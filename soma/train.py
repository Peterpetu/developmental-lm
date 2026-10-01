"""`LIFETIME`: one complete training run of a phenotype from a germline.

SPEC: 07 §7.2
SPEC: 01 §1.3.1
SPEC: 02 §2.6.4

§01.3.1's four rate regimes are the loop's structure and not a commentary on it:
`r_0` is the within-token recurrence inside the mixer, `r_1` is the optimizer
step this function iterates, `r_2` is `SELECT` at period `T_sel(n)`, and `r_3`
is the germline update that happens *outside* this function entirely — which is
what the Weismann barrier means operationally. §02.6.4's `FITNESS` is computed
at the end from the curve this loop records.

```
Algorithm LIFETIME(Gamma, seed) -> (metrics, archetypes)

  1.  U       <- SEED_POPULATION(Gamma.Inventory, |U|_0, side)      # §02.2.4
  2.  Theta_S <- INIT(Gamma)                                        # §01.5
  3.  ACCEPTANCE_TEST()                                             # §02.5 -- AFTER init
  4.  record birth performance on the held-out set, no updates
  5.  Lambda  <- SEED_ARCHIVE()                                     # §06.3.3
      init next-event counters: n_sel, n_arch, n_cert, n_log, n_val # (1.15)
  6.  for n in 0 .. N_step-1:
        a. batch <- STREAM(n)                                       # §06.3.4
        b. forward; L_task (§06.2), L_met (§01.6), L_z (§05.6)
        c. backward; ASSERT_BARRIER; Muon/AdamW step on Theta_S only
        d. renormalise p_u to the unit sphere
        e. update nu_u (§03.4), masks m_u, ages; retire units whose mask reached 0
        f. push credits into the reservoir buffer (§04.6)
        g. if n >= n_ref:  update theta_lag (§06.1.2)
        h. if n >= n_sel and n >= max(T_0, M_mat):  SELECT (§04.2)
        i. if n in growth_schedule:  GROW (§03.5.1)
        j. if n >= n_arch and f_oe > 0:  ARCHIVE_STEP (§06.3.3)
        k. if n >= n_cert and f_oe > 0:  CERTIFY, live / frozen / ramp (§06.4)
        l. if n >= n_val:  held-out evaluation; update n_crit if first crossing
        m. if n >= n_log:  log diagnostics (§7.6)
  7.  return metrics, and the gene sets of units surviving with nu_hat above the median
  postcondition: Theta_S is discarded; only metrics and archetypes cross the barrier
```

**[NORMATIVE]** Step 7 is the Weismann barrier in operational form: no weights,
no optimizer state, no activations are inherited — only selection statistics of
germline objects.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from soma.config import SomaConfig
from soma.data.stream import Batch, SyntheticPacker, TrainingStream, f_oe_at
from soma.development.bifurcation import epsilon_bif_from
from soma.development import (
    BifurcationMonitor,
    UtilityState,
    grow,
    measure_sd_probe,
    nu_hat,
    update_utility,
)
from soma.eval.metrics import (
    CurveTracker,
    Fitness,
    compute_fitness,
    erank_probe,
    held_out_ce,
)
from soma.germline import Genome, GermlineRegistry, assert_barrier
from soma.model.init import acceptance_test, apply_calibration, calibrate
from soma.model.metabolic import d_unif, nocollapse_ok
from soma.model.model import SomaModel
from soma.openended.archive import TaskArchive
from soma.openended.certify import (
    BatchRing,
    CertificationVerdict,
    CheckpointRing,
    capture_cadence,
    certify,
    RampController,
)
from soma.openended.references import LaggedSelf, ReferenceCache
from soma.openended.tasks import TaskParams, generate
from soma.openended.tokensel import ExploreState, masked_cross_entropy, select_tokens
from soma.optim.muon import build_optimizers, wsd_lr, zero_slot_state
from soma.selection import CaseReservoir, CreditAudit, select
from soma.utils.logging import EventCounter, JsonlLogger, RunPaths
from soma.utils.rng import RngBundle

__all__ = ["LifetimeConfig", "LifetimeResult", "lifetime"]


def _autocast(device: torch.device, dtype: Optional[torch.dtype]):
    """§05.9's `bfloat16` activations, as a context manager.

    A no-op on CPU or when `dtype is None`; `bfloat16` needs no loss scaling,
    which is why §05.9 chooses it over `float16` and why no `GradScaler`
    appears anywhere in this file.
    """
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


@dataclass
class LifetimeConfig:
    """Everything a lifetime needs that is not the germline."""

    cfg: SomaConfig
    run_id: str = "run"
    runs_root: str = "runs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    corpus: object = None
    eval_batches: Sequence[Batch] = ()
    probe_tokens: Optional[torch.Tensor] = None
    L_star: Optional[float] = None
    slope_ref: float = 1.0
    ref_cache: Optional[ReferenceCache] = None
    rows_per_batch: int = 32
    max_steps: Optional[int] = None
    """Overrides `N_step`, for smoke runs. Recorded in the manifest."""
    use_mean_utility: bool = False
    """Ablation A2."""
    pre_gate: bool = False
    """Ablation A12."""
    dense_sites: bool = False
    """M0 / B1 baseline (§08.4)."""
    beta_unsigned: bool = False
    """**A7** (§08.6): `beta_mix = sigmoid`, range `[0,1]`. Removes §05.2.2's
    signed transitions, which is what K7 is adjudicated against. M1 only."""
    psi_off: bool = False
    """**A10** (§05.5, §08.6): `psi == 1`, the constant context gate."""
    freeze_body_plan: bool = False
    """**A9** (§08.6): freeze `p_u` and `g_u`; train `A`, `B`, `W_shared`.

    §08.6's own note (audit's D-56 resolution): A9 freezes the **body plan**
    while leaving the parts trainable, so a difference is attributable to the
    plan rather than to having frozen half the model."""
    random_router: bool = False
    """**B3** (§08.4, §05.4): the random router, `r_ut = 1/k_route`, and
    `L_wire` disabled. §05.4 makes it **required in every routing experiment**;
    §10.1 K10 is written on it."""
    frozen_field: bool = False
    """**B2** (§08.4), the Regret Ecology: the germline field is frozen at its
    **random initialisation** — not removed, so the Site still exists and (5.20)
    is well defined. Births, deaths, selection and `L_met` are disabled;
    token selection and the task stream stay on. Spec v1 said "no germline",
    under which the Site has no forward pass — and B2 is the baseline whose
    defeat C1 requires."""
    enable_selection: bool = True
    """Ablation A3 sets this False (births/deaths off)."""
    log_to_disk: bool = True
    save_final_state: bool = False
    """Write `checkpoints/final.pt` at the end of the lifetime.

    §07.4.2 lists `theta_m0` — the frozen reference whose per-token losses §06.2
    caches — as one of three artifacts M0 produces. v3.0's `run_m0.py` recorded
    the *path* to that checkpoint in its report and nothing ever wrote the file,
    so §06.2's Rho-1 reference had no source (audit A-70). Off by default: a
    generation curve runs hundreds of lifetimes and does not want hundreds of
    checkpoints."""
    verbose: bool = True
    eval_every: Optional[int] = None
    """Overrides `T_val`, for smoke runs."""
    grad_checkpoint: bool = True
    """§01.9's `B_tok` does not fit in 16 GB without it; see
    `SomaModel.forward`. Disabled only for the profiling comparison."""
    micro_batches: int = 1
    """Gradient accumulation. `B_tok` (§01.9) is the **optimizer-step** batch;
    `rows_per_batch` is the micro-batch. §06.2 already presumes this split —
    "Ranking is over the **micro-batch**", because ranking over a full
    accumulated batch would need every micro-batch's `Delta` before any backward
    pass, which is not implementable in one pass."""
    amp_dtype: Optional[torch.dtype] = torch.bfloat16
    """§05.9: "Parameters and activations `bfloat16`; optimizer state and
    Newton-Schulz `float32`."

    Implemented as autocast rather than by casting the parameters, so the master
    weights and both optimizers stay in `float32` while the activations - which
    dominate memory at M1 - are `bfloat16`. Every reduction the specification
    calls out as delicate promotes itself back: the UT triangular solves and the
    log-space decay accumulation (§05.2.3, §05.9), every normalisation (§05.9's
    floor rule), and the `logsumexp` and excess losses of §06.2. `None` disables
    it, which is what the `float64` oracle comparisons need."""


@dataclass
class LifetimeResult:
    """What crosses the Weismann barrier (§07.2 step 7)."""

    fitness: Fitness
    curve: CurveTracker
    archetypes: List[Tuple[int, ...]]
    mean_utility: float
    mean_clade_size: float
    metrics: Dict[str, object]
    wall_seconds: float
    steps_run: int
    certifications: List[Dict[str, object]] = field(default_factory=list)
    selection_events: List[Dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "fitness": self.fitness.as_dict(),
            "curve": self.curve.as_dict(),
            "n_archetypes": len(self.archetypes),
            "mean_utility": self.mean_utility,
            "mean_clade_size": self.mean_clade_size,
            "wall_seconds": self.wall_seconds,
            "steps_run": self.steps_run,
            **self.metrics,
        }


# ---------------------------------------------------------------------------


def lifetime(
    genome: Genome,
    lc: LifetimeConfig,
    seed: int,
    *,
    archive: Optional[TaskArchive] = None,
    solve_rate_fn: Optional[Callable[[object, Genome, TaskParams, int], float]] = None,
) -> LifetimeResult:
    """One lifetime. See the algorithm in the module docstring.

    `solve_rate_fn(model, genome, params, n_eval) -> s_task` supplies §06.3.3's
    greedy decode protocol. It takes the model **and the genome** because both
    are constructed or moved to the device *here*, and `s_task` is by definition
    a measurement of the **current** model: a closure captured before the
    lifetime starts would hold a CPU genome and a model that does not yet exist.
    """
    cfg = lc.cfg
    device = torch.device(lc.device)
    t0 = time.time()
    N = lc.max_steps if lc.max_steps is not None else cfg.N_step
    soma_on = not lc.dense_sites
    # B2 keeps the Site and the stream and removes the population dynamics
    # (§08.4). It is not `dense_sites`: the Site must still exist.
    # A9 freezes the **body plan**: `p_u` and `g_u` stop changing, while `A`,
    # `B` and `W_shared` keep training. `g_u` changes only through birth and
    # mutation, so freezing it means no structural events; `p_u` is a parameter,
    # so it is removed from the optimizer in `_apply_ablations`.
    structural_on = (
        lc.enable_selection and not lc.frozen_field and not lc.freeze_body_plan
    )
    # B2 additionally disables `L_met` (§08.4): the metabolic price is a
    # statement about a population that B2 does not have.
    metabolic_on = soma_on and not lc.frozen_field

    # Torch generators must live on the device they draw for; NumPy streams are
    # device-independent. `RngBundle` keeps both per purpose (§02.6.3).
    rngs = RngBundle.create(seed, device=device)
    paths = RunPaths.create(lc.runs_root, lc.run_id) if lc.log_to_disk else None
    logger = JsonlLogger(paths.log) if paths else None

    def log(kind: str, step: int, **f: object) -> None:
        if logger is not None:
            logger.log(kind, step, **f)

    # ---------------------------------------------------------------- 1, 2
    genome = genome.to(device)
    registry = GermlineRegistry()
    genome.register(registry)

    # `nn.Linear`/`nn.Embedding`/`nn.init.normal_` take no generator, so they
    # draw from the process-global stream. Seeding it here is what makes the
    # initial weights a function of `seed` alone (§02.6.3, audit A-64).
    rngs.seed_global("init")
    model = SomaModel(cfg, dense_sites=lc.dense_sites).to(device)
    _apply_ablations(model, lc)
    # §01.7.2: in-lifetime policy writes act on a DEEP COPY taken at step 0.
    # Only pre-lifetime values enter ES_STEP; without this the writes would be
    # Lamarckian inheritance that ASSERT_BARRIER cannot detect.
    policy = copy.deepcopy(genome.constants())

    neighbour_table = genome.field_.neighbour_table() if soma_on else None
    if soma_on:
        model.seed_populations(genome, rngs.np("init"), rngs.pt("init"))

    # ---------------------------------------------------------------- 3
    acceptance = None
    if soma_on:
        cal = calibrate(
            cfg, genome.field_, model.parts.A.data,
            generator=rngs.pt("init"), kappa=model.parts.kappa.data,
            W_ctx=model.sites[0].W_ctx.weight.data,
        )
        apply_calibration(model, cal, generator=rngs.pt("init"))
        acceptance = acceptance_test(model, genome, registry, generator=rngs.pt("init"))
        log("acceptance", 0, **acceptance.as_dict())
        if lc.verbose:
            print(
                f"  acceptance: var_ratio={acceptance.variance_ratio:.3f} "
                f"grad_ratio={acceptance.grad_ratio:.3f} "
                f"gain_act={acceptance.gain_act:.4f} N_eff={acceptance.n_eff_large:.3f}"
            )

    muon, adamw = build_optimizers(model, cfg)
    opt_param_ids = [id(p) for g in muon.param_groups for p in g["params"]] + [
        id(p) for g in adamw.param_groups for p in g["params"]
    ]

    # ---------------------------------------------------------------- 4
    curve = CurveTracker(cfg, L_star=lc.L_star)
    L_birth = float("nan")
    if lc.eval_batches:
        L_birth, nt = held_out_ce(
            model, lc.eval_batches, genome=genome if soma_on else None, device=device
        )
        curve.record(0, 0, L_birth, nt)
        log("birth", 0, L_birth=L_birth)

    # ---------------------------------------------------------------- 5
    frozen_archive: Optional[TaskArchive] = None
    bound_solve_rate = (
        (lambda params, n_eval: solve_rate_fn(model, genome, params, n_eval))
        if solve_rate_fn is not None else None
    )
    if soma_on and policy.f_oe > 0 and bound_solve_rate is not None:
        if archive is None:
            archive = TaskArchive(T_ctx=cfg.T_ctx)
            archive.seed(bound_solve_rate, rngs.np("archive"))
        # The frozen control arm: Lambda fixed after SEED_ARCHIVE (§06.4).
        frozen_archive = TaskArchive(T_ctx=cfg.T_ctx)
        frozen_archive.seed(bound_solve_rate, rngs.np("archive"))

    stream = TrainingStream(
        cfg, lc.corpus, archive, rows_per_batch=lc.rows_per_batch, rng=rngs.np("data")
    )

    t_val = lc.eval_every if lc.eval_every is not None else cfg.T_val
    c_sel = EventCounter(cfg.T_sel, warmup=max(cfg.T_0, policy.M_mat))
    c_arch = EventCounter(cfg.T_arch)
    c_cert = EventCounter(cfg.T_cert)
    c_val = EventCounter(t_val)
    c_log = EventCounter(cfg.T_log)
    c_ref = EventCounter(cfg.T_ref)
    c_erank = EventCounter(10 * cfg.T_log)
    c_explore = EventCounter(10 * cfg.T_log)
    c_capture = EventCounter(capture_cadence(N, 32))
    c_ckpt = EventCounter(capture_cadence(N, 8))

    n_sites = cfg.L if soma_on else 0
    reservoirs = [CaseReservoir(cfg.q_cases, cfg.k_route, device=device) for _ in range(n_sites)]
    util = [UtilityState.zeros(cfg.U_max, device) for _ in range(n_sites)]
    explore = ExploreState(V=cfg.V)
    audit = CreditAudit()

    lagged = LaggedSelf(model, cfg.kappa_lag, cfg.T_ref) if soma_on else None
    lag_model = SomaModel(cfg, dense_sites=lc.dense_sites).to(device) if soma_on else None

    sd_probe = 0.0
    if soma_on and lc.probe_tokens is not None:
        probe = lc.probe_tokens.to(device)

        def probe_loss() -> float:
            with torch.no_grad():
                return _probe_loss(
                    model(probe, genome=genome, want_metabolic=False), probe
                )

        sd_probe = measure_sd_probe(probe_loss)
    # §01.7.2 (audit A-58): `3 * sd_probe` alone is zero on deterministic
    # hardware, which makes the alarm fire on any non-zero difference at all.
    probe_scale = 1.0
    if soma_on and lc.probe_tokens is not None:
        with torch.no_grad():
            probe_scale = _probe_loss(
                model(lc.probe_tokens.to(device), genome=genome,
                      want_metabolic=False),
                lc.probe_tokens.to(device),
            )
    epsilon_bif = epsilon_bif_from(sd_probe, probe_scale)
    bif = BifurcationMonitor(epsilon_bif, cfg.phi_ann)

    # §04.5.1's cost governor, and §07.6's required selection wall-clock share.
    # Both were absent in v3.0 (audit A-75): `d_rate` was read in eight places
    # and written in none, and nothing anywhere timed `select()`.
    sel_governor = SelectionGovernor(policy)

    ckpt_ring = CheckpointRing(N)
    rings: Dict[str, BatchRing] = {"live": BatchRing(), "frozen": BatchRing(), "ramp": BatchRing()}
    # §06.4's ramp arm is **matched** to the live arm's mean solve-rate
    # trajectory (audit A-79); the controller is what carries the state
    # between captures, so it lives beside the rings.
    ramp = RampController()
    packer = SyntheticPacker(cfg.T_ctx)
    certifications: List[Dict[str, object]] = []
    selection_events: List[Dict[str, object]] = []
    tokens_seen = 0
    d_unif_ref = d_unif(cfg.m_dim)

    if lc.verbose:
        print(f"  lifetime: {N} steps  cfg={cfg.name}  device={device}  seed={seed}")

    # ---------------------------------------------------------------- 6
    model.train()
    for n in range(N):
        t_step = time.time()
        lr = wsd_lr(n, cfg)
        for g in muon.param_groups:
            g["lr"] = lr
        for g in adamw.param_groups:
            g["lr"] = lr

        # --- a-c. micro-batched forward/backward ----------------------------
        # `B_tok` (§01.9) is the **optimizer-step** batch; `rows_per_batch` is
        # the micro-batch, and `micro_batches` accumulates to reach it. §06.2
        # already presumes the split: "Ranking is over the **micro-batch**",
        # because ranking over a full accumulated batch would need every
        # micro-batch's `Delta` before any backward pass.
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)

        accum = max(1, lc.micro_batches)
        L_task = L_z = None
        sel = None
        out = None
        batch = None
        loss = None
        for micro in range(accum):
            batch = stream.batch(n, policy.f_oe).to(device)
            tokens_seen += int(batch.valid.sum())

            with _autocast(device, lc.amp_dtype):
                out = model(
                    batch.tokens,
                    genome=genome if soma_on else None,
                    doc_id=batch.doc_id,
                    valid=batch.valid,
                    pre_gate=lc.pre_gate,
                    want_metabolic=metabolic_on,
                    grad_checkpoint=lc.grad_checkpoint,
                )

            # Per-token loss first, so token selection can rank it; the second
            # call applies the realised mask. The ranking pass runs under
            # `no_grad`: only the resulting *mask* is used, and letting it build
            # a graph over `(B, T, V)` logits doubles the largest activation in
            # the model for a quantity that is immediately discarded.
            with torch.no_grad():
                _, _, loss_per_token = masked_cross_entropy(
                    out.logits, batch.targets, batch.scored
                )

            ref = None
            if lc.ref_cache is not None and batch.flat_index is not None:
                idx = batch.flat_index.clamp_min(0).cpu().numpy()
                vals = lc.ref_cache.lookup(idx.reshape(-1)).reshape(idx.shape).to(device)
                ref = torch.where(batch.flat_index >= 0, vals, torch.zeros_like(vals))

            sel = select_tokens(
                loss_per_token, ref, batch.is_synthetic, batch.scored,
                policy.k_tok, explore, generator=rngs.pt("explore"),
            )
            explore.observe(batch.tokens, sel.include)

            L_task, L_z, _ = masked_cross_entropy(
                out.logits, batch.targets, sel.include
            )
            loss = L_task + L_z
            if out.metabolic is not None:
                loss = loss + out.metabolic.total

            if not torch.isfinite(loss):
                log("abort", n, reason="non-finite loss", micro=micro)
                raise FloatingPointError(
                    f"non-finite loss at step {n} (micro-batch {micro}); "
                    f"§07.4.3 criterion 1 requires no NaN/Inf in loss or "
                    f"gradients for the whole lifetime"
                )

            # Each micro-batch contributes its share of the step's gradient.
            (loss / accum).backward()

            if soma_on and micro < accum - 1:
                # Population state advances once per optimizer step, not once
                # per micro-batch, so the utility/reservoir updates below run
                # only on the last one. Credits from earlier micro-batches are
                # still offered to the reservoir, which is what keeps the case
                # sample uniform over the interval.
                _offer_credits(
                    model, out, batch, cfg, reservoirs, policy, rngs,
                    lag_model, lagged, genome, device,
                )
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        # §07.4.3 criterion 1 requires no NaN/Inf in loss **or gradients** for
        # the whole lifetime. The clipped total norm is NaN iff some gradient is,
        # so this costs one scalar read that is already computed -- and it
        # catches the failure one step before it reaches the parameters, where
        # the diagnosis is still possible.
        if not math.isfinite(grad_norm):
            offenders = [
                n for n, p in model.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()
            ]
            log("abort", n, reason="non-finite gradient", offenders=offenders[:8])
            raise FloatingPointError(
                f"non-finite gradient norm at step {n} (loss={float(loss):.4f}); "
                f"offending parameters: {offenders[:8]}"
            )
        assert_barrier(registry, optimizer_param_ids=opt_param_ids)
        muon.step()
        adamw.step()

        # --- d. renormalise p_u --------------------------------------------
        if soma_on:
            with torch.no_grad():
                for pop in model.populations:
                    pop.p_u.data.div_(
                        pop.p_u.data.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    )

        # --- e-f. utility, masks, ages, reservoir --------------------------
        if soma_on:
            # §06.1.2: theta_lag is evaluated on a d_rate fraction of the ROWS,
            # and exactly those rows are offered to the reservoir. The row
            # sample is uniform, so the reservoir still sees an unbiased sample
            # of the interval's tokens (§04.6).
            n_rows = batch.tokens.shape[0]
            n_lag = max(1, int(round(policy.d_rate * n_rows)))
            rows = torch.from_numpy(
                rngs.np("downsample").choice(n_rows, size=n_lag, replace=False)
            ).to(device)
            w_rows = _lag_rows(lag_model, lagged, genome, batch, loss_per_token, rows)
            keep = batch.scored[rows].reshape(-1)

            for li, pop in enumerate(model.populations):
                so = out.site_outputs[li]
                slots_all = so.slots.reshape(-1, cfg.k_route)
                r_all = so.r_ut.reshape(-1, cfg.k_route).detach()
                dn_all = so.delta_norm.reshape(-1, cfg.k_route).detach()

                util[li].accumulate(slots_all, r_all, dn_all)
                update_utility(pop, util[li], genome.field_, model.parts, policy.eta_ema)
                util[li].reset()

                retired = pop.step_masks(n)
                if bool(retired.any()):
                    dead = torch.nonzero(retired, as_tuple=False).squeeze(-1)
                    pop.retire(dead)
                    zero_slot_state(adamw, pop.p_u, dead)

                if bool(keep.any()):
                    reservoirs[li].push(
                        batch.tokens[rows].reshape(-1)[keep],
                        so.slots[rows].reshape(-1, cfg.k_route)[keep],
                        so.delta_norm[rows].reshape(-1, cfg.k_route).detach()[keep],
                        so.r_ut[rows].reshape(-1, cfg.k_route).detach()[keep],
                        w_rows.reshape(-1)[keep],
                        rngs.np("downsample"),
                    )

        # --- g. theta_lag ---------------------------------------------------
        if lagged is not None and c_ref.maybe(n):
            lagged.update(model)

        # --- h. SELECT ------------------------------------------------------
        if soma_on and structural_on and c_sel.due(n):
            c_sel.fire(n)
            t_sel = time.time()
            for li, pop in enumerate(model.populations):
                ev = select(
                    pop, reservoirs[li], n,
                    cfg=cfg, field_=genome.field_, neighbour_table=neighbour_table,
                    rng_lexicase=rngs.np("lexicase"),
                    rng_mutation=rngs.np("mutation"),
                    rng_downsample=rngs.np("downsample"),
                    generator=rngs.pt("mutation"),
                    rho_scale=bif.rho_scale, site=li, G_gen=genome.G_gen,
                    use_mean_utility=lc.use_mean_utility,
                )
                if ev is not None:
                    d = ev.as_dict()
                    selection_events.append(d)
                    bif.register_cohort(li, ev.births, n)
                    curve.annotate(n, "select", site=li,
                                   births=len(ev.births), deaths=len(ev.deaths))
                    log("selection", n, **{k: v for k, v in d.items() if k != "step"})
                    if li == 0 and lc.probe_tokens is not None:
                        rho = _credit_audit(
                            model, genome, pop, ev, audit, cfg,
                            lc.probe_tokens.to(device), device,
                            rngs.np("downsample"),
                            lag_model=lag_model, lagged=lagged,
                        )
                        if rho is not None:
                            log("credit_audit", n, site=li, **audit.as_dict())
            sel_governor.add_selection(time.time() - t_sel)

        # --- i. GROW --------------------------------------------------------
        if soma_on and structural_on and n in cfg.growth_schedule and n > 0:
            for li, pop in enumerate(model.populations):
                created = grow(
                    pop, n, cfg=cfg, field=genome.field_,
                    neighbour_table=neighbour_table, rng=rngs.np("growth"),
                    generator=rngs.pt("growth"), G_gen=genome.G_gen,
                )
                bif.register_cohort(li, created, n, origin="grow")
                curve.annotate(n, "grow", site=li, created=len(created))
                log("growth", n, site=li, created=len(created), n_live=pop.n_live)

        # --- bifurcation, measured AFTER the anneal (1.21) ------------------
        if soma_on and lc.probe_tokens is not None:
            for cohort in bif.due(n):
                fired = _evaluate_bifurcation(
                    model, genome, bif, cohort, lc.probe_tokens.to(device), n,
                    model.populations[cohort.site].n_live,
                )
                if fired:
                    log("bifurcation", n, **{
                        k: v for k, v in bif.history[-1].items() if k != "step"
                    })

        # --- j. ARCHIVE_STEP ------------------------------------------------
        f_now = f_oe_at(n, cfg, policy.f_oe)
        if archive is not None and bound_solve_rate is not None and f_now > 0 and c_arch.maybe(n):
            rec = archive.step(bound_solve_rate, rngs.np("archive"), n)
            log("archive", n, **{
                k: v for k, v in rec.items()
                if k not in ("parent", "child", "step")
            })

        # --- capture rings, cadence decoupled from T_cert (A-23) ------------
        if archive is not None and f_now > 0 and c_capture.maybe(n):
            _capture_arms(rings, archive, frozen_archive, packer, n, N, cfg,
                          rngs, ramp, bound_solve_rate)
        if archive is not None and f_now > 0 and c_ckpt.maybe(n):
            ckpt_ring.offer(
                n, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )

        # --- k. CERTIFY -----------------------------------------------------
        if archive is not None and f_now > 0 and c_cert.maybe(n):
            v = _certify_three_arms(model, genome, ckpt_ring, rings, n, device)
            if v is not None:
                d = v.as_dict()
                certifications.append(d)
                log("certification", n, **{k: x for k, x in d.items() if k != "step"})

        # --- l. held-out evaluation -----------------------------------------
        if lc.eval_batches and c_val.maybe(n):
            ce, nt = held_out_ce(
                model, lc.eval_batches, genome=genome if soma_on else None, device=device
            )
            curve.record(n, tokens_seen, ce, nt)
            log("eval", n, ce=ce, tokens=tokens_seen, n_scored=nt)

        # --- m. diagnostics --------------------------------------------------
        if c_log.maybe(n):
            # Attribution, at log cadence: the per-step guards cover the loss
            # and the gradients, but a parameter can also be corrupted by a
            # failing device-side kernel with every gradient finite -- which is
            # exactly how audit A-55 presented. One pass over the parameters
            # every `T_log` steps names the tensor while the run can still say
            # when it happened.
            nonfinite = [
                pname for pname, pval in model.named_parameters()
                if not torch.isfinite(pval).all()
            ]
            if nonfinite:
                log("abort", n, reason="non-finite parameter", offenders=nonfinite[:8])
                raise FloatingPointError(
                    f"non-finite parameters at step {n}: {nonfinite[:8]} "
                    f"(loss and gradients were finite; suspect a device-side "
                    f"kernel -- see AUDIT A-55)"
                )
            rec: Dict[str, object] = {
                "lr": lr, "loss": float(loss), "L_task": float(L_task),
                "L_z": float(L_z), "grad_norm": grad_norm, "tokens": tokens_seen,
                "n_selected": sel.n_selected,
                "synthetic_share": sel.synthetic_share,
                "n_explore": sel.n_explore, "p_explore": explore.p_explore,
                "f_oe": f_now, "sep": _separation(n, cfg, policy),
                "barrier_ok": True,
                **sel_governor.as_dict(),
            }
            if out.metabolic is not None:
                rec.update(out.metabolic.as_dict())
            if soma_on:
                pop0 = model.populations[0]
                rec.update(pop0.state_summary())
                ok, realised, thresh = nocollapse_ok(
                    pop0.p_u.data, pop0.live, cfg.m_dim, d_unif_ref
                )
                w_in, w_bt = pop0.clade_distances()
                rec.update({
                    "mean_pairwise_p": realised, "nocollapse_threshold": thresh,
                    "nocollapse_ok": ok, "clade_within": w_in, "clade_between": w_bt,
                    "bif_alarm_rate": bif.alarm_rate, "rho_scale": bif.rho_scale,
                    "reservoir_len": len(reservoirs[0]),
                })
            # **[NORMATIVE]** §01.7.1: `sep(n) <= 1` is **asserted** and logged
            # for `n_warm <= n < n_dec` only. v3.0 logged it and asserted
            # nothing, so a configuration violating the timescale separation ran
            # silently to completion and §01.8's falsifier rested on an
            # unenforced claim (audit A-75).
            if cfg.n_warm <= n < cfg.n_dec and float(rec["sep"]) > 1.0 + SEP_TOL:
                log("abort", n, reason="timescale separation violated",
                    sep=rec["sep"])
                raise AssertionError(
                    f"sep({n}) = {float(rec['sep']):.4f} > 1 in [n_warm, n_dec) "
                    f"= [{cfg.n_warm}, {cfg.n_dec}); (1.20) requires the "
                    f"structural rate to fall at least as fast as the learning "
                    f"rate (§01.7.1)"
                )
            log("step", n, **rec)
            if lc.verbose and c_log.fired % 20 == 1:
                print(
                    f"    n={n:6d} loss={float(loss):.4f} lr={lr:.2e} "
                    f"|U|={rec.get('n_live', '-')} tok={tokens_seen:,}"
                )

        reduced = sel_governor.end_step(time.time() - t_step)
        if reduced is not None:
            log("d_rate_reduced", n, **reduced)

        if c_explore.maybe(n):
            explore.adjust(policy.k_tok)

        if soma_on and lc.probe_tokens is not None and c_erank.maybe(n):
            ranks = erank_probe(model, lc.probe_tokens.to(device), genome)
            log("erank", n, erank=ranks, mean=float(np.mean(ranks)))

    # ---------------------------------------------------------------- 7
    archetypes, mean_util, mean_clade = _harvest(model, policy, soma_on)

    fitness = compute_fitness(
        curve, cfg,
        L_birth=L_birth if math.isfinite(L_birth) else cfg.L_rand,
        slope_ref=lc.slope_ref,
    )

    metrics: Dict[str, object] = {
        "selection_events": len(selection_events),
        # Counted rather than inferred: the §07.4.3 machinery gate asserts that
        # every mechanism fired at least once, and "the population changed" is
        # not the same claim as "births happened and deaths happened" -- a run
        # in which only growth fired would satisfy the first and not the second.
        "total_births": sum(int(e["n_births"]) for e in selection_events),
        "total_deaths": sum(int(e["n_deaths"]) for e in selection_events),
        "growth_events": bif.n_cohorts.get("grow", 0),
        "certifications": len(certifications),
        "archive_size": len(archive) if archive is not None else 0,
        "bif_alarm_rate": bif.alarm_rate,
        "bif_growth_alarm_rate": bif.growth_alarm_rate,
        "bif_alarms_per_birth": bif.births_alarm_rate,
        "bif_cohorts": dict(bif.n_cohorts),
        "bif_births": bif.n_births,
        "epsilon_bif": epsilon_bif,
        "sd_probe": sd_probe,
        "credit_audit": audit.as_dict(),
        # §06.4's ramp arm reports how well matched it actually was; a large
        # `ramp_matching_error` means `live - ramp` compared two difficulty
        # regimes and K5's verdict is not attributable (audit A-79).
        **ramp.as_dict(),
        "tokens_seen": tokens_seen,
        "final_n_live": model.populations[0].n_live if soma_on else 0,
        "acceptance": acceptance.as_dict() if acceptance is not None else None,
        "reservoir_bytes": reservoirs[0].memory_bytes() * len(reservoirs) if reservoirs else 0,
    }
    if lc.save_final_state and paths is not None:
        final = paths.checkpoints / "final.pt"
        # The key is `"model"`, which is what `scripts/eval_tinystories.py`
        # reads. A writer and a reader that disagree on a key produce a
        # `KeyError` at the *end* of an M1 run, hours after the run that
        # produced the file (audit A-71).
        #
        # `model.state_dict()` carries the unit populations as buffers -- 66
        # entries at M-nano: `p_u`, `g_u`, `live`, `dying`, `m_u`, `age_u`,
        # `nu_u`, `h_bar_u` and the anneal state per Site -- so the population
        # is restored with the weights and needs no separate record. The
        # **germline** is not in the state dict by construction (§02.1: it is
        # not a parameter of the phenotype), so it is saved alongside.
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in
                          model.state_dict().items()},
                "genome": genome.to("cpu"),
                "config": cfg.summary(),
                "seed": seed,
                "steps": N,
                "dense_sites": lc.dense_sites,
            },
            final,
        )
        log("checkpoint", N, path=str(final))

    if logger is not None:
        logger.log("lifetime_end", N, **metrics, **fitness.as_dict())
        logger.close()

    return LifetimeResult(
        fitness, curve, archetypes, mean_util, mean_clade, metrics,
        time.time() - t0, N, certifications, selection_events,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SEP_TOL = 1e-6
"""Slack on §01.7.1's `sep(n) <= 1` assertion.

(1.20) is a ratio of two exactly-computed schedules, so the only spread is
floating point; the tolerance exists so a value of `1 + 2e-16` cannot abort a
run, and is small enough that a real violation cannot hide under it."""


class SelectionGovernor:
    """§04.5.1's cost governor, and §07.6's selection wall-clock share.

    SPEC: 04 §4.5.1
    SPEC: 07 §7.6

    ```
    if mean( t_select / t_step ) over a 1000-step window > 0.02:
        d_rate <- max( 0.02 , 0.7 * d_rate )
    ```

    **[NORMATIVE - v3.1, audit A-75]** v3.0 implemented neither the governor nor
    its measurement. `d_rate` was read in eight places and written in none; no
    call site anywhere timed `select()`; `SelectionEvent.as_dict()` carried no
    wall-clock field; and §07.8's profiling row for selection ("capped at 2% of
    step time") was therefore uncheckable.

    §01.7.2 names this as one of exactly **two** in-lifetime policy writes, the
    other being `rho`. Like that one it acts on the **deep copy** of `Policy`
    taken at `LIFETIME` step 0, so the reduction is not inherited and
    `ASSERT_BARRIER` -- which cannot see a direct write -- does not need to.

    The window closes on **step count**, not on elapsed time, so the decision is
    reproducible: two runs with the same seed reduce `d_rate` at the same steps
    even on differently loaded hardware.
    """

    WINDOW = 1000
    SHARE_CAP = 0.02
    FACTOR = 0.7
    FLOOR = 0.02

    def __init__(self, policy) -> None:
        self.policy = policy
        self.t_sel = 0.0
        self.t_step = 0.0
        self.n_in_window = 0
        self.last_share = 0.0
        self.n_reductions = 0

    def add_selection(self, seconds: float) -> None:
        self.t_sel += float(seconds)

    def end_step(self, seconds: float) -> Optional[Dict[str, object]]:
        """Accumulate one step; at the window boundary apply §04.5.1.

        Returns a log record when the window closed over budget, else `None`.
        """
        self.t_step += float(seconds)
        self.n_in_window += 1
        if self.n_in_window < self.WINDOW:
            return None
        share = self.t_sel / max(self.t_step, 1e-12)
        self.last_share = share
        self.t_sel = self.t_step = 0.0
        self.n_in_window = 0
        if share <= self.SHARE_CAP:
            return None
        before = float(self.policy.d_rate)
        after = max(self.FLOOR, self.FACTOR * before)
        if after >= before:
            # Already at the floor: the governor has done all it can. Reported
            # anyway, so a run that stays over budget says so rather than
            # looking like a run that never exceeded it.
            return {"share": share, "d_rate": before, "at_floor": True}
        object.__setattr__(self.policy, "d_rate", after)
        self.n_reductions += 1
        return {"share": share, "d_rate_before": before,
                "d_rate_after": after, "at_floor": False}

    def as_dict(self) -> Dict[str, object]:
        return {
            "selection_share": self.last_share,
            "d_rate": float(self.policy.d_rate),
            "d_rate_reductions": self.n_reductions,
        }


N_EVAL_RAMP = 8
"""Instances per ramp solve-rate measurement (§06.4).

Smaller than the archive's `n_eval = 32` deliberately: the ramp rate is a
**control signal** for a one-step ratchet, not a reported statistic, and it is
measured at every capture — `K_ring` times per run per arm. At `1/8` resolution
the ratchet's `0.05` tolerance is still well above the measurement quantum."""


def _apply_ablations(model, lc) -> None:
    """Set the module-level ablation switches (§08.6, §08.4).

    SPEC: 08 §8.4
    SPEC: 08 §8.6

    **[NORMATIVE — v3.1, audit A-80]** §08.6 states that *"every novel component
    carries exactly one ablation and one kill criterion"*, and v3.0 could reach
    four of them. A7, A9, A10, B2 and B3 were fully specified and had **no
    switch anywhere**, so §05.11's falsifiers for the router and the context
    gate, §01.8's falsifiers for the Moment regime, and K7 and K10 could not be
    evaluated at all.

    Called immediately after construction and **before** the optimizer is built,
    because B3 freezes parameters and a frozen tensor left in an optimizer group
    still collects weight decay.
    """
    if lc.beta_unsigned:                                    # A7
        for m in model.modules():
            if hasattr(m, "beta_unsigned"):
                m.beta_unsigned = True
    if lc.psi_off:                                          # A10
        for m in model.modules():
            if hasattr(m, "psi_off"):
                m.psi_off = True
    if lc.random_router:                                    # B3
        for site in getattr(model, "sites", []):
            site.router.freeze_for_b3()
    if lc.freeze_body_plan:                                 # A9
        # `p_u` is the only *parameter* of the body plan; `g_u` is a buffer that
        # changes only through birth and mutation, and `structural_on` stops
        # those. Freezing here rather than skipping the optimizer group keeps
        # `optimizer_groups()` exhaustive-and-disjoint, which §05.7 requires.
        for pop in getattr(model, "populations", []):
            pop.p_u.requires_grad_(False)


MAX_AUDIT_UNITS = 128
"""Distinct units the credit audit may switch off in one event (§04.4).

Twice `CreditAudit.min_pairs`, so one event still over-supplies the 64-pair
minimum; see `_credit_audit` for the cost arithmetic this bounds."""


@torch.no_grad()
def _credit_audit(
    model, genome, pop, event, audit, cfg, probe, device, rng,
    lag_model=None, lagged=None,
) -> Optional[float]:
    """§04.4's leave-one-out validity check, over **(unit, case) pairs**.

    SPEC: 04 §4.3
    SPEC: 04 §4.4

    ```
    proxy(u, t) = nu_ut = r_ut * ||delta_ut||_2 * w_t                       (4.2)
    exact(u, t) = L_t( m_u = 0 ) - L_t
    ```

    **[NORMATIVE]** Spearman between the two is logged; below `0.5` on `>= 64`
    pairs the audit fraction **doubles**, capped at 10%. §04.10 makes the same
    threshold a falsifier.

    **[NORMATIVE — v3.1 correction, audit A-67]** v3.0 compared, per *unit*, the
    quantity `sum_t r_ut ||delta_ut||` against the change in the **mean** probe
    loss. That is not (4.2) and it is not `L(t | m_u = 0)`:

    * `w_t` was **absent**. (4.2) is a product of three factors and §04.6's own
      note says all three must come from the same moment on the same token; the
      proxy the audit scored was the two-factor magnitude, which is not the
      quantity selection uses and not the quantity §04.10 is a falsifier for.
    * The units were compared on a quantity summed over **all** probe tokens
      against a counterfactual averaged over **all** probe tokens, so a single
      pair mixed every case together. §04.4 says "a random `1%` of **cases**",
      and `nu_ut` is indexed by `(u, t)`: the pairs are per case, which is also
      what makes `>= 64` of them reachable within a lifetime.
    * The exact term was the change in the batch **mean**, diluting a per-case
      effect by `1/(B*T)` and mixing in every other case's response.

    Measured consequence: `rho = 0.155` on a healthy M-nano lifetime, tripping
    §04.10's falsifier on a comparison that was not testing it. The diagnostic
    that identified it: 45% of the per-unit "exact" values were **negative**,
    which an unsigned two-factor magnitude cannot rank even in principle.

    **Counterfactual by full forward.** §04.4 permits a partial forward from the
    Site's layer to the head as a cost saving; a full forward is strictly more
    faithful and is affordable because the cost is one forward per **distinct
    unit** in the sample, not one per pair.

    Returns the Spearman when the accumulated buffer reached `min_pairs`, else
    `None`.
    """
    # (4.1) requires theta_lag. Before its first update the lagged self *is* the
    # model, `w_t` is identically zero, and every proxy would tie at zero -- a
    # correlation over a constant vector is undefined, not small.
    if lag_model is None or lagged is None or lagged.n_updates == 0:
        return None

    base_out = model(probe, genome=genome, want_metabolic=False)
    base_tok = _token_losses(base_out.logits, probe)          # (B, T-1)
    lagged.load_into(lag_model)
    lag_model.eval()
    lag_tok = _token_losses(
        lag_model(probe, genome=genome, want_metabolic=False).logits, probe
    )
    w = (lag_tok - base_tok).clamp_min(0.0)                   # (B, T-1)

    so = base_out.site_outputs[0]
    slots = so.slots[:, : w.shape[1]]                         # (B, T-1, k_route)
    r_ut = so.r_ut[:, : w.shape[1]]
    dn = so.delta_norm[:, : w.shape[1]]

    # A case is eligible when it carries a non-zero improvement signal and at
    # least one live routed unit. Cases with `w_t = 0` contribute a block of
    # ties at proxy zero whose exact values are unconstrained -- they would
    # depress the correlation without measuring the proxy (§04.4's reason for
    # skipping units that carry no traffic, applied to cases).
    eligible = (w > 0) & (slots >= 0).any(dim=-1)
    cand = torch.nonzero(eligible, as_tuple=False)
    if cand.shape[0] < 2:
        return None
    # §04.4's `1%` of cases is the *floor* from the doubling rule, not a
    # sampling plan: at the integration configuration it yields 2 cases and
    # ~8 pairs per event, so a whole lifetime accumulated **40** pairs against
    # the 64-pair minimum and the Spearman was never computed at all (audit
    # A-84). Sampling at least `min_pairs / k_route` cases per event makes the
    # minimum reachable in about four events at any scale, while the doubling
    # rule still raises the fraction when the correlation is poor.
    floor_cases = -(-audit.min_pairs // max(1, cfg.k_route))
    n_cases = max(2, floor_cases, audit.due(int(cand.shape[0])))
    take = rng.choice(int(cand.shape[0]), size=min(n_cases, int(cand.shape[0])),
                      replace=False)

    # Group the pairs by unit so the counterfactual costs one forward per
    # distinct unit rather than one per pair.
    by_slot: Dict[int, List[Tuple[int, int, float]]] = {}
    for idx in take:
        b, t = int(cand[idx, 0]), int(cand[idx, 1])
        wt = float(w[b, t])
        for j in range(cfg.k_route):
            slot = int(slots[b, t, j])
            if slot < 0:
                continue
            nu = float(r_ut[b, t, j]) * float(dn[b, t, j]) * wt
            by_slot.setdefault(slot, []).append((b, t, nu))

    # **Cost bound.** The counterfactual costs one forward per *distinct unit*,
    # and the doubling rule (§04.4) raises `fraction` from 1% to 10% when the
    # correlation is poor. At M1 that is `0.10 * 8 * 512 = 410` cases times
    # `k_route = 8`, i.e. up to 3280 forwards on a `(8, 512)` probe **per
    # selection event** -- which would take longer than the training between
    # events and make the audit the run.
    #
    # 128 distinct units is twice `min_pairs`, so a single event still supplies
    # more than the 64 pairs §04.4 requires and the doubling rule keeps its
    # effect on the *estimate*; only its effect on the *cost* is capped. The
    # units kept are the first 128 of an already-uniform sample, so no bias is
    # introduced.
    if len(by_slot) > MAX_AUDIT_UNITS:
        keep = list(by_slot)[:MAX_AUDIT_UNITS]
        by_slot = {k: by_slot[k] for k in keep}

    proxy: List[float] = []
    exact: List[float] = []
    for slot, entries in by_slot.items():
        saved = float(pop.m_u[slot])
        pop.m_u[slot] = 0.0
        pop.invalidate_active_count()
        off_tok = _token_losses(
            model(probe, genome=genome, want_metabolic=False).logits, probe
        )
        pop.m_u[slot] = saved
        pop.invalidate_active_count()
        for b, t, nu in entries:
            proxy.append(nu)
            exact.append(float(off_tok[b, t]) - float(base_tok[b, t]))

    if len(proxy) < 2:
        return None
    return audit.record(proxy, exact)


@torch.no_grad()
def _token_losses(logits: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
    """Per-position cross-entropy `L_t`, aligned as §04.6 aligns the reservoir.

    `logits[:, t]` predicts `probe[:, t + 1]`, so the returned tensor is
    `(B, T - 1)` and index `t` is the *case* at position `t` — the same
    convention `_lag_rows` and `CaseReservoir.push` use, which is what makes
    `r_ut`, `||delta_ut||` and `w_t` in (4.2) come from one moment on one token.
    """
    lf = logits.float()[:, :-1]
    logZ = torch.logsumexp(lf, dim=-1)
    tgt = probe[:, 1:]
    g = lf.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return logZ - g


def _probe_loss(out, probe: torch.Tensor) -> float:
    lf = out.logits.float()
    logZ = torch.logsumexp(lf, dim=-1)
    tgt = torch.zeros_like(probe)
    tgt[:, :-1] = probe[:, 1:]
    g = lf.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float((logZ - g)[:, :-1].mean())


@torch.no_grad()
def _offer_credits(
    model, out, batch, cfg, reservoirs, policy, rngs, lag_model, lagged,
    genome, device,
) -> None:
    """Offer one micro-batch's credits to the reservoirs (§04.6).

    Called for every micro-batch except the last, which the main loop handles
    together with the utility and mask updates. Splitting it this way keeps the
    reservoir a uniform sample over the whole interval — sampling only the last
    micro-batch of each step would bias it toward whatever the stream happened
    to emit there.
    """
    n_rows = batch.tokens.shape[0]
    n_lag = max(1, int(round(policy.d_rate * n_rows)))
    rows = torch.from_numpy(
        rngs.np("downsample").choice(n_rows, size=n_lag, replace=False)
    ).to(device)
    with torch.no_grad():
        lf = out.logits.float()
        logZ = torch.logsumexp(lf, dim=-1)
        gathered = lf.gather(-1, batch.targets.unsqueeze(-1)).squeeze(-1)
        lpt = logZ - gathered
    w_rows = _lag_rows(lag_model, lagged, genome, batch, lpt, rows)
    keep = batch.scored[rows].reshape(-1)
    if not bool(keep.any()):
        return
    for li in range(len(reservoirs)):
        so = out.site_outputs[li]
        reservoirs[li].push(
            batch.tokens[rows].reshape(-1)[keep],
            so.slots[rows].reshape(-1, cfg.k_route)[keep],
            so.delta_norm[rows].reshape(-1, cfg.k_route).detach()[keep],
            so.r_ut[rows].reshape(-1, cfg.k_route).detach()[keep],
            w_rows.reshape(-1)[keep],
            rngs.np("downsample"),
        )


@torch.no_grad()
def _lag_rows(
    lag_model, lagged: Optional[LaggedSelf], genome, batch: Batch,
    loss_per_token: torch.Tensor, rows: torch.Tensor,
) -> torch.Tensor:
    """(4.1): `w_t = max(0, L_lag(t) - L_theta(t))` on the offered rows.

    SPEC: 04 §4.3
    SPEC: 06 §6.1.2

    Evaluated **at push time**, on the `d_rate` fraction of rows the reservoir is
    being offered — the 10% §06.1.2 budgets — so all three factors of (4.2) come
    from the same moment on the same token (§04.6 v3.1, audit A-53).

    Returns `(len(rows), T)` weights. Before `theta_lag` has received its first
    update it *is* the model, so `w_t` is identically zero; §04.5's `SHUFFLE`
    falls back to a uniform permutation there, which is the correct degenerate
    behaviour rather than an error.

    **[NORMATIVE — polarity]** `L_lag - L_theta`, not the reverse. A lagged self
    knows **less** than the current model, so `L_theta <= L_lag` almost
    everywhere; spec v1's `max(0, L_theta - L_lag)` was identically zero.
    """
    sub_loss = loss_per_token[rows]
    if lagged is None or lag_model is None or lagged.n_updates == 0:
        return torch.zeros_like(sub_loss)
    # A no-op unless theta_lag has moved since the last call, i.e. at most once
    # per T_ref steps -- otherwise a full state_dict copy runs every step.
    lagged.load_into(lag_model)
    lag_model.eval()
    o = lag_model(
        batch.tokens[rows],
        genome=genome,
        doc_id=batch.doc_id[rows],
        valid=batch.valid[rows],
        want_metabolic=False,
    )
    lf = o.logits.float()
    logZ = torch.logsumexp(lf, dim=-1)
    tgt = batch.targets[rows]
    gathered = lf.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    lag_loss = logZ - gathered
    return (lag_loss - sub_loss).clamp_min(0.0)


@torch.no_grad()
def _evaluate_bifurcation(
    model, genome, bif, cohort, probe, n: int, n_live: int = 1
) -> bool:
    """(1.21) against a masked counterfactual.

    Both terms use the same weights and the same fixed probe batch and differ
    only in whether the cohort born at `n_b` is switched on. Measuring **at** the
    birth instead — as v2 did — gives a difference Theorem 3.1a makes identically
    zero, so the alarm was unreachable and `rho <- rho/2` was dead code
    (audit A-36).

    **[NORMATIVE — the statistic is `L_task`, not a proxy for it; audit A-73]**
    v3.0 measured `logits.square().mean()` here and in both inputs to
    `epsilon_bif`. That is a scalar with **no dependence on the targets** — it
    responds to the scale of the output distribution, not to the model's loss —
    and (1.21) names `L_task`. The two are not a monotone reparameterisation of
    each other: masking cohorts of 4, 8 and 16 units moved the surrogate by
    `0.73`, `0.22` and `0.03` times the corresponding relative change in
    `L_task`, so the alarm's ordering over cohorts was not the ordering (1.21)
    specifies. `_probe_loss` — which computes exactly the probe cross-entropy —
    already existed in this module and was **dead code**.

    This is the fourth defect in this criterion (A-36, A-58, A-61, A-73), and
    the first three were all about the *threshold*. This one is about the
    *statistic*, which no amount of threshold work could have reached.
    """
    pop = model.populations[cohort.site]
    saved = pop.m_u[cohort.slots].clone()
    active = _probe_loss(
        model(probe, genome=genome, want_metabolic=False), probe
    )
    pop.m_u[cohort.slots] = 0.0
    pop.invalidate_active_count()
    masked = _probe_loss(
        model(probe, genome=genome, want_metabolic=False), probe
    )
    pop.m_u[cohort.slots] = saved
    pop.invalidate_active_count()
    return bif.evaluate(cohort, active, masked, n, n_live)


def _capture_arms(
    rings: Dict[str, BatchRing], archive: TaskArchive,
    frozen_archive: Optional[TaskArchive], packer: SyntheticPacker,
    n: int, N: int, cfg: SomaConfig, rngs: RngBundle,
    ramp: RampController, solve_rate_fn=None,
) -> None:
    """Capture one batch for **each** certification arm (§06.4).

    The three arms differ only in which generator produced the batch:

    * **live**   — the evolving `Lambda`;
    * **frozen** — `Lambda` fixed after `SEED_ARCHIVE`;
    * **ramp**   — a monotone ratchet on `n_len` alone, **matched** to the live
      arm's mean solve rate (§06.4, audit A-79).

    Capturing all three from the same step against the same checkpoint ring is
    what makes `live - ramp` attributable to the archive rather than to the
    observer.
    """
    rng = rngs.np("archive")
    rows = 8

    def pack(params_list: List[TaskParams]) -> torch.Tensor:
        insts = [generate(p, rng) for p in params_list]
        toks, _, _ = packer.pack(insts)
        return torch.from_numpy(toks)

    rings["live"].offer(n, pack([archive.sample_params(rng) for _ in range(rows)]))
    if frozen_archive is not None:
        rings["frozen"].offer(
            n, pack([frozen_archive.sample_params(rng) for _ in range(rows)])
        )
    # §06.4: matched to the live arm's mean solve-rate trajectory. Both rates
    # are measured on the **same** model at the **same** step, which is what
    # makes the ratchet a tracker of the live arm rather than of the schedule.
    # With no `solve_rate_fn` there is nothing to match against, and the arm
    # falls back to holding its current `n_len` rather than silently ramping on
    # a schedule -- an unmatched ramp makes `live - ramp` a comparison of two
    # task distributions instead of two archive dynamics (audit A-79).
    ramp_params = TaskParams("PARITY", n_len=ramp.n_len)
    if solve_rate_fn is not None:
        s_live = archive.mean_solve_rate()
        s_ramp = float(solve_rate_fn(ramp_params, N_EVAL_RAMP))
        ramp.observe(s_live, s_ramp)
    rings["ramp"].offer(
        n, pack([TaskParams("PARITY", n_len=ramp.n_len) for _ in range(rows)])
    )


@torch.no_grad()
def _certify_three_arms(
    model, genome, ckpt_ring: CheckpointRing, rings: Dict[str, BatchRing],
    n: int, device,
) -> Optional[CertificationVerdict]:
    """Build (6.11) per arm and run the Mann-Kendall tests (§06.4)."""
    ckpts = ckpt_ring.entries()
    if len(ckpts) < 2:
        return None
    if any(len(r.entries()) < 2 for r in rings.values()):
        return None

    saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    steps_t = [s for s, _ in ckpts]
    results = {}
    try:
        mats: Dict[str, np.ndarray] = {
            arm: np.zeros((len(ckpts), len(rings[arm].entries()))) for arm in rings
        }
        for j, (_, snap) in enumerate(ckpts):
            model.load_state_dict({k: v.to(device) for k, v in snap.items()})
            for arm, ring in rings.items():
                for k, (_, xb) in enumerate(ring.entries()):
                    mats[arm][j, k] = _nll(model, genome, xb.to(device))
    finally:
        model.load_state_dict({k: v.to(device) for k, v in saved.items()})

    for arm, ring in rings.items():
        steps_T = [s for s, _ in ring.entries()]
        results[arm] = certify(mats[arm], steps_t, steps_T, arm=arm)
    return CertificationVerdict(results["live"], results["frozen"], results["ramp"], n)


@torch.no_grad()
def _nll(model, genome, xb: torch.Tensor) -> float:
    o = model(xb, genome=genome, want_metabolic=False)
    lf = o.logits.float()
    logZ = torch.logsumexp(lf, -1)
    tgt = torch.zeros_like(xb)
    tgt[:, :-1] = xb[:, 1:]
    g = lf.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float((logZ - g)[:, :-1].mean())


def _harvest(model, policy, soma_on: bool):
    """§07.2 step 7: the gene sets of units surviving with `nu_hat` above the
    median. **Only** selection statistics of germline objects cross the barrier."""
    if not soma_on:
        return [], 0.0, 0.0
    archetypes: List[Tuple[int, ...]] = []
    total_util = 0.0
    for pop in model.populations:
        nh = nu_hat(pop, policy.eta_ema)
        live = pop.live
        if not bool(live.any()):
            continue
        med = float(nh[live].median())
        keep = live & (nh > med)
        for slot in torch.nonzero(keep, as_tuple=False).squeeze(-1).tolist():
            archetypes.append(tuple(int(g) for g in pop.g_u[slot].tolist()))
        total_util += float(nh[live].mean())
    counts = model.populations[0].clade_counts()
    mean_clade = float(np.mean(list(counts.values()))) if counts else 0.0
    return archetypes, total_util / max(1, len(model.populations)), mean_clade


def _separation(n: int, cfg: SomaConfig, policy) -> float:
    """(1.20): `sep(n) = (b_str(n)/b_str(0)) / (a_lr(n)/lr_max)`.

    SPEC: 01 §1.7.1

    Logged for every step but **asserted** only for `n_warm <= n < n_dec`. v2
    exempted warmup but not decay; as `a_lr -> 0` the ratio diverges
    (`sep = 6.31` at `n = 12299` at M1), so the assertion aborted the final steps
    of every run.
    """
    lr = wsd_lr(n, cfg)
    if lr <= 0:
        return float("inf")
    b0 = policy.rho_0 / cfg.T_0 if cfg.T_0 else 0.0
    if b0 <= 0:
        return 0.0
    bn = policy.rho_0 * math.exp(-n / cfg.n_c) / cfg.T_sel(n)
    return (bn / b0) / (lr / cfg.lr_max)
