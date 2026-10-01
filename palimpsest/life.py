"""The life: phases, births, the free lifetime with its labelled regularisers, consolidation,
the scores and the compute accounting.

SPEC: docs/palimpsest/01-model.md §1.2 (the lifetime), §1.5-1.8 (consolidation, the life,
the controls, the scores)
"""

from __future__ import annotations

import copy
import dataclasses
import math
import statistics
import time
from typing import Callable, Optional

import torch

from anlage.lifetime import evaluate
from anlage.model import Phenotype, free_init
from anlage.recipe import TrainConfig, batch_at, build_optimizers, clip_grads, micro_batch_for, set_lr
from bottleneck.loop import fisher_diag

from .distil import distil, distil_loss, dream
from .envs import Environment, LifeSpec, MixPool
from .expand import FULL, core_of, expand, ratio, shrink

__all__ = ["LifeConfig", "ARMS", "run_life", "scores_of", "train_phase", "replay_tokens_for"]

ARMS = ("copy", "scratch", "shrink_perturb", "sieve", "sieve-nodream", "sieve-fresh", "sieve-life", "trunc",
        "replay", "ewc", "lwf", "iid", "copy-dream")
SIEVE_ARMS = ("sieve", "sieve-nodream", "sieve-fresh", "sieve-life")
EVAL_FRACS = (0.10, 0.25, 0.50, 1.0)


@dataclasses.dataclass(frozen=True)
class LifeConfig:
    """Every dial of 01 §1.9 with its default."""

    arm: str
    core: str = "c8"                      # the core for sieve/trunc arms; the byte reference for replay
    tokens_per_phase: int = 8_000_000
    batch: int = 32
    T: int = 512
    noise: float = 0.01
    eps_gain: float = 0.02
    new_units: str = "copy"
    distil_tokens: int = 2_000_000
    dream_tokens: int = 500_000
    T_dream: int = 256
    lam_dream: float = 1.0
    temperature: float = 1.0
    core_init: str = "prev"
    dream_start: str = "mixed"           # eos | uniform | mixed (P-P.5)
    replay_frac: float = 0.1
    replay_tokens: int = 0               # 0 = the byte rule of §1.7 for `core`
    lam_ewc: float = 1e3
    lam_lwf: float = 1.0
    sp_lambda: float = 0.6
    sp_p: float = 0.1
    eval_fracs: tuple[float, ...] = EVAL_FRACS
    compile: Optional[str] = None
    n_fisher: int = 32
    dream_batch: int = 32

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm in {ARMS}")
        if self.core_init not in ("prev", "shrink"):
            raise ValueError("core_init in prev|shrink")
        if self.new_units not in ("copy", "fresh"):
            raise ValueError("new_units in copy|fresh")
        if self.dream_start not in ("eos", "uniform", "mixed"):
            raise ValueError("dream_start in eos|uniform|mixed")
        if not (0 <= self.replay_frac < 1):
            raise ValueError("replay_frac in [0, 1)")


def replay_tokens_for(core_cfg, vocab: int) -> int:
    """The byte rule: the buffer's bytes at 2 per token equal the core's at 4 per parameter.
    SPEC: 01 §1.7."""
    return 2 * core_cfg.n_params(vocab)


# ------------------------------------------------------------------------------ the lifetime
def train_phase(model: Phenotype, pool, tc: TrainConfig, *, order_seed: int, evals: dict,
                eval_fracs=EVAL_FRACS, extra_loss: Optional[Callable] = None,
                batch_transform: Optional[Callable] = None, log=None, device: str = "cuda",
                train_log_every: int = 50) -> dict:
    """ANLAGE's lifetime loop (`anlage/lifetime.py::train`) with two hooks for the labelled
    controls: `extra_loss(model, window) -> scalar` added to the objective, and
    `batch_transform(window, step) -> window` applied before the step (replay). Evaluations on
    every environment of `evals` at the fractions of the phase. SPEC: 01 §1.2, §1.7."""
    model.to(device).train()
    muon, adam = build_optimizers(model, tc)
    gen = pool.batches(batch=tc.batch_final, steps=1 << 30, seed=order_seed)
    tokens_done, step = 0, 0
    fracs = sorted(set(float(f) for f in eval_fracs) | {1.0})
    marks = [(f, int(round(f * tc.tokens))) for f in fracs]
    next_mark = 0
    curve: dict[str, dict[str, float]] = {}
    train_curve, step_ms = [], []
    ema = None
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ev_a = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    ev_b = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    while tokens_done < tc.tokens:
        p = tokens_done / tc.tokens
        B = batch_at(p, tc)
        set_lr(p, tc, muon, adam)
        window = next(gen)[:B]
        if batch_transform is not None:
            window = batch_transform(window, step)
        if ev_a is not None:
            ev_a.record()
        micro = B if tc.micro_batch <= 0 else min(B, tc.micro_batch)
        ce_acc = 0.0
        for w in window.split(micro):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                ce_m, zl_m = model.loss(w)
                tot = ce_m + zl_m
                if extra_loss is not None:
                    tot = tot + extra_loss(model, w)
            (tot * (w.shape[0] / B)).backward()
            ce_acc += float(ce_m) * (w.shape[0] / B)
        gnorm = clip_grads(model, tc)
        if not math.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        muon.step(); adam.step()
        muon.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True)
        if ev_b is not None:
            ev_b.record(); ev_b.synchronize()
            step_ms.append(ev_a.elapsed_time(ev_b))
        tokens_done += B * tc.T
        step += 1
        ema = ce_acc if ema is None else 0.98 * ema + 0.02 * ce_acc
        if train_log_every and step % train_log_every == 0:
            train_curve.append((tokens_done, ema))
            if log:
                log(f"    step {step:6d}  tokens {tokens_done:>12,}  train {ema:.4f}  "
                    f"lr_M {muon.param_groups[0]['lr']:.2e}  B {B}  "
                    f"{step_ms[-1] if step_ms else float('nan'):.0f} ms")
        while next_mark < len(marks) and tokens_done >= marks[next_mark][1]:
            f = marks[next_mark][0]
            curve[f"{f:.2f}"] = {name: evaluate(model, ev)[0] for name, ev in evals.items()}
            if log:
                log(f"  eval @{f:.2f}  " + "  ".join(f"{n} {c:.4f}" for n, c in curve[f"{f:.2f}"].items()))
            next_mark += 1
    if "1.00" not in curve:
        curve["1.00"] = {name: evaluate(model, ev)[0] for name, ev in evals.items()}
    seconds = time.time() - t0
    return {"evals": curve, "tokens": tokens_done, "steps": step,
            "ms_per_step_median": statistics.median(step_ms) if step_ms else float("nan"),
            "tokens_per_s": tokens_done / max(seconds, 1e-9), "seconds": seconds,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            "train_curve": train_curve}


# ------------------------------------------------------------------------------- the controls
class EWC:
    """`(lam/2) sum_j sum_i F_{j,i} (w_i - w_{j,i})^2`. SPEC: 01 §1.7."""

    def __init__(self, lam: float):
        self.lam = lam
        self.stores: list[tuple[dict, dict]] = []

    def add(self, model: Phenotype, windows: torch.Tensor) -> None:
        F = fisher_diag(model, windows)
        W = {k: v.detach().clone() for k, v in model.weights().items()}
        self.stores.append((F, W))

    def penalty(self, model: Phenotype) -> torch.Tensor:
        live = model.weights()
        acc = None
        for F, W in self.stores:
            for k, w in live.items():
                term = (F[k] * (w.float() - W[k]) ** 2).sum()
                acc = term if acc is None else acc + term
        if acc is None:
            return next(model.parameters()).new_zeros(())
        return 0.5 * self.lam * acc


class Replay:
    """A buffer of windows; `replay_frac` of every batch's rows replaced by buffer windows
    drawn by a seeded generator. SPEC: 01 §1.7."""

    def __init__(self, capacity_tokens: int, T: int, frac: float, seed: int, device: str):
        self.capacity = capacity_tokens // (T + 1)
        self.frac, self.device = frac, device
        self.buffer: Optional[torch.Tensor] = None
        self.g = torch.Generator(device="cpu").manual_seed(int(seed))

    def add(self, windows: torch.Tensor) -> None:
        w = windows.to(self.device)
        self.buffer = w if self.buffer is None else torch.cat([self.buffer, w], 0)
        if self.buffer.shape[0] > self.capacity:
            keep = torch.randperm(self.buffer.shape[0], generator=self.g)[: self.capacity].to(self.device)
            self.buffer = self.buffer.index_select(0, keep)

    def __call__(self, window: torch.Tensor, step: int) -> torch.Tensor:
        if self.buffer is None or self.frac <= 0:
            return window
        n = int(math.ceil(self.frac * window.shape[0]))
        L = window.shape[1]
        if self.buffer.shape[1] >= L:
            idx = torch.randint(0, self.buffer.shape[0], (n,), generator=self.g).to(self.device)
            rows = self.buffer.index_select(0, idx)[:, :L]
        else:
            # shorter buffer windows (dreams of T_dream tokens): several are concatenated per
            # row, so a rehearsal row is a sequence of dreams, each a document from the start token
            per = -(-L // self.buffer.shape[1])
            idx = torch.randint(0, self.buffer.shape[0], (n * per,), generator=self.g).to(self.device)
            rows = self.buffer.index_select(0, idx).view(n, per * self.buffer.shape[1])[:, :L]
        out = window.clone()
        out[:n] = rows
        return out

    def n_tokens(self) -> int:
        return 0 if self.buffer is None else int(self.buffer.shape[0] * (self.buffer.shape[1] - 1))


# ---------------------------------------------------------------------------------- births
@torch.no_grad()
def birth(cfg: LifeConfig, W_prev: Phenotype, core: Optional[Phenotype], *, seed: int, k: int,
          vocab: int, full=FULL, device: str = "cuda") -> tuple[Phenotype, dict]:
    """`BIRTH(arm, W, C, seed, k)` of 01 §1.6."""
    arm = cfg.arm
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 100003 + k)
    if arm in ("copy", "replay", "ewc", "lwf", "copy-dream"):
        return W_prev, {"birth": "copy"}
    if arm == "scratch":
        return Phenotype(full, vocab, seed=seed * 1009 + k).to(device), {"birth": "scratch"}
    if arm == "shrink_perturb":
        fresh = Phenotype(full, vocab, seed=seed * 1009 + k).to(device)
        for key, w in W_prev.weights().items():
            r, l = key
            w.mul_(cfg.sp_lambda).add_(cfg.sp_p * fresh.get(r, l))
        return W_prev, {"birth": "shrink_perturb", "lambda": cfg.sp_lambda, "p": cfg.sp_p}
    if arm in SIEVE_ARMS:
        if core is None:
            raise RuntimeError("a sieve birth needs a core")
        new = "fresh" if arm == "sieve-fresh" else cfg.new_units
        W0 = expand(core, full, noise=cfg.noise, eps_gain=cfg.eps_gain, generator=g, new=new)
        return W0, {"birth": "expand", "new": new, "core": core.cfg.summary()}
    if arm == "trunc":
        C = shrink(W_prev, core_of(cfg.core, full))
        W0 = expand(C, full, noise=cfg.noise, eps_gain=cfg.eps_gain, generator=g, new=cfg.new_units)
        return W0, {"birth": "truncate+expand", "core": C.cfg.summary()}
    raise ValueError(arm)


# ------------------------------------------------------------------------------ the scores
def scores_of(record: dict) -> dict:
    """The scores of 01 §1.8 from a record's phases and consolidations."""
    phases = record["phases"]
    envs_all = []
    for ph in phases:
        if ph["env"] not in envs_all:
            envs_all.append(ph["env"])
    final = phases[-1]["evals"]["1.00"]
    U = statistics.mean(final[e] for e in envs_all)
    lam_terms = []
    retention, adaptation = {}, {}
    seen: list[str] = []
    best_seen: dict[str, float] = {}
    for k, ph in enumerate(phases, 1):
        e_k = ph["env"]
        if e_k not in seen:
            seen.append(e_k)
        ev1 = ph["evals"]["1.00"]
        lam_terms.append(statistics.mean(ev1[e] for e in seen))
        for e in seen:
            if e != e_k and e in best_seen:
                retention[f"{k}:{e}"] = ev1[e] - best_seen[e]
        adaptation[str(k)] = {f: ph["evals"][f][e_k] for f in ph["evals"]}
        for e in seen:
            best_seen[e] = min(best_seen.get(e, float("inf")), ev1[e])
    out = {"U_K": U, "Lambda": statistics.mean(lam_terms), "retention": retention,
           "adaptation": adaptation, "envs": envs_all, "final": final}
    cons = record.get("consolidations") or []
    if cons and cons[-1].get("core_evals"):
        ce = cons[-1]["core_evals"]
        out["G_K"] = statistics.mean(ce[e] for e in envs_all if e in ce)
        out["core_final"] = ce
        out["core_curve"] = [c.get("core_evals") for c in cons]
    return out


# ------------------------------------------------------------------------------- the life
def run_life(spec: LifeSpec, cfg: LifeConfig, *, seed: int, envs: dict[str, Environment],
             log: Callable[[str], None], device: str = "cuda", full=FULL) -> dict:
    """`LIFE(arm, phases, T_phi, core, seed)` of 01 §1.6, returning the record."""
    vocab = next(iter(envs.values())).vocab
    N_full = full.n_params_nonemb(vocab)
    core_cfg = core_of(cfg.core, full)
    N_core = core_cfg.n_params_nonemb(vocab)
    tc = TrainConfig(tokens=cfg.tokens_per_phase, batch_final=cfg.batch, T=cfg.T,
                     micro_batch=micro_batch_for(full.n_params(vocab), cfg.batch),
                     lr_provenance="measured: ANLAGE R2 at s0 (docs/anlage/RESULTS.md)")
    evals = {name: e.test for name, e in envs.items()}
    record = {"spec": spec.as_list(), "cfg": dataclasses.asdict(cfg), "seed": seed,
              "full": full.summary(), "core": core_cfg.summary(), "ratio": ratio(full, core_cfg, vocab),
              "n_params": full.n_params(vocab), "n_params_core": core_cfg.n_params(vocab),
              "phases": [], "consolidations": [], "births": [], "complete": False,
              "flops": {"train": 0.0, "overhead": 0.0}}
    W = Phenotype(full, vocab, seed=seed).to(device)
    core: Optional[Phenotype] = None
    ewc = EWC(cfg.lam_ewc) if cfg.arm == "ewc" else None
    replay = None
    if cfg.arm == "replay":
        cap = cfg.replay_tokens or replay_tokens_for(core_cfg, vocab)
        replay = Replay(cap, cfg.T, cfg.replay_frac, seed * 31 + 7, device)
        record["replay_capacity_tokens"] = cap
    teacher_prev: Optional[Phenotype] = None
    t_life = time.time()
    if cfg.arm == "iid":
        pools = [envs[e].pool(p) for e, p, _ in spec.phases]
        mix = MixPool(pools)
        tc_all = dataclasses.replace(tc, tokens=cfg.tokens_per_phase * len(spec.phases))
        log(f"iid: one lifetime of {tc_all.tokens:,} tokens on the mixture of {len(pools)} pools")
        res = train_phase(W, mix, tc_all, order_seed=spec.phases[0][2], evals=evals,
                          eval_fracs=cfg.eval_fracs, log=log, device=device)
        record["phases"].append({"k": 1, "env": "+".join(spec.envs), **res})
        # the union score of `iid` is its final evaluation on every environment
        record["phases"][-1]["env"] = spec.envs[-1]
        record["flops"]["train"] = 6.0 * N_full * tc_all.tokens
        record["scores"] = {"U_K": statistics.mean(res["evals"]["1.00"][e] for e in spec.envs),
                            "final": res["evals"]["1.00"], "envs": spec.envs, "Lambda": float("nan"),
                            "retention": {}, "adaptation": {}}
        record["seconds"] = time.time() - t_life
        record["complete"] = True
        return record
    for k, (env_name, pool_seed, order_seed) in enumerate(spec.phases, 1):
        env = envs[env_name]
        birth_evals = None
        if k > 1 and cfg.arm in ("sieve-life", "copy-dream"):
            # SPEC: 01 section 1.7 -- rehearsal in the lifetime from a *generated* buffer: the
            # core's dreams (sieve-life) or the previous phenotype's own samples (copy-dream);
            # the buffer's token count by the byte rule, so these arms rehearse the same volume
            # as `replay` does from stored data
            src = core if cfg.arm == "sieve-life" else W
            if src is None:
                raise RuntimeError("sieve-life needs a core")
            cap = cfg.replay_tokens or replay_tokens_for(core_cfg, vocab)
            n_seq = max(1, cap // cfg.T_dream)
            t1 = time.time()
            buf = dream(src, n_seq, cfg.T_dream, temperature=cfg.temperature, seed=seed * 19 + k, device=device,
                        start=cfg.dream_start)
            N_src = src.cfg.n_params_nonemb(vocab)
            record["flops"]["overhead"] += 2.0 * N_src * n_seq * cfg.T_dream
            replay = Replay(cap, cfg.T_dream, cfg.replay_frac, seed * 31 + 7 + k, device)
            replay.add(buf)
            log(f"  rehearsal buffer: {n_seq} dreams x {cfg.T_dream} from "
                f"{'the core' if cfg.arm == 'sieve-life' else 'the previous phenotype'} in {time.time() - t1:.0f}s")
            del buf
        if k > 1:
            W, info = birth(cfg, W, core, seed=seed, k=k, vocab=vocab, full=full, device=device)
            info["k"] = k
            birth_evals = {name: evaluate(W, ev)[0] for name, ev in evals.items()}
            info["evals"] = birth_evals
            record["births"].append(info)
            log(f"phase {k}: birth {info}")
        pool = env.pool(pool_seed)
        extra = None
        if ewc is not None and ewc.stores:
            extra = lambda m, w, _e=ewc: _e.penalty(m)                        # noqa: E731
        if cfg.arm == "lwf" and teacher_prev is not None:
            extra = lambda m, w, _t=teacher_prev, _l=cfg.lam_lwf: _l * distil_loss(m, _t, w)   # noqa: E731
        log(f"phase {k}/{len(spec.phases)}: {env_name}  pool {pool_seed} ({pool.digest()[:8]})  order {order_seed}  "
            f"{cfg.tokens_per_phase:,} tokens  arm {cfg.arm}")
        res = train_phase(W, pool, tc, order_seed=order_seed, evals=evals, eval_fracs=cfg.eval_fracs,
                          extra_loss=extra, batch_transform=replay, log=log, device=device)
        res.update({"k": k, "env": env_name, "pool_digest": pool.digest()})
        if birth_evals is not None:
            res["evals"]["0.00"] = birth_evals
        record["phases"].append(res)
        record["flops"]["train"] += 6.0 * N_full * res["tokens"]
        if cfg.arm == "lwf" and teacher_prev is not None:
            record["flops"]["overhead"] += 2.0 * N_full * res["tokens"]
        # ---- the slow timescale --------------------------------------------------------
        if cfg.arm in SIEVE_ARMS:
            core, cons = consolidate(cfg, W, core, env, k=k, seed=seed, vocab=vocab, full=full,
                                     device=device, log=log, N_full=N_full, N_core=N_core)
            cons["core_evals"] = {name: evaluate(core, ev)[0] for name, ev in evals.items()}
            log(f"  core after phase {k}: " + "  ".join(f"{n} {c:.4f}" for n, c in cons["core_evals"].items()))
            record["consolidations"].append(cons)
            record["flops"]["overhead"] += cons["flops"]
        if cfg.arm == "replay" and replay is not None:
            n_add = replay.capacity // max(1, len(spec.phases) - 1)
            add = next(pool.batches(batch=min(n_add, pool.n_windows), steps=1, seed=order_seed + 99))
            replay.add(add)
            record.setdefault("replay_tokens", []).append(replay.n_tokens())
        if ewc is not None:
            ewc.add(W, env.fitness_windows(cfg.n_fisher))
            record["flops"]["overhead"] += 6.0 * N_full * cfg.n_fisher * cfg.T
        if cfg.arm == "lwf":
            teacher_prev = copy.deepcopy(W).eval()
            for p in teacher_prev.parameters():
                p.requires_grad_(False)
        if cfg.arm == "trunc":
            core = None
        del pool
        torch.cuda.empty_cache()
    record["scores"] = scores_of(record)
    record["flops"]["overhead_frac"] = record["flops"]["overhead"] / max(record["flops"]["train"], 1.0)
    record["seconds"] = time.time() - t_life
    record["complete"] = True
    return record


def consolidate(cfg: LifeConfig, W: Phenotype, core_prev: Optional[Phenotype], env: Environment, *,
                k: int, seed: int, vocab: int, full, device: str, log, N_full: int, N_core: int) -> tuple:
    """Definition 7: the new core from the phenotype on the current pool and from the previous
    core on its dreams. Returns `(core, record)`."""
    core_cfg = core_of(cfg.core, full)
    t0 = time.time()
    if core_prev is None or cfg.core_init == "shrink":
        core = shrink(W, core_cfg)
        init = "shrink"
    else:
        core = copy.deepcopy(core_prev)
        init = "prev"
    # the consolidation's own pool: a seed disjoint from every training pool of the life
    pool = env.pool(900_000 + 10 * k + seed)
    batches = pool.batches(batch=cfg.batch, steps=1 << 30, seed=seed * 7 + k)
    tc_c = TrainConfig(tokens=cfg.distil_tokens, batch_final=cfg.batch, T=cfg.T, micro_batch=16,
                       lr_provenance="the phenotype's recipe at the core's size")
    dreams = None
    n_dream = 0
    flops = 0.0
    if core_prev is not None and cfg.lam_dream > 0 and cfg.dream_tokens > 0 and cfg.arm != "sieve-nodream":
        n_dream = max(1, cfg.dream_tokens // cfg.T_dream)
        t1 = time.time()
        dreams = dream(core_prev, n_dream, cfg.T_dream, temperature=cfg.temperature,
                       seed=seed * 13 + k, device=device, start=cfg.dream_start)
        flops += 2.0 * N_core * n_dream * cfg.T_dream          # the cached sampler: one forward per token
        log(f"  dreamed {n_dream} x {cfg.T_dream} tokens from the previous core in {time.time() - t1:.0f}s")
    res = distil(core, W, batches, tc=tc_c, dreams=dreams, prev_core=core_prev, lam_dream=cfg.lam_dream,
                 dream_batch=cfg.dream_batch, log=log, device=device, seed=seed * 17 + k)
    flops += 2.0 * N_full * res["data_tokens"] + 6.0 * N_core * res["data_tokens"]
    flops += (2.0 * N_core + 6.0 * N_core) * res["dream_tokens"]
    for p in W.parameters():
        p.requires_grad_(True)
    rec = {"k": k, "init": init, "core": core_cfg.summary(), "pool_digest": pool.digest(),
           "n_dream": n_dream, "T_dream": cfg.T_dream, "flops": flops, "seconds": time.time() - t0,
           "distil": {kk: (v if kk not in ("curve", "dream_curve") else v[::5]) for kk, v in res.items()},
           "distil_final": res["curve"][-1] if res["curve"] else float("nan"),
           "dream_final": res["dream_curve"][-1] if res["dream_curve"] else float("nan")}
    del pool
    return core, rec
