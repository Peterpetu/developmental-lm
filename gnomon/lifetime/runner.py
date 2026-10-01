"""A reusable, compiled lifetime runner.

SPEC: 05 §5.1
SPEC: 05 §5.4
SPEC: 05 §5.5

The evolutionary loop of §08 runs thousands of lifetimes, so the cost of one is
the project's binding constraint. Measured at M-pico on an RTX 4080 SUPER:

| | ms / step |
|---|---|
| eager | 79.5 |
| `torch.compile(mode="default")` | 28.2 |
| **`torch.compile(mode="reduce-overhead")`** | **11.5** |

The step is **CPU-dispatch bound, not GPU bound** -- profiled at 131 ms of CPU
against 25 ms of CUDA per step, across roughly 4,000 kernel launches, because
the decode is a long chain of cheap elementwise operations over `(R, C, d_h)`
tensors. `reduce-overhead` captures the step as a CUDA graph and replays it,
which is why it wins where fusion alone (`default`) does not.

Reusing the graph is what makes it pay, and that dictates the design here: one
`Gnomon` instance for the whole run, into which each genome's values are
**copied**, rather than a fresh module per genome. A new module object would
retrigger a ~30 s compile every lifetime and lose more than the graph saves.
Everything that varies per step -- `eps(n)`, the batch -- is a device tensor
copied into a static buffer, never a Python float, so no step recompiles.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import time
import warnings
from typing import Callable, Optional

import torch

# Inductor probes for a C++ compiler whenever it emits a CPU kernel. On a
# Windows box without MSVC that raises `Compiler: cl is not found`, which would
# otherwise abort a run mid-generation. Skipping the vectorised-ISA probe
# removes most of it; `_compile` below handles the rest by degrading.
try:
    torch._inductor.config.cpp.vec_isa_ok = False
except Exception:                                    # pragma: no cover
    pass


def _compile(fn: Callable, mode: Optional[str], label: str) -> tuple[Callable, str]:
    """Compile `fn`, degrading rather than failing. SPEC: 07 §7.1.

    Every genome in an evolutionary run decodes its own graph, and some of those
    graphs make inductor emit a CPU kernel it cannot build in this environment.
    A run must not die on one architecture, so the ladder is
    `reduce-overhead -> default -> eager`, and which rung was used is returned
    so that a reported timing can never be silently attributed to the wrong
    backend.
    """
    if mode is None:
        return fn, "eager"
    ladder = ["reduce-overhead", "default"] if mode == "reduce-overhead" else [mode]
    for m in ladder:
        try:
            c = torch.compile(fn, mode=m)
            c()                                       # force the compile now
            return c, m
        except Exception as exc:                      # pragma: no cover - env
            warnings.warn(f"{label}: torch.compile(mode={m!r}) failed "
                          f"({type(exc).__name__}); degrading")
            torch._dynamo.reset()
    return fn, "eager"

from ..config import GenomeShape, PhenoConfig
from ..data.stream import (Shard, TokenStream, counts_digest, eval_batches,
                           token_counts)
from ..genome import codes as C
from ..genome.decoder import Decoder
from ..pheno.model import Gnomon
from ..pheno.plan import Arch
from .graph import capture_step
from .plasticity import ADAM_POINT, DecodedRule, N_RULE
from .train import LifetimeResult, eps_at, wsd_lr

__all__ = ["LifetimeRunner"]


class LifetimeRunner:
    """Owns one compiled phenotype and runs many lifetimes through it."""

    #: SPEC: 02 §2.9. ``freq`` is the specified token coding; ``bijection`` is
    #: §2.4's construction, retained *only* as the control that measured the
    #: difference. It is not a tuning knob: §2.9 records +0.0999 nats
    #: 95% BCa [+0.0288, +0.2647] for ``freq`` with plain AdamW.
    TOKEN_CODES = ("freq", "bijection")

    def __init__(self, cfg: PhenoConfig, shape: GenomeShape, *,
                 train_shard: Shard, eval_shard: Shard, device: str = "cuda",
                 arch: Optional[Arch] = None, compile_mode: Optional[str] =
                 "reduce-overhead", n_eval_batches: int = 8,
                 token_code: str = "freq"):
        # SPEC: 07 §7.1 -- one compiled graph per distinct architecture.
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 256)
        torch._dynamo.config.accumulated_cache_size_limit = max(
            getattr(torch._dynamo.config, "accumulated_cache_size_limit", 0), 2048)
        self.cfg, self.shape, self.device = cfg, shape, device
        self.train_shard, self.eval_shard = train_shard, eval_shard
        self.model = Gnomon(cfg, shape, arch=arch).to(device)
        # SPEC: 02 §2.9 -- the innate coordinate system, installed *before* any
        # lifetime and before the step is bound, so no graph is ever captured
        # against a code table it will not use. The rank is a pure function of
        # the training shard, shared by every genome in every generation, costs
        # zero genome reals and is never mutated; `token_provenance` carries it
        # into the run manifest so a code derived from data is reproducible
        # from that data.
        if token_code not in self.TOKEN_CODES:
            raise ValueError(f"token_code must be one of {self.TOKEN_CODES}, "
                             f"got {token_code!r}")
        self.token_code = token_code
        self.token_provenance: dict = {"token_code": token_code}
        if token_code == "freq":
            counts = token_counts(train_shard, cfg.vocab)
            self.token_provenance.update(
                counts_digest=counts_digest(counts),
                n_counted=int(counts.sum() - counts.size),
                n_unseen=int((counts == 1.0).sum()),
            )
            self.model.decoder.set_token_rank(
                C.frequency_rank(counts).to(self.model.decoder.W_row.device))
            self.model._prepare()
        for p in self.model.genome_parameters():
            p.requires_grad_(False)
        self.ev = eval_batches(eval_shard, batch=cfg.batch, T=cfg.T,
                               n_batches=n_eval_batches, device=device)

        # Static buffers: CUDA-graph replay needs stable addresses, and a Python
        # float for `eps` would specialise the graph on its value and recompile
        # at every step.
        self._tok = torch.zeros(cfg.batch, cfg.T + 1, dtype=torch.long,
                                device=device)
        self._eps = torch.zeros((), device=device)
        self._W_rand: dict = {}

        self.compile_mode = compile_mode
        self._raw_step = self._build_step()
        self._raw_eval = self._build_eval()
        self._backend = "?"
        self._graphs: dict = {}
        self._bind()

    # ------------------------------------------------------------------ graphs

    def _build_step(self):
        m, cfg = self.model, self.cfg

        def step():
            W = m.decode()
            Wa = {k: self._eps * self._W_rand[k] + (1.0 - self._eps) * v
                  for k, v in W.items()}
            loss = m.loss(self._tok, weights=Wa)
            coef = m.plasticity(W)
            return loss, coef["z"], coef["zeta"]

        return step

    def _build_eval(self):
        m = self.model

        def ev(tokens):
            with torch.no_grad():
                return m.loss(tokens, weights=m.decode())

        return ev

    # ------------------------------------------------------------------- state

    def _bind(self) -> None:
        """(Re)compile for the architecture now in force, caching per shape.

        Keyed on the architecture so a repeated one reuses its graph and a new
        one pays a single compile. `torch.compile` would cache internally, but
        the ladder in `_compile` has to be re-run per architecture because which
        rung succeeds depends on the graph.
        """
        key = (self.model.arch, self.cfg.batch, self.cfg.T)
        hit = self._graphs.get(key)
        if hit is None:
            if self.compile_mode == "cudagraph":
                hit = (self._capture(), self._raw_eval, "cudagraph")
            else:
                step, b1 = _compile(self._raw_step, self.compile_mode, "step")
                ev, b2 = _compile(lambda: self._raw_eval(self.ev[0]),
                                  self.compile_mode, "eval")
                hit = (step, self._raw_eval if b2 == "eager" else
                       torch.compile(self._raw_eval, mode=b2), b1)
            self._graphs[key] = hit
        self._step, self._eval, self._backend = hit

    def _capture(self):
        """Capture the step as a CUDA graph. SPEC: 05 §5.1, SPEC: 07 §7.1.

        The third rung, and the one `graph.py` was written for and never
        connected to: it existed as 110 unimported lines for the whole of v1.0
        while `evolve()` ran eager at 21.7 s per lifetime, which is the sort of
        hollow shell this project's own standards forbid.

        Why it is here rather than `torch.compile`. The step is CPU-dispatch
        bound, so what pays is issuing it as one launch instead of ~4,000.
        `torch.compile(mode="reduce-overhead")` does that too, but pays a ~30 s
        inductor compile **per architecture** and every genome decodes its own
        (§07), and on a machine without MSVC some of those compiles fail
        outright. Capture costs about a second, so paying it per architecture is
        affordable -- which is what makes it usable inside the search rather
        than only in a fixed-architecture run.

        `W_rand` must already be allocated, because capture bakes in the
        addresses it reads: `_ensure_w_rand` owns that, and `run` copies values
        into those tensors rather than rebinding them.
        """
        self._ensure_w_rand()
        # SPEC: 04 §4.1 -- `F.embedding`'s backward sorts its indices and calls
        # `unique_by_key`, which synchronises, and a synchronising kernel cannot
        # be captured (`cudaErrorStreamCaptureUnsupported`). The atomic-add form
        # can be, at the cost of a summation order that is not fixed. That trade
        # is what this backend *is*, so it is made here and nowhere else.
        self.model.capturable = True
        # SPEC: 03 §3.7 -- checkpointing recomputes inside backward, which a
        # captured graph cannot express. Capture trades memory for launches;
        # checkpointing trades compute for memory. They are not composable, so
        # the backend picks one and says which.
        self.model.decoder.checkpoint_decode = False
        try:
            return capture_step(self._raw_step, self.model.soma_parameters())
        except Exception:
            self.model.capturable = False
            self.model.decoder.checkpoint_decode = True
            raise

    def _ensure_w_rand(self) -> None:
        """Allocate `W_rand` once per architecture. SPEC: 05 §5.5.

        A captured graph reads these tensors at fixed addresses, so a lifetime
        must **write into** them and never replace them; `random_weights`
        allocates fresh ones, which under capture would leave the graph reading
        freed memory. Allocating here and copying in `run` makes the eager and
        captured paths do exactly the same thing, which is what lets one test
        assert they agree.
        """
        if self._W_rand:
            return
        self._W_rand = {k: torch.zeros_like(v)
                        for k, v in self.model.decode().items()}

    def set_token_code(self, rank, *, label: Optional[str] = None) -> None:
        """Switch the token code after construction. SPEC: 02 §2.9.

        `rank = None` restores §2.4's seeded bijection, which exists only as the
        control that measured §2.9's +0.0999 nats. Changing the code changes
        every decoded embedding row, so the compiled graph is discarded and
        rebound: replaying a graph captured against the old code table would
        keep decoding the old embedding, which is the class of staleness
        §05.1's line 5 exists to forbid.
        """
        self.model.decoder.set_token_rank(
            None if rank is None else rank.to(self.model.decoder.W_row.device))
        self.model._prepare()
        self.token_code = label or ("bijection" if rank is None else "freq")
        self.token_provenance = dict(self.token_provenance,
                                     token_code=self.token_code)
        self._W_rand = {}
        self._graphs.clear()
        self._bind()

    @property
    def backend(self) -> str:
        """Which rung of `_compile`'s ladder this architecture ended on."""
        return self._backend

    def set_arch(self, arch: Arch) -> None:
        """Point the runner at a different decoded architecture.

        SPEC: 07 §7.1. Every genome in an evolutionary run decodes its own
        graph, so the compiled callables must survive a shape change. They do:
        `torch.compile` keys its cache on the traced shapes, so a repeated
        architecture reuses its graph and a new one pays one compile. The
        dynamo cache limit is raised accordingly -- at the default of 8 a run
        over dozens of architectures would thrash and silently fall back to
        eager, which is the failure mode this exists to avoid.
        """
        if arch == self.model.arch:
            return
        self.model.set_arch(arch)
        self._W_rand = {}
        self._bind()

    def load_genome(self, genome: Decoder | dict) -> None:
        """Copy a genome's values in place, preserving the compiled graph."""
        src = genome.state_dict() if isinstance(genome, Decoder) else genome
        with torch.no_grad():
            for name, p in self.model.decoder.named_parameters():
                p.copy_(src[name].to(p.device, p.dtype))

    def reset_soma(self, seed: int) -> None:
        """SPEC: 04 §4.5 -- reinitialised each lifetime from the lifetime seed."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for name, p in self.model.soma.named_parameters():
                p.copy_(torch.randn(*p.shape, generator=g).to(p.device, p.dtype))

    # -------------------------------------------------------------- evaluation

    def evaluate(self) -> float:
        """Mean held-out CE in nats over the fixed evaluation batches."""
        return sum(float(self._eval(b)) for b in self.ev) / len(self.ev)

    # ---------------------------------------------------------------- lifetime

    #: SPEC: 08 §8.5. The Lamarckian arm's step size for `Gamma`, as a multiple
    #: of the soma's own schedule. Untuned, and said so: the arm is reported as
    #: an upper bound, and an upper bound tuned against the arms it bounds would
    #: not be one. It is below unity because every one of `Gamma`'s 5,279 reals
    #: touches every decoded weight, so an equal step moves the phenotype
    #: further by orders of magnitude.
    LAMARCK_LR_RATIO = 0.1

    def run(self, genome: Decoder | dict, *, seed: int, anneal: bool = True,
            rule: str = "dpr", coeff_override: Optional[tuple] = None,
            lr_field: Optional[dict] = None,
            eval_every: Optional[int] = None, weight_decay: float = 0.0,
            max_identity_rms: Optional[float] = None,
            probe: bool = True, lamarck: bool = False,
            lamarck_lr_ratio: Optional[float] = None, load: bool = True,
            checkpoint: Optional[str] = None, checkpoint_every: int = 0,
            resume: bool = False, progress: bool = False) -> LifetimeResult:
        """SPEC: 05 §5.1, SPEC: 10 §10.2. Returns held-out CE at birth and at `N_life`.

        `rule` selects one of §10.6's arms:

        ``adam``        coefficients pinned to `ADAM_POINT`; plain AdamW.
        ``eta``         only `B` decoded: AdamW times a per-identity gain. This
                        is v0.9's design, kept as the ornament it was.
        ``dpr``         all four coefficients decoded, recomputed every step.
        ``dpr-frozen``  all four decoded **once at birth** and held, isolating
                        whether the coefficient field must track a moving soma.

        `lamarck` opens §08.5's `A-LAMARCK`: the lifetime loss back-propagates
        **through the decode into `Gamma`**. This is the PNAS paper's own inner
        step, which the paper itself calls *a form of Lamarckian evolution, and
        as such biologically unrealistic*. It is expected to be far stronger per
        generation than the Darwinian arms and is reported as an **upper
        bound**, never as GNOMON's result; `LifetimeResult.lamarck` labels every
        artifact it produces, as §01.7 requires.

        `load=False` continues from the genome already in the runner rather than
        reloading one. It exists for `A-LAMARCK`, where consecutive seeds are
        consecutive *lives of one lineage*: an acquired change discarded between
        seeds would not be inherited, and inheritance of acquired change is the
        whole content of the arm.

        `checkpoint` with `checkpoint_every > 0` writes the lifetime's whole
        mutable state every `n` steps, and `resume=True` continues from it.
        SPEC: 05 §5.7 L8 -- "checkpoint at `N_life/2`, resume, and match the
        uninterrupted run bit-for-bit". Until v1.1 only the *stream* was
        resumable and the lifetime was not, so §09.1's 8-hour M1 run had no way
        to survive an interruption: everything from a crash at hour seven was
        lost. What is saved is exactly what makes a step: `Z`, both optimizer
        states including the step count and the neuromodulator's baseline, the
        stream position, and the curves. `Gamma` is not, because in the
        Darwinian arms it does not change -- except under `lamarck`, where it
        does, and is.
        """
        if rule not in ("adam", "eta", "dpr", "dpr-frozen"):
            raise ValueError(f"unknown lifetime rule {rule!r}")
        t0 = time.time()
        cfg, m = self.cfg, self.model
        if load:
            self.load_genome(genome)
        self.reset_soma(seed)
        before = [p.detach().clone() for p in m.genome_parameters()]

        # SPEC: 05 §5.1 line 9. The gradient path into `Gamma` is opened here
        # and nowhere else, and closed again before this method returns, so no
        # later Darwinian lifetime can inherit an open one.
        # SPEC: 08 §8.5 -- the Lamarckian arm runs eager. Capture would have to
        # include `Gamma`'s gradients, and the arm is an upper bound run a
        # handful of times rather than the inner loop of a search, so the
        # throughput it would buy is not worth a second captured path whose
        # equivalence would have to be established separately.
        captured = self._backend == "cudagraph" and not lamarck
        if self._backend == "cudagraph" and lamarck:
            raise ValueError(
                "A-LAMARCK cannot run under compile_mode='cudagraph': the "
                "capture does not include dL/dGamma. Use compile_mode=None "
                "(SPEC: 08 §8.5)"
            )
        opt_g = None
        if lamarck:
            for p in m.genome_parameters():
                p.requires_grad_(True)
                p.grad = None
            ratio = (self.LAMARCK_LR_RATIO if lamarck_lr_ratio is None
                     else lamarck_lr_ratio)
            opt_g = torch.optim.AdamW(m.genome_parameters(), lr=cfg.lr * ratio,
                                      betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=0.0)
            self._lamarck_ratio = ratio

        # SPEC: 05 §5.5. Written into, never rebound: a captured graph holds
        # these addresses, and `anneal=False` is `W_rand = 0` rather than an
        # absent tensor so that the graph's shapes do not depend on the arm.
        self._ensure_w_rand()
        if anneal:
            fresh = m.random_weights(seed=seed ^ 0x5A5A)
            with torch.no_grad():
                for k, v in self._W_rand.items():
                    v.copy_(fresh[k])
        else:
            with torch.no_grad():
                for v in self._W_rand.values():
                    v.zero_()
        self._eps.fill_(0.0)

        stream = TokenStream(self.train_shard, batch=cfg.batch, T=cfg.T,
                             seed=seed ^ 0xA5A5)
        params = {"z": m.soma.z, "zeta": m.soma.zeta}
        opt = DecodedRule(params, weight_decay=weight_decay,
                          max_identity_rms=max_identity_rms)
        # `y` is tiny (L x d_y) and has no identity axis to decode a rule for,
        # so it keeps a plain AdamW; §10.2's rule is about the identity tables.
        opt_y = torch.optim.AdamW([m.soma.y], lr=cfg.lr, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.0)
        every = eval_every or max(1, cfg.N_life // 10)
        # SPEC: 04 §4.6, second paragraph -- `G-NOFREE` asserts that the
        # optimizers' parameter groups cover `Z` and, in the Darwinian arms,
        # nothing else. The model holding exactly `Gamma` and `Z` does not
        # establish that: a lifetime is defined by what *moves*. Recorded here,
        # where the optimizers are built, so the gate reads the real groups
        # rather than a restatement of them.
        self.last_optimized = (
            list(params.values())
            + [q for g in opt_y.param_groups for q in g["params"]]
            + ([q for g in opt_g.param_groups for q in g["params"]]
               if opt_g is not None else [])
        )

        birth = self.evaluate()
        curve, train_curve = [(0, birth)], []
        # SPEC: 03 §3.8 D5 -- the decode's health, sampled during the lifetime
        # rather than only at birth. A run that slides back to the unigram
        # entropy over 34,000 steps did so with every existing gate green,
        # because every existing gate looks at initialisation.
        health = [dict(step=0, **m.health())] if probe else []
        start = 0
        if resume and checkpoint and pathlib.Path(checkpoint).exists():
            start, birth = self._load_checkpoint(checkpoint, opt, opt_y, stream,
                                                 curve, train_curve, opt_g)
            print(f"    resumed from {checkpoint} at step {start}", flush=True)
        # SPEC: 10 §10.6 -- `coeff_override` pins (A,B,C,D) globally, which is
        # how the rule *space* is probed: if no fixed point in it beats Adam,
        # then a decoded rule has nothing to find and the mechanism is honest
        # but pointless. That question is prior to whether evolution can find
        # anything, and far cheaper to answer.
        if coeff_override is not None:
            pinned = self._pinned_coeffs(params, coeff_override)
            rule = "adam"
        else:
            pinned = self._pinned_coeffs(params) if rule == "adam" else None
        if lr_field is not None:
            # SPEC: 10 §10.7 -- a hand-designed per-identity field, to measure
            # the ceiling of the one hypothesis §10.5b's control leaves standing.
            pinned = dict(pinned if pinned is not None
                          else self._pinned_coeffs(params))
            for k, v in lr_field.items():
                pinned[k] = v.to(pinned[k].device, pinned[k].dtype)
            rule = "adam"
        frozen = None

        for n in range(start, cfg.N_life):
            lr = wsd_lr(n, cfg)
            for grp in opt_y.param_groups:
                grp["lr"] = lr
            self._tok.copy_(stream.next(self.device))
            self._eps.fill_(eps_at(n, cfg) if anneal else 0.0)

            if captured:
                # A captured step replays its own backward into the *same*
                # `.grad` allocations every time, so they are zeroed before the
                # replay and never set to `None`: `set_to_none` would detach the
                # tensors the graph writes and the optimizer would then read
                # stale zeros for the rest of the lifetime.
                for p in m.soma_parameters():
                    p.grad.zero_()
                loss, cz, czeta = self._step()
            else:
                loss, cz, czeta = self._step()
                for p in m.soma_parameters():
                    if p.grad is not None:
                        p.grad = None
                if opt_g is not None:
                    for p in m.genome_parameters():
                        p.grad = None
                loss.backward()
            torch.nn.utils.clip_grad_norm_(m.soma_parameters(), 1.0)
            if opt_g is not None:
                # The same clip as the soma's, so the arm differs from the
                # Darwinian ones in *where* the gradient goes and not in how it
                # is conditioned; and the same warmup-stable-decay shape, so it
                # differs in scale rather than in schedule (SPEC: 05 §5.4).
                torch.nn.utils.clip_grad_norm_(m.genome_parameters(), 1.0)
                for grp in opt_g.param_groups:
                    grp["lr"] = lr * self._lamarck_ratio

            coef = {"z": cz, "zeta": czeta}
            if rule == "adam":
                coef = pinned
            elif rule == "eta":
                coef = self._eta_only(coef, pinned or self._pinned_coeffs(params))
            elif rule == "dpr-frozen":
                if frozen is None:
                    frozen = {k: v.clone() for k, v in coef.items()}
                coef = frozen
            opt.step(coef, float(loss), lr)
            opt_y.step()
            if opt_g is not None:
                # SPEC: 08 §8.5 -- the germline step. Taken *after* the soma's,
                # so within a step the two timescales see the same `W`: an
                # update to `Gamma` between the soma's gradient and the soma's
                # step would make the soma's displacement one computed against a
                # decode that no longer exists.
                opt_g.step()

            train_curve.append((n, float(loss)))
            if (n + 1) % every == 0 or n == cfg.N_life - 1:
                curve.append((n + 1, self.evaluate()))
                hz = dict(step=n + 1, **m.health()) if probe else {}
                if probe:
                    health.append(hz)
                if progress:
                    extra = (f"  |z| {hz['z_rms']:8.3g}  |zeta| {hz['zeta_rms']:8.3g}"
                             f"  resp {hz['responsive']:.3f}  erank {hz['erank']:.4f}"
                             if probe else "")
                    print(f"    step {n+1:6d}  train {float(loss):7.4f}  "
                          f"held-out {curve[-1][1]:7.4f}{extra}", flush=True)
            if (checkpoint and checkpoint_every
                    and (n + 1) % checkpoint_every == 0):
                self._save_checkpoint(checkpoint, n + 1, birth, opt, opt_y,
                                      stream, curve, train_curve, opt_g)

        if opt_g is None:
            for p, b in zip(m.genome_parameters(), before):
                assert torch.equal(p.detach(), b), (
                    "the genome was written during a lifetime (§05.7 L3)"
                )
        else:
            # The arm is only the arm if `Gamma` actually moved. A silently
            # severed path would turn `A-LAMARCK` into a relabelled Darwinian
            # run whose "upper bound" bounds nothing -- which is exactly the
            # class of defect §00's RC1 describes.
            moved = any(not torch.equal(p.detach(), b)
                        for p, b in zip(m.genome_parameters(), before))
            assert moved, (
                "A-LAMARCK ran and `Gamma` is unchanged: the gradient path "
                "into the decode is severed (SPEC: 08 §8.5)"
            )
            for p in m.genome_parameters():
                p.requires_grad_(False)
                p.grad = None
        self.last_rule_report = opt.contributions(
            {"z": cz.detach(), "zeta": czeta.detach()})
        return LifetimeResult(
            final=curve[-1][1], birth=birth, curve=curve,
            train_curve=train_curve, seconds=time.time() - t0,
            n_genome=m.n_genome(), n_soma=m.n_soma(), n_decoded=m.n_decoded(),
            lamarck=lamarck, health=health,
        )

    def _save_checkpoint(self, path, n, birth, opt, opt_y, stream, curve,
                         train_curve, opt_g) -> None:
        """SPEC: 05 §5.7 L8. Written atomically: a crash mid-write must not
        leave a checkpoint that loads and is wrong, which is worse than none."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "n": int(n), "birth": float(birth),
            "soma": {k: v.detach().cpu().clone()
                     for k, v in self.model.soma.state_dict().items()},
            "rule": opt.state_dict(),
            "opt_y": opt_y.state_dict(),
            "stream": stream.state(),
            "curve": list(curve), "train_curve": list(train_curve),
            "arch": self.model.arch.summary(),
            "token_code": self.token_code,
        }
        if opt_g is not None:
            blob["genome"] = self.export_genome()
            blob["opt_g"] = opt_g.state_dict()
        tmp = p.with_suffix(p.suffix + ".partial")
        torch.save(blob, tmp)
        tmp.replace(p)

    def _load_checkpoint(self, path, opt, opt_y, stream, curve, train_curve,
                         opt_g) -> tuple[int, float]:
        """SPEC: 05 §5.7 L8. Restore every mutable thing a step reads."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("arch") != self.model.arch.summary():
            raise ValueError(
                f"checkpoint was written for architecture "
                f"[{blob.get('arch')}] and this runner holds "
                f"[{self.model.arch.summary()}]"
            )
        if blob.get("token_code") != self.token_code:
            raise ValueError(
                f"checkpoint used token code {blob.get('token_code')!r} and "
                f"this runner uses {self.token_code!r} (SPEC: 02 §2.9)"
            )
        with torch.no_grad():
            for k, v in self.model.soma.state_dict().items():
                v.copy_(blob["soma"][k].to(v.device, v.dtype))
        opt.load_state_dict(blob["rule"])
        opt_y.load_state_dict(blob["opt_y"])
        stream.seek(blob["stream"]["step"])
        curve[:] = [tuple(x) for x in blob["curve"]]
        train_curve[:] = [tuple(x) for x in blob["train_curve"]]
        if opt_g is not None and "opt_g" in blob:
            self.load_genome(blob["genome"])
            opt_g.load_state_dict(blob["opt_g"])
        return int(blob["n"]), float(blob["birth"])

    def export_genome(self) -> dict:
        """`Gamma_w` as it now stands, detached. SPEC: 08 §8.5.

        The Darwinian arms never need this: the genome they were given is the
        genome they end with. `A-LAMARCK` writes to `Gamma` during the lifetime,
        and an acquired change that is not read back out is not inherited.
        """
        return {n: p.detach().clone().cpu()
                for n, p in self.model.decoder.named_parameters()}

    @staticmethod
    def _pinned_coeffs(params: dict, point=ADAM_POINT) -> dict:
        """One `(A,B,C,D)` for every identity. SPEC: 10 §10.6, the `R-ADAM` arm."""
        out = {}
        for k, p in params.items():
            c = torch.zeros(p.shape[0], N_RULE, device=p.device, dtype=p.dtype)
            for j, v in enumerate(point):
                c[:, j] = v
            out[k] = c
        return out

    @staticmethod
    def _eta_only(coef: dict, pinned: dict) -> dict:
        """Keep the decoded `B`, pin the rest. SPEC: 10 §10.6, the `R-ETA` arm.

        This is exactly v0.9's `eta`: AdamW's displacement multiplied by a
        per-identity decoded gain, and nothing else.
        """
        out = {}
        for k in coef:
            c = pinned[k].clone()
            c[:, 1] = coef[k][:, 1]
            out[k] = c
        return out
