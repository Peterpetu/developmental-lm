"""One lifetime: decode a phenotype, train its soma, measure it.

SPEC: 05 §5.1
SPEC: 05 §5.2
SPEC: 05 §5.3
SPEC: 05 §5.4
SPEC: 05 §5.5

```
LIFETIME(Gamma, seed, cfg):
  1  arch   <- DECODE_ARCH(Gamma_a, cfg)
  2  Z      <- INIT_SOMA(seed, cfg)
  3  W_rand <- random init of every role at s_target
  4  for n = 1 .. N_life:
  5      W    <- DECODE(Gamma, Z, arch)            # inside the loop; Z moves
  6      W_n  <- eps(n) W_rand + (1 - eps(n)) W
  8      L    <- cross_entropy(FORWARD(W_n, batch))
  9      grad <- dL/dZ
 10      Z    <- STEP(Z, grad, eta, n)
```
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Optional

import torch

from ..config import GenomeShape, PhenoConfig
from ..data.stream import Shard, TokenStream, eval_batches
from ..genome.decoder import Decoder
from ..pheno.model import Gnomon
from ..pheno.plan import Arch

__all__ = ["LifetimeResult", "lifetime", "evaluate", "eps_at", "wsd_lr"]


def eps_at(n: int, cfg: PhenoConfig) -> float:
    """SPEC: 05 §5.5. `eps(n) = exp(-n/lambda)`, `lambda = 0.04 N_life`.

    The paper reports the annealing as a necessity rather than an optimisation:
    p-networks decoded from random g-networks are *"impossible to train"*
    because initialisation errors diverge exponentially with depth. Generation 0
    decodes from random genomes and is exactly that regime.
    """
    return math.exp(-n / cfg.lam)


def wsd_lr(n: int, cfg: PhenoConfig, *, warm: float = 0.05, decay: float = 0.2) -> float:
    """Warmup-stable-decay. SPEC: 05 §5.4."""
    N = cfg.N_life
    n_w, n_d = max(1, int(warm * N)), max(1, int(decay * N))
    if n < n_w:
        return cfg.lr * (n + 1) / n_w
    if n >= N - n_d:
        return cfg.lr * max(0.0, (N - n) / n_d)
    return cfg.lr


@dataclasses.dataclass
class LifetimeResult:
    """Everything one lifetime produces. SPEC: 05 §5.3."""

    final: float                       # held-out CE at n = N_life  (the fitness)
    birth: float                       # held-out CE at n = 0, eps = 0 (§06.4)
    curve: list[tuple[int, float]]     # (step, held-out CE)
    train_curve: list[tuple[int, float]]
    seconds: float
    n_genome: int
    n_soma: int
    n_decoded: int
    #: SPEC: 03 §3.8 D5 -- the decode's health at each evaluation: identity
    #: norms, decoder saturation and decoded effective rank. Empty when the
    #: probe is off.
    health: list = dataclasses.field(default_factory=list)
    #: SPEC: 08 §8.5, SPEC: 01 §1.7. True if the lifetime loss reached `Gamma`.
    #: §01.7 requires **every** artifact of the Lamarckian arm to be labelled,
    #: and a label carried on the result is one no reporting path can forget.
    lamarck: bool = False

    @property
    def slope(self) -> float:
        """Late-phase slope of held-out CE against `log10` step. Reported, not selected on."""
        pts = [(s, v) for s, v in self.curve if s >= self.curve[-1][0] // 2 and s > 0]
        if len(pts) < 2:
            return float("nan")
        xs = [math.log10(s) for s, _ in pts]
        ys = [v for _, v in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        return float("nan") if den == 0 else sum(
            (x - mx) * (y - my) for x, y in zip(xs, ys)) / den


@torch.no_grad()
def evaluate(model: Gnomon, batches: list[torch.Tensor]) -> float:
    """Mean held-out cross-entropy in nats, over fixed batches. SPEC: 05 §5.2."""
    was = model.training
    model.eval()
    W = model.decode()
    total = 0.0
    for b in batches:
        total += float(model.loss(b, weights=W))
    model.train(was)
    return total / len(batches)


def lifetime(
    decoder: Decoder,
    cfg: PhenoConfig,
    shape: GenomeShape,
    *,
    seed: int,
    train_shard: Shard,
    eval_shard: Shard,
    arch: Optional[Arch] = None,
    device: str = "cpu",
    n_eval_batches: int = 8,
    eval_every: Optional[int] = None,
    anneal: bool = True,
    rule: str = "dpr",
    token_code: str = "freq",
    compile_mode: Optional[str] = None,
    progress: bool = False,
) -> LifetimeResult:
    """One lifetime. SPEC: 05 §5.1.

    **[NORMATIVE — there is exactly one implementation of §05.1, and this is a
    call into it.]** Until v1.1 there were two: this function, and
    `LifetimeRunner.run`. Every acceptance test L1..L9 exercised *this* one and
    every run that produced a number used the *other*, and they had drifted --
    this one silently ran plain `dpr` when asked for `dpr-frozen`, had no
    `coeff_override` and so could not express §10.6's rule-space probe, and
    after §02.9 became the default it would have kept decoding §2.4's token
    code. A suite that certifies a path the science does not take is the exact
    shape of §00.3's defect, one level down, so the duplicate is gone and this
    is a thin wrapper: the tests now bind the code that runs.

    A `LifetimeRunner` is constructed per call, which is the wrong economics for
    a search -- §08 builds one runner and pushes thousands of genomes through it
    -- and the right ones for a single lifetime, a smoke run or a test.
    """
    from .runner import LifetimeRunner

    runner = LifetimeRunner(cfg, shape, train_shard=train_shard,
                            eval_shard=eval_shard, device=device, arch=arch,
                            compile_mode=compile_mode,
                            n_eval_batches=n_eval_batches,
                            token_code=token_code)
    return runner.run(decoder, seed=seed, anneal=anneal, rule=rule,
                      eval_every=eval_every, progress=progress)
