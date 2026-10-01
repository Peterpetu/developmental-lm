"""Sleep: consolidation *during* the lifetime.

SPEC: 05 §5.7
SPEC: 06 §6.4
SPEC: 08 §8.5

R4 measured what a single regression of an unconstrained adult gives at birth:
6.7% of the trained gain (K-BIRTH fired, `docs/anlage/RESULTS.md`). The PNAS loop
never relied on a single regression -- its "intermittent training" alternates
gradient steps on the phenotype with regression of the genome and regeneration
of the phenotype from it for hundreds of rounds, so that the phenotype is
*trained to be expressible*. This module is that loop as a lifetime hook:

  * every `every` optimiser steps the genome is refit to the current weights
    (`n_refit` warm-started Adam steps on (5.3); C4 continuity) and the decode
    `D = D(Gamma, Z)` is cached;
  * `mode = "anchor"`: on every step the weights are pulled toward the cached
    decode, `W <- W - eta(p) (W - D)`, `eta(p) = eta_anchor · lr(p)/lr_max`,
    i.e. decoupled decay toward `D` instead of toward `0` (the Gaussian-prior
    term of Proposition 1 applied in the lifetime, §05.7);
  * `mode = "regenerate"`: at each refit `W <- (1 - eps) W + eps D` at once,
    the PNAS regeneration step, as the ablation.

The decode never runs inside the training step: it runs at refits, and the
per-step pull reads a cached tensor (G-NOFREE-INVERSE holds; the sentinel test
covers the hook).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch

from .genome import DecodeContext, Genome
from .model import Phenotype
from .recipe import TrainConfig, lr_at
from .regress import RegressConfig, objective

__all__ = ["Sleeper"]


class Sleeper:
    def __init__(self, G: Genome, ctx: DecodeContext, model: Phenotype, tc: TrainConfig, *,
                 every: int, n_refit: int, eta_anchor: float, mode: str = "anchor",
                 regen_eps: float = 0.5, rc: RegressConfig = RegressConfig(),
                 eval_decode: Optional[Callable[[dict], float]] = None,
                 log: Optional[Callable[[str], None]] = None):
        if every < 1 or n_refit < 0:
            raise ValueError("every >= 1 and n_refit >= 0")
        if mode not in ("anchor", "regenerate", "none"):
            raise ValueError(f"unknown sleep mode {mode!r}")
        if not 0.0 <= eta_anchor < 1.0:
            raise ValueError("eta_anchor in [0, 1)")
        if not 0.0 <= regen_eps <= 1.0:
            raise ValueError("regen_eps in [0, 1]")
        self.G, self.ctx, self.model, self.tc = G, ctx, model, tc
        self.every, self.n_refit = int(every), int(n_refit)
        self.eta_anchor, self.mode, self.regen_eps = float(eta_anchor), mode, float(regen_eps)
        self.rc = rc
        self.eval_decode = eval_decode
        self.log = log
        self.params = list(G.parameters()) + list(ctx.bank.parameters())
        # one persistent optimiser: the refits are one long regression, interleaved
        self.opt = torch.optim.Adam(self.params, lr=rc.lr, betas=rc.betas)
        self.D: Optional[dict] = None
        self.records: list[dict] = []
        self.seconds_asleep = 0.0

    # ------------------------------------------------------------------ refit
    @torch.no_grad()
    def _residual(self, W: dict) -> float:
        if self.D is None:
            return float("nan")
        r = [(self.D[k].float() - W[k].float()).pow(2).sum() / W[k].float().pow(2).sum().clamp_min(1e-30)
             for k in W]
        return float(torch.stack(r).mean())

    def refit(self, step: int) -> dict:
        t0 = time.time()
        W = {k: p.detach() for k, p in self.model.weights().items()}
        drift = self._residual(W)                       # how far W moved from the last decode
        self.G.requires_grad_(True)
        self.ctx.bank.requires_grad_(True)
        last = float("nan")
        for _ in range(self.n_refit):
            self.opt.zero_grad(set_to_none=True)
            J, res = objective(self.G, self.ctx, W, self.rc)
            J.backward()
            self.opt.step()
            last = float(torch.stack(list(res.values())).mean())
        self.G.requires_grad_(False)
        self.ctx.bank.requires_grad_(False)
        with torch.no_grad():
            self.D = {k: v.detach() for k, v in self.G.decode_all(self.ctx).items()}
            if self.mode == "regenerate" and self.regen_eps > 0:
                for k, p in self.model.weights().items():
                    p.data.lerp_(self.D[k].to(p.dtype), self.regen_eps)
        ce_D = self.eval_decode(self.D) if self.eval_decode is not None else float("nan")
        dt = time.time() - t0
        self.seconds_asleep += dt
        rec = {"step": step, "drift": drift, "residual": last, "decode_ce": ce_D, "seconds": dt}
        self.records.append(rec)
        if self.log:
            self.log(f"  sleep at step {step}: drift {drift:.3f} -> residual {last:.3f}  "
                     f"CE(D) {ce_D:.4f}  {dt:.1f}s")
        return rec

    # ------------------------------------------------------------------- hook
    def before_step(self, step: int, tokens_done: int) -> None:
        if step > 0 and step % self.every == 0:
            self.refit(step)
        if self.D is not None and self.mode == "anchor" and self.eta_anchor > 0:
            p = tokens_done / self.tc.tokens
            eta = self.eta_anchor * lr_at(p, 1.0, self.tc)
            if eta > 0:
                with torch.no_grad():
                    for k, w in self.model.weights().items():
                        w.data.sub_(eta * (w.data - self.D[k].to(w.dtype)))
        return None

    def as_dict(self) -> dict:
        return {"every": self.every, "n_refit": self.n_refit, "eta_anchor": self.eta_anchor,
                "mode": self.mode, "regen_eps": self.regen_eps, "records": self.records,
                "seconds_asleep": self.seconds_asleep}
