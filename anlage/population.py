"""The population: mutation, selection over strata, crossover by merging.

SPEC: 07 §7.2
SPEC: 07 §7.3
SPEC: 07 §7.4
SPEC: 07 §7.5

```
    matrices:  t' = t + sigma_mut · rms(t) · (A B^T) / sqrt(r)          r = 4
    vectors:   t' = t + sigma_mut · max(rms(t), 0.1) · N(0, I)
    W_x = W_0 + sum_p c_{p,l,m} (W_p - W_0)                              (7.1)
    accept iff CE_fit(W_x) <= min_k CE_fit(W_k) + sigma_e               (7.2)
```
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from soma.selection.lexicase import epsilon_lexicase
from .data import EvalSet
from .genome import Genome
from .lifetime import evaluate
from .model import Phenotype, key_name

__all__ = ["mutate", "select_parents", "merge", "MergeReport", "MUT_RANK", "MUT_FLOOR"]

MUT_RANK = 4
MUT_FLOOR = 0.1


@torch.no_grad()
def mutate(G: Genome, sigma_mut: float, generator: torch.Generator) -> Genome:
    """`MUTATE` of §07.3: a low-rank, scale-relative perturbation of the kernel.

    Every trailing matrix of a tensor (each `O_m^{(h)}` separately) gets its own
    rank-`MUT_RANK` perturbation; vectors get Gaussian noise at the floor.
    """
    M = copy.deepcopy(G)
    for _, p in M.named_parameters():
        t = p.data
        if t.dim() >= 2:
            flat = t.reshape(-1, t.shape[-2], t.shape[-1])
            for i in range(flat.shape[0]):
                m, n = flat[i].shape
                A = torch.randn(m, MUT_RANK, generator=generator).to(t.device)
                B = torch.randn(n, MUT_RANK, generator=generator).to(t.device)
                rms = float(flat[i].pow(2).mean().sqrt())
                flat[i] += sigma_mut * rms * (A @ B.T) / math.sqrt(MUT_RANK)
        else:
            rms = max(float(t.pow(2).mean().sqrt()), MUT_FLOOR)
            t += sigma_mut * rms * torch.randn(t.shape, generator=generator).to(t.device)
    return M


def select_parents(scores: Sequence[Sequence[float]], eps: Sequence[float], n_par: int,
                   rng: np.random.Generator) -> list[int]:
    """ε-lexicase over strata. SPEC: 07 §7.4.

    `scores[k][c]` is sibling `k`'s CE on stratum `c` (lower is better);
    `eps[c] = sigma_e(c)`. Strata with no tokens (NaN) are dropped.
    """
    K = len(scores)
    S = np.asarray(scores, dtype=np.float64)                  # (K, C)
    ok = np.all(np.isfinite(S), axis=0)
    S = S[:, ok]
    e = np.asarray(eps, dtype=np.float64)[ok]
    e = np.where(np.isfinite(e), e, 0.0)
    # The vendored operator maximises a non-negative credit over cases; credit
    # is CE reflected and shifted so that every entry is positive.
    nu = torch.tensor((S.max() + 1.0 - S).T)                 # (C, K), > 0
    res = epsilon_lexicase(list(range(K)), nu, torch.tensor(e), torch.ones(nu.shape[0]),
                           n_par, rng)
    return res.parents


class _LossModule(nn.Module):
    """A module whose forward is the phenotype's loss, for `functional_call`."""

    def __init__(self, model: Phenotype):
        super().__init__()
        self.model = model

    def forward(self, window):
        ce, zl = self.model.loss(window)
        return ce


@dataclasses.dataclass
class MergeReport:
    coefficients: dict[str, list[float]]
    ce_fit_merge: float
    ce_fit_parents: list[float]
    ce_fit_best: float
    accepted: bool
    chosen: str                                        # "merge" or "sibling k"
    fit_curve: list[float]

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def merge(parents: list[Phenotype], W_0: dict, fit_windows: torch.Tensor,
          score_set: EvalSet, sigma_e: float, *, steps: int = 200, lr: float = 0.05,
          log: Optional[Callable[[str], None]] = None) -> tuple[Phenotype, MergeReport]:
    """(7.1)-(7.2). SPEC: 07 §7.4.

    `fit_windows (n, T+1)` are the coefficient-fit windows (disjoint from
    `score_set`, S8); `score_set` scores the merge and the parents for (7.2).
    Returns the merge if accepted, else the best parent, with the report.
    """
    P = len(parents)
    cfg, vocab, tied = parents[0].cfg, parents[0].vocab, parents[0].tied
    keys = cfg.keys(vocab, tied=tied)
    device = next(parents[0].parameters()).device
    base = {k: W_0[k].detach().to(device).float() for k in keys}
    deltas = [{k: p.get(*k).detach().float() - base[k] for k in keys} for p in parents]
    c = torch.full((P, len(keys)), 1.0 / P, device=device, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    child = Phenotype(cfg, vocab, tied=tied, weights=base).to(device)
    child.requires_grad_(False)
    wrapper = _LossModule(child)

    def build(cc):
        return {"model.W." + key_name(*k): base[k] + sum(cc[p, i] * deltas[p][k] for p in range(P))
                for i, k in enumerate(keys)}

    curve = []
    for n in range(steps):
        opt.zero_grad(set_to_none=True)
        params = build(c)
        tot = torch.zeros((), device=device)
        for i in range(0, fit_windows.shape[0], 16):
            w = fit_windows[i:i + 16]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=w.is_cuda):
                ce = functional_call(wrapper, params, (w,))
            tot = tot + ce * w.shape[0]
        loss = tot / fit_windows.shape[0]
        loss.backward()
        opt.step()
        curve.append(float(loss))
        if log and (n + 1) % 50 == 0:
            log(f"    merge fit step {n + 1}: CE {float(loss):.4f}")
    with torch.no_grad():
        final = build(c.detach())
        for name, t in final.items():
            child.W[name[len("model.W."):]].copy_(t)
    ce_merge, _ = evaluate(child, score_set)
    ce_par = [evaluate(p, score_set)[0] for p in parents]
    best = int(np.argmin(ce_par))
    accepted = ce_merge <= ce_par[best] + sigma_e
    coeff = {f"{k[0]}_{k[1]}": c[:, i].detach().tolist() for i, k in enumerate(keys)}
    rep = MergeReport(coefficients=coeff, ce_fit_merge=ce_merge, ce_fit_parents=ce_par,
                      ce_fit_best=ce_par[best], accepted=accepted,
                      chosen="merge" if accepted else f"sibling {best}", fit_curve=curve)
    return (child if accepted else parents[best]), rep
