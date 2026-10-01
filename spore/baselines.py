"""Count-model baselines on the same stream and the same evaluation windows. SPEC: 02 §2.2.

Unigram (add-one), bigram and trigram with Jelinek-Mercer interpolation whose weights are fit
on the fitness set by golden-section search. Exact counts: the bigram table is dense `(V, V)`;
trigram counts are a sorted key table (`torch.unique` on int64 keys) looked up by
`searchsorted`.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

from anlage.data import EvalSet, N_STRATA
from spore.life import SequentialStream
from spore.readout import golden_section

__all__ = ["CountModel", "fit_count_models"]


class CountModel:
    def __init__(self, V: int, device: str = "cuda"):
        self.V = V
        self.device = device
        self.c1 = torch.zeros(V, dtype=torch.float64, device=device)
        self.c2 = torch.zeros(V * V, dtype=torch.float64, device=device)
        self.tri_keys: list[torch.Tensor] = []
        self.N = 0
        self.lam2, self.lam3 = 0.5, 0.5

    def add(self, w: torch.Tensor) -> None:
        """`w (B, T+1)`: counts of unigrams, bigrams and trigrams over every target position."""
        V = self.V
        tgt = w[:, 1:].reshape(-1)
        prev = w[:, :-1].reshape(-1)
        self.c1 += torch.bincount(tgt, minlength=V).double()
        self.c2 += torch.bincount(prev * V + tgt, minlength=V * V).double()
        pp = w[:, :-2].reshape(-1)
        k3 = (pp * V + w[:, 1:-1].reshape(-1)) * V + w[:, 2:].reshape(-1)
        self.tri_keys.append(k3)
        self.N += int(tgt.numel())

    def finalise(self) -> None:
        keys = torch.cat(self.tri_keys)
        self.tri_u, self.tri_c = torch.unique(keys, return_counts=True)
        self.tri_c = self.tri_c.double()
        self.tri_keys = []
        # context counts of trigrams: c(ab) over positions with a preceding token
        ab = self.tri_u // self.V
        self.ab_u, ab_idx = torch.unique(ab, return_inverse=True)
        self.ab_c = torch.zeros(self.ab_u.numel(), dtype=torch.float64, device=self.device).index_add_(0, ab_idx, self.tri_c)

    def probs(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per target position of `(B, T+1)`: `(p1, p2, p3)` where `p2` is the raw bigram
        conditional (nan where the context is unseen) and `p3` the raw trigram conditional."""
        V = self.V
        tgt = w[:, 1:].reshape(-1)
        prev = w[:, :-1].reshape(-1)
        p1 = (self.c1[tgt] + 1.0) / (self.N + V)
        c_prev = self.c1_ctx[prev]
        p2 = self.c2[prev * V + tgt] / c_prev.clamp_min(1)
        p2 = torch.where(c_prev > 0, p2, torch.full_like(p2, float("nan")))
        B, L = w.shape
        k3 = torch.full((B, L - 1), -1, dtype=torch.long, device=w.device)
        k3[:, 1:] = (w[:, :-2] * V + w[:, 1:-1]) * V + w[:, 2:]
        k3 = k3.reshape(-1)
        pos = torch.searchsorted(self.tri_u, k3).clamp_max(self.tri_u.numel() - 1)
        hit = self.tri_u[pos] == k3
        c3 = torch.where(hit, self.tri_c[pos], torch.zeros_like(self.tri_c[pos]))
        ab = k3 // V
        pos_ab = torch.searchsorted(self.ab_u, ab).clamp_max(self.ab_u.numel() - 1)
        hit_ab = (self.ab_u[pos_ab] == ab) & (k3 >= 0)
        cab = torch.where(hit_ab, self.ab_c[pos_ab], torch.zeros_like(self.ab_c[pos_ab]))
        p3 = torch.where(cab > 0, c3 / cab.clamp_min(1), torch.full_like(c3, float("nan")))
        return p1, p2, p3

    def prepare(self) -> None:
        # context counts for the bigram: c(a) over positions that have a successor = column sums of c2
        self.c1_ctx = self.c2.reshape(self.V, self.V).sum(1)

    def interpolate(self, p1, p2, p3, lam2: float, lam3: float) -> torch.Tensor:
        q2 = torch.where(torch.isnan(p2), p1, lam2 * p2 + (1 - lam2) * p1)
        q3 = torch.where(torch.isnan(p3), q2, lam3 * p3 + (1 - lam3) * q2)
        return q3

    def ce(self, ev: EvalSet, order: int, lam2: float, lam3: float) -> tuple[float, list[float]]:
        total = torch.zeros(N_STRATA, dtype=torch.float64, device=self.device)
        count = torch.zeros(N_STRATA, dtype=torch.float64, device=self.device)
        for i in range(0, ev.n_windows, 32):
            w = ev.windows[i:i + 32]
            s = ev.strata[i:i + 32].reshape(-1)
            p1, p2, p3 = self.probs(w)
            if order == 1:
                p = p1
            elif order == 2:
                p = self.interpolate(p1, p2, torch.full_like(p1, float("nan")), lam2, 0.0)
            else:
                p = self.interpolate(p1, p2, p3, lam2, lam3)
            l = -torch.log(p.clamp_min(1e-12))
            total.index_add_(0, s, l)
            count.index_add_(0, s, torch.ones_like(l))
        ce = float(total.sum() / count.sum())
        per = (total / count.clamp_min(1)).tolist()
        return ce, [x if c > 0 else float("nan") for x, c in zip(per, count.tolist())]

    def fit_lambdas(self, ev: EvalSet) -> None:
        self.lam2, _ = golden_section(lambda x: self.ce(ev, 2, x, 0.0)[0], 0.01, 0.99)
        self.lam3, _ = golden_section(lambda x: self.ce(ev, 3, self.lam2, x)[0], 0.01, 0.99)


def fit_count_models(shard, *, budget: int, offset: int, T: int, batch_windows: int, V: int,
                     ev_fit: EvalSet, ev_test: EvalSet, device: str = "cuda",
                     log: Callable[[str], None] = print) -> dict:
    t0 = time.time()
    cm = CountModel(V, device)
    stream = SequentialStream(shard, offset=offset, T=T, batch_windows=batch_windows, device=device)
    seen = 0
    while seen < budget:
        cm.add(stream.next())
        seen += batch_windows * T
    cm.finalise()
    cm.prepare()
    cm.fit_lambdas(ev_fit)
    out = {"tokens_seen": seen, "lam2": cm.lam2, "lam3": cm.lam3, "n_trigram_types": int(cm.tri_u.numel()),
           "seconds": time.time() - t0}
    for order, name in ((1, "unigram"), (2, "bigram"), (3, "trigram")):
        ce_f, _ = cm.ce(ev_fit, order, cm.lam2, cm.lam3)
        ce_t, strata = cm.ce(ev_test, order, cm.lam2, cm.lam3)
        out[name] = {"fitness_ce": ce_f, "test_ce": ce_t, "test_strata": strata}
        log(f"{name}: fitness {ce_f:.4f}  test {ce_t:.4f}")
    return out
