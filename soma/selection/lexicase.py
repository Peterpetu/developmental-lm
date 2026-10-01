"""ε-lexicase selection over units. **This is the novel core.**

SPEC: 04 §4.4
SPEC: 04 §4.5
SPEC: 04 §4.5.1

```
nu_ut = r_ut * || delta_ut ||_2 * w_t                                    (4.2)

C_t   = { u : nu_ut > 0 }                                                (4.6)
eps_t = median_{u in C_t} | nu_ut - median_{v in C_t} nu_vt |  if C_t != {}
      = 0                                                      if C_t == {}  (4.7)
```

```
Algorithm EPSILON_LEXICASE(U_parent, cases, nu, eps, w, n_par) -> parents
  precondition: |cases| >= D_min >= 8 ; |U_parent| >= 2 ; eps_t >= 0
  parents <- empty multiset
  repeat n_par times:
      S_pool <- U_parent
      T_q    <- SHUFFLE(cases, weights = w)
      while |T_q| >= 1 and |S_pool| > 1:
          t      <- pop_front(T_q)
          elite  <- max_{u in S_pool} nu_ut          # on the CURRENT pool
          S_pool <- { u in S_pool : nu_ut >= elite - eps_t }
      parents <- parents + { UNIFORM_RANDOM(S_pool) }
  return parents
```

**Two defects this implementation exists to avoid.**

*Spec v1's `eps_t` was identically zero.* It computed the MAD over the **whole
population**; since only `k_route ~ 8` of `~1e2`-`1e3` units are routed to any
token, `nu_ut = 0` for the overwhelming majority, so `median_v nu_vt = 0` and
`MAD_t = 0` identically. Every `eps_t` would have been zero, the filter would
have degenerated to strict `nu_ut >= elite`, the pool would collapse to one unit
after the first case, and Proposition 4.3 would not hold. ε-lexicase — the
architecture's central novel mechanism — would have been silently inert.

*Spec v2's `eps_t` was NaN.* Taking the MAD over `C_t` fixed the first defect
but `C_t` is **empty** whenever `w_t = 0`, which (4.1) makes common: any case on
which the model has not improved since the lag. A median over the empty set is
undefined (audit A-20). (4.7) defines `eps_t = 0` there, which reduces the
filter to strict lexicase on a case carrying no learnability signal — the
correct behaviour, since such a case should not widen the pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "credit_matrix",
    "epsilon_per_case",
    "epsilon_lexicase",
    "mean_utility_parents",
    "case_win_histogram",
    "LexicaseResult",
]


def credit_matrix(
    slots: torch.Tensor,
    delta_norm: torch.Tensor,
    r_ut: torch.Tensor,
    w_t: torch.Tensor,
    U_max: int,
) -> torch.Tensor:
    """(4.2) as a dense `(q, U_max)` matrix.

    SPEC: 04 §4.4

    `nu_ut = 0` for every unit not routed to `t` — which is why `eps_t` must be
    computed over non-zero credits only (§04.5).

    Dense rather than sparse because `q_cases * U_max` is `1638 x 256 = 419k`
    floats at M-nano and `6554 x 1024 = 6.7M` at M1 (27 MB), while the lexicase
    inner loop indexes it by (case, pool) repeatedly — a sparse layout would
    make the hot loop a gather.
    """
    q, k = slots.shape
    nu = torch.zeros(q, U_max, device=slots.device, dtype=torch.float32)
    valid = slots >= 0
    if not bool(valid.any()):
        return nu
    rows = torch.arange(q, device=slots.device).unsqueeze(-1).expand_as(slots)
    contrib = (r_ut * delta_norm).float() * w_t.unsqueeze(-1).float()
    nu.index_put_(
        (rows[valid], slots[valid]), contrib[valid], accumulate=True
    )
    return nu


def epsilon_per_case(nu: torch.Tensor) -> torch.Tensor:
    """(4.6)-(4.7): per-case MAD over units with non-zero credit.

    SPEC: 04 §4.5

    `nu` is `(q, U_max)`; returns `(q,)`.
    """
    q = nu.shape[0]
    eps = torch.zeros(q, device=nu.device, dtype=nu.dtype)
    positive = nu > 0
    counts = positive.sum(dim=-1)
    for t in torch.nonzero(counts > 0, as_tuple=False).squeeze(-1).tolist():
        vals = nu[t][positive[t]]
        med = vals.median()
        eps[t] = (vals - med).abs().median()
    return eps


@dataclass
class LexicaseResult:
    parents: List[int]
    """A **multiset**: a unit selected twice appears twice."""
    pool_trace: List[int]
    """Pool size after the first case of each selection, for diagnostics."""
    cases_used: List[int]
    """How many cases each selection consumed before the pool reached 1."""
    eps_zero_fraction: float
    """The regression guard for the v1 defect: if this is 1.0, `eps_t` is
    identically zero and the mechanism has degenerated to strict lexicase."""


def epsilon_lexicase(
    eligible: Sequence[int],
    nu: torch.Tensor,
    eps: torch.Tensor,
    w_t: torch.Tensor,
    n_par: int,
    rng: np.random.Generator,
) -> LexicaseResult:
    """`EPSILON_LEXICASE` of §04.5.

    `eligible` are slot indices drawn from `U_parent`; `nu` is `(q, U_max)`;
    `eps` and `w_t` are `(q,)`.

    **Lemma 4.1 (elite guarantee).** Every returned parent is within
    `eps_{t_1}` of the pool maximum on the first case used to select it —
    because the first iteration filters `S_pool` to exactly that set, and
    `S_pool` only shrinks.

    **Lemma 4.2 (termination).** The inner loop terminates in at most `|cases|`
    iterations, since `pop_front` strictly decreases `|T_q|`.

    The loop guard is `|T_q| >= 1`, so the **final case is used**; spec v1's
    `> 1` discarded it.
    """
    elig = np.asarray(list(eligible), dtype=np.int64)
    if elig.size < 2:
        raise ValueError(
            f"EPSILON_LEXICASE needs |U_parent| >= 2; got {elig.size}"
        )
    q = int(nu.shape[0])
    if q < 1:
        raise ValueError("EPSILON_LEXICASE needs at least one case")

    nu_np = nu.detach().cpu().numpy()
    eps_np = eps.detach().cpu().numpy()
    w_np = w_t.detach().cpu().numpy().astype(np.float64)
    w_sum = float(w_np.sum())

    parents: List[int] = []
    pool_trace: List[int] = []
    cases_used: List[int] = []

    for _ in range(int(n_par)):
        pool = elig.copy()
        order = _weighted_permutation(q, w_np, w_sum, rng)
        used = 0
        first = True
        for t in order:
            if pool.size <= 1:
                break
            col = nu_np[t][pool]
            elite = col.max()
            keep = col >= elite - eps_np[t]
            pool = pool[keep]
            used += 1
            if first:
                pool_trace.append(int(pool.size))
                first = False
        if first:
            pool_trace.append(int(pool.size))
        cases_used.append(used)
        parents.append(int(pool[int(rng.integers(0, pool.size))]))

    return LexicaseResult(
        parents, pool_trace, cases_used, float((eps_np == 0).mean())
    )


def _weighted_permutation(
    q: int, w: np.ndarray, w_sum: float, rng: np.random.Generator
) -> np.ndarray:
    """`SHUFFLE(cases, weights = w)` of §04.5.

    Draws a permutation with probability proportional to `w_t` at each draw
    without replacement, and **falls back to a uniform permutation when
    `sum_t w_t = 0`** — reachable whenever the model has improved on no sampled
    case since the lag, which (4.1) makes an ordinary occurrence rather than an
    edge case.

    Implemented by the Gumbel/exponential-race construction, which is exactly
    equivalent to sequential proportional-without-replacement draws and is
    `O(q log q)` rather than `O(q^2)`.
    """
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        return rng.permutation(q)
    pos = w > 0
    keys = np.full(q, np.inf)
    # E_i / w_i with E_i ~ Exp(1): ascending order is the sequential draw.
    keys[pos] = rng.exponential(size=int(pos.sum())) / w[pos]
    # Zero-weight cases retain a random relative order at the tail.
    tail = rng.random(q - int(pos.sum()))
    keys[~pos] = 1e30 + tail
    return np.argsort(keys, kind="stable")


def mean_utility_parents(
    eligible: Sequence[int], nu_hat_vec: torch.Tensor, n_par: int
) -> List[int]:
    """Ablation **A2**: parents = top-`n_par` by `nu_hat` (§08.6).

    SPEC: 08 §8.6

    Spec v1's A2 specified only a death rule, leaving the ablated arm with **no
    way to produce births**. A2 now replaces both.
    """
    elig = list(eligible)
    if not elig:
        return []
    scores = nu_hat_vec[torch.tensor(elig, device=nu_hat_vec.device)]
    k = min(int(n_par), len(elig))
    top = torch.topk(scores, k=k).indices.tolist()
    return [elig[i] for i in top]


def case_win_histogram(
    eligible: Sequence[int], nu: torch.Tensor, eps: torch.Tensor
) -> Dict[int, int]:
    """The case-win diagnostic of §04.10, **independent of the operator**.

    SPEC: 04 §4.10

    A "case win" is *membership of the surviving pool after the first case of a
    selection event*. It is computed as a diagnostic independent of the
    selection operator, so it exists in **both** arms of A2; spec v1 defined it
    only inside lexicase, so K3's comparison had data in one arm only.

    Here it is evaluated over every case rather than over one sampled first
    case, which is the operator-free version of the same quantity: the number of
    cases on which a unit would survive the first filter.
    """
    elig = torch.tensor(list(eligible), device=nu.device)
    if elig.numel() == 0:
        return {}
    sub = nu[:, elig]                                  # (q, |elig|)
    elite = sub.max(dim=-1, keepdim=True).values
    keep = sub >= (elite - eps.unsqueeze(-1))
    counts = keep.sum(dim=0)
    return {int(elig[i]): int(counts[i]) for i in range(elig.numel())}
