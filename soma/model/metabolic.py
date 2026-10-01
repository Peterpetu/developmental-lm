"""The metabolic functional: wiring, dispersion, tissue, and the population cap.

SPEC: 01 §1.6
SPEC: 01 §1.6.1

```
L_wire = lambda_wire * (1/|T_set|) * sum_{t in T_set} sum_{u in R_t}
              r_ut * m_u * || q_t - p_u ||^2                            (1.8)

L_disp = lambda_disp * ( 2 / (|U_live|(|U_live|-1)) ) *
              sum_{u < v, both live} exp( - ||p_u - p_v||^2 / (2 s_rep^2) )  (1.9)

L_size = lambda_size * (1/|U_live|) * sum_{u in U_live} m_u             (1.10)
L_met  = L_wire + L_disp + L_size                                       (1.11)
```

**Why (1.9) exists.** `L_wire >= 0` with equality **iff `p_u = q_t` for every
routed pair** — the configuration in which every position and every query
collapse to one point. That point lies on the sphere, so it is feasible, and it
is the **global** minimiser. Worse, `W_pos` appears in no other term, so `q_t`
has no gradient source other than `L_wire` and is driven purely toward collapse.
Spec v2 claimed the sphere constraint meant `L_wire` "cannot drive a collapse to
a point"; the sphere bounds the term's *value*, not its *argmin*, and the claim
was false (audit A-9). (1.9) is a Gaussian repulsion whose equilibrium is a
dispersed configuration, as in a Thomson problem on `S^{m_dim-1}` — the
character-displacement mechanism the design's ecological analogy invoked and
never implemented.

**`L_size` has identically zero gradient** (`m_u` is bookkeeping). It is the
reported metabolic bill, not the mechanism bounding the population; that
mechanism is the explicit cap `BUDGET_HEADROOM` (1.14). Ablation A1b is
therefore defined on `U_cap`, not on `lambda_size`, which would be a provable
no-op (audit A-31).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from soma.config import NOCOLLAPSE_FRACTION, S_REP, PolicyConstants, SomaConfig
from soma.utils.shapes import check_shape

__all__ = [
    "MetabolicTerms",
    "wiring_loss",
    "dispersion_loss",
    "size_loss",
    "metabolic_loss",
    "budget_headroom",
    "mean_pairwise_distance",
    "d_unif",
    "d_unif_monte_carlo",
    "nocollapse_ok",
]


@dataclass
class MetabolicTerms:
    L_wire: torch.Tensor
    L_disp: torch.Tensor
    L_size: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.L_wire + self.L_disp + self.L_size

    def as_dict(self) -> dict:
        return {
            "L_wire": float(self.L_wire),
            "L_disp": float(self.L_disp),
            "L_size": float(self.L_size),
            "L_met": float(self.total),
        }


def wiring_loss(
    q_t: torch.Tensor,
    p_u: torch.Tensor,
    slots: torch.Tensor,
    r_ut: torch.Tensor,
    lambda_wire: float,
) -> torch.Tensor:
    """(1.8). `q_t` is `(B, T, m_dim)`, `slots`/`r_ut` are `(B, T, k_route)`.

    Both positions and queries lie on the unit sphere, so `||q - p||^2` lies in
    `[0, 4]`. Weighted by `r_ut` (and by `m_u`, which the caller has already
    folded into `r_ut`), so a unit that is barely routed to a token pays barely
    anything for the distance — which is the point: the price is on *realised*
    connections, not on possible ones.

    Normalised by `|T_set|`, the number of scored tokens, so the term does not
    scale with batch size.
    """
    if lambda_wire == 0.0:
        return torch.zeros((), dtype=q_t.dtype, device=q_t.device)
    check_shape(q_t, [None, None, p_u.shape[-1]], name="q_t")
    B, T, k = slots.shape
    safe = slots.clamp_min(0)
    p_sel = p_u[safe]                                    # (B,T,k,m_dim)
    d2 = (q_t.unsqueeze(2) - p_sel).square().sum(-1)     # (B,T,k)
    valid = (slots >= 0).to(d2.dtype)
    total = (r_ut * valid * d2).sum()
    return lambda_wire * total / max(1, B * T)


def dispersion_loss(
    p_u: torch.Tensor,
    live: torch.Tensor,
    lambda_disp: float,
    s_rep: float = S_REP,
) -> torch.Tensor:
    """(1.9). Gaussian repulsion between live positions.

    `p_u` is `(U_max, m_dim)`, `live` is `(U_max,)` bool.

    Normalised by the number of unordered live pairs so the term does not grow
    quadratically with the population — otherwise the cheapest way to reduce it
    would be to kill units, and the dispersion price would silently become a
    second population cap fighting the first.
    """
    if lambda_disp == 0.0:
        return torch.zeros((), dtype=p_u.dtype, device=p_u.device)
    idx = torch.nonzero(live, as_tuple=False).squeeze(-1)
    n = int(idx.numel())
    if n < 2:
        return torch.zeros((), dtype=p_u.dtype, device=p_u.device)
    p = p_u[idx]
    d2 = torch.cdist(p, p).square()
    iu = torch.triu_indices(n, n, offset=1, device=p_u.device)
    pair = torch.exp(-d2[iu[0], iu[1]] / (2.0 * s_rep * s_rep))
    return lambda_disp * (2.0 / (n * (n - 1))) * pair.sum()


def size_loss(m_u: torch.Tensor, live: torch.Tensor, lambda_size: float) -> torch.Tensor:
    """(1.10). Reported diagnostic; **identically zero gradient** (§01.6.1).

    Normalised by `|U_live|`, not by the token count. `L_size` sums over *units*
    and does not depend on `t`, so spec v1's division by `|T_set|` made it
    inversely proportional to batch size — contradicting the sentence that
    justified it.
    """
    n = int(live.sum())
    if n == 0 or lambda_size == 0.0:
        return torch.zeros((), dtype=m_u.dtype, device=m_u.device)
    return lambda_size * m_u[live].sum() / n


def metabolic_loss(
    q_t: torch.Tensor,
    p_u: torch.Tensor,
    slots: torch.Tensor,
    r_ut: torch.Tensor,
    m_u: torch.Tensor,
    live: torch.Tensor,
    policy: PolicyConstants,
) -> MetabolicTerms:
    """(1.8)-(1.11)."""
    return MetabolicTerms(
        wiring_loss(q_t, p_u, slots, r_ut, policy.lambda_wire),
        dispersion_loss(p_u, live, policy.lambda_disp),
        size_loss(m_u, live, policy.lambda_size),
    )


# ---------------------------------------------------------------------------
# Population bound (§01.6.1)
# ---------------------------------------------------------------------------


def budget_headroom(
    U_cap: int, n_live: int, n_death: int, *, n_dying: int = 0, U_max: int = 0
) -> int:
    """(1.14): the births a cohort may add, under **two** bounds.

    SPEC: 01 §1.6.1

    ```
    BUDGET_HEADROOM = min( U_cap - |{u : live and not dying}| ,
                           U_max - |U_live| )                            (1.14)
    ```

    Two bounds, because two distinct things must not be exceeded and they are
    not the same thing:

    * `U_cap` bounds the **effective** population — the units that actually
      compute. A dying unit is annealing to `m_u = 0` and is excluded from
      routing already (§05.4), so it does not count against it.
    * `U_max` bounds the **slots**. A dying unit still occupies one for
      `phi_ann` steps, and `FREE_SLOT_NEAREST` raises when the grid is full, so
      the slot bound must count it.

    **[NORMATIVE — v3.1 correction]** v3.0 computed `U_cap - (|U_live| - n_death)`,
    which assumes the event's deaths free their slots immediately. They do not:
    `DEATH` sets `dying` and keeps `live_u = True` for `phi_ann` further steps
    (§03.4). The realised population therefore **exceeded the cap** — measured
    `|U_live| = 243` against `U_cap = 240` — and with `U_max - U_cap = 16` of
    slack at M-nano a few concurrent events could exhaust the grid entirely.

    This is audit A-17's defect approached from the other side: v2 evaluated
    headroom *before* deaths and births became permanently zero at the cap; v3.0
    evaluated it net of deaths and the cap stopped binding. Separating the two
    bounds makes both impossible.
    """
    if n_death < 0 or n_dying < 0:
        raise ValueError("n_death and n_dying must be non-negative")
    n_nondying = int(n_live) - int(n_dying) - int(n_death)
    by_cap = int(U_cap) - n_nondying
    by_slots = (int(U_max) - int(n_live)) if U_max else by_cap
    return max(0, min(by_cap, by_slots))


# ---------------------------------------------------------------------------
# Collapse detection (1.13)
# ---------------------------------------------------------------------------


def mean_pairwise_distance(p_u: torch.Tensor, live: torch.Tensor) -> float:
    """Mean `||p_u - p_v||` over unordered live pairs."""
    idx = torch.nonzero(live, as_tuple=False).squeeze(-1)
    n = int(idx.numel())
    if n < 2:
        return 0.0
    p = p_u[idx].float()
    d = torch.cdist(p, p)
    iu = torch.triu_indices(n, n, offset=1, device=p_u.device)
    return float(d[iu[0], iu[1]].mean())


def d_unif(m_dim: int) -> float:
    """(1.13a): the exact mean chord length on `S^{m_dim-1}`.

    SPEC: 01 §1.6

    For `p, q` independent and uniform on `S^{m-1}`, `<p,q>` has density
    proportional to `(1 - t^2)^{(m-3)/2}` on `[-1,1]`, and

        D_unif(m) = E||p - q|| = 2^{m-1} Gamma(m/2)^2 / ( sqrt(pi) Gamma(m-1/2) )

    giving `4/pi` at `m = 2`, `4/3` at `m = 3`, and `1.389328` at `m = 8`.

    Spec v3.0 tabulated `D_unif(8) = 1.375` as "measured" — wrong by 1.0%
    (audit A-48). The closed form removes the possibility of that class of error
    entirely: a threshold derived from a literal that nothing checks is a
    threshold that drifts.
    """
    if m_dim < 2:
        raise ValueError(f"the sphere S^(m-1) needs m >= 2; got {m_dim}")
    return math.exp(
        (m_dim - 1) * math.log(2.0)
        + 2.0 * math.lgamma(m_dim / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma(m_dim - 0.5)
    )


def d_unif_monte_carlo(
    m_dim: int, n: int = 4096, generator: Optional[torch.Generator] = None
) -> float:
    """Monte-Carlo estimate of `D_unif`, used to check (1.13a) independently."""
    from soma.germline.field import sample_sphere

    p = sample_sphere(n, m_dim, generator, dtype=torch.float64)
    d = torch.cdist(p, p)
    iu = torch.triu_indices(n, n, offset=1)
    return float(d[iu[0], iu[1]].mean())


def nocollapse_ok(
    p_u: torch.Tensor, live: torch.Tensor, m_dim: int, d_unif_value: float
) -> Tuple[bool, float, float]:
    """(1.13): `mean ||p_u - p_v|| >= 0.60 * D_unif(m_dim)`.

    SPEC: 01 §1.6

    Returns `(ok, realised, threshold)`. Spec v2 referenced this gate without
    ever defining a threshold, so `G-NOCOLLAPSE` was a name with no criterion.
    """
    realised = mean_pairwise_distance(p_u, live)
    threshold = NOCOLLAPSE_FRACTION * d_unif_value
    return realised >= threshold, realised, threshold
