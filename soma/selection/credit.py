"""Case weights and the credit audit.

SPEC: 04 §4.3
SPEC: 04 §4.4
SPEC: 01 §1.7.3

```
w_t = max( 0 , L_lag(t) - L_theta(t) )                                   (4.1)
```

`w_t` is the **improvement** on case `t` since the lag — a rate, not a level.

**[NORMATIVE — v1 correction, polarity]** Spec v1 wrote
`w_t = max(0, L_theta - L_ref)` with `L_ref` the lagged self. That has the wrong
sign: a lagged self knows **less** than the current model, so
`L_theta <= L_lag` almost everywhere and `w_t` would be **identically zero** —
the same degeneracy the auditors found at `n = 0`, but permanent. (4.1) reverses
the difference.

**[NORMATIVE — v1 correction, the missing rate term]** Spec v1 additionally
declared credit must be scored on "rate of improvement over a randomised horizon
`H`", gave a formula requiring a per-case loss history that nothing maintained,
and then defined `nu_ct` with no horizon term at all — two `[NORMATIVE]`
statements in direct contradiction. The contradiction is resolved by (4.1): the
*case weight itself* is the improvement signal, so no separate horizon machinery
is needed. `H_sel`, `H_min`, `H_max` are **deleted**.

**Relationship to token selection.** §06.2 uses a *different* reference — a
frozen, better model — because Rho-1-style token selection requires a reference
that knows **more**. The two mechanisms have different jobs and correctly
different references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["case_weights", "CreditAudit", "spearman"]


def case_weights(
    lag_losses: Optional[torch.Tensor],
    cur_losses: Optional[torch.Tensor],
    q: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """(4.1): `w_t = max(0, L_lag(t) - L_theta(t))`.

    Both arguments are `(q,)` per-case losses. `None` for either — which happens
    before `theta_lag` has diverged from `theta`, i.e. in the first `T_ref`
    steps — yields an all-zero weight vector, and §04.5's `SHUFFLE` falls back
    to a uniform permutation there. That is the correct degenerate behaviour:
    with no improvement signal, every case is equally informative.
    """
    device = torch.device(device)
    if lag_losses is None or cur_losses is None:
        return torch.zeros(q, device=device)
    if lag_losses.shape != cur_losses.shape:
        raise ValueError(
            f"lag and current losses must match: {tuple(lag_losses.shape)} vs "
            f"{tuple(cur_losses.shape)}"
        )
    w = (lag_losses.float() - cur_losses.float()).clamp_min(0.0)
    return w.to(device)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation, computed directly.

    Used for the credit audit of §04.4, whose trip rule depends on the exact
    value, so it is computed here rather than through a dependency whose tie
    convention could change between versions.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size != y.size:
        raise ValueError("spearman needs equal-length inputs")
    if x.size < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


@dataclass
class CreditAudit:
    """The leave-one-out validity check of §04.4.

    SPEC: 04 §4.4

    **Exact leave-one-out.** Spec v1 claimed the counterfactual
    `L(t | m_u = 0)` is "obtainable by a single additional Site evaluation". It
    is not: it requires a forward pass from the Site's layer through every
    subsequent block and the output head. The audit path therefore evaluates a
    **partial forward from layer `l` to the head** on a random `1%` of cases.

    **[NORMATIVE]** Spearman correlation between (4.2) and the **magnitude** of
    the exact per-case effect, `|L_t(m_u = 0) - L_t|`, is logged; if it falls
    below `0.5` on a sample of at least **64 pairs**, the audit fraction
    **doubles**, capped at 10%. §04.10 makes the same threshold a falsifier.

    **[NORMATIVE — the target is a magnitude; v3.1, audit A-68]** v3.0 compared
    (4.2) against the **signed** difference. (4.2) is a product of three
    non-negative factors, so `nu_ut >= 0` identically, and a rank correlation
    between a non-negative statistic and a target that is negative on about half
    its sample is bounded near zero **whatever the proxy is**. That is not a
    failing test, it is an unsatisfiable one — the same defect class as the three
    unfireable thresholds already recorded (A-36, A-58, A-61), in the opposite
    direction.

    Both correlations are computed and both are reported. Measured on 472 pairs
    from an M-nano lifetime:

    * `rho_signed    = +0.001` — (4.2) carries **no** information about whether a
      unit helps or hurts;
    * `rho_magnitude = +0.601` — (4.2) tracks **how much** a unit changes the
      case, which is the quantity it is defined to estimate.

    This is a statement about what SOMA selects on, and it is a design property
    rather than a defect: (4.2) and §03.4.1's `y_u` are both magnitudes of
    change, so selection ranks units by **influence**, not by benefit. Benefit is
    the gradient's job. `rho_signed` remains in the log because a large positive
    value would be news.
    """

    fraction: float = 0.01
    min_pairs: int = 64
    threshold: float = 0.5
    max_fraction: float = 0.10
    last_rho: float = float("nan")
    """The gated statistic: Spearman against `|exact|`."""
    last_rho_signed: float = float("nan")
    """Reported, never gated: Spearman against the signed difference."""
    n_pairs: int = 0
    tripped: int = 0
    _proxy: List[float] = field(default_factory=list)
    _exact: List[float] = field(default_factory=list)
    n_evaluations: int = 0

    def due(self, n_cases: int) -> int:
        """How many cases to audit this event: `fraction` of them, at least 0."""
        return int(round(self.fraction * max(0, n_cases)))

    def record(self, proxy: Sequence[float], exact: Sequence[float]) -> float:
        """Accumulate pairs and evaluate once `min_pairs` are available.

        §04.4 requires the Spearman "on a sample of at least **64 pairs**". A
        single selection event supplies far fewer — `1%` of `q_cases` is 2 at
        smoke scale and 65 at M1, and units carrying no probe traffic are
        skipped — so the pairs are **accumulated across events** until the
        minimum is reached, then evaluated and the buffer cleared. Evaluating a
        2-pair sample instead would produce a correlation whose sampling
        distribution is almost uniform on `[-1, 1]`: it would trip the doubling
        rule at random, and §04.10's falsifier would fire on noise.

        Returns the Spearman when an evaluation happened, `NaN` otherwise.
        """
        self._proxy.extend(float(v) for v in proxy)
        self._exact.extend(float(v) for v in exact)
        self.n_pairs = len(self._proxy)
        if self.n_pairs < self.min_pairs:
            # `last_rho` is **not** cleared: it holds the most recent *actual*
            # evaluation, and §04.10's falsifier is a statement about that
            # measurement. Overwriting it with NaN on every under-full call
            # discards the only real number the audit produces.
            return float("nan")
        magnitude = [abs(v) for v in self._exact]
        rho = spearman(self._proxy, magnitude)
        self.last_rho = rho
        self.last_rho_signed = spearman(self._proxy, self._exact)
        self.n_evaluations += 1
        if rho < self.threshold:
            self.tripped += 1
            self.fraction = min(self.max_fraction, self.fraction * 2.0)
        self._proxy.clear()
        self._exact.clear()
        return rho

    @property
    def status(self) -> str:
        """`"unmeasured"`, `"valid"` or `"invalid"` — three states, not two.

        **[NORMATIVE — v3.1, audit A-84]** v3.0's `valid` returned
        `isnan(last_rho) or last_rho >= threshold`, i.e. **`True` when the audit
        had never produced a number**. §04.10's falsifier is a statement about a
        *measurement*; reporting `valid: true` for a lifetime in which the
        Spearman was never computed asserts the opposite of what is known. It is
        the same distinction `soma.eval.killcriteria` draws between
        `applicable=False` and `fires=False`, and for the same reason: a report
        must be able to tell "the proxy passed" from "nobody looked".

        Observed on the integration configuration: a whole lifetime accumulated
        40 pairs against a 64-pair minimum, so `spearman` stayed `NaN`,
        `n_evaluations` stayed `0`, and `valid` read `true`.
        """
        import math as _m

        if self.n_evaluations == 0 or _m.isnan(self.last_rho):
            return "unmeasured"
        return "valid" if self.last_rho >= self.threshold else "invalid"

    @property
    def valid(self) -> Optional[bool]:
        """`True`/`False` once measured, and **`None`** before.

        `None` rather than a default in either direction: a consumer that treats
        it as a boolean gets a falsy value and looks again, which is the failure
        mode worth having.
        """
        st = self.status
        return None if st == "unmeasured" else st == "valid"

    def as_dict(self) -> dict:
        return {
            "audit_fraction": self.fraction,
            "spearman": self.last_rho,
            "spearman_signed": self.last_rho_signed,
            "n_pairs": self.n_pairs,
            "pending_pairs": len(self._proxy),
            "n_evaluations": self.n_evaluations,
            "tripped": self.tripped,
            "valid": self.valid,
            "status": self.status,
        }
