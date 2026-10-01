"""Open-endedness certification: the three-arm trend test.

SPEC: 06 §6.4

```
Novelty:      forall t, forall T>t, exists T' > T :  E[l(t,T')] > E[l(t,T)]   (6.9)
Learnability: forall T, forall t<t'<T :             E[l(t',T)] < E[l(t,T)]    (6.10)
```

**[NORMATIVE — the claim is weaker than (6.9)-(6.10)]** These are quantified
over unbounded time and cannot be decided from a finite matrix; literal novelty
would require unbounded loss growth, which a bounded NLL cannot supply. What is
implemented is a **finite-sample trend test**, which is a strictly weaker claim,
and it is labelled as such.

**Instantiation.** Observer = frozen checkpoint from the ring; artifact = the
synthetic batch emitted at step `T_k`, stored in a ring of `K_ring = 32`;
`l(t,T)` = mean next-token NLL of `theta^{(t_j)}` on batch `X^{(T_k)}`.

```
Lhat[j,k] = NLL( theta^{(t_j)} ; X^{(T_k)} )                             (6.11)
```

**Three arms, because a frozen control is not enough (audit A-28).** The frozen
arm separates "the archive changes" from "the archive is static", which is *not*
the confound §6.4 names. A pure difficulty-tracker — one scalar ratcheted
against the current model — also beats a frozen archive on every novelty
statistic, because it too keeps producing instances older checkpoints have not
seen. The **ramp** arm is what distinguishes genuine open-endedness from
monotone difficulty ramping, and K5 is adjudicated on `live - ramp`.

**Granularity.** With `J_ring = 8` and the `n >= 6` exclusion, **7 rows**
qualify, so the per-firing novelty fraction is quantised to multiples of `1/7`.
§08.5's K5 margin is `0.34`, which exceeds that; v2's `0.10` lay entirely inside
the gap between two attainable values, so neither TOST could reject and K5 was
undecidable in both directions (audit A-29).

**[NORMATIVE — v3.1, audit A-51/A-77]** v3.0 stated "at most 3 rows qualify …
quantised to `{0, 1/3, 2/3, 1}`" here and at `granularity` below. The realised
count is 7 and the quantum is `1/7`. The error was **conservative** — a margin
chosen against a coarser grid than the real one is still above the real one — so
nothing downstream was wrong, which is exactly why it survived a repair pass that
corrected the specification and left both docstrings behind. `granularity` itself
was always computed as `1 / n_rows_tested` and was never wrong; only the prose
was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from soma.utils.stats import benjamini_hochberg, mann_kendall

__all__ = [
    "RampController",
    "J_RING",
    "K_RING",
    "MIN_USABLE",
    "CertificationResult",
    "certify",
    "CheckpointRing",
    "BatchRing",
    "capture_cadence",
]

J_RING = 8
"""§06.1.3: frozen checkpoints, CPU-resident, `bfloat16`."""
K_RING = 32
"""§06.4: stored synthetic batches."""
MIN_USABLE = 6
"""§06.4: rows/columns with `n < 6` usable points are excluded from **both**
numerator and denominator of the reported fraction."""
ALPHA_SIG = 0.05


def capture_cadence(N_step: int, ring: int) -> int:
    """`ceil(N_step / ring)` — the decoupled capture cadence (§06.4).

    SPEC: 06 §6.4

    Batches are captured every `ceil(N_step / K_ring)` steps, so the ring
    **spans the whole run**; certification *fires* every `T_cert` steps (1.19).
    v2 tied capture to `T_cert / K_ring`, which is self-contradictory: "the ring
    spans the run" requires `T_cert ~ N_step`, at which point certification
    fires at most once — and with the counter of (1.15) starting at `T_cert`,
    zero times (audit A-23).
    """
    return max(1, math.ceil(N_step / max(1, ring)))


class CheckpointRing:
    """`J_ring = 8` frozen checkpoints spanning the run logarithmically.

    SPEC: 06 §6.1.3

    **[NORMATIVE]** Eviction keeps the checkpoints nearest to `N_step * 2^-j`
    for `j = 0..J_ring-1`, so the retained set spans the run logarithmically and
    does **not** require knowing `N_step` in advance beyond the configured value.

    Spec v1 said "ring buffer ... spaced logarithmically", which is
    self-contradictory (a ring evicts oldest; log spacing evicts the middle),
    gave no residency, and omitted the memory from §05.10.
    """

    def __init__(self, N_step: int, size: int = J_RING):
        self.N_step = int(N_step)
        self.size = int(size)
        self.targets = [self.N_step * (2.0 ** -j) for j in range(self.size)]
        self.slots: List[Optional[Tuple[int, object]]] = [None] * self.size

    def offer(self, step: int, snapshot: object) -> Optional[int]:
        """Offer a checkpoint; returns the slot it took, or `None`."""
        best = min(range(self.size), key=lambda j: abs(self.targets[j] - step))
        cur = self.slots[best]
        if cur is None or abs(self.targets[best] - step) < abs(self.targets[best] - cur[0]):
            self.slots[best] = (step, snapshot)
            return best
        return None

    def entries(self) -> List[Tuple[int, object]]:
        """Retained `(step, snapshot)` pairs, ascending by step."""
        return sorted([s for s in self.slots if s is not None], key=lambda s: s[0])


class BatchRing:
    """`K_ring = 32` stored synthetic batches, captured across the run."""

    def __init__(self, size: int = K_RING):
        self.size = int(size)
        self.items: List[Tuple[int, object]] = []

    def offer(self, step: int, batch: object) -> None:
        self.items.append((step, batch))
        if len(self.items) > self.size:
            # Keep an even spread across the run rather than the most recent:
            # a ring of the last 32 emitted batches spans ~32 captures and the
            # novelty test would measure adjacent-step noise (§06.4).
            keep_idx = np.linspace(0, len(self.items) - 1, self.size).round().astype(int)
            self.items = [self.items[i] for i in sorted(set(keep_idx.tolist()))]

    def entries(self) -> List[Tuple[int, object]]:
        return sorted(self.items, key=lambda s: s[0])


@dataclass
class CertificationResult:
    """One arm's verdict."""

    arm: str
    novelty_fraction: float
    learnability_fraction: float
    n_rows_tested: int
    n_cols_tested: int
    row_p_values: List[float] = field(default_factory=list)
    col_p_values: List[float] = field(default_factory=list)
    matrix_shape: Tuple[int, int] = (0, 0)

    def as_dict(self) -> Dict[str, object]:
        return {
            "arm": self.arm,
            "novelty_fraction": self.novelty_fraction,
            "learnability_fraction": self.learnability_fraction,
            "n_rows_tested": self.n_rows_tested,
            "n_cols_tested": self.n_cols_tested,
            "matrix_shape": list(self.matrix_shape),
        }


def certify(
    L_hat: np.ndarray,
    checkpoint_steps: Sequence[int],
    batch_steps: Sequence[int],
    *,
    arm: str = "live",
    alpha: float = ALPHA_SIG,
) -> CertificationResult:
    """Run the Mann-Kendall trend tests on (6.11)'s matrix.

    SPEC: 06 §6.4

    `L_hat[j, k] = NLL(theta^{(t_j)} ; X^{(T_k)})`, with `checkpoint_steps[j]`
    and `batch_steps[k]` giving `t_j` and `T_k`.

    **Novelty**: for each row `j`, test `L_hat[j, k]` for a **positive** trend in
    `k` over `T_k > t_j`.
    **Learnability**: for each column `k`, test for a **negative** trend in `j`
    over `t_j < T_k`.

    Fully specified per §06.4: original MK statistic, tie-corrected variance,
    exact null for `n <= 10`, one-sided, rows with `n < 6` usable points excluded
    from **both** numerator and denominator, Benjamini-Hochberg across rows.
    """
    L = np.asarray(L_hat, dtype=float)
    if L.ndim != 2:
        raise ValueError("L_hat must be a 2-D (checkpoint x batch) matrix")
    J, K = L.shape
    t = np.asarray(checkpoint_steps, dtype=float)
    T = np.asarray(batch_steps, dtype=float)
    if t.size != J or T.size != K:
        raise ValueError(
            f"step vectors {t.size}/{T.size} do not match matrix shape {L.shape}"
        )

    # --- novelty: rows, ascending k, restricted to T_k > t_j ---------------
    row_p: List[float] = []
    for j in range(J):
        sel = T > t[j]
        vals = L[j][sel]
        if vals.size < MIN_USABLE:
            continue                      # excluded from numerator AND denominator
        row_p.append(mann_kendall(vals, alternative="increasing").p_value)
    novelty = (
        float(benjamini_hochberg(row_p, alpha).mean()) if row_p else 0.0
    )

    # --- learnability: columns, ascending j, restricted to t_j < T_k -------
    col_p: List[float] = []
    for k in range(K):
        sel = t < T[k]
        vals = L[:, k][sel]
        if vals.size < MIN_USABLE:
            continue
        col_p.append(mann_kendall(vals, alternative="decreasing").p_value)
    learnability = (
        float(benjamini_hochberg(col_p, alpha).mean()) if col_p else 0.0
    )

    return CertificationResult(
        arm, novelty, learnability, len(row_p), len(col_p), row_p, col_p, (J, K)
    )


@dataclass
class CertificationVerdict:
    """The three-arm comparison and K5's decision (§06.4, §10 K5)."""

    live: CertificationResult
    frozen: CertificationResult
    ramp: CertificationResult
    step: int

    @property
    def live_minus_ramp(self) -> float:
        """The reported criterion (§06.4)."""
        return self.live.novelty_fraction - self.ramp.novelty_fraction

    @property
    def live_minus_frozen(self) -> float:
        """The sanity floor: live must also exceed frozen, or the machinery is
        inert."""
        return self.live.novelty_fraction - self.frozen.novelty_fraction

    @property
    def granularity(self) -> float:
        """The smallest attainable non-zero difference in the novelty fraction.

        `1 / n_rows_tested`. §08.5 requires every TOST margin to exceed the
        granularity of its statistic; K5's margin of `0.34` is set against the
        `1/7` this returns at `J_ring = 8` (audit A-29, corrected by A-51).
        """
        n = max(self.live.n_rows_tested, 1)
        return 1.0 / n

    def as_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "live": self.live.as_dict(),
            "frozen": self.frozen.as_dict(),
            "ramp": self.ramp.as_dict(),
            "live_minus_ramp": self.live_minus_ramp,
            "live_minus_frozen": self.live_minus_frozen,
            "granularity": self.granularity,
        }


class RampController:
    """The **ramp** control arm: a monotone ratchet on `n_len`, **matched**.

    SPEC: 06 §6.4

    This is the confound the frozen arm cannot exclude. A pure difficulty
    tracker produces instances older checkpoints have not seen, so it beats a
    frozen archive on novelty without being open-ended in any interesting sense.

    **[NORMATIVE — the ramp is matched to the live arm; v3.1, audit A-79]**
    §06.4 specifies *"a scripted monotone ratchet on `n_len` alone, **matched to
    the live arm's mean solve-rate trajectory**"*, and K5 is adjudicated on
    `live - ramp`. v3.0 implemented an **unmatched** linear interpolation of
    `n_len` from 4 to 32 across the run, with a docstring asserting a linear ramp
    was "the simplest schedule with that property" and nothing measuring the
    property. Nothing anywhere read the live arm's solve rate.

    An unmatched ramp makes `live - ramp` a comparison of two different task
    *distributions* rather than of two archive *dynamics*, which is exactly the
    attribution A-28 introduced the arm to obtain. If the ramp is harder than the
    live archive, the ramp wins on novelty for a reason that has nothing to do
    with open-endedness; if it is easier, the live arm does.

    **The matching rule.** The controller holds `n_len` and, at each capture,
    compares the live archive's mean `s_task` with its own last measured solve
    rate:

    * live is **easier** than the ramp (`s_live - s_ramp > tol`) -> ratchet
      `n_len` up by one, bounded by `hi`;
    * otherwise hold.

    It is a **ratchet**: `n_len` never decreases, which is what "monotone" means
    and what keeps the arm a difficulty *tracker* rather than a second adaptive
    archive. Matching is therefore one-sided by construction — the ramp can be
    made harder to catch up, never easier to fall back — and the residual is
    reported as `matching_error` so a certification says how well matched it
    actually was rather than assuming it.
    """

    def __init__(self, lo: int = 4, hi: int = 32, tol: float = 0.05) -> None:
        self.lo = int(lo)
        self.hi = int(hi)
        self.tol = float(tol)
        self.n_len = int(lo)
        self.s_ramp = 0.0
        self.history: List[Dict[str, float]] = []

    def observe(self, s_live: float, s_ramp: float) -> int:
        """Record both arms' solve rates and ratchet if the live arm is easier.

        Returns the `n_len` to use for the **next** capture.
        """
        self.s_ramp = float(s_ramp)
        before = self.n_len
        if float(s_live) - float(s_ramp) > self.tol and self.n_len < self.hi:
            self.n_len += 1
        self.history.append({
            "s_live": float(s_live), "s_ramp": float(s_ramp),
            "n_len_before": before, "n_len_after": self.n_len,
        })
        return self.n_len

    @property
    def matching_error(self) -> float:
        """Mean `|s_live - s_ramp|` over the run.

        Reported with every certification: a large value means `live - ramp`
        compared two different difficulty regimes and K5's verdict is not
        attributable, which the arm exists to prevent."""
        if not self.history:
            return float("nan")
        return sum(abs(h["s_live"] - h["s_ramp"]) for h in self.history) / len(
            self.history
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "ramp_n_len": self.n_len,
            "ramp_s": self.s_ramp,
            "ramp_matching_error": self.matching_error,
            "ramp_observations": len(self.history),
        }


def ramp_schedule(n: int, N_step: int, lo: int = 4, hi: int = 32) -> int:
    """The **unmatched** linear ratchet, retained for the A4-style control only.

    SPEC: 06 §6.4

    Kept because a matched ramp and an unmatched one are different controls and
    the difference between them is itself informative; `RampController` is what
    §06.4 specifies and what the certification uses (audit A-79).
    """
    frac = min(1.0, max(0.0, n / max(1, N_step)))
    return int(round(lo + frac * (hi - lo)))
