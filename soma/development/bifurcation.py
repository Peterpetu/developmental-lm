"""The bifurcation alarm: is a birth a controlled probe or a state change?

SPEC: 01 §1.7.2
SPEC: 03 §3.3
SPEC: 03 §3.7

```
bif = | L_task( probe ; theta_{n_b + phi_ann}, cohort active )
      - L_task( probe ; theta_{n_b + phi_ann}, cohort forced to m_u = 0 ) |   (1.21)
```

Both terms use the same weights and the same fixed probe batch and differ only
in whether the cohort born at `n_b` is switched on.

**[NORMATIVE — v2 correction]** v2 measured
`|L_task(probe; after) - L_task(probe; before)|` **at** a birth. Theorem 3.1a
guarantees that difference is zero to floating-point reduction order, so the
alarm was **unreachable**, `rho <- rho/2` was dead code, M1 criterion 2 was
trivially satisfied and §03.7's falsifier was inert. Measuring after the anneal,
against a masked counterfactual, is what makes the quantity non-trivial.

`epsilon_bif = 3 * sd_probe`, where `sd_probe` is the standard deviation of
`L_task` on the fixed probe across `k_probe = 8` repeated forward passes under
the run's nondeterminism. v2 defined it as a "batch-to-batch" standard deviation
of a batch it also mandated be **fixed** — a quantity that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

__all__ = ["BifurcationMonitor", "PendingCohort", "measure_sd_probe"]

K_PROBE = 8
"""§01.7.2: repeated forward passes used to estimate `sd_probe`."""

BIF_PROPORTIONALITY = 3.0
"""§01.7.2: a cohort may change the probe by up to this multiple of its share.

The alarm asks whether a birth cohort behaves as a **controlled probe**. A
cohort of `c` units in a population of `|U_live|` carries roughly `c/|U_live|`
of the routed mass, so changing the probe loss by about that fraction is
*expected* — it is what those units do. A bifurcation is a change
**disproportionate** to the cohort's size, and the criterion says so directly:

    fire when   |L_active - L_masked| / |L_active|  >  3 * c / |U_live|

Two earlier forms both failed, in opposite directions, and are recorded because
the shape of the failure is instructive (audit A-58, A-61):

* `3 * sd_probe` alone is **zero** on deterministic hardware, so every cohort
  fired;
* `max(3 * sd_probe, 1e-3 * |L_probe(0)|)` anchors an **absolute** threshold to
  the step-0 logit scale, which grows by orders of magnitude during training
  while the threshold does not — measured, **67%** of selection cohorts fired on
  a healthy run, and the criterion was reporting the loss scale rather than the
  operator.

Both were unfireable or always-firing rather than informative. A relative,
size-aware criterion is scale-free over the run and size-free over the schedule.
"""

EPSILON_BIF_REL_FLOOR = 1e-3
"""An absolute floor on the *relative* difference, so a cohort of one unit in a
large population is not judged against a threshold indistinguishable from
floating-point noise."""


def epsilon_bif_from(sd_probe: float, loss_scale: float) -> float:
    """The **relative** noise floor, `max(3 sd_probe / |L|, rel_floor)`.

    SPEC: 01 §1.7.2

    Returned as a fraction, so it composes with the proportionality term of
    `BifurcationMonitor.threshold_for`.
    """
    scale = max(abs(float(loss_scale)), 1e-12)
    return max(3.0 * float(sd_probe) / scale, EPSILON_BIF_REL_FLOOR)


def measure_sd_probe(
    loss_fn: Callable[[], float], k_probe: int = K_PROBE
) -> float:
    """`sd_probe`: the run's own nondeterminism on a **fixed** probe batch.

    SPEC: 01 §1.7.2

    Repeated forward passes on the *same* batch with the *same* weights. Any
    non-zero spread comes from cuBLAS reduction order, atomics and TF32 — the
    floor below which no difference is meaningful.

    This is **not** `noise_floor` (§07.7), which is the seed-to-seed standard
    deviation over five differently-seeded M0 repeats and is used for the TOST
    margins. §07.7 v3 forbids substituting one for the other: they differ by
    orders of magnitude and measure different things.
    """
    vals = [float(loss_fn()) for _ in range(k_probe)]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return var ** 0.5


@dataclass
class PendingCohort:
    """A birth cohort awaiting its post-anneal measurement."""

    site: int
    slots: List[int]
    born_at: int
    due_at: int
    """`born_at + phi_ann` — when the anneal completes."""
    origin: str = "select"
    """`"select"` or `"grow"`. §03.7's falsifier is about **births behaving as a
    controlled probe**, and a scheduled `GROW` event that adds `3|U_live|` units
    at once is not a probe — it is a deliberate capacity change whose whole
    purpose is to alter the function. Mixing the two would make the falsifier
    report the growth schedule rather than the birth operator."""


@dataclass
class BifurcationMonitor:
    """Schedules and evaluates (1.21) for every birth cohort.

    On `bif > epsilon_bif`, `rho <- rho / 2` for the remainder of this lifetime,
    and the event is logged (§01.7.2).

    **[NORMATIVE — in-lifetime policy writes are local]** The `rho` reduction
    acts on a **deep copy** of `Policy` taken at `LIFETIME` step 0. Only the
    pre-lifetime values enter `ES_STEP`. Without this the write would be
    Lamarckian inheritance across generations, and `ASSERT_BARRIER` — which
    checks `requires_grad`, `.grad` and optimizer membership — cannot detect a
    direct write.
    """

    epsilon_bif: float
    phi_ann: int
    pending: List[PendingCohort] = field(default_factory=list)
    n_births: int = 0
    n_alarms: int = 0
    n_cohorts: Dict[str, int] = field(default_factory=dict)
    n_alarms_by_origin: Dict[str, int] = field(default_factory=dict)
    rho_scale: float = 1.0
    """Multiplies `rho(n)`; halved on every alarm, for this lifetime only."""
    history: List[Dict[str, float]] = field(default_factory=list)

    def register_cohort(
        self, site: int, slots: Sequence[int], n: int, origin: str = "select"
    ) -> None:
        if not slots:
            return
        self.n_births += len(slots)
        self.n_cohorts[origin] = self.n_cohorts.get(origin, 0) + 1
        self.pending.append(
            PendingCohort(site, list(slots), n, n + self.phi_ann, origin)
        )

    def due(self, n: int) -> List[PendingCohort]:
        ready = [c for c in self.pending if n >= c.due_at]
        self.pending = [c for c in self.pending if n < c.due_at]
        return ready

    def threshold_for(self, cohort_size: int, n_live: int) -> float:
        """The relative threshold for a cohort of this size (§01.7.2).

        `max( noise_floor , 3 * cohort_size / |U_live| )`.
        """
        share = cohort_size / max(1, n_live)
        return max(self.epsilon_bif, BIF_PROPORTIONALITY * share)

    def evaluate(
        self,
        cohort: PendingCohort,
        loss_active: float,
        loss_masked: float,
        n: int,
        n_live: int = 1,
    ) -> bool:
        """Apply (1.21) to one cohort. Returns True if the alarm fired."""
        scale = max(abs(float(loss_active)), 1e-12)
        bif = abs(float(loss_active) - float(loss_masked)) / scale
        thresh = self.threshold_for(len(cohort.slots), n_live)
        fired = bif > thresh
        if fired:
            self.n_alarms += 1
            self.n_alarms_by_origin[cohort.origin] = (
                self.n_alarms_by_origin.get(cohort.origin, 0) + 1
            )
            # Only a *selection* cohort's alarm slows structural change. A
            # growth cohort is expected to alter the function -- that is what
            # growth is -- so letting it halve `rho` would make the schedule
            # silence the very mechanism it was scheduled alongside.
            if cohort.origin == "select":
                self.rho_scale *= 0.5
        self.history.append(
            {
                "step": n,
                "site": cohort.site,
                "born_at": cohort.born_at,
                "cohort_size": len(cohort.slots),
                "bif_relative": bif,
                "threshold": thresh,
                "share": len(cohort.slots) / max(1, n_live),
                "noise_floor": self.epsilon_bif,
                "fired": bool(fired),
                "rho_scale": self.rho_scale,
            }
        )
        return fired

    @property
    def alarm_rate(self) -> float:
        """§03.7 / M1 criterion 2, on **selection** cohorts.

        The denominator is *cohorts evaluated*, not units born. (1.21) is
        evaluated once per cohort, so one alarm on a 176-unit growth cohort is
        one event, not `1/176` of one; dividing by units born makes a large
        cohort's alarm arithmetically incapable of crossing a 1% bar, which is
        the same class of defect as A-22's unreachable `cap_off`.

        Growth cohorts are excluded: §03.7's falsifier asks whether **births
        behave as a controlled probe**, and a scheduled capacity change is not a
        probe. Its rate is reported separately as `growth_alarm_rate`.
        """
        n = self.n_cohorts.get("select", 0)
        return self.n_alarms_by_origin.get("select", 0) / max(1, n)

    @property
    def growth_alarm_rate(self) -> float:
        """The same statistic over growth cohorts, reported but not a falsifier."""
        n = self.n_cohorts.get("grow", 0)
        return self.n_alarms_by_origin.get("grow", 0) / max(1, n)

    @property
    def births_alarm_rate(self) -> float:
        """The v3.0 statistic — alarms per unit born — retained for comparison."""
        return self.n_alarms / max(1, self.n_births)
