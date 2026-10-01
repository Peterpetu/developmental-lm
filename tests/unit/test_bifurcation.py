"""The bifurcation alarm: §03.7's falsifier and M1 criterion 2.

SPEC: 01 §1.7.2
SPEC: 03 §3.3
SPEC: 03 §3.7

Gate `G-BIFURCATION`. A full M-nano lifetime reports `alarm_rate = 0.0` over 42
selection cohorts, and that number means **nothing** on its own: an alarm that
cannot fire reports zero exactly as reliably as a healthy population does. This
project has already recorded three thresholds that could not report what they
claimed — A-36 (unreachable), A-58 (always firing on a zero threshold), A-61
(anchored to a drifting scale, 67% of cohorts) — so the alarm's *reachability*
is tested here before its silence is believed.

Zero is nonetheless the **prediction**: Theorem 3.1a makes a birth exactly
function-preserving, with `m_u` annealing from zero, so a healthy birth cohort
should move the probe by roughly its share of the routed mass and no more. The
alarm exists to catch a violation of that guarantee, and the tests below are the
violation and the non-violation, side by side.
"""

from __future__ import annotations

import pytest

from soma.development.bifurcation import (
    BIF_PROPORTIONALITY,
    EPSILON_BIF_REL_FLOOR,
    BifurcationMonitor,
    epsilon_bif_from,
    measure_sd_probe,
)


def _monitor(eps: float = EPSILON_BIF_REL_FLOOR, phi_ann: int = 50):
    return BifurcationMonitor(epsilon_bif=eps, phi_ann=phi_ann)


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------


def test_the_noise_floor_is_never_zero_on_deterministic_hardware() -> None:
    """Audit A-58: `3 * sd_probe` is **exactly zero** when repeated forward
    passes are bit-identical, and a threshold of zero is not a threshold — the
    measured consequence was an alarm rate of 1.12% on differences of `1e-9`."""
    assert measure_sd_probe(lambda: 4.0, k_probe=8) == 0.0
    assert epsilon_bif_from(0.0, loss_scale=4.0) == EPSILON_BIF_REL_FLOOR
    assert epsilon_bif_from(0.0, loss_scale=0.0) == EPSILON_BIF_REL_FLOOR


def test_the_noise_floor_is_relative_so_it_does_not_drift_with_the_loss() -> None:
    """Audit A-61: an absolute floor anchored to the step-0 logit scale fires on
    67% of cohorts once the loss has moved. The floor is a *fraction*."""
    a = epsilon_bif_from(sd_probe=0.004, loss_scale=8.0)
    b = epsilon_bif_from(sd_probe=0.001, loss_scale=2.0)
    assert a == pytest.approx(b), (a, b)


def test_the_threshold_is_proportional_to_the_cohort_share() -> None:
    """(1.22): a cohort of `c` in `|U_live|` carries about `c/|U_live|` of the
    routed mass, so moving the probe by that much is *expected*."""
    m = _monitor()
    assert m.threshold_for(8, 240) == pytest.approx(BIF_PROPORTIONALITY * 8 / 240)
    assert m.threshold_for(176, 240) == pytest.approx(BIF_PROPORTIONALITY * 176 / 240)
    # A one-unit cohort in a large population falls back to the noise floor
    # rather than to a bar indistinguishable from floating-point noise.
    assert m.threshold_for(1, 100_000) == pytest.approx(EPSILON_BIF_REL_FLOOR)


def test_the_threshold_is_scale_free_over_the_run() -> None:
    """The same *relative* difference must give the same verdict at CE 8 and at
    CE 2. This is the property A-61's absolute form lacked."""
    m = _monitor()
    fired_hi = m.evaluate(_cohort(m, size=4, site=0), 8.0, 8.0 * 1.5, n=100, n_live=8)
    m2 = _monitor()
    fired_lo = m2.evaluate(_cohort(m2, size=4, site=0), 2.0, 2.0 * 1.5, n=100, n_live=8)
    assert fired_hi == fired_lo


# ---------------------------------------------------------------------------
# Reachability — the control that licenses reading a zero rate
# ---------------------------------------------------------------------------


def _cohort(monitor, *, size: int, site: int, born_at: int = 0, origin: str = "select"):
    monitor.register_cohort(site, list(range(size)), born_at, origin=origin)
    return monitor.pending[-1]


def test_a_disproportionate_cohort_fires_the_alarm() -> None:
    """The positive control. Without it, `alarm_rate = 0.0` is not evidence."""
    m = _monitor()
    c = _cohort(m, size=4, site=0)          # share 4/240 -> threshold 0.05
    fired = m.evaluate(c, loss_active=4.0, loss_masked=8.0, n=50, n_live=240)
    assert fired
    assert m.n_alarms == 1
    assert m.alarm_rate == pytest.approx(1.0)


def test_a_cohort_that_moves_the_probe_by_its_own_share_does_not_fire() -> None:
    """The negative control, and the *predicted* behaviour of a function-
    preserving birth: a cohort carrying `c/|U|` of the mass may move the probe
    by about that much without being a bifurcation."""
    m = _monitor()
    c = _cohort(m, size=4, site=0)
    share = 4 / 240
    fired = m.evaluate(c, 4.0, 4.0 * (1.0 + share), n=50, n_live=240)
    assert not fired
    assert m.alarm_rate == 0.0


def test_an_exactly_function_preserving_cohort_never_fires() -> None:
    """Theorem 3.1a: a birth changes nothing at all, so `bif = 0`."""
    m = _monitor()
    c = _cohort(m, size=8, site=0)
    assert not m.evaluate(c, 3.5, 3.5, n=50, n_live=64)


# ---------------------------------------------------------------------------
# The rate, and what an alarm is allowed to change
# ---------------------------------------------------------------------------


def test_the_rate_denominator_is_cohorts_not_units_born() -> None:
    """Audit A-60: (1.21) is evaluated **once per cohort**, so one alarm on a
    176-unit cohort is one event and not `1/176` of one. Dividing by units born
    made a large cohort's alarm arithmetically unable to cross a 1% bar."""
    m = _monitor()
    big = _cohort(m, size=176, site=0)
    m.evaluate(big, 4.0, 40.0, n=50, n_live=240)
    assert m.n_births == 176
    assert m.alarm_rate == pytest.approx(1.0), "per cohort"
    assert m.births_alarm_rate == pytest.approx(1 / 176), "the v3.0 statistic"


def test_growth_cohorts_are_counted_separately_and_never_halve_rho() -> None:
    """§03.7 asks whether **births behave as a controlled probe**. A scheduled
    capacity change is not a probe — it is meant to alter the function — so
    letting it halve `rho` would make the growth schedule silence the very
    mechanism it is scheduled alongside (audit A-60)."""
    m = _monitor()
    g = _cohort(m, size=176, site=0, origin="grow")
    fired = m.evaluate(g, 4.0, 40.0, n=50, n_live=240)
    assert fired
    assert m.rho_scale == 1.0, "a growth alarm must not slow structural change"
    assert m.alarm_rate == 0.0, "growth is excluded from the falsifier"
    assert m.growth_alarm_rate == pytest.approx(1.0)


def test_a_selection_alarm_halves_rho_each_time() -> None:
    m = _monitor()
    for i in range(3):
        c = _cohort(m, size=4, site=0, born_at=i)
        assert m.evaluate(c, 4.0, 40.0, n=50 + i, n_live=240)
    assert m.rho_scale == pytest.approx(0.125)


def test_the_two_rates_do_not_contaminate_each_other() -> None:
    m = _monitor()
    m.evaluate(_cohort(m, size=4, site=0), 4.0, 40.0, n=1, n_live=240)          # fires
    m.evaluate(_cohort(m, size=4, site=0, born_at=1), 4.0, 4.001, n=2, n_live=240)
    m.evaluate(_cohort(m, size=8, site=0, born_at=2, origin="grow"),
               4.0, 4.001, n=3, n_live=240)
    assert m.n_cohorts == {"select": 2, "grow": 1}
    assert m.alarm_rate == pytest.approx(0.5)
    assert m.growth_alarm_rate == 0.0


# ---------------------------------------------------------------------------
# When a cohort is evaluated
# ---------------------------------------------------------------------------


def test_a_cohort_is_evaluated_after_the_anneal_not_at_birth() -> None:
    """Audit A-36: measuring **at** the birth gives a difference Theorem 3.1a
    makes identically zero, so the alarm was unreachable and `rho <- rho/2` was
    dead code."""
    m = _monitor(phi_ann=50)
    m.register_cohort(0, [1, 2, 3], 100)
    assert m.due(100) == [], "a cohort due at birth would be unreachable"
    assert m.due(149) == []
    ready = m.due(150)
    assert len(ready) == 1 and ready[0].born_at == 100 and ready[0].due_at == 150
    assert m.due(150) == [], "a cohort is evaluated once"


def test_registering_an_empty_cohort_is_a_no_op() -> None:
    m = _monitor()
    m.register_cohort(0, [], 10)
    assert m.pending == [] and m.n_births == 0 and m.n_cohorts == {}


def test_every_evaluation_is_recorded_with_its_threshold() -> None:
    """The history is the audit trail: a verdict without the bar it was judged
    against cannot be re-read later."""
    m = _monitor()
    m.evaluate(_cohort(m, size=4, site=2), 4.0, 4.5, n=77, n_live=240)
    assert len(m.history) == 1
    rec = m.history[0]
    for k in ("bif_relative", "threshold", "share", "noise_floor"):
        assert k in rec, sorted(rec)
    assert rec["share"] == pytest.approx(4 / 240)
    assert rec["threshold"] == pytest.approx(BIF_PROPORTIONALITY * 4 / 240)
