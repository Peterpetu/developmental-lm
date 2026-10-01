"""§04.5.1's selection cost governor, and §01.7.1's `sep(n) <= 1` assertion.

SPEC: 04 §4.5.1
SPEC: 01 §1.7.1
SPEC: 07 §7.6

Gate `G-GOVERNOR`. Both mechanisms were fully specified and entirely absent from
v3.0 (audit A-75): `d_rate` was read in eight places and written in none, nothing
timed `select()`, and `sep(n)` was logged every `T_log` steps under a docstring
restating the requirement to assert it.

An unimplemented governor is invisible from every direction that matters — the
run completes, the log field it should have produced is simply not there, and
§07.8's profiling row that cites it has nothing to read.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from soma.config import M1, M_NANO
from soma.train import SEP_TOL, SelectionGovernor, _separation


def _gov(d_rate: float = 0.05) -> SelectionGovernor:
    return SelectionGovernor(replace(M_NANO.policy, d_rate=d_rate))


def _window(gov: SelectionGovernor, share: float, steps: int | None = None) -> object:
    """Run one full window at a given selection share. Returns the verdict."""
    steps = steps or gov.WINDOW
    out = None
    for _ in range(steps):
        gov.add_selection(share)
        out = gov.end_step(1.0)
    return out


def test_a_window_under_budget_does_not_touch_d_rate() -> None:
    gov = _gov(0.05)
    assert _window(gov, share=0.01) is None
    assert gov.policy.d_rate == pytest.approx(0.05)
    assert gov.n_reductions == 0
    assert gov.last_share == pytest.approx(0.01)


def test_a_window_over_budget_multiplies_d_rate_by_0_7() -> None:
    """§04.5.1: `d_rate <- max(0.02, 0.7 * d_rate)`."""
    gov = _gov(0.05)
    rec = _window(gov, share=0.05)
    assert rec is not None and not rec["at_floor"]
    assert rec["d_rate_before"] == pytest.approx(0.05)
    assert rec["d_rate_after"] == pytest.approx(0.035)
    assert gov.policy.d_rate == pytest.approx(0.035)
    assert gov.n_reductions == 1


def test_the_reduction_stops_at_the_floor_and_says_so() -> None:
    """A governor pinned at its floor while still over budget must report that,
    not look like a governor that never fired."""
    gov = _gov(0.05)
    for _ in range(20):
        rec = _window(gov, share=0.5)
    assert gov.policy.d_rate == pytest.approx(0.02)
    assert rec is not None and rec["at_floor"] is True
    assert gov.last_share == pytest.approx(0.5)


def test_the_boundary_is_exactly_two_percent() -> None:
    """`> 0.02` fires, `== 0.02` does not."""
    below = _gov()
    assert _window(below, share=0.02) is None
    above = _gov()
    assert _window(above, share=0.0200001) is not None


def test_the_window_is_a_thousand_steps_and_closes_on_step_count() -> None:
    """Closing on step count rather than elapsed time is what makes the decision
    reproducible: two runs with the same seed reduce `d_rate` at the same steps
    even on differently loaded hardware."""
    gov = _gov()
    assert gov.WINDOW == 1000
    for i in range(gov.WINDOW - 1):
        gov.add_selection(1.0)
        assert gov.end_step(1.0) is None, f"fired early at step {i}"
    gov.add_selection(1.0)
    assert gov.end_step(1.0) is not None


def test_the_share_is_a_ratio_of_accumulated_times_not_a_mean_of_ratios() -> None:
    """One expensive selection event in a window of cheap steps must count for
    what it cost. A mean of per-step ratios would dilute it by the window."""
    gov = _gov()
    gov.add_selection(30.0)          # one event costing 30 s
    for _ in range(gov.WINDOW - 1):
        gov.end_step(1.0)
    rec = gov.end_step(1.0)
    assert rec is not None
    assert gov.last_share == pytest.approx(30.0 / 1000.0)


def test_the_log_fields_07_6_requires_are_present() -> None:
    gov = _gov()
    d = gov.as_dict()
    for k in ("selection_share", "d_rate", "d_rate_reductions"):
        assert k in d, sorted(d)


def test_the_governor_writes_only_its_own_policy_copy() -> None:
    """§01.7.2: an in-lifetime policy write acts on the deep copy taken at step
    0, so it is not inherited. The shared module constant must be untouched."""
    before = M_NANO.policy.d_rate
    gov = _gov(before)
    _window(gov, share=0.9)
    assert gov.policy.d_rate < before
    assert M_NANO.policy.d_rate == pytest.approx(before)


# ---------------------------------------------------------------------------
# §01.7.1's separation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [M_NANO, M1], ids=["m-nano", "m1"])
def test_separation_holds_over_the_asserted_window(cfg) -> None:
    """The assertion the training loop now makes, checked over the whole window
    it applies to rather than at one point."""
    worst = max(
        _separation(n, cfg, cfg.policy)
        for n in range(cfg.n_warm, cfg.n_dec, max(1, (cfg.n_dec - cfg.n_warm) // 200))
    )
    assert worst <= 1.0 + SEP_TOL, f"worst sep = {worst:.4f} at {cfg.name}"


@pytest.mark.parametrize("cfg", [M_NANO, M1], ids=["m-nano", "m1"])
def test_separation_diverges_in_decay_which_is_why_the_window_excludes_it(cfg) -> None:
    """Audit A-20's other half: as `a_lr -> 0` the ratio diverges, so asserting
    it outside `[n_warm, n_dec)` would abort the final steps of every run. The
    exclusion is a fact about the schedule, and is checked as one."""
    late = _separation(cfg.N_step - 1, cfg, cfg.policy)
    assert late > 1.0, f"sep at the end of decay is {late:.4f}, not > 1"


def test_the_bound_is_structural_on_the_plateau_not_merely_observed() -> None:
    """Why no reachable configuration can violate it — stated, not assumed.

    On `[n_warm, n_dec)` the WSD schedule holds `lr = lr_max`, so (1.20) reduces
    to `sep(n) = (rho(n)/rho_0) * (T_0/T_sel(n))`. With (3.10)
    `rho(n) = rho_0 exp(-n/n_c)` and (3.11) `T_sel(n) = T_0 (1 + n/n_c)`, this is

        sep(n) = exp(-n/n_c) / (1 + n/n_c)  <=  1   for every n >= 0,

    with equality only at `n = 0`. Both factors are individually `<= 1`. The
    assertion in `LIFETIME` is therefore a guard against a **future change to
    (3.10) or (3.11)**, not a live check on the current schedules — and saying so
    is the difference between a guard and an assertion that looks like evidence.
    """
    import math

    for cfg in (M_NANO, M1):
        for n in range(cfg.n_warm, cfg.n_dec, max(1, (cfg.n_dec - cfg.n_warm) // 50)):
            closed = math.exp(-n / cfg.n_c) / (1.0 + n / cfg.n_c)
            assert _separation(n, cfg, cfg.policy) == pytest.approx(closed)
            assert closed <= 1.0


def test_a_violating_schedule_is_caught() -> None:
    """The positive control, against a deliberately broken schedule.

    Since no *reachable* configuration violates the bound (above), the control
    substitutes schedules that do: a structural rate that **grows** with `n`
    while the learning rate is flat. If this did not exceed 1, the assertion in
    `LIFETIME` would be unfireable machinery rather than a guard."""

    class _BrokenSchedule:
        """(3.11) inverted, and canalization switched off.

        Both are needed. `_separation` applies (3.10)'s `exp(-n/n_c)` itself, so
        inverting `T_sel` alone gives `sep(n) = (1 + n/n_c) exp(-n/n_c)`, whose
        maximum over `n >= 0` is exactly 1 — a near miss, and a useful reminder
        that (1.20) is bounded by more than one of its factors. Taking `n_c`
        large removes the decay and leaves the inversion exposed."""

        name = "broken"
        T_0 = M_NANO.T_0
        n_c = 1e12                       # canalization effectively off
        n_warm = M_NANO.n_warm
        n_dec = M_NANO.n_dec
        N_step = M_NANO.N_step
        lr_max = M_NANO.lr_max

        def T_sel(self, n: int) -> float:
            return max(1.0, self.T_0 / (1.0 + n / 1000.0))

    class _NoCanalization:
        """(3.10) with no decay: `rho(n) = rho_0`."""

        rho_0 = M_NANO.policy.rho_0

    broken = _BrokenSchedule()
    # `_separation` reads `policy.rho_0` and applies (3.10)'s exponential itself,
    # so the violation is driven entirely by `T_sel`.
    worst = max(
        _separation(n, broken, _NoCanalization())
        for n in range(broken.n_warm, broken.n_dec, 5)
    )
    assert worst > 1.0 + SEP_TOL, (
        f"worst sep = {worst:.4f} on a deliberately inverted schedule; the "
        f"assertion in LIFETIME cannot fire and is not a guard"
    )
