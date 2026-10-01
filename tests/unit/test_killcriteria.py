"""K1 … K10 as mechanical decisions.

SPEC: 10 §10.1
SPEC: 08 §8.5

Gate `G-KILL`. Spec v1's kill criteria were all unfireable — "unchanged on every
metric", with no equivalence test and a conjunction over dozens of fields — and
§10.1 replaced each with a small pre-registered metric set and a mechanical
decision. v3.0 then implemented **two** of the ten (K1 inside
`soma/generations.py`, K8 inline in `scripts/eval_tinystories.py`) and left the
other eight as prose, while `scripts/spec_trace.py` exempted §10.1 by naming a
module that did not exist (audit A-78).

Every criterion is tested **both ways**: an input that must fire it, and one that
must not. A kill criterion that can only be shown not to fire is the defect §10.1
was written to remove.
"""

from __future__ import annotations

import pytest

from soma.eval.killcriteria import (
    DELTA_NOVELTY,
    DELTA_Q_ROUTE,
    MIXER_TOL,
    evaluate_all,
    k1_generation_curve,
    k2_development,
    k3_case_selection,
    k4_connection_cost,
    k5_open_endedness,
    k6_mixer_oracle,
    k7_state_tracking,
    k8_plasticity,
    k9_ng_scaling,
    k10_unit_identity,
)


def _seeds(v, n=5, spread=0.001):
    return [v + spread * (i - n / 2) / n for i in range(n)]


# ---------------------------------------------------------------------------
# The distinction the whole module rests on
# ---------------------------------------------------------------------------


def test_absent_inputs_are_not_evaluated_rather_than_not_fired() -> None:
    """"K3 did not fire" and "K3 was never evaluated" must not be the same line
    in a report. Nothing defaults to a decision."""
    v = evaluate_all()
    assert v.fired == []
    assert len(v.not_evaluated) == 10
    for verdict in v.verdicts:
        assert verdict.applicable is False
        assert verdict.reason, f"{verdict.key} gives no reason"


def test_k6_and_k9_are_checked_first_and_are_named_blocking() -> None:
    """§10.1: both invalidate everything downstream, so a report cannot bury a
    mixer mismatch under nine other rows."""
    v = evaluate_all(mixer_max_rel_error=1e-3, mixer_engineer_days=4.0)
    assert [x.key for x in v.verdicts][:2] == ["K6", "K9"]
    assert v.blocking == ["K6"]


# ---------------------------------------------------------------------------
# K6, K9
# ---------------------------------------------------------------------------


def test_k6_fires_only_after_the_three_engineer_day_window() -> None:
    """The window is part of the condition: a mismatch fixed inside three days
    is a bug, not a kill."""
    assert not k6_mixer_oracle(1e-3, engineer_days_spent=1.0).fires
    assert k6_mixer_oracle(1e-3, engineer_days_spent=3.0).fires
    assert not k6_mixer_oracle(MIXER_TOL, engineer_days_spent=99.0).fires
    assert not k6_mixer_oracle(1e-9, engineer_days_spent=99.0).fires


def test_k9_reads_the_sweep_and_flags_an_argmin_at_the_grid_edge() -> None:
    fires = k9_ng_scaling({"log2_slope_b": 0.95, "K9_fires": True,
                           "argmin_at_grid_edge": [], "argmin_n_g": {"96": 1024}})
    assert fires.fires
    quiet = k9_ng_scaling({"log2_slope_b": 0.05, "K9_fires": False,
                           "argmin_at_grid_edge": [384], "argmin_n_g": {}})
    assert not quiet.fires
    assert "grid edge" in quiet.reason, quiet.reason


# ---------------------------------------------------------------------------
# K1
# ---------------------------------------------------------------------------


def _k1(crit_sig, slope_sig, censored=0.1, seeds=5):
    return {"b_criterion": 0.1, "b_criterion_significant": crit_sig,
            "b_slope": 0.1, "b_slope_significant": slope_sig,
            "interpretable": censored <= 0.5, "censored_fraction": censored,
            "n_seeds": seeds,
            # K1 is decided on `main - a4` (audit A-82); a verdict without this
            # flag came from one arm's absolute curve and is refused.
            "arms_differenced": True}


def test_k1_refuses_a_single_arm_verdict() -> None:
    """Audit A-82: an arm's absolute curve is confounded by recipe drift."""
    d = _k1(False, False)
    del d["arms_differenced"]
    v = k1_generation_curve(d)
    assert not v.applicable and "difference" in v.reason.lower()


def test_k1_is_a_conjunction_of_two_failures() -> None:
    """§07.5 read "both", §10 read "either", and the two decide the project's
    central verdict differently."""
    assert k1_generation_curve(_k1(False, False)).fires
    assert not k1_generation_curve(_k1(True, False)).fires
    assert not k1_generation_curve(_k1(False, True)).fires
    assert not k1_generation_curve(_k1(True, True)).fires


def test_k1_refuses_an_uninterpretable_curve_rather_than_firing() -> None:
    """§07.5's censoring rule: a verdict from a degenerate statistic is not a
    verdict, and "uninterpretable" must not read as "K1 fired"."""
    v = k1_generation_curve(_k1(False, False, censored=0.8))
    assert not v.applicable and not v.fires
    assert "censored" in v.reason


def test_k1_refuses_fewer_than_five_seeds() -> None:
    v = k1_generation_curve(_k1(False, False, seeds=3))
    assert not v.applicable and "n_seed" in v.reason


# ---------------------------------------------------------------------------
# K2, K3, K4, K10 — equivalence, not failure-to-reject
# ---------------------------------------------------------------------------


def test_k2_fires_on_equivalence_and_not_on_a_wide_inconclusive_sample() -> None:
    """§08.5: "unchanged" means both one-sided tests reject. A noisy sample that
    merely fails to show a difference is **inconclusive**, and must not fire."""
    tight = k2_development(_seeds(2.0), _seeds(2.0005), noise_floor_ce=0.01)
    assert tight.fires

    noisy_a = [2.0, 2.4, 1.6, 2.5, 1.5]
    noisy_b = [2.1, 1.7, 2.4, 1.6, 2.4]
    loose = k2_development(noisy_a, noisy_b, noise_floor_ce=0.01)
    assert not loose.fires
    assert loose.detail["held_out_ce"]["verdict"] in ("inconclusive", "different")


def test_k2_needs_a_noise_floor_and_says_so() -> None:
    v = k2_development(_seeds(2.0), _seeds(2.0), noise_floor_ce=None)
    assert not v.applicable and "noise_floor" in v.reason


def test_k3_requires_all_three_primary_metrics_to_be_equivalent() -> None:
    same = {"held_out_ce": (_seeds(2.0), _seeds(2.0)),
            "Q_route": (_seeds(0.3), _seeds(0.3)),
            "case_win_ks": (_seeds(0.1), _seeds(0.1))}
    assert k3_case_selection(same, noise_floor_ce=0.01, noise_floor_ks=0.01).fires

    one_differs = dict(same)
    one_differs["Q_route"] = (_seeds(0.30), _seeds(0.60))
    v = k3_case_selection(one_differs, noise_floor_ce=0.01, noise_floor_ks=0.01)
    assert not v.fires
    assert v.detail["Q_route"]["verdict"] != "equivalent"


def test_k4_uses_the_absolute_margin_specified_for_q_route() -> None:
    assert DELTA_Q_ROUTE == 0.02
    assert k4_connection_cost(_seeds(0.30), _seeds(0.3005)).fires
    assert not k4_connection_cost(_seeds(0.30), _seeds(0.40)).fires


def test_k10_is_conditional_on_k4_having_fired() -> None:
    """B3 equivalence alone says routing carries no information *given* a
    connection cost that may itself be inert. K10 asserts both."""
    metrics = {"held_out_ce": (_seeds(2.0), _seeds(2.0)),
               "Q_route": (_seeds(0.3), _seeds(0.3))}
    k4_fired = k4_connection_cost(_seeds(0.30), _seeds(0.3005))
    k4_quiet = k4_connection_cost(_seeds(0.30), _seeds(0.40))
    assert k4_fired.fires and not k4_quiet.fires

    assert k10_unit_identity(metrics, noise_floor_ce=0.01, k4=k4_fired).fires
    assert not k10_unit_identity(metrics, noise_floor_ce=0.01, k4=k4_quiet).fires

    absent = k10_unit_identity(metrics, noise_floor_ce=0.01, k4=None)
    assert not absent.applicable


# ---------------------------------------------------------------------------
# K5, K7, K8
# ---------------------------------------------------------------------------


def test_k5_is_a_disjunction_over_the_two_control_arms() -> None:
    assert DELTA_NOVELTY == 0.34
    ramp_zero = k5_open_endedness(live_minus_ramp=_seeds(0.0, spread=0.01),
                                  live_minus_frozen=[0.9] * 5)
    assert ramp_zero.fires and "live_minus_ramp" in ramp_zero.reason

    both_big = k5_open_endedness(live_minus_ramp=[0.9] * 5,
                                 live_minus_frozen=[0.9] * 5)
    assert not both_big.fires


def test_k7_needs_a_flat_curve_not_merely_a_low_point() -> None:
    """A low point with a *falling* curve is a length-generalisation limit; a low
    point with a *flat* curve is the §05.2.2 claim failing. Only the second is a
    kill, and §08.2 evaluates the curve for exactly that reason."""
    flat = k7_state_tracking(0.40, {40: 0.41, 64: 0.40, 96: 0.40, 256: 0.40})
    assert flat.fires

    falling = k7_state_tracking(0.40, {40: 0.99, 64: 0.90, 96: 0.70, 256: 0.40})
    assert not falling.fires and "not flat" in falling.reason

    passing = k7_state_tracking(0.97, {40: 0.99, 64: 0.98, 96: 0.98, 256: 0.97})
    assert not passing.fires


def test_k8_needs_both_significance_and_a_material_slope() -> None:
    """"Does not decline monotonically" — spec v1's wording — is satisfied
    trivially by any noisy statistic."""
    steep = k8_plasticity([10.0 - 0.02 * i for i in range(20)])
    assert steep.fires

    tiny = k8_plasticity([10.0 - 1e-7 * i for i in range(20)])
    assert not tiny.fires, tiny.detail

    noisy = k8_plasticity([10.0, 9.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1])
    assert not noisy.fires

    assert not k8_plasticity([1.0, 2.0]).applicable
