"""The generation curve: the decisive experiment's own analysis.

SPEC: 07 §7.5
SPEC: 10 §10.1

Gate `G-GENERATIONS`. `soma/generations.py` had **0% test coverage** — 114
statements, including `run_generations`, the A4 control arm, `analyse_curves`,
`K1Verdict` and `budget_report`: the module that runs and adjudicates the
project's central experiment (audit A-82). That is why the defect below stood.

K1 was read off the `main` arm's **absolute** curve while the module's own
docstring says the absolute curves are confounded by ordinary recipe drift and
are not interpretable alone. Two arms improving identically — an inheritance
effect of exactly zero — reported success, and the A4 arm, half the M3 GPU
budget, never entered the decision.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from soma.config import M_NANO
from soma.generations import (
    GenerationCurve,
    GenerationRecord,
    analyse_arm_difference,
    analyse_curves,
    budget_report,
)


def _curve(arm: str, seed: int, drift: float, G: int, rng, censor=False):
    c = GenerationCurve(seed, arm)
    for g in range(G):
        v = drift * math.log1p(g) + rng.normal(0.0, 0.02)
        c.records.append(GenerationRecord(
            seed=seed, g=g, criterion=v, slope_score=v, early_slope=-v,
            birth=0.0, fitness=[v], censored=[censor], all_censored=censor,
            wall_seconds=1.0, n_c_frac=0.2 - 0.002 * g, rho_0=0.1,
        ))
    return c


def _arm(arm, drift, seeds=(1, 2, 3, 4, 5), G=20, seed=0, censor=False):
    rng = np.random.default_rng(seed)
    return [_curve(arm, s, drift, G, rng, censor) for s in seeds]


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


def test_k1_fires_when_both_arms_drift_identically() -> None:
    """Audit A-82. Identical drift in both arms **is** an inheritance effect of
    zero, so K1 must fire. Reading the `main` arm alone reports success."""
    main = _arm("main", 0.30, seed=0)
    a4 = _arm("a4", 0.30, seed=1)

    # The v3.0 reading, kept here so the difference between the two is visible
    # rather than asserted: each arm's own curve rises significantly.
    per_arm = analyse_curves(main, rng=np.random.default_rng(1))
    assert per_arm.b_criterion_significant
    assert not per_arm.fires, "the per-arm reading reports success"

    d = analyse_arm_difference(main, a4, rng=np.random.default_rng(1))
    assert abs(d.b_criterion) < 0.05, d.b_criterion
    assert not d.b_criterion_significant
    assert d.fires, "K1 must fire on a zero inheritance effect"
    assert d.detail["arms_differenced"] is True


def test_k1_does_not_fire_when_inheritance_actually_helps() -> None:
    """The other side: a main arm that improves faster than its control."""
    main = _arm("main", 0.45, seed=3)
    a4 = _arm("a4", 0.30, seed=4)
    d = analyse_arm_difference(main, a4, rng=np.random.default_rng(1))
    assert d.b_criterion > 0.05, d.b_criterion
    assert d.b_criterion_significant
    assert not d.fires


def test_the_difference_is_paired_by_seed() -> None:
    """The two arms share seeds by construction, so the seed-level difference
    removes the seed's own recipe draw — which is the confound the control was
    introduced to remove. A seed in one arm only is dropped, and recorded."""
    main = _arm("main", 0.30, seeds=(1, 2, 3, 4, 5, 6), seed=0)
    a4 = _arm("a4", 0.30, seeds=(1, 2, 3, 4, 5, 9), seed=1)
    d = analyse_arm_difference(main, a4, rng=np.random.default_rng(1))
    assert d.detail["n_seeds"] == 5
    assert d.detail["seeds_dropped"] == [6, 9]


def test_too_few_shared_seeds_is_refused_not_raised() -> None:
    """§07.5 requires `n_seed >= 5`, and the BCa routine raises below it. An
    analysis that cannot be computed must report that — a `ValueError` at the
    end of a 46-hour run is not a verdict."""
    main = _arm("main", 0.30, seeds=(1, 2, 3), seed=0)
    a4 = _arm("a4", 0.30, seeds=(1, 2, 3), seed=1)
    d = analyse_arm_difference(main, a4, rng=np.random.default_rng(1))
    assert not d.interpretable and not d.fires
    assert d.detail["n_seeds"] == 3
    assert "n_seed >= 5" in d.detail["why"]


def test_a_heavily_censored_difference_is_not_interpretable() -> None:
    """§07.5's censoring rule survives the differencing: a verdict computed from
    a degenerate statistic is not a verdict."""
    main = _arm("main", 0.30, seed=0, censor=True)
    a4 = _arm("a4", 0.30, seed=1, censor=True)
    d = analyse_arm_difference(main, a4, rng=np.random.default_rng(1))
    assert d.censored_fraction > 0.5
    assert not d.interpretable


def test_k1_in_killcriteria_refuses_an_undifferenced_verdict() -> None:
    """`soma.eval.killcriteria.k1_generation_curve` must not accept a single
    arm's analysis; the confound is the whole point of the control."""
    from soma.eval.killcriteria import k1_generation_curve

    single = analyse_curves(_arm("main", 0.30), rng=np.random.default_rng(1))
    v = k1_generation_curve(single.as_dict())
    assert not v.applicable, v.reason
    assert "difference" in v.reason.lower()

    paired = analyse_arm_difference(
        _arm("main", 0.30, seed=0), _arm("a4", 0.30, seed=1),
        rng=np.random.default_rng(1),
    )
    v2 = k1_generation_curve(paired.as_dict())
    assert v2.applicable and v2.fires


# ---------------------------------------------------------------------------
# The curve's own accessors
# ---------------------------------------------------------------------------


def test_the_curve_exposes_every_series_the_analysis_needs() -> None:
    c = _curve("main", 1, 0.3, 6, np.random.default_rng(0))
    assert [g for g, _ in c.criterion_points()] == list(range(6))
    assert [g for g, _ in c.slope_points()] == list(range(6))
    assert [g for g, _ in c.birth_points()] == list(range(6))
    # §09.4's secondary confirmation: `n_c` should fall across generations.
    can = c.canalization_points()
    assert [g for g, _ in can] == list(range(6))
    assert can[-1][1] < can[0][1]
    assert c.censored_fraction == 0.0


def test_a_fully_censored_curve_reports_it() -> None:
    c = _curve("main", 0.3, 0.3, 4, np.random.default_rng(0), censor=True)
    assert c.censored_fraction == 1.0


def test_the_record_serialises_every_field_a_report_reads() -> None:
    c = _curve("main", 1, 0.3, 2, np.random.default_rng(0))
    d = c.records[0].as_dict()
    for k in ("seed", "g", "criterion", "slope_score", "early_slope", "birth",
              "fitness", "censored", "all_censored", "wall_seconds",
              "n_c_frac", "rho_0"):
        assert k in d, sorted(d)


# ---------------------------------------------------------------------------
# The budget, which decides whether the experiment is affordable at all
# ---------------------------------------------------------------------------


def test_budget_report_scales_with_both_arms_and_the_es_population() -> None:
    """§07.4.1: the M3 budget is `arms x n_seed x G_gen x P_es x T_life`, and it
    is recomputed from the **measured** `T_life` because §01.9 leaves it
    `[OPEN]`."""
    r = budget_report(6.93, M_NANO, n_seed=5, n_generations=20, P_es=2)
    assert r["arms"] == 2
    assert r["total_minutes"] == pytest.approx(2 * 5 * 20 * 2 * 6.93)
    assert r["total_hours"] == pytest.approx(r["total_minutes"] / 60)
    assert not r["within_12h"], "46 h must not report as within a 12 h run"

    ok = budget_report(1.5, M_NANO, n_seed=5, n_generations=20, P_es=2)
    assert ok["within_12h"]


def test_the_budget_note_names_the_constraint_that_may_not_be_reduced() -> None:
    """§07.5 fixes `n_seed >= 5`: it is what makes `alpha_sig = 0.05` attainable
    at all, so the lever is `G_gen` or `N_step`."""
    r = budget_report(6.93, M_NANO, n_seed=5, n_generations=20)
    assert "n_seed >= 5" in r["note"]
