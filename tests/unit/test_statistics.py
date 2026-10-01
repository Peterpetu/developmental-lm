"""The decision procedures: Mann-Kendall, TOST, BCa, and the metrics.

SPEC: 06 §6.4
SPEC: 07 §7.5
SPEC: 08 §8.1
SPEC: 08 §8.5

Gates `G-MK`, `G-METRICS`.

Every test that decides something in this project runs through this module, so
each is checked against a value computed independently — a closed form, an
exhaustive enumeration, or a construction with a known answer.
"""

from __future__ import annotations

import math
from itertools import permutations

import numpy as np
import pytest

from soma.openended.certify import (
    J_RING,
    K_RING,
    MIN_USABLE,
    CertificationVerdict,
    capture_cadence,
    certify,
    ramp_schedule,
)
from soma.utils.stats import (
    bca_bootstrap_slope,
    benjamini_hochberg,
    effective_rank,
    ks_statistic,
    log_slope,
    mann_kendall,
    median_absolute_deviation,
    pass_at_k,
    scaled_accuracy,
    tost,
)


# ---------------------------------------------------------------------------
# G-MK
# ---------------------------------------------------------------------------


def test_mann_kendall_exact_null_matches_enumeration() -> None:
    """§06.4: **exact** null distribution for `n <= 10`.

    Checked against an exhaustive enumeration of permutations, which is the only
    way to know the Mahonian recursion is right.
    """
    n = 6
    xs = list(range(n))
    counts = {}
    for perm in permutations(xs):
        S = sum(
            np.sign(perm[j] - perm[i]) for i in range(n) for j in range(i + 1, n)
        )
        counts[int(S)] = counts.get(int(S), 0) + 1
    total = math.factorial(n)

    for S_obs in sorted(counts):
        exact_p = sum(v for k, v in counts.items() if k >= S_obs) / total
        # Build a sequence realising S_obs by picking a permutation with it.
        target = next(p for p in permutations(xs)
                      if sum(np.sign(p[j] - p[i])
                             for i in range(n) for j in range(i + 1, n)) == S_obs)
        res = mann_kendall(list(target), alternative="increasing")
        assert res.method == "exact"
        assert res.S == S_obs
        assert res.p_value == pytest.approx(exact_p, abs=1e-12)


def test_mann_kendall_strictly_increasing_is_maximally_significant() -> None:
    n = 8
    res = mann_kendall(list(range(n)), alternative="increasing")
    assert res.S == n * (n - 1) // 2
    assert res.p_value == pytest.approx(1.0 / math.factorial(n))
    assert res.tau == pytest.approx(1.0)
    assert res.slope == pytest.approx(1.0)


def test_mann_kendall_falls_back_to_normal_with_ties() -> None:
    """The exact null is the distribution over permutations of **distinct**
    values, so it is not the null when ties are present. Applying it anyway
    would understate the p-value."""
    res = mann_kendall([1, 1, 2, 2, 3, 3, 4, 4], alternative="increasing")
    assert res.method == "normal"
    assert 0.0 < res.p_value < 0.05


def test_mann_kendall_decreasing_alternative() -> None:
    res = mann_kendall(list(range(8))[::-1], alternative="decreasing")
    assert res.p_value == pytest.approx(1.0 / math.factorial(8))
    assert res.slope == pytest.approx(-1.0)


def test_mann_kendall_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        mann_kendall([1.0, 2.0])


def test_benjamini_hochberg_controls_the_step_up() -> None:
    """Thresholds are `k*alpha/m`; the largest `k` with `p_(k) <= t_k` sets the
    cut, and everything at or below it is rejected."""
    # m=5, alpha=0.05 -> thresholds 0.01, 0.02, 0.03, 0.04, 0.05.
    # p_(3)=0.039 > 0.03 and p_(4)=0.041 > 0.04, so the cut is at k=2.
    assert benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.9], 0.05).tolist() == [
        True, True, False, False, False
    ]
    # The step-up property: a LARGER p-value passing its own threshold pulls in
    # every smaller one, including ones that fail theirs. Here p_(3)=0.029 <=
    # 0.03, so k=3 is the cut and p_(2)=0.025 is rejected despite 0.025 > 0.02.
    assert benjamini_hochberg([0.001, 0.025, 0.029, 0.5, 0.9], 0.05).tolist() == [
        True, True, True, False, False
    ]
    assert benjamini_hochberg([], 0.05).size == 0
    assert not benjamini_hochberg([0.9, 0.8], 0.05).any()
    assert benjamini_hochberg([0.001] * 4, 0.05).all()


# ---------------------------------------------------------------------------
# Certification (§06.4)
# ---------------------------------------------------------------------------


def _matrix(J=J_RING, K=K_RING, N=12208, novelty=0.5, learnability=0.4, seed=0):
    t = np.array([N * 2.0 ** -j for j in range(J)])[::-1]
    T = np.linspace(1, N, K)
    rng = np.random.default_rng(seed)
    L = np.zeros((J, K))
    for j in range(J):
        for k in range(K):
            L[j, k] = (
                1.0
                + novelty * (T[k] / N)
                - learnability * (t[j] / N)
                + rng.normal(0, 0.005)
            )
    return L, t, T


def test_certification_detects_a_planted_trend() -> None:
    L, t, T = _matrix()
    res = certify(L, t, T, arm="live")
    assert res.novelty_fraction > 0.9
    assert res.learnability_fraction > 0.9


def test_certification_reports_zero_on_pure_noise() -> None:
    J, K, N = J_RING, K_RING, 12208
    t = np.array([N * 2.0 ** -j for j in range(J)])[::-1]
    T = np.linspace(1, N, K)
    rng = np.random.default_rng(1)
    L = rng.normal(0, 1, (J, K))
    res = certify(L, t, T)
    assert res.novelty_fraction < 0.35


def test_rows_with_fewer_than_six_usable_points_are_excluded_from_both() -> None:
    """§06.4: excluded from **numerator and denominator** of the reported
    fraction."""
    L, t, T = _matrix()
    res = certify(L, t, T)
    n_qualifying = sum(int((T > t[j]).sum()) >= MIN_USABLE for j in range(len(t)))
    assert res.n_rows_tested == n_qualifying


def test_the_qualifying_row_count_is_seven_not_three() -> None:
    """Audit A-51: v3.0 stated "at most 3 rows qualify", giving a granularity of
    `1/3`. The realised count is 7 and the granularity `1/7`."""
    L, t, T = _matrix()
    res = certify(L, t, T)
    assert res.n_rows_tested == 7
    v = CertificationVerdict(res, res, res, 0)
    assert v.granularity == pytest.approx(1.0 / 7.0)
    assert 0.34 > v.granularity, "the K5 margin must exceed the granularity"


def test_capture_cadence_spans_the_run() -> None:
    """§06.4 / audit A-23: v2 tied capture to `T_cert / K_ring`, which is
    self-contradictory."""
    from soma.config import M1, M_NANO

    for cfg in (M_NANO, M1):
        c = capture_cadence(cfg.N_step, K_RING)
        assert c * K_RING >= cfg.N_step - c
        assert cfg.realised_periodic_events(cfg.T_cert) >= 3


def test_ramp_schedule_is_monotone_and_bounded() -> None:
    """The ramp control arm: a scripted monotone ratchet on `n_len` alone."""
    N = 1000
    vals = [ramp_schedule(n, N) for n in range(0, N + 1, 50)]
    assert vals == sorted(vals)
    assert vals[0] == 4 and vals[-1] == 32


def test_verdict_reports_both_differences() -> None:
    L, t, T = _matrix(novelty=0.5)
    Lr, _, _ = _matrix(novelty=0.2, seed=2)
    Lf, _, _ = _matrix(novelty=0.0, seed=3)
    v = CertificationVerdict(
        certify(L, t, T, arm="live"),
        certify(Lf, t, T, arm="frozen"),
        certify(Lr, t, T, arm="ramp"),
        100,
    )
    assert v.live_minus_ramp >= 0.0
    assert v.live_minus_frozen >= 0.0
    d = v.as_dict()
    assert set(d) >= {"live", "frozen", "ramp", "live_minus_ramp", "granularity"}


# ---------------------------------------------------------------------------
# TOST (§08.5)
# ---------------------------------------------------------------------------


def test_tost_declares_equivalence_for_a_tight_paired_sample() -> None:
    a = [1.00, 1.01, 0.99, 1.02, 0.98]
    b = [1.00, 1.00, 1.01, 1.01, 0.99]
    r = tost(a, b, delta=0.05)
    assert r.equivalent and not r.different
    assert r.verdict == "equivalent"


def test_tost_declares_difference_for_a_clear_shift() -> None:
    a = [1.0, 1.1, 0.9, 1.05, 0.95]
    b = [2.0, 2.1, 1.9, 2.05, 1.95]
    r = tost(a, b, delta=0.05)
    assert r.different and not r.equivalent
    assert r.verdict == "different"


def test_tost_reports_inconclusive_as_a_third_outcome() -> None:
    """§08.5: "Neither rejecting is reported as **inconclusive**, which is an
    honest third outcome"."""
    rng = np.random.default_rng(3)
    a = list(rng.normal(0, 1, 5))
    b = list(rng.normal(0, 1, 5))
    r = tost(a, b, delta=0.01)
    assert r.verdict == "inconclusive"
    assert not r.equivalent and not r.different


def test_tost_refuses_a_zero_margin() -> None:
    """Audit A-12: v2's noise floor over identical seeds made every margin
    exactly zero, and the failure was silent."""
    with pytest.raises(ValueError, match="zero margin|must be positive"):
        tost([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], delta=0.0)


def test_tost_paired_requires_matched_shapes() -> None:
    with pytest.raises(ValueError, match="matched samples"):
        tost([1, 2, 3], [1, 2], delta=0.1)


# ---------------------------------------------------------------------------
# BCa (§07.5)
# ---------------------------------------------------------------------------


def test_bca_detects_a_planted_positive_slope() -> None:
    rng = np.random.default_rng(0)
    curves = [
        [(g, 0.1 * math.log1p(g) + rng.normal(0, 0.01)) for g in range(20)]
        for _ in range(5)
    ]
    r = bca_bootstrap_slope(curves, n_resamples=2000, rng=rng)
    assert r.estimate == pytest.approx(0.1, abs=0.02)
    assert r.significant and r.ci_low > 0


def test_bca_does_not_detect_a_flat_curve() -> None:
    rng = np.random.default_rng(1)
    curves = [
        [(g, rng.normal(0, 0.05)) for g in range(20)] for _ in range(5)
    ]
    r = bca_bootstrap_slope(curves, n_resamples=2000, rng=rng)
    assert not r.significant


def test_bca_requires_five_seeds() -> None:
    """§07.5: spec v1 specified a bootstrap over **3** seeds, which admits 27
    distinct resamples and a minimum achievable `p = 0.037`."""
    rng = np.random.default_rng(2)
    curves = [[(g, float(g)) for g in range(10)] for _ in range(3)]
    with pytest.raises(ValueError, match="n_seed >= 5"):
        bca_bootstrap_slope(curves, n_resamples=100, rng=rng)


def test_bca_resamples_seeds_not_lifetimes() -> None:
    """The resampling unit is the **seed**: lifetimes within a seed share a data
    order and an initialisation, so resampling them treats correlated
    observations as independent and understates the interval."""
    rng = np.random.default_rng(4)
    curves = [[(g, 0.05 * g + s * 0.5) for g in range(10)] for s in range(5)]
    r = bca_bootstrap_slope(curves, n_resamples=1000, rng=rng)
    assert r.n_units == 5


# ---------------------------------------------------------------------------
# G-METRICS
# ---------------------------------------------------------------------------


def test_pass_at_k_is_k_dependent() -> None:
    """(8.1). Spec v1 defined `pass@k` as the per-sample success rate — which is
    independent of `k`, so the curve is flat and §08.7 rule 1's whole argument
    is unobservable."""
    n, c = 128, 10
    curve = [pass_at_k(n, c, k) for k in (1, 2, 4, 8, 16, 32, 64, 128)]
    assert curve == sorted(curve)
    assert curve[0] == pytest.approx(c / n)
    assert curve[-1] == pytest.approx(1.0)
    assert pass_at_k(10, 0, 3) == 0.0
    assert pass_at_k(10, 10, 1) == pytest.approx(1.0)


def test_pass_at_k_matches_hypergeometric_enumeration() -> None:
    from math import comb

    n, c, k = 12, 5, 4
    expected = 1 - comb(n - c, k) / comb(n, k)
    assert pass_at_k(n, c, k) == pytest.approx(expected, rel=1e-12)


def test_pass_at_k_validates_its_arguments() -> None:
    with pytest.raises(ValueError):
        pass_at_k(10, 11, 1)
    with pytest.raises(ValueError):
        pass_at_k(10, 5, 0)


def test_scaled_accuracy_zeroes_chance() -> None:
    """§08.2: raw accuracy is never reported — for parity, chance is 0.5."""
    assert scaled_accuracy(0.5, 0.5) == 0.0
    assert scaled_accuracy(1.0, 0.5) == 1.0
    assert scaled_accuracy(0.75, 0.5) == pytest.approx(0.5)


def test_an_empty_evaluation_cannot_reach_the_criterion() -> None:
    """§07.1 / audit A-90: `held_out_ce` returned **0.0** when nothing was
    scored, and `CurveTracker.record` accepts any `ce <= L_star` — so one
    evaluation batch with no scorable natural token instantly un-censored the
    lifetime and set `n_crit` to that step.

    `CriterionScore` is the primary statistic of the decisive experiment, and a
    zero from an empty batch is the one value that cannot be a measurement.
    """
    from soma.config import M_NANO
    from soma.eval.metrics import CurveTracker

    c = CurveTracker(M_NANO, L_star=2.0)
    c.record(100, 1000, 0.0, n_scored=0)
    assert c.n_crit is None and c.censored, "an empty batch reached the criterion"
    c.record(200, 2000, float("nan"), n_scored=64)
    assert c.n_crit is None and c.censored, "a NaN reached the criterion"
    assert len(c.points) == 2, "the gap must still be recorded on the curve"

    c.record(300, 3000, 1.5, n_scored=64)
    assert c.n_crit == 300 and not c.censored


def test_effective_rank_matches_known_spectra() -> None:
    """(3.9)."""
    assert effective_rank([1, 1, 1, 1]) == pytest.approx(4.0)
    assert effective_rank([1, 0, 0, 0]) == pytest.approx(1.0)
    assert effective_rank([1.0, 1e-12, 1e-12]) == pytest.approx(1.0, abs=1e-6)
    assert 1 < effective_rank([4, 2, 1, 1]) < 4


def test_mad_returns_zero_on_the_empty_set() -> None:
    """(4.7): returning 0.0 rather than NaN is what makes the ε-lexicase
    precondition `eps_t >= 0` satisfiable."""
    assert median_absolute_deviation([]) == 0.0
    assert median_absolute_deviation([1, 1, 1]) == 0.0
    assert median_absolute_deviation([1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_ks_statistic_matches_a_hand_computation() -> None:
    assert ks_statistic([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 0.0
    assert ks_statistic([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]) == pytest.approx(0.4)


def test_log_slope_recovers_a_planted_slope() -> None:
    xs = [10 ** i for i in range(1, 6)]
    ys = [3.0 - 0.5 * math.log(x) for x in xs]
    assert log_slope(list(zip(xs, ys))) == pytest.approx(-0.5, abs=1e-9)
    assert log_slope([(0, 1.0), (0, 2.0)]) == 0.0


def test_bits_per_byte() -> None:
    from soma.eval.metrics import bits_per_byte

    assert bits_per_byte(math.log(2), 1, 1) == pytest.approx(1.0)
    assert bits_per_byte(math.log(2), 100, 400) == pytest.approx(0.25)
