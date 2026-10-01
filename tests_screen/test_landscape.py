"""Tests for the landscape topology module.

The module's job is to say whether a population is needed. A bug here would not produce a
crash; it would produce a confident verdict about the central question of the project. So
every test plants a landscape whose answer is known and checks the module recovers it —
and, following `qd/mapelites.py`'s own rule that *"a control that has never been shown to
fire in both directions is not a control"*, the suite plants smooth landscapes and rugged
ones and requires the right answer to each.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from screen.landscape import Landscape


def _noisy(f: float, n: int = 5, sd: float = 0.01, rng=None) -> list[float]:
    rng = rng or np.random.default_rng(0)
    return [f + float(x) for x in rng.normal(0, sd, n)]


def _cube(fitness, *, loci=4, n=5, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return {g: _noisy(fitness(g), n, sd, rng)
            for g in itertools.product((0, 1), repeat=loci)}


# ------------------------------------------------------------------- smooth

def test_L1_additive_landscape_has_exactly_one_peak():
    """K=0 in Kauffman's terms: every locus independent, one globally attractive peak.

    This is the case where an evolutionary algorithm provably cannot help, and it is what
    the S1 axis table suggests the real body-plan landscape looks like.
    """
    w = [0.10, 0.05, 0.03, 0.02]
    land = Landscape(_cube(lambda g: -sum(wi * gi for wi, gi in zip(w, g))))
    rep = land.report()
    assert rep["n_local_optima"] == 1
    assert rep["sign_epistasis"]["reciprocal_sign_epistasis"] == 0
    assert rep["additive_r2"] > 0.95
    assert rep["funnel_ratio"] == 1.0
    assert land.global_optimum() == (1, 1, 1, 1)
    assert "SINGLE PEAK" in rep["verdict"]


def test_L2_diminishing_returns_still_has_one_peak():
    """Magnitude epistasis does not create local optima.

    The measured S1 non-additivity is exactly this shape — two delta layers deliver 78% of
    the four-layer benefit — and the point of the test is that 5.2 sigma of *real* epistasis
    is still consistent with a single peak. Concave and monotone: one optimum, reachable
    greedily.
    """
    land = Landscape(_cube(lambda g: -0.20 * math.sqrt(sum(g))))
    rep = land.report()
    assert rep["n_local_optima"] == 1
    assert rep["sign_epistasis"]["reciprocal_sign_epistasis"] == 0
    assert rep["additive_r2"] < 0.999          # genuinely non-additive
    assert land.global_optimum() == (1, 1, 1, 1)


# ------------------------------------------------------------------- rugged

def test_L3_planted_two_peak_landscape_is_found():
    """Reciprocal sign epistasis planted deliberately; both peaks must be recovered."""
    base = {g: 0.0 for g in itertools.product((0, 1), repeat=4)}
    base[(0, 0, 0, 0)] = -0.20
    base[(1, 1, 0, 0)] = -0.25
    base[(1, 0, 0, 0)] = +0.10
    base[(0, 1, 0, 0)] = +0.10
    land = Landscape(_cube(lambda g: base[g]))
    rep = land.report()
    assert rep["n_local_optima"] >= 2
    peaks = {tuple(p["genome"]) for p in rep["peaks"]}
    assert (0, 0, 0, 0) in peaks and (1, 1, 0, 0) in peaks
    assert rep["sign_epistasis"]["reciprocal_sign_epistasis"] >= 1
    assert land.global_optimum() == (1, 1, 0, 0)


def test_L4_saona_bound_holds_on_every_planted_landscape():
    """#RSE >= #peaks - 1 (Saona, Kondrashov & Khudiakova 2022, Theorem 1).

    A published inequality the module must not violate. If it ever does, either the RSE
    counter or the peak counter is wrong, and the test says which pair to look at.
    """
    rng = np.random.default_rng(3)
    for trial in range(12):
        vals = {g: float(rng.normal(0, 0.3))
                for g in itertools.product((0, 1), repeat=4)}
        land = Landscape(_cube(lambda g: vals[g], seed=trial))
        rep = land.report()
        rse = rep["sign_epistasis"]["reciprocal_sign_epistasis"]
        assert rse >= rep["n_local_optima"] - 1, (
            f"trial {trial}: {rse} RSE against {rep['n_local_optima']} peaks")


def test_L5_a_multi_peak_landscape_with_one_funnel_is_flagged_navigable():
    """Papkou et al. (Science 2023): 514 peaks, still easily navigable.

    Peak count alone must not gate anything. A landscape with a second, tiny-basin peak is
    solved by random-restart hill-climbing and the verdict has to say so.
    """
    def f(g):
        # 0000 beats its four single-mutant neighbours (-0.15 < -0.10), so it is a
        # genuine local optimum -- but it captures only itself, because every one of
        # those neighbours is pulled onward to 1111. One shallow peak, one deep funnel.
        s = sum(g)
        return {0: -0.15, 1: -0.10, 2: -0.20, 3: -0.30, 4: -0.60}[s]
    land = Landscape(_cube(f))
    rep = land.report()
    assert rep["n_local_optima"] >= 2
    assert rep["funnel_ratio"] >= 0.8
    assert "NAVIGABLE" in rep["verdict"]


# -------------------------------------------------------------------- noise

def test_L6_pure_noise_does_not_manufacture_peaks():
    """The margin exists so measurement error cannot become ruggedness."""
    land = Landscape(_cube(lambda g: 0.0, sd=0.01, seed=7))
    rep = land.report()
    assert rep["n_local_optima"] <= 1
    assert rep["sign_epistasis"]["reciprocal_sign_epistasis"] == 0


def test_L7_indistinguishable_squares_are_reported_not_resolved():
    """Ivankov & Zorin (2026): the RSE theorem assumes distinct fitnesses.

    Our regime has statistically indistinguishable genotypes by construction, so a square
    containing a pair inside the margin must be counted as `indistinguishable` and excluded
    from the epistasis tallies rather than resolved in whichever direction the noise fell.
    """
    land = Landscape(_cube(lambda g: 0.0, sd=0.001, seed=11))
    se = land.report()["sign_epistasis"]
    assert se["indistinguishable"] == se["squares_examined"]
    assert se["simple_sign_epistasis"] == 0
    assert se["reciprocal_sign_epistasis"] == 0


def test_L8_zero_resolvable_peaks_is_not_reported_as_a_single_peak():
    """'Unresolved' and 'single-peaked' are different claims and must read differently."""
    land = Landscape(_cube(lambda g: 0.0, sd=0.001, seed=13))
    rep = land.report()
    assert rep["n_local_optima"] == 0
    assert "UNRESOLVED" in rep["verdict"]
    assert "SINGLE PEAK" not in rep["verdict"]


# ------------------------------------------------------------ housekeeping

def test_L9_an_incomplete_landscape_is_rejected():
    """An exact local-optimum count needs every vertex; a sample cannot support one."""
    tab = _cube(lambda g: -sum(g) * 0.1)
    tab.pop((1, 1, 1, 1))
    with pytest.raises(ValueError, match="incomplete"):
        Landscape(tab)


def test_L10_the_margin_defaults_to_two_standard_errors():
    land = Landscape(_cube(lambda g: -0.1 * sum(g), n=5, sd=0.02, seed=5))
    s, _ = land.sigma_e()
    assert land.margin == pytest.approx(2.0 * s / math.sqrt(5), rel=1e-9)


def test_L11_lower_cross_entropy_is_better_everywhere():
    """One sign convention, asserted, because inverting it would invert the verdict."""
    land = Landscape(_cube(lambda g: -0.1 * sum(g)))
    assert land.global_optimum() == (1, 1, 1, 1)
    assert land.better((1, 1, 1, 1), (0, 0, 0, 0))
    assert not land.better((0, 0, 0, 0), (1, 1, 1, 1))


def test_L12_simulated_searchers_run_and_respect_their_budget():
    land = Landscape(_cube(lambda g: -0.1 * sum(g)))
    for nm in ("random", "rmhc", "restart_rmhc", "ga"):
        r = land.simulate(nm, budget=8, runs=200, seed=2)
        assert r.budget == 8
        assert 0.0 <= r.p_found_global <= 1.0
        assert math.isfinite(r.mean_best)


def test_L13_searchers_face_the_measured_noise_not_the_mean():
    """A searcher that saw noiseless fitness would be flattered.

    On a landscape whose arms are separated by less than the seed noise, no searcher should
    reliably identify the true optimum — if one does, it is being handed the means.
    """
    rng = np.random.default_rng(17)
    tab = {g: _noisy(-0.002 * sum(g), n=5, sd=0.05, rng=rng)
           for g in itertools.product((0, 1), repeat=4)}
    land = Landscape(tab)
    r = land.simulate("random", budget=8, runs=2000, seed=3)
    assert r.p_found_global < 0.6
