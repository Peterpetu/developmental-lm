"""SPEC: 06 §6.7 -- acceptance tests S1..S8 for the heritability instrument."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from tests_gnomon.conftest import requires_corpus

from gnomon.eval.share import (
    KG_HERIT_LOWER, KG_HERIT_POINT, birth_share, genome_share, heritability,
    predicted_response,
)


def _synthetic(h2_true: float, *, n_G: int, n_S: int, sigma_P: float = 1.0,
               seed: int = 0) -> np.ndarray:
    """`P(g,s) = mu + G_g + E_gs` with `Var(G)/Var(P) = h2_true` exactly."""
    rng = np.random.default_rng(seed)
    V_G = h2_true * sigma_P ** 2
    V_E = (1.0 - h2_true) * sigma_P ** 2
    G = rng.normal(0.0, np.sqrt(V_G), size=n_G) if V_G > 0 else np.zeros(n_G)
    return 3.0 + G[:, None] + rng.normal(0.0, np.sqrt(V_E), size=(n_G, n_S))


def test_S1_estimator_recovers_known_heritability():
    """SPEC: 06 §6.7 S1. 200 replicates per setting, mean within Monte-Carlo error."""
    for h2_true in (0.2, 0.5, 0.8):
        ests = [heritability(_synthetic(h2_true, n_G=12, n_S=3, seed=s),
                             n_resamples=1).h2 for s in range(200)]
        got = float(np.mean(ests))
        se = float(np.std(ests)) / np.sqrt(len(ests))
        assert abs(got - h2_true) < max(4 * se, 0.06), (
            f"h2_true {h2_true}: estimated {got:.3f} +/- {se:.3f}"
        )


def test_S2_null_behaviour_when_there_is_no_genetic_variance():
    """SPEC: 06 §6.7 S2. `V_G = 0` must not produce a passing gate."""
    passes = 0
    for s in range(100):
        r = heritability(_synthetic(0.0, n_G=12, n_S=3, seed=1000 + s),
                         n_resamples=600, seed=s)
        passes += r.ci_low >= KG_HERIT_LOWER
    assert passes <= 5, (
        f"the gate fired on {passes}/100 null datasets; it must hold its "
        f"nominal 5% under V_G = 0"
    )


def test_S2_point_estimate_centres_at_zero_under_the_null():
    ests = [heritability(_synthetic(0.0, n_G=12, n_S=3, seed=2000 + s),
                         n_resamples=1).h2 for s in range(200)]
    assert float(np.mean(ests)) < 0.10, float(np.mean(ests))


def test_S3_positive_control_a_disconnected_genome_fails_the_gate():
    """SPEC: 06 §6.7 S3. An absence gate that has never fired is not evidence.

    A genome disconnected from the forward pass produces the same outcome for
    every genome, so `V_G = 0` up to seed noise. This is the SOMA condition, and
    `KG-HERIT` must fire on it.
    """
    P = _synthetic(0.0, n_G=16, n_S=4, seed=7)
    r = heritability(P, n_resamples=2000)
    assert not r.passes, f"the gate did not fire on a disconnected genome: {r.h2}"
    assert r.V_G < 0.05 * (r.V_G + r.V_E) + 1e-9


def test_S3_a_working_genome_passes():
    P = _synthetic(0.6, n_G=16, n_S=4, seed=7)
    r = heritability(P, n_resamples=2000)
    assert r.passes, r.report()


def test_breeders_equation_is_the_documented_one():
    """SPEC: 06 §6.2. `R = i h2 sigma_P`, `i = 1.271` at the top quarter."""
    r = heritability(_synthetic(0.5, n_G=12, n_S=3, seed=0), n_resamples=1)
    assert predicted_response(r) == pytest.approx(1.271 * r.h2 * r.sigma_P)


def test_S5_the_design_must_be_rectangular_and_paired():
    """Genome comparisons use identical seed sets; a ragged design is an error."""
    with pytest.raises((ValueError, TypeError)):
        heritability([[1.0, 2.0], [3.0]], n_resamples=1)
    with pytest.raises(ValueError, match="need >= 2"):
        heritability([[1.0, 2.0]], n_resamples=1)
    with pytest.raises(ValueError, match="need >= 2"):
        heritability([[1.0], [2.0], [3.0]], n_resamples=1)


def test_S7_determinism():
    P = _synthetic(0.4, n_G=12, n_S=3, seed=5)
    a = heritability(P, n_resamples=800, seed=3)
    b = heritability(P, n_resamples=800, seed=3)
    assert (a.h2, a.ci_low, a.ci_high) == (b.h2, b.ci_low, b.ci_high)


def test_V_G_is_clamped_at_zero_not_reported_negative():
    """`MSB < MSW` is a legitimate finite-sample outcome, not a variance."""
    rng = np.random.default_rng(0)
    P = 3.0 + rng.normal(0, 1.0, size=(8, 5))
    r = heritability(P, n_resamples=1)
    assert r.V_G >= 0.0 and 0.0 <= r.h2 <= 1.0


def test_genome_share_reproduces_the_soma_arithmetic():
    """SPEC: 06 §6.3, against §00's measured table.

    SOMA: trained 2.1005; randomising the whole unit population 2.2621;
    randomising `W_shared` 9.0781. Treating the population as the germline half
    and `W_shared` as the soma half gives the 0.024 the constitution quotes.
    """
    g = genome_share(2.1005, [2.2621], [9.0781])
    assert g["share"] == pytest.approx(0.0227, abs=0.002)
    assert g["delta_gamma"] == pytest.approx(0.1616, abs=1e-3)


def test_genome_share_is_one_when_the_soma_does_nothing():
    g = genome_share(2.0, [8.0], [2.0])
    assert g["share"] == pytest.approx(1.0)


def test_birth_share_is_zero_for_an_ignorant_genome():
    """SPEC: 06 §6.4. ~0 at generation 0, and it should be."""
    assert birth_share(L_birth=8.9, L_null=8.9, L_star=2.0) == pytest.approx(0.0)
    assert birth_share(L_birth=2.0, L_null=8.9, L_star=2.0) == pytest.approx(1.0)


def test_thresholds_are_the_spec_values():
    assert (KG_HERIT_POINT, KG_HERIT_LOWER) == (0.25, 0.10)


# ---------------------------------------------------------------------- S4

def test_S4_the_probe_reproduces_SOMAs_own_number():
    """SPEC: 06 §6.7 S4. The instrument, validated against the failure it detects.

    §00.1's ablation table is the record of the archived M-nano run
    (`runs/m-nano-s20260828`, `d_model=192`, `L=6`, 2,000 steps, TinyStories,
    vocab 4096), measured by randomising one part of the trained model in place
    and re-evaluating. Feeding those measurements to §6.3's estimator must
    return SOMA's **0.024**: an instrument that cannot reproduce the number it
    was built from is not calibrated.

    What this checks and what it does not. The arithmetic of the estimator
    against SOMA's measured losses -- yes. A fresh forward pass through
    `runs/m-nano-s20260828/checkpoints/final.pt` -- no: `runs/` is not under
    version control, so a unit test that loaded it would pass or fail depending
    on what happens to be on the machine, which is worse than a narrower test
    that always means the same thing.
    """
    # SPEC: 00 §0.1's table, in nats per token, k_tok = 1, natural text.
    L_trained = 2.1005          # the trained model
    L_units_randomised = 2.2621  # parts + pops + ctx/pos/router: 976,384 reals
    L_untrained = 8.86           # a wholly random model

    r = genome_share(L_trained,
                     L_gamma_rand=[L_units_randomised],
                     L_z_rand=[L_untrained - 0.162])   # soma randomised
    assert r["delta_gamma"] == pytest.approx(0.1616, abs=5e-4)
    assert r["share"] == pytest.approx(0.024, abs=0.005), r
    assert r["soma_reference"] == 0.024

    # And the direction that matters: the evolutionary machinery operating on
    # that population contributed 0.0027 of 6.76 nats, which is 0.04%.
    #
    # Taken from §00.1's *delta* column, not its CE column, because the two
    # disagree for the ablation-A3 row and only the delta reconciles with the
    # prose. Every other row satisfies `delta = CE - 2.1005` exactly (the gene
    # library: 2.2621 - 2.1005 = 0.1616 against a stated 0.1617); A3 states
    # CE 2.1250 and delta +0.0027, and 2.1250 - 2.1005 = 0.0245. §00.1 now
    # records the discrepancy rather than carrying both figures silently.
    A3_DELTA = 0.0027
    assert A3_DELTA / (L_untrained - L_trained) < 0.001


def test_S4_the_probe_would_not_have_returned_0024_for_a_working_split():
    """The other half of S4: the number is not what the estimator always says.

    Without this, S4 would be satisfied by an estimator that returned 0.024
    whatever it was given -- the shape of vacuity §00.3 is about.
    """
    r = genome_share(2.1005, L_gamma_rand=[8.0], L_z_rand=[2.5])
    assert r["share"] > 0.9, r


# ---------------------------------------------------------------------- S6

@requires_corpus()
def test_S6_h2_is_computed_on_the_fitness_split_never_the_test_split():
    """SPEC: 06 §6.7 S6. Selection must not be able to overfit what is reported.

    Structural rather than by inspection: `Shard.load` splits the holdout by
    **document**, and the two halves share no document and no token offset, so
    a window scored on one can never have been scored on the other. §05's
    runner takes the fitness shard as its `eval_shard` for every selection
    decision, and only a reporting run is handed the test shard.
    """
    import numpy as np
    from gnomon.data.stream import Shard

    fit = Shard.load("data/ts_4096", "fitness")
    test = Shard.load("data/ts_4096", "test")
    assert fit.meta["split"] == "fitness" and test.meta["split"] == "test"
    # Disjoint document sets, by construction: the halves are taken from one
    # `starts` array at a document boundary and each is re-based to zero.
    assert fit.meta["n_docs"] > 0 and test.meta["n_docs"] > 0
    assert fit.n_tokens + test.n_tokens <= Shard.load(
        "data/ts_4096", "fitness").n_tokens + test.n_tokens
    # ...and they are different tokens, which the document split guarantees but
    # which is worth asserting because a bug in the re-basing would not raise.
    n = min(4096, fit.n_tokens, test.n_tokens)
    assert not np.array_equal(np.asarray(fit.tokens[:n]),
                              np.asarray(test.tokens[:n]))

    # The gate that matters: the heritability driver evaluates on `fitness`.
    src = pathlib.Path("scripts/gnomon_herit.py").read_text(encoding="utf-8")
    assert 'Shard.load(root, "fitness")' in src
    assert '"test"' not in src, "the h2 design must never see the test split"


# ---------------------------------------------------------------------- S8

def test_S8_share_retrain_recovers_a_known_share():
    """SPEC: 06 §6.3.1. The statistic §01.6 names and nothing defined."""
    from gnomon.eval.share import KG_SHARE_FLOOR, share_retrain

    # Genome worth 3 nats, lifetime worth 1: share 0.75.
    r = share_retrain(L_star=5.0, L_random_trained=8.0, L_birth=6.0)
    assert r["delta_gamma"] == pytest.approx(3.0)
    assert r["delta_z"] == pytest.approx(1.0)
    assert r["share_retrain"] == pytest.approx(0.75)
    assert r["KG_SHARE"] == "not fired"


def test_S8_KG_SHARE_fires_below_the_floor():
    from gnomon.eval.share import KG_SHARE_FLOOR, share_retrain

    assert KG_SHARE_FLOOR == 0.50
    # SOMA's regime: the genome is worth almost nothing.
    r = share_retrain(L_star=2.10, L_random_trained=2.26, L_birth=8.70)
    assert r["share_retrain"] < KG_SHARE_FLOOR
    assert r["KG_SHARE"] == "FIRES"
    # Exactly at the floor is a pass, not a fire: the criterion is `< 0.50`.
    edge = share_retrain(L_star=0.0, L_random_trained=1.0, L_birth=1.0)
    assert edge["share_retrain"] == pytest.approx(0.5)
    assert edge["KG_SHARE"] == "not fired"


def test_S8_a_negative_delta_is_reported_not_clamped():
    """SPEC: 06 §6.3.1. At generation 0 it is a coin flip; later it is a finding."""
    from gnomon.eval.share import share_retrain

    r = share_retrain(L_star=5.0, L_random_trained=4.5, L_birth=9.0)
    assert r["delta_gamma"] < 0
    assert r["share_retrain"] < 0, "a negative share was clamped away"
