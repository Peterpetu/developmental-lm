"""The K-CRIT trigger: both limbs, the standard-error gate, and depth invariance.

SPEC: 07 §7.6 (the trigger and the per-block band), §7.2 (no verdict from a displacement
inside `SE_GATE` standard errors), 11 §P-C.33 and §P-D.1.

`trigger` and `needs_control` decide whether K-CRIT is evaluated at all, and they were
rewritten by the second adversarial round after the first form compared `|Λ|` with the band
and never looked at a standard error at all. A rewritten decision function with no test is the
shape of defect that round kept finding, so the cases below cover both limbs, both directions
of the gate, the depth invariance the per-block correction was made for, and the adapter in
`scripts/tempo_analyse.py` that renames the record's keys on the way in.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

from tempo.criticality import BAND, SE_GATE, needs_control, trigger

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

import tempo_analyse as A  # noqa: E402


def profile(per_block: float, *, n_layers: int = 8, se: float = 0.001,
            block_se: float = 0.001, worst=None) -> dict:
    """A profile whose `2 n_layers` sub-blocks all sit at `per_block`, except that one ffn
    block is moved to `worst` when asked. `Lambda` is their sum, as (7.3) defines it."""
    n = 2 * n_layers
    mix = {str(l): per_block for l in range(n_layers)}
    ffn = {str(l): per_block for l in range(n_layers)}
    if worst is not None:
        ffn["0"] = worst
    lam = sum(mix.values()) + sum(ffn.values())
    return {"Lambda": lam, "n_blocks": n, "lambda_block": lam / n, "lambda_block_se": se,
            "Lambda_se": se * n, "mix": mix, "ffn": ffn,
            "mix_se": {k: block_se for k in mix}, "ffn_se": {k: block_se for k in ffn}}


class TestNeitherLimb:
    def test_a_profile_well_inside_the_band_does_not_fire(self):
        t = trigger(profile(-0.03))
        assert t["fires"] is False
        assert t["stack_excess_se"] == 0.0 and t["block_excess_se"] == 0.0

    def test_a_profile_exactly_at_the_band_does_not_fire(self):
        """The band is closed: a displacement of exactly zero is not a displacement."""
        t = trigger(profile(BAND))
        assert t["fires"] is False and t["stack_excess_se"] == 0.0

    def test_the_sign_does_not_matter(self):
        assert trigger(profile(-0.6, se=0.01))["fires"] is trigger(profile(0.6, se=0.01))["fires"]


class TestTheStackLimb:
    def test_it_fires_when_the_per_block_mean_leaves_the_band_by_more_than_the_gate(self):
        t = trigger(profile(-0.6, se=0.01, block_se=1.0))     # block se large: that limb is gated off
        assert t["fires"] is True
        assert t["stack_excess_se"] == pytest.approx(0.1 / 0.01)
        assert t["block_excess_se"] < SE_GATE

    def test_the_gate_suppresses_a_displacement_inside_two_standard_errors(self):
        """The defect this closes: the first `needs_control` compared the value with the band
        and would have returned a verdict on a displacement of a fifth of a standard error."""
        t = trigger(profile(-0.51, se=0.05, block_se=1.0))    # excess 0.01 = 0.2 se
        assert t["stack_excess_se"] == pytest.approx(0.2)
        assert t["fires"] is False

    def test_a_missing_error_bar_does_not_suppress_a_real_displacement(self):
        """No error bar must fail toward evaluating K-CRIT, not toward skipping it: firing
        costs two arms of compute, suppressing costs a criterion."""
        p = profile(-0.9)
        p["lambda_block_se"], p["Lambda_se"] = float("nan"), float("nan")
        p["mix_se"], p["ffn_se"] = {}, {}
        t = trigger(p)
        assert t["fires"] is True and t["stack_excess_se"] == math.inf

    def test_the_per_block_mean_is_derived_from_the_sum_when_absent(self):
        p = profile(-0.6, se=0.01)
        lam, n, se = p["Lambda"], p["n_blocks"], p["Lambda_se"]
        del p["lambda_block"], p["lambda_block_se"]
        t = trigger(p)
        assert t["lambda_block"] == pytest.approx(lam / n)
        assert t["lambda_block_se"] == pytest.approx(se / n)


class TestTheBlockLimb:
    def test_one_bad_block_fires_while_the_stack_is_calm(self):
        t = trigger(profile(-0.01, se=1.0, block_se=0.01, worst=-0.8))
        assert t["fires"] is True
        assert t["worst_block"] == pytest.approx(0.8)
        assert t["worst_block_key"] == ["ffn", "0"]
        assert t["block_excess_se"] == pytest.approx(0.3 / 0.01)
        assert t["stack_excess_se"] == 0.0

    def test_the_gate_applies_to_this_limb_too(self):
        t = trigger(profile(-0.01, se=1.0, block_se=0.5, worst=-0.6))   # excess 0.1 = 0.2 se
        assert t["block_excess_se"] == pytest.approx(0.2) and t["fires"] is False

    def test_the_worst_block_is_the_largest_in_absolute_value(self):
        p = profile(0.1, block_se=0.01, worst=-0.4)
        assert trigger(p)["worst_block"] == pytest.approx(0.4)

    def test_a_profile_with_no_per_block_entries_reports_no_worst_block(self):
        p = profile(-0.03)
        p["mix"], p["ffn"] = {}, {}
        t = trigger(p)
        assert t["worst_block_key"] is None and t["block_excess_se"] == 0.0


class TestDepthInvariance:
    """11 §P-C.33: `Λ` is a sum over `K_a` sub-blocks, so a band on it is a band on depth.
    The same per-block behaviour must give the same verdict at the primordium and the adult."""

    @pytest.mark.parametrize("per_block", (-0.03, -0.2, -0.6, 0.55))
    def test_the_verdict_is_the_same_at_eight_and_sixteen_layers(self, per_block):
        a = trigger(profile(per_block, n_layers=8, se=0.01, block_se=0.01))
        b = trigger(profile(per_block, n_layers=16, se=0.01, block_se=0.01))
        assert a["fires"] is b["fires"]
        assert a["lambda_block"] == pytest.approx(b["lambda_block"])

    def test_the_stack_sum_doubles_while_the_per_block_mean_does_not(self):
        a, b = profile(-0.03, n_layers=8), profile(-0.03, n_layers=16)
        assert b["Lambda"] == pytest.approx(2 * a["Lambda"])
        assert trigger(b)["lambda_block"] == pytest.approx(trigger(a)["lambda_block"])


class TestNeedsControl:
    def test_it_is_any_over_the_profiles(self):
        calm, loud = profile(-0.03), profile(-0.9, se=0.01, block_se=0.01)
        assert needs_control([calm, calm]) is False
        assert needs_control([calm, loud, calm]) is True

    def test_no_profile_is_not_a_trigger(self):
        """An unmeasured life must not fire K-CRIT; 10 §10.5 makes it NOT EVALUATED instead."""
        assert needs_control([]) is False


class TestTheRecordedProfileOfR1:
    """A regression lock on the published number (11 §P-D.1), so that a change to the band,
    the gate or the exponent has to come past this test and be recorded as a correction."""

    LAMBDA = -0.511059045791626
    LAMBDA_SE = 0.052901215851306915
    BLOCK = -0.031941190361976624
    BLOCK_SE = 0.003306325990706682
    WORST = -0.0461            # ffn, layer 0

    def _profile(self) -> dict:
        return {"Lambda": self.LAMBDA, "Lambda_se": self.LAMBDA_SE, "n_blocks": 16,
                "lambda_block": self.BLOCK, "lambda_block_se": self.BLOCK_SE,
                "mix": {"0": -0.04}, "ffn": {"0": self.WORST},
                "mix_se": {"0": 0.004}, "ffn_se": {"0": 0.004}}

    def test_it_does_not_fire_either_limb(self):
        t = trigger(self._profile())
        assert t["fires"] is False
        assert t["stack_excess_se"] == 0.0 and t["block_excess_se"] == 0.0

    def test_the_headroom_is_the_order_of_magnitude_the_ledger_records(self):
        """11 §P-D.1 records 15.7x on the stack limb and 10.8x on the worst-block limb. If the
        band is ever recalibrated these change, and that ledger entry must change with them."""
        assert BAND / abs(self.BLOCK) == pytest.approx(15.7, abs=0.1)
        assert BAND / abs(self.WORST) == pytest.approx(10.8, abs=0.1)

    def test_the_old_reading_that_would_have_fired_it_was_arithmetically_wrong(self):
        """The first draft called `|Λ| = 0.511` 'outside the band by about 2 Λ_se'. It is
        0.21 Λ_se. The 9.7 sigma in that number is the distance from *zero*, which the trigger
        does not ask about."""
        assert (abs(self.LAMBDA) - 0.5) / self.LAMBDA_SE == pytest.approx(0.21, abs=0.01)
        assert abs(self.LAMBDA) / self.LAMBDA_SE == pytest.approx(9.66, abs=0.02)


class TestTheAnalyserAdapter:
    """`scripts/tempo_analyse.py::profiles` renames the record's `lambda_mix` to the `mix`
    that `trigger` reads. If that rename is ever dropped the block limb does not fail: it
    silently sees an empty dict and reports 0.0, and K-CRIT quietly loses half its trigger."""

    def test_a_record_episode_reaches_both_limbs(self):
        rec = {"episodes": [{"Lambda": -1.8, "Lambda_se": 0.16, "n_blocks": 16,
                             "lambda_block": -0.1125, "lambda_block_se": 0.01,
                             "lambda_mix": {"3": -0.9}, "lambda_ffn": {"3": -0.05},
                             "lambda_mix_se": {"3": 0.01}, "lambda_ffn_se": {"3": 0.01}}]}
        p = A.profiles(rec)
        assert len(p) == 1
        t = trigger(p[0])
        assert t["worst_block"] == pytest.approx(0.9), "the block limb saw an empty profile"
        assert t["worst_block_key"] == ["mix", "3"] and t["fires"] is True

    def test_an_episode_with_no_exponent_is_dropped_rather_than_read_as_zero(self):
        assert A.profiles({"episodes": [{"Lambda": None}, {}]}) == []
