"""The verdict layer, against synthetic records whose right answer is known.

SPEC: 10 §10.5 (the one verdict rule), §10.1 (a sigma per criterion; no missing or incomplete
run is a pass), §10.2 (the criteria).

`scripts/tempo_analyse.py` decides every verdict of this project and, until the second
adversarial round, had no test at all: a defect there produces a wrong *conclusion* rather
than a crash or a bad loss, and nothing would have caught it. Every case below builds a run
directory on disk and asks the real `analyse()` what it says.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

import tempo_analyse as A  # noqa: E402


# =============================================================================
# builders
# =============================================================================

def life(scores, *, F0=1e14, dF=1e14, complete=True, stream="text", rewards=None,
         evals=None, extra_ep=None):
    """A life record: one episode per entry of `scores`, each with a cumulative-FLOP point."""
    eps = []
    for i, s in enumerate(scores):
        ev = {"stream": stream, "fresh": s, "continuing": s, "n": 8}
        if stream in A.REWARD_STREAMS:
            ev["reward"] = -s                                  # a loss enters as minus the reward
        e = {"episode": i, "L_check": s, "F_cum": F0 + i * dF, "F_episode": dF,
             "eval_final": {stream: ev}}
        if rewards is not None:
            e["reward"] = rewards[i] if i < len(rewards) else rewards[-1]
        if extra_ep:
            e.update(extra_ep)
        eps.append(e)
    return {"episodes": eps, "evals": evals or [], "complete": complete}


def result(nll, *, complete=True, **kw):
    return {"test_nll_targets_K_to_TK": nll, "complete": complete, "flop_ratio": 1.0,
            "width": 256, **kw}


def write(root: pathlib.Path, name: str, blob: dict, fname="record.json"):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(json.dumps(blob), encoding="utf-8")


def arms(tmp_path, mech_scores, ctrl_scores, *, run="R4", mech="tempo", ctrl="a_gamma0", **kw):
    """One mechanism arm and one control arm, `n` seeds each, plus the R1 noise arm."""
    root = tmp_path / "runs"
    for s, v in enumerate(mech_scores):
        write(root, f"{run}-{mech}-ts-s{s}", life(v, **kw))
    for s, v in enumerate(ctrl_scores):
        write(root, f"{run}-{ctrl}-ts-s{s}", life(v, **kw))
    return root


def verdict_of(root, key, config="ts"):
    rep = A.analyse(A.load_runs(root), config)
    return rep["verdicts"].get(key, {}).get("verdict"), rep


# =============================================================================
# the rule itself
# =============================================================================

class TestTheVerdictRule:
    @pytest.mark.parametrize("m,lo,hi,bar,fires,expect", [
        # "not better": DID NOT FIRE iff m <= -bar and hi < 0
        (-1.0, -1.2, -0.8, 0.5, "not_better", A.DID_NOT_FIRE),
        (-0.5, -0.7, -0.3, 0.5, "not_better", A.DID_NOT_FIRE),   # m exactly -bar, hi < 0
        (-0.5, -1.5, +0.5, 0.5, "not_better", A.INCONCLUSIVE),   # by the bar but not significant
        (-0.2, -0.3, -0.1, 0.5, "not_better", A.INCONCLUSIVE),   # better, not by the bar
        (+0.0, -0.1, +0.1, 0.5, "not_better", A.FIRED),          # not better at all
        (+1.0, +0.8, +1.2, 0.5, "not_better", A.FIRED),
        # "worse": FIRED iff m >= bar and lo > 0
        (+1.0, +0.8, +1.2, 0.5, "worse", A.FIRED),
        (+0.5, +0.1, +0.9, 0.5, "worse", A.FIRED),               # m exactly bar, lo > 0
        (+1.0, -0.2, +2.2, 0.5, "worse", A.INCONCLUSIVE),        # worse but not significant
        (+0.2, +0.1, +0.3, 0.5, "worse", A.DID_NOT_FIRE),        # hi < bar
        (-1.0, -1.2, -0.8, 0.5, "worse", A.DID_NOT_FIRE),
    ])
    def test_both_forms(self, m, lo, hi, bar, fires, expect):
        c = {"n": 4, "mean": m, "lo": lo, "hi": hi, "sd": 0.1}
        assert A.verdict(c, bar, fires_when=fires) == expect

    def test_fewer_than_two_paired_seeds_is_not_evaluated(self):
        """10 §10.5: one seed means the contrast was not run, which is not 'unclear'."""
        c = {"n": 1, "mean": -9.0, "lo": float("nan"), "hi": float("nan"), "sd": float("nan")}
        assert A.verdict(c, 0.5, fires_when="not_better") == A.NOT_EVALUATED
        assert A.verdict(None, 0.5, fires_when="not_better") == A.NOT_EVALUATED

    def test_a_missing_bar_is_not_evaluated(self):
        c = {"n": 4, "mean": -9.0, "lo": -9.1, "hi": -8.9, "sd": 0.1}
        assert A.verdict(c, None, fires_when="not_better") == A.NOT_EVALUATED
        assert A.verdict(c, float("nan"), fires_when="not_better") == A.NOT_EVALUATED

    def test_a_zero_bar_still_decides(self):
        """A control arm with no spread gives bar 0; that is a real bar, not a missing one."""
        c = {"n": 4, "mean": -0.5, "lo": -0.6, "hi": -0.4, "sd": 0.05}
        assert A.verdict(c, 0.0, fires_when="not_better") == A.DID_NOT_FIRE


# =============================================================================
# completeness and missing arms
# =============================================================================

class TestCompleteness:
    def test_an_incomplete_record_is_not_loaded(self, tmp_path):
        root = tmp_path / "runs"
        write(root, "R4-tempo-ts-s0", life([1.0, 0.9], complete=False))
        assert A.load_runs(root) == {}

    def test_an_incomplete_result_is_not_loaded(self, tmp_path):
        """The same standard for `result.json` as for `record.json`."""
        root = tmp_path / "runs"
        write(root, "R2-token-ts-s0", result(2.0, complete=False), "result.json")
        assert A.load_runs(root) == {}
        write(root, "R2-token-ts-s1", result(2.0), "result.json")
        assert set(A.load_runs(root)) == {("R2", "token", "ts")}

    def test_a_truncated_result_is_not_loaded(self, tmp_path):
        root = tmp_path / "runs"
        d = root / "R2-token-ts-s0"
        d.mkdir(parents=True)
        (d / "result.json").write_text('{"test_nll": 2.0', encoding="utf-8")
        assert A.load_runs(root) == {}

    def test_k_cont_with_incomplete_baselines_is_not_evaluated(self, tmp_path):
        root = tmp_path / "runs"
        for s in range(4):
            write(root, f"R1-nolat-ts-s{s}", life([3.0 + 0.01 * s]))
            write(root, f"R2-token-ts-s{s}", result(2.0, complete=False), "result.json")
        v, _ = verdict_of(root, "K-CONT")
        assert v == A.NOT_EVALUATED

    def test_k_cont_fires_when_the_span_agent_is_worse(self, tmp_path):
        root = tmp_path / "runs"
        for s in range(4):
            write(root, f"R1-nolat-ts-s{s}", life([3.0 + 0.001 * s]))
            write(root, f"R2-token-ts-s{s}", result(2.0 + 0.001 * s), "result.json")
        v, rep = verdict_of(root, "K-CONT")
        assert v == A.FIRED
        assert rep["verdicts"]["K-CONT"]["contrast"]["mean"] == pytest.approx(1.0, abs=1e-9)


# =============================================================================
# the criteria that were wrong
# =============================================================================

class TestKHorizon:
    def test_a_small_positive_difference_does_not_fire(self, tmp_path):
        """The defect: K-HORIZON fired on any positive point estimate, with no bar and no
        interval. A 0.005 loss against a 0.024 bar is not a fire."""
        root = arms(tmp_path,
                    [[1.0, 0.905], [1.0, 0.925], [1.0, 0.945], [1.0, 0.965]],
                    [[1.0, 0.900], [1.0, 0.920], [1.0, 0.940], [1.0, 0.960]],
                    rewards=[0.05, 0.05])
        v, rep = verdict_of(root, "K-HORIZON")
        c = rep["verdicts"]["K-HORIZON"]
        assert c["contrast"]["mean"] == pytest.approx(0.005, abs=1e-6)
        assert c["bar"] > c["contrast"]["mean"]
        assert v == A.DID_NOT_FIRE

    def test_a_missing_control_is_not_evaluated(self, tmp_path):
        root = tmp_path / "runs"
        for s in range(4):
            write(root, f"R4-tempo-ts-s{s}", life([1.0, 0.9], rewards=[0.05, 0.05]))
        v, _ = verdict_of(root, "K-HORIZON")
        assert v == A.NOT_EVALUATED

    def test_it_needs_a_positive_reward_as_well_as_a_worse_score(self, tmp_path):
        """A rule that is merely worse has not shown the short-horizon *bias*: the bias is
        winning per-episode rewards while losing the outcome."""
        root = arms(tmp_path,
                    [[1.0, 1.9], [1.0, 1.91], [1.0, 1.92], [1.0, 1.93]],
                    [[1.0, 0.90], [1.0, 0.91], [1.0, 0.92], [1.0, 0.93]],
                    rewards=[-0.5, -0.5])
        v, rep = verdict_of(root, "K-HORIZON")
        assert rep["verdicts"]["K-HORIZON"]["mean_reward"] < 0
        assert v == A.DID_NOT_FIRE


class TestKXfer:
    def test_streams_that_were_never_run_are_not_losses(self, tmp_path):
        """The defect: an unevaluated stream counted against the criterion, so two missing
        streams could fire it."""
        root = tmp_path / "runs"
        evs = [{"episode": 0, "n": 25, "control": {"fresh": 5.0, "reward": 0.5}}]
        for s in range(4):
            write(root, f"R9-tempo-tempo-ts-s{s}", life([1.0], evals=evs))
            write(root, f"R9-tempo-reset-ts-s{s}", life([1.0], evals=evs))
        v, rep = verdict_of(root, "K-XFER")
        assert rep["verdicts"]["K-XFER"]["pairs_evaluated"] < len(A.XFER_STREAMS)
        assert v == A.NOT_EVALUATED


class TestKBlend:
    def test_a_missing_specific_arm_is_not_a_pass(self, tmp_path):
        """10 §10.2: without the stream-specific arms, NOT EVALUATED, never passed."""
        root = tmp_path / "runs"
        for s in range(4):
            write(root, f"R8-tempo-ts-s{s}", life([1.0]))
        v, rep = verdict_of(root, "K-BLEND")
        assert v == A.NOT_EVALUATED
        assert rep["verdicts"]["K-BLEND"]["missing"]


class TestStreamScore:
    def test_a_missing_reward_is_not_replaced_by_the_loss(self):
        """The defect: a `null` reward fell through to `fresh`, putting a loss in nats where a
        negative reward belonged, and reporting the change of units as a regression."""
        assert A.stream_score({"fresh": 9.9, "reward": None}, "control") is None
        assert A.stream_score({"fresh": 9.9}, "control") is None
        assert A.stream_score({"fresh": 9.9, "reward": 0.5}, "control") == -0.5
        assert A.stream_score({"fresh": 9.9}, "text") == 9.9
        assert A.stream_score({"fresh": None}, "text") is None
        assert A.stream_score(None, "text") is None


class TestNonFinite:
    def test_a_null_reward_or_lambda_does_not_crash_the_analysis(self, tmp_path):
        root = arms(tmp_path, [[1.0, 0.9]] * 4, [[1.0, 0.95]] * 4, rewards=[None, 0.1])
        rep = A.analyse(A.load_runs(root), "ts")
        assert isinstance(rep["lines"], list) and rep["lines"]

    def test_a_null_f_cum_drops_the_point_rather_than_raising(self, tmp_path):
        root = tmp_path / "runs"
        rec = life([1.0, 0.9])
        rec["episodes"][0]["F_cum"] = None
        write(root, "R4-tempo-ts-s0", rec)
        assert A.curve(rec) == [(rec["episodes"][1]["F_cum"], 0.9)]


# =============================================================================
# equal-compute comparison
# =============================================================================

class TestEqualCompute:
    def test_the_budget_is_the_smallest_final_one(self, tmp_path):
        a = {0: life([1.0, 0.9, 0.8])}                       # F to 3e14
        b = {0: life([1.0, 0.9])}                            # F to 2e14
        assert A.common_budget(a, b) == 2e14

    def test_a_run_with_no_curve_refuses_the_comparison(self):
        a = {0: life([1.0, 0.9])}
        b = {0: {"episodes": [], "complete": True}}
        assert A.common_budget(a, b) is None

    def test_interpolation_and_its_edges(self):
        r = life([1.0, 0.8, 0.6])                            # F = 1e14, 2e14, 3e14
        assert A.score_at(r, 1e14) == pytest.approx(1.0)
        assert A.score_at(r, 1.5e14) == pytest.approx(0.9)
        assert A.score_at(r, 3e14) == pytest.approx(0.6)
        assert A.score_at(r, 0.5e14) is None                 # below the first point
        assert A.score_at(r, 4e14) is None                   # beyond the last
        assert A.score_at(r, None) is None

    def test_an_arm_that_spends_more_per_episode_gets_fewer_episodes(self, tmp_path):
        """The point of comparing at equal compute: a costlier arm is read earlier in its life."""
        root = tmp_path / "runs"
        for s in range(4):
            write(root, f"R4-tempo-ts-s{s}", life([1.0, 0.5, 0.4], F0=2e14, dF=2e14))
            write(root, f"R4-a_gamma0-ts-s{s}", life([1.0, 0.9, 0.8, 0.7, 0.6], F0=1e14, dF=1e14))
        rep = A.analyse(A.load_runs(root), "ts")
        c = rep["verdicts"]["K-GENE vs a_gamma0"]
        assert c["contrast"] is not None
        assert "at F=" in c["note"]


class TestCombine:
    def test_a_demonstrated_fire_beats_a_missing_arm(self):
        parts = [{"verdict": A.FIRED}, {"verdict": A.NOT_EVALUATED}]
        assert A.combine(parts) == A.FIRED

    def test_a_missing_arm_otherwise_blocks_a_pass(self):
        parts = [{"verdict": A.DID_NOT_FIRE}, {"verdict": A.NOT_EVALUATED}]
        assert A.combine(parts) == A.NOT_EVALUATED

    def test_all_clear_passes(self):
        assert A.combine([{"verdict": A.DID_NOT_FIRE}] * 3) == A.DID_NOT_FIRE


class TestEarlyScore:
    def test_it_reads_the_first_evaluation_of_the_first_episode(self):
        rec = life([1.0], evals=[
            {"episode": 1, "n": 25, "task": {"fresh": 9.0, "reward": 0.9}},
            {"episode": 0, "n": 50, "task": {"fresh": 5.0, "reward": 0.5}},
            {"episode": 0, "n": 25, "task": {"fresh": 4.0, "reward": 0.4}},
        ])
        assert A.early_score(rec, "task") == pytest.approx(-0.4)
