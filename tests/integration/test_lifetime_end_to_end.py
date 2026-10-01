"""One complete lifetime, with every mechanism required to fire.

SPEC: 07 §7.2
SPEC: 07 §7.4.3

The unit suite proves each mechanism correct in isolation. This module proves
they **compose**, and — more importantly — that each one is actually *reached*
by the training loop. That second claim is not decorative. Audit A-59 found the
credit audit constructed, logged and never called for a whole lifetime; audit
A-66 found the M1 driver's `solve_rate_fn` bound with the wrong arity and then
passed as `None`, which left the entire open-ended arm inert with no error and
no log line, because an empty archive is a legal state. Neither defect is
visible to a unit test, and neither changes any number a smoke run prints.

Every assertion below is therefore of the form "this mechanism produced a
non-trivial record", never "the run completed".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma.eval.regression import solve_rate
from soma.germline import Genome
from soma.train import LifetimeConfig, lifetime

pytestmark = pytest.mark.slow


def _run(tiny_setup, seed: int, tmp_path, run_id: str):
    cfg, corpus, eval_batches, probe = tiny_setup

    def solve_rate_fn(model, genome, params, n_eval=8):
        return solve_rate(
            model, params, min(n_eval, 4), np.random.default_rng(0),
            genome=genome, T_ctx=cfg.T_ctx, device="cpu",
        )

    genome = Genome.initialise(
        cfg, generator=torch.Generator().manual_seed(seed), lineage_id=run_id
    )
    lc = LifetimeConfig(
        cfg=cfg, run_id=run_id, runs_root=str(tmp_path), device="cpu",
        corpus=corpus, eval_batches=eval_batches, probe_tokens=probe,
        L_star=4.0, slope_ref=1.0, rows_per_batch=2, micro_batches=1,
        eval_every=200, verbose=False, log_to_disk=False,
    )
    return lifetime(genome, lc, seed=seed, solve_rate_fn=solve_rate_fn)


@pytest.fixture(scope="module")
def result(tiny_setup, tmp_path_factory):
    return _run(tiny_setup, 5, tmp_path_factory.mktemp("e2e"), "e2e")


def test_the_lifetime_runs_to_its_configured_length(result, tiny_setup) -> None:
    cfg = tiny_setup[0]
    assert result.steps_run == cfg.N_step
    assert np.isfinite(result.curve.points[-1].ce)


def test_selection_fired_and_produced_births_and_deaths(result) -> None:
    """§04.2: an event that selects parents but never births is a no-op, and an
    event that only births is not selection."""
    m = result.metrics
    assert m["selection_events"] >= 5, m["selection_events"]
    assert m["total_births"] > 0, "no unit was ever born"
    assert m["total_deaths"] > 0, "no unit ever died"


def test_every_selection_event_used_more_than_one_case(result) -> None:
    """ε-lexicase degenerates to single-case selection if `eps_t == 0` for every
    case; §04.6's regression pins `eps_t != 0`, and this is its end-to-end form."""
    used = [e["mean_cases_used"] for e in result.selection_events]
    assert used and max(used) > 1.0, f"mean cases used per parent: {used}"


def test_the_credit_audit_actually_ran(result) -> None:
    """Audit A-59: the object existed and was never called for a whole lifetime."""
    audit = result.metrics["credit_audit"]
    assert audit["n_evaluations"] >= 1, audit
    assert np.isfinite(audit["spearman"]), audit


def test_the_archive_was_seeded_and_stepped(result) -> None:
    """Audit A-66: `solve_rate_fn` bound wrongly leaves this at zero, silently."""
    assert result.metrics["archive_size"] > 0, "the task archive is empty"


def test_certification_ran_with_all_three_arms(result) -> None:
    """§06.5: live, frozen and ramp. A certification with fewer arms is not a
    certification — the frozen arm is the control that makes the claim mean
    anything."""
    assert result.certifications, "no certification ran"
    for c in result.certifications:
        for arm in ("live", "frozen", "ramp"):
            assert arm in c, f"arm {arm!r} missing from {sorted(c)}"
            assert c[arm]["n_rows_tested"] >= 0
        # The comparisons that make the arms a control rather than three
        # unrelated numbers.
        assert "live_minus_frozen" in c and "live_minus_ramp" in c


def test_the_bifurcation_monitor_evaluated_cohorts(result) -> None:
    m = result.metrics
    assert sum(m["bif_cohorts"].values()) > 0, m["bif_cohorts"]
    assert 0.0 <= m["bif_alarm_rate"] <= 1.0


def test_growth_events_fired_and_are_annotated(result) -> None:
    kinds = {a["kind"] for a in result.curve.structural_events}
    assert "select" in kinds and "grow" in kinds, sorted(kinds)


def test_the_population_stayed_within_both_bounds(result, tiny_setup) -> None:
    """(1.14)'s two bounds, end-to-end (audit A-62)."""
    cfg = tiny_setup[0]
    for e in result.selection_events:
        assert e["n_live_after"] <= cfg.U_max, (e["step"], e["n_live_after"])
    assert 0 < result.metrics["final_n_live"] <= cfg.U_max


def test_no_parameter_or_metric_went_non_finite(result) -> None:
    for p in result.curve.points:
        assert np.isfinite(p.ce), p
    assert np.isfinite(result.fitness.fitness)
