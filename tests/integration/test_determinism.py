"""Two lifetimes with the same seed are the same lifetime.

SPEC: 02 §2.6.3
SPEC: 07 §7.7

`RngBundle` gives every purpose its own stream, and the unit suite checks that
those streams are independent and round-trip. What it cannot check is the claim
that actually matters: that a **whole lifetime** — model construction, data
order, selection, mutation, growth and the archive — is a function of `seed`
alone.

That claim was false until audit A-64. `nn.Linear.reset_parameters` and
`nn.init.normal_` take no `generator`, so module construction drew from the
process-global stream that `RngBundle` deliberately never touches, and two
invocations of the same script with the same seed produced different models.
It was found by noticing that a *failure* moved between runs — the loss curve
itself never looked wrong.

The run is cut at `max_steps` so that at least two selection events fire (`T_0`
is 57 at this configuration) while the test still costs under a minute per arm.
Determinism through the structural machinery is the part that was broken; the
optimizer's own determinism is not in question.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma.eval.regression import solve_rate
from soma.germline import Genome
from soma.train import LifetimeConfig, lifetime

pytestmark = pytest.mark.slow

STEPS = 120


def _run(tiny_setup, seed: int, tmp_path, run_id: str):
    cfg, corpus, eval_batches, probe = tiny_setup

    def solve_rate_fn(model, genome, params, n_eval=8):
        return solve_rate(
            model, params, min(n_eval, 4), np.random.default_rng(0),
            genome=genome, T_ctx=cfg.T_ctx, device="cpu",
        )

    genome = Genome.initialise(
        cfg, generator=torch.Generator().manual_seed(11), lineage_id="det"
    )
    lc = LifetimeConfig(
        cfg=cfg, run_id=run_id, runs_root=str(tmp_path), device="cpu",
        corpus=corpus, eval_batches=eval_batches, probe_tokens=probe,
        L_star=4.0, slope_ref=1.0, rows_per_batch=2, micro_batches=1,
        eval_every=40, verbose=False, log_to_disk=False, max_steps=STEPS,
    )
    return lifetime(genome, lc, seed=seed, solve_rate_fn=solve_rate_fn)


def _trace(res):
    return [(p.step, round(p.ce, 10)) for p in res.curve.points]


@pytest.fixture(scope="module")
def paired(tiny_setup, tmp_path_factory):
    root = tmp_path_factory.mktemp("det")
    # Deliberately advance the process-global stream between the two runs. If
    # anything in the lifetime still draws from it, the arms will differ — which
    # is exactly the A-64 defect, and is what makes this test able to fail.
    a = _run(tiny_setup, 21, root, "det-a")
    torch.randn(4096)
    np.random.default_rng().random(4096)
    b = _run(tiny_setup, 21, root, "det-b")
    c = _run(tiny_setup, 22, root, "det-c")
    return a, b, c


def test_the_same_seed_gives_the_same_held_out_curve(paired) -> None:
    a, b, _ = paired
    assert _trace(a) == _trace(b), (_trace(a), _trace(b))


def test_the_same_seed_gives_the_same_structural_history(paired) -> None:
    """The part a loss curve would hide: which units were born, when, and where."""
    a, b, _ = paired
    assert a.selection_events == b.selection_events
    assert a.curve.structural_events == b.curve.structural_events
    assert a.metrics["total_births"] == b.metrics["total_births"]
    assert a.metrics["total_deaths"] == b.metrics["total_deaths"]
    assert a.archetypes == b.archetypes


def test_a_different_seed_gives_a_different_run(paired) -> None:
    """Without this the two assertions above pass on a run that ignores its seed."""
    a, _, c = paired
    assert _trace(a) != _trace(c)
