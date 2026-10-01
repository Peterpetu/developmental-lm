"""§08.2's behavioural regression suite: its specification and its harness.

SPEC: 08 §8.2
SPEC: 06 §6.3.5

Gate `G-STATETRACK` has two halves that are usually conflated and must not be:

1. **The suite is specified correctly** — the right families, the right
   `acc_random` for each, the right pass bars, and evaluation settings that lie
   **outside** the archive's holdout so a bar cannot be met by having trained on
   it. This half is checkable now, and it is what this module checks.
2. **A trained model clears the bars.** That needs an M1 checkpoint and is a
   Phase-6 run, not a unit of code.

Every scoring path here is exercised against **both** a perfect decoder and a
chance-level one. A harness that reports `1.0` for everything and a harness that
reports `0.0` for everything both look like "the suite ran"; only the pair of
controls distinguishes a working scorer from either.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import soma.eval.regression as R
from soma.config import M1, M_NANO
from soma.openended.archive import HOLDOUT
from soma.openended.tasks import generate, verify
from soma.utils.stats import scaled_accuracy

GENERALISES_ALONG = {
    "PARITY": "n_len",
    "MODCOUNT3": "n_len",
    "MODCOUNT5": "n_len",
    "PERM": "n_len",
    "DYCK": "depth",
    "MQAR": "P_mqar",
    "COPY": "n_len",
}
"""The parameter each family is scored *out of distribution* along (§08.2).

**[NORMATIVE — v3.1, audit A-88]** The check that each row lies outside the
archive's hold-out was written as **any-of**: a row passed if *any* hold-out
parameter was outside its band, and `n_len = 40` satisfied that for every row.
So the `DYCK` case the test's own docstring named — the archive training at the
very `depth` the suite scores — was vacuous, and `depth`'s band `(2, 8)` did
admit depth 8 while the suite scored depth 8. `P_mqar` was genuinely held out
and `depth` was not, and an any-of test cannot tell those apart."""


# §08.2's table, transcribed from the specification and never from the code.
EXPECTED = {
    "PARITY": (0.5, 0.95),
    "MODCOUNT3": (1 / 3, 0.80),
    "MODCOUNT5": (1 / 5, 0.80),
    "PERM": (1 / 120, 0.80),
    "DYCK": (0.023, 0.80),
    "MQAR": (1 / 32, 0.90),
    "COPY": (0.0, 0.95),
}


def test_the_suite_matches_the_specification_table() -> None:
    suite = R._suite(M1.eval_lengths)
    assert [s.name for s in suite] == list(EXPECTED), [s.name for s in suite]
    for spec in suite:
        acc_random, bar = EXPECTED[spec.name]
        assert spec.acc_random == pytest.approx(acc_random), spec.name
        assert spec.pass_bar == pytest.approx(bar), spec.name


def test_every_declared_chance_baseline_matches_the_generator() -> None:
    """§08.2 / audit A-87: `acc_random` is the best **constant answer**.

    `solve_rate` scores exact match on the whole answer, so the baseline is the
    modal answer's probability — not the chance of one token. Two declared
    values were wrong under that reading: `DYCK` at `0.5` (as if the answer were
    a balanced/unbalanced bit; it is the residual open-bracket stack, 339
    distinct values) and `COPY` at `1/alpha_sz` (the per-token chance; exact
    match is `alpha_sz ** -n_copy`).

    A wrong `r` moves the bar rather than the score: `scaled_acc = (acc - r) /
    (1 - r)`, so `DYCK`'s `0.80` bar demanded raw `0.90` instead of raw `0.805`.
    """
    import collections

    from soma.eval.regression import ACC_RANDOM_TOL
    from soma.openended.tasks import generate

    for spec in R._suite(M1.eval_lengths):
        lengths = spec.eval_lengths or (spec.params.n_len,)
        p = replace(spec.params, n_len=lengths[-1])
        rng = np.random.default_rng(20260828)
        counts = collections.Counter()
        for _ in range(2000):
            counts[tuple(generate(p, rng).answer)] += 1
        modal = max(counts.values()) / sum(counts.values())
        assert abs(modal - spec.acc_random) <= ACC_RANDOM_TOL, (
            f"{spec.name}: declared acc_random {spec.acc_random:.4f}, "
            f"measured modal-answer frequency {modal:.4f} over "
            f"{len(counts)} distinct answers"
        )


def test_every_evaluation_setting_lies_outside_the_archive_holdout() -> None:
    """§06.3.5 / audit A-30: the holdout covers every parameter the suite varies.

    A bar met at a setting the archive was allowed to train on measures memory,
    not generalisation — and `MQAR` and `DYCK` generalise along `P_mqar` and
    `depth`, not along length, which is the case v2 missed.
    """
    for cfg in (M1, M_NANO):
        for spec in R._suite(cfg.eval_lengths):
            lengths = spec.eval_lengths or (spec.params.n_len,)
            axis = GENERALISES_ALONG[spec.name]
            lo, hi = HOLDOUT[axis]
            for L in lengths:
                p = replace(spec.params, n_len=L)
                v = getattr(p, axis)
                assert not (lo <= v <= hi), (
                    f"{spec.name} at n_len={L} is scored at {axis}={v}, which "
                    f"the archive's holdout {HOLDOUT[axis]} ADMITS — so the "
                    f"archive may train at the exact setting the bar tests"
                )


def test_m_nano_uses_the_reduced_band_and_the_bars_are_unchanged() -> None:
    """§08.6 / audit A-32: `T_ctx = 128` cannot hold a length-256 prompt plus its
    answer, so M-nano gets a reduced band — but not easier bars."""
    assert max(M_NANO.eval_lengths) < max(M1.eval_lengths)
    nano = {s.name: s.pass_bar for s in R._suite(M_NANO.eval_lengths)}
    full = {s.name: s.pass_bar for s in R._suite(M1.eval_lengths)}
    assert nano == full


def test_scaled_accuracy_is_zero_at_chance_and_one_at_perfect() -> None:
    for _, (acc_random, _) in EXPECTED.items():
        assert scaled_accuracy(acc_random, acc_random) == pytest.approx(0.0)
        assert scaled_accuracy(1.0, acc_random) == pytest.approx(1.0)
        # Below chance is worse than nothing and must not read as progress —
        # for every family whose chance is above zero. `COPY`'s exact-match
        # chance is `alpha_sz ** -n_copy` and is declared 0 (audit A-87), so
        # there is nothing below it to be below.
        if acc_random > 0.0:
            assert scaled_accuracy(acc_random / 2, acc_random) < 0.0


# ---------------------------------------------------------------------------
# The harness, against a perfect decoder and a chance decoder
# ---------------------------------------------------------------------------


class _Oracle:
    """Stands in for `greedy_decode`, returning the instance's true answer."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model, prompt, max_new, **kw):
        self.calls += 1
        return list(model["answers"][tuple(prompt)])


class _Chance:
    """Returns a fixed wrong-length answer: nothing can verify."""

    def __call__(self, model, prompt, max_new, **kw):
        return [0]


@pytest.fixture
def patched(monkeypatch):
    def install(decoder):
        monkeypatch.setattr(R, "greedy_decode", decoder)
    return install


def _answer_table(cfg, seed: int = 0):
    """A dict from prompt to answer, covering the instances `run_suite` draws."""
    table: dict = {}

    def record(params, rng):
        inst = generate(params, rng)
        table[tuple(inst.prompt)] = inst.answer
        return inst

    return table, record


def test_a_perfect_decoder_scores_one_and_passes_every_bar(patched) -> None:
    """Positive control: without it, a scorer stuck at zero looks identical to a
    model that has learned nothing."""
    cfg = M_NANO
    table, _ = _answer_table(cfg)

    real_generate = R.generate

    def capturing(params, rng):
        inst = real_generate(params, rng)
        table[tuple(inst.prompt)] = inst.answer
        return inst

    patched(_Oracle())
    import soma.eval.regression as mod
    orig = mod.generate
    mod.generate = capturing
    try:
        rows = R.run_suite({"answers": table}, cfg, np.random.default_rng(1), n_eval=4)
    finally:
        mod.generate = orig

    assert rows, "the suite produced no rows"
    scored = [r for r in rows if not r.skipped]
    assert scored, "every row was skipped"
    for r in scored:
        assert r.accuracy == pytest.approx(1.0), r.as_dict()
        assert r.scaled == pytest.approx(1.0), r.as_dict()
        assert r.passed, r.as_dict()


def test_a_chance_decoder_fails_every_bar(patched) -> None:
    """Negative control: the bars are reachable *and* refusable."""
    patched(_Chance())
    rows = R.run_suite(None, M_NANO, np.random.default_rng(1), n_eval=4)
    assert rows
    scored = [r for r in rows if not r.skipped]
    assert scored, "every row was skipped"
    for r in scored:
        assert r.accuracy == pytest.approx(0.0), r.as_dict()
        assert not r.passed, r.as_dict()
    # A row that cannot be serialised is **present and marked**, never absent
    # (audit A-91): a downstream `all(r.passed)` gate must be able to see it.
    for r in rows:
        if r.skipped:
            assert not r.passed and r.n_eval == 0


def test_a_row_that_cannot_be_serialised_is_skipped_not_scored_zero(patched) -> None:
    """§08.2 / audit A-32: a row that does not fit `T_ctx` is **absent**, never a
    zero — a zero would read as a capability failure the model never had a
    chance to demonstrate."""
    patched(_Chance())
    tiny = replace(M_NANO, T_ctx=128, eval_lengths=(40, 64, 96))
    rows = R.run_suite(None, tiny, np.random.default_rng(1), n_eval=1)
    names = {r.name for r in rows}
    lengths = {(r.name, r.n_len) for r in rows}
    assert names, "every row was skipped, so the check is vacuous"
    assert all(n_len is None or n_len <= tiny.T_ctx for _, n_len in lengths)


def test_every_verifier_accepts_its_own_generated_answer() -> None:
    """The property the whole suite rests on: `verify(prompt, answer) == 1` for
    the answer `generate` produced. If this ever fails, a perfect model scores
    zero and every bar becomes unreachable."""
    rng = np.random.default_rng(7)
    for spec in R._suite(M_NANO.eval_lengths):
        for _ in range(8):
            inst = generate(spec.params, rng)
            assert verify(inst.prompt, list(inst.answer), spec.params) == 1, spec.name
