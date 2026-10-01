"""The streams: window layouts, re-fetchability, the fixed sets, and determinism.

SPEC: 02 §2.9 (the seven streams), 08 §8.3 (lanes, boundaries, group rows, the fixed sets),
06 §6.9 (a window is re-fetchable by identity, for rehearsal and for the salience regression).

`tempo/streams.py` is the module the corrections rewrote most and the only one that had no
test file of its own. The invariants below are the ones the second adversarial round found
broken: a window that does not come back the same when re-fetched silently corrupts the
salience regression (which subtracts a re-scored loss from a recorded one), and a supervised
row that is not teacher-forced is not the likelihood of anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma.openended.tasks import EOS
from tempo.config import TINY
from tempo.streams import (
    ANSWER_BUDGET_SPANS, EVAL_SEED_BASE, FITNESS_OFFSET, TOKEN_MODALITIES, _eval_seed,
    build_streams, scored_tokens,
)

cfg = TINY
_rng = np.random.default_rng(0)
TOKENS = _rng.integers(8, cfg.vocab, size=20_000).astype(np.uint16)
EVAL = _rng.integers(0, 20_000 - cfg.T - cfg.K, size=8)
PASSIVE = ("text", "pattern", "interleave")


EVAL_ROWS = _rng.integers(8, cfg.vocab, size=(8, cfg.T + cfg.K)).astype(np.int64)
FIT_ROWS = _rng.integers(8, cfg.vocab, size=(8, cfg.T + cfg.K)).astype(np.int64)


@pytest.fixture()
def streams():
    """As production builds them: the text stream gets both fixed sets as explicit rows (the
    v3 test and fitness documents), not an offset list."""
    return build_streams(cfg, TOKENS, eval_tokens=EVAL_ROWS, fitness_tokens=FIT_ROWS,
                         seed=0, G=2, control_horizon=16)


def _same_window(a, b) -> bool:
    """Two windows are the same when their inputs, targets and score mask agree.

    `t_w` is compared **when both windows carry one**: 02 (2.13) makes the time code a Fourier
    transform of the event's absolute wall-clock time, so two windows at different `t_w` are
    different inputs however identical their content. An earlier docstring waived the comparison
    on the grounds that the code is "a function of the window's own offsets", which is false;
    the waiver was removed once `fetch` learned to take a `t0` (11 §P-D.12). A window re-fetched
    without a clock legitimately starts at 0, so the comparison is skipped in that case only.
    """
    if a.ident != b.ident or set(a.inputs) != set(b.inputs) or set(a.targets) != set(b.targets):
        return False
    if not np.array_equal(a.score_mask, b.score_mask) or not np.array_equal(a.mod, b.mod):
        return False
    if float(a.t_w[0]) > 0.0 and float(b.t_w[0]) > 0.0 and not np.array_equal(a.t_w, b.t_w):
        return False
    for m in a.inputs:
        pa, ra = a.inputs[m]
        pb, rb = b.inputs[m]
        if not np.array_equal(pa, pb) or not torch.equal(ra, rb):
            return False
    for m in a.targets:
        pa, ra = a.targets[m]
        pb, rb = b.targets[m]
        if not np.array_equal(pa, pb) or not torch.equal(ra, rb):
            return False
    return True


class TestRefetchability:
    """06 §6.9: a window must come back exactly from its identity."""

    @pytest.mark.parametrize("name", PASSIVE)
    def test_a_passive_lane_window_refetches_exactly(self, streams, name):
        st = streams[name]
        cur = st.cursor(0, 11)
        for _ in range(3):
            w = st.next_windows(cur, None, None, torch.Generator())[0]
            assert _same_window(w, st.fetch(w.ident)), (name, w.ident)

    def test_the_pattern_stream_refetches_its_first_window_too(self, streams):
        """The defect: window `k = 0` was produced by the generator's own run and re-fetched by
        continuing from a float32-rounded cached state. On a chaotic map those are different
        trajectories, and only the *first* window of every lane was affected."""
        st = streams["pattern"]
        cur = st.cursor(0, 5)
        first = st.next_windows(cur, None, None, torch.Generator())[0]
        assert first.ident[3] == 0
        again = st.fetch(first.ident)
        pa = first.inputs[first.ident[1]][1]
        pb = again.inputs[again.ident[1]][1]
        assert torch.equal(pa, pb), float((pa - pb).abs().max())

    def test_a_pattern_lane_continues_and_every_window_refetches(self, streams):
        st = streams["pattern"]
        cur = st.cursor(0, 5)
        ws = [st.next_windows(cur, None, None, torch.Generator())[0] for _ in range(4)]
        assert [w.ident[3] for w in ws] == [0, 1, 2, 3]
        for w in ws:
            assert _same_window(w, st.fetch(w.ident)), w.ident
        # the trajectory is continuous: a window's last observation is the next one's first
        kind = ws[0].ident[1]
        for a, b in zip(ws[:-1], ws[1:]):
            assert torch.allclose(a.targets[kind][1][-1], b.inputs[kind][1][0], atol=1e-6)


class TestSupervisedRowsAreTeacherForced:
    """08 §8.3: the truth is the input at the answer positions of a supervised row."""

    def test_task(self, streams):
        st = streams["task"]
        w = st.eval_windows(1)[0]
        inp = w.inputs["task"][1]
        tpos, tgt = w.targets["task"]
        for j, p in enumerate(tpos):
            if not w.score_mask[p]:
                continue
            assert torch.equal(inp[p + 1], tgt[j]) or p + 1 >= inp.shape[0]

    def test_speech_transcription(self, streams):
        """The defect: transcription's supervised row passed an empty answer, so the truth was
        never an input and half its scored targets were predicted from a padded zero frame."""
        st = streams["speech"]
        w = st.fetch(("speech", _eval_seed("speech", 0, fitness=True), "trans"))
        assert "task" in w.inputs
        task_in_pos, task_in = w.inputs["task"]
        tpos, tgt = w.targets["task"]
        scored = [int(p) for p in tpos if w.score_mask[p]]
        assert scored, "the transcription row must score something"
        # every scored target position is followed by an input carrying that same truth span
        pos_to_raw = {int(p): task_in[i] for i, p in enumerate(task_in_pos)}
        for j, p in enumerate(tpos):
            if not w.score_mask[p] or int(p) + 1 not in pos_to_raw:
                continue
            assert torch.equal(pos_to_raw[int(p) + 1], tgt[j]), int(p)

    def test_a_rollout_row_carries_no_supervised_target(self, streams):
        for name in ("task", "vision", "speech"):
            st = streams[name]
            cur = st.cursor(0, 3)

            class Dummy:
                def perceive(self, lane, raw, m, t):
                    return torch.zeros(cfg.d_x)

                def decode_text(self, y, g, *, greedy=False):
                    return torch.full((cfg.K,), EOS, dtype=torch.long)

                def decode_head(self, y, h, g, *, greedy=False):
                    return torch.zeros(64)

                def fork(self, lane):
                    return (None, None, True)

                def restore(self, lane, snap):
                    lane.S, lane.cache, lane.boundary_pending = None, None, True

                def state_of(self, lane):
                    return None

            from tempo.life import Lane
            ws = st.next_windows(cur, Dummy(), Lane(0, name, cur, None, None, True), torch.Generator())
            sup = [w for w in ws if w.rollout == 0]
            roll = [w for w in ws if w.rollout > 0]
            assert len(sup) == 1 and roll
            assert sup[0].rl is None and sup[0].targets
            for w in roll:
                assert not w.targets and w.n_tokens == 0


class TestTokenCounts:
    """The batch denominator is built from these, so they must be what the loss sums over."""

    @pytest.mark.parametrize("name", ("text", "task", "vision", "speech", "pattern", "interleave"))
    def test_n_tokens_equals_the_scored_text_tokens(self, streams, name):
        st = streams[name]
        w = st.eval_windows(1)[0]
        expected = sum(
            int(np.asarray([w.score_mask[p] for p in pos]).sum()) * cfg.K
            for m, (pos, _) in w.targets.items() if m in TOKEN_MODALITIES and len(pos)
        )
        assert w.n_tokens == expected, (name, w.n_tokens, expected)

    def test_scored_tokens_counts_every_modality(self, streams):
        w = streams["vision"].eval_windows(1)[0]
        total = scored_tokens(w, cfg.K)
        assert total >= w.n_tokens
        assert total > 0

    def test_a_control_window_scores_nothing_supervised(self, streams):
        w = streams["control"].eval_windows(1)[0]
        assert scored_tokens(w, cfg.K) == 0 and w.n_tokens == 0
        assert w.rl is not None                       # it carries a return instead


class TestFixedSets:
    """10 §10.1: two disjoint fixed sets per stream, deterministic and seed-independent."""

    @pytest.mark.parametrize("name", ("text", "task", "control", "vision", "speech", "pattern", "interleave"))
    def test_evaluation_and_fitness_are_disjoint_and_deterministic(self, streams, name):
        st = streams[name]
        ev = [w.ident for w in st.eval_windows(6)]
        fit = [w.ident for w in st.fitness_windows(6)]
        assert len(ev) == len(fit) == 6
        assert not (set(ev) & set(fit)), name
        assert [w.ident for w in st.eval_windows(6)] == ev

    def test_the_sets_do_not_depend_on_the_model_seed(self):
        a = build_streams(cfg, TOKENS, eval_tokens=EVAL_ROWS, fitness_tokens=FIT_ROWS, seed=0, G=2, control_horizon=16)
        b = build_streams(cfg, TOKENS, eval_tokens=EVAL_ROWS, fitness_tokens=FIT_ROWS, seed=99, G=2, control_horizon=16)
        for name in ("task", "control", "vision", "speech", "pattern"):
            assert [w.ident for w in a[name].eval_windows(4)] == [w.ident for w in b[name].eval_windows(4)], name

    def test_the_seed_layout_has_headroom(self):
        """A stream's evaluation seeds must not reach into its own fitness block."""
        assert _eval_seed("text", 0) + 400 < _eval_seed("text", 0, fitness=True)
        assert _eval_seed("text", 0, fitness=True) < _eval_seed("task", 0)


class TestCRN:
    """10 §10.1: arms sharing an order seed read the same windows, whatever their model seed."""

    @pytest.mark.parametrize("name", ("text", "task", "control", "vision", "speech", "pattern", "interleave"))
    def test_the_window_order_is_a_function_of_the_order_seed(self, name):
        a = build_streams(cfg, TOKENS, eval_tokens=EVAL_ROWS, fitness_tokens=FIT_ROWS, seed=0, G=2, control_horizon=16)[name]
        b = build_streams(cfg, TOKENS, eval_tokens=EVAL_ROWS, fitness_tokens=FIT_ROWS, seed=7, G=2, control_horizon=16)[name]
        ca, cb = a.cursor(2, 1234), b.cursor(2, 1234)
        for _ in range(3):
            wa = a.next_windows(ca, None, None, torch.Generator())
            wb = b.next_windows(cb, None, None, torch.Generator())
            assert [w.ident for w in wa] == [w.ident for w in wb], name


class TestAnswerBudget:
    def test_it_does_not_depend_on_the_truth(self, streams):
        """The budget must be a constant: deriving it from `len(inst.answer)` lets the truth's
        length decide how long the agent may speak in the rollout that scores it."""
        st = streams["task"]
        import inspect
        src = inspect.getsource(st.act_group)
        assert "len(insts" not in src and "inst.answer" not in src
        assert "ANSWER_BUDGET_SPANS" in src and ANSWER_BUDGET_SPANS > 0


class TestTimeCodeSurvivesARefetch:
    """02 (2.13) [NORMATIVE]: the time code is `τ(t_w)` over periods `2^-6 … 2^7` seconds, and
    `t_w` is the event's wall-clock time. `Perception.forward` adds it to every span
    (`modalities.py:141`), and `life.py:231` feeds it straight from `w.t_w`, so two windows
    with the same content at different `t_w` are **different inputs**.

    Every `fetch` hardcodes `t0 = 0.0`. That breaks two things:

    * 06 (6.11)'s salience target `Δℓ_w = ℓ_w^{(e)} − ℓ_w^{(e+1,end)}` is only the *model's*
      improvement if the window is identical at both scorings. It is not: the first is recorded
      live at the true `t_w`, the second re-scored from a fetch at `t = 0`.
    * `Rehearsal` stores only idents, so the 5 % of each episode spent replaying presents its
      windows with a time code of ≈ 0 instead of their own.

    Marked `xfail(strict=True)`, so it fails loudly the moment `fetch` learns to reproduce
    `t_w` and the correction has to be recorded rather than absorbed.
    """

    def test_a_text_window_refetches_at_its_own_wall_clock_time(self, streams):
        """Corrected 11 §P-D.12: `fetch` takes the `t0` the window had, so a re-fetch given that
        clock reproduces `t_w` exactly. A re-fetch given no clock still starts at 0 -- that is
        the fixed evaluation sets' case, which has no live counterpart to match."""
        st = streams["text"]
        cur = st.cursor(0, 11)
        ws = [st.next_windows(cur, None, None, torch.Generator())[0] for _ in range(4)]
        live = ws[-1]
        assert float(live.t_w[0]) > 0.0, "the fixture must reach a non-zero wall clock"
        again = st.fetch(live.ident, t0=float(live.t_w[0]))
        assert np.array_equal(live.t_w, again.t_w)
        assert _same_window(live, again)
        assert float(st.fetch(live.ident).t_w[0]) == 0.0

    def test_the_time_code_actually_differs_at_those_two_times(self, streams):
        """Not an xfail: the damage is real only if `τ(0)` and `τ(t_w)` are far apart. The
        shortest period is 2^-6 s and the longest 2^7 s, so a window hundreds of seconds into a
        life shares no phase with the same window at zero."""
        from tempo.modalities import TimeCode

        st = streams["text"]
        cur = st.cursor(0, 11)
        ws = [st.next_windows(cur, None, None, torch.Generator())[0] for _ in range(4)]
        t_live = torch.tensor(ws[-1].t_w, dtype=torch.float64)
        assert float(t_live[0]) > 0.0
        tc = TimeCode(cfg.d_x)
        f_live = tc.features(t_live)
        f_zero = tc.features(torch.zeros_like(t_live))
        # cosine similarity of the 28-dimensional feature rows, averaged over the window
        num = (f_live * f_zero).sum(-1)
        den = f_live.norm(dim=-1) * f_zero.norm(dim=-1)
        cos = float((num / den.clamp_min(1e-12)).mean())
        assert cos < 0.9, f"the two time codes are nearly identical (cos {cos:.3f}); no defect"
