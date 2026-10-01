"""Task generators and verifiers against independent brute force.

SPEC: 06 §6.3.2
SPEC: 06 §6.3.3
SPEC: 06 §6.3.5

Gates `G-VERIFIER`, `G-RESERVED`, `G-ARCHIVE`.

**[NORMATIVE]** Every generator emits `(prompt, answer, lambda_task)` and every
verifier is a **total** function returning `{0,1}`.

Each family is checked against a *separately written* solver in the test, not
against the implementation's own. A verifier compared only to itself proves
nothing, and `s_task` — the archive's central quantity — would then measure the
generator rather than the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from soma.config import M1, M_NANO
from soma.openended.archive import (
    A_MAX,
    EPS_ARCH,
    HOLDOUT,
    N_EVAL,
    R_REF,
    TAU_ADMIT,
    TaskArchive,
    learn,
    priority,
)
from soma.openended.tasks import (
    BOS,
    EOS,
    EX,
    FAMILIES,
    GRID_TRANSFORMS,
    N_SYMBOLS,
    ROW,
    SEP,
    SYM_BASE,
    TaskParams,
    assert_no_task_identity,
    edit,
    family_lower_bounds,
    generate,
    serialised_length,
    solve,
    verify,
)
from soma.openended.tasks import _apply_transform, _parse_grids, _serialise_grid, sym


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# G-VERIFIER: every family against an independent implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_len", [1, 2, 3, 7, 12])
def test_parity_against_xor(n_len: int, rng) -> None:
    p = TaskParams("PARITY", n_len=n_len)
    for _ in range(60):
        inst = generate(p, rng)
        bits = [t - SYM_BASE for t in inst.prompt]
        assert inst.answer == [sym(bits.count(1) % 2)]
        assert verify(inst.prompt, inst.answer, p) == 1


@pytest.mark.parametrize("m", [2, 3, 5, 9])
def test_modcount_against_counting(m: int, rng) -> None:
    p = TaskParams("MODCOUNT", n_len=10, m_mod=m)
    for _ in range(60):
        inst = generate(p, rng)
        bits = [t - SYM_BASE for t in inst.prompt]
        assert inst.answer == [sym(bits.count(1) % m)]


@pytest.mark.parametrize("g", [3, 4, 5, 6])
def test_perm_against_composition(g: int, rng) -> None:
    p = TaskParams("PERM", p_grp=g, n_len=8)
    for _ in range(60):
        inst = generate(p, rng)
        s = [t - SYM_BASE for t in inst.prompt]
        perm = list(range(g))
        for k in range(0, len(s) - 1, 2):
            a, b = s[k], s[k + 1]
            if 0 <= a < g and 0 <= b < g:
                perm[a], perm[b] = perm[b], perm[a]
        assert inst.answer == [sym(v) for v in perm]


def test_dyck_against_a_stack_automaton(rng) -> None:
    p = TaskParams("DYCK", b_types=2, depth=4, n_len=12)
    for _ in range(200):
        inst = generate(p, rng)
        stack = []
        for t in inst.prompt:
            s = t - SYM_BASE
            if s % 2 == 0:
                stack.append(s // 2)
            else:
                stack.pop()
        expected = [sym(2 * b) for b in stack] if stack else [sym(N_SYMBOLS - 2)]
        assert inst.answer == expected


def test_dyck_reports_invalid_for_a_mismatched_close() -> None:
    """The verifier is **total**, so an invalid string has a defined answer."""
    p = TaskParams("DYCK", b_types=2, depth=4, n_len=2)
    assert solve([sym(0), sym(3)], p) == [sym(N_SYMBOLS - 1)]
    assert solve([sym(1)], p) == [sym(N_SYMBOLS - 1)]


def test_mqar_against_dict_lookup(rng) -> None:
    p = TaskParams("MQAR", P_mqar=8, alpha_sz=16, distract=4)
    for _ in range(100):
        inst = generate(p, rng)
        body, q = inst.prompt[:-1], inst.prompt[-1]
        table = {body[i]: body[i + 1] for i in range(0, len(body) - 1, 2)}
        assert inst.answer == [table.get(q, sym(N_SYMBOLS - 1))]


def test_copy_against_identity(rng) -> None:
    p = TaskParams("COPY", alpha_sz=8, n_len=10, gap=8)
    for _ in range(100):
        inst = generate(p, rng)
        assert inst.answer == inst.prompt[:10]


@pytest.mark.parametrize("tf", GRID_TRANSFORMS)
def test_grid_against_numpy(tf: str, rng) -> None:
    """§06.3.2's transform set, enumerated explicitly. Spec v1 left "transform
    class" undefined, making the family unimplementable."""
    p = TaskParams("GRID", h=3, w=4, tf=tf, n_ex=2)
    for _ in range(20):
        inst = generate(p, rng)
        grids = _parse_grids(inst.prompt, p)
        assert grids, "GRID prompt must parse into at least one grid"
        assert inst.answer == _serialise_grid(_apply_transform(grids[-1], p))


@pytest.mark.parametrize("family", FAMILIES)
def test_verifiers_are_total(family: str, rng) -> None:
    """Any candidate is admissible input, including wrong length, out-of-range
    symbols and the empty sequence. Totality is what makes `s_task` well defined
    for a model that has not yet learned to stop."""
    p = next(q for q in family_lower_bounds() if q.family == family)
    inst = generate(p, rng)
    for cand in ([], [0], [10**6], [-1], list(range(300)), [SYM_BASE] * 5):
        assert verify(inst.prompt, cand, p) in (0, 1)
    assert verify(inst.prompt, inst.answer, p) == 1


@pytest.mark.parametrize("family", FAMILIES)
def test_generate_and_solve_never_disagree(family: str, rng) -> None:
    """The answer is produced by `solve`, never independently — a class of bug
    that would silently make `s_task` a measurement of the generator."""
    p = next(q for q in family_lower_bounds() if q.family == family)
    for _ in range(50):
        inst = generate(p, rng)
        assert inst.answer == solve(inst.prompt, p)


# ---------------------------------------------------------------------------
# G-RESERVED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_serialisation_stays_in_the_reserved_block(family: str, rng) -> None:
    p = next(q for q in family_lower_bounds() if q.family == family)
    for _ in range(30):
        inst = generate(p, rng)
        assert all(0 <= t < 64 for t in inst.tokens())
        assert_no_task_identity(inst)


def test_loss_mask_covers_the_answer_span_only(rng) -> None:
    """§06.3.2: the loss is masked to the answer span."""
    p = TaskParams("PARITY", n_len=6)
    inst = generate(p, rng)
    toks, mask = inst.tokens(), inst.loss_mask()
    assert len(toks) == len(mask)
    assert not any(mask[: 1 + len(inst.prompt) + 1]), "prompt and SEP are not scored"
    assert sum(mask) == len(inst.answer) + 1, "the answer plus EOS is scored"
    assert toks[1 + len(inst.prompt)] == SEP


def test_no_family_marker_reaches_the_model(rng) -> None:
    """§08.7 rule 2: a **static assertion** that no task-identity feature reaches
    the model. Spec v1 required "the ablation confirming this", but an absent
    input cannot be ablated."""
    controls_by_family = {}
    for p in family_lower_bounds():
        inst = generate(p, rng)
        controls_by_family[p.family] = set(inst.tokens()) & {BOS, EOS, SEP, ROW, EX}
    # Only GRID uses the structural separators; every other family is
    # indistinguishable by control-symbol usage alone.
    non_grid = [v for k, v in controls_by_family.items() if k != "GRID"]
    assert all(v == {BOS, EOS, SEP} for v in non_grid)


def test_the_tokenizer_never_emits_the_reserved_block() -> None:
    """§06.3.2, with a planted positive control."""
    from soma.data.tokenizer import assert_reserved_block_unused, build_tokenizer

    docs = ["the cat sat on the mat and looked at the dog"] * 200
    tok = build_tokenizer(docs, 300)
    assert_reserved_block_unused(tok, docs[:20])

    class Planted:
        vocab_size = 300

        def encode(self, text):
            return [5, 100, 200]

    with pytest.raises(AssertionError, match="reserved block"):
        assert_reserved_block_unused(Planted(), ["x"])


# ---------------------------------------------------------------------------
# G-ARCHIVE
# ---------------------------------------------------------------------------


def test_seed_archive_creates_all_seven_families(rng) -> None:
    """§06.3.3 / audit A-24: v2 never initialised `Lambda`, so `ARCHIVE_STEP`'s
    first firing sampled a parent from an empty set."""
    arc = TaskArchive(T_ctx=M_NANO.T_ctx)
    arc.seed(lambda p, n: 0.5, rng)
    assert len(arc) == len(FAMILIES)
    assert {m.params.family for m in arc.members.values()} == set(FAMILIES)
    assert all(m.s_H == m.s_task for m in arc.members.values())


def test_seeds_are_admitted_unconditionally(rng) -> None:
    """At `LIFETIME` step 5 the model is at initialisation, so every generator
    reads `s_task ~ 0` and would fail `tau_admit` — leaving `Lambda` empty
    forever."""
    arc = TaskArchive(T_ctx=M_NANO.T_ctx)
    arc.seed(lambda p, n: 0.0, rng)
    assert len(arc) == len(FAMILIES)
    assert all(learn(m.s_task) <= TAU_ADMIT for m in arc.members.values())


def test_archive_step_on_an_empty_lambda_raises(rng) -> None:
    arc = TaskArchive(T_ctx=M_NANO.T_ctx)
    with pytest.raises(RuntimeError, match="SEED_ARCHIVE"):
        arc.step(lambda p, n: 0.5, rng, 1)


def test_a_mastered_archive_does_not_freeze(rng) -> None:
    """§06.3.3: v1 sampled purely proportional to `Pri`; as `s -> 1` every
    `Pri -> 0`, no parent can be sampled, and the archive freezes permanently."""
    arc = TaskArchive(T_ctx=M_NANO.T_ctx)
    arc.seed(lambda p, n: 1.0, rng)
    assert max(m.priority for m in arc.members.values()) == 0.0
    rec = arc.step(lambda p, n: 1.0, rng, 1)
    assert rec["sampled"] in ("uniform", "uniform-fallback", "priority")
    assert rec["outcome"] in (
        "inserted", "duplicate", "inadmissible", "edit_abandoned",
        "rejected_learnability",
    )


def test_tau_admit_excludes_the_near_vacuous_band() -> None:
    """§06.3.3: at `n_eval = 32`, `Learn(1/32) = 0.121 > 0.05`, so v1's gate
    admitted any generator with 1-31 successes out of 32."""
    assert learn(1 / N_EVAL) > 0.05
    assert learn(1 / N_EVAL) < TAU_ADMIT
    assert learn(0.08) < TAU_ADMIT
    assert learn(0.92) < TAU_ADMIT
    assert learn(0.5) > TAU_ADMIT
    assert learn(0.30) > TAU_ADMIT


def test_staleness_bound_is_achievable(rng) -> None:
    """§06.3.3: round-robin oldest-first, so every member is refreshed within
    `ceil(A_max / R_ref)` archive steps. v1's "random 5%" left a third of the
    archive stale at all times."""
    arc = TaskArchive(T_ctx=M_NANO.T_ctx)
    arc.seed(lambda p, n: 0.5, rng)
    for n in range(1, 120):
        arc.step(lambda p, s: float(np.clip(0.5 + rng.normal(0, 0.05), 0, 1)), rng, n)
    oldest = min(m.last_refreshed for m in arc.members.values())
    assert arc.n_steps - oldest <= -(-A_MAX // R_REF)


def test_archive_respects_every_holdout_and_the_context_clip(rng) -> None:
    """§06.3.5 / audit A-30: the holdout covers every parameter the suite varies,
    and `lambda`s serialising beyond `T_ctx - 2` are rejected."""
    arc = TaskArchive(T_ctx=M_NANO.T_ctx)
    arc.seed(lambda p, n: 0.5, rng)
    for n in range(1, 300):
        arc.step(lambda p, s: 0.5, rng, n)
    for m in arc.members.values():
        for name, (lo, hi) in HOLDOUT.items():
            assert lo <= getattr(m.params, name) <= hi, (name, m.params)
        assert serialised_length(m.params, rng) <= M_NANO.T_ctx - 2


def test_grid_8x8x4_is_rejected_at_both_configurations(rng) -> None:
    """The concrete instance audit A-30 names: `(2 n_ex + 2) h w = 640` symbols
    against `T_ctx = 128`."""
    p = TaskParams("GRID", h=8, w=8, tf="identity", n_ex=4)
    assert serialised_length(p, rng) > M_NANO.T_ctx
    assert serialised_length(p, rng) > M1.T_ctx
    for cfg in (M_NANO, M1):
        arc = TaskArchive(T_ctx=cfg.T_ctx)
        assert not arc._admissible_shape(p, rng)


def test_edit_perturbs_exactly_one_parameter_and_clamps(rng) -> None:
    """§06.3.3: exactly one parameter perturbed by one step, every value
    clamped, a no-op resampled up to 8 times then abandoned."""
    p = TaskParams("MQAR", P_mqar=2, alpha_sz=2, distract=0)
    changed_counts = []
    for _ in range(500):
        e = edit(p, rng)
        if e is None:
            continue
        diffs = [
            k for k in p.as_dict()
            if p.as_dict()[k] != e.as_dict()[k]
        ]
        changed_counts.append(len(diffs))
        assert HOLDOUT["P_mqar"][0] <= e.P_mqar <= HOLDOUT["P_mqar"][1]
    assert changed_counts and set(changed_counts) == {1}


def test_edit_only_touches_parameters_the_family_uses(rng) -> None:
    """Editing `m_mod` on a PARITY instance would produce a distinct archive key
    for an identical generator, filling `Lambda` with duplicates that
    `reject if lambda' already in Lambda` cannot see."""
    p = TaskParams("PARITY", n_len=8)
    for _ in range(200):
        e = edit(p, rng)
        if e is None:
            continue
        assert e.m_mod == p.m_mod and e.h == p.h and e.tf == p.tf
        assert e.n_len != p.n_len


def test_learn_and_priority_shapes() -> None:
    """(6.6)-(6.8): `Learn` is 0 at both ends, maximal at 1/2, bounded."""
    assert learn(0.0) == 0.0
    assert learn(1.0) == 0.0
    assert learn(0.5) == 1.0
    assert all(0.0 <= learn(s) <= 1.0 for s in np.linspace(0, 1, 101))
    assert priority(0.5, 0.5) == pytest.approx(EPS_ARCH)  # eta_imp == eps_arch == 0.1
    assert priority(0.5, 0.0) > priority(0.5, 0.5), "improvement must raise priority"
