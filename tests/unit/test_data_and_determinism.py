"""Tokenizer, corpus packing, the stream, and determinism.

SPEC: 06 §6.1.1
SPEC: 06 §6.2
SPEC: 06 §6.3.4
SPEC: 07 §7.7

Gates `G-DETERMINISM`, `G-RESUME`, `G-STREAM`.

**[NORMATIVE]** One RNG stream **per purpose**, all seeded from the run seed and
recorded. Spec v1 mandated separation in one bullet and shared a generator
between lexicase and archive in the next — under which an unrelated change to
one mechanism's call count silently changes the other's trajectory.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from soma.config import M_NANO
from soma.data import (
    TokenizedCorpus,
    TrainingStream,
    build_tokenizer,
    corpus_hash,
    f_oe_at,
    tokenize_corpus,
)
from soma.openended import TaskArchive
from soma.openended.references import (
    LaggedSelf,
    RefCacheBinding,
    RefCacheMismatch,
    ReferenceCache,
)
from soma.openended.tokensel import ExploreState, select_tokens
from soma.utils.rng import PURPOSES, RngBundle, derive_seed

CFG = M_NANO
DOCS = [
    "Once upon a time there was a little girl named Lily who liked the park.",
    "Tom and Sara went to the shop and bought a red ball and a blue kite.",
    "The cat sat on the mat and looked at the dog, and the dog barked.",
] * 200


@pytest.fixture(scope="module")
def corpus_dir():
    td = tempfile.mkdtemp()
    tok = build_tokenizer(DOCS, 400, out_path=os.path.join(td, "tok.json"))
    tokenize_corpus(DOCS, tok, td, "train", progress_every=0)
    yield td, tok


# ---------------------------------------------------------------------------
# RNG streams
# ---------------------------------------------------------------------------


def test_every_purpose_has_its_own_stream() -> None:
    b = RngBundle.create(1234)
    seeds = b.seeds()
    assert set(seeds) == set(PURPOSES)
    assert len(set(seeds.values())) == len(PURPOSES), "stream seeds must differ"


def test_neighbouring_run_seeds_give_uncorrelated_streams() -> None:
    """BLAKE2b derivation rather than arithmetic: `run_seed` and `run_seed + 1`
    would otherwise give `data` streams offset by one."""
    a = derive_seed(1000, "data")
    b = derive_seed(1001, "data")
    assert abs(a - b) > 1_000_000


def test_streams_are_independent_of_each_others_call_counts() -> None:
    """The property spec v1's shared generator destroyed."""
    b1 = RngBundle.create(7)
    b2 = RngBundle.create(7)
    for _ in range(100):
        b1.np("lexicase").random()
    assert b1.np("archive").random() == b2.np("archive").random()


def test_rng_state_round_trips() -> None:
    b = RngBundle.create(11)
    for _ in range(5):
        b.np("data").random()
        torch.rand(3, generator=b.pt("init"))
    state = b.state()
    expected_np = b.np("data").random()
    expected_pt = torch.rand(3, generator=b.pt("init"))

    b2 = RngBundle.create(11)
    b2.load_state(state)
    assert b2.np("data").random() == expected_np
    assert torch.equal(torch.rand(3, generator=b2.pt("init")), expected_pt)


def test_load_state_refuses_a_different_run_seed() -> None:
    b = RngBundle.create(1)
    with pytest.raises(ValueError, match="run seed mismatch"):
        RngBundle.create(2).load_state(b.state())


def test_model_construction_is_a_function_of_the_seed_alone() -> None:
    """§02.6.3 / audit A-64: `nn.Linear` and `nn.init.normal_` take no generator,
    so module construction draws from the **process-global** stream that
    `RngBundle` deliberately never touches.

    The test is two-sided on purpose. The positive half — same seed, identical
    weights — passes trivially in a process whose global stream happens not to
    have moved, so it proves nothing on its own. The negative half runs the
    identical construction with the global stream deliberately advanced and
    requires the weights to be **the same anyway**; and the control below shows
    that without the seeding call they are not. Without the control this gate
    cannot fail.
    """
    from soma.model.model import SomaModel
    from soma.utils.rng import RngBundle

    def build(seed: int, *, disturb: int = 0) -> torch.Tensor:
        rngs = RngBundle.create(seed, device="cpu")
        if disturb:
            torch.randn(disturb)          # move the global stream
        rngs.seed_global("init")
        m = SomaModel(M_NANO)
        return torch.cat([p.detach().reshape(-1) for p in m.parameters()])

    a0 = build(11)
    a1 = build(11, disturb=997)
    assert torch.equal(a0, a1), "same seed gave different weights"

    b0 = build(12)
    assert not torch.equal(a0, b0), "different seeds gave identical weights"

    # Positive control: the disturbance is real, and *without* the seeding call
    # it changes the model. If this assertion ever fails, the two above have
    # stopped measuring anything.
    def build_unseeded(disturb: int) -> torch.Tensor:
        torch.manual_seed(4242)
        if disturb:
            torch.randn(disturb)
        m = SomaModel(M_NANO)
        return torch.cat([p.detach().reshape(-1) for p in m.parameters()])

    assert not torch.equal(build_unseeded(0), build_unseeded(997)), (
        "the positive control did not perturb construction, so the test above "
        "would pass even with the seeding removed"
    )


def test_unknown_purpose_is_refused() -> None:
    b = RngBundle.create(1)
    with pytest.raises(KeyError):
        b.np("not_a_purpose")


# ---------------------------------------------------------------------------
# Corpus and stream
# ---------------------------------------------------------------------------


def test_corpus_index_and_gather(corpus_dir) -> None:
    td, tok = corpus_dir
    tc = TokenizedCorpus(td, "train")
    assert len(tc) == len(DOCS)
    w = tc.windows(CFG.T_ctx)
    assert w.shape[1] == 2
    toks, docs, valid, flat = tc.gather(w[:8], CFG.T_ctx)
    assert toks.shape == (8, CFG.T_ctx)
    assert docs[:, 0].tolist() == w[:8, 0].tolist()
    assert (flat[valid] >= 0).all()
    tc.close()


def test_a_window_never_crosses_a_document(corpus_dir) -> None:
    """§06.1.1's `doc_boundary_policy = "masked"`: a token's conditioning
    depends only on its own document, which is what makes the `L_ref` cache
    valid."""
    td, tok = corpus_dir
    tc = TokenizedCorpus(td, "train")
    w = tc.windows(CFG.T_ctx)
    _, docs, _, _ = tc.gather(w[:32], CFG.T_ctx)
    assert (docs == docs[:, :1]).all(), "a row must belong to exactly one document"
    tc.close()


def test_flat_index_is_document_keyed_and_shuffle_invariant(corpus_dir) -> None:
    """The `L_ref` sidecar is addressed by `(doc_id, position_in_doc)`, so
    §07.3's data-order jitter reorders *windows* and changes no token's key."""
    td, tok = corpus_dir
    tc = TokenizedCorpus(td, "train")
    w = tc.windows(CFG.T_ctx)
    a = tc.gather(w[5:6], CFG.T_ctx)[3]
    b = tc.gather(w[5:6], CFG.T_ctx)[3]
    assert (a == b).all()
    # A different window of the same document has different keys, and the keys
    # depend only on the document, never on batch position.
    c = tc.gather(w[6:7], CFG.T_ctx)[3]
    assert not (a == c).all()
    tc.close()


def test_stream_mixes_and_anneals_f_oe(corpus_dir) -> None:
    """§06.3.4: `f_oe` anneals to zero across the LR decay phase."""
    td, tok = corpus_dir
    tc = TokenizedCorpus(td, "train")
    rng = np.random.default_rng(0)
    arc = TaskArchive(T_ctx=CFG.T_ctx)
    arc.seed(lambda p, n: 0.5, rng)
    st = TrainingStream(CFG, tc, arc, rows_per_batch=64, rng=rng)

    assert f_oe_at(0, CFG, 0.25) == 0.25
    assert f_oe_at(CFG.n_dec, CFG, 0.25) == pytest.approx(0.25)
    assert f_oe_at(CFG.N_step - 1, CFG, 0.25) < 0.01

    early = st.batch(0, 0.25)
    late = st.batch(CFG.N_step - 1, 0.25)
    assert early.n_synthetic_rows > 0

    # `n_syn ~ Binomial(rows, f_oe)`, so the late count is a **draw**, not a
    # constant: at `f_oe < 0.01` over 64 rows the mean is 0.64 and a 1 is
    # ordinary. Asserting `== 0` made the test a statement about the RNG
    # stream's position — it broke when an unrelated change to a task
    # generator shifted the shared stream by a few draws (audit A-88).
    # The property the anneal actually guarantees is the one asserted here.
    assert late.n_synthetic_rows < early.n_synthetic_rows
    assert late.n_synthetic_rows <= 3, late.n_synthetic_rows
    tail = [st.batch(CFG.N_step - 1, 0.25).n_synthetic_rows for _ in range(40)]
    assert sum(tail) / len(tail) < 0.05 * 64, sum(tail) / len(tail)
    tc.close()


def test_synthetic_rows_score_only_the_answer_span(corpus_dir) -> None:
    """§06.3.2: the loss is masked to the answer span."""
    td, tok = corpus_dir
    tc = TokenizedCorpus(td, "train")
    rng = np.random.default_rng(1)
    arc = TaskArchive(T_ctx=CFG.T_ctx)
    arc.seed(lambda p, n: 0.5, rng)
    st = TrainingStream(CFG, tc, arc, rows_per_batch=64, rng=rng)
    b = st.batch(0, 0.5)
    syn = b.is_synthetic
    assert int(syn.any(dim=1).sum()) > 0
    frac_syn = float((b.scored & syn).sum()) / max(1, float((b.valid & syn).sum()))
    frac_nat = float((b.scored & ~syn).sum()) / max(1, float((b.valid & ~syn).sum()))
    assert frac_syn < 0.6, "a synthetic row must not score its prompt"
    assert frac_nat > 0.85
    tc.close()


def test_synthetic_doc_ids_cannot_collide_with_natural_ones(corpus_dir) -> None:
    td, tok = corpus_dir
    tc = TokenizedCorpus(td, "train")
    rng = np.random.default_rng(2)
    arc = TaskArchive(T_ctx=CFG.T_ctx)
    arc.seed(lambda p, n: 0.5, rng)
    st = TrainingStream(CFG, tc, arc, rows_per_batch=32, rng=rng)
    b = st.batch(0, 0.5)
    assert bool((b.doc_id[b.is_synthetic] < 0).all())
    assert bool((b.doc_id[~b.is_synthetic] >= 0).all())
    tc.close()


def test_stream_checkpoint_resume_is_exact(corpus_dir) -> None:
    """§07.7: the streaming dataloader checkpoints its position."""
    td, tok = corpus_dir
    tc = TokenizedCorpus(td, "train")
    s1 = TrainingStream(CFG, tc, None, rows_per_batch=8, rng=np.random.default_rng(42))
    for _ in range(3):
        s1.batch(0, 0.0)
    snapshot = s1.state_dict()
    expected = s1.batch(0, 0.0)

    s2 = TrainingStream(CFG, tc, None, rows_per_batch=8, rng=np.random.default_rng(42))
    s2.load_state_dict(snapshot)
    got = s2.batch(0, 0.0)
    assert torch.equal(expected.tokens, got.tokens)
    assert torch.equal(expected.doc_id, got.doc_id)
    tc.close()


def test_two_runs_with_the_same_seed_agree(corpus_dir) -> None:
    """`G-DETERMINISM`."""
    td, tok = corpus_dir
    tc = TokenizedCorpus(td, "train")
    outs = []
    for _ in range(2):
        st = TrainingStream(CFG, tc, None, rows_per_batch=8, rng=np.random.default_rng(9))
        outs.append(torch.cat([st.batch(i, 0.0).tokens for i in range(4)]))
    assert torch.equal(outs[0], outs[1])
    tc.close()


def test_different_seeds_give_different_data_order(corpus_dir) -> None:
    """§07.3's jitter is applied to the **data order**, not the budget, so two
    seeds differ in order without differing in which windows exist."""
    td, tok = corpus_dir
    tc = TokenizedCorpus(td, "train")
    a = TrainingStream(CFG, tc, None, rows_per_batch=8, rng=np.random.default_rng(1))
    b = TrainingStream(CFG, tc, None, rows_per_batch=8, rng=np.random.default_rng(2))
    ta = torch.cat([a.batch(i, 0.0).tokens for i in range(4)])
    tb = torch.cat([b.batch(i, 0.0).tokens for i in range(4)])
    assert not torch.equal(ta, tb)
    assert a._windows.shape == b._windows.shape
    tc.close()


def test_corpus_hash_is_stable_and_content_sensitive() -> None:
    assert corpus_hash(DOCS) == corpus_hash(DOCS)
    assert corpus_hash(DOCS) != corpus_hash(DOCS[::-1])


# ---------------------------------------------------------------------------
# References (§06.1)
# ---------------------------------------------------------------------------


def test_ref_cache_binding_refuses_every_mismatch() -> None:
    """§06.1.1 / audit A-27."""
    with tempfile.TemporaryDirectory() as td:
        b = RefCacheBinding("tok1", "corp1", "m0-abc", 512)
        c = ReferenceCache(os.path.join(td, "lref.f16"), b)
        c.save(np.random.rand(1000))
        with ReferenceCache(os.path.join(td, "lref.f16"), b):
            pass
        for bad in (
            RefCacheBinding("tok2", "corp1", "m0-abc", 512),
            RefCacheBinding("tok1", "corp2", "m0-abc", 512),
            RefCacheBinding("tok1", "corp1", "m0-xyz", 512),
            RefCacheBinding("tok1", "corp1", "m0-abc", 128),
            RefCacheBinding("tok1", "corp1", "m0-abc", 512, packing_scheme="flat"),
            RefCacheBinding("tok1", "corp1", "m0-abc", 512, doc_boundary_policy="none"),
        ):
            with pytest.raises(RefCacheMismatch):
                ReferenceCache(os.path.join(td, "lref.f16"), bad).load()


def test_ref_cache_lookup_returns_float32() -> None:
    """§05.9: token-selection excess losses are accumulated in `float32`."""
    with tempfile.TemporaryDirectory() as td:
        b = RefCacheBinding("t", "c", "m", 128)
        c = ReferenceCache(os.path.join(td, "l.f16"), b)
        c.save(np.arange(100, dtype=np.float32))
        got = c.lookup(np.array([0, 5, 99]))
        assert got.dtype == torch.float32
        assert got.tolist() == [0.0, 5.0, 99.0]
        c.close()


def test_lagged_self_half_life_matches_6_2() -> None:
    """(6.2): `kappa` is per **update**, not per step."""
    from soma.model.model import SomaModel

    torch.manual_seed(0)
    model = SomaModel(CFG, dense_sites=True)
    kappa = LaggedSelf.kappa_from_half_life(CFG.T_ref, CFG.h_lag)
    lag = LaggedSelf(model, kappa, CFG.T_ref)
    ref = model.E_emb.weight.detach().clone()

    with torch.no_grad():
        model.E_emb.weight.add_(1.0)
    n_updates = int(round(CFG.h_lag / CFG.T_ref))
    for _ in range(n_updates):
        lag.update(model)

    shadow = lag._shadow["E_emb.weight"].float()
    moved = (shadow - ref).abs().mean() / 1.0
    assert float(moved) == pytest.approx(0.5, abs=0.05), (
        "after one half-life of updates the shadow must have moved half way"
    )


def test_lagged_self_load_into_round_trips() -> None:
    from soma.model.model import SomaModel

    torch.manual_seed(1)
    a = SomaModel(CFG, dense_sites=True)
    b = SomaModel(CFG, dense_sites=True)
    lag = LaggedSelf(a, 0.9, 32)
    lag.load_into(b, force=True)
    x = torch.randint(0, CFG.V, (1, 8))
    ya = a(x).logits
    yb = b(x).logits
    # §06.1.2 stores the shadow in `bfloat16` (~8 mantissa bits, 0.4% relative),
    # and the error compounds over L blocks. The assertion is therefore on the
    # RELATIVE error against the logit scale, not an absolute bound: an absolute
    # bound would be a statement about this configuration's logit magnitude
    # rather than about `load_into`.
    rel = float((ya - yb).abs().max()) / float(ya.abs().max())
    assert rel < 0.05, f"load_into round trip relative error {rel:.4f}"


def test_lagged_self_rejects_a_degenerate_kappa() -> None:
    from soma.model.model import SomaModel

    model = SomaModel(CFG, dense_sites=True)
    with pytest.raises(ValueError, match="kappa_lag"):
        LaggedSelf(model, 1.0, 32)


# ---------------------------------------------------------------------------
# Token selection (§06.2)
# ---------------------------------------------------------------------------


def test_within_source_ranking_gives_each_source_its_own_fraction() -> None:
    """§06.2: spec v1 ranked jointly, so synthetic tokens would have crowded out
    natural ones and confounded A5 with A6."""
    torch.manual_seed(0)
    B, T = 4, 64
    loss = torch.rand(B, T) * 3
    ref = torch.rand(B, T) * 3
    syn = torch.zeros(B, T, dtype=torch.bool)
    syn[:, :20] = True
    scored = torch.ones(B, T, dtype=torch.bool)
    sel = select_tokens(loss, ref, syn, scored, 0.65, ExploreState(V=256, p_explore=0.0))
    nat_n = int((scored & ~syn).sum())
    syn_n = int((scored & syn).sum())
    assert int((sel.include & ~syn).sum()) / nat_n == pytest.approx(0.65, abs=0.02)
    assert int((sel.include & syn).sum()) / syn_n == pytest.approx(0.65, abs=0.02)


def test_p_explore_adds_tokens_outside_the_top_fraction() -> None:
    torch.manual_seed(1)
    B, T = 4, 64
    loss = torch.rand(B, T)
    scored = torch.ones(B, T, dtype=torch.bool)
    syn = torch.zeros(B, T, dtype=torch.bool)
    base = select_tokens(loss, None, syn, scored, 0.65, ExploreState(V=256, p_explore=0.0))
    more = select_tokens(
        loss, None, syn, scored, 0.65, ExploreState(V=256, p_explore=0.2),
        generator=torch.Generator().manual_seed(1),
    )
    assert more.n_selected > base.n_selected
    assert more.n_explore > 0


def test_degeneracy_rule_is_two_sided_and_occurrence_conditioned() -> None:
    """§06.2 / audit A-40: v2's rule could only ever raise `p_explore`, and
    triggered on a quantity dominated by items that never occur."""
    torch.manual_seed(2)
    toks = torch.randint(0, 64, (200, 32))

    healthy = ExploreState(V=64, p_explore=0.10)
    healthy.observe(toks, torch.rand(200, 32) < 0.65)
    p_down = healthy.adjust(0.65)
    assert p_down < 0.10 and healthy.last_f_starved < 0.10

    starved = ExploreState(V=64, p_explore=0.05)
    inc = torch.rand(200, 32) < 0.65
    inc[toks < 10] = False
    starved.observe(toks, inc)
    p_up = starved.adjust(0.65)
    assert p_up > 0.05 and starved.last_f_starved > 0.10

    floor = ExploreState(V=64, p_explore=0.05)
    floor.observe(toks, torch.rand(200, 32) < 0.65)
    assert floor.adjust(0.65) == pytest.approx(0.05), "p_explore has a floor"

    ceil = ExploreState(V=64, p_explore=0.20)
    inc2 = torch.rand(200, 32) < 0.65
    inc2[toks < 20] = False
    ceil.observe(toks, inc2)
    assert ceil.adjust(0.65) == pytest.approx(0.20), "and a ceiling"


def test_normaliser_is_the_realised_count() -> None:
    """§06.2: the normaliser is `sum_i I_i`, not `N_tok * k_tok`."""
    from soma.openended.tokensel import masked_cross_entropy

    torch.manual_seed(3)
    logits = torch.randn(2, 8, 16)
    targets = torch.randint(0, 16, (2, 8))
    include = torch.zeros(2, 8, dtype=torch.bool)
    include[0, :3] = True
    L, _, per_tok = masked_cross_entropy(logits, targets, include)
    assert float(L) == pytest.approx(float(per_tok[0, :3].mean()), rel=1e-5)


def test_empty_selection_does_not_divide_by_zero() -> None:
    from soma.openended.tokensel import masked_cross_entropy

    logits = torch.randn(2, 8, 16)
    targets = torch.randint(0, 16, (2, 8))
    L, Lz, _ = masked_cross_entropy(
        logits, targets, torch.zeros(2, 8, dtype=torch.bool)
    )
    assert float(L) == 0.0 and torch.isfinite(L)
