"""SPEC: 02 §2.8 -- the acceptance tests T1..T10 for the innate codes."""

from __future__ import annotations

import math

import pytest
import torch

from gnomon.config import GenomeShape, N_ROLES, PhenoConfig
from gnomon.genome import codes as C

SHAPE = GenomeShape()
WIDTHS = [16, 32, 64, 128, 192, 256, 512, 1024, 2048, 4096]


def test_T1_codomain_is_pm1():
    for d in WIDTHS:
        b = C.channel_codes(d, SHAPE)
        assert set(b.unique().tolist()) <= {-1.0, 1.0}
        assert b.shape == (d, SHAPE.n_c)
    for V in (256, 4096, 8192):
        assert set(C.token_codes(V, SHAPE).unique().tolist()) <= {-1.0, 1.0}
    assert set(C.layer_codes(12, SHAPE).unique().tolist()) <= {-1.0, 1.0}
    assert set(C.role_codes(SHAPE).unique().tolist()) <= {-1.0, 1.0}


def test_T2_injective():
    for d in WIDTHS:
        b = C.channel_codes(d, SHAPE)
        assert torch.unique(b, dim=0).shape[0] == d


def test_T3_nesting_channels_tokens_roles():
    """The property that makes a genome evolved at one width decode at another."""
    big = C.channel_codes(4096, SHAPE)
    for d in WIDTHS:
        torch.testing.assert_close(C.channel_codes(d, SHAPE), big[:d])
    bigV = C.token_codes(8192, SHAPE)
    for V in (256, 1024, 4096):
        torch.testing.assert_close(C.token_codes(V, SHAPE), bigV[:V])
    torch.testing.assert_close(C.role_codes(SHAPE), C.role_codes(SHAPE))


def test_T4_spread_bit_balance_and_hamming():
    for d in WIDTHS:
        b = C.channel_codes(d, SHAPE)
        bal = C.bit_balance(b)
        assert bal.abs().max() <= 3.0 / math.sqrt(d) + 1e-6, (
            f"d={d}: bit balance {bal.tolist()} -- the image is not a "
            f"well-spread sample of the Hamming cube"
        )
        assert C.min_hamming(b) >= 1


def test_T5_no_bit_position_is_constant():
    for d in WIDTHS:
        b = C.channel_codes(d, SHAPE)
        assert (b.std(dim=0) > 0).all(), f"d={d}: a bit position is constant"


def test_T5_positive_control_prefix_coding_fails():
    """SPEC: 02 §2.3. The absence check must be shown to fire.

    A prefix coding is exactly the naive construction the spec rejects. If this
    test ever passes, T5 is measuring nothing.
    """
    idx = torch.arange(128, dtype=torch.int64)
    planted = C.bits_to_pm1(idx, SHAPE.n_c)
    assert not (planted.std(dim=0) > 0).all(), "the positive control did not fire"
    assert C.bit_balance(planted).abs().max() == pytest.approx(1.0)


def test_T5_positive_control_stride_coding_fails():
    idx = torch.arange(128, dtype=torch.int64) * (4096 // 128)
    planted = C.bits_to_pm1(idx, SHAPE.n_c)
    assert not (planted.std(dim=0) > 0).all()


def test_T6_determinism_and_seed_sensitivity():
    a = C.channel_codes(256, SHAPE)
    b = C.channel_codes(256, GenomeShape(seed_codes=SHAPE.seed_codes))
    torch.testing.assert_close(a, b)
    c = C.channel_codes(256, GenomeShape(seed_codes=SHAPE.seed_codes + 1))
    assert not torch.equal(a, c)


def test_T7_gray_adjacency():
    """Adjacent depths differ in exactly one bit when the quantised depth steps by 1."""
    L = 1 << SHAPE.n_L                      # every depth level distinct, step exactly 1
    r = C.layer_codes(L, SHAPE)
    diffs = (r[1:] != r[:-1]).sum(-1)
    assert diffs.tolist() == [1] * (L - 1)


def test_T8_depth_is_relative_not_absolute():
    r4 = C.layer_codes(4, SHAPE)
    r12 = C.layer_codes(12, SHAPE)
    torch.testing.assert_close(r4[0], r12[0])
    torch.testing.assert_close(r4[-1], r12[-1])
    # And a middle layer is *not* shared -- relativity is not degeneracy.
    assert not torch.equal(r4[1], r12[1])


def test_T8_single_layer_is_defined():
    r1 = C.layer_codes(1, SHAPE)
    assert r1.shape == (1, SHAPE.n_L)
    torch.testing.assert_close(r1[0], C.layer_codes(4, SHAPE)[0])


def test_T9_overflow_raises_and_names_the_width():
    with pytest.raises(ValueError, match=r"n_bits must grow to 13"):
        C.channel_codes(4097, SHAPE)
    with pytest.raises(ValueError, match=r"n_bits must grow to 15"):
        C.token_codes(16385, SHAPE)
    with pytest.raises(ValueError, match=r"exceeds 2\^n_L"):
        C.layer_codes(33, SHAPE)


def test_T10_zero_cost():
    """No parameters, no state: the codes are a pure function."""
    import inspect

    from gnomon.genome import codes as mod

    for _, obj in inspect.getmembers(mod):
        assert not isinstance(obj, torch.nn.Parameter)
        assert not isinstance(obj, torch.nn.Module)


def test_vector_column_is_unambiguous():
    """SPEC: 02 §2.6. The flag bit separates 'no column' from every real one."""
    vc = C.vector_column(SHAPE)
    assert vc.shape == (1, SHAPE.n_c + 1)
    assert vc[0, -1] == -1.0
    assert (vc[0, :-1] == 0.0).all()
    real = C.column_codes(C.channel_codes(4096, SHAPE))
    assert (real[:, -1] == 1.0).all()
    assert not (real == vc).all(dim=-1).any()


def test_roles_fit_their_width():
    r = C.role_codes(SHAPE)
    assert r.shape == (N_ROLES, SHAPE.n_M)
    assert torch.unique(r, dim=0).shape[0] == N_ROLES


def test_genome_size_is_scale_free():
    """SPEC: 03 §3.5 (D8) -- the claim, asserted at the level of the shape."""
    from gnomon.config import M1, M_NANO, M_PICO, assert_scale_free

    assert_scale_free(SHAPE, [M_PICO, M_NANO, M1])
    assert SHAPE.n_params() == 4483, SHAPE.n_params()
    # H log N: one more channel bit costs 2*d_h and doubles the addressable width.
    wide = GenomeShape(n_c=SHAPE.n_c + 1)
    assert wide.n_params() - SHAPE.n_params() == 2 * SHAPE.d_h
    assert wide.d_max == 2 * SHAPE.d_max
    # One more vocabulary bit costs d_h.
    deep = GenomeShape(n_v=SHAPE.n_v + 1)
    assert deep.n_params() - SHAPE.n_params() == SHAPE.d_h


# ---------------------------------------------------------------- T11..T13
# SPEC: 02 §2.9 -- the frequency-ordered token code.

def _toy_counts(V=512, seed=0):
    """Zipf-ish counts, so `rank` is meaningful without touching the corpus."""
    g = torch.Generator().manual_seed(seed)
    z = 1.0 / (torch.arange(1, V + 1, dtype=torch.float64) ** 1.1)
    perm = torch.randperm(V, generator=g)      # ids are NOT in frequency order
    out = torch.empty(V, dtype=torch.float64)
    out[perm] = z * 1e6
    return out


def test_T11_the_frequency_code_carries_frequency():
    """SPEC: 02 §2.8 T11. And the bijection control, without which it is vacuous."""
    import numpy as np

    V = 512
    counts = _toy_counts(V)
    rank = C.frequency_rank(counts)
    y = np.log(counts.numpy())

    def r2(b):
        X = np.c_[b, np.ones(V)]
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 1 - ((y - X @ w) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    got = r2(C.token_codes(V, SHAPE, rank=rank).numpy())
    ctrl = r2(C.token_codes(V, SHAPE).numpy())
    assert got >= 0.30, f"the frequency code recovers only R^2 = {got:.3f}"
    assert ctrl <= 0.05, (
        f"the bijection control recovers R^2 = {ctrl:.3f}; T11 would pass for "
        f"any code and is then measuring nothing"
    )


def test_T12_the_frequency_code_is_still_a_code():
    V = 512
    rank = C.frequency_rank(_toy_counts(V))
    b = C.token_codes(V, SHAPE, rank=rank)
    assert set(b.unique().tolist()) <= {-1.0, 1.0}
    assert torch.unique(b, dim=0).shape[0] == V, "codes collide"
    assert (b.std(dim=0) > 0).all(), "a bit position is constant"
    assert C.bit_balance(b).abs().max() <= C.STRUCTURED_BALANCE_MAX
    # Adjacent *ranks* have nearby codes: the property a decoder needs, on the
    # axis that turned out to have the topology.
    #
    # Not exactly one bit. Exact 1-bit adjacency holds only when `V = 2^n_v`, so
    # that consecutive ranks map to consecutive `q`; at `V = 512` with 14 bits
    # the step is 32 and adjacent ranks differ in 2-3 bits. The claim is
    # smoothness, and the random control is what makes it a claim at all.
    order = torch.argsort(rank)
    diffs = (b[order][1:] != b[order][:-1]).sum(-1).float()
    rnd = C.token_codes(V, SHAPE)
    rnd_diffs = (rnd[1:] != rnd[:-1]).sum(-1).float()
    assert float(diffs.mean()) < 0.5 * float(rnd_diffs.mean()), (
        f"adjacent ranks differ in {float(diffs.mean()):.2f} bits against "
        f"{float(rnd_diffs.mean()):.2f} for a random code; the code is not "
        f"smooth in rank"
    )
    assert int(diffs.max()) <= 5, int(diffs.max())


def test_T13_the_frequency_code_is_rank_relative():
    """The most and least frequent type get the same code at any `V`."""
    small, big = 256, 1024
    rs = C.frequency_rank(_toy_counts(small, seed=1))
    rb = C.frequency_rank(_toy_counts(big, seed=2))
    bs = C.token_codes(small, SHAPE, rank=rs)
    bb = C.token_codes(big, SHAPE, rank=rb)
    torch.testing.assert_close(bs[torch.argmin(rs)], bb[torch.argmin(rb)])
    torch.testing.assert_close(bs[torch.argmax(rs)], bb[torch.argmax(rb)])


def test_T13_frequency_rank_is_a_permutation():
    counts = _toy_counts(64)
    rank = C.frequency_rank(counts)
    assert sorted(rank.tolist()) == list(range(64))
    assert int(rank[int(torch.argmax(counts))]) == 0
    with pytest.raises(ValueError, match="rank has"):
        C.token_codes(64, SHAPE, rank=torch.arange(63))


# --------------------------------------------------------------- §2.9 wiring
#
# The construction was implemented and tested from the day §2.9 was written, and
# for the whole of v1.0 nothing but `scripts/gnomon_tokencode.py` ever installed
# it: every evolutionary run, every baseline and every transfer silently used
# §2.4's bijection, which §2.9 measures at +0.0999 nats worse. These tests bind
# the *default*, not merely the capability.

def test_T14_the_frequency_rank_is_the_default_token_code():
    """SPEC: 02 §2.9. A runner uses the specified code unless told otherwise."""
    from gnomon.data.stream import Shard
    from gnomon.lifetime.runner import LifetimeRunner
    import numpy as np

    cfg = PhenoConfig(name="tiny", d_model=32, L=2, n_h=2, n_kv=1, d_head=16,
                      d_ff=64, vocab=64, T=64, N_life=2, chunk_size=32,
                      window=32, batch=2, lr=3e-3)
    rng = np.random.default_rng(0)
    # A Zipf-like draw, so the ranks are not all ties.
    p = 1.0 / np.arange(1, 65)
    toks = rng.choice(64, size=20000, p=p / p.sum()).astype(np.uint16)
    sh = Shard(toks, np.arange(0, 20000, 100, dtype=np.int64),
               np.full(200, 100, dtype=np.int64), {"n_tokens": 20000})

    r = LifetimeRunner(cfg, SHAPE, train_shard=sh, eval_shard=sh, device="cpu",
                       compile_mode=None, n_eval_batches=1)
    assert r.token_code == "freq"
    assert r.model.decoder._tok_rank is not None
    assert r.token_provenance["n_counted"] == 20000
    assert "counts_digest" in r.token_provenance

    # ...and the §2.4 control is still reachable, because §2.9's claim is a
    # measured comparison and a comparison needs both arms.
    r.set_token_code(None)
    assert r.token_code == "bijection"
    assert r.model.decoder._tok_rank is None


def test_T14_switching_the_token_code_invalidates_the_compiled_graph():
    """A graph captured against the old code table would decode the old rows.

    Exactly the staleness §05.1's line 5 forbids, one level up: the embedding is
    decoded from the token code, so a cached graph outlives the code change
    unless the runner discards it.
    """
    from gnomon.data.stream import Shard
    from gnomon.lifetime.runner import LifetimeRunner
    import numpy as np

    cfg = PhenoConfig(name="tiny", d_model=32, L=2, n_h=2, n_kv=1, d_head=16,
                      d_ff=64, vocab=64, T=64, N_life=2, chunk_size=32,
                      window=32, batch=2, lr=3e-3)
    toks = np.random.default_rng(1).integers(0, 64, 20000).astype(np.uint16)
    sh = Shard(toks, np.arange(0, 20000, 100, dtype=np.int64),
               np.full(200, 100, dtype=np.int64), {"n_tokens": 20000})
    r = LifetimeRunner(cfg, SHAPE, train_shard=sh, eval_shard=sh, device="cpu",
                       compile_mode=None, n_eval_batches=1)
    a = r.evaluate()
    r.set_token_code(None)
    assert not r._graphs or list(r._graphs) == [(r.model.arch, cfg.batch, cfg.T)]
    b = r.evaluate()
    assert a != b, "the two codes decoded identical embeddings"


def test_T14_token_counts_covers_the_whole_shard():
    """SPEC: 02 §2.9. Counted over all of it, not a window."""
    from gnomon.data.stream import Shard, token_counts
    import numpy as np

    toks = np.arange(4096, dtype=np.uint16).repeat(3)
    sh = Shard(toks, np.array([0], dtype=np.int64),
               np.array([len(toks)], dtype=np.int64), {"n_tokens": len(toks)})
    c = token_counts(sh, 4096, chunk=97)          # a chunk that does not divide
    assert c.shape == (4096,)
    assert np.allclose(c, 4.0)                    # 3 occurrences + Laplace 1


def test_T14_frequency_rank_breaks_ties_by_ascending_token_id():
    """SPEC: 02 §2.9. 508 of `ts_4096`'s 4,096 types are tied with another.

    `torch.argsort` without `stable=True` leaves a tie's order unspecified, and
    on the real corpus its output differs from the stable one -- so the code was
    a function of sort internals rather than of the corpus, which §2.9's
    reproducibility requirement forbids. The rule is fixed here.
    """
    counts = torch.tensor([5.0, 3.0, 3.0, 3.0, 9.0, 1.0, 1.0, 5.0])
    assert C.frequency_rank(counts).tolist() == [1, 3, 4, 5, 0, 6, 7, 2]

    # And a heavy-tie input is still a permutation, repeatably.
    flat = torch.ones(512)
    r = C.frequency_rank(flat)
    assert sorted(r.tolist()) == list(range(512))
    assert torch.equal(r, C.frequency_rank(flat))
    assert torch.equal(r, torch.arange(512))     # all tied -> id order exactly
