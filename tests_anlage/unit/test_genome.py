"""Acceptance tests D1-D14 of docs/anlage/03. SPEC: 03 §3.9"""

from __future__ import annotations

import pytest
import torch

from anlage.config import (
    MATRIX_ROLES, ROLE_CONV_Q, ROLE_EMB, ROLE_LOGIT_SCALE, ROLE_NORM_MIX, ROLE_Q,
    ROLE_UP, VECTOR_ROLES, GenomeShape, PILE, StageConfig, TINYSTORIES,
    assert_scale_free,
)
from anlage.genome import DecodeContext, Genome, offset, s_target
from anlage.latents import LatentBank
from gnomon.utils.stats import effective_rank
from tests_anlage.conftest import make_tables


def _ctx64(shape, cfg, vocab):
    tables = make_tables(shape, cfg, vocab)
    bank = LatentBank(shape, cfg, vocab, seed=3).double()
    ctx = DecodeContext(cfg, tables, bank, vocab)
    return ctx


def test_D1_reference_agreement(small_shape, tiny_stage, vocab):
    ctx = _ctx64(small_shape, tiny_stage, vocab)
    G = Genome(small_shape, seed=1).double()
    for role, l in ((ROLE_Q, 0), (ROLE_NORM_MIX, 1), (ROLE_EMB, -1), (ROLE_CONV_Q, 2),
                    (ROLE_LOGIT_SCALE, -1)):
        fast = G.decode(ctx, role, l)
        ref = G.decode_reference(ctx, role, l)
        assert fast.shape == ref.shape
        assert torch.allclose(fast, ref, atol=1e-10, rtol=1e-10), (role, l)


def test_D2_every_genome_tensor_receives_gradient(small_shape, tiny_stage, vocab):
    ctx = _ctx64(small_shape, tiny_stage, vocab)
    G = Genome(small_shape, seed=1).double()
    W = G.decode_all(ctx)
    loss = sum((w ** 2).sum() for w in W.values())
    loss.backward()
    for name, p in G.named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, name


def test_D3_finite_differences(small_shape, tiny_stage, vocab):
    ctx = _ctx64(small_shape, tiny_stage, vocab)
    G = Genome(small_shape, seed=2).double()
    ctx.bank.z_res.requires_grad_(True)
    target = torch.randn(*ctx.shape(ROLE_Q, 0), dtype=torch.float64,
                         generator=torch.Generator().manual_seed(9))

    def J():
        return ((G.decode(ctx, ROLE_Q, 0) - target) ** 2).sum()

    params = list(G.named_parameters()) + [("z_res", ctx.bank.z_res)]
    loss = J()
    grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    g = torch.Generator().manual_seed(4)
    eps = 1e-5
    for (name, p), gr in zip(params, grads):
        if gr is None:
            continue
        flat = p.detach().view(-1)
        for _ in range(3):
            i = int(torch.randint(flat.numel(), (1,), generator=g))
            old = float(flat[i])
            with torch.no_grad():
                flat[i] = old + eps
            up = float(J())
            with torch.no_grad():
                flat[i] = old - eps
            dn = float(J())
            with torch.no_grad():
                flat[i] = old
            fd = (up - dn) / (2 * eps)
            an = float(gr.view(-1)[i])
            assert abs(fd - an) <= 1e-5 * max(1.0, abs(an)), (name, fd, an)


def test_D4_scale_free_and_the_configured_genome():
    shape = GenomeShape()
    assert_scale_free(shape, [TINYSTORIES, PILE])
    n = Genome(shape).n_params()
    assert n == shape.n_params()
    # the same genome object serves every stage of both lineages
    for lin in (TINYSTORIES, PILE):
        for st in lin.stages:
            assert st.widest_axis(lin.vocab) <= shape.d_max


def test_D5_rank_floor_at_initialisation():
    shape = GenomeShape(d_h=64, H=4, d_g=32, d_z=8, d_y=4, d_sigma=8)
    cfg = StageConfig(0, d_model=128, L=4, n_h=2, n_kv_attn=1, d_ff=256, d_head=64)
    vocab = 512
    ctx = _ctx64(shape, cfg, vocab)
    G = Genome(shape, seed=3).double()
    for role, l in ((ROLE_Q, 0), (ROLE_UP, 1), (ROLE_EMB, -1)):
        W = G.decode(ctx, role, l)
        er = effective_rank(torch.linalg.svdvals(W).tolist())
        assert er >= 0.20 * min(W.shape), (role, er, W.shape)


def test_D6_scale(small_shape, tiny_stage, vocab):
    ctx = _ctx64(small_shape, tiny_stage, vocab)
    G = Genome(small_shape, seed=1).double()
    for role in MATRIX_ROLES:
        l = -1 if role == ROLE_EMB else 0
        if role in (15,):            # HEAD_OUT absent while tied
            continue
        if l != -1 and role not in tiny_stage.roles(l):
            continue
        W = G.decode(ctx, role, l)
        shp = ctx.shape(role, l)
        want = s_target(role, shp, tiny_stage) * float(torch.exp(G.log_sigma(ctx, l, role)))
        assert abs(float(W.pow(2).mean().sqrt()) - want) <= 1e-6 * want
    for role in VECTOR_ROLES:
        l = -1 if role in (18, 19) else 0
        v = G.decode(ctx, role, l)
        assert torch.allclose(v, torch.full_like(v, offset(role)))      # o = 0 at init


def test_D7_nesting(small_shape, tiny_stage, tiny_stage_next, vocab):
    G = Genome(small_shape, seed=1).double()
    ctx_a = _ctx64(small_shape, tiny_stage, vocab)
    # the wider stage keeps the narrow stage's latents on RES and layer 0's units
    bank_b = LatentBank(small_shape, tiny_stage_next, vocab, seed=11).double()
    with torch.no_grad():
        bank_b.z_res[: tiny_stage.d_model] = ctx_a.bank.z_res
        bank_b.z_inner[0, : tiny_stage.d_ff] = ctx_a.bank.z_inner[0]
        bank_b.z_head[0, : tiny_stage.n_h * 8] = ctx_a.bank.z_head[0]
        bank_b.y[0] = ctx_a.bank.y[0]
    # same depth code needs same L: compare on the global roles and on a
    # same-L wider stage instead.
    wide_same_L = StageConfig(1, d_model=24, L=4, n_h=3, n_kv_attn=1, d_ff=32, d_head=8)
    bank_c = LatentBank(small_shape, wide_same_L, vocab, seed=11).double()
    with torch.no_grad():
        bank_c.z_res[: tiny_stage.d_model] = ctx_a.bank.z_res
        bank_c.z_inner[:, : tiny_stage.d_ff] = ctx_a.bank.z_inner
        bank_c.z_head[:, : tiny_stage.n_h * 8] = ctx_a.bank.z_head
        bank_c.z_gate[:, : 2 * tiny_stage.n_h] = ctx_a.bank.z_gate
        bank_c.y.copy_(ctx_a.bank.y)
        bank_c.z_tok.copy_(ctx_a.bank.z_tok)
    ctx_c = DecodeContext(wide_same_L, make_tables(small_shape, wide_same_L, vocab), bank_c, vocab)
    for role, l in ((ROLE_Q, 0), (ROLE_UP, 1), (ROLE_EMB, -1)):
        Wa = G.decode(ctx_a, role, l)
        Wc = G.decode(ctx_c, role, l)
        ra, rc = Wa.shape[0], Wa.shape[1]
        ratio = Wc[:ra, :rc] / Wa
        assert float((ratio - ratio.mean()).abs().max()) <= 1e-6 * float(ratio.abs().mean())


def test_D8_determinism(small_shape, tiny_stage, vocab):
    ctx = _ctx64(small_shape, tiny_stage, vocab)
    G = Genome(small_shape, seed=1).double()
    a, b = G.decode_all(ctx), G.decode_all(ctx)
    for k in a:
        assert torch.equal(a[k], b[k])


def test_D9_chunking(small_shape, tiny_stage, vocab):
    ctx = _ctx64(small_shape, tiny_stage, vocab)
    G = Genome(small_shape, seed=1).double()
    full = G.decode_matrix(ctx, ROLE_UP, 0)
    chunked = G.decode_matrix(ctx, ROLE_UP, 0, chunk_elems=7 * ctx.shape(ROLE_UP, 0)[0])
    assert torch.allclose(full, chunked, atol=1e-6)


def test_D11_totality(small_shape, tiny_stage, vocab):
    ctx = _ctx64(small_shape, tiny_stage, vocab)
    for seed in range(20):
        G = Genome(small_shape, seed=seed).double()
        W = G.decode_all(ctx)
        assert all(torch.isfinite(w).all() for w in W.values())


def test_D13_counts_of_section_3_6():
    shape = GenomeShape()
    G = Genome(shape)
    t = G.count_table()
    assert t["W_1"] == 256 * 45 and t["W_1_tok"] == 256 * 47
    assert t["W_1b"] == 256 * 256 and t["W_2"] == 256 * 256
    assert t["O"] == 16 * 4 * 64 * 64 and t["kappa_hat"] == 16 and t["o"] == 4 * 64
    assert t["A_sigma"] == 16 * 12 and t["b_sigma"] == 16 and t["w_sigma"] == 16
    assert G.n_params() == 418_288 == shape.n_params()


def test_D14_uninformative_genome(small_shape, tiny_stage, vocab):
    ctx = _ctx64(small_shape, tiny_stage, vocab)
    G = Genome(small_shape, seed=1).double()
    with torch.no_grad():
        G.O.zero_()
        G.o.zero_()
    assert torch.all(G.decode(ctx, ROLE_Q, 0) == 0.0)
    assert torch.allclose(G.decode(ctx, ROLE_NORM_MIX, 0), torch.ones(tiny_stage.d_model,
                                                                        dtype=torch.float64))
