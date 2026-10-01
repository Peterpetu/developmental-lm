"""Acceptance tests P1-P13 of docs/anlage/04. SPEC: 04 §4.10"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from anlage import ops
from anlage.config import (
    ROLE_AB, ROLE_AGATE, ROLE_CONV_Q, ROLE_EMB, ROLE_GATE, ROLE_K, ROLE_LOGIT_SCALE,
    ROLE_NORM_FINAL, ROLE_NORM_MIX, ROLE_O, ROLE_Q, ROLE_V, PILE, StageConfig,
    TINYSTORIES,
)
from anlage.model import Phenotype, free_init, key_name
from gnomon.pheno import ops as gops

CFG = StageConfig(0, d_model=32, L=4, n_h=4, n_kv_attn=2, d_ff=48, d_head=8, chunk_size=8)
VOCAB = 64


def _model(seed=0, cfg=CFG, dtype=torch.float32):
    return Phenotype(cfg, VOCAB, seed=seed).to(dtype)


def test_P1_shapes_at_every_stage():
    for lin in (TINYSTORIES, PILE):
        for st in lin.stages:
            for role, l in st.keys(lin.vocab):
                shp = st.role_shape(role, l, lin.vocab)
                assert all(s >= 1 for s in shp), (lin.name, st.summary(), role, l)
    m = _model()
    for role, l in CFG.keys(VOCAB):
        assert tuple(m.get(role, l).shape) == CFG.role_shape(role, l, VOCAB)


def test_P2_attention_oracle_without_gate():
    torch.manual_seed(0)
    B, T, d, n_h, n_kv, dk = 2, 16, 32, 4, 2, 8
    h = torch.randn(B, T, d, dtype=torch.float64)
    Wq, Wk, Wv = (torch.randn(n_h * dk, d, dtype=torch.float64) * 0.2,
                  torch.randn(n_kv * dk, d, dtype=torch.float64) * 0.2,
                  torch.randn(n_kv * dk, d, dtype=torch.float64) * 0.2)
    Wo = torch.randn(d, n_h * dk, dtype=torch.float64) * 0.2
    Wg = torch.zeros(n_h * dk, d, dtype=torch.float64)
    # gate fixed at 1: offset -> +inf; no RoPE, to match the NoPE oracle.
    ours = ops.attention_gated(h, Wq, Wk, Wv, Wo, Wg, n_h=n_h, n_kv=n_kv, d_head=dk,
                               gate_offset=60.0, use_rope=False)
    # the oracle uses repeat_interleave; make the two groupings coincide by
    # comparing at n_kv = n_h too, and at n_kv < n_h by re-ordering KV heads.
    sink = torch.full((n_h,), -1e4, dtype=torch.float64)
    # oracle grouping: head g reads kv g // rep; ours: g mod n_kv. Build an
    # oracle K/V whose interleaved order reproduces ours.
    rep = n_h // n_kv
    idx = torch.tensor([g % n_kv for g in range(n_h)])
    # oracle assigns kv j to heads [j*rep, (j+1)*rep); permute Q rows instead.
    perm = torch.argsort(idx, stable=True)                  # heads sorted by kv id
    Wq_p = Wq.view(n_h, dk, d)[perm].reshape(n_h * dk, d)
    Wo_p = Wo.view(d, n_h, dk)[:, perm].reshape(d, n_h * dk)
    ref = gops.attention(h, Wq_p, Wk, Wv, Wo_p, sink, n_h=n_h, n_kv=n_kv, d_head=dk)
    assert torch.allclose(ours, ref, atol=1e-10)


def test_P3_delta_oracle():
    torch.manual_seed(1)
    m = _model(dtype=torch.float64)
    toks = torch.randint(0, VOCAB, (2, 33))
    m.use_reference = False
    a = m(toks)
    m.use_reference = True
    b = m(toks)
    assert torch.allclose(a, b, atol=1e-9)


def test_P4_causality_and_the_conv_control():
    torch.manual_seed(2)
    m = _model()
    toks = torch.randint(0, VOCAB, (1, 20))
    alt = toks.clone()
    alt[0, 12:] = torch.randint(0, VOCAB, (8,))
    with torch.no_grad():
        la, lb = m(toks), m(alt)
    assert torch.allclose(la[0, :12], lb[0, :12], atol=1e-5)
    assert not torch.allclose(la[0, 12:], lb[0, 12:])
    # positive control: a conv that pads on the right leaks the future
    x = torch.randn(1, 20, 3)
    W = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * 3)
    good = ops.causal_conv4(x, W)
    xp = F.pad(x.transpose(1, 2), (0, 3))                                  # broken pad
    bad = F.conv1d(xp, W.flip(-1).unsqueeze(1), groups=3).transpose(1, 2)
    assert torch.allclose(good[:, 1:], x[:, :-1])
    assert not torch.allclose(bad[:, :-1], good[:, :-1])


def test_P5_kv_grouping_is_g_mod_n_kv():
    torch.manual_seed(3)
    B, T, d, n_h, n_kv, dk = 1, 8, 32, 4, 2, 8
    h = torch.randn(B, T, d)
    Wq = torch.randn(n_h * dk, d) * 0.2
    Wk = torch.randn(n_kv * dk, d) * 0.2
    Wv = torch.randn(n_kv * dk, d) * 0.2
    Wo = torch.eye(n_h * dk)[: d] if d <= n_h * dk else None
    Wo = torch.randn(d, n_h * dk) * 0.2
    Wg = torch.zeros(n_h * dk, d)
    out = ops.attention_gated(h, Wq, Wk, Wv, Wo, Wg, n_h=n_h, n_kv=n_kv, d_head=dk,
                              gate_offset=60.0, use_rope=False)
    # swap the two KV heads: heads 0,2 and 1,3 swap roles under `g mod n_kv`;
    # under repeat_interleave heads 0,1 and 2,3 would.
    Wk2 = Wk.view(n_kv, dk, d)[[1, 0]].reshape(-1, d)
    Wv2 = Wv.view(n_kv, dk, d)[[1, 0]].reshape(-1, d)
    Wq2 = Wq.view(n_h, dk, d)[[1, 0, 3, 2]].reshape(-1, d)
    Wo2 = Wo.view(d, n_h, dk)[:, [1, 0, 3, 2]].reshape(d, -1)
    out2 = ops.attention_gated(h, Wq2, Wk2, Wv2, Wo2, Wg, n_h=n_h, n_kv=n_kv, d_head=dk,
                               gate_offset=60.0, use_rope=False)
    assert torch.allclose(out, out2, atol=1e-5)
    Wq3 = Wq.view(n_h, dk, d)[[2, 3, 0, 1]].reshape(-1, d)      # the interleave mutant
    Wo3 = Wo.view(d, n_h, dk)[:, [2, 3, 0, 1]].reshape(d, -1)
    out3 = ops.attention_gated(h, Wq3, Wk2, Wv2, Wo3, Wg, n_h=n_h, n_kv=n_kv, d_head=dk,
                               gate_offset=60.0, use_rope=False)
    assert not torch.allclose(out, out3, atol=1e-5)


def test_P6_initialisation_scales():
    cfg = StageConfig(0, d_model=256, L=8, n_h=4, n_kv_attn=2, d_ff=1024, d_head=64)
    m = Phenotype(cfg, 8192, seed=0)
    for role, l in cfg.keys(8192):
        W = m.get(role, l)
        shp = tuple(W.shape)
        if role in (ROLE_Q, ROLE_K, ROLE_V, ROLE_O, ROLE_AB, ROLE_GATE):
            want = shp[1] ** -0.5
            assert abs(float(W.pow(2).mean().sqrt()) - want) <= 0.1 * want, key_name(role, l)
        if role == ROLE_NORM_MIX:
            assert torch.allclose(W, torch.full_like(W, (2 * cfg.L) ** -0.5))
        if role == ROLE_CONV_Q:
            assert torch.all(W[:, 0] == 1) and torch.all(W[:, 1:] == 0)
    toks = torch.randint(0, 8192, (2, 64))
    x = F.embedding(toks, m.get(ROLE_EMB, -1)) * cfg.d_model ** 0.5
    rms = float(x.pow(2).mean().sqrt())
    assert 0.9 <= rms <= 1.1


def test_P7_softcap_and_zloss():
    m = _model()
    toks = torch.randint(0, VOCAB, (2, 17))
    logits = m(toks)
    assert float(logits.abs().max()) < ops.SOFTCAP
    x = torch.randn(5, 32)
    E = torch.randn(VOCAB, 32) * 0.2
    ls = torch.zeros(1)
    ce, zl = ops.chunked_cross_entropy(x, E, ls, torch.arange(5), chunk=2)
    lg = ops.softcap_logits(x @ E.T, ls)
    lse = torch.logsumexp(lg, -1)
    assert torch.allclose(zl, ops.Z_LOSS * (lse ** 2).mean(), atol=1e-6)
    assert torch.allclose(ce, F.cross_entropy(lg, torch.arange(5)), atol=1e-6)


def test_P8_optimizer_groups_by_name():
    m = _model()
    muon, adam = m.optimizer_groups()
    ids = {id(p) for p in muon} | {id(p) for p in adam}
    assert len(ids) == len(list(m.parameters()))
    assert not ({id(p) for p in muon} & {id(p) for p in adam})
    assert id(m.get(ROLE_AB, 0)) in {id(p) for p in muon}
    assert id(m.get(ROLE_CONV_Q, 0)) in {id(p) for p in adam}
    assert id(m.get(ROLE_EMB, -1)) in {id(p) for p in adam}


def test_P10_chunked_ce_matches_unchunked():
    torch.manual_seed(4)
    x = torch.randn(11, 32)
    E = torch.randn(VOCAB, 32) * 0.3
    ls = torch.tensor([0.3])
    t = torch.randint(0, VOCAB, (11,))
    a, _ = ops.chunked_cross_entropy(x, E, ls, t, chunk=4)
    b, _ = ops.chunked_cross_entropy(x, E, ls, t, chunk=11)
    assert torch.allclose(a, b, atol=1e-6)


def test_P11_determinism():
    m = _model()
    toks = torch.randint(0, VOCAB, (2, 16))
    with torch.no_grad():
        a, b = m(toks), m(toks)
    assert torch.equal(a, b)
    m2 = _model()
    assert all(torch.equal(p, q) for p, q in zip(m.parameters(), m2.parameters()))


def test_P12_counts_from_named_parameters():
    for lin in (TINYSTORIES, PILE):
        st = lin.stages[0]
        m = Phenotype(st, lin.vocab, seed=0)
        assert m.n_params() == st.n_params(lin.vocab)
        assert m.n_params_nonemb() == st.n_params_nonemb(lin.vocab)


def test_P13_no_free_tensor_outside_the_roles():
    m = _model()
    names = {n for n, _ in m.named_parameters()}
    want = {"W." + key_name(r, l) for r, l in CFG.keys(VOCAB)}
    assert names == want
    m.untie()
    assert "W." + key_name(15, -1) in {n for n, _ in m.named_parameters()}


def test_backward_runs_and_loss_is_finite():
    m = _model()
    w = torch.randint(0, VOCAB, (2, 25))
    ce, zl = m.loss(w, chunk=7)
    (ce + zl).backward()
    assert math.isfinite(float(ce)) and float(ce) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_P9_schedule_boundaries_and_decay_to_exactly_zero():
    from anlage.recipe import TrainConfig, batch_at, lr_at
    tc = TrainConfig(tokens=1_000_000, batch_final=32, T=512, lr_provenance="test")
    lr = 0.02
    assert lr_at(0.0, lr, tc) == 0.0
    assert abs(lr_at(tc.p_warm, lr, tc) - lr) < 1e-12
    assert abs(lr_at(0.5 * tc.p_warm, lr, tc) - 0.5 * lr) < 1e-12
    assert lr_at(0.1, lr, tc) == lr
    assert lr_at(tc.p_dec, lr, tc) == lr
    assert abs(lr_at(0.5 * (tc.p_dec + 1.0), lr, tc) - 0.5 * lr) < 1e-12
    assert lr_at(1.0, lr, tc) == 0.0
    assert batch_at(0.0, tc) == int(round(tc.batch_final * tc.batch_min_frac / 8)) * 8
    assert batch_at(tc.p_batch_warm, tc) == tc.batch_final
    assert batch_at(1.0, tc) == tc.batch_final
    prev = 0
    for i in range(101):
        b = batch_at(i / 100, tc)
        assert b % 8 == 0 and b >= prev
        prev = b
