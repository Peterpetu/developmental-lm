"""Acceptance tests C1-C10 of docs/anlage/05. SPEC: 05 §5.9"""

from __future__ import annotations

import torch

from anlage.config import ROLE_EMB, ROLE_Q, ROLE_UP
from anlage.genome import DecodeContext, Genome
from anlage.latents import LatentBank
from anlage.regress import RegressConfig, expression, objective, regress, shuffled
from tests_anlage.conftest import make_tables


def _ctx(shape, cfg, vocab, seed=3):
    bank = LatentBank(shape, cfg, vocab, seed=seed)
    return DecodeContext(cfg, make_tables(shape, cfg, vocab), bank, vocab)


def test_C1_planted_recovery(small_shape, tiny_stage, vocab):
    ctx_star = _ctx(small_shape, tiny_stage, vocab, seed=3)
    G_star = Genome(small_shape, seed=1)
    with torch.no_grad():
        G_star.o.normal_(0, 0.5)                 # make the vector roles informative too
    W = {k: v.detach() for k, v in G_star.decode_all(ctx_star).items()}
    ctx = _ctx(small_shape, tiny_stage, vocab, seed=3)       # same latents as the planted
    G = Genome(small_shape, seed=1)
    with torch.no_grad():
        for p, q in zip(G.parameters(), G_star.parameters()):
            p.copy_(q + 0.02 * torch.randn_like(q))          # a start near the planted genome
    rep = regress(G, ctx, W, RegressConfig(n_fit=600, lr=2e-3, lambda_z=0.0, lambda_g=0.0,
                                           report_every=50, plateau_window=1000))
    assert rep.residual_mean < 1e-3, rep.residual


def test_C2_objective_closed_form(small_shape, tiny_stage, vocab):
    ctx = _ctx(small_shape, tiny_stage, vocab)
    G = Genome(small_shape, seed=2)
    W = {k: v.detach() + 0.1 for k, v in G.decode_all(ctx, keys=[(ROLE_Q, 0), (ROLE_UP, 1)]).items()}
    rc = RegressConfig(lambda_z=0.0, lambda_y=0.0, lambda_g=0.0)
    J, res = objective(G, ctx, W, rc)
    D = G.decode_all(ctx, keys=list(W))
    want = sum(((D[k] - W[k]) ** 2).sum() / (W[k] ** 2).sum() for k in W)
    assert torch.allclose(J, want, atol=1e-8)


def test_C3_the_prior_prunes_the_latents(small_shape, tiny_stage, vocab):
    G0 = Genome(small_shape, seed=5)
    W = {k: v.detach() + 0.3 * torch.randn_like(v) for k, v in
         G0.decode_all(_ctx(small_shape, tiny_stage, vocab, seed=9)).items()}
    out = {}
    for lam in (1e-3, 10.0):
        ctx = _ctx(small_shape, tiny_stage, vocab, seed=3)
        G = Genome(small_shape, seed=5)
        # Adam moves each coordinate by ~lr per step whatever the gradient, so
        # the shrinkage needs lr x steps >> |z| ~ 1: 0.02 x 400 = 8.
        rep = regress(G, ctx, W, RegressConfig(n_fit=400, lr=0.02, lambda_z=lam,
                                               report_every=50, plateau_window=1000))
        out[lam] = (float(sum(p.norm() ** 2 for n, p in ctx.bank.named_parameters()
                              if n != "y").sqrt()), rep.residual_mean)
    assert out[10.0][0] < out[1e-3][0] / 10
    assert out[10.0][1] > out[1e-3][1]


def test_C4_continuity_starts_at_the_previous_residual(small_shape, tiny_stage, vocab):
    ctx = _ctx(small_shape, tiny_stage, vocab, seed=3)
    G = Genome(small_shape, seed=1)
    W = {k: v.detach() + 0.05 * torch.randn_like(v) for k, v in G.decode_all(ctx).items()}
    rc = RegressConfig(lambda_z=0.0, lambda_y=0.0, lambda_g=0.0)
    J0, _ = objective(G, ctx, W, rc)
    rep = regress(G, ctx, W, RegressConfig(n_fit=100, report_every=100, plateau_window=1000))
    J1, _ = objective(G, ctx, W, rc)
    assert float(J1) < float(J0)
    # a second regression from the fitted state starts where the first ended
    J2, _ = objective(G, ctx, W, rc)
    assert abs(float(J2) - float(J1)) < 1e-6


def test_C6_the_dial(small_shape, tiny_stage, vocab):
    ctx = _ctx(small_shape, tiny_stage, vocab)
    G = Genome(small_shape, seed=1)
    D = {k: v.detach() for k, v in G.decode_all(ctx).items()}
    W_prev = {k: torch.randn_like(v) for k, v in D.items()}
    assert all(torch.equal(expression(W_prev, D, 1.0)[k], W_prev[k]) for k in D)
    assert all(torch.equal(expression(W_prev, D, 0.0)[k], D[k]) for k in D)
    half = expression(W_prev, D, 0.5)
    assert all(torch.allclose(half[k], 0.5 * W_prev[k] + 0.5 * D[k]) for k in D)
    # a unit absent from W_prev takes the decode
    small = {k: v[:2] if v.dim() == 2 else v for k, v in W_prev.items()}
    e = expression(small, D, 0.5)
    k = (ROLE_UP, 0)
    assert torch.allclose(e[k][2:], D[k][2:]) and torch.allclose(e[k][:2], 0.5 * small[k] + 0.5 * D[k][:2])


def test_C7_shuffle_permutes_rows(small_shape, tiny_stage, vocab):
    ctx = _ctx(small_shape, tiny_stage, vocab)
    G = Genome(small_shape, seed=1)
    D = {k: v.detach() for k, v in G.decode_all(ctx).items()}
    S = shuffled(D, torch.Generator().manual_seed(0))
    k = (ROLE_EMB, -1)
    assert not torch.equal(S[k], D[k])
    assert torch.allclose(S[k].sort(0).values, D[k].sort(0).values)


def test_C9_determinism(small_shape, tiny_stage, vocab):
    W = {k: v.detach() for k, v in Genome(small_shape, seed=7).decode_all(
        _ctx(small_shape, tiny_stage, vocab, seed=9)).items()}
    outs = []
    for _ in range(2):
        ctx = _ctx(small_shape, tiny_stage, vocab, seed=3)
        G = Genome(small_shape, seed=1)
        regress(G, ctx, W, RegressConfig(n_fit=50, report_every=50, plateau_window=1000))
        outs.append(torch.cat([p.detach().flatten() for p in G.parameters()]))
    assert torch.equal(outs[0], outs[1])


def test_C8_proposition_2_raw_entries_nest_and_the_ratio_is_one_scalar(small_shape, tiny_stage, vocab):
    """05 §5.8: same layer set and latents, wider channel axes -> `raw` agrees on `A_s`."""
    import dataclasses
    from anlage.genome import MAT_INDEX, DecodeContext, Genome, s_target
    from anlage.latents import LatentBank
    from tests_anlage.conftest import make_tables
    cfg_s = tiny_stage
    cfg_w = dataclasses.replace(cfg_s, d_model=24, n_h=3, d_ff=32)
    G = Genome(small_shape, seed=3)
    bank_s = LatentBank(small_shape, cfg_s, vocab, seed=5)
    bank_w = bank_s.grown(cfg_w, list(range(cfg_s.L)), torch.Generator().manual_seed(0))
    ctx_s = DecodeContext(cfg_s, make_tables(small_shape, cfg_s, vocab), bank_s, vocab)
    ctx_w = DecodeContext(cfg_w, make_tables(small_shape, cfg_w, vocab), bank_w, vocab)
    n_checked = 0
    for role, l in cfg_s.keys(vocab, tied=True):
        if role not in MAT_INDEX:
            continue
        raw_s, rms_s = G.raw_matrix(ctx_s, role, l)
        raw_w, rms_w = G.raw_matrix(ctx_w, role, l)
        r, c = raw_s.shape
        assert raw_w.shape[0] >= r and raw_w.shape[1] >= c
        assert torch.allclose(raw_w[:r, :c], raw_s, atol=1e-6, rtol=1e-5), (role, l)
        D_s = G.decode_matrix(ctx_s, role, l)
        D_w = G.decode_matrix(ctx_w, role, l)
        # the per-matrix ratio is one scalar: rms(raw_s)/rms(raw_w) times the
        # change of the innate s_target with fan-in/fan-out (also one scalar)
        expected = (rms_s / rms_w) * s_target(role, tuple(D_w.shape), cfg_w) / s_target(role, tuple(D_s.shape), cfg_s)
        ratio = D_w[:r, :c] / D_s
        assert torch.allclose(ratio, torch.full_like(ratio, float(expected)), atol=1e-5, rtol=1e-4), (role, l)
        n_checked += 1
    assert n_checked >= 6
