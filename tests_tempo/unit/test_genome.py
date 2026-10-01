"""Acceptance rows E1-E8, E10, E12 of docs/tempo/04 on the genome and its expression
(E9 and E11 need the life loop and live in test_life.py).

SPEC: 04 §4.10
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from anlage.data import systematic_counts
from tempo.backbone import Backbone
from tempo.config import TINY
from tempo.genome import (
    DIFFERENTIABLE_KINDS, GENOME_NORM_BOUND, RANGES, EpisodeGains, Expression, RoleRegistry,
    backbone_gains, genome_prior, lr_multipliers, mixture, svf_effective,
)
from tempo.morphology import SwitchPattern


@pytest.fixture
def parts():
    cfg = TINY
    bb = Backbone(cfg)
    reg = RoleRegistry(bb)
    g0 = genome_prior(cfg.d_g)
    ex = Expression(reg, cfg.d_g, cfg.r, g0=g0)
    eg = EpisodeGains(reg)
    return cfg, bb, reg, ex, eg


class TestE1Registry:
    def test_every_matrix_has_one_role_and_nothing_else_does(self, parts):
        cfg, bb, reg, _, _ = parts
        names = {r.name for r in reg.roles if r.kind == "matrix"}
        assert names == set(bb.matrix_names())
        for n, p in bb.W.items():
            if p.dim() == 2 and n.split("_")[-1] not in ("CQ", "CK", "CV"):
                assert n in names
            if n.endswith(("CQ", "CK", "CV")) or n == "IN_b":
                assert n not in reg.by_name
        for r in reg.roles:
            if r.kind == "matrix":
                assert r.size == r.n_in + r.n_out == sum(bb.W[r.name].shape)
        tab = reg.count_table()
        assert tab["offset"] == (3, 2 * cfg.h * cfg.n_delta_layers + cfg.L)
        assert tab["morph"] == (3, 2 + 2 * cfg.L) and tab["mixture"] == (1, 7) and tab["lr"] == (1, 4)
        assert sum(v[1] for v in tab.values()) == reg.n_gamma


class TestE2E3Identity:
    def test_e2_identity_at_birth(self, parts):
        cfg, bb, reg, ex, eg = parts
        g = genome_prior(cfg.d_g)                                   # the prior: the expression is zero there
        eg.set_from(ex.express(g))
        gamma = eg.gamma()
        assert torch.count_nonzero(gamma) == 0
        gains = backbone_gains(reg, gamma)
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        y0, _, _, _ = bb(x)
        y1, _, _, _ = bb(x, gains=gains, zeta=torch.zeros(2 * cfg.L))
        assert torch.equal(y0, y1)
        assert lr_multipliers(reg, gamma) == [1.0] * 4
        mu = mixture(reg, gamma)
        assert torch.allclose(mu, torch.full((7,), 1 / 7, dtype=torch.float64))
        pat = SwitchPattern.from_gains(cfg, gamma[reg.slice("MORPH_Q")], gamma[reg.slice("MORPH_H")],
                                       gamma[reg.slice("MORPH_F")])
        assert pat == SwitchPattern.primordium(cfg)

    def test_e3_ranges(self, parts):
        cfg, bb, reg, ex, eg = parts
        torch.manual_seed(0)
        with torch.no_grad():
            ex.B_flat.normal_(0, 3.0)
        d = torch.randn(cfg.d_g)
        g = genome_prior(cfg.d_g) + d / d.norm() * GENOME_NORM_BOUND(cfg.d_g)   # at the bound about the prior
        gamma = reg.gamma(ex.express(g))
        for r in reg.roles:
            seg = gamma[r.offset:r.offset + r.size]
            assert seg.abs().max() <= RANGES[r.kind]          # tanh saturates to 1 in fp32
        gains = backbone_gains(reg, gamma)
        for k, v in gains.items():
            if ":" in k or "MIX" in k or "FFN" in k or "rho" in k:
                assert (1.0 + v > 0).all()


class TestE4E5Gradients:
    def test_e4_expression_is_live(self, parts):
        cfg, bb, reg, ex, eg = parts
        with torch.no_grad():
            ex.B_flat.normal_(0, 0.1)
        g = torch.randn(cfg.d_g)
        u0 = ex.express(g)
        u1 = ex.express(g + 1e-2 * torch.randn(cfg.d_g))
        assert (u0 - u1).abs().max() > 0
        x = torch.randn(1, cfg.T_s, cfg.d_x)
        eg.set_from(u0)
        y0, _, _, _ = bb(x, gains=backbone_gains(reg, eg.gamma()))
        eg.set_from(u1)
        y1, _, _, _ = bb(x, gains=backbone_gains(reg, eg.gamma()))
        assert (y0 - y1).abs().max() > 0
        # liveness at the bound about the prior: tanh is not saturated (A ~ N(0, 1/(4 d_g)))
        d = torch.randn(cfg.d_g)
        gb = genome_prior(cfg.d_g) + d / d.norm() * GENOME_NORM_BOUND(cfg.d_g)
        assert (ex.express(gb) - ex.express(gb + 1e-2 * torch.randn(cfg.d_g))).abs().max() > 1e-6

    def test_e4b_programme_kinds_move_under_a_genome_step(self, parts):
        """11 §P-C.25: the programme kinds (lr, morphology, mixture) are zero at the prior and
        move with the genome step through the fixed frame, with B = 0 for every other role."""
        cfg, bb, reg, ex, eg = parts
        g0 = genome_prior(cfg.d_g)
        u0 = ex.express(g0)
        assert torch.count_nonzero(u0) == 0
        tau = 0.2 * (cfg.d_g ** 0.5)
        d = torch.randn(cfg.d_g)
        u1 = ex.express(g0 + tau * d / d.norm())
        prog = ~reg.diff_of_row
        assert (u1[prog]).abs().max() > 1e-3                         # the programme rows moved
        assert torch.count_nonzero(u1[~prog]) == 0                  # the differentiable rows did not (B = 0)
        gamma = reg.gamma(u1)
        a = gamma[reg.slice("MORPH_H")]
        assert a.abs().max() > 0.05                                  # of the order of a switch's threshold (0.108)

    def test_e5_no_gradient_reaches_g(self, parts):
        cfg, bb, reg, ex, eg = parts
        g = torch.randn(cfg.d_g)
        assert not g.requires_grad
        eg.set_from(ex.express(g))
        x = torch.randn(1, cfg.T_s, cfg.d_x)
        y, _, _, _ = bb(x, gains=backbone_gains(reg, eg.gamma()))
        y.square().sum().backward()
        assert g.grad is None
        assert eg.u_diff.grad is not None and eg.u_diff.grad.abs().sum() > 0
        assert not eg.u_prog.requires_grad
        assert ex.A.grad is None and ex.B_flat.grad is None     # theta is not in the step graph


class TestE6E7Regression:
    def test_e6_regression_recovers_a_history_in_the_map_range(self, parts):
        cfg, bb, reg, ex, _ = parts
        torch.manual_seed(1)
        g0 = genome_prior(cfg.d_g)
        with torch.no_grad():
            ex.B_flat.normal_(0, 0.3)
        pairs = [(g0 + torch.randn(cfg.d_g), None) for _ in range(6)]
        pairs = [(g, ex.express(g)) for g, _ in pairs]
        other = Expression(reg, cfg.d_g, cfg.r, g0=g0, seed=9)
        with torch.no_grad():
            other.B_flat.normal_(0, 0.3)
        mask = reg.diff_of_row
        g, u = pairs[-1]
        r0 = float(((other(g) - u)[mask]).square().sum() / u[mask].square().sum())
        R = other.fit(pairs, steps=500, lr=1e-2)
        assert r0 > 0.5 and R < 1e-2, (r0, R)

    def test_e6b_a_single_pair_is_always_fittable_and_a_history_is_not(self, parts):
        """04 §4.5: at one point the fit is exact for any target; the history constrains it."""
        cfg, bb, reg, ex, _ = parts
        torch.manual_seed(2)
        g0 = genome_prior(cfg.d_g)
        one = [(g0 + torch.randn(cfg.d_g), torch.randn(reg.n_gamma))]
        R1 = ex.fit(one, steps=400, lr=1e-2)
        assert R1 < 1e-2, R1
        many = [(g0 + torch.randn(cfg.d_g), torch.randn(reg.n_gamma)) for _ in range(12)]
        ex2 = Expression(reg, cfg.d_g, cfg.r, g0=g0)
        R12 = ex2.fit(many, steps=400, lr=1e-2)
        assert R12 > 0.1, R12                                        # r = 4 cannot carry 12 random targets

    def test_e7_regeneration_reproduces_the_fitted_gains(self, parts):
        cfg, bb, reg, ex, eg = parts
        g = genome_prior(cfg.d_g) + torch.randn(cfg.d_g)
        with torch.no_grad():
            eg.u_diff.normal_(0, 0.2)
        target = eg.u_target()
        ex.fit([(g, target)], steps=300, lr=1e-2)
        eg2 = EpisodeGains(reg)
        eg2.set_from(ex.express(g))
        assert torch.allclose(eg2.u()[reg.diff_of_row], target[reg.diff_of_row], atol=1e-2)


class TestE10E12Programme:
    def test_e10_mixture_floor_and_systematic_counts(self, parts):
        cfg, bb, reg, ex, eg = parts
        gamma = torch.zeros(reg.n_gamma)
        gamma[reg.slice("MIX")] = torch.tensor([4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0])
        mu = mixture(reg, gamma, floor=0.02)
        assert abs(float(mu.sum()) - 1.0) < 1e-9 and float(mu.min()) >= 0.02 - 1e-12
        for _ in range(50):                                          # exact on adversarial logits
            gamma[reg.slice("MIX")] = torch.randn(7) * 6
            mu = mixture(reg, gamma, floor=0.02)
            assert abs(float(mu.sum()) - 1.0) < 1e-9 and float(mu.min()) >= 0.02 - 1e-12
        counts = systematic_counts(32, mu.numpy(), 0.37)
        assert counts.sum() == 32 and len(counts) == 7

    def test_e12_svf_at_zero_reproduces_w(self, parts):
        W = torch.randn(12, 7)
        assert torch.allclose(svf_effective(W, torch.zeros(7)), W, atol=1e-5)
        a = torch.full((7,), 0.5)
        assert torch.allclose(svf_effective(W, a), 1.5 * W, atol=1e-5)

    def test_lr_multipliers_clamp(self, parts):
        cfg, bb, reg, ex, eg = parts
        gamma = torch.zeros(reg.n_gamma)
        gamma[reg.slice("LR")] = torch.tensor([math.log(2.0), -3.0, 3.0, 0.0])
        m = lr_multipliers(reg, gamma)
        assert abs(m[0] - 2.0) < 1e-6 and m[1] == 0.25 and m[2] == 4.0 and m[3] == 1.0
