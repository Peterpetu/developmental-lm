"""Acceptance rows C1-C7, C9-C11 of docs/tempo/06 on consolidation (C8 and C12 need
the life loop and live in test_life.py).

SPEC: 06 §6.11
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import stats

from tempo.consolidate import (
    ACCEPTED_MAX, SIGMA_MIN, TAU_FACTOR, Consolidator, Reinforce, Rehearsal, RestEM, WindowSummaries, env_loss,
    es_proposal, genome_step, kl_diag, L_check, reward, salience_fit, salience_targets, trust_region,
)

D, DX, DG, DC = 8, 6, 12, 16


def _cons(seed=0):
    return Consolidator(D, DX, DG, d_c=DC, n_layers=2, n_heads=2, d_ff=32, chunk_size=4, seed=seed)


def _summaries(n_steps=3, B=4, N=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    ws = WindowSummaries(D, DX, N)
    for n in range(n_steps):
        ws.append(torch.randn(B, D, generator=g), torch.rand(B, generator=g) + 1.0,
                  torch.randn(B, DX, generator=g), n, [("text", n * B + i) for i in range(B)], ["text"] * B)
    return ws


class TestC1Summaries:
    def test_shapes_and_env_losses(self):
        ws = _summaries()
        assert ws.n_windows == 12
        r = ws.inputs()
        assert r.shape == (12, D + 1 + DX + 1)
        assert torch.allclose(r[:, D], ws.losses())
        assert torch.allclose(r[:, -1], ws.steps().float() / ws.N)
        assert abs(ws.mean_loss() - float(ws.losses().mean())) < 1e-6
        last = ws.mean_loss(last_frac=0.34)                              # the last step of N = 3
        assert abs(last - float(ws.losses()[ws.steps() == 2].mean())) < 1e-6
        e = env_loss(torch.tensor([2.0, -1.0]), 4.0)
        assert torch.allclose(e, torch.tensor([-0.5, 0.25]))
        d = ws.state_dict()
        ws2 = WindowSummaries.from_state_dict(d)
        assert torch.equal(ws2.inputs(), ws.inputs()) and ws2.ident == ws.ident

    def test_rehearsed_windows_are_excluded_and_rescale_is_per_stream(self):
        ws = WindowSummaries(D, DX, 2)
        ws.append(torch.zeros(2, D), torch.tensor([1.0, 3.0]), torch.zeros(2, DX), 0,
                  [("text", 0), ("control", 1)], ["text", "control"], rehearsed=[True, False])
        ws.append(torch.zeros(2, D), torch.tensor([5.0, 7.0]), torch.zeros(2, DX), 1,
                  [("text", 2), ("control", 3)], ["text", "control"])
        assert ws.mean_loss() == (3.0 + 5.0 + 7.0) / 3
        assert ws.mean_loss(exclude_rehearsed=False) == 4.0
        ws.rescale("control", 0.5)
        assert ws.losses().tolist() == [1.0, 1.5, 5.0, 3.5]


class TestC2ReverseOrder:
    def test_mixer_sees_the_summaries_last_to_first(self):
        cons = _cons()
        ws = _summaries()
        r_in = ws.inputs()
        seen = {}
        orig = cons.mixer

        def spy(seq):
            seen["seq"] = seq.detach().clone()
            return orig(seq)

        cons.mixer = spy
        g = torch.zeros(DG)
        R, c = cons.read(r_in, g, reverse=True)
        expected = torch.nn.functional.linear(r_in, cons.W_r).flip(0)
        assert torch.allclose(seen["seq"][0], expected)
        assert c.shape == (12, DC) and R.shape == (DC,)
        R2, c2 = cons.read(r_in, g, reverse=False)
        assert torch.allclose(seen["seq"][0], torch.nn.functional.linear(r_in, cons.W_r))
        assert not torch.allclose(c, c2)

    def test_the_read_out_is_the_last_reached_output_in_both_orders(self):
        cons = _cons()
        r_in = _summaries().inputs()
        feat_r, c_r = cons.features(r_in, reverse=True)
        feat_f, c_f = cons.features(r_in, reverse=False)
        assert torch.allclose(feat_r[:DC], c_r[0]) and torch.allclose(feat_r[DC:], c_r.mean(0))     # reversed: window 0, reached last
        assert torch.allclose(feat_f[:DC], c_f[-1]) and torch.allclose(feat_f[DC:], c_f.mean(0))   # forward: the last window


class TestC3C4ProposalAndStep:
    def test_c3_proposal_at_birth_draws_a_norm_tau_step_and_the_scale_head_is_alive(self):
        """(6.4): at birth `mu = 0` and `sigma` sits just inside the top of `[0.02, 0.2]`, so the
        typical draw has norm `tau` and the trust region does not bind.

        And -- the property two earlier forms of this head silently lost -- **the scale head has
        a gradient**. A `clamp` whose bound the birth pre-activation already violates has zero
        gradient there, so `W_sigma` could never move and every proposal for the whole life had
        the same scale (11 §P-C.36). The squashing map is differentiable everywhere.
        """
        cons = _cons()
        R = torch.randn(DC)
        mu, ls = cons.propose(R)
        assert torch.count_nonzero(mu) == 0
        sigma = float(ls.exp()[0])
        assert 0.95 * TAU_FACTOR < sigma < TAU_FACTOR, sigma        # just inside the top
        assert SIGMA_MIN < sigma < TAU_FACTOR
        g = torch.Generator().manual_seed(0)
        draws = torch.stack([cons.sample(mu, ls, g) for _ in range(10_000)])
        se = 1 / math.sqrt(10_000)
        assert draws.mean(0).abs().max() < TAU_FACTOR * (3 * se * 1.5 + 0.03)
        assert (draws.var(0) - sigma ** 2).abs().max() < sigma ** 2 * (3 * math.sqrt(2.0) * se * 1.5 + 0.03)
        tau = TAU_FACTOR * math.sqrt(DG)
        assert abs(float(draws.norm(dim=1).mean()) - tau) < 0.1 * tau

        # 11 §P-D.8: the head is deliberately FROZEN. It was clamped and dead twice, then made
        # alive, and alive it collapsed to the floor within one boundary -- the null probe then
        # winning 6 of 10 episodes. With `G` candidates and one elite there is no spread to
        # estimate a scale from, so the scale is a constant of the design at `tau / sqrt(d_g)`.
        assert cons.scale_is_frozen()
        cons.zero_grad()
        mu_, ls_ = cons.propose(R)
        cons.log_prob(torch.randn(DG) * 0.05, mu_, ls_).backward()
        assert cons.W_sigma.grad is None and cons.b_sigma.grad is None
        assert [id(p) for p in cons.head_parameters()] == [id(cons.W_R), id(cons.W_mu)]
        # the squashing map is still correct where it matters -- it simply is not trained
        with torch.no_grad():
            cons.b_sigma.fill_(-20.0)
        assert float(cons.propose(R)[1].exp()[0]) < 1.05 * SIGMA_MIN                 # reaches the floor
        with torch.no_grad():
            cons.b_sigma.fill_(20.0)
        assert float(cons.propose(R)[1].exp()[0]) > 0.99 * TAU_FACTOR                # and the ceiling

    def test_c3b_the_elite_regression_moves_the_mean_and_leaves_the_scale_alone(self):
        """11 §P-D.8, and the inversion of an earlier test. The elite regression must pull the
        MEAN onto its elites and must NOT move the scale.

        The earlier form asserted the opposite -- that the regression could concentrate `sigma`
        -- because at the time the defect was a clamp with zero gradient. Made alive, the same
        mechanism collapsed `sigma` to its floor after a single boundary (measured over ten TINY
        episodes: 0.191 -> 0.020, the null probe winning 4 of 10), because `W_mu` is
        over-parameterised against an accepted set of at most 8 pairs by 32x at the TS shape, so
        the mean fits every elite exactly and the MLE scale of a zero residual is zero. With one
        elite per generation there is no spread to estimate a scale from at all."""
        cons = _cons()
        r_in = _summaries().inputs()
        g = torch.zeros(DG)
        sigma0 = float(cons.propose(cons.read(r_in, g)[0])[1].exp()[0])
        est = RestEM(steps=400, lr=1e-2, lambda_kl=0.0)
        elite = torch.full((DG,), 0.05)                       # an elite the mean must move toward
        with torch.no_grad():
            feat, _ = cons.features(r_in)
        est.accepted = [(feat.clone(), g.clone(), elite.clone())]
        with torch.no_grad():
            olds = [cons.propose(cons.readout(feat, g))]
            mu0 = cons.propose(cons.read(r_in, g)[0])[0].clone()
        est._heads_fit(cons, est.accepted, [(m.detach(), l.detach()) for m, l in olds])
        mu1, ls1 = cons.propose(cons.read(r_in, g)[0])
        sigma1 = float(ls1.exp()[0])
        # the mean concentrates on the elite ...
        assert float((mu1 - elite).norm()) < float((mu0 - elite).norm())
        # ... and the scale does not move at all
        assert sigma1 == pytest.approx(sigma0, rel=1e-6), (sigma0, sigma1)
        assert sigma1 == pytest.approx(TAU_FACTOR, rel=0.05)  # still tau/sqrt(d_g) per coordinate

    def test_c4_trust_region_decay_and_bound_about_the_prior(self):
        tau = 0.2 * math.sqrt(DG)
        big = torch.randn(DG) * 10
        assert abs(float(trust_region(big, tau).norm()) - tau) < 1e-6
        small = torch.randn(DG) * 0.01
        assert torch.allclose(trust_region(small, tau), small)
        g, g0 = torch.randn(DG), torch.randn(DG)
        g2 = genome_step(g, torch.zeros(DG), tau=tau, lambda_dec=0.01, bound=100.0, prior=g0)
        assert torch.allclose(g2, g0 + 0.99 * (g - g0))
        g3 = genome_step(g0 + g * 100, torch.zeros(DG), tau=tau, lambda_dec=0.0, bound=3 * math.sqrt(DG), prior=g0)
        assert abs(float((g3 - g0).norm()) - 3 * math.sqrt(DG)) < 1e-4
        assert torch.allclose(genome_step(g, torch.zeros(DG), tau=tau, lambda_dec=0.01, bound=100.0), 0.99 * g)


class TestC5Reward:
    def test_planted(self):
        assert abs(reward(2.0, 1.9, 2.0, 1.0) - 0.05) < 1e-12
        assert abs(L_check({"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 0.5}) - 2.0) < 1e-12
        assert abs(L_check({"a": 2.0}, {}) - 2.0) < 1e-12
        ws = WindowSummaries(D, DX, 10)
        for n in range(10):
            ws.append(torch.zeros(2, D), torch.tensor([float(n), float(n)]), torch.zeros(2, DX), n,
                      [("text", 2 * n), ("task", 2 * n + 1)], ["text", "task"])
        assert ws.mean_loss(last_frac=0.1) == 9.0                      # the last 10 % of steps
        assert ws.mean_loss(stream="task") == 4.5
        with pytest.raises(ValueError):
            reward(1.0, 1.0, 0.0, 1.0)
        with pytest.raises(ValueError):
            L_check({}, {})


class TestC6C7Estimators:
    def test_c6_reinforce_gradient_and_baseline_and_kl(self):
        cons = _cons()
        with torch.no_grad():
            cons.W_mu.normal_(0, 0.1)
            cons.W_sigma.normal_(0, 0.1)
        ws = _summaries()
        r_in = ws.inputs()
        g = torch.randn(DG)
        with torch.no_grad():
            R, _ = cons.read(r_in, g)
            mu, ls = cons.propose(R)
        dg = cons.sample(mu, ls, torch.Generator().manual_seed(1))
        # the module's log-prob equals torch.distributions'
        ref = torch.distributions.Normal(mu, ls.exp()).log_prob(dg).sum()
        assert abs(float(cons.log_prob(dg, mu, ls)) - float(ref)) < 1e-5
        # the gradient of (6.9) on W_mu equals (r - b) * d log N / d W_mu
        est = Reinforce(baseline=0.3)
        cons.zero_grad()
        R_, _ = cons.read(r_in, g)
        mu_, ls_ = cons.propose(R_)
        (0.7 * cons.log_prob(dg, mu_, ls_)).backward()
        manual = cons.W_mu.grad.clone()
        opt = torch.optim.SGD(cons.parameters(), lr=0.0)
        est.update(cons, opt, r_in, g, dg, r_e=1.0)
        assert torch.allclose(cons.W_mu.grad, -manual, atol=1e-6)
        assert abs(est.baseline - (0.9 * 0.3 + 0.1 * 1.0)) < 1e-12
        # the KL closed form against torch.distributions
        a, b = torch.randn(DG), torch.randn(DG) * 0.3
        c, d = torch.randn(DG), torch.randn(DG) * 0.3
        ref_kl = torch.distributions.kl_divergence(torch.distributions.Normal(a, b.exp()),
                                                   torch.distributions.Normal(c, d.exp())).sum()
        assert abs(float(kl_diag(a, b, c, d)) - float(ref_kl)) < 1e-5

    def test_c7_elite_regression_selects_the_argmax_and_moves_toward_accepted(self):
        cons = _cons()
        ws = _summaries()
        r_in = ws.inputs()
        g = torch.randn(DG)
        est = RestEM(steps=50, lr=1e-2, lambda_kl=0.1)
        assert est.select([0.1, 0.5, -0.2, 0.4]) == 1
        dg_star = torch.randn(DG) * 0.2
        with torch.no_grad():
            ll0 = float(cons.log_prob(dg_star, *cons.propose(cons.read(r_in, g)[0])))
        est.update(cons, r_in, g, dg_star)
        assert len(est.accepted) == 1
        ll1 = est.log_likelihood(cons)
        assert ll1 > ll0
        with torch.no_grad():
            mu, ls = cons.propose(cons.read(r_in, g)[0])
        assert float((mu - dg_star).norm()) < float(dg_star.norm())
        # later episodes: D grows and is bounded; the past pairs keep their (cached) read-outs
        for i in range(ACCEPTED_MAX + 2):
            info = est.update(cons, r_in * (0.5 + 0.1 * i), g, torch.randn(DG) * 0.2)
        assert info["n_accepted"] == ACCEPTED_MAX and len(est.accepted) == ACCEPTED_MAX
        assert math.isfinite(est.log_likelihood(cons))

    def test_grpo_variant_moves_toward_the_better_candidate(self):
        cons = _cons()
        r_in = _summaries().inputs()
        g = torch.randn(DG)
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            mu, ls = cons.propose(cons.read(r_in, g)[0])
        cands = [cons.sample(mu, ls, gen) for _ in range(4)]
        rewards = [0.0, 1.0, 0.0, 0.0]
        with torch.no_grad():
            lp0 = [float(cons.log_prob(c, mu, ls)) for c in cands]
        est = RestEM()
        for _ in range(20):
            est.update_grpo(cons, r_in, g, cands, rewards, lr=1e-2)
        with torch.no_grad():
            mu2, ls2 = cons.propose(cons.read(r_in, g)[0])
            lp1 = [float(cons.log_prob(c, mu2, ls2)) for c in cands]
        assert lp1[1] - lp0[1] > max(lp1[i] - lp0[i] for i in (0, 2, 3))
        assert est.update_grpo(cons, r_in, g, cands, [1.0] * 4) == {"skipped": True}


class TestC9C10C11:
    def test_c9_null_step_leaves_g(self):
        g, g0 = torch.randn(DG), torch.randn(DG)
        assert torch.equal(genome_step(g, torch.zeros(DG), tau=1.0, lambda_dec=0.0, bound=1e9), g)
        assert torch.allclose(genome_step(g, torch.zeros(DG), tau=1.0, lambda_dec=0.0, bound=1e9, prior=g0), g)

    def test_c10_salience_regression_buffer_and_draw(self):
        torch.manual_seed(0)
        cons = _cons()
        n = 300
        c = torch.randn(n, DC)
        w_true = torch.randn(DC)
        delta_ell = c @ w_true + 0.1 * torch.randn(n)
        target = salience_targets(delta_ell)
        mixer_before = [p.detach().clone() for p in cons.mixer_parameters()]
        salience_fit(cons, c, target, steps=200, lr=5e-2)
        assert all(torch.equal(a, b) for a, b in zip(mixer_before, cons.mixer_parameters()))   # w_s alone moves
        pred = cons.salience(c).detach()
        r2 = 1 - (pred - target).square().sum() / (target - target.mean()).square().sum()
        assert float(r2) > 0.9, float(r2)
        buf = Rehearsal(n_reh=10)
        idents = [("text", i) for i in range(n)]
        # 11 §P-D.12: the buffer carries each window's wall clock beside its identity, and
        # `draw` returns `(ident, t0)` pairs -- a replayed window must be presented at the time
        # it actually had, because 02 (2.13) makes the time code a function of it.
        t0s = [float(i) * 0.25 for i in range(n)]
        buf.build(idents, pred, t0=t0s)
        assert len(buf) == 10 and buf.mode == "salience"
        top = set(int(i) for i in torch.topk(pred, 10).indices)
        assert {i for _, i in buf.ident} == top
        assert buf.t0 == [t0s[i] for _, i in buf.ident]        # the clock travels with the identity
        g = torch.Generator().manual_seed(0)
        counts = np.zeros(10)
        for _ in range(10_000):
            ident, t0 = buf.draw(1, g)[0]
            assert t0 == t0s[ident[1]]
            j = buf.ident.index(ident)
            counts[j] += 1
        expected = (buf.s.double() / buf.s.double().sum()).numpy()
        expected = expected / expected.sum() * counts.sum()
        p = stats.chisquare(counts, expected).pvalue
        assert p > 0.01, p
        d = buf.state_dict()
        buf2 = Rehearsal(1)
        buf2.load_state_dict(d)
        assert buf2.ident == buf.ident
        # the uniform arm and the degenerate-salience guard: a uniform sample, drawn uniformly
        uni = Rehearsal(n_reh=10)
        uni.build(idents, pred, generator=torch.Generator().manual_seed(1), uniform=True)
        assert uni.mode == "uniform" and len(uni) == 10 and torch.equal(uni.s, torch.ones(10))
        zero = Rehearsal(n_reh=10)
        zero.build(idents, torch.zeros(n), generator=torch.Generator().manual_seed(1))
        assert zero.mode == "uniform" and zero.ident == uni.ident

    def test_c11_es_on_a_quadratic_takes_norm_tau_steps(self):
        torch.manual_seed(0)
        g_star = torch.randn(DG) * 3
        g = torch.zeros(DG)
        tau = 0.2 * math.sqrt(DG)
        sigma = tau / math.sqrt(DG)
        gen = torch.Generator().manual_seed(0)
        for _ in range(8):
            eps = torch.randn(8, DG, generator=gen)
            rp = -((g + sigma * eps) - g_star).square().sum(1)
            rm = -((g - sigma * eps) - g_star).square().sum(1)
            dg = es_proposal(eps, rp, rm, sigma_es=sigma, tau=tau)
            assert abs(float(dg.norm()) - tau) < 1e-5
            before = float((g - g_star).norm())
            g = g + dg
            assert float((g - g_star).norm()) < before
        assert torch.count_nonzero(es_proposal(eps, rp, rp, sigma_es=sigma, tau=tau)) == 0
