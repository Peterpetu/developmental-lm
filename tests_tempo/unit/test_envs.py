"""Acceptance rows I8, I10 (point-mass), I11 of docs/tempo/02 on the reward-bearing streams.

SPEC: 02 §2.6, §2.9.2, §2.9.3, §2.10
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from soma.openended.tasks import BOS, EOS, PAD, SEP, TaskParams, family_lower_bounds, solve
from tempo.envs import (
    MLPPolicy, PointMass, SymbolicTaskStream, grpo_loss, group_advantage, pad_to_spans,
)


class TestPointMass:
    def test_hand_computed_step(self):
        env = PointMass()
        s = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5, -0.5]])
        a = torch.tensor([[1.0, 0.0]])
        s2, r = env.step(s, a, 0)
        assert torch.allclose(s2[0, 2:4], torch.tensor([0.05, 0.0]))
        assert torch.allclose(s2[0, 0:2], torch.tensor([0.0025, 0.0]))
        expected = -math.hypot(0.0025 - 0.5, 0.0) - 0.01 * 1.0
        assert abs(float(r[0]) - expected) < 1e-6

    def test_collision_penalty_and_clipping(self):
        env = PointMass()
        s = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.005, 0.0]])
        a = torch.tensor([[5.0, 0.0]])                         # clipped to 1
        s2, r = env.step(s, a, 0)
        assert torch.allclose(s2[0, 2:4], torch.tensor([0.05, 0.0]))
        assert float(r[0]) < -5.0

    def test_reset_is_deterministic_and_keeps_the_obstacle_away(self):
        env = PointMass()
        s1 = env.reset(64, torch.Generator().manual_seed(0))
        s2 = env.reset(64, torch.Generator().manual_seed(0))
        assert torch.equal(s1, s2) and s1.shape == (64, 8)
        assert ((s1[:, 6:8] - s1[:, 0:2]).norm(dim=-1) >= 0.3).all()
        assert ((s1[:, 6:8] - s1[:, 4:6]).norm(dim=-1) >= 0.3).all()

    def test_rollout_shapes_and_manoeuvre(self):
        env = PointMass(manoeuvre=True, horizon=10)
        s0 = env.reset(4, torch.Generator().manual_seed(1))
        states, actions, rewards = env.rollout(lambda s, t: torch.zeros(s.shape[0], 2), s0)
        assert states.shape == (4, 11, 8) and actions.shape == (4, 10, 2) and rewards.shape == (4, 10)
        g0, g5 = env.goal_at(s0, 0), env.goal_at(s0, 5)
        assert torch.allclose((g0 - s0[:, 4:6]).norm(dim=-1), torch.full((4,), 0.5))
        assert not torch.allclose(g0, g5)
        assert env.observation(s0, 5).shape == (4, 8)

    def test_mlp_policy(self):
        pol = MLPPolicy()
        s = torch.randn(3, 8)
        a = pol.sample(s, torch.Generator().manual_seed(0))
        assert a.shape == (3, 2) and pol.log_prob(s, a).shape == (3,)


class TestI8PolicyGradient:
    def test_bandit_learns_the_rewarded_arm(self):
        logits = torch.zeros(3, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=0.1)
        g = torch.Generator().manual_seed(0)
        p0 = torch.softmax(logits, 0)[0].item()
        for _ in range(50):
            probs = torch.softmax(logits, 0)
            arms = torch.multinomial(probs.detach(), 8, replacement=True, generator=g)
            rewards = (arms == 0).float()
            adv, keep = group_advantage(rewards)
            if not keep:
                continue
            logp = torch.log(probs[arms])[:, None]
            loss = grpo_loss(logp, logp.detach(), adv, torch.ones(8, 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert torch.softmax(logits, 0)[0].item() > p0 + 0.3

    def test_dropped_groups_give_zero_gradient(self):
        adv, keep = group_advantage(torch.ones(8))
        assert not keep and torch.count_nonzero(adv) == 0
        logits = torch.zeros(3, requires_grad=True)
        logp = torch.log_softmax(logits, 0)[torch.zeros(8, dtype=torch.long)][:, None]
        loss = grpo_loss(logp, logp.detach(), adv, torch.ones(8, 1))
        loss.backward()
        assert torch.count_nonzero(logits.grad) == 0

    def test_clip_is_asymmetric_and_masked_tokens_do_not_count(self):
        adv = torch.tensor([1.0, -1.0])
        old = torch.zeros(2, 3)
        new = torch.log(torch.tensor([[1.5, 1.5, 1.5], [0.5, 0.5, 0.5]]))
        loss_all = grpo_loss(new, old, adv, torch.ones(2, 3))
        # positive advantage: ratio 1.5 clipped to 1.28 -> min(1.5, 1.28) = 1.28
        # negative advantage: ratio 0.5 -> min(-0.5, -0.8) = -0.8
        assert abs(float(loss_all) - (-(3 * 1.28 + 3 * -0.8) / 6)) < 1e-6
        mask = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        loss_m = grpo_loss(new, old, adv, mask)
        assert abs(float(loss_m) - (-1.28)) < 1e-6


class TestSymbolicTasks:
    def test_instances_verify_against_their_own_answers(self):
        st = SymbolicTaskStream(T_ctx=256, seed=0)
        for _ in range(30):
            inst = st.sample_instance()
            assert st.reward(inst, inst.answer) == 1
            assert st.reward(inst, inst.answer + [EOS, 5]) == 1
            assert st.reward(inst, inst.answer[:-1] if inst.answer else [3]) == 0 or not inst.answer
            assert st.prompt_tokens(inst)[0] == BOS and st.prompt_tokens(inst)[-1] == SEP

    def test_padding_to_spans(self):
        toks, mask = pad_to_spans([1, 2, 3, 4, 5], [False, False, True, True, True], 4)
        assert toks.tolist() == [1, 2, 3, 4, 5, PAD, PAD, PAD]
        assert mask.tolist() == [False, False, True, True, True, False, False, False]

    def test_archive_seeding_with_a_planted_solve_rate(self):
        st = SymbolicTaskStream(T_ctx=256, seed=0)
        st.seed_archive(lambda params, n: 0.5)
        assert len(st.archive) == len(family_lower_bounds())
        p = st.sample_params()
        assert isinstance(p, TaskParams)
        out = st.archive_step(lambda params, n: 0.5, 1)
        assert isinstance(out, dict)
