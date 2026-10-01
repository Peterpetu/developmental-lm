"""Acceptance rows M1-M4, M6-M10 of docs/tempo/05 on latent morphology (M5's timing is
measured in R1; its hook clause is here).

SPEC: 05 §5.8
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from tempo.backbone import Backbone
from tempo.config import TINY, TS_ADULT, TempoConfig
from tempo.morphology import (
    A_BASE, A_SKIP, EPS_SKIP, SwitchPattern, apply_births, born_flags_at_birth, control_pattern,
    flops_per_span, flops_per_window, hypercloning_split, n_active_params,
)


def _on(cfg, groups: set[str]) -> SwitchPattern:
    lm = tuple(1.0 if (l in cfg.primordium_layers or f"Q{cfg.quarter_of(l)}" in groups) else 0.0
               for l in range(cfg.L))
    hm = tuple(1.0 if f"H{l}" in groups else 0.0 for l in range(cfg.L))
    fm = tuple(1.0 if f"F{l}" in groups else 0.0 for l in range(cfg.L))
    return SwitchPattern(cfg, lm, hm, fm)


class TestM1M2:
    def test_m1_switching_a_group_off_equals_the_backbone_without_it(self):
        cfg = TINY
        bb = Backbone(cfg)
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        S0 = {l: torch.randn(2, cfg.h, cfg.d_head, cfg.d_head) * 0.1 for l in range(cfg.L) if cfg.op(l) == "delta"}
        prim = SwitchPattern.primordium(cfg)
        y_ref, _, _, _ = bb(x, state=S0, pattern=prim)
        for grp in ("Q1", "H0", "F0", "Q3"):
            pat_on = _on(cfg, {grp})
            pat_tiny = SwitchPattern(cfg, tuple(1e-30 if (m == 1.0 and l not in cfg.primordium_layers) else m
                                                for l, m in enumerate(pat_on.layer_m)),
                                     tuple(1e-30 if m == 1.0 else 0.0 for m in pat_on.head_m),
                                     tuple(1e-30 if m == 1.0 else 0.0 for m in pat_on.slice_m))
            y_on, _, _, _ = bb(x, state=S0, pattern=pat_on)
            y_tiny, _, _, _ = bb(x, state=S0, pattern=pat_tiny)
            assert not torch.allclose(y_on, y_ref, atol=1e-6), grp          # the group does something at m = 1
            assert torch.allclose(y_tiny, y_ref, atol=1e-6), grp            # and nothing at m -> 0

    def test_m2_a_born_block_learns_at_rate_m(self):
        cfg = TINY
        bb = Backbone(cfg)
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        l = cfg.primordium_layers[0]
        n0 = cfg.primordium_heads * cfg.d_head

        def grad_norm(m):
            pat = SwitchPattern(cfg, SwitchPattern.primordium(cfg).layer_m,
                                tuple(m if k == l else 0.0 for k in range(cfg.L)), (0.0,) * cfg.L)
            bb.zero_grad()
            y, _, _, _ = bb(x, pattern=pat)
            y.sum().backward()
            return bb.W[f"L{l}_Q"].grad[n0:].norm().item()

        # G2 is local: the block's gradient is m times its output sensitivity at the current
        # network state; doubling m from eps_skip doubles it to O(m) (the downstream state moves by O(m))
        g1, g2 = grad_norm(EPS_SKIP), grad_norm(2 * EPS_SKIP)
        assert abs(g2 / g1 - 2.0) < 0.05, (g1, g2)
        g_full = grad_norm(1.0)
        assert g1 < 0.1 * g_full


class TestM3M8Births:
    def test_m3_births_copy_with_one_percent_noise(self):
        cfg = TINY
        bb = Backbone(cfg)
        before = {n: p.detach().clone() for n, p in bb.W.items()}
        born = born_flags_at_birth(cfg)
        prim = SwitchPattern.primordium(cfg)
        new = _on(cfg, {"H0", "F0", "Q1"})
        out = apply_births(bb, prim, new, born, torch.Generator().manual_seed(0))
        assert set(out) == {"Q1", "H0", "F0"} and born["H0"] and born["Q1"] and not born["H2"]
        n0, dh = cfg.primordium_heads * cfg.d_head, cfg.d_head
        src = before["L0_Q"][:n0]
        dst = bb.W["L0_Q"].detach()[n0:2 * n0]
        ratio = (dst - src).pow(2).mean().sqrt() / src.pow(2).mean().sqrt()
        assert 0.005 < ratio < 0.02
        assert torch.equal(bb.W["L0_Q"].detach()[:n0], src)                  # the source is unchanged
        F0 = cfg.primordium_slices * cfg.slice
        assert not torch.equal(bb.W["L0_GATE"].detach()[F0:], before["L0_GATE"][F0:])
        assert torch.equal(bb.W["L0_GATE"].detach()[:F0], before["L0_GATE"][:F0])
        # the newborn quarter-stack copies layer l - L/4 (the same relative depth, the same operator)
        for l in cfg.layers_of_quarter(1):
            src_l = l - cfg.quarter
            assert cfg.op(src_l) == cfg.op(l)
            d = (bb.W[f"L{l}_UP"].detach() - before[f"L{src_l}_UP"]).pow(2).mean().sqrt()
            assert d / before[f"L{src_l}_UP"].pow(2).mean().sqrt() < 0.02
            if cfg.op(l) == "delta":
                for role in ("CQ", "CK", "CV", "A0", "G0"):          # taps and offsets copied exactly
                    assert torch.equal(bb.W[f"L{l}_{role}"].detach(), before[f"L{src_l}_{role}"])
        # taps of a born head group are exact copies
        assert torch.equal(bb.W["L0_CQ"].detach()[n0:2 * n0], before["L0_CQ"][:n0])

    def test_births_ignore_groups_of_skipped_layers(self):
        """05 §5.1: a head of an inactive layer is inactive; it is neither born nor flagged."""
        cfg = TINY
        bb = Backbone(cfg)
        born = born_flags_at_birth(cfg)
        prim = SwitchPattern.primordium(cfg)
        l_off = cfg.layers_of_quarter(1)[0]
        raised = SwitchPattern(cfg, prim.layer_m, tuple(1.0 if k == l_off else 0.0 for k in range(cfg.L)), (0.0,) * cfg.L)
        before = bb.W[f"L{l_off}_Q"].detach().clone()
        out = apply_births(bb, prim, raised, born, torch.Generator().manual_seed(0))
        assert out == [] and not born[f"H{l_off}"] and torch.equal(bb.W[f"L{l_off}_Q"].detach(), before)

    def test_m8_reactivation_is_not_a_rebirth(self):
        cfg = TINY
        bb = Backbone(cfg)
        born = born_flags_at_birth(cfg)
        prim = SwitchPattern.primordium(cfg)
        on = _on(cfg, {"H0"})
        apply_births(bb, prim, on, born, torch.Generator().manual_seed(0))
        after_birth = bb.W["L0_Q"].detach().clone()
        with torch.no_grad():
            bb.W["L0_Q"].add_(1.0)
        out = apply_births(bb, prim, on, born, torch.Generator().manual_seed(1))
        assert out == []
        assert torch.equal(bb.W["L0_Q"].detach(), after_birth + 1.0)


class TestM4M5Order:
    def test_execution_order_and_skipped_layers_never_run(self):
        cfg = TINY
        bb = Backbone(cfg)
        calls = []
        orig = bb._ffn

        def spy(l, hn, pattern, gains):
            calls.append(l)
            return orig(l, hn, pattern, gains)

        bb._ffn = spy
        x = torch.randn(1, cfg.T_s, cfg.d_x)
        bb(x, pattern=SwitchPattern.primordium(cfg))
        assert calls == list(cfg.primordium_layers)
        calls.clear()
        bb(x, pattern=_on(cfg, {"Q1"}))
        assert calls == sorted(set(cfg.primordium_layers) | set(cfg.layers_of_quarter(1)))


class TestM6M9Pattern:
    def test_m6_roundtrip_and_key(self):
        cfg = TINY
        pat = _on(cfg, {"H1", "F2"})
        assert SwitchPattern.from_dict(cfg, pat.to_dict()) == pat
        assert pat.key() != SwitchPattern.primordium(cfg).key()

    def test_from_gains_thresholds(self):
        cfg = TINY
        z = torch.zeros(cfg.L)
        p0 = SwitchPattern.from_gains(cfg, torch.zeros(2), z, z)
        assert p0 == SwitchPattern.primordium(cfg)
        on = SwitchPattern.from_gains(cfg, torch.tensor([8.0, 0.0]), z, z)
        assert on.layer_active(cfg.layers_of_quarter(1)[0]) and not on.layer_active(cfg.layers_of_quarter(3)[0])
        assert abs(on.layer_m[cfg.layers_of_quarter(1)[0]] - 1 / (1 + math.exp(-(A_BASE + 8.0)))) < 1e-9
        edge = SwitchPattern.from_gains(cfg, torch.tensor([A_SKIP - A_BASE - 1e-6, A_SKIP - A_BASE + 1e-6]), z, z)
        assert not edge.layer_active(cfg.layers_of_quarter(1)[0]) and edge.layer_active(cfg.layers_of_quarter(3)[0])

    def test_m9_primordium(self):
        for cfg in (TINY, TS_ADULT):
            pat = SwitchPattern.primordium(cfg)
            assert pat.active_layers() == cfg.primordium_layers
            assert all(pat.h_act(l) == cfg.primordium_heads for l in range(cfg.L))
        assert TS_ADULT.primordium_layers == (0, 1, 2, 3, 8, 9, 10, 11)
        assert n_active_params(TS_ADULT, SwitchPattern.primordium(TS_ADULT)) == 17_886_264


class TestM7Flops:
    def test_hand_computation_at_tiny(self):
        cfg = TINY
        pat = SwitchPattern.primordium(cfg)
        d, dh, h, F0 = cfg.d, cfg.d_head, cfg.primordium_heads, cfg.primordium_slices * cfg.slice
        t = 5
        hand = 6 * (2 * d * cfg.d_x)
        for l in cfg.primordium_layers:
            hand += 6 * (3 * d * F0 + 4 * d)
            if cfg.op(l) == "delta":
                hand += 6 * (3 * h * dh * d + 2 * h * d + 2 * h * dh * d) + 6 * h * 2 * dh * dh
            else:
                hand += 6 * (2 * h * dh * d + 2 * cfg.h_kv * dh * d + d * h * dh) + 6 * h * 2 * dh * (t + 1)
        assert flops_per_span(cfg, pat, t) == hand
        # the effective form weights the switchable groups by their multiplier
        on = SwitchPattern(cfg, pat.layer_m, (0.5,) * cfg.L, (0.0,) * cfg.L)
        full = SwitchPattern(cfg, pat.layer_m, (1.0,) * cfg.L, (0.0,) * cfg.L)
        f_eff = flops_per_span(cfg, on, t, effective=True)
        assert flops_per_span(cfg, pat, t) < f_eff < flops_per_span(cfg, full, t) == flops_per_span(cfg, on, t)
        assert flops_per_window(cfg, SwitchPattern.adult(cfg)) > flops_per_window(cfg, pat)

    def test_against_the_flop_counter_on_a_forward(self):
        cfg = TINY
        bb = Backbone(cfg)
        x = torch.randn(4, cfg.T_s, cfg.d_x)
        pat = SwitchPattern.adult(cfg)
        with FlopCounterMode(display=False) as fc:
            bb(x, pattern=pat)
        measured = fc.get_total_flops() / 4
        formula = flops_per_window(cfg, pat, train=False)
        ratio = measured / formula
        print(f"flop counter / formula (forward): {ratio:.3f}")
        assert 0.7 < ratio < 1.3, ratio


class TestM10Controls:
    def test_control_trajectories(self):
        cfg = TINY
        for f in (0.0, 0.1, 0.16, 0.3, 0.7, 1.0):
            a = control_pattern(cfg, "schedule", f)
            b = control_pattern(cfg, "oneshot", f)
            c = control_pattern(cfg, "flat", f)
            assert c == SwitchPattern.adult(cfg)
            assert b == (SwitchPattern.adult(cfg) if f >= 0.5 else SwitchPattern.primordium(cfg))
            assert (a.head_m[0] > 0) == (f >= 0.15) and (a.slice_m[0] > 0) == (f >= 0.25)
            assert a.layer_active(cfg.layers_of_quarter(1)[0]) == (f >= 0.60)
        a = math.log(0.98 / 0.02) * 0.2                              # a_G ramps linearly to sigma^-1(0.98)
        assert abs(control_pattern(cfg, "schedule", 0.16).head_m[0] - 1 / (1 + math.exp(-a))) < 1e-9

    def test_hypercloning_split_preserves_the_function_at_the_move(self):
        cfg = TINY
        bb = Backbone(cfg)
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        y_before, _, _, _ = bb(x, pattern=SwitchPattern.primordium(cfg))
        born = born_flags_at_birth(cfg)
        out = hypercloning_split(bb, SwitchPattern.primordium(cfg), born, torch.Generator().manual_seed(0), noise=0.0)
        assert all(born.values()) and len(out) == len(born)
        y_after, _, _, _ = bb(x, pattern=SwitchPattern.adult(cfg))
        # the newborn quarter-stacks change the function (they are not halved); restrict to
        # the primordium layers: heads and slices doubled with the split preserve every layer's output
        pat = SwitchPattern(cfg, SwitchPattern.primordium(cfg).layer_m, (1.0,) * cfg.L, (1.0,) * cfg.L)
        y_split, _, _, _ = bb(x, pattern=pat)
        assert torch.allclose(y_split, y_before, atol=1e-5), (y_split - y_before).abs().max()
