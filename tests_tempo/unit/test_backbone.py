"""Acceptance rows P1-P11 of docs/tempo/03 on the backbone.

SPEC: 03 §3.10
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from anlage import ops as aops
from anlage.config import StageConfig
from anlage.model import Phenotype
from tempo.backbone import Backbone, _rope_at
from tempo.config import PILE_ADULT, TINY, TS_ADULT, TempoConfig
from tempo.genome import EpisodeGains, RoleRegistry, backbone_gains
from tempo.morphology import SwitchPattern, n_active_params


def provisioned_params(cfg: TempoConfig) -> int:
    """The formula of 03 §3.9, per block."""
    d, dh, h, hkv, Fw = cfg.d, cfg.d_head, cfg.h, cfg.h_kv, cfg.F
    delta = 5 * h * dh * d + 2 * h * d + 3 * 4 * h * dh + 3 * d * Fw + 4 * d + 1 + 2 * h
    attn = 3 * h * dh * d + 2 * hkv * dh * d + 3 * d * Fw + 4 * d + 1
    return cfg.n_delta_layers * delta + cfg.n_attn_layers * attn + 2 * d * cfg.d_x + 2 * d


class TestP1Shapes:
    @pytest.mark.parametrize("cfg", [TINY, TS_ADULT, PILE_ADULT])
    def test_counts_match_the_formula(self, cfg):
        bb = Backbone(cfg)
        assert bb.n_params() == provisioned_params(cfg)
        for l in range(cfg.L):
            n_kv = cfg.h if cfg.op(l) == "delta" else cfg.h_kv
            assert tuple(bb.W[f"L{l}_Q"].shape) == (cfg.h * cfg.d_head, cfg.d)
            assert tuple(bb.W[f"L{l}_K"].shape) == (n_kv * cfg.d_head, cfg.d)
            assert tuple(bb.W[f"L{l}_O"].shape) == (cfg.d, cfg.h * cfg.d_head)
            assert tuple(bb.W[f"L{l}_DOWN"].shape) == (cfg.d, cfg.F)
        assert tuple(bb.W["IN_W"].shape) == (cfg.d, cfg.d_x)
        assert tuple(bb.W["OUT_W"].shape) == (cfg.d_x, cfg.d)

    def test_spec_numbers_of_the_ts_adult(self):
        """G-SPECNUM: the integers of 03 §3.9 and 04 §4.2-4.3 from the module tree."""
        cfg = TS_ADULT
        bb = Backbone(cfg)
        assert bb.n_params() == 70_198_480
        assert n_active_params(cfg, SwitchPattern.primordium(cfg)) == 17_886_264
        reg = RoleRegistry(bb)
        assert reg.n_gamma == 243_133 and reg.n_roles == 215
        assert reg.n_theta(cfg.d_g, cfg.r) == 11_302_816
        assert reg.count_table() == {"matrix": (142, 209_600), "norm": (65, 33_280), "offset": (3, 208),
                                     "lr": (1, 4), "morph": (3, 34), "mixture": (1, 7)}


class TestP2P3P4Operators:
    def test_p2_chunk_equals_the_float64_reference_with_a_state(self):
        cfg = TINY
        bb = Backbone(cfg).double()
        x = torch.randn(2, cfg.T_s, cfg.d_x, dtype=torch.float64)
        S0 = {l: torch.randn(2, cfg.h, cfg.d_head, cfg.d_head, dtype=torch.float64) * 0.1
              for l in range(cfg.L) if cfg.op(l) == "delta"}
        y1, S1, _, _ = bb(x, state=S0)
        bb.use_reference = True
        y2, S2, _, _ = bb(x, state=S0)
        assert torch.allclose(y1, y2, atol=1e-10)
        for l in S1:
            assert torch.allclose(S1[l], S2[l], atol=1e-10)

    def test_p3_streaming_equivalence_with_state_and_cache(self):
        cfg = TINY
        bb = Backbone(cfg)
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        y, S, C, _ = bb(x)
        T1 = cfg.T_s // 2
        ya, Sa, Ca, _ = bb(x[:, :T1])
        yb, Sb, Cb, _ = bb(x[:, T1:], state=Sa, cache=Ca)
        assert torch.allclose(torch.cat([ya, yb], 1), y, atol=1e-5)
        for l in S:
            assert torch.allclose(Sb[l], S[l], atol=1e-5)

    def test_p4_kv_cache_span_by_span(self):
        cfg = TINY
        bb = Backbone(cfg)
        x = torch.randn(1, cfg.T_s, cfg.d_x)
        y, _, _, _ = bb(x)
        S, C, outs = None, None, []
        for t in range(cfg.T_s):
            yt, S, C, _ = bb(x[:, t:t + 1], state=S, cache=C)
            outs.append(yt)
        assert torch.allclose(torch.cat(outs, 1), y, atol=1e-5)
        for l, (k, v) in C.items():
            assert k.shape[2] == cfg.T_s

    def test_p5_gqa_and_rope(self):
        x = torch.randn(2, 3, 4, 16)
        assert torch.allclose(_rope_at(x, 0), aops.rope(x))
        r = _rope_at(x, 5)
        full = aops.rope(torch.cat([torch.zeros(2, 3, 5, 16), x], dim=2))[:, :, 5:]
        assert torch.allclose(r, full, atol=1e-6)
        cfg = TempoConfig(d=32, L=8, h=4, h_kv=2, d_head=8, F=64, slice=16, d_x=16, K=2, T=16,
                          d_g=8, r=2, d_c=16, chunk_size=4, vocab=64, attn_every=2,
                          primordium_heads=2, primordium_slices=2)
        bb = Backbone(cfg)
        l = 1
        with torch.no_grad():                               # heads 0 and 2 share q rows -> they read the same kv head 0
            bb.W[f"L{l}_Q"][2 * 8:3 * 8] = bb.W[f"L{l}_Q"][0:8]
            bb.W[f"L{l}_OG"][2 * 8:3 * 8] = bb.W[f"L{l}_OG"][0:8]
            bb.W[f"L{l}_Q"][1 * 8:2 * 8] = bb.W[f"L{l}_Q"][0:8]
            bb.W[f"L{l}_OG"][1 * 8:2 * 8] = bb.W[f"L{l}_OG"][0:8]
        hn = torch.randn(1, 6, 32)
        out_by_head = {}

        def capture(mod, inp, out):
            pass

        a, _ = bb._attn(l, hn, None, SwitchPattern.adult(cfg), None)
        # per-head outputs before W_o: recompute by inverting nothing; compare via W_o restricted to one head
        Wo = bb.W[f"L{l}_O"].detach().clone()
        with torch.no_grad():
            heads = []
            for j in range(4):
                Wo_j = torch.zeros_like(Wo)
                Wo_j[:, j * 8:(j + 1) * 8] = Wo[:, j * 8:(j + 1) * 8]
                bb.W[f"L{l}_O"].copy_(Wo_j)
                a_j, _ = bb._attn(l, hn, None, SwitchPattern.adult(cfg), None)
                heads.append(a_j)
            bb.W[f"L{l}_O"].copy_(Wo)
        # head 0 and head 2 read kv head 0 (same q, same kv) -> identical pre-W_o outputs
        pre = [torch.linalg.lstsq(Wo[:, j * 8:(j + 1) * 8], heads[j].reshape(-1, 32).T).solution for j in range(4)]
        assert torch.allclose(pre[0], pre[2], atol=1e-4)
        assert not torch.allclose(pre[0], pre[1], atol=1e-3)     # head 1 reads kv head 1


class TestExplicitAttention:
    def test_explicit_softmax_path_equals_sdpa_with_and_without_cache(self):
        cfg = TINY
        bb = Backbone(cfg)
        l = [k for k in range(cfg.L) if cfg.op(k) == "attn"][0]
        hn = torch.randn(2, cfg.T_s, cfg.d)
        pat = SwitchPattern.adult(cfg)
        a1, c1 = bb._attn(l, hn, None, pat, None)
        a2, c2 = bb._attn(l, hn, None, pat, None, explicit=True)
        assert torch.allclose(a1, a2, atol=1e-5)
        more = torch.randn(2, 3, cfg.d)
        b1, _ = bb._attn(l, more, c1, pat, None)
        b2, _ = bb._attn(l, more, c2, pat, None, explicit=True)
        assert torch.allclose(b1, b2, atol=1e-5)


class TestP6Identity:
    def test_matches_anlage_phenotype_block_for_block(self):
        d, L, h, hkv, dh, Fw, V = 32, 8, 2, 1, 16, 64, 8
        acfg = StageConfig(stage=0, d_model=d, L=L, n_h=h, n_kv_attn=hkv, d_ff=Fw, d_head=dh,
                           chunk_size=4, use_ve=False, ops_override=("delta", "attn") * 4)
        pheno = Phenotype(acfg, V, tied=True, seed=3)
        tcfg = TempoConfig(d=d, L=L, h=h, h_kv=hkv, d_head=dh, F=Fw, slice=16, d_x=V, K=2, T=16,
                           d_g=8, r=2, d_c=16, chunk_size=4, vocab=V, attn_every=2,
                           primordium_heads=1, primordium_slices=2)
        bb = Backbone(tcfg, seed=3)
        with torch.no_grad():
            for l in range(L):
                for tr, ar in (("Q", "Q"), ("K", "K"), ("V", "V"), ("O", "O"), ("GATE", "GATE"), ("UP", "UP"),
                               ("DOWN", "DOWN"), ("NMIX", "NORM_MIX"), ("NFFN", "NORM_FFN")):
                    bb.W[f"L{l}_{tr}"].copy_(pheno.get(getattr(__import__("anlage.config", fromlist=["x"]), f"ROLE_{ar}"), l))
                if acfg.op(l) == "delta":
                    from anlage.config import ROLE_AB, ROLE_CONV_K, ROLE_CONV_Q, ROLE_CONV_V, ROLE_OGATE
                    bb.W[f"L{l}_OG"].copy_(pheno.get(ROLE_OGATE, l))
                    bb.W[f"L{l}_AB"].copy_(pheno.get(ROLE_AB, l))
                    bb.W[f"L{l}_CQ"].copy_(pheno.get(ROLE_CONV_Q, l))
                    bb.W[f"L{l}_CK"].copy_(pheno.get(ROLE_CONV_K, l))
                    bb.W[f"L{l}_CV"].copy_(pheno.get(ROLE_CONV_V, l))
                else:
                    from anlage.config import ROLE_AGATE
                    bb.W[f"L{l}_OG"].copy_(pheno.get(ROLE_AGATE, l))
            from anlage.config import ROLE_EMB, ROLE_NORM_FINAL
            E = pheno.get(ROLE_EMB, -1)
            bb.W["IN_W"].copy_((E * acfg.emb_scale).T)
            bb.W["IN_b"].zero_()
            bb.W["OUT_rho"].copy_(pheno.get(ROLE_NORM_FINAL, -1))
        # random weights so that every path is exercised (the offsets stay at their weight values)
        with torch.no_grad():
            for l in range(L):
                for r in ("NMIX", "NFFN"):
                    bb.W[f"L{l}_{r}"].mul_(1.0)
        tokens = torch.randint(0, V, (2, 8))
        ref = pheno.forward(tokens, return_hidden=True)
        x = F.one_hot(tokens, V).float()
        _, _, _, hidden = bb(x)
        assert torch.allclose(hidden, ref, atol=1e-6), (hidden - ref).abs().max()


class TestEveryGainIsLive:
    def test_perturbing_any_gain_role_changes_the_output(self):
        """04 E1 / 03 §3.8: a role whose gain never enters the computation is a dead role
        (`W_in`'s was, 11 §P-C.26)."""
        cfg = TINY
        bb = Backbone(cfg)
        reg = RoleRegistry(bb)
        eg = EpisodeGains(reg)
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        with torch.no_grad():
            y0, _, _, _ = bb(x, pattern=SwitchPattern.adult(cfg), gains=backbone_gains(reg, eg.gamma()))
        base = backbone_gains(reg, eg.gamma())
        gen = torch.Generator().manual_seed(0)
        for key in base:
            g2 = {k: v.clone() for k, v in base.items()}
            # a non-uniform perturbation: the uniform component of an output-side gain is a
            # global scale that the post-norm of Peri-LN removes (04 §4.4, a null direction)
            g2[key] = g2[key] + 0.3 * torch.randn(g2[key].shape, generator=gen)
            with torch.no_grad():
                y1, _, _, _ = bb(x, pattern=SwitchPattern.adult(cfg), gains=g2)
            assert not torch.allclose(y0, y1, atol=1e-6), key


class TestP7Fold:
    def test_identity_at_zero_gains_and_finite_difference_gradient(self):
        cfg = TINY
        bb = Backbone(cfg).double()
        reg = RoleRegistry(bb)
        eg = EpisodeGains(reg).double()
        gains = backbone_gains(reg, eg.gamma())
        name = "L0_Q"
        assert torch.equal(bb._eff(name, gains), bb.W[name])
        # gradient of a loss through the fold wrt W and wrt the gains, by finite differences
        x = torch.randn(1, 4, cfg.d_x, dtype=torch.float64)
        with torch.no_grad():
            eg.u_diff.normal_(0, 0.5)

        def loss_fn():
            g = backbone_gains(reg, eg.gamma())
            y, _, _, _ = bb(x, gains=g)
            return y.square().sum()

        loss = loss_fn()
        loss.backward()
        W = bb.W[name]
        i, j = 1, 2
        eps = 1e-5
        with torch.no_grad():
            W[i, j] += eps
            lp = loss_fn().item()
            W[i, j] -= 2 * eps
            lm = loss_fn().item()
            W[i, j] += eps
        assert abs((lp - lm) / (2 * eps) - W.grad[i, j].item()) < 1e-4 * max(1.0, abs(W.grad[i, j].item()))
        k = int(reg.slice(name).start) + 3                     # a coordinate of L0_Q's a_in
        idx = int(reg.diff_of_row[:k].sum())
        with torch.no_grad():
            eg.u_diff[idx] += eps
            lp = loss_fn().item()
            eg.u_diff[idx] -= 2 * eps
            lm = loss_fn().item()
            eg.u_diff[idx] += eps
        assert abs((lp - lm) / (2 * eps) - eg.u_diff.grad[idx].item()) < 1e-4 * max(1.0, abs(eg.u_diff.grad[idx].item()))


class TestP8P9P10Switches:
    def test_p8_skipped_equals_the_zero_limit(self):
        cfg = TINY
        bb = Backbone(cfg)
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        prim = SwitchPattern.primordium(cfg)
        tiny = SwitchPattern(cfg, tuple(1.0 if l in cfg.primordium_layers else 1e-30 for l in range(cfg.L)),
                             (1e-30,) * cfg.L, (1e-30,) * cfg.L)
        y0, S0, _, _ = bb(x, pattern=prim)
        y1, S1, _, _ = bb(x, pattern=tiny)
        assert torch.allclose(y0, y1, atol=1e-6)
        # the states of the primordium heads of the primordium layers agree; a layer executed
        # at m -> 0 still updates its own state (the multiplier is on its output), which a
        # skipped layer does not, and that state is not read (05 §5.4)
        for l in S0:
            if l in cfg.primordium_layers:
                assert torch.allclose(S0[l][:, :cfg.primordium_heads], S1[l][:, :cfg.primordium_heads], atol=1e-6)

    def test_p9_views_leave_parameters_untouched_and_route_gradients(self):
        cfg = TINY
        bb = Backbone(cfg)
        ids = {n: id(p) for n, p in bb.W.items()}
        vals = {n: p.detach().clone() for n, p in bb.W.items()}
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        prim = SwitchPattern.primordium(cfg)
        y, _, _, _ = bb(x, pattern=prim)
        y.sum().backward()
        assert all(id(p) == ids[n] for n, p in bb.W.items())
        assert all(torch.equal(p.detach(), vals[n]) for n, p in bb.W.items())
        l = cfg.primordium_layers[0]
        n0 = cfg.primordium_heads * cfg.d_head
        assert bb.W[f"L{l}_Q"].grad[:n0].abs().sum() > 0
        assert bb.W[f"L{l}_Q"].grad[n0:].abs().sum() == 0
        assert bb.W[f"L{l}_O"].grad[:, n0:].abs().sum() == 0
        F0 = cfg.primordium_slices * cfg.slice
        assert bb.W[f"L{l}_GATE"].grad[F0:].abs().sum() == 0
        for l2 in range(cfg.L):
            if l2 not in cfg.primordium_layers:
                assert bb.W[f"L{l2}_Q"].grad is None

    def test_p10_causality(self):
        cfg = TINY
        bb = Backbone(cfg)
        x1 = torch.randn(1, cfg.T_s, cfg.d_x)
        x2 = x1.clone()
        t = 3
        x2[:, t + 1:] += 1.0
        y1, _, _, _ = bb(x1)
        y2, _, _, _ = bb(x2)
        assert torch.allclose(y1[:, :t + 1], y2[:, :t + 1], atol=1e-6)
        assert not torch.allclose(y1[:, t + 1], y2[:, t + 1])


class TestP11Numerics:
    def test_bf16_autocast_training_has_no_nan(self):
        cfg = TINY
        bb = Backbone(cfg)
        opt = torch.optim.AdamW(bb.parameters(), lr=1e-3)
        for step in range(30):
            x = torch.randn(2, cfg.T_s, cfg.d_x)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y, S, _, _ = bb(x)
                loss = y.float().square().mean() * (1000.0 if step % 7 == 0 else 1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bb.parameters(), 1.0)
            opt.step()
            assert torch.isfinite(loss)
            for s in S.values():
                assert s.dtype == torch.float32 and torch.isfinite(s).all()
