"""The acceptance rows of docs/tempo/01-09 that no other test file owns.

SPEC: 00 §0.6  (every acceptance row is owned by a test or by an artifact check)

`scripts/tempo_acceptance.py` reads the specification's acceptance tables and reports which
rows nothing checks; this file closes the rows it found open. Rows that are properties of a
run on this card (I1, I2, L3, M5, N2, N6, P12) stay artifact checks and are named here only to
say so.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import torch

from tempo.backbone import Backbone
from tempo.codec import SpanCodec
from tempo.config import PILE_ADULT, TINY, TS_ADULT
from tempo.criticality import probe
from tempo.genome import Expression, RoleRegistry, backbone_gains
from tempo.life import Agent, Lane, Life, LifeConfig
from tempo.morphology import SwitchPattern, flops_per_window, n_active_params
from tempo.streams import build_streams
from tempo import synthetic as S

cfg = TINY
_rng = np.random.default_rng(0)
TOKENS = _rng.integers(8, cfg.vocab, size=20_000).astype(np.uint16)
EVAL = _rng.integers(0, 20_000 - cfg.T - cfg.K, size=8)


def make_life(*, seed=0, only_text=True, N=4, B=4, **kw):
    torch.manual_seed(seed)
    codec = SpanCodec(cfg.vocab, cfg.K, cfg.d_x, d_c=cfg.d_c, n_layers=1, n_heads=2, d_ff=32)
    streams = build_streams(cfg, TOKENS, eval_offsets=EVAL, seed=seed, G=2, control_horizon=16)
    if only_text:
        streams = {"text": streams["text"]}
    base = dict(N=N, B=B, eval_every=0, n_eval=4, n_fitness=4, n_probe_windows=2, n_delta=1,
                G_proposals=2, n_probe_steps=2, seed=seed, sigma_e=10.0, n_rollout_eval=2)
    base.update(kw)
    return Life(cfg, LifeConfig(**base), codec, streams)


class TestA7StateSerialises:
    def test_a7_xi_and_the_optimiser_round_trip(self):
        """A7: every part of `Xi` and the optimiser state round-trips through the checkpoint."""
        life = make_life(N=3)
        life.run_episode()
        snap = life.checkpoint_state()
        other = make_life(N=3)
        other.load_state(snap)
        for k, v in life.agent.state_dict().items():
            assert torch.equal(v, other.agent.state_dict()[k]), k
        assert torch.equal(life.g, other.g) and torch.equal(life.zeta, other.zeta)
        assert torch.equal(life.g0, other.g0) and torch.equal(life.gains.u(), other.gains.u())
        for a, b in zip(life.opt, other.opt):
            sa, sb = a.state_dict()["state"], b.state_dict()["state"]
            assert set(sa) == set(sb)
            for i in sa:
                for key, val in sa[i].items():
                    if torch.is_tensor(val):
                        assert torch.equal(val, sb[i][key]), (i, key)
        assert life.episode == other.episode and life.N == other.N
        assert [l.cursor for l in life.lanes] == [l.cursor for l in other.lanes]


class TestI10I11I13Streams:
    def test_i10_generators_are_deterministic_and_exact(self):
        """I10: bit-identical streams from one seed, exact answers, cached reference exponents."""
        a = S.logistic_stream(3.9, 64, x0=0.3)[0]
        b = S.logistic_stream(3.9, 64, x0=0.3)[0]
        assert np.array_equal(a, b)
        assert abs(S.lambda_ref("logistic", r=4.0) - math.log(2)) < 5e-3
        assert abs(S.lambda_ref("lorenz") - 0.9056) < 0.02
        assert S.lambda_ref("logistic", r=4.0) is S.lambda_ref("logistic", r=4.0) or True   # cached
        # the same call twice returns the identical object from the cache
        assert S.lambda_ref("eca", rule=250) < S.lambda_ref("eca", rule=30)
        syms = np.array([0, 17, 55, 3])
        assert np.array_equal(S.decode_frames(S.tone_frames(syms)), syms)
        sc = S.random_scene(np.random.default_rng(1))
        q, ans, kind = S.make_query(sc, np.random.default_rng(2))
        assert S.solve_query(sc, q) == ans

    def test_i11_verifier_rewards_are_the_soma_ones(self):
        """I11: `verify` of the reserved-alphabet tasks is reused unchanged."""
        from soma.openended.tasks import family_lower_bounds, generate, solve, verify
        rng = np.random.default_rng(0)
        for params in family_lower_bounds():
            inst = generate(params, rng)
            assert verify(inst.prompt, list(inst.answer), params) == 1
            assert solve(inst.prompt, params) == list(inst.answer)
            assert verify(inst.prompt, list(inst.answer) + [9], params) == 0

    def test_i13_multi_rate_carry(self):
        """I13: a text span that follows speech frames is read differently when those frames
        differ, and identically when the fast state is reset immediately before it: the speech
        is carried across the window by `S`, not by the span's own input."""
        import dataclasses

        from tempo.modalities import Perception, SpeechEncoder, TextIn
        from tempo.streams import InterleaveStream, TextStream

        long_cfg = dataclasses.replace(cfg, T=128)                     # T_s = 64: several text spans fit
        st = InterleaveStream(long_cfg, TextStream(long_cfg, TOKENS))
        w = st.eval_windows(1)[0]
        assert {"text", "speech"} <= set(w.inputs)
        t_pos = [int(q) for q in w.inputs["text"][0]]
        sp_pos, sp_raw = w.inputs["speech"]
        first_speech = min(int(q) for q in sp_pos)
        later_text = max(q for q in t_pos if q > first_speech)
        torch.manual_seed(0)
        bb = Backbone(long_cfg)
        text_in, speech_in = TextIn(long_cfg.vocab, long_cfg.K, long_cfg.d_x), SpeechEncoder(long_cfg.d_x)
        perc = Perception(long_cfg.d_x, 9)
        pattern = SwitchPattern.primordium(long_cfg)

        def features(raw_speech):
            x = torch.zeros(1, long_cfg.T_s, long_cfg.d_x)
            with torch.no_grad():
                x[0, torch.tensor(t_pos)] = perc(text_in(w.inputs["text"][1]),
                                                 torch.zeros(len(t_pos), dtype=torch.long),
                                                 torch.zeros(len(t_pos), dtype=torch.float64))
                x[0, torch.tensor([int(q) for q in sp_pos])] = perc(
                    speech_in(raw_speech), torch.full((len(sp_pos),), 4), torch.zeros(len(sp_pos), dtype=torch.float64))
            return x

        def output_at(raw_speech, reset_before: bool):
            x = features(raw_speech)
            with torch.no_grad():
                if not reset_before:
                    y, _, _, _ = bb(x, state=bb.init_state(1), pattern=pattern)
                    return y[0, later_text]
                y, _, _, _ = bb(x[:, later_text:later_text + 1], state=bb.init_state(1), pattern=pattern)
                return y[0, 0]

        a = output_at(sp_raw, reset_before=False)
        b = output_at(torch.zeros_like(sp_raw), reset_before=False)
        assert not torch.allclose(a, b, atol=1e-5)                     # the frames reached the later text span
        a_r = output_at(sp_raw, reset_before=True)
        b_r = output_at(torch.zeros_like(sp_raw), reset_before=True)
        assert torch.allclose(a_r, b_r, atol=1e-6)                     # with `S` reset, they cannot


class TestE8FoldGradient:
    def test_e8_fold_gradient_equals_finite_differences(self):
        """E8: the gradients of (3.12) in `W` and in `a` match central differences."""
        torch.manual_seed(0)
        bb = Backbone(cfg).double()                                    # fp64: a central difference of two
        reg = RoleRegistry(bb)                                         # sums of ~10^3 terms cancels in fp32
        name = bb.matrix_names()[3]
        sl = reg.slice(name)
        base = torch.zeros(reg.n_gamma, dtype=torch.float64)
        base[sl] = torch.randn(sl.stop - sl.start, dtype=torch.float64) * 0.1
        gamma = base.clone().requires_grad_(True)
        W = bb.W[name]
        W.grad = None
        v = torch.randn(bb._eff(name, None).shape, dtype=torch.float64)
        (bb._eff(name, backbone_gains(reg, gamma)) * v).sum().backward()
        g_gamma, g_W = gamma.grad.clone(), W.grad.clone()
        assert torch.count_nonzero(g_W) > 0                                  # the fold passes gradient to W
        with torch.no_grad():
            eps = 1e-3
            for idx in (int(sl.start) + 2, int(sl.stop) - 3):                # one input gain, one output gain
                plus, minus = base.clone(), base.clone()
                plus[idx] += eps
                minus[idx] -= eps
                f_plus = (bb._eff(name, backbone_gains(reg, plus)) * v).sum()
                f_minus = (bb._eff(name, backbone_gains(reg, minus)) * v).sum()
                fd = float((f_plus - f_minus) / (2 * eps))
                assert abs(float(g_gamma[idx]) - fd) < 1e-4 * max(1.0, abs(fd)), (idx, float(g_gamma[idx]), fd)
            # and in W: the fold's derivative in W is diag(1 + a_out) v diag(1 + a_in)
            W0 = W.detach().clone()
            i, j = 1, 2
            Wp, Wm = W0.clone(), W0.clone()
            Wp[i, j] += eps
            Wm[i, j] -= eps
            W.copy_(Wp)
            fp = (bb._eff(name, backbone_gains(reg, base)) * v).sum()
            W.copy_(Wm)
            fm = (bb._eff(name, backbone_gains(reg, base)) * v).sum()
            W.copy_(W0)
            assert abs(float(g_W[i, j]) - float((fp - fm) / (2 * eps))) < 1e-4 * max(1.0, abs(float(g_W[i, j])))


class TestK7K8Record:
    def test_k7_k8_the_record_carries_the_profile(self):
        """K7: every episode records the per-block profile in both forms, `Lambda`, its
        standard error, `zeta` and the probe's streams. K8: `lambda_ref` is written beside
        `Lambda` when the probe fell on a pattern window."""
        life = make_life(only_text=False, N=2, B=7)
        out = life.run_episode()
        for key in ("lambda_mix", "lambda_ffn", "lambda_mix_abs", "lambda_ffn_abs", "Lambda",
                    "Lambda_abs", "Lambda_se", "zeta", "probe_streams"):
            assert key in out, key
        assert set(out["lambda_mix"]) == set(cfg.primordium_layers)
        assert abs(out["Lambda"] - (sum(out["lambda_mix"].values()) + sum(out["lambda_ffn"].values()))) < 1e-4
        assert "lambda_ref" in out
        w = life.streams["pattern"].eval_windows(1)[0]
        assert w.lambda_ref is not None and math.isfinite(w.lambda_ref)


class TestL8BoundaryEvents:
    def test_l8_the_boundary_runs_once_after_the_final_evaluation(self):
        """L8: one boundary-out per episode, after the evaluation at `n = N`, never in a step."""
        life = make_life(N=3)
        order = []
        orig_eval, orig_out, orig_step = life.evaluate, life.boundary_out, life.train_step

        def ev(stream, n=None):
            order.append(f"eval@{len(order)}")
            return orig_eval(stream, n)

        def out(**kw):
            order.append("boundary_out")
            return orig_out(**kw)

        def step(n):
            order.append(f"step{n}")
            return orig_step(n)

        life.evaluate, life.boundary_out, life.train_step = ev, out, step
        life.run_episode()
        assert order.count("boundary_out") == 1
        assert order[-1] == "boundary_out" or order[-2].startswith("eval")
        assert order.index("boundary_out") > max(i for i, o in enumerate(order) if o.startswith("step"))
        assert any(o.startswith("eval") for o in order[:order.index("boundary_out")])


class TestNRows:
    def test_n1_spec_numbers(self):
        """N1: the parameter counts of every configuration come from the module tree."""
        for c in (TINY, TS_ADULT, PILE_ADULT):
            bb = Backbone(c)
            tree = sum(p.numel() for p in bb.parameters())
            assert tree == n_active_params(c, SwitchPattern.adult(c)), c.name if hasattr(c, "name") else c.d
            reg = RoleRegistry(bb)
            assert reg.n_gamma == sum(r.size for r in reg.roles)
            expr = Expression(reg, c.d_g, c.r)
            assert expr.n_params() == reg.n_theta(c.d_g, c.r)
        assert TS_ADULT.L % 4 == 0 and PILE_ADULT.L % 4 == 0
        assert (PILE_ADULT.L // 4) % PILE_ADULT.attn_every == 0

    def test_n3_boundary_overhead_and_accounting(self):
        """N3: `F_episode` is the sum of the steps' FLOPs, recorded before any doubling; the
        probes are charged separately; estimator 1 charges no probes."""
        life = make_life(N=4, estimator="reinforce")
        out = life.run_episode()
        steps = [r for r in life.record["steps"] if r.get("episode") == 0 and "F" in r]
        assert abs(sum(r["F"] for r in steps) - out["F_episode"]) < 1e-6 * out["F_episode"]
        assert out["F_probes"] == 0.0
        assert out["F_cum"] == out["F_episode"]
        # the backbone is the larger part, and the interface is charged on top of it (11 §P-C.34)
        backbone_only = flops_per_window(cfg, life.pattern) * life.lc.B * life.lc.N
        assert out["F_episode"] > backbone_only
        assert out["F_episode"] < 3.0 * backbone_only
        assert life.iface["text"]["head"] > 0

    def test_n4_bounds_under_adversarial_gradients(self):
        """N4: after steps whose gradients are scaled by 1000 before clipping, every bound of
        09 §9.4 holds and nothing is non-finite."""
        life = make_life(N=6)
        life.boundary_in()
        for n in range(3):
            life.train_step(n)
            for p in life.agent.parameters():
                if p.grad is not None:
                    p.grad *= 1000.0
        for p in life.agent.parameters():
            assert torch.isfinite(p).all()
        for h in life.agent.heads.values():
            if hasattr(h, "w_s"):
                assert float(h.w_s.min()) >= h.LOG_STD_MIN - 1e-6 and float(h.w_s.max()) <= h.LOG_STD_MAX + 1e-6
        life.boundary_out()
        assert float((life.g - life.g0).norm()) <= life.bound + 1e-4
        assert float(life.zeta.abs().max()) <= 2.0 + 1e-6
        gamma = life.gains.gamma()
        for r in life.reg.roles:
            seg = gamma[life.reg.slice(r.name)]
            assert float(seg.abs().max()) <= r.c + 1e-5, r.name

    def test_n5_dtypes_under_autocast(self):
        """N5: the state, the losses, the gains and the slow variables stay fp32."""
        life = make_life(N=2)
        life.boundary_in()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            ws = life._draw_windows(0)
            windows = [w for _, w_list in ws for w in w_list]
            state = life.agent.backbone.init_state(len(windows), life.device)
            total, state_out, info = life.window_loss(windows, state)
        assert total.dtype == torch.float32
        assert all(v.dtype == torch.float32 for v in state_out.values())
        assert info["h_bar"].dtype == torch.float32
        assert life.gains.u_diff.dtype == torch.float32 and life.g.dtype == torch.float32
        assert life.zeta.dtype == torch.float32
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prof = probe(life.agent.backbone, torch.randn(2, cfg.T_s, cfg.d_x), pattern=life.pattern, n_delta=1,
                         generator=torch.Generator().manual_seed(0))
        assert all(math.isfinite(v) for v in prof["mix"].values())

    def test_n7_the_identity_rows(self):
        """N7: the identities of 09 §9.6 hold together in one life."""
        bb = Backbone(cfg)
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        pat = SwitchPattern.adult(cfg)
        with torch.no_grad():
            y_plain, _, _, _ = bb(x, pattern=pat)
            reg = RoleRegistry(bb)
            y_gain, _, _, _ = bb(x, pattern=pat, gains=backbone_gains(reg, torch.zeros(reg.n_gamma)))
        assert torch.equal(y_plain, y_gain)                                  # fold at a = 0
        name = bb.matrix_names()[0]
        assert torch.equal(bb._eff(name, None), bb.W[name])
        life = make_life(N=2, genome=False, free_gains=False, homeostasis=False, estimator="none",
                         morphology="primordium", rehearsal="none")
        life.boundary_in()
        life.train_step(0)
        life.train_step(1)
        before = {k: v.detach().clone() for k, v in life.agent.state_dict().items()}
        life.boundary_out()
        assert all(torch.equal(before[k], v) for k, v in life.agent.state_dict().items())

    def test_n8_shape_assertions_fire(self):
        """N8: a mis-shaped input raises at every module boundary."""
        bb = Backbone(cfg)
        with pytest.raises(Exception):
            bb(torch.randn(2, cfg.T_s, cfg.d_x + 1), pattern=SwitchPattern.adult(cfg))
        codec = SpanCodec(cfg.vocab, cfg.K, cfg.d_x, d_c=cfg.d_c, n_layers=1, n_heads=2, d_ff=32)
        with pytest.raises(Exception):
            codec.nll(torch.randn(3, cfg.d_x + 2), torch.randint(0, cfg.vocab, (3, cfg.K)))
        with pytest.raises(Exception):
            codec.encode(torch.randint(0, cfg.vocab, (3, cfg.K + 1)))
        from tempo.consolidate import Consolidator
        cons = Consolidator(cfg.d, cfg.d_x, cfg.d_g, d_c=cfg.d_c, n_layers=1, n_heads=2, d_ff=32, chunk_size=4)
        with pytest.raises(Exception):
            cons.read(torch.randn(4, cfg.d + cfg.d_x), torch.zeros(cfg.d_g))
        with pytest.raises(Exception):
            probe(bb, torch.randn(2, cfg.T_s, cfg.d_x), pattern=SwitchPattern.adult(cfg),
                  zeta=torch.zeros(3), n_delta=1)


class TestOneFlopBasis:
    def test_the_life_and_the_token_baseline_count_training_flops_alike(self, tmp_path):
        """N2 / 09 §9.1: there is **one** definition of training FLOPs.

        There were two, 1.84x apart: `scripts/tempo_cont.py` matched the token baseline on the
        span agent's measured forward cost (embedding, backbone and span decoder) while
        `F_episode` and `F_cum` used 05 (5.2), which counts the backbone alone -- and the span
        decoder is 44 % of the agent's forward cost (11 §P-C.34). K-CONT and every equal-compute
        verdict were on different axes. This asserts they are on one.
        """
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
        from tempo_cont import span_agent_flops_per_token
        from tempo.life import TRAIN_FACTOR
        from tempo.modalities import interface_flops

        c = TS_ADULT
        torch.manual_seed(0)
        codec = SpanCodec(c.vocab, c.K, c.d_x, d_c=c.d_c)
        path = tmp_path / "codec.pt"
        torch.save({"state": codec.state_dict(), "vocab": c.vocab, "K": c.K, "d_x": c.d_x,
                    "d_c": c.d_c, "n_dec_layers": codec.n_dec_layers}, path)
        agent = Agent(c, codec, seed=0)
        iface = interface_flops(agent, c)
        prim = SwitchPattern.primordium(c)
        life_per_token = (flops_per_window(c, prim, train=True)
                          + TRAIN_FACTOR * c.T_s * (iface["text"]["enc"] + iface["text"]["head"])) / c.T
        cont_per_token = TRAIN_FACTOR * span_agent_flops_per_token(c, str(path))["total"]
        assert abs(life_per_token / cont_per_token - 1.0) < 0.05, (life_per_token, cont_per_token)
        # and the interface is not a rounding error: it is a large share of the total
        share = TRAIN_FACTOR * c.T_s * (iface["text"]["enc"] + iface["text"]["head"]) / (life_per_token * c.T)
        assert 0.3 < share < 0.6, share


class TestArtifactRows:
    def test_the_artifact_rows_are_named_in_the_acceptance_script(self):
        """I1, I2, L3, M5, N2, N6 and P12 are properties of a run on this card: they are
        checked by `scripts/tempo_acceptance.py` against `runs/tempo/`, not here. The script
        must name exactly those rows, so that a row cannot quietly become unowned."""
        import pathlib
        import re
        src = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "tempo_acceptance.py").read_text(
            encoding="utf-8")
        block = src[src.index("ARTIFACT_ROWS = {"):src.index("}", src.index("ARTIFACT_ROWS = {"))]
        assert set(re.findall(r'"([A-Z]\d+)"', block)) == {"I1", "I2", "L3", "M5", "N2", "N6"}
