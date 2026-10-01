"""Acceptance rows A1-A8 (01), L1-L10 (08), E9/E11 (04), C8/C12 (06), K6 (07) on the life,
at the CPU test shape over a synthetic token array.

SPEC: 01 §1.6, 08 §8.7, 04 §4.10, 06 §6.11, 07 §7.7
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from tempo.codec import SpanCodec
from tempo.config import TINY
from tempo.life import Lane, Life, LifeConfig, code_digest, gstat_check
from tempo.streams import build_streams

cfg = TINY
_rng = np.random.default_rng(0)
TOKENS = _rng.integers(8, cfg.vocab, size=20_000).astype(np.uint16)
EVAL_OFFSETS = _rng.integers(0, 20_000 - cfg.T - cfg.K, size=8)


def make_life(*, seed=0, only_text=True, N=4, B=4, **kw):
    torch.manual_seed(seed)                                                  # the codec's initialisation
    codec = SpanCodec(cfg.vocab, cfg.K, cfg.d_x, d_c=cfg.d_c, n_layers=1, n_heads=2, d_ff=32)
    streams = build_streams(cfg, TOKENS, eval_offsets=EVAL_OFFSETS, seed=seed, G=2, control_horizon=16)
    if only_text:
        streams = {"text": streams["text"]}
    base = dict(N=N, B=B, eval_every=0, n_eval=4, n_probe_windows=2, n_delta=1, G_proposals=2,
                n_probe_steps=2, seed=seed, sigma_e=10.0)                # sigma_e large: G-STAT passes
    base.update(kw)
    return Life(cfg, LifeConfig(**base), codec, streams)


def _flat(life):
    return {k: v.detach().clone() for k, v in life.agent.state_dict().items()} | \
        {"g": life.g.clone(), "zeta": life.zeta.clone(), "u": life.gains.u().detach().clone()}


class TestA_Dynamics:
    def test_a1_one_code_path(self):
        life = make_life()
        life.run_episode()
        lane = Lane(0, "text", {}, None, None, True)
        raw = torch.randint(8, cfg.vocab, (cfg.K,))
        y, emitted = life.step_inference(lane, raw, "text", 0.0, act="text")
        assert y.shape == (cfg.d_x,) and emitted.shape == (cfg.K,)
        # inference did not touch a parameter (no optimiser step) and did advance the lane's state
        assert lane.S is not None and not lane.boundary_pending

    def test_a3_fast_state_persists_and_boundaries_reset(self):
        life = make_life(N=3)
        life.boundary_in()
        life.train_step(0)
        S1 = {l: s.clone() for l, s in life.lanes[0].S.items()}
        life.train_step(1)
        S2 = life.lanes[0].S
        assert any(not torch.equal(S1[l], S2[l]) for l in S1)
        life.lanes[0].boundary_pending = True
        state = life._lane_states([life.lanes[0]])
        assert all(torch.count_nonzero(s) == 0 for s in state.values())
        life.lanes[0].boundary_pending = False
        state = life._lane_states([life.lanes[0]])
        assert all(torch.equal(state[l][0], S2[l][0]) for l in state)

    def test_a4_no_gradient_reaches_g_and_the_expression_is_live(self):
        life = make_life(N=2)
        life.boundary_in()
        life.train_step(0)
        assert not life.g.requires_grad and life.g.grad is None
        u0 = life.expr.express(life.g)
        with torch.no_grad():
            life.expr.B_flat.normal_(0, 0.1)
        u1 = life.expr.express(life.g)
        u2 = life.expr.express(life.g + 0.1)
        assert (u1 - u2).abs().max() > 0 and (u0 - u1).abs().max() > 0

    def test_a5_l10_gstat_doubles_N(self):
        # the statistic: a loss that keeps falling at the same rate fails; a transient that
        # ends passes; a flat end passes by the sigma_e clause
        falling = [10.0 - 0.1 * i for i in range(100)]
        assert not gstat_check(falling, 1e-9)["ok"]
        transient = [10.0 * (0.9 ** i) for i in range(100)]
        assert gstat_check(transient, 1e-9)["ok"]
        flat = [5.0 + 0.001 * ((i % 2) - 0.5) for i in range(100)]
        assert gstat_check(flat, 0.01)["ok"] and not gstat_check([5.0 - 0.1 * i for i in range(100)], 0.01)["ok"]
        # in the life: a planted falling record doubles N and writes the reason
        life = make_life(N=4, sigma_e=1e-9)
        life.boundary_in()
        for n in range(4):
            life.train_step(n)
        for i, r in enumerate(life.record["steps"][-4:]):
            r["loss"] = 10.0 - i
        out = life.boundary_out()
        assert not out["gstat_ok"] and life.N == 8 and out["N_doubled_to"] == 8
        # at most once per life: a second failure is flagged, N stays
        life.boundary_in()
        for n in range(8):
            life.train_step(n)
        for i, r in enumerate(life.record["steps"][-8:]):
            r["loss"] = 10.0 - i
        out = life.boundary_out()
        assert not out["gstat_ok"] and life.N == 8 and out.get("gstat_flagged")
        life2 = make_life(N=4, sigma_e=10.0)
        out2 = life2.run_episode()
        assert out2["gstat_ok"] and life2.N == 4

    def test_a6_determinism(self):
        a, b = make_life(seed=3), make_life(seed=3)
        for _ in range(2):
            a.run_episode()
            b.run_episode()
        fa, fb = _flat(a), _flat(b)
        assert all(torch.equal(fa[k], fb[k]) for k in fa)
        assert a.pattern == b.pattern

    def test_a8_c2_ratio_is_recorded(self):
        life = make_life()
        out = life.run_episode()
        assert "C2_ratio" in out and out["C2_ratio"] > 0


class TestL_Life:
    def test_l1_order_of_the_boundary(self):
        life = make_life(N=2)
        calls = []
        for name in ("boundary_in", "train_step", "boundary_out"):
            orig = getattr(life, name)

            def wrapped(*a, _n=name, _o=orig, **k):
                calls.append(_n)
                return _o(*a, **k)

            setattr(life, name, wrapped)
        life.run_episode()
        assert calls == ["boundary_in", "train_step", "train_step", "boundary_out"]

    def test_l2_bit_exact_resume_on_cpu(self):
        life = make_life(N=4)
        life.run_episode()
        snap = life.checkpoint_state()
        life.run_episode()
        l1 = [r["loss"] for r in life.record["steps"][-4:]]
        f1 = _flat(life)
        life.load_state(snap)
        life.run_episode()
        l2 = [r["loss"] for r in life.record["steps"][-4:]]
        f2 = _flat(life)
        assert l1 == l2
        assert all(torch.equal(f1[k], f2[k]) for k in f1)

    def test_l2b_save_and_load_through_a_file(self, tmp_path):
        life = make_life(N=2)
        life.run_episode()
        p = tmp_path / "ckpt.pt"
        life.save(p)
        other = make_life(N=2)
        other.load(p)
        fa, fb = _flat(life), _flat(other)
        assert all(torch.equal(fa[k], fb[k]) for k in fa) and other.episode == 1

    def test_l4_lanes_read_consecutively(self):
        life = make_life(N=3)
        life.boundary_in()
        offs = []
        for n in range(3):
            pairs = life._draw_windows(n)
            offs.append(pairs[0][1][0].ident[1])
        assert offs[1] == offs[0] + cfg.T and offs[2] == offs[1] + cfg.T

    def test_l5_fresh_state_evaluation(self):
        life = make_life(N=2)
        life.run_episode()
        e1 = life.evaluate("text", 4)
        life.lanes[0].S = {l: torch.randn_like(s) for l, s in life.lanes[0].S.items()}
        e2 = life.evaluate("text", 4)
        assert e1["fresh"] == e2["fresh"] and "continuing" in e1

    def test_l6_rehearsal_first_steps(self):
        life = make_life(N=40, B=4)
        life.run_episode()
        assert len(life.rehearsal) > 0
        life.boundary_in()
        pairs = life._draw_windows(0)
        assert all(ln.index >= 1000 for ln, _ in pairs)                  # rehearsal rows
        pairs = life._draw_windows(int(0.05 * life.N) + 1)
        assert all(ln.index < 1000 for ln, _ in pairs)

    def test_l7_inference_closure(self):
        life = make_life(N=2)
        life.run_episode()
        lane = Lane(0, "text", {}, None, None, True)
        raw = torch.randint(8, cfg.vocab, (cfg.K,))
        y1, emitted = life.step_inference(lane, raw, "text", 0.0, act="text")
        y2, _ = life.step_inference(lane, emitted, "text", 0.25)         # the emission re-enters
        assert y2.shape == (cfg.d_x,) and lane.cache is not None

    def test_l9_provenance_digest(self, tmp_path):
        life = make_life(N=1)
        life.run_episode()
        p = tmp_path / "c.pt"
        life.save(p)
        s = torch.load(p, weights_only=False)
        s["provenance"]["code_digest"] = "deadbeefdeadbeef"
        torch.save(s, p)
        with pytest.raises(RuntimeError):
            make_life(N=1).load(p)
        make_life(N=1).load(p, allow_digest_mismatch=True)
        assert len(code_digest()) == 16


class TestE_C_K_OncePerEpisode:
    def test_e11_c12_k6_sentinels(self):
        """G-NOFREE-INVERSE: expression, regression, consolidation and the probe run once per
        boundary and never inside the step loop."""
        life = make_life(N=3)
        life.run_episode()
        # the expression runs at birth and once per boundary (the new genome's gains, reused
        # by the next boundary-in): e + 1 calls after e episodes
        assert life._expression_calls == 2 and life._regression_calls == 1
        assert life._consolidations == 1 and life._probe_calls == 1
        life.run_episode()
        assert life._expression_calls == 3 and life._regression_calls == 2
        assert life._consolidations == 2 and life._probe_calls == 2
        # inside a step nothing of the boundary runs: the counters are unchanged across steps
        life.boundary_in()
        before = (life._expression_calls, life._regression_calls, life._consolidations, life._probe_calls)
        life.train_step(0)
        assert (life._expression_calls, life._regression_calls, life._consolidations, life._probe_calls) == before
        life._in_step = True
        with pytest.raises(RuntimeError):
            life.boundary_out()

    def test_e9_learning_rate_role_scales_the_groups(self):
        life = make_life(N=1)
        life.boundary_in()
        base = [g["base_lr"] for opt in life.opt for g in opt.param_groups]
        gamma = torch.zeros(life.reg.n_gamma)
        gamma[life.reg.slice("LR")] = torch.tensor([math.log(2.0), 0.0, 0.0, 0.0])
        from tempo.genome import lr_multipliers
        opts = life._optimizers(lr_multipliers(life.reg, gamma))
        scaled = [g["base_lr"] for opt in opts for g in opt.param_groups]
        assert abs(scaled[0] - 2 * base[0]) < 1e-6 * base[0] and scaled[1:] == base[1:]   # gamma is float32

    def test_c8_estimator2_probes_consume_identical_windows(self):
        life = make_life(N=2, estimator="restem")
        seen = []
        orig = life.streams["text"].next_windows

        def spy(cursor, actor, lane, gen):
            ws = orig(cursor, actor, lane, gen)
            seen.append(tuple(w.ident for w in ws))
            return ws

        life.streams["text"].next_windows = spy
        life.run_episode()
        n_probe = life.lc.G_proposals * life.lc.n_probe_steps * life.lc.B
        probe_idents = seen[-n_probe:]
        half = n_probe // 2
        assert probe_idents[:half] == probe_idents[half:]                 # CRN: the same windows in order

    def test_c9_null_consolidation_leaves_xi(self):
        life = make_life(N=2, genome=False, free_gains=False, homeostasis=False, estimator="none",
                         morphology="primordium", rehearsal="none")
        life.boundary_in()
        life.train_step(0)
        life.train_step(1)
        before = _flat(life)
        life.boundary_out()
        after = _flat(life)
        assert all(torch.equal(before[k], after[k]) for k in before)


class TestBlend:
    def test_all_streams_run_and_record(self):
        life = make_life(only_text=False, N=3, B=8, estimator="reinforce")
        out = life.run_episode()
        seen = life.prev_summaries.streams_present()
        assert set(seen) >= {"text", "task", "control", "vision", "speech", "pattern", "interleave"}
        assert "L_by_stream" in out and set(out["L_by_stream"]) == set(seen)
        assert all(math.isfinite(v) for v in out["L_by_stream"].values())
        assert "control" in life.R_scale and life.R_scale["control"] >= 1.0
        out2 = life.run_episode()
        assert "reward" in out2 and math.isfinite(out2["reward"])
        assert set(out2["kappa"]) >= {"act", "speech", "image"}

    def test_groups_arrive_together(self):
        """An interactive lane returns its teacher-forced supervised row (rollout 0, no policy
        term) and its `G` rollout rows (one group) in one step; control returns the `G`
        rollouts' k-th windows with the states they entered with."""
        life = make_life(only_text=False, N=1, B=8)
        life.boundary_in()
        pairs = life._draw_windows(0)
        for lane, ws in pairs:
            if lane.stream in ("task", "vision", "speech"):
                groups = {w.rl.group for w in ws if w.rl is not None}
                assert len(ws) == 3 and len(groups) == 1
                assert ws[0].rollout == 0 and ws[0].rl is None and ws[0].targets
                assert all(w.rollout > 0 and not w.targets for w in ws[1:])
            if lane.stream == "control":
                assert len(ws) == 2 and len({w.rl.group for w in ws}) == 1 and ws[0].rollout == 0


class TestArms:
    def test_e12_svf_arm_runs_and_is_the_identity_at_zero(self):
        life = make_life(N=2, svf=True)
        assert all(r.size == min(r.n_in, r.n_out) for r in life.reg.roles if r.kind == "matrix")
        life.boundary_in()
        assert len(life.agent.backbone.svf) == len(life.agent.backbone.matrix_names())
        x = torch.randn(1, cfg.T_s, cfg.d_x)
        with torch.no_grad():
            y0, _, _, _ = life.agent.backbone(x, pattern=life.pattern)
            y1, _, _, _ = life.agent.backbone(x, pattern=life.pattern, gains=life.gains_dict())
        assert torch.allclose(y0, y1, atol=1e-6)                         # B = 0 at birth: the identity
        with torch.no_grad():
            life.gains.u_diff.normal_(0, 1.0)
            y2, _, _, _ = life.agent.backbone(x, pattern=life.pattern, gains=life.gains_dict())
        assert not torch.allclose(y0, y2, atol=1e-4)
        life.train_step(0)
        life.train_step(1)
        out = life.boundary_out()
        assert "dg_norm" in out

    def test_grpo_and_es_arms_run(self):
        for est in ("grpo", "es"):
            life = make_life(N=2, estimator=est)
            out = life.run_episode()
            assert "dg_norm" in out and math.isfinite(out["dg_norm"])


class TestDecoderTrains:
    def test_the_span_decoder_trains_and_the_encoder_does_not(self):
        life = make_life(N=4)                      # the schedule is zero at p = 0 and p = 1: use its middle
        life.boundary_in()
        enc_before = [p.detach().clone() for p in life.agent.codec.encoder_parameters()]
        dec_before = [p.detach().clone() for p in life.agent.codec.decoder_parameters()]
        for n in range(3):
            life.train_step(n)
        assert all(torch.equal(a, b) for a, b in zip(enc_before, life.agent.codec.encoder_parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(dec_before, life.agent.codec.decoder_parameters()))
        assert all(not p.requires_grad for p in life.agent.codec.encoder_parameters())

class TestL11L12Batching:
    """L11 (the genome reset), L12 (compute accounting) and the equality of the lockstep
    rollout path with the one-at-a-time path (08 §8.3)."""

    def test_l11_genome_reset(self):
        life = make_life(N=2)
        life.run_episode()
        life.run_episode()
        w_before = {k: v.detach().clone() for k, v in life.agent.state_dict().items()}
        psi_before = {k: v.detach().clone() for k, v in life.cons.state_dict().items()}
        zeta_before = life.zeta.clone()
        assert not torch.equal(life.g, life.g0)                     # the genome moved
        life.reset_genome()
        assert torch.equal(life.g, life.g0)
        assert torch.allclose(life.gains.u(), life.expr.express(life.g0))
        assert not life.fit_pairs and not life.restem.accepted and len(life.rehearsal) == 0
        assert life.prev_summaries is None and life.L_check_prev is None and life.pattern_frozen
        assert all(torch.equal(w_before[k], v) for k, v in life.agent.state_dict().items())
        assert all(torch.equal(psi_before[k], v) for k, v in life.cons.state_dict().items())
        assert torch.equal(zeta_before, life.zeta)
        pattern = life.pattern
        life.run_episode()
        assert life.pattern == pattern                              # frozen: the shape cannot move

    def test_l12_compute_accounting(self):
        life = make_life(N=2, estimator="restem")
        out = life.run_episode()
        assert out["F_episode"] > 0 and out["F_cum"] >= out["F_episode"]
        assert out["F_probes"] > 0                                  # the probes are charged
        steps = [r for r in life.record["steps"] if r.get("episode") == 0 and "F" in r]
        assert abs(sum(r["F"] for r in steps[:life.lc.N]) - out["F_episode"]) < 1e-6 * out["F_episode"]
        rec = life.record["probes"][-1]
        assert rec["null_reward"] is not None and len(rec["probe_rewards"]) == life.lc.G_proposals + 1

    def test_lockstep_rollouts_equal_the_sequential_path(self):
        """The batched acting path is an optimisation, not a different computation.

        The earlier form of this test compared only the *rewards* of `evaluate`, which are
        identically `0.0` for every text-emitting stream on an untrained agent -- so it held
        no matter what the two paths did, and it passed while the batched path was destroying
        the KV cache (11 §P-C.35). It now compares what the agent **emitted**, which differs
        long before a reward does, and it checks the property that actually broke: an instance
        scored inside a heterogeneous group must emit what it emits when scored alone.
        """
        life = make_life(only_text=False, N=1, B=4, n_rollout_eval=6)
        life.boundary_in()
        life.train_step(0)

        class Serial:
            """The same actor with the batched entry points hidden, so the helpers fall back."""

            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, k):
                if k in ("perceive_many", "decode_text_many", "decode_head_many"):
                    raise AttributeError(k)
                return getattr(self._inner, k)

        serial = Serial(life)
        from tempo.streams import _eval_seed

        # (a) a task group whose prompts have different lengths: each row must emit what it
        #     emits alone. This is the case the group-by-prompt-length partition exists for.
        ts = life.streams["task"]
        seeds = [_eval_seed("task", i, fitness=True) for i in range(6)]
        insts = [ts.instance(seeds[i], ts._fixed_params(i)) for i in range(6)]
        n_p = {len(ts.tasks.prompt_tokens(i)) for i in insts}
        assert len(n_p) > 1, "the fixture must contain prompts of differing length"
        lanes = [Lane(700 + i, "task", {}, None, None, True) for i in range(6)]
        grp, _ = ts.act_group(life, lanes, insts, torch.Generator().manual_seed(seeds[0]), 0.0, greedy=True)
        for i in range(6):
            one = [Lane(800, "task", {}, None, None, True)]
            alone, _ = ts.act_group(life, one, [insts[i]], torch.Generator().manual_seed(seeds[0]), 0.0, greedy=True)
            assert grp[i] == alone[0], (i, grp[i], alone[0])

        # (b) greedy evaluation, batched against sequential, on every reward-bearing stream
        for name in ("task", "speech", "vision", "control"):
            st = life.streams[name]
            a = st.evaluate(life, Lane(0, name, {}, None, None, True), 6, fitness=True)
            b = st.evaluate(serial, Lane(0, name, {}, None, None, True), 6, fitness=True)
            assert len(a) == len(b) == 6
            assert max(abs(x - y) for x, y in zip(a, b)) < 5e-3, (name, a, b)

        # (c) the sampled training draw: the two paths consume the generator differently (one
        #     batched draw against n row draws), so their tokens differ while the structure and
        #     the determinism do not
        for name in ("task", "speech", "vision"):
            st = life.streams[name]
            c1, c2 = st.cursor(0, 7), st.cursor(0, 7)
            w1 = st.next_windows(c1, life, Lane(0, name, c1, None, None, True), torch.Generator().manual_seed(1))
            c3 = st.cursor(0, 7)
            w2 = st.next_windows(c3, life, Lane(0, name, c3, None, None, True), torch.Generator().manual_seed(1))
            assert [w.ident for w in w1] == [w.ident for w in w2]
            for x, y in zip(w1, w2):
                assert (x.rl is None) == (y.rl is None)
                if x.rl is not None:
                    assert torch.equal(x.rl.samples, y.rl.samples)      # deterministic
            ws = st.next_windows(c2, serial, Lane(0, name, c2, None, None, True), torch.Generator().manual_seed(1))
            assert [w.ident for w in ws] == [w.ident for w in w1]        # same windows either way

    def test_perceive_many_refuses_a_mixed_cache_phase(self):
        """A group must be stepped in lockstep, every row every span. The batched call refuses
        a set of lanes whose caches have different lengths rather than silently running with
        no cache and writing the length-one result back over every lane (11 §P-C.35)."""
        life = make_life(N=1, B=2)
        life.boundary_in()
        lanes = [Lane(900 + i, "text", {}, None, None, True) for i in range(2)]
        raw = torch.randint(8, cfg.vocab, (2, cfg.K))
        life.perceive_many(lanes, list(raw), "text", [0.0, 0.0])
        life.perceive_many(lanes, list(raw), "text", [0.25, 0.25])
        lens = [int(next(iter(ln.cache.values()))[0].shape[2]) for ln in lanes]
        assert lens == [2, 2]                                    # both advanced
        life.perceive(lanes[0], raw[0], "text", 0.5)             # one lane pulled out of step
        with pytest.raises(RuntimeError, match="cache phase"):
            life.perceive_many(lanes, list(raw), "text", [0.75, 0.75])

class TestPauseResume:
    """L2c (08 §8.5): a life paused *inside* an episode and resumed from its checkpoint
    reproduces the uninterrupted life step for step, and does not re-run the boundary."""

    def test_l2c_mid_episode_pause_and_resume_is_bit_exact(self):
        from tempo.life import Paused
        a = make_life(N=6, seed=5)
        a.run_episode()
        losses_ref = [r["loss"] for r in a.record["steps"] if r.get("episode") == 1] if False else None
        ref = make_life(N=6, seed=5)
        ref.run_episode()
        ref.run_episode()
        ref_losses = [r["loss"] for r in ref.record["steps"][-6:]]
        ref_flat = _flat(ref)

        # the same life, paused before step 3 of its second episode
        paused = make_life(N=6, seed=5)
        paused.run_episode()
        with pytest.raises(Paused) as caught:
            paused.run_episode(should_pause=lambda n: n == 3)
        assert caught.value.episode == 1 and caught.value.next_step == 3
        assert paused._in_episode and paused._resume_step == 3
        snap = paused.checkpoint_state()

        # resumed in a fresh process-equivalent: the boundary must not run again
        resumed = make_life(N=6, seed=5)
        resumed.load_state(snap)
        assert resumed._in_episode and resumed._resume_step == 3
        calls = []
        orig_in = resumed.boundary_in

        def spy():
            calls.append("boundary_in")
            return orig_in()

        resumed.boundary_in = spy
        out = resumed.run_episode()
        assert calls == []                                   # B1-B5 were not repeated
        assert out["episode"] == 1 and resumed.episode == 2
        assert not resumed._in_episode and resumed._resume_step is None
        got = [r["loss"] for r in resumed.record["steps"][-6:]]
        assert got == ref_losses
        got_flat = _flat(resumed)
        assert all(torch.equal(ref_flat[k], got_flat[k]) for k in ref_flat)

    def test_a_pause_that_never_fires_changes_nothing(self):
        a = make_life(N=3, seed=6)
        a.run_episode(should_pause=lambda n: False)
        b = make_life(N=3, seed=6)
        b.run_episode()
        fa, fb = _flat(a), _flat(b)
        assert all(torch.equal(fa[k], fb[k]) for k in fa)
        assert not a._in_episode and a._resume_step is None
