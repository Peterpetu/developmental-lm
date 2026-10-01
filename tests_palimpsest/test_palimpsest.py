"""Acceptance tests T1-T11 of docs/palimpsest/01-model.md §1.11 (CPU)."""

from __future__ import annotations

import copy
import hashlib
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anlage.config import ROLE_EMB, ROLE_LOGIT_SCALE, ROLE_NORM_MIX, StageConfig  # noqa: E402
from anlage.data import EvalSet  # noqa: E402
from anlage.model import Phenotype  # noqa: E402
from anlage.recipe import TrainConfig  # noqa: E402
from bottleneck.transfer import TaskShard  # noqa: E402
from palimpsest import analyse  # noqa: E402
from palimpsest.distil import EOS_ID, distil, distil_loss, dream  # noqa: E402
from palimpsest.envs import LifeSpec, MixPool, phases_for  # noqa: E402
from palimpsest.expand import CORES, FULL, core_config, core_of, expand, ratio, shrink  # noqa: E402
from palimpsest.life import EWC, LifeConfig, Replay, birth, replay_tokens_for, run_life, scores_of  # noqa: E402
from screen.crn import WindowPool  # noqa: E402

torch.manual_seed(0)
V = 64
TINY = StageConfig(0, d_model=64, L=8, n_h=4, n_kv_attn=2, d_ff=128, d_head=16, chunk_size=16, emb_scale=8.0)


class FakeEnv:
    """The three attributes `run_life` reads from an environment."""

    def __init__(self, name: str, seed: int, T: int = 16, n_eval: int = 8):
        self.name, self.vocab, self.T, self.device = name, V, T, "cpu"
        rng = np.random.default_rng(seed)
        self.tokens = rng.integers(0, V, size=20_000, dtype=np.int64)
        ev = rng.integers(0, V, size=(n_eval, T + 1), dtype=np.int64)
        self.test = EvalSet(torch.from_numpy(ev), torch.zeros(n_eval, T, dtype=torch.long), "test",
                            hashlib.blake2b(ev.tobytes(), digest_size=8).hexdigest())
        self.fitness = self.test

    def pool(self, pool_seed: int, *, n_windows: int = 256):
        return WindowPool(TaskShard(self.tokens), n_windows=n_windows, T=self.T, pool_seed=pool_seed, device="cpu")

    def fitness_windows(self, n: int = 4):
        return self.test.windows[:n]


def rand_tokens(n=3, T=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (n, T), generator=g)


# ---------------------------------------------------------------- T1 counts
def test_T1_counts_and_ratios():
    expected = {"c2": 5_388_417, "c4": 3_218_561, "c8": 2_136_385, "c16": 1_330_369}
    assert FULL.n_params(8192) == 13_011_201
    for name in CORES:
        c = core_of(name)
        assert c.n_params(8192) == expected[name], (name, c.n_params(8192))
        assert abs(ratio(FULL, c, 8192) - 13_011_201 / expected[name]) < 1e-9
        assert abs(c.emb_scale * CORES[name][0] - FULL.emb_scale) < 1e-9


# ---------------------------------------------------------------- T2 Theorem 1
@pytest.mark.parametrize("name", list(CORES))
@pytest.mark.parametrize("new", ["copy", "fresh"])
def test_T2_expansion_preserves_the_function(name, new):
    k_d, k_L = CORES[name]
    cfg_c = core_config(TINY, k_d, k_L)
    core = Phenotype(cfg_c, V, seed=3)
    with torch.no_grad():
        core.get(ROLE_LOGIT_SCALE, -1).fill_(math.log(4.0))       # a peaked head, so the test bites
    W = expand(core, TINY, noise=0.0, eps_gain=0.0, new=new)
    assert W.cfg.d_model == TINY.d_model and W.cfg.L == TINY.L and W.cfg.n_h == TINY.n_h
    assert W.cfg.ops_override is None                               # the canonical layout is recovered
    x = rand_tokens()
    with torch.no_grad():
        lc, lw = core(x), W(x)
    assert torch.allclose(lc, lw, atol=5e-4, rtol=1e-4), float((lc - lw).abs().max())   # float32 reduction order over L=8 at a x4 logit scale
    # at the defaults the deviation is bounded
    W2 = expand(core, TINY, new=new, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ce_c = torch.nn.functional.cross_entropy(lc[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        ce_w = torch.nn.functional.cross_entropy(W2(x)[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
    assert abs(float(ce_c - ce_w)) < 0.5      # a random network with a x4 logit scale; the trained deviation is measured on the card (ledger P-P.1)


def test_T2b_shrink_then_expand_round_trips_the_kept_block():
    W = Phenotype(TINY, V, seed=5)
    C = shrink(W, core_of("c2", TINY))
    W2 = expand(C, TINY, noise=0.0, eps_gain=0.0)
    with torch.no_grad():
        assert torch.allclose(C(rand_tokens()), W2(rand_tokens()), atol=1e-4)


# ---------------------------------------------------------------- T3 KL
def test_T3_kl_matches_the_closed_form():
    s, t = Phenotype(TINY, V, seed=1), Phenotype(TINY, V, seed=2)
    with torch.no_grad():
        t.get(ROLE_LOGIT_SCALE, -1).fill_(math.log(3.0))
    w = rand_tokens(2, 6, seed=4)
    d = distil_loss(s, t, w, chunk=3)
    with torch.no_grad():
        ls_, lt = s(w[:, :-1]).float(), t(w[:, :-1]).float()
        pt = torch.softmax(lt, -1)
        ref = (pt * (torch.log_softmax(lt, -1) - torch.log_softmax(ls_, -1))).sum(-1).mean()
    assert abs(float(d) - float(ref)) < 1e-5
    assert abs(float(distil_loss(s, t, w, chunk=100)) - float(ref)) < 1e-5


# ---------------------------------------------------------------- T4 dreams
def test_T4_dreams():
    C = Phenotype(core_of("c2", TINY), V, seed=1)
    a = dream(C, 5, 7, seed=11, device="cpu", batch=2, start="eos")
    b = dream(C, 5, 7, seed=11, device="cpu", batch=3, start="eos")
    c = dream(C, 5, 7, seed=12, device="cpu", batch=2, start="eos")
    assert a.shape == (5, 8) and (a[:, 0] == EOS_ID).all() and a.min() >= 0 and a.max() < V
    assert torch.equal(a, b) and not torch.equal(a, c)
    m = dream(C, 6, 7, seed=11, device="cpu", batch=4, start="mixed")
    assert (m[0::2, 0] == EOS_ID).all() and m.shape == (6, 8) and m.min() >= 0 and m.max() < V
    u = dream(C, 40, 3, seed=11, device="cpu", batch=8, start="uniform")
    assert len(set(u[:, 0].tolist())) > 5


def test_T4b_cached_sampler_equals_the_full_forward():
    from palimpsest.sample import Sampler
    for name in ("c2", "c8"):
        C = Phenotype(core_of(name, TINY), V, seed=4)
        with torch.no_grad():
            C.get(ROLE_LOGIT_SCALE, -1).fill_(math.log(3.0))
        x = rand_tokens(3, 20, seed=9)
        with torch.no_grad():
            full = C(x)
            s = Sampler(C, 3, "cpu")
            steps = torch.stack([s.step(x[:, t]) for t in range(20)], 1)
        assert torch.allclose(full, steps, atol=2e-4, rtol=1e-4), float((full - steps).abs().max())
    # and the two dream paths agree on the same seed
    C = Phenotype(core_of("c2", TINY), V, seed=1)
    a = dream(C, 6, 9, seed=5, device="cpu", batch=4, cached=True)
    b = dream(C, 6, 9, seed=5, device="cpu", batch=6, cached=False)
    assert torch.equal(a, b)


# ---------------------------------------------------------------- T5 distillation
def test_T5_distillation_recovers_a_planted_teacher():
    cfg_c = core_of("c2", TINY)
    teacher = Phenotype(cfg_c, V, seed=1)
    with torch.no_grad():
        teacher.get(ROLE_LOGIT_SCALE, -1).fill_(math.log(6.0))
    student = Phenotype(cfg_c, V, seed=2)
    env = FakeEnv("a", 0)
    pool = env.pool(1)
    w0 = next(pool.batches(batch=8, steps=1, seed=0))
    d0 = float(distil_loss(student, teacher, w0))
    tc = TrainConfig(tokens=8 * 16 * 120, batch_final=8, T=16, micro_batch=8, lr_muon=0.02)
    res = distil(student, teacher, pool.batches(batch=8, steps=1 << 20, seed=1), tc=tc, device="cpu")
    d1 = float(distil_loss(student, teacher, w0))
    assert res["steps"] == 120 and d1 < 0.5 * d0, (d0, d1)


# ---------------------------------------------------------------- T6 births
def test_T6_births():
    cfg = LifeConfig(arm="copy", core="c2")
    W = Phenotype(TINY, V, seed=1)
    same, info = birth(cfg, W, None, seed=0, k=2, vocab=V, full=TINY, device="cpu")
    assert same is W and info["birth"] == "copy"
    cfg_s = LifeConfig(arm="scratch", core="c2")
    S, _ = birth(cfg_s, W, None, seed=0, k=2, vocab=V, full=TINY, device="cpu")
    assert not torch.equal(S.get(ROLE_EMB, -1), W.get(ROLE_EMB, -1))
    cfg_sp = LifeConfig(arm="shrink_perturb", core="c2")
    W2 = Phenotype(TINY, V, seed=1)
    old = W2.get(ROLE_EMB, -1).detach().clone()
    fresh = Phenotype(TINY, V, seed=0 * 1009 + 2)
    SP, _ = birth(cfg_sp, W2, None, seed=0, k=2, vocab=V, full=TINY, device="cpu")
    assert torch.allclose(SP.get(ROLE_EMB, -1), 0.6 * old + 0.1 * fresh.get(ROLE_EMB, -1), atol=1e-6)
    cfg_v = LifeConfig(arm="sieve", core="c2", noise=0.0, eps_gain=0.0)
    C = Phenotype(core_of("c2", TINY), V, seed=3)
    B, info = birth(cfg_v, W, C, seed=0, k=2, vocab=V, full=TINY, device="cpu")
    with torch.no_grad():
        assert torch.allclose(B(rand_tokens()), C(rand_tokens()), atol=1e-4)
    with pytest.raises(RuntimeError):
        birth(cfg_v, W, None, seed=0, k=2, vocab=V, full=TINY, device="cpu")


# ---------------------------------------------------------------- T8 replay
def test_T8_replay_byte_rule_and_fraction():
    for name in CORES:
        c = core_of(name)
        assert replay_tokens_for(c, 8192) == 2 * c.n_params(8192)
    R = Replay(17 * 10, T=9, frac=0.3, seed=0, device="cpu")
    R.add(torch.full((30, 10), 7))
    assert R.buffer.shape[0] == 17
    w = torch.zeros(8, 10, dtype=torch.long)
    out = R(w, 0)
    assert int((out[:, 0] == 7).sum()) == math.ceil(0.3 * 8) and (out[3:] == 0).all()
    # shorter buffer windows are concatenated to the batch's length
    R2 = Replay(100, T=3, frac=0.5, seed=0, device="cpu")
    R2.add(torch.arange(4).repeat(5, 1))
    out2 = R2(torch.full((4, 10), -1), 0)
    assert out2.shape == (4, 10) and (out2[:2] >= 0).all() and (out2[2:] == -1).all()


# ---------------------------------------------------------------- T9 EWC
def test_T9_ewc_penalty_closed_form():
    W = Phenotype(TINY, V, seed=1)
    env = FakeEnv("a", 0)
    e = EWC(lam=2.0)
    e.add(W, env.fitness_windows(4))
    F, Wst = e.stores[0]
    assert all((f >= 0).all() for f in F.values()) and all(p.grad is None for p in W.parameters())
    with torch.no_grad():
        W.get(ROLE_NORM_MIX, 0).add_(0.5)
    ref = 0.5 * 2.0 * float((F[(ROLE_NORM_MIX, 0)] * 0.25).sum())
    assert abs(float(e.penalty(W)) - ref) < 1e-5


# ---------------------------------------------------------------- T7 + T10 the life
def _life(arm, seed=0, core="c2", **kw):
    envs = {"a": FakeEnv("a", 0), "b": FakeEnv("b", 1)}
    spec = LifeSpec([("a", 1, 2), ("b", 3, 4), ("a", 5, 6)])
    cfg = LifeConfig(arm=arm, core=core, tokens_per_phase=8 * 16 * 4, batch=8, T=16, distil_tokens=8 * 16 * 3,
                     dream_tokens=64, T_dream=8, n_fisher=4, dream_batch=4, **kw)
    return run_life(spec, cfg, seed=seed, envs=envs, log=lambda s: None, device="cpu", full=TINY)


@pytest.mark.parametrize("arm", ["copy", "scratch", "shrink_perturb", "sieve", "sieve-nodream", "sieve-fresh",
                                 "sieve-life", "copy-dream", "trunc", "replay", "ewc", "lwf", "iid"])
def test_T10_life_runs_every_arm(arm):
    rec = _life(arm)
    assert rec["complete"]
    sc = rec["scores"]
    assert set(sc["final"]) == {"a", "b"} and math.isfinite(sc["U_K"])
    if arm != "iid":
        assert len(rec["phases"]) == 3 and "3:b" in sc["retention"] and "0.10" in sc["adaptation"]["2"]
    if arm.startswith("sieve"):
        assert len(rec["consolidations"]) == 3 and "G_K" in sc
        assert (rec["consolidations"][1]["n_dream"] > 0) == (arm != "sieve-nodream")
        # T7: the consolidation never reads a training pool
        train_digests = {p["pool_digest"] for p in rec["phases"]}
        assert all(c["pool_digest"] not in train_digests for c in rec["consolidations"])
    if arm == "replay":
        assert rec["replay_tokens"][-1] > 0
    if arm in ("sieve-life", "copy-dream"):
        assert rec["flops"]["overhead"] > 0 and len(rec["phases"]) == 3
    if arm == "sieve-life":
        assert len(rec["consolidations"]) == 3
    if arm == "ewc":
        assert rec["flops"]["overhead"] > 0


def test_T10b_life_is_reproducible():
    a, b = _life("sieve"), _life("sieve")
    assert a["scores"]["U_K"] == b["scores"]["U_K"] and a["scores"]["G_K"] == b["scores"]["G_K"]


def test_T10c_scores_of_hand_record():
    rec = {"phases": [{"env": "a", "evals": {"0.10": {"a": 3.0, "b": 5.0}, "1.00": {"a": 2.0, "b": 5.0}}},
                      {"env": "b", "evals": {"0.10": {"a": 2.5, "b": 4.0}, "1.00": {"a": 3.0, "b": 2.0}}}],
           "consolidations": [{"core_evals": {"a": 2.5, "b": 2.5}}]}
    sc = scores_of(rec)
    assert sc["U_K"] == 2.5 and sc["Lambda"] == (2.0 + 2.5) / 2 and sc["retention"] == {"2:a": 1.0}
    assert sc["adaptation"]["2"]["0.10"] == 4.0 and sc["G_K"] == 2.5


def test_phases_for_crn_and_mixpool():
    s = phases_for("shift", 0)
    assert [e for e, _, _ in s.phases] == ["stories", "code", "stories", "code"] and s.envs == ["stories", "code"]
    t = phases_for("stat", 0)
    assert len({p for _, p, _ in t.phases}) == 4                      # a different pool every phase
    env = FakeEnv("a", 0)
    mix = MixPool([env.pool(1), env.pool(2)])
    b = next(mix.batches(batch=8, steps=1, seed=0))
    assert b.shape == (8, 17)


# ---------------------------------------------------------------- T11 analysis
def _rec(arm, u, seed=0, core="c8", ratio_=10.76):
    return {"life": "shift", "seed": seed, "cfg": {"arm": arm, "core": core, "lam_ewc": 1e3}, "ratio": ratio_,
            "scores": {"U_K": u, "Lambda": u, "final": {"stories": u, "code": u}, "retention": {}, "adaptation": {}},
            "flops": {"overhead_frac": 0.0}, "complete": True}


def test_T11_verdicts_on_planted_records():
    interior = {f"r{i}": r for i, r in enumerate([_rec("copy", 3.0), _rec("copy", 3.02, seed=1), _rec("sieve", 2.8), _rec("sieve", 2.81, seed=1),
                                                    _rec("scratch", 3.5), _rec("scratch", 3.5, seed=1)])}
    arms = analyse.by_life(interior, "shift")
    v = analyse.verdict_sieve(arms)
    assert v["verdict"] == "LOAD-BEARING" and v["argmin"] == "sieve-c8"
    monotone = {f"r{i}": r for i, r in enumerate([_rec("copy", 3.0), _rec("copy", 3.01, seed=1), _rec("sieve", 3.2), _rec("sieve", 3.2, seed=1),
                                                    _rec("scratch", 3.5), _rec("scratch", 3.5, seed=1)])}
    v2 = analyse.verdict_sieve(analyse.by_life(monotone, "shift"))
    assert v2["verdict"] == "FIRES"
    assert analyse.verdict_sieve(analyse.by_life({"x": _rec("copy", 3.0)}, "shift"))["verdict"] == "NOT EVALUATED"
    for r in monotone.values():
        r["life"] = "stat"
    assert analyse.verdict_stat(analyse.by_life(monotone, "stat"))["verdict"] == "MONOTONE"
    text = analyse.report(interior)
    assert "K-SIEVE  LOAD-BEARING" in text
