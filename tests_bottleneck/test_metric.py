"""Acceptance tests I11-I14 and M-I4 of docs/bottleneck/00-replication.md §11, on CPU at a tiny shape.

SPEC: docs/bottleneck/00-replication.md §11, §8
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from anlage.config import ROLE_EMB, StageConfig
from anlage.model import Phenotype
from bottleneck.genome import BottleneckGenome
from bottleneck.loop import Intermittent, epsilon, fisher_diag, fisher_sampling_weights
from bottleneck.meta import MetaPool, TaskDistribution

VOCAB = 64
CFG = StageConfig(stage=0, d_model=32, L=2, n_h=2, n_kv_attn=1, d_ff=64, d_head=16, chunk_size=16)


def _rank(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(VOCAB, generator=g)


def _model(seed: int = 0) -> Phenotype:
    torch.manual_seed(seed)
    return Phenotype(CFG, VOCAB, seed=seed)


def _genome(model: Phenotype, H: int = 8, H2: int = 4, seed: int = 0) -> BottleneckGenome:
    return BottleneckGenome(model, _rank(), H=H, H2=H2, H2_vector=3, seed=seed)


def _windows(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n, 17), generator=g)


# ------------------------------------------------------------------ I11: the Fisher diagonal
class TestFisher:
    def test_i11_fisher_is_the_mean_of_squared_micro_batch_gradients(self):
        m = _model()
        win = _windows(6)
        F = fisher_diag(m, win, micro=2)
        live = m.weights()
        m.eval()
        acc = {k: torch.zeros(p.shape, dtype=torch.float32) for k, p in live.items()}
        for a in range(0, 6, 2):
            m.zero_grad(set_to_none=True)
            m.loss(win[a:a + 2])[0].backward()
            for k, p in live.items():
                if p.grad is not None:
                    acc[k] += p.grad.float() ** 2
        m.zero_grad(set_to_none=True)
        assert set(F) == set(live)
        for k, p in live.items():
            assert F[k].shape == p.shape, k
            assert torch.all(F[k] >= 0), k
            assert torch.allclose(F[k], acc[k] / 3, rtol=1e-4, atol=1e-12), k
        assert any(float(v.sum()) > 0 for v in F.values())
        assert all(p.grad is None for p in live.values()), "the hook leaves gradients as the trainer keeps them"
        with pytest.raises(ValueError):
            fisher_diag(m, win, micro=0)

    def test_i11_hook_records_the_fisher_residual_with_a_tensor_or_a_callable(self):
        for probe in (_windows(4), lambda: _windows(4)):
            m = _model()
            G = _genome(m)
            hook = Intermittent(G, m, T_gen=100, lam=20.0, n_refit=2, fraction=1.0, cap=256, lr=1e-3,
                                evaluate=lambda model: 1.0, seed=0, fisher_windows=probe, fisher_alpha=0.1)
            hook.before_step(0, 100)
            rec = hook.records[-1]
            assert rec["fisher_seconds"] > 0
            assert set(rec["residual_fisher"]) == set(rec["residual"])
            assert math.isfinite(rec["residual_fisher_mean_matrix"])
            assert hook.as_dict()["fisher"] is True and hook.as_dict()["fisher_alpha"] == 0.1


# ------------------------------------------------------------- I12: the weights and sampler
class TestSampler:
    def test_i12_sampling_weights(self):
        F = {"a": torch.tensor([[0.0, 4.0], [0.0, 0.0]]), "z": torch.zeros(3)}
        w1 = fisher_sampling_weights(F, ["a", "z"], alpha=1.0)
        assert torch.equal(w1["a"], torch.ones(4)) and torch.equal(w1["z"], torch.ones(3))
        w0 = fisher_sampling_weights(F, ["a", "z"], alpha=0.0)
        assert torch.equal(w0["a"], torch.tensor([0.0, 4.0, 0.0, 0.0]))       # F / mean(F)
        assert torch.equal(w0["z"], torch.ones(3)), "a zero Fisher falls back to the uniform sampler"
        wh = fisher_sampling_weights(F, ["a"], alpha=0.5)
        assert torch.allclose(wh["a"], torch.tensor([0.5, 2.5, 0.5, 0.5]))
        with pytest.raises(ValueError):
            fisher_sampling_weights(F, ["a"], alpha=1.5)

    def test_i12_sampler_draws_in_proportion_and_reproduces_randint_without_weights(self):
        m = _model()
        G = _genome(m)
        key = (ROLE_EMB, -1)
        N = math.prod(G.shapes[key])
        gen = torch.Generator().manual_seed(0)
        one_hot = torch.zeros(N)
        one_hot[5] = 1.0
        assert torch.all(G.sample_entries(key, N, 100, gen, one_hot) == 5)
        flat = G.sample_entries(key, N, 20000, gen, torch.ones(N))
        assert int(flat.min()) >= 0 and int(flat.max()) < N
        assert abs(float(flat.float().mean()) - (N - 1) / 2) < 0.02 * N
        half = torch.zeros(N)
        half[: N // 2] = 1.0
        assert int(G.sample_entries(key, N, 5000, gen, half).max()) < N // 2
        g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        assert torch.equal(G.sample_entries(key, N, 50, g1, None), torch.randint(0, N, (50,), generator=g2))
        with pytest.raises(ValueError):
            G.sample_entries(key, N, 5, gen, torch.ones(N + 1))

    def test_i12_weighted_fit_step_runs_and_reduces_the_loss(self):
        m = _model()
        G = _genome(m, H=32, H2=16)
        W = {k: p.detach() for k, p in m.weights().items()}
        F = fisher_diag(m, _windows(4), micro=2)
        weights = fisher_sampling_weights(F, G.compressed_keys(), alpha=0.1)
        opt = torch.optim.Adam(G.parameters(), lr=3e-3)
        gen = torch.Generator().manual_seed(0)
        first = G.fit_step(W, fraction=1.0, cap=4096, generator=gen, opt=opt, weights=weights)
        for _ in range(150):
            last = G.fit_step(W, fraction=1.0, cap=4096, generator=gen, opt=opt, weights=weights)
        improved = [last[k] < first[k] for k in first]
        assert sum(improved) >= 0.9 * len(improved)


# ------------------------------------------------------------------ I13: weighted residual
class TestWeightedResidual:
    def test_i13_uniform_weights_give_the_plain_residual(self):
        m = _model()
        G = _genome(m)
        W = {k: p.detach() for k, p in m.weights().items()}
        D = G.decode_all(W)
        ones = {k: torch.ones(math.prod(G.shapes[k])) for k in G.compressed_keys()}
        r, rF = G.residuals(W, D), G.weighted_residuals(W, D, ones)
        for k in r:
            assert abs(r[k] - rF[k]) <= 1e-6 * max(1.0, r[k]), k
        key = (ROLE_EMB, -1)
        one_hot = {k: torch.zeros(math.prod(G.shapes[k])) for k in G.compressed_keys()}
        for k in one_hot:
            one_hot[k][0] = 1.0
        rr = G.weighted_residuals(W, D, one_hot)
        w0, d0 = float(W[key].reshape(-1)[0]), float(D[key].reshape(-1)[0])
        assert abs(rr[key] - (d0 - w0) ** 2 / (w0 ** 2)) < 1e-5 * max(1.0, rr[key])
        with pytest.raises(ValueError):
            G.weighted_residuals(W, D, {k: torch.ones(3) for k in G.compressed_keys()})


# --------------------------------------------------------------------------- I14: selection
class TestSelection:
    def test_i14_rejection_restores_the_genome_and_keeps_the_decode_in_force(self):
        m = _model()
        G = _genome(m)
        fit_seq = iter([2.0, 3.0, 1.5])          # gen 1 burn-in; gen 2 worse -> rejected; gen 3 better
        hook = Intermittent(G, m, T_gen=100, lam=20.0, n_refit=1, fraction=1.0, cap=256, lr=1e-2,
                            evaluate=lambda model: 1.0, evaluate_fit=lambda model: next(fit_seq),
                            select_from=2, seed=0)
        hook.before_step(0, 100)
        assert hook.records[-1]["accepted"] is None and hook.accepted["generation"] == 1
        assert hook.records[-1]["ce_birth_fit"] == 2.0 and hook.n_proposals == 0
        D1 = {k: v.clone() for k, v in hook.accepted["D"].items()}
        state1 = {k: v.clone() for k, v in G.state_dict().items()}
        opt1 = copy.deepcopy(hook.opt.state_dict())
        W_before = None

        def spy(model):                                   # the phenotype's weights at regeneration time
            nonlocal W_before
            if W_before is None:
                W_before = {k: p.detach().clone() for k, p in model.weights().items()}
            return 1.0
        hook.evaluate = spy
        hook.before_step(1, 200)
        rec = hook.records[-1]
        assert rec["accepted"] is False and hook.n_proposals == 1 and hook.n_accepted == 0
        assert rec["ce_birth_fit"] == 3.0 and rec["ce_birth_used"] == 1.0
        for k, v in G.state_dict().items():
            assert torch.equal(v, state1[k]), k
        s0, s1 = hook.opt.state_dict()["state"], opt1["state"]
        assert set(s0) == set(s1)
        for i in s0:
            for name in s0[i]:
                a, b = s0[i][name], s1[i][name]
                assert torch.equal(a, b) if torch.is_tensor(a) else a == b, (i, name)
        assert hook.accepted["generation"] == 1
        for k in D1:
            assert torch.equal(hook.accepted["D"][k], D1[k]), k
        eps = epsilon(2, 20.0)                            # regenerated from D1, not from the rejected decode
        for k, p in m.weights().items():
            expect = eps * W_before[k] + (1.0 - eps) * D1[k].to(p.dtype)
            assert torch.allclose(p.detach(), expect, atol=1e-6), k
        hook.before_step(2, 300)
        rec = hook.records[-1]
        assert rec["accepted"] is True and hook.n_accepted == 1 and hook.accepted["generation"] == 3
        assert any(not torch.equal(v, state1[k]) for k, v in G.state_dict().items())
        d = hook.as_dict()
        assert d["select"] and d["select_from"] == 2 and d["n_proposals"] == 2 and d["accepted_generation"] == 3

    def test_i14_before_select_from_every_proposal_is_taken(self):
        m = _model()
        G = _genome(m)
        fit_seq = iter([1.0, 2.0, 3.0])
        hook = Intermittent(G, m, T_gen=100, lam=20.0, n_refit=1, fraction=1.0, cap=256, lr=1e-2,
                            evaluate=lambda model: 1.0, evaluate_fit=lambda model: next(fit_seq),
                            select_from=10, seed=0)
        states = []
        for step, toks in enumerate([100, 200, 300]):
            hook.before_step(step, toks)
            states.append({k: v.clone() for k, v in G.state_dict().items()})
        assert [r["accepted"] for r in hook.records] == [None, None, None]
        assert hook.accepted["generation"] == 3 and hook.accepted["fit"] == 3.0 and hook.n_proposals == 0
        for a, b in zip(states, states[1:]):
            assert any(not torch.equal(a[k], b[k]) for k in a)

    def test_no_selection_records_nothing(self):
        m = _model()
        G = _genome(m)
        hook = Intermittent(G, m, T_gen=100, lam=20.0, n_refit=1, fraction=1.0, cap=256, lr=1e-3,
                            evaluate=lambda model: 1.0, seed=0)
        hook.before_step(0, 100)
        rec = hook.records[-1]
        assert rec["accepted"] is None and math.isnan(rec["ce_birth_fit"]) and rec["residual_fisher"] is None
        assert hook.as_dict()["select"] is False and hook.as_dict()["fisher"] is False


# ------------------------------------------------------------------- I15: a kept role
class TestKeep:
    def test_i15_kept_role_is_copied_through_counted_and_reloaded(self, tmp_path):
        import json
        from anlage.model import save_phenotype
        from bottleneck.transfer import load_source
        m = _model()
        key = (ROLE_EMB, -1)
        G = BottleneckGenome(m, _rank(), H=8, H2=4, H2_vector=3, seed=0, keep=(ROLE_EMB,))
        G0 = _genome(m)
        W = {k: p.detach() for k, p in m.weights().items()}
        D = G.decode_all(W)
        assert torch.equal(D[key], W[key].float()), "the kept table is the trained table, verbatim"
        assert key not in G.compressed_keys() and "EMB_-1" not in G.nets and key in G.keys
        assert key not in G.residuals(W, D)
        assert G.n_params() == G0.n_params() - G0.nets["EMB_-1"].n_params() + W[key].numel()
        assert G.compression_ratio(m) < G0.compression_ratio(m)
        s = G.summary()
        assert s["keep"] == ["EMB"] and s["keep_roles"] == [ROLE_EMB]
        assert s["n_params_uncompressed"] == W[key].numel() + 1          # the table and LOGIT_SCALE
        opt = torch.optim.Adam(G.parameters(), lr=1e-3)
        losses = G.fit_step(W, fraction=1.0, cap=256, generator=torch.Generator().manual_seed(0), opt=opt)
        assert key not in losses
        with pytest.raises(KeyError):
            BottleneckGenome(m, _rank(), H=8, H2=4, H2_vector=3, seed=0, keep=(10 ** 6,))
        # round trip: load_source rebuilds the genome with the same kept set (the fit step above
        # moved the g-networks, so the decode is taken again before saving)
        D = G.decode_all(W)
        (tmp_path / "loop.json").write_text(json.dumps(
            {"arm": "t", "args": {"seed": 0}, "genome": {"compression_ratio": G.compression_ratio(m)}}), encoding="utf-8")
        torch.save({"state": G.state_dict(), "summary": s, "keys": G.keys, "shapes": G.shapes}, tmp_path / "genome.pt")
        save_phenotype(m, tmp_path / "phenotype.pt")
        src = load_source(tmp_path, _rank(), device="cpu", which="final")
        assert torch.equal(src.D[key], W[key].float())
        for k in D:
            assert torch.equal(src.D[k], D[k]), k


# ------------------------------------------------------------- I16: the hidden activation
class TestActivation:
    def test_i16_leaky_keeps_a_gradient_path_through_a_dead_layer_and_reloads(self, tmp_path):
        import json
        from anlage.model import save_phenotype
        from bottleneck.genome import PairMLP, index_codes
        from bottleneck.transfer import load_source
        x = index_codes(64)                                   # 64 codes of 6 bits
        for act in ("relu", "leaky"):
            net = PairMLP(6, 8, 4, seed=0, activation=act)
            with torch.no_grad():
                net.net[2].bias.fill_(-10.0)                   # every second-layer unit inactive
            y = net(x)
            ((y - 1.0) ** 2).mean().backward()
            g = float(net.net[0].weight.grad.abs().sum())
            if act == "relu":
                assert torch.allclose(y, y[0].expand_as(y)), "a dead ReLU layer decodes a constant"
                assert g == 0.0, "and nothing below it receives a gradient"
            else:
                assert g > 0.0, "LeakyReLU keeps the path open"
                assert not torch.allclose(y, y[0].expand_as(y))
        with pytest.raises(ValueError):
            PairMLP(6, 8, 4, seed=0, activation="tanh")
        m = _model()
        G = BottleneckGenome(m, _rank(), H=8, H2=4, H2_vector=3, seed=0, activation="leaky")
        assert G.summary()["activation"] == "leaky"
        assert all(isinstance(n.net[1], torch.nn.LeakyReLU) for n in G.nets.values())
        W = {k: p.detach() for k, p in m.weights().items()}
        D = G.decode_all(W)
        (tmp_path / "loop.json").write_text(json.dumps(
            {"arm": "t", "args": {"seed": 0}, "genome": {"compression_ratio": G.compression_ratio(m)}}), encoding="utf-8")
        torch.save({"state": G.state_dict(), "summary": G.summary(), "keys": G.keys, "shapes": G.shapes}, tmp_path / "genome.pt")
        save_phenotype(m, tmp_path / "phenotype.pt")
        src = load_source(tmp_path, _rank(), device="cpu", which="final")
        for k in D:
            assert torch.equal(src.D[k], D[k]), k
        # the same state under ReLU decodes differently: the activation is part of the genome
        G_relu = BottleneckGenome(m, _rank(), H=8, H2=4, H2_vector=3, seed=0)
        G_relu.load_state_dict(G.state_dict())
        D_relu = G_relu.decode_all(W)
        assert any(not torch.equal(D_relu[k], D[k]) for k in G.compressed_keys())


# --------------------------------------------------------------------- I17: a rank role
class TestRank:
    def test_i17_rank_role_is_the_truncated_svd_counted_and_reloaded(self, tmp_path):
        import json
        from anlage.model import save_phenotype
        from bottleneck.genome import truncated_svd
        from bottleneck.transfer import load_source
        m = _model()
        key = (ROLE_EMB, -1)
        r = 4
        G = BottleneckGenome(m, _rank(), H=8, H2=4, H2_vector=3, seed=0, rank_roles={ROLE_EMB: r})
        G0 = _genome(m)
        W = {k: p.detach() for k, p in m.weights().items()}
        D = G.decode_all(W)
        U, S, Vh = torch.linalg.svd(W[key].float(), full_matrices=False)
        expect = (U[:, :r] * S[:r]) @ Vh[:r]
        assert torch.allclose(D[key], expect, atol=1e-6)
        assert torch.linalg.matrix_rank(D[key]) <= r
        # Eckart-Young: no other rank-r matrix is closer -- a random projection is worse
        g = torch.Generator().manual_seed(0)
        P = torch.randn(W[key].shape[1], r, generator=g)
        Q, _ = torch.linalg.qr(P)
        other = W[key].float() @ Q @ Q.T
        assert ((D[key] - W[key]) ** 2).sum() < ((other - W[key]) ** 2).sum()
        assert key not in G.compressed_keys() and key in G.rank_keys() and key in G.decoded_keys()
        assert "EMB_-1" not in G.nets
        assert key in G.residuals(W, D) and G.residuals(W, D)[key] < 1.0
        n_out, n_in = W[key].shape
        assert G.n_params_rank() == r * (n_out + n_in)
        assert G.n_params() == G0.n_params() - G0.nets["EMB_-1"].n_params() + r * (n_out + n_in)
        s = G.summary()
        assert s["rank"] == {"EMB": r} and s["rank_roles"] == [[ROLE_EMB, r]]
        opt = torch.optim.Adam(G.parameters(), lr=1e-3)
        losses = G.fit_step(W, fraction=1.0, cap=256, generator=torch.Generator().manual_seed(0), opt=opt)
        assert key not in losses
        with pytest.raises(ValueError):
            BottleneckGenome(m, _rank(), H=8, H2=4, H2_vector=3, seed=0, keep=(ROLE_EMB,), rank_roles={ROLE_EMB: r})
        with pytest.raises(ValueError):
            BottleneckGenome(m, _rank(), H=8, H2=4, H2_vector=3, seed=0, rank_roles={ROLE_EMB: 10 ** 6})
        with pytest.raises(ValueError):
            truncated_svd(torch.zeros(3), 1)
        # round trip: the reloaded genome decodes the table by the same rank
        D = G.decode_all(W)
        (tmp_path / "loop.json").write_text(json.dumps(
            {"arm": "t", "args": {"seed": 0}, "genome": {"compression_ratio": G.compression_ratio(m)}}), encoding="utf-8")
        torch.save({"state": G.state_dict(), "summary": s, "keys": G.keys, "shapes": G.shapes}, tmp_path / "genome.pt")
        save_phenotype(m, tmp_path / "phenotype.pt")
        src = load_source(tmp_path, _rank(), device="cpu", which="final")
        for k in D:
            assert torch.allclose(src.D[k], D[k], atol=1e-6), k


# ------------------------------------------------------------------- M-I4: the meta probe
class TestMetaProbe:
    def test_m_i4_probe_follows_the_task(self):
        pool = MetaPool(TaskDistribution(), n_tasks=3, n_windows=8, T=16, seed=0, device="cpu")
        p0 = pool.probe(4).clone()
        assert p0.shape == (4, 17)
        pool.advance()
        p1 = pool.probe(4)
        assert not torch.equal(p0, p1)
        assert torch.equal(p1, pool._windows(1)[:4])
        with pytest.raises(ValueError):
            pool.probe(0)
