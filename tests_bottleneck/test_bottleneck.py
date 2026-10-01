"""Acceptance tests I1-I10 of docs/bottleneck/00-replication.md §8, on CPU at a tiny shape.

SPEC: docs/bottleneck/00-replication.md §8
"""

from __future__ import annotations

import math

import pytest
import torch

from anlage.config import (AX_TOK, MATRIX_ROLES, ROLE_AXES, ROLE_EMB, ROLE_LOGIT_SCALE,
                           ROLE_NORM_MIX, StageConfig)
from anlage.model import Phenotype
from bottleneck.genome import (BottleneckGenome, PairMLP, gray_rank_codes, index_codes,
                               n_bits)
from bottleneck.loop import Intermittent, epsilon, regenerate, shuffle_rows

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


# ----------------------------------------------------------------------------- I1, I2: codes
class TestCodes:
    def test_i1_index_codes_are_distinct_minimal_pm1(self):
        for n in (1, 2, 3, 4, 7, 8, 9, 256, 1024):
            c = index_codes(n)
            assert c.shape == (n, n_bits(n))
            assert n_bits(n) == max(1, math.ceil(math.log2(n))) if n > 1 else n_bits(n) == 1
            assert set(c.unique().tolist()) <= {-1.0, 1.0}
            assert len({tuple(r.tolist()) for r in c}) == n

    def test_i2_gray_over_rank_adjacent_ranks_differ_in_one_bit(self):
        rank = _rank()
        c = gray_rank_codes(rank)
        assert c.shape == (VOCAB, n_bits(VOCAB))
        by_rank = c[torch.argsort(rank)]                      # row r = the code of rank r
        assert torch.all(by_rank[0] == -1.0), "rank 0 is the all -1 code"
        flips = ((by_rank[1:] != by_rank[:-1]).sum(dim=-1))
        assert torch.all(flips == 1), flips
        assert len({tuple(r.tolist()) for r in c}) == VOCAB


# ------------------------------------------------------------------------ I3, I4: the decode
class TestDecode:
    def test_i3_decode_has_the_tensor_shape_and_is_finite(self):
        m = _model()
        G = _genome(m)
        W = m.weights()
        D = G.decode_all({k: p.detach() for k, p in W.items()})
        for k, p in W.items():
            assert D[k].shape == p.shape, k
            assert torch.isfinite(D[k]).all(), k
        assert (ROLE_NORM_MIX, 0) in D and D[(ROLE_NORM_MIX, 0)].dim() == 1

    def test_i3_pair_mlp_rejects_the_wrong_width(self):
        net = PairMLP(6, 4, 3, seed=0)
        with pytest.raises(Exception):
            net(torch.zeros(5, 7))

    def test_i4_logit_scale_is_copied_through(self):
        m = _model()
        G = _genome(m)
        W = {k: p.detach() for k, p in m.weights().items()}
        W[(ROLE_LOGIT_SCALE, -1)] = torch.tensor([3.25])
        D = G.decode_all(W)
        assert torch.equal(D[(ROLE_LOGIT_SCALE, -1)], torch.tensor([3.25]))
        assert (ROLE_LOGIT_SCALE, -1) not in G.compressed_keys()

    def test_token_axis_reads_the_rank_code(self):
        m = _model()
        G = _genome(m)
        rows, cols = G.codes_for((ROLE_EMB, -1))
        assert ROLE_AXES[ROLE_EMB][0] == AX_TOK
        assert torch.equal(rows, gray_rank_codes(_rank()))
        assert cols.shape == (CFG.d_model, n_bits(CFG.d_model))


# ------------------------------------------------------------------------------ I5: capacity
class TestRegression:
    def test_i5_a_wide_gnetwork_fits_a_small_matrix(self):
        torch.manual_seed(0)
        net = PairMLP(8, 128, 32, seed=1)
        rows, cols = index_codes(16), index_codes(16)
        g = torch.Generator().manual_seed(2)
        target = torch.randn(16, 16, generator=g) * 0.05
        flat = torch.arange(256)
        x = torch.cat([rows[flat // 16], cols[flat % 16]], dim=-1)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for _ in range(600):
            opt.zero_grad()
            loss = torch.mean((net(x) - target.reshape(-1)) ** 2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            resid = float(((net(x).view(16, 16) - target) ** 2).sum() / (target ** 2).sum())
        assert resid < 0.05, resid

    def test_fit_step_reduces_every_matrix_loss(self):
        m = _model()
        G = _genome(m, H=32, H2=16)
        W = {k: p.detach() for k, p in m.weights().items()}
        opt = torch.optim.Adam(G.parameters(), lr=3e-3)
        gen = torch.Generator().manual_seed(0)
        first = G.fit_step(W, fraction=1.0, cap=4096, generator=gen, opt=opt)
        for _ in range(150):
            last = G.fit_step(W, fraction=1.0, cap=4096, generator=gen, opt=opt)
        improved = [last[k] < first[k] for k in first]
        assert sum(improved) >= 0.9 * len(improved), (first, last)


# ------------------------------------------------------------------- I6, I7, I8: the loop
class TestLoop:
    def test_i6_eps_one_is_bit_identical_and_eps_zero_is_the_decode(self):
        m = _model()
        G = _genome(m)
        W = {k: p.detach().clone() for k, p in m.weights().items()}
        D = G.decode_all(W)
        regenerate(m, D, 1.0)
        for k, p in m.weights().items():
            assert torch.equal(p.detach(), W[k]), k
        regenerate(m, D, 0.0)
        for k, p in m.weights().items():
            assert torch.equal(p.detach(), D[k].to(p.dtype)), k
        regenerate(m, D, 0.5)
        for k, p in m.weights().items():
            expect = 0.5 * D[k] + 0.5 * D[k]              # from D at eps=0.5 the mix is D again
            assert torch.allclose(p.detach(), expect.to(p.dtype)), k

    def test_i7_epsilon_schedule_and_boundaries(self):
        assert epsilon(0, 20.0) == 1.0
        assert abs(epsilon(20, 20.0) - math.exp(-1)) < 1e-12
        assert abs(epsilon(60, 20.0) - math.exp(-3)) < 1e-12
        with pytest.raises(ValueError):
            epsilon(-1, 20.0)
        m = _model()
        G = _genome(m)
        hook = Intermittent(G, m, T_gen=100, lam=20.0, n_refit=1, fraction=1.0, cap=256,
                            lr=1e-3, evaluate=lambda model: 1.0, shuffle_every=2, seed=0)
        for step, toks in enumerate([0, 50, 99, 100, 150, 200, 250]):
            hook.before_step(step, toks)
        assert hook.generation == 2
        assert [r["tokens"] for r in hook.records] == [100, 200]
        assert [r["eps"] for r in hook.records] == [epsilon(1, 20.0), epsilon(2, 20.0)]
        assert hook.decode_calls == 2, "I10: one decode per generation, none elsewhere"
        assert not math.isnan(hook.records[0]["ce_shuffle"]) and not math.isnan(hook.records[1]["ce_shuffle"])

    def test_i10_no_decode_below_the_boundary(self):
        m = _model()
        G = _genome(m)
        hook = Intermittent(G, m, T_gen=1000, lam=20.0, n_refit=1, fraction=1.0, cap=256,
                            lr=1e-3, evaluate=lambda model: 1.0, seed=0)
        for step, toks in enumerate(range(0, 1000, 100)):
            hook.before_step(step, toks)
        assert hook.decode_calls == 0 and hook.generation == 0

    def test_i8_shuffle_preserves_the_multiset_of_rows(self):
        m = _model()
        G = _genome(m)
        D = G.decode_all({k: p.detach() for k, p in m.weights().items()})
        S = shuffle_rows(D, torch.Generator().manual_seed(0))
        for k in D:
            if k[0] in MATRIX_ROLES and D[k].dim() == 2:
                a = sorted(tuple(r.tolist()) for r in D[k])
                b = sorted(tuple(r.tolist()) for r in S[k])
                assert a == b, k
            else:
                assert torch.equal(D[k], S[k]), k


# ---------------------------------------------------------------------------- I9: accounting
class TestAccounting:
    def test_i9_ratio_is_phenotype_over_genome(self):
        m = _model()
        G = _genome(m)
        n_p = sum(p.numel() for p in m.weights().values())
        n_g = sum(p.numel() for p in G.parameters()) + 1          # + the uncompressed logit scale
        assert G.n_params() == n_g
        assert abs(G.compression_ratio(m) - n_p / n_g) < 1e-12
        s = G.summary()
        assert s["n_gnets"] == len(G.compressed_keys())
        assert s["n_params_genome"] == n_g

    def test_wider_genome_lower_ratio(self):
        m = _model()
        assert _genome(m, H=8, H2=4).compression_ratio(m) > _genome(m, H=64, H2=16).compression_ratio(m)


# ------------------------------------------------------------------- §3.2: the best checkpoint
class TestBestCheckpoint:
    def test_best_state_is_written_when_birth_improves(self, tmp_path):
        import json
        from bottleneck.transfer import load_source
        m = _model()
        G = _genome(m)
        # evaluate is called before, at birth, (at generation 1 and every shuffle_every: for the
        # shuffle control), and after; make each generation's birth better than the last so the
        # checkpoint must be rewritten each time
        seq = iter([5.0, 4.0, 9.0, 4.0,      # gen 1: before, birth, shuffle, after
                    5.0, 3.0, 3.0,           # gen 2
                    5.0, 2.0, 2.0])          # gen 3
        hook = Intermittent(G, m, T_gen=100, lam=20.0, n_refit=1, fraction=1.0, cap=256, lr=1e-3,
                            evaluate=lambda model: next(seq), seed=0, out_dir=tmp_path)
        for step, toks in enumerate([100, 200, 300]):
            hook.before_step(step, toks)
        assert hook.best_generation == 3 and hook.best_birth == 2.0
        assert (tmp_path / "genome_best.pt").exists() and (tmp_path / "phenotype_best.pt").exists()
        best = json.loads((tmp_path / "best.json").read_text(encoding="utf-8"))
        assert best["generation"] == 3 and best["ce_birth"] == 2.0
        # the saved genome is the live one: reloading it decodes identically
        (tmp_path / "loop.json").write_text(json.dumps(
            {"arm": "t", "args": {"seed": 0}, "genome": {"compression_ratio": 1.0}}), encoding="utf-8")
        torch.save({"state": G.state_dict(), "summary": G.summary(), "keys": G.keys, "shapes": G.shapes},
                   tmp_path / "genome.pt")
        from anlage.model import save_phenotype
        save_phenotype(m, tmp_path / "phenotype.pt")
        src = load_source(tmp_path, _rank(), device="cpu", which="best")
        assert src.which == "best" and src.generation == 3
        D_live = G.decode_all({k: p.detach() for k, p in m.weights().items()})
        for k in D_live:
            assert torch.equal(src.D[k], D_live[k]), k
        assert load_source(tmp_path, _rank(), device="cpu", which="final").which == "final"
