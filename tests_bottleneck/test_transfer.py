"""Acceptance tests T-I1..T-I5 of docs/bottleneck/00-replication.md §9.4, on CPU.

SPEC: docs/bottleneck/00-replication.md §9
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from anlage.config import ROLE_EMB, ROLE_LOGIT_SCALE, ROLE_NORM_FINAL, StageConfig
from anlage.lifetime import evaluate
from anlage.model import Phenotype, save_phenotype
from bottleneck.genome import BottleneckGenome
from bottleneck.transfer import (LAYER_SETS, Source, initialise, layer_keys, load_source,
                                 task_eval_set, task_pool, task_token_array)
from soma.openended.tasks import BOS, EOS, RESERVED_SIZE, SEP

VOCAB = 64
CFG = StageConfig(stage=0, d_model=32, L=4, n_h=2, n_kv_attn=1, d_ff=64, d_head=16, chunk_size=16)


def _rank(seed=0):
    return torch.randperm(VOCAB, generator=torch.Generator().manual_seed(seed))


class TestTarget:
    def test_ti1_task_tokens_are_deterministic_and_reserved(self):
        a = task_token_array(5000, seed=3)
        b = task_token_array(5000, seed=3)
        c = task_token_array(5000, seed=4)
        assert np.array_equal(a, b) and not np.array_equal(a[:len(c)], c[:len(a)])
        assert a.min() >= 0 and a.max() < RESERVED_SIZE
        assert a[0] == BOS and SEP in a and EOS in a
        assert len(a) >= 5000

    def test_ti1_pool_and_eval_set_run_through_train_machinery(self):
        pool, shard = task_pool(64, T=16, seed=1, pool_seed=2, device="cpu")
        b = next(pool.batches(batch=8, steps=1, seed=0))
        assert b.shape == (8, 17) and b.max() < RESERVED_SIZE
        ev = task_eval_set(12, T=16, seed=5, device="cpu")
        assert ev.n_windows == 12 and ev.windows.shape == (12, 17) and ev.strata.shape == (12, 16)
        m = Phenotype(CFG, VOCAB, seed=0)
        ce, per = evaluate(m, ev)
        assert np.isfinite(ce) and ce > 0
        assert not np.isnan(per[0]) and all(np.isnan(p) for p in per[1:]), "one stratum only"


class TestSource:
    def test_ti2_genome_round_trip_decodes_identically(self, tmp_path: pathlib.Path):
        torch.manual_seed(0)
        m = Phenotype(CFG, VOCAB, seed=0)
        rank = _rank()
        G = BottleneckGenome(m, rank, H=8, H2=4, H2_vector=3, seed=0)
        W = {k: p.detach().clone() for k, p in m.weights().items()}
        D_run = G.decode_all(W)
        torch.save({"state": G.state_dict(), "summary": G.summary(), "keys": G.keys, "shapes": G.shapes},
                   tmp_path / "genome.pt")
        save_phenotype(m, tmp_path / "phenotype.pt")
        (tmp_path / "loop.json").write_text(json.dumps(
            {"arm": "GN8", "args": {"seed": 0}, "genome": {"compression_ratio": 12.5}}), encoding="utf-8")
        src = load_source(tmp_path, rank, device="cpu")
        assert src.arm == "GN8" and src.ratio == 12.5 and src.D is not None
        for k in D_run:
            assert torch.equal(src.D[k], D_run[k]), k
            assert torch.equal(src.W[k], W[k]), k


class TestInitialise:
    def _source(self):
        m = Phenotype(CFG, VOCAB, seed=1)
        W = {k: p.detach().clone() for k, p in m.weights().items()}
        D = {k: torch.full_like(v, 0.123) for k, v in W.items()}
        return Source(pathlib.Path("."), "x", CFG, VOCAB, W, D, 5.0, 1)

    def test_ti3_layer_sets_partition_as_specified(self):
        m = Phenotype(CFG, VOCAB, seed=0)
        keys = list(m.weights().keys())
        low, high, all_ = (layer_keys(CFG, keys, w) for w in ("low", "high", "all"))
        assert all_ == set(keys)
        assert (ROLE_EMB, -1) in low and all(k[1] in (0, 1) for k in low if k[1] >= 0)
        assert (ROLE_NORM_FINAL, -1) in high and (ROLE_LOGIT_SCALE, -1) in high
        assert all(k[1] in (CFG.L - 2, CFG.L - 1) for k in high if k[1] >= 0)
        assert not (low & high)

    @pytest.mark.parametrize("which", LAYER_SETS)
    @pytest.mark.parametrize("mode", ("copy", "decode"))
    def test_ti3_initialise_writes_only_the_chosen_keys(self, which, mode):
        src = self._source()
        m = Phenotype(CFG, VOCAB, seed=7)
        fresh = {k: p.detach().clone() for k, p in m.weights().items()}
        written = initialise(m, src, mode, which)
        chosen = layer_keys(CFG, fresh.keys(), which)
        assert len(written) == len(chosen)
        ref = src.W if mode == "copy" else src.D
        for k, p in m.weights().items():
            if k in chosen:
                assert torch.equal(p.detach(), ref[k]), k
            else:
                assert torch.equal(p.detach(), fresh[k]), k

    def test_ti3_random_writes_nothing(self):
        m = Phenotype(CFG, VOCAB, seed=7)
        fresh = {k: p.detach().clone() for k, p in m.weights().items()}
        assert initialise(m, None, "random", "all") == []
        for k, p in m.weights().items():
            assert torch.equal(p.detach(), fresh[k])
        with pytest.raises(ValueError):
            initialise(m, None, "copy", "all")
