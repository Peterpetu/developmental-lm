"""Acceptance tests M-I1..M-I3 of docs/bottleneck/00-replication.md §10.4, on CPU.

SPEC: docs/bottleneck/00-replication.md §10
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from anlage.config import StageConfig
from anlage.model import Phenotype
from bottleneck.genome import BottleneckGenome
from bottleneck.meta import (HELDOUT_FAMILIES, HELDOUT_LENGTHS, AdvanceOnly, MetaIntermittent, MetaPool,
                             TaskDistribution)
from soma.openended.tasks import RESERVED_SIZE

VOCAB = 64
CFG = StageConfig(stage=0, d_model=32, L=2, n_h=2, n_kv_attn=1, d_ff=64, d_head=16, chunk_size=16)


class TestDistribution:
    def test_mi1_deterministic_and_disjoint(self):
        d = TaskDistribution()
        # one generator per sequence: re-seeding inside the comprehension would draw the same
        # task 50 times, which is how the first version of this test produced 500 identical draws
        ra, rb = np.random.default_rng(0), np.random.default_rng(0)
        a = [d.sample(ra) for _ in range(50)]
        b = [d.sample(rb) for _ in range(50)]
        assert a == b and len({(p.family, p.n_len) for p in a}) > 5
        r1, r2 = np.random.default_rng(1), np.random.default_rng(2)
        train = [d.sample(r1) for _ in range(500)]
        held = [d.holdout_sample(r2) for _ in range(500)]
        assert not any(d.is_heldout(p) for p in train)
        assert all(d.is_heldout(p) for p in held)
        assert any(p.family in HELDOUT_FAMILIES for p in held) and any(p.n_len in HELDOUT_LENGTHS for p in held)
        assert {p.family for p in train} == set(d.train_families)
        with pytest.raises(ValueError):
            TaskDistribution(train_families=("GRID",), heldout_families=("GRID",))


class TestMetaPool:
    def test_mi2_advance_switches_task_and_batches_never_mix(self):
        d = TaskDistribution()
        pool = MetaPool(d, n_tasks=4, n_windows=32, T=16, seed=3, device="cpu")
        assert len(pool.sequence) == 4
        d0 = pool.current_digest()
        gen = pool.batches(batch=8, steps=100, seed=0)
        b = next(gen)
        assert b.shape == (8, 17) and b.max() < RESERVED_SIZE
        assert torch.equal(b[0], pool._windows(0)[pool._order[0]])       # from task 0's windows
        pool.advance()
        assert pool.current == 1 and pool.current_digest() != d0
        b2 = next(gen)
        assert all(any(torch.equal(row, w) for w in pool._windows(1)) for row in b2), "after advance, every row is task 1's"
        # the last task is held: advancing past the end stays on it
        pool.advance(); pool.advance(); pool.advance()
        assert pool.current == 3
        assert pool.digest() == MetaPool(d, n_tasks=4, n_windows=32, T=16, seed=3, device="cpu").digest()


class TestMetaHook:
    def _model_genome(self):
        torch.manual_seed(0)
        m = Phenotype(CFG, VOCAB, seed=0)
        rank = torch.randperm(VOCAB, generator=torch.Generator().manual_seed(0))
        return m, BottleneckGenome(m, rank, H=8, H2=4, H2_vector=3, seed=0)

    def test_mi3_meta_hook_advances_once_per_generation_at_the_boundary(self):
        m, G = self._model_genome()
        pool = MetaPool(TaskDistribution(), n_tasks=5, n_windows=16, T=16, seed=1, device="cpu")
        hook = MetaIntermittent(G, m, pool, T_gen=100, lam=20.0, n_refit=1, fraction=1.0, cap=64,
                                lr=1e-3, evaluate=lambda model: 1.0, seed=0)
        seen = []
        for step, toks in enumerate([0, 50, 100, 150, 200, 250, 300]):
            hook.before_step(step, toks)
            seen.append(pool.current)
        assert hook.generation == 3 and pool.current == 3
        assert seen == [0, 0, 1, 1, 2, 2, 3]
        r = hook.records[0]
        assert r["task_before"] == dataclasses.asdict(pool.sequence[0])
        assert r["task_next"] == dataclasses.asdict(pool.sequence[1])

    def test_mi3_advance_only_matches_the_meta_schedule(self):
        m, _ = self._model_genome()
        pool = MetaPool(TaskDistribution(), n_tasks=5, n_windows=16, T=16, seed=1, device="cpu")
        hook = AdvanceOnly(pool, T_gen=100, evaluate=lambda model: 2.0, model=m)
        for step, toks in enumerate([0, 50, 100, 150, 200, 250, 300]):
            hook.before_step(step, toks)
        assert hook.generation == 3 and pool.current == 3
        assert [r["tokens"] for r in hook.records] == [100, 200, 300]
        assert hook.records[0]["ce_before"] == 2.0
