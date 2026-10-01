"""Environments: a training pool under common random numbers, a fixed evaluation set, a
fitness slice. SPEC: docs/palimpsest/01-model.md §1.1 (Definitions 1, 2).

`stories` and `code` are prepared shards (`anlage/data.py::Corpus`, the v3 evaluation rule);
`tasks` is the reserved-alphabet verifier stream of `bottleneck/transfer.py`.
"""

from __future__ import annotations

import pathlib
from typing import Iterator, Optional

import numpy as np
import torch

from anlage.data import Corpus, EvalSet
from screen.crn import WindowPool

__all__ = ["Environment", "MixPool", "LifeSpec", "phases_for", "EVAL_SEEDS", "ENV_ROOTS"]

REPO = pathlib.Path(__file__).resolve().parent.parent
ENV_ROOTS = {"stories": "data/ts_8192", "code": "data/py_8192"}
EVAL_SEEDS = {"test": 20260902, "fitness": 20260903}
"""ANLAGE's v3 seeds (scripts/anlage_common.py), so `stories` scores are comparable to every
number in the repository."""
TASK_SEEDS = {"pool": 20260910, "test": 20260911}


class Environment:
    """One environment: `pool(seed)`, `test`, `fitness_windows(n)`. SPEC: Definition 1."""

    def __init__(self, name: str, *, vocab: int = 8192, T: int = 512, n_eval: int = 256,
                 device: str = "cuda", cache_dir: str | pathlib.Path = "runs/palimpsest/cache"):
        self.name, self.vocab, self.T, self.device = name, vocab, T, device
        if name in ENV_ROOTS:
            self.corpus = Corpus(REPO / ENV_ROOTS[name], vocab, cache_dir=cache_dir)
            shards = [self.corpus.fitness, self.corpus.test]
            self.test = EvalSet.sample_documents(shards, self.corpus.rank, n_windows=n_eval, T=T, split="test",
                                                 seed=EVAL_SEEDS["test"], mod=3, residue=2, device=device)
            self.fitness = EvalSet.sample_documents(shards, self.corpus.rank, n_windows=n_eval, T=T,
                                                    split="fitness", seed=EVAL_SEEDS["fitness"], mod=3,
                                                    residue=0, device=device)
            self._task = None
        elif name == "tasks":
            from bottleneck.transfer import task_eval_set
            self.corpus = None
            self.test = task_eval_set(n_eval, T=T, seed=TASK_SEEDS["test"], device=device)
            self.fitness = task_eval_set(n_eval, T=T, seed=TASK_SEEDS["test"] + 1, device=device)
            self._task = True
        else:
            raise KeyError(f"unknown environment {name!r}; have {sorted(ENV_ROOTS) + ['tasks']}")

    def pool(self, pool_seed: int, *, n_windows: int = 200_000) -> WindowPool:
        if self._task:
            from bottleneck.transfer import task_pool
            # the task stream is generated in Python: 60k windows (7.7M tokens) suffice for a
            # phase of 8M tokens at batch 32 (the pool wraps into a different order past its end)
            pool, _ = task_pool(min(n_windows, 60_000), T=self.T, seed=TASK_SEEDS["pool"] + pool_seed,
                                pool_seed=pool_seed, device=self.device)
            return pool
        return WindowPool(self.corpus.train, n_windows=n_windows, T=self.T, pool_seed=pool_seed,
                          device=self.device)

    def fitness_windows(self, n: int = 32) -> torch.Tensor:
        return self.fitness.windows[:n]

    def manifest(self) -> dict:
        m = {"name": self.name, "vocab": self.vocab, "T": self.T, "test_digest": self.test.digest,
             "fitness_digest": self.fitness.digest}
        if self.corpus is not None:
            m.update(self.corpus.manifest())
        return m


class MixPool:
    """The i.i.d. mixture of several pools: every batch takes an equal share of rows from
    each, in the order of one seed. SPEC: 01 §1.6 (`iid`)."""

    def __init__(self, pools: list[WindowPool]):
        if not pools:
            raise ValueError("at least one pool")
        self.pools = pools
        self.T = pools[0].T
        self.n_windows = sum(p.n_windows for p in pools)

    def digest(self) -> str:
        return "+".join(p.digest() for p in self.pools)

    def batches(self, *, batch: int, steps: int, seed: int) -> Iterator[torch.Tensor]:
        n = len(self.pools)
        share = [batch // n + (1 if i < batch % n else 0) for i in range(n)]
        gens = [p.batches(batch=max(1, s), steps=steps, seed=seed + 7919 * i)
                for i, (p, s) in enumerate(zip(self.pools, share)) if s > 0]
        for _ in range(steps):
            yield torch.cat([next(g) for g in gens], 0)


class LifeSpec:
    """The phases of a life: `(env, pool_seed, order_seed)` triples. SPEC: Definition 2."""

    def __init__(self, phases: list[tuple[str, int, int]]):
        if not phases:
            raise ValueError("a life has at least one phase")
        self.phases = [(str(e), int(p), int(o)) for e, p, o in phases]

    @property
    def envs(self) -> list[str]:
        seen = []
        for e, _, _ in self.phases:
            if e not in seen:
                seen.append(e)
        return seen

    def as_list(self) -> list[list]:
        return [list(p) for p in self.phases]


def phases_for(life: str, seed: int, *, K: int = 4) -> LifeSpec:
    """The registered lives. SPEC: 01 §1.1. Pool and order seeds derive from the run seed
    and the phase index so that every arm at one seed reads the same windows in the same
    order (CRN), and the stationary double sees a *different* pool every phase."""
    base = 1000 * (seed + 1)
    if life == "shift":
        envs = ["stories", "code"] * ((K + 1) // 2)
    elif life == "stat":
        envs = ["stories"] * K
    elif life == "shift3":
        envs = ["stories", "code", "tasks"]
    elif life == "statcode":
        envs = ["code"] * K
    else:
        raise KeyError(f"unknown life {life!r}")
    envs = envs[:K] if life != "shift3" else envs
    return LifeSpec([(e, base + 10 * k, base + 10 * k + 5) for k, e in enumerate(envs)])
