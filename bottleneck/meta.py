"""The loop over a distribution of tasks: the routine aimed at learning how to learn.

SPEC: docs/bottleneck/00-replication.md §10

Object implemented. `TaskDistribution` is a seeded sampler over `(family, params)` with a
held-out region (an unseen family, an unseen length). `MetaPool` is a `WindowPool`-compatible
pool whose windows come from the *current* task and which `advance()`s to the next drawn task;
a batch never mixes tasks. `MetaIntermittent` is §3's hook with one addition: it advances the
pool at every generation boundary, before regeneration, so the phenotype meets the next task
from the decode. `AdvanceOnly` is the same schedule of task switches with no genome -- the
sequential multi-task control of §10.2.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Callable, Iterator, Optional

import numpy as np
import torch

from anlage.model import Phenotype
from soma.openended.tasks import TaskParams

from .genome import BottleneckGenome
from .loop import Intermittent
from .transfer import TaskShard, task_token_array

__all__ = ["TaskDistribution", "MetaPool", "MetaIntermittent", "AdvanceOnly"]

TRAIN_FAMILIES = ("PARITY", "MODCOUNT", "PERM", "DYCK", "COPY", "MQAR")
HELDOUT_FAMILIES = ("GRID",)
TRAIN_LENGTHS = (4, 6, 8, 12)
HELDOUT_LENGTHS = (16,)
# one family-specific difficulty axis each (§10.1); values drawn uniformly
AXES: dict[str, tuple[str, tuple[int, ...]]] = {
    "PARITY": ("n_len", ()),                 # length is its only axis
    "MODCOUNT": ("m_mod", (2, 3, 4)),
    "PERM": ("p_grp", (3, 4)),
    "DYCK": ("depth", (2, 3)),
    "COPY": ("gap", (0, 2, 4)),
    "MQAR": ("P_mqar", (2, 3)),
    "GRID": ("n_ex", (1, 2)),
}


class TaskDistribution:
    """SPEC: §10.1. `sample(rng)` draws from the training region; `holdout_sample(rng)` from the
    held-out one. The two are disjoint by construction (M-I1): a held-out draw has either a
    family no training draw has, or a length no training draw has."""

    def __init__(self, *, train_families=TRAIN_FAMILIES, train_lengths=TRAIN_LENGTHS,
                 heldout_families=HELDOUT_FAMILIES, heldout_lengths=HELDOUT_LENGTHS):
        if set(train_families) & set(heldout_families):
            raise ValueError("held-out families must be disjoint from training families")
        if set(train_lengths) & set(heldout_lengths):
            raise ValueError("held-out lengths must be disjoint from training lengths")
        self.train_families, self.train_lengths = tuple(train_families), tuple(train_lengths)
        self.heldout_families, self.heldout_lengths = tuple(heldout_families), tuple(heldout_lengths)

    @staticmethod
    def _draw(rng: np.random.Generator, families, lengths) -> TaskParams:
        f = str(families[int(rng.integers(0, len(families)))])
        n = int(lengths[int(rng.integers(0, len(lengths)))])
        kw = {"family": f, "n_len": n}
        axis, values = AXES[f]
        if values:
            kw[axis] = int(values[int(rng.integers(0, len(values)))])
        return TaskParams(**kw)

    def sample(self, rng: np.random.Generator) -> TaskParams:
        return self._draw(rng, self.train_families, self.train_lengths)

    def holdout_sample(self, rng: np.random.Generator) -> TaskParams:
        # half the held-out draws are the unseen family at a seen length, half a seen family at
        # the unseen length, so both kinds of novelty are represented
        if rng.random() < 0.5:
            return self._draw(rng, self.heldout_families, self.train_lengths)
        return self._draw(rng, self.train_families, self.heldout_lengths)

    def is_heldout(self, p: TaskParams) -> bool:
        return p.family in self.heldout_families or p.n_len in self.heldout_lengths

    def sampler(self, *, heldout: bool) -> Callable[[np.random.Generator], TaskParams]:
        return self.holdout_sample if heldout else self.sample

    def describe(self) -> dict:
        return {"train_families": self.train_families, "train_lengths": self.train_lengths,
                "heldout_families": self.heldout_families, "heldout_lengths": self.heldout_lengths,
                "axes": {k: v for k, v in AXES.items()}}


class MetaPool:
    """SPEC: §10, M-I2. The task sequence is drawn once from `seed`; task `i`'s shard is built on
    first use and cached. `batches()` yields windows of the current task only."""

    def __init__(self, dist: TaskDistribution, *, n_tasks: int, n_windows: int, T: int, seed: int,
                 device: str):
        if n_tasks < 1 or n_windows < 1:
            raise ValueError("n_tasks >= 1 and n_windows >= 1")
        self.dist, self.n_tasks, self.n_windows, self.T, self.seed, self.device = dist, int(n_tasks), int(n_windows), int(T), int(seed), device
        rng = np.random.default_rng(int(seed))
        self.sequence: list[TaskParams] = [dist.sample(rng) for _ in range(self.n_tasks)]
        self._shards: dict[int, torch.Tensor] = {}
        self.current = 0
        self._cursor, self._epoch, self._order, self._order_seed = 0, 0, None, 0

    def params(self, i: Optional[int] = None) -> TaskParams:
        return self.sequence[self.current if i is None else i]

    def _windows(self, i: int) -> torch.Tensor:
        if i not in self._shards:
            p = self.sequence[i]
            arr = task_token_array((self.n_windows + 8) * (self.T + 1) // 2, seed=self.seed * 100_003 + i,
                                   sampler=lambda rng, p=p: p)
            shard = TaskShard(arr)
            rng = np.random.default_rng(self.seed * 100_003 + i + 1)
            offs = rng.integers(0, shard.n_tokens - self.T - 1, size=self.n_windows, dtype=np.int64)
            idx = offs[:, None] + np.arange(self.T + 1, dtype=np.int64)[None, :]
            self._shards[i] = torch.from_numpy(shard.tokens[idx]).to(self.device)
        return self._shards[i]

    def probe(self, n: int) -> torch.Tensor:
        """The first `n` windows of the current task: the Fisher probe of §11.1 for the task being
        learned, so it changes with `advance()` (M-I4)."""
        if n < 1:
            raise ValueError("n >= 1")
        return self._windows(self.current)[:n]

    def advance(self) -> int:
        """Move to the next task in the sequence (M-I2); the order restarts for the new task."""
        if self.current + 1 < self.n_tasks:
            self.current += 1
        self._cursor, self._epoch, self._order = 0, 0, None
        return self.current

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for p in self.sequence:
            h.update(repr(dataclasses.astuple(p)).encode())
        h.update(str((self.n_windows, self.T, self.seed)).encode())
        return h.hexdigest()

    def current_digest(self) -> str:
        return hashlib.blake2b(self._windows(self.current).cpu().numpy().tobytes(), digest_size=16).hexdigest()

    def batches(self, *, batch: int, steps: int, seed: int) -> Iterator[torch.Tensor]:
        if batch < 1 or steps < 1 or batch > self.n_windows:
            raise ValueError("batch in [1, n_windows] and steps >= 1")
        self._order_seed = int(seed)
        for _ in range(steps):
            win = self._windows(self.current)
            if self._order is None or self._cursor + batch > self.n_windows:
                self._epoch = 0 if self._order is None else self._epoch + 1
                g = torch.Generator(device="cpu").manual_seed(self._order_seed + 1_000_003 * self._epoch + 7 * self.current)
                self._order = torch.randperm(self.n_windows, generator=g)
                self._cursor = 0
            take = self._order[self._cursor:self._cursor + batch].to(self.device)
            self._cursor += batch
            yield win.index_select(0, take)


class MetaIntermittent(Intermittent):
    """§3's hook, advancing the task before regeneration (M-I3)."""

    def __init__(self, genome: BottleneckGenome, model: Phenotype, pool: MetaPool, **kw):
        super().__init__(genome, model, **kw)
        self.pool = pool

    def generation_step(self, step: int, tokens_done: int) -> dict:
        rec = super().generation_step(step, tokens_done)
        rec["task_before"] = dataclasses.asdict(self.pool.params())
        self.pool.advance()
        rec["task_next"] = dataclasses.asdict(self.pool.params())
        return rec


class AdvanceOnly:
    """The same task switches with no genome: sequential multi-task training (§10.2)."""

    def __init__(self, pool: MetaPool, *, T_gen: int, max_generations: Optional[int] = None,
                 evaluate: Optional[Callable[[Phenotype], float]] = None, model: Optional[Phenotype] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.pool, self.T_gen, self.max_generations = pool, int(T_gen), max_generations
        self.evaluate, self.model, self.log = evaluate, model, log
        self.generation, self.next_boundary, self.records = 0, int(T_gen), []

    def before_step(self, step: int, tokens_done: int) -> None:
        if self.max_generations is not None and self.generation >= self.max_generations:
            return None
        if tokens_done >= self.next_boundary:
            n = self.generation + 1
            ce = float(self.evaluate(self.model)) if (self.evaluate and self.model) else float("nan")
            rec = {"generation": n, "step": step, "tokens": tokens_done, "ce_before": ce,
                   "task_before": dataclasses.asdict(self.pool.params())}
            self.pool.advance()
            rec["task_next"] = dataclasses.asdict(self.pool.params())
            self.records.append(rec)
            self.generation, self.next_boundary = n, self.next_boundary + self.T_gen
            if self.log:
                self.log(f"  gen {n:4d}  tokens {tokens_done:>12,}  before {ce:.4f}  -> task {rec['task_next']['family']} n_len {rec['task_next']['n_len']}")
        return None

    def as_dict(self) -> dict:
        return {"T_gen": self.T_gen, "generations": self.generation, "records": self.records}
