"""Transfer through the bottleneck: the paper's Fig. 3G and Fig. 5G, on a language model.

SPEC: docs/bottleneck/00-replication.md §9

Object implemented. A *source* run of the loop leaves two artefacts: the trained phenotype
`W_src` and the genome `θ_src`. On a *target* distribution the phenotype never saw, three
initialisations are trained under one recipe, one seed and one window order:

    random      a fresh phenotype                                         (the paper's "No GN")
    copy        `W_src` copied into the phenotype                          ("direct transfer")
    decode      `g_{θ_src}(codes)` decoded into the phenotype              ("g-network mediated")

each on a layer subset -- every layer, the lowest two, or the highest two -- because the
paper's one positive transfer result is layer-specific (layer 1 of a CNN, 1.5x, p. 6) and its
two negatives are layer-specific too (layers 8-9 worse than random; MNIST->F-MNIST worse than
random). "Learns how to learn" is then a statement about the *target* learning curve of
`decode` against `random` and against `copy`, which §9 pre-registers.

The target distribution is the reserved-alphabet verifier stream of `soma/openended/tasks.py`
-- the same tokenizer, ids the phenotype has never seen in natural text, exact structure --
packed into `(T+1)` windows exactly as the corpus is, so `anlage.lifetime.train` and
`evaluate` run on it unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from typing import Iterable, Optional

import numpy as np
import torch

from anlage.config import (GLOBAL_ROLES, ROLE_EMB, ROLE_LOGIT_SCALE, ROLE_NORM_FINAL,
                           StageConfig)
from anlage.data import EvalSet
from anlage.model import Phenotype, load_phenotype
from screen.crn import WindowPool
from soma.openended.tasks import BOS, EOS, FAMILIES, RESERVED_SIZE, SEP, TaskParams, generate

from .genome import BottleneckGenome

__all__ = ["task_token_array", "TaskShard", "task_pool", "task_eval_set", "Source",
           "load_source", "layer_keys", "initialise", "LAYER_SETS"]

LAYER_SETS = ("all", "low", "high")


# ------------------------------------------------------------------- the target distribution
def task_token_array(n_tokens: int, *, seed: int, families: Iterable[str] = FAMILIES,
                     sampler=None) -> np.ndarray:
    """At least `n_tokens` tokens of `BOS prompt SEP answer EOS` instances, from `seed` alone.
    SPEC: §9.1. By default the families round-robin at their default (lower-bound) parameters;
    with `sampler(rng) -> TaskParams` each instance's parameters are drawn by the sampler (§10).

    Every id lies in the reserved block `[0, RESERVED_SIZE)`, which natural text never emits
    (`soma/data/tokenizer.py::assert_reserved_block_unused`), so the phenotype meets these ids
    for the first time on the target."""
    if n_tokens < 1:
        raise ValueError("n_tokens >= 1")
    rng = np.random.default_rng(int(seed))
    fams = tuple(families)
    out: list[int] = []
    i = 0
    while len(out) < n_tokens:
        params = sampler(rng) if sampler is not None else TaskParams(family=fams[i % len(fams)])
        inst = generate(params, rng)
        out.extend(inst.tokens())
        i += 1
    arr = np.asarray(out, dtype=np.int64)
    if arr.min() < 0 or arr.max() >= RESERVED_SIZE:
        raise ValueError("task tokens must lie in the reserved block")
    return arr


class TaskShard:
    """The one attribute `WindowPool` reads from a shard."""

    def __init__(self, tokens: np.ndarray):
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.n_tokens = int(self.tokens.shape[0])
        self.digest = hashlib.blake2b(self.tokens.tobytes(), digest_size=16).hexdigest()


def task_pool(n_windows: int, *, T: int, seed: int, pool_seed: int, device: str,
              sampler=None) -> tuple[WindowPool, TaskShard]:
    """A `WindowPool` over a task token array: the same object the corpus pool is."""
    shard = TaskShard(task_token_array((n_windows + 8) * (T + 1) // 4, seed=seed, sampler=sampler))
    return WindowPool(shard, n_windows=n_windows, T=T, pool_seed=pool_seed, device=device), shard


def task_eval_set(n_windows: int, *, T: int, seed: int, device: str, sampler=None) -> EvalSet:
    """A fixed evaluation set of task windows, one stratum, disjoint from the pool by seed.
    `evaluate` reads `windows`, `strata` and `n_windows` and nothing else. SPEC: §9.1."""
    shard = TaskShard(task_token_array((n_windows + 8) * (T + 1), seed=seed, sampler=sampler))
    rng = np.random.default_rng(int(seed) + 17)
    offs = rng.integers(0, shard.n_tokens - T - 1, size=n_windows, dtype=np.int64)
    idx = offs[:, None] + np.arange(T + 1, dtype=np.int64)[None, :]
    win = shard.tokens[idx]
    strata = np.zeros((n_windows, T), dtype=np.int64)
    digest = hashlib.blake2b(win.tobytes(), digest_size=16).hexdigest()
    return EvalSet(torch.from_numpy(win).to(device), torch.from_numpy(strata).to(device), "task", digest,
                   offsets=offs, sources=None)


# ------------------------------------------------------------------------------- the source
@dataclasses.dataclass
class Source:
    """A finished source run: its trained weights and its genome, decoded. SPEC: §9."""

    run_dir: pathlib.Path
    arm: str
    cfg: StageConfig
    vocab: int
    W: dict                              # `{(role, l): tensor}` the trained phenotype
    D: Optional[dict]                    # `{(role, l): tensor}` the decode, or None for `free`
    ratio: Optional[float]
    seed: int
    which: str = "final"                 # `best` (the generation of best at-birth CE, §3.2) or `final`
    generation: Optional[int] = None


def load_source(run_dir, rank: torch.Tensor, *, device: str, which: str = "best") -> Source:
    """Rebuild the genome under the run's own `H, H₂, seed`, load its state, and decode it against
    the trained phenotype it was fitted to. `which="best"` reads `genome_best.pt` and
    `phenotype_best.pt` -- the state at the generation of best at-birth CE (§3.2), the paper's
    reported quantity -- and falls back to the final state when no best checkpoint exists; the
    choice actually made is recorded in `Source.which`. A round trip test (T-I2) holds the decode
    here equal to the run's own decode."""
    if which not in ("best", "final"):
        raise ValueError("which is 'best' or 'final'")
    run_dir = pathlib.Path(run_dir)
    rep = json.loads((run_dir / "loop.json").read_text(encoding="utf-8"))
    use_best = which == "best" and (run_dir / "genome_best.pt").exists() and (run_dir / "phenotype_best.pt").exists()
    model = load_phenotype(run_dir / ("phenotype_best.pt" if use_best else "phenotype.pt"), device=device)
    W = {k: p.detach().clone() for k, p in model.weights().items()}
    D, ratio, generation = None, None, None
    gp = run_dir / ("genome_best.pt" if use_best else "genome.pt")
    if gp.exists():
        blob = torch.load(gp, map_location=device, weights_only=False)
        s = blob["summary"]
        G = BottleneckGenome(model, rank, H=s["H"], H2=s["H2"], H2_vector=s["H2_vector"],
                             seed=int(rep["args"]["seed"]), device=device, keep=s.get("keep_roles", ()),
                             activation=s.get("activation", "relu"),
                             rank_roles={int(r): int(k) for r, k in s.get("rank_roles", [])})
        G.load_state_dict(blob["state"])
        D = G.decode_all(W)
        ratio = float(rep["genome"]["compression_ratio"])
        generation = blob.get("generation")
    return Source(run_dir, rep["arm"], model.cfg, int(model.vocab), W, D, ratio, int(rep["args"]["seed"]),
                  "best" if use_best else "final", generation)


# -------------------------------------------------------------------------- initialisation
def layer_keys(cfg: StageConfig, keys: Iterable[tuple[int, int]], which: str) -> set[tuple[int, int]]:
    """The keys a layer set names. SPEC: §9.2.

    `all`: everything. `low`: layers 0 and 1 plus the embedding (the input side, the paper's
    "layer 1"). `high`: the last two layers plus the final norm and logit scale (the output side,
    the paper's "layers 8-9")."""
    if which not in LAYER_SETS:
        raise ValueError(f"layer set must be one of {LAYER_SETS}")
    keys = list(keys)
    if which == "all":
        return set(keys)
    if which == "low":
        return {k for k in keys if k[1] in (0, 1) or k == (ROLE_EMB, -1)}
    L = cfg.L
    return {k for k in keys if k[1] in (L - 2, L - 1) or k in ((ROLE_NORM_FINAL, -1), (ROLE_LOGIT_SCALE, -1))}


@torch.no_grad()
def initialise(model: Phenotype, source: Optional[Source], mode: str, which: str) -> list[str]:
    """Set `model`'s weights for one arm: `random` leaves the fresh initialisation; `copy` writes
    `W_src` and `decode` writes `D_src` on the layer set `which`, everything else staying fresh.
    Returns the names of the keys written, for the record. SPEC: §9.2."""
    if mode == "random":
        return []
    if source is None:
        raise ValueError("copy and decode need a source")
    src = source.W if mode == "copy" else source.D
    if src is None:
        raise ValueError(f"source {source.arm} has no decode")
    if mode not in ("copy", "decode"):
        raise ValueError(f"unknown init mode {mode!r}")
    live = model.weights()
    chosen = layer_keys(model.cfg, live.keys(), which)
    written = []
    for k, p in live.items():
        if k in chosen:
            if src[k].shape != p.shape:
                raise ValueError(f"{k}: source shape {tuple(src[k].shape)} vs target {tuple(p.shape)}")
            p.data.copy_(src[k].to(p.dtype))
            written.append(f"{k[0]}_{k[1]}")
    return written
