"""Shared plumbing for the `anlage_*` run scripts: manifests, corpus, eval sets.

SPEC: 08 §8.5
"""

from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from anlage.config import Lineage, get_lineage  # noqa: E402
from anlage.data import Corpus, EvalSet  # noqa: E402
from gnomon.utils.report import write_json  # noqa: E402

__all__ = ["REPO", "manifest", "load_corpus", "eval_sets", "log_to", "write_json",
           "get_lineage", "Lineage"]


def _git() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).decode().strip())
    except Exception:                                          # noqa: BLE001
        commit, dirty = "unknown", True
    return {"commit": commit, "dirty": dirty}


def manifest(extra: dict) -> dict:
    """SPEC: 08 §8.5 -- the fields every run records."""
    m = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "eval_version": EVAL_VERSION,
    }
    m.update(extra)
    return m


def load_corpus(lin: Lineage) -> Corpus:
    return Corpus(REPO / lin.data_root, lin.vocab)


EVAL_VERSION = 3
"""SPEC: 08 §8.2 -- v1 (R1-R4 artifacts): the first consecutive windows of each split; v2:
windows sampled uniformly over the split (the two validation halves differ in composition,
R6); v3 (from R3's re-evaluation on): windows sampled over the union of both validation
shards, split by document index mod 3 -- fitness 0, merge-fit 1, test 2 -- so the three sets
share no document and have one composition. The version is recorded in every manifest."""
EVAL_SEEDS = {"test": 20260902, "fitness": 20260903, "fitw": 20260904}


def eval_sets(corpus: Corpus, lin: Lineage, *, n_test: int = 256, n_fit: int = 256,
              device="cuda", version: int = EVAL_VERSION) -> tuple[EvalSet, EvalSet]:
    """`(test, fitness)` fixed sets of `n` windows each. SPEC: 08 §8.2.

    v2: the fitness set is drawn from the first half of the fitness shard and the merge's
    coefficient-fit windows (`fit_windows`) from the second half, so they share no token (S8)."""
    if version == 1:
        test = EvalSet.build(corpus.test, corpus.rank, n_windows=n_test, T=lin.T, split="test", device=device)
        fit = EvalSet.build(corpus.fitness, corpus.rank, n_windows=n_fit, T=lin.T, split="fitness", device=device)
        return test, fit
    if version == 2:
        test = EvalSet.sample(corpus.test, corpus.rank, n_windows=n_test, T=lin.T, split="test",
                              seed=EVAL_SEEDS["test"], device=device)
        fit = EvalSet.sample(corpus.fitness, corpus.rank, n_windows=n_fit, T=lin.T, split="fitness",
                             seed=EVAL_SEEDS["fitness"], region=(0.0, 0.5), device=device)
        return test, fit
    shards = [corpus.fitness, corpus.test]
    test = EvalSet.sample_documents(shards, corpus.rank, n_windows=n_test, T=lin.T, split="test",
                                    seed=EVAL_SEEDS["test"], mod=3, residue=2, device=device)
    fit = EvalSet.sample_documents(shards, corpus.rank, n_windows=n_fit, T=lin.T, split="fitness",
                                   seed=EVAL_SEEDS["fitness"], mod=3, residue=0, device=device)
    return test, fit


def eval_shards(corpus: Corpus) -> list:
    """The shards v3 evaluation sets draw from, in order (for `classify_eval_windows`)."""
    return [corpus.fitness, corpus.test]


def fit_windows(corpus: Corpus, lin: Lineage, *, n: int = 64, device="cuda", version: int = EVAL_VERSION):
    """The merge's coefficient-fit windows, disjoint from the fitness set (S8)."""
    if version == 1:
        return EvalSet.build(corpus.fitness, corpus.rank, n_windows=n, T=lin.T, split="fitness",
                             device=device, offset_windows=256).windows
    if version == 2:
        return EvalSet.sample(corpus.fitness, corpus.rank, n_windows=n, T=lin.T, split="fitness",
                              seed=EVAL_SEEDS["fitw"], region=(0.5, 1.0), device=device).windows
    return EvalSet.sample_documents([corpus.fitness, corpus.test], corpus.rank, n_windows=n, T=lin.T,
                                    split="fitness", seed=EVAL_SEEDS["fitw"], mod=3, residue=1,
                                    device=device).windows


def log_to(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    return log
