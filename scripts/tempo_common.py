"""Shared pieces of the TEMPO run scripts: the corpus, the v3 evaluation windows for the
span agent, manifests, logging, run directories.

SPEC: 10 §10.1  (evaluation sets, noise, CRN, one chain)
SPEC: 10 §10.4  (manifests and the register)
"""

from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from anlage.data import Corpus, EvalSet  # noqa: E402
from gnomon.utils.report import write_json  # noqa: E402
from tempo.config import PILE_ADULT, TS_ADULT, TempoConfig  # noqa: E402

__all__ = ["REPO", "RUNS", "manifest", "load_corpus", "eval_windows", "log_to", "write_json",
           "config_by_name", "EVAL_VERSION", "EVAL_SEEDS", "run_dir", "file_sha256", "array_digest",
           "scripts_digest", "step_time_outliers"]

RUNS = REPO / "runs" / "tempo"
EVAL_VERSION = 3
EVAL_SEEDS = {"test": 20260902, "fitness": 20260903}
"""ANLAGE's v3 seeds (scripts/anlage_common.py), so the sets are the same documents."""


def _git() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).decode().strip())
    except Exception:                                          # noqa: BLE001
        commit, dirty = "unknown", True
    return {"commit": commit, "dirty": dirty}


def manifest(extra: dict) -> dict:
    """SPEC: 10 §10.4 -- the fields every run records."""
    from tempo.life import code_digest
    m = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git(), "code_digest": code_digest(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(), "python": sys.version.split()[0],
        "eval_version": EVAL_VERSION,
    }
    m.update(extra)
    return m


def file_sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(a: np.ndarray) -> str:
    """The digest of a fixed evaluation set (10 §10.4): its shape and its bytes."""
    import hashlib
    h = hashlib.sha256()
    h.update(str(tuple(a.shape)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def scripts_digest() -> str:
    """A hash of `scripts/tempo_*.py` (the run scripts are part of the provenance)."""
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    for f in sorted((REPO / "scripts").glob("tempo_*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()


def step_time_outliers(seconds: list, factor: float = 3.0) -> dict:
    """Steps slower than `factor` x the median (the card is shared: 10 §10.4 records them)."""
    if not seconds:
        return {"n": 0, "median": float("nan"), "outliers": 0, "max": float("nan")}
    med = float(np.median(seconds))
    return {"n": len(seconds), "median": med, "outliers": int(sum(1 for s in seconds if s > factor * med)),
            "max": float(max(seconds))}


def config_by_name(name: str) -> TempoConfig:
    """A base configuration, optionally followed by `-<key><value>` modifiers.

    Bases: `ts` (K = 4), `tsk2`, `tsk1`, `tsk1dx512`, `tsk1dc512`, `tsk1wide`, `pile`.
    Modifiers: `dg<int>` (genome dimension), `dx<int>` (interface width), `dc<int>` (codec and
    consolidator width), `r<int>` (expression rank). So `tsk1wide-dg32` is `K = 1` with the wide
    interface and a 32-dimensional genome.

    Why composable (11 §P-D.9). Two axes now need sweeping and their product is not expressible
    as a table. The interface axis is being decided by the factorial of §P-D.4; the genome axis
    must be swept because the search captures `2.3 %` of the available improvement at
    `d_g = 512` against `41 %` at `d_g = 32`, and that sweep has to run at whichever interface
    the factorial selects. `d_g` is a free parameter of the design in the sense that matters:
    the trust region `tau = 0.2 sqrt(d_g)`, the bound `3 sqrt(d_g)`, the expression map
    `A (n_roles, r, d_g)` and the read-out `W_R` over `[feat; g]` all scale with it, so changing
    it changes nothing else.

    The parser is strict: an unknown key or a non-integer value raises. A typo that silently
    returned the base configuration would run the wrong shape under the right name, and nothing
    downstream could detect it.
    """
    import dataclasses
    base_name, *mods = name.split("-")
    bases = {"ts": TS_ADULT, "tsk2": TS_ADULT, "tsk1": TS_ADULT, "tsk1dx512": TS_ADULT,
             "tsk1dc512": TS_ADULT, "tsk1wide": TS_ADULT, "pile": PILE_ADULT}
    if base_name not in bases:
        raise KeyError(f"unknown configuration base {base_name!r}; known: {sorted(bases)}")
    cfg = bases[base_name]
    preset = {"tsk2": {"K": 2}, "tsk1": {"K": 1}, "tsk1dx512": {"K": 1, "d_x": 512},
              "tsk1dc512": {"K": 1, "d_c": 512}, "tsk1wide": {"K": 1, "d_x": 512, "d_c": 512}}
    if base_name in preset:
        cfg = dataclasses.replace(cfg, **preset[base_name])
    keys = {"dg": "d_g", "dx": "d_x", "dc": "d_c", "r": "r", "K": "K"}
    for m in mods:
        for k in sorted(keys, key=len, reverse=True):
            if m.startswith(k) and m[len(k):].isdigit():
                cfg = dataclasses.replace(cfg, **{keys[k]: int(m[len(k):])})
                break
        else:
            raise ValueError(f"unknown configuration modifier {m!r} in {name!r}; "
                             f"known keys {sorted(keys)} each followed by an integer")
    return cfg


def load_corpus(root: str = "data/ts_8192", vocab: int = 8192) -> Corpus:
    return Corpus(REPO / root, vocab, cache_dir=str(RUNS / "cache"))


def eval_windows(corpus: Corpus, cfg: TempoConfig, *, n: int = 256, split: str = "test") -> np.ndarray:
    """The v3 windows of `T + K` tokens for the span agent: the same documents ANLAGE's v3
    sets draw (union of both validation shards, document index mod 3: test 2, fitness 0),
    each window `K - 1` tokens longer so that the scored tokens `K .. T + K - 1` number `T`.
    SPEC: 10 §10.1."""
    residue = {"test": 2, "fitness": 0}[split]
    ev = EvalSet.sample_documents([corpus.fitness, corpus.test], corpus.rank, n_windows=n,
                                  T=cfg.T + cfg.K - 1, split=split, seed=EVAL_SEEDS[split],
                                  mod=3, residue=residue, device="cpu")
    return ev.windows.numpy().astype(np.int64)


def run_dir(name: str) -> pathlib.Path:
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_to(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    return log
