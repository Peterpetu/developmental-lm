"""Structured run logging and the periodic-hook counter.

SPEC: 07 §7.6
SPEC: 01 §1.7

Two things live here because they are the same concern — *when* something is
recorded and *what* is recorded — and because the counter is subtle enough that
one implementation, tested once, is safer than a counter inlined at each of the
six hook sites.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

__all__ = ["EventCounter", "JsonlLogger", "RunPaths"]


@dataclass
class EventCounter:
    """The explicit next-event counter of (1.15).

    SPEC: 01 §1.7

    `T_sel(n)` is a non-integer increasing function, so `n mod T_sel(n) == 0` is
    **not** a schedule: for most `n` the modulus is never zero, and for some it
    is zero at several consecutive steps. (1.15) replaces it with

        fire when n >= n_next and n > 0 ;  then  n_next <- n + ceil(T_x(n))

    The `n > 0` guard is load-bearing. Spec v1's modulus fires at `n = 0`, where
    no token has been scored, no unit has a utility, and the case-weight vector
    is identically zero (§04.2's warmup correction).
    """

    period: Callable[[int], float] | float
    """Either a constant period or `n -> T_x(n)`."""

    warmup: int = 0
    """Extra guard: the hook does not fire before this step (§04.2's
    `n >= max(T_0, M_mat)`)."""

    n_next: int = field(init=False)
    fired: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.n_next = self._period_at(0)

    def _period_at(self, n: int) -> int:
        p = self.period(n) if callable(self.period) else float(self.period)
        return max(1, math.ceil(p))

    def due(self, n: int) -> bool:
        """Whether the hook fires at step `n`. Does not advance the counter."""
        return n > 0 and n >= self.n_next and n >= self.warmup

    def fire(self, n: int) -> None:
        """Advance after firing. Must follow a `due(n)` that returned True."""
        self.n_next = n + self._period_at(n)
        self.fired += 1

    def maybe(self, n: int) -> bool:
        """`due` then `fire`, as a single call for the common case."""
        if self.due(n):
            self.fire(n)
            return True
        return False

    def state(self) -> Dict[str, int]:
        return {"n_next": self.n_next, "fired": self.fired}

    def load_state(self, s: Mapping[str, int]) -> None:
        self.n_next = int(s["n_next"])
        self.fired = int(s["fired"])


@dataclass
class RunPaths:
    """Canonical layout of `runs/<run_id>/` (§07.6, §02.3)."""

    root: Path

    @classmethod
    def create(cls, root: str | os.PathLike, run_id: str) -> "RunPaths":
        p = Path(root) / run_id
        p.mkdir(parents=True, exist_ok=True)
        (p / "checkpoints").mkdir(exist_ok=True)
        return cls(p)

    @property
    def log(self) -> Path:
        return self.root / "log.jsonl"

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def env(self) -> Path:
        return self.root / "env.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"


class JsonlLogger:
    """Newline-delimited JSON to `runs/<run_id>/log.jsonl` (§07.6).

    Every record carries `step`, `wall` and a `kind` discriminator, so a single
    file holds the optimisation, population, structure, plasticity, selection,
    stream and barrier channels §07.6 enumerates without a schema per channel.

    Floats are rounded on write to keep the file readable and diffable; the
    rounding is applied at 6 significant figures, which is far below any
    threshold the specification tests against (the tightest is `1e-5`, and that
    is checked in-process, not from the log).
    """

    def __init__(self, path: str | os.PathLike, *, flush_every: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._t0 = time.time()
        self._n = 0
        self._flush_every = max(1, flush_every)

    def log(self, kind: str, step: int, **fields: Any) -> None:
        rec: Dict[str, Any] = {
            "kind": kind,
            "step": int(step),
            "wall": round(time.time() - self._t0, 3),
        }
        for k, v in fields.items():
            rec[k] = _jsonable(v)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._n += 1
        if self._n % self._flush_every == 0:
            self._fh.flush()

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def read(path: str | os.PathLike, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read records back, optionally filtered by `kind`."""
        out: List[Dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            return out
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if kind is None or rec.get("kind") == kind:
                    out.append(rec)
        return out


def _jsonable(v: Any) -> Any:
    """Coerce tensors, arrays and numpy scalars to JSON-safe values."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return float(f"{v:.6g}")
    # torch / numpy without importing them at module scope
    if hasattr(v, "detach") and hasattr(v, "cpu"):
        v = v.detach().cpu()
        if getattr(v, "ndim", 1) == 0:
            return _jsonable(float(v))
        return [_jsonable(x) for x in v.tolist()]
    if hasattr(v, "tolist") and hasattr(v, "shape"):
        if getattr(v, "ndim", 1) == 0:
            return _jsonable(v.item())
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)
