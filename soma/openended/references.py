"""The two references: `theta_m0` (frozen, better) and `theta_lag` (EMA, worse).

SPEC: 06 §6.1
SPEC: 06 §6.1.1
SPEC: 06 §6.1.2

**[NORMATIVE — v1 correction]** Spec v1 used one object, an EMA "lagged self",
for three roles. Two of those roles need **opposite polarity**, so one object
cannot serve both.

| Role | Needs a reference that knows... | Object |
|---|---|---|
| Token selection (§6.2) | **more** than the model | `theta_m0`, the frozen M0 baseline |
| Selection case weights (§04.3) | **less** than the model | `theta_lag`, the EMA lagged self |
| Open-endedness observer (§6.4) | a *fixed past* model | the frozen checkpoint ring |

An EMA lagged self gives `L_theta <= L_lag` almost everywhere, so Rho-1-style
excess loss `max(0, L_theta - L_lag)` is identically zero.

**[NORMATIVE — the `L_ref` cache is bound to its packing]** `L_ref` is
context-conditional, so a cached value is valid only under the exact packing
that produced it. The sidecar records
`(tokenizer_hash, corpus_hash, m0_checkpoint_id, T_ctx, packing_scheme,
doc_boundary_policy)` and any mismatch is **refused at load**, exactly as §2.3
refuses a germline hash mismatch (audit A-27).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

__all__ = [
    "RefCacheBinding",
    "RefCacheMismatch",
    "ReferenceCache",
    "LaggedSelf",
]


class RefCacheMismatch(RuntimeError):
    """The `L_ref` sidecar does not match the current packing (§06.1.1)."""


@dataclass(frozen=True)
class RefCacheBinding:
    """Everything a cached `L_ref` value depends on.

    SPEC: 06 §6.1.1

    Two consequences v2 left implicit, both honoured here: attention is masked
    at **document boundaries**, so a token's conditioning depends only on its own
    document and the cached value does not depend on what was packed beside it;
    and the cache is keyed by `(doc_id, position_in_doc)` rather than by position
    in a shuffled stream — otherwise §07.3's data-order jitter would silently
    invalidate every cached value while every recorded hash still matched.
    """

    tokenizer_hash: str
    corpus_hash: str
    m0_checkpoint_id: str
    T_ctx: int
    packing_scheme: str = "doc_keyed"
    doc_boundary_policy: str = "masked"

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def check(self, other: "RefCacheBinding") -> None:
        if self != other:
            diffs = [
                f"{k}: cached {getattr(other, k)!r} vs current {getattr(self, k)!r}"
                for k in asdict(self)
                if getattr(self, k) != getattr(other, k)
            ]
            raise RefCacheMismatch(
                "L_ref cache binding mismatch; the cache is refused at load "
                "(§06.1.1):\n  " + "\n  ".join(diffs)
            )


class ReferenceCache:
    """Precomputed per-token `L_ref` from the frozen M0 checkpoint.

    SPEC: 06 §6.1.1

    **[NORMATIVE — precomputation]** Because `theta_m0` is frozen and the
    natural-text corpus is fixed, `L_ref(x_i)` is **precomputed once over the
    entire tokenized corpus** and stored as `float16` beside the token stream
    (2 bytes/token). Training-time cost is **zero**.

    Spec v1 claimed the reference cost "under 3% of step time" while §6.2 and
    §04.3 required a reference forward for **every token of every batch at every
    step** — an extra full forward, ~`+25-33%` of step time, an order of
    magnitude above the claim, propagating into every wall-clock number in
    §07-§09. Precomputation removes the cost entirely rather than restating it.

    Stored memory-mapped: at M1 the sidecar is `2 B x 8e8 = 1.6 GB`, which is
    read by index alongside the token stream and is never resident in full.
    """

    def __init__(self, path: str | os.PathLike, binding: RefCacheBinding):
        self.path = Path(path)
        self.binding = binding
        self._data: Optional[np.memmap] = None

    # ------------------------------------------------------------------

    @property
    def meta_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".meta.json")

    def save(self, values: np.ndarray) -> None:
        """Write the sidecar and its binding."""
        arr = np.asarray(values, dtype=np.float16)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        arr.tofile(self.path)
        self.meta_path.write_text(
            json.dumps({"binding": asdict(self.binding), "n": int(arr.size)}, indent=2),
            encoding="utf-8",
        )

    def load(self) -> np.memmap:
        """Load, **refusing on any binding mismatch**."""
        if not self.path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"no L_ref sidecar at {self.path}")
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        cached = RefCacheBinding(**meta["binding"])
        self.binding.check(cached)
        self._data = np.memmap(self.path, dtype=np.float16, mode="r")
        if self._data.size != int(meta["n"]):
            raise RefCacheMismatch(
                f"L_ref sidecar length {self._data.size} != recorded {meta['n']}"
            )
        return self._data

    def close(self) -> None:
        """Release the memmap handle.

        Windows holds an exclusive lock on a mapped file, so a cache that is
        never closed cannot be deleted or replaced — which matters because the
        sidecar is regenerated whenever the binding changes (§06.1.1), and a
        stale locked file would make regeneration fail with a permission error
        rather than a clear mismatch.
        """
        if self._data is not None:
            self._data._mmap.close()  # type: ignore[attr-defined]
            self._data = None

    def __enter__(self) -> "ReferenceCache":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def lookup(self, flat_index: np.ndarray) -> torch.Tensor:
        """`L_ref` at `(doc_id, position_in_doc)` flattened indices.

        Returned as `float32`: §05.9 requires token-selection excess losses to be
        accumulated in `float32`, since a `bfloat16` difference of two `O(1)`
        losses is dominated by rounding late in training and would rank noise.
        """
        if self._data is None:
            self.load()
        assert self._data is not None
        return torch.from_numpy(
            np.asarray(self._data[flat_index], dtype=np.float32)
        )


class LaggedSelf:
    """`theta_lag`: an EMA of the model, updated every `T_ref` steps.

    SPEC: 06 §6.1.2

    ```
    theta_lag <- kappa_lag * theta_lag + (1 - kappa_lag) * theta   every T_ref steps  (6.1)
    kappa_lag  = 0.5 ^ ( T_ref / h_lag )                                              (6.2)
    ```

    `kappa_lag` is stated **per update**, not per step, and derived from a target
    half-life `h_lag = 0.05 * N_step`. Spec v1 wrote a per-step EMA indexed by
    `n` in its equation and "applied every 32 steps" in its prose; at
    `kappa = 0.999` applied once per 32 steps the effective half-life is ~22k
    steps, i.e. `theta_lag` would have remained approximately the initialisation
    for a whole M1 lifetime.

    **[NORMATIVE — the forward cost is budgeted]** §04.3's `w_t` requires
    `L_lag(t)`, i.e. a forward pass of `theta_lag`, which **cannot** be
    precomputed because `theta_lag` changes. The cost is bounded by restricting
    evaluation to the reservoir: `L_lag` is computed **only on the `q_cases`
    cases the buffer retains** (§04.6), never on every token of the batch. At M1
    that is `q_cases = 6554` against `B_tok = 65536` — a forward on 10% of the
    batch, `2ND` against the step's `6ND`, so **+3.3%** of step time
    (audit A-26). `theta_lag` is therefore **GPU-resident**, not CPU: it is
    evaluated, not merely stored.
    """

    def __init__(self, model: torch.nn.Module, kappa: float, T_ref: int):
        if not 0.0 < kappa < 1.0:
            raise ValueError(f"kappa_lag must lie in (0,1); got {kappa}")
        self.kappa = float(kappa)
        self.T_ref = int(T_ref)
        self.n_updates = 0
        self._shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().to(torch.bfloat16)
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }
        self._non_float: Dict[str, torch.Tensor] = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if not v.is_floating_point()
        }
        self._dirty = True
        """The shadow has changed since the last `load_into`.

        `theta_lag` is updated every `T_ref` steps (6.1) but `L_lag` is needed
        every step, so a naive implementation reloads a full `state_dict` -- a
        copy of every parameter -- 32 times more often than the shadow actually
        moves. At M1 that is 24 M parameters copied per step for no reason, and
        it dominates step time. The flag makes the reload happen exactly when
        the shadow has moved.
        """

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """(6.1), applied once per `T_ref` steps by the caller."""
        sd = model.state_dict()
        for k, shadow in self._shadow.items():
            cur = sd[k]
            shadow.mul_(self.kappa).add_(cur.detach().to(shadow.dtype), alpha=1.0 - self.kappa)
        for k in self._non_float:
            self._non_float[k].copy_(sd[k].detach())
        self.n_updates += 1
        self._dirty = True

    @torch.no_grad()
    def load_into(self, model: torch.nn.Module, *, force: bool = False) -> None:
        """Write the shadow into a scratch model for the `L_lag` forward.

        A no-op unless the shadow has moved since the last call, which is at
        most once per `T_ref` steps (6.1). `force=True` reloads regardless, for
        the round-trip test.
        """
        if not (self._dirty or force):
            return
        sd = {}
        target = model.state_dict()
        for k, v in self._shadow.items():
            sd[k] = v.to(target[k].dtype)
        for k, v in self._non_float.items():
            sd[k] = v
        model.load_state_dict(sd, strict=True)
        self._dirty = False

    def memory_bytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in self._shadow.values()) + sum(
            v.numel() * v.element_size() for v in self._non_float.values()
        )

    @staticmethod
    def kappa_from_half_life(T_ref: int, h_lag: float) -> float:
        """(6.2): `0.5 ** (T_ref / h_lag)`."""
        if h_lag <= 0:
            raise ValueError("h_lag must be positive")
        return 0.5 ** (T_ref / h_lag)
