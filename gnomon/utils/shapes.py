# VENDORED from `soma/utils/shapes.py` at commit `7d29cc1` (SOMA v3.1).
#
# SPEC: 04 §4.1 -- the numerical cores are already pure functions of
# activations and weights, so they are vendored verbatim below this header
# and `soma/` can be archived without breaking GNOMON. Their SOMA oracle
# tests are vendored with them and must pass unchanged.
"""Shape assertions at module boundaries.

SOMA-SPEC: 00 §0.4
SOMA-SPEC: 05 §5.0

`00 §0.4` and `05 §5.0` both require every module boundary to assert the shapes
of its inputs and outputs against the declared dimensions. This module supplies
the assertion so the requirement can be met uniformly and cheaply.

The checks are always on. A broadcast that silently produces the wrong shape is
the single most common way a tensor program computes a plausible wrong answer,
and the cost of a tuple comparison is negligible beside a matmul.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch

__all__ = ["check_shape", "check_finite", "check_dtype", "ShapeError"]

Dim = Union[int, str, None]


class ShapeError(AssertionError):
    """Raised when a tensor's shape does not match its declaration."""


def check_shape(
    t: torch.Tensor,
    spec: Sequence[Dim],
    *,
    name: str = "tensor",
    bindings: Optional[dict] = None,
) -> Tuple[int, ...]:
    """Assert `t.shape` matches `spec`, returning the realised shape.

    `spec` entries may be:

    * an ``int``   -- the dimension must equal it exactly;
    * a ``str``    -- a named dimension, bound on first use and checked
                      thereafter (within one ``bindings`` dict);
    * ``None``     -- the dimension is unconstrained.

    Named dimensions make relational constraints checkable, which is what the
    plain per-axis assertion misses: ``check_shape(q, ["B", "T", "d"])`` followed
    by ``check_shape(k, ["B", "T", "d"], bindings=b)`` catches a mismatch between
    two tensors that are each individually well shaped.
    """
    if not isinstance(t, torch.Tensor):
        raise ShapeError(f"{name}: expected a Tensor, got {type(t).__name__}")
    shape = tuple(t.shape)
    if len(shape) != len(spec):
        raise ShapeError(
            f"{name}: expected {len(spec)} dims {tuple(spec)}, got {len(shape)} {shape}"
        )
    for i, (got, want) in enumerate(zip(shape, spec)):
        if want is None:
            continue
        if isinstance(want, int):
            if got != want:
                raise ShapeError(
                    f"{name}: dim {i} is {got}, expected {want} (full shape {shape}, "
                    f"spec {tuple(spec)})"
                )
        elif isinstance(want, str):
            if bindings is None:
                raise ShapeError(
                    f"{name}: named dim {want!r} used without a bindings dict"
                )
            prev = bindings.get(want)
            if prev is None:
                bindings[want] = got
            elif prev != got:
                raise ShapeError(
                    f"{name}: dim {i} named {want!r} is {got}, but was bound to {prev} "
                    f"earlier (full shape {shape})"
                )
        else:
            raise TypeError(f"bad shape spec element {want!r}")
    return shape


# SOMA-SPEC: 05 §5.1. `check_finite` forces a device->host sync, which is illegal
# inside a CUDA-graph capture and would abort it with an opaque "operation
# failed due to a previous error during capture". The lifetime runner disables
# the checks for the duration of a capture and restores them afterwards; they
# remain on everywhere else, including every test and every eager run, which is
# where a NaN would actually be diagnosed.
FINITE_CHECKS = True


def set_finite_checks(enabled: bool) -> bool:
    """Toggle `check_finite`; returns the previous setting."""
    global FINITE_CHECKS
    was, FINITE_CHECKS = FINITE_CHECKS, bool(enabled)
    return was


def check_finite(t: torch.Tensor, *, name: str = "tensor") -> torch.Tensor:
    """Assert no NaN or Inf.

    Used at the boundaries the specification identifies as numerically delicate:
    the UT triangular solves (§05.2.5), the log-space decay accumulation
    (§05.2.3), and the loss (M1 criterion 1, §07.4.3). It is *not* applied to
    every tensor, because a `.isfinite().all()` forces a device sync.
    """
    if not FINITE_CHECKS:
        return t
    if not torch.isfinite(t).all():
        n_nan = int(torch.isnan(t).sum())
        n_inf = int(torch.isinf(t).sum())
        raise FloatingPointError(
            f"{name}: {n_nan} NaN and {n_inf} Inf entries in a tensor of "
            f"{t.numel()} elements (shape {tuple(t.shape)})"
        )
    return t


def check_dtype(
    t: torch.Tensor, dtype: torch.dtype | Sequence[torch.dtype], *, name: str = "tensor"
) -> torch.Tensor:
    """Assert dtype, for the places §05.9 mandates a specific precision."""
    allowed = (dtype,) if isinstance(dtype, torch.dtype) else tuple(dtype)
    if t.dtype not in allowed:
        raise ShapeError(
            f"{name}: dtype is {t.dtype}, expected one of {allowed} "
            f"(§05.9 mandates the precision at this boundary)"
        )
    return t
