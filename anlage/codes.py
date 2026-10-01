"""Innate identity codes: pure functions of index and configuration.

SPEC: 02 §2.1
SPEC: 02 §2.2
SPEC: 02 §2.3
SPEC: 02 §2.4
SPEC: 02 §2.5
SPEC: 02 §2.7

Nothing here is stored or learned; the module holds no `Parameter`. The three
constructions GNOMON measured to be right are reused from
`gnomon/genome/codes.py` (the seeded bijection, Gray over relative depth, Gray
over frequency rank with stable ties) and adapted to ANLAGE's widths, the layer
flag bit of §02.4 and the axis table of §02.1.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from gnomon.genome.codes import bits_to_pm1, seeded_bijection, frequency_rank  # noqa: F401
from .config import (
    AX_GATE, AX_HEAD, AX_INNER, AX_KV, AX_ONE, AX_RES, AX_TAP, AX_TOK,
    GenomeShape, N_ROLES,
)

__all__ = [
    "bits_to_pm1", "seeded_bijection", "frequency_rank", "channel_codes",
    "token_codes", "layer_codes", "global_layer_code", "role_codes", "axis_codes",
    "bit_balance", "CodeTables",
]

CHANNEL_AXES = (AX_RES, AX_INNER, AX_HEAD, AX_KV, AX_GATE, AX_TAP, AX_ONE)


def channel_codes(n: int, shape: GenomeShape, device="cpu") -> torch.Tensor:
    """`(n, n_c)` in `{-1,+1}`: the first `n` codes of the seeded bijection.

    SPEC: 02 §2.2. Nesting (P1): unit `i` has the same code at every `n >= i+1`.
    """
    if n > shape.d_max:
        raise ValueError(
            f"cannot address {n} identities with n_c={shape.n_c} bits; n_c must "
            f"grow to {max(1, (n - 1).bit_length())} (§02.1)")
    perm = seeded_bijection(shape.n_c, shape.seed_codes)[:n]
    return bits_to_pm1(perm.to(device), shape.n_c)


def token_codes(vocab: int, shape: GenomeShape, rank: torch.Tensor,
                device="cpu") -> torch.Tensor:
    """`(vocab, n_v)`: Gray code over frequency rank. SPEC: 02 §2.3.

    `rank[v] = 0` for the most frequent type (`frequency_rank`). Relative rank
    is what makes the code nested across vocabulary sizes in the sense §02.3
    states: the most frequent type has the same code at every `V`.
    """
    if vocab > shape.v_max:
        raise ValueError(f"vocab {vocab} exceeds 2^n_v={shape.v_max} (§02.1)")
    if rank.numel() != vocab:
        raise ValueError(f"rank has {rank.numel()} entries, vocab is {vocab}")
    hi = (1 << shape.n_v) - 1
    r = rank.to(torch.float64)
    q = torch.round(r * hi / max(vocab - 1, 1)).to(torch.int64)
    return bits_to_pm1((q ^ (q >> 1)).to(device), shape.n_v)


def layer_codes(L: int, shape: GenomeShape, device="cpu") -> torch.Tensor:
    """`(L, n_L + 1)`: Gray over relative depth, flag `+1`. SPEC: 02 §2.4."""
    if L > shape.L_max:
        raise ValueError(f"L={L} exceeds 2^n_L={shape.L_max} (§02.1)")
    hi = (1 << shape.n_L) - 1
    if L == 1:
        q = torch.zeros(1, dtype=torch.int64, device=device)
    else:
        l = torch.arange(L, dtype=torch.float64, device=device)
        q = torch.round(l * hi / (L - 1)).to(torch.int64)
    gray = bits_to_pm1(q ^ (q >> 1), shape.n_L)
    flag = torch.ones(L, 1, dtype=gray.dtype, device=device)
    return torch.cat([gray, flag], dim=-1)


def global_layer_code(shape: GenomeShape, device="cpu") -> torch.Tensor:
    """`(1, n_L + 1)`: zero code bits, flag `-1`. SPEC: 02 §2.4."""
    out = torch.zeros(1, shape.n_layer_code, dtype=torch.float32, device=device)
    out[0, -1] = -1.0
    return out


def role_codes(shape: GenomeShape, device="cpu") -> torch.Tensor:
    """`(N_ROLES, n_M)` plain binary. SPEC: 02 §2.5."""
    idx = torch.arange(N_ROLES, dtype=torch.int64, device=device)
    return bits_to_pm1(idx, shape.n_M)


def bit_balance(codes: torch.Tensor) -> torch.Tensor:
    return codes.mean(dim=0)


class CodeTables:
    """The code tables one phenotype needs, built once per (stage, device).

    SPEC: 02 §2.1. `axis_codes(axis, n)` returns the first `n` codes of the
    axis's table, so a wider stage's table is a prefix-extension of a narrower
    stage's (P1) on every channel axis.
    """

    def __init__(self, shape: GenomeShape, *, vocab: int, rank: torch.Tensor,
                 L: int, device="cpu"):
        self.shape = shape
        self.device = torch.device(device)
        self.vocab = vocab
        self.L = L
        self._chan = channel_codes(shape.d_max, shape, device)     # full table
        self._tok = token_codes(vocab, shape, rank, device)
        self._layer = layer_codes(L, shape, device)
        self._global = global_layer_code(shape, device)
        self._role = role_codes(shape, device)
        self._one = torch.ones(1, shape.n_c, dtype=torch.float32, device=device)

    def axis_codes(self, axis: str, n: int) -> torch.Tensor:
        """`(n, width)` codes for the first `n` units of `axis`."""
        if axis == AX_TOK:
            if n != self.vocab:
                raise ValueError(f"token axis has {self.vocab} units, asked {n}")
            return self._tok
        if axis == AX_ONE:
            if n != 1:
                raise ValueError("the ONE axis has a single unit")
            return self._one
        if axis in CHANNEL_AXES:
            if n > self.shape.d_max:
                raise ValueError(
                    f"{axis}: {n} units exceed 2^n_c={self.shape.d_max} (§02.1)")
            return self._chan[:n]
        raise KeyError(axis)

    def layer_code(self, l: int) -> torch.Tensor:
        """`(1, n_L + 1)`; `l = -1` is the global pseudo-layer."""
        if l == -1:
            return self._global
        return self._layer[l:l + 1]

    def role_code(self, m: int) -> torch.Tensor:
        return self._role[m:m + 1]
