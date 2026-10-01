"""Configurations: the provisioned shape, the primordium, the interface widths.

SPEC: 03 §3.1
SPEC: 05 §5.1
SPEC: 05 §5.5
SPEC: 04 §4.1

One frozen dataclass describes a lineage's *provisioned* shape (what is allocated
at birth) together with the constants of the interface (`d_x`, `K`), the genome
(`d_g`, `r`) and the consolidator (`d_c`). The *active* shape is a property of
the switch pattern (`tempo/morphology.py`), not of the configuration.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

__all__ = ["TempoConfig", "TS_ADULT", "PILE_ADULT", "TINY"]


@dataclasses.dataclass(frozen=True)
class TempoConfig:
    """The provisioned shape of one lineage. SPEC: 03 §3.1."""

    d: int = 512                  # residual width (provisioned, never switched)
    L: int = 16                   # layers (provisioned)
    h: int = 8                    # heads per layer (provisioned)
    h_kv: int = 2                 # key-value heads of an attention layer (lineage constant)
    d_head: int = 64              # head width (lineage constant)
    F: int = 2048                 # FFN inner width (provisioned)
    slice: int = 256              # FFN units per slice (a switchable block)
    d_x: int = 256                # the interface width: perception and output
    K: int = 4                    # tokens per text span
    T: int = 512                  # tokens per window
    d_g: int = 512                # genome dimension (scale free: the same at every shape)
    r: int = 32                   # rank of the expression map
    d_c: int = 256                # consolidator and codec width
    chunk_size: int = 64          # spans per chunk of the delta rule
    attn_every: int = 4           # attention at `l mod attn_every == attn_every - 1` (3:1)
    vocab: int = 8192
    primordium_heads: int = 4     # heads `0..primordium_heads-1` are permanently on
    primordium_slices: int = 4    # slices `0..primordium_slices-1` are permanently on

    def __post_init__(self) -> None:
        if self.L % 4:
            raise ValueError("L must be a multiple of 4 (four quarter-stacks)")
        if self.F % self.slice:
            raise ValueError("F must be a multiple of the slice width")
        if self.h % self.h_kv:
            raise ValueError("h must be a multiple of h_kv")
        if self.T % self.K:
            raise ValueError("T must be a multiple of K")
        if not (0 < self.primordium_heads <= self.h and 0 < self.primordium_slices <= self.n_slices):
            raise ValueError("the primordium must be a non-empty subset of the provisioned blocks")
        if self.chunk_size < 1:
            raise ValueError("chunk_size >= 1")
        ops_all = {self.op(l) for l in range(self.L)}
        ops_prim = {self.op(l) for l in self.primordium_layers}
        if ops_prim != ops_all:
            raise ValueError("the primordium must contain every operator of the layout "
                             "(05 §5.5: births copy a layer with the same operator)")
        if self.quarter % self.attn_every:
            raise ValueError("a quarter-stack must be a whole number of operator periods, so that "
                             "layer l copies layer l - L/4 of the same operator (05 §5.3)")
        if self.h != 2 * self.primordium_heads or self.n_slices != 2 * self.primordium_slices:
            raise ValueError("the switchable head group and slice group are the primordium's doubles "
                             "(05 §5.1: heads ph..2ph-1 copy heads 0..ph-1)")

    # ------------------------------------------------------------------ derived
    @property
    def T_s(self) -> int:
        """Spans per window."""
        return self.T // self.K

    @property
    def n_slices(self) -> int:
        return self.F // self.slice

    @property
    def quarter(self) -> int:
        """Layers per quarter-stack. SPEC: 05 §5.1."""
        return self.L // 4

    def op(self, l: int) -> str:
        """`"attn"` or `"delta"` for layer `l`. SPEC: 03 §3.1."""
        if not 0 <= l < self.L:
            raise IndexError(l)
        return "attn" if l % self.attn_every == self.attn_every - 1 else "delta"

    def quarter_of(self, l: int) -> int:
        return l // self.quarter

    def layers_of_quarter(self, q: int) -> tuple[int, ...]:
        if not 0 <= q < 4:
            raise IndexError(q)
        return tuple(range(q * self.quarter, (q + 1) * self.quarter))

    @property
    def primordium_layers(self) -> tuple[int, ...]:
        """Quarter-stacks 0 and 2. SPEC: 05 §5.5."""
        return self.layers_of_quarter(0) + self.layers_of_quarter(2)

    @property
    def switchable_quarters(self) -> tuple[int, ...]:
        return (1, 3)

    @property
    def n_delta_layers(self) -> int:
        return sum(1 for l in range(self.L) if self.op(l) == "delta")

    @property
    def n_attn_layers(self) -> int:
        return self.L - self.n_delta_layers

    # ------------------------------------------------------------------ identity
    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def digest(self) -> str:
        return hashlib.blake2b(json.dumps(self.as_dict(), sort_keys=True).encode(),
                               digest_size=8).hexdigest()


TS_ADULT = TempoConfig()
"""SPEC: 03 §3.1 -- the TinyStories lineage."""

PILE_ADULT = TempoConfig(d=1024, L=32, h=16, h_kv=4, F=4096, d_x=512,
                         primordium_heads=8, primordium_slices=8)
"""SPEC: 03 §3.1 -- the Pile lineage: quarter-stacks of eight layers (two `d d d a` periods),
so the primordium keeps 3:1 and every newborn layer has a same-operator source at `l - 8`."""

TINY = TempoConfig(d=32, L=8, h=2, h_kv=1, d_head=16, F=64, slice=16, d_x=16, K=2, T=16,
                   d_g=24, r=4, d_c=16, chunk_size=4, vocab=64, attn_every=2,
                   primordium_heads=1, primordium_slices=2)
"""A CPU test shape: every mechanism runs in a second. Quarter-stacks of two layers
(`d a`), so the primordium `{0, 1, 4, 5}` holds both operators."""
