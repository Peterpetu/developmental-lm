"""`Arch`: a decoded architecture and the phenotype shapes it implies.

SPEC: 07 §7.1
SPEC: 07 §7.4

Separated from both `genome/arch.py` (which decodes one of these from
`Gamma_a`) and `pheno/model.py` (which builds a network from one) so that
neither has to import the other, and so that a hand-written architecture is
exactly as valid an object as a decoded one -- which is what makes the
`A-NOARCH` ablation and the fixed-architecture wave-1 runs the *same* code path
as the evolved ones.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

from ..config import (
    ROLE_DELTA_BETA, ROLE_EMB, ROLE_FFN_DOWN, ROLE_FFN_GATE, ROLE_FFN_UP,
    ROLE_K, ROLE_LOGIT_SCALE, ROLE_NORM_FFN, ROLE_NORM_FINAL, ROLE_NORM_MIX,
    ROLE_O, ROLE_PLAST_CHAN, ROLE_PLAST_TOK, ROLE_Q, ROLE_SINK, ROLE_V,
    N_RULE, PhenoConfig,
)

__all__ = ["Arch", "GLOBAL_ROLES"]

GLOBAL_ROLES = (ROLE_EMB, ROLE_NORM_FINAL, ROLE_PLAST_TOK, ROLE_LOGIT_SCALE)

# Roles a layer needs, by operator. Q/K/V/O are shared between the two (§04.2):
# a mutation in `Gamma_a` that flips a layer's operator then lands in a
# partially prepared genome rather than on untrained ground.
_ATTN = (ROLE_Q, ROLE_K, ROLE_V, ROLE_O, ROLE_SINK)
_DELTA = (ROLE_Q, ROLE_K, ROLE_V, ROLE_O, ROLE_DELTA_BETA)
_FFN = (ROLE_FFN_GATE, ROLE_FFN_UP, ROLE_FFN_DOWN)
_ALWAYS = (ROLE_NORM_MIX, ROLE_NORM_FFN, ROLE_PLAST_CHAN)


@dataclasses.dataclass(frozen=True)
class Arch:
    """One architecture. Every field is per layer except where noted."""

    ops: tuple[str, ...]                  # "attn" | "delta"
    n_h: tuple[int, ...]
    d_ff: tuple[int, ...]
    window: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.ops)
        for name in ("n_h", "d_ff", "window"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} entries, "
                                 f"ops has {n}")
        if any(o not in ("attn", "delta") for o in self.ops):
            raise ValueError(f"unknown operator in {self.ops}")

    @property
    def L(self) -> int:
        return len(self.ops)

    @staticmethod
    def uniform(cfg: PhenoConfig, ops: Optional[tuple[str, ...]] = None) -> "Arch":
        """The configuration's own shape at every layer.

        Alternating attention and delta mixers by default: a hybrid rather than
        either pure form, so the recurrent layers give a linear-time path and
        the attention layers an exact global one.
        """
        ops = ops or tuple("attn" if l % 2 else "delta" for l in range(cfg.L))
        L = len(ops)
        return Arch(ops, (cfg.n_h,) * L, (cfg.d_ff,) * L, (cfg.window,) * L)

    # ------------------------------------------------------------- shapes

    def role_shape(self, cfg: PhenoConfig, role: int, layer: int) -> tuple[int, ...]:
        """SPEC: 04 §4.2, with the per-layer widths of §07.1."""
        d, V = cfg.d_model, cfg.vocab
        if role in GLOBAL_ROLES:
            return {ROLE_EMB: (V, d), ROLE_NORM_FINAL: (d,),
                    ROLE_PLAST_TOK: (V, N_RULE), ROLE_LOGIT_SCALE: (1,)}[role]
        h = self.n_h[layer]
        d_qk, d_kv, d_ff = h * cfg.d_head, cfg.n_kv * cfg.d_head, self.d_ff[layer]
        return {
            ROLE_Q: (d_qk, d), ROLE_K: (d_kv, d), ROLE_V: (d_kv, d),
            ROLE_O: (d, d_qk), ROLE_DELTA_BETA: (2 * h, d),
            ROLE_FFN_GATE: (d_ff, d), ROLE_FFN_UP: (d_ff, d),
            ROLE_FFN_DOWN: (d, d_ff), ROLE_NORM_MIX: (d,), ROLE_NORM_FFN: (d,),
            ROLE_SINK: (h,), ROLE_PLAST_CHAN: (d, N_RULE),
        }[role]

    def roles_for(self, layer: int) -> tuple[int, ...]:
        base = _ATTN if self.ops[layer] == "attn" else _DELTA
        return base + _FFN + _ALWAYS

    def decode_keys(self) -> list[tuple[int, int]]:
        keys = [(r, l) for l in range(self.L) for r in self.roles_for(l)]
        return keys + [(r, -1) for r in GLOBAL_ROLES]

    def widest_axis(self, cfg: PhenoConfig) -> int:
        """The largest channel-like axis, for the innate code table (§02.2)."""
        return max([cfg.d_model, cfg.n_kv * cfg.d_head]
                   + [h * cfg.d_head for h in self.n_h] + list(self.d_ff))

    # --------------------------------------------------------- descriptors

    def n_decoded(self, cfg: PhenoConfig) -> int:
        return sum(math.prod(self.role_shape(cfg, r, l))
                   for r, l in self.decode_keys())

    def attn_fraction(self) -> float:
        """`B1`. SPEC: 07 §7.4."""
        return sum(o == "attn" for o in self.ops) / self.L

    def descriptors(self, cfg: PhenoConfig) -> tuple[float, float]:
        """`(B1, B2)`. SPEC: 07 §7.4 -- computed without running a lifetime."""
        return self.attn_fraction(), math.log10(self.n_decoded(cfg))

    def summary(self) -> str:
        return " ".join(f"{o[0]}{h}/{f}" for o, h, f
                        in zip(self.ops, self.n_h, self.d_ff))
