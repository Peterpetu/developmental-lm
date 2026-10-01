"""Configuration objects: the genome shape and the stage configurations.

SPEC: 02 §2.1
SPEC: 03 §3.6
SPEC: 04 §4.2
SPEC: 04 §4.5

Two kinds of object with different lifetimes, kept apart on purpose:

``GenomeShape``  the shape of the genome. Code widths, feature width, heads,
                 kernel width, latent widths. **Independent of every phenotype
                 dimension** -- that independence is the `H log N` claim of
                 §03.6 and it is asserted by `assert_scale_free`, not assumed.
``StageConfig``  the shape of one developmental stage of one lineage. Changing
                 it must not change `|Gamma|`.
``Lineage``      the ordered stages, the vocabulary, the context length and the
                 budget fractions of §06.5.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Sequence

# ----------------------------------------------------------------------------
# SPEC: 04 §4.2 -- the 20 roles, fixed for the life of the project. The order is
# the role's binary code (§02.5) and must never be changed.
ROLE_Q = 0
ROLE_K = 1
ROLE_V = 2
ROLE_O = 3
ROLE_AB = 4
ROLE_OGATE = 5
ROLE_CONV_Q = 6
ROLE_CONV_K = 7
ROLE_CONV_V = 8
ROLE_AGATE = 9
ROLE_VE = 10
ROLE_GATE = 11
ROLE_UP = 12
ROLE_DOWN = 13
ROLE_EMB = 14
ROLE_HEAD_OUT = 15
ROLE_NORM_MIX = 16
ROLE_NORM_FFN = 17
ROLE_NORM_FINAL = 18
ROLE_LOGIT_SCALE = 19

N_ROLES = 20

ROLE_NAMES = {
    ROLE_Q: "Q", ROLE_K: "K", ROLE_V: "V", ROLE_O: "O", ROLE_AB: "AB",
    ROLE_OGATE: "OGATE", ROLE_CONV_Q: "CONV_Q", ROLE_CONV_K: "CONV_K",
    ROLE_CONV_V: "CONV_V", ROLE_AGATE: "AGATE", ROLE_VE: "VE", ROLE_GATE: "GATE",
    ROLE_UP: "UP", ROLE_DOWN: "DOWN", ROLE_EMB: "EMB", ROLE_HEAD_OUT: "HEAD_OUT",
    ROLE_NORM_MIX: "NORM_MIX", ROLE_NORM_FFN: "NORM_FFN",
    ROLE_NORM_FINAL: "NORM_FINAL", ROLE_LOGIT_SCALE: "LOGIT_SCALE",
}
ROLE_BY_NAME = {v: k for k, v in ROLE_NAMES.items()}

MATRIX_ROLES = tuple(range(ROLE_Q, ROLE_HEAD_OUT + 1))          # 0..15
VECTOR_ROLES = (ROLE_NORM_MIX, ROLE_NORM_FFN, ROLE_NORM_FINAL, ROLE_LOGIT_SCALE)
GAIN_ROLES = frozenset({ROLE_NORM_MIX, ROLE_NORM_FFN, ROLE_NORM_FINAL})
GLOBAL_ROLES = (ROLE_EMB, ROLE_HEAD_OUT, ROLE_NORM_FINAL, ROLE_LOGIT_SCALE)
CONV_ROLES = (ROLE_CONV_Q, ROLE_CONV_K, ROLE_CONV_V)
N_TAPS = 4

# Axes of §02.1. Each role has a (row axis, column axis); vector roles have a row
# axis only. Axis names are the ones the codes module keys its tables by.
AX_RES, AX_INNER, AX_HEAD, AX_KV, AX_GATE, AX_TAP, AX_TOK, AX_ONE = (
    "RES", "INNER", "HEAD", "KV", "GATE", "TAP", "TOK", "ONE")

ROLE_AXES: dict[int, tuple[str, str | None]] = {
    ROLE_Q: (AX_HEAD, AX_RES), ROLE_K: (AX_KV, AX_RES), ROLE_V: (AX_KV, AX_RES),
    ROLE_O: (AX_RES, AX_HEAD), ROLE_AB: (AX_GATE, AX_RES),
    ROLE_OGATE: (AX_HEAD, AX_RES), ROLE_CONV_Q: (AX_HEAD, AX_TAP),
    ROLE_CONV_K: (AX_KV, AX_TAP), ROLE_CONV_V: (AX_KV, AX_TAP),
    ROLE_AGATE: (AX_HEAD, AX_RES), ROLE_VE: (AX_TOK, AX_KV),
    ROLE_GATE: (AX_INNER, AX_RES), ROLE_UP: (AX_INNER, AX_RES),
    ROLE_DOWN: (AX_RES, AX_INNER), ROLE_EMB: (AX_TOK, AX_RES),
    ROLE_HEAD_OUT: (AX_TOK, AX_RES), ROLE_NORM_MIX: (AX_RES, None),
    ROLE_NORM_FFN: (AX_RES, None), ROLE_NORM_FINAL: (AX_RES, None),
    ROLE_LOGIT_SCALE: (AX_ONE, None),
}

# Which roles a layer of each operator carries (§04.2's "present in" column).
_EVERY = (ROLE_Q, ROLE_K, ROLE_V, ROLE_O, ROLE_GATE, ROLE_UP, ROLE_DOWN,
          ROLE_NORM_MIX, ROLE_NORM_FFN)
_DELTA_ONLY = (ROLE_AB, ROLE_OGATE, ROLE_CONV_Q, ROLE_CONV_K, ROLE_CONV_V)
_ATTN_ONLY = (ROLE_AGATE,)


def roles_for(op: str, *, use_ve: bool) -> tuple[int, ...]:
    """The roles a layer of operator `op` carries. SPEC: 04 §4.2."""
    if op == "delta":
        return _EVERY + _DELTA_ONLY
    if op == "attn":
        return _EVERY + _ATTN_ONLY + ((ROLE_VE,) if use_ve else ())
    raise ValueError(f"unknown operator {op!r}")


# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class GenomeShape:
    """SPEC: 02 §2.1, SPEC: 03 §3.6. Every field is a property of the genome."""

    n_c: int = 13          # channel address bits   -> 8192 channel-like units
    n_v: int = 15          # token address bits     -> 32768 vocabulary items
    n_L: int = 6           # relative-depth bits    -> 64 depths (+1 flag bit)
    n_M: int = 5           # role bits              -> 32 roles (20 used)
    d_z: int = 16          # unit latent width
    d_y: int = 8           # layer latent width
    d_h: int = 256         # feature-map hidden width
    H: int = 4             # kernel heads
    d_g: int = 64          # kernel width per head
    d_sigma: int = 16      # per-matrix scale map width
    seed_codes: int = 20260902

    def __post_init__(self) -> None:
        if 2 ** self.n_M < N_ROLES:
            raise ValueError(f"n_M={self.n_M} cannot address {N_ROLES} roles")
        for name in ("n_c", "n_v", "n_L", "n_M", "d_h", "H", "d_g", "d_sigma", "d_y"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.d_z < 0:
            raise ValueError("d_z must be >= 0 (d_z = 0 is the innate-only arm)")

    # -- widths --------------------------------------------------------------
    @property
    def d_max(self) -> int:
        return 1 << self.n_c

    @property
    def v_max(self) -> int:
        return 1 << self.n_v

    @property
    def L_max(self) -> int:
        return 1 << self.n_L

    @property
    def n_layer_code(self) -> int:
        """`n_L` Gray bits plus the global flag (§02.4)."""
        return self.n_L + 1

    @property
    def in_chan(self) -> int:
        """Feature-map input width on the channel path: (3.1)."""
        return self.n_c + self.d_z + self.n_layer_code + self.d_y + 1

    @property
    def in_tok(self) -> int:
        return self.n_v + self.d_z + self.n_layer_code + self.d_y + 1

    # -- the count of §03.6 -------------------------------------------------
    def n_params(self) -> int:
        """`|Gamma|`, recomputed from the shape. SPEC: 03 §3.6.

        Independent of every phenotype dimension by construction; `G-SPECNUM`
        asserts the table row by row.
        """
        d_h, H, d_g = self.d_h, self.H, self.d_g
        n_mat, n_vec = len(MATRIX_ROLES), len(VECTOR_ROLES)
        return (
            d_h * self.in_chan + d_h                 # W_1, c_1
            + d_h * self.in_tok + d_h                # W_1^tok, c_1^tok
            + d_h * d_h + d_h                        # W_1b, c_1b
            + H * d_g * d_h + H * d_g                # W_2, c_2
            + n_mat * H * d_g * d_g                  # O
            + n_mat                                  # kappa_hat
            + n_vec * d_g                            # o
            + self.d_sigma * (self.n_layer_code + self.n_M)   # A_sigma
            + self.d_sigma + self.d_sigma            # b_sigma, w_sigma
        )

    def key(self) -> str:
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class StageConfig:
    """One developmental stage. SPEC: 04 §4.5."""

    stage: int
    d_model: int
    L: int
    n_h: int
    n_kv_attn: int
    d_ff: int
    d_head: int = 64
    chunk_size: int = 64
    use_ve: bool = True
    #: SPEC: 04 §4.3 -- the embedding read scale of (4.1). A *lineage* constant,
    #: `sqrt(d_0)` of the primordium, not `sqrt(d_s)`: a scale that moved with the
    #: width would change the ratio of the embedding to every branch write at a
    #: growth event and break Theorem G1 (found by G1 in tests_anlage).
    emb_scale: float = 0.0
    #: SPEC: 06 §6.2 step 3 -- a grown stage carries its operator layout
    #: explicitly (existing layers keep their operators and their order); the
    #: lineage's canonical stages use the `l mod 4 == 3` rule.
    ops_override: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.emb_scale <= 0.0:
            object.__setattr__(self, "emb_scale", float(self.d_model) ** 0.5)
        if self.ops_override is not None:
            if len(self.ops_override) != self.L:
                raise ValueError("ops_override must have one entry per layer")
            if any(o not in ("attn", "delta") for o in self.ops_override):
                raise ValueError("unknown operator in ops_override")
        if self.n_h % self.n_kv_attn:
            raise ValueError("n_h must be a multiple of n_kv_attn")
        if self.L < 1 or self.d_model < 1 or self.d_ff < 1:
            raise ValueError("dimensions must be >= 1")

    @property
    def d_qk(self) -> int:
        return self.n_h * self.d_head

    def op(self, l: int) -> str:
        """SPEC: 04 §4.4 -- attention at `l mod 4 == 3`, delta elsewhere (canonical)."""
        if l < 0 or l >= self.L:
            raise IndexError(f"layer {l} outside 0..{self.L - 1}")
        if self.ops_override is not None:
            return self.ops_override[l]
        return "attn" if l % 4 == 3 else "delta"

    def canonical_ops(self) -> tuple[str, ...]:
        return tuple("attn" if l % 4 == 3 else "delta" for l in range(self.L))

    def ops(self) -> tuple[str, ...]:
        return tuple(self.op(l) for l in range(self.L))

    def n_kv(self, l: int) -> int:
        """SPEC: 04 §4.2 -- full heads for delta layers, `n_kv_attn` for attention."""
        return self.n_h if self.op(l) == "delta" else self.n_kv_attn

    def d_kv(self, l: int) -> int:
        return self.n_kv(l) * self.d_head

    def roles(self, l: int) -> tuple[int, ...]:
        return roles_for(self.op(l), use_ve=self.use_ve)

    def role_shape(self, role: int, l: int, vocab: int, *, tied: bool = True
                   ) -> tuple[int, ...]:
        """SPEC: 04 §4.2. `l = -1` for global roles."""
        d, F, h, dk = self.d_model, self.d_ff, self.n_h, self.d_head
        if role in GLOBAL_ROLES:
            if role == ROLE_HEAD_OUT and tied:
                raise KeyError("HEAD_OUT does not exist while the head is tied")
            return {ROLE_EMB: (vocab, d), ROLE_HEAD_OUT: (vocab, d),
                    ROLE_NORM_FINAL: (d,), ROLE_LOGIT_SCALE: (1,)}[role]
        n_kv = self.n_kv(l)
        return {
            ROLE_Q: (h * dk, d), ROLE_K: (n_kv * dk, d), ROLE_V: (n_kv * dk, d),
            ROLE_O: (d, h * dk), ROLE_AB: (2 * h, d), ROLE_OGATE: (h * dk, d),
            ROLE_CONV_Q: (h * dk, N_TAPS), ROLE_CONV_K: (n_kv * dk, N_TAPS),
            ROLE_CONV_V: (n_kv * dk, N_TAPS), ROLE_AGATE: (h * dk, d),
            ROLE_VE: (vocab, n_kv * dk), ROLE_GATE: (F, d), ROLE_UP: (F, d),
            ROLE_DOWN: (d, F), ROLE_NORM_MIX: (d,), ROLE_NORM_FFN: (d,),
        }[role]

    def axis_size(self, axis: str, l: int, vocab: int) -> int:
        """Units on an axis at this stage. SPEC: 02 §2.1."""
        if axis == AX_RES:
            return self.d_model
        if axis == AX_INNER:
            return self.d_ff
        if axis == AX_HEAD:
            return self.n_h * self.d_head
        if axis == AX_KV:
            return self.d_kv(l)
        if axis == AX_GATE:
            return 2 * self.n_h
        if axis == AX_TAP:
            return N_TAPS
        if axis == AX_TOK:
            return vocab
        if axis == AX_ONE:
            return 1
        raise KeyError(axis)

    def keys(self, vocab: int, *, tied: bool = True) -> list[tuple[int, int]]:
        """Every `(role, layer)` the phenotype holds at this stage; `-1` global."""
        out = [(r, l) for l in range(self.L) for r in self.roles(l)]
        out += [(ROLE_EMB, -1), (ROLE_NORM_FINAL, -1), (ROLE_LOGIT_SCALE, -1)]
        if not tied:
            out.append((ROLE_HEAD_OUT, -1))
        return out

    def n_params(self, vocab: int, *, tied: bool = True) -> int:
        return sum(math.prod(self.role_shape(r, l, vocab, tied=tied))
                   for r, l in self.keys(vocab, tied=tied))

    def n_params_nonemb(self, vocab: int, *, tied: bool = True) -> int:
        """Non-embedding count, the `N` of the `6 N T` FLOP budget (§06.5)."""
        return self.n_params(vocab, tied=tied) - vocab * self.d_model

    def widest_axis(self, vocab: int) -> int:
        return max(self.d_model, self.d_ff, self.n_h * self.d_head, 2 * self.n_h)

    def summary(self) -> str:
        return (f"s{self.stage}:d{self.d_model}/L{self.L}/h{self.n_h}/"
                f"kv{self.n_kv_attn}/F{self.d_ff}")


@dataclasses.dataclass(frozen=True)
class Lineage:
    """SPEC: 04 §4.5, SPEC: 06 §6.5."""

    name: str
    vocab: int
    T: int
    stages: tuple[StageConfig, ...]
    data_root: str
    #: fractions of `C_total` per stage (§06.5); must sum to 1.
    budget_fractions: tuple[float, ...] = (0.15, 0.25, 0.60)

    def __post_init__(self) -> None:
        if len(self.budget_fractions) != len(self.stages):
            raise ValueError("one budget fraction per stage")
        if abs(sum(self.budget_fractions) - 1.0) > 1e-9:
            raise ValueError("budget fractions must sum to 1")
        for a, b in zip(self.stages, self.stages[1:]):
            for name in ("d_model", "L", "n_h", "d_ff"):
                if getattr(b, name) < getattr(a, name):
                    raise ValueError(f"{name} must be non-decreasing across stages")
            if b.n_kv_attn != a.n_kv_attn:
                raise ValueError("n_kv_attn is fixed per lineage (§04.2)")
            if b.n_kv_attn * b.d_head != a.n_kv_attn * a.d_head:
                raise ValueError("KV width is fixed per lineage")
            if abs(b.emb_scale - a.emb_scale) > 1e-9:
                raise ValueError("emb_scale is a lineage constant (§04.3)")

    def flops(self, s: int, tokens: int) -> float:
        """`6 N_s T` with `N_s` the stage's non-embedding count. SPEC: 06 §6.5."""
        return 6.0 * self.stages[s].n_params_nonemb(self.vocab) * tokens

    def stage_tokens(self, C_total: float, *, n_stages: "int | None" = None) -> tuple[int, ...]:
        """`T_s = floor(f_s C_total / (6 N_s))`. SPEC: 06 §6.5.

        With `n_stages` the fractions are renormalised over the first `n_stages`
        stages (a lineage cut short for a screen keeps the same proportions).
        """
        n = len(self.stages) if n_stages is None else int(n_stages)
        if not 1 <= n <= len(self.stages):
            raise ValueError(f"n_stages={n} outside 1..{len(self.stages)}")
        fr = self.budget_fractions[:n]
        tot = sum(fr)
        return tuple(int((f / tot) * C_total // (6 * st.n_params_nonemb(self.vocab)))
                     for f, st in zip(fr, self.stages[:n]))

    def sibling_tokens(self, C_total: float, K: int, *, n_stages: "int | None" = None) -> int:
        """`C_sib = f_0 C_total / K` at stage 0, in tokens. SPEC: 07 §7.5."""
        if K < 1:
            raise ValueError("K must be >= 1")
        return self.stage_tokens(C_total, n_stages=n_stages)[0] // K

    @property
    def adult(self) -> StageConfig:
        return self.stages[-1]

    def stage(self, s: int) -> StageConfig:
        return self.stages[s]


def assert_scale_free(shape: GenomeShape, lineages: Sequence[Lineage]) -> None:
    """SPEC: 02 §2.1, SPEC: 03 §3.6 (D4). `|Gamma|` must not move when the phenotype does."""
    n = shape.n_params()
    for lin in lineages:
        for st in lin.stages:
            widest = st.widest_axis(lin.vocab)
            if widest > shape.d_max:
                raise ValueError(
                    f"{lin.name} {st.summary()}: widest channel-like axis {widest} "
                    f"exceeds 2^n_c={shape.d_max}; n_c must grow to "
                    f"{math.ceil(math.log2(widest))} (§02.1)")
            if lin.vocab > shape.v_max:
                raise ValueError(f"{lin.name}: vocab {lin.vocab} exceeds 2^n_v={shape.v_max}")
            if st.L > shape.L_max:
                raise ValueError(f"{lin.name}: L={st.L} exceeds 2^n_L={shape.L_max}")
    assert shape.n_params() == n, "genome size moved while inspecting phenotypes"


# ----------------------------------------------------------------------------
# SPEC: 04 §4.5 -- the two lineages.
TINYSTORIES = Lineage(
    name="tinystories", vocab=8192, T=512, data_root="data/ts_8192",
    stages=(
        StageConfig(0, d_model=256, L=8, n_h=4, n_kv_attn=2, d_ff=1024, emb_scale=16.0),
        StageConfig(1, d_model=384, L=12, n_h=6, n_kv_attn=2, d_ff=1536, emb_scale=16.0),
        StageConfig(2, d_model=512, L=16, n_h=8, n_kv_attn=2, d_ff=2048, emb_scale=16.0),
    ),
)

PILE = Lineage(
    name="pile", vocab=16384, T=512, data_root="data/pile_16384",
    stages=(
        StageConfig(0, d_model=512, L=12, n_h=8, n_kv_attn=4, d_ff=2048, emb_scale=512 ** 0.5),
        StageConfig(1, d_model=768, L=18, n_h=12, n_kv_attn=4, d_ff=3072, emb_scale=512 ** 0.5),
        StageConfig(2, d_model=1024, L=24, n_h=16, n_kv_attn=4, d_ff=4096, emb_scale=512 ** 0.5),
    ),
)

LINEAGES = {l.name: l for l in (TINYSTORIES, PILE)}


def get_lineage(name: str) -> Lineage:
    if name not in LINEAGES:
        raise KeyError(f"unknown lineage {name!r}; have {sorted(LINEAGES)}")
    return LINEAGES[name]
