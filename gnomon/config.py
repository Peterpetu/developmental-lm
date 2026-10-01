"""Configuration objects for GNOMON.

SPEC: 02 §2.2
SPEC: 03 §3.5
SPEC: 04 §4.4
SPEC: 05 §5.4

Two objects, deliberately kept apart because they have different lifetimes:

``GenomeShape``   the shape of the genome. Code widths, decoder width, identity
                  widths. **Independent of `d_model`, `L` and `V`** -- that
                  independence is the whole `H log N` claim (§03.5), so it is
                  asserted rather than assumed.
``PhenoConfig``   the shape of the phenotype a genome is decoded into. Changing
                  it must not change `|Gamma|`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Literal

# SPEC: 04 §4.2 -- the 16 matrix roles, fixed for the life of the project.
ROLE_Q = 0
ROLE_K = 1
ROLE_V = 2
ROLE_O = 3
ROLE_FFN_GATE = 4
ROLE_FFN_UP = 5
ROLE_FFN_DOWN = 6
ROLE_EMB = 7
ROLE_NORM_MIX = 8
ROLE_NORM_FFN = 9
ROLE_NORM_FINAL = 10
ROLE_SINK = 11
ROLE_DELTA_BETA = 12
ROLE_PLAST_CHAN = 13
ROLE_PLAST_TOK = 14
ROLE_LOGIT_SCALE = 15

N_ROLES = 16

ROLE_NAMES = {
    ROLE_Q: "Q", ROLE_K: "K", ROLE_V: "V", ROLE_O: "O",
    ROLE_FFN_GATE: "FFN_GATE", ROLE_FFN_UP: "FFN_UP", ROLE_FFN_DOWN: "FFN_DOWN",
    ROLE_EMB: "EMB", ROLE_NORM_MIX: "NORM_MIX", ROLE_NORM_FFN: "NORM_FFN",
    ROLE_NORM_FINAL: "NORM_FINAL", ROLE_SINK: "SINK",
    ROLE_DELTA_BETA: "DELTA_BETA", ROLE_PLAST_CHAN: "PLAST_CHAN",
    ROLE_PLAST_TOK: "PLAST_TOK", ROLE_LOGIT_SCALE: "LOGIT_SCALE",
}

# SPEC: 10 §10.3 -- the width of the decoded plasticity rule, (A, B, C, D).
N_RULE = 4

# Roles whose decode is a vector rather than a matrix (§03.3). The two
# plasticity roles left this set at v1.0: they now decode `(n, N_RULE)` rule
# coefficients rather than a single gain (§10.3), which is a deliberate,
# recorded, versioned change -- genomes from before it decode differently and
# are marked incomparable rather than silently reused.
VECTOR_ROLES = frozenset({
    ROLE_NORM_MIX, ROLE_NORM_FFN, ROLE_NORM_FINAL, ROLE_SINK,
    ROLE_LOGIT_SCALE,
})
# Vector roles that multiply an activation and must decode to ~1 when the genome
# is uninformative, so an untrained decode is the identity transformation.
GAIN_ROLES = frozenset({ROLE_NORM_MIX, ROLE_NORM_FFN, ROLE_NORM_FINAL})
# Roles indexed by the vocabulary rather than by the residual channel.
TOKEN_ROLES = frozenset({ROLE_EMB, ROLE_PLAST_TOK})

Operator = Literal["attn", "delta"]


@dataclasses.dataclass(frozen=True)
class GenomeShape:
    """SPEC: 02 §2.2, SPEC: 03 §3.5.

    Every field here is a property of the *genome*. None of them may depend on
    the phenotype being decoded; `assert_scale_free` enforces that.
    """

    n_c: int = 12          # channel address bits  -> 4096 identities
    n_v: int = 14          # token address bits    -> 16384 vocabulary items
    n_L: int = 5           # relative-depth bits   -> 32 depth levels
    n_M: int = 4           # matrix-role bits      -> 16 roles
    d_h: int = 32          # decoder hidden width
    d_z: int = 16          # acquired identity width (the dial, §06.5)
    d_y: int = 8           # acquired layer-state width
    seed_codes: int = 20260829   # fixes pi_c and pi_v (§02.4); never mutated

    def __post_init__(self) -> None:
        if 2 ** self.n_M < N_ROLES:
            raise ValueError(f"n_M={self.n_M} cannot address {N_ROLES} roles")
        for name in ("n_c", "n_v", "n_L", "n_M", "d_h", "d_y"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.d_z < 0:
            raise ValueError("d_z must be >= 0 (d_z = 0 is the fully-innate arm)")

    @property
    def d_max(self) -> int:
        return 1 << self.n_c

    @property
    def v_max(self) -> int:
        return 1 << self.n_v

    def n_params(self) -> int:
        """`|Gamma_w|` (§03.5). Independent of `d_model`, `L`, `V` by design."""
        d_h, d_z = self.d_h, self.d_z
        return (
            d_h * (self.n_c + d_z)              # W_row
            + d_h * (self.n_c + 1 + d_z)        # W_col  (+1 = the vector flag, §02.6)
            + d_h * (self.n_v + d_z)            # W_tok
            + d_h * (self.n_L + self.n_M + self.d_y)   # W_ctx
            + d_h                               # c1
            + d_h * d_h + d_h                   # W_2, c2
            + d_h + 1                           # w_out, o_0
            + d_h + 1                           # w_sig, sig_0
            + 1                                 # log_gain
        )

    def key(self) -> str:
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class PhenoConfig:
    """SPEC: 04 §4.4. The shape a genome is decoded into."""

    name: str
    d_model: int
    L: int
    n_h: int
    n_kv: int
    d_head: int
    d_ff: int
    vocab: int
    T: int
    N_life: int
    chunk_size: int = 64            # delta-rule chunk (SOMA §05.2)
    window: int = 128               # sliding-window attention span
    batch: int = 32
    lr: float = 3e-3
    #: SPEC: 10 §10.6 -- where this configuration's `lr` came from.
    #:
    #: §10.6 records, as a defect in every comparison the project had made,
    #: that `lr` was chosen from a sweep at 2,000 steps and then used at 300 and
    #: 600. A learning rate is only meaningful together with the horizon and
    #: batch it was measured at, so the two travel together and the string ends
    #: up in every run manifest. A value that has *not* been measured here says
    #: so, in those words, rather than being indistinguishable from one that
    #: has -- which is precisely how §10.6's defect survived being written down.
    lr_provenance: str = "NOT MEASURED at this configuration"
    c_eta: float = 1.5              # SPEC: 05 §5.4
    lam_ratio: float = 0.04         # SPEC: 05 §5.5, the paper's 500/12500
    decode_chunk_elems: int = 1 << 24   # SPEC: 03 §3.7

    def __post_init__(self) -> None:
        if self.n_h % self.n_kv:
            raise ValueError("n_h must be a multiple of n_kv (GQA)")
        if self.T % self.chunk_size:
            raise ValueError("T must be a multiple of chunk_size")
        if self.L < 1:
            raise ValueError("L must be >= 1")

    @property
    def lam(self) -> float:
        return max(self.lam_ratio * self.N_life, 1e-9)

    @property
    def d_qk(self) -> int:
        return self.n_h * self.d_head

    @property
    def d_kv(self) -> int:
        return self.n_kv * self.d_head

    def role_shape(self, role: int) -> tuple[int, ...]:
        """SPEC: 04 §4.2. The decoded shape of each role."""
        d, V = self.d_model, self.vocab
        return {
            ROLE_Q: (self.d_qk, d),
            ROLE_K: (self.d_kv, d),
            ROLE_V: (self.d_kv, d),
            ROLE_O: (d, self.d_qk),
            ROLE_FFN_GATE: (self.d_ff, d),
            ROLE_FFN_UP: (self.d_ff, d),
            ROLE_FFN_DOWN: (d, self.d_ff),
            ROLE_EMB: (V, d),
            ROLE_NORM_MIX: (d,),
            ROLE_NORM_FFN: (d,),
            ROLE_NORM_FINAL: (d,),
            ROLE_SINK: (self.n_h,),
            # 2*n_h: alpha (decay) and beta (write strength) packed into one
            # role, keeping the table at 16 and the two gates of one head
            # adjacent in the decoder's row-identity space (§04.1).
            ROLE_DELTA_BETA: (2 * self.n_h, d),
            ROLE_PLAST_CHAN: (d, N_RULE),
            ROLE_PLAST_TOK: (V, N_RULE),
            ROLE_LOGIT_SCALE: (1,),
        }[role]

    def n_decoded(self) -> int:
        """Decoded reals under the configuration's own uniform architecture.

        Delegates to `Arch`, which knows that an attention layer never decodes
        `DELTA_BETA` and a delta layer never decodes `SINK`. Counting every
        per-layer role at every layer -- as this method first did -- overstated
        M-nano by 6,930 reals.
        """
        from .pheno.plan import Arch
        return Arch.uniform(self).n_decoded(self)

    def n_soma(self, shape: GenomeShape) -> int:
        """`|Z|` (§04.5)."""
        return (self.d_model + self.vocab) * shape.d_z + self.L * shape.d_y


def assert_scale_free(shape: GenomeShape, cfgs: list[PhenoConfig]) -> None:
    """SPEC: 03 §3.5 (D8). `|Gamma_w|` must not move when the phenotype does.

    Stated as a runtime assertion rather than only a test because it is the
    project's central claim: a change that made the genome depend on `d_model`
    would silently turn GNOMON into a hypernetwork, which is exactly the defect
    the PNAS paper distinguishes itself from and exactly the one this project
    already made once (docs/gnomon/00 §0.4).
    """
    n = shape.n_params()
    for cfg in cfgs:
        # Every channel-like axis shares one code table: the residual stream and
        # the widest layer-internal axis (§03 AXIS_KIND). `d_ff` is normally the
        # binding one, which is why it is checked and not only `d_model`.
        widest = max(cfg.d_model, cfg.d_ff, cfg.d_qk, cfg.d_kv, cfg.n_h)
        if widest > shape.d_max:
            raise ValueError(
                f"{cfg.name}: widest channel-like axis {widest} exceeds "
                f"2^n_c={shape.d_max}; n_c must grow to "
                f"{math.ceil(math.log2(widest))} (+{2 * shape.d_h} genome reals) "
                f"-- this is where H log N bites"
            )
        if cfg.vocab > shape.v_max:
            raise ValueError(
                f"{cfg.name}: vocab={cfg.vocab} exceeds 2^n_v={shape.v_max}; "
                f"n_v must grow to {math.ceil(math.log2(cfg.vocab))} "
                f"(+{shape.d_h} genome reals)"
            )
        if cfg.L > (1 << shape.n_L):
            raise ValueError(f"{cfg.name}: L={cfg.L} exceeds 2^n_L={1 << shape.n_L}")
    assert shape.n_params() == n, "genome size moved while inspecting phenotypes"


# SPEC: 10 §10.6 -- each `lr` is measured at the configuration's own `N_life`
# and batch, by `scripts/gnomon_lrsweep.py`, and the sweep is on disk. Where the
# top values are statistically indistinguishable the smallest is taken: the
# conservative end of a region the data cannot separate degrades more gracefully
# when the horizon or the batch moves.
M_PICO = PhenoConfig(
    name="m-pico", d_model=128, L=4, n_h=4, n_kv=2, d_head=32, d_ff=512,
    vocab=4096, T=128, N_life=600, window=128, batch=32, lr=0.06,
    lr_provenance=(
        "measured: runs/gnomon/lrsweep-m-pico-n600-b32.json, N_life=600, "
        "batch=32, rule=adam, token_code=freq, 3 genomes x 2 seeds paired. "
        "Best 0.06 at CE 5.3181; {0.06, 0.09, 0.12, 0.18} indistinguishable by "
        "BCa; the curve turns inside the grid (0.03 -> 0.06 -> 0.09 gives "
        "5.4016 -> 5.3181 -> 5.3503). The previous 3e-3 is off the bottom of "
        "the grid, worse than 0.015 which is +0.2690 [+0.2100, +0.3266]."
    ),
)
M_NANO = PhenoConfig(
    name="m-nano", d_model=192, L=6, n_h=6, n_kv=2, d_head=32, d_ff=768,
    vocab=4096, T=128, N_life=2000, window=128, batch=32, lr=3e-3,
    lr_provenance=(
        "NOT MEASURED at this configuration -- the sweep at N_life=2000 has not "
        "been read in. §06.5's FRONTIER runs here, so running it at this value "
        "would repeat §10.6's defect at the very run that fixes the operating "
        "point. Run: python scripts/gnomon_lrsweep.py --config m-nano "
        "--steps 2000 --batch 32"
    ),
)
M1 = PhenoConfig(
    name="m1", d_model=512, L=12, n_h=8, n_kv=2, d_head=64, d_ff=2048,
    vocab=8192, T=512, N_life=20000, window=256, batch=24, lr=2e-3,
    lr_provenance=(
        "NOT MEASURED at this configuration. A 9-point sweep at N_life=20000 is "
        "54 full M1 lifetimes, which is many GPU-days and larger than the M1 "
        "run itself; §09.1 budgets 8 h for one. Whatever is used here is an "
        "extrapolation from M-pico and M-nano and must be reported as one."
    ),
)

CONFIGS = {c.name: c for c in (M_PICO, M_NANO, M1)}


def get_config(name: str) -> PhenoConfig:
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; have {sorted(CONFIGS)}")
    return CONFIGS[name]
