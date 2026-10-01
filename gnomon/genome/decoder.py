"""The decoder: one function that writes every weight in the network.

SPEC: 01 §1.3
SPEC: 03 §3.1
SPEC: 03 §3.2
SPEC: 03 §3.3
SPEC: 03 §3.4
SPEC: 03 §3.5
SPEC: 03 §3.6
SPEC: 03 §3.7

```
    u_row(i)    = W_row [ b_i ; z_i ]
    u_col(j)    = W_col [ b_j ; +1 ; z_j ]
    u_ctx(l,m)  = W_ctx [ r_l ; t_m ; y_l ]
    h1          = tanh( s * (u_row + u_col + u_ctx) + c1 )
    h2          = tanh( W_2 h1 + c2 )
    raw[i,j]    = w_out . h2 + o_0
    W_{l,m}     = raw / rms(raw) * sigma_{l,m} * s_target(l,m)
```

`Gamma_w` is every tensor in `Decoder` below and nothing else: independent of
`d_model`, `L` and `V`, which is the whole `H log N` claim.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..config import (
    GAIN_ROLES, ROLE_DELTA_BETA, ROLE_EMB, ROLE_FFN_DOWN, ROLE_FFN_GATE,
    ROLE_FFN_UP, ROLE_K, ROLE_LOGIT_SCALE, ROLE_NORM_FFN, ROLE_NORM_FINAL,
    ROLE_NORM_MIX, ROLE_O, ROLE_PLAST_CHAN, ROLE_PLAST_TOK, ROLE_Q, ROLE_SINK,
    ROLE_V, VECTOR_ROLES, GenomeShape, PhenoConfig,
)
from . import codes as C

__all__ = ["Decoder", "Soma", "s_target", "vector_offset", "AXIS_KIND",
           "decode_reference", "DEFAULT_GAIN"]

# SPEC: 03 §3.6. Measured, not chosen: see the sweeps recorded there.
DEFAULT_GAIN = 4.0
# SPEC: 04 §4.3. The innate cosine-logit temperature and the genome's room
# around it: `tau = exp(TAU_SPREAD * raw + log TAU_INNATE)`.
TAU_INNATE = 4.0
TAU_SPREAD = 0.3

# SPEC: 03 §3.1 -- what kind of entity each axis of each role indexes.
#
# `chan`  the residual stream. Shared by every layer, and the only channel-like
#         thing whose identity a lifetime may acquire (§01.3).
# `tok`   the vocabulary. Likewise acquired.
# `inner` a layer-internal unit: an attention head slot, an FFN hidden unit.
#         **Purely innate.** Giving these acquired identities would cost
#         `L * d_ff * d_z` = 393K reals of soma at M1, three times the rest of
#         `Z` put together, and would misplace the claim: the residual stream
#         and the vocabulary are the network's *interface*, which is what a
#         lifetime learns the meaning of; a layer's internal units are machinery
#         the genome builds, and their identities are the genome's business.
# `vec`   the "no column" argument of a vector decode (§02.6).
AXIS_KIND: dict[int, tuple[str, str]] = {
    ROLE_Q: ("inner", "chan"),
    ROLE_K: ("inner", "chan"),
    ROLE_V: ("inner", "chan"),
    ROLE_DELTA_BETA: ("inner", "chan"),
    ROLE_O: ("chan", "inner"),
    ROLE_FFN_GATE: ("inner", "chan"),
    ROLE_FFN_UP: ("inner", "chan"),
    ROLE_FFN_DOWN: ("chan", "inner"),
    ROLE_EMB: ("tok", "chan"),
    ROLE_NORM_MIX: ("chan", "vec"),
    ROLE_NORM_FFN: ("chan", "vec"),
    ROLE_NORM_FINAL: ("chan", "vec"),
    ROLE_SINK: ("inner", "vec"),
    # SPEC: 10 §10.3 -- the four rule coefficients get four distinct innate
    # identities from the shared inner table, so a decoded learning rule
    # needs no new genome tensor and no new code path.
    ROLE_PLAST_CHAN: ("chan", "inner"),
    ROLE_PLAST_TOK: ("tok", "inner"),
    ROLE_LOGIT_SCALE: ("inner", "vec"),
}


def s_target(role: int, shape_out: tuple[int, ...], cfg: PhenoConfig,
             L: Optional[int] = None) -> float:
    """The innate scale of a decoded object. SPEC: 03 §3.2.

    A pure function of the *shape*, so it costs no genome and supplies
    dimensional correctness independently of `Gamma`. The genome's `sigma_{l,m}`
    then supplies only the deviation from it, initialised at exactly 1.
    """
    if role in VECTOR_ROLES:
        # Not rms-normalised (§03.3): `raw` is already O(1) because `h2` is
        # bounded and `||w_out|| ~ 1`, so `s_target` sets only how far a decode
        # may sit from the identity transformation.
        if role == ROLE_LOGIT_SCALE:
            return TAU_SPREAD
        return 0.5 if role in GAIN_ROLES else 1.0
    if role == ROLE_EMB:
        # Xavier over `(V, d_model)` gives ~0.02 and a residual stream two orders
        # of magnitude below unit RMS. An embedding row should have unit RMS,
        # which for entrywise-iid rows means entrywise scale 1.
        return 1.0
    fan_out, fan_in = shape_out
    sc = math.sqrt(2.0 / (fan_in + fan_out))
    if role in (ROLE_O, ROLE_FFN_DOWN):
        # Residual-branch output projections, scaled by 1/sqrt(2L) so the
        # residual stream's variance does not grow with depth (GPT-2's rule).
        # `L` is the *decoded* depth when an architecture is supplied, not the
        # configuration's maximum: a genome that decodes 3 layers must be
        # scaled for 3, not for the configuration's 12.
        sc /= math.sqrt(2.0 * (cfg.L if L is None else L))
    return sc


def vector_offset(role: int) -> float:
    """Innate additive offset, so an uninformative genome decodes to identity."""
    if role == ROLE_LOGIT_SCALE:
        return math.log(TAU_INNATE)
    return 1.0 if role in GAIN_ROLES else 0.0


class Soma(nn.Module):
    """`Z = (z, zeta, y)` -- everything a lifetime may write. SPEC: 04 §4.5.

    Initialised at unit variance, not `1/d_z`. Under `N(0, 1/d_z)` the acquired
    block of `[b_i ; z_i]` would contribute `1/(n_c + d_z)` of the
    pre-activation variance against the innate block's `n_c/(n_c + d_z)`: the
    acquired half of an identity would be numerically invisible at birth, and
    §06's heritability would be measuring a network in which only the codes
    matter. Unit variance makes the halves comparable, which is what §01.3
    claims they are.
    """

    def __init__(self, cfg: PhenoConfig, shape: GenomeShape, *, seed: int):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.cfg, self.shape = cfg, shape
        self.z = nn.Parameter(torch.randn(cfg.d_model, shape.d_z, generator=g))
        self.zeta = nn.Parameter(torch.randn(cfg.vocab, shape.d_z, generator=g))
        self.y = nn.Parameter(torch.randn(cfg.L, shape.d_y, generator=g))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Decoder(nn.Module):
    """`Gamma_w`. SPEC: 03 §3.5 -- these tensors and nothing else."""

    def __init__(self, shape: GenomeShape, *, seed: int = 0):
        super().__init__()
        self.shape = shape
        d_h, d_z = shape.d_h, shape.d_z
        g = torch.Generator(device="cpu").manual_seed(seed)

        def normal(*sz, fan: int) -> nn.Parameter:
            return nn.Parameter(torch.randn(*sz, generator=g) / math.sqrt(fan))

        # SPEC: 03 §3.6 -- the initialisation table.
        self.W_row = normal(d_h, shape.n_c + d_z, fan=shape.n_c + d_z)
        self.W_col = normal(d_h, shape.n_c + 1 + d_z, fan=shape.n_c + 1 + d_z)
        self.W_tok = normal(d_h, shape.n_v + d_z, fan=shape.n_v + d_z)
        n_ctx = shape.n_L + shape.n_M + shape.d_y
        self.W_ctx = normal(d_h, n_ctx, fan=n_ctx)
        self.c1 = nn.Parameter(torch.zeros(d_h))
        # Seeded orthogonal init. `nn.init.orthogonal_` draws from the *global*
        # RNG, which would make `Decoder(shape, seed=k)` irreproducible -- the
        # gain sweep in §03.6 showed run-to-run drift before this was fixed.
        Q, R = torch.linalg.qr(torch.randn(d_h, d_h, generator=g))
        self.W_2 = nn.Parameter((Q * torch.sign(torch.diagonal(R))).contiguous())
        self.c2 = nn.Parameter(torch.zeros(d_h))
        self.w_out = normal(d_h, fan=d_h)
        self.o_0 = nn.Parameter(torch.zeros(()))
        self.w_sig = nn.Parameter(torch.zeros(d_h))
        self.sig_0 = nn.Parameter(torch.zeros(()))
        # SPEC: 03 §3.6 -- `s = exp(log_gain)` is the decoder's most consequential
        # scalar and its default is measured, not chosen. See §03.6's table.
        self.log_gain = nn.Parameter(torch.full((), math.log(DEFAULT_GAIN)))
        # The decoded depth in force, set by `prepare`. `None` means "use the
        # configuration's L". Never a parameter: it is a property of the
        # architecture being decoded, not of the genome.
        self._L: Optional[int] = None
        self._n_inner: Optional[int] = None
        # SPEC: 02 §2.9 -- the token frequency rank, when the run uses the
        # frequency-ordered token code. `None` keeps §2.4's seeded bijection.
        self._tok_rank: Optional[torch.Tensor] = None
        # SPEC: 03 §3.7 -- recompute the decode's interior during backward
        # rather than holding it. Not a parameter and not part of `Gamma`: a
        # memory/compute trade that changes no value. Turned off under
        # CUDA-graph capture, where a recomputation inside the captured region
        # is not something the graph can express.
        self.checkpoint_decode: bool = True

        assert self.n_params() == shape.n_params(), (
            f"|Gamma_w| = {self.n_params()} but GenomeShape says "
            f"{shape.n_params()}; §03.5's table and this module have diverged"
        )

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---------------------------------------------------------------- codes

    def codes(self, cfg: PhenoConfig, n_inner: Optional[int] = None) -> dict:
        """Innate code tables for one phenotype shape. Cached, derived, free.

        Built **eagerly**, never inside a traced region: the construction is
        integer bit arithmetic with a Python-level bounds check, which
        `torch.compile` cannot trace and which would break the CUDA graph the
        lifetime runner depends on. Once built, this is a dict lookup.
        """
        dev, dt = self.W_row.device, self.W_row.dtype
        # The prepared width wins over the configuration's default. Without
        # this, `decode_many` -- which does not know the architecture -- rederived
        # `n_inner` from `cfg` and silently rebuilt a table too small for a
        # decoded `d_ff`, surfacing as a `torch.cat` size mismatch three frames
        # away from the cause.
        n_inner = (n_inner or getattr(self, "_n_inner", None)
                   or max(cfg.d_ff, cfg.d_qk, cfg.d_kv, cfg.n_h, 1))
        key = (cfg.name, cfg.d_model, cfg.vocab, cfg.L, n_inner,
               str(dev), str(dt), self._tok_rank is not None)
        cache = getattr(self, "_codes", None)
        if cache is None or cache["key"] != key:
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "code tables built inside a traced region; call "
                    "Decoder.prepare(cfg) before compiling (SPEC: 02 §2.4)"
                )
            sh = self.shape
            cache = {
                "key": key,
                "chan": C.channel_codes(cfg.d_model, sh, dev).to(dt),
                "inner": C.channel_codes(n_inner, sh, dev).to(dt),
                "tok": C.token_codes(cfg.vocab, sh, dev,
                                     rank=self._tok_rank).to(dt),
                "layer": C.layer_codes(cfg.L, sh, dev).to(dt),
                "role": C.role_codes(sh, dev).to(dt),
                "vec_col": C.vector_column(sh, dev, dt),
            }
            object.__setattr__(self, "_codes", cache)
        return cache

    def set_token_rank(self, rank: Optional[torch.Tensor]) -> "Decoder":
        """Use §02.9's frequency-ordered token code. SPEC: 02 §2.9.

        Not part of `Gamma`: it is a coordinate system, like the choice of
        basis, shared by every genome in every generation.
        """
        self._tok_rank = rank
        object.__setattr__(self, "_codes", None)
        return self

    def prepare(self, cfg: PhenoConfig, n_inner: Optional[int] = None,
                L: Optional[int] = None) -> "Decoder":
        """Build the code tables now, so no traced region ever has to."""
        self._L, self._n_inner = L, n_inner
        self.codes(cfg, n_inner)
        return self

    def _axis(self, kind: str, n: int, soma: "Soma", cache: dict,
              *, column: bool) -> torch.Tensor:
        """Features for one axis: `(n, n_in)`, ready for `W_row`/`W_col`/`W_tok`.

        `inner` axes carry a zero acquired block rather than a borrowed one:
        borrowing `z[:n]` would give an FFN unit the acquired identity of a
        residual channel it has nothing to do with, and tiling the table (the
        first version of this module did) would give units `i` and `i + d_model`
        *identical* identities and therefore identical weight rows -- a
        degeneracy exactly like the rank collapse §03.1 rejects.
        """
        d_z = self.shape.d_z
        if kind == "tok":
            b, z = cache["tok"], soma.zeta
        elif kind == "chan":
            b, z = cache["chan"], soma.z
        elif kind == "inner":
            if n > cache["inner"].shape[0]:
                raise ValueError(
                    f"inner axis of width {n} exceeds the prepared code table "
                    f"({cache['inner'].shape[0]}); call "
                    f"Decoder.prepare(cfg, n_inner=arch.widest_axis(cfg))"
                )
            b = cache["inner"][:n]
            z = torch.zeros(n, d_z, dtype=b.dtype, device=b.device)
        else:
            raise ValueError(f"unknown axis kind {kind!r}")
        if kind in ("tok", "chan"):
            if n > b.shape[0]:
                raise ValueError(
                    f"axis of width {n} exceeds the {kind} table ({b.shape[0]})"
                )
            b, z = b[:n], z[:n]
        if column:
            b = C.column_codes(b)
        return torch.cat([b, z], dim=-1)

    # ------------------------------------------------------------- the trunk

    def _u_ctx(self, layer: int, role: int, soma: Soma, cache: dict) -> torch.Tensor:
        """`(d_h,)`. `layer < 0` means a once-per-model role; depth 0 is used."""
        ll = 0 if layer < 0 else layer
        feat = torch.cat([cache["layer"][ll], cache["role"][role], soma.y[ll]], -1)
        return F.linear(feat, self.W_ctx)

    def _readout(self, u_row: torch.Tensor, u_col: torch.Tensor,
                 u_ctx=0.0) -> torch.Tensor:
        """`(R, d_h) x (C, d_h) -> (R, C)`. SPEC: 03 §3.1.

        The broadcast *is* the factorisation: `u_row`, `u_col` and `u_ctx` each
        cost `O(n d_h)`, and only this expression touches `R x C x d_h`.
        """
        pre = u_row.unsqueeze(1) + u_col.unsqueeze(0) + u_ctx
        h1 = torch.tanh(torch.exp(self.log_gain) * pre + self.c1)
        h2 = torch.tanh(F.linear(h1, self.W_2, self.c2))
        return h2 @ self.w_out + self.o_0

    def _readout_chunked(self, u_row, u_col, u_ctx, max_elems: int) -> torch.Tensor:
        """SPEC: 03 §3.7 -- bound peak activation memory at `max_elems` entries.

        Chunking bounds the *width* of one readout; **gradient checkpointing**
        bounds how many of them are alive at once, and without it the chunking
        does not help at all: every chunk's `h1` and `h2` stay in the autograd
        graph until backward, so the peak is the whole decode regardless of how
        it was sliced.

        The arithmetic is why M1 did not fit. `h1` and `h2` are each
        `d_out x d_in x d_h`, which is `d_h = 32` times the weight count, so the
        retained activation for a 49.9M-weight decode is
        `2 x 32 x 49.9M = 3.19e9` floats -- **12.8 GB in fp32**, measured as a
        17.5 GB peak on a 16 GB card at batch 8, which then spilled and ran at
        4.4 s/step. §3.7 always said the decode "is gradient-checkpointed"; it
        was not, and this implements it.

        The cost is one extra forward through the trunk during backward, which
        is the "about 3% with recomputation" §3.7 already budgeted for. It is
        exact rather than approximate: the decode contains no RNG and no
        data-dependent control flow, so the recomputation reproduces the
        original activations bit-for-bit.
        """
        R, Cn, d_h = u_row.shape[0], u_col.shape[0], self.shape.d_h
        rows = R if R * Cn * d_h <= max_elems else max(
            1, max_elems // max(1, Cn * d_h))

        def run(a: int) -> torch.Tensor:
            blk = u_row[a:a + rows]
            if not (self.checkpoint_decode and torch.is_grad_enabled()):
                return self._readout(blk, u_col, u_ctx)
            # `use_reentrant=False`: the reentrant path cannot handle a chunk
            # whose inputs do not all require grad, which is exactly the `inner`
            # axis (§03's AXIS_KIND -- a purely innate identity).
            return torch.utils.checkpoint.checkpoint(
                self._readout, blk, u_col, u_ctx, use_reentrant=False)

        if rows >= R:
            return run(0)
        return torch.cat([run(a) for a in range(0, R, rows)], dim=0)

    def _sigma(self, u_ctx: torch.Tensor) -> torch.Tensor:
        """`sigma_{l,m}`, the genome's deviation from the innate scale. §03.2."""
        return torch.exp(
            torch.clamp(self.w_sig @ torch.tanh(u_ctx) + self.sig_0, -3.0, 3.0)
        )

    # -------------------------------------------------------------- decoding

    def decode(self, role: int, layer: int, soma: Soma, cfg: PhenoConfig,
               *, shape_out: Optional[tuple[int, ...]] = None,
               chunk_elems: Optional[int] = None) -> torch.Tensor:
        """One role in one layer, shaped as `cfg.role_shape(role)`.

        SPEC: 03 §3.1-§3.3. `layer = -1` for once-per-model roles.
        """
        cache = self.codes(cfg)
        shape_out = cfg.role_shape(role) if shape_out is None else shape_out
        row_kind, col_kind = AXIS_KIND[role]
        u_ctx = self._u_ctx(layer, role, soma, cache)

        row_feat = self._axis(row_kind, shape_out[0], soma, cache, column=False)
        u_row = F.linear(row_feat, self.W_tok if row_kind == "tok" else self.W_row)

        if col_kind == "vec":
            u_col = F.linear(cache["vec_col"],
                             self.W_col[:, : self.shape.n_c + 1])
            raw = self._readout(u_row, u_col, u_ctx).squeeze(-1)
            sig = self._sigma(u_ctx)
            return (raw * sig * s_target(role, shape_out, cfg, self._L)
                    + vector_offset(role))

        col_feat = self._axis(col_kind, shape_out[1], soma, cache, column=True)
        u_col = F.linear(col_feat, self.W_col)
        raw = self._readout_chunked(u_row, u_col, u_ctx,
                                    chunk_elems or cfg.decode_chunk_elems)
        rms = torch.sqrt(raw.pow(2).mean() + 1e-12)
        return (raw / rms * self._sigma(u_ctx)
                * s_target(role, shape_out, cfg, self._L))

    def decode_many(self, requests: list[tuple[int, int]], soma: Soma,
                    cfg: PhenoConfig, *, shapes: Optional[dict] = None,
                    chunk_elems: Optional[int] = None) -> dict:
        """Decode many `(role, layer)` requests in a few grouped passes.

        SPEC: 03 §3.1, §3.7. Numerically identical to calling `decode` on each
        request -- §03.8's D1 checks exactly that -- but 48 separate decodes at
        M-pico cost about 0.5 ms *each* in kernel-launch and Python overhead
        alone, including a `(128,)` vector decode that does almost no
        arithmetic. Grouping them makes the decode compute-bound instead, which
        is what brings a 60-generation evolutionary run inside its budget.

        The grouping is exact because `u_ctx(l, m)` is constant across a
        request's rows: it can be folded into that request's `u_row` block, and
        blocks sharing a column axis then differ only in their rows, so they
        concatenate. The per-matrix `rms` normalisation is applied after the
        split, on each block separately, exactly as `decode` does.
        """
        cache = self.codes(cfg)

        def shp_of(role: int, layer: int) -> tuple[int, ...]:
            return (cfg.role_shape(role) if shapes is None
                    else shapes[(role, layer)])

        groups: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for role, layer in requests:
            _, col_kind = AXIS_KIND[role]
            width = 1 if col_kind == "vec" else shp_of(role, layer)[1]
            groups.setdefault((col_kind, width), []).append((role, layer))

        out: dict = {}
        for (col_kind, width), reqs in groups.items():
            if col_kind == "vec":
                u_col = F.linear(cache["vec_col"],
                                 self.W_col[:, : self.shape.n_c + 1])
            else:
                feat = self._axis(col_kind, width, soma, cache, column=True)
                u_col = F.linear(feat, self.W_col)

            blocks, sizes = [], []
            for role, layer in reqs:
                row_kind = AXIS_KIND[role][0]
                n_row = shp_of(role, layer)[0]
                rf = self._axis(row_kind, n_row, soma, cache, column=False)
                u_row = F.linear(rf, self.W_tok if row_kind == "tok" else self.W_row)
                # Folding u_ctx into the row block is what makes the grouping
                # exact: it is constant over the block's rows.
                blocks.append(u_row + self._u_ctx(layer, role, soma, cache))
                sizes.append(n_row)

            raw = self._readout_chunked(
                torch.cat(blocks, 0), u_col, 0.0,
                chunk_elems or cfg.decode_chunk_elems)

            at = 0
            for (role, layer), n_row in zip(reqs, sizes):
                blk = raw[at:at + n_row]
                at += n_row
                shp = shp_of(role, layer)
                sig = self._sigma(self._u_ctx(layer, role, soma, cache))
                sc = s_target(role, shp, cfg, self._L)
                if col_kind == "vec":
                    out[(role, layer)] = (blk.squeeze(-1) * sig * sc
                                          + vector_offset(role))
                else:
                    rms = torch.sqrt(blk.pow(2).mean() + 1e-12)
                    out[(role, layer)] = blk / rms * sig * sc
        return out

    def decode_rule(self, layer: int, role: int, soma: Soma, cfg: PhenoConfig,
                    *, shape_out: Optional[tuple[int, ...]] = None
                    ) -> torch.Tensor:
        """`(n, N_RULE)` plasticity-rule coefficients. SPEC: 10 §10.2, §10.3.

        A matrix decode under a role that already exists, so the learning rule
        adds no genome tensor and no code path: it is a second readout of the
        same representation of identity that produced the weights, which is the
        coupling §03.4 argues for, now extended from *what the weights are* to
        *how the identities change*.
        """
        assert role in (ROLE_PLAST_CHAN, ROLE_PLAST_TOK)
        from ..lifetime.plasticity import coefficients_from

        raw = self.decode(role, layer, soma, cfg, shape_out=shape_out)
        return coefficients_from(raw)


def decode_reference(dec: Decoder, role: int, layer: int, soma: Soma,
                     cfg: PhenoConfig,
                     shape_out: Optional[tuple[int, ...]] = None) -> torch.Tensor:
    """A naive entry-by-entry decode. SPEC: 03 §3.8 D1 -- the oracle.

    Written from §03.1's equations rather than from `Decoder`, so that agreement
    between the two is evidence about the equations and not about a shared
    implementation. `O(R*C)` Python iterations: test-only.
    """
    cache = dec.codes(cfg)
    shape_out = cfg.role_shape(role) if shape_out is None else shape_out
    row_kind, col_kind = AXIS_KIND[role]
    u_ctx = dec._u_ctx(layer, role, soma, cache)
    s = torch.exp(dec.log_gain)

    rf = dec._axis(row_kind, shape_out[0], soma, cache, column=False)
    rows = rf @ (dec.W_tok if row_kind == "tok" else dec.W_row).T
    if col_kind == "vec":
        cols = cache["vec_col"] @ dec.W_col[:, : dec.shape.n_c + 1].T
    else:
        cols = dec._axis(col_kind, shape_out[1], soma, cache, column=True) @ dec.W_col.T

    out = torch.empty(rows.shape[0], cols.shape[0], dtype=rows.dtype,
                      device=rows.device)
    for i in range(rows.shape[0]):
        for j in range(cols.shape[0]):
            h1 = torch.tanh(s * (rows[i] + cols[j] + u_ctx) + dec.c1)
            h2 = torch.tanh(dec.W_2 @ h1 + dec.c2)
            out[i, j] = dec.w_out @ h2 + dec.o_0

    sig = dec._sigma(u_ctx)
    sc = s_target(role, shape_out, cfg, dec._L)
    if col_kind == "vec":
        return out.squeeze(-1) * sig * sc + vector_offset(role)
    return out / torch.sqrt(out.pow(2).mean() + 1e-12) * sig * sc
