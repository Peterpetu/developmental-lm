"""The genome: a neural kernel over identity space, and its decode `D`.

SPEC: 03 §3.1
SPEC: 03 §3.2
SPEC: 03 §3.3
SPEC: 03 §3.4
SPEC: 03 §3.5
SPEC: 03 §3.6
SPEC: 03 §3.7
SPEC: 03 §3.8

```
    xi_{a,l,side}  =  [ b_a ; z_a ; r_l ; y_l ; side ]
    f(xi)          =  W_2 GELU( W_1b GELU( W_1 xi + c_1 ) + c_1b ) + c_2         in R^{H x d_g}     (3.1)
    raw_{l,m}[i,j] =  sum_h  g_m( u_{i,l}^{(h)T} O_m^{(h)} v_{j,l}^{(h)} ),   g_m(x) = x + kappa_m x^3   (3.2)-(3.3)
    W_hat_{l,m}    =  raw / rms(raw) * s_target(m, shape) * sigma_{l,m}                              (3.4)
    log sigma_{l,m}=  w_sigma . tanh( A_sigma [ r_l ; t_m ] + b_sigma )                              (3.5)
    v_hat_{l,m}[i] =  u_{i,l}^{(1)} . o_m + offset(m)                                                (3.6)
```

The decode is one batched GEMM per matrix and head plus an elementwise map;
it runs at birth, at growth, inside `REGRESS` and inside `MERGE`, and never
inside a training step (§03.8, `G-NOFREE-INVERSE`).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnomon.utils.shapes import check_shape
from .codes import CodeTables
from .config import (
    AX_KV, AX_ONE, AX_RES, AX_TAP, AX_TOK, CONV_ROLES, GAIN_ROLES, MATRIX_ROLES,
    ROLE_AXES, ROLE_EMB, ROLE_HEAD_OUT, ROLE_LOGIT_SCALE, ROLE_VE, VECTOR_ROLES,
    GenomeShape, StageConfig,
)
from .latents import LatentBank

__all__ = ["Genome", "s_target", "offset", "DecodeContext"]

MAT_INDEX = {m: i for i, m in enumerate(MATRIX_ROLES)}
VEC_INDEX = {m: i for i, m in enumerate(VECTOR_ROLES)}


def s_target(role: int, shape: tuple[int, ...], cfg: StageConfig) -> float:
    """SPEC: 03 §3.3 -- the innate scale of a role, a function of shape only."""
    if role in (ROLE_EMB, ROLE_HEAD_OUT):
        return cfg.d_model ** -0.5
    if role == ROLE_VE:
        return shape[1] ** -0.5
    if role in CONV_ROLES:
        return 0.5
    if role in MATRIX_ROLES:
        return shape[1] ** -0.5                     # fan_in of F.linear(x, W)
    if role in GAIN_ROLES:
        return 1.0
    if role == ROLE_LOGIT_SCALE:
        return 1.0
    raise KeyError(role)


def offset(role: int) -> float:
    """SPEC: 03 §3.4 -- `1` for gain-like roles, `0` otherwise."""
    return 1.0 if role in GAIN_ROLES else 0.0


class DecodeContext:
    """Everything a decode needs besides `Gamma`: tables, bank, stage, vocab."""

    def __init__(self, cfg: StageConfig, tables: CodeTables, bank: LatentBank,
                 vocab: int, *, tied: bool = True):
        if tables.L != cfg.L:
            raise ValueError(f"code tables built for L={tables.L}, stage has L={cfg.L}")
        if bank.cfg.L != cfg.L:
            raise ValueError(f"latent bank built for L={bank.cfg.L}, stage has L={cfg.L}")
        self.cfg, self.tables, self.bank, self.vocab, self.tied = cfg, tables, bank, vocab, tied

    def keys(self) -> list[tuple[int, int]]:
        return self.cfg.keys(self.vocab, tied=self.tied)

    def shape(self, role: int, l: int) -> tuple[int, ...]:
        return self.cfg.role_shape(role, l, self.vocab, tied=self.tied)


class Genome(nn.Module):
    """`Gamma`. SPEC: 03 §3.6 enumerates every tensor; nothing else exists here."""

    def __init__(self, shape: GenomeShape, *, seed: int = 0):
        super().__init__()
        self.shape = shape
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        d_h, H, d_g = shape.d_h, shape.H, shape.d_g
        n_mat, n_vec = len(MATRIX_ROLES), len(VECTOR_ROLES)

        def normal(*size, std):
            return nn.Parameter(torch.randn(*size, generator=g) * std)

        # SPEC: 03 §3.7 -- initialisation.
        self.W_1 = normal(d_h, shape.in_chan, std=shape.in_chan ** -0.5)
        self.c_1 = nn.Parameter(torch.zeros(d_h))
        self.W_1_tok = normal(d_h, shape.in_tok, std=shape.in_tok ** -0.5)
        self.c_1_tok = nn.Parameter(torch.zeros(d_h))
        self.W_1b = normal(d_h, d_h, std=d_h ** -0.5)
        self.c_1b = nn.Parameter(torch.zeros(d_h))
        self.W_2 = normal(H * d_g, d_h, std=d_h ** -0.5)
        self.c_2 = nn.Parameter(torch.zeros(H * d_g))
        self.O = normal(n_mat, H, d_g, d_g, std=d_g ** -0.5)
        self.kappa_hat = nn.Parameter(torch.full((n_mat,), math.log(math.expm1(0.1))))
        self.o = nn.Parameter(torch.zeros(n_vec, d_g))
        n_ctx = shape.n_layer_code + shape.n_M
        self.A_sigma = normal(shape.d_sigma, n_ctx, std=n_ctx ** -0.5)
        self.b_sigma = nn.Parameter(torch.zeros(shape.d_sigma))
        # SPEC: 03 §3.7 -- `w_sigma` at N(0, 1e-6) rather than exactly zero:
        # sigma = 1 to within 0.1% at initialisation, and `A_sigma`'s gradient path
        # (which is proportional to `w_sigma`) is live from the first step (D2).
        self.w_sigma = normal(shape.d_sigma, std=1e-3)

    # ------------------------------------------------------------------ counts
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_table(self) -> dict[str, int]:
        """The rows of §03.6, for `G-SPECNUM`."""
        return {name: p.numel() for name, p in self.named_parameters()}

    # ---------------------------------------------------------------- features
    def features(self, codes: torch.Tensor, latents: torch.Tensor,
                 layer_code: torch.Tensor, y: torch.Tensor, side: float, *,
                 token: bool) -> torch.Tensor:
        """(3.1). `codes (n, n_code)`, `latents (n, d_z)` -> `(n, H, d_g)`."""
        n = codes.shape[0]
        dt, dev = self.W_1.dtype, self.W_1.device
        parts = [codes.to(dt), latents.to(dt),
                 layer_code.to(dt).expand(n, -1), y.to(dt).expand(n, -1),
                 torch.full((n, 1), float(side), dtype=dt, device=dev)]
        xi = torch.cat(parts, dim=-1)
        if token:
            check_shape(xi, [n, self.shape.in_tok], name="xi_tok")
            h = F.gelu(F.linear(xi, self.W_1_tok, self.c_1_tok))
        else:
            check_shape(xi, [n, self.shape.in_chan], name="xi_chan")
            h = F.gelu(F.linear(xi, self.W_1, self.c_1))
        h = F.gelu(F.linear(h, self.W_1b, self.c_1b))
        out = F.linear(h, self.W_2, self.c_2)
        return out.view(n, self.shape.H, self.shape.d_g)

    def _axis_features(self, ctx: DecodeContext, axis: str, l: int, side: float,
                       cache: Optional[dict] = None) -> torch.Tensor:
        n = ctx.cfg.axis_size(axis, l, ctx.vocab)
        # Features depend on the layer through (r_l, y_l) for every axis, so the
        # cache key carries the layer even for RES and TOK.
        key = (axis, l, side, n)
        if cache is not None and key in cache:
            return cache[key]
        codes = ctx.tables.axis_codes(axis, n)
        latents = ctx.bank.unit_latents(axis, l, n)
        out = self.features(codes, latents, ctx.tables.layer_code(l),
                            ctx.bank.layer_latent(l), side, token=(axis == AX_TOK))
        if cache is not None:
            cache[key] = out
        return out

    # ------------------------------------------------------------------ scale
    def log_sigma(self, ctx: DecodeContext, l: int, role: int) -> torch.Tensor:
        """(3.5)."""
        r = ctx.tables.layer_code(l).to(self.A_sigma.dtype)
        t = ctx.tables.role_code(role).to(self.A_sigma.dtype)
        h = torch.tanh(F.linear(torch.cat([r, t], dim=-1), self.A_sigma, self.b_sigma))
        return (h * self.w_sigma).sum()

    def kappa(self, role: int) -> torch.Tensor:
        return F.softplus(self.kappa_hat[MAT_INDEX[role]])

    # ----------------------------------------------------------------- decode
    def raw_matrix(self, ctx: DecodeContext, role: int, l: int, *,
                   cache: Optional[dict] = None,
                   chunk_elems: int = 1 << 26) -> tuple[torch.Tensor, torch.Tensor]:
        """(3.2)-(3.3) for one matrix role, before (3.4): `(raw, rms(raw))`.

        Proposition 2 (05 §5.8) is a statement about `raw`: its entries on the
        nested units are the same at every stage, and only `rms` changes.
        """
        rows_axis, cols_axis = ROLE_AXES[role]
        shp = ctx.shape(role, l)
        u = self._axis_features(ctx, rows_axis, l, +1.0, cache)        # (rows, H, d_g)
        v = self._axis_features(ctx, cols_axis, l, -1.0, cache)        # (cols, H, d_g)
        check_shape(u, [shp[0], self.shape.H, self.shape.d_g], name="u")
        check_shape(v, [shp[1], self.shape.H, self.shape.d_g], name="v")
        O = self.O[MAT_INDEX[role]]                                     # (H, d_g, d_g)
        kappa = self.kappa(role)
        rows, cols = shp
        chunk = max(1, min(cols, chunk_elems // max(rows, 1)))
        pieces = []
        # P^{(h)} = u^{(h)} O^{(h)}: (H, rows, d_g), formed once per matrix.
        P = torch.einsum("rhd,hde->hre", u, O)
        for c0 in range(0, cols, chunk):
            vc = v[c0:c0 + chunk]                                       # (chunk, H, d_g)
            S = torch.einsum("hre,che->hrc", P, vc)                     # (H, rows, chunk)
            raw = (S + kappa * S.pow(3)).sum(0)                         # (rows, chunk)
            pieces.append(raw)
        raw = torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
        rms = torch.sqrt(raw.pow(2).mean() + 1e-12)
        return raw, rms

    def decode_matrix(self, ctx: DecodeContext, role: int, l: int, *,
                      cache: Optional[dict] = None,
                      chunk_elems: int = 1 << 26) -> torch.Tensor:
        """(3.2)-(3.4) for one matrix role."""
        raw, rms = self.raw_matrix(ctx, role, l, cache=cache, chunk_elems=chunk_elems)
        shp = ctx.shape(role, l)
        scale = s_target(role, shp, ctx.cfg) * torch.exp(self.log_sigma(ctx, l, role))
        return raw / rms * scale

    def decode_vector(self, ctx: DecodeContext, role: int, l: int, *,
                      cache: Optional[dict] = None) -> torch.Tensor:
        """(3.6) for a vector or scalar role."""
        rows_axis, _ = ROLE_AXES[role]
        shp = ctx.shape(role, l)
        u = self._axis_features(ctx, rows_axis, l, +1.0, cache)        # (n, H, d_g)
        check_shape(u, [shp[0], self.shape.H, self.shape.d_g], name="u_vec")
        return u[:, 0, :] @ self.o[VEC_INDEX[role]] + offset(role)

    def decode(self, ctx: DecodeContext, role: int, l: int, *,
               cache: Optional[dict] = None) -> torch.Tensor:
        if role in MATRIX_ROLES:
            return self.decode_matrix(ctx, role, l, cache=cache)
        return self.decode_vector(ctx, role, l, cache=cache)

    def decode_all(self, ctx: DecodeContext, keys: Optional[list[tuple[int, int]]] = None
                   ) -> dict[tuple[int, int], torch.Tensor]:
        """Every `(role, layer)` of the stage. Features are cached per (axis, layer, side)."""
        cache: dict = {}
        out = {}
        for role, l in (keys if keys is not None else ctx.keys()):
            out[(role, l)] = self.decode(ctx, role, l, cache=cache)
        return out

    # ---------------------------------------------------------------- reference
    @torch.no_grad()
    def decode_reference(self, ctx: DecodeContext, role: int, l: int) -> torch.Tensor:
        """A naive per-entry evaluation of (3.1)-(3.6), for D1. Slow on purpose."""
        rows_axis, cols_axis = ROLE_AXES[role]
        shp = ctx.shape(role, l)
        n_r = shp[0]
        cu = ctx.tables.axis_codes(rows_axis, n_r)
        zu = ctx.bank.unit_latents(rows_axis, l, n_r)
        rl, yl = ctx.tables.layer_code(l), ctx.bank.layer_latent(l)
        u = torch.stack([self.features(cu[i:i + 1], zu[i:i + 1], rl, yl, +1.0,
                                       token=(rows_axis == AX_TOK))[0] for i in range(n_r)])
        if role in VECTOR_ROLES:
            return torch.stack([u[i, 0] @ self.o[VEC_INDEX[role]] + offset(role)
                                for i in range(n_r)])
        n_c = shp[1]
        cv = ctx.tables.axis_codes(cols_axis, n_c)
        zv = ctx.bank.unit_latents(cols_axis, l, n_c)
        v = torch.stack([self.features(cv[j:j + 1], zv[j:j + 1], rl, yl, -1.0,
                                       token=(cols_axis == AX_TOK))[0] for j in range(n_c)])
        O = self.O[MAT_INDEX[role]]
        kappa = self.kappa(role)
        raw = torch.zeros(n_r, n_c, dtype=u.dtype, device=u.device)
        for i in range(n_r):
            for j in range(n_c):
                acc = torch.zeros((), dtype=u.dtype, device=u.device)
                for h in range(self.shape.H):
                    x = u[i, h] @ O[h] @ v[j, h]
                    acc = acc + x + kappa * x ** 3
                raw[i, j] = acc
        rms = torch.sqrt((raw ** 2).mean() + 1e-12)
        return raw / rms * s_target(role, shp, ctx.cfg) * torch.exp(self.log_sigma(ctx, l, role))
