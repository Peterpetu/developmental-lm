"""Cores and the function-preserving expansion (development as decompression).

SPEC: docs/palimpsest/01-model.md §1.3 (cores), §1.4 (Definition 4, Theorem 1, §1.4.1 depth)

The core is a phenotype of the same family at integer factors on width (`k_d`) and depth
(`k_L`). The expansion tiles every tensor along its unit axes -- copies of channels, heads
and inner units -- and divides by the column tiling factor wherever the matrix *reads* a
duplicated vector, so that the expanded network computes the core's function exactly
(Theorem 1). The embedding is the one tied object: divided like a reader, with the read
scale `emb_scale` raised in step so that the residual stream enters as `k` copies of the
core's. Depth is added by ANLAGE's placement rule with the new layers' branch gains at
`eps_gain` of their source's.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch

from anlage.config import (AX_GATE, AX_HEAD, AX_INNER, AX_KV, CONV_ROLES, ROLE_AXES, ROLE_DOWN,
                           ROLE_NORM_FFN, ROLE_NORM_MIX, ROLE_O, ROLE_VE, StageConfig)
from anlage.grow import layer_placement, shrink
from anlage.model import Phenotype, free_init, key_name

__all__ = ["FULL", "CORES", "core_config", "ratio", "expand", "shrink", "core_of"]

FULL = StageConfig(0, d_model=256, L=8, n_h=4, n_kv_attn=2, d_ff=1024, emb_scale=16.0)
"""SPEC: 01 §1.2 -- ANLAGE's stage 0, the phenotype of every arm (13,011,201 parameters at V = 8192)."""

CORES: dict[str, tuple[int, int]] = {"c2": (2, 1), "c4": (2, 2), "c8": (4, 1), "c16": (4, 2)}
"""SPEC: 01 §1.3 -- `name -> (k_d, k_L)`."""


def core_config(full: StageConfig, k_d: int, k_L: int) -> StageConfig:
    """The core's configuration at factors `(k_d, k_L)` of `full`. SPEC: 01 §1.3.

    `kv_c = gcd(h_c, kv)` keeps the attention KV width where the head count allows it and
    drops to one KV head when `h_c = 1`; `emb_scale_c = emb_scale / k_d` so that the
    expansion's `k_d · emb_scale_c` is the lineage constant of the full phenotype (4.1)."""
    if k_d < 1 or k_L < 1:
        raise ValueError("factors must be >= 1")
    if full.d_model % (k_d * full.d_head) or full.n_h % k_d or full.d_ff % k_d or full.L % k_L:
        raise ValueError(f"({k_d}, {k_L}) does not divide {full.summary()}")
    h = full.n_h // k_d
    return StageConfig(full.stage, d_model=full.d_model // k_d, L=full.L // k_L, n_h=h,
                       n_kv_attn=math.gcd(h, full.n_kv_attn), d_ff=full.d_ff // k_d,
                       d_head=full.d_head, chunk_size=full.chunk_size, use_ve=full.use_ve,
                       emb_scale=full.emb_scale / k_d)


def core_of(name: str, full: StageConfig = FULL) -> StageConfig:
    k_d, k_L = CORES[name]
    return core_config(full, k_d, k_L)


def ratio(full: StageConfig, core: StageConfig, vocab: int, *, tied: bool = True) -> float:
    """`r = |W| / |C|` in parameters. SPEC: 01 §1.3."""
    return full.n_params(vocab, tied=tied) / core.n_params(vocab, tied=tied)


def _rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt()) if t.numel() else 1.0


@torch.no_grad()
def expand(core: Phenotype, full: StageConfig, *, noise: float = 0.01, eps_gain: float = 0.02,
           generator: Optional[torch.Generator] = None, new: str = "copy") -> Phenotype:
    """`X(C) -> W_0`: Definition 4 with the depth rule of §1.4.1.

    `new = "copy"`: duplicated heads and inner units are copies (plus the noise of (4.5));
    `new = "fresh"`: the duplicated heads and inner units are replaced by the free
    initialisation and their output columns (`O`, `DOWN`) zeroed, so that the function is
    still the core's and the fresh units learn from their first gradient. The residual
    width is duplicated in both modes (a reparameterisation, not new capacity)."""
    if new not in ("copy", "fresh"):
        raise ValueError("new in copy|fresh")
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    cfg_c, vocab, tied = core.cfg, core.vocab, core.tied
    device = next(core.parameters()).device
    k = full.d_model // cfg_c.d_model
    if k * cfg_c.d_model != full.d_model or full.d_ff % cfg_c.d_ff or full.n_h % cfg_c.n_h:
        raise ValueError(f"{cfg_c.summary()} does not divide {full.summary()} on width")
    if abs(full.emb_scale - k * cfg_c.emb_scale) > 1e-6:
        raise ValueError("emb_scale of the full configuration must be k_d times the core's (4.1)")
    if full.d_head != cfg_c.d_head:
        raise ValueError("d_head is a lineage constant")
    # ---- width: every tensor of the core's layers at the full width --------------------
    cfg_mid = dataclasses.replace(full, L=cfg_c.L, ops_override=cfg_c.ops())
    wide: dict[tuple[int, int], torch.Tensor] = {}
    for role, l in cfg_c.keys(vocab, tied=tied):
        A = core.get(role, l).detach().float()
        shp = cfg_mid.role_shape(role, l, vocab, tied=tied)
        f_r = shp[0] // A.shape[0]
        if f_r * A.shape[0] != shp[0]:
            raise ValueError(f"{key_name(role, l)}: rows {A.shape[0]} -> {shp[0]} not integral")
        if A.dim() == 1:
            B = A.repeat(f_r)
        else:
            f_c = shp[1] // A.shape[1]
            if f_c * A.shape[1] != shp[1]:
                raise ValueError(f"{key_name(role, l)}: cols {A.shape[1]} -> {shp[1]} not integral")
            B = A.repeat(f_r, f_c)
            if f_c > 1 and role != ROLE_VE:          # a reader of a duplicated vector; VE writes
                B = B / f_c
        wide[(role, l)] = B
    if new == "fresh":
        for role, l in cfg_c.keys(vocab, tied=tied):
            if l == -1:
                continue
            rax, cax = ROLE_AXES[role]
            shp = cfg_mid.role_shape(role, l, vocab, tied=tied)
            old_rows = cfg_c.axis_size(rax, l, vocab)
            if rax in (AX_HEAD, AX_INNER, AX_GATE) or (rax == AX_KV and cfg_c.op(l) == "delta"):
                if shp[0] > old_rows:
                    fresh = free_init(role, shp, cfg_mid, generator)
                    wide[(role, l)][old_rows:] = fresh[old_rows:]
            if role in (ROLE_O, ROLE_DOWN):
                # the kept units are now the only contributors: undo the reader's division by
                # the column factor on their columns and zero the fresh units' columns (T2 found
                # the first form at 1/k -- hidden by the gain-free norm that follows every branch)
                old_cols = cfg_c.axis_size(cax, l, vocab)
                f_c = shp[1] // old_cols
                wide[(role, l)][:, :old_cols] *= f_c
                wide[(role, l)][:, old_cols:] = 0.0
    # ---- depth: the placement rule; new layers copy the nearest same-operator layer ------
    layer_map, ops = layer_placement(cfg_mid, full)
    cfg_full = dataclasses.replace(full, ops_override=None if ops == full.canonical_ops() else ops)
    out: dict[tuple[int, int], torch.Tensor] = {}
    for role, l_new in cfg_full.keys(vocab, tied=tied):
        if l_new == -1:
            src = wide[(role, -1)]
        elif layer_map[l_new] >= 0:
            src = wide[(role, layer_map[l_new])]
        else:
            same = [j for j in range(full.L) if layer_map[j] >= 0 and cfg_full.op(j) == cfg_full.op(l_new)]
            if same:
                j = min(same, key=lambda e: abs(e - l_new))
                src = wide[(role, layer_map[j])].clone()
            else:
                # no layer of this operator exists in the core (a core too shallow to hold an
                # attention layer): the new layer takes the free initialisation
                src = free_init(role, cfg_full.role_shape(role, l_new, vocab, tied=tied), cfg_full, generator)
            if role in (ROLE_NORM_MIX, ROLE_NORM_FFN):
                src = src * eps_gain
        t = src.clone()
        if t.dim() == 2 and noise > 0 and role not in CONV_ROLES:
            t = t + noise * _rms(t) * torch.randn(*t.shape, generator=generator).to(t.device)
        out[(role, l_new)] = t
    return Phenotype(cfg_full, vocab, tied=tied, weights=out).to(device)
