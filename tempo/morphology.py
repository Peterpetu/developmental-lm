"""Latent morphology: the switch pattern, active views, births by copy, FLOP
accounting and the K-MORPH controls.

SPEC: 05 §5.1  (switches and groups; primordium blocks permanently on)
SPEC: 05 §5.2  (the branch multiply and compute skipping)
SPEC: 05 §5.3  (births by copy with 1 % noise; re-activation without a copy)
SPEC: 05 §5.5  (the primordium: quarter-stacks 0 and 2, the first heads and slices)
SPEC: 05 §5.6  (FLOPs per span, per window)
SPEC: 05 §5.7  (the controls: fixed schedule, one-shot doubling, flat adult)
SPEC: 03 §3.8  (active views are prefixes: heads `0..h_act-1`, slices `0..F_act-1`)

A group's pre-activation is `a_G = A_BASE + gamma_G` with the genome gain
`gamma_G in (-8, 8)` (04 §4.3) and the baseline `A_BASE = -4`, so that at birth
(`gamma = 0`) every switchable group is skipped (`sigma(-4) = 0.018 < eps_skip`)
and the range reaches `sigma(4) = 0.982`. (05 §5.1 wrote `a_G = gamma_G`, which
would switch every group on at birth; the baseline is the correction, recorded
in 11 §P-C.17.)
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch

from .config import TempoConfig

__all__ = [
    "EPS_SKIP", "A_BASE", "A_SKIP", "SwitchPattern", "n_active_params", "flops_per_span",
    "flops_per_window", "apply_births", "born_flags_at_birth", "control_pattern",
    "hypercloning_split",
]

EPS_SKIP = 0.02
"""SPEC: 05 §5.1 -- a block with `m_b < eps_skip` is skipped."""
A_BASE = -4.0
"""The baseline of every switchable group's pre-activation (module docstring)."""
A_SKIP = math.log(EPS_SKIP / (1.0 - EPS_SKIP))          # sigma^{-1}(0.02) = -3.892
"""`a_G` below this is skipped."""


def _sigmoid(a: float) -> float:
    return 1.0 / (1.0 + math.exp(-a))


@dataclasses.dataclass(frozen=True)
class SwitchPattern:
    """The multipliers of every switchable group; `0.0` means skipped. SPEC: 05 §5.1.

    `layer_m[l]` is `1.0` for a primordium layer, `m_Q` for a layer of an active
    switchable quarter-stack, `0.0` for a skipped one. `head_m[l]` and
    `slice_m[l]` are the multipliers of layer `l`'s switchable head group and
    slice group (`0.0` when skipped). Primordium heads and slices always have
    multiplier `1.0`.
    """

    cfg: TempoConfig
    layer_m: tuple[float, ...]
    head_m: tuple[float, ...]
    slice_m: tuple[float, ...]

    def __post_init__(self) -> None:
        L = self.cfg.L
        if len(self.layer_m) != L or len(self.head_m) != L or len(self.slice_m) != L:
            raise ValueError("one multiplier per layer")
        for l in self.cfg.primordium_layers:
            if self.layer_m[l] != 1.0:
                raise ValueError("primordium layers are permanently on")

    # ------------------------------------------------------------- constructors
    @classmethod
    def primordium(cls, cfg: TempoConfig) -> "SwitchPattern":
        lm = tuple(1.0 if l in cfg.primordium_layers else 0.0 for l in range(cfg.L))
        return cls(cfg, lm, (0.0,) * cfg.L, (0.0,) * cfg.L)

    @classmethod
    def adult(cls, cfg: TempoConfig) -> "SwitchPattern":
        return cls(cfg, (1.0,) * cfg.L, (1.0,) * cfg.L, (1.0,) * cfg.L)

    @classmethod
    def from_gains(cls, cfg: TempoConfig, gamma_q, gamma_h, gamma_f) -> "SwitchPattern":
        """From the morphology gains: `gamma_q (2,)` for `Q_1, Q_3`, `gamma_h (L,)`,
        `gamma_f (L,)`. SPEC: 05 §5.1."""
        gq = [float(v) for v in gamma_q]
        gh = [float(v) for v in gamma_h]
        gf = [float(v) for v in gamma_f]
        if len(gq) != 2 or len(gh) != cfg.L or len(gf) != cfg.L:
            raise ValueError("gamma_q (2,), gamma_h (L,), gamma_f (L,)")

        def m(a: float) -> float:
            return _sigmoid(a) if a >= A_SKIP else 0.0

        lm = []
        for l in range(cfg.L):
            if l in cfg.primordium_layers:
                lm.append(1.0)
            else:
                idx = cfg.switchable_quarters.index(cfg.quarter_of(l))
                lm.append(m(A_BASE + gq[idx]))
        hm = [m(A_BASE + gh[l]) for l in range(cfg.L)]
        fm = [m(A_BASE + gf[l]) for l in range(cfg.L)]
        return cls(cfg, tuple(lm), tuple(hm), tuple(fm))

    # ------------------------------------------------------------------ queries
    def layer_active(self, l: int) -> bool:
        return self.layer_m[l] > 0.0

    def active_layers(self) -> tuple[int, ...]:
        return tuple(l for l in range(self.cfg.L) if self.layer_active(l))

    def h_act(self, l: int) -> int:
        return self.cfg.h if self.head_m[l] > 0.0 else self.cfg.primordium_heads

    def F_act(self, l: int) -> int:
        return self.cfg.F if self.slice_m[l] > 0.0 else self.cfg.primordium_slices * self.cfg.slice

    def head_mask(self, l: int, device, dtype) -> torch.Tensor:
        """`(h_act d_head,)`: 1 on primordium heads, the group's multiplier on the rest."""
        cfg = self.cfg
        n0 = cfg.primordium_heads * cfg.d_head
        n1 = (self.h_act(l) - cfg.primordium_heads) * cfg.d_head
        return torch.cat([torch.ones(n0, device=device, dtype=dtype),
                          torch.full((n1,), self.head_m[l], device=device, dtype=dtype)])

    def slice_mask(self, l: int, device, dtype) -> torch.Tensor:
        cfg = self.cfg
        n0 = cfg.primordium_slices * cfg.slice
        n1 = self.F_act(l) - n0
        return torch.cat([torch.ones(n0, device=device, dtype=dtype),
                          torch.full((n1,), self.slice_m[l], device=device, dtype=dtype)])

    def key(self) -> tuple:
        """The active *shape* (what the views depend on), not the multipliers."""
        return (tuple(self.layer_active(l) for l in range(self.cfg.L)),
                tuple(self.h_act(l) for l in range(self.cfg.L)),
                tuple(self.F_act(l) for l in range(self.cfg.L)))

    def to_dict(self) -> dict:
        return {"layer_m": list(self.layer_m), "head_m": list(self.head_m), "slice_m": list(self.slice_m)}

    @classmethod
    def from_dict(cls, cfg: TempoConfig, d: dict) -> "SwitchPattern":
        return cls(cfg, tuple(d["layer_m"]), tuple(d["head_m"]), tuple(d["slice_m"]))

    def groups_on(self) -> dict[str, bool]:
        """Every switchable group and whether it is active: keys `Q1, Q3, H{l}, F{l}`."""
        cfg = self.cfg
        out = {}
        for q in cfg.switchable_quarters:
            out[f"Q{q}"] = self.layer_active(cfg.layers_of_quarter(q)[0])
        for l in range(cfg.L):
            out[f"H{l}"] = self.head_m[l] > 0.0
            out[f"F{l}"] = self.slice_m[l] > 0.0
        return out


# =============================================================================
# Counting. SPEC: 03 §3.9, 05 §5.6
# =============================================================================

def n_active_params(cfg: TempoConfig, pattern: SwitchPattern) -> int:
    """The parameters the active shape uses (rows/columns of active heads and slices)."""
    d, dh = cfg.d, cfg.d_head
    total = d * cfg.d_x + d + cfg.d_x * d + d                          # W_in, b_in, W_out, rho_fin
    for l in pattern.active_layers():
        h, F = pattern.h_act(l), pattern.F_act(l)
        total += 3 * d * F + 4 * d                                     # FFN, four norm vectors
        if cfg.op(l) == "delta":
            total += 3 * h * dh * d + 3 * 4 * h * dh + 2 * h * d + h * dh * d + d * h * dh + 2 * h + 1
        else:
            total += h * dh * d + 2 * cfg.h_kv * dh * d + h * dh * d + d * h * dh + 1
    return int(total)


def flops_per_span(cfg: TempoConfig, pattern: SwitchPattern, position: int, *, train: bool = True,
                   effective: bool = False) -> float:
    """(5.2): the FLOPs of one span at window position `position` (0-based); the attention
    term is scored per span at the window's `T_s` (its mean over positions), so the sum
    over a window is exact. `effective=True` weights each switchable group's blocks by
    their multiplier `m` (the *effective* size the reward charges, 05 §5.6: a block entering
    at `m = 0.02` contributes and is charged 2 %)."""
    mult = 6.0 if train else 2.0
    d, dh = cfg.d, cfg.d_head
    f = mult * (d * cfg.d_x + cfg.d_x * d)
    ph, ps = cfg.primordium_heads, cfg.primordium_slices * cfg.slice
    for l in pattern.active_layers():
        h, F = pattern.h_act(l), pattern.F_act(l)
        m_l = pattern.layer_m[l] if (effective and l not in cfg.primordium_layers) else 1.0
        if effective:
            h = ph + (h - ph) * pattern.head_m[l]
            F = ps + (F - ps) * pattern.slice_m[l]
        f_l = mult * 3 * d * F + mult * 4 * d
        if cfg.op(l) == "delta":
            f_l += mult * (3 * h * dh * d + 2 * h * d + h * dh * d + d * h * dh)
            f_l += mult * h * 2 * dh * dh
        else:
            f_l += mult * (h * dh * d + 2 * cfg.h_kv * dh * d + h * dh * d + d * h * dh)
            f_l += mult * h * 2 * dh * (position + 1)              # scores and values, forward + backward
        f += m_l * f_l
    return float(f)


def flops_per_window(cfg: TempoConfig, pattern: SwitchPattern, *, train: bool = True,
                     effective: bool = False) -> float:
    return float(sum(flops_per_span(cfg, pattern, t, train=train, effective=effective) for t in range(cfg.T_s)))


# =============================================================================
# Births. SPEC: 05 §5.3
# =============================================================================

def born_flags_at_birth(cfg: TempoConfig) -> dict[str, bool]:
    """Every switchable group starts never-born."""
    flags = {f"Q{q}": False for q in cfg.switchable_quarters}
    for l in range(cfg.L):
        flags[f"H{l}"] = False
        flags[f"F{l}"] = False
    return flags


def _source_layer(cfg: TempoConfig, l: int, active: tuple[int, ...]) -> int:
    """`l - L/4`: the layer at the same relative depth of the previous quarter-stack, which
    has the same operator because a quarter-stack is a whole number of operator periods
    (config check); it must be active in the pre-move pattern (05 §5.3)."""
    src = l - cfg.quarter
    if src < 0 or src not in active or cfg.op(src) != cfg.op(l):
        raise ValueError(f"layer {l} has no active same-operator source at {src}")
    return src


@torch.no_grad()
def apply_births(backbone, prev: SwitchPattern, new: SwitchPattern, born: dict[str, bool],
                 generator: torch.Generator, *, noise: float = 0.01) -> list[str]:
    """Copy-initialise every group that is active in `new`, inactive in `prev`, and
    never born; mark it born; zero its delta states. Returns the born groups.

    Heads `ph..h-1` copy heads `0..ph-1` (`j <- j - ph`), slices likewise, a layer
    copies its source layer (05 §5.3). Noise `N(0, (noise * RMS(source))^2)`.
    """
    cfg = backbone.cfg
    W = backbone.W
    out = []

    def eta(src):
        return noise * src.float().pow(2).mean().sqrt() * torch.randn(
            src.shape, generator=generator, dtype=torch.float32).to(src.device, src.dtype)

    def copy_rows(name, dst_rows, src_rows, exact=False):
        src = W[name][src_rows]
        W[name][dst_rows] = src if exact else src + eta(src)

    def copy_cols(name, dst_cols, src_cols):
        src = W[name][:, src_cols]
        W[name][:, dst_cols] = src + eta(src)

    dh, ph, ps = cfg.d_head, cfg.primordium_heads, cfg.primordium_slices * cfg.slice
    # layers first, so that head/slice births inside a newborn layer copy its copied blocks
    prev_active = prev.active_layers()
    for q in cfg.switchable_quarters:
        key = f"Q{q}"
        layers = cfg.layers_of_quarter(q)
        if new.layer_active(layers[0]) and not prev.layer_active(layers[0]) and not born[key]:
            for l in layers:
                src = _source_layer(cfg, l, prev_active)
                for name in backbone.layer_param_names(l):
                    sname = name.replace(f"L{l}_", f"L{src}_", 1)
                    s = W[sname]
                    exact = name.endswith(("_CQ", "_CK", "_CV", "_A0", "_G0", "_GOG"))
                    W[name].copy_(s if exact else s + eta(s))
                backbone.reset_state_layer(l)
            born[key] = True
            out.append(key)
    for l in range(cfg.L):
        # a head or slice group is active only when its layer is (05 §5.1); a group of a
        # skipped layer is neither born nor flagged
        h_new = new.head_m[l] > 0.0 and new.layer_active(l)
        h_prev = prev.head_m[l] > 0.0 and prev.layer_active(l)
        key = f"H{l}"
        if h_new and not h_prev and not born[key]:
            dst = slice(ph * dh, cfg.h * dh)
            src = slice(0, ph * dh)
            for role in ("Q", "OG"):
                copy_rows(f"L{l}_{role}", dst, src)
            copy_cols(f"L{l}_O", dst, src)
            if cfg.op(l) == "delta":
                for role in ("K", "V"):
                    copy_rows(f"L{l}_{role}", dst, src)
                for role in ("CQ", "CK", "CV"):
                    copy_rows(f"L{l}_{role}", dst, src, exact=True)
                copy_rows(f"L{l}_AB", slice(2 * ph, 2 * cfg.h), slice(0, 2 * ph))
                for role in ("A0", "G0"):
                    W[f"L{l}_{role}"][ph:] = W[f"L{l}_{role}"][:ph].clone()
                backbone.reset_state_heads(l, ph)
            born[key] = True
            out.append(key)
        f_new = new.slice_m[l] > 0.0 and new.layer_active(l)
        f_prev = prev.slice_m[l] > 0.0 and prev.layer_active(l)
        key = f"F{l}"
        if f_new and not f_prev and not born[key]:
            dst = slice(ps, cfg.F)
            src = slice(0, ps)
            copy_rows(f"L{l}_GATE", dst, src)
            copy_rows(f"L{l}_UP", dst, src)
            copy_cols(f"L{l}_DOWN", dst, src)
            born[key] = True
            out.append(key)
    return out


@torch.no_grad()
def hypercloning_split(backbone, prev: SwitchPattern, born: dict[str, bool], generator: torch.Generator,
                       *, noise: float = 0.01) -> list[str]:
    """The one-shot doubling control (05 §5.7 (b)): copy every switchable group from its
    primordium counterpart *exactly* on the input side and halve the *output-side* weights
    of both halves, `W_o[:, src] <- W_o[:, src]/2`, `W_o[:, new] <- W_o[:, src]/2 + eta`
    (`W_down` likewise; layers by plain copy), so the function at `m = 1` equals the source
    up to `eta`. Marks the caller's `born` flags; returns the born groups."""
    cfg = backbone.cfg
    W = backbone.W
    dh, ph, ps = cfg.d_head, cfg.primordium_heads, cfg.primordium_slices * cfg.slice
    out = apply_births(backbone, prev, SwitchPattern.adult(cfg), born, generator, noise=0.0)
    for l in range(cfg.L):
        for name, src_c, dst_c in ((f"L{l}_O", slice(0, ph * dh), slice(ph * dh, cfg.h * dh)),
                                   (f"L{l}_DOWN", slice(0, ps), slice(ps, cfg.F))):
            W[name][:, src_c] *= 0.5
            src = W[name][:, src_c]
            W[name][:, dst_c] = src + noise * src.float().pow(2).mean().sqrt() * torch.randn(
                src.shape, generator=generator, dtype=torch.float32).to(src.device, src.dtype)
    return out


def control_pattern(cfg: TempoConfig, arm: str, frac: float) -> SwitchPattern:
    """The switch pattern of a K-MORPH control arm at budget fraction `frac`. SPEC: 05 §5.7.

    (a) `"schedule"`: head groups on at 0.15, slice groups at 0.25, quarter-stacks at
    0.60, each entering at `m = 0.5` and ramping linearly to `0.98` over 5 % of the
    budget; (b) `"oneshot"`: everything on at `m = 1` from 0.5; (c) `"flat"`: the adult.
    """
    if arm == "flat":
        return SwitchPattern.adult(cfg)
    if arm == "oneshot":
        return SwitchPattern.adult(cfg) if frac >= 0.5 else SwitchPattern.primordium(cfg)
    if arm != "schedule":
        raise ValueError(arm)

    def ramp(t0: float) -> float:
        """`a_G` from `sigma^-1(0.5) = 0` to `sigma^-1(0.98)` linearly over 5 % of the budget."""
        if frac < t0:
            return 0.0
        a = math.log(0.98 / 0.02) * min(1.0, (frac - t0) / 0.05)
        return _sigmoid(a)

    mh, mf, mq = ramp(0.15), ramp(0.25), ramp(0.60)
    lm = tuple(1.0 if l in cfg.primordium_layers else mq for l in range(cfg.L))
    return SwitchPattern(cfg, lm, (mh,) * cfg.L, (mf,) * cfg.L)
