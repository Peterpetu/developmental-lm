"""Growth events: Definition 2 of docs/anlage/06, and its controls.

SPEC: 06 §6.2
SPEC: 06 §6.3
SPEC: 06 §6.6

A growth event maps a stage-`s` phenotype to a stage-`s+1` phenotype:

1. new identities get sampled latents (`LatentBank.grown`);
2. entries with all indices in `A_s` keep their trained values; entries with any
   index in `A_new` are decoded from the stage-`s+1` decode (ANLAGE-GROW), or
   copied with noise (B-COPY), or drawn from the free initialisation (B-RAND);
3. new layers are placed by the uniform-spreading rule of §6.2 step 3;
4. the ramp gates of §6.3 start at `rho_0` and reach 1 after `n_ramp` steps.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from gnomon.utils.shapes import check_shape
from .config import (
    AX_GATE, AX_HEAD, AX_INNER, AX_KV, AX_RES, AX_TAP, AX_TOK, AX_ONE,
    GLOBAL_ROLES, MATRIX_ROLES, ROLE_AXES, ROLE_EMB, ROLE_HEAD_OUT, StageConfig,
    VECTOR_ROLES,
)
from .model import Phenotype, free_init, key_name

__all__ = ["unit_prior", "shrink", "layer_placement", "GrowthPlan", "grow", "Ramp", "RHO_0"]

RHO_0 = 0.1
"""SPEC: 06 §6.2 step 4."""


def layer_placement(cfg_old: StageConfig, cfg_new: StageConfig
                    ) -> tuple[list[int], tuple[str, ...]]:
    """`(layer_map, ops)`: `layer_map[l_new] = l_old` or `-1`. SPEC: 06 §6.2 step 3.

    Existing layers are spread uniformly over the target depth **in their
    original order** -- layer `i` of `L` goes to slot `round(i (L' - 1)/(L - 1))`
    -- and keep their operators. The free slots become attention layers only as
    far as needed to reach the canonical target's attention count, each chosen
    nearest to an unclaimed canonical attention position; the rest are delta.

    An earlier rule spread each operator's layers independently; it reordered
    an old attention layer ahead of an old delta layer and broke Theorem G1
    (found by test G1).
    """
    L, L_new = cfg_old.L, cfg_new.L
    if L_new < L:
        raise ValueError("depth cannot shrink")
    layer_map = [-1] * L_new
    ops: list = [None] * L_new
    ops_old = cfg_old.ops()
    for i in range(L):
        j = 0 if L == 1 else int(round(i * (L_new - 1) / (L - 1)))
        if layer_map[j] != -1:
            raise RuntimeError("two existing layers mapped to one slot")
        layer_map[j] = i
        ops[j] = ops_old[i]
    canon = cfg_new.canonical_ops()
    unclaimed = [l for l in range(L_new) if canon[l] == "attn"]
    # existing attention layers claim their nearest canonical positions
    for l in range(L_new):
        if ops[l] == "attn" and unclaimed:
            c = min(unclaimed, key=lambda c: abs(c - l))
            unclaimed.remove(c)
    free = [l for l in range(L_new) if layer_map[l] == -1]
    for c in unclaimed:
        if not free:
            break
        l = min(free, key=lambda l: abs(l - c))
        free.remove(l)
        ops[l] = "attn"
    for l in free:
        ops[l] = "delta"
    return layer_map, tuple(ops)


def _old_extent(role: int, cfg_old: StageConfig, l_old: int, vocab: int,
                tied: bool) -> tuple[int, ...]:
    """The `A_s` extent of a role's tensor inside the grown tensor."""
    if l_old < 0:
        return ()
    return cfg_old.role_shape(role, l_old, vocab, tied=tied)


@dataclasses.dataclass
class GrowthPlan:
    cfg_old: StageConfig
    cfg_new: StageConfig
    layer_map: list[int]


@torch.no_grad()
def grow(model: Phenotype, cfg_new: StageConfig, *, mode: str,
         decoded: Optional[dict] = None, generator: Optional[torch.Generator] = None,
         noise: float = 0.01, Lambda: Optional[dict] = None) -> tuple[Phenotype, GrowthPlan]:
    """Grow `model` to `cfg_new`. SPEC: 06 §6.2, SPEC: 12 §12.7.

    `mode`: `"genome"` (new entries from `decoded`, the stage-`s+1` decode; kept
    for the record, §11), `"copy"` (B-COPY: duplicated existing units with
    symmetry-breaking noise at `noise` of the RMS), `"random"` (B-RAND: free
    initialisation), `"prior"` (§12.7: new units at the scale of typical trained
    units from `Lambda = unit_prior(model)`, random direction).
    """
    cfg_old, vocab, tied = model.cfg, model.vocab, model.tied
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    layer_map, ops = layer_placement(cfg_old, cfg_new)
    cfg_new = dataclasses.replace(cfg_new, ops_override=ops)
    plan = GrowthPlan(cfg_old, cfg_new, layer_map)
    device = next(model.parameters()).device
    new_w: dict[tuple[int, int], torch.Tensor] = {}
    for role, l_new in cfg_new.keys(vocab, tied=tied):
        shp = cfg_new.role_shape(role, l_new, vocab, tied=tied)
        l_old = -1 if l_new == -1 else layer_map[l_new]
        has_old = (l_new == -1) or (l_old >= 0 and model.has(role, l_old))
        # --- the new full tensor, by mode ---------------------------------
        if mode == "genome":
            if decoded is None or (role, l_new) not in decoded:
                raise ValueError(f"genome growth needs the decode of {key_name(role, l_new)}")
            full = decoded[(role, l_new)].detach().to(device).float().clone()
            check_shape(full, list(shp), name=key_name(role, l_new))
        elif mode == "random":
            full = free_init(role, shp, cfg_new, generator).to(device)
        elif mode == "copy":
            full = _copy_grow(model, role, l_old, shp, cfg_old, cfg_new, generator,
                              noise, has_old, layer_map, l_new).to(device)
        elif mode == "prior":
            if Lambda is None:
                raise ValueError("prior growth needs Lambda = unit_prior(model)")
            full = _prior_grow(role, shp, cfg_new, generator, Lambda,
                               old_shape=(tuple(model.get(role, -1 if l_new == -1 else l_old).shape)
                                          if has_old else None)).to(device)
        else:
            raise ValueError(f"unknown growth mode {mode!r}")
        # --- keep the trained entries of A_s -----------------------------
        if has_old:
            old = model.get(role, -1 if l_new == -1 else l_old).detach()
            sl = tuple(slice(0, s) for s in old.shape)
            full[sl] = old
        new_w[(role, l_new)] = full
    grown = Phenotype(cfg_new, vocab, tied=tied, weights=new_w).to(device)
    return grown, plan


def _copy_grow(model, role, l_old, shp, cfg_old, cfg_new, generator, noise, has_old,
               layer_map, l_new):
    """B-COPY: new units are perturbed duplicates of existing ones. SPEC: 06 §6.6."""
    device = next(model.parameters()).device
    if not has_old:
        # a new layer: copy the nearest existing layer of the same operator
        existing = [l for l in range(cfg_new.L) if layer_map[l] >= 0
                    and cfg_new.op(l) == cfg_new.op(l_new)]
        src_new = min(existing, key=lambda e: abs(e - l_new))
        src = model.get(role, layer_map[src_new]).detach()
    else:
        src = model.get(role, l_old).detach()
    full = torch.empty(*shp, device=device, dtype=torch.float32)
    if src.dim() == 1:
        n = src.shape[0]
        reps = -(-shp[0] // n)
        full[:] = src.repeat(reps)[: shp[0]]
    else:
        r, c = src.shape
        rr, cc = -(-shp[0] // r), -(-shp[1] // c)
        tiled = src.repeat(rr, cc)[: shp[0], : shp[1]]
        # HyperCloning-style: split duplicated *columns* (fan-in) so the
        # function is preserved up to the noise when inputs are duplicated too.
        if shp[1] > c:
            tiled = tiled * (c / shp[1])
        full[:] = tiled
    rms = float(src.float().pow(2).mean().sqrt()) if src.numel() else 1.0
    full += noise * rms * torch.randn(*shp, generator=generator).to(device)
    return full


@torch.no_grad()
def unit_prior(model: Phenotype) -> dict[str, dict[str, float]]:
    """`Lambda` of §12.7: per role, over all layers, the mean and standard deviation of
    the trained matrices' row norms (output units) and column norms (input units); for
    vector roles, of the entries. Scale-free: a few numbers per role."""
    from .config import ROLE_NAMES
    acc: dict[str, dict[str, list[float]]] = {}
    for (role, l), w in model.weights().items():
        name = ROLE_NAMES[role]
        d = acc.setdefault(name, {"row": [], "col": [], "entry": []})
        w = w.detach().float()
        if w.dim() == 2:
            d["row"] += w.norm(dim=1).tolist()
            d["col"] += w.norm(dim=0).tolist()
        else:
            d["entry"] += w.flatten().tolist()
    out: dict[str, dict[str, float]] = {}
    for name, d in acc.items():
        o: dict[str, float] = {}
        for key in ("row", "col", "entry"):
            if d[key]:
                t = torch.tensor(d[key], dtype=torch.float64)
                o[f"{key}_mu"] = float(t.mean())
                o[f"{key}_sd"] = float(t.std(unbiased=False)) if t.numel() > 1 else 0.0
        out[name] = o
    return out


def _prior_grow(role, shp, cfg_new, generator, Lambda, *, old_shape):
    """§12.7: random directions at the norms of typical trained units."""
    from .config import ROLE_NAMES
    name = ROLE_NAMES[role]
    pr = Lambda.get(name)
    full = free_init(role, shp, cfg_new, generator).float()
    if pr is None:
        return full
    if full.dim() == 1:
        if "entry_mu" in pr and old_shape is not None and shp[0] > old_shape[0]:
            n_new = shp[0] - old_shape[0]
            draw = pr["entry_mu"] + pr["entry_sd"] * torch.randn(n_new, generator=generator)
            full[old_shape[0]:] = draw.to(full.dtype)
        elif "entry_mu" in pr and old_shape is None:
            full[:] = (pr["entry_mu"] + pr["entry_sd"] * torch.randn(shp[0], generator=generator)).to(full.dtype)
        return full
    r_old, c_old = old_shape if old_shape is not None else (0, 0)
    r_new, c_new = shp
    # new rows (output units): random direction, norm from the row prior
    if r_new > r_old and "row_mu" in pr:
        block = full[r_old:]
        n = block.norm(dim=1, keepdim=True).clamp_min(1e-12)
        target = (pr["row_mu"] + pr["row_sd"] * torch.randn(r_new - r_old, 1, generator=generator)).clamp_min(0.0)
        full[r_old:] = block / n * target
    # new columns of the old rows (input units): norm from the column prior, scaled by the
    # fraction of the rows that are old so a full column would have the prior's norm
    if c_new > c_old and r_old > 0 and "col_mu" in pr:
        block = full[:r_old, c_old:]
        n = block.norm(dim=0, keepdim=True).clamp_min(1e-12)
        target = (pr["col_mu"] + pr["col_sd"] * torch.randn(1, c_new - c_old, generator=generator)).clamp_min(0.0)
        full[:r_old, c_old:] = block / n * target * (r_old / r_new) ** 0.5
    return full


@torch.no_grad()
def shrink(model: Phenotype, cfg_small: StageConfig) -> Phenotype:
    """The primordium block of a grown phenotype. SPEC: 12 §12.3 line 1.

    The inverse of `grow` on the kept entries: unit prefixes on every channel axis
    and the layers at the spread positions `round(i (L - 1)/(L_s - 1))` (G13's rule),
    with the small stage's canonical operator layout. If the large model was grown
    from a stage-`s` model by `grow`, the returned weights equal that model's on
    every kept entry (G15); on a trained adult they are its sub-network.
    """
    big = model.cfg
    if (cfg_small.d_model > big.d_model or cfg_small.d_ff > big.d_ff or cfg_small.n_h > big.n_h
            or cfg_small.L > big.L):
        raise ValueError("shrink needs a stage no larger than the model's on every axis")
    Ls, L = cfg_small.L, big.L
    sel = [0 if Ls == 1 else int(round(i * (L - 1) / (Ls - 1))) for i in range(Ls)]
    ops = tuple(big.op(j) for j in sel)
    cfg_s = dataclasses.replace(cfg_small, ops_override=ops)
    new_w = {}
    for role, l in cfg_s.keys(model.vocab, tied=model.tied):
        src_l = -1 if l == -1 else sel[l]
        if not model.has(role, src_l):
            raise ValueError(f"the model lacks {key_name(role, src_l)} needed by the small stage")
        shp = cfg_s.role_shape(role, l, model.vocab, tied=model.tied)
        w = model.get(role, src_l).detach()
        new_w[(role, l)] = w[tuple(slice(0, s_) for s_ in shp)].clone()
    return Phenotype(cfg_s, model.vocab, tied=model.tied, weights=new_w).to(next(model.parameters()).device)


class Ramp:
    """The gates of §6.3 during a ramp of `n_ramp` steps. SPEC: 06 §6.2 step 4, §6.3."""

    def __init__(self, model: Phenotype, plan: GrowthPlan, *, n_ramp: int,
                 rho_0: float = RHO_0):
        self.model, self.plan, self.n_ramp, self.rho_0 = model, plan, n_ramp, rho_0
        self.n = 0
        self._apply(rho_0)

    def rho(self, n: int) -> float:
        if self.n_ramp <= 0:
            return 1.0
        return min(1.0, self.rho_0 + (1.0 - self.rho_0) * n / self.n_ramp)

    def _apply(self, rho: float) -> None:
        m, po, pn = self.model, self.plan.cfg_old, self.plan.cfg_new
        dev = next(m.parameters()).device
        if rho >= 1.0:
            m.res_gate, m.inner_gate, m.head_gate, m.layer_gate = None, {}, {}, {}
            return
        if pn.d_model > po.d_model:
            g = torch.ones(pn.d_model, device=dev)
            g[po.d_model:] = rho
            m.res_gate = g
        for l_new in range(pn.L):
            l_old = self.plan.layer_map[l_new]
            if l_old < 0:
                m.layer_gate[l_new] = rho
                continue
            if pn.d_ff > po.d_ff:
                g = torch.ones(pn.d_ff, device=dev)
                g[po.d_ff:] = rho
                m.inner_gate[l_new] = g
            if pn.n_h > po.n_h:
                g = torch.ones(pn.n_h * pn.d_head, device=dev)
                g[po.n_h * po.d_head:] = rho
                m.head_gate[l_new] = g

    def step(self) -> float:
        self.n += 1
        r = self.rho(self.n)
        self._apply(r)
        return r

    @property
    def done(self) -> bool:
        return self.rho(self.n) >= 1.0
