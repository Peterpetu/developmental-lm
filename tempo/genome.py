"""The genome, the role registry, the low-rank expression map, the episode gains
and the regression.

SPEC: 04 §4.1  (g in R^{d_g}, prior 0, norm bound)
SPEC: 04 §4.2  (the role registry: kinds, sizes, counts)
SPEC: 04 §4.3  (u_rho = B_rho tanh(A_rho g); gamma_rho = c_rho tanh(u_rho); ranges)
SPEC: 04 §4.4  (how each kind enters)
SPEC: 04 §4.5  (the episode cycle: express once, free differentiable gains, refit theta)
SPEC: 04 §4.7  (no gradient reaches g)
SPEC: 04 §4.8  (the SVF ablation)

`Expression` holds `theta = {A_rho, B_rho}` packed: `A (n_roles, r, d_g)` and
`B_flat (n_gamma, r)` with a row-to-role index, so that expressing every role is
one gather and one product. `EpisodeGains` holds the episode's `u`: a leaf
parameter for the differentiable kinds (matrix, norm, offset) and a buffer for
the programme kinds (lr, morph, mixture). `backbone_gains` turns `gamma` into
the dictionary `tempo/backbone.py` consumes.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch
import torch.nn as nn

from gnomon.utils.shapes import check_shape
from .backbone import Backbone
from .config import TempoConfig

__all__ = [
    "KINDS", "RANGES", "DIFFERENTIABLE_KINDS", "PROGRAMME_KINDS", "Role", "RoleRegistry",
    "Expression", "EpisodeGains", "backbone_gains", "lr_multipliers", "mixture", "genome_prior",
    "GENOME_NORM_BOUND", "N_LR_GROUPS", "N_STREAMS", "svf_effective",
]

KINDS = ("matrix", "norm", "offset", "lr", "morph", "mixture")
RANGES = {"matrix": 0.5, "norm": 0.2, "offset": 2.0, "lr": math.log(4.0), "morph": 8.0, "mixture": 4.0}
"""SPEC: 04 §4.3 -- `c_rho` per kind."""
DIFFERENTIABLE_KINDS = ("matrix", "norm", "offset")
PROGRAMME_KINDS = ("lr", "morph", "mixture")
N_LR_GROUPS = 4
"""SPEC: 04 §4.2 -- matrices, interface embeddings, vectors, heads."""
N_STREAMS = 7
GENOME_NORM_BOUND_FACTOR = 3.0
B_PROG_SCALE = 0.2
"""SPEC: 04 §4.3 -- the fixed frame of the programme kinds (11 §P-C.25)."""
HISTORY = 16
"""SPEC: 04 §4.5 -- the regression fits the last HISTORY (g, u) pairs of the life."""


def GENOME_NORM_BOUND(d_g: int) -> float:
    """SPEC: 04 §4.1 -- `||g - g_0|| <= 3 sqrt(d_g)`: the bound is centred at the prior."""
    return GENOME_NORM_BOUND_FACTOR * math.sqrt(d_g)


def genome_prior(d_g: int, seed: int = 0) -> torch.Tensor:
    """`g_0 ~ N(0, I)` from `seed`, `||g_0|| = sqrt(d_g)` in expectation. SPEC: 04 §4.1.

    Not zero: at `g = 0` the expression `B tanh(A g)` is identically zero whatever
    `B`, so the regression of 04 (4.4) could fit nothing at birth (11 §P-C.19). The
    identity at birth is carried by `B = 0`, not by `g`."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 7 + 3)
    return torch.randn(d_g, generator=g)


@dataclasses.dataclass(frozen=True)
class Role:
    name: str
    kind: str
    size: int
    offset: int                 # first row in the flat gain vector
    n_in: int = 0               # matrix roles: the input side comes first, then the output side
    n_out: int = 0

    @property
    def c(self) -> float:
        return RANGES[self.kind]


class RoleRegistry:
    """The roles of a backbone at its provisioned shape. SPEC: 04 §4.2."""

    def __init__(self, backbone: Backbone, *, n_streams: int = N_STREAMS, n_lr_groups: int = N_LR_GROUPS,
                 svf: bool = False):
        cfg = backbone.cfg
        self.cfg = cfg
        self.svf = bool(svf)
        roles: list[Role] = []
        off = 0

        def add(name, kind, size, n_in=0, n_out=0):
            nonlocal off
            roles.append(Role(name, kind, size, off, n_in, n_out))
            off += size

        for name in backbone.matrix_names():
            n_out, n_in = backbone.W[name].shape
            if self.svf:
                add(name, "matrix", min(n_in, n_out), n_in, n_out)      # singular-direction gains (04 §4.8)
            else:
                add(name, "matrix", n_in + n_out, n_in, n_out)
        for l in range(cfg.L):
            for r in ("RMIX", "RFFN", "NMIX", "NFFN"):
                add(f"L{l}_{r}", "norm", cfg.d)
        add("OUT_rho", "norm", cfg.d)
        n_delta = cfg.n_delta_layers
        add("A0", "offset", cfg.h * n_delta)
        add("G0", "offset", cfg.h * n_delta)
        add("GOG", "offset", cfg.L)
        add("LR", "lr", n_lr_groups)
        add("MORPH_Q", "morph", len(cfg.switchable_quarters))
        add("MORPH_H", "morph", cfg.L)
        add("MORPH_F", "morph", cfg.L)
        add("MIX", "mixture", n_streams)
        self.roles = tuple(roles)
        self.n_gamma = off
        self.by_name = {r.name: r for r in roles}
        self.delta_layers = tuple(l for l in range(cfg.L) if cfg.op(l) == "delta")
        # row -> role index, row -> range, row -> differentiable
        role_of_row = torch.empty(off, dtype=torch.long)
        c_of_row = torch.empty(off)
        diff_of_row = torch.empty(off, dtype=torch.bool)
        for i, r in enumerate(roles):
            role_of_row[r.offset:r.offset + r.size] = i
            c_of_row[r.offset:r.offset + r.size] = r.c
            diff_of_row[r.offset:r.offset + r.size] = r.kind in DIFFERENTIABLE_KINDS
        self.role_of_row, self.c_of_row, self.diff_of_row = role_of_row, c_of_row, diff_of_row

    @property
    def n_roles(self) -> int:
        return len(self.roles)

    def n_theta(self, d_g: int, r: int) -> int:
        """`sum_rho (r d_g + n_rho r)`. SPEC: 04 §4.3."""
        return self.n_roles * r * d_g + self.n_gamma * r

    def count_table(self) -> dict[str, tuple[int, int]]:
        """`kind -> (roles, gains)` for G-SPECNUM."""
        out = {}
        for k in KINDS:
            rs = [r for r in self.roles if r.kind == k]
            out[k] = (len(rs), sum(r.size for r in rs))
        return out

    def slice(self, name: str) -> slice:
        r = self.by_name[name]
        return slice(r.offset, r.offset + r.size)

    def gamma(self, u: torch.Tensor) -> torch.Tensor:
        """`gamma = c tanh(u)` row-wise. SPEC: 04 (4.2)."""
        check_shape(u, [self.n_gamma], name="u")
        return self.c_of_row.to(u.device, u.dtype) * torch.tanh(u)


class Expression(nn.Module):
    """`theta = {A_rho, B_rho}`. SPEC: 04 (4.1a), §4.5.

    Differentiable kinds: `u_rho = B_rho tanh(A_rho g)` with `B_rho = 0` at birth (the
    identity) and `B_rho` refit by regression to the life's last `HISTORY` pairs
    `(g_e, u_e^{end})`. Programme kinds (learning rate, morphology, mixture) have no
    per-step gradient to regress to; they carry a *fixed* seeded frame `B_prog` and the
    differential form `u_rho = B_prog (tanh(A_rho g) - tanh(A_rho g_0))`, zero at the
    prior for every `B_prog` and moved by the genome step alone (11 §P-C.25: with `B = 0`
    and no target they could never move). `A ~ N(0, 1/(4 d_g))` keeps `tanh` unsaturated
    inside the bound `||g - g_0|| <= 3 sqrt(d_g)`.
    """

    def __init__(self, registry: RoleRegistry, d_g: int, r: int, *, g0: Optional[torch.Tensor] = None,
                 seed: int = 0):
        super().__init__()
        self.reg, self.d_g, self.r = registry, d_g, r
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.A = nn.Parameter(torch.randn(registry.n_roles, r, d_g, generator=g) / (2.0 * math.sqrt(d_g)))
        self.B_flat = nn.Parameter(torch.zeros(registry.n_gamma, r))
        self.register_buffer("role_of_row", registry.role_of_row.clone())
        self.register_buffer("diff_row", registry.diff_of_row.clone())
        B_prog = torch.randn(registry.n_gamma, r, generator=g) * B_PROG_SCALE
        self.register_buffer("B_prog", B_prog * (~registry.diff_of_row)[:, None].float())
        self.register_buffer("g0", torch.zeros(d_g) if g0 is None else g0.detach().clone().float())

    def n_params(self) -> int:
        return self.A.numel() + self.B_flat.numel()

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """`u (n_gamma,)` from `g (d_g,)`: the regressed map on the differentiable rows, the
        differential fixed frame on the programme rows."""
        check_shape(g, [self.d_g], name="g")
        t = torch.tanh(self.A @ g.to(self.A.dtype))                          # (n_roles, r)
        t0 = torch.tanh(self.A @ self.g0.to(self.A.dtype))
        rows = self.role_of_row
        u_diff = (self.B_flat * t[rows]).sum(-1)
        u_prog = (self.B_prog * (t[rows] - t0[rows])).sum(-1)
        return torch.where(self.diff_row, u_diff, u_prog)

    @torch.no_grad()
    def express(self, g: torch.Tensor) -> torch.Tensor:
        """The once-per-episode expression, detached (04 §4.7)."""
        return self.forward(g).detach().clone()

    def fit(self, pairs, *, steps: int = 500, lr: float = 3e-3) -> float:
        """(4.4): regress `theta` on the differentiable rows so that `Phi_theta(g_e)` reproduces
        `u_e` for every pair of the history (a single pair is always exactly fittable; the
        history is what constrains the map, 04 §4.5); returns the relative residual on the
        *last* pair, `R = ||Phi(g) - u||^2 / ||u||^2` over the differentiable rows.
        `pairs`: an iterable of `(g (d_g,), u (n_gamma,))`; the last is the current one."""
        pairs = [(g.detach().to(self.A.device), u.detach().to(self.A.device)) for g, u in pairs]
        if not pairs:
            raise ValueError("fit needs at least one pair")
        mask = self.diff_row
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = sum(((self.forward(g) - u)[mask]).square().sum() for g, u in pairs) / len(pairs)
            loss.backward()
            opt.step()
        with torch.no_grad():
            g, u = pairs[-1]
            res = ((self.forward(g) - u)[mask]).square().sum()
            denom = u[mask].square().sum().clamp_min(1e-12)
            return float(res / denom)


class EpisodeGains(nn.Module):
    """The episode's `u`: a leaf for the differentiable kinds, a buffer for the
    programme kinds. SPEC: 04 §4.5."""

    def __init__(self, registry: RoleRegistry):
        super().__init__()
        self.reg = registry
        n = registry.n_gamma
        self.register_buffer("diff_mask", registry.diff_of_row.clone())
        self.u_diff = nn.Parameter(torch.zeros(int(registry.diff_of_row.sum())))
        self.register_buffer("u_prog", torch.zeros(n - self.u_diff.numel()))

    @torch.no_grad()
    def set_from(self, u: torch.Tensor) -> None:
        check_shape(u, [self.reg.n_gamma], name="u")
        self.u_diff.copy_(u[self.diff_mask])
        self.u_prog.copy_(u[~self.diff_mask])

    def u(self) -> torch.Tensor:
        """The full `u (n_gamma,)`, differentiable in the differentiable kinds."""
        out = torch.empty(self.reg.n_gamma, device=self.u_diff.device, dtype=self.u_diff.dtype)
        out = out.masked_scatter(self.diff_mask, self.u_diff)
        out = out.masked_scatter(~self.diff_mask, self.u_prog)
        return out

    def gamma(self) -> torch.Tensor:
        return self.reg.gamma(self.u())

    def u_target(self) -> torch.Tensor:
        """The regression target of (4.4): the episode's final `u`, detached."""
        return self.u().detach().clone()


def backbone_gains(registry: RoleRegistry, gamma: torch.Tensor) -> dict[str, torch.Tensor]:
    """The dictionary `Backbone.forward` consumes. SPEC: 04 §4.4.

    Matrix roles give `name:in` and `name:out`; norm roles their gain; the offset
    roles are split per layer (`L{l}_A0` of `h`, `L{l}_G0`, `L{l}_GOG` of 1).
    """
    cfg = registry.cfg
    out = {}
    for r in registry.roles:
        seg = gamma[r.offset:r.offset + r.size]
        if r.kind == "matrix":
            if registry.svf:
                out[r.name + ":svf"] = seg
            else:
                out[r.name + ":in"] = seg[:r.n_in]
                out[r.name + ":out"] = seg[r.n_in:]
        elif r.kind == "norm":
            out[r.name] = seg
    for i, l in enumerate(registry.delta_layers):
        out[f"L{l}_A0"] = gamma[registry.slice("A0")][i * cfg.h:(i + 1) * cfg.h]
        out[f"L{l}_G0"] = gamma[registry.slice("G0")][i * cfg.h:(i + 1) * cfg.h]
    for l in range(cfg.L):
        out[f"L{l}_GOG"] = gamma[registry.slice("GOG")][l:l + 1]
    return out


def lr_multipliers(registry: RoleRegistry, gamma: torch.Tensor) -> list[float]:
    """`exp(lambda_grp)` in `[1/4, 4]`. SPEC: 04 §4.4."""
    seg = gamma[registry.slice("LR")]
    return [float(min(4.0, max(0.25, math.exp(float(v))))) for v in seg]


def mixture(registry: RoleRegistry, gamma: torch.Tensor, floor: float = 0.02) -> torch.Tensor:
    """`mu = floor(softmax(gamma_mu))` over the streams: every entry `>= floor`, sums to 1."""
    p = torch.softmax(gamma[registry.slice("MIX")].detach().cpu().double(), dim=0)     # a CPU vector: it seeds the lanes
    n = p.numel()
    if floor * n > 1.0:
        raise ValueError("floor too large")
    # project onto {p >= floor, sum p = 1}: clip and redistribute the deficit over the free entries
    p = p.clone()
    fixed = torch.zeros(n, dtype=torch.bool)
    for _ in range(n):
        below = (p < floor) & ~fixed
        if not below.any():
            break
        fixed |= below
        p[fixed] = floor
        free = ~fixed
        p[free] = p[free] * (1.0 - floor * int(fixed.sum())) / p[free].sum()   # exact: the free mass is what remains
    return p


def svf_effective(W: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """The SVF ablation (04 §4.8): `U diag(sigma (1 + a)) V^T`."""
    U, s, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return (U * (s * (1.0 + a.float()))[None, :]) @ Vh
