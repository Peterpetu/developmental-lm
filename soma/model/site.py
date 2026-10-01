"""The Site: the unit population's forward pass.

SPEC: 05 §5.5
SPEC: 01 §1.3
SPEC: 01 §1.4
SPEC: 02 §2.4

```
z          = A x_t                                                in R^{N_G}  (5.19)
psi(x_t)   = sigmoid( kappa ( x_t W_ctx ) / sqrt(d_key) )         in [0,1]^{N_G}
gate_u     = phi(p_u) (*) psi(x_t) (*) 1_{g_u}                    k_g-sparse
Site(x_t)  = sum_{u in R_t} r_ut * m_u * B [ relu( gate_u (*) z )^2 ]
             + W_shared x_t                                       in R^{d_model} (5.20)
```

**The activation is applied AFTER the gate.** This is the correction the second
audit forced (A-8). With the activation *before* the gate, `B` and the gate are
both linear, so

    sum_u r_ut m_u B[ gate_u (*) relu(z)^2 ]
      == B[ (sum_u r_ut m_u gate_u) (*) relu(z)^2 ]

**exactly** — every unit would be a linear reweighting of one shared nonlinear
feature, and the forbidden aggregate form would compute bit-for-bit the same
function at `1/k_route` of the cost. The post-gate placement costs nothing,
because `gate_u` is `k_g`-sparse and therefore so is `relu(gate_u (*) z)^2`.

Because the claim was asserted once and was wrong, the corrected claim carries a
mandatory ablation: **A12** runs the pre-gate placement at exact parameter
parity, and `pre_gate=True` below is what it flips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from soma.config import SomaConfig
from soma.germline.field import Field
from soma.model.routing import Router, RoutingResult
from soma.utils.shapes import check_shape

__all__ = ["PartsLibrary", "Site", "SiteOutput"]


class PartsLibrary(nn.Module):
    """`A`, `B`, `kappa` — shared across all Sites (§05.5).

    SPEC: 05 §5.5

    There is **no `d_ff`**: the Site's inner width *is* `N_G`, so the gene
    library plays the role of the feed-forward inner dimension and `B` maps gene
    activations straight back to the residual stream. That is what makes
    `d_in = d_out = d_model` and removes the output projection spec v1 needed.

    Shared across Sites because a controlled cross-layer sharing study found
    that sharing attention costs approximately nothing while sharing
    feed-forward parameters is where the damage concentrates — so the
    differentiation budget is spent at the Site, in `W_shared` and in the unit
    population, not in a per-Site copy of the parts.
    """

    def __init__(self, cfg: SomaConfig):
        super().__init__()
        self.cfg = cfg
        self.A = nn.Parameter(torch.empty(cfg.N_G, cfg.d_model))
        self.B = nn.Parameter(torch.empty(cfg.d_model, cfg.N_G))
        self.kappa = nn.Parameter(torch.empty(cfg.N_G, cfg.d_key))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """(1.6): `Var(a_g) = 1 / d_in`. `Var(b_g)` is fitted by CALIBRATE."""
        with torch.no_grad():
            nn.init.normal_(self.A, std=self.cfg.d_model ** -0.5)
            nn.init.normal_(self.B, std=self.cfg.N_G ** -0.5)
            nn.init.normal_(self.kappa, std=self.cfg.d_key ** -0.5)


@dataclass
class SiteOutput:
    """The Site's output and the per-unit quantities selection needs."""

    y: torch.Tensor
    """`(B, T, d_model)` — (5.20)."""
    delta_norm: torch.Tensor
    """`(B, T, k_route)` — `||delta_ut||_2`, the per-unit contribution norm
    required by (4.2)'s credit and (3.5)'s utility. §01.4: even were the
    aggregate form equivalent, it would destroy the signal selection runs on."""
    slots: torch.Tensor
    """`(B, T, k_route)` — flat slot indices, `-1` where unrouted."""
    r_ut: torch.Tensor
    """`(B, T, k_route)` — routing weights, exactly as §05.4 defines them:
    `softmax over R_t of (s1[i1] + s2[i2])`, with **no** `m_u` factor.

    **[NORMATIVE — v3.1, audit A-76]** v3.0 returned `r_ut * m_u` here, folding
    the mask into the quantity every downstream consumer reads. Only (1.8)
    wants that product; (4.2) and §08.1.1's `Q_route` are written on `r_ut`
    alone, and a newborn's credit was therefore scaled by `m_u in (0,1)` for the
    whole of its `phi_ann`-step anneal — up to 50 of the 99 steps of an M-nano
    selection interval, systematically under-crediting exactly the units
    ε-lexicase is deciding whether to keep.

    (3.5) is unaffected either way: `h_u` is a **ratio** whose numerator and
    denominator both carry the same per-unit `m_u`, so it cancels identically.
    That is why the defect survived — two of its three consumers were correct.

    `m_u` is available separately as `mask_u` for the one consumer that needs
    it."""
    mask_u: torch.Tensor
    """`(B, T, k_route)` — `m_u` for each routed slot, `0` where unrouted.

    Kept beside `r_ut` rather than folded into it, so (1.8) can form
    `r_ut * m_u * ||q_t - p_u||^2` and nothing else has to."""
    z: torch.Tensor
    """`(B, T, N_G)` — `A x_t`, retained for the `erank` probe (§03.4)."""
    n_candidates: torch.Tensor


class Site(nn.Module):
    """One Site: the routed unit population plus the shared bypass (§05.5)."""

    def __init__(self, cfg: SomaConfig, parts: PartsLibrary):
        super().__init__()
        self.cfg = cfg
        self.parts = parts
        self.router = Router(cfg)
        self.W_shared = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_ctx = nn.Linear(cfg.d_model, cfg.d_key, bias=False)
        # §01.6: q_t = normalize(x_t W_pos), used only by L_wire and L_disp.
        self.W_pos = nn.Linear(cfg.d_model, cfg.m_dim, bias=False)
        self.psi_scale = 1.0 / math.sqrt(cfg.d_key)
        # **Ablation A10 (§05.5, §08.6):** `psi == 1`. The context gate is
        # removed and `gate_u` reduces to `phi(p_u) (*) 1_{g_u}`, so a unit's
        # expression stops depending on the token. §01.8 lists this as a
        # falsifier of the Moment regime; §05.11 makes it the context gate's.
        self.psi_off = False
        with torch.no_grad():
            nn.init.normal_(self.W_shared.weight, std=cfg.d_model ** -0.5)
            nn.init.normal_(self.W_ctx.weight, std=cfg.d_model ** -0.5)
            nn.init.normal_(self.W_pos.weight, std=cfg.d_model ** -0.5)

    # ------------------------------------------------------------------

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        """`sigmoid( kappa (x W_ctx) / sqrt(d_key) )` in `[0,1]^{N_G}` (§05.5).

        Identically `1` under ablation A10, which removes the context gate from
        `gate_u` without changing any shape.
        """
        if self.psi_off:
            return torch.ones(
                x.shape[0], x.shape[1], self.parts.kappa.shape[0],
                dtype=x.dtype, device=x.device,
            )
        c = self.W_ctx(x)                                     # (B,T,d_key)
        return torch.sigmoid((c @ self.parts.kappa.transpose(0, 1)) * self.psi_scale)

    def positions_query(self, x: torch.Tensor) -> torch.Tensor:
        """`q_t = normalize(x_t W_pos)` in `S^{m_dim-1}` (§01.6, §05.4)."""
        q = self.W_pos(x)
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        *,
        field: Field,
        p_u: torch.Tensor,
        g_u: torch.Tensor,
        m_u: torch.Tensor,
        active: torch.Tensor,
        pre_gate: bool = False,
        routing: Optional[RoutingResult] = None,
        keep_z: bool = True,
        n_active: Optional[int] = None,
    ) -> SiteOutput:
        """(5.19)-(5.20).

        Parameters
        ----------
        p_u : `(U_max, m_dim)` — unit positions (soma).
        g_u : `(U_max, k_g)` — expressed gene indices (germline-addressed).
        m_u : `(U_max,)` — MSG masks in `[0, 1]`.
        active : `(U_max,)` bool — `live and not dying and m_u > 0`.
        pre_gate : ablation **A12**. Applies `relu(.)^2` to `z` *before* the
            gate, which §01.4 proves makes every unit a linear reweighting of
            one shared nonlinear feature. Present so the corrected claim is
            testable rather than merely asserted.
        """
        cfg = self.cfg
        check_shape(x, [None, None, cfg.d_model], name="site input")
        check_shape(p_u, [cfg.U_max, cfg.m_dim], name="p_u")
        check_shape(g_u, [cfg.U_max, cfg.k_g], name="g_u")
        check_shape(m_u, [cfg.U_max], name="m_u")
        B, T, _ = x.shape

        # (5.19): z = A x_t, computed once per token and shared by every routed
        # unit. This is the term that dominates Theorem 1's cost.
        z = x @ self.parts.A.transpose(0, 1)                    # (B,T,N_G)
        psi = self.psi(x)                                       # (B,T,N_G)

        route = (
            routing if routing is not None
            else self.router(x, active, n_active=n_active)
        )
        slots, r = route.slots, route.weights                   # (B,T,k)
        k_route = slots.shape[-1]

        safe_slots = slots.clamp_min(0)
        genes = g_u[safe_slots]                                 # (B,T,k,k_g)
        phi_u = field.phi_at(
            p_u, g_u
        )                                                       # (U_max,k_g)
        phi_sel = phi_u[safe_slots]                             # (B,T,k,k_g)
        mask_u = m_u[safe_slots]                                # (B,T,k)

        # Gather only the `k_g` coordinates each unit expresses. This is what
        # makes the cost `O(k_route k_g d_out)` rather than `O(k_route N_G)`:
        # `gate_u` is `k_g`-sparse, so materialising the full `N_G`-vector would
        # defeat Theorem 1 while computing the same thing.
        #
        # The gather is taken from the **unexpanded** `(B, T, N_G)` tensor.
        # Writing it as `gather(z.unsqueeze(2).expand(B,T,k,N_G), 3, genes)` is
        # free in the forward -- `expand` is a stride-0 view -- but autograd's
        # backward scatters into a buffer of the **expanded** shape:
        # `B*T*k_route*N_G` floats, **537 MB per Site** at M1 with 16 rows, and
        # 6.4 GB across the stack. Gathering along `dim=2` of the real tensor
        # scatters into `(B, T, N_G)` instead -- a `k_route`-fold reduction.
        flat_genes = genes.reshape(B, T, k_route * cfg.k_g)
        z_sel = torch.gather(z, 2, flat_genes).reshape(B, T, k_route, cfg.k_g)
        psi_sel = torch.gather(psi, 2, flat_genes).reshape(B, T, k_route, cfg.k_g)

        gate = phi_sel * psi_sel                                # (B,T,k,k_g)

        if pre_gate:
            # A12: the placement §01.4 proves is degenerate. Retained so the
            # corrected claim carries a test.
            act = gate * torch.relu(z_sel).square()
        else:
            act = torch.relu(gate * z_sel).square()

        # B applied to a `k_g`-sparse vector touches `k_g` columns, so
        #     delta[b,t,u] = sum_g act[b,t,u,g] * B[:, g_u[u,g]]
        # is a weighted sum of `k_g` columns of `B`. Computing it as
        # `einsum("btkg,btkgd->btkd", act, B.T[genes])` first *materialises* the
        # gather `B.T[genes]`, which is `B*T*k_route*k_g*d_model` floats -- at M1
        # with 16 rows that is **1.07 GB per Site**, and it is retained for the
        # backward pass across all 12 Sites: 12.9 GB, which does not fit beside
        # the rest of the model in 16 GB.
        #
        # `embedding_bag` computes exactly the same weighted sum without ever
        # materialising the gather, so the intermediate is `M x d_model` rather
        # than `M x k_g x d_model` -- an 8x reduction at `k_g = 8`. It is the
        # same arithmetic, and `tests/unit/test_backbone.py` pins the two forms
        # against each other.
        M = B * T * k_route
        delta = torch.nn.functional.embedding_bag(
            genes.reshape(M, cfg.k_g),
            self.parts.B.transpose(0, 1),
            per_sample_weights=act.reshape(M, cfg.k_g),
            mode="sum",
        ).reshape(B, T, k_route, cfg.d_model)

        delta_norm = delta.norm(dim=-1)                         # (B,T,k)

        weight = (r * mask_u).unsqueeze(-1)                     # (B,T,k,1)
        routed = (weight * delta).sum(dim=2)                    # (B,T,d_model)

        # §05.5: W_shared x_t sits **outside** the routed sum. Spec v1 placed it
        # inside, giving it the token-dependent coefficient sum_u r_ut m_u in
        # [0,1], so the shared bypass was silently attenuated during every birth
        # anneal and vanished entirely at a Site whose units were all newborn.
        y = routed + self.W_shared(x)

        check_shape(y, [B, T, cfg.d_model], name="site output")
        # `z` is `(B, T, N_G)` -- 67 MB per Site at M1 with 16 rows, 0.8 GB
        # across the stack -- and only the `erank` probe of §03.4 reads it, at a
        # cadence of `10 * T_log`. Retaining it on every step is 0.8 GB of the
        # 16 GB budget spent on a diagnostic that runs on 0.2% of steps.
        # `r_ut` is returned **unmasked** (audit A-76): (4.2) and `Q_route`
        # are written on the routing weight, not on the weight times the anneal
        # mask. `mask_u` travels beside it for (1.8), which is the only consumer
        # of the product.
        return SiteOutput(
            y, delta_norm, slots, r, mask_u * (slots >= 0).to(mask_u.dtype),
            z if keep_z else z.detach()[:0], route.n_candidates,
        )
