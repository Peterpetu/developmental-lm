"""Routing tokens to units: the fixed slot grid and product-key addressing.

SPEC: 05 §5.4

Each Site owns a preallocated grid of `U_max = side^2` slots. A unit occupies a
slot; `live_u` and `dying_u` are per-slot flags. **The grid size never changes**,
so no tensor is reshaped by a birth or a death and no unit is re-addressed by a
population change. Spec v1 sized the key tables as `sqrt(|U|)`, so adding one
unit changed `sqrt(|U|)` and re-addressed **every** unit, which made Theorem 3.1
false.

```
qk       = x_t W_key                                     in R^{d_key}
s1       = qk[:d_key/2] K_unit1^T / sqrt(d_key/2)        in R^{side}
s2       = qk[d_key/2:] K_unit2^T / sqrt(d_key/2)        in R^{side}
active[i1,i2] <- live and (not dying) and (m_u > 0)
s1, s2  <- masked to -inf on rows/cols with no `active` slot
I1, I2   = topk_{k_route}(s1), topk_{k_route}(s2)
cand     = { (i1,i2) : i1 in I1, i2 in I2, active[i1,i2] }
while |cand| < k_route and I1,I2 can widen:
    widen I1 and I2 by one rank each; recompute cand
R_t      = topk_{min(k_route,|cand|)} over cand of  s1[i1] + s2[i2]      (5.18)
r_ut     = softmax over R_t of ( s1[i1] + s2[i2] )
```

**One predicate**, `active`, governs the row/column mask, `cand`, and the
softmax alike. Spec v2 masked rows on "has a live slot" while excluding newborns
only from `cand`, in a prose paragraph the algorithm block did not implement; a
newborn that was the first live slot in its row would flip that row from `-inf`
to finite, changing `I1` and hence **every other unit's** `r_ut`, so
Theorem 3.1a did not hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from soma.config import SomaConfig
from soma.utils.shapes import check_shape

__all__ = ["Router", "RoutingResult"]


@dataclass
class RoutingResult:
    """The routed set and its weights, in slot-index form.

    `slots` is `(B, T, k_route)` of flat slot indices in `[0, U_max)`, with
    `-1` where fewer than `k_route` candidates existed. `weights` is
    `(B, T, k_route)` with `0` at those positions, so
    `weights.sum(-1) in {0, 1}` — zero exactly when `cand` was empty and the
    Site output is the shared bypass alone (§05.4).
    """

    slots: torch.Tensor
    weights: torch.Tensor
    n_candidates: torch.Tensor
    """`(B, T)` — the realised `|cand|`, logged so the widening path is
    observable rather than inferred."""

    @property
    def any_empty(self) -> torch.Tensor:
        return self.n_candidates == 0


class Router(nn.Module):
    """Product-key routing over a fixed `side x side` slot grid (§05.4)."""

    def __init__(self, cfg: SomaConfig):
        super().__init__()
        self.cfg = cfg
        self.side = cfg.side
        self.U_max = cfg.U_max
        self.k_route = cfg.k_route
        self.d_key = cfg.d_key
        half = cfg.d_key // 2

        self.W_key = nn.Linear(cfg.d_model, cfg.d_key, bias=False)
        # §05.4/§02.1.1: K_unit1, K_unit2 are **soma**. Were they germline,
        # d|Gamma|/d|U_max| would be non-zero and §09.3's bottleneck claim would
        # be false by construction, firing K9 definitionally without any
        # experiment.
        self.K_unit1 = nn.Parameter(torch.empty(cfg.side, half))
        self.K_unit2 = nn.Parameter(torch.empty(cfg.side, half))
        self.scale = 1.0 / math.sqrt(half)
        # **Baseline B3 (§08.4, §05.4):** "routing assignment fixed per unit slot
        # at initialisation and held constant for the run; `r_ut = 1/k_route`;
        # `L_wire` disabled", and §05.4 makes it **required in every routing
        # experiment**.
        #
        # **[NORMATIVE — v3.1 reading, audit A-80]** "fixed at initialisation" is
        # implemented as freezing the routing **function**, not as freezing one
        # assignment for all tokens. The literal alternative — one fixed score
        # per slot, so every token selects the same `k_route` units — leaves
        # `U_max - k_route` units permanently unrouted, and a baseline whose
        # population is dead by construction cannot test whether the population
        # carries information: it assumes the conclusion. Freezing `W_key`,
        # `K_unit1` and `K_unit2` at their random initialisation gives a routing
        # function fixed for the run, carrying no *learned* information, and
        # still exercising the whole population.
        self.random_router = False
        with torch.no_grad():
            nn.init.normal_(self.K_unit1, std=half ** -0.5)
            nn.init.normal_(self.K_unit2, std=half ** -0.5)

    # ------------------------------------------------------------------

    def freeze_for_b3(self) -> None:
        """Enter baseline B3: freeze the routing function and go uniform.

        Called once, before the optimizer is built, so the frozen tensors are
        not merely unused but absent from the parameter groups — a frozen tensor
        still in an optimizer collects weight decay and drifts.
        """
        self.random_router = True
        for p in self.parameters():
            p.requires_grad_(False)

    def scores(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(s1, s2)`, each `(B, T, side)`."""
        qk = self.W_key(x)
        half = self.d_key // 2
        s1 = (qk[..., :half] @ self.K_unit1.transpose(0, 1)) * self.scale
        s2 = (qk[..., half:] @ self.K_unit2.transpose(0, 1)) * self.scale
        return s1, s2

    def forward(
        self,
        x: torch.Tensor,
        active: torch.Tensor,
        *,
        n_active: Optional[int] = None,
    ) -> RoutingResult:
        """Route each token to at most `k_route` active slots.

        `x` is `(B, T, d_model)`; `active` is `(U_max,)` bool — the single
        predicate `live and (not dying) and (m_u > 0)`.

        `n_active` is the population's active count as tracked on the **host**.
        Passing it avoids a `.sum()` / `.any()` GPU->CPU synchronisation per
        Site per step — 12 stalls per forward at M1 — for a number that changes
        only at a selection event, a growth event, or a mask-anneal retirement.
        """
        check_shape(x, [None, None, self.cfg.d_model], name="router input")
        check_shape(active, [self.U_max], name="active")
        B, T, _ = x.shape
        side = self.side
        device = x.device

        s1, s2 = self.scores(x)                                  # (B,T,side)
        grid = active.view(side, side)
        row_has = grid.any(dim=1)                                # (side,)
        col_has = grid.any(dim=0)

        neg_inf = torch.finfo(s1.dtype).min
        s1 = s1.masked_fill(~row_has, neg_inf)
        s2 = s2.masked_fill(~col_has, neg_inf)

        n_act = n_active if n_active is not None else int(active.sum())
        if n_act == 0:
            # No active slot anywhere: R_t is empty for every token and the Site
            # output is the shared bypass alone. Reachable at a Site whose units
            # are all newborn (every m_u = 0), which is precisely the state
            # Theorem 3.1a must survive.
            return RoutingResult(
                torch.full((B, T, self.k_route), -1, dtype=torch.long, device=device),
                torch.zeros(B, T, self.k_route, dtype=x.dtype, device=device),
                torch.zeros(B, T, dtype=torch.long, device=device),
            )

        # --- §05.4's widening, per token, exactly ---------------------------
        #
        # ```
        #   I1, I2 = topk_{k_route}(s1), topk_{k_route}(s2)
        #   cand   = { (i1,i2) : i1 in I1, i2 in I2, active[i1,i2] }
        #   while |cand| < k_route and I1,I2 can widen:
        #       widen I1 and I2 by one rank each; recompute cand
        #   R_t = topk_{min(k_route,|cand|)} over cand of s1[i1] + s2[i2]   (5.18)
        # ```
        #
        # **[NORMATIVE — v3.1, audit A-74]** The widening rank `w_t` is a
        # property of the **token**, not of the Site. v3.0 chose one `w` for the
        # whole batch from an expected-occupancy argument
        # (`w >= sqrt(k_route * U_max / n_active)`) and claimed it "reaches the
        # same fixed point the loop converges to, in one shot". It does not, in
        # either direction. Measured at M-nano over 256 tokens:
        #
        # | `n_active` | `w` used | tokens whose `R_t` differs | tokens left short of `k_route` |
        # |---|---|---|---|
        # | 240 (at cap) | 4 |  0 / 256 |  0 |
        # | 128          | 4 |  3 / 256 |  3 |
        # |  96          | 5 | 113 / 256 | 0 |
        # |  64 (`|U|_0`)| 5 | 85 / 256 | 25 |
        # |  32          | 7 | 123 / 256 | 5 |
        #
        # At `n_active = 64` — the seeded population every M-nano lifetime
        # starts from — a third of tokens were routed to a different unit set
        # than §05.4 specifies, and a tenth to *fewer* than `k_route` units where
        # the loop would have widened and found a full set. `r_ut` is the input
        # to (1.8), (3.5), (4.2) and `Q_route`, so the deviation propagated into
        # the metabolic gradient, the utility, the credit matrix and K4.
        #
        # The loop is implemented without the per-iteration `.all()`
        # synchronisation that motivated the shortcut. Both axes are ranked in
        # full once; `A[w]`, the number of active pairs in the leading `w x w`
        # block, is a 2-D prefix sum whose diagonal gives every `w` at once; and
        # `w_t` is the first index at which it reaches `k_route`. Selection runs
        # detached — only the `k_route` gathered scores carry gradient — so the
        # `(B, T, side, side)` intermediates never enter the autograd graph.
        # `side` is 16 at M-nano and 32 at M1, i.e. 256 or 1024 pairs per token.
        I1 = torch.topk(s1, k=side, dim=-1).indices                 # (B,T,side)
        I2 = torch.topk(s2, k=side, dim=-1).indices

        with torch.no_grad():
            # active[i1, i2] under each token's own ranking of the two axes
            M = grid[I1.unsqueeze(-1), I2.unsqueeze(-2)]            # (B,T,s,s)
            # A[w] = |cand| at rank w+1 (0-based w), by 2-D prefix sum
            A = M.to(torch.int32).cumsum(-1).cumsum(-2).diagonal(
                dim1=-2, dim2=-1
            )                                                       # (B,T,side)
            enough = A >= self.k_route
            # The loop starts at rank `k_route`, so `w_t` is never below it; and
            # it stops at `side`, where `cand` is every active pair there is.
            first = torch.argmax(enough.to(torch.uint8), dim=-1) + 1
            w_t = torch.where(
                enough.any(dim=-1), first, torch.full_like(first, side)
            ).clamp_(min=self.k_route)                              # (B,T)

            ar = torch.arange(side, device=device)
            lim = w_t.unsqueeze(-1).unsqueeze(-1)
            within = (ar.view(-1, 1) < lim) & (ar.view(1, -1) < lim)
            valid = M & within                                      # (B,T,s,s)

            d1 = torch.gather(s1.detach(), -1, I1).unsqueeze(-1)
            d2 = torch.gather(s2.detach(), -1, I2).unsqueeze(-2)
            flat_valid = valid.flatten(2)                           # (B,T,s*s)
            n_cand = flat_valid.sum(-1)                             # (B,T)
            masked = (d1 + d2).flatten(2).masked_fill(~flat_valid, neg_inf)

            k = min(self.k_route, masked.shape[-1])
            top_idx = torch.topk(masked, k=k, dim=-1).indices

            # Flat pair index -> (rank_1, rank_2) -> slot id.
            a_idx = top_idx // side
            b_idx = top_idx % side
            i1 = torch.gather(I1, -1, a_idx)
            i2 = torch.gather(I2, -1, b_idx)
            slots = i1 * side + i2                                   # (B,T,k)

        # Only these `k_route` scores carry gradient, and they are recomputed
        # from `s1`/`s2` rather than gathered out of a detached tensor, so
        # `W_key` and `K_unit*` receive exactly the gradient (5.18) implies.
        top_vals = torch.gather(s1, -1, i1) + torch.gather(s2, -1, i2)

        selected_valid = torch.gather(flat_valid, -1, top_idx)
        top_vals = top_vals.masked_fill(~selected_valid, neg_inf)

        # Softmax over R_t only. Dead, masked and dying slots score -inf in
        # **both** the top-k and the softmax, which is what makes Theorem 3.1a
        # true: a newborn neither displaces a real unit from the selection nor
        # enters the normalisation.
        if self.random_router:
            # B3: `r_ut = 1/k_route`, uniform over the **realised** routed set —
            # a token routed to fewer than `k_route` units must still have
            # weights summing to 1, exactly as the learned softmax does. Applied
            # here because with untrained scores a softmax would report the
            # arbitrary scale of a random projection as a routing weight.
            n_sel = selected_valid.sum(-1, keepdim=True).clamp_min(1)
            weights = selected_valid.to(x.dtype) / n_sel.to(x.dtype)
        else:
            weights = torch.softmax(top_vals.float(), dim=-1).to(x.dtype)
        weights = weights * selected_valid.to(weights.dtype)
        slots = slots.masked_fill(~selected_valid, -1)

        if k < self.k_route:
            pad = self.k_route - k
            slots = torch.cat(
                [slots, torch.full((B, T, pad), -1, dtype=slots.dtype, device=device)],
                dim=-1,
            )
            weights = torch.cat(
                [weights, torch.zeros(B, T, pad, dtype=weights.dtype, device=device)],
                dim=-1,
            )

        return RoutingResult(slots, weights, n_cand)
