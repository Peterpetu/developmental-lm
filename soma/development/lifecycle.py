"""Birth, death, growth, and the utility that decides deaths.

SPEC: 03 §3.2
SPEC: 03 §3.4
SPEC: 03 §3.4.1
SPEC: 03 §3.5
SPEC: 03 §3.5.1

Development is the change of the unit population `U_l` and of gene sets `g_u`
over a single lifetime. It runs at rate `r_2`, it is discrete, and it is
therefore the exclusive province of selection.

**[NORMATIVE — a negative result we obey]** Multi-step *generative* growth
processes, in which a program is unrolled for developmental steps to produce a
network, were tested against a one-shot decoding baseline and showed **no
benefit**; the same line reports that such programs often contain more
parameters than the networks they produce, inverting the bottleneck. SOMA
therefore implements **no** temporally-unrolled morphogenesis program.
Development here is birth and death of units under selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from soma.config import SomaConfig
from soma.development.population import LineageRecord, UnitPopulation
from soma.development.variation import crossover, mutate
from soma.germline.field import Field
from soma.model.metabolic import budget_headroom

__all__ = [
    "birth",
    "death",
    "grow",
    "update_utility",
    "calibrate_tau_death",
    "nu_raw",
    "nu_hat",
    "UtilityState",
]


# ---------------------------------------------------------------------------
# Utility (§03.4.1)
# ---------------------------------------------------------------------------


def nu_raw(pop: UnitPopulation, eta_ema: float) -> torch.Tensor:
    """Bias-corrected utility `nu_u / (1 - eta_ema^{age_u})` (3.8, numerator).

    SPEC: 03 §3.4.1

    This is the quantity `tau_death` is calibrated on and compared against.
    (3.8) then divides by the population median, and **that** normalised form
    has median exactly 1 at every step by construction — so a "fixed" threshold
    on it still selects a fixed fraction, which is the extinction behaviour v1's
    correction note claimed to have removed (audit A-18). Only the un-normalised
    quantity can carry an absolute threshold.
    """
    age = pop.age_u.clamp_min(1).float()
    correction = (1.0 - eta_ema ** age).clamp_min(1e-8)
    return pop.nu_u / correction


def nu_hat(pop: UnitPopulation, eta_ema: float) -> torch.Tensor:
    """(3.8): `nu_raw` normalised by the median over live, mature units.

    Used by (3.1)'s somatic hypermutation and by ablation A2's mean-utility
    selection. `nu_hat = 0` at `age_u = 0` by definition (the expression is
    `0/0` otherwise).
    """
    raw = nu_raw(pop, eta_ema)
    mature = pop.U_death
    if not bool(mature.any()):
        return torch.zeros_like(raw)
    med = raw[mature].median().clamp_min(1e-8)
    out = raw / med
    out[pop.age_u == 0] = 0.0
    return out


@dataclass
class UtilityState:
    """Per-step accumulators feeding (3.5)-(3.7)."""

    sum_r_delta: torch.Tensor
    """`sum_t r_ut ||delta_ut||_2` per slot."""
    sum_r: torch.Tensor
    """`sum_t r_ut` per slot."""

    @classmethod
    def zeros(cls, U_max: int, device) -> "UtilityState":
        return cls(torch.zeros(U_max, device=device), torch.zeros(U_max, device=device))

    def accumulate(
        self, slots: torch.Tensor, r_ut: torch.Tensor, delta_norm: torch.Tensor
    ) -> None:
        """Scatter one batch's routed contributions into the per-slot sums."""
        valid = slots >= 0
        flat_slots = slots[valid]
        if flat_slots.numel() == 0:
            return
        self.sum_r_delta.index_add_(0, flat_slots, (r_ut * delta_norm)[valid].float())
        self.sum_r.index_add_(0, flat_slots, r_ut[valid].float())

    def reset(self) -> None:
        self.sum_r_delta.zero_()
        self.sum_r.zero_()


@torch.no_grad()
def update_utility(
    pop: UnitPopulation,
    acc: UtilityState,
    field: Field,
    parts,
    eta_ema: float,
    eps: float = 1e-8,
) -> None:
    """(3.5)-(3.7), applied once per optimizer step.

    SPEC: 03 §3.4.1

    ```
    h_u      = sum_t r_ut ||delta_ut||_2 / max(eps, sum_t r_ut)                (3.5)
    h_bar_u  = eta_ema * h_bar_u + (1 - eta_ema) * h_u                        (3.5b)
    out_u    = sum_{g in g_u} phi_g(p_u) ||b_g||_1
    in_u     = sum_{g in g_u} phi_g(p_u) ||a_g||_1
    y_u      = |h_u - h_bar_u| * out_u / max(eps, in_u)                        (3.6)
    nu_u    <- eta_ema * nu_u + (1 - eta_ema) * y_u                            (3.7)
    ```

    (3.5b) is new in v3: `h_bar_u` was used in (3.6) and listed in the symbol
    table but **defined nowhere** in v2, so the entire death rule was not
    implementable (audit A-19). Defining it as a per-unit EMA of `h_u` makes
    `y_u` measure a unit's *deviation from its own recent contribution*, which
    is what makes it a novelty signal rather than a magnitude signal — a unit
    that contributes steadily is not thereby fit; a unit whose contribution is
    changing is doing something.

    Spec v1's `y_u` was **not computable**: a unit owns no weights (it owns
    `p_u` and a gene index set into the shared `a_g, b_g`), `h_u` was a vector
    with no norm specified, and the result had no normalisation, so it
    contaminated (3.1)'s `sigma_mut_0/(1 + nu_hat)` with an arbitrary activation
    scale.
    """
    h_u = acc.sum_r_delta / acc.sum_r.clamp_min(eps)

    phi = field.phi_at(pop.p_u.data, pop.g_u)                # (U_max, k_g)
    b_l1 = parts.B.abs().sum(dim=0)                          # (N_G,) column L1
    a_l1 = parts.A.abs().sum(dim=1)                          # (N_G,) row  L1
    out_u = (phi * b_l1[pop.g_u]).sum(dim=-1)
    in_u = (phi * a_l1[pop.g_u]).sum(dim=-1)

    y_u = (h_u - pop.h_bar_u).abs() * out_u / in_u.clamp_min(eps)

    live = pop.live
    pop.h_bar_u[live] = eta_ema * pop.h_bar_u[live] + (1.0 - eta_ema) * h_u[live]
    pop.nu_u[live] = eta_ema * pop.nu_u[live] + (1.0 - eta_ema) * y_u[live]


@torch.no_grad()
def calibrate_tau_death(pop: UnitPopulation, eta_ema: float) -> float:
    """Calibrate `tau_death` **once**, at the first selection event (§04.4).

    SPEC: 04 §4.4

    The 10th percentile of the **un-normalised** bias-corrected utility over
    `U_death`, thereafter fixed for the lifetime.

    Spec v1 recomputed a percentile every event, guaranteeing a constant 10%
    kill rate whether or not any unit was unfit — deaths structurally exceeded
    births from the first event and §04.2's postcondition was violated
    immediately. v2 made it absolute but stated it on `nu_hat`, which has median
    exactly 1 by construction, so a fixed threshold on it still selects a fixed
    fraction (audit A-18).
    """
    if pop.tau_death is not None:
        return pop.tau_death
    eligible = pop.U_death
    if not bool(eligible.any()):
        pop.tau_death = 0.0
        return 0.0
    raw = nu_raw(pop, eta_ema)[eligible]
    pop.tau_death = float(torch.quantile(raw.float(), 0.10))
    return pop.tau_death


# ---------------------------------------------------------------------------
# Birth (§03.2)
# ---------------------------------------------------------------------------


@torch.no_grad()
def birth(
    pop: UnitPopulation,
    parent1: int,
    parent2: Optional[int],
    n: int,
    *,
    cfg: SomaConfig,
    field: Field,
    neighbour_table: torch.Tensor,
    rng: np.random.Generator,
    generator: torch.Generator,
    run: str = "run",
    G_gen: int = 0,
    parent_nu_hat: float = 0.0,
) -> int:
    """`BIRTH` of §03.2. Returns the child's slot.

    ```
    precondition: u_p1 live and not dying ; BUDGET_HEADROOM(n_death) >= 1 ;
                  a free slot exists in the grid
    slot_child <- FREE_SLOT_NEAREST(slot(u_p1))
    p_child    <- normalize( p_{u_p1} + xi ),  xi ~ N(0, sigma_pos^2 I_{m_dim})
    g_child    <- MUTATE(g_{u_p1}, nu_hat_{u_p1})   if u_p2 is None
                  CROSSOVER(g_{u_p1}, g_{u_p2})     otherwise
    m_child <- 0 ; age_child <- 0 ; nu_child <- 0
    postcondition: m_child = 0, so by Theorem 3.1a the phenotype is unchanged
    ```

    `nu_child = 0`, not the parent's value: inheriting it invalidates the bias
    correction of (3.8) (whose derivation assumes initialisation at zero) and
    inflates young utilities by up to `100x`. `M_mat` provides the protection
    instead.

    The child does **not** inherit key-table rows. §03.2 v3.1 (audit A-50): a
    `K_unit1` row is shared by every slot in its grid row, so writing it would
    change every other unit's `r_ut` and make Theorem 3.1a false at the moment
    of birth. What the child inherits is `p_parent + xi` and `MUTATE(g_parent)`,
    which determine its *function*; its routing is deliberately fresh, which is
    what makes siblings differentiate rather than duplicate.
    """
    if not bool(pop.live[parent1]) or bool(pop.dying[parent1]):
        raise ValueError(f"parent slot {parent1} is not live-and-not-dying")

    slot_child = pop.free_slot_nearest(parent1)

    xi = torch.randn(
        cfg.m_dim, generator=generator, device=pop.p_u.device, dtype=pop.p_u.dtype
    ) * cfg.policy.sigma_pos
    p_child = pop.p_u.data[parent1] + xi
    p_child = p_child / p_child.norm().clamp_min(1e-6)

    g_parent = pop.g_u[parent1].tolist()
    if parent2 is None:
        nearest = _nearest_live_genes(pop, p_child, exclude=parent1)
        g_child, _outcome = mutate(
            g_parent,
            parent_nu_hat,
            policy=cfg.policy,
            N_G=cfg.N_G,
            neighbour_table=neighbour_table,
            rng=rng,
            nearest_unit_genes=nearest,
        )
    else:
        g_child = crossover(g_parent, pop.g_u[parent2].tolist(), rng=rng)

    pop.p_u.data[slot_child] = p_child
    pop.g_u[slot_child] = torch.tensor(
        sorted(g_child), dtype=torch.long, device=pop.g_u.device
    )
    pop.live[slot_child] = True
    pop.dying[slot_child] = False
    pop.m_u[slot_child] = 0.0
    pop.age_u[slot_child] = 0
    pop.nu_u[slot_child] = 0.0
    pop.h_bar_u[slot_child] = 0.0
    pop.anneal_dir[slot_child] = 1
    pop.anneal_start[slot_child] = n
    pop.offspring_this_event[parent1] += 1

    parent_rec = pop.lineage.get(parent1)
    depth = (parent_rec.depth + 1) if parent_rec is not None else 1
    pop.lineage[slot_child] = LineageRecord(
        run, G_gen, parent1, parent2, n, slot_child, depth
    )
    return slot_child


def _nearest_live_genes(
    pop: UnitPopulation, p_child: torch.Tensor, exclude: int
) -> Optional[List[int]]:
    """Genes of the live unit nearest to `p_child` (§03.2.1's `dup` kind).

    Returns `None` when no such unit exists, which triggers `MUTATE`'s explicit
    `neighbour(i)` fallback — the fallback spec v1 omitted, leaving the rule
    undefined at the first birth into an empty neighbourhood.
    """
    live = pop.live.clone()
    live[exclude] = False
    idx = torch.nonzero(live, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return None
    d = (pop.p_u.data[idx] - p_child.unsqueeze(0)).norm(dim=-1)
    return pop.g_u[idx[int(torch.argmin(d))]].tolist()


# ---------------------------------------------------------------------------
# Death (§03.4)
# ---------------------------------------------------------------------------


@torch.no_grad()
def death(pop: UnitPopulation, slot: int, n: int) -> None:
    """`DEATH` of §03.4.

    ```
    precondition: live_u and not dying_u ; age_u > M_mat ;
                  u not selected as a parent this event ; nu_raw_u < tau_death
    1. dying_u <- True ; schedule m_u to decay linearly to 0 over phi_ann steps
    2. while dying: u is excluded from cand, from parents, from deaths,
       from n_par, and from the tau_death calibration set
    3. on m_u reaching 0: live_u <- False ; zero p_u and its optimizer state;
       retain L_u in the lineage log
    ```

    Spec v1's death step transferred a unit's average contribution "into the
    bias of each consumer". **The model has no biases** (§05.1), and "consumer
    of `u`" is not an object in this architecture — units contribute additively
    through the shared `B`. The step is **deleted**. Death is therefore
    *approximately*, not exactly, function-preserving; the mask anneal is what
    makes the change gradual, and consequently the bifurcation alarm applies to
    **births only** (§01.7.2).
    """
    if not bool(pop.live[slot]) or bool(pop.dying[slot]):
        raise ValueError(f"slot {slot} is not live-and-not-dying")
    pop.dying[slot] = True
    pop.anneal_dir[slot] = -1
    pop.anneal_start[slot] = n


# ---------------------------------------------------------------------------
# Growth (§03.5.1)
# ---------------------------------------------------------------------------


@torch.no_grad()
def grow(
    pop: UnitPopulation,
    n: int,
    *,
    cfg: SomaConfig,
    field: Field,
    neighbour_table: torch.Tensor,
    rng: np.random.Generator,
    generator: torch.Generator,
    run: str = "run",
    G_gen: int = 0,
) -> List[int]:
    """`GROW` of §03.5.1. Returns the slots created.

    ```
    precondition: n in growth_schedule ; BUDGET_HEADROOM(0) > 0
    r_left <- number of growth events remaining, including this one
    n_new  <- min( floor((g_growth - 1) * |U_live|),
                   floor( BUDGET_HEADROOM(0) / r_left ) )
    for j in 1..n_new:
        parent <- sampled from U_live with probability proportional to nu_hat
        BIRTH(parent, None, n)
    postcondition: every new unit has m_u = 0, so Theorem 3.1a applies to each
    ```

    Dividing the remaining headroom by the number of remaining events is what
    makes the schedule length matter. v2 sized an event as
    `min(floor((g_growth-1)|U_live|), BUDGET_HEADROOM())`, which with
    `g_growth = 4` requests `3|U_live|` new units and **saturates `U_cap` on the
    very first event** at both configurations; the 2nd and 3rd scheduled events
    then violate `GROW`'s own precondition, so all three arms of ablation
    **A8** would have produced an identical trajectory (audit A-21).
    """
    schedule = cfg.growth_schedule
    if n not in schedule:
        raise ValueError(f"step {n} is not in growth_schedule {schedule}")
    r_left = sum(1 for s in schedule if s >= n)

    headroom = budget_headroom(
        cfg.policy.U_cap, pop.n_live, 0,
        n_dying=pop.n_dying, U_max=cfg.U_max,
    )
    if headroom <= 0:
        return []

    n_new = min(
        int((cfg.policy.g_growth - 1.0) * pop.n_live),
        headroom // max(1, r_left),
        int((~pop.live).sum()),
    )
    if n_new <= 0:
        return []

    eligible = torch.nonzero(pop.U_parent, as_tuple=False).squeeze(-1)
    if eligible.numel() == 0:
        return []
    scores = nu_hat(pop, cfg.policy.eta_ema)[eligible].clamp_min(0.0).cpu().numpy()
    if scores.sum() <= 0:
        probs = np.full(len(scores), 1.0 / len(scores))
    else:
        probs = scores / scores.sum()

    nh = nu_hat(pop, cfg.policy.eta_ema)
    created: List[int] = []
    for _ in range(n_new):
        if int((~pop.live).sum()) == 0:
            break
        j = int(rng.choice(len(eligible), p=probs))
        parent = int(eligible[j])
        if not bool(pop.live[parent]) or bool(pop.dying[parent]):
            continue
        created.append(
            birth(
                pop, parent, None, n,
                cfg=cfg, field=field, neighbour_table=neighbour_table,
                rng=rng, generator=generator, run=run, G_gen=G_gen,
                parent_nu_hat=float(nh[parent]),
            )
        )
    assert int((pop.live & ~pop.dying).sum()) <= cfg.policy.U_cap, (
        "GROW exceeded U_cap"
    )
    assert pop.n_live <= cfg.U_max, "GROW exceeded the slot grid"
    return created
