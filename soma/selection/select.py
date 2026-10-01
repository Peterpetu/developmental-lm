"""`SELECT`: one selection event.

SPEC: 04 §4.2
SPEC: 04 §4.3

```
Algorithm SELECT(U_l, buffer, n) -> (parents, deaths, births)
  precondition: n > 0 and n >= max(T_0, M_mat)          # warmup guard
                |{u : live_u and not dying_u}| >= 2
                buffer holds >= D_min scored cases

  1. cases   <- buffer.sample(q_cases)
  2. nu_ut   <- credit(u, t) for u in U_parent, t in cases            (4.2)
  3. eps_t   <- per-case MAD over { u : nu_ut > 0 }, else 0           (4.6)-(4.7)
  4. parents <- EPSILON_LEXICASE(U_parent, cases, nu, eps, w, n_par)
  5. n_death <- min( floor(rho(n) * |U_live|),
                     |{ u in U_death : u not in parents and nu_raw_u < tau_death }| )
     deaths  <- the n_death such units with lowest nu_raw
  6. n_birth <- min( n_par, floor(rho(n)*|U_live|), BUDGET_HEADROOM(n_death) )
     births  <- n_birth draws uniformly from the parents multiset,
                subject to cap_off offspring per unit
  7. apply deaths (§03.4) and births (§03.2)

  postcondition: | |U_live'| - |U_live| | <= floor(rho(n) * |U_live|)
                 n_birth is computed net of n_death
                 no unit with age_u <= M_mat was killed
                 no dying unit appears in parents, deaths, or the tau_death set
```

**The deliberate asymmetry.** Births are decided **case-by-case**; deaths by
**aggregate** utility plus failure to be selected. A unit that is best at nothing
is expendable, so aggregation is safe for removal; it is not safe for
reproduction, which is exactly where averaging destroys specialists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from soma.config import SomaConfig
from soma.development.lifecycle import (
    birth,
    calibrate_tau_death,
    death,
    nu_hat,
    nu_raw,
)
from soma.development.population import UnitPopulation
from soma.model.metabolic import budget_headroom
from soma.selection.buffer import CaseBatch, CaseReservoir
from soma.selection.lexicase import (
    LexicaseResult,
    case_win_histogram,
    credit_matrix,
    epsilon_lexicase,
    epsilon_per_case,
    mean_utility_parents,
)

__all__ = ["SelectionEvent", "select"]


@dataclass
class SelectionEvent:
    """What one event did, for the log and for the postcondition assertions."""

    step: int
    site: int
    n_cases: int
    parents: List[int]
    deaths: List[int]
    births: List[int]
    eps_zero_fraction: float
    mean_pool_after_first_case: float
    mean_cases_used: float
    cap_bound: bool
    """Whether `cap_off` bound at least one parent this event (§04.6, §04.10)."""
    n_live_before: int
    n_live_after: int
    tau_death: float
    w_sum: float
    case_wins: Dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "site": self.site,
            "n_cases": self.n_cases,
            "n_parents": len(self.parents),
            "n_deaths": len(self.deaths),
            "n_births": len(self.births),
            "eps_zero_fraction": self.eps_zero_fraction,
            "mean_pool_after_first_case": self.mean_pool_after_first_case,
            "mean_cases_used": self.mean_cases_used,
            "cap_bound": self.cap_bound,
            "n_live_before": self.n_live_before,
            "n_live_after": self.n_live_after,
            "tau_death": self.tau_death,
            "w_sum": self.w_sum,
        }


def select(
    pop: UnitPopulation,
    reservoir: CaseReservoir,
    n: int,
    *,
    cfg: SomaConfig,
    field_,
    neighbour_table: torch.Tensor,
    rng_lexicase: np.random.Generator,
    rng_mutation: np.random.Generator,
    rng_downsample: np.random.Generator,
    generator: torch.Generator,
    rho_scale: float = 1.0,
    site: int = 0,
    run: str = "run",
    G_gen: int = 0,
    use_mean_utility: bool = False,
) -> Optional[SelectionEvent]:
    """One selection event (§04.2). Returns `None` if a precondition is unmet.

    `use_mean_utility` selects ablation **A2**, which replaces **both** the
    birth and the death rule with top-`n_par` by `nu_hat`.
    """
    policy = cfg.policy

    # ---- preconditions (§04.2) -------------------------------------------
    if n <= 0 or n < max(cfg.T_0, policy.M_mat):
        return None
    parent_mask = pop.U_parent
    if int(parent_mask.sum()) < 2:
        return None
    if len(reservoir) < 8:  # D_min
        return None

    n_live_before = pop.n_live

    # ---- 1. cases ---------------------------------------------------------
    # `w_t` was evaluated at push time, on the rows the reservoir was offered
    # (§04.6 v3.1, audit A-53), so all three factors of (4.2) come from the same
    # moment on the same token.
    cases = reservoir.sample(cfg.q_cases, rng_downsample)
    q = len(cases)
    w_t = cases.w_t

    # ---- 2-3. credit and epsilon -----------------------------------------
    nu = credit_matrix(cases.slots, cases.delta_norm, cases.r_ut, w_t, cfg.U_max)
    eps = epsilon_per_case(nu)

    eligible = torch.nonzero(parent_mask, as_tuple=False).squeeze(-1).tolist()

    # The case-win histogram is computed **before** any operator runs, so it
    # exists identically in both arms of A2 (§04.10).
    wins = case_win_histogram(eligible, nu, eps)

    # ---- 4. parents -------------------------------------------------------
    nh = nu_hat(pop, policy.eta_ema)
    if use_mean_utility:
        parents = mean_utility_parents(eligible, nh, policy.n_par)
        lex = LexicaseResult(parents, [len(eligible)], [0], float((eps == 0).float().mean()))
    else:
        lex = epsilon_lexicase(eligible, nu, eps, w_t, policy.n_par, rng_lexicase)
        parents = lex.parents

    # ---- 5. deaths --------------------------------------------------------
    tau = calibrate_tau_death(pop, policy.eta_ema)
    raw = nu_raw(pop, policy.eta_ema)
    rho_n = policy.rho_0 * math.exp(-n / cfg.n_c) * rho_scale
    cap = int(rho_n * n_live_before)

    parent_set = set(parents)
    death_mask = pop.U_death.clone()
    for p in parent_set:
        death_mask[p] = False
    death_mask &= raw < tau
    candidates = torch.nonzero(death_mask, as_tuple=False).squeeze(-1)
    n_death = min(cap, int(candidates.numel()))
    if n_death > 0:
        order = torch.argsort(raw[candidates])
        deaths = candidates[order[:n_death]].tolist()
    else:
        deaths = []

    # ---- 6. births --------------------------------------------------------
    headroom = budget_headroom(
        policy.U_cap, n_live_before, len(deaths),
        n_dying=pop.n_dying, U_max=cfg.U_max,
    )
    free = int((~pop.live).sum())
    n_birth = min(policy.n_par, cap, headroom, free)

    births: List[int] = []
    cap_bound = False
    if n_birth > 0 and parents:
        pop.offspring_this_event.zero_()
        pool = list(parents)
        rng_lexicase.shuffle(pool)
        cursor = 0
        attempts = 0
        max_attempts = 8 * max(1, n_birth)
        while len(births) < n_birth and attempts < max_attempts:
            attempts += 1
            if not pool:
                break
            parent = pool[cursor % len(pool)]
            cursor += 1
            if int(pop.offspring_this_event[parent]) >= cfg.cap_off:
                cap_bound = True
                if all(
                    int(pop.offspring_this_event[p]) >= cfg.cap_off for p in set(pool)
                ):
                    break
                continue
            if not bool(pop.live[parent]) or bool(pop.dying[parent]):
                continue
            # Recombination with probability p_rec, using a second distinct
            # parent from the same multiset (§03.2, ablation A11 pins p_rec = 0).
            partner = None
            if rng_mutation.random() < policy.p_rec:
                others = [p for p in set(pool) if p != parent and bool(pop.live[p])]
                if others:
                    partner = int(others[int(rng_mutation.integers(0, len(others)))])
            births.append(
                birth(
                    pop, int(parent), partner, n,
                    cfg=cfg, field=field_, neighbour_table=neighbour_table,
                    rng=rng_mutation, generator=generator, run=run, G_gen=G_gen,
                    parent_nu_hat=float(nh[parent]),
                )
            )

    # ---- 7. apply deaths (births already applied above) -------------------
    # §04.2 orders deaths before births; the headroom above is already net of
    # them, and `death` only sets the dying flag, so the order of the two calls
    # does not change the outcome. Deaths are applied here so that a slot freed
    # by this event's deaths is not reused by this event's births — a unit must
    # complete its anneal before its slot returns to the pool.
    for d in deaths:
        death(pop, int(d), n)

    n_live_after = pop.n_live

    # ---- postconditions ---------------------------------------------------
    assert abs(n_live_after - n_live_before) <= max(cap, len(births)), (
        f"postcondition violated: |U_live| moved {n_live_before} -> {n_live_after}, "
        f"bound floor(rho*|U|) = {cap}"
    )
    # (1.14)'s two bounds, asserted rather than assumed (audit A-62).
    n_nondying = int((pop.live & ~pop.dying).sum())
    assert n_nondying <= policy.U_cap, (
        f"effective population {n_nondying} exceeds U_cap {policy.U_cap}"
    )
    assert pop.n_live <= cfg.U_max, (
        f"live units {pop.n_live} exceed the slot grid U_max {cfg.U_max}"
    )
    for d in deaths:
        assert pop.age_u[d] > policy.M_mat, "killed a unit with age <= M_mat"
        assert d not in parent_set, "killed a selected parent"
    assert not any(bool(pop.dying[p]) and p not in deaths for p in parents), (
        "a dying unit was selected as a parent"
    )

    reservoir.reset()

    return SelectionEvent(
        step=n,
        site=site,
        n_cases=q,
        parents=list(parents),
        deaths=deaths,
        births=births,
        eps_zero_fraction=lex.eps_zero_fraction,
        mean_pool_after_first_case=float(np.mean(lex.pool_trace)) if lex.pool_trace else 0.0,
        mean_cases_used=float(np.mean(lex.cases_used)) if lex.cases_used else 0.0,
        cap_bound=cap_bound,
        n_live_before=n_live_before,
        n_live_after=n_live_after,
        tau_death=tau,
        w_sum=float(w_t.sum()),
        case_wins=wins,
    )
