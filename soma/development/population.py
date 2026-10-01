"""The unit population: per-slot state on a fixed grid.

SPEC: 01 §1.1
SPEC: 03 §3.3
SPEC: 05 §5.4

A unit `u` in `U_l` occupies a fixed slot in a preallocated `side x side` grid.
**The grid size never changes**, so no tensor is reshaped by a birth or a death
and no unit is re-addressed by a population change (§05.4). All per-unit tensors
are preallocated at `U_max`; a death zeroes the freed slot's parameter and
optimizer state, and a birth writes the child's position and leaves its
optimizer state at zero (§05.7).

| Field | Shape | Side |
|---|---|---|
| `slot_u` | implicit (the index) | bookkeeping |
| `live`, `dying` | `(U_max,)` bool | bookkeeping |
| `p_u` | `(U_max, m_dim)` | **soma** (an `nn.Parameter`, AdamW group) |
| `g_u` | `(U_max, k_g)` | **germline-addressed** |
| `m_u` | `(U_max,)` in `[0,1]` | bookkeeping — the MSG mask |
| `age_u` | `(U_max,)` | bookkeeping |
| `nu_u`, `h_bar_u` | `(U_max,)` | bookkeeping — selection state |
| `L_u` | list | bookkeeping — lineage records |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from soma.config import SomaConfig
from soma.utils.shapes import check_shape

__all__ = ["LineageRecord", "UnitPopulation"]


@dataclass
class LineageRecord:
    """`L_u = (run, G_gen, id(u_p1), id(u_p2) | None, n)` (§03.2).

    Spec v1's `BIRTH` took one parent while recombination needs two, and `L_u`
    could record only one; both are extended (§03.2 correction (b)).
    """

    run: str
    G_gen: int
    parent1: int
    parent2: Optional[int]
    step: int
    slot: int
    depth: int = 0
    """Generations from a seeded founder — the quantity §07.6's clade cut uses."""


class UnitPopulation(nn.Module):
    """One Site's unit population.

    `p_u` is an `nn.Parameter` because §02.4 makes it **soma**: a unit's
    *identity* (which genes it expresses) is inherited and discrete, its
    *placement* is learned and continuous. Because `phi_g` is smooth in `p`
    (§01.2) the gradient exists; because gene membership is a set, no gradient
    is needed for it. That seam is the whole point of the position/gene split.
    """

    def __init__(self, cfg: SomaConfig, site_index: int = 0):
        super().__init__()
        self.cfg = cfg
        self.site_index = site_index
        U, m, k = cfg.U_max, cfg.m_dim, cfg.k_g

        self.p_u = nn.Parameter(torch.zeros(U, m))

        self.register_buffer("g_u", torch.zeros(U, k, dtype=torch.long))
        self.register_buffer("live", torch.zeros(U, dtype=torch.bool))
        self.register_buffer("dying", torch.zeros(U, dtype=torch.bool))
        self.register_buffer("m_u", torch.zeros(U))
        self.register_buffer("age_u", torch.zeros(U, dtype=torch.long))
        self.register_buffer("nu_u", torch.zeros(U))
        self.register_buffer("h_bar_u", torch.zeros(U))
        # Mask anneal bookkeeping: `anneal_dir` is +1 while a newborn's mask
        # rises, -1 while a dying unit's mask falls, 0 when static.
        self.register_buffer("anneal_dir", torch.zeros(U, dtype=torch.int8))
        self.register_buffer("anneal_start", torch.zeros(U, dtype=torch.long))
        # Offspring produced in the current selection event, for `cap_off`.
        self.register_buffer("offspring_this_event", torch.zeros(U, dtype=torch.long))

        self._n_active_cached: Optional[int] = None
        self.lineage: Dict[int, LineageRecord] = {}
        self.tau_death: Optional[float] = None
        """Calibrated once at the first selection event and then fixed (§04.4)."""

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    @property
    def active(self) -> torch.Tensor:
        """§05.4's single predicate: `live and (not dying) and (m_u > 0)`.

        One predicate governs the row/column mask, `cand`, and the softmax
        alike. Spec v2 masked rows on "has a live slot" while excluding newborns
        only from `cand`; a newborn that was the first live slot in its row would
        flip that row from `-inf` to finite, changing `I1` and hence **every
        other unit's** `r_ut`, so Theorem 3.1a did not hold.
        """
        return self.live & (~self.dying) & (self.m_u > 0)

    @property
    def U_parent(self) -> torch.Tensor:
        """`{ u : live_u and not dying_u }` — may reproduce (§04.2).

        A newborn's exclusion is from *routing* until `m_u > 0`, **not** from
        reproduction. Spec v2 passed a single `U_elig` to lexicase and then
        defined it with the maturity condition, while the following sentence
        said parents come from "all live, non-dying units"; the two readings
        disagreed on whether newborns can reproduce.
        """
        return self.live & (~self.dying)

    @property
    def U_death(self) -> torch.Tensor:
        """`{ u : live_u and not dying_u and age_u > M_mat }` — may be culled."""
        return self.U_parent & (self.age_u > self.cfg.policy.M_mat)

    @property
    def n_live(self) -> int:
        return int(self.live.sum())

    @property
    def n_dying(self) -> int:
        return int(self.dying.sum())

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    @property
    def n_active_cached(self) -> int:
        """The active count, maintained on the **host**.

        `Router` needs it to size §05.4's widening, and reading it from the
        device costs a synchronisation per Site per step — 12 stalls per forward
        at M1 — for a number that changes only at a selection event, a growth
        event, or a mask-anneal retirement. Every mutation invalidates it and
        the next read recomputes it once.
        """
        if self._n_active_cached is None:
            self._n_active_cached = int(self.active.sum())
        return self._n_active_cached

    def invalidate_active_count(self) -> None:
        self._n_active_cached = None

    def free_slots(self) -> torch.Tensor:
        return torch.nonzero(~self.live, as_tuple=False).squeeze(-1)

    # ------------------------------------------------------------------
    # Grid geometry
    # ------------------------------------------------------------------

    def slot_coords(self, slot: int) -> Tuple[int, int]:
        return divmod(int(slot), self.cfg.side)

    def free_slot_nearest(self, slot: int) -> int:
        """`FREE_SLOT_NEAREST(slot)` — grid-L1 nearest free slot (§03.2).

        Spec v1's `BIRTH` never assigned a routing address, so a child could
        land anywhere in the grid and receive a completely different token
        distribution from its parent — destroying both the differentiated-
        siblings argument (§02.4) and the clade prediction of §03.2.

        Ties are broken by the lower flat index, deterministically, so the same
        seed produces the same grid layout.
        """
        free = self.free_slots()
        if free.numel() == 0:
            raise RuntimeError(
                "no free slot in the grid; BUDGET_HEADROOM should have "
                "prevented this birth (§01.6.1)"
            )
        side = self.cfg.side
        i1, i2 = self.slot_coords(slot)
        f1 = free // side
        f2 = free % side
        dist = (f1 - i1).abs() + (f2 - i2).abs()
        # `torch.argmin` returns the first minimum in flat order, and `free` is
        # ascending, so the tie-break is the lower flat index.
        return int(free[int(torch.argmin(dist))])

    # ------------------------------------------------------------------
    # Mask annealing (§03.3)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step_masks(self, n: int) -> torch.Tensor:
        """Advance `m_u` one step; return the slots that finished dying.

        `m_u` rises linearly `0 -> 1` over `phi_ann` steps for a newborn and
        falls linearly `1 -> 0` for a dying unit. A unit whose mask reaches 0 is
        retired: `live <- False`, and its position and optimizer state are zeroed
        by the caller (§05.7).
        """
        phi = self.cfg.phi_ann
        growing = self.anneal_dir == 1
        shrinking = self.anneal_dir == -1

        if bool(growing.any()):
            frac = ((n - self.anneal_start[growing]).float() / phi).clamp(0.0, 1.0)
            self.m_u[growing] = frac
            done = growing.clone()
            done[growing] = frac >= 1.0
            self.anneal_dir[done] = 0
            self.m_u[done] = 1.0

        retired = torch.zeros_like(self.live)
        if bool(shrinking.any()):
            frac = ((n - self.anneal_start[shrinking]).float() / phi).clamp(0.0, 1.0)
            self.m_u[shrinking] = 1.0 - frac
            done = shrinking.clone()
            done[shrinking] = frac >= 1.0
            retired = done
            self.m_u[done] = 0.0

        self.age_u[self.live] += 1
        self._n_active_cached = None
        return retired

    @torch.no_grad()
    def retire(self, slots: torch.Tensor) -> None:
        """Free a slot completely (§03.4 step 3, §05.7)."""
        if slots.numel() == 0:
            return
        self.live[slots] = False
        self.dying[slots] = False
        self.m_u[slots] = 0.0
        self.age_u[slots] = 0
        self.nu_u[slots] = 0.0
        self.h_bar_u[slots] = 0.0
        self.anneal_dir[slots] = 0
        self.p_u.data[slots] = 0.0
        self.g_u[slots] = 0
        self._n_active_cached = None

    # ------------------------------------------------------------------
    # Seeding (§02.2.4)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def seed(
        self,
        inventory,
        rng: np.random.Generator,
        generator: torch.Generator,
        *,
        run: str = "run",
        G_gen: int = 0,
        mutate_fn=None,
    ) -> None:
        """`SEED_POPULATION` of §02.2.4.

        ```
        1. slots <- |U|_0 slots chosen by a low-discrepancy sweep of the grid
        2. for each slot:
             if Inventory non-empty: g_u <- archetype sampled prop. to
                                            generations-survived, then MUTATE'd once
             else:                   g_u <- k_g indices uniform w/o replacement
             p_u   <- uniform on S^{m_dim-1}
             m_u   <- 1        # seeded units are mature
             age_u <- M_mat    # immediately eligible, having no parent to protect
             nu_u  <- 0
        ```

        Seeded units start **mature** (`m_u = 1`): were they to start at
        `m_u = 0`, the model would compute nothing at step 0 and "birth
        performance" would be undefined rather than merely low (§02.2.4).
        """
        from soma.germline.field import sample_sphere

        cfg = self.cfg
        n0 = cfg.U_0
        slots = _low_discrepancy_slots(cfg.side, n0)

        self.live.zero_()
        self.dying.zero_()
        self.m_u.zero_()
        self.age_u.zero_()
        self.nu_u.zero_()
        self.h_bar_u.zero_()
        self.anneal_dir.zero_()
        self.lineage.clear()

        pos = sample_sphere(
            n0, cfg.m_dim, generator, device=self.p_u.device, dtype=self.p_u.dtype
        )
        for j, slot in enumerate(slots):
            if inventory is not None and not inventory.is_empty():
                genes = list(inventory.sample_gene_set(rng))
                if mutate_fn is not None:
                    genes = list(mutate_fn(genes, 0.0))
            else:
                genes = list(rng.choice(cfg.N_G, size=cfg.k_g, replace=False))
            self.g_u[slot] = torch.tensor(
                sorted(int(g) for g in genes), dtype=torch.long, device=self.g_u.device
            )
            self.p_u.data[slot] = pos[j]
            self.live[slot] = True
            self.m_u[slot] = 1.0
            # §04.2 defines U_death with `age_u > M_mat`, so seeding at exactly
            # M_mat would leave a seeded unit ineligible for one step, making
            # §02.2.4's "immediately eligible" false. M_mat + 1 makes it true.
            self.age_u[slot] = cfg.policy.M_mat + 1
            self.lineage[int(slot)] = LineageRecord(
                run, G_gen, -1, None, 0, int(slot), 0
            )

        self._n_active_cached = None
        assert self.n_live == n0, f"seeded {self.n_live} units, expected {n0}"
        assert bool(
            (self.g_u[self.live].sort(dim=-1).values.diff(dim=-1) > 0).all()
        ), "a seeded unit has duplicate genes"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def clade_of(self, slot: int, max_depth: int = 3) -> int:
        """The MRCA-at-depth-`max_depth` ancestor id, for §07.6's clade cut.

        §07.6: "clade" is defined by cutting the lineage forest at
        `MRCA depth <= 3` generations; the cut rule is fixed and logged. Spec v1
        required "number of distinct clades" with no cut rule at all.
        """
        cur = int(slot)
        rec = self.lineage.get(cur)
        if rec is None:
            return cur
        hops = 0
        while rec is not None and rec.parent1 >= 0 and hops < max_depth:
            cur = rec.parent1
            rec = self.lineage.get(cur)
            hops += 1
        return cur

    def clade_counts(self, max_depth: int = 3) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for slot in torch.nonzero(self.live, as_tuple=False).squeeze(-1).tolist():
            root = self.clade_of(slot, max_depth)
            out[root] = out.get(root, 0) + 1
        return out

    def clade_distances(self, max_depth: int = 3) -> Tuple[float, float]:
        """Mean within-clade and between-clade position distance (§07.6).

        §03.2's clade prediction is that `p_child = normalize(p_parent + xi)`
        places offspring near their parent, so under `L_wire` a lineage occupies
        a compact region and modules arise as **clades** rather than as a
        designed partitioning. §03.7 makes failure of this prediction a stated
        falsifier, so it must be measured, not assumed.
        """
        slots = torch.nonzero(self.live, as_tuple=False).squeeze(-1)
        if slots.numel() < 2:
            return 0.0, 0.0
        roots = torch.tensor(
            [self.clade_of(int(s), max_depth) for s in slots.tolist()],
            device=slots.device,
        )
        p = self.p_u.data[slots]
        d = torch.cdist(p, p)
        same = roots.unsqueeze(0) == roots.unsqueeze(1)
        n = slots.numel()
        iu = torch.triu_indices(n, n, offset=1, device=slots.device)
        dv = d[iu[0], iu[1]]
        sv = same[iu[0], iu[1]]
        within = float(dv[sv].mean()) if bool(sv.any()) else 0.0
        between = float(dv[~sv].mean()) if bool((~sv).any()) else 0.0
        return within, between

    def state_summary(self) -> Dict[str, float]:
        live = self.live
        ages = self.age_u[live].float()
        return {
            "n_live": self.n_live,
            "n_dying": self.n_dying,
            "n_active": self.n_active,
            "age_mean": float(ages.mean()) if live.any() else 0.0,
            "age_median": float(ages.median()) if live.any() else 0.0,
            "clade_count": len(self.clade_counts()),
            "lineage_depth_max": max(
                (r.depth for r in self.lineage.values()), default=0
            ),
        }


def _low_discrepancy_slots(side: int, n: int) -> List[int]:
    """`n` slots spread over a `side x side` grid (§02.2.4 step 1).

    A stride sweep by the golden ratio in flat index space: successive slots are
    far apart in *both* grid coordinates, which matters because `L_wire` and the
    routing keys are both spatial — seeding a contiguous block would give the
    initial population a systematic addressing bias that the first selection
    event would then be measuring.
    """
    total = side * side
    if n > total:
        raise ValueError(f"cannot seed {n} units into {total} slots")
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    stride = max(1, int(round(total / phi)))
    while _gcd(stride, total) != 1:
        stride += 1
    out: List[int] = []
    cur = 0
    seen = set()
    while len(out) < n:
        if cur not in seen:
            out.append(cur)
            seen.add(cur)
        cur = (cur + stride) % total
    return sorted(out)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a
