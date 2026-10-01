"""`Inventory` — discovered archetypes, and `SEED_POPULATION`.

SPEC: 02 §2.2.4

`{ (g^(j), stats_j) }` with `|Inventory| <= J_max = 256`. `stats` records
generations survived, mean utility and mean clade size. Eviction is by lowest
generations-survived, ties broken by lowest mean utility.

The Inventory is the *only* object besides the field and the policy that crosses
a lifetime boundary, and it carries no weights — only gene index sets. That is
what makes §07.2 step 7 the Weismann barrier in operational form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from soma.config import GERMLINE_SCALE, GermlineScale
from soma.germline.registry import GermlineRegistry
from soma.utils.shapes import check_shape

__all__ = ["Archetype", "Inventory"]


@dataclass
class Archetype:
    """One heritable gene set with its selection statistics."""

    genes: Tuple[int, ...]
    generations_survived: int = 0
    """Distinct **generations** in which this gene set appeared.

    Not appearances: `insert` is called once per surviving unit and `_harvest`
    emits one archetype per unit, so counting appearances made one generation
    contribute ~240 increments at M-nano (audit A-86). §02.2.4 weights
    `SEED_POPULATION`'s sampling by this field, and `_evict` orders by it."""
    mean_utility: float = 0.0
    mean_clade_size: float = 0.0
    last_generation: Optional[int] = None
    """The generation of the most recent insert, so the counter is idempotent
    within a generation. `None` means the caller supplied no generation, in
    which case every insert counts — the legacy behaviour, kept for the unit
    tests that construct inventories directly."""

    def key(self) -> Tuple[int, ...]:
        return tuple(sorted(self.genes))


@dataclass
class Inventory:
    """The germline archetype store.

    SPEC: 02 §2.2.4

    Held as a Python structure rather than a tensor because its operations are
    set membership, insertion and eviction — none of which vectorise — and
    because `J_max * k_g = 2048` indices is small. It is exposed to the registry
    as a single packed tensor so that `|Gamma|` is measurable in one place
    (§09.3's `O(J_max k_g)` term).
    """

    scale: GermlineScale = GERMLINE_SCALE
    archetypes: List[Archetype] = field(default_factory=list)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.archetypes)

    def is_empty(self) -> bool:
        return not self.archetypes

    def packed(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """`(J_max, k_g)` int32 view, `-1`-padded, for size accounting.

        Registered with the `GermlineRegistry` so `numel()` includes the
        Inventory term of (9.1). The padding to `J_max` is deliberate: the
        germline's *capacity* is what is independent of `|U|`, `d_model` and
        `L`, and a ragged store would make `|Gamma|` depend on how many
        archetypes happened to be discovered.
        """
        out = torch.full(
            (self.scale.J_max, self.scale.k_g), -1, dtype=torch.int32,
            device=torch.device(device),
        )
        for j, a in enumerate(self.archetypes[: self.scale.J_max]):
            out[j, : len(a.genes)] = torch.tensor(
                a.genes, dtype=torch.int32, device=out.device
            )
        return out

    def register(self, registry: GermlineRegistry, prefix: str = "") -> torch.Tensor:
        t = self.packed()
        registry.register(f"{prefix}inventory", t)
        return t

    # ------------------------------------------------------------------

    def insert(
        self,
        genes: Sequence[int],
        *,
        mean_utility: float = 0.0,
        mean_clade_size: float = 0.0,
        generation: Optional[int] = None,
    ) -> None:
        """Insert or refresh an archetype, evicting if over capacity."""
        genes_t = tuple(sorted(int(g) for g in genes))
        if len(set(genes_t)) != len(genes_t):
            raise ValueError(f"archetype gene set has duplicates: {genes_t}")
        if len(genes_t) != self.scale.k_g:
            raise ValueError(
                f"archetype must have exactly k_g = {self.scale.k_g} genes; "
                f"got {len(genes_t)}"
            )
        for a in self.archetypes:
            if a.key() == genes_t:
                # **[NORMATIVE — the counter counts *generations*; v3.1, audit
                # A-86]** `insert` is called once per surviving **unit**, and
                # `_harvest` emits one archetype per unit — about 120 at
                # M-nano — so v3.0's unconditional `+= 1` made a single
                # generation contribute ~240 increments. The field's name is its
                # contract, and three consumers read it as one: `_evict`'s
                # ordering, §02.2.4's `SEED_POPULATION` sampling weight
                # ("proportional to its generations-survived"), and the running
                # mean below. Measured: 1 unit over 40 generations and 40 units
                # in 1 generation both reported `generations_survived = 40`.
                #
                # `last_generation` makes the increment idempotent within a
                # generation, so the counter means what it says.
                if generation is None or a.last_generation != generation:
                    a.generations_survived += 1
                    a.last_generation = generation
                n = a.generations_survived
                a.mean_utility += (mean_utility - a.mean_utility) / n
                a.mean_clade_size += (mean_clade_size - a.mean_clade_size) / n
                return
        self.archetypes.append(
            Archetype(genes_t, 1, mean_utility, mean_clade_size,
                      last_generation=generation)
        )
        self._evict()

    def _evict(self) -> None:
        """§02.2.4: lowest generations-survived, ties by lowest mean utility."""
        while len(self.archetypes) > self.scale.J_max:
            worst = min(
                range(len(self.archetypes)),
                key=lambda i: (
                    self.archetypes[i].generations_survived,
                    self.archetypes[i].mean_utility,
                ),
            )
            self.archetypes.pop(worst)

    def sample_gene_set(self, rng: np.random.Generator) -> Tuple[int, ...]:
        """Sample an archetype with probability proportional to survival.

        SPEC: 02 §2.2.4

        Weighting by generations-survived is the heritable signal: an archetype
        that has persisted across germline generations is one whose lineages
        kept being selected, which is the only cross-lifetime information the
        barrier permits.
        """
        if not self.archetypes:
            raise ValueError("cannot sample from an empty Inventory")
        w = np.array(
            [max(1, a.generations_survived) for a in self.archetypes], dtype=float
        )
        w = w / w.sum()
        j = int(rng.choice(len(self.archetypes), p=w))
        return self.archetypes[j].genes

    def to_json(self) -> List[Dict[str, object]]:
        return [
            {
                "genes": list(a.genes),
                "generations_survived": a.generations_survived,
                "mean_utility": a.mean_utility,
                "mean_clade_size": a.mean_clade_size,
            }
            for a in self.archetypes
        ]

    @classmethod
    def from_json(
        cls, rows: Sequence[Dict[str, object]], scale: GermlineScale = GERMLINE_SCALE
    ) -> "Inventory":
        inv = cls(scale)
        for r in rows:
            inv.archetypes.append(
                Archetype(
                    tuple(int(g) for g in r["genes"]),  # type: ignore[arg-type]
                    int(r["generations_survived"]),      # type: ignore[arg-type]
                    float(r["mean_utility"]),            # type: ignore[arg-type]
                    float(r["mean_clade_size"]),         # type: ignore[arg-type]
                )
            )
        inv._evict()
        return inv

    def clone(self) -> "Inventory":
        return Inventory(
            self.scale, [Archetype(a.genes, a.generations_survived,
                                   a.mean_utility, a.mean_clade_size)
                         for a in self.archetypes],
        )
