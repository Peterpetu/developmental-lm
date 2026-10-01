"""The MAP-Elites loop, independent of GNOMON.

SPEC: 08 §8.2
SPEC: 08 §8.6

Separated from `evolve.py` so that the algorithm can be driven by a synthetic
fitness function with **no GPU and no model at all**. That is what makes §08.6's
E5 and E6 possible: a control that has never been shown to fire in both
directions -- indistinguishable from random search on a structureless landscape,
and better than it on a landscape with stepping stones -- is not a control.

It is also what makes `A-RAND` an exactly matched ablation: random search is
this same loop with `p_random = 1`, so it shares the archive, the insertion
rule, the budget and the accounting, and differs only in where genomes come
from.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, TypeVar

import torch

from .archive import Archive, Member

__all__ = ["map_elites"]

G = TypeVar("G")


def map_elites(
    archive: Archive,
    *,
    n_init: int,
    n_pop: int,
    G_gen: int,
    p_random: float,
    random_genome: Callable[[int], G],
    vary: Callable[[Member, Member, torch.Generator], G],
    evaluate: Callable[[G, int], Member],
    rng: torch.Generator,
    on_generation: Optional[Callable[[int, dict], None]] = None,
) -> Archive:
    """SPEC: 08 §8.2. Returns the archive it was given, filled.

    The budget is exact: `n_init + G_gen * n_pop` evaluations, asserted at the
    end (§08.6 E9), because an off-by-one in the loop would make every
    comparison against a matched control invalid by the amount of the error.
    """
    n_eval = 0
    for i in range(n_init):
        archive.insert(evaluate(random_genome(i), 0))
        n_eval += 1
    row = archive.record(0)
    if on_generation:
        on_generation(0, row)

    for g in range(1, G_gen + 1):
        for i in range(n_pop):
            fresh = (len(archive.cells) < 2
                     or float(torch.rand((), generator=rng)) < p_random)
            if fresh:
                genome = random_genome(n_init + (g - 1) * n_pop + i)
            else:
                p1, p2 = archive.sample_parents(2, rng)
                genome = vary(p1, p2, rng)
            archive.insert(evaluate(genome, g))
            n_eval += 1
        row = archive.record(g)
        if on_generation:
            on_generation(g, row)

    assert n_eval == n_init + G_gen * n_pop, (n_eval, n_init, G_gen, n_pop)
    return archive
