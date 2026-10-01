"""Variation operators: `MUTATE` and `CROSSOVER`.

SPEC: 03 §3.2.1

```
Algorithm MUTATE(g, nu_hat) -> g'
  sigma_mut <- sigma_mut_0 / (1 + nu_hat)                                (3.1)
  kind      <- ONE categorical draw from (p_point, p_dup, p_del)
  g'        <- copy(g)
  with probability min(1, sigma_mut / sigma_mut_0):
      if kind == point:  replace a uniformly chosen i in g' by neighbour(i)
      if kind == dup:    replace a uniformly chosen i in g' by a gene expressed
                         by the live unit nearest to p_child; if none, neighbour(i)
      if kind == del:    replace a uniformly chosen i in g' by a uniform draw
                         from {0..N_G-1} \\ g'
  postcondition: |g'| = k_g exactly ; g' subset of {0..N_G-1} ; no duplicates
```

**[NORMATIVE — somatic hypermutation]** (3.1) makes the mutation *probability*
inversely proportional to parent fitness: high-fitness lineages are explored
finely, low-fitness lineages coarsely. This is the affinity-maturation rule from
adaptive immunity and has no analogue in gradient training, which has a step
size but no notion of searching harder around bad solutions.

Spec v1's version was wrong in four ways, all fixed here: it drew three
independent Bernoullis while calling the probabilities a simplex (so zero to
three mutations could occur); its postcondition `1 <= |g'| <= k_g` contradicted
its own "truncated or padded to `|g'| = k_g`" with no padding source; it left
`neighbour` undefined; and it gave the "nearest unit" rule no fallback.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["mutate", "crossover", "MutationOutcome"]


class MutationOutcome:
    """What a call to `mutate` actually did, for the mutation log."""

    __slots__ = ("kind", "applied", "replaced", "inserted", "sigma_ratio")

    def __init__(
        self,
        kind: str,
        applied: bool,
        replaced: Optional[int],
        inserted: Optional[int],
        sigma_ratio: float,
    ):
        self.kind = kind
        self.applied = applied
        self.replaced = replaced
        self.inserted = inserted
        self.sigma_ratio = sigma_ratio

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"MutationOutcome(kind={self.kind!r}, applied={self.applied}, "
            f"{self.replaced}->{self.inserted}, p={self.sigma_ratio:.3f})"
        )


def _pick_replacement(
    genes: List[int],
    idx: int,
    candidates: Sequence[int],
    N_G: int,
    rng: np.random.Generator,
) -> int:
    """Choose a replacement for `genes[idx]` that keeps the set duplicate-free.

    Tries the supplied candidates in order, then falls back to a uniform draw
    from the complement. The postcondition `|g'| = k_g` with no duplicates is
    what makes `k_g`-sparsity exact — a duplicated index would make `gate_u`
    double-weight one gene while the accounting still called it `k_g` parts.
    """
    present = set(genes)
    present.discard(genes[idx])
    for c in candidates:
        c = int(c)
        if c not in present:
            return c
    # Uniform from the complement. Rejection sampling is O(1) in expectation
    # because |g'| = k_g = 8 against N_G = 2048.
    while True:
        c = int(rng.integers(0, N_G))
        if c not in present:
            return c


def mutate(
    genes: Sequence[int],
    nu_hat: float,
    *,
    policy,
    N_G: int,
    neighbour_table: torch.Tensor,
    rng: np.random.Generator,
    nearest_unit_genes: Optional[Sequence[int]] = None,
) -> Tuple[List[int], MutationOutcome]:
    """`MUTATE` of (3.1) and §03.2.1.

    Parameters
    ----------
    nu_hat : the **parent's** normalised utility. Higher fitness -> lower
        mutation probability, per (3.1).
    neighbour_table : `(N_G, n_nbr)` from `Field.neighbour_table()`.
    nearest_unit_genes : the gene set of the live unit nearest to `p_child`,
        for the `dup` kind. `None` triggers the explicit `neighbour(i)`
        fallback that spec v1 omitted.
    """
    g = [int(x) for x in genes]
    k_g = len(g)
    if len(set(g)) != k_g:
        raise ValueError(f"parent gene set has duplicates: {g}")

    # (3.1). sigma_mut / sigma_mut_0 = 1 / (1 + nu_hat), so the acceptance
    # probability is min(1, 1/(1+nu_hat)) — which is 1 at nu_hat = 0 (a unit of
    # unknown fitness is explored at the base rate) and falls toward 0 for a
    # unit far above the population median.
    sigma_ratio = 1.0 / (1.0 + max(0.0, float(nu_hat)))
    p_apply = min(1.0, sigma_ratio)

    kinds = ("point", "dup", "del")
    probs = np.array(policy.mutation_simplex, dtype=float)
    probs = probs / probs.sum()
    kind = kinds[int(rng.choice(3, p=probs))]

    if rng.random() >= p_apply:
        return g, MutationOutcome(kind, False, None, None, sigma_ratio)

    idx = int(rng.integers(0, k_g))
    old = g[idx]

    if kind == "point":
        cands = [int(x) for x in neighbour_table[old].tolist()]
        rng.shuffle(cands)
    elif kind == "dup":
        if nearest_unit_genes:
            cands = [int(x) for x in nearest_unit_genes]
            rng.shuffle(cands)
            cands = cands + [int(x) for x in neighbour_table[old].tolist()]
        else:
            cands = [int(x) for x in neighbour_table[old].tolist()]
            rng.shuffle(cands)
    else:  # del
        cands = []

    new = _pick_replacement(g, idx, cands, N_G, rng)
    g[idx] = new

    assert len(g) == k_g, "MUTATE changed the gene-set size"
    assert len(set(g)) == k_g, "MUTATE produced a duplicate gene"
    return g, MutationOutcome(kind, True, old, new, sigma_ratio)


def crossover(
    g1: Sequence[int], g2: Sequence[int], *, rng: np.random.Generator
) -> List[int]:
    """`CROSSOVER` of §03.2.1.

    ```
    pool <- g1 union g2
    g'   <- k_g elements sampled from pool without replacement, each element
            weighted 1.0 if in g1 & g2, else 0.5
    postcondition: |g'| = k_g
    ```

    **[OURS]** Crossover is available to no gradient method. Ablation **A11**
    tests whether it earns its place.

    The weighting is the mechanism: a gene both parents express is twice as
    likely to survive as one only a single parent carries, so the shared core of
    a lineage is conserved while the divergent periphery recombines. Weighted
    sampling *without replacement* is implemented by the exponential-race
    (Gumbel top-k) construction, which is exact for this scheme.
    """
    a = {int(x) for x in g1}
    b = {int(x) for x in g2}
    k_g = len(list(g1))
    pool = sorted(a | b)
    if len(pool) < k_g:
        raise ValueError(
            f"crossover pool has {len(pool)} < k_g = {k_g} genes; the parents "
            f"cannot both have had k_g distinct genes"
        )
    weights = np.array([1.0 if x in a and x in b else 0.5 for x in pool])
    # Efraimidis-Spirakis: keys w_i^(1/U_i); top-k without replacement is exact
    # for probability-proportional-to-weight sampling.
    keys = rng.random(len(pool)) ** (1.0 / weights)
    order = np.argsort(-keys)
    out = sorted(int(pool[i]) for i in order[:k_g])
    assert len(out) == k_g and len(set(out)) == k_g
    return out
