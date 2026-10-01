"""`Q_route` — bipartite modularity of the token-to-unit routing graph.

SPEC: 08 §8.1.1

**[NORMATIVE — v1 correction]** Spec v1 named `Q_route` in a metric table and
defined it nowhere, while making it the **sole** quantity in K4 and one of three
in K3.

* **Graph.** Bipartite, per Site `l`: left nodes are the `2048` tokens of the
  fixed probe batch, right nodes are the live units of `U_l`.
* **Edges.** Weight `w(t,u) = r_ut` for `u in R_t`; edges with `r_ut < 1e-3` are
  dropped.
* **Modularity.** **Barber bipartite modularity** `Q_B` (the bipartite null
  model; Newman's `Q` is inappropriate for a bipartite graph and its range is
  not `[0,1]`).
* **Communities.** 10 restarts at a fixed seed, best `Q_B` retained.
* **Aggregation.** Computed per Site and **averaged over the `L` Sites**.
* **Range.** `Q_B` in `[-1, 1]`.

**[NORMATIVE — v3.1 correction, the community algorithm]** §08.1.1 v3.0
specified **Leiden** at resolution 1.0. Leiden optimises Newman modularity (or
CPM), **not** Barber's bipartite null model, so using it to find communities and
then *scoring* them with `Q_B` systematically under-reports the very quantity
K4 is adjudicated on at a margin of `0.02`. The algorithm Barber introduced for
`Q_B` is **BRIM** (bipartite, recursively induced modules), which maximises
`Q_B` directly by alternating optimal assignment of each side given the other.
BRIM is used here, with the restart discipline unchanged. It also removes a
dependency (`igraph`/`leidenalg`) that does not build cleanly on this platform,
which R8 identifies as a standing risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["barber_modularity", "brim", "q_route", "RoutingGraph", "EDGE_FLOOR"]

EDGE_FLOOR = 1e-3
"""§08.1.1: edges with `r_ut < 1e-3` are dropped."""
Q_SEED = 20260828
"""§08.1.1's fixed seed."""
N_RESTARTS = 10


@dataclass
class RoutingGraph:
    """The bipartite adjacency of one Site's probe batch."""

    A: np.ndarray
    """`(n_tokens, n_units)` of edge weights."""
    unit_ids: np.ndarray
    """Slot index of each right node."""

    @property
    def m(self) -> float:
        return float(self.A.sum())


def build_graph(
    slots: torch.Tensor, r_ut: torch.Tensor, live: torch.Tensor
) -> RoutingGraph:
    """Assemble the bipartite graph from one probe batch's routing.

    `slots` and `r_ut` are `(n_tokens, k_route)`; `live` is `(U_max,)`.
    """
    unit_ids = torch.nonzero(live, as_tuple=False).squeeze(-1).cpu().numpy()
    if unit_ids.size == 0:
        return RoutingGraph(np.zeros((0, 0)), unit_ids)
    remap = -np.ones(int(live.numel()), dtype=np.int64)
    remap[unit_ids] = np.arange(unit_ids.size)

    s = slots.reshape(-1).cpu().numpy()
    w = r_ut.reshape(-1).float().cpu().numpy()
    rows = np.repeat(np.arange(slots.shape[0]), slots.shape[1])

    keep = (s >= 0) & (w >= EDGE_FLOOR)
    keep &= np.where(s >= 0, remap[np.clip(s, 0, None)] >= 0, False)
    if not keep.any():
        return RoutingGraph(np.zeros((slots.shape[0], unit_ids.size)), unit_ids)

    A = np.zeros((slots.shape[0], unit_ids.size), dtype=np.float64)
    np.add.at(A, (rows[keep], remap[s[keep]]), w[keep])
    return RoutingGraph(A, unit_ids)


def barber_modularity(
    A: np.ndarray, red: np.ndarray, blue: np.ndarray
) -> float:
    """Barber's bipartite modularity `Q_B`.

    SPEC: 08 §8.1.1

    ```
    Q_B = (1/m) * sum_{i in R, j in B} ( A_ij - k_i d_j / m ) * delta(g_i, g_j)
    ```

    with `m` the total edge weight, `k_i` the degree of red node `i` and `d_j`
    the degree of blue node `j`. The null model preserves the bipartite
    structure, which Newman's `Q` does not: Newman's null admits red-red and
    blue-blue edges that cannot exist, so it under-counts the expected
    within-community weight and inflates `Q`.
    """
    m = float(A.sum())
    if m <= 0 or A.size == 0:
        return 0.0
    k = A.sum(axis=1)
    d = A.sum(axis=0)
    B = A - np.outer(k, d) / m
    same = red[:, None] == blue[None, :]
    return float((B * same).sum() / m)


def brim(
    A: np.ndarray,
    n_communities: int,
    rng: np.random.Generator,
    *,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """BRIM: maximise `Q_B` by alternating optimal assignment of each side.

    SPEC: 08 §8.1.1

    With `B~ = A - k d^T / m`, `Q_B = (1/m) trace(R^T B~ T)` for indicator
    matrices `R` (red x c) and `T` (blue x c). Holding `T` fixed, each red node's
    optimal community is `argmax_c (B~ T)[i, c]`; holding `R` fixed, each blue
    node's is `argmax_c (B~^T R)[j, c]`. Both steps are exact and non-decreasing
    in `Q_B`, so the alternation converges monotonically to a local optimum —
    which is why the restart discipline matters and is retained.
    """
    n_r, n_b = A.shape
    m = float(A.sum())
    if m <= 0 or n_r == 0 or n_b == 0:
        return np.zeros(n_r, np.int64), np.zeros(n_b, np.int64), 0.0
    c = max(1, min(n_communities, n_r, n_b))
    k = A.sum(axis=1)
    d = A.sum(axis=0)
    Bt = A - np.outer(k, d) / m

    blue = rng.integers(0, c, size=n_b)
    prev_q = -np.inf
    red = np.zeros(n_r, dtype=np.int64)
    for _ in range(max_iter):
        T = np.zeros((n_b, c))
        T[np.arange(n_b), blue] = 1.0
        red = np.asarray((Bt @ T).argmax(axis=1), dtype=np.int64)

        R = np.zeros((n_r, c))
        R[np.arange(n_r), red] = 1.0
        blue = np.asarray((Bt.T @ R).argmax(axis=1), dtype=np.int64)

        q = barber_modularity(A, red, blue)
        if q <= prev_q + 1e-12:
            break
        prev_q = q
    return red, blue, barber_modularity(A, red, blue)


def q_route_site(
    graph: RoutingGraph,
    *,
    seed: int = Q_SEED,
    restarts: int = N_RESTARTS,
    community_grid: Sequence[int] = (2, 4, 8, 16, 32),
) -> Tuple[float, int]:
    """`Q_B` for one Site: best over restarts and community counts.

    §08.1.1's "10 restarts, best `Q_B` retained", extended over the community
    count because BRIM — unlike Leiden — takes `c` as input. The grid is fixed
    and part of the metric's definition, so two runs are comparable.
    """
    if graph.A.size == 0 or graph.m <= 0:
        return 0.0, 0
    best_q, best_c = -1.0, 0
    for c in community_grid:
        for r in range(restarts):
            rng = np.random.default_rng(seed + 1000 * c + r)
            _, _, q = brim(graph.A, c, rng)
            if q > best_q:
                best_q, best_c = q, c
    return best_q, best_c


def q_route(
    site_slots: Sequence[torch.Tensor],
    site_r: Sequence[torch.Tensor],
    site_live: Sequence[torch.Tensor],
    *,
    seed: int = Q_SEED,
) -> Dict[str, object]:
    """`Q_route`: per-Site `Q_B`, averaged over the `L` Sites (§08.1.1)."""
    per_site: List[float] = []
    per_site_c: List[int] = []
    for slots, r, live in zip(site_slots, site_r, site_live):
        g = build_graph(slots, r, live)
        q, c = q_route_site(g, seed=seed)
        per_site.append(q)
        per_site_c.append(c)
    mean = float(np.mean(per_site)) if per_site else 0.0
    if not -1.0 <= mean <= 1.0:
        raise AssertionError(
            f"Q_B = {mean} outside its declared range [-1, 1]; §01.9's earlier "
            f"[0,1] was wrong and this assertion is what keeps the corrected "
            f"range honest"
        )
    return {
        "Q_route": mean,
        "per_site": per_site,
        "communities": per_site_c,
        "n_sites": len(per_site),
    }
