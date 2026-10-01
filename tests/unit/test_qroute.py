"""`Q_route`: Barber bipartite modularity, maximised by BRIM.

SPEC: 08 §8.1.1

Gate `G-QROUTE`. `Q_route` is the metric that decides whether routing has
organised into modules, and audit A-52 found the v2 pipeline reporting Barber's
`Q_B` while *optimising* Newman's `Q` through Leiden — a community structure
chosen to maximise one objective, scored on a different one. BRIM removes the
mismatch by maximising `Q_B` directly.

Every check here is against a hand-constructible ground truth: a graph whose
modularity can be computed by hand, a graph with no structure at all, and a
planted block structure BRIM must recover. Testing a community-detection
algorithm against its own output would be circular.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma.eval.qroute import (
    EDGE_FLOOR,
    barber_modularity,
    brim,
    build_graph,
    q_route,
    q_route_site,
)


def _blocks(n_r: int, n_b: int, k: int, off: float = 0.0) -> np.ndarray:
    """`k` planted blocks: within-block weight 1, between-block weight `off`."""
    A = np.full((n_r, n_b), off, dtype=float)
    for i in range(n_r):
        for j in range(n_b):
            if (i * k) // n_r == (j * k) // n_b:
                A[i, j] = 1.0
    return A


# ---------------------------------------------------------------------------
# Barber's Q_B against hand computation
# ---------------------------------------------------------------------------


def test_barber_modularity_matches_a_hand_computation() -> None:
    """A 2x2 graph small enough to evaluate the definition by hand.

    ```
    A = [[1, 0],      m = 2,  k = (1, 1),  d = (1, 1)
         [0, 1]]      B~_ij = A_ij - k_i d_j / m = A_ij - 0.5
    ```
    With the perfect assignment `red = blue = (0, 1)` the two diagonal terms
    contribute `(1 - 0.5)` each, so `Q_B = (0.5 + 0.5) / 2 = 0.5`.
    """
    A = np.eye(2)
    perfect = np.array([0, 1])
    assert barber_modularity(A, perfect, perfect) == pytest.approx(0.5)
    # The anti-assignment picks out the two off-diagonal terms instead.
    anti = np.array([1, 0])
    assert barber_modularity(A, perfect, anti) == pytest.approx(-0.5)


def test_barber_modularity_is_zero_on_a_complete_graph() -> None:
    """A complete bipartite graph has no community structure: every `B~_ij` is
    exactly zero, so **every** assignment scores zero."""
    A = np.ones((5, 7))
    rng = np.random.default_rng(0)
    for _ in range(5):
        red = rng.integers(0, 3, size=5)
        blue = rng.integers(0, 3, size=7)
        assert barber_modularity(A, red, blue) == pytest.approx(0.0, abs=1e-12)


def test_barber_modularity_handles_the_empty_graph() -> None:
    assert barber_modularity(np.zeros((3, 3)), np.zeros(3, int), np.zeros(3, int)) == 0.0
    assert barber_modularity(np.zeros((0, 0)), np.zeros(0, int), np.zeros(0, int)) == 0.0


# ---------------------------------------------------------------------------
# BRIM
# ---------------------------------------------------------------------------


def test_brim_recovers_a_planted_block_structure_given_its_restarts() -> None:
    """Three planted 4x4 blocks. The optimum is computable by hand:
    `m = 48`, `k_i = d_j = 4`, so every in-block term is `1 - 16/48 = 2/3` and
    `Q_B = 48 * (2/3) / 48 = 2/3`.

    A **single** BRIM start reaches it about 60% of the time and otherwise stops
    at `4/9` — the alternation is monotone in `Q_B` and therefore converges to a
    *local* optimum. That is precisely why §08.1.1 mandates restarts, and it is
    asserted here rather than assumed: the single-start failure is part of the
    metric's specification, not a flaw to hide."""
    A = _blocks(12, 12, 3)
    truth = np.repeat(np.arange(3), 4)
    optimum = barber_modularity(A, truth, truth)
    assert optimum == pytest.approx(2 / 3)

    singles = [brim(A, 3, np.random.default_rng(s))[2] for s in range(10)]
    assert max(singles) == pytest.approx(optimum)
    assert min(singles) < optimum - 0.1, (
        "single starts no longer get stuck, so the restart discipline is "
        "untested by this case"
    )

    from soma.eval.qroute import RoutingGraph
    q, _ = q_route_site(RoutingGraph(A, np.arange(12)))
    assert q == pytest.approx(optimum), "the restart discipline missed the optimum"


def test_brim_never_scores_below_a_random_assignment_of_the_same_graph() -> None:
    """BRIM's two half-steps are each exact argmaxes, so `Q_B` is non-decreasing.
    The claim is checked against random assignments rather than asserted."""
    A = _blocks(12, 12, 3, off=0.15)
    _, _, q = brim(A, 3, np.random.default_rng(2))
    rng = np.random.default_rng(3)
    worst = max(
        barber_modularity(A, rng.integers(0, 3, 12), rng.integers(0, 3, 12))
        for _ in range(50)
    )
    assert q >= worst - 1e-12, (q, worst)


def test_brim_is_near_zero_on_a_structureless_graph() -> None:
    """The control that makes a high `Q_route` mean something: a graph with no
    modules must not be reported as modular."""
    A = np.ones((16, 16))
    _, _, q = brim(A, 4, np.random.default_rng(4))
    assert abs(q) < 1e-9, q


def test_brim_optimises_the_objective_it_reports() -> None:
    """Audit A-52: v2 optimised Newman's `Q` via Leiden and reported Barber's
    `Q_B`. Here the returned score **is** `barber_modularity` of the returned
    partition, which is what makes the two agree by construction."""
    A = _blocks(10, 14, 2, off=0.2)
    red, blue, q = brim(A, 2, np.random.default_rng(5))
    assert q == pytest.approx(barber_modularity(A, red, blue))


def test_brim_degenerates_gracefully() -> None:
    for A in (np.zeros((4, 4)), np.zeros((0, 5)), np.zeros((5, 0))):
        red, blue, q = brim(A, 3, np.random.default_rng(6))
        assert q == 0.0
        assert red.shape[0] == A.shape[0] and blue.shape[0] == A.shape[1]


def test_q_route_site_is_deterministic_for_a_fixed_seed() -> None:
    """§08.1.1 fixes the seed and the restart count, so two runs of the metric on
    the same graph are comparable numbers rather than two samples."""
    from soma.eval.qroute import RoutingGraph

    g = RoutingGraph(_blocks(12, 12, 3), np.arange(12))
    a = q_route_site(g)
    b = q_route_site(g)
    assert a == b


def test_q_route_site_beats_a_single_community_on_structured_input() -> None:
    from soma.eval.qroute import RoutingGraph

    structured = RoutingGraph(_blocks(16, 16, 4), np.arange(16))
    flat = RoutingGraph(np.ones((16, 16)), np.arange(16))
    q_s, c_s = q_route_site(structured)
    q_f, _ = q_route_site(flat)
    assert q_s > 0.5 and c_s >= 2, (q_s, c_s)
    assert abs(q_f) < 1e-9, q_f


# ---------------------------------------------------------------------------
# The graph the metric is actually computed on
# ---------------------------------------------------------------------------


def test_build_graph_drops_edges_below_the_floor_and_dead_units() -> None:
    """§08.1.1: `r_ut < 1e-3` is dropped, and a dead slot is not a node."""
    slots = torch.tensor([[[0, 1, 2]], [[0, 1, 3]]])          # (B=2, T=1, k=3)
    r_ut = torch.tensor([[[0.9, EDGE_FLOOR / 2, 0.1]],
                         [[0.5, 0.4, 0.1]]])
    live = torch.tensor([True, True, True, False])
    g = build_graph(slots.reshape(2, 3), r_ut.reshape(2, 3), live)

    assert 3 not in g.unit_ids.tolist(), "a dead unit became a node"
    col = {int(u): i for i, u in enumerate(g.unit_ids)}
    # The sub-floor edge was dropped; the others survive with their weights.
    assert g.A[0, col[1]] == pytest.approx(0.0)
    assert g.A[0, col[0]] == pytest.approx(0.9)
    assert g.A[1, col[1]] == pytest.approx(0.4)


def test_q_route_averages_over_sites_and_stays_in_range() -> None:
    """§01.9's earlier `[0, 1]` was wrong: `Q_B` is bounded in `[-1, 1]`."""
    rng = np.random.default_rng(7)
    slots, rs, lives = [], [], []
    for _ in range(3):
        s = torch.from_numpy(rng.integers(0, 8, size=(24, 2)))
        r = torch.from_numpy(rng.random((24, 2)))
        slots.append(s)
        rs.append(r)
        lives.append(torch.ones(8, dtype=torch.bool))
    out = q_route(slots, rs, lives)
    assert out["n_sites"] == 3
    assert len(out["per_site"]) == 3
    assert -1.0 <= out["Q_route"] <= 1.0
    assert out["Q_route"] == pytest.approx(float(np.mean(out["per_site"])))
