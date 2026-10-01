"""Population stability, the metabolic terms, and the collapse gate.

SPEC: 01 §1.6
SPEC: 01 §1.6.1
SPEC: 03 §3.2.1
SPEC: 03 §3.4
SPEC: 03 §3.5.1
SPEC: 04 §4.2

Gates `G-POPSTABLE`, `G-NOCOLLAPSE`, `G-MUTATE`.

The v1 population went extinct within one selection event: `tau_death` was the
10th percentile of `nu_hat`, killing ~10% of eligible units per event while
births were capped at `rho_0 = 2%`. Deaths structurally exceeded births from the
first event and §04.2's postcondition was violated immediately. These tests are
the regression against that and the three related repairs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from soma.config import M_NANO, NOCOLLAPSE_FRACTION
from soma.development import (
    UtilityState,
    birth,
    calibrate_tau_death,
    crossover,
    death,
    grow,
    mutate,
    nu_hat,
    nu_raw,
    update_utility,
)
from soma.germline import Genome
from soma.germline.field import sample_sphere
from soma.model.metabolic import (
    budget_headroom,
    d_unif,
    dispersion_loss,
    mean_pairwise_distance,
    nocollapse_ok,
    size_loss,
    wiring_loss,
)
from soma.model.model import SomaModel
from soma.selection import CaseReservoir, select

CFG = M_NANO


@pytest.fixture()
def pop_fixture():
    g = torch.Generator().manual_seed(6)
    rng = np.random.default_rng(6)
    genome = Genome.initialise(CFG, generator=g)
    model = SomaModel(CFG)
    model.seed_populations(genome, rng, g)
    return genome, model, genome.field_.neighbour_table(), rng, g


# ---------------------------------------------------------------------------
# G-MUTATE
# ---------------------------------------------------------------------------


def test_mutate_preserves_size_and_uniqueness(pop_fixture) -> None:
    """§03.2.1's postcondition: `|g'| = k_g` exactly, no duplicates.

    Spec v1 had `1 <= |g'| <= k_g` contradicting its own "truncated or padded to
    `|g'| = k_g`" with no padding source.
    """
    genome, model, nb, rng, g = pop_fixture
    pop = model.populations[0]
    for _ in range(2000):
        out, _ = mutate(
            pop.g_u[0].tolist(), 0.0, policy=CFG.policy, N_G=CFG.N_G,
            neighbour_table=nb, rng=rng, nearest_unit_genes=pop.g_u[1].tolist(),
        )
        assert len(out) == CFG.k_g
        assert len(set(out)) == CFG.k_g
        assert all(0 <= x < CFG.N_G for x in out)


def test_mutate_draws_exactly_one_categorical_kind(pop_fixture) -> None:
    """Spec v1 drew three independent Bernoullis while calling the
    probabilities a simplex, so zero to three mutations could occur."""
    genome, model, nb, rng, g = pop_fixture
    pop = model.populations[0]
    kinds = {}
    for _ in range(6000):
        _, oc = mutate(
            pop.g_u[0].tolist(), 0.0, policy=CFG.policy, N_G=CFG.N_G,
            neighbour_table=nb, rng=rng,
        )
        kinds[oc.kind] = kinds.get(oc.kind, 0) + 1
    total = sum(kinds.values())
    assert kinds["point"] / total == pytest.approx(CFG.policy.p_point, abs=0.03)
    assert kinds["dup"] / total == pytest.approx(CFG.policy.p_dup, abs=0.03)
    assert kinds["del"] / total == pytest.approx(CFG.policy.p_del, abs=0.03)


@pytest.mark.parametrize("nu", [0.0, 1.0, 4.0, 9.0])
def test_somatic_hypermutation_rate(pop_fixture, nu: float) -> None:
    """(3.1): the mutation *probability* is inversely proportional to parent
    fitness — the affinity-maturation rule, with no analogue in gradient
    training."""
    genome, model, nb, rng, g = pop_fixture
    pop = model.populations[0]
    applied = sum(
        mutate(
            pop.g_u[0].tolist(), nu, policy=CFG.policy, N_G=CFG.N_G,
            neighbour_table=nb, rng=rng,
        )[1].applied
        for _ in range(4000)
    )
    assert applied / 4000 == pytest.approx(1.0 / (1.0 + nu), abs=0.03)


def test_mutate_falls_back_when_no_nearest_unit_exists(pop_fixture) -> None:
    """§03.2.1: spec v1 gave the "nearest unit" rule no fallback."""
    genome, model, nb, rng, g = pop_fixture
    pop = model.populations[0]
    for _ in range(500):
        out, _ = mutate(
            pop.g_u[0].tolist(), 0.0, policy=CFG.policy, N_G=CFG.N_G,
            neighbour_table=nb, rng=rng, nearest_unit_genes=None,
        )
        assert len(set(out)) == CFG.k_g


def test_crossover_conserves_the_shared_core(pop_fixture) -> None:
    """§03.2.1: a gene both parents express is twice as likely to survive."""
    genome, model, nb, rng, g = pop_fixture
    g1 = list(range(8))
    g2 = list(range(4, 12))
    counts = {}
    for _ in range(4000):
        for x in crossover(g1, g2, rng=rng):
            counts[x] = counts.get(x, 0) + 1
    shared = np.mean([counts[x] for x in range(4, 8)])
    single = np.mean([counts[x] for x in list(range(4)) + list(range(8, 12))])
    assert shared > single * 1.2
    assert all(len(set(crossover(g1, g2, rng=rng))) == 8 for _ in range(100))


def test_crossover_refuses_an_undersized_pool() -> None:
    """A defensive guard: unreachable for valid parents, since two `k_g`-sized
    duplicate-free sets always union to at least `k_g`. Exercised by feeding a
    parent that violates that invariant, so the guard is known to work if the
    invariant is ever broken upstream."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="crossover pool"):
        crossover([1, 1, 2, 3], [1, 2, 3], rng=rng)
    # And the reachable case does not raise.
    assert len(crossover([1, 2, 3], [1, 2, 3], rng=rng)) == 3


# ---------------------------------------------------------------------------
# G-POPSTABLE
# ---------------------------------------------------------------------------


def test_budget_headroom_is_net_of_deaths() -> None:
    """(1.14) / audit A-17: v2 evaluated headroom **before** deaths, and `DEATH`
    keeps `live_u = True` for `phi_ann` more steps — so once the cap was reached
    births were permanently zero."""
    assert budget_headroom(240, 240, 0) == 0
    assert budget_headroom(240, 240, 5) == 5
    assert budget_headroom(240, 100, 0) == 140
    with pytest.raises(ValueError):
        budget_headroom(240, 100, -1)


def test_budget_headroom_excludes_dying_units_from_the_cap() -> None:
    """(1.14) / audit A-62: `DEATH` keeps `live_u = True` for `phi_ann` more
    steps, so the v3.0 expression `U_cap - (|U_live| - n_death)` counted a slot
    as freed `phi_ann` steps before it was. Measured consequence:
    `|U_live| = 243` against `U_cap = 240`."""
    # 240 live, of which 20 are already annealing to death from earlier events.
    # The effective population is 220, so 20 births are affordable -- and the
    # v3.0 expression would have said 0, which is the A-17 failure.
    assert budget_headroom(240, 240, 0, n_dying=20, U_max=256) == 16
    # ...but only 16 *slots* remain, and that is the binding constraint. Grant
    # 20 and the 257th unit has nowhere to go: `FREE_SLOT_NEAREST` raises.
    assert budget_headroom(240, 240, 0, n_dying=20, U_max=280) == 20
    # This event's own deaths count as freeing the cap (they stop routing at
    # once) but not the slots (they anneal like any other).
    assert budget_headroom(240, 240, 5, n_dying=0, U_max=256) == 5
    assert budget_headroom(240, 250, 0, n_dying=0, U_max=256) == 0
    with pytest.raises(ValueError):
        budget_headroom(240, 100, 0, n_dying=-1, U_max=256)


def test_budget_headroom_never_admits_a_unit_the_grid_cannot_hold() -> None:
    """The slot bound of (1.14), swept: for every population and every number of
    concurrently-dying units, `|U_live| + BUDGET_HEADROOM <= U_max`.

    A property rather than a case, because A-62's defect was invisible in every
    individual case that had ever been checked -- it needed a *dying* unit to be
    present at the same event, which no unit test constructed."""
    U_cap, U_max = 240, 256
    checked = 0
    for n_live in range(0, U_max + 1):
        for n_dying in range(0, n_live + 1):
            # (1.14) *preserves* the invariant; it cannot repair a state that
            # already violates it, so states with the effective population
            # already over the cap are out of scope by precondition.
            if n_live - n_dying > U_cap:
                continue
            for n_death in range(0, min(8, n_live - n_dying) + 1):
                h = budget_headroom(
                    U_cap, n_live, n_death, n_dying=n_dying, U_max=U_max
                )
                assert h >= 0
                assert n_live + h <= U_max
                assert (n_live - n_dying - n_death) + h <= U_cap
                checked += 1
    # The sweep must actually contain the states A-62 was found in: a full
    # population with concurrent anneals. Without this the loop above passes
    # vacuously if the ranges are ever narrowed.
    assert checked > 100_000
    assert budget_headroom(U_cap, U_max, 0, n_dying=16, U_max=U_max) == 0


def test_tau_death_is_calibrated_on_the_un_normalised_utility(pop_fixture) -> None:
    """§04.4 / audit A-18: (3.8) divides by the population median, so `nu_hat`
    has median exactly 1 **by construction** and a fixed threshold on it still
    selects a fixed fraction."""
    genome, model, nb, rng, g = pop_fixture
    pop = model.populations[0]
    with torch.no_grad():
        pop.nu_u[pop.live] = torch.rand(int(pop.live.sum()))
        pop.age_u[pop.live] = 500

    nh = nu_hat(pop, CFG.policy.eta_ema)
    assert float(nh[pop.U_death].median()) == pytest.approx(1.0, abs=1e-5), (
        "nu_hat has median 1 by construction, which is why tau_death cannot "
        "live on it"
    )
    tau = calibrate_tau_death(pop, CFG.policy.eta_ema)
    raw = nu_raw(pop, CFG.policy.eta_ema)
    frac = float((raw[pop.U_death] < tau).float().mean())
    assert frac == pytest.approx(0.10, abs=0.03)

    # Now improve everyone; the ABSOLUTE threshold must kill fewer.
    with torch.no_grad():
        pop.nu_u[pop.live] += 1.0
    raw2 = nu_raw(pop, CFG.policy.eta_ema)
    frac2 = float((raw2[pop.U_death] < tau).float().mean())
    assert frac2 < 0.02, (
        "a fitter population must lose fewer units; a percentile recomputed "
        "every event would keep killing exactly 10%"
    )
    assert calibrate_tau_death(pop, CFG.policy.eta_ema) == tau, "tau_death is fixed"


def test_selection_event_respects_every_postcondition(pop_fixture) -> None:
    """§04.2's postconditions, asserted inside `select` and re-checked here."""
    genome, model, nb, rng, g = pop_fixture
    pop = model.populations[0]
    with torch.no_grad():
        pop.age_u[pop.live] = 500
        pop.nu_u[pop.live] = torch.rand(int(pop.live.sum()))

    res = CaseReservoir(CFG.q_cases, CFG.k_route)
    n_tok = 4096
    res.push(
        torch.randint(0, CFG.V, (n_tok,)),
        torch.randint(0, CFG.U_0, (n_tok, CFG.k_route)),
        torch.rand(n_tok, CFG.k_route),
        torch.softmax(torch.randn(n_tok, CFG.k_route), -1),
        torch.rand(n_tok),
        rng,
    )
    n = max(CFG.T_0, CFG.policy.M_mat) + 1
    before = pop.n_live
    ev = select(
        pop, res, n, cfg=CFG, field_=genome.field_, neighbour_table=nb,
        rng_lexicase=rng, rng_mutation=rng, rng_downsample=rng, generator=g,
    )
    assert ev is not None
    cap = int(CFG.policy.rho_0 * math.exp(-n / CFG.n_c) * before)
    assert abs(pop.n_live - before) <= max(cap, len(ev.births))
    assert all(int(pop.age_u[d]) > CFG.policy.M_mat for d in ev.deaths)
    assert not (set(ev.deaths) & set(ev.parents))
    assert res.n_seen == 0, "the reservoir is reset after each event"


def test_population_neither_dies_out_nor_pins_over_many_events(pop_fixture) -> None:
    """`G-POPSTABLE`: over a simulated lifetime the population neither goes
    extinct nor pins at the cap."""
    genome, model, nb, rng, g = pop_fixture
    pop = model.populations[0]
    res = CaseReservoir(CFG.q_cases, CFG.k_route)
    sizes = []
    n = max(CFG.T_0, CFG.policy.M_mat) + 1
    for event in range(12):
        with torch.no_grad():
            pop.age_u[pop.live] += 200
            pop.nu_u[pop.live] = torch.rand(int(pop.live.sum()))
        n_tok = 2048
        live_idx = torch.nonzero(pop.U_parent).squeeze(-1)
        slots = live_idx[torch.randint(0, live_idx.numel(), (n_tok, CFG.k_route))]
        res.push(
            torch.randint(0, CFG.V, (n_tok,)), slots,
            torch.rand(n_tok, CFG.k_route),
            torch.softmax(torch.randn(n_tok, CFG.k_route), -1),
            torch.rand(n_tok), rng,
        )
        select(
            pop, res, n, cfg=CFG, field_=genome.field_, neighbour_table=nb,
            rng_lexicase=rng, rng_mutation=rng, rng_downsample=rng, generator=g,
        )
        for step in range(n, n + CFG.phi_ann + 1):
            retired = pop.step_masks(step)
            if bool(retired.any()):
                pop.retire(torch.nonzero(retired).squeeze(-1))
        n += int(CFG.T_sel(n))
        sizes.append(pop.n_live)

    assert min(sizes) > 0, f"population went extinct: {sizes}"
    assert max(sizes) <= CFG.policy.U_cap
    assert not all(s == CFG.policy.U_cap for s in sizes[3:]), "pinned at the cap"


def test_grow_divides_headroom_by_remaining_events(pop_fixture) -> None:
    """§03.5.1 / audit A-21: v2 saturated `U_cap` on the very first event, so
    all three arms of ablation A8 produced an identical trajectory."""
    from dataclasses import replace

    genome, _, nb, rng, g = pop_fixture
    for fracs in ((0.35,), (0.10, 0.20, 0.35), tuple(0.05 * i for i in range(1, 9))):
        cfg = replace(CFG, growth_schedule_fracs=fracs)
        model = SomaModel(cfg)
        model.seed_populations(genome, np.random.default_rng(1), torch.Generator().manual_seed(1))
        pop = model.populations[0]
        cohorts = []
        for step in cfg.growth_schedule:
            created = grow(
                pop, step, cfg=cfg, field=genome.field_, neighbour_table=nb,
                rng=rng, generator=g,
            )
            cohorts.append(len(created))
            for s in range(step, step + cfg.phi_ann + 1):
                pop.step_masks(s)
        assert all(c > 0 for c in cohorts), (
            f"with {len(fracs)} events the cohorts were {cohorts}; every "
            f"scheduled event must do something or A8 is inert"
        )
        assert pop.n_live <= cfg.policy.U_cap


# ---------------------------------------------------------------------------
# G-NOCOLLAPSE and the metabolic terms
# ---------------------------------------------------------------------------


def test_dispersion_term_separates_collapsed_from_dispersed() -> None:
    """(1.9) / audit A-9: the sphere bounds `L_wire`'s *value*, not its *argmin*;
    collapse is the global minimiser and `W_pos` has no other gradient."""
    g = torch.Generator().manual_seed(2)
    live = torch.zeros(CFG.U_max, dtype=torch.bool)
    live[:64] = True

    dispersed = sample_sphere(CFG.U_max, CFG.m_dim, g)
    collapsed = sample_sphere(1, CFG.m_dim, g).repeat(CFG.U_max, 1)
    collapsed = collapsed + 1e-3 * torch.randn(CFG.U_max, CFG.m_dim, generator=g)
    collapsed = collapsed / collapsed.norm(dim=-1, keepdim=True)

    l_d = float(dispersion_loss(dispersed, live, 1.0))
    l_c = float(dispersion_loss(collapsed, live, 1.0))
    assert l_c > 10 * l_d


def test_nocollapse_gate_has_a_numeric_threshold() -> None:
    """(1.13): v2 referenced the gate without ever defining a threshold."""
    g = torch.Generator().manual_seed(3)
    live = torch.zeros(CFG.U_max, dtype=torch.bool)
    live[:64] = True
    ref = d_unif(CFG.m_dim)

    ok, realised, thresh = nocollapse_ok(
        sample_sphere(CFG.U_max, CFG.m_dim, g), live, CFG.m_dim, ref
    )
    assert ok and realised == pytest.approx(ref, abs=0.05)
    assert thresh == pytest.approx(NOCOLLAPSE_FRACTION * ref)

    p = sample_sphere(1, CFG.m_dim, g).repeat(CFG.U_max, 1)
    ok2, realised2, _ = nocollapse_ok(p, live, CFG.m_dim, ref)
    assert not ok2 and realised2 < 1e-5


def test_dispersion_gradient_disperses_a_collapsed_configuration() -> None:
    """The mechanism, not just the detector: `L_disp`'s equilibrium is a
    dispersed configuration, as in a Thomson problem on `S^{m_dim-1}`."""
    g = torch.Generator().manual_seed(4)
    live = torch.zeros(CFG.U_max, dtype=torch.bool)
    live[:64] = True
    p = sample_sphere(1, CFG.m_dim, g).repeat(CFG.U_max, 1)
    p = p + 1e-2 * torch.randn(CFG.U_max, CFG.m_dim, generator=g)
    p = (p / p.norm(dim=-1, keepdim=True)).requires_grad_(True)

    opt = torch.optim.SGD([p], lr=0.5)
    for _ in range(300):
        opt.zero_grad()
        dispersion_loss(p, live, 1.0).backward()
        opt.step()
        with torch.no_grad():
            p.data /= p.data.norm(dim=-1, keepdim=True)

    ok, realised, thresh = nocollapse_ok(p.detach(), live, CFG.m_dim, d_unif(CFG.m_dim))
    assert ok, f"L_disp alone must clear (1.13); reached {realised:.3f} < {thresh:.3f}"


def test_L_size_has_no_gradient() -> None:
    """§01.6.1 / audit A-31: `m_u` is bookkeeping, so `lambda_size = 0` would be
    a provable no-op — which is why A1b is defined on `U_cap`."""
    live = torch.zeros(CFG.U_max, dtype=torch.bool)
    live[:32] = True
    m_u = torch.rand(CFG.U_max)
    loss = size_loss(m_u, live, CFG.policy.lambda_size)
    assert not loss.requires_grad


def test_wiring_loss_is_weighted_by_routing() -> None:
    """(1.8): the price is on *realised* connections, not possible ones."""
    g = torch.Generator().manual_seed(5)
    q = torch.nn.functional.normalize(torch.randn(2, 8, CFG.m_dim, generator=g), dim=-1)
    p = sample_sphere(CFG.U_max, CFG.m_dim, g)
    slots = torch.randint(0, 64, (2, 8, CFG.k_route), generator=g)
    r_full = torch.softmax(torch.randn(2, 8, CFG.k_route, generator=g), -1)
    high = float(wiring_loss(q, p, slots, r_full, 1.0))
    low = float(wiring_loss(q, p, slots, r_full * 0.1, 1.0))
    assert low == pytest.approx(high * 0.1, rel=1e-5)
    assert float(wiring_loss(q, p, slots, r_full, 0.0)) == 0.0


def test_wiring_loss_ignores_unrouted_slots() -> None:
    g = torch.Generator().manual_seed(6)
    q = torch.nn.functional.normalize(torch.randn(1, 4, CFG.m_dim, generator=g), dim=-1)
    p = sample_sphere(CFG.U_max, CFG.m_dim, g)
    slots = torch.full((1, 4, CFG.k_route), -1, dtype=torch.long)
    r = torch.ones(1, 4, CFG.k_route)
    assert float(wiring_loss(q, p, slots, r, 1.0)) == 0.0


# ---------------------------------------------------------------------------
# Utility (3.5)-(3.8)
# ---------------------------------------------------------------------------


def test_utility_is_a_deviation_not_a_magnitude(pop_fixture) -> None:
    """(3.5b) / audit A-19: `h_bar_u` was used in (3.6) and defined nowhere, so
    the death rule was not implementable. Defining it as a per-unit EMA makes
    `y_u` a **novelty** signal."""
    genome, model, nb, rng, g = pop_fixture
    pop = model.populations[0]
    acc = UtilityState.zeros(CFG.U_max, torch.device("cpu"))
    slots = torch.randint(0, CFG.U_0, (256, CFG.k_route))
    r = torch.softmax(torch.randn(256, CFG.k_route), -1)

    # A unit whose contribution is CONSTANT accumulates no utility, because
    # h_u converges to h_bar_u.
    for _ in range(200):
        acc.accumulate(slots, r, torch.ones(256, CFG.k_route))
        update_utility(pop, acc, genome.field_, model.parts, CFG.policy.eta_ema)
        acc.reset()
    steady = pop.nu_u[pop.live].clone()

    # A unit whose contribution CHANGES does accumulate.
    for i in range(200):
        acc.accumulate(slots, r, torch.full((256, CFG.k_route), 1.0 + 3.0 * (i % 2)))
        update_utility(pop, acc, genome.field_, model.parts, CFG.policy.eta_ema)
        acc.reset()
    varying = pop.nu_u[pop.live]

    assert float(varying.mean()) > float(steady.mean())


def test_nu_hat_is_zero_at_age_zero(pop_fixture) -> None:
    """§03.4.1: `nu_hat_u = 0` at `age_u = 0` (the expression is 0/0
    otherwise)."""
    genome, model, nb, rng, g = pop_fixture
    pop = model.populations[0]
    parent = int(torch.nonzero(pop.U_parent)[0])
    child = birth(
        pop, parent, None, 0, cfg=CFG, field=genome.field_,
        neighbour_table=nb, rng=rng, generator=g,
    )
    assert float(nu_hat(pop, CFG.policy.eta_ema)[child]) == 0.0
    assert float(pop.nu_u[child]) == 0.0, (
        "nu_child = 0, not the parent's value: inheriting it invalidates the "
        "bias correction of (3.8) and inflates young utilities by up to 100x"
    )
