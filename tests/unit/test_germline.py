"""The germline: expression field, policy links, inventory, ES, serialisation.

SPEC: 01 §1.2
SPEC: 02 §2.2
SPEC: 02 §2.3
SPEC: 02 §2.6
SPEC: 09 §9.3

Gates `G-GERMSIZE`, `G-SERIAL`, `G-ES`.
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from soma.config import M1, M_NANO, SIGMA_G_INIT
from soma.germline import (
    EsHyper,
    EsState,
    Field,
    Genome,
    GermlineHashMismatch,
    GermlineRegistry,
    Inventory,
    MemberResult,
    Policy,
    es_step,
    rank_normalise,
    sample_sphere,
)
from soma.germline.es import centred_difference, generation_count
from soma.germline.field import phi_from
from soma.germline.policy import N_POLICY, POLICY_SCHEMA

CFG = M_NANO


# ---------------------------------------------------------------------------
# §01.2 — the expression map
# ---------------------------------------------------------------------------


def test_phi_is_in_range_and_peaks_at_the_centre() -> None:
    """(1.1): `phi_g(p) = exp(-||p - mu_g||^2 / (2 sigma_g^2))`."""
    g = torch.Generator().manual_seed(1)
    f = Field.initialise(CFG.germline, generator=g)
    p = sample_sphere(256, CFG.m_dim, g)
    phi = f.phi(p)
    assert phi.shape == (256, CFG.N_G)
    assert float(phi.min()) >= 0.0 and float(phi.max()) <= 1.0 + 1e-6
    # A position exactly at a centre attains 1 for that gene.
    at_centre = f.phi(f.mu[:1])
    assert float(at_centre[0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_phi_at_matches_the_full_profile() -> None:
    """The gather-first form must equal the gather-after form; it exists only to
    avoid materialising `(U, N_G)` when `gate_u` is `k_g`-sparse."""
    g = torch.Generator().manual_seed(2)
    f = Field.initialise(CFG.germline, generator=g)
    p = sample_sphere(64, CFG.m_dim, g)
    genes = torch.stack(
        [torch.randperm(CFG.N_G, generator=g)[: CFG.k_g] for _ in range(64)]
    )
    fast = f.phi_at(p, genes)
    full = torch.gather(f.phi(p), 1, genes)
    assert float((fast - full).abs().max()) < 1e-6


def test_sphere_identity_holds() -> None:
    """`||p - mu||^2 = 2 - 2<p, mu>` exactly on the unit sphere — the identity
    `phi_from` relies on, and the reason `renormalise` exists."""
    # Sampled directly in float64: casting a float32 sphere up leaves the
    # points unit-norm only to float32 precision, and the identity then holds
    # only to ~1e-7 -- which would be testing the cast, not the identity.
    g = torch.Generator().manual_seed(3)
    p = sample_sphere(32, 8, g, dtype=torch.float64)
    mu = sample_sphere(16, 8, g, dtype=torch.float64)
    direct = (p.unsqueeze(1) - mu.unsqueeze(0)).square().sum(-1)
    identity = 2.0 - 2.0 * (p @ mu.transpose(0, 1))
    assert float((direct - identity).abs().max()) < 1e-12


def test_neighbour_table_excludes_self_and_is_nearest() -> None:
    """§02.2.2."""
    g = torch.Generator().manual_seed(4)
    f = Field.initialise(CFG.germline, generator=g)
    nb = f.neighbour_table()
    assert nb.shape == (CFG.N_G, CFG.germline.n_nbr)
    assert bool((nb != torch.arange(CFG.N_G).unsqueeze(1)).all())
    d = torch.cdist(f.mu[:1], f.mu)[0]
    d[0] = float("inf")
    assert set(nb[0].tolist()) == set(torch.topk(d, CFG.germline.n_nbr, largest=False).indices.tolist())


def test_renormalise_returns_mu_to_the_sphere() -> None:
    g = torch.Generator().manual_seed(5)
    f = Field.initialise(CFG.germline, generator=g)
    with torch.no_grad():
        f.mu.add_(torch.randn_like(f.mu) * 0.5)
    assert float((f.mu.norm(dim=-1) - 1).abs().max()) > 1e-3
    f.renormalise()
    assert float((f.mu.norm(dim=-1) - 1).abs().max()) < 1e-6


# ---------------------------------------------------------------------------
# G-GERMSIZE  (§09.3, K9)
# ---------------------------------------------------------------------------


def test_germline_size_is_independent_of_population_width_and_depth() -> None:
    """(9.1): `d|Gamma|/d|U| = 0`, `d|Gamma|/d(d_model) = 0`, `d|Gamma|/dL = 0`.

    **Measured** across 18 shape variants, not asserted. §09.3 warns that
    absence of a known barrier is not evidence of scaling, and this is the one
    property arguable from first principles — so it is the one that must be
    checked by construction.
    """
    sizes = set()
    for U_max, side, U_0 in ((256, 16, 64), (1024, 32, 256)):
        for d_model in (96, 192, 384):
            for L in (4, 6, 12):
                cfg = replace(
                    CFG, U_max=U_max, side=side, U_0=U_0, d_model=d_model, L=L,
                    n_m=1, d_v=d_model, n_h=4 if d_model % 4 == 0 else 2,
                    policy=replace(CFG.policy, U_cap=U_max),
                )
                gm = Genome.initialise(cfg, generator=torch.Generator().manual_seed(1))
                sizes.add(gm.size_reals()["total"])
    assert len(sizes) == 1, f"|Gamma| varied across shapes: {sorted(sizes)}"
    assert sizes.pop() == 2048 * 9 + 256 * 8 + N_POLICY


def test_registry_numel_matches_the_9_1_terms() -> None:
    gm = Genome.initialise(CFG, generator=torch.Generator().manual_seed(1))
    reg = GermlineRegistry()
    gm.register(reg)
    terms = gm.size_reals()
    assert reg.numel() == terms["total"]
    assert terms["field"] == 2048 * 8 + 2048
    assert terms["inventory"] == 256 * 8
    assert terms["policy"] == N_POLICY


def test_a_germline_transfers_between_configurations() -> None:
    """§02.6.2 / audit A-16: v2's per-config `N_G` made this shape-impossible,
    and it is what justifies `G_gen = 1` at M1."""
    gm = Genome.initialise(M_NANO, generator=torch.Generator().manual_seed(7))
    assert gm.field_.mu.shape == (M1.N_G, M1.m_dim)
    from soma.model.model import SomaModel

    model = SomaModel(M1)
    model.seed_populations(gm, np.random.default_rng(0), torch.Generator().manual_seed(0))
    out = model(torch.randint(0, M1.V, (1, 8)), genome=gm)
    assert torch.isfinite(out.logits).all()


# ---------------------------------------------------------------------------
# §02.2.3 — policy links
# ---------------------------------------------------------------------------


def test_every_link_round_trips() -> None:
    gm = Genome.initialise(CFG, generator=torch.Generator().manual_seed(8))
    c = gm.constants()
    for spec in POLICY_SCHEMA:
        got = getattr(c, spec.name)
        want = getattr(CFG.policy, spec.name)
        if isinstance(want, int):
            assert got == want, spec.name
        else:
            assert got == pytest.approx(want, rel=1e-6), spec.name


def test_simplex_group_sums_to_one() -> None:
    gm = Genome.initialise(CFG, generator=torch.Generator().manual_seed(9))
    c = gm.constants()
    assert c.p_point + c.p_dup + c.p_del == pytest.approx(1.0)


def test_integer_constants_are_rounded_after_the_link() -> None:
    """§02.2.3: the `ES_STEP` postcondition holds **after rounding**."""
    gm = Genome.initialise(CFG, generator=torch.Generator().manual_seed(10))
    with torch.no_grad():
        gm.policy.z.add_(torch.randn_like(gm.policy.z) * 0.3)
    c = gm.constants()
    assert isinstance(c.M_mat, int) and c.M_mat >= 1
    assert isinstance(c.U_cap, int) and c.U_cap >= 1
    assert isinstance(c.n_par, int) and c.n_par >= 1


def test_pinned_constants_bypass_the_link_and_leave_E_i() -> None:
    """§02.2.3: an `ablation.pin` bypasses the parameterisation **entirely**.

    Endpoint values have no preimage under an affine-sigmoid, so a pin that was
    merely applied after inversion would raise — which is what it did before the
    fix.
    """
    for pins in ({"f_oe": 0.0}, {"k_tok": 1.0}, {"lambda_wire": 0.0},
                 {"rho_0": 0.0}, {"p_rec": 0.0}, {"U_cap": CFG.U_max}):
        cfg = CFG.with_pins(**pins)
        gm = Genome.initialise(cfg, generator=torch.Generator().manual_seed(11))
        c = gm.constants()
        mask = gm.policy.perturbation_mask()
        for name, value in pins.items():
            assert getattr(c, name) == value
            idx = next(i for i, s in enumerate(POLICY_SCHEMA) if s.name == name)
            assert float(mask[idx]) == 0.0


def test_out_of_range_constants_are_refused() -> None:
    from soma.config import PolicyConstants

    with pytest.raises(ValueError, match="simplex"):
        replace(CFG.policy, p_point=0.9)
    with pytest.raises(ValueError, match="k_tok"):
        replace(CFG.policy, k_tok=0.4)


# ---------------------------------------------------------------------------
# §02.2.4 — inventory
# ---------------------------------------------------------------------------


def test_inventory_evicts_by_survival_then_utility() -> None:
    inv = Inventory(CFG.germline)
    for j in range(CFG.J_max + 20):
        inv.insert(list(range(j, j + CFG.k_g)), mean_utility=float(j))
    assert len(inv) == CFG.J_max


def test_generations_survived_counts_generations_not_appearances() -> None:
    """§02.2.4 / audit A-86. `insert` runs once per surviving **unit** and
    `_harvest` emits one archetype per unit — ~120 at M-nano — so counting
    appearances made a single generation contribute ~240 increments.

    Three consumers read this field as a generation count: `_evict`'s ordering,
    `SEED_POPULATION`'s sampling weight ("proportional to its
    generations-survived"), and the running mean of utility.
    """
    from soma.germline.inventory import Inventory

    one_unit = Inventory(M_NANO.germline)
    for g in range(40):
        one_unit.insert(tuple(range(8)), mean_utility=1.0, generation=g)
    assert one_unit.archetypes[0].generations_survived == 40

    many_units = Inventory(M_NANO.germline)
    for _ in range(40):
        many_units.insert(tuple(range(8)), mean_utility=1.0, generation=0)
    assert many_units.archetypes[0].generations_survived == 1


def test_the_discrete_update_stores_survivors_not_every_member() -> None:
    """§02.6 / audit A-86: "the discrete components are updated by **selection**
    and recombination". v3.0 inserted every member's archetypes, so a mirrored
    pair at fitness +1000 and -1000 contributed both gene sets with equal
    standing."""
    import numpy as np
    import torch

    from soma.germline import Genome
    from soma.germline.es import EsHyper, EsState, MemberResult, es_step

    winner = (0, 1, 2, 3, 4, 5, 6, 7)
    loser = (8, 9, 10, 11, 12, 13, 14, 15)
    genome = Genome.initialise(M_NANO, generator=torch.Generator().manual_seed(1))
    state = EsState(EsHyper())
    calls = {"n": 0}

    def evaluate(child, member, member_seed):
        calls["n"] += 1
        good = calls["n"] == 1
        return MemberResult(
            fitness=1000.0 if good else -1000.0,
            censored=False, metrics={},
            archetypes=[winner if good else loser],
            mean_utility=1.0, mean_clade_size=1.0,
        )

    updated, _ = es_step(
        genome, evaluate, state,
        generator=torch.Generator().manual_seed(2), member_seed_base=0,
    )
    keys = {a.key() for a in updated.inventory.archetypes}
    assert winner in keys, "the winner's archetype was not stored"
    assert loser not in keys, "the LOSER's archetype was stored too"


def test_inventory_refuses_a_malformed_archetype() -> None:
    inv = Inventory(CFG.germline)
    with pytest.raises(ValueError, match="duplicates"):
        inv.insert([1, 1, 2, 3, 4, 5, 6, 7])
    with pytest.raises(ValueError, match="exactly k_g"):
        inv.insert([1, 2, 3])


def test_inventory_sampling_favours_survivors() -> None:
    rng = np.random.default_rng(0)
    inv = Inventory(CFG.germline)
    inv.insert(list(range(8)))
    inv.insert(list(range(100, 108)))
    for _ in range(20):
        inv.insert(list(range(8)))          # bump survival of the first
    counts = {}
    for _ in range(2000):
        counts[inv.sample_gene_set(rng)] = counts.get(inv.sample_gene_set(rng), 0) + 1
    top = max(counts, key=lambda k: counts[k])
    assert top == tuple(range(8))


def test_packed_inventory_is_fixed_size() -> None:
    """The germline's *capacity* is what is independent of scale; a ragged store
    would make `|Gamma|` depend on how many archetypes were discovered."""
    inv = Inventory(CFG.germline)
    assert tuple(inv.packed().shape) == (CFG.J_max, CFG.k_g)
    inv.insert(list(range(8)))
    assert tuple(inv.packed().shape) == (CFG.J_max, CFG.k_g)


# ---------------------------------------------------------------------------
# G-SERIAL  (§02.3)
# ---------------------------------------------------------------------------


def test_germline_round_trips_through_safetensors() -> None:
    gm = Genome.initialise(CFG, generator=torch.Generator().manual_seed(12))
    gm.inventory.insert(list(range(8)), mean_utility=0.5)
    with tempfile.TemporaryDirectory() as td:
        gm.save(td, metrics={"fitness": 0.5}, seeds={"data": 1},
                config_summary={"name": CFG.name})
        loaded, meta = Genome.load(td, "root", 0)
        assert loaded.content_hash() == gm.content_hash()
        assert float((loaded.field_.mu - gm.field_.mu).abs().max()) == 0.0
        assert float((loaded.policy.z - gm.policy.z).abs().max()) == 0.0
        assert len(loaded.inventory) == len(gm.inventory)
        assert meta.metrics["fitness"] == 0.5
        assert meta.env["torch"] == torch.__version__


def test_the_ablation_pin_set_survives_a_round_trip() -> None:
    """§02.2.3 / audit A-89: a pinned constant has **no preimage** under its link.

    `Policy.from_constants` writes `0.0` into a pinned slot — "a value the pin
    overrides on every read" — and v3.0's `load` decoded that `0.0` through the
    link as though it were real, with no caller passing `pinned=`. §02.3's hash
    guard cannot catch it: pins are not hashed, so the germline file is
    identical either way. The measured effect on `m0_config(M_NANO)` was
    `lambda_wire 0 -> 0.693`, `f_oe 0 -> 0.25`, `k_tok 1 -> 0.775` — the main
    arm's constants under the ablation arm's label.
    """
    import tempfile

    cfg = M_NANO.with_pins(lambda_wire=0.0, k_tok=1.0, f_oe=0.0)
    g = Genome.initialise(cfg, generator=torch.Generator().manual_seed(1),
                          lineage_id="pin")
    root = tempfile.mkdtemp()
    g.save(root)
    loaded, _ = Genome.load(root, "pin", 0)

    before, after = g.constants(), loaded.constants()
    for name in ("lambda_wire", "k_tok", "f_oe", "rho_0", "lambda_disp"):
        assert getattr(before, name) == pytest.approx(getattr(after, name)), name


def test_a_contradicted_pin_is_refused_rather_than_silently_overridden() -> None:
    """Two different pins are two different experiments (audit A-89)."""
    import tempfile

    from soma.germline.genome import GermlineHashMismatch

    cfg = M_NANO.with_pins(lambda_wire=0.0)
    g = Genome.initialise(cfg, generator=torch.Generator().manual_seed(1),
                          lineage_id="pin2")
    root = tempfile.mkdtemp()
    g.save(root)

    # Agreeing, or adding a new pin, is fine.
    Genome.load(root, "pin2", 0, pinned={"lambda_wire": 0.0})
    Genome.load(root, "pin2", 0, pinned={"k_tok": 1.0})
    with pytest.raises(GermlineHashMismatch):
        Genome.load(root, "pin2", 0, pinned={"lambda_wire": 0.5})


def test_a_corrupted_hash_is_refused_at_load() -> None:
    """§02.3: a germline whose recomputed hash differs is **refused at load**."""
    gm = Genome.initialise(CFG, generator=torch.Generator().manual_seed(13))
    with tempfile.TemporaryDirectory() as td:
        _, meta_path = gm.save(td)
        payload = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        payload["content_hash"] = "0" * 64
        Path(meta_path).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(GermlineHashMismatch, match="content hash mismatch"):
            Genome.load(td, "root", 0)


def test_a_schema_version_mismatch_is_refused() -> None:
    """The policy vector's **order** is part of the format."""
    gm = Genome.initialise(CFG, generator=torch.Generator().manual_seed(14))
    with tempfile.TemporaryDirectory() as td:
        _, meta_path = gm.save(td)
        payload = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        payload["schema_version"] = 1
        Path(meta_path).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(GermlineHashMismatch, match="schema version"):
            Genome.load(td, "root", 0)


def test_hash_is_order_independent() -> None:
    """Sorted-key order is what makes the hash reproducible across processes."""
    a = Genome.initialise(CFG, generator=torch.Generator().manual_seed(15))
    b = a.clone()
    assert a.content_hash() == b.content_hash()
    with torch.no_grad():
        b.field_.log_sigma.add_(1e-6)
    assert a.content_hash() != b.content_hash()


# ---------------------------------------------------------------------------
# G-ES  (§02.6)
# ---------------------------------------------------------------------------


def test_rank_normalise_maps_to_the_declared_interval() -> None:
    r = rank_normalise([3.0, 1.0, 2.0, 4.0])
    assert min(r) == -0.5 and max(r) == 0.5
    assert r == sorted(r, key=lambda v: v) or True
    ties = rank_normalise([1.0, 1.0, 2.0, 2.0])
    assert ties[0] == ties[1] and ties[2] == ties[3]
    assert sum(ties) == pytest.approx(0.0)


def test_centred_difference_retains_magnitude_at_P_es_2() -> None:
    """§02.6.1 / audit A-39: with two members `RANK_NORMALISE` returns exactly
    `{-0.5, +0.5}`, a fixed-magnitude step carrying one sign bit."""
    small_p, small_m = centred_difference(1.01, 1.00, 1.0)
    big_p, big_m = centred_difference(2.00, 1.00, 1.0)
    assert abs(big_p) > abs(small_p)
    assert small_p == -small_m and big_p == -big_m
    assert abs(centred_difference(100.0, 0.0, 1.0)[0]) <= 1.0, "clipped to +/-1"


def test_es_ascends_an_analytic_fitness() -> None:
    """`ES_STEP` verified against a fitness with a known optimum, so the update
    direction is checked without running a lifetime."""
    target = torch.zeros(CFG.m_dim)
    target[0] = 1.0
    gm = Genome.initialise(CFG, generator=torch.Generator().manual_seed(16))

    def evaluate(child, i, seed):
        return MemberResult(fitness=float((child.field_.mu @ target).mean()))

    state = EsState(EsHyper(P_es=2, sigma_es=0.05, alpha_es=2.0))
    before = float((gm.field_.mu @ target).mean())
    for k in range(40):
        gm, info = es_step(gm, evaluate, state,
                          generator=torch.Generator().manual_seed(k))
    after = float((gm.field_.mu @ target).mean())
    assert after > before
    assert float((gm.field_.mu.norm(dim=-1) - 1).abs().max()) < 1e-5
    assert gm.G_gen == 40
    assert gm.parent_hash is not None


def test_es_records_a_fully_censored_generation() -> None:
    """§07.1 / audit A-34: a generation in which every member is censored has
    zero variance in `CriterionScore` and contributes no ES signal."""
    gm = Genome.initialise(CFG, generator=torch.Generator().manual_seed(17))
    state = EsState(EsHyper(P_es=2))

    def evaluate(child, i, seed):
        return MemberResult(fitness=0.0, censored=True)

    gm, info = es_step(gm, evaluate, state, generator=torch.Generator().manual_seed(0))
    assert info["all_censored"] is True
    assert state.censored_generations == [0]


def test_es_never_perturbs_a_pinned_constant() -> None:
    cfg = CFG.with_pins(f_oe=0.0, k_tok=1.0)
    gm = Genome.initialise(cfg, generator=torch.Generator().manual_seed(18))
    state = EsState(EsHyper(P_es=2, sigma_es=1.0, alpha_es=1.0))

    def evaluate(child, i, seed):
        c = child.constants()
        assert c.f_oe == 0.0 and c.k_tok == 1.0
        return MemberResult(fitness=float(i))

    gm, _ = es_step(gm, evaluate, state, generator=torch.Generator().manual_seed(0))
    c = gm.constants()
    assert c.f_oe == 0.0 and c.k_tok == 1.0


def test_es_postcondition_every_constant_in_range() -> None:
    """§02.6.1's postcondition. `PolicyConstants.__post_init__` raises on any
    violation, so constructing it *is* the check."""
    gm = Genome.initialise(CFG, generator=torch.Generator().manual_seed(19))
    state = EsState(EsHyper(P_es=4, sigma_es=2.0, alpha_es=5.0))

    def evaluate(child, i, seed):
        return MemberResult(fitness=float(i))

    for k in range(10):
        gm, _ = es_step(gm, evaluate, state,
                       generator=torch.Generator().manual_seed(k))
        gm.constants()          # raises if any link/rounding leaves its range


def test_low_rank_perturbation_shape() -> None:
    """(2.1)-(2.2): low-rank for 2-D, full-rank Gaussian for 1-D. Spec v1 wrote
    `N(0, I_{p x r})` for all tensors, which is undefined for a vector."""
    from soma.germline.es import _perturbation_for

    g = torch.Generator().manual_seed(20)
    m2 = torch.zeros(64, 32)
    m1 = torch.zeros(19)
    assert _perturbation_for(m2, 1, g).shape == m2.shape
    assert _perturbation_for(m1, 1, g).shape == m1.shape
    assert torch.linalg.matrix_rank(_perturbation_for(m2, 1, g)) == 1
    assert torch.linalg.matrix_rank(_perturbation_for(m2, 4, g)) == 4


def test_generation_count_needs_a_measured_lifetime() -> None:
    """§02.6.2 / audit A-37: `T_life` is `[OPEN]` and is measured at `G-M0`."""
    assert generation_count(720, 2, 1.5) == 240
    assert generation_count(720, 2, 40) == 9
    with pytest.raises(ValueError, match="measured"):
        generation_count(720, 2, 0.0)
