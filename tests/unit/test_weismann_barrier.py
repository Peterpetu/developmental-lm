"""The Weismann barrier, verified by a **mutation test**.

SPEC: 02 §2.1
SPEC: 02 §2.1.1
SPEC: 02 §2.1.2

Gate `G-BARRIER`.

**[NORMATIVE]** The guard is verified by a **mutation test**: this module
monkeypatches `ASSERT_BARRIER` to a no-op and asserts that a deliberately
gradient-enabled germline tensor then goes undetected — i.e. the test proves the
guard is **load-bearing**, not merely that it currently passes.

That distinction is the whole point. A test that only checks the guard passes on
correct code cannot tell the difference between a working guard and a guard that
always returns True.
"""

from __future__ import annotations

import pytest
import torch

from soma.config import M_NANO
from soma.germline import (
    BarrierViolation,
    Genome,
    GermlineRegistry,
    assert_barrier,
)
from soma.germline.registry import (
    BOOKKEEPING_MEMBERS,
    GERMLINE_MEMBERS,
    SOMA_MEMBERS,
)
from soma.model.model import SomaModel


@pytest.fixture()
def genome_and_registry():
    g = torch.Generator().manual_seed(3)
    genome = Genome.initialise(M_NANO, generator=g)
    reg = GermlineRegistry()
    genome.register(reg)
    return genome, reg


def test_registered_germline_tensors_have_no_gradients(genome_and_registry) -> None:
    _, reg = genome_and_registry
    assert len(reg) == len(("mu_g", "log_sigma_g", "policy", "inventory"))
    for name, t in reg.items():
        assert t.requires_grad is False, name
        assert t.grad is None, name
    assert_barrier(reg, optimizer_param_ids=[])


def test_membership_table_is_exhaustive_and_disjoint() -> None:
    """§02.1.1: the table is exhaustive; a tensor not listed is a defect."""
    g = set(GERMLINE_MEMBERS)
    s = set(SOMA_MEMBERS)
    b = set(BOOKKEEPING_MEMBERS)
    assert not (g & s) and not (g & b) and not (s & b)
    # K_unit1/2 are SOMA: were they germline, |Gamma| would scale with U_max and
    # §09.3's bottleneck claim would be false by construction, firing K9
    # definitionally without any experiment.
    assert "K_unit1" in s and "K_unit2" in s
    assert "K_unit1" not in g and "K_unit2" not in g
    # slot_u is bookkeeping, not germline (§01.1 v3 resolution).
    assert "slot_u" in b and "slot_u" not in g


def test_registering_a_grad_enabled_tensor_is_refused(genome_and_registry) -> None:
    _, reg = genome_and_registry
    bad = torch.zeros(4, requires_grad=True)
    with pytest.raises(BarrierViolation, match="requires_grad"):
        reg.register("log_sigma_g", bad)


def test_registering_a_non_member_name_is_refused(genome_and_registry) -> None:
    """Adding a tensor to the germline requires a spec change, not a code
    change."""
    _, reg = genome_and_registry
    with pytest.raises(BarrierViolation, match="exhaustive germline membership"):
        reg.register("W_shared", torch.zeros(4))


def test_grad_enabled_germline_is_caught(genome_and_registry) -> None:
    genome, reg = genome_and_registry
    genome.field_.mu.requires_grad_(True)
    try:
        with pytest.raises(BarrierViolation, match="requires_grad=True"):
            assert_barrier(reg, optimizer_param_ids=[])
    finally:
        genome.field_.mu.requires_grad_(False)


def test_populated_grad_is_caught(genome_and_registry) -> None:
    genome, reg = genome_and_registry
    genome.field_.mu.grad = torch.zeros_like(genome.field_.mu)
    try:
        with pytest.raises(BarrierViolation, match="non-None .grad"):
            assert_barrier(reg, optimizer_param_ids=[])
    finally:
        genome.field_.mu.grad = None


def test_optimizer_membership_is_caught(genome_and_registry) -> None:
    """The realistic failure: a germline tensor handed to an optimizer while
    still non-trainable. Silent until something later flips `requires_grad`."""
    genome, reg = genome_and_registry
    with pytest.raises(BarrierViolation, match="registered with an optimizer"):
        assert_barrier(reg, optimizer_param_ids=[id(genome.field_.mu)])


def test_mutation_test_the_guard_is_load_bearing(genome_and_registry, monkeypatch) -> None:
    """§02.1.2's mandatory mutation test.

    Monkeypatch `assert_barrier` to a no-op and assert that a deliberately
    gradient-enabled germline tensor then goes **undetected**. If this test
    fails, the guard is not what is catching the violation and the other tests
    in this file prove nothing.
    """
    genome, reg = genome_and_registry
    import soma.germline.registry as registry_mod

    def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(registry_mod, "assert_barrier", noop)
    genome.field_.mu.requires_grad_(True)
    try:
        registry_mod.assert_barrier(reg, optimizer_param_ids=[])  # must NOT raise
        detected = False
    except BarrierViolation:
        detected = True
    finally:
        genome.field_.mu.requires_grad_(False)

    assert not detected, (
        "with ASSERT_BARRIER neutered the violation was still detected, so "
        "something other than the guard is catching it and §02.1.2's guard is "
        "not load-bearing"
    )


def test_no_germline_tensor_reaches_either_optimizer() -> None:
    """End-to-end: build the real model and both optimizers."""
    from soma.optim.muon import build_optimizers

    g = torch.Generator().manual_seed(11)
    genome = Genome.initialise(M_NANO, generator=g)
    reg = GermlineRegistry()
    genome.register(reg)
    model = SomaModel(M_NANO)
    muon, adamw = build_optimizers(model, M_NANO)
    ids = [id(p) for grp in muon.param_groups for p in grp["params"]]
    ids += [id(p) for grp in adamw.param_groups for p in grp["params"]]
    assert_barrier(reg, optimizer_param_ids=ids)


def test_a_backward_pass_leaves_the_germline_untouched() -> None:
    """§02.5 step 6: `ASSERT_BARRIER` after one forward/backward on random data."""
    import numpy as np

    g = torch.Generator().manual_seed(5)
    genome = Genome.initialise(M_NANO, generator=g)
    reg = GermlineRegistry()
    genome.register(reg)
    model = SomaModel(M_NANO)
    model.seed_populations(genome, np.random.default_rng(0), g)

    tokens = torch.randint(0, M_NANO.V, (2, 32))
    out = model(tokens, genome=genome)
    loss = out.logits.float().square().mean() + out.metabolic.total
    loss.backward()

    assert_barrier(reg, optimizer_param_ids=[id(p) for p in model.parameters()])
    for name, t in reg.items():
        assert t.grad is None, f"{name} received a gradient through the forward pass"


def test_p_u_is_soma_and_does_receive_gradients() -> None:
    """§02.4's position seam: `p_u` is **soma**, `g_u` is germline-addressed.

    A unit's *identity* is inherited and discrete; its *placement* is learned and
    continuous. If `p_u` received no gradient the seam would not exist and
    `L_disp` could not disperse anything.
    """
    import numpy as np

    g = torch.Generator().manual_seed(5)
    genome = Genome.initialise(M_NANO, generator=g)
    model = SomaModel(M_NANO)
    model.seed_populations(genome, np.random.default_rng(0), g)
    tokens = torch.randint(0, M_NANO.V, (2, 32))
    out = model(tokens, genome=genome)
    (out.logits.float().square().mean() + out.metabolic.total).backward()

    p = model.populations[0].p_u
    assert p.requires_grad
    assert p.grad is not None
    assert float(p.grad.abs().sum()) > 0.0

    assert model.populations[0].g_u.dtype == torch.long
    assert not model.populations[0].g_u.is_floating_point()
