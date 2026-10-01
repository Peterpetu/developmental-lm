"""Theorem 3.1a/3.1b: function preservation at birth and at capacity growth.

SPEC: 03 §3.3

Gates `G-MSG`, `G-RMSMASK`.

**Theorem 3.1a (exact function preservation at unit birth).** With
`m_child = 0` and the routing exclusion of §05.4, the Site output (5.20) is
unchanged by the addition of `u_child`, **for any values of the child's `p`,
`g`, or key rows**, up to floating-point reduction order.

**[NORMATIVE]** Claims of *bitwise* equality are withdrawn: adding a
zero-valued summand changes floating-point reduction order. The claim is
equality **up to reduction order**, and the test tolerance is `1e-6`.

The "for any values" clause is what the adversarial cases below exercise. Spec
v1's proof relied on a parenthetical "(implementation requirement)" for softmax
exclusion that §05.4 did not implement, did not exclude the newborn from
**top-k** selection, and rested on a key-table shape that changed with every
birth.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma.config import M_NANO
from soma.development import birth, death, grow
from soma.germline import Genome
from soma.model.model import SomaModel
from soma.model.rmsnorm import masked_rms_norm

TOL = 1e-6


@pytest.fixture()
def fixture():
    g = torch.Generator().manual_seed(4)
    rng = np.random.default_rng(4)
    genome = Genome.initialise(M_NANO, generator=g)
    model = SomaModel(M_NANO)
    model.seed_populations(genome, rng, g)
    nb = genome.field_.neighbour_table()
    tokens = torch.randint(0, M_NANO.V, (2, 32), generator=g)
    return genome, model, nb, rng, g, tokens


def _logits(model, genome, tokens):
    with torch.no_grad():
        return model(tokens, genome=genome, want_metabolic=False).logits.clone()


def test_birth_preserves_the_function_exactly(fixture) -> None:
    genome, model, nb, rng, g, tokens = fixture
    pop = model.populations[0]
    before = _logits(model, genome, tokens)

    parent = int(torch.nonzero(pop.U_parent)[0])
    for _ in range(8):
        birth(pop, parent, None, 10, cfg=M_NANO, field=genome.field_,
              neighbour_table=nb, rng=rng, generator=g)

    after = _logits(model, genome, tokens)
    assert float((after - before).abs().max()) < TOL


def test_birth_preserves_the_function_for_adversarial_child_values(fixture) -> None:
    """"for any values of the child's `p`, `g`, or key rows" — including
    adversarially large ones (§03.3's mandatory test)."""
    genome, model, nb, rng, g, tokens = fixture
    pop = model.populations[0]
    before = _logits(model, genome, tokens)

    parent = int(torch.nonzero(pop.U_parent)[0])
    slots = [
        birth(pop, parent, None, 10, cfg=M_NANO, field=genome.field_,
              neighbour_table=nb, rng=rng, generator=g)
        for _ in range(4)
    ]
    with torch.no_grad():
        pop.p_u.data[slots[0]] = torch.full((M_NANO.m_dim,), 1e6)
        pop.p_u.data[slots[1]] = torch.full((M_NANO.m_dim,), -1e6)
        pop.p_u.data[slots[2]] = torch.zeros(M_NANO.m_dim)
        pop.g_u[slots[3]] = torch.arange(M_NANO.k_g)

    after = _logits(model, genome, tokens)
    assert float((after - before).abs().max()) < TOL


def test_a_newborn_does_not_displace_a_unit_from_top_k(fixture) -> None:
    """§05.4: `active` governs the row/column mask, `cand`, and the softmax
    alike, so a newborn changes no other unit's `r_ut`."""
    genome, model, nb, rng, g, tokens = fixture
    pop = model.populations[0]
    site = model.sites[0]
    x = torch.randn(2, 16, M_NANO.d_model)

    before = site.router(x, pop.active)
    parent = int(torch.nonzero(pop.U_parent)[0])
    for _ in range(6):
        birth(pop, parent, None, 10, cfg=M_NANO, field=genome.field_,
              neighbour_table=nb, rng=rng, generator=g)
    after = site.router(x, pop.active)

    assert torch.equal(before.slots, after.slots)
    assert float((before.weights - after.weights).abs().max()) == 0.0


def test_growth_event_preserves_the_function(fixture) -> None:
    """§03.5.1: every new unit has `m_u = 0`, so Theorem 3.1a applies to each."""
    genome, model, nb, rng, g, tokens = fixture
    pop = model.populations[0]
    before = _logits(model, genome, tokens)
    created = grow(pop, M_NANO.growth_schedule[0], cfg=M_NANO, field=genome.field_,
                   neighbour_table=nb, rng=rng, generator=g)
    assert len(created) > 0
    after = _logits(model, genome, tokens)
    assert float((after - before).abs().max()) < TOL


def test_death_is_only_approximately_preserving(fixture) -> None:
    """§03.4: death is *approximately*, not exactly, function-preserving; the
    mask anneal is what makes the change gradual.

    Spec v1's bias-transfer step is **deleted** — the model has no biases
    (§05.1) and "consumer of `u`" is not an object in this architecture. This
    test records the consequence rather than pretending it does not exist, and
    it is why the bifurcation alarm applies to **births only**.
    """
    genome, model, nb, rng, g, tokens = fixture
    pop = model.populations[0]
    before = _logits(model, genome, tokens)
    victim = int(torch.nonzero(pop.U_death)[0])
    death(pop, victim, 100)
    immediately = _logits(model, genome, tokens)
    # dying_u excludes it from `active` at once, so the change is immediate --
    # which is exactly why it is not function-preserving.
    assert float((immediately - before).abs().max()) > 0.0


def test_mask_anneals_linearly_over_phi_ann(fixture) -> None:
    genome, model, nb, rng, g, tokens = fixture
    pop = model.populations[0]
    parent = int(torch.nonzero(pop.U_parent)[0])
    slot = birth(pop, parent, None, 0, cfg=M_NANO, field=genome.field_,
                 neighbour_table=nb, rng=rng, generator=g)
    seen = []
    for n in range(1, M_NANO.phi_ann + 2):
        pop.step_masks(n)
        seen.append(float(pop.m_u[slot]))
    assert seen[0] == pytest.approx(1.0 / M_NANO.phi_ann, abs=1e-6)
    assert seen[M_NANO.phi_ann - 1] == pytest.approx(1.0)
    assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:])), "anneal must be monotone"


# ---------------------------------------------------------------------------
# Theorem 3.1b and (3.4)  --  G-RMSMASK
# ---------------------------------------------------------------------------


def test_masked_rmsnorm_reduces_to_layernorm_at_zero_mean() -> None:
    """(3.4) `[OURS]`: the derivation is immediate — RMSNorm omits the mean, so
    only the masked second moment needs correction. §03.3 makes this a mandatory
    test of the extension."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 32, generator=g, dtype=torch.float64)
    x = x - x.mean(dim=-1, keepdim=True)          # zero mean
    mask = torch.ones(32, dtype=torch.float64)

    rms = masked_rms_norm(x, mask, eps=0.0)
    # (3.3) LayerNorm at zero mean: var' = mean(x^2), so the two coincide.
    var = (x.square() * mask).sum(-1, keepdim=True) / mask.sum()
    ln = (x - 0.0) * torch.rsqrt(var)
    assert float((rms - ln).abs().max()) < 1e-12


def test_masked_rmsnorm_is_invariant_to_adding_masked_coordinates() -> None:
    """Theorem 3.1b: the grown module's output equals the ungrown module's at
    `c = 0`.

    An unmasked RMSNorm divides by `sqrt(mean over ALL d')`, which shrinks by
    `sqrt(d_old/d_new)` the moment a coordinate is added — so the invariance
    fails before any new weight has a non-zero value.
    """
    g = torch.Generator().manual_seed(2)
    x = torch.randn(16, 8, generator=g, dtype=torch.float64)
    mask = torch.ones(8, dtype=torch.float64)
    base = masked_rms_norm(x, mask, eps=0.0)

    x_grown = torch.cat([x, torch.randn(16, 8, generator=g, dtype=torch.float64) * 1e3], -1)
    mask_grown = torch.cat([mask, torch.zeros(8, dtype=torch.float64)])
    grown = masked_rms_norm(x_grown, mask_grown, eps=0.0)

    assert float((grown[:, :8] - base).abs().max()) < 1e-12
    assert float(grown[:, 8:].abs().max()) == 0.0

    # The positive control: an *unmasked* norm does NOT have this property.
    naive = x_grown * torch.rsqrt(x_grown.square().mean(-1, keepdim=True))
    assert float((naive[:, :8] - base).abs().max()) > 1e-3


def test_width_growth_identity_at_zero_mask() -> None:
    """(3.2): `y' = c (*) sigma_act(W' x) = [y ; 0]`."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(4, 8, generator=g, dtype=torch.float64)
    W = torch.randn(8, 8, generator=g, dtype=torch.float64)
    W_new = torch.randn(8, 8, generator=g, dtype=torch.float64) * 1e6
    W_grown = torch.cat([W, W_new], dim=1)
    c = torch.cat([torch.ones(8, dtype=torch.float64), torch.zeros(8, dtype=torch.float64)])

    y = torch.relu(x @ W).square()
    y_grown = c * torch.relu(x @ W_grown).square()
    assert float((y_grown[:, :8] - y).abs().max()) < 1e-12
    assert float(y_grown[:, 8:].abs().max()) == 0.0
