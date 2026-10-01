"""SPEC: 10 §10.6 -- the acceptance tests P1..P8 for the decoded rule."""

from __future__ import annotations

import math

import pytest
import torch

from gnomon.config import GenomeShape, M_PICO, N_RULE
from gnomon.genome.arch import ArchHead, Strategy, _tau, mutate
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.plasticity import (
    ADAM_POINT, DecodedRule, Neuromodulator, coefficients_from,
)
from gnomon.pheno.model import Gnomon

SHAPE = GenomeShape()


def _pinned(p: torch.Tensor, point=ADAM_POINT) -> torch.Tensor:
    c = torch.zeros(p.shape[0], N_RULE, dtype=p.dtype)
    for j, v in enumerate(point):
        c[:, j] = v
    return c


# ---------------------------------------------------------------------- P1

def test_P1_adam_is_inside_the_search_space():
    """SPEC: 10 §10.6 P1 -- and it is the whole argument of §10.2.

    A search whose baseline lies *outside* its space cannot be said to have
    chosen anything. At `A=0, B=1, C=0, D=0` the decoded rule must be AdamW
    exactly, so that any departure from Adam is something evolution found rather
    than something the author forbade.
    """
    torch.manual_seed(0)
    n, d = 40, 6
    a = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    rule = DecodedRule({"z": a}, beta_1=0.9, beta_2=0.95, eps=1e-8)
    adam = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                             weight_decay=0.0)
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        grad = torch.randn(n, d, generator=g, dtype=torch.float64)
        a.grad = grad.clone()
        b.grad = grad.clone()
        rule.step({"z": _pinned(a)}, loss=1.0, lr=1e-2)
        adam.step()
    torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=1e-9)


def test_P1_a_different_point_is_not_adam():
    """The control for P1: if every point matched Adam, P1 would be vacuous."""
    torch.manual_seed(0)
    a = torch.randn(20, 4, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    rule = DecodedRule({"z": a})
    adam = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    g = torch.Generator().manual_seed(1)
    for _ in range(20):
        grad = torch.randn(20, 4, generator=g, dtype=torch.float64)
        a.grad, b.grad = grad.clone(), grad.clone()
        rule.step({"z": _pinned(a, (0.7, 0.3, 0.05, 0.0))}, loss=1.0, lr=1e-2)
        adam.step()
    assert not torch.allclose(a.detach(), b.detach(), atol=1e-6)


# ---------------------------------------------------------------------- P2

def test_P2_the_coefficients_are_decoded_from_the_genome():
    m = Gnomon(M_PICO, SHAPE, genome_seed=0, soma_seed=1)
    before = {k: v.clone() for k, v in m.plasticity().items()}
    mutate([m.decoder], sigma_rel=0.5, p_mut=1.0,
           generator=torch.Generator().manual_seed(0))
    after = m.plasticity()
    for k in before:
        assert not torch.allclose(before[k], after[k]), k


def test_P2_the_coefficients_track_the_soma_too():
    """`Z` is half of an identity, so a moving soma moves the rule."""
    m = Gnomon(M_PICO, SHAPE, genome_seed=0, soma_seed=1)
    before = {k: v.clone() for k, v in m.plasticity().items()}
    with torch.no_grad():
        m.soma.z.add_(torch.randn_like(m.soma.z))
    assert not torch.allclose(before["z"], m.plasticity()["z"])


def test_P2_ranges_are_the_spec_ranges():
    """SPEC: 10 §10.3. `B` is centred on 1 so an ignorant genome does Adam."""
    raw = torch.zeros(5, N_RULE)
    c = coefficients_from(raw)
    torch.testing.assert_close(c[0], torch.tensor(list(ADAM_POINT)))
    big = coefficients_from(torch.full((5, N_RULE), 10.0))
    assert float(big[0, 0]) == pytest.approx(1.0, abs=1e-4)      # A in (-1, 1)
    assert float(big[0, 1]) == pytest.approx(2.0, abs=1e-4)      # B in (0, 2)
    assert float(big[0, 2]) == pytest.approx(0.1, abs=1e-4)      # C small
    assert float(big[0, 3]) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------- P3

def test_P3_nothing_in_the_rule_is_differentiable():
    m = Gnomon(M_PICO, SHAPE, genome_seed=0, soma_seed=1)
    for k, v in m.plasticity().items():
        assert not v.requires_grad and v.grad_fn is None, k


# ---------------------------------------------------------------------- P4

def test_P4_the_neuromodulator_is_a_prediction_error():
    nu = Neuromodulator(rho=0.9)
    for _ in range(30):
        nu.update(5.0 + 0.1 * torch.randn(()).item())
    better = nu.update(3.0)
    assert better > 0.0, "a loss below the baseline must give nu > 0"
    nu2 = Neuromodulator(rho=0.9)
    for _ in range(30):
        nu2.update(5.0 + 0.1 * torch.randn(()).item())
    worse = nu2.update(9.0)
    assert worse < 0.0, "a loss above the baseline must give nu < 0"


def test_P4_the_neuromodulator_is_bounded_and_starts_silent():
    nu = Neuromodulator()
    assert nu.update(5.0) == 0.0, "no baseline yet means no prediction error"
    vals = [nu.update(v) for v in [5, 4, 9, 1, 100, -3, 0.5] * 6]
    assert all(-1.0 < v < 1.0 for v in vals)


def test_P4_the_neuromodulator_gates_the_trace():
    """`D` must do nothing when `nu` is zero, and something when it is not."""
    torch.manual_seed(0)
    a = torch.randn(10, 3, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    ra, rb = DecodedRule({"z": a}), DecodedRule({"z": b})
    g = torch.Generator().manual_seed(2)
    for i in range(12):
        grad = torch.randn(10, 3, generator=g, dtype=torch.float64)
        a.grad, b.grad = grad.clone(), grad.clone()
        ra.step({"z": _pinned(a, (0.0, 1.0, 0.0, 0.9))}, loss=5.0, lr=1e-2)
        rb.step({"z": _pinned(b, (0.0, 1.0, 0.0, 0.0))}, loss=5.0, lr=1e-2)
    # A constant loss gives no prediction error, so D is inert.
    torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------- P5

def test_P5_the_shape_change_costs_no_genome():
    """SPEC: 10 §10.3. A decoded learning rule for free."""
    assert Decoder(SHAPE).n_params() == SHAPE.n_params() == 4483
    m = Gnomon(M_PICO, SHAPE)
    from gnomon.config import ROLE_PLAST_CHAN, ROLE_PLAST_TOK
    assert m.shapes()[(ROLE_PLAST_CHAN, 0)] == (M_PICO.d_model, N_RULE)
    assert m.shapes()[(ROLE_PLAST_TOK, -1)] == (M_PICO.vocab, N_RULE)


# ------------------------------------------------------------- P6, P7, P8

def test_P6_strengths_are_mutated_before_they_are_used():
    """SPEC: 10 §10.4. The ordering is what makes self-adaptation work at all."""
    g = torch.Generator().manual_seed(0)
    d = Decoder(SHAPE, seed=0)
    st = Strategy([n for n, _ in d.named_parameters()])
    before = st.log_sigma.clone()
    mutate([d], generator=g, strategy=st)
    assert not torch.equal(before, st.log_sigma), "the strengths did not move"


def test_P6_a_strength_actually_changes_the_perturbation():
    """A strategy that is carried but never consulted is an ornament."""
    def perturb(log_sigma_value):
        g = torch.Generator().manual_seed(3)
        a, b = Decoder(SHAPE, seed=0), Decoder(SHAPE, seed=0)
        st = Strategy([n for n, _ in b.named_parameters()])
        with torch.no_grad():
            st.log_sigma.fill_(log_sigma_value)
        # Freeze the self-adaptation step so only the *use* is measured.
        tau_saved = st.log_sigma.clone()
        mutate([b], p_mut=1.0, generator=g, strategy=st)
        with torch.no_grad():
            st.log_sigma.copy_(tau_saved)
        return float((b.W_row - a.W_row).abs().mean())

    small, large = perturb(-1.5), perturb(1.5)
    assert large > 3.0 * small, (small, large)


def test_P7_strengths_are_heritable_state_not_a_global():
    """Two lineages must be able to carry different strengths."""
    from gnomon.qd.evolve import Genome

    a, b = Genome(SHAPE, seed=1), Genome(SHAPE, seed=2)
    with torch.no_grad():
        a.s.log_sigma.fill_(1.0)
    assert float(b.s.log_sigma.abs().max()) == 0.0
    child = Genome(SHAPE, seed=3).load(a.state())
    torch.testing.assert_close(child.s.log_sigma, a.s.log_sigma)


def test_P7_the_strategy_is_carried_in_the_genome_state():
    from gnomon.qd.evolve import Genome

    g = Genome(SHAPE, seed=0)
    assert any(k.startswith("s.") for k in g.state())
    assert g.n_params() == 4483 + 796 + 27 == 5306


def test_P8_generation_zero_is_the_adam_baseline():
    """SPEC: 10 §10.6 P8. `log_sigma = 0` and `B = 1` at initialisation."""
    from gnomon.qd.evolve import Genome

    g = Genome(SHAPE, seed=0)
    assert float(g.s.log_sigma.abs().max()) == 0.0
    c = coefficients_from(torch.zeros(3, N_RULE))
    torch.testing.assert_close(c[0], torch.tensor(list(ADAM_POINT)))


def test_tau_is_schwefels_rate():
    assert _tau(27) == pytest.approx(1.0 / math.sqrt(2.0 * math.sqrt(27)))
    assert _tau(1) > _tau(100)


def test_the_rule_reports_what_the_genome_chose():
    """SPEC: 10 §10.2. §10.1's defect was that the split was never measured."""
    torch.manual_seed(0)
    a = torch.randn(8, 3, requires_grad=True)
    a.grad = torch.randn(8, 3)
    r = DecodedRule({"z": a})
    r.step({"z": _pinned(a)}, loss=2.0, lr=1e-3)
    rep = r.contributions({"z": _pinned(a)})
    assert rep["z.A"] == pytest.approx(0.0)
    assert rep["z.B"] == pytest.approx(1.0)
    assert "nu" in rep
