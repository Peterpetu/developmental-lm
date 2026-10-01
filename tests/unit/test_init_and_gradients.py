"""Initialisation acceptance and finite-difference gradient checks.

SPEC: 01 §1.5
SPEC: 02 §2.5

Gates `G-HYPERFAN`, `G-GRAD`.

**[RATIONALE]** A mis-initialised weight-generating network is reported to fail
to train at any learning rate, with one variant showing a ~270-epoch dead zone,
and batch normalization corrects vanishing but **not** exploding decoded
weights. The check happens before training.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from soma.config import M1, M_NANO
from soma.germline import Genome, GermlineRegistry
from soma.germline.field import Field, sample_sphere
from soma.model.init import (
    GRAD_FLOW_HI,
    GRAD_FLOW_LO,
    GRAD_FLOW_MAX,
    VAR_HI,
    VAR_LO,
    AcceptanceFailed,
    CalibrationFailed,
    acceptance_test,
    apply_calibration,
    calibrate,
    measure_gain_act,
)
from soma.model.model import SomaModel


def _acceptance(cfg, seed: int, *, break_emb_scale: bool = False):
    """Build a model exactly as `LIFETIME` does, then run §02.5.

    The ordering matters and is the ordering the training loop uses:
    seed the global stream, construct, seed the populations, CALIBRATE, apply,
    accept. A test that calibrates in a different order measures a different
    model.
    """
    from soma.germline import GermlineRegistry
    from soma.model.init import acceptance_test, apply_calibration, calibrate
    from soma.utils.rng import RngBundle

    rngs = RngBundle.create(seed, device="cpu")
    genome = Genome.initialise(cfg, generator=torch.Generator().manual_seed(seed))
    reg = GermlineRegistry()
    genome.register(reg)
    rngs.seed_global("init")
    model = SomaModel(cfg)
    if break_emb_scale:
        model.emb_scale = 1.0                      # reinstates audit A-57
    model.seed_populations(genome, rngs.np("init"), rngs.pt("init"))
    cal = calibrate(cfg, genome.field_, model.parts.A.data,
                    generator=rngs.pt("init"), kappa=model.parts.kappa.data,
                    W_ctx=model.sites[0].W_ctx.weight.data)
    apply_calibration(model, cal)
    return acceptance_test(model, genome, reg, generator=rngs.pt("init"),
                           raise_on_failure=False)

CFG = M_NANO


@pytest.fixture()
def built():
    g = torch.Generator().manual_seed(20260828)
    genome = Genome.initialise(CFG, generator=g)
    reg = GermlineRegistry()
    genome.register(reg)
    model = SomaModel(CFG)
    model.seed_populations(genome, np.random.default_rng(1), g)
    return genome, reg, model, g


# ---------------------------------------------------------------------------
# G-HYPERFAN
# ---------------------------------------------------------------------------


def test_calibrate_reaches_the_target_variance(built) -> None:
    """(1.7): `Var(b_g)` is **fitted** so `Var(unit path output) = Var(x)`."""
    genome, reg, model, g = built
    res = calibrate(
        CFG, genome.field_, model.parts.A.data, generator=g,
        kappa=model.parts.kappa.data, W_ctx=model.sites[0].W_ctx.weight.data,
    )
    assert res.realised_var == pytest.approx(res.target_var, rel=0.05)
    assert res.var_b > 0
    assert res.iterations <= 20


def test_calibrate_replaces_a_wrong_analytic_prefactor(built) -> None:
    """Audit A-13: v2 used `N_eff` where `k_g` is correct.

    `N_eff = k_g E[phi]` is a *first* moment while `E2` already carries `phi^2`,
    so the product is cubic in `phi` where the variance is quadratic. The
    realised discrepancy is measured here, and matches the ~3.94x §01.5 records.
    """
    genome, reg, model, g = built
    res = calibrate(
        CFG, genome.field_, model.parts.A.data, generator=g,
        kappa=model.parts.kappa.data, W_ctx=model.sites[0].W_ctx.weight.data,
    )
    var_b_v2 = 1.0 / (res.N_eff * res.E2_hat)      # v2's formula at gain = 1
    ratio = res.var_b / var_b_v2
    # CALIBRATE fits `Var(b) = s / (k_g E2)` with `s` converged, so the ratio is
    # `s N_eff / k_g` -- not a closed form, which is the point: the analytic
    # chain v2 relied on cannot be checked against itself. What can be checked
    # is the SIZE of the discrepancy, and §01.5 records the realised
    # `d_in Var(W)` under v2's formula as `3.94 * gain_act`.
    assert 3.5 < ratio < 4.5, (
        f"the v2 prefactor error measures {ratio:.2f}x; §01.5 records ~3.94x. "
        f"If this drifts, either the field moments or §01.5's recorded figure "
        f"is stale."
    )
    assert res.N_eff == pytest.approx(CFG.k_g * 0.5147, rel=0.05), (
        "N_eff = k_g E[phi] is a FIRST moment; E2 already carries phi^2, which "
        "is why their product is cubic in phi where the variance is quadratic"
    )


def test_calibrate_aborts_on_a_degenerate_field() -> None:
    """§01.5's guards: `N_eff >= 1` and `E2 >= 1e-4` both indicate a field in
    which almost no gene is expressed. At `sigma_g = 0.6`, `N_eff = 0.79`."""
    g = torch.Generator().manual_seed(1)
    bad = Field.initialise(CFG.germline, generator=g, sigma_g=0.6)
    A = torch.randn(CFG.N_G, CFG.d_model) * CFG.d_model ** -0.5
    with pytest.raises(CalibrationFailed, match="N_eff"):
        calibrate(CFG, bad, A, generator=g)


def test_calibrate_requires_the_specified_probe_size() -> None:
    g = torch.Generator().manual_seed(2)
    f = Field.initialise(CFG.germline, generator=g)
    A = torch.randn(CFG.N_G, CFG.d_model) * CFG.d_model ** -0.5
    with pytest.raises(ValueError, match="N_probe >= 4096"):
        calibrate(CFG, f, A, n_probe=100, generator=g)


def test_gain_act_is_measured_not_assumed() -> None:
    """§02.5 step 5. For `relu(z)^2` the gain is far from He's 2 and depends on
    the pre-activation scale — which the gate scales down by `phi * psi ~ 0.25`.
    """
    g = torch.Generator().manual_seed(3)
    for scale in (0.1, 0.25, 1.0, 4.0):
        z = torch.randn(200_000, generator=g) * scale
        gain = measure_gain_act(z)
        assert gain > 0
        # A borrowed constant of 2 would be wrong by a scale-dependent factor.
        assert abs(gain - 2.0) > 0.1 or scale == pytest.approx(1.0, abs=0.6)


def test_acceptance_test_passes_on_a_calibrated_model(built) -> None:
    """§02.5, all six checks."""
    genome, reg, model, g = built
    res = calibrate(
        CFG, genome.field_, model.parts.A.data, generator=g,
        kappa=model.parts.kappa.data, W_ctx=model.sites[0].W_ctx.weight.data,
    )
    apply_calibration(model, res)
    report = acceptance_test(model, genome, reg, generator=g)
    assert report.passed
    assert 0.5 <= report.variance_ratio <= 2.0
    # §02.5 step 2 v3.1: the criterion is the MEDIAN per-block residual-gradient
    # ratio, not the total. The total is depth-dependent even for a healthy
    # stack (13 at L=6, 24 at L=12), which is why the v3.0 form passed at one
    # configuration and failed at the other -- audit A-57.
    assert 0.60 <= report.grad_ratio_per_block <= 1.70
    assert report.grad_ratio > 1.0, "the residual gradient grows toward the input"
    assert report.n_eff_large >= 1.0
    assert report.e2_large >= 1e-4


def test_acceptance_uses_two_independent_estimators(built) -> None:
    """§02.5 step 3: estimate at `N_probe = 4096`, then **independently** at
    `16384`, and assert agreement within 10%. Spec v1 compared an estimator
    against itself, which is vacuous."""
    genome, reg, model, g = built
    res = calibrate(
        CFG, genome.field_, model.parts.A.data, generator=g,
        kappa=model.parts.kappa.data, W_ctx=model.sites[0].W_ctx.weight.data,
    )
    apply_calibration(model, res)
    report = acceptance_test(model, genome, reg, generator=g)
    assert report.n_eff_small != report.n_eff_large, "the estimators must differ"
    assert abs(report.n_eff_small - report.n_eff_large) / report.n_eff_large < 0.10


def test_acceptance_test_aborts_on_a_bad_initialisation(built) -> None:
    """Failure aborts (§02.5). The positive control: an uncalibrated `B` that is
    far too large must be caught."""
    genome, reg, model, g = built
    with torch.no_grad():
        model.parts.B.data.mul_(1000.0)
    with pytest.raises(AcceptanceFailed, match="variance test"):
        acceptance_test(model, genome, reg, generator=g)


# ---------------------------------------------------------------------------
# G-GRAD
# ---------------------------------------------------------------------------


def _finite_difference(fn, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Central differences, in `float64`."""
    grad = torch.zeros_like(x)
    flat = x.reshape(-1)
    gflat = grad.reshape(-1)
    for i in range(flat.numel()):
        old = float(flat[i])
        flat[i] = old + eps
        fp = float(fn())
        flat[i] = old - eps
        fm = float(fn())
        flat[i] = old
        gflat[i] = (fp - fm) / (2 * eps)
    return grad


@pytest.mark.parametrize("cfg", [M_NANO, M1], ids=["m-nano", "m1"])
def test_acceptance_passes_at_both_configurations_and_every_seed(cfg) -> None:
    """§02.5 / audit A-83: the variance band's lower edge **was** the design value.

    `CALIBRATE` targets `Var(y) = 1` against a He-normal reference of `Var = 2`,
    so the ratio is `0.5` by construction — and v3.0's band began at exactly
    `0.5`. Measured at M1: `0.4991, 0.4996, 0.4999, 0.5031` across seeds, so
    **three of four aborted the run at `LIFETIME` step 3**. M-nano's `0.62` sat
    comfortably inside, which is why an M-nano-only test never saw it.
    """
    for seed in (1, 3, 7, 11, 13):
        r = _acceptance(cfg, seed)
        assert r.passed, f"{cfg.name} seed {seed}: {r.failures}"
        assert VAR_LO <= r.variance_ratio <= VAR_HI


@pytest.mark.parametrize("cfg", [M_NANO, M1], ids=["m-nano", "m1"])
def test_the_gradient_flow_test_detects_the_defect_it_was_written_for(cfg) -> None:
    """§02.5 / audit A-83: the **median** per-block ratio cannot see A-57.

    A-57 was a 43–61x discontinuity concentrated in block 0, and a median over
    `L-1` ratios is by construction immune to one outlier: reinstating the defect
    moves the median by `0.04` and the **maximum** by a factor of 15 to 24. The
    positive control is the defect itself — remove the `sqrt(d_model)` embedding
    read-scale and the test must abort.
    """
    for seed in (1, 3, 7):
        healthy = _acceptance(cfg, seed)
        broken = _acceptance(cfg, seed, break_emb_scale=True)

        assert healthy.passed and healthy.grad_ratio_max < GRAD_FLOW_MAX
        assert not broken.passed, "the A-57 defect passed §02.5"
        assert broken.grad_ratio_max > 4 * healthy.grad_ratio_max

        # The median, which v3.0 gated on, barely moves. Stated so the reason
        # for keeping both statistics is visible rather than assumed.
        assert abs(broken.grad_ratio_per_block - healthy.grad_ratio_per_block) < 0.1
        assert GRAD_FLOW_LO <= broken.grad_ratio_per_block <= GRAD_FLOW_HI


def test_mixer_gradients_match_finite_differences() -> None:
    from soma.model.mixer_chunk import gated_delta_rule_chunk

    torch.manual_seed(0)
    B, H, T, dk, dv = 1, 2, 12, 4, 4
    q = torch.randn(B, H, T, dk, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, H, T, dk, dtype=torch.float64)
    k = (k / k.norm(dim=-1, keepdim=True)).requires_grad_(True)
    v = torch.randn(B, H, T, dv, dtype=torch.float64, requires_grad=True)
    a = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64)).requires_grad_(True)
    b = torch.rand(B, H, T, dtype=torch.float64).mul(1.5).requires_grad_(True)

    def loss():
        o, _ = gated_delta_rule_chunk(q, k, v, a, b, chunk_size=4)
        return o.square().sum()

    analytic = torch.autograd.grad(loss(), [q, v, a, b])
    for tensor, g_a in zip((q, v, a, b), analytic):
        with torch.no_grad():
            g_n = _finite_difference(loss, tensor)
        rel = float((g_a - g_n).abs().max()) / (float(g_n.abs().max()) + 1e-12)
        assert rel < 1e-5, f"gradient mismatch {rel:.2e}"


def test_site_gradients_match_finite_differences() -> None:
    """Including through `p_u`, which is the position seam of §02.4."""
    from soma.model.site import PartsLibrary, Site

    torch.manual_seed(1)
    g = torch.Generator().manual_seed(1)
    cfg = M_NANO
    field = Field.initialise(cfg.germline, generator=g)
    field = Field(field.mu.double(), field.log_sigma.double(), cfg.germline)
    parts = PartsLibrary(cfg).double()
    site = Site(cfg, parts).double()

    p_u = sample_sphere(cfg.U_max, cfg.m_dim, g, dtype=torch.float64).requires_grad_(True)
    g_u = torch.stack(
        [torch.randperm(cfg.N_G, generator=g)[: cfg.k_g] for _ in range(cfg.U_max)]
    )
    active = torch.zeros(cfg.U_max, dtype=torch.bool)
    active[:16] = True
    m_u = active.double()
    x = torch.randn(1, 4, cfg.d_model, dtype=torch.float64)

    def loss():
        out = site(x, field=field, p_u=p_u, g_u=g_u, m_u=m_u, active=active)
        return out.y.square().sum()

    analytic = torch.autograd.grad(loss(), p_u)[0]
    sub = p_u.detach()[:4].clone()

    def loss_sub():
        p = p_u.detach().clone()
        p[:4] = sub
        out = site(x, field=field, p_u=p, g_u=g_u, m_u=m_u, active=active)
        return out.y.square().sum()

    with torch.no_grad():
        num = _finite_difference(loss_sub, sub, eps=1e-6)
    rel = float((analytic[:4] - num).abs().max()) / (float(num.abs().max()) + 1e-12)
    assert rel < 1e-4, f"p_u gradient mismatch {rel:.2e}"


def test_metabolic_gradients_match_finite_differences() -> None:
    """(1.8)-(1.9) carry gradients through `p_u` and `W_pos`; (1.10) does not."""
    from soma.model.metabolic import dispersion_loss, wiring_loss

    torch.manual_seed(2)
    g = torch.Generator().manual_seed(2)
    cfg = M_NANO
    live = torch.zeros(cfg.U_max, dtype=torch.bool)
    live[:12] = True
    p = sample_sphere(cfg.U_max, cfg.m_dim, g, dtype=torch.float64).requires_grad_(True)

    def loss():
        return dispersion_loss(p, live, 1.0)

    analytic = torch.autograd.grad(loss(), p)[0]
    sub = p.detach()[:3].clone()

    def loss_sub():
        pp = p.detach().clone()
        pp[:3] = sub
        return dispersion_loss(pp, live, 1.0)

    with torch.no_grad():
        num = _finite_difference(loss_sub, sub, eps=1e-6)
    rel = float((analytic[:3] - num).abs().max()) / (float(num.abs().max()) + 1e-12)
    assert rel < 1e-4


def test_full_model_backward_produces_finite_gradients(built) -> None:
    genome, reg, model, g = built
    tokens = torch.randint(0, CFG.V, (2, 24), generator=g)
    out = model(tokens, genome=genome)
    loss = out.logits.float().square().mean() + out.metabolic.total
    loss.backward()
    n_with_grad = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"
            n_with_grad += 1
    assert n_with_grad > 0.8 * len(list(model.parameters()))


def test_token_selection_gradient_flows_only_through_selected_tokens() -> None:
    """(6.5): the loss is normalised over the **realised** selected count."""
    from soma.openended.tokensel import ExploreState, masked_cross_entropy, select_tokens

    torch.manual_seed(3)
    logits = torch.randn(2, 8, 16, requires_grad=True)
    targets = torch.randint(0, 16, (2, 8))
    include = torch.zeros(2, 8, dtype=torch.bool)
    include[0, :4] = True

    L, Lz, per_tok = masked_cross_entropy(logits, targets, include)
    L.backward()
    gz = logits.grad
    assert float(gz[0, :4].abs().sum()) > 0
    assert float(gz[0, 4:].abs().sum()) == 0.0
    assert float(gz[1].abs().sum()) == 0.0
    assert per_tok.shape == (2, 8)


def test_z_loss_shares_the_denominator_with_L_task() -> None:
    """§05.6: averaged over **the same token set** as `L_task`, so the two share
    a denominator and A5 cannot silently change the regulariser."""
    from soma.openended.tokensel import masked_cross_entropy

    torch.manual_seed(4)
    logits = torch.randn(2, 8, 16)
    targets = torch.randint(0, 16, (2, 8))
    full = torch.ones(2, 8, dtype=torch.bool)
    half = torch.zeros(2, 8, dtype=torch.bool)
    half[:, :4] = True

    _, z_full, _ = masked_cross_entropy(logits, targets, full, lambda_z=1.0)
    _, z_half, _ = masked_cross_entropy(logits, targets, half, lambda_z=1.0)
    logZ = torch.logsumexp(logits.float(), -1)
    assert float(z_full) == pytest.approx(float(logZ.square().mean()), rel=1e-5)
    assert float(z_half) == pytest.approx(
        float(logZ[:, :4].square().mean()), rel=1e-5
    )
