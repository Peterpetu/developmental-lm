"""Initialisation: `CALIBRATE` and the §02.5 acceptance test.

SPEC: 01 §1.5
SPEC: 02 §2.5

```
Var(a_g) = 1 / d_in                                                       (1.6)
Var(b_g) : chosen by CALIBRATE so that Var( unit path output ) = Var(x)   (1.7)
           for x ~ N(0, I) rescaled to ||x|| = sqrt(d_in)
```

**Why there is no closed form.** The Site's unit path is nonlinear and
`relu(.)^2` is positively homogeneous of degree 2, so no single fan-in rule
determines the scale. Spec v2 asserted
`Var(W_ij) = N_eff * E2 * Var(a) * Var(b)` with `N_eff = E_u[||phi (*) 1_g||_1]`;
for a sum of `k_g` independent rank-one terms the correct prefactor is **`k_g`**,
since `N_eff = k_g E[phi]` is a *first* moment while `E2` already carries
`phi^2`, so the product is cubic in `phi` where the variance is quadratic.
Measured at the v2 initialisation the realised `d_in Var(W)` is `3.94 gain_act`
— outside §02.5's own `[0.5, 2.0]` band, so v2 as written would have aborted
every run at `LIFETIME` step 3 (audit A-13).

Calibration removes the fragile analytic chain entirely: the acceptance test
becomes the definition rather than a check on a formula that may itself be
wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from soma.config import SomaConfig
from soma.germline.field import Field, sample_sphere
from soma.germline.genome import Genome
from soma.germline.registry import GermlineRegistry, assert_barrier

__all__ = [
    "CalibrationFailed",
    "AcceptanceFailed",
    "CalibrationResult",
    "calibrate",
    "acceptance_test",
    "measure_gain_act",
]


GRAD_FLOW_LO = 0.60
GRAD_FLOW_HI = 1.70
VAR_TARGET = 0.5
"""The **design** value of `Var(y)/Var(y_ref)` (§02.5).

`CALIBRATE` drives the unit-path output variance to `Var(y) = 1`, and the He-normal
reference has `Var(y_ref) = 2`, so the ratio is `0.5` **by construction**."""

VAR_LO = 0.25
VAR_HI = 1.00
"""A factor of two either side of `VAR_TARGET`, in log.

**[NORMATIVE — v3.1, audit A-83]** v3.0's band was `[0.5, 2.0]`, whose **lower
edge is the design value itself**. A test whose pass region begins exactly at the
quantity's expected value rejects roughly half of all seeds, and it did: measured
at M1, `Var(y)/Var(y_ref) = 0.4991, 0.4996, 0.4999, 0.5031` across four seeds —
**three of four abort the run at `LIFETIME` step 3** with
`AcceptanceFailed`. §07.5 requires `n_seed >= 5`; a quarter to a half of them
would not have started. M-nano's `0.62-0.64` sat comfortably inside, which is
why the M-nano-only unit test never saw it — the same one-configuration blindness
as A-32.

The intent of §02.5's variance test is "within a factor of two of the reference",
and the band now expresses that around the value the calibration actually
targets, rather than around a reference the calibration deliberately does not
reproduce."""

GRAD_FLOW_MAX = 8.0
"""Bound on the **largest** per-block residual-gradient ratio (§02.5).

**[NORMATIVE — v3.1, audit A-83]** The median alone could not detect the defect
it was introduced for. A-57 was a `43-61x` gradient discontinuity concentrated in
**block 0**, and a median over `L - 1` ratios is by construction immune to one
outlier. Measured, healthy against the A-57 defect reinstated:

| | median (gated in v3.0) | **max** | total |
|---|---|---|---|
| M-nano, healthy   | 1.37 – 1.43 | **2.96 – 3.22** | 12 – 16 |
| M-nano, A-57      | 1.43 – 1.47 | **42.4 – 52.3** | 256 – 301 |
| M1, healthy       | 1.21 – 1.23 | **2.64 – 2.75** | 23 – 25 |
| M1, A-57          | 1.23 – 1.25 | **60.9 – 63.9** | 733 – 779 |

The median moves by `0.04`; the maximum moves by a factor of **15 to 24**, and
does so identically at both depths — which the *total* does not, since it
compounds over `L`. `8.0` sits 2.5x above the worst healthy value and 5x below
the mildest defective one.

The two statistics are kept as a pair on purpose: the median is robust and
catches a *systematic* drift across the stack, the maximum is sensitive and
catches a *single* discontinuous block. A-57's own conclusion — "a check that
measures the wrong quantity does not merely fail to catch a problem, it
certifies its absence" — applied verbatim to the repair for A-57."""
"""§02.5 step 2: the band on the **median** per-block residual-gradient ratio.

A healthy Peri-LN stack measures 1.38 (M-nano) and 1.21 (M1). Genuine pathology
is orders of magnitude away: before the embedding read-scaling of §05.6 the
first block alone measured 43x-61x, and a vanishing stack would sit far below
0.6."""


class CalibrationFailed(RuntimeError):
    """`CALIBRATE` did not converge within `iters` (§01.5)."""


class AcceptanceFailed(RuntimeError):
    """§02.5's acceptance test failed. Failure aborts the run."""


@dataclass
class CalibrationResult:
    var_b: float
    E2_hat: float
    N_eff: float
    realised_var: float
    target_var: float
    iterations: int
    gain_act: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "var_b": self.var_b,
            "E2_hat": self.E2_hat,
            "N_eff": self.N_eff,
            "realised_var": self.realised_var,
            "target_var": self.target_var,
            "iterations": self.iterations,
            "gain_act": self.gain_act,
        }


def _probe_inputs(
    n: int, d: int, generator: Optional[torch.Generator], device, dtype
) -> torch.Tensor:
    """`x ~ N(0, I)` rescaled to `||x|| = sqrt(d)` (§01.5, §02.5)."""
    x = torch.randn(n, d, generator=generator, device=device, dtype=dtype)
    return x * (math.sqrt(d) / x.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def measure_gain_act(pre_activation: torch.Tensor) -> float:
    """§02.5 step 5: the variance gain of `relu(.)^2` on the realised
    pre-activation distribution.

    Measured numerically rather than assumed to be 2 (the He constant for a
    plain ReLU). For `relu(z)^2` at `z ~ N(0, s^2)` the gain is far from 2 and
    depends on `s`, so a borrowed constant would be wrong by a factor that grows
    with the pre-activation scale — which is exactly the regime the Site
    operates in, since `gate` scales `z` down by `phi * psi ~ 0.25`.
    """
    z = pre_activation.float()
    v_in = float(z.var())
    v_out = float(torch.relu(z).square().var())
    return v_out / max(v_in, 1e-12)


@torch.no_grad()
def calibrate(
    cfg: SomaConfig,
    field: Field,
    A: torch.Tensor,
    *,
    n_probe: int = 4096,
    tol: float = 0.05,
    iters: int = 20,
    generator: Optional[torch.Generator] = None,
    kappa: Optional[torch.Tensor] = None,
    W_ctx: Optional[torch.Tensor] = None,
) -> CalibrationResult:
    """`CALIBRATE` of (1.7).

    SPEC: 01 §1.5

    ```
    precondition: A initialised per (1.6); N_probe >= 4096 units sampled from §02.2.1
    E2_hat <- Monte-Carlo estimate of E_{u, g in g_u}[ phi_g(p_u)^2 psi_g^2 ]
    s <- 1.0
    repeat iters times:
        draw B with Var(b_g) = s / (k_g * E2_hat)
        v <- measured Var over probes of  B[ relu( gate(u,x) (*) (A x) )^2 ]
        if | v / Var(x) - 1 | <= tol:  return Var(b_g)
        s <- s * ( Var(x) / v )                # secant on a monotone positive scalar
    raise CalibrationFailed
    postcondition: measured output variance within tol of Var(x); the fitted
                   value is recorded in the run manifest
    ```

    The secant update is exact for a linear response and converges in one step
    when the map `s -> v` is proportional. It is not: `relu(.)^2` makes `v`
    proportional to `Var(b)` (the outer map *is* linear in `B`, since `B` enters
    once), so convergence is in fact immediate — the loop exists because the
    Monte-Carlo estimate of `v` is noisy, and the tolerance is on the estimate.
    """
    if n_probe < 4096:
        raise ValueError(f"§01.5 requires N_probe >= 4096; got {n_probe}")
    device = A.device
    dtype = A.dtype
    d_in = A.shape[1]

    p_u = sample_sphere(n_probe, cfg.m_dim, generator, device=device, dtype=dtype)
    g_u = torch.stack(
        [
            torch.randperm(cfg.N_G, generator=generator, device=device)[: cfg.k_g]
            for _ in range(n_probe)
        ]
    )
    phi = field.phi_at(p_u, g_u)                             # (n_probe, k_g)

    x = _probe_inputs(n_probe, d_in, generator, device, dtype)
    target_var = float(x.var())

    z = x @ A.transpose(0, 1)                                # (n_probe, N_G)
    z_sel = torch.gather(z, 1, g_u)                          # (n_probe, k_g)

    if kappa is not None and W_ctx is not None:
        c = x @ W_ctx.transpose(0, 1)
        psi_full = torch.sigmoid((c @ kappa.transpose(0, 1)) / math.sqrt(kappa.shape[1]))
        psi = torch.gather(psi_full, 1, g_u)
    else:
        # Without a context projection the expectation of psi is 1/2 by symmetry
        # of the sigmoid about zero; using the constant makes E2_hat an upper
        # bound rather than an estimate, and the loop corrects the difference.
        psi = torch.full_like(phi, 0.5)

    E2_hat = float((phi.square() * psi.square()).mean())
    N_eff = float(phi.sum(dim=-1).mean())

    # §01.5 guards: both indicate a degenerate expression field.
    if N_eff < 1.0:
        raise CalibrationFailed(
            f"N_eff = {N_eff:.4f} < 1: almost no gene is expressed. At "
            f"sigma_g = 0.6 this is 0.79, which is why §01.2 sets 1.2."
        )
    if E2_hat < 1e-4:
        raise CalibrationFailed(f"E2_hat = {E2_hat:.3e} < 1e-4: degenerate field")

    gate = phi * psi
    act = torch.relu(gate * z_sel).square()                  # (n_probe, k_g)
    gain_act = measure_gain_act(gate * z_sel)

    s = 1.0
    var_b = s / (cfg.k_g * E2_hat)
    realised = float("nan")
    for it in range(1, iters + 1):
        var_b = s / (cfg.k_g * E2_hat)
        B_cols = torch.randn(
            n_probe, cfg.k_g, cfg.d_model, generator=generator, device=device, dtype=dtype
        ) * math.sqrt(var_b)
        y = torch.einsum("ng,ngd->nd", act, B_cols)          # (n_probe, d_model)
        realised = float(y.var())
        if realised <= 0.0:
            raise CalibrationFailed("unit path output has zero variance")
        if abs(realised / target_var - 1.0) <= tol:
            return CalibrationResult(
                var_b, E2_hat, N_eff, realised, target_var, it, gain_act
            )
        s = s * (target_var / realised)

    raise CalibrationFailed(
        f"CALIBRATE did not reach |v/Var(x) - 1| <= {tol} in {iters} iterations; "
        f"last realised variance {realised:.4f} against target {target_var:.4f}"
    )


# ---------------------------------------------------------------------------
# The acceptance test (§02.5)
# ---------------------------------------------------------------------------


@dataclass
class AcceptanceReport:
    variance_ratio: float
    grad_ratio: float
    grad_ratio_per_block: float
    grad_ratio_max: float
    per_block_steps: tuple
    n_eff_small: float
    n_eff_large: float
    e2_small: float
    e2_large: float
    gain_act: float
    calibration: CalibrationResult
    passed: bool
    failures: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "variance_ratio": self.variance_ratio,
            "grad_ratio": self.grad_ratio,
            "grad_ratio_per_block": self.grad_ratio_per_block,
            "grad_ratio_max": self.grad_ratio_max,
            "per_block_steps": list(self.per_block_steps),
            "n_eff_small": self.n_eff_small,
            "n_eff_large": self.n_eff_large,
            "e2_small": self.e2_small,
            "e2_large": self.e2_large,
            "gain_act": self.gain_act,
            "calibration": self.calibration.as_dict(),
            "passed": self.passed,
            "failures": list(self.failures),
        }


def acceptance_test(
    model,
    genome: Genome,
    registry: GermlineRegistry,
    *,
    generator: Optional[torch.Generator] = None,
    raise_on_failure: bool = True,
) -> AcceptanceReport:
    """§02.5, executed **after** phenotype construction.

    SPEC: 02 §2.5

    1. **Variance test.** Draw `x ~ N(0, I_{d_model})` rescaled to
       `||x|| = sqrt(d_model)`. For `N_probe = 4096` sampled units compute the
       realised Site output variance. Assert `0.5 <= Var(y)/Var(y_ref) <= 2.0`,
       where `y_ref` uses `W ~ N(0, 2/d_model)` (He normal).
    2. **Gradient-flow test.** Ratio of gradient norms between the first and
       last block in `[0.1, 10]` at step 0.
    3. **`N_eff`/`E2` cross-check.** Estimate with `N_probe = 4096` at init,
       then **independently** with `N_probe = 16384`; assert agreement within
       10%. Spec v1 compared an estimator against itself, which is vacuous.
    4. **Guards.** `N_eff >= 1`, `E2 >= 1e-4`.
    5. **`gain_act` calibration**, recorded in the run manifest.
    6. **Barrier test.** `ASSERT_BARRIER` after one forward/backward on random
       data.

    `y_ref` is read as a He-normal *linear* map `W x` with
    `W ~ N(0, 2/d_model)`: at `||x|| = sqrt(d_model)` its per-coordinate
    variance is exactly `2`, which makes the band a statement about the Site's
    output scale against a standard well-initialised layer of the same width.
    That is the only reading under which the single matrix `W` the spec names is
    well defined.
    """
    cfg = model.cfg
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    failures: List[str] = []

    # ---- 3 + 4: two independent estimators at different sample sizes -------
    small = calibrate(
        cfg, genome.field_, model.parts.A.data, n_probe=4096,
        generator=generator, kappa=model.parts.kappa.data,
        W_ctx=model.sites[0].W_ctx.weight.data,
    )
    large = calibrate(
        cfg, genome.field_, model.parts.A.data, n_probe=16384,
        generator=generator, kappa=model.parts.kappa.data,
        W_ctx=model.sites[0].W_ctx.weight.data,
    )
    if abs(small.N_eff - large.N_eff) / max(large.N_eff, 1e-12) > 0.10:
        failures.append(
            f"N_eff cross-estimator disagreement: {small.N_eff:.4f} (4096) vs "
            f"{large.N_eff:.4f} (16384), > 10%"
        )
    if abs(small.E2_hat - large.E2_hat) / max(large.E2_hat, 1e-12) > 0.10:
        failures.append(
            f"E2 cross-estimator disagreement: {small.E2_hat:.5f} vs {large.E2_hat:.5f}"
        )
    if large.N_eff < 1.0:
        failures.append(f"guard: N_eff = {large.N_eff:.4f} < 1")
    if large.E2_hat < 1e-4:
        failures.append(f"guard: E2 = {large.E2_hat:.3e} < 1e-4")

    # ---- 1: variance test --------------------------------------------------
    n_probe = 4096
    x = _probe_inputs(n_probe, cfg.d_model, generator, device, dtype).unsqueeze(0)
    pop = model.populations[0]
    with torch.no_grad():
        site_out = model.sites[0](
            x, field=genome.field_, p_u=pop.p_u, g_u=pop.g_u,
            m_u=pop.m_u, active=pop.active,
        )
        var_y = float(site_out.y.var())
    var_ref = 2.0  # W ~ N(0, 2/d_model) at ||x|| = sqrt(d_model)
    ratio = var_y / var_ref
    if not (VAR_LO <= ratio <= VAR_HI):
        failures.append(
            f"variance test: Var(y)/Var(y_ref) = {ratio:.4f} outside "
            f"[{VAR_LO}, {VAR_HI}] (Var(y) = {var_y:.4f}; CALIBRATE targets "
            f"Var(y) = 1, so the design ratio is {VAR_TARGET})"
        )

    # ---- 2: gradient-flow test --------------------------------------------
    # §02.5 step 2, v3.1. Measured on the **residual-stream activation**
    # gradient at the first and last block, and expressed **per block**.
    #
    # v3.0 compared raw *parameter*-gradient norms of block 0 against block
    # L-1. Two things are wrong with that. The blocks are not the same size --
    # at M1 block 0 is a mixer (1.32 M parameters) and block 11 is attention
    # (0.65 M) -- so the comparison conflates parameter count with gradient
    # flow. And the band `[0.1, 10]` is stated on the *total* ratio across the
    # stack, which is not depth-invariant: an equally healthy model measured
    # 7.9 at `L = 6` and 14.1 at `L = 12`, passing at one depth and failing at
    # the other. Both are ~1.3x per block, which is exactly what a healthy
    # Peri-LN residual stack does. See AUDIT A-57.
    tokens = torch.randint(
        0, cfg.V, (2, min(cfg.T_ctx, 64)), generator=generator, device=device
    )
    model.zero_grad(set_to_none=True)
    out = model(tokens, genome=genome, want_metabolic=False, capture_residuals=True)
    out.logits.float().square().mean().backward()

    def residual_grad_norm(i: int) -> float:
        g = out.residuals[i].grad
        return float(g.detach().float().norm()) if g is not None else 0.0

    norms = [residual_grad_norm(i) for i in range(cfg.L)]
    g_first, g_last = norms[0], norms[-1]
    grad_ratio = g_first / max(g_last, 1e-30)
    steps = [norms[i] / max(norms[i + 1], 1e-30) for i in range(cfg.L - 1)]
    # The **median** per-block ratio, not the geometric mean. Block 0 is
    # legitimately larger than the rest -- two sublayers each add RMS 1 to a
    # stream at RMS 1, so the residual grows by sqrt(3) there and by less
    # thereafter as the stream's own norm rises -- and with few blocks that one
    # legitimate factor dominates a geometric mean (1.68 at L=6 against 1.34 at
    # L=12 for the *same* healthy initialisation). The median is robust to it
    # and is comparable across depths: 1.38 at M-nano, 1.21 at M1.
    per_block = float(sorted(steps)[len(steps) // 2]) if steps else 1.0
    per_block_max = float(max(steps)) if steps else 1.0
    if not (GRAD_FLOW_LO <= per_block <= GRAD_FLOW_HI):
        failures.append(
            f"gradient-flow test: median per-block residual-gradient ratio "
            f"{per_block:.4f} outside [{GRAD_FLOW_LO}, {GRAD_FLOW_HI}] "
            f"(total {grad_ratio:.4f} across {cfg.L} blocks; "
            f"per-block {[round(x, 2) for x in steps]})"
        )
    # The median is blind to a discontinuity in one block, which is exactly the
    # shape A-57 had (audit A-83).
    if per_block_max > GRAD_FLOW_MAX:
        failures.append(
            f"gradient-flow test: largest per-block residual-gradient ratio "
            f"{per_block_max:.4f} exceeds {GRAD_FLOW_MAX} at block "
            f"{steps.index(max(steps))} (median {per_block:.4f} is inside its "
            f"band, which is why the median alone cannot see this; "
            f"per-block {[round(x, 2) for x in steps]})"
        )

    # ---- 6: barrier test ---------------------------------------------------
    try:
        assert_barrier(registry, optimizer_param_ids=[id(p) for p in model.parameters()])
    except Exception as exc:  # BarrierViolation
        failures.append(f"barrier test: {exc}")
    model.zero_grad(set_to_none=True)

    report = AcceptanceReport(
        ratio, grad_ratio, per_block, per_block_max, tuple(steps),
        small.N_eff, large.N_eff, small.E2_hat,
        large.E2_hat, large.gain_act, large, not failures, failures,
    )
    if failures and raise_on_failure:
        raise AcceptanceFailed(
            "§02.5 acceptance test failed; the run aborts:\n  "
            + "\n  ".join(failures)
        )
    return report


@torch.no_grad()
def apply_calibration(
    model, result: CalibrationResult, generator: Optional[torch.Generator] = None
) -> None:
    """Rescale `B` to the calibrated `Var(b_g)` (§01.5).

    **[NORMATIVE — a rescale, not a resample; v3.1, audit A-85]** v3.0 called
    `B.normal_(0, std)` with **no generator**, which did two wrong things at
    once. It **replaced** `B` rather than rescaling it, contradicting its own
    one-line docstring; and `Tensor.normal_` *does* accept `generator=`, so this
    was not one of §02.6.3's sanctioned global-RNG exceptions but the A-64
    defect in a place nobody had looked. `B` is the largest tensor the Site
    multiplies by — here rescaled by a factor of ~161 — and its values depended
    on how many global draws module construction happened to consume between
    `rngs.seed_global("init")` and this call, so any refactor upstream silently
    changed `B` while every explicit stream stayed identical.

    Rescaling preserves the *direction* of every gene's output vector, which is
    what "rescale" means and what makes the calibration a change of scale rather
    than a second initialisation.
    """
    target = math.sqrt(result.var_b)
    cur = float(model.parts.B.data.std())
    if cur > 0:
        model.parts.B.data.mul_(target / cur)
    else:
        # A degenerate `B` cannot be rescaled into anything; draw once, from an
        # explicit stream.
        model.parts.B.data.normal_(0.0, target, generator=generator)
