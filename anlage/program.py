"""The programme genome `Gamma = (mu, omega, eta, s, Lambda)`.

SPEC: 12 §12.1
SPEC: 12 §12.4
SPEC: 12 §12.5

What is heritable is what does not copy (A8, §11.5): the data programme `mu`
(sampling weights over data classes per stage), the developmental schedule
`omega`, the recipe `eta` in scale-free units, the scale profile `s` (the
vector genome of §03 restricted to vector roles) and the unit prior `Lambda`.
This module holds the object, its variation operator (the Dirichlet
neighbourhood, 12.4), the outcome surrogate (12.1) and the trust-region step
(12.2) -- the generation-level consolidation of Definition 1 applied to the
programme instead of to weights.

Everything here is CPU float64 and deterministic given its seeds (M11); the
solver has no data-dependent branching that could differ across machines.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np
import torch

__all__ = ["Program", "Surrogate", "MU_MIN", "ALPHA_0", "TAU", "LAMBDA_F", "clip_simplex",
           "dirichlet_neighbourhood", "quadratic_features", "fit_surrogate",
           "project_simplex_floor", "solve_trust_region"]

MU_MIN = 0.02
"""SPEC: 12 §12.4 -- no class weight below this (no class is ever unseen)."""
ALPHA_0 = 50.0
"""SPEC: 12 §12.4 -- initial Dirichlet concentration; set by G-LIVE-mu."""
TAU = 0.2
"""SPEC: 12 §12.5 -- trust-region radius (l2) around the current programme."""
LAMBDA_F = 1e-3
"""SPEC: 12 §12.5 -- ridge penalty of the surrogate fit."""


# ----------------------------------------------------------------------------- object
@dataclasses.dataclass
class Program:
    """`Gamma`. `mu (S+1, n_C)` rows on the simplex; `omega` FLOP fractions summing to 1;
    `eta` scale-free recipe multipliers; `s_state` the vector genome's tensors (or None);
    `Lambda` per-axis moments; `classes` the class manifest (names, edges)."""

    mu: torch.Tensor
    omega: tuple[float, ...]
    eta: dict[str, float]
    s_state: Optional[dict]
    Lambda: dict[str, dict[str, list[float]]]
    classes: dict

    def __post_init__(self) -> None:
        self.mu = torch.as_tensor(self.mu, dtype=torch.float64)
        if self.mu.dim() != 2:
            raise ValueError("mu must be (S+1, n_C)")
        self.validate()

    def validate(self) -> None:
        if not torch.isfinite(self.mu).all():
            raise ValueError("mu has a non-finite entry")
        if (self.mu < MU_MIN - 1e-12).any():
            raise ValueError(f"a class weight is below MU_MIN = {MU_MIN}")
        if (self.mu.sum(1) - 1.0).abs().max() > 1e-9:
            raise ValueError("every row of mu must sum to 1")
        if abs(sum(self.omega) - 1.0) > 1e-9 or any(f <= 0 for f in self.omega):
            raise ValueError("omega must be positive fractions summing to 1")
        if len(self.omega) != self.mu.shape[0]:
            raise ValueError("one mu row per stage of omega")

    @property
    def n_classes(self) -> int:
        return int(self.mu.shape[1])

    def n_reals(self) -> int:
        """The size of `Gamma` in reals (M8: scale-free, identical at every stage)."""
        n = self.mu.numel() + len(self.omega) + len(self.eta)
        for ax in self.Lambda.values():
            for v in ax.values():
                n += len(v) if isinstance(v, (list, tuple)) else 1
        if self.s_state is not None:
            n += sum(int(t.numel()) for t in self.s_state.values())
        return int(n)

    def save(self, path) -> None:
        torch.save({"mu": self.mu.clone(), "omega": tuple(float(f) for f in self.omega),
                    "eta": dict(self.eta), "s_state": self.s_state, "Lambda": self.Lambda,
                    "classes": self.classes}, path)

    @staticmethod
    def load(path) -> "Program":
        b = torch.load(path, map_location="cpu", weights_only=False)
        return Program(mu=b["mu"], omega=tuple(b["omega"]), eta=dict(b["eta"]),
                       s_state=b["s_state"], Lambda=b["Lambda"], classes=b["classes"])

    def equals(self, other: "Program") -> bool:
        if not torch.equal(self.mu, other.mu) or self.omega != other.omega or self.eta != other.eta:
            return False
        if self.Lambda != other.Lambda or self.classes != other.classes:
            return False
        if (self.s_state is None) != (other.s_state is None):
            return False
        if self.s_state is not None:
            if self.s_state.keys() != other.s_state.keys():
                return False
            return all(torch.equal(self.s_state[k], other.s_state[k]) for k in self.s_state)
        return True


# --------------------------------------------------------------------------- simplex
def clip_simplex(mu: torch.Tensor, mu_min: float = MU_MIN) -> torch.Tensor:
    """Raise every entry below `mu_min` to it and renormalise the rest so the result is
    on the simplex to `1e-12` (M5). Iterates because renormalising can push another
    entry below the floor; terminates in at most `n_C` rounds."""
    x = torch.as_tensor(mu, dtype=torch.float64).clone()
    n = x.numel()
    if n * mu_min > 1.0 + 1e-12:
        raise ValueError(f"{n} classes cannot all have weight >= {mu_min}")
    x = x / x.sum()
    fixed = torch.zeros(n, dtype=torch.bool)
    for _ in range(n):
        low = (x < mu_min) & ~fixed
        if not low.any():
            break
        fixed |= low
        x[fixed] = mu_min
        free = ~fixed
        mass = 1.0 - float(fixed.sum()) * mu_min
        if free.any():
            x[free] = x[free] / x[free].sum() * mass
    return x


def project_simplex_floor(v: torch.Tensor, mu_min: float = MU_MIN) -> torch.Tensor:
    """Euclidean projection onto `{mu : sum mu = 1, mu >= mu_min}`: shift by the floor and
    project onto the simplex of mass `1 - n mu_min` (Duchi, Shalev-Shwartz, Singer & Chandra,
    ICML 2008, Fig. 1), then shift back."""
    v = torch.as_tensor(v, dtype=torch.float64)
    n = v.numel()
    z = 1.0 - n * mu_min
    if z < 0:
        raise ValueError("floor infeasible")
    u = v - mu_min
    s, _ = torch.sort(u, descending=True)
    css = torch.cumsum(s, 0)
    k = torch.arange(1, n + 1, dtype=torch.float64)
    cond = s - (css - z) / k > 0
    rho = int(torch.nonzero(cond).max()) + 1
    theta = (css[rho - 1] - z) / rho
    w = torch.clamp(u - theta, min=0.0)
    return w + mu_min


def dirichlet_neighbourhood(mu_0: torch.Tensor, alpha: float, K: int, seed: int, *,
                            mu_min: float = MU_MIN, include_centre: bool = True) -> torch.Tensor:
    """`K` mixtures around `mu_0`: `mu_k ~ Dirichlet(alpha mu_0)`, clipped to the floor.
    SPEC: 12 §12.4. With `include_centre` the first row is `mu_0` itself (S-CTRL)."""
    mu_0 = torch.as_tensor(mu_0, dtype=torch.float64)
    if K < 1 or alpha <= 0:
        raise ValueError("K >= 1 and alpha > 0")
    rng = np.random.default_rng(int(seed))
    n_draw = K - 1 if include_centre else K
    draws = rng.dirichlet(alpha * mu_0.numpy(), size=n_draw) if n_draw > 0 else np.zeros((0, mu_0.numel()))
    rows = [clip_simplex(mu_0, mu_min)] if include_centre else []
    rows += [clip_simplex(torch.from_numpy(d), mu_min) for d in draws]
    return torch.stack(rows, 0)


# -------------------------------------------------------------------------- surrogate
def quadratic_features(mu: torch.Tensor, *, linear_only: bool = False) -> torch.Tensor:
    """`[1, mu, upper triangle of mu mu^T]` per row. (12.1)"""
    mu = torch.as_tensor(mu, dtype=torch.float64)
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    n, d = mu.shape
    cols = [torch.ones(n, 1, dtype=torch.float64), mu]
    if not linear_only:
        iu = torch.triu_indices(d, d)
        cols.append(mu[:, iu[0]] * mu[:, iu[1]])
    return torch.cat(cols, 1)


@dataclasses.dataclass
class Surrogate:
    """(12.1) per stratum: `F_hat_q(mu) = phi(mu)^T beta_q`."""

    beta: torch.Tensor            # (n_features, Q)
    n_classes: int
    linear_only: bool
    lambda_F: float
    n_samples: int
    loo_rmse: float               # held-one-out RMSE over the fitting samples

    def predict(self, mu: torch.Tensor) -> torch.Tensor:
        """`(Q,)` for one mixture, `(n, Q)` for a batch."""
        phi = quadratic_features(mu, linear_only=self.linear_only)
        out = phi @ self.beta
        return out[0] if torch.as_tensor(mu).dim() == 1 else out

    def objective(self, mu: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """`sum_q w_q F_hat_q(mu)`, differentiable in `mu`."""
        return (self.predict(mu) * w).sum()


def fit_surrogate(mus: torch.Tensor, F: torch.Tensor, *, lambda_F: float = LAMBDA_F,
                  weights: Optional[torch.Tensor] = None, force_linear: bool = False,
                  w_q: Optional[torch.Tensor] = None, select: str = "loo") -> Surrogate:
    """Ridge least squares of (12.1), one regression per stratum, in float64.

    Identifiability (M6): the quadratic has `1 + n_C + n_C (n_C + 1) / 2` coefficients;
    with fewer than `4 n_C` samples the fit is linear. With `select = "loo"` (§12.5) the
    quadratic is used only if its held-one-out error on the `w_q`-weighted objective is
    below the linear fit's -- the rule R3 forced (a quadratic on 25 samples predicted
    worse than a line). `weights` (per sample) implement the `gamma^age` re-weighting.
    The intercept is not penalised.
    """
    mus = torch.as_tensor(mus, dtype=torch.float64)
    F = torch.as_tensor(F, dtype=torch.float64)
    if mus.dim() != 2 or F.dim() != 2 or mus.shape[0] != F.shape[0]:
        raise ValueError("mus (n, n_C) and F (n, Q) with matching n")
    n, n_C = mus.shape
    if not force_linear and n >= 4 * n_C and select == "loo":
        lin = _fit(mus, F, lambda_F, weights, True, w_q)
        quad = _fit(mus, F, lambda_F, weights, False, w_q)
        return quad if quad.loo_rmse < lin.loo_rmse else lin
    return _fit(mus, F, lambda_F, weights, force_linear or n < 4 * n_C, w_q)


def _fit(mus: torch.Tensor, F: torch.Tensor, lambda_F: float, weights: Optional[torch.Tensor],
         linear_only: bool, w_q: Optional[torch.Tensor]) -> Surrogate:
    n, n_C = mus.shape
    phi = quadratic_features(mus, linear_only=linear_only)
    p = phi.shape[1]
    W = torch.ones(n, dtype=torch.float64) if weights is None else torch.as_tensor(weights, dtype=torch.float64)
    reg = torch.eye(p, dtype=torch.float64) * lambda_F
    reg[0, 0] = 0.0
    A = phi.T @ (phi * W[:, None]) + reg
    B = phi.T @ (F * W[:, None])
    beta = torch.linalg.solve(A, B)
    # held-one-out error by the ridge leave-one-out identity: r_i / (1 - h_ii), on the
    # w_q-weighted objective (the quantity (12.2) minimises) when w_q is given
    H = phi @ torch.linalg.solve(A, (phi * W[:, None]).T)
    h = torch.diagonal(H)
    resid = F - phi @ beta
    loo = resid / (1.0 - h).clamp_min(1e-12)[:, None]
    if w_q is not None:
        wq = torch.as_tensor(w_q, dtype=torch.float64)
        loo_obj = (loo * wq[None, :]).sum(1)
        loo_rmse = float(torch.sqrt((loo_obj.pow(2) * W).sum() / W.sum()))
    else:
        loo_rmse = float(torch.sqrt((loo.pow(2) * W[:, None]).sum() / (W.sum() * F.shape[1])))
    return Surrogate(beta=beta, n_classes=n_C, linear_only=linear_only, lambda_F=lambda_F,
                     n_samples=n, loo_rmse=loo_rmse)


# ----------------------------------------------------------------------- trust region
def _project_ball(x: torch.Tensor, centre: torch.Tensor, tau: float) -> torch.Tensor:
    d = x - centre
    nrm = float(d.norm())
    if nrm <= tau:
        return x
    return centre + d * (tau / nrm)


def project_feasible(x: torch.Tensor, mu_0: torch.Tensor, tau: float, mu_min: float,
                     *, iters: int = 200) -> torch.Tensor:
    """Projection onto `{simplex with floor} ∩ {||mu - mu_0||_2 <= tau}` by Dykstra's
    alternating projections (Boyle & Dykstra 1986), which converges to the exact
    projection onto an intersection of closed convex sets."""
    p = torch.zeros_like(x)
    q = torch.zeros_like(x)
    y = x.clone()
    for _ in range(iters):
        a = project_simplex_floor(y + p, mu_min)
        p = y + p - a
        y_new = _project_ball(a + q, mu_0, tau)
        q = a + q - y_new
        if float((y_new - y).abs().max()) < 1e-14:
            y = y_new
            break
        y = y_new
    return y


def solve_trust_region(sur: Surrogate, w: torch.Tensor, mu_0: torch.Tensor, *, tau: float = TAU,
                       mu_min: float = MU_MIN, iters: int = 5000, lr: float = 0.05
                       ) -> tuple[torch.Tensor, dict]:
    """(12.2): `argmin_mu sum_q w_q F_hat_q(mu)` over the simplex with floor, within the
    `l2` trust region of radius `tau` around `mu_0`. Projected gradient with a fixed
    iteration count (deterministic, M11); the result is verified to be feasible and the
    projected-gradient residual is reported (M7)."""
    mu_0 = torch.as_tensor(mu_0, dtype=torch.float64)
    w = torch.as_tensor(w, dtype=torch.float64)
    if abs(float(mu_0.sum()) - 1.0) > 1e-9:
        raise ValueError("mu_0 must be on the simplex")
    x = project_feasible(mu_0.clone(), mu_0, tau, mu_min)
    best, best_val = x.clone(), float("inf")
    for _ in range(iters):
        xg = x.clone().requires_grad_(True)
        val = sur.objective(xg, w)
        (g,) = torch.autograd.grad(val, xg)
        fv = float(val)
        if fv < best_val:
            best, best_val = x.clone(), fv
        x_new = project_feasible(x - lr * g, mu_0, tau, mu_min)
        if float((x_new - x).abs().max()) < 1e-12:
            x = x_new
            break
        x = x_new
    xg = best.clone().requires_grad_(True)
    val = sur.objective(xg, w)
    (g,) = torch.autograd.grad(val, xg)
    residual = float((best - project_feasible(best - lr * g, mu_0, tau, mu_min)).norm()) / lr
    info = {"value": float(val), "pg_residual": residual, "distance": float((best - mu_0).norm()),
            "on_boundary": abs(float((best - mu_0).norm()) - tau) < 1e-6,
            "simplex_error": abs(float(best.sum()) - 1.0), "min_weight": float(best.min())}
    return best, info
