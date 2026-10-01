"""The readout: the lifetime's convex object. SPEC: 01 §1.4 (Definitions 4-6).

Ridge regression to one-hot targets accumulated in one streaming pass (the Gram and the
cross-covariance), a Cholesky solve, a two-scalar calibration by bracketed search on the
fitness set, and the delta-rule refinement of the multinomial logistic objective with the
closed-form gradient (1.7) -- no autograd anywhere.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable

import torch

from gnomon.utils.shapes import check_shape

__all__ = ["GramAccumulator", "golden_section", "calibrate", "calibrate_linear", "linear_log_probs",
           "AdamState", "delta_rule_step", "cross_entropy_rows", "SOFTCAP"]

SOFTCAP = 30.0
"""01 §1.10: logits are softcapped at 30 to bound the calibration search."""


def softcap(x: torch.Tensor, c: float = SOFTCAP) -> torch.Tensor:
    return c * torch.tanh(x / c)


class GramAccumulator:
    """`G = sum z z^T` and `C = sum z 1_v^T` in float32 on the device; grows when units are born."""

    def __init__(self, F: int, V: int, device: str = "cuda"):
        if F < 1 or V < 1:
            raise ValueError("F, V >= 1")
        self.F, self.V = int(F), int(V)
        self.G = torch.zeros(F, F, dtype=torch.float32, device=device)
        self.C = torch.zeros(F, V, dtype=torch.float32, device=device)
        self.n = 0

    def add(self, Z: torch.Tensor, targets: torch.Tensor) -> None:
        """`G += Z^T Z` (a deterministic GEMM) and `C += Z^T Y` with `Y` the one-hot targets as a
        CSR sparse matrix (cuSPARSE SpMM, deterministic and 3 ms). `index_add_` on CUDA uses
        atomics and changed `C` at 1e-6 between two identical calls, which moved a life's CE by
        1.6e-4 (G-DET, 03 §P-S.26)."""
        check_shape(Z, ["n", self.F], name="Z", bindings={})
        check_shape(targets, ["n"], name="targets", bindings={})
        Zf = Z.float()
        n = int(Z.shape[0])
        self.G.addmm_(Zf.T, Zf)
        idx = torch.stack([targets, torch.arange(n, device=targets.device)])
        Yt = torch.sparse_coo_tensor(idx, torch.ones(n, device=Z.device, dtype=torch.float32), (self.V, n)).coalesce()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                       # the CSR "beta" notice
            self.C += torch.sparse.mm(Yt.to_sparse_csr(), Zf).T
        self.n += n

    def grow(self, F_new: int) -> None:
        """A newborn unit's activation is zero on every token seen before its birth (01 §1.6 step 4)."""
        if F_new < self.F:
            raise ValueError("cannot shrink")
        if F_new == self.F:
            return
        G = torch.zeros(F_new, F_new, dtype=torch.float32, device=self.G.device)
        G[: self.F, : self.F] = self.G
        self.G = G
        del G
        C = torch.zeros(F_new, self.V, dtype=torch.float32, device=self.C.device)
        C[: self.F] = self.C
        self.C, self.F = C, int(F_new)

    def solve_targets(self, lam: float, Y: torch.Tensor, *, float64_below: int = 12288) -> torch.Tensor:
        """`(G + lam Gbar I)^{-1} (C Y)` for a `(V, k)` target map `Y`: the ridge predictors of the
        targets `Y^T 1_v` from the accumulated features (01 Definition 11b). Returns `(F, k)`."""
        if lam <= 0:
            raise ValueError("lam > 0")
        Gbar = float(self.G.diagonal().mean()) if self.n > 0 else 1.0
        if not math.isfinite(Gbar) or Gbar <= 0:
            raise FloatingPointError(f"degenerate Gram: mean diagonal {Gbar}")
        dtype = torch.float64 if self.F <= float64_below else torch.float32
        M = self.G.to(dtype) / Gbar
        M.diagonal().add_(lam)
        L = torch.linalg.cholesky(M)
        rhs = (self.C @ Y.to(self.C.dtype)).to(dtype) / Gbar
        W = torch.cholesky_solve(rhs, L)
        del M, L
        return W.float()

    def solve(self, lam: float, *, float64_below: int = 12288) -> torch.Tensor:
        """(1.6): `W = (G + lam Gbar I)^{-1} C`, Cholesky in float64 when `F <= float64_below`,
        else float32 (01 §1.10). Raises on a non-positive pivot; never silently regularised."""
        if lam <= 0:
            raise ValueError("lam > 0")
        Gbar = float(self.G.diagonal().mean()) if self.n > 0 else 1.0
        if not math.isfinite(Gbar) or Gbar <= 0:
            raise FloatingPointError(f"degenerate Gram: mean diagonal {Gbar}")
        dtype = torch.float64 if self.F <= float64_below else torch.float32
        M = self.G.to(dtype) / Gbar
        M.diagonal().add_(lam)
        if not torch.isfinite(M).all():
            raise FloatingPointError("non-finite Gram before the solve")
        L = torch.linalg.cholesky(M)
        W = torch.cholesky_solve(self.C.to(dtype) / Gbar, L)
        del M, L
        if not torch.isfinite(W).all():
            raise FloatingPointError("non-finite readout after the solve (03 §P-S.29)")
        return W.float()


def cross_entropy_rows(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-row CE in float32 with the max subtracted."""
    lg = logits.float()
    lse = torch.logsumexp(lg, dim=-1)
    return lse - lg.gather(1, targets.unsqueeze(1)).squeeze(1)


def golden_section(f: Callable[[float], float], lo: float, hi: float, *, tol: float = 1e-3,
                   max_iter: int = 40) -> tuple[float, float]:
    """Minimise `f` on `[lo, hi]` by golden-section search; returns `(x*, f(x*))`.
    Terminates after at most `max_iter` evaluations or when the bracket is below `tol`."""
    if hi <= lo:
        raise ValueError("hi > lo")
    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    n = 2
    while (b - a) > tol and n < max_iter:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
        n += 1
    x = c if fc < fd else d
    return float(x), float(min(fc, fd))


def calibrate(S: torch.Tensor, M: torch.Tensor | None, targets: torch.Tensor, *,
              tau_s_range=(0.5, 400.0), tau_m_range=(0.0, 60.0), rounds: int = 2,
              chunk: int = 8192) -> tuple[float, float, float]:
    """Definition 5: `(tau_s, tau_m)` minimising the mean CE of `softmax(tau_s S + tau_m M)` on
    the calibration rows by coordinate-wise golden-section search (`tau_s` in log space).
    `S`, `M` are `(n, V)` (M may be None -> tau_m = 0). Returns `(tau_s, tau_m, ce)`."""
    n = S.shape[0]

    def ce(ts: float, tm: float) -> float:
        tot = 0.0
        for i in range(0, n, chunk):
            lg = ts * S[i:i + chunk].float()
            if M is not None and tm != 0.0:
                lg = lg + tm * M[i:i + chunk].float()
            tot += float(cross_entropy_rows(softcap(lg), targets[i:i + chunk]).sum())
        return tot / n

    ts, tm = 1.0, 0.0
    best = ce(ts, tm)
    for _ in range(rounds):
        lts, best = golden_section(lambda x: ce(math.exp(x), tm), math.log(tau_s_range[0]),
                                   math.log(tau_s_range[1]))
        ts = math.exp(lts)
        if M is not None:
            tm, best = golden_section(lambda x: ce(ts, x), tau_m_range[0], tau_m_range[1])
    return float(ts), float(tm), float(best)


class AdamState:
    """Adam moments for the delta-rule refinement (Definition 6); no autograd."""

    def __init__(self, W: torch.Tensor, *, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.m = torch.zeros_like(W)
        self.v = torch.zeros_like(W)
        self.lr, self.b1, self.b2, self.eps = float(lr), beta1, beta2, eps
        self.t = 0

    def step(self, W: torch.Tensor, grad: torch.Tensor, lr_scale: float = 1.0) -> None:
        self.t += 1
        self.m.mul_(self.b1).add_(grad, alpha=1 - self.b1)
        self.v.mul_(self.b2).addcmul_(grad, grad, value=1 - self.b2)
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        denom = (self.v / bc2).sqrt_().add_(self.eps)
        W.addcdiv_(self.m, denom, value=-(self.lr * lr_scale) / bc1)


def delta_rule_step(W: torch.Tensor, Z: torch.Tensor, targets: torch.Tensor, offset: torch.Tensor | None,
                    adam: AdamState, lr_scale: float, *, U: torch.Tensor | None = None,
                    E_out: torch.Tensor | None = None, adam_U: "AdamState | None" = None) -> float:
    """(1.7): one mini-batch of the multinomial logistic readout, closed-form gradient
    `Z^T (p - 1_v) / n`, applied by Adam. Returns the batch's mean CE (before the step).

    With `U (F, e)` and `E_out (V, e)` (Definition 6b, output-side semantic sharing) the logits
    are `Z W + (Z U) E_out^T`; the gradient in `U` is `Z^T ((p - 1_v) E_out) / n`. The logits stay
    linear in `(W, U)`, so the objective stays convex and the rule stays single-layer."""
    n = Z.shape[0]
    lg = Z @ W
    if U is not None:
        lg = lg + (Z @ U) @ E_out.T
    if offset is not None:
        lg = lg + offset
    lg = softcap(lg)
    ce = float(cross_entropy_rows(lg, targets).mean())
    p = torch.softmax(lg, dim=-1)
    p[torch.arange(n, device=p.device), targets] -= 1.0
    # the softcap's derivative: d/dx c tanh(x/c) = 1 - tanh^2(x/c)
    p = p * (1.0 - (lg / SOFTCAP) ** 2)
    if U is not None:
        grad_U = Z.T @ (p @ E_out) / n
        adam_U.step(U, grad_U, lr_scale)
    grad = Z.T @ p / n
    adam.step(W, grad, lr_scale)
    return ce


def linear_log_probs(S: torch.Tensor, M: torch.Tensor | None, log_uni: torch.Tensor, mu: float, nu: float,
                     tau_m: float, eps: float = 1e-6) -> torch.Tensor:
    """The linear head (Definition 5b): the least-squares scores used *as* probabilities.

    `p = (1 - mu - nu) * clip(S, 0) / sum + mu * p_uni + nu * softmax(tau_m M)`; returns `log p`,
    `(n, V)`. The ridge scores are least-squares estimates of `P(v | z)` and sum to one with
    a bias, so clipping and renormalising is the projection onto the simplex in L1; the
    unigram and the memory's cleanup enter as mixture components with weights fit on the
    fitness set. The softmax of a probability estimate flattens the tail (a token at 0.01
    and one at 0.001 get nearly the same logit); this head does not."""
    S = S.float()
    p = S.clamp_min(0.0) + eps
    p = p / p.sum(dim=-1, keepdim=True)
    out = (1.0 - mu - nu) * p + mu * log_uni.exp().unsqueeze(0)
    if M is not None and nu > 0.0:
        out = out + nu * torch.softmax(tau_m * M.float(), dim=-1)
    return out.clamp_min(1e-12).log()


def calibrate_linear(S: torch.Tensor, M: torch.Tensor | None, targets: torch.Tensor, log_uni: torch.Tensor, *,
                     rounds: int = 2, chunk: int = 8192) -> tuple[float, float, float, float]:
    """`(mu, nu, tau_m, ce)` for the linear head by coordinate-wise golden-section search."""
    n = S.shape[0]

    def ce(mu: float, nu: float, tm: float) -> float:
        tot = 0.0
        for i in range(0, n, chunk):
            lp = linear_log_probs(S[i:i + chunk], None if M is None else M[i:i + chunk], log_uni, mu, nu, tm)
            tot += float(-lp.gather(1, targets[i:i + chunk].unsqueeze(1)).sum())
        return tot / n

    mu, nu, tm = 0.1, 0.0, 10.0
    best = ce(mu, nu, tm)
    for _ in range(rounds):
        mu, best = golden_section(lambda x: ce(x, nu, tm), 0.0, 0.98 - nu)
        if M is not None:
            nu, best = golden_section(lambda x: ce(mu, x, tm), 0.0, 0.98 - mu)
            tm, best = golden_section(lambda x: ce(mu, nu, x), 0.0, 60.0)
    return float(mu), float(nu), float(tm), float(best)
