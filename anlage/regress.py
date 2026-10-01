"""The generation-level consolidation: `REGRESS` a phenotype into `(Gamma, Z)`.

SPEC: 05 §5.1
SPEC: 05 §5.2
SPEC: 05 §5.4
SPEC: 05 §5.5
SPEC: 05 §5.6

```
    J(Gamma, Z) = sum_{(l,m)} omega_{l,m} ||D_{l,m} - W_{l,m}||_F^2 / ||W_{l,m}||_F^2
                  + lambda_z sum_a ||z_a||^2 / d_z + lambda_y sum_l ||y_l||^2 / d_y
                  + lambda_g ||Gamma||^2                                              (5.3)
```

Adam on `(Gamma, Z)`, every step one full decode of the stage; per-role
relative residuals recorded every 100 steps; early stop on a plateau. The
regression is the third instance of Definition 1 and the only place the
genome's tensors receive a gradient.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable, Optional

import torch

from gnomon.utils.stats import effective_rank
from .config import MATRIX_ROLES, ROLE_NAMES, VECTOR_ROLES
from .genome import DecodeContext, Genome
from .latents import LatentBank

__all__ = ["RegressConfig", "RegressReport", "objective", "regress", "expression",
           "shuffled", "decoded_erank_report"]


@dataclasses.dataclass(frozen=True)
class RegressConfig:
    """SPEC: 05 §5.2."""

    n_fit: int = 3000
    lr: float = 2e-3
    betas: tuple[float, float] = (0.9, 0.99)
    lr_floor: float = 0.1
    lambda_z: float = 1e-3
    lambda_y: float = 1e-3
    lambda_g: float = 1e-6
    plateau_window: int = 300
    plateau_tol: float = 1e-4
    report_every: int = 100


@dataclasses.dataclass
class RegressReport:
    residual: dict[str, float]            # per role, final relative residual
    residual_mean: float
    history: list[tuple[int, float]]      # (step, mean residual)
    steps: int
    seconds: float
    n_genome: int
    n_latents: int
    n_phenotype: int
    ratio_genome: float
    ratio_genome_latents: float
    stopped_early: bool

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _rel(D: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return (D - W).pow(2).sum() / W.pow(2).sum().clamp_min(1e-12)


def objective(G: Genome, ctx: DecodeContext, W: dict, rc: RegressConfig, *,
              omega: Optional[dict] = None
              ) -> tuple[torch.Tensor, dict[tuple[int, int], torch.Tensor]]:
    """(5.3). Returns the objective and the per-(role, layer) relative residuals."""
    D = G.decode_all(ctx, keys=list(W.keys()))
    res = {}
    total = torch.zeros((), dtype=torch.float32, device=next(G.parameters()).device)
    for k, w in W.items():
        w32 = w.detach().float()
        r = _rel(D[k].float(), w32)
        res[k] = r
        total = total + (omega[k] if omega else 1.0) * r
    bank = ctx.bank
    d_z = max(ctx.bank.shape.d_z, 1)
    zterm = sum(p.pow(2).sum() for n, p in bank.named_parameters() if n != "y") / d_z
    yterm = bank.y.pow(2).sum() / max(bank.shape.d_y, 1)
    gterm = sum(p.pow(2).sum() for p in G.parameters())
    total = total + rc.lambda_z * zterm + rc.lambda_y * yterm + rc.lambda_g * gterm
    return total, res


def regress(G: Genome, ctx: DecodeContext, W: dict, rc: RegressConfig = RegressConfig(),
            *, log: Optional[Callable[[str], None]] = None, omega: Optional[dict] = None
            ) -> RegressReport:
    """`REGRESS` of §05.2, in place on `G` and `ctx.bank`."""
    t0 = time.time()
    G.requires_grad_(True)
    ctx.bank.requires_grad_(True)
    params = list(G.parameters()) + list(ctx.bank.parameters())
    opt = torch.optim.Adam(params, lr=rc.lr, betas=rc.betas)
    history: list[tuple[int, float]] = []
    last_res: dict = {}
    stopped = False
    for n in range(1, rc.n_fit + 1):
        frac = (n - 1) / max(rc.n_fit - 1, 1)
        lr = rc.lr * (rc.lr_floor + (1 - rc.lr_floor) * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        J, res = objective(G, ctx, W, rc, omega=omega)
        J.backward()
        opt.step()
        if n % rc.report_every == 0 or n == rc.n_fit:
            mean = float(torch.stack(list(res.values())).mean())
            history.append((n, mean))
            last_res = {k: float(v) for k, v in res.items()}
            if log:
                log(f"    regress step {n:5d}  J {float(J):.4f}  mean residual {mean:.4f}  lr {lr:.2e}")
            k = rc.plateau_window // rc.report_every
            if len(history) > k and history[-1 - k][1] - mean < rc.plateau_tol:
                stopped = True
                break
    G.requires_grad_(False)
    ctx.bank.requires_grad_(False)
    per_role: dict[str, list[float]] = {}
    for (role, l), v in last_res.items():
        per_role.setdefault(ROLE_NAMES[role], []).append(v)
    residual = {r: sum(v) / len(v) for r, v in per_role.items()}
    n_pheno = sum(w.numel() for w in W.values())
    n_g, n_z = G.n_params(), ctx.bank.n_params()
    return RegressReport(
        residual=residual, residual_mean=history[-1][1] if history else float("nan"),
        history=history, steps=history[-1][0] if history else 0, seconds=time.time() - t0,
        n_genome=n_g, n_latents=n_z, n_phenotype=n_pheno,
        ratio_genome=n_pheno / n_g, ratio_genome_latents=n_pheno / (n_g + n_z),
        stopped_early=stopped)


@torch.no_grad()
def expression(W_prev: Optional[dict], D: dict, eps: float) -> dict:
    """(5.4): `W_0 = eps W_prev + (1 - eps) D`; units absent from `W_prev` take `D`."""
    out = {}
    for k, d in D.items():
        if W_prev is None or k not in W_prev or eps == 0.0:
            out[k] = d.detach().clone()
            continue
        w = W_prev[k].detach().to(d.device, d.dtype)
        if w.shape == d.shape:
            out[k] = eps * w + (1 - eps) * d
        else:
            # a grown shape: blend on the overlap, decode elsewhere
            o = d.detach().clone()
            sl = tuple(slice(0, s) for s in w.shape)
            o[sl] = eps * w + (1 - eps) * d[sl]
            out[k] = o
    return out


@torch.no_grad()
def shuffled(D: dict, generator: torch.Generator) -> dict:
    """The PNAS shuffle control: rows of every matrix role permuted. SPEC: 05 §5.6."""
    out = {}
    for (role, l), d in D.items():
        if role in MATRIX_ROLES and d.dim() == 2:
            perm = torch.randperm(d.shape[0], generator=generator).to(d.device)
            out[(role, l)] = d[perm]
        else:
            out[(role, l)] = d
    return out


@torch.no_grad()
def decoded_erank_report(D: dict, W: dict, keys: list[tuple[int, int]]) -> dict[str, dict]:
    """D5: decoded vs trained effective rank per selected role. SPEC: 03 §3.9."""
    out = {}
    for k in keys:
        d, w = D[k].float(), W[k].detach().float()
        if d.dim() != 2:
            continue
        ed = effective_rank(torch.linalg.svdvals(d).tolist())
        ew = effective_rank(torch.linalg.svdvals(w).tolist())
        out[f"{ROLE_NAMES[k[0]]}_{k[1]}"] = {"decoded": ed, "trained": ew,
                                              "min_dim": min(d.shape)}
    return out


# =============================================================================
# The scale profile `s`: the vector genome. SPEC: 12 §12.6
# =============================================================================

def vector_keys(W: dict) -> list[tuple[int, int]]:
    """The keys of `W` that are vector roles (everything the kernel decodes by (12.3))."""
    from .genome import MAT_INDEX
    return [k for k in W if k[0] not in MAT_INDEX]


def fit_scale_profile(model, shape, tables, *, rc: RegressConfig = RegressConfig(n_fit=500, lr=2e-3),
                      seed: int = 1, log=None):
    """Fit `(Gamma, Z)` to the *vector* roles of a trained phenotype only (12.3), the part of
    §03 that R4 measured to regress to `< 1%`. Returns `(G, bank, report)`; the matrix roles
    are never a target."""
    from .genome import DecodeContext, Genome
    from .latents import LatentBank
    device = next(model.parameters()).device
    G = Genome(shape, seed=seed).to(device)
    bank = LatentBank(shape, model.cfg, model.vocab, seed=seed).to(device)
    ctx = DecodeContext(model.cfg, tables, bank, model.vocab, tied=model.tied)
    W = {k: v.detach() for k, v in model.weights().items()}
    W_vec = {k: W[k] for k in vector_keys(W)}
    rep = regress(G, ctx, W_vec, rc, log=log)
    return G, bank, rep


@torch.no_grad()
def express_scale_profile(G, ctx, W_free: dict) -> dict:
    """The `s`-birth of §12.6: every vector role of `W_free` replaced by its decode (12.3),
    every matrix role left at the innate initialisation it came with (M9)."""
    keys = vector_keys(W_free)
    D = G.decode_all(ctx, keys=keys)
    out = dict(W_free)
    for k in keys:
        out[k] = D[k].to(W_free[k].dtype)
    return out
