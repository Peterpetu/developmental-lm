# VENDORED from `soma/optim/muon.py` at commit `7d29cc1` (SOMA v3.1).
#
# SPEC: 04 §4.1 -- the numerical cores are already pure functions of
# activations and weights, so they are vendored verbatim below this header
# and `soma/` can be archived without breaking GNOMON. Their SOMA oracle
# tests are vendored with them and must pass unchanged.
"""Muon, AdamW, and the warmup-stable-decay schedule.

SOMA-SPEC: 05 §5.7
SOMA-SPEC: 05 §5.8

```
G      = grad
M_buf  = mu_muon * M_buf + G                          mu_muon = 0.95
G'     = G + mu_muon * M_buf                          (Nesterov)
O      = NewtonSchulz5( G' / ||G'||_F )
W     <- W - lr(n) * 0.2 * sqrt(max(rows,cols)) * O                       (5.21)
```

`NewtonSchulz5` runs 5 iterations of `X <- a X + b (X X^T) X + c (X X^T)^2 X`
with `(a,b,c) = (3.4445, -4.7750, 2.0315)`, in `float32`.

**[NORMATIVE — optimizer state under structural change]** All per-unit tensors
(`p_u` and its optimizer state) are preallocated at `U_max`. A death **zeroes**
the freed slot's parameter and optimizer state. A birth writes the child's
position and leaves its optimizer state at **zero** — momentum is not inherited,
since inheriting it would inject a non-zero update into a unit whose mask is `0`
and would violate the spirit of Theorem 3.1a.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

__all__ = ["Muon", "newton_schulz5", "wsd_lr", "build_optimizers", "zero_slot_state"]


NS_COEFFS = (3.4445, -4.7750, 2.0315)
"""§05.7's quintic coefficients."""


def newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalisation by a quintic Newton-Schulz iteration.

    SOMA-SPEC: 05 §5.7

    `X <- a X + b (X X^T) X + c (X X^T)^2 X`, five iterations, in `float32`.

    The iteration is applied to the **shorter** orientation: for a wide matrix
    `X X^T` is `rows x rows`, which is the cheaper Gram matrix; transposing
    first when `rows > cols` keeps the per-iteration cost at
    `O(min(m,n)^2 max(m,n))` instead of `O(max(m,n)^2 min(m,n))`.

    The quintic does not converge to the exact orthogonal factor — it drives the
    singular values toward, but not onto, 1. That is the intended behaviour and
    is what `G-MUON` asserts: the spread of the singular values must contract
    sharply, not become exactly unity.
    """
    if G.dim() != 2:
        raise ValueError(f"Newton-Schulz operates on matrices; got shape {tuple(G.shape)}")
    a, b, c = NS_COEFFS
    X = G.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.transpose(0, 1)
    # The iteration's basin of attraction is the unit spectral ball; the
    # Frobenius normalisation guarantees sigma_max <= 1 for any input.
    X = X / X.norm().clamp_min(1e-7)
    for _ in range(steps):
        A = X @ X.transpose(0, 1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(0, 1)
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """(5.21). One momentum buffer, no second moment.

    SOMA-SPEC: 05 §5.7

    **[RATIONALE]** Muon reports roughly `2x` compute efficiency against AdamW
    at compute-optimal training and is also a memory win (one momentum buffer,
    no second moment). §05.10 budgets it at 6 bytes/param against AdamW's 12.

    Weight decay `0.1` applies to group **M** only (§05.7).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        scale: float = 0.2,
    ):
        if lr < 0:
            raise ValueError("lr must be non-negative")
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            ns_steps=ns_steps, scale=scale,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError(
                        f"Muon's group is enumerated by name in §05.7 and contains "
                        f"only 2-D matrices; got a {p.dim()}-D parameter of shape "
                        f"{tuple(p.shape)}"
                    )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            steps = group["ns_steps"]
            scale = group["scale"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                # Nesterov: the update looks one momentum step ahead.
                g_nesterov = g.add(buf, alpha=mu)
                o = newton_schulz5(g_nesterov, steps)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(o, alpha=-lr * scale * math.sqrt(max(p.shape[0], p.shape[1])))
        return loss


def wsd_lr(n: int, cfg) -> float:
    """Warmup-stable-decay (§05.8).

    ```
    lr(n) = lr_max * n / n_warm                          n < n_warm
          = lr_max                                       n_warm <= n < n_dec
          = lr_max * (1 - (n - n_dec)/(N_step - n_dec))  n >= n_dec
    ```
    """
    if n < cfg.n_warm:
        return cfg.lr_max * n / max(1, cfg.n_warm)
    if n < cfg.n_dec:
        return cfg.lr_max
    span = max(1, cfg.N_step - cfg.n_dec)
    return cfg.lr_max * max(0.0, 1.0 - (n - cfg.n_dec) / span)


def build_optimizers(
    model, cfg
) -> Tuple[Muon, torch.optim.AdamW]:
    """§05.7's two optimizers over the two named groups."""
    muon_params, adamw_params = model.optimizer_groups()
    muon = Muon(muon_params, lr=cfg.lr_max, momentum=0.95, weight_decay=0.1)
    adamw = torch.optim.AdamW(
        adamw_params, lr=cfg.lr_max, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )
    return muon, adamw


@torch.no_grad()
def zero_slot_state(
    optimizer: torch.optim.Optimizer, param: torch.Tensor, slots: torch.Tensor
) -> None:
    """Zero a freed slot's optimizer state (§05.7).

    A death zeroes the parameter and its state; a birth leaves the child's state
    at zero. Inheriting momentum would inject a non-zero update into a unit whose
    mask is `0`, and Theorem 3.1a's guarantee is that a newborn changes nothing
    until its mask rises — a guarantee a stale momentum buffer would break at the
    very first step.
    """
    if slots.numel() == 0:
        return
    state = optimizer.state.get(param)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
        buf = state.get(key)
        if isinstance(buf, torch.Tensor) and buf.shape[0] == param.shape[0]:
            buf[slots] = 0.0
