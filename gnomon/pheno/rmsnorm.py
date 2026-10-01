# VENDORED from `soma/model/rmsnorm.py` at commit `7d29cc1` (SOMA v3.1).
#
# SPEC: 04 §4.1 -- the numerical cores are already pure functions of
# activations and weights, so they are vendored verbatim below this header
# and `soma/` can be archived without breaking GNOMON. Their SOMA oracle
# tests are vendored with them and must pass unchanged.
"""RMSNorm, its head-wise gain-free variant, and the masked form of (3.4).

SOMA-SPEC: 05 §5.1
SOMA-SPEC: 03 §3.3

Three normalisations, kept together because their only differences are which
statistics they compute over and whether a gain is learned — and because
confusing them is exactly how the QK-norm bound of §05.3 and the growth
invariance of Theorem 3.1b get silently broken.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..utils.shapes import check_shape

__all__ = ["RMSNorm", "rms_norm_headwise", "masked_rms_norm"]

EPS = 1e-6
"""(5.1)."""


class RMSNorm(nn.Module):
    """(5.1): `x / sqrt(mean(x^2) + eps) * gain_rms`.

    SOMA-SPEC: 05 §5.1

    `gain_rms in R^{d_model}` is learned and initialised to 1. **No bias terms
    appear anywhere in the model** (§05.1) — which is why §03.4's death rule
    performs no bias transfer: spec v1's death step tried to fold a dying unit's
    average contribution "into the bias of each consumer", and there are no
    biases and no consumers.

    The reduction is computed at **no less than** `float32` --
    `promote_types(ambient, float32)`: the mean of squares of a `bfloat16`
    vector of width 512 loses ~3 bits to the accumulation alone, which then
    multiplies the whole residual stream. It is a floor, not an exact dtype;
    downcasting a `float64` call would cap the achievable precision at the
    `float32` epsilon, which is the defect audit A-47 records for the mixer's
    triangular solve.
    """

    def __init__(self, dim: int, eps: float = EPS, *, elementwise_gain: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        if elementwise_gain:
            self.gain_rms = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("gain_rms", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_shape(x, [*([None] * (x.dim() - 1)), self.dim], name="RMSNorm input")
        dtype = x.dtype
        acc = torch.promote_types(dtype, torch.float32)
        xf = x.to(acc)
        norm = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        out = norm.to(dtype)
        if self.gain_rms is not None:
            out = out * self.gain_rms
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, gain={self.gain_rms is not None}"


def rms_norm_headwise(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Gain-free RMSNorm over the last (head) dimension.

    SOMA-SPEC: 05 §5.3

    Used for QK-norm. **Gain-free** is normative: with a learned gain the bound
    `|s| <= sqrt(d_h)` does not hold, since the gain is unbounded — spec v1
    asserted the bound while specifying a gained norm.

    The bound is what the attention-entropy argument rests on: entropy collapses
    at a rate exponential in the spectral norm of the logits, so bounding
    `||q||, ||k||` caps that norm and permits a higher stable learning rate.
    A learned gain reintroduces exactly the unboundedness the norm removed.
    """
    dtype = x.dtype
    acc = torch.promote_types(dtype, torch.float32)
    xf = x.to(acc)
    out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return out.to(dtype)


def masked_rms_norm(
    x: torch.Tensor, mask: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """(3.4) `[OURS]`: RMSNorm whose second moment is taken over live coordinates.

    SOMA-SPEC: 03 §3.3

    ```
    RMS:  var' = ((x')^2 . c) / sum(c)                              (3.4)
    ```

    The source operator specifies LayerNorm only; (3.4) is our extension, and
    the derivation is immediate — RMSNorm omits the mean, so only the masked
    second moment needs correction. `tests/unit/test_masked_norm.py` asserts
    (3.4) reduces to (3.3) at zero mean.

    Why it is needed: under width growth with a per-coordinate mask `c`, the
    grown module's output must equal the ungrown module's at `c = 0`
    (Theorem 3.1b). An unmasked RMSNorm divides by `sqrt(mean over ALL d')`,
    which shrinks by `sqrt(d_old / d_new)` the moment a coordinate is added —
    so the invariance fails before any new weight has a non-zero value.

    `mask` broadcasts against `x`'s last dimension and holds values in `[0, 1]`.
    """
    if mask.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"mask last dim {mask.shape[-1]} != x last dim {x.shape[-1]}"
        )
    dtype = x.dtype
    acc = torch.promote_types(dtype, torch.float32)
    xf = x.to(acc)
    m = mask.to(acc)
    denom = m.sum(-1, keepdim=True).clamp_min(1.0)
    var = (xf.square() * m).sum(-1, keepdim=True) / denom
    out = xf * torch.rsqrt(var + eps)
    # Coordinates outside the mask contribute nothing downstream; zeroing them
    # here makes the invariance exact rather than merely approximate, since a
    # masked coordinate's value would otherwise leak through a later unmasked op.
    return (out * m).to(dtype)
