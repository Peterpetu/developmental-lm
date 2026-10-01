"""The mixer block: feature construction and the gated output path.

SPEC: 05 §5.2.2
SPEC: 05 §5.2.4

```
W_qkv   in R^{d_model x D_qkv}         D_qkv = n_m * (2 d_k + d_v)
conv    depthwise causal, kernel 4, one filter per channel of D_qkv
w_alpha, w_beta in R^{d_model x n_m}
W_gate  in R^{d_model x (d_model e_v)}
OutProj in R^{(d_model e_v) x d_model}

q_raw, k_raw, v_raw = split( x W_qkv )
q_mix = L2Norm( SiLU( conv(q_raw) ) )        # unit norm per head
k_mix = L2Norm( SiLU( conv(k_raw) ) )
v_mix =         SiLU( conv(v_raw) )
alpha_mix = sigmoid( x w_alpha )
beta_mix  = min( 2-1e-3, 2*sigmoid( x w_beta ) )
y = OutProj( RMSNorm(o) * SiLU( x W_gate ) )
```

**[EMPIRICAL — component priority]** From a 400M/15B ablation, cost of removal
in average perplexity / average accuracy: `alpha_mix` gate **+3.52 / -2.14**
(largest); L2 replaced by L1+SiLU **+2.83 / -1.17**; output gate
**+1.77 / -1.80**; short convolution **+1.60 / -1.10**; output norm
**+0.20 / -0.19**. §05.2.4 therefore forbids removing the `alpha_mix` gate or
the L2 normalisation except as a labelled experiment, and permits dropping the
output norm under memory pressure.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from soma.config import SomaConfig
from soma.model.mixer_chunk import BETA_MAX, gated_delta_rule_chunk
from soma.model.mixer_ref import gated_delta_rule_reference
from soma.model.rmsnorm import RMSNorm
from soma.utils.shapes import check_shape

__all__ = ["Mixer", "CausalDepthwiseConv1d"]


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal convolution, kernel 4, one filter per channel (§05.2.4).

    Causal by left-padding with `kernel - 1` zeros rather than by masking, so no
    future position can contribute at any point in the computation — masking
    after the fact leaves the value in the tensor and relies on every downstream
    consumer honouring the mask.
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=False
        )
        # Initialised as an identity-in-time filter (a delta at the current
        # position) so the block starts as if there were no convolution, and the
        # short-range mixing is learned rather than imposed at initialisation.
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, -1] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is `(B, T, C)`; returns `(B, T, C)`."""
        check_shape(x, [None, None, self.channels], name="conv input")
        h = x.transpose(1, 2)                                  # (B, C, T)
        h = F.pad(h, (self.kernel_size - 1, 0))
        h = self.conv(h)
        return h.transpose(1, 2)


class Mixer(nn.Module):
    """One mixer sublayer (§05.2).

    The recurrence itself is delegated to `mixer_chunk` (production) or
    `mixer_ref` (the oracle). §05.2.6 maintains both and asserts their agreement;
    keeping the feature construction here means the two paths differ only in the
    recurrence, which is what makes the comparison a test of the recurrence.
    """

    def __init__(self, cfg: SomaConfig):
        super().__init__()
        self.cfg = cfg
        self.n_m = cfg.n_m
        self.d_k = cfg.d_k
        self.d_v = cfg.d_v
        self.d_model = cfg.d_model
        self.d_inner = cfg.d_model * cfg.e_v
        self.C_chunk = cfg.C_chunk

        self.W_qkv = nn.Linear(cfg.d_model, cfg.D_qkv, bias=False)
        self.conv = CausalDepthwiseConv1d(cfg.D_qkv, 4)
        self.w_alpha = nn.Linear(cfg.d_model, cfg.n_m, bias=False)
        self.w_beta = nn.Linear(cfg.d_model, cfg.n_m, bias=False)
        self.W_gate = nn.Linear(cfg.d_model, self.d_inner, bias=False)
        self.OutProj = nn.Linear(self.d_inner, cfg.d_model, bias=False)
        self.out_norm = RMSNorm(self.d_inner)

        # **Ablation A7 (§08.6):** `beta_mix = sigmoid`, range `[0, 1]` instead
        # of `[0, 2)`. §05.2.2's signed transitions need `beta > 1` to reach a
        # **negative** eigenvalue, so A7 removes exactly the mechanism §05.11's
        # falsifier and K7 are written about, leaving a gated delta rule that can
        # forget but never reverse. A module flag rather than a config constant,
        # because it changes the forward and not a number.
        self.beta_unsigned = False

    # ------------------------------------------------------------------

    def features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """§05.2.4's feature construction. Returns `(q, k, v, alpha, beta)`.

        `q, k` are `(B, n_m, T, d_k)`, `v` is `(B, n_m, T, d_v)`,
        `alpha, beta` are `(B, n_m, T)`.
        """
        B, T, _ = x.shape
        qkv = self.conv(self.W_qkv(x))                          # (B, T, D_qkv)
        nq = self.n_m * self.d_k
        nk = self.n_m * self.d_k
        q_raw, k_raw, v_raw = torch.split(qkv, [nq, nk, self.n_m * self.d_v], dim=-1)

        def heads(t: torch.Tensor, d: int) -> torch.Tensor:
            return t.view(B, T, self.n_m, d).transpose(1, 2)    # (B, n_m, T, d)

        q = heads(F.silu(q_raw), self.d_k)
        k = heads(F.silu(k_raw), self.d_k)
        v = heads(F.silu(v_raw), self.d_v)

        # L2 normalisation is per head, which is what gives (5.5) its `||k|| = 1`
        # premise. Without it the transition matrix is not a scaled Householder
        # and the eigenvalue statement does not hold.
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        alpha = torch.sigmoid(self.w_alpha(x)).transpose(1, 2)  # (B, n_m, T)
        scale = 1.0 if self.beta_unsigned else 2.0          # A7 (§08.6)
        beta = (scale * torch.sigmoid(self.w_beta(x))).clamp(max=BETA_MAX)
        beta = beta.transpose(1, 2)
        return q, k, v, alpha, beta

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        use_reference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`x` is `(B, T, d_model)`; returns `(y, final_state)`.

        `use_reference` selects the naive recurrence, for the equivalence test
        and for debugging. It is never used in training: the reference is
        `O(T)` Python-level iterations.
        """
        check_shape(x, [None, None, self.d_model], name="mixer input")
        q, k, v, alpha, beta = self.features(x)

        if valid is not None:
            # (B, T) -> (B, n_m, T), broadcast across heads.
            valid = valid.unsqueeze(1).expand(-1, self.n_m, -1)

        runner = gated_delta_rule_reference if use_reference else gated_delta_rule_chunk
        kwargs = {} if use_reference else {"chunk_size": self.C_chunk}
        o, new_state = runner(
            q, k, v, alpha, beta, initial_state=state, valid=valid, **kwargs
        )

        B, _, T, _ = o.shape
        o = o.transpose(1, 2).reshape(B, T, self.n_m * self.d_v)
        y = self.OutProj(self.out_norm(o) * F.silu(self.W_gate(x)))
        check_shape(y, [B, T, self.d_model], name="mixer output")
        return y, new_state
