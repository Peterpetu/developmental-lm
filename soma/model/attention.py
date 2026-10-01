"""Full causal attention with GQA, gain-free QK-norm, attention sinks and NoPE.

SPEC: 05 §5.3

At M-nano and M1 the attention layers are **full causal attention**; the sliding
window is disabled. Spec v1 specified a 256-window against a context of 512 for
*every* attention layer, which left the model with **no global receptive field
at all**, contradicted §05.3's own reference to "the global path", and
invalidated the transfer of the cited hybrid study (whose non-linear layers were
full attention). Windowing is an `[OPEN]` scaling-path option enabled only for
`T_ctx > 2048`.

```
q, k   = RMSNorm_headwise(q), RMSNorm_headwise(k)     # gain-free
s[t,j] = ( q[t] . k[j] ) / sqrt(d_h)      for j <= t
a[t]   = softmax( [ s[t, 0..t] ; z_sink[h] ] )
o[t]   = sum_j a[t,j] v[j]                            # the sink's value is 0
```
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from soma.config import SomaConfig
from soma.model.rmsnorm import rms_norm_headwise
from soma.utils.shapes import check_shape

__all__ = ["Attention", "causal_mask", "document_mask"]


def causal_mask(T: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """`(T, T)` boolean: True where attention is permitted (`j <= t`)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def document_mask(doc_id: torch.Tensor) -> torch.Tensor:
    """`(B, T, T)` boolean: True where `t` and `j` lie in the same document.

    SPEC: 06 §6.1.1

    §06.1.1 requires attention to be masked at **document boundaries**, so a
    token's conditioning depends only on its own document. That is what makes
    the precomputed `L_ref` cache valid: `L_ref` is context-conditional, and
    without boundary masking a cached value would depend on whatever happened to
    be packed beside it in the stream, which the data-order jitter of §07.3
    changes every run.
    """
    check_shape(doc_id, [None, None], name="doc_id")
    return doc_id.unsqueeze(-1) == doc_id.unsqueeze(-2)


class Attention(nn.Module):
    """One attention sublayer (§05.3)."""

    def __init__(self, cfg: SomaConfig):
        super().__init__()
        self.cfg = cfg
        self.n_h = cfg.n_h
        self.n_kv = cfg.n_kv
        self.d_h = cfg.d_h
        self.d_model = cfg.d_model
        self.repeat = cfg.n_h // cfg.n_kv

        self.W_q = nn.Linear(cfg.d_model, cfg.n_h * cfg.d_h, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.n_kv * cfg.d_h, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.n_kv * cfg.d_h, bias=False)
        self.W_o = nn.Linear(cfg.n_h * cfg.d_h, cfg.d_model, bias=False)

        # §05.3: a learned per-head sink logit, appended to each row with an
        # implicit zero value vector, so the attention weights sum to *less than
        # one* and a head may decline to attend. Softmax otherwise forces a
        # distribution, so a head with nothing to retrieve improvises a sink on
        # the first token and distorts the residual stream.
        self.z_sink = nn.Parameter(torch.zeros(cfg.n_h))

    def forward(
        self,
        x: torch.Tensor,
        *,
        doc_id: Optional[torch.Tensor] = None,
        key_padding_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """`x` is `(B, T, d_model)`; returns `(B, T, d_model)`.

        No positional encoding is applied (**NoPE**, §05.3): causal masking
        alone breaks permutation symmetry, the recurrent layers carry position
        implicitly, and removing explicit position improves length
        extrapolation. RoPE is `[OPEN]`, added only if the M2
        length-extrapolation curve regresses against a RoPE control.
        """
        check_shape(x, [None, None, self.d_model], name="attention input")
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.n_h, self.d_h).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv, self.d_h).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv, self.d_h).transpose(1, 2)

        # Gain-free QK RMSNorm (§05.3). With a learned gain the bound
        # |s| <= sqrt(d_h) does not hold, since the gain is unbounded.
        q = rms_norm_headwise(q)
        k = rms_norm_headwise(k)

        # GQA: each kv head serves `repeat` query heads.
        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=1)
            v = v.repeat_interleave(self.repeat, dim=1)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_h)   # (B,n_h,T,T)

        allowed = causal_mask(T, x.device).unsqueeze(0).unsqueeze(0)
        if doc_id is not None:
            allowed = allowed & document_mask(doc_id).unsqueeze(1)
        if key_padding_valid is not None:
            allowed = allowed & key_padding_valid[:, None, None, :]

        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed, neg_inf)

        # The sink is one extra logit per row with a zero value vector, so it is
        # implemented as an extra softmax column that is then dropped rather
        # than as a real key/value pair.
        sink = self.z_sink.view(1, self.n_h, 1, 1).expand(B, self.n_h, T, 1)
        full = torch.cat([scores, sink.to(scores.dtype)], dim=-1)
        attn = torch.softmax(full.float(), dim=-1).to(x.dtype)
        attn = attn[..., :T]                                        # drop the sink column

        o = attn @ v                                                # (B,n_h,T,d_h)
        o = o.transpose(1, 2).reshape(B, T, self.n_h * self.d_h)
        y = self.W_o(o)
        check_shape(y, [B, T, self.d_model], name="attention output")
        return y

    def attention_weights(
        self, x: torch.Tensor, *, doc_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """The realised `a[t, j]`, for the sink-algebra test and diagnostics.

        Returned **without** the sink column, so `attn.sum(-1) < 1` is exactly
        the property (5.17) claims and `G-ATTN` can assert it.
        """
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.n_h, self.d_h).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv, self.d_h).transpose(1, 2)
        q = rms_norm_headwise(q)
        k = rms_norm_headwise(k)
        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_h)
        allowed = causal_mask(T, x.device).unsqueeze(0).unsqueeze(0)
        if doc_id is not None:
            allowed = allowed & document_mask(doc_id).unsqueeze(1)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        sink = self.z_sink.view(1, self.n_h, 1, 1).expand(B, self.n_h, T, 1)
        full = torch.cat([scores, sink.to(scores.dtype)], dim=-1)
        return torch.softmax(full.float(), dim=-1)[..., :T]
