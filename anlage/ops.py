"""The functional operators of the phenotype.

SPEC: 04 §4.1
SPEC: 04 §4.3
SPEC: 04 §4.4

Every operator is a pure function of activations and weights. The delta-rule
recurrence is the vendored, oracle-verified `gated_delta_rule_chunk`; the
attention path uses `scaled_dot_product_attention` with the sigmoid output gate
of §04.4, and is checked against `gnomon/pheno/ops.py::attention` (with a
zero-mass sink column) as its float64 oracle (P2).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from gnomon.pheno.mixer_chunk import gated_delta_rule_chunk, BETA_MAX
from gnomon.pheno.mixer_ref import gated_delta_rule_reference
from gnomon.pheno.rmsnorm import masked_rms_norm, rms_norm_headwise
from gnomon.utils.shapes import check_shape

__all__ = [
    "rms_norm_gf", "rope", "causal_conv4", "attention_gated", "delta_mixer",
    "swiglu", "softcap_logits", "chunked_cross_entropy", "A0_DECAY", "G0_GATE",
    "SOFTCAP", "Z_LOSS",
]

A0_DECAY = 4.0
"""SPEC: 04 §4.4 -- innate offset of the decay gate: alpha = sigmoid(a_r + 4) ~ 0.98 at init."""
G0_GATE = 2.0
"""SPEC: 04 §4.4 -- innate offset of the attention output gate: sigmoid(2) ~ 0.88 at init."""
SOFTCAP = 15.0
"""SPEC: 04 §4.3 -- logits = c tanh(z / c)."""
Z_LOSS = 1e-4
"""SPEC: 04 §4.3 -- z-loss coefficient."""


def rms_norm_gf(x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                eps: float = 1e-6) -> torch.Tensor:
    """Gain-free RMSNorm at >= float32; masked form (3.4) when a channel gate is given.

    SPEC: 04 §4.3, SPEC: 06 §6.3.
    """
    if mask is not None:
        return masked_rms_norm(x, mask, eps)
    acc = torch.promote_types(x.dtype, torch.float32)
    xf = x.to(acc)
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def rope(x: torch.Tensor, theta: float = 1e4) -> torch.Tensor:
    """Rotary embedding over all `d_k` dims of `(B, H, T, d_k)`. SPEC: 04 §4.4."""
    B, H, T, d = x.shape
    if d % 2:
        raise ValueError("RoPE needs an even head dimension")
    half = d // 2
    freqs = theta ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    ang = pos[:, None] * freqs[None, :]                                  # (T, half)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


def causal_conv4(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """`conv_4(x; W)[t, c] = sum_{tau=0}^{3} W[c, tau] x[t - tau, c]`. SPEC: 04 §4.4.

    `x` is `(B, T, C)`, `W` is `(C, 4)`. Causal by a left pad of 3; P4 verifies
    that breaking the pad leaks the future.
    """
    B, T, C = x.shape
    check_shape(W, [C, 4], name="conv weight")
    xp = F.pad(x.transpose(1, 2), (3, 0))                                # (B, C, T+3)
    # depthwise conv: flip so that weight index tau multiplies x[t - tau]
    w = W.flip(-1).unsqueeze(1).to(x.dtype)                              # (C, 1, 4)
    y = F.conv1d(xp, w, groups=C)                                        # (B, C, T)
    return y.transpose(1, 2)


def attention_gated(
    h: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor,
    W_o: torch.Tensor, W_gate: torch.Tensor, *, n_h: int, n_kv: int, d_head: int,
    ve: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None,
    gate_offset: float = G0_GATE, use_rope: bool = True,
) -> torch.Tensor:
    """(4.3): SDPA attention with QK-norm, RoPE, GQA by `g mod n_kv`, sigmoid output gate.

    SPEC: 04 §4.4. `ve`, if given, is `(B, T, n_kv d_head)` added to `v` before
    the head split (value embeddings). `head_mask` `(n_h d_head,)` gates head
    channels before `W_o` (growth, §06.3).
    """
    B, T, _ = h.shape
    q = F.linear(h, W_q).view(B, T, n_h, d_head).transpose(1, 2)
    k = F.linear(h, W_k).view(B, T, n_kv, d_head).transpose(1, 2)
    v = F.linear(h, W_v)
    if ve is not None:
        v = v + ve
    v = v.view(B, T, n_kv, d_head).transpose(1, 2)
    q, k = rms_norm_headwise(q), rms_norm_headwise(k)
    if use_rope:
        q, k = rope(q), rope(k)
    if n_kv != n_h:
        rep = n_h // n_kv
        # `repeat`, not `repeat_interleave`: head g reads KV head g mod n_kv (§04.2).
        k = k.repeat(1, rep, 1, 1)
        v = v.repeat(1, rep, 1, 1)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                       scale=1.0 / math.sqrt(d_head))
    a = a.transpose(1, 2).reshape(B, T, n_h * d_head)
    gate = torch.sigmoid(F.linear(h, W_gate) + gate_offset)
    a = a * gate
    if head_mask is not None:
        a = a * head_mask.to(a.dtype)
    return F.linear(a, W_o)


def delta_mixer(
    h: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor,
    W_o: torch.Tensor, W_ab: torch.Tensor, W_og: torch.Tensor,
    W_cq: torch.Tensor, W_ck: torch.Tensor, W_cv: torch.Tensor, *,
    n_h: int, d_head: int, chunk_size: int, head_mask: Optional[torch.Tensor] = None,
    decay_offset: float = A0_DECAY, use_reference: bool = False,
    state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(4.4): the gated delta rule with short conv, L2 keys, gates. SPEC: 04 §4.4.

    Delta layers use full heads (`n_kv = n_h`, §04.2).
    """
    B, T, _ = h.shape
    q = F.silu(causal_conv4(F.linear(h, W_q), W_cq))
    k = F.silu(causal_conv4(F.linear(h, W_k), W_ck))
    v = F.silu(causal_conv4(F.linear(h, W_v), W_cv))
    q = q.view(B, T, n_h, d_head).transpose(1, 2)
    k = k.view(B, T, n_h, d_head).transpose(1, 2)
    v = v.view(B, T, n_h, d_head).transpose(1, 2)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ab = F.linear(h, W_ab).transpose(1, 2)                                # (B, 2h, T)
    # SPEC: 04 §4.4 -- the GATE axis is interleaved per head: row 2g is head
    # g's decay gate, row 2g+1 its write gate. A block layout (all alphas, then
    # all betas) would break prefix nesting the moment a head is appended:
    # old beta_0 would land on new alpha_2's row (found by G1).
    alpha = torch.sigmoid(ab[:, 0::2] + decay_offset)
    beta = (2.0 * torch.sigmoid(ab[:, 1::2])).clamp(max=BETA_MAX)
    if use_reference:
        o, S = gated_delta_rule_reference(q, k, v, alpha, beta, initial_state=state)
    else:
        o, S = gated_delta_rule_chunk(q, k, v, alpha, beta, chunk_size=chunk_size,
                                      initial_state=state)
    o = rms_norm_headwise(o).transpose(1, 2).reshape(B, T, n_h * d_head)
    o = o * F.silu(F.linear(h, W_og))
    if head_mask is not None:
        o = o * head_mask.to(o.dtype)
    return F.linear(o, W_o), S


def swiglu(h: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor,
           W_down: torch.Tensor, inner_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`W_down( SiLU(W_gate h) * (W_up h) )`, with an optional inner gate (§06.3)."""
    f = F.silu(F.linear(h, W_gate)) * F.linear(h, W_up)
    if inner_mask is not None:
        f = f * inner_mask.to(f.dtype)
    return F.linear(f, W_down)


def softcap_logits(z: torch.Tensor, log_scale: torch.Tensor, c: float = SOFTCAP) -> torch.Tensor:
    """`c tanh(exp(lambda) z / c)`. SPEC: 04 §4.3."""
    return c * torch.tanh(z * torch.exp(log_scale) / c)


def chunked_cross_entropy(x: torch.Tensor, E: torch.Tensor, log_scale: torch.Tensor,
                          targets: torch.Tensor, *, chunk: int = 4096,
                          z_loss: float = Z_LOSS) -> tuple[torch.Tensor, torch.Tensor]:
    """CE in float32 over chunks of positions, with the z-loss. SPEC: 04 §4.3, §4.6.

    `x` is `(N, d)` already normalised and gained, `E` the head `(V, d)`,
    `targets (N,)`. Returns `(ce, zloss)` as means over `N`. The logits are never
    materialised at `N x V` in float32 all at once.
    """
    N = x.shape[0]
    ce_sum = x.new_zeros((), dtype=torch.float32)
    z_sum = x.new_zeros((), dtype=torch.float32)
    for a in range(0, N, chunk):
        xb, tb = x[a:a + chunk], targets[a:a + chunk]
        logits = softcap_logits(F.linear(xb, E), log_scale).float()
        lse = torch.logsumexp(logits, dim=-1)
        ce_sum = ce_sum + (lse - logits.gather(-1, tb[:, None]).squeeze(-1)).sum()
        z_sum = z_sum + (lse ** 2).sum()
    return ce_sum / N, z_loss * z_sum / N
