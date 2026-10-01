"""The operators, as pure functions of activations and weights.

SPEC: 04 §4.1
SPEC: 04 §4.3

No operator owns an `nn.Parameter`. A module that owned weights would be a
`W_shared` waiting to happen, and §00's RC1 is that whatever backpropagation can
bypass, it bypasses. Every weight below arrives as an argument, decoded.

Refactored from SOMA's `model/attention.py` and `model/mixer.py`, whose numerics
are preserved exactly; the recurrence itself is the vendored
`gated_delta_rule_chunk`, still checked against `gated_delta_rule_reference`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .mixer_chunk import gated_delta_rule_chunk
from .mixer_ref import gated_delta_rule_reference
from .rmsnorm import rms_norm_headwise
from ..utils.shapes import check_shape

__all__ = ["embed", "rms_norm", "causal_mask", "document_mask", "attention", "delta_mixer",
           "swiglu", "BETA_MAX"]

# SOMA-SPEC: 05 §5.2.2. `beta in [0, 2)` is what lets the delta-rule transition
# reach a negative eigenvalue; the clamp keeps it strictly inside the interval.
BETA_MAX = 1.999


def rms_norm(x: torch.Tensor, gain, eps: float = 1e-6) -> torch.Tensor:
    """`x / rms(x) * gain`, with the gain decoded rather than owned.

    `gain` may be the scalar `1.0` for the gain-free normalisation Peri-LN uses
    on the way *into* a sublayer (§04.3).
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gain


def causal_mask(T: int, device="cpu") -> torch.Tensor:
    """`(T, T)` boolean: True where attention is permitted (`j <= t`)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def document_mask(doc_id: torch.Tensor) -> torch.Tensor:
    """`(B, T, T)`: True where `t` and `j` lie in the same document."""
    check_shape(doc_id, [None, None], name="doc_id")
    return doc_id.unsqueeze(-1) == doc_id.unsqueeze(-2)


def attention(
    x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor,
    W_o: torch.Tensor, sink: torch.Tensor, *, n_h: int, n_kv: int, d_head: int,
    window: Optional[int] = None, doc_id: Optional[torch.Tensor] = None,
    return_weights: bool = False,
) -> torch.Tensor:
    """GQA + gain-free QK-norm + attention sinks + NoPE. SPEC: 04 §4.1.

    ```
    q, k   = rms_norm_headwise(q), rms_norm_headwise(k)      # gain-free
    s[t,j] = (q[t] . k[j]) / sqrt(d_head)        for j <= t
    a[t]   = softmax([ s[t, 0..t] ; sink[h] ])
    o[t]   = sum_j a[t,j] v[j]                   # the sink's value is 0
    ```

    The sink is one extra softmax column that is then dropped, so a head may
    decline to attend and `sum_j a[t,j] < 1`. Without it, softmax forces a
    distribution and a head with nothing to retrieve improvises a sink on the
    first token, distorting the residual stream.

    No positional encoding: causal masking alone breaks permutation symmetry.
    """
    B, T, _ = x.shape
    q = F.linear(x, W_q).view(B, T, n_h, d_head).transpose(1, 2)
    k = F.linear(x, W_k).view(B, T, n_kv, d_head).transpose(1, 2)
    v = F.linear(x, W_v).view(B, T, n_kv, d_head).transpose(1, 2)

    # Gain-free: with a learned gain the bound |s| <= sqrt(d_head) does not hold.
    q, k = rms_norm_headwise(q), rms_norm_headwise(k)
    rep = n_h // n_kv
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d_head)
    allowed = causal_mask(T, x.device).unsqueeze(0).unsqueeze(0)
    if window is not None and window < T:
        idx = torch.arange(T, device=x.device)
        allowed = allowed & ((idx[:, None] - idx[None, :]) < window)
    if doc_id is not None:
        allowed = allowed & document_mask(doc_id).unsqueeze(1)
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

    col = sink.view(1, n_h, 1, 1).expand(B, n_h, T, 1).to(scores.dtype)
    full = torch.cat([scores, col], dim=-1)
    # Promote a low-precision softmax to float32, but never *demote* one: SOMA's
    # `.float()` was unconditional, which silently capped a float64 run at
    # float32 and would have weakened every f64 equivalence test that runs
    # through attention (measured: a sink at -30 leaked 1.4e-7 of the mass
    # instead of 3e-15).
    acc = torch.float32 if full.dtype in (torch.float16, torch.bfloat16) else full.dtype
    attn = torch.softmax(full.to(acc), dim=-1).to(x.dtype)
    attn = attn[..., :T]
    if return_weights:
        return attn
    o = (attn @ v).transpose(1, 2).reshape(B, T, n_h * d_head)
    return F.linear(o, W_o)


def delta_mixer(
    x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor,
    W_ab: torch.Tensor, W_o: torch.Tensor, *, n_h: int, n_kv: int, d_head: int,
    chunk_size: int, use_reference: bool = False,
    state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gated delta rule, SOMA (5.6)-(5.16). SPEC: 04 §4.1.

    `W_ab` is `(2 n_h, d_model)` and supplies both gates: `alpha = sigmoid(.)` in
    `[0,1]` is the decay, `beta = 2 sigmoid(.)` in `[0,2)` is the write strength.
    Packing them into one role keeps §04.2's role table at 16 entries and keeps
    the two gates of one head adjacent in the decoder's row-identity space.

    **Deviations from SOMA's mixer, both deliberate and both recorded in
    §04.1:** the causal depthwise convolution and the SiLU output gate are
    dropped. The convolution is a separate weight with no role in the delta-rule
    identity, and the output gate duplicates work the block's SwiGLU already
    does. Neither touches the recurrence, which is the verified part.
    """
    B, T, _ = x.shape
    q = F.linear(x, W_q).view(B, T, n_h, d_head).transpose(1, 2)
    k = F.linear(x, W_k).view(B, T, n_kv, d_head).transpose(1, 2)
    v = F.linear(x, W_v).view(B, T, n_kv, d_head).transpose(1, 2)
    q, k, v = F.silu(q), F.silu(k), F.silu(v)

    rep = n_h // n_kv
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    # L2 per head: (5.5)'s `||k|| = 1` premise, without which the transition is
    # not a scaled Householder and the eigenvalue statement does not hold.
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    ab = F.linear(x, W_ab).transpose(1, 2)                     # (B, 2 n_h, T)
    alpha = torch.sigmoid(ab[:, :n_h])
    beta = (2.0 * torch.sigmoid(ab[:, n_h:])).clamp(max=BETA_MAX)

    if use_reference:
        o, new_state = gated_delta_rule_reference(
            q, k, v, alpha, beta, initial_state=state)
    else:
        o, new_state = gated_delta_rule_chunk(
            q, k, v, alpha, beta, chunk_size=chunk_size, initial_state=state)
    o = o.transpose(1, 2).reshape(B, T, n_h * d_head)
    return F.linear(rms_norm_headwise(o.view(B, T, n_h, d_head)).view(B, T, -1),
                    W_o), new_state


def swiglu(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor,
           W_down: torch.Tensor) -> torch.Tensor:
    """`W_down ( silu(W_gate x) * (W_up x) )`."""
    return F.linear(F.silu(F.linear(x, W_gate)) * F.linear(x, W_up), W_down)


def embed(tokens: torch.Tensor, E: torch.Tensor, *, capturable: bool = False
          ) -> torch.Tensor:
    """Look up token rows of the decoded embedding. SPEC: 04 §4.1, §4.3.

    Two formulations of the same gather, and the difference is entirely in the
    **backward**:

    ``F.embedding``  accumulates the gradient by sorting the indices, which is
                     deterministic -- the same lifetime gives bit-identical `Z`
                     and CE, which §05.7's L7 requires. Its sort calls
                     `unique_by_key`, which **synchronises**, and a synchronising
                     kernel cannot be captured into a CUDA graph:
                     `cudaErrorStreamCaptureUnsupported`.
    ``E[tokens]``    accumulates with `atomicAdd`, which *is* capturable and
                     whose summation order is therefore not fixed. Two runs can
                     differ in the last bits of a float.

    So the choice is exactly a trade of bit-exact reproducibility against the
    ~3,900 kernel launches a captured step collapses into one, and it is made
    explicit rather than silently: `capturable` is set only by the runner's
    `compile_mode="cudagraph"` backend, every other path keeps `F.embedding`,
    and L7's determinism is asserted on the deterministic one.
    """
    return E[tokens] if capturable else F.embedding(tokens, E)
