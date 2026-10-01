"""The mixer's chunkwise parallel form — WY representation and the UT transform.

SPEC: 05 §5.2.3
SPEC: 05 §5.2.5

This module implements (5.6)-(5.16), including **the equation an audit found
wrong in spec v1** (5.16). The v1 form omitted `Gam_decay`:

    O = Q<- S^T + (Q K^T (*) M)(U_tilde - W<- S^T)              # WRONG

which yields `sum_i (k^iT q^r) u~^i` — missing `g_r/g_i` — and
`- sum_i (k^iT q^r) g_i (S w^i)^T`, carrying `g_i` where `g_r` is required. It
is correct **only** in the ungated limit `alpha == 1`. That is the failure mode
worth naming: a mixer that trains, produces plausible losses, and passes every
shape test while computing a different operator from the one specified.
`G-MIXER-UNGATED` pins the `alpha == 1` equivalence so the error cannot silently
return.

Numerical policy (§05.2.3, §05.9):

* `alpha in (0,1)` is a **positive scalar** and is the only quantity entering
  the intra-chunk cumulative product, so `log g_r = sum_{i<=r} log alpha^i` is
  well defined and is accumulated in `float32`;
* `beta in (0,2)` carries the **sign** of the transition eigenvalue but appears
  only inside the Householder factor, never accumulated as a scalar product; it
  enters through the UT transform, an explicit triangular solve;
* the triangular solves (5.11), (5.14) are computed at **no less than**
  `float32` -- `promote_types(ambient, float32)` -- because their conditioning
  degrades in `bfloat16`. §05.9 v2 said "`float32` regardless of ambient dtype",
  which downcasts a `float64` call and makes §05.2.6's own `1e-10` `float64`
  oracle test unsatisfiable (audit A-47); the requirement is a floor.

The mechanisms are compositional: the sign lives in `beta`, the log-space
reduction in `alpha`, and they do not interact.
"""

from __future__ import annotations

import contextlib
import math
from typing import Optional, Tuple

import torch

from soma.utils.shapes import check_finite, check_shape

__all__ = [
    "gated_delta_rule_chunk",
    "ut_condition_number",
    "BETA_MAX",
    "CHECK_FINITE",
]

CHECK_FINITE = False
"""Whether the chunkwise form asserts finiteness of its two solves.

Each assertion is a `.all()` reduction, i.e. a **GPU->CPU synchronisation**, and
the mixer runs 9 times per forward at M1 -- 18 pipeline stalls per step, which
measured as a large fraction of step time. §07.4.3's criterion 1 is enforced
once per step on the loss and on the clipped gradient norm (audit A-56), which
covers the same failure for one scalar instead of eighteen stalls.

The tests set it True: there the cost is irrelevant and a localised message is
worth more than a step-level one.
"""

BETA_MAX = 2.0 - 1e-3
"""§05.2.2 / §05.2.3: `beta_mix <= 2 - 1e-3` is clamped.

Unbounded `beta` with `||k|| = 1` admits inverses whose entries grow with chunk
length; `tests/regression/test_signed_gdn.py` asserts
`cond(I + strictlower(diag(beta) K K^T)) < 1e6` over the test grid.
"""


def _pad_to_multiple(
    x: torch.Tensor, chunk: int, dim: int, value: float = 0.0
) -> Tuple[torch.Tensor, int]:
    """Pad `dim` up to a multiple of `chunk`; return the tensor and the padding."""
    T = x.shape[dim]
    rem = (-T) % chunk
    if rem == 0:
        return x, 0
    pad_shape = list(x.shape)
    pad_shape[dim] = rem
    pad = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim), rem


def ut_condition_number(
    beta: torch.Tensor, k: torch.Tensor, gam: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """`cond(I + strictlower(diag(beta) (Gam (*) K K^T)))`, per chunk.

    SPEC: 05 §5.2.3

    Returned rather than asserted so the regression test owns the threshold and
    the training loop can log the realised value. `beta` is `(..., C)`, `k` is
    `(..., C, d_k)`.
    """
    kkt = k.float() @ k.float().transpose(-1, -2)
    if gam is not None:
        kkt = kkt * gam.float()
    C = k.shape[-2]
    tril = torch.tril(torch.ones(C, C, device=k.device, dtype=torch.float32), -1)
    M = torch.eye(C, device=k.device, dtype=torch.float32) + (
        beta.float().unsqueeze(-1) * kkt
    ) * tril
    return torch.linalg.cond(M)


def gated_delta_rule_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
    initial_state: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
    return_condition: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(5.6)-(5.16). Mathematically identical to `gated_delta_rule_reference`.

    Shapes are as in the reference: `q, k` are `(B, H, T, d_k)`, `v` is
    `(B, H, T, d_v)`, `alpha, beta` are `(B, H, T)`, the optional
    `initial_state` is `(B, H, d_v, d_k)`.

    §05.2.5: sequences are padded to a multiple of `C_chunk` and pad positions
    are masked; `C_chunk` is **not** required to divide `T_ctx`.
    """
    with _no_autocast(q.device):
        return _chunk_impl(
            q, k, v, alpha, beta, chunk_size=chunk_size,
            initial_state=initial_state, valid=valid,
            return_condition=return_condition,
        )


def _no_autocast(device: torch.device):
    """Disable autocast for the chunkwise form.

    SPEC: 05 §5.9

    §05.9 sets a **floor** on the precision of the triangular solves, and
    `promote_types(ambient, float32)` implements it -- but under
    `torch.autocast` that is not enough: autocast intercepts `matmul` and
    downcasts `float32` operands back to `bfloat16`, so `K K^T` and therefore
    `A_w` return to `bfloat16` and `solve_triangular` runs at a precision the
    specification forbids. Measured, it produces an all-`NaN` `W_WY`.

    Disabling autocast here is what makes the floor a floor. It costs little:
    the solve is `O(C^2 d_k)` against the block's `O(T d_model^2)`.
    """
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", enabled=False)


def _chunk_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
    initial_state: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
    return_condition: bool = False,
):
    b = {}
    check_shape(q, ["B", "H", "T", "dk"], name="q_mix", bindings=b)
    check_shape(k, ["B", "H", "T", "dk"], name="k_mix", bindings=b)
    check_shape(v, ["B", "H", "T", "dv"], name="v_mix", bindings=b)
    check_shape(alpha, ["B", "H", "T"], name="alpha_mix", bindings=b)
    check_shape(beta, ["B", "H", "T"], name="beta_mix", bindings=b)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")

    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    out_dtype, device = q.dtype, q.device
    # §05.9 (A-47): a floor of float32, not an exact dtype. bfloat16/float16 are
    # promoted; float64 is preserved, which is what makes the §05.2.6 oracle
    # comparison at 1e-10 attainable at all.
    solve_dtype = torch.promote_types(out_dtype, torch.float32)

    # A padded position must not write, must not decay the state, and must emit
    # zero. Doing this before padding also handles a caller-supplied `valid`.
    if valid is not None:
        check_shape(valid, ["B", "H", "T"], name="valid", bindings=b)
        alpha = torch.where(valid, alpha, torch.ones_like(alpha))
        beta = torch.where(valid, beta, torch.zeros_like(beta))
        q = q * valid.unsqueeze(-1).to(q.dtype)

    # --- pad ---------------------------------------------------------------
    q_p, pad = _pad_to_multiple(q, chunk_size, 2)
    k_p, _ = _pad_to_multiple(k, chunk_size, 2)
    v_p, _ = _pad_to_multiple(v, chunk_size, 2)
    # alpha = 1 on the pad so `g` is unchanged; beta = 0 so nothing is written.
    a_p, _ = _pad_to_multiple(alpha, chunk_size, 2, value=1.0)
    b_p, _ = _pad_to_multiple(beta, chunk_size, 2, value=0.0)

    Tp = q_p.shape[2]
    n_chunks = Tp // chunk_size
    C = chunk_size

    # --- reshape into chunks ----------------------------------------------
    # float32 throughout the solve path (§05.9).
    q_c = q_p.reshape(B, H, n_chunks, C, d_k).to(solve_dtype)
    k_c = k_p.reshape(B, H, n_chunks, C, d_k).to(solve_dtype)
    v_c = v_p.reshape(B, H, n_chunks, C, d_v).to(solve_dtype)
    a_c = a_p.reshape(B, H, n_chunks, C).to(solve_dtype)
    b_c = b_p.reshape(B, H, n_chunks, C).to(solve_dtype)

    b_c = b_c.clamp(min=0.0, max=BETA_MAX)

    # --- (5.6) g_r in log space -------------------------------------------
    # g_r = prod_{i=1..r} alpha^i, g_0 = 1. Accumulated as a cumulative sum of
    # log alpha in float32; alpha > 0 so the log is defined, which is exactly
    # the property §05.2.3 identifies as making the composition work.
    log_a = torch.log(a_c.clamp_min(1e-30))
    log_g = torch.cumsum(log_a, dim=-1)                     # (B,H,N,C)
    g = torch.exp(log_g)

    # Gam_decay[r, i] = g_r / g_i for r >= i, else 0.
    # Computed as exp(log g_r - log g_i) so the ratio never forms a quotient of
    # two underflowed products.
    log_ratio = log_g.unsqueeze(-1) - log_g.unsqueeze(-2)   # (B,H,N,C,C) [r,i]
    causal = torch.tril(torch.ones(C, C, device=device, dtype=torch.bool))
    gam = torch.where(causal, torch.exp(log_ratio.clamp(max=0.0)),
                      torch.zeros_like(log_ratio))

    # --- (5.11)-(5.12) WY / UT --------------------------------------------
    kkt = k_c @ k_c.transpose(-1, -2)                       # (B,H,N,C,C)
    eye = torch.eye(C, device=device, dtype=solve_dtype)
    strict_lower = torch.tril(torch.ones(C, C, device=device, dtype=solve_dtype), -1)

    # T_UT = [ I + strictlower( diag(beta) K K^T ) ]^{-1} diag(beta)
    A_w = eye + (b_c.unsqueeze(-1) * kkt) * strict_lower
    diag_beta_K = b_c.unsqueeze(-1) * k_c                   # diag(beta) K
    W_WY = torch.linalg.solve_triangular(A_w, diag_beta_K, upper=False, unitriangular=True)
    if CHECK_FINITE:
        check_finite(W_WY, name="W_WY")

    # --- (5.13)-(5.14) gated value accumulation ---------------------------
    A_u = eye + (b_c.unsqueeze(-1) * (gam * kkt)) * strict_lower
    diag_beta_V = b_c.unsqueeze(-1) * v_c                   # diag(beta) V
    U_tilde = torch.linalg.solve_triangular(A_u, diag_beta_V, upper=False, unitriangular=True)
    if CHECK_FINITE:
        check_finite(U_tilde, name="U_tilde")

    cond = None
    if return_condition:
        cond = torch.linalg.cond(A_u)

    # --- arrow rescalings --------------------------------------------------
    #   Q<-[r] = g_r q^r ,  W<-[r] = g_r w^r ,  K->[i] = (g_C/g_i) k^i , S-> = g_C S
    #
    # `Q<-` and `W<-` scale **by** `g_r <= 1`: underflowing to zero is the
    # numerically correct answer, since the contribution really has decayed away.
    #
    # `K->` is a **ratio** `g_C / g_i`, and forming it as a quotient is the one
    # thing this file's `gam` block already forbids -- for a reason that is
    # sharper in the backward pass than in the forward one. `d(g_C/g_i)/dg_i =
    # -g_C/g_i^2`, and `g_i^2` underflows at the square root of `g_i`'s own
    # limit: in float32 the *gradient* is already `NaN` at
    # `sum_j log alpha_j = -96` (`g_i ~ 2e-42`), where the forward value is still
    # perfectly finite. With `C = 64` that is a sustained `alpha ~ 0.22` -- an
    # unremarkable value for a decay gate. Measured: an M-nano lifetime with a
    # finite loss of `4.04` and a non-finite gradient reaching **only**
    # `w_alpha`, the sole parameter feeding `g` (audit A-65).
    #
    # The log-space form has derivative bounded by the ratio itself (`<= 1`) and
    # is finite wherever the mathematics is.
    g_last = g[..., -1:]                                    # (B,H,N,1)
    log_g_last = log_g[..., -1:]                            # (B,H,N,1)
    Q_arrow = q_c * g.unsqueeze(-1)
    W_arrow = W_WY * g.unsqueeze(-1)
    K_arrow = k_c * torch.exp((log_g_last - log_g).clamp(max=0.0)).unsqueeze(-1)

    # --- scan over chunks --------------------------------------------------
    if initial_state is None:
        S = torch.zeros(B, H, d_v, d_k, dtype=solve_dtype, device=device)
    else:
        check_shape(initial_state, ["B", "H", "dv", "dk"], name="S_mix", bindings=b)
        S = initial_state.to(solve_dtype).clone()

    outs = []
    # (Q K^T) (*) Gam (*) M_causal, materialised once per chunk.
    attn = (q_c @ k_c.transpose(-1, -2)) * gam              # gam already carries M

    for c in range(n_chunks):
        S_T = S.transpose(-1, -2)                           # (B,H,d_k,d_v)

        # (U_tilde - W<- S^T): (B,H,C,d_v)
        WS = W_arrow[:, :, c] @ S_T                         # (B,H,C,d_v)
        inner = U_tilde[:, :, c] - WS

        # (5.16)  O = Q<- S^T + ( Q K^T (*) Gam (*) M ) ( U_tilde - W<- S^T )
        O_c = Q_arrow[:, :, c] @ S_T + attn[:, :, c] @ inner
        outs.append(O_c)

        # (5.15)  S_next = S-> + ( U_tilde - W<- S^T )^T K->
        S = S * g_last[:, :, c].unsqueeze(-1) + inner.transpose(-1, -2) @ K_arrow[:, :, c]

    O = torch.stack(outs, dim=2).reshape(B, H, Tp, d_v)
    if pad:
        O = O[:, :, :T, :]

    O = O.to(out_dtype)
    S = S.to(out_dtype)
    if return_condition:
        return O, S, cond  # type: ignore[return-value]
    return O, S
