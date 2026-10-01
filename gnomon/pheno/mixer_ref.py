# VENDORED from `soma/model/mixer_ref.py` at commit `7d29cc1` (SOMA v3.1).
#
# SPEC: 04 §4.1 -- the numerical cores are already pure functions of
# activations and weights, so they are vendored verbatim below this header
# and `soma/` can be archived without breaking GNOMON. Their SOMA oracle
# tests are vendored with them and must pass unchanged.
"""The mixer's naive recurrence — **the oracle**.

SOMA-SPEC: 05 §5.2.6

```
S_mix[t] = S_mix[t-1] ( alpha_mix[t] ( I - beta_mix[t] k[t] k[t]^T ) )
           + beta_mix[t] v[t] k[t]^T                                     (5.2)
o[t]     = S_mix[t] q[t]                                                 (5.3)
```

§05.2.6 makes this the correctness oracle: `mixer_chunk` is verified against it
to `1e-10` in `float64` and `1e-5` in `float32`, across chunk sizes and
sequence lengths that do not divide the chunk. It is written for transparency,
not speed — a Python loop over `t` — because an oracle that shares an
optimisation with the thing it checks is not an oracle.

**[DERIVED]** (5.2) is one step of online gradient descent on
`l(S) = (1/2)||S k - v||^2` with `alpha_mix` acting as adaptive weight decay:

```
S - beta ( (alpha S) k - v ) k^T   applied to   alpha S
  = S ( alpha ( I - beta k k^T ) ) + beta v k^T                          (5.4)
```

so the mixer is a test-time learner over `S_mix`, and `alpha_mix` is its
forgetting gate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..utils.shapes import check_shape

__all__ = ["gated_delta_rule_reference", "transition_eigenvalues"]


@torch.no_grad()
def transition_eigenvalues(
    alpha: torch.Tensor, beta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(5.5): the eigenvalues of `alpha (I - beta k k^T)` with `||k|| = 1`.

    SOMA-SPEC: 05 §5.2.2

    ```
    alpha * (1 - beta)   along k          in (-alpha, alpha)
    alpha                orthogonal to k                                 (5.5)
    ```

    An earlier draft omitted `alpha` and claimed eigenvalues `1 - beta` and `1`.
    The exact value `-1` is therefore **not** attainable: a Householder
    reflection is approached only as `alpha -> 1`. Returned as
    `(along_k, orthogonal)` so a test can assert the range rather than trusting
    the algebra.
    """
    return alpha * (1.0 - beta), alpha


def gated_delta_rule_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(5.2)-(5.3), evaluated one position at a time.

    Parameters
    ----------
    q, k : `(B, H, T, d_k)`
        Queries and keys. `k` is expected L2-normalised per position
        (§05.2.4); this function does not renormalise, so a caller passing
        un-normalised keys gets the recurrence it asked for and the eigenvalue
        statement of (5.5) no longer applies.
    v : `(B, H, T, d_v)`
    alpha : `(B, H, T)` in `(0, 1)` -- the forgetting gate.
    beta : `(B, H, T)` in `(0, 2 - 1e-3]` -- the write strength; values above 1
        make the transition eigenvalue along `k` negative, which is the point
        (§05.2.2, Proposition 5.1).
    initial_state : `(B, H, d_v, d_k)`, optional
        `S_mix[-1]`. Zero if omitted.
    valid : `(B, H, T)` bool, optional
        False at padded positions. A padded position writes nothing
        (`beta <- 0`), does not decay the state (`alpha <- 1`) and emits zero
        (§05.2.5: "sequences are padded to a multiple of `C_chunk` and pad
        positions are masked").

    Returns
    -------
    o : `(B, H, T, d_v)`
    final_state : `(B, H, d_v, d_k)`
    """
    b = {}
    check_shape(q, ["B", "H", "T", "dk"], name="q_mix", bindings=b)
    check_shape(k, ["B", "H", "T", "dk"], name="k_mix", bindings=b)
    check_shape(v, ["B", "H", "T", "dv"], name="v_mix", bindings=b)
    check_shape(alpha, ["B", "H", "T"], name="alpha_mix", bindings=b)
    check_shape(beta, ["B", "H", "T"], name="beta_mix", bindings=b)

    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    dtype, device = q.dtype, q.device

    if valid is not None:
        check_shape(valid, ["B", "H", "T"], name="valid", bindings=b)
        alpha = torch.where(valid, alpha, torch.ones_like(alpha))
        beta = torch.where(valid, beta, torch.zeros_like(beta))
        q = q * valid.unsqueeze(-1).to(q.dtype)

    if initial_state is None:
        S = torch.zeros(B, H, d_v, d_k, dtype=dtype, device=device)
    else:
        check_shape(initial_state, ["B", "H", "dv", "dk"], name="S_mix", bindings=b)
        S = initial_state.clone()

    out = torch.empty(B, H, T, d_v, dtype=dtype, device=device)
    for t in range(T):
        k_t = k[:, :, t, :]                       # (B, H, d_k)
        v_t = v[:, :, t, :]                       # (B, H, d_v)
        q_t = q[:, :, t, :]                       # (B, H, d_k)
        a_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)
        b_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)    # (B, H, 1, 1)

        # S ( alpha ( I - beta k k^T ) ) = alpha S - alpha beta (S k) k^T
        Sk = torch.einsum("bhvk,bhk->bhv", S, k_t)         # (B, H, d_v)
        S = a_t * S - a_t * b_t * Sk.unsqueeze(-1) * k_t.unsqueeze(-2)
        # + beta v k^T
        S = S + b_t * v_t.unsqueeze(-1) * k_t.unsqueeze(-2)

        out[:, :, t, :] = torch.einsum("bhvk,bhk->bhv", S, q_t)

    return out, S
