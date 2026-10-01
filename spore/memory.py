"""The in-context memory: the delta rule over seeded projections of the codes.
SPEC: 01 §1.5 (Definitions 7-9).

The one object that learns *inside* a text, and it learns by online least squares. The
projections are seeded and fixed; the state starts at zero in every window; the pair written
at `t` is `(k_{t-1}, v_t)` -- the previous context and the current token -- and the query is
`k_t`, so no target enters the state before it is known.
"""

from __future__ import annotations

import torch

from gnomon.pheno.mixer_chunk import BETA_MAX, gated_delta_rule_chunk
from gnomon.utils.shapes import check_shape

__all__ = ["Memory"]


class Memory:
    def __init__(self, A: torch.Tensor, *, d_k: int, seed: int, beta: float, chunk_size: int = 64):
        V, D = A.shape
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.P_k = (torch.randn(d_k, D, generator=g) / D ** 0.5).to(A.device)      # (d_k, D)
        self.P_v = (torch.randn(d_k, D, generator=g) / D ** 0.5).to(A.device)      # (d_v, D), d_v = d_k
        self.AP = A @ self.P_v.T                                                    # (V, d_v): the cleanup table
        self.d_k = int(d_k)
        self.beta = float(min(beta, BETA_MAX))
        self.chunk = int(chunk_size)

    def retrieve(self, code_k: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """(1.8)-(1.9): `r_t = S_t k_t` for `(B, T, D)` key codes and token atoms; returns `(B, T, d_v)`."""
        check_shape(code_k, ["B", "T", "D"], name="code_k", bindings={})
        check_shape(emb, ["B", "T", "D"], name="emb", bindings={})
        B, T, D = code_k.shape
        k = code_k @ self.P_k.T
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)                        # (B, T, d_k)
        v = emb @ self.P_v.T                                                          # (B, T, d_v)
        k_prev = torch.zeros_like(k)
        k_prev[:, 1:] = k[:, :-1]
        alpha = torch.ones(B, 1, T, device=k.device, dtype=torch.float32)
        beta = torch.full((B, 1, T), self.beta, device=k.device, dtype=torch.float32)
        beta[:, :, 0] = 0.0                                                            # nothing to write at t = 0
        o, _ = gated_delta_rule_chunk(k.unsqueeze(1).float(), k_prev.unsqueeze(1).float(),
                                      v.unsqueeze(1).float(), alpha, beta, chunk_size=self.chunk)
        return o[:, 0]

    def cleanup(self, r: torch.Tensor) -> torch.Tensor:
        """(1.10): `m_t = (P_v A)^T r_t` -> `(..., V)`."""
        return r @ self.AP.T
