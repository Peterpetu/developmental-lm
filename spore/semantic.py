"""Semantic atoms without gradients: the PPMI-SVD of the bigram counts, lifted to `D`.
SPEC: 01 §1.1 variant 1b (03 §P-S.8).

Levy & Goldberg (NeurIPS 2014, "Neural word embedding as implicit matrix factorization",
Thm in §3) show that skip-gram with negative sampling factorises the shifted PMI matrix; the
truncated SVD of the positive PMI matrix gives embeddings of the same quality in closed form.
Here the counts are the bigram table of the first developmental stage (a `bincount`), the
matrix is `PPMI_{uv} = max(log(N_{uv} N / (N_{u.} N_{.v})), 0)`, and a token's embedding is the
concatenation of its left-context and right-context factors, `[U_d S_d^{1/2} ; V_d S_d^{1/2}]`
(both roles, since the codes bind tokens on either side of a position). The embedding is
lifted to the substrate's `D` by a seeded Gaussian matrix (a Johnson-Lindenstrauss map, so
the cosine geometry survives) and mixed with the random atom, `a'_v = normalise((1 - eta) a_v
+ eta lift(E_v))`, so that `eta = 0` is the pure seeded organism and `eta = 1` the pure
distributional one. No parameter of this module is fit by a gradient.
"""

from __future__ import annotations

import torch

from spore.life import SequentialStream

__all__ = ["bigram_counts", "ppmi_svd", "lift", "semantic_atoms"]


def bigram_counts(shard, *, offset: int, tokens: int, T: int, batch_windows: int, V: int,
                  device: str = "cuda") -> torch.Tensor:
    """`(V, V)` float32 counts of `(v_t, v_{t+1})` over the first `tokens` targets of the stream."""
    stream = SequentialStream(shard, offset=offset, T=T, batch_windows=batch_windows, device=device)
    N = torch.zeros(V * V, dtype=torch.float32, device=device)
    seen = 0
    while seen < tokens:
        w = stream.next()
        prev = w[:, :-1].reshape(-1)
        nxt = w[:, 1:].reshape(-1)
        N += torch.bincount(prev * V + nxt, minlength=V * V).float()
        seen += batch_windows * T
    return N.reshape(V, V)


def ppmi_svd(N: torch.Tensor, d: int, *, eps: float = 1e-8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """`(E (V, 2d), singular values (d,))`: rows of `[U_d S_d^{1/2} ; V_d S_d^{1/2}]` of the PPMI matrix.
    The randomised SVD draws its test matrix from the global RNG, so it runs under a forked,
    seeded RNG state to keep a life deterministic (G-DET; 03 §P-S.14)."""
    if N.dim() != 2 or N.shape[0] != N.shape[1]:
        raise ValueError("N must be square")
    V = N.shape[0]
    if not 1 <= d <= V:
        raise ValueError("1 <= d <= V")
    N = N.double()
    total = N.sum().clamp_min(1.0)
    row = N.sum(1, keepdim=True).clamp_min(eps)
    col = N.sum(0, keepdim=True).clamp_min(eps)
    pmi = torch.log((N * total / (row * col)).clamp_min(eps))
    ppmi = pmi.clamp_min(0.0)
    ppmi[N == 0] = 0.0
    # randomised SVD (Halko, Martinsson & Tropp 2011, Alg. 4.4 with power iterations): the
    # full SVD of a V x V matrix is minutes on this card, the rank-d factorisation seconds
    devices = [ppmi.device] if ppmi.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        U, S, Vt = torch.svd_lowrank(ppmi.float(), q=min(d + 32, V), niter=6)
    s = S[:d].sqrt()
    E = torch.cat([U[:, :d] * s, Vt[:, :d] * s], dim=1)
    return E, S[:d]


def lift(E: torch.Tensor, D: int, seed: int) -> torch.Tensor:
    """A seeded Gaussian map `R^{2d} -> R^D`, rows normalised: `(V, D)` unit vectors."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    R = (torch.randn(E.shape[1], D, generator=g) / E.shape[1] ** 0.5).to(E.device)
    X = E @ R
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-6)


def semantic_atoms(A: torch.Tensor, N: torch.Tensor, *, d: int, eta: float, seed: int) -> torch.Tensor:
    """`a'_v = normalise((1 - eta) a_v + eta lift(E_v))` for the random atoms `A (V, D)`."""
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta in [0, 1]")
    if eta == 0.0:
        return A
    E, _ = ppmi_svd(N, d, seed=seed)
    L = lift(E, A.shape[1], seed)
    X = (1.0 - eta) * A + eta * L
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-6)
