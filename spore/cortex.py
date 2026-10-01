"""The cortex: units born by residual correlation, frozen thereafter. SPEC: 01 §1.6.

Candidates are seeded sparse directions over the features present at birth; a wave scores
every candidate by the cascade-correlation statistic against the current prediction's
residual (Fahlman & Lebiere 1990, Eq. 1, the sum over outputs taken in the 2-norm) and keeps
the top `k_wave`. The `random` control keeps a uniformly random subset instead.
"""

from __future__ import annotations

import torch

from gnomon.utils.shapes import check_shape

__all__ = ["make_candidates", "activate", "ResidualScorer", "Cortex", "hash_targets"]


def hash_targets(V: int, k: int, seed: int, device: str = "cuda") -> torch.Tensor:
    """`(V, k)` seeded `+-1` hashes of the vocabulary: the random token subsets whose ridge
    predictors become units (01 Definition 11b)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return (torch.randint(0, 2, (V, k), generator=g).float() * 2 - 1).to(device)


def make_candidates(M: int, F_in: int, k_w: int, seed: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """`(W_c (M, F_in), phase (M,))`: `k_w` nonzeros of `+-1/sqrt(k_w)` per row, seeded phases in `[0, 2pi)`."""
    if k_w > F_in:
        raise ValueError("k_w <= F_in")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    W = torch.zeros(M, F_in, dtype=torch.float32)
    idx = torch.stack([torch.randperm(F_in, generator=g)[:k_w] for _ in range(M)])       # (M, k_w)
    sign = (torch.randint(0, 2, (M, k_w), generator=g).float() * 2 - 1) / k_w ** 0.5
    W.scatter_(1, idx, sign)
    phase = torch.rand(M, generator=g) * 2 * torch.pi
    return W.to(device), phase.to(device)


def activate(X: torch.Tensor, W: torch.Tensor, phase: torch.Tensor, psi: str) -> torch.Tensor:
    """`psi(<w_u, x_t>)` for `(n, F_in)` inputs and `(M, F_in)` units -> `(n, M)`."""
    a = X @ W.T
    if psi == "relu":
        return torch.relu(a)
    if psi == "sign":
        return torch.sign(a)
    if psi == "cos":
        return torch.cos(a * (W.shape[1] / max(int((W[0] != 0).sum()), 1)) ** 0.5 + phase)
    raise ValueError(f"unknown psi {psi}")


class ResidualScorer:
    """Accumulates the centred residual correlation of `M` candidates over a scoring pass."""

    def __init__(self, M: int, V: int, device: str = "cuda"):
        self.R = torch.zeros(M, V, dtype=torch.float32, device=device)     # sum_t h_u,t (1_v - p_t)^T
        self.S_h = torch.zeros(M, dtype=torch.float32, device=device)
        self.S_hh = torch.zeros(M, dtype=torch.float32, device=device)
        self.S_r = torch.zeros(V, dtype=torch.float32, device=device)      # sum_t (1_v - p_t)
        self.n = 0

    def add(self, H: torch.Tensor, resid: torch.Tensor) -> None:
        """`H (n, M)` activations, `resid (n, V) = 1_v - p` for the same rows."""
        check_shape(H, ["n", "M"], name="H", bindings={})
        check_shape(resid, ["n", "V"], name="resid", bindings={})
        Hf = H.float()
        self.R.addmm_(Hf.T, resid.float())
        self.S_h += Hf.sum(0)
        self.S_hh += (Hf * Hf).sum(0)
        self.S_r += resid.float().sum(0)
        self.n += int(H.shape[0])

    def scores(self) -> torch.Tensor:
        """`rho_u = ||R_u - hbar_u S_r||_2 / sqrt(var_u n)` (01 Definition 11 step 2)."""
        if self.n == 0:
            raise RuntimeError("no rows scored")
        hbar = self.S_h / self.n
        Rc = self.R - hbar.unsqueeze(1) * self.S_r.unsqueeze(0)
        var = (self.S_hh - self.n * hbar * hbar).clamp_min(1e-12)
        return Rc.norm(dim=1) / (var * self.n).sqrt()


class Cortex:
    """The frozen unit bank, grown wave by wave. `F_in` is the width of the cascade input
    `[codes ; r ; h_prev]` at the moment of each wave; units are stored per wave with their
    own input width so later units may read earlier ones (01 Definition 10)."""

    def __init__(self, psi: str):
        self.psi = psi
        self.waves: list[tuple[torch.Tensor, torch.Tensor]] = []      # (W (k, F_in), phase (k,))

    @property
    def M(self) -> int:
        return sum(int(W.shape[0]) for W, _ in self.waves)

    def add_wave(self, W: torch.Tensor, phase: torch.Tensor) -> None:
        self.waves.append((W.contiguous(), phase.contiguous()))

    def forward_into(self, Z: torch.Tensor, start: int, col: int) -> None:
        """Write the units' activations into `Z[:, col:]` wave by wave, each wave reading the
        cascade input `Z[:, start:col]` (its own `F_in`) -- no copies of the input."""
        for W, phase in self.waves:
            F_in = W.shape[1]
            if col - start != F_in:
                raise RuntimeError(f"cascade input width {col - start} != unit width {F_in}")
            k = W.shape[0]
            Z[:, col:col + k] = activate(Z[:, start:col], W, phase, self.psi)
            col += k

    def forward(self, X0: torch.Tensor) -> torch.Tensor:
        """`X0 (n, F0)`: the base features (codes, retrieval). Returns `h (n, M)`; wave `w`'s
        units read `[X0 ; h_{<w}]` truncated or matched to their own `F_in`."""
        if not self.waves:
            return X0.new_zeros(X0.shape[0], 0)
        hs = []
        X = X0
        for W, phase in self.waves:
            F_in = W.shape[1]
            if X.shape[1] != F_in:
                raise RuntimeError(f"cascade input width {X.shape[1]} != unit width {F_in}")
            h = activate(X, W, phase, self.psi)
            hs.append(h)
            X = torch.cat([X, h], dim=1)
        return torch.cat(hs, dim=1)
