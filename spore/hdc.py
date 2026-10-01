"""The hyperdimensional substrate: seeded atoms, the permutation, the context codes.
SPEC: 01 §1.1, §1.2, §1.3.

Everything here is regenerated from seeds and is never fit (00 §0.5 S2). The codes are
causal within a window and are the *same information* a transformer scored on the same
window receives.
"""

from __future__ import annotations

import torch

from gnomon.utils.shapes import check_shape

__all__ = ["atoms", "permutation", "permutation_powers", "apply_power", "ContextCoder", "Reservoir"]


def atoms(V: int, D: int, seed: int, device: str = "cuda") -> torch.Tensor:
    """(1.1): `A` as `(V, D)` with unit rows, `g_v ~ N(0, I_D)` from `PRNG(seed)`.

    Rows rather than columns because token lookup is a row gather."""
    if V < 1 or D < 1:
        raise ValueError("V, D >= 1")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(V, D, generator=g, dtype=torch.float32)
    A = A / A.norm(dim=1, keepdim=True)
    return A.to(device)


def permutation(D: int, seed: int, device: str = "cuda") -> torch.Tensor:
    """Definition 2: a coordinate permutation `pi` of `{0..D-1}` from `PRNG(seed)`;
    `(rho x)_i = x_{pi(i)}`."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(int(D), generator=g).to(device)


def permutation_powers(perm: torch.Tensor, J: int) -> torch.Tensor:
    """`(J, D)` index tensor: row `j` is `pi^j` (row 0 the identity), so that
    `apply_power(x, powers, j) == rho^j(x)`. `pi^{j+1} = pi^j o pi`, i.e. `perm_j[perm]`."""
    D = perm.numel()
    out = torch.empty(J, D, dtype=torch.long, device=perm.device)
    cur = torch.arange(D, device=perm.device)
    for j in range(J):
        out[j] = cur
        cur = cur[perm]
    return out


def apply_power(x: torch.Tensor, powers: torch.Tensor, j: int) -> torch.Tensor:
    """`rho^j(x)` on the last axis."""
    return x[..., powers[j]]


class ContextCoder:
    """The codes of Definition 3 for a batch of windows. SPEC: 01 §1.2.

    `orders`: the exact n-gram orders `N`; `gamma_b`: the bag decay; `gamma_e`, `J_e`: the
    fading trajectory's decay and truncation. `binding=False` sets `rho = I` (the K-HDC arm).
    """

    def __init__(self, A: torch.Tensor, perm: torch.Tensor, *, orders, gamma_b: float,
                 gamma_e: float, J_e: int, binding: bool = True):
        self.A = A
        self.V, self.D = A.shape
        self.orders = tuple(sorted(set(int(o) for o in orders)))
        self.gamma_b = float(gamma_b)
        self.gamma_e = float(gamma_e)
        self.J_e = int(J_e)
        self.binding = bool(binding)
        J = max(max(self.orders), self.J_e)
        if self.binding:
            self.powers = permutation_powers(perm, J)
        else:
            self.powers = torch.arange(self.D, device=A.device).unsqueeze(0).expand(J, self.D).contiguous()
        self._decay_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    @property
    def n_blocks(self) -> int:
        return len(self.orders) + 2

    def _decay_matrix(self, T: int, device) -> torch.Tensor:
        key = (T, device)
        if key not in self._decay_cache:
            t = torch.arange(T, device=device, dtype=torch.float32)
            lag = t.unsqueeze(1) - t.unsqueeze(0)                          # [t, s] = t - s
            M = torch.where(lag >= 0, self.gamma_b ** lag.clamp_min(0), torch.zeros_like(lag))
            self._decay_cache[key] = M
        return self._decay_cache[key]

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """`(B, T) -> (B, T, D)`: the atoms of the input tokens."""
        check_shape(tokens, ["B", "T"], name="tokens", bindings={})
        return self.A[tokens]

    def codes(self, tokens: torch.Tensor) -> dict:
        """All codes at every position `t` of `(B, T)` input tokens, each `(B, T, D)`:
        `c{n}` for `n in orders` (1.2), `b` (1.3, post-normalised fading sum), `e` (1.4,
        truncated at `J_e` lags)."""
        emb = self.embed(tokens)
        B, T, D = emb.shape
        acc = {f"c{n}": torch.zeros_like(emb) for n in self.orders}
        e = torch.zeros_like(emb)
        J = max(max(self.orders), self.J_e)
        for j in range(min(J, T)):                     # lags beyond the window contribute nothing
            if j == 0:
                L = emb
            else:
                L = torch.zeros_like(emb)
                L[:, j:] = emb[:, :T - j]
            Lj = apply_power(L, self.powers, j)
            for n in self.orders:
                if j < n:
                    acc[f"c{n}"] += Lj
            if j < self.J_e:
                e += (self.gamma_e ** j) * Lj
        out = {}
        for n in self.orders:
            a = acc[f"c{n}"]
            out[f"c{n}"] = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # the bag: sum_{s<=t} gamma_b^{t-s} emb_s, then normalised
        M = self._decay_matrix(T, emb.device)
        b = torch.einsum("ts,bsd->btd", M, emb)
        out["b"] = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        out["e"] = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        out["emb"] = emb
        return out


class Reservoir:
    """An echo-state block (Jaeger 2001): `x_t = tanh(W_res x_{t-1} + W_in a_{v_t})`, seeded,
    `W_res` scaled to spectral radius `rho`, `x_0 = 0`, reset at every window. Variant 2b:
    implicit recurrence against the explicit binding of Definition 3."""

    def __init__(self, D: int, seed: int, rho: float, device: str = "cuda"):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        W = torch.randn(D, D, generator=g) / D ** 0.5
        ev = torch.linalg.eigvals(W).abs().max()
        self.W_res = (W * (rho / float(ev))).to(device)
        self.W_in = (torch.randn(D, D, generator=g) / D ** 0.5).to(device)
        self.D = D

    def run(self, emb: torch.Tensor) -> torch.Tensor:
        """`(B, T, D)` atoms -> `(B, T, D)` states, sequentially over `T`."""
        B, T, D = emb.shape
        u = emb @ self.W_in.T
        x = emb.new_zeros(B, D)
        out = torch.empty_like(emb)
        for t in range(T):
            x = torch.tanh(x @ self.W_res.T + u[:, t])
            out[:, t] = x
        return out
