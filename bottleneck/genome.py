"""The paper's g-networks on a language model's weight matrices.

SPEC: docs/bottleneck/00-replication.md §1 (the g-network), §2 (codes), §4 (compression)

Object implemented. For every weight `W ∈ R^{n_out × n_in}` of a `Phenotype`, one MLP
`g_θ : {−1,+1}^{b_row + b_col} → R` with `Ŵ_ij = g_θ([c_row(i); c_col(j)])` (1.1); for every
gain vector `w ∈ R^{n}` one MLP on the row code alone; `LOGIT_SCALE` copied through. Codes are
fixed functions of index (§2): Gray over frequency rank on the token axis, a plain binary
index elsewhere. Nothing here is on the gradient path of the phenotype's loss -- the genome is
fitted to weights by regression (§3 step b) and read by decoding (§3 step c), which is the
paper's method and the thing this repository's earlier attempts did not do (11 §P-D.44).
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn

from anlage.config import (AX_TOK, MATRIX_ROLES, ROLE_AXES, ROLE_LOGIT_SCALE, ROLE_NAMES,
                           VECTOR_ROLES)
from anlage.model import Phenotype
from gnomon.genome.codes import bits_to_pm1
from gnomon.utils.shapes import check_shape

__all__ = ["n_bits", "index_codes", "gray_rank_codes", "truncated_svd", "PairMLP", "BottleneckGenome",
           "UNCOMPRESSED_ROLES", "ACTIVATIONS"]

UNCOMPRESSED_ROLES = (ROLE_LOGIT_SCALE,)
"""SPEC: §1 -- copied through, by the paper's rule for objects smaller than their g-network."""

ACTIVATIONS = {"relu": nn.ReLU, "leaky": lambda: nn.LeakyReLU(0.01)}
"""SPEC: §1 (ReLU, the paper's unstated choice) and §11.5 (LeakyReLU, slope 0.01, Maas et al.
2013 -- the standard remedy for dead units)."""


def n_bits(n: int) -> int:
    """`⌈log₂ n⌉` for `n ≥ 2`, and 1 for `n = 1`: the paper's "`log₂ N` binary tags" (p. 9)."""
    if n < 1:
        raise ValueError("an axis has at least one unit")
    return max(1, (n - 1).bit_length())


def index_codes(n: int, device="cpu") -> torch.Tensor:
    """`(n, b)` in `{−1,+1}`: the plain binary index of each unit. SPEC: §2 (axes without a
    topology -- the paper's hidden-unit case)."""
    return bits_to_pm1(torch.arange(n, device=device), n_bits(n)).float()


@torch.no_grad()
def truncated_svd(W: torch.Tensor, r: int) -> torch.Tensor:
    """The best rank-`r` approximation of `W` in Frobenius norm, `U_r Σ_r V_rᵀ` (Eckart-Young-
    Mirsky), in float32 on `W`'s device. SPEC: §11.6."""
    if W.dim() != 2 or r < 1 or r > min(W.shape):
        raise ValueError("a matrix and 1 <= r <= min(shape)")
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


def gray_rank_codes(rank: torch.Tensor, device="cpu") -> torch.Tensor:
    """`(V, b)` in `{−1,+1}`: Gray code over frequency rank on `b = ⌈log₂ V⌉` bits. SPEC: §2.

    `rank[v] = 0` for the most frequent type. `q_v = round(rank · (2^b − 1) / (V − 1))`, then the
    Gray map `q XOR (q >> 1)`, so adjacent ranks differ in one bit (I2)."""
    check_shape(rank, [None], name="rank")
    V = int(rank.numel())
    b = n_bits(V)
    hi = (1 << b) - 1
    q = torch.round(rank.to(torch.float64) * hi / max(V - 1, 1)).to(torch.int64)
    return bits_to_pm1((q ^ (q >> 1)).to(device), b).float()


class PairMLP(nn.Module):
    """One g-network: `Linear(b, H) ∘ ReLU ∘ Linear(H, H₂) ∘ ReLU ∘ Linear(H₂, 1)`. SPEC: (1.1).

    The paper's `(20 − H − 10 − 1)` (Table 1) with the input width set by the axes. The
    activation is the paper's unstated choice, taken as ReLU. The last layer is initialised at
    `0.01×` its default scale so an untrained genome decodes to small weights rather than `O(1)`
    (§1, recorded); at generation 0 the decode is unused (`ε = 1`), so this is conditioning only.
    """

    def __init__(self, b: int, H: int, H2: int, *, seed: int, activation: str = "relu"):
        super().__init__()
        if b < 1 or H < 1 or H2 < 1:
            raise ValueError("widths must be >= 1")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation is one of {sorted(ACTIVATIONS)}")
        self.b, self.H, self.H2, self.activation = int(b), int(H), int(H2), activation
        # §11.5: `leaky` keeps a gradient path through an inactive hidden unit; at H = 419 a
        # quarter of the ReLU g-networks died in their second layer at generation 1 (11 §P-D.50)
        self.net = nn.Sequential(nn.Linear(b, H), ACTIVATIONS[activation](), nn.Linear(H, H2),
                                 ACTIVATIONS[activation](), nn.Linear(H2, 1))
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            for lin in (self.net[0], self.net[2], self.net[4]):
                bound = 1.0 / math.sqrt(lin.in_features)               # nn.Linear's default range
                lin.weight.copy_(torch.empty_like(lin.weight).uniform_(-bound, bound, generator=g))
                lin.bias.copy_(torch.empty_like(lin.bias).uniform_(-bound, bound, generator=g))
            self.net[4].weight.mul_(0.01)
            self.net[4].bias.mul_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_shape(x, [None, self.b], name="pair codes")
        return self.net(x).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BottleneckGenome(nn.Module):
    """Every g-network of one phenotype, with the code tables they read. SPEC: §1, §2, §4.

    `keys` are the phenotype's `(role, layer)` pairs. Matrix roles get a two-axis g-network on
    `[c_row; c_col]`; the gain vectors a one-axis g-network on `c_row`; `UNCOMPRESSED_ROLES` are
    copied through. Codes are built once per `(axis, n_units)` so every matrix sharing an axis
    reads the same code for the same unit.
    """

    def __init__(self, model: Phenotype, rank: torch.Tensor, *, H: int, H2: int = 10,
                 H2_vector: int = 5, seed: int = 0, device="cpu", keep: Iterable[int] = (),
                 activation: str = "relu", rank_roles: Optional[dict] = None):
        super().__init__()
        self.H, self.H2, self.H2_vector = int(H), int(H2), int(H2_vector)
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation is one of {sorted(ACTIVATIONS)}")
        self.activation = activation
        # §11.6: roles carried as a rank-r factorisation -- the closed-form regression of
        # Algorithm 1's step (b) under a rank constraint (Eckart-Young), r(n_out + n_in)
        # parameters counted. A content code for the one axis without a topology (11 §P-D.49).
        self.rank_roles: dict[int, int] = {int(r): int(k) for r, k in (rank_roles or {}).items()}
        for r, k in self.rank_roles.items():
            if r not in ROLE_NAMES or r not in MATRIX_ROLES:
                raise KeyError(f"rank_roles: {r} is not a matrix role")
            if k < 1:
                raise ValueError("a rank is >= 1")
        # §11.4: roles kept uncompressed beyond the paper's rule -- copied through the decode and
        # COUNTED in the genome's size, so the ratio stays honest (the ablation of 11 §P-D.49
        # measured the vocabulary table alone at +4 nats: a coordinate axis with no topology)
        self.keep = tuple(sorted({int(r) for r in keep}))
        for r in self.keep:
            if r not in ROLE_NAMES:
                raise KeyError(f"keep: unknown role {r}")
        self.uncompressed = frozenset(UNCOMPRESSED_ROLES) | frozenset(self.keep)
        if self.uncompressed & set(self.rank_roles):
            raise ValueError("a role is either kept or rank-coded, not both")
        self.no_gnet = self.uncompressed | frozenset(self.rank_roles)
        self.device = torch.device(device)
        self.vocab = int(model.vocab)
        self._codes: dict[tuple[str, int], torch.Tensor] = {}
        self._rank = rank.detach().to("cpu")
        self.keys: list[tuple[int, int]] = []
        self.shapes: dict[tuple[int, int], tuple[int, ...]] = {}
        self.nets = nn.ModuleDict()
        for k, (key, p) in enumerate(model.weights().items()):
            role, l = key
            self.keys.append(key)
            self.shapes[key] = tuple(p.shape)
            if role in self.rank_roles:
                if p.dim() != 2 or self.rank_roles[role] > min(p.shape):
                    raise ValueError(f"{ROLE_NAMES[role]}: rank {self.rank_roles[role]} exceeds the matrix {tuple(p.shape)}")
            if role in self.no_gnet:
                continue
            row_axis, col_axis = ROLE_AXES[role]
            if role in MATRIX_ROLES:
                if p.dim() != 2 or col_axis is None:
                    raise ValueError(f"{ROLE_NAMES[role]}: expected a matrix role with two axes")
                b = n_bits(p.shape[0]) + n_bits(p.shape[1])
                self.nets[self._name(key)] = PairMLP(b, self.H, self.H2, seed=seed * 1009 + k, activation=activation)
            elif role in VECTOR_ROLES:
                if p.dim() != 1:
                    raise ValueError(f"{ROLE_NAMES[role]}: expected a vector role")
                b = n_bits(p.shape[0])
                self.nets[self._name(key)] = PairMLP(b, self.H2_vector, self.H2_vector, seed=seed * 1009 + k,
                                                     activation=activation)
            else:
                raise KeyError(f"role {role} is neither a matrix nor a vector role")
            self._axis_codes(row_axis, p.shape[0])
            if col_axis is not None:
                self._axis_codes(col_axis, p.shape[1])
        self.to(self.device)

    # ------------------------------------------------------------------ codes
    @staticmethod
    def _name(key: tuple[int, int]) -> str:
        return f"{ROLE_NAMES[key[0]]}_{key[1]}"

    def _axis_codes(self, axis: str, n: int) -> torch.Tensor:
        """The `(n, b)` code table of `axis` at `n` units, built once. SPEC: §2."""
        k = (axis, int(n))
        if k not in self._codes:
            if axis == AX_TOK:
                if n != self.vocab:
                    raise ValueError(f"token axis has {self.vocab} units, asked {n}")
                self._codes[k] = gray_rank_codes(self._rank, self.device)
            else:
                self._codes[k] = index_codes(n, self.device)
        return self._codes[k]

    def codes_for(self, key: tuple[int, int]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        role = key[0]
        row_axis, col_axis = ROLE_AXES[role]
        shape = self.shapes[key]
        rows = self._axis_codes(row_axis, shape[0])
        cols = self._axis_codes(col_axis, shape[1]) if (col_axis is not None and len(shape) == 2) else None
        return rows, cols

    def compressed_keys(self) -> list[tuple[int, int]]:
        """The keys with a g-network."""
        return [k for k in self.keys if k[0] not in self.no_gnet]

    def rank_keys(self) -> list[tuple[int, int]]:
        """The keys carried as rank-r factorisations (§11.6)."""
        return [k for k in self.keys if k[0] in self.rank_roles]

    def decoded_keys(self) -> list[tuple[int, int]]:
        """Every key whose decode is not a copy: g-network keys and rank keys."""
        return [k for k in self.keys if k[0] not in self.uncompressed]

    # ----------------------------------------------------------------- decode
    def _inputs(self, key: tuple[int, int], flat: torch.Tensor) -> torch.Tensor:
        """`(len(flat), b)` pair codes for flat entry indices of `key`'s tensor."""
        rows, cols = self.codes_for(key)
        if cols is None:
            return rows[flat]
        n_in = self.shapes[key][1]
        return torch.cat([rows[flat // n_in], cols[flat % n_in]], dim=-1)

    @torch.no_grad()
    def decode(self, key: tuple[int, int], *, chunk: int = 1 << 18) -> torch.Tensor:
        """`Ŵ` for one key, in the tensor's shape, decoded `chunk` entries at a time. SPEC: §3(c)."""
        shape = self.shapes[key]
        N = math.prod(shape)
        out = torch.empty(N, device=self.device, dtype=torch.float32)
        net = self.nets[self._name(key)]
        for a in range(0, N, chunk):
            flat = torch.arange(a, min(a + chunk, N), device=self.device)
            out[a:a + chunk] = net(self._inputs(key, flat))
        return out.view(*shape)

    @torch.no_grad()
    def decode_all(self, W_ref: dict, *, chunk: int = 1 << 18) -> dict:
        """`{key: Ŵ}` over every key; uncompressed roles (the paper's rule and `keep`) are copied
        from `W_ref`."""
        out = {}
        for key in self.keys:
            if key[0] in self.uncompressed:
                out[key] = W_ref[key].detach().clone().float()
            elif key[0] in self.rank_roles:
                out[key] = truncated_svd(W_ref[key].detach(), self.rank_roles[key[0]])
            else:
                out[key] = self.decode(key, chunk=chunk)
        return out

    # -------------------------------------------------------------------- fit
    def sample_entries(self, key: tuple[int, int], N: int, k: int, generator: torch.Generator,
                       weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`k` flat entry indices of `key`'s tensor, drawn with replacement: uniformly (Algorithm
        1's sampler) or in proportion to `weights` (§11.1 -- the Fisher-weighted objective by
        importance sampling, so the same code path minimises (11.1) with the unweighted MSE on
        the sample). Drawn on the CPU generator so every arm shares the sampler and its seed."""
        if weights is None:
            flat = torch.randint(0, N, (k,), generator=generator, device="cpu")
        else:
            if weights.dim() != 1 or weights.numel() != N:
                raise ValueError(f"{self._name(key)}: weights must be flat over the tensor's {N} entries")
            flat = torch.multinomial(weights.float(), k, replacement=True, generator=generator)
        return flat.to(self.device)

    def fit_step(self, W: dict, *, fraction: float, cap: int, generator: torch.Generator,
                 opt: torch.optim.Optimizer, weights: Optional[dict] = None) -> dict[tuple[int, int], float]:
        """One Adam step of MSE regression for every g-network at once, each on a fresh random
        subset of `min(fraction · N, cap)` of its tensor's entries. SPEC: §3(b); with `weights`
        (flat, per compressed key, on the CPU) the entries are drawn in proportion to them and
        the step minimises the weighted objective of (11.1) in expectation.

        The losses are summed; the g-networks have disjoint parameters and Adam is
        per-parameter, so this is exactly independent regression of each g-network with one
        optimiser, which is Algorithm 1's `g_optimizer` shared across layers.
        """
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction in (0, 1]")
        opt.zero_grad(set_to_none=True)
        losses: dict[tuple[int, int], float] = {}
        total = None
        for key in self.compressed_keys():
            w = W[key].detach().float().reshape(-1)
            N = w.numel()
            k = max(1, min(int(fraction * N), int(cap)))
            flat = self.sample_entries(key, N, k, generator, None if weights is None else weights[key])
            pred = self.nets[self._name(key)](self._inputs(key, flat))
            loss = torch.mean((pred - w[flat]) ** 2)
            losses[key] = float(loss)
            total = loss if total is None else total + loss
        if total is not None:
            total.backward()
            opt.step()
        return losses

    @torch.no_grad()
    def residuals(self, W: dict, D: dict) -> dict[tuple[int, int], float]:
        """Relative residual `‖Ŵ − W‖² / ‖W‖²` per decoded key (g-network and rank keys). SPEC: §5."""
        out = {}
        for key in self.decoded_keys():
            w = W[key].detach().float()
            d = D[key].float()
            out[key] = float(((d - w) ** 2).sum() / (w ** 2).sum().clamp_min(1e-30))
        return out

    @torch.no_grad()
    def weighted_residuals(self, W: dict, D: dict, weights: dict) -> dict[tuple[int, int], float]:
        """`Σ F̃ (Ŵ − W)² / Σ F̃ W²` per compressed key: the residual in the metric of (11.1),
        which is what a Fisher-weighted refit minimises and what §11.3's alignment reading
        correlates with the decode's CE. With uniform weights it equals `residuals`."""
        out = {}
        for key in self.compressed_keys():
            w = W[key].detach().float().reshape(-1)
            d = D[key].float().reshape(-1)
            f = weights[key].to(w.device).float()
            if f.numel() != w.numel():
                raise ValueError(f"{self._name(key)}: weights must be flat over the tensor's entries")
            out[key] = float((f * (d - w) ** 2).sum() / (f * w ** 2).sum().clamp_min(1e-30))
        return out

    # ------------------------------------------------------------- accounting
    def n_params(self) -> int:
        """Genome parameters: every g-network, every uncompressed entry (the paper's scalar and
        any kept role alike) and `r (n_out + n_in)` per rank-coded key. SPEC: §4, §11.4, §11.6."""
        uncompressed = sum(math.prod(self.shapes[k]) for k in self.keys if k[0] in self.uncompressed)
        return sum(p.numel() for p in self.parameters()) + uncompressed + self.n_params_rank()

    def n_params_rank(self) -> int:
        return int(sum(self.rank_roles[k[0]] * (self.shapes[k][0] + self.shapes[k][1]) for k in self.rank_keys()))

    def compression_ratio(self, model: Phenotype) -> float:
        """`n_params(phenotype) / n_params(genome)`, the paper's definition (Tables 1-2)."""
        return sum(p.numel() for p in model.weights().values()) / self.n_params()

    def summary(self) -> dict:
        return {"H": self.H, "H2": self.H2, "H2_vector": self.H2_vector, "activation": self.activation,
                "keep": [ROLE_NAMES[r] for r in self.keep], "keep_roles": list(self.keep),
                "rank": {ROLE_NAMES[r]: k for r, k in self.rank_roles.items()},
                "rank_roles": [[r, k] for r, k in sorted(self.rank_roles.items())],
                "n_params_rank": self.n_params_rank(),
                "n_params_uncompressed": int(sum(math.prod(self.shapes[k]) for k in self.keys if k[0] in self.uncompressed)),
                "n_gnets": len(self.nets), "n_params_genome": self.n_params(),
                "bits": {self._name(k): (self.nets[self._name(k)].b) for k in self.compressed_keys()}}


def iter_matrix_keys(genome: BottleneckGenome) -> Iterable[tuple[int, int]]:
    return (k for k in genome.compressed_keys() if k[0] in MATRIX_ROLES)
