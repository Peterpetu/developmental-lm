"""The genome: seeds and dials, never a fit. SPEC: 01 §1.7, §1.8.

A `Genome` is the complete description of an organism's structure; a lifetime is a
deterministic function of `(genome, budget, corpus offset)` (01 §1.7, G-DET). Every gene has
a default, a range and a mutation rule; ranges are enforced at construction so a mutated
genome is always valid.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Sequence

import numpy as np

__all__ = ["Genome", "GENE_RANGES", "mutate", "crossover"]

D_CHOICES = (1024, 2048, 4096, 8192)
DK_CHOICES = (256, 512, 1024, 2048)
NK_CHOICES = (1, 2, 3, 4)
E_CHOICES = (0, 1, 2, 3)
PSI_CHOICES = ("relu", "sign", "cos")
DSEM_CHOICES = (64, 128, 256, 512)
ORDER_UNIVERSE = (1, 2, 3, 4, 5, 6, 7, 8)

GENE_RANGES = {
    "gamma_b": (0.5, 0.999),
    "gamma_e": (0.3, 0.95),
    "beta": (0.05, 1.999),
    "lam": (1e-4, 1.0),
    "eta_ri": (0.0, 2.0),
    "eta_sem": (0.0, 1.0),
    "rho_res": (0.1, 1.2),
}
"""SPEC: 01 §1.7 -- the real-valued genes' ranges."""


@dataclasses.dataclass(frozen=True)
class Genome:
    # seeds (01 (1.1), Definition 2, Definition 7, Definition 10)
    s_A: int = 0
    s_rho: int = 1
    s_M: int = 2
    s_U: int = 3
    # substrate
    D: int = 4096
    orders: tuple = (1, 2, 3, 4, 6)
    gamma_b: float = 0.95
    gamma_e: float = 0.7
    J_e: int = 24                 # truncation of the fading trajectory; gamma_e^J_e < 1e-3 at the default
    # memory
    n_k: int = 2
    beta: float = 1.0
    d_k: int = 1024
    # readout
    lam: float = 0.01
    E: int = 1
    # development
    waves: tuple = (0.1, 0.25, 0.5)
    k_wave: int = 512
    cand_mult: int = 8
    psi: str = "relu"
    k_w: int = 64
    # variants
    eta_ri: float = 0.0
    onehot: bool = False          # the exact order-1 block at D_1 = V (01 §1.2 variant 2a)
    eta_sem: float = 0.0          # semantic atoms: PPMI-SVD of the bigram counts, mixed in (01 §1.1 variant 1b)
    d_sem: int = 256              # rank of the PPMI-SVD
    sem_frac: float = 1.0         # fraction of the budget counted before the atoms are formed
    reservoir: bool = False       # an echo-state block over the atoms (00 §0.4 candidate B; 01 §1.2 variant 2b)
    lro: bool = False             # output-side semantic sharing in the refinement (01 Definition 6b); needs eta_sem > 0
    rho_res: float = 0.9          # its spectral radius

    def __post_init__(self):
        if self.D not in D_CHOICES:
            raise ValueError(f"D must be one of {D_CHOICES}; got {self.D}")
        if self.d_k not in DK_CHOICES:
            raise ValueError(f"d_k must be one of {DK_CHOICES}; got {self.d_k}")
        if self.n_k not in NK_CHOICES:
            raise ValueError(f"n_k must be one of {NK_CHOICES}; got {self.n_k}")
        if self.E not in E_CHOICES:
            raise ValueError(f"E must be one of {E_CHOICES}; got {self.E}")
        if self.d_sem not in DSEM_CHOICES:
            raise ValueError(f"d_sem must be one of {DSEM_CHOICES}; got {self.d_sem}")
        if not 0.0 < self.sem_frac <= 1.0:
            raise ValueError("sem_frac in (0, 1]")
        if self.psi not in PSI_CHOICES:
            raise ValueError(f"psi must be one of {PSI_CHOICES}; got {self.psi}")
        orders = tuple(sorted(set(int(o) for o in self.orders)))
        if not orders or any(o not in ORDER_UNIVERSE for o in orders):
            raise ValueError(f"orders must be a non-empty subset of {ORDER_UNIVERSE}; got {self.orders}")
        object.__setattr__(self, "orders", orders)
        for name, (lo, hi) in GENE_RANGES.items():
            x = float(getattr(self, name))
            if not lo <= x <= hi:
                raise ValueError(f"{name} = {x} outside [{lo}, {hi}]")
        waves = tuple(sorted(float(w) for w in self.waves))
        if any(not 0.0 < w < 1.0 for w in waves):
            raise ValueError("every wave fraction must lie in (0, 1)")
        object.__setattr__(self, "waves", waves)
        if self.k_wave < 0 or self.cand_mult < 1 or self.k_w < 1 or self.J_e < 1:
            raise ValueError("k_wave >= 0, cand_mult >= 1, k_w >= 1, J_e >= 1")
        for s in (self.s_A, self.s_rho, self.s_M, self.s_U):
            if not 0 <= int(s) < 2 ** 31:
                raise ValueError("seeds must lie in [0, 2^31)")

    # -- serialisation ------------------------------------------------------------------
    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["orders"] = list(self.orders)
        d["waves"] = list(self.waves)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        d = dict(d)
        d["orders"] = tuple(d.get("orders", (1, 2, 3, 4, 6)))
        d["waves"] = tuple(d.get("waves", (0.1, 0.25, 0.5)))
        return Genome(**d)

    def digest(self) -> str:
        return hashlib.blake2b(json.dumps(self.as_dict(), sort_keys=True).encode(), digest_size=8).hexdigest()

    def n_features(self, V: int = 8192) -> int:
        """`F` at birth (01 (1.5)): `(|N| + 2) D + d_v + 1` (+ `V` with the exact order-1 block)."""
        return (len(self.orders) + 2) * self.D + self.d_k + 1 + (V if self.onehot else 0)

    def replace(self, **kw) -> "Genome":
        return dataclasses.replace(self, **kw)


def _step_choice(x, choices: Sequence, rng: np.random.Generator):
    i = list(choices).index(x)
    j = min(max(i + int(rng.choice([-1, 1])), 0), len(choices) - 1)
    return choices[j]


def _lognormal(x: float, lo: float, hi: float, rng: np.random.Generator, sigma: float = 0.3) -> float:
    y = x * math.exp(sigma * float(rng.standard_normal()))
    return float(min(max(y, lo), hi))


def mutate(g: Genome, rng: np.random.Generator, *, p: float | None = None) -> Genome:
    """SPEC: 01 Definition 13 -- each gene mutates with probability `p` (default `1/|genes|`):
    seeds are redrawn; integers step by one on their ordered range; reals move log-normally
    with `sigma = 0.3` and are clamped; the order set adds or removes one order; the wave
    schedule moves one wave by a log-normal factor inside (0, 1)."""
    genes = ["s_A", "s_rho", "s_M", "s_U", "D", "orders", "gamma_b", "gamma_e", "n_k", "beta",
             "d_k", "lam", "E", "waves", "k_wave", "psi", "k_w", "eta_ri", "onehot", "eta_sem", "d_sem", "reservoir", "rho_res", "lro"]
    p = 1.0 / len(genes) if p is None else p
    kw = {}
    for name in genes:
        if rng.random() >= p:
            continue
        if name in ("s_A", "s_rho", "s_M", "s_U"):
            kw[name] = int(rng.integers(0, 2 ** 31 - 1))
        elif name == "D":
            kw[name] = _step_choice(g.D, D_CHOICES, rng)
        elif name == "d_k":
            kw[name] = _step_choice(g.d_k, DK_CHOICES, rng)
        elif name == "n_k":
            kw[name] = _step_choice(g.n_k, NK_CHOICES, rng)
        elif name == "E":
            kw[name] = _step_choice(g.E, E_CHOICES, rng)
        elif name == "psi":
            kw[name] = str(rng.choice([c for c in PSI_CHOICES if c != g.psi]))
        elif name == "k_w":
            kw[name] = int(min(max(round(g.k_w * math.exp(0.3 * rng.standard_normal())), 8), 512))
        elif name == "k_wave":
            kw[name] = int(min(max(round(g.k_wave * math.exp(0.3 * rng.standard_normal())), 0), 2048))
        elif name == "orders":
            s = set(g.orders)
            if len(s) > 1 and rng.random() < 0.5:
                s.remove(int(rng.choice(sorted(s))))
            else:
                free = [o for o in ORDER_UNIVERSE if o not in s]
                if free:
                    s.add(int(rng.choice(free)))
            kw[name] = tuple(sorted(s))
        elif name == "onehot":
            kw[name] = not g.onehot
        elif name == "reservoir":
            kw[name] = not g.reservoir
        elif name == "lro":
            kw[name] = not g.lro
        elif name == "d_sem":
            kw[name] = _step_choice(g.d_sem, DSEM_CHOICES, rng)
        elif name == "waves":
            w = list(g.waves)
            if w:
                i = int(rng.integers(0, len(w)))
                w[i] = float(min(max(w[i] * math.exp(0.3 * rng.standard_normal()), 0.02), 0.9))
            kw[name] = tuple(sorted(w))
        else:
            lo, hi = GENE_RANGES[name]
            kw[name] = _lognormal(float(getattr(g, name)), lo, hi, rng)
    return g.replace(**kw)


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    """SPEC: 01 Definition 13 -- uniform crossover, gene by gene."""
    da, db = a.as_dict(), b.as_dict()
    child = {k: (da[k] if rng.random() < 0.5 else db[k]) for k in da}
    return Genome.from_dict(child)
