# VENDORED from `soma/utils/rng.py` at commit `7d29cc1` (SOMA v3.1).
#
# SPEC: 04 §4.1 -- the numerical cores are already pure functions of
# activations and weights, so they are vendored verbatim below this header
# and `soma/` can be archived without breaking GNOMON. Their SOMA oracle
# tests are vendored with them and must pass unchanged.
"""Per-purpose random streams.

SOMA-SPEC: 02 §2.6.3
SOMA-SPEC: 07 §7.7

One generator per purpose, all seeded from the run seed and all recorded in
`meta.json`. Spec v1 mandated separation by purpose in one bullet and then put
lexicase and archive in a shared generator in the next; sharing a stream makes
two mechanisms' draws depend on each other's call counts, so an unrelated change
to one silently changes the other's trajectory and no run is reproducible under
refactoring.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Tuple

import numpy as np
import torch

__all__ = ["PURPOSES", "RngBundle", "derive_seed", "seed_global_construction"]


PURPOSES: Tuple[str, ...] = (
    "data",
    "init",
    "dropout",
    "mutation",
    "lexicase",
    "archive",
    "es_perturb",
    "member",
    "downsample",
    "growth",
    "explore",
)
"""The exhaustive list of §02.6.3. A purpose not listed is a defect."""


def derive_seed(run_seed: int, purpose: str) -> int:
    """Deterministically derive a 63-bit stream seed from the run seed.

    Uses BLAKE2b rather than arithmetic on the run seed so that neighbouring run
    seeds do not produce correlated streams: `run_seed` and `run_seed + 1` would
    otherwise give `data` streams offset by one, which is exactly the structure
    a seed sweep is meant to avoid.
    """
    if purpose not in PURPOSES:
        raise KeyError(f"unknown RNG purpose {purpose!r}; known: {PURPOSES}")
    h = hashlib.blake2b(
        f"{int(run_seed)}:{purpose}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(h, "big") & ((1 << 63) - 1)


def seed_global_construction(run_seed: int, purpose: str = "init") -> int:
    """Seed torch's **global** generators from the `purpose` stream seed.

    SOMA-SPEC: 02 §2.6.3

    Every other draw in SOMA takes an explicit `Generator`, which is what makes
    the streams independent under refactoring. Module *construction* is the one
    place where that is not possible: `nn.Linear.reset_parameters`,
    `nn.Embedding.reset_parameters` and `nn.init.normal_` have no `generator`
    argument in torch 2.7, so they draw from the process-global default
    generator regardless of what a bundle holds.

    **[NORMATIVE — v3.1, audit A-64]** A run therefore seeds the global
    generators from `derive_seed(run_seed, "init")` immediately before the model
    is constructed. Without it the initial weights came from wherever the
    process's global stream happened to be, and two invocations of the *same*
    lifetime with the *same* seed produced different models — measured, the
    §02.5 acceptance statistics differed between identical invocations
    (`var_ratio` 0.640 vs 0.635, `grad_ratio` 12.739 vs 13.045). Every
    reproducibility claim in §07.7, and every seed-paired ablation in §08.6,
    rests on this call.

    The CUDA generators are seeded too, because a module constructed directly on
    a CUDA device draws from those instead.

    Returns the seed used, so it can be recorded in `meta.json`.
    """
    s = derive_seed(run_seed, purpose)
    # `manual_seed` takes int64; the derived seed is 63-bit, so it fits.
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s


@dataclass
class RngBundle:
    """The full set of per-purpose streams for one run.

    Each purpose owns both a NumPy `Generator` (for the combinatorial machinery:
    lexicase shuffles, reservoir sampling, mutation draws) and a torch
    `Generator` (for tensor-valued draws: initialisation, ES perturbations,
    positional jitter). They are seeded from the same derived seed but advance
    independently, which is intentional: a change in how many torch draws a
    routine makes must not shift the NumPy stream of the same purpose.
    """

    run_seed: int
    device: torch.device
    _np: Dict[str, np.random.Generator]
    _pt: Dict[str, torch.Generator]

    @classmethod
    def create(cls, run_seed: int, device: torch.device | str = "cpu") -> "RngBundle":
        device = torch.device(device)
        np_streams: Dict[str, np.random.Generator] = {}
        pt_streams: Dict[str, torch.Generator] = {}
        for purpose in PURPOSES:
            seed = derive_seed(run_seed, purpose)
            np_streams[purpose] = np.random.default_rng(seed)
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            pt_streams[purpose] = g
        return cls(int(run_seed), device, np_streams, pt_streams)

    def np(self, purpose: str) -> np.random.Generator:
        try:
            return self._np[purpose]
        except KeyError:
            raise KeyError(f"unknown RNG purpose {purpose!r}; known: {PURPOSES}") from None

    def pt(self, purpose: str) -> torch.Generator:
        try:
            return self._pt[purpose]
        except KeyError:
            raise KeyError(f"unknown RNG purpose {purpose!r}; known: {PURPOSES}") from None

    def seed_global(self, purpose: str = "init") -> int:
        """Seed the process-global torch generators from this run's stream.

        See `seed_global_construction`. Called once per lifetime, immediately
        before the model is built.
        """
        if purpose not in PURPOSES:
            raise KeyError(f"unknown RNG purpose {purpose!r}; known: {PURPOSES}")
        return seed_global_construction(self.run_seed, purpose)

    def seeds(self) -> Dict[str, int]:
        """The `seeds` block of `meta.json` (§02.3)."""
        return {p: derive_seed(self.run_seed, p) for p in PURPOSES}

    def state(self) -> Dict[str, object]:
        """Full stream state, for checkpoint/resume equality (§07.7)."""
        return {
            "run_seed": self.run_seed,
            "np": {p: self._np[p].bit_generator.state for p in PURPOSES},
            "pt": {p: self._pt[p].get_state() for p in PURPOSES},
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        if int(state["run_seed"]) != self.run_seed:  # type: ignore[arg-type]
            raise ValueError(
                f"run seed mismatch on resume: checkpoint {state['run_seed']!r} "
                f"vs bundle {self.run_seed!r}"
            )
        np_state = state["np"]  # type: ignore[index]
        pt_state = state["pt"]  # type: ignore[index]
        for p in PURPOSES:
            self._np[p].bit_generator.state = np_state[p]  # type: ignore[index]
            self._pt[p].set_state(pt_state[p])  # type: ignore[index]

    def __iter__(self) -> Iterator[str]:
        return iter(PURPOSES)
