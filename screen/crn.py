"""Common random numbers: the variance reduction that makes small screens work.

Two arms that see *different* data differ by their genome and by their data.
Two arms that see *identical* data in *identical order* differ only by their
genome, and the paired difference has far smaller variance than either arm's
spread around its own mean. Measured here on a free-weight transformer: an
unpaired arm's seed sd was 0.0141 nats while a paired arm's was 0.0024.

The contract is exactly one sentence, and every screen depends on it:

    **arm `i` seed `s` and arm `j` seed `s` consume the same windows in the same
    order.** `seed` selects the initialisation and the pool permutation; it
    never selects *which* windows exist.

This repository already had the idea -- `qd/evolve.py:generation_seeds` pairs
seeds within a generation, and `runs/gnomon/lrsweep-m-pico-n600-b32.json`
records "3 genomes x 2 seeds paired" -- but no module made it reusable, so most
measurements did not get it.
"""

from __future__ import annotations

import hashlib
from typing import Iterator

import numpy as np
import torch

__all__ = ["WindowPool"]


class WindowPool:
    """A fixed, GPU-resident pool of `(T+1)`-token training windows.

    Drawn once, from `pool_seed` alone. Every arm of every screen that shares a
    `pool_seed` shares this exact pool, byte for byte, which is what `digest()`
    is for: a screen report that does not carry the digest cannot be compared
    against another screen's.
    """

    def __init__(self, shard, *, n_windows: int, T: int, pool_seed: int,
                 device: str = "cuda"):
        if n_windows < 1 or T < 1:
            raise ValueError(f"n_windows={n_windows}, T={T} must both be >= 1")
        tokens = shard.tokens
        n_tok = int(tokens.shape[0])
        if n_tok < T + 1:
            raise ValueError(f"shard has {n_tok} tokens, need >= {T + 1}")

        # Offsets are drawn from the pool seed only. Contiguous windows, so a
        # window may straddle a document boundary -- that is the same convention
        # `data/stream.py:TokenStream` uses, and keeping it means a screen and a
        # GNOMON-era run are measuring the same objective.
        rng = np.random.default_rng(pool_seed)
        offs = rng.integers(0, n_tok - T - 1, size=n_windows, dtype=np.int64)
        idx = offs[:, None] + np.arange(T + 1, dtype=np.int64)[None, :]
        win = np.asarray(tokens)[idx].astype(np.int64)

        self.T = T
        self.n_windows = n_windows
        self.pool_seed = pool_seed
        self.device = device
        self.windows = torch.from_numpy(win).to(device)
        self._digest = hashlib.blake2b(
            np.ascontiguousarray(win, dtype=np.int64).tobytes(), digest_size=16
        ).hexdigest()

    def digest(self) -> str:
        return self._digest

    def batches(self, *, batch: int, steps: int, seed: int
                ) -> Iterator[torch.Tensor]:
        """`steps` batches of `(batch, T+1)`, ordered by `seed` and nothing else.

        The permutation is regenerated whenever the pool is exhausted, so a run
        longer than `n_windows / batch` steps wraps into a *different* order
        rather than repeating the first epoch verbatim.
        """
        if batch < 1 or steps < 1:
            raise ValueError(f"batch={batch}, steps={steps} must both be >= 1")
        if batch > self.n_windows:
            raise ValueError(
                f"batch={batch} exceeds pool size {self.n_windows}")
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        order = torch.randperm(self.n_windows, generator=g)
        cursor, epoch = 0, 0
        for _ in range(steps):
            if cursor + batch > self.n_windows:
                epoch += 1
                g = torch.Generator(device="cpu").manual_seed(
                    int(seed) + 1_000_003 * epoch)
                order = torch.randperm(self.n_windows, generator=g)
                cursor = 0
            take = order[cursor:cursor + batch].to(self.device)
            cursor += batch
            yield self.windows.index_select(0, take)


def eval_windows(shard, *, n_windows: int, T: int, device: str = "cuda"
                 ) -> torch.Tensor:
    """A fixed evaluation set: the *head* of a shard, never sampled.

    Deterministic and shared by every arm, so two arms' cross-entropies are
    computed on identical tokens. `data/stream.py:eval_batches` does the same
    thing for the same reason; this is its pool-shaped twin.
    """
    tokens = np.asarray(shard.tokens)
    need = n_windows * (T + 1)
    if tokens.shape[0] < need:
        raise ValueError(f"shard has {tokens.shape[0]} tokens, need {need}")
    win = tokens[:need].reshape(n_windows, T + 1).astype(np.int64)
    return torch.from_numpy(win).to(device)
