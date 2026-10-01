"""The case reservoir: bounded-memory down-sampling of scored tokens.

SPEC: 04 §4.6

**[NORMATIVE]** The buffer is a **reservoir sample** of fixed size
`q_cases = max(D_min, round(d_rate * B_tok))`, refilled continuously and reset
after each selection event.

Spec v1 said the buffer "holds every scored token from the last `T_sel` steps"
and then down-sampled. At M1 that is `T_sel * B_tok * k_route` credit records —
tens of gigabytes, absent from the memory budget. Reservoir sampling bounds it
to `q_cases * (2 k_route + 2)` floats per Site: **5.1 MB** across all 12 Sites
at M1, which §05.10 records.

Per (case, unit) it stores only scalars: `||delta_ut||_2` and `r_ut`. Per case it
stores the token id and **`w_t` itself**, evaluated at the moment the case was
scored (§04.6 v3.1, audit A-53) — not a context window to be re-evaluated later.
Both reasons matter: `U` is per-Site so a context store is `L` times larger than
v3.0 costed it, and (4.2)'s product `r_ut * ||delta_ut||_2 * w_t` is only
coherent if all three factors come from the same moment.

**[NORMATIVE — the justification, stated correctly]** Down-sampling is justified
because it **allows more individuals to be evaluated per unit of compute**. It
is *not* justified by a "changing environment" effect: the source's own
experiments give evidence **against** that hypothesis and against a
generalisation explanation. We adopt the mechanism together with the source's
own explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

__all__ = ["CaseReservoir", "CaseBatch"]


@dataclass
class CaseBatch:
    """The cases drawn for one selection event."""

    token_id: torch.Tensor
    """`(q,)` int64."""
    slots: torch.Tensor
    """`(q, k_route)` int64; `-1` where the case was routed to fewer units."""
    delta_norm: torch.Tensor
    """`(q, k_route)` float."""
    r_ut: torch.Tensor
    """`(q, k_route)` float."""
    w_t: torch.Tensor
    """`(q,)` float — (4.1), evaluated when the case was scored."""

    def __len__(self) -> int:
        return int(self.token_id.numel())


class CaseReservoir:
    """Uniform reservoir sample over the scored tokens of a selection interval.

    Algorithm R: the `i`-th item (0-based) replaces a uniformly chosen slot with
    probability `capacity / (i + 1)`. Every token offered since the last reset
    has exactly equal inclusion probability, which is the property the
    down-sampling evidence is about — a biased sample would make `eps_t` and the
    case-win histogram statistics of the *sampler* rather than of the
    population.
    """

    def __init__(
        self,
        capacity: int,
        k_route: int,
        *,
        device: torch.device | str = "cpu",
    ):
        if capacity < 1:
            raise ValueError("reservoir capacity must be >= 1")
        self.capacity = int(capacity)
        self.k_route = int(k_route)
        self.device = torch.device(device)

        self._token = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self._slots = torch.full(
            (capacity, k_route), -1, dtype=torch.long, device=self.device
        )
        self._delta = torch.zeros(capacity, k_route, device=self.device)
        self._r = torch.zeros(capacity, k_route, device=self.device)
        self._w = torch.zeros(capacity, device=self.device)
        self._n_seen = 0
        self._n_filled = 0

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_filled

    @property
    def n_seen(self) -> int:
        return self._n_seen

    def reset(self) -> None:
        """Reset after each selection event (§04.6)."""
        self._n_seen = 0
        self._n_filled = 0
        self._slots.fill_(-1)

    def memory_bytes(self) -> int:
        """Realised footprint, for §05.10's table."""
        return (
            self._token.numel() * 8
            + self._slots.numel() * 8
            + self._delta.numel() * 4
            + self._r.numel() * 4
            + self._w.numel() * 4
        )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def push(
        self,
        token_id: torch.Tensor,
        slots: torch.Tensor,
        delta_norm: torch.Tensor,
        r_ut: torch.Tensor,
        w_t: torch.Tensor,
        rng: np.random.Generator,
    ) -> None:
        """Offer one batch of scored tokens.

        `token_id` and `w_t` are `(n,)`; `slots`, `delta_norm`, `r_ut` are
        `(n, k_route)`.
        """
        n = int(token_id.numel())
        if n == 0:
            return
        if slots.shape != (n, self.k_route):
            raise ValueError(
                f"slots must be ({n}, {self.k_route}); got {tuple(slots.shape)}"
            )

        start = 0
        if self._n_filled < self.capacity:
            take = min(self.capacity - self._n_filled, n)
            sl = slice(self._n_filled, self._n_filled + take)
            self._token[sl] = token_id[:take].to(self.device)
            self._slots[sl] = slots[:take].to(self.device)
            self._delta[sl] = delta_norm[:take].to(self.device).float()
            self._r[sl] = r_ut[:take].to(self.device).float()
            self._w[sl] = w_t[:take].to(self.device).float()
            self._n_filled += take
            self._n_seen += take
            start = take

        if start >= n:
            return

        rest = n - start
        seen_before = self._n_seen + np.arange(rest, dtype=np.int64)
        j = rng.integers(0, seen_before + 1)
        accept = j < self.capacity
        self._n_seen += rest
        if not accept.any():
            return
        src = torch.from_numpy(np.nonzero(accept)[0] + start).to(self.device)
        dst = torch.from_numpy(j[accept]).to(self.device)

        self._token[dst] = token_id[src].to(self.device)
        self._slots[dst] = slots[src].to(self.device)
        self._delta[dst] = delta_norm[src].to(self.device).float()
        self._r[dst] = r_ut[src].to(self.device).float()
        self._w[dst] = w_t[src].to(self.device).float()

    @torch.no_grad()
    def sample(
        self, q: Optional[int] = None, rng: Optional[np.random.Generator] = None
    ) -> CaseBatch:
        """Draw the cases for one selection event (§04.2 step 1).

        The reservoir is already a uniform sample of the interval, so a request
        for `q >= len(self)` returns everything; a smaller `q` sub-samples
        without replacement.
        """
        n = self._n_filled
        if n == 0:
            raise ValueError("reservoir is empty; SELECT's precondition is unmet")
        if q is None or q >= n:
            idx = torch.arange(n, device=self.device)
        else:
            if rng is None:
                raise ValueError("sub-sampling the reservoir needs an RNG")
            idx = torch.from_numpy(rng.choice(n, size=q, replace=False)).to(self.device)
        return CaseBatch(
            self._token[idx],
            self._slots[idx],
            self._delta[idx],
            self._r[idx],
            self._w[idx],
        )
