"""The coevolving task archive: `SEED_ARCHIVE` and `ARCHIVE_STEP`.

SPEC: 06 §6.3.3

```
Learn(lambda) = 4 * s_task * ( 1 - s_task )                              (6.6)
Imp(lambda)   = max( 0 , s_task - s_H ) / H_arch                         (6.7)
Pri(lambda)   = Learn(lambda) * ( eta_imp + Imp(lambda) )                (6.8)
```

`Learn` is `0` at `s = 0` (no signal available) and at `s = 1` (nothing left to
learn), maximal at `s = 1/2`, bounded in `[0,1]`, and needs only the verifier.

**[NORMATIVE — `s_task` protocol]** `s_task` is the fraction of `n_eval = 32`
freshly generated instances whose **greedy** continuation (temperature 0,
`max_new` = the answer length, stop at the answer-span terminator) verifies
exactly. No sampling. Spec v1 gave no decode protocol for the archive's central
quantity.

Three v1 defects this module exists to avoid, and one v2 defect:

* *Freeze.* v1 sampled purely proportional to `Pri`. As the model masters the
  archive `s -> 1`, `Learn -> 0`, `Pri -> 0` for every member, no parent can be
  sampled and the archive **freezes permanently**. The `eps_arch` mixture makes
  this state unreachable.
* *Staleness.* v1 refreshed "a random 5%", under which
  `P(unrefreshed for 20 steps) = 0.95^20 = 0.36`, so its own postcondition would
  fail for a third of the archive at all times. Round-robin oldest-first makes
  the bound achievable.
* *Admission.* `tau_admit` is raised from `0.05` to **`0.30`**. At
  `n_eval = 32`, `Learn(1/32) = 0.121 > 0.05`, so the v1 gate admitted any
  generator with 1-31 successes out of 32 — near-vacuous. `0.30` excludes
  `s < 0.08` and `s > 0.92`.
* *Uninitialised.* v2 never created `Lambda` at all, so `ARCHIVE_STEP`'s first
  firing sampled a parent from an empty set and the entire task stream was
  undefined (audit A-24). `SEED_ARCHIVE` fixes this.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from soma.openended.tasks import (
    FAMILIES,
    TaskParams,
    edit,
    family_lower_bounds,
    serialised_length,
)

__all__ = [
    "ArchiveMember",
    "TaskArchive",
    "A_MAX",
    "TAU_ADMIT",
    "EPS_ARCH",
    "ETA_IMP",
    "H_ARCH",
    "R_REF",
    "N_EVAL",
    "learn",
    "improvement",
    "priority",
]

A_MAX = 512
TAU_ADMIT = 0.30
EPS_ARCH = 0.1
ETA_IMP = 0.1
H_ARCH = 5
"""**Archive steps**, not optimizer steps — the collision with §04's deleted `H`
is why the symbol is renamed."""
R_REF = 16
N_EVAL = 32

# §06.3.5 (audit A-30): the holdout covers every parameter the regression suite
# varies, not only `n_len`. v2 held out `n_len` alone, so the MQAR and DYCK bars
# of §08.2 -- which generalise along `P_mqar` and `depth`, not along length --
# were confounded by archive training at the very settings they score.
HOLDOUT = {
    "n_len": (4, 32),
    "P_mqar": (2, 16),
    # **[NORMATIVE — v3.1, audit A-88]** `(2, 8)` *admitted* depth 8, which is
    # exactly the depth §08.2's `DYCK` row scores, so the archive was free to
    # train at the setting the suite tests and the hold-out was a **no-op**.
    # `P_mqar` was genuinely held out — archive `<= 16`, suite at `32` — and
    # `depth` was not. The band is now `(2, 6)` and `EDIT`'s ceiling moves with
    # it, so the suite's depth 8 sits outside as `P_mqar = 32` does.
    "depth": (2, 6),
}


def learn(s: float) -> float:
    """(6.6): `4 s (1 - s)`."""
    return 4.0 * s * (1.0 - s)


def improvement(s: float, s_H: float) -> float:
    """(6.7): `max(0, s - s_H) / H_arch`."""
    return max(0.0, s - s_H) / H_ARCH


def priority(s: float, s_H: float) -> float:
    """(6.8): `Learn * (eta_imp + Imp)`."""
    return learn(s) * (ETA_IMP + improvement(s, s_H))


@dataclass
class ArchiveMember:
    """One `lambda` with its solve-rate history.

    `s_H` is the **oldest** entry in a ring of the last `H_arch` estimates, and
    is initialised to `s_task` at insertion, so `Imp` is defined for a fresh
    member (§06.3.3).

    **[RATIONALE — `Imp` is usually zero, and that is intended]** With
    `A_target = 20` archive steps per lifetime and round-robin refresh of
    `R_ref = 16` members per step, only a member that survives a large fraction
    of the lifetime accumulates `H_arch = 5` distinct estimates. For every other
    member the ring is short, `s_H = s_task`, `Imp = 0`, and `Pri` reduces to
    `eta_imp * Learn`. The improvement term is a bonus for persistently improving
    lineages, not the primary driver.
    """

    params: TaskParams
    ring: Deque[float]
    inserted_at: int
    last_refreshed: int

    @property
    def s_task(self) -> float:
        return self.ring[-1] if self.ring else 0.0

    @property
    def s_H(self) -> float:
        return self.ring[0] if self.ring else 0.0

    @property
    def learn(self) -> float:
        return learn(self.s_task)

    @property
    def priority(self) -> float:
        return priority(self.s_task, self.s_H)

    def observe(self, s: float, step: int) -> None:
        self.ring.append(float(s))
        self.last_refreshed = step


SolveRate = Callable[[TaskParams, int], float]
"""`(params, n_eval) -> s_task`: the greedy-decode protocol of §06.3.3."""


@dataclass
class TaskArchive:
    """`Lambda`, holding at most `A_max = 512` parameterisations."""

    T_ctx: int
    members: Dict[Tuple, ArchiveMember] = field(default_factory=dict)
    n_steps: int = 0
    _refresh_cursor: int = 0
    history: List[Dict[str, object]] = field(default_factory=list)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.members)

    def is_empty(self) -> bool:
        return not self.members

    def seed(self, solve_rate: SolveRate, rng: np.random.Generator) -> None:
        """`SEED_ARCHIVE` of §06.3.3, called once at `LIFETIME` step 5.

        SPEC: 06 §6.3.3

        ```
        for family in {PARITY, MODCOUNT, PERM, DYCK, COPY, MQAR, GRID}:
            lambda        <- family at the lower bound of every EDIT range
            s_task(lambda) <- estimate over n_eval instances
            ring(lambda)   <- [ s_task ]        # so s_H = s_task and Imp = 0
            insert unconditionally
        postcondition: |Lambda| = 7
        ```

        **[RATIONALE]** The seeds are admitted **unconditionally** because
        `tau_admit` is a *learnability* gate on a proposed variation, and at step
        5 the model is at initialisation, where every generator reads
        `s_task ~ 0` and would be rejected — leaving `Lambda` empty forever.
        """
        self.members.clear()
        for params in family_lower_bounds():
            if not self._admissible_shape(params, rng):
                raise ValueError(
                    f"family {params.family} is inadmissible at its own lower "
                    f"bound under T_ctx = {self.T_ctx}; the configuration cannot "
                    f"run the task stream"
                )
            s = float(solve_rate(params, N_EVAL))
            self.members[params.key()] = ArchiveMember(
                params, deque([s], maxlen=H_ARCH), 0, 0
            )
        assert len(self.members) == len(FAMILIES), (
            f"SEED_ARCHIVE produced {len(self.members)} members, expected "
            f"{len(FAMILIES)}"
        )

    # ------------------------------------------------------------------

    def _admissible_shape(self, params: TaskParams, rng: np.random.Generator) -> bool:
        """§06.3.5's three rejection rules, all of them.

        SPEC: 06 §6.3.5

        1. every held-out parameter within its cap (`n_len <= 32`,
           `P_mqar <= 16`, `depth <= 8`);
        2. the serialised `(prompt, answer)` pair within `T_ctx - 2`.

        v2 clipped no parameter against context length: `GRID` at
        `h = w = 8, n_ex = 4` serialises `640`+ symbols against `T_ctx = 128` at
        M-nano, so a large majority of nominally admissible members were
        untrainable and unscoreable, and `s_task` would have been measured on
        truncated prompts (audit A-30).
        """
        for name, (lo, hi) in HOLDOUT.items():
            v = getattr(params, name)
            if not lo <= v <= hi:
                return False
        try:
            length = serialised_length(params, rng)
        except Exception:
            return False
        return length <= self.T_ctx - 2

    def step(
        self, solve_rate: SolveRate, rng: np.random.Generator, n: int
    ) -> Dict[str, object]:
        """`ARCHIVE_STEP` of §06.3.3.

        ```
        1. with probability eps_arch = 0.1 sample lambda uniformly from Lambda;
           otherwise sample proportional to Pri(lambda)
        2. lambda' <- EDIT(lambda)
        3. reject if lambda' already in Lambda, or outside the held-out caps,
           or serialising beyond T_ctx - 2
        4. estimate s_task(lambda') over n_eval instances with the current model
        5. if Learn(lambda') > tau_admit: insert lambda'
        6. if |Lambda| > A_max: evict argmin Pri
        7. refresh s_task for the R_ref = 16 oldest members (round-robin)
        postcondition: |Lambda| <= A_max ; every member's estimate is at most
                       ceil(A_max / R_ref) archive steps old
        ```
        """
        if self.is_empty():
            raise RuntimeError(
                "ARCHIVE_STEP on an empty Lambda; SEED_ARCHIVE must run at "
                "LIFETIME step 5 (§06.3.3, audit A-24)"
            )
        self.n_steps += 1
        keys = list(self.members)
        rec: Dict[str, object] = {"step": n, "archive_step": self.n_steps}

        # 1. epsilon-uniform mixture: as the model masters the archive every
        # Pri -> 0, and a purely proportional sampler would freeze permanently.
        if rng.random() < EPS_ARCH:
            parent_key = keys[int(rng.integers(0, len(keys)))]
            rec["sampled"] = "uniform"
        else:
            pri = np.array([self.members[k].priority for k in keys], dtype=float)
            if pri.sum() <= 0:
                parent_key = keys[int(rng.integers(0, len(keys)))]
                rec["sampled"] = "uniform-fallback"
            else:
                parent_key = keys[int(rng.choice(len(keys), p=pri / pri.sum()))]
                rec["sampled"] = "priority"
        parent = self.members[parent_key]

        # 2-3. propose and screen
        child = edit(parent.params, rng)
        rec["parent"] = parent.params.as_dict()
        if child is None:
            rec["outcome"] = "edit_abandoned"
        elif child.key() in self.members:
            rec["outcome"] = "duplicate"
        elif not self._admissible_shape(child, rng):
            rec["outcome"] = "inadmissible"
        else:
            # 4-5. estimate and admit
            s = float(solve_rate(child, N_EVAL))
            rec["child"] = child.as_dict()
            rec["child_s_task"] = s
            rec["child_learn"] = learn(s)
            if learn(s) > TAU_ADMIT:
                self.members[child.key()] = ArchiveMember(
                    child, deque([s], maxlen=H_ARCH), self.n_steps, self.n_steps
                )
                rec["outcome"] = "inserted"
            else:
                rec["outcome"] = "rejected_learnability"

        # 6. evict
        while len(self.members) > A_MAX:
            worst = min(self.members, key=lambda k: self.members[k].priority)
            self.members.pop(worst)
            rec["evicted"] = rec.get("evicted", 0) + 1  # type: ignore[operator]

        # 7. round-robin refresh of the R_ref oldest
        self._refresh_oldest(solve_rate, min(R_REF, len(self.members)))

        assert len(self.members) <= A_MAX
        rec["size"] = len(self.members)
        rec["mean_s_task"] = float(
            np.mean([m.s_task for m in self.members.values()])
        )
        self.history.append(rec)
        return rec

    def _refresh_oldest(self, solve_rate: SolveRate, k: int) -> None:
        """Round-robin over the `k` least recently refreshed members.

        This is what makes the staleness postcondition achievable: every member
        is refreshed within `ceil(A_max / R_ref)` archive steps, because the
        cursor advances deterministically. v1's "random 5%" left a third of the
        archive stale at all times.
        """
        if k <= 0:
            return
        order = sorted(self.members, key=lambda key: self.members[key].last_refreshed)
        for key in order[:k]:
            m = self.members[key]
            m.observe(float(solve_rate(m.params, N_EVAL)), self.n_steps)

    # ------------------------------------------------------------------

    def mean_solve_rate(self) -> float:
        """Mean `s_task` over the live members — the quantity §06.4's ramp arm
        is matched to.

        The archive's members each hold an `s_task` refreshed round-robin
        (§06.3.3), so this is the trajectory the ramp tracks. An empty archive
        returns `0.0`, which makes the ratchet hold rather than run away on a
        signal that does not exist.
        """
        if not self.members:
            return 0.0
        vals = [m.s_task for m in self.members.values()]
        return float(sum(vals) / len(vals))

    def sample_params(self, rng: np.random.Generator) -> TaskParams:
        """Draw a generator for the training stream, proportional to `Learn`.

        Training samples follow *learnability*, not *priority*: `Pri` includes
        the improvement bonus, which is an archive-curation signal, whereas what
        the model should see is the frontier where `s ~ 1/2`.
        """
        keys = list(self.members)
        w = np.array([max(1e-6, self.members[k].learn) for k in keys])
        return self.members[keys[int(rng.choice(len(keys), p=w / w.sum()))]].params

    def solve_rate_distribution(self) -> List[float]:
        return [m.s_task for m in self.members.values()]

    def summary(self) -> Dict[str, object]:
        s = self.solve_rate_distribution()
        return {
            "size": len(self.members),
            "archive_steps": self.n_steps,
            "s_task_mean": float(np.mean(s)) if s else 0.0,
            "s_task_min": float(np.min(s)) if s else 0.0,
            "s_task_max": float(np.max(s)) if s else 0.0,
            "families": sorted({m.params.family for m in self.members.values()}),
            "mean_priority": float(
                np.mean([m.priority for m in self.members.values()])
            ) if self.members else 0.0,
        }
