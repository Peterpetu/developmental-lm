"""MAP-Elites over genomes.

SPEC: 08 §8.1
SPEC: 08 §8.2
SPEC: 08 §8.3
SPEC: 08 §8.4

Mouret & Clune, *Illuminating search spaces by mapping elites*, arXiv:1504.04909
(2015); Cully, Clune, Tarapore & Mouret, Nature 521:503-507 (2015).

C4 of the constitution, and the only thing that improves `Gamma`. The archive
*is* the size-quality frontier: §07.4 makes decoded parameter count a behaviour
descriptor rather than a constraint, so each niche's elite is the best
architecture at that size and the frontier is the result, not a single number.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import math
import pathlib
from typing import Callable, Iterable, Optional, Sequence

import torch

from ..utils.report import finite_or_none

__all__ = ["Niche", "Member", "Archive", "bin_index"]


def bin_index(value: float, edges: Sequence[float]) -> int:
    """Which bin `value` falls in. SPEC: 08 §8.6 E2.

    Bins are closed-open `[e_i, e_{i+1})` except the top, which is closed, so a
    genome at exactly the upper limit lands in the last bin rather than off the
    end of the map.
    """
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return len(edges) - 2
    return bisect.bisect_right(edges, value) - 1


Niche = tuple[int, int]


@dataclasses.dataclass
class Member:
    """One elite: the genome, its fitness, and everything needed to report it."""

    state: dict                      # {tensor name -> cpu tensor}, Gamma_w + Gamma_a
    fitness: float                   # mean held-out CE over the seed set (lower is better)
    descriptors: tuple[float, float]
    niche: Niche
    generation: int
    n_decoded: int
    arch_summary: str
    per_seed: list[float]
    birth: float = float("nan")
    erank: float = float("nan")
    #: SPEC: 08 §8.5, SPEC: 01 §1.7 -- every artifact of the Lamarckian arm is
    #: labelled, including a genome sitting in an archive somebody reloads a
    #: month later with no memory of which arm produced it.
    lamarck: bool = False

    def clone_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.state.items()}


class Archive:
    """A 2-D MAP-Elites grid over `(attention fraction, log10 decoded params)`."""

    def __init__(self, *, b1_edges: Sequence[float], b2_edges: Sequence[float],
                 L_null: float):
        self.b1_edges = list(b1_edges)
        self.b2_edges = list(b2_edges)
        self.L_null = float(L_null)
        self.cells: dict[Niche, Member] = {}
        self.n_attempts = 0
        self.n_inserts = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ shape

    @property
    def n_niches(self) -> int:
        return (len(self.b1_edges) - 1) * (len(self.b2_edges) - 1)

    def niche_of(self, descriptors: tuple[float, float]) -> Niche:
        b1, b2 = descriptors
        return (bin_index(b1, self.b1_edges), bin_index(b2, self.b2_edges))

    # ------------------------------------------------------------- insertion

    def insert(self, member: Member) -> bool:
        """Elitist, per niche. SPEC: 08 §8.2.

        Replaces only if strictly better, so archive quality is monotone in
        every niche and `qd_score` is monotone overall -- which makes a *falling*
        QD score a bug rather than a fluctuation, a useful property for a long
        run to have.
        """
        self.n_attempts += 1
        n = member.niche
        cur = self.cells.get(n)
        if cur is None or member.fitness < cur.fitness:
            self.cells[n] = member
            self.n_inserts += 1
            return True
        return False

    # ------------------------------------------------------------ statistics

    def coverage(self) -> float:
        """Filled niches over total. SPEC: 08 §8.6 E8 -- filled, never attempted."""
        return len(self.cells) / self.n_niches

    def qd_score(self) -> float:
        """`sum over filled niches of (L_null - L_elite)`. SPEC: 08 §8.4.

        Non-negative, and an empty niche contributes exactly zero, which is what
        makes it comparable across runs with different coverage.
        """
        return sum(max(0.0, self.L_null - m.fitness) for m in self.cells.values())

    def best(self) -> Optional[Member]:
        return min(self.cells.values(), key=lambda m: m.fitness, default=None)

    def mean_elite(self) -> float:
        if not self.cells:
            return float("nan")
        return sum(m.fitness for m in self.cells.values()) / len(self.cells)

    def mean_erank(self) -> float:
        """Mean decoded `erank/min(d)` over the filled niches. SPEC: 07 §7.4.

        A diagnostic rather than a descriptor: it is a property of `Gamma_w`,
        not of the architecture. It is averaged over the *archive's elites*
        because that is the population `KG-RANK` asks about -- a single elite's
        rank can move for reasons that are not degeneration.
        """
        vals = [m.erank for m in self.cells.values()
                if m.erank == m.erank]                 # drop NaN
        return sum(vals) / len(vals) if vals else float("nan")

    def mean_birth(self) -> float:
        """Mean at-birth CE over the filled niches. SPEC: 06 §6.4."""
        vals = [m.birth for m in self.cells.values() if m.birth == m.birth]
        return sum(vals) / len(vals) if vals else float("nan")

    #: A history value, or `None` where there is none. SPEC: 08 §8.4.
    #: Not `NaN`: `json.dumps` writes a bare `NaN`, which is not valid JSON,
    #: and `NaN != NaN` means a row containing one does not compare equal to
    #: itself -- so neither E4's determinism nor E7's round-trip can be
    #: asserted over a history that has one in it.
    _num = staticmethod(finite_or_none)

    def record(self, generation: int) -> dict:
        b = self.best()
        row = {
            "generation": generation,
            "coverage": self.coverage(),
            "n_filled": len(self.cells),
            "qd_score": self.qd_score(),
            "best": self._num(b.fitness if b else None),
            "mean_elite": self._num(self.mean_elite()),
            "n_attempts": self.n_attempts,
            "n_inserts": self.n_inserts,
            "best_arch": b.arch_summary if b else "",
            "best_n_decoded": b.n_decoded if b else 0,
            # SPEC: 07 §7.4 -- `KG-RANK` fires if the archive's elites show
            # decoded effective rank collapsing, and it is a trend test over
            # generations. Until v1.1 `erank` was computed for every `Member`
            # and never written to the history, so the series the criterion is
            # defined over did not exist and `KG-RANK` could not be evaluated
            # from a completed run at all. Same for `birth`, which §06.4
            # requires reported as a trajectory across generations.
            "mean_erank": self._num(self.mean_erank()),
            "best_erank": self._num(b.erank if b else None),
            "mean_birth": self._num(self.mean_birth()),
            "best_birth": self._num(b.birth if b else None),
            # SPEC: 01 §1.7 -- a history file must say which arm wrote it.
            "lamarck": bool(b.lamarck) if b else False,
        }
        self.history.append(row)
        return row

    # ------------------------------------------------------------- selection

    def sample_parents(self, k: int, generator: torch.Generator) -> list[Member]:
        """Uniform over **filled niches**, not over fitness. SPEC: 08 §8.2.

        Uniform-over-niches is what makes this quality-*diversity*: a poor elite
        in an unexplored region is as likely to be a parent as the best genome in
        the archive, which is the mechanism that keeps stepping stones alive.
        """
        elites = list(self.cells.values())
        if not elites:
            return []
        idx = torch.randint(0, len(elites), (k,), generator=generator)
        return [elites[int(i)] for i in idx]

    # ------------------------------------------------------------ persistence

    def save(self, path: str | pathlib.Path) -> None:
        """SPEC: 08 §8.6 E7 -- round-trip must be bit-identical."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "b1_edges": self.b1_edges, "b2_edges": self.b2_edges,
            "L_null": self.L_null, "n_attempts": self.n_attempts,
            "n_inserts": self.n_inserts, "history": self.history,
            "cells": {f"{k[0]},{k[1]}": dataclasses.asdict(m)
                      for k, m in self.cells.items()},
        }, path)

    @staticmethod
    def load(path: str | pathlib.Path) -> "Archive":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        a = Archive(b1_edges=blob["b1_edges"], b2_edges=blob["b2_edges"],
                    L_null=blob["L_null"])
        a.n_attempts, a.n_inserts = blob["n_attempts"], blob["n_inserts"]
        a.history = blob["history"]
        for key, d in blob["cells"].items():
            i, j = (int(x) for x in key.split(","))
            d["descriptors"] = tuple(d["descriptors"])
            d["niche"] = (i, j)
            a.cells[(i, j)] = Member(**d)
        return a

    def summary(self) -> str:
        b = self.best()
        lines = [
            f"archive: {len(self.cells)}/{self.n_niches} niches "
            f"({self.coverage():.0%})  QD {self.qd_score():.2f}  "
            f"best {b.fitness:.4f}" if b else "archive: empty",
        ]
        for i in range(len(self.b1_edges) - 1):
            row = []
            for j in range(len(self.b2_edges) - 1):
                m = self.cells.get((i, j))
                row.append(f"{m.fitness:6.3f}" if m else "     ." )
            lo, hi = self.b1_edges[i], self.b1_edges[i + 1]
            lines.append(f"  attn {lo:.2f}-{hi:.2f} | " + " ".join(row))
        return "\n".join(lines)
