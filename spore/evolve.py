"""Selection across lifetimes: the (mu + lambda) loop of Definition 13. SPEC: 01 §1.8, 02 K-EVO.

Fitness is the fitness-set CE after a screening-budget life; the test CE is recorded in every
life's record for the analysis of transfer but never read by selection. Lives are cached by
genome digest so a surviving elite is not re-lived.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Callable

import numpy as np

from spore.genome import Genome, crossover, mutate

__all__ = ["evolve"]


def evolve(base: Genome, evaluate: Callable[[Genome], dict], *, pop: int, elites: int, generations: int,
           seed: int, out: pathlib.Path, wall_seconds: float, log: Callable[[str], None] = print,
           init_mutation_p: float = 0.35) -> dict:
    """`evaluate(genome) -> record` with keys `fitness_ce`, `test_ce`. Returns the history."""
    if elites < 1 or pop < elites:
        raise ValueError("1 <= elites <= pop")
    rng = np.random.default_rng(seed)
    cache: dict[str, dict] = {}
    history = {"base": base.as_dict(), "pop": pop, "elites": elites, "seed": seed, "generations": []}
    out.mkdir(parents=True, exist_ok=True)

    def score(g: Genome) -> dict:
        d = g.digest()
        if d not in cache:
            t = time.time()
            r = evaluate(g)
            r["genome"] = g.as_dict()
            r["digest"] = d
            r["seconds"] = time.time() - t
            cache[d] = r
        return cache[d]

    t0 = time.time()
    population = [base] + [mutate(base, rng, p=init_mutation_p) for _ in range(pop - 1)]
    scored = [(score(g), g) for g in population]
    scored.sort(key=lambda x: x[0]["fitness_ce"])
    gen = 0
    while True:
        elite = scored[:elites]
        history["generations"].append({
            "gen": gen, "elapsed": time.time() - t0,
            "population": [{"digest": r["digest"], "fitness_ce": r["fitness_ce"], "test_ce": r.get("test_ce"),
                            "genome": g.as_dict()} for r, g in scored],
            "elite_fitness": elite[0][0]["fitness_ce"], "elite_test": elite[0][0].get("test_ce"),
        })
        (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        log(f"gen {gen}: elite fitness {elite[0][0]['fitness_ce']:.4f} test {elite[0][0].get('test_ce'):.4f} "
            f"(pop best..worst {scored[0][0]['fitness_ce']:.4f}..{scored[-1][0]['fitness_ce']:.4f}; {time.time() - t0:.0f} s)")
        gen += 1
        if gen > generations or (time.time() - t0) > wall_seconds:
            break
        children = []
        for _ in range(pop):
            a = elite[int(rng.integers(0, len(elite)))][1]
            b = scored[int(rng.integers(0, len(scored)))][1]
            child = mutate(crossover(a, b, rng), rng)
            if child.digest() in cache or any(child.digest() == c.digest() for c in children):
                child = mutate(child, rng, p=0.5)
            children.append(child)
        scored = scored + [(score(c), c) for c in children]
        # (mu + lambda): the best `pop` of parents and children survive, by fitness only
        scored.sort(key=lambda x: x[0]["fitness_ce"])
        seen = set()
        uniq = []
        for r, g in scored:
            if r["digest"] not in seen:
                seen.add(r["digest"])
                uniq.append((r, g))
        scored = uniq[:pop]
    history["elite"] = scored[0][0]
    history["n_lives"] = len(cache)
    history["seconds"] = time.time() - t0
    (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    return history
