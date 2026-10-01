"""Landscape topology: the measurement that decides whether a population is needed.

An evolutionary algorithm earns its place over repeated local search only on a landscape
whose *shape* defeats local search. Everything here measures shape.

Three results from the fitness-landscape literature set the design, and each of them
corrects a mistake an earlier draft of this work had made:

**Peak count is necessary, not sufficient.** Papkou, Garcia-Pastor, Escudero & Wagner
(*Science* 382:eadh3860, 2023) mapped >260,000 DHFR genotypes and found **514 peaks in a
landscape that is nevertheless "easily navigable"**, because the high peaks share one very
large basin. So `local_optima` alone cannot gate anything; `basins` runs beside it.

**Reciprocal sign epistasis is necessary for multiple peaks** — Poelwijk, Tanase-Nicola,
Kiviet & Tans (*J. Theor. Biol.* 272(1):141-144, 2011): *"systems incapable of reciprocal
sign epistasis will always possess a single fitness peak"*, and by a Morse-theory argument
no *sufficient* condition can be written in terms of local interactions at all. Saona,
Kondrashov & Khudiakova (*Bull. Math. Biol.* 84(8):74, 2022) add **#RSE >= #peaks - 1**.

**...but that theorem assumes distinct fitnesses, and we have a noise floor.** Ivankov &
Zorin (*Genome Biol. Evol.* 18(2):evag024, 2026) give a two-peaked landscape with no RSE
when the intermediates are neutral. Every comparison here therefore takes an explicit
`margin`, and pairs closer than it are reported as *indistinguishable* rather than silently
resolved in favour of whichever way the noise fell. Manufacturing ruggedness out of
measurement error would be this project's old mistake in a new costume.

Fitness is **cross-entropy in nats, lower is better**, throughout. That convention is stated
once here and never inverted, because a sign error in this module would invert the project's
central verdict.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import statistics
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = ["Landscape", "Peak", "SearchResult"]

Genome = tuple[int, ...]


@dataclasses.dataclass
class Peak:
    genome: Genome
    fitness: float
    basin_size: int
    margin_over_neighbours: float


@dataclasses.dataclass
class SearchResult:
    name: str
    budget: int
    mean_best: float
    sd_best: float
    p_found_global: float
    mean_evals_to_global: Optional[float]


class Landscape:
    """A complete, measured genotype-to-fitness map over a discrete cube.

    `table` maps each genome (a tuple of small non-negative ints, one per locus) to the
    list of its per-seed cross-entropies. The map must be *complete* over the product of
    the observed alphabets -- a sampled landscape cannot support an exact local-optimum
    count, and reporting one from a sample would be a stronger claim than the data makes.
    """

    def __init__(self, table: Mapping[Genome, Sequence[float]], *,
                 margin: Optional[float] = None, paired: bool = True,
                 t_crit: float = 3.0):
        """`paired` uses the seed pairing that common random numbers provide.

        This is not a refinement, it is the difference between a result and a
        non-result. Comparing *means* against an absolute margin throws away exactly
        the variance reduction CRN was run to obtain. On the measured operator
        landscape the unpaired rule resolved **zero** vertices and returned
        "unresolved"; the paired rule resolves four endpoints with equal basins, from
        the same 80 runs. The margin rule is kept and reported beside it because it is
        the conservative bound, and a disagreement between the two is information about
        how much the conclusion rests on the pairing.
        """
        if not table:
            raise ValueError("empty landscape")
        self.table = {tuple(k): list(v) for k, v in table.items()}
        self.paired = paired
        self.t_crit = t_crit
        self.n_loci = len(next(iter(self.table)))
        if any(len(g) != self.n_loci for g in self.table):
            raise ValueError("genomes differ in length")
        self.alphabet = [
            sorted({g[i] for g in self.table}) for i in range(self.n_loci)]
        expected = math.prod(len(a) for a in self.alphabet)
        if len(self.table) != expected:
            raise ValueError(
                f"landscape is incomplete: {len(self.table)} genomes against "
                f"{expected} in the product of the observed alphabets. An exact "
                f"local-optimum count needs every vertex.")
        self.mean = {g: statistics.fmean(v) for g, v in self.table.items()}
        self.margin = self.default_margin() if margin is None else margin

    # ------------------------------------------------------------ the noise floor

    def sigma_e(self) -> tuple[float, int]:
        """Pooled within-genome sd and its degrees of freedom."""
        ss = df = 0
        for v in self.table.values():
            if len(v) < 2:
                continue
            m = statistics.fmean(v)
            ss += sum((x - m) ** 2 for x in v)
            df += len(v) - 1
        if df == 0:
            raise ValueError("no genome has >= 2 seeds; sigma_e is not estimable")
        return math.sqrt(ss / df), df

    def default_margin(self) -> float:
        """`2 x` the standard error of a genome's mean.

        Two vertices closer than this are indistinguishable at the seed count actually
        run, and the topology routines refuse to rank them.
        """
        s, _ = self.sigma_e()
        n = statistics.fmean([len(v) for v in self.table.values()])
        return 2.0 * s / math.sqrt(n)

    # ------------------------------------------------------------- graph structure

    def neighbours(self, g: Genome) -> list[Genome]:
        """Single-locus mutants. The neighbourhood defines the topology, so it is
        explicit rather than implied: 'local optimum' is meaningless without it."""
        out = []
        for i, alpha in enumerate(self.alphabet):
            for a in alpha:
                if a != g[i]:
                    out.append(g[:i] + (a,) + g[i + 1:])
        return out

    def paired_delta(self, a: Genome, b: Genome) -> tuple[float, float, float, int]:
        """`(mean, sd, t, n)` of `fitness(b) - fitness(a)` across matched seeds.

        Valid only because `crn.py` guarantees that seed `s` of every genome consumed
        the same windows in the same order. Without that guarantee this is an ordinary
        two-sample comparison wearing a paired test's clothes.
        """
        va, vb = self.table[a], self.table[b]
        n = min(len(va), len(vb))
        if n < 2:
            return float("nan"), float("nan"), float("nan"), n
        diff = [vb[i] - va[i] for i in range(n)]
        m = statistics.fmean(diff)
        s = statistics.stdev(diff)
        t = m / (s / math.sqrt(n)) if s > 0 else math.copysign(math.inf, m) if m else 0.0
        return m, s, t, n

    def better(self, a: Genome, b: Genome) -> bool:
        """`a` beats `b`. Lower cross-entropy is better.

        Paired by default: `a` wins only if the seed-matched difference clears
        `t_crit` standard errors. Falls back to the unpaired margin when pairing is
        disabled or the seed counts do not permit it.
        """
        if not self.paired:
            return self.mean[b] - self.mean[a] > self.margin
        m, _, t, n = self.paired_delta(a, b)
        if n < 2 or not math.isfinite(t):
            return self.mean[b] - self.mean[a] > self.margin
        return m > 0 and t > self.t_crit

    def better_unpaired(self, a: Genome, b: Genome) -> bool:
        """The conservative bound, reported alongside so the two can disagree visibly."""
        return self.mean[b] - self.mean[a] > self.margin

    def local_optima(self) -> list[Peak]:
        """Vertices that beat every neighbour by more than the margin."""
        peaks = []
        for g in self.table:
            nb = self.neighbours(g)
            if all(self.better(g, h) for h in nb):
                gap = min(self.mean[h] - self.mean[g] for h in nb)
                peaks.append(Peak(g, self.mean[g], 0, gap))
        sizes = self.basins()
        for p in peaks:
            p.basin_size = sizes.get(p.genome, 0)
        return sorted(peaks, key=lambda p: p.fitness)

    def global_optimum(self) -> Genome:
        return min(self.mean, key=self.mean.get)

    # --------------------------------------------------------------- adaptive walks

    def greedy_walk(self, start: Genome, rng: np.random.Generator
                    ) -> tuple[Genome, int]:
        """Steepest-ascent walk. Returns the endpoint and the number of steps.

        Ties inside the margin are broken at random rather than by index order: a
        deterministic tie-break would make basin sizes an artefact of how the alphabet
        happens to be numbered.
        """
        g, steps = start, 0
        while True:
            nb = [h for h in self.neighbours(g) if self.better(h, g)]
            if not nb:
                return g, steps
            best = min(self.mean[h] for h in nb)
            tied = [h for h in nb if self.mean[h] - best <= self.margin]
            g = tuple(tied[int(rng.integers(len(tied)))])
            steps += 1

    def adaptive_walks(self, *, seed: int = 0) -> dict:
        """Greedy ascent from every vertex, and — the part that decides things —
        **how much worse the worst endpoint is than the best**.

        Distinct endpoints alone do not establish that hill-climbing fails. If every
        endpoint is within a fraction of a standard deviation of the global optimum,
        a hill-climber loses nothing by stopping at any of them, and a population has
        nothing to win. `endpoint_cost_sigma` is that quantity, and it is the number
        the verdict turns on.
        """
        rng = np.random.default_rng(seed)
        lengths, ends = [], []
        for g in self.table:
            end, n = self.greedy_walk(g, rng)
            lengths.append(n)
            ends.append(end)
        gopt = self.global_optimum()
        s, _ = self.sigma_e()
        worst = max(ends, key=lambda e: self.mean[e])
        cost = self.mean[worst] - self.mean[gopt]
        return {
            "mean_length": statistics.fmean(lengths),
            "max_length": max(lengths),
            "fraction_reaching_global": sum(e == gopt for e in ends) / len(ends),
            "distinct_endpoints": len(set(ends)),
            "endpoints": sorted({"".join(map(str, e)) for e in ends}),
            "worst_endpoint_cost_nats": cost,
            "endpoint_cost_sigma": cost / s if s > 0 else float("inf"),
        }

    def basins(self, *, seed: int = 0, repeats: int = 32) -> dict[Genome, int]:
        """How many starts each optimum captures under steepest ascent.

        This is the Papkou correction made operational: a landscape with many peaks but
        one dominant basin is navigable by random-restart hill-climbing and does not need
        a population. Repeated because the tie-break is stochastic.
        """
        rng = np.random.default_rng(seed)
        counts: dict[Genome, float] = {}
        for _ in range(repeats):
            for g in self.table:
                end, _ = self.greedy_walk(g, rng)
                counts[end] = counts.get(end, 0.0) + 1.0 / repeats
        return {k: int(round(v)) for k, v in counts.items()}

    def funnel_ratio(self) -> float:
        """Largest basin as a fraction of the space. 1.0 means a single funnel."""
        sizes = self.basins()
        return max(sizes.values()) / len(self.table) if sizes else 1.0

    # ------------------------------------------------------------------- epistasis

    def sign_epistasis(self) -> dict:
        """Count simple and reciprocal sign epistasis over every 2-locus square.

        A square `(i, j)` on background `b` shows **reciprocal** sign epistasis when both
        `00` and `11` beat both `01` and `10` (or the mirror image) -- i.e. two corners of
        the square are local optima within it. That is the motif Poelwijk et al. prove is
        necessary for multiple peaks.

        Every comparison uses the margin, and a square containing any indistinguishable
        pair is counted as `indistinguishable` rather than assigned. Ivankov & Zorin's
        counterexample is exactly a landscape of neutral intermediates, so this is not
        pedantry -- it is the boundary of the theorem being respected.
        """
        simple = reciprocal = indistinct = total = 0
        rse_examples = []
        for i, j in itertools.combinations(range(self.n_loci), 2):
            if len(self.alphabet[i]) < 2 or len(self.alphabet[j]) < 2:
                continue
            for ai, bi in itertools.combinations(self.alphabet[i], 2):
                for aj, bj in itertools.combinations(self.alphabet[j], 2):
                    others = [self.alphabet[k] for k in range(self.n_loci)
                              if k not in (i, j)]
                    for rest in itertools.product(*others):
                        def build(vi, vj):
                            g, it = [], iter(rest)
                            for k in range(self.n_loci):
                                g.append(vi if k == i else vj if k == j else next(it))
                            return tuple(g)

                        c00, c01 = build(ai, aj), build(ai, bj)
                        c10, c11 = build(bi, aj), build(bi, bj)
                        total += 1
                        f = {k: self.mean[k] for k in (c00, c01, c10, c11)}
                        # Every one of the six pairs in the square must be
                        # distinguishable, by the same rule the topology uses. A
                        # square with any tie is Ivankov & Zorin's counterexample in
                        # miniature and is excluded rather than resolved by noise.
                        corners = (c00, c01, c10, c11)
                        if any(not (self.better(x, y) or self.better(y, x))
                               for x, y in itertools.combinations(corners, 2)):
                            indistinct += 1
                            continue
                        # Effect of flipping locus i, on each background of j.
                        e_i_at_aj = f[c10] - f[c00]
                        e_i_at_bj = f[c11] - f[c01]
                        e_j_at_ai = f[c01] - f[c00]
                        e_j_at_bi = f[c11] - f[c10]
                        if (e_i_at_aj > 0) != (e_i_at_bj > 0):
                            simple += 1
                        elif (e_j_at_ai > 0) != (e_j_at_bi > 0):
                            simple += 1
                        # Reciprocal: two opposite corners each beat both others.
                        corner_a = f[c00] < f[c01] and f[c00] < f[c10]
                        corner_b = f[c11] < f[c01] and f[c11] < f[c10]
                        corner_c = f[c01] < f[c00] and f[c01] < f[c11]
                        corner_d = f[c10] < f[c00] and f[c10] < f[c11]
                        if (corner_a and corner_b) or (corner_c and corner_d):
                            reciprocal += 1
                            if len(rse_examples) < 8:
                                rse_examples.append(
                                    {"loci": (i, j), "square":
                                     [list(c00), list(c01), list(c10), list(c11)],
                                     "fitness": [f[c00], f[c01], f[c10], f[c11]]})
        return {
            "squares_examined": total,
            "indistinguishable": indistinct,
            "simple_sign_epistasis": simple,
            "reciprocal_sign_epistasis": reciprocal,
            "examples": rse_examples,
            # Saona et al. 2022: #RSE >= #peaks - 1, so RSE = 0 bounds peaks at 1.
            "peak_upper_bound_from_rse": reciprocal + 1,
        }

    def additive_r2(self) -> float:
        """Fraction of between-genome variance explained by a main-effects-only model.

        `1.0` means the landscape is exactly additive: every locus contributes
        independently, a coordinate hill-climb is optimal, and no population can help.
        """
        genomes = list(self.table)
        cols = []
        for i, alpha in enumerate(self.alphabet):
            for a in alpha[1:]:
                cols.append([1.0 if g[i] == a else 0.0 for g in genomes])
        X = np.column_stack([np.ones(len(genomes))] + cols)
        y = np.array([self.mean[g] for g in genomes])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0

    # --------------------------------------------------------- classical statistics

    def fitness_distance_correlation(self) -> float:
        """Pearson correlation of fitness with Hamming distance to the global optimum.

        Requires knowing the optimum, which is fine on an enumerated landscape and is the
        measure's main practical limitation elsewhere. Reported with its counterexample
        on the record: Altenberg (ICGA '97) showed FDC can mispredict GA performance
        because it ignores the algorithm's dynamics. It is a descriptor here, never a gate.
        """
        gopt = self.global_optimum()
        d = np.array([sum(a != b for a, b in zip(g, gopt)) for g in self.table])
        f = np.array([self.mean[g] for g in self.table])
        if d.std() == 0 or f.std() == 0:
            return float("nan")
        return float(np.corrcoef(d, f)[0, 1])

    def autocorrelation(self, *, lags: int = 6, walk: int = 4000,
                        seed: int = 0) -> list[float]:
        """Fitness autocorrelation along a random walk (Weinberger 1990).

        Note for the report: this and FDC are sensitive to monotone rescaling of fitness
        (Naudts & Kallel, *IEEE TEVC* 4(1):1-15, 2000), and cross-entropy, perplexity and
        bits-per-byte are monotone rescalings of one another. The local-optimum count is
        invariant under them, which is why it is the primary measurement and these are not.
        """
        rng = np.random.default_rng(seed)
        genomes = list(self.table)
        g = genomes[int(rng.integers(len(genomes)))]
        series = []
        for _ in range(walk):
            series.append(self.mean[g])
            nb = self.neighbours(g)
            g = nb[int(rng.integers(len(nb)))]
        x = np.array(series)
        x = x - x.mean()
        denom = float((x * x).sum())
        return [float((x[:len(x) - k] * x[k:]).sum() / denom) if denom else float("nan")
                for k in range(1, lags + 1)]

    # ------------------------------------------------- searchers, at zero GPU cost

    def _noisy(self, g: Genome, rng: np.random.Generator) -> float:
        """One evaluation, resampled from the measured seeds.

        A searcher must face the same noise a real run would, or the simulation flatters
        it. Resampling the observed values rather than adding Gaussian noise keeps the
        noise distribution empirical.
        """
        v = self.table[g]
        return v[int(rng.integers(len(v)))]

    def simulate(self, name: str, *, budget: int, runs: int = 10_000,
                 seed: int = 0) -> SearchResult:
        """`name` in {'random', 'rmhc', 'restart_rmhc', 'ga'}.

        The benchmark to beat is not a strawman. On Royal Road R1 -- a function built to
        favour genetic algorithms -- random-mutation hill-climbing found the optimum in
        about 6,179 evaluations, roughly an order of magnitude faster than the GA
        (Mitchell, Forrest & Holland, ECAL 1992; the cause, hitchhiking, is diagnosed in
        Forrest & Mitchell, FOGA 2, 1993). Any claim that a population helps here has to
        clear RMHC on the measured landscape first.
        """
        rng = np.random.default_rng(seed)
        gopt = self.global_optimum()
        genomes = list(self.table)
        bests, hits, first = [], 0, []

        for _ in range(runs):
            seen_best, best_g, evals_to_global = math.inf, None, None
            n_ev = 0

            def ev(g):
                nonlocal seen_best, best_g, n_ev, evals_to_global
                n_ev += 1
                val = self._noisy(g, rng)
                if val < seen_best:
                    seen_best, best_g = val, g
                if g == gopt and evals_to_global is None:
                    evals_to_global = n_ev
                return val

            if name == "random":
                while n_ev < budget:
                    ev(genomes[int(rng.integers(len(genomes)))])

            elif name in ("rmhc", "restart_rmhc"):
                cur = genomes[int(rng.integers(len(genomes)))]
                cur_f = ev(cur)
                stall = 0
                while n_ev < budget:
                    nb = self.neighbours(cur)
                    cand = nb[int(rng.integers(len(nb)))]
                    cand_f = ev(cand)
                    if cand_f <= cur_f:
                        cur, cur_f, stall = cand, cand_f, 0
                    else:
                        stall += 1
                    if name == "restart_rmhc" and stall > 2 * len(nb):
                        cur = genomes[int(rng.integers(len(genomes)))]
                        cur_f, stall = ev(cur), 0

            elif name == "ga":
                mu = max(4, min(16, len(genomes) // 4))
                pop = [genomes[int(rng.integers(len(genomes)))] for _ in range(mu)]
                fit = [ev(g) for g in pop]
                while n_ev < budget:
                    order = sorted(range(len(pop)), key=lambda i: fit[i])
                    parents = [pop[i] for i in order[:max(2, mu // 2)]]
                    a = parents[int(rng.integers(len(parents)))]
                    b = parents[int(rng.integers(len(parents)))]
                    child = tuple(a[i] if rng.random() < 0.5 else b[i]
                                  for i in range(self.n_loci))
                    if rng.random() < 0.5:
                        i = int(rng.integers(self.n_loci))
                        alt = [x for x in self.alphabet[i] if x != child[i]]
                        if alt:
                            child = (child[:i]
                                     + (alt[int(rng.integers(len(alt)))],)
                                     + child[i + 1:])
                    cf = ev(child)
                    worst = max(range(len(pop)), key=lambda i: fit[i])
                    if cf < fit[worst]:
                        pop[worst], fit[worst] = child, cf
            else:
                raise ValueError(f"unknown searcher {name!r}")

            bests.append(self.mean[best_g])
            if best_g == gopt:
                hits += 1
            if evals_to_global is not None:
                first.append(evals_to_global)

        return SearchResult(
            name=name, budget=budget,
            mean_best=statistics.fmean(bests),
            sd_best=statistics.stdev(bests) if len(bests) > 1 else 0.0,
            p_found_global=hits / runs,
            mean_evals_to_global=statistics.fmean(first) if first else None,
        )

    # ------------------------------------------------------------------ the verdict

    def report(self) -> dict:
        s, df = self.sigma_e()
        peaks = self.local_optima()
        rse = self.sign_epistasis()
        walks = self.adaptive_walks()
        funnel = self.funnel_ratio()
        n_sep = len(peaks)
        cost_sigma = walks["endpoint_cost_sigma"]

        # The gate turns on what hill-climbing COSTS, not on how many peaks exist.
        # Papkou et al. (Science 2023) found 514 peaks in an easily navigable
        # landscape; conversely a plateau of statistical ties has several endpoints
        # and no cost at all. Peak count is a description; the cost is the decision.
        # `n_local_optima` counts STRICT peaks -- vertices beating every neighbour.
        # On a plateau of statistical ties that count is zero while greedy ascent
        # still has well-defined endpoints, so it describes the landscape but cannot
        # decide anything. The decision uses the walks: if nothing improves anywhere,
        # the measurement failed; otherwise what matters is what stopping early costs.
        if walks["mean_length"] == 0:
            verdict = ("UNRESOLVED -- no vertex improves on any neighbour at "
                       f"t>{self.t_crit}; the landscape is flat at this resolution. "
                       "Not evidence of a single peak.")
        elif cost_sigma < 2.0:
            verdict = (f"NAVIGABLE -- {n_sep} endpoint(s), but the worst costs only "
                       f"{cost_sigma:.1f} sigma_e over the global optimum. "
                       f"Hill-climbing loses nothing; a population has nothing to win.")
        elif funnel >= 0.8:
            verdict = (f"RUGGED BUT SINGLE-FUNNEL -- worst endpoint costs "
                       f"{cost_sigma:.1f} sigma_e, but one basin holds "
                       f"{funnel:.0%} of starts; random-restart hill-climbing suffices.")
        else:
            verdict = (f"RUGGED AND MULTI-FUNNEL -- worst endpoint costs "
                       f"{cost_sigma:.1f} sigma_e across {n_sep} endpoints with no "
                       f"dominant basin. A population may have a job; confirm with "
                       f"the simulated searchers before spending GPU time.")

        # The conservative bound, reported so a disagreement is visible rather than
        # buried in a choice of statistic.
        unpaired = [g for g in self.table
                    if all(self.better_unpaired(g, h) for h in self.neighbours(g))]
        return {
            "n_genomes": len(self.table),
            "n_loci": self.n_loci,
            "paired": self.paired, "t_crit": self.t_crit,
            "sigma_e": s, "sigma_e_df": df, "margin": self.margin,
            "n_local_optima": n_sep,
            "n_local_optima_unpaired": len(unpaired),
            "peaks": [dataclasses.asdict(p) for p in peaks],
            "global_optimum": list(self.global_optimum()),
            "funnel_ratio": funnel,
            "additive_r2": self.additive_r2(),
            "adaptive_walks": walks,
            "sign_epistasis": rse,
            "fitness_distance_correlation": self.fitness_distance_correlation(),
            "autocorrelation": self.autocorrelation(),
            "verdict": verdict,
        }
