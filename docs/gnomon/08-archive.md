# 08 — Open-ended search: MAP-Elites over genomes

C4 of the constitution. This is the **only** thing that improves `Gamma`, and it
is where the project's requirement for substantial open-ended search is
discharged.

## 8.1 Why quality-diversity rather than a hill climb

**[NORMATIVE]** The algorithm is MAP-Elites (Mouret & Clune, *Illuminating
search spaces by mapping elites*, arXiv:1504.04909, 2015; Cully, Clune, Tarapore
& Mouret, *Robots that can adapt like animals*, Nature 521:503–507, 2015).

Three reasons, and the third is the one that decides it:

1. **The result is a frontier, not a number.** §07.4 makes decoded size a
   behaviour descriptor, so the archive *is* the size-quality frontier, and each
   niche's elite is the best architecture at that size. A scalar hill climb
   would return one point on it and discard the rest of the information it paid
   for.
2. **Stepping stones.** An architecture that is poor now may be the parent of a
   good one; MAP-Elites keeps it because it occupies a different niche, which
   is the mechanism that makes the search open-ended rather than convergent.
3. **It has a falsifiable null.** Random search into the same archive is a
   complete, cheap, matched control, and `KG-QD` (§09) fires if MAP-Elites is
   not distinguishable from it. SOMA's open-ended stream had no such control,
   which is part of why its contribution was never measurable.

## 8.2 The loop

**[NORMATIVE]**

```
EVOLVE(cfg, G_gen, n_pop, n_seed):
  1  archive <- {}
  2  for i = 1 .. n_init:                       # generation 0
  3      Gamma <- random genome (§03.6, §07)
  4      INSERT(archive, Gamma, EVALUATE(Gamma))
  5  for g = 1 .. G_gen:
  6      batch <- []
  7      for i = 1 .. n_pop:
  8          if |archive| < 2 or U(0,1) < p_random:  Gamma <- random genome
  9          else:
 10              p1, p2 <- two uniformly random elites
 11              Gamma  <- MUTATE(CROSSOVER(p1, p2))        # §07.5
 12          batch.append(Gamma)
 13      for Gamma in batch:  INSERT(archive, Gamma, EVALUATE(Gamma))
 14      record(g, coverage, qd_score, best, archive)
```

**[NORMATIVE]** `EVALUATE(Gamma)` runs `n_seed` lifetimes (§05) on the **fitness
split** and returns their mean held-out CE. Seeds are **shared across all
genomes in a generation**, so every comparison within a generation is
seed-paired and the between-genome differences are not draws.

**[NORMATIVE]** `INSERT` places a genome in the niche its descriptors give
(§07.4) and replaces the incumbent **only if strictly better**. Elitism is
per-niche and unconditional: an elite is never displaced by a worse genome, so
the archive's quality is monotone in every niche and `qd_score` is monotone
overall. This makes a *falling* QD score a bug, which is a useful property for
a long run to have.

**[NORMATIVE]** `p_random = 0.10`. A tenth of every generation is drawn fresh
rather than from the archive, which bounds the loss of diversity when the
archive is small and keeps a floor under exploration for the whole run.

## 8.3 Parameters

**[NORMATIVE]** Defaults, with their justification. Two of them were **measured**
before the runs rather than chosen, and the measurements are given.

**The fitness horizon.** How long a lifetime must be to rank genomes at all is
an empirical question, so `HORIZON` (§09.1) asked it. M-pico, batch 128, 12
genomes x 3 seeds at each length:

| `N_life` | tokens | s / lifetime | `h2` | 95% BCa | `sigma_P` | best..worst | `KG-HERIT` |
|---|---|---|---|---|---|---|---|
| 50 | 0.8 M | 6.0 | 0.402 | [0.058, 0.699] | 0.187 | 7.99 .. 8.50 | **FAIL** |
| 100 | 1.6 M | 11.1 | 0.310 | [0.052, 0.677] | 0.325 | 7.47 .. 8.21 | **FAIL** |
| **150** | **2.5 M** | **15.7** | **0.398** | **[0.138, 0.726]** | **0.446** | 7.00 .. 7.99 | **PASS** |

The point estimate of `h2` is stable across all three; what changes is
`sigma_P`, which more than doubles from 50 steps to 150 as the models actually
separate, and with it the *evidence* for heritability. At 50 steps the models
have barely moved off `ln V = 8.32` and most of the between-genome variance is
still seed noise. `N_life = 150` is therefore the shortest admissible fitness
horizon, and it is used.

**Batch size.** The step is CPU-dispatch bound (§05's runner), so batch 128
costs 100 ms/step against batch 32's 82 ms while carrying four times the tokens:
**6.15 ms per thousand tokens against 20.1**, a 3.3x gain for nothing. The
`N_life = 150` at batch 128 above is the same 2.5M-token budget as the
`N_life = 600` at batch 32 that produced §09.5's `h2 = 0.389`, and it reproduces
it (0.398) in a third of the wall-clock.

| | value | why |
|---|---|---|
| `N_life` | 150 | measured above; the shortest that passes `KG-HERIT` |
| `batch` | 128 | measured above |
| `n_init` | 32 | fills a usable fraction of a 36-niche map before variation starts |
| `n_pop` | 24 | one generation is `24 x n_seed` lifetimes, about 13 min |
| `n_seed` | 2 | `V_E = 0.045` against `V_G = 0.029` puts the standard error of a genome's mean at 0.15 nats over two seeds, against a between-genome sd of 0.17: the point where more seeds buy less than more genomes |
| `G_gen` | 20-40 | set by the wall-clock available; §09.6 rule 4 requires the value used to be stated with every curve |
| niches | 6 x 6 = 36 | `B1` in 6 bins over `[0,1]`, `B2` in 6 over §07.4's computed range |
| `sigma_rel`, `p_mut`, `p_random` | 0.15, 0.25, 0.10 | §07.5, §8.2 |

At 15.7 s per lifetime, `(32 + G_gen * 24) * 2` lifetimes cost **3.1 h** at
`G_gen = 20` and **6.6 h** at `G_gen = 30`.

## 8.4 Reported quantities

**[NORMATIVE]** Per generation:

```
    coverage   = |filled niches| / |niches|
    qd_score   = sum over filled niches of (L_null - L_elite)     # >= 0, monotone
    best       = min over the archive of L_elite
    mean_elite = mean over filled niches of L_elite
```

`qd_score` uses `L_null - L_elite` rather than `-L_elite` so that it is
non-negative and an empty niche contributes exactly zero, which is what makes it
comparable across runs with different coverage.

**[NORMATIVE — the generation curve.]** The headline figure is `best` and
`mean_elite` against generation, three run-seeds, with the Mann–Kendall trend
test (exact null, vendored `utils/stats.mann_kendall`) and a BCa interval on the
slope. `KG-FLAT` (§09) fires if no significant downward trend exists.

**[NORMATIVE — the registered prediction.]** §06.2 measured `h2 = 0.389` and
`sigma_P = 0.273` at generation 0, giving `R = i h2 sigma_P = 0.135` nats per
generation under truncation selection on the top quarter. MAP-Elites is not
truncation selection, so this is an **upper bound on the early response** rather
than a point prediction: elitism per niche is a weaker selection differential
than global truncation because parents are drawn uniformly over niches. §09
reports the observed early slope against this bound. It was recorded before the
run.

## 8.5 Ablations

**[NORMATIVE]** Each is the same budget with one mechanism removed.

| | arm | what it isolates |
|---|---|---|
| A-RAND | random search into the same archive | is MAP-Elites earning its place? (`KG-QD`) |
| A-NOARCH | `Gamma_a` frozen at its initial value | is architecture search earning its place? |
| A-NOX | mutation only, no crossover | §07.5's linkage argument |
| A-1D | one descriptor (`B1` only), 6 niches | is the size axis doing work? |
| A-ANNEAL | `eps = 0` throughout | does the PNAS paper's necessity transfer to language? |
| A-FIXSIG | `log_sigma` frozen at 0 | is §10.4's self-adaptation earning its place? |
| R-ADAM | the lifetime rule pinned to `ADAM_POINT` | the bare baseline, and **the default** since §10.5b |
| R-ETA | only `B` decoded | v0.9's `eta`, kept as the ornament §10.1 diagnosed |
| R-DPR-FROZEN | all four coefficients decoded once at birth | whether the field must track a moving soma |
| A-LAMARCK | gradient into `Gamma` through the decode | the paper's own algorithm, as the upper bound |

**[NORMATIVE — this table named an arm that no longer exists.]** It listed
`A-ETA`, "`eta = 1` throughout the lifetime", which was v0.9's vocabulary: at
v1.0 `eta` stopped being a scalar gain and became §10.2's four-coefficient rule,
and the ablation that asks the same question became `R-ADAM` — the rule pinned
to the point where it *is* AdamW. The row is replaced by the four arms §10.6
actually defines, together with `A-FIXSIG`, which was implemented and never
listed here. The set in this table and the set in `scripts/gnomon_evolve.py` are
checked against each other by a test, because a table of arms that disagrees
with the arms is how a "missing arm" becomes invisible to §09.6 rule 4.

**[NORMATIVE]** `A-LAMARCK` is the paper's inner step, which the paper itself
calls *"a form of Lamarckian evolution, and as such biologically unrealistic"*.
It is expected to be far stronger per generation than the Darwinian arms and is
reported as an **upper bound**, never as GNOMON's result. Every artifact it
produces is labelled: `EvolveConfig.label()`, `LifetimeResult.lamarck`,
`Member.lamarck` and every row of the run history, so a genome reloaded from an
archive months later still says which arm made it.

**[NORMATIVE — the arm was specified here and implemented nowhere until v1.1.]**
§01.7 and §05.1 both promised it and both pointed at "§07.6", which is an
acceptance-test table; the arm was specified only by this one table row.
`EvolveConfig` carried no flag and `LifetimeRunner` froze `Gamma`
unconditionally, so the upper bound every Darwinian result was to be measured
against did not exist. Three definitional choices had to be made to implement
it, and each is recorded here rather than left in the code:

1. **The germline step is taken after the soma's.** Within one step the two
   timescales then see the same `W`: updating `Gamma` between the soma's
   gradient and the soma's step would make the soma's displacement one computed
   against a decode that no longer exists.

2. **Consecutive seeds are consecutive lives of one lineage.** In every other
   arm a genome's `n_seed` lifetimes are independent replicates and their mean
   is an unbiased estimate of its fitness. Here the genome is loaded once and
   the seeds run on it in sequence, because an acquired change discarded
   between seeds is not inherited and inheritance of acquired change is the
   whole content of the arm. The consequence is stated rather than hidden: the
   later seeds start from a genome the earlier ones improved, so the mean over
   seeds is **not** an unbiased fitness estimate in this arm, and it is not
   compared across arms as though it were.

3. **The step size is `0.1 x` the soma's schedule, and is untuned.** An upper
   bound tuned against the arms it bounds would not be one. The ratio is below
   unity because every one of `Gamma`'s 5,279 reals touches every decoded
   weight, so an equal step moves the phenotype by orders of magnitude more.
   Any future tuning of it is a change to the reported bound and is recorded as
   such.

**[NORMATIVE]** The arm asserts that `Gamma` actually moved, per tensor. A
severed gradient into the decode does not produce a visibly broken run: it
produces a plausible one, numerically identical to the Darwinian arm and
labelled as the ceiling — §00's RC1 relocated from the model into the
instrument. `G-MUTATE` M23 plants the detached decode and M24 plants a path
left open into the next lifetime, which would make a Darwinian control run in
the same process a second Lamarckian run.

## 8.6 Acceptance tests

**[NORMATIVE]** The archive and the loop are accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| E1 | elitism | an inserted genome never displaces a better incumbent; `qd_score` is non-decreasing over any run |
| E2 | niching | a genome's niche is exactly the one its descriptors give; bin edges are closed-open and the top bin is closed |
| E3 | seed pairing | all genomes in a generation are evaluated on the same seed set; a mutant that unpairs them fails |
| E4 | determinism | a whole run is reproducible bit-for-bit from its seed |
| E5 | **the null is reachable** | on a synthetic fitness function with no structure, MAP-Elites and random search are statistically indistinguishable by TOST — the control for `KG-QD` |
| E6 | **the null is beaten** | on a synthetic fitness function with a known stepping-stone structure, MAP-Elites beats random search. Without both E5 and E6, `KG-QD` is untested in both directions |
| E7 | archive round-trip | save and reload an archive; every genome, fitness and descriptor is bit-identical |
| E8 | coverage accounting | `coverage` counts filled niches, never attempted ones |
| E9 | budget | the number of lifetimes run equals `n_init + G_gen * n_pop` times `n_seed`, exactly |
| E10 | **the Lamarckian arm exists and is labelled** | `A-LAMARCK` moves every tensor of `Gamma`, closes the gradient path before returning, and is labelled on the config, the result, the member and every history row |
