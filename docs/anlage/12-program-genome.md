# 12. ANLAGE-P — the program genome

**Status: normative (revision of §02–§07 after §11).** This section replaces the weight
genome of §03 as the load-bearing heritable object. It keeps the phenotype (§04), the
lifetime (§04.6), the population machinery (§07), growth (§06) and the protocol discipline
(§08), and re-discharges the two constitutional requirements of §01 with mechanisms whose
benefit has published evidence and a measured landscape rather than a biological analogy.

## 12.0 The principle, stated

**Heritable is what does not copy (A8).** In a lineage whose adults can be copied, a genome
earns its place only by carrying what copying cannot: the objects that must span sizes, runs
and machines without loss, and that are cheaper to acquire once than to re-learn every
lifetime. For a language model these are

| symbol | object | why it does not copy | why it is worth inheriting |
|---|---|---|---|
| `mu` | the **data programme**: sampling weights over data classes, per stage | it is a property of the corpus and the learner's stage, not of the weights; it transfers across sizes (P-1, P-2) | choosing it well is worth a constant factor in tokens (P-1: 2.6x at 8B from a 280M proxy) |
| `omega` | the **developmental schedule**: when to grow, by how much | a property of the lineage's compute path | growth is worth 2–4x tokens at the right time (P-6, P-7) and the timing is a landscape |
| `eta` | the **recipe** in scale-free units: learning-rate multipliers, gate offsets, warm-up and ramp fractions | it transfers across sizes under `muP`-style parametrisation (P-5) | a population searching it is population-based training (P-4: +0.35 BLEU on a Transformer from a population of 32), the population method that is used in practice |
| `s` | the **scale profile**: every vector role as a function of (role, relative depth, unit code) | scale-free by construction; regressed to `< 1%` (R4) | a non-random initialisation of every stage's gains, offsets and scales (K-INIT) |
| `Lambda` | the **unit prior**: moments of trained units' statistics per axis | small, scale-free | new units at growth are born typical rather than random (K-GROW) |

`Gamma = (mu, omega, eta, s, Lambda)`: `10^2`–`10^4` reals. Its expression into any stage is
`O(|Gamma|)`, trivially parallel, and identical at 13M and 3B. The weights are the second part
of the two-part code (Proposition 1) and are inherited by copying within a lineage and by
function-preserving expansion across sizes (§06). The bottleneck is therefore *real* (three
to seven orders of magnitude), *priced* (A2) and *not on the learner's capacity* (A3).

Where the neural network enters (constitutional requirement §01.2, "neural networks
fundamentally"): the phenotype is the full-capacity hybrid of §04; the population's outcome
surrogate `F` (12.5) is fit by regression and is a neural network when the sibling count
supports it (G-SURR decides against a quadratic); the vector genome `s` is the neural kernel
of §03 restricted to vector roles.

## 12.1 Objects and notation

**[NORMATIVE]** | symbol | type | meaning |
|---|---|---|
| `c in C` | data class, `|C| = n_C` | a partition of the training pool's windows (12.2); TinyStories `n_C = 6`, Pile `n_C = 22` (its domains) |
| `mu_s in Delta^{n_C - 1}` | simplex vector | sampling weights over classes at stage `s`; `mu = (mu_0, ..., mu_S)` |
| `P_c` | distribution over windows | the empirical distribution of class `c`'s windows in the pool |
| `P_mu = sum_c mu_c P_c` | mixture | the lifetime's training distribution at a stage |
| `q in Q`, `|Q| = 12` | output stratum (§07.4) | position-in-document × target-frequency class; the cases of lexicase |
| `omega = (f_0, ..., f_S)` | FLOP fractions | `sum_s f_s = 1`; stage `s` trains for `f_s C_total` FLOPs (§06.5) |
| `eta` | vector | `(log eta_M/eta_M^0, log eta_A/eta_A^0, a_0, g_0, p_warm, p_batch, rho_0, p_ramp)` |
| `s` | vector-genome parameters | `theta_f, o_m, A_sigma, b_sigma, w_sigma` of §03 restricted to vector roles (12.6) |
| `Lambda` | per-axis moments | `(mu_X, s_X)` of §02.6 over trained units' statistics (12.7) |
| `K` | siblings per generation | 4 initially (§07.5) |
| `F: Delta^{n_C-1} -> R^{|Q|}` | outcome surrogate | fitted map from a sibling's mixture to its per-stratum CE (12.5) |

## 12.2 Data classes and the mixed lifetime

**[NORMATIVE]** Every window of the pool carries a class `c(w)`. For a corpus with domain
labels (The Pile) the class is the domain of the document the window starts in. For
TinyStories, which has no labels, the class is a property partition of the document the
window starts in: length tercile (`L_doc` in tokens, edges at the corpus terciles) × rarity
half (fraction of the document's tokens with frequency rank `>= 1000`, split at the corpus
median), `n_C = 6`. The edges are computed once from the training shard and recorded in the
manifest (M1); they are innate (a property of the corpus), not learned.

**Sampling under CRN.** A `ClassifiedPool` holds one permutation per class, seeded by the
order seed. A batch at mixture `mu` draws `n_c = round_multinomial(B, mu)` windows from the
head of each class's permutation (a systematic-sampling rounding so that `sum_c n_c = B` and
`E[n_c] = B mu_c`, M2). Two siblings with different `mu` consume the *same* class streams at
different rates, so their shared windows are identical (CRN across mixtures, M3); a sibling
at the corpus mixture `mu^corpus_c = |c| / sum |c'|` reproduces the unclassified pool's
distribution in expectation (M4).

The lifetime objective at stage `s` is `E_{w ~ P_{mu_s}} [CE(W; w)]` with everything else as
in §04.6; the hot path is unchanged (class bookkeeping is index arithmetic on the CPU).

## 12.3 The generation

**[NORMATIVE]** ```
GENERATION-P(Gamma = (mu, omega, eta, s, Lambda), W_adult(g-1), g):
  1  W_0 <- W_adult(g-1) restricted to stage 0 (the primordium block; g = 0: init from s)
  2  for k = 1..K:
  3      mu_k    <- Dirichlet(alpha mu_0)                       # 12.4: the variation operator
  4      eta_k   <- eta + sigma_eta * xi_k                        # only after G-LIVE-eta
  5      W_k     <- LIFETIME(W_0, s = 0, f_0 C_total / K, P_{mu_k}, eta_k, order seed k)
  6      F_k     <- per-stratum CE on the fitness split (the cases)
  7  parents <- EPSILON_LEXICASE({F_k}, eps_q = sigma_e(q), n_par)
  8  W_x     <- MERGE(parents, W_0)  with (7.1)-(7.2)           # crossover in weight space
  9  mu*     <- argmin_{mu in T} sum_q w_q F_hat_q(mu)            # 12.5: the program regression
 10  W_adult <- GROW/TRAIN(W_x, stages 1..S at omega, P_{mu*}, eta)  # §06 with Lambda, s
 11  Gamma'  <- (mu*, omega', eta', s', Lambda')                  # 12.6-12.7: refits from W_adult
 12  return Gamma', W_adult
```

Line 1 is the warm start that §11.3 shows is never beaten by a decode; the primordium block
of a grown adult is its first `d_0` channels, `F_0` inner units, `h_0` heads and the `L_0`
layers at the spread positions (`LatentBank.shrunk`'s rule, G13), which are exactly the
stage-0 units that were trained longest. Line 5 spends `f_0 C_total` FLOPs in total (S9).

## 12.4 Variation: the Dirichlet neighbourhood

**[NORMATIVE]** `mu_k ~ Dirichlet(alpha mu_0)` with concentration `alpha` set so that the
siblings' per-stratum CE spread is `2–3 sigma_e(q)` on the strata the mixture moves
(G-LIVE-mu, measured before any generation runs; `alpha = 50` initially). A sibling whose
`mu_k` gives a class weight below `mu_min = 0.02` is clipped and renormalised (no class is
ever unseen, M5). The corpus mixture is always one of the `K` (the control sibling, S-CTRL).

This is the population's search axis, and it is chosen because it is the one axis with
*measured, replicated, interior* optima at scale: the domain-weight landscape of The Pile is
not monotone (down-weighting some domains improves every domain's perplexity, P-1), and the
optimum found by a small population transfers to a model 30x (P-1) and 1000x (P-2) larger.
The SCREEN rule of §00.3 ("no population on a monotone landscape") is satisfied by
measurement, not by hope. Two cautions are on the record and shape the protocol: the
small-scale optimum does not always transfer (P-8, AutoScale), which is why K-MIX is
evaluated on the *adult*, not on the siblings; and online mixing methods barely beat
stratified sampling on homogeneous data (P-9, Aioli), which is why G-LIVE-mu is measured on
TinyStories before anything is claimed there.

## 12.5 The program regression (generation-level consolidation, A6 revised)

**[NORMATIVE]** From the `K` siblings' `(mu_k, F_k)` (and those of the previous `G_mem`
generations, re-weighted by `gamma^{age}`), fit

```
    F_hat_q(mu) = a_q + b_q^T mu + mu^T C_q mu,       C_q symmetric,      (12.1)
```

by ridge least squares (`lambda_F`), one regression per stratum `q`; with `K G_mem >= 4 n_C`
samples the quadratic is identifiable, otherwise `C_q = 0` (M6). When it is identifiable, the
quadratic is used only if its held-one-out error on the `w_q`-weighted objective is below the
linear fit's (R3 measured a quadratic on 25 samples to predict *worse* than a line; the rule
is the same one G-SURR applies to a neural surrogate). The next programme is

```
    mu* = argmin_{mu in T(mu_0)} sum_q w_q F_hat_q(mu),   T = {mu : ||mu - mu_0||_1 <= tau},   (12.2)
```

a trust region of radius `tau = 0.3` around the current programme (the surrogate is local; the
step is bounded), solved by projected gradient on the simplex to `1e-8` (M7). `w_q` are the
stratum token frequencies of the test split (the adult objective is the plain CE). Because
line 9 consumes only `(mu_k, F_k)` scalars, the regression is a least-squares projection of
the population's outcome onto `n_C` numbers per stage — Definition 1's third instance, now on
the programme.

A neural surrogate (`F_hat` a 2-layer MLP over `mu`) is the registered ablation G-SURR; it
replaces (12.1) only if its held-one-out error over the siblings is lower by more than the
quadratic's bootstrap CI (P-2 uses gradient-boosted trees for the same map).

## 12.6 The scale profile `s` (the vector genome)

The kernel of §03 keeps its vector roles: for `m in {AB, OGATE, AGATE, NORM_MIX, NORM_FFN,
NORM_FINAL, LOGIT_SCALE, CONV_Q, CONV_K, CONV_V}`,

```
    v_hat_{l,m}[i] = u_i^{(1)} . o_m + offset(m),    u_i = f([b_i; z_i; r_l; y_l; +1]),          (12.3)
```

with `z_i` and `y_l` fit by (5.3) restricted to vector roles (`lambda_z` as before). R4 measured
this regression to `< 1%` residual. The birth of a stage from `s` sets every vector role from
(12.3) and every matrix role from the innate initialisation of §04.3 (the free init at the
scales `s_target`); K-INIT measures the value of `s` against the plain free init at the
primordium (§12.9). At growth, new layers' vectors are decoded from (12.3) at their relative
depth; new units' gains from the axis prior (12.7).

Matrix roles are **not** decoded anywhere (§11.3). `Z` on channel and token axes survives only
as the sampling prior of new units (§02.6) and as the identity input of (12.3).

## 12.7 The unit prior `Lambda` and growth

**[NORMATIVE]** Definition 2 (§06.2) stands with one substitution in step 2: entries with an index in `A_new`
are **not** decoded; new rows/columns are initialised from `Lambda` — per axis, the mean and
standard deviation of trained units' row norms, and for gate rows the mean gate pre-activation —
so that a new unit has the *scale* of a typical trained unit and random direction (B-RAND is
the same at the innate scale; B-COPY the HyperCloning copy; the three are the K-GROW arms).
The ramp gates `rho` and Theorem G1 are unchanged. `omega` is a genome coordinate; its
mutation among siblings (heterochrony) is enabled only after K-GROW passes and G-LIVE-omega
shows the timing landscape is live (§08's rule).

## 12.8 Kill criteria and gates (revised from §08.5)

**[NORMATIVE]** All at `alpha = 0.05`, paired by CRN; `>= 4` seeds; screening runs use the `t` fallback of
§08.2 and are labelled.

| id | fires when | consequence |
|---|---|---|
| G-NOISE, G-BASE, G-NOFREE-INVERSE, G-SPECNUM | as in §08.5 | as in §08.5 |
| **G-LIVE-mu** | the Dirichlet neighbourhood at `alpha` moves no output stratum's fitness CE by `>= 3 sigma_e(q)` across `K` siblings, at any `alpha in {200, 50, 10}` | the mixture axis is dead on this corpus; the population is decorative there and the lineage runs SINGLE; reported |
| **K-MIX** | the adult trained on `mu*` is not better than the adult trained on `mu^corpus` by `>= 3 sigma_e` at equal compute (paired) | the programme regression buys nothing; `mu` is fixed at the corpus mixture; reported |
| **K-POP** | lexicase + merge not better than the best sibling AND not better than SINGLE at equal compute | as in §08.5 |
| **K-INIT** | birth from `s` (vectors decoded, matrices innate) is not better than the free init by `>= 3 sigma_e` at 10% of the primordium budget | `s` is dropped; the genome is `(mu, omega, eta, Lambda)` |
| **K-GROW** | `Lambda`-seeded growth not better than B-COPY by `>= 3 sigma_e` at matched compute (B-COPY `<=` B-FLAT re-measured) | growth stays as B-COPY; `Lambda` removed; reported |
| **K-LOOP** | generation `g+1`'s adult not better than SINGLE-WARM (one lineage, corpus mixture, same total compute, same growth) by `>= 3 sigma_e` | the generational loop buys nothing beyond a warm start; reported as the central negative result |
| **K-SOTA** | as in §08.5 | as in §08.5 |
| G-SURR | the neural surrogate's held-one-out error is not below the quadratic's | (12.1) stays |

The weight-genome criteria K-BIRTH, K-SCALE and K-ANCHOR are closed by §11 with their
measured outcomes and are not re-run.

## 12.9 Protocol (revision of §08.1)

**[NORMATIVE]** | run | measures | gate |
|---|---|---|
| R1 NOISE | done (`sigma_e = 0.0060` at TS s0) | G-NOISE |
| **R2 CLASSES** | the class partition's manifest; per-class token counts; the corpus mixture; the per-stratum CE of the R1 seeds *by class* (does the free model already differ across classes?) | M1–M4 |
| **R3 LIVE** | `K = 8` siblings at `alpha in {200, 50, 10}` from one warm start, 6.25M tokens each; per-stratum spread against `sigma_e(q)`; also the `eta` axis (LR multipliers `{0.5, 1, 2}`) | G-LIVE-mu, G-LIVE-eta |
| R4 INIT | birth from `s` vs free init, 4 seeds, 2.5M tokens | K-INIT |
| R5 GROW | as before with the `Lambda` arm replacing "genome" | K-GROW |
| **R6 MIX** | `mu*` from R3's regression vs `mu^corpus`, 4 seeds, full stage-0 budget | K-MIX |
| R7 POP | lexicase + merge vs best sibling vs SINGLE, from R3's siblings | K-POP |
| R8 LOOP | 3 generations on TinyStories vs SINGLE-WARM at equal compute | K-LOOP |
| R9 PILE | the Pile lineage with its 22 domains (`n_C = 22`), where P-1/P-2 predict the mixture axis is live | K-MIX, K-LOOP, K-SOTA |

TinyStories is homogeneous by construction; the honest expectation, recorded before R3 runs,
is that G-LIVE-mu may fire there and pass on the Pile. A dead mixture axis on TinyStories
does not falsify §12; it moves the population's test to the corpus that has domains.

## 12.10 Acceptance tests

**[NORMATIVE]** `anlage/program.py` and `anlage/data.py`'s classified pool are accepted only
when all of the following hold.

| # | property | test |
|---|---|---|
| M1 | classes are innate | the TinyStories partition edges are a deterministic function of the training shard and are recorded in the manifest; two builds agree |
| M2 | systematic rounding | `n_c` sums to `B` for every `mu` and `B`; `E[n_c] = B mu_c` over 10,000 draws to `1e-2` |
| M3 | CRN across mixtures | two pools at different `mu` with one order seed: the windows a class supplies are the same sequence |
| M4 | corpus mixture | at `mu^corpus` the class histogram of 10,000 sampled windows matches the pool's to `1e-2` |
| M5 | floor | no class weight below `mu_min` after clipping; the result is on the simplex to `1e-12` |
| M6 | identifiability | with fewer than `4 n_C` samples the fit is linear; with more, the quadratic recovers a planted `(a, b, C)` to `1e-6` |
| M7 | trust region | the solver's output is on the simplex and within `tau` in `l_1` of `mu_0`; on a planted convex surrogate it matches the interior optimum to `1e-6` when that lies inside `T` and lies on the boundary otherwise |
| M8 | program round-trip | `Gamma` serialises and reloads bit-exactly; its size is scale-free (identical for every stage of both lineages) |
| M9 | K-INIT protocol | the `s`-birth of a stage decodes every vector role from (12.3) and leaves every matrix role at the innate init (asserted by role) |
| M10 | budget | S9 with the mixture: `K` siblings at `f_0 C_total / K` each; the merged elite's adult at `(1 - f_0) C_total` |
| M11 | lineage determinism | `GENERATION-P` from one `seed_g` is bit-reproducible on CPU (S10 extended to the programme regression) |
| M12 | no decode in step | G-NOFREE-INVERSE unchanged; the mixed pool adds no GPU work to the step (step time within 2% of the unclassified pool's, measured) |

## 12.11 Implementation plan

| module | contents | reuse |
|---|---|---|
| `anlage/data.py` | `classify_documents` (length terciles × rarity), `ClassifiedPool.batches(batch, steps, seed, mu)` with systematic rounding; Pile domain labels from the shard's document table | `screen/crn.py` |
| `anlage/program.py` | `Program` dataclass `(mu, omega, eta, s_state, Lambda)`, `dirichlet_neighbourhood`, `fit_surrogate` (12.1), `solve_trust_region` (12.2), serialisation | `gnomon/utils/stats.py` |
| `anlage/genome.py` | unchanged for vectors; matrix decode retained for the record and for tests, unused by the lifetime | |
| `anlage/grow.py` | `mode = "prior"` (12.7) beside `copy`/`random` | |
| `scripts/anlage_classes.py`, `anlage_live.py`, `anlage_init.py`, `anlage_mix.py`, revised `anlage_pop.py`, `anlage_generation.py` | R2–R8 | `anlage_common.py` |
| `tests_anlage/unit/test_program.py` | M1–M12 | |

## 12.12 What may be claimed

As §03.10 of the plan, with one addition: the claim "the genome carries the phenotype" is
**withdrawn** for weight matrices (§11) and is not made for the programme either — what is
claimed, if measured, is that a lineage whose programme is fit from a primordium population
reaches a lower adult loss at equal compute than one lineage on the corpus mixture (K-LOOP),
and by how much.
