# ANLAGE — results register

**Status: record.** Every number here is re-measured from the artifacts named beside it
(§08.6 rule 3). Negative results are recorded in the same table as positive ones (rule 4).
Screening runs use 4 seeds and Student-t paired intervals (§08.2); they are labelled as
screening.

## R1 NOISE — TinyStories stage 0

Configuration `s0:d256/L8/h4/kv2/F1024` (13,011,201 parameters), 25M tokens, batch 32 × 512,
Muon `eta_M = 0.02`, Adam `eta_emb = 3e-3`, pool seed 20260902, 4 seeds under CRN, eager
(no compile), 69k tok/s, peak 8.35 GB. Artifact: `runs/anlage/noise-tinystories-s0/noise.json`
(commit `8519d9d`).

| seed | test CE (nats) |
|---|---|
| 0 | 1.8571 |
| 1 | 1.8530 |
| 2 | 1.8432 |
| 3 | 1.8480 |
| **mean** | **1.8503** |
| **`sigma_e`** | **0.0060** (3 df) |

`G-NOISE` is discharged for this configuration. The 3-`sigma_e` bar for any mechanism at
stage 0 is therefore **0.018 nats**. The fitness split evaluates ~0.23 nats above the test
split for every seed (the two validation halves differ in content); contrasts are always
within one split.


**Re-evaluation on the v2 sets (08 §8.2; `noise.json`, v1 kept as `noise-v1.json`).** The same four adults on 256 windows sampled uniformly over each split: test CE 1.7935, 1.7926, 1.7902, 1.7922 (mean 1.7921), `sigma_e` = **0.0014** (3 df); fitness 1.9413, 1.9412, 1.9366, 1.9419. Every run from R3 on is measured on the v2 sets; the 3-`sigma_e` bar there is 0.004 nats.


**Re-evaluation on the v3 sets (08 §8.2; `noise.json`; v1 and v2 kept beside it).** The two validation halves differ in composition (v2: the test half holds 134 of 256 windows in the long classes, the fitness half 100), so from R3's re-evaluation on every set is drawn from the union of both halves by document index mod 3. The same four adults: test CE 1.8070, 1.8114, 1.8063, 1.8120 (mean 1.8092), fitness 1.8376, 1.8389, 1.8301, 1.8404 (the two sets now agree to 0.03 nats), `sigma_e` = **0.0029** (3 df). R6 and everything after it are measured on v3; the 3-`sigma_e` bar is 0.009 nats.

Per-stratum seed noise (the `eps_c` of §07.4, fitness split), by position-in-document ×
target-frequency class:

| c | stratum | test tokens | `sigma_e(c)` test | `sigma_e(c)` fitness |
|---|---|---|---|---|
| 0 | pos<32 x rank<100 | 17,487 | 0.0041 | 0.0033 |
| 1 | pos<32 x rank 100-1000 | 4,899 | 0.0069 | 0.0060 |
| 2 | pos<32 x rank>=1000 | 1,635 | 0.0456 | 0.0128 |
| 3 | 32-128 x rank<100 | 43,500 | 0.0037 | 0.0023 |
| 4 | 32-128 x rank 100-1000 | 20,898 | 0.0082 | 0.0064 |
| 5 | 32-128 x rank>=1000 | 6,859 | 0.0270 | 0.0152 |
| 6 | 128-256 x rank<100 | 21,092 | 0.0054 | 0.0042 |
| 7 | 128-256 x rank 100-1000 | 10,809 | 0.0064 | 0.0126 |
| 8 | 128-256 x rank>=1000 | 3,549 | 0.0550 | 0.0247 |
| 9 | 256+ x rank<100 | 205 | 0.0387 | 0.0159 |
| 10 | 256+ x rank 100-1000 | 92 | 0.0971 | 0.0528 |
| 11 | 256+ x rank>=1000 | 47 | 0.1139 | 0.2139 |

The rare-token strata at deep positions (c = 9–11) hold 47–205 tokens and have seed noise
of 0.04–0.21 nats; lexicase with `eps_c = sigma_e(c)` treats them as ties unless a sibling
moves them by more than that.

## R4 BIRTH — regression of the stage-0 phenotype into the genome

**K-BIRTH fires.** Genome dials `d_g = 64, H = 4, d_z = 16, d_h = 256` (`|Gamma|` = 418,288,
`|Z|` = 300,096; ratio 31.1x without and 18.1x with the latents), fit
by 3,000 Adam steps of (5.3) to R1's seed-0 phenotype (657 s).
Artifacts: `runs/anlage/birth-tinystories-s0-dg64H4z16/{birth.json,diag.json,log.txt}`.

| quantity | value (nats) |
|---|---|
| trained adult (R1 seed 0) | 1.8571 |
| `L_null = ln V` | 9.0109 |
| **at-birth CE of `D(Gamma, Z)`** | **8.5335** |
| `birth_share` (K-BIRTH needs ≥ 0.5) | **0.067** |
| random genome at birth | 13.7031 |
| shuffle control (rows permuted) | 11.0021 |
| `d_z = 0` arm: residual / at-birth | 0.549 / 9.23 |
| mean relative residual over keys | 0.509 |

**The mean residual hides a bimodal fit.** Role by role (`diag.json`; "W←D" is the trained
phenotype with only that role decoded, "D←W" the decoded phenotype with only that role's
trained weights restored):

| role | residual | W←D | D←W |
|---|---|---|---|
| Q | 0.812 | 1.9702 (+0.113) | 8.9493 (+0.416) |
| K | 0.829 | 2.3159 (+0.459) | 7.9039 (-0.630) |
| V | 0.800 | 3.0400 (+1.183) | 7.9358 (-0.598) |
| O | 0.809 | 3.3683 (+1.511) | 8.6553 (+0.122) |
| OGATE | 0.776 | 2.1372 (+0.280) | 8.7903 (+0.257) |
| AGATE | 0.616 | 1.8700 (+0.013) | 8.6009 (+0.067) |
| VE | 0.845 | 1.8582 (+0.001) | 8.5358 (+0.002) |
| GATE | 0.927 | 3.9206 (+2.063) | 8.3509 (-0.183) |
| UP | 0.924 | 3.2155 (+1.358) | 8.3130 (-0.220) |
| DOWN | 0.928 | 4.0224 (+2.165) | 9.2785 (+0.745) |
| EMB | 0.429 | 7.3221 (+5.465) | 12.0198 (+3.486) |
| AB | 0.004 | 1.8573 (+0.000) | 8.5252 (-0.008) |
| CONV_Q | 0.002 | 1.8572 (+0.000) | 8.5249 (-0.009) |
| CONV_K | 0.001 | 1.8572 (+0.000) | 8.5220 (-0.012) |
| CONV_V | 0.001 | 1.8573 (+0.000) | 8.4867 (-0.047) |
| NORM_MIX | 0.004 | 1.8587 (+0.002) | 8.5616 (+0.028) |
| NORM_FFN | 0.003 | 1.8576 (+0.000) | 8.5524 (+0.019) |
| NORM_FINAL | 0.001 | 1.8580 (+0.001) | 8.5416 (+0.008) |
| LOGIT_SCALE | 0.000 | 1.8571 (+0.000) | 8.5335 (-0.000) |

Every vector role and the causal convolutions (a few thousand entries) are fit to `< 1%`
residual; every weight matrix is fit to 7–20% of its variance (residual 0.78–0.93), the tied
embedding to 57%. The function is far more sensitive than the Frobenius residual: decoding
the FFN alone costs 1.4–2.2 nats, the lexicon alone 5.5 nats, and the decoded phenotype with
its lexicon and vectors restored to trained values is still at 12.01 nats. One layer decoded
(the rest trained) costs 0.14–0.48 nats: 0: 1.994 1: 2.065 2: 2.089 3: 2.113 4: 2.100 5: 2.107 6: 2.149 7: 2.338.

**Effective rank.** Decoded Q/UP/DOWN/O keep 178–212 of 256 (trained 204–240): the cubic term
lifts the rank, so the failure is not rank but *content*: with `d_z = 16` a unit's row in a
256-wide matrix is described by 16 acquired numbers plus its innate code, and a trained matrix
with a flat spectrum (erank 0.8–0.94 of full) has no such per-unit low-dimensional structure to
find. The residual of 0.8–0.93 on the matrices is what an information count predicts
(`d_z / d_model ≈ 0.06`). The decoded embedding has erank 26 against 218: token rows collapse
onto the code's frequency-rank manifold.

**K-SCALE (same run).** The stage-0 genome expresses stage 1 at 9.92 nats (pure decode), above
`L_null`; growing the trained stage-0 model to stage 1 with all gates at 1 gives genome-decoded
new units 2.916, copied units 2.799, random new units 1.993 (the trained
model was 1.8571). Full-scale decoded units inject more disturbance than small random ones;
Definition 2's ramp from `rho_0 = 0.1` is what makes any of these usable, and the registered
comparison (R5, after the ramp and training) is pending.

**Reading.** A single regression of an unconstrained trained adult into a neural-kernel genome
at 18x does not carry the phenotype: the pre-registered consequence of K-BIRTH is to raise the
dials and, failing that, to hold the genome design falsified for this use. The two follow-ups
below separate *capacity* (does the decodable set contain a good stage-0 model at all?) from
*assimilation* (can a lifetime trained with intermittent regression and a pull toward its
decode, the PNAS loop, land on one?).

### R4-probe — the manifold's capacity: training `D(Gamma, Z)` end to end

Same dials as R4, every weight decoded on every step, Adam on `(Gamma, Z)` (lr 2e-3, the
regression's priors `lambda_z = 1e-3, lambda_g = 1e-6`, clip 1.0), R1's pool, order and
budget (25M tokens). Artifact: `runs/anlage/manifold-tinystories-s0-dg64H4z16h256/manifold.json`.

| quantity | value |
|---|---|
| untrained decode (random genome) | 13.7031 nats |
| held-out CE at 5M / 10M / 15M / 20M / 25M tokens | 0M: 13.7031  5M: 9.0109  10M: 9.0109  15M: 9.0109  20M: 9.0109  25M: 9.0109  25M: 9.0109 |
| **final test CE** | **9.0109** (`L_null` = 9.0109) |
| effective rank of the decoded matrices at the end | Q_0 1.1/256, K_0 1.0/256, V_0 1.0/256, O_0 1.0/256, GATE_0 1.1/256, UP_0 1.1/256, DOWN_0 1.0/256, AB_0 1.0/8 |
| throughput | 42,928 tok/s, 343 ms/step (free training: 69,020 tok/s, 213 ms) |
| latent RMS, start → end | 1.00 → 0.43 |

The decoded model reaches the uniform predictor within 5M tokens and never leaves it; the
latents shrink under the prior along a flat loss and every decoded matrix collapses to rank
one (all units identical). This is GNOMON's failure (§00.2: decoded erank collapsed 38x over
40,000 steps at M1) reproduced in ten minutes on a well-conditioned decoder with a fixed
output scale. Gradient descent through a per-unit-identity decoder from a random genome does
not find the good models of the decodable set, if there are any; the regression from the best
possible target (R4) says the set does not contain the trained model to better than 20% of
its variance per matrix.

### R4b — assimilation: the PNAS loop as a lifetime hook

R1's seed-0 lifetime with `anlage.sleep.Sleeper` attached: every 100 optimiser steps the
genome is refit to the weights (100 warm Adam steps of (5.3), one persistent optimiser) and
the decode cached; on every step the weights are pulled toward the cached decode with
`eta = 0.01 · lr(p)/lr_max` (decoupled decay toward `D`). Same dials as R4. Artifact:
`runs/anlage/sleep-tinystories-s0-anchor-e100-r100-a0.01-dg64H4z16-seed0/sleep.json`.

| quantity | value |
|---|---|
| lifetime test CE (paired with R1 seed 0 = 1.8571) | **2.0021** (+0.1450 nats = +24 `sigma_e`; **K-ANCHOR fires**) |
| refits | 16, 359 s asleep of 737 s (95% overhead) |
| final regression (1,000 steps) residual over keys | 0.394 |
| **at-birth CE of the final decode** | **13.07** (share -0.58; `L_null` 9.01) |

Every third refit (drift = relative residual of the weights against the previous decode
before the refit; residual = after; CE(D) = the cached decode's test CE):

| step | drift | residual | CE(D) |
|---|---|---|---|
| 100 | — | 0.704 | 12.48 |
| 400 | 0.597 | 0.568 | 12.47 |
| 700 | 0.571 | 0.539 | 12.32 |
| 1000 | 0.539 | 0.506 | 13.87 |
| 1300 | 0.506 | 0.475 | 11.62 |
| 1600 | 0.427 | 0.413 | 11.64 |

The weights never become expressible: the gradient moves them away from the decodable set
as fast as the pull draws them toward it (drift stays 0.43–0.65 between refits), the decode
never falls below 11 nats, and the lifetime pays 0.145 nats for being pulled toward it.
Observation 3 of §11.3 is measured: assimilation is a cost, and at this ratio it buys no
birth competence at all. The three trials together (R4, the probe, R4b) close the weight
genome for this loop; §11 records the reading and §12 the design that follows.

## R4 INIT — the scale-profile birth against the free initialisation (§12.6, K-INIT)

`s` (the vector genome: NORM_MIX, NORM_FFN, NORM_FINAL, LOGIT_SCALE) fit to R1's seed-0 adult in 500 steps to
residual 0.0027; the `s`-birth of seed `k` decodes those roles and leaves every
matrix at seed `k`'s innate initialisation (M9). Both arms train 2.5M tokens (10% of the
primordium budget; batch 16), paired by seed and order. Evaluation sets v1 (the run started
before the v2 switch; its noise reference is v1's `sigma_e` = 0.0060).
Artifact: `runs/anlage/init-tinystories-s0/init.json`.

| seed | at birth, free | at birth, `s` | after 2.5M, free | after 2.5M, `s` | Δ |
|---|---|---|---|---|---|
| 1 | 10.2710 | 11.3050 | 2.5360 | 2.5213 | -0.0147 |
| 2 | 10.2659 | 11.2003 | 2.5307 | 2.5151 | -0.0156 |
| 3 | 10.2365 | 11.2375 | 2.5243 | 2.5143 | -0.0100 |
| 4 | 10.2353 | 11.1465 | 2.5212 | 2.5050 | -0.0162 |

Paired: at birth `s` − free = +0.9702 (the inherited gains and logit scale make an untrained
network more confidently wrong); after 2.5M tokens `s` − free = **-0.0141**
[-0.0186, -0.0096] (t),
= -2.3 `sigma_e`. **K-INIT fires** on this screen: the scale profile is a
heritable, scale-free object of a few thousand reals that shortens early learning; its value
at the full budget is measured in R8 (the primordium block of a grown adult carries it too).

## R2 CLASSES — the data-class partition of TinyStories (§12.2)

Rule: length tercile × rarity half per document, edges from the training shard
(length edges 168 / 208 tokens, rarity median 0.0913 with
rank ≥ 1000 as rare). The classified pool holds the same 200,000 windows as R1's plain
pool (digest equal: True). Artifact: `runs/anlage/classes-tinystories/classes.json`.

| c | class | train documents | corpus mixture `mu^corpus_c` (pool windows) | free model CE, fitness (mean of 4 R1 seeds) | seed SD |
|---|---|---|---|---|---|
| 0 | `len_short|rare_lo` | 373,267 | 0.1192 | 1.9480 | 0.0041 |
| 1 | `len_short|rare_hi` | 323,449 | 0.1033 | 2.0339 | 0.0033 |
| 2 | `len_mid|rare_lo` | 335,576 | 0.1374 | 2.0044 | 0.0032 |
| 3 | `len_mid|rare_hi` | 357,206 | 0.1485 | 2.1558 | 0.0054 |
| 4 | `len_long|rare_lo` | 341,067 | 0.2497 | 2.0624 | 0.0033 |
| 5 | `len_long|rare_hi` | 368,924 | 0.2419 | 2.1936 | 0.0054 |

The classes differ in the free model's loss by 0.246 nats (41 `sigma_e`): rare-heavy
stories are harder than common-vocabulary ones and long stories are harder than short. That
the classes are *distinguishable in loss* is necessary for a mixture to matter and is not
sufficient (Aioli, P-9); R3 measures whether moving the mixture moves the outcome.


**On the v2 sets** (`classes.json`; the fitness set's class histogram is now the corpus's, [33, 34, 40, 49, 51, 49]): class means 1.7866, 1.9902, 1.8588, 2.1071, 1.8350, 2.0178; range 0.321 nats.


**On the v3 sets** (`classes.json`; fitness windows per class [39, 31, 35, 44, 58, 49], test [28, 31, 35, 39, 65, 58]): class means 1.6553, 1.9894, 1.7665, 1.9995, 1.7317, 1.9127; range 0.344 nats.

## R3 LIVE — is the mixture axis alive? (§12.4, G-LIVE-mu)

Warm start: R1's seed-0 adult; every arm continues 6.25M tokens (the R1 recipe with its
warm-up restarted, batch 32 × 512) on the classified pool; the centre (`mu^corpus`) is
replicated at 4 order seeds for the protocol's own noise; each `alpha` has 7 Dirichlet
siblings at order seed 0. Artifact: `runs/anlage/live-tinystories-s0/live.json` (arm
checkpoints beside it). **Measured on the v3 evaluation sets** (the v1 and v2 scorings are kept as
`live-v1.json`, `live-v2.json`; the mixture axis was alive by stratum on every version).

**Protocol noise (4 centre replicates):** test 0.0034 nats; per stratum
[0.0043, 0.0079, 0.0213, 0.004, 0.0059, 0.0222, 0.0033, 0.0107, 0.0194, 0.0136, 0.0173, 0.0285]; centre test CE 1.8881, 1.8828, 1.8817, 1.8803 (the
continuation re-warms the learning rate, so all arms sit above R1's 1.8571 — paired, this
cancels).

| `alpha` | sibling test CE range | max |ΔCE_test| vs centre (in protocol σ) | best sibling − centre | max stratum spread / `sigma_e(q)` | max class spread / σ_c | G-LIVE-mu |
|---|---|---|---|---|---|---|
| 200 | 1.8813 – 1.8867 | 0.0068 (2.0) | -0.0068 | 3.9 | 6.0 | alive |
| 50 | 1.8803 – 1.8893 | 0.0078 (2.3) | -0.0078 | 2.9 | 7.8 | not alive |
| 10 | 1.8842 – 1.8971 | 0.0089 (2.6) | -0.0039 | 5.1 | 10.5 | alive |

The `eta` axis (learning-rate multipliers at the corpus mixture):

| arm | Δ test CE vs centre (σ) |
|---|---|
| lr ×0.5 | -0.0870 (-25.4) |
| lr ×2 | +0.0993 (+29.0) |

**Programme regression on all 25 mixture arms** (quadratic surrogate, held-one-out RMSE
0.0026 against a protocol noise of 0.0034); the trust-region step (12.2) gives `mu*`
(on the boundary, distance 0.200, predicted CE 1.9006):

| class | `mu^corpus` | `mu*` | Δ |
|---|---|---|---|
| `len_short|rare_lo` | 0.1192 | 0.2915 | +0.1723 |
| `len_short|rare_hi` | 0.1033 | 0.0836 | -0.0197 |
| `len_mid|rare_lo` | 0.1374 | 0.0460 | -0.0914 |
| `len_mid|rare_hi` | 0.1485 | 0.1133 | -0.0352 |
| `len_long|rare_lo` | 0.2497 | 0.2359 | -0.0137 |
| `len_long|rare_hi` | 0.2419 | 0.2296 | -0.0123 |

R6 tests `mu*` at the full stage-0 budget from scratch, paired by seed (K-MIX).

## R6 MIX — the regressed programme against the corpus mixture (K-MIX)

Four seeds, 25M tokens each from the free initialisation of the seed on the classified pool
(the same windows as R1), the R1 recipe; the arms differ only in the sampling mixture.
Evaluation sets v2; `sigma_e` = 0.0029. Artifact: `runs/anlage/mix-tinystories-s0/mix.json`.

| class | `mu^corpus` | `mu*` (R3's linear surrogate, trust-region step) |
|---|---|---|
| `len_short|rare_lo` | 0.1192 | 0.2915 |
| `len_short|rare_hi` | 0.1033 | 0.0836 |
| `len_mid|rare_lo` | 0.1374 | 0.0460 |
| `len_mid|rare_hi` | 0.1485 | 0.1133 |
| `len_long|rare_lo` | 0.2497 | 0.2359 |
| `len_long|rare_hi` | 0.2419 | 0.2296 |

| seed | corpus | `mu*` | Δ |
|---|---|---|---|
| 0 | 1.8047 | 1.8067 | +0.0020 |
| 1 | 1.8094 | 1.8139 | +0.0045 |
| 2 | 1.8065 | 1.8087 | +0.0022 |
| 3 | 1.8073 | 1.8065 | -0.0008 |

Paired `mu*` − corpus = **+0.0020** [-0.0015, +0.0054] (t) = +0.7 `sigma_e`.
**K-MIX fires**: a mixture step of radius 0.2 from the corpus mixture, chosen by a linear surrogate fit on 25 short continuations, does not improve a 25M-token lifetime on the corpus distribution. On a homogeneous corpus this is the outcome §12.9 recorded as the honest expectation; the Pile lineage (R9) is where the mixture axis has its evidence (P-1, P-2).

## R7 POP — lexicase and the merge against the best sibling and SINGLE (K-POP)

Populations: R3's centre (order seed 0) plus its seven Dirichlet siblings at each `alpha`,
each 6.25M tokens from R1's seed-0 adult. ε-lexicase over the 12 output strata with
`eps_q` = R3's protocol noise per stratum, `n_par` = 4; the merge (7.1)–(7.2) fit on 64
windows disjoint from the fitness set. SINGLE: one continuation of 50,000,000 tokens
(= K × 6.25M, the population's compute) at the corpus mixture from the same adult: test
1.6713. Protocol σ (test) = 0.0034. Artifact: `runs/anlage/pop2-tinystories-s0/pop.json`
(one run per `alpha`: screening).

| `alpha` | parents | accepted | child (test) | best sibling | centre | merge of all 8 | child − best | child − SINGLE | K-POP |
|---|---|---|---|---|---|---|---|---|---|
| 200 | a200 k3, a200 k5, a200 k7 | merge | 1.7879 | 1.8845 (a200 k7) | 1.8881 | 1.7971 | -0.0967 (-28.2σ) | +0.1166 (+34.0σ) | fires |
| 50 | a50 k5, a50 k7 | merge | 1.7879 | 1.8803 (a50 k1) | 1.8881 | 1.7961 | -0.0924 (-27.0σ) | +0.1166 (+34.0σ) | fires |
| 10 | centre s0, a10 k4, a10 k6 | merge | 1.7857 | 1.8842 (a10 k4) | 1.8881 | 1.7964 | -0.0985 (-28.8σ) | +0.1143 (+33.4σ) | fires |

## R5 GROW — unit-prior growth against copy, random and flat (K-GROW)

Stage 0 → stage 1 (`s1`) from R1's seed checkpoints (25M tokens at stage 0), then
20,000,000 tokens at stage 1 with the ramp of Definition 2; `flat` trains stage 1 from scratch
for the FLOP-matched 28,336,560 tokens. Seeds [0, 1, 2]. Artifact: `runs\anlage\grow-tinystories-s0to1\grow.json`.

| arm | test CE per seed | mean |
|---|---|---|
| prior | 1.7389, 1.7440, 1.7366 | 1.7398 |
| copy | 1.7217, 1.7193, 1.7172 | 1.7194 |
| random | 1.7479, 1.7516, 1.7461 | 1.7485 |
| flat | 1.7138, 1.7169, 1.7177 | 1.7161 |

(The verdict with its paired interval is printed by `scripts/anlage_analyse.py`.)

## R8 LOOP — generations of ANLAGE-P against SINGLE-WARM (K-LOOP)

`tokens_total` 50M adult-token-equivalents per generation (`C_total` = 6 · N_adult · 50M), split
by `omega` = (0.15, 0.25, 0.60) over the three TinyStories stages; K = 4 siblings share stage 0's
budget; growth by the unit prior; generation 1 is born from generation 0's adult (its primordium
block) and programme. SINGLE-WARM: the same warm start, growth and compute with one lifetime
at stage 0 and no programme step. One lineage per arm (screening).

| gen | arm | birth (stage-0 block) | stage 0 / 1 / 2 test CE | adult | merge | `mu*` |
|---|---|---|---|---|---|---|
| 0 | K=4 | 10.2955 | 1.9556 / 1.6817 / 1.5983 | **1.5983** | merge | [0.145, 0.02, 0.258, 0.156, 0.303, 0.119] |
| 0 | SINGLE-WARM | 10.2955 | 1.6877 / 1.6448 / 1.5870 | **1.5870** | — | [0.119, 0.103, 0.137, 0.149, 0.25, 0.242] |
| 1 | K=4 | 6.5024 | 1.7713 / 1.6324 / 1.5757 | **1.5757** | merge | [0.094, 0.02, 0.373, 0.09, 0.403, 0.02] |
| 1 | SINGLE-WARM | 5.5007 | 1.6400 / 1.6207 / 1.5693 | **1.5693** | — | [0.119, 0.103, 0.137, 0.149, 0.25, 0.242] |

(K-LOOP's verdict per generation is printed by `scripts/anlage_analyse.py`.)

## Gates and kill criteria

`python scripts/anlage_analyse.py`, 2026-09-02 18:27 local, commit `1c6c4cc`, every number from the artifacts on disk
(evaluation sets v3 from R3's re-evaluation on; R1–R4 also on v1 where stated):

```
G-NOISE      PASS  sigma_e = 0.0029 nats (3 df), mean test CE 1.8092 over 4 seeds
K-BIRTH      INCONCLUSIVE (residual > 0.5 on some role)  birth_share 0.067  at-birth 8.5335  adult 1.8571  shuffle 11.0021  random genome 13.7031  mean residual 0.509  [birth-tinystories-s0-dg64H4z16]
K-SCALE      FIRES  grown genome 2.9162 vs random 1.9931 vs copy 2.7990 (one seed, no gates)
K-GROW       FIRES  prior - copy = +0.0204 over 3 seeds  t [+0.0108, +0.0300]  (fires unless prior better by 3 sigma_e = 0.0088)  [grow-tinystories-s0to1]
H-GROW(2)    INCONCLUSIVE  copy - flat = +0.0033  t [-0.0071, +0.0137]
   prior    1.7398  ['1.7389', '1.7440', '1.7366']
   copy     1.7194  ['1.7217', '1.7193', '1.7172']
   random   1.7485  ['1.7479', '1.7516', '1.7461']
   flat     1.7161  ['1.7138', '1.7169', '1.7177']
K-POP        NOT EVALUABLE -- no population run
K-ANCHOR     FIRES  sleep - free = +0.1450 nats (+49.4 sigma_e, n=1 screening)  at-birth of the final decode 13.07  [sleep-tinystories-s0-anchor-e100-r100-a0.01-dg64H4z16-seed0]
G-LIVE-mu    PASS  alive at alpha: {'200': True, '50': False, '10': True}  max stratum spread 5.1 sigma_e(q) at alpha 10; test spread {'200': 0.0068, '50': 0.0078, '10': 0.0089}  (protocol sigma_test 0.0034)  [live-tinystories-s0]
   G-LIVE-eta lr x0.5: -0.0870 nats (-25.4 sigma of the protocol)
   G-LIVE-eta lr x2: +0.0993 nats (+29.0 sigma of the protocol)
K-MIX        FIRES  star - corpus = +0.0020 over 4 seeds  t [-0.0015, +0.0054]  (fires unless star better by 3 sigma_e = 0.0088)  [mix-tinystories-s0]
K-POP(a200)  FIRES  child - best sibling -0.0967 (-28.2 sigma), child - SINGLE +0.1166 (+34.0 sigma); merge  n=1 screening  [pop2-tinystories-s0]
K-POP(a50)  FIRES  child - best sibling -0.0924 (-27.0 sigma), child - SINGLE +0.1166 (+34.0 sigma); merge  n=1 screening  [pop2-tinystories-s0]
K-POP(a10)  FIRES  child - best sibling -0.0985 (-28.8 sigma), child - SINGLE +0.1143 (+33.4 sigma); merge  n=1 screening  [pop2-tinystories-s0]
K-LOOP(g0)  FIRES  ANLAGE-P adult - SINGLE-WARM adult = +0.0113 nats (+3.8 sigma_e, n=1 screening)  adults 1.5983 vs 1.5870  [gen0-tinystories-K4-copy-s0 vs gen0-tinystories-single-s0]
K-LOOP(g1)  FIRES  ANLAGE-P adult - SINGLE-WARM adult = +0.0065 nats (+2.2 sigma_e, n=1 screening)  adults 1.5757 vs 1.5693  [gen1-tinystories-K4-copy-s0 vs gen1-tinystories-single-s0]
K-INIT       INCONCLUSIVE  s-birth - free after 2,500,000 tokens = -0.0141  t [-0.0186, -0.0096]  (at birth +0.9702); sigma_e of the run's eval version 0.0060  [init-tinystories-s0]
K-SOTA       NOT EVALUABLE -- the Pile lineage has not run
K-BIRTH/K-SCALE are closed by section 11 (fired; the weight genome is not load-bearing)
```

Every criterion of §08.5 and §12.8 that could be evaluated on TinyStories has been evaluated. The
ones that fired are the design's record; K-SOTA and the Pile lineage (R9) have not run.
