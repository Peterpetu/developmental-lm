# SOMA — Specification 02: The Weismann Barrier, Germline, and Soma

**Spec version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Normative for `soma/germline/`.
Serves constitutional requirement **C1**.

---

## 2.1 The barrier

**[NORMATIVE]** Parameters are partitioned into two disjoint sets with different write
permissions, enforced in code.

### 2.1.1 Exhaustive membership

**[NORMATIVE]** This table is exhaustive. Any tensor not listed is a specification defect.

| Object | Side | Shape | Rationale |
|---|---|---|---|
| `mu_g` | **germline** | `N_G x m_dim` | gene field centres |
| `log_sigma_g` | **germline** | `N_G` | gene field widths |
| `policy` | **germline** | `n_policy` reals | §2.2.3 |
| `Inventory` | **germline** | `<= J_max` gene sets + stats | §2.2.4 |
| `g_u` (per unit) | **germline-addressed** | `k_g` indices | discrete, selection-only |
| `slot_u`, `live_u`, `dying_u` | bookkeeping | — | grid occupancy |
| `A`, `B`, `kappa` | soma | `N_G x d_model`, `d_model x N_G`, `N_G x d_key` | the parts themselves |
| `W_shared`, `W_key`, `W_ctx`, `W_pos` | soma | per Site | projections |
| `K_unit1`, `K_unit2` | **soma** | `side x d_key/2` | unit routing addresses |
| `p_u` | soma | `U_max x m_dim` | unit positions |
| mixer, attention, embeddings, norms | soma | §05 | phenotype |
| `m_u`, `age_u`, `nu_u` | bookkeeping | — | selection state |

**[NORMATIVE]** `K_unit1, K_unit2` are **soma**. Were they germline, `|Gamma|` would scale
with `U_max` and §09.3's bottleneck claim (9.1) would be false by construction, firing K9
definitionally without any experiment.

### 2.1.2 Enforcement

```
Algorithm ASSERT_BARRIER(GermlineRegistry, optimizer)
  # called immediately before every optimizer step
  for tensor in GermlineRegistry:
      assert tensor.requires_grad is False
      assert tensor.grad is None
  assert GermlineRegistry.ids() & optimizer.param_ids() == empty
```

A violation raises. **[NORMATIVE]** The guard is verified by a **mutation test**:
`tests/unit/test_weismann_barrier.py` monkeypatches `ASSERT_BARRIER` to a no-op and asserts
that a deliberately gradient-enabled germline tensor then goes undetected — i.e. the test
proves the guard is load-bearing, not merely that it currently passes.

### 2.1.3 Why the barrier is forced

**[RATIONALE]** Two independent facts, plus one positive reason.

*(a) Population search cannot afford the phenotype.* Measured cost of hyperscale ES on
phenotype weights is approximately **180× the GPU-hours of a backprop baseline**, reaching
3.40 bits/byte against 3.58 for a plain-SGD control — a worse-than-Adam result at two orders
of magnitude more compute. A `1e4`–`1e5`-dimensional germline is the only affordable regime.

*(b) The germline's operations admit no gradient.* Birth, death, duplication, gene-set
mutation and recombination are discrete set operations.

*(c) ES selects for the right thing.* ES optimises the perturbation-smoothed objective
`E_xi[ F(theta + sigma xi) ]`, i.e. for parameters **robust to perturbation** — which is
canalization. No gradient method optimises for this. ES on the germline is therefore not a
fallback; its implicit bias is the property a genome should have.

---

## 2.2 Germline contents

```
  Gamma = ( Field , Policy , Inventory )
```

**[NORMATIVE — v1 correction]** Spec v1 listed a fourth component, `Library`, holding
gene-side product-key tables `K_gene1, K_gene2`. **These are deleted.** Nothing ever issued
a query against them: `g_u` is created by `BIRTH` and modified by `MUTATE` as a *set*, never
by retrieval. They were dead parameters consuming ES budget. `neighbour(g)` is now defined
geometrically (§2.2.2), which uses structure that already exists.

### 2.2.1 `Field` — the expression map

`mu` in `R^{N_G x m_dim}` (renormalised to the unit sphere after every germline update),
`log_sigma` in `R^{N_G}`. Cost `(m_dim + 1) N_G` reals, independent of `|U|`, `d_model`,
`L`.

**[NORMATIVE — initialisation]** Constants and measured moments are in §01.2.
```
  mu_g        ~ uniform on S^{m_dim-1}          m_dim = 8
  log_sigma_g  = log(1.2)          # measured median phi = 0.4995 (v2 used 0.6 -> 0.062)
  p_u          ~ uniform on S^{m_dim-1}
  g_u          = k_g indices drawn uniformly without replacement from {0..N_G-1}
```
Spec v1 specified none of these while requiring `N_eff` to be estimated "from the initial
position distribution".

### 2.2.2 Gene adjacency

**[NORMATIVE]** `neighbour(g)` returns a uniformly random element of the `n_nbr = 8` genes
with smallest `|| mu_g' - mu_g ||` (excluding `g`). It is well defined for every `g` and
requires no extra parameters.

**[RATIONALE — what geometric locality does and does not buy]** v2 claimed this "makes a
point mutation a small change rather than an arbitrary one". That is **false**: `neighbour(g)`
shares only the RBF centre, while the *content* `(a_g, b_g, kappa_g)` of the neighbouring gene
is an independent random draw, so swapping `g -> neighbour(g)` replaces one rank-one part
with an unrelated one at a similar expression coefficient — a maximally arbitrary functional
change. What locality actually buys is that the mutated unit's **expression profile** moves
little, so its *routing* and its metabolic cost are approximately preserved while its
*function* changes. That is a meaningful and testable property (it keeps the clade compact
under `L_wire`), and it is the only one claimed.

### 2.2.3 `Policy` — the canonical constant table

**[NORMATIVE]** This is the single canonical table. §03, §04 and §06 cite it and do not
restate it. Every constant is stored as an **unconstrained real** and mapped by a fixed
link; ES perturbs the unconstrained reals.

| Constant | Link | Range | M-nano | M1 |
|---|---|---|---|---|
| `lambda_wire` | softplus | `> 0` | 1e-3 | 1e-3 |
| `lambda_disp` | softplus | `> 0` | 1e-3 | 1e-3 |
| `lambda_size` | softplus | `> 0` | 1e-6 | 1e-6 |
| `rho_0` | affine-sigmoid | `[0, 0.2]` | 0.05 | 0.02 |
| `n_c / N_step` | affine-sigmoid | `[0.10, 0.50]` | 0.25 | 0.25 |
| `M_mat` | round∘softplus | `N` | 100 | 400 |
| `sigma_pos` | exp | `> 0` | 0.10 | 0.10 |
| `sigma_mut_0` | exp | `> 0` | 0.05 | 0.05 |
| `(p_point, p_dup, p_del)` | **softmax** | simplex | (0.5,0.3,0.2) | (0.5,0.3,0.2) |
| `p_rec` | affine-sigmoid | `[0, 0.5]` | 0.10 | 0.10 |
| `d_rate` | affine-sigmoid | `[0.02, 0.25]` | 0.10 | 0.10 |
| `k_tok` | affine-sigmoid | `[0.55, 1.0]` | 0.65 | 0.65 |
| `f_oe` | affine-sigmoid | `[0, 0.5]` | 0.25 | 0.25 |
| `U_cap` | round∘softplus | `N` | 240 | 960 |
| `n_par` | round∘softplus | `N` | 8 | 24 |
| `cap_off` | **derived**, §04.6 | `N` | 6 | 10 |
| `eta_ema` | affine-sigmoid | `[0.9, 0.999]` | 0.99 | 0.99 |
| `g_growth` | affine-sigmoid | `[2, 5]` | 4 | 4 |

**[NORMATIVE — link definitions]** Spec v1 mandated `sigmoid`/`softplus` only, which cannot
express a simplex, an integer, or an interval other than `(0,1)`, so its own postcondition
was false.

**[NORMATIVE — every default must be interior to its range]** `affine-sigmoid` maps onto the
**open** interval `(lo, hi)`, and `softplus` onto `(0, inf)`; neither attains an endpoint. A
default equal to an endpoint therefore has **no preimage**, so the germline cannot be
initialised to the table's own value. Config validation asserts that every default is strictly
interior.

v3 set `g_growth`'s range to `[2, 4]` with a default of `4` — the endpoint — so
`Policy.from_constants` could not invert it and generation 0 could not be constructed at all.
The range is widened to `[2, 5]`, keeping the documented default interior. This defect was
found by **running the code**, not by reading the specification, which is what §AUDIT Part D
predicted would happen and why the deferral of the remaining `[OPEN]` items to measurement is
the right disposition rather than a fourth reading pass.

**[NORMATIVE]** `lambda_disp` (1.9) is a germline policy constant, not a hard-coded literal:
it is the coefficient of the dispersion term that prevents the positional collapse the sphere
constraint does not (audit A-9), so ES must be able to tune it exactly as it tunes
`lambda_wire`.
```
  softplus(z)            = log(1 + e^z)
  affine-sigmoid(z;lo,hi)= lo + (hi - lo) * sigmoid(z)
  softmax(z)             for simplex-valued constants
  round∘softplus(z)      = max(1, round(softplus(z)))       for integer constants
```
**[NORMATIVE]** The `ES_STEP` postcondition holds *after rounding*.

**[NORMATIVE — ablation override]** Ablations set a constant to an endpoint value that the
links cannot reach (`lambda_wire = 0`, `k_tok = 1.0`, `f_oe = 0`, `p_rec = 0`,
`rho_0 = 0`). A config path `ablation.pin = {name: value}` **bypasses the germline
parameterisation entirely** for the named constants and **excludes them from `E_i`**, so the
intervention is clean and the main arm cannot drift into the ablated condition. Spec v1's
claim that "selection may drive `rho_0` to zero" was unreachable under a sigmoid and is
withdrawn.

`phi_ann`, `T_0`, `T_arch`, `T_cert`, `q_cases`, `cap_off`, `epsilon_bif`, `tau_death` are
**derived**, not policy: see §03.3, (1.17)–(1.19), §01.10, §04.6, §01.7.2, §04.4.

**[NORMATIVE — germline-scale constants are not policy]** `N_G = 2048`, `k_g = 8`,
`m_dim = 8`, `J_max = 256` are fixed at every configuration (§01.2), are **not** perturbed by
ES, and do not appear in this table. v2 varied `N_G` between configurations, which made the
M-nano → M1 germline inheritance of §2.6.2 shape-impossible.

### 2.2.4 `Inventory` — discovered archetypes

`{ (g^(j), stats_j) }`, `|Inventory| <= J_max = 256`. `stats` records generations survived,
mean utility, mean clade size. Eviction by lowest generations-survived, ties broken by
lowest mean utility.

**[DERIVED]** The Inventory contributes `J_max * k_g` discrete indices plus statistics to
`|Gamma|`, so (9.1) is corrected to include an `O(J_max k_g)` term. It remains independent
of `|U|`, `d_model` and `L`.

```
Algorithm SEED_POPULATION(Inventory, |U|_0, side) -> U
  precondition: |U|_0 <= U_max
  1. slots <- |U|_0 slots chosen by a low-discrepancy sweep of the side x side grid
  2. for each slot:
       if Inventory is non-empty:
           g_u <- gene set of an archetype sampled with probability proportional to
                  its generations-survived, then MUTATE'd once
       else:
           g_u <- k_g indices uniform without replacement            # generation 0, and A4
       p_u  <- uniform on S^{m_dim-1}
       m_u  <- 1        # seeded units are mature; only BIRTH-created units anneal
       age_u<- M_mat+1  # immediately eligible: §04.2's U_death is `age_u > M_mat`
       nu_u <- 0
  postcondition: |U_live| = |U|_0 ; every live unit has |g_u| = k_g
```

**[NORMATIVE]** Seeded units start **mature** (`m_u = 1`). Were they to start at `m_u = 0`,
the model would compute nothing at step 0 and "birth performance" would be undefined rather
than merely low.

---

## 2.3 Serialisation

```
  runs/genomes/<lineage_id>/<G_gen>.germline.safetensors
  runs/genomes/<lineage_id>/<G_gen>.meta.json
```

`meta.json` (all fields required): `schema_version`, `lineage_id`, `G_gen`, `parent_hash`,
`content_hash` (BLAKE2b-256 over tensor bytes in sorted-key order), `config`, `metrics`
(`birth_score`, `n_crit`, `early_slope`, `final_ce`, `fitness`), `env`
(`python/torch/cuda/gpu/driver`), `seeds` (§2.6.3), `created_utc`.

**[NORMATIVE]** A germline whose recomputed hash differs from `content_hash` is **refused at
load**.

---

## 2.4 The position seam

**[NORMATIVE]** `p_u` is **soma** while `g_u` is germline-addressed. A unit's *identity*
(which genes it expresses) is inherited and discrete; its *placement* is learned and
continuous. Because `phi_g` is smooth in `p` (§01.2), the gradient exists; because gene
membership is a set, no gradient is needed for it.

**[DERIVED]** Two units with identical gene sets but different positions express those genes
with different weights, so duplication followed by gradient descent produces *differentiated*
siblings without explicit symmetry-breaking noise on the parts.

---

## 2.5 Initialisation acceptance test

**[NORMATIVE]** Executed **after** phenotype construction (spec v1 made it a precondition of
`LIFETIME` while it depends on tensors `LIFETIME` creates at step 2). Failure aborts.

1. **Variance test.** Draw `x ~ N(0, I_{d_model})` rescaled to `||x|| = sqrt(d_model)`.
   For `N_probe = 4096` sampled units compute the realised Site output variance. Assert
   `0.5 <= Var(y)/Var(y_ref) <= 2.0`, where `y_ref` uses `W ~ N(0, 2/d_model)` (He normal).
2. **Gradient-flow test.** The **median per-block ratio of the residual-stream gradient
   norm** must lie in `[0.60, 1.70]` at step 0. The total first-to-last ratio and the full
   per-block series are reported alongside.

   **[NORMATIVE — v3.1 correction]** v3.0 stated "ratio of gradient norms between the first
   and last block in `[0.1, 10]`", which is wrong in two independent ways.

   *It compared unlike quantities.* Taken over **parameter** gradients, block 0 is a mixer
   (`1.32 M` parameters at M1) and block `L-1` is attention (`0.65 M`), so the ratio conflates
   parameter count with gradient flow. The residual-stream activation gradient is the quantity
   "gradient flow" names, and it is parameter-count independent.

   *It was not depth-invariant.* A band on the **total** ratio across the stack cannot hold at
   two depths: the same healthy initialisation measured `13` at `L = 6` and `24` at `L = 12`,
   and before the §05.6 fix, `304` and `799`. Stating the criterion per block makes it
   comparable across configurations.

   The **median** is used rather than the geometric mean because block 0 is *legitimately*
   larger than the rest — two sublayers each add RMS 1 to a stream at RMS 1, so the residual
   grows by `sqrt(3)` there and by less thereafter — and with few blocks that one legitimate
   factor dominates a geometric mean (`1.68` at `L = 6` against `1.34` at `L = 12` for the same
   model). The median is robust to it: `1.38` at M-nano, `1.21` at M1.
3. **`N_eff`/`E2` cross-check.** Estimate with `N_probe = 4096` at init, then **independently**
   with `N_probe = 16384` at test time; assert agreement within 10%. Spec v1 compared an
   estimator against itself, which is vacuous.
4. **Guards.** `N_eff >= 1`, `E2 >= 1e-4` (§01.5).
5. **`gain_act` calibration.** Measure the variance gain of `relu(.)^2` numerically on the
   realised pre-activation distribution and record it in the run manifest.
6. **Barrier test.** `ASSERT_BARRIER` after one forward/backward on random data.

**[RATIONALE]** A mis-initialised weight-generating network is reported to fail to train at
any learning rate, with one variant showing a ~270-epoch dead zone, and batch normalization
corrects vanishing but **not** exploding decoded weights. The check happens before training.

---

## 2.6 The phylogeny rate: ES on the germline

**[NORMATIVE]** Between lifetimes, the germline's **continuous** components (`mu`,
`log_sigma`, and the unconstrained policy reals) are updated by **single-point evolution
strategies with mirrored sampling**. The **discrete** components (`Inventory` membership and
archetype gene sets) are updated by selection and recombination, not by ES.

**[NORMATIVE — v1 correction]** Spec v1 described a single `Gamma` updated by `ES_STEP`
(§2.6) while §07.5 described "germlines drawn from the current population" and "update the
germline population" — two different algorithms. Single-point ES is normative; §07.5 is
corrected to match, and the serialisation path `<G_gen>.germline.safetensors` (one file per
generation) is consistent with it.

### 2.6.1 The step

```
Algorithm ES_STEP(Gamma, P_es, sigma_es, alpha_es, r_rank) -> Gamma'
  precondition: P_es even and >= 2 ; sigma_es > 0 ; r_rank >= 1

  for i in 1 .. P_es/2:                                   # mirrored pairs
      for each germline tensor M:
          if M is 2-D of shape (p,q):
              A_i ~ N(0, I_{p x r_rank}) ; B_i ~ N(0, I_{q x r_rank})
              E_i[M] <- (1/sqrt(r_rank)) A_i B_i^T                              (2.1)
          else:                                            # 1-D or scalar
              E_i[M] ~ N(0, I_shape(M))                                          (2.2)
      f_{2i-1} <- FITNESS( LIFETIME( Gamma + sigma_es * E_i , seed_{2i-1} ) )
      f_{2i}   <- FITNESS( LIFETIME( Gamma - sigma_es * E_i , seed_{2i}   ) )

  f_hat <- RANK_NORMALISE(f_1 .. f_{P_es})
  for each germline tensor M:
      M <- M + (alpha_es / P_es) * sum_i f_hat_i * E_i^{(+/-)}[M]                (2.3)
  renormalise mu rows to the unit sphere
  postcondition: every policy constant, after its link and rounding, lies in its range
```

**[NORMATIVE — at `P_es = 2` the update uses the centred fitness difference, not ranks]**
With two members, `RANK_NORMALISE` returns exactly `{-0.5, +0.5}`, so the update reduces to
`M <- M +/- (alpha_es/2) E` — a fixed-magnitude step whose only information is one sign bit,
which is precisely what v2 criticised v1 for. Rank normalisation discards the magnitude that
mirroring exists to provide. Therefore:

```
  if P_es == 2:   f_hat_1 = -f_hat_2 = clip( (f_+ - f_-) / (2 * sigma_f), -1, +1 )
  else:           f_hat   = RANK_NORMALISE(f_1..f_{P_es})
```

with `sigma_f` a running standard deviation of the fitness over the last 8 generations
(initialised to `1`). This keeps the estimator scale-free while retaining magnitude.

**[NORMATIVE]** 1-D germline tensors (`log_sigma`, the policy vector) use full-rank
Gaussian perturbation (2.2); the low-rank form (2.1) applies only to 2-D tensors. Spec v1
wrote `N(0, I_{p x r})` for all tensors, which is undefined for a vector.

```
RANK_NORMALISE(f_1..f_P):
  r_i <- average rank of f_i in ascending order, ties averaged
  return  (r_i - 1)/(P - 1) - 0.5                       in [-0.5, +0.5]
```

**[DERIVED]** The low-rank estimator converges to the full-rank ES update at `O(1/r_rank)`,
so `r_rank = 1` is usable; perturbation storage falls from `O(pq)` to `O(r(p+q))`.
**[NORMATIVE]** `r_rank = 1` by default.

### 2.6.2 Generation arithmetic

**[NORMATIVE]** `G_gen = floor( T_run / (P_es * T_life) )`.

**[NORMATIVE — v1 correction]** Spec v1 stated `G = 9..18` in §02.6 and `G = 20..40` in
§07.5 for the same inputs; neither matched `floor(720/(2*T_life))`, and the kill criterion
demanded `G >= 20`, which was unreachable at M1 lifetimes. The repair is the M-nano
configuration:

**[OPEN — `T_life` is measured, not asserted]** §01.9 marks `T_life` `[OPEN]` at both
configurations: v2's M1 figure of 40 min implies 47% of this card's dense bf16 peak from the
`6ND` term alone, before attention, routing, sparse gathers, `float32` triangular solves,
held-out evaluation, `erank` SVDs, certification and archive decoding. The table below is
therefore an **illustration of the arithmetic**, not a claim; `G_gen` is recomputed from the
`T_life` measured at gate `G-M0` before the decisive experiment is scheduled.

| Config | illustrative `T_life` | `P_es` | `G_gen` in a 12 h run |
|---|---|---|---|
| M-nano | 1.5 min | 2 | `floor(720 / 3)` = **240** |
| M1 | 40 min | 2 | `floor(720 / 80)` = 9 |

At M-nano, `G_gen >= 20` for three seeds **and** the A4 control fits comfortably. At M1,
`G_gen = 1` in practice and the germline is **inherited** from the M-nano lineage — which is
the transfer claim §07.5 tests.

### 2.6.3 Random streams

**[NORMATIVE]** One generator per purpose, all seeded from the run seed and all recorded in
`meta.json`: `data`, `init`, `dropout`, `mutation`, `lexicase`, `archive`, `es_perturb`,
`member`, `downsample`, `growth`, `explore`. Spec v1 mandated separation by purpose in one
bullet and then put lexicase and archive in a shared generator in the next.

**[NORMATIVE — the one draw that cannot take a generator; v3.1, audit A-64]** Every draw in
SOMA takes an **explicit** generator, which is what makes the streams independent under
refactoring — with one unavoidable exception. `nn.Linear.reset_parameters`,
`nn.Embedding.reset_parameters` and `nn.init.normal_` accept no `generator` argument in
torch 2.7, so **module construction draws from the process-global default generator**, which
`RngBundle` by design never touches.

`LIFETIME` therefore seeds the global CPU and CUDA generators from
`derive_seed(run_seed, "init")` immediately before the model is constructed, and records the
value in `meta.json`. Without it the initial weights were a function of whatever the
process's global stream happened to hold: two invocations of the same lifetime with the same
`seed` produced **different models**, measured as §02.5 acceptance statistics of
`var_ratio = 0.640 / 0.635` and `grad_ratio = 12.739 / 13.045` on back-to-back identical
runs. §07.7's reproducibility claim and every seed-paired comparison in §08.6 rest on this
call, and its regression test carries a **positive control** — the same construction with the
seeding removed must differ — because the assertion "two runs agree" passes trivially in a
process whose global stream has not moved.

`dropout` is reserved and currently unused: SOMA applies no dropout (§05.1), and the stream
exists so that adding it later cannot shift any other stream's trajectory.

### 2.6.4 Fitness

`FITNESS` is (7.1). **[NORMATIVE]** It is weighted toward **adaptation efficiency**
(time-to-criterion and early slope), not final loss, per §01.7.2 and the finding recorded in
`AUDIT-v1.md` A-2 that `BirthScore` is ~0 by construction for a pure-scaffold germline.

---

## 2.7 Falsification

- If a germline evolved for `G_gen` generations does not improve **adaptation efficiency**
  over a fresh random germline (ablation A4), the phylogeny rate is doing nothing and C1 is
  unmet (K1).
- If `BirthScore` is **significantly non-zero**, information is leaking across the barrier;
  this is a **bug signal**, not a success (§07.3).
- If `N_G` must grow with `|U|` or `d_model` to maintain quality, (9.1) fails and the
  architecture is a hypernetwork inheriting the known ceiling (K9).
- If the barrier guard can be removed without any test failing, the partition is convention
  rather than architecture (§2.1.2 mutation test).
