# 07 — The population: siblings, selection over strata, crossover by merging

**Status (2026-09-02, after §12):** the siblings' variation is the Dirichlet neighbourhood of the programme's mixture (§12.4) rather than genome mutation; lexicase, the merge (7.1)-(7.2), CRN pairing and S1-S10 are unchanged.

*(Constitutional requirement C2; axiom A4.)* The Darwinian part of ANLAGE lives where
individuals are cheap: at the primordium stage, among siblings born from one genome. Their
fitness is case-based over held-out strata so that specialists survive; their crossover is
weight-space merging, which is valid because they share a birth; the merged elite is what
grows into the adult. Every claim in this file has a control and a kill criterion.

## 7.1 Why here, and why this

**[RATIONALE]** Three facts set the design. (i) SOMA and GNOMON put the population where
individuals were either meaningless (rank-one units) or too expensive (whole lifetimes), and
the population contributed 0.04% and one improvement in 432 evaluations respectively (§00).
(ii) Every population-like pretraining method that works at scale is merge-based — model
soups, Branch-Train-Merge, DiLoCo — and merging works when the branches share an ancestor
(linear mode connectivity is established within the first few percent of training, §10 E-6,
E-17). (iii) Case-based selection preserves specialists that an averaged loss deletes (§10
E-18), and a language corpus has natural cases: its domains, and — label-free — its
positional and lexical strata.

## 7.2 Siblings

**[NORMATIVE]** Generation `g` at stage 0 runs `K` siblings (`K = 4` by default). Sibling `k`
is defined by:

```
    Gamma_k  =  MUTATE(Gamma_g, sigma_mut)          for k ≥ 1;   Gamma_0 = Gamma_g   (no mutation)
    W_{k,0}  =  eps · W_prev,0  +  (1 - eps) · D(Gamma_k, Z_g, 0)                    (5.4)
    order_k  =  derive_seed(seed_g, "member") ⊕ k    the data-order seed
    mu_k     =  the corpus mixture (TinyStories) | Dirichlet(alpha_mix · mu_corpus) (Pile)
    W_k      =  LIFETIME(W_{k,0}, s = 0, C_sib, order_k, mu_k)                        §06 / §04
```

with `C_sib = f_0 C_total / K` FLOPs each, `f_0` the stage-0 fraction of §06.5. At generation
0 there is no genome, so siblings differ by data order (and mixture) only.

**[NORMATIVE — the CRN contract for siblings]** All siblings of a generation draw from one
window pool (`screen/crn.py::WindowPool`, `pool_seed = seed_g`); sibling `k`'s *order* differs.
When two *treatments* are compared (mutation on/off, sampling on/off), sibling `k` of each arm
uses `order_k`, so the contrast is paired and the population's own diversity is not confused
with the treatment's effect.

## 7.3 Mutation

**[NORMATIVE]** `MUTATE` perturbs the genome's feature-map matrices `W_1, W_1^tok, W_1b, W_2`
and every `O_m^{(h)}` with scale-relative, **low-rank** noise, and its vectors with Gaussian
noise:

```
    matrices:  t' = t + sigma_mut · rms(t) · (A B^T) / sqrt(r),   A ~ N(0, I_{m x r}), B ~ N(0, I_{n x r}),  r = 4
    vectors:   t' = t + sigma_mut · max(rms(t), 0.1) · N(0, I)
```

A low-rank perturbation of the kernel moves every decoded weight coherently — the structured
variation weight-space noise cannot provide (EGGROLL's form, §10 E-7). `sigma_mut` is set by
**G-LIVE** (§7.6): the smallest value on the grid `{0.02, 0.05, 0.1, 0.2}` at which the
at-birth CE across 16 mutants has standard deviation ≥ `2 sigma_e`; if none does, mutation is
reported inert and siblings differ by data order only.

## 7.4 Selection and crossover

**[NORMATIVE — cases]** The cases `C` are strata of the **fitness split** (never the test
split):

| lineage | strata | count |
|---|---|---|
| TinyStories | position-in-document bucket `{[0,32), [32,128), [128,256), [256,∞)}` × target-frequency class `{rank < 100, 100 ≤ rank < 1000, rank ≥ 1000}` (the upper position edge is 256, not 512: measured on `data/ts_8192`, no story reaches position 512 and an edge at 512 left a bucket empty) | 12 |
| Pile | source domain (the 22 Pile subsets), with the same 12 position × frequency strata reported as diagnostics | 22 |

A sibling's score on case `c` is its mean held-out CE over the tokens of that stratum,
`F_k(c)`. Lower is better throughout. The strata are label-free on TinyStories and measure
distinct competences: context use (position) and lexical coverage (frequency).

**[NORMATIVE — ε-lexicase]** `soma/selection/lexicase.py::epsilon_lexicase` with
`nu_{k,c} = -F_k(c)` (so that "higher is better" inside the vendored operator), case weights
uniform, and **`eps_c = sigma_e(c)`**, the seed noise of that stratum measured by §08 R1 —
not a MAD, which at `K = 4` is meaningless. `n_par = K/2` parents are drawn (a multiset;
Lemma 4.1 and 4.2 of `docs/spec/04` apply unchanged). A sibling within `sigma_e(c)` of the
best on every case it is filtered on survives; a specialist on one stratum survives the
orderings that put that stratum first.

**[NORMATIVE — crossover = merging]** With `P` the distinct parents and `W_0` the common birth
weights (5.4) of the generation:

```
    W_x  =  W_0  +  sum_{p in P} c_{p,l,m} ( W_p,{l,m} - W_0,{l,m} )     per role and layer      (7.1)
```

The coefficients `c in R^{|P| x |roles x layers|}` are fit by Adam (lr 0.05, 200 steps,
`c` initialised at `1/|P|`) to minimise CE on 64 windows of the fitness split disjoint from
the windows on which `F_k(c)` was scored; then the merge is accepted iff

```
    CE_fit(W_x)  ≤  min_k CE_fit(W_k)  +  sigma_e                                                 (7.2)
```

and otherwise `W_x <- W_{argmin_k CE_fit(W_k)}` and the rejection is recorded. (7.1) is a merge
of *task vectors* relative to the common birth (§10 E-17); a TIES-style sign-consensus merge
is a registered ablation. The merge can never lose to the best sibling by more than `sigma_e`
by construction of (7.2), so K-POP is decided by whether it *wins*.

## 7.5 The compute split, and what the population is compared against

**[NORMATIVE]** `C_pop = K C_sib = f_0 C_total`; the merged elite is then grown and trained
on `C_total - C_pop`. The control **SINGLE** spends the same `f_0 C_total` at stage 0 on one
lineage (`K = 1`), then grows and trains identically. K-POP asks whether the population arm
beats SINGLE at matched `C_total` by `≥ 3 sigma_e` **and** whether the merge beats the best
sibling; if only the second holds, selection is doing work and the split is re-examined; if
neither, the population is decorative and removed.

`f_0 = 0.20` and `K = 4` initially. `f_0` is the dial C2 names (§01.4); it is raised only on
K-POP's evidence.

## 7.6 G-LIVE and G-SEARCHWORTHY for the diversity axes

**[NORMATIVE]** Before a generation is run, each axis of sibling diversity is screened
(`screen/variance.py::decompose`, `searchworthy`):

| axis | live if | searchworthy if |
|---|---|---|
| data order | (always live) | — |
| genome mutation | at-birth CE spread ≥ `2 sigma_e` at some `sigma_mut` on the grid | the sibling landscape over `sigma_mut` is not monotone |
| domain mixture (Pile) | per-domain CE moves ≥ `3 sigma_e(c)` between two mixtures | — |
| growth timing | a `±25%` shift of `C_0` moves adult CE ≥ `3 sigma_e` | K-GROW has passed |

An axis that is not live is not varied, and the report says so.

## 7.7 Acceptance tests

**[NORMATIVE]** `anlage/population.py` is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| S1 | CRN pairing | sibling `k` of two arms consumes identical windows in identical order; a mutant that unpairs them fails |
| S2 | mutation is low-rank and scale-relative | the perturbation of each matrix has rank `≤ r` and RMS `sigma_mut · rms(t)` within 10%; vectors are Gaussian at the floor |
| S3 | strata | every fitness-split token is assigned to exactly one stratum; the 12 TinyStories strata are non-empty at `V = 8192`; per-stratum CE of a model equals a hand computation on a small batch |
| S4 | lexicase | with `eps_c = sigma_e(c)`, a planted specialist (best on one stratum by `3 sigma_e`, worst on the rest) is selected in at least one of 200 draws; under mean-CE selection it never is (control) |
| S5 | merge algebra | (7.1) with `c = 1/|P|` equals the plain average of the parents; with a one-hot `c` equals that parent |
| S6 | merge acceptance | a merge worse than the best sibling by more than `sigma_e` is rejected and recorded; one within `sigma_e` is accepted |
| S7 | coefficient fit | on two planted parents whose task vectors are orthogonal, the fit recovers coefficients within `0.05` of the optimum |
| S8 | disjoint windows | the coefficient-fit windows and the scoring windows of the fitness split do not overlap; neither overlaps the test split |
| S9 | budget | `K C_sib = f_0 C_total` to within one batch; SINGLE spends the same |
| S10 | determinism | a generation's population phase is reproducible bit-for-bit from `seed_g` |
