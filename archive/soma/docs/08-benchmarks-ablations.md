# SOMA — Specification 08: Benchmarks, Ablations, and Reporting Discipline

**Spec version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Normative for `soma/eval/`.

---

## 8.1 Metric definitions

| Metric | Definition |
|---|---|
| `CE` | mean next-token cross-entropy in nats, held-out, `k_tok = 1`, natural text only |
| `BPB` | `(CE / ln 2) * (tokens / bytes)` |
| `PPL` | `exp(CE)` — **reporting only, never a decision basis** |
| `n_crit` | first evaluation step with `CE <= L_star`, right-censored (§07.1) |
| `early_slope` | slope of `CE` vs `log(tokens)` over `[0.02, 0.20] * N_step` |
| `late_slope` | slope of `CE` vs `log(tokens)` over the final 50% |
| `erank` | (3.9) on the fixed probe batch |
| `Q_route` | §8.1.1 |
| `pass_at_k` | §8.1.2 |
| `scaled_acc` | `(acc - acc_random) / (1 - acc_random)` |

**[NORMATIVE]** Every reported curve is **held-out `L_task` with `k_tok = 1` on natural text
only**, with structural events annotated. The training objective (which sums `L_met` and
normalises over selected tokens) is never plotted as a loss curve.

### 8.1.1 `Q_route` — fully defined

**[NORMATIVE — v1 correction]** Spec v1 named `Q_route` in a metric table and defined it
nowhere, while making it the **sole** quantity in K4 and one of three in K3.

- **Graph.** Bipartite, per Site `l`: left nodes are the `2048` tokens of the fixed probe
  batch, right nodes are the live units of `U_l`.
- **Edges.** Weight `w(t,u) = r_ut` for `u in R_t`; edges with `r_ut < 1e-3` are dropped.
- **Modularity.** **Barber bipartite modularity** `Q_B` (the bipartite null model; Newman's
  `Q` is inappropriate for a bipartite graph and its range is not `[0,1]`).
- **Communities.** **BRIM** (Barber's bipartite recursively induced modules), seed fixed at
  `20260828`, 10 restarts per community count over the fixed grid `c in {2,4,8,16,32}`, best
  `Q_B` retained.

**[NORMATIVE — v3.1 correction, the community algorithm]** v3.0 specified **Leiden** at
resolution 1.0. Leiden optimises **Newman** modularity (or CPM), not Barber's bipartite null
model, so finding communities with Leiden and then *scoring* them with `Q_B` systematically
under-reports the quantity K4 is adjudicated on — at a margin of `0.02`, on a metric whose
range is `[-1,1]`. The algorithm Barber introduced **for** `Q_B` is BRIM, which maximises it
directly: with `B~ = A - k d^T / m`, `Q_B = (1/m) tr(R^T B~ T)`, and holding either indicator
matrix fixed the other's optimum is an exact `argmax`, so the alternation is monotone in `Q_B`.
The restart discipline is unchanged, and extended over the community count because BRIM takes
`c` as an input where Leiden infers it; the grid is fixed and is part of the metric's
definition, so two runs remain comparable.

This also removes a dependency (`igraph` / `leidenalg`) that does not build cleanly on this
platform, which R8 records as a standing toolchain risk.
- **Aggregation.** Computed per Site and **averaged over the `L` Sites**.
- **Range.** `Q_B` in `[-1, 1]`. §01.9's earlier `[0,1]` was wrong.

### 8.1.2 `pass_at_k` — corrected

**[NORMATIVE]** With `n = 128` samples per instance at temperature `0.8` and `c` correct:

```
  pass_at_k = 1 - C(n - c, k) / C(n, k)                                          (8.1)
```

evaluated on the grid `k in {1, 2, 4, 8, 16, 32, 64, 128}`.

**[NORMATIVE — v1 correction]** Spec v1 defined `pass@k` as "fraction of `k` samples solving
a verified task", which is the per-sample success rate — **independent of `k`**. Under that
definition the `k`-curve is flat and §8.6 rule 1's entire argument (base model wins at large
`k`) would be unobservable.

---

## 8.2 Behavioural regression suite

**[NORMATIVE]** Runs from M1 onward at every checkpoint. Deterministic; gates every
capability claim. All tasks are serialised over the reserved symbol block (§06.3.2), and all
evaluation lengths lie **outside** the archive's admissible range (§06.3.5).

| Suite | Train lengths | Eval lengths | `acc_random` | Pass bar (scaled) |
|---|---|---|---|---|
| `PARITY` | 3–32 | 40–256 | 0.5 | `>= 0.95` at 256 |
| `MODCOUNT` (`m_mod` 3, 5) | 3–32 | 40–256 | `1/m_mod` | `>= 0.80` |
| `PERM` (`p_grp = 5`, swaps) | 32 | 256 | `1/120` | `>= 0.80` sequence acc |
| `DYCK` (`b_types = 2`, `depth = 8`) | 3–32 | 40–256 | 0.5 | `>= 0.80` |
| `MQAR` | `P_mqar = 16` | `P_mqar = 32` | `1/alpha_sz` | `>= 0.90` |
| `COPY` with gap | 32 | 96 | `1/alpha_sz` | `>= 0.95` |

**[NORMATIVE — v1 corrections]** Spec v1 (i) gave no `acc_random` for `COPY` and `MQAR`, so
`scaled_acc` was uncomputable for them; (ii) set `COPY` eval length 512 and `PERM` 500
against `T_ctx = 512`, which cannot fit prompt **and** answer. Lengths are restated against
the real context budget.

**[NORMATIVE]** Raw accuracy is never reported for these tasks: for parity, chance is 0.5.

`PARITY` and `PERM` are the direct tests of §05.2.2 (K7). **[NORMATIVE]** K7 is evaluated on
the **accuracy-vs-length curve** over `{40, 64, 96, 128, 192, 256}`, not a single point, and
against the **median** of 3 seeds — the cited source's own medians are materially weaker than
its best-of-3.

---

## 8.3 Language evaluation

**Stage A (M1–M3).** TinyStories V2 and a second story corpus for generator independence.

**[NORMATIVE — generation quality, frozen before any comparison]** A versioned artifact
`eval/generation_rubric_v1.json` fixes: 60 prompts; a 5-point rubric with written anchors
for **grammar**, **consistency**, and **completion plausibility**; decoding `temperature 0.8,
top_p 0.95, max_new 200, seed fixed`; 3 samples per prompt; and the grader. The grader is an
LLM judge with a **pinned model id**, and **inter-rater agreement** against 60 human-labelled
items must reach Krippendorff `alpha >= 0.6` before the rubric may be used. The grader,
rubric, prompts and decoding parameters are never changed between compared runs; if any
changes, every compared run is re-graded.

**Stage B (M4).** Per-domain `BPB` on a held-out multi-domain set. **[NORMATIVE]** Reported
per domain, not only as a mean: an aggregate can hide a model that traded one domain for
another, which is precisely the failure case-based selection is meant to prevent.

---

## 8.4 Baselines

**[NORMATIVE]** Three, all required.

- **B1 — matched dense transformer.** The §05 stack with the Site replaced by a dense
  `ReLU^2` FFN of inner width `N_G`; identical width, depth, tokenizer, data, optimizer,
  schedule and **token budget**. This is the M0 artifact (M0 at M1 scale, M0-nano at M-nano
  scale — §07.1).

  **[NORMATIVE — the parameter mismatch is stated, not matched away]** B1 is matched on
  width, depth and inner width but **not** on parameter count: SOMA additionally carries
  `W_ctx`, `kappa`, `W_key`, `W_pos`, `K_unit1`, `K_unit2` and `p_u`, and lacks nothing B1
  has. At M1 this is `+1.31 M` on `24.41 M`, i.e. SOMA is **5.7% larger**. A
  parameter-matched B1 would require widening its FFN, which changes the very quantity `N_G`
  the K9 sweep varies, so the mismatch is **reported with every B1 comparison** rather than
  removed. Any SOMA advantage smaller than the advantage a 5.7%-larger dense model would be
  expected to show is **not** reported as an advantage; the scaling-law estimate of that
  quantity, and its uncertainty, accompany the comparison. v2 called B1 "matched" without
  qualification.
- **B2 — the Regret Ecology.** **[NORMATIVE]** The germline field is **frozen at its random
  initialisation** — *not* removed — so the Site still exists and (5.20) is well defined.
  Disabled: births, deaths, selection, `L_met`, and inheritance. Enabled: token selection
  and the task stream. Spec v1 said "no germline", under which the Site is undefined and B2
  had no forward pass, yet B2 is the baseline whose defeat C1 requires.
- **B3 — random router.** Routing assignment fixed **per unit slot at initialisation** and
  held constant for the run; `r_ut = 1 / k_route`; `L_wire` **disabled** (it is meaningless
  under random routing, and would otherwise collapse all `p_u` toward the mean query).

---

## 8.5 Equivalence testing — how "unchanged" is decided

**[NORMATIVE — v1's structural defect]** Every spec-v1 kill criterion had the form "if X is
unchanged, the component is decorative". A non-significant difference is **not** evidence of
equivalence, and no margin, TOST, or non-inferiority bound was defined anywhere. As written,
A1–A3, A5, A6, A10, A11 and K2, K3, K4, K10 could not be adjudicated in either direction.

**[NORMATIVE]** Each ablation declares one **primary metric** and an **equivalence margin**
`delta`. "Unchanged" means the **two one-sided tests (TOST)** both reject at
`alpha_sig = 0.05` over `n_seed >= 5` seeds, i.e. the 90% CI of the difference lies entirely
within `(-delta, +delta)`. "Changed" means the ordinary two-sided test rejects. Neither
rejecting is reported as **inconclusive**, which is an honest third outcome.

| Ablation | Primary metric | `delta` |
|---|---|---|
| A1a, A1b | `Q_route` | 0.02 |
| A2 | case-win histogram (KS statistic) | 0.10 |
| A3 | held-out `CE` | `2 * noise_floor(CE)` |
| A4 | `CriterionScore` trend `b` | 0.02 |
| A5, A6 | held-out `CE`; suite `scaled_acc` | `2 * noise_floor` each |
| A7 | `PARITY` `scaled_acc` at 256 | 0.05 |
| A8 | held-out `CE` | `2 * noise_floor(CE)` |
| A9 | `CriterionScore` trend `b` | 0.02 |
| A10, A11, **A12** | held-out `CE` | `2 * noise_floor(CE)` |
| B3 | held-out `CE` and `Q_route` | `2 * noise_floor`; 0.02 |
| **K5** (certification) | mean `live - ramp` novelty fraction | **0.34** (§06.4) |

**[NORMATIVE — every margin must exceed the granularity of its statistic]** A TOST margin
narrower than the smallest attainable non-zero difference in the metric makes the equivalence
region a gap between attainable values, so neither TOST can reject and the criterion is
undecidable in both directions. Two margins are set by this rule rather than by judgement:

- **K5** at `0.34`, just above the `1/3` quantisation of the per-firing novelty fraction
  (§06.4). v2's `0.10` was below it and K5 could never fire.
- **A2** at `0.10` on the KS statistic, which with `|U_live| >= 64` samples has granularity
  `1/64 = 0.016`; this one is comfortably above and is retained.

**[NORMATIVE]** `noise_floor` is measured per §07.7 over **5 differently-seeded** M0 repeats.
Under v2's identical-seed definition every such floor was exactly zero and every margin in
this table that references one was zero.

---

## 8.6 The ablation matrix

**[NORMATIVE]** Every novel component carries exactly one ablation and one kill criterion.
Every ablation **pins** its parameter through `ablation.pin` (§02.2.3) and **excludes it from
`E_i`**, so the intervention is clean and the main arm cannot drift into the ablated
condition.

| # | Ablation | Setting | Stage | Kill |
|---|---|---|---|---|
| **A1a** | no wiring price | `lambda_wire = 0` | nano + M1 | K4 |
| **A1b** | no population cap | `U_cap = U_max` | nano | — |
| **A2** | mean-utility selection | parents = top-`n_par` by `nu_hat`; deaths likewise | nano + M1 | K3 |
| **A3** | no development | births/deaths off, `\|U\|` fixed at the measured equilibrium | nano + M1 | K2 |
| **A4** | no inheritance | germline reset to random each generation | nano | **K1** |
| **A5** | no token selection | `k_tok = 1.0` | nano + M1 | — |
| **A6** | no task stream | `f_oe = 0` | nano + M1 | — |
| **A7** | unsigned transitions | `beta_mix = sigmoid` (range `[0,1]`) | **M1 only** | K7 |
| **A8** | growth-event count | `{1, 3, 8}` events | nano | — |
| **A9** | frozen body plan | freeze `p_u` and `g_u`; train `A`,`B`,`W_shared` | nano | — |
| **A10** | constant context gate | `psi == 1` | nano + M1 | — |
| **A11** | no recombination | `p_rec = 0` | nano | — |
| **A12** | pre-gate activation | `relu(z)^2` gated *after* the activation (§01.4) | nano + M1 | — |

**[NORMATIVE — v1 corrections]**
- **A1 split** into A1a/A1b. Spec v1 changed two variables jointly and attributed the result
  to one; and its `lambda_size = 0` removed the quantity from which `BUDGET_HEADROOM` was
  derived, so births were unbounded and the arm would OOM.
- **A1b is redefined in v3.** §01.6.1 establishes that `L_size` has **identically zero
  gradient** (`m_u` is bookkeeping), so v2's `lambda_size = 0` was a provable **no-op**: the
  ablation could not change a single trajectory, and reporting it as an ablation of "the
  tissue price" would have been reporting a null result about a mechanism that was never
  connected. A1b now ablates the mechanism that actually bounds the population — the cap — by
  setting `U_cap = U_max`, which leaves the grid as the only bound. This is safe (the grid is
  preallocated, so the arm cannot OOM) and it is the real question: does bounding the
  population do anything, or does the death rule bound it anyway?
- **A2** now specifies **both** birth and death rules; spec v1 gave only a death rule,
  leaving the ablated arm unable to produce births. The case-win histogram is computed as an
  operator-independent diagnostic (§04.10) so it exists in both arms.
- **A3**'s `|U|` is fixed at the **measured equilibrium** of a prior full run, not at an
  unstated constant.
- **A4** — a random germline is accepted only if it passes §02.5 on the **first** attempt;
  resampling until acceptance would itself be a selection process contaminating the control.
  The empty-`Inventory` seeding path (§02.2.4) applies. The ES perturbation scale is
  **matched** across arms so the curves differ only by inheritance.
- **A7 runs at M1 only.** Its primary metric is `PARITY scaled_acc` at length **256**
  (§08.5), and M-nano's `T_ctx = 128` cannot hold a length-256 prompt plus its answer, so the
  metric is not computable at M-nano and the v2 stage assignment "nano + M1" was
  unsatisfiable. The same applies to every `40..256` row of §08.2. **[NORMATIVE]** At M-nano
  the regression suite runs a reduced band, eval lengths `40..96`, with pass bars unchanged;
  the archive holdout of §06.3.5 (`n_len <= 32`) already excludes it. K7 is adjudicated at M1
  on the full `{40, 64, 96, 128, 192, 256}` curve.
- **A9** is redefined. ANIL is **not well-defined here**: ANIL freezes a *meta-learned* body,
  but SOMA's outer loop never trains `A`, `B`, `W_shared` — they are soma, re-randomised
  every lifetime — so freezing all but the head would train a random-features model and the
  stated inference would not follow. A9 now freezes the **body plan** (`p_u`, `g_u`) while
  training the parts: a genuine test of adaptivity within a fixed structure.
- **A10** is **not parameter-matched** (it removes `W_ctx` and `kappa`); the confound is
  stated with every A10 result.
- **A12** is new in v3 and is mandatory. §01.4 proves that with the activation *before* the
  gate every unit is a linear reweighting of one shared nonlinear feature, so the population
  is provably not a population of distinct functions. v2 asserted the opposite and was wrong.
  A12 runs the pre-gate placement — which is **exactly parameter-matched**, differing only in
  the order of two elementwise operations — and tests whether the corrected claim does any
  work. If A12 is equivalent under the declared margin, the per-unit nonlinearity of §01.4 is
  real algebraically but inert empirically, and §01.8's fifth falsifier has fired. A claim
  that was found false once is not permitted to stand on its correction alone.
- Stage assignment is complete for all of A1a–A11; spec v1 assigned A8 to no Stage-A slot and
  A9/A11 to no stage at all, while §03 committed to their results.

**[NORMATIVE]** Every ablation is compared at **matched tokens**, with measured wall-clock
reported alongside; A3 and A1b change per-step cost, so a step-matched comparison would not
be compute-matched.

### 8.6.1 The `N_G` scaling sweep (for K9)

**[NORMATIVE]** K9 asks whether `N_G` must grow with `d_model` or `|U|`. Spec v1 declared it
"cheap to check" and "checked first" while **no experiment anywhere tested it**. The sweep:
`N_G in {256, 512, 1024, 2048, 4096}` x `d_model in {96, 192, 384}` at M-nano, with `|U|_0`
fixed, measuring held-out `CE` at matched tokens. K9 fires if the `CE`-minimising `N_G` grows
proportionally to `d_model`.

**[NORMATIVE — the sweep must bracket the default]** `N_G = 2048` is the germline-scale
default at every configuration (§01.2), so a grid topping out at 1024 could not observe
whether the optimum sits at, below, or above the value actually used — and an optimum pinned
at the grid's upper edge is uninformative about proportionality. The grid brackets 2048 on
both sides. v2's grid `{128, 256, 512, 1024}` predates the unification of `N_G` across
configurations and excludes the default entirely.

**[NORMATIVE — this sweep varies a germline-scale constant deliberately]** §01.2 fixes `N_G`
at every configuration precisely so that this sweep is interpretable: the default holds it
constant, so any dependence the sweep finds is attributable to the sweep.

---

## 8.7 Reporting discipline

**[NORMATIVE]**

1. **`pass_at_k` at large `k`**, full curve, never `pass@1` alone. Methods that reshape a
   distribution can win at `k = 1` while the base model wins at `k = 128`; reporting only
   `pass@1` would let sharpening masquerade as capability expansion.
2. **No task-identity conditioning.** A **static assertion** verifies that no task-identity
   feature reaches the model. Spec v1 required "the ablation confirming this", but an absent
   input cannot be ablated.
3. **Per-step trajectories** for any iterative refinement — a comparable published result was
   found on replication to achieve most of its accuracy at the *first* step.
4. **Measured wall-clock and memory** with every efficiency claim.
5. **Components whose gain shrinks with compute are flagged as suspect.** A comparable
   adaptive-depth architecture beat its baseline at small scale and **lost at 1.7B**. The
   scale-direction of every effect is reported with its magnitude.
6. **Negative results at equal prominence.** A published ablation table in this area asserts
   all components contribute while its own numbers show one ablation with better perplexity
   than the full model. Any row contradicting our claim is stated in the text.
7. **Verification status carried** — peer-reviewed vs preprint, author-reported vs replicated
   (§11).
8. **Inconclusive is a reportable outcome** (§8.5).

---

## 8.8 Out of scope locally

**[NORMATIVE]** Not reachable below ~`1e9` parameters and excluded from success criteria:
frontier abstract-reasoning suites, competitive software-engineering benchmarks, long-horizon
agentic evaluations, graduate-level science QA, multi-hour agentic coherence. §09 states what
the scaling path would need to demonstrate before any such claim becomes admissible.
