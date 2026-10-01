# SOMA — Specification 07: Training Protocol

**Spec version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Normative for `soma/train.py`, `soma/generations.py`
and `configs/`.

---

## 7.1 Definitions

**[NORMATIVE — one meaning per word]** Spec v1 used "generation" for two different things
and gave both the symbol `G`, leaving kill criterion K1's "`G >= 20`" ambiguous between
~80 GPU-hours and a quantity that was already unreachable.

| Term | Symbol | Definition |
|---|---|---|
| **step** | `n` | one optimizer update, `0 .. N_step-1` |
| **event** | `E_sel` | one selection event (§04.2). **Never called a generation.** |
| **lifetime** | — | one complete training run of a phenotype from init to `N_step`, given a germline |
| **generation** | `G_gen` | one germline update over `P_es` lifetimes (§02.6). **The only meaning of "generation".** |
| **run** | — | one invocation of the trainer; contains one or many lifetimes |
| **birth performance** | — | held-out `L_task` of the decoded phenotype at step 0, no updates |
| **early slope** | — | slope of held-out CE against `log(tokens)` over `[0.02, 0.20] * N_step` |
| **time-to-criterion** | `n_crit` | first **evaluation step** at which held-out CE `<= L_star` |

**[NORMATIVE]** `T_val = max(50, floor(N_step / 100))`, so `n_crit` has at least
100-point resolution. Spec v1 defined `n_crit` without specifying the evaluation cadence,
leaving the entire ES fitness a function of an unstated constant.

**[NORMATIVE — `L_star` and `slope_ref` come from the matched M0, per configuration]**
`L_star` is M0's held-out CE **at a token budget equal to the SOMA lifetime's `D_tok`**, not
at M0's full budget, and `slope_ref` is that same run's early slope. Both are measured on the
**same corpus, tokenizer and context length** as the lifetime that will be scored against
them, which means **M-nano requires its own M0**.

**[NORMATIVE — M0-nano]** §7.4.2's M0 is defined at `V = 8192`, `T_ctx = 512`. M-nano runs at
`V = 4096`, `T_ctx = 128`, so M0's `L_star`, `slope_ref` and `theta_m0` are all
**incomparable** at M-nano: the CEs are over different vocabularies, and `theta_m0`'s
`L_ref` cache is refused at load by §6.1.1's binding rule. v2 defined one M0 while making
`CriterionScore` — the largest fitness term — and the whole of §6.2 depend on it at both
configurations, so the decisive experiment at M-nano had no reference at all. **M0-nano** is
therefore a required artifact: the §7.4.2 recipe re-run at the M-nano corpus settings,
producing `L_star_nano`, `slope_ref_nano` and `theta_m0_nano` with its own `L_ref` sidecar.
Its cost is one M-nano-scale dense run and it gates the M-nano lineage exactly as M0 gates M1.

Spec v1 used M0's *final* loss; since SOMA lifetimes are short and carry developmental
overhead, no lifetime would ever have reached it, `CriterionScore` would have
been identically 0, and the `w_c` term — deliberately the largest — would have carried zero
signal. **[NORMATIVE]** If `L_star` is not reached, `n_crit` is **right-censored** at `N_step` and
flagged; it is not scored as 0.

**[NORMATIVE — what censoring means for the estimator]** A censored lifetime yields
`CriterionScore = 1 - N_step/N_step_ref = 0` at `N_step = N_step_ref`, so a generation in which
*every* member is censored has zero variance in the primary statistic and contributes no ES
signal. Two consequences are handled explicitly rather than left to chance:

1. **`ES_STEP` falls back to the secondary statistic** when all `P_es` members are censored:
   `f_i` is computed from `SlopeScore` alone for that generation, which is defined whether or
   not `L_star` is reached. The substitution is logged per generation.
2. **The censored fraction is a reported field** of §7.5's analysis. If it exceeds `0.5` over
   the run of generations, `L_star` was set too aggressively for the lifetime budget and the
   generation curve is **not interpretable**; the finding is reported and `L_star` is re-derived
   at the measured `D_tok`, with every affected generation re-run.

v2 specified censoring but not its consequence, leaving `CriterionScore` silently constant —
which is the same class of defect as A-2's zero-by-construction `BirthScore`.

**[NORMATIVE]** `L_rand = ln V` (9.01 nats at `V = 8192`, 8.32 at `V = 4096`) — a
shape-independent constant. Spec v1 defined it as "a randomly initialised model of the same
shape", but `|U_live|` varies per lifetime so the shape is not fixed.

---

**[NORMATIVE — `L_star` must be reachable; v3.1, audit A-81]** `L_star` is the M0 baseline's
held-out CE at a **matched token budget**, and the M0 report publishes that CE at
`{10, 25, 50, 75, 100}%` of the budget. `L_star` is taken at the **largest fraction a
generation-0 lifetime reaches before `0.8 * N_step`**; the default is `50%`.

The full-budget value is **not** the default, and the reason is measured rather than argued.
At M-nano, M0 reaches `2.0205` nats and a full SOMA lifetime reaches `2.1249` on the same
30.7M tokens, so `n_crit` is undefined and the lifetime is right-censored. That is not a
property of one lifetime: it would make `CriterionScore` **identically zero across every
generation, both arms and all five seeds**, and §07.5's BCa bootstrap would fit a slope
through a constant. A criterion no run reaches and a criterion every run reaches instantly are
the same defect — the statistic does not vary — and §07.5's censoring rule catches it only
*after* the budget is spent.

`0.50` at M-nano gives `n_crit = 1850` of `2442` steps, leaving 24% of the run as headroom for
a faster-adapting germline to show an earlier crossing, which is the entire quantity §07.5
measures. `0.75` gives `2300` of `2442`: any improvement beyond 6% saturates against the end
of the run, which is the same censoring one step removed.

---

## 7.2 The lifetime

```
Algorithm LIFETIME(Gamma, seed) -> (metrics, archetypes)

  1.  U     <- SEED_POPULATION(Gamma.Inventory, |U|_0, side)          # §02.2.4
  2.  Theta_S <- INIT(Gamma)                                          # §01.5 hyperfan
  3.  ACCEPTANCE_TEST()                                               # §02.5 -- AFTER init
  4.  record birth performance on the held-out set, no updates
  5.  Lambda <- SEED_ARCHIVE()                                        # §06.3.3
      init next-event counters: n_sel, n_arch, n_cert, n_log, n_val   # (1.15)
  6.  for n in 0 .. N_step-1:
        a. batch <- STREAM(n)                                # §06.3.4 mixing
        b. forward; L_task (§06.2), L_met (§01.6), L_z (§05.6)
        c. backward; ASSERT_BARRIER; Muon/AdamW step on Theta_S only
        d. renormalise p_u to the unit sphere
        e. update nu_u (§03.4), masks m_u, ages; retire units whose mask reached 0
        f. push credits into the reservoir buffer (§04.6)
        g. if n >= n_ref:  update theta_lag (§06.1.2)
        h. if n >= n_sel and n >= max(T_0, M_mat):  SELECT (§04.2);  n_sel += ceil(T_sel(n))
        i. if n in growth_schedule:  GROW (§03.5.1)
        j. if n >= n_arch and f_oe > 0:  ARCHIVE_STEP (§06.3.3)
        k. if n >= n_cert and f_oe > 0:  CERTIFY, live / frozen / ramp arms (§06.4)
        l. if n >= n_val:  held-out evaluation; update n_crit if first crossing
        m. if n >= n_log:  log diagnostics (§7.6)
  7.  return metrics, and the gene sets of units surviving with nu_hat above the median
  postcondition: Theta_S is discarded; only metrics and archetypes cross the barrier
```

**[NORMATIVE]** Step 7 is the Weismann barrier in operational form: no weights, no optimizer
state, no activations are inherited — only selection statistics of germline objects.

**[NORMATIVE]** The acceptance test is step **3**, after `INIT`. Spec v1 made it a
precondition of `LIFETIME` while it depends on tensors `LIFETIME` creates at step 2.

---

## 7.3 Lifetime fitness

**[NORMATIVE — v1 correction, the primary statistic]** An audit established that
`BirthScore` is **~0 by construction** (`AUDIT-v1.md` A-2): every tensor that touches data is
soma and is re-randomised each lifetime, so inheriting gene index sets cannot improve step-0
loss. Fitness is therefore weighted toward **adaptation efficiency**, which is what a
scaffold can improve.

```
  CriterionScore = 1 - min(n_crit, N_step_ref) / N_step_ref     # = 0 when censored at
                                                                # n_crit = N_step = N_step_ref
  SlopeScore     = clip( -early_slope / slope_ref , 0 , 2 ) / 2
  FinalScore     = ( L_rand - L_final ) / L_rand
  BirthScore     = ( L_rand - L_birth ) / L_rand                   # diagnostic only

  FITNESS = w_c * CriterionScore + w_s * SlopeScore + w_f * FinalScore           (7.1)
```

**[NORMATIVE]** `w_c = 0.5`, `w_s = 0.3`, `w_f = 0.2`. `BirthScore` carries **weight zero**.

**[NORMATIVE — `BirthScore` is a bug detector]** It is *predicted* to be `~0`. A
significantly non-zero `BirthScore` means information is crossing the Weismann barrier, and
**fails gate `G-BARRIER`**. Spec v1 made it the primary statistic of the decisive experiment
and of K1, where it had no mechanism by which to improve.

**[NORMATIVE — budget]** The lifetime budget is fixed in **tokens** (`D_tok`), and
`CriterionScore` is measured against a **fixed reference** `N_step_ref` from the config, not
against the realised `N_step`. Spec v1 fixed neither: under a fixed-step budget ES could buy
quality with compute (larger `|U|`, larger `k_g`); under a fixed wall-clock budget
`1 - n_crit/N_step` rewards whichever member happened to get a larger `N_step` for the same
`n_crit`. §01.9 sets `N_step_ref = N_step` at both configurations, so the two coincide by
default and diverge only under a deliberate budget change. **[NORMATIVE]** The ±10% jitter of spec v1 is applied to the **data order**, not
the budget, so it cannot bias the fitness.

`slope_ref` is M0's early slope on the same corpus, measured once.

---

## 7.4 Configurations and milestones

### 7.4.1 The two configurations

**[NORMATIVE]** Spec v1 had one configuration, under which — at its own numbers — `SELECT`
fired 0.7–3 times per lifetime, `ARCHIVE_STEP` 0–1, and `CERTIFY` **zero** times, leaving the
developmental arm inert and K5 unable to fire; and K1 cost 80–160 GPU-hours against a stated
12-hour budget. The repair is two purposed configurations.

**[NORMATIVE]** These values are the ones in §01.9 and nowhere else; this table restates
them and may not diverge. Gate `G-SPECNUM` asserts the agreement mechanically.

| | **M-nano** | **M1** |
|---|---|---|
| Purpose | evolution dynamics, the generation curve, K1/A4 | TinyStories language quality |
| Parameters (§05.10) | **2,935,752** | **24,411,160** |
| `T_ctx` / `V` | 128 / 4096 | 512 / 8192 |
| `D_tok` / `B_tok` / `N_step` | 4e7 / 16384 / **2442** | 8e8 / 65536 / **12208** |
| `T_0` from (1.17) | **99** | **492** |
| `T_arch` / `T_cert` (1.18)–(1.19) | 122 / 610 | 610 / 3052 |
| Realised `E_sel` / archive / cert per lifetime | 10 / 20 / 4 | 10 / 20 / 3 |
| Baseline reference | **M0-nano** (§7.1) | M0 |
| `T_life` | **[OPEN]** — measured at `G-M0` | **[OPEN]** — measured at `G-M0` |
| `G_gen` in 12 h (`P_es = 2`) | `floor(720 / (2 T_life))`, recomputed from the measurement | idem |

**[NORMATIVE — v2 correction]** v2's row read `N_step` 2000 / 12300 and `T_0` 81 / 495, which
are neither the §01.9 values nor consistent with `N_step = D_tok / B_tok` at the configured
`B_tok`; and it asserted `T_life` and hence `G_gen` as fact. §01.9 is the single source and
`T_life` is `[OPEN]` until measured.

**[DERIVED — conditional on the measured `T_life`]** At M-nano, `G_gen = 20` for the
`n_seed = 5` required by §07.5, across the main arm **and** the A4 control, costs
`2 arms * 5 seeds * 20 gens * 2 members * T_life`. At an illustrative `T_life = 1.5 min` that
is `600 min = 10 h`, one overnight run.

**[NORMATIVE — the budget is re-derived, not assumed]** If the measured `T_life` exceeds
`1.8 min` the M3 budget exceeds 12 h and one of `n_seed`, `G_gen` or `N_step` must be reduced
**before** the experiment starts, with the reduction and its power consequence recorded in the
run manifest. §7.5's `n_seed >= 5` is the constraint that may **not** be reduced, since it is
what makes `alpha_sig = 0.05` attainable at all.

At M1 the germline is **inherited** from the M-nano lineage — which the germline-scale
constants of §01.2 make shape-possible — and one lifetime is run.

### 7.4.2 M0 — the baseline gate

**[NORMATIVE]** No SOMA-specific component is enabled until M0 passes.

- Model: the §05 stack with the Site replaced by a dense `ReLU^2` FFN of inner width `N_G`,
  no germline, no units, no selection, no stream.
- Corpus: TinyStories V2 (GPT-4 subset), the project tokenizer at `V = 8192`, `T_ctx = 512`.
- Budget: `D_tok = 8e8` tokens.
- **Numeric target:** held-out CE `<= 1.45` nats (`PPL <= 4.26`). **[EMPIRICAL]** This is
  set from published TinyStories reference checkpoints of comparable size on the same corpus;
  the exact reference run and its checkpoint id are recorded in `configs/m0_baseline.yaml`.
- Pass: stable (no NaN/Inf, no divergence), target met, and wall-clock, tokens/s, peak memory
  and MFU recorded.

**[NORMATIVE — v1 correction]** Spec v1's M0 gate named no reference, no corpus, no token
count, no tokenizer, and used "within 2% of the published number" (2% of what?) plus "the
loss curve matches the reference shape", which is not mechanically checkable.

M0 additionally produces `theta_m0`, the frozen reference whose per-token losses are
precomputed for §06.2. **[NORMATIVE — v3.1, audit A-70]** It is written to
`runs/<run_id>/checkpoints/final.pt` by the M0 run itself. v3.0 listed the artifact and
recorded its *path* in the M0 report while nothing ever wrote the file, so §06.2's Rho-1
reference had no source — a dangling artifact, which is worse than a missing one because the
report asserted it existed.

**[NORMATIVE — the numeric target is per configuration; v3.1, audit A-70]** `CE <= 1.45` nats
is stated for **M1's setting**: TinyStories V2, `V = 8192`, `T_ctx = 512`, `D_tok = 8e8`,
against published reference checkpoints on that corpus. §07.1 (audit A-33) separately requires
an M0 **per configuration**, and M-nano runs at `V = 4096`, `T_ctx = 128`.

A cross-entropy over a 4096-token vocabulary at a quarter of the context is not comparable to
one over 8192 at full context, and no published reference exists for it. The M-nano M0 is
therefore an **artifact-producing** run, not a gated one: its purpose is to measure `L_star`,
`slope_ref`, `theta_m0` and `T_life`, and its pass condition is **stability** — finite loss
throughout, no divergence. Carrying M1's number across would be a bar invented by
transcription, which is the class of defect §07.4.2's own v1 correction was written about.

### 7.4.3 M1 — SOMA learns

**[NORMATIVE]** All criteria are numeric.

1. No NaN/Inf in loss or gradients for the whole lifetime.

   **[NORMATIVE — how criterion 1 is checked]** v3.0 stated the criterion without saying what
   enforces it. Two checks run every step, both effectively free:

   * the loss is tested for finiteness before the backward pass;
   * the **total gradient norm returned by clipping** is tested for finiteness after it. It is
     `NaN` or `inf` exactly when some gradient is, so this costs one scalar that is already
     computed — and it catches the failure one step *before* it reaches the parameters, where
     the offending tensors can still be named. On a trip the run **aborts** and logs the
     offending parameter names.

   **[NORMATIVE — v3.1, a third check at `T_log` cadence]** Both per-step checks read the
   *loss* and the *gradients*. Neither reads the **parameters**, and a parameter can be
   corrupted with both of those finite: a failing device-side kernel writes into a buffer
   between the optimizer step and the next forward, which is exactly how A-55 presented and
   exactly what A-63 re-exhibited. A single sweep over `model.parameters()` every `T_log`
   steps closes that gap. It costs one pass over `|Theta_S|` elements per `T_log` steps —
   below measurement noise at both configurations — and on a trip it **aborts** naming the
   offending tensors, so the failure is attributed to a bounded window of `T_log` steps
   rather than to the arbitrary downstream operation that first produced a `NaN`.

   Without the second check the first symptom is a `NaN` several operations downstream, in a
   tensor that has nothing to do with the cause; that is precisely how the defect recorded in
   A-55 presented, and how much of the time spent on it was spent.
2. Bifurcation alarm fires on `< 1%` of **birth** events (§01.7.1).
3. Held-out CE at `N_step` is below held-out CE at `0.10 * N_step` by at least
   `2 * noise_floor(CE)` (§7.7).
4. `erank` shows **no significant negative** Mann–Kendall trend over the final 50% of the
   run at `alpha_sig = 0.05`. Spec v1 said "does not decline monotonically", which a noisy
   statistic satisfies trivially.
5. `|U_live|` in `[0.25 * |U|_0, U_cap]` at the end of the run.
6. Generation quality (§08.3) `>= 3.0 / 5` mean on the frozen rubric.
7. The chunkwise mixer matches the oracle at every checkpoint (§05.2.6).

### 7.4.4 M2, M3, M4

- **M2** — SOMA beats a parameter-matched M0 under identical **token** budget on held-out CE
  and on graded generation quality, both at `alpha_sig = 0.05`.
- **M3** — the decisive experiment (§7.5), run at **M-nano**.
- **M4** — larger corpus; per-domain BPB `= (CE_nats/ln 2) * (tokens/bytes)` on a held-out
  multi-domain set, with any deviation between the training mirror and the evaluation set's
  composition stated in every reported table.

---

## 7.5 The decisive experiment

**[NORMATIVE]** The central claim is **not** that a single run beats a baseline; growth at
matched total compute is approximately a wash. The claim is about capital formation across
generations:

> **Adaptation efficiency improves monotonically across germline generations at fixed
> per-lifetime token budget.**

**Protocol.** Run at **M-nano**.
1. `G_gen = 0` starts from a random germline.
2. Each generation runs `P_es = 2` mirrored lifetimes, scores by (7.1), and applies
   `ES_STEP` (§02.6.1). **Single-point ES**; spec v1's §07.5 described drawing germlines
   "from the current population", which is a different algorithm from its own §02.6.
3. Record `CriterionScore(g)`, `early_slope(g)`, and `BirthScore(g)` for `g = 0 .. G_gen`.

**Analysis.** **[NORMATIVE]** Primary statistic `y(g) = CriterionScore(g)`; secondary
`early_slope(g)`. Fit `y(g) = a + b log(1 + g)`. Test `b > 0` one-sided at
`alpha_sig = 0.05` by **BCa bootstrap over seeds**, resampling **seeds** (not lifetimes),
with `n_seed >= 5`.

**[NORMATIVE — v1 correction]** Spec v1 specified a bootstrap over **3** seeds, which admits
27 distinct resamples and a minimum achievable `p ≈ 0.037`, with no stated statistic,
resampling unit, or sidedness. Five seeds are required for the claimed `alpha_sig`.

**[NORMATIVE — kill rule, stated as a boolean]** K1 fires iff
`NOT( b_criterion > 0 significant )` **AND** `NOT( b_slope > 0 significant )`, i.e. the
project is killed only when **both** pre-registered tests fail. Spec v1's §07.5 said "either"
and §10 K1 said "or", and the two readings decide the project's central verdict differently.

**Control.** Ablation **A4** runs the identical protocol with the germline reset to random
each generation. The **difference** between the curves is the inheritance effect; the
absolute curves are confounded by ordinary recipe drift and are not interpretable alone.

---

## 7.6 Required logging

**[NORMATIVE]** Every `T_log = 50` steps, as newline-delimited JSON to
`runs/<run_id>/log.jsonl`, with `config.json` and `env.json` alongside.

*Optimisation* — step, tokens, held-out and training `L_task`, `L_wire`, `L_size`, `L_z`,
grad norm, `lr`, tokens/s, peak memory, MFU, `sep(n)` (1.19).
*Population* — `|U_live|`, `|U_dying|`, births, deaths, mean/median age, offspring
histogram, lineage depth, clade count.
*Structure* — mean and variance of `||q_t - p_u||^2` over routed pairs weighted by `r_ut`;
mean within- vs between-clade position distance; `Q_route` (§08.1) every `10 * T_log`.
*Plasticity* — `erank` per Site every `10 * T_log`; dead-unit fraction; mean `|w|`.
*Selection* — case-win histogram; hyperselection-cap hit rate; realised `eps_t` distribution
and the fraction that are zero (regression guard for B17); selection wall-clock share;
credit-audit Spearman.
*Stream* — realised `f_oe`; synthetic share of **selected** tokens; archive size; solve-rate
distribution; certification verdicts (live and frozen arms) with p-values.
*Barrier* — the per-step assertion result, aggregated.

**[NORMATIVE]** "Clade" is defined by cutting the lineage forest at
`MRCA depth <= 3` generations; the cut rule is fixed and logged. Spec v1 required "number of
distinct clades" with no cut rule.

---

## 7.7 Determinism and reproducibility

**[NORMATIVE]** One RNG stream **per purpose** (§02.6.3), all seeded from the run seed and
recorded. Spec v1 mandated separation in one bullet and shared a generator between lexicase
and archive in the next.

**[NORMATIVE — noise floor, per metric]** M0 is run **5 times with *different* run seeds**
(and therefore different data order, initialisation and dropout streams); for each reported
metric the noise floor is the **standard deviation** across those repeats.

**[NORMATIVE — v2 correction, and why this one mattered]** v2 specified "3 times with
identical seeds". Under §7.7's own determinism requirement, identical seeds produce identical
runs, so every such standard deviation is **exactly zero** — making every
`delta = 2 * noise_floor` margin zero, and every TOST that depends on one (A3, A5, A6, A8,
A10, A11, B3, and kill criteria K2, K3, K10) undecidable in the equivalence direction. Only
seed-to-seed variation measures the quantity the margin is meant to represent. Five seeds
matches §07.5's requirement and gives the standard deviation 4 degrees of freedom.

**[NORMATIVE]** Nondeterminism that survives seeding — cuBLAS reduction order, atomics — is
measured **separately** as `sd_probe` (§01.7.2) over `k_probe = 8` repeats at fixed seed, and
is used only for `epsilon_bif`. The two quantities are not interchangeable and are never
substituted for one another. No
claimed difference smaller than `2 x noise_floor(metric)` is reported as a difference. Spec
v1 defined "spread" for CE only and then applied it as a universal gate to `Q_route`,
histograms and scaled accuracy, which are on different scales.

The streaming dataloader checkpoints its position; checkpoint/resume equality is an
integration test.

---

## 7.8 Compute budgeting

`C = 6 * N_par * D_tok` is the planning estimate, with **measured** throughput from M0
substituted for peak.

**[NORMATIVE]** Any efficiency or Pareto claim is accompanied by **measured wall-clock and
peak memory**, never by analytic FLOPs alone: parameter-counting FLOP estimates omit
normalisation, non-linearity, attention-kernel overhead and router cost.

**[NORMATIVE — the profiling gate]** Gate `G-PROFILE` runs a 200-step M-nano lifetime with
every hook live under `torch.profiler` and asserts, against the table below, that each
budgeted item's **measured** share of step time is within a factor of 2 of its stated figure.
An item exceeding that is a specification defect, not a performance issue: it means a cost was
claimed that the design does not have. The measured shares are written to the run manifest and
every wall-clock figure in §07–§09 cites them.

**[NORMATIVE — costs the v1 spec omitted]** The following are budgeted and reported:

| Item | Cost |
|---|---|
| `theta_m0` reference losses | one forward pass over the corpus, **precomputed once** |
| `theta_lag` update | one parameter copy every `T_ref = 32` steps |
| `theta_lag` **evaluation** | one forward on `q_cases` retained cases **per step** — `+3.3%` at M1 (§06.1.2) |
| Certification | `J_ring * K_ring = 256` forwards on stored batches per `T_cert`, **x2** for the frozen control arm |
| Archive `s_task` | `(n_eval + R_ref * n_eval) = 32 + 512` greedy generations per `T_arch` |
| `erank` | 1 forward on the probe batch + `L` SVDs of `2048 x N_G` per `10 * T_log` |
| Selection | `O(n_par * q_cases * |U_elig|)` per event; capped at 2% of step time (§04.5.1) |
| Credit audit | partial forward on 1% of cases per event |
