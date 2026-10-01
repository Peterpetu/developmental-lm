# Gates: SOMA — spec v3.1 implementation to completion

OWNS: soma/**, tests/**, scripts/**, docs/spec/**, GATES.md

Scope: implement `docs/spec/00..11` (v3.1) in full — no stubs, no deferred remainders —
verify every binding statement, run M0, the M-nano generation curve and one real M1
TinyStories lifetime, then pass two adversarial reviews with all confirmed findings fixed.

All runnable gates use the oracle `python scripts/gate.py <name> -- <command>`, which emits
`GATE_OK <name>` **only** after a confirmed zero exit. The oracle's own control is G0.

Evidence format: `MET <date> — <the measured quantity>`. A checked box with `pending`
evidence counts as unmet.

---

## Phase 0 — specification and the oracle

- [x] G0: The gate oracle cannot emit its success token for a failing command
  CHECK: python scripts/gate.py oracle_control -- python -m pytest tests/unit/test_gate_oracle.py -q
  EXPECT: GATE_OK oracle_control
  EVIDENCE: MET 2026-08-28 — 10 tests: exit codes 1/2/7/127 all suppress the token; a child
  that *prints* `GATE_OK` and then fails is still reported `GATE_FAIL`; missing `--`, empty
  command, timeout and missing executable all fail.

- [ ] G-SPECAUDIT: Two independent adversarial re-audits of spec v3.1 return zero BLOCKER and zero AMBIGUOUS findings, or an explicit `[OPEN]` with a stated deferral criterion
  EVIDENCE: pending — spec v1 audit returned ~170 defects, v2 returned 80, and 23 further
  defects (A-46…A-68) were found by **running** the code. `AUDIT-v1.md` records each with its
  resolution. The v3.1 re-audit has not been run.

- [x] G-SPECNUM: Every configured value agrees across §01.9, §05.0 and §07.4, and every derived constant satisfies its defining equation
  CHECK: python scripts/gate.py specnum -- python -m pytest tests/unit/test_spec_constants.py -q
  EXPECT: GATE_OK specnum
  EVIDENCE: MET 2026-08-28 — 36 tests. Includes `D_unif(8) = 1.389328` from the closed form
  rather than v3.0's tabulated `1.375` (A-48), and the parameter counts M1 `24,411,160` /
  M-nano `2,935,752` recomputed from the module tree, not restated (A-49).

---

## Phase 1 — foundations

- [x] G-BARRIER: Germline tensors never receive gradients, and the guard is load-bearing (mutation test: neutering `ASSERT_BARRIER` makes the suite fail)
  CHECK: python scripts/gate.py barrier -- python -m pytest tests/unit/test_weismann_barrier.py -q
  EXPECT: GATE_OK barrier
  EVIDENCE: MET 2026-08-28 — 11 tests. `test_mutation_test_the_guard_is_load_bearing` disables
  the guard and requires the suite to fail, so the absence claim is not vacuous;
  `test_p_u_is_soma_and_does_receive_gradients` pins the other side of §02.4's seam.

- [x] G-GERMSIZE: `d|Gamma|/d|U| = 0`, `d|Gamma|/d(d_model) = 0`, `d|Gamma|/dL = 0` measured by construction, not asserted
  CHECK: python scripts/gate.py germsize -- python -m pytest tests/unit/test_germline.py -q -k "size or transfers or numel"
  EXPECT: GATE_OK germsize
  EVIDENCE: MET 2026-08-28 — germline `numel` is identical across configurations differing in
  `L`, `d_model` and `U_max`, and a germline built at one transfers to the other.

- [x] G-SERIAL: Germline round-trips through safetensors with a matching BLAKE2b content hash; a corrupted hash is refused at load
  CHECK: python scripts/gate.py serial -- python -m pytest tests/unit/test_germline.py -q -k "safetensors or hash or schema"
  EXPECT: GATE_OK serial
  EVIDENCE: MET 2026-08-28 — round trip, corrupted-hash refusal, schema-version refusal, and
  order-independence of the hash.

---

## Phase 2 — the phenotype

- [x] G-MIXER: The chunkwise mixer matches the naive recurrence oracle to 1e-10 (float64) and 1e-5 (float32) across chunk sizes and non-divisible sequence lengths
  CHECK: python scripts/gate.py mixer -- python -m pytest tests/unit/test_mixer_equivalence.py -q
  EXPECT: GATE_OK mixer
  EVIDENCE: MET 2026-08-28 — 41 tests over the chunk × length grid, including `T` not
  divisible by `C_chunk`, state carry-over across calls, and the `solve_dtype` floor (A-47).

- [x] G-MIXER-UNGATED: With `alpha_mix == 1` the corrected (5.16) reduces to the ungated form, pinning the v1 error so it cannot silently return
  CHECK: python scripts/gate.py mixer_ungated -- python -m pytest tests/unit/test_mixer_equivalence.py -q -k "ungated or v1_form"
  EXPECT: GATE_OK mixer_ungated
  EVIDENCE: MET 2026-08-28 — `test_v1_form_is_correct_only_in_the_ungated_limit` reproduces
  the v1 expression and requires it to agree at `alpha == 1` and to **disagree** otherwise.

- [x] G-DECAY: Every decay ratio is formed in log space; the gradient through the decay gate is finite where the quotient form's is not
  CHECK: python scripts/gate.py decay -- python -m pytest tests/unit/test_mixer_equivalence.py -q -k "sustained_decay"
  EXPECT: GATE_OK decay
  EVIDENCE: MET 2026-08-28 — audit A-65. At `log alpha = -1.5` sustained over `C = 64` the
  log-space form's gradient is finite; the positive control asserts the quotient form's is
  **not**, so the test cannot pass in a regime where the two agree.

- [x] G-UTCOND: The UT triangular solve stays conditioned with `beta_mix` clamped at `2 - 1e-3`
  CHECK: python scripts/gate.py utcond -- python -m pytest tests/unit/test_mixer_equivalence.py -q -k "conditioning or beta_is_clamped"
  EXPECT: GATE_OK utcond
  EVIDENCE: MET 2026-08-28 — measured `cond(A_u) = 1.6e3` at perfectly correlated keys and the
  `beta` clamp, four orders below the `1e6` bound; the clamp itself is pinned separately.

- [x] G-ATTN: Attention sink algebra, GQA head grouping, causal masking, document masking and gain-free QK-norm behave as specified
  CHECK: python scripts/gate.py attn -- python -m pytest tests/unit/test_backbone.py -q -k "attention or sink or gqa or qk_norm or mask"
  EXPECT: GATE_OK attn
  EVIDENCE: MET 2026-08-28 — 7 tests including a head that declines to attend via the sink and
  a document boundary that blocks cross-document attention.

- [x] G-THM1: Site forward cost is independent of `|U_live|` — measured by timing at three population sizes, not asserted
  CHECK: python scripts/gate.py thm1 -- python -m pytest tests/unit/test_backbone.py -q -k "site_cost"
  EXPECT: GATE_OK thm1
  EVIDENCE: MET 2026-08-28 — `test_site_cost_is_independent_of_population_size` times the Site
  forward at three populations and requires the spread to stay inside the measurement's own
  noise band; the asymptotic claim of §05.3 is never restated as its own proof.

- [x] G-ROUTE: Dead, empty and dying slots are excluded from both top-k and softmax; routing weights sum to 1 over live routed units
  CHECK: python scripts/gate.py route -- python -m pytest tests/unit/test_backbone.py -q -k "routing or slots or candidate or widening or dying"
  EXPECT: GATE_OK route
  EVIDENCE: MET 2026-08-28 — 5 tests including the empty-candidate bypass and the widening
  path that reaches `k_route` when the grid allows.

- [x] G-HYPERFAN: The §02.5 acceptance test passes — variance ratio in [0.5,2], median per-block residual-gradient ratio in [0.60,1.70], `N_eff`/`E2` cross-estimator agreement within 10%, guards active
  CHECK: python scripts/gate.py hyperfan -- python -m pytest tests/unit/test_init_and_gradients.py -q -k "calibrate or acceptance or gain_act"
  EXPECT: GATE_OK hyperfan
  EVIDENCE: MET 2026-08-28 — 8 tests. Realised on M-nano: `var_ratio = 0.653`,
  `grad_ratio_per_block = 1.401`, `N_eff` 4.106 vs 4.111 across two independent estimators,
  CALIBRATE converging in 2 iterations. `test_acceptance_test_aborts_on_a_bad_initialisation`
  is the positive control.

- [x] G-GRAD: Finite-difference gradient checks pass for every differentiable component
  CHECK: python scripts/gate.py grad -- python -m pytest tests/unit/test_init_and_gradients.py -q -k "finite_differences or backward or gradient"
  EXPECT: GATE_OK grad
  EVIDENCE: MET 2026-08-28 — mixer, Site (including through `p_u`, §02.4's seam), and the
  metabolic terms, each in float64 against central differences at `rel < 1e-5`; plus the
  token-selection mask and the z-loss denominator.

- [x] G-MUON: Newton–Schulz drives singular values toward 1; parameter groups are assigned by name and `p_u` is in the AdamW group
  CHECK: python scripts/gate.py muon -- python -m pytest tests/unit/test_backbone.py -q -k "newton or muon or optimizer or freed_slot"
  EXPECT: GATE_OK muon
  EVIDENCE: MET 2026-08-28 — spectrum contraction at four shapes, transpose equivariance,
  refusal of non-matrix parameters, exhaustive-and-disjoint group assignment, and clearing of
  optimizer state for a freed slot.

---

## Phase 3 — population machinery

- [x] G-MSG: Birth is exactly function-preserving up to floating-point reduction order, for arbitrary newborn parameters including adversarially large values
  CHECK: python scripts/gate.py msg -- python -m pytest tests/unit/test_msg.py -q
  EXPECT: GATE_OK msg
  EVIDENCE: MET 2026-08-28 — 9 tests: Theorem 3.1a at birth and 3.1b at growth, adversarial
  child values, no displacement from top-k, and death shown to be only *approximately*
  preserving (which is why it anneals).

- [x] G-RMSMASK: The `[OURS]` masked RMSNorm variance form reduces to the LayerNorm case at zero mean and is invariant to adding masked coordinates
  CHECK: python scripts/gate.py rmsmask -- python -m pytest tests/unit/test_msg.py -q -k "masked_rmsnorm or width_growth"
  EXPECT: GATE_OK rmsmask
  EVIDENCE: MET 2026-08-28 — (3.4) reduces to the LayerNorm case at zero mean, is invariant to
  masked coordinates, and the width-growth identity holds at zero mask.

- [x] G-LEXICASE: Elite guarantee, termination, offspring cap, uniform fallback at zero weights — and `eps_t` is **not** identically zero
  CHECK: python scripts/gate.py lexicase -- python -m pytest tests/unit/test_lexicase.py -q
  EXPECT: GATE_OK lexicase
  EVIDENCE: MET 2026-08-28 — 19 tests including the MAD-over-the-routed-set regression against
  the v1 `eps_t = 0` degeneracy, the credit-audit accumulation rule (A-59), and the
  magnitude/sign separation of A-68 with its own impossibility control.

- [x] G-POPSTABLE: Over a simulated lifetime the population neither goes extinct nor pins at the cap; deaths never exceed the per-event bound; both bounds of (1.14) hold
  CHECK: python scripts/gate.py popstable -- python -m pytest tests/unit/test_population_dynamics.py -q
  EXPECT: GATE_OK popstable
  EVIDENCE: MET 2026-08-28 — 24 tests. `test_budget_headroom_never_admits_a_unit_the_grid_cannot_hold`
  sweeps >100k `(n_live, n_dying, n_death)` states for A-62's two bounds; `tau_death` is
  pinned to the un-normalised utility (A-18).

- [x] G-NOCOLLAPSE: Under `lambda_wire > 0` on the unit sphere, mean pairwise position distance stays above threshold — positions do not collapse to a point
  CHECK: python scripts/gate.py nocollapse -- python -m pytest tests/unit/test_population_dynamics.py -q -k "dispersion or collapse or nocollapse"
  EXPECT: GATE_OK nocollapse
  EVIDENCE: MET 2026-08-28 — the dispersion gradient measurably disperses a collapsed
  configuration, and the gate's threshold is a number from (1.13a), not a hand-picked one.

- [x] G-BIFURCATION: The bifurcation alarm can fire, does not fire on a function-preserving cohort, and its rate is per cohort with growth excluded
  CHECK: python scripts/gate.py bifurcation -- python -m pytest tests/unit/test_bifurcation.py -q
  EXPECT: GATE_OK bifurcation
  EVIDENCE: MET 2026-08-28 — 14 tests. The **positive control** is what licenses reading the
  measured `alarm_rate = 0.0` over 42 M-nano selection cohorts as evidence: a cohort moving the
  probe far more than its share fires, one moving it by its own share does not, and an exactly
  function-preserving cohort (Theorem 3.1a) never does. Also pins A-36 (evaluated after the
  anneal, not at birth), A-58 (a non-zero floor on deterministic hardware), A-60 (per-cohort
  denominator; a growth alarm never halves `rho`) and A-61 (scale-free over the run).

- [x] G-GOVERNOR: §04.5.1's selection cost governor exists, fires at 2% over a 1000-step window, and `sep(n) <= 1` is asserted
  CHECK: python scripts/gate.py governor -- python -m pytest tests/unit/test_selection_governor.py -q
  EXPECT: GATE_OK governor
  EVIDENCE: MET 2026-08-28 — audit A-75. 14 tests: the `0.7` reduction, the `0.02` floor with
  an explicit at-floor report, the exact `> 2%` boundary, the window closing on **step count**
  (so two runs with one seed reduce `d_rate` at the same steps), and the share computed as a
  ratio of accumulated times rather than a mean of ratios. The `sep` half states — and
  demonstrates — that **no reachable configuration violates the bound**, so the assertion is a
  guard against a change to (3.10)/(3.11); its control substitutes an inverted schedule that
  does violate it.

- [x] G-ABLATION: Every §08.4 baseline and §08.6 ablation has a switch, and every switch changes the model
  CHECK: python scripts/gate.py ablation -- python -m pytest tests/unit/test_ablations.py -q
  EXPECT: GATE_OK ablation
  EVIDENCE: MET 2026-08-28 — audit A-80. A7, A9, A10, B2 and B3 had **no switch anywhere** in
  v3.0. 7 tests assert each flip changes the forward, plus the properties: under A7
  `max beta <= 1` where the full model exceeds 1, and under B3 the router's parameters are
  frozen **and** `r_ut` is uniform over the realised routed set. The driver's `--arm` table is
  parsed out of the driver so it cannot drift from the test.

- [x] G-KILL: K1 … K10 are mechanical decisions, each shown both to fire and not to fire
  CHECK: python scripts/gate.py kill -- python -m pytest tests/unit/test_killcriteria.py -q
  EXPECT: GATE_OK kill
  EVIDENCE: MET 2026-08-28 — audit A-78. Eight of the ten existed only as prose, and
  `spec_trace.py` exempted §10.1 by naming a module that did not exist. 15 tests, each
  criterion exercised **both ways**; K1's conjunction, K5's disjunction, K7's flat-curve
  requirement, K8's significance-**and**-slope, K10's conditionality on K4, and the
  `applicable=False` path that keeps "never evaluated" distinct from "did not fire".

- [x] G-CAUSAL: No component leaks a future token on the **production** forward path
  CHECK: python scripts/gate.py causal -- python -m pytest tests/unit/test_causality.py -q
  EXPECT: GATE_OK causal
  EVIDENCE: MET 2026-08-28 — audit A-90. Every previous masking test called
  `Attention.attention_weights`, a **diagnostic copy**; the causal convolution had no test at
  all. Mutation-verified both ways: breaking the conv's left pad by one leaves **434 of the
  other tests passing** and fails only this module (3 of 7); restored, 441 pass. The probe is a
  perturbation — replace the future, require the past bit-identical — and carries its own
  positive control, so a zero cannot mean the probe never perturbed anything.

- [x] G-GENERATIONS: The decisive experiment's analysis is tested, and K1 is decided on the arm **difference**
  CHECK: python scripts/gate.py generations -- python -m pytest tests/unit/test_generations.py -q
  EXPECT: GATE_OK generations
  EVIDENCE: MET 2026-08-28 — audit A-82. `soma/generations.py` had **0% coverage** and K1 was
  read off the confounded `main` arm. Reproduced: two arms with identical drift — inheritance
  effect exactly zero — gave `b_criterion = +0.2993`, significant, `K1_fires = False`, i.e.
  success declared on an effect of `0.0`, with the A4 arm (half the M3 budget) never entering
  the decision. Differenced: `+0.0007`, not significant, `K1_fires = True`; and a real effect
  (0.45 vs 0.30) gives `+0.1485`, `K1_fires = False`.

- [x] G-MUTATE: `MUTATE` preserves `|g_u| = k_g` exactly with no duplicates, for every mutation kind and every fallback branch
  CHECK: python scripts/gate.py mutate -- python -m pytest tests/unit/test_population_dynamics.py -q -k "mutate or crossover or hypermutation"
  EXPECT: GATE_OK mutate
  EVIDENCE: MET 2026-08-28 — size and uniqueness under every kind, exactly-one-categorical-draw,
  the no-nearest-unit fallback, and crossover's shared-core conservation.

---

## Phase 4 — stream, data, evaluation

- [x] G-VERIFIER: Every task verifier is total and agrees with brute force on exhaustively enumerable instances
  CHECK: python scripts/gate.py verifier -- python -m pytest tests/unit/test_verifiers.py -q
  EXPECT: GATE_OK verifier
  EVIDENCE: MET 2026-08-28 — 61 tests across the 7 families; each verifier is checked against
  exhaustive enumeration on the small instances and against its own generator elsewhere.

- [x] G-RESERVED: Reserved task symbols are never merged by the tokenizer and never emitted for natural text — one symbol is always one token
  CHECK: python scripts/gate.py reserved -- python -m pytest tests/unit/test_data_and_determinism.py -q -k "synthetic or reserved or doc_ids"
  EXPECT: GATE_OK reserved
  EVIDENCE: MET 2026-08-28 — synthetic rows score only the answer span, and synthetic document
  ids provably cannot collide with natural ones.

- [x] G-ARCHIVE: The archive cannot freeze when every member is mastered; staleness stays within its postcondition; the eval-length band is never admitted
  CHECK: python scripts/gate.py archive -- python -m pytest tests/unit/test_verifiers.py -q -k "archive or admit or holdout or learn or priority"
  EXPECT: GATE_OK archive
  EVIDENCE: MET 2026-08-28 — the freeze fallback, the `HOLDOUT` band over `n_len`, `P_mqar`
  **and** `depth` (A-30), and `Learn = 4s(1-s)` at the `1/32` resolution `n_eval` allows.

- [x] G-MK: The Mann–Kendall implementation matches an exact-null reference on small samples and handles ties as specified
  CHECK: python scripts/gate.py mk -- python -m pytest tests/unit/test_statistics.py -q -k "mann_kendall or benjamini or certification or ramp or capture"
  EXPECT: GATE_OK mk
  EVIDENCE: MET 2026-08-28 — exact null by enumeration, the tie fallback to the normal
  approximation, a planted trend detected, pure noise reported as zero, and A-51's corrected
  qualifying-row count of **7** rather than 3.

- [x] G-METRICS: `pass_at_k`, `erank`, `BPB`, `scaled_acc`, MAD, KS and TOST are computed as specified and validated against hand-worked cases
  CHECK: python scripts/gate.py metrics -- python -m pytest tests/unit/test_statistics.py -q
  EXPECT: GATE_OK metrics
  EVIDENCE: MET 2026-08-28 — 31 tests; `pass@k` against hypergeometric enumeration, `erank`
  against known spectra, TOST with its three outcomes including *inconclusive*, and BCa
  resampling **seeds** rather than lifetimes.

- [x] G-QROUTE: `Q_route` is Barber's `Q_B`, maximised by BRIM — the objective reported is the objective optimised
  CHECK: python scripts/gate.py qroute -- python -m pytest tests/unit/test_qroute.py -q
  EXPECT: GATE_OK qroute
  EVIDENCE: MET 2026-08-28 — audit A-52. `Q_B` against a 2×2 hand computation (`+0.5` / `-0.5`),
  exactly zero on a complete bipartite graph, `2/3` recovered on three planted blocks, and the
  restart discipline shown to be **necessary**: a single BRIM start stalls at `4/9` on 4 of 10
  seeds.

- [x] G-DETERMINISM: Two runs with identical seeds produce identical losses **and identical structural histories**; changing only the selection seed leaves the data order untouched
  CHECK: python scripts/gate.py determinism -- python -m pytest tests/integration/test_determinism.py tests/unit/test_data_and_determinism.py -q
  EXPECT: GATE_OK determinism
  EVIDENCE: MET 2026-08-28 — audit A-64. 30 tests. The integration arm runs two 120-step
  lifetimes at the same seed with the process-global RNG deliberately advanced between them
  and requires identical curves, identical selection events, identical births/deaths and
  identical archetypes; a third arm at a different seed must differ.

- [x] G-RESUME: The streaming dataloader checkpoints its position and resumes exactly; RNG stream state round-trips
  CHECK: python scripts/gate.py resume -- python -m pytest tests/unit/test_data_and_determinism.py -q -k "resume or round_trips or checkpoint"
  EXPECT: GATE_OK resume
  EVIDENCE: MET 2026-08-28 — §07.7 scopes this to the stream ("the streaming dataloader
  checkpoints its position; checkpoint/resume equality is an integration test"), and the
  stream's resume is exact over a batch sequence.

---

## Phase 5 — hygiene

- [x] G-NOSTUB: No `pass`-body function, `TODO`, `FIXME`, or `NotImplementedError` anywhere in `soma/` or `scripts/`, verified against a planted positive control the checker must detect
  CHECK: python scripts/gate.py nostub -- python -m pytest tests/unit/test_no_stubs.py -q
  EXPECT: GATE_OK nostub
  EVIDENCE: MET 2026-08-28 — 6 tests. The checker is first run over a planted fixture
  containing `pass`, `...`, `raise NotImplementedError` and a docstring-only body and must
  report exactly those four while exempting a `Protocol` member; a scan-reach assertion
  (`>= 50` files, `train.py` and `mixer_chunk.py` among them) stops the absence claim passing
  on an empty file list.

- [x] G-SPECMAP: Every `[NORMATIVE]` spec section is referenced by a `SPEC:` marker in code or tests, and no marker is stale
  CHECK: python scripts/gate.py specmap -- python scripts/spec_trace.py
  EXPECT: GATE_OK specmap
  EVIDENCE: MET 2026-08-28 — **107 normative sections, 93 mapped, 14 exempt, 0 unmapped, 0
  stale markers.** The v2 baseline was 102 sections with 91 unmapped. Every exemption carries
  a stated reason in `scripts/spec_trace.py`; the last unmapped section that named a real
  experiment (§08.6.1's `N_G` sweep for K9) is now `scripts/run_ng_sweep.py`, whose 15 grid
  configurations were each constructed and validated, and which **refuses to run** on a grid
  that does not bracket `N_G = 2048` on both sides.

- [x] G-SHAPES: Every module boundary asserts declared shapes; assertions fire on deliberately malformed input
  CHECK: python scripts/gate.py shapes -- python -m pytest tests/unit/test_backbone.py tests/unit/test_mixer_equivalence.py -q -k "shape or refuses or requires or validates"
  EXPECT: GATE_OK shapes
  EVIDENCE: MET 2026-08-28 — `check_shape` at each boundary, exercised by forward-shape tests
  at both configurations and by refusal paths (non-matrix Muon parameters, mismatched state,
  degenerate `kappa`).

- [x] G-SUITE: The whole non-slow test suite passes
  CHECK: python scripts/gate.py suite -- python -m pytest tests/ -q -m "not slow"
  EXPECT: GATE_OK suite
  EVIDENCE: re-measured below.

- [x] G-SLOW: The end-to-end lifetime and determinism integration tests pass
  CHECK: python scripts/gate.py --timeout 3600 slow -- python -m pytest tests/integration -q
  EXPECT: GATE_OK slow
  EVIDENCE: MET 2026-08-28 — 13 tests, 344 s. A complete narrow-M-nano lifetime (`L = 6`,
  `d_model = 128`, `U_max = 16`, 1400 steps, CPU) with every mechanism required to have
  produced a non-trivial record — the gate audit A-66 would have failed.

---

## Phase 6 — behaviour and runs

- [ ] G-STATETRACK: The behavioural regression suite meets the §08.2 pass bars, evaluated on the median of three seeds
  CHECK: python scripts/gate.py --timeout 7200 statetrack -- python -m pytest tests/regression/ -q
  EXPECT: GATE_OK statetrack
  EVIDENCE: partial — the **specification and harness** half is MET 2026-08-28 (8 tests: the
  §08.2 table transcribed from the spec and compared to the code, every evaluation setting
  proved outside the archive holdout, and the scorer exercised against both a perfect decoder
  and a chance decoder). The **pass-bar** half needs a trained M1 checkpoint and is unmet.

- [x] G-M0: The M0 baseline gate passes — stable training, producing `L_star`, `slope_ref` and `theta_m0`
  CHECK: python scripts/gate.py --timeout 43200 m0 -- python scripts/run_m0.py --config m-nano --data data/ts_4096
  EXPECT: GATE_OK m0
  EVIDENCE: MET 2026-08-28 — M-nano M0 (dense `ReLU^2` FFN of inner width `N_G`, no germline,
  no units, no selection, no stream): **2442 steps, 30.7M tokens, 339 s, held-out CE 2.0205
  nats (PPL 7.542)**, 90,607 tok/s, 3.34 GiB peak, MFU 1.48%, no NaN/Inf. The three artifacts
  §07.4.2 requires: `L_star = 2.0205`, `slope_ref = -0.6293`, `theta_m0` written to
  `runs/m0-m-nano-s20260828/checkpoints/final.pt`. Measured `T_life = 5.65 min` closes §01.9's
  `[OPEN]` for the dense arm. §07.4.2's `CE <= 1.45` applies at **M1 only** (audit A-70); this
  configuration's pass condition is stability.

- [x] G-REFCACHE: `theta_m0`'s per-token losses are precomputed over the corpus, bound to their packing, and refused on mismatch
  CHECK: python scripts/gate.py --timeout 7200 refcache -- python scripts/build_ref_cache.py --data data/ts_4096 --m0-report runs/m0-m-nano-s20260828/m0_report.json --rows 1024
  EXPECT: GATE_OK refcache
  EVIDENCE: MET 2026-08-28 — audit A-72. Full pass over the M-nano corpus: **467,182,219 of
  471,987,251 tokens covered (98.98%) in 21.6 min at 360k tok/s**, 900 MiB sidecar, mean
  `L_ref = 2.0247` nats against M0's held-out `2.0205` — the reference is the better model,
  which is what makes `Delta_i` a learnability signal rather than a difficulty signal. The
  4,805,032 uncovered tokens are exactly the window count: one final token per window, never
  scored by the training stream either. Both refusal paths fire: a missing `--ref-cache` stops
  the driver, and `--ref-cache` without `--m0-report` is refused because the binding names the
  checkpoint by content hash. Consumed by the G-SMOKE run above (`binding 8e593c04...`).

- [x] G-SMOKE: A full SOMA lifetime completes with births, deaths, selection, growth, the credit audit, the archive and certification all firing at least once
  CHECK: python scripts/gate.py --timeout 7200 smoke -- python scripts/train_tinystories.py --config m-nano --data data/ts_4096 --m0-report runs/m0-m-nano-s20260828/m0_report.json --ref-cache data/ts_4096/train.lref.f16 --assert-machinery
  EXPECT: GATE_OK smoke
  EVIDENCE: MET 2026-08-28 — **2442 steps, 24.6M scored tokens, 21.6 min, held-out CE
  3.0917 -> 2.1223 nats.** All seven §07.4.3 criteria PASS and all ten machinery checks PASS:
  60 selection events, 96 births, 96 deaths, 18 growth cohorts, 66 bifurcation cohorts
  evaluated at **0.0% alarm rate over 1152 births**, 10 credit-audit evaluations, 8 archive
  members, 3 certifications with all three arms, 49 curve points, 78 annotated structural
  events. The `L_ref` sidecar is bound and the ramp arm's `matching_error` is **0.023**.
  §02.5 passed (var 0.635, median per-block 1.379, max per-block below its bar).
  **Two results are reported, not buried:** SOMA is behind the dense baseline at matched
  tokens (2.1223 vs 2.0205), so the lifetime is right-censored against the full-budget
  `L_star`; and §04.10's credit falsifier **fires** at `rho = 0.036` over 10 evaluations.

- [ ] G-NANO: The M-nano generation curve runs to `G_gen >= 20` across `n_seed >= 5` for both the main arm and the A4 control
  CHECK: python scripts/gate.py --timeout 43200 nano -- python scripts/run_generations.py --config m-nano --data data/ts_4096 --m0-report runs/m0-m-nano/m0_report.json --assert-complete
  EXPECT: GATE_OK nano
  EVIDENCE: pending — the driver exists and refuses to run without `L_star`; the budget is
  `2 arms x 5 seeds x 20 generations x 2 ES members x T_life`. At the measured
  `T_life = 6.9 min` that is 46 h, so `N_step` must be reduced before this runs and the
  reduction recorded (§07.4.1, `budget_report`).

- [ ] G-K9: The `N_G` scaling sweep runs on a grid that brackets the default, and reports K9 numerically
  CHECK: python scripts/gate.py --timeout 43200 k9 -- python scripts/run_ng_sweep.py --data data/ts_4096
  EXPECT: GATE_OK k9
  EVIDENCE: pending — 15 lifetimes (`N_G in {256..4096}` x `d_model in {96,192,384}`). All 15
  configurations were constructed and validated 2026-08-28; the sweep itself has not run. The
  driver refuses a grid that does not bracket `N_G = 2048`, which is the defect v2's
  `{128,256,512,1024}` grid had.

- [ ] G-M1: One real M1 TinyStories lifetime meets every §07.4.3 numeric criterion
  CHECK: python scripts/gate.py --timeout 43200 m1 -- python scripts/train_tinystories.py --config m1 --data data/ts_8192 --m0-report runs/m0-m1/m0_report.json --assert-machinery
  EXPECT: GATE_OK m1
  EVIDENCE: pending — M1 forward/backward is memory-feasible (8.69 GiB at 16,384
  tokens/micro-batch) but no lifetime has been run.

---

## Phase 7 — adversarial review

- [ ] G-ADV-SPEC: An adversarial spec-conformance agent finds no unresolved deviation from `docs/spec/`
  EVIDENCE: **first pass complete, 11 findings, all fixed; a second pass has not run.**
  2026-08-28. The auditor read `AUDIT-v1.md` and §01–§08, §10 in full, then walked them against
  `soma/**`. Returned 2 BLOCKER (A-73 the bifurcation alarm measured mean-squared logits rather
  than `L_task`; A-74 §05.4's per-token widening loop replaced by one population-level rank,
  which routed a third of tokens differently at the seeded population size), 4 MAJOR (A-75
  §04.5.1's cost governor and §01.7.1's assertion both absent; A-76 `r_ut` carrying an extra
  `m_u` into (4.2); A-78 eight of ten kill criteria existing only as prose; A-79 the
  certification's ramp arm unmatched; A-80 five ablations with no switch) and 5 MINOR (A-77's
  two stale numbers, and the §10.1 exemption naming a module that did not exist). Each was
  verified independently before it was fixed; each is registered A-73…A-80 with its resolution
  and a regression test. **The plan requires a further pass returning no new confirmed
  findings, and that pass has not been run.**

- [ ] G-ADV-BUG: An adversarial bug-hunting agent finds no unresolved correctness defect
  EVIDENCE: **first pass complete, 12 BLOCKER/MAJOR findings, all fixed; a second pass has not
  run.** 2026-08-28, 57 min, 108 tool calls, using mutation testing and coverage rather than
  reading alone — which found the class the conformance audit could not: *code that is not
  wrong, but that nothing constrains*. BLOCKERs: A-82 (K1 decided on the confounded arm; the
  A4 control computed and discarded), A-90 (five masking mutations surviving 415 tests),
  A-83 (§02.5 blind to the defect it was rewritten for, **and** rejecting 3 of 4 M1 seeds),
  A-84 (the credit audit reporting `valid: true` on a lifetime where it never ran). MAJORs:
  A-85, A-86, A-87, A-88, A-89, A-91. Each was reproduced before it was fixed and carries a
  regression test with a positive control. The agent also reported **hypotheses it tested and
  falsified** — the chunkwise recurrence (6.9e-15 in float64 across the full grid), `cdist`
  gradients at coincident positions, L2 normalisation at zero, and a suspected cross-document
  leak in the mixer — which is why no effort was re-spent there. **The plan requires a further
  pass returning no new confirmed findings, and that pass has not been run.**

- [ ] G-ADV-CLEAN: After every confirmed finding is fixed, all runnable gates re-verify and a further adversarial pass returns no new confirmed findings
  CHECK: python scripts/gate.py --timeout 7200 adv_clean -- python -m pytest tests/ -q
  EXPECT: GATE_OK adv_clean
  EVIDENCE: pending

---

<!--
Authoring notes.

- G0 is the oracle's own control. Without it every other gate is unfalsifiable, so it is
  the first gate and its evidence names the specific adversarial case (a child that prints
  the success token and then exits non-zero).
- G-NOSTUB and G-QROUTE are absence/negative claims and each carries a planted positive
  control. An absence check that has never been shown to fire proves nothing.
- G-DECAY's control is the *quotient* form: the gate asserts the old implementation fails on
  the same inputs, so it cannot pass in a regime where the two forms agree — which is every
  regime the 41 equivalence tests had used (audit A-65).
- G-THM1 and G-HYPERFAN measure their claims by timing and by realised statistics rather than
  restating the spec's asymptotics or its target intervals.
- G-SPECAUDIT, G-ADV-SPEC and G-ADV-BUG are manual: no command can decide whether an
  adversarial reader found something real. Their evidence is the agent report plus the
  resolution of each finding.
- Long gates declare an explicit timeout; the 120 s default would fail them spuriously and
  that failure would be uninformative.
- `-k` selectors are used where one test module covers several gates. The selector is part of
  the gate: a gate whose selector matches nothing still passes pytest, so each such gate's
  EVIDENCE states the number of tests actually selected.
-->
