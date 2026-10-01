# 08 — Protocol, gates, kill criteria, and what may be claimed

Everything here is fixed before the runs it governs. Where a number was measured rather than
chosen, the measurement is cited.

## 8.1 The runs

**[NORMATIVE]** In this order. A run's budget is stated in FLOPs (`6 N T`, non-embedding
`N`) and its realised tokens, wall-clock and peak memory are recorded. "Screening" budgets
are used first (the user's instruction of 2026-09-02: small experiments before the full
architecture); "full" budgets follow only for mechanisms that survive screening.

| id | run | configuration | budget | decides |
|---|---|---|---|---|
| **R1 NOISE** | 4 seeds of the baseline recipe under CRN; `sigma_e` pooled and per stratum | TinyStories s0; later every configuration used | screening: 40M tokens at s0 | G-NOISE; every `eps_c` of §07.4 |
| **R2 LR** | `eta_M` grid × 2 paired seeds | s0 | screening | the recipe's `lr_provenance` |
| **R3 BASE** | B-FLAT at the adult; profile | TinyStories adult; Pile adult | screening: 100M tokens; full: 400M (TinyStories), 2B (Pile) | G-BASE, throughput targets |
| **R4 BIRTH** | `REGRESS` on the trained s0 and adult; at-birth CE; shuffle and random controls; `d_z = 0`; nesting decode at s1 | TinyStories | minutes | K-BIRTH, K-SCALE |
| **R5 GROW** | ANLAGE-GROW vs B-COPY vs B-RAND vs B-FLAT (+ MEAN, ZERO), 4 seeds CRN | TinyStories s0 → s1 (screening), s0 → s2 (full) | screening: `C_total` = 6 · N_s1 · 60M | K-GROW, H-GROW |
| **R6 POP** | K = 4 siblings + lexicase + merge vs SINGLE, 4 seeds; G-LIVE per axis; `eps` dial | TinyStories | screening | K-POP, G-LIVE |
| **R7 LOOP** | 3 generations vs B-WARM and B-FLAT at equal `C_total` | TinyStories | full | K-LOOP, K-ANCHOR, K-SOTA |
| **R8 PILE** | the surviving design at the Pile lineage | Pile | full | K-SOTA (Pile) |
| **R9 AUDIT** | second adversarial pass: spec conformance + mutation testing | — | — | G-ADV-CLEAN |

**[NORMATIVE — TinyStories first]** No Pile run starts before R1–R7 have reported on
TinyStories; the corpus is the fast-iteration substrate and every mechanism is screened there.

## 8.2 Statistics

**Evaluation sets, v2 (2026-09-02, after R3).** The v1 sets (R1–R4 artifacts) were the first
256 consecutive windows of each split; their data-class composition is that of the split's
head, not the corpus's, and the fitness and test heads differ (R2: `[27, 38, 41, 55, 37, 58]`
against `[58, 43, 52, 48, 20, 35]` windows per class). A mixture fit on the fitness set then
partly learns that set's composition. From R3 on, **[NORMATIVE]** every evaluation set is
drawn by the pool's own rule — window offsets uniform over the split from a fixed seed
(`EvalSet.sample`; test seed 20260902; fitness seed 20260903 over the first half of the
fitness shard; the merge's coefficient-fit windows, seed 20260904, over the second half, so
S8's disjointness holds by construction). `sigma_e` and `sigma_e(q)` are re-measured on the
v2 sets from R1's saved checkpoints; every manifest records `eval_version`.

**v3 (2026-09-02, after the first R6 arm).** The two validation halves are not identically
distributed: on v2 sets the test half holds `[29, 27, 29, 37, 80, 54]` windows per class and the
fitness half `[33, 34, 40, 49, 51, 49]` (the test half is long-document heavy), so a mixture
fit on the fitness set shifts toward its composition and is scored on a different one. From
R3's re-evaluation on, **[NORMATIVE]** the fitness, merge-fit and test sets are drawn from
the *union* of both validation shards by `EvalSet.sample_documents`: offsets uniform over
the union's tokens, a window kept when its starting document's global index is `0`, `1` or
`2 mod 3` respectively. The three sets share no document and have one composition in
expectation. R6 and everything after it are measured on v3.

**[NORMATIVE]**

* `sigma_e` is the pooled within-arm standard deviation of final held-out CE over seeds
  under CRN (`screen/variance.py::pooled_within_sd`), measured by R1 at every configuration
  before anything else is run there; it is never inherited from another configuration
  (GNOMON's 0.171 against SCREEN's 0.0100 is why).
* Every contrast is **paired by seed** under CRN and reported as the mean paired difference
  with a 95% interval over seeds: BCa (`gnomon/utils/stats.py::bca_bootstrap`) at `≥ 5`
  seeds, the vendored bootstrap's own floor; a paired Student-t interval at 3–4 seeds,
  labelled `t` in every report; the screening runs of §8.1 use 4 seeds and are therefore
  labelled, and any claim they support is stated as a screening result.
* "Better by `≥ 3 sigma_e`" means the paired mean difference exceeds `3 sigma_e` **and** the
  BCa interval excludes zero. "Equivalent" means TOST at margin `2 sigma_e` rejects both
  one-sided tests. Neither is "inconclusive", which is a reportable outcome.
* Every reported CE is on the **test split** at `T = 512`, natural text, in nats, with
  bits per byte alongside (`soma/eval/metrics.py::bits_per_byte` at the measured
  tokens-per-byte). The fitness split is used for selection, merging and `sigma_e(c)`, and
  for nothing that is reported as a result.
* A priced quantity (parameter count, FLOPs) never enters a fitness; budgets are matched, not
  optimised (SCREEN's deterministic-term rule).

## 8.3 Gates and kill criteria

**[NORMATIVE]** "Fires" means a design claim is false, not that a run was unlucky; every one
that fires is reported at the same prominence as a pass. `scripts/anlage_analyse.py`
evaluates each from artifacts on disk and prints **NOT EVALUABLE** for any run that does not
exist.

| id | fires when | consequence |
|---|---|---|
| **G-NOISE** | a number is reported from a configuration whose `sigma_e` was not measured | the number is withdrawn |
| **G-BASE** | the free-weight recipe on this card, at matched tokens and tokenizer, is worse than a reproduced llama2.c-style TinyStories run by more than `0.02` nats (compared in bits per byte), or the Pile adult is worse than Pythia's per-token loss at matched tokens by more than `0.05` nats | the substrate is not a control; fix before any mechanism is measured |
| **G-NOFREE-INVERSE** | a decode is executed inside a training step (a planted decode is the positive control) | throughput regression; a bug |
| **K-BIRTH** | after a regression whose per-role residuals are all `< 0.5`, `birth_share < 0.5` | the genome reproduces the weights but not the function; raise `(H, d_g, d_h, d_z)` by §8.4, else the neural-kernel genome is falsified |
| **K-SCALE** | a genome fit at stage `s` expresses stage `s+1`'s new units no better than random: the at-birth CE of the grown decode is not below the ZERO/random arm's by `3 sigma_e` | scale-freedom is false in practice; growth reverts to B-COPY and the finding is reported |
| **K-GROW** | ANLAGE-GROW is not better than B-COPY by `≥ 3 sigma_e` at matched `C_total` | growth stays if B-COPY beats B-FLAT; the genome's role in it is removed and reported |
| **K-POP** | the population arm is not better than SINGLE by `≥ 3 sigma_e` **and** the merge is not better than the best sibling | the population is decorative; ANLAGE reduces to genome + growth; reported |
| **K-ANCHOR** | every `lambda_anchor > 0` is worse than or equivalent to `0` | the anchor is dropped |
| **K-LOOP** | generation `g+1`'s adult is not better than B-WARM (`eps = 1`) at equal `C_total` by `≥ 3 sigma_e` | compression through the genome buys nothing over inheriting the weights; the design's central claim fails and is reported as such |
| **K-SOTA** | the full system is not better than B-FLAT at equal `C_total` and equal wall-clock on this card | not SOTA; reported |
| **G-LIVE / G-SEARCHWORTHY** | a diversity axis moves CE by `< 3 sigma_e`, or its landscape is monotone | the axis is not varied (§07.6) |
| **G-ADV-CLEAN** | the second adversarial pass returns a confirmed finding | fixed, registered, re-run |

## 8.4 Dials, and the order in which they are turned

**[NORMATIVE]** `(H, d_g, d_h, d_z, d_y)` of the genome, `f_0`, `K`, `sigma_mut`, `rho_0`,
`n_ramp`, `lambda_anchor`, `eps`. A dial is turned only by a run that names it, in the grid
that run specifies, and the value used thereafter carries the run's id as provenance. The
genome dials are raised only on K-BIRTH's instruction, in the order `d_g → H → d_z → d_h`,
each doubling, with the residual report after each step.

## 8.5 The run manifest

**[NORMATIVE]** Every run writes `runs/anlage/<run_id>/manifest.json` with: the git commit and
dirty flag; `torch`, CUDA, driver, GPU; the configuration and stage; the tokenizer hash,
corpus hash and counts digest; every seed of the `RngBundle`; the pool seed and pool digest;
`lr_provenance`; the realised tokens, FLOPs, wall-clock, tokens/s (median, IQR) and peak
memory; the compile backend; and, for a genome, `|Gamma|`, `|Z|`, both ratios, the per-role
residuals and the at-birth CE. Results go through `gnomon/utils/report.py::write_json`
(`allow_nan=False`).

## 8.6 Reporting rules

**[NORMATIVE]**

1. Every curve is held-out CE on the test split, natural text, in nats, bits per byte
   alongside; growth events, regressions and merges are annotated, never smoothed.
2. Every arm that failed to run, was cut for budget, or was skipped is listed with the reason;
   a missing arm is never silently omitted.
3. Every number in a final report is re-measured from the artifacts on disk immediately
   before reporting, never quoted from a message.
4. Negative results at equal prominence: a mechanism removed by a kill criterion is recorded
   in `docs/anlage/RESULTS.md` with its measurement, in the same table as the survivors.
5. Anything at `< 4` seeds is labelled as such.
6. Published numbers are context, reported in bits per byte at the nearest matched token
   budget, never as the basis of a claim.

## 8.7 Acceptance tests

**[NORMATIVE]** The protocol's instruments are accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| R1 | `sigma_e` | `pooled_within_sd` recovers a planted noise level; the per-stratum estimator agrees with a hand computation |
| R2 | paired contrast | the BCa interval of a planted `3 sigma_e` difference excludes zero at 4 seeds; a planted zero difference is TOST-equivalent at margin `2 sigma_e` |
| R3 | analyse | every criterion of §8.3 prints NOT EVALUABLE on an empty runs directory, and the correct verdict on planted reports in both directions |
| R4 | manifest | every field of §8.5 is present; a NaN is refused at write |
| R5 | gate oracle | `scripts/gate.py` cannot emit its success token for a failing command (vendored G0) |
