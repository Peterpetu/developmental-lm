# SOMA — Specification 10: Risks and Kill Criteria

**Spec version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Normative for project conduct.

---

## 10.1 Kill criteria

**[NORMATIVE — v1's structural defect]** Every spec-v1 criterion had the form "if X is
unchanged / matches on **every metric**, the component is decorative". Two problems made
them unfireable: no equivalence test existed (a non-significant difference is not evidence
of equivalence), and a conjunction over the dozens of fields in §07.6 will always have one
field differing by chance across seeds. §08.5 supplies TOST margins, and each criterion below
now names a **small pre-registered primary metric set** and a **mechanical decision**.

| K | Condition (all at `alpha_sig = 0.05`, `n_seed >= 5`) | Decision |
|---|---|---|
| **K1** | `NOT(b_criterion > 0)` **AND** `NOT(b_slope > 0)` in §07.5's BCa bootstrap | **C1 unmet.** Report the negative result at full prominence. Select the ablated variant with the lowest held-out `CE` at matched tokens and re-describe the work as what it then is. |
| **K2** | A3 equivalent to full SOMA on `{held-out CE}` under `delta = 2 x noise_floor` | Development is decorative. Remove births/deaths; retain selection over a fixed population only if A2 still shows a difference. |
| **K3** | A2 equivalent on `{held-out CE, Q_route, case-win KS}` | Case-based selection is decorative. Remove it; the residue is a metabolically-priced MoE and is described as such. |
| **K4** | A1a equivalent on `{Q_route}` under `delta = 0.02` | The connection-cost mechanism does not operate here. Drop it from the claims; retain the term only if it independently improves `CE`. |
| **K5** | Certification's **live-minus-ramp** novelty fraction (§06.4) equivalent to zero under `delta = 0.34`, **or** live-minus-frozen equivalent to zero | The stream is not open-ended. Stop calling it open-ended; re-describe as curriculum shaping; reduce `f_oe` and re-evaluate whether it earns its compute by the A6 result. |
| **K6** | `max abs(chunk - ref) / (abs(ref) + 1e-8) > 1e-5` on the §05.2.6 grid after 3 engineer-days | Halt all downstream work. No result computed with a wrong mixer is admissible. |
| **K7** | `PARITY` scaled accuracy median-of-3 below `0.95` at length 256, **and** the accuracy-vs-length curve is flat over `{40..256}` | Withdraw the §05.2.2 claim for our implementation; investigate the log-space interaction (§05.2.3); if unresolved, disable signed transitions and report. |
| **K8** | `erank` shows a **significant negative Mann–Kendall trend** over the final 50% with slope below `-1e-4 / step` | The plasticity argument for death does not operate here. Report; investigate optimizer interaction. |
| **K9** | In the §08.6.1 sweep, the `CE`-minimising `N_G` grows proportionally to `d_model` | The bottleneck claim (9.1) is false; the architecture is a hypernetwork with the known ceiling. **Most damaging single finding; checked early.** |
| **K10** | B3 equivalent on `{held-out CE, Q_route}` **and** K4 has already fired | Unit identity carries no information. The population is a fiction and the work reduces to a dense model. |

**[NORMATIVE]** K6 and K9 are checked **first**: both invalidate everything downstream and
both are cheap.

**[NORMATIVE — v1 corrections, per criterion]**
- **K1** — spec v1's §07.5 read "kill if **both** fail" and §10 K1 read "kill if **either**
  fails". The boolean is now stated once, here, as a conjunction of two failures.
- **K5** — two repairs. v2's margin `delta = 0.10` lay **below the granularity** of the
  statistic it tested: with `J_ring = 8` and the `n >= 6` exclusion the per-firing novelty
  fraction is quantised to multiples of `1/7` (v3.0 said `1/3`; the realised row count is
  **7**, audit A-51), so the equivalence region `(-0.10, +0.10)` sat entirely inside the gap
  between two attainable values and neither TOST could ever reject —
  K5 was undecidable in both directions, which is the same structural defect §08.5 was written
  to remove. The margin is now `0.34`. Second, v2's control was the **frozen** archive alone,
  which does not separate open-endedness from monotone difficulty ramping; §06.4 adds the
  **ramp** arm and K5 is adjudicated on `live - ramp`, with `live - frozen` retained as a
  sanity floor. Spec v1's certification had no control arm at all, so it could not distinguish
  an open-ended archive from a static difficulty-tracker: learnability is satisfied by ordinary
  training progress on any data, and novelty is produced by construction because
  `Learn = 4s(1-s)` pins difficulty against the current model. §06.4's frozen-archive control
  makes K5 meaningful.
- **K6** — "cannot be made to match" was a claim about effort, not a condition. Now a norm, a
  grid, and a time-box.
- **K7** — spec v1 tested one length. A `[0,1]`-eigenvalue model can pass at bounded length
  for reasons unrelated to the theorem (three full-attention layers are present), so A7 would
  have drawn the wrong inference. The curve is now the criterion.
- **K8** — "declines monotonically" is essentially never satisfied by a noisy statistic, so
  K8 could never fire and M1's criterion 4 was trivially passed. Now a trend test with a slope
  threshold.
- **K9** — declared "cheap to check" while **no experiment tested it**. §08.6.1 adds the
  sweep. Note also that had `K_unit1/2` remained germline (as spec v1 left ambiguous), K9
  would have fired *definitionally*; §02.1.1 places them in the soma.

---

## 10.2 Risk register

### R1 — The four high-risk mechanisms have no scaling precedent
The spatial expression map beyond ~`1e6` params, unit-level ε-lexicase, metabolic pricing in
an LM, and cross-run inheritance (§09.2). *Mitigation:* each is separately ablatable
(A1a–A4), so a failure is attributable. *Residual:* if all four fail, SOMA reduces to a
well-engineered hybrid transformer, and we say so.

### R2 — Repeated structural events may be harmful
One-shot growth beat gradual multi-stage growth in the cited study, and measured speedups
**decay** with longer training (53.1% at 4e11 tokens vs 31.0% at 7e11), while persistence to
8e12 is an extrapolation its own points contradict. *Mitigation:* A8; few growth events plus
many selection events, which are different operations; `rho_0` may be pinned to zero.

### R3 — Decoder initialisation pathology
Mis-initialised weight-generating networks fail to train at any learning rate, and
normalization corrects vanishing but **not** exploding decoded weights. *Mitigation:*
corrected hyperfan (§01.5), guards on `N_eff` and `E2`, and a hard acceptance test (§02.5).
*Residual:* our composition is not identical to the analysed case; the test is empirical.

### R4 — Small-scale wins that die at scale
A comparable adaptive-computation architecture beat its baseline at small scale and **lost at
1.7B**, its advantage shrinking as compute grew. *Mitigation:* §08.7 rule 5; scale-direction
reported with magnitude.

### R5 — Fine-grained routing instability at small scale
Extremely sparse expert routing makes standard load-balancing ineffective in lower layers.
*Mitigation:* `k_route` is large relative to `|U|_0` at M-nano; ε-lexicase is itself a
diversity mechanism; the hyperselection guard targets collapse directly.

### R6 — Short-horizon bias
*Mitigation:* case weights are an improvement signal (§04.3); fitness weights
time-to-criterion above final loss (§07.3); jitter is applied to data order, not budget.

### R7 — Open-endedness that only sharpens
Verifier-free self-improvement sharpens rather than expands. *Mitigation:* every generator is
verifier-backed; `pass_at_k` at large `k` is mandatory; certification has a control arm.

### R8 — Toolchain fragility
Triton on Windows, a CUDA build without `scaled_mm`, a Linux-first kernel ecosystem.
*Mitigation:* the pure-PyTorch reference path is the oracle and is always available.

### R9 — Post-training incompatibility
Per-token dynamic routing may break fixed rollout paths in standard RL tooling, as reported
for a dynamic-depth architecture. *Mitigation:* **none yet**. Recorded as an identified,
**unmitigated** risk (§09.6).

### R10 — Novelty overclaim
Case-based selection over networks exists. *Mitigation:* §04.1 states the prior art and
narrows the claim to unit-level, in-model, learnability-weighted selection. **[OPEN]** The
absence of lexicase in LM training rests on three non-exhaustive searches; a citation-graph
sweep must precede any published novelty claim.

### R11 — Evaluation self-deception
Task-identity leakage; refinement claims where most accuracy arrives at step 1; aggregates
hiding domain trades; ablation captions contradicted by their own tables. *Mitigation:*
§08.7, all eight rules, without exception.

### R12 — The specification itself
**[NORMATIVE]** Spec v1 contained ~170 defects including a wrong equation (5.15), an inert
selection mechanism (`eps_t = 0`), a population that went extinct, positions that collapse to
a point, and a decisive experiment measuring a statistic that is zero by construction.

**A second audit of the v2 repair found ~80 further defects**, of which the most serious were
*introduced by the repair itself*: the claim that per-unit nonlinearity survives a pre-gate
activation is algebraically **false**; the sphere constraint does **not** prevent positional
collapse, which is the global minimiser of `L_wire`; `T_arch` and `T_cert` had no value
anywhere; `q_cases` was defined inversely to `d_rate`, making the cost governor amplify cost;
and the noise floor was defined over identical seeds, making every TOST margin exactly zero.
All are recorded with resolutions in `AUDIT-v1.md` (Parts A and A').

*Mitigation:* Phase 0's exit gate re-runs the adversarial auditors on the repaired specs and
requires zero BLOCKER and zero AMBIGUOUS findings, or an explicit `[OPEN]` with a stated
deferral criterion (`G-SPECAUDIT`). *Residual:* **the second audit is the evidence that one
audit is not enough.** The defect count fell 170 → 80 and the character shifted from
architectural to missing-constants and wrong-signs, which is convergence, not completion. The
register is versioned so a third audit is comparable, and §08 is written on the assumption
that some defects will only be found by running the code.

---

## 10.3 Standing epistemic caveats

**[NORMATIVE]** These accompany every external presentation.

1. Several load-bearing results are **preprints**, several key numbers **author-reported
   without replication**. §11 carries per-claim status.
2. Some numbers were read from figures or a single extraction pass and are flagged in §11.3.
   **[NORMATIVE]** None may be the sole support for a decision; where one currently is, the
   decision carries `[OPEN]` (§11.3 now enforces this, which spec v1 asserted but violated in
   at least three places).
3. Results at `1e6`–`1e7` parameters do not license claims about `1e9`+ behaviour in either
   direction.
4. This project's results will be small-scale. The correct frame is: *these mechanisms are or
   are not viable at research scale, and here is what would have to be measured next.*

---

## 10.4 The commitment

**[NORMATIVE]** If K1 fires, the developmental thesis is falsified **in this setting**, and
the conclusion — committed to here, in advance — is that **the value of the
genomic-bottleneck program to language modelling lies in its practical and scientific uses
and not in capability parity through evolvability**. We will report that, in those words,
with the same prominence as success.

Negative results are results. A project that cannot state in advance what would refute it is
not doing research.
