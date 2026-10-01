# SOMA — Audit v1 Defect Register and Resolutions

Two adversarial auditors read `docs/spec/00..11` in full before any code was written and
returned **~170 defects** (spec v1 → v2, Parts A and B). A second audit then read the
**repaired** specs and returned **~80 further defects, 37 of them blockers** (spec v2 → v3,
Part A'), several of which the first repair had itself introduced.

A third pass then found **26 more (A-46 … A-71), by running the code** — spec v3.0 → v3.1.
Not one of them was reachable by reading: a default with no preimage under its own link, a
gradient that is `NaN` where the forward value is finite, a cap that counted a resource as
released 50 steps before it was, an audit object constructed and never called, a driver
whose callable was bound with the wrong arity and then passed as `None`, and an
asynchronous GPU fault that no amount of reasoning about the mathematics could have located.

This register records every defect, its resolution, and the spec delta it produced, so that
the repair is itself auditable — including where the repair was wrong, which it was three
times (A-17→A-62, A-58→A-61, A-70→A-71).

**Status key:** `FIXED` resolved in spec v2 or v3 · `OPEN` deferred with a stated criterion ·
`WONTFIX` judged not a defect, with reason.

**Severity key:** `BLK` blocker · `AMB` ambiguous · `COS` cosmetic.

---

## Part A — Design resolutions (defects that changed the design, not just the text)

### A-1 · Equation (5.15) is wrong · BLK · FIXED
Re-derived independently and confirmed. The spec's form gives `sum_i (k^iT q^r) u~^i`
(missing `g^r/g^i`) and `- sum_i (k^iT q^r) g^i (S w^i)^T` (`g^i` where `g^r` is needed).
Correct only in the ungated limit `g == 1`.
**Resolution:** `O = Q<- S^T + ( Q K^T (*) Gam_decay (*) M ) ( U~ - W<- S^T )`, with
`Gam_decay[r,i] = g^r/g^i` for `r >= i` else `0`. This supplies the missing factor on the
`U~` term and cancels `W<-`'s `g^i` to leave `g^r`. Spec §05.2.5 rewritten with the full
derivation and a worked check at `g == 1`.

### A-2 · `BirthScore` is ~0 by construction · BLK · FIXED
Every tensor that touches data is soma and re-randomised each lifetime, so inheriting gene
index sets cannot improve step-0 loss.
**Resolution.** This is the correct behaviour of a pure-scaffold germline — the genome
encodes where answers will be stored, not the answers. Therefore:
- **Primary fitness statistic becomes adaptation efficiency** (`n_crit`, early-phase
  slope). §07.3 fitness weights re-derived.
- **`BirthScore` is retained as a barrier diagnostic**, predicted `~0`; a significantly
  non-zero value indicates leakage across the Weismann barrier and fails a gate.
- §10 K1 restated over the new primary statistic with an explicit boolean rule.

### A-3 · The experimental protocol is arithmetically impossible · BLK · FIXED
At the specified budget `SELECT` fires 0.7–3 times per lifetime, `ARCHIVE_STEP` 0–1,
`CERTIFY` 0. K1 as written costs 80–160 GPU-hours.
**Resolution:** two purposed configurations, and `T_0` derived from a target event count
rather than asserted.

| Config | Purpose | Params | `T_ctx` | `N_step` | `T_0` | Lifetime |
|---|---|---|---|---|---|---|
| **M-nano** | evolution dynamics, generation curve, K1/A4 | ~2.6M | 128 | 2000 | 81 | ~1.5 min |
| **M1** | TinyStories language quality | ~24.3M | 512 | 12300 | 495 | ~40 min |

`T_0 = ceil(n_c * ln(1 + N_step/n_c) / E_target)` with `E_target = 10` selection events per
lifetime, inverted from (1.17). See A-6 for a correction to this formula made during the
repair. Both configs are recorded in §07.4 and `configs/`.

### A-4 · The lagged self has the wrong sign for token selection · BLK · FIXED
*(Found during repair, not by the auditors.)* Rho-1 selection needs a reference that knows
**more** than the model. An EMA lagged self knows **less**, so `L_theta - L_ref <= 0`
almost everywhere and `w_t = max(0, .)` is identically zero — the same degeneracy the
auditors found at `n = 0` (A2), but permanent.
**Resolution:** two references with correct polarity, each keeping its own justification.
- **Token selection (§06.2)** uses a *frozen, better* reference: the **M0 baseline
  checkpoint**, which the project trains anyway. Faithful to the cited evidence.
- **Selection case weights (§04.3)** use the **lagged self improvement**
  `w_t = max(0, L_lag(t) - L_theta(t))` — literally a rate of improvement, which is what
  §04.3's "rate, not level" requirement asked for. `H_sel` is deleted as redundant.

### A-5 · Reference-forward cost · BLK · FIXED
Auditors correctly found the "+3% of step time" claim understates a full extra forward
(+25–33%).
**Resolution:** the reference is **frozen** and the natural-text corpus is **fixed**, so
`L_ref` is **precomputed once over the whole tokenized corpus** and stored as `float16`
beside the tokens (2 bytes/token). Training-time cost becomes **zero**. Synthetic tokens
have no cached reference and are ranked within their own source (also resolving D-24).

### A-6 · `T_0` derivation was inverted · BLK · FIXED
*(Found during the repair itself, not by the auditors.)* Repair A-3 introduced (1.18) to
derive `T_0` from a target selection-event count, but wrote
`T_0 = ceil(n_c / (E_target * ln(1 + N_step/n_c)))`. Inverting (1.17) correctly gives
`T_0 = ceil(n_c * ln(1 + N_step/n_c) / E_target)`. The erroneous form yields `T_0 = 32`
(M-nano) and `197` (M1), i.e. ~25 selection events instead of 10 — a 2.5x error in structural
churn, in the direction that would have *destabilised* training.
**Resolution:** (1.18) corrected; tabled values `T_0 = 81` (M-nano) and `495` (M1), verified
numerically to give `E_sel = 9.93` and `10.00`.

### A-7 · Seed count inconsistent between protocol and analysis · AMB · FIXED
§07.5 and §08.5 require `n_seed >= 5` (D-25: a 3-seed bootstrap admits only 27 resamples and
a minimum `p ≈ 0.037`), while §07.4.1's cost calculation assumed 3 seeds.
**Resolution:** the cost calculation is restated for `n_seed = 5` across two arms:
`2 * 5 * 20 * 2 * 1.5 min = 10 h`, still one overnight run.

---

## Part B — Grouped resolutions

### R1 · Notation hygiene — S1–S19, D-84, D-85 · BLK/COS · FIXED
Canonical renames applied across all files; §01.9 extended to cover every symbol.

| Was | Now | Was | Now |
|---|---|---|---|
| `N` (steps) | `N_step` | `Gamma` (decay matrix) | `Gam_decay` |
| `N` (batch tokens) | `N_tok` | `gamma` (RMSNorm gain) | `gain_rms` |
| `N` (params) | `N_par` | `gamma^r` | `g_r` |
| `T` (context) | `T_ctx` | `S` (mixer state) | `S_mix` |
| `T` (UT matrix) | `T_UT` | `S` (lexicase pool) | `S_pool` |
| `T` (token set) | `T_set` | `U~` | `U_tilde` |
| `C` (chunk) | `C_chunk` | `W` (WY matrix) | `W_WY` |
| `P` (ES population) | `P_es` | `P^r` (Householder) | `P_WY` |
| `P` (MQAR pairs) | `P_mqar` | `K1,K2` (genes) | `K_gene1, K_gene2` |
| `G` (selection events) | `E_sel` | `K1,K2` (units) | `K_unit1, K_unit2` |
| `G` (germline gens) | `G_gen` | `r` (ES rank) | `r_rank` |
| `H` (§04) | *deleted* (see A-4) | `r` (chunk position) | `r_pos` |
| `H` (§06) | `H_arch` | `k` (units/token) | `k_route` |
| `d` (growth tokens) | `D_grow` | `m` (expression dim) | `m_dim` |
| `alpha` (mixer) | `alpha_mix` | `m` (MODCOUNT modulus) | `m_mod` |
| `alpha` (significance) | `alpha_sig` | `q` (`\|cases\|`) | `q_cases` |
| `mu` (Muon momentum) | `mu_muon` | `Phi` (erank matrix) | `Act` |

`d_in`/`d_out` are bound explicitly: at a Site, **`d_in = d_out = d_model`** (see R2/B1).

### R2 · Model dimensions — B1, B5–B8, C3, C5, C6, D-08, D-45, D-46 · BLK · FIXED
- **B1** Resolved more cleanly than the auditor proposed. Rather than adding an output
  projection, **`d_out` is bound to `d_model`** and the gene library `N_G` becomes the Site's
  inner width, so `B` maps gene activations straight back to the residual stream. There is
  **no `d_ff`** and no `W_down`. This also cuts the M1 parameter count from ~38 M to
  ~24.3 M.
- **B5** `U` is **per-Site**: `U = disjoint union of U_l`. Selection, `L_met`, `R_t`,
  `r_ut` all carry a layer index. One population per Site, `L` populations total.
- **B6** The mixer is **not** cross-layer tied; §05.5's "shares the mixer" sentence was
  wrong and is replaced by the accurate statement that the *parts library* is shared while
  Sites differentiate.
- **B7/D-45** `N_G` pinned per config: M-nano `512`, M1 `2048`. The perfect-square
  constraint is **dropped** — it existed only for gene-side product keys, which R9/B36
  deletes. Only `U_max` must be a perfect square (unit routing). §05.10 recomputed from
  scratch.
- **C3** At M1/M-nano the attention layers are **full attention** (`T_ctx` is short);
  sliding-window is a scaling-path option enabled only for `T_ctx > 2048`. Removes the
  "no global path" contradiction and restores the cited study's applicability.
- **C5** Optimizer groups enumerated by name, not by tensor rank. `p_u` goes to the AdamW
  group (Newton–Schulz on an `|U| x 2` matrix is degenerate).
- **C6/D-08** Slot-preallocated tensors (`U_max`) with a liveness mask, so no optimizer
  state is ever reshaped; freed slots have their state zeroed; newborn momentum is zero.
  Full M1 and M-nano config blocks added to §07.4.
- **D-46** `D_tok` is fixed first, then `B_tok` from the batch law, then
  `N_step = D_tok / B_tok`. Circularity removed.

### R3 · Mixer — M1, M11, M12, A9, A10, C16, C17, D-79 · BLK · FIXED
- **M1** see A-1.
- **M11** Eigenvalue statement corrected to `alpha_mix * (1 - beta)` in
  `(-alpha_mix, alpha_mix)`. The `-1` transposition is approached as `alpha_mix -> 1`,
  which `sigmoid` attains to within `1e-4`; over 256 steps `0.9999^256 = 0.975`, so the
  state-tracking argument survives. A test asserts trained `alpha_mix -> 1` on parity.
- **M12** `RMSNorm_head` in QK-norm is **gain-free**, restoring the `|s| <= sqrt(d_h)` bound.
- **A10** `beta_t <= 2 - 1e-3` clamp; a condition-number assertion on the UT inverse is
  added to the regression test.
- **A9** `W_gate in R^{d_model x (d_model e_v)}`, `OutProj in R^{(d_model e_v) x d_model}`.
- **C16/D-79** The factor of 2 lives **only** in `beta_t = 2*sigmoid(.)`; the write term is
  `beta_t v_t k_t^T` with no further scaling. The double-count sentence is deleted and the
  equations are declared authoritative over the prose.
- **C17** Padding sentence rewritten: sequences are padded to a multiple of `C_chunk` and
  the pad positions are masked.

### R4 · Initialisation — M3–M7, B13, B38, N2, N3 · BLK · FIXED
Single stated criterion `d_in * Var(W_ij) = gain_act`, with
`Var(W_ij) = N_eff * E2 * Var(a) * Var(b)` and `E2 = E_{u, g in g_u}[phi_g(p_u)^2 psi_g^2]`
(second moment, not centred variance, and including `psi` — M4, M5). Solved as:

```
  Var(a_g) = 1 / d_in
  Var(b_g) = gain_act / ( N_eff * E2 )
```

removing the double-count (M3). `gain_act` for ReLU^2 is **calibrated numerically** in the
acceptance test rather than assumed to be 2 (M6). Guards `N_eff >= 1` and `E2 >= 1e-4`
abort the run (M7).
Init distributions (B13): `p_u` uniform on the unit sphere `S^{m_dim-1}`; `mu_g` uniform on
the same sphere; `log sigma_g = log(0.6)` so typical `phi ~ 0.5`; generation-0 `g_u` is
`k_g` indices drawn uniformly without replacement. `|U|_0` given per config.
The acceptance test moves to **after** phenotype construction (B38), uses two independent
estimators at different sample sizes (N2), and specifies `x ~ N(0, I)` rescaled to
`||x|| = sqrt(d_in)` with `y_ref` from He-normal (N3).

### R5 · Routing — B9–B12, A8, D-47 · BLK · FIXED
Fixed slot grid `U_max = side^2` with a liveness mask. `K_unit1, K_unit2 in
R^{side x (d_key/2)}` declared **soma** (D-47 — if they were germline, §09.3's bottleneck
claim would be false by construction). Dead, masked, and dying slots score `-inf` in
**both** the top-k and the softmax, which makes Theorem 3.1 true (B9, B10). A newborn is
assigned the nearest free slot to its parent by grid distance, and inherits the parent's
key rows as its initial addressing (B11). Scores are divided by `sqrt(d_key/2)` (A8).

### R6 · Selection — B17–B25, B32, A1–A5, A16, D-16, D-27, D-57 · BLK · FIXED
- **B17** (the degeneracy that would have made ε-lexicase inert)
  `eps_t = MAD over { u : nu_ut > 0 }`, i.e. over units with non-zero credit on that case,
  not over the whole population.
- **B18/B19** Resolved by A-4: `w_t` is itself the improvement signal. `H_sel`, `H_min`,
  `H_max` deleted; no per-case loss history is needed.
- **B20** Utility made computable:
  `h_u = sum_t r_ut ||Delta_ut||_2 / sum_t r_ut`;
  `out_u = sum_{g in g_u} phi_g(p_u) ||b_g||_1`; `in_u = sum_{g in g_u} phi_g(p_u) ||a_g||_1`;
  `y_u = |h_u - h_bar_u| * out_u / in_u`, then normalised by the population median.
- **B21/A16** `tau_death` is an **absolute** threshold, calibrated once at the first
  selection event from the 10th percentile over `{u : age > M_mat, not dying}` and
  thereafter fixed; deaths are capped symmetrically with births at `floor(rho * |U|)`.
- **B22** `nu_child = 0` and `M_mat` provides protection; `nu_hat` is defined as `0` at
  `age = 0`.
- **B23** All periodic hooks guarded by `n > 0`; first selection additionally requires
  `n >= max(T_0, M_mat)`.
- **B24/D-16** The buffer is a **reservoir sample** of fixed size `ceil(D_min / d_rate)`,
  storing only scalars per (case, unit): `||Delta_ut||_2`, `r_ut`, plus `w_t` per case.
  Memory is bounded and added to §05.10.
- **B25** The leave-one-out cost claim is corrected to a partial forward from the Site's
  layer to the head, on a 1% audit subset.
- **B32** A `dying` flag excludes units from `parents`, `deaths`, `n_par`, the `tau_death`
  calibration, and routing.
- **A1** Births are drawn uniformly from the parent multiset subject to `cap_off`.
- **A2** `SHUFFLE` falls back to a uniform permutation when `sum w_t = 0`.
- **A3** Inner loop guard corrected to `|T'| >= 1`.
- **D-27** "case win" defined as *being in the surviving pool after the first case of a
  selection event*; computed as a **diagnostic independent of the selection operator** so
  it exists in both arms of A2 (D-57).
- **D-57** A2 replaces both birth and death rules: parents are the top-`n_par` by
  `nu_hat`.

### R7 · Development — B26–B31, C9, C10, D-22 · BLK · FIXED
- **B26** The bias-transfer step is **deleted** (the model has no biases). Death anneals
  `m_u -> 0` over `phi_ann` and then frees the slot.
- **B27/B28/N7** The bifurcation alarm applies to **births only**. Estimator:
  `|L_task(probe; after) - L_task(probe; before)|` on the fixed probe batch of §03.4.
  `epsilon_bif = 3 * sd` where `sd` is the probe's batch-to-batch standard deviation
  measured at init. On trip, `rho <- rho / 2`.
- **B29/D-22** `GROW` defined: at M1/M-nano the only grown dimension is the **unit
  population**, and a growth event is a scheduled large birth of `g_growth * |U|` units
  through the ordinary MSG path. The token-count law (3.9) applies at M4+ only; at
  M1/M-nano the schedule is explicit fractions of `N_step`. A8 becomes `{1, 3, 8}` events.
- **B30** `MUTATE` draws **one** categorical kind; postcondition `|g'| = k_g` exactly;
  padding re-draws from the parent's genes; `neighbour(i)` defined as grid-adjacent under
  the `(i1,i2)` decomposition; the "nearest unit" rule has a uniform-random fallback.
- **B31** `BIRTH(u_p1, u_p2 | None, n)`; `L_u` carries a parent tuple; the crossover pool
  is the **parent multiset** (the Inventory is used only for seeding).
- **C9** Theorem 3.1 split into 3.1a (unit birth — needs only routing exclusion) and
  3.1b (width/depth/head growth — needs the masked-normalisation lemma).
- **C10** "bitwise" weakened to "unchanged up to floating-point reduction order"; the test
  tolerance stays `1e-6`.

### R8 · Metabolic and population control — B14–B16, A6, A7, D-15, D-36 · BLK · FIXED
- **B14/B15/D-15** The price and the cap are separated. `U_cap` is an explicit germline
  integer; `BUDGET_HEADROOM() = max(0, U_cap - |U_live|)`. The `lambda_size` term is
  retained in the loss only as a *reported diagnostic* with an honest note that it has no
  gradient; the equilibrium claim is restated as a claim about the **cap plus the
  death rule**, not about a gradient.
- **B16** Positions are constrained to the unit sphere `S^{m_dim-1}` (renormalised after
  each step) and `q_t` is likewise normalised, so the wiring term is bounded in `[0,4]`
  and cannot drive a collapse to a point. Collapse is nonetheless measured by gate
  `G-NOCOLLAPSE` (mean pairwise position distance must not fall below a threshold).
- **A6** The wiring term is weighted by `r_ut`.
- **A7** Both `L_met` terms are per-token normalised.
- **D-36** A1 split into **A1a** (`lambda_wire = 0`) and **A1b** (`lambda_size = 0`);
  since headroom no longer depends on `lambda_size`, A1b cannot cause unbounded growth.

### R9 · Germline and ES — B33–B37, C7, C14, A14, A15, D-17, D-34, D-35, D-38, D-49, D-50 · BLK · FIXED
- **B33/D-38** **Single-point ES** with **mirrored sampling**: `P_es` is even and members
  are `(+E_i, -E_i)` pairs. §07.5's "population" wording is corrected. `G_gen` arithmetic
  restated as `G_gen = floor(T_run / (P_es * T_life))` in both places, with the M-nano
  numbers that make `G_gen >= 20` actually reachable.
- **B34/D-34** Link functions extended: `softmax` for simplex constants, affine-composed
  sigmoid `lo + (hi-lo)*sigmoid(z)` for bounded intervals, `round(softplus(z))` for
  integers. The postcondition is restated as holding *after rounding*.
- **D-35** Ablations **pin** their parameter *and* exclude it from `E_i`, via an explicit
  `ablation.pin` config path that bypasses the germline parameterisation entirely — which
  also makes endpoint values (`0`, `1.0`) reachable.
- **B35** One canonical policy table in §02.2.3; §03.6 and §04.8 now cite it rather than
  restating it.
- **B36** Gene-side product keys are **deleted**. `g_u` is a set manipulated by
  `BIRTH`/`MUTATE`; `neighbour(i)` is defined combinatorially on the `(i1,i2)`
  decomposition of the gene index. This removes dead germline parameters from the ES
  search space.
- **B37** `N_G` constrained to be a perfect square (satisfied by 256 and 1024).
- **C7** `log sigma_pos` stored unconstrained with a numeric default; the
  `0.1 * mean(sigma_g)` expression survives only as the generation-0 seeding heuristic.
- **C14** Germline/soma membership stated exhaustively in one table.
- **A14** 1-D germline tensors are perturbed full-rank; only 2-D tensors use the low-rank
  form. Notation corrected.
- **A15** Mirrored sampling makes `P_es = 2` a proper antithetic pair (see B33).
- **D-17** `RANK_NORMALISE` defined: average ranks for ties, mapped to `[-0.5, 0.5]`.
- **D-49** One RNG stream **per purpose**, all listed in the seeds schema: `data`,
  `init`, `dropout`, `mutation`, `lexicase`, `archive`, `es_perturb`, `member`,
  `horizon`, `downsample`, `growth`, `explore`.
- **D-50** `r_3`'s period restated as "one germline generation (`P_es` lifetimes)".
- **D-18** `SEED_POPULATION` fully specified, including the empty-Inventory case.

### R10 · Open-ended stream — D-01 to D-14, D-19, D-23, D-24, D-29, D-33, D-43, D-51, D-68, D-69, D-87 · BLK · FIXED
- **D-01/D-02** All periodic hooks use explicit next-event counters
  `n_next = n + ceil(T_x(n))`; eq (4.0) restated for that recursion.
- **D-03** `H_arch` defined in archive-steps with a default; added to §01.9.
- **D-04** A complete `EDIT` table: parameter, type, min, max, step, selection weight;
  duplicate `lambda` rejected.
- **D-05** `GRID`'s transform set enumerated explicitly (identity, rotate90/180/270,
  reflect-h/v, translate, recolour-permutation, tile) with a total verifier.
- **D-06** `s(lambda)` protocol fixed: greedy decode, `max_new = answer length`,
  `n_eval = 32` instances, verified by exact match; no sampling.
- **D-07** `tau_admit` raised to `0.30` (excludes `s < 0.08` and `s > 0.92`).
- **D-10/D-11** `p_explore` adaptation rule made explicit; (6.4) normalised by
  `sum_i I_i`, and the batch-token symbol renamed `N_tok`.
- **D-12** `kappa` is stated **per update** and derived from a target half-life in steps.
- **D-13** The frozen-checkpoint ring keeps checkpoints nearest to `N_step * 2^-j`,
  residency **CPU**, and is added to the memory table.
- **D-14** Certification batches are captured every `T_cert / K_ring` steps so the ring
  spans the run.
- **D-19** Mann–Kendall fully specified: original statistic, tie-corrected variance, exact
  null for `n <= 10`, one-sided, rows with `n < 6` excluded from numerator **and**
  denominator, Benjamini–Hochberg across rows.
- **D-23** Synthetic tasks are serialised over a **reserved symbol block** (ids `0..63`)
  that the tokenizer never merges, guaranteeing one symbol per token; the loss is masked
  to the answer span.
- **D-24** Token selection ranks **within source** (natural vs synthetic), which also
  de-confounds A5 from A6. The realised synthetic share of selected tokens is logged.
- **D-29** Archive sampling uses an epsilon-uniform mixture (`eps_arch = 0.1`) so a
  fully-mastered archive cannot freeze; `s_H` initialised to `s` at insertion.
- **D-33** Refresh is **round-robin oldest-first**, making the staleness postcondition
  achievable.
- **D-43** An eval-length band is **held out** from the archive's admissible `lambda`
  range, and the exclusion is asserted at admission time.
- **D-51** "asserts ... and logs a warning" resolved to **raise**; the archive's inference
  cost is added to §07.8.
- **D-68** `Delta` is computed in `float32` for both terms.
- **D-69** Ranking is defined over the **micro-batch**, with the change stated explicitly.
- **D-87** `f_oe` decay branch guarded; archive and certification are suspended once
  `f_oe = 0`.

### R11 · Protocol and budget — D-30, D-31, D-37, D-39, D-40, D-41, D-42, D-65, D-66, D-90 · BLK · FIXED
- **D-30/D-31** See A-3.
- **D-37** "generation" now means **germline generation** (`G_gen`) exclusively; a
  selection event is an **event** (`E_sel`). Both added to §01.9.
- **D-39** The lifetime budget is fixed in **tokens**, and `CriterionScore` is computed
  against a fixed reference `N_step^ref` so a larger realised `N_step` confers no
  advantage. Horizon jitter is applied to the *data order*, not the budget.
- **D-40** See A-2.
- **D-41** `L*` is M0's validation CE **at a token budget equal to `T_life`**, not at M0's
  full budget; unreached criteria are **right-censored**, not scored `0`.
- **D-42** `L_rand = ln V` (`= 9.01` nats at `V = 8192`), a shape-independent constant.
- **D-65** The M0 reference is named numerically per config, with corpus, tokenizer, token
  count, and the target CE; "matches the reference shape" is replaced by a numeric bar.
- **D-66** M1 criteria given numeric bars; criterion 4 replaced by a Mann–Kendall trend
  test (see D-70).
- **D-90** The `n_c(g)` early-warning framing is dropped; the required `G_gen` for power is
  stated instead.

### R12 · Evaluation — D-20, D-21, D-25, D-26, D-52 to D-64, D-70 to D-77 · BLK · FIXED
- **D-20** `Q_route` fully defined: bipartite graph of the fixed probe batch's tokens
  against live units, edge weight `sum_t r_ut`, **Barber bipartite modularity**, Leiden
  communities with a fixed seed and resolution `1.0`, computed per Site and averaged;
  range corrected to `[-1, 1]`.
- **D-21** Generation quality frozen as a versioned artifact: fixed prompt set, fixed
  rubric with anchors, fixed decoding parameters, fixed grader with a pinned model id, and
  a stated inter-rater agreement check.
- **D-25** The generation-curve test is specified: statistic, resampling unit (seed), one
  sided, BCa interval; the required seed count for `alpha_sig = 0.05` is stated.
- **D-26** "Noise floor" defined **per metric** as the standard deviation across three
  identical-seed M0 repeats.
- **D-52/D-53** **Equivalence is now decidable.** Each ablation declares a primary metric
  and a TOST margin; "unchanged" means the two one-sided tests reject at `alpha_sig`. K2,
  K3, K4, K10 name a small pre-registered primary metric set rather than "every metric".
- **D-54** `pass@k` corrected to `1 - C(n-c, k)/C(n, k)` with `n`, the `k` grid, and the
  decoding temperature stated.
- **D-55** B2's forward pass specified exactly: the germline field is **frozen at its
  random initialisation** (not removed), so the Site still exists; only births, deaths,
  selection, the metabolic term, and inheritance are disabled.
- **D-56** B3 specified: routing assignment fixed per unit slot at init and held constant,
  `r_ut = 1/k_route`, and the wiring term disabled (it is meaningless under random
  routing).
- **D-58** A3 fixes `|U|` at the full model's **measured equilibrium** from a prior run.
- **D-59** A9 redefined for this architecture: freeze `p_u` and the gene sets, train
  `A`, `B`, `W_shared` — a genuine test of adaptivity within a fixed body plan. The ANIL
  framing is dropped as inapplicable.
- **D-60** A4 specified: a random germline is accepted only if it passes §02.5 on the
  **first** attempt (no resampling, which would itself be selection); the retry policy and
  the empty-Inventory seeding rule are given; the perturbation scale is matched across arms.
- **D-61** Stage assignment completed for all of A1a–A11; §00.3's "A1-A9" corrected.
- **D-62** Every ablation declares its matching basis (matched **tokens**, with measured
  wall-clock reported alongside).
- **D-63** Rule 2 replaced by a static assertion that no task-identity feature reaches the
  model.
- **D-64** The `pass@k` grid and its inference budget are specified.
- **D-70** K8 and M1 criterion 4 replaced by a Mann–Kendall trend test on `erank` with a
  stated slope threshold.
- **D-71** Certification gains a **control arm** (frozen archive), and the criterion is
  the *difference* between evolving and frozen. Without it the test cannot distinguish an
  open-ended archive from difficulty tracking.
- **D-72** K1 restated as an explicit boolean over two pre-registered tests.
- **D-73** K4 now rests on the defined `Q_route`, the split A1a, and the ablation pin path.
- **D-74** K6 given a mechanical form: `max |chunk - ref| / (|ref| + eps) > 1e-5` on the
  §05.2.6 grid.
- **D-75** An explicit `N_G x d_model x |U|` sweep is added to §08 so K9 is testable.
- **D-76** K7 gains a length-sweep (accuracy vs length curve, not a single point).
- **D-77** Every §10 decision is made mechanical.
- **D-67** `erank` cadence coarsened to `10 * T_log` and costed.

### R13 · Bibliography — D-78, D-80 to D-83 · AMB/COS · FIXED
- **D-78** Added C-23 batch-size law, C-24 cross-layer sharing, C-25 Peri-LN, C-26 QK-norm
  entropy collapse, C-27 attention sinks, C-28 ReLU^2, C-29 embedding tying threshold.
- **D-80** `[OPEN]` markers added to §03.3 (masked-growth scale), §04.5 (specialist
  evidence), §02.2.2 (hypernetwork ceiling) where a single unverified source is the sole
  support.
- **D-81** Both citation identifiers re-verified and corrected.
- **D-82** The `rho` unit conversion (per-step vs per-event) stated explicitly in §03.5.
- **D-83** C-13's limitations carried into §09.2's Risk column.

### Untestable normative claims — N1–N9 · FIXED
- **N1** A `[RATIONALE]` marker is introduced and prose-only claims are reclassified;
  `[NORMATIVE]` now marks only mechanically checkable statements.
- **N2/N3** See R4.
- **N4** The barrier guard is verified by a **mutation test**: the suite must fail when
  `ASSERT_BARRIER` is monkeypatched to a no-op.
- **N5** The licensing statement is reclassified `[RATIONALE]`.
- **N6** "Structural change" defined as any change to `U`, `g_u`, or `p_u`; `R_t` and
  `psi` are explicitly exempt.
- **N8** Selection-cost trip rule: `d_rate <- max(0.02, 0.7 * d_rate)` over a 1000-step
  window.
- **N9** Credit-audit rule: minimum 64 pairs, doubling on trip, capped at 10%.

### Remaining cosmetic items
C4, C8, C11, C12, C13, C15, A17, A18, M8, D-86, D-88, D-89 — all applied as written in the
audit reports.

---

## Part A' — Audit v2: defects in the v1 *repair* (spec v2 → v3)

A second adversarial audit read the twelve v2 files in full and returned **~80 defects, 37 of
them blockers**. The defect count fell 170 → 80 and the character shifted from architectural
to missing-constants and wrong-signs — convergence, not completion.

**The finding that matters most about this table:** several of the worst entries were
*introduced by the v1 repair*. A repair pass is itself a source of defects, and A-8 in
particular is a claim the author asserted confidently and in writing before checking the
algebra. This is recorded here because a register that logs only the original author's
mistakes, and not the auditor's, is not a register.

### A-8 · Per-unit nonlinearity was false under the v2 placement · BLK · FIXED
v2 applied `sigma_act` to `z` **before** the gate and then forbade the aggregate form on the
grounds that aggregating "destroys the premise that a unit is a distinct function". With the
activation before the gate both `B` and the gate are linear, so

```
  sum_u r_ut m_u B[ gate_u (*) sigma_act(z) ]  ==  B[ (sum_u r_ut m_u gate_u) (*) sigma_act(z) ]
```

**exactly**. Every unit already *was* a linear reweighting of one shared nonlinear feature,
the two forms computed bit-for-bit the same function, and v2 mandated a `k_route`-fold cost
increase to preserve a distinction that did not exist.
**Resolution.** The activation moves **after** the gate: `F = B[sigma_act(gate (*) (A x))]`
(1.3), (5.19)–(5.20). The support is still `k_g`-sparse, so this costs nothing, and (1.5) is
now a genuine inequality. Because the original claim was asserted and wrong, the corrected
claim carries a **mandatory ablation, A12**, which runs the pre-gate placement at exact
parameter parity and tests whether the algebraic distinction does empirical work.

### A-9 · The sphere does not prevent positional collapse · BLK · FIXED
v2 (register entry B16) claimed the unit-sphere constraint meant `L_wire` "cannot drive a
collapse to a point". False: `L_wire >= 0` with equality **iff** every routed `p_u` equals its
`q_t`, that configuration lies on the sphere, and it is the **global minimiser**. Worse,
§05.4 states `W_pos` is "used only by `L_wire`", so `q_t` had **no other gradient source** and
was driven purely toward collapse. The sphere bounds the term's *value*, not its *argmin*.
**Resolution.** A dispersion term `L_disp` (1.9) — Gaussian repulsion between live positions,
`s_rep = 0.5` — makes the equilibrium a dispersed configuration (a Thomson problem on
`S^{m_dim-1}`), which is the character-displacement mechanism the ecological analogy invoked
and never implemented. `G-NOCOLLAPSE` is given a numeric threshold (1.13),
`mean ||p_u - p_v|| >= 0.60 * D_unif(8) = 0.825`; v2 referenced the gate without ever
defining one.

### A-10 · `T_arch` and `T_cert` had no value anywhere · BLK · FIXED
The v1 repair derived `T_0` from a target event count and **deleted v1's formula for the other
two hooks without replacing it**. Whether the open-ended machinery fires at all — the largest
single defect v1 had — was therefore made *undecidable* rather than repaired.
**Resolution.** (1.18)–(1.19): `T_arch = max(1, floor(N_step/A_target))` with `A_target = 20`,
`T_cert = max(1, floor(N_step/C_target))` with `C_target = 4`. Realised counts are tabulated
in §01.9 and asserted by `G-SPECNUM`: 20/4 at M-nano and 20/3 at M1.

### A-11 · `q_cases` was inversely proportional to `d_rate` · BLK · FIXED
v2 set `q_cases = ceil(D_min / d_rate)`. §04.5.1's cost governor reduces `d_rate` when
selection exceeds 2% of step time, so each trip **raised** the case count and therefore the
cost it was meant to bound — unstable positive feedback running `80 → 115 → 163 → ... → 400`
at the `d_rate` floor. It also inverted the semantics of the cited down-sampling evidence, in
which a lower rate means *fewer* cases.
**Resolution.** `q_cases = max(D_min, round(d_rate * B_tok))` (§01.10).

### A-12 · The noise floor was exactly zero, so every TOST margin was zero · BLK · FIXED
v2 defined the per-metric noise floor as the standard deviation over "**3 runs with identical
seeds**". Under §07.7's own determinism requirement identical seeds give identical runs, so
every noise floor was **exactly 0**, every `delta = 2 * noise_floor` margin was 0, and every
TOST depending on one (A3, A5, A6, A8, A10, A11, A12, B3; kill criteria K2, K3, K10) was
undecidable in the equivalence direction — reintroducing precisely the structural defect
§08.5 was written to remove.
**Resolution.** Five repeats at **different** run seeds (§07.7). The seed-invariant
nondeterminism (cuBLAS reduction order) is measured separately as `sd_probe` over `k_probe`
repeats and is used only for `epsilon_bif`; the two are never substituted.

### A-13 · The v2 initialisation failed its own acceptance band · BLK · FIXED
v2 asserted `Var(W_ij) = N_eff * E2 * Var(a) * Var(b)` with `N_eff = E_u[||phi (*) 1_g||_1]`.
For a sum of `k_g` independent rank-one terms the correct prefactor is **`k_g`**, not
`N_eff`: `N_eff = k_g E[phi]` is a *first* moment while `E2` already carries `phi^2`, so the
product is cubic in `phi` where the variance is quadratic. Measured at the v2 initialisation
the realised `d_in Var(W)` is `3.94 * gain_act` — outside §02.5's own `[0.5, 2.0]` band, so
v2 as written would have **aborted every run** at `LIFETIME` step 3.
**Resolution.** The fragile analytic chain is removed entirely. `Var(a_g) = 1/d_in` (1.6) and
`Var(b_g)` is **fitted** by `CALIBRATE` (1.7) against the *measured* output variance, so the
acceptance test becomes the definition rather than a check on a formula that may itself be
wrong. `N_eff` survives as a reported diagnostic with its abort guards.

### A-14 · `sigma_g = 0.6` gave a median `phi` of 0.062, not 0.5 · BLK · FIXED
v2 set `log sigma_g = log(0.6)` with the comment "so a typical `phi` is near 0.5". Measured
over `4e5` samples at `m_dim = 8`, the median is **0.062** and `N_eff` falls to `0.79` —
below §01.5's own abort guard `N_eff >= 1`.
**Resolution.** `sigma_g = 1.2`, at which `E[phi] = 0.5147`, `E[phi^2] = 0.2809`,
`median[phi] = 0.4995`. The moments are recorded as `[EMPIRICAL]` with their sample size.

### A-15 · M-nano violated its own mixer shape constraint · BLK · FIXED
v2 set `n_m = 2, d_v = 64` at M-nano against `d_model = 192`, violating `n_m d_v = d_model e_v`
(`128 != 192`) and producing a hard shape error in §05.2.4, where `RMSNorm(o)` is multiplied
elementwise by a `d_model`-dimensional gate.
**Resolution.** `n_m = 3` at M-nano (§01.9, §05.0).

### A-16 · `N_G` differed between configurations, making inheritance shape-impossible · BLK · FIXED
v2 (register entry B7) pinned M-nano `N_G = 512` and M1 `N_G = 2048`. But §02.6.2's whole
argument for `G_gen = 1` at M1 is that the germline is **inherited from the M-nano lineage**,
and `mu_g` has shape `N_G x m_dim` — so the inheritance was shape-impossible. §09.3's claim
that `|Gamma|` is independent of scale was also left resting on an argument rather than on
construction.
**Resolution.** `N_G = 2048`, `k_g = 8`, `m_dim = 8`, `J_max = 256` are **germline-scale
constants, identical at every configuration** (§01.2). Only the soma scales. §08.6.1's `N_G`
sweep is correspondingly re-centred on `{256, 512, 1024, 2048, 4096}` so that it brackets the
default rather than stopping below it. The perfect-square constraint on `N_G` (register entry
B37) is withdrawn — it existed only for the gene-side product keys that B36 deleted.

### A-17 · `BUDGET_HEADROOM` was evaluated before deaths, permanently zeroing births · BLK · FIXED
v2's `BUDGET_HEADROOM() = max(0, U_cap - |U_live|)` was evaluated *before* the event's deaths,
and `DEATH` keeps `live_u = True` for a further `phi_ann` steps. Once the cap was reached
births were therefore **permanently zero**, and ε-lexicase would have produced parents that
could never reproduce.
**Resolution.** `BUDGET_HEADROOM(n_death) = max(0, U_cap - (|U_live| - n_death))` (1.14),
evaluated net of the deaths applied in the same event; §04.2 applies deaths before births.

### A-18 · `tau_death` on `nu_hat` could not be absolute · BLK · FIXED
v1 recomputed a 10th percentile every event, guaranteeing a 10% kill rate regardless of
fitness. v2 made the threshold absolute but stated it on `nu_hat`, which (3.8) divides by the
population median — so `nu_hat` has median exactly 1 at every step **by construction** and a
fixed threshold on it still selects a fixed fraction. The v2 repair did not remove the
behaviour it claimed to remove.
**Resolution.** `tau_death` is calibrated on, and compared against, the **un-normalised**
bias-corrected utility `nu_raw = nu_u / (1 - eta_ema^age)` (§04.4).

### A-19 · `h_bar_u` was used but defined nowhere · BLK · FIXED
(3.6)'s `y_u = |h_u - h_bar_u| * out_u / in_u` and §01.11's symbol table both referenced
`h_bar_u`, which v2 never defined. The entire death rule was not implementable.
**Resolution.** (3.5b): a per-unit EMA of `h_u` at the same decay as the utility, initialised
to zero — which makes `y_u` a *deviation from the unit's own recent contribution*, i.e. a
novelty signal rather than a magnitude signal.

### A-20 · `eps_t` was NaN whenever `w_t = 0` · BLK · FIXED
v2 fixed v1's identically-zero `eps_t` by taking the MAD over `C_t = {u : nu_ut > 0}`. But
`C_t` is **empty** whenever `w_t = 0`, which (4.1) makes common — any case on which the model
has not improved since the lag. A median over the empty set is undefined, so `eps_t` was NaN
on those cases and the precondition `eps_t >= 0` could not be met.
**Resolution.** (4.7): `eps_t = 0` when `C_t` is empty, which reduces the filter to strict
lexicase on a case carrying no learnability signal — the correct behaviour, since such a case
should not widen the pool.

### A-21 · `GROW` saturated the cap on its first event, making A8 inert · BLK · FIXED
v2 sized a growth event as `min(floor((g_growth-1)|U_live|), BUDGET_HEADROOM())`. With
`g_growth = 4` that requests `3|U_live|` new units and **saturates `U_cap` on the very first
event** at both configurations; the 2nd and 3rd scheduled events then violate `GROW`'s own
precondition, so all three arms of ablation **A8** would have produced an identical
trajectory.
**Resolution.** The remaining headroom is divided by the number of remaining scheduled events
(§03.5.1), so the schedule length actually matters.

### A-22 · `cap_off` was unreachable, zeroing two diagnostics · BLK · FIXED
v2 fixed `cap_off = 6` at M1 while `n_birth <= floor(rho_0 |U|) <= 19`, but offspring are
drawn from at most `n_par` parents — and at the realised numbers no unit could exceed 6 draws.
§04.10's falsifier ("the cap binds in more than 20% of events") and §07.6's hit-rate log were
therefore identically zero **by arithmetic**, not by the population being healthy.
**Resolution.** `cap_off = max(2, ceil(0.5 * n_birth_max))` with
`n_birth_max = floor(rho_0 * U_cap)` (§04.6), so the cap is reachable and the diagnostic
carries information: 6 at M-nano, 10 at M1.

### A-23 · The certification ring's cadence contradicted its own firing rate · BLK · FIXED
v2 captured batches every `T_cert / K_ring` steps "so the ring spans the run". That requires
`T_cert ~ N_step`, at which point certification fires at most once — and with the counter of
(1.15) starting at `T_cert`, **zero** times.
**Resolution.** Capture cadence is decoupled from certification cadence: capture every
`ceil(N_step / K_ring)` steps, fire every `T_cert` steps (§06.4). Realised: 4 certifications
at M-nano, 3 at M1, each over a ring spanning the whole run.

### A-24 · The task archive was never initialised · BLK · FIXED
`ARCHIVE_STEP` step 1 samples a parent from `Lambda`, and nothing anywhere in v2 created
`Lambda`. The first firing sampled from an empty set; the entire task stream was undefined.
**Resolution.** `SEED_ARCHIVE` (§06.3.3), called at `LIFETIME` step 5, inserts each of the
seven families at the lower bound of every `EDIT` range, unconditionally — unconditionally
because `tau_admit` is a learnability gate on a *variation*, and at initialisation every
generator reads `s_task ~ 0` and would be rejected, leaving `Lambda` empty forever.

### A-25 · `Delta_i` is undefined for synthetic tokens · BLK · FIXED
`L_ref` exists only for natural text, so (6.3) has no value on the synthetic source. v2's
"within-source ranking" cross-reference fixes the *comparability* of the ranking, not the
*existence* of the quantity ranked.
**Resolution.** Synthetic tokens are ranked by **raw** `L_theta` within their own source
(§06.1.1), stated explicitly as a weaker criterion, justified by the archive's own
`Learn = 4s(1-s)` eviction keeping surviving generators away from the already-learned regime.

### A-26 · `theta_lag`'s forward cost was unbudgeted · BLK · FIXED
§04.3's `w_t` needs `L_lag(t)`, i.e. a forward pass of `theta_lag`, which cannot be
precomputed because `theta_lag` changes. v2 budgeted only "one parameter copy every `T_ref`
steps" and placed `theta_lag` on the **CPU** — reintroducing under a different name exactly
the unbudgeted reference forward that A-5 removed for `theta_m0`.
**Resolution.** `L_lag` is evaluated **only on the `q_cases` cases the reservoir retains**
(§06.1.2), so the cost is a forward on 10% of the batch, `+3.3%` of step time, asserted by
the §07.8 profiling gate. `theta_lag` becomes GPU-resident and the buffer stores each retained
case's context window (§04.6, §05.10).

### A-27 · The `L_ref` cache was not bound to its packing · BLK · FIXED
`L_ref` is context-conditional, so a cached value is valid only under the packing that
produced it — yet §07.3 mandates data-order jitter, which would silently invalidate every
cached value.
**Resolution.** The sidecar records
`(tokenizer_hash, corpus_hash, m0_checkpoint_id, T_ctx, packing_scheme, doc_boundary_policy)`
and is refused at load on mismatch; attention is masked at document boundaries; and the cache
is keyed by `(doc_id, position_in_doc)` rather than by stream position (§06.1.1).

### A-28 · A frozen-archive control cannot test what K5 asks · BLK · FIXED
v2's single control arm separates "the archive changes" from "the archive is static" — which
is **not** the confound §06.4 itself names. A pure difficulty-tracker also beats a frozen
archive on novelty, so the test fired identically for genuine open-endedness and for monotone
difficulty ramping.
**Resolution.** A third **ramp** arm — a scripted monotone ratchet on `n_len` alone, matched
to the live arm's solve-rate trajectory. K5 is adjudicated on `live - ramp`, with
`live - frozen` retained as a sanity floor (§06.4, §10 K5).

### A-29 · K5's margin was below the granularity of its own statistic · BLK · FIXED
With `J_ring = 8` and the `n >= 6` exclusion, at most 3 rows qualify, so the per-firing
novelty fraction is quantised to `{0, 1/3, 2/3, 1}`. v2's `delta = 0.10` equivalence region
sat entirely inside the gap between two attainable values: **neither TOST could ever reject**
and K5 was undecidable in both directions.
**Resolution.** `delta = 0.34` (§06.4, §08.5), with a general rule added to §08.5 that every
margin must exceed the granularity of its statistic.

### A-30 · Three inconsistent `n_len` rules, and no clipping against `T_ctx` · BLK · FIXED
§06.3.5 stated `n_len <= 32`, the `EDIT` table permitted `4..96`, and step 3 rejected only the
`40..256` band — which admits `33..39`. Separately, no parameter was clipped against context
length: `GRID` at `h = w = 8, n_ex = 4` serialises 640 symbols against `T_ctx = 128` at
M-nano. And the holdout covered `n_len` only, while the suite also generalises along `P_mqar`
and `depth`, so the MQAR and DYCK bars were confounded by archive training.
**Resolution.** One rule, `4 <= n_len <= 32`, superseding the table; `P_mqar <= 16`,
`depth <= 8`; and rejection of any `lambda` serialising beyond `T_ctx - 2` (§06.3.5).

### A-31 · A1b was a provable no-op · BLK · FIXED
§01.6.1 establishes `L_size` has **identically zero gradient** (`m_u` is bookkeeping), so
v2's A1b (`lambda_size = 0`) could not change a single trajectory. It would have been reported
as a null result about a mechanism that was never connected.
**Resolution.** A1b is redefined on the mechanism that actually bounds the population:
`U_cap = U_max` (§08.6).

### A-32 · A7 was unmeasurable at its assigned stage · BLK · FIXED
A7's primary metric is `PARITY scaled_acc` at length **256**, and M-nano's `T_ctx = 128`
cannot hold a length-256 prompt plus its answer. The v2 stage assignment "nano + M1" was
unsatisfiable, as was every `40..256` row of §08.2 at M-nano.
**Resolution.** A7 runs at **M1 only**; M-nano runs a reduced `40..96` evaluation band with
pass bars unchanged (§08.6).

### A-33 · M-nano had no baseline reference at all · BLK · FIXED
M0 is defined at `V = 8192, T_ctx = 512`; M-nano runs at `V = 4096, T_ctx = 128`. `L_star`,
`slope_ref` and `theta_m0` are therefore all incomparable at M-nano — the CEs are over
different vocabularies and the `L_ref` cache is refused by A-27's binding rule — while
`CriterionScore` (the largest fitness term) and all of §06.2 depend on them. The decisive
experiment, which runs **at M-nano**, had no reference.
**Resolution.** **M0-nano** is a required artifact: the §07.4.2 recipe at M-nano corpus
settings, producing `L_star_nano`, `slope_ref_nano`, `theta_m0_nano` and its own `L_ref`
sidecar (§07.1).

### A-34 · Censoring made `CriterionScore` silently constant · BLK · FIXED
v2 specified right-censoring but not its consequence: a generation in which every member is
censored has zero variance in the primary statistic and contributes no ES signal — the same
class of defect as A-2's zero-by-construction `BirthScore`.
**Resolution.** `ES_STEP` falls back to `SlopeScore` for a fully-censored generation, the
substitution is logged, and the censored fraction is a reported field with a `> 0.5`
interpretability threshold (§07.1).

### A-35 · `sep(n)` aborted the final steps of every run · BLK · FIXED
v2 exempted warmup from the `sep(n) <= 1` assertion but not decay. As `a_lr -> 0` the ratio
diverges (`sep = 6.31` at `n = 12299` at M1), so the assertion fired near the end of every
lifetime.
**Resolution.** The assertion holds for `n_warm <= n < n_dec` only, and §01.7.1 states
explicitly what is and is not claimed: `b_str` is summable **by design**, so the slow variable
freezes rather than converging jointly, deliberately failing the persistent-excitation
condition of two-timescale stochastic approximation. No joint-equilibrium convergence is
claimed.

### A-36 · The bifurcation alarm was unreachable and its threshold undefined · BLK · FIXED
v2 measured `|L_task(after) - L_task(before)|` **at** a birth. Theorem 3.1a guarantees that
difference is zero to reduction order, so the alarm was unreachable, `rho <- rho/2` was dead
code, M1 criterion 2 was trivially satisfied and §03.7's falsifier was inert. `epsilon_bif`
was additionally defined as the "batch-to-batch standard deviation" of a batch the same
sentence mandated be **fixed** — a quantity that does not exist.
**Resolution.** The alarm is measured **after the mask anneal completes**, against a masked
counterfactual (1.21), and `epsilon_bif = 3 * sd_probe` where `sd_probe` is the standard
deviation over `k_probe = 8` repeated passes on the fixed probe under the run's
nondeterminism (§01.7.2).

### A-37 · `T_life` was asserted, not measured · OPEN
v2 asserted `T_life = 40` min at M1. That implies `6 N_par D_tok / t = 48.6 TFLOP/s`, about
**47% of this card's dense bf16 peak**, from the `6ND` term alone and before attention, norms,
routing, sparse gathers, `float32` triangular solves, held-out evaluation, `erank` SVDs,
certification and archive decoding — while §07.8's own rule forbids asserting an unmeasured
wall-clock figure.
**Deferral criterion.** `T_life` is `[OPEN]` at both configurations (§01.9) and is measured at
gate `G-M0`; `G_gen` and the whole M3 budget are recomputed from the measurement before the
decisive experiment is scheduled (§07.4.1). If the measured `T_life` exceeds 1.8 min at
M-nano, one of `n_seed`, `G_gen`, `N_step` is reduced **before** the experiment starts, with
`n_seed >= 5` the constraint that may not be reduced.

### A-38 · Geometric gene locality did not buy what was claimed · AMB · FIXED
v2 claimed `neighbour(g)` "makes a point mutation a small change rather than an arbitrary
one". `neighbour(g)` shares only the RBF **centre**; the *content* `(a_g, b_g, kappa_g)` of
the neighbouring gene is an independent random draw, so the swap replaces one rank-one part
with an unrelated one — a maximally arbitrary functional change.
**Resolution.** The claim is narrowed to what is true and testable: the mutated unit's
**expression profile** moves little, so its routing and metabolic cost are approximately
preserved while its function changes, keeping the clade compact under `L_wire` (§02.2.2).

### A-39 · At `P_es = 2`, rank normalisation discarded the magnitude mirroring provides · AMB · FIXED
With two members `RANK_NORMALISE` returns exactly `{-0.5, +0.5}`, so the update reduces to a
fixed-magnitude step carrying one sign bit — precisely what v2 criticised v1 for.
**Resolution.** At `P_es = 2` the update uses the centred fitness difference
`clip((f_+ - f_-)/(2 sigma_f), -1, +1)` with `sigma_f` a running fitness standard deviation
(§02.6.1).

### A-40 · The `p_explore` degeneracy rule was inert in both directions · AMB · FIXED
v2's rule triggered on "the fraction of vocabulary items with zero inclusions exceeds 0.5",
which at `V = 8192` is dominated by items that never **occur** and therefore sits near its
ceiling permanently rather than responding to a selection pathology; and the rule could only
ever raise `p_explore`, with no cadence and no decay.
**Resolution.** A two-sided rule on occurrence-conditioned starvation, evaluated every
`10 * T_log` steps (§06.2).

### A-41 · `phi_ann` could exceed the selection interval · AMB · FIXED
Nothing in v2 forbade `phi_ann >= T_0`, which is the configuration (v1's `phi_ann = 5000`
against `T_sel = 2000`) that §03.4's dying-unit exclusion set was added to contain.
**Resolution.** `phi_ann < T_0` is asserted at config validation (§03.3). At M-nano
`50 < 99`; at M1 `244 < 492`.

### A-42 · `p_u` in the AdamW group was justified by a wrong dimension · COS · FIXED
§05.7 justified the (correct) placement by "`m_dim = 2`", contradicting §01.2's `m_dim = 8`.
**Resolution.** The reason is restated correctly: Newton–Schulz would drive all 8 singular
values of the `U_max x 8` position matrix toward 1, destroying the relative geometry on `S^7`
that `phi_g` reads and `L_disp` shapes, and Muon's `0.2 sqrt(max(rows,cols))` scaling gives a
`6.4x` multiplier on a sphere-constrained parameter (§05.7).

### A-43 · §09.3's germline arithmetic used `m_dim = 2` · COS · FIXED
v2 computed `|Gamma| = 2048*3 + 256*8 + ~20 = 8.2e3` against its own `m_dim = 8`, and quoted
it for M1 alone while M-nano then had `N_G = 512`.
**Resolution.** `2048*9 + 256*8 + ~20 = 20,500` reals, **identical at both configurations**
by A-16 (§09.3).

### A-44 · §05.10's memory table deferred to itself · COS · FIXED
v2 wrote that SOMA's additional residents "are unchanged from v2 and are listed above"; they
were not listed above, and `theta_lag`'s residency had changed (A-26).
**Resolution.** §05.10 enumerates all eight additional residents with sizes and locations.

### A-45 · B1 was called "matched" while being 5.7% smaller · AMB · FIXED
SOMA carries `W_ctx`, `kappa`, `W_key`, `W_pos`, `K_unit1`, `K_unit2` and `p_u` that B1 lacks
(`+1.31 M` on `24.41 M` at M1), and lacks nothing B1 has.
**Resolution.** The mismatch is **reported with every B1 comparison** rather than removed —
matching parameters would require widening B1's FFN, which changes the `N_G` that §08.6.1's
K9 sweep varies. Any SOMA advantage smaller than the expected advantage of a 5.7%-larger dense
model is not reported as an advantage (§08.4).

### A-46 · `g_growth`'s default was the endpoint of its own range · BLK · FIXED
*(Found by the implementation, not by an auditor.)* §02.2.3 gives `g_growth` the link
`affine-sigmoid` over `[2, 4]` and the default `4`. An affine-sigmoid maps onto the **open**
interval, so `4` has no preimage: `Policy.from_constants` cannot invert the table's own
default, and **generation 0 cannot be constructed**. Every other bounded constant's default is
interior; this was the only one at an endpoint.
**Resolution.** Range widened to `[2, 5]`, default unchanged at `4`. §02.2.3 gains a normative
rule that every default must be strictly interior to its range, asserted at config validation.
Separately, `lambda_disp` — introduced by A-9 and used by (1.9) — was missing from the policy
table entirely, so the dispersion coefficient was outside the ES search space; it is added.

**Why this entry matters beyond its size.** It is the first defect found by *running* the
specification rather than reading it, at the first point where code touched §02.2.3. Part D's
claim that "code is the next auditor" is not a rhetorical flourish; the remaining `[OPEN]`
items are deferred to measurement on exactly this reasoning.

### A-47 · §05.9 downcast the mixer solve, making §05.2.6's own test unsatisfiable · BLK · FIXED
*(Found by the implementation.)* §05.9 required the UT triangular solves to be computed in
`float32` **"regardless of ambient dtype"**, while §05.2.6 requires `mixer_chunk` to match the
`mixer_ref` oracle to **`1e-10` in `float64`**. Read literally the first forces the second to
fail: a `float64` call is downcast to `float32` inside the solve and the realised error is
`1.9e-7`, the `float32` epsilon. Gate `G-MIXER` could never pass, and K6 — "halt all downstream
work" — would have fired on a correct implementation.
**Resolution.** The requirement is a **floor**, not an exact dtype:
`promote_types(ambient, float32)`. `bfloat16` and `float16` are promoted to `float32` as
intended; `float64` is preserved. §05.9 restated.

### A-48 · `D_unif(8)` was tabulated 1.0% wrong · AMB · FIXED
*(Found by the implementation.)* §01.6 tabulated `D_unif(8) = 1.375` as "measured", making
the `G-NOCOLLAPSE` threshold `0.825`. The exact mean chord length on `S^{m-1}` is
`2^{m-1} Gamma(m/2)^2 / (sqrt(pi) Gamma(m-1/2))`, which at `m = 8` is **`1.389328`**; a
Monte-Carlo estimate over 8192 points agrees to four decimals. The tabled value was wrong by
1.0% and the threshold correspondingly loose.
**Resolution.** (1.13a) states the closed form; the threshold is **computed** from it rather
than read from a literal, so it cannot drift again. New threshold `0.83360`. The magnitude is
small, but a "measured" constant that was not measured is exactly the class of defect the
register exists to catch.

### A-49 · §05.10's parameter table omitted the mixer output norm · AMB · FIXED
*(Found by the implementation.)* §05.2.4 specifies `y = OutProj( RMSNorm(o) * SiLU(x W_gate) )`,
so every mixer block carries a `d_model * e_v` norm gain. §05.10's table counted it in neither
the "Mixer blocks" row (which matches `1,320,960 * 9` exactly, i.e. the block *without* it) nor
the "Norm gains" row (`25,088` = four Peri-LN gains per block plus the output norm), so the
stated total was low by `4,608` at M1 and `768` at M-nano.
**Resolution.** Totals corrected to **24,411,160** (M1) and **2,935,752** (M-nano), with the
Muon/AdamW split re-derived as `19,947,520 / 4,463,640` and `2,108,416 / 827,336`. The table is
now produced by `SomaModel.parameter_report()`, which enumerates `named_parameters()`, so it
cannot disagree with the model again. Note the Muon figure was already exact — the omission
fell entirely in the AdamW group, which is where norm gains live (§05.7).

### A-50 · Key-row inheritance at birth is incompatible with Theorem 3.1a · BLK · FIXED
*(Found by the implementation.)* §03.2's `BIRTH` stated "the child inherits its parent's
key-table rows as its initial addressing". `K_unit1` has shape `side x (d_key/2)` and row `i1`
is **shared by all `side` slots in grid row `i1`** (§05.4), so writing the parent's row into
the child's row changes `s1` for every slot in that row — hence `I1`, hence every other unit's
`r_ut`. Theorem 3.1a would be false **at the moment of birth**, which is exactly the failure
§05.4's own v3 correction removed from a different route into it.

Worse, the clause's stated purpose was unattainable anyway: grid adjacency confers no routing
similarity, because adjacent rows carry independent random keys. v2's justification for
`FREE_SLOT_NEAREST` ("a child could land anywhere and receive a completely different token
distribution") therefore described a property the mechanism does not deliver.
**Resolution.** The clause is deleted. §03.2 now states what actually makes a child resemble
its parent — `p_child ~ p_parent` and `g_child ~ g_parent`, which determine the *function* —
and records that routing is **deliberately** fresh: a unit with its parent's function but a
different token distribution is what §02.4 means by a differentiated sibling, whereas
inheriting both would produce a duplicate that selection cannot choose between.
`FREE_SLOT_NEAREST` is retained for the modest benefit it does confer (clade contiguity in
slot space, which makes `Q_route`'s communities spatially interpretable) and that benefit alone
is claimed.

A residual discontinuity is documented rather than papered over: at the first anneal step the
child becomes active, and if its row or column had no other active slot the row flag flips and
`I1` changes for every token. The mask anneal does not smooth this. It is inside the quantity
the bifurcation alarm (1.21) measures — forcing the cohort to `m_u = 0` also clears the flag —
so §03.7's ">1% of birth events" falsifier covers it.

### A-51 · §06.4's qualifying-row count was wrong (3, not 7) · COS · FIXED
*(Found by the implementation.)* v3.0 stated that with `J_ring = 8` and the `n >= 6` exclusion
"at most 3 rows qualify", giving a novelty-fraction granularity of `1/3`. The realised count is
**7**: row `j` qualifies when at least 6 of the 32 evenly-captured batches satisfy `T_k > t_j`,
i.e. `t_j < N_step (1 - 6/32)`, which every `j >= 1` of `t_j = N_step 2^-j` satisfies. The
granularity is `1/7 = 0.1429`.
**Resolution.** §06.4 states the count as a computed quantity with its derivation, and
`CertificationVerdict.granularity` reports the realised value so the K5 margin is checkable
mechanically. The margin `0.34` is unchanged and remains valid — it exceeds `1/7` by 2.4x — so
the error was conservative in effect. It is recorded anyway: a number the code contradicts is a
defect whichever way it errs, and this one sat inside the justification for a kill criterion's
threshold.

### A-52 · `Q_route` optimised a different objective from the one it reports · BLK · FIXED
*(Found by the implementation.)* §08.1.1 specified **Barber bipartite modularity** `Q_B` as the
reported statistic and **Leiden** as the community algorithm. Leiden optimises Newman
modularity (or CPM); Barber's null model is different, and the difference is not cosmetic —
Newman's null admits red-red and blue-blue edges that cannot exist in a bipartite graph, so
partitions good under one are systematically not the argmax of the other. Finding communities
with Leiden and scoring them with `Q_B` therefore reports a value biased **below** the maximum,
on the sole quantity in K4 and one of three in K3, against a margin of `0.02`.
**Resolution.** **BRIM** (Barber 2007), which maximises `Q_B` directly: with
`B~ = A - k d^T / m` and `Q_B = (1/m) tr(R^T B~ T)`, each side's optimum given the other is an
exact `argmax`, so the alternation is monotone. Ten restarts at the fixed seed are retained and
extended over a fixed community grid, since BRIM takes `c` as input. As a secondary benefit
this removes the `igraph`/`leidenalg` dependency, which does not build cleanly on this platform
(risk R8).

### A-53 · The reservoir's context store was 12x its budget, and mistimed `w_t` · BLK · FIXED
*(Found by the implementation.)* v3.0's A-26 repair required the case reservoir to store each
retained case's context window so `L_lag` could be re-run at the selection event. Two problems.

*Size.* `U` is **per-Site** (§01.1 / register entry B5), so the store is `L` times the quoted
figure: `12 * 6554 * 513 * 4 B = 161 MB` at M1 against the `13.4 MB` §05.10 budgeted. The
budget had been written for a single Site's worth.

*Timing.* (4.2) is a product `nu_ut = r_ut * ||delta_ut||_2 * w_t`. The first two are
necessarily measured when the token was routed; re-evaluating `w_t` at the selection event —
up to `T_sel` steps later, against a `theta_lag` that has moved — forms a product of quantities
from different moments, and the resulting credit is not the credit of any actual event.
**Resolution.** `w_t` is evaluated **at push time**, on the `d_rate` fraction of the batch
being offered to the reservoir, and the scalar is stored. The `theta_lag` forward cost is
unchanged at 10% of the batch (`+3.3%` of step time, the figure A-26 established); the context
store disappears; and all three factors of (4.2) are measured at the same moment on the same
token. §04.6, §06.1.2 and §05.10 restated; realised footprint **5.1 MB** across all Sites.

### A-54 · Normalisations downcast `float64`; seeded units were ineligible for one step · COS · FIXED
*(Both found by the implementation.)* Two small defects of the same shape as larger ones
already recorded.

*(a)* `RMSNorm`, the gain-free QK norm and the masked form (3.4) all accumulated in `float32`
unconditionally, so a `float64` call was capped at the `float32` epsilon (`3.8e-7` measured).
This is A-47's defect in a second place. §05.9 now states the floor rule once and applies it to
both the normalisations and the triangular solves: `promote_types(ambient, float32)`.

*(b)* §02.2.4 seeded `age_u <- M_mat` with the comment "seeded units are immediately
eligible", while §04.2 defines `U_death` with `age_u > M_mat` — so a seeded unit was ineligible
for exactly one step and the comment was false. Seeding at `M_mat + 1` makes it true. The
practical effect is nil (the first selection event fires at `n >= max(T_0, M_mat)`, by which
time ages far exceed `M_mat`), but a comment contradicted by the code two files away is how
the larger defects in this register began.

### A-55 · A cuSOLVER SVD corrupted later kernels asynchronously · BLK · FIXED
*(Found by running the code; the single hardest defect to localise so far.)* §03.4's `erank`
diagnostic took `torch.linalg.svdvals` of a `2048 x 2048` matrix on the GPU. cuSOLVER's batched
Jacobi driver fails to converge on it, emits a warning, and falls back — but leaves an
**asynchronous** error state that corrupts *later, unrelated* kernels. The symptom was `NaN`
inside the mixer's triangular solve on a **subsequent** forward pass, with **every parameter
and buffer still finite**, at a step that moved between runs with identical seeds, and which
**vanished entirely** under `CUDA_LAUNCH_BLOCKING=1` or when extra synchronising
`torch.isfinite` checks were inserted.

Two hypotheses were tested and **falsified** before the cause was found, and both are recorded
because each would have been a plausible place to stop:
* *UT ill-conditioning.* Measured `cond(I + strictlower(diag(beta) K K^T))` as within-chunk key
  correlation rises to 1.0 at `beta = 2 - 1e-3`: it reaches only `1.6e3`, and the chunkwise
  form still matches the oracle to `1.1e-12`. The conditioning bound of §05.2.3 holds even in
  the degenerate case.
* *`solve_triangular` itself.* 5000 trials at the exact mixer shapes, interleaved with other
  GPU work: **zero** non-finite outputs from finite inputs.

**Resolution.** The SVD runs on CPU (§03.4). `erank` is a diagnostic at `10 * T_log` cadence,
so the transfer is affordable; the corruption was not. **The general lesson is recorded in
Part D:** a `NaN` whose location moves between identical-seed runs, and which disappears when
you add a check for it, is an asynchronous fault — not arithmetic — and no amount of reasoning
about the mathematics will find it.

### A-56 · §07.4.3 criterion 1 named no mechanism · AMB · FIXED
*(Found while diagnosing A-55.)* M1 criterion 1 requires "no NaN/Inf in loss **or gradients**
for the whole lifetime" and v3.0 specified nothing that checks it. The loss was tested; the
gradients were not, so a non-finite gradient first surfaced as a `NaN` several operations
downstream in an unrelated tensor.
**Resolution.** `clip_grad_norm_` already returns the total norm, which is non-finite exactly
when some gradient is — so the check costs a scalar that is already computed. It runs every
step, aborts the run, and **logs the offending parameter names**. It immediately earned its
place: on a deliberately degenerate corpus it fired at step 1007 naming `E_emb.weight`, turning
"a NaN appeared somewhere" into "the tied embedding's gradient overflowed", which is a
diagnosis rather than an observation.

### A-57 · The embedding scale broke gradient flow, and §02.5's test could not detect it · BLK · FIXED
*(Found by running the code: §02.5 aborted the first real-data lifetime.)* Two coupled defects.

*(a) The initialisation.* §05.6 specified a tied embedding with no read scaling. Peri-LN
normalises every sublayer output to RMS 1 before adding it, so the residual stream should start
at RMS 1; with `Var(E) = 1/d_model` it starts at `d_model^{-1/2}` = `0.072` at M-nano, and
`Norm_pre` amplifies it `14x` at block 0. Measured, the residual **gradient** jumps **43x**
(M-nano) / **61x** (M1) at block 0 against `1.1x`–`2.4x` at every other block — the entire
gradient-flow failure, in one place.
**Resolution.** The embedding is read scaled by `sqrt(d_model)` and the head is not, which is
the only way one tied matrix can set both scales: the residual starts at RMS 1 and
`Norm_out(x) E^T` still yields logits of order 1. Total first-to-last gradient ratio falls
`304 -> 13` (M-nano) and `799 -> 24` (M1).

*(b) The test.* §02.5 step 2 compared **parameter**-gradient norms of block 0 against block
`L-1`. At M1 those blocks differ in size by 2x (mixer vs attention), so the ratio conflates
parameter count with gradient flow — and it reported `7.9` where the residual gradient ratio
was `304`, i.e. it **passed** on a stack with a 43x discontinuity in it. Its band `[0.1, 10]`
was also stated on the total across the stack, which is not depth-invariant: the same healthy
model measured `13` at `L = 6` and `24` at `L = 12`, so a single band cannot hold at both.
**Resolution.** The test measures the **residual-stream activation gradient** and states the
band **per block**, on the **median** — robust to block 0's legitimate `sqrt(3)` growth, which
dominates a geometric mean at small `L`. Realised: `1.38` at M-nano, `1.21` at M1, band
`[0.60, 1.70]`.

**Why this one is worth reading twice.** Defect (b) is what let defect (a) exist. A check that
measures the wrong quantity does not merely fail to catch a problem — it certifies its absence,
and every downstream number is then reported against a stack whose first block loses a factor
of 50. It was found only because a *different* configuration made the wrong statistic cross its
own threshold by 40%.

### A-58 · `epsilon_bif` is zero on deterministic hardware · BLK · FIXED
*(Found by running a full lifetime.)* §01.7.2 set `epsilon_bif = 3 * sd_probe`, with
`sd_probe` the spread of `L_task` on a **fixed** probe across `k_probe` repeated passes "under
the run's nondeterminism". On this card, with these kernels, that spread is **exactly zero** —
the passes are bit-identical — so `epsilon_bif = 0` and the alarm fires on any non-zero
difference at all. Measured on a healthy 1400-step lifetime: **1068 births, alarm rate 1.12%**,
above §03.7's `1%` falsifier, on differences of order `1e-9`.

This is A-36's defect returning through the other side. v2's alarm was **unreachable** because
the difference was identically zero; v3.0's is **unavoidable** because the threshold is. Both
make M1 criterion 2 and §03.7's falsifier report something other than what they name.
**Resolution.** A relative floor: `epsilon_bif = max(3 sd_probe, 1e-3 |L_task(probe)|)`. The
alarm becomes a claim about magnitude rather than about the hardware's reproducibility, and it
degrades correctly on hardware that *is* nondeterministic, where `3 sd_probe` dominates.

### A-59 · The §04.4 credit audit was specified and never run · BLK · FIXED
*(Found by reading a completed run's metrics.)* §04.4 requires the Spearman correlation between
(4.2) and the exact leave-one-out to be **logged**, with a doubling rule on `< 0.5` and a
§04.10 falsifier at the same threshold — "the credit proxy is invalid and every selection
decision taken with it is suspect". The `CreditAudit` object existed, was carried in the
lifetime metrics, and was **never called**: a completed run reported `n_pairs: 0` and
`spearman: NaN`, i.e. the single check standing between ε-lexicase and a meaningless credit
signal had never executed.
**Resolution.** The audit runs at every selection event on Site 0. The counterfactual is
computed by **switching the unit off** (`m_u <- 0`) and re-running the forward, which is exact;
§04.4 permits a partial forward from the Site's layer to the head as a cost saving, and a full
forward is strictly more faithful and affordable at event cadence over a handful of units.
Units carrying no probe traffic are **skipped** rather than scored zero, since a block of ties
at zero would inflate the correlation with pairs that measure nothing.

### A-60 · The alarm rate's denominator made §03.7's falsifier unfireable · BLK · FIXED
*(Found by running a full lifetime.)* §03.7's falsifier reads "fires on more than 1% of
**birth** events", and v3.0 implemented the denominator as **units born**. But (1.21) is
evaluated **once per cohort**: one alarm on a 176-unit growth cohort counts as one alarm over
176 births, `0.57%`, which cannot cross a 1% bar however badly the cohort behaves. Measured on
a healthy smoke run: **`0.94%` per unit born against `45%` per cohort** — the reported
statistic passed on a run where nearly half the evaluated cohorts tripped. This is A-22's
defect in a different mechanism: a threshold that arithmetic prevents from being reached.

A second, independent problem surfaced with it. A scheduled `GROW` event adds `3|U_live|` units
at once, and **altering the function is its purpose**; its alarms say nothing about whether the
*birth operator* behaves as a controlled probe, which is what §03.7 asks. Worse, each such
alarm halved `rho`, so the growth schedule was silencing the selection machinery scheduled
alongside it — `rho_scale` reached `1/1024` in a 1400-step run.
**Resolution.** The rate is per **selection cohort**; growth cohorts are counted and reported
separately as `growth_alarm_rate` and do **not** halve `rho`. The v3.0 per-unit statistic is
retained as `bif_alarms_per_birth` so the two are comparable.

### A-61 · The bifurcation threshold was absolute where the quantity is not · BLK · FIXED
*(Found by running a lifetime after A-58.)* A-58 replaced a zero threshold with
`max(3 sd_probe, 1e-3 |L_probe(0)|)`. That is **absolute** and anchored to the step-0 logit
scale, which grows by orders of magnitude during training while the threshold does not — so the
criterion ends up reporting the loss scale rather than the birth operator. Measured on a
healthy run: **67% of selection cohorts fired**. It is also **size-blind**: a selection cohort
of 8 units and a growth cohort of 176 were judged against the same bar.
**Resolution.** (1.22): the alarm is **relative** to the current probe loss and **proportional**
to the cohort's share of the population, firing when
`|L_active - L_masked| / |L_active| > max(noise_floor, 3 c / |U_live|)`. A cohort carrying
`c/|U_live|` of the routed mass is *expected* to move the probe by about that much; a
bifurcation is a **disproportionate** change. Scale-free over the run, size-free over the
schedule.

Three attempts were needed to get this threshold right (A-36, A-58, A-61), each failing for a
different reason: unreachable, then always-firing, then measuring the wrong scale. The common
thread is that **the quantity being thresholded was never dimensionless** until now. Where a
criterion compares two quantities that both drift during a run, the ratio is the statistic —
not either one.

### A-62 · The population cap did not bind: dying units were counted as freed · BLK · FIXED
*(Found by running a lifetime after A-60.)* (1.14) read
`max(0, U_cap - (|U_live| - n_death))`, which assumes the event's deaths free their slots
immediately. They do not: `DEATH` sets `dying_u` and keeps `live_u = True` for a further
`phi_ann` steps while `m_u` anneals to zero (§03.4). Every concurrent anneal therefore bought
the same event an extra birth. Measured on an M-nano lifetime: **`|U_live| = 243` against
`U_cap = 240`**, with no abort and no diagnostic — the cap is the *only* mechanism bounding
the population (§01.6.1 makes `L_size` gradient-free by construction), so nothing else could
have caught it. At M-nano `U_max - U_cap = 16`, so a handful of overlapping selection events
would have exhausted the slot grid and `FREE_SLOT_NEAREST` would have raised mid-lifetime.

**Resolution.** (1.14) carries **two** bounds, because two distinct resources are exhaustible
and a dying unit consumes exactly one of them:

```
  BUDGET_HEADROOM = max( 0 , min( U_cap - n_nondying , U_max - |U_live| ) )
```

`U_cap` bounds the **effective** population — dying units are excluded from routing already,
so they do not count against it. `U_max` bounds the **slots** — dying units occupy one, so
they do. Both are asserted as postconditions of `SELECT` and of `GROW`, not assumed.

This is A-17 approached from the other side. v2 evaluated headroom *before* deaths and births
became permanently zero at the cap; v3.0 evaluated it net of deaths and the cap stopped
binding. The pattern worth carrying: when one expression is asked to bound two resources that
are released on **different schedules**, it will be wrong for at least one of them, and which
one depends only on which way the single expression was last corrected.

### A-63 · A diagnostic ended the run instead of reporting · MAJ · FIXED
*(Found by running a lifetime after A-60.)* The same lifetime ended at step ~500 with
`torch._C._LinAlgError: linalg.svd: the input matrix contained non-finite values`, raised
inside §03.4's `erank` probe. Every parameter and every gradient was finite at the preceding
step, and `grad_norm = 0.992`.

Two defects, one symptom:

1. **The probe aborted.** `erank` is a plasticity diagnostic on a `10 * T_log` cadence. Its
   contract is to *measure*, and an exception carries strictly less information than the
   measurement's own missing-value would: `erank` is logged per Site, so `NaN` names the Site
   and the step and leaves the run alive to record what happens next, whereas the exception
   says only "something, somewhere, was not finite" and destroys the evidence by ending the
   run. It now returns `NaN` per Site on non-finite input.
2. **Nothing checked the parameters.** §07.4.3's two guards read the loss and the gradient
   norm. A parameter can be corrupted with both finite — a device-side kernel writing into a
   buffer after the optimizer step — which is precisely the A-55 failure mode. A sweep over
   `model.parameters()` at `T_log` cadence now closes the gap and aborts naming the offending
   tensors, attributing the failure to a window of `T_log` steps.

The general rule, now stated in §03.4 and §07.4.3: **the components that abort a run are
named in the protocol, and a diagnostic is not one of them.** A monitor that can kill the
process it monitors converts every one of its own edge cases into a run-ending failure, and
the one thing that is certain about a numerical edge case is that it will appear first in the
code that was written to look for it.

### A-64 · Module construction drew from the global RNG, so no run was reproducible · BLK · FIXED
*(Found by comparing two identical invocations.)* `RngBundle` gives every purpose its own
NumPy and torch generator and every draw in SOMA takes one explicitly — with one exception
nobody had looked for. `nn.Linear.reset_parameters`, `nn.Embedding.reset_parameters` and
`nn.init.normal_` accept **no `generator` argument** in torch 2.7, so **model construction
draws from the process-global default generator**, which `RngBundle` by design never touches.

The initial weights were therefore a function of wherever the process's global stream
happened to be. Evidence: two back-to-back invocations of the same lifetime script with the
same `seed=1` reported §02.5 acceptance statistics of `var_ratio = 0.640 / 0.635` and
`grad_ratio = 12.739 / 13.045`. Every reproducibility claim in §07.7, every seed-paired
ablation in §08.6, and every `[OURS]` measurement rested on a model that was not a function
of the seed.

It also made **debugging non-reproducible**, which is how it was found: the same failure
appeared at step 412 in one run and step 1025 in the next.

**Resolution.** `seed_global_construction(run_seed, "init")` seeds the global CPU and CUDA
generators from `derive_seed(run_seed, "init")`, called by `LIFETIME` immediately before the
model is built (§02.6.3). The regression test carries a **positive control** — the same
construction with the seeding removed must differ — because "two runs agree" passes
trivially in a process whose global stream has not moved, and a test that cannot fail is
worse than no test.

The general lesson: **an explicit-generator discipline is only as good as its least
cooperative dependency.** The audit that matters is not "does our code pass a generator"
but "does every draw that happens during our code accept one".

### A-65 · The decay ratio `K->` overflowed in backward while the forward was finite · BLK · FIXED
*(Found by running a lifetime; localised with a global module backward hook.)* §07.4.3's
gradient guard fired with a **finite loss of `4.04`** and a non-finite gradient norm. A
module-level backward hook placed the non-finiteness precisely: at the failing layer, the
`Mixer`'s incoming gradient was **finite**, its outgoing gradient was not, and of its seven
child modules exactly one — `w_alpha` — had a non-finite gradient. `w_alpha` is the sole
source of `alpha_mix`, and therefore of `g_r = prod_{j<=r} alpha^j`.

§05.2.5's arrow rescalings contain `K->[i] = (g_C / g_i) k^i`. `Gam_decay` was already
computed as `exp(log g_r - log g_i)` with a comment saying why; `K->` was written as the
quotient `g_last / g`. The two look equally safe in the forward pass, and are not:

```
  d(g_C/g_i)/dg_i = -g_C / g_i^2
```

`g_i^2` underflows at the **square root** of `g_i`'s own limit. In `float32` the gradient is
already `NaN` at `sum_j log alpha^j = -96` (`g_i ~ 2e-42`), where the forward value is still
exactly representable. Swept and confirmed: at `log alpha = -1.5` sustained over a chunk of
`C = 64` the quotient form's forward is finite and its gradient is `NaN`, while the
log-space form is finite in both; at `log alpha = -2.0` even the forward is `NaN`.
`alpha_mix ~ 0.22` is an unremarkable value for a decay gate.

**Resolution.** `K->` is computed as `exp((log g_C - log g_i).clamp(max=0))`, the same rule
`Gam_decay` already obeyed. `Q<-` and `W<-` are left alone: they scale **by** `g_r <= 1`, so
underflow to zero is the correct answer and no ratio is formed.

Two things made this survive every test the mixer had. First, the 41 equivalence tests all
draw `alpha ~ sigmoid(N(0,1))`, whose chunk-cumulative log is `~ -40` at `C = 64` — a regime
in which the two forms agree to machine precision. Second, and more importantly, **every one
of those tests checks values, not gradients.** An equivalence oracle that compares forward
outputs cannot see a defect whose entire signature is in the backward pass. The new
regression checks the gradient and carries a positive control that the quotient form fails
on the identical inputs.

The rule now stated in §05.2.5: **every decay ratio is formed in log space**, and the reason
is recorded as being about the derivative, not the value — because the value is exactly what
made it look safe.

### A-66 · The M1 driver's `solve_rate_fn` was dead, leaving the open-ended arm inert · BLK · FIXED
*(Found by reading the driver while writing its integration test.)* `scripts/train_tinystories.py`
defined `solve_rate_fn(params, n_eval)` closing over a `model_holder` dict that nothing ever
filled, and then called `lifetime(..., solve_rate_fn=None)`. Two independent defects in four
lines: the arity did not match the `(model, genome, params, n_eval)` contract, and the callable
was not passed at all.

The consequence is silent. With no solve rate, `SEED_ARCHIVE` admits nothing, `archive` stays
`None`, `f_oe` mixes no synthetic rows, and **the entire open-ended arm — constitutional
requirement C2 — does not run**. Nothing raises, because an empty archive is a legal state
and `archive_size: 0` is a legal number.

**Resolution.** The callable takes the contract's four arguments and is passed. The
end-to-end integration test asserts `archive_size > 0`, `n_evaluations >= 1` on the credit
audit, and a certification with all three arms — because the class of defect here is
"a mechanism was never reached", which no unit test and no smoke-run summary can see.

### A-67 · The credit audit compared the wrong quantity on the wrong token set · BLK · FIXED
*(Found by investigating a failing falsifier.)* An M-nano lifetime reported
`credit_audit: {spearman: 0.155, valid: false}` — §04.10 firing. Before accepting a falsifier
one has to check that it is testing what it claims to, and it was not:

* **`w_t` was missing.** (4.2) is `r_ut * ||delta_ut|| * w_t`; the audit scored
  `sum_t r_ut ||delta_ut||`. §04.6's own note says all three factors must come from the same
  moment on the same token, and the audited proxy dropped one of them.
* **The pairs were per unit, not per `(u, t)`.** `nu_ut` is indexed by unit *and* case, and
  §04.4 says "a random `1%` of **cases**". The audit summed the proxy over all probe tokens
  and compared it to the change in the probe **mean**, so one pair mixed every case together
  and a per-case effect was diluted by `1/(B T_ctx)`.

**Resolution.** Pairs are `(u, t)`; the proxy is (4.2) entire, with `w_t` computed from
`theta_lag` on the same probe; the exact term is the change in that case's own loss. The
counterfactual costs one forward per **distinct unit** in the sample rather than one per pair.
The audit is skipped — not scored zero — before `theta_lag` has moved, since `w_t == 0` there
makes every proxy tie and a correlation over a constant is undefined, not small.

### A-68 · The audit's `0.5` bar was unsatisfiable: a non-negative proxy against a signed target · BLK · FIXED
*(Found by re-measuring after A-67.)* With the comparison corrected, the Spearman **fell** to
`+0.0008` over 472 pairs. That is the interesting number, and it is not a failure of (4.2).

(4.2) is a product of three **non-negative** factors, so `nu_ut >= 0` identically. The target
`L_t(m_u=0) - L_t` was **negative on 52%** of the sample — removing a unit lowered that case's
loss about half the time. A rank correlation between a non-negative statistic and a target
that changes sign on half its sample is bounded near zero *whatever the proxy is*: the test
was unsatisfiable, not failed. It is the same defect class as A-36, A-58 and A-61 — a
threshold that cannot report what it claims to — in the opposite direction.

Both correlations were then measured on the same 472 pairs:

| statistic | value |
|---|---|
| `rho_signed` — (4.2) vs `L_t(m_u=0) - L_t` | `+0.001` |
| `rho_magnitude` — (4.2) vs `\|L_t(m_u=0) - L_t\|` | `+0.601` (`+0.657` over non-zero proxies) |

**Resolution.** §04.4's gated statistic is the **magnitude** correlation; the signed
correlation is logged beside it and never gated. The finding this produces is a statement
about the architecture, not a repair: **(4.2) estimates how much a unit changes a case and
carries no information about whether the change helps.** §03.4.1's `y_u` is likewise a
magnitude of change. SOMA selects on **influence**; benefit is the gradient's job — the same
division of labour §01.7.2 draws between the lifetime and generation rates.

**[OPEN — recorded against myself]** `rho_magnitude = 0.601` clears the `0.5` bar by `0.10`,
on **one** M-nano measurement at `n <= 700`, and the criterion was re-specified **after** that
number was seen. That is precisely the situation pre-registration exists to prevent. The
argument for the change is structural rather than empirical — a non-negative statistic cannot
rank a sign-changing target, and that is true before any measurement — but the reader is owed
the sequence in which it actually happened.

**[OPEN — §04.10 is currently firing]** On the **full** M-nano lifetime that followed
(1400 steps, 3 audit evaluations, `>= 64` pairs each) the value is:

```
  spearman        = 0.4572     (magnitude; the gated statistic)
  spearman_signed = 0.1397
  tripped         = 1          audit_fraction doubled 0.01 -> 0.02
  valid           = false
```

`0.457 < 0.50`, so **§04.10's falsifier fires**, and it is left firing. The bar is not moved
to accommodate it. Three things are true at once and all three are reported:

1. the value is **marginal** — 0.043 below a bar chosen for roundness, not derived;
2. it is **lower than the 472-pair probe's 0.601** measured over steps 100–611, so the
   correlation appears to *fall* as the population saturates the cap, which is the direction
   substitutability predicts: with `|U_live| = 240` and `k_route = 4`, removing one unit is
   increasingly "replace it with the next-best unit" rather than "remove its contribution";
3. it is **one lifetime at one configuration**. §04.10 is adjudicated at M1.

If it also fires at M1, the honest conclusion is not that SOMA is broken but that (4.2) is a
*local* credit signal being audited by a *global* ablation, and the audit — not the proxy —
needs a counterfactual that holds routing fixed. That change would itself have to be
pre-registered before the M1 run, not chosen after seeing its number, and it is **not** made
here.

### A-69 · The test suite's own model constructions were unseeded · MAJ · FIXED
*(Found by running the gate ledger.)* A-64 fixed `LIFETIME`. It did not fix the **tests**:
21 `SomaModel(cfg)` constructions across the test modules, none of which seeded the
process-global stream, so every one of them was built from whatever the previous test left
behind.

The evidence is a single failure of `G-GRAD` during the gate sweep that did **not** reproduce
in fourteen subsequent runs. The tests are tolerance-based assertions on finite differences
and on the §02.5 acceptance bands; they had been passing on every initialisation they happened
to be handed, and failed on one they had not.

A suite whose passes are not reproducible is not evidence, whatever its green count says — and
the defect it is least able to find is precisely this one: an assertion with a tolerance that
holds for most initialisations.

**Resolution.** An autouse fixture in `tests/conftest.py` resets `torch`, `torch.cuda`, `numpy`
and `random` before **every** test. Seeding there rather than at 21 call sites is deliberate:
a convention that must be remembered at every construction will be forgotten at one of them,
and that failure is silent — a passing test, on a model nobody chose. Verified by six
consecutive full-suite runs at 365 passed, 0 failed.

The pattern, now twice: **an explicit-generator discipline is only as strong as the places
that are exempt from it.** A-64 was the production path; A-69 was the path that verifies it.

### A-70 · M0 named `theta_m0` as an artifact and produced no such file · BLK · FIXED
*(Found by reading the M0 driver before running it.)* `run_m0.py` wrote
`"theta_m0_checkpoint": f"{run_id}/checkpoints/final.pt"` into its report, and nothing
anywhere saved a checkpoint. `lifetime` did not even return the model.

§07.4.2 lists `theta_m0` as one of the three artifacts M0 exists to produce, and §06.2's
Rho-1 token selection needs it as the *frozen, better* reference — the one that must know
**more** than the current model, as distinct from §04.3's lagged self which must know less. So
the M1 run would have started with a reference path in its manifest pointing at a file that
had never existed. A dangling artifact is worse than a missing one: the report asserts it.

A second defect in the same function: `--target-ce` defaulted to `1.45` at **both**
configurations. That number is §07.4.2's, and §07.4.2 states it for TinyStories V2 at
`V = 8192`, `T_ctx = 512`, `D_tok = 8e8` against published checkpoints. M-nano runs at
`V = 4096`, `T_ctx = 128`. A cross-entropy over a quarter of the vocabulary at a quarter of
the context is a different quantity, and the M-nano M0 would have been failed — or worse,
passed — against a bar imported by transcription.

**Resolution.** `LifetimeConfig.save_final_state` writes `checkpoints/final.pt` with the
state dict, the config and the seed; `run_m0.py` sets it and records the resolved absolute
path. The target applies at M1 only; at M-nano the M0 is an artifact-producing run whose pass
condition is **stability**, and §07.4.2 now says so. An explicit `--target-ce` at M-nano is
honoured and announces that it is outside the specified setting.

### A-71 · The checkpoint writer and the checkpoint reader disagreed on a key · BLK · FIXED
*(Found immediately after A-70, by reading the consumer of the artifact just written.)* A-70's
fix wrote `{"state_dict": ...}`. `scripts/eval_tinystories.py` reads `blob["model"]`.

The failure mode is what makes this worth a register entry rather than a silent correction:
it is a `KeyError` at the **end** of an M1 run, in the evaluation step, hours after the run
that produced the file — and it would have been reached only by executing the exact sequence
`run_m0 -> train_tinystories -> eval_tinystories`, which no test did, because each script's
own tests exercise its own side.

**Resolution.** The writer uses `"model"`, and `tests/integration/test_checkpoint_contract.py`
pins the contract end to end: a real `lifetime` writes the file, and the assertions are the
reader's own expectations — the key, the genome beside it (the germline is **not** in the
state dict by construction, §02.1), the 66 population buffers that make a reload resume the
same population rather than a freshly seeded one, and a forward pass on the reloaded model
that reproduces the original logits bit for bit.

Fixing A-70 introduced A-71 in the same edit. That is the third time in this register a repair
has introduced its own defect (A-17→A-62, A-58→A-61, A-70→A-71), and the pattern is now
explicit enough to state as a rule: **after writing an artifact, read its consumer before
claiming the artifact exists.** A produced file is not a delivered interface.

### A-72 · §06.2's reference had no producer, so token selection was silently a different mechanism · BLK · FIXED
*(Found by reading the consumer chain before the first gated run — the discipline A-71 taught.)*
`ReferenceCache` was implemented, tested, and wired into `soma/train.py`. `LifetimeConfig`
carried a `ref_cache` field. **Nothing ever built a cache, and no driver ever set the field.**

`select_tokens` handles `ref_per_token=None` gracefully and documents the fallback:
`delta = L`, raw loss. That fallback is correct for M0's own run and for a smoke test. It is
**not** §06.2. Rho-1 token selection is defined on the *excess* loss over a frozen, **better**
reference — the whole point of the mechanism, and the reason §06.2's reference is deliberately
different from §04.3's lagged self (which must know **less**; audit A-4). Selecting on raw
loss instead selects whatever the model currently finds hard, including the tokens that are
hard because they are unpredictable, which is the failure Rho-1 exists to avoid.

So an M1 run would have reported "token selection: on" while running a different criterion,
with no error and no log line — the same shape as A-66, and the second time a constitutional
mechanism was found inert rather than wrong.

**Resolution.** `scripts/build_ref_cache.py` computes `L_ref` over the corpus from `theta_m0`
and writes the `float16` sidecar §06.1.1 specifies, binding it by tokenizer hash, corpus hash,
`T_ctx` and a **content hash of the checkpoint file** — a run id would let two M0 runs with
the same id and different weights share a cache. It reproduces the training stream's own
`targets`/`scored` construction line for line, because a cache aligned to different tokens
than the lookup expects is undetectable downstream, and it reports coverage against the
expected shortfall of one token per document.

`train_tinystories.py` now **refuses to run** without `--ref-cache` unless
`--allow-no-ref-cache` is passed, which records the fallback in the report. Measured on the
M-nano corpus: 195k tok/s, 472M tokens, mean `L_ref = 2.11` nats against M0's held-out
`2.02` — the reference is the better model, as §06.2 requires.

The rule from A-71 generalises again: **an artifact that is consumed but never produced is
invisible from both ends.** The consumer has a documented fallback; the producer does not
exist; each side reads as complete on its own.

## Part A'' — the adversarial spec-conformance audit (Agent A)

Run against the **implementation**, not the specification, after A-72. The auditor read
`AUDIT-v1.md` in full and then walked §01–§08 and §10 line by line against `soma/**`. Every
finding below was verified independently before it was fixed.

### A-73 · The bifurcation alarm measured mean-squared logits, not `L_task` · BLK · FIXED
(1.21) names `L_task(probe)`. All three places that needed it —
`_evaluate_bifurcation`'s two terms, `measure_sd_probe`'s `probe_loss`, and `epsilon_bif`'s
`probe_scale` — computed `logits.square().mean()` instead.

That is a scalar **with no dependence on the targets**. It responds to the scale of the output
distribution, not to the model's loss, and the two are not a monotone reparameterisation of
each other: masking cohorts of 4, 8 and 16 units moved the surrogate by `0.73`, `0.22` and
`0.03` times the corresponding relative change in `L_task`, so the alarm's *ordering* over
cohorts was not (1.21)'s ordering. `bif` drives `rho_scale *= 0.5` for the rest of a lifetime,
so this changed the realised structural churn of every run, and it is the statistic behind
§03.7's falsifier and §07.4.3's criterion 2.

The corroborating detail: `_probe_loss`, which computes exactly the probe cross-entropy (1.21)
asks for, **already existed in the same module and was dead code**.

**Resolution.** All three call sites use `_probe_loss`. This is the fourth defect in this one
criterion (A-36, A-58, A-61, A-73) and the first three were all about the **threshold**. This
one was about the **statistic**, which no amount of threshold work could have reached — the
lesson being that a criterion has two halves and fixing one repeatedly is not evidence about
the other.

### A-74 · §05.4's per-token widening loop was replaced by one population-level rank · BLK · FIXED
§05.4 specifies a widening loop **per token**: start at `topk_{k_route}` on both axes, widen
by one rank until `|cand| >= k_route` or the grid is exhausted, then take (5.18) within that
block. v3.0 chose a single `w` for the whole batch from an expected-occupancy argument,
`w >= sqrt(k_route * U_max / n_active)`, with a comment claiming it *"reaches the same fixed
point the loop converges to, in one shot"*.

It does not, in either direction. Measured at M-nano against a literal transcription of the
spec's loop, over 256 tokens:

| `n_active` | `w` used | tokens routed differently | tokens left short of `k_route` |
|---|---|---|---|
| 240 (at cap) | 4 | 0 / 256 | 0 |
| 128 | 4 | 3 / 256 | 3 |
| 96 | 5 | **113 / 256** | 0 |
| 64 (`\|U\|_0`) | 5 | **85 / 256** | **25** |
| 32 | 7 | **123 / 256** | 5 |

At `n_active = 64` — the population **every M-nano lifetime is seeded with**, and the state it
occupies until it grows — a third of tokens were routed to a different unit set than §05.4
specifies, and a tenth to *fewer* than `k_route` units where the loop would have widened and
found a full set. `r_ut` is the input to (1.8), (3.5), (4.2) and `Q_route`, so the deviation
propagated into the metabolic gradient, the utility, the credit matrix and K4's statistic.

**Resolution.** The loop is implemented exactly, and without the per-iteration `.all()`
synchronisation that motivated the shortcut: both axes are ranked in full once, `A[w]` (the
number of active pairs in the leading `w x w` block) is read off the diagonal of a 2-D prefix
sum, and `w_t` is the first index at which it reaches `k_route`. Selection runs detached — only
the `k_route` gathered scores carry gradient — so the `(B, T, side, side)` intermediates never
enter the autograd graph. Verified against the literal transcription at six occupancies:
**0/96 mismatched tokens at every one**, where the shortcut differed at four of six.

The occupancy sweep is the test, and the reason the defect survived: **at the cap the shortcut
and the loop agree**, so any test written at a full population would have passed on the broken
code.

### A-75 · §04.5.1's cost governor did not exist, and `sep(n)` was logged but never asserted · MAJ · FIXED
Two unimplemented `[NORMATIVE]` statements, both of the "specified, plausible, absent" kind.

**§04.5.1's governor.** *"Measured selection wall-clock is a required log field. If it exceeds
2% of step time averaged over a 1000-step window, then `d_rate <- max(0.02, 0.7 * d_rate)`."*
`d_rate` was **read in eight places and written in none**; nothing anywhere timed `select()`;
`SelectionEvent.as_dict()` carried no wall-clock field; §07.6's required
`selection wall-clock share` log field did not exist; and §07.8's profiling row citing the 2%
cap had nothing to read. §01.7.2 names this as one of exactly **two** in-lifetime policy
writes, and the other one (`rho`) was implemented.

Note that `lexicase.py` carries `SPEC: 04 §4.5.1`, so `scripts/spec_trace.py` reported the
section as **mapped**. A marker claims ownership; it does not claim implementation. That is
worth stating because G-SPECMAP is otherwise easy to over-read.

**§01.7.1's assertion.** *"`sep(n) <= 1` is asserted and logged for `n_warm <= n < n_dec`
only."* It was logged every `T_log` steps, under a docstring that restated the requirement to
assert it.

**Resolution.** `SelectionGovernor` times selection against step time over a 1000-step window
closed on **step count** — so two runs with the same seed reduce `d_rate` at the same steps
even on differently loaded hardware — writes to the step-0 deep copy of `Policy` (§01.7.2's
Lamarckian guard), and reports `selection_share`, `d_rate` and `d_rate_reductions` every
`T_log`. `LIFETIME` asserts `sep(n) <= 1` over the specified window.

**[OPEN — and it is the honest half of this entry]** The `sep` assertion **cannot fire under
the current schedules**. On `[n_warm, n_dec)` the WSD schedule holds `lr = lr_max`, so (1.20)
reduces to `sep(n) = exp(-n/n_c) / (1 + n/n_c)` by (3.10) and (3.11) — a product of two factors
each `<= 1`, with equality only at `n = 0`. It is therefore a **guard against a future change
to (3.10) or (3.11)**, not a live check, and its test says so and demonstrates it by
substituting an inverted schedule that does violate the bound. Calling it an assertion without
that note would be the same over-reading as calling a `SPEC:` marker an implementation.

### A-76 · `r_ut` carried an extra `m_u` factor into (4.2) · MAJ · FIXED
`SiteOutput.r_ut` was returned as `r * m_u`. §05.4 defines `r_ut` as the softmax alone. Of its
three consumers, exactly one wants the product:

* (1.8) `L_wire` — wants `r_ut * m_u`, and had it;
* (3.5) `h_u` — **unaffected either way**, because it is a *ratio* whose numerator and
  denominator carry the same per-unit factor, which cancels identically;
* (4.2) `nu_ut` and §08.1.1's `Q_route` — want `r_ut`, and got `r_ut * m_u`.

So a newborn's ε-lexicase credit was scaled by `m_u in (0,1)` for the whole of its
`phi_ann`-step anneal — up to 50 of the 99 steps of an M-nano selection interval — systematically
under-crediting exactly the units selection is deciding whether to keep. Same class as A-67's
missing `w_t`, in the opposite direction.

**Resolution.** `SiteOutput` carries `r_ut` unmasked and `mask_u` beside it; (1.8) forms the
product itself. Two of the three consumers being correct is why it survived: a defect that is
right in the place people look at (the metabolic term, which has its own gate) and wrong in
the place they do not.

### A-77 · Two withdrawn numbers survived in prose that nothing compared · MIN · FIXED
Two restatements of corrected values, both left behind by the repair that corrected them.

**§07.4.1's parameter row** still read `2,934,984 / 24,406,552` — the counts A-49 withdrew
when it found §05.10's table had omitted the mixer output norm. The same section says: *"These
values are the ones in §01.9 and nowhere else; this table restates them and may not diverge.
Gate `G-SPECNUM` asserts the agreement mechanically."* `G-SPECNUM` asserted the counts against
the **code** and never against the other table, so the sentence claiming a mechanism described
one that did not exist. A restatement nothing compares is a second source of truth.

**`certify.py`'s two granularity docstrings** still said *"at most 3 rows qualify … quantised
to `{0, 1/3, 2/3, 1}`"*, which A-51 withdrew in favour of **7 rows** and `1/7`; §10.1's K5
bullet carried the same figure. The computation was always `1 / n_rows_tested` and was never
wrong — only the prose was — and the error was **conservative**, since a margin chosen against
a coarser grid than the real one still clears the real one. That is why it survived: nothing
downstream was affected, so nothing downstream complained.

**Resolution.** Both numbers corrected, and `G-SPECNUM` now parses §05.10's and §07.4.1's
tables **out of the spec files** and requires them to agree with each other and with
`parameter_report()`. A future edit to one table that does not touch the other now fails.

### A-78 · §10.1's kill criteria existed only as prose · MAJ · FIXED
`scripts/spec_trace.py` exempted §10.1 with the reason *"kill criteria are evaluated in
analysis, mapped via eval/killcriteria"* — naming a module that **did not exist**. Two of the
ten were implemented in scattered form (`K1Verdict` in `soma/generations.py`, K8 inline in
`scripts/eval_tinystories.py`); the other eight were prose.

D-77 made every §10 decision mechanical precisely because spec v1's criteria all read "if X is
unchanged on **every** metric" — unfireable, since no equivalence test existed and a
conjunction over §07.6's dozens of fields always has one field differing by chance. A criterion
that is only prose is not pre-registered in any sense that binds: it is decided after the fact,
by whoever is reading the numbers, which is the situation the whole section exists to prevent.

The exemption is also worth recording on its own account: it was **false**, and
`spec_trace.py` reported "unmapped: 0" partly on its strength. An exemption with a stated
reason is only as good as the reason.

**Resolution.** `soma/eval/killcriteria.py` implements all ten as boolean decisions over run
artifacts, with §10.1's checking order (K6 and K9 first, both invalidating everything
downstream) and `Verdicts.blocking` so a report cannot bury a mixer mismatch under nine other
rows. Every criterion returns `applicable=False` with a reason when its inputs are absent:
**"K3 did not fire" and "K3 was never evaluated" must not be the same line in a report.** The
15 tests exercise each criterion **both ways** — an input that must fire it and one that must
not — because a kill criterion that can only be shown not to fire is the defect §10.1 was
written to remove.

### A-79 · The certification's ramp arm was not matched to the live arm · MAJ · FIXED
§06.4 specifies the ramp as *"a scripted monotone ratchet on `n_len` alone, **matched to the
live arm's mean solve-rate trajectory**"*, and K5 is adjudicated on `live - ramp`. v3.0
implemented an **unmatched** linear interpolation of `n_len` from 4 to 32 across the run, under
a docstring asserting that a linear ramp was "the simplest schedule with that property" while
nothing measured the property and nothing anywhere read the live arm's solve rate.

An unmatched ramp makes `live - ramp` a comparison of two different task **distributions**
rather than of two archive **dynamics** — which is the attribution A-28 introduced the arm to
obtain. Harder ramp: the ramp wins on novelty for a reason unrelated to open-endedness.
Easier ramp: the live arm does.

**Resolution.** `RampController` measures both arms' solve rates on the **same model at the
same step** and ratchets `n_len` up by one whenever the live arm is easier by more than a
tolerance. Matching is one-sided by construction — the ratchet is monotone, which is what keeps
the arm a difficulty *tracker* and not a second adaptive archive — so the residual is reported
as `ramp_matching_error` with every certification rather than assumed. With no `solve_rate_fn`
the arm **holds** rather than ramping on a schedule, because an unmatched ramp is not a
control.

### A-80 · Five specified ablations and baselines had no switch anywhere · MAJ · FIXED
§08.6 states that *"every novel component carries exactly one ablation and one kill
criterion"*. v3.0 could reach four: A2, A3, A12 and B1. **A7, A9, A10, B2 and B3 were fully
specified and had no implementation switch at all**, and the three that did exist were
unreachable from any driver's CLI (`--ablation` parses `name=value` with `float(v)`, so only
scalar policy pins arrived).

The consequences are specific, not general: §05.11's falsifier for the router needs B3, which
§05.4 additionally makes **required in every routing experiment**; §05.11's falsifier for the
context gate needs A10; §01.8 lists A10 and A12 as falsifiers of the Moment regime and only
A12 existed; K7 is adjudicated against A7; and K10 is written on B3. Five falsifiers and two
kill criteria could not be evaluated.

**Resolution.** Each arm is exactly one switch on `LifetimeConfig`, applied before the
optimizer is built, and reachable as `--arm {A2,A3,A7,A9,A10,A12,B1,B2,B3}`. Two readings had
to be settled and are recorded rather than assumed:

* **B3.** §08.4 says *"routing assignment fixed per unit slot at initialisation"*. Read
  literally — one fixed score per slot — every token selects the **same** `k_route` units and
  `U_max - k_route` units are permanently unrouted. A baseline whose population is dead by
  construction cannot test whether the population carries information; it assumes the
  conclusion. B3 therefore freezes the routing **function** at its random initialisation and
  forces `r_ut = 1/k_route` over the realised routed set, with `lambda_wire` pinned to 0 as
  §08.4 requires.
* **A9.** *"freeze `p_u` and `g_u`; train `A`, `B`, `W_shared`"*. `g_u` changes only through
  birth and mutation, so freezing it means no structural events; `p_u` is a parameter and is
  frozen in place rather than dropped from a group, which keeps §05.7's
  exhaustive-and-disjoint property.

Every switch's test asserts that flipping it **changes the model's output**, and A7's and B3's
additionally assert the property (`beta <= 1`; frozen parameters *and* uniform weights). A
wired-but-inert ablation flag is worse than a missing one: the arm runs, the report names it,
and the comparison is against the same model.

### A-81 · `L_star` at the full matched budget censors every lifetime, so the decisive experiment's primary statistic is a constant · BLK · FIXED
*(Found by running the M0 baseline and one full SOMA lifetime back to back, before scheduling
the generation curve.)* This is a **pre-run** finding, which is the only kind that saves the
compute it is about.

§07.1 defines `L_star` as the M0 baseline's held-out CE **at a matched token budget**, and
`CriterionScore` as time-to-criterion `n_crit` against it. Measured at M-nano, both arms on
30.7M tokens:

| | held-out CE |
|---|---|
| M0 (dense `ReLU^2`, no SOMA) | **2.0205** |
| SOMA, full machinery | **2.1249** |

SOMA does not reach `L_star`. So `n_crit` is `None`, the lifetime is **right-censored**,
`CriterionScore = 0` and `SlopeScore = 0` — and that is not a property of *this* lifetime but
of every lifetime the experiment would run. §07.5's primary statistic would be **identically
zero across all generations, both arms and all five seeds**, and the BCa bootstrap would fit a
slope through a constant. The 46 GPU-hours of §07.5 would have measured nothing, and K1 would
have "fired" on a statistic that could not have moved.

§07.5's censoring rule anticipated the shape of this — *"if the censored fraction exceeds 0.5
the curve is not interpretable and `L_star` is re-derived at the measured `D_tok`"* — but the
rule fires **after** the run, and `analyse_curves` would have reported `interpretable: False`
having already spent the budget.

**Resolution.** `L_star` is selected from the M0 curve at the largest budget fraction a
generation-0 lifetime reaches **with headroom**, and the whole curve is published in the M0
report so the choice is made from data and recorded rather than argued. Measured at M-nano:

| `L_star` | M0 at | SOMA reaches it at | as a fraction of `N_step` |
|---|---|---|---|
| 2.8170 | 10% | step 350 | 0.14 |
| 2.4467 | 25% | step 900 | 0.37 |
| **2.2486** | **50%** | **step 1850** | **0.76** |
| 2.1675 | 75% | step 2300 | 0.94 |
| 2.0205 | 100% | never | censored |

`0.50` is the default: it is the largest fraction whose `n_crit` lands before `0.8 * N_step`,
so a *faster*-adapting germline has room to show an earlier crossing, which is the entire
quantity §07.5 measures. At `0.75` the crossing is at 94% of the run and an improvement of
more than 6% saturates against the run's end — the same censoring, one step removed.

`scripts/run_generations.py` **refuses to start** against an M0 report that does not carry the
curve, rather than falling back to the full-budget value.

**The general point.** A criterion no run reaches and a criterion every run reaches instantly
are the same defect: the statistic is constant and the experiment is uninformative. This
register has recorded four thresholds that could not fire (A-36, A-58, A-61, A-73) and one
that could not pass (A-68); this is the first one on the **primary statistic of the decisive
experiment**, and it was found only because a real baseline and a real lifetime were run
against each other before the experiment was scheduled.

**[OPEN — the substantive result underneath]** SOMA at M-nano is **behind** a parameter-matched
dense baseline at matched tokens (2.125 vs 2.020 nats), and that is reported at full
prominence, not buried in this entry. §04 of the plan fixed the reading in advance — SOMA
spends `f_oe` of its stream on synthetic tasks, scores only `k_tok` of natural tokens, has
non-stationary capacity, and the thesis lives in the **generation** curve rather than in one
lifetime — but the number is what it is, and M2's comparison is a comparison SOMA currently
loses.

## Part A''' — the adversarial bug hunt (Agent B)

Run against the implementation with mutation testing and coverage: the auditor broke things
deliberately and measured whether the suite noticed. That method found a class of defect the
conformance audit could not — **not code that is wrong, but code that nothing constrains.**

### A-82 · K1 was decided on the confounded arm; the A4 control was computed and discarded · BLK · FIXED
`soma/generations.py`'s own docstring:

> The **difference** between the curves is the inheritance effect; the absolute curves are
> confounded by ordinary recipe drift and **are not interpretable alone**.

`scripts/run_generations.py` then did `report["K1_fires"] = verdicts["main"]["K1_fires"]`, and
nothing anywhere subtracted. K5 in the same codebase *is* adjudicated on `live - ramp`, so the
differencing pattern was known and simply not applied to the criterion that decides C1.

**Failure scenario, reproduced.** Two arms improving identically — pure recipe drift,
inheritance effect exactly zero — over 5 seeds x 20 generations:

```
  main arm  : b_criterion = +0.2993  significant  K1_fires = False
  a4 control: b_criterion = +0.2985  significant  K1_fires = False
  report["K1_fires"] -> False
```

The project would have been **declared to have met C1 on an inheritance effect of 0.0**, and
half the M3 GPU budget — the entire A4 arm — never entered the decision. This is audit A-28's
defect (a control that cannot test what the criterion asks) inside the criterion A-28 exists
to protect.

**Resolution.** `analyse_arm_difference(main, a4)` bootstraps the slope of the **seed-paired
difference**; the two arms share seeds by construction, so pairing removes each seed's own
recipe draw. `k1_generation_curve` **refuses** a verdict that is not marked
`arms_differenced`. Verified on the reproduction above: the differenced verdict gives
`b_criterion = +0.0007`, not significant, `K1_fires = True` — and a genuine effect (0.45 vs
0.30 drift) gives `+0.1485`, significant, `K1_fires = False`.

**Why it stood:** `soma/generations.py` had **0% test coverage**. 114 statements — the whole
decisive experiment — with no test file importing it. `tests/unit/test_generations.py` now
covers it, and that is the general repair: the module that adjudicates the central claim was
the least tested in the repository.

### A-83 · §02.5 could not detect the defect it was rewritten for, and rejected M1 · BLK · FIXED
Two defects in the acceptance test, found by reinstating A-57 and by sweeping seeds.

**The gradient-flow test was blind to A-57.** A-57's pathology was a `43-61x` gradient
discontinuity concentrated in **block 0**; the repair made the gated statistic the **median**
of `L-1` per-block ratios, which is by construction immune to a single outlier. Reinstating
the defect:

| | median (gated) | max | total |
|---|---|---|---|
| M-nano healthy | 1.37 – 1.43 | 2.96 – 3.22 | 12 – 16 |
| M-nano, A-57   | 1.43 – 1.47 | **42.4 – 52.3** | 256 – 301 |
| M1 healthy     | 1.21 – 1.23 | 2.64 – 2.75 | 23 – 25 |
| M1, A-57       | 1.23 – 1.25 | **60.9 – 63.9** | 733 – 779 |

The median moves by `0.04` and the test passes at both configurations. A-57's own conclusion —
*"a check that measures the wrong quantity does not merely fail to catch a problem, it
certifies its absence"* — applied verbatim to the repair written for A-57.

**The variance test's lower edge was its own design value.** `CALIBRATE` drives `Var(y) = 1`
against a He-normal reference of `Var(y_ref) = 2`, so the ratio is `0.5` **by construction**,
and v3.0's band was `[0.5, 2.0]`. Measured at M1: `0.4991, 0.4996, 0.4999, 0.5031` — **three of
four seeds abort at `LIFETIME` step 3**. §07.5 requires `n_seed >= 5`; a quarter to a half of
them would never have started. M-nano's `0.62` sat comfortably inside, which is why the
M-nano-only unit test never saw it — the same one-configuration blindness as A-32.

**Resolution.** A **maximum** per-block bound of `8.0` beside the median (2.5x above the worst
healthy value, 5x below the mildest defect, and depth-invariant where the *total* is not), and
the variance band recentred on the design value as `[0.25, 1.00]`. Verified over 10
seed/configuration pairs: **10/10 healthy pass, 10/10 A-57 rejected**, each naming the block.

The two statistics are kept as a pair deliberately: the median is robust and catches a
systematic drift across the stack; the maximum is sensitive and catches one discontinuous
block. Neither alone is the test.

### A-84 · The credit audit reported `valid: true` on a lifetime in which it never ran · BLK · FIXED
`valid` returned `isnan(last_rho) or last_rho >= threshold` — **`True` when no measurement
existed**. And on the integration configuration a whole lifetime accumulated **40** pairs
against the 64-pair minimum, so `spearman` stayed `NaN` and `n_evaluations` stayed `0`: the
metrics asserted the proxy was valid on a run where nobody had looked. §04.10's own regression
test was failing on a clean tree and the `valid` flag hid it from any consumer reading only the
metrics.

**Resolution.** `status` is three-valued — `unmeasured` / `valid` / `invalid` — and `valid`
returns `None` before a measurement, which is the same distinction `killcriteria` draws between
`applicable=False` and `fires=False`. And the audit now samples at least `min_pairs / k_route`
cases per event, so the 64-pair minimum is reachable in ~4 events at any scale while the
doubling rule keeps its effect on the estimate.

### A-85 · `median_phi` was the median of gene 0, and `apply_calibration` resampled `B` from the global RNG · MAJ · FIXED
**The stride.** `ph.flatten()[:: ph.numel() // 4096]` over a `(4096, N_G)` tensor has stride
exactly `N_G`, so it selected **column 0 of every row**: the median of **1 of 2048 genes**. It
is exact only while `log_sigma` is uniform — i.e. only at generation 0, before ES has touched
the field — so the diagnostic became wrong precisely when it started carrying information.
`median_phi` is the `[EMPIRICAL]` statistic §01.2 used to set `SIGMA_G_INIT` after audit A-14
found the v2 value wrong by 8x.

**The calibration.** `apply_calibration` called `B.normal_(0, std)` with **no generator**. Two
things at once: it **replaced** `B` rather than rescaling it, contradicting its own one-line
docstring, and `Tensor.normal_` *does* accept `generator=`, so this was not one of §02.6.3's
sanctioned global-RNG exceptions but the A-64 defect in a place nobody had looked. `B` is the
largest tensor the Site multiplies by, rescaled here by ~161x, and its values depended on how
many global draws module construction happened to consume upstream.

**Resolution.** Random sampling over both axes for the median; a true rescale
(`B *= target/current`, cosine similarity 1.0 with the pre-calibration tensor) with an explicit
generator for the degenerate case. `sigma_g` is reported as `sigma_g_gene0` and
`sigma_g_median` rather than as one number that was gene 0's.

### A-86 · The discrete germline update selected nothing, and `generations_survived` counted appearances · MAJ · FIXED
§02.6 `[NORMATIVE]`: *"the discrete components ... are updated by **selection** and
recombination"*. `es_step` inserted **every** member's archetypes, losers included: a mirrored
pair at fitness `+1000` and `-1000` contributed both gene sets with equal standing, making the
Inventory a record of what *appeared* rather than of what *survived* — which is the whole
distinction §02.2.4's sampling weight rests on.

And `Inventory.insert` incremented `generations_survived` once per **call**, while `_harvest`
emits one archetype per surviving unit — ~120 at M-nano — so a single generation contributed
~240 increments. Measured: **1 unit over 40 generations and 40 units in 1 generation both
reported `40`.** Three consumers read the field as a generation count: `_evict`'s ordering,
`SEED_POPULATION`'s sampling weight, and the running mean of utility.

**Resolution.** Only members at or above the generation's mean fitness contribute — the
smallest selection `P_es = 2` admits — and `insert` takes the generation, making the increment
idempotent within one. Now `40 / 1` where it was `40 / 40`.

### A-87 · Two chance baselines were wrong, and both made the bar harder · MAJ · FIXED
`solve_rate` scores **exact match on the whole answer**, so `acc_random` is the probability the
best *constant answer* is right. Two declared values were not that:

| family | declared | measured modal-answer frequency |
|---|---|---|
| `DYCK` | `0.5`, as if the answer were a balanced/unbalanced bit | **0.023** (339 distinct answers; it is the residual open-bracket stack) |
| `COPY` | `1/alpha_sz = 0.0625`, the per-**token** chance | **0.0003** (exact match is `alpha_sz^-n_copy ~ 2e-10`) |

Under `scaled_acc = (acc - r)/(1 - r)` a wrong `r` moves the **bar**, not the score: `DYCK`'s
`0.80` bar demanded raw `0.90` instead of raw `0.805`, so a model at raw `0.85` reported `0.70`
and failed K7 where the honest figure is `0.847` and it passes.

**Resolution.** Both corrected, and `tests/regression` re-measures **every** row's declared
`acc_random` against the generator at a stated protocol and tolerance, so a value that drifts
from the task fails rather than quietly re-scaling a kill criterion.

### A-88 · MQAR's distractors were parsed into the lookup table, and the `depth` hold-out admitted the depth the suite scores · MAJ · FIXED
**MQAR.** `generate` appended `distract` **loose symbols** after the key-value pairs and
`_mqar_solve` parses the whole body two at a time — so every distractor was read **into the
table**, an odd `distract` shifted the alignment of everything after it, and a distractor
colliding with a real key overwrote that key's value. Measured before the fix: **7.17% of the
suite row's ground truths were wrong** and 2.9% of answers fell outside the declared alphabet.
`distract` was not a distractor parameter — raising it enlarged the table. `test_verifiers.py`
reimplemented the same broken parse, so it confirmed the defect rather than catching it.

A distractor in MQAR is an additional pair whose key is **never queried**. Keys are now drawn
distinct across real and distractor pairs and the query is taken from the real ones only.
Measured after: **0 ground-truth failures and 0 out-of-alphabet answers** at `distract` 0, 8
and 16.

**The `depth` hold-out.** `HOLDOUT["depth"] = (2, 8)` *admits* depth 8, `EDIT`'s ceiling was
8, and §08.2's `DYCK` row scores depth 8 — so the archive was free to train at exactly the
setting the bar tests. `P_mqar` was genuinely held out (archive `<= 16`, suite at `32`);
`depth` was not.

The guarding test could not tell the difference: it asked whether **any** hold-out parameter
was outside its band, and `n_len = 40` satisfied that for every row, so the `DYCK` case its own
docstring named was vacuous. Both are fixed — band and `EDIT` ceiling moved to `(2, 6)`, and
the guard is now per-family along the axis that family is scored out of distribution on — and
the guard is verified by a positive control: restoring `(2, 8)` makes it fail.

### A-89 · `Genome.load` silently discarded the ablation pin set · MAJ · FIXED
`Policy.from_constants` deliberately writes `0.0` into a pinned slot — "a value the pin
overrides on every read" — because a pinned constant has **no preimage** under its affine
link (§02.2.3). `Genome.load(..., pinned=None)` then decoded that `0.0` **through the link**
as if it were a real value, and **no caller in the repository passed `pinned=`**. Measured on
`m0_config(M_NANO)`, save then load:

```
  name          in-memory   after load
  f_oe                  0        0.25
  k_tok                 1       0.775
  lambda_wire           0    0.693147
  rho_0                 0         0.1
```

No exception is raised, and §02.3's hash guard cannot catch it: **pins are not hashed**, so
the germline file is byte-identical either way. The consequence is that
`--ablation lambda_wire=0 --genome ...` ran the **main arm's constants under the ablation
arm's label**, and §02.6.2's M-nano -> M1 germline inheritance lost every pin.

**Resolution.** The pin set is written into `meta.json` beside the weights and read back at
load. An explicit `pinned=` still takes precedence, but a pin present in the file and
**contradicted** by the caller raises rather than resolving silently — those are two different
experiments, and choosing one for the reader is exactly the failure this entry is about.

The pattern, for the third time in this register (A-70, A-72, A-89): **a field that exists,
is populated, and is read by nobody.** The germline's own `meta.json` already recorded
`config.ablation_pin`; `load` simply never looked at it.

### A-90 · Nothing constrained masking on the **production** forward path · BLK · FIXED
The masking logic exists **twice**. `test_attention_is_causal` and
`test_document_mask_blocks_cross_document_attention` both call
`Attention.attention_weights(...)` — the **diagnostic copy** — and never `Attention.forward`.
The depthwise causal convolution had no causality test of any kind.

**Verified by mutation, both ways.** Changing the conv's left pad from `(k-1, 0)` to
`(k-2, 1)` — a one-character leak of the next token — and running the suite:

```
  with the leak:  434 passed, 18 deselected     <- everything else
  with the leak:  3 failed                      <- the new causality module only
  restored:       441 passed
```

Agent B measured the same for `Attention.forward`'s causal mask, its document mask, its
key-padding mask, and `mixer_chunk`'s pad neutralisation: **five masking mutations, all
surviving 415 tests.** This is the worst failure shape available — the loss goes *down*,
nothing raises, and the model has been reading its own answers.

**Resolution.** `tests/unit/test_causality.py` probes by **perturbation**: replace every token
from position `j` onward and require the outputs before `j` to be **bit-identical**. It makes
no assumption about how masking is implemented — only that changing the future must not change
the past, which is what causality *means* and what has to survive any refactor of how it is
achieved. Attention, the short convolution, the mixer, the Site (position-wise, so not merely
not-from-the-future) and the whole model are covered, plus the document mask that §06.1.1's
`L_ref` cache binding depends on. The probe carries its own positive control: a deliberately
acausal function of the same shape must register a leak, so a zero cannot mean "the probe never
perturbed anything".

The general lesson, and the reason this was a BLOCKER rather than a test gap: **a second
implementation of a rule is a place for the rule to be true in one copy and false in the
other**, and the tested copy is by construction the one that is not running.

### A-91 · Three defects that changed what a report would say · MAJ · FIXED
**An empty evaluation reached the criterion.** `held_out_ce` returned `total / max(1, count)`,
i.e. **`0.0`** when no token was scored, and `CurveTracker.record` accepts any `ce <= L_star`.
One evaluation batch with no scorable natural token therefore set `n_crit` to that step and
**un-censored the lifetime** — on `CriterionScore`, the primary statistic of the decisive
experiment. It now returns `NaN` with a zero count, and `record` requires `n_scored > 0` and a
finite `ce` before the criterion can be reached, while still recording the gap on the curve: a
missing evaluation is a fact about the run.

**A skipped suite row vanished.** `_fits`' docstring says a row that does not serialise inside
`T_ctx` is "skipped and **reported as skipped**"; `run_suite` omitted it entirely. At M-nano
that silently drops `COPY@96`, and a downstream `all(r.passed)` gate cannot tell a suite that
passed six rows from one that passed five and lost the sixth. Skipped rows are now emitted with
`skipped=True`, `n_eval=0` and `NaN` accuracy.

**The ES postcondition covered only the bounded links.** §02.6.1 says "every constant, after
link and rounding, in range", and `PolicyConstants.__post_init__` range-checks the
affine-sigmoid rows and the simplex only — so `constants()` accepted `U_cap = 100000` against
`U_max = 256` and `sigma_pos = 5.2e21`. The `softplus`/`round_softplus`/`exp` links map onto
the positive reals and declare no upper bound, so what is checked now is what is unambiguous:
`U_cap`, `n_par` and `M_mat` at least 1, and `sigma_pos`, `sigma_mut_0` strictly positive and
finite. Their *upper* bounds are configuration-dependent and stay where the configuration is
known — `SomaConfig.validate()` requires `U_cap <= U_max` — because the germline is scale-free
and cannot know `U_max`. Inventing a bound in `es_step` would make it not scale-free.

### Superseded register entries
The following Part B entries were correct against spec v1 but are **superseded** by the
resolutions above. They are retained rather than edited, because the point of the register is
that the repair is auditable.

| Entry | Superseded by | Was |
|---|---|---|
| R4 (init: `Var(b_g) = gain_act/(N_eff E2)`; `log sigma_g = log(0.6)`) | A-13, A-14 | analytic prefactor wrong; `sigma_g` gave median `phi` = 0.062 |
| R8 / B16 ("the sphere ... cannot drive a collapse to a point") | A-9 | false; collapse is the global minimiser |
| R2 / B7 (`N_G` per config: 512 / 2048) and B37 (`N_G` a perfect square) | A-16 | made M-nano → M1 inheritance shape-impossible |
| R6 / B24 (`q_cases = ceil(D_min / d_rate)`) | A-11 | inversely proportional to `d_rate` |
| R6 / B21 (`tau_death` on `nu_hat`) | A-18 | `nu_hat` has median 1 by construction |
| R7 / B27–B28 (bifurcation estimator at the birth) | A-36 | zero by Theorem 3.1a; alarm unreachable |
| R7 / B29 (`GROW` sized by `g_growth`) | A-21 | saturates the cap on event 1; A8 inert |
| R8 / B14 (`BUDGET_HEADROOM()` pre-deaths) | A-17 | births permanently zero at the cap |
| R10 / D-14 (capture every `T_cert / K_ring`) | A-23 | contradicts the firing rate |
| R10 / D-24 (synthetic within-source ranking as the fix for `Delta_i`) | A-25 | fixes comparability, not existence |
| R10 / D-43 (`n_len` holdout only) | A-30 | three inconsistent rules; no `T_ctx` clip |
| R11 / D-41 (censoring) | A-34 | consequence for the estimator unstated |
| R12 / D-26 (noise floor over identical seeds) | A-12 | exactly zero, so every margin zero |
| R12 / D-71 (frozen-archive control) | A-28 | does not separate open-endedness from ramping |
| R12 / D-36 (A1b on `lambda_size`) | A-31 | provable no-op |
| R9 / A15 (mirroring makes `P_es = 2` an antithetic pair) | A-39 | true, but rank normalisation discards the magnitude |

---

## Part C — Items deliberately left `[OPEN]`

| ID | Item | Deferral criterion |
|---|---|---|
| O-1 | `float8` for the head and `B` projection | a profile showing those matmuls above 15% of step time |
| O-2 | SwiGLU vs `ReLU^2` at the Site | an M2 A/B at matched parameters |
| O-3 | Sliding-window attention | only enabled for `T_ctx > 2048`; untested below |
| O-4 | Cross-layer parts-library sharing granularity | measured at the `N_G` sweep of D-75 |

---

## Part D — Verification of the repair

Phase 0 exits when the adversarial spec auditors are re-run on the repaired specs and
return **zero BLOCKER and zero AMBIGUOUS** findings, or an explicit `[OPEN]` with a stated
deferral criterion. Gate `G-SPECAUDIT`.

**[NORMATIVE — what the second audit established about this gate]** The v2 audit returned
80 defects against specs that had already passed one repair pass, and several of its worst
findings were defects the repair *introduced*. Two conclusions are carried forward and
bind the rest of the project:

1. **A clean audit is not a proof of correctness**, only evidence that a particular class
   of reading found nothing further. The defect trajectory 170 → 80 with a shift from
   architectural to missing-constant defects is convergence, not completion.
2. **Code is the next auditor.** The remaining `[OPEN]` items — `T_life`, `gain_act`, the
   `E[phi]` calibration, the realised profiling shares — cannot be settled on paper, and
   are deliberately deferred to measurement rather than argued to a conclusion. The
   adversarial review of §08 of the plan is run against the **implementation**, not against
   the specification again.

   Entries A-46 through A-63 are the evidence for this. **Eighteen** defects were found by
   running the code, and **not one of them was reachable by reading**: a default with no preimage under
   its own link (A-46), a dtype floor stated as an exact dtype (A-47, A-54), a "measured"
   constant that was never measured (A-48), a parameter table that omitted a tensor the same
   document specifies (A-49), an inheritance rule incompatible with a theorem four sections
   away (A-50), a granularity claim off by a factor of two (A-51), a community algorithm that
   optimises a different objective from the one it reports (A-52), a store sized for one Site
   in an architecture with `L` of them (A-53), and an asynchronous GPU fault that no amount of
   reasoning about the mathematics could have located (A-55).

   The seven found after that (A-57 … A-63) sharpen the point rather than repeating it: every
   one is a **quantity whose realised value on a running system differed from the value its
   own specification implied**. An embedding scale that made a `sqrt(d_model)` factor a 43x
   gradient discontinuity (A-57); a noise floor that was exactly zero on deterministic
   hardware (A-58); an audit object that was constructed, logged, and never called (A-59); a
   rate whose denominator made its own falsifier unfireable (A-60); a threshold anchored to a
   scale that grows while the threshold does not (A-61); a cap that counted a resource as
   released `phi_ann` steps before it was (A-62); and a diagnostic that ended the run it was
   measuring (A-63). None of these is a mathematical error. Each is a **statement about the
   system's own behaviour that had never been checked against the system**, which is the only
   category of defect a specification cannot audit itself for.

3. **Diagnosis discipline, learned from A-55.** A `NaN` whose location moves between runs with
   identical seeds, and which disappears when a check for it is added, is an **asynchronous
   fault**, not arithmetic. Reason about the mathematics only after `CUDA_LAUNCH_BLOCKING=1`
   has been ruled out. Two mathematically plausible hypotheses were tested and falsified before
   the real cause was found; both are recorded under A-55 so the next reader does not repeat
   them.
