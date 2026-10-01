# 02 — Protocol: pre-registered readings, kill criteria, and what may be claimed

**Status: normative; written before the first run (2026-09-09, 22:20).** Every criterion is
decided by `scripts/spore_analyse.py` from artifacts in `runs/spore/`; a criterion whose
deciding run does not exist is reported `NOT EVALUATED`, never as passing. Verdicts at
`α = 0.05`; contrasts are paired by corpus offset and window set; the noise reference is
`σ_e` over atom seeds (§2.1). The test set (v3, residue 2, 256 windows × 512 targets — the
set every model in this repository is scored on) is read only where a row says so.

## 2.1 The instrument first (S6)

**G-NOISE.** Three atom seeds `s_A ∈ {0, 1, 2}` of the base genome at the screening budget
(5M tokens) and at the report budget (25M): `σ_e` is the sample standard deviation of the
test CE over seeds (2 df). No contrast is reported at a budget whose `σ_e` is unmeasured.
Expected: `σ_e ≈ 0.01` (the codes' interference is `1/√D`; the readout averages it over
millions of tokens).

## 2.2 Nulls and baselines (the floor and the ceiling)

| id | arm | what it is |
|---|---|---|
| N-UNI | unigram | the closed-form count model, add-one; the entropy floor of a memoryless predictor |
| N-BI | bigram, Kneser–Ney-free interpolation | exact counts `(v_t, v_{t+1})` with Jelinek–Mercer interpolation to the unigram, `λ_JM` fit on the fitness set |
| N-TRI | trigram, likewise | exact counts of triples; interpolation weights fit on the fitness set |
| N-SKETCH | the HDC sketch (T4 candidate A) | codes `c^{(n)}` bundled with the next atom by superposition, no readout: `ℓ_t = τ Σ_n ⟨A, Σ_s c_s^{(n)} 1_{v_{s+1}}ᵀ ⟩ c_t^{(n)}` — the pure superposition-counting language model |
| N-RES | the reservoir (candidate B) | `x_t = tanh(W_res x_{t−1} + W_in a_{v_t})`, seeded, spectral radius 0.9, `D_res = 4096`, one ridge readout |
| **B-GD** | the 13M transformer (`s0:d256/L8`) | the backprop ceiling: test CE **3.20 at 1.6M tokens, 2.85 at 2.5M, 2.2255 at 10M, 1.9975 at 20M, 1.857 at 25M (4 seeds), 1.804 at 201M** (`runs/bottleneck/free-tinystories-s0-seed0/loop.json`, `docs/anlage/RESULTS.md` R1) |

## 2.3 The readings of the base organism

| id | reading | passes when |
|---|---|---|
| R-BIRTH | the innate predictor (`p_t = softmax(τ_m m_t)`, no pass over the corpus; `τ_m` fit on the fitness set) | test CE **< 9.011** (the uniform null) by ≥ 3 `σ_e`; expected: far below null on repeated names and far above on the rest; the number is reported per stratum |
| R-PARADIGM | the base organism at 25M tokens | test CE **< N-TRI** by ≥ 3 `σ_e` (the paradigm carries more than counting) |
| R-CURVE | the base at 1.6M, 2.5M, 5M, 10M, 25M tokens | the curve is reported beside B-GD's at the same tokens; the **gap** and its **slope** (nats per doubling of tokens) are the K-SOTA numbers |

## 2.4 Kill criteria of the mechanisms (each against its matched control)

| id | fires when | consequence |
|---|---|---|
| **G-BPF** | any tensor of the organism has `requires_grad`, or `torch.autograd.backward` / `Tensor.backward` is called during a life (a sentinel test), or `torch.autograd.grad` appears in `spore/` | C2 is violated: a bug, fixed before any number is reported |
| **K-HDC** | fewer than two of the three ablations cost ≥ 3 `σ_e` at 5M tokens against the base: `binding = off` (`ρ = I`), `memory = off`, `D → D/4` | the HDC substrate is decorative; reported as such (C1 fails) |
| **K-DEV** | `growth = select` does not beat both `growth = off` (matched tokens; fewer features) **and** `growth = random` (matched features and tokens) by ≥ 3 `σ_e` at 5M tokens | development is not load-bearing, or its selection is not; reported *(fired in this form, 03 §P-S.19: selected = random; the criterion is read a second time for Definition 11b — units born by a ridge solve against the same `off` and `random` controls — where it does not fire, §P-S.25; both readings are reported)* |
| **K-EVO** | after the wall-clock budget of generations, `CE_test(elite) − CE_test(base) > −3 σ_e` (the elite is not better than the base on the *unseen* test set), or the fitness gain is carried by a gene whose effect is monotone above noise (a hyperparameter, not a search) — reported separately, not a fire | selection over lifetimes is not load-bearing (C3 fails as a search; the monotone genes are still reported as findings) *(fired, 03 §P-S.32–33: the elite's gain is the second refinement pass; `base + E 2` beats the elite at 5M and on the loop's own screen)* |
| **K-REFINE** | the delta-rule refinement (`E = 1`) does not improve on the ridge solution by ≥ 3 `σ_e` | the organism is fully closed-form; `E = 0` becomes the default (not a failure of the paradigm) |
| **K-SOTA** | the gap to B-GD at 25M tokens is reported together with the slope of the base's curve; fires when the slope over 2.5M → 25M is **flatter** than B-GD's (the paradigm loses ground with tokens), which would say the ceiling is representational, not budgetary | the honest reading of C6: the paradigm's place on the curve, and whether more tokens or more substrate is the lever |
| G-DET | two lives with the same genome, budget and offset differ in test CE by more than `10^{−4}` | a bug (the life must be deterministic given its arguments) *(fired at 1.6e-4 and 1.5e-4: CUDA atomics in `index_add_` and cuBLAS split-K reductions; both replaced, 03 §P-S.26/28; passes at 2.0e-5)* |

## 2.5 Reporting rules

1. Every run writes `runs/spore/<name>/life.json` with the manifest (`scripts/anlage_common.py::manifest`), the genome, the budget, the corpus offset, the evaluation digests, the timings, the per-stratum CE and the criterion inputs; the analyser recomputes every verdict from those files.
2. A criterion is decided only by the arm it names, at the budget it names.
3. Negative results are written in the ledger (`03-evidence-ledger.md`) with the number, the artifact and the reason, in the order they were obtained.
4. The test set is read for: G-NOISE, R-*, K-HDC/K-DEV/K-REFINE/K-SOTA arms, the evolutionary base and its final elite. Never inside selection. *(Correction, 23:25: every life records its test CE beside its fitness CE so that the transfer of fitness gains to the unseen set can be analysed per generation; `spore/evolve.py` sorts by `fitness_ce` only, and the reported K-EVO number is the elite's.)*
5. Which genes selection moved, in which direction, and whether a one-dimensional sweep of that gene reproduces the gain (monotone → a hyperparameter finding; non-monotone or interacting → a search finding) is reported with K-EVO.

## 2.6 The order of work tonight

R0 nulls and G-NOISE at 5M → R-BIRTH → the base at 5M and its ablations (K-HDC) → growth arms
(K-DEV) → refinement (K-REFINE) → the base at 25M and its curve (R-PARADIGM, K-SOTA) → the
evolutionary loop for the remaining wall-clock (K-EVO), with the elite re-scored at 5M and 25M
→ the variants (semantic atoms; the exact order-1 block; `N-RES`; `N-SKETCH`) as time allows.
*(Correction, 23:25: the evolutionary loop's screening budget is 1M tokens, not 5M — a 5M life
is 5–6 min and two hours would give two generations; at 1M a life is ~1.5 min and the same
two hours give seven or eight generations of eight. The elite is re-scored at 5M against the
base at 5M, which is the K-EVO reading; the 1M fitness is the selection signal only.)* Every run under
thirty minutes; a longer run only after a shorter one justified it (`tempo/11 §P-D.43`).
