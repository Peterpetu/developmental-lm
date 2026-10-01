# 09 — Accounting and numerics

**Status: normative.** Parameters and compute per component (§9.1), the per-token accounting
against a token model (§9.2), memory (§9.3), initialisation and bounds (§9.4), dtypes and
stability (§9.5), the identity tests that bind every rule (§9.6), and the acceptance tests
(§9.7).

## 9.1 Parameters and compute (TS-adult)

**[NORMATIVE]**

| component | parameters (formula value) | per-span cost | when |
|---|---|---|---|
| backbone matrices and vectors (03 §3.9) | `70,198,480` provisioned; `17,886,264` active at birth | one backbone step over the active blocks: `F_span` of 05 (5.2) | τ0/τ1, every span |
| text interface: `T_emb` `8192 × 128`, `W_x` `256 × 512`, `μ_m` `7 × 256`, `W_τ` `256 × 28` | `1,189,376` | one embedding gather and one `(256 × 512)` product | every text span |
| span codec (02 §2.2): encoder frozen, decoder trained on | `≈ 9.6M` | at `λ_lat = 0` the **encoder does not run in training** (nothing needs `x̄`); the decoder's `K` causal steps run per text span | the decoder every text span; the encoder only under the `λ_lat > 0` ablation and in R0 |
| other encoders/decoders (02 §2.9): `E_img` (4 convs, `≈ 0.39M`), `Dec_img` (`≈ 0.39M`), `E_sp` (`≈ 0.12M`), `W_prop, w_rew`, pattern encoders (linear), Gaussian/Bernoulli heads, `D_flow` (`≈ 0.66M`) | `≈ 1.8M` | per span of their modality | when present |
| expression map `θ` (04 §4.3) | `11,302,816` | once per episode (4.3), plus the 500-step regression (4.4) on `11.3M` parameters | τ2 |
| consolidator `ψ` (06 §6.10) | `≈ 5.2M` | one reverse read of `32,000` summaries per episode | τ2 |
| genome `g`, homeostatic `ζ` | `512`, `32` | — | τ2 |
| criticality probe (07 §7.4) | — | `n_pr n_δ = 64` stack-JVPs (`≈ 128` window-stack forwards) per episode | τ2 |
| **per-step overhead of the genome machinery** | | the elementwise fold, `< 0.1 %` of the step | τ1 |

Boundary cost in total: the regression (`500 × 2 × 11.3M × 16` pairs `≈ 1.8 · 10^{11}` FLOPs),
the reverse read (`≈ 32,000 × 2 × 5.2M ≈ 3 · 10^{11}`), the criticality probe (`≈ 10^{12}`),
and the fitness evaluations of 06 (6.6) (fresh-state scoring of the fixed sets and, for the
reward-bearing streams, `n` greedy rollouts) against an episode of
`N B T_s F_span ≈ 1000 × 32 × 128 × 6 × 17.9M ≈ 4.4 · 10^{14}` FLOPs at the primordium:
`< 0.5 %` with estimator 1; estimator 2 adds its `(G + 1) n_probe / N = 50 %` of probe steps,
which are **charged to the episode** and enter every equal-compute comparison (10 §10.1).

**What `F_episode` counts.** The training rows' forward and backward (`μ = 6` in 05 (5.2)),
**the interface** at the same factor (05 (5.2b): each input span's encoder and each scored
target span's head, measured on the real modules), the rollouts' inference forwards (`μ = 2`,
scaled by the fraction of the window each rollout actually used), and, added to `F_cum` at the
boundary, the probes' steps. It is recorded *before* any G-STAT doubling of `N`, so an
episode's cost and its `N` always match. There is exactly **one** definition of training
FLOPs in this specification, and `scripts/tempo_cont.py` uses it too; an acceptance test
asserts the two agree to 5 %.

**Rollout batching.** A group's `G` rollouts and an evaluation set's `n` rollouts are
independent, and the acting path steps them **in lockstep** through one batched backbone call
per span (08 §8.3). This changes no result — the batched and the one-at-a-time paths are
asserted equal on every reward-bearing stream — but at batch size one the card is idle, and
the blend runs (R8, R9) are dominated by these forwards.

## 9.2 Per-token accounting against a token model

**[NORMATIVE]** A token model of width `d` runs one backbone step per token. TEMPO runs one
per `K` tokens plus, per token, one causal decoder step at `d_c = 256`. **Measured** at the TS
shape (the FLOP counter, forward): backbone `9.23M` per token, decoder `7.37M` per token,
input embedding `0.07M` — the decoder is `44 %` of the total, not the `≈ 1.6M` (`about a
third of the backbone`) this section first estimated, which was wrong by `2.3x` and is
corrected here (11 §P-C.34). At `K = 4` the backbone's per-token cost is a quarter of the
token model's at equal active size; the decoder adds most of that back,
so the span agent at the primordium is expected near `2.5x` the token model's tokens per
second at equal active parameters, before the delta rule's and attention's sequence terms.
K-CONT (10 §10.2) is decided at matched FLOPs, so the span agent may be up to `K` times
wider than its token control; R1 measures the actual ratio and records it against ANLAGE's
throughput table (03 §3.9).

## 9.3 Memory

**[NORMATIVE]** At `B = 32`, `T = 512`, bf16 activations and fp32 optimiser state, the
adult's provisioned `70.2M` matrices cost `0.28 GB` (weights) `+ 0.56 GB` (Muon momentum and
AdamW state) `+ 0.28 GB` (gradients); activations of the active network at micro-batch 8
(`anlage/recipe.py::micro_batch_for`'s rule for `78M`) are the bound and are measured in R1
with the median-of-CUDA-events rule; the fast state `S` costs `B × L_delta × h × d_head² ×
4 B = 32 × 12 × 8 × 4096 × 4 = 50 MB`; the window summaries `32,000 × 770 × 2 B = 49 MB`;
the rehearsal buffer holds identities only. The peak is recorded per configuration and must
stay under `13 GB` on this card (leaving the user's other use of it, 00 §0.6); micro-batches
are chosen by the card's rule to hold that.

## 9.4 Initialisation and bounds

**[NORMATIVE]**

* Matrices `N(0, 1/fan_in)`; conv taps the identity; branch gains `n = (2L)^{-1/2}`; norm
  gains `1`; the delta offsets `a_0 = 4` (decay `σ(4) = 0.982`), `g_0 = 0` (write `2σ(0) = 1`);
  the attention output-gate offset `2`, the delta gate a SiLU with offset `0` (ANLAGE 04 §4.4,
  §4.7; `anlage/ops.py`).
* Gains `1 + γ` with `γ = c_ρ tanh(u)` (04 (4.2)): matrix gains in `[0.5, 1.5]`, norm gains in
  `[0.8, 1.2]`, offsets within `±2`, learning-rate multipliers in `[¼, 4]`, switch
  pre-activations in `(−8, 8)`, mixture logits in `(−4, 4)`; no gain reaches zero.
* `‖g − g_0‖ ≤ 3 sqrt(d_g)` (06 (6.8)); `‖Π_τ Δg‖ ≤ τ = 0.2 sqrt(d_g)`;
  `σ_ψ ∈ [0.02, 0.2]` per coordinate, the upper bound being `τ / sqrt(d_g)` so that a draw at
  the bound has norm `τ` (06 §6.2); the KL penalty bounds the drift of the multi-step
  regression of 06 §6.6.
* `ζ ∈ [−2, 2]`, moved by at most `κ = 0.5` per boundary (07 (7.4)).
* `β_t ≤ β_max = 1.999` (03 (3.5)); keys unit-norm; the delta state's spectral radius is
  bounded by `α_t ≤ 1` and the projection `(I − β k kᵀ)` with `‖k‖ = 1` and `β < 2` (SOMA 05
  §5.3: eigenvalues `1` and `1 − β ∈ (−1, 1]`).
* Softcap `15` on every logit (the codec's, the Bernoulli head's); z-loss `10^{-4}` on the
  codec's logits, applied **per scored row** (a sum over rows divided by the batch, not a mean
  that changes with how many rows a window happens to score, 11 §P-C.26); `w_s ∈ [−5, 2]` on
  the Gaussian heads, kept by **projection after each optimiser step**, not by a clamp inside
  the graph: a parameter clamped in the forward pass has zero gradient outside the range and
  can never come back.
* The span latent is unit-RMS; `λ_lat = 0` is the default, so `ŷ` is defined by the jointly
  trained decoder and heads (02 §2.2, 11 §P-C.23).

## 9.5 Dtypes and stability

**[NORMATIVE]** bf16 autocast for the backbone's matrix products and the interface; fp32 for
every norm, the delta rule's state and solve (`mixer_chunk.py`'s floor), the softmax of
attention (SDPA's own accumulation), every loss, the optimiser state, the gains `u`, `ζ`, `g`,
`θ`, `ψ`, the criticality probe (07 K5), the codec's KL. Gradient clipping at `1.0` on the
global norm. NaN/Inf in any loss aborts the step, records the event, and skips the optimiser
update (a counter in the record; more than `3` in an episode aborts the run with the state
saved). Every module asserts its input and output shapes at its boundary
(`gnomon/utils/shapes.py`).

## 9.6 The identity tests that bind every rule

**[NORMATIVE]** Each of the following is a test in `tests_tempo/` and a gate (G-IDENT):

| identity | statement |
|---|---|
| plain backbone | `u = 0`, `ζ = 0`, all groups on: the output equals ANLAGE's `Phenotype` forward on the same weights with the embedding replaced by `W_in`; a *non-uniform* perturbation of any role changes it (P6) |
| null consolidation | `Δg = 0`, `κ = 0`, no births, regression skipped: `Ξ` unchanged bit for bit (C9) |
| token special case | `K = 1`: the decoder has no layers and the head is the softmax of 02 (2.4b); `L_txt` is the mean NLL of the same tokens the token model scores (I5) |
| fold | `a = 0`: `W_eff = W` bit for bit (P7) |
| switch | `m_b = 0` or skipped: the function without the block (M1) |
| regeneration | `Δg = 0` after the regression: the next episode starts at the fitted gains (E7) |

## 9.7 Acceptance tests

**[NORMATIVE]**

| # | property | test |
|---|---|---|
| N1 | spec numbers | `test_spec_numbers` builds every configuration and asserts the parameter counts of §9.1, 03 §3.9, 04 §4.2–4.3 and 06 §6.10 as exact integers from the module tree (G-SPECNUM) |
| N2 | FLOPs | `flops_per_span` at the primordium equals a hand computation and is within 5 % of `torch.utils.flop_counter` on one step (M7) |
| N3 | boundary overhead | the recorded `F_probes` is zero under estimator 1 and equals the sum of the probes' step FLOPs under estimator 2; `F_episode` equals the sum of the episode's per-step `F` and is recorded before any `N` doubling; the boundary's non-probe cost (regression, read, criticality probe) is `< 5 %` of a 20-step CPU episode's and scales as `1/N` (the `0.5 %` of §9.1 is the `N = 1000` value) |
| N4 | bounds | after 200 steps of bf16 training on random data with adversarial gradients (scaled ×1000 before clipping), every bound of §9.4 holds and no NaN/Inf occurs (P11) |
| N5 | dtypes | the state, norms, losses, gains and optimiser state are fp32 under autocast (asserted per tensor) |
| N6 | memory | R1 records the peak memory per configuration and it is `< 13 GB`; the micro-batch chosen by the rule is recorded |
| N7 | identities | every row of §9.6 passes |
| N8 | shape assertions | a deliberately mis-shaped input at every module boundary raises the module's assertion |
