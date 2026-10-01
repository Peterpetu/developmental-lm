# 08 — The life and inference: one loop

**Status: normative.** Birth (§8.1), the loop with every ordering (§8.2), lanes, boundaries,
group rows and the attention context (§8.3), inference as the same loop (§8.4), the checkpoint
and resume (§8.5), the recorded quantities (§8.6), and the acceptance tests (§8.7).
Symbols: 01–07. Implementation: `tempo/life.py`, `tempo/streams.py`.

## 8.1 Birth

**[NORMATIVE]** A life starts from a **model seed** (the initialisation, the proposals, the
probe draws) and an **order seed** (the window order; defaults to the model seed). Arms of a
contrast share the order seed and so read the same windows in the same order (CRN, 10 §10.1),
while the model seed sets what the agent is. With a configuration (03 §3.1):

| object | at birth |
|---|---|
| `W` (provisioned shape) | matrices `N(0, 1/fan_in)` (ANLAGE 04 §4.7, `anlage/model.py::free_init`); `n_{W,l} = (2L)^{-1/2}`; `ρ_W = 1`, `ρ_fin = 1`; taps the identity `[1, 0, 0, 0]`; `b_in = 0`; `a_0 = 4`, `g_0 = 0` (write gate `2σ(0) = 1`), `g_og = 2` (attention) and `0` (delta) as weight values before the offset roles |
| codec `E_txt, D_txt` | loaded from R0's artifact; `E_txt` frozen (it defines the target `x̄`), `D_txt` trained on at τ1 in the heads' optimiser group (02 §2.2); `E^{in}_txt`, `μ_m`, `W_τ`, every other `E_m, D_m` at their module defaults (`N(0, 0.02²)` for embeddings, Kaiming-uniform for convolutions, `w_s = 0` for Gaussian heads, `b ~ N(0, 0.5²)` for the affine linear encoders of 02 §2.9.3) |
| `g` | `g_0`, the seeded prior `N(0, I_{d_g})` drawn from the model seed (04 §4.1); **not** the origin |
| `ζ` | `0` |
| `θ` | `A_ρ ~ N(0, 1/(4 d_g))`; `B_ρ = 0` on the differentiable rows (trainable) and `B_prog ~ N(0, 0.2²)` fixed on the programme rows, which act in the differential frame of 04 §4.3 |
| `u` | `Φ_θ(g_0) = 0` (so every gain is the identity at birth, by the differential frame) |
| `ψ` | `W_μ = W_σ = 0`, `w_s = 0`, the rest by module defaults (06 §6.2) |
| switches | the primordium pattern (05 §5.5); born flags false outside it |
| `S` | `0` for every delta head and lane; the KV cache empty |
| `κ_m` | `1` for text and the task stream's text; `0` for every other decoder until its rule fires (02 §2.4) |
| buffers | rehearsal empty; accepted set `D` empty; the regression history of pairs empty; baseline `b = 0`; `R_scale` and the fitness scales `s_m` unset until the stream's first evaluation |
| RNG | `torch`, CUDA, `numpy` and `random` seeded from the model seed; the life's own CPU generator from it; the probe's generator from `(order seed, episode)` |

## 8.2 The loop

**[NORMATIVE]** `tempo/life.py::Life.run_episode(learn=True)` executes, for episode
`e = 0, 1, …`:

```
   boundary-in (τ2, once):
      B1  u ← Φ_θ(g_e) for every role (04 (4.3)); the expression computed at the *previous* boundary is
          reused (it is the same call); programme kinds fixed, differentiable kinds become τ1 leaves;
          u^{start}_e recorded for A8
      B2  switch pattern from the morphology roles (or the control arm's schedule at the spent budget
          fraction, 05 §5.7); births for groups crossing ε_skip (05 §5.3, hypercloning under the oneshot
          arm); active views rebuilt. A frozen pattern (after a genome reset, §8.4) is kept
      B3  optimisers rebuilt with groups scaled by exp(λ_grp); their state zero (the episode ends at
          η_1 = 0, 01 §1.2)
      B4  the episode's mixture μ (04 §4.4); lane counts by systematic_counts at the order seed; lanes
          whose stream is unchanged keep their cursor and fast state
      B5  the rehearsal buffer of episode e−1 attached; fresh window summaries; the FLOP counter reset
   steps n = 1..N (τ1):
      S1  draw B windows: for n < p_reh N from the rehearsal buffer ∝ s_w (§8.3), otherwise the next
          windows of each lane — an interactive lane contributes its teacher-forced supervised row and
          its G rollout rows in one step
      S2  forward: spans through 02 (2.1), the backbone 03 with each row's entering state (§8.3), ŷ per span
      S3  losses: the terms of 02 §2.4 present in each span's modality on the *supervised* rows, summed
          over scored positions and divided by the batch's token count -- the **scored** supervised
          tokens (a text or task position counts K, any other counts 1) plus the policy-gradient
          positions of the complete groups, and nothing else. A rollout row carries no supervised
          target and so contributes no supervised tokens (11 §P-C.36); the policy-gradient term
          (02 (2.12)) is formed over each complete rollout group
      S4  backward; clip at 1.0; Muon on matrices, AdamW on vectors and the interface, at η_1(n) exp(λ_grp);
          the Gaussian heads' log-std bound re-imposed by projection (09 §9.4). Skipped when learn=False;
          a non-finite gradient norm skips the step and is counted (09 §9.5)
      S5  each lane's S ← the final state of its rollout-0 row, detached (truncated BPTT of one window);
          the attention KV cache is dropped (window-local attention, §8.3)
      S6  the window summaries r_w (06 (6.1)) appended with their identities, streams and rehearsed flags
      S7  the step's FLOPs (training rows and rollout forwards) added to the episode's counter;
          every eval_every steps the fixed evaluation sets are scored from a fresh state
   evaluation at n = N (before the boundary changes anything): every stream's evaluation set, recorded
   boundary-out (τ2, once):
      O1  R_scale set on an environment stream's first episode; that episode's stored ℓ_w rescaled (06 §6.1)
      O2  G-STAT on the episode's steps after the rehearsal phase (01 §1.4); if it fails, N is doubled
          once per life and recorded; F_episode is recorded *before* the doubling
      O3  the fitness sets scored from a fresh state; s_m set at a stream's first evaluation; Ľ_e (06 (6.6));
          the reward r_{e−1} of the previous proposal (06 (6.7))
      O4  regression of θ over the history of (g, u) pairs (04 (4.4)); R_e recorded; G-REGRESS
      O5  the criticality probe (07 §7.4) on the episode's last-step windows with their entering
          states, under the episode's gains and switch pattern; Λ, Λ/K_a, Λ_se, the per-block
          profile with its standard errors, and the K-CRIT trigger's verdict recorded.
          **Before O6**: estimator 2 lives G+1 probe episodes and may adopt one, which replaces
          the weights, the gains and the pattern, so a probe run afterwards would measure a
          different network on a different step (11 §P-C.36)
      O6  estimator 1: the ψ update on the previous proposal, *then* this boundary's read and proposal
          (06 §6.5). Estimator 2: the read, the null probe and the G candidate probes, the winner kept,
          the elite regression (06 §6.6)
      O7  the genome step (06 (6.8)); ‖g − g_0‖ recorded
      O8  the ζ rule (07 (7.6)) on the relative profile
      O9  κ_m updated from each head's own loss over the episode (the action head from minus the mean
          return); the values that will apply next episode are recorded
      O10 the rehearsal buffer for episode e+1; the salience regression of the previous probe set on the
          cached c_w (06 (6.11)); the new probe set P_e; the task curriculum stepped (02 §2.9.2);
          u_{e+1} = Φ_θ(g_{e+1}) expressed and A8's C2 ratio recorded; the record; the checkpoint (§8.5)
```

`Life.run_episode(learn=False)` is the inference path: it executes everything above and skips
only S4 (A1).

## 8.3 Lanes, boundaries, group rows and the attention context

**[NORMATIVE]** The stream is read by `B` **lanes**. A lane is a cursor over the episode's
stream: for a corpus stream, consecutive windows of `T + K` tokens read in corpus order from a
seeded start offset (the cursor advances by `T`, so the fast state carried across a lane's
windows continues the same text); for a pattern stream, the *continuing* trajectory of one
generator instance seeded by `(order seed, lane)` — the last observation of a window is the
first input of the next, and a window is re-fetchable from its `(kind, seed, index)` because
the generator's state at that window's start is reproducible; for the interactive streams
(tasks, control, vision queries, speech), one prompt or start state per draw. Each lane's `S`
persists across its windows, steps and episodes (A3).

**Declared boundaries** reset a lane's `S` to zero and drop its KV cache. They are declared
(a) at birth and at a lane's first window; (b) when a lane's source is exhausted and restarts;
(c) by an explicit reset event in a stream (used by tests and by the evaluation protocol);
(d) at each new instance of an interactive stream (a task instance, a scene run, an utterance,
a control rollout group), which is a new source and not a continuation of the previous one;
(e) never otherwise — in particular not between the consecutive windows of a corpus lane, a
pattern lane or a control rollout.

**Group rows and their entering states.** An interactive stream's draw yields one
teacher-forced **supervised row** (the truth is the input at the answer positions, so the
likelihood of 02 (2.6) is a likelihood of the truth given the truth's prefix) and `G`
**rollout rows** (the agent's own emission is the input; no supervised target; the
policy-gradient positions and the group's reward). The rows of a group are returned together,
in one step, so the group advantage of 02 (2.12) is formed within the step. Every row carries
the fast state it was produced under: the rollouts were sampled from a fork of the lane's
state, so `state_in` of each row is that fork, and a control rollout's `k`-th window carries
the state the rollout had entered that window with. The lane continues from the supervised row
(rollout 0). Scoring a row under a state it never had is the defect this fixes (11 §P-C.29).

**Attention context.** Attention is causal over the current window's `T_s = 128` spans only;
the KV cache is not carried across windows in training (it would cost `T_s` spans of memory
per lane and layer for a context the delta state already carries) and a full cache starts a new
window rather than growing. At inference the cache is window-local in the same way. The long
memory of the agent is the fast state `S`; attention is the precise short memory; this
division is the reason one layer in four is attention (00 §0.2, 11 §P-A.4) and it is what P4
and P3 test.

**Evaluation and fitness sets.** Two disjoint fixed sets per stream (10 §10.1): the
**evaluation set** reports the run's score, the **fitness set** feeds the reward of 06 §6.3.
Both are scored from a fresh state at each window (a declared boundary), so the number is a
function of `W`, the gains and `ζ` only and is comparable with the token baseline, which has
no state across windows. For the reward-bearing streams both are greedy rollouts on reserved
seeds, scored by the exact reward. The *continuing-state* score (the same windows read
consecutively with `S` carried) is recorded as a second number and never used in a verdict.

## 8.4 Inference, and the genome reset

**[NORMATIVE]** Inference is `run_episode(learn=False)`; `Life.step_inference(lane, raw, m, t)`
is one span of it:

```
   x_t = E_m(raw_t) + μ_m + τ(t)                                            02 (2.1)
   ŷ_{t+1} = F(x_t; S, W, u, ζ)        one backbone step; S in place; the window-local KV cache appended
   if act:  raw_out ~ D_{m'}(· | ŷ_{t+1})   text by 02 §2.3; actions by 02 §2.5; frames by the Gaussian head
            the emission is perceived at t+1 through E_{m'} (01 (1.4))
   every N spans, or at a declared episode boundary: the boundary-out of §8.2 with η_1 = 0
```

Per span: one backbone step over the active blocks plus a `K`-step decoder of `d_c = 256`; per
token, `1/K` of a token model's backbone steps at equal width (09 §9.2).

**The genome reset** (R7, R9's control arm) sets `g ← g_0`, regenerates the gains from it,
clears the consolidation's memory (the regression history, the accepted set, the rehearsal
buffer, the previous summaries and probe set, the reward's previous value), and **freezes the
switch pattern**, while keeping `W`, `ψ`, `ζ`, the interface and the optimiser groups. It is
the "same weights, no genome" control: everything the fast system learned is kept, everything
the slow channel wrote is not, and the shape cannot change under the reset arm alone.

## 8.5 The checkpoint and resume

**[NORMATIVE]** `Life.save(path)` writes one file containing, and `Life.load(path)` restores
exactly:

| group | contents |
|---|---|
| parameters | `W` at the provisioned shape (inactive blocks included), the interface (`E^{in}_txt, μ_m, W_τ, E_m, D_m`) and the codec's trained decoder, `u`, `ζ`, `g`, `g_0`, `g_{e−1}`, `θ` (including the fixed `B_prog` and `g_0` buffers), `ψ` and its Adam state |
| morphology | the switch pattern, the frozen flag, the born flags |
| fast state | `S` of every lane, fp32, with each lane's stream, cursor and boundary flag |
| optimisers | Muon momentum buffers, AdamW `m, v, step` for every group, the group multipliers |
| episode | `e`, `N`, whether an episode is in flight and the step it resumes at, the number of `N` doublings, the non-finite-step count, whether the task curriculum could be seeded, the lane cursors, `κ_m`, `R_scale`, the fitness scales `s_m`, `Ľ` and the fitness of the previous boundary, the estimator-1 baseline, the accepted set `D`, the regression history of `(g, u)` pairs, the rehearsal buffer (identities, weights and mode), the probe set identities with their `ℓ_w`, the previous episode's summaries and cached `c_w`, the previous proposal and its read, `u^{start}`, the expressed `u` for the next episode, the per-head loss accumulators, the FLOP counters, the task archive |
| RNG | the `torch` CPU generator state, the life's own generator, every CUDA generator state, `numpy` and `random` states |
| record | everything of §8.6 so far |
| provenance | the configuration, the model and order seeds, the code digest (a hash of `tempo/*.py`), `torch.__version__` |

**Pausing and resuming inside an episode.** The card is shared with the user's other use of
it, so a run must be able to stop on request and continue without losing work or changing a
number. `Life.run_episode(should_pause=...)` consults the caller after every step; when it
says to stop, the life raises `Paused`, and the caller writes the checkpoint and exits. The
checkpoint records that the episode is **in flight** and which step comes next, so the resumed
life continues the step loop and does **not** re-run the boundary: the gains, the switch
pattern, the optimisers and the lane assignment of that episode were drawn once and are in the
checkpoint, and drawing them again would silently change the episode. The operational form is
a sentinel file (`runs/tempo/PAUSE`): the running life stops at its next step, exits with a
distinct code, the chain stops without touching anything else, and removing the file and
re-running the same command continues from where it stopped (10 §10.4). A run interrupted any
other way — a crash, a reboot — has no paused record and is simply re-run from its start.

Resume is **bit-exact on CPU** (L2, L2c): a life saved at the end of an episode, or in the
middle of one, and resumed produces the same `Ξ`, losses and record as the uninterrupted life. On CUDA the resumed life
matches the uninterrupted one to `1e-5` relative over 10 steps; the residual difference comes
from non-deterministic reductions (SDPA backward, `index_add`), is recorded, and is the reason
CRN pairing rather than bit-identity is the basis of every contrast (10 §10.1). Loading a
checkpoint whose code digest differs from the running code raises unless the caller passes
`allow_digest_mismatch` (L9).

## 8.6 The record

**[NORMATIVE]** Per step, **one row per step, always**: `n`, the episode, the mean per-token loss, the latent
term, `η_1(n)`'s fraction, the gradient norm, the token count, the loss's denominator, the
wall-clock seconds, the policy-gradient term, the step's FLOPs, whether the step was a
rehearsal step, and whether it was skipped for a non-finite gradient. One row per step matters:
a second row for a skipped step pushed the episode's first steps out of the last-`N` window
that G-STAT reads (11 §P-C.36).

Per evaluation: the fresh-state and continuing-state scores per stream, and the greedy reward
for the reward-bearing streams.

Per episode: `L̄_e`, `L^{fin}_e`, the per-stream means, the fitness scores and their scales,
`Ľ_e`, `F_episode`, `F_cum`, `F_probes`, `r_{e−1}`, `‖Δg‖`, the proposal's mean scale,
`‖g − g_0‖`, the regression residual `R_e` and G-REGRESS, `Λ`, `Λ_abs`, `Λ_se` and every
`λ_l` in both forms, `λ_ref` when the probe fell on a pattern window, `ζ`, the switch pattern,
the active parameter count and the births, `κ_m`, `μ`, the G-STAT statistic with its standard
error and ratio, the A8 ratio `C2` with its numerator and denominator, the probes' rewards
including the null, which candidate was kept, the rehearsal mode, the salience regression's
error, the curriculum's state, the median step time and its outliers, the wall-clock, the peak
memory. All in one JSON per life beside the checkpoint, with the manifest of 10 §10.4 and a
`"complete"` flag that is true only when every requested episode finished (or the compute
budget was reached); an incomplete record is not a result (10 §10.5).

## 8.7 Acceptance tests

**[NORMATIVE]** `tempo/life.py` is accepted only when all of the following hold (with A1–A8
of 01). Implemented in `tests_tempo/unit/test_life.py`.

| # | property | test |
|---|---|---|
| L1 | the order | a 2-episode CPU life with hooks records the boundary steps in exactly the order B1–B5, S1–S7, O1–O10 |
| L2 | bit-exact resume (CPU) | checkpoint after an episode; run on; restore; run the same episode again: the per-step losses, the final `Ξ` and the RNG states agree bit for bit |
| L2c | bit-exact pause *inside* an episode | a life paused before step `k` of episode `e` and resumed reproduces the uninterrupted life's remaining steps and final `Ξ` exactly, and its boundary-in does not run a second time; a `should_pause` that never fires changes nothing |
| L3 | resume (CUDA) | as L2 on CUDA: `1e-5` relative agreement over 10 steps after resume; the difference is written to the record |
| L4 | lanes | each lane's windows are consecutive in its source (asserted by offsets); a lane's `S` after window `k` is the initial state of window `k+1` and is zero after a declared boundary; a group's rows carry the state they were sampled under |
| L5 | fresh-state evaluation | the evaluation score of a window is independent of what the lane read before it; the continuing-state score is recorded separately |
| L6 | rehearsal | the first `p_reh N` steps draw only from the buffer; after them, lanes resume where they were; rehearsed windows are flagged in the summaries |
| L7 | inference is the loop | `learn=False` executes S1–S3, S5–S7 and skips S4 (the parameters do not move); the emitted text span re-enters as the next input through `E^{in}_txt` |
| L8 | boundary events | the boundary-out runs once per episode, after the evaluation at `n = N`, and never inside a step (a sentinel raises) |
| L9 | provenance | the checkpoint carries the code digest and configuration; loading a checkpoint whose digest differs raises unless allowed |
| L10 | G-STAT enforcement | a planted non-stationary tail doubles `N` once and flags a second failure (A5) |
| L11 | the genome reset | after a reset, `g = g_0`, the gains are those of `g_0`, the consolidation buffers are empty, the pattern is frozen, and `W`, `ψ`, `ζ` are unchanged |
| L12 | compute accounting | `F_episode` counts the training rows and the rollouts' forwards; the probes' FLOPs are added to `F_cum` and reported separately; `F_episode` is recorded before any `N` doubling |
