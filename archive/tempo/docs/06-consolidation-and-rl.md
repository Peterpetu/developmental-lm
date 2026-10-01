# 06 — Consolidation: the reverse-time process and the reinforcement learning that trains it

**Status: normative.** The window summaries (§6.1), the consolidator (§6.2), the reward
(§6.3), the genome step (§6.4), estimator 1 (§6.5), estimator 2 (§6.6), the short-horizon
defences (§6.7), the evolution-strategies alternative (§6.8), salience and rehearsal (§6.9),
the parameter count and cost (§6.10), and the acceptance tests (§6.11). Symbols: 01, 03, 04.
Implementation: `tempo/consolidate.py`; the orchestration (which candidate is kept, the order
at the boundary) is 08 §8.2 and `tempo/life.py`.

## 6.1 The unit of the reverse read: the window summary

**[NORMATIVE]** The consolidator reads the episode at the resolution of the *window* (`T_s`
spans; one row of a rollout group for an environment stream), not the span: an episode has
`N · B = 32,000` windows and `4.1M` spans, and per-span residuals would cost `4 GB` of storage
where per-window summaries cost `50 MB`. Every window `w` processed in episode `e` leaves

```
   r_w = W_r · [ h̄_w ; ℓ_w ; μ̄_w ; n_w / N ]     ∈ R^{d_c},   d_c = 256,   W_r: (d_c × (d + 1 + d_x + 1))
   h̄_w = mean_{t ∈ w} h^{(L)}_t ∈ R^d   (the final residual, averaged over the window's spans, stored in bf16)
   ℓ_w = the window's mean per-token (text) or per-span (other) loss (02 §2.4, in nats);
         for a rollout row with no likelihood target, −return / R_scale
   μ̄_w = mean_{t ∈ w} μ_{m(t)} ∈ R^{d_x}       n_w = the step at which the window was processed             (6.1)
```

together with the window's identity (a tuple that re-fetches it exactly, 08 §8.3) and a
**rehearsed flag**: windows drawn from the rehearsal buffer (§6.9) are re-reads of old
material and are excluded from the episode's own per-stream means, which would otherwise mix
progress with repetition. `R_scale` is set once per environment stream, at that stream's first
episode, as the standard deviation of the return over the episode's distinct rollouts, floored
at 1; the summaries already stored in that episode are rescaled by it, so one episode never
mixes two scales.

## 6.2 The consolidator

**[NORMATIVE]** At the end of episode `e` the `N B` summaries are ordered by `(n_w, w)` and
read **in reverse**:

```
   c^{seq} = Mixer_C( r_{NB}, r_{NB−1}, …, r_1 )              4 gated-delta layers (03 §3.3–3.4 at d_c = 256,
                                                              4 heads of 64, FFN 1024, no attention layer,
                                                              no gains, no switches), chunk 64               (6.2)
   R  = W_R · [ c^{seq}_{last} ; mean_w c_w ; g_e ]           ∈ R^{d_c},   W_R: (d_c × (2 d_c + d_g))
                                                              c^{seq}_{last} is the output at the sequence's
                                                              end: window 1 when reversed, window NB forward (6.3)
   μ_ψ = W_μ R,   log σ_ψ = clamp( W_σ R, log 0.02, log 0.2 ),   W_μ, W_σ: (d_g × d_c)
   Δg ~ N( μ_ψ, diag σ_ψ² )                                   the proposal                                  (6.4)
   s_w = σ( w_sᵀ c_w )                                        salience per window ∈ (0, 1),  w_s ∈ R^{d_c}   (6.5)
```

`ψ = {W_r, Mixer_C, W_R, W_μ, W_σ, w_s}`; at birth `W_μ = W_σ = 0` and `w_s = 0`. `c_w` is
always indexed in the episode's original order, whichever direction the mixer read.

**The read-out is the last-reached output, in both directions** (11 §P-C.28). The mixer is
recurrent and its state decays, so its most informative output is the one at the end of the
sequence it consumed. Reversed (the mechanism) that is the *first* window's output, reached
last: the summary is anchored on how the episode ended and unwound toward how it began.
Forward (the K-REV control) it is the *last* window's output. The first form of (6.3) took
`c_1` in both directions, which for the forward read is the output after a single window and
made K-REV a comparison between a full read and almost no read: the control was not the same
mechanism with the order changed. K-REV (10) now tests the direction alone with the same `ψ`,
the same read-out rule and the same cost.

**The proposal's scale** lies in `(0.02, 0.2)` per coordinate, by a *squashing* map, not a
clamp:

```
   log σ_ψ = log 0.02 + ( log 0.2 − log 0.02 ) · sigmoid( W_σ R + b_σ ),   b_σ = 3.892        (6.4b)
```

The upper end is `τ / sqrt(d_g)`: a draw there has typical norm `τ`, exactly the trust region's
radius (6.8), and `b_σ` is set so that birth sits at `σ ≈ 0.191`, just inside it — the policy
proposes a step of the intended size from the first boundary and the projection is almost never
active.

**A clamp killed this head twice** (11 §P-C.28, §P-C.36). The first form clamped `log σ` to
`[−5, 0]`, whose upper bound `σ = 1` makes the birth draw have norm `5 τ`, so every early
proposal was clipped to the trust region's boundary and only its direction survived. The second
form clamped to `[log 0.02, log 0.2]` — and at birth `W_σ = 0`, so the pre-activation is `0`,
which is *above* the upper bound. `clamp` has zero gradient outside its bounds, so `W_σ.grad`
was **exactly zero** from every estimator and `σ` sat at `0.2` for the entire life. The
mechanism the correction was written to restore was dead again, from the opposite direction.
The sigmoid is differentiable everywhere, so the head can move from wherever it starts; `b_σ`
is a parameter of the head and belongs to the head's optimiser group, which is the other half
of the same lesson.

The consolidator's own fast states are reset at every consolidation (it reads one episode, not
a life).

## 6.3 The reward: fitness sets, not the training curve

**[NORMATIVE]** The reward compares the agent *before* and *after* the episode on fixed,
held-out material, not on the windows it happened to train on. For every stream `m` present in
the life there is a **fitness set**, disjoint from the evaluation set that reports the run's
score (10 §10.1): for text, the v3 fitness rows (document residue 0); for the generated
streams, the reserved seeds `10^6 + 1000 i + 500 + j`. The fitness score of stream `m` at a
boundary is measured **from a fresh fast state** and is

```
   f_m = mean over the fitness windows of the per-token (or per-span) loss          (text, patterns, interleave)
   f_m = − mean greedy return / reward over the fitness rollouts                    (task, control, vision, speech)
```

so that lower is better for every stream. The reward-bearing streams are scored by *acting*
(greedy decoding, mean actions), because their loss terms score the agent's own samples and
would move with the sampling temperature rather than with competence.

```
   s_m  = the standard deviation of f_m's own windows at stream m's first evaluation, floored at 0.1
   Ľ_e  = (1 / |M|) Σ_{m ∈ M} f_m / s_m                    M = the streams present, uniform weights            (6.6)
   r_e  = ( Ľ_e − Ľ_{e+1} ) · F_ref / F_{e+1}                                                                (6.7)
```

`F_{e+1}` is the FLOPs episode `e+1` spent including its probes and rollouts (09 §9.1),
`F_ref` the primordium's episode (05 §5.5), so `r_e` is standardised improvement per
primordium-episode of compute: the reward of the proposal `Δg_e` (applied before episode
`e+1`) is what the agent's own next episode gained per unit of compute.

Three corrections are folded into (6.6) (11 §P-C.29). First, the *mixture* weighting of the
first form made the reward controllable by the genome's own mixture role: a proposal could
raise the reward by moving `μ` onto whichever stream had the lowest numerical loss, with no
learning at all. The weights are therefore uniform over the streams present. Second, the
per-stream scale `s_m` is required because the streams' losses are incommensurable — text is
`≈ 2` nats per token, a 64-band Gaussian speech frame is `≈ 60` nats per span, the point-mass
return is `≈ −100` — and an unstandardised mean is a reward for the loudest stream. Third, the
training-curve form (`½ mean + ½ last tenth` of the episode's own windows) rewarded reading
easy windows; the fitness sets are fixed, so the only way to move `Ľ` is to be better at them.

## 6.4 The genome step

**[NORMATIVE]** After the proposal is chosen (§6.5 or §6.6):

```
   g_{e+1} = g_0 + P_{ρ} [ ( 1 − λ_dec ) · ( g_e + Π_τ(Δg) − g_0 ) ],      Π_τ(Δg) = Δg · min( 1, τ / ‖Δg‖ ),
   τ = 0.2 sqrt(d_g) = 4.525,   λ_dec = 0.01,   ρ = 3 sqrt(d_g) = 67.9,   P_ρ(x) = x · min(1, ρ / ‖x‖)   (6.8)
```

Every operation is **about the prior** `g_0` (04 §4.1): the decay pulls toward `g_0`, and the
bound is on `‖g − g_0‖`, the distance the genome has travelled from what it was born with. The
decay alone bounds that distance by `τ (1 − λ_dec) / λ_dec ≈ 448`; the projection to
`3 sqrt(d_g)` is the binding bound. `‖g_{e+1} − g_0‖` is recorded per episode.

The order of the whole boundary — regression, estimator update, proposal, probe, genome step,
`ζ` rule, insulation, switches and births, expression, record — is fixed and specified once, in
08 §8.2 (O1–O10); this document specifies only the pieces.

## 6.5 Estimator 1: on-policy single sample (the production rule)

**[NORMATIVE]** At the boundary of episode `e+1` the reward `r_e` of the *previous* proposal
is known. The order within the boundary is therefore **update, then propose**:

```
   (i)  b_{e+1} = 0.9 b_e + 0.1 r_e,   b_0 = 0
        ψ ← ψ + η_2 ( r_e − b_e ) ∇_ψ log N( Δg_e ; μ_ψ(R_e), σ_ψ(R_e) )        η_2 = 10^{-4} (Adam)      (6.9)
   (ii) R_{e+1} from (6.3) on this episode's summaries;   Δg_{e+1} ~ N( μ_ψ(R_{e+1}), σ_ψ(R_{e+1}) )
```

so that every proposal is a sample of the policy whose gradient will score it: the estimator
is on-policy. The first form drew the proposal first and updated afterwards, which scores a
sample of a policy that no longer exists. `R_e` is recomputed from the stored summaries of
episode `e` (the reverse read is re-run for the gradient; its cost is §6.10).

**No KL term.** The plan's (6.9) carried `− λ_KL ∇_ψ KL(ψ ‖ ψ^{old})`. With a *single*
gradient step per boundary, `ψ^{old}` is `ψ` at the point the gradient is taken, and
`KL(p ‖ p)` is stationary there: the term contributes exactly zero and is not implemented
(11 §P-C.18). What bounds the policy's drift is the single small step, the `log σ` clamp of
(6.4), and the trust region of (6.8). The KL is live only in the multi-step regression of
§6.6, where `ψ^{old}` is genuinely a different policy.

## 6.6 Estimator 2: the null probe, best-of-G, and the elite regression (the screening rule)

**[NORMATIVE]** From the checkpoint at the end of episode `e`, `G = 4` proposals
`Δg_i ~ N(μ_ψ, σ_ψ)` are drawn, and a **null proposal** `Δg_0 = 0` is prepended. Each of the
`G + 1` candidates is applied by (6.8), expressed, and lived for `n_probe = N/10` steps from
the *same* checkpoint on the *same* windows (common random numbers by the order seed); each
probe's score is `Ľ` of (6.6) on the fitness sets and its reward `r_i` is (6.7) with the
probe's own FLOPs. Then

```
   i* = argmax_{i ≥ 1} r_i
   if r_{i*} ≤ r_0 :   Δg = 0 and the life continues from the null probe's end               (the rule declines)
   else            :   Δg = Δg_{i*}, the life continues from probe i*'s end, and
                       D ← (D ∪ {(feat_e, g_e, Δg_{i*})}) truncated to its last 8 pairs
                       ψ ← argmin_ψ Σ_{(f, g, Δg) ∈ D} − log N( Δg ; μ_ψ(W_R[f; g]), σ_ψ(·) ) + λ_KL KL(ψ ‖ ψ^{old})
                       by 200 Adam steps (η_2 = 10^{-3}) on the heads W_R, W_μ, W_σ,
                       preceded by one step on the mixer through this episode's read           (6.10)
```

**The null probe** is the control variate this estimator needs. `r_i` measures the improvement
of a candidate against the *previous* boundary, so it is dominated by the improvement the
agent would have made anyway; without a null the "best of G" is the noisiest draw as often as
the best one, and a rule that never helps still applies a step every episode. With it, the
proposal is applied only when it beats doing nothing on the same windows from the same
checkpoint, and `r_0` is recorded as the episode's baseline.

**The winner's state is kept.** The probe's `n_probe` steps are real training on real windows;
discarding them and re-living the episode would waste `(G+1) n_probe` steps of compute and
break the pairing. The life continues from the kept candidate's post-probe state, and *every*
probe's compute is charged to the episode (09 §9.1), so an arm that probes pays for probing in
every equal-compute comparison (10 §10.5).

**What this update is, and is not.** It is the cross-entropy method's elite regression: sample
`G`, keep the top `1/G`, refit the sampling distribution to the kept elites, with a KL leash to
the pre-update policy. It is **not** ReST-EM (Singh et al. 2024), which refits the *base* model
on fresh samples each round and keeps no buffer; the earlier text called it ReST-EM and that
name is withdrawn (11 §P-C.28). Two honest caveats: (a) the accepted set is bounded at 8 pairs
because each pair stores the read-out features `feat` computed at the time it was accepted, and
those features go stale as the mixer moves — a long buffer regresses the heads onto features
the current mixer no longer produces; (b) the mixer itself gets one gradient step per boundary,
through the current episode's read, since it is the only pair whose summaries still exist.

`n_probe = N / 10` (100 at `N = 1000`; 25 at the screens' `N = 250`), so the overhead
`(G + 1) n_probe / N = 50 %` of the episode's steps is the same at every `N`. It is reported
per episode and charged; estimator 2 is used in the screens that decide K-GENE, K-REV, K-SAL
and K-MORPH (R4–R6, R8), estimator 1 in the long run (R10). The GRPO variant (advantage
`Â_i = (r_i − mean r) / (std r + 10^{-6})`, the clipped ratio of 02 (2.12) at `1 ± 0.2/0.28`)
is R4's arm (g), an ablation.

## 6.7 The short-horizon bias, designed against

**[NORMATIVE]** A rule rewarded for next-episode improvement is greedy in the way the
learned-optimiser literature documents (11 §P-B.3: learning rates driven down by orders of
magnitude to win the short horizon). Three defences:

1. the learning-rate role is bounded to `[¼, 4]` (04 §4.3);
2. the reward is measured on **held-out fitness sets from a fresh state** (6.6), so a rule that
   makes the training curve cheap without making the agent better is not paid, and the reward
   window is the whole next episode;
3. **K-HORIZON** (10 §10.2): the rule wins per-episode rewards (their mean is positive) while
   its score at equal compute is worse than arm (a)'s; the reward window is then lengthened to
   two episodes (`r_e = (Ľ_e − Ľ_{e+2}) F_ref / (F_{e+1} + F_{e+2})`) and the screen repeated;
   if it fires again, K-GENE fires.

## 6.8 The evolution-strategies alternative (arm (f))

**[NORMATIVE]** The learned proposal is compared with a policy-free rule at equal probe
compute: mirrored evolution strategies on `g` directly (Salimans et al. 2017; the local
precedent is `soma/germline/es.py::es_step`). `G/2 = 2` mirrored pairs `g_e ± σ_es ε_i`,
`ε_i ~ N(0, I)`, `σ_es = τ / sqrt(d_g) = 0.2`, each probed for `n_probe` steps under CRN, and

```
   ĝ = (1 / (2 n σ_es)) Σ_i ( r_i^+ − r_i^− ) ε_i ,        Δg = τ · ĝ / ‖ĝ‖   (zero when every pair ties)  (6.8b)
```

The estimator gives the *direction*; the step has the explicit length `τ`, so the ES arm and
the proposal head move the genome the same distance per episode and the comparison is of
directions, not of step sizes. (The first form divided by the standard deviation of the
rewards, which makes the step length an uncontrolled function of the reward scale.) If ES on
`g` matches the trained `ψ` (R4 arm (f) against TEMPO), the consolidator's proposal head is not
load-bearing and is removed; the reverse read then serves salience only (recorded as a partial
K-GENE).

## 6.9 Salience and rehearsal

**[NORMATIVE]** The buffer holds `n_reh = 0.10 · N · B = 3,200` windows of episode `e`, the
top by `s_w`; the first `p_reh = 5 %` of episode `e+1`'s steps draw their windows from it with
probability `∝ s_w`, without replacement within a step, re-fetched by identity. The buffer is
**twice** the size the rehearsal phase consumes, so that the `∝ s_w` draw actually selects: at
`n_reh = p_reh N B` the phase would consume the whole buffer whatever the weights, and the
salience head could not affect what is rehearsed (11 §P-C.28). Under the uniform arm (K-SAL's
control) the buffer is a uniform sample of the episode's windows and the draw is uniform; the
same path is taken, with the reason recorded, if `s_w` is degenerate (all zero or non-finite).

The salience head is fit at every boundary by regression on the *cached* window outputs of the
previous episode's read: a probe set `P_e` of 512 windows of episode `e` (uniform) is re-scored
without gradient at the end of episode `e+1`, the target is the measured improvement
`Δℓ_w = ℓ_w^{(e)} − ℓ_w^{(e+1, end)}`, and

```
   w_s ← argmin_{w_s}  Σ_{w ∈ P_e} ( σ(w_sᵀ c_w) − σ(Δℓ_w / σ_ℓ) )²          200 Adam steps, η = 10^{-2}   (6.11)
```

with `σ_ℓ` the standard deviation of `Δℓ_w` over the set (so the target is a rank-like
transform in `(0, 1)`) and `c_w` the outputs cached when episode `e`'s read was taken. **Only
`w_s` moves.** The first form let this objective train the mixer too, which moves the policy
`ψ` between the moment a proposal is drawn and the moment its gradient is taken — an
off-policy update introduced by a term that has nothing to do with the proposal. The mixer is
moved only by the estimators of §6.5–6.6.

Salience is therefore a prediction of "what the next episode will fix", and rehearsal spends
5 % of the episode on the windows predicted to improve most; **K-SAL** (10 §10.2) fires if
salience-ranked rehearsal does not beat uniform rehearsal of the same size, in which case the
buffer is drawn uniformly (the head stays as a recorded diagnostic).

## 6.10 Parameter count and cost

**[NORMATIVE]** `|ψ|`: `W_r` `256 · 770 = 197,120`; `Mixer_C` four layers of `(5 · 256 · 256
+ 8 · 256 + 3 · 4 · 256 + 3 · 256 · 1024 + 4 · 256)` `≈ 1.12M` each; `W_R` `256 · 1024 =
262,144`; `W_μ, W_σ` `2 · 512 · 256`; `w_s` `256`: `≈ 5.2M` (formula value; asserted by a
test). Cost per boundary: the reverse read over `32,000` summaries through `5.2M` parameters,
`< 0.1 %` of an episode's FLOPs; the fitness evaluations of (6.6); estimator 2 adds
`(G + 1) n_probe / N = 50 %` of the episode's steps, all charged.

## 6.11 Acceptance tests

**[NORMATIVE]** `tempo/consolidate.py` is accepted only when all of the following hold.
Implemented in `tests_tempo/unit/test_consolidate.py` (C1–C7, C9–C11) and `test_life.py`
(C8, C12).

| # | property | test |
|---|---|---|
| C1 | summaries | every window of a 3-step CPU episode leaves one `r_w` with the shape of (6.1); `ℓ_w` equals the window's loss recomputed from the model; rollout rows carry `−return / R_scale`; rehearsed windows are excluded from the per-stream means; a per-stream rescale moves only that stream's entries |
| C2 | reverse order and the read-out | the mixer sees the summaries in the order `(n_w, w)` descending (a distinct marker per window read back from the mixer's input); the read-out equals the output at the end of the consumed sequence in *both* directions (`c_1` reversed, `c_{NB}` forward) and `c_w` is returned in the episode's order |
| C3 | the proposal at birth | with `W_μ = W_σ = 0`, `μ_ψ = 0` and `σ_ψ = 0.2` for every `R`; over 10,000 draws the sample mean and variance match to 3 standard errors and the mean draw norm is `τ` to 10 %; a large negative `W_σ R` clamps at `σ = 0.02` |
| C4 | trust region, decay and bound about the prior | `‖Π_τ(Δg)‖ ≤ τ` with equality when `‖Δg‖ > τ`; (6.8) with `Δg = 0` maps `g` to `g_0 + (1 − λ_dec)(g − g_0)`; `‖g − g_0‖ ≤ 3 sqrt(d_g)` after the projection; with `prior = None` the same holds about the origin |
| C5 | reward | on planted values (`Ľ_e = 2.0`, `Ľ_{e+1} = 1.9`, `F_{e+1} = 2 F_ref`) `r_e = 0.05` exactly; `Ľ` is the uniform mean of `f_m / s_m` over the streams present; an empty stream set and a zero `F` raise |
| C6 | estimator 1 | the gradient of (6.9) equals `torch.distributions.Normal`'s log-prob gradient times `(r − b)` to `1e-6`; the EMA baseline follows `0.9/0.1`; `kl_diag` equals `torch.distributions.kl_divergence` to `1e-5` (it is used by §6.6) |
| C7 | estimator 2's elite regression | with planted rewards the argmax is selected; after (6.10) the log-likelihood of the accepted set increases and the policy's mean moves toward the accepted proposal; the accepted set is bounded at 8; the GRPO variant raises the winner's log-probability most and declines a tied group |
| C8 | CRN and the null probe | the `G + 1` probes of estimator 2 consume identical window sequences (asserted by the fetched identities); with the null probe best, `Δg = 0` is applied and the record says the rule declined; the kept probe's state is the one the life continues from |
| C9 | null consolidation | with `Δg = 0`, the `ζ` rule off, no births and the regression skipped, `Ξ` is unchanged bit for bit (G-IDENT) |
| C10 | salience and rehearsal | on planted `Δℓ_w` the head reaches `R² > 0.9` in 200 steps and **the mixer's parameters do not move**; the buffer holds the top `n_reh` by `s_w`; the draw is `∝ s_w` (chi-square on 10,000 draws at `α = 0.01`); the uniform arm and a degenerate `s_w` both give a uniform buffer with the mode recorded |
| C11 | ES arm | (6.8b) on a quadratic `r(g) = −‖g − g*‖²` moves `g` toward `g*` in every one of 8 steps, each of norm exactly `τ`; a tied group gives the zero step |
| C12 | once per episode | the reverse read, the proposal and the `ψ` updates are called once per boundary and never in the step loop (G-NOFREE-INVERSE sentinel) |
