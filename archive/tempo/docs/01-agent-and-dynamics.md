# 01 — The agent and its dynamics

**Status: normative.** Defines the agent as a state and a map (§1.1), the three timescales and
their update rules (§1.2), the timescale-separation heuristic, what it does *not* claim, and
the two quantities measured in its place (§1.3), the empirical premise G-STAT (§1.4), the life
in one figure (§1.5), and the acceptance tests (§1.6).
Notation: vectors are columns, `⊙` the Hadamard product, `σ` the logistic function,
`RMS(x) = sqrt(mean_i x_i² + ε)`, `‖·‖` the Euclidean norm, shapes `(rows × cols)`.

## 1.1 Definition 1 (agent)

**[NORMATIVE]** TEMPO is the tuple `(E, F, D, C, Φ)` and the state

```
   Ξ_{e,t} = ( S_{e,t},  W_e,  g_e,  ζ_e )
             fast state · slow weights · genome · homeostatic log-gains
```

indexed by episode `e ∈ ℕ` and span `t ∈ ℕ` within the episode. The maps are

```
   perception   x_t   = E_{m(t)}( raw_t ) + μ_{m(t)} + τ(t)                       ∈ R^{d_x}      (1.1)
   output       ŷ_{t+1} = F( x_{≤t} ; S_{e,t}, W_e, Φ(g_e), ζ_e )                  ∈ R^{d_x}      (1.2)
   emission     raw^{out}_{t+1} ~ D_{m'}( · | ŷ_{t+1} )                                           (1.3)
   closure      raw_{t+1} includes raw^{out}_{t+1} perceived through E_{m'}  (the egocentric loop) (1.4)
```

where `m(t)` is the modality of span `t`, `μ_m ∈ R^{d_x}` its learned embedding, `τ(t)` the
time code (02 §2.8), `E_m` and `D_m` the modality encoder and decoder (02), `F` the backbone
(03) with the fast state `S`, the weights `W`, the genome expressed into gains by `Φ` (04) and
the homeostatic gains `ζ` (07), and `C` the consolidator (06).

There is no train/deploy distinction. The life (08) is the trajectory of `Ξ` under the three
update rules of §1.2, applied by one code path (`tempo/life.py`) whether the raw stream is a
corpus, a conversation, a video, a control loop or a blend of them; deployment is a life with
`η_1` possibly zero.

## 1.2 The three timescales

**[NORMATIVE]**

| timescale | state | update rule | rate | where |
|---|---|---|---|---|
| τ0, per span | `S_t^{(l,j)} ∈ R^{d_v × d_k}` of every delta head; the attention KV cache | `S_t = S_{t−1} α_t (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ` (03 §3.4): one least-squares step on `(k_t, v_t)` | every span | `tempo/backbone.py` |
| τ1, per step | `W` (backbone matrices and vectors), `u` (the episode's differentiable gain pre-activations, 04 §4.5), `ψ_s` (salience head), `E^{in}_m, μ_m, D_m` (trainable interface parts) | Muon on matrices, AdamW on vectors: one step per `B` windows on the span losses (02 §2.4) | every step | `tempo/life.py` |
| τ2, per episode | `g` (genome), `θ` (expression map, refit by regression to the episode's gains, 04 §4.5), `ζ` (homeostatic gains), `ψ` (proposal head), morphology switches | (06 §6.3–6.5): one consolidation per episode of `N` steps; (04 §4.5); (07 §7.3); (05 §5.3) | every episode | `tempo/consolidate.py` |

The per-span rule (τ0) is the exact least-squares step of the gated delta rule (SOMA
(5.4)–(5.16); `gnomon/pheno/mixer_chunk.py`, verified to 1e-10 against the float64 oracle).
The per-step rule (τ1) is the ANLAGE recipe (`anlage/recipe.py`): Muon (Newton–Schulz 5,
`0.2 sqrt(max(m, n))` scaling, weight decay 0.1) for 2-D parameters, AdamW (`β = (0.9, 0.95)`)
for vectors and the interface, gradient clipping at 1.0, bf16 autocast with fp32 optimiser
state, micro-batches by the card's rule. The per-episode rule (τ2) is the subject of 06.

**Step sizes.** `η_1` follows the warm-up / constant / decay-to-zero schedule *within each
episode* (`p_warm = 0.02`, `p_dec = 0.8`, batch warm-up over the first 10 %), so an episode
ends at zero learning rate and the optimiser is rebuilt for the next (ANLAGE 06 §6.2 step 5,
measured reason: the state left at zero learning rate is that of a stopped optimiser). The
genome moves at most `τ = 0.2 sqrt(d_g)` per episode (06 §6.4). Locally `N = 1000` steps,
`B = 32` windows of `T = 512` tokens (`T_s = 128` spans), so an episode is `16.4M` tokens.

## 1.3 Heuristic 1 (timescale separation) - what is claimed and what is not

**Setting.** Within episode `e` the fast variable is `W` (with `u`, the decoder, the interface
and the salience head), moved by `N` steps of a stochastic gradient method with step sizes
`eta_1(n)`, `n = 1..N`, on the span loss `L(W; g, zeta, data)`. Between episodes the **slow
variables move once each**: the genome `g` by a step of norm at most `tau` (06 §6.4), the
homeostatic gains `zeta` by (07 §7.3), the expression map `theta` by the regression of 04
§4.5, the consolidator `psi` by one estimator update (06 §6.5-6.6), the morphology switches by
05 §5.2, and the insulation scales `kappa_m` by 02 §2.4.

**[NORMATIVE] Statement (heuristic, not a theorem).** TEMPO is *designed so that* the slow
variables move slowly enough that the consolidation rule's reward is a stable function of
`g`: the fast system sees a slow variable that is nearly constant within an episode, and the
slow sequence approximately tracks

```
   g_{e+1} ~ g_e + E[ Pi_tau(Delta g)  |  W at the end of an episode lived under g_e ]              (1.5)
```

i.e. the rule is evaluated at the weights the current genome *leads to*, not at a transient.
That is the design intent, and the reason the reward of 06 §6.3 is measured after a whole
episode rather than after a few steps. It is **not** a convergence theorem, and this document
does not claim one.

**Why not.** The two-timescale theorem of Borkar (2008, ch. 6, Thm 2) requires, for coupled
recursions with step-size ratio `b(n)/a(n) -> 0`, that the fast ODE `x' = h(x, y)` have a
*globally asymptotically stable* equilibrium `lambda(y)` for each slow value `y`, together
with martingale-difference noise of bounded second moment. Three of its hypotheses fail here
and cannot be repaired by choosing constants:

1. **No global attractor.** The fast loss is a non-convex deep network's; `lambda(g, zeta)` is
   at best a set of local attractors, and which one is reached depends on the trajectory, so
   the averaged slow recursion is not a function of `g` alone.
2. **The slow variables are not one variable.** `psi` (a 3.2M-parameter network trained by
   policy gradient), `theta` (refit by regression to a moving target), and the *discrete*
   switch set of 05 all move at tau2; the switches make the slow dynamics discontinuous,
   which the theorem's ODE limit does not admit.
3. **The step sizes do not vanish.** `eta_1` follows a within-episode schedule that restarts
   every episode, so `sum eta_1 = inf`, `sum eta_1^2 < inf` are not satisfied along the life;
   and the slow step has a *fixed* size `tau`, not a vanishing one.

**What is measured instead.** Two recorded quantities stand in for the theorem, and both are
checked in every run rather than assumed:

* **A8, the step-size ratio in gain space.** The genome's move and the fast system's move are
  compared in the *same* space, the expressed gain pre-activations `u` (04 §4.5), where both
  act:

  ```
     C2 = || u_{e+1}^{start} - u_e^{end} ||  /  || u_e^{end} - u_e^{start} ||                       (1.6)
  ```

  the numerator being what the genome's step changes at the boundary, the denominator what
  tau1 changed during the episode. `C2 << 1` is the operational form of "the slow variable is
  slow"; it is recorded per episode and flagged when it exceeds `1`. Comparing `tau` with
  `N eta_1` (a norm in genome space against a norm in weight space) compares incommensurable
  quantities and was dropped (11 §P-C.27).
* **G-STAT (§1.4)**, the transient test on the episode's own losses.

**What Heuristic 1 does not say.** It does not say the RL rule finds a good `g`; it says the
rule's reward is intended to be a stable function of `g`, and gives the two measurements that
say whether it was. Whether the rule improves learning is K-GENE (10 §10.2), and the verdict
rests on the paired design of 10 §10.1, not on this section.

## 1.4 G-STAT: the empirical premise

**[NORMATIVE]** Before any consolidation is trained, and in every recorded life, the episode's
transient should be over when the slow step is taken. On the per-step training losses `l_n` of
the episode **after the rehearsal phase** (the first `p_reh N` steps re-read old windows, 06
§6.9, and their losses are not on the episode's own curve), with `k = max(1, round(N/10))`
and `N` the number of remaining steps,

```
   D_first = mean(l_{0..k-1}) - mean(l_{k..2k-1}),   D_last = mean(l_{N-2k..N-k-1}) - mean(l_{N-k..N-1})
   se_last = std(l_{N-2k..N-1}) sqrt(2/k)                                                            (1.7)
   G-STAT holds when   ( D_last < 0.25 max(D_first, 0)  and  D_last > -2 max(sigma_e, se_last) )
                  or   |D_last| < 2 max(sigma_e, se_last)                                            (1.8)
```

i.e. the loss's rate of decrease over the last fifth of the episode is below a quarter of the
rate over the first fifth *and the loss is not rising*, or the last fifth is flat to twice the
larger of the evaluation noise `sigma_e` and the statistic's own standard error. The test is
**two-sided** (11 §P-C.22): a loss that is *increasing* at the end of an episode passes the
one-sided form trivially while being the clearest possible violation of "the transient is
over". `se_last` enters because `D_last` is a difference of two means of `k` noisy steps and a
bar of `2 sigma_e` alone can be tighter than the statistic's own resolution at small `N`; it
is computed only when the tail has at least eight steps.

The ratio `D_last / D_first` is recorded at every boundary. If (1.8) does not hold, `N` is
doubled once per life (recorded); a further failure is flagged in the record and the life
continues. This is the check that the fast system has passed its transient when the slow step
is taken. It cannot hold exactly in this regime: the first form of the rule, "the last 20 %
flat to `2 sigma_e`", failed by `0.18` nats on a plain life, because the per-episode
decay-to-zero schedule makes the final tenth the steepest part of the episode, and a lifelong
learner keeps improving on new data at every budget (R1: the ratio is `0.20` at 25M tokens,
`0.10` on the constant-learning-rate plateau; 11 §P-C.22). What carries the verdicts is not
stationarity but the pairing: every arm of a contrast lives the same windows in the same order
(10 §10.1), so the continued fast improvement is a common term that cancels in the paired
difference.

## 1.5 The life, in one figure

```
   birth ──► [ episode e: N steps at τ1 over spans (τ0 inside every step) ] ──► consolidate (τ2) ──► episode e+1 ──► …
                      ▲                                                          │
                      │              rehearsal buffer (salience), gains Φ(g), switches, ζ, κ_m
                      └──────────────────────────────────────────────────────────┘
```

The full loop with every ordering is 08 §8.2; inference is the same loop with `η_1 = 0` (08
§8.3).

## 1.6 Acceptance tests

**[NORMATIVE]** `tempo/life.py` and `tempo/backbone.py` are accepted only when all of the
following hold.

| # | property | test |
|---|---|---|
| A1 | one code path | the same `Life.run_episode` runs with `learn=True` (training) and `learn=False` (inference, `η_1 = 0`); the two differ only in whether the optimiser step is taken, and `Life.step_inference` is one span of that path |
| A2 | rates are separable | with consolidation disabled and switches frozen, the life reduces bit for bit to the ANLAGE lifetime on the same windows and seed (up to the span interface) |
| A3 | the fast state persists | the delta heads' `S` after a window equals the `initial_state` of the next window's call; resetting `S` only at declared boundaries is observed (a planted boundary resets it, a missing boundary does not) |
| A4 | no gradient reaches `g` or `ζ` from τ1 | `g.requires_grad` is False through a full step; a finite-difference perturbation of `g` changes the loss (the expression is live) while `g.grad` is None after backward |
| A5 | G-STAT is computed and enforced | on a planted non-stationary episode (a loss that keeps falling) the life doubles `N` once and flags a second failure; a *rising* end fails too (the two-sided form); on a stationary one it proceeds |
| A6 | determinism | two lives from one `seed` on CPU are bit-identical over two episodes including a consolidation (state, weights, `g`, `ζ`, switches) |
| A7 | the state serialises | `Ξ` and the optimiser state round-trip through the checkpoint of 08 §8.4 bit for bit and the resumed life continues identically (08 L-rows own the details) |
| A8 | step-size ordering in gain space | the recorded life reports `C2` of (1.6) per episode, the genome's boundary move over the episode's τ1 move in the *same* space `u`; a run with `C2 > 1` is flagged |
