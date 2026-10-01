# 04 — The genome and its expression

**Status: normative.** The genome `g` (§4.1), the role registry with counts (§4.2), the
low-rank expression map with its ranges (§4.3), how each role enters the computation (§4.4),
the episode cycle of the gains, which is the PNAS loop in gain space (§4.5), the zero per-token
cost (§4.6), the no-gradient rule (§4.7), the SVF ablation (§4.8), the control arms of K-GENE
(§4.9), and the acceptance tests (§4.10). Symbols: 01 §1.1, 03.

## 4.1 The genome

**[NORMATIVE]**

```
   g ∈ R^{d_g},   d_g = 512,   prior g_0 ~ N(0, I) from the life's model seed (recorded),
   ‖g − g_0‖ ≤ 3 sqrt(d_g) always (06 §6.4)
```

The prior is not zero: at `g = 0` the expression (4.1a) is identically zero whatever `B_ρ`, so
the regression of §4.5 could fit nothing at birth and the decay of 06 (6.8) would pull the
genome toward a degenerate point (11 §P-C.19). The identity at birth is carried by `B_ρ = 0`
on the differentiable rows and by the *differential frame* on the programme rows (§4.3), not
by the prior. Every operation of 06 (6.8) — the decay and the bound — is about `g_0`, and what
is bounded and recorded is the distance travelled, `‖g − g_0‖`, not `‖g‖`: a bound on `‖g‖`
would depend on where the seed happened to put the prior, which is not a property of the
agent (11 §P-C.25).

`d_g` does not depend on the configuration (03 §3.1), the modality (02 §2.9) or the active
shape (05): the same 512 numbers are the agent's slow channel at every size and in every
domain, which is what B5's "translating between domains" and the user's "scale free" require
of it. `g` is a τ2 state: it moves only by consolidation (06) and by the decay of 06 §6.4.

## 4.2 The role registry

**[NORMATIVE]** A *role* is one named vector of gains with a fixed size, a range `c_ρ`, a
kind, and a rule for entering the computation (§4.4). The registry at the TS-adult:

| kind | roles | size per role | roles at TS-adult | gains at TS-adult |
|---|---|---|---|---|
| matrix (`a_in ; a_out`) | one per weight matrix `W: (n_out × n_in)` of 03 §3.9: delta layer `W_q, W_k, W_v, W_o, W_og, W_ab, W_gate, W_up, W_down` (9), attention layer `W_q, W_k, W_v, W_o, W_og, W_gate, W_up, W_down` (8), `W_in`, `W_out` | `n_in + n_out` | `12·9 + 4·8 + 2 = 142` | `12·13,328 + 4·12,032 + 1,536 = 209,600` |
| norm | `ρ^{mix}_l, ρ^{ffn}_l, n^{mix}_l, n^{ffn}_l` per layer, `ρ_fin` | `d` | `4·16 + 1 = 65` | `33,280` |
| offset | `a_0` (decay) and `g_0` (write) over all delta heads; `g_og` over all layers | `h·L_delta`, `h·L_delta`, `L` | `3` | `96 + 96 + 16 = 208` |
| learning rate | `log λ_grp` over the optimiser groups (matrices, interface embeddings, vectors, heads) | `4` | `1` | `4` |
| morphology | pre-activations of the quarter-stacks `{4..7}, {12..15}`; of the head group `{4..7}` per layer; of the slice group `{4..7}` per layer (05 §5.1) | `2`, `16`, `16` | `3` | `34` |
| mixture | `μ` over the seven streams of 02 §2.9 (logits) | `7` | `1` | `7` |
| **total** | | | **`215`** | **`n_γ = 243,133`** |

The depthwise convolution taps `W_{cq,ck,cv}` and the biases `b_in` carry no role. The Pile
registry follows the same rules from its shapes (03 §3.1). The integers above are formula
values; `test_spec_numbers` (G-SPECNUM) asserts the registry built from the module tree.

## 4.3 The expression map

**[NORMATIVE]** For every role `ρ` with size `n_ρ`:

```
   differentiable kinds:   u_ρ = B_ρ · tanh( A_ρ g )                                                (4.1a)
   programme kinds:        u_ρ = B_ρ^{prog} · ( tanh( A_ρ g ) − tanh( A_ρ g_0 ) )                   (4.1b)
   both:                   γ_ρ = c_ρ · tanh( u_ρ )   ∈ (−c_ρ, c_ρ)^{n_ρ}                            (4.2)
                           A_ρ: (r × d_g),  B_ρ, B_ρ^{prog}: (n_ρ × r),  r = 32
```

`A_ρ ~ N(0, 1/(4 d_g))`, so `‖A_ρ g‖` is of order `sqrt(r)/2` at the prior and `tanh` is not
saturated at birth: at the earlier scale `1/d_g` the pre-activations of a genome at the bound
saturated the nonlinearity and the map's Jacobian in `g` vanished, which no amount of
regression could repair.

The two frames are the correction of 11 §P-C.25. The **differentiable** rows (matrix, norm,
offset) start at `B_ρ = 0`, so `u = γ = 0` and every gain is the identity at birth (G-IDENT);
`B_ρ` is trainable and the regression of §4.5 gives it content from the first boundary on.
The **programme** rows (learning rate, morphology, mixture) are consumed by the optimiser, the
switches and the data draw and are therefore *never* differentiable in the span loss: nothing
would ever move their `B_ρ` away from zero, so under (4.1a) they would be identically zero for
the whole life and the genome could not act on them at all — the mixture role, the morphology
role and the learning-rate role would be dead by construction. They use instead the fixed
random frame `B_ρ^{prog} ~ N(0, 0.2²)` (a buffer, never trained) applied to the *displacement*
of `tanh(A_ρ g)` from its value at the prior: at `g = g_0` the expression is exactly zero (the
identity at birth is preserved), and any move of the genome away from `g_0` moves the
programme gains along a fixed random projection of that move. The genome's action on the
programme kinds is therefore a genuine function of `g`, learned by the consolidation rule
through the reward rather than by a gradient.

The ranges:

| kind | `c_ρ` | effect bound |
|---|---|---|
| matrix | `0.5` | each row/column of `W_eff` scaled within `[0.5, 1.5]` |
| norm | `0.2` | `ρ, n` scaled within `[0.8, 1.2]` |
| offset | `2.0` | `a_0, g_0, g_og` shifted within `±2` (a decay of `σ(4 ± 2)`, a write of `2σ(2 ± 2)`) |
| learning rate | `log 4` | `η_grp ∈ [η_1/4, 4 η_1]` (the short-horizon defence of 06 §6.7) |
| morphology | `8` | `m = σ(a_b) ∈ [3.4·10^{-4}, 0.9997]`; `ε_skip = 0.02` is inside the range |
| mixture | `4` | a stream's share within `[e^{-8}, e^{8}]` of another's before the floor of 02 §2.9 |

`θ = {A_ρ, B_ρ}_ρ` has `Σ_ρ (r d_g + n_ρ r) = 215 · 16,384 + 32 · 243,133 = 11,302,816`
entries at the TS-adult (asserted by G-SPECNUM), of which the `B^{prog}` rows are fixed
buffers and the rest trainable. `θ` is a τ2 quantity (§4.5).

## 4.4 How each kind enters

**[NORMATIVE]**

| kind | enters as | where |
|---|---|---|
| matrix | `W_eff = diag(1 + a_out) · W · diag(1 + a_in)` | 03 (3.12), in-graph at every step |
| norm | `ρ_l = ρ_{W,l} ⊙ (1 + γ_{ρ,l})`; `n_l = n_{W,l} ⊙ (1 + γ_{n,l}) · exp(ζ_l)` | 03 (3.3) |
| offset | additive in 03 (3.5) and (3.8): `a_{0,j} = 4 + γ_{a_0,j}`, `g_{0,j} = 2 + γ_{g_0,j}`, `g_{og,l} = 2 + γ_{og,l}` | 03 |
| learning rate | `η_grp = η_1(n) · exp(λ_grp)` | `anlage/recipe.py::build_optimizers` groups |
| morphology | `m_b = σ(a_b)`, `a_b = −4 + γ_b` of the block's group (the baseline `−4` keeps every switchable group skipped at birth, `σ(−4) = 0.018 < ε_skip`; 11 §P-C.17) | 05 §5.1 |
| mixture | `μ = floor_{0.02}( softmax(γ_μ) )` over the seven streams | 02 §2.9, the episode's window draw |

## 4.5 The episode cycle of the gains (the PNAS loop, in gain space)

**[NORMATIVE]** The roles split into the *differentiable* kinds (matrix, norm, offset), whose
gains the span loss is differentiable in, and the *programme* kinds (learning rate,
morphology, mixture), which the optimiser, the switches and the data draw consume. Within an
episode:

```
   at the start of episode e:
       u_ρ ← B_ρ tanh(A_ρ g_e)              for every role                                 (4.3)   expression, once
       programme kinds:  fixed for the episode at γ_ρ = c_ρ tanh(u_ρ)
       differentiable kinds:  u_ρ becomes a free τ1 parameter (AdamW, lr η_1, no weight decay)
   during the episode (τ1):
       the loss is differentiable in γ_ρ = c_ρ tanh(u_ρ) through 03 (3.12), (3.3), (3.5), (3.8); u_ρ moves with W
   at the end of episode e (τ2), before the genome step:
       H ← the last 16 pairs (g_j, u_j^{end}), j ≤ e                                  the regression history
       θ ← argmin_θ  Σ_{(g, u) ∈ H} Σ_ρ ‖ B_ρ tanh(A_ρ g) − u_ρ ‖²                          (4.4)   regression
       by 500 Adam steps (lr 3·10^{-3}) on the differentiable rows from the current θ;
       record R_e = the relative residual on the *newest* pair
   then the genome step g_{e+1} of 06 (6.8), the criticality rule (07);
   at the start of episode e+1:  (4.3) with g_{e+1}   — the gains are regenerated from the genome (ε = 0)

The regression is over a **history of pairs**, not the current pair alone (11 §P-C.25). With
one pair `(g_e, u_e)` the problem is trivially solvable to machine precision — `B_ρ` has
`n_ρ × r` free entries and one `r`-vector of constraints per row — so a single-pair residual
of `10^{-6}` says nothing about whether the map *generalises* to a different genome, which is
exactly what the next episode's expression `Φ_θ(g_{e+1})` asks of it. Sixteen pairs make (4.4)
overdetermined in the direction that matters, and `R_e` becomes an informative number: it is
the residual on the newest pair under a map that must also fit the previous fifteen. The
acceptance test E6 asserts both facts (a single pair is fittable to `10^{-6}`; a history of
sixteen is not, and its residual is the reported one).
```

This is the loop of the PNAS paper (00 §0.2: train, fit the genome to the phenotype,
regenerate) applied to the object that the record says a compact genome can carry (vector
roles fit to `< 1 %`, 00 §0.1), run at every episode boundary (B7: hundreds of times per life),
with full regeneration (`ε = 0`) because the fit is near-exact; the fast weights `W` are never
regenerated (A8). The regression residual `R_e` is a recorded quantity; **G-REGRESS** fires
if `R_e > 0.01` at any boundary (the map cannot carry the gains; `r` is doubled and the run
restarted, recorded). The programme kinds' targets in (4.4) are their unchanged values.

**What the genome step does.** `u_{e+1} = B tanh(A(g_e + Π_τ Δg))`: the proposal moves the
*starting* gains of the next episode along the directions the trained map exposes, and the
reward (06 §6.3) is what that start does to the next episode's learning.

**A null direction to be aware of.** Under Peri-LN (03 (3.3)) a *uniform* rescaling of a
norm gain `ρ_l` is absorbed by the RMS normalisation that follows it, so that direction of
gain space has no effect on the function. It is not removed — it costs nothing and the
regression handles it — but it means a test that perturbs the gains uniformly and expects the
output to move is testing nothing; the acceptance tests perturb non-uniformly (03 P6). Without a genome step
(`Δg = 0`, the K-GENE arm (d)) the cycle regenerates the same gains it fit: the loop's only
effect is the compression itself, which is the PNAS paper's simplicity-bias claim in gain
space, and which the arms of §4.9 separate from the RL rule's effect.

## 4.6 Zero per-token cost

**[NORMATIVE]** The per-step path contains, of the genome machinery, only the parameter
tensors `u_ρ` (differentiable kinds) and the elementwise fold 03 (3.12): `Σ_matrices
(n_out + n_in + n_out n_in)` multiplications per step, `< 0.1 %` of the step's matrix
multiply-adds. The expression (4.3), the regression (4.4), the consolidator (06) and the
criticality probe (07) execute only in the boundary path. The sentinel test G-NOFREE-INVERSE
instruments the expression and regression entry points and asserts they are called exactly
once per episode and never between two consecutive optimiser steps.

## 4.7 No gradient reaches the genome

**[NORMATIVE]** `g.requires_grad` is `False`; the expression (4.3) is computed under
`torch.no_grad` and its result assigned to the leaf `u_ρ`. A perturbation of `g` changes the
loss of the next episode (the map is live), and no per-step loss can move `g` (A4 in 01). This
is B8's "the genome is not continuous" and the structural reason GNOMON's runaway cannot
recur: nothing pushes `g` at τ1.

## 4.8 The SVF ablation of the matrix roles

**[NORMATIVE]** The registered alternative to the diagonal unit gains of 03 (3.12) is
singular-direction gains (Transformer², 11 §P-B.1): `W_eff = U diag(σ ⊙ (1 + a_svf)) Vᵀ` from
one SVD per matrix per episode, `a_svf ∈ R^{min(n_in, n_out)}`, `c = 0.5`. It has fewer gains
(`min` instead of `n_in + n_out`) and costs one SVD per matrix per episode (`≈ 0.5 s` for the
TS-adult on this card). It is the ablation of R4 arm (e), not the default; the default is the
form with no per-episode decomposition.

## 4.9 The control arms of K-GENE

**[NORMATIVE]** R4 (10) runs, from the same birth and windows (CRN):

| arm | genome cycle | isolates |
|---|---|---|
| (a) `γ ≡ 0` | no gains, no regression, no genome | the plain backbone |
| (b) homeostatic only | `ζ` rule of 07 on, `γ ≡ 0` | criticality maintenance without a genome |
| (c) random `Δg` | the full cycle with `Δg` drawn isotropically and scaled to norm exactly `τ` | the RL rule against a random walk of the same size. The arm keeps the full consolidation cost (its `Δg` is free, but the reverse read and the fitness evaluations still run) so that the equal-compute comparison is of directions, not of overheads |
| (d) free gains | differentiable roles free at τ1, carried over between episodes, no regression, no genome | the value of free gains without compression |
| (e) SVF | as TEMPO with §4.8 | the matrix-gain form |
| **TEMPO** | the full cycle with the trained rule (06) | |

K-GENE (10 §10.2) fires unless TEMPO beats (a), (b), (c) and (d) on the test score at equal
cumulative compute over a 20-episode life, by each arm's own bar (10 §10.1); the comparison
with (d) is the one that tests whether the *genome*, as opposed to the *gains*, is doing
anything, and the comparison with (c) whether the *rule*, as opposed to the *step*, is.

## 4.10 Acceptance tests

**[NORMATIVE]** `tempo/genome.py` is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| E1 | registry | the registry built from the TS-adult module tree has exactly the roles, sizes and total `n_γ` of §4.2; every backbone parameter of matrix kind has exactly one role; the taps and `b_in` have none |
| E2 | identity at birth | with `B_ρ = 0` every `γ_ρ = 0`, `W_eff = W` bit for bit, the norms and offsets equal their weight values, `η_grp = η_1`, `m` equals the primordium pattern, `μ` is uniform; the backbone's output equals the plain backbone's (P6) |
| E3 | ranges | for random `g` with `‖g‖ = 4 sqrt(d_g)` and random `θ`, every `γ_ρ` lies in `(−c_ρ, c_ρ)`; every `1 + a > 0` |
| E4 | expression is live | a finite-difference perturbation of `g` changes `u` and the loss at the next episode's start, **for the programme kinds as well as the differentiable ones**: with `B = 0` on the differentiable rows (birth) a move of `g` still moves the learning-rate, morphology and mixture gains through the differential frame (4.1b), and at `g = g_0` every gain is exactly the identity |
| E5 | no gradient to `g` | after a full step `g.grad is None` and `g` is unchanged bit for bit; `u_ρ.grad` is populated for the differentiable kinds and absent for the programme kinds |
| E6 | regression over a history | on gains produced by a random `(A, B)` from sixteen random `g`, (4.4) from a different initial `θ` reaches `R < 10^{-2}` on the newest pair in 500 steps; a *single* pair is fittable to `10^{-6}` while a history of sixteen unrelated targets is not (both asserted, so the reported residual is known to be informative); on gains outside the map's range the residual is reported and G-REGRESS fires when `> 0.01` |
| E7 | regeneration | after (4.4) with `Δg = 0`, the next episode starts with `u` equal to the fitted values to `1e-5`, and the loss at its first step equals the previous episode's last-step loss on the same windows to `2 σ_step` (the regeneration loses nothing measurable) |
| E8 | fold gradient | `∇_W` and `∇_a` through 03 (3.12) equal finite differences at `1e-4` (relative) |
| E9 | learning-rate role | with `λ_grp = log 2` the optimiser's step on that group is exactly twice the step at `λ_grp = 0` on the same gradient; the clamp to `[¼, 4]` holds |
| E10 | mixture role | the episode's window draw counts equal `systematic_counts(μ, n)` for the expressed `μ`; the floor holds at `0.02` |
| E11 | once per episode | the expression and regression entry points are called exactly once per episode boundary and never inside the step loop (the G-NOFREE-INVERSE sentinel, instrumented over two episodes) |
| E12 | SVF ablation | `U diag(σ ⊙ (1 + a)) Vᵀ` at `a = 0` reproduces `W` to `1e-5`; the SVD is recomputed once per episode and never per step (sentinel) |
