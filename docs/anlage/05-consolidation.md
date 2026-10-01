# 05 — Consolidation: one primitive at three timescales

**Status (2026-09-02, after §11):** the regression (5.3) is retained for vector roles only (§12.6); Proposition 1 stands with the programme as the first part of the code; the expression dial and the anchor were measured (R4, R4b) and closed.

This is the mathematics that makes the two constitutional timescales one object rather than
two mechanisms. It defines the primitive, exhibits its three instances, states the regression
that writes the genome, proves the two propositions the design rests on, and fixes the dial
that turns the treatment into its own control.

## 5.1 Definition 1 — least-squares consolidation

**[NORMATIVE]** Let a memory at horizon `tau` be a parameter `P_tau` and let the faster memory
`P_{tau-1}` supply a stream of targets `{(x_i, y_i)}` through a readout `R(P; x)`. The
**consolidation of `P_{tau-1}` into `P_tau`** is

```
    P_tau  <-  argmin_P  sum_i w_i || R(P; x_i) - y_i ||^2  +  lambda_tau Reg_tau(P)          (5.1)
```

or one gradient step toward it. Three instances, each already verified or specified elsewhere
in this repository:

| horizon | `P_tau` | targets `(x_i, y_i)` from | readout `R` | the step | verified |
|---|---|---|---|---|---|
| **token** | `S_t in R^{d_v x d_k}` per head | the context: `(k_t, v_t)` | `S k` | `S_t = S_{t-1} alpha_t (I - beta_t k_t k_t^T) + beta_t v_t k_t^T`, one step on `(1/2)||S k - v||^2` with `alpha_t` as decay (SOMA (5.4)) | `mixer_chunk.py` vs oracle, 1e-10 |
| **step** | `W` | the data: `(context, next token)` | the phenotype's logits | a Muon/Adam step on `L_CE` | §04 |
| **generation** | `(Gamma, Z)` | the weights: `((a_i, a_j, l, m), W_{l,m}[i,j])` | `D_{l,m}(Gamma, Z)[i,j]` | Adam steps on (5.3) | this file |

**[RATIONALE]** The delta rule is literally an online least-squares regression (Schlag, Irie
& Schmidhuber 2021; SOMA (5.4)); gradient descent on cross-entropy is the maximum-likelihood
instance of (5.1) with the Fisher metric; and the genome fit is (5.1) with the weights as
targets. Nothing is being claimed about joint convergence across horizons: the claim is that
one operator, instantiated three times, is what the design consists of, and that the two
outer instances are what SOMA and GNOMON lacked — a substrate the timescales share.

## 5.2 The generation-level regression

**[NORMATIVE]** Given a phenotype `W` at stage `s` and its identity tables,

```
    J(Gamma, Z)  =  sum_{(l,m)} omega_{l,m} · || D_{l,m}(Gamma, Z) - W_{l,m} ||_F^2 / || W_{l,m} ||_F^2
                    +  lambda_z sum_{a} || z_a ||^2 / d_z  +  lambda_y sum_l || y_l ||^2 / d_y
                    +  lambda_g || Gamma ||^2                                                       (5.3)
```

with `omega_{l,m} = 1` for every role (the importance-weighted variant of §5.7 is a registered
ablation), `lambda_z = lambda_y = 1e-3`, `lambda_g = 1e-6`. Every term is a relative error, so a
role's contribution is its reconstruction fraction and the objective is a sum of numbers in
`[0, 1]` at the start of the fit.

**[NORMATIVE — `REGRESS`]**

```
REGRESS(W, Gamma_init, Z_init, s) -> (Gamma, Z, report)
  1  (Gamma, Z) <- (Gamma_init, Z_init), with Z extended by sampled latents (§02.6) for any unit
                    W has and Z_init lacks
  2  opt <- Adam over (Gamma, Z), lr 2e-3, betas (0.9, 0.99), cosine to 0.1 over N_fit steps
  3  for n = 1 .. N_fit:                                 # N_fit = 3000 at stage 0, 4000 at the adult
  4      decode every role present at stage s              # one full decode, §03.2
  5      J <- (5.3);  step
  6      if n mod 100 == 0: record per-role relative residual  r_{l,m} = ||D - W||_F^2 / ||W||_F^2
  7      stop early when the mean residual improved by < 1e-4 over the last 300 steps
  8  report: per-role residual, decoded erank / trained erank per role, |Gamma|, |Z|, both ratios of §03.6,
             at-birth CE (§5.5), wall-clock
```

**[NORMATIVE — continuity across generations]** `Gamma_init` is the previous generation's
genome and `Z_init` its latent bank restricted to the units that still exist; generation 0
fits from §03.7's initialisation. The genome therefore accumulates across generations rather
than being refit from nothing.

**[NORMATIVE — targets are the adult's weights after the schedule's decay]** The regression
sees the final weights of a lifetime (after the decay to zero), never a mid-schedule
checkpoint, except in the optional in-lifetime consolidation of §06.4, whose report is
labelled as such.

## 5.3 Proposition 1 — the genome is the first part of a two-part code

**Proposition 1.** Write `W_{l,m} = D_{l,m}(Gamma, Z) + E_{l,m}` and assume the residual
entries are `E_{l,m}[i,j] ~ N(0, s_{l,m}^2)` with `s_{l,m}^2 = c · ||W_{l,m}||_F^2 / |W_{l,m}|`
for a fixed `c`, the latents `z_a ~ N(0, sigma_z^2 I)`, `y_l ~ N(0, sigma_y^2 I)` and the genome
`Gamma ~ N(0, sigma_g^2 I)`. Then the Shannon codelength of `W` under this generative model,
up to constants independent of `(Gamma, Z)`, is

```
    L(W)  =  (1/(2c)) sum_{(l,m)} ||W_{l,m} - D_{l,m}||_F^2 / ||W_{l,m}||_F^2 · |W_{l,m}|
             + (1/(2 sigma_z^2)) sum_a ||z_a||^2 + (1/(2 sigma_y^2)) sum_l ||y_l||^2 + (1/(2 sigma_g^2)) ||Gamma||^2,
```

so that minimising `J` in (5.3) with `omega_{l,m} ∝ |W_{l,m}|`, `lambda_z = c d_z / sigma_z^2`,
`lambda_y = c d_y / sigma_y^2`, `lambda_g = c / sigma_g^2` is exactly minimising the two-part
description length of the phenotype given the identity tables.

*Proof.* Under the model, `-log p(W, Z, Gamma) = -log p(W | Gamma, Z) - log p(Z) - log p(Gamma)`.
The first term is a sum over entries of `E^2 / (2 s^2)` plus normalising constants that do
not depend on the parameters; substituting `s_{l,m}^2` gives the first line. The prior terms
are the remaining three. The identification of the multipliers is term-by-term. □

**[NORMATIVE — what is and is not claimed]** With `omega = 1` rather than `∝ |W_{l,m}|`,
(5.3) weights roles equally rather than by their entry count, which is a deliberate departure
from the strict codelength (it stops the FFN from dominating the fit) and is recorded here.
PAC-Bayes and MDL bounds (§10 E-19) bound the generalisation gap of a model by its
description length; applied here they bound the **birth model** `D(Gamma, Z)`, not the adult
after a lifetime. The bottleneck's benefit for the *lifetime* is therefore an empirical
question and is measured by K-ANCHOR and K-LOOP, never assumed from the bound.

## 5.4 Expression: birth, and the dial

**[NORMATIVE]** Birth at stage `s` of generation `g+1`:

```
    W_0  =  eps · W_prev  +  (1 - eps) · D(Gamma_g, Z_g, s)                                (5.4)
```

`W_prev` is the previous generation's weights at stage `s` (its stage-0 checkpoint for a
stage-0 birth). `eps = 0` is genome birth (the treatment); `eps = 1` is the warm-start
control **B-WARM** with no genome at all; the curve of final adult CE over
`eps in {0, 0.25, 0.5, 0.75, 1}` is a reported quantity of §08 R6. This is the PNAS paper's
annealing coefficient turned into the instrument that decides whether compression through the
genome buys anything over inheriting the weights directly. At generation 0 there is no
genome and `eps = 1` with `W_prev` = §04.7's initialisation (§01.5).

**[NORMATIVE]** Where a unit exists in `W_prev` and in `D(...)` the blend is entrywise; a unit
present only in the decode (a wider stage) takes the decoded value.

## 5.5 The at-birth measurement

**[NORMATIVE]** After every regression, the decoded phenotype `D(Gamma, Z, s)` is evaluated
with **no training** on the test split:

```
    L_birth     =  CE( D(Gamma, Z, s) )
    birth_share =  (L_null - L_birth) / (L_null - L_adult),        L_null = ln V
```

`L_adult` is the CE of the phenotype the genome was fit to. `birth_share` is the PNAS paper's
headline statistic in this setting (MNIST: 94% of trained accuracy at 322×), reported every
generation. K-BIRTH (§08.3) fires if `birth_share < 0.5` after a regression whose per-role
residuals are all below 0.5 — that is, if the genome reproduces the weights and the reproduced
weights do not compute the function, which would mean the residual is concentrated where the
function lives.

## 5.6 Controls of the regression itself

**[NORMATIVE]** Reported beside every regression:

| control | what it isolates |
|---|---|
| `d_z = 0` | the innate-only genome: how much of the fit the latent bank carries |
| a random genome (§03.7) decoded at the same stage | `L_null`-level reference for `birth_share` |
| shuffled decode: the decoded matrices with rows permuted | the PNAS shuffle control; must be at `L_null` |

## 5.7 Registered ablations

**[NORMATIVE — not defaults; each is an arm of §08 R6]**

* **Importance weighting.** `omega_{l,m}[i,j] = 1 + a_i^{(l)}` where `a^{(l)}` is the row-wise
  mean squared activation of the trained phenotype on a probe batch, so the fit spends its
  capacity where activations flow.
* **Baldwinian refinement.** After `REGRESS`, `N_refine = 300` steps of gradient descent on
  `L_CE(D(Gamma, Z); batch)` with respect to `(Gamma, Z)` through the decode — the PNAS
  paper's abandoned end-to-end variant, used only as a refinement of at-birth loss. Cost:
  `N_refine` decodes plus phenotype forward/backward.
* **Anchor.** During a lifetime, `L = L_CE + lambda_anchor sum_{(l,m)} ||W_{l,m} -
  D_{l,m}||_F^2 / ||W_{l,m}||_F^2` with `D` decoded once at birth and held fixed;
  `lambda_anchor in {1e-4, 1e-3, 1e-2}`. Kill: K-ANCHOR.

## 5.8 Proposition 2 — nesting

**Proposition 2.** Let `A_s ⊂ A_{s'}` on every channel and token axis, `s < s'`, and let
`Z_{s'}` restrict to `Z_s` on `A_s`, with the same layer set and layer latents. Then for every
role `m`, layer `l`, and `i, j in A_s`,

```
    raw_{l,m}^{(s')}[i, j]  =  raw_{l,m}^{(s)}[i, j],
```

and `D_{l,m}^{(s')}[i,j] / D_{l,m}^{(s)}[i,j] = rms(raw^{(s)}) / rms(raw^{(s')})`, a single
scalar per matrix.

*Proof.* By (3.1) the features `u_{i,l}, v_{j,l}` depend only on `(b_i, z_i, r_l, y_l, side)`,
and by P1 the codes of `i, j in A_s` are identical at `s` and `s'`; by hypothesis so are their
latents and the layer's. (3.2) is entrywise in `(i, j)` given the features, so the `raw`
entries agree. (3.4) divides by the RMS over the whole matrix, which differs between the two
stages only through the new entries, and multiplies by `s_target sigma_{l,m}`, which depends
on the shape and on `(r_l, t_m)` alone; the ratio follows. □

**[NORMATIVE — the layer axis]** When `L` grows, the existing layers' codes change (§02.4)
and Proposition 2 does not apply to them; §06.2 keeps their trained weights and does not
re-decode them, so no discontinuity results. The next `REGRESS` refits the genome to the new
codes, and the report records the per-layer residual before and after.

## 5.9 Acceptance tests

**[NORMATIVE]** `anlage/regress.py` is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| C1 | **planted recovery** | a phenotype generated by a known `(Gamma*, Z*)` is recovered by `REGRESS` from a random start to per-role residual `< 1e-4` at a small stage |
| C2 | objective | (5.3) on a hand-built two-role case equals the closed form to `1e-10` |
| C3 | priors prune | with `lambda_z = 10` the fitted `||Z||` is at least 10× smaller than with `lambda_z = 1e-3`, and the residual rises — the price is visible |
| C4 | continuity | `REGRESS` from `(Gamma_g, Z_g)` starts at the residual `(Gamma_g, Z_g)` had, not from scratch; new units are sampled per §02.6 |
| C5 | at-birth protocol | `L_birth` is computed on the test split with no optimiser step; `birth_share` of a random genome is within `0.05` of `0` |
| C6 | the dial | (5.4) at `eps = 1` returns `W_prev` bit-exactly and at `eps = 0` returns the decode; units absent from `W_prev` take the decoded value |
| C7 | shuffle control | the shuffled decode scores within `0.1` nats of `L_null` |
| C8 | Proposition 2 | `raw` entries agree to `1e-6` on `A_s` across two stages from one `(Gamma, Z)`; the per-matrix ratio is a single scalar |
| C9 | determinism | two regressions from one seed give bit-identical `(Gamma, Z)` |
| C10 | report | every field of step 8 is present and finite in the JSON report; `write_json` refuses a NaN |
