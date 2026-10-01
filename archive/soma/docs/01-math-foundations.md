# SOMA — Specification 01: Mathematical Foundations

**Spec version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Normative for all modules.

Every symbol used anywhere in this specification appears in §1.11. A symbol used but absent
from that table is a specification defect.

**Marker vocabulary.** `[NORMATIVE]` marks only statements that are **mechanically
checkable**. Motivation carries `[RATIONALE]` and binds nothing. `[DERIVED]`,
`[EMPIRICAL]`, `[OPEN]`, `[OURS]` are as in §00.5.

---

## 1.1 Objects

SOMA's state is a triple `(U, Gamma, Theta_S)`:

- `U` — the **unit population**, a disjoint union over Sites,
  `U = U_0 (+) ... (+) U_{L-1}`, one population per Site (§05.5). Population-valued
  quantities carry a Site index `l`, suppressed where a statement holds for every Site.
- `Gamma` — the **germline**: small, heritable, gradients prohibited (§02).
- `Theta_S` — the **soma**: large, gradient-trained, discarded at the end of a lifetime.

A unit `u` in `U_l` occupies a fixed **slot** in a preallocated `side x side` grid:

| Field | Shape | Domain | Side | Meaning |
|---|---|---|---|---|
| `slot_u` | `(i1,i2)` | `{0..side-1}^2` | bookkeeping | routing address (§05.4) |
| `live_u`, `dying_u` | bool | — | bookkeeping | slot occupancy, retirement flag |
| `p_u` | `R^{m_dim}` | `S^{m_dim-1}` | **soma** | position in expression space |
| `g_u` | index set, `\|g_u\| = k_g` | `{0..N_G-1}` | **germline-addressed** | expressed genes |
| `m_u` | scalar | `[0,1]` | bookkeeping | MSG mask |
| `age_u` | scalar | `N` | bookkeeping | steps since birth |
| `L_u` | tuple | discrete | bookkeeping | lineage record |
| `nu_u` | scalar | `>= 0` | bookkeeping | running utility (§04.4) |

**[NORMATIVE]** `slot_u`, `live_u`, `dying_u` are **bookkeeping**, not germline: `BIRTH`
assigns a slot within a lifetime and nothing about it is inherited. (v2 §1.1 called
`slot_u` germline-addressed while §02.1.1 called it bookkeeping; the latter is correct.)

**[NORMATIVE — structural change]** A *structural change* is any change to `U` (births,
deaths), to `g_u`, or to `p_u` outside a gradient step. Per-token variation of the routed
set `R_t` and of the context gate `psi` is **not** structural change.

---

## 1.2 The expression map

Gene `g` is a radial basis function on the unit sphere with centre `mu_g` in
`S^{m_dim-1}`, width `sigma_g > 0`:

```
  phi_g(p) = exp( - || p - mu_g ||^2 / (2 sigma_g^2) )                          (1.1)
  phi(p)   = ( phi_0(p), ..., phi_{N_G-1}(p) )   in   [0,1]^{N_G}
```

**[NORMATIVE]** The germline cost of the expression map is `(m_dim + 1) * N_G` reals,
independent of `|U|`, `d_model` and `L`. This is the bottleneck property; §09.3 and K9 rest
on it.

**[NORMATIVE — germline-scale constants]** `N_G = 2048`, `k_g = 8`, `m_dim = 8`,
`J_max = 256` are **identical at every configuration**. Only the soma scales. Three
consequences, all intended:

1. a germline is transferable between configurations without reshaping — v2 set M-nano
   `N_G = 512` and M1 `N_G = 2048`, which made the M-nano→M1 inheritance that justifies
   `G_gen = 1` at M1 **shape-impossible**;
2. §09.3's claim that `|Gamma|` is independent of model scale becomes true by construction
   rather than by argument;
3. the `N_G` sweep of §08.6.1 varies `N_G` against `d_model` deliberately, which is only
   interpretable if the default holds `N_G` fixed.

**[NORMATIVE — why `m_dim = 8`]** With `m_dim = 2`, `N_G = 2048` centres lie on a circle at
a mean spacing of `2 pi / 2048 = 0.003` rad while an RBF spans `~sigma_g` rad, so at most
`~2 pi / sigma_g ≈ 5` profiles are resolvable and the expression map would have about five
effective degrees of freedom regardless of `N_G`. `m_dim = 8` gives `S^7` room for 2048
centres. Germline cost `2048 * 9 = 18432` reals, inside the `1e4`–`1e5` target of §02.1.

**[EMPIRICAL — initialisation, measured over 4e5 samples]** `mu_g` uniform on
`S^{m_dim-1}`; `p_u` uniform on `S^{m_dim-1}`; `log sigma_g = log(1.2)`. At
`m_dim = 8, sigma_g = 1.2`:

```
  E[phi] = 0.5147      E[phi^2] = 0.2809      median[phi] = 0.4995
```

v2 set `sigma_g = 0.6` with the comment "so a typical `phi` is near 0.5"; the measured
median there is **0.062**, and `N_eff` falls to 0.79 at `m_dim = 8` — below the abort guard
of §1.5. A median of 0.50 is obtained at `sigma_g = 1.2` for every `m_dim` tested.

**[DERIVED]** `phi_g` is smooth in `p`, so unit positions are gradient-trainable while gene
membership is discrete and reachable only by selection. This is the seam between the two
write-permission regimes.

---

## 1.3 The expression operator

**[NORMATIVE — v2 correction: the operator generates a gate]** v2 defined
`E(u,c) = sum_g phi_g psi_g Omega_g` as a *linear* map and then asserted
`E(u,x) x = B[gate_u (*) sigma_act(A x)]`. That is not an identity — the right side is
nonlinear, the left is not — so the object the specification defined was not the object the
backbone computed. The construction is restated so the two coincide.

**Definition 1 (expression gate).** For unit `u` under context `c`,

```
  gate(u, c) = phi(p_u) (*) psi(c) (*) 1_{g_u}     in [0,1]^{N_G},  k_g-sparse   (1.2)
```

**Definition 2 (unit transform).** With shared `A in R^{N_G x d_in}`,
`B in R^{d_out x N_G}` and elementwise activation `sigma_act`,

```
  F(u, c, x) = B [ sigma_act( gate(u,c) (*) (A x) ) ]                            (1.3)
```

**[NORMATIVE]** `sigma_act` is applied **after** the gate. This is what makes units distinct
nonlinear functions; §1.4 shows the pre-gate placement makes them provably indistinct.

**Definition 3 (connectivity form, diagnostics only).** With `Omega_conn(c)` in
`R^{N_G x N_G}`, `E_conn(u,v,c) = phi(p_u)^T Omega_conn(c) phi(p_v)`. At constant `c` this
is the developmental encoding `W = X Omega_conn X^T`. Used only in §08.

### 1.3.1 The four rate regimes

**[RATIONALE]** The organising claim: attention, learning, development and inheritance are
one construction read at different rates of change of its arguments.

| Regime | Varying | Held | Reads as |
|---|---|---|---|
| Moment | `psi(c)` per token | `p_u`, `A`, `B`, `U` | content-dependent connectivity |
| Lifetime | `p_u`, `A`, `B` per step | `U`, `mu`, `sigma` | learning within a body plan |
| Generation | `U`, `g_u` per event | `A`, `B`, `mu`, `sigma` | development |
| Phylogeny | `mu_g`, `sigma_g`, policy | — | inheritance |

**[NORMATIVE — scope of the Moment claim]** (1.2)–(1.3) supply *content-dependent
connectivity* only; they have no dependence on tokens other than `t`. The fast-weight
correspondence with attention belongs to the mixer state `S_mix` of §05.2, which aggregates
over `t' < t`. The Moment regime is the two together. Ablation **A10** tests whether `psi`
contributes.

**[NORMATIVE — canalization constraint]** `U`, `p_u`, `g_u` may not vary within a forward
pass. Only `psi` and `R_t` may.

---

## 1.4 Cost, and why units are materialised separately

**Theorem 1 (cost independent of population size).** For a token routed to `R_t` with
`|R_t| = k_route`, the Site output (5.20) is evaluable in

```
  O( N_G * d_in  +  k_route * k_g * d_out )                                      (1.4)
```

operations, with no dependence on `|U|`, and without materialising any `d_out x d_in`
matrix.

*Proof.* `z = A x_t` costs `O(N_G d_in)` once per token and is shared by every routed unit.
`gate(u, x_t)` is supported on `g_u`, of size `k_g`, so `sigma_act(gate (*) z)` is
`k_g`-sparse and costs `O(k_g)`. Applying `B` to a `k_g`-sparse vector touches `k_g`
columns, costing `O(k_g d_out)`. Summing over `k_route` units gives `O(k_route k_g d_out)`.
`psi` costs `O(N_G d_key) ⊆ O(N_G d_in)`, and `W_shared` costs `O(d_model^2) ⊆ O(N_G d_in)`
since `d_out <= N_G`. No term involves `|U|`. QED

**[NORMATIVE — the aggregate form is not equivalent]** Let
`G_t = sum_{u in R_t} r_ut m_u gate(u, x_t)`. Because `sigma_act` is applied after gating,

```
  sum_u r_ut m_u B[ sigma_act( gate(u,x_t) (*) z ) ]  !=  B[ sigma_act( G_t (*) z ) ]  (1.5)
```

in general. Units are therefore genuinely distinct nonlinear functions of the same input,
differing in **which** coordinates of `z` they see (`g_u`) and **how strongly**
(`phi(p_u)`).

**[NORMATIVE — v2 correction, recorded]** v2 applied `sigma_act` to `z` *before* gating and
then forbade the aggregate form on the grounds that aggregating "destroys the premise that a
unit is a distinct function". With the activation before the gate, both `B` and the gate are
linear, so

```
  sum_u r_ut m_u B[ gate_u (*) sigma_act(z) ]  ==  B[ G_t (*) sigma_act(z) ]     exactly
```

The two forms computed bit-for-bit the same function, and every unit already **was** a
linear reweighting of one shared nonlinear feature. v2 therefore mandated a `k_route`-fold
cost increase on a distinction that did not exist. Moving the activation after the gate
costs nothing — the support is still `k_g`-sparse — and makes the claim true.

**[DERIVED]** A second, independent reason to materialise per-unit outputs: `delta_ut`, the
per-unit contribution, is required by the credit assignment of (4.2) and the utility of
(3.6). Even were the two forms equivalent, the aggregate form would destroy the signal that
selection runs on.

**[DERIVED]** At M1 (`N_G = 2048`, `d_in = d_out = 512`, `k_route = 8`, `k_g = 8`):
`1.05e6 + 3.3e4` operations per token per Site — the shared `A x_t` dominates and the entire
population costs about **3%** of the Site.

---

## 1.5 Initialisation

**[NORMATIVE — the criterion is on the realised output; the free scale is calibrated]** The
Site's unit path is nonlinear and `relu(.)^2` is positively homogeneous of degree 2, so no
single closed-form fan-in rule determines the scale. The criterion is stated on the measured
output and the remaining freedom is fitted:

```
  Var(a_g) = 1 / d_in                                                            (1.6)
  Var(b_g) : chosen by CALIBRATE so that Var( unit path output ) = Var(x)        (1.7)
             for x ~ N(0, I) rescaled to ||x|| = sqrt(d_in)
```

```
Algorithm CALIBRATE(A, field, k_g, tol = 0.05, iters = 20)
  precondition: A initialised per (1.6); N_probe >= 4096 units sampled from §02.2.1
  E2_hat <- Monte-Carlo estimate of E_{u, g in g_u}[ phi_g(p_u)^2 psi_g^2 ]
  s <- 1.0
  repeat iters times:
      draw B with Var(b_g) = s / (k_g * E2_hat)
      v <- measured Var over probes of  B[ relu( gate(u,x) (*) (A x) )^2 ]
      if | v / Var(x) - 1 | <= tol:  return Var(b_g)
      s <- s * ( Var(x) / v )                    # secant on a monotone positive scalar
  raise CalibrationFailed
  postcondition: measured output variance within tol of Var(x); the fitted value is
                 recorded in the run manifest
```

**[NORMATIVE — v2 correction]** v2 asserted `Var(W_ij) = N_eff * E2 * Var(a) * Var(b)` with
`N_eff = E_u[ ||phi (*) 1_{g_u}||_1 ]`. For a sum of `k_g` independent rank-one terms the
correct prefactor is **`k_g`**, not `N_eff`: `N_eff = k_g E[phi]` is a *first* moment while
`E2` already carries `phi^2`, so the product is cubic in `phi` where the variance is
quadratic. Measured at the v2 initialisation this gives a realised `d_in Var(W)` of
`3.94 * gain_act` — outside §02.5's own `[0.5, 2.0]` acceptance band, so v2 as written would
have aborted every run at `LIFETIME` step 3. Calibration removes the fragile analytic chain
entirely: the acceptance test becomes the definition rather than a check on a formula that
may itself be wrong.

**[NORMATIVE]** `N_eff = k_g * E[phi]` is retained as a **reported diagnostic only**.
Guards `N_eff >= 1` and `E2_hat >= 1e-4` abort the run; both indicate a degenerate
expression field in which almost no gene is expressed.

**[RATIONALE]** Normalisation does not rescue a bad initialisation: the cited evidence is
that batch normalization corrects vanishing but **not** exploding decoded weights, and that
a mis-initialised weight-generating network can fail to train at any learning rate.

---

## 1.6 The metabolic functional

Positions and queries lie on the unit sphere: `||p_u|| = 1`,
`q_t = normalize(x_t W_pos)` in `S^{m_dim-1}`, so `||q_t - p_u||^2` lies in `[0, 4]`.

```
  L_wire = lambda_wire * (1/|T_set|) * sum_{t in T_set} sum_{u in R_t}
                r_ut * m_u * || q_t - p_u ||^2                                  (1.8)

  L_disp = lambda_disp * ( 2 / (|U_live|(|U_live|-1)) ) *
                sum_{u < v, both live} exp( - ||p_u - p_v||^2 / (2 s_rep^2) )   (1.9)

  L_size = lambda_size * (1/|U_live|) * sum_{u in U_live} m_u                   (1.10)

  L_met  = L_wire + L_disp + L_size                                             (1.11)
  L      = L_task + L_met + lambda_z * (log Z)^2                                (1.12)
```

**[NORMATIVE — the dispersion term is required; v2 correction]** `L_wire >= 0` with
equality **iff `p_u = q_t` for every routed pair** — the configuration in which every
position and every query collapse to one point. That point lies on the sphere, so it is
feasible, and it is the **global** minimiser. Worse, `W_pos` appears in no other term
(§05.4 says it is "used only by `L_wire`"), so `q_t` has **no gradient source other than
`L_wire`** and is driven purely toward collapse. v2 claimed the sphere constraint meant
`L_wire` "cannot drive a collapse to a point"; the sphere bounds the term's *value*, not its
*argmin*, and the claim was false.

(1.9) is a Gaussian repulsion between live positions, `s_rep = 0.5`, with
`lambda_disp` a germline policy constant (§02.2.3), default `1e-3`. It is the
**character-displacement** mechanism the design's ecological analogy invoked and never
implemented: two units occupying the same niche pay for it. The equilibrium is a dispersed
configuration, as in a Thomson problem on `S^{m_dim-1}`.

**[NORMATIVE — `G-NOCOLLAPSE`, defined numerically]** The mean pairwise distance between
live positions must satisfy

```
  mean_{u<v} || p_u - p_v ||  >=  0.60 * D_unif(m_dim)                          (1.13)
```

**[DERIVED — `D_unif` in closed form]** For `p, q` independent and uniform on `S^{m-1}`,
`<p,q>` has density proportional to `(1 - t^2)^{(m-3)/2}` on `[-1,1]`, and

```
  D_unif(m) = E|| p - q ||  =  2^{m-1} * Gamma(m/2)^2 / ( sqrt(pi) * Gamma(m - 1/2) )  (1.13a)
```

which gives `D_unif(2) = 4/pi = 1.27324`, `D_unif(3) = 4/3`, `D_unif(4) = 1.35812`, and
**`D_unif(8) = 1.389328`**, so the (1.13) threshold at `m_dim = 8` is **`0.83360`**.

**[NORMATIVE — v3 correction]** v3.0 tabulated `D_unif(8) = 1.375` as "measured". It is
wrong by 1.0%: both (1.13a) and a Monte-Carlo estimate over `8192` points agree on
`1.3893`. The threshold is computed from (1.13a), never from a tabulated literal, so it
cannot drift again.

**[NORMATIVE — `L_size` normalisation; v2 correction]** (1.10) is normalised by `|U_live|`,
not `|T_set|`. `L_size` sums over *units* and does not depend on `t`, so dividing by the
token count made it inversely proportional to batch size — contradicting the sentence that
justified it. It is stated in units of live mask mass rather than a per-unit footprint
`s_u = k_g(d_in + d_out)`, because a unit does not own those parameters: it owns `p_u` and
`k_g` indices into shared tensors, as §03.4.1 states explicitly.

### 1.6.1 Population size is bounded by a cap, not by a gradient

**[NORMATIVE]** `L_size` has **identically zero gradient** (`m_u` is bookkeeping). It is a
reported diagnostic — the metabolic bill — and is **not** the mechanism bounding the
population. That mechanism is an explicit cap:

```
  n_nondying      = |{ u : live_u and not dying_u }|  -  n_death
  BUDGET_HEADROOM = max( 0 , min( U_cap - n_nondying , U_max - |U_live| ) )     (1.14)
```

**[NORMATIVE — (1.21)'s statistic is `L_task`, not a proxy for it; v3.1, audit A-73]** Every
term of (1.21) and (1.22) — `L_active`, `L_masked`, and both inputs to `epsilon_bif` — is the
probe **cross-entropy**. v3.0 computed `mean(logits^2)` in all three places: a scalar with no
dependence on the targets, which responds to the scale of the output distribution rather than
to the loss. The two are not a monotone reparameterisation of each other, so the alarm's
ordering over cohorts was not (1.21)'s ordering — masking cohorts of 4, 8 and 16 units moved
the surrogate by `0.73`, `0.22` and `0.03` times the corresponding relative change in
`L_task`. This is the **fourth** defect in this one criterion; A-36, A-58 and A-61 were all
about the *threshold*, and no amount of work on a threshold reaches the statistic it is
applied to.

**[NORMATIVE]** Headroom is evaluated **net of the deaths applied in the same event**. v2
evaluated it before deaths and `DEATH` keeps `live_u = True` for a further `phi_ann` steps,
so once the cap was reached births were permanently zero — ε-lexicase would have produced
parents that could never reproduce.

**[NORMATIVE — v3.1 correction, two bounds]** (1.14) carries **two** bounds because two
distinct resources are exhaustible and they are not the same resource:

* `U_cap` bounds the **effective** population — the units that actually compute. A dying
  unit is annealing `m_u -> 0` and is already excluded from routing (§05.4), so it does not
  count against this bound.
* `U_max = side^2` bounds the **slots**. A dying unit still occupies one for `phi_ann`
  further steps, and `FREE_SLOT_NEAREST` (§03.2) raises when the grid is full, so the slot
  bound must count it.

v3.0's single expression `U_cap - (|U_live| - n_death)` assumes the event's deaths free
their slots immediately. They do not. The realised population therefore **exceeded the
cap** — measured `|U_live| = 243` against `U_cap = 240` on an M-nano lifetime — and with
`U_max - U_cap = 16` slots of slack at that scale, a few concurrent selection events could
have exhausted the grid outright.

This is audit A-17's defect approached from the opposite side: v2 evaluated headroom
*before* deaths and births became permanently zero at the cap; v3.0 evaluated it net of
deaths and the cap stopped binding at all. Separating the two bounds makes both failures
unreachable, and both are asserted as postconditions of `SELECT` and `GROW` rather than
assumed (audit A-62).

`L_wire` and `L_disp` carry gradients through `p_u` and `W_pos`, and are what ablation
**A1a** tests. Because `L_size` has no gradient, an ablation setting `lambda_size = 0`
would be a no-op; §08.6's A1b is therefore defined on `U_cap`, the actual mechanism.

---

## 1.7 The rate hierarchy

| Rate | Symbol | Period | Object | Update |
|---|---|---|---|---|
| Moment | `r_0` | 1 token | `S_mix`, `psi` | recurrence (§05.2) |
| Lifetime | `r_1` | 1 step | `Theta_S` | Muon / AdamW, `a_lr(n)` |
| Generation | `r_2` | `T_sel(n)` steps | `U`, `g_u` | selection (§04) |
| Phylogeny | `r_3` | one germline generation | `Gamma` | mirrored ES (§02.6) |

**[NORMATIVE — event scheduling]** `T_sel(n)` is a non-integer increasing function, so
`n mod T_sel(n) == 0` is not a schedule. Every periodic hook keeps an explicit counter:

```
  fire when n >= n_next and n > 0 ;  then  n_next <- n + ceil( T_x(n) )         (1.15)
```

The event count follows `n_{j+1} = n_j + T_sel(n_j)`, whose continuum solution is

```
  E_sel(N_step) = ( n_c / T_0 ) * ln( 1 + N_step / n_c )                        (1.16)
```

**[NORMATIVE — every hook period is derived from a target count]**

```
  T_0    = ceil( n_c * ln(1 + N_step/n_c) / E_target )      E_target = 10        (1.17)
  T_arch = max(1, floor( N_step / A_target ))               A_target = 20        (1.18)
  T_cert = max(1, floor( N_step / C_target ))               C_target = 4         (1.19)
```

**[NORMATIVE — v2 correction]** v2 derived `T_0` but left `T_arch` and `T_cert` with **no
value or formula anywhere**, so whether the open-ended machinery fires at all — the largest
single defect of v1 — became undecidable rather than repaired. (1.18)–(1.19) fix this. **[NORMATIVE]** The *realised* count is
`floor((N_step - T_x)/T_x) + 1` and may differ from the target by one; the realised counts
are tabulated in §1.9 and asserted by gate `G-SPECNUM`, so no section may assume the target
is attained exactly. §06.1.3 decouples the certification ring's **capture cadence** from `T_cert`, so "the ring
spans the run" and "certification fires `C_target` times" are no longer in conflict.

### 1.7.1 Timescale separation

With `a_lr(n)` the soma learning rate and `b_str(n) = rho(n)/T_sel(n)` the structural churn
per step, the separation quantity is dimensionless:

```
  sep(n) = ( b_str(n) / b_str(0) ) / ( a_lr(n) / lr_max )                        (1.20)
```

**[NORMATIVE]** `sep(n) <= 1` is asserted and logged for `n_warm <= n < n_dec` **only**. v2
exempted warmup but not decay; as `a_lr -> 0` the ratio diverges (`sep = 6.31` at
`n = 12299` at M1), so the assertion aborted the final steps of every run.

**[NORMATIVE — what is and is not claimed]** `b_str(n)` is summable by design (Lemma 3.2):
the slow variable is intended to **freeze**, not to converge jointly with the fast one. This
deliberately fails the persistent-excitation condition of two-timescale stochastic
approximation. SOMA claims only that (i) structural churn decays to zero, so the system
reduces to a single-timescale problem after canalization, and (ii) structural change is slow
relative to weight change in the sense of (1.20). No joint-equilibrium convergence is
claimed.

### 1.7.2 The bifurcation diagnostic

**[NORMATIVE — v2 correction]** v2 measured
`|L_task(probe; after) - L_task(probe; before)|` **at** a birth. Theorem 3.1a guarantees
that difference is zero to floating-point reduction order, so the alarm was **unreachable**,
`rho <- rho/2` was dead code, M1 criterion 2 was trivially satisfied and §03.7's falsifier
was inert.

The alarm is measured **after the mask anneal completes**, against a masked counterfactual:

```
  bif = | L_task( probe ; theta_{n_b + phi_ann}, cohort active )
        - L_task( probe ; theta_{n_b + phi_ann}, cohort forced to m_u = 0 ) |   (1.21)
```

Both terms use the same weights and the same fixed probe batch and differ only in whether
the cohort born at `n_b` is switched on.

**[NORMATIVE — the criterion is relative and size-aware]** The alarm fires when

```
  | L_active - L_masked |  /  | L_active |   >   max( noise_floor , 3 * c / |U_live| )   (1.22)
  noise_floor = max( 3 * sd_probe / |L_active| , 1e-3 )
```

with `c` the cohort size and `sd_probe` the standard deviation of `L_task` on the fixed probe
across `k_probe = 8` repeated forward passes under the run's nondeterminism.

**[RATIONALE]** The alarm asks whether a birth cohort behaves as a **controlled probe**. A
cohort of `c` units in a population of `|U_live|` carries roughly `c/|U_live|` of the routed
mass, so changing the probe by about that fraction is *expected* — it is what those units do.
A bifurcation is a change **disproportionate** to the cohort's size, and (1.22) says exactly
that.

**[NORMATIVE — two earlier forms, both wrong, v3.1]** Recorded because they failed in opposite
directions and the shape of each failure is instructive.

*`3 * sd_probe` alone is zero.* On this card with these kernels the eight passes are
bit-identical, so `sd_probe = 0`, the threshold is `0`, and the alarm fires on **any** non-zero
difference. Realised: `1.12%` of births, above §03.7's falsifier, on differences of order
`1e-9`. A threshold of zero is not a threshold (audit A-58).

*An absolute floor anchored to the initial scale is worse.* `max(3 sd_probe, 1e-3 |L(0)|)` is
fixed while the logit scale grows by orders of magnitude during training, so the criterion ends
up reporting the loss scale rather than the operator: **67%** of selection cohorts fired on a
healthy run (audit A-61). It is also size-blind — a cohort of 8 units and a cohort of 176 were
judged against the same bar.

(1.22) is scale-free over the run and size-free over the schedule, which is what makes §03.7's
falsifier a statement about the birth operator. v2 defined it as a "batch-to-batch" standard deviation of a batch it also
mandated be **fixed** — a quantity that does not exist. On `bif > epsilon_bif`,
`rho <- rho / 2` for the remainder of this lifetime, and the event is logged.

**[NORMATIVE — in-lifetime policy writes are local]** Any in-lifetime mutation of a policy
constant (`rho` here, `d_rate` in §04.5.1) acts on a **deep copy** of `Policy` taken at
`LIFETIME` step 0. Only the pre-lifetime values enter `ES_STEP`. Without this the writes
would be Lamarckian inheritance across generations, and `ASSERT_BARRIER` — which checks
`requires_grad`, `.grad` and optimizer membership — cannot detect a direct write.

### 1.7.3 Short-horizon bias

**[NORMATIVE]** Selection cases are scored on an **improvement** signal (§04.3, a difference
against a lagged self) and lifetimes primarily on **time-to-criterion** rather than final
loss (§07.3), because an outer loop scoring the *level* of a loss after `k` inner steps is
biased toward conservatively small effective learning rates.

---

## 1.8 What would falsify the mathematics

- If `L_wire + L_disp` with `lambda_wire > 0` does not raise `Q_route` above the
  `lambda_wire = 0` control under the §08.5 margin, the metabolic claim fails (A1a, K4).
- If (1.13) is violated, positions have collapsed and unit differentiation has failed.
- If the bifurcation alarm rate does not fall as `rho` falls, (1.20)'s separation claim
  fails.
- If `psi == 1` (A10) is equivalent under the declared margin, the Moment regime contributes
  nothing.
- If replacing (1.3) with the pre-gate activation is equivalent under the declared margin,
  the per-unit nonlinearity of §1.4 is not doing work. (**A12**, added because v2's claim
  here was false and the corrected claim must itself be tested.)

---

## 1.9 Configured dimensions

**[NORMATIVE]** Germline-scale constants are identical at both configurations:
`N_G = 2048`, `k_g = 8`, `m_dim = 8`, `J_max = 256`.

| Symbol | M-nano | M1 | Meaning |
|---|---|---|---|
| `L` | 6 | 12 | blocks |
| `d_model` | 192 | 512 | residual width (`= d_in = d_out` at a Site) |
| `n_h` / `n_kv` / `d_h` | 4 / 2 / 48 | 8 / 2 / 64 | attention heads |
| `n_m` / `d_k` / `d_v` / `e_v` | **3** / 64 / 64 / 1 | 4 / 128 / 128 / 1 | mixer (`n_m d_v = d_model e_v`) |
| `C_chunk` | 32 | 64 | mixer chunk |
| `T_ctx` | 128 | 512 | context |
| `V` | 4096 | 8192 | vocabulary |
| `k_route` | 4 | 8 | units routed per token |
| `U_max` / `side` | 256 / 16 | 1024 / 32 | slot grid (`U_max = side^2`) |
| `\|U\|_0` | 64 | 256 | initial live units per Site |
| `U_cap` | 240 | 960 | population cap per Site |
| `d_key` | 32 | 64 | routing key width |
| `D_tok` | 4e7 | 8e8 | training tokens |
| `B_tok` | 16384 | 65536 | tokens per step (power of two) |
| `N_step` | 2442 | 12208 | `= D_tok / B_tok` |
| `N_step_ref` | 2442 | 12208 | fitness reference horizon (§07.3) |
| `lr_max` | 3e-3 | 1e-3 | Muon peak LR |
| `n_c` | `0.25 N_step` | `0.25 N_step` | canalization constant |
| `T_0` (1.17) | 99 | 492 | 10 selection events (realised: 10, 10) |
| `T_arch` (1.18) | 122 | 610 | 20 archive steps (realised: 20, 20) |
| `T_cert` (1.19) | 610 | 3052 | 4 certifications (realised: 4, 3) |
| `P_es` | 2 | 2 | ES mirrored pair |
| `T_life` | **[OPEN]** | **[OPEN]** | measured at M0 |

**[NORMATIVE]** `n_m = 3` at M-nano. v2 set `n_m = 2, d_v = 64`, violating its own
constraint `n_m d_v = d_model e_v` (`128 != 192`) and producing a hard shape error in
§05.2.4, where `RMSNorm(o)` would be multiplied elementwise by a `d_model`-dimensional gate.

**[OPEN — `T_life`]** v2 asserted 1.5 min (M-nano) and 40 min (M1). The M1 figure implies
`6 N_par D_tok / t = 1.17e17 / 2400 s = 48.6 TFLOP/s`, about **47% of this card's dense bf16
peak**, from the `6ND` estimate alone and before attention, norms, routing, sparse gathers,
`float32` triangular solves, held-out evaluation, `erank` SVDs, certification and archive
decoding. **Deferral criterion:** `T_life` is measured at M0 (gate `G-M0`), and
`G_gen = floor(T_run / (P_es * T_life))` together with the whole M3 budget is recomputed
from the measured value before the decisive experiment is scheduled. §07.8's own rule
forbids asserting an unmeasured wall-clock figure, and v2 violated it.

---

## 1.10 Derived quantities

```
  n_c        = 0.25 * N_step
  phi_ann    = min( 5000, max(50, floor(0.02 * N_step)) )                 (§03.3)
  q_cases    = max( D_min, round( d_rate * B_tok ) )                      (§04.6)
  D_unif(8)  = 1.389328                                                   (1.13a)
  L_rand     = ln V                                                       (§07.1)
```

**[NORMATIVE]** `q_cases` is **proportional** to `d_rate`. v2 set
`q_cases = ceil(D_min / d_rate)`, which is inversely proportional, so §04.5.1's cost
governor (`d_rate <- 0.7 d_rate` on trip) *increased* the case count and therefore the cost
it was meant to bound — an unstable positive feedback running to a 5x cost floor. It also
inverted the semantics of the cited evidence, in which a lower rate means *fewer* cases.

---

## 1.11 Symbol table

Complete. A symbol used anywhere in this specification and absent here is a defect.

**Objects and germline** — `U`, `U_l`, `U_live`, `U_elig`, `U_parent`, `U_death`, `Gamma`,
`Theta_S`, `slot_u`, `live_u`, `dying_u`, `p_u`, `q_t`, `g_u`, `m_u`, `age_u`, `L_u`,
`nu_u`, `nu_hat_u`, `mu_g`, `sigma_g`, `phi_g`, `phi`, `psi_g`, `psi`, `kappa_g`, `a_g`,
`b_g`, `A`, `B`, `gate(u,c)`, `G_t`, `F(u,c,x)`, `Omega_conn`, `E_conn`, `Inventory`,
`J_max`, `n_nbr`, `n_policy`, `N_eff`, `E2_hat`.

**Dimensions** — `L`, `d_model`, `d_in`, `d_out`, `d_h`, `d_k`, `d_v`, `d_key`, `n_h`,
`n_kv`, `n_m`, `e_v`, `C_chunk`, `T_ctx`, `V`, `m_dim`, `N_G`, `k_g`, `k_route`, `U_max`,
`side`, `|U|_0`, `U_cap`, `D_tok`, `B_tok`, `N_step`, `N_step_ref`, `N_par`, `N_tok`,
`N_probe`, `k_probe`, `lr_max`, `n_warm`, `n_dec`, `D_qkv`.

**Site and backbone** — `x_t`, `z`, `sigma_act`, `gain_act`, `delta_ut`, `W_shared`,
`W_pos`, `W_key`, `W_ctx`, `W_qkv`, `W_gate`, `OutProj`, `W_q`, `W_k`, `W_v`, `W_o`,
`w_alpha`, `w_beta`, `K_unit1`, `K_unit2`, `R_t`, `cand`, `r_ut`, `E_emb`, `gain_rms`,
`z_sink`, `y_blk`.

**Mixer** — `S_mix`, `S_next`, `q_mix`, `k_mix`, `v_mix`, `alpha_mix`, `beta_mix`, `g_r`,
`Gam_decay`, `P_WY`, `W_WY`, `T_UT`, `U_tilde`, `Q_ch`, `K_ch`, `V_ch`, `M_causal`,
`r_pos`, `Q<-`, `K->`, `S->`, `W<-`.

**Selection and development** — `E_sel`, `T_sel(n)`, `T_0`, `E_target`, `n_c`, `rho(n)`,
`rho_0`, `b_str`, `a_lr`, `sep`, `w_t`, `nu_ut`, `eps_t`, `C_t`, `S_pool`, `T_q`,
`q_cases`, `D_min`, `d_rate`, `n_par`, `cap_off`, `tau_death`, `M_mat`, `eta_ema`, `h_u`,
`h_bar_u`, `out_u`, `in_u`, `y_u`, `phi_ann`, `sigma_pos`, `sigma_mut_0`, `p_point`,
`p_dup`, `p_del`, `p_rec`, `g_growth`, `growth_schedule`, `epsilon_bif`, `sd_probe`, `bif`,
`Act`, `erank`, `s_rep`, `D_unif`, `n_death`, `n_birth`.

**Germline / ES** — `G_gen`, `P_es`, `r_rank`, `sigma_es`, `alpha_es`, `E_i`, `f_hat_i`,
`T_life`, `T_run`, `A_target`, `C_target`.

**Stream** — `theta_lag`, `kappa_lag`, `h_lag`, `T_ref`, `theta_m0`, `L_ref`, `L_lag`,
`L_theta`, `Delta_i`, `k_tok`, `p_explore`, `f_oe`, `Lambda`, `lambda_task`, `A_max`,
`tau_admit`, `s_task`, `s_H`, `H_arch`, `R_ref`, `Learn`, `Imp`, `Pri`, `eta_imp`,
`eps_arch`, `n_eval`, `J_ring`, `K_ring`, `Lhat`, `T_arch`, `T_cert`, `T_log`, `T_val`,
`n_len`, `m_mod`, `p_grp`, `tpt`, `b_types`, `depth`, `alpha_sz`, `gap`, `P_mqar`,
`distract`, `h_grid`, `w_grid`, `tf`, `n_ex`.

**Losses, prices, metrics** — `L_task`, `L_met`, `L_wire`, `L_disp`, `L_size`, `L_z`,
`lambda_wire`, `lambda_disp`, `lambda_size`, `lambda_z`, `T_set`, `L_star`, `L_rand`,
`L_birth`, `L_final`, `n_crit`, `slope_ref`, `early_slope`, `late_slope`, `CE`, `BPB`,
`PPL`, `Q_route`, `Q_B`, `pass_at_k`, `scaled_acc`, `acc_random`, `CriterionScore`,
`SlopeScore`, `FinalScore`, `BirthScore`, `FITNESS`, `w_c`, `w_s`, `w_f`, `alpha_sig`,
`n_seed`, `delta_tost`, `noise_floor`, `b_criterion`, `b_slope`, `resp`.

**[NORMATIVE — collisions resolved]** `V` is the vocabulary size only; the stacked chunk
value matrix is `V_ch`. `Delta_i` is the token excess loss; the unit contribution vector is
`delta_ut`. `E_emb` is the embedding table, `E_i` an ES perturbation, `E_sel` an event
index, `E_conn` the connectivity form. `y_u` is a utility, `y_blk` a block output, and the
regression response of §07.5 is `resp(g)`.
