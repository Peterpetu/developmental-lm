# SOMA — Specification 04: Selection

**Spec version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Normative for `soma/selection/`.
Serves constitutional requirements **C1** and **C2**. This is the novel core.

---

## 4.1 What are the individuals?

Case-based (lexicase) selection has been applied to neural-network training once at any
scale of note, and there the **individuals were whole networks**: `p = 4` model copies, each
gradient-trained on a disjoint `1/p` shard, with lexicase choosing the surviving *network*
case by case. Reported: `+0.25%` to `+0.68%` on vision at ~`4x` compute, a regression on one
architecture, and the stated limitation that the method "may not work directly with
architectures that have been extensively optimized".

Tokens are **cases**, not individuals; making tokens the individuals is thresholded
importance weighting, a different mechanism (§4.9).

**[NORMATIVE] SOMA's answer: individuals are units inside one model; cases are tokens.**

| | Whole-network lexicase | SOMA |
|---|---|---|
| Individual | a model copy | a unit |
| Population | `p = 4` | `\|U_live\|` ~ `1e2`–`1e3` per Site |
| Cost of the population | `~4x` a normal run | ~3% of Site cost (§01.4) |
| Case score | raw per-case error | learnability-weighted credit |
| What selection decides | which network survives | which units are born and die |
| Domain | vision, fixed topology | language, growing topology |

---

## 4.2 The selection event

One event is one `E_sel`, fired by the counter of (1.15) with period `T_sel(n)` (3.11).

```
Algorithm SELECT(U_l, buffer, n) -> (parents, deaths, births)

  precondition: n > 0  and  n >= max(T_0, M_mat)          # warmup guard
                |{u : live_u and not dying_u}| >= 2
                buffer holds >= D_min scored cases

  1. cases   <- buffer.sample(q_cases)                    # §4.6
  2. nu_ut   <- credit(u, t) for u in U_parent, t in cases           # (4.2)
  3. eps_t   <- per-case MAD over { u : nu_ut > 0 }, else 0          # (4.6)-(4.7)
  4. parents <- EPSILON_LEXICASE(U_parent, cases, nu, eps, w, n_par)
  5. n_death <- min( floor(rho(n) * |U_live|),
                     |{ u in U_death : u not in parents and nu_raw_u < tau_death }| )
     deaths  <- the n_death such units with lowest nu_raw
  6. n_birth <- min( n_par, floor(rho(n)*|U_live|), BUDGET_HEADROOM )
     births  <- n_birth draws uniformly from the parents multiset,
                subject to cap_off offspring per unit
  7. apply deaths (§03.4) and births (§03.2)

  postcondition: | |U_live'| - |U_live| | <= floor(rho(n) * |U_live|)
                 n_birth is computed net of n_death, so a saturated cap does not
                 permanently zero births
                 no unit with age_u <= M_mat was killed
                 no dying unit appears in parents, deaths, or the tau_death set
```

**[NORMATIVE — two named sets, v2 correction]**

```
  U_parent = { u : live_u and not dying_u }                    # may reproduce
  U_death  = { u : live_u and not dying_u and age_u > M_mat }  # may be culled
```

`EPSILON_LEXICASE` is called on `U_parent`; step 5 draws from `U_death`. v2 passed a single
`U_elig` to lexicase and then defined it with the maturity condition, while the following
sentence said parents come from "all live, non-dying units" — the two readings disagree on
whether newborns can reproduce. They can: a newborn's exclusion is from *routing* until
`m_u > 0`, not from reproduction.

**[NORMATIVE — v1 corrections]**
- *Warmup.* Spec v1's `n mod T_sel(n) == 0` fires at `n = 0`, where no token has been
  scored, no unit has utility, and the case-weight vector is all zero. The guard forbids it.
- *Extinction.* Spec v1 set `tau_death` to "the 10th percentile of `nu_hat`", killing ~10%
  of eligible units per event while births were capped at `rho_0 = 2%`. Deaths structurally
  exceeded births from the first event and the postcondition was violated immediately.
  Step 5 now caps deaths symmetrically with births.

**Termination.** Steps 1–3 are finite maps; step 4 terminates by Lemma 4.2; steps 5–7 are
finite set operations.

---

## 4.3 Case weights: learnability as an improvement signal

**[NORMATIVE]**

```
  w_t = max( 0 , L_lag(t) - L_theta(t) )                                         (4.1)
```

where `L_lag` is the loss of the **lagged self** `theta_lag` (§06.1) and `L_theta` the
current model's. `w_t` is therefore the **improvement** on case `t` since the lag — a rate,
not a level.

**[NORMATIVE — v1 correction, polarity]** Spec v1 wrote `w_t = max(0, L_theta - L_ref)` with
`L_ref` the lagged self. That has the wrong sign: a lagged self knows **less** than the
current model, so `L_theta <= L_lag` almost everywhere and `w_t` would be **identically
zero** — the same degeneracy the auditors found at `n = 0`, but permanent. (4.1) reverses the
difference.

**[NORMATIVE — v1 correction, the missing rate term]** Spec v1 additionally declared credit
must be scored on "rate of improvement over a randomised horizon `H`", gave a formula
`delta_ut(H)` requiring a per-case loss history that nothing maintained, and then defined
`nu_ct` with no horizon term at all — two `[NORMATIVE]` statements in direct contradiction,
with `H`, `H_min`, `H_max` used by nothing. The contradiction is resolved by (4.1): the
*case weight itself* is the improvement signal, so no separate horizon machinery is needed.
`H_sel`, `H_min`, `H_max` are **deleted**.

**[DERIVED]** This satisfies §01.7.2's short-horizon-bias requirement: selection pressure
follows where learning is currently happening, not where the loss is merely low.

**Relationship to token selection.** §06.2 uses a *different* reference — a frozen, better
model — because Rho-1-style token selection requires a reference that knows **more**. The
two mechanisms have different jobs and correctly different references; see `AUDIT-v1.md` A-4.

---

## 4.4 Per-unit per-case credit

```
  nu_ut = r_ut * || delta_ut ||_2 * w_t                                          (4.2)
```

- `r_ut` in `[0,1]` — routing weight, zero if `u not in R_t`;
- `delta_ut = B[ relu( gate_u (*) z_t )^2 ]` in `R^{d_model}` — the unit's additive
  contribution (renamed from `Delta_ut` to avoid collision with the token excess loss
  `Delta_i` of §06.2);
- `w_t` from (4.1).

**[DERIVED]** `nu_ut = 0` for every unit not routed to `t`. This is why `eps_t` must be
computed over non-zero credits only (§4.5).

**Exact leave-one-out.** **[NORMATIVE — v1 correction]** Spec v1 claimed the counterfactual
`L(t | m_u = 0)` is "obtainable by a single additional Site evaluation". It is not: it
requires a forward pass from the Site's layer through every subsequent block and the output
head. The audit path therefore evaluates a **partial forward from layer `l` to the head** on
a random `1%` of cases; a full forward is strictly more faithful and is what the
implementation does, because the cost is one forward per **distinct unit** in the sample, not
one per pair.

**[NORMATIVE — the pairs are `(u, t)`, and the proxy is (4.2) entire; v3.1, audit A-67]**
`nu_ut` is indexed by unit **and** case, so an audit pair is one unit on one case:

```
  proxy(u, t) = r_ut * ||delta_ut||_2 * w_t                       (4.2)
  exact(u, t) = L_t( m_u = 0 ) - L_t
```

v3.0's implementation compared, per *unit*, the two-factor sum `sum_t r_ut ||delta_ut||`
against the change in the **mean** probe loss. `w_t` was absent, so the audited quantity was
not the one selection uses; and both terms were aggregated over all probe tokens, so a single
pair mixed every case together and the per-case effect was diluted by `1/(B T_ctx)`. The
consequence was a reported `rho = 0.155` on a healthy lifetime — §04.10 firing on a
comparison that was not testing it.

**[NORMATIVE — the target is a magnitude; v3.1, audit A-68]** The Spearman is computed
against `|L_t(m_u = 0) - L_t|`. (4.2) is a product of three **non-negative** factors, so
`nu_ut >= 0` identically; a rank correlation between a non-negative statistic and a target
that is negative on about half its sample is bounded near zero *whatever the proxy is*. A
`0.5` bar against a signed target is unsatisfiable by construction — the same defect class as
the three unfireable thresholds recorded at A-36, A-58 and A-61, in the opposite direction.

Both correlations are computed and both are logged. Measured on **472 pairs** from an M-nano
lifetime (steps 100–611, `|U_live|` 64→240):

| statistic | value |
|---|---|
| `rho_signed` — (4.2) vs `L_t(m_u=0) - L_t` | `+0.001` |
| `rho_magnitude` — (4.2) vs `\|L_t(m_u=0) - L_t\|` | `+0.601` |
| fraction of cases where removing the unit **lowered** the loss | `0.52` |

The reading is exact and is a **design property, not a defect**: (4.2) estimates *how much* a
unit changes a case and carries no information about *whether* the change helps. §03.4.1's
`y_u` is likewise a magnitude of change. SOMA therefore selects on **influence**, and benefit
is the gradient's job — which is the same division of labour §01.7.2 states between the
lifetime and generation rates. `rho_signed` stays in the log because a large positive value
would be news.

**[OPEN — the bar is close]** `rho_magnitude = 0.601` clears `0.5` by `0.10` on one M-nano
measurement at `n <= 700`. This criterion was **re-specified after seeing that number**, which
is exactly the situation pre-registration exists to prevent; it is recorded here so a reader
can discount it accordingly. The M1 run re-measures it on an independent lifetime, and the
value there — not this one — is what §04.10 is adjudicated on.

**[NORMATIVE]** If the magnitude correlation falls below `0.5` on a sample of at least **64
pairs**, the audit fraction **doubles**, capped at 10%.

**[NORMATIVE — the audit's cost is bounded at 128 units per event; v3.1]** The counterfactual
costs one forward per **distinct unit** in the sample. At M1 the `10%` cap means
`0.10 * 8 * 512 = 410` cases times `k_route = 8`, i.e. up to **3280 forwards on a `(8, 512)`
probe per selection event** — longer than the training between events, so the doubling rule
would make the audit the run. At most `128` distinct units are switched off per event.

`128` is twice `min_pairs`, so a single event still over-supplies the 64 pairs the Spearman
needs and the doubling rule keeps its effect on the *estimate*; only its effect on the *cost*
is capped. The retained units are a prefix of an already-uniform sample, so the bound
introduces no bias in which units are audited.

**[NORMATIVE — the pairs accumulate across events, v3.1]** One selection event supplies far
fewer than 64 usable pairs: `1%` of `q_cases` is 2 at smoke scale, and units carrying no probe
traffic are skipped rather than scored zero (a block of ties at zero would inflate the
correlation with pairs that measure nothing). The pairs are therefore **accumulated across
events** until the minimum is reached, then evaluated and the buffer cleared. Evaluating a
2-pair sample instead gives a correlation whose sampling distribution is nearly uniform on
`[-1,1]`: it would trip the doubling rule at random and §04.10's falsifier would fire on noise.
v3.0 specified the minimum without saying where the pairs come from, and the audit consequently
reported `n_pairs = 0` for an entire lifetime (audit A-59).

Aggregate utility for death is (3.5)–(3.8) in §03.4.

**[NORMATIVE — `tau_death`]** An **absolute** threshold on the **un-normalised**
bias-corrected utility `nu_raw_u = nu_u / (1 - eta_ema^{age_u})`, calibrated **once** at the
first selection event as the 10th percentile of `nu_raw` over `U_death`, and thereafter
**fixed** for the lifetime.

Spec v1 recomputed a percentile every event, guaranteeing a constant 10% kill rate whether or
not any unit was unfit. v2 made it absolute but stated it on `nu_hat`, which (3.8) divides by
the population median — so `nu_hat` has median exactly 1 at every step **by construction**
and a fixed threshold on it still selects a fixed fraction. Only the un-normalised quantity
can carry an absolute threshold.

**[NORMATIVE — the deliberate asymmetry]** Births are decided **case-by-case**; deaths by
**aggregate** utility plus failure to be selected. A unit that is best at nothing is
expendable, so aggregation is safe for removal; it is not safe for reproduction, which is
exactly where averaging destroys specialists.

---

## 4.5 Semi-dynamic epsilon-lexicase

**[NORMATIVE]** `eps_t` is the median absolute deviation **over units with non-zero credit
on case `t`**:

```
  C_t   = { u : nu_ut > 0 }                                                      (4.6)
  eps_t = median_{u in C_t} | nu_ut - median_{v in C_t} nu_vt |   if C_t != {}
        = 0                                                       if C_t == {}    (4.7)
```

**[NORMATIVE — v2 correction]** `C_t` is **empty** whenever `w_t = 0`, which (4.1) makes
common: any case on which the model has not improved since the lag. A median over the empty
set is undefined, so v2's `eps_t` was NaN on those cases and the precondition `eps_t >= 0`
could not be met. (4.7) defines `eps_t = 0` there, which reduces the filter to strict
lexicase on a case that carries no learnability signal — the correct behaviour, since such a
case should not widen the pool.

**[NORMATIVE — v1 correction, the defect that would have made the mechanism inert]**
Spec v1 computed the MAD over the **whole population**. Since only `k_route ~ 8` of
`~1e2`–`1e3` units are routed to any token, `nu_ut = 0` for the overwhelming majority, so
`median_v nu_vt = 0` and `MAD_t = 0` **identically**. Every `eps_t` would have been zero,
the filter would have degenerated to strict `nu_ut >= elite`, the pool would collapse to one
unit after the first case, and Proposition 4.3 would not hold. ε-lexicase — the architecture's
central novel mechanism — would have been silently inert. (4.6)–(4.7) fix this.

```
Algorithm EPSILON_LEXICASE(U_parent, cases, nu, eps, w, n_par) -> parents
  precondition: |cases| >= D_min >= 8 ; |U_parent| >= 2 ; eps_t >= 0

  parents <- empty multiset
  repeat n_par times:
      S_pool <- U_parent
      T_q    <- SHUFFLE(cases, weights = w)
      while |T_q| >= 1 and |S_pool| > 1:
          t      <- pop_front(T_q)
          elite  <- max_{u in S_pool} nu_ut                  # on the CURRENT pool
          S_pool <- { u in S_pool : nu_ut >= elite - eps_t }
      parents <- parents + { UNIFORM_RANDOM(S_pool) }
  return parents
```

**[NORMATIVE]** The loop guard is `|T_q| >= 1`, so the final case is used; spec v1's
`> 1` discarded it.

**[NORMATIVE]** `SHUFFLE(cases, weights = w)` draws a permutation with probability
proportional to `w_t` at each draw without replacement, and **falls back to a uniform
permutation when `sum_t w_t = 0`** — reachable whenever the model has improved on no sampled
case since the lag.

**Lemma 4.1 (elite guarantee).** Every returned parent is within `eps_{t_1}` of the pool
maximum on the first case used to select it.
*Proof.* The first iteration filters `S_pool` to exactly that set, and `S_pool` only shrinks.
QED

**Lemma 4.2 (termination).** The inner loop terminates in at most `|cases|` iterations.
*Proof.* `pop_front` strictly decreases `|T_q|`. QED

**Proposition 4.3 (specialist preservation).** The reachable parent set is the set of
ε-Pareto-set boundary points of `U_parent` in the `|cases|`-dimensional credit space; a unit
best on a single rare case survives, where mean-utility selection requires its mean to
exceed a threshold. **[OPEN]** The supporting evidence is an abstract-only reading of a
paper distinct from the down-sampling source; §11 C-11 carries the `[OPEN]` marker.

### 4.5.1 Complexity

With `q_cases = |cases|`, `P = |U_parent|`:
- building `nu`: `O(q_cases * k_route)` — **sparse**, since `nu_ut = 0` off `R_t`;
- `eps_t`: `O(q_cases * |C_t| log|C_t|)`, `|C_t| <= k_route`;
- one parent selection: worst case `O(q_cases * P)`; in practice the pool winnows quickly.

**[NORMATIVE]** Measured selection wall-clock is a required log field. If it exceeds **2% of
step time averaged over a 1000-step window**, then `d_rate <- max(0.02, 0.7 * d_rate)`.

---

## 4.6 Down-sampling and the buffer

**[NORMATIVE]** The buffer is a **reservoir sample** of fixed size
`q_cases = max(D_min, round(d_rate * B_tok))` (§01.10), refilled continuously and reset after
each selection event.

**[NORMATIVE — v2 correction]** v2 set `q_cases = ceil(D_min / d_rate)`, which is
**inversely** proportional to `d_rate`. §4.5.1's cost governor reduces `d_rate` on a trip, so
each trip *raised* the case count and therefore the cost it was meant to bound — an unstable
positive feedback running `80 -> 115 -> 163 -> ... -> 400` at the floor. It also inverted the
semantics of the cited evidence, in which a lower rate means *fewer* cases.
Per (case, unit) it stores only scalars: `||delta_ut||_2` and `r_ut`. Per case it stores the
token id and **`w_t` itself**, evaluated at the moment the case was scored.

**[NORMATIVE — `w_t` is computed at push time, not at the selection event]** v3.0 required the
buffer to store each case's context window and target so that `L_lag` could be re-run at the
selection event. That is both expensive and semantically worse:

*Expensive.* `U` is per-Site (§01.1), so the store is `L` times the figure v3.0 quoted:
`12 * 6554 * 513 * 4 B = 161 MB` at M1, not `13.4 MB`, and §05.10's budget was written for one
Site's worth.

*Semantically worse.* `nu_ut = r_ut * ||delta_ut||_2 * w_t` (4.2) multiplies three quantities.
`r_ut` and `delta_ut` are necessarily measured **when the token was routed**; re-evaluating
`w_t` at the selection event would multiply them by an improvement measured up to `T_sel`
steps later, against a `theta_lag` that had moved in the meantime. Evaluating all three at the
same moment on the same token is what makes (4.2) a coherent product.

The implementation therefore runs the `theta_lag` forward on the **rows being pushed** each
step — a `d_rate` fraction of the batch, which is exactly the 10% §06.1.2 budgets — and stores
the resulting scalar. Realised footprint: `q_cases * (2 k_route + 2)` floats per Site,
**5.1 MB** across all 12 Sites at M1.

**[NORMATIVE — v1 correction]** Spec v1 said the buffer "holds every scored token from the
last `T_sel` steps" and then down-sampled. At M1 that is `T_sel * B_tok * k_route` credit
records — tens of gigabytes, absent from the memory budget. Reservoir sampling bounds it to
`q_cases * k_route * 3` floats ≈ 8 KB, which §05.10 records.

**[EMPIRICAL]** Performance is statistically flat for down-sample rates from `0.25` to
`0.02` and erratic at `0.01`. Default `d_rate = 0.10`, `D_min = 8`.

**[NORMATIVE — the justification, stated correctly]** Down-sampling is justified because it
**allows more individuals to be evaluated per unit of compute**. It is *not* justified by a
"changing environment" effect: the source's own experiments give evidence **against** that
hypothesis and against a generalisation explanation. We adopt the mechanism together with
the source's own explanation.

**Guards.**
- *Hyperselection.* Offspring per unit per event are capped at
  `cap_off = max(2, ceil(0.5 * n_birth_max))` with `n_birth_max = floor(rho_0 * U_cap)`, so
  the cap is **reachable**. v2 fixed `cap_off = 6` at M1 while `n_birth <= floor(0.02|U|) <= 5`,
  so no unit could ever exceed it from at most 5 draws: §4.10's falsifier ("cap binds in more
  than 20% of events") and §07.6's hit-rate log were identically zero by arithmetic. The
  fraction of events where the cap binds is logged; persistent binding means monoculture.
- *Combinatorial ceiling.* With `q_cases` cases at most `q_cases!` credit orderings are
  distinguishable; `D_min = 8` gives `40320`, comfortably above realised diversity.
- *Headroom.* `BUDGET_HEADROOM` from (1.14) caps births under both of its bounds —
  effective population against `U_cap`, occupied slots against `U_max` — evaluated net of the
  deaths applied in the same event.

---

## 4.7 Why this is not "MoE plus pruning"

| Property | Fine-grained MoE + pruning | SOMA |
|---|---|---|
| Heredity | none | lineage records, parents, clades |
| Fitness aggregation | mean utility / magnitude | **case-by-case for reproduction** |
| Variation operators | none | duplication, mutation, **crossover** |
| Capacity | fixed hyperparameter | bounded by `U_cap` + death rule |
| Mutation rate | n/a | **inversely proportional to fitness** |
| Persistence | dies with the run | archetypes enter the Inventory |

This is the distinction between borrowing evolution's *arithmetic* — a group-relative
baseline over a transient sample — and implementing its *process*.

---

## 4.8 Constants

All live in the canonical table of §02.2.3. Derived: `q_cases = max(D_min, round(d_rate * B_tok))`
(§01.10), `cap_off` (§4.6), `tau_death` (§4.4), `T_0` (1.17).

---

## 4.9 Relation to token selection

| | Token selection (§06.2) | Unit selection (this section) |
|---|---|---|
| Individuals | none | units |
| Cases | none | tokens |
| Operation | global threshold on a scalar | case-by-case filtering |
| Aggregation | ranks tokens by one number | never aggregates |
| Reference | frozen **better** model (M0) | lagged **worse** self |
| Effect | masks the loss | decides birth and death |
| Evidence | measured LM gains | none — the research bet |

They compose: token selection gates in *token space*, the event counter gates in *time*, and
unit selection operates on the population.

---

## 4.10 Falsification

- **A2** — replace ε-lexicase with mean-utility selection for **both** parents and deaths
  (parents = top-`n_par` by `nu_hat`). Spec v1's A2 specified only a death rule, leaving the
  ablated arm with no way to produce births.
- **[NORMATIVE — the case-win diagnostic]** A "case win" is *membership of the surviving pool
  after the first case of a selection event*. It is computed as a **diagnostic independent of
  the selection operator**, so it exists in **both** arms of A2; spec v1 defined it only
  inside lexicase, so K3's comparison had data in one arm only.
- If the case-win histogram under ε-lexicase is equivalent to that under mean selection under
  the declared TOST margin (§08.5), specialists are not being preserved and K3 fires.
- If the hyperselection cap binds in more than 20% of events, the population has collapsed to
  a monoculture.
- If Spearman correlation between (4.2) and the **magnitude** of exact leave-one-out
  (§04.4, audit A-68) falls below `0.5` on `>= 64`
  pairs, the credit proxy is invalid and every selection decision taken with it is suspect.
