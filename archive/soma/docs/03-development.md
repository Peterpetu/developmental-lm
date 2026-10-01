# SOMA — Specification 03: Development — Birth, Death, and Canalization

**Spec version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Normative for `soma/development/`.
Serves constitutional requirement **C1**.

---

## 3.1 What development is, and what it deliberately is not

Development is the change of the unit population `U_l` and of gene sets `g_u` over a single
lifetime. It runs at rate `r_2`, it is discrete, and it is therefore the exclusive province
of selection.

**[NORMATIVE — a negative result we obey]** Multi-step *generative* growth processes, in
which a program is unrolled for developmental steps to produce a network, were tested
against a one-shot decoding baseline and showed **no benefit**; the same line reports that
such programs often contain more parameters than the networks they produce, inverting the
bottleneck. SOMA therefore implements **no** temporally-unrolled morphogenesis program.
Development here is birth and death of units under selection.

---

## 3.2 Birth

**[NORMATIVE]** Units are created by **duplication of a selected parent**, optionally with a
second parent for recombination.

```
Algorithm BIRTH(u_p1, u_p2 | None, n) -> u_child
  precondition: u_p1 live and not dying ; BUDGET_HEADROOM >= 1 ;
                a free slot exists in the grid

  slot_child <- FREE_SLOT_NEAREST(slot(u_p1))       # grid-L1 nearest free slot
  p_child    <- normalize( p_{u_p1} + xi ),  xi ~ N(0, sigma_pos^2 I_{m_dim})
  g_child    <- MUTATE(g_{u_p1}, nu_hat_{u_p1})          if u_p2 is None
                CROSSOVER(g_{u_p1}, g_{u_p2})            otherwise
  # NOTE: the child does NOT inherit key-table rows -- see the v3.1 correction below
  m_child    <- 0
  age_child  <- 0
  nu_child   <- 0
  L_child    <- (run, G_gen, id(u_p1), id(u_p2) | None, n)
  live[slot_child] <- True

  postcondition: m_child = 0, so by Theorem 3.1a the phenotype function is unchanged
```

**[NORMATIVE — v1 corrections]** Three defects are repaired here.
*(a) Slot assignment.* Spec v1's `BIRTH` never assigned a routing address at all, so nothing
determined where in the grid a child landed. `FREE_SLOT_NEAREST` places the child at the
grid-L1 nearest free slot.

**[NORMATIVE — v3.1 correction: the child does not inherit key rows, and could not]**
v2 and v3.0 stated that the child "inherits its parent's key-table rows as its initial
addressing". **This is impossible.** `K_unit1` has shape `side x (d_key/2)` and row `i1` is
shared by all `side` slots in grid row `i1` (§05.4), so writing the parent's row into the
child's row changes `s1` for every slot in that row — hence `I1`, hence **every other unit's
`r_ut`**. That is precisely the failure mode §05.4's own correction identifies, and it would
make Theorem 3.1a false at the moment of birth rather than merely after the anneal. The clause
is deleted.

**[NORMATIVE — what actually makes a child resemble its parent]** Grid adjacency confers **no
routing similarity**: adjacent rows carry independent random keys, so `FREE_SLOT_NEAREST`
does not give the child a token distribution like its parent's, and v2's justification for it
was wrong. Two mechanisms do the work, and they are the right ones:

1. **Function** is inherited through `p_child = normalize(p_parent + xi)` and
   `g_child = MUTATE(g_parent)`, which together determine
   `gate_u = phi(p_u) (*) psi (*) 1_{g_u}`. A child computes very nearly its parent's
   function.
2. **Routing is deliberately fresh.** A unit with its parent's function but a *different*
   token distribution is exactly what §02.4 means by differentiated siblings; inheriting both
   would produce a duplicate, not an offspring, and selection would have nothing to choose
   between.

`FREE_SLOT_NEAREST`'s remaining purpose is bookkeeping: it keeps a clade in a contiguous block
of slots, which makes §08.1.1's `Q_route` community structure spatially interpretable and
keeps the free-slot search local. That is a real but modest benefit, and it is the only one
claimed.

**[NORMATIVE — the anneal's residual discontinuity, and why it is already covered]** While
`m_child = 0` the child is inactive, so Theorem 3.1a holds exactly. At the first anneal step
`m_child > 0` and the child becomes active; if its grid row or column previously contained no
active slot, that row flips from `-inf` to finite and `I1` changes for every token. This is a
genuine discontinuity that the mask anneal does not smooth. It is **not** left undetected: the
bifurcation alarm (1.21) compares the cohort active against the cohort forced to `m_u = 0`
after the anneal completes, and forcing `m_u = 0` also clears the row flag — so the row-flip
effect is inside the quantity `bif` measures, and §03.7's "alarm fires on more than 1% of
birth events" is the falsifier that covers it.
*(b) Two parents.* Spec v1's signature took one parent while recombination needs two, and
`L_u` could record only one. Both are extended.
*(c) Inherited utility.* Spec v1 set `nu_child = nu_parent`, which invalidates the bias
correction (4.6) (whose derivation assumes initialisation at zero) and inflates young
utilities by up to `100x`. `nu_child = 0`; `M_mat` provides the protection instead.

**[EMPIRICAL — why duplication]** Across a systematic comparison of growth operators —
direct duplication, learned linear maps, zero-initialisation, masked random initialisation
— **whole-module duplication won**. SOMA applies it at unit granularity.

**[DERIVED — the clade prediction]** `p_child = normalize(p_parent + xi)` places offspring
near their parent on the sphere. Under `L_wire` a lineage occupies a compact region, shares
gene expression, and incurs low intra-lineage wiring cost. Modules should therefore arise as
**clades** rather than as a designed partitioning. §07.6 logs within- versus between-clade
mean distance and §3.7 makes failure of this prediction a stated falsifier.

### 3.2.1 Mutation

```
Algorithm MUTATE(g, nu_hat) -> g'
  sigma_mut <- sigma_mut_0 / (1 + nu_hat)                                        (3.1)
  kind      <- ONE categorical draw from (p_point, p_dup, p_del)      # simplex
  g'        <- copy(g)
  with probability min(1, sigma_mut / sigma_mut_0):
      if kind == point:  replace a uniformly chosen i in g' by neighbour(i)   (§02.2.2)
      if kind == dup:    replace a uniformly chosen i in g' by a gene expressed by the
                         live unit nearest to p_child; if no such unit, by neighbour(i)
      if kind == del:    replace a uniformly chosen i in g' by a uniform draw from
                         {0..N_G-1} \ g'
  postcondition: |g'| = k_g exactly ; g' subset of {0..N_G-1} ; no duplicates
```

**[NORMATIVE — v1 corrections]** Spec v1 (i) drew three independent Bernoullis while calling
the probabilities a simplex, so zero to three mutations could occur; (ii) had a postcondition
`1 <= |g'| <= k_g` contradicting its own "truncated or padded to `|g'| = k_g`" with no
padding source; (iii) left `neighbour` undefined; (iv) gave the "nearest unit" rule no
fallback. All four are fixed: exactly one categorical draw, size preserved by *replacement*
rather than add/remove, `neighbour` defined geometrically in §02.2.2, and an explicit
fallback.

**[NORMATIVE — somatic hypermutation]** (3.1) makes the mutation *probability* inversely
proportional to parent fitness: high-fitness lineages are explored finely, low-fitness
lineages coarsely. This is the affinity-maturation rule from adaptive immunity and has no
analogue in gradient training, which has a step size but no notion of searching harder
around bad solutions.

```
Algorithm CROSSOVER(g1, g2) -> g'
  pool <- g1 union g2
  g'   <- k_g elements sampled from pool without replacement, each element weighted
          1.0 if in g1 & g2, else 0.5
  postcondition: |g'| = k_g
```

**[OURS]** Crossover is available to no gradient method. Ablation **A11** tests it.

---

## 3.3 Birth does not damage the phenotype

**Theorem 3.1a (exact function preservation at unit birth).** With `m_child = 0` and the
routing exclusion of §05.4, the Site output (5.20) is unchanged by the addition of
`u_child`, for any values of the child's `p`, `g`, or key rows, up to floating-point
reduction order.

*Proof.* The child contributes to (5.20) only through `r_ut * m_u * B(gate_u (*) z)`. With
`m_child = 0` the summand is exactly zero. The child is additionally excluded from `cand`
in (5.18) while `m_u = 0`, so it neither displaces another unit from the top-`k_route`
selection nor enters the softmax normalisation — hence every other unit's `r_ut` is also
unchanged. `W_shared x_t` lies outside the sum (§05.5) and is untouched. QED

**[NORMATIVE — v1 corrections]** Spec v1's proof (i) relied on a parenthetical
"(implementation requirement)" for softmax exclusion that §05.4 did not implement,
(ii) did not exclude the newborn from **top-k selection**, so it could displace a real unit,
and (iii) rested on a key-table shape `sqrt(|U|)` that changed with every birth, re-addressing
every unit. §05.4's fixed slot grid and `-inf` masking make the theorem true.

**Theorem 3.1b (function preservation at capacity growth).** For width, depth, or head
growth with a per-coordinate mask `c` and masked normalisation statistics, the block output
is unchanged at `c = 0`.

```
  width:  y' = c (*) sigma_act(W' x) = [ y ; 0 ]                                 (3.2)
  LN:     mu'  = (x' . c)/sum(c) ,   var' = ((x' - mu')^2 . c)/sum(c)            (3.3)
  RMS:    var' = ((x')^2 . c)/sum(c)                                  [OURS]     (3.4)
  MHA:    mask the VALUE path only: v'_i = 0  =>  o'_i = 0
  depth:  x_n = c (*) Layer_n(x_{n-1}) + (1 - c) (*) x_{n-1}
```

**[NORMATIVE — v1 correction]** Spec v1 stated a single Theorem 3.1 whose proof conflated
the **unit mask** `m_u` (a scalar per unit, changing no vector width) with the
**coordinate mask** `c` (a per-coordinate vector for width growth), invoking a normalisation
lemma about a different operation. The two are now separate theorems with separate proofs.

**[OPEN]** The source operator's use at `1.01e11` parameters is **[UNVERIFIED]** (§11 C-5)
and is the sole support for expecting this operator to hold at scale. Deferral criterion: a
measured function-preservation check at every growth event, which §03.3's mandatory test
performs at our scale; no claim is made beyond it.

**[OURS]** (3.4) is our extension — the source operator specifies LayerNorm only. The
derivation is immediate (RMSNorm omits the mean, so only the masked second moment needs
correction). **Mandatory test:** `tests/unit/test_msg.py` asserts (3.4) reduces to (3.3) at
zero mean, and that the grown module's output equals the ungrown module's to `1e-6` at
`c = 0` for random `W'` including adversarially large values.

**[NORMATIVE]** Claims of *bitwise* equality are withdrawn: adding a zero-valued summand
changes floating-point reduction order. The claim is equality **up to reduction order**, and
the test tolerance is `1e-6`.

**Mask annealing.** `m_u` rises linearly `0 -> 1` over `phi_ann` steps.

**[NORMATIVE]** `phi_ann = min( 5000, max(50, floor(0.02 * N_step)) )` — one formula,
replacing two inconsistent ones in spec v1. At the §01.9 values (`N_step` 2442 / 12208) this
is **50** steps at M-nano and **244** at M1.

**[NORMATIVE — the anneal must fit inside a selection interval]** `phi_ann < T_0` is asserted
at config validation, so a unit born at one event completes its anneal (or its death) before
the next. At M-nano `50 < 99`; at M1 `244 < 492`. v1 scheduled `phi_ann = 5000` against
`T_sel = 2000`, which is exactly the defect §3.4's dying-unit exclusion was added to contain;
the assertion prevents the configuration from recurring.

**[EMPIRICAL]** The source ablation finds all anneal lengths above 500 stable at its scale.
M-nano's 50 is far below that; its stability is therefore **verified by the bifurcation
alarm** rather than assumed, and a failure raises `rho` reduction (§01.7.1).

**[DERIVED — growth as an instrument]** Because birth is exactly function-preserving,
any discontinuity in `L_task` across a birth is not an operator artefact; it is the
bifurcation signal of §01.7.1.

---

## 3.4 Death

```
Algorithm DEATH(u, n)
  precondition: live_u and not dying_u ; age_u > M_mat ;
                u not selected as a parent this event ; nu_hat_u < tau_death
  1. dying_u <- True ; schedule m_u to decay linearly to 0 over phi_ann steps
  2. while dying: u is excluded from cand (5.18), from parents, from deaths,
     from n_par, and from the tau_death calibration set
  3. on m_u reaching 0: live_u <- False ; zero p_u and its optimizer state;
     retain L_u in the lineage log
  postcondition: |U_live| decreased by 1 ; the freed slot is reusable
```

**[NORMATIVE — v1 correction]** Spec v1's death step transferred a unit's average
contribution "into the bias of each consumer". **The model has no biases** (§05.1), and
"consumer of `u`" is not an object in this architecture — units contribute additively
through the shared `B`. The step is **deleted**. Death is therefore *approximately*, not
exactly, function-preserving; the mask anneal is what makes the change gradual.

**[NORMATIVE]** Consequently the **bifurcation alarm applies to births only** (§01.7.1).
Spec v1 applied a `<1%` threshold to "every structural event" while its own death operator
could not preserve the function, so a death-induced discontinuity would have been
misattributed to a bifurcation.

**[NORMATIVE — v1 correction, dying units]** Spec v1 scheduled death over `phi_ann = 5000`
steps while `T_sel = 2000`, so a dying unit remained in `U` across two selection events and
could be selected as a parent. Step 2's exclusion set fixes this.

### 3.4.1 Utility

```
  h_u      = sum_t r_ut ||delta_ut||_2 / max(eps, sum_t r_ut)                    (3.5)
  h_bar_u  = eta_ema * h_bar_u + (1 - eta_ema) * h_u        # per-unit EMA, init 0  (3.5b)
  out_u   = sum_{g in g_u} phi_g(p_u) ||b_g||_1
  in_u    = sum_{g in g_u} phi_g(p_u) ||a_g||_1
  y_u      = |h_u - h_bar_u| * out_u / max(eps, in_u)                            (3.6)
  nu_u   <- eta_ema * nu_u + (1 - eta_ema) * y_u                                 (3.7)
  nu_hat_u= ( nu_u / (1 - eta_ema^{age_u}) ) / median_{v live, mature} (...)      (3.8)
```

**[NORMATIVE — v2 correction]** `h_bar_u` was used in (3.6) and listed in the symbol table
but **defined nowhere** in v2, so the entire death rule was not implementable. (3.5b) defines
it as a per-unit EMA of `h_u` with the same decay as the utility, initialised to zero — so
`y_u` measures a unit's *deviation from its own recent contribution*, which is what makes it
a novelty signal rather than a magnitude signal.

**[NORMATIVE — v1 correction]** Spec v1's `y_u = (|h_u - h_bar_u| * sum_out|w|)/(sum_in|w|)`
was **not computable**: a unit owns no weights (it owns `p_u` and a gene index set into the
shared `a_g, b_g`), `h_u` was a vector with no norm specified, and the result had no
normalisation, so it contaminated (3.1)'s `sigma_mut_0/(1 + nu_hat)` with an arbitrary
activation scale. (3.5)–(3.8) define every term and normalise by the population median.

`nu_hat_u = 0` at `age_u = 0` by definition (the expression is `0/0` otherwise).

**[NORMATIVE — v2 correction, `tau_death` must be calibrated on the un-normalised utility]**
(3.8) divides by the population median, so `nu_hat` has median exactly 1 at every step **by
construction**. A threshold "calibrated once as the 10th percentile of `nu_hat`" is therefore
absolute only in a coordinate that is renormalised each step, and would still select about
10% of the population at every event — exactly the extinction behaviour v1's correction note
claimed to have removed. `tau_death` is calibrated on the **un-normalised** bias-corrected
utility `nu_u / (1 - eta_ema^{age_u})` and compared against that same quantity.

**[NORMATIVE — why death is mandatory]** The peer-reviewed finding is that *"plasticity is
maintained indefinitely only by algorithms that continually inject diversity into the
network"*; networks without it fall toward linear-model performance, with dying units,
growing weight magnitude, and falling effective rank. Adam-family optimizers suffer
catastrophic plasticity loss in this setting. SOMA trains against a non-stationary stream,
so the birth/death cycle is load-bearing.

**[NORMATIVE]** Effective rank is logged every `10 * T_log` steps:

```
  p_i        = s_i / ||s||_1                     s_i = singular values of Act
  erank(Act) = exp( - sum_i p_i log p_i )                                        (3.9)
```

`Act` in `R^{2048 x N_G}` is the **post-activation Site inner state** `relu(A x)^2` on a
**fixed** probe batch of 2048 tokens held constant for the whole run.

**[NORMATIVE — the SVD runs on CPU]** `Act` is `2048 x 2048` at both configurations, and
cuSOLVER's batched Jacobi driver does not reliably converge on it: it emits a convergence
warning, falls back, and the resulting **asynchronous** error state was observed to corrupt
*later, unrelated* kernels — producing `NaN` in a subsequent forward pass while **every
parameter and buffer remained finite**, and disappearing entirely under
`CUDA_LAUNCH_BLOCKING=1`. `erank` is a diagnostic computed once per `10 * T_log` steps, so the
host transfer is affordable where the corruption is not; LAPACK's `gesdd` on CPU is
deterministic. See `AUDIT-v1.md` A-55.

**[NORMATIVE — a diagnostic reports, it does not abort]** If `Act` for a Site contains a
non-finite entry, `erank` for that Site is **`NaN`** and the run continues. It is not an
exception. `linalg.svdvals` raises `_LinAlgError` on non-finite input, and that exception
carries strictly less information than the `NaN` does: `erank` is logged per Site, so a
`NaN` names the Site, the step, and leaves the run alive to record what happens next,
whereas the exception says only "something, somewhere, was not finite" and destroys the
evidence by ending the run. §07.4.3's guards are the mechanisms that abort; a plasticity
probe is not one of them. Observed: an M-nano lifetime ended at step 500 inside this SVD
with every parameter and every gradient finite (audit A-63).

Spec v1 left `Act`
unspecified, collided its name with `phi`, and logged every `T_log = 50` steps, which costs
12 SVDs of a `2048 x N_G` matrix at that cadence; the coarser cadence is costed in §07.8.

---

## 3.5 Canalization

```
  rho(n)   = rho_0 * exp( - n / n_c )                                            (3.10)
  T_sel(n) = T_0 * ( 1 + n / n_c )                                              (3.11)
  b_str(n) = rho(n) / T_sel(n)                                                   (3.12)
```

with `T_0` **derived** from a target event count by (1.17), not asserted.

**Lemma 3.2.** `b_str(n)` is summable.
*Proof.* `b_str(n) <= (rho_0/T_0) e^{-n/n_c}`; a geometric series converges. QED

**[NORMATIVE]** By §01.7.1 this is the intended behaviour — the slow variable freezes. The
"Phase I / Phase II" framing of spec v1 is **withdrawn**: it described `b_str` as
approximately constant early, whereas (3.10) makes it decay from the first step (a factor
`e^{-1} = 0.37` by `n = n_c`).

**[EMPIRICAL — the window]** Networks become stable to SGD noise at approximately 1.5%,
3%, 16% and 20% of training across four studied settings; **the window widens with scale and
difficulty**. Two consequences: structural change is only worth doing early; and allocating a
large fraction of a *small* run to development is defensible because the production-scale
window is expected to be wider.

**[NORMATIVE — the extrapolation is flagged]** These measurements are from vision CNNs.
**No critical-period study exists for transformers or language models.** Our use of the
window is an extrapolation across architecture family, strengthened but not established by
the finding that critical periods appear in deep *linear* networks.

**[NORMATIVE — unit conversion]** The cited replacement rate is `1e-4` **per step**;
`rho_0 = 0.02` here is a fraction **per selection event**, i.e. `rho_0 / T_0 = 5.3e-5` per
step at M1. The two are within a factor of two once converted, and the conversion is stated
because the plasticity argument (K8) rests on turnover volume.

### 3.5.1 Capacity growth

**[NORMATIVE]** `GROW` is defined. At M-nano and M1 the **only** grown dimension is the unit
population.

```
Algorithm GROW(l, g_growth, n)
  precondition: n in growth_schedule ; BUDGET_HEADROOM > 0
  r_left <- number of growth events remaining in growth_schedule, including this one
  n_new  <- min( floor((g_growth - 1) * |U_live(l)|),
                 floor( BUDGET_HEADROOM / r_left ) )
  for j in 1..n_new:
      parent <- sampled from U_live(l) with probability proportional to nu_hat
      BIRTH(parent, None, n)
  postcondition: every new unit has m_u = 0, so Theorem 3.1a applies to each
```

**[NORMATIVE]** `growth_schedule` is an explicit list of steps. Defaults:
`{0.35}` for 1 event, `{0.10, 0.20, 0.35}` for 3, and
`{0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40} * N_step` for 8. Ablation **A8** compares
`{1, 3, 8}` events, all under the same `U_cap`.

**[NORMATIVE — v2 correction]** v2 sized a growth event as
`min(floor((g_growth-1)|U_live|), BUDGET_HEADROOM)`, which with `g_growth = 4` requests
`3|U_live|` new units and therefore **saturates `U_cap` on the very first event** at both
configurations. The 2nd and 3rd scheduled events then violate `GROW`'s own precondition, so
all three arms of ablation **A8** would have produced an identical trajectory and the
ablation would have been inert. Dividing the remaining headroom by the number of remaining
events makes the schedule length actually matter.

**[NORMATIVE — v1 correction]** Spec v1 gave a token-count law
`log10(D_grow) = 0.88 log10(N_par) + 163.27/log10(C) - 5.74` and a factor `g_growth = 4`,
but no algorithm connecting them, and never named the grown dimension. Numerically, at M1
(`N_par ~ 2.4e7`, `D_tok = 8e8`) that law yields `D_grow ~ 2e10` tokens — **25× the entire
training budget** — so no growth event would ever have occurred and §03.3's masking
machinery and ablation A8 would have been dead code. The law is retained for **M4 and above
only**; at M-nano and M1 the explicit schedule applies.

**[NORMATIVE — a warning we act on]** The same source found **one-shot growth beat gradual
multi-stage growth**, and that the measured speedup **decays** with longer training
(53.1% at 4e11 tokens vs 31.0% at 7e11 at fixed model size), while its claim of persistence
to 8e12 tokens is an extrapolation of a log fit its own two measured points contradict in
direction. SOMA therefore defaults to few growth events plus many selection events, treats
them as different operations, and makes the event count an explicit ablation. We do not
borrow evidence that repeated growth compounds, because none exists.

---

## 3.6 Constants

All developmental constants live in the canonical table of §02.2.3. Derived quantities:
`phi_ann` (§3.3), `T_0` (1.17), `epsilon_bif` (§01.7.2), `tau_death` (§04.4),
`growth_schedule` (§3.5.1).

---

## 3.7 Falsification

- **A3** — disable births and deaths. If equivalent under the declared TOST margin (§08.5),
  development is decorative and C1 is unmet (K2).
- If `erank` shows a significant negative Mann–Kendall trend with turnover enabled, the
  plasticity argument for death does not operate here (K8).
- If lineages do **not** form spatially compact clades under `lambda_wire > 0`, the module
  formation mechanism of §3.2 does not work as claimed.
- If the bifurcation alarm fires on more than 1% of **selection birth cohorts**, births are
  not behaving as a controlled probe.

  **[NORMATIVE — v3.1, the denominator and the exclusion]** Two corrections, both found by
  running a lifetime.

  *The denominator is cohorts, not units.* (1.21) is evaluated **once per cohort**, so one
  alarm on a 176-unit cohort is one event, not `1/176` of one. Dividing by units born makes a
  large cohort's alarm arithmetically incapable of crossing a 1% bar — the same class of defect
  as A-22's unreachable `cap_off`. Measured on a healthy smoke run: `0.94%` per unit born
  against `45%` per cohort, i.e. the per-unit form reported a pass on a run where nearly half
  the cohorts tripped.

  *Growth cohorts are excluded.* A scheduled `GROW` adding `3|U_live|` units is **not a
  probe** — altering the function is its purpose — so its alarms say nothing about whether the
  *birth operator* is controlled. They are reported separately as `growth_alarm_rate`, and a
  growth alarm does **not** halve `rho`: letting it would make the growth schedule silence the
  selection machinery scheduled alongside it.
- **A8** — if one growth event is reliably better than many, we adopt one and report it.
- **A11** — if removing recombination is equivalent, the crossover operator is removed.
