# SOMA — Specification 06: The Open-Ended Stream

**Spec version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Normative for `soma/openended/`.
Serves constitutional requirement **C2**.

---

## 6.0 The honest starting position

**[RATIONALE]** A systematic review finds that **no member of the open-endedness /
quality-diversity family has ever improved a language-modelling objective** — not novelty
search, MAP-Elites, POET, regret-based environment design, minimal-criterion coevolution,
nor classical Hebbian plasticity. Their demonstrated wins are in RL generalisation, program
discovery, model merging, and reasoning post-training.

SOMA therefore places open-ended search only where selection-driven search has measured
gains on a language objective, and treats everything else as a hypothesis with a kill
criterion.

| Component | Evidence class | § |
|---|---|---|
| Token-level selection over experience | large measured LM gains | 6.2 |
| Coevolving verifier-backed task stream | strong RL evidence, **no LM precedent** | 6.3 |
| Open-endedness certification | no precedent; makes the claim falsifiable | 6.4 |

---

## 6.1 The two references

**[NORMATIVE — v1 correction]** Spec v1 used one object, an EMA "lagged self", for three
roles. Two of those roles need **opposite polarity**, so one object cannot serve both.

| Role | Needs a reference that knows... | Object |
|---|---|---|
| Token selection (§6.2) | **more** than the model | `theta_m0`, the frozen M0 baseline |
| Selection case weights (§04.3) | **less** than the model | `theta_lag`, the EMA lagged self |
| Open-endedness observer (§6.4) | a *fixed past* model | the frozen checkpoint ring |

An EMA lagged self gives `L_theta <= L_lag` almost everywhere, so Rho-1-style excess loss
`max(0, L_theta - L_lag)` is identically zero. §04.3 uses the reversed difference (an
improvement signal); §6.2 uses a genuinely better reference.

### 6.1.1 `theta_m0` — the frozen reference, and why it is free

**[NORMATIVE]** `theta_m0` is the M0 baseline checkpoint (§07.4), which the project trains
regardless as its gate.

**[NORMATIVE — precomputation]** Because `theta_m0` is frozen and the natural-text corpus is
fixed, `L_ref(x_i)` is **precomputed once over the entire tokenized corpus** and stored as
`float16` beside the token stream (2 bytes/token). Training-time cost is **zero**.

**[NORMATIVE — the producer, and the refusal; v3.1, audit A-72]** The sidecar is built by
`scripts/build_ref_cache.py`, which reproduces the training stream's own `targets`/`scored`
construction line for line — a cache aligned to different tokens than the lookup expects is
undetectable from either side — and reports coverage against the expected shortfall of exactly
one token per document (the last, which has no next token).

**A run that means to implement §06.2 must supply the cache.** `select_tokens` accepts a
missing reference and falls back to `Delta_i = L_theta`, raw loss. That fallback is correct
for M0's own run and for a smoke test, and it is a **different mechanism** — it selects
whatever the model currently finds hard, including tokens that are hard because they are
unpredictable, which is the failure Rho-1 exists to avoid. v3.0 implemented the cache, tested
it, wired it into `LIFETIME`, and built one **nowhere**; the driver never set the field, so
every run would have reported "token selection: on" while running the fallback, with no error
and no log line. `train_tinystories.py` therefore **refuses to start** without `--ref-cache`
unless `--allow-no-ref-cache` is passed, which records the fallback in the run report.

Measured on the M-nano corpus: 195k tok/s over 472M tokens; mean `L_ref = 2.11` nats against
M0's held-out `2.02` — the reference is the better model, which is what makes `Delta_i` a
learnability signal rather than a difficulty signal.

**[NORMATIVE — the cache is bound to its packing and refused on mismatch]** `L_ref(x_i)` is
**context-conditional**, so a cached value is valid only under the exact packing that produced
it. The sidecar records
`(tokenizer_hash, corpus_hash, m0_checkpoint_id, T_ctx, packing_scheme, doc_boundary_policy)`
and any mismatch is **refused at load**, exactly as §2.3 refuses a germline hash mismatch.

Two consequences v2 left implicit, both of which the implementation must honour: attention is
masked at **document boundaries**, so a token's conditioning depends only on its own document
and the cached value does not depend on what was packed beside it; and the cache is keyed by
`(doc_id, position_in_doc)`, not by position in a shuffled stream — otherwise §07.3's
data-order jitter would silently invalidate every cached value while every recorded hash still
matched.

**[NORMATIVE — v1 correction]** Spec v1 claimed the reference cost "under 3% of step time"
while §6.2 and §04.3 required a reference forward for **every token of every batch at every
step** — an extra full forward, ≈ **+25–33%** of step time, an order of magnitude above the
claim, propagating into every wall-clock number in §07–§09. Precomputation removes the cost
entirely rather than restating it.

**[NORMATIVE — synthetic tokens are ranked by raw loss, and that is a weaker rule]**
`L_ref` exists only over the natural-text corpus, so `Delta_i` is **undefined** on the
synthetic source. Within-source ranking does not repair this, as v2's cross-reference implied:
it fixes the *comparability* of the ranking, not the *existence* of the quantity being ranked.
Synthetic tokens are therefore ranked by **raw** `L_theta` within their own source, and each
source contributes its own `k_tok` fraction independently.

This is acceptable here, and only here, because the archive already supplies the learnability
signal that `Delta_i` supplies for natural text: a mastered generator is evicted by
`Learn = 4 s (1 - s)` (§6.3.3), so raw loss over the surviving generators is bounded away from
the already-learned regime by construction. The substitution is stated rather than glossed,
and §06.6's A6 is what tests whether it earns its place.

### 6.1.2 `theta_lag` — the lagged self

```
  theta_lag <- kappa_lag * theta_lag + (1 - kappa_lag) * theta      every T_ref steps (6.1)
```

**[NORMATIVE]** `kappa_lag` is stated **per update**, not per step, and derived from a target
half-life `h_lag` in steps:

```
  kappa_lag = 0.5 ^ ( T_ref / h_lag )                                            (6.2)
```

with `T_ref = 32` and `h_lag = 0.05 * N_step`. Spec v1 wrote a per-step EMA indexed by `n`
in its equation and "applied every 32 steps" in its prose; at `kappa = 0.999` applied once
per 32 steps the effective half-life is ~22k steps, i.e. `theta_lag` would have remained
approximately the initialisation for a whole M1 lifetime.

Storage: `bfloat16`, **GPU**-resident, 0.049 GB at M1 (§05.10).

**[NORMATIVE — `theta_lag` costs forward passes, and they are budgeted]** §04.3's `w_t`
requires `L_lag(t)`, i.e. a forward pass of `theta_lag`. v2 budgeted only "one parameter copy
every `T_ref` steps" (§07.8), reintroducing under a different name exactly the unbudgeted
reference forward that resolution A-5 removed for `theta_m0`. Unlike `L_ref`, `L_lag`
**cannot** be precomputed, because `theta_lag` changes during the run; and it cannot be
CPU-resident, because it is evaluated rather than merely stored.

The cost is bounded by restricting evaluation to a `d_rate` fraction of each batch — the rows
the reservoir is offered — rather than to every token. At M1 that is `d_rate * B_tok = 6554`
tokens against `B_tok = 65536`: a forward on **10%** of the batch, `2ND` against the step's
`6ND`, so **+3.3%** of step time. The figure is asserted by the profiling gate of §07.8, not
assumed.

**[NORMATIVE — v3.1]** `w_t` is evaluated **at push time**, on the rows being offered, and the
scalar is stored; the buffer does **not** retain context windows. §04.6 gives both reasons: the
context store is `L` times larger than v3.0 costed it (161 MB, not 13.4 MB, because `U` is
per-Site), and (4.2) multiplies `w_t` by `r_ut` and `||delta_ut||_2`, which are necessarily
measured when the token was routed — so evaluating `w_t` later would form a product of
quantities from different moments.

### 6.1.3 The certification checkpoint ring

**[NORMATIVE]** `J_ring = 8` frozen checkpoints, **CPU**-resident, `bfloat16`. Eviction
keeps the checkpoints nearest to `N_step * 2^-j` for `j = 0..J_ring-1`, so the retained set
spans the run logarithmically and does **not** require knowing `N_step` in advance beyond
the configured value.

**[NORMATIVE — v1 correction]** Spec v1 said "ring buffer ... spaced logarithmically",
which is self-contradictory (a ring evicts oldest; log spacing evicts the middle), gave no
residency, and omitted the memory from §05.10. Both are now specified and costed (0.39 GB
at M1, CPU).

---

## 6.2 Token-level selection

```
  Delta_i = L_theta(x_i) - L_ref(x_i)                                            (6.3)
  I_i     = 1 if rank(Delta_i) is in the top k_tok fraction *within its source*
            or with probability p_explore, else 0                                (6.4)
  L_task  = - ( 1 / max(1, sum_i I_i) ) * sum_i I_i * log P(x_i | x_{<i}; theta) (6.5)
```

**[NORMATIVE]** `L_ref` is `theta_m0`'s cached per-token loss for natural tokens. Both terms
of (6.3) are accumulated in **`float32`**; a `bfloat16` difference of two `O(1)` losses is
dominated by rounding late in training and would rank noise.

**[NORMATIVE — within-source ranking]** Ranking is performed **separately within the natural
and synthetic sources**, and each source contributes its own `k_tok` fraction. Spec v1
ranked jointly; synthetic tokens carry persistently high `Delta` and would have crowded out
natural tokens, silently re-weighting the objective and confounding ablation **A5** with
**A6**. The realised synthetic share of selected tokens is a required log field.

**[NORMATIVE]** The normaliser is `sum_i I_i`, the realised count, not `N_tok * k_tok`;
`p_explore` inclusions make the two differ.

**[NORMATIVE]** Ranking is over the **micro-batch**. Ranking over a full gradient-accumulated
batch would require all micro-batches' `Delta` before any backward pass, which is not
implementable in one pass; the change is stated rather than glossed.

**[EMPIRICAL]** Token-loss trajectories partition as: low→low **51%** (already learned),
high→low **26%** (the learnable band), high→high **11%** (unlearnable), low→high **12%**.
Reported gains at 7B: `GSM8K 42.9 -> 66.9`, `MATH 22.2 -> 31.0`, baseline reached ~`10x`
faster, `+6.8` points over 15 general benchmarks. The ratio curve is flat from `k_tok = 0.9`
to `0.6` and falls **below** the no-selection baseline at `0.5`.

**[NORMATIVE]** `k_tok = 0.65`; values below `0.55` are rejected by config validation.

**[NORMATIVE — the maturity gap, OURS]** Token selection has no analogue of `M_mat`: a token
that becomes temporarily hard can be deselected indefinitely. With probability
`p_explore = 0.05` a token is included regardless of rank. **Degeneracy rule.** Every `10 * T_log` steps, restricted to vocabulary items that
**occurred at least 32 times** in that window, let `f_starved` be the fraction whose realised
inclusion rate is below `0.5 * k_tok`. Then

```
  if f_starved > 0.10:   p_explore <- min( 1.5 * p_explore , 0.20 )
  else:                  p_explore <- max( 0.05 , p_explore / 1.5 )
```

Spec v1 left the rule unspecified inside a `[NORMATIVE]` block. v2 supplied one that was inert
in both directions: over `10 * T_log` steps the fraction of the *whole* vocabulary with zero
inclusions is dominated by items that never **occur** at all, so at `V = 8192` it sits near its
ceiling permanently instead of crossing `0.5` on a selection pathology; and the rule could only
ever raise `p_explore`, with no cadence and no decay, so once raised it never returned.

---

## 6.3 The coevolving task stream

### 6.3.1 Why verifier-backed

**[RATIONALE]** Two constraints. First, verifier-free self-improvement **sharpens** rather
than expands: RLVR models win at `pass@1` while the base model wins at large `k`, and the
reachable support demonstrably shrinks. Ground-truth generators are the channel through
which new support can enter. Second, bounded-depth transformers and diagonal linear
recurrences both lie in `TC^0` and provably cannot do arbitrary-length parity or permutation
composition; §05.2.2 gives the substrate the capacity, and the task stream supplies the
pressure to use it.

### 6.3.2 Task families and serialisation

**[NORMATIVE — reserved symbol block]** Token ids `0..63` are a **reserved block** that the
tokenizer never merges and never emits for natural text. Every synthetic task is serialised
over this block, guaranteeing **one symbol = one token**.

**[NORMATIVE — v1 correction]** Spec v1 specified no serialisation. With a BPE vocabulary
learned on story text, runs of bits and brackets would merge into multi-symbol tokens,
changing the structure of PARITY and DYCK and invalidating both the length-generalisation
claims of §08.2 and the expressivity argument of §05.2.2.

**[NORMATIVE]** The loss is masked to the **answer span** only.

| Family | `lambda_task` parameters | Verifier |
|---|---|---|
| `PARITY` | `n_len` | XOR reduction |
| `MODCOUNT` | `m_mod`, `n_len` | running count mod `m_mod` |
| `PERM` | `p_grp`, `tpt` (transitions per token), `n_len` | permutation composition |
| `DYCK` | `b_types`, `depth`, `n_len` | stack automaton |
| `COPY` | `alpha_sz`, `n_len`, `gap` | identity on the copied span |
| `MQAR` | `P_mqar`, `distract`, `n_len` | key→value lookup |
| `GRID` | `h`, `w`, `tf`, `n_ex` | apply `tf` |

**[NORMATIVE]** `GRID`'s transform set `tf` is enumerated: `identity`, `rot90`, `rot180`,
`rot270`, `flip_h`, `flip_v`, `translate(dx,dy)`, `recolour(perm)`, `tile(k)`. Spec v1 left
"transform class" undefined, making the family unimplementable.

**[NORMATIVE]** Every generator emits `(prompt, answer, lambda_task)` and every verifier is
a **total** function returning `{0,1}`. Generators run on CPU in dataloader workers. If
generator latency exceeds GPU step time the implementation **raises**; spec v1 said it
"asserts ... and logs a warning otherwise", which is neither, and contradicted §02.1.

### 6.3.3 The archive

`Lambda` holds at most `A_max = 512` parameterisations with statistics.

```
  Learn(lambda) = 4 * s_task(lambda) * ( 1 - s_task(lambda) )                    (6.6)
  Imp(lambda)   = max( 0 , s_task(lambda) - s_H(lambda) ) / H_arch               (6.7)
  Pri(lambda)   = Learn(lambda) * ( eta_imp + Imp(lambda) )                      (6.8)
```

`Learn` is `0` at `s = 0` (no signal available) and at `s = 1` (nothing left to learn),
maximal at `s = 1/2`, bounded in `[0,1]`, and needs only the verifier.

**[NORMATIVE — `s_task` protocol]** `s_task` is the fraction of `n_eval = 32` freshly
generated instances whose **greedy** continuation (temperature 0, `max_new` = the answer
length, stop at the answer-span terminator) verifies exactly. No sampling. Spec v1 gave no
decode protocol for the archive's central quantity.

**[NORMATIVE]** `H_arch = 5` **archive steps** (not optimizer steps; the collision with
§04's deleted `H` is why the symbol is renamed). Each member keeps a ring of its last `H_arch`
solve-rate estimates and `s_H` is the **oldest** entry in that ring. It is initialised to
`s_task` at insertion, so `Imp` is defined for a fresh member.

**[RATIONALE — `Imp` is usually zero, and that is the intended behaviour]** With
`A_target = 20` archive steps per lifetime and round-robin refresh of `R_ref = 16` members per
step, only a member that survives a large fraction of the lifetime accumulates `H_arch = 5`
distinct estimates. For every other member the ring is short, `s_H = s_task`, `Imp = 0`, and
`Pri` reduces to `eta_imp * Learn`. `Imp` is therefore a bonus for persistently improving
lineages rather than the primary driver, and the archive does not stall where it is
undefined.

**[NORMATIVE]** `eta_imp = 0.1`, so a stalled-but-learnable task retains priority.

```
Algorithm SEED_ARCHIVE() -> Lambda
  # Called once, at LIFETIME step 5. v2 never initialised Lambda, so the first firing of
  # ARCHIVE_STEP sampled a parent from an empty set and the whole task stream was undefined.
  Lambda <- {}
  for family in {PARITY, MODCOUNT, PERM, DYCK, COPY, MQAR, GRID}:        # §6.3.2
      lambda        <- family at the lower bound of every EDIT range (§6.3.3), clipped
                       by the serialised-length rule below
      s_task(lambda) <- estimate over n_eval instances with the current model
      ring(lambda)   <- [ s_task(lambda) ]        # so s_H = s_task and Imp = 0 initially
      insert lambda unconditionally     # the tau_admit gate of step 5 governs EDITs only
  postcondition: |Lambda| = 7 ; every member carries a solve-rate estimate and an s_H ring
```

**[RATIONALE]** The seeds are admitted unconditionally because `tau_admit` is a *learnability*
gate on a proposed variation, and at step 5 the model is at initialisation, where every
generator reads as `s_task ~ 0` and would be rejected — leaving `Lambda` empty forever.

```
Algorithm ARCHIVE_STEP  (fired by the counter of (1.15) with period T_arch (1.18))
  1. with probability eps_arch = 0.1 sample lambda uniformly from Lambda;
     otherwise sample proportional to Pri(lambda)
  2. lambda' <- EDIT(lambda)
  3. reject if lambda' already in Lambda, or if lambda'.n_len lies in the
     held-out evaluation band (§6.3.5)
  4. estimate s_task(lambda') over n_eval instances with the current model
  5. if Learn(lambda') > tau_admit: insert lambda'
  6. if |Lambda| > A_max: evict argmin Pri
  7. refresh s_task for the R_ref = 16 **oldest** members (round-robin)
  postcondition: |Lambda| <= A_max ; every member's estimate is at most
                 ceil(A_max / R_ref) archive steps old
```

**[NORMATIVE — v1 corrections]**
- *Freeze.* Spec v1 sampled purely proportional to `Pri`. As the model masters the archive
  `s -> 1`, `Learn -> 0`, `Pri -> 0` for every member, no parent can be sampled and the
  archive **freezes permanently**. The `eps_arch` mixture makes this state unreachable.
- *Staleness.* Spec v1 refreshed "a random 5%", under which `P(unrefreshed for 20 steps)
  = 0.95^20 = 0.36`, so its own postcondition would fail for a third of the archive at all
  times. Round-robin oldest-first makes the bound achievable.
- *Admission.* `tau_admit` is raised from `0.05` to **`0.30`**. At `n_eval = 32`,
  `Learn(1/32) = 0.121 > 0.05`, so the v1 gate admitted any generator with 1–31 successes
  out of 32 — near-vacuous. `0.30` excludes `s < 0.08` and `s > 0.92`.

**[NORMATIVE — `EDIT`]** Exactly one parameter is perturbed by one step on its own scale.

| Parameter | Type | Min | Max | Step | Weight |
|---|---|---|---|---|---|
| `n_len` | int | 4 | 96 | `x (1 +/- 0.25)`, rounded | 3 |
| `m_mod` | int | 2 | 9 | `+/- 1` | 1 |
| `p_grp` | int | 3 | 6 | `+/- 1` | 1 |
| `tpt` | int | 1 | 4 | `+/- 1` | 1 |
| `b_types` | int | 1 | 4 | `+/- 1` | 1 |
| `depth` | int | 2 | 12 | `+/- 1` | 1 |
| `alpha_sz` | int | 2 | 32 | `x2` / `/2` | 1 |
| `gap` | int | 0 | 32 | `+/- 4` | 1 |
| `P_mqar` | int | 2 | 32 | `+/- 2` | 2 |
| `distract` | int | 0 | 64 | `+/- 4` | 1 |
| `h`, `w` | int | 2 | 8 | `+/- 1` | 1 |
| `tf` | categorical | — | — | uniform over the enumerated set | 2 |
| `n_ex` | int | 1 | 4 | `+/- 1` | 1 |

Every value is clamped to its range; a no-op edit is resampled up to 8 times then abandoned.

### 6.3.4 Mixing

```
  P(sample is synthetic at step n) = f_oe                       n < n_dec
                                   = f_oe * (1 - (n-n_dec)/(N_step-n_dec))   n >= n_dec
```

**[NORMATIVE]** `f_oe` anneals to zero across the LR decay phase; placing distributionally
unusual data where the learning rate is smallest wastes it, and the final model must be
calibrated on natural text. **[NORMATIVE]** Once `f_oe` reaches 0, `ARCHIVE_STEP` and
`CERTIFY` are **suspended** (the batch ring would otherwise go stale against a stream that
no longer contains synthetic data).

### 6.3.5 Evaluation-band holdout

**[NORMATIVE — one rule, not three]** The archive's admissible length is
**`4 <= n_len <= 32`** at every configuration. The `EDIT` table's `n_len` row is clipped to
this range, which **supersedes its nominal upper bound of 96**, and step 3 of `ARCHIVE_STEP`
asserts it.

Spec v1 let `EDIT` mutate length by ±25% with no upper bound at all, so the model could be
**trained on length-256 parity by its own archive**, confounding the length-generalisation
claim, K7, and A7's inference. v2 then stated three mutually inconsistent rules within this one
subsection: the `EDIT` table permitted `4..96`, this paragraph said `<= 32`, and step 3
rejected only the `40..256` band — which admits `33..39`.

**[NORMATIVE — the holdout covers every parameter the suite varies]** The archive is
additionally capped at `P_mqar <= 16` and `depth <= 8`, against suite values of `32` and `8`.
v2 held out `n_len` alone, so the MQAR and DYCK bars of §08.2 — which generalise along
`P_mqar` and `depth`, not along length — were confounded by archive training at the very
settings they score.

**[NORMATIVE — serialised length is clipped against `T_ctx`]** Step 3 also rejects any
`lambda` whose serialised `(prompt, answer)` pair exceeds `T_ctx - 2`. v2 clipped no parameter
against the context budget: `GRID` at `h = w = 8, n_ex = 4` serialises
`(2 n_ex + 2) h w = 640` symbols against `T_ctx = 128` at M-nano, so a large majority of
nominally admissible archive members were untrainable and unscoreable at that configuration,
and `s_task` would have been measured on truncated prompts.

---

## 6.4 Certification

**[NORMATIVE]** SOMA does not assert that its stream is open-ended; it measures it and
reports the measurement either way.

```
  Novelty:      forall t, forall T>t, exists T'>T :  E[l(t,T')] > E[l(t,T)]      (6.9)
  Learnability: forall T, forall t<t'<T :           E[l(t',T)] < E[l(t,T)]       (6.10)
```

**[NORMATIVE — the claim is weaker than (6.9)–(6.10)]** These are quantified over unbounded
time and cannot be decided from a finite matrix; literal novelty would require unbounded loss
growth, which a bounded NLL cannot supply. What is implemented is a **finite-sample trend
test**, which is a strictly weaker claim, and it is labelled as such.

**Instantiation.** Observer = frozen checkpoint `theta^{(t_j)}` from the ring (§6.1.3);
artifact = the synthetic batch emitted at step `T_k`, stored in a ring of `K_ring = 32`;
`l(t,T)` = mean next-token NLL of `theta^{(t_j)}` on batch `X^{(T_k)}`.

**[NORMATIVE — capture cadence is decoupled from certification cadence]** Batches are
captured every `ceil(N_step / K_ring)` steps, so the ring **spans the whole run**;
certification *fires* every `T_cert` steps (1.19). The frozen checkpoints of §6.1.3 follow the
same rule at `ceil(N_step / J_ring)`.

Spec v1 left the cadence unspecified, so a ring of the last 32 emitted batches would have
spanned ~32 steps and the "novelty" test would have measured adjacent-step noise. v2's repair
tied capture to `T_cert / K_ring`, which is self-contradictory: "the ring spans the run"
requires `T_cert ~ N_step`, at which point certification fires at most once — and with the
counter of (1.15) starting at `T_cert`, zero times. Decoupling the two removes the conflict.
Realised counts at the §01.9 values are **4 certifications at M-nano and 3 at M1**, each over
a ring spanning the entire run.

```
  Lhat[j,k] = NLL( theta^{(t_j)} ; X^{(T_k)} )                                   (6.11)
```

`J_ring * K_ring = 256` forward passes on stored batches, no backward. At M1
(`32 x 512` tokens/batch) this is `4.19e6` tokens of forward compute per certification,
every `T_cert` steps.

**[NORMATIVE — the test]** Mann–Kendall, fully specified:
- **original** MK statistic `S = sum_{i<j} sign(x_j - x_i)`;
- **tie-corrected** variance;
- **exact** null distribution for `n <= 10`, normal approximation with continuity correction
  otherwise (with `J_ring = 8` the normal approximation is invalid, so the exact null is what
  actually runs);
- **one-sided**;
- rows/columns with `n < 6` usable points are excluded from **both** numerator and
  denominator of the reported fraction;
- **Benjamini–Hochberg** across rows at `alpha_sig = 0.05`.

Novelty: for each row `j`, test `Lhat[j,k]` for a positive trend in `k` over `T_k > t_j`.
Learnability: for each column `k`, test for a negative trend in `j` over `t_j < T_k`.

**[NORMATIVE — the reported statistic and its granularity]** The **novelty fraction** is the
number of rows whose one-sided test survives Benjamini–Hochberg, divided by the number of rows
with `n >= 6` usable points.

The qualifying row count is **computed, not asserted**. Row `j` qualifies when at least
`MIN_USABLE = 6` of the `K_ring = 32` captured batches satisfy `T_k > t_j`. Capture is on the
`ceil(N_step / K_ring)` cadence, so the `T_k` are evenly spaced across the run and row `j`
qualifies iff `t_j < N_step * (1 - 6/32)`. With checkpoints at `t_j = N_step * 2^-j` for
`j = 0..J_ring-1`, every `j >= 1` satisfies it: **7 rows qualify**, and the per-firing novelty
fraction is quantised to multiples of `1/7 = 0.1429`.

**[NORMATIVE — v3.1 correction]** v3.0 stated "at most 3 rows qualify ... quantised to
`{0, 1/3, 2/3, 1}`". That is wrong: it is 7 rows and `1/7`. The error was **conservative** —
the K5 margin derived from it is larger than the granularity requires, not smaller — but a
stated count the code contradicts is a defect whichever way it errs, and this one sat inside
the justification for a kill criterion's threshold. `CertificationVerdict.granularity` now
reports the realised value, so the margin is checked against a measurement rather than against
a remembered figure.

**A TOST margin below the granularity is not decidable**: the equivalence region would lie
entirely inside the gap between two attainable values, so neither one-sided test could reject
and the criterion would be undecidable in **both** directions. v2 declared `delta = 0.10` for
K5, below `1/7`, so K5 could never fire. §08.5 sets `delta = 0.34`, which exceeds the
granularity by 2.4x — deliberately, and on substantive grounds as well as arithmetic ones: a
`live - ramp` difference smaller than a third of the tested rows is not a difference worth
calling open-endedness.

**[NORMATIVE]** The difference `live - ramp` is averaged over the `C_target` firings and over
`n_seed` seeds before the TOST is applied, which is what makes a margin finer than the
per-firing granularity meaningful at the level of the *mean*; the per-firing statistic remains
quantised and is reported as such.

**[NORMATIVE — three arms, because a frozen control is not enough]** Certification runs
**three** times per firing:

| Arm | Archive | Purpose |
|---|---|---|
| **live** | the evolving `Lambda` | the claim |
| **frozen** | `Lambda` fixed after `SEED_ARCHIVE` | sanity floor: is the machinery inert? |
| **ramp** | `Lambda` replaced by a scripted monotone ratchet on `n_len` alone, matched to the live arm's mean solve-rate trajectory | the actual confound |

**[NORMATIVE — the matching is measured, not scripted; v3.1, audit A-79]** The ramp arm is
`soma.openended.certify.RampController`. It measures both arms' solve rates on the **same
model at the same step** and ratchets `n_len` up by one whenever the live arm is easier by
more than a tolerance. Matching is one-sided by construction — the ratchet is monotone, which
is what keeps the arm a difficulty *tracker* rather than a second adaptive archive — so the
residual is **reported** as `ramp_matching_error` with every certification rather than
assumed. With no solve-rate function available the arm **holds** its current `n_len` rather
than ramping on a schedule.

v3.0 implemented an unmatched linear interpolation of `n_len` from 4 to 32 across the run,
under a docstring asserting that a linear ramp was "the simplest schedule with that property"
while nothing measured the property and nothing anywhere read the live arm's solve rate. An
unmatched ramp makes `live - ramp` a comparison of two task **distributions** rather than of
two archive **dynamics**, which is the attribution the arm was introduced to obtain.

The reported criterion is **live minus ramp**; the live arm must additionally exceed the
frozen arm, or the machinery is doing nothing at all.

The frozen arm separates "the archive changes" from "the archive is static", which is **not**
the confound this section names. A pure difficulty-tracker — one scalar ratcheted against the
current model — also beats a frozen archive on every novelty statistic, because it too keeps
producing instances older checkpoints have not seen. A frozen-only control therefore fires
identically for genuine open-endedness and for monotone difficulty ramping, which is precisely
the distinction K5 exists to make. The ramp arm makes it.

**[NORMATIVE — the control arm, and why certification alone proves nothing]** Certification's
live arm is compared against controls; the raw live-arm statistic is never reported alone.

Rationale: learnability is satisfied by ordinary training progress on *any* data, and
novelty is produced *by construction*, because `Learn = 4s(1-s)` pins the archive near a 50%
solve rate **against the current model**, which necessarily makes later batches harder for
older checkpoints. Without a control the test cannot distinguish an open-ended archive from a
static difficulty-tracker, and spec v1's claim that certification is "falsifiable by
construction" was not established.

**[NORMATIVE — verdict]** If the live arm does not exceed the **ramp** arm on the novelty
fraction under the declared TOST margin (§08.5), the stream is **declared not open-ended**
and the finding is reported, not suppressed; `f_oe` is then reduced. This is what prevents
constitutional requirement C2 from being satisfied decoratively.

---

## 6.5 What is not claimed

- Not that open-ended search improves perplexity. **A6** may show it does not.
- Not novelty for token-level selection, which is prior work. The novelty claimed is the
  **two-reference construction** (§6.1) and the **controlled certification harness** (§6.4).
- Not that verifier-backed synthetic tasks transfer to natural language; §08's held-out
  natural-text evaluation measures exactly that.

---

## 6.6 Falsification

- **A6** — `f_oe = 0`. If the state-tracking and length-generalisation suites are equivalent
  under the declared margin, the stream is decorative.
- **A5** — `k_tok = 1.0`. If equivalent, the best-evidenced C2 component does not transfer
  to this architecture, which is itself an important negative result.
- If certification's **live-minus-ramp** difference is null while an ablation shows a gain,
  the gain is real but **not** attributable to open-endedness, and the mechanism is
  re-described as curriculum shaping. If live-minus-**frozen** is also null, the archive is
  inert and the finding is a bug report, not a result.
