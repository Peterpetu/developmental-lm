# 10 — The mathematics of change is itself under selection

This section replaces two hand-picked mechanisms with decoded ones. It exists
because of a question that exposed the same defect §00 records: the germline was
evolutionary and the lifetime was stock deep learning with an ornament.

## 10.1 The defect

**[NORMATIVE — stated as a defect, because that is what it was.]** Up to v0.9,
GNOMON's two timescales were not symmetric:

| | mechanism | who chose it |
|---|---|---|
| germline | MAP-Elites; Gaussian mutation at `sigma_rel = 0.15` | **the author** |
| lifetime | AdamW; a decoded `eta` multiplying its step | **the author**, except `eta` |

`eta` multiplies a displacement AdamW has already computed. If AdamW supplies
most of the update and `eta` modulates a few percent of it, that is structurally
identical to SOMA's `W_shared`: a conventional mechanism doing the work with an
evolutionary decoration riding along, and §00's RC1 says that is exactly what
gradient descent will settle into. The ratio was never measured.

**[NORMATIVE — what is *not* the defect.]** The gradient is not the problem, and
removing it would be asceticism rather than principle. `dL/dz` is information
about the environment, and biological learning has error signals of its own —
climbing-fibre input, dopaminergic prediction error. The problem is narrower and
sharper: **the map from error to change was fixed by the author.** AdamW's
`m / (sqrt(v) + eps)` is a design decision imported from outside the
constitution. Every weight and the whole architecture are decoded from `Gamma`,
and then a hand-picked rule decides how the one thing a lifetime owns actually
moves.

## 10.2 The decoded plasticity rule

**[NORMATIVE]** The lifetime update for an acquired identity `z_i` is

```
    g_i      = dL/dz_i                                   the error signal
    m_i      <- beta_1 m_i + (1 - beta_1) g_i            an eligibility trace
    v_i      <- beta_2 v_i + (1 - beta_2) g_i^2          a second-moment trace
    ghat_i   = g_i / (sqrt(v_i) + eps)                   scale-free error
    mhat_i   = m_i / (sqrt(v_i) + eps)                   scale-free trace
    nu       = tanh( (Lbar - L) / (sigma_L + eps) )      the neuromodulator

    dz_i     = - lr(n) * [ A_i ghat_i + B_i mhat_i + C_i z_i + D_i nu mhat_i ]
```

with `(A_i, B_i, C_i, D_i)` **decoded from identity** (§10.3) and `lr(n)` the
warmup-stable-decay schedule of §05.4.

**[NORMATIVE — the four terms, and why exactly these.]** The basis is not
arbitrary; it is what is locally available to a channel, one term per source of
information:

| term | what it is | what it buys |
|---|---|---|
| `A_i ghat_i` | the instantaneous error, normalised | plain sign-descent; no memory |
| `B_i mhat_i` | the accumulated trace | momentum; **this is the Adam term** |
| `C_i z_i` | the identity's own value | homeostasis, decay, weight-decay, and the only term that can act with no error at all |
| `D_i nu mhat_i` | the trace **gated** by a global signal | three-factor plasticity: change is licensed by whether the system is beating its own recent baseline |

**[NORMATIVE — AdamW is inside the search space, and that is the point.]**
Setting `A = 0`, `B = 1`, `C = 0`, `D = 0` makes the rule **exactly** AdamW with
decoupled weight decay `0`. The rule the author would otherwise have imposed is
therefore a single point in the space the genome searches, and §10.6's P1 proves
the identity numerically rather than asserting it. A search whose baseline lies
*outside* its space cannot be said to have chosen anything; one whose baseline
lies inside it can rediscover momentum, and if it does, that is a finding.

**[NORMATIVE — the neuromodulator.]** `nu in (-1, 1)` compares the current batch
loss against an exponential baseline `Lbar` of the recent past, scaled by the
running standard deviation `sigma_L`:

```
    Lbar    <- rho Lbar + (1 - rho) L,     rho = 0.99
    sigma_L  = sqrt( rho s2 + (1 - rho) (L - Lbar)^2 )
```

It is positive when the model is doing better than its recent self. This is a
reward-prediction error, and gating plasticity on one is the defining feature of
three-factor learning rules (Frémaux & Gerstner, *Neuromodulated
spike-timing-dependent plasticity, and theory of three-factor learning rules*,
Front. Neural Circuits 9:85, 2016). It costs one scalar per step, is **global**
— every channel sees the same `nu` — and is **detached**.

**[NORMATIVE]** `A, B, C, D` and `nu` are all detached. Differentiating through
the update would make it a second-order object and would let a lifetime minimise
its loss by switching its own plasticity off, which is a degenerate optimum and
not a learning mechanism. This is §05.4's argument, unchanged.

## 10.3 Where the coefficients come from

**[NORMATIVE]** They are decoded by the same trunk, under the roles that already
exist. `ROLE_PLAST_CHAN` and `ROLE_PLAST_TOK` change **shape** from a vector to a
matrix of width `N_RULE = 4`:

```
    PLAST_CHAN : (d_model, 4)     rows = channels, columns = (A, B, C, D)
    PLAST_TOK  : (V, 4)           rows = tokens
```

The column axis is `inner` (§03's `AXIS_KIND`), so the four coefficients get four
distinct innate identities from the shared code table, and the decoder needs no
new tensor, no new role and no new code path. `|Gamma|` is **unchanged**.

**[NORMATIVE — this is a versioned change, made deliberately.]** §04.2 forbids
*adding* a role after genomes exist, because that would silently change what
every existing genome decodes. Changing the *shape* of two existing roles has
the same effect and is therefore made explicitly, recorded here, and genomes
from before this change are marked incomparable rather than silently reused.

**[NORMATIVE — the coefficients' ranges, and their initialisation.]**

```
    A_i = a_scale * tanh(raw_A)          a_scale = 1.0
    B_i = 1 + b_scale * tanh(raw_B)      b_scale = 1.0     so B in (0, 2), centred on Adam
    C_i = c_scale * tanh(raw_C)          c_scale = 0.1     decay is a small effect or none
    D_i = d_scale * tanh(raw_D)          d_scale = 1.0
```

`B` is centred on **1**, not on 0: an uninformative genome then performs Adam
exactly, so generation 0 starts at the strong baseline rather than at a random
rule, and evolution has to *earn* any departure from it. `C` is scaled down
because a decay term acts every step whether or not there is an error, so the
range that is useful is small and the range that is destructive is not.

## 10.4 Self-adaptive variation

**[NORMATIVE]** The germline's mutation scale is also decoded — from the genome
itself, by self-adaptation (Rechenberg 1973; Schwefel 1977; the diagonal case of
Wierstra et al., *Natural Evolution Strategies*, JMLR 15:949–980, 2014):

```
    for each genome tensor t:
        log_sigma_t <- log_sigma_t + tau * N(0, 1),     tau = 1 / sqrt(2 sqrt(n_t))
        sigma_t      = sigma_rel * exp(log_sigma_t) * max(rms(t), SIGMA_FLOOR)
        t            <- t + sigma_t * N(0, I) * Bernoulli(p_mut)
```

**[NORMATIVE — the mutation strengths are mutated *first*, then used.]** That
ordering is what makes self-adaptation work at all: a genome is evaluated with
the strengths it will pass on, so a strength that produces good offspring is
selected together with the offspring it produced. Applying the old strength and
storing a new one severs that link and the strengths drift at random.

**[NORMATIVE — per tensor, not per coordinate.]** A per-coordinate `log_sigma`
would add `|Gamma|` reals and double the genome, destroying the compression
claim for a second-order effect. Per tensor it is **27 reals** — twelve in
`Gamma_w`, fifteen in `Gamma_a` — against 5,279, and it captures the effect
§07.5 actually measured: tensors drift apart in scale across generations, and a
single global `sigma_rel` is then wrong for all of them.

**[NORMATIVE]** `log_sigma` is initialised at `0`, so generation 0 mutates
exactly as §07.5 specifies and self-adaptation is a departure evolution has to
earn — the same discipline as `B = 1` above.

## 10.5 What this is not claimed to do

**[NORMATIVE — recorded before the arms are run.]** Evolved local learning rules
are not known to beat gradient descent on language modelling, and the literature
that reports them working does so on small policies and small networks. The
prediction registered here is that the decoded rule will be **at best comparable**
to AdamW at equal budget, because AdamW is inside its space and evolution has
few generations to improve on it.

What the change buys is not expected to be loss. It is that **the learning rule
is under selection**, so `A-RULE` below can answer a question the previous design
could not even pose: how much of the lifetime's progress is attributable to a
mechanism the genome controls?

## 10.5b Measured: the mechanism does not pay, and why

**[NORMATIVE — the result, recorded in full because it is negative.]** §10.5
predicted the decoded rule would be *at best comparable* to AdamW. It is
slightly worse than that, and the reason is worth more than the prediction was.

**Generation 0, four arms, 4 genomes x 2 seeds, M-pico, 600 steps**
(`runs/gnomon/rules.json`):

```
    R-ADAM         5.4123 +/- 0.173
    R-ETA          5.4185 +/- 0.207     +0.006 vs Adam
    R-DPR          5.4427 +/- 0.185     +0.030
    R-DPR-FROZEN   5.4457 +/- 0.204     +0.033
```

All six pairwise comparisons are **equivalent** by TOST at a 0.10-nat margin;
the standard error of each mean is 0.061. Two things follow. `R-ETA` ties bare
AdamW, which **confirms §10.1's diagnosis by measurement**: v0.9's `eta` was an
ornament. And `R-DPR` sits on Adam at generation 0, which is exactly what
§10.3's centring of `B` on 1 requires — so this comparison cannot say whether
the mechanism is *beneficial*, only that it is not broken.

**The rule space, probed directly** (`runs/gnomon/rulespace.json`; 16 points with
`(A,B,C,D)` pinned globally, 300 steps): the space is **not** flat — it spans a
full nat, 5.545 to 6.551 — and six points beat Adam, the best by 0.171 nats.

**[NORMATIVE — and then the confound, which takes all of it away.]** `B`
multiplies the entire displacement, so `B = 2` at `lr` is *arithmetically*
`B = 1` at `2 lr`; `D` adds `E[nu] D ~ 0.33 D` to the same scale. The control
(`runs/gnomon/rulecontrol.json`) swept Adam's own `lr` across every effective
scale the rule points imply, and took the residual:

| rule point | effective `lr` | rule | Adam at that `lr` | residual |
|---|---|---|---|---|
| `B = 2.0` | 0.060 | 5.545 | 5.544 | **+0.002** |
| `D = +1.0` | 0.040 | 5.607 | 5.627 | −0.020 |
| `A=.6 B=.6 D=.5` | 0.041 | 5.577 | 5.619 | −0.042 |
| `B = 1.5` | 0.045 | 5.652 | 5.589 | +0.062 |
| `D = −0.5` | 0.025 | 5.832 | 5.769 | +0.063 |
| **`D = +1.0, B = 0.67`** | **0.030** | **5.922** | **5.716** | **+0.206** |
| **`D = −1.0, B = 1.33`** | **0.030** | **5.883** | **5.716** | **+0.167** |

The apparent structure was a learning rate. `B = 2` reproduces Adam at twice the
`lr` to **0.002 nats**; Adam's own curve runs 5.919 at `lr = 0.015` down to
5.468 at 0.12, and every rule point tested at `lr = 0.03` is beaten by raising
`lr` alone.

**[NORMATIVE — the sharpest result, and the one that decides.]** The last two
rows hold the *effective scale fixed* at Adam's and move the gating into `D`.
Both are **worse than plain Adam by 0.17 to 0.21 nats**, and worse in the same
direction for **both signs of `D`**. A term that hurts whichever way it points is
not carrying signal; it is injecting a time-varying multiplier correlated with
recent loss, which from the optimiser's point of view is noise.

**[NORMATIVE — the conclusion, stated plainly.]**

1. The **constitutional argument of §10.1 stands**: a fixed exogenous optimiser
   is an unprincipled component in a model whose every other part is decoded,
   and that reasoning was not wrong.
2. The **empirical payoff is zero and slightly negative.** The only exploitable
   dimension of §10.2's rule space is a global scale that `lr` already exposes,
   and the neuromodulator costs 0.17–0.21 nats at matched scale.
3. The decoded rule is therefore **retained as an arm, not promoted to the
   default**. `R-ADAM` is the default until something is measured to beat it.
   Building the instrument that could say so is the value delivered here; a
   mechanism kept because it is principled and never measured is precisely
   §00.3's defect.

**[NORMATIVE — a defect in every comparison reported before this section.]**
`lr` was `0.03` throughout, chosen from a sweep at 2,000 steps and then used at
300 and 600. Adam's curve above shows `0.03` is far from optimal at those
horizons. Every GNOMON-versus-baseline number recorded earlier is therefore at
an untuned `lr` for at least one side, and §09.1's `BASE-FREE` comparison in
particular tuned GNOMON's `lr` to `0.1` while leaving the baseline at `0.003` —
which understates the baseline, so the measured gap is a **lower bound** on the
compression cost rather than an estimate of it. Those runs are re-baselined
before any of them is reported as a result.

## 10.7 The one hypothesis left standing

**[NORMATIVE]** Every point in §10.5b pins `(A,B,C,D)` **globally**. A
*per-identity* scale is the one part of §10.2's rule the control does not touch:
rare tokens plausibly want a larger step than frequent ones, and `B` decoded per
token is exactly that field. `R-ETA` ties Adam only because a random genome's
field is near-uniform.

Its **ceiling** is measurable with no evolution at all: give each token a
hand-designed rate and see whether any shape beats uniform.

**[NORMATIVE — the normalisation is the whole design of that test, and the first
attempt got it wrong.]** Fields were first scaled to *arithmetic* mean 1, and
uniform won by 0.03 to 0.46 nats with the losses ordering almost perfectly by
the field's max/min spread — the signature of an artefact, not a finding. A
heavy-tailed field with arithmetic mean 1 puts most of its mass far below 1, so
the common tokens that dominate the loss all take smaller steps, and the field
loses for a reason unrelated to its shape. Fields are therefore normalised so
the **occurrence-weighted** mean is 1, `sum_v freq_v * field_v = 1`, which
matches the expected step per token *seen*.

**[NORMATIVE — measured, and it is not null.]** M-pico, 300 steps, `lr = 0.09`,
`B` per token and `A = C = D = 0`:

| field | max/min | held-out CE | vs uniform |
|---|---|---|---|
| **`freq^-0.25`** | 34.7 | **5.4340** | **−0.067** |
| `freq^-0.10` | 4.1 | 5.4609 | −0.040 |
| uniform | 1.0 | 5.5012 | — |
| `freq^+0.10` | 4.1 | 5.5114 | +0.010 |
| `freq^+0.25` | 34.7 | 5.5138 | +0.013 |
| random log-normal | 149.8 | 5.5435 | +0.042 |
| `freq^-0.50` | 1206 | 5.6079 | +0.107 |

A **dose-response** monotone in the exponent with a turning point — rare tokens
want larger steps, but only moderately; at `-0.50` it overshoots — and a random
field of comparable spread that *loses*, so it is the shape and not the spread.
Confirmed seed-paired at n = 12: `freq^-0.25` beats uniform by **+0.0619 nats,
95% BCa [+0.0165, +0.2103]**, clearing zero. `freq^-0.10` does not
(+0.036, [−0.039, +0.136]).

**[NORMATIVE — why the effect is small, and where the rest of it went.]** AdamW's
second moment already implements a per-token adaptive step: `v` decays for a
token absent from the batch, so a rare token arrives with small `v` and takes a
large step automatically. That the `-0.50` field *overshoots* is direct evidence
of it. The residual 0.062 nats is what the denominator leaves on the table.

**[NORMATIVE — and this is what the finding actually led to.]** A genome cannot
decode a frequency-shaped field from a code carrying no frequency information,
and §2.4's token code carried none (`R^2 = 0.004`, chance). Fixing that —
§02.9's Gray code of relative frequency rank, `R^2 = 0.488` — is worth
**+0.0999 nats, 95% BCa [+0.0288, +0.2647]** on its own, with **plain AdamW and
no decoded plasticity at all**. The larger of the two effects was therefore not
in the learning rule but in the innate code, and it was found only because the
plasticity measurement asked what signal the rule would need.

## 10.6 Arms and acceptance tests

**[NORMATIVE]** Four arms at matched budget, each removing exactly one thing:

| arm | lifetime rule | isolates |
|---|---|---|
| `R-ADAM` | AdamW, no `eta` | the bare baseline |
| `R-ETA` | AdamW x decoded `eta` | v0.9's design; the ornament |
| `R-DPR` | the decoded rule of §10.2 | the mechanism |
| `R-DPR-FROZEN` | the decoded rule, coefficients fixed at their `A=0,B=1,C=0,D=0` values | whether the *decoding* matters or only the form |

**[NORMATIVE]** `R-DPR-FROZEN` is the control that makes `R-DPR` interpretable.
If `R-DPR` beats `R-ADAM` but ties `R-DPR-FROZEN`, the gain came from the rule's
*form* and not from the genome deciding it — which would be the same decorative
outcome in new clothes, and would be reported as such.

| # | property | test |
|---|---|---|
| P1 | **Adam is inside the space** | with `A=0,B=1,C=0,D=0` the rule's displacement equals `torch.optim.AdamW`'s to `1e-6` over 50 steps |
| P2 | coefficients are decoded | perturbing `Gamma` changes `A,B,C,D`; perturbing nothing else does |
| P3 | detached | no coefficient and no `nu` carries `grad_fn`; a mutant that attaches one fails |
| P4 | the neuromodulator is a prediction error | `nu > 0` iff the batch loss is below its running baseline; `nu` is bounded in `(-1,1)` |
| P5 | `\|Gamma\|` is unchanged | the shape change costs zero genome reals |
| P6 | self-adaptation mutates strengths first | a mutant that applies the old strength and then updates it fails |
| P7 | self-adaptation is heritable | over 200 generations of pure drift, `log_sigma` under selection separates from `log_sigma` under none |
| P8 | generation 0 is the baseline | with an uninitialised genome, `R-DPR` and `R-ADAM` produce identical trajectories |
