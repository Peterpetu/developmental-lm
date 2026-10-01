# 05 — The lifetime

One lifetime is: sample a genome, decode a phenotype, initialise the soma, train the soma
for `N_life` steps, measure. It is the inner loop of §08's evolution and it is also the
only thing that happens in the single-genome runs of §09.

## 5.1 The loop

**[NORMATIVE]**

```
LIFETIME(Gamma, seed, cfg) -> (L_final, curve, Z)
  1  arch   <- DECODE_ARCH(Gamma_a, cfg)                      # §07; fixed for the lifetime
  2  Z      <- INIT_SOMA(seed, cfg)                           # §04.5
  3  W_rand <- random init of every role at s_target           # §05.5, the eps(n) anchor
  4  for n = 1 .. N_life:
  5      W    <- DECODE(Gamma, Z, arch)                        # §03, one decode per step
  6      W_n  <- eps(n) * W_rand + (1 - eps(n)) * W            # §05.5
  7      batch<- next(stream)
  8      L    <- cross_entropy( FORWARD(W_n, batch) )
  9      grad <- dL/dZ                                         # dL/dGamma only in the Lamarckian arm
 10      Z    <- STEP(Z, grad, eta, n)                         # §05.4
 11      if n mod N_eval == 0: record held-out CE
 12  return held-out CE at n = N_life, the curve, and Z
```

**[NORMATIVE]** Line 5 is inside the loop. `W` is a function of `Z`, and `Z` changes every
step, so a decode cached across steps would be stale and would silently sever the gradient
path from the loss to the identities — which is exactly the class of defect that made
SOMA's routing gradient severable with 452 tests passing. `G-MUTATE` plants that defect
and requires the suite to fail on it.

**[NORMATIVE — there is exactly one implementation of this loop.]** Until v1.1
there were two: `lifetime()` and `LifetimeRunner.run`. Every acceptance test
L1..L9 exercised the first and every run that produced a number used the second,
and they had drifted apart: the first ran plain `dpr` when asked for
`dpr-frozen`, could not express §10.6's pinned-coefficient probe at all, and
after §02.9 became the default would have gone on decoding §2.4's token code.
A suite that certifies a path the science does not take is §00.3's defect one
level down. `lifetime()` is now a wrapper, and L1..L9 bind the runner.

The correction exposed a second defect it had been hiding: L3 asserted that
`lifetime()` cleared `requires_grad` on the `Decoder` handed to it, which is a
side effect of that implementation rather than this section's invariant. Under
one implementation the assertion became vacuously true. It is restated over the
tensors that actually take part in a lifetime, and `G-MUTATE` M22 plants a
germline write to show the restated form can still fail.

**[NORMATIVE]** Line 9: in the Darwinian arms `Gamma` carries `requires_grad=False` and no
gradient reaches it. In the Lamarckian arm (`A-LAMARCK`, §08.5) it does, and every
artifact of that arm is labelled.

## 5.2 The objective

**[NORMATIVE]** Plain next-token cross-entropy over all tokens, in nats. No token
selection, no auxiliary loss, no metabolic term.

This is a deliberate reversal of SOMA, whose training objective summed a normalised
task loss with a metabolic price and a token-selection mask, with the consequence that the
quantity being optimised was not the quantity being reported and no curve could be
compared to a baseline's. Here the training objective, the fitness and the reported metric
are **the same quantity** measured on three different splits.

**[NORMATIVE]** Every reported curve is held-out cross-entropy on natural text, in nats
per token, at the configuration's `T`. Bits-per-byte is reported alongside for
comparability with the literature, computed as in SOMA `eval/metrics.bits_per_byte`.

## 5.3 Fitness

**[NORMATIVE]** The fitness of a genome is

```
    F(Gamma) = - E_{seed}[ held-out CE after N_life steps ]
```

estimated over `n_seed` lifetimes (default 2 at M-pico, paired by seed across genomes so
that comparisons between genomes are seed-matched).

**[NORMATIVE — what fitness is *not*.]** SOMA's fitness mixed a birth score, a slope
score and a criterion score with tuned weights, and its birth term was ~0 by construction
(§00 RC2). GNOMON's fitness is one number with one meaning. The *decomposition* — at-birth
loss, early slope, time to criterion — is recorded for every lifetime and reported, but
it does not enter selection, so no weighting decision can be made after seeing results.

## 5.4 The optimizer

**[NORMATIVE]** `Z` is optimised by AdamW with decoupled weight decay `0` on `z`, `zeta`,
`y`, `beta = (0.9, 0.95)`, `eps = 1e-8`, gradient clipping at global norm `1.0`, and a
warmup–stable–decay schedule (SOMA `optim.wsd_lr`, vendored) with warmup `0.05 N_life` and
decay over the final `0.2 N_life`.

**[NORMATIVE — decoded plasticity]** The per-parameter step is scaled by the decoded
`eta` of §03.4:

```
    z[i]    <- z[i]    - lr(n) * eta_chan[i] * adam_dir(z[i])
    zeta[v] <- zeta[v] - lr(n) * eta_tok[v]  * adam_dir(zeta[v])
```

`eta` is recomputed from the current `Gamma` and `Z` at every step and is **detached**: it
scales the step, it is not differentiated through. Differentiating through it would make
the update a second-order object and would let the lifetime reduce its own plasticity to
minimise loss, which is a degenerate optimum rather than a learning mechanism.

**[NORMATIVE]** `c_eta = 1.5`, so `eta in [0.223, 4.48]`: a factor of 20 between the least
and most plastic channel. The bound exists because AdamW's step is already normalised, so
an unbounded multiplier is an unbounded learning rate; 20x is wide enough for the genome
to silence a channel or to drive one hard, and narrow enough that no setting diverges at
the schedule's peak `lr`. `A-ETA` (§09) ablates `eta = 1` to measure what it is worth.

**[NORMATIVE]** Muon is **not** used. Muon orthogonalises gradients of 2-D weight
matrices; `Z` is a set of identity tables, not weight matrices, and the orthogonality
prior does not apply to them. It remains vendored for the free-weight baselines of §09,
where it does apply and where using anything weaker would understate the baseline.

## 5.5 Developmental annealing

**[NORMATIVE]**

```
    W_n = eps(n) * W_rand + (1 - eps(n)) * DECODE(Gamma, Z_n)
    eps(n) = exp(-n / lambda),      lambda = 0.04 * N_life
```

The paper reports this as a necessity, not an optimisation:

> if g-networks are initialized randomly, they produce p-networks that are far from the
> optimal fixed point. We found that such p-networks are **impossible to train**. This
> problem is exacerbated in moderately deep p-networks due to the exponential divergence
> of initialization errors from layer to layer. In practice, such p-networks return zero
> activations, which yield no gradients of weights.

Generation 0 of §08 decodes from random genomes and is exactly that regime. The ratio
`lambda / n_iterations = 0.04` is the paper's (500 of 12,500); GNOMON preserves the ratio
rather than the absolute value, since `N_life` differs by two orders of magnitude between
configurations.

**[NORMATIVE — the lifetime is checkpointable, and until v1.1 it was not.]**
§09.1 budgets an 8-hour `M1` run. Only the token *stream* was resumable, and
the test named L8 tested only the stream, so a crash at hour seven lost
everything and the acceptance test said nothing about it — an acceptance test
passing while the requirement it names is unimplemented. `LifetimeRunner.run`
takes `checkpoint` and `checkpoint_every`, writes atomically through a
`.partial` file so a crash mid-write cannot leave a checkpoint that loads and is
wrong, and `resume=True` continues from one.

**[NORMATIVE]** `W_rand` is drawn once per lifetime at `s_target` scale from the lifetime
seed and held fixed. By `n = 3 lambda` (12% of the lifetime) `eps < 0.05`; by `n = 7
lambda`, `eps < 1e-3`. The reported metric is always taken at `n = N_life`, where
`eps < 1e-10` at every configuration, so no reported number is contaminated by `W_rand`.

**[NORMATIVE]** `A-ANNEAL` (§09) runs `eps = 0` throughout. If the paper's finding
transfers, generation-0 lifetimes fail to train without it; if it does not transfer, that
is a reportable difference between vision and language settings and the annealing is
dropped.

## 5.6 Data

**[NORMATIVE]** TinyStories, tokenized at vocab 4096 (M-pico, M-nano) and 8192 (M1) with
the vendored SOMA tokenizer; the existing `data/ts_4096` and `data/ts_8192` shards are
reused unchanged, and their `tokenizer_hash` is recorded in every run manifest. Train /
validation / test splits are the existing ones. The held-out split used for fitness during
evolution is **disjoint** from the test split used for every reported number, so that
selection cannot overfit the reported metric.

## 5.7 Acceptance tests

**[NORMATIVE]** The lifetime is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| L1 | the decode is inside the loop | a mutant that hoists it out of the loop fails the suite (`G-MUTATE`) |
| L2 | `dL/dZ` reaches every tensor of §04.5 | non-zero gradient asserted per tensor |
| L3 | `Gamma` is not written in Darwinian arms | `Gamma` bit-identical before and after a lifetime |
| L4 | `eta` is detached | `eta.requires_grad` is `False` at the step; a mutant that attaches it fails |
| L5 | annealing endpoints | `eps(0) == 1`, `eps(N_life) < 1e-10`, and `W_1 == W_rand` exactly |
| L6 | schedule | `wsd_lr` warmup, plateau and decay boundaries are exact at `N_life` |
| L7 | determinism | two lifetimes with one `(Gamma, seed)` give bit-identical final `Z` and CE |
| L8 | resumability | checkpoint mid-run, resume, and match the uninterrupted run **bit-for-bit** — the whole lifetime, not only the stream. What is saved is `Z`, both optimizer states *including* the rule's step count `t` and the neuromodulator's running baseline, the stream position and the curves; a resume that restarted `t` would scale its first step by up to `1/(1-beta_1)`, and one that restarted the baseline would take a different `nu` immediately. A checkpoint refuses to load into a different architecture or a different token code |
| L9 | split disjointness | the fitness split and the test split share no document id |
| L10 | **one implementation** | `LifetimeRunner.run` is the only implementation of §5.1; `lifetime()` is a wrapper that calls it. L1..L9 bind the runner directly |
