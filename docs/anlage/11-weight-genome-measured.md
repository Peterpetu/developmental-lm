# 11. The weight genome, measured — and what a genome can carry

**Status: record and revision.** This section is written after R4 (§10, `RESULTS.md`), the
role-wise diagnostic, the manifold probe and the assimilation trial R4b. It records what the
neural-kernel weight genome of §03 was measured to do, derives from those measurements what
*any* genome that compresses adult weights can do in this loop, and revises the axioms of §00
accordingly. §12 specifies the design that follows. Nothing in §01–§10 is deleted: the
measurements are the reason the record exists (§00.4, RC6).

## 11.1 What was measured

All numbers are re-measured from the artifacts named in `RESULTS.md`; TinyStories stage 0
(`d 256, L 8, h 4, F 1024`, 13,011,201 parameters), 25M tokens, `sigma_e = 0.0060` (R1).

| measurement | value |
|---|---|
| trained stage-0 phenotype (R1 seed 0) | 1.857 nats |
| genome `(Gamma, Z)` at `d_g 64, H 4, d_z 16, d_h 256`: 418,288 + 300,096 reals, 18.1x | |
| regression (5.3), 3,000 Adam steps: residual on vector roles and convolutions | `< 0.01` |
| residual on every weight matrix (Q, K, V, O, OGATE, GATE, UP, DOWN) | 0.78 – 0.93 |
| residual on the tied embedding | 0.43 |
| **at-birth CE of the decode** | **8.53** (`L_null` 9.01; share 0.067; K-BIRTH fires) |
| trained phenotype with only the FFN decoded / only the lexicon decoded | 3.2 – 4.0 / 7.32 |
| decoded phenotype with the lexicon and all vectors restored to trained values | 12.01 |
| stage-1 expression of the stage-0 genome (K-SCALE) | 9.92, above `L_null` |
| manifold probe: `D(Gamma, Z)` trained end to end, 25M tokens | 9.011 from 5M tokens on; decoded rank 1.0 of 256 |
| assimilation trial R4b (anchor 0.01, refit every 100 steps) | lifetime 2.0021 vs 1.8571 free (+24 `sigma_e`); final decode 13.07; 49% overhead |

The effective ranks of the decoded matrices after regression (178–212 of 256 against 204–240
trained) show that the cubic term of (3.3) delivers the rank it was designed for; the failure
is not rank. The failure is content.

## 11.2 The information count

Fix one role matrix `W in R^{n x m}` of a trained phenotype. Under §03 its decode is

```
    D[i, j] = sum_h g( u_i^{(h)T} O^{(h)} v_j^{(h)} ),   u_i = f([b_i; z_i; r_l; y_l; +1]),  v_j = f([b_j; z_j; r_l; y_l; -1]),
```

so the set of decodable matrices for this `(l, m)`, with `Gamma` held fixed, is the image of a
smooth map from `R^{(n + m) d_z}` (the latents of its rows and columns; the innate codes are
constants). Its dimension is at most `(n + m) d_z`, against the `n m` reals of `W`; with `Gamma`
free and shared over the ~50 matrices of the stage, the total is `|Gamma| + |Z| = 718,384`
against 13,011,201: 5.5% of the phenotype's coordinates.

A trained matrix at this scale has a flat spectrum (erank 0.80–0.94 of `min(n, m)`, R4): it is
not close to any low-dimensional structured family that is not built from it. The standard
counting argument gives the scale of the loss. For a Lipschitz decoder on a bounded parameter
set of dimension `k`, the image has `epsilon`-covering number at most `(C/epsilon)^k`; the set
of matrices sharing `W`'s spectrum has covering number `(C'/epsilon)^{~nm}`; a generic member
of the latter is therefore at relative distance of order `1 - k/(nm)` in energy from the image
— i.e. a `k`-parameter family can carry about `k/(nm)` of a generic matrix's variance. With
`k/(nm) = d_z (n + m) / (n m) ≈ 2 d_z / d = 0.125` for the `256 x 256` roles and `0.078` for the
`1024 x 256` roles, the measured 7–20% of variance explained is the count, not a bug.

The same count says what raising the dials buys. `d_z = 128` gives `2 d_z / d = 1` — a genome
the size of the phenotype, no bottleneck. There is no setting of `(d_g, H, d_h)` that changes
the count: they shape the kernel, not the per-unit information. The consequence pre-registered
for K-BIRTH ("raise `d_g, H, d_z`, else the genome design is falsified") resolves to the second
branch without running the sweep, because the sweep's outcome is a counting identity.

## 11.3 Why no weight-compressing genome can be load-bearing in this loop

The point is not specific to the neural kernel. Let `G` be any genome of `M` reals with an
expression map `E`, fit to an adult `W_g` of generation `g` by any procedure, and let the next
generation be born as `W_{g+1,0} = E(G)` (the loop of §07.4 with `eps = 0`).

**Observation 1 (copying dominates at equal size).** If the next generation has the adult's
size, the alternative birth `W_{g+1,0} = W_g` (B-WARM, `eps = 1`) is available at zero cost and
is the adult itself. `E(G)` is a function of `W_g` through the fit, so any information in
`E(G)` is in `W_g`; `CE(E(G)) >= CE(W_g)` holds for every genome that is fit to `W_g` by a
procedure that does not use held-out data, up to the regularisation effect measured by K-ANCHOR
and K-TRANSFER. Birth competence is therefore never a *gain* over the warm start at equal size;
the most a perfect genome can do is tie it, at the price of `M` reals of storage saved — which
is not a training objective.

**Observation 2 (across sizes, the lossless expansion is available for the copied block).**
For growth `s -> s+1`, the entries of `W_{s+1}` with both indices in `A_s` can be copied
(Definition 2 step 2), and HyperCloning/G_stack-style expansion of the rest is function
preserving. A genome can only compete on the *new* entries, and there its content is the
genome's *prior over units*, which is small (the moments of §02.6) and does not need the
phenotype to be compressed to exist.

**Observation 3 (assimilation is a cost, not a benefit).** The PNAS loop obtains birth
competence by training the phenotype to be expressible (intermittent regeneration). In this
loop that is a constraint on the lifetime, whose cost is measured directly by K-ANCHOR
(`RESULTS.md` §R4b). A constraint on a data-limited learner cannot raise the adult's held-out
performance except through regularisation, and the regularisation of a per-unit-identity
prior was measured in GNOMON (§00.2) and here to be destructive, not helpful.

Together: **in a loop where the adult's weights can be copied, a genome that compresses those
weights has no mechanism by which it can improve the lineage's performance at equal compute.**
Its biological necessity (DNA is small; synapses are not inherited) has no counterpart in a
system that inherits synapses for free. What a genome *can* carry usefully is what is
**not copyable across the objects the lineage must span** — sizes, and, at production scale,
machines and runs — and what is **cheaper to acquire once than to re-learn every lifetime**.

## 11.4 What survived the measurement

Three parts of §02–§07 were measured to work and are kept:

1. **The vector genome.** Norm gains, gate biases, decay/write offsets, the logit scale and
   the convolutions were regressed to `< 1%` residual by the same kernel, as functions of
   (role, relative depth, unit code). These are the phenotype's *scale and gating profile*: a
   few thousand reals, scale-free by construction, the "innate" part of the phenotype in the
   sense of §01.3. Their at-birth value is a better-than-random initialisation of every stage;
   it is small and it is measured, not assumed (K-INIT in §12).
2. **The latent prior for growth.** New units sampled from the moments of trained units'
   statistics (§02.6) is a generative model of *units*, not of weights; it needs no
   compression of the phenotype and competes on K-GROW as before.
3. **The population machinery.** ε-lexicase over strata, CRN pairing, the task-vector merge
   (7.1)–(7.2) and its acceptance rule are independent of what the genome contains; their
   inputs are siblings born from one set of weights, which a warm start provides.

What is removed from the load-bearing path: the decode of weight matrices from identities
(§03 (3.1)–(3.6) for matrix roles), the regression of matrices (5.3 restricted to vectors),
`Z` for channel and token axes as a *carrier of weights* (it remains as the sampling prior of
growth), and K-BIRTH as stated (replaced by K-INIT).

## 11.5 Revised axioms

| # | axiom (§00.5) | revision |
|---|---|---|
| A1 | Sequential, not adversarial: the genome is written from the lifetime and read into the next. | Kept. What is written is the *program* (§12), fit by regression to the outcomes of a population of lifetimes, not the weights. |
| A2 | The bottleneck is priced, not imposed. | Kept, and now *satisfied by construction*: a program of `10^2–10^3` reals against `10^7–10^9` weights. The two-part code is program + weights; the weights are the second part and are inherited by copying within a lineage. |
| A3 | Full-capacity lifetime. | Kept, strengthened: the weights are never constrained to a decodable set (R4b measures why). |
| A4 | Population where individuals are cheap. | Kept; the siblings' *variation* is now in the program (data mixture, schedule, recipe), which is where the population has a measured non-monotone landscape to search (§12.3, G-LIVE). |
| A5 | Development is growth sampled from the genome. | Revised: development is growth *timed and shaped* by the genome (schedule `omega`, unit prior `Lambda`), with the copied block function-preserving. |
| A6 | One consolidation primitive across timescales. | Revised: the generation-level consolidation is the regression of the program onto the population's per-stratum outcomes (RegMix's regression is the instance), not of weights onto identities. |
| A7 | Measure before machinery. | Kept; it is the axiom that produced §11. |
| **A8** | **Heritable is what does not copy.** The genome carries what must span sizes, runs and machines without loss and what is cheaper to acquire once than to re-learn: the data programme, the developmental schedule, the recipe and the scale profile. What copies — the weights — is inherited by copying. | new |
