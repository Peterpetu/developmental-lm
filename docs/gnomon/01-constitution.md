# 01 — The constitution

GNOMON is a language model in which **every weight is an output of a decoder**, that
decoder's parameters are the **genome**, and the genome is searched by **evolution over an
open-ended archive**. The genomic bottleneck and architecture search are not features
added to a network; they are the object of which the network is an image.

The test each mechanism below had to pass: *remove it, and the model does not degrade —
it ceases to exist.*

## 1.1 The three parameter classes

**[NORMATIVE]** Every real number in a GNOMON run belongs to exactly one of three classes,
and the class determines who may write it.

| class | symbol | written by | size at M1 | persists across lifetimes |
|---|---|---|---|---|
| **germline** | `Gamma` | evolution only (§07, §08) | 5,306 | yes |
| **soma** | `Z = (z, zeta, y)` | lifetime learning only (§05) | 139,360 | no — reinitialised each lifetime |
| **innate** | `B` (codes) | nobody — derived from indices (§02) | 0 | n/a, they are a pure function |

**[NORMATIVE]** There is no fourth class. In particular there is **no free weight
tensor**: no `torch.nn.Parameter` holding a `W` that the phenotype uses directly. §04.6
makes this a mechanically checked absence with a planted positive control.

`Gamma` and `Z` are disjoint by construction (different objects, different optimizers, and
`Gamma` carries `requires_grad=False` during a lifetime except in the Lamarckian arm
`A-LAMARCK`, §08.5). `B` is a deterministic function of an index and a configuration; it is never
stored, never learned, and costs zero parameters.

## 1.2 C1 — Total decode

**[NORMATIVE]** For every weight matrix `W_{l,m}` of the phenotype, every entry is

```
    W_{l,m}[i, j]  =  g_Gamma( r_l , t_m , y_l ; b_i , z_i ; b_j , z_j )
```

where `g_Gamma` is the single decoder of §03, shared across **all** layers and **all**
matrix roles. `m` ranges over the matrix roles of §04.2; `l` over layers. Vectors (norm
gains, biases where present) are decoded by the same trunk with a degenerate column
argument (§03.6).

*Consequence, and the point of the mechanism:* randomising `Gamma` destroys every weight
in the network simultaneously. There is no `W_shared`. §00's RC1 is structurally
impossible.

## 1.3 C2 — Innate + acquired identity

**[NORMATIVE]** Each addressable entity — channel, token, layer — has two parts to its
identity:

* an **innate** code `b in {-1,+1}^n`, derived from its index (§02), identical in every
  lifetime and in every generation, costing nothing;
* an **acquired** vector `z in R^{d_z}`, initialised at birth and written only by
  lifetime learning.

**[NORMATIVE]** Lifetime learning writes `Z` and nothing else. The division of labour is
exact and is the architecture's central claim:

> the **genome** learns what a *relationship between two identities* means;
> the **lifetime** learns what each *thing is*.

For a language model this is the right split on its face: the *format* of a token
embedding is innate, its *meaning* is acquired. §06 measures whether the split holds in
fact rather than in intention.

## 1.4 C3 — The architecture is decoded

**[NORMATIVE]** The layer count `L`, and for each layer its operator type, head count,
FFN expansion and attention window, are outputs of an architecture head
`Gamma_a subset Gamma` reading the layer code `r_l` and global genome scalars (§07). They
are **innate**: `Gamma_a` takes no acquired input, so a lifetime cannot change the
architecture it was born with.

*Consequence:* mutating `Gamma` changes the computation graph. Architecture search and
genome search are one search, over one object, with one set of operators. RC5 is answered
not by adding a search but by removing the distinction.

## 1.5 C4 — Evolution is quality-diversity over genomes

**[NORMATIVE]** `Gamma` is improved only by §08's MAP-Elites archive: a population of
genomes, each evaluated by decoding a phenotype and running a short lifetime, placed in a
behaviour-space niche read off its decoded architecture, and varied by mutation and
crossover.

*Consequence:* the population lives in genome space, where 32 individuals of 5,306 reals
cost 663 KB, instead of inside one layer where SOMA's units were interchangeable. RC4 is
answered by moving the population to where fitness variance is heritable and
consequential — a genome's phenotype *is* its trait.

The archive is also where the requirement for **substantial open-ended search** is
discharged: MAP-Elites illuminates rather than optimises, and its coverage of behaviour
space is an explicit reported quantity (§08.5).

## 1.6 C5 — The genome share is a standing measurement

**[NORMATIVE]** Every certification point computes `share_retrain` (§06): the fraction of
attainable learning attributable to `Gamma` rather than to `Z`, measured by randomising
one and **re-training** the other from scratch. It is reported with every result, and
`KG-SHARE` (§09) fires below **0.50**.

*Consequence:* the failure mode of §00 is instrumented rather than hoped against. SOMA's
value was 0.024.

## 1.7 The Weismann barrier is compression, not policy

**[NORMATIVE]** SOMA policed a barrier: a registry, a write guard, a mutation test that
germline tensors were not written during a lifetime. GNOMON does not need one, and the
substitution is the deepest structural change in this design.

`|Gamma| : |phenotype|` is **1 : 9,458** at M1 (§03.5, and `G-SPECNUM` recomputes
it). A genome of 5,306 reals *cannot* memorise 49.9M weights; the only thing it can
carry across generations is the **regularity** in them. The barrier is therefore a
theorem about capacity rather than a rule about permissions, and no bug can violate
it. (The figure stood at `1 : 11,200` until v1.1 — an arithmetic overstatement of
the same true property, corrected by measurement rather than by argument.)

**[NORMATIVE]** The Lamarckian arm (`A-LAMARCK`, §08.5) deliberately opens a gradient path from the
lifetime loss into `Gamma`. Even there the barrier holds in the sense that matters: what
flows back is compressed 9,458-fold. This is why the arm is admissible science rather
than a violation, and it is labelled in every artifact it produces.

## 1.8 The two timescales

*(Constitutional requirement C1 of the project brief.)*

**[NORMATIVE]**

| timescale | writes | horizon | mechanism |
|---|---|---|---|
| **lifetime** | `Z` | `N_life` optimizer steps (10^2–10^4) | §05, gradient descent under §10.2's decoded plasticity rule |
| **germline** | `Gamma` | `G_gen` generations (10^1–10^2) | §08, MAP-Elites with mutation, crossover, optional ES |

They share a substrate: `W = g_Gamma(B, Z)` depends on both, multiplicatively and
inseparably. Neither can be removed. RC3 is answered.

**[NORMATIVE — a third timescale, within development]** `eps(n) = exp(-n/lambda)` (§05.5)
blends a random initialisation into the decode over the first steps of a lifetime, per the
PNAS paper's report that p-networks decoded from random g-networks are *"impossible to
train"*. This is developmental time and is distinct from both of the above.

## 1.9 Novel learning mechanisms

*(Constitutional requirement C2 of the project brief.)*

**[NORMATIVE]** Three, each load-bearing rather than decorative:

1. **Learning in identity space, not weight space.** The optimizer's variables are
   `d_z`-dimensional identities, 139,360 reals at M1, from which 49.9M weights are
   recomputed each step. No published language model learns this way.
2. **Decoded plasticity.** The genome emits a per-channel gain `eta_i` (§03.4) multiplying
   that channel's step. The genome therefore specifies *how much of what* is allowed to
   change — evolution of the learning process, not only of the learned.
3. **Open-ended quality-diversity architecture search** as the sole germline update (C4).

## 1.10 What GNOMON is not claimed to be

**[NORMATIVE — stated before any run, so it cannot be adjusted afterwards]**

* It is **not** claimed that a model with ~140K free parameters matches a 30M free-weight
  transformer on TinyStories. §09's baselines are matched on *free parameter count* and
  on *decoded parameter count* separately, and both numbers are reported.
* The compression ratios are **not** comparable to the PNAS paper's: theirs compress a
  network already trained on the target task; GNOMON's genome has seen the task only
  through the fitness of previous generations.
* The paper found that genomic compression **did not** accelerate learning, only raised
  the starting point. §09 records this as a prediction registered before the run.

## 1.11 Symbols

**[NORMATIVE]** The names below are binding; code identifiers mirror them.

| symbol | code | type | meaning |
|---|---|---|---|
| `Gamma` | `Gamma` | `R^{|Gamma|}` | the genome: every germline real (§03.5) |
| `Gamma_w` | `Gamma.w` | — | the weight-decoder subset of `Gamma` (§03) |
| `Gamma_a` | `Gamma.a` | — | the architecture-head subset of `Gamma` (§07) |
| `Z` | `Z` | — | all soma: `(z, zeta, y)` |
| `z_i` | `z` | `R^{d_max x d_z}` | acquired channel identity, row `i` |
| `zeta_v` | `zeta` | `R^{V x d_z}` | acquired token identity |
| `y_l` | `y` | `R^{L_max x d_y}` | acquired layer state |
| `b_i` | `b_chan` | `{-1,+1}^{n_c}` | innate channel code (§02) |
| `e_v` | `b_tok` | `{-1,+1}^{n_v}` | innate token code |
| `r_l` | `b_layer` | `{-1,+1}^{n_L}` | innate layer code |
| `t_m` | `b_role` | `{-1,+1}^{n_M}` | innate matrix-role code |
| `d_h` | `d_h` | `N` | decoder hidden width |
| `d_z` | `d_z` | `N` | acquired-identity width (the capacity dial, §06.4) |
| `d_y` | `d_y` | `N` | acquired layer-state width |
| `s` | `gain` | `R_+` | decoder input gain, `exp(log_gain)` |
| `sigma_{l,m}` | `sigma` | `R_+` | decoded per-matrix output scale (§03.3) |
| `eta_i` | `eta` | `R_+` | decoded per-channel plasticity (§03.4) |
| `eps(n)` | `eps` | `[0,1]` | developmental annealing coefficient (§05.5) |
| `share` | `share` | `[0,1]` | genome share, co-adapted estimator (§06.2) |
| `share_retrain` | `share_retrain` | `[0,1]` | genome share, retrained estimator (§06.3) |
