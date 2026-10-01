# 01 — The constitution

ANLAGE is a language model in which **a neural genome is fit by regression to whatever a
lineage has learned, birth decodes it, growth samples it, siblings mutate it, merging
recombines what they learned, and every timescale consolidates into the next by one
least-squares primitive.** The genome and the developmental program are not features added
to a network; the network is one expression of them.

Read this file first; it is normative for the rest. Symbols are fixed in §1.8.

## 1.1 The four parameter classes

**[NORMATIVE]** Every real number in an ANLAGE run belongs to exactly one class.

| class | symbol | written by | lives | size at the Pile adult (§04.5) |
|---|---|---|---|---|
| **phenotype** | `W = {W_{l,m}}` | the lifetime (gradient descent, §04, §06) | one lifetime, then compressed into `Gamma, Z` and discarded | ≈ 3.5 × 10⁸ |
| **genome** | `Gamma` | regression from a phenotype (§05); mutation (§07) | across generations | ≈ 5 × 10⁵, independent of phenotype size |
| **latent bank** | `Z = {z_a}, {y_l}` | regression (§05); sampling at growth (§06) | across generations; grows with the phenotype | `d_z |A_s| + d_y L_s` |
| **innate codes** | `B = {b_a}, {r_l}, {t_m}` | nobody — pure functions of index and configuration (§02) | forever | 0 |

**[NORMATIVE]** There is no fifth class. The phenotype is a set of free `torch.nn.Parameter`
tensors and **nothing else** trains during a lifetime (A3). `Gamma` and `Z` carry
`requires_grad=False` during a lifetime; they receive gradients only inside `REGRESS` (§05)
and inside `MERGE`'s coefficient fit (§07). `G-NOFREE-INVERSE` (§08) asserts that no decode
runs inside a training step.

## 1.2 The four rates

**[NORMATIVE]**

| rate | object | horizon | mechanism | section |
|---|---|---|---|---|
| **token** | fast-weight state `S_t` of each delta-rule head | one sequence | the delta rule, one online least-squares step per token | §04.3 |
| **step** | `W` | `N_s` optimiser steps per stage | Muon / Adam on next-token CE | §04.6 |
| **sibling** | the population `{W_k}` at stage 0 | `T_sib` tokens per sibling | variation, ε-lexicase over strata, merging | §07 |
| **generation** | `Gamma, Z` | one loop of §07.4 | regression of the adult into the genome | §05 |

They share a substrate: `W` at birth is `D(Gamma, Z)`, `W` after growth carries decoded new
units, and `Gamma, Z` after a generation are a function of `W`. Neither timescale can be
removed without the other's object changing — which is the property SOMA lacked (RC3) and
GNOMON purchased by starving the lifetime (X3).

## 1.3 Constitutional requirement C1 — two evolving timescales

*(Requirement 1 of the project brief: "an AI architecture which utilises evolving in at least
2 timescales, the individual lifetime and the genome lifetime; intelligence is developmental.")*

**[NORMATIVE]** C1 is discharged by three mechanisms, each with a control and a kill criterion
(§08.3):

1. **Birth from the genome.** Generation `g+1`'s siblings start from `D(Gamma_g, Z_g)` at
   stage 0. Control: the same run started from a plain warm start of generation `g`'s
   stage-0 weights (`eps = 1` in §05.4). Kill: K-LOOP.
2. **Growth sampled from the genome.** Every growth event decodes its new units from
   `(Gamma, Z)` with latents drawn from the fitted prior (§06). Control: copy-based growth
   (HyperCloning / G_stack) and random initialisation of the new units. Kill: K-GROW.
3. **Consolidation into the genome.** The adult is regressed into `(Gamma, Z)` (§05); the
   genome accumulates what the lineage has learned, at fixed size. Control: none needed —
   the mechanism is what makes 1 and 2 possible; its own gate is K-BIRTH.

**[NORMATIVE — the developmental thesis, stated as the claim it is]** "Neurons are able to
function as they do only after the brain has developed to a certain form" is realised as:
a phenotype born from a genome and grown in stages reaches a lower held-out loss at equal
compute than one trained from a random initialisation at its final size. That is K-SOTA
against B-FLAT, and it is the claim the whole design is held to.

## 1.4 Constitutional requirement C2 — novel learning mechanisms

*(Requirement 2 of the brief: "novel mechanisms or combinations which have not been tried
before in LLM next-token prediction; e.g. part purposeful training, part open-ended search.")*

**[NORMATIVE]** Three, each load-bearing, none decorative; the 2026 survey in §10 finds no
published instance of any of them in language-model pretraining:

1. **A neural kernel genome fit by regression, expressed by sampling** (§03, §05, §06). The
   genome is a conditional generative model of *neurons* — weights conditioned on identity —
   not a compressed copy of one network. Growth is a draw from it.
2. **A primordium-stage population with case-based selection and weight-space crossover**
   (§07). The individuals are cheap because they are small and short-lived; they are
   mergeable because they share a birth; selection is over held-out strata, so specialists
   survive; the merged elite is what grows.
3. **Three-level consolidation by one primitive** (§05 Definition 1): the delta rule, gradient
   descent and genome regression are the same least-squares projection at three horizons,
   and the design treats them as one object with three instances.

**[NORMATIVE — the compute split is a dial.]** The share of a generation's compute spent on
the population, `C_pop / (C_pop + C_adult)`, starts at 0.25 and is raised only if K-POP shows
the population paying at the margin (§07.5). No fraction is assumed.

## 1.5 Generation zero is the baseline

**[NORMATIVE]** Generation 0 has no genome to be born from. Its stage-0 siblings start from
the free-weight initialisation of §04.7, and its adult is exactly the free-weight baseline
recipe (B-FLAT when no growth is used, B-COPY when copy-based growth is). The first genome is
`REGRESS(W_adult^{(0)})`. Consequently **the control and the treatment are one code path**
distinguished by the expression dial `eps` (§05.4) and the growth-initialisation switch
(§06.3): nothing about the comparison depends on a separately maintained baseline.

## 1.6 What ANLAGE is not claimed to be

**[NORMATIVE — stated before any run, so it cannot be adjusted afterwards]**

* It is not claimed that the genome carries language competence by itself. The at-birth
  loss of `D(Gamma, Z)` is reported at every generation as the paper's own headline quantity
  (§05.5); it is a *measurement*, and generation 1's at-birth loss is expected to be far
  above the adult's.
* It is not claimed that compression accelerates the lifetime's learning trajectory; the
  PNAS paper found it does not (§10 E-1). What is claimed, and tested, is that birth,
  growth and the population change the *starting points* of successive stages enough to
  win at equal compute.
* It is not claimed that any of this matches models trained on 10²–10³ times more tokens.
  Published numbers are reported as context in bits per byte; every comparative claim is at
  matched compute on this hardware (§08.2).
* Nothing is claimed about reasoning, agentic or long-horizon capability below ~10⁹
  parameters; §09 states what the scaling path would have to demonstrate.

## 1.7 The Weismann barrier, reconsidered

GNOMON argued that a small genome *cannot* memorise the phenotype, so no barrier need be
policed. ANLAGE inherits the argument and adds the price: `Z` is large enough to memorise
per-unit detail, and the Gaussian prior on `Z` (§05.2) is what stops it from doing so
except where the detail is worth its description length. The barrier is therefore neither a
rule (SOMA) nor a capacity theorem alone (GNOMON) but a **cost**: Proposition 1 of §05.

## 1.8 Symbols

**[NORMATIVE]** Binding; code identifiers mirror them.

| symbol | code | type | meaning |
|---|---|---|---|
| `s` | `stage` | `0..S` | developmental stage; 0 is the primordium |
| `d_s, L_s, F_s, h_s` | `d_model, L, d_ff, n_h` | naturals | residual width, depth, FFN width, heads at stage `s` |
| `d_head` | `d_head` | natural | head dimension, 64 at every stage |
| `n_kv(l)` | `n_kv` | natural | KV heads of layer `l`: `h_s` for delta layers, `max(1, h_s/4)` for attention |
| `V` | `vocab` | natural | vocabulary |
| `T` | `T` | natural | context length |
| `W_{l,m}` | `W[(m, l)]` | tensor | phenotype weight of role `m` in layer `l` (`l = -1` global) |
| `M` | `ROLES` | set | the 20 roles of §04.2 |
| `A_s` | — | set | addressable units at stage `s`; `A_s ⊂ A_{s+1}` on channel and token axes |
| `b_a` | `codes.*` | `{-1,+1}^n` | innate code of unit `a` |
| `r_l` | `layer_codes` | `{-1,+1}^{n_L+1}` | relative-depth Gray code with the global flag |
| `t_m` | `role_codes` | `{-1,+1}^{n_M}` | role code |
| `z_a` | `Z.*` | `R^{d_z}` | acquired latent of unit `a` |
| `y_l` | `Z.y` | `R^{d_y}` | acquired latent of layer `l` |
| `Gamma` | `Genome` | — | `(theta_f, {O_m}, {kappa_m}, {o_m}, theta_sigma, omega)` |
| `u_{a,l}^{(h)}, v_{a,l}^{(h)}` | `feat_row, feat_col` | `R^{d_g}` | row-side / column-side features of unit `a` in layer `l`, head `h` |
| `O_m^{(h)}` | `O` | `R^{d_g x d_g}` | interaction matrix of role `m`, head `h` |
| `kappa_m` | `kappa` | `R_+` | cubic kernel coefficient of role `m` |
| `sigma_{l,m}` | `sigma` | `R_+` | decoded per-matrix scale |
| `s_target(m, shape)` | `s_target` | `R_+` | innate initialisation scale of a role |
| `D` | `Genome.decode` | map | `(Gamma, Z, s) -> W` |
| `J` | `regress.objective` | scalar | the regression objective (5.3) |
| `eps` | `eps` | `[0,1]` | expression dial: `W_0 = eps W_prev + (1-eps) D(Gamma, Z)` |
| `rho(n)` | `ramp` | `[rho_0, 1]` | growth safety ramp |
| `omega` | `Schedule` | — | developmental schedule `{(d_s, L_s, F_s, h_s, T_s)}` |
| `K, T_sib, T_adult` | — | naturals | siblings per generation; sibling and adult token budgets |
| `C` | `strata` | set | held-out strata used as selection cases |
| `sigma_e` | `sigma_e` | nats | pooled seed noise under CRN at a configuration |
| `S_t` | `S` | `R^{d_v x d_k}` | delta-rule fast-weight state |
