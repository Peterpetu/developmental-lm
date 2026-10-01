# 00 — The record of five models read for one question, the argument, and the constitution of the sixth

**Status: record and constitution.** Written 2026-09-10 after reading every model document
in this repository from its beginning (`docs/spec` 00–11 and its audit; `docs/gnomon` 00–10
and its audit; `docs/anlage` 00–12, README, RESULTS; `docs/tempo` 00–11, ARCHITECTURE-AS-BUILT;
`docs/bottleneck/00`; `docs/spore` 00–03), the founding conversation, and the run artifacts
under `runs/`. The question the user put: **how can the compression be made load-bearing?**
§0.1 states the question exactly; §0.2 is the record, reduced to what bears on it; §0.3 is the
first-principles argument (a theorem, its three exits, and the measured evidence for each);
§0.4 the transformative creative-reasoning pass; §0.5 the constitution and axioms of the
sixth model; §0.6 what is reused. Negative results are results (project rule).

PALIMPSEST (*palimpsest*: a manuscript scraped and written over, on which the earlier text
survives beneath the later) is the sixth model. Its genome is a small network; its
phenotype is written over by every environment the lineage lives in; what survives the
scraping is what the genome carried.

## 0.1 The question, and the reading of the brief

The repository's founding prompt (2026-09-02) asks for an architecture in which *evolution on
at least two timescales* is load-bearing, *intelligence is developmental*, the mechanisms are
*novel*, and the result is *SOTA on TinyStories and the Pile* on one 16 GB card. The lineage
of ideas is the **genomic bottleneck** (Zador 2019; Shuvaev, Lachi, Koulakov & Zador, PNAS
2024): a compact genome, decoded by development into a large network, that carries innate
competence and a simplicity bias. Five models put a compression mechanism beside, under or
around a language model, and in every one the compression was measured to be a *cost* at
matched compute. The user's question is therefore not "does the bottleneck work" — the record
answers that for the forms tried — but **under what construction can a lossy compression of
what a lineage has learned *improve* what the lineage does next.**

Also recorded, per the user's message of 2026-09-10: *growth was a success* — the one
mechanism the record reads positively (TEMPO `§P-D.42`: growth ahead by 0.02 after break-even
at `F ≈ 15`; ANLAGE R5: copy-growth equal to from-scratch at matched FLOPs, and generation 1
better than generation 0 through the inherited block; SPORE P-S.25: ridge-born units 0.046
over none). The sixth model keeps development as growth and makes it the *expression* of the
genome (§0.5).

**The constitution read.** The requirements are the repository's own (two timescales,
development, novel mechanisms, SOTA at matched compute, the scale path stated) with the
genomic bottleneck as *the* constitutional mechanism. SPORE's two constraints (HDC at the
core; no backpropagation) came from the sibling project's pasted prompt and were answered by
SPORE; they are not carried here, and the design is agnostic to the lifetime's learning rule
(§0.5 P3): a convex lifetime could be substituted, at SPORE's measured price of 0.55 nats.
One line redirects this if the user intends otherwise.

## 0.2 The record, reduced to the question

Every number is from the register named; `σ_e` is the seed noise under common random numbers
of the configuration it is quoted for.

| model | what was compressed, into what | the measurement that decides | reason recorded in its own ledger |
|---|---|---|---|
| **SOMA** (`docs/spec`, `docs/gnomon/00`) | nothing: a germline of scaffolding (20.5k reals) beside a gradient-trained soma; birth competence zero *by construction* | evolutionary machinery 0.04 % of learning; a trained parts library transplanted into a random model 8.87 vs 8.86 | RC1 "anything gradient descent can bypass, it bypasses"; RC2 the genome could not write weights |
| **GNOMON** (`docs/gnomon`) | every weight decoded from 5,306 reals by a per-entry MLP over identity codes; the lifetime writes 139k identity coordinates | best 5.22 nats vs 2.02 for a free 2.9M model; MAP-Elites one improvement in 432 evaluations; identity norms ran away | X2: the PNAS paper compresses an *already trained* network; a 10⁴-real genome as the sole substrate caps capacity (ANLAGE 00 §0.4) |
| **ANLAGE** (`docs/anlage/11`, RESULTS) | the adult's weights regressed into a neural-kernel genome (418k) + latents (300k), 18×; then the *programme* | at-birth 8.53 (null 9.01, trained 1.857); every matrix fit to 7–20 % of its variance — a counting identity (`k/(nm) = 2d_z/d`); the assimilated lifetime +0.145; K-LOOP fires at two generations (+0.011, +0.007) | **A8: heritable is what does not copy.** "In a loop where the adult's weights can be copied, a genome that compresses them has no mechanism by which it can improve the lineage at equal compute" (§11.3) |
| **TEMPO** (`docs/tempo/11`) | 512 numbers expressed into 243k gains by an 11.3M map, written by RL from a reversed episode | K-GENE fires against a random `Δg` (+0.013 [−0.057, +0.082]); 99.98 % of the gains have leverage below noise (sd 0.0006); the rule learned to *throttle* the learning rate; a kept genome forgets less under shift (retention −0.14 to −0.22), which is the same throttling | "leverage smaller than overhead loses at matched compute … it is the ratio" (§P-D.43) |
| **the faithful replication** (`docs/bottleneck/00`; `tempo/11 §P-D.44–57`) | Algorithm 1 as printed: 200 generations of train → refit g-networks by MSE → regenerate, `ε(n) = e^{−n/20}` | 107–175×: a random walk between 7 and 9.5 nats; the token table is the binding constraint (+4 nats alone; no topology on the token axis); **table carried, layers at 107× (5.9× whole): WORKS, 2.86 and 2.84 at birth on two seeds; lifetime +0.37**; **Fisher-metric refit alone WORKS at 107× (5.54)**; hot start on the source distribution worth ≈ 9M tokens and converging to the copied weights within 0.02; on foreign targets the source hinders and the decode hinders *less* (−0.06, and forgets less: 7.73 vs 9.78); the meta routine's decode is a generalist (−0.22 vs its own pre-regeneration state) but the free sequential control leads by 0.35 on the mixture | "the compression measures the task's information content, not the method … there is no redundancy to remove" (§P-D.44); "the constrained phenotype pays what the constraint always costs" (§P-D.57) |
| **SPORE** (`docs/spore`) | a seed and twenty dials regenerate the whole organism; a convex, backprop-free lifetime | 2.41 at 25M vs 1.857 (transformer) and 3.17 (trigram); K-EVO fires (a two-life sweep beats the selection loop); K-SOTA fires on the slope | the genome carries no learned content; the gap is representational |

**An artifact the record never absorbed** (found on this reading; `runs/bottleneck/xfer-heldout-from-*/transfer.json`, complete, written after the last ledger entry): on the *held-out* task family of the meta routine (GRID, and length 16, never trained), against three random starts at 25 / 50 / 100 % of a 5M-token budget — the free sequentially trained network's weights, **copied**, are `+1.70 / +1.43 / +1.33` nats *worse* than random; the genome-regenerated network, **decoded**, is `+0.105 / +0.008 / +0.007`: neutral. Nothing beats a random start on a family the source never saw; but the copy carries 1.3 nats of damage at the end of the budget and the decode carries none. Recorded as TEMPO `§P-D.58` and as `P-P.0` of this model's ledger.

**What the six compression mechanisms have in common.** Each was scored on one criterion:
*final held-out cross-entropy on a stationary corpus at matched compute, with the previous
weights available to copy.* On that criterion every one lost, by amounts from 0.007 to 4
nats, and the amounts track the constraint's tightness, not the method.

## 0.3 The first-principles argument

### 0.3.1 The copy theorem

Let a lineage produce adults `W_1, W_2, …`, each by a lifetime `W_g = L(I_g, D_g)` from an
initial state `I_g` on data `D_g`, with `L` a fixed learning procedure. A **genome** is any
object `G_g = C(W_g, G_{g−1})` with an expression `E(G_g)` of the phenotype's shape, and a
genomic lineage is one with `I_{g+1} = E(G_g)`; a **copying** lineage has `I_{g+1} = W_g`.

**Proposition 1 (copying dominates on a stationary criterion).** Suppose (a) `D_{g+1}` is
drawn from the same distribution as `D_g`, (b) the criterion is the expected held-out loss
of `W_{g+1}` at fixed compute of `L`, and (c) `L` is monotone in its start in the sense that
a start with lower expected loss on the distribution gives a lower expected final loss.
Then the copying lineage is at least as good as any genomic lineage whose expression is a
function of `W_g` alone, up to the regularising effect of the map `E ∘ C`.

*Proof.* `E(G_g)` is a function of `W_g`, so any information in `E(G_g)` about the
distribution is information in `W_g`; by (a) the distribution has not changed, so
`loss(E(G_g)) ≥ loss(W_g)` unless `E ∘ C` moves `W_g` to a point of lower expected loss on
the *same* distribution using no new data — which is the definition of a regulariser acting
on `W_g`. By (c) the start with the lower loss gives the lower final loss. □

The proposition is a restatement of ANLAGE §11.3 (Observations 1–3) with its assumptions
named, because the assumptions are where the answer lives.

### 0.3.2 The information count, restated

The regularising exit of Proposition 1 was the hope of every design here, and it fails for
a reason that is a count, not a bug (ANLAGE §11.2, confirmed by the replication's per-role
residuals at 107× and 21× alike): a trained language model's matrices have flat spectra
(effective rank 0.80–0.94 of full), and a `k`-parameter family carries about `k/(nm)` of a
generic flat-spectrum matrix's variance. A compression that pulls the weights toward a
structured family imposes *false* structure, and the lifetime pays for it (+0.145 at 18×,
+0.37 at 5.9×, +0.75 at 107× with the Fisher metric, +1.6–1.9 without it). The one
regulariser that is measured to help language models — weight decay — pulls toward zero,
which imposes no structure at all. **On the stationary criterion, the regularising exit is
closed by measurement at every ratio from 6× to 175×.**

### 0.3.3 The three exits

Proposition 1 has three assumptions; each is an exit, and the record has evidence at each.

**Exit 1 — no copy exists (the fresh start).** When a lifetime cannot start from the previous
weights (a new size that the copy cannot fill; a new machine; a new substrate; the first
lifetime), the genome competes with a *random* start, and there the record is positive:
the decode starts 1.14 nats ahead at 0.5M tokens and is worth about nine million tokens on
the source distribution (`§P-D.56`); the inherited scale profile is worth 0.014 after 2.5M
tokens (ANLAGE R4-INIT); the frequency-rank token code 0.10 (GNOMON §2.9). The gain is a
head start, and its value is bounded by the tokens it saves — small beside a lifetime.

**Exit 2 — the copy carries damage (non-stationarity).** When `D_{g+1}` differs from `D_g`,
`W_g` carries what the old environment needed *and* what it over-fitted, and the lifetime
on the new environment overwrites both: the classical results are catastrophic forgetting
(McCloskey & Cohen 1989), warm-start generalisation loss (Ash & Adams 2020: a warm-started
network generalises worse than one trained from scratch on the enlarged data, and *shrinking
toward the initialisation* repairs it), and loss of plasticity (Dohare et al. 2024:
re-initialising low-utility units restores trainability). In this repository: the copied
network is a hindrance of 0.32 on foreign tasks and the decode hinders 0.06 less
(`§P-D.55`); the copy forgets to 9.78 and the decode to 7.73; the copied free network is
1.33 nats worse than random on a held-out family while the decode is neutral (§0.2); the
regenerated network is 0.22 better on the task mixture than the network its last task left
(`§P-D.57`). **Every positive reading of compression in this repository is a reading of
damage the copy carried and the compression did not.** And every negative reading is on a
stationary criterion where there was no damage to filter.

**Exit 3 — the compression changes the compute.** A compressed structure that is *cheaper to
apply* (structured matrices, sparse activation, weight sharing across depth) buys capacity
per FLOP. Not tested here; the published record at this scale is negative for
mixture-of-experts below 100M under parameter matching (ANLAGE ledger §10.7) and for
low-rank structure against flat spectra. It is recorded as the exit this model does not take.

### 0.3.4 The conclusion the design is built on

> **Compression is load-bearing exactly when what it discards is harmful to what comes next.**
> On a stationary corpus with a copy available, nothing the lifetime learned is harmful to
> the next lifetime, and the theorem holds. Across environments, what the last environment
> over-wrote and what it over-fitted are harmful, and a bounded store that must keep the
> general and drop the specific is the mechanism that filters them. The genome's job is not
> to *carry* the phenotype; it is to *forget the right things*.

Two corollaries fix the design. First, the compression must be in **function space**, not
weight space: the weight-space decoders of GNOMON, ANLAGE and the replication fail on the
information count whatever their form, while a smaller *network* can match a larger one's
function on the data manifold (distillation: Hinton, Vinyals & Dean 2015; Busbridge et al.
2025's distillation scaling laws), and its capacity — not a decoder's family — is the
bottleneck. Second, the expression of the genome must be **development**: a function-
preserving expansion of the small network into the large one (HyperCloning, Samragh et al.
2024; the record's copy-growth), so that the phenotype is born computing exactly what the
genome knows and gains fresh, symmetric-broken capacity to learn the next environment —
"born small, grows" is the decompression.

The design's central claim is therefore falsifiable in a form the record has never tested:
**a double dissociation.** Under a shift of environments, the lineage's loss on the union of
its environments, as a function of the compression ratio `r ∈ {1 (copy), …, ∞ (scratch)}`,
should have an interior minimum; under no shift, the same protocol should be monotone with
its minimum at `r = 1`. If the first fails, compression is not load-bearing even where the
theorem allows it; if the second fails, the theorem is wrong and the record needs rereading.

## 0.4 The creative-reasoning pass (Universe of Thoughts, Tier 3 — transformative)

**Problem framing.** *One sentence:* make a lossy compression of what a language-model
lineage has learned improve what the lineage does next, measurably, at matched compute,
against the strongest conventional alternative, with development and two timescales
load-bearing. *Hard constraints:* the record of §0.2 (six losses on the stationary
criterion; the information count); one 16 GB card, one serial GPU job; `σ_e` first; every
mechanism against a matched control with a pre-registered kill criterion; short experiments.
*Tier:* 3 — the rules every prior design obeyed are the object.

### T1 — rules, explicit and hidden

| id | rule every prior compression design here obeyed |
|---|---|
| H1 | the object compressed is the **weights** (or gains) of the phenotype |
| H2 | the compressed object is used for **initialisation** (birth) or as an **anchor** (a pull during the lifetime) |
| H3 | the compression is toward a **family chosen a priori** (identity codes, kernels, coordinate g-networks) |
| H4 | the criterion is **final loss on the same distribution** at matched compute, with the previous weights available to copy |
| H5 | there is **one environment**: a life sees one corpus, i.i.d. |
| H6 | the genome is **read once** (birth) and **written once** (regression); nothing in between |
| H7 | what the genome discards is treated as **loss**; the benefit is looked for in what it keeps |
| H8 | the fair control is **copying** the previous weights |
| H9 | development (growth) is a **separate** mechanism from the genome's expression |
| H10 | the genome's store is **replaced** each generation (fit to the latest adult), not accumulated across environments |

### T2 — analogous rules from distant domains

| domain | the same tension | rule harvested |
|---|---|---|
| **statistics (shrinkage; James–Stein 1961)** | a biased estimator with less variance beats the unbiased one | *what compression discards can be the estimator's variance; the gain is in the discard* |
| **complementary learning systems (McClelland, McNaughton & O'Reilly 1995)** | a fast store learns the episode; a slow store must keep the general without the episode's interference | *the slow store is smaller and is written by replaying the fast one; forgetting in the fast store is by design* |
| **generative replay (Shin et al. 2017); self-synthesised rehearsal for LMs (Huang et al. 2024)** | a bounded memory cannot hold the past; a generative model of the past can be *sampled* | *the compressed store rehearses itself: dreams instead of data* |
| **knowledge distillation; born-again networks (Furlanello et al. 2018)** | a small network can match a large one's function on the data manifold, and the student can beat the teacher | *compress the function, not the weights* |
| **function-preserving expansion (Net2Net, Chen et al. 2016; HyperCloning, Samragh et al. 2024)** | a larger network must start where a smaller one finished, then learn what the small one could not | *decompression is growth: the same function, more capacity* |
| **shrink-and-perturb (Ash & Adams 2020); continual backprop (Dohare et al. 2024)** | warm-started weights generalise worse and lose plasticity; partial reset to the prior repairs both | *the copy carries damage; regeneration toward a prior removes it* |
| **evolution: the Weismann barrier as a filter** | offspring do not inherit the parent's injuries or acquired idiosyncrasies, only what passed the germ line | *the barrier is a sieve, and the organism benefits from what it refuses to transmit* |

### T3 — rule mutations

| mutation | new rule | opens |
|---|---|---|
| Drop H1 + H3 (distillation) | the genome is a **small network of the same family**, fit in function space by distillation | the information count no longer applies: the bottleneck is capacity, which distillation is measured to fill |
| Drop H4 + H5 (CLS, Weismann) | the criterion is the lineage's loss over a **sequence of environments**, with retention and adaptation measured; the control is the copy, which carries the damage | Exit 2 of §0.3.3 — where the record's positives are |
| Drop H7 (shrinkage, Weismann) | the **discard is the mechanism**: the genome must be small so that it cannot keep the specific | a testable frontier in the ratio `r`, with an interior optimum as the claim |
| Drop H9 (Net2Net) | **expression = development**: the small network is expanded function-preservingly into the phenotype, which is born computing the genome and grows into the environment | "born small, grows" is the decompression; growth (the record's positive) becomes load-bearing by construction |
| Drop H6 (generative replay) | the genome **dreams**: at consolidation it samples its own text, on which its successor is distilled from it; the phenotype may rehearse the dreams during the lifetime | no stored data of the past; the compressed store is the rehearsal |
| Drop H10 (CLS) | the genome **accumulates**: each consolidation distils the current phenotype *and* the previous genome (on its dreams) into the new genome | the store spans the whole life at bounded size |
| Drop H8 partly | the controls are the copy, the copy with **real** replay at the genome's byte size, the copy with EWC (Kirkpatrick et al. 2017), the copy with shrink-and-perturb, learning-without-forgetting (Li & Hoiem 2017, the *unbounded* store), from scratch, and the i.i.d. mixture as the upper bound | the claim is placed among the strongest conventional alternatives, not against a straw copy |

Kept: the strongest lifetime recipe this repository has measured (the 3:1 gated-delta /
attention hybrid, Muon, the constant-then-decay schedule); common random numbers; `σ_e`
first; every number re-measured from an artifact.

### T4 — candidates

**A, "the sieve as anchor"**: keep the replication's weight-space genome (GN50-E with the
Fisher metric) and use it only at environment switches. *Rejected on the record:* the
decode must be assimilated over hundreds of generations to carry the function (2.86 only
after 200; 7–9 after one), and the assimilation costs +0.37 on every environment.

**B, "the core with real replay"**: a small network distilled from the phenotype, expanded
by growth, plus a stored buffer of past data. *Kept as a control arm*, not the model: the
buffer is exactly what a genome exists to avoid, and its byte size is the honest match.

**C, "PALIMPSEST"** (selected): every mutation of T3 composed — a small core network as the
genome; distillation as consolidation, on the current environment's data and on the
previous core's dreams; function-preserving expansion as birth; the free lifetime on the
new environment; the frontier in `r` and the double dissociation as the claim.

### T5 — evaluation

| candidate | feasibility | utility | novelty | creativity |
|---|---|---|---|---|
| A | pass | 0.2 (the assimilation cost is measured) | 0.3 | 0.5 |
| B | pass | 0.6 (replay is the strongest continual method) | 0.2 (standard) | 0.8 |
| **C** | pass (every part exists or is a page of code; a life is minutes) | **0.7** conditional on Exit 2 | **0.8** (no published continual language model consolidates by distilling into a smaller network that is then expanded by function-preserving growth and rehearses from its own samples; the pieces are each published) | **1.5** |

**Core creative insight.** *The genome is a sieve, not a store: it is load-bearing when what
it refuses to transmit would have harmed the next lifetime, and the only place a language
model's lifetime leaves harm behind is where the environment changes. So compress the
function into a network too small to keep the specific, decompress it by growth, let it
dream its past, and measure the lineage across environments — with the copy, which keeps
everything, as the control that must lose.*

## 0.5 The constitution and axioms of PALIMPSEST

**[NORMATIVE]** Each is constitutional; a change is a change of design and is recorded.

| # | axiom | replaces |
|---|---|---|
| **P1** | **The genome is a network, and its compression is in function space.** The heritable object is a small phenotype of the same family (the *core*), written from the adult by distillation. No weight of the adult is ever regressed into a code. | H1, H3; GNOMON C1, ANLAGE §03 |
| **P2** | **Expression is development.** The phenotype is born from the core by a function-preserving expansion (width by tiling with halved fan-in, the embedding read-scale raised in step; depth by inserting layers with near-zero branch gains) and grows nothing else: at birth it computes the core's function exactly, and its extra capacity is what the environment fills. | H9; ANLAGE A5, TEMPO A5 |
| **P3** | **Full-capacity, unconstrained lifetime.** Within an environment the phenotype is a free-weight hybrid trained by the strongest measured recipe; no term of any loss pulls it toward the core except in the labelled rehearsal arm. The design is agnostic to the learning rule. | ANLAGE A3, TEMPO A3 |
| **P4** | **Consolidation accumulates by dreaming.** At every consolidation the new core is distilled from the current phenotype on the current environment's data *and* from the previous core on the previous core's own samples. No data of a past environment is stored. | H6, H10 |
| **P5** | **The discard is the mechanism.** The core's size is a dial `r` (the ratio of phenotype to core parameters), reported never targeted; the claim is a frontier in `r`, and `r = 1` is the copy. | H7; ANLAGE A2 |
| **P6** | **The criterion is the life, not the lifetime.** A lineage is scored on the union of the environments it has lived in, on retention after a switch and on adaptation at a switch, at matched compute, against the copy, the copy with real replay at the genome's byte size, EWC, shrink-and-perturb, learning-without-forgetting, from scratch, and the i.i.d. mixture. | H4, H5, H8 |
| **P7** | **The double dissociation is the claim.** Compression is load-bearing iff the frontier in `r` has an interior minimum under a shift of environments *and* is monotone at `r = 1` without one. A one-sided result is reported as what it is. | new |
| **P8** | **Measure before machinery; heritable is what does not copy.** `σ_e` first; every contrast paired under common random numbers; every criterion pre-registered in `02-protocol.md`; every number re-measured from disk. A8 is kept with its content made concrete: what does not copy *usefully* across environments is the general, and the sieve is what separates it. | ANLAGE A7, A8 |

**Two timescales.** The lifetime (gradient descent within an environment) and the genome
(distillation and dreaming at consolidation) share the substrate — the phenotype is born
from the core and the core is written from the phenotype — and neither can be removed
without the other's object changing. K-SIEVE decides whether the slow one is load-bearing.

**Development.** Growth is the expression (P2); its controls are the truncation of ANLAGE
(the adult's block, no distillation) and random new units (K-GROW-C).

**Novel mechanisms.** Consolidation by distillation into a smaller network that is then
expanded function-preservingly; the core's generative rehearsal of its own past; the
frontier in the ratio as the object of the claim. Each part is published; the composition as
a lineage of language models is not (T5).

**SOTA.** Read as the strongest model *at the end of a life* on *every* environment of that
life, at matched compute, against the conventional continual controls and against the
i.i.d. mixture that needs all data at once (K-SOTA-C, `02`). The stationary single-corpus
number is reported beside it as the theorem predicts: the copy.

## 0.6 What is reused, and why

| component | path | reuse |
|---|---|---|
| the phenotype (3:1 gated-delta/attention hybrid, tied head, softcap, z-loss) and its recipe (Muon, AdamW, schedule, batch warm-up, micro-batching) | `anlage/model.py`, `anlage/ops.py`, `anlage/recipe.py`, `gnomon/pheno/mixer_chunk.py` | the lifetime (P3), unchanged |
| the lifetime trainer with its hooks, the evaluation with strata | `anlage/lifetime.py` | the environment lifetime |
| shards, window pools under CRN, the v3 evaluation sets, token counts and frequency rank | `gnomon/data/*`, `screen/crn.py`, `anlage/data.py`, `scripts/anlage_common.py` | environments (`01 §1.1`) |
| the primordium block (`shrink`) | `anlage/grow.py::shrink` | the truncation control of K-GROW-C |
| the diagonal empirical Fisher | `bottleneck/loop.py::fisher_diag` | the EWC control |
| the reserved-alphabet verifier stream | `bottleneck/transfer.py::task_pool, task_eval_set` | the third environment |
| paired contrasts, `t` intervals, BCa, TOST | `screen/variance.py`, `gnomon/utils/stats.py` | the protocol |

Not reused: every weight-space decoder (`bottleneck/genome.py`, `anlage/genome.py`,
`gnomon/genome`), the gain genome, the consolidator, the population and merge, the mixture
programme. They are the record.

## 0.7 The process rule

**[NORMATIVE]** The specification (`01`, `02`) is written before the code; every acceptance
row is owned by a test; `σ_e` of every environment is measured before any contrast; runs are
short (a life is minutes) and iterated; the bottleneck queue of `runs/bottleneck/` stays
paused by the user's standing order; one serial GPU job per card; `screen/landscape.py` is
the user's and is not touched.
