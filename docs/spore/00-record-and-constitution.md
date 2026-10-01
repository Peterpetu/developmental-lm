# 00 — The record of four models, the constitution of the fifth, and the creative-reasoning pass that produced it

**Status: record and constitution.** Written 2026-09-09 (22:00) at the start of an autonomous
night, before any code of the fifth model existed. §0.1 reads the brief; §0.2 is the record
that constrains any design here; §0.3 is the first-principles argument; §0.4 the explicit
creative-reasoning pass (Tier 3, transformative); §0.5 the constitution and axioms; §0.6 what
is reused. Nothing is softened: negative results are results (project rule: research integrity).

SPORE (*spore*: a complete organism folded into a seed; everything it becomes is regenerated
from a few bytes and one growing season) is the fifth language-model design in this
repository, after SOMA (`main`), GNOMON (`gnomon`), ANLAGE (`anlage`) and TEMPO (`tempo`, with
the faithful genomic-bottleneck replication in `bottleneck/`).

## 0.1 The brief, and how it is read

The message that opened this night pasted the founding prompt of the sibling project
`HDC_LM` (its two constitutional items: *hyperdimensional computing at the core*;
*backpropagation-free learning*) and asked for a fresh start in **this** repository, whose
own founding prompt (2026-09-02, the first message of session `924c942c`) had two different
constitutional items: *evolution on at least two timescales, an individual lifetime and a
genome lifetime, intelligence being developmental*; and *novel learning mechanisms or
combinations never tried in next-token prediction, open-ended search included*. Both prompts
otherwise coincide word for word (whitepaper first; first principles; reset the axioms;
16 GB card; TinyStories and the Pile; a path to production scale; the transformative
creative-reasoning pass; plan documents; rigorous implementation).

**[NORMATIVE] The reading adopted.** The fifth model satisfies *both* sets of constitutional
items, and §0.3 argues that this is not a compromise but a necessity: the record of this
repository shows that the two-timescale, developmental design cannot be load-bearing while
backpropagation is available to the lifetime, and the sibling project's constraint is the
condition under which it can be. If the user intended only one of the two readings, every
result below still stands under it; the interpretation is recorded here so it can be
redirected in one line.

Requirements as read (each is a hypothesis with a criterion in `02-protocol.md`):

| id | requirement | operational meaning |
|---|---|---|
| C1 | **HDC at the core, not decorative** | representations are seeded hypervectors; context is built by binding and bundling; memory is superposition; ablating any of these must degrade the model (K-HDC) |
| C2 | **backpropagation-free learning** | no error signal crosses a nonlinearity; every parameter is fit by a closed-form or single-layer local rule (G-BPF asserts it mechanically) |
| C3 | **two timescales, both load-bearing** | the lifetime writes readouts and memories by convex rules only; the genome (seeds and dials) is the *only* object that can change structure, and it is searched by selection over lifetimes (K-EVO) |
| C4 | **development** | the organism is born small and grows units over its life; a unit is fit to what the organism cannot yet predict and is frozen after birth (K-DEV) |
| C5 | **novel mechanisms** | the combination — seeded HDC substrate, closed-form lifetime, delta-rule in-context memory, cascade growth, evolved dials — has no published instance as a next-token model (§0.4 T5) |
| C6 | **SOTA on TinyStories and the Pile; throughput; scale path** | the gap to the strongest backprop model at matched tokens is measured and reported (K-SOTA), with its slope in `D`, `F` and tokens; the scale path is stated, not claimed (`01 §1.9`) |

## 0.2 The record that constrains any design here

Every number is from the registers named; the ledgers are `docs/gnomon/00-postmortem.md`,
`docs/anlage/RESULTS.md`, `docs/anlage/11-weight-genome-measured.md`,
`docs/tempo/11-evidence-ledger.md` (§P-D.1–57), `docs/bottleneck/00-replication.md`.

| model | what it was | what was measured | root cause named in its own ledger |
|---|---|---|---|
| **SOMA** (v3.1, 25k lines, 474 tests) | a transformer with a routed population of mortal units under ε-lexicase, an ES germline of scaffolding, a metabolic price | the evolutionary machinery contributed **0.04 %** of learning; the unit population 2.4 %; one dense bypass 6.98 of 6.76 nats; a trained parts library transplanted into a random model scored 8.87 vs 8.86 random | **RC1: anything gradient descent can bypass, it bypasses.** RC2–RC5: the genome could not write weights; the timescales shared no substrate; selection had nothing heritable to select on; function-preserving events were no-ops |
| **GNOMON** (275 tests) | every weight decoded from a 5,306-real genome by a per-entry MLP over identity codes; MAP-Elites over the genome | best **5.22** nats against **2.02** for a free 2.9M transformer; MAP-Elites: one improvement in 432 evaluations; identity norms ran away (`zeta` × 188,401) | per-token information had to reach 49.9M weights through 139k saturating coordinates; the genome capped capacity (axiom X2 of ANLAGE 00 §0.4) |
| **SCREEN** | the instrument: `σ_e` under common random numbers; landscape topology | `σ_e = 0.0100` (GNOMON reasoned with 0.171); operator and sharing axes monotone above noise (`h2 = 0.991`, hill-climbable in ≤ 4 flips); a priced fitness has `σ = 0` and the search optimises the price | *an axis monotone above noise is a hypothesis with a known answer, not a search space* |
| **ANLAGE** | free-weight hybrid lifetime; genome by regression of the adult; growth; siblings; programme | one regression of the adult decodes to **8.53** (null 9.01): a trained matrix has a flat spectrum and 16 latents per unit carry 7–20 % of a row (a counting identity, `11 §11.2`); the assimilated lifetime cost **+0.145**; four siblings merged lose **0.115** to one long life; the mixture programme **+0.002** (noise); the LR axis 0.09; copy-growth = from-scratch at matched FLOPs | **A8: heritable is what does not copy.** In a loop where the adult's weights can be copied, a genome that compresses them can only lose |
| **TEMPO** (275 tests) | lifespan agent, three timescales, gain genome written by RL from a reversed episode, latent morphology, criticality, continuous span interface | K-CONT fires at every `K` (+0.55 at `K = 4`); K-GENE fires against a random `Δg` (the policy learned to *throttle* the learning rate); 99.98 % of the genome's outputs have leverage below noise; growth pays after `F ≈ 15` but a coin beats the learned timing; criticality drift bounded without the rule | leverage below overhead loses at matched compute, and the ratio does not improve with scale (`§P-D.43`); the load-bearing hidden rules were R3 (from scratch on this card) and R8 (spec before number) |
| **the faithful bottleneck replication** (Shuvaev et al. 2024, Algorithm 1 as written) | 200 generations of train → refit g-nets → regenerate | at 107–175× the loop is a random walk (7–9.5 nats, between the thresholds); the token table is the binding constraint (+4 nats alone; no topology on the token axis); **with the table carried (5.9×) the loop WORKS: 2.86 at birth on two seeds; Fisher-weighted refit alone WORKS at 107× (5.54)**; hot start worth ~9M tokens on the source distribution; transfer to foreign targets: the source hinders, the decode hinders less; the meta routine's decode is a generalist that beats *copying* a sequentially trained network on a new family by 1.3–1.6 nats but does not beat a random start | the paper's product is a simplicity bias that transfers, obtained by repeatedly forcing the phenotype onto a compressible set; on a language model the constrained lifetime pays +0.37 nats at best |

**What drives learning on this repository's own numbers** (`tempo/11 §P-D.43`): compute through
gradient descent (0.050 nats per doubling); the optimiser and schedule (0.09 nats across the LR
axis); representation granularity (0.55 nats); the 3:1 gated-delta/attention hybrid (won 6 of
6); growth timed to the budget (+0.02 after break-even). *Not* drivers, measured: channels
carrying weights or gains; populations; assimilation; continuous interfaces; criticality
control; RL-trained update rules fitted inside one life.

**Operational record** that binds this night: one serial GPU job per card; `σ_e` measured
before any contrast; sweep contrasts across budgets, never one point; `pkill` is inert here
(enumerate with `Get-CimInstance`, kill with `Stop-Process`); two heredocs in one command
break the tool; the pre-existing modification to `screen/landscape.py` is the user's; the
bottleneck queue is **paused** by the user's order and stays paused.

## 0.3 The first-principles argument

**The observation that every ledger repeats.** Four designs put an evolutionary or genomic
mechanism *beside* a substrate that backpropagation could train, and in every case the
mechanism's leverage was below its overhead at matched compute. The ledgers name it as a
ratio (per-FLOP information: a gradient assigns credit to every parameter every step; a
selection loop returns one scalar per lifetime) and as a bypass (RC1: the gradient finds the
path that does not run through the mechanism). Both are consequences of one fact: **as long
as the lifetime can change the structure by gradient, the genome has nothing to do that the
lifetime does not do better.**

**The consequence.** A design in which the two timescales are load-bearing *by construction*
must make the lifetime unable to change structure. The cleanest such lifetime is a **convex**
one: given the structure, every learnable object is the solution of a convex problem
(least squares, ridge regression, multinomial logistic regression of a linear readout) —
solved in closed form or by a single-layer local rule. Then:

1. the lifetime needs no backpropagation (C2) — the chain rule never crosses a
   nonlinearity, because no nonlinearity has learnable parameters upstream of another
   learnable object; the developmental rule of `01 §1.6` is itself a convex fit to a residual;
2. the structure — which features exist, how context is bound, how memory decays, how the
   organism grows — is *unreachable* from inside the lifetime, so the genome that specifies
   it is the only thing that can improve it (C3), and no bypass exists;
3. a convex lifetime on 25M tokens costs minutes rather than GPU-hours, so the outer loop's
   information rate rises from bits per GPU-hour (ANLAGE X4) to bits per minute, which is
   the regime in which selection has ever moved anything;
4. the genome can be tiny **by construction, not by fitting**: a hyperdimensional substrate
   is regenerated from seeds (Kanerva 2009 §2; Plate 1995 §II), so the two-part code of
   ANLAGE A2 becomes *seed and dials* + *what one closed-form pass writes*, and "at birth"
   the organism is a complete, valid predictor with an innate in-context memory (`01 §1.5`)
   before it has read a token.

**The price, stated before it is measured.** A convex lifetime over fixed or grown features
is a random-features / kernel model (Rahimi & Recht 2007), and the 2026 record of such
models on language is weak against end-to-end gradient training (the sibling project's own
backprop-free result: 3.78 nats against its own gradient twin's 3.47, both far above the
1.80 a 13M transformer reaches here). SPORE does not assume that gap away. It puts three
mechanisms against it that a fixed random-feature model lacks — a **delta-rule in-context
memory** (an exact online least-squares n-gram learner with no trained projections, the
operator this repository verified to 1e-10 in `mixer_chunk.py`), **developmental growth**
of units fit one at a time to the residual (cascade-correlation, Fahlman & Lebiere 1990,
with random candidate generation), and **selection over the structure** — and it measures
the gap to the backprop model at matched tokens as its central number (K-SOTA), with the
slope of that gap in `D`, `F` and tokens as the scale-path claim.

## 0.4 The creative-reasoning pass (Universe of Thoughts, Tier 3 — transformative)

**Problem framing.** *One sentence:* design a next-token language model whose representations
are hyperdimensional, whose lifetime learning uses no backpropagation, whose structure is
searched by selection over cheap lifetimes, and which develops by growing units — and make
each of those load-bearing on TinyStories within one night of short experiments on one card.
*Hard constraints:* C1–C6; the record of §0.2; one GPU job at a time; `σ_e` first; every
mechanism against a matched control with a pre-registered kill criterion.
*Tier:* 3 — the record indicts the rules the four prior designs shared, not their details.

### T1 — rules, explicit and hidden

| id | rule every prior design here (and standard LLM practice) obeyed |
|---|---|
| H1 | credit is assigned by backpropagation: the chain rule crosses every layer |
| H2 | representations are learned parameters (the embedding table is trained) |
| H3 | the trainable object is the weights; the structure is fixed by hand |
| H4 | a lifetime is expensive, so the outer loop is starved (bits per GPU-hour) |
| H5 | depth is a stack of jointly trained differentiable layers |
| H6 | corpus knowledge lives in weights; context lives in a cache; they are different mechanisms |
| H7 | the fitness of a structure is known only after training it |
| H8 | randomness is noise to be averaged out; seeds are incidental |
| H9 | a genome must be *fit* to a phenotype (the bottleneck paper's refit loop) |
| H10 | a novel mechanism must beat backprop on backprop's own substrate |

### T2 — analogous rules harvested from distant domains (far-then-analogical)

| domain | the same tension | rule harvested |
|---|---|---|
| **random features / kernel methods** (Rahimi & Recht 2007) | expressive features vs. a tractable fit | *fix the representation randomly, solve the readout exactly; choose the kernel by validation* |
| **hyperdimensional computing / VSA** (Kanerva 2009; Plate 1995; Gayler 2003) | store structure without learning | *random high-dimensional atoms are as good as designed ones; binding is invertible; superposition stores; a seed regenerates any atom* |
| **cascade-correlation** (Fahlman & Lebiere 1990) | depth without joint training | *grow one unit at a time, fit it to the residual, freeze it* |
| **gradient boosting** (Friedman 2001) | depth in function space | *each stage is a convex fit to the pseudo-residual of the ensemble* |
| **clonal selection** (the immune system) | find good units in a huge space | *generate many random candidates, select by affinity, expand the winners* |
| **procedural generation from a seed** (a world from 64 bits) | a large object from a tiny description | *the genome is a seed; the Kolmogorov complexity of the substrate is the seed's length* |
| **reservoir computing** (Jaeger 2001) | dynamics vs. training | *fixed random dynamics, a trained linear readout; the dials of the dynamics are tuned by search* |
| **recursive least squares / the delta rule** (Widrow & Hoff 1960; Schlag et al. 2021) | learning inside a sequence | *the exact online least-squares update is a local rule and is the same operator as linear attention* |
| **count-min sketch / Bloom filters** | count in less space than the table | *hash into superposition; accept bounded collision noise* |
| **random indexing** (Sahlgren 2005) | word meaning without gradients | *a word's vector is the bundle of the contexts it occurs in; Hebbian, one pass* |

### T3 — mutations

| mutation | new rule | what it opens |
|---|---|---|
| **drop H1** | no error signal crosses a nonlinearity; every parameter is fit by a closed-form or single-layer local rule | C2 by construction; training cost ≈ one forward pass |
| **drop H2, H8** | representations are *generated from seeds*; what must be learned about tokens is learned by counting (random indexing), not by gradient | the genome is a seed (C1, C3); the vocabulary table — the binding constraint of the bottleneck replication — no longer needs compressing |
| **vary H3, H7 by H4** | the lifetime is convex and costs minutes, so structure is the genome's object and its fitness is cheap | selection is affordable *and* unbypassable (C3) |
| **vary H5** (cascade-correlation, boosting) | depth is developmental: units are added over the life, each fit to the residual, each frozen after birth; the schedule and the candidate generator are genomic | C4; "a neuron functions only once the substrate has developed" is literal |
| **drop H6** (RLS = linear attention) | corpus knowledge is a least-squares readout; within-document knowledge is a least-squares fast memory; one operator at two timescales, no trained projections | an innate in-context learner at birth (C1) |
| **drop H9** | the genome is never fit to weights; it *is* the seed and the dials; at-birth competence is the memory's, and the first lifetime pass writes the rest | the two-part code of ANLAGE A2 satisfied by construction |
| **drop H10** | the paradigm is judged on its own controls; the backprop model is the *ceiling to report*, not the competitor at matched compute | honesty about C6 and a measurable scale path |

### T4 — candidates in the transformed space

* **A — the HDC n-gram sketch.** Superposition counting only; no learned readout. The null of
  the paradigm (control arm `sketch`).
* **B — the reservoir LM.** Seeded random recurrent dynamics over atoms, one ridge readout.
  Tests whether implicit recurrence beats explicit binding (arm `reservoir`).
* **C — SPORE.** Seeded atoms; multi-scale permutation-bound context codes; a delta-rule
  in-context memory over the same codes; a developmental cortex of nonlinear units grown by
  residual correlation; a closed-form ridge readout refined by the delta rule; a genome of
  seeds and dials searched by selection over lifetimes; random-indexing atoms as a registered
  variant.

### T5 — evaluation

| candidate | feasibility | utility | novelty | creativity |
|---|---|---|---|---|
| A | pass | 0.2 (a hashed n-gram model) | 0.3 | 0.5 |
| B | pass | 0.4 (echo-state language models are weak at the word level) | 0.3 | 0.7 |
| **C** | pass (every part is a matmul or a seeded draw; a 25M-token life is minutes) | **0.7** | **0.8** (no published next-token model composes seeded HDC codes, a delta-rule memory with fixed projections, cascade growth by residual correlation and evolved dials with zero backpropagation) | **1.5** |

**Core creative insight.** *Make the lifetime convex: then it needs no backpropagation and
costs minutes; the structure it cannot touch becomes the genome's, so selection is both
affordable and unbypassable; and development is the growth of units fit one at a time to
what the organism cannot yet predict.*

## 0.5 The constitution and the axioms of SPORE

**[NORMATIVE]** Each axiom is constitutional; a change to one is a change of design.

| # | axiom | replaces |
|---|---|---|
| **S1** | **The lifetime is convex.** Given the structure, every learnable object is the minimiser of a convex objective: the readout (ridge, then multinomial logistic by its own gradient), the in-context memory (online least squares), the unit weights of the cortex (fixed at birth; their *selection* is by correlation with a residual). No error signal crosses a nonlinearity. | X1, RC1, H1 |
| **S2** | **The genome is a seed and dials, never a fit.** Atoms, permutations and candidate directions are regenerated from seeds; the dials are a few dozen numbers. The genome is the only object that can change structure. | X2, H9, A2 |
| **S3** | **Heritable is what does not copy** (ANLAGE A8, kept): what the lifetime writes (readout, memory) is *not* inherited; what regenerates from the seed and what the dials say is. A lineage's next lifetime starts from the genome and reads the corpus again. | — |
| **S4** | **Development is growth by residual.** A unit is born when a growth wave fires (a genomic schedule), chosen from seeded candidates by its correlation with what the current readout cannot predict, and is frozen thereafter. | X6, X7, A5 |
| **S5** | **Selection over cheap lifetimes.** Fitness is the fitness-set cross-entropy after a fixed token budget; the test set is never seen by selection; the elite's test CE is the only reported number. | X4, A4 |
| **S6** | **Measure before machinery** (ANLAGE A7, kept): `σ_e` over atom seeds first; a mechanism is kept only if its ablation costs ≥ 3 `σ_e` at matched features and tokens. | — |
| **S7** | **The backprop model is the ceiling to report.** K-SOTA records the gap at matched tokens and its slope; the paradigm's claim is its own controls (K-HDC, K-DEV, K-EVO), not parity. | H10 |

## 0.6 What is reused, and why

| component | path | use |
|---|---|---|
| the gated delta rule, chunkwise, verified to 1e-10 against a float64 oracle | `gnomon/pheno/mixer_chunk.py::gated_delta_rule_chunk` | the in-context memory (`01 §1.5`) with fixed, seeded projections — the one learned object inside the sequence, and it learns by least squares |
| corpus shards, the v3 evaluation sets, strata | `gnomon/data/stream.py`, `anlage/data.py`, `scripts/anlage_common.py` | the same 256 test windows every model here is scored on |
| paired contrasts, `σ_e`, BCa | `screen/variance.py`, `gnomon/utils/stats.py` | the protocol |
| the run-manifest and JSON conventions | `scripts/anlage_common.py::manifest`, `gnomon/utils/report.py` | every run |

Not reused: anything that trains by gradient (the phenotypes of all four models), the
weight decoders, populations of siblings, the RL consolidator. They stay on their branches
as the record. The bottleneck replication's queue is paused and untouched.
