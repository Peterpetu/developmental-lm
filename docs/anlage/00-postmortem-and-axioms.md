# 00 — Post-mortem of two models, and the axioms that replace theirs

**Status (2026-09-02):** §11 adds the measurement of this design's own genome and the axiom A8 it forced; §12 is the design that follows. §0.5's axioms stand as revised in §11.5.

**Status: record and constitution.** §0.1–§0.3 are the evidence; §0.4 names the axioms both
prior designs shared; §0.5 states the axioms ANLAGE is built on. Negative results are results
(project rule: research integrity): nothing here is softened.

ANLAGE (*Anlage*, embryology: the primordium from which an organ develops) is the third
language-model design in this repository. SOMA (branches `main`, `genomic-bottleneck`) and
GNOMON (branch `gnomon`) were each implemented to a research-grade standard of conformance
and each was inert or worse at the level of the mechanism it existed to test. The failure in
both cases was in the rules the design obeyed without stating them, not in the code.

## 0.1 SOMA, measured

SOMA v3.1: a transformer (attention + gated delta-rule mixer + FFN) whose FFN carried a routed
population of mortal "units" under ε-lexicase selection, a germline of gene-field scaffolding
(20,499 reals) updated by ES, a metabolic price, and a verifier-backed synthetic task stream.
25,347 lines, 474 tests, 91 audited defects, 43/51 gates met. From `docs/gnomon/00-postmortem.md`:

| measurement (M-nano, 2.94M params, 2,000 steps, TinyStories, V = 4096) | value |
|---|---|
| trained held-out CE | 2.1005 nats |
| untrained CE | 8.86 nats (total learning 6.76 nats) |
| contribution of the whole unit population | 0.162 nats (2.4%) |
| contribution of the evolutionary machinery acting on it | 0.003 nats (0.04%) |
| contribution of the single dense bypass `W_shared` | 6.98 nats |
| trained parts library transplanted into a random model | 8.87 vs 8.86 for random parts |
| credit-assignment correlation | 0.03 |
| the dense baseline M0 at the same token budget | 2.0205 nats |

Root causes as recorded there: **RC1** evolution was a parallel path and gradient descent
bypassed it; **RC2** the genome could not write weights, so birth performance was zero *by
construction*; **RC3** the two timescales shared no substrate; **RC4** selection had nothing
heritable and consequential to select on; **RC5** function-preserving structural events were
no-ops at the instant they occurred; **RC6** (process) a conformance suite cannot detect that
the thing being conformed to does not matter — the 2.4% was measurable in hour six and was
measured in week three.

## 0.2 GNOMON, measured

GNOMON v1.1: every weight decoded by one per-entry MLP (4,483 reals) from innate binary
identity codes and acquired identity vectors (`Z`, 139,360 reals at M1); the architecture
decoded by a head of 796 reals; the 5,306-real genome searched by MAP-Elites over an archive;
a decoded plasticity rule. 275 tests, 26 planted defects, three gates green.

| measurement | value | source |
|---|---|---|
| heritability at generation 0, M-pico | `h2 = 0.389 [0.172, 0.620]`, gate PASS | `docs/gnomon/09 §9.5` |
| MAP-Elites, 20 generations, 432 evaluations | best 6.72 → 6.46 nats at generation 1, then **no improvement for 19 generations** | `runs/gnomon/evo-GNOMON-s0/history.json` |
| M1 lifetime, 512×12, 49.9M decoded weights, 40,000 steps, 9.65 GPU-h | best **5.22** nats at step 6,000; final **5.99**, above the unigram entropy 5.96; samples are word salad | `runs/gnomon/m1-long/m1.json` |
| a free-weight transformer of 2.94M parameters on the same corpus | **2.02** nats | `GATES.md` G-M0 |
| decoded plasticity rule vs AdamW | equivalent by TOST; the neuromodulator costs 0.17–0.21 nats at matched scale | `docs/gnomon/10 §10.5b` |
| the runaway identity norm | `z → ∞`, `zeta` grew 188,401×, decoder responsiveness 32% → 0.1%, decoded erank collapsed 38× | commit `f083282` |
| frequency-rank token code vs random bijection | **+0.0999 nats [+0.029, +0.265]** with plain AdamW — the largest effect found | `docs/gnomon/02 §2.9` |

The mechanism that starved GNOMON is named exactly: the per-token information of language
had to reach 49.9M weights through 139,360 identity coordinates behind a saturating map, and
Adam's fixed-size steps on saturated coordinates drove them to infinity.

## 0.3 SCREEN, measured

Branch `screen` (commits `02304d5`, `32688cc`) built the instrument neither design had:
signal against seed noise under common random numbers (CRN), and landscape topology.

| measurement (free-weight transformer d = 128, L = 4, T = 128, 2,000 steps, 4 seeds, CRN) | value |
|---|---|
| seed noise `sigma_e` | **0.0100 nats** (GNOMON had reasoned with 0.171 — a property of its decoder — 17× too high) |
| operator axis, attention → delta | −0.185 nats (18.5 `sigma_e`), **monotone** |
| cross-layer sharing vs none | +0.129 nats (12.9 `sigma_e`), monotone |
| sharing pattern, receptive field, width pattern | 1.9, 1.1, 0.7 `sigma_e`: below the floor |
| broad-sense `h2` of the body-plan landscape | 0.991, and a coordinate hill-climb finds the optimum in ≤ 4 flips |
| a priced fitness (parameter count, FLOPs) | `sigma = 0` against CE's `sigma_e > 0`: the search optimises the price |

**[NORMATIVE — binding on this design]** Heritability measures transmission fidelity, not
search difficulty. An axis that is monotone above noise is a hypothesis with a known answer,
not a search space. A mechanism is admitted only after its effect on held-out CE has been
measured against `sigma_e` under CRN (§08).

## 0.4 The axioms both designs shared, and which were wrong

| # | shared axiom | what the evidence says |
|---|---|---|
| **X1** | The evolutionary mechanism must be structurally separated from gradient descent (a Weismann barrier, distinct optimisers) and must contribute *against* gradient descent to the same objective. | Set up as competitors for the same credit on the same substrate, per-FLOP information decides it, and backprop wins every time: 0.04% (SOMA); one improvement in 432 evaluations (GNOMON). |
| **X2** | The genome must be tiny (10³–10⁴ reals) and must cap the model's capacity: the bottleneck *is* the smallness. | The PNAS bottleneck compresses an *already trained* network and is evaluated on at-birth performance and transfer (§10 ledger E-1). It never claims a 10⁴-real genome can be the sole substrate of competence. GNOMON's 5.2–6.0 nats against 2.0 is the measured cost of the misreading. |
| **X3** | Lifetime learning may write only a small non-weight object (unit positions; identities behind a saturating decoder). | The lifetime is where per-token information arrives; starving it starves the model, and identities behind `tanh` are unstable under Adam. |
| **X4** | Fitness is one scalar per lifetime, a lifetime is a whole training run, and a small population over a few dozen generations suffices. | The information rate of such a loop is bits per GPU-hour. Nothing in this repository or in the literature (§10 ledger E-7) shows it moving a language objective at this compute. |
| **X5** | Open-endedness lives in a synthetic task stream or an archive of architectures. | No member of the open-ended / quality-diversity family has ever improved a language-modelling objective (`docs/spec/06 §6.0`, re-confirmed by the 2026 survey in §10); SCREEN shows the architecture landscape needs no population. |
| **X6** | Structural events must be function-preserving at the instant they occur. | Then they are gradient-descent conveniences, not developmental events (RC5). |
| **X7** | Development is birth/death of units or a one-shot decode plus a short anneal; the model's size is fixed for its lifetime. | Neither changes the network's size over training. Staged growth is the one developmental mechanism with large, replicated compute wins (§10 ledger E-4, E-5). |

## 0.5 The axioms of ANLAGE

**[NORMATIVE]** Each is constitutional: a change to one is a change of design, recorded as
such, never a tuning decision.

| # | axiom | replaces |
|---|---|---|
| **A1** | **Sequential, not adversarial.** Gradient descent is the lifetime. The genome is written *from* the lifetime by compression (regression) and read *into* the next lifetime by expression (birth) and development (growth). The two are stages of one loop and never compete for credit: the genome is not on the gradient's path. | X1 |
| **A2** | **The bottleneck is priced, not imposed.** The genome is the first part of a two-part code (§05 Proposition 1); its size is a dial set by description length and measured benefit. The compression ratio is reported, never targeted. | X2 |
| **A3** | **Full-capacity lifetime.** The phenotype is a free-weight hybrid trained by the strongest known recipe (§04). Throughput on the training hot path equals the baseline's: no decode runs inside a training step. | X3 |
| **A4** | **Population where individuals are cheap.** Selection and crossover act on primordium-stage siblings born from one genome, on case-based fitness over held-out strata, with weight-space merging as crossover (§07). | X4, X5 |
| **A5** | **Development is growth sampled from the genome.** Structural events are real — the function changes — but safe — the change is ramped. New capacity is a draw from the genome's generative model of units (§06). The schedule is heritable. | X6, X7 |
| **A6** | **One consolidation primitive across timescales.** Context → state (the delta rule), data → weights (gradient descent), weights → genome (regression): each is a least-squares projection onto a lower-capacity parameterisation (§05 Definition 1). | — |
| **A7** | **Measure before machinery.** `sigma_e` under CRN is measured first at every configuration. A mechanism is retained only if it moves held-out CE by at least `3 sigma_e` against its best conventional replacement at matched compute; otherwise it is removed and the result reported (§08). | RC6 |

## 0.6 What survives from the two prior designs

**[NORMATIVE]** Reused, with provenance headers, and their oracle tests re-run unchanged:

| module | what | verified by |
|---|---|---|
| `gnomon/pheno/mixer_chunk.py`, `mixer_ref.py` | the gated delta rule, chunkwise (SOMA (5.6)–(5.16)) and its naive oracle | agreement to 1e-10 in float64 over the chunk × length grid, incl. the (5.16) `Gam_decay` correction |
| `gnomon/pheno/rmsnorm.py` | RMSNorm, gain-free head-wise norm, masked RMSNorm (3.4) | reduction at ≥ float32; (3.4) reduces to (3.3) at zero mean |
| `gnomon/pheno/ops.py::attention` | causal attention with GQA, gain-free QK-norm, sink column | used here only as the float64 **oracle** for the SDPA path |
| `gnomon/optim/muon.py` | Muon (Newton–Schulz 5), `wsd_lr` | spectrum contraction, transpose equivariance |
| `gnomon/genome/codes.py` | seeded bijection, Gray over relative depth, Gray over frequency rank, `frequency_rank` with stable ties | T1–T14 of `docs/gnomon/02` |
| `gnomon/data/stream.py`, `tokenizer.py`, `corpus.py` | shards, streams, token counts, tokenizer | the prepared `data/ts_8192` corpus |
| `gnomon/utils/stats.py` | BCa bootstrap, TOST, Mann–Kendall, effective rank | 31 tests |
| `gnomon/utils/shapes.py`, `rng.py`, `report.py` | shape assertions, per-purpose RNG, strict JSON | — |
| `screen/crn.py`, `screen/variance.py`, `screen/landscape.py` | CRN window pools, paired contrasts, `sigma_e`, landscape topology | 71 tests |
| `soma/selection/lexicase.py` | ε-lexicase with the Gumbel-race shuffle | 19 tests |
| `scripts/gate.py` | the gate oracle | G0 |

**Not reused:** GNOMON's per-entry decoder, its identity-space learning, its architecture
head and MAP-Elites archive; SOMA's unit population, metabolic price, task stream and
certification harness. Each is retired on the measurement in §0.1–§0.2, not on argument.

## 0.7 The process rule

**[NORMATIVE]** The instrument runs before the machinery: `scripts/anlage_noise.py` (§08 R1)
is the first script executed at every new configuration, and no number is reported from a
configuration whose `sigma_e` has not been measured. This is RC6 made procedural.
