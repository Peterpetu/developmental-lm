# GNOMON — a language model whose every weight is decoded from a genome

> **Context (October 2026).** This is GNOMON's own README as written in September 2026. Read its compression table beside the quality numbers it does not print: the evolved elite reaches 5.235 nats at 2,000 steps where a free-weight model of the same architecture reaches 3.020 and a 75k-parameter free model reaches 4.295 (`runs/gnomon/gen-elite-2000.json`, `base-big-2000.json`, `base-free-2000.json`). The flat-archive and no-architecture ablation arms were never run.

**Branch status (v1.1):** implementation complete against `docs/gnomon/00..10`;
275 tests, 26 planted defects, three gates green. §09.1's experimental schedule
is *not* complete — see [what is still to run](#what-is-still-to-run).

This branch supersedes SOMA, which is on `main` and `genomic-bottleneck`.
`docs/gnomon/00-postmortem.md` is why: SOMA trained, and was **inert**. Its
evolutionary machinery contributed 0.04% of the model's learning, its unit
population 2.4%, and a single dense bypass carried 6.98 of 6.76 nats. That
measurement was available in hour six and was taken in week three.

## The idea in one paragraph

Every weight matrix of the phenotype is an output of one decoder shared across
all layers and all matrix roles; that decoder's parameters are the **genome**
(5,306 reals), and the genome is searched by **MAP-Elites over an open-ended
archive**. Identity is split in two — an **innate** code derived from an index,
costing nothing, and an **acquired** vector written only by lifetime learning —
so the genome learns what a *relationship between two identities* means and the
lifetime learns what each *thing is*. The layer count, per-layer operator, head
count, FFN width and attention window are decoded too, so mutating the genome
changes the computation graph and architecture search is not a second loop.

The test each mechanism had to pass: *remove it, and the model does not degrade
— it ceases to exist.*

| | at M1 (`512 x 12`, V = 8192) |
|---|---|
| genome `\|Gamma\|` | 5,306 reals — 20.7 KB |
| soma `\|Z\|` | 139,360 reals — 544 KB |
| decoded weights | 49,926,705 |
| compression | **9,458x**, and it *improves* with scale |

## Specification

Read `01` first; it is normative for the rest.

| File | Contents |
|---|---|
| [`00-postmortem.md`](docs/gnomon/00-postmortem.md) | why SOMA failed, measured; what that constrains |
| [`01-constitution.md`](docs/gnomon/01-constitution.md) | the three parameter classes; C1–C5; the symbol table |
| [`02-codes.md`](docs/gnomon/02-codes.md) | innate codes: the seeded bijection, Gray-over-depth, Gray-over-frequency-rank |
| [`03-decoder.md`](docs/gnomon/03-decoder.md) | the trunk, output normalisation, the genome enumerated, cost |
| [`04-phenotype.md`](docs/gnomon/04-phenotype.md) | operators as pure functions; the 16 matrix roles; the absence gate |
| [`05-lifetime.md`](docs/gnomon/05-lifetime.md) | the loop, the objective, fitness, the optimizer, developmental annealing |
| [`06-share.md`](docs/gnomon/06-share.md) | does the genome carry the function? heritability, the probes, the `d_z` frontier |
| [`07-architecture.md`](docs/gnomon/07-architecture.md) | the architecture head, totality, behaviour descriptors, variation |
| [`08-archive.md`](docs/gnomon/08-archive.md) | MAP-Elites over genomes; the ablation arms |
| [`09-protocol.md`](docs/gnomon/09-protocol.md) | the runs, the kill criteria, what may be claimed |
| [`10-plasticity.md`](docs/gnomon/10-plasticity.md) | the decoded learning rule, and the measurement that says it does not pay |
| [`AUDIT-v1.1.md`](docs/gnomon/AUDIT-v1.1.md) | the audit register: every finding and what was done about it |

## Running it

Everything needs the tokenised corpora, which are 6.4 GB and gitignored:

```
python scripts/build_tokenizer.py && python scripts/prepare_data.py
```

Then, cheapest first:

```
python -m pytest tests_gnomon -q            # the suite
python scripts/gnomon_spec_trace.py         # G-SPECMAP: every normative section owned
python scripts/gnomon_acceptance.py         # G-ACCEPT:  every acceptance row has a test
python scripts/gnomon_smoke.py --steps 200  # one lifetime, reported honestly

python scripts/gnomon_herit.py              # KG-HERIT, the gate that runs before the machinery
python scripts/gnomon_lrsweep.py --steps 150 --batch 128
python scripts/gnomon_evolve.py --gens 40 --pop 24 --seed 0
python scripts/gnomon_analyse.py runs/gnomon/evo-GNOMON-s*/history.json \
       --against runs/gnomon/evo-A-RAND-s*/history.json
python scripts/gnomon_frontier.py           # the d_z frontier; sets the operating point
python scripts/gnomon_m1.py --archive runs/gnomon/evo-GNOMON-s0
```

`--compile-mode cudagraph` captures the training step and replays it: 1.60x
measured, at the cost of bit-reproducibility across backends (see
`gnomon/lifetime/graph.py`). The default is eager.

## The gates

| gate | what it checks |
|---|---|
| `G-SPECMAP` | every section with a [NORMATIVE] statement is owned by code or a test |
| `G-ACCEPT` | every acceptance-table row has a test named for it |
| `G-SPECNUM` | every configured constant, recomputed from the module tree as an equality |
| `G-MUTATE` | 26 planted defects, each with the test that must reject it |
| `G-NOFREE` | the model holds exactly `Gamma ∪ Z`, and the optimizers step exactly `Z` |
| `G-INFER` | the inference cache equals decoding per step, and is faster |
| `KG-HERIT` | does the genome carry the function? Fires below `h2 = 0.25` |
| `KG-FLAT`, `KG-QD`, `KG-SCALE`, `KG-RANK`, `KG-SHARE` | the kill criteria of §09.2 |

## What is still to run

Implementation is complete; the schedule is not. `scripts/gnomon_analyse.py`
reports a criterion as **NOT EVALUABLE** rather than as passing when the run
that decides it does not exist.

| run | status |
|---|---|
| `G0-HERIT`, `HORIZON` | done — `h2 = 0.389` [0.172, 0.620], `KG-HERIT` **passes** |
| `EVO` | 1 of 3 run-seeds, and that one is superseded: it used §2.4's token code and an `lr` §10.6 records as defective |
| the ablation arms | not run |
| `FRONTIER`, `TRANSFER`, `M1` | not run; drivers exist |
| `BASE-FREE`, `BASE-BIG` | to be re-baselined at a measured `lr` |
| the M-nano `lr` sweep | running; `M_NANO.lr` is marked `NOT MEASURED` until it lands, and `FRONTIER` runs at M-nano |

## Project standards

The project's research-integrity rules govern. In short: no stubs, no partial specifications, no
unverified claims, every hyperparameter justified, every algorithm testable in
isolation, and **negative results reported with equal prominence** — of which
§10.5b, where the decoded learning rule was measured and did not pay, is the
worked example.
