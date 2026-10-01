# SOMA — Specification 00: Overview and Reading Guide

**SOMA** — *Selected Ontogeny of Modular Architecture*.
**Specification version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Status: normative.

---

## 0.1 What SOMA is

SOMA is a language-model architecture in which the parameters are partitioned into two
populations with **different write permissions**, and in which a population of **mortal,
positioned computational units** is subjected to **case-based selection** inside a single
training run.

The design rests on one primitive, the **expression operator** (§01.3), read at four rates
of change. Attention, learning, development and inheritance are not four mechanisms; they
are one operator whose arguments change at different speeds.

## 0.2 The two constitutional requirements

**C1 — Multi-timescale evolution.** The system evolves on at least two timescales, an
individual lifetime and a germline lifetime, structurally and **performance-beneficially**.
Thesis: intelligence is developmental; units function as they do only after the substrate
has developed into a certain form.

**C2 — Novel learning mechanisms.** Combinations never tried in LLM next-token prediction,
including substantial open-ended search, measurably improving performance.

Every section states which requirement it serves and §10 states what would falsify it.

## 0.3 Document map

Read `01` first; it is normative for all others.

| File | Contents | Normative for |
|---|---|---|
| `00-overview.md` | this file; notation; markers; map | — |
| `01-math-foundations.md` | expression operator; four rates; cost theorem; initialisation; metabolic functional; rate hierarchy; **complete symbol table** | all |
| `02-germline-soma.md` | Weismann barrier; germline contents; policy table; acceptance test; ES | `soma/germline/` |
| `03-development.md` | birth, mutation, crossover, death, function preservation, canalization, `GROW` | `soma/development/` |
| `04-selection.md` | ε-lexicase over units; case weights; credit; utility; guards | `soma/selection/` |
| `05-backbone.md` | mixer (recurrent + chunkwise), attention, routing, Site, optimizer, precision, accounting | `soma/model/` |
| `06-open-ended-stream.md` | two references; token selection; task archive; certification | `soma/openended/` |
| `07-training-protocol.md` | lifetime, fitness, configurations, milestones, logging, reproducibility | `soma/train.py`, `soma/generations.py` |
| `08-benchmarks-ablations.md` | metrics, `Q_route`, baselines, **TOST equivalence**, ablations A1a–A12, reporting rules | `soma/eval/` |
| `09-scaling-path.md` | per-component scaling evidence; the bottleneck argument; production gaps | claims only |
| `10-risks-kill-criteria.md` | kill criteria K1–K10; risk register R1–R12 | project conduct |
| `11-bibliography.md` | evidence ledger C-1…C-29 with per-claim status and failure modes | all claims |
| `AUDIT-v1.md` | the ~170 defects of spec v1 and the ~80 of spec v2, each with its resolution and spec delta | the repair itself |

Background that inspired the project (not normative): `docs/inspiration_text_2.txt`,
`docs/original_inspiration_conversation_summary.md`.

## 0.4 Notation conventions

- Scalars lowercase; matrices uppercase; sets in prose, uppercase in code.
- Time indices: `t` token position, `n` optimizer step, `E_sel` selection event,
  `G_gen` germline generation.
- **[NORMATIVE]** Every symbol used anywhere in this specification appears in the table of
  §01.10. A symbol used but absent from that table is a specification defect.
- **[NORMATIVE]** Every module boundary asserts the shapes of its inputs and outputs against
  the declared dimensions.

**One word, one meaning.** Spec v1 used "generation" for both a selection event and a
germline update, and gave both the symbol `G`, which made kill criterion K1 ambiguous
between two experiments differing by an order of magnitude in cost. `E_sel` is an **event**;
`G_gen` is a **generation**; the two are never conflated.

## 0.5 Status markers

| Marker | Meaning |
|---|---|
| **[NORMATIVE]** | must be implemented exactly as written, **and is mechanically checkable** |
| **[RATIONALE]** | motivation only; binds nothing |
| **[DERIVED]** | follows from a normative statement; included for clarity |
| **[EMPIRICAL]** | a value from cited evidence; revisable, and the citation must accompany any revision |
| **[OPEN]** | a decision deliberately deferred, with the deferral criterion stated |
| **[OURS]** | an extension present in no cited source; carries a mandatory test |

**[NORMATIVE — v2 change]** `[RATIONALE]` is new. Spec v1 marked prose with no implementable
content as `[NORMATIVE]` (for example "the justification must be stated correctly"), diluting
a marker its own §0.5 defined as "must be implemented exactly as written". `[NORMATIVE]` now
implies mechanical checkability, which is what makes the traceability gate `G-SPECMAP`
meaningful.

## 0.6 Relationship to prior work

Prior research is inspiration and a source of engineering substrate. It is not the design.
`11-bibliography.md` records, for every borrowed claim, the exact statement relied upon, its
verification status, and its known failure mode. `AUDIT-v1.md` additionally records two
design options **eliminated** on the strength of primary-source reading, and one novelty
claim **narrowed**; recording eliminations is as much the ledger's function as recording
adoptions.

The phenotype substrate (§05) is deliberately conventional and deliberately strong. Inventing
a new substrate *and* a new learning paradigm simultaneously would make any failure
unattributable. The substrate is the experimental control; the germline, selection,
development and open-ended machinery are the variables.

## 0.7 Specification history

| Version | Change |
|---|---|
| 1.0 | initial design, 12 files |
| **3.0** | **audit-v2 repair.** A second adversarial audit of the v2 specs returned ~80 defects, 37 of them blockers, several **introduced by the v1 repair**: a false per-unit-nonlinearity claim (§01.4, fixed by moving the activation after the gate, plus new ablation A12), a positional-collapse claim the sphere does not support (§01.6, fixed by the dispersion term (1.9) and a numeric `G-NOCOLLAPSE` threshold), `T_arch`/`T_cert` with no value anywhere ((1.18)–(1.19)), `q_cases` inversely proportional to `d_rate` (§01.10), a noise floor over identical seeds making every TOST margin zero (§07.7), `n_m d_v != d_model` at M-nano, an uninitialised task archive (`SEED_ARCHIVE`), an unbudgeted `theta_lag` forward, and a K5 margin below its statistic's granularity. Germline-scale constants are now identical at every configuration, which makes the M-nano → M1 inheritance shape-possible. |
| **2.0** | **audit-v1 repair.** Two adversarial auditors read all twelve files before any code was written and returned ~170 defects, including a wrong equation in the chunkwise mixer (5.16), an ε-lexicase epsilon that was identically zero, a population that went extinct within one selection event, positions that collapse to a point under the wiring term, a fitness statistic that is zero by construction, and an experimental protocol whose inner machinery fired zero to three times per lifetime. All are recorded with resolutions in `AUDIT-v1.md`. |
