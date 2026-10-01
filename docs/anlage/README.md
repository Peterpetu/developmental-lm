# ANLAGE — a language model grown from a heritable programme

*Anlage* (embryology): the primordium from which an organ develops.

ANLAGE is the third language-model design in this repository, after SOMA and GNOMON
(`00-postmortem-and-axioms.md`). It was specified in full before any code (`01`–`10`),
implemented to its acceptance tests, and then **measured** — and the measurements changed
it. This README is the map; every number in it is re-measured from an artifact under
`runs/anlage/` and recorded with its provenance in `RESULTS.md`.

## The short version

1. **The weight genome was falsified, cleanly.** ANLAGE was designed around a *neural-kernel
   genome*: a compact network that decodes every weight of every stage from innate unit codes
   and small acquired latents, written from a trained adult by regression (the PNAS 2024
   "genomic bottleneck" loop, made scale-free). Implemented and verified (D1–D14, 67 tests),
   it was then regressed onto a trained 13M-parameter stage-0 model at 18x compression:
   vector roles fit to `< 1%` residual, **every weight matrix to 7–20% of its variance**, and
   the decoded model scored 8.53 nats against a null of 9.01 and a trained 1.857. Training the
   decode end to end collapsed to the uniform predictor (rank 1); the PNAS-style assimilated
   lifetime cost 0.145 nats and still decoded to 13.07. `11-weight-genome-measured.md` gives
   the information count that predicts these numbers and the argument that **no genome that
   compresses adult weights can be load-bearing in a loop where the adult can be copied**.
2. **What a genome can carry is what does not copy (axiom A8).** `12-program-genome.md`
   specifies ANLAGE-P: the heritable object is the *developmental and learning programme* —
   the data-class mixture per stage `mu`, the growth schedule `omega`, the recipe `eta`, the
   scale profile `s` (the one part of the weight genome that was measured to work) and the
   unit prior `Lambda` for growth. It is `10^2`–`10^4` reals against `10^7`–`10^9` weights,
   scale-free by construction, and its expression is trivially parallel. The weights are the
   second part of the two-part code and are inherited by copying and function-preserving
   growth. Every mechanism has published evidence in language-model pretraining
   (`10-evidence-ledger.md` §10.9, re-verified against the primary sources on 2026-09-02).
3. **Two timescales, both load-bearing or removed.** The lifetime is gradient descent on a
   free-weight 3:1 gated-delta/attention hybrid with Muon (`04-phenotype.md`). The generation
   is a population of cheap primordium-stage siblings on a Dirichlet neighbourhood of the
   programme's mixture, ε-lexicase selection over label-free output strata, a weight-space
   merge of the parents, a regression of the programme from the population's outcomes
   (RegMix's map, with a trust-region step), growth to the adult, and the adult's block as the
   next warm start. Each step has a kill criterion with a conventional control at matched
   compute (`12-program-genome.md` §12.8) and is reported either way.

## State of the evidence (TinyStories, `V = 8192`, stage 0 = 13.0M parameters, 25M tokens)

| run | what it measured | result |
|---|---|---|
| R1 NOISE | seed noise under common random numbers, 4 seeds | `sigma_e = 0.0060` nats on the v1 sets (mean 1.8503); 0.0029 on the v3 sets (mean 1.8092) |
| R4 BIRTH | the weight genome regressed onto R1's seed 0 | at-birth 8.53 (share 0.067) — **K-BIRTH fires** |
| R4-probe | the decode trained end to end | 9.011 = uniform; decoded rank 1 |
| R4b | assimilation (refit + pull) during the lifetime | lifetime 2.0021 vs 1.8571; decode 13.07 — **K-ANCHOR fires** |
| R2 CLASSES | the data-class partition | classes differ by 0.25 nats (40 `sigma_e`) in the free model |
| R4 INIT | the scale-profile birth vs the free init, 4 seeds | worse at birth, −0.014 [−0.019, −0.010] after 2.5M tokens: real, below the 3-`sigma_e` bar |
| R3 LIVE | does the mixture axis move the outcome? | marginal on TinyStories: stratum spread up to 5.1 `sigma_e(q)` (v3), test spread 2–2.6 protocol σ, best sibling −0.004 to −0.008; the learning-rate axis moves 0.09 nats |
| R6 MIX | `mu*` vs the corpus mixture, 4 seeds × 25M tokens | +0.002 [−0.002, +0.005]: no difference — **K-MIX fires** on TinyStories (the homogeneous-corpus outcome §12.9 expected) |
| R7 POP | lexicase + merge vs best sibling vs SINGLE at equal compute | merge beats every sibling by 0.09, but SINGLE (one 50M-token lifetime) beats the merged child by 0.115 — **K-POP fires** |
| R5 GROW | unit-prior growth vs copy / random / flat, 3 seeds | prior − copy = +0.020 [+0.011, +0.030] — **K-GROW fires**; copy − flat = +0.003 [−0.007, +0.014]: growth is only equivalent to from-scratch at this budget |
| R8 LOOP | generations of ANLAGE-P (K=4, copy growth, 50M adult-token budget) vs SINGLE-WARM | gen 0: adults 1.598 vs 1.587 (+0.011); gen 1: 1.576 vs 1.569 (+0.007) — **K-LOOP fires** both generations, one lineage each; both lineages improve across generations by inheriting the adult's block |

`python scripts/anlage_analyse.py` prints every criterion's verdict from the artifacts on
disk (PASS / FIRES / INCONCLUSIVE / NOT EVALUABLE — a missing run is never a pass). The
protocol on TinyStories is complete (2026-09-02, 18:27): every mechanism that could be
measured there was measured, and the best model in the repository is the conventionally
trained, copy-grown 78M hybrid of the SINGLE-WARM lineage at generation 1, **1.569 nats**
test CE on the v3 sets (`runs/anlage/gen1-tinystories-single-s0/adult.pt`).

## Layout

| path | contents |
|---|---|
| `docs/anlage/00`–`10` | the original specification (kept in full; status notes mark what §11 superseded) |
| `docs/anlage/11` | the weight genome, measured; the information count; axiom A8 |
| `docs/anlage/12` | ANLAGE-P, the programme genome: objects, mathematics, kill criteria, protocol, acceptance tests |
| `docs/anlage/RESULTS.md` | the results register — every number with its artifact |
| `anlage/` | `config`, `codes`, `latents`, `genome` (kept; vector roles live), `model`, `ops`, `recipe`, `data` (classified pool), `lifetime`, `regress`, `grow` (prior growth, shrink), `population`, `program`, `sleep` |
| `tests_anlage/` | acceptance tests I, D, P, C, G, S, R, M and the G-NOFREE-INVERSE sentinel (`python -m pytest tests_anlage -q`) |
| `scripts/anlage_*.py` | one script per run of the protocol; `anlage_analyse.py` the verdicts; `anlage_spec_trace.py` the spec ↔ code binding |

## Reproducing

```
python -m pytest tests_anlage -q                       # ~2 min, CPU
python scripts/anlage_spec_trace.py                    # every normative section owned
python scripts/anlage_noise.py --tokens 25000000 --seeds 4      # R1, ~30 min
python scripts/anlage_classes.py                       # R2, ~1 min
python scripts/anlage_live.py                          # R3, ~45 min
python scripts/anlage_mix.py --seeds 0 1 2 3           # R6, ~55 min
python scripts/anlage_pop2.py                          # R7, ~15 min
python scripts/anlage_grow.py --arms prior copy random flat --seeds 0 1 2   # R5, ~1.5 h
python scripts/anlage_generation.py --gen 0 --K 4      # R8 (then --gen 1 with --program/--prev-adult)
python scripts/anlage_analyse.py
```

Hardware: one RTX 4080 Super (16 GB), Windows 11, Python 3.11, torch 2.7.1+cu118; the
manifests in every JSON record the commit, the pool digest and the evaluation-set digests.

## What is not claimed

Parity with published models trained on 100–1000x more tokens; anything about the Pile
lineage before it runs (R9); that the mixture axis is live on a homogeneous corpus before
R3 says so. The weight genome's negative result is claimed, with its artifacts.
