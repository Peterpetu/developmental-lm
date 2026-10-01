# SPORE — a language model with no backpropagation, grown from a seed

**Branch `tempo`, built and measured in one night (2026-09-09 22:00 → 2026-09-10).** The fifth
model in this repository. Every number below is re-measured from `runs/spore/` by
`scripts/spore_analyse.py`; every finding, correction and "why" is in
[`03-evidence-ledger.md`](03-evidence-ledger.md) in the order it was obtained.

## The idea in one paragraph

Four earlier models put an evolutionary or genomic mechanism *beside* a network that
backpropagation could train, and backpropagation bypassed it every time
([`00 §0.2`](00-record-and-constitution.md)). SPORE removes backpropagation from the lifetime.
The organism's structure is a **seed and twenty dials** (the genome): from the seed it
regenerates hyperdimensional atoms for every token, a permutation that binds order, the
projections of an in-context memory and the candidates of its growth. Its lifetime is
**convex**: context codes are built by binding and bundling (exact n-gram codes, a fading
bag, a fading trajectory), a delta-rule memory learns inside every text with no trained
projection, units are born from a ridge solve over what the organism has already learned, and
a log-linear readout is fit by ridge regression and then by the delta rule — a single-layer
local rule with a closed-form gradient. Because the lifetime cannot change the structure, the
genome is the only thing that can, and selection over lifetimes is both affordable (a life is
minutes) and unbypassable. Semantic geometry is obtained without gradients: the organism's
first developmental stage counts bigrams, factorises their PPMI matrix, and mixes the factors
into its atoms.

## What was measured (TinyStories, vocabulary 8192, the v3 test windows every model here uses)

| model at the same tokens | 5M tokens | 25M tokens |
|---|---|---|
| unigram / bigram / trigram counts | 5.94 / 3.84 / 3.38 | 5.94 / 3.75 / 3.17 |
| **SPORE** (best measured recipe) | **2.63** | **2.41** |
| the 13M gradient-trained transformer (`s0:d256/L8`) | ≈ 2.54 (interpolated) | 1.857 |

* **Innate behaviour** (R-BIRTH): before reading a token, the in-context memory alone scores
  7.4–7.6 nats against a null of 9.01 (it predicts repeated names and phrases).
* **Every HDC mechanism is load-bearing** (K-HDC does not fire): removing order binding costs
  0.037, the memory 0.122, half the code width 0.093 — each 17–55 times the seed noise
  (`σ_e` = 0.0007 over atom seeds).
* **Development**: units chosen by residual correlation are no better than random units
  (K-DEV fires in that form, +0.019 either way); units *born by a ridge solve* of random token
  subsets beat random units by 0.021 and none by 0.046 (K-DEV does not fire in that form).
* **The readout's optimisation is the largest lever**: one delta-rule pass takes the ridge
  solution from 3.81 to 2.97; three passes at a larger step to 2.63.
* **Semantic atoms** (PPMI-SVD of the organism's own counts, no gradient): −0.155 at 5M and
  half a nat at birth.
* **K-SOTA fires**: from 5M to 25M tokens SPORE improves by 0.11 nats per doubling, the
  transformer by 0.29. The ceiling at 25M is representational (fixed codes, a fixed memory,
  1,536 units), not budgetary: the lever for more data is richer development.
* **Null results, recorded**: the exact one-hot order-1 block (+0.04 at eight times the cost —
  paging), output-side semantic sharing (0), `D = 4096` (−0.03 at 2.4× the cost), pure
  distributional atoms (worse than the mixture), `d_sem = 512`.
* **Selection across lifetimes** (K-EVO fires): 32 lives in four generations at a 1M-token
  screen; the elite beats the base by 0.15 at 5M, but the base with the elite's one monotone
  gene (a second refinement pass) beats the elite by 0.013 — what the screen selected (a
  smaller memory, fewer units, the reservoir at a low radius) does not carry to 5M. The
  loop did surface the reservoir block, which on its own is worth −0.036 (P-S.31/32).

## The constitution, read against the measurements

| requirement (`00 §0.1`) | verdict |
|---|---|
| C1 HDC at the core, not decorative | **holds**: binding +0.037, memory +0.122, width +0.093 when removed (K-HDC does not fire) |
| C2 backprop-free | **holds by construction and by test** (G-BPF); every rule is closed-form or single-layer local |
| C3 two timescales, both load-bearing | lifetime: yes; genome by selection: **not shown** — K-EVO fires (the screen's gains do not carry) |
| C4 development | **partly**: units born by a ridge solve pay 0.046 over none and 0.021 over random; units selected by residual correlation do not (K-DEV fires in that form) |
| C5 novel mechanisms | the composition has no published instance as a next-token model; each part is cited (`00 §0.4`) |
| C6 SOTA on TinyStories | **not reached**: 2.41 vs 1.857 at 25M; 0.10 behind at 5M; K-SOTA fires on the slope (representational ceiling); the Pile lineage was not run |

## Files

| file | contents |
|---|---|
| [`00-record-and-constitution.md`](00-record-and-constitution.md) | the record of four models; the reading of the brief; the first-principles argument; the Tier-3 creative-reasoning pass; axioms S1–S7 |
| [`01-model.md`](01-model.md) | the full mathematics: atoms, codes, the memory, the readout and its heads, development (11, 11b), the genome, selection, accounting, the scale path |
| [`02-protocol.md`](02-protocol.md) | pre-registered readings and kill criteria; reporting rules; corrections |
| [`03-evidence-ledger.md`](03-evidence-ledger.md) | P-S.1–: every finding and correction with its artifact |
| `spore/` | `genome.py`, `hdc.py`, `memory.py`, `readout.py`, `cortex.py`, `semantic.py`, `life.py`, `baselines.py`, `evolve.py` |
| `tests_spore/` | 23 tests: identities, planted-signal recoveries, the autograd cross-check of the delta rule, G-BPF |
| `scripts/spore_*.py` | `life` (one lifetime), `baselines` (the count models), `evolve` (selection), `analyse` (every verdict) |

## Running it

```
python -m pytest tests_spore -q
python scripts/spore_baselines.py --name counts-5M --budget 5000000
python scripts/spore_life.py --name sem05-E3-5M --budget 5000000 --D 2048 --eta-sem 0.5 --E 3 --refine-lr 0.01 --growth ridge --d-k 2048
python scripts/spore_evolve.py --name evo-1M --budget 1000000 --D 2048 --eta-sem 0.5 --k-wave 256 --pop 8 --elites 2 --generations 12 --wall-hours 1.5
python scripts/spore_analyse.py
```

A 5M-token life is 5–10 minutes on the RTX 4080 SUPER; the 25M report run is 55 minutes,
of which forty are the three delta-rule passes. There is no backward pass anywhere: the test
`test_no_organism_tensor_requires_grad_and_backward_is_never_called` is the mechanical
statement of that.
