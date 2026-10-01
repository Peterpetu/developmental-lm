# 00 — Post-mortem: why SOMA failed, and what it constrains

> **Context (October 2026).** The randomise-in-place table in §0.1 is the output of a probe whose results were not archived; the archived SOMA reports give final CE 2.1223 (full) vs 2.1250 (no selection/birth/death) vs 2.0205 for the plain M0 baseline (`archive/soma/runs/`). The conclusion — the evolutionary machinery was inert — stands on the archived numbers.

**Status: record, not specification.** This document is the evidence that motivates
§01's constitution. Negative results are results (project rule: research integrity); the
project keeps them and builds on them.

SOMA v3.1 is on branches `main` and `genomic-bottleneck`: 25,347 lines, 474 passing
tests, 91 audited defects, 43/51 gates met, spec traceability green. It trained. It was
not wrong in the sense of being buggy. It was wrong in the sense of being **inert**.

## 0.1 The measurements

Trained M-nano checkpoint (`runs/m-nano-s20260828`, `d_model=192`, `L=6`, 2.94M
parameters, 2,000 steps, TinyStories, vocab 4096). Held-out cross-entropy in nats per
token, `k_tok = 1`, natural text only. Each row randomises one part of the trained model
**in place** and re-evaluates; no re-training.

| removed / randomised | reals touched | held-out CE | Δ from trained |
|---|---|---|---|
| — (the trained model) | — | **2.1005** | — |
| ablation A3: all selection, birth, death, growth | (mechanism) | 2.1250 | **+0.0027** |
| the gene library `parts.A`, `parts.B` | 786,432 | 2.2621 | +0.1617 |
| the unit populations `p_u, g_u, m_u, …` | 38,400 | 2.2621 | +0.1617 |
| everything a unit is (parts + pops + ctx/pos/router) | 976,384 | 2.2621 | +0.1617 |
| all units switched off (`m_u ← 0`) | (mechanism) | 2.2621 | +0.1617 |
| `W_shared`, the Site's dense bypass | 221,184 | 9.0781 | **+6.9776** |
| the mixers + attention | — | 7.2226 | +5.1221 |
| the embedding | — | 8.7432 | +6.6427 |

An untrained model scores 8.86, so total learning is **6.76 nats**. The unit population
contributed **0.162 (2.4%)**; the evolutionary machinery operating on that population
contributed **0.003 (0.04%)**.

**A discrepancy in the A3 row, recorded rather than resolved.** Every other row
satisfies `delta = CE - 2.1005` exactly — the gene library's `2.2621 - 2.1005 =
0.1616` against a stated `0.1617`. The ablation-A3 row states `CE 2.1250` and
`delta +0.0027`, and `2.1250 - 2.1005 = 0.0245`. The two cannot both be right.
The prose above, §0.1's third corroborating signal, and the 0.04% figure all
reconcile with the **delta**, so `+0.0027` is the quantity every later section
and `S4` use; the CE entry is left as it was written because this document is a
record of an archived run on another branch and re-deriving it would mean
re-running SOMA. Which of the two is the typo is not established here, and the
conclusion does not turn on it: at either value the evolutionary machinery
contributed under 0.4% of the total learning.

Three further measurements, each independent:

1. **Magnitude vs. information.** Inside Site 0 the routed term
   `sum_u r_ut m_u delta_ut` has RMS **8.49** against `W_shared x` at **1.44** — a ratio
   of **5.89**. The population was not small. It was loud and uninformative; the network
   learned to route around it.
2. **Transplant.** A fully trained, *uncompressed* parts library transplanted into an
   otherwise-random model scored **8.87** against **8.86** for random parts. The trained
   library carried **nothing** transferable.
3. **Corroborating signals.** Credit-assignment correlation `rho ≈ 0.03`; bifurcation
   alarm rate 0.0%; ablation A3 (`3.75x` fewer units, no selection) within 0.13% of full
   SOMA.

## 0.2 Root causes

**RC1 — Evolution was a parallel path.** SOMA was a transformer (attention + gated
delta-rule mixer + FFN) with a routed population decorating the FFN. Backpropagation
could reach the loss through `W_shared` alone, so it did. **Anything gradient descent can
bypass, it bypasses.** This is the governing lesson.

**RC2 — The genome could not write weights.** The germline stored scaffolding — gene
indices, positions on a sphere, policy constants: 20,499 reals — while every tensor that
touched data was soma and was re-randomised each lifetime. `BirthScore ≈ 0` was therefore
true *by construction*, and generations could only ever tune hyperparameters. §02.2's
"experience-expectant" framing rationalised this; it was a rationalisation.

**RC3 — The two timescales shared no substrate.** Lifetime learning wrote `theta`;
germline evolution wrote `Gamma`. They met only through a scalar fitness, once per
lifetime — about 4 bits per 46-GPU-hour generation against a 22,262-real genome.

**RC4 — Selection had nothing to select on.** ε-lexicase over units-as-individuals with
tokens-as-cases requires fitness variance that is *heritable* and *consequential*.
Units were interchangeable rank-1 decorations; all three properties failed.

**RC5 — Architecture search was never implemented.** Nothing in SOMA searched over
structure. MSG made every structural event function-preserving, so each birth was a no-op
at the instant it occurred — a gradient-descent convenience, not an evolutionary event.

## 0.3 A sixth cause, in the process rather than the design

The 2.4% figure was measurable on the first smoke run. It was measured in week three,
after 25k lines, 91 audit entries and two adversarial verification passes — every one of
which was measuring *conformance to a specification that specified an inert model*. A
conformance suite cannot detect that the thing being conformed to does not matter.

**This is why §06's genome-share probe is a gate on the smoke run and not a final
metric.** The instrument that would have caught SOMA in hour six is promoted to a
first-class module with a kill criterion attached.

## 0.4 What survives

Verified and reused: the corrected chunkwise gated delta rule (`mixer_chunk.py`, matched
to a closed-form oracle at 1e-10 in f64, including the (5.15) derivation error that would
otherwise have shipped); attention (GQA/SWA/sink/NoPE); RMSNorm; Muon; the tokenizers and
tokenized TinyStories at vocab 4096 and 8192; the statistics module (exact Mann–Kendall,
BH, TOST, BCa bootstrap, effective rank); the behavioural regression suite; the gate and
spec-traceability oracles; the randomise-and-measure probe that produced §0.1.

Also reused: `soma/germline/gnet.py`, the genomic-bottleneck decoder — read from and
corrected against Shuvaev, Lachi, Koulakov & Zador (PNAS 2024) rather than a summary of
it. It is the seed of §03's decoder. Two errors that correction found are recorded there
and are binding on §03: **the labels must not be stored** (storing them cost 94% of the
genome and made the model a hypernetwork by the paper's own distinction), and **the
decoder needs two hidden layers, not one** (one layer is additive at initialisation scale
and yields a rank-2 library: measured effective rank 6.76 of 192, two genes at cosine
0.999).
