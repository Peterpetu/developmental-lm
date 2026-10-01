# 02 — Protocol: pre-registered readings, kill criteria, reporting

**Status: normative; fixed before any life runs (2026-09-10).** Every criterion names its
arms, its metric, its bar and its consequence. `scripts/palimpsest_analyse.py` evaluates each
from the records on disk and prints NOT EVALUATED for any run that does not exist.

## 2.1 Measurement rules

* **Noise, per environment.** `σ_e(ε)` is the standard deviation over seeds of the `copy` arm's
  final `CE(W_K; ε)` on the life being read, under common random numbers; the union score's
  noise `σ_U` is the copy arm's spread in `U_K`. A criterion whose control has fewer than two
  complete seeds is NOT EVALUATED (G-NOISE). First-wave contrasts at one seed are **screens**
  and are labelled so; a claim needs `n ≥ 2` seeds with the paired `t` interval and `n ≥ 4`
  where it is final.
* **Contrasts** are paired by seed and order under CRN; reported as the paired mean
  difference with its interval beside the bar `3 σ`.
* **Compute.** Arms are compared at equal phenotype-training tokens in the first wave, with
  each arm's overhead recorded; the FLOP-matched copy (`copy-m`) is the strict control in the
  second wave (01 §1.8).
* **Every reported number** is on an environment's evaluation set (residue 2), never on the
  fitness set or the training pool; the union score weights environments equally.

## 2.2 The frontier and the double dissociation

**K-SIEVE (the claim).** On the `SHIFT` life, over the arms
`{copy, sieve-c2, sieve-c4, sieve-c8, sieve-c16, scratch}` (ratios `1, 2.4, 4.0, 6.1, 9.8, ∞`):

```
   load-bearing   iff   argmin_r U_K(r) is interior (a sieve arm)   and   U_K(copy) − U_K(r*) ≥ 3 σ_U   and   the interval excludes 0
   fires          iff   argmin_r U_K(r) = copy,   or the margin is inside the bar
```

**K-STAT (the theorem's half).** On the `STAT` life, the same arms: the reading is **monotone**
iff `U_K(copy) ≤ U_K(r)` for every sieve arm within `σ_U`; a sieve arm beating the copy under
no shift by `≥ 3 σ_U` **contradicts Proposition 1** and is reported as such, with the
regularising mechanism named (01 §0.3.2) — not as a success.

**The verdict of the model** is the pair `(K-SIEVE, K-STAT)`:

| K-SIEVE | K-STAT | reading |
|---|---|---|
| load-bearing | monotone | **the double dissociation**: compression is load-bearing where the copy carries damage and only there |
| load-bearing | not monotone | compression helps regardless; the theorem's assumption (c) or the count is wrong — reread |
| fires | monotone | compression is not load-bearing even where the theorem allows it; the record closes on the argument of 00 §0.3 |
| fires | not monotone | the protocol is not measuring what it claims; a bug before a finding |

## 2.3 The controls

| id | fires when | consequence |
|---|---|---|
| **K-REPLAY** | `U_K(sieve-c*) ≥ U_K(replay)` at the byte-matched buffer | the genome loses to storing the data it is supposed to replace; compression is not the mechanism, storage is; reported |
| **K-EWC / K-LWF / K-SP** | `U_K(sieve-c*)` not better than `ewc` (best `λ_E`), `lwf`, `shrink_perturb` by `3 σ_U` each | the sieve is no better than the conventional regulariser named; the margin is reported per control |
| **K-DREAM** | `U_K(sieve-c*) ≥ U_K(sieve-nodream-c*)` | dreaming does not pay; the core is consolidated from the phenotype alone |
| **K-DREAM-LIFE** | `U_K(sieve-life-c*) ≥ U_K(sieve-c*)`, **or** `≥ U_K(copy-dream)` | rehearsal in the lifetime does not pay, or pays without the sieve (the uncompressed store's own samples do the same): the sieve is not what the rehearsal needs |
| **K-GROW-C** | `U_K(sieve-c*) ≥ U_K(trunc-c*)` (no distillation) **or** `≥ U_K(sieve-fresh-c*)` | the distillation or the copied units is not what pays; the cheaper form is kept |
| **K-GENOME** | `G_K(c*) ≥ U_K(copy)` — the core, as a model, is worse than the copied phenotype on the union | the genome does not carry the life; the sieve's value, if any, is in the rebirth only |
| **K-SOTA-C** | `U_K(sieve-c*) ≥ U_K(iid)` at equal total tokens | the continual lineage is behind the mixture that needs all data at once; the gap is the price of sequence, reported |
| **G-NOISE** | a control's `σ` unmeasured | that criterion NOT EVALUATED |
| **G-EXPAND** | T2 of `01 §1.11` fails at `η = 0` | a bug; nothing runs |

`c*` is the argmin of K-SIEVE's frontier; if K-SIEVE fires, every control is read against the
best sieve arm and labelled as such.

## 2.4 Secondary readings (reported, not adjudicated)

* Retention `R_k(ε)` and adaptation `A_k(q)` per arm (01 §1.8): the sieve's expected signature
  is *lower* forgetting at the switch after the first (the core carries the past) and *equal
  or faster* adaptation (fresh capacity); the copy's is the reverse.
* The frontier's shape in `r` on both lives, with the width cores (`c2`, `c8`) and the depth
  cores (`c4`, `c16`) marked.
* The core's own curve `G_k` over the life (the compressed model's competence).
* The overhead per arm.

## 2.5 The runs

| run | what | arms | seeds | wall |
|---|---|---|---|---|
| **R0 CODE** | build `data/py_8192`; measure the story-trained phenotype's CE on it | — | — | minutes |
| **R1 NOISE** | `copy` on `SHIFT` and `STAT` | copy | 2 (then 4) | 4 × 9 min |
| **R2 FRONTIER** | K-SIEVE on `SHIFT` | sieve-c2/c4/c8/c16, scratch | 1, then 2 | 5 × 12 min |
| **R3 STAT** | K-STAT on `STAT` | sieve-c4, sieve-c8, scratch | 1, then 2 | 3 × 12 min |
| **R4 CONTROLS** | K-REPLAY, K-EWC, K-LWF, K-SP, K-DREAM, K-DREAM-LIFE, K-GROW-C, K-GENOME on `SHIFT` | replay, ewc(2), lwf, shrink_perturb, sieve-nodream-c*, sieve-life-c*, copy-dream, trunc-c*, sieve-fresh-c* | 1, then 2 | 10 × 10 min |
| **R5 IID** | K-SOTA-C | iid | 1 | 9 min |
| **R6 MATCHED** | `copy-m` and the second-wave arms (`sieve-life`) | — | 2 | — |
| **R7 SHIFT3** | the three-environment life (moved into the first wave, P-P.4): K-DREAM-LIFE where accumulation can show | copy, copy-dream, sieve-c8, sieve-life-c8, replay, scratch | 1, then 2 | 6 × 12 min |

Order: R0 → R1 → R2 → R3 → R4 → R5 → R6 → R7. A life is one job; the chain is serial and
pausable (`runs/palimpsest/PAUSE`).

## 2.6 Reporting rules

1. Every curve is held-out CE on an environment's evaluation set in nats; switches are
   annotated, never smoothed.
2. Every arm that failed, was cut, or was skipped is listed with the reason.
3. Every number in a report is re-measured from `runs/palimpsest/<life>/record.json`.
4. Negative results at equal prominence, in the same table.
5. Anything at one seed is a screen and says so; `docs/palimpsest/RESULTS.md` is rendered by
   the analyser and never edited by hand.
