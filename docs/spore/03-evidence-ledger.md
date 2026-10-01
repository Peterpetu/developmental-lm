# 03 — Evidence ledger: every finding, every correction, every "why", in the order obtained

**Status: record.** Entries are numbered `P-S.n` and are never rewritten; a later entry
corrects an earlier one in place by reference. Every number names its artifact under
`runs/spore/`; verdicts are those printed by `scripts/spore_analyse.py`. Times are local
(2026-09-09/10). The bottleneck queue is paused throughout by the user's order.

1. **The reading of the brief, and why the fifth model is backprop-free and hyperdimensional
   rather than another gradient-trained phenotype with a genome beside it.** *22:00.* The
   pasted prompt is the sibling project's (`HDC_LM`: HDC core; backprop-free). This
   repository's own prompt (2026-09-02) asked for two timescales and novel mechanisms. The
   record (`00 §0.2`) shows every two-timescale mechanism built beside a backprop substrate
   was bypassed or out-competed at matched compute (SOMA 0.04 %; GNOMON 5.2 vs 2.0; ANLAGE
   −0.115, −0.145, +0.002; TEMPO K-CONT, K-GENE; the replication's +0.37 at best). A convex,
   backprop-free lifetime is the one construction under which the genome is *unbypassable*
   and lifetimes are cheap enough for selection to move anything (`00 §0.3`). The two
   readings of the brief therefore converge on one design, and both constitutions are
   stated as satisfied or falsified by the same criteria (`02`). *Why recorded:* if the user
   intended only one reading, this entry is where to redirect.

2. **Design decisions taken before any number, with their reasons** (`01`). (a) Seeded
   atoms and a seeded permutation, never fit — the genome is a seed (Kanerva 2009); the token
   table, the binding constraint of the bottleneck replication (`tempo/11 §P-D.49`), no longer
   needs compressing. (b) Exact n-gram codes by permutation binding, normalised, so that
   suffix overlap is inner product `m/n` — the code *is* backoff and the readout learns the
   interpolation. (c) The in-context memory is the delta rule over fixed projections of the
   same codes, computed by the kernel verified to 1e-10 in this repository; the pair written
   at `t` is `(k_{t−1}, v_t)` so no target leaks (tested). (d) The readout is ridge in one
   streaming pass; the Gram is accumulated in float32 and solved by Cholesky. (e) Development
   is cascade-correlation with seeded candidates and selection by residual correlation; the
   `random` control decides whether selection is the mechanism. (f) Selection across
   lifetimes is `(μ + λ)` with elitism, fitness on the fitness set only.

3. **The pipeline runs end to end; a first smoke life exposes a modelling error in the head,
   fixed before any reading.** *22:15–22:24, `runs/spore/smoke-05M/` (D 1024, 0.5M tokens,
   three waves of 128 units, F 8577, 174–203 s, peak 6.8 GB).* The innate memory-only
   predictor scores **8.139** on the test set (null 9.011) with one scalar fit on the fitness
   set — the organism predicts repeated within-window contexts before it has read the
   corpus. The ridge scores passed through a calibrated softmax scored **6.011**, at the
   unigram entropy (5.96): the softmax of a *probability estimate* is exponential in the
   probability, so tokens at `p = 0.01` and `p = 0.001` receive nearly the same logit and the
   tail, where the loss mass lives, is flattened to uniform. **Correction:** the linear head
   (`01` Definition 5b, `spore/readout.py::linear_log_probs`): the least-squares scores are
   conditional-probability estimates and sum to one with the bias, so they are clipped,
   renormalised and mixed with the unigram (from the bias row of `C`) and the memory's
   cleanup, three scalars fit on the fitness set. Same life, same ridge solution: **4.411**.
   The log-linear refinement (the delta rule, Adam) from the softmax warm start reached
   **4.375** in 30 steps once the step size was raised from 3e-4 to 5e-3 (at 3e-4 it moved
   0.2 nats; the objective is convex, the step was simply too small for 30 updates).
   *Why:* the model class of the head, not the substrate, set the first number; both heads
   are recorded in every life from here on and the fitness set chooses.

4. **The count-model nulls on the same stream and windows** (`runs/spore/counts-05M/`,
   `counts-5M/`; `scripts/spore_baselines.py`; interpolation weights fit on the fitness set):

   | tokens | unigram | bigram (JM) | trigram (JM) | B-GD, the 13M transformer |
   |---|---|---|---|---|
   | 0.5M | 5.963 | 4.154 | 3.875 | 4.31 at 0.8M (train CE) |
   | 5M | 5.941 | 3.839 | 3.375 | 2.85 at 2.5M, 2.23 at 10M (≈ 2.5 at 5M, interpolated) |

   The smoke life at 0.5M and `D = 1024` (4.41 / 4.375) sits between the unigram and the
   bigram: with `D < V` the order-1 block is a rank-1024 image of a rank-8192 table. `D` is
   the first lever to measure (the chain's `D2048-05M`, `D4096o123-05M`).

5. **Throughput and the card's ceiling, measured.** At `D = 2048`, `F = 15,361`: 387 ms per
   batch of 32 × 512 tokens (features 337 ms of it) → a 5M-token pass in 2 min, 25M in 10;
   solve 0.4 s; peak 7.7 GB. At `D = 4096` with five orders, `F = 29,697`: **30.7 s per
   batch** and a 91 s solve at a 21.7 GB peak — the card's memory is oversubscribed and
   Windows pages CUDA allocations to host memory. `01 §1.3`'s stated ceiling (`F ≈ 40k`) was
   wrong by a factor of two; corrected to `F ≈ 22k` at batch 32 in float32 (`D = 4096` needs
   three orders). *Consequence:* `D = 2048` is tonight's workhorse; the regime `D ≥ V` that
   HDC actually wants needs a blockwise solver (`01 §1.9`'s scale path, moved forward if the
   D-slope says `D` is the lever).

6. **The first serial chain** (`scratchpad/spore_chain.py`, log `runs/spore/chain.log`;
   detached; one GPU job at a time; the queue file is re-read before every step):
   `base-5M-s0`, the D-slope at 0.5M (`D2048-05M`, `D4096o123-05M`), the K-HDC arms
   (`nomem-5M`, `nobind-5M`, later `D1024-5M`), the K-DEV arms (`growoff-5M`, `growrand-5M`),
   G-NOISE seeds (`base-5M-s1`, `-s2`), then `base-25M-s0`. Every 5M life is ~6 min.

7. **The first chain life ran five times slower than the probe, and the D-slope at 0.5M
   says `D` is the lever but the card cannot hold it as dense atoms.** *22:28–22:33.*
   `base-5M-s0`'s first pass averaged 1,916 ms per batch (probe: 387): the feature assembly
   held three 1 GB copies of the feature rows (`X0`, `Z`, `X_casc`) beside the Gram, and at
   the card's limit Windows pages. **Correction:** one buffer written block by block, the
   cascade input as a view, the cache released after growth (`spore/life.py::features`,
   `spore/cortex.py::forward_into`); the chain was stopped, patched and relaunched (4 min of
   compute lost). `D2048-05M` after the patch: 63 s for the whole life, peak 10.35 GB.
   Readings at 0.5M tokens, three waves of 128 units, the fitness set choosing the stage:

   | D | ridge, linear head | log-linear refinement (1 pass) | bigram counts | trigram counts |
   |---|---|---|---|---|
   | 1024 (`smoke-05M`) | 4.411 | 4.375 | 4.154 | 3.875 |
   | 2048 (`D2048-05M`) | 4.377 | **4.196** | 4.154 | 3.875 |

   The organism at `D = 2048` is still *behind the exact bigram counts*. The reason is a
   counting identity, not a bug: the order-1 code is a random `D`-dimensional image of the
   `V = 8192` one-hots, so the least-squares readout on that block is a rank-`D`
   approximation of the `V × V` bigram table; at `D < V` it cannot hold the table. HDC's own
   answer is `D ≫ V` (Kanerva's 10,000-dimensional atoms for vocabularies of thousands), but
   `D ≥ 8192` for seven blocks is `F ≈ 60k`, beyond this card's Gram (§P-S.5).
   **Decision (01 §1.2 variant 2a, `onehot`):** the order-1 block is the *identity code*
   (`D_1 = V`, the one-hot: the orthonormal atom set at `D = V`, exact by construction) and the
   higher orders keep their random atoms at `D`. `F` rises by `V` to `23.5k` (Gram 2.2 GB).
   This is the honest reading of "the substrate must be at least as wide as the vocabulary
   for the first order", and the higher-order blocks are where binding does its work. The
   refinement's gain grew with `D` (−0.04 at 1024, −0.18 at 2048): the log-linear readout
   benefits from rank where the linear head does not. Arm `onehot-5M` queued second.

8. **Two more levers built while the chain runs, both gradient-free, both registered before
   their numbers.** *22:40.* (a) **Semantic atoms (01 §1.1 variant 1b, `spore/semantic.py`,
   genes `eta_sem`, `d_sem`, `sem_frac`).** The one-hot block gives the exact bigram; what an
   n-gram model lacks and a neural one has is *generalisation across similar tokens*. Levy &
   Goldberg (2014) showed that skip-gram embeddings are a factorisation of the shifted PMI
   matrix, so the truncated SVD of the positive PMI of the bigram counts gives embeddings of
   the same kind in closed form. The organism's first developmental stage counts bigrams over
   `sem_frac` of its budget (a `bincount`), factorises the PPMI matrix by randomised SVD
   (1.3 s at `V = 8192`, `d = 256`), lifts `[U S^{1/2} ; V S^{1/2}]` to `D` by a seeded Gaussian
   map, and mixes it into the random atoms: `a'_v = normalise((1 − η) a_v + η lift(E_v))`.
   Every code, the memory's keys and values and the cleanup table are then built on atoms in
   which distributionally similar tokens are similar, so the ridge readout on a higher-order
   code generalises to unseen n-grams of similar words. Test: two planted token classes
   separate in the embedding and in the mixed atoms. Arms `sem05-5M` (η 0.5) and `sem1-5M`
   (η 1) queued after `onehot-5M`. (b) **The reservoir block (variant 2b, candidate B of the
   creative-reasoning pass; gene `reservoir`, `rho_res`)**: an echo-state recurrence over the
   atoms, spectral radius 0.9, reset per window, as an added feature block — the test of
   implicit recurrence against explicit binding; arm `res-5M`. (c) **G-DET** queued as
   `det-05M`, a repeat of `D2048-05M` under a different run name.

9. **The base organism at 5M tokens beats the trigram count model by a quarter of a nat, and
   the log-linear refinement is the largest single lever measured so far.** *22:38,
   `runs/spore/base-5M-s0/` (D 2048, five orders + bag + trajectory + memory, three waves of
   512 units → F 16,897, 308 s, peak 11.2 GB).*

   | stage | fitness | test |
   |---|---|---|
   | innate (memory only, `τ_m` 22.7) | — | 8.120 |
   | ridge, linear head (`μ` 0.041, `ν` 0.039) | 3.902 | 3.889 |
   | ridge, softmax head | 5.867 | 5.847 |
   | **log-linear refinement, one pass** | **3.149** | **3.122** |
   | trigram counts (same tokens) | 3.411 | 3.375 |
   | B-GD, the 13M transformer at 5M (interpolated) | — | ≈ 2.54 |

   R-BIRTH passes (−0.89 below the null with no pass over the corpus). R-PARADIGM at 5M
   passes on its face (−0.25 against the trigram; `σ_e` pending). The K-SOTA gap at 5M tokens
   is **+0.59 nats**. *Reading.* The ridge solution, used as probabilities, is at the level of
   an interpolated bigram (its order-1 block is rank-limited, §P-S.7); the same features
   under the multinomial-logistic readout, reached by one pass of the delta rule from the
   ridge warm start, are a quarter of a nat better than exact trigram counts. The log-linear
   readout is the right model class for these features (a product of experts over the order
   codes, the memory and the units — the log-linear form is what n-gram interpolation cannot
   express), and it is trained by a single-layer local rule: the paradigm's first number that
   a count model cannot reach. Next levers, in order of expected size: the exact order-1
   block (running), semantic atoms (queued), more refinement passes (`E = 3`, queued), `D`.

10. **The exact order-1 block buys 0.04 nats at eight times the cost; paging, not compute,
    is the cost, and the Gram was still alive during the refinement.** *23:18,
    `runs/spore/onehot-5M/` (F 25,089; 2,426 s; peak 16.2 GB).* Linear head 3.798 (base
    3.889: the exact bigram is worth 0.09 to the linear readout); refined **3.084** (base
    3.122: 0.04). The log-linear readout recovers most of what the exact bigram block adds
    from the random atoms — the rank limit of §P-S.7 binds the *linear* head, not the
    log-linear one. The run paged throughout (1.4–2.8 s per batch; the refinement pass took
    28 min against 2.3 at `F = 17k`): with `F = 25k` the Gram (2.5 GB) and cross-covariance
    (0.8 GB) sat on the card beside the readout, its two Adam moments and the feature batch.
    **Corrections:** the Gram is freed before the refinement (it is never read again);
    `onehot` stays a registered gene but is *off* in every further arm tonight (the semantic
    and `E = 3` arms were re-queued without it; the semantic arm that had started with it
    was killed a minute in and its directory removed). *Why:* a lever worth 0.04 that costs
    eight times the wall-clock is not the lever for a night of short experiments; it is
    recorded for the scale path, where a blockwise solve makes it free.

11. **Semantic atoms are a real lever at zero cost: −0.155 nats at 5M tokens, and the innate
    predictor improves by half a nat.** *23:24, `runs/spore/sem05-5M/` (η 0.5, d 256, counts
    over the whole budget; F 16,897; 308 s — the same wall-clock as the base).*

    | stage | base (random atoms) | semantic atoms, η = 0.5 | Δ |
    |---|---|---|---|
    | innate, memory only | 8.120 | **7.571** | −0.549 |
    | ridge, linear head | 3.889 | 3.813 | −0.076 |
    | **log-linear refinement** | 3.122 | **2.967** | **−0.155** |

    The K-SOTA gap at 5M falls from +0.59 to **+0.43**. *Reading.* Every code, the memory's
    keys and values and the cleanup are built on atoms in which distributionally similar
    tokens are similar (PPMI-SVD of the organism's own first-stage counts, no gradient), so
    (i) the in-context memory generalises — a name stored after "said" is retrieved after
    "shouted" — which is the half-nat at birth; (ii) the readout on the higher-order codes
    generalises to n-grams of similar words, which is the 0.15 after refinement. This is the
    lever a count model lacks and a neural one buys with its embedding table, obtained here
    by a closed-form factorisation of counts: the organism's first developmental stage
    (counting) shapes the substrate of its second (coding and reading out). Arms queued:
    η = 1 (running), `d_sem` 512, memory `d_k` 2048 on the semantic base.

12. **Where the gap to the transformer lives, by stratum: rare tokens.** *23:30.* Per-stratum
    test CE of `sem05-5M` (5M tokens) against the 13M transformer's final strata (201M
    tokens — the *pattern*, not a matched-token contrast; `free-tinystories-s0-seed0`):

    | position × target frequency rank | transformer | SPORE (sem05, 5M) | SPORE innate |
    |---|---|---|---|
    | any × rank < 100 (63 % of targets) | 0.80–1.26 | 1.27–2.05 | 6.5–7.4 |
    | any × rank 100–1000 (28 %) | 2.37–2.62 | 3.82–4.15 | 7.9–8.7 |
    | any × rank ≥ 1000 (9 %) | 3.91–4.70 | **6.51–7.06** | 8.0–8.8 |

    The organism is within 0.5–0.8 nats on frequent tokens and **three nats behind on rare
    ones**: each rare token's readout column (`F` numbers) is learned from a few dozen
    occurrences, whereas a neural model's tied output embedding shares statistical strength
    across similar tokens. The innate memory is best at deep positions (6.5 at `pos ≥ 256`,
    rank < 100: repeated names and phrases). **Decision (01 Definition 6b, gene `lro`):** an
    output-side sharing path in the refinement, `ℓ = Zᵀ W + (Zᵀ U) E_outᵀ + τ_m m`, with `E_out`
    the normalised PPMI-SVD embedding of the *next* token (the same factorisation as the atoms)
    and `U ∈ R^{F×2d}` trained by the same delta rule; the logits stay linear in `(W, U)`, the
    objective convex, the rule single-layer (tested against autograd). Arm `sem05-lro-5M` queued.

13. **Three passes of the delta rule instead of one: −0.29 nats. The convex readout is far
    from its optimum after one pass, so the lifetime is optimisation-limited, not
    representation-limited, at this point.** *23:39, `runs/spore/E3-5M/` (the base
    organism, `E = 3`, 575 s).* Refined test **2.836** (one pass: 3.122; pass-mean train CE
    3.34 → 3.04 → 2.94, still falling). `sem1-5M` (η = 1, 23:29): innate 7.170, refined 2.996 —
    worse than η = 0.5 after refinement (2.967) though better at birth: the pure distributional
    atoms lose the orthogonality that carries token *identity*; the mixture keeps both, and
    η = 0.5 is the semantic base from here on. *Reading.* The log-linear readout over `F ≈ 17k`
    features and `V = 8192` outputs is a convex problem with 138M parameters; Adam at 5e-3 for
    306 mini-batches does not reach its optimum, and every further pass over the *same* tokens
    (information-matched with the baselines; compute is reported separately) buys a tenth of a
    nat or more. Combined with the semantic atoms this projects to ≈ 2.7 at 5M tokens against
    the transformer's ≈ 2.54. Queued: `sem05-E3-5M` (both levers), `lr1e-2-5M` (a larger step
    for one pass, to see whether passes or step size is the lever).

14. **Output-side sharing is a null result, and its run exposed a determinism defect.** *23:44,
    `runs/spore/sem05-lro-5M/`.* Refined 2.964 against 2.967 without it; every stratum within
    0.02; the rare-token strata unchanged (7.08 vs 7.06). Adam already equalises the step size
    of rare and frequent columns, and one pass with `U` from zero adds nothing the full-rank `W`
    does not already learn; the rare-token gap is not a step-size problem. Recorded as a
    negative result; `lro` stays off. **The defect:** the innate CE of this run (7.5594)
    differed from `sem05-5M`'s (7.5714) although the atoms should be identical — the
    randomised SVD (`torch.svd_lowrank`) draws its test matrix from the *global* RNG, which
    is not seeded per process, so the semantic atoms were not a deterministic function of the
    genome. **Fixed:** the SVD runs under a forked RNG state seeded from `s_A` (test added:
    two calls with an intervening global draw give identical factors). Arms before this fix
    (`sem05-5M`, `sem1-5M`, `sem05-lro-5M`, `sem05-d512-5M` if started before 23:47) carry
    `O(0.01)` of this non-determinism in their numbers; G-DET is re-checked on the semantic
    base with `sem05-05M` / `sem05-det-05M` at the end of the chain.

15. **Two dials of the semantic base, one pass each.** *23:49–23:56.* `d_sem = 512`
    (`sem05-d512-5M`): refined 2.983, innate 7.732 — worse than 256 (2.967, 7.571) on both;
    the extra factors add noise to the atoms faster than structure. Memory `d_k = d_v = 2048`
    (`sem05-dk2048-5M`, F 17,921, 376 s): refined **2.944**, innate 7.517 — −0.023 for a 20 %
    longer life; the retrieval block's rank is a small lever. The semantic base of the
    evolutionary loop is therefore `D 2048, η 0.5, d 256, d_k 1024, E 1` (the cheaper memory;
    `d_k` is a gene the loop can move), and the report-budget configuration adds `d_k 2048`
    and `E = 3` if `sem05-E3-5M` confirms the two levers add.

16. **The two levers add: 2.719 nats at 5M tokens, 0.18 behind the 13M transformer at the
    same tokens.** *00:05, `runs/spore/sem05-E3-5M/` (η 0.5, d 256, d_k 1024, E 3; F 16,897;
    576 s).*

    | organism at 5M tokens | test CE |
    |---|---|
    | base (random atoms, one pass) | 3.122 |
    | + three passes (`E3-5M`) | 2.836 |
    | + semantic atoms (`sem05-5M`) | 2.967 |
    | **+ both (`sem05-E3-5M`)** | **2.719** |
    | trigram counts | 3.375 |
    | the 13M transformer (interpolated in log-tokens between its 2.5M and 10M points) | ≈ 2.54 |

    Additive to 0.04 (predicted 2.68 from the two single-lever gains). Per stratum the rare
    tokens moved most (7.0 → 6.1–6.2), the frequent ones from 1.27–2.05 to 1.18–1.93. *Reading.*
    A backprop-free organism — seeded hypervectors, permutation binding, a delta-rule memory,
    units grown by residual correlation, a ridge readout refined by the delta rule on atoms
    factorised from its own counts — is within a fifth of a nat of a gradient-trained
    transformer at equal tokens on TinyStories at 5M. What it costs: 576 s against the
    transformer's ≈ 100 s for the same tokens (S7: tokens are matched, compute is reported).
    **From here the organism of record is the semantic base**, so the K-HDC, K-DEV and G-NOISE
    arms are re-queued on it (η 0.5, E 1 for cost), and the 25M report run is `sem05-E3-25M`.

17. **The step size is half the lever the pass count is, at a third of the cost.** *00:10,
    `runs/spore/lr1e-2-5M/` (base organism, one pass at Adam 1e-2 instead of 5e-3).* Refined
    **2.976** against 3.122 at 5e-3 (−0.146) and 2.836 with three passes at 5e-3 (−0.286). The
    readout's optimisation is the lever in both forms; the evolutionary screen at 1M tokens
    uses 1e-2 with one pass (cheap), the report run keeps 5e-3 × 3 passes so that its 5M and
    25M points share one recipe for the K-SOTA slope. Queued: 2e-2 (one pass) and the
    combination η 0.5 × 1e-2 × 3 passes, after the report run.

18. **`D` is a weak lever at its cost.** *00:13, `runs/spore/D4096o123-05M/` (D 4096, orders
    {1, 2, 3}, F 21,889; 0.5M tokens).* Refined 4.166 against 4.196 for `D = 2048` with five
    orders (−0.03) at 2.4× the wall-clock (153 s vs 63 s, peak 12.8 GB, no paging). With the
    log-linear readout the substrate's width above 2048 buys little; the levers that paid
    tonight are the readout's optimisation and the atoms' geometry. `D = 2048` stays.

19. **The instrument and the constitutional verdicts on the semantic base at 5M tokens:
    K-HDC does not fire (every HDC mechanism is load-bearing); K-DEV fires (growth is worth
    0.02 and its selection nothing).** *00:45; `scripts/spore_analyse.py` output quoted.*

    **G-NOISE.** Three atom seeds of `sem05-5M`: test CE 2.9670, 2.9656, 2.9667 → **`σ_e` =
    0.0007 nats** (2 df); the linear head's 0.0015. The organism is deterministic given the
    corpus, and the seed's only trace is the `1/√D` interference of the codes, which the
    readout averages over five million tokens. The 3 `σ_e` bar is 0.0022 nats — every contrast
    below is far above it, so the verdicts are not marginal.

    | arm (semantic base, one pass) | test CE | Δ vs base | linear-head Δ |
    |---|---|---|---|
    | `sem05-5M` (base) | 2.967 | — | — |
    | `s-nobind-5M` (`ρ = I`: bags, no order) | 3.004 | **+0.037** | +0.008 |
    | `s-nomem-5M` (no in-context memory) | 3.089 | **+0.122** | +0.027 |
    | `s-D1024-5M` (D halved; the smallest the genome allows) | 3.060 | **+0.093** | +0.077 |
    | `s-growoff-5M` (no units, F 15,361) | 2.986 | **+0.019** | −0.001 |
    | `s-growrand-5M` (1536 units kept at random) | 2.962 | **−0.005** | +0.000 |

    **K-HDC does not fire**: order binding, the superposition memory and the code width each
    cost ≥ 17 `σ_e` when removed. The memory is the largest (its loss concentrates at deep
    positions and rare tokens — the copy mechanism for names; the innate predictor without
    binding falls from 7.57 to 8.06 because a bag key cannot tell "A said to B" from "B said
    to A"). The width ablation is `D/2`, not the pre-registered `D/4`: 512 is below the
    genome's range, which is recorded as a deviation. **K-DEV fires**: the 1536 units grown
    by residual correlation are worth 0.019 nats (≥ 3 `σ_e`, so growth is not nothing), but
    units kept *at random* from the same candidates are worth the same (−0.005, i.e. slightly
    more). The cascade-correlation score (kept 0.017 vs all 0.011) discriminates candidates,
    yet what a sparse random ReLU over 64 coordinates of the code adds to a log-linear readout
    does not depend on which one is kept. Development, as implemented, is decorative at this
    budget: reported as such (C4 fails in this form). Arms queued to test whether the *unit
    class* is the reason: dense units over 2048 coordinates, with their random control, and
    cosine (random Fourier) units. **K-REFINE does not fire** (−0.85 nats). **R-BIRTH** passes
    at −1.44 nats below the null.

20. **Step size, continued: 2e-2 gives 2.890 in one pass.** *00:50, `runs/spore/lr2e-2-5M/`
    (base organism).* 5e-3 → 3.122, 1e-2 → 2.976, 2e-2 → **2.890**, against 2.836 for three
    passes at 5e-3. The convex readout is still under-optimised at every setting tried; the
    step-size axis has not saturated. Queued `lr4e-2-5M` (one pass) after the report run; the
    report run (`sem05-E3-25M`, already running at 5e-3 × 3) keeps its recipe for the slope
    reading, and a second 25M run at the best recipe follows if the night allows.

21. **The report budget: 2.462 nats at 25M tokens — beyond any count model, 0.60 behind the
    13M transformer, and K-SOTA fires: the organism's curve is flatter (−0.11 nats per doubling
    of tokens against the transformer's −0.29), so its ceiling is representational.** *01:35,
    `runs/spore/sem05-E3-25M/` (η 0.5, d 256, d_k 1024, E 3 at 5e-3; F 16,897; 2,662 s of
    which 1,999 s are the three refinement passes; peak as at 5M).*

    | tokens | ridge, linear head | refined (E 3) | trigram counts | 13M transformer | gap |
    |---|---|---|---|---|---|
    | 5M | 3.812 | 2.719 | 3.375 | ≈ 2.54 (interp.) | +0.18 |
    | 25M | 3.704 | **2.462** | (`counts-25M` running) | 1.857 (4 seeds) | **+0.60** |

    Per stratum at 25M: frequent tokens 1.09–1.80, mid 3.16–3.44, rare 5.0–5.35 (from 6.1–6.2
    at 5M: the rare tokens gained most from tokens, as they should). The refinement's
    pass-mean train CE went 2.839 → 2.566 → 2.530: the third pass adds a thirtieth of the
    second's gain, so at 25M the convex readout is near its optimum for these features —
    unlike at 5M, where more passes and larger steps kept paying. *Reading (S7).* At 5M the
    organism is optimisation-limited and within a fifth of a nat of the transformer; at 25M
    it is feature-limited: a log-linear readout over fixed codes, a fixed memory and 1,536
    random units cannot absorb five times the data the way a network that learns its features
    can. This is the pre-registered meaning of K-SOTA firing: the lever for more tokens is
    richer features, i.e. development that is not decorative — exactly the mechanism K-DEV
    found inert in its current form (§P-S.19). **Decision:** a unit class grown *from the
    organism's own statistics in closed form* (§P-S.22) is built and tested before the second
    25M run.

22. **The 25M nulls, and a unit class born from the organism's own statistics in closed
    form.** *01:36–01:40.* `counts-25M`: unigram 5.940, bigram 3.747, trigram **3.168** (weights
    0.98 / 0.78): R-PARADIGM at 25M passes by **0.71 nats**. **Definition 11b (`growth = ridge`,
    `spore/readout.py::solve_targets`, `spore/cortex.py::hash_targets`).** A wave draws `k`
    seeded `±1` hashes of the vocabulary `H ∈ {±1}^{V×k}` — random token subsets — and each
    unit is the ridge predictor of its subset's membership from the features accumulated so
    far, `w_u = (G + λ Ḡ I)^{−1} C h_u`, normalised, with `ψ` on top. The solve reuses the
    Gram the lifetime already holds (one Cholesky with `k` right-hand sides; no scoring pass);
    the unit is "the best linear guess at whether the next token lies in subset `S_u`", and
    its rectification is a nonlinear feature a log-linear readout can use as a gate. It is
    developmental in the strict sense of S4 — born from what the organism has learned, frozen
    after birth — and convex (S1). Test: on a planted problem the units correlate `> 0.3`
    with their subsets and equal `W_ridge H` (the projection commutes with the solve). Arm
    `s-gridge-5M` queued against the base (2.967) and the random-unit control (2.962).

23. **Passes and step size add: 2.634 at 5M tokens, a tenth of a nat behind the transformer.**
    *01:44, `runs/spore/sem05-E3-lr1e-2-5M/` (η 0.5, E 3 at 1e-2; 577 s).* Against 2.719 at
    E 3 × 5e-3 and 2.967 at E 1 × 5e-3; pass means 3.117 → 2.859 → 2.748, still falling at the
    third pass. The K-SOTA gap at 5M is **+0.10** (2.634 vs ≈ 2.54 interpolated). The recipe
    of the second report run is decided by `lr4e-2-5M` (one pass) and `s-gridge-5M` (the
    ridge-born units), both running next.

24. **The step-size axis saturates at 2e-2 for one pass.** *01:50, `runs/spore/lr4e-2-5M/`.*
    4e-2 → 2.905 against 2.890 at 2e-2 (5e-3 → 3.122, 1e-2 → 2.976). The second report run
    therefore uses the *measured* multi-pass recipe (E 3 at 1e-2, §P-S.23) with `d_k 2048`
    (§P-S.15), plus the ridge-born units if `s-gridge-5M` shows a gain.

25. **Ridge-born units beat random units by 0.021 and no units by 0.046, in less time: with
    Definition 11b, development is load-bearing over its random control.** *01:55,
    `runs/spore/s-gridge-5M/` (η 0.5, E 1, three waves of 512 ridge units; 285 s against 309
    for the scored waves — no scoring pass).*

    | growth (semantic base, one pass) | linear head | refined | time |
    |---|---|---|---|
    | off | 3.812 | 2.986 | 247 s |
    | random units (11) | 3.813 | 2.962 | 309 s |
    | selected by residual correlation (11) | 3.813 | 2.967 | 308 s |
    | **ridge-born (11b)** | **3.747** | **2.941** | 285 s |

    The linear head gains most (−0.066): a unit that is the best linear predictor of "is the
    next token in subset S" is informative on its own, where a sparse random ReLU is not.
    Under the log-linear readout the gain is 0.021 over random units (30 `σ_e`) and 0.046
    over none. K-DEV in its first form stands as fired (§P-S.19: selection among random
    candidates is not the mechanism); in its 11b form it does not fire — the *birth rule*
    is the mechanism, and the analyser reports both. The effect is small at 5M; whether it
    grows with tokens is what the second report run measures. **Second report run
    (`final-25M`, queued next):** η 0.5, d 256, `d_k` 2048, E 3 at 1e-2, ridge-born units.

26. **G-DET fires at 1.6e-4 (bar 1e-4); the cause is found, fixed and re-queued.** *01:56–02:05,
    `runs/spore/D2048-05M/` vs `det-05M/`: 4.195676 vs 4.195517.* Two identical lives differ by
    1.6e-4 nats — a quarter of `σ_e` and 1 % of the smallest contrast read, so no verdict
    moves, but the gate was pre-registered at 1e-4 and it fired. Cause, isolated in a
    12-line test: `index_add_` on CUDA (the cross-covariance `C += Zᵀ Y`) uses atomic adds and
    returns values differing at 1e-6 between two calls on identical inputs; the Gram's GEMM
    is bit-reproducible. **Fix:** `C` is accumulated as a cuSPARSE sparse-dense product with
    the one-hot targets as a CSR matrix — bit-reproducible in the same test, and 3 ms per
    batch. The re-check pair `det2a-05M` / `det2b-05M` is queued after the semantic pair; the
    runs before this fix carry ±2e-4 of this noise. (`final-25M`, already running, is one of
    them.)

27. **The second report run: 2.408 nats at 25M tokens.** *02:49, `runs/spore/final-25M/`
    (η 0.5, d 256, `d_k` 2048, E 3 at 1e-2, ridge-born units; F 17,921; 3,211 s, of which
    2,406 s the three refinement passes; run before the determinism fix, ±2e-4).*

    | organism at 25M tokens | linear head | refined |
    |---|---|---|
    | first recipe (`sem05-E3-25M`: `d_k` 1024, 5e-3, scored units) | 3.704 | 2.462 |
    | **second recipe (`final-25M`)** | **3.638** | **2.408** |
    | trigram counts | | 3.168 |
    | the 13M transformer (4 seeds) | | 1.857 |

    The three levers added at 5M (−0.023, −0.085, −0.026) add at 25M to −0.054. Per stratum:
    frequent 1.08–1.77, mid 3.08–3.37, rare 4.84–5.23. Pass means 2.816 → 2.546 → 2.489: the
    readout is close to its optimum. **The night's headline:** a next-token model with no
    backpropagation anywhere — seeded hypervectors, permutation-bound codes, a delta-rule
    memory, units born by a ridge solve, a log-linear readout trained by the delta rule on
    atoms factorised from its own counts — scores **2.41 nats at 25M tokens** on the test set
    every model here is scored on, 0.76 below the best count model and **0.55 above** a 13M
    gradient-trained transformer at the same tokens; at 5M tokens the gap is 0.10. K-SOTA
    stands as fired on the slope: the paradigm's curve flattens where the transformer's does
    not, and the reason is representational (§P-S.21).

28. **G-DET passes once cuBLAS is made deterministic: 2.0e-5 between two identical lives.**
    *02:56.* The CSR fix alone was not enough: `sem05-05M` / `sem05-det-05M`, both after it,
    still differed by 1.5e-4, and a mixed pair (`det2a` before, `det2b` after the second fix)
    by 2.9e-4. The remaining source is cuBLAS, which picks split-K reductions by available
    workspace unless `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA initialises; the
    scripts now set it, and the life runs under `torch.use_deterministic_algorithms(True,
    warn_only=True)`. The clean pair `det3a-05M` / `det3b-05M`: 4.195673 vs 4.195653,
    **|Δ| = 2.0e-5**, under the 1e-4 bar (not bit-identical: the float64 stratum sums of the
    evaluation still use an atomic `index_add_`, which is where the last 2e-5 lives and which
    touches no training quantity). Every run before 02:52 carries up to ±3e-4 of this noise;
    the smallest contrast any verdict rests on is 0.019.

29. **Dense random units are no better than sparse ones; the cosine arm crashed on a
    non-finite statistic and a guard now makes such a life fail at once.** *03:01–03:20.*
    `s-kw2048-5M` (2048 coordinates per unit, selected): 2.9625; `s-kw2048-rand-5M`: 2.9617 —
    both equal to the sparse arms (2.967 / 2.962). Over random directions the unit class does
    not matter; the ridge-born units (2.941) remain the only development that pays.
    `s-cos-kw2048-5M` (random Fourier units) died at the end of its life with
    `KeyError: None`: every head's CE was NaN, so no stage could be chosen. On CPU at small
    scale the cosine features, the memory and the codes are finite, so the non-finite value
    arose in the GPU life's solve or calibration and was not isolated tonight; the arm is
    recorded as **NOT EVALUATED (crashed)**. **Correction:** the solve now raises
    `FloatingPointError` on a non-finite Gram or readout and the life raises on a non-finite
    CE at either head (`01 §1.10`'s rule made mechanical; test added), so a failure of this
    kind is named at its source instead of surfacing as a missing key.

30. **The evolutionary loop's own record: 32 lives, four generations, 1.84 h; the elite is
    0.27 nats below the base at the 1M screen, and three of its four moved genes are known
    monotone levers.** *03:11–05:02, `runs/spore/evo-1M/` (base: η 0.5, D 2048, k_wave 256,
    E 1, refinement 1e-2 × 1 pass; pop 8, elites 2, (μ + λ), initial mutation probability
    0.35 per gene, then `1/|genes|`).* The plan of `02 §2.6` (correction) assumed 1.5-minute
    lives; generation 0 took 48 minutes because heavy mutants (`D 4096`, the one-hot block,
    the reservoir, `E 2`) cost up to six minutes each, and one child of generation 1 exceeded
    the card's memory (scored unfit, discarded) while one of generation 2 was invalid (its
    memory key order outside its coded orders — a constraint the mutation operator should
    enforce; scored unfit, discarded). Elites by generation on the fitness set (test in
    brackets): 3.3806 (3.3452) → 3.3806 → 3.1919 (3.1727) → **3.1909 (3.1703)**; the base at
    the same budget 3.4437 (3.4352). The final elite's genome moved: `E` 1 → 2, `d_k` 1024 →
    512, `reservoir` off → on with `ρ` 0.61, and the memory seed. *Reading, per `02 §2.5`
    rule 5:* `E` is a monotone lever measured earlier (§P-S.13/23), `d_k` a measured weak one
    (§P-S.15, in the other direction at 5M); the reservoir block is the one gene whose effect
    was *unknown* before the loop, and generation 2's runner-up reached 3.2183 with `E 2` and
    `d_k 512` alone. The one-dimensional sweeps at the screening budget (`sw-E2-1M`,
    `sw-res-1M`, `sw-E2-res-dk512-1M`, …) and the elite re-scored at 5M against the base and
    against the base with `E 2` (`elite-5M`, `sem05-E2-5M`) are queued; K-EVO is read from
    those, not from the screen.

31. **The reservoir block is a real lever (−0.036 at 5M), and the evolutionary loop found it
    before the pre-registered arm ran.** *05:08, `runs/spore/s-res-5M/` (semantic base + an
    echo-state block of width 2048 at spectral radius 0.9; F 18,945; 388 s).* Refined
    **2.931** against 2.967 (50 `σ_e`); linear head 3.818 vs 3.813 (the block's value is in
    the log-linear readout, not the linear one). Candidate B of the creative-reasoning pass
    (`00 §0.4` T4) was registered as a control against explicit binding; as an *added* block
    it complements the codes: a nonlinear fading trace of the whole window beside the exact
    n-gram codes and the linear fading bag. The loop's elite switched it on (at `ρ` 0.61)
    from a base that had it off, which is the one gene of §P-S.30 whose effect no earlier
    arm had measured.

32. **K-EVO, read at 5M as pre-registered: the elite beats the base by 0.153, but the base
    with the elite's one monotone gene beats the elite by 0.013 — selection over lifetimes
    found nothing that carries to the report budget beyond a lever already known.** *05:24,
    `runs/spore/elite-5M/` (the loop's final genome: E 2, `d_k` 512, reservoir at ρ 0.61,
    `k_wave` 256 → 768 units, a new memory seed) and `runs/spore/sem05-E2-5M/` (the base with
    E 2 only); both at the life script's default refinement step 5e-3, as was the base.*

    | at 5M tokens | test CE |
    |---|---|
    | base (`sem05-5M`) | 2.967 |
    | elite of the loop (`elite-5M`) | 2.814 |
    | base + E 2 (`sem05-E2-5M`, the one-gene sweep) | **2.801** |
    | base + reservoir at ρ 0.9 (`s-res-5M`, §P-S.31) | 2.931 |

    The pre-registered inequality `CE(elite) − CE(base) ≤ −3 σ_e` holds (−0.153 = 220 `σ_e`),
    and rule 5 of `02 §2.5` decides what it means: the whole of the gain and more is the
    second refinement pass, a gene whose effect was measured to be monotone before the loop
    ran (§P-S.13/23); the loop's other moves — `d_k` 512, fewer units, the reservoir at 0.61
    — sum to **+0.013** at 5M against E 2 alone, although the reservoir on its own at 0.9 is
    worth −0.036. **K-EVO fires** in the sense that matters: what the screen at 1M selected
    (a smaller memory, fewer units) does not carry to 5M, and the one genuinely new find (the
    reservoir) was found at a setting that does not pay when combined. The search was also
    starved: 32 lives in 1.84 h against a plan of ~60, because mutation was free to propose
    six-minute genomes and two invalid ones. What would make a next loop informative is
    recorded, not done: a cost-aware fitness or a mutation operator that respects the card's
    ceiling and the genome's own constraints (`n_k ∈ N`), a screening budget closer to the
    report budget, and the known monotone genes fixed at their measured optimum so that
    selection is asked only the questions the sweeps cannot answer. The 1M sweeps
    (`sw-*-1M`) complete the record of which genes moved and how at the screen.

33. **The one-gene sweeps at the screening budget: a two-life sweep of the elite's monotone
    gene beats the loop's 32-life elite on the loop's own screen.** *05:25–05:35,
    `runs/spore/sw-*-1M/` (1M tokens, the loop's configuration: `k_wave` 256, refinement 1e-2).*

    | genome at the 1M screen | test CE | vs base |
    |---|---|---|
    | base (`sw-base-1M`) | 3.4352 | — |
    | base + E 2 | **3.1487** | −0.287 |
    | base + γ_b 0.83 | 3.4481 | +0.013 |
    | base + E 2 + γ_b 0.83 | 3.1731 | −0.262 |
    | base + reservoir (ρ 0.61) | 3.4031 | −0.032 |
    | base + E 2 + reservoir + `d_k` 512 (the elite's genes) | 3.1739 | −0.261 |
    | the loop's elite (E 2, reservoir 0.61, `d_k` 512, memory seed) | 3.1703 | −0.265 |

    Every gain the loop reported is the second refinement pass; γ_b 0.83 (generation 2's
    elite) is a null gene that rode on E 2; the reservoir is a real but small lever (−0.032,
    matching −0.036 at 5M); `d_k` 512 costs 0.025 beside E 2. The elite is 0.022 *worse* than
    `base + E 2` on the very set that selected it: the genome `base + E 2` was never sampled,
    because mutation moved several genes at once. This is the cleanest statement of
    §P-S.32: K-EVO fires; the loop's information rate over 1.84 hours was below that of two
    lives placed by the sweeps the protocol already required.

34. **The night, closed.** *05:40. GPU idle; the bottleneck queue still paused by the user's
    order; `screen/landscape.py` untouched.* Built from nothing at 22:00: a specification in
    three documents, a package of nine modules with 23 tests (G-BPF among them), a serial
    runner, 60 lives and count models, 34 ledger entries, seven commits. What stands:

    | claim | status | where |
    |---|---|---|
    | a next-token model with no backpropagation anywhere reaches **2.41 nats at 25M tokens** and **2.63 at 5M** on the test set every model here uses (count models 3.17 / 3.38; the 13M transformer 1.857 / ≈ 2.54) | measured; G-DET 2e-5; `σ_e` 0.0007 | §27, §23, §28, §19 |
    | the HDC substrate is load-bearing: binding +0.037, memory +0.122, width +0.093 when removed | K-HDC does not fire | §19 |
    | an innate in-context predictor at birth: 7.4–7.6 vs 9.01 | R-BIRTH passes | §3, §11 |
    | semantic geometry without gradients: −0.155 at 5M, half a nat at birth | measured | §11 |
    | the convex readout's optimisation is the largest lever (passes, step size: −0.49 at 5M) | measured | §13, §17, §20, §23 |
    | development: ridge-born units pay 0.046 over none, 0.021 over random; residual-correlation selection does not | K-DEV fires (11), does not fire (11b) | §19, §25 |
    | selection over lifetimes: the loop's gain is a known monotone gene; a two-life sweep beats its 32 lives | K-EVO fires | §30, §32, §33 |
    | the curve is flatter than the transformer's (−0.11 vs −0.29 nats per doubling): the ceiling at 25M is representational | K-SOTA fires | §21 |
    | nulls: the exact order-1 block at its cost, output-side sharing, `D` 4096, pure distributional atoms, `d_sem` 512, dense random units, γ_b 0.83; the cosine arm crashed (guards added) | recorded | §10, §14, §15, §18, §29, §33 |

    *What the record says to do next, in order of evidence.* (1) Development that adds
    features with the data: ridge-born units are the one growth rule that paid; growing them
    in more waves and from richer inputs (units of units) is the direct test of whether the
    slope can be bent. (2) A blockwise solver so that `D ≥ V` and the exact order-1 block are
    free (the one-hot block was worth 0.04 to 0.09 at a cost the card could not pay). (3)
    A selection loop whose mutations respect the card and the genome's constraints, with the
    monotone genes fixed and a cost-aware fitness — or none: the sweeps did its job. (4) The
    Pile lineage, which the brief names and which was not run. None of these is claimed.
