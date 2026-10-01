# 01 — The model: a lineage that lives across environments, consolidates into a small core, and is reborn from it by growth

**Status: normative.** Every object has its dimensions, every rule its inputs, outputs and
termination, every constant its default, range and reason. Symbols are fixed in §1.9. Code:
`palimpsest/`; tests: `tests_palimpsest/`.

## 1.1 Environments and the life

**Definition 1 (environment).** An environment `ε` is a token distribution given as a
training shard, with a fixed **evaluation set** `Ev(ε)` of `n_ev = 256` windows of `T + 1`
tokens (targets `T = 512`) drawn by the v3 rule (`anlage/data.py::EvalSet.sample_documents`,
document residue 2) and a disjoint **fitness set** (residue 0) from the same holdout, its
token counts and frequency rank, and a **window pool** `Pool(ε, s)` of `n_pool = 200,000`
windows drawn from the training shard by the pool seed `s` (`screen/crn.py::WindowPool`).
Every phenotype is scored on `Ev(ε)` as the mean per-token next-token cross-entropy in nats.

The local environments, all at the vocabulary `V = 8192` of the story tokenizer (byte-level;
it round-trips every source exactly):

| name | shard | what it is | tokens |
|---|---|---|---|
| `stories` | `data/ts_8192` | TinyStories V2 (GPT-4 subset) | 452.9M train |
| `code` | `data/py_8192` | Python source of the local interpreter's `Lib` and `site-packages`: one file per document, de-duplicated, seeded order, first 3,000 files held out (`scripts/palimpsest_build_code.py`) | ≈ 75M train at 0.51 tokens/char |
| `tasks` | generated | the reserved-alphabet verifier stream (`bottleneck/transfer.py`): `PARITY, MODCOUNT, PERM, DYCK, COPY, MQAR` at default parameters, ids in `[0, 64)`, which natural text never emits | unbounded, seeded |

**Definition 2 (life).** A life is a sequence of **phases** `(ε_1, s_1), …, (ε_K, s_K)`, each
of `T_φ` training tokens on `Pool(ε_k, s_k)` in the order of the order seed `o_k`. The
**shift life** alternates environments; the **stationary double** uses one environment in
every phase with distinct pool seeds. Both are run under common random numbers: two arms of
one life read the same windows in the same order (P8).

```
   SHIFT:  (stories, s_1) → (code, s_2) → (stories, s_3) → (code, s_4)          K = 4
   STAT:   (stories, s_1) → (stories, s_2) → (stories, s_3) → (stories, s_4)    K = 4
   SHIFT3: (stories, s_1) → (code, s_2) → (tasks, s_3)                          K = 3
   T_φ = 8·10^6 tokens (default; range 2·10^6 … 25·10^6; measured to place the 13M phenotype at
        ≈ 2.4–2.6 nats on stories, where the copy has something to forget)
```

## 1.2 The phenotype and its lifetime

**[NORMATIVE]** `W` is ANLAGE's stage-0 hybrid `s0: d = 256, L = 8, h = 4, n_kv = 2, F = 1024,
d_head = 64`, tied head, `13,011,201` parameters (`anlage/model.py::Phenotype`; `04 §4.3–4.5`
of `docs/anlage`), trained within a phase by the ANLAGE recipe (`anlage/recipe.py`: Muon on
matrices at `η_M = 0.02`, AdamW on the table at `3·10^{-3}` and vectors at `10^{-3}`,
warm-up `2 %`, constant, decay to zero over the last `20 %` of the phase, batch warm-up over
the first `10 %`, `B = 32 × 512`, clipping `1.0`, bf16 autocast). Each phase restarts the
schedule and rebuilds the optimiser: an arm that continues (`copy`) continues its weights,
not its optimiser state, which the decay to zero has stopped (ANLAGE `06 §6.2` step 5).

The lifetime is free (P3): the only loss is `CE + 10^{-4} (log Z)^2`. The regularised
controls of §1.7 add one labelled term each.

## 1.3 The core

**Definition 3 (core).** The core `C` is a phenotype of the same family at a smaller
configuration `c = (d_c, L_c, h_c, kv_c, F_c)` with `d_head = 64`, obtained from the full
configuration by integer factors `k_d` on width (`d_c = d/k_d`, `F_c = F/k_d`, `h_c = h/k_d`,
`kv_c = max(1, kv/k_d)`) and `k_L` on depth (`L_c = L/k_L`). The **ratio** is
`r = |W| / |C|` in parameters, tied head counted once.

| core | `(d_c, L_c, h_c, kv_c, F_c)` | `|C|` | `r` |
|---|---|---|---|
| `copy` (no core) | — | 13,011,201 | 1 |
| `c2` | (128, 8, 2, 2, 512) | 5,388,417 | 2.41 |
| `c4` | (128, 4, 2, 2, 512) | 3,218,561 | 4.04 |
| `c8` | (64, 8, 1, 1, 256) | 2,136,385 | 6.09 |
| `c16` | (64, 4, 1, 1, 256) | 1,330,369 | 9.78 |
| `scratch` | — | 0 | ∞ |

The table and the value embeddings (`V × kv d_h` per attention layer) do not shrink with the
width factor, which is why the ratios are below `k_d k_L`: at `c16` the lexicon and the value
embeddings are 60 % of the core. The frontier spans a factor of four in `r`; a deeper cut
(`L_c = 2`) would move the attention layer off its canonical slot and is not a core.

(counts recomputed from the module tree by `tests_palimpsest` T1; the table is written from
that test). `emb_scale` of a core is `sqrt(d_c)`; of an expanded phenotype, `k_d · sqrt(d_c)`
(§1.4), which equals `sqrt(d)` only at `k_d = 1` — it is a property of the *lineage*, carried
in the configuration, exactly as ANLAGE `04 §4.3` requires.

## 1.4 Expression: the function-preserving expansion (development)

**Definition 4 (expansion).** `X(C) → W_0` at the full configuration. Write `tile_r(A, a)`
for `A` repeated `a` times along rows and `tile_c(A, b)` along columns, `k = k_d`,
`k_kv = kv / kv_c`, and `η` the symmetry-breaking noise scale. For every role of a kept
layer (§1.4.1 places layers):

```
   EMB   (V × d_c)          →  tile_c(E, k) / k;      emb_scale_full = k · emb_scale_c                   (4.1)
   Q, OGATE, AGATE (h_c d_h × d_c)  →  tile_r( tile_c(A, k) / k , k )                                    (4.2)
   K, V   attention (kv_c d_h × d_c) →  tile_r( tile_c(A, k) / k , k_kv );   delta (h_c d_h × d_c) → as (4.2)
   O     (d_c × h_c d_h)     →  tile_r( tile_c(A, k) / k , k )                                            (4.3)
   AB    (2 h_c × d_c)       →  tile_r( tile_c(A, k) / k , k )        (interleaved per head, so the block repeats heads)
   CONV_{Q,K,V} (rows × 4)   →  tile_r(A, k)  (attention K/V convs do not exist; delta convs by k)
   VE    (V × kv_c d_h)      →  tile_c(A, k_kv)                                                          (4.4)
   GATE, UP (F_c × d_c)      →  tile_r( tile_c(A, k) / k , k );   DOWN (d_c × F_c) → tile_r( tile_c(A, k) / k , k )
   NORM_MIX, NORM_FFN, NORM_FINAL (d_c) → tile(g, k);   LOGIT_SCALE unchanged
   every matrix              +=  η · rms(A) · N(0, I)                                                     (4.5)
```

**Theorem 1 (function preservation).** At `η = 0` and `k_L = 1`, `X(C)` and `C` produce
identical logits on every input, up to floating-point reduction order.

*Proof.* Induct over the block with the invariant "the residual stream is `[x; …; x]`
(`k` copies)". Input: by (4.1) `x_0' = k · emb_scale_c · [e/k; …; e/k] = emb_scale_c [e; …; e]`.
The gain-free RMS norm is invariant to duplication (`rms([x;…;x]) = rms(x)`), so every
sublayer receives `[ĥ; …; ĥ]`. A matrix `tile_c(A, k)/k` applied to `[ĥ; …; ĥ]` gives `A ĥ`,
and `tile_r` repeats the result: heads and inner units are exact copies; attention head
`g + j h_c` reads KV head `(g + j h_c) mod (k_kv kv_c) ≡ g (mod kv_c)` because `h_c` is a
multiple of `kv_c`, and by (4.4)/(4.2) that KV head is a copy of head `g mod kv_c`'s; so the
duplicated heads compute identical outputs `o' = [o; …; o]`. The output matrices (4.3) sum
`k` identical copies at weight `1/k` and repeat the result over the `k` residual blocks, and
the tiled gains scale each block alike, so the residual write is `[y; …; y]`. The FFN is the
same with inner units in place of heads. At the head, `[x_n; …; x_n] · (tile_c(E, k)/k)ᵀ =
x_n Eᵀ`, and the softcap and the logit scale are unchanged. ∎ (Test T2 asserts it to `10^{-4}`
in float32 on random inputs for every core of §1.3 at `k_L = 1`.)

### 1.4.1 Depth

**[NORMATIVE]** For `k_L > 1` the `L_c` layers are placed at slots `round(i (L − 1)/(L_c − 1))`
in their order with their operators (ANLAGE `06 §6.2` step 3, `anlage/grow.py::layer_placement`);
free slots become attention only as far as the canonical target's count requires, nearest to an
unclaimed canonical attention position, the rest delta (for `L_c = 4 → L = 8` this yields
exactly the canonical `d d d a d d d a`). A new layer copies the nearest existing layer of the
same operator (after width expansion) and sets its two branch gains to `ε_g · g_source`,
`ε_g = 0.02`: its residual writes are `2 %` of the source's, so the function moves by
`O(ε_g)` and the new layer receives a gradient from its first step (ANLAGE Theorem G2's
argument). At `ε_g = 0` the expansion is exact by Theorem 1; the default `0.02` is the price
of a live gradient and is measured (T2 reports the logit deviation at the default).

**The new-unit rule.** `new = "copy"` (the default; the record's B-COPY with `η = 0.01`) or
`new = "fresh"`: the duplicated blocks are replaced by the free initialisation of
`anlage/model.py::free_init` at the innate scale, with their *output* rows (`O`, `DOWN`
columns) zeroed so that the function is still preserved and the fresh units learn from
their first gradient. K-GROW-C compares them.

## 1.5 Consolidation: distillation and dreaming

**Definition 5 (per-position distillation).** For a teacher `Θ` and a student `S` on a
window `w ∈ V^{T+1}`, with `p_Θ(·|w_{≤t})` and `p_S(·|w_{≤t})` the next-token distributions
at positions `t = 1..T`:

```
   D(S ‖ Θ; w)  =  (1/T) Σ_{t=1}^{T} Σ_{v ∈ V} p_Θ(v | w_{≤t}) [ log p_Θ(v | w_{≤t}) − log p_S(v | w_{≤t}) ]       (5.1)
```

computed in float32 from the softcapped logits over chunks of positions so that no
`(B, T, V)` float32 tensor is held at once. Temperature 1 (a registered dial, not swept).

**Definition 6 (dreams).** `Dream(C; n, T_d, τ, seed)` samples `n` sequences of `T_d + 1`
tokens from `C` autoregressively at temperature `τ = 1` by ancestral sampling
(`palimpsest/sample.py::Sampler`, the exact recurrence with a carried state and a key/value
cache, one forward per token). The **start state** is `mixed` by default: even sequences
begin at `EOS` (id 2, the state every short document begins in), odd ones at a token drawn
uniformly from the vocabulary. The EOS-only start (`eos`) is the registered control: a
domain whose documents are long — the Python files average 7,500 tokens, so a training
window rarely contains a document start — is never entered from `EOS` by a model that has
only seen its windows, and its dreams then contain nothing of it (P-P.5, measured: the core
lost code from 2.10 to 5.83 at the next consolidation). Dreams are windows like any other:
`T_d = 256` (`128` in the first wave).

**Definition 7 (consolidation).** After phase `k`, with phenotype `W_k` and previous core
`C_{k−1}` (`C_0 = ∅`):

```
   C_k^{(0)}  =  C_{k−1}  if k > 1 and init = "prev"  (default);   shrink(W_k) otherwise               (7.1)
   B_data     =  N_d tokens of Pool(ε_k, s_k') in the order o_k' (a pool seed and order seed disjoint from training)
   B_dream    =  Dream(C_{k−1}; N_δ / T_d, T_d)        (empty at k = 1 or when λ_δ = 0)
   C_k        =  argmin_C  E_{w ∈ B_data} D(C ‖ W_k; w)  +  λ_δ · E_{w ∈ B_dream} D(C ‖ C_{k−1}; w)      (7.2)
```

by `N_d / (B T)` steps of the recipe of §1.2 at the core's size on the data batches, each step
followed by one step on a dream batch when `λ_δ > 0` (so the two expectations are weighted by
their batch counts; `λ_δ = 1` by default, range `[0, 4]`), warm-up `2 %` and decay `20 %` as
in a phase. Defaults `N_d = 2·10^6`, `N_δ = 5·10^5`. **The data teacher is the phenotype; the
dream teacher is the previous core** — the phenotype has been trained on `ε_k` only since its
birth and may have overwritten the past; the previous core is the store of the past, and on
its own samples it is the only teacher that knows what they should say (P4). Nothing of a
past environment's data is read (T7 asserts it: the consolidation's window digests are all
from `ε_k`'s pool and the dreams).

**Termination.** Every step count is fixed by the token budget; dreams are `T_d` forward
passes; the argmin is `N_d / (B T)` optimiser steps.

## 1.6 The life, in full

```
LIFE(arm, phases (ε_k, s_k, o_k)_{k=1..K}, T_φ, core c, seed)  →  record
   W ← Phenotype(s0, seed)                      # every arm, the same initialisation under CRN
   C ← ∅ ;  R ← ∅  (replay buffer) ;  F ← ∅  (EWC store) ;  Θ_prev ← ∅  (LwF teacher)
   for k = 1 .. K:
       if k > 1:   W ← BIRTH(arm, W, C, seed, k)                                            (§1.7)
       W ← TRAIN(W, Pool(ε_k, s_k), o_k, T_φ, regulariser(arm, R, F, Θ_prev, C))          (§1.2, §1.7)
                 with evaluations of W on Ev(ε_j) for every j ≤ K at 10, 25, 50, 100 % of the phase
       if arm ∈ SIEVE:  C ← CONSOLIDATE(W, C, ε_k)   (Def. 7);   evaluate C on every Ev(ε_j)
       if arm = replay: R ← R ∪ sample(Pool(ε_k), n_R tokens)
       if arm = ewc:    F ← F ∪ {(fisher_diag(W, fitness windows of ε_k), W)}
       if arm = lwf:    Θ_prev ← W (frozen copy)
   return the record: every evaluation, the scores of §1.8, the compute of every stage, the seeds, the digests
```

**BIRTH by arm.**

| arm | `W_{k,0}` |
|---|---|
| `copy`, `replay`, `ewc`, `lwf` | `W_{k−1}` |
| `scratch` | `Phenotype(s0, seed ⊕ k)` |
| `shrink_perturb` | `λ_s W_{k−1} + p_s · N(0, σ_init²)`, `λ_s = 0.6`, `p_s = 0.1` (Ash & Adams 2020, their defaults) |
| `sieve-c`, `sieve-life-c` | `X(C_{k−1})` (Def. 4) |
| `copy-dream` | `W_{k−1}` |
| `trunc-c` | `X(shrink(W_{k−1}))` — the primordium block, no distillation (ANLAGE R8's inheritance) |
| `iid` | one lifetime of `K T_φ` tokens on the mixture of the `K` pools (not a continual arm; the upper bound) |

## 1.7 The controls, exactly

* **`replay`**: a buffer of `n_R` tokens per past environment, stored as windows; each training
  batch replaces a fraction `ρ_R = 0.1` of its rows by buffer windows (uniformly over the
  buffer). `n_R` is set by the **byte rule**: the buffer's bytes at 2 per token equal the
  core's bytes at 4 per parameter — for `c8`, `2.14M × 4 / 2 = 4.27M` tokens over all past
  environments. The genome and the buffer are then the same storage.
* **`ewc`**: `L = CE + (λ_E / 2) Σ_j Σ_i F_{j,i} (w_i − w_{j,i})²` with `F_j` the diagonal
  empirical Fisher of the mean token CE at the end of phase `j` on `32` fitness windows of
  `ε_j` (`bottleneck/loop.py::fisher_diag`; Kirkpatrick et al. 2017 Eq. 3), summed over past
  phases. `λ_E ∈ {10^3, 10^4}` (screened; the better is reported with both).
* **`lwf`**: `L = CE + λ_L · D(W ‖ Θ_prev; w)` on the current batch with `Θ_prev` the frozen
  previous adult (Li & Hoiem 2017), `λ_L = 1`. This is the *unbounded* store (13M
  parameters) against which the core's size is the question.
* **`sieve-nodream-c`**: Def. 7 with `λ_δ = 0` (K-DREAM).
* **`sieve-fresh-c`**: Def. 4 with `new = "fresh"` (K-GROW-C).
* **`sieve-life-c`**: the sieve, and during phase `k > 1` the phenotype rehearses the core's
  dreams: a buffer of `n_R` tokens (the byte rule) of `Dream(C_{k−1})` is generated at the
  switch and `ρ_R` of every batch's rows are drawn from it (as data — the samples *are* the
  core's distribution; several dreams of `T_d` tokens are concatenated to fill a row). No
  data of the past is stored; the rehearsal is generated from the compressed store
  (K-DREAM-LIFE against `sieve-c`, and against `copy-dream`).
* **`copy-dream`**: the copy, rehearsing in the same way from the *previous phenotype's* own
  samples (pseudo-rehearsal, Robins 1995) — the same generative rehearsal with no compression,
  so that `sieve-life − copy-dream` isolates the sieve from the rehearsal.

## 1.8 Scores and accounting

With `CE(M; ε)` the mean cross-entropy of model `M` on `Ev(ε)`, `W_k` the phenotype at the
end of phase `k`, `W_k(q)` at fraction `q` of it, and `E_K = {ε_1, …, ε_K}` as a *set* (an
environment lived twice counts once):

```
   union score       U_K   =  mean_{ε ∈ E_K} CE(W_K; ε)                                    the model at the end of the life
   life score        Λ     =  mean_{k=1..K} mean_{ε ∈ E_k} CE(W_k; ε)                      the area under the life
   retention         R_k(ε) =  CE(W_k; ε) − min_{j<k} CE(W_j; ε)      for ε ∉ {ε_k}          forgetting since ε was last seen
   adaptation        A_k(q) =  CE(W_k(q); ε_k),  q ∈ {0.10, 0.25, 0.50}                      speed on the phase's environment
   the genome's own  G_K   =  mean_{ε ∈ E_K} CE(C_K; ε)                                    what the core carries, as a model
```

**Compute.** Every arm trains the phenotype for `K T_φ` tokens. The sieve arms add, per
consolidation, `N_d` teacher forwards (`2 |W|` FLOP/token) and student steps (`6 |C|`), the
dreams (`N_δ` tokens at `2 |C|` each with the cached sampler), and `N_δ` previous-core forwards; the
controls add their own (`replay`: none beyond its batch; `ewc`: the Fisher probes; `lwf`: one
teacher forward per training token, `2 |W|`). `overhead(arm) = FLOPs beyond the phenotype's
training / the phenotype's training FLOPs`, recorded per life. The **FLOP-matched copy**
(`copy-m`) trains `T_φ (1 + overhead(sieve-c*))` tokens per phase, where `c*` is the frontier's
argmin, and is the strict control of K-SIEVE's margin (second wave).

## 1.9 Symbols and defaults

| symbol | code | default | range | why |
|---|---|---|---|---|
| `T` | `T` | 512 | fixed | the evaluation window of every model here |
| `T_φ` | `tokens_per_phase` | 8·10^6 | 2·10^6–25·10^6 | §1.1 |
| `K` | `len(phases)` | 4 | 2–8 | the shift life alternates twice |
| `n_pool` | `n_windows` | 200,000 | ≥ 10^4 | ANLAGE's pool; 102M tokens, never exhausted at `T_φ` |
| `r`, `c` | `core` | `c8` | §1.3's table | the frontier |
| `k_d, k_L` | `k_d, k_L` | 4, 1 | integers dividing `d/64`, `L` | §1.3 |
| `η` | `noise` | 0.01 | [0, 0.1] | the record's symmetry-breaking scale |
| `ε_g` | `eps_gain` | 0.02 | [0, 1] | §1.4.1 |
| `N_d` | `distil_tokens` | 2·10^6 | ≥ 5·10^5 | one quarter of a phase at the core's cost |
| `N_δ` | `dream_tokens` | 5·10^5 | [0, N_d] | dreams cost `T_d/2` forwards each |
| `T_d` | `T_dream` | 256 | 64–512 | §1.6 cost |
| `λ_δ` | `lam_dream` | 1.0 | [0, 4] | Def. 7 |
| `τ` | `temperature` | 1.0 | fixed | samples from the model's own distribution |
| `init` | `core_init` | `prev` | `{prev, shrink}` | (7.1) |
| `ρ_R`, `n_R` | `replay_frac`, `replay_tokens` | 0.1, byte rule | — | §1.7 |
| `λ_E` | `lam_ewc` | 10^3 / 10^4 | screened | §1.7 |
| `λ_L` | `lam_lwf` | 1.0 | [0.1, 10] | §1.7 |
| `λ_s, p_s` | `sp_lambda, sp_p` | 0.6, 0.1 | Ash & Adams | §1.6 |
| `n_ev` | `n_eval` | 256 | fixed | the v3 sets |

## 1.10 Numerics

Distillation in float32 logits (the softcapped logits of both models are bounded by 15, so
`log_softmax` is stable); the KL is summed over the vocabulary per position in chunks of
`2048` positions. Dreams are sampled under `torch.no_grad` in bf16 autocast with the
categorical draw in float32 from a seeded CPU generator moved per step (bit-reproducible on
one device). The expansion is done in float32 on the CPU generator for `η`. Every shape is
asserted at the module boundary (`gnomon/utils/shapes.py::check_shape`). A non-finite loss or
gradient norm raises. bf16 autocast is not bitwise deterministic across runs (TEMPO
`§P-D.47`); determinism tests run on the CPU.

## 1.11 Acceptance tests (`tests_palimpsest/test_palimpsest.py`)

| # | property | test |
|---|---|---|
| T1 | counts | `|C|` and `r` of every core of §1.3 recomputed from the module tree as equalities |
| T2 | **Theorem 1** | for every core, `X(C)` at `η = 0, ε_g = 0` reproduces `C`'s logits to `10^{-4}` on random tokens (width, kv doubling, depth); at the defaults the deviation is reported and below `0.05` nats of CE on the same tokens |
| T3 | KL | (5.1) on a hand-computed two-position example equals the closed form to `10^{-6}`; the chunked form equals the unchunked |
| T4 | dreams | shapes `(n, T_d + 1)`, first token `EOS`, all ids in `[0, V)`, two draws from one seed identical, different seeds differ |
| T5 | distillation recovers a planted teacher | a core distilled from a *core-sized* teacher on random windows reaches `D < 0.05` in 300 steps (the student can represent the teacher), and a size-mismatched student's residual is reported |
| T6 | births | `copy` returns the same tensors; `scratch` differs; `shrink_perturb` has the stated mean and noise; `sieve` equals `X(C)` |
| T7 | no past data | the consolidation's consumed windows are all from the current pool (digests) plus dreams; a planted past-pool window fails |
| T8 | replay byte rule | `n_R` equals `4|C|/2` for every core; the buffer's fraction per batch is `ρ_R` |
| T9 | EWC | the penalty on a two-parameter example equals the closed form; the Fisher is non-negative and leaves gradients `None` |
| T10 | the life loop | a 3-phase life at a tiny configuration on the CPU runs every arm end to end, records every score of §1.8, and is bit-reproducible from its seed |
| T11 | analysis | the frontier verdict of `02` on planted records: interior minimum → load-bearing; monotone → not; a missing arm → NOT EVALUATED |
