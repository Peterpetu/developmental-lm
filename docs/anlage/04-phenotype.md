# 04 — The phenotype: the strongest conventional hybrid

The network a genome expresses and a lifetime trains. It is deliberately conventional and
deliberately strong: the substrate is the experimental control (SOMA §00.6's principle,
retained), so every difference between ANLAGE and its baselines is attributable to the
genome, the growth or the population, never to the backbone. Nothing here is novel and
nothing here is meant to be.

## 4.1 Operators are functions of activations and weights

**[NORMATIVE]** Every operator is a pure function `y = op(x, W_1, ..., W_k; cfg)`. The module
that owns the weights (`anlage/model.py::Phenotype`) holds them as free `nn.Parameter`s named
by role and layer, and calls the functional operators of `gnomon/pheno/ops.py`
(`delta_mixer`'s recurrence, `rms_norm`, `swiglu`) and `anlage/ops.py` (SDPA attention with
the output gate, the causal short convolution, RoPE, the softcapped head). The vendored
recurrence is used unchanged; `tests_anlage/unit/test_ops.py` re-runs its oracle test.

## 4.2 Roles

**[NORMATIVE]** Exactly 20 roles, `n_M = 5` bits, fixed for the life of the project. Shapes
are `(rows, cols)` with the row axis first; `d = d_s`, `F = F_s`, `h = h_s`, `d_k = d_head`,
`n_kv = n_kv(l)`.

| `m` | name | row axis | column axis | shape | present in |
|---|---|---|---|---|---|
| 0 | `Q` | `HEAD(l)` | `RES` | `(h d_k, d)` | every layer |
| 1 | `K` | `KV(l)` | `RES` | `(n_kv d_k, d)` | every layer |
| 2 | `V` | `KV(l)` | `RES` | `(n_kv d_k, d)` | every layer |
| 3 | `O` | `RES` | `HEAD(l)` | `(d, h d_k)` | every layer |
| 4 | `AB` | `GATE(l)` | `RES` | `(2h, d)` | delta layers |
| 5 | `OGATE` | `HEAD(l)` | `RES` | `(h d_k, d)` | delta layers |
| 6 | `CONV_Q` | `HEAD(l)` | `TAP` | `(h d_k, 4)` | delta layers |
| 7 | `CONV_K` | `KV(l)` | `TAP` | `(n_kv d_k, 4)` | delta layers |
| 8 | `CONV_V` | `KV(l)` | `TAP` | `(n_kv d_k, 4)` | delta layers |
| 9 | `AGATE` | `HEAD(l)` | `RES` | `(h d_k, d)` | attention layers |
| 10 | `VE` | `TOK` | `KV(l)` | `(V, n_kv d_k)` | attention layers, if `use_ve` |
| 11 | `GATE` | `INNER(l)` | `RES` | `(F, d)` | every layer |
| 12 | `UP` | `INNER(l)` | `RES` | `(F, d)` | every layer |
| 13 | `DOWN` | `RES` | `INNER(l)` | `(d, F)` | every layer |
| 14 | `EMB` | `TOK` | `RES` | `(V, d)` | global |
| 15 | `HEAD_OUT` | `TOK` | `RES` | `(V, d)` | global, only after untying |
| 16 | `NORM_MIX` | `RES` | — | `(d,)` | every layer |
| 17 | `NORM_FFN` | `RES` | — | `(d,)` | every layer |
| 18 | `NORM_FINAL` | `RES` | — | `(d,)` | global |
| 19 | `LOGIT_SCALE` | `ONE` | — | `(1,)` | global |

Roles 0–15 are matrix roles (`|M_mat| = 16`); 16–19 are vector/scalar roles (`|M_vec| = 4`).

**[NORMATIVE — KV grouping]** Attention head `g` reads KV head `g mod n_kv`. This is
`repeat`, not `repeat_interleave`, and the choice is load-bearing for §06: when heads are
appended, an existing head keeps its KV source. `n_kv(l) = h_s` for delta layers and a fixed
`n_kv_attn` per lineage for attention layers (§4.5), so that the grouping never changes an
existing head's source across stages.

## 4.3 The block

**[NORMATIVE]** With `RMSNorm_gf` the gain-free norm of `gnomon/pheno/rmsnorm.py`
(reduction at ≥ float32) and `⊙` elementwise:

```
    x_0[t]   =  sqrt(d_0) E[tok_t]         d_0 = the primordium's width, a lineage constant     (4.1)
    for l = 0 .. L-1:
        h    =  RMSNorm_gf(x)
        y    =  Mix_l(h)                                    (4.3) attention or (4.4) delta
        x    =  x + g^mix_l ⊙ RMSNorm_gf(y)                 Peri-LN; g^mix_l = W_{l,NORM_MIX}
        h    =  RMSNorm_gf(x)
        f    =  W_DOWN ( SiLU(W_GATE h) ⊙ (W_UP h) )        SwiGLU
        x    =  x + g^ffn_l ⊙ RMSNorm_gf(f)                                              (4.2)
    z        =  ( RMSNorm_gf(x) ⊙ g^final ) E^T             tied;  W_HEAD_OUT^T after untying
    logits   =  c · tanh( exp(lambda) z / c ),     c = 15,  lambda = W_LOGIT_SCALE
    L        =  CE(logits, targets)  +  1e-4 · (log sum_v exp logits_v)^2                (4.5)
```

**[NORMATIVE — why each piece]** (4.1)'s read scaling and the unscaled head follow SOMA
§05.6's measured 43× gradient discontinuity at block 0 without it. The scale is `sqrt(d_0)`
of the *primordium*, held fixed across stages: a scale that moved with `d_s` would change
the ratio of the embedding to every branch write at a growth event and break Theorem G1 of
§06 (found by test G1; at generation 0 it equals `sqrt(d)` exactly). Peri-LN (normalise
into and out of every sublayer) is the arrangement least sensitive to weight scale, which
matters here because decoded and grown weights are structured rather than i.i.d. The branch
gains `g^mix_l, g^ffn_l` carry the depth scaling (§4.7) so that no forward-pass constant
depends on `L`, which is what lets depth grow without a discontinuity in the existing layers.
The softcap and z-loss are the speedrun's and PaLM's; their measured gains are in §10.

## 4.4 The two mixers

**[NORMATIVE — attention, (4.3)]** Layers with `l mod 4 == 3`.

```
    q = RoPE( RMSNorm_head_gf( W_Q h ) ),    k = RoPE( RMSNorm_head_gf( W_K h ) ),    v = W_V h  (+ VE[tok] if use_ve)
    a = SDPA_causal( q, k[g mod n_kv], v[g mod n_kv] )                 per head g, scale d_k^{-1/2}
    y = W_O ( a ⊙ sigmoid( W_AGATE h + g_0 ) )                            g_0 = 2 (innate)
```

RoPE with `theta = 10^4` over all `d_k` dimensions. The QK-norm is gain-free (the bound
`|s| ≤ sqrt(d_k)` requires it; SOMA §05.3). The sigmoid output gate is the 2025 "gated
attention" mechanism (§10 E-12): it removes the need for a sink column, tolerates a larger
learning rate, and lets the phenotype use `torch.nn.functional.scaled_dot_product_attention`
directly. `gnomon/pheno/ops.py::attention` with `sink = -10^4` (a column of zero mass) is the
float64 oracle for the SDPA path (P2).

**[NORMATIVE — gated delta rule, (4.4)]** Layers with `l mod 4 != 3`.

```
    q_r, k_r, v_r  =  W_Q h,  W_K h,  W_V h
    q  =  L2norm_head( SiLU( conv_4(q_r; W_CONV_Q) ) ),     k  =  L2norm_head( SiLU( conv_4(k_r; W_CONV_K) ) )
    v  =  SiLU( conv_4(v_r; W_CONV_V) )
    (a_r, b_r) = deinterleave( W_AB h )     row 2g is head g's decay gate, row 2g+1 its write gate, each (h,)
    alpha  =  sigmoid( a_r + a_0 ),        a_0 = 4  (innate)
    beta   =  min( 2 - 1e-3, 2 sigmoid( b_r ) )
    o  =  GDR_chunk( q, k, v, alpha, beta )                             SOMA (5.6)–(5.16), chunk 64
    o  =  RMSNorm_head_gf(o) ⊙ SiLU( W_OGATE h )
    y  =  W_O o
    conv_4(x; W)[t, c]  =  sum_{tau=0}^{3} W[c, tau] x[t - tau, c]        causal, depthwise
```

**[NORMATIVE — the `GATE(l)` axis is interleaved per head]** Row `2g` of `W_AB` is head
`g`'s decay gate and row `2g+1` its write gate. A block layout (all decay rows, then all
write rows) breaks prefix nesting (§02.2 P1) the moment a head is appended — an existing
head's write row would land on a new head's decay row — and G1 of §06.7 catches it.

`a_0 = 4` places the decay at `alpha ≈ 0.98` at initialisation (a half-life of ~35 tokens)
rather than at `0.5`; there are no biases in the model, so the offsets `a_0, g_0` are innate
constants and are not parameters. The short convolution and the SiLU output gate are the two
components the Gated DeltaNet ablation prices at +1.60 and +1.77 perplexity (§10 E-8), which
is why GNOMON's decision to drop them is reversed here. `L2norm_head` is what makes the
transition a scaled Householder and (5.5)'s eigenvalue statement true.

## 4.5 Configurations

**[NORMATIVE]** Stages of the two lineages. `d_head = 64` everywhere; `n_kv_attn` is fixed per
lineage; attention at `l mod 4 == 3`; `T = 512`.

| lineage | stage | `d` | `L` | `h` | `n_kv_attn` | `F` | attention layers |
|---|---|---|---|---|---|---|---|
| TinyStories (`V = 8192`) | 0 | 256 | 8 | 4 | 2 | 1024 | 2 |
| | 1 | 384 | 12 | 6 | 2 | 1536 | 3 |
| | 2 (adult) | 512 | 16 | 8 | 2 | 2048 | 4 |
| Pile (`V = 16384`) | 0 | 512 | 12 | 8 | 4 | 2048 | 3 |
| | 1 | 768 | 18 | 12 | 4 | 3072 | 4 |
| | 2 (adult) | 1024 | 24 | 16 | 4 | 4096 | 6 |

**[NORMATIVE — parameter counts are computed, never restated]** `Phenotype.n_params()`
enumerates `named_parameters()`; `tests_anlage/unit/test_spec_numbers.py` asserts the counts
below as equalities recomputed from the module tree (`G-SPECNUM`), and this table is written
from that test's output. A mismatch is a specification defect.

| configuration | parameters (tied `EMB`) | non-embedding `N` | `|Z|` (§02.6) | `|Gamma| : |W|` | `(|Gamma| + |Z|) : |W|` |
|---|---|---|---|---|---|
| TinyStories s0 | 13,011,201 | 10,914,049 | 300,096 | 1 : 31 | 1 : 18 |
| TinyStories s1 | 35,875,201 | 32,729,473 | 508,256 | 1 : 86 | 1 : 39 |
| TinyStories s2 (adult) | 78,307,841 | 74,113,537 | 798,848 | 1 : 187 | 1 : 64 |
| Pile s0 | 73,804,289 | 65,415,681 | 765,024 | 1 : 176 | 1 : 62 |
| Pile s1 | 207,116,033 | 194,533,121 | 1,387,408 | 1 : 495 | 1 : 115 |
| Pile s2 (adult) | 461,186,049 | 444,408,833 | 2,257,088 | 1 : 1,103 | 1 : 172 |

`|Gamma| = 418,288` at every row (§03.6). The ratios grow with the phenotype because the
genome does not; the latent bank grows with the number of units (`d_z |A_s| + d_y L_s`), which
is why both ratios are reported (A2). The plan's "≈ 350M" for the Pile adult was an estimate;
the module tree says 461M, and at 16 GB that stage trains at ~8K tokens per micro-step with
gradient accumulation (§4.8).

## 4.6 The recipe

**[NORMATIVE — optimiser groups, by role name]**

| group | roles | optimiser |
|---|---|---|
| **M** | `Q, K, V, O, AB, OGATE, AGATE, GATE, UP, DOWN` | Muon (`gnomon/optim/muon.py`: Newton–Schulz 5, Nesterov 0.95, step `0.2 sqrt(max(m,n))`), weight decay 0.1 |
| **A** | `EMB, HEAD_OUT, VE, CONV_*, NORM_*, LOGIT_SCALE` | AdamW `(0.9, 0.95)`, `eps 1e-8`, weight decay 0 |

**[NORMATIVE — learning rates carry provenance]** `eta_M`, `eta_A` are swept at stage 0
(§08 R2) and each configuration records `lr_provenance` beginning with `measured:` or
`NOT MEASURED` (GNOMON `AUDIT-v1.1 C-1`'s rule). Initial grid: `eta_M in {0.01, 0.02, 0.03,
0.05}`, `eta_A = 3e-3` for embeddings, `eta_A = 1e-3` for gains and scalars.

**[NORMATIVE — schedule]**

```
    lr(n)   =  lr_max · n / n_warm                       n <  n_warm = 0.02 N
            =  lr_max                                    n_warm ≤ n < n_dec = 0.8 N
            =  lr_max · (N - n) / (N - n_dec)            n ≥ n_dec          (linear decay to zero)
    B(n)    =  B_final · clip( 0.25 + 0.75 n / (0.1 N), 0.25, 1 )   rounded to a multiple of 8 sequences
```

Decay-to-zero over the last 20% follows the cooldown and D2Z results (§10 E-13); the
batch-size warm-up follows the critical-batch-size result (§10 E-14). Gradient clipping at
global norm 1.0. On a growth event the schedule of the new stage restarts its warm-up and
batch warm-up over `n_ramp` (§06.3).

**[NORMATIVE — precision and compilation]** Parameters float32, forward under bfloat16
autocast, Newton–Schulz and every normalisation reduction at ≥ float32, the delta-rule
triangular solves at ≥ float32 (the vendored `_no_autocast`), cross-entropy in float32 over
chunks of 4,096 positions so the logits never materialise at `B T V` in float32.
`torch.compile(mode="default")` when available, eager otherwise; the backend is recorded in
the manifest. Attention uses SDPA; a FlexAttention path is optional and gated by a runtime
check.

## 4.7 Free-weight initialisation (generation 0; the "random" control)

**[NORMATIVE]**

| role | initialisation |
|---|---|
| `Q, K, V, O, AB, OGATE, AGATE, GATE, UP, DOWN, HEAD_OUT` | `N(0, 1/fan_in)` |
| `EMB` | `N(0, 1/d)` |
| `VE` | `N(0, 1/(n_kv d_k))` |
| `CONV_*` | tap 0 = `1`, taps 1–3 = `0` (the identity convolution) |
| `NORM_MIX, NORM_FFN` | `(2 L_0)^{-1/2}` |
| `NORM_FINAL` | `1` |
| `LOGIT_SCALE` | `0` |

drawn through an explicit generator seeded from `derive_seed(run_seed, "init")`. The branch
gains at `(2L_0)^{-1/2}` reproduce SOMA/GNOMON's fixed `1/sqrt(2L)` branch scale at
generation 0 as a *learned* quantity, which is what lets §06 grow depth without a forward-pass
constant changing.

## 4.8 Throughput and memory

**[NORMATIVE]** Every efficiency claim carries measured tokens/s (median of per-step CUDA
events, with IQR — `screen/harness.py`'s rule, because this GPU is shared with a desktop) and
peak memory. Targets on the RTX 4080 Super, asserted by `scripts/anlage_baseline.py --profile`
rather than assumed: TinyStories adult ≥ 40k tok/s; Pile adult ≥ 18k tok/s at `B_tok = 8k`
per micro-step with accumulation to `B_final`. A growth or regression event costs at most
`1%` of the tokens between two events (D10).

## 4.9 The baselines

**[NORMATIVE]** Three, all the same code path:

| id | definition |
|---|---|
| **B-FLAT** | the adult configuration trained from §4.7 for the whole budget `C_total` (FLOPs) |
| **B-COPY** | the stage schedule `omega` with copy-based growth (§06.3: HyperCloning width, G_stack depth, symmetry-breaking noise) from §4.7 |
| **B-WARM** | generation `g+1` started from generation `g`'s adult weights directly (`eps = 1`, §05.4) |

## 4.10 Acceptance tests

**[NORMATIVE]** `anlage/model.py` and `anlage/ops.py` are accepted only when all of the
following hold.

| # | property | test |
|---|---|---|
| P1 | shapes | every role has the shape of §4.2 at every stage of both lineages, asserted at the module boundary |
| P2 | attention oracle | the SDPA path with the sigmoid gate fixed at 1 matches `gnomon/pheno/ops.py::attention` with `sink = -10^4` to `1e-5` in float32 and `1e-10` in float64, with and without GQA, with RoPE applied to both |
| P3 | delta oracle | the delta path matches `gated_delta_rule_reference` to `1e-10` in float64 at `T in {1, 31, 64, 65, 512}` |
| P4 | causality | a token's logits are invariant to every future token, on the production forward path, mutation-verified by breaking the conv's left pad (positive control) |
| P5 | KV grouping | head `g` reads KV head `g mod n_kv`; a `repeat_interleave` mutant fails |
| P6 | initialisation | realised RMS of every role after §4.7 is within 10% of its target; the residual stream enters block 0 at RMS `1.0 ± 0.1` |
| P7 | softcap and z-loss | `|logits| < 15` always; the z-loss term equals `1e-4 (log Z)^2` on a hand-computed batch |
| P8 | optimiser groups | group assignment by name is exhaustive and disjoint; `AB` and `CONV_*` are in the groups the table says |
| P9 | schedule | `lr(n)` and `B(n)` at the boundaries `n_warm, n_dec, N` and at `0.1 N`; the decay reaches exactly zero at `N` |
| P10 | chunked CE | the chunked cross-entropy equals the unchunked one to `1e-6` on a batch whose size is not a multiple of the chunk |
| P11 | determinism | two forward passes of one `(W, batch)` are bit-identical; two lifetimes with one seed give identical curves |
| P12 | counts | §4.5's table recomputed from `named_parameters()` as equalities |
| P13 | no free tensor outside the roles | the model's parameters are exactly the roles present at its stage, and nothing else |
