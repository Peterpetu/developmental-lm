# 03 — The backbone: one span step, every shape

**Status: normative.** The backbone `F` of 01 (1.2): the configurations (§3.1), the residual
block with gains and switches (§3.2), the delta operator (§3.3–3.4), the attention operator
(§3.5), the FFN (§3.6), input and output (§3.7), the gain fold and the active views (§3.8),
parameter and FLOP accounting (§3.9), and the acceptance tests (§3.10). Everything below is the
TS-adult unless a table gives others. Reused operators: `anlage/ops.py` (`delta_mixer`,
`attention_gated`, `swiglu`, `causal_conv4`, `rms_norm_gf`, `softcap_logits`),
`gnomon/pheno/mixer_chunk.py` (the gated delta rule), `gnomon/pheno/rmsnorm.py`.

## 3.1 Configurations

**[NORMATIVE]**

| name | `d` | `L` | operators | `h` | `h_kv` | `d_head` | `F` | slices | `d_x` | `K` |
|---|---|---|---|---|---|---|---|---|---|---|
| TS-primordium (active at birth) | 512 | 8 of 16: `{0,1,2,3,8,9,10,11}` | `d d d a` twice (attention at 3 and 11) | 4 of 8 | 2 | 64 | 1024 of 2048 | 4 of 8 | 256 | 4 |
| TS-adult (provisioned) | 512 | 16 | `d d d a` × 4 (attention at `l mod 4 = 3`) | 8 | 2 | 64 | 2048 | 8 × 256 | 256 | 4 |
| Pile-adult (provisioned) | 1024 | 32 | `d d d a` × 8 | 16 | 4 | 64 | 4096 | 16 × 256 | 512 | 4 |

The *provisioned* shape is allocated at birth; the *active* shape is the set of blocks whose
switches are on (05). The residual width `d` is provisioned; heads, FFN slices and layers are
switched. `h_kv` and `d_head` are lineage constants (ANLAGE 04 §4.2, G6). Windows: `T = 512`
tokens = `T_s = 128` spans; batch `B = 32`; chunk size 64 spans for the delta rule.

## 3.2 The residual block

**[NORMATIVE]** The stream enters as `h^{(0)}_t = W_in x_t + b_in` (`W_in: d × d_x`, `b_in: d`).
Layer `l = 0..L−1` with operator `Op_l ∈ {delta, attn}`:

```
   ĥ  = ρ_l^{mix} ⊙ h / RMS(h)                                   pre-norm with gain (a vector role)
   y  = Op_l(ĥ);   h ← h + m_l · ( n_l^{mix} ⊙ y / RMS(y) )       post-norm (gain-free), then the branch gain n,
                                                                 init (2L)^{-1/2}: Peri-LN as in ANLAGE 04 §4.3   (3.1)
   ĥ  = ρ_l^{ffn} ⊙ h / RMS(h)
   f  = FFN_l(ĥ);  h ← h + m_l · ( n_l^{ffn} ⊙ f / RMS(f) )                                           (3.2)
```

`m_l = σ(a_l)` is the layer's switch (05), `ρ, n ∈ R^d` the norm vector roles carried by the
weights and modulated by the genome (04) and, for `n`, by `ζ` (07); `ρ_W = 1` and the post-norm
is gain-free, so at `γ = 0` the block is exactly ANLAGE's `Phenotype` block (P6):

```
   ρ_l = ρ_{W,l} ⊙ (1 + γ_{ρ,l}),      n_l = n_{W,l} ⊙ (1 + γ_{n,l}) · exp(ζ_l)                       (3.3)
```

RMS is over the `d` coordinates with `ε = 10^{-6}`; all norms in fp32.

## 3.3 The delta operator: projections

**[NORMATIVE]** For a delta layer with `h` heads of `d_k = d_v = d_head`:

```
   q̃ = W_q ĥ,   k̃ = W_k ĥ,   ṽ = W_v ĥ                    W_q, W_k, W_v: (h d_head × d) = (512 × 512)
   q = SiLU(Conv4(q̃)),  k = SiLU(Conv4(k̃)),  v = SiLU(Conv4(ṽ))   depthwise causal conv over spans,
                                                              taps W_cq, W_ck, W_cv: (h d_head × 4)     (3.4)
   per head j:  k_j ← k_j / ‖k_j‖,   q_j ← q_j / ‖q_j‖                     unit keys and queries (SOMA (5.7); anlage/ops.py)
   [α̃; β̃] = W_ab ĥ                                          W_ab: (2h × d), rows interleaved per head
   α_{t,j} = σ( α̃_{t,j} + a_{0,j} ),   β_{t,j} = min( 2 σ( β̃_{t,j} + g_{0,j} ), β_max ),  β_max = 1.999   (3.5)
```

`a_{0,j}, g_{0,j}` are the decay and write offsets (initial values `4` and `0`; genome roles
04). The interleaved row layout (decay row `2j`, write row `2j+1`) is the nesting-safe layout
ANLAGE's G1 required.

## 3.4 The delta operator: the fast state

**[NORMATIVE]** Per head, with `S ∈ R^{d_v × d_k}`:

```
   S_t = S_{t−1} α_t ( I − β_t k_t k_tᵀ ) + β_t v_t k_tᵀ                                            (3.6)
   o_t = S_t q_t                                                                                   (3.7)
   Op(ĥ)_t = W_o · ( RMSNorm_headwise(o_t) ⊙ SiLU( W_og ĥ_t + g_og ) )     W_o: (d × h d_head), W_og: (h d_head × d),
                                                              g_og = 0 for delta layers (a SiLU gate, anlage/ops.py)   (3.8)
```

(3.6) is exactly one gradient step on `½‖S k_t − v_t‖²` with step `β_t` and decay `α_t`
(SOMA (5.4)); computed chunkwise by `gated_delta_rule_chunk(q, k, v, alpha, beta,
chunk_size=64, initial_state=S_{start})`, which returns the final state; **`S` persists across
steps and episodes** and is reset only at a declared boundary of the raw stream (01 A3). The
solve dtype is fp32 (the module's floor); the output is cast to the compute dtype.

## 3.5 The attention operator

**[NORMATIVE]** For an attention layer (every fourth):

```
   q = W_q ĥ  (h d_head × d),   k = W_k ĥ,  v = W_v ĥ  (h_kv d_head × d)
   per head: q_j, k_{j mod h_kv} ← RMSNorm_headwise (gain-free);  RoPE on the span index
   o = SDPA_causal( q, repeat(k), repeat(v) ),  scale 1/sqrt(d_head)      head j reads KV head j mod h_kv
   Op(ĥ) = W_o · ( o ⊙ σ( W_og ĥ + g_og ) ),   g_og = 2 for attention layers (a sigmoid gate)        (3.9)
```

`W_o: (d × h d_head)`, `W_og: (h d_head × d)`. At inference the keys and values of past spans
are cached per layer (the KV cache), so a span step costs one query against the cache. The
cache is **window-local** (08 §8.3): a cache that already holds `T_s` spans starts a new
window rather than growing, and a call whose offset plus new spans would exceed `T_s` raises.
Without that rule the RoPE positions of the new spans continue past `T_s` while the training
path always starts at zero, so a cached inference path and the training path would compute
different functions of the same window (11 §P-C.26).

## 3.6 The FFN

**[NORMATIVE]**

```
   FFN(ĥ) = W_down · ( SiLU( W_gate ĥ ) ⊙ ( W_up ĥ ) )          W_gate, W_up: (F × d), W_down: (d × F)   (3.10)
```

The `F` inner units are grouped into `F/256` slices; a slice's rows of `W_gate, W_up` and
columns of `W_down` are a block (05).

## 3.7 Output

**[NORMATIVE]**

```
   ŷ_{t+1} = W_out · ( ρ_fin ⊙ h^{(L)}_t / RMS(h^{(L)}_t) )        W_out: (d_x × d), ρ_fin: d           (3.11)
```

No vocabulary, no softmax: every decoder (02) is downstream of `ŷ`.

## 3.8 The gain fold and the active views

**[NORMATIVE]** For every matrix role `W_{(l,m)}: (n_out × n_in)` the genome supplies input and
output gain vectors `a_in ∈ R^{n_in}`, `a_out ∈ R^{n_out}` (04). The backbone computes with

```
   W_eff = diag(1 + a_out) · W · diag(1 + a_in)                                                  (3.12)

The switch multiplier `m_b` of 05 §5.2 multiplies a **head's** or a **slice's** contribution
*before* the output projection (`W_o` for heads, `W_down` for slices) and a **layer's** whole
branch after it, exactly as (3.8), (3.9) and the FFN write it; it is not folded into `W_eff`,
because `W_eff` is shared by every head of a layer while `m_b` is per block.
```

computed in-graph at every step from the current `W` and the current gains (one elementwise
product per matrix per step, `< 0.1 %` of the matrix's GEMM); autograd delivers
`∇_W L = diag(1 + a_out) ∇_{W_eff} L diag(1 + a_in)` and, for the differentiable gain roles
that are free within the episode (04 §4.5), `∇_a L`. The *expression* of the genome into the
gains' starting values happens once per episode (04 §4.5); nothing of the genome is evaluated
per step. Because `1 + a > 0` (04 §4.3), the fold is invertible and `W_eff` at `a = 0` is `W`
exactly (G-IDENT).

**Active views.** At the start of every episode, for each layer the active heads
`J_l ⊆ {0..h−1}` and active slices `Σ_l ⊆ {0..F/256−1}` are gathered into contiguous views:
`W_q^{act} = W_q[rows of J_l]`, `W_o^{act} = W_o[:, cols of J_l]`, `W_ab^{act}` (the two rows
of each active head), `W_og^{act}`, `W_{cq,ck,cv}^{act}`; `W_gate^{act}, W_up^{act}` (rows of
`Σ_l`), `W_down^{act}` (columns of `Σ_l`). A layer with `m_l < ε_skip` is not executed at all.
The switch value `m_b` of an *active* block multiplies its branch as in 05 §5.1. A skipped
block contributes exactly zero, which equals the limit `m_b → 0` (Theorem G1, 05 §5.4).

## 3.9 Parameter and FLOP accounting

**[NORMATIVE]** Counts are formulas; the exact integers are asserted by `test_spec_numbers`
(G-SPECNUM) from the module tree, never restated by hand.

| block | parameters | TS-adult, provisioned |
|---|---|---|
| delta layer, mixer | `3 d h d_head` (q, k, v) `+ 3 · 4 h d_head` (taps) `+ 2 h d` (ab) `+ h d_head d` (og) `+ d h d_head` (o) `+ 2 h` (`a_0, g_0`) `+ 1` (`g_og`) | `1,325,073` |
| attention layer, mixer | `d h d_head` (q) `+ 2 d h_kv d_head` (k, v) `+ h d_head d` (og) `+ d h d_head` (o) `+ 1` (`g_og`) | `917,505` |
| FFN, every layer | `3 d F` | `3,145,728` |
| norms, every layer | `4 d` (`ρ^{mix}, ρ^{ffn}, n^{mix}, n^{ffn}`) | `2,048` |
| in / out | `d d_x + d` (`W_in, b_in`) `+ d_x d + d` (`W_out, ρ_fin`) | `263,168` |
| **delta layer** | | `4,472,849` |
| **attention layer** | | `4,065,281` |
| **total provisioned** | `12 × 4,472,849 + 4 × 4,065,281 + 263,168` | **`70,198,480`** (asserted by G-SPECNUM, `test_backbone.py`) |
| **active at birth** | 6 delta + 2 attention layers at 4 heads and 4 slices | **`17,886,264`** (asserted) |
| Pile-adult provisioned / at birth | the same formulas at `d = 1024, L = 32, h = 16, h_kv = 4, F = 4096, d_x = 512` | formula value, asserted by G-SPECNUM against the module tree |

The plan's "78.3M" was ANLAGE's count including a tied token embedding; TEMPO's backbone has
no embedding table. The first draft of this table (`70,198,592`) mis-added the in/out row;
the tree is the authority (G-SPECNUM).

**FLOPs per span** (training ≈ `6 ×` the multiply-adds of the active matrices):
`F_span = 6 · N_active^{mat}`; the attention operator adds `4 · h d_head · t` for a span at
position `t` in its window; the delta rule adds `O(h d_head²)` per span. `F_episode = Σ_spans
F_span` is the quantity the reward (06 §6.3) normalises by; it is computed exactly from the
active shape by `tempo/morphology.py::flops_per_span`.

**Throughput reference (measured, ANLAGE, this card, eager):** 13M 69,020 tok/s at 8.35 GB;
36M 33,271 at 9.0 GB; 78M 15,118 at 8.5 GB (`T = 512`, batch 32). TEMPO runs one backbone
step per `K = 4` tokens, so its per-token throughput at a given active size is measured in R1
and recorded against these.

## 3.10 Acceptance tests

**[NORMATIVE]** `tempo/backbone.py` is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| P1 | shapes at every configuration | every parameter of §3.9 has its declared shape at TS-primordium, TS-adult and Pile-adult; the tree's total equals the formula (G-SPECNUM) |
| P2 | delta oracle | (3.6)–(3.7) computed chunkwise equal the float64 reference (`gnomon/pheno/mixer_ref.py`) to `1e-10` on random inputs, with and without an initial state |
| P3 | streaming equivalence | one window processed as two consecutive halves with `S` carried equals the whole window to `1e-5` (bf16 off) |
| P4 | KV cache equivalence | span-by-span attention with the cache equals full-window attention to `1e-5` *within a window*; a full cache starts a new window (the positions restart) and an over-long call raises |
| P5 | GQA and RoPE | head `j` reads KV head `j mod h_kv`; RoPE on the span index reproduces `anlage/ops.rope` |
| P6 | every gain is live | with `γ = 0` the backbone equals the plain backbone bit for bit; perturbing **any one** role's gains by a *non-uniform* random vector changes the output. The perturbation must not be uniform: under Peri-LN a uniform rescaling of a norm gain is absorbed by the normalisation that follows it, so a uniform test would report a live gain as dead (04 §4.5) |
| P7 | fold | `W_eff` at `a = 0` equals `W` bit for bit; the gradient of the loss w.r.t. `W` through the fold equals `diag(1+a_out) ∇_{W_eff} diag(1+a_in)` by finite differences at `1e-4` |
| P8 | skipping | a layer with `m_l < ε_skip` and the same layer computed with `m_l = 0` give bit-identical outputs; an inactive head/slice gathered out gives outputs equal to `m_b = 0` to `1e-6` |
| P9 | active views | after a switch change the views are rebuilt and the parameter tensors are untouched (their ids and values); gradients flow into the correct rows/columns of the full tensors |
| P10 | causality | changing span `t+1` of the input does not change `ŷ_{t+1}` (the output at `t`) in either operator |
| P11 | numerics | no NaN/Inf over 200 steps of bf16 training on random data with the softcap and clipping in place; `S` accumulates in fp32 |
| P12 | throughput and memory | R1 records spans/s, tokens/s and peak memory at the primordium and the adult with the median-of-CUDA-events rule; the numbers appear in the register |
