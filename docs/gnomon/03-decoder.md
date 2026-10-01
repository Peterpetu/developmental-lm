# 03 — The decoder

One function decodes every weight in the network. This section specifies it completely:
equations, dimensions, initialisation, genome size, cost, and the acceptance tests that
decide whether the implementation is correct.

## 3.1 The trunk

**[NORMATIVE]** For a weight matrix of role `m` in layer `l`, entry `(i, j)`:

```
    u_row(i)     =  W_row [ b_i ; z_i ]                          in R^{d_h}
    u_col(j)     =  W_col [ b_j ; +1 ; z_j ]                     in R^{d_h}
    u_ctx(l, m)  =  W_ctx [ r_l ; t_m ; y_l ]                    in R^{d_h}

    h1(i,j,l,m)  =  tanh( s * ( u_row(i) + u_col(j) + u_ctx(l,m) ) + c1 )
    h2(i,j,l,m)  =  tanh( W_2 h1 + c2 )
    raw[i,j]     =  w_out . h2 + o_0
```

with `s = exp(log_gain)`. For the **token** axis (the embedding and the tied output head)
`W_tok [ e_v ; zeta_v ]` replaces `W_row`, since the token code is `n_v` bits wide rather
than `n_c`.

**[NORMATIVE — two hidden layers, not one, *and* a gain above unity.]** The
paper's g-networks are `(20-30-10-1)` and `(20-10-5-1)`: two hidden layers
between the identity pair and the weight. One layer is not a simplification, it
is a different model. At small `s`, `tanh` is near-linear, a single layer makes
the decoder **additive** in its arguments, and `raw[i,j] ~ alpha_i + beta_j` — a
rank-2 weight matrix.

Measured here (`erank` of the decoded `Q` matrix as a fraction of `min(d_out,
d_in)`, mean of 3 seeds; the SOMA measurement that motivated this was 6.76 of
192 = **3.5%**):

| | pico `d=128` | nano `d=192` | M1 `d=512` | `erank ~ d^alpha` |
|---|---|---|---|---|
| 1 hidden layer, `s = 1` | 0.106 | 0.083 | **0.031** | 0.41 |
| 2 hidden layers, `s = 1` | 0.177 | 0.123 | 0.083 | 0.60 |
| 1 hidden layer, `s = 4` | 0.282 | 0.249 | 0.124 | 0.41 |
| **2 hidden layers, `s = 4`** | **0.377** | **0.314** | **0.260** | **0.73** |

**[NORMATIVE — the correct reading of this table, which is not the obvious one.]**
The gain is the *larger* of the two effects and the second layer is the
*qualitative* one. Neither alone is sufficient: at `s = 4` a single layer still
reaches only 12.4% at M1, and at `s = 1` two layers reach only 8.3%. What
distinguishes them is the **scaling exponent**: a single-layer decoder's rank
grows as `d^0.41` regardless of gain — 13.6 at `d = 128` and 16.0 at `d = 512`,
a factor of 1.2 against a fourfold width — while the two-layer decoder at
`s = 4` grows as `d^0.73`. A rank that barely moves with width is a decoder of
fixed complexity, which is the degeneracy; a rank that grows sublinearly is a
low-rank prior, which is the compression working. §3.8's D5b tests the exponent
and D6 uses the single-layer decoder as its positive control.

**[NORMATIVE — the floor is 0.20, not 0.25.]** An earlier draft of this section
set `0.25 * min(d)` as an absolute floor at every scale. That is the wrong shape
of criterion: because rank grows as `d^0.73`, any fixed fraction must eventually
fail at sufficient width for a decoder that is working exactly as designed. The
gate is therefore `0.20 * min(d)` *and* an exponent above `0.6`, which the
single-layer control fails by construction.

**[NORMATIVE — the first layer is factorised, and this is what makes it affordable.]**
`u_row`, `u_col` and `u_ctx` are computed once per matrix at cost `O((d_out + d_in) d_h)`
and broadcast; only layers 2 onward touch the `d_out x d_in x d_h` tensor. This is also
the paper's own form (`W_1g b_g + W_1c b_i`).

## 3.2 Output normalisation, and why it is not a cheat

**[NORMATIVE]** The decoded matrix is

```
    W_{l,m}  =  ( raw / rms(raw) ) * sigma_{l,m} * s_target(l,m)

    rms(raw)          =  sqrt( mean_{i,j} raw[i,j]^2 + 1e-12 )
    sigma_{l,m}       =  exp( clamp( w_sig . tanh(u_ctx(l,m)) + sig_0 , -3, +3 ) )
    s_target(l,m)     =  sqrt( 2 / (fan_in + fan_out) )        # innate: a function of shape
```

Three reasons, in order of importance:

1. **Every genome must decode to a trainable network.** An evolutionary search whose
   mutants decode to networks with exploding or vanishing weights spends its entire budget
   on dead individuals. Per-matrix RMS normalisation makes the decoded scale independent
   of `Gamma`, so a mutation can change the *pattern* of the weights but can never make
   the network untrainable by scale alone. This is a property of the search, not a
   convenience.
2. It is the structural half of the paper's own finding that p-networks decoded from
   random g-networks are *"impossible to train"* because of "exponential divergence of
   initialization errors from layer to layer". §05.5's `eps(n)` annealing is the
   developmental half; GNOMON uses both.
3. `s_target` is **innate** — a pure function of the matrix shape, costing nothing — so
   dimensional correctness is supplied by the architecture and the genome is left to
   supply only the deviation from it. `sigma_{l,m}` is initialised at exactly `1` and is
   free to move.

**[NORMATIVE]** The normalisation does **not** weaken the bottleneck. `rms(raw)` is one
scalar per matrix; all `d_out x d_in` degrees of freedom of the pattern still come from
`Gamma`. §06 measures this rather than assuming it.

## 3.3 Vector-valued decodes

**[NORMATIVE]** RMSNorm gains, the logit scale, the attention sink scalars and the
plasticity fields are decoded by the same trunk with the column argument set to the
vector form of §02.6 (`[ 0_{n_c} ; -1 ]`, `z_j = 0`):

```
    v_{l,m}[i] = raw[i] * sigma_{l,m} * s_target(l,m) + offset(m)
    offset(m)  = +1 for the gain-like roles, 0 otherwise
    s_target   = 0.5 for the gain-like roles, 1 otherwise
```

**[NORMATIVE — vector decodes are *not* rms-normalised, and matrix decodes are.]** A
matrix decode is normalised because its scale must not depend on `Gamma` (§3.2's first
reason). A vector decode must not be, for two reasons. It would be degenerate at length 1
(`ROLE_LOGIT_SCALE`), where `raw/rms(raw) = +/-1` carries no information at all. And it
would destroy the quantity these roles exist to carry: a gain field's *magnitude* is its
content, so normalising it away and multiplying by one decoded scalar would collapse
`d_model` degrees of freedom to one. Normalisation is unnecessary here in any case:
`raw = w_out . h2 + o_0` with `h2` bounded in `[-1,1]^{d_h}` and `||w_out|| ~ 1` is
already `O(1)` for every genome.

The `+1` offset on gain-like roles means an uninformative genome decodes to the identity
transformation rather than to zero, so a random genome at generation 0 produces a network
that runs.

## 3.4 Decoded plasticity `eta`

**[NORMATIVE]** The per-channel plasticity gain is a **vector decode under a reserved
role**, and therefore introduces no new machinery, no new genome tensor and no new code
path:

```
    eta_l[i]  =  exp( clamp( decode_vector(l, ROLE_PLAST_CHAN)[i], -c_eta, +c_eta ) )
    eta_tok[v]=  exp( clamp( decode_vector(., ROLE_PLAST_TOK )[v], -c_eta, +c_eta ) )
    c_eta     =  1.5      # so eta in [0.223, 4.48]; justified in §05.4
```

`eta` multiplies that channel's optimizer step (§05.4). The genome therefore specifies
**how much of what is allowed to change**, which is the third of §01.9's novel learning
mechanisms. Because `eta` is decoded by the same trunk, evolution cannot tune plasticity
independently of the weights it applies to — they are two readouts of one representation
of identity, which is the biologically correct coupling and the parsimonious one.

## 3.5 The genome, enumerated

**[NORMATIVE]** `Gamma_w` is exactly the following tensors and nothing else:

| tensor | shape | reals at defaults |
|---|---|---|
| `W_row` | `d_h x (n_c + d_z)` | 896 |
| `W_col` | `d_h x (n_c + 1 + d_z)` | 928 |
| `W_tok` | `d_h x (n_v + d_z)` | 960 |
| `W_ctx` | `d_h x (n_L + n_M + d_y)` | 544 |
| `c1` | `d_h` | 32 |
| `W_2` | `d_h x d_h` | 1024 |
| `c2` | `d_h` | 32 |
| `w_out` | `d_h` | 32 |
| `o_0` | scalar | 1 |
| `w_sig` | `d_h` | 32 |
| `sig_0` | scalar | 1 |
| `log_gain` | scalar | 1 |
| | **`\|Gamma_w\|`** | **4,483** |

at `d_h = 32`, `n_c = 12`, `n_v = 14`, `n_L = 5`, `n_M = 4`, `d_z = 16`, `d_y = 8`.
`Gamma_a` (§07.2) adds exactly **796**, giving `|Gamma| = 5,279`; §10.4's
self-adapted mutation strengths add **27** more — one per genome tensor, twelve
here and fifteen in the head — so the object that is stored, mutated and crossed
over is **5,306 reals**, 20.7 KB in float32.

**[NORMATIVE — these are equalities, and they were not.]** Until v1.1 this
section carried `~550` and `~5,033` against §07.2's exact 796 and 5,279, and the
only test touching the ratio below asserted that it exceeded 100. Every constant
in both tables is now recomputed from the module tree by
`tests_gnomon/unit/test_spec_numbers.py` (`G-SPECNUM`) and asserted exactly. A
number appears in this specification only if it appears there.

**[DERIVED — `H log N`, read off the table]** Not one entry depends on `d_model`, `L` or
`V`. Adding one bit of channel address — doubling the addressable width — costs
`2 d_h = 64` reals; one bit of vocabulary address costs `d_h = 32`. `n_c = 12` rather than 10 because every channel-like axis shares one code table and `d_ff = 2048` at M1 is the binding one, not `d_model`.

| | phenotype | `\|Gamma\|` | ratio |
|---|---|---|---|
| M-pico `128 x 4`, V=4096 | 1,528,969 | 5,279 | 289.6x |
| M-nano `192 x 6`, V=4096 | 4,060,883 | 5,279 | 769.3x |
| M1 `512 x 12`, V=8192 | 49,926,705 | 5,279 | **9,457.6x** |

**[NORMATIVE — the phenotype column is measured under `Arch.uniform`.]** The
decoded size depends on the architecture, so "the phenotype" is undefined until
an architecture is named: these are the configuration's own alternating
delta/attention arrangement, which is what `n_decoded` and every
fixed-architecture run use. An evolved genome decodes a different count, which
is precisely why §07.4 makes it a *descriptor* rather than a constant.

The values this replaces — `1.57 M`, `3.4 M`, `54.5 M`, and ratios of `312x`,
`676x`, `10,830x` — were carried from a draft and never recomputed, so the
project's headline compression claim was **overstated by 15%**. The property
that matters is unchanged and is now checked: the genome does not grow at all
between these three configurations, only the phenotype does, so the ratio
*improves* with scale.

The genome does not grow at all between these three; only the phenotype does. The
compression ratio therefore *improves* with scale, which is the property an `O(m+n)`
factorisation cannot deliver and which is the reason §02 refuses to store labels.

For reference, the paper's own ratios: **322x** and **1,038x** on MNIST, **92x** on
CIFAR-10, **492x**–**3,570x** on BeamRider, **4.9x** on HalfCheetah. §01.10 records why
these are not directly comparable.

## 3.6 Initialisation of `Gamma`

**[NORMATIVE]** At generation 0, and for every freshly sampled genome:

| tensor | initialisation | reason |
|---|---|---|
| `W_row`, `W_col`, `W_tok` | `N(0, 1/(n + d_z))` per row-block | unit-scale pre-activation into `tanh` |
| `W_ctx` | `N(0, 1/(n_L + n_M + d_y))` | as above |
| `W_2` | orthogonal, gain `1` | preserves `h1`'s spectrum; avoids the additive-collapse regime |
| `c1`, `c2`, `o_0`, `sig_0` | `0` | zero-mean decode; `sigma = 1` exactly |
| `w_out` | `N(0, 1/d_h)` | `E\|\|w_out\|\|^2 = 1` |
| `w_sig` | `0` | `sigma_{l,m} = 1` for every matrix at generation 0 |
| `log_gain` | **`log 4.0`** | measured; see the table below |

**[NORMATIVE — `log_gain`, the one hyperparameter measured rather than chosen.]**
`s = exp(log_gain)` is initialised at **`log 4.0`**. It is the decoder's most
consequential scalar and the default is set by the sweep in §3.1's table
together with the saturation cost below, not by convention.

| `s` | `erank/min(d)` at M1 | fraction of `h1` units with `|h1| <= 0.99` |
|---|---|---|
| 1 | 0.083 | 1.00 |
| 3 | 0.23 | 0.45 |
| **4** | **0.26** | **0.34** |
| 5 | 0.29 | 0.28 |
| 8 | 0.34 | 0.18 |

Small `s` puts `tanh` in its linear regime, where §3.1's collapse returns even
with two layers. Large `s` saturates it: at `s = 8` only 18% of the decoder's
hidden units are still responsive, and the gradient reaching `W_row`, `W_col`
and `log_gain` itself falls with them. `s = 4` is the smallest value clearing
§3.8's floor at **all three** configurations while leaving a third of the
decoder responsive. Valid range `[2, 8]`; `log_gain` is evolved (§07) and is
differentiable in the Lamarckian arm, so the initialisation need only place the
genome in a workable basin.

**[NORMATIVE — `W_2`'s orthogonal initialisation must be seeded.]**
`torch.nn.init.orthogonal_` draws from the **global** RNG and ignores a passed
generator, which made `Decoder(shape, seed=k)` irreproducible; the gain sweep
above drifted between runs until it was found. `W_2` is built by a seeded QR
with the sign correction that makes the factorisation unique.

## 3.7 Computational cost

**[NORMATIVE]** Per decoded matrix of shape `d_out x d_in`:

```
    row/col/ctx projections   O( (d_out + d_in) * d_h * (n + d_z) )
    h1 (broadcast + tanh)     O( d_out * d_in * d_h )
    h2 (matmul + tanh)        O( d_out * d_in * d_h^2 )
    readout                   O( d_out * d_in * d_h )
    peak activation memory    2 * d_out * d_in * d_h  reals
```

At M1 the whole decode is **~57.5 GFLOP** against a phenotype step of **~5.2 TFLOP** at
`B*T = 16384` — about **1%**, and about **3%** with recomputation in the backward pass.

**[NORMATIVE]** The decode is recomputed once per optimizer step, not once per token, and
is gradient-checkpointed: the backward pass through the phenotype sees only `W`, and the
decode graph is rebuilt when `dL/dW` is available. The column axis of FFN matrices is
chunked so that no single `h1` exceeds `chunk_elems = 2^24` entries (134 MB in fp32).

**[NORMATIVE — inference]** At inference the decode runs **once** and `W` is cached.
Throughput is then exactly that of an ordinary transformer of the same shape, and the
shipped artifact is `|Gamma| + |Z|` — **20.7 KB + 544.4 KB at M1**, 565 KB
in all — that expands to 49.9M weights. This is the production path of §01 and it is checked by `G-INFER`.

## 3.8 Acceptance tests

**[NORMATIVE]** `gnomon/genome/decoder.py` is accepted only when all of the following
hold.

| # | property | test |
|---|---|---|
| D1 | **reference agreement** | the vectorised decode matches a naive `for i: for j:` reference to `1e-10` in float64 on random `Gamma`, for a matrix, a vector, and a token matrix |
| D2 | factorisation is exact | removing the broadcast and forming `h1` from explicitly concatenated `(i,j)` inputs gives identical output |
| D3 | **gradient reachability** | `dL/dGamma` is non-zero for **every** tensor in §3.5's table; a severed path is a test failure, not a silent regression |
| D4 | finite differences | central differences on 20 random entries of every genome tensor agree with autograd to `1e-5` (float64) |
| D5a | **rank floor** | decoded `erank` (spectral entropy of the singular values, vendored `utils/stats.effective_rank`) exceeds `0.20 * min(d_out, d_in)` at initialisation, 3 seeds x 3 configurations |
| D5b | **rank scaling** | fitting `erank ~ d^alpha` across pico/nano/M1 gives `alpha > 0.6` |
| D6 | **one-layer control** | the single-hidden-layer decoder collapses at `s = 1` (`< 5%` of `min(d)` at M1, deepening with width) and **fails D5b** (`alpha = 0.41`) at every gain. Without this control, D5b measures nothing |
| D7 | scale | `rms(W_{l,m}) == sigma_{l,m} * s_target(l,m)` to `1e-6`, for every role |
| D8 | scale-freedom | `|Gamma|` is bit-identical across M-pico, M-nano and M1 construction; decoding all three from one genome succeeds |
| D9 | nesting | decoding at `d` and at `2d` from one genome gives, for `i, j < d`, matrices that agree in the codes used (`b_i` identical, per §02 T3) |
| D10 | determinism | two decodes of one `(Gamma, Z)` are bit-identical on CPU and on CUDA independently |
| D11 | chunking | chunked and unchunked decodes agree to `1e-6`; chunk boundaries are exercised by a non-divisible shape |
| D12 | no free weights | the decoder module holds no `Parameter` outside §3.5's table (§04.6's absence gate, positive control included) |
