# 03 — The genome: a neural kernel over identity space

**Status (2026-09-02, after §11):** the decode of *matrix* roles specified here was measured (R4, the probe, R4b; `RESULTS.md`) and is no longer load-bearing; it is kept as the record and for the tests. The decode of *vector* roles (3.7) survives as the scale profile `s` of §12.6.

One object decodes every weight of every phenotype the lineage will ever express. This file
specifies it completely: equations, dimensions, cost, size, initialisation, and the tests that
decide whether an implementation is correct.

The form is chosen against GNOMON's, and the difference is the point. GNOMON evaluated a
per-entry MLP `g(b_i, b_j)` on every `(i, j)` — `O(d_out d_in d_h^2)` per matrix, a
saturating `tanh` on the sum of the two identities, and a measured effective rank of 26% at
`d = 512`. ANLAGE evaluates a **neural feature map once per unit** and forms the matrix by a
**bilinear kernel** plus a cheap pointwise nonlinearity: one batched GEMM per matrix, no
saturation on the identity path, and the interaction — the part the PNAS paper says the
genome should carry, "what a relationship between two identities means" — is an explicit
matrix `O_m` per role.

## 3.1 Feature maps

**[NORMATIVE]** For a unit `a` on a channel-like axis, in layer context `l` (or the global
pseudo-layer), on side `side in {+1 (row), -1 (column)}`:

```
    xi_{a,l,side}  =  [ b_a ; z_a ; r_l ; y_l ; side ]                 in R^{n_c + d_z + n_L + 1 + d_y + 1}
    h_1            =  GELU( W_1 xi + c_1 )                              in R^{d_h}
    h_2            =  GELU( W_1b h_1 + c_1b )                           in R^{d_h}
    f(xi)          =  W_2 h_2 + c_2                                     in R^{H d_g}, reshaped to (H, d_g)     (3.1)
```

`y_{-1} = 0`. For a token `v`, `xi` uses the `n_v`-bit token code and its own first layer
`W_1^tok in R^{d_h x (n_v + d_z + n_L + 1 + d_y + 1)}`, `c_1^tok`; `W_1b, c_1b, W_2, c_2` are
shared by every axis. `TAP` and `ONE` units use the channel-code path with `z = 0`.

**[NORMATIVE]** Row-side features `u_{a,l}^{(h)} = f(xi_{a,l,+1})^{(h)}` and column-side
features `v_{a,l}^{(h)} = f(xi_{a,l,-1})^{(h)}` are computed **once per (unit, layer, side)**,
cost `O(|A_s| L_s d_h (n + d_z + d_y + H d_g))`, and cached for the decode.

**[NORMATIVE — two hidden layers]** GNOMON §03.1 measured that a single hidden layer makes
an additive decoder low-rank. The kernel of §3.2 is multiplicative by construction, so the
argument does not transfer directly; the second layer is kept because it is cheap
(`d_h^2`) and because D5 measures rather than assumes the rank. `d_h = 256`.

## 3.2 The kernel decode of a matrix role

**[NORMATIVE]** For a matrix role `m` in layer `l` with row axis `R` and column axis `C`
(§04.2):

```
    raw_{l,m}[i, j]  =  sum_{h=1}^{H}  g_m( u_{i,l}^{(h)T} O_m^{(h)} v_{j,l}^{(h)} ),        i in R, j in C     (3.2)
    g_m(x)           =  x + kappa_m x^3,        kappa_m = softplus(kappa_hat_m)                                  (3.3)
    W_hat_{l,m}      =  raw_{l,m} / rms(raw_{l,m})  *  s_target(m, shape)  *  sigma_{l,m}                       (3.4)
    rms(R)           =  sqrt( mean_{i,j} R[i,j]^2 + 1e-12 )
```

`O_m^{(h)} in R^{d_g x d_g}` per role and head; `H = 4`, `d_g = 64`.

**[NORMATIVE — cost]** (3.2) is, per head, one GEMM `(|R| x d_g) (d_g x |C|)` followed by an
elementwise map: `2 H d_g + O(H)` FLOPs per decoded weight, `512 + O(1)` at the defaults. A
350M-weight phenotype decodes in `≈ 1.8 x 10^11` FLOPs. Peak memory is `H` copies of the
largest matrix in float32 for the elementwise stage unless heads are accumulated in place,
which the implementation does; column chunking at `chunk_cols` keeps any single temporary
under `2^26` elements.

**[NORMATIVE — rank]** Before `g_m`, each head's term has rank `≤ d_g`, so `raw` has rank
`≤ H d_g = 256` plus whatever the cubic term adds (a degree-3 polynomial kernel in `(u, v)`,
whose feature space has dimension `O(d_g^3)`). D5 measures the effective rank of decoded
matrices against the trained matrices they are fit to, per role; the dials `(H, d_g, d_h,
d_z)` are raised by §08.4's protocol if the fit residual of any role exceeds its floor.

## 3.3 The per-matrix scale

**[NORMATIVE]**

```
    log sigma_{l,m}  =  w_sigma . tanh( A_sigma [ r_l ; t_m ] + b_sigma ),     A_sigma in R^{d_sigma x (n_L + 1 + n_M)}     (3.5)
    s_target(m, shape) =  fan_in^{-1/2}      for matrix roles other than EMB, HEAD_OUT, VE, CONV_*
                       =  d_s^{-1/2}         for EMB and HEAD_OUT  (rows are tokens; §04.4's read scaling)
                       =  (n_kv d_head)^{-1/2} for VE
                       =  1/4                for CONV_*  (four taps, the identity tap carries 1)
```

`d_sigma = 16`, `w_sigma` initialised at `N(0, 10^{-6})` so `sigma_{l,m} = 1` to within 0.1% for
every matrix at initialisation (§3.7 says why it is not exactly zero). `s_target` is innate — a function of shape — so the genome supplies only the
deviation from a correctly scaled initialisation, and no genome, mutated or random, decodes to
a network that is untrainable by scale alone (GNOMON §03.2's first reason, retained).

## 3.4 Vector and scalar roles

**[NORMATIVE]** For a vector role `m` over axis `R` (norm gains) or the scalar role
`LOGIT_SCALE` over `ONE`:

```
    v_hat_{l,m}[i]  =  u_{i,l}^{(1)} . o_m  +  offset(m),        o_m in R^{d_g}                                 (3.6)
    offset(m)       =  1  for NORM_MIX, NORM_FFN, NORM_FINAL;    0 for LOGIT_SCALE
```

Not RMS-normalised: a gain field's magnitude is its content (GNOMON §03.3). `o_m` is
initialised at `0`, so an uninformative genome decodes every gain to `1` and the logit scale
to `0` — the identity transformation.

## 3.5 Embedding and head

**[NORMATIVE]** `EMB` has row axis `TOK` and column axis `RES`, decoded by (3.2)–(3.4) with
`s_target = d_s^{-1/2}`. While the head is tied (§04.4) there is no `HEAD_OUT`; after untying,
`HEAD_OUT` is a distinct role with its own `O_m`, fit to the untied head like any other
matrix.

## 3.6 The genome, enumerated

**[NORMATIVE]** `Gamma` is exactly the following tensors and nothing else. Counts at the
defaults `n_c = 13, n_v = 15, n_L = 6, n_M = 5, d_z = 16, d_y = 8, d_h = 256, H = 4, d_g = 64,
d_sigma = 16`, with `|M_mat| = 16` matrix roles and `|M_vec| = 4` vector/scalar roles (§04.2):

| tensor | shape | reals |
|---|---|---|
| `W_1` (channel path) | `d_h x (n_c + d_z + n_L + 1 + d_y + 1)` = `256 x 45` | 11,520 |
| `c_1` | `d_h` | 256 |
| `W_1^tok` | `d_h x (n_v + d_z + n_L + 1 + d_y + 1)` = `256 x 47` | 12,032 |
| `c_1^tok` | `d_h` | 256 |
| `W_1b`, `c_1b` | `d_h x d_h`, `d_h` | 65,792 |
| `W_2`, `c_2` | `H d_g x d_h`, `H d_g` | 65,792 |
| `O` | `|M_mat| x H x d_g x d_g` | 262,144 |
| `kappa_hat` | `|M_mat|` | 16 |
| `o` | `|M_vec| x d_g` | 256 |
| `A_sigma`, `b_sigma`, `w_sigma` | `d_sigma x (n_L + 1 + n_M)`, `d_sigma`, `d_sigma` | 224 |
| | **`|Gamma|`** | **418,288** |

Not one entry depends on `d_s, L_s, F_s, h_s` or `V`. The schedule `omega` (§06.5) is
serialised with the genome and holds integers, not reals; the mutation strengths of §07.3 are
per-tensor scalars (11 reals) held beside `Gamma`, as GNOMON §10.4 did.

**[NORMATIVE — every count above is an equality asserted by `G-SPECNUM`]** GNOMON's headline
ratio was stated three times and agreed with the code in none of them (`docs/gnomon/AUDIT-v1.1
A-3`). `tests_anlage/unit/test_spec_numbers.py` recomputes each row from the module tree.

**[DERIVED — the ratios]** Against the phenotype counts of §04.5 (asserted there by
`G-SPECNUM`, from which this table is written):

| phenotype | `|W|` | `|Gamma|` | `|Z|` | ratio (genome) | ratio (genome + latents) |
|---|---|---|---|---|---|
| TinyStories s0 | 13,011,201 | 418,288 | 300,096 | 31× | 18× |
| TinyStories adult | 78,307,841 | 418,288 | 798,848 | 187× | 64× |
| Pile adult | 461,186,049 | 418,288 | 2,257,088 | 1,103× | 172× |

Both ratios are reported with every result (A2); neither is a target.

## 3.7 Initialisation

**[NORMATIVE]** A fresh genome (generation 0, before any regression; also the "random
genome" control of §05.6) is drawn as:

| tensor | initialisation |
|---|---|
| `W_1, W_1^tok, W_1b` | `N(0, 1/fan_in)`, biases `0` |
| `W_2` | `N(0, 1/d_h)`, `c_2 = 0` |
| `O` | `N(0, 1/d_g)` per entry, so `u^T O v = O(1)` for unit-scale features |
| `kappa_hat` | `softplus^{-1}(0.1)` |
| `o` | `0` |
| `A_sigma` | `N(0, 1/(n_L + 1 + n_M))`; `b_sigma = 0`; `w_sigma ~ N(0, 10^{-6})` — not exactly zero, so that `sigma_{l,m} = 1` to within 0.1% while `A_sigma`'s gradient path, which is proportional to `w_sigma`, is live from the first regression step (D2) |

seeded from `derive_seed(run_seed, "init")` through an explicit generator. The decode of a
fresh genome is therefore a structured random matrix at exactly the innate scale: trainable
(by (3.4)) and distinct from the free-weight initialisation of §04.7 (by its rank).

## 3.8 What the decode is and is not used for

**[NORMATIVE]** `D` runs: at birth (§05.4), at every growth event (§06.2), inside `REGRESS`
(§05.2) and inside `MERGE` (§07.4). It **never** runs inside a training step. A lifetime holds
`W` as free parameters; `G-NOFREE-INVERSE` (§08) walks the compiled training step and asserts
that no genome tensor is read by it. This is A3, and it is the exact inversion of GNOMON
§05.1's line 5 — for the reason §00.2 measures.

## 3.9 Acceptance tests

**[NORMATIVE]** `anlage/genome.py` is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| D1 | **reference agreement** | the batched decode matches a naive per-entry evaluation of (3.1)–(3.4) to `1e-10` in float64 on a random genome, for a matrix role, a vector role, the embedding and a `CONV_*` role |
| D2 | gradient reachability | `dJ/dGamma` is non-zero for every tensor of §3.6 under a regression step on random targets; a severed path is a test failure |
| D3 | finite differences | central differences on 20 random entries of every genome tensor and of `Z` agree with autograd to `1e-5` in float64 |
| D4 | **scale-free** | `|Gamma|` is bit-identical when constructed for the three configurations of §04.5; decoding all three from one genome succeeds |
| D5 | **rank** | effective rank of the decoded `Q`, `UP` and `EMB` at initialisation exceeds `0.20 min(shape)` at every stage; after a regression to a trained phenotype the decoded erank is within a factor `2` of the trained matrix's, per role, or the residual report names the role |
| D6 | scale | `rms(W_hat_{l,m}) == s_target sigma_{l,m}` to `1e-6` for every matrix role; vector roles decode to `offset(m)` exactly at `o_m = 0` |
| D7 | **nesting (Proposition 2)** | decoding at stage `s` and `s+1` from one `(Gamma, Z)` gives, for `i, j in A_s`, `raw` entries that agree to `1e-6` on every channel and token axis |
| D8 | determinism | two decodes of one `(Gamma, Z)` are bit-identical on CPU and on CUDA independently |
| D9 | chunking | column-chunked and unchunked decodes agree to `1e-6` at a non-divisible chunk |
| D10 | cost | the adult decode's wall-clock is under `1%` of `N_sleep` training steps' wall-clock (§06.4), measured on this card |
| D11 | totality | 1,000 random genomes each decode every role of every stage to finite tensors |
| D12 | no decode in step | `G-NOFREE-INVERSE`: a planted decode inside the training step is detected (positive control) |
| D13 | counts | every row of §3.6 recomputed from the module tree as an equality |
| D14 | uninformative genome | with `O = 0` and `o = 0` a decode gives zero matrices and identity gains, so an "empty" genome is a well-defined object |
