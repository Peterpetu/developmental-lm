# SOMA — Specification 05: The Phenotype Backbone

**Spec version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Normative for `soma/model/`.

This section specifies the substrate on which development and selection act. It is
deliberately conventional and deliberately strong: inventing a new substrate *and* a new
learning paradigm simultaneously would render any failure unattributable. The substrate is
the experimental control.

**[NORMATIVE]** Every tensor shape below is asserted at runtime at the module boundary.

---

## 5.0 Configured dimensions

Normative values live in §01.9. Repeated here for locality:

| Symbol | M-nano | M1 | Constraint |
|---|---|---|---|
| `L` | 6 | 12 | `L mod 4 = 0` for M1; M-nano uses period 3 (see §5.1) |
| `d_model` | 192 | 512 | `d_model mod n_h = 0` |
| `N_G` | **2048** | **2048** | Site inner width; germline-scale, identical at both configs (§01.2) |
| `n_h` / `n_kv` | 4 / 2 | 8 / 2 | `n_h mod n_kv = 0` |
| `d_h` | 48 | 64 | `= d_model / n_h` |
| `n_m` / `d_k` / `d_v` | **3** / 64 / 64 | 4 / 128 / 128 | `n_m * d_v = d_model * e_v` (v2 had `2*64 = 128 != 192`) |
| `e_v` | 1 | 1 | mixer value expansion |
| `C_chunk` | 32 | 64 | power of two |
| `T_ctx` | 128 | 512 | context |
| `V` | 4096 | 8192 | vocabulary |
| `U_max` / `side` | 256 / 16 | 1024 / 32 | **`U_max = side^2`** (unit routing is product-key) |
| `k_route` / `k_g` | 4 / **8** | 8 / 8 | units per token; `k_g` is germline-scale |
| `d_key` | 32 | 64 | routing key width, even |
| `W_swa` | — | — | **disabled**; see §5.3 |

**[NORMATIVE]** There is **no `d_ff`**. The Site's inner width *is* `N_G`: the gene library
plays the role of the feed-forward inner dimension. Consequently `d_in = d_out = d_model`
at a Site, and no output projection is required.

---

## 5.1 Block layout

**[NORMATIVE]** Blocks alternate mixer and attention on a fixed period:

```
  M1      (L = 12, period 4):  attention at  i mod 4 == 3   ->  3 attention, 9 mixer
  M-nano  (L =  6, period 3):  attention at  i mod 3 == 2   ->  2 attention, 4 mixer
```

**[EMPIRICAL]** A controlled 72-model study found language-modelling loss nearly flat
across the linear-to-full ratio while **recall** improves sharply below 3:1, recommending
3:1 to 6:1. M1 adopts 3:1. **[NORMATIVE]** Any ablation of this ratio is evaluated on
recall-intensive probes (§08), never on loss alone, because the study's own finding is that
perplexity is not the signal on this axis.

Each block:

```
  x <- x + Norm_post( Mix ( Norm_pre(x) ) )
  x <- x + Norm_post( Site( Norm_pre(x) ) )
```

**[NORMATIVE] Peri-LN.** Both the input and the output of every sublayer are normalised.
Pre-norm residual streams exhibit variance growth exponential in depth; post-norm has
stable variance but degraded gradient flow. `Norm` is RMSNorm:

```
  RMSNorm(x) = x / sqrt( mean(x^2) + eps ) * gain_rms ,   eps = 1e-6             (5.1)
```

`gain_rms` in `R^{d_model}` is learned, initialised to `1`. **No bias terms appear anywhere
in the model.** (Consequently §03.4's death rule performs no bias transfer — see §03.)

**[NORMATIVE]** The mixer is **not** cross-layer weight-tied. An earlier draft asserted
"SOMA shares the mixer and differentiates the FFN"; that was wrong. What is shared is the
**parts library** `(A, B, kappa)`, which is global across Sites; every mixer, every
`W_shared`, and every unit population is per-block.

---

## 5.2 The recurrent mixer

### 5.2.1 Recurrent definition

State `S_mix` in `R^{d_v x d_k}` per head, per sequence. For each head independently:

```
  S_mix[t] = S_mix[t-1] ( alpha_mix[t] ( I - beta_mix[t] k_mix[t] k_mix[t]^T ) )
             + beta_mix[t] v_mix[t] k_mix[t]^T                                   (5.2)
  o[t]     = S_mix[t] q_mix[t]                                                   (5.3)
```

`q_mix, k_mix` in `R^{d_k}` with `||k_mix|| = 1`; `v_mix` in `R^{d_v}`;
`alpha_mix` in `(0,1)`; `beta_mix` in `(0, 2-1e-3]`.

**[DERIVED]** (5.2) is one step of online gradient descent on
`l(S) = (1/2)||S k - v||^2` with `alpha_mix` acting as adaptive weight decay:

```
  S - beta ( (alpha S) k - v ) k^T  applied to  alpha S
    = S ( alpha ( I - beta k k^T ) ) + beta v k^T                                (5.4)
```

so the mixer is a test-time learner over `S_mix`, and `alpha_mix` is its forgetting gate.

### 5.2.2 Signed transitions

**[NORMATIVE]** `beta_mix[t] = min( 2 - 1e-3 , 2 * sigmoid( x_t w_beta ) )`.

**[NORMATIVE — corrected eigenvalue statement]** The transition matrix is
`alpha_mix (I - beta_mix k k^T)`, whose eigenvalues are

```
  alpha_mix * (1 - beta_mix)   along k        in  ( -alpha_mix , alpha_mix )
  alpha_mix                    orthogonal to k                                   (5.5)
```

An earlier draft omitted `alpha_mix` and claimed eigenvalues `1 - beta` and `1`. The
exact value `-1` is therefore **not** attainable: a Householder reflection is approached
only as `alpha_mix -> 1`.

**[DERIVED — why this still works]** `sigmoid` attains `1 - 1e-4` comfortably. At
`alpha_mix = 0.9999`, a state persists over 256 steps with factor `0.9999^256 = 0.975`, so
the parity construction survives with 2.5% decay. **[NORMATIVE]** A regression test asserts
that the trained `alpha_mix` on the parity task exceeds `0.99` in at least one head.

**Proposition 5.1.** A finite-precision linear RNN whose state-transition matrices have
only non-negative real eigenvalues cannot recognise `(11)*` at arbitrary length. Parity is
non-star-free but is recognised by a two-state automaton, so this is an expressivity gap.

**[EMPIRICAL]** Extending the eigenvalue range from `[0,1]` to `[-1,1]`: parity
`0.017 -> 1.000`, modular arithmetic `0.314 -> 0.971`, `S_5` with swaps solved in one
layer. **[OPEN]** The cited source reports that medians are materially weaker than
best-of-3 (`0.826 +/- 0.146` vs `0.971`) and that language metrics *degrade* for diagonal
models; our regression bar (§08.2) is set against the median, not the best.

### 5.2.3 Numerical stability

The cited source states its modification is incompatible with computing cumulative products
in log space, and reports degraded performance for a diagonal model's chunkwise variant.

**Analysis.** The two multiplicative quantities play different roles:
- `alpha_mix` in `(0,1)` is a **positive scalar** and is the only quantity entering the
  intra-chunk cumulative product `g_r = prod_{i<=r} alpha_mix^i`. Since `alpha_mix > 0`,
  `log g_r = sum_{i<=r} log alpha_mix^i` is well defined and is accumulated in `float32`.
- `beta_mix` in `(0,2)` carries the **sign** of the transition eigenvalue but appears only
  inside the Householder factor, which is never accumulated as a scalar product. It enters
  the chunkwise algorithm through the UT transform (§5.2.5), an explicit triangular solve.

**[DERIVED]** The mechanisms are compositional: the sign lives in `beta_mix`, the log-space
reduction in `alpha_mix`, and they do not interact.

**[NORMATIVE — conditioning]** `beta_mix <= 2 - 1e-3` is clamped, and
`tests/regression/test_signed_gdn.py` asserts `cond(I + strictlower(diag(beta) K K^T))
< 1e6` over the test grid. Unbounded `beta` with `||k|| = 1` admits inverses whose entries
grow with chunk length.

**[NORMATIVE]** The test must verify (a) chunkwise agrees with the naive recurrence to
`1e-5` in `float32` and `1e-10` in `float64`, and (b) parity scaled accuracy at length 256
after training on lengths 3..40 exceeds `0.95`. On failure the signed variant is disabled
and the failure reported, not worked around.

### 5.2.4 Feature construction

**[NORMATIVE]** Per block, with `D_qkv = n_m * (2 d_k + d_v)`:

```
  W_qkv   in R^{d_model x D_qkv}
  conv    depthwise causal, kernel 4, one filter per channel of D_qkv
  w_alpha, w_beta in R^{d_model x n_m}
  W_gate  in R^{d_model x (d_model e_v)}
  OutProj in R^{(d_model e_v) x d_model}

  q_raw,k_raw,v_raw = split( x W_qkv )
  q_mix = L2Norm( SiLU( conv(q_raw) ) )        # unit norm per head
  k_mix = L2Norm( SiLU( conv(k_raw) ) )
  v_mix =         SiLU( conv(v_raw) )
  alpha_mix = sigmoid( x w_alpha )
  beta_mix  = min( 2-1e-3, 2*sigmoid( x w_beta ) )
  y = OutProj( RMSNorm(o) * SiLU( x W_gate ) )
```

**[EMPIRICAL — component priority]** From a 400M/15B ablation, cost of removal in average
perplexity / average accuracy: `alpha_mix` gate **+3.52 / -2.14** (largest); L2 replaced by
L1+SiLU **+2.83 / -1.17**; output gate **+1.77 / -1.80**; short convolution
**+1.60 / -1.10**; output norm **+0.20 / -0.19**.

**[NORMATIVE]** The `alpha_mix` gate and the L2 normalisation of `q_mix, k_mix` may not be
removed except as a labelled experiment. The output norm may be dropped under memory
pressure.

### 5.2.5 Chunkwise parallel form

Fix a chunk of `C_chunk` positions, indexed `r_pos = 1..C_chunk`. Let `S` be the state
entering the chunk and

```
  g_r = prod_{i=1}^{r} alpha_mix^i ,     g_0 = 1                                 (5.6)
```

**Lemma 5.2 (the gate factors out).** With
`P_WY^r = prod_{i=1}^{r} ( I - beta^i k^i k^{iT} )` (increasing `i` applied on the right),

```
  prod_{i=1}^{r} alpha^i ( I - beta^i k^i k^{iT} ) = g_r * P_WY^r                (5.7)
```

*Proof.* Each `alpha^i` is a scalar; scalars commute with matrices; collect them. QED

Unrolling (5.2) across the chunk:

```
  S^r = S * g_r * P_WY^r  +  sum_{i=1}^{r} (g_r / g_i) * beta^i v^i k^{iT} * P_WY^{i+1..r}
                                                                                 (5.8)
```

**WY representation.**

```
  P_WY^r = I - sum_{i=1}^{r} w^i k^{iT}                                          (5.9)
  w^r    = beta^r ( k^r - sum_{i=1}^{r-1} w^i ( k^{iT} k^r ) )                  (5.10)
```

*Proof by induction.* At `r = 1`, (5.10) gives `w^1 = beta^1 k^1` and (5.9) matches.
Assume (5.9) at `r-1`. Then

```
  P^r = P^{r-1}( I - beta^r k^r k^{rT} )
      = P^{r-1} - beta^r ( P^{r-1} k^r ) k^{rT}
      = I - sum_{i<r} w^i k^{iT} - beta^r ( k^r - sum_{i<r} w^i (k^{iT}k^r) ) k^{rT}
```
which is (5.9) at `r` with `w^r` as in (5.10). QED

**UT transform.** Stacking `K, V, beta` over the chunk, (5.10) is unit-lower-triangular:

```
  T_UT  = [ I + strictlower( diag(beta) K K^T ) ]^{-1} diag(beta)    in R^{C x C} (5.11)
  W_WY  = T_UT K                                                     in R^{C x d_k}(5.12)
```

**Gated value accumulation.** With `Gam_decay[r,i] = g_r / g_i` for `r >= i`, else `0`:

```
  u~^r    = beta^r ( v^r - sum_{i<r} u~^i ( (g_r/g_i) k^{iT} k^r ) )             (5.13)
  U_tilde = [ I + strictlower( diag(beta) ( Gam_decay (*) K K^T ) ) ]^{-1} diag(beta) V
                                                                                 (5.14)
```

**Chunk update and output.** With the arrow rescalings

```
  Q<-[r] = g_r q^r ,   W<-[r] = g_r w^r ,   K->[i] = (g_C / g_i) k^i ,   S-> = g_C S
```

the state update is

```
  S_next = S->  +  ( U_tilde - W<- S^T )^T K->                     in R^{d_v x d_k} (5.15)
```

and — **this is the equation an audit found wrong in spec v1, and it is corrected here** —
the output is

```
  O = Q<- S^T  +  ( Q K^T (*) Gam_decay (*) M_causal ) ( U_tilde - W<- S^T )     (5.16)
```

*Derivation.* From (5.8)–(5.9), `o^r = S^r q^r` expands to

```
  o^r = g_r S q^r
        - g_r sum_{i<=r} (S w^i)( k^{iT} q^r )
        + sum_{i<=r} (g_r/g_i) u~^i ( k^{iT} q^r )
```

The first term is `Q<- S^T`. In the third term the factor `g_r/g_i` is exactly
`Gam_decay[r,i]`, giving `(Q K^T (*) Gam_decay (*) M) U_tilde`. In the second term,
`Gam_decay[r,i] * g_i = g_r`, so `-(Q K^T (*) Gam_decay (*) M) W<- S^T` contributes
`-g_r sum_i (k^{iT}q^r)(S w^i)^T` as required. QED

**[NORMATIVE — the v1 error, recorded]** Spec v1 wrote
`O = Q<- S^T + (Q K^T (*) M)(U_tilde - W<- S^T)`, i.e. without `Gam_decay`. That form
yields `sum_i (k^{iT}q^r) u~^i` — missing `g_r/g_i` — and `-sum_i (k^{iT}q^r) g_i (S w^i)^T`,
carrying `g_i` where `g_r` is required. It is correct **only** in the ungated limit
`alpha_mix == 1`. A regression test pins the `alpha_mix == 1` case so this equivalence is
checked rather than assumed.

**[DERIVED — complexity]** Per chunk: two triangular solves and matmuls of cost
`O(C^2 d_k + C d_k d_v)`. Over `T_ctx/C` chunks the total is
`O(T_ctx * C * d_k + T_ctx * d_k * d_v)`, linear in `T_ctx`.

**[NORMATIVE]** Sequences are padded to a multiple of `C_chunk` and pad positions are
masked; `C_chunk` is not required to divide `T_ctx`.

**[NORMATIVE — every decay *ratio* is formed in log space; v3.1, audit A-65]** `Gam_decay`
and `K->` both contain the ratio `g_r / g_i`. Neither may be computed as a quotient. Both
are computed as `exp( (log g_r - log g_i) clamped to <= 0 )`, where
`log g_r = sum_{j<=r} log alpha^j` is the cumulative sum already required by §05.2.3.

The reason is **stronger in the backward pass than in the forward one**, which is why
inspecting the forward values does not reveal it. `d(g_C/g_i)/dg_i = -g_C/g_i^2`, and `g_i^2`
underflows at the square root of `g_i`'s own limit: in `float32` the *gradient* is already
`NaN` at `sum_j log alpha^j = -96` (`g_i ~ 2e-42`), a point at which the forward value is
still exactly representable and the loss is finite. With `C_chunk = 64` that is a sustained
`alpha_mix ~ 0.22` — an ordinary value for a decay gate, reached by ordinary training.

Observed: an M-nano lifetime with a finite loss of `4.04` whose gradient was non-finite, and
whose non-finiteness reached exactly one parameter tensor, `w_alpha` — the sole source of
`alpha_mix`, and therefore of `g`. `Q<-` and `W<-` are *not* ratios: they scale by
`g_r <= 1`, so underflow to zero is the numerically correct answer there and no log-space
treatment is required.

The regression test carries a **positive control**: the quotient form, on the identical
inputs, must be non-finite. Without it the test would pass at any decay strength mild enough
that the two forms agree, which is every decay strength the equivalence tests had used.

### 5.2.6 Reference implementation and the oracle

**[NORMATIVE]** Two implementations are maintained:
1. `soma/model/mixer_ref.py` — the naive recurrence (5.2)–(5.3), tested in `float64`. This
   is the **oracle**.
2. `soma/model/mixer_chunk.py` — the chunkwise form (5.11)–(5.16).

`tests/unit/test_mixer_equivalence.py` asserts agreement to `1e-5` (`float32`) and `1e-10`
(`float64`) for `C_chunk` in `{16,32,64}` and `T_ctx` in `{1,31,32,33,63,64,65,512}`.

**[RATIONALE — licensing]** The mixer is implemented from the equations above. The
reference code accompanying the source publication is under a non-commercial licence and is
not consulted or vendored.

---

## 5.3 The attention layer

**[NORMATIVE]** At M-nano and M1 the attention layers are **full causal attention**. The
sliding window is **disabled** (`W_swa` unset).

**[NORMATIVE — v1 correction]** Spec v1 specified a sliding window of 256 against a
context of 512 for *every* attention layer, which left the model with **no global receptive
field at all**, contradicted §5.3's own reference to "the global path", and invalidated the
transfer of the cited hybrid study (whose non-linear layers were full attention). Windowing
is an `[OPEN]` scaling-path option enabled only for `T_ctx > 2048`.

```
  W_q in R^{d_model x n_h d_h} ;  W_k, W_v in R^{d_model x n_kv d_h}
  W_o in R^{n_h d_h x d_model} ;  z_sink in R^{n_h}

  q, k = RMSNorm_headwise(q), RMSNorm_headwise(k)          # gain-free, see below
  s[t,j] = ( q[t] . k[j] ) / sqrt(d_h)     for  j <= t
  a[t]   = softmax( [ s[t, 0..t] ; z_sink[h] ] )
  o[t]   = sum_j a[t,j] v[j]                               # the sink's value is 0
```

**[NORMATIVE]** The QK RMSNorm is **gain-free** (no learned scale). With a learned gain the
bound `|s| <= sqrt(d_h)` does not hold, since the gain is unbounded; spec v1 asserted the
bound while specifying a gained norm.

**[RATIONALE]** Attention entropy collapses at a rate exponential in the spectral norm of
the logits; bounding `||q||,||k||` caps that norm and permits a higher stable learning rate.
Logit soft-capping is **not** additionally used; the two are redundant.

**The sink.** `z_sink[h]` is a learned per-head logit appended to the row, with an implicit
zero value vector, so

```
  a[t,j] = exp(s[t,j]) / ( sum_i exp(s[t,i]) + exp(z_sink[h]) )                  (5.17)
```

and the attention weights sum to less than 1: a head may decline to attend. Softmax
otherwise forces a distribution, so a head with nothing to retrieve improvises a sink on
the first token and distorts the residual stream.

**Positional encoding.** **[NORMATIVE]** NoPE — no positional encoding on the attention
path. Causal masking alone breaks permutation symmetry, the recurrent layers carry position
implicitly, and removing explicit position improves length extrapolation. **[OPEN]** RoPE
(`theta = 1e4`) is added to the attention path if the M2 length-extrapolation curve
regresses relative to a RoPE control.

---

## 5.4 Routing tokens to units

**[NORMATIVE — fixed slot grid]** Each Site owns a preallocated grid of `U_max = side^2`
slots. A unit occupies a slot; `live_u` and `dying_u` are per-slot flags. **The grid size
never changes**, so no tensor is ever reshaped by a birth or a death, and no unit is ever
re-addressed by a population change.

**[NORMATIVE — v1 correction]** Spec v1 sized the key tables as `sqrt(|U|)`, so adding one
unit changed `sqrt(|U|)` and re-addressed **every** unit, which made Theorem 3.1 false. The
fixed grid removes this.

```
  K_unit1, K_unit2 in R^{side x (d_key/2)}          # SOMA, not germline
  W_key            in R^{d_model x d_key}

  qk       = x_t W_key                                          in R^{d_key}
  s1       = qk[:d_key/2] K_unit1^T / sqrt(d_key/2)             in R^{side}
  s2       = qk[d_key/2:] K_unit2^T / sqrt(d_key/2)             in R^{side}
  active[i1,i2] <- live and (not dying) and (m_u > 0)
  s1, s2  <- s1, s2 masked to -inf on rows/cols with no `active` slot
  I1, I2   = topk_{k_route}(s1), topk_{k_route}(s2)
  cand     = { (i1,i2) : i1 in I1, i2 in I2, active[i1,i2] }
  while |cand| < k_route and I1,I2 can widen:
      widen I1 and I2 by one rank each; recompute cand        # bounded by `side`
  R_t      = topk_{min(k_route,|cand|)} over cand of  s1[i1] + s2[i2]            (5.18)
  r_ut     = softmax over R_t of ( s1[i1] + s2[i2] )
```

**[NORMATIVE]** One predicate, `active` = live and not dying and `m_u > 0`, governs the
row/column mask, `cand`, and the softmax alike. v2 masked rows on "has a live slot" while
excluding newborns only from `cand`, in a prose paragraph the algorithm block did not
implement; a newborn that was the first live slot in its row would flip that row from `-inf`
to finite, changing `I1` and hence **every other unit's** `r_ut`, so Theorem 3.1a did not
hold.

**[NORMATIVE — `|cand|` is not guaranteed]** `cand` may be smaller than `k_route` or empty:
at M-nano initialisation (64 active of 256 slots, `k_route = 4`) the expected `|cand|` is
about 4 and `P(|cand| = 0)` is roughly 0.5% per token. The widening loop runs first; if
`cand` is still empty then `R_t = {}` and the Site output is the shared bypass alone.
`|R_t| = min(k_route, |cand|)`, and the shape assertion admits `|R_t| <= k_route`.

**[NORMATIVE]** `K_unit1, K_unit2` are **soma**. Were they germline, `d|Gamma|/d|U_max|`
would be non-zero and §09.3's bottleneck claim would be false by construction.

**[NORMATIVE — mandatory control]** Learned routing is worth far less than it appears: in a
directly comparable study a **random router** reached `2.9335` NLL / `39.1` accuracy against
the best learned router's `2.8667` / `40.1` — the entire learned-routing gain is
`-0.067` NLL and `+1.0` point. Baseline **B3** (random router) is therefore required in
every routing experiment, and routing is justified by the metabolic and structural story,
never by an expected routing gain.

**Expression-space query.** Separately, `q_t = normalize( x_t W_pos )` in `S^{m_dim-1}`,
used only by `L_wire` (§01.6).

---

## 5.5 The Site

**[NORMATIVE]** Unit populations live at the Site (feed-forward) positions. A controlled
cross-layer sharing study found that sharing attention costs approximately nothing
(`+0.1` to `-0.7` on a benchmark average) while sharing feed-forward parameters is where
the damage concentrates; the differentiation budget is therefore spent at the Site.

```
  A in R^{N_G x d_model}   (shared across all Sites)
  B in R^{d_model x N_G}   (shared across all Sites)
  kappa in R^{N_G x d_key} (shared across all Sites)
  W_shared, W_ctx per Site

  z          = A x_t                                                 in R^{N_G}   (5.19)
  psi(x_t)   = sigmoid( kappa ( x_t W_ctx ) / sqrt(d_key) )          in [0,1]^{N_G}
  gate_u     = phi(p_u) (*) psi(x_t) (*) 1_{g_u}                     k_g-sparse
  Site(x_t)  = sum_{u in R_t} r_ut * m_u * B [ relu( gate_u (*) z )^2 ]
               + W_shared x_t                                        in R^{d_model} (5.20)
```

**[NORMATIVE — the activation is applied AFTER the gate]** `relu(.)^2` acts on
`gate_u (*) z`, **not** on `z` alone. This is what makes units distinct nonlinear functions.
With the activation *before* the gate, `B` and the gate are both linear, so

```
  sum_u r_ut m_u B[ gate_u (*) relu(z)^2 ] == B[ (sum_u r_ut m_u gate_u) (*) relu(z)^2 ]
```

**exactly** — every unit would be a linear reweighting of one shared nonlinear feature, and
the forbidden aggregate form would compute bit-for-bit the same function at `1/k_route` of
the cost. §01.4 records this. The post-gate placement costs nothing, because `gate_u` is
`k_g`-sparse and therefore so is `relu(gate_u (*) z)^2`.

**[NORMATIVE]** `W_shared x_t` sits **outside** the routed sum.

Three defects are corrected here. v1 placed `W_shared x_t` *inside* the sum, giving it the
token-dependent coefficient `sum_u r_ut m_u` in `[0,1]`, so the shared bypass was silently
attenuated during every birth anneal and vanished entirely at a Site whose units were all
newborn. v1 also left the activation's position ambiguous across three readings. v2 removed
the ambiguity but chose the placement that makes units provably indistinct.

**[DERIVED]** Because `gate_u` is `k_g`-sparse, `relu(gate_u (*) z)^2` is `k_g`-sparse and
`B` applied to it costs `O(k_g d_model)`, so (5.20) costs
`O(N_G d_model + k_route k_g d_model)` — Theorem 1.

**[NORMATIVE]** `psi == 1` is ablation **A10**, which tests whether the context gate earns
its place. A10 must be reported with its parameter-count confound stated (it removes
`W_ctx` and `kappa`).

---

## 5.6 Embeddings, head, losses

- Token embedding `E` in `R^{V x d_model}`, `Var(E_ij) = 1 / d_model`. **[NORMATIVE]** Tied
  to the output head at M-nano and M1; untied at M4 and above.

  **[NORMATIVE — the embedding is read scaled by `sqrt(d_model)`; the head is not]**

  ```
    x_0    = E[tokens] * sqrt(d_model)              # residual stream starts at RMS 1
    logits = Norm_out(x_L) E^T                      # unscaled, so logits are O(1)
  ```

  Under weight tying one matrix sets two scales that want different values. Peri-LN normalises
  every sublayer's output to RMS 1 before adding it, so the residual stream should **start** at
  RMS 1; the head should produce logits of order 1. With `Var(E) = 1/d_model` and no scaling
  the residual starts at RMS `d_model^{-1/2}` — `0.072` at M-nano — so `Norm_pre` amplifies it
  by `14x` at block 0 and the stream jumps ~20x in a single block.

  **Measured**, that is a **43x** (M-nano) / **61x** (M1) discontinuity in the residual
  gradient at block 0, against `1.1x`–`2.4x` at every other block: the entirety of §02.5's
  gradient-flow failure, concentrated in one place. Scaling on read fixes the residual scale
  and leaves the head's — `Norm_out(x) E^T` sums `d_model` terms of size `d_model^{-1/2}`, so
  logits stay `O(1)`. Total first-to-last gradient ratio falls from `304 -> 13` at M-nano and
  `799 -> 24` at M1. See `AUDIT-v1.md` A-57.
- **z-loss** `L_z = lambda_z (log Z)^2`, `lambda_z = 1e-5`, `Z` the softmax partition
  function, averaged over **the same token set as `L_task`** (i.e. the selected tokens,
  §06.2), so the two terms share a denominator.
- No weight decay on embeddings or norm gains.

---

## 5.7 Optimizer

**[NORMATIVE — groups enumerated by name, not by tensor rank]**

| Group | Members | Optimizer |
|---|---|---|
| **M** | `W_qkv`, `W_gate`, `OutProj`, `W_q`, `W_k`, `W_v`, `W_o`, `A`, `B`, `W_shared`, `W_key`, `W_ctx`, `kappa`, `K_unit1`, `K_unit2` | Muon |
| **A** | `E` (and untied head), all `gain_rms`, `z_sink`, `w_alpha`, `w_beta`, `W_pos`, **`p_u`**, conv filters | AdamW |

**[NORMATIVE]** `p_u` is in group **A**. It is a `U_max x m_dim` matrix with `m_dim = 8`
(§01.2), so it is extremely wide-aspect: at M1, `1024 x 8`. Newton–Schulz orthogonalisation
drives all 8 singular values toward 1, destroying the *relative* geometry of positions on
`S^7` that `phi_g` reads and `L_disp` shapes; and Muon's `0.2 sqrt(max(rows,cols))` scaling
with `rows = U_max = 1024` gives a step multiplier of `6.4`, applied to a parameter
constrained to the unit sphere. (v2 justified this placement by asserting `m_dim = 2`, which
contradicts §01.2's `m_dim = 8`; the placement is correct, the stated reason was not.) Spec v1's
"all 2-D soma matrices" rule wrongly captured both `p_u` and the embeddings.

Muon:

```
  G      = grad
  M_buf  = mu_muon * M_buf + G                          mu_muon = 0.95
  G'     = G + mu_muon * M_buf                          (Nesterov)
  O      = NewtonSchulz5( G' / ||G'||_F )
  W     <- W - lr(n) * 0.2 * sqrt(max(rows,cols)) * O                            (5.21)
```

`NewtonSchulz5` runs 5 iterations of `X <- a X + b (X X^T) X + c (X X^T)^2 X` with
`(a,b,c) = (3.4445, -4.7750, 2.0315)`, in `float32`.

AdamW: `betas = (0.9, 0.95)`, `eps = 1e-8`. Weight decay `0.1` on group **M** only.
Gradient clipping at global norm `1.0`.

**[NORMATIVE — optimizer state under structural change]** All per-unit tensors (`p_u` and
its optimizer state) are preallocated at `U_max`. A death **zeroes** the freed slot's
parameter and optimizer state. A birth writes the child's position and leaves its optimizer
state at **zero** — momentum is not inherited, since inheriting it would inject a non-zero
update into a unit whose mask is `0` and would violate the spirit of Theorem 3.1a.

**[RATIONALE]** Muon reports roughly `2x` compute efficiency against AdamW at
compute-optimal training and is also a memory win (one momentum buffer, no second moment).
**[NORMATIVE]** Adam-family optimizers are reported to suffer catastrophic plasticity loss
under long non-stationary training with a collapse in effective rank; `erank` is logged
(§03.4) and the birth/death cycle is the designed mitigation.

---

## 5.8 Schedule and batch

**[NORMATIVE — the circularity is removed]** `D_tok` and `B_tok` are **configured**
(§01.9); `N_step = ceil(D_tok / B_tok)` is derived. v1 derived `B_tok` from `D_tok` while
`D_tok = B_tok * N_step` was itself unknown.

**[RATIONALE]** A published critical-batch-size fit gives
`round_pow2(22.91 * D_tok^0.47)`, i.e. `65536` at `D_tok = 4e7` and `262144` at `8e8`. The
configured values (`16384`, `65536`) are deliberately **4x smaller**, to keep activation
memory and the selection buffer inside 16 GB and to give the selection machinery more events
per token. v2 asserted the configured values were "consistent with" the law; they are not,
and the deviation is stated rather than claimed away. Both are powers of two and both give
exactly **128 sequences per step**.

Warmup–stable–decay:

```
  lr(n) = lr_max * n / n_warm                                 n < n_warm
        = lr_max                                              n_warm <= n < n_dec
        = lr_max * (1 - (n - n_dec)/(N_step - n_dec))         n >= n_dec
```

with `n_warm = 0.01 N_step`, `n_dec = 0.86 N_step`.

**[NORMATIVE]** `f_oe` is annealed to zero across the decay phase (§06.3.4). Placing
distributionally unusual data where the learning rate is smallest wastes it, and the final
model must be calibrated on natural text.

---

## 5.9 Numerical precision

- Parameters and activations `bfloat16`; optimizer state and Newton–Schulz `float32`.
- **[NORMATIVE]** Every normalisation reduction (RMSNorm, the gain-free QK norm, and the
  masked form (3.4)) accumulates at **no less than `float32`**, by the same floor rule as the
  triangular solves below: a `bfloat16` mean-of-squares over width 512 loses ~3 bits to the
  accumulation alone, and that factor then multiplies the whole residual stream.
- The UT triangular solves (5.11), (5.14) are computed at **no less than `float32`** —
  formally `promote_types(ambient, float32)` — because their conditioning degrades in
  `bfloat16`. **[NORMATIVE — v3 correction]** v2 wrote "`float32` regardless of ambient
  dtype", which *downcasts* a `float64` call and makes §05.2.6's own `1e-10` `float64`
  oracle test unsatisfiable: the realised error is then `~2e-7`, the `float32` epsilon, and
  gate `G-MIXER` could never pass. The requirement is a floor, not an exact dtype.
- The log-space decay accumulation `log g_r` is `float32`.
- **[NORMATIVE]** Token-selection excess losses (§06.2) are accumulated in `float32`; a
  `bfloat16` difference of two `O(1)` losses is dominated by rounding late in training and
  would rank noise.
- **[OPEN]** `float8` for the head and `B`; deferral criterion is a profile showing those
  matmuls above 15% of step time. The installed CUDA build does not expose `scaled_mm`.

---

## 5.10 Parameter and memory accounting

**M1** (`L=12`, `d_model=512`, `N_G=2048`, `n_m=4`, `d_k=d_v=128`, `V=8192`, `U_max=1024`,
`side=32`, `d_key=64`, `m_dim=8`):

| Component | Count |
|---|---|
| Embedding (tied) `8192*512` | 4,194,304 |
| Mixer blocks (9), incl. the §5.2.4 output norm | 11,893,248 |
| Attention blocks (3) | 1,966,104 |
| Sites (12), incl. `p_u` | 4,104,192 |
| Parts library (shared: `A`, `B`, `kappa`) | 2,228,224 |
| Block and output norm gains | 25,088 |
| **Total** | **24,411,160** (24.41 M; non-embedding 20.22 M) |

Optimizer split by the named groups of §5.7: **Muon 19,947,520**, **AdamW
4,463,640**, summing to the total. (v2 stated "≈18 M / ≈6 M", which does not
follow from enumerating §5.7's groups.)

**[NORMATIVE — v3.1 correction]** The v3.0 table omitted the mixer's **output
norm** — the `RMSNorm(o)` of §5.2.4's `y = OutProj( RMSNorm(o) * SiLU(x W_gate) )`
— from both the "Mixer blocks" row and the "Norm gains" row, understating the
total by `9 * d_model * e_v = 4,608` at M1 and `4 * 192 = 768` at M-nano. The
"Norm gains" row counts the four Peri-LN gains per block plus the output norm
(`12 * 4 * 512 + 512 = 25,088`) and never included the mixer-internal one. The
counts above are produced by `SomaModel.parameter_report()`, which enumerates
`named_parameters()`, so the table can no longer disagree with the model.

Muon state at 6 bytes/param ≈ 0.120 GB; AdamW state at 12 bytes/param
≈ 0.054 GB.

**M-nano** (`L=6`, `d_model=192`, `N_G=2048`, `n_m=3`, `d_k=d_v=64`, `V=4096`,
`U_max=256`, `side=16`, `d_key=32`, `m_dim=8`): **2,935,752** (2.94 M);
Muon 2,108,416, AdamW 827,336. v2 quoted "≈2.6 M" against a configuration it had not
recomputed; v3.0 quoted 2,934,984, omitting the mixer output norm as above.

Additional residents introduced by SOMA machinery, at M1:

| Resident | Size | Location |
|---|---|---|
| `theta_lag` parameters (`bfloat16`) | 0.049 GB | **GPU** (§06.1.2: it is forward-evaluated) |
| Certification checkpoint ring, `J_ring = 8` (`bfloat16`) | 0.39 GB | CPU |
| Certification batch ring, `K_ring = 32` x `32 x 512` int32 | 0.002 GB | CPU |
| Selection reservoir, all 12 Sites: `q_cases * (2 k_route + 2)` f32 | 0.005 GB | GPU |
| `L_ref` sidecar, 2 B/token over `D_tok` | 1.6 GB | disk, memory-mapped |
| Muon momentum, 6 B/param over 19.95 M | 0.120 GB | GPU |
| AdamW state, 12 B/param over 4.46 M | 0.054 GB | GPU |

The `L_ref` sidecar is read by index alongside the token stream and is never resident in
full. v2 asserted these were "unchanged from v2" without enumerating them, and omitted
`theta_lag`'s residency change entirely.

---

## 5.11 Falsification

- If the chunkwise mixer does not match the oracle to `1e-5`, nothing downstream is
  trustworthy and work stops (K6).
- If parity does not exceed `0.95` at length 256, §5.2.2 does not hold in our
  implementation and the claim is withdrawn (K7).
- If a random router (B3) is equivalent to the learned router under the declared TOST
  margin, routing carries no information and only `L_wire` justifies unit identity (K10).
- If `psi == 1` (A10) is equivalent under the declared margin, the context gate contributes
  nothing.
