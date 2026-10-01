# 01 — SPORE: the model, fully specified

**Status: normative.** Every symbol has a dimension; every rule has inputs, outputs and a
termination guarantee; every hyperparameter has a default, a range and a reason. Code:
`spore/`. Tests: `tests_spore/`. Notation: vectors are columns; `⟨·,·⟩` the Euclidean inner
product; `‖·‖` its norm; `⊙` the Hadamard product; `1_v ∈ {0,1}^V` the one-hot of token `v`;
shapes are written `(rows × cols)`; `V = 8192` (the TinyStories tokenizer), `T = 512` (the
window length every model here is scored on). Constitution: `00 §0.5`.

## 1.0 The organism in one page

```
   genome  Γ = (seeds, dials)                                     a few dozen numbers            §1.7
      │  regenerate (deterministic, no fitting)
      ▼
   atoms  A = [a_1 … a_V] ∈ R^{D×V}, unit columns                 seeded hypervectors            §1.1
   permutation ρ (a seeded coordinate permutation)                                               §1.2
   context codes  c_t = (c_t^{(n)} for n in N, b_t, e_t)          binding + bundling             §1.2
   in-context memory  S_t (delta rule over the codes)             innate; learns inside a text   §1.5
   cortex  U = {u_1 … u_M}  units born over the life              development                    §1.6
   ─────────────────────────────────────────────────────────────────────────────────────────────
   feature vector  z_t = [c_t ; r_t ; h_t ; 1] ∈ R^F                                            §1.3
   readout  W ∈ R^{F×V}, fit by ridge then by the delta rule       the lifetime's convex object   §1.4
   logits  ℓ_t = Wᵀ z_t + τ_m m_t,   p_t = softmax(ℓ_t)           m_t = the memory's cleanup      §1.5
```

Two timescales: the **lifetime** (§1.3–1.6) reads a token budget `B` once and writes `W`,
the calibration scalars, and the cortex's unit set; the **genome** (§1.7–1.8) is what selection
writes across lifetimes. Nothing in the lifetime can change the structure; nothing in the
genome is fit to weights (`00 §0.5` S1–S3).

## 1.1 Atoms (the seeded vocabulary)

**Definition 1 (atoms).** For `D ∈ {1024, 2048, 4096, 8192}` and the seed `s_A`,

```
   g_v ~ N(0, I_D) drawn from PRNG(s_A, v),     a_v = g_v / ‖g_v‖,     v = 1..V.            (1.1)
```

`A = [a_1 … a_V] ∈ R^{D×V}`. Properties (Kanerva 2009 §2.1; Plate 1995 Thm 1 for the
variance of dot products of random vectors): `⟨a_u, a_v⟩` for `u ≠ v` has mean 0 and standard
deviation `1/√D` (0.0156 at `D = 4096`); `A` has full column rank almost surely when `D ≥ V`
and is a Johnson–Lindenstrauss embedding of the one-hot basis when `D < V`.

**Variant 1a (random-indexing atoms, a registered variant; Sahlgren 2005).** After one
counting pass over the budget, `a'_v = normalise( a_v + η_RI · ĉ_v )` where `ĉ_v` is the
normalised bundle of the fading-bag codes `b_t` (1.4) at every position where `v_{t+1} = v`.
This is Hebbian (a product of pre- and post-synaptic activity summed over the pass), single
pass, gradient-free, and gives distributionally similar tokens similar atoms. `η_RI ∈ [0, 2]`,
default 0 (off); the variant arm sets it to 1. Reason for the default: the base organism must
first be measured with *pure* seeded atoms (C1's cleanest form).

## 1.2 Context codes (binding and bundling)

**Definition 2 (position binding).** `ρ` is a coordinate permutation of `R^D` drawn from
`PRNG(s_ρ)`; `ρ^j` is its `j`-th power (`ρ^0 = I`). `ρ` is orthogonal, so `‖ρ^j x‖ = ‖x‖`, and
for a random unit vector `a`, `⟨ρ^j a, a⟩` has mean 0 and sd `1/√D` for every `j ≠ 0` such that
`ρ^j` has no fixed points in expectation (a random permutation has one fixed point in
expectation, i.e. `1/D` of the coordinates; the bias is `O(1/D)`). Permutation binding is the
standard sequence encoding of HDC (Kanerva 2009 §3.3; Jones & Mewhort 2007 for the
holographic-order variant).

**Definition 3 (the codes).** With `v_t` the token at position `t` of a window (`t = 1..T`,
`v_0 := ⊥` treated as a zero vector), the organism forms, for every `n ∈ N` (the set of
orders, a dial; default `N = {1, 2, 3, 4, 6}`):

```
   exact n-gram code      c_t^{(n)} = normalise( Σ_{j=0}^{n−1} ρ^j a_{v_{t−j}} )                ∈ R^D        (1.2)
   fading bag (gist)      b_t = normalise( γ_b b_{t−1} + a_{v_t} ),   b_0 = 0                  ∈ R^D        (1.3)
   fading trajectory      e_t = normalise( γ_e ρ(e_{t−1}) + a_{v_t} ),   e_0 = 0               ∈ R^D        (1.4)
```

`c_t^{(n)}` is a unit vector (normalised exactly; its summands are orthogonal in expectation, so
the normalisation is by `√n` up to `O(1/√D)`): two positions with
the same last `n` tokens have the same code; positions sharing a suffix of length `m < n`
have inner product `m/n` in expectation — the code *is* backoff, and the readout learns how
to weigh the orders (§1.4). `b_t` is order-free and carries the topic; `e_t` carries all
orders with geometric weights (`γ_e^j`) — the BEAGLE order vector with decay. Dials:
`γ_b ∈ [0.5, 0.999]` (default 0.95), `γ_e ∈ [0.3, 0.95]` (default 0.7). All codes are causal
and are computed within the window (the same information a transformer scored on the same
window receives). Cost per token: `|N| + 2` permutation-gathers and adds of `D` numbers.

**Control `binding = off` (K-HDC arm):** `ρ = I`. Then `c^{(n)}` is a bag of the last `n`
tokens and order information is lost.

**Variant 2a (the exact order-1 block, gene `onehot`; 03 §P-S.7).** At `D < V` the order-1
code is a random `D`-dimensional image of the `V` one-hots and the least-squares readout on
it is a rank-`D` approximation of the `V × V` bigram table (measured: behind the exact
bigram counts at `D = 2048`). With `onehot = True` the order-1 block is the identity code —
the one-hot `1_{v_t} ∈ R^V`, i.e. the orthonormal atom set at `D_1 = V`, exact by
construction — and the higher orders keep their random atoms at `D`. `F` grows by `V`. This
is the regime `D ≥ V` that HDC prescribes for the first order, at the cost the card can pay;
the higher-order blocks are where binding and bundling do their work.

## 1.3 The feature vector

```
   z_t = [ c_t^{(n)} for n ∈ N ;  b_t ;  e_t ;  r_t ;  h_t ;  1 ]   ∈ R^F,
   F = (|N| + 2) D + d_v + M + 1.                                                             (1.5)
```

`r_t ∈ R^{d_v}` is the memory's retrieval (§1.5), `h_t ∈ R^M` the cortex (§1.6), the final `1`
the bias. At `D = 2048`, `d_v = 1024`, `M = 0` at birth: `F = 15,361`; the Gram matrix of
§1.4 is `F × F` in `float32` (0.94 GB) beside the `F × V` cross-covariance (0.5 GB). **Measured
ceiling (03 §P-S.5):** at `F = 29,697` (`D = 4096`, five orders) the card's 16 GB is
oversubscribed and Windows pages CUDA memory (30 s per batch, 21.7 GB peak at the solve);
the practical ceiling at batch 32 in float32 is `F ≈ 22,000`, so `D = 4096` is run with three
orders and `D ≥ V` needs the blockwise solver of §1.9. The default `D` is therefore 2048 for
the screens of `02`, with `D` as a registered lever.

## 1.4 The readout: the lifetime's convex object

**Definition 4 (the ridge solution).** Over the budget's `B` target positions,

```
   G = Σ_t z_t z_tᵀ ∈ R^{F×F},     C = Σ_t z_t 1_{v_{t+1}}ᵀ ∈ R^{F×V},
   W_ridge = (G + λ Ḡ I)^{−1} C,    Ḡ = tr(G)/F,   λ ∈ [10^{−4}, 1], default 10^{−2}.         (1.6)
```

Both sums are accumulated in `float32` on the card (`G` by one `Zᵀ Z` per batch, `C` by an
`index_add_` over targets); the solve is a Cholesky factorisation in `float64` of a matrix
scaled to unit mean diagonal (the relative ridge `λ Ḡ` makes `λ` dimensionless). Because a
bias is included and `Σ_v 1_v = 1` at every position, `Σ_v (W_ridgeᵀ z)_v = 1` exactly for
every `z` (the fit of a constant target is exact) — the ridge scores are least-squares
estimates of `P(v | z)` (the regression function of a one-hot is the conditional
probability).

**Definition 5 (calibration).** The scores `s_t = W_ridgeᵀ z_t` are turned into a distribution by
`p_t = softmax( τ_s s_t + τ_m m_t )` with `(τ_s, τ_m)` fit by golden-section search on the
**fitness** set (two scalars, convex-in-each coordinate is not guaranteed and is not needed:
the search is bracketed and terminates in 40 evaluations at tolerance `10^{−3}`). `m_t` is
the memory's cleanup (1.10).

**Definition 6 (delta-rule refinement, the multinomial logistic readout).** Starting from
`W_0 = τ_s W_ridge`, `E` passes over a fresh `B_r` tokens with the exact gradient of the
convex objective `L(W) = −Σ_t log softmax(Wᵀ z_t + τ_m m_t)_{v_{t+1}}`:

```
   ∇_W L = Σ_t z_t (p_t − 1_{v_{t+1}})ᵀ,                                                        (1.7)
```

applied with Adam (`η_r = 10^{−3}`, decaying to zero, weight decay 0) on mini-batches. (1.7)
is the Widrow–Hoff delta rule for a softmax unit; it touches one weight layer and no error
term crosses a nonlinearity. It is implemented *without autograd* (the gradient is the
closed form), which is what `G-BPF` checks. The objective is convex in `W`, so the refinement
can only improve the fitness-set objective in expectation and its termination is by the
pass count. Default `E = 1`, range `[0, 3]`; the arm `refine = 0` reports the ridge solution
alone (the fully closed-form organism).

**Why ridge first.** The ridge solve gives the global least-squares readout in one pass
regardless of conditioning; the refinement then optimises the right loss from a good start,
so one pass suffices where SGD from zero would need many (measured in `03 §E-4`).

## 1.5 The in-context memory (innate, learns inside a text)

**Definition 7 (projections).** Seeded, fixed, never fit: `P_k ∈ R^{d_k×D}` and
`P_v ∈ R^{d_v×D}` with i.i.d. `N(0, 1/d)` entries from `PRNG(s_M)`, `d_k = d_v = 1024`
(range 256–2048; a JL projection of the codes and atoms that keeps the state at
`d_v d_k = 1M` numbers per sequence). Keys, values and queries:

```
   k_t = normalise(P_k c_t^{(n_k)}),    v_t = P_v a_{v_t},    q_t = k_t,                       (1.8)
```

with `n_k` the memory's key order (a dial, default 2). **The value at `t` is the *current*
token's atom and the key is the *previous* position's code**: the memory is updated with the
pair `(k_{t−1}, v_t)` and queried with `k_t`, so no target enters the state before it is
known.

**Definition 8 (the delta rule).** With `S_0 = 0 ∈ R^{d_v×d_k}` at the start of every window,

```
   S_t = S_{t−1} ( I − β k_{t−1} k_{t−1}ᵀ ) + β v_t k_{t−1}ᵀ,       r_t = S_t k_t,               (1.9)
```

the exact online least-squares update (Widrow & Hoff 1960; Schlag et al. 2021 Eq. 6; Yang
et al. 2024 Eq. 10 with `α = 1`): after the update, `S_t k_{t−1} = β v_t + (1 − β) S_{t−1} k_{t−1}`,
so at `β = 1` the memory *overwrites* the association of the previous context with the
token that followed it, and retrieval at a repeated context returns that token's value
vector plus interference from contexts with overlapping suffixes (which is generalisation
over similar keys, at `1/√d_k` noise per stored pair). `β ∈ (0, 2)` is a dial (default 1).
It is computed by `gated_delta_rule_chunk` (verified to 1e-10 against the float64 oracle)
with `α ≡ 1` (no decay within a window; the window is the document's horizon here) and one
head; the state is `(1, 1, d_v, d_k)` per sequence.

**Definition 9 (cleanup).** The retrieval is read back into token scores by the atoms:

```
   m_t = (P_v A)ᵀ r_t ∈ R^V,                                                                  (1.10)
```

which is the HDC cleanup memory: `m_{t,v}` is large when `r_t ≈ P_v a_v`. Two paths reach the
logits: `r_t` as a block of `z_t` (the readout learns what to do with the retrieval, e.g. how
much to trust it after a topic word) and `τ_m m_t` directly (Definition 5), which is what
the organism predicts with **at birth**, before any pass: `p_t = softmax(τ_m m_t)` is a
within-window `n_k`-gram copy predictor with no trained parameter. Its CE is the innate
reading `R-BIRTH` (`02 §2.3`).

**Control `memory = off` (K-HDC arm):** `r_t = 0`, `m_t = 0`.

## 1.6 The cortex: development by residual

**Definition 10 (units).** A unit `u` is a fixed pair `(w_u ∈ R^{F_in}, ψ)` with
`h_{u,t} = ψ(⟨w_u, x_t⟩)`, `x_t = [c_t ; r_t ; h_t^{(<u)}]` the features present when it is
born (so later units may read earlier ones: the cascade), `w_u` a seeded sparse direction
(`k_w = 64` nonzero entries of `±1/√k_w` from `PRNG(s_U, i)`), `ψ` the nonlinearity (a dial:
`relu`, `sign`, `cos` — `cos(⟨w,x⟩ + φ_u)` with a seeded phase is a random Fourier feature,
Rahimi & Recht 2007 Eq. 2; default `relu`).

**Definition 11 (a growth wave).** At the wave's token count `B_g` (the schedule is a dial:
`waves = (0.1, 0.25, 0.5)` of the budget, `k_wave = 512` units per wave, `M_cand = 8 k_wave`
candidates), with the current readout `W` (Definition 4 on the tokens so far):

1. draw `M_cand` candidate units from the seed and the wave index;
2. over the next `B_c` tokens (default `2^{18}`), compute for every candidate the
   **residual correlation** `ρ_u = ‖ Σ_t h̃_{u,t} (1_{v_{t+1}} − p_t)ᵀ ‖_2 / (‖h̃_u‖ ·  √B_c)`,
   with `h̃_u` the candidate's centred activation and `p_t` the current prediction — the
   cascade-correlation score (Fahlman & Lebiere 1990 Eq. 1, the sum over outputs taken in
   the 2-norm, without their gradient ascent on the candidate's weights: the candidates are
   seeded draws and the *selection* is the learning);
3. keep the top `k_wave` by `ρ_u`; freeze them; append their activations to `z_t`;
4. the Gram and cross-covariance grow by the new rows and columns, accumulated from this
   point on (a newborn unit's activation is zero on every token seen before its birth,
   which is the exact statement that it did not exist then; the block structure keeps
   `G` positive semidefinite as a sum of Gram matrices of two feature sets);
5. re-solve (1.6).

**Definition 11b (ridge-born units; 03 §P-S.22).** With `growth = ridge` a wave draws `k_wave`
seeded `±1` hashes of the vocabulary `H ∈ {±1}^{V×k}` and each unit is the ridge predictor of
its subset's membership from the features accumulated so far, `w_u = (G + λ Ḡ I)^{−1} C h_u`
(normalised), with `ψ` on top: no candidates, no scoring pass, one Cholesky solve with `k`
right-hand sides from the Gram the lifetime already holds. The unit is born from what the
organism has learned (S4) by a convex rule (S1).

**Termination.** Each wave is a finite pass with a fixed candidate count (11) or one solve
(11b); the life has a fixed number of waves. **Controls (K-DEV):** `growth = off` (no units, `M = 0`); `growth =
random` (the same waves and `k_wave`, candidates kept uniformly at random instead of by
`ρ_u`) — the arm that decides whether *selection* is what development contributes.

## 1.7 The genome

**[NORMATIVE] Γ is the complete description of the structure; a lifetime is a deterministic
function of `(Γ, budget, corpus offset)`.**

| gene | type | default | range | reason |
|---|---|---|---|---|
| `s_A, s_ρ, s_M, s_U` | seeds | (0, 1, 2, 3) | `[0, 2^31)` | regenerate atoms, permutation, projections, candidates |
| `D` | int | 4096 | {1024, 2048, 4096, 8192} | code dimension; sets `1/√D` interference (Kanerva 2009 §2) |
| `N` | set of ints | {1, 2, 3, 4, 6} | subsets of {1..8} | which n-gram orders are coded |
| `γ_b`, `γ_e` | reals | 0.95, 0.7 | [0.5, 0.999], [0.3, 0.95] | bag and trajectory decays |
| `n_k` | int | 2 | {1, 2, 3, 4} | the memory's key order |
| `β` | real | 1.0 | (0, 2) | the memory's write rate |
| `d_k = d_v` | int | 1024 | {256, 512, 1024, 2048} | memory state size |
| `λ` | real | 0.01 | [1e-4, 1] | relative ridge |
| `E` | int | 1 | {0, 1, 2, 3} | refinement passes |
| `waves`, `k_wave`, `ψ`, `k_w` | schedule | (0.1, 0.25, 0.5), 512, relu, 64 | see §1.6 | development |
| `η_RI` | real | 0 | [0, 2] | random-indexing strength (variant 1a) |

Genome length: 4 seeds and 16 numbers. Kolmogorov reading: the organism's structure is the
program "PRNG + these numbers"; everything else the lifetime writes from the corpus.

## 1.8 Selection across lifetimes

**Definition 12 (fitness).** `f(Γ) = CE_fitness(Γ, B_screen)`, the token-weighted cross-entropy
on the fixed fitness set (256 windows, v3) of the organism after a lifetime of `B_screen`
tokens (default 5M — the screening budget at which one life costs about a minute).

**Definition 13 (the loop).** Population `P = 8`, elites `2`, generations `G_max` (a wall-clock
budget). Offspring: uniform crossover of two elites' genes, then mutation of each gene with
probability `1/|Γ|`... per-gene rules: seeds → redraw; integers → a step of ±1 on the ordered
range; reals → log-normal with `σ = 0.3`; sets → add or remove one order. Selection is
`(μ + λ)`: the best `P` of parents and offspring survive. The **test** set is scored only for
the elite at the end of the run and at the generation-0 base (`00 §0.5` S5), so selection
cannot overfit the reported number; the reading is `CE_test(elite) − CE_test(base)`.

**Why this and not ES/REINFORCE.** The genome has integer and set-valued genes and a
landscape whose noise is the atom seed (`σ_e`, measured first); a `(μ + λ)` evolutionary
algorithm with elitism is monotone in the fitness set by construction (an elite is never
replaced by a worse individual), which is the guarantee that makes K-EVO's reading
well-defined. ANLAGE's SCREEN result — axes that are monotone above noise are hypotheses,
not search spaces — is respected by *reporting* which genes moved and whether their effect
is monotone (`02 §2.5`).

## 1.9 Accounting, throughput and the scale path

| object | size (defaults, `D = 4096`, `M = 1536` after three waves) | when |
|---|---|---|
| genome | 20 numbers | inherited |
| atoms `A`, permutation, projections | `4096 × 8192` + `4096` + `2 × 1024 × 4096` = 42M numbers, regenerated from seeds | at birth |
| readout `W` | `F × V = 31,233 × 8192 = 256M` numbers | written by the lifetime |
| Gram `G` | `F × F` float32 = 3.9 GB | during the lifetime only |
| memory state | `d_v × d_k = 1M` per sequence | during a text |

Per-token cost: codes `O((|N| + 2) D)`; memory `O(d_k d_v)` (the chunkwise kernel); units
`O(M k_w)`; readout `O(F V)`; Gram accumulation `O(F^2)` during the lifetime only. There is
**no backward pass**: a lifetime is one forward pass plus the Gram accumulation plus `E`
refinement passes, each a forward pass with a rank-one update per token. At `F = 30k` the
Gram term dominates (`9 × 10^8` per token), i.e. a 25M-token life is `2 × 10^{16}` FLOPs — the
FLOPs the 13M transformer spends on `0.3M` tokens of training.

**What the night measured as levers (03), largest first.** The readout's optimisation:
passes and step size of the delta rule (−0.29 for three passes, −0.23 for a fourfold step,
adding to −0.49 at 5M); the atoms' geometry (semantic atoms −0.155, half a nat at birth);
the memory's size (−0.02); ridge-born units (−0.046 over none); `D` (−0.03 at 2.4× the cost).
Not levers: units over random directions (sparse or dense, selected or not), output-side
sharing, the exact order-1 block at its cost, pure distributional atoms. The curve is flatter
than the transformer's (K-SOTA fires: −0.11 vs −0.29 nats per doubling from 5M to 25M), so
the next lever is the one that adds *features with the data*: development beyond a fixed
bank of units born from a fixed feature set.

**Scale path (stated, not claimed).** `D` and `d_k` scale the substrate; `M` scales the cortex
linearly in cost; `F` beyond the card's Gram ceiling is handled by *block* solves (the cortex
in blocks of 8k units, each block ridge-solved against the residual of the previous blocks —
gradient boosting in feature blocks, Friedman 2001 §4, exact in the limit of one unit per
block) or by conjugate gradients on `G` without forming it; the readout `F × V` is the same
object as a language model's output layer and scales as one. Production scale needs (i) the
Pile lineage at `V = 16384`, (ii) the random-indexing atoms (1a) or a co-occurrence SVD
for a vocabulary whose atoms must generalise, (iii) multi-window memory (`α < 1` in (1.9)
with a document-length horizon) — each is a dial or a variant already in the genome, none
is a new mechanism.

## 1.10 Initialisation and numerics

* Atoms and codes are unit vectors; the codes' sums are divided by `√n` so every block of
  `z_t` has unit norm and the Gram's diagonal blocks are commensurate (the relative ridge
  `λ Ḡ` then acts uniformly).
* `G` and `C` are accumulated in `float32` with the batch products formed in `bfloat16`
  matmuls promoted to `float32` accumulators only where the card's tensor cores are used
  (`Zᵀ Z` at `F ≤ 40k` is formed in `float32`; the bfloat16 path is a dial for the Pile
  lineage and is *off* by default because K-SOTA's resolution is `σ_e ≈ 0.01`).
* The solve is `float64` Cholesky of `(G + λ Ḡ I)/Ḡ`; failure (a non-positive pivot) raises
  and the life is recorded as failed — never silently regularised.
* Softmax is computed in `float32` with the max subtracted; logits are softcapped at 30 to
  bound the calibration search.
* The delta rule runs at the kernel's `float32` floor; `β` is clamped at `2 − 10^{−3}`.
* Every module asserts its shapes at the boundary (`gnomon/utils/shapes.py::check_shape`).
* The whole lifetime runs under `torch.no_grad()`; `G-BPF` asserts that no tensor in the
  organism requires grad and that `torch.autograd.backward` is never called (a monkeypatched
  sentinel in the test).
