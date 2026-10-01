# 02 — Identities: innate codes and the latent bank

The genome's inputs are identities. An identity has two parts: an **innate code**, a pure
function of an index that costs nothing and never changes (GNOMON's construction, retained
where it was measured to be right), and an **acquired latent**, a small vector fit by
regression and priced by a prior (new here: GNOMON learned its latents by gradient through a
saturating map and they ran away; §00.2).

## 2.1 Axes

**[NORMATIVE]** Every addressable unit belongs to exactly one axis. Codes are drawn from
three tables; latents are stored per axis.

| axis | units at stage `s` | code table | code width | latent | latent shared across layers? |
|---|---|---|---|---|---|
| `RES` | residual channels `i in [d_s]` | channel bijection | `n_c` | `z^RES_i in R^{d_z}` | **yes** — a residual channel is the same feature in every layer |
| `INNER(l)` | FFN hidden units `j in [F_s]` of layer `l` | channel bijection | `n_c` | `z^INN_{l,j}` | no |
| `HEAD(l)` | head channels `c in [h_s d_head]` of layer `l`; `KV(l)` is its prefix `[n_kv(l) d_head]` | channel bijection | `n_c` | `z^HEAD_{l,c}` | no |
| `GATE(l)` | the `2 h_s` decay/write gate rows of a delta layer | channel bijection | `n_c` | `z^GATE_{l,r}` | no |
| `TAP` | convolution taps `t in [4]` | channel bijection (prefix 4) | `n_c` | none | — |
| `TOK` | tokens `v in [V]` | frequency-rank Gray | `n_v` | `z^TOK_v` | — |
| `LAYER` | layers `l in [L_s]`, plus the global pseudo-layer `l = -1` | relative-depth Gray + flag | `n_L + 1` | `y_l in R^{d_y}` (none for `-1`) | — |
| `ROLE` | roles `m in M` | plain binary | `n_M` | none | — |
| `ONE` | the single unit of scalar roles | all `+1` | `n_c` | none | — |

**[NORMATIVE]** Widths are properties of the genome, not of the configuration being decoded,
and are chosen at the largest target scale of §09:

```
    n_c = 13    addresses 8192 channel-like units      (the Pile adult's F = 4096, with one doubling in reserve)
    n_v = 15    addresses 32768 vocabulary items        (V = 16384, with one doubling in reserve)
    n_L =  6    addresses 64 relative depths
    n_M =  5    addresses 32 roles (20 used, §04.2)
```

**[NORMATIVE]** Decoding an axis wider than its table **raises**; it never truncates, wraps or
collides. That boundary is exactly where `H log N` says the genome must grow by one input
column per feature-map row, and growing it is a deliberate, versioned act.

**[NORMATIVE]** All codes take values in `{-1, +1}`, never `{0, 1}` (GNOMON §02.1's reason: a
zero bit contributes nothing to a linear map, so `{0,1}` makes low-index identities
systematically smaller in norm).

## 2.2 The channel table: a seeded bijection

**[NORMATIVE]** Reused verbatim from `gnomon/genome/codes.py::seeded_bijection` and
`channel_codes`. Fix a permutation `pi` of `{0, ..., 2^{n_c} - 1}` drawn once from a generator
seeded by `seed_codes`; unit `i` of any channel-like axis has code
`b_i = 2 bits_{n_c}(pi(i)) - 1`.

Three properties, each an acceptance test (§2.8):

* **(P1) Nesting.** `image(kappa_d) ⊂ image(kappa_{d'})` for `d ≤ d'`, and unit `i` has the
  **same** code at every width. A channel that exists at stage `s` keeps its identity at
  stage `s+1`; the new channels `d_s ≤ i < d_{s+1}` are new identities the genome has never
  seen, which is what makes Proposition 2 of §05 meaningful.
* **(P2) Spread.** The image is a uniformly random `d`-subset of the Hamming cube: no bit
  position is constant, so no genome column is untrained noise at a wider decode.
* **(P3) Determinism.** `pi` is a pure function of `seed_codes` and `n_c`.

**[NORMATIVE]** `pi` is not part of `Gamma` and is never mutated: it is the coordinate system
shared by every genome of every generation. `seed_codes = 20260902`.

**[NORMATIVE — why channels take no structured code]** A residual channel has no
pre-training order; a smoothness prior over an arbitrary permutation is a prior over nothing
(GNOMON §02.9's last paragraph). `INNER`, `HEAD`, `GATE` and `TAP` share the table because a
single shared table is what makes the count in §03.6 independent of how many axes exist.

## 2.3 The token table: Gray code over frequency rank

**[NORMATIVE]** Reused verbatim from `gnomon/genome/codes.py::token_codes(rank=…)` and
`frequency_rank`. With `rank(v) = 0` for the most frequent type:

```
    q_v  =  round( rank(v) (2^{n_v} - 1) / (V - 1) )
    e_v  =  2 bits_{n_v}( q_v XOR (q_v >> 1) ) - 1
```

Counts are taken over the whole training shard (`gnomon/data/stream.py::token_counts`,
Laplace `+1`), ties broken by ascending token id through a stable sort, and the counts'
digest is written to the run manifest. This is the construction GNOMON measured at
**+0.0999 nats [+0.029, +0.265]** against the bijection under plain AdamW, and the reason is
structural rather than a learning-rule effect: the embedding rows are decoded from the token
code, and frequency rank is the one topology a vocabulary has before training. Relative rank
transfers across vocabulary sizes (rank 0 is "the most frequent type" at every `V`), which is
the token axis's form of P1.

## 2.4 The layer table: Gray code over relative depth, with a global flag

**[NORMATIVE]** For a phenotype of depth `L`:

```
    q_l  =  round( l (2^{n_L} - 1) / (L - 1) )        (L ≥ 2);   q_0 = 0   (L = 1)
    r_l  =  [ 2 bits_{n_L}( q_l XOR (q_l >> 1) ) - 1 ;  +1 ]
    r_{-1} = [ 0_{n_L} ; -1 ]                          the global pseudo-layer
```

The last bit is the flag: `+1` for a real layer, `-1` for the global roles (`EMB`, `HEAD_OUT`,
`NORM_FINAL`, `LOGIT_SCALE`), whose code bits are then genuinely unused rather than
ambiguously equal to layer 0's.

**[NORMATIVE — the layer code is relative, and therefore not nested]** When depth grows from
`L_s` to `L_{s+1}`, an existing layer's `q_l` changes. This is intended: the genome learns
"what an early layer looks like", never "what layer 3 looks like", so that a genome fit at
`L = 8` is interpolating rather than extrapolating at `L = 12`. The consequence is stated so
it is not mistaken for a defect: Proposition 2 (nesting) of §05 holds on channel and token
axes; on the layer axis the existing layers keep their **trained weights** through a growth
event (§06.2) and their latents `y_l` travel with them, while their *codes* are re-derived,
and the next regression refits the genome to the new codes.

## 2.5 The role table

**[NORMATIVE]** `t_m` is the plain binary code of the role's index in §04.2's table; 20 roles,
`n_M = 5`. Adding a role after genomes exist would silently change what every existing genome
decodes; the table is fixed for the life of the project.

## 2.6 The latent bank and its prior

**[NORMATIVE]** `Z` holds, at stage `s`:

| tensor | shape | count |
|---|---|---|
| `z^RES` | `(d_s, d_z)` | `d_s d_z` |
| `z^INN` | `(L_s, F_s, d_z)` | `L_s F_s d_z` |
| `z^HEAD` | `(L_s, h_s d_head, d_z)` | `L_s h_s d_head d_z` |
| `z^GATE` | `(L_s, 2 h_s, d_z)` | `2 L_s h_s d_z` |
| `z^TOK` | `(V, d_z)` | `V d_z` |
| `y` | `(L_s, d_y)` | `L_s d_y` |

with `d_z = 16`, `d_y = 8` at every configuration (dials of §08.4).

**[NORMATIVE — the prior]** Every latent carries an isotropic Gaussian prior
`z ~ N(0, sigma_z^2 I)`, `sigma_z = 1`, which enters the regression objective as the term
`lambda_z ||z||^2` of §05 (5.3) and makes `Z` a *priced* part of the description (§05
Proposition 1). Nothing about `Z` is learned by gradient descent on the language loss.

**[NORMATIVE — sampling for growth]** At a growth event, a new unit on axis `X` of layer `l`
receives `z ~ N(mu_X, diag(s_X^2))` where `mu_X, s_X` are the per-coordinate mean and standard
deviation of the fitted latents on that axis (over all layers for `RES`, over layer `l`'s
existing units for `INNER(l)`, `HEAD(l)`, `GATE(l)`; over the nearest existing layers for a
newly inserted layer's `y`). `d_z = 0` is a valid configuration (identity entirely innate; the
control arm of §05.6), in which sampling is the identity map.

## 2.7 What the construction buys

**[NORMATIVE — registered before it is tested]** Because every table is fixed and `|Gamma|`
depends on `(n_c, n_v, n_L, n_M, d_z, d_y, d_h, H, d_g, d_sigma)` and not on
`(d_s, L_s, F_s, h_s, V)`, one genome decodes a phenotype of any size up to the addressable
maximum with no change to its shape. `K-SCALE` (§08.3) tests it: a genome regressed at stage
`s` must express stage `s+1`'s new units better than random at birth.

## 2.8 Acceptance tests

**[NORMATIVE]** `anlage/codes.py` and `anlage/latents.py` are accepted only when all of the
following hold (test ids are bound by `scripts/anlage_acceptance.py`).

| # | property | test |
|---|---|---|
| I1 | codomain | every code entry is exactly `-1.0` or `+1.0` |
| I2 | injectivity | the channel map is injective at every `d in {2^k}` up to `2^{n_c}` |
| I3 | **nesting (P1)** | for all `d ≤ d'` and `i < d`: `b_i(d) == b_i(d')` elementwise on the channel axes; the frequency code of a given type is unchanged when `V` grows only through its rank |
| I4 | spread (P2) | per-bit mean over the image within `[-3/sqrt(d), +3/sqrt(d)]`; a planted prefix coding fails (positive control) |
| I5 | determinism (P3) | two constructions from one `seed_codes` are bit-identical; different seeds differ |
| I6 | Gray adjacency | `r_l, r_{l+1}` differ in one code bit whenever `q_{l+1} = q_l + 1`; adjacent frequency ranks differ in fewer than half as many bits as a random code's neighbours |
| I7 | relative depth | for `L = 8` and `L = 16`: `r_0` agrees, `r_{L-1}` agrees; the global flag of `r_{-1}` is `-1` and its code bits are zero |
| I8 | overflow | decoding `d > 2^{n_c}`, `V > 2^{n_v}` or `L > 2^{n_L}` raises, naming the width that must grow |
| I9 | zero cost | the codes module holds no `Parameter` and no buffer whose size depends on the phenotype other than the derived tables |
| I10 | frequency carried | a linear probe recovers `R^2 ≥ 0.3` of log frequency from the token code and `≤ 0.05` from the bijection (the control) |
| I11 | latent bank shapes | `Z.count(s)` equals the table of §2.6 at three stages, asserted as equalities |
| I12 | sampling | `sample_new(axis, n)` returns latents whose per-coordinate mean and sd match `(mu_X, s_X)` within `4/sqrt(n)` at `n = 4096`, and returns zeros at `d_z = 0` |
