# 02 — Innate codes

The decoder's inputs are *identities*. This section specifies them completely: what an
identity is, how it is derived from an index, and why the derivation is what makes the
genome scale-free.

Nothing here is stored or learned. Every quantity in this section is a pure function of
an integer index and the configuration, and contributes **zero** to `|Gamma|` and zero to
`|Z|`. That is the whole point: it is the reason the genome grows as `H log N` rather
than `O(N)`.

## 2.1 The rule inherited from the PNAS paper

**[NORMATIVE]** Shuvaev, Lachi, Koulakov & Zador (PNAS 121(38):e2409160121, 2024) use
**Gray code where the index carries a topology** — a pixel's `x` and `y` — and **plain
binary where it does not**, for hidden units "since their order is of no particular
importance". GNOMON applies the same rule, axis by axis:

| axis | topology? | coding | reason |
|---|---|---|---|
| channel `i` | no | pseudorandom bijection (§2.4) | channel order is arbitrary; a smoothness prior over it would be a prior over nothing |
| token `v` | **yes** | Gray code over relative **frequency rank** (§2.9) | BPE merges the most frequent pair first, so merge order *is* frequency order. §2.1 first placed this axis in the "no topology" class; §2.9 records the measurement that corrected it |
| layer `l` | **yes** | Gray code over **relative depth** (§2.5) | layer 3 is genuinely between 2 and 4; and relative depth is what transfers across `L` |
| matrix role `m` | no | plain binary (§2.6) | the 12 roles are a nominal set |

**[NORMATIVE]** All codes take values in `{-1, +1}`, never `{0, 1}`. A zero bit
contributes nothing to `W b`, so a `{0,1}` code would make low-index identities
systematically smaller in norm than high-index ones — an artifact of the encoding rather
than of the identity it encodes.

## 2.2 Widths are allocated at the largest target scale

**[NORMATIVE]** Code widths are properties of the **genome**, not of the configuration
being decoded:

```
    n_c = 12     addresses  d_max = 4096   channel-like units
    n_v = 14     addresses  V_max = 16384  vocabulary items
    n_L =  5     addresses  32 distinguishable relative depths
    n_M =  4     addresses  16 matrix roles (all 16 reachable, §04.2)
```

**[NORMATIVE — `n_c` is 12, not 10, and the binding axis is not `d_model`.]**
An earlier draft of this section set `n_c = 10` for `d_max = 1024`, reasoning
about the residual stream. One code table is shared by *every* channel-like
axis (`AXIS_KIND`'s `chan` and `inner`), and the widest is the FFN's hidden
axis: `d_ff = 2048` at M1, which `MENU_F`'s top entry may double to **4096**.
At `n_c = 10` a genome that decoded `d_ff = 2048` would have raised, so the
architecture head's own menu fixes the width. The cost is `2 d_h = 64` genome
reals per bit, exactly as `H log N` says, and it is paid deliberately.

Likewise "12 used": all 16 roles are reachable. Roles 0–3 are shared between
the two operators and `SINK`/`DELTA_BETA` are exclusive to one each (§04.2),
so no single *layer* decodes all 16 — but the union over the operator choices
is the whole table, and a role no architecture could reach would be dead
genome.

A configuration with `d < d_max` uses a **subset** of the channel code space, never a
narrower code. This is what makes §2.7's transfer claim meaningful: the same genome
decodes at every width without changing shape.

**[NORMATIVE]** Decoding a configuration wider than `2^n_c` channels, or with more than
`2^n_v` vocabulary items, **raises**. It does not truncate, wrap, or silently collide.
That boundary is precisely where `H log N` says the genome must grow by `d_h` reals per
added bit, and growing it is a deliberate act.

## 2.3 Why the naive constructions fail

Let `kappa_d : {0..d-1} -> {0..2^{n_c}-1}` be the injection from a configuration's
channel index to its code index. Three obvious choices, all wrong:

**Prefix, `kappa_d(i) = i`.** At `d = 128`, `n_c = 10`, the top 3 bits are `0` for every
channel. Those three columns of `W_row` receive no gradient and no selection pressure
during evolution, and are therefore arbitrary noise when a wider configuration first
exercises them. The transfer claim would be vacuous.

**Stride, `kappa_d(i) = i * 2^{n_c}/d`.** The same defect with the bits reversed: the low
3 bits are `0` for every channel.

**Bit-reversal, `kappa_d(i) = bitrev(i)`.** Also the same defect. Any *linear* map from
`p = log2 d` varying input bits into `n_c` output bits has an image lying in a
`p`-dimensional affine subcube; `n_c - p` bit positions are constant by construction.

**[NORMATIVE]** The requirement is stronger than "every bit varies somewhere". The genome
is trained on `d` codes and must generalise to `2^{n_c}` of them, so the training codes
must be a **well-spread sample of the Hamming cube**, not a face of it.

## 2.4 The construction: a seeded bijection, used as a prefix

**[NORMATIVE]** Fix a permutation `pi_c` of `{0, ..., 2^{n_c} - 1}` drawn once from a
generator seeded by `seed_codes` (a genome-level constant, §03.5, recorded in the run
manifest and in the genome hash). Then

```
    kappa_d(i)  =  pi_c(i)        for i = 0 .. d-1
    b_i         =  2 * bits_{n_c}( kappa_d(i) ) - 1     in {-1,+1}^{n_c}
```

and identically for tokens with `pi_v` over `{0, ..., 2^{n_v} - 1}`.

This construction has exactly the three properties required, and they are the acceptance
tests of §2.8:

**(P1) Nesting.** `image(kappa_d) = pi_c({0..d-1}) subset pi_c({0..d'-1}) = image(kappa_{d'})`
for every `d <= d'`. Channel `i` has the **same code at every width**. A genome evolved at
`d = 128` and decoded at `d = 512` finds its first 128 channels exactly where it left
them; the remaining 384 are new identities it has never seen, which is the honest form of
the test.

**(P2) Spread.** The image is a uniformly random `d`-subset of the cube, so each bit
position is `+1` for a `Binomial(d, 1/2)` share of channels. §2.8 tests per-bit balance
and minimum pairwise Hamming distance against the values expected of a random subset.

**(P3) Determinism.** `pi_c` is a pure function of `seed_codes` and `n_c`; the code table
is reproducible on any machine and is covered by the genome hash.

**[NORMATIVE]** `pi_c` is **not** part of `Gamma` and is never mutated. It is a fixed
coordinate system, like the choice of basis, shared by every genome in every generation.
Mutating it would make genomes from different generations incomparable.

## 2.5 Layer codes: Gray over relative depth

**[NORMATIVE]** Depth has topology, and the quantity that transfers across configurations
of different depth is *relative* depth, not absolute index:

```
    q_l  =  round( l * (2^{n_L} - 1) / (L - 1) )      in {0, ..., 2^{n_L}-1}     (L >= 2)
    q_0  =  0                                                                    (L  = 1)
    r_l  =  2 * bits_{n_L}( q_l XOR (q_l >> 1) ) - 1
```

The Gray transform makes adjacent depths differ in one bit, so a decoder smooth in its
input is smooth in depth — which is the prior we actually hold about transformers, whose
layers vary gradually with depth.

**[NORMATIVE — the reason this is right, not merely defensible]** Under this coding the
first layer of an `L = 4` phenotype and the first layer of an `L = 12` phenotype receive
the *same* code, as do the last of each. The genome learns "what an early layer looks
like", never "what layer 3 looks like", and depth transfer (§2.7) is therefore
interpolation between depths it has seen rather than extrapolation past them. An absolute
layer index would make the `L = 4 -> L = 12` transfer a pure extrapolation and the claim
of §2.7 unsupportable.

## 2.6 Role codes and the vector flag

**[NORMATIVE]** Matrix roles (§04.2) are a nominal set of at most 16; `t_m` is the plain
binary `{-1,+1}` code of the role's index.

**[NORMATIVE]** Some decoded objects are vectors, not matrices — RMSNorm gains, the
logit-scale, the per-head sink scalars (§04.2). The column argument then has no referent.
Rather than leave `b_j` undefined or set it to zeros (which would break the `{-1,+1}`
invariant and make the "no column" case indistinguishable from a real channel whose code
happened to be near zero), the column side carries **one extra bit**:

```
    b_j^ext  =  [ b_j        ; +1 ]     for a matrix, column j
    b_j^ext  =  [ 0_{n_c}    ; -1 ]     for a vector
```

The flag bit is `-1` exactly in the vector case, and the `n_c` code bits are then
genuinely unused rather than ambiguously used. Cost: `d_h` reals of genome
(`n_c + 1` input columns instead of `n_c` in `W_col`).

## 2.7 What the construction buys: scale-free decode

**[NORMATIVE — the claim, registered before it is tested]** Because the codes at every
width are drawn from one fixed table, and because `|Gamma|` depends on `n_c, n_v, n_L,
n_M, d_h, d_z` and **not** on `d_model`, `L` or `V`, one genome decodes phenotypes of any
size up to the addressable maximum with no change to its shape.

`KG-SCALE` (§09) tests it: a genome evolved on a 1.53M-parameter phenotype, decoded
zero-shot at 49.9M, must produce a model that trains measurably better than one decoded
from a random genome of the same shape. If it does not, the claim is false and is reported
as false.

## 2.9 The token axis takes a Gray code of frequency rank

**[NORMATIVE — this section corrects §2.1, and the correction was found by
measurement rather than by argument.]**

§2.1 placed the token axis in the "no topology" class and gave it §2.4's seeded
bijection, on the grounds that *"token id order is an artifact of BPE merge
order"*. **BPE merges the most frequent pair first, so merge order is frequency
order.** That is not an artifact; it is information, and frequency rank has
exactly the topology §2.1's own rule asks about. The rule was right and its
application to this axis was wrong.

Measured on `data/ts_4096` (4,096 types, 20M tokens):

```
    Spearman(token id, frequency rank)                    +0.292   p ~ 1e-81
    R^2 predicting log frequency from log(token id)         0.035
    R^2 predicting log frequency from §2.4's bijection      0.004   (chance)
    R^2 predicting log frequency from §2.9's code           0.488
```

**[NORMATIVE]** The construction is §2.5's, with frequency in place of depth:

```
    rank(v) = 0 for the most frequent type, V-1 for the least
    q_v     = round( rank(v) * (2^n_v - 1) / (V - 1) )
    e_v     = 2 * bits( q_v XOR (q_v >> 1) ) - 1
```

**[NORMATIVE — why it matters, and it is not a small effect.]** Held-out CE at
M-pico, 300 steps, `lr = 0.09`, **plain AdamW** with no decoded plasticity,
12 seed-paired lifetimes per arm:

```
    §2.4 seeded bijection      5.5374 +/- 0.169
    §2.9 frequency rank        5.4375 +/- 0.131
    difference                +0.0999 nats   95% BCa [+0.0288, +0.2647]
```

The interval clears zero. The gain is in the **decoded weights** -- chiefly the
embedding, whose rows are decoded from the token code -- and not in any
learning rule: the comparison uses the same optimiser on both sides.

**[NORMATIVE — the second reason, which is about §2.7 rather than about loss.]**
Relative rank transfers where absolute identity cannot. Rank 0 means "the most
frequent type" at every vocabulary size, so a genome evolved at `V = 4096`
decodes the same relationship at `V = 8192`, exactly as §2.5's relative depth
makes an `L = 4` genome meaningful at `L = 12`. §2.4's bijection gave a token
its identity from a random draw, which carries across scales only by accident.

**[NORMATIVE — ties are broken by token id, and they are not rare.]**
`rank` is a total order and the counts are not. Measured on `data/ts_4096`,
counting all 471,987,251 training tokens: **508 of 4,096 types share a
count** with at least one other type, and the largest tie group is the
**173 types that never occur in the shard at all** — merges the tokenizer
holds and this corpus never exercises. No counting window recovers them; at
20M tokens the figure is 243, so a bounded count makes the tail worse and
buys nothing (the full pass costs 1.6 s against a lifetime's 15).

`torch.argsort` leaves the order within a tie unspecified unless
`stable=True`, and on exactly this input its output **differs** from the
stable one. The first implementation therefore produced a code that was a
function of sort internals rather than of the corpus — not reproducible
across versions or devices, which this section's own reproducibility
requirement and §2.8's T6 both forbid. The order is fixed twice over: a
stable sort of the negated counts, so that ties resolve by **ascending token
id**. Counts are Laplace-smoothed (`+1`) so an unseen type still receives a
rank rather than a division by zero downstream.

**[NORMATIVE — this is the default, not an option.]** Every entry point that
runs a lifetime installs it before the first step, from its own training
shard. §2.4's bijection remains reachable **only** as the control that
measured the difference above; it is not a tuning choice. The counts'
`blake2b` digest, the number of tokens counted and the number of unseen
types are written to the run manifest, because a code derived from data must
be reproducible from that data.

**[NORMATIVE — the code depends on the corpus, and that is admissible.]**
`rank` is computed from the training shard. This is exactly as legitimate as
fitting the tokenizer on that shard, which every configuration already does: it
is an **innate coordinate system shaped by the environment the lineage evolved
in**, fixed before any lifetime, identical for every genome in every generation,
and costing zero genome reals. It is recorded in the run manifest and covered by
the corpus hash, because a code derived from data must be reproducible from it.
`rank` is **not** part of `Gamma` and is never mutated (§2.4's rule, unchanged).

**[NORMATIVE — the spread test does not apply to a structured code.]** §2.8's T4
bounds per-bit balance by `3/sqrt(n)`, which is the right test for §2.4's
*random* construction and the wrong one here: a frequency-ordered code is
deliberately correlated with rank, which is the entire point of it. Its worst
bit balance is **0.333** against the bijection's 0.026. T4 therefore applies to
the bijection, and a structured code is held to `STRUCTURED_BALANCE_MAX = 0.40`
together with the requirements that every bit still varies, every code is still
unique, and adjacent ranks still differ in one bit.

**[NORMATIVE — what this does *not* license.]** The channel axis keeps §2.4's
bijection. A residual channel has no counterpart of frequency rank: nothing
orders channels before training, and imposing an order would be a smoothness
prior over an arbitrary permutation, which is the defect §2.1 correctly warns
about. The correction here is specific to an axis that turned out to have a
topology, not a general licence to order things.

## 2.8 Acceptance tests

**[NORMATIVE]** `gnomon/genome/codes.py` is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| T1 | codomain | every code entry is exactly `-1.0` or `+1.0` |
| T2 | injectivity | `kappa_d` is injective for every `d` in `{2^k}` up to `d_max` |
| T3 | **nesting (P1)** | for all `d <= d'` and all `i < d`: `b_i(d) == b_i(d')`, elementwise, for channels, tokens and roles |
| T4 | **spread (P2)** | per-bit mean over the image lies in `[-3s, +3s]` with `s = 1/sqrt(d)`; minimum pairwise Hamming distance is at least 1 and the mean is within 10% of `n_c/2` |
| T5 | **prefix defect absent** | no bit position is constant across the image, for every `d >= 16`; a *planted* prefix coding fails this test (positive control) |
| T6 | determinism (P3) | two constructions from the same `seed_codes` are bit-identical; different seeds differ |
| T7 | Gray adjacency | `r_l` and `r_{l+1}` differ in exactly one bit whenever `q_{l+1} = q_l + 1` |
| T8 | depth relativity | for `L = 4` and `L = 12`: `r_0` agrees, `r_{L-1}` agrees |
| T9 | overflow | decoding `d > 2^{n_c}` or `V > 2^{n_v}` raises, and the message names the bit width that must grow |
| T10 | zero cost | the module holds no `torch.nn.Parameter` and no persistent buffer whose size depends on `d_model` other than the code table itself, which is derived |
| T11 | **§2.9 carries frequency** | a linear probe recovers `R^2 >= 0.3` of log-frequency from the frequency-ordered token code, and `<= 0.05` from §2.4's bijection — the second half is the control, without which T11 would pass for any code at all |
| T12 | §2.9 is still a code | every bit varies, every code is unique, per-bit balance is within `STRUCTURED_BALANCE_MAX`, and the code is **smooth in rank**: adjacent ranks differ in fewer than half as many bits as a random code's neighbours. Not *exactly one* bit — that holds only when `V = 2^n_v`, so that consecutive ranks map to consecutive `q`; at `V = 512` with 14 bits the step is 32 and adjacent ranks differ in 2 to 3 bits against a random code's 7 |
| T13 | §2.9 is rank-relative | the most frequent type receives the same code at `V = 4096` and `V = 8192`, as does the least |
| T14 | **§2.9 is the default, and deterministic under ties** | a `LifetimeRunner` installs the frequency code with no argument, records its provenance, and discards any compiled graph when the code is switched; `frequency_rank` orders tied types by ascending id and is bit-identical on repetition. Without the first half the construction is tested and unused, which is what it was for the whole of v1.0 |
