# 07 — The architecture is decoded

C3 of the constitution: the computation graph is an output of the genome, so
**mutating `Gamma` changes the architecture** and there is no separate
architecture search to run. This section specifies the head that does it.

## 7.1 What is decoded, and what is not

**[NORMATIVE]** `Gamma_a` decodes, per model and per layer:

| quantity | domain | granularity |
|---|---|---|
| `L`, the layer count | `{2, 3, 4, 6, 8, 12}` capped at the configuration's `L_max` | per model |
| operator | `{attn, delta}` | per layer |
| `n_h`, heads | `{n_h_max/4, n_h_max/2, n_h_max, 2 n_h_max}` | per layer |
| `d_ff`, FFN width | `{1, 2, 4, 8} x d_model` | per layer |
| `window` | `{T/4, T/2, T, infinity}` | per layer |

**[NORMATIVE — what is *not* decoded, and why.]** `d_model` and the vocabulary
are **not**. They are the configuration's body plan — the compute envelope the
experiment is run inside — and letting the genome move `d_model` would make a
generation's genomes incomparable in a way that has nothing to do with the
question being asked. Everything the genome does change is a choice *within* a
fixed residual width, which is the level at which architecture search is
interesting and at which the decoder's identities remain meaningful across
genomes (a channel's code does not move when a layer's head count does).

**[NORMATIVE — the architecture is innate.]** `Gamma_a` reads only `r_l` (the
relative-depth code, §02.5) and a global genome vector `g_glob subset Gamma_a`.
It takes **no acquired input**: a lifetime cannot change the architecture it was
born with, which is what makes the two timescales distinct rather than nested.

## 7.2 The head

**[NORMATIVE]**

```
    a1        = tanh( A_1 [ r_l ; g_glob ] + a_c1 )      in R^{d_a}
    a2        = tanh( A_2 a1 + a_c2 )                    in R^{d_a}
    op_l      = argmax( A_op a2 + b_op )                 in {attn, delta}
    heads_l   = MENU_H [ argmax( A_h a2 + b_h ) ]
    ffn_l     = MENU_F [ argmax( A_f a2 + b_f ) ]
    win_l     = MENU_W [ argmax( A_w a2 + b_w ) ]
    L         = MENU_L [ argmax( A_d g_glob + b_d ) ]
```

with `d_a = 16` and `g_glob in R^{8}`. Two layers for the same reason §03.1
gives: one is additive in `(r_l, g_glob)` and would make every layer's
architecture a monotone function of depth, which is a strictly smaller search
space than the one being claimed.

| tensor | shape | reals |
|---|---|---|
| `A_1`, `a_c1` | `d_a x (n_L + 8)`, `d_a` | 208 + 16 |
| `A_2`, `a_c2` | `d_a x d_a`, `d_a` | 256 + 16 |
| `A_op`, `b_op` | `2 x d_a`, `2` | 32 + 2 |
| `A_h`, `b_h` | `4 x d_a`, `4` | 64 + 4 |
| `A_f`, `b_f` | `4 x d_a`, `4` | 64 + 4 |
| `A_w`, `b_w` | `4 x d_a`, `4` | 64 + 4 |
| `A_d`, `b_d` | `6 x 8`, `6` | 48 + 6 |
| `g_glob` | `8` | 8 |
| | **`\|Gamma_a\|`** | **796** |

Total genome `|Gamma| = 4,483 + 796 = 5,279` reals, plus §10.4's 27 self-adapted
mutation strengths — **5,306** reals, 20.7 KB in float32 — against 49,926,705
decoded parameters at M1: **9,458x**. (`10,324x` here and `10,830x` in §03.5
were both arithmetic on a phenotype count that was never recomputed; `G-SPECNUM`
now asserts every one of these as an equality.)

## 7.3 Totality

**[NORMATIVE]** `DECODE_ARCH` is **total**: every point in `R^{|Gamma_a|}` maps
to a valid architecture. `argmax` over a fixed menu cannot fail, cannot produce
an out-of-range value, and needs no repair operator, no rejection sampling and
no constraint handling in the search. This is a deliberate choice over the
obvious alternative of decoding integers directly and clamping them: a search
whose mutations can produce invalid individuals spends its budget on repair, and
a repair operator is a second, undocumented, un-evolved piece of design.

**[NORMATIVE]** Menus are clamped to the configuration at decode time — `L` to
`L_max`, `n_h` so that `n_h*d_head <= 2^n_c` and `n_h % n_kv == 0`, `d_ff` to
`2^n_c` — and clamping is by **saturation to the nearest admissible menu
entry**, never by rejection. §7.6's T4 asserts totality by decoding 10,000
random genomes at every configuration and requiring every one to build and run.

## 7.4 Behaviour descriptors

**[NORMATIVE]** MAP-Elites (§08) places a genome by two descriptors, both read
off `Gamma_a` **without running a lifetime**:

```
    B1 = (# attention layers) / L                        in [0, 1]
    B2 = log10( decoded parameter count )
```

`B1` is the architectural axis the operator choice moves. `B2` is the size axis,
and making it a **descriptor rather than a constraint** is the substantive
decision here: fitness is held-out loss, so a larger architecture wins on
fitness alone, and left unconstrained the search would collapse onto the largest
menu entry. As a descriptor it instead makes MAP-Elites illuminate the
**size-quality frontier**, which is the object this project is actually about —
each niche's elite is the best architecture *at that size*, and the frontier is
the result rather than a single number.

**[NORMATIVE]** Descriptors computable before evaluation is not an incidental
convenience: it lets §08 report archive coverage over *attempted* niches and
lets a batch be composed to fill empty ones.

**[NORMATIVE]** Decoded effective rank (§03.8 D5) is recorded for every elite as
a **diagnostic**, not a descriptor. It is a property of `Gamma_w` rather than of
the architecture, and `KG-RANK` (§09) fires if the archive's elites show it
collapsing.

## 7.5 Mutation and crossover

**[NORMATIVE]** Variation acts on `Gamma = (Gamma_w, Gamma_a)` as one vector.

```
    MUTATE(Gamma):  for each tensor t:  t' = t + sigma_t * N(0, I) * m
                    sigma_t = sigma_rel * rms(t)                    (scale-relative)
                    m ~ Bernoulli(p_mut) elementwise                (sparse)
    sigma_rel = 0.15,  p_mut = 0.25
```

**[NORMATIVE — the perturbation is scale-relative, and it is load-bearing from
generation 0, not only after drift.]** This section has now said three different
things about the same measurement, and the third is the measured one.

The first draft said the tensors "differ in scale by more than an order of
magnitude" at initialisation. The second corrected that to "0.17 to 0.20 — a
factor of 1.55", concluded that scale-relative sigma buys nothing yet, and
pointed T10 at an evolved genome instead. **Both were wrong, and the second was
wrong because it was written before §03.6's sweep moved `log_gain` off zero.**
Measured over 5 seeds at the default shape:

| tensor | rms at generation 0 |
|---|---|
| `log_gain` | **1.3863** ( = `log 4.0`, §03.6's measured default) |
| `W_ctx` | 0.2360 |
| `W_row` | 0.1941 |
| `W_col` | 0.1910 |
| `W_2` | 0.1768 |
| `W_tok` | 0.1749 |
| `w_out` | 0.1596 |
| `c1`, `c2`, `o_0`, `sig_0`, `w_sig` | 0 |
| `Gamma_a`'s nonzero tensors | 0.2036 .. 0.5178 |

Nonzero rms spans **0.137 to 1.386** across seeds: a factor of **10.1**, not
1.55. An absolute `sigma = 0.15` would therefore perturb `w_out` by 94% of its
own scale and `log_gain` by 11% of its, at generation 0, before any drift. The
mechanism is doing its work from the first mutation.

`log_gain` is the outlier and it is not an artifact: §03.6 measured `s = 4` and
set `log_gain = log 4 = 1.386`, so a scalar whose valid range is `[log 2, log 8]
= [0.69, 2.08]` sits at rms 1.386 while every matrix sits near 0.18. Under
scale-relative sigma one mutation moves it by `0.15 x 1.386 = 0.21`, about a
seventh of its admissible range — which is a deliberate step for an evolved
hyperparameter and would be a 0.15 step, three times smaller in relative terms,
under an absolute sigma.

The argument for an *evolved* genome is unchanged and still holds: after tens of
generations the tensors drift further apart, and an absolute `sigma` perturbs a
shrunken tensor out of existence while leaving a grown one untouched. §7.6's T10
tests that regime, and `G-SPECNUM` pins the generation-0 spread above so this
section cannot drift from it a fourth time.

`rms(t)` is taken over the tensor, with a floor of `SIGMA_FLOOR = 0.1` below
which `sigma` stops tracking scale, so that a zero-initialised tensor — `c1`,
`c2`, `w_sig`, `sig_0`, `o_0`, and every bias in `Gamma_a` — can be reached at
all. At `0.1` such a tensor moves by `0.015` per mutation and reaches `O(0.1)`
over the tens of generations §08.3 budgets.

**[NORMATIVE — the floor must be well below the tensors it is protecting.]** It
was `1.0` in the first implementation. Every genome tensor has `rms` below 1 at
initialisation, so that floor applied to **all** of them and the perturbation
was absolute throughout: the floor swallowed the very scaling it existed to
protect, and the module did the opposite of what this section says. Found by
`G-MUTATE` M17, once T10 was itself corrected to divide by `rms` rather than by
`max(rms, 1)` — the test had the same defect as the code, which is exactly how a
test comes to be unable to fail.

```
    CROSSOVER(Gamma_1, Gamma_2):  per *tensor*, take one parent's whole tensor
                                  with probability 1/2
```

**[NORMATIVE — crossover is per tensor, not per element.]** The decoder's
tensors are functionally coupled: `W_row` and `W_col` feed one shared `tanh`,
and blending them elementwise between two genomes that organised their hidden
space differently produces a decoder that is neither parent's, in the way that
matters. Per-tensor crossover preserves each tensor's internal coherence, which
is the standard argument for a linkage-respecting operator.

## 7.6 Acceptance tests

**[NORMATIVE]** The architecture head is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| T1 | shape | `\|Gamma_a\| = 796` exactly; `\|Gamma\| = 5,279` |
| T2 | innate | `DECODE_ARCH` is invariant to `Z`: perturbing the soma arbitrarily changes nothing |
| T3 | determinism | one `Gamma_a` gives one architecture, on CPU and CUDA |
| T4 | **totality** | 10,000 random `Gamma_a` at each of M-pico / M-nano / M1 all decode to an architecture that builds and completes a forward pass |
| T5 | clamping | a genome whose menu choice exceeds the configuration saturates rather than raising, and the saturated value is admissible |
| T6 | expressivity | over 1,000 random genomes, every operator, every head count, every FFN width and every depth in the menus is reached at least once |
| T7 | **mutation changes the graph** | over 200 mutations of one genome at `sigma_rel = 0.15`, the decoded architecture differs in at least 10% of cases. A search in which mutation never changes the graph is not architecture search |
| T8 | one-layer control | a planted single-hidden-layer head makes every layer's architecture a monotone function of depth, and fails T6's coverage |
| T9 | descriptors | `B1` and `B2` are computed without a lifetime, are in range, and agree with the architecture actually built |
| T10 | mutation is scale-relative | mutating `W_2` (rms ~ 0.18) and `W_tok` (rms ~ 0.17) changes both by comparable *relative* amounts; an absolute-sigma control fails this |
| T11 | crossover | a child's every tensor is bit-identical to one parent's; over 200 crossovers both parents contribute |
