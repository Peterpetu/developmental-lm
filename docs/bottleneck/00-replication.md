# The genomic bottleneck on a language model — a faithful replication

> **Context (October 2026).** The headline birth result (2.863 / 2.836 nats, two seeds) is for GN50-E, which keeps the vocabulary table uncompressed: 2,097,153 of the 2,217,620 genome parameters are that table, so the overall ratio is 5.9× and the 107× figure applies to the compressed layers only. The fully compressed arms (GN30, GN50) are reported in the same run directory.

**Status: normative for `bottleneck/` and `scripts/bottleneck_loop.py`; pre-registered before
any run (2026-09-07).** The source is Shuvaev, Lachi, Koulakov & Zador, *Encoding innate ability
through a genomic bottleneck*, PNAS 121(38):e2409160121 (2024), read end to end and compared
with this repository's three prior attempts in `docs/tempo/11-evidence-ledger.md` §P-D.44.
Page numbers below are the publisher PDF's.

## 0. Why this exists

Three designs in this repository (GNOMON, ANLAGE, TEMPO) took the paper as their point of
departure and none ran its method. §P-D.44 identifies three departures: training the phenotype
*through* the decoder (which the paper tried and abandoned, bioRxiv v2); a fixed-`ε` pull at 1×
compute instead of the paper's annealed regeneration (ANLAGE R4b); and learnable identity
vectors where the paper uses fixed codes. Every negative about "the PNAS loop on language" in
this repository is therefore about a different procedure. This document specifies the paper's
procedure as printed, on the 13M stage-0 language model whose seed noise is already measured,
with the readings written before the numbers exist.

## 1. The g-network (paper pp. 2–3, 10; Tables 1–2)

For every weight matrix `W ∈ R^{n_out × n_in}` of the phenotype there is one g-network
`g_θ : {−1,+1}^{b_row + b_col} → R`,

```
    Ŵ_ij = g_θ( [ c_row(i) ; c_col(j) ] ),          g_θ = Linear(b, H) ∘ ReLU ∘ Linear(H, H₂) ∘ ReLU ∘ Linear(H₂, 1)     (1.1)
```

with `c_row(i)` the code of unit `i` on the matrix's row axis and `c_col(j)` on its column axis
(§2). The paper's structure is `(20 − H − 10 − 1)` per layer with `H ∈ {5, 20, 30, 50}` (Table
1, Table 2); here `b = b_row + b_col` depends on the axes and `H₂ = 10` at the paper's `H`. The
activation is not stated in the paper; ReLU is used and recorded as a choice. **One g-network per
matrix**, as the paper has one per layer: layer identity is implicit, so no layer code enters.

Vector roles (`NORM_MIX`, `NORM_FFN`, `NORM_FINAL`, each `R^{d}`) are compressed by a single-axis
g-network `g_θ(c_row(i))` of structure `(b_row − H₂ − 1)`, matching the paper's bias networks
(`10 − 5 − 1`, Table 1). `LOGIT_SCALE ∈ R^1` is **not** compressed, by the paper's own rule: "the
ten biases for the output layer were not compressed since the corresponding g-network would
include more than ten parameters" (p. 10).

Every g-network's last layer is initialised at `0.01×` the default scale so that an untrained
genome decodes to weights of small magnitude rather than `O(1)`. This is a conditioning choice
with no effect on the loop (at generation 0, `ε = 1` and the decode is not used) and is recorded.

## 2. Codes (paper p. 10)

The paper's rule: *a Gray code on an axis that has a topology; a plain binary index on an axis
that does not* ("the neurons in the hidden layer were represented by simple binary codes ... since
their order is of no particular importance"). A unit on an axis of `N` units carries
`b = ⌈log₂ N⌉` bits, the paper's "`log₂ N` binary tags" (p. 9), in `{−1, +1}`.

| axis | units at stage 0 | bits | code | why |
|---|---|---|---|---|
| `TOK` | 8,192 | 13 | **Gray over frequency rank** | the one topology a vocabulary has before training; GNOMON's largest measured positive (`docs/anlage/00` §0.2) |
| `RES`, `INNER`, `HEAD`, `KV`, `GATE`, `TAP` | 256, 1024, 256, 128, 8, 4 | 8, 10, 8, 7, 3, 2 | plain binary index | no topology; the paper's hidden-unit case |

Gray over rank: with `rank(v) = 0` the most frequent type and `hi = 2^b − 1`,
`q_v = round( rank(v) · hi / (V − 1) )`, `code_v = 2·bits_b( q_v XOR (q_v >> 1) ) − 1`. Adjacent
ranks differ in one bit (I2 below). Frequency ranks come from the training shard's counts with
Laplace `+1` (`gnomon/data/stream.py::token_counts`), as every prior design here used them.

**Deviation from the paper, stated:** the paper's Gray codes are over *two-dimensional pixel
position*, a genuine spatial topology under which a weight is a smooth function of position
(its recovered filters are Gabor-like, Fig. 2F). Frequency rank is one-dimensional and is not a
topology of the *function* the weight computes; it is a prior that similar-frequency tokens get
similar rows. §P-D.44 names this as the structural reason the method may not carry. It is used
because it is the best available and because it measured positive.

## 3. Intermittent training (paper pp. 3, 10–11; Algorithm 1; Eq. 3)

Generations `n = 1, …, N`. Each:

```
    (a) train the phenotype freely for T_gen tokens (Muon on matrices, AdamW on vectors and
        embeddings, the R1 recipe; one learning-rate schedule over the whole run, not one per
        generation — see §3.1)                                                                  →  W_n
    (b) refit every g-network by MSE regression onto W_n, warm-started, on a random
        fraction f of each matrix's entries per step, n_refit Adam steps                       →  θ_n
    (c) decode                                                                                 →  Ŵ_n = g_{θ_n}(codes)
    (d) regenerate        W ← ε(n) W_n + (1 − ε(n)) Ŵ_n,       ε(n) = exp(−n / λ),   λ = 20     (3.1) = paper Eq. 3
```

`ε(1) = 0.95`, `ε(20) = 0.37`, `ε(60) = 0.05`, `ε(100) = 0.007`: after about sixty generations
the phenotype is regenerated from the genome before every training burst, which is the paper's
mechanism for forcing the phenotype onto the decodable set. The paper uses `N = 500`
generations. Here `N = 200` by default (`ε(200) = 4.5·10⁻⁵`; the regime is fully entered by
`n ≈ 100`), with `N` a parameter and the paper's value recorded.

The p-network's optimiser state is **not** reset at regeneration: the paper does not mention
resetting it, and Algorithm 1 holds one `p_optimizer` across generations.

### 3.1 The learning-rate schedule — corrected 2026-09-07 15:05

The paper trains the p-network at a constant learning rate per generation (SGD `0.05`,
momentum `0.9` for CIFAR, p. 10). The R1 recipe is warm-up / constant / decay-to-zero over a
budget. Running one R1 schedule *per generation* would re-warm and decay 200 times, which TEMPO
measured to produce a saw-tooth that dominated every per-episode measurement (§P-D.25,
§P-D.28).

**As first written**, one R1 schedule over the whole run was used, "constant for its middle
78 %". **That was wrong in a way the first GN30 run measured** (11 §P-D.46): the recipe's decay
begins at `p_dec = 0.8` of the budget — generation 160 of 200 — and from generation ~165 the
loop collapsed: the residual fell to `0.04` while the decode returned to null and the trained
phenotype degraded from `3.3` to `4.5`. With the learning rate gone, a burst can no longer move
the phenotype off its decode, refit-and-regenerate becomes self-referential, and Adam on a
vanishing residual walks the genome away from the trained manifold with the phenotype following.
The paper has no decay and cannot exhibit this.

**As now specified:** warm-up (`p_warm = 0.02`) and then a **constant** rate for the rest of the
run (`p_dec = 0.999`, i.e. no decay phase), the paper's regime. Every arm, the tabula-rasa
control included, uses it. The GN30 run under the decayed schedule is kept as
`runs/bottleneck/GN30-tinystories-s0-seed0-v1-decay/` and its plateau (generations 70–165) is
the reading §P-D.46 reports; its end is the schedule's artefact.

### 3.2 The state transfer reads

The paper reports the *best* at-birth performance. The end of a run is not that state, and
under a decayed schedule it was the worst. At every generation whose at-birth CE is the best so
far, the loop writes `genome_best.pt` and `phenotype_best.pt` — the genome and the trained
weights `W_n` it was fitted to, before regeneration — plus `best.json`. §9's transfer reads these
(`--which best`, the default) and records which state it used; `--which final` is the run's end.

## 4. Compression

`ratio = n_params(phenotype) / n_params(all g-networks + the uncompressed scalars)`, the paper's
definition (Tables 1–2: p-network parameters over g-network parameters). Arms:

| arm | `H` | `H₂` | why |
|---|---|---|---|
| `GN30` | 30 | 10 | the paper's MNIST/CIFAR structure |
| `GN50` | 50 | 10 | the paper's best CIFAR structure |
| `mild` | chosen for `≈ 5×` | `H / 8` | the paper's HalfCheetah ratio (`4.9×`), its hardest task — never tested here; `H₂ = 10` would cap a wide g-network at rank 10 and make "5×" a rank-10 compression in disguise |

The ratio each arm actually reaches is computed at build time and recorded, not assumed.

## 5. Controls and metrics

* **Tabula rasa**: the free phenotype trained for the same `N · T_gen` tokens under the same
  recipe, seed and window order (CRN), no genome. The paper's "No GN" curve.
* **Shuffle**: the decode with the rows of every matrix permuted (paper Fig. 2D, dashed), at
  birth; separates "learned the weight distribution" from "learned the wiring". Every 10th
  generation and at the end.
* **Random genome**: the decode at generation 0, before any fit.
* **At-birth CE** of `Ŵ_n` on the v3 test set at every generation — the paper's claim.
* **Regeneration jolt**: the phenotype's test CE immediately before and after step (d).
* **Lifetime**: the regenerated phenotype's final test CE against the tabula-rasa arm's at the
  same training tokens — this repository's criterion.
* Per-matrix relative residual `‖Ŵ − W‖² / ‖W‖²` at every refit.

`σ_e = 0.0029` nats on the v3 sets (`runs/anlage/noise-tinystories-s0/noise.json`, 3 df);
`L_null = ln 8192 = 9.011`; the trained R1 mean is `1.809`.

## 6. Pre-registered readings

| reading | condition |
|---|---|
| **works** | at-birth `CE(Ŵ_N) < 6.0` at any arm — three nats of innate competence, a third of the way from null to trained |
| **fails** | at-birth `CE(Ŵ_N) > 8.5` at every arm — what one regression already gave (ANLAGE R4: 8.53) |
| between | reported as what it is, with the shuffle control deciding whether it is wiring or distribution |
| lifetime | `CE_regen − CE_free` at matched training tokens: the paper predicts `≈ 0` (Fig. 3E, "hot-started"); worse by `> 3σ_e = 0.009` is the measured cost of the constraint |

Whatever the outcome, the ledger records it with the artefact, and the next step is decided by
the reading above and not after it.

## 7. Budget and compute accounting

`T_gen = 10⁶` tokens (61 steps at `B = 32, T = 512`), `N = 200` → `2·10⁸` training tokens per
arm, about 50 minutes of card at R1's measured 69k tok/s, plus 200 refits and decodes (§8).
The tabula-rasa arm costs the same training. Refit and decode compute is reported per generation
and **not** FLOP-matched: the paper does not match it, and the at-birth claim does not depend on
it. The lifetime comparison is at matched *training* tokens with the overhead stated beside it.

## 8. Acceptance tests (`tests_bottleneck/`)

| # | property |
|---|---|
| I1 | every axis's codes are distinct rows, `b = ⌈log₂ N⌉` bits, entries in `{−1, +1}` |
| I2 | Gray over rank: adjacent ranks differ in exactly one bit; rank 0 is the all-`−1` code |
| I3 | a g-network's decode has the matrix's shape and is finite; the single-axis form likewise |
| I4 | `LOGIT_SCALE` is copied through the decode unchanged |
| I5 | regression capacity: a g-network with `H` above a matrix's rank fits a random low-rank matrix to relative residual `< 10⁻²` |
| I6 | regeneration: `ε = 1` leaves every weight bit-identical; `ε = 0` makes every compressed weight equal the decode |
| I7 | `ε(n) = exp(−n/λ)` exactly; the hook fires once per generation at the right token boundary |
| I8 | the shuffle control preserves each matrix's multiset of rows |
| I9 | the compression ratio equals the parameter count of the phenotype over that of the genome, and each arm's ratio is recorded |
| I10 | the decode never runs inside a training step (G-NOFREE-INVERSE): it runs in the hook only |
| I11 | `fisher_diag` equals the mean over micro-batches of the squared gradients computed by hand, has every key's shape, is non-negative, and leaves the gradients `None`; the hook records the Fisher residual from a tensor or a callable probe (§11.1) |
| I12 | `F̃` at `α = 1` is uniform and a zero Fisher gives uniform; the sampler draws only the entries with weight and, with no weights, reproduces `randint` bit for bit; the weighted refit reduces the loss (§11.1) |
| I13 | the weighted residual equals the plain one under uniform weights and one entry's own relative error under a one-hot weight (§11.1) |
| I14 | a rejected proposal restores the genome and its optimiser bit for bit and the phenotype is regenerated from the decode in force; before `select_from` every proposal is taken; with neither variant the records carry nothing (§11.2) |
| I15 | a kept role is copied through the decode verbatim, has no g-network, is skipped by the refit and the residuals, is counted in the genome's size and the ratio, and reloads through `load_source` with the same kept set (§11.4) |
| I16 | a ReLU g-network whose second layer is inactive decodes a constant and passes no gradient below it; the leaky one passes gradient; the activation is recorded in the summary, reloads through `load_source`, and the same state decodes differently under the other activation (§11.5) |
| I17 | a rank-coded role decodes to its truncated SVD exactly (rank `≤ r`, closer than a random rank-`r` projection), has no g-network, is skipped by the refit, appears in the residuals, is counted as `r (n_out + n_in)`, cannot also be kept, rejects a rank above the matrix, and reloads through `load_source` (§11.6) |

## 9. Transfer — the claim that matters, tested unconditionally

**Status: normative, pre-registered 2026-09-07 14:10, before any transfer run.** §5's "(c) only
if (a) succeeds" is withdrawn: the paper's most valuable claim is that the bottleneck extracts
structure that *transfers* (Fig. 3G, Fig. 5G–J), and "learns how to learn" is worth a compute
cost that at-birth performance is not. It is tested whatever §6 reads.

### 9.1 The target

The reserved-alphabet verifier stream (`soma/openended/tasks.py`): `BOS prompt SEP answer EOS`
instances, families round-robin at their default parameters, packed into `(T+1)` windows exactly
as the corpus is. Every id lies in the reserved block `[0, 64)` that natural text never emits, so
the phenotype meets them for the first time on the target — a *distant* shift in the paper's
terms (its MNIST→F-MNIST case), which by the paper's own rule predicts no benefit. A fixed
evaluation set of task windows, one stratum, disjoint from the pool by seed.

### 9.2 The arms

From a finished source run (`W_src`, `θ_src`), on the target under one recipe, seed and window
order:

| arm | initialisation | the paper's name |
|---|---|---|
| `random` | fresh | "No GN" |
| `copy-{all,low,high}` | `W_src` on the layer set, fresh elsewhere | "direct transfer" |
| `decode-{all,low,high}` | `g_{θ_src}(codes)` on the layer set | "g-network mediated" |

`low` = layers 0–1 plus the embedding (the paper's "layer 1"); `high` = the last two layers plus
the final norm and logit scale (its "layers 8–9"). The target's own `σ_e` is measured first from
`random` over three seeds (G-NOISE); every contrast is read against `3σ_e` of the target.

### 9.3 Readings, pre-registered

Contrasts are paired differences of target test CE at 25 %, 50 % and 100 % of the target
budget (`5·10⁶` tokens), positive meaning worse than the reference.

| reading | condition | what it would mean |
|---|---|---|
| **T1** | `decode-all − copy-all ≈ 0` (inside `3σ_e`) at every point | the genome carries what the weights carry — the paper's L1-9 |
| **T2** | `decode-low − copy-low < −3σ_e` at 25 % or 50 % | the bottleneck's low layers transfer *better* than copied ones — the paper's one positive, its L1 |
| **T3** | `decode-high − random > 3σ_e` | late layers through the bottleneck hurt — the paper's L8-9 and F-MNIST |
| **META** | `decode-* − random < −3σ_e` at 25 % or 50 %, on a distribution the genome never saw | **faster learning of a new distribution**: learns how to learn |
| hot-start only | `decode − random < 0` at birth, `≈ 0` in slope thereafter | the paper's Fig. 3E — an offset, not a rate |
| negative transfer | `decode − random > 3σ_e` | the F-MNIST outcome |

The `free` source has no decode; its `copy` arms are the multi-task baseline of §10.

## 10. The routine aimed at learning how to learn

**Status: normative, pre-registered 2026-09-07 14:10.** The loop of §3 run on **one** task
learns that task at birth. The same loop with a **fresh task every generation** fits the
genome to *adapted* weights across many tasks, so that what survives regression is what helps
adaptation in general — an initialisation that learns fast. Step (d) with `ε → 0` is then
Reptile-like (Nichol, Achiam & Schulman, arXiv:1803.02999, 2018: move the initialisation toward
the adapted weights) with a lossy channel in between. The paper never ran this; it is the
training routine its own transfer claim points at.

### 10.1 The distribution

Families `PARITY, MODCOUNT, PERM, DYCK, COPY, MQAR` at `n_len ∈ {4, 6, 8, 12}` with one
family-specific difficulty axis each (`m_mod`, `p_grp`, `depth`, `gap`, `P_mqar`), sampled
uniformly by a seeded generator; each generation draws one `(family, params)` and trains on
windows of that task alone. **Held out entirely**: the family `GRID`, and `n_len = 16` for every
family — an unseen family and an unseen length. The evaluation set during training is drawn
from the training distribution; the transfer test of §9 is run on the held-out set.

### 10.2 The arms

| arm | what | control for |
|---|---|---|
| `meta-GN30`, `meta-mild` | §3's loop, task per generation | — |
| `meta-free` | the identical task sequence, sequential multi-task training, no genome | "is it the bottleneck or just the data" |
| (§4's single-task sources) | already run | "is it the distribution or just the loop" |

Every meta arm at a seed sees the same task sequence in the same order (CRN over tasks).

### 10.3 Readings, pre-registered

On the **held-out** distribution, §9's protocol: `decode(meta-GN)` against `random`, against
`copy(meta-free)` (the multi-task initialisation from the same data), and against
`decode(single-task GN)`. **Learns how to learn** = `decode(meta-GN) − random < −3σ_e` at 25 % or
50 % of the held-out budget **and** `decode(meta-GN) − copy(meta-free) ≤ 0`: faster than a
fresh start, and no worse than the best initialisation plain training produces from the same
tasks. If `decode(meta-GN)` beats `copy(meta-free)` by more than `3σ_e`, the bottleneck itself
is doing meta-learning work the data alone does not — the paper's regulariser claim, on
adaptation. If `copy(meta-free)` wins, the data did it and the genome is a lossy copy of what
sequential training already found.

### 10.4 Acceptance tests

| # | property |
|---|---|
| T-I1 | task tokens are deterministic from the seed and lie in the reserved block |
| T-I2 | a saved genome reloads and decodes to exactly the run's final decode |
| T-I3 | `initialise` writes exactly the layer set's keys and nothing else |
| M-I1 | the task distribution is deterministic from its seed; held-out tasks are disjoint from every training draw |
| M-I2 | the meta pool switches task at `advance()` and its digest changes; a batch never mixes tasks |
| M-I3 | the meta hook advances the pool once per generation, at the boundary, before regeneration |
| M-I4 | the meta pool's Fisher probe is the current task's windows and changes with `advance()` (§11.1) |

## 11. The metric-corrected refit, and selection — variants, pre-registered 2026-09-07 19:40

**Why.** Three runs (11 §P-D.46–48) measured that Algorithm 1's proxy — mean-square error on
the weights — carries no information about the decode's *function* at 107–174× on this network:
across runs a better fit decoded worse (§P-D.47); within runs the correlation between the
residual and the at-birth CE over generations 100–200 has a random sign (`−0.40`, `+0.25`,
`+0.17`; §P-D.48); and once `ε ≈ 0` the loop is a walk in genome space driven by the burst
deltas with nothing selecting for function (§P-D.48's interpretation). The paper never met this
because on its networks the proxy and the function move together (§P-D.44). Two changes,
separable, each keeping the paper's structure — the genome is fitted to weights and is never
trained through the decode, the route GNOMON and ANLAGE found to collapse (§P-D.44):

### 11.1 Fisher-weighted regression (`--fisher`)

The quantity the paper's objective `E₀(G)` (its Eq. 1) wants small is `L(Ŵ) − L(W)`. To second
order at the trained `W`,

```
    L(Ŵ) − L(W)  ≈  ∇L(W)·(Ŵ − W)  +  ½ (Ŵ − W)ᵀ H (Ŵ − W)
```

the gradient term vanishes in expectation at a trained point, and `H` is approximated by the
diagonal empirical Fisher `F_ij = E_b[(∂ℓ_b/∂W_ij)²]` (Kirkpatrick et al. 2017, PNAS 114:3521,
Eq. 3; the Laplace approximation, MacKay 1992). The refit's objective per key becomes

```
    J_F(θ) = Σ_ij F̃_ij (Ŵ_ij − W_ij)² / Σ_ij F̃_ij,        F̃ = (1 − α) F / mean(F) + α,    α = 0.1        (11.1)
```

in place of the plain mean square. **Implementation.** Entries are drawn in proportion to `F̃`
(with replacement, the same count per step as §3) and the unweighted MSE on the sample is
minimised; its expectation is (11.1), and `α = 1` is the sampler of §3 exactly (I12). The floor
`α` keeps a tenth of the uniform mass on entries the probe's gradient never reaches — the
embedding row of a token absent from the probe — which `α = 0` would leave unfitted and hence
arbitrary at decode. The probe is a fixed slice of `n_fisher = 32` windows (16k tokens, one
training batch) of the **fitness** set — never the test set — in micro-batches of 4, eight
squared gradients averaged; the cross terms inside a micro-batch are EWC's standard
approximation and are recorded as such. `F` is taken at `W_n`, before the refit, once per
generation: eight forward-backward passes of `4 × 512` tokens, about `0.4` s against the hook's
`7` s. Reported beside the plain residual: `r_F = Σ F̃ (Ŵ − W)² / Σ F̃ W²` per key (I13).

| hyperparameter | default | range | why |
|---|---|---|---|
| `α` | 0.1 | `[0, 1]` | 0 unfits unseen entries; 1 is the paper; 0.1 keeps every entry fitted at a tenth of the uniform rate |
| `n_fisher` | 32 | `1 … 256` | one training batch of tokens; the fitness set has 256 windows |
| `micro` | 4 | `≥ 1` | memory of one backward pass at `T = 512` |

### 11.2 Selection on the fitness set (`--select`)

After (c) the decode's CE on the fitness set is compared with that of the *decode in force*; if
worse, the genome and its optimiser are restored to their state before the refit and the decode
in force is used in (d). This is the elitist `(1+1)` acceptance rule of evolution strategies
(Rechenberg 1973) with the regression as the proposal; it turns §P-D.48's walk into a hill-climb
on the decode's function. It starts at `select_from = 30` (`ε(30) = 0.22`, where the paper's
regime begins): before that the decode of a barely-fitted genome is meaningless as an elite and
every proposal is taken. The test set decides nothing — the at-birth CE stays an honest held-out
number — and the fitness set is the one ANLAGE 08 §8.2 reserves for such fits. Cost: one
fitness evaluation per generation (`~1.5` s) and a 1 MB snapshot. Reported: accepted or rejected
per generation, the acceptance rate, the generation of the decode in force. **Not applicable to
§10's routine**, where a decode's fitness is not comparable across tasks; §10 takes 11.1 only.

### 11.4 The vocabulary table left uncompressed (`--keep EMB`) — added 2026-09-07 19:35

The per-role ablation of the two best decodes (11 §P-D.49) measured the token embedding as the
binding constraint: its decode alone, with every other weight exact, costs `+4.05` nats (GN50)
and `+4.18` (GN30 v2) — the whole gap between trained and decoded — and leaving it exact inside
the full decode takes GN50's `7.29` to `5.53` and GN30 v2's `7.67` to `6.87`. Its relative
residual (`0.007`) had said nothing: `92 %` of the table's squared norm is its mean row, and the
row-specific part — the content — is what the g-network cannot express. The reason is
structural, and it is the paper's own premise failing at one place: the g-network's input is a
*coordinate* code, and the paper's coordinates (pixel positions, and hidden units indexed at
random) carry the network's structure because nearby pixels share filters. The token axis has
no such topology — a Gray code over frequency rank puts adjacent ranks one bit apart, and
adjacent ranks are unrelated words — so the table is a lookup of 8192 independent vectors that
no function of its coordinates can compress. An MNIST network has no such object; a language
model's first layer is exactly one.

`--keep EMB` copies the table through the decode as the paper copies scalars, **counts every
entry in the genome's size** (§4's ratio stays honest: `13.0M / (0.12M + 2.1M) ≈ 5.9×` at
`H = 50`), and leaves everything else as Algorithm 1. It is a *reallocation* of the genome, not
a relaxation: at the same ratio class as `mild` (5.0×, everything compressed at `H = 419`) it
puts the whole budget where the ablation says the function is.

### 11.5 The g-network's hidden activation (`--activation leaky`) — added 2026-09-07 20:45

`mild` (`H = 419`) measured **20 of its 88 matrix g-networks with no active second-layer unit
from generation 1** (11 §P-D.50): dead ReLU layers, which decode a constant `≈ 0` matrix, so a
quarter of the phenotype's matrices were regenerated as zeros in every generation, and the arm's
"5×" was nominal. No `H = 50` g-network died (GN30, GN50: 0 of 88). The paper does not state
its activation; §1 took ReLU. `--activation leaky` uses LeakyReLU with slope `0.01` (Maas et
al. 2013, the standard remedy) in both hidden layers, which keeps a gradient path through an
inactive unit so that it can recover, and changes nothing else. ReLU stays the default so the
`H = 50` arms remain comparable with GN50 under CRN; the activation is recorded in the genome's
summary and reloaded with it (I16).

Arm `mild-L` (seed 0, everything else as `mild`) is the honest `5×` control that R-E1 reads
against; `mild` is kept as the record of the defect. Readings: **R-L1** no matrix key of
`mild-L` has residual `> 0.98` after generation 20; **R-L2** no burst of `mild-L` diverges
(`before < 8.5` at every generation). `mild`'s burst at generation 121 went from the best
decode of any arm (`6.81`) to the uniform state within two hundred steps and the loop then
fitted the dead phenotype for nine generations (residual `0.25`); the H = 50 arms never
diverged in 400 bursts. If R-L2 fails with the dead networks gone, the divergence belongs to
the constant learning rate under regeneration, and is recorded as such. `mild` seed 1 is
dropped from the queue: a second seed of a defective configuration has no reading.

### 11.6 The vocabulary table as a rank-`r` factorisation (`--emb-rank r`) — added 2026-09-08 01:20

GN50-E reads *works* on two seeds (11 §P-D.52–53) with the table carried verbatim, which makes
the table `95 %` of the genome and sets the whole-network ratio at `5.9×` whatever the layers
do. The table needs a *content* code — a function of what the rows are, not of where they
sit — and the simplest one with an optimality guarantee is its rank-`r` truncated SVD
`U_r Σ_r V_rᵀ`, the closest rank-`r` matrix in Frobenius norm (Eckart–Young–Mirsky), computed
from the trained table at every generation in closed form: Algorithm 1's step (b) solved
exactly for that object under a rank constraint. `r (V + d)` parameters are counted
(`r = 16: 0.135M`, `32: 0.27M`, `64: 0.54M`). Regeneration then pulls the table onto the
rank-`r` set and the burst trains it freely, as with every other object of the loop. The
residual of the table is reported like a g-network's; the table has no g-network and is not
refit by Adam.

Arms at `H = 50`, seed 0, CRN with GN50: `GN50-R32` (whole-network ratio `≈ 33×`) and
`GN50-R16` (`≈ 51×`); `GN50-R64` (`≈ 20×`) if the card allows. Readings: **R-R1** at-birth
CE against §6's thresholds per rank, and the curve of best at-birth CE against the honest
whole-network ratio through `GN50-E (5.9×)`, `R64`, `R32`, `R16`, `GN50 (107×, table
compressed by coordinates: 7.29)`; **R-R2** the table's own residual against its centred
content (`0.09` was the g-network's centred residual at `H = 50`, §P-D.49). Test I17.

### 11.3 Arms and readings, pre-registered (11.4's arm added 19:35, 11.5's at 20:45, 11.6's at 01:20 the next day, before any of them ran)

Arms at `H = 50` (GN50's structure), seed 0, the same windows as GN50 (CRN): `GN50-E` (11.4,
ratio `≈ 5.9×`), `GN50-F` (11.1 alone, 106.6×) and `GN50-FS` (11.1 and 11.2, 106.6×); `GN50`
(§P-D.48) is the control, and `mild-L` (11.5, `5.0×`, `H = 419`, LeakyReLU) the honest control
at the `5×` class — `mild` itself (§P-D.50) ran with a quarter of its g-networks dead. Order of
running: E, F, FS, mild-L — cheapest and most strongly predicted first.

| reading | condition |
|---|---|
| R-E1 allocation | GN50-E's best at-birth CE below `mild-L`'s (and `mild`'s `6.81`) by more than the two-nat run-to-run spread, at the same ratio class: where the genome goes matters more than how big it is. The ablation's `5.53` is the point prediction for GN50-E at its best generation if the loop's dynamics are unchanged by the reallocation |
| R-L1, R-L2 | `mild-L`: no dead g-network after generation 20; no diverging burst (§11.5) |
| R-F1 alignment | over generations 100–200 of GN50-F, `corr(r_F, CE_birth) > +0.5`, where the plain residual gave `−0.40 / +0.25 / +0.17` (§P-D.48): the metric is aligned with the function |
| R-F2 at birth | §6's thresholds unchanged (works `< 6.0`, fails `> 8.5`); the arm's best and plateau against GN50's `7.29 / 7.76`, with the two-nat run-to-run spread (§P-D.47) as the bar for any claim of a difference |
| R-S1 no walk | in GN50-FS the decode in force never worsens on the fitness set (by construction) and the test at-birth CE of the decode in force has no excursion above `8.5` after generation 60: the walk is gone |
| R-S2 not frozen | acceptance rate over generations 30–200 above `10 %`; below it selection froze the loop, recorded as such |
| lifetime | as §6, against `free` |

**Conditional branch, fixed now:** if any variant arm reads *works* (`< 6.0`), §10's routine is
rerun with that arm's flags — the arm with the lowest best at-birth CE among those below `6.0`;
selection is never carried, 11.2 — and transferred to `heldout`; otherwise it is not. Tests
I11–I15, M-I4 (§8, §10.4).

**Combinations, added 2026-09-08 11:45** (11 §P-D.54: GN50-E works on two seeds, and GN50-F
read works at generation 175 with the table compressed): `GN50-FE` (11.1 + 11.4) and
`GN50-FR32` (11.1 + 11.6 at rank 32). Reading **R-C1**: the combination's best at-birth CE
below its single-variant parent's (GN50-E `2.84–2.86`; GN50-R32) by more than `0.1` — three
times the largest paired seed-to-seed difference measured for GN50-E (§P-D.53) — with the
lifetime and the shuffle read as in §6.
