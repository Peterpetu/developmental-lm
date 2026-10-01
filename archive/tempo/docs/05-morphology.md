# 05 — Latent morphology: the shape as a slow state

**Status: normative.** The switches and their groups (§5.1), the branch multiply and compute
skipping (§5.2), births by copy (§5.3), Theorems G1 and G2 (§5.4), the primordium (§5.5), FLOP
accounting (§5.6), the controls of K-MORPH (§5.7), and the acceptance tests (§5.8). Symbols:
03; the morphology roles of 04 §4.2.

## 5.1 Switches and groups

**[NORMATIVE]** A *block* is one of: a delta or attention head `(l, j)`, an FFN slice `(l, s)`
of 256 inner units, or a layer `l`. Blocks are switched in *groups*, one pre-activation per
group (the large-move rule, 11 §P-B.12: one-shot doubling beats gradual stacking):

| group | blocks | pre-activation | at birth |
|---|---|---|---|
| quarter-stack `Q_1 = {4, 5, 6, 7}` | layers 4–7 (`d d d a`) | `a_{Q_1}` | off |
| quarter-stack `Q_3 = {12, 13, 14, 15}` | layers 12–15 | `a_{Q_3}` | off |
| head group `H_l = {(l, 4), …, (l, 7)}` for every layer `l` | heads 4–7 of layer `l` | `a_{H_l}` | off |
| slice group `F_l = {(l, 4), …, (l, 7)}` for every layer `l` | slices 4–7 of layer `l` | `a_{F_l}` | off |
| the primordium (§5.5) | layers `{0..3, 8..11}`, heads `{0..3}`, slices `{0..3}` of every layer | none: permanently on | on |

`a_G = −4 + γ_G` with `γ_G ∈ (−8, 8)` the group's genome gain (04 §4.3, kind morphology) and
`−4` the baseline, so that at birth (`γ = 0`) every switchable group is skipped
(`σ(−4) = 0.018 < ε_skip`) and the range reaches `σ(4) = 0.982` (11 §P-C.17); `m_G = σ(a_G)`.
Every block `b ∈ G` has `m_b = m_G`; a block of the primordium has `m_b = 1`. A block is
**active** when `m_b ≥ ε_skip = 0.02` (i.e. `a_G ≥ σ^{-1}(0.02) = −3.892`, `γ_G ≥ 0.108`),
otherwise **skipped**. A head of an inactive layer is inactive whatever its group's value. Pile-adult
groups follow the same rule with its `L = 24` (quarter-stacks of six layers; `Q_1` and `Q_3`
switchable) and `h = 16` (head groups of eight).

## 5.2 The branch multiply and compute skipping

**[NORMATIVE]** For a layer, the residual update 03 (3.1)–(3.2) is multiplied by `m_l`. For
an active head `(l, j)` its contribution to `o` in 03 (3.8)/(3.9) is multiplied by `m_{(l,j)}`
before `W_o`; for an active slice its inner activations `SiLU(W_gate ĥ) ⊙ W_up ĥ` on its 256
units are multiplied by `m_{(l,s)}` before `W_down`. Skipped blocks are not computed: the
active views of 03 §3.8 are rebuilt at the start of every episode (and only then) from the
switch pattern, and a skipped layer is not executed. The switch pattern is a τ2 state written
into the checkpoint (08 §8.4).

## 5.3 Births by copy

**[NORMATIVE]** When a consolidation raises a group's `a_G` across `σ^{-1}(ε_skip)` from
below, every block of the group is *born*:

```
   head (l, j), j ∈ {4..7}:       rows of W_q, W_k, W_v, W_og for head j  ← rows of head j − 4  + η
                                  columns of W_o for head j               ← columns of head j − 4 + η
                                  taps, a_{0,j}, g_{0,j}                  ← those of head j − 4
                                  S_{(l,j)} ← 0
   slice (l, s), s ∈ {4..7}:      rows of W_gate, W_up for slice s         ← rows of slice s − 4 + η
                                  columns of W_down for slice s            ← columns of slice s − 4 + η
   layer l ∈ Q_1 (resp. Q_3):     every parameter of layer l               ← that of layer l − L/4, + η
                                  (the layer at the same relative depth of the previous
                                  quarter-stack; it has the same operator because a
                                  quarter-stack is a whole number of operator periods, which
                                  the configuration checks, and it must be active in the
                                  pre-move pattern or the birth raises)
   η ~ N(0, (0.01 · RMS(source))²) elementwise;  optimiser state of the born tensors ← 0                       (5.1)
```

The born block then contributes `m_b × (its output)` with `m_b = σ(a_G) ≥ ε_skip`; the
consolidator raises or lowers `a_G` in later episodes (06), so the ramp is the consolidator's
decision, not a schedule. **Deviation from the plan, recorded:** the plan's HyperCloning
split `[W/2 + η, W/2 − η]` on the shared-input side is *not* used for births, because it halves
the source block's contribution and so changes the function at `m_b < 1`, which breaks
Theorem G1 during the ramp; the split is used only by the one-shot doubling control of §5.7,
where `m_b = 1` at once. A born block's gain roles (`a_in, a_out` of its rows and columns) are
already in the registry (the registry is built at the provisioned shape) and take their
current `u` values.

A group whose `a_G` falls back below `σ^{-1}(ε_skip)` is skipped again; its parameters and
optimiser state are kept (a later rise re-uses them without a copy: births happen only from
the never-active state, tracked by a per-group flag in the checkpoint).

Two rules the implementation must obey and the acceptance tests assert (11 §P-C.25): a head or
slice group of a layer that is **not active** in the new pattern is not born (its parameters
would be copied from a source that is itself dormant, and the copy would be overwritten when
the layer is later born from its own source); and the symmetry-breaking noise `η` is drawn on
the CPU generator and moved to the parameter's device, so that a life on CUDA and the same
life on CPU make the same copy. The taps and the head offsets are copied **exactly** (no
noise): they are the block's identity in time, and perturbing them makes the born head read a
different window than the head it was copied from.

## 5.4 Theorems G1 and G2

**Theorem G1 (function preservation at the switch).** For any block `b` with `m_b = 0`, or
skipped, the backbone's output equals the output of the backbone without that block, for
every input and state.

*Proof.* A layer's update is `h ← h + m_l (n ⊙ Op(ĥ))`, `h ← h + m_l (n ⊙ FFN(ĥ))`; at
`m_l = 0` both are `h ← h`; a skipped layer is not executed, which is the same map. A head's
contribution enters `o` as `m_{(l,j)} o_j` before the linear `W_o`; at `0` the term vanishes and
`W_o` acts on the remaining heads, which is what the gathered view computes; the same for a
slice under `W_down`. The delta state of a skipped head is not read. ∎ (ANLAGE's G1 test,
`tests_anlage/unit/test_grow.py`, is carried over and re-run against `tempo/backbone.py`.)

**Theorem G2 (the born block learns at rate `m_b`).** Let `z_b` be a block's output and
`θ_b` its parameters. Then `∂L/∂θ_b = m_b · (∂L/∂z_b)ᵀ ∂z_b/∂θ_b`, where `∂L/∂z_b` is the
loss's sensitivity to the block's output at the current network state, because `m_b`
multiplies `z_b` linearly and nothing else of the block depends on `m_b`. The statement is
local: the downstream state itself depends on `m_b`, so the gradient at `m_b` is not `m_b`
times the gradient at `m_b = 1`, but doubling `m_b` doubles the block's gradient up to
`O(m_b)` (M2 tests the ratio at `ε_skip` and `2 ε_skip`). *Consequence.* At
`m_b = ε_skip = 0.02` a born block receives two per cent of the gradient it would receive at
full multiplier and the same sensitivity; the consolidator's raising of `a_G` is therefore
what makes a birth effective, and K-MORPH's comparison against the one-shot control (`m = 1`
at once) is the test of whether that gradual entry is worth anything. ∎

## 5.5 The primordium

**[NORMATIVE]** At birth the active shape is layers `{0, 1, 2, 3, 8, 9, 10, 11}` (two
quarter-stacks, each `d d d a`: attention at 3 and 11, so the 3:1 ratio is kept), heads
`{0..3}` of every active layer, slices `{0..3}` of every active layer. Every layer keeps its
provisioned index, so when `Q_1` switches on its layers sit between 3 and 8 as their indices
say: relative depth is preserved (the order-preserving placement ANLAGE's G1 required, 00
§0.5). Active parameters at birth `≈ 17.9M` (03 §3.9, formula value; asserted).

## 5.6 FLOP accounting

**[NORMATIVE]** `tempo/morphology.py::flops_per_span(pattern, position)` returns, for a
switch pattern and a span position `t` in its window, the training FLOPs of one span:

```
   F_span = μ · Σ_{active matrices} n_out^{act} n_in^{act}          the matrix multiply-adds
          + μ · Σ_{active attention heads} 2 d_head (t + 1)          the SDPA over the cached spans
          + μ · Σ_{active delta heads} 2 d_head²                     the state update and read
          + μ · Σ_{active layers} 4 d + μ · 2 d_x d                  norms, W_in, W_out
          with μ = 6 for training (forward and backward) and μ = 2 for inference             (5.2)
   F_iface  = μ · [ Σ_{input spans} enc_{m(t)}  +  Σ_{scored target spans} head_{m(t)} ]      (5.2b)
   F_episode = Σ_{steps} [ B · Σ_{t=0}^{T_s−1} F_span(pattern, t)  +  F_iface  +  the rollouts' forwards ]  (5.3)
   F_ref     = F_episode at the primordium pattern on text alone, with no rollouts               (5.4)
```

**(5.2) counts the backbone; (5.2b) counts the interface, and it is not small.** `enc_m` and
`head_m` are the forward FLOPs of a modality's encoder and of its likelihood head, **measured
once per life on the real modules** with `torch.utils.flop_counter` — the same instrument
`scripts/tempo_cont.py` uses for the token baseline, so that the two sides of K-CONT are
counted by one method rather than by two hand-derived formulas that have to be trusted to
agree. At the TS shape the span decoder alone is `44 %` of the span agent's forward cost.

Leaving it out was not a harmless rescaling (11 §P-C.34). The protocol had **two definitions
of training FLOPs differing by a factor of 1.84**: K-CONT matched the token baseline on the
measured cost including the decoder, while `F_episode`, `F_cum` and the reward's denominator
used (5.2) alone. And because the interface's cost per token is the same for every morphology
while the backbone's is not, omitting it changes the *ratio* the equal-compute comparison
reads: adult over primordium is `3.93` on the backbone alone and `2.61` with the interface
charged to both, so K-MORPH would have given the flat-adult arm about half again as many
episodes as it is owed.

The attention term carries the same factor `μ` as every other term: the scores and the value
mixing are differentiated like any other product, and scoring them at `μ = 1` (the first form)
under-charged the adult attention layers by a factor of six and so *flattered* every arm that
switched attention heads on. The per-span attention term is scored at the span's own position
`t`, so the sum over a window is exact rather than an average.

**The effective charge.** A block entering at `m_b = ε_skip` contributes two per cent of its
output and is charged two per cent of its FLOPs by the `effective` form of (5.2), in which each
switchable group's blocks are weighted by their multiplier. The *executed* cost is the full
one (a block at `m = 0.02` is still computed), so the two forms differ, and the record carries
both: the executed cost is what the wall-clock reflects and what the equal-compute comparisons
of 10 §10.1 use; the effective cost is reported beside it as the size the reward would charge
if the ramp were free. No verdict is read off the effective form.

The reward of 06 §6.3 uses the executed `F_episode` including the probes; K-MORPH (10) runs
every arm to the same cumulative `Σ_e F_episode`, not to the same number of episodes and not
to the same number of tokens.

## 5.7 The controls of K-MORPH

**[NORMATIVE]** R6 (10) runs from the same birth and windows, to the same cumulative FLOP
budget `F_budget`:

| arm | growth | switches |
|---|---|---|
| (a) fixed schedule | ANLAGE's `(0.15, 0.25, 0.60)`: at those fractions of the **spent budget** the head groups, then the slice groups, then the quarter-stacks switch on by copy (5.1) with `a_G` set to `σ^{-1}(0.5)` and ramped linearly to `σ^{-1}(0.98)` over the following 5 % of the budget | scheduled |
| (b) one-shot doubling | at `0.5` of the spent budget every group switches on at `m = 1` with the HyperCloning split: the input side is copied **exactly**, and the *output-side* weights of both halves are halved, `W_o[:, src] ← W_o[:, src]/2`, `W_o[:, new] ← W_o[:, src]/2 + η` (heads), `W_down` likewise (slices), layers by plain copy | one move |
| (c) flat adult | every group on from birth (`m = 1`) | none |
| **TEMPO** | the consolidator's `a_G` (06) with births by §5.3 | learned |

The schedule variable is the **spent fraction of the compute budget** (`F_cum / F_budget`),
not the episode index: the arms have different per-episode costs, so an episode-indexed
schedule would place (a)'s moves at different compute points than (b)'s and the comparison
would not be at equal compute. Every arm runs until `F_cum` reaches `F_budget` and stops, so
the arms differ in how many episodes they get, which is the point.

In the split, the noise is added on the **output** side only, after the halving. Adding it on
the input side as well would make the two halves compute different functions of the input
whose *sum* is no longer the source's output, so the move would not preserve the function even
at `m = 1`; M10 asserts the preservation to `1e-6`.

K-MORPH (10 §10.2) fires unless TEMPO beats (a), (b) and (c) at the budget by each control's
own bar (10 §10.1). The record's prior is that growth is compute-neutral at this card's
budgets (ANLAGE R5, copy − flat `+0.003 [−0.007, +0.014]`).

## 5.8 Acceptance tests

**[NORMATIVE]** `tempo/morphology.py` is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| M1 | G1 | for every group, switching it off (`a_G = −8`, skipped) gives outputs bit-identical to a backbone built without those blocks, on random inputs and a nonzero fast state |
| M2 | G2 | the gradient norm of a born block's parameters at `m_b = 0.02` equals `0.02 ×` its norm at `m_b = 1` to `1e-5` when the two are evaluated at the *same* network state (the statement of §5.4 is local: the downstream state itself depends on `m_b`, so the test freezes it) |
| M3 | births copy | after a birth, the born rows/columns equal the source's plus noise of RMS `0.01 × RMS(source)` (measured to 10 %); the taps and head offsets are copied **exactly**; the source is unchanged; the born optimiser state is zero; the born delta state is zero; a group of an inactive layer is not born; the source is `l − L/4` and an inactive source raises |
| M4 | order preserved | when `Q_1` switches on, the executed layer order is `0..15` restricted to the active set (asserted by a forward hook trace) |
| M5 | skipped costs nothing | a skipped layer's forward hook is never called; the measured step time at the primordium is `< 60 %` of the adult's at the same batch (CUDA events, medians; recorded) |
| M6 | pattern in the checkpoint | the switch pattern, the born flags and `a_G` round-trip through the checkpoint; a resumed life has the same active views |
| M7 | FLOPs | `flops_per_span` at the primordium and the adult equals a hand computation from §5.6 and 03 §3.9, with the attention term carrying the training factor; the adult's value is within 5 % of the measured training FLOPs from `torch.utils.flop_counter` on one step; the `effective` form at `m = ε_skip` charges a switchable group `ε_skip` of its executed cost and equals the executed form at `m = 1` |
| M8 | re-activation | a group switched off and on again is not re-born (its parameters persist; the flag prevents the copy) |
| M9 | primordium | at birth the active set is exactly §5.5 and the active parameter count equals the formula value |
| M10 | controls | arms (a), (b), (c) of §5.7 produce the documented switch trajectories against the *spent budget* (fractions asserted), and (b)'s split preserves the function at the moment of the move to `1e-6` at zero noise (the sum of the halves equals the source's contribution) |
