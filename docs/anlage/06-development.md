# 06 — Development: growth sampled from the genome

*(Constitutional requirement C1; axiom A5.)* A phenotype is born small and grows in stages
during its lifetime. Every unit it gains is a draw from the genome's generative model of
units; every growth event changes the function — continuously, under a ramp — and is measured
against the two conventional ways of growing a network and against not growing at all.

## 6.1 Why growth, and why it is linked to the genome

**[RATIONALE]** Staged growth is the one developmental mechanism with large, replicated
compute savings in language-model pretraining (§10 E-4, E-5: 54.6% at 7B/300B tokens; 2.2–4×
for width cloning; up to ~5× for layer expansion), and the savings are largest at the low
tokens-per-parameter regime this hardware lives in. The literature grows by *copying*
(HyperCloning, G_stack): new units are perturbed duplicates of old ones, and symmetry must be
broken by noise. ANLAGE grows by *sampling*: a new unit's weights are the genome's decode at a
fresh identity with a latent drawn from the fitted prior (§02.6), so new units are distinct,
structured, and "typical" of the lineage by construction. Proposition 2 (§05.8) is what makes
the old units' weights and the new units' decodes live in one coordinate system.

## 6.2 Definition 2 — a growth event

**[NORMATIVE]** At the transition from stage `s` to `s+1` (`omega` of §6.5), with new units
`A_new = A_{s+1} \ A_s` per axis:

1. **Identities.** New units on channel axes take the next codes of the channel table
   (nested, §02.2); new tokens do not occur (`V` is fixed per lineage). Each new unit's latent
   is sampled per §02.6: `z ~ N(mu_X, diag(s_X^2))` from the fitted moments of its axis
   (ablation: `z = mu_X`; control: `z = 0`).
2. **Weights.** Every role's tensor is reshaped to stage `s+1`. Entries whose indices all lie
   in `A_s` **keep their trained values**; entries with any index in `A_new` are **decoded**
   from `(Gamma, Z_{s+1})` — the decode of the full stage-`s+1` matrix, restricted to the new
   entries. The RMS normaliser of (3.4) is the stage-`s+1` matrix's, so a new row is at the
   same scale as the decoded old rows would be.
3. **Layers.** Depth growth inserts `L_{s+1} - L_s` new layers. Positions: **all** existing
   layers are spread uniformly over the target depth **in their original order** — existing
   layer `i` of `L_s` goes to slot `round(i (L_{s+1} - 1)/(L_s - 1))` — and keep their
   operators. The free slots become attention layers only as far as needed to reach the
   canonical target's attention count (`l mod 4 == 3` gives `floor(L_{s+1}/4)`), each chosen
   nearest to a canonical attention position not already claimed by an existing attention
   layer; the remaining free slots are delta layers. The grown stage therefore carries an
   explicit operator layout (`StageConfig.ops_override`) with the canonical attention count,
   and its attention positions may differ from the canonical stage's by the spreading. A new
   layer's weights are decoded in full from `(Gamma, Z_{s+1})` with its relative-depth code
   and a latent `y` sampled from its two nearest existing layers' moments. Existing layers
   keep their trained weights (their codes change, §02.4; their latents travel with them).

   *(An earlier draft spread each operator's layers independently over that operator's
   slots. That rule can move an existing attention layer ahead of an existing delta layer,
   which changes the function even at `rho = 0`; test G1 found it, and Theorem G1's proof
   requires the order-preserving rule above.)*
4. **The ramp.** A gate `rho(n) = min(1, rho_0 + (1 - rho_0) n / n_ramp)`, `rho_0 = 0.1`,
   `n_ramp = 0.02 N_{s+1}`, multiplies every *new* unit's write into the residual stream
   (§6.3), so the function moves continuously from the stage-`s` function (at `rho_0`, up to
   the `rho_0`-scaled contribution) to the full grown function. The loss discontinuity at the
   event, `CE(step n_event + 1) - CE(step n_event)`, is a **measured** quantity and must be
   within `2 sigma_e` after `n_ramp` steps (G4).
5. **Optimiser state.** The optimiser is rebuilt at the boundary: every parameter, old and
   new, starts from zero momentum (Muon) and zero moments (Adam). Stage `s` ends with its
   learning rate decayed to zero (§04.6), so the state it leaves is that of a stopped
   optimiser and carrying it over would preserve nothing. The schedule of stage `s+1`
   restarts its warm-up and batch warm-up over `n_ramp` (SPARKLING's asymmetric reset,
   §10 E-5). `optimizer_groups` of the grown phenotype must cover every parameter exactly
   once, matrices under Muon (G7).

## 6.3 The gates, per axis — and Theorem G1

**[NORMATIVE]** The ramp is applied through one gate vector per grown axis, and the block of
§04.3 reads them as follows:

| axis grown | gate | where it acts |
|---|---|---|
| `RES` (`d_s -> d_{s+1}`) | `m_res in [0,1]^{d_{s+1}}`, `1` on old channels, `rho(n)` on new | every gain-free norm of the block becomes `masked_rms_norm(x, m_res)` (`gnomon/pheno/rmsnorm.py` (3.4)), whose output is multiplied by `m_res`; so every reader of the residual sees the new channels at gain `rho` and the RMS statistic excludes them at `rho = 0` |
| `INNER(l)` (`F_s -> F_{s+1}`) | `m_inner^l` | the FFN hidden activation is multiplied by `m_inner^l` before `W_DOWN` |
| `HEAD(l)` (`h_s -> h_{s+1}`) | `m_head^l` over head channels | the mixer's per-head output is multiplied by `m_head^l` before `W_O` |
| a new layer | scalar `rho(n)` | both of its residual writes are multiplied by `rho(n)` |

When `rho(n)` reaches `1` the gates are removed and the block is the plain block of §04.3;
no discontinuity occurs because every gate is then identically `1`.

**Theorem G1 (continuity at the event).** Let `W^{(s)}` be the stage-`s` phenotype and
`W^{(s+1)}` its growth by Definition 2 with all gates at `rho = 0`. Then for every input the
logits of the two are equal up to floating-point reduction order.

*Proof.* Induct over the block. With `m_res = 0` on new channels, `masked_rms_norm` (3.4)
computes the second moment over old channels only and zeroes the new ones on output, so every
sublayer receives exactly the stage-`s` normalised input. Inside a sublayer, new head channels
and new FFN units are multiplied by `0` before `W_O` / `W_DOWN`, so their contribution is
zero and old units' outputs are unchanged (their weights are unchanged by step 2). New layers
write `0`. Old channels of the residual therefore evolve exactly as at stage `s`; new channels
receive writes but are never read at gain above `0`, including by the final norm and the tied
head, whose `EMB` columns for new channels are read through `m_res`. □

**[NORMATIVE — what the theorem does and does not license]** It licenses starting a growth
event at `rho_0 = 0.1` with a bounded, measured discontinuity rather than at `rho = 0`, where
new units would receive no gradient through their writes. It is not a function-preservation
guarantee at `rho_0`; the discontinuity is reported, and G4 bounds it after the ramp.

## 6.4 The event's relation to the genome: regression before growth

**[NORMATIVE]** A growth event that decodes new units from `(Gamma, Z)` presupposes a genome
fit to the *current* phenotype. Therefore `REGRESS` (§05.2) runs immediately before every
growth event on the stage-`s` phenotype (this is the in-lifetime consolidation, labelled as
such in its report); the new units are then decoded from the freshly fit genome. The cost is
`REGRESS`'s (seconds to minutes) once per stage. Optional additional consolidations every
`N_sleep` steps (`N_sleep = 0` by default) are a registered ablation, as is the anchor of
§05.7.

## 6.5 The schedule `omega`

**[NORMATIVE]** `omega = ((d_s, L_s, F_s, h_s, C_s))_{s=0..S}` with `C_s` the **FLOP** budget
of stage `s` (`6 N_s T_s` with `N_s` the stage's non-embedding parameter count and `T_s` its
tokens). Budgets are in FLOPs, not tokens, because the point of growth is to spend fewer FLOPs
per token early; the tokens each stage sees are derived and reported.

Defaults (fractions of the lineage's `C_total`):

| stage | fraction of `C_total` | rationale |
|---|---|---|
| 0 (primordium, siblings + merged elite) | 0.15 | the population lives here (§07); `C_pop` is part of this fraction |
| 1 | 0.25 | |
| 2 (adult) | 0.60 | most compute at full size, where the from-scratch loss is set |

The G_stack timing law `log10 d = 0.88 log10 N + 163.27/log10 C - 5.74` is the prior from
which these fractions were chosen and is recomputed and reported for each lineage (§10 E-4
records its limits). `omega` is serialised with the genome; **it is mutated among siblings
only after K-GROW has passed with this fixed schedule** (SCREEN's rule, §00.3).

## 6.6 Controls and the registered hypothesis

**[NORMATIVE]** Matched on `C_total` and on the schedule, seed-paired by CRN, `≥ 4` seeds:

| arm | growth initialisation of new units |
|---|---|
| **ANLAGE-GROW** | Definition 2: decoded from the genome with sampled latents |
| **B-COPY** | width: HyperCloning `W' = [[W/2 + n_1, W/2 - n_1], [W/2 + n_2, W/2 - n_2]]`-style duplication with symmetry-breaking noise at 1% of the RMS; depth: copy of the nearest existing layer of the same operator with the same noise; heads and FFN units: copies of existing ones with noise |
| **B-RAND** | new units from §04.7's initialisation |
| **B-FLAT** | no growth; the adult trained from scratch for `C_total` |
| **MEAN** (ablation) | Definition 2 with `z = mu_X` (no sampling) |
| **ZERO** (ablation) | Definition 2 with `z = 0` (innate identity only) |

**Registered hypothesis H-GROW.** `CE(ANLAGE-GROW) ≤ CE(B-COPY) - 3 sigma_e` and
`CE(B-COPY) ≤ CE(B-FLAT)` at matched `C_total`. The second inequality is the literature's
claim, re-measured here at this scale because §10 E-5 reports the gains shrinking above
~20 tokens per parameter; if it fails, growth itself is not paying at this budget, and that is
reported before any claim about the genome's part in it.

## 6.7 Acceptance tests

**[NORMATIVE]** `anlage/grow.py` is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| G1 | **Theorem G1** | at `rho = 0` the grown model's logits equal the ungrown model's to `1e-5` in float32 on random input, for width, FFN, head and depth growth separately and combined |
| G2 | old weights untouched | after growth, every entry with all indices in `A_s` is bit-identical to before |
| G3 | new units decoded | every entry with an index in `A_new` equals the stage-`s+1` decode restricted to those entries, to `1e-6` |
| G4 | ramp bound | on a trained stage-0 phenotype, the held-out CE `n_ramp` steps after a growth event is within `2 sigma_e` of the CE immediately before it; the discontinuity at the event is recorded |
| G5 | layer placement | for `(8 -> 12)` and `(12 -> 16)` the positions of §6.2 step 3 are the ones listed in the test, every existing layer keeps its operator and the existing layers' relative order, and the attention count equals the canonical target's |
| G6 | KV grouping survives | an existing attention head reads the same KV head before and after head growth (P5's `g mod n_kv`) |
| G7 | optimiser state | after a growth event the rebuilt optimisers cover every parameter of the grown phenotype exactly once (matrices under Muon, the rest under Adam), all with zero state |
| G8 | sampled latents | new units' latents have the moments of §02.6 within `4/sqrt(n)`; a `MEAN` arm's are exactly `mu_X` |
| G9 | budgets | the tokens per stage derived from `omega` reproduce `C_s` to within one batch |
| G10 | new units are distinct | pairwise cosine similarity between new units' decoded rows has mean below `0.5`; B-COPY's copies are above `0.9` before noise (positive control) |
| G11 | gates removed | after `n_ramp` the forward pass uses the plain block and no gate tensor remains |
| G12 | resume | a lifetime checkpointed mid-ramp resumes bit-exactly, including the gate values |
