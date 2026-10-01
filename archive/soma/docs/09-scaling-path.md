# SOMA — Specification 09: The Scaling Path

**Spec version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Non-normative for code; **normative for what may
be claimed**.

---

## 9.1 What this section does and does not establish

The local machine is a research and validation rig. A design that works only at `1e6`
parameters is a curiosity, not a paradigm.

**[NORMATIVE]** No component may be described as "scalable" on the strength of the argument
below. The argument establishes only that no component has a **known** barrier. Absence of a
known barrier is not evidence of scaling.

---

## 9.2 Component-by-component

| Component | Largest demonstrated scale (any author) | SOMA's additional requirement | Risk |
|---|---|---|---|
| Gated delta mixer, 3:1 hybrid | 1.3e9 params / 1e11 tokens | none — used as published | low |
| Signed `[-1,1]` transitions | 1.3e9 / 1e11 | must compose with the gated chunkwise kernel (§05.2.3) | **medium** — and see the limitations below |
| Muon | 1.6e10 (MoE) / 5.7e12 tokens | none | low |
| Product-key addressing | 1.28e11 addressed params / 1e12 tokens | used for *unit* addressing, not memory lookup | low |
| Masked function-preserving growth | ~1e8; an unverified claim exists at 1.01e11 | applied at unit granularity; the RMSNorm form is `[OURS]` (§03.3) | medium |
| Token-level excess-loss selection | 7e9 params | precomputed reference (§06.1.1) | low |
| Spatial expression map, `\|Gamma\|` independent of `\|U\|` | **~1e6 params** | **four orders of magnitude beyond any demonstration**; §08.6.1's sweep is the only direct measurement | **high** |
| ε-lexicase over units | **never done in an LM** | entirely new | **high** |
| Metabolic pricing of wiring | ~1e2-node evolved networks | entirely new in an LM | **high** |
| Cross-lifetime germline inheritance | **never done** | entirely new | **high** |

**[NORMATIVE — the signed-transition limitations, carried from §11 C-13]** Spec v1's table
named only kernel composition as the additional requirement. The source also reports that
(a) language metrics **degrade** for the diagonal model under the modification, (b) results
at 1.3e9 are **mixed**, and (c) medians are far below best-of-3 (`0.826 +/- 0.146` vs
`0.971`). §08.2's pass bar is therefore set against the **median**.

**[NORMATIVE]** The four `high`-risk rows are the research contribution **and** the rows with
no scaling evidence. Both facts appear together in any description of this work.

---

## 9.3 Why the germline does not grow with the phenotype

This is the load-bearing scaling property and the one arguable from first principles.

```
  |Gamma| = O( N_G * (m_dim + 1) )        # gene fields
          + O( J_max * k_g )              # Inventory archetypes
          + O(1)                          # policy constants
                                                                                 (9.1)
```

**[NORMATIVE — v1 correction]** Spec v1's (9.1) contained a term
`O(sqrt(N_G) * d_key)` for gene-side product keys, which were **deleted** in audit repair
R9/B36 (they addressed nothing), and **omitted** the Inventory, which is germline and scales
as `J_max * k_g`. Both are corrected. Crucially, the Inventory term is still independent of
`|U|`, `d_model` and `L`.

**[DERIVED]** `d|Gamma| / d|U| = 0`, `d|Gamma| / d(d_model) = 0`, `d|Gamma| / dL = 0`.
Widening the model or adding units changes the soma budget and leaves the germline untouched.

At **both** configurations, since §01.2 fixes `N_G = 2048`, `m_dim = 8`, `k_g = 8`,
`J_max = 256`:

```
  gene fields   2048 * (8 + 1)   = 18,432 reals
  Inventory      256 * 8         =  2,048 discrete indices  (+ 3 stats per archetype)
  policy                          ~    20 unconstrained reals
                                  --------
                                   20,500
```

comfortably inside the `1e4`–`1e5` target of §02.1, and **identical at M-nano and M1** —
which is the property (9.1) claims and the reason §2.6.2's M-nano → M1 inheritance is
shape-possible at all. v2 wrote `2048 * 3 + 256 * 8 + ~20 = 8.2e3`, using `m_dim + 1 = 3`
against its own `m_dim = 8`, and quoted the figure for M1 alone while M-nano then had
`N_G = 512`.

**[OPEN]** The `1.4e6`-parameter hypernetwork ceiling is `[EXTRACT]`, single-pass (§11 C-8),
and is the sole cited support for the contrast drawn below. Deferral criterion: §08.6.1's
`N_G` sweep measures the property directly. Note the design does not *depend* on the cited
figure — (9.1) is derived, not borrowed — so the marker qualifies the rhetoric, not the
architecture.

**[NORMATIVE — the falsifier]** This is why the `1.4e6`-parameter ceiling binding
full-weight-generating decoders does not bind here: **we never generate a weight from a
code**; we select which shared, gradient-trained parts a unit expresses. If at any scale
`N_G` must grow proportionally to `d_model` or `|U|` to maintain quality, (9.1) fails, the
bottleneck is not a bottleneck, and the architecture has degenerated into a hypernetwork.
§08.6.1 is the experiment; K9 is the criterion.

**[NORMATIVE]** `K_unit1, K_unit2` are **soma** (§02.1.1). Had they been germline, `|Gamma|`
would scale with `U_max` and (9.1) would be false by construction — K9 would fire
definitionally, without any experiment.

---

## 9.4 Why the developmental phase should shrink as a fraction

At generation `G_gen`, the germline already encodes a body plan that worked. Development
exists to *discover* that plan; as `G_gen` grows, less discovery is needed, and `rho_0` and
`n_c` — both germline constants under ES — should evolve downward.

**[DERIVED]** This predicts a measurable signature: `n_c` selected at generation `g` should
decrease in `g`.

**[NORMATIVE — v1 correction]** Spec v1 offered this as a cheap "early warning". It is not:
`n_c` is a single evolving scalar expressed as a fraction of `N_step`, and detecting a trend
in it requires roughly the same `G_gen` and seed count as the primary test of §07.5. It is
therefore reported as a **secondary confirmation**, not an early warning, and the required
`G_gen` is the same as for K1.

This is also the answer to the objection that the developmental phase is wasted compute.
Within one lifetime it is a cost; across lifetimes it is capital formation, and the amortised
cost per lifetime should fall with `g`. If it does not, the thesis is wrong and §07.5 says so.

---

## 9.5 Throughput at scale

The expression operator costs `O(N_G d_model + k_route k_g d_model)` per token (§01.4), never
materialises a per-unit weight matrix, and reduces to one dense matmul plus a sparse gather.
It is therefore a **dense** primitive on the accelerator and does not inherit the documented
gap between theoretical and realised speedup that afflicts unstructured sparsity.

The mixer is linear in sequence length; the attention layers are one in four. Full attention
is used at M-nano and M1 because `T_ctx` is short (§05.3); windowing is `[OPEN]` for
`T_ctx > 2048`.

**[NORMATIVE]** These are architectural properties, not measurements. Each is verified by
profiling at M2 and reported as measured tokens/s and peak memory (§07.8).

---

## 9.6 What production would additionally require

Stated so the gap is explicit.

1. **Multi-device sharding of the unit population.** Units are independent given the shared
   parts, so the natural partition is by unit with an all-gather on `A x`. Designed, not
   implemented; communication cost unmeasured.
2. **Selection that does not synchronise all devices.** ε-lexicase as specified is a global
   operation over a Site's population. An island model with periodic migration is the standard
   answer but changes the selection semantics and needs its own ablation.
3. **Longer context.** M-nano and M1 use short contexts. The mixer is linear in length, but
   the attention layers, the sink, and the routing statistics are unvalidated beyond a few
   thousand tokens.
4. **Instruction and preference post-training.** Unaddressed, and carrying an identified risk:
   an architecture with dynamic per-token computation was found **incompatible with standard
   RL post-training** because variable-depth execution breaks fixed rollout paths in inference
   engines. SOMA's unit routing is per-token and may inherit this. **Unmitigated** (R9).

**[NORMATIVE]** Until (1)–(4) are addressed, SOMA is a research architecture with a scaling
*argument*, not a production architecture. Any claim beyond that is unsupported.
