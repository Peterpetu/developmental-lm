# 07 — Criticality: the edge of chaos, measured and maintained

**Status: normative.** The propagated tangent and the depth-wise finite-time Lyapunov
exponents (§7.1), the stack's exponent and the relative form the rule uses (§7.2), the
homeostatic rule on `ζ` (§7.3), the probe's batch and cost (§7.4), the references and the
standing of the claim (§7.5), the conditional test K-CRIT (§7.6), and the acceptance tests
(§7.7). Symbols: 03. Implementation: `tempo/criticality.py`.

## 7.1 The propagated tangent and the per-sub-block exponents

**[NORMATIVE]** Order the active stack's sub-blocks `k = 1..K_a`,
`K_a = 2 · |{active layers}|`, in execution order (`mix` then `ffn` of each active layer, by
increasing layer index), and write `f_k` for the residual map of sub-block `k` on a whole
window `H ∈ R^{T_s × d}` (all spans at once, so that the sequence mixing of the delta rule and
of attention is inside the Jacobian) with the fast state entering the window held fixed:

```
   f_k(H) = H + m_l ( n_l^{mix} ⊙ Op_l( ρ_l^{mix} ⊙ H / RMS(H) ) )        (k a mixing sub-block of layer l)
   f_k(H) = H + m_l ( n_l^{ffn} ⊙ FFN_l( ρ_l^{ffn} ⊙ H / RMS(H) ) )       (k an FFN sub-block of layer l)   (7.1)
```

The probe takes `h_0 = W_in x + b_in` (the residual stream entering layer 0 of a probe
window, 03 (3.1)), draws `δ_0 ~ N(0, I_{T_s × d})`, and **propagates one tangent through the
whole active stack** in a single forward-mode pass that returns every sub-block's output and
its tangent:

```
   h_k = f_k(h_{k−1}),        δ_k = J_k(h_{k−1}) δ_{k−1},        k = 1..K_a                          (7.2)
   λ_k = ½ log ( ‖δ_k‖² / ‖δ_{k−1}‖² )                                                                (7.3)
```

with `J_k` the Jacobian of `f_k` at the point the stack actually reaches. This is the
correction of 11 §P-C.30: the first form of this section drew an **independent** isotropic
`δ` at each sub-block, which measures each map's response to an isotropic perturbation but
whose sum is *not* the exponent of the composed stack, because the perturbation that arrives
at layer `k` is `J_{k−1} ⋯ J_1 δ_0`, not an isotropic vector. With (7.2) the composition is
exact and (7.5) telescopes.

**The relative exponent.** The residual stream grows with depth (every sub-block adds to it)
and the output is RMS-normalised, so a perturbation that grows exactly as fast as the stream
leaves the *relative* perturbation `‖δ_k‖ / RMS(h_k)` unchanged and is the neutral case. The
quantity the rule and the band use is therefore

```
   λ_k^{rel} = λ_k − log ( RMS(h_k) / RMS(h_{k−1}) )                                                  (7.4)
```

`λ_k` (absolute) and `λ_k^{rel}` (relative) are both recorded; every threshold in this
document and in 10 is on the relative form. The JVP is forward-mode (`torch.func.jvp`) with
the exact double-backward trick as the fallback, in fp32 with autocast disabled (K5); the
attention sub-block is differentiated through an explicit softmax path, because the fused
SDPA kernels support neither (11 §P-C.18); the explicit path equals the fused one to `1e-5`
(a backbone test). Skipped layers are not in the list and contribute nothing (their map is
the identity, `λ = 0`). A non-finite exponent raises rather than moving a gain (K4).

## 7.2 The stack's exponent

**[NORMATIVE]**

```
   Λ_abs = Σ_k λ_k = ½ log ( ‖δ_{K_a}‖² / ‖δ_0‖² ),        Λ = Σ_k λ_k^{rel}                          (7.5)
```

The first equality is exact by (7.3): the per-sub-block exponents of one propagated tangent
telescope to the exponent of the composed stack, which is what makes the per-layer profile a
decomposition of `Λ` rather than a collection of unrelated numbers (K2b tests the identity to
`1e-4`). `Λ = 0` is the edge of chaos of the *trained* network on its own data: a relative
perturbation neither dies through the stack (order, `Λ < 0`: depth is wasted, the deep layers
compute a function of little of the input) nor explodes (chaos, `Λ > 0`: the gradient signal
through depth is noise). The profile is averaged over the probe's windows and draws; its
standard error across windows, `Λ_se`, is recorded beside it, and no verdict is read off a
`Λ` whose displacement from the band is inside `2 Λ_se`.

The pattern streams of 02 §2.9.6 carry their own *temporal* Lyapunov exponent `λ_ref` (a
property of the data's dynamics, cached over 100k steps of the generator, 02 §2.9.6), a
different quantity from the network's depth-wise `Λ`; R3 records both side by side (the probe
window's `λ_ref` when the probe fell on a pattern window) and reports the association as a
description, not a target.

## 7.3 The homeostatic rule

**[NORMATIVE]** `ζ ∈ R^{2L}` (one entry per residual sub-block; `|ζ| = 32` at the TS-adult)
enters the branch gains multiplicatively, 03 (3.3): `n_l^{mix} = n_{W,l}^{mix} ⊙ (1 +
γ_{n,l}^{mix}) · exp(ζ_l^{mix})`, likewise the FFN branch. At every consolidation, after the
probe (08 §8.2 O8):

```
   ζ_l^{mix} ← clamp( ζ_l^{mix} − κ · clip( λ_l^{mix,rel}, −1, 1 ),  −2, 2 ),   κ = 0.5,  likewise ffn   (7.6)
```

so a branch whose relative perturbations grow is turned down and one whose relative
perturbations die is turned up, by at most a factor `exp(κ) = 1.65` per episode and
`exp(2) = 7.4` over a life: synaptic scaling (Turrigiano 2008) applied to the residual
branches, at the slow rate, while the fast weights learn on top. Sub-blocks the probe did not
reach keep their `ζ`. `ζ` persists across episodes beside `g` (both are the agent's slow
channel) and is in the checkpoint. The genome's `γ_n` acts on the same gains, so the RL rule
can counteract (7.6) if criticality maintenance hurts learning; R3 records whether it does.

## 7.4 The probe's batch and its cost

**[NORMATIVE]** The probe runs at the boundary, after the episode's last step, on the
**episode's own last-step windows with the fast states they entered with** (up to
`n_pr = 16` windows, taken from the step's rollout-0 rows in batch order), not on a fresh
evaluation batch: the profile is a property of the network *as the episode left it*, on the
data it was reading, at the states it had. The probe uses its own generator seeded by
`(order_seed, episode)`, so the profile is reproducible and is not a function of the training
draw.

Cost per boundary: `n_pr n_δ = 16 · 4 = 64` stack-JVPs, each about twice the stack's forward,
i.e. `≈ 128` window-stack forwards, against the episode's `N B = 32,000` window-stack forward
and backward passes: `< 0.4 %`. The probe never runs inside the step loop
(G-NOFREE-INVERSE, K6), and its compute is charged to the episode's FLOPs (09 §9.1).

## 7.5 References and the standing of the claim

Mean-field theory locates trainability at the boundary `χ_1 = 1` for *initialisations*
(Poole et al., NIPS 2016: `χ_1 = σ_w² ∫ Dz φ′(√q* z)²`; Schoenholz et al., ICLR 2017:
depth scale `ξ_c^{-1} = −log χ_1`, trainable when `L ≲ 6 ξ_c`), and says residual networks
with normalisation sit near it by construction (Yang & Schoenholz, NIPS 2017: residual tanh
networks are "always at the edge"; Doshi, He & Gromov, NeurIPS 2023: LayerNorm plus residual
is "everywhere-critical"). Regularising toward criticality *during* training exists only at
toy scale (Tomihari & Karakida, NeurIPS 2025, `LE ≈ 0.1`; Benati 2025), and no verified work
holds a language model at criticality while it trains (11 §P-B.9–11). (7.3)–(7.4) are the
finite-time, trained-network, data-conditioned counterpart of `χ_1`; the expectation from the
theory is `Λ ≈ 0` under RMSNorm and residuals, in which case the rule (7.6) is idle. That is
why 10 makes K-CRIT conditional: **the profile is measured first (R3)**, and the rule is
tested only if the profile leaves the band during a life. The pattern streams give B6 its
"edge of chaos" training signal; (7.3)–(7.6) give the phrase a measured meaning inside the
network.

## 7.6 K-CRIT

**[NORMATIVE]** *Trigger.* K-CRIT is evaluated only if R3 (a 20-episode life at `γ ≡ 0` with
the rule off) shows, at some consolidation, either `|Λ| / K_a > 0.5` or `max_k |λ_k^{rel}| > 0.5`,
with the displacement larger than `2` of that limb's own standard error. **Both limbs are
per block**; `Λ` itself is recorded beside them and is not a limb, for the reason given
immediately below. *(Corrected 2026-09-05, 11 §P-D.26: this sentence still read `|Λ| > 0.5`
after the per-block correction below had been made and implemented, so the normative line and
`needs_control` disagreed. They first disagreed about an actual verdict on this date, when
`R4-tempo-tsk1-dg32-s0` reached `Λ = −0.628 ± 0.010` at episode 2 — over the stale line's
threshold by 12 σ, and inside the implemented band by a factor of 25.)* Otherwise the record says "not needed: the
normalised residual network stays critical", with the full profile, and K-CRIT is
**NOT NEEDED**, never a pass.

**The band is per block, and the gate is part of the trigger.** Two corrections, both from
the second adversarial round (11 §P-C.33):

*Per block.* `Λ` is a **sum** over the `K_a = 2 × |active layers|` sub-blocks. A fixed band on
it is therefore a band on depth as much as on criticality: the same per-block behaviour that
gives `Λ = −0.51` at the 8-layer primordium gives `−1.02` at the 16-layer adult, so growth
alone would fire the trigger. Both limbs are now on depth-intensive quantities — `Λ / K_a` and
`max_k |λ_k^{rel}|` — and `Λ` is recorded beside them.

*The gate is code, not prose.* §7.2's "no verdict is read off a `Λ` whose displacement from the
band is inside `2 Λ_se`" was written in this document and implemented nowhere: `needs_control`
compared `|Λ|` with the band and never looked at a standard error. It now returns the whole
trigger — both limbs, each displacement in units of its own standard error, and the verdict —
and the per-block standard errors it needs are computed by the probe.

**The observation that prompted this, corrected.** The first episode measured with the
propagated tangent (`runs/tempo/R1-nolat-ts-s0`, one episode of 25M tokens at the primordium,
`γ ≡ 0`, the rule off) recorded `Λ = −0.511 ± 0.053` over 16 blocks. An earlier draft of this
paragraph said that was "outside the band by about `2 Λ_se`". **That was wrong**: `|Λ|`
exceeds `0.5` by `0.011`, which is `0.21 Λ_se`, not two. The `9.7 σ` in that number is the
distance of `Λ` from *zero*, which is not what the trigger asks. Under the corrected trigger
the same measurement gives `Λ / K_a = −0.032` and `max_k |λ_k^{rel}| = 0.046`, both far inside
the band, and K-CRIT does **not** trigger on it. R3 still decides; this is one episode of one
seed.

**[NORMATIVE]** *The band is not calibrated, and R3 must report the distribution, not only the
verdict.* Two facts about that same profile say why. First, `±0.5` **per block** against a
measured `0.032` is `15.7x` of headroom on the stack limb and `10.8x` on the worst-block limb;
a per-block exponent of `0.5` compounds to `e^{8} ≈ 3000` across the adult stack, which is far
past any usable sense of "the edge of chaos". The per-block correction above fixed the
depth-dependence and inherited `±0.5` from the plan's absolute form without recalibrating it,
so as written the trigger is close to unfireable — a defect in the band, not a property of the
network. Second, the **absolute** exponent on that profile is `Λ_abs = +0.111`, of the opposite
sign to the relative `Λ = −0.511`: the raw tangent grows through depth but more slowly than
the activations, so (7.4)'s normalisation flips the sign. `λ^{rel}` is defined by this
specification and reported by no prior work, so neither reading has an external calibration.
R3 therefore records the **distribution** of `λ_k^{rel}` and `λ_k^{abs}` over episodes, layers
and seeds; the band is re-derived from that spread with the form and constant fixed before the
data are read; and the re-derivation is recorded as a correction (11 §P-D.1), never applied
quietly. A band nothing can cross is not evidence that nothing is wrong.

*Contrast, if evaluated.* The arm that isolates the rule is `b_homeo` (the `ζ` rule alone:
`γ ≡ 0`, no consolidation, `κ = 0.5`) against `a_gamma0` (the same life with `κ = 0`), paired
by seed under CRN, 4 seeds, compared at equal compute by the rule of 10 §10.5. K-CRIT fires
if `b_homeo` does not beat `a_gamma0`; the consequence is that (7.6) is removed and `ζ ≡ 0`.
Comparing the full agent against itself with `κ = 0` (the arm `nohomeo`) confounds the rule
with the genome's action on the same gains and is recorded as a secondary contrast only.

## 7.7 Acceptance tests

**[NORMATIVE]** `tempo/criticality.py` is accepted only when all of the following hold.
Implemented in `tests_tempo/unit/test_criticality.py` (K1–K5) and `test_life.py` (K6).

| # | property | test |
|---|---|---|
| K1 | JVP is a derivative | `J δ` from `torch.func.jvp` equals the central finite difference `(f(H + sδ) − f(H − sδ)) / 2s` at `s = 10^{-3}` to `1e-4` relative, on a random sub-block with a nonzero fast state |
| K2 | known exponent | on a sub-block replaced by `H ↦ c H` (a scalar map), `λ = log c` to `1e-6` for `c ∈ {0.5, 1, 2}`; on `H ↦ Q H` with `Q` orthogonal, `λ = 0` to `1e-6` |
| K2b | the tangent telescopes | for one window and one `δ`, `Σ_k λ_k` equals `½ log(‖J_stack δ‖²/‖δ‖²)` computed by a single JVP through the composed stack, to `1e-4`; the relative and absolute forms differ block by block |
| K3 | identity sub-block | with `m_l = 0`, `λ = 0` exactly; skipped layers are not probed (their keys are absent from the profile); `Λ` equals the sum of the reported per-block entries |
| K3b | the entering states matter | the profile computed with the windows' live entering states differs from the profile at a zero state (the probe is not silently state-free) |
| K4 | the rule's direction and the trigger | planting `λ^{rel} = +0.4` decreases `ζ` by `0.2`; planting `λ^{rel} = −3` increases it by `0.5` (the clip); the clamp at `±2` holds and touches **only** the sub-blocks the probe reached; a non-finite `λ` raises `FloatingPointError`. The trigger fires only when a *per-block* displacement past the band exceeds `2` standard errors: a profile at `Λ = −0.511 ± 0.053` over 16 blocks does **not** fire it |
| K5 | fp32 | the probe runs in fp32 with autocast disabled (asserted by the dtype of every returned tangent; a bf16 map raises) |
| K6 | once per episode | the probe and the rule execute once per boundary and never in the step loop (sentinel counters in `Life`) |
| K7 | record | every episode's record carries `λ_l^{mix,rel}, λ_l^{ffn,rel}`, their absolute forms and their standard errors for every active layer, `Λ`, `Λ_abs`, `Λ_se`, `Λ/K_a`, `K_a`, `ζ`, the trigger's verdict with both displacements in units of their standard error, the probe's streams and **its seed** |
| K8 | the reference exponents | the generators' `λ_ref` (02 §2.9.6) equal the literature values within tolerance (I10) and are written beside `Λ` in the record when the probe fell on a pattern window |
