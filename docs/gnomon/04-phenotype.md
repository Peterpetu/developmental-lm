# 04 — The phenotype

The network that `Gamma` decodes. Nothing here is novel and nothing here is meant to be:
the operators are the well-understood ones, vendored from SOMA where they were already
verified, so that every difference between GNOMON and a strong baseline is attributable
to the decoder rather than to the backbone.

## 4.1 Operators are functions, not modules

**[NORMATIVE]** Every operator is a **pure function of activations and weights**:

```
    y = attention(x, W_q, W_k, W_v, W_o, sink, gain, *, cfg)
    y = delta_mixer(x, W_q, W_k, W_v, W_beta, W_o, gain, *, cfg)
    y = swiglu(x, W_gate, W_up, W_down)
```

No operator owns an `nn.Parameter`. This is the structural enforcement of C1: a module
that owned weights would be a `W_shared` waiting to happen, and §00's RC1 says that
whatever backpropagation can bypass, it bypasses. The vendored SOMA modules
(`Attention`, `Mixer`) hold their own `nn.Linear`s and are therefore **refactored to
functional form** during vendoring; the numerical cores (`gated_delta_rule_chunk`,
`gated_delta_rule_reference`, `causal_mask`, `document_mask`, `rms_norm`) are pure already
and are vendored verbatim with a provenance header.

**[NORMATIVE]** The refactor is required to be behaviour-preserving: the vendored tests
`test_mixer_equivalence`, `test_attention_*` and `test_mixer_chunk_oracle` are re-run
against the functional forms and must pass unchanged. In particular the corrected
chunkwise identity (SOMA's (5.15), whose original derivation dropped the `g^r/g^i` factor
and would have shipped a subtly wrong mixer) is preserved and re-verified against
`gated_delta_rule_reference` at `1e-10` in float64.

## 4.2 Matrix roles

**[NORMATIVE]** Exactly 16 roles, `n_M = 4` bits, fixed for the life of the project.
Adding a role after genomes exist would silently change what every existing genome
decodes.

| `m` | name | shape | kind |
|---|---|---|---|
| 0 | `Q` | `n_h*d_head x d_model` | matrix |
| 1 | `K` | `n_kv*d_head x d_model` | matrix |
| 2 | `V` | `n_kv*d_head x d_model` | matrix |
| 3 | `O` | `d_model x n_h*d_head` | matrix |
| 4 | `FFN_GATE` | `d_ff x d_model` | matrix |
| 5 | `FFN_UP` | `d_ff x d_model` | matrix |
| 6 | `FFN_DOWN` | `d_model x d_ff` | matrix |
| 7 | `EMB` | `V x d_model` | token matrix (tied to the output head) |
| 8 | `NORM_MIX` | `d_model` | vector, gain-like |
| 9 | `NORM_FFN` | `d_model` | vector, gain-like |
| 10 | `NORM_FINAL` | `d_model` | vector, gain-like |
| 11 | `SINK` | `n_h` | vector |
| 12 | `DELTA_BETA` | `n_h*d_head x d_model` | matrix (the delta-rule write strength) |
| 13 | `PLAST_CHAN` | `d_model` | vector, `eta` (§03.4) |
| 14 | `PLAST_TOK` | `V` | token vector, `eta` (§03.4) |
| 15 | `LOGIT_SCALE` | `1` | vector of length 1 |

**[NORMATIVE]** Roles 0–3 and 12 are shared between the attention and delta-mixer
operators: a layer decoded as attention uses `{Q,K,V,O,SINK}`, one decoded as a delta
mixer uses `{Q,K,V,O,DELTA_BETA}`. The sharing is deliberate — it means the *same* genome
region is under selection whichever operator a layer is assigned, so a mutation in `Gamma_a`
that flips a layer's operator (§07) lands in a partially prepared genome rather than on
untrained ground.

## 4.3 The block

**[NORMATIVE]** Peri-LN residual placement, as SOMA §05.1, which is retained because it
was verified and because it is the arrangement least sensitive to weight scale — a
property that matters more here, where the weights come from a decoder that a mutation
may have moved.

```
    h  <- x + Op_l( rms_norm(x, gain=NORM_MIX_l) )
    x' <- h + swiglu( rms_norm(h, gain=NORM_FFN_l) )
```

with a final `rms_norm(·, gain=NORM_FINAL)` before the tied output head, and logits scaled
by the decoded `LOGIT_SCALE`.

## 4.4 Configurations

**[NORMATIVE]** Three, fixed now so that no configuration is chosen after seeing a result.

| | M-pico | M-nano | M1 |
|---|---|---|---|
| purpose | evolution | verification, the `d_z` frontier | language |
| `d_model` | 128 | 192 | 512 |
| `L` (max, `Gamma_a` may decode fewer) | 4 | 6 | 12 |
| `n_h` / `n_kv` / `d_head` | 4 / 2 / 32 | 6 / 2 / 32 | 8 / 2 / 64 |
| `d_ff` | 512 | 768 | 2048 |
| vocab | 4096 | 4096 | 8192 |
| `T` | 128 | 128 | 512 |
| decoded params (under `Arch.uniform`) | 1,528,969 | 4,060,883 | 49,926,705 |
| `N_life` | 600 | 2000 | 20000 |

`d_z` is **not** fixed here: it is the dial of §06.5 and is set by measurement, not by
choice.

## 4.5 Soma, enumerated

**[NORMATIVE]** `Z` is exactly these tensors:

| tensor | shape | reals at M1, `d_z = 16` |
|---|---|---|
| `z` (channel identities) | `d_model x d_z` | 8,192 |
| `zeta` (token identities) | `V x d_z` | 131,072 |
| `y` (layer states) | `L x d_y` | 96 |
| | **`\|Z\|`** | **139,360** |

There is nothing else. `Z` is reinitialised at the start of every lifetime from
`N(0, 1/d_z)` under a per-lifetime seed derived from the run seed and the generation
index, so that two lifetimes of one genome differ only by that seed.

## 4.6 The absence gate

**[NORMATIVE]** `G-NOFREE` walks the whole model and asserts that the set of tensors with
`requires_grad=True` is exactly `Gamma_w ∪ Gamma_a ∪ Z`, and that no buffer or parameter
anywhere has a shape matching any decoded matrix role. It is an **absence** check, and an
absence check is worthless unless it has been shown to fire: the gate runs first against a
**planted positive control** — a model with one extra free `d_model x d_model` parameter
added to the residual path, exactly SOMA's `W_shared` — and the gate must fail on it
before its pass on the real model is credited.

**[NORMATIVE]** `G-NOFREE` additionally asserts that the optimizer's parameter groups
(§05.4) cover `Z` and, in the Darwinian arms, nothing else — a free tensor that exists but
is not optimized is not a bypass, and a tensor that is optimized but was not declared is.

## 4.7 Acceptance tests

**[NORMATIVE]** The phenotype is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| P1 | **twin equivalence** | decode `Gamma` into `W`, load the identical `W` into a conventional free-weight transformer of the same shape; logits agree to `1e-6` on random input. This is what makes the compression claim about *this* model rather than about a different one |
| P2 | vendored numerics | the SOMA oracle tests pass against the functional refactor, unchanged |
| P3 | shapes | every decoded role has the shape in §4.2's table, asserted at the module boundary |
| P4 | masking | causal and document masks leak nothing: a token's logits are invariant to every future token and to tokens of other documents |
| P5 | `G-NOFREE` | as §4.6, positive control first |
| P6 | determinism | two forward passes of one `(Gamma, Z, batch)` are bit-identical |
| P7 | inference cache | decode-once-then-cache produces logits identical to decode-per-step, and is measurably faster (`G-INFER`) |
