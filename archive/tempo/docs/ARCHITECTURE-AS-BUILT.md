# TEMPO — the architecture as actually built

**What this document is.** A description of the TEMPO model reconstructed **from the source
code alone**, written 2026-09-04 while the R2 protocol stage was running. No other document in
this directory was consulted while writing it. Where the code and a specification disagree, the
code is what this file reports, because the code is what ran and produced the numbers in §17.

**This is the second draft.** The first was audited by three adversarial reviewers reading only
the source and the run artifacts. They returned **45 findings, 8 of them critical** — including
three places where the first draft described a mechanism that does not exist. Everything they
found is corrected below, and §19 lists what was wrong, because a document that was checked and
a document that was right the first time are not the same object and a reader should be able to
tell them apart.

**How to check any claim.** Line references are given as `file:line` against commit `aeb98e7`.
The authoritative modules:

```
  tempo/config.py        shapes and the provisioned/active distinction
  tempo/modalities.py    the input path: encoders, modality codes, time codes
  tempo/codec.py         the output path: span codec, text likelihood, continuous heads
  tempo/backbone.py      the network: one span step
  tempo/genome.py        the genome, the role registry, the expression map
  tempo/morphology.py    switches, births, FLOP counting
  tempo/consolidate.py   the reverse-time consolidator, the reward, the estimators
  tempo/criticality.py   the Lyapunov probe and the homeostatic rule
  tempo/envs.py          the policy-gradient term and the verifier environments
  tempo/streams.py       the seven data sources
  tempo/life.py          the loop that runs all of it
```

---

## 1. The object in one page

TEMPO is one network that consumes a stream of **continuous vectors** and emits **continuous
vectors**. Text, images, sound, proprioception and control actions are all encoded into the
same vector space `R^256` on the way in, and all decoded from the same vector space on the way
out.

The backbone has **no vocabulary-sized output layer**: it predicts a 256-vector, and a small
decoder turns that into a distribution over 4 tokens. There *is* a token embedding on the input
side (`TextIn.T_emb`, modalities.py:81, 1.18M parameters) — the asymmetry is deliberate and is
described in §4.

Three things change at three different rates. That is the whole idea.

```
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │                            ONE AGENT, THREE CLOCKS                           │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                                                                              │
   │  τ0   FAST STATE  S          every span (~every K tokens)                     │
   │       a 64×64 matrix per head of the 12 DELTA layers, rewritten by an         │
   │       online least-squares rule.  The 4 ATTENTION layers have no S — they     │
   │       keep a KV cache instead.        backbone.py:150  backbone.py:281        │
   │                                                                              │
   │  τ1   WEIGHTS                every optimiser step (B windows)                 │
   │       78,801,485 trainable parameters — the backbone (70.2M) PLUS the         │
   │       interface, the heads, the codec's decoder and the differentiable        │
   │       gains.  Muon on 2-D matrices, AdamW on everything else.                 │
   │                                       life.py:346                             │
   │                                                                              │
   │  τ2   GENOME  g ∈ R^512      once per episode (N steps; N = 250 by default,   │
   │       expressed into 243,133 gains; moved by an RL-trained rule that reads     │
   │       the episode BACKWARDS.          life.py:1002                            │
   │                                                                              │
   └──────────────────────────────────────────────────────────────────────────────┘
```

`N` is a **run parameter, not an architectural constant**: `LifeConfig.N = 250` is the default
(life.py:136) and the R1 runs of §17 used `N = 1525`. Numbers below that depend on `N` are
given for both.

The claim the project is testing is that the slowest clock — a 512-number genome that stores no
knowledge, only **gains** (multipliers on weights, learning rates, and which parts of the
network are switched on) — makes the faster clocks learn better, and that the rule for updating
it can be learned by reinforcement learning whose reward is *the agent's own improvement per
unit of compute*.

### The data path, end to end

```
   raw input                                                         raw output
   (tokens, pixels,     ┌───────────┐   x_t ∈ R^256   ┌──────────┐   (tokens, pixels,
    frames, joints)  ──►│  ENCODER  │───────────────► │ BACKBONE │──► ŷ ∈ R^256 ──┐  frames,
                        │   E_m     │                 │    F     │                │  torques)
                        └───────────┘                 └──────────┘                │
                              ▲                             │                     ▼
                              │                             │              ┌────────────┐
                     + μ_m (modality code)                  │              │  DECODER   │
                     + τ(t)  (wall-clock code)              │              │    D_m     │
                                                            │              └────────────┘
                                              fast state S (delta layers)         │
                                              KV cache      (attn layers)         │
                                                                                  │
        the agent's own emission is fed back in through E_m ◄──────────────────────┘
                              (the egocentric loop, life.py:1365)
```

`Perception.forward` (modalities.py:135) is the only way into the backbone:

```
   x_t  =  unit_rms( E_m(raw_t) )  +  μ_m  +  τ(t_w)
           └── encoder output ──┘   └ what ┘  └ when ┘
```

`μ_m` is a learned 256-vector per modality (9 modalities). `τ(t_w)` is a Fourier code of the
event's wall-clock time over 14 periods from 1/64 s to 128 s (modalities.py:66) — this is what
lets a 100 Hz pattern stream and a 4 Hz text stream coexist in one sequence.

---

## 2. Shapes and counts (TS-adult)

From `tempo/config.py:126`, verified by instantiating the modules:

```
   ┌─────────────────────────┬──────────┬────────────────────────────────────────────┐
   │ d                       │      512 │ residual width (never switched off)        │
   │ L                       │       16 │ layers, provisioned                        │
   │ h / h_kv / d_head       │  8 / 2/64│ heads, key-value heads, head width         │
   │ F / slice               │ 2048/ 256│ FFN width, switchable slice size           │
   │ d_x                     │      256 │ THE INTERFACE WIDTH — in and out           │
   │ K                       │        4 │ tokens per text span                       │
   │ T / T_s                 │  512/ 128│ tokens / spans per window                  │
   │ d_g                     │      512 │ genome dimension                           │
   │ r                       │       32 │ rank of the expression map                 │
   │ d_c                     │      256 │ codec and consolidator width               │
   │ vocab                   │     8192 │ tokenizer size                             │
   └─────────────────────────┴──────────┴────────────────────────────────────────────┘
```

### Parameter budget (measured)

```
   backbone (provisioned)         70,198,480
   span codec                      9,650,176   ← of which 3,809,280 FROZEN (the encoder)
   span embedding (text in)        1,179,648
   other encoders                    583,616
   continuous heads                  989,373
   perception (μ + τ)                  9,472
                                  ──────────
   the agent                      82,610,765
   of which trainable at τ1       78,801,485   ← + the differentiable gains u_diff

   expression map θ               11,302,816   ← turns 512 numbers into 243,133 gains
   consolidator ψ                  5,201,152   ← the reverse-time reader
   genome g                              512   ← THE SLOW CHANNEL
   ζ (homeostatic log-gains)              32   ← one per SUB-BLOCK: 2 per layer
```

The genome is 512 numbers; the machinery that reads it (11.3M) and writes it (5.2M) are each
four orders of magnitude larger. The bottleneck is structural, not a regulariser.

### The layer layout

`config.op(l)` (config.py:86) puts attention at every 4th layer:

```
   layer   0  1  2  3   4  5  6  7   8  9 10 11  12 13 14 15
   op      d  d  d  a   d  d  d  a   d  d  d  a   d  d  d  a
           └──────────┘ └──────────┘ └──────────┘ └──────────┘
             quarter 0    quarter 1    quarter 2    quarter 3
           ██ PRIMORDIUM               ██ PRIMORDIUM

   d = gated delta rule (linear, O(1) state)      12 layers  ── these own S
   a = softmax attention, GQA 8 query / 2 kv       4 layers  ── these own a KV cache
```

**Provisioned vs active.** All 16 layers are allocated at birth; only quarters 0 and 2 are
switched on.

```
   active parameters   primordium 17,886,264 (25.5 %)   adult 70,198,480 (100 %)
   backbone train FLOPs/window   prim 1.3923e10   adult 5.4662e10   ratio 3.926
```

---

## 3. The three timescales

```
  ══════════════════════════════════════════════════════════════════════════════════
   τ0 — THE FAST STATE            backbone.py:_delta   (DELTA LAYERS ONLY)
  ══════════════════════════════════════════════════════════════════════════════════

   Each head of each of the 12 delta layers owns  S ∈ R^{64×64}, updated once per span:

        S_t  =  S_{t-1} · α_t · ( I − β_t k_t k_tᵀ )  +  β_t v_t k_tᵀ
                └── forget ──┘  └─ erase old key ─┘     └─ write new ─┘
        o_t  =  S_t q_t

   α_t = sigmoid(w_α·h + a_0)          per-span forget gate,  a_0 init 4.0
   β_t = 2·sigmoid(w_β·h + g_0), clamped                      g_0 init 0.0
   k and q are L2-normalised.

   a_0 and g_0 are ordinary τ1 PARAMETERS (backbone.py:110); the genome adds a
   bounded offset to them (|γ| ≤ 2.0).  The genome shifts these gates; it does
   not own them.

   S survives across windows and episodes; it is reset only at a declared boundary.

  ══════════════════════════════════════════════════════════════════════════════════
   τ1 — THE WEIGHTS               life.py:849
  ══════════════════════════════════════════════════════════════════════════════════

   B windows × T_s spans → one forward, one backward, one optimiser step.
   Muon on the 2-D matrices, AdamW on vectors/embeddings/heads/decoder.
   LR: warm-up 2 %, flat, linear decay-to-zero from 80 % — WITHIN each episode.

  ══════════════════════════════════════════════════════════════════════════════════
   τ2 — THE GENOME                life.py:1002
  ══════════════════════════════════════════════════════════════════════════════════

   Once per episode.  THE GENOME RECEIVES NO GRADIENT FROM ANY LOSS
   (genome.py:express is @torch.no_grad).  It moves only by the rule of §10.
```

---

## 4. The input path

```
   text (K tokens) ─► TextIn: embed each token (128-d), concatenate, linear → 256
   image 3×32×32   ─► 4 stride-2 convs 3→32→64→128→256, GELU, mean-pool, → 256
   speech 64-bin   ─► per-frame MLP 64→128→256→256   (NO temporal convolution)
   proprio 8 / reward 1 / logistic 8 / lorenz 3 / eca 64
                   ─► AFFINE  W·raw + b → 256                 modalities.py:221
                                    │
                                    ▼
                    unit_rms(·) + μ_m + τ(t_w)                modalities.py:141
                                    │
                                    ▼   x_t ∈ R^256
```

**The small encoders are affine, not linear** (modalities.py:221, `bias=True`). Because
`Perception` divides by RMS, a purely linear map would destroy the magnitude of the input — a
scalar reward would be reduced to its sign.

**The speech encoder is per-frame** (modalities.py:201). A temporal convolution inside the
encoder would give a frame different features when the agent is *acting* (one frame at a time)
than when it is *training* (a whole window). The backbone carries the temporal context.

**The text input path shares no parameter with the text output path.** `TextIn` and `SpanCodec`
have separate embedding tables. The code states this as a design invariant — the input side is
trained with the backbone at τ1, and the codec serves the output likelihood only
(modalities.py:74). *No ablation of a shared embedding exists in this repository*, so the
separation is a stated design decision, not a measured result.

---

## 5. The output path: the span codec

```
   ENCODER  E_txt   (FROZEN after R0 — it defines the latent space)
   ═══════════════════════════════════════════════════════════════════════════

     v₁ v₂ v₃ v₄ ─► embed+pos ─► 2 BIDIRECTIONAL blocks (d_c=256)
                                      │
                            LayerNorm per position, THEN mean-pool    codec.py:134
                                      │
                                 ┌────┴────┐
                                 ▼         ▼
                                μ_z      log s
                                 │
                          x̄ = unit_rms(μ_z)

   DECODER  D_txt   (TRAINS WITH THE BACKBONE — it is a conditional LM)
   ═══════════════════════════════════════════════════════════════════════════

     ŷ ──► W_c ──┐
                 ▼
     BOS v₁ v₂ v₃ ─► embed+pos ─► ⊕ ─► 2 CAUSAL blocks ─► LN, W_v, softcap
                                                              │
                                    p(v₁|ŷ) p(v₂|ŷ,v₁) p(v₃|…) p(v₄|…)

     L_txt = − Σ_{i=1..K} log p(v_i | ŷ, v_<i)        codec.py:152
```

**What the backbone is actually trained to do.** At the as-built default `λ_lat = 0`
(codec.py:50, life.py:158) the *only* text loss is the decoder NLL (plus a z-loss, §7). **`x̄`
is not a regression target.** It defines the geometry the frozen encoder induces and is what
the decoder was trained to consume, but nothing pulls `ŷ` toward it. The `λ_lat = 0.1` variant
that *does* regress `ŷ` to `x̄` is a recorded ablation, and it costs 0.23 nats (§17). When
`λ_lat = 0` the encoder is not even executed during training (life.py:588).

**The codec is trained under noise on purpose** (codec.py:171): the latent handed to the decoder
during codec training is `unit_rms(μ_z + s·ε₁) + 0.3·ε₂` with 15 % of coordinates dropped, so
the decoder tolerates a *predicted* latent that lands near but not on the true one.

**Reconstruction ceiling, measured:** 0.00137 nats/token at K=4, 0.00071 at K=2 — negligible
against ~1.9–2.4 nats of language modelling.

### The other decoders

```
   GaussianHead   a ~ N(W_a ŷ, diag exp(2 w_s))    ← control actions ("act"), speech
                                                     frames, logistic, lorenz
   BernoulliHead  bits ~ sigmoid(softcap(W_b ŷ))   ← cellular-automaton stream
   ImageDecoder   ŷ → 256×2×2 → 4 transposed convs → 3×32×32, squared error
   FlowHead       rectified flow, 10 Euler steps   ← IMPLEMENTED AND UNIT-TESTED,
                                                     BUT NOT WIRED IN.  The production
                                                     action head is GaussianHead
                                                     (life.py:202).
```

---

## 6. The backbone: one span step

```
  x_t ∈ R^256  ──►  W_in + b  ──►  h⁰ ∈ R^512
      │
      ├──────────────────  for each ACTIVE layer l  ───────────────────┐
      │   ĥ = RMSNorm(h) ⊙ ρ_l^mix                                     │
      │   y = Op_l(ĥ)                    delta rule OR attention       │
      │   h ← h + m_l · n_l^mix · exp(ζ[2l])   · RMSNorm(y)            │
      │                                                                │
      │   ĥ = RMSNorm(h) ⊙ ρ_l^ffn                                     │
      │   f = W_down·( SiLU(W_gate ĥ) ⊙ W_up ĥ )        SwiGLU         │
      │   h ← h + m_l · n_l^ffn · exp(ζ[2l+1]) · RMSNorm(f)            │
      └────────────────────────────────────────────────────────────────┘
      │
      ▼  RMSNorm ⊙ ρ_fin,  then W_out
  ŷ_{t+1} ∈ R^256          ← no softmax, no vocabulary
```

This is **Peri-LN**: normalise going into the branch, and normalise the branch output before
adding it back. `ζ` has one entry per **sub-block** — `ζ[2l]` for the mixer, `ζ[2l+1]` for the
FFN (backbone.py:345, 354).

The **gain fold** (backbone.py:190) is how the genome touches a matrix:

```
        W_eff  =  diag(1 + a_out) · W · diag(1 + a_in)          |a| ≤ 0.5
```

applied *in the graph*, so `∂loss/∂a` exists and the differentiable gains train at τ1. It costs
two matrix-shaped elementwise products per matrix per forward — cheaper than a matmul, but not
free.

**Skipping does both things.** `h_act(l)` / `F_act(l)` (morphology.py:113, 116) slice the weight
tensors to a prefix, *and* the branch is multiplied by a mask carrying the group multiplier
(backbone.py:243, 308). Note the consequence: `h_act` is all-or-nothing, so a group ramping in
at `m = 0.02` **costs full compute while contributing 2 %**.

---

## 7. What the loss actually is

```
   text / task spans  ──►  L_txt = −Σᵢ log p(vᵢ | ŷ, v_<ᵢ)   +   z_loss·Σᵢ lse²
                                                                  Z_LOSS = 1e-4
   speech / patterns  ──►  Gaussian or Bernoulli NLL
   images             ──►  squared error through the image decoder
   control actions    ──►  no supervised target

   plus, per complete rollout group:

       Â_i = (r_i − mean r) / (std r + 1e-6)          envs.py:157
       a group with < 2 members or zero spread is DROPPED entirely

   total  ←  (Σ everything) / DENOMINATOR
```

**The policy-gradient term is REINFORCE, not a clipped-ratio method.** `window_loss` calls
`grpo_loss(LP, LP.detach(), …)` (life.py:640), so `ρ = exp(LP − LP.detach()) ≡ 1` and
`min(ρA, clip(ρ)A) = A` at every evaluation. The clipping machinery exists (envs.py:179) but
never binds: it is a single on-policy sample with a group-normalised advantage.

**The denominator** is the number of things the loss actually sums over: scored supervised
tokens plus the policy-gradient positions of complete groups (life.py:822). A rollout row with
no supervised target contributes `0`, not a whole window.

**Gradient insulation** (modalities.py:280): each non-text head reaches the backbone through
`κ·ŷ + (1−κ)·stopgrad(ŷ)`, with `κ = 0` until the head's own loss has **fallen by half the
first episode's magnitude** (`first − loss ≥ 0.5·|first|`, modalities.py:307 — not "halved",
which would be ill-defined for a density whose NLL can be negative), then a linear ramp to 1
over one episode.

---

## 8. The genome

```
   g ∈ R^512
     │   u_ρ = B_ρ·tanh(A_ρ g)                          DIFFERENTIABLE roles
     │   u_ρ = B_prog·(tanh(A_ρ g) − tanh(A_ρ g₀))      PROGRAMME roles
     ▼
   γ = c_ρ·tanh(u)
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  kind      roles   gains      c_ρ      where it acts                     │
   ├──────────────────────────────────────────────────────────────────────────┤
   │  matrix      142   209,600    0.5      diag gains on every weight matrix │
   │  norm         65    33,280    0.2      RMSNorm gains, branch gains       │
   │  offset        3       208    2.0      α/β gates (192), output gate (16) │
   │  lr            1         4    log 4    learning-rate multiplier ×4 groups│
   │  morph         3        34    8.0      WHICH BLOCKS ARE SWITCHED ON      │
   │  mixture       1         7    4.0      probability over the 7 streams    │
   ├──────────────────────────────────────────────────────────────────────────┤
   │  TOTAL       215   243,133             compression 474.9 : 1             │
   └──────────────────────────────────────────────────────────────────────────┘
```

The 16 output-gate offsets are on **every** layer, not only the attention ones: `GOG` is the
delta layers' SiLU output gate as well (backbone.py:242, 298), initialised 0.0 for delta and
2.0 for attention.

**Two kinds of role.** The *differentiable* kinds enter the loss, so `u` for those rows is a
free parameter at τ1 and `B` is re-fitted by regression to whatever τ1 discovered
(genome.py:203). The *programme* kinds (learning rate, morphology, stream mixture) are consumed
by the optimiser, the switch pattern and the data sampler — **none of which is differentiable**.
There is no gradient to regress toward, so those rows use a fixed random frame applied to the
displacement from the prior, which is exactly zero at `g = g₀` for any frame. Without this,
three of the design's five claims could never leave zero.

**The prior is not zero** (genome.py:59): `g₀ ~ N(0, I)`. The identity at birth is carried by
`B = 0`, not by `g = 0` — at `g = 0` the expression `B·tanh(0)` is zero whatever `B`, so the
regression would have nothing to fit.

**Bounds.** `‖g − g₀‖ ≤ 3√512 ≈ 67.9`; each step projected to `‖Δg‖ ≤ 0.2√512 ≈ 4.53`; 1 %
decay toward the prior per episode.

---

## 9. Latent morphology

```
   a_G = −4 + γ_G          m_G = sigmoid(a_G), and SKIPPED when m_G < 0.02

        γ_G = 0      →  m = 0.018  →  skipped          (birth)
        γ_G ≥ 0.108  →  m ≥ 0.020  →  the block turns on
        γ_G = +8     →  m = 0.982

   BIRTH: layers 0-3, 8-11 · heads 0-3 of 8 · slices 0-3 of 8 · 17.9M active
   ADULT: all 16 layers   · all 8 heads    · all 8 slices    · 70.2M active
```

**Growth is by copy** (morphology.py:apply_births): heads `ph..h-1` copy `0..ph-1`, slices
likewise, layer `l` copies layer `l − L/4` (same operator by construction — checked in
config.py:64). Most weights get 1 % RMS noise, but the **convolution taps and the gate offsets
(`CQ, CK, CV, A0, G0, GOG`) are copied exactly**, with no noise (morphology.py:270, 291, 294).

A block cannot be born at `m = 0.018`, because that value *is* the skip threshold: the genome
must first push `γ_G` past 0.108. Once born, the branch multiplier ramps in with `m_G`.

---

## 10. Consolidation: the reverse-time process and the RL that trains it

```
   ══════════════════════════════════════════════════════════════════════════════
    STEP 1 — SUMMARISE EVERY WINDOW                         consolidate.py:80
   ══════════════════════════════════════════════════════════════════════════════
        r_w = [ h̄_w (512) ; ℓ_w (1) ; μ̄_w (256) ; n/N (1) ]
        N·B windows per episode  (48,800 at N=1525·B=32; 8,000 at N=250)

   ══════════════════════════════════════════════════════════════════════════════
    STEP 2 — READ THEM BACKWARDS                            consolidate.py:274
   ══════════════════════════════════════════════════════════════════════════════

      stored order:      w₁  w₂  w₃ ............ w_{n-1}  w_n
      processed order:   w_n ......................... w₂  w₁
                                                            ▲
                                              read-out taken HERE
                                              (c_seq[-1] — "last reached")

      The mixer is causal and recency-weighted, so the read-out is dominated by
      whatever it processed MOST RECENTLY.  Reading backwards, that is the
      episode's BEGINNING.  Reading forwards (the K-REV control), it is the
      episode's END.  Both take "last reached + mean over all windows"; only the
      direction differs, so K-REV asks whether a summary anchored on how the
      episode began beats one anchored on how it ended.

   ══════════════════════════════════════════════════════════════════════════════
    STEP 3 — PROPOSE A GENOME CHANGE                        consolidate.py:293
   ══════════════════════════════════════════════════════════════════════════════
      μ     = W_μ R
      log σ = log 0.02 + (log 0.2 − log 0.02)·sigmoid(W_σ R + b_σ)
      Δg    ~ N(μ, diag σ²)

      b_σ is INITIALISED to 3.892 (σ ≈ 0.19, near the top of the range) and is a
      TRAINED parameter in head_parameters() — not a constant.

      The scale head is a SQUASHED map, not a clamped one.  With a clamp and
      W_σ = 0 the pre-activation sits outside the bound, where clamp has zero
      gradient — the head was dead by construction twice before this form.

   ══════════════════════════════════════════════════════════════════════════════
    STEP 4 — THE REWARD                                     consolidate.py:367
   ══════════════════════════════════════════════════════════════════════════════
                    ( L_e  −  L_{e+1} )        F_ref
        r_e   =     ───────────────────   ·   ───────
                            1                  F_{e+1}

      L is L_check: the uniform mean over the streams present of each stream's
      score on a RESERVED FITNESS SET (disjoint from the evaluation sets,
      streams.py:55), scored from a fresh state, each divided by that stream's
      own scale s_m.  Because of that division L_check is DIMENSIONLESS — the
      reward is a standardised improvement per primordium-episode of compute,
      not a number of nats.

   ══════════════════════════════════════════════════════════════════════════════
    STEP 5 — THE POLICY UPDATE
   ══════════════════════════════════════════════════════════════════════════════

   (1) REINFORCE, one sample, zero overhead        consolidate.py:384
       ψ ← ψ + η₂(r_e − b)∇_ψ log N(Δg ; μ, σ),  b an EMA of past rewards.
       Applied BEFORE the new proposal is drawn, so every proposal is a sample
       of the policy whose gradient scores it.

   (2) BEST-OF-G with an elite regression          life.py:1236
       G = 4 candidates + a NULL probe (Δg = 0), n_probe steps each under common
       random numbers; keep the winner if it beats the null; regress the policy
       toward it with a KL penalty.

       n_probe is set by the protocol as max(5, N//10) — 25 at the default
       N = 250 (tempo_protocol.py:113), NOT the module default of 100.
       Overhead = (G+1)·n_probe / N = 5·25/250 = 50 %.

           ┌─── snapshot ───┬────────┬────────┬────────┬────────┐
           │   Δg = 0 null  │  Δg₁   │  Δg₂   │  Δg₃   │  Δg₄   │
           └───────┬────────┴───┬────┴───┬────┴───┬────┴───┬────┘
                 r_null        r₁       r₂       r₃       r₄
                                    argmax → keep its state, regress ψ toward it
```

**Salience and rehearsal.** The same reverse read gives `s_w = sigmoid(w_sᵀ c_w)`. The buffer
holds `0.10·N·B` windows — **4,880 at N=1525, 800 at the default N=250** (life.py:91, 292) —
twice what the 5 % rehearsal phase consumes, so the ranking can actually select. The salience
head is trained by regression on the *measured* improvement of a 512-window probe set.

---

## 11. Criticality

```
   ONE tangent is propagated through the WHOLE active stack:

      δ₀ = random,   δ_k = J_k δ_{k−1}      through each sub-block

      λ_k     = ½ log( ‖δ_k‖² / ‖δ_{k−1}‖² )              ABSOLUTE
      λ_k^rel = λ_k − log( RMS(h_k)/RMS(h_{k−1}) )        RELATIVE   criticality.py:137

   Σ_k λ_k  telescopes exactly to ½ log(‖δ_last‖²/‖δ₀‖²)  — this is Lambda_abs.
   Σ_k λ_k^rel does NOT telescope — and THIS is what the code calls `Lambda`
   (criticality.py:182), and what the rule, the band, trigger() and every record use.
```

The relative form exists because the residual stream grows with depth: for a *relative*
perturbation the neutral point is not 0 but the stream's own growth rate. The homeostatic rule
then nudges each branch toward `λ^rel = 0`:

```
      ζ_j ← clamp( ζ_j − 0.5·clip(λ_j^rel, −1, 1), −2, +2 )   and the branch ×= exp(ζ_j)
```

Computed by forward-mode AD (`torch.func.jvp`) in fp32 with autocast disabled, on 16 windows ×
4 draws, once per episode at O5 — never inside a training step.

---

## 12. The seven streams

```
  ┌───┬────────────┬──────────┬───────┬─────────────────────────────────────────────┐
  │ 1 │ text       │ tokens   │ 4 Hz  │ language          (reads the corpus shards) │
  │ 2 │ task       │ tokens   │ 4 Hz  │ algorithmic reasoning, EXACT verifier reward│
  │ 3 │ control    │ 8-d + 1  │ 20 Hz │ continuous actions, point-mass environment  │
  │ 4 │ vision     │ 3×32×32  │ 5 Hz  │ scene understanding, high-d slow input      │
  │ 5 │ speech     │ 64 bins  │62.5Hz │ transcription AND production                │
  │ 6 │ pattern    │ 3–64 d   │100 Hz │ ◄ THE FASTEST RATE. logistic / Lorenz /     │
  │   │            │          │       │   cellular automata — streams whose OWN     │
  │   │            │          │       │   Lyapunov exponent is known                │
  │ 7 │ interleave │ mixed    │ mixed │ two streams merged by wall-clock time       │
  └───┴────────────┴──────────┴───────┴─────────────────────────────────────────────┘

  Reward-bearing (they produce rollouts and returns): task, control, vision, speech.
```

Five of the seven are **generated deterministically in-process** with no external data. Text is
not: `TextStream` reads a flat token array from the TinyStories shards (streams.py:177), and
`InterleaveStream` is built over that same text stream.

Every stream implements the same **five** operations (streams.py:115): `cursor`,
`next_windows`, `fetch(identity)`, `eval_windows` and `fitness_windows` — the last two being
what §10's reward depends on.

---

## 13. Training: one step and one boundary

```
  ┌─ BOUNDARY IN ──────────────────────────────────────────────────────────────┐
  │  B1  express the genome:  u ← Φ_θ(g),  γ ← c·tanh(u)                        │
  │  B2  build the switch pattern; COPY-INITIALISE any newly born block         │
  │  B3  rebuild the optimisers with the genome's learning-rate multipliers     │
  │  B4  draw the stream mixture μ from the genome; assign the B lanes          │
  │  B5  fresh window-summary buffer                                            │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ STEP n = 0 … N−1 ─────────────────────────────────────────────────────────┐
  │  S1  draw B windows  (first 5 % from the REHEARSAL buffer)                  │
  │  S2  encode → backbone → ŷ                                                  │
  │  S3  losses; divide by the token denominator                                │
  │  S4  backward; clip grad norm to 1.0; SKIP the step if non-finite           │
  │  S5  optimiser step; project the Gaussian heads' log-std                    │
  │  S6  append window summaries for the reverse read                           │
  │  S7  charge FLOPs: training rows + rollout forwards + interface             │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ BOUNDARY OUT ─────────────────────────────────────────────────────────────┐
  │  O1  per-stream return scales                                               │
  │  O2  G-STAT: has the episode's transient finished?                          │
  │  O3  score the FITNESS sets → L_check → the reward r_e                      │
  │  O4  re-fit the expression map θ to the last 16 (g, u) pairs                │
  │  O5  CRITICALITY PROBE  ◄ BEFORE the estimator, so no probe episode can     │
  │                           move what it measures                             │
  │  O6  reverse read → policy update → propose Δg                              │
  │  O7  genome step: trust region, decay, bound                                │
  │  O8  the ζ homeostatic rule                                                 │
  │  O9  update the insulation κ from each head's own loss                      │
  │  O10 build the rehearsal buffer; fit the salience head                      │
  └────────────────────────────────────────────────────────────────────────────┘
```

**G-STAT** (life.py:127) passes if

```
   ( d_last < ¼·max(d_first, 0)  AND  d_last > −2·max(σ_e, se_last) )
   OR  |d_last| < 2·max(σ_e, se_last)
```

— the rate limb is two-sided, so an episode whose loss is *rising* at the end fails it. `N` is
doubled at most once per life.

`boundary_out` raises if it is somehow called inside a step (life.py:1005).

---

## 14. Inference

Inference is the same loop with `η₁ = 0` (life.py:1365): perceive, optionally emit, and the
emission re-enters through `E_m`. Per token the backbone runs `1/K` of a token model's steps —
plus a decoder step that is **not** small (§15).

**Batched rollouts run in lockstep.** `perceive_many` (life.py:459) steps every row every span
with no mask, and now **raises** if the lanes' caches differ in length or mix fresh with cached
(life.py:497, 504). Earlier it silently ran with no cache and wrote a length-1 cache back over
the active rows; that failure is historical, and the raise is what prevents its return.

---

## 15. Compute accounting

```
   F_episode = Σ_steps [ backbone(pattern)   ← analytic, morphology.py:205
                       + interface           ← MEASURED, modalities.py:349
                       + rollout forwards ]
```

Measured forward FLOPs per token **at the primordium** (tempo_cont.py:62 — this is the shape
the K-CONT baseline is matched against, not the adult):

```
   ┌──────────────────────┬──────────────┬────────┐
   │ span embedding       │   6.554e+04  │   0.4% │
   │ backbone (primordium)│   9.229e+06  │  55.4% │
   │ span decoder         │   7.373e+06  │  44.2% │  ← at the ADULT this share is ~17 %
   │ total                │   1.667e+07  │ 100.0% │
   └──────────────────────┴──────────────┴────────┘
```

Charging the interface matters because it is the same per token for every morphology while the
backbone is not: adult/primordium is **3.93** on the backbone alone and **2.61** with the
interface charged to both — and that ratio is exactly what an equal-compute comparison reads.

---

## 16. How a verdict is reached

```
   contrast = paired difference over seeds sharing common random numbers
   bar      = 3σ
   fires_when = "not_better":              fires_when = "worse":
     mean ≤ −bar AND hi < 0 → DID NOT FIRE   mean ≥ bar AND lo > 0 → FIRED
     mean ≥ 0               → FIRED          hi < bar              → DID NOT FIRE
     otherwise              → INCONCLUSIVE   otherwise             → INCONCLUSIVE

   < 2 paired seeds, a missing arm, an incomplete run, or an unmeasured bar
                                          →  NOT EVALUATED
```

**Where σ comes from is per-criterion, not universal.** The default is the *control* arm's
spread on that criterion's own metric, but `criterion()` accepts overrides
(tempo_analyse.py:269) and three criteria use them: **K-CONT** takes σ from the R1 arm on the
test metric (tempo_analyse.py:375) — i.e. `σ_e`, the span agent's own seed noise, which is the
methodologically right bar for it; **G-ACT** and **K-SOTA** override too. The claim "every
criterion uses the control arm's spread" is false, and the docstring in the code repeats it.

`NOT EVALUATED` and `INCONCLUSIVE` are different statements: the first says the contrast was not
run, the second says it was run and was unclear.

---

## 17. What has actually been measured

R0–R1 complete; R2 in progress at the time of writing.

### R0 — the codec ceiling

```
   K = 4   0.00137 nats/token        K = 2   0.00071 nats/token
```

*Provenance caveat:* the codec runs have **no `record.json`**, and their `"complete": true` was
backfilled by hand (each `result.json` carries a `complete_backfilled` note saying so).

### R1 — seed noise, and an ablation

Four seeds, one episode of ~25M tokens, primordium shape, genome inert:

```
   arm      test NLL (nats/token)          σ
   ─────────────────────────────────────────────
   nolat    2.4067 2.3977 2.4032 2.4027    0.0037     ← λ_lat = 0   (default)
   lat01    2.6340 2.6267 2.6362 2.6185    0.0080     ← λ_lat = 0.1 (ablation)

   paired:  +0.2263 nats  (sd of the four differences 0.0074;
                           95 % t-interval [+0.2145, +0.2380])  =  60.6 σ_e
```

**σ_e = 0.0037337** is the scale every later verdict is read against. Throughput 90,238 tok/s,
5.72 GB peak, RTX 4080 SUPER.

### R2 — K-CONT

```
   span agent (K=4)   2.4026 nats/token
   token model        1.8560 nats/token     width 192, 8 layers, 7.56M params
                      ─────────────────
   difference         +0.5465  [+0.5390, +0.5541]   bar 3σ_e = 0.0112
                      = 146 σ_e                ──►  K-CONT FIRED
```

**A caveat on "equal compute", found by review.** The recorded `flop_ratio` is 0.9999999991,
but that is an *analytic* match: `tempo_cont.py` sizes the baseline against an assumed 25,000,000
span-agent tokens at 16.668e6 FLOP/token. The span agent actually consumed **24,985,600** tokens
and charged itself **1.237629e15** FLOPs; the token model actually trained on **29,982,720**
tokens for **1.250490e15**. The **delivered** ratio is **1.0104** — a 1.0 % compute advantage to
the token model. This does not touch the verdict (1 % of compute cannot produce 0.55 nats), but
the two sides of K-CONT are still counted by two slightly different methods, which is precisely
what the interface-FLOPs work was meant to eliminate.

### The criticality profile

```
                        Λ (relative)          Λ_abs (absolute)
   arm         total    per block     total    per block     worst single block
   ────────────────────────────────────────────────────────────────────────────
   nolat      −0.550     −0.0344      +0.075     +0.0047      0.0458 … 0.0473
   lat01      −0.754     −0.0471      −0.158     −0.0099      0.0830 … 0.0900
```

**(a) The band is loose, but by less than the first draft claimed.** `trigger()` has **two**
limbs (criticality.py:209): the per-block stack average *and* the **worst single sub-block**.
The worst-block limb binds first: `0.5 / 0.047 = 10.6×` headroom at nolat and `0.5 / 0.090 =
5.6×` at lat01, not the ~15× that the stack limb alone suggests. The λ_lat intervention — worth
60.6 σ_e in test loss — moved the worst block by about 0.040, i.e. **8 % of the band**, so
roughly **11** such interventions would reach it, not the ~35 the stack limb implies. The band
is still very loose, and K-CRIT is still unlikely ever to be evaluated; it is not as unfireable
as first stated.

**(b) The two exponents disagree about the sign.** `Λ_abs` sits essentially at zero and changes
sign between arms (+0.0047 → −0.0099); `Λ_rel` is robustly negative. Mean-field theory for
residual stacks with normalisation predicts a trained network of this shape sits *at*
criticality — which is what `Λ_abs` shows, to within 0.01 per block, across two objectives 60 σ
apart. The order `Λ_rel` reports comes from dividing out the activations' own growth, a
normalisation this codebase introduced and no prior work reports.

*A claim the first draft made and the review refuted:* that `Λ_rel` is "much more tightly
clustered" than `Λ_abs`. In absolute units it is not — the per-seed sd is 0.0280 (rel) vs
0.0238 (abs) for nolat and 0.0125 vs 0.0077 for lat01. `Λ_abs` has the **smaller** spread in
both arms. The relative form's apparent stability is an artefact of its larger magnitude.

So the open question is not "does the network leave criticality" but **which exponent
criticality should be measured in** — and the band was placed on the one that disagrees with the
literature's prediction.

### Provenance

`code_digest` is identical across every R1 and R2 run, so the model code did not change between
them — which is the property comparability actually requires. The *commits* differ: the nolat
seeds ran at `3e7425c`, the lat01 seeds at `686e61f` / `a84e515` / `85f221c`, the R2 token runs
at `aeb98e7`, every one with a dirty working tree.

---

## 18. What is not yet decided

```
   MEASURED AND SETTLED
     · the codec ceiling is negligible                             R0
     · σ_e = 0.0037                                                R1
     · λ_lat = 0.1 costs 0.23 nats — correctly off by default      R1
     · K-CONT FIRES at K = 4: the token model wins by 146 σ_e      R2
     · the criticality band is 5.6–10.6× looser than the measured
       worst block, so K-CRIT is very unlikely to be evaluated     R1

   SCHEDULED IN THE RUNNING STAGE (no artifacts yet)
     · K-CONT at K = 2  (span-agent seeds running; token arm not yet)
     · K-CONT at K = 1  (not started; only a superseded -v1 codec exists)

   NOT YET RUN — every claim the design rests on
     · K-GENE  · K-REV  · K-MORPH  · K-SAL  · K-CRIT
     · K-TRANSFER · K-XFER · K-BLEND · K-SOTA · G-ACT
```

Everything in §§3–16 is built and unit-tested; nothing in the last list has been *shown to
work*. The one end-to-end comparison that has run went against the architecture's central
efficiency premise at its default setting.

If K-CONT also fires at K=2 and K=1, the span interface has argued itself out of the design: at
K=1 the codec is a token head and TEMPO reduces to a conventional token model carrying extra
machinery. That is a possible outcome of this protocol and it is written here before the runs
that decide it.

---

## 19. What the adversarial review found in the first draft

Three reviewers read only the source and the run artifacts. **45 findings, 8 critical.** The
ones that changed a described mechanism:

| # | the first draft said | the code says |
|---|---|---|
| 1 | the fast state is "a matrix per **attention** head" | it belongs to the 12 **delta** layers; attention layers keep a KV cache |
| 2 | `x̄` is "the target the backbone is asked to predict" | at `λ_lat = 0` (the default) nothing regresses `ŷ` to `x̄`; the only text loss is the decoder NLL |
| 3 | FlowHead is "the production action head" | FlowHead is never instantiated outside a unit test; the action head is `GaussianHead` |
| 4 | reading backwards "anchors the summary on how the episode **ended**" | the read-out is taken last, after processing w₁ — so it is anchored on how the episode **began** |
| 5 | the rehearsal buffer holds 3,200 windows | it holds `0.10·N·B` — 4,880 at N=1525, 800 at N=250 |
| 6 | estimator 2 probes for 100 steps, ~33 % overhead | the protocol sets `n_probe = max(5, N//10)` = 25, giving **50 %** |
| 7 | "σ is always the control arm's spread" | K-CONT, K-SOTA and G-ACT override it; K-CONT uses the R1 arm |
| 8 | both K-CONT arms had equal compute to 8 digits | analytically yes; **delivered** ratio is 1.0104 |

And the ones that changed a number: the token-model mean is 1.8560 (not 1.8561); the K-CONT gap
is 146 σ_e (not 148 — the first draft divided by a rounded σ); three cells of the criticality
table were misrounded; "±0.0074" was a standard deviation labelled as an interval; the FLOP
breakdown is measured at the **primordium**, not the adult; the pattern stream at 100 Hz is
faster than speech at 62.5 Hz; the `Stream` protocol has five methods, not three; text and
interleave are **not** generated in-process; `Λ` (relative) does not telescope, only `Λ_abs`
does; the policy-gradient term's clip is **inert** (`ρ ≡ 1`), making it REINFORCE; the text loss
also carries a z-loss; several births copy **exactly**, without noise; and eight `file:line`
citations pointed at the wrong lines.

One claim was withdrawn as unsupported: that feeding the codec latent back as input "was
measured to be worse". No such run exists here; the separation is a stated design invariant.

---

*Written from the source at commit `aeb98e7`, 2026-09-04; second draft after adversarial review.
Every number in §17 is reproducible from `runs/tempo/*/record.json` and `result.json`.*
