# 02 — The interface: one perception stream, one continuous output, the modalities

**Status: normative.** Specifies the perception stream (§2.1), the text interface and its
codec (§2.2), decoding (§2.3), the losses per modality and the gradient insulation (§2.4), the
continuous heads (§2.5), the τ1 policy-gradient term (§2.6), the pretrained-encoder forms
(§2.7), time codes and multi-rate interleaving (§2.8), the local modality blend with every
generator defined exactly (§2.9), and the acceptance tests (§2.10). Symbols: 01 §1.1;
`d_x = 256` (TS), `512` (Pile); `K = 4`.

## 2.1 The perception stream

**[NORMATIVE]** Every span `t` of the life carries one vector

```
   x_t = E_{m(t)}(raw_t) + μ_{m(t)} + τ(t)         ∈ R^{d_x}                                    (2.1)
```

with `E_m(raw)` RMS-normalised to unit RMS (`E_m(raw) ← E_m(raw) / RMS(E_m(raw))`), `μ_m ∈ R^{d_x}`
a learned modality embedding (τ1, one per modality, initialised `N(0, 0.02²)`), and `τ(t)` the
time code of §2.8. The backbone (03) consumes `x_t` and nothing else: no vocabulary, no
modality-specific table, no separate action input. The agent's own emissions re-enter the
stream through the encoder of their modality (01 (1.4)).

## 2.2 The text interface

**[NORMATIVE]** A text span is `K = 4` consecutive tokens `(v_1, …, v_K) ∈ V^K`,
`V = 8192` (the existing tokenizer `data/tokenizer_8192.json`; the reserved block of 64 ids is
never emitted for natural text, `soma/data/tokenizer.py::assert_reserved_block_unused`). Two
maps carry text, and they are different by design (the codec latent fed back as *input*
costs quality: CALM, 11 §P-A.6):

**Input side, trained at τ1 with the backbone:**

```
   E^{in}_txt(v_{1..K}) = W_x · [ T_emb[v_1] ; … ; T_emb[v_K] ]         T_emb: (V × d_e), d_e = 128,
                                                                        W_x: (d_x × K d_e)               (2.2)
```

**Output codec, trained first (R0); the encoder then frozen, the decoder trained on with the
backbone (11 §P-C.21):**

```
   encoder  E_txt:   u_i = T'_emb[v_i] + P_i                          (K × d_c),  d_c = 256, P learned
                     ũ   = Enc_2(u)                                     2 bidirectional pre-LN layers,
                                                                        4 heads of 64, FFN 1024, GELU
                     μ_z = W_z · mean_i ũ_i,   log s_z = W_s · mean_i ũ_i     (d_x each)
                     x̄   = μ_z / RMS(μ_z)                               the unit-RMS target latent          (2.3)
   decoder  D_txt:   c_i = T''_emb[v_{i−1}] + P'_i + W_c x̄               v_0 := BOS (id 1); causal in i
                     c̃   = Dec_2(c)                                     2 causal pre-LN layers, same sizes
                     p(v_i | x̄, v_{<i}) = softmax( softcap_15( W_v c̃_i ) )                                (2.4)
   codec loss        L_codec = − Σ_{i=1}^{K} log p(v_i | x̃, v_{<i})  +  β Σ_j max( KL_j, λ_floor )
                     x̃ = drop_{0.15}( x̄ + ε ),  ε ~ N(0, σ_z² I),  σ_z = 0.3,
                     KL_j = ½ ( μ_{z,j}² + s_{z,j}² − log s_{z,j}² − 1 ),  β = 10^{-3},  λ_floor = 0.5     (2.5)
```

`drop_{0.15}` zeroes 15 % of the latent coordinates (a coordinate-wise dropout applied only
during codec training); the KL is on the unnormalised `μ_z` with a per-dimension floor (the
CALM recipe: the floor applies to the batch-mean KL of each coordinate). The latent the
decoder is trained on is the reparameterised posterior sample `z = μ_z + s_z ⊙ ε_1`,
normalised, plus the noise and the dropout, so that `s_z` is used by the decoder and the KL
is a VAE regulariser rather than a term on an unused scale (11 §P-C.15); the target `x̄` the
backbone predicts is deterministic.

**What is frozen and what trains on.** After R0 the *encoder* `E_txt` is frozen: it defines
the target latent `x̄` and the latent regression term of (2.6). The *decoder* `D_txt` is
initialised from R0 and **trained at τ1 with the backbone** as the conditional language model
of the span given `ŷ`, `p(v_i | ŷ, v_{<i})`. The reason is measured (11 §P-C.21): the next
span is almost never predictable with certainty, so the best deterministic `ŷ` is the
conditional *mean* of the span codes, a low-norm blur the frozen reconstruction decoder
(trained only within `σ_z` of true codes) cannot turn into a distribution over spans; the
frozen-decoder agent stayed near unigram level (`5.12` nats per token at 25M tokens) while its
latent regression never fell below `‖x̄‖²`. With the decoder trained on, the likelihood
(2.6) stays proper, `ŷ` is the context summary the decoder conditions on, and the `K = 1`
case is a token softmax head. The regression term was meant to keep `ŷ` in the codec's
latent space, where every other head reads it; measured, it costs `0.223` nats at 25M tokens
(four seeds, `σ_e = 0.0014`), so `λ_lat = 0` is the default and the output space is defined by
the jointly trained decoder and heads (11 §P-C.23). At the end of R0 the codec is frozen; its per-token NLL on true latents,
`L_codec/K` without noise or dropout, is the **codec ceiling** and must be `< 0.01` nats or
`K` is reduced to 2. Parameters: `Enc_2` and `Dec_2` `≈ 0.79M` per layer (3.1M), `T'_emb,
T''_emb: 2 × V × 256`, `W_v: V × 256` (6.3M), `W_z, W_s, W_c` (0.2M): `≈ 9.6M`, frozen.

The span's *target* is `x̄_{t+1}` from (2.3); the span's *input* is (2.2).

**The `K = 1` head.** At `K = 1` a span is one token and there is nothing to predict *within*
the span, so the decoder has **no layers**: the logits are

```
   p(v | ŷ) = softmax( softcap_15( W_v · LN( T''_emb[BOS] + P'_1 + W_c ŷ ) ) )                       (2.4b)
```

a softmax head on a `d_c`-wide affine image of `ŷ`, with no within-span recurrence and no
self-attention. Keeping the two causal decoder layers at `K = 1` would give the span agent a
two-layer transformer the token baseline does not have, on a sequence of length one, and would
charge it FLOPs for a computation that cannot use them. This is the form the K-CONT fallback
`K = 1` runs, and it is the honest "the token model is the `K = 1` special case" claim
(11 §P-C.26).

**The default `K`.** `K = 4` is the registered default, the value CALM measured as the best of
`{1, 2, 4, 8}` at this scale (11 §P-A.6). K-CONT decides it on this card, and the fallbacks
run in order `4 → 2 → 1`: R2 fired at `K = 4` (`+0.511` nats against the FLOP-matched token
model) and at `K = 2` (`+0.278`), so the `K = 1` block is what the protocol runs next; the
final default is whatever survives, and 03 §3.1's `T_s = T/K` follows it.

## 2.3 Decoding text at inference

**[NORMATIVE]** Given `ŷ_{t+1}`, the next span is sampled within the span autoregressively
from (2.4) with `x̄ := ŷ_{t+1}` (the decoder is the conditional model of the span given `ŷ`,
trained with the backbone, §2.2): `v_i ~ p(· | ŷ, v_{<i})` with temperature `T` and nucleus `p`; the
emitted span is re-encoded by (2.2) as the next input (01 (1.4)). Greedy decoding is
`T → 0`. The evaluation NLL of a span is (2.6) with true tokens; sampling never enters it.

## 2.4 Losses per modality and the gradient insulation

**[NORMATIVE]** Per span, with `ŷ = ŷ_{t+1}` and the true next raw:

```
   text        L_txt   = − Σ_{i=1}^{K} log p_{D_txt}( v_i | ŷ, v_{<i} )  +  λ_lat ‖ ŷ − x̄ ‖²,   λ_lat = 0,
                         plus z-loss 10^{-4} on the decoder's logits                                       (2.6)
               (λ_lat = 0.1, the plan's value, is the recorded ablation: it cost 0.223 nats at 25M tokens
                and pushed the depth profile into the ordered phase; 11 §P-C.23)
   Gaussian    L_gau   = − log N( a ; W_a ŷ + b_a, diag exp(2 w_s) )                                      (2.7)
   Bernoulli   L_ber   = − Σ_j [ y_j log σ(W_b ŷ)_j + (1 − y_j) log (1 − σ(W_b ŷ)_j) ]                   (2.8)
   flow        L_fm    = E_{s ~ Beta(1.5, 1), ε ~ N(0, I)} ‖ v_φ( a_s, s, ŷ ) − (ε − a) ‖²,  a_s = (1−s) a + s ε   (2.9)
   reward      L_rl    = the policy-gradient term of §2.6 on reward-bearing streams
   frame       L_img   = ‖ Dec_img(ŷ) − frame ‖²                                                          (2.10)
```

`L_txt / K` is a per-token negative log-likelihood of the same tokens the token baseline
scores, so K-CONT (10) compares like with like. The episode loss is the sum over spans of the
terms present in the span's modality (each modality declares its terms in §2.9).

**Gradient insulation.** A randomly initialised continuous head sends gradients that corrupt
the backbone (Knowledge Insulation, 11 §P-A.9). Every non-text decoder `D_m` reaches the
backbone through a scale on `ŷ`:

```
   ŷ_m = κ_m · ŷ + (1 − κ_m) · stopgrad(ŷ),      κ_m ∈ [0, 1]                                         (2.11)
```

The head itself always trains at full gradient (the scale is on `ŷ`, not on the head's
parameters). `κ_m` is a τ2 quantity, driven by the head's **own** loss over the episode
(for the action head, minus the mean return, since its "loss" is a return):

```
   let ℓ_m^{(0)} be the head's mean loss over its first episode and ℓ_m^{(e)} over episode e;
   the head is released at the first episode e with  ℓ_m^{(0)} − ℓ_m^{(e)} ≥ ½ |ℓ_m^{(0)}|;
   κ_m = 0 before episode e+1,  = the fraction of episode e+1 elapsed during it,  = 1 after      (2.11b)
```

The rule is stated as an **improvement** from the head's own starting point, not as "below
half its initial value", because several of these losses are negative densities (a Gaussian
frame head's NLL is `≈ −60` nats) for which "half its value" is not a decrease at all, and
because a scalar return has no meaningful zero. The ramp fills a whole episode, so `κ_m` never
jumps from `0` to `1` between two consecutive steps — a jump the first form allowed, and which
delivers to the backbone, in one step, the full gradient of a head that was insulated from it
until then (11 §P-C.26). `κ_m` is recorded per episode (08 §8.6). Text and the task stream's
text have `κ = 1` always.

## 2.5 The continuous heads

**[NORMATIVE]** `D_gau` (2.7): `W_a: (d_a × d_x)`, `b_a: d_a`, `w_s: d_a` (log standard
deviations, initialised `0`, clamped to `[−5, 2]`); sampling `a = W_a ŷ + b_a + exp(w_s) ⊙ ξ`.
`D_flow` (2.9): `v_φ` a 3-layer MLP `(d_a + 1 + d_x) → 512 → 512 → d_a` with SiLU, `s` embedded
by 16 Fourier features (`sin/cos(π f s)`, `f = 1..8`, which are the periodic functions of
`s ∈ [0, 1]` — the plain `sin(2^k s)` features of the first form are not periodic on the unit
interval and alias); sampling by 10 Euler steps of `a ← a − (1/10) v_φ(a, s, ŷ)` from
`a_0 ~ N(0, I)` at `s = 1` down to `s = 0` (`v = da_s/ds = ε − a`, so the step toward the data
is against `v`; `s ~ Beta(1.5, 1)` by inverse CDF `s = u^{2/3}`); both take `ŷ_m` of (2.11).

**Which head carries the policy gradient.** The Gaussian head is the screening default *and*
the only head the policy-gradient term of §2.6 uses. A flow-matching head has no tractable
density: `L_fm` is a regression on the velocity field, not a log-likelihood, so `log p_θ(a)`
and the ratio `ρ` of (2.12) do not exist for it. The flow head is the production form for
imitation-style training (the π0 setting, 11 §P-A.8) and is declared, tested for its own
identity rows, and **not** used in any reinforcement-learning arm on this card. Any future use
of it under (2.12) requires a stated density estimator, not a proxy.

## 2.6 The τ1 policy-gradient term

**[NORMATIVE]** On a reward-bearing stream (symbolic tasks, control), each prompt or start
state is rolled out `G = 8` times from the same state with CRN by rollout seed, the rewards
`r_i` are observed (exact verification or the environment's return), and

```
   Â_i = ( r_i − mean_j r_j ) / ( std_j r_j + 10^{-6} ),
   L_rl = − (1 / Σ_i |o_i|) Σ_i Σ_{t ∈ o_i} min( ρ_{i,t} Â_i,  clip(ρ_{i,t}, 1 − 0.2, 1 + 0.28) Â_i ),
   ρ_{i,t} = p_θ(o_{i,t} | ·) / p_{θ_old}(o_{i,t} | ·)                                                   (2.12)
```

the group-normalised advantage of GRPO with DAPO's asymmetric clip and token-level (here:
span-level and within-span-token-level) averaging (11 §P-B.6); prompts whose rewards are all
equal are dropped (dynamic sampling); prompt and tool spans are masked, only the agent's
emitted spans are scored; one policy update per rollout batch (so `ρ = 1` at the first
pass and the clip is active only under reuse). For control, `o_i` is the action sequence and `p_θ` the Gaussian head's density (2.7); the
flow head is not used here (§2.5). No KL to a reference policy (there is none in a life);
entropy is monitored, not regularised, and recorded.

**The egocentric closure of an action.** What the agent emits re-enters through an encoder,
as 01 (1.4) requires. For text and speech the emission is perceived through that modality's
own encoder. For a *control action* there is no action encoder: the agent perceives the
environment's response — the next proprioception span and the reward span — and not the action
it took. This is a deliberate reading of (1.4) and is recorded as such: the action's effect
enters through the world, and adding a separate `E_act` would double-count it. The consequence
for the rollout rows is that the fast state after an action reflects the world's reply, which
is exactly what the next action must condition on.

**Supervised rows and rollout rows.** A reward-bearing draw produces *two kinds of row*
(08 §8.3): a teacher-forced **supervised row** whose input at the answer positions is the
**truth** and whose targets are scored by (2.6), and `G` **rollout rows** whose input is the
agent's own emission and which carry (2.12) and no supervised target. Scoring the truth as a
target while feeding the model its own (usually wrong) emission — the first form — is not the
likelihood of anything: it asks for the truth given a prefix the truth never had.

## 2.7 Pretrained-encoder forms (declared, not evaluated locally)

**[NORMATIVE]** `E_vid(frames) = W_vid · JEPA(frames)`, `W_vid: (d_x × 1024)`, at the video's
frame rate, JEPA frozen; `E_sp(frames) = W_sp · codec(frames)` for a neural speech codec's
frame latents. Both are the interface of §2.1 with a frozen `E_m`; their tests exercise the
projection and the stream with random stand-in features; 10 marks their criteria as not
evaluable until data exist.

## 2.8 Time codes and multi-rate interleaving

**[NORMATIVE]** Every raw event carries a wall-clock time `t_w` (seconds; simulated for
generated streams from their rates, real at deployment). Spans are ordered by `t_w`; the
backbone's RoPE position is the span's index in the ordered stream. The time code is

```
   τ(t) = W_τ · [ sin(2π t_w / 2^k), cos(2π t_w / 2^k) ]_{k = −6..7}        W_τ: (d_x × 28)          (2.13)
```

so that periods from 1/64 s to 128 s are represented; `W_τ` is learned at τ1. Multi-rate
interleaving (§2.9.7) merges streams by `t_w`; the fast state carries the fast stream across
the slow one, which I13 tests.

## 2.9 The local modality blend

**[NORMATIVE]** The first build generates every stream below deterministically from seeds
(`tempo/synthetic.py`, `tempo/envs.py`) with no download. Each has its own raw dimensionality,
rate, encoder, decoder and terms of §2.4. The mixture over the seven, `μ ∈ Δ^6` with floor
`0.02`, is a genome role (04 §4.3) expressed once per episode; episodes draw windows by
systematic rounding (`anlage/data.py::systematic_counts`).

### 2.9.1 Text

TinyStories at `V = 8192` through the classified pool (`anlage/data.py::ClassifiedPool`, six
length × rarity classes), evaluation sets v3, strata as in ANLAGE 07 §7.4. Terms: (2.6).
Rate: 4 spans per second of simulated time (one token every 62.5 ms, a reading pace).

### 2.9.2 Symbolic tasks (the verifier environments)

`soma/openended/tasks.py`: families PARITY, MODCOUNT, PERM, DYCK, COPY, MQAR, GRID with the
nine grid transforms, over the reserved alphabet (`SYM_BASE = 8`, 56 symbols; PAD 0, BOS 1,
EOS 2, SEP 3, ROW 4, EX 5). An instance is `[BOS] + prompt + [SEP] + answer + [EOS]`; the span
interface applies to the token sequence; the loss mask scores answer tokens only
(`Instance.loss_mask`). Terms: (2.6) on answer spans (supervised) and (2.12) with
`r = verify(prompt, candidate, params) ∈ {0, 1}` on sampled answers. Curriculum:
`TaskArchive` (learnability gate `learn(s) = 4s(1−s) > 0.30`, priority sampling, 512 members)
with `solve_rate` from the agent's own greedy decoding, queried by the life at every
boundary (08 §8.2 O10) — the archive is seeded at the first boundary and takes one step per
boundary after, so the curriculum is a τ2 state like every other slow variable. Every draw
yields one teacher-forced supervised row and `G` rollout rows (§2.6). A window is re-fetchable
because the instance's parameters travel in its identity, so a rehearsed task window is the
same instance whatever the curriculum did since. A configuration whose context is too short for
a family's lower bound cannot seed the archive; the life then draws from the family lower
bounds and records the reason rather than aborting. Rate: as text.

### 2.9.3 Continuous control (the point-mass)

State `s ∈ R^8 = (p_x, p_y, v_x, v_y, q_x, q_y, o_x, o_y)`: position, velocity, goal, one
disc obstacle of radius `0.15`, all in `[−1, 1]²`. Dynamics at `dt = 0.05` (20 Hz):

```
   v ← v + (a − c v) dt,  c = 0.5,  ‖a‖_∞ ≤ 1;   p ← clip(p + v dt, −1, 1);
   r = − ‖p − q‖ − 0.01 ‖a‖² − 5 · [ ‖p − o‖ < 0.15 ]                                          (2.14)
```

An episode is 100 ticks; goal and obstacle are resampled per episode; the *manoeuvre*
variant moves the goal on a circle of radius 0.5 at 0.1 Hz. Perception: `E_prop(s) = W_prop s + b_prop` (`d_x × 8`) and, as a separate modality,
`E_rew(r) = w_rew r + b_rew` (`d_x × 1`), so the agent perceives its reward. Both are
**affine**, and the bias is initialised `N(0, 0.5²)`: (2.1) divides the encoder's output by its
own RMS, so a bias-free linear map would send `raw` and `2 raw` to the *same* perception
vector — a scalar reward would be reduced to its sign and the proprioception to its direction
(11 §P-C.26). With a generic bias, `raw ↦ (W raw + b)/RMS(·)` is injective on a bounded
domain. Output: `D_act` (2.7 screening, 2.9 production) with `d_a = 2`. Terms: (2.7) or
(2.9) as behaviour-cloning of the agent's own accepted actions is not used; the term is
(2.12) with the episode return as `r_i` over `G = 8` rollouts from the same start state. A
dedicated 2-layer MLP policy trained with the same estimator and compute is the G-ACT control.

### 2.9.4 Synthetic vision

Scenes of 1–4 shapes (circle, square, triangle) in six colours on a `4 × 4` cell grid,
rendered at `32 × 32 × 3` (values in `[0, 1]`); every 4 frames the scene undergoes one of the
nine `GRID_TRANSFORMS` (identity, rotations, flips, translate, recolour, tile). Frame rate
5 Hz. A query is a reserved-alphabet token string among `COUNT <colour> <shape>`,
`WHERE <shape>` (answer: cell index), `AFTER <transform> <cell>` (answer: the shape at that cell
after the transform), `SAME <cell> <cell>` (answer: yes/no), generated with its exact answer.
Encoder `E_img`: four `3 × 3` convolutions `3 → 32 → 64 → 128 → 256`, stride 2, GELU, global
mean pool, `W_img: (d_x × 256)`; trained at τ1. Decoder `Dec_img`: the mirror (four
transposed convolutions from a `256 × 2 × 2` seed produced by `W'_img ŷ`), loss (2.10).
Answers go through the text codec (they are reserved-alphabet spans). Terms: (2.10) on
frames and (2.6) on answers on the teacher-forced supervised row; (2.12) with
`r = [answer exact]`, averaged over the window's queries, on the `G` rollout rows (§2.6).

### 2.9.5 Tone-coded speech

A symbol `s ∈ {0..55}` is a tone of frequency `f_s = 62.5 (s + 4)` Hz (250 Hz to 3687.5 Hz,
on the frame's DFT grid and below Nyquist) lasting 4 frames; sample rate 8 kHz; a frame is
128 samples (16 ms, 62.5 Hz); waveform `w(n) = h(n) sin(2π f_s n / 8000)` with a Hann envelope
`h` over the symbol's 512 samples; frame feature `φ ∈ R^{64} = log(1 + |DFT_{128}(frame)|_{0..63})`
(the first 64 magnitude bins). (The plan's `220 · 2^{s/12}` Hz at 4 kHz put the upper symbols
above Nyquist and the lower ones inside one bin: 11 §P-C.16.) A
"speech" event is a sequence of symbol frames. Encoder `E_sp`: a **per-frame** MLP `64 → 128 → 256 → d_x` (GELU), applied to each frame
independently. Not a convolution over consecutive frames: the agent perceives one frame at a
time when it acts and a whole window at once when it trains, and a temporal context *inside*
the encoder makes those two paths compute different features for the same frame — the acting
path would see the frame with zero left context and the training path with three frames of it
(11 §P-C.26). The temporal context belongs to the backbone, which has it. Two tasks:
**transcription** (hear the frames, emit the symbol sequence through the codec; exact reward)
and **production** (see the symbol sequence as text, emit frames through a Gaussian head with
`d_a = 64`; decoded by nearest tone in the bank, exact reward). Terms on the supervised row: (2.7) on the *true* frames (production) and (2.6) on the
*true* transcript (transcription); on the `G` rollout rows, (2.12) with the exact rewards.

### 2.9.6 Pattern streams at the edge of chaos

Three generators with known Lyapunov exponents, rate 100 Hz, linear encoders and the heads
named:

* **logistic map** `x_{n+1} = r x_n (1 − x_n)`, `r ~ U[3.5, 4.0]` per trajectory; observation
  `(x_n, …, x_{n−7}) ∈ R^8` (eight lags); `λ_ref = lim (1/n) Σ_k log |r (1 − 2 x_k)|`; Gaussian
  head.
* **Lorenz** `ẋ = σ(y − x), ẏ = x(ρ − z) − y, ż = xy − βz`, `σ = 10, ρ = 28, β = 8/3`,
  integrated by RK4 at `dt = 0.01`, observed every step as `(x, y, z)/20 ∈ R^3`;
  `λ_ref ≈ 0.9056` (the generator recomputes it by the Benettin method over its run);
  Gaussian head.
* **elementary cellular automata** of width 64, rules `{250, 4, 90, 30, 110, 184}` (Wolfram
  classes 1, 2, 3, 4 represented), random initial rows; observation the row `∈ {0,1}^{64}`;
  `λ_ref` the damage-spreading rate of a single-cell perturbation,
  `mean_trials mean_{t ≤ w/2} (1/t) log max(H_t, 1/w)` with `H_t` the Hamming distance
  (Bagnoli, Rechtman & Ruffo 1992; the time average because additive rules such as 90 have
  `H_t = 2^{popcount(t)}`, degenerate at any single horizon); Bernoulli head.

**`λ_ref` is a property of the generator, not of a window.** It is computed once per
parameter setting over `10^5` steps and cached, never over the `T_s = 128` observations of a
window: a 128-step estimate of a Lyapunov exponent is noise of the same order as the exponent,
and writing that noise beside the network's `Λ` would invite a correlation that is an artefact
of the estimator (11 §P-C.25).

**A lane's trajectory continues.** A pattern lane holds one generator instance whose
trajectory runs across the lane's windows — the last observation of a window is the first
input of the next — so the fast state has something to carry, which is the point of putting
these streams in the blend at 100 Hz. A window is re-fetchable from `(kind, seed, index)`.

These streams give B6 its "pattern training" and give 07 an external reference: the network's
own depth-wise exponent `λ_l` is read beside the stream's `λ_ref`.

### 2.9.7 Multi-rate interleaving

Any two streams merged by `t_w`. The registered test pair is tone speech (62.5 Hz) with
text (4 spans/s): 15–16 speech frames between consecutive text spans. The fast state must
carry the speech stream across the text spans (I13).

## 2.10 Acceptance tests

**[NORMATIVE]** `tempo/codec.py`, `tempo/modalities.py`, `tempo/synthetic.py`, `tempo/envs.py`
are accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| I1 | codec ceiling | after R0 the per-token NLL of `D_txt` on true latents is `< 0.01` nats on the v3 test set (artifact check); on a synthetic 64-token vocabulary a 200-step codec reaches `< 0.05` (unit) |
| I2 | noise tolerance | the decoder's NLL at `x̄ + ε`, `‖ε‖ = σ_z sqrt(d_x)`, is within `0.05` nats of its NLL at `x̄` after training under (2.5) |
| I3 | the input is not the codec | `E^{in}_txt` has no parameter in common with `E_txt`; perturbing the frozen codec leaves (2.2) unchanged |
| I4 | likelihood is proper | for every `x̄`, `Σ_{v ∈ V^K} p(v | x̄)` computed exactly on `V = 8` and `K = 2` equals 1 to `1e-6` |
| I5 | the `K = 1` head | at `K = 1` the decoder has no layers and (2.4b) holds: the logits are an affine function of `ŷ` through `W_c` and `W_v` with no self-attention (asserted by the module tree and by the gradient's support), and `L_txt / K` is the mean NLL of the same tokens the token model scores |
| I6 | Gaussian head | (2.7) matches `scipy.stats.multivariate_normal.logpdf` on random inputs to `1e-5`; the log-std bound is kept by **projection** after the optimiser step, not by a clamp in the graph (a clamped parameter outside the range has zero gradient and can never return, 11 §P-C.26) |
| I7 | flow head | on a 2-D bimodal target, 10-step Euler samples from a trained (2.9) head recover both modes (`≥ 90 %` of samples within 0.6 of a mode, balance in `[0.4, 0.6]`, within-mode spread within a factor 2 of the target's; the two-sample KS p-value is reported beside them, not asserted: a 10-step Euler integration carries a small bias that a KS on 2048 points detects), and the Gaussian head does not (`< 50 %` near a mode); the time features are periodic on `s ∈ [0, 1]` (`sin/cos(π f s)`) and the head is **not** used under (2.12) |
| I8 | policy gradient | (2.12) on a bandit with `G = 8` increases the probability of the rewarded arm over 50 updates; dropped groups (all-equal rewards) contribute zero gradient |
| I9 | insulation | with `κ_m = 0` the backbone's gradient from `D_m` is exactly zero and the head's is not; with `κ_m = 1` the backbone's gradient equals the un-insulated one; the rule (2.11b) releases a head on an **improvement** of half its first value (so it is well posed for negative losses), the ramp fills exactly one episode, and the state round-trips through the checkpoint |
| I10 | generators are deterministic and exact | every generator of §2.9 reproduces bit-identical streams from one seed; the point-mass obeys (2.14) against a hand computation and its manoeuvre obstacle is kept off the goal's orbit; scene queries are non-degenerate (an informative query is drawn with probability ½ and `SAME` names two distinct cells) and their answers equal a brute-force solver's; tone frames decode back to their symbols with zero error; `λ_ref` is cached over `10^5` steps and matches the literature values (`ln 2` at `r = 4`; `0.9056 ± 0.01`) |
| I11 | verifier rewards | `verify` of the reserved-alphabet tasks is reused unchanged and its tests pass (`tests/unit/test_verifiers.py`) |
| I12 | time codes | (2.13) distinguishes `t_w` and `t_w + 1/62.5` and is periodic in 128 s to `1e-6` |
| I13 | multi-rate carry | a planted speech pattern before a text span is decodable from the fast state after the span in the interleaved stream and not when `S` is reset between them |
| I14 | unit RMS, magnitude and shapes | every `E_m` output has RMS `1 ± 1e-5` and shape `(d_x,)` after (2.1); every `D_m` accepts `(d_x,)` and returns its declared shape; shape assertions fire on any mismatch; the **affine** linear encoders keep the magnitude of their input (two raws differing only in scale give different perception vectors) and the speech encoder gives a frame the same features whether it is presented alone or inside a window |
