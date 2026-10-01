# 11 — The evidence ledger

**Status: record.** What was verified before the design was fixed (2026-09-04): the machine
and the repository (§P-0), the timescale, continuous-output and world-model literature
(§P-A, verification agent A), the RL-adapter, learned-rule, criticality and growth
literature (§P-B, verification agent B), and the corrections the evidence forced on the
design (§P-C). Every number below was checked against the primary source by the agent that
reported it; where a claim could not be verified it is marked so. Cross-references elsewhere
in `docs/tempo/` use the labels `P-A.n`, `P-B.n`.

## P-0 The machine and the repository (inventory agent)

**Installed and usable:** torch 2.7.1+cu118, torchvision 0.22.1, torchaudio 2.7.1,
transformers 4.57.6, datasets 4.0.0, tokenizers 0.22.2, sentencepiece, einops, triton-windows
3.3.1, numpy 2.3.1, scipy 1.17.0, scikit-learn 1.8.0, pillow, opencv-python 4.13, pytest 9.
**Not installed:** gymnasium, gym, mujoco, dm_control, pybullet, timm, flash_attn, xformers,
imageio, av, decord, soundfile, librosa, wandb, tensorboard. Hardware: RTX 4080 SUPER
(15.99 GiB, SM 8.9), 31 GiB RAM, 16 logical CPUs, 235 GB free disk.

**Data:** TinyStories shards at `data/ts_8192` (train 452.9M tokens, 2.10M documents;
validation 4.3M tokens, 20k documents) and `data/ts_4096`; tokenizers with a reserved block
of 64 ids (`soma/openended/tasks.py::RESERVED_SIZE`); the plain-text corpus at
`data/raw/tinystories_train.txt` (1.9 GB). **No images, video, audio or robotics data exist
anywhere on the machine.** `data/pile_16384` does not exist; the Pile subset must be built
(`scripts/prepare_data.py --dataset local`).

**Environments with exact rewards, already in the repository:** `soma/openended/tasks.py`
(families PARITY, MODCOUNT, PERM, DYCK, COPY, MQAR, GRID with 9 grid transforms;
`generate(params, rng) → Instance`, `verify(prompt, candidate, params) → {0,1}`,
`solve(prompt, params)`), `soma/eval/regression.py::solve_rate(model, params, n_eval, rng, …)`
(greedy decoding), and `soma/openended/archive.py::TaskArchive` (a learnability-gated
auto-curriculum, `learn(s) = 4s(1−s) > 0.30`, 512 members, priority sampling). A continuous
control environment does not exist and no library for one is installed; 02 §2.9.3 adds a
dependency-free 2-D point-mass environment in pure torch (`tempo/envs.py`); it is a toy and is
labelled one.

**Reusable code confirmed (signatures verified):** `gnomon/pheno/mixer_chunk.py::
gated_delta_rule_chunk(q, k, v, alpha, beta, *, chunk_size, initial_state, valid)` returns
`(O, S)` and `anlage/ops.py::delta_mixer(..., state=None)` returns `(y, S)`: the fast state
already streams across calls; `attention_gated`, `swiglu`, `causal_conv4`, `rms_norm_gf`,
`softcap_logits`, `chunked_cross_entropy`; `anlage/model.py::Phenotype` with growth gates as
plain attributes (`res_gate, inner_gate, head_gate, layer_gate`); `anlage/recipe.py`
(`TrainConfig`, `lr_at`, `batch_at`, `build_optimizers`, `micro_batch_for`);
`anlage/lifetime.py::train(..., hooks={"before_step", "after_eval"})`; `anlage/data.py`
(`Corpus`, `EvalSet.sample_documents` v3, `ClassifiedPool.at(mu)`, `systematic_counts`);
`anlage/grow.py` (`grow(mode="copy")`, `Ramp`, `unit_prior`, `shrink`);
`gnomon/optim/muon.py::Muon` (2-D only); `screen/crn.py::WindowPool`; `screen/variance.py`
(paired contrasts, power); `soma/germline/es.py::es_step` (mirrored ES, `evaluate` injected);
`gnomon/utils/{stats,shapes,rng,report}.py`. Test suites: `tests_anlage` (91 pass),
`tests_gnomon`, `tests_screen`, `tests` (SOMA, incl. `test_verifiers.py`).

**Measured throughput on this card (median CUDA-event timing, eager, `T = 512`, batch 32):**
13M 69,020 tok/s at 8.35 GB; 36M 33,271 tok/s at 9.0 GB (micro-batch 16); 78M 15,118 tok/s at
8.5 GB (micro-batch 8). `torch.compile` falls back to eager on this Windows toolchain in
every recorded run. Best model on disk: 1.5693 nats (78M,
`runs/anlage/gen1-tinystories-single-s0/adult.pt`).

**What does not exist and must be written:** any RL loop, any continuous head, any episodic
rollout, any flow-matching or Gaussian decoder, any checkpoint/resume of a lifetime. The
working tree carries one pre-existing uncommitted modification, `screen/landscape.py`, which
is not touched.

**Decisions taken with the user (2026-09-04):** the control environment is the pure-torch
point-mass (no installs); the first build's modality scope is the *generated* blend of 02
§2.9 taken as far as local generation allows, with the genome as the one scale-free object
across all of them; pretrained image/speech encoders are interfaces with tests and no data
until data exist (10 states them as not evaluable, never as passes).

## P-A Timescales, continuous output, world models (verification agent A)

**P-A.1 Nested Learning / HOPE** (Behrouz, Razaviyayn, Zhong, Mirrokni; NeurIPS 2025;
arXiv 2512.24695). The closest published framing of B9: a model is a set of components
partially ordered by *update frequency* (Def. 2), each with its own context flow; the
Continuum Memory System (§7.1, Eqs. 70–71) is a chain of MLP blocks whose level `ℓ` updates
every `C^(ℓ)` tokens by the optimiser's error term on the task loss; HOPE adds self-referential
Titans memory (Eqs. 83–89). Table 2 (from scratch, FineWeb-Edu): 760M/30B Hope Wiki ppl 18.68
vs Titans 20.08, Transformer++ 24.18; 1.3B/100B Hope 14.39 / LMB 10.08 / avg acc 58.04 vs
Titans 15.60 / 11.41 / 56.82, Transformer++ 17.92 / 17.73 / 53.38. Failure modes stated by the
authors: catastrophic forgetting "is not solved"; attention-free Hope is far below
Transformers on multi-key retrieval (29.4 vs 79.4 at 4K); §1.1 explicitly **defers offline
("sleep") consolidation**, the level TEMPO's τ2 occupies. *Relied upon for:* the legitimacy
and vocabulary of nested update frequencies (01 §1.2). *Difference:* TEMPO's slowest level is a
compact gain genome moved by an RL-trained reverse pass, not a weight block moved by gradient.

**P-A.2 Titans** (arXiv 2501.00663): surprise-gated deep memory with momentum and forgetting
(Eqs. 12–14); 760M/30B MAC 19.93 Wiki ppl vs GDN-H2 19.88; weak on multi-key NIAH later.
**P-A.3 TTT** (arXiv 2407.04620), **TTT-E2E** (arXiv 2512.23675: only the last quarter's MLPs
updated at test time, `W_0` meta-learned; 3B tracks full attention to 128K, 2.7x faster there,
but training is 3.4x slower at 8K and full attention "dramatically outperforms" on NIAH),
**LaCT** (arXiv 2505.23884), **ATLAS** (arXiv 2505.23735: 1.3B 14.97 Wiki ppl vs Titans
15.60), **MIRAS** (arXiv 2504.13173: Yaad 1.3B 15.18 vs GDN 16.42). Caveat from the field:
"TTT with KV binding is secretly linear attention" (arXiv 2602.21204, 2026). *Relied upon
for:* the fast state as a least-squares learner (03 (3.6)) is the established τ0.

**P-A.4 Gated DeltaNet** (Yang, Kautz, Hatamizadeh, ICLR 2025, arXiv 2412.06464): Eq. 10 is
03 (3.6) exactly; 1.3B/100B GDN 16.42 / 12.17 / 55.32 vs Mamba2 16.56, GDN-H1 16.07.
Production hybrids at **3:1**: Qwen3-Next-80B-A3B (12 × (3 GDN → 1 gated attention)) and the
Feb-2026 **Qwen3.5-4B/9B** (the same layout); Kimi Linear (arXiv 2510.26692, KDA, 3:1; 7:1
"significantly worse"); GDN-2 (arXiv 2605.22791). GLM-5 (arXiv 2602.15763) chose sparse
attention over a GDN hybrid for fine-grained retrieval (−11.28 on RULER-128K in its
ablation). *Relied upon for:* the backbone (03) is the layout of the strongest small open
models; the retrieval gap of linear layers is why one layer in four stays attention.

**P-A.5 The third, slowest level in 2025–26 work.** SEAL (MIT, arXiv 2506.10943): the model
emits *self-edits* applied by LoRA SFT, an outer loop of ReST-EM with a binary reward on
held-out improvement; SQuAD no-context 32.7 → 47.0; catastrophic forgetting under sequential
edits (Fig. 6); 30–45 s per reward evaluation. "Do Language Models Need Sleep? Offline
Recurrence for Improved Online Inference" (CMU/UMD, arXiv 2605.26099): when the KV cache
fills, N offline recurrent passes over the buffered context update the SSM fast weights,
trained end to end; 16-hop retrieval only solved with 4 loops; GSM-Infinite 6-op 0.742 →
0.812. "Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories" (Google,
arXiv 2606.03979): consolidation by upward distillation then RL "dreaming" (numbers not
verified). TTCD (arXiv 2608.01672), SDFT (arXiv 2601.19897); survey arXiv 2607.25380.
Sleep-time compute (arXiv 2504.13171) rewrites the context, not the weights. *Relied upon
for:* an offline pass over the episode that writes a slow state is an active published line
with positive results; **none writes a compact gain genome, none reads the episode in
reverse, and only SEAL trains the rule by RL.** K-REV and K-GENE are where TEMPO's 06 must
earn its composition of them.

**P-A.6 CALM** (arXiv 2510.27688): `K = 4` tokens → 128-d latent (hidden 512), `> 99.9 %`
reconstruction under Gaussian noise `σ ≈ 0.3`, VAE regulariser `β = 10^{-3}` with a
per-dimension KL floor, 15 % latent dropout, 15 % token masking; a likelihood-free
energy-score head; **inputs remain discrete tokens** (feeding the latent back as input drops
BrierLM 4.70 → 3.25); Table 1: CALM-M 371M reaches BrierLM 5.72 at `3.7·10^{20}` train FLOPs
against Transformer-S 281M at 6.05 for `6.6·10^{20}`; `K = 8` "larger performance drop",
`K = 1` worse than discrete; the autoencoder does not improve with scale; no likelihood.
*Corrections applied:* 02 §2.2 (input side a learned span embedding; codec output-only with
`σ_z = 0.3`, a weak KL with floor, latent dropout; a proper likelihood through the within-span
decoder; K-CONT at matched FLOPs). **P-A.7** Large Concept Models (arXiv 2412.08821) and
Coconut (arXiv 2412.06769, COLM 2025) are context, not mechanisms used.

**P-A.8 Continuous action heads.** π0 (arXiv 2410.24164, RSS 2025): PaliGemma-3B + a 300M
flow-matching action expert, loss `E‖v_θ(A^τ, o) − u(A^τ | A)‖²` with beta-sampled `τ`, 10
Euler steps, chunks of 50 at up to 50 Hz; π0.5 (arXiv 2504.16054): pre-training with
everything as discrete tokens (FAST), post-training adds the flow expert with `CE + α·flow`,
`α = 10`. GR00T N1 (arXiv 2503.14734): VLM at 10 Hz, diffusion transformer at 120 Hz. Helix
(Figure, blog only): System 2 at 7–9 Hz, System 1 at 200 Hz, one continuous latent between
them. Gemini Robotics 1.5 (arXiv 2510.03342): mechanism undisclosed. *Relied upon for:* the
flow-matching form of `D_flow` (02 (2.9)) and the multi-rate structure of 02 §2.8.

**P-A.9 One output space for tokens and actions:** no verified 2025–26 paper trains a single
*continuous* output shared by language and control. VLA-0 (arXiv 2510.13054) and FAST (arXiv
2501.09747) unify them as *discrete* tokens (FAST: 700 → 53 tokens per 1 s chunk; naive
binning fails as control rate rises); OpenVLA-OFT (arXiv 2502.19645) Table I: discrete AR
76.5 → continuous L1 95.3 → diffusion 95.4; **Knowledge Insulation** (arXiv 2505.23705):
gradients from a randomly initialised continuous expert corrupt the VLM (fix: stop-gradient
into the backbone during co-training; LIBERO-90 96.0 vs 60.2). *Corrections applied:* the
gradient insulation of 02 §2.4; "one continuous output space" is a hypothesis with G-ACT as
its first test and the discrete-token route as the fallback.

**P-A.10–11 World models.** V-JEPA 2 (arXiv 2506.09985): ViT-g latent prediction on >1M
hours; V-JEPA 2-AC = frozen encoder + 300M block-causal predictor on 62 h of unlabelled robot
video, planning by CEM in latent space (16 s per action on an RTX 4090). Dreamer 4 (arXiv
2509.24527): causal tokenizer + 1.6B dynamics model with shortcut forcing; first offline
Minecraft diamonds; 256–1024 TPU-v5p. *Relied upon for:* the frozen-encoder-plus-projection
form of `E_vid` (02 §2.7) when video data exist; nothing else is claimed locally.

**P-A.12 Looped depth as another per-token time axis.** Huginn (arXiv 2502.05171), Ouro
(arXiv 2510.25741: loop `T = 4` with learned exits; 2.6B rivals Qwen3-8B on several suites;
**capacity stays ≈ 2 bits/param looped or not, gains are "manipulation, not storage";
degrades beyond T = 4; RL gave no gain over SFT**), Mixture-of-Recursions (arXiv 2507.10524).
*Relied upon for:* the decision **not** to make the third axis a depth loop: it buys reasoning
steps, not memory, and TEMPO's third axis is a memory process (06).

**P-A.13 The benchmark landscape B11 names (third-party trackers, 1–3 Sept 2026; vendor
cards for small models).** Frontier: MMLU-Pro ≈ 90, GPQA-Diamond ≈ 94–96, HLE ≈ 60–65,
SWE-bench Verified ≈ 95 (with a documented contamination caveat), τ²-bench ≈ 99, BrowseComp
≈ 91, OSWorld-Verified ≈ 86, ARC-AGI-3 single digits. The strongest sub-10B open model is
**Qwen3.5-9B** (Feb 2026), a 3:1 GDN/gated-attention hybrid: MMLU-Pro 82.5, GPQA-D 81.7,
LiveCodeBench v6 65.6, τ² 79.1, OSWorld-V 41.8; Qwen3.5-4B 79.1 / 76.2; Gemma 4 E4B 69.4 /
58.6. No ≤ 9B model appears on SWE-bench Verified, BrowseComp or HLE leaderboards.
*Consequence for B11:* the target is a *path*: TEMPO's backbone family is the one the best
small open models use; matched-compute parity locally (K-SOTA) is the claim this card can
support; the benchmarks above discriminate at ≥ 4B and are evaluated only when a TEMPO
lineage of that size exists.

## P-B RL-trained adapters, learned update rules, criticality, growth (verification agent B)

**P-B.1 Transformer²** (Sun, Cetin, Tang, Sakana; ICLR 2025; arXiv 2501.06252). Singular
Value Fine-tuning: `W′ = U (Σ ⊗ diag(z)) Vᵀ`, only `z ∈ R^r` trained; Llama3-8B 0.58M
parameters for MLP + attention experts vs 6.82M for a LoRA on attention (Table 4); trained by
REINFORCE with `r ∈ {−1, +1}` and a KL penalty (Eq. 1; lr 2e-3, `z` init mean 0.1). Table 1
(base / LoRA / SVF): Llama3-8B GSM8K 75.89 / 77.18 / 79.15; Mistral-7B 42.83 / 44.66 / 49.74;
Table 4: **SVF by next-token loss 60.50 vs by RL 79.23**; LoRA under the same policy gradient
collapses to 57.92; two-pass inference costs 13–47 % (Table 3); cross-model transfer of the
vectors works (Table 5). Negatives: gains at 70B within noise on two of three tasks; sparse
rewards when the base is weak. *Used for:* the genome as gains (04) and the RL rule over
gains (06); the SVF ablation (04 §4.8).

**P-B.2 Hypernetwork-generated adapters.** Text-to-LoRA (Sakana, ICML 2025, arXiv
2506.06105): zero-shot 73.9 vs task-specific oracle 75.8 vs multi-task LoRA 71.9 (Table 2);
description sensitivity (random strings 63.5, Table 5). HyperTuning (ICML 2023), HINT (ACL
2023), HyperLoRA (EMNLP 2024 Findings), Zhyper (arXiv 2510.19733: `A·diag(z)·B`, 65.9 % at
26x fewer params than T2L), Drag-and-Drop (arXiv 2506.16406: worse than base on MATH),
Meta-Tool (ACL 2026 Findings, arXiv 2604.20148: **no improvement over few-shot**), Model of
Models (arXiv 2608.21386: one-pass adapters recover 11–14 % of the in-context gain), the
Override Gap (arXiv 2604.23750). *Pattern:* generated adapters reach the multi-task-LoRA level
and stay below task-specific ones; **low-dimensional conditioning matches full generation at
10–26x fewer parameters**, the reason 04 (4.1a) is low-rank with `r = 32` and generates
gains, not matrices.

**P-B.3 Learned optimizers and meta-learned rules.** VeLO (arXiv 2211.09760; 4,000
TPU-months): degrades beyond ~500M parameters, an 8B transformer underperforms untuned
Adafactor (Fig. 9), fails past the prompted step count; Rezk et al. (arXiv 2310.18191): "does
not necessarily outperform competitors"; Celo2 (ICLR 2026, arXiv 2602.19142) shows a 1.3B
curve without loss values and no Muon comparison; ELO (arXiv 2607.06772) ≤ 350M. **No verified
meta-learned optimizer beats tuned AdamW or Muon at ≥ 1B.** Short-horizon bias: Wu et al.
(ICLR 2018, arXiv 1803.02021): meta-optimisation "chooses too small a learning rate by
multiple orders of magnitude, even … 100 steps"; Metz et al. (arXiv 1810.10180): chaotic
meta-gradients under truncation. *Used for:* the three defences of 06 §6.7 and K-HORIZON.

**P-B.4 Meta-initialisation and SEAL.** MAML-en-LLM (KDD 2024, +2–4 %, LR-sensitive), ABMLL
(arXiv 2508.14285), MeTA-LoRA (arXiv 2510.11598, mixed). SEAL (MIT, arXiv 2506.10943):
self-edits applied by SFT; outer loop **ReST-EM with a binary held-out-improvement reward,
after PPO/GRPO were unstable in their setting (§3.1)**; version dependence of Table 2 (v1:
SEAL 43.8 above GPT-4.1 data 39.4 at n = 200 with LoRA; v2: SEAL 58.2 below 59.4 with full
fine-tuning); forgetting under sequential edits (81 % → 24 % after 8, Fig. 6); 30–45 s per
reward. *Used for:* estimator 2's filtered regression (06 (6.10)); the forgetting decay
`λ_dec` (06 (6.8)).

**P-B.5 Compact per-corpus state.** Cartridges (Stanford, arXiv 2506.06266): trained KV
prefixes with self-study + context distillation, 38.6x less memory and 26.4x higher throughput
at ICL-matching quality, composable; Memory Decoder (arXiv 2508.09874): a 0.5B plug-in
mimicking kNN-LM, avg ppl −6.17. *Context for:* a slow compact state per corpus is an
established, useful object; TEMPO's genome is smaller (512 reals) and carries gains rather
than memory content.

**P-B.6 GRPO and successors.** DeepSeekMath Eq. 3 (clipped ratio, group-normalised advantage
`Â = (r − mean r)/std r`, KL Eq. 4; `ε` never stated), R1 (rule rewards; neural RM rejected
for hacking), DAPO (arXiv 2503.14476: clip-higher 0.2/0.28, dynamic sampling, token-level
loss; AIME24 30 → 50), Dr. GRPO (length and std biases), GSPO (sequence-level ratio, Qwen3),
CISPO, Clip-Cov/KL-Cov (entropy collapse: ΔH ≈ −η·Cov(log π, π·A)), ProRL, spurious rewards
(arXiv 2506.10947: random rewards +21 on Qwen-Math, rarely on others), Yue et al. (NeurIPS
2025: base beats RL at large k). *Used for:* the τ1 action term (02 (2.12)) uses the
group-normalised advantage with the DAPO clip-higher and token-level loss; the τ2 rule uses
the ReST-EM form.

**P-B.7 Agentic RL.** WebRL (ICLR 2025: Llama-3.1-8B 4.8 → 42.4 on WebArena-Lite with a
self-evolving curriculum and an ORM), DigiRL, Search-R1 (retrieved-token masking; PPO 0.431
vs GRPO 0.350), ReTool, ToolRL (length reward hurts), RAGEN's "Echo Trap", SWE-RL (41.0 %
Verified), SkyRL-Agent (24.4 → 39.4 % by pure RL on 4.5K tasks, 4,601 H100-h), GiGPO
(anchor-state credit, +12–13 points), Kimi K2 / DeepSeek-V3.2 recipes (≈ 85K agentic tasks
kept iff `pass@100 > 0`). *Pattern:* sparse verifiable outcome rewards, tool-token masking,
light format shaping; dense and length rewards fail; long-horizon credit is unsolved. *Used
for:* `tempo/envs.py` uses exact verifiers and a learnability-gated curriculum (the SOMA
archive) and masks non-agent tokens.

**P-B.8 Loss improvement as a reward.** An exact Δ-training-loss RL reward exists only at
MNIST/CIFAR scale (Xu et al. 2017, arXiv 1705.11159); nearest at LM scale: SEAL (binary
downstream improvement), Montessori-Instruct (reference-loss decrease consumed by DPO), PIPO
(standardised verifier gain), RLP (information gain over an EMA baseline). **No LLM-scale work
uses pretraining-loss reduction after an update as the reward of a meta-policy.** *Status:*
K-GENE's hypothesis.

**P-B.9–11 Criticality.** Poole et al. (NIPS 2016): `χ_1 = σ_w² ∫Dz φ′(√q* z)²`, boundary
`χ_1 = 1`; Schoenholz et al. (ICLR 2017): `ξ_c^{−1} = −log χ_1` in the ordered phase,
diverging at the edge; "trainable when `L ≲ 6 ξ_c`" (empirical); Pennington et al. (NIPS
2017): `χ = 1` necessary not sufficient, dynamical isometry unachievable for ReLU; Yang &
Schoenholz (NIPS 2017): residual tanh networks are "always at the edge", ReLU residual nets
grow exponentially; Hayou et al. (ICML 2019); Cowsik et al. (ICML 2024, arXiv 2403.02579):
token-angle and gradient exponents at init predict a ViT's test loss; Doshi–He–Gromov
(NeurIPS 2023): LN + residual is "everywhere-critical". Regularising toward criticality
during training exists only at toy scale (Benati 2025; Tomihari & Karakida, NeurIPS 2025, LE
≈ 0.1; Zhang 2021); "Intelligence at the Edge of Chaos" (ICLR 2025, arXiv 2410.02536) is
correlational with a frozen backbone and no perplexity. μP fixes width exponents, not
`(σ_w, σ_b)`; depth-μP has no non-degenerate limit for block depth ≥ 2 (TP6). Edge of
stability: no LLM-scale measurement; Muon's threshold is set by the mean gradient singular
value (arXiv 2605.13079); the practical instability at scale is attention-logit explosion
(QK-clip in Kimi K2). *Used for:* 07's framing (measure first; expect near-criticality under
normalisation; the rule is a hypothesis); QK-norm stays in the backbone.

**P-B.12 Growth.** G_stack (NeurIPS 2024): 7B/300B 54.6 %, **gain decays with training length
(410M: 53.1 % at 400B → 31.0 % at 700B)**, interior optimal timing (a "valley"), one-shot
beats gradual stacking (+2.4 points), function preservation "neither necessary nor
sufficient"; HyperCloning 2.2–4x (width only; early forgetting); MSG, LiGO, TokenFormer,
Staged Training (grow when `dL/dC` hits a threshold), SPARKLING (ICML 2026, up to 35 % at 2x
width), Bu (ICML 2026: one expansion at ~80 % of a WSD horizon, ~5x at 7B), **Mallik et al.
(arXiv 2605.13405): 2x is the most reliable factor, gains most pronounced under 20 tokens per
parameter and diminishing above**; RaPTr: dropping layers can beat stacking. RL-controlled
growth exists only for CNN NAS (Cai et al. 2018); **no verified work lets an RL policy choose
a transformer LM's growth timing.** *Used for:* large-move morphology (05 §5.1); the one-shot
doubling control (05 §5.7); the local expectation that growth is compute-neutral at this
budget (ANLAGE R5) stands.

**P-B.13 MoE growth.** Sparse upcycling (ICLR 2023), BTX, aux-loss-free routing (DeepSeek-V3),
PEER, Drop-Upcycling, Expert Upcycling (arXiv 2604.19835: add experts at `τ/T ≤ 0.25`, LR
re-warmed); the upcycling scaling law (ICML 2025) finds it inefficient against
compute-optimal from-scratch. *Used for:* births re-warm the learning rate (the per-episode
schedule already does) and happen early in a life or not at all.

**P-B.14 Negative evidence on decoding weights.** Falk et al. (arXiv 2510.02096): a GPT-2
125M generated from a weight-space latent is "slightly above or at random guessing" at birth;
SANE cross-task 0.6 % vs random 0.5 %; RPG regenerates only from checkpoints of the same
model; HyperTransformer: generating only the last layer suffices from 8 channels up; Ha et
al.'s HyperLSTM generates row-scaling vectors; Hyper-Compression is below int4.
Multi-timescale memories at scale: TTT-E2E's advantage vanishes above ≈ 760M / 48B tokens
(its Fig. 5) with NIAH 0.06 vs 0.99 for full attention; GDN Table 4 and Waleffe et al. (8B):
pure recurrent/memory models keep a large recall gap that only attention-containing hybrids
close; LaCT Fig. 8(b): large-chunk nonlinear memories lose to per-token GDN at matched state;
Titans Revisited cannot reproduce Titans; RULER: RWKV/Mamba degrade by 8K. *Consequences:*
ANLAGE 11's verdict (00 §0.4 A8) stands and is anchored in the field; the τ0 fast state is a
substrate, not a claim; the backbone keeps one attention layer in four.

## P-C Corrections the evidence forced on the design

1. The text input side is a learned span embedding; the codec is output-only, with
   `σ_z = 0.3`, a weak KL with a floor and latent dropout (CALM, P-A.6). → 02 §2.2.
2. Gradient insulation of the continuous heads (Knowledge Insulation, P-A.9); "one continuous
   output space" is a hypothesis with G-ACT and a FAST-style discrete fallback. → 02 §2.4, 10.
3. Estimator 2 is best-of-G with a filtered regression update (SEAL's ReST-EM finding, P-B.4);
   GRPO's normalised advantage is an ablation. → 06 §6.6.
4. The reward weighs the episode's final loss beside its mean; the learning-rate role is
   bounded; K-HORIZON is registered (short-horizon bias, P-B.3). → 06 §6.3, §6.7.
5. Criticality is measured before it is controlled; the prior expectation is
   near-criticality under normalisation; K-CRIT is conditional on R3 (P-B.9–11). → 07.
6. Morphology acts in large, grouped moves; one-shot doubling at half the budget is a control
   (P-B.12). → 05 §5.1, §5.7.
7. The precedents for gain-space heritable objects are cited (HyperLSTM, SVF, Zhyper) and the
   SVF form is a registered ablation (P-B.1, P-B.2, P-B.14). → 04 §4.8.
8. Loss improvement as the meta-reward and RL-timed growth are stated as untested hypotheses
   at this scale, each with its kill criterion (P-B.8, P-B.12). → 10.
9. The third axis is the reverse consolidation, not a depth loop (Ouro, P-A.12). → 06.
10. B11 is stated as a path anchored on the best small open model's layout (Qwen3.5-9B, a 3:1
    GDN hybrid, P-A.13), with matched-compute parity as the local claim. → 10 K-SOTA.
11. **(Found while writing 04.)** The plan trained the expression map `θ` at τ1 by gradient
    through the gains while also requiring expression once per episode and no genome
    evaluation in the step path; these conflict. Resolution: the differentiable gains are free
    τ1 parameters within an episode and `θ` is refit by regression to them at every boundary,
    then the gains are regenerated from `g` (04 §4.5). This is the PNAS loop in gain space,
    run at every episode; it adds G-REGRESS and the K-GENE arm (d) "free gains, no genome"
    that isolates the genome's effect from the gains' effect.
12. **(Found while writing 05.)** The plan's HyperCloning split at birth halves the source
    block's contribution and so changes the function at `m_b < 1`, contradicting Theorem G1
    during the ramp. Resolution: births are copies with 1 % noise entering at `m_b = ε_skip`
    (05 §5.3); the split is used only by the one-shot control where `m_b = 1` at once.
13. **(Found while writing 06.)** Per-span summaries of an episode would cost `4 GB`; the
    consolidator reads window summaries (`32,000` per episode, `50 MB`), and salience and
    rehearsal are per window, re-fetched by identity (06 §6.1, §6.9). K-SAL is added so that
    salience-ranked rehearsal has its own kill criterion.
14. **(Found while writing 08.)** A batched life needs a definition of what the fast state
    continues across: lanes (08 §8.3), each a contiguous cursor over the stream, with
    declared boundaries enumerated; attention is window-local and the delta state is the long
    memory; evaluation is from a fresh state for comparability with token models, with the
    continuing-state score recorded beside it.
15. **(Found while writing `tempo/codec.py`.)** With the decoder's training noise added to
    the deterministic latent only, the posterior scale `s_z` entered the loss through the
    KL alone and the KL would set it to a constant; the decoder is trained on the
    reparameterised posterior sample, normalised, plus the fixed noise and the dropout (02
    §2.2). The target the backbone predicts stays deterministic.
16. **(Found while writing `tempo/synthetic.py`.)** Semitone tones from 220 Hz at a 4 kHz
    sample rate put symbols 32–55 above the Nyquist frequency and pack the lowest symbols
    inside one 62.5 Hz DFT bin; the tones are placed on the frame's DFT grid at 8 kHz
    (02 §2.9.5). The ECA damage-spreading exponent is a time average over the horizon
    because rule 90's damage set is `2^{popcount(t)}` and cancels under the periodic boundary
    at `t = w/2` (02 §2.9.6). I7's criterion is mode recovery with the KS p-value reported,
    not asserted (02 §2.10).
17. **(Found while writing `tempo/morphology.py`.)** With `a_G = γ_G` and `γ = 0` at birth,
    `m_G = σ(0) = 0.5` would switch every group on at birth. The pre-activation carries the
    baseline `−4`: `a_G = −4 + γ_G` (05 §5.1, 04 §4.4). Theorem G2 is restated in its local
    form (05 §5.4): the block's gradient is `m_b` times its output sensitivity at the current
    state, so doubling `m_b` doubles it up to `O(m_b)`; the earlier "m times the gradient at
    `m = 1`" ignored that the downstream state depends on `m_b`. A test shape must keep both
    operators inside its primordium (the birth copy needs a source of the same operator);
    `TempoConfig` now rejects layouts that do not, and the CPU test shape is `L = 8` with
    attention every second layer.
18. **(Found while writing `tempo/criticality.py`.)** The fused scaled-dot-product attention
    kernels (CPU flash, and the CUDA flash/memory-efficient kernels) implement neither
    forward-mode AD nor a double backward, so the probe differentiates an explicit softmax
    attention (07 §7.1); training keeps the fused path. With one gradient step per boundary
    the KL penalty of estimator 1 has zero gradient at the point it is taken (06 §6.5 note
    in `tempo/consolidate.py`); it is live under inner steps, which estimator 2 uses.
19. **(Found by the first life on CPU.)** With the prior `g_0 = 0`, `B tanh(A g)` is zero for
    every `B`: the regression of the episode's gains (04 (4.4)) returned a residual of exactly
    1 and the decay of (6.8) pulled the genome toward the degenerate point. The prior is a
    seeded `N(0, I)` vector and the decay pulls toward it (04 §4.1, 06 §6.4). The interactive
    streams size their utterances, scene runs and prompts by `T_s` and clip every position to
    the window, so the CPU test shape (`T_s = 8`) runs the same code as the adult
    (`tempo/streams.py`). The text lanes read the shard in corpus order from seeded offsets;
    ANLAGE's class mixture is not used inside the text stream (the stream mixture role
    replaces it), which 08 §8.3 now says.
20. **(Found by R1's first steps on the card.)** The recorded text score included the latent
    regression term `λ_lat ‖ŷ − x̄‖²` of (2.6), which trains the backbone but is not a
    likelihood (it added `≈ 10` nats per token at initialisation). The record, the window
    summaries and the evaluation carry the NLL through the frozen decoder alone; the latent
    term is recorded beside it. R0 at 50M tokens left the codec ceiling at `0.0126`; the
    budget is 200M tokens (ceiling `0.00137`, noise gap `0.00013`), the 50M artifact kept.
21. **(Found by R1 seed 0 on the card, `runs/tempo/R1frozen-a_gamma0-ts-s0`.)** With the
    decoder frozen, the span agent at the primordium reached only `5.12` nats per token after
    25M tokens (the token model: `1.85`), with the latent regression term flat at
    `≈ ‖x̄‖² = 256` (`6.0` nats per token at `λ_lat = 0.1`) and the training NLL declining
    from `10.2` to `5.0`. The cause is structural: a deterministic `ŷ` can only be the
    conditional mean of the next span's code, and a reconstruction decoder trained within
    `σ_z` of true codes has no meaning there, so the likelihood's gradient carried no
    information beyond the regression to a blur. The decoder `D_txt` is now initialised from
    R0 and trained at τ1 with the backbone (02 §2.2); the encoder stays frozen. The
    likelihood remains proper; the `K = 1` case is a token softmax; `λ_lat = 0` is a
    registered ablation. R1 was restarted with the same codec.
22. **(Found by R1 with the trainable decoder.)** The span agent reached `2.64` nats per
    token at 25M tokens (the token model `1.85`); its curve was still falling steeply in the
    decay phase and the latent term stayed at `≈ 0.8 ‖x̄‖²`, so the `λ_lat = 0` ablation
    (`nolat`) runs beside R1 before K-CONT is read. G-STAT's first form ("the last 20 %
    flat to `2 σ_e`") failed by `0.18` nats on a plain life: the decay-to-zero schedule makes
    the last tenth of every episode the steepest, so the rule would have doubled `N` at every
    boundary. G-STAT is now the transient test (1.6): the rate of decrease over the last
    fifth under a quarter of the first fifth's, or flat to `2 σ_e`, with `N` doubled at most
    once per life. Measured on R1 seeds 0 and 1 (1525 steps, 25M tokens): `Δ_first = 0.90,
    0.96`, `Δ_last = 0.18, 0.19` (ratio `0.20`), and `0.09, 0.11` on the constant-rate
    plateau (ratio `0.10`): the fast system keeps learning at every budget in this regime,
    which is the lifelong setting itself; (C1) holds only approximately and the paired CRN
    contrasts, in which the continued improvement is a common term, carry the verdicts.
23. **(R1, `runs/tempo/R1-nolat-ts-s{0..3}` against `R1-a_gamma0-ts-s{0..3}`.)** The latent
    regression term of (2.6) at `λ_lat = 0.1`: test NLL `2.638, 2.619, 2.626, 2.629` (mean
    `2.628`); at `λ_lat = 0`: `2.403, 2.406, 2.405, 2.406` (mean `2.405`, `σ_e = 0.0014`);
    paired difference `+0.223` nats at 25M tokens, the same windows in the same order. With
    the term, the depth profile sat at `Λ ≈ −0.24` (ordered); without it, `Λ ≈ +0.04`. The
    regression to a conditional mean fights the likelihood's gradient and is dropped:
    `λ_lat = 0` is the default of every later run and `lambda_lat = 0.1` is the recorded
    ablation arm (`lat01`). K-CONT compares the token baseline with the `λ_lat = 0` agent.
24. **(R2, K-CONT at `K = 4` and `K = 2`.)** At matched training FLOPs per token (the token
    model's width chosen so that its forward FLOPs per token are nearest the span agent's,
    backbone plus codec, measured by `torch.utils.flop_counter`; ratios `1.16` and `1.17`)
    and scored on the same target tokens `K .. T + K − 1` of the v3 test windows:
    `K = 4`: span agent `2.405` (σ_e `0.0014`) against the `d = 256, L = 8` token model
    `1.892`, paired difference `+0.511 [+0.506, +0.516]`, bar `0.004`: **FIRED**.
    `K = 2` (codec ceiling `0.00071`): span agent `2.108, 2.115, 2.119, 2.107` (mean `2.112`,
    σ_e `0.0057`) against the `d = 320` token model `1.836`, difference `+0.278 [+0.269, +0.286]`,
    bar `0.017`: **FIRED**. The span agent's throughput was the same at both `K` (`77k` against
    `78k` tokens per second): the backbone step is not the bottleneck at the primordium, the
    per-token decoder and the data path are, so halving `K` did not halve the throughput either.
    Per 10 §10.2 the protocol proceeds to `K = 1` (the token codec).

### The adversarial review of 2026-09-04 (corrections 25-31)

The user asked for two adversarial passes over the finished work: one over the specification
documents 00-11, one over the code, each "truth seeking, transparent, objective", with the
instruction to fix everything that arose. Seven reviewers ran (three on the documents, four on
the code); every finding below was reproduced by hand before it was accepted, and findings
that did not survive that check are recorded at the end of §P-C.31 rather than acted on. The
protocol was stopped for the duration; R1 and R2 are re-run with the corrected code, and the
runs made with the first-pass code are kept on disk with a `-v1` suffix as the record.

25. **(Genome and expression.)** Four defects in 04, all of them silently disabling a
    mechanism the design depends on.
    (a) *The programme kinds could never move.* (4.1a)'s `B_ρ` starts at zero and is trained
    only by the span loss's gradient; the learning-rate, morphology and mixture roles are
    consumed by the optimiser, the switch pattern and the data draw, which are not
    differentiable, so nothing would ever move their `B_ρ` from zero: `γ ≡ 0` for those roles
    for the whole life. The genome's action on *how to learn*, on *when to grow* and on *what
    to read* — three of the design's five claims — was dead by construction. Fixed by the
    differential frame (4.1b): a fixed random `B^{prog}` applied to `tanh(A g) − tanh(A g_0)`,
    zero at the prior and live everywhere else.
    (b) *The regression was trivially exact.* Fitting `θ` to the single pair `(g_e, u_e)` has
    far more free parameters than constraints, so `R_e` was `10^{-6}` by construction and said
    nothing about whether `Φ_θ` generalises to `g_{e+1}`, which is exactly what the next
    episode asks of it. Fixed by regressing over a history of the last 16 pairs, with `R_e`
    reported on the newest; E6 now asserts both that one pair is exactly fittable and that
    sixteen are not.
    (c) *`A` saturated `tanh`.* At `A ~ N(0, 1/d_g)` a genome at the bound gives
    pre-activations of order `sqrt(r)`, deep in `tanh`'s flat region, so the map's Jacobian in
    `g` vanished where the trust region puts the proposals. `A ~ N(0, 1/(4 d_g))`.
    (d) *The bound was on the wrong quantity.* `‖g‖ ≤ 4 sqrt(d_g)` bounds the distance from the
    origin, but the prior `g_0` is itself a random vector of norm `≈ sqrt(d_g)`, so the bound
    was a function of the seed. The bound is now `‖g − g_0‖ ≤ 3 sqrt(d_g)`, the distance
    travelled, and the decay of (6.8) is about `g_0` as it always claimed to be.
    Two further items in 05 belong here: births were applied to head and slice groups of layers
    that were themselves inactive (copying from a dormant source), and the symmetry-breaking
    noise was drawn on the parameter's device, so a CUDA life and a CPU life made different
    copies from the same seed. Both fixed; the source layer is `l − L/4` and an inactive source
    raises rather than silently copying.

26. **(The interface.)** Eight defects in 02 and 03, found by reading the acting path against
    the training path.
    (a) *`W_in`'s gains were dead.* The input projection has a matrix role in the registry, but
    the forward pass used `self.W["IN_W"]` directly instead of the gain fold, so the genome's
    first `1,536` gains did nothing. Fixed; P6 now perturbs every role.
    (b) *Bias-free linear encoders erased magnitude.* (2.1) divides by the RMS, so
    `E_prop(s) = W s` sends `s` and `2 s` to the same perception vector: a scalar reward became
    its sign. The linear encoders are affine with a random bias.
    (c) *The speech encoder was a convolution over frames.* The acting path perceives one frame
    with zero left context, the training path a whole window with three: the same frame gave
    different features on the two paths. It is now per frame.
    (d) *The KV cache corrupted the window.* A cache carried across a window boundary continues
    the RoPE positions past `T_s` while training always starts at zero. The cache is now
    window-local: a full cache starts a new window, and an over-long call raises.
    (e) *The insulation scale jumped.* `κ_m` went from `0` to `1` between two consecutive steps,
    delivering a previously insulated head's full gradient to the backbone in one step; and the
    release rule ("below half its initial value") is not a decrease at all for the negative
    densities several of these heads have. The rule is now an improvement of half the head's
    first value, followed by a ramp that fills one episode.
    (f) *The z-loss scaled with the batch.* It was divided by the number of rows, so a window
    that scores few tokens got a larger per-row penalty. It is now per row.
    (g) *The Gaussian head's log-std bound was a graph clamp*, which has zero gradient outside
    the range and can never return; it is now a projection after the optimiser step.
    (h) *`K = 1` kept two decoder layers*, giving the fallback a two-layer transformer over a
    sequence of length one that the token baseline does not have. The `K = 1` head is now the
    layer-free softmax of (2.4b).
    Also in this group: the flow head has no tractable density and cannot carry the
    policy-gradient term, which the text implied it could; its time features were not periodic
    on `[0, 1]`; and the scene generator's queries were degenerate (`SAME` could name one cell
    twice). All fixed or stated.

27. **(The life loop.)** The order seed was `hash(seed, e)`, so arms of a contrast at the same
    seed read different windows unless their model seeds also matched — the CRN pairing the
    whole protocol rests on was not actually enforced when an arm's initialisation differed.
    The order seed is now a separate, explicit input, defaulting to the model seed. Two other
    items: `F_episode` was read after a G-STAT doubling of `N` (so an episode's recorded cost
    and its `N` disagreed), and the A8 ratio compared `τ` (a genome-space norm) with `N η̄_1`
    (a weight-space one), which is not a ratio of comparable quantities. A8 is now measured in
    gain space, where both the genome step and the τ1 step act (01 (1.6)).

28. **(Consolidation.)** Six defects in 06.
    (a) *The proposal's scale head was dead.* `log σ` clamped to `[−5, 0]` means `σ ≤ 1`, so
    the birth draw has norm `sqrt(d_g) = 5 τ` and *every* early proposal was clipped to the
    trust region's boundary: only its direction survived and the head's output was irrelevant.
    The clamp is now `[0.02, 0.2]`, the upper bound being `τ / sqrt(d_g)`.
    (b) *Estimator 1 was off-policy by one step.* The proposal was drawn and then `ψ` updated,
    so the gradient scored a sample of a policy that no longer existed. The order is now
    update-then-propose.
    (c) *The KL term of (6.9) was inert* — `KL(p ‖ p)` is stationary at a single step — and is
    removed with the reason recorded.
    (d) *The forward read was crippled.* Both directions took `c_1` as the read-out, which for
    the forward read is the output after a single window: K-REV was comparing a full read
    against almost no read. Both now take the output at the end of the sequence consumed.
    (e) *The salience objective trained the mixer*, moving the policy between the drawing of a
    proposal and the taking of its gradient. It now trains `w_s` alone, on cached outputs.
    (f) *The rehearsal buffer could not select.* At `n_reh = p_reh N B` the rehearsal phase
    consumes the whole buffer whatever the weights are, so the `∝ s_w` draw was a no-op and
    K-SAL would have measured nothing. The buffer is now twice the phase's size.
    Also: the accepted set of estimator 2 grew without bound while storing read-out features
    that go stale as the mixer moves (now bounded at 8, with the caveat stated); the update was
    called ReST-EM, which it is not (it is the cross-entropy method's elite regression, and the
    name is withdrawn); and the ES arm's step length was an uncontrolled function of the reward
    scale (now exactly `τ`).

29. **(The reward, and what a rollout row is.)** Three defects that together made the reward
    of 06 §6.3 measure the wrong thing.
    (a) *The reward was mixture-weighted*, and the mixture is a genome role: a proposal could
    raise its own reward by moving `μ` onto whichever stream had the numerically smallest loss,
    with no learning at all. The weights are now uniform over the streams present.
    (b) *The streams' losses were incommensurable* — `≈ 2` nats per text token against `≈ 60`
    per speech frame against a return of `≈ −100` — so an unstandardised mean was a reward for
    the loudest stream. Each stream now enters divided by the spread of its own first
    evaluation.
    (c) *The reward was read off the training curve*, so it paid for reading easy windows. It is
    now read off fixed fitness sets, disjoint from the evaluation sets, from a fresh state, and
    the reward-bearing streams are scored by *acting* (greedy rollouts, exact reward) rather
    than by the likelihood of their own samples.
    In the same pass, the rollout rows themselves were wrong: one row carried both the agent's
    emitted answer as *input* and the true answer as *target*, which is not the likelihood of
    anything, and the rows of a group were scored under whatever state the batch happened to
    give them rather than the state they were sampled under. A draw now yields one
    teacher-forced supervised row and `G` rollout rows, each carrying its own entering state.

30. **(Criticality.)** The per-sub-block exponents were computed from an *independent*
    isotropic perturbation at each sub-block, so their sum was not the exponent of the composed
    stack and `Λ` was not the quantity the band and the rule claimed to control. One tangent is
    now propagated through the whole active stack, which makes the per-block profile an exact
    decomposition of `Λ` (K2b). A second point: under Peri-LN the residual stream grows with
    depth, so a perturbation that grows at the stream's own rate leaves the *relative*
    perturbation unchanged and is the neutral case; the rule and the band now use the relative
    exponent, with the absolute one recorded beside it. The probe also ran on a fresh
    evaluation batch with zero states; it now runs on the episode's last windows with the
    states they entered with, under its own seeded generator.

31. **(The protocol and the scripts.)** The verdict rule was stated in two incompatible ways
    (§10.5's point-estimate rule and §10.2's per-criterion prose), every bar used one global
    `σ_e` even for criteria whose metric is a per-stream reward, K-BLEND named stream-specific
    arms that no run produced (so it could only ever report "did not fire" on missing data),
    K-MORPH compared arms at equal *episodes* while their per-episode costs differ, K-SOTA's
    bar was a number from a different budget and shape, and the K-CONT token baseline was run
    without value embeddings and at equal *tokens* rather than equal training FLOPs. All are
    fixed in 10 and in `scripts/tempo_analyse.py`: one verdict rule with the confidence
    interval, a per-criterion `σ` from the control arm's own seeds, equal-compute
    interpolation of the score curves, `only-<stream>` arms in the chain, an `R10-token`
    baseline at the adult budget, and NOT EVALUATED wherever an arm is missing or a record does
    not say `"complete": true`.

    *Findings raised by the reviewers and not acted on, with the reason.* (i) "The delta
    rule's state should be reset between rollout groups" — it is, by the boundary rule (d) of
    08 §8.3; the reviewer had read an earlier draft. (ii) "The consolidator should read spans,
    not windows" — the storage cost is `4 GB` against `50 MB` and the design's own §6.1 states
    the choice; recorded as a limitation, not a defect. (iii) "The morphology roles should be
    per block rather than per group" — that is the large-move rule of 05 §5.1, which the growth
    literature supports (11 §P-B.12); it stays. (iv) "`Λ` should be compared with the pattern
    streams' `λ_ref`" — they are different quantities (07 §7.2) and the record keeps them side
    by side without a target, which is what the reviewer's own argument implies.

32. **(Pause and resume, and the device bug it exposed.)** The card is shared, so the chain
    must be stoppable on request. `Life.run_episode` now takes a `should_pause` callback and
    raises `Paused` after a step, leaving the life resumable *inside* the episode; the
    checkpoint records the in-flight episode and its next step, and the resumed life skips the
    boundary rather than re-drawing the episode's gains, pattern, optimisers and lanes. The
    end-to-end test of that path on CUDA immediately found a defect no CPU test could:
    `torch.load(map_location="cuda")` moves *every* tensor in a checkpoint to the device,
    including the RNG byte tensors (which `set_rng_state` refuses) and the window summaries
    (which are a CPU structure, so the next `append` concatenated across devices). Both are
    now brought back explicitly, and the accepted set of estimator 2 is moved to the
    consolidator's device. The same defect would have broken `--load` in R7 and R9, which are
    the only other paths that read a checkpoint from disk on CUDA — a reminder that a resume
    path is only verified by resuming on the device the runs use. A paused and resumed life
    reproduces the uninterrupted one to every recorded digit (L2c, and measured end to end at
    `N = 40` on this card: `L_bar 6.3490`, `test 5.3459`, `Lambda -0.310` in both).

### The second adversarial round of 2026-09-04 (corrections 33-36)

The user asked whether a second verification round was worth running. It was. Five reviewers
went over the *changed* surface -- the four modules the first round's fixes rewrote, the new
reward, the batched acting path, and the analysis scripts, which no round had touched -- and
found about as many confirmed defects as the first round did, several of them **in the fixes
themselves**. Two were introduced by the first round's own corrections, and one by an
optimisation added the same day. Every finding below was reproduced by hand before it was
accepted; the reviewers' own reproductions are in the session's scratchpad.

The lesson worth keeping is narrow and repeatable: **a mechanism can be disabled by the shape
of its parametrisation, and a test that exercises it will still pass.** Three of the four
corrections below are of that kind. The counter-measure is not more tests but a specific
question, asked of every mechanism: *can anything move this parameter at all, from where it
starts?*

33. **(Criticality.)** Two defects in the trigger, one of them in this document.
    (a) *The significance gate was prose.* §7.2 says "no verdict is read off a `Λ` whose
    displacement from the band is inside `2 Λ_se`" and §7.6 repeats it as part of the trigger.
    `needs_control` compared `|Λ|` with the band and never read a standard error; `Λ_se` was
    computed, recorded, and used by nothing in the repository. The trigger is now a function
    that returns both limbs, each displacement in units of its own standard error, and the
    verdict; the probe computes the per-block standard errors it needs.
    (b) *The band was on a depth-extensive quantity.* `Λ` is a sum over `K_a = 2 × |active
    layers|` sub-blocks, so a fixed band on it is a band on depth: the same per-block behaviour
    that gives `Λ = −0.51` at the 8-layer primordium gives `−1.02` at the 16-layer adult, and
    **growth alone would fire the trigger** — in a design whose whole point is that the network
    grows. Both limbs are now per block.
    Also: an arbitrary `RuntimeError` inside the model was being swallowed into the
    double-backward fallback, which runs with `strict=False` and returns *zeros* for an output
    it believes independent of the input — a dead tangent entering the profile as `log 0`
    rather than as a failure; the fallback is now reached only by an operator with no
    forward-mode rule, and runs strict. The logarithms have a floor. `zeta_update` no longer
    clamps entries the probe did not reach.
    **And a correction to a claim made in this document.** §7.6 said the first measured episode
    sat "outside the band by about `2 Λ_se`". It does not: `|Λ| = 0.511` exceeds `0.5` by
    `0.011`, which is `0.21 Λ_se`. The `9.7 σ` in that measurement is the distance of `Λ` from
    *zero*, which is not what the trigger asks. Under the corrected trigger the same numbers
    give `Λ/K_a = −0.032` and `max_k |λ_k^{rel}| = 0.046`, both far inside the band, and K-CRIT
    does not trigger on them. The open observation stands as an open observation (§P-D.1); what
    was wrong was the arithmetic asserted about it.

34. **(The FLOP basis.)** The protocol had **two mutually inconsistent definitions of training
    FLOPs, differing by a factor of 1.84**, and the acceptance tests covered the one K-CONT does
    not use. `scripts/tempo_cont.py` matches the token baseline on the span agent's *measured*
    forward cost — input embedding, backbone and span decoder — while `F_episode`, `F_cum`, the
    reward's denominator and every equal-compute comparison use 05 (5.2), which counts the
    backbone alone. The span decoder is **44 % of the span agent's forward cost** at the TS
    shape and is absent from (5.2) entirely.
    This is not a rescaling that cancels. The adult-over-primordium cost ratio is `3.93` on the
    backbone alone and `2.61` with the interface charged to both, so K-MORPH's equal-FLOP
    comparison would hand the flat-adult arm about 50 % more episodes than it is owed, and
    K-CONT's token baseline is trained on 1.84× the FLOPs the span agent's own record says it
    spent.

35. **(The batched acting path — an optimisation added the same day, and broken.)** To make the
    blend runs feasible, a group's rollouts were made to step in lockstep through one batched
    forward. Three defects, all mine:
    (a) *The mask destroyed the attention context.* Holding a finished branch inactive made the
    lanes' KV caches diverge in length; a batched call whose caches differ has no batched cache
    at all, so the call ran with **no cache** and then wrote the resulting length-one cache back
    over every *active* lane. The context never recovered. Rollouts were therefore sampled with
    attention over a single span while `window_loss` re-scored them with the full window — the
    off-policy defect §P-C.29 exists to prevent, reintroduced through the cache instead of the
    fast state. The mask is gone: every row steps, always, and a finished branch is stepped with
    a pad whose output the caller discards. A mixed cache phase now raises.
    (b) *Rows with different prompt lengths cannot be padded to a common length*, because the
    state a short row would then decode from is the state after its prompt *plus pad spans*.
    Rows are partitioned by prompt length first.
    (c) *The guard test could not fail.* It compared the rewards returned by `evaluate`, which
    are identically `0.0` for every text-emitting stream on an untrained agent. It passed for
    the whole time the cache was being destroyed. It now compares **emissions**, and asserts the
    property that actually broke: an instance scored inside a heterogeneous group must emit what
    it emits when scored alone.
    Two more found in the same pass: a task rollout's emission budget was derived from
    `len(inst.answer)`, letting the truth's length decide how long the agent may speak in the
    rollout that produces its reward (now a constant); and the task stream's parameters were
    drawn from a long-lived generator seeded by the *model* seed, so two arms of one contrast
    read different task families and a window was not reproducible from its cursor — CRN, which
    every paired verdict rests on, was broken on that stream (now a function of the order seed).

36. **(The life loop, the reward, and a head dead for the second time.)**
    (a) *The scale head was dead again.* §P-C.28 replaced a clamp of `[−5, 0]` — whose upper
    bound made every draw hit the trust region — with a clamp of `[log 0.02, log 0.2]`. At birth
    `W_σ = 0`, so the pre-activation is `0`, which is **above** the new upper bound; `clamp` has
    zero gradient outside its bounds, so `W_σ.grad` was exactly zero from every estimator and
    `σ` sat at `0.2` for the entire life. The correction reproduced the failure it was written
    to remove. The scale is now a sigmoid-squashed affine map, differentiable everywhere, with a
    bias placing birth just inside the top of the range — and the bias is in the head's
    optimiser group, which the first attempt at this fix also got wrong.
    (b) *The loss denominator was not the batch's token count.* It was
    `sum(w.n_tokens or w.T_s)`, and a rollout row has `n_tokens = 0` because it carries no
    supervised target, so `0 or T_s` charged it a whole window. Measured on the blend: **70 % of
    the denominator came from rows with zero scored positions**, and the inflation ran from
    `1.0×` on a text-only batch to `16×` on a control-only one. The stream mixture is a genome
    role, so the genome was moving the effective step size — and, through the gradient clip,
    which is not scale-invariant, the clip too — by a bookkeeping artefact rather than by the
    learning-rate role it is supposed to use.
    (c) *The criticality probe measured a discarded probe episode.* Three of the five reviewers
    found this independently. O5's estimator-2 probes overwrite the windows and states the
    probe reads, and an adopted candidate replaces the weights, gains and switch pattern; O6
    then measured that, and O8's `ζ` rule was driven by it. The boundary now measures the
    episode **before** the estimator runs, and the record's pattern, mixture, non-finite count,
    step-time median and A8 ratio are captured from the episode rather than from an adopted
    probe.
    (d) Smaller, same pass: a skipped non-finite step wrote **two** record rows, pushing the
    episode's first steps out of the last-`N` window G-STAT reads; `R_scale` was set for
    `control` alone though every reward-bearing stream's rollout rows carry returns; the control
    stream has no teacher-forced row, so the `rollout > 0` filter never charged one rollout in
    `G` of its acting compute; the mid-episode and end-of-episode evaluations used 16 and 32
    windows, so the curve stepped at `n = N` for no reason to do with the agent; the
    curriculum's fallback flag was neither initialised nor checkpointed; the SVF arm could not
    be resumed at all, because the SVD cache is filled only by a boundary and a mid-episode
    resume skips it; and the rehearsal buffer's degeneracy guard tested for a *zero* salience
    when the degenerate case at birth is a *constant* one, so the buffer was the episode's
    earliest windows, recorded as if it had been ranked.


37. **(The analysis layer.)** `scripts/tempo_analyse.py` decides every verdict of this project
    and had **no test at all**: a defect there produces a wrong *conclusion* rather than a
    crash or a bad loss, so nothing would have caught it. The second round found eighteen, of
    which these change a published verdict: K-HORIZON fired on any positive point estimate,
    with neither a bar nor an interval, and read a missing or unmeasurable contrast as "did not
    fire"; `result.json` was loaded without checking `complete`, so a partially written
    baseline counted; K-BLEND passed with its stream-specific arms missing, which 10 §10.2
    explicitly forbids; K-XFER counted a stream that was never run as a lost pair, so two
    missing streams could fire it; a stream whose greedy reward failed to record fell through
    to its NLL, putting a loss in nats where a negative reward belonged and reporting the change
    of units as a ten-nat regression; fewer than two paired seeds gave INCONCLUSIVE, which says
    "run and unclear" about a contrast that was not run; a control arm with exactly zero spread
    made the criterion vanish through a truthiness test; and a `null` in a record raised rather
    than being dropped, so one non-finite reward suppressed the entire report.
    The layer is rewritten around one `criterion()` function that applies 10 §10.5 once, and
    `tests_tempo/unit/test_analyse.py` feeds it synthetic records whose right answer is known —
    every boundary case of both forms of the rule, incomplete and truncated records, missing
    arms, null rewards, and the equal-compute interpolation's edges.
    Two protocol-level gaps closed with it: `codec` and `G-ACT` results now carry `complete`
    (the existing codec artifacts were backfilled with the flag *and a marker saying so*, rather
    than the reader being taught to assume it), and the analyser reports every configuration
    present rather than only `ts`, so the `K = 2` and `K = 1` fallback runs the chain produces
    are actually read.


38. **(The streams.)** `tempo/streams.py` is the module the corrections rewrote most and the
    only one that had no test file of its own; four of the second round's findings live in it,
    and each is cited from the code by this number.

    *A chaotic map re-derived from a rounded state is a different trajectory.* `PatternStream`
    cached the state at each window's start and continued from it on a re-fetch, but the cached
    state was the **float32 observation**. On the logistic map at `r → 4` the exponent is about
    `ln 2` per step, so 128 steps amplify a float32 rounding to `O(1)`: a window rehearsed or
    probed by identity was not the window whose loss had been recorded, and because only the
    *first* window of a lane was reconstructed this way, the corruption was confined to `k = 0`
    and invisible in any aggregate. The cache is gone. `_observations(kind, seed, k)` is now a
    pure function that replays from the generator's own parameters, which is what 08 §8.3
    requires of an identity; the lane still pays nothing to continue, because it carries the
    live float64 tail in its cursor, and a test asserts the two routes agree.

    *A rollout group is a new source.* 08 §8.3 (d) makes every rollout group start from a reset
    state. `ControlStream` forked the lane and kept its state, so the `boundary` flag on every
    control window was inert — the `G` branches of a group began from whatever the lane had
    accumulated, and the group's rollouts were neither independent of each other's history nor
    reproducible from the identity. The branches are now reset explicitly.

    *A row that scores frames must not claim text tokens.* The speech **production** row's
    `n_tokens` counted `(n_p − 1) K` — the prompt's tokens — although the row scores frames
    through a Gaussian head and no text at all. `n_tokens` is the per-token accounting's
    denominator, so every batch containing a production row divided its text loss by a token
    count that included tokens no text head had scored. It is now `0`, and `scored_tokens`
    exists for the callers that want every modality.

    *A supervised row must be teacher-forced.* Speech **transcription**'s supervised row was
    laid out with an empty answer, so the truth was never an input: half its scored targets
    were predicted from a padded zero frame, and the row was not the likelihood of anything.
    08 §8.3 requires the truth at the answer positions of a supervised row. It is now passed,
    and `tests_tempo/unit/test_streams.py::TestSupervisedRowsAreTeacherForced` checks, for the
    task, vision and speech streams, that every scored target position is followed by an input
    carrying that same truth span.

    *How the omission of this entry was found.* It was not. The four citations `(11 §P-C.38)`
    were written into `streams.py` while the entry itself was never added, so for a day the
    code pointed at a section that did not exist. `scripts/tempo_spec_trace.py` was extended in
    the same round to check **doc → doc** cross-references and does not check **code → doc**
    citations; that gap is now on the round-3 work order. A pointer that resolves to nothing is
    the cheapest possible kind of unverifiable claim, and it survived a round whose whole
    purpose was to find such things.


## P-D. Open observations (recorded before the run that decides them)

Items here are **not results**. They are things a single run showed that would matter if they
survive, written down before the deciding run so that the reading cannot be chosen afterwards.
Each says what would confirm it, what would refute it, and what either would mean.

1. **The criticality band is very loose, and K-CRIT may never be evaluated at all.**
   *Observed:* `runs/tempo/R1-nolat-ts-s0`, the first episode measured with the propagated
   tangent of 07 (7.2)-(7.4): at the primordium with `γ ≡ 0`, the `ζ` rule off, `λ_lat = 0`,
   after 25M tokens, `Λ = −0.511 ± 0.053` over `K_a = 16` blocks, i.e.
   `Λ / K_a = −0.0319 ± 0.0033`, `max_k |λ_k^{rel}| = 0.0461` (ffn, layer 0). The sixteen
   per-block exponents are tightly clustered in `[−0.046, +0.007]`: fifteen negative, one
   marginally positive, no outlier. This is a uniform, gentle contraction per block, not a
   pathological layer.

   *The correction that produced this entry.* An earlier draft of this item, and of 07 §7.6,
   said `Λ = −0.511` was "outside the band by about `2 Λ_se`" and that K-CRIT might therefore
   have to run. **That was wrong twice.** Arithmetically: `|Λ|` exceeds `0.5` by `0.011`, which
   is `0.21 Λ_se`, not two — the `9.7 σ` in that number is the distance of `Λ` from *zero*,
   which the trigger does not ask about. Semantically: the band is now on the **per-block**
   exponent (11 §P-C.33), because `Λ` is a sum over `K_a` sub-blocks and a fixed band on a sum
   is a band on depth — the same per-block behaviour gives `−0.51` at the 8-layer primordium
   and `−1.02` at the 16-layer adult, so growth alone would have fired the old trigger. Under
   the corrected trigger this profile does not fire either limb, by `0.00` standard errors.

   *What is actually open, and it is the opposite of what the first draft claimed.* The band is
   `±0.5` **per block** against a measured `0.032`: **15.7x of headroom** on the stack limb and
   **10.8x** on the worst-block limb. A per-block band of `0.5` permits each sub-block to
   amplify a perturbation by `e^{0.5} ≈ 1.65`, which compounds to `e^{8} ≈ 3000` across the
   adult stack — a network that reached the band's edge would be far past any usable notion of
   "the edge of chaos". As written, K-CRIT is therefore not a conditional criterion so much as
   an unfireable one: on the only profile in hand it would take a tenfold change to evaluate it.
   *That is a defect in the band, not a property of the network*, and it was introduced by the
   per-block correction, which fixed the depth-dependence and inherited the `±0.5` from the
   plan's absolute form without recalibrating it.

   *A second reading the profile forces.* The **absolute** exponent is `Λ_abs = +0.111`, of the
   **opposite sign** to the relative `Λ = −0.511`. The raw tangent grows through depth; it
   grows more slowly than the activations themselves, so measured against the representation's
   own scale (07 (7.4)) perturbations contract. Both are defensible readings of "criticality"
   and this specification picked the relative one; no prior work reports `λ^{rel}`, so there is
   no external calibration for either. A band chosen on one is not a band on the other, and at
   this profile they do not even agree on which side of zero the network sits.

   *R1 supplies the calibration early, and it is decisive.* The `λ_lat` ablation is a large,
   real intervention on the same agent — `nolat` against `lat01` (`λ_lat = 0.1`), paired by seed
   under CRN, four seeds each, complete:

   | arm | test (nats) | `Λ / K_a` | `Λ_abs / K_a` |
   |---|---|---|---|
   | `nolat` | 2.4026 ± 0.0037 | −0.0344 ± 0.0017 | **+0.0047** |
   | `lat01` | 2.6289 ± 0.0080 | −0.0471 ± 0.0008 | **−0.0099** |

   Paired: the objective change costs `+0.2263 ± 0.0074` nats, **`+60.6 σ_e`** — about as
   emphatic as a contrast in this project gets — and it moves the per-block exponent by
   `−0.01277 ± 0.00236`, which is **2.55 % of the band** on the stack limb.

   *Corrected 2026-09-04, same day, by adversarial review of the derived architecture note:*
   the first version of this paragraph put the consequence at "roughly 35 stacked interventions",
   which reads the **wrong limb**. `trigger()` (criticality.py:209) fires on *either* the
   per-block stack average *or* the **worst single sub-block**, and the worst-block limb binds
   first: `0.5 / 0.047 = 10.6x` of headroom at `nolat` and `0.5 / 0.090 = 5.6x` at `lat01`,
   against `15.7x` for the stack average. On that limb the same intervention moves the worst
   block by about `0.040`, i.e. **8 % of the band**, so the honest figure is roughly **11**
   stacked interventions, not 35. The conclusion is unchanged in direction and weaker in degree:
   the band is very loose and K-CRIT is unlikely ever to be evaluated, but "not a threshold this
   protocol can cross" was stated with more confidence than the binding limb supports. R3's value
   remains the distribution rather than the verdict.

   *And the sign question sharpens into the real finding.* The **absolute** per-block exponent
   sits essentially at zero and **changes sign between the arms** (`+0.0047` → `−0.0099`), while
   the **relative** one is robustly negative.

   *Also corrected the same day:* this entry first added that the relative exponent is "far more
   tightly clustered". **It is not.** Per-seed standard deviations are `sd(Λ_rel) = 0.0280` vs
   `sd(Λ_abs) = 0.0238` at `nolat`, and `0.0125` vs `0.0077` at `lat01` — the **absolute**
   exponent has the smaller spread in both arms. The relative form only looks steadier because
   its magnitude is larger; as a coefficient of variation it is, and saying so without the
   qualifier turned a ratio into a false claim about stability. The sign argument below does not
   depend on it. §7.5's cited prior —
   Doshi–He–Gromov's "everywhere-critical" result for residual stacks with normalisation, and
   Yang & Schoenholz — predicts a trained network of this shape sits *at* criticality. That is
   what `λ_abs` shows, to within `0.01` per block, across two objectives that differ by sixty
   standard errors. The order that `λ^{rel}` reports is produced by dividing out the
   activations' own growth through depth (07 (7.4)), which is a choice this specification made
   and which no prior work reports.

   So the open question is no longer "does the network leave criticality". It is **which
   exponent criticality is measured in**, and R1 already shows the two answering differently in
   the way that matters: one agrees with the literature's prediction and one does not, and the
   band was placed on the one that does not. R3 must therefore report both distributions and
   the band must be re-derived on whichever is chosen, with the choice argued before the data
   are read. Recording that the network "stays critical" on the strength of a band it could
   never have crossed would be the empty result this entry exists to prevent.

   *Decided by:* R3, twenty episodes on two seeds, which is where 07 §7.6 reads the trigger.
   R3 must report the **distribution** of `λ_k^{rel}` and of `λ_k^{abs}` over episodes, layers
   and seeds, not only whether the trigger fired — the trigger's own verdict is nearly
   foreordained by the paragraph above and carries little information.

   *If the profile stays where it is:* the record says "not needed: the normalised residual
   network stays critical", with the profile, as 07 §7.6 requires — and the band is re-derived
   from R3's measured spread (a plausible form: `± c · sd_k(λ_k^{rel})` with `c` fixed in
   advance) and the re-derivation recorded here as a correction, never applied quietly. A band
   nothing can cross is not evidence that nothing is wrong.

   *If it moves by the order of magnitude the band needs:* K-CRIT stops being conditional and
   becomes a criterion that actually has to be run — the arms `b_homeo` against `a_gamma0`. That
   would be the interesting outcome, because §7.5's cited mean-field result is that residual
   stacks with normalisation sit near criticality *by construction* (Doshi-He-Gromov's
   "everywhere-critical"; Yang & Schoenholz), and a trained, growing, gain-modulated network
   leaving that neighbourhood would be a measurement the literature does not predict — a finding
   in its own right whichever way the `b_homeo` contrast then goes.

2. **What the `K = 1` arm will and will not tell us (registered before it runs).**
   *Written 2026-09-04 at 21:5x, while the `K = 2` token seeds were still running and no
   `K = 1` artifact of any kind existed on disk.*

   K-CONT has now fired twice at matched training FLOPs, on the `ts` and `tsk2`
   configurations: `+0.5465 [+0.5390, +0.5541] n=4` at `K = 4` (146 sigma_e) and
   `+0.3310 [+0.2729, +0.3891] n=2` at `K = 2` (35 sigma_e of that configuration's own
   `sigma_e = 0.0095`). Halving `K` recovers about 40 % of the gap and costs 56 % more compute
   per token to do it (`F` rises from `1.250e15` to `1.945e15`), so the deficit is a property of
   the interface rather than of one choice of `K`.

   *Why `K = 1` is not a third repetition.* `scripts/tempo_codec.py:47` builds the `K = 1` codec
   with `n_dec_layers = 0`: a softmax head on an affine image of `y_hat`, with no within-span
   recurrence (the correction that removed two decoder layers the token baseline does not have).
   At `K = 1` there is therefore no sequence shortening and no span decoder. The two arms differ
   by exactly one structure:

   ```
     span agent   backbone(d = 512) -> W_out -> y_hat in R^256 -> W_c -> 256 -> LN -> W_v -> softmax
                                                 \____ the d_x = 256 interface ____/
     token model  backbone(d ~ 256-320) ---------------------------------------------> softmax
   ```

   *The three outcomes and what each licenses, fixed now:*

   * **Gap indistinguishable from zero.** The codec machinery is neutral overhead, and the
     `K = 4` and `K = 2` deficits are the price of predicting a *span* rather than a token. The
     failure localises to the span idea; the continuous-output interface survives, and 02's
     `d_x` bottleneck is not implicated.
   * **Gap still positive by the bar.** Predicting a continuous latent and decoding it is worse
     than a softmax over the vocabulary *even with no shortening to pay for it*. That implicates
     the `d_x = 256` interface itself, not `K`, and it is the more consequential negative result:
     the whole design routes every modality through that interface, so the finding would not be
     confined to text. The follow-up is a `d_x` sweep, not a `K` sweep.
   * **Gap negative.** The codec helps once it stops shortening. Unlikely; it would invert the
     reading and require explaining why a bottleneck plus an extra affine stage beats a direct
     softmax at equal compute.

   *What none of the three licenses:* a claim that TEMPO's other mechanisms are unaffected.
   K-GENE, K-REV, K-MORPH and K-SAL are contrasts *within* the span agent, so they remain
   well-posed at whatever `K` survives -- but if the interface is the defect, every one of them
   is being measured on a substrate that is already behind a token model, and that has to be
   said whenever they are reported.

   *A caveat that applies to all three.* The `K = 4` comparison is matched analytically to eight
   digits but *delivers* a ratio of `1.0104` in the token model's favour (`tempo_cont.py` sizes
   the baseline against an assumed 25,000,000 span-agent tokens; the agent consumed 24,985,600
   and charged itself `1.2376e15` against the analytic `1.2501e15`). One per cent of compute
   cannot produce half a nat, so no verdict above turns on it -- but the `K = 1` gap may be small
   enough that it does turn on it, and the accounting must be unified before that gap is read.


3. **Neither Lyapunov exponent is invariant across `K`, and the two move in opposite
   directions.** *Observed 2026-09-04 on the three completed R1 `nolat` arms, four seeds each,
   before the `K = 1` token baseline existed.*

   P-D.1 leaves open which exponent criticality should be measured in, and suggested that
   `lambda_abs` might be the network's own property (it sits near zero, where the mean-field
   result for normalised residual stacks predicts a trained network of this shape should sit)
   with `lambda^rel` the artefact of dividing out the stream's growth. The `K` sweep is a third,
   independent probe of that, and it **refutes the clean version of the hope**:

   | config | `Lambda_rel` | per block | `Lambda_abs` | per block | `sigma_e` |
   |---|---|---|---|---|---|
   | `K = 4` | −0.5500 ± 0.0280 | −0.0344 | +0.0753 ± 0.0238 | +0.0047 | 0.0037 |
   | `K = 2` | −0.7230 ± 0.0167 | −0.0452 | +0.1299 ± 0.0080 | +0.0081 | 0.0095 |
   | `K = 1` | −0.7693 ± 0.0236 | −0.0481 | +0.2221 ± 0.0164 | +0.0139 | 0.0108 |

   Both are monotone in `K` and both move well outside their seed spreads; `lambda_abs` **triples**
   across the sweep while `lambda^rel` falls by 40 %. They move in **opposite** directions. So
   stability across configurations cannot be used to choose between them, and the argument in
   P-D.1 must rest on what the two quantities *mean* rather than on which one holds still.
   `lambda_abs` does remain small in absolute terms (`+0.005` to `+0.014` per block, still near
   critical by any reading), which is the one part of the earlier framing that survives.

   *The confound, stated because it is fatal to a strong reading.* These three arms differ in
   more than `K`. The window holds `T_s = T/K` spans, so it is 128 spans at `K = 4` and **512**
   at `K = 1`: the delta rule's state accumulates over four times as many steps, and the
   operator's Jacobian -- which is what the probe differentiates -- genuinely depends on that.
   The codec differs too (two decoder layers at `K = 4` and `K = 2`, none at `K = 1`), and the
   compute differs by `2.49x`. This is a suggestive cross-cut, **not a controlled experiment**.
   The clean test is to vary `T_s` at fixed `K`, which no run in this protocol does; until one
   does, "the exponents depend on `K`" is the honest statement and "the exponents depend on
   sequence length" is a hypothesis.

   *A second observation from the same table, recorded because it was mis-called once.*
   `sigma_e` rises monotonically as `K` falls: `0.0037 -> 0.0095 -> 0.0108`. An interim reading
   at three `K = 1` seeds gave `0.0034` and was reported as "`K = 2` is the outlier, not a
   trend"; the fourth seed (`1.8694`, against a mean of `1.8903` for the first three) moved it to
   `0.0108` and restored the monotone reading. Recorded as a caution: at `n = 3` this project's
   own bar of two paired seeds is met while the spread is not yet estimable, and a `sigma`
   quoted from three runs should be treated as provisional. Every K-CONT bar is `3 sigma_e`, so
   `K = 1`'s bar is `0.0324` -- nearly three times `K = 4`'s `0.0112`.


4. **A softmax bottleneck as the candidate cause of the residual `K = 1` gap, and the factorial
   that tests it.** *Registered 2026-09-05 05:00, before any cell of the factorial had run.*

   *The observation to be explained.* K-CONT fired at every `K` (P-D.2): `+0.5465` at `K = 4`,
   `+0.3361` at `K = 2`, `+0.0902 [+0.0654, +0.1151]` at `K = 1` against a bar of `0.0324`. The
   first two mix two costs -- the interface, and predicting several tokens from one latent. The
   third does not: at `K = 1` the span interface buys no shortening at all, so `+0.0902` is the
   price of the interface alone.

   *What the code actually computes at `K = 1`.* `scripts/tempo_codec.py:47` builds that codec
   with `n_dec_layers = 0`, so `SpanCodec.dec` is empty; and in `decode_logits`, `prev` is
   `[BOS]` followed by `tokens[:, :-1]`, which is empty when `K = 1`. So `T_dec(prev) + P_dec`
   is a **constant** and the entire output path collapses to two linear maps in series:

   ```
      h in R^512  --W_out-->  y in R^{d_x}  --W_c-->  c in R^{d_c}  --LN-->  --W_v-->  logits in R^8192
                   (d_x x 512)              (d_c x d_x)                    (8192 x d_c)
   ```

   *Why this should cost something.* The pre-norm vector lies in an affine subspace of dimension
   at most `min(d_x, d_c, d) = 256`, and `LayerNorm` is a smooth map, which cannot raise that
   dimension. The achievable logit vectors therefore lie on a manifold of dimension at most
   **256** embedded in `R^8192`. The FLOP-matched token baseline at `K = 1` is width **384** with
   a single `unembed` from its own hidden state, so its logits lie on a **384**-dimensional
   manifold. This is exactly the *softmax bottleneck* of Yang, Dai, Salakhutdinov & Cohen (ICLR
   2018, "Breaking the Softmax Bottleneck"): the log-probability matrix of natural language is
   empirically high-rank, and a low-rank logit map cannot match it however well the rest of the
   model is trained. `+0.09` nats is the right order for a `256` against `384` restriction at
   this vocabulary.

   *Why the rank account is nearly the only structural candidate here.* At `K = 1` with
   `lambda_lat = 0` the frozen encoder **never runs during training** (`life.py:588`), so it
   constrains nothing; `x_bar` is not a target; and every part of the decoder (`W_c`, `ln_dec`,
   `W_v`, the constant) trains at tau_1 with the backbone. The span agent at `K = 1` is
   therefore a token model with a wider backbone (512 against 384) and a rank-256 bottleneck
   inserted before its softmax, at equal training FLOPs. What remains different besides the rank
   is small and recorded here for completeness: the token baseline ties its input and output
   embeddings and uses value embeddings, the span agent does neither.

   *The prediction, and why the controls make it a mechanism test rather than a scan.* The rank
   of a composition is bounded by the **narrower** map, so widening only one of the two changes
   nothing at all:

   | cell | `d_x` | `d_c` | rank | prediction |
   |---|---|---|---|---|
   | `tsk1` | 256 | 256 | 256 | baseline, gap about `+0.09` |
   | `tsk1dx512` | 512 | 256 | 256 | **no improvement** (`W_c` is still 256 wide) |
   | `tsk1dc512` | 256 | 512 | 256 | **no improvement** (`W_out` is still 256 wide) |
   | `tsk1wide` | 512 | 512 | 512 | **the gap closes** |

   A parameter scan would have run `d_x` at 256/384/512 and could not have distinguished "width
   helps" from "rank helps". The two single-widened cells add parameters and FLOPs without
   adding rank, so they are the controls that separate the two. **If either of them improves
   substantially, the rank account is wrong**, and the deficit is something else -- which is the
   outcome that would teach us the most and must be reported as such rather than explained away.

   *What a confirmation would mean, and it is not good news for the design.* If the gap closes
   only at `d_x = d_c = 512`, then TEMPO's central premise -- that every modality passes through
   one narrow continuous interface of width `d_x` -- imposes a rank-`d_x` ceiling on any softmax
   the agent ever has to produce. The interface would then have to be at least as wide as a
   competitive token model's hidden state, which is most of what the span idea was supposed to
   save. The efficiency claim and the one-interface claim would be in direct tension, and that
   tension is a result about the architecture rather than a tuning problem.

   *Design choices and why.* Two seeds per cell: the verdict layer needs two paired seeds, two
   give a directional answer in about three hours of shared GPU, and any cell that looks decisive
   is extended to four before it is reported as a claim. `--micro-batch 16` on **every** cell:
   roughly 6.8 GB of the card is held by other processes, and gradient accumulation is exact
   here (`denom` is computed over the whole batch before the chunk loop, so the accumulated
   gradient equals the full-batch one; only the floating-point summation order changes). It is
   applied uniformly so the four cells are comparable to each other, which is what this
   experiment needs; they are not bit-comparable to the earlier `tsk1` runs, which is why `tsk1`
   is re-run as a cell rather than reused. Each cell gets its own codec (its shapes change) and
   its own FLOP-matched baseline sized against the span agent's own charged compute.


5. **Why K-CONT fired: the span/compute exchange rate, derived from the three completed
   contrasts.** *2026-09-05. A reading of completed data, not a pre-registration; its limits are
   stated at the end.*

   Three K-CONT contrasts at four seeds each are enough to separate the two costs the span
   interface incurs, because one of them is `K`-invariant and the other is not. Fitting the
   measured gaps against `log2(K)`:

   ```
      gap(K) = 0.0961 + 0.2281 * log2(K)          max residual 0.0118 nats over three points

      measured   K = 4  +0.5465     fit  +0.5524
                 K = 2  +0.3361     fit  +0.3243
                 K = 1  +0.0902     fit  +0.0961
   ```

   The **intercept** is the interface cost, which does not depend on `K`; the **slope** is the
   cost of the span mechanism itself, `0.228` nats per doubling of `K`. The intercept `0.0961`
   is an independent estimate of the same quantity the `K = 1` arm measures directly at
   `+0.0902 [+0.0654, +0.1151]`, and the two agree -- a useful internal check, since they come
   from different combinations of the same runs.

   *The exchange rate, and why it is so unfavourable.* Doubling `K` saves `0.66` doublings of
   training compute (measured: `3.0764e15` at `K = 1` against `1.2376e15` at `K = 4`, so `2.49x`
   for a `4x` shortening). What is that compute worth? Our own two token baselines answer it
   without any external scaling law: `1.31` doublings of compute buy `0.0662` nats, i.e.
   **`0.050` nats per doubling**. So the compute a doubling of `K` saves is worth `0.033` nats,
   and the same doubling costs `0.228` nats of quality. **Spans cost `6.9x` what they save.**

   The decisive term is not the span mechanism's cost but the *cheapness of compute at this
   scale*: the compute-quality slope is shallow (`0.05` nats per doubling), so an efficiency
   mechanism has very little room to pay for itself, and one that costs `0.228` nats per
   doubling cannot. An efficiency mechanism is only worth having when its quality cost is well
   under the compute-quality slope, and this one is over it by nearly an order of magnitude.

   *Why the saving is only `2.49x` for a `4x` shortening, which matters.* The span decoder runs
   `K` times per span -- once per token -- so the interface's cost is `K`-invariant and dilutes
   every saving the backbone makes. At the primordium the decoder is `44 %` of the forward cost
   (09 §9.2), so shortening the sequence can never save more than about half the total however
   large `K` grows. The mechanism's ceiling is therefore low before any quality cost is counted.

   *A mechanism hypothesis, stated as such.* Twelve of the sixteen layers are the gated delta
   rule, which is **linear** in sequence length; only four are attention. Shortening a
   sequence that is already processed in linear time saves compute in proportion, with no
   super-linear win to be had. Spans and a linear-time mixer are two mechanisms attacking the
   same cost, and composing them buys the smaller of the two savings while paying both
   overheads. This would predict that spans *are* worth having in a model whose mixer is
   quadratic in sequence length, where a `K`-fold shortening is a `K^2`-fold saving in the
   dominant term. **Not tested here**, and this card cannot test it: it needs an
   attention-only backbone at a sequence length where attention dominates.

   *Limits, so this is not over-read.* Three points, one card, one scale (25M tokens, a 22M-
   parameter active shape), one corpus, one tokenizer. The fit has two free parameters for three
   points, so its functional form is a choice rather than a finding -- what is measured is that
   the gap falls monotonically and roughly evenly per halving of `K`, and that the `K = 1`
   residual is non-zero. The compute-quality slope is estimated from **two** baselines and is
   the weakest number here; a third point would improve it. None of this is a scaling law, and
   nothing here licenses a claim about what happens at 1B parameters.


6. **K-GENE's bar is a between-seed scale applied to a within-pair contrast, and the effect it
   is looking for may sit under it.** *Registered 2026-09-05, before R4 has run.*

   *The arithmetic.* `criterion()` sets `bar = 3 * sigma`, where `sigma` is the **control arm's
   spread across seeds** on that criterion's metric (`tempo_analyse.py:269`); for K-GENE the
   metric is the score at equal cumulative FLOPs. A contrast is credited to the mechanism only
   when `mean <= -bar` *and* the interval excludes zero. Our measured single-episode spreads are
   `sigma_e = 0.0037` at `K = 4` and `0.0108` at `K = 1`, giving bars of `0.011` and `0.032`
   nats. R4's own control spread will differ -- it is a 20-episode equal-`F` score, not a
   one-episode test score -- but these are the only estimates in hand.

   *What effect size is expected.* The record this project inherited (00 §0.1, from the ANLAGE
   lineage) measures the two things a gain genome can plausibly do: an inherited scale profile
   was worth **`+0.014` nats** after 2.5M tokens on every seed, and the learning-rate axis moved
   **`0.09` nats**, while the data-mixture programme was noise. So the genome's plausible reach
   is roughly `0.014` to `0.09` nats.

   Set against a bar of `0.011`-`0.032`: **the upper end of that range clears the bar and the
   lower end cannot.** An effect of `0.014` nats that reproduced on every seed with a tight
   paired interval would still be recorded as K-GENE FIRING -- that is, as the genome being
   decorative -- because it is under three times the between-seed spread.

   *Why the two scales are different, which is the substantive point.* The arms of K-GENE are
   paired by seed under common random numbers: they share the initialisation and the window
   order and differ only in the genome. The variance that CRN cancels is exactly the between-seed
   variance the bar is built from, so the bar is calibrated on a noise source the design has
   already removed. This is the paired-test analogue of judging a within-subject effect against
   the between-subject standard deviation: statistically the wrong yardstick, though defensible
   as a *practical* one -- "an effect smaller than ordinary run-to-run variation is not worth
   11.3M parameters of expression map, 5.2M of consolidator and 50 % probe overhead" is a
   coherent position, and it is the position the protocol takes.

   *What is registered here, before R4.* Both readings are reported, and neither is allowed to
   stand alone:

   * the **paired contrast** with its interval -- does the genome move the score at all, and
     reproducibly? (the analyser already prints `mean [lo, hi] n=`);
   * the **3-sigma verdict** -- is the move large enough to be worth the machinery?

   A result that is statistically solid and under the bar is to be reported as *"real but below
   the practical threshold"*, with both numbers, and **not** as "the genome does not work". The
   converse also holds: a contrast that clears the bar with an interval touching zero is not a
   result either. This is not a change to the verdict rule -- the rule stays as pre-registered --
   it is a statement of how its output must be read, fixed before the data exist so that it
   cannot be chosen afterwards.

   *A design consequence worth noting.* Because the bar does not shrink with seeds while the
   interval does, adding seeds sharpens the statistical question and does nothing for the
   practical one. If R4's control spread turns out large, the honest response is to say the
   experiment cannot resolve effects of the expected size and to report the paired estimate with
   its interval, rather than to add seeds and pretend the bar moved.


7. **The premise of P-D.4, verified directly on the map -- and the two instruments that got it
   wrong first.** *2026-09-05, while the factorial's first cell was training.*

   P-D.4's hypothesis rests on a claim about the *architecture*, not about training: that the
   achievable logit family at `K = 1` has at most `min(d_x, d_c)` degrees of freedom. That is
   checkable in seconds on the CPU, and it should be checked before the training runs are read,
   because if it is false the factorial's two single-widened cells are not controls at all.

   *Attempt 1, wrong instrument.* Push 2048 random latents through the real decoder and take the
   singular spectrum of the resulting `(2048 x 8192)` logit matrix. Result: **full rank (2047)
   in every cell**, no cliff anywhere. The measure is confounded. `decode_logits` ends in
   `SOFTCAP * tanh(z / SOFTCAP)`, a pointwise nonlinearity, so the logits are a *curved* image
   of the latent space, and a finite sample of a curved `k`-dimensional manifold generically
   spans far more than `k` directions. **Matrix rank is the wrong instrument the moment a
   nonlinearity is in the path** -- and note this also weakens the naive form of the softmax-
   bottleneck argument, which is usually stated for a *linear* head: softcap already breaks the
   linear rank bound. What it cannot break is the count of degrees of freedom.

   *Attempt 2, right instrument, wrong precision.* The dimension of the image manifold is the
   rank of the Jacobian `d(logits)/dy`. Measured on the float32 modules it gave
   `tsk1dx512 -> 512`, contradicting the algebra (`W_c` is `256 x 512`, so the Jacobian must
   factor through 256 dimensions and exactly half its singular values must vanish). The cause
   was the instrument again: float32 has a relative noise floor near `1e-7`, and the spectrum's
   tail had already fallen to `1.6e-7` by index 255, so the floor sat *above* the cliff and hid
   it. A threshold cannot resolve a structure finer than the arithmetic that produced it.

   *Attempt 3, in float64.* Casting the modules to double drops the floor to `~1e-16`:

   | cell | `d_x` | `d_c` | bound | measured dof | the cliff |
   |---|---|---|---|---|---|
   | `tsk1` | 256 | 256 | 256 | **255** | `sv[255]/sv[0] = 6.8e-16` |
   | `tsk1dx512` | 512 | 256 | 256 | **255** | `sv[255] = 9.6e-16`, `sv[256] = 9.4e-16` |
   | `tsk1dc512` | 256 | 512 | 256 | **256** | no cliff at 255 (`sv[255] = 2.0e-2`) |
   | `tsk1wide` | 512 | 512 | 512 | **511** | `sv[511]/sv[0] = 6.0e-16` |

   The bound bites exactly where predicted. The `-1` is `LayerNorm` removing the mean direction.
   **`tsk1dx512` adds 262,144 parameters and their FLOPs and buys precisely zero extra degrees
   of freedom** -- which is what makes it a control rather than another point on a scan.

   *What this does and does not establish.* It establishes the *necessary* half: the narrow
   cells cannot express a richer family of next-token distributions than the baseline, whatever
   training does. It does not establish that the missing degrees of freedom are what the loss is
   paying for -- a model may fail to use the freedom it has. Only `tsk1wide`'s trained score can
   say that, which is what the factorial is for. Recorded now so that the architectural claim
   and the empirical one stay distinguishable when the runs land.


8. **Both estimators are degenerate, in opposite directions -- found before R4 rather than
   after.** *2026-09-05, on CPU at the TINY shape, while the factorial held the GPU.*

   *Why this was checked at all.* Every run of this session has had `genome = False`. R4 is
   roughly eighteen GPU-hours and rests entirely on the tau_2 path, so before committing to it:
   does a genome life run, does `g` move, is the reward finite, and do both estimators behave?
   The first three pass. The fourth does not.

   *What was measured.* Ten consecutive episodes at TINY (`d_g = 24`, `G = 2`, `n_probe = 2`),
   recording the proposal scale, the step norm and which probe was kept:

   ```
     estimator 2 (elite regression)          estimator 1 (REINFORCE)
     ep  sigma   ||dg||   kept               ep  sigma   ||dg||
      0  0.1910  0.9539   candidate           0  0.1910  0.9539
      1  0.0281  0.5719   candidate           1  0.1910  0.9380
      2  0.0201  0.8411   candidate           2  0.1910  0.9185
      3  0.0201  0.7699   candidate           ...  (0.1910 at every episode)
      4  0.0206  0.0000   NULL                9  0.1910  0.8158
      5  0.0201  0.0000   NULL
      6  0.0202  0.6132   candidate
      7  0.0200  0.0000   NULL
      8  0.0200  0.6818   candidate
      9  0.0200  0.0000   NULL
   ```

   **Estimator 2 collapses.** `sigma` falls to `SIGMA_MIN = 0.02` after a single elite
   regression and never recovers in ten episodes. The policy becomes effectively deterministic:
   it proposes `mu` with almost no spread, so the `G` candidates of the next boundary are near
   copies of one another and best-of-`G` is selecting among noise. The null probe -- `dg = 0` --
   wins **four of ten** episodes, which is the estimator correctly reporting that its own
   deterministic proposal is worse than doing nothing.

   **Estimator 1 never moves.** `sigma` sits at its initialisation `0.1910` for all ten
   episodes. One REINFORCE step per boundary at `eta_2 = 1e-4` is roughly nothing over the
   twenty episodes a life contains.

   *Why the collapse is structural, and worse in production than in the test.* `_heads_fit`
   (consolidate.py:426) runs 200 Adam steps maximising `sum_i log N(dg_i ; mu(feat_i),
   sigma(feat_i))` over the accepted set, resisted only by a KL penalty at `lambda_KL = 0.1`.
   `W_mu` has `d_g x d_c` parameters against an accepted set of at most `ACCEPTED_MAX = 8`
   pairs:

   | shape | `W_mu` parameters | targets (`8 x d_g`) | over-parameterisation |
   |---|---|---|---|
   | TINY (where this was observed) | `24 x 16 = 384` | 192 | `2x` |
   | **TS production** | `512 x 256 = 131,072` | 4,096 | **`32x`** |

   An over-parameterised conditional mean fits every accepted `dg` exactly; the residuals go to
   zero; and the maximum-likelihood `sigma` of a zero residual is zero. The floor is the only
   thing stopping it. Production is sixteen times more over-parameterised than the shape where
   the collapse was observed, so it will collapse **faster and harder**, not less.

   *The first-principles reading.* This is the cross-entropy method with an elite fraction of
   `1/G`, i.e. **one elite per generation**. CEM shrinks its sampling scale from the *spread of
   the elite set*; with a single elite there is no spread, so there is no statistical basis for
   estimating a scale at all. Fitting one by maximum likelihood against an over-parameterised
   mean is not an estimate, it is an artefact. A parameter that cannot be estimated should not
   be fitted.

   *The history this closes, which is worth stating plainly.* The scale head has now failed
   three ways: **clamped** (11 §P-C.28) -- the pre-activation sat outside the clamp, whose
   gradient is zero there, so the head was dead; **clamped again** (§P-C.36) -- the same defect
   in the form written to cure it; **squashed and alive** -- and now it collapses to its floor
   within one boundary. Each fix was correct about the defect it addressed and revealed the
   next. The pattern suggests the problem was never the parameterisation: it is that `G = 4`
   with one elite cannot support a learned scale.

   *The correction to apply (prepared, not yet applied -- the factorial holds the GPU and
   `tempo/*.py` is in its code digest).* The elite regression fits the **mean only**; the
   proposal scale is held at the value 06 (6.4) specifies, `tau / sqrt(d_g)`, which makes a
   typical draw of norm `tau` -- the trust region the step is allowed to take, and therefore the
   natural exploration scale for best-of-`G`. The adaptive-scale capability is **withdrawn**,
   with this measurement as the reason, rather than left in place producing a floor value that
   looks like a decision and is not.

   *The correction, verified before it was applied.* The fix was simulated without touching the
   repository (the factorial owned the code digest): freeze `W_sigma` and `b_sigma`, filter the
   optimisers, and let `head_parameters()` return the read-out and the mean only. The same
   ten-episode trace, rerun:

   ```
     as built      sigma  0.191 0.191 0.021 0.024 0.028 0.022 0.020 0.020 0.020 0.020
                   null probe wins 6/10     mean ||dg|| 0.281
     with the fix  sigma  0.191 x 10  (= tau/sqrt(d_g), held)
                   null probe wins 2/10     mean ||dg|| 1.189
   ```

   The null-probe win rate is the behavioural measure that matters: it is the fraction of
   boundaries on which the estimator's own proposal was **worse than doing nothing**. As built
   that is `60 %`; with the scale held it is `20 %`, and the steps are `4.2x` larger, close to
   the trust region `tau = 0.98` the design intends. Exploration survives the whole life.
   The correction is therefore not a tidy-up: it is the difference between an estimator that
   proposes and one that does not.

   *What this does not settle.* Whether estimator 1's inertia is also a defect. One gradient
   step per episode is what 06 §6.5 specifies and its whole appeal is zero overhead, but a
   policy that does not move in twenty episodes cannot be said to have been tested either. R4's
   arm (f) compares the two estimators; with estimator 2 collapsing and estimator 1 static, that
   contrast currently compares two failures, and that must be said when it is reported.


9. **The genome's dimension is roughly eighteen times too large for the budget that searches
   it.** *2026-09-05, simulated on CPU while the factorial held the GPU; before R4.*

   *The question.* P-D.8 found both estimators degenerate. Fixing them raises a prior question
   that no amount of estimator repair addresses: `d_g = 512`, and estimator 2 sees
   `(G + 1) x 20 = 100` noisy evaluations in a life. Can a search of this shape find anything in
   that many dimensions with that many samples? This is answerable without a GPU, because it is
   a property of the *rule*, not of TinyStories.

   *The simulation.* The rule exactly as implemented -- `G = 4` isotropic candidates plus the
   null probe, trust-region projection at `tau = 0.2 sqrt(d_g)`, decay `0.01` toward the prior,
   the frozen `sigma` of P-D.8's correction, 20 boundaries -- against a smooth objective whose
   optimum sits `10 tau` away, so the budget could in principle reach it. Measured: the fraction
   of the achievable improvement captured, where 1.0 means moving straight at the optimum every
   boundary. 200 lives per cell.

   | `d_g` | `tau` | noiseless | with probe noise | `sqrt(2 ln G / d)` |
   |---|---|---|---|---|
   | 16 | 0.80 | 0.577 | 0.574 | 0.416 |
   | 32 | 1.13 | 0.421 | 0.413 | 0.294 |
   | 64 | 1.60 | 0.283 | 0.267 | 0.208 |
   | 128 | 2.26 | 0.176 | 0.158 | 0.147 |
   | 256 | 3.20 | 0.099 | 0.072 | 0.104 |
   | **512 (the design)** | 4.53 | **0.044** | **0.023** | 0.074 |

   The last column is the expected cosine similarity between the best of `G` isotropic
   directions and the gradient, which is the standard `sqrt(2 ln G / d)` of random search. The
   measured curve tracks it, so the simulation is behaving as the theory says and the result is
   not an artefact of the synthetic objective's shape.

   **At the design's `d_g = 512` the search captures 2.3 % of the available improvement. At
   `d_g = 32` it captures 41 %** -- eighteen times more from exactly the same number of
   evaluations.

   *What this means for K-GENE.* R4 would very likely record K-GENE as FIRING, and the finding
   would be uninterpretable: a null result cannot distinguish "a gain genome does not help" from
   "the search could not move". Combined with P-D.6 -- the expected effect (`0.014`-`0.09` nats)
   sitting at or under the `3 sigma` bar -- R4 as specified is close to a foregone conclusion,
   and running it unchanged would spend eighteen GPU-hours to learn very little.

   *The redesign, and why it costs nothing the design cares about.* `d_g = 512` is described in
   `config.py` as "scale free: the same at every shape", and that property is about `d_g` being
   *independent of the network's size*, not about it being large. `d_g = 32` is equally scale
   free. The prior evidence for how much genomic variety is actually used points the same way:
   the ANLAGE lineage found the useful variation concentrated in a learning-rate axis of four
   numbers and roughly one scale-profile direction. A 512-dimensional genome is parameterising a
   space whose useful subspace appears to be one or two orders of magnitude smaller, and paying
   for it in search efficiency.

   *Registered before R4:* `d_g` is swept over `{32, 128, 512}` on the genome-versus-`gamma = 0`
   contrast before the full eight-arm R4 is run. If the `d_g = 32` arm moves the score and the
   `d_g = 512` arm does not, the design's failure was the search budget and not the idea, and
   K-GENE must be re-registered at the corrected dimension. If neither moves, that is a much
   stronger negative result about the gain genome than the current design could have produced.

   *Limits.* The simulation models the **search** component of the rule, not its **learned
   policy** component: the real consolidator can in principle propose better-than-random
   directions from the episode summary. But it has 20 (summary, proposal, reward) tuples in a
   life to learn that from, so the random-search bound is the right first approximation, and
   nothing in the measurement of P-D.8 suggests the policy is learning to beat it. The objective
   is quadratic and isotropic; a real reward landscape with a low-dimensional active subspace
   would be kinder, which is a further argument for making `d_g` match that subspace rather than
   the number 512.


10. **An adopted probe's births vanish from the record, so the growth trajectory is
    unauditable.** *2026-09-05, found while checking that the genome can switch a block on
    at all.*

    *What was measured.* A twelve-episode TINY life under estimator 2 with `morphology =
    "genome"` ended with **fourteen `born` flags set and `record["births"] == []`**. The
    morphology had grown and the record said nothing had happened.

    *The mechanism.* Under estimator 2 the morphology almost always changes inside a **probe**
    episode, because a probe applies a candidate genome and that moves the morph gains. The
    boundary then goes: `checkpoint_state` deep-copies the record into the snapshot; the probes
    run and append their births to the live record; `load_state(snapshot)` restores the record,
    correctly discarding the probes' births; and then, if a probe is kept,
    `_adopt_probe_state` calls `load_state(probe_state)` -- which makes the probe's record live
    again -- and immediately overwrites it with the pre-probe record. The adopted probe's
    **weights and `born` flags are kept while its birth entries are thrown away.**

    *What this does and does not break.* K-MORPH compares arms by their scores, not by the birth
    list, so no verdict is corrupted. What is lost is the ability to say *when* the agent grew
    and *what each new block copied* -- and 05 §5.3 makes growth-by-copy a mechanism with a
    theorem attached, so being unable to show it happened is not a cosmetic gap in a project
    whose standard is that every claim is traceable to an artifact.

    *A second observation from the same trace, not a defect.* `SwitchPattern.groups_on()`
    reports a head or slice group as on whenever its multiplier is positive, **without checking
    whether its layer is active**. At TINY the primordium layers are `{0, 1, 4, 5}`, so `F2` and
    `H3` read as "on" while their quarter-stack is off and they contribute nothing. That is a
    loose *reporter*, not a bug in the switching: `apply_births` checks
    `new.head_m[l] > 0 and new.layer_active(l)` and correctly declines to birth them. It is
    recorded because it cost an hour of misreading -- the first conclusion drawn from this trace
    was "growth-by-copy never fires", which was wrong.

    *Correction prepared with P-D.8's, to apply at the same boundary:* the births an adopted
    probe performed are carried into the kept record and tagged `adopted_from_probe`, so the
    record's birth list once again describes the weights the agent is actually carrying.


11. **The two estimators do not receive the same quality of signal, and the one the
    specification calls "the production rule" receives the worse one.** *2026-09-05, from the
    code and the traces of P-D.8.*

    The two rewards are built differently, and the difference is not a detail:

    ```
      estimator 1 (life.py:1064)   r_e = reward(L_check_prev, L_now,  F_episode, F_ref)
                                         ^^^^^^^^^^^^^^^^^^  ACROSS episodes
      estimator 2 (life.py:1273)   r_i = reward(L_now,        L_probe, F_episode, F_ref * n_probe/N)
                                         ^^^^^^                WITHIN one boundary
    ```

    Estimator 2's `G + 1` probes all start from **one checkpoint** and read the same windows
    under common random numbers, so the ordinary training progress they share cancels when they
    are compared, and the null probe (`dg = 0`) is an explicit control variate for whatever
    remains. Its selection signal is the genome's effect and nothing else, up to probe noise.

    Estimator 1's reward is the improvement from one episode to the next. That quantity is
    dominated by ordinary training progress -- the model gets better every episode whether or
    not the genome does anything -- and the genome's contribution rides on top of it. The EMA
    baseline removes the systematic part but not the episode-to-episode variation. In the
    ten-episode traces of P-D.8 the rewards were `+0.674, +0.037, +0.057, +0.039, -0.056,
    +0.143, +0.038, +0.024, -0.018`: a spread of about `0.06` in `L_check` units around a small
    mean, against a genome effect that the inherited record (P-D.6) puts one to two orders of
    magnitude lower. REINFORCE averages that noise down only as `sqrt(n)`, and `n` is 20.

    So the design gives its **zero-overhead** estimator an **uncontrolled** signal and its
    **50 %-overhead** estimator a **controlled** one. 06 §6.5 nominates estimator 1 as the
    production rule and estimator 2 as "the screening rule", which is exactly backwards with
    respect to signal quality: the cheap rule is cheap because it does not run the controls that
    make the measurement mean anything.

    *What follows.* Estimator 1's inertia in P-D.8 -- `sigma` unmoved over ten episodes -- is
    consistent with a policy receiving almost no usable gradient, and is not separately
    diagnosable from "one Adam step per boundary at `1e-4` is small". Both are true and both
    point the same way. R4's arm (f) compares the two estimators; that contrast should be read
    as comparing two *signal designs*, not two optimisers, and the production/screening
    nomenclature of 06 §6.5 should be revisited on the evidence rather than carried forward.

    *Not a correction, and deliberately so.* Nothing here is changed. Estimator 1 is what the
    specification registered as the production rule, and swapping it now on an argument rather
    than a measurement would be exactly the goalpost-moving this ledger exists to prevent. What
    is registered is that the asymmetry is structural and known in advance, so that when R4
    reports arm (f) the reading is already fixed.


12. **`t_w` now survives a re-fetch (the correction prepared for the same boundary).**
    *2026-09-05; the defect itself is measured in the commit that added
    `TestTimeCodeSurvivesARefetch`.*

    02 (2.13) is [NORMATIVE] that the time code is a Fourier transform of the event's absolute
    wall clock, and `Perception.forward` adds it to every span. Every `fetch` hardcoded
    `t0 = 0.0`. Measured on the text stream: a window taken live at `t_w = 14.0 s` and re-fetched
    by its identity returns identical content at `t_w = 0.0`, and the two time-code feature
    vectors have cosine similarity `0.44` (`0.34` at 10 s). Within one episode `t_w` reaches
    hundreds of seconds, where they share no phase.

    Two things were corrupted. 06 (6.11)'s salience target `Δℓ_w = ℓ_w^{(e)} − ℓ_w^{(e+1,end)}`
    is the *model's* improvement only if the window is identical at both scorings; it was not,
    the first being live at the true clock and the second re-scored from a fetch at zero. And
    rehearsal stores only identities, so the 5 % of every episode spent replaying presented its
    windows at `t ≈ 0` rather than their own time.

    The correction gives `fetch` a `t0`, and makes the callers that must reproduce a live window
    carry it: `WindowSummaries` records `t0` beside each identity, `Rehearsal` returns
    `(ident, t0)` from `draw`, and `Life.probe_t0` holds the probe set's clocks beside its
    losses (checkpointed, and restored on the adopt path too). The fixed evaluation and fitness
    sets keep `t0 = 0` by default and stay self-consistent -- they have no live counterpart to
    match. `TestTimeCodeSurvivesARefetch` stops being an `xfail` and becomes an assertion, and
    `_same_window` compares `t_w` again.

13. **The genome's *static* leverage is below seed noise, so its case rests entirely on the
    dynamics.** *2026-09-05, TINY, CPU.*

    Every question so far has been whether the rule can *find* a good genome. This is the prior
    one: is there a good genome to find? The gains are bounded by construction (`c_rho`: matrix
    `0.5`, norm `0.2`, offset `2.0`), so the genome reaches a bounded neighbourhood of the
    trained network. Sampling genomes uniformly on spheres at fractions of the norm bound,
    expressing each, and scoring **the same batch** under each:

    | `\|g − g₀\|` | n | mean | sd | min | max | best beat `γ=0` by |
    |---|---|---|---|---|---|---|
    | 3.67 (¼ bound) | 24 | 4.21890 | 0.00040 | 4.21785 | 4.21928 | 0.00122 |
    | 7.35 (½) | 24 | 4.21865 | 0.00060 | 4.21756 | 4.21922 | 0.00150 |
    | 14.70 (full) | 24 | 4.21866 | 0.00061 | 4.21727 | 4.21945 | 0.00180 |

    A random genome at the **full** norm bound moves the loss by a standard deviation of
    `0.0006` nats -- against `sigma_e` of `0.0037` (`K = 4`) to `0.0108` (`K = 1`). The genome
    cannot reshape what the network computes by anything approaching the noise floor.

    *What this does and does not say.* It is a **static** measure, and the design's claim is
    dynamic: the genome is supposed to change how the network *learns* over an episode, not what
    it computes at an instant. The learning-rate, morphology and mixture roles act on the
    optimiser, the switch pattern and the data draw, none of which `window_loss` touches -- so
    they contribute **nothing** to the numbers above by construction. What is bounded here is
    the direct effect of the `243,088` differentiable gains (matrix, norm, offset), which is
    `99.98 %` of the genome's output.

    This corroborates the record the project inherited rather than contradicting it: 00 §0.1
    reports the inherited scale profile as *worse at the instant of birth and better by `0.014`
    nats after 2.5M tokens*. Static leverage near zero with a dynamic gain an order of magnitude
    larger is exactly that pattern, measured again from the other side.

    *The consequence for reading K-GENE.* The genome's case rests on `45` programme gains (4
    learning-rate, 34 morphology, 7 mixture) and on the differentiable gains' effect on the
    *trajectory* rather than the function. A K-GENE null would therefore not be evidence that
    "gains cannot help"; it would be evidence about a specific and rather small set of dynamic
    levers. Registered so the null, if it comes, is not over-read.

    *A caveat on the last column.* "Best beat `γ=0` by" is the best of 24 **random** draws in a
    512-dimensional space, which is a weak optimiser -- it is a lower bound on what a
    well-chosen genome could achieve, not an upper bound. The `sd` is the number to trust, and
    it is the one that matters for the search: it is the scale of the differences the
    best-of-`G` selection has to resolve against probe noise.

    *An error worth recording.* The first version of this measurement built the life with
    `genome=False` and reported `sd = 0.00000` at every radius -- an apparently decisive "the
    genome does nothing". `Life.gains_dict()` returns `None` unless `genome` or `free_gains` is
    set, so `window_loss` was not consulting the gains at all. The harness, not the mechanism,
    was flat.


14. **A suspicion tested and NOT supported: the roles are not badly entangled.**
    *2026-09-05. Recorded because it was a plausible mechanism for the genome's weakness and it
    turned out to be wrong.*

    *The suspicion.* The roles share one map (`u = B tanh(A g)` with a single `A` and rank
    `r = 32`), and the inherited record puts the leverage in the learning-rate axis -- 4 gains
    of 243,133. If a step in `g` cannot reach those 4 without dragging the other 243,088, the
    search is fighting the parameterisation and no estimator repair would help.

    *The measurement.* A step of norm `tau = 4.53` in `g` at the TS shape, 32 random draws,
    decomposed by role kind:

    | kind | gains | share of `\|dgamma\|^2` | per-gain rms |
    |---|---|---|---|
    | matrix | 209,600 | 82.5 % | 0.023 |
    | norm | 33,280 | 2.0 % | 0.009 |
    | offset | 208 | 1.3 % | 0.090 |
    | lr | 4 | 0.053 % | 0.123 |
    | morph | 34 | 13.6 % | 0.733 |
    | mixture | 7 | 0.57 % | 0.316 |

    *Why the suspicion fails.* The `0.053 %` share looks damning and is an artefact of counting
    squared change over 209,600 matrix gains against 4 learning-rate gains. The number that
    matters is the **per-gain** movement, and there the learning-rate role moves `0.123` in
    log-space per boundary -- a `13 %` change in the multiplier -- while the matrix gains move
    `2.3 %` each. Morphology moves most of all (`0.733`, its range being `c = 8`), which matches
    the separate observation that groups switch on within the first few boundaries. A `tau`-step
    moves every role at a rate that is, if anything, generous.

    So the parameterisation is not the problem. The coupling that does exist is also cheap:
    P-D.13 measured the matrix gains' static leverage at `sd = 0.0006` nats, so a step that
    improves the learning-rate role while perturbing 209,600 matrix gains is not paying much for
    the perturbation.

    *What this leaves standing.* The genome's difficulties are the ones already measured -- the
    search dimension against the evaluation budget (P-D.9), the estimators' degeneracy (P-D.8),
    and the static leverage floor (P-D.13) -- and not the shape of the expression map. Recorded
    so that the map is not "fixed" later on the strength of an argument that was checked and did
    not hold.


15. **An operational error: two GPU jobs ran concurrently for about fifty seconds.**
    *2026-09-05 05:58:25-05:59:15. Recorded because the standing constraint of this project is
    one serial chain per card, and because `runs/tempo/factorial.log` contains interleaved lines
    that a later reader would otherwise have to explain.*

    *What happened.* The factorial's cells were reordered mid-run so that the two cells with
    different predictions (`tsk1`, `tsk1wide`) would run first. The reordering was done by a
    helper that waited for the current cell to finish, set the driver's `PAUSE` sentinel, waited
    for the running child to exit, then relaunched with the new order. Its wait was

    ```
      until ! ps -W | grep -q "tempo_life\|tempo_cont\|tempo_codec"; do sleep 20; done
    ```

    and on this shell `ps -W` prints only the executable path -- `...\python.exe` -- never the
    arguments. The pattern therefore **could not match**, the wait returned immediately, and the
    new driver started `codec-tsk1wide-s0` while the old one was still training
    `codec-tsk1dx512-s0`. Two codec jobs shared the card for about fifty seconds until the old
    driver and its child were killed by PID.

    *What it cost.* Nothing measurable. Both processes were codec training, which is
    deterministic given its seed, so contention changes speed and not numbers; no run was
    completed or recorded during the window; the partially trained `codec-tsk1dx512-s0`
    directory contained a log and no artifacts and was removed, so that cell will rebuild
    cleanly when it is reached. The surviving `codec-tsk1wide-s0` was unaffected beyond being
    slowed.

    *The lesson, which is not about this script.* A wait condition that can never be true is
    indistinguishable from a wait condition that is satisfied. The check asserted the absence of
    something it had no ability to observe, and absence-of-evidence read as evidence-of-absence.
    The same shape appears twice elsewhere in this project's history: a `clamp` whose gradient is
    zero outside its bounds (P-C.28, P-C.36) and a `groups_on()` that reports a group as active
    without checking whether its layer is (P-D.10). Any guard should be tested by making it
    *fire* once, and none of these three ever was.


16. **The analyser cannot see any configuration outside a hardcoded list, and skips them
    silently.** *2026-09-05, found by checking whether the factorial would be readable before
    it finished.*

    `NAME_RE` and the codec pattern in `scripts/tempo_analyse.py` fix the configuration to
    `ts|tsk2|tsk1|pile`. Every cell of the rank factorial except `tsk1` -- and every `-dg<n>`
    run of the genome probe -- fails to match, and `load_runs` **`continue`s on a non-match**, so
    those runs do not appear as incomplete; they do not appear at all. The verdict layer would
    then report NOT EVALUATED: the right output for entirely the wrong reason, which is worse
    than an error, because an error would have been noticed.

    This is the same failure shape as P-D.15's wait condition and P-C.28/36's clamp: a check
    whose negative result is indistinguishable from the thing not existing. The factorial's
    numbers were never at risk -- they are read from the records directly -- but its *verdict*
    would have been silently absent.

    *The correction (prepared for the same boundary).* The run-name grammar is
    `{run}-{arm}[-{tag}]-{config}-s{seed}`, and a configuration may now contain hyphens
    (`tsk1wide-dg32`), which is ambiguous against the optional tag. It is disambiguated by
    enumerating the bases longest-first -- so `tsk1wide` is not matched as `tsk1` with a stray
    tail -- and giving the modifiers a shape nothing else has (`dg32`, `dx512`). Verified on 18
    names in isolation before being applied: every real run form parses to the right
    `(run, arm, config, seed)`, and the retired `-v1` / `-v2` directories and non-runs are still
    rejected.


17. **K-CONT compares a deliberately thin shape against a conventionally proportioned one, so
    it does not isolate the span interface.** *2026-09-05, found while trying to explain why
    doubling the interface's degrees of freedom closed only about a quarter of the gap.*

    *What the comparison actually is.* R1's arm `nolat` sets `morphology = "primordium"`
    (`tempo_life.py` ARMS), so every K-CONT span run is the born-small shape. Its active
    proportions, against the FLOP-matched token baseline's:

    | | `d` | `d_ff / d` | `(n_heads x d_head) / d` | active params | tokens |
    |---|---|---|---|---|---|
    | span agent (primordium) | 512 | **2.00** | **0.50** | 17,886,264 | 24,985,600 |
    | token baseline | 384 | 4.00 | 1.00 | 24,965,505 | 21,598,208 |

    A transformer is conventionally `d_ff = 4d` with `n_heads x d_head = d`. The primordium is
    **half of both**, at a wider residual -- a thin model wearing a wide stream, because half its
    heads and half its FFN slices are switched off and waiting to be grown into. The FLOP
    matching accounts for the compute that shape consumes. It does not ask whether the shape is
    a good way to spend it.

    *Why this matters more than it first looks.* The span agent is **wider** (512 vs 384) and
    sees **more tokens** (25.0M vs 21.6M) and still loses by `0.099` nats. That rules out the
    easy explanations -- it is not a smaller model and not a shorter run. What is left is where
    the FLOPs go *inside* the model, and one of the things inside is a shape with half the
    conventional FFN and attention width.

    So the K-CONT gap conflates two costs: the span interface, which is what the criterion is
    named for, and the primordium's proportions, which belong to the **growth** mechanism and
    have nothing to do with `K`. Nothing in 10 §10.2 separates them, and the criterion has been
    reported four times (`K = 4, 2, 1`, and the factorial's baseline) without the separation
    being noticed.

    *What is and is not damaged.* The rank factorial is unaffected as a *contrast*: all four
    cells run the same primordium, so the degrees-of-freedom effect is measured cleanly against
    a common shape. What is damaged is the reading of the **absolute** gap -- and therefore the
    conclusion in P-D.2 that a positive `K = 1` gap "implicates the `d_x` interface". It may
    implicate the primordium instead, or both.

    *The diagnostic, registered before it runs.* Repeat K-CONT at `K = 1` with the **adult**
    shape (`morphology = "flat"`: all 16 layers, all heads, full FFN, so `d_ff/d = 4` and
    `n_heads x d_head = d`), against its own FLOP-matched baseline. Three outcomes:
    * the gap largely closes -- K-CONT was measuring the primordium's proportions, and the span
      interface is much cheaper than four reported criteria have claimed;
    * the gap persists -- the interface is the cost after all, and the primordium is exonerated;
    * the gap grows -- the primordium was *flattering* the span agent, which would be the most
      surprising outcome and the most informative.

    Whatever it shows, K-CONT's four existing verdicts must be re-reported with the confound
    named, because a criterion that cannot separate the mechanism it is named for from an
    unrelated one has not tested that mechanism.


18. **The softmax-bottleneck hypothesis is REFUTED: degrees of freedom do not explain the
    `K = 1` interface gap.** *2026-09-05. The prediction was registered in §P-D.4 before any
    cell ran; this is its result.*

    | cell | `d_x` | `d_c` | dof | span | token | gap | vs baseline |
    |---|---|---|---|---|---|---|---|
    | `tsk1` | 256 | 256 | 255 | 1.8920 | 1.7930 | **+0.0989** | -- |
    | `tsk1wide` | 512 | 512 | 511 | 1.8696 | 1.7749 | **+0.0947** | **−0.0042** |

    Per seed the paired differences are `−0.0110` and `+0.0025` -- **opposite signs** -- with mean
    `−0.0042` and sd `0.0095`. At two seeds that is indistinguishable from zero. Doubling the
    logit family's degrees of freedom moves the gap by about `4 %`, against a prediction that it
    would close.

    *What went wrong with the argument, precisely.* The dof bound is real and was measured
    exactly where the algebra put it (§P-D.7: a cliff at `min(d_x, d_c)`, and `tsk1dx512` buying
    262,144 parameters and zero freedom). The error was assuming a binding constraint on the
    *map* is a binding constraint on the *loss*. The span agent did improve, by `0.0224` nats --
    but a wider interface costs more compute, the baseline is FLOP-matched and is therefore paid
    that compute back (24.33M tokens against 21.60M), and it improved by almost as much. I had
    accounted for this and underestimated it by half: the predicted baseline was `1.786` and the
    measured one `1.7749`.

    A model can have far more expressive freedom than it uses. 255 degrees of freedom are
    evidently enough for the next-token distributions this corpus and this budget produce, and
    the `d_x = 256` interface was never the binding constraint on quality -- only on the
    geometry.

    *The two control cells were dropped, and why.* `tsk1dx512` and `tsk1dc512` exist to
    attribute the effect -- rank, or merely more parameters. With the effect at `−0.0042 ± 0.0095`
    they can only attribute something already indistinguishable from zero, and would cost 90
    minutes of a shared card. That is a deviation from the design registered in §P-D.4 and is
    recorded as one rather than quietly absorbed: the *decisive* cell ran, the prediction failed,
    and the controls are only worth running if the effect ever becomes resolvable. The codecs and
    configurations for them remain in place.

    *What this leaves.* The `K = 1` gap of `+0.0989` nats is `27 sigma_e` and now has no
    accepted explanation. §P-D.17's confound is the leading candidate and the only one with an
    effect of the right size available to it: every K-CONT span run has been the born-small
    shape at `d_ff/d = 2.0` and `(n_heads x d_head)/d = 0.5`, half the conventional proportions
    and half the FLOP-matched baseline's. The adult-shape diagnostic is next, and it now carries
    the whole question rather than being a supplementary check.


19. **CORRECTED. The deficit is invariant to shape: the primordium is exonerated and the gap
    neither grows nor closes.** *2026-09-05. The diagnostic and its three readings were
    registered in §P-D.17 before it ran; this is the **second** of them.*

    **This entry first reported the third reading -- "the gap grew by `+0.0352`, the primordium
    was flattering the span agent" -- and that was wrong.** It rested on a baseline whose width
    was clipped at the end of `CANDIDATE_WIDTHS`, a confound the entry itself identified, named
    as working *against* its own conclusion, and then let stand as a headline for twenty minutes
    while the check ran. The check overturned it. The original text is kept below the correction
    because a conclusion that was published and withdrawn is part of the record, not an
    embarrassment to be tidied away.

    | shape | baseline | width | tokens | ratio | span | token | gap |
    |---|---|---|---|---|---|---|---|
    | primordium | free | 384 | 21.60M | 1.150 | 1.8920 | 1.7930 | **+0.0989** |
    | adult | **clipped** | 640 | 30.26M | 0.820 | 1.8264 | 1.6923 | +0.1342 |
    | adult | **free** | 704 | 25.35M | 0.979 | 1.8264 | 1.7230 | **+0.1034** |

    The clip was worth **`+0.0307` nats** to the baseline -- nearly the whole of the `+0.0352`
    "growth" -- because at `1.112e16` FLOPs the compute-optimal allocation is roughly
    `N ~ D ~ 43M`, and width 640 with 30.3M tokens sits closer to it than width 704 with 25.4M.
    Against a *free* baseline the adult gap is `+0.1034` against the primordium's `+0.0989`: a
    change of `+0.0045`, comfortably inside the seed noise of either arm.

    *The finding, corrected.* The `K = 1` deficit is **shape-invariant**: about `0.10` nats
    across `d_ff/d` of 2.0 and 4.0, attention width of 0.50 and 1.00, 17.9M and 70.2M active
    parameters, and a `3.6x` range of compute. It neither closes when the shape is made
    conventional nor grows when the model does.

    *Why this is better news than what was first written, and still not good news.* A deficit
    that **grew with scale** would have pointed the opposite way from B11 and made the
    architecture's prospects worse the larger it got. A **constant** `0.10` nats is a tax --
    payable, in principle, for a single continuous interface that every modality shares, if that
    interface ever buys something the tax is worth. What it is not is free, and nothing measured
    so far shows what it buys.

    *What survives of the elimination.* Three candidate explanations were proposed and all three
    remain eliminated, only the third for a different reason than first written:
    * the **span idea** -- at `K = 1` there is no shortening to pay for;
    * the interface's **degrees of freedom** -- §P-D.18, refuted at 4 %, seeds disagreeing in sign;
    * the **born-small shape** -- exonerated here, because growing it up changes nothing.

    *What the confound cost, and what it bought.* `CANDIDATE_WIDTHS` now runs to 1024 and
    `result.json` records `width_at_boundary`, so a clipped selection is visible instead of
    inferred. The episode is also a measurement of something worth knowing: a clipped baseline
    was worth `0.031` nats here, which is three times `K = 1`'s `3 sigma_e` bar. Any earlier
    verdict whose baseline sat at a boundary would have been affected the same way -- none did
    (the `ts`, `tsk2` and primordium `tsk1` baselines all chose interior widths), which is now
    checkable rather than assumed.

    *Remaining untested candidates* (unchanged): the two extra linear stages between the backbone
    and the softmax; the decoder's **warm start**, fitted by R0 to the encoder's `x_bar` and then
    asked to decode the backbone's `y_hat`; and the baseline's tied and value embeddings. The
    warm start is being tested next -- it is the cheapest, needs no new baseline, and is the only
    one that would be an own goal rather than an inherent cost.

    ---

    *The superseded text, as published:*

    | shape | `d_ff/d` | attn`/d` | `F_episode` | span | token | gap |
    |---|---|---|---|---|---|---|
    | primordium (K-CONT as run) | 2.0 | 0.50 | 3.076e15 | 1.8920 | 1.7930 | **+0.0989** |
    | adult (conventional) | 4.0 | 1.00 | 1.112e16 | 1.8264 | 1.6923 | **+0.1342** |

    **The gap grew by `+0.0352` nats, about 36 %.** Growing the shape up -- all 16 layers, all
    heads, the full FFN, `3.6x` the compute -- makes the span agent *relatively worse*, not
    better. The born-small proportions were helping it.

    *What this eliminates.* Three explanations for the `K = 1` deficit have now been proposed and
    all three are gone:
    * the **span idea** -- at `K = 1` there is no shortening to pay for, so the gap is not that;
    * the interface's **degrees of freedom** -- §P-D.18, my own registered hypothesis, refuted at
      4 % with the seeds disagreeing in sign;
    * the **born-small shape** -- this entry, refuted with the sign reversed.

    *Why a growing gap is worse news than a fixed one.* A constant `0.1` nats would be a tax one
    could choose to pay for a single continuous multimodal interface. A deficit that grows with
    scale -- `+0.0989` at 17.9M active parameters, `+0.1342` at 70.2M -- is a statement about how
    the architecture *scales*, and it points the opposite way from B11. Two points do not make a
    trend line and this must not be read as one; what is measured is a direction, on two shapes,
    at two seeds each.

    *The confound, and the correction to my own reasoning.* The adult baseline chose width 640,
    the **last element** of `CANDIDATE_WIDTHS`, at a per-token ratio of `0.820` -- a clipped
    choice, not a free one. I first wrote that the clip "flatters the span agent". **That is
    backwards.** At `1.112e16` FLOPs the compute-optimal allocation is roughly `N ~ D ~ 43M`;
    width 640 gives ~44M parameters and 30.3M tokens, so the baseline was already
    over-parameterised and under-trained, and a wider uncapped baseline is *further* from
    optimal and therefore worse. The clip **helped the baseline**, so it works *against* this
    entry's conclusion. A confound that opposes one's own conclusion still has to be measured:
    the candidate list now runs to 1024, `result.json` records `width_at_boundary`, and the
    uncapped baseline (width 704, ratio 0.979, 25.3M tokens -- predicted before it ran) is being
    measured beside the capped one so the clip's worth is a number rather than an argument.

    *Where this leaves the project's central efficiency claim.* Across `K = 4, 2, 1` and both
    shapes, at matched training FLOPs, the span agent is behind a token baseline by `0.09` to
    `0.55` nats, and no proposed mechanism survives as the cause. The remaining untested
    candidates are recorded so the search does not start over: the two extra linear stages
    between the backbone and the softmax; the decoder's **warm start**, which R0 fits to the
    *encoder's* latents `x_bar` and which is then asked to decode the *backbone's* predictions
    `y_hat` -- a warm start on a different input distribution, where a token model's unembed
    simply trains from scratch; and the baseline's tied and value embeddings, which the span
    agent has neither of. The decoder's warm start is the cheapest to test and the only one that
    would be an own goal rather than an inherent cost.


20. **The decoder's warm start is exonerated too -- it HELPS.** *2026-09-05. The fourth
    candidate explanation for the `K = 1` deficit, and the fourth eliminated.*

    R0 fits `D_txt` to decode `x_bar`, the frozen encoder's latent on the unit-RMS sphere; the
    life then hands the same weights `y_hat`, the backbone's prediction. That is a warm start on
    a different input distribution and it puts a prior on the geometry the backbone must produce,
    where a token model's unembed starts from noise and co-adapts freely. It was the cheapest
    remaining candidate and the only one that would have been an **own goal** rather than an
    inherent cost, so it was tested first.

    | decoder | span | gap vs the same baseline |
    |---|---|---|
    | warm-started from R0 | 1.8920 | **+0.0989** |
    | freshly initialised | 1.8988 | **+0.1058** |

    Paired per seed the difference is `−0.0085` and `−0.0052`, agreeing in sign. **The warm start
    is worth `0.0068` nats of benefit**, not a cost. Removing it makes the agent slightly worse,
    which is the opposite of the hypothesis -- the second time tonight a sign came out reversed
    (§P-D.19 was the first).

    The diagnostic needed no new baseline: `--fresh-decoder` changes no shape and no FLOP, so the
    existing primordium baselines applied unmodified and the whole test was two span runs of
    eleven minutes.

    *The tally.* Four explanations proposed for the `K = 1` deficit and four eliminated:
    the **span idea** (no shortening to pay for at `K = 1`), the interface's **degrees of
    freedom** (§P-D.18, refuted at 4 %), the **born-small shape** (§P-D.19, invariant), and the
    decoder's **warm start** (this entry, beneficial). The deficit is `~0.10` nats and stands
    unexplained.

    *The two candidates left, and why the next one is worth running.* The span agent's head is
    `h -> W_out -> y_hat -> W_c -> c -> LN -> W_v`, two linear stages more than a token model's
    single unembed. And the baseline has two features the span agent simply does not: **tied
    input/output embeddings** and **value embeddings** (`Phenotype(..., tied=True)`,
    `StageConfig(use_ve=True)`). The second is the more useful test because it is *actionable*:
    running the baseline without them measures how much of the gap is "the span architecture is
    worse" against "the baseline is better equipped", and if they account for it the remedy is to
    give the span agent the same features rather than to abandon the interface. It is also cheap
    -- two baseline runs at the primordium budget.


21. **A quarter of the deficit is the baseline's two extra features, and it is recoverable.**
    *2026-09-05. The fifth candidate, and the first to account for anything.*

    The baseline is `Phenotype(..., tied=True)` with `StageConfig(use_ve=True)`: tied
    input/output embeddings and value embeddings. The span agent has neither -- its `TextIn`
    embedding and the codec's `W_v` are separate matrices and there is no value-embedding path.
    Neither feature changes counted matmul FLOPs (a value embedding is a gather; tying is
    parameter sharing), so the stripped baseline runs at **identical width, identical token
    budget and identical compute** -- the cleanest comparison in this ledger.

    | baseline | params | test | gap |
    |---|---|---|---|
    | with tied + value embeddings | 24,965,505 | 1.7930 | **+0.0989** |
    | without them | 26,014,081 | 1.8175 | **+0.0744** |

    **The two features are worth `+0.0245` nats, a quarter of the gap.** Note the stripped
    baseline has *more* parameters -- untying adds a separate output embedding -- and is still
    worse, so this is an efficiency gain rather than a capacity one.

    *The tally, updated.* Of the `+0.0989` nats at `K = 1`:
    * `0.0245` (25 %) is **the baseline carrying two features the span agent was never given**;
    * `0.0744` (75 %) is structural and unexplained.

    Four earlier candidates account for none of it: the span idea (no shortening at `K = 1`),
    the interface's degrees of freedom (§P-D.18), the born-small shape (§P-D.19), and the
    decoder's warm start (§P-D.20, which *helps*).

    *Why this piece is different from the others.* It is **actionable**. Tied embeddings and
    value embeddings are standard, cost nothing in FLOPs, and nothing about the design forbids
    them: ~~`TextIn.T_emb` and `SpanCodec.W_v` are both `vocab x d`-shaped and could share weights
    through the interface~~ **-- corrected in §P-D.32: they are `(8192, 128)` and `(8192, 256)`,
    cannot share weights as they stand, and tying them requires `d_e = d_c = 256` and a new R0.
    The remedy is a design change, not a flag, and calling it cheap here was wrong** -- and a
    value-embedding path is a backbone change independent of the codec. Every previous finding narrowed what the deficit *is not*; this one names something
    that can be built. It is not proposed as a change here -- 02 §2.2 makes the separation of the
    input and output text paths a design invariant (I3), and inverting an invariant needs its own
    registered argument, not a corridor decision at the end of a long night.

    *What remains.* One candidate: the two extra linear stages between the backbone and the
    softmax (`h -> W_out -> y_hat -> W_c -> c`, against a token model's single unembed). If it
    does not account for the remaining `0.0744`, the deficit has no identified cause and that
    should be stated as the result rather than pursued indefinitely -- five candidates is already
    a thorough search, and "unexplained" is an honest finding when it is the true one.



22. **A three-and-a-quarter-hour stall, a fix that was right, and a diagnosis that was wrong.**
    *2026-09-05. An operational finding and a correction to my own reasoning, in that order.*

    At 10:29 the genome probe stopped making progress inside probe 3 of episode 0. The probes
    either side of it had taken 17 and 11 seconds; this one took **3 hours 22 minutes**, at
    12.87 GB of a 16.38 GB card. One episode of twenty had consumed 3.4 hours, which puts the
    life at roughly 68 hours and the four-run cell at eleven days. The run was stopped.

    *The mechanism I found, which is real.* `_best_of_G` holds three `checkpoint_state()` dicts
    alive at once -- `snapshot`, `null_state`, `best_state` -- and every tensor in each was a
    `copy.deepcopy` of a **CUDA** tensor. Measured at the TS shape: `~710 MB` per checkpoint,
    `~2.1 GB` of copies sitting beside the live model, its optimiser state and its activations.
    Probe 3 is the point at which the third of them becomes live. The fix is to keep the copies
    in host memory (`Life._to_host`): an fp32 round trip through the host is exact, so it is
    bit-neutral, and the cost is a PCIe transfer per capture against a boundary that already
    spends `(G + 1) * n_probe = 125` training steps. Committed as `099fa3d`.

    *The diagnosis, twice.* At 12:0x I wrote that the stall was **GPU contention** -- World of
    Warcraft on the same card -- and that the checkpoint copies, though real, "were not the
    cause"; the commit message `099fa3d` says so in those words. **Three hours later the
    relaunched run refuted that**, and the refutation is in this project's own log
    (`runs/tempo/genome.log`), one line above the other:

    | run | peak | micro-batch | checkpoints | WoW | episode 0 |
    |---|---|---|---|---|---|
    | 10:29 | 12.87 GB | 16 | CUDA-resident | **running** | **12,315 s** |
    | 14:56 | 6.27 GB | 8 | host memory | **running** | **411 s** |

    World of Warcraft is running in *both* rows. A 30x speed-up with the contention unchanged
    says the contention was not sufficient, and the variable that did change is **peak memory**:
    a 16.38 GB card, `~5-6 GB` held by the game, and a job asking for 12.87 GB. That is
    oversubscription, and on Windows an allocation that cannot be served does not fail -- WDDM
    pages into system memory, which is exactly how one probe takes three hours while the probes
    on either side take eleven seconds.

    *So the correct account is the third one, not the first or the second.* The cause was
    **memory oversubscription on a shared card**. The game was a *precondition* -- it consumed
    the headroom -- and the CUDA-resident checkpoints were the largest single contributor under
    my control. The first diagnosis was right about the mechanism and could not have known the
    magnitude; the second over-corrected, discarding a correct cause because I had found a
    contributing one. Withdrawing "the fix was not the cause" is the fourth correction in this
    section, and the pattern in all four is the same: I published the moment the evidence
    pointed somewhere, instead of the moment it pointed *only* there.

    *What is still confounded, and how to settle it.* The relaunch changed **two** things at
    once -- the checkpoints moved to the host **and** the micro-batch went 16 -> 8 -- and both
    act on the same variable. The 30x is therefore attributable to "peak memory fell from 12.87
    to 6.27 GB" and not, on this evidence, to either change alone. One episode at micro-batch
    16 *with* the host-memory checkpoints separates them (`~7` minutes of card): fast means the
    checkpoints were the binding constraint, slow means micro-batch 16 oversubscribes on its
    own. Registered here as a follow-up rather than run now, because the card is committed to
    the genome probe and the answer changes an operating parameter, not a result.

    *A second error inside the fix, caught before it was applied.* The patch's `load_state`
    restored the lanes' fast states with a plain `.clone()`. That was harmless while the
    checkpoint was CUDA-resident and becomes a device mismatch the moment it is not:
    `load_state_dict` casts to the parameters' device for the modules and the optimisers, and
    this was the one restore path that did not. It is now `.to(self.device)`. The near-miss is
    the generic hazard of moving state across devices: the *capture* side is where you think,
    and the *restore* side is where the bug lands.

    *What it changes operationally.* The card is shared with a game by design -- that was agreed
    at the outset, and slower runs are acceptable -- but "acceptable" now has a number attached:
    the agent's peak must leave the game its `~6 GB`, or the run does not slow down gracefully,
    it falls off a cliff. The standing rule is therefore that a genome life runs at
    `--micro-batch 8` rather than 16 while the card is shared, and that `peak_mem_gb` is read
    at every episode boundary rather than at the end. Gradient accumulation
    here is exact (the denominator is computed over the whole batch before the chunk loop), and
    it is applied to **both** arms, so it changes the memory footprint and not the arithmetic or
    the contrast. The relaunched probe holds `~0.9 s` per step against `0.507 s` on a quiet card:
    a 1.8x tax, paid, not fought.

23. **The `gamma == 0` control arm is bit-identical across genome dimensions, so one control
    serves both cells of the factorial.** *2026-09-05. Measured before it was assumed.*

    The genome probe is a `{tempo, a_gamma0} x d_g in {32, 512} x 2 seeds` factorial. `a_gamma0`
    sets `genome=False, free_gains=False, regenerate=False, estimator="none"`: `gamma == 0` for
    the whole life and no consolidation ever fires. Nothing in that arm *reads* the genome, so
    its two `d_g` cells should be the same run twice -- but "should" is not a measurement, and
    the coupling that would break it is not exotic. `Life.__init__` builds `Expression` and
    `Consolidator` **unconditionally**, both sized by `d_g`; if either drew from the global torch
    RNG, the number of draws would differ between 32 and 512 and every later global-RNG consumer
    would be displaced. That is precisely the class of silent coupling this project has been bitten
    by three times (§P-D.15, §P-D.16, and the inert clip of the audit).

    So it was run: two lives at the CPU `TINY` shape, arm `a_gamma0`, seed 0, identical in every
    field but `d_g`, two episodes of six steps each, comparing the recorded per-step loss and
    gradient norm **exactly** -- `repr`-level equality, not a tolerance, because a displaced RNG
    stream shows in the low bits immediately.

    | quantity | result |
    |---|---|
    | per-step `loss` and `gnorm`, all 12 steps | identical, digit for digit |
    | `max abs` difference over the whole agent state dict after 2 episodes | `0.000e+00` |

    *The consequence, and its cost.* The `d_g = 512` control cell is a verbatim rerun. It is
    dropped, and the `d_g = 32` control runs serve both contrasts: `~2.5` of the probe's `~12.6`
    GPU-hours saved at the shared card's `0.9 s` per step, with nothing given up. **What is given
    up is independence**: the two K-GENE contrasts now share a control arm, so their errors are
    correlated and the two verdicts must not be treated as two independent tests of the genome.
    The compensation is on the other side of the same coin -- the direct `tempo(32)` against
    `tempo(512)` comparison becomes *cleaner*, because the shared control removes a source of
    between-cell variance from the difference of differences, and that comparison is the one
    §P-D.9 predicted would matter (`41 %` of the available improvement captured at `d_g = 32`
    against `2.3 %` at `512`).

    *Why this is worth a ledger entry at all.* It is a negative result about the code -- "no
    hidden coupling here" -- and negative results about code are the ones that get assumed
    instead of measured. The check cost ninety seconds on the CPU and bought a decision about
    two and a half GPU-hours; the version of this project that skips it either burns the hours
    or, worse, saves them on an assumption that was never tested.

24. **The probe as first launched could not have produced a K-GENE verdict: two of the
    criterion's three controls were missing.** *2026-09-05. My error, found by reading the
    criterion against the runner two hours after the runner started.*

    10 §10.2 defines K-GENE as firing unless the RL-trained genome beats, on loss per FLOP over
    20 episodes, **each** of three controls: (a) `gamma == 0`, (b) the homeostatic rule alone,
    (c) a random `Delta g` **of the same norm**. The runner queued `tempo` against `a_gamma0`
    and nothing else. Twelve GPU-hours would have produced a number against (a) and no verdict,
    because a criterion with three conjuncts cannot be evaluated from one.

    *Why (c) is the one that matters, and why its absence would have been worse than a null.*
    `c_random` is `estimator="random"`: `Delta g = tau * xi / ||xi||`, an isotropic draw scaled
    to exactly the trust region. It differs from `tempo` in ~~**one** place -- where the proposal
    comes from~~ **-- corrected in §P-D.35: in TWO places, because `estimator="random"` also runs
    no probes, so the arms differ by `tempo`'s measured 21.3 % search overhead as well** -- and shares the expression, the regeneration, the homeostatic rule and the
    genome morphology. So it separates *"the learned proposal helps"* from *"a move of this size
    helps"*, and those are not the same claim. A perturbation of fixed norm applied at every
    episode boundary is a plausible regulariser on its own; without (c), a positive `tempo`
    against `a_gamma0` would have been reported as evidence for B4 when it might have been
    evidence for jitter. This is the same shape as §P-D.18's refutation: the interesting
    hypothesis and a boring one predict the same sign, and only the control tells them apart.

    *Fixed.* The queue at `d_g = 32` is now `tempo`, `a_gamma0`, `c_random`, `b_homeo`, two
    seeds each -- a complete K-GENE at the dimension where §P-D.9 says the search can move --
    and `d_g = 512` runs the mechanism arm last, reusing the controls under §P-D.23. Two seeds
    is a **screen** and is labelled one wherever it is reported; 10 §10.5 wants four for a claim.

25. **The first readings of the genome, at two boundaries: the fast learner works, the slow one
    has moved, and the guard that was supposed to fire did.** *2026-09-05,
    `runs/tempo/_pilot-R4-tempo-tsk1-dg32-s0`.*

    | | ep 0 | ep 1 |
    |---|---|---|
    | `L_bar` (train, per token) | 3.4877 | 2.5844 |
    | `L_check` (fitness) | 7.0778 | 6.2307 |
    | held-out `fresh` at the episode's end | 2.6255 | 2.2886 |
    | `reward` | -- (no previous episode) | **+0.3387** |
    | `abs(g - g_0)` | 1.108 | 1.718 |
    | `Lambda` | -0.288 | **-0.435** |
    | `gstat_ok` | True | **False** |
    | `lr_mult` (muon, emb, vec, head) | 1, 1, 1, 1 | **1.108, 1.087, 0.881, 1.109** |

    *The fast learner works, and the identity holds at birth.* Train NLL `5.06 -> 2.75` within
    episode 0 and held-out `fresh` `3.73 -> 2.63` monotonically over five evaluations, then
    `-> 2.29` through episode 1. Episode 0's `lr_mult` is exactly `[1, 1, 1, 1]` and its mixture
    exactly uniform, which is G-IDENT holding where it must: `B_rho = 0` at birth, so the first
    episode is the plain backbone and the genome cannot take credit for it.

    *G-STAT fires at episode 1, and the reason is not noise.* `d_last = +0.1199` against
    `d_first = +0.0289`: the episode improves **faster at its end than at its start**. That is
    not an unfinished transient, it is an episode that begins with a jolt -- the per-episode D2Z
    schedule re-warms the learning rate from near zero and the optimisers' momentum is rebuilt at
    every boundary, so `fresh` goes `2.6255` (end of ep 0) to `2.6792` (ep 1 step 50) before
    recovering to `2.2886` -- and then makes real progress throughout. At `N = 250` the fast
    system is nowhere near `W*(g)`, so **Proposition 1's premise does not hold at the screening
    length** and the two-timescale convergence argument is not licensed here. The guard is doing
    exactly its job; the registered remedy (lengthen `N`) costs 4x, which this card does not have
    while the controls are still unrun.

    *What that does and does not invalidate.* It invalidates the *theory*, not the *measurement*.
    Estimator 2 selects among proposals lived from a **common checkpoint under common random
    numbers**, so the ordinary training progress that dominates `L_check` between episodes --
    `0.85` nats, against a genome step worth hundredths -- is common mode and cancels in the
    ranking. This is why the design chose best-of-G over REINFORCE for the screens (06 §6.6), and
    a non-equilibrium screen is precisely the case it was chosen for. The paired K-GENE contrast
    at equal FLOPs is unaffected: both arms live the same non-equilibrium.

    *Every probe reward is negative, including the null, and that is expected.* At boundary 0:
    null `-0.2103`, candidates `-0.1124, -0.1352, -0.0983, -0.1308`. Twenty-five probe steps make
    the fitness **worse** than the boundary state, for every proposal and for doing nothing,
    because a probe is a 25-step episode with its own warm-up and fresh momentum -- the same jolt,
    with no time to recover. The offset is common to all five, which is exactly what the null
    probe is in the design for (06 §6.6, the control variate): the ranking is what is used, all
    four candidates beat the null, and the winner's state was kept at both boundaries.

    *The first K-HORIZON reading, and it points the right way.* The documented failure of learned
    update rules under a short-horizon reward is greed: they turn the learning rate down by orders
    of magnitude to win the near term (Wu et al. 2018; Metz et al. 2018; plan §8.3.3). A probe
    reward measured over 25 steps from a jolted state is about as greed-inviting a signal as this
    design contains. The genome's first expressed multipliers go the **other** way -- Muon
    `x1.108`, embeddings `x1.087`, heads `x1.109`, and only the vector group down at `x0.881`.
    One boundary is an anecdote, not evidence; it is recorded now because it is the specific
    number K-HORIZON will be read on, and a prediction written before the data arrive is worth
    more than one written after. **It was written before the data and the data went against it:
    §P-D.33 shows the multipliers turning at episode 3 and falling monotonically for the rest of
    the life, Muon `1.108 -> 0.527`. The anecdote was the wrong way round.**

    *One thing to watch that is not yet a finding.* `Lambda` moved `-0.288 -> -0.435` in one
    episode, toward the ordered side. **The first draft of this paragraph read that against a
    band on `Lambda` itself and is corrected in §P-D.26**: the trigger's limbs are per block,
    `Lambda` is a sum over `K_a = 2 x |active layers|` of them, and the stack got deeper between
    those two episodes, so most of that move is depth and not criticality. The quantity to watch
    is `max_k |lambda_k^rel|`, and it is watched in §P-D.26.

26. **The genome's first act was to switch on the whole stack. That reframes three criteria and
    corrects a criticality reading I had just written.** *2026-09-05,
    `runs/tempo/_pilot-R4-tempo-tsk1-dg32-s0`, episodes 0-2.*

    | ep | active params | blocks `K_a` | groups born | `Lambda` | `Lambda / K_a` | worst block | `zeta` mean |
    |---|---|---|---|---|---|---|---|
    | 0 | 17,886,264 | 16 | **13** (`Q1 Q3 H1 H6 H7 H9 H13 F3 F5 F7 F8 F9 F13`) | -0.2880 | -0.01800 | +0.03596 | +0.0045 |
    | 1 | 47,989,904 | 32 | 4 (`F2 H5 H11 H15`) | -0.4346 | -0.01358 | +0.05784 | +0.0113 |
    | 2 | 50,349,200 | 32 | 2 (`H0 F6`) | -0.6284 | -0.01964 | +0.08688 | +0.0211 |

    ("worst block" is `max_k` of the absolute relative per-block exponent; it was `['ffn', 0]`
    at all three consolidations.)

    At the **first** consolidation the morphology roles switched on thirteen groups, including
    both missing quarter-stacks (`Q1`, `Q3`): the agent went from the 8-layer primordium to the
    full 16-layer stack and from `17.9M` to `48.0M` active parameters in one boundary, and it
    kept growing. Every birth carries `adopted_from_probe: true`, so these are the *winning
    probe's* births, kept along with its state.

    *The correction this forces on §P-D.25.* I had just written that `Lambda` drifting
    `-0.288 -> -0.435` was criticality moving toward the ordered side, and read it against a
    `[-0.5, +0.5]` band. That is wrong twice over. `Lambda` is a **sum** over `K_a` sub-blocks
    and `K_a` doubled between those episodes, so the stack exponent grew mostly because the stack
    grew -- **per block it went the other way**, `-0.01800 -> -0.01358`. And 07 §7.6's corrected
    trigger is per block for exactly this reason ("growth alone would fire the trigger"), which I
    had read, and then failed to apply to the first number that needed it.

    *A defect found by the same check.* 07 §7.6's **[NORMATIVE]** trigger sentence still read
    `|Lambda| > 0.5` after the per-block correction had been made in the paragraph beneath it and
    implemented in `needs_control`. Prose and code disagreed, and on this date they disagreed
    about an actual verdict for the first time: `Lambda = -0.628 +- 0.010` is over the stale line
    by twelve standard errors and inside the implemented band by a factor of twenty-five. A
    reader following the normative sentence would have reported K-CRIT as triggered. Fixed in
    place, with the date and this entry cited.

    *A prediction, written before the data exist.* `Lambda/K_a` is not drifting monotonically,
    but the worst block is: `+0.036 -> +0.058 -> +0.087`, always the same block, a factor of
    about `1.5` per episode. Continued, it crosses the `0.5` band at **episode 6 or 7**. If it
    does, K-CRIT stops being a hypothesis this project can decline to test; if it flattens, that
    is a measurement of the homeostatic rule catching up. `zeta` is pushing the right way -- mean
    `+0.0045 -> +0.0113 -> +0.0211`, which is `exp(zeta) ~ 1.02`, a two-percent branch gain
    against a drift of fifty percent per episode -- so on present evidence the rule is a weak
    lever, and that is itself the thing 07 §7.5 said no published work had measured.

    *What growth does to K-GENE, and why the arm order already changed.* The reward is loss per
    FLOP and growth is charged in the denominator, so switching on the stack is not free. But it
    is evaluated over a **25-step probe**, where more capacity nearly always pays, against a cost
    borne for the rest of the life. That is the short-horizon bias of §8.3.3 appearing in the
    **morphology** dimension rather than the learning-rate one K-HORIZON was registered to catch.
    The consequences for the contrasts are concrete:

    * `tempo` against `a_gamma0` is **confounded**. `a_gamma0` fixes `morphology="primordium"`
      and never grows, so the contrast measures the genome's gains *and* its growth timing
      together; by episode 2 the arms differ by `2.8x` in active parameters and by `12.3` against
      `3` reference-episodes of spent compute. It is still the registered arm (a) and will be
      reported, with this caveat attached wherever it appears.
    * `tempo` against `c_random` is **not** confounded *by growth*. Both carry
      `morphology="genome"`, both grow, both regenerate, both run the homeostatic rule. They do
      differ in a second place that §P-D.35 corrects: `estimator="random"` runs no probes, so
      `tempo` carries a measured 21.3 % search overhead that `c_random` does not. §P-D.24 moved it to second in the queue because it asks B4's
      question; this is a second and stronger reason.
    * K-MORPH's hypothesis is in trouble before its run. "RL-timed growth beats a fixed schedule"
      is not testable by a policy that switches on thirteen groups at its first opportunity.
      Either the morphology roles are too sensitive at `c_rho = 8` against a trust region of
      `tau = 0.2 sqrt(d_g)`, or the probe horizon is too short to price capacity, or both. This
      is registered as the question R6 must answer and **not** as a defect: growing early is what
      G_stack and Mallik et al. predict is *correct* at this token-per-parameter ratio, so "the
      policy grows immediately because growing immediately is right" is a live alternative to
      "the policy is myopic". The two are separated by whether the grown arm wins at matched
      compute, which is what the contrast now running will say.

27. **The arms were matched on episodes and needed to be matched on compute; the run was
    stopped and restarted at four episodes.** *2026-09-05. A design change made against live
    data, with the compute it cost stated.*

    §P-D.26 measured the mechanism arm growing from `17.9M` to `50.3M` active parameters in
    three consolidations. The compute consequence was visible in the same log line and I did not
    read it until episode 3:

    | ep | `F_cum` (reference-episodes) | per episode |
    |---|---|---|
    | 0 | 1.86 | 1.86 |
    | 1 | 5.70 | 3.84 |
    | 2 | 12.29 | 6.59 |
    | 3 | 18.95 | 6.66 |

    A reference-episode is `250` primordium steps at `B = 32` (`F_ref = F_window_ref * B * lc.N`,
    with `lc.N` the *configured* 250 and not the doubled 500). So the grown mechanism arm costs
    `~6.7` reference-episodes per episode while a control pinned to `morphology="primordium"`
    costs `2` after the G-STAT doubling. Twenty episodes each would have been `~126`
    reference-episodes against `38`, and since every K-GENE contrast is scored at the arms'
    **common** budget, **thirteen of the mechanism arm's twenty episodes would have had no
    control to be scored against** -- about `3.8` hours per seed, `7.6` hours over the pair,
    bought and then discarded by the analyser.

    *Which clause of the criterion wins.* 10 §10.2 states K-GENE as "loss per FLOP over 20
    episodes". Those two halves were written before it was known that an arm could grow, and
    they now conflict. The matched-compute clause wins: B10 makes a matched-compute control the
    condition for reporting any mechanism at all, whereas "20 episodes" is a horizon convention.
    Every arm therefore now runs `--episodes 20 --F-budget 40`.

    *Why 40 and not a round number chosen for looking round.* It is what the registered
    20-episode primordium life **costs**: two episodes at `N = 250` plus eighteen at `N = 500`
    after the doubling, `2 + 36 = 38` reference-episodes. So the controls still bind on their
    registered horizon of 20 episodes and only the mechanism binds on FLOPs, stopping at about
    its seventh consolidation. Four arms x two seeds then costs `~12.3` hours instead of `~20`,
    and -- the point -- a **complete** K-GENE screen fits in one night where an incomplete one
    would not have.

    *What it costs, honestly.* The mechanism arm gets about **seven** consolidations instead of
    twenty. Seven ReST-EM updates is thin for a policy, and if K-GENE fails to fire it will be
    partly because the mechanism had few boundaries. That is not a flaw in the comparison, it is
    the finding underneath it: **a growing agent buys capacity with the compute it would
    otherwise have spent on boundaries**, and at matched compute it must show that the trade paid.
    Stating it in advance is the difference between a result and an excuse.

    *The cost of the change.* The run was paused at episode 4 step 50 -- cleanly, `rc = 42`, the
    checkpoint resumable -- and restarted rather than resumed, because its configuration is no
    longer the one under test. `54` minutes of card. The four episodes are kept at
    `runs/tempo/_pilot-R4-tempo-tsk1-dg32-s0` (the leading underscore keeps it out of the run-name
    grammar and so out of every analysis), and §P-D.25 and §P-D.26 cite it there: their findings
    -- the growth, the criticality profile, the first `lr_mult`, the G-STAT failure -- are facts
    about that life and are not withdrawn. They are simply not part of a K-GENE contrast.

    *The general lesson, which is the third instance tonight.* The number that mattered was
    printed in every log line from the first episode (`F 1.86 ref-episodes`, then `5.70`, then
    `12.29`) and I read it as progress rather than as a budget. §P-D.22 was a memory number read
    the same way, §P-D.26 a depth number. Three times in one session a quantity was on screen,
    correct, and unread until something else forced attention onto it. The mitigation is not
    "look harder": it is that a monitored run should **assert** its invariants -- here, that the
    arms of a live contrast stay within a factor of the same spent compute -- rather than print
    them and trust the reader.

28. **The probe pays a fixed tax about seven times larger than the signal it is used to measure,
    and the mechanism is optimiser state, not the learning-rate schedule.** *2026-09-05, the four
    consolidations of `runs/tempo/_pilot-R4-tempo-tsk1-dg32-s0`.*

    | ep | null probe | candidate mean | candidate sd | null, in candidate sd | best - null | 25 clean steps would be worth |
    |---|---|---|---|---|---|---|
    | 0 | -0.2103 | -0.1192 | 0.0170 | **-5.35** | +0.1120 | -- |
    | 1 | -0.0625 | -0.0553 | 0.0036 | **-1.97** | +0.0105 | +0.0847 |
    | 2 | -0.0687 | -0.0616 | 0.0028 | **-2.55** | +0.0110 | +0.0358 |
    | 3 | -0.0607 | -0.0543 | 0.0055 | **-1.17** | +0.0132 | +0.0128 |

    ("25 clean steps" is the episode's own measured `L_check` gain prorated to 25 of its steps --
    what a probe *should* deliver if a probe were simply a short piece of an episode.)

    *The tax.* After the first boundary the null probe returns a near-constant `-0.06`, while the
    same 25 steps taken inside an episode are worth `+0.013` to `+0.085`. The gap -- about
    `0.08` nats -- is paid by **every** probe, the null and all four candidates alike, and it is
    six to seven times the `0.011`-`0.013` by which the best candidate beats the null. The
    quantity the estimator selects on is a small difference riding on a large common artefact.

    *The mechanism.* **Downgraded to a hypothesis by §P-D.29, which tested it and found no
    support, and which identifies a better candidate. Read that entry before relying on this
    paragraph.** The obvious suspect is the per-episode D2Z schedule, but that one **scales**: `p_warm = 0.02` is 0.5 steps of a 25-step
    probe and 10 steps of a 500-step episode, the same fraction. What does not scale is the
    **optimiser state**. `boundary_in` rebuilds `Muon` and `AdamW` from scratch at every
    boundary and at every probe, so momentum starts at zero with a fixed time constant --
    `1/(1 - 0.95) ~ 20` steps for both Muon's momentum and AdamW's second moment. Twenty steps
    is `4 %` of a 500-step episode and `80 %` of a 25-step probe. **The probe therefore measures
    how fast a configuration recovers from a cold optimiser, and only secondarily how well it
    learns.**

    *Why that is a distinct hazard from the one already registered.* K-HORIZON watches for the
    learned rule turning the learning rate **down** to win the near term. This is a different
    bias with the same root: a genome that eases the re-accumulation of momentum -- larger branch
    gains, gentler curvature, anything that makes the first twenty steps of a cold optimiser
    cheaper -- is selected over a genome that makes the agent learn better over five hundred.
    Nothing in the design detects that, and §P-D.25's reading of the first `lr_mult` (three of
    four multipliers **up**) is consistent with it as well as with the benign story.

    *How much of this is my configuration rather than the design.* `N_PROBE_STEPS = 100` is the
    specification's default. The runner passes `max(5, N//10) = 25` to fit the screen into a
    night. At 100 the momentum rebuild would be `20 %` of a probe rather than `80 %`, so a
    substantial part of the tax measured here is a choice I made, not a property of estimator 2.
    That is worth saying plainly before the number is quoted anywhere as a fact about the design.

    *Two fixes, registered and not applied tonight.* (i) `n_probe_steps = 100`, the default, which
    amortises the rebuild. (ii) Carry the optimiser state across the boundary into the probes
    instead of rebuilding it -- the momentum buffers describe the loss surface, which the genome
    perturbs but does not replace, and rebuilding them discards information at every one of the
    `(G+1)` probe restarts. **Neither is applied to the running comparison.** The run has already
    been restarted once tonight (§P-D.27); the estimator demonstrably works as built, with the
    null placed last at all four boundaries; and a second mid-flight change would split the
    provenance of the contrast a second time for a refinement that is not needed to reach a
    verdict. They belong to the run after this one.

    *One reading that is not a finding yet, and one that is close.* The null is beaten by the
    candidate mean at every boundary (`-5.35, -1.97, -2.55, -1.17` in units of the candidates'
    own spread), which says a nonzero `Delta g` beats no move -- but **not** that the *learned*
    move beats a random one, which is exactly `c_random`'s job (§P-D.24). Closer to a finding:
    the sampled proposal norm grows past what pure noise would give. With `sigma = 0.191` and
    `d_g = 32` an isotropic draw has norm `1.080`; the observed `|Delta g|` runs
    `1.119, 1.214, 1.297, 1.210`, implying a policy mean `|mu|` of `0.29, 0.55, 0.72, 0.55`. The
    consolidator is acquiring a preferred direction rather than sampling isotropically. It is
    still sub-dominant -- `|mu| / (sigma sqrt(d_g)) ~ 0.5`, so sampling noise governs any single
    proposal -- and four boundaries of one seed is an anecdote. But it is the first evidence in
    this project that `C_psi` is learning anything at all.

29. **§P-D.28's mechanism is withdrawn. The test that was supposed to confirm it did not
    reproduce the phenomenon, and the pilot's own numbers point somewhere else.** *2026-09-05,
    within an hour of writing it.*

    §P-D.28 asserted that the probe's `~0.08`-nat tax is **optimiser state**: `boundary_in`
    rebuilds Muon and AdamW at every probe, momentum restarts from zero with a fixed `~20`-step
    time constant, and 20 steps is `80 %` of a 25-step probe. The argument is sound as an
    argument. It was published as a mechanism on the strength of the argument alone, which after
    §P-D.22 and §P-D.26 is a habit this session has been punished for twice already.

    *The test.* At the CPU `TINY` shape, an arm with `genome=False, free_gains=False,
    morphology="primordium"` -- so a boundary expresses nothing, changes no gain and grows
    nothing, and the **only** difference between conditions is whether the optimiser survives it
    -- four seeds, one shared checkpoint each, `n` probe steps under `fresh` (rebuilt, as built)
    against `carried` (state restored):

    | `n` | fresh | carried | carried - fresh | sd over 4 seeds |
    |---|---|---|---|---|
    | 5 | +0.06382 | +0.04139 | **-0.02243** | 0.0665 |
    | 10 | +0.08238 | +0.08809 | +0.00571 | 0.0720 |
    | 20 | +0.07607 | +0.10656 | +0.03049 | 0.0439 |
    | 40 | +0.08004 | +0.11797 | +0.03794 | 0.0288 |
    | 80 | +0.09233 | +0.12534 | +0.03301 | 0.0359 |

    The prediction was that carrying the state helps **most** at small `n`, where the rebuild is
    most of the probe, and fades as a fraction as `n` grows. The measurement is the **opposite**:
    nothing at `n = 5` (and of the wrong sign), nothing distinguishable at `n = 10`, and a benefit
    only from `n = 20` upward -- at exactly the lengths where the hypothesis predicted it should
    matter least. At `2.6` standard errors on its best cell and four seeds, this is a weak test;
    but a weak test that points the wrong way is not support.

    *And the deeper defect in the test, which is the part worth remembering.* Look at the `fresh`
    column: every entry is **positive**. At `TINY` a probe *improves* the fitness. On the real
    run every probe *degrades* it. **The testbed does not reproduce the phenomenon it was built
    to explain**, so it could not have confirmed or refuted anything about it; it measured a fix
    for a symptom that was not present. Checking that a testbed exhibits the effect before using
    it to explain the effect is elementary, and I did not do it. That check is now the first step
    of any such diagnostic here.

    *A better candidate, from the pilot's own numbers.* At boundary 0 the null probe scored
    `-0.2103` and the candidates `-0.098` to `-0.135` -- a gap of `+0.112`, ten times the
    `~0.011` gap at every later boundary. Boundary 0 is also the one where thirteen groups were
    born (§P-D.26). And the null probe cannot grow: `Delta g = 0` leaves `g = g_0 = 0`, which
    expresses the primordium, while every candidate moves `g` and switches blocks on. So at
    boundary 0 the contrast was not "learned proposal against no proposal" but **"2.7x the
    capacity against the primordium, priced over 25 steps"** -- and capacity won by a distance
    that never recurred once every arm was already grown. That is a concrete, checkable
    explanation of both the size of the boundary-0 gap and its collapse afterwards, and it
    sharpens §P-D.26: the best-of-G selection did not merely permit growth, at its first
    opportunity it was *measuring* growth and almost nothing else.

    *What is left standing.* The tax itself is a measurement on the real run and is not
    withdrawn: the null probe returns `~-0.06` where 25 in-episode steps are worth `+0.013` to
    `+0.085`. Its **cause is now unknown**. The D2Z scaling argument still rules out the schedule
    as a fixed-fraction explanation, and that is all it does. The decisive test is at the real
    shape -- vary `n_probe_steps` at `d_g = 32` with everything else fixed, and separately hold
    the morphology pattern constant across probes so growth cannot be what is being priced -- and
    it needs a card that is currently committed. Registered, not run.

30. **The probe would have produced runs the analyser could not score. Found by dry-running the
    analyser against the live configuration instead of waiting to need it.** *2026-09-05.*

    `analyse` keys every lookup by the exact configuration string. The probe's runs are at
    `tsk1-dg32`; the noise runs `R1-nolat` exist at `tsk1`. So `sigma_e` came back `None`, and
    **G-NOISE makes everything below it NOT EVALUATED** -- including K-GENE, with all four arms
    present and complete. Twelve GPU-hours would have landed in a report reading
    `K-GENE  NOT EVALUATED (an arm is missing or incomplete)` for a reason that has nothing to do
    with the arms.

    Dry run against the live configuration, with one partial run present:

    ```
    ================ tsk1-dg32 ================
    R1 NOISE  NOT EVALUATED (G-NOISE: nothing below is reportable)
    K-GENE vs a_gamma0         NOT EVALUATED (an arm is missing or incomplete)
    ```

    against the same command at the base configuration:

    ```
    ================ tsk1 ================
    R1 NOISE  n=2 test 1.8920 sigma_e 0.0036 tok/s 27584 mem 5.59 GB
    ```

    *The fix, and why it is a fact rather than a convenience.* `R1-nolat` is a `genome=False`
    arm, and §P-D.23 **measured** such an arm to be bit-identical across `d_g` -- two `TINY`
    lives differing only in `d_g`, twelve steps identical digit for digit, zero parameter
    difference. So an `R1-nolat` run at `tsk1-dg32` would be the `tsk1` run rerun, and reading
    the noise from `tsk1` is not an approximation of anything. The same holds for the expression
    rank `r`. It does **not** hold for `dx` or `dc`, which change the interface widths and so
    change the model whose noise is being measured, so the fallback is restricted to `dg` and `r`
    and the report prints `sigma_e 0.0036 (from tsk1, 11 P-D.30)` whenever it is used. A bar
    should never come from a run the reader cannot identify.

    *Written, tested, reverted, and not applied.* The patch was applied to a live checkout, the
    dry run confirmed `sigma_e 0.0036 (from tsk1)` at `tsk1-dg32`, and it was then reverted --
    `scripts_digest` back to `948c2fb8d08b3cc5`, byte-identical to what the running life and the
    pilot both recorded in their manifests. `scripts/tempo_life.py` writes that digest into every
    run's `manifest.json`, so editing any `scripts/tempo_*.py` mid-chain would split one
    comparison's provenance across two digests. The analyser is needed only once the runs exist,
    so there is nothing to buy by applying it early and a provenance split to pay. It goes in when
    the chain finishes.

    *The general point.* This is the second thing tonight caught by exercising the *downstream*
    machinery before it was needed (§P-D.27 was the first, caught by reading the compute column).
    A run is not "in flight and fine" because it is producing records; it is fine when something
    has read those records end to end and produced the verdict they are for. Dry-running the
    analyser costs seconds and should be a precondition of launching any chain, not a step taken
    at the end.

31. **§P-D.26's criticality prediction is already in doubt, on the fourth point of the series it
    was fitted to.** *2026-09-05, an hour after making it.*

    §P-D.26 registered a prediction: the worst block's relative exponent was growing by a factor
    of about `1.5` per episode (`+0.036 -> +0.058 -> +0.087`) and, continued, would cross the
    `0.5` band at episode 6 or 7, making K-CRIT a test this project could no longer decline. It
    was fitted to three points, which were all that existed when it was written. The pilot's
    fourth arrived minutes later:

    | ep | 0 | 1 | 2 | 3 |
    |---|---|---|---|---|
    | worst block | 0.03596 | 0.05784 | 0.08688 | **0.10181** |
    | ratio | -- | 1.609 | 1.502 | **1.172** |
    | increment | -- | +0.02188 | +0.02904 | **+0.01493** |

    The ratio is **decelerating** and the increment turned over. Extrapolating the increment's
    own decay rather than a constant ratio, the series stalls at about `0.118` and the band is
    never approached. On present evidence the prediction is **wrong**, and it was wrong in the
    way exponential fits to three points usually are.

    *Recorded now rather than when it resolves, and why.* The live run is at episode 2 and will
    pass episode 7 in about ninety minutes, which is when this actually settles. Waiting would
    have produced a tidier entry. But the prediction was published, evidence against it is in
    hand, and the interval between "I know this is probably wrong" and "the record says so" is
    exactly the interval in which a wrong published claim does its damage. §P-D.19 was withdrawn
    twenty-five minutes after publication and that was too slow.

    *What it would mean if it holds.* The exponent settling near `0.12` per block, against a band
    of `0.5`, is 07 §7.6's own "the trigger is close to unfireable -- a defect in the band, not a
    property of the network" observed a second time from a different direction. It would also be
    a real result about the mechanism: a network that grows `2.8x` and trains for tens of
    thousands of steps drifts *bounded* distances from criticality, with a homeostatic rule
    running that is far too weak (`~2 %` branch gain per episode) to be what is holding it. If
    the drift is self-limiting **without** the rule doing the limiting, then K-CRIT is not merely
    "not needed", it is not needed *for a reason* -- and that reason belongs in 07 §7.5, where
    the standing claim is that nobody has measured this.

32. **The registered argument for the I3 decision, and a correction to §P-D.21's claim that the
    remedy is cheap.** *2026-09-05. Written for a decision that is the user's, not mine, and
    written so it can be taken without re-deriving it.*

    §P-D.21 measured that `+0.0245` of the `+0.0989`-nat `K = 1` deficit -- a quarter -- is the
    token baseline carrying **tied input/output embeddings** and **value embeddings**, which the
    span agent has neither of. It called the remedy *actionable* on the grounds that "`TextIn.T_emb`
    and `SpanCodec.W_v` are both `vocab x d`-shaped and could share weights through the
    interface". **That sentence is wrong, and the shapes say so:**

    | matrix | shape | parameters | status at `K = 1`, `lambda_lat = 0` |
    |---|---|---|---|
    | `TextIn.T_emb` | `(8192, 128)` | 1,048,576 | **live** -- the input path |
    | `SpanCodec.W_v` | `(8192, 256)` | 2,097,152 | **live** -- the output projection |
    | `SpanCodec.T_dec` | `(8192, 256)` | 2,097,152 | **one row live**: at `n_dec_layers = 0` the logits are `softcap(W_v LN(T''[BOS] + P' + W_c y))`, so only the BOS row is ever read -- 8,191 rows receive no gradient |
    | `SpanCodec.T_enc` | `(8192, 256)` | 2,097,152 | **dead**: frozen after R0, and at `lambda_lat = 0` the encoder is never executed during training at all |

    `T_emb` is 128 wide and `W_v` is 256 wide. They cannot share weights as they stand. Tying
    them requires `d_e = d_c = 256`, which also doubles `T_emb` and changes `W_x` from
    `(d_x x K*128)` to `(d_x x K*256)`. The remedy is a design change with its own cost, not a
    flag, and §P-D.21 should not have called it cheap.

    *What the shapes reveal, which is more interesting than the correction.* The span agent
    carries **four** vocabulary-rowed matrices totalling `7.34M` parameters. At `K = 1` with
    `lambda_lat = 0`, `4.19M` of them are **dead** -- the whole frozen encoder table and all but
    one row of the decoder's input table -- and the `3.15M` that are live are **split across two
    tables that never see each other's gradient**. The FLOP-matched baseline at width 640 carries
    **one** table, `8192 x 640 = 5.24M`, entirely live, receiving gradient from the input side and
    the output side at once. So "the baseline has tied embeddings" understates the difference:
    the span agent learns its vocabulary representation **twice, in two smaller unshared pieces,
    while carrying a third and a fourth copy that learn nothing**. That is a specific, mechanical
    account of where a `0.0245`-nat efficiency gap could come from, and it predicts the gap should
    *shrink* at larger `K`, where the decoder's tables come alive and the encoder's table earns
    its keep through `lambda_lat > 0`.

    *The invariant, and the fact that it is broader than its own justification.* 02 §2.2 says the
    two maps "are different by design" and cites exactly one reason: CALM's measurement that
    feeding the codec **latent** back as *input* costs quality (§P-A.6, BrierLM `4.70 -> 3.25`).
    That is a finding about a **latent**, not about a **weight matrix**. The acceptance test I3
    is nevertheless categorical -- `text_in` may share **no parameter** with the codec, and
    perturbing *any* codec parameter must leave (2.2) unchanged:

    ```python
    ids_codec = {id(p) for p in codec.parameters()}
    assert not any(id(p) in ids_codec for p in text_in.parameters())
    ```

    So the implemented invariant forbids a class of designs that its stated rationale does not
    speak to. Tying an embedding table is not feeding the latent back: the input path would still
    be `W_x [T_emb[v]]` and would still never see `x_bar`. Whether that distinction should be
    honoured is a judgement about what I3 is *for*, and it is not mine to make unilaterally --
    an invariant is a commitment, and quietly reinterpreting one because a measurement made it
    inconvenient is how specifications stop meaning anything.

    *The decision, stated so it can be taken.*

    * **Keep I3 as implemented.** The `0.0245` stays on the books as a cost of the design. The
      argument for: the separation is one of the few load-bearing commitments the interface makes,
      it was made on a measured finding, and `0.0245` nats is a quarter of a deficit that is
      itself `~5 %` of the loss. Nothing else in the design depends on the tie.
    * **Narrow I3 to what it was for**, i.e. "the input path never consumes `x_bar` or any
      encoder output", and permit input/output table sharing. Cost: `d_e = d_c = 256`, a new R0
      (the codec's `W_v` becomes a shared object), and a re-run of every `K = 1` comparison. Gain:
      up to `0.0245` of a `0.0989` gap, and a cleaner story about why four vocabulary tables exist.
    * **Do neither yet.** The observation above predicts the gap shrinks at `K > 1`, where three
      of the four tables come alive, and `K = 4` is the design's default. If that prediction holds,
      the whole question is an artefact of the `K = 1` fallback and is not worth an invariant.

    **My recommendation is the third**, and it is cheap to settle: the `K = 4` and `K = 2` K-CONT
    runs already exist, and the same `--no-ve --no-tied` baselines that produced §P-D.21 at
    `K = 1` would give the same decomposition at `K = 4` for two baseline runs. If the two
    features are worth much less at `K = 4` than at `K = 1`, I3 is not implicated and the deficit
    was never about embeddings. That is a measurement, not a judgement, and it should come before
    anyone touches an invariant.

33. **The genome learned to throttle the learning rate. K-HORIZON's failure mode is visible in
    the first completed arm, and it refutes §P-D.25's reading of it.** *2026-09-05,
    `runs/tempo/R4-tempo-tsk1-dg32-s0`, eight episodes to a 40-reference-episode budget, 1.97 h.*

    | ep | `F` | active | `fresh` | reward | `lr_mult` (muon, emb, vec, head) | born |
    |---|---|---|---|---|---|---|
    | 0 | 1.86 | 17,886,264 | 2.6268 | -- | `1.000, 1.000, 1.000, 1.000` | 13 |
    | 1 | 5.71 | 47,989,904 | 2.2910 | +0.3490 | `1.108, 1.087, 0.881, 1.109` | 4 |
    | 2 | 12.36 | 51,011,736 | 2.0194 | +0.1337 | `1.110, 1.066, 0.752, 1.000` | 2 |
    | 3 | 18.92 | 50,618,520 | 1.9335 | +0.0475 | `1.098, 1.017, 0.709, 0.835` | 0 |
    | 4 | 25.56 | 51,011,736 | 1.8546 | +0.0457 | `0.868, 0.941, 0.632, 0.751` | 0 |
    | 5 | 32.13 | 51,011,736 | 1.7904 | +0.0281 | `0.627, 0.918, 0.667, 0.647` | 0 |
    | 6 | 38.51 | 50,618,520 | 1.7549 | +0.0226 | `0.524, 0.785, 0.683, 0.561` | 0 |
    | 7 | 44.83 | 50,618,520 | 1.7430 | +0.0030 | `0.527, 0.787, 0.686, 0.564` | 0 |

    *The finding.* After episode 3 every expressed learning-rate multiplier falls, monotonically
    and together: Muon `1.108 -> 0.527`, the heads `1.109 -> 0.564`, the embeddings
    `1.087 -> 0.787`. **The rule more than halved the learning rate of the group that trains the
    matrices.** The bound is `[1/4, 4]`, so `0.527` is not a clamp, it is a choice. The
    per-episode reward decays alongside it -- `+0.349, +0.134, +0.048, +0.046, +0.028, +0.023,
    +0.003` -- reaching effectively zero at the last boundary.

    That is the documented failure of learned update rules under a short-horizon reward, in the
    exact form the literature describes: "chooses too small a learning rate by multiple orders of
    magnitude, even ... 100 steps" (Wu et al., ICLR 2018; Metz et al. 2018; plan §8.3.3). Not
    orders of magnitude here -- the bounded role and the trust region see to that, and those
    defences are working -- but the **direction is unambiguous and the trajectory is monotone
    over five consecutive boundaries.**

    *This refutes §P-D.25.* Eight hours ago, on the strength of the **first** boundary, I wrote:
    "the genome's first expressed multipliers go the other way -- Muon `x1.108`, embeddings
    `x1.087`, heads `x1.109` ... a prediction written before the data arrive is worth more than
    one written after." The prediction was written before the data and the data went against it.
    The first two boundaries raised the multipliers; from the third they turned and fell for the
    rest of the life. One boundary was an anecdote, I said so at the time, and the anecdote was
    the wrong way round.

    *The mechanism, and it connects to the probe.* Why would lowering the learning rate win a
    25-step probe? Because §P-D.28 measured that every probe **degrades** the fitness -- a
    `~0.08`-nat tax paid by the null and every candidate alike -- and the most direct way to
    reduce a jolt is to take smaller steps. The estimator selects on which proposal loses least
    over 25 steps from a re-warmed schedule, and a damped learner loses least. §P-D.29 withdrew
    the *explanation* of the tax, and the tax itself stands; what is new here is a rule visibly
    **exploiting** it. The short-horizon bias is not entering through the reward's horizon alone,
    it is entering through the probe's artefact.

    *What is NOT yet established, and the free test that settles it.* The expressed multipliers
    are `Phi_theta(g)`, and `theta` trains at `tau_1` while `g` moves by consolidation, so the
    fall could in principle come from either. The discriminator costs nothing: `c_random` runs
    the **same** expression map under the **same** `tau_1` training with a **random** `Delta g`
    of the same norm. If its multipliers fall the same way, the decline is a property of the
    machinery and not of the learned policy, and K-HORIZON is about the design rather than the
    rule. If they do not, the policy chose it. That comparison is already queued and needs no
    extra run.

    *Replicated on the second seed, and the sign agrees in every group.* `s1` completed
    minutes later, also eight episodes to the budget, also 1.96 h:

    | group | `s0` ep 1 -> ep 7 | `s1` ep 1 -> ep 7 |
    |---|---|---|
    | muon | `1.108 -> 0.527` (`x0.48`) | `0.876 -> 0.731` (`x0.83`) |
    | emb | `1.087 -> 0.787` (`x0.72`) | `1.018 -> 0.606` (`x0.60`) |
    | vec | `0.881 -> 0.686` (`x0.78`) | `0.869 -> 0.840` (`x0.97`) |
    | head | `1.109 -> 0.564` (`x0.51`) | `1.226 -> 0.742` (`x0.61`) |

    **All eight multipliers fell, and all eight end below unity** -- against four of eight above
    unity at the first boundary. The magnitude differs (`s0` halves Muon's rate, `s1` barely
    touches it) but no group in either seed rose. Final held-out `fresh`: `1.7430` and `1.7652`.
    The seeds disagree about *where* the throttling lands, not about whether it happens.

    *Growth, for the record.* Births run `13, 4, 2, 0, 0, 0, 0, 0` on `s0`: front-loaded and then
    **stopped**, with the active count settling near `51M` and oscillating slightly as individual
    blocks cross `eps_skip` both ways. So the policy is not a switch that flips everything on and
    saturates -- it grows hard, then stops, which is a timing decision and is comparable against
    a schedule. §P-D.26's worry that K-MORPH would be untestable is eased. `s1` grew on a
    **different** schedule -- `12, 0, 4, 3, 1, 1, 1, 2`, still adding blocks at the last boundary,
    ending at `54.0M` against `s0`'s `50.6M` -- so growth timing is a genuinely seed-dependent
    decision rather than a fixed consequence of the machinery. That is the first sign that the
    morphology roles carry a *policy* and not just a threshold.

    *And §P-D.31's prediction resolves: refuted, as expected.* The worst block's relative
    exponent ran `0.0370, 0.0614, 0.0914, 0.1071, 0.1290, 0.1444, 0.1572, 0.1609` -- increments
    `0.0244, 0.0300, 0.0157, 0.0219, 0.0154, 0.0128, 0.0037`, decelerating to a near-halt. It
    ended at `0.161` against a band of `0.5`, having flattened rather than crossed at episode 6 or
    7 as §P-D.26 predicted. `Lambda / K_a` likewise settled at `-0.034`. A network that grew
    `2.8x` and trained through 3,500 steps drifted a **bounded** distance from criticality, and
    the homeostatic rule was far too weak to be what bounded it.

34. **A stale watcher of my own stopped the chain 56 seconds into the sharp control arm. Second
    instance of the same platform failure as §P-D.15.** *2026-09-05, 19:43.*

    A cooperative-stop watcher had been written at 15:07 to halt the chain before a duplicate
    control run (§P-D.23), targeting the completion of `R4-tempo-tsk1-dg32-s1`. The design
    changed twice afterwards and the watcher was retargeted and relaunched twice; each time I
    "killed" the previous copy with `pkill -f pause_after_dg512`. At 19:42 `tempo-s1` completed,
    the chain started `c_random-s0`, and **three surviving watchers all fired at once**: `PAUSE`
    appeared, `tempo_life.py` honoured it at step 62 of episode 0, and the run exited `rc = 42`,
    taking the chain down with it.

    *The cause.* `pkill -f` does not kill these processes under Git Bash on Windows. It reports
    nothing and exits, and the process keeps running. This is the **second** appearance of the
    same shape: §P-D.15 was `ps -W`, which prints only executable paths and so could never match
    a pattern in a command line, and it let two GPU jobs run concurrently. Then it was a *check*
    that silently could not see; now it is an *action* that silently does not act. Both reported
    success. Both were process control through a POSIX tool that Windows does not back.

    *What makes this one worse than the first.* After §P-D.15 the fix was written down: enumerate
    with `Get-CimInstance Win32_Process` and match on `CommandLine`. I used exactly that for the
    **detection** here -- and then reached for `pkill` for the **kill**. Having learned the
    platform lesson for one half of the operation and not the other is not bad luck; the rule
    was recorded as "how to find a process", and it needed to be recorded as "how to find *and
    end* one". The standing rule is now: on this machine, `pkill`, `killall` and `ps` are never
    used for anything that matters. Enumerate with `Get-CimInstance Win32_Process`, end with
    `taskkill /F /PID`, and **verify the count afterwards** -- which is how the three copies were
    finally found.

    *Cost and repair.* `56` seconds of card. The partial `R4-c_random-tsk1-dg32-s0` was deleted
    rather than resumed -- the resume path is designed and tested, but this is the arm the whole
    K-GENE reading turns on and a minute of compute is not worth introducing a variable into it.
    The three watchers were killed by PID, the watcher script was **deleted** so nothing can
    revive it, `PAUSE` was cleared, and the chain relaunched and confirmed running. The runner's
    own `done()` check skipped both completed `tempo` runs, which is the mechanism working as
    intended.

    *And the reason the watcher existed at all is gone.* It was needed because the *first*
    launch's runner carried the pre-§P-D.23 loop and would have run a duplicate control. The
    runner has been rewritten twice since and now skips duplicates itself (`[reuse]`), so the
    watcher had been redundant for four hours before it fired. **A guard that outlives its
    reason is not neutral** -- it is a live process with a trigger condition that the thing it
    was guarding no longer avoids.

35. **The sharp control beats the mechanism on the first seed, and checking it exposed an error
    in how I have been describing the contrast.** *2026-09-05, one seed of two; the second is
    running. Recorded as a measurement, not as a verdict.*

    *First, the error, because it changes what the number means.* §P-D.24 and §P-D.26 both claim
    `c_random` "differs from `tempo` in **one** place -- where the proposal comes from". That is
    false. `estimator="random"` draws `Delta g = tau * xi / ||xi||` **and runs no probes at all**;
    it never enters `_best_of_G`. Measured over the completed runs:

    | run | episodes to `F = 40` | `F_probes` | probe share | mean active params |
    |---|---|---|---|---|
    | `tempo` s0 | 8 | 9.54 | **21.3 %** | 46,345,867 |
    | `tempo` s1 | 8 | 9.94 | **21.7 %** | 47,297,274 |
    | `c_random` s0 | **12** | 0.00 | 0 % | 36,313,554 |

    So the two arms differ in the proposal source **and** in whether a search is run for it. At
    equal FLOPs that is accounted for -- it is why `c_random` gets twelve episodes to `tempo`'s
    eight -- and K-GENE's criterion, being loss **per FLOP**, is well posed as written. What is
    not licensed is the mechanistic gloss: a loss here means "the learned proposal *together with
    its 21 % search cost* is worse", not "the learned proposal is worse".

    *The measurement, scored by the analyser's own `score_at` at the common budget.*

    | | `tempo` | `c_random` | difference |
    |---|---|---|---|
    | seed 0 at `F* = 43.40` | 1.7457 | **1.7282** | **+0.0175** (the mechanism is worse) |

    ~~The `tsk1` bar is `3 sigma_e = 3 x 0.0036 = 0.0107`, so the gap is `1.6x` the bar on one
    seed.~~ **Corrected in §P-D.36: `criterion` takes the bar from the CONTROL ARM's spread on
    the criterion's own metric, not from R1's `sigma_e`. The bar is `0.0429`.** **One seed is not a verdict** -- 10 §10.5 needs the paired contrast over at least two
    -- and `c_random` s1 is running now. What would reverse it: a seed-1 difference of `-0.0175`
    or worse, which would put the mean at zero. `tempo`'s own seeds ended `0.0222` apart
    (`1.7430`, `1.7652`), so a swing of that size is not impossible, and the entry will be
    updated either way rather than quietly left standing.

    *The §P-D.33 discriminator, provisionally answered.* §P-D.33 could not tell whether the
    monotone learning-rate throttling belonged to the learned policy or to the machinery, since
    `Phi_theta` trains at `tau_1` in both arms. `c_random`'s Muon multiplier, boundary by
    boundary: `0.921, 0.847, 0.747, 0.555, 0.584, 0.646, 0.791, 0.877, 0.735, 0.738, 0.643`. It
    **wanders and mean-reverts** in a band around `0.7`. `tempo`'s: `1.108, 1.110, 1.098, 0.868,
    0.627, 0.524, 0.527` -- it falls and **stays down**. So there is a machinery component (both
    arms sit below unity, so `tau_1` training of the expression map drags the multipliers down on
    its own) and a policy component (only the learned arm drives one direction and holds it).
    That is the shape §P-D.33 predicted, on one seed of each.

    *What this makes decisive, and it is now queued.* If `tempo` loses to `c_random`, the two
    candidate explanations are "learning the proposal does not help" and "**searching** for the
    proposal costs more than the proposal is worth". They are separated by **estimator 1**:
    REINFORCE learns the policy from the episode reward with **no probes and therefore no
    overhead** (06 (6.9); the design offers it as the production rule and estimator 2 only as the
    screening rule). An arm at `--estimator reinforce` costs what `c_random` costs and answers
    the question that `c_random` cannot. It is queued behind K-GENE's remaining controls rather
    than ahead of them: a complete criterion first, then the mechanism behind its verdict.

36. **K-GENE FIRES against its sharp control. The RL-trained genome is not better than a random
    `Delta g` of the same norm at matched compute.** *2026-09-05, two seeds, screening strength.
    The first verdict on this project's central claim.*

    | | `tempo` | `c_random` | difference |
    |---|---|---|---|
    | seed 0 at `F* = 40.13` | 1.7498 | 1.7426 | **+0.0072** |
    | seed 1 at `F* = 40.13` | 1.7809 | 1.7628 | **+0.0181** |
    | **paired mean** | | | **+0.0127** `[-0.0570, +0.0824]` `n = 2` |

    Computed with the analyser's own `score_at`, `common_budget` and `criterion`, imported and
    not modified:

    ```
    K-GENE vs c_random   +0.0127 [-0.0570, +0.0824] n=2  sigma 0.014285  bar 0.0429 -> FIRED
    ```

    *Why it fires on an interval that spans zero, and why that is right.* K-GENE's rule is
    `fires_when="not_better"`: the criterion is escaped only by `m <= -bar and hi < 0`, that is,
    only by the mechanism being **demonstrably better**. The burden of proof is on the mechanism,
    which is the correct direction for a criterion whose job is to stop a decorative mechanism
    from being kept. A mean of `+0.0127` -- the wrong sign -- fires it. **Both seeds agree in
    sign.**

    *A correction to §P-D.35, which quoted the wrong bar.* I wrote "the `tsk1` bar is
    `3 sigma_e = 0.0107`". That is R1's evaluation noise. `criterion` computes the bar from the
    **control arm's own spread on the criterion's metric**, which here is
    `3 x sd(1.7426, 1.7628) = 0.0429`. The mistake did not change the verdict -- under
    `not_better` the bar only governs escape, and the mean is positive -- but a bar quoted from
    the wrong arm is exactly the sort of thing that changes a verdict in some other cell.

    *What the criterion's registered consequence says.* 10 §10.2: "the RL rule is decorative; the
    genome reduces to (b) or is removed; reported." That is now the standing verdict at
    `d_g = 32`, at screening strength, pending (a) and (b).

    *And why the night should now ask WHY rather than measure again THAT.* More seeds cannot turn
    `FIRED` into `DID_NOT_FIRE` unless the paired mean goes negative, and both seeds are positive;
    they would tighten an interval that does not enter the verdict. The informative question is
    which of two explanations holds, because they have opposite consequences for the design:

    * **"The learned proposal is no better than random."** The genome's action space is
      `d_g = 32` numbers expressed into gains; if a policy trained on the agent's own loss
      improvement cannot beat an isotropic draw of the same norm, B4 fails on its own terms.
    * **"Searching for the proposal costs more than the proposal is worth."** `tempo` spends a
      measured **21.3 %** of its compute in `_best_of_G` and `c_random` spends none (§P-D.35), so
      at matched FLOPs `c_random` trains twelve episodes to `tempo`'s eight. The proposal could
      be worth something and still lose, if the search is priced above it.

    These are separated by **estimator 1**. REINFORCE learns the same policy from the episode
    reward with **no probes and no overhead** (06 (6.9)); the design offers it as the *production*
    rule and estimator 2 only as the *screening* rule, so this is the design's own answer rather
    than a rescue invented after the fact. An arm at `--estimator reinforce --tag reinf` costs
    what `c_random` costs. It is queued next, ahead of control (b), which the criterion no longer
    needs in order to have a verdict.

37. **A reckoning: what five refutations across four models actually say, and the mis-test that
    makes tonight's the weakest of them.** *2026-09-06. Written after K-GENE fired, before the
    remaining arms, so that the reading is on record before the data that might flatter it.*

    *The through-line.* Five compact-slow-channel mechanisms have now been measured against a
    matched-compute control in this repository, and five have lost:

    | mechanism | model | result |
    |---|---|---|
    | a genome carrying **weights** | ANLAGE | one regression of an adult decodes to `8.5` nats against a `9.0` null |
    | a **population** of siblings | ANLAGE | `-0.115` nats against one long lifetime at equal compute |
    | **assimilation** over 15 rounds | ANLAGE | `-0.145` nats |
    | the **span interface** | TEMPO | K-CONT fires at every `K`; spans cost `6.9x` what they save |
    | the **RL-trained gain genome** | TEMPO | K-GENE fires against a random `Delta g` of the same norm (§P-D.36) |

    *The structural fact underneath, which changes what tonight's verdict means.* The genome has
    `n_gamma = 243,133` outputs. **`243,088` of them -- `99.98 %` -- are differentiable function
    gains whose measured static leverage is `sd = 0.0006` nats** (§P-D.13), against a noise floor
    `sigma_e` of `0.0037` to `0.0108`. They cannot move the loss even in principle. The genome's
    real action space is **45 numbers**: 4 learning-rate multipliers, 34 morphology
    pre-activations, 7 mixture weights.

    In the experiment that just fired K-GENE, those three live levers stood as follows:

    | lever | numbers | what it did tonight |
    |---|---|---|
    | mixture over streams | 7 | **structurally inert** -- every run logged `streams ['text']`, so seven weights selected among one available stream |
    | learning rate | 4 | **used, and used against the agent** -- monotone throttling on both seeds (§P-D.33) |
    | morphology | 34 | used, front-loaded, seed-dependent, and **never compared against a schedule** (K-MORPH unrun) |

    So the criterion fired on a configuration in which one of the genome's three live levers was
    **disabled by my own experimental setup**, one was actively harmful, and one is untested.

    *And the repository had already recorded why that matters.* 00 §0.1, from ANLAGE: "the
    data-mixture programme was noise on a homogeneous corpus; the learning-rate axis moved `0.09`
    nats." The mixture axis was measured to need **heterogeneity**, and the genome probe was then
    run on a single homogeneous stream. That is a mis-test, it is mine, and it was avoidable by
    reading a line in the document that motivates the design.

    *The genome's actual claim was never tested at all.* K-GENE asks whether the genome makes
    **same-task loss** fall faster per FLOP -- the weakest reading of B4. B5 is the claim that the
    genome is "the slow memory of *how to learn here*" that **transfers**. K-TRANSFER, K-XFER,
    K-BLEND and G-ACT are all `NOT EVALUATED`, never run. A mechanism can fail to accelerate a
    corpus it has already converged on and still carry what B5 claims; nothing measured so far
    speaks to that either way.

    *Three axioms this evidence puts up for reset.*

    * **A1. "The genome carries gains" (05 §5.6).** `99.98 %` of its output is inert. The reset is
      that a compact slow object should carry **control** variables -- learning rates, morphology
      switches, the data mixture, schedule shape -- and no function parameters at all. That is not
      a tuning change: it deletes the rank-32 expression map's `11.3M` parameters and takes
      `d_g` from `512` to order `16`, which is also where §P-D.9 measured the search to work.
    * **A2. "The reward is the agent's own loss improvement over the next episode" (06 §6.3).** It
      is gameable and it was gamed: the policy throttled the learning rate on both seeds
      (§P-D.33), and the probe that estimates it is dominated by an artefact seven times the
      signal (§P-D.28). A reward that can be won by **learning more slowly** is the wrong reward.
      The reset is a signal that cannot be won that way -- solve rate on the verifiable curriculum,
      or improvement measured on a segment the agent was not allowed to train on.
    * **A3. "Prediction loss is the target."** The brief itself contradicts this: B11 asks for
      capability, and everything here is measured in nats. At this scale the two correlate, but
      the *mechanisms* under test are supposed to pay in **adaptation**, which nats on one corpus
      cannot see.

    *What the data positively supports as the drivers, as opposed to what has been assumed.*
    Compute through ordinary gradient descent (`0.050` nats per doubling, §P-D.5 -- the rate every
    mechanism is charged against); the architecture itself, a `3:1` gated-delta/attention hybrid,
    which is what the strongest small open models use; the learning-rate schedule, whose `0.09`
    nats is the **largest** leverage any compact object has shown anywhere in this repository --
    and which already has a good hand-tuned answer, so the genome's one real lever is the one
    least in need of a controller; and one long life over many short ones (`0.115` nats). Untested
    here but field-supported: heterogeneity with a learnability-gated curriculum, and RL on
    sparse **verifiable outcome** rewards, both of which exist in this repository
    (`soma/openended/tasks.py`, `TaskArchive`, `tempo/envs.py`) and were used in none of tonight's
    runs.

    *The order the remaining work should take, cheapest and sharpest first.*

    1. **Test B5 on the lives that already exist.** `--load life.pt` and `--reset-genome` are
       implemented; four trained lives came out of tonight. The contrast is the same weights on a
       shifted stream, genome **kept** against genome **reset**. No compute confound, no probe
       overhead, no search cost -- only "does this object carry anything". It is the sharpest test
       of the genome available and it costs about an hour.
    2. **Re-run K-GENE with the mixture lever live** (`--streams all`): seven heterogeneous
       generated streams, verifiable rewards in the reward, the curriculum on. Same arms, same
       criterion. If it fires there too, the genome is refuted on its own ground.
    3. ~~**The `reinforce` arm**, which separates "the proposal is worthless" from "the search
       costs more than the proposal is worth" (§P-D.36).~~ **Withdrawn in §P-D.39: §P-D.8 already
       measured that estimator 1's policy never moves, so the arm would have been `c_random` with
       a differently shaped noise source at a cost of four hours.**

    *And what would close the line.* If (1) and (2) both come back null, then four models and six
    mechanisms have said the same thing, and the honest conclusion is that the drivers are the
    substrate -- architecture, compute, data, and RL on verifiable outcomes -- not a sixth compact
    slow channel. That conclusion should then be written as the result, not treated as a failure
    to be engineered around.

38. **K-GENE's verdict is robust to the budget, and the deficit DECAYS by a factor of twenty
    across the life. That is the signature of a compute tax being amortised, not of a harmful
    mechanism.** *2026-09-06. A robustness check that changed the reading.*

    §P-D.36 scored the contrast at a single common budget. Scoring at one point is fragile when a
    curve has eight to twelve evaluations and is not monotone, so the paired difference was swept
    across the whole overlapping range:

    | `F` (ref-ep) | s0 | s1 | paired mean |
    |---|---|---|---|
    | 3.85 | +0.2199 | +0.2214 | **+0.2207** |
    | 9.75 | +0.1852 | +0.1226 | +0.1539 |
    | 17.62 | +0.1420 | +0.0348 | +0.0884 |
    | 25.49 | +0.0608 | +0.0318 | +0.0463 |
    | 33.36 | +0.0277 | +0.0343 | +0.0310 |
    | 39.26 | -0.0022 | +0.0199 | +0.0089 |
    | 41.23 | +0.0072 | +0.0181 | +0.0127 |

    **The mechanism is worse at all 21 budgets sampled, in both seeds at 19 of them.** The
    verdict does not depend on where it is scored, which is what the check was for.

    *But the magnitude is the finding.* The deficit falls from `+0.221` to `+0.009` -- a factor of
    **twenty-four** -- monotonically over the measured range. A mechanism that is *harmful* does
    not do that; its damage compounds or holds. A mechanism paying a **fixed compute tax** does
    exactly this: it starts far behind because it spent `21.3 %` of its budget on probes instead
    of steps (§P-D.35), and closes as the loss curve flattens and the control's extra steps buy
    progressively less. The shape of the gap is the shape of the overhead being amortised.

    *What that does and does not license.* It does **not** license "it will win at larger
    budgets" -- §P-D.31 is the standing lesson about extrapolating a decaying series, and the last
    two points are not even monotone (`+0.0089` then `+0.0127`). What it licenses is a sharper
    statement of the verdict: **at every budget measured, the RL-trained genome is behind a random
    `Delta g` of the same norm by approximately its own search cost, and no more.** Whether the
    policy is worth anything at all is therefore *still* the open question -- it is bounded above
    by the overhead and below by nothing yet measured.

    *Two experiments this makes concrete, and one of them was already queued for the right
    reason.* Estimator 1 (`--estimator reinforce`) carries **no probes**: if the deficit is the
    search cost, that arm should start without it, and any difference from `c_random` is then the
    learned proposal alone. And running the same contrast to `F = 80` would say whether the
    decaying gap crosses zero or asymptotes above it -- a direct test of "the search amortises"
    against "the search never pays for itself", at `2x` tonight's cost per run.

    *A caution recorded against my own first reading.* A three-arm comparison at `F = 38.00`
    showed `tempo` ahead of `c_random` on seed 0 by `-0.0058`, and for some minutes I took that as
    the sign flipping. It was not: seed 0's curve happens to cross near `F = 37-39` and seed 1's
    never does, so a single-seed slice at that budget is the one place in the whole range where
    the picture looks reversed. The sweep is the reason that did not become an entry. **Scoring a
    verdict at one budget, on one seed, at a point chosen by whichever arm finished shortest, is
    how a robust result gets reported backwards.**

39. **The `reinforce` arm is withdrawn before it ran, and the reason reframes the whole
    difficulty: the policy has about ten noisy samples with which to fit a 32-dimensional
    action.** *2026-09-06. A recommendation of mine from four hours ago, reversed by a
    measurement already in this ledger that I did not re-read before making it.*

    §P-D.36 and §P-D.37 both nominated estimator 1 as the decisive next arm, on the argument that
    it carries no probes and so would separate "the learned proposal is worthless" from "the
    search costs more than the proposal is worth". The argument is sound. It is also answered
    already, twice, in entries written before tonight:

    * §P-D.8 measured estimator 1 over ten TINY episodes: `sigma` sits at its initialisation
      `0.1910` at **every** boundary. "One REINFORCE step per boundary at `eta_2 = 1e-4` is
      roughly nothing over the twenty episodes a life contains." **The policy does not move.**
    * §P-D.10 measured why it cannot: the reward is the episode-to-episode improvement, which is
      dominated by ordinary training progress. Observed rewards spread about `0.06` in `L_check`
      units around a small mean, against a genome effect the inherited record puts one to two
      orders of magnitude lower. REINFORCE averages that down as `sqrt(n)`, and `n = 20`.

    So an arm at `--estimator reinforce` would have produced, at a cost of four hours, a genome
    whose policy never moved -- **`c_random` with a differently shaped noise source**. It would
    have answered nothing.

    *The reframing, which is worth more than the arm.* Put the two measurements together with
    tonight's and the difficulty stops being about estimators:

    | | |
    |---|---|
    | action dimension | `d_g = 32` continuous |
    | policy samples available in one life | `8` to `20` boundaries |
    | reward effect size | `~0.001` to `0.01` nats |
    | reward noise | `~0.06` nats |

    **Ten noisy samples to fit a 32-dimensional continuous policy is not a hard learning problem,
    it is an impossible one**, and no choice of estimator repairs it. Estimator 2 buys a
    controlled signal (common checkpoint, CRN, null probe) and pays `21.3 %` of the budget for it;
    estimator 1 is free and gets a signal buried in training progress. Neither changes the sample
    count. That is the structural fact, and every result tonight is consistent with it.

    *Which makes one further thing clear about where headroom is not.* The genome's largest live
    lever is the learning-rate role, and 00 §0.1's `0.09` nats for the learning-rate axis is the
    **range** of that axis -- best against worst -- not headroom above a **tuned** default. The
    D2Z schedule is already tuned. A search whose biggest lever points at an already-good setting
    has nothing to win there, which is consistent with the genome having moved it and moved it
    *wrongly* (§P-D.33).

    *So the levers with possible headroom are the two that were never given a chance*: morphology
    timing, which K-MORPH has not run, and the **data mixture**, which had seven weights choosing
    among **one** stream in every run tonight (§P-D.37). 00 §0.1 recorded the mixture programme as
    "noise on a homogeneous corpus" -- which is a statement about homogeneous corpora, and the
    heterogeneous case has never been tried here. That is the next experiment, and it is the first
    one in this project that would give the genome a lever with measurable headroom to pull.

40. **The seven-stream local blend is not runnable on this machine, and the two-stream one is.
    Measured before committing a night to either.** *2026-09-06.*

    | stream set | streams | seconds per step | against text |
    |---|---|---|---|
    | `text` | 1 | `~0.80` | 1x |
    | `tasks` | 2 (`text`, `task`) | `2.53`, `2.53`, `2.75` | **3.2x** |
    | `all` | 7 (`control, interleave, pattern, speech, task, text, vision`) | **> 30** | **> 33x** |

    (The `all` figure is a bound: launched at 03:10:25 and killed at 03:15:30 with no step-10 line
    written, so fewer than ten steps in 305 seconds including start-up. Same shape, same
    micro-batch, same card, nothing else running.)

    *What that costs the specification.* R8 (K-BLEND) and R9 (K-XFER) are defined on the
    seven-stream blend of 02 §2.9. At `> 30` s per step, one 20-episode life at `N = 500` is
    `10,000` steps, or **more than 80 hours**; K-BLEND additionally needs modality-specific agents
    and K-XFER four transfer pairs. **The blend as specified is weeks of card, and K-BLEND,
    K-XFER and G-ACT are therefore not evaluable on this machine under the current protocol.**
    That is a feasibility finding about 02 §2.9, not a correctness one -- and it lands squarely on
    the choice §2.9 was proudest of, that every modality is *generated* locally so that nothing
    need be downloaded. Nothing was downloaded, and the generation is the cost.

    *Where the cost is, and the fix that is not being applied tonight.* The generators run on the
    **CPU inside the step loop**, `B = 32` windows at a time: 32x32 scenes rendered per window,
    tone waveforms synthesised and filtered per window, Lorenz and logistic trajectories
    integrated per window. None of it touches the GPU, which explains the card sitting at `35 %`
    while the step rate collapsed. Every generator is documented as **deterministic from a seed**
    (I10), so the whole blend can be **pre-generated once and cached** exactly as the text corpus
    already is -- moving the cost out of the loop and paying it once. That is the right fix, it is
    a day's work, and it is registered rather than started, because tonight has a card and a
    question and the question does not need seven streams.

    *What is affordable, and what it buys.* `tasks` at `3.2x` is the configuration the argument
    actually needs (§P-D.39): text plus the reserved-alphabet verifier families, through the same
    codec and the same tokenizer, so only the **data** changes -- and with it, for the first time
    in this project, the genome's mixture role has more than one stream to choose between, the
    fitness carries exact verifier rewards, and the learnability-gated curriculum is live. Two
    streams is not the blend 02 §2.9 describes. It is the smallest configuration in which the
    heterogeneity that 00 §0.1 says the mixture axis requires actually exists.

41. **The genome's first strongly positive result, and the measurement error that makes it the
    wrong number.** *2026-09-06, `runs/tempo/XFER-tsk1-dg32`, both seeds, 8.1 h of card.*

    Both arms started from the same `life.pt` -- same weights, same optimiser state, same fast
    state, same windows in the same order -- and differed in exactly one object: `g`. One kept the
    genome its life had written (`|g - g_0| = 4.434` and `4.908`); the other had `g <- g_0`, the
    identity gains, with the switch pattern frozen so it kept the grown shape. Then both continued
    on a shifted stream set: text **plus** the reserved-alphabet verifier families. No probes, no
    search, no growth difference, **no compute confound of any kind.**

    | n | seed 0 keep | seed 0 reset | diff | seed 1 keep | seed 1 reset | diff |
    |---|---|---|---|---|---|---|
    | 10 | 1.8696 | 2.0212 | **-0.1515** | 1.8771 | 2.0155 | **-0.1384** |
    | 20 | 1.9305 | 2.1167 | -0.1862 | 1.8964 | 2.0676 | -0.1712 |
    | 40 | 1.9668 | 2.1891 | -0.2223 | 1.9163 | 2.1222 | -0.2059 |
    | 60 | 1.9616 | 2.1500 | -0.1884 | 1.8947 | 2.0934 | -0.1987 |

    Negative means the kept genome is **better**. It is better at every point of both seeds by
    `0.14` to `0.22` nats, against a bar of about `0.10` from the control arm's own spread and
    `0.011` from `sigma_e`. Nothing else in this project has produced an effect of that size in
    the mechanism's favour.

    *And it is the wrong measurement.* `fresh` and `continuing` are the **text** strata. The
    driver filtered `e.get("text")` and dropped every other stream, so what is tabulated above is
    how much of the **old** domain each arm kept while training on the new one. That is
    **retention**. K-TRANSFER asks how fast the agent **adapts to the new domain**, which is the
    task stream, which was evaluated and thrown away. I built a transfer test that does not
    measure transfer.

    *One trivial explanation refuted, on the data.* The genome sets the stream mixture, so "the
    kept genome simply draws more text" would explain a smaller text degradation arithmetically.
    Expressed mixtures, renormalised over the two streams present:

    | seed | keep | reset | rand (same norm) |
    |---|---|---|---|
    | 0 | **0.432** text | 0.500 | 0.062 |
    | 1 | **0.876** text | 0.500 | 0.039 |

    Seed 0's kept genome draws **less** text than the reset one and still degrades less; seed 1's
    draws more. The seeds disagree in the direction of the mixture and agree in the outcome, so
    the mixture is not the common cause.

    *One trivial explanation that survives, and it is §P-D.33 wearing a different hat.* The kept
    genomes' learning-rate multipliers are **all below unity** -- seed 0 `[0.53, 0.79, 0.69,
    0.57]`, seed 1 `[0.74, 0.62, 0.86, 0.77]` -- while the reset arm's are exactly `[1, 1, 1, 1]`
    by construction. **A damped learner forgets less under a distribution shift.** The throttling
    that made the genome lose K-GENE is the same throttling that would make it "win" a retention
    measurement, and on this evidence the two are the same fact seen from opposite sides.

    *A second confound, structural.* The weights were trained **with** the kept gains folded in,
    so `g <- g_0` removes gains a co-adapted network depends on. 00 §0.1 recorded exactly this
    shape from ANLAGE -- an inherited scale profile "worse at the instant of birth and better by
    `0.014` after 2.5M tokens". The `reset` arm is that instant of birth. It is a real comparison
    and it is the one 10 §10.2 registers, but it cannot distinguish "this genome carries
    something" from "any gains beat no gains on a network that grew up with gains".

    *What is running instead.* All three arms -- `keep`, `reset`, and a **random genome at the
    same distance from the prior** -- on both seeds, recording **every stream's** evaluation
    rather than text alone, plus each arm's expressed mixture and learning-rate multipliers. Then
    the question separates: better on **text** alone is retention, and most likely damping; better
    on **task** is adaptation, which is what B5 claims and what nothing here has yet measured.

    *An operational failure to record beside it.* The first pass finished at 11:37 and the card sat
    **idle for twelve hours** before the follow-up was launched, because the session was waiting on
    a result that had already arrived. A run that completes and is not noticed is indistinguishable
    from a run that is still going, and nothing in this setup distinguishes them. The monitor
    watches `runs/tempo/genome.log`, which this experiment does not write to.

42. **Growth is a compute investment with a crossover near `F = 15`, and both of my earlier
    readings of it were scored at a single budget.** *2026-09-07.*

    §P-D.38 established that a contrast between arms of different cost must be swept across
    budgets, not scored at one. I then said, of control (a), that "the `tau_2` stack as a whole --
    growth plus perturbation -- does beat doing nothing", on the strength of **one budget**
    (`F = 38`) and **one seed**. Swept, seed 0, where `a_gamma0` runs to `F = 38`:

    | `F` | `a_gamma0` (fixed primordium) | `c_random` (grows) | `tempo` (grows) | best grower - fixed |
    |---|---|---|---|---|
    | 4 | 2.0574 | 2.2230 | 2.4403 | **+0.1656** fixed wins |
    | 8 | 1.9210 | 1.9905 | 2.1975 | +0.0695 fixed wins |
    | 12 | 1.8689 | 1.8882 | 2.0340 | +0.0194 fixed wins |
    | **14** | 1.8475 | 1.8478 | 1.9979 | **+0.0004** the crossover |
    | 16 | 1.8318 | 1.8227 | 1.9718 | -0.0091 growth wins |
    | 24 | 1.8026 | 1.7944 | 1.8732 | -0.0082 growth wins |
    | 32 | 1.7867 | 1.7620 | 1.7917 | **-0.0247** growth wins |
    | 38 | 1.7782 | 1.7635 | 1.7577 | -0.0204 growth wins |

    And at every budget both seeds of every arm reached (`F <= 12`, because `a_gamma0`'s seed 1
    was stopped at episode 7 to free the card):

    | `F` | `a_gamma0` | `c_random` | `tempo` | best grower - fixed |
    |---|---|---|---|---|
    | 4 | 2.0545 | 2.2187 | 2.4356 | +0.1641 fixed wins |
    | 8 | 1.9222 | 2.0025 | 2.1862 | +0.0803 fixed wins |
    | 12 | 1.8668 | 1.9165 | 2.0227 | +0.0497 fixed wins |

    *The finding.* Growth is exactly what its literature says it is at this token-per-parameter
    ratio: **an investment**. It costs `0.17` nats at `F = 4`, breaks even at `F ~ 15`, and is
    ahead by `0.02` at `F = 32`. Neither of my two earlier statements about it was wrong at the
    budget it was made at, and both were wrong as statements about growth.

    *What the two-seed evidence actually covers.* Only `F <= 12`, where growth **loses**. The
    crossover and everything past it rests on **one seed of the control arm**, because I stopped
    `a_gamma0-s1` at episode 7 on the argument that conjunct (a) could not change a verdict K-GENE
    had already reached. That argument was right about the verdict and wrong about this: the run I
    stopped is the one that would have made the growth result a two-seed claim. Resuming it costs
    about two hours and its checkpoint is intact (`rc = 42`, resumable).

    *Two things this settles about the shape of the whole night.* First, `tempo` is worse than
    `c_random` at **every** budget in the table, which is §P-D.38's conclusion arriving a second
    time from a different arm. Second, the deficit that K-GENE measured is now legible as two
    separate compute investments stacked: `~21 %` on probes that buy nothing, and a growth
    programme that does not break even until `F ~ 15`. `tempo` pays both; `c_random` pays only the
    second; `a_gamma0` pays neither and leads until the second one matures.

    *And it makes K-MORPH the next criterion worth running,* not K-GENE's remaining conjunct. The
    arms exist (`schedule`, `oneshot`, `flat`), the question -- does RL-timed growth beat a fixed
    schedule -- is the one place the machinery has shown a positive return, and the crossover
    measured here says any such comparison must be run to at least `F = 30` and **swept**, because
    a K-MORPH scored at `F = 12` would report the opposite of one scored at `F = 32`.

43. **Where the project stands, what drives learning on the evidence, and the fork that decides
    the fifth model. Written at the user's request as a pause, before any further design.**
    *2026-09-07.*

    *The state of TEMPO's five mechanisms.*

    | mechanism | claim | measured | status |
    |---|---|---|---|
    | span / continuous interface (B8) | one continuous output, `K` tokens a step | K-CONT fires at every `K`: `+0.5465` at `K = 4` (4 seeds), `+0.0989` at `K = 1`; spans cost `6.9x` what they save | **dead** |
    | gain genome + RL (B4) | 512 numbers written by RL make the agent learn better | `99.98 %` of its output has leverage below noise; loses to a random `Delta g` of the same norm by its own `21 %` search cost; learned to throttle the learning rate | **dead as designed** |
    | genome transfer (B5) | carries "how to learn here" to "there" | never measured correctly -- the one pass measured retention, not adaptation; rerun in flight | **open** |
    | latent morphology (B3) | born small, grow when it pays | growth pays: `-0.17` at `F = 4`, break-even `F ~ 15`, `+0.02` at `F = 32`; but *random* timing beats *learned* timing | **positive for the mechanism, negative for its controller** |
    | criticality (B6) | hold the edge of chaos | drift is bounded (`0.16` against a band of `0.5`) and the rule is too weak to be what bounds it | **decorative** |

    Four models, six compact-slow-channel mechanisms, six losses at matched compute, one positive
    (growth), and that one's controller loses to a coin.

    *The one explanation.* Plain gradient descent returns **`0.050` nats per doubling** of compute
    here (§P-D.5), and every mechanism is charged against that rate. Their leverage: gains `0.0006`
    (below noise); the learning rate `0.09` of range but no headroom above a tuned schedule; growth
    `+0.02` after a `0.17` investment; spans `-0.10` to `-0.55`; the probe search `21 %` for nothing
    measurable. Leverage smaller than overhead loses at matched compute, and that is not a property
    of any estimator or configuration -- those were varied -- it is the ratio. Nor does it improve
    with scale: the compute-return exponent barely moves, so a mechanism that loses at 50M has no
    argument for 5B. The exceptions are the two things that do not appear in same-task loss at all
    -- transfer across domains, and a mixture over heterogeneous data -- which are exactly the two
    this machine can barely measure (§P-D.40).

    *What drives learning, on this repository's own numbers.*

    | driver | where | size |
    |---|---|---|
    | compute through gradient descent on the task loss | §P-D.5 | `0.050` nats / doubling -- everything else is charged against it |
    | the optimiser and its schedule (Muon, D2Z) | ANLAGE; the control arm of every contrast | `0.09` nats across the LR axis, the largest compact-object leverage measured here |
    | representation granularity | K-CONT | `0.55` nats between `K = 4` and `K = 1`, the largest number in the project |
    | capacity growth timed to the token budget | §P-D.42 | `+0.02` after break-even |
    | the 3:1 gated-delta / attention hybrid | the control arm | won 6 of 6 |
    | heterogeneity with a curriculum; RL on verifiable outcomes | **untested here** | the field's source of *capability* as distinct from loss |

    Not drivers, measured: channels carrying weights or gains; populations; assimilation;
    continuous interfaces; criticality control; RL-trained update rules fitted inside one life.

    *The hidden rules every design obeyed, and which two are the load-bearing errors.* All four
    designs assumed: the interesting object is a compact slow channel beside the weights (R1);
    improvement is corpus loss (R2); everything trains from scratch on this card (R3); one agent,
    alone, no external models (R4); mechanisms are tested at one size at this card's scale (R5);
    the meta-signal is the agent's own loss improvement (R6); modalities are generated locally
    (R7); the full architecture is designed before anything is measured (R8); the design must be
    novel in mechanism (R9). The record indicts R1 (six losses), R6 (§P-D.33), R7 (§P-D.40) and R9
    (the unoriginal control arm won every time). **R3 and R8 are the load-bearing ones.** R3 makes
    B11 unreachable by arithmetic -- 78M on TinyStories against 9B on trillions of tokens -- so no
    mechanism could ever have satisfied the brief under it. R8 is why each design was a 700-line
    bet placed before a number existed, and why each failed at its axioms rather than at a detail.

    *The three options, with what each can and cannot reach.*

    1. **PAIDEIA -- the regimen is the research.** Drop R3/R4: the substrate is a downloaded
       Qwen3.5 hybrid, the *same* 3:1 gated-delta family TEMPO built from scratch, so `tau_0` is
       already real and load-bearing. The agent lives on the verifier-task stream that already
       exists (`soma/openended/tasks.py`, `TaskArchive`, `tempo/envs.py`), trained by GRPO on LoRA.
       `tau_1` is the LoRA weights under RL; `tau_2` is a **compact regimen** -- mixture,
       learnability gate, growth timing, schedule -- the 45-number control space that was the only
       live part of TEMPO's genome, now with a lever that has headroom. Growth is rank or expert
       addition at a scheduled point. Every property survives only if a 20-minute paired screen
       says so. **The only option that can reach B11.** Removes the `0.55`-nat interface tax and
       the `21 %` search tax on day one. Needs one download, which was a first-build constraint of
       TEMPO and is now the decision.
    2. **Scaling-law screening -- keep the substrate, change the question.** Vary R5/R8: every
       candidate mechanism is screened at three widths, two seeds, `F = 15`, twenty minutes each,
       and the verdict is the **slope** of its benefit against size. Fully local. Cannot reach B11,
       and that must be said plainly; its product is *which mechanisms are worth scaling*, which is
       the prerequisite for option 1's choices and the discipline TEMPO's protocol lacked.
    3. **Close the ledger.** Finish the transfer rerun (in flight) and the cross-life
       meta-learning run (built, smoke-tested, ~4 h). If both are null: six mechanisms, four models,
       one conclusion, with the measurement apparatus -- pre-registered criteria, matched-compute
       controls, budget sweeps, one verdict rule, every correction on record -- as the reusable
       artefact. High as science; zero toward B11.

    *The recommendation.* 3 first, because it is cheap and in flight and closes the genome
    honestly rather than by attrition. Then 1, with 2 as its screening discipline. The fork is one
    decision and it is the user's: **does the fifth model start from a downloaded checkpoint?** If
    not, the ceiling is option 2 and B11 comes off the table explicitly. If so, the first four
    screens are already specified by the machinery in this repository: GRPO on `TaskArchive` from
    the base for `sigma_e` and a baseline; `+ growth` against fixed rank at matched compute;
    `+ learnability-gated curriculum` against uniform; `+ fast state carried across episodes`
    against reset -- the `tau_0` claim, testable in an afternoon for the first time.

    *"Faster experiments", concretely.* Tonight's runs took two to eight hours because of `F = 40`
    budgets, `50M` grown shapes, 128-window evaluations with task rollouts, and the seven-stream
    blend. The data says the effects are visible far cheaper: the K-CONT gap at `13M`, the growth
    crossover by `F = 15`, the K-GENE deficit largest at `F < 10`. Twenty-minute screens at `13M`,
    `F = 15`, `n_eval = 32`, with the generated streams pre-cached (§P-D.40's fix), are feasible
    with what exists. The rule from continuous delivery -- no run over thirty minutes without a
    twenty-minute screen that justified it -- is adopted as protocol for whatever comes next.

    *A note on the run in flight.* The transfer rerun's first arm has been computing for over
    three hours at `15.6 GB` and `100 %` utilisation, against `2.5` h for the same arm in the first
    pass with twice the evaluations. It is not the paging signature (utilisation would fall), it is
    slower than it should be, and the cause is not yet known. The watchdog covers a stall; the
    throughput question is open until the arm lands.

44. **The genomic-bottleneck paper, read end to end against what this repository did with it.
    Three mistakes of ours, and two structural reasons the paper's result does not carry to
    language modelling on the lifetime criterion.** *2026-09-07. Shuvaev, Lachi, Koulakov &
    Zador, PNAS 121(38):e2409160121, publisher PDF, all 12 pages, text extracted with PyMuPDF;
    page numbers below are the PDF's.*

    *What the paper actually does (Methods, pp. 10-11; Algorithm 1, p. 11).*

    | element | the paper | what this repository built |
    |---|---|---|
    | g-network | a **separate tiny MLP per layer** (and per bias vector), `20 - H - 10 - 1`, `H in {5, 20, 30, 50}`, 613 to 14,958 parameters in total (Tables 1-2) | GNOMON: **one** per-entry MLP for every weight of every layer, plus 139,360 *learnable* identity coordinates `Z`; ANLAGE: a neural-kernel genome with acquired latents `d_z = 16` per unit |
    | the labels the g-network reads | **fixed** binary codes, and *where a layer has a topology the code is that topology*: MNIST input pixels are **5-bit Gray `x` + 5-bit Gray `y`**; CNN kernels are `2 + 2` Gray bits of kernel position; hidden units are plain binary indices "since their order is of no particular importance"; outputs are one-hot (p. 10) | GNOMON/ANLAGE: Gray over **frequency rank** for tokens, Gray over **relative depth** for layers, and *learned* vectors `Z` for everything else, because channel indices have no topology |
    | how the g-network is fitted | **MSE regression** onto the current weights, by backprop on the g-network alone; end-to-end training of the p-network *through* the g-network was tried and found "inefficient as it involves generation of the entire set of weight matrices for each mini-batch" (bioRxiv v2) | GNOMON trained `Z` **through** the decoder every step; ANLAGE's R4-probe trained the decode end to end -- and both collapsed (rank 1; `z -> infinity`) |
    | the loop | **500 generations**; each: train the p-network briefly (MNIST 1/6 epoch; CIFAR **10 full epochs**), refit the g-networks, then **regenerate**: `W_n = eps(n) W_{n-1} + (1 - eps(n)) W_hat_{n-1}`, `eps(n) = exp(-n/20)` (Eq. 3, p. 11). After `~60` generations `eps < 0.05`: for 440 of 500 rounds the p-network is **re-decoded from the genome before every training burst** and is never allowed to leave the decodable set | ANLAGE R4b: refit every 100 steps and a **pull** toward the cached decode at `eta = 0.01` -- **`eps` was never annealed toward 0**; one 25M-token lifetime; the network drifted 0.43-0.65 between refits and the decode never fell below 11 nats |
    | compute | CIFAR: `500 x 10 = 5,000` epochs of p-network training, roughly `50-100x` a normal run; the authors "have not attempted to optimize this approach to compete with state-of-the-art benchmarks" (p. 10) | every criterion here is at **matched compute**, and ANLAGE's axiom A3 forbids a decode inside the training step for throughput |
    | the claim | **at-birth** performance and transfer; explicitly **not** learning speed: "genomic compression did not affect the learning trajectory; the only speedup was due to the higher initial performance, as though the p-network was hot-started" (p. 4, Fig. 3E); "the present work ... is focused not on faster learning but rather on enhanced initial performance" (p. 9) | every K-criterion here asks whether the genome makes the **lifetime** better per FLOP |
    | tasks and sizes | MNIST MLP 636,010 params; CIFAR-10 9-layer all-conv 1.37M; BeamRider dueling DQN 3.3M; HalfCheetah 6,092 params; ImageNet/Caltech in the SI only. **No sequence model, no recurrence, no attention, nothing above 3.3M** | language models of 2.9M-78M |
    | evolution | none: "both the inner and outer loops are optimized by gradient descent" (p. 2); "a form of Lamarckian evolution, and is, as such, biologically unrealistic" (p. 9) | SOMA put ES in the loop; GNOMON put MAP-Elites in the loop |
    | statistics | 4 independent runs for HalfCheetah (Fig. 5E); "SD of the mean" shading on the SVHN transfer; ANOVA `p = 0.015` and `0.0005` for the two positive transfer layers; MNIST and CIFAR curves with no stated seed count | `sigma_e` under CRN, 3-4 seeds, paired intervals, a `3 sigma_e` bar |

    *The paper's numbers, exactly.* MNIST: 94 % at birth at 322x (GN30), 91 % at 470x, 79-80 %
    at 1,038x, against 98 % trained and 10 % random; a **shuffle control** at chance but with
    "markedly accelerated" learning (Fig. 2D). CIFAR-10: 76 % at 92x (GN50), 72 % at 146x, 56 %
    at 361x, against 89 % trained -- a 13-point gap at the mildest compression (Fig. 3C).
    BeamRider: "nearly perfect" at 492x and "excellent" to 3,570x. HalfCheetah: **4.9x**
    compression only. Transfer CIFAR->SVHN: the whole compressed network is "indistinguishable"
    from copying the trained weights; **layer 1 alone** through the bottleneck reaches 50 % in 3.2
    epochs against 4.8 by direct copy, a 1.5x; layers 8-9 through the bottleneck are **worse than
    random**; MNIST->F-MNIST is worse than random ("as though the network first had to unlearn
    the MNIST dataset", p. 4). HalfCheetah under 1.5x gravity: the compressed policy re-adapts
    faster than the copied one, because the copied one had reward-hacked (tumbling) and "must
    first unlearn these unconventional strategies" (p. 7).

    *Mistakes of ours, three, each identifiable against the text.*

    1. **GNOMON trained through the decoder.** The paper fits the g-network by regression to a
       weight matrix that gradient descent produced freely, and it says in so many words that
       backpropagating the task loss through the decoder was tried and abandoned. GNOMON made the
       identity coordinates learnable and pushed the task gradient through a saturating map every
       step; the runaway norm and the 38x rank collapse (`f083282`) are what the paper's method
       avoids by construction. ANLAGE's R4-probe repeated the same experiment deliberately, as a
       capacity test, and got the same collapse. **Self-inflicted, and known to the source.**
    2. **ANLAGE never ran the paper's loop.** R4b ran a *pull* (`eps = 0.01`, fixed) with refits
       every 100 steps inside one ordinary lifetime, and concluded "assimilation is a cost". The
       paper anneals `eps` to zero over `~60` generations and then **regenerates the whole network
       from the genome 440 more times**, spending `50-100x` a normal training budget to force the
       phenotype onto the decodable set. That regime was never tried here. It could not have been
       tried under axiom A3 and the matched-compute rule, which is a legitimate choice -- but the
       ledger should not say the PNAS loop failed on language when what failed was a fixed-`eps`
       pull at `1x` compute. This is the one place where a faithful replication is still owed.
    3. **We invented topologies for coordinates that have none, and the one real topology was the
       one that worked.** The paper's compression lives on the layers whose *input side has a
       coordinate system* -- pixels with `(x, y)`, kernels with position -- where a weight is a
       smooth function of position (the recovered filters are Gabor-like, Fig. 2F). Its own
       hidden units get arbitrary binary indices and are, in effect, memorised. A transformer's
       `d_model` channels, heads and FFN units are the arbitrary case with no pixel case beside
       it. GNOMON's frequency-rank token code, the **only** structured label a vocabulary admits,
       was GNOMON's single largest positive (`+0.0999`, §00.2). Everything else was a label
       invented for a coordinate that carries no rule.

    *Structural reasons it does not carry, two, neither of which is a mistake.*

    * **The compression measures the task's information content, not the method.** MNIST at
      636k parameters is grossly over-parameterised for a task a 2k-parameter linear model solves
      at 92 %; the paper's own harder task already costs 13 points at 92x. ANLAGE §11.2 derived
      the information count for a language model's flat-spectrum matrices and it predicted the
      7-20 % residuals R4 then measured. There is no redundancy to remove, so there is no ratio.
    * **The paper's benefit is at birth, and it reports that the lifetime is unchanged.** Fig. 3E
      shifts every compressed curve onto the tabula-rasa curve. That is the paper saying, of its
      own method, what ANLAGE R4 INIT measured on the scale profile (worse at birth, `-0.014`
      after 2.5M tokens) and what every K-criterion here found: **the bottleneck does not make a
      network learn faster.** Our lifetime nulls *agree* with the paper. The disagreement is only
      on at-birth performance, and the two points above account for it.

    *What the paper claims that this repository has never tested.* Transfer. The positive results
    are specific and small -- layer 1 of a CNN, 1.5x; a 6k-parameter policy under changed gravity
    -- and negative for late layers and for MNIST->F-MNIST, with the authors' own rule "for complex
    but not for simple test problems". The one transfer measurement here (§P-D.41) measured
    retention of the old domain, not adaptation to the new. A faithful test is: a small language
    model, the PNAS loop **as specified** (`eps -> 0`, hundreds of regenerations, at-birth loss as
    the metric), then a corpus shift, genome-mediated initialisation against copied weights on the
    **new** corpus's learning curve. It is affordable at 13M and it has not been run.

45. **The faithful replication is specified, built, tested and launched; the readings are on
    record before the numbers.** *2026-09-07, 12:40. Specification `docs/bottleneck/00-replication.md`;
    code `bottleneck/`; tests `tests_bottleneck/` (14, all passing on CPU in 13 s); script
    `scripts/bottleneck_loop.py`; runs `runs/bottleneck/`.*

    *What was built, and each departure from the paper stated with its reason.* One g-network per
    weight matrix, `(b − H − H₂ − 1)` with ReLU, exactly the paper's Table 1 shape with the input
    width set by the axes; one single-axis g-network per gain vector, matching its bias networks;
    `LOGIT_SCALE` copied through by its own rule for objects smaller than their g-network. Codes
    are the paper's rule applied to a language model: Gray over frequency rank on the token axis
    (`⌈log₂ 8192⌉ = 13` bits), plain binary index on every channel axis. MSE regression onto
    freely-trained weights, warm-started across generations, one Adam over all g-networks -- which
    is exactly independent regression per network, since Adam is per-parameter and the networks
    share nothing. Regeneration by Eq. 3 with `λ = 20`. Three departures: **(i)** one
    learning-rate schedule over the whole run rather than one per generation, because TEMPO
    measured the per-episode restart to dominate every per-episode number (§P-D.25, §P-D.28), and
    the paper's constant per-generation rate is closer to the former; **(ii)** `N = 200`
    generations rather than 500, with `ε(100) = 0.007` already fully in the regenerating regime;
    **(iii)** the last layer of every g-network initialised at `0.01×`, so the generation-0 decode
    is small rather than `O(1)` -- unused at `ε(0) = 1`, conditioning only. Each is in the
    specification with the reason beside it.

    *What the smoke run taught before an hour was spent.* Three generations at `10⁵` tokens each:
    only two regenerations fired, because the hook fires at the *start* of the step that reaches a
    boundary and `train` stops the moment the budget is met, so a budget of exactly
    `N · T_gen` ends on the last boundary. Fixed by one extra burst -- `(N + 1) · T_gen` for every
    arm including the control, so tokens stay matched -- which is also Algorithm 1's shape. And
    the card has the game on it again (`7.4 GB`), so the 13M run takes `--micro-batch 16` to sit
    near half of R1's `8.35 GB` peak; accumulation is exact (ANLAGE `04 §4.8`, the micro-batch identity).

    *The arms, with the ratio each actually reaches.*

    | arm | `H` | `H₂` | genome params | **ratio** | what it is |
    |---|---|---|---|---|---|
    | `GN30` | 30 | 10 | 74,569 | **174.5×** | the paper's MNIST/CIFAR structure |
    | `GN50` | 50 | 10 | 122,089 | **106.6×** | the paper's best CIFAR structure |
    | `mild` | 419 | 52 | 2,607,895 | **5.0×** | the paper's HalfCheetah ratio, solved for `H` at build time |
    | `free` | -- | -- | -- | -- | tabula rasa: same tokens, seed, windows, order |

    Note that the paper's `H = 30` gives `174×` here against `322×` on its MNIST MLP: the stage-0
    model has 108 compressed tensors against the paper's three, so the same per-network size buys
    less. The ratio is what the information argument is about, and it is recorded per arm.

    *Budget.* `201 × 10⁶` training tokens per arm (`~50` min at R1's 69k tok/s on a free card),
    plus 200 refits of 50 Adam steps and 200 decodes, measured at `~6` s per generation at 20
    steps in the smoke -- call it `~40` min of hook per genome arm. Four arms, one serial chain,
    launched **detached** so it survives a session exit (§P-D.41's lesson): roughly five hours
    free, longer shared.

    *The readings, pre-registered (00-replication §6, restated so this entry stands alone).*
    **Works**: at-birth `CE(Ŵ_N) < 6.0` at any arm -- three nats of innate competence against
    `L_null = 9.011` and the trained `1.809`. **Fails**: at-birth `> 8.5` at every arm, what one
    regression gave ANLAGE (8.53). **Between**: reported as what it is, with the shuffle control
    deciding whether the decode carries wiring or only a weight distribution. **Lifetime**:
    `CE_regen − CE_free` at matched training tokens; the paper predicts `≈ 0` ("hot-started",
    Fig. 3E); worse by `> 3σ_e = 0.009` is the cost of the constraint, measured.

    *What the first two generations of the smoke already show, and what they do not.* At `174×`
    with 20 refit steps on a network trained for `10⁵` tokens, the decode sat at `9.012` and
    `8.993` (null `9.011`), the shuffle at `9.011`, matrix residuals `0.95` and `0.89`; the
    regeneration jolt was `+0.05` at `ε = 0.95` and `+0.15` at `ε = 0.90`. That is a genome that
    has learned nothing yet blended into a network that has, and it is exactly the paper's
    generation-1 state ("if g-networks are initialized randomly, they produce p-networks that are
    far from the optimal fixed point", p. 10 -- the reason Eq. 3 exists). Nothing about the
    outcome is readable from it. The question the run answers is whether, as `ε → 0`, the
    phenotype is pulled onto the decodable set or destroyed by it.

46. **GN30 complete: the paper's method at 174× extracts 1.9 nats of innate wiring and costs
    1.6 nats of lifetime — and its last thirty-five generations collapsed under a schedule I
    chose, so the arm is kept as `v1-decay` and the chain restarted.** *2026-09-07, 15:10.
    `runs/bottleneck/GN30-tinystories-s0-seed0-v1-decay/`, 200 generations, 2.42 h (39.9 min in
    the hook), 201M training tokens.*

    *The trajectory, three phases and a collapse.*

    | gen | ε | before | **birth** | after | shuffle | resid |
    |---|---|---|---|---|---|---|
    | 1 | 0.95 | 3.39 | 8.97 | 3.44 | 9.00 | 0.864 |
    | 5 | 0.78 | 2.43 | **14.08** | 2.57 | — | 0.784 |
    | 10 | 0.61 | 2.39 | 10.30 | 2.89 | 9.15 | 0.772 |
    | 30 | 0.22 | 2.61 | 9.02 | 6.35 | 9.01 | 0.724 |
    | 70 | 0.03 | 3.42 | 7.72 | 7.34 | 9.01 | 0.641 |
    | 100 | 0.007 | 3.50 | 7.48 | 7.38 | 9.01 | 0.540 |
    | 130 | 0.0015 | 3.42 | 7.68 | 7.66 | 9.01 | 0.526 |
    | **156** | 0.0004 | 3.37 | **7.09** | 7.08 | — | 0.51 |
    | 165 | 0.0003 | 3.33 | 7.52 | 7.52 | — | 0.503 |
    | 180 | 0.0001 | 3.29 | 8.34 | 8.34 | 9.01 | 0.397 |
    | 190 | 0.0001 | 3.31 | 8.01 | 8.00 | 9.01 | 0.275 |
    | 200 | 0.0000 | **4.50** | **9.01** | 9.01 | 9.01 | **0.041** |

    (*before*: the phenotype after its 1M-token burst; *birth*: the decode alone; *after*: the
    regenerated phenotype; `L_null = 9.011`; the random genome decoded to `9.0108`.)

    **Phase 1 (1–10).** Free learning; the decode worse than uniform, `14.08` at generation 5: a
    regression explaining 22 % of the variance gives weights of the right scale and the wrong
    structure, which is confidently wrong. **Phase 2 (10–30).** The jolt grows to `+3.7` as the
    phenotype is dragged onto a null-level decode. **Phase 3 (30–165), the paper's regime.** The
    phenotype lives in a cycle -- regenerate to `~7.4`, train to `~3.4` -- and inside it the
    decode improves: `9.02 → 7.72 → 7.09` at its best (generation 156), residual `0.72 → 0.50`,
    the shuffle at **`9.0109` at every one of 17 checks**. **The collapse (165–200).** The
    residual falls to `0.041` while the decode returns to `9.01` and the trained phenotype
    degrades from `3.3` to `4.5` (final test `5.93`); at the end every role's residual is `~0`
    except `DOWN` at `0.145`.

    *The cause of the collapse, and it is mine.* The R1 recipe decays the learning rate to zero
    from `p_dec = 0.8` of the budget -- generation 160 of 200 -- and the collapse begins at
    generation ~165. With the rate gone a burst can no longer move the phenotype off its decode;
    refit-and-regenerate becomes self-referential (the genome fitting its own previous output,
    hence the vanishing residual); and Adam on a vanishing residual, which normalises tiny
    gradients to full-size steps, walks the genome away from the trained manifold with the
    phenotype following it, since nothing can pull the phenotype back. **The paper uses a constant
    per-generation rate and cannot exhibit this.** §3.1 of the specification had chosen one R1
    schedule over the whole run as "closest to the paper's regime"; the decay phase makes it the
    opposite of the paper's regime exactly where the loop is most sensitive. Corrected: warm-up,
    then a constant rate (`p_dec = 0.999`), every arm including the control. The run is kept
    under `-v1-decay` with its collapse on record, and the chain is restarted from GN30.

    *A second consequence, caught before it cost the night.* Stage 2's transfer runs read a
    source's **final** `genome.pt` and `phenotype.pt` -- which for this run are the collapsed
    state. Nine hours of transfer from a null decode would have been worthless. The loop now
    checkpoints the genome and the trained weights it was fitted to at the generation of best
    at-birth CE (§3.2, `genome_best.pt`, `phenotype_best.pt`), which is also the quantity the
    paper reports; transfer reads that state by default and records which state it used.

    *The reading, from the plateau, against the pre-registration (§6).* **Between.** The best decode
    is `7.09` and the plateau `7.3–7.7`: not below `6.0` (works) and well below `8.5` (fails);
    every plateau generation beats what one regression gave ANLAGE (`8.53`, R4). The shuffle
    control decides what the `1.9` nats is: **wiring**, not a weight distribution -- the paper's
    Fig. 2D distinction, on the paper's side. The **lifetime** is the paper's claim that fails: the
    best the constrained phenotype reaches per burst is `~3.4` against `~1.8` for a free network
    at these tokens (R1 v3 mean `1.809` at 25M; the `free` arm will give the matched number), a
    cost of `~1.6` nats, more than five hundred `σ_e`, against the paper's `≈ 0` ("hot-started",
    Fig. 3E). And the hot-start itself does not hold: from the decode at `7.4`, 1M tokens reach
    `3.4`; from a random initialisation at generation 1, 1M tokens reached `3.39`. Starting 1.6
    nats ahead ends level -- the decode's wiring does not shorten gradient descent at all. (The
    learning rate and batch differ between those two points; the `free` arm's curve at 1M tokens
    is the clean comparison and is queued.)

    *What this says about the paper's mechanism on a language model at this ratio.* The loop does
    what the paper describes: it forces the phenotype onto the decodable set (residual falling,
    decode improving, shuffle at null) and extracts far more than a single regression. What the
    paper never had to report -- because on MNIST the decodable set contains a 94 % network -- is
    the price of living on that set, and at 174× on a language model the set contains no good
    model, so the phenotype pays `1.6` nats to stay there. That is Eq. 1's trade-off,
    `J = E₀(G) + γH(G)`, measured at a fixed `H(G)` where minimising `E₀` sacrifices the lifetime.
    Whether a mild ratio changes this qualitatively is what the `5×` arm is for.

    *Cost of the restart.* GN30 re-runs under the corrected schedule (`~2.4` h), and GN50's
    140 s under the old one were discarded. Stage 2 exited on the pause flag as designed and is
    relaunched with the chain.

47. **GN30 under the corrected schedule did not reproduce GN30 under the old one where the two
    are the same configuration. The loop amplifies floating-point nondeterminism into a two-nat
    difference in the decode, and a better weight fit gave a worse decode.** *2026-09-07, 17:00.
    `runs/bottleneck/GN30-tinystories-s0-seed0/` (v2, `p_dec = 0.999`, 1.83 h) against
    `.../GN30-tinystories-s0-seed0-v1-decay/` (v1, `p_dec = 0.8`).*

    *The two runs are identical in every setting the loop reads until generation 160.* Same seed,
    same window order, same pool digest, same micro-batching, same warm-up, and the same
    constant learning rate from 2 % of the run until the decay begins at 80 % -- generation 160 --
    in v1 only. Everything before that boundary should agree, and it agrees only to the third
    decimal after the first burst (`before` `3.3934` vs `3.3908` at generation 1: bf16 autocast
    and atomics are not bitwise-deterministic on this card). The loop then does this with it:

    | gen | v1 birth | v2 birth | v1 before | v2 before | v1 resid | v2 resid |
    |---|---|---|---|---|---|---|
    | 10 | 10.30 | 10.90 | 2.39 | 2.39 | 0.772 | 0.770 |
    | 50 | 8.91 | 8.29 | 3.29 | 3.21 | 0.692 | 0.683 |
    | 70 | 7.72 | 8.05 | 3.42 | 3.45 | 0.641 | 0.620 |
    | 100 | 7.48 | 8.49 | 3.50 | 3.63 | 0.540 | 0.532 |
    | 120 | 7.53 | **10.00** | 3.46 | 3.75 | 0.517 | 0.470 |
    | 140 | 7.72 | 9.60 | 3.46 | 3.71 | 0.500 | 0.443 |
    | 156 | **7.09** | 8.09 | 3.37 | 3.68 | 0.51 | 0.42 |
    | 160 | 7.51 | 7.73 | 3.38 | 3.64 | 0.517 | 0.427 |
    | 180 | 8.34 (decay) | **10.77** | 3.29 | 3.65 | 0.397 | 0.431 |
    | 200 | 9.01 (collapsed) | 9.18 | 4.50 | 3.50 | 0.041 | 0.477 |

    Plateau means over generations 100-160: v1 **7.53**, v2 **9.56**. Best decode: v1 `7.09` at
    156, v2 `7.67` at 78 (the checkpoint stage 2 will read).

    *What the schedule fix did and did not do.* It fixed what it was for: v2's phenotype never
    collapses (final test `3.52` against v1's `5.93`) and its residual never goes self-referential
    (`0.43-0.48` throughout against v1's `0.04` at the end). §P-D.46's diagnosis of v1's collapse
    stands -- the signature is specific and absent here. But removing the decay did not make the
    loop *work*. It revealed that the loop's outcome on the decode is **high-variance across
    numerically-identical runs**: one plateaued at `7.5`, the other spent most of its second half
    at or above null, with excursions to `10.0` and `10.8` -- decodes *confidently wrong*, the
    generation-5 state recurring at generation 180.

    *And the residual points the wrong way.* v2 fits the weights **better** than v1 at every
    matched generation (`0.43` against `0.51` at 160) and decodes **worse**. This is the structural
    fact §P-D.44 predicted and this run measures: the paper's Algorithm 1 minimises **MSE on the
    weights** as a proxy for its stated objective `E₀(G)`, the *performance* of the decode (Eq. 1),
    and on a network whose matrices have a flat spectrum the proxy is not aligned with the target.
    A fit that explains 55 % of the variance can decode to `7.5` or to `10.8` depending on *which*
    55 % it captured, and nothing in the loop selects for the good one: the regression sees
    weights, not outputs, and the next burst's gradient is then only 55 % captured in turn. On
    MNIST's 800-unit hidden layer -- redundant units, smooth spatial filters -- weight MSE and
    accuracy move together, which is why the paper never had to confront this. On a language
    model they do not, which is why the paper's own alternative -- fitting the decode to the
    *output* by training through it -- was the thing it tried and abandoned, and the thing GNOMON
    and ANLAGE's probe found to collapse.

    *What this changes about reading the results.* A single run per arm cannot support an
    at-birth reading: the seed noise of the decode's CE is of order **two nats**, against the
    three-nat span between the "works" and "fails" thresholds. §6's readings must be taken over at
    least two seeds, and the arm they matter for is `mild` (5×), where the regression should
    capture far more of the variance and the proxy should be far better aligned; a second seed of
    `mild` is added to tonight's queue. The lifetime reading is not affected: both GN30 runs put
    the constrained phenotype at `3.4-3.7` against `~1.8` free, and that is the paper's hot-start
    claim failing by 1.6-1.9 nats in both.

    *Recorded against §P-D.46.* That entry attributed the *difference between the end of v1 and
    its plateau* to the decay, correctly. It did not know that the plateau itself was one draw of
    a two-nat-wide distribution. The reading "between, best 7.09" was a reading of one run; the
    honest reading of GN30 at 174× is now "between on one run, fails on the other, and the decode
    is not reproducible at this ratio".

48. **GN50 complete: at 107× the loop lands where 174× did — a plateau between the thresholds
    and a walk to null at the end with the weight fit unchanged — and the per-role residuals
    show what the genome holds: every vector and the embedding table to the second decimal, the
    matrices at 17–33 %.** *2026-09-07, 18:50. `runs/bottleneck/GN50-tinystories-s0-seed0/`,
    `H = 50`, `H₂ = 10`, genome 122,089 parameters, ratio 106.6×, 200 generations × 1M tokens,
    1.62 h, constant learning rate (§3.1 as corrected), best-at-birth checkpoint (§3.2).*

    *The trajectory.*

    | gen | ε | before | **birth** | after | shuffle | resid |
    |---|---|---|---|---|---|---|
    | 1 | 0.95 | 3.39 | 9.05 | 3.44 | 9.00 | 0.837 |
    | 5 | 0.78 | 2.42 | **13.72** | 2.56 | — | 0.779 |
    | 10 | 0.61 | 2.39 | 10.67 | 2.84 | 9.26 | 0.765 |
    | 30 | 0.22 | 2.59 | 9.06 | 6.23 | 9.011 | 0.710 |
    | 50 | 0.08 | 3.15 | 9.55 | 6.86 | 9.011 | 0.681 |
    | 70 | 0.03 | 3.36 | 9.27 | 8.81 | 9.011 | 0.625 |
    | 80 | 0.018 | 3.48 | 7.75 | 7.46 | 9.011 | 0.586 |
    | 100 | 0.007 | 3.49 | 7.83 | 7.70 | 9.011 | 0.531 |
    | **108** | 0.005 | 3.40 | **7.29** | 7.24 | — | 0.531 |
    | 120 | 0.003 | 3.50 | 7.99 | 7.97 | 9.011 | 0.500 |
    | 140 | 0.001 | 3.43 | 7.36 | 7.35 | 9.011 | 0.460 |
    | 160 | 0.0003 | 3.56 | 9.18 | 9.18 | 9.011 | 0.448 |
    | 170 | 0.0002 | 3.38 | 7.75 | 7.75 | 9.011 | 0.451 |
    | 180 | 0.0001 | 3.31 | 9.16 | 9.15 | 9.011 | 0.459 |
    | 200 | 0.0000 | 3.51 | **9.00** | 9.00 | 9.011 | 0.453 |

    Final test `3.617`; the final decode `9.003` against its shuffle `9.011` and `L_null = 9.011`;
    the shuffle sits at `9.011` at every check from generation 20 (the `9.26` at generation 10 is
    the shuffle of a confidently-wrong decode, worse than uniform either way). The last six
    generations decode at or above null (`9.77, 9.52, 9.19, 9.01, 9.01, 9.00`).

    *Reading against §6.* At birth: best `7.29` (generation 108), plateau `7.76` over generations
    100–150 and `8.30` over 150–200, null at the end — **between** on the plateau, **fails** at the
    end, the same shape as both GN30 runs. Lifetime: the phenotype's best per burst (`before`)
    averages `3.44` over generations 100–200, the same `~1.6`-nat cost as GN30 pending the
    `free` arm's number at these tokens. Three completed runs at two ratios:

    | run | ratio | best birth (gen) | 100–150 | 150–200 | end |
    |---|---|---|---|---|---|
    | GN30 v1 (decay) | 174× | 7.09 (156) | 7.52 (100–160) | collapse from 165 | 9.01, resid 0.04 |
    | GN30 v2 | 174× | 7.67 (78) | 10.21 | 8.89 | 9.18, resid 0.48 |
    | GN50 | 107× | 7.29 (108) | 7.76 | 8.30 | 9.00, resid 0.45 |

    The ratio does not order the outcome: the 107× run sits between the two 174× runs, and the
    run-to-run spread at one ratio (§P-D.47's two nats) is larger than anything between 107× and
    174×. GN50 is a third draw from one distribution — a plateau between the thresholds and a
    walk that visits null.

    *The walk to null is not §P-D.46's collapse.* There the residual fell to `0.04` as the decode
    went to null and the trained phenotype degraded to `4.5`. Here the residual holds at `0.45`,
    the phenotype trains to `3.3–3.5` from every decode including the null ones, and the decode
    still walks to `9.00` in the last six generations. Within a run the residual does not predict
    the decode either: over generations 100–200 the correlation between the mean matrix residual
    and the at-birth CE is `−0.40` in GN50, `+0.25` in GN30 v2 and `+0.17` in GN30 v1 (100–160) —
    opposite signs, all weak. §P-D.47 measured the misalignment *across* runs (a better fit
    decoded worse); this is the *within*-run half: at these ratios the weight-MSE proxy carries
    no information about the decode's function.

    *Why the decode walks — an interpretation, labelled as one.* Once `ε ≈ 0` (generation > 100)
    each generation is `W_n = burst(D_{n−1})` then `D_n = fit(W_n)`: the regression's target is
    the genome's own previous output plus one burst's delta, and fifty Adam steps absorb a
    fraction of that delta. Nothing in the loop scores `D_n` by its function. The genome is driven
    only by the sequence of burst deltas, which where the proxy is misaligned is a random walk in
    genome space, and the decode's CE is an observable of that walk: a plateau at its typical
    level with excursions to null, and a "best at birth" that improves with generations because it
    is the running minimum of the walk, not because the plateau moves. On the paper's MNIST network
    the same walk is harmless because MSE and accuracy are aligned there (§P-D.44, §P-D.47); here
    they are not, at either ratio tried.

    *What the genome holds, per role, at the best generation (108).* Relative residual
    `‖Ŵ − W‖² / ‖W‖²`:

    | role | resid | role | resid |
    |---|---|---|---|
    | NORM_MIX, NORM_FFN, NORM_FINAL, CONV_{Q,K,V} | 0.00 | Q | 0.78 |
    | EMB | 0.01 | K | 0.77 |
    | VE | 0.07 | V | 0.67 |
    | AB | 0.08 | O | 0.83 |
    | | | GATE | 0.68 |
    | | | UP | 0.67 |
    | | | DOWN | 0.83 |
    | | | AGATE | 0.81 |
    | | | OGATE | 0.69 |

    The vectors fit exactly — they are the roles ANLAGE found compressible below 1 % (00 §2) — and
    the seven big matrices of every layer fit at 17–33 % of their variance. A decode is "the right
    vectors on a fifth to a third of each matrix", and that is what `7.3` nats looks like. Two of
    these numbers are suspect in opposite directions. The embedding table's `0.01` — a 8192 × 256
    table with no coordinate topology beyond frequency rank, fitted to 99 %? — is what a relative
    residual reads when a matrix's norm is dominated by a component common to all its rows: a fit
    of the mean row explains it while the row-specific part, the whole content of an embedding,
    can be unfitted. And the matrices' `0.7–0.8` says nothing about *which* fifth was captured.
    Neither question has an answer in any residual, and both decide what the next arm should
    change. A CPU ablation was launched with this entry (`scratchpad/ablate_roles.py`, beside the
    running card, `~20` s per evaluation): on the best checkpoints of GN50 and GN30 v2 it replaces
    one role at a time — the decode into the trained weights (the cost of compressing that role
    alone) and the trained weights into the decode (the gain of leaving it uncompressed) — and the
    same per layer, and records per key the share of `‖W‖²` in the mean row and the residual of
    the row-centred parts. Its round trip holds: on CPU in fp32 the checkpoint's `W` scores
    `3.3951` and its `D` `7.2859` against the loop's `3.395` and `7.287` in bf16. The result is
    §P-D.49.

    *Stage 2's sources are fixed.* Transfer reads GN50 at generation 108 (decode `7.29`) and GN30
    v2 at generation 78 (`7.67`); `mild` (5.0×, `H = 419`) is running.

49. **Where the decode's damage lives, measured: the vocabulary table alone costs four nats
    behind a residual of 0.007; the matrices' damage is additive across roles and sits in `O`
    and in the first and last layers; giving back any single role recovers almost nothing; and
    more refit steps do not help. Three variants are pre-registered and queued ahead of
    stage 2: the table kept uncompressed (§11.4), the Fisher-weighted refit (§11.1), and
    selection (§11.2).** *2026-09-07, 19:55. `scratchpad/ablate_roles.py → ablate.json` (CPU,
    fp32, each run's own 256-window test set, ~15 s per evaluation, 96 evaluations);
    `scratchpad/refit_more.py → refit_GN50.json`; spec `docs/bottleneck/00-replication.md` §11
    (`96821a8` and the commit of this entry); tests I11–I15, M-I4, 41 pass.*

    *The ablation.* On the best checkpoints of GN50 (generation 108: `W` `3.395`, `D` `7.286`)
    and GN30 v2 (generation 78: `W` `3.649`, `D` `7.673`), one role at a time: **damage** =
    CE(`W` with that role's decode) − CE(`W`), the cost of compressing that role alone;
    **recover** = CE(`D` with that role exact) − CE(`D`), what leaving it uncompressed buys
    inside the full decode. (The CPU round trip holds: `W` and `D` score `3.3951` / `7.2859`
    against the loop's `3.395` / `7.287`.)

    | role | params | GN50 damage | recover | GN30 v2 damage | recover | resid (GN50) |
    |---|---|---|---|---|---|---|
    | **EMB** | 2.10M | **+4.05** | **−1.76** | **+4.18** | **−0.80** | 0.007 (row-centred 0.09) |
    | O | 0.52M | +1.10 | −0.15 | +1.57 | **+0.68** | 0.83 |
    | V | 0.46M | +0.84 | +0.15 | +0.85 | +0.08 | 0.67 |
    | DOWN | 2.10M | +0.42 | −0.19 | +0.57 | −0.15 | 0.83 |
    | UP | 2.10M | +0.40 | −0.01 | +0.35 | −0.07 | 0.67 |
    | GATE | 2.10M | +0.27 | +0.09 | +0.21 | −0.07 | 0.68 |
    | OGATE | 0.39M | +0.20 | −0.01 | +0.23 | −0.04 | 0.69 |
    | Q | 0.52M | +0.14 | 0.00 | +0.09 | 0.00 | 0.78 |
    | K | 0.46M | +0.12 | 0.00 | +0.21 | 0.00 | 0.77 |
    | AGATE | 0.13M | +0.06 | −0.07 | +0.06 | −0.06 | 0.81 |
    | VE, AB, convs, norms | 2.2M | ≤ +0.002 | 0.00 | ≤ +0.02 | 0.00 | ≤ 0.17 |

    Per layer (every role of the layer at once), GN50 / GN30 v2: layer 0 `+0.58 / +0.44`,
    layers 1–2 `≤ +0.04`, layer 3 `+0.26 / +0.16`, layers 4–5 `≤ +0.12`, layer 6
    `+0.32 / +0.32`, **layer 7 `+0.92 / +0.87`**; leaving layer 7 exact inside the decode makes
    it *worse* (`+0.19 / +0.15`).

    *Four readings.*

    **(1) The table is the binding constraint, and the residual hid it.** Compressing the
    embedding alone, with the other 11M weights exact, gives `7.44` — above the full decode's
    `7.29`. Exact, inside the full decode, it gives **`5.53`** in GN50 — below §6's "works"
    line — and `6.87` in GN30 v2. Its relative residual is `0.007` because `92.5 %` of its
    squared norm is its mean row (`94.7 %` in GN30 v2: the anisotropy of trained embeddings);
    the row-centred residual is `0.09` (`0.19`). A fit of the mean row explains the norm; the
    content is the 9 %. Why the g-network cannot reach it: its input is a *coordinate* code, and
    the paper's coordinates carry the structure — nearby pixels share filters — whereas a Gray
    code over frequency rank puts adjacent *ranks* one bit apart, and adjacent ranks are
    unrelated words. The table is 8192 independent vectors on an axis without topology, and no
    function of that axis compresses it. An MNIST network has no such object; a language model's
    input layer is exactly one. This is the paper's premise failing at one identifiable place,
    measured at four nats.

    **(2) Across the matrices the damage is additive; across layers it is not, and the ends of
    the stack carry it.** The matrix roles' damages sum to `3.56` (GN50) and `4.16` (GN30 v2)
    against total gaps of `3.89` and `4.02` — each hurts alone by about what it contributes to
    the whole — while the layer damages sum to `2.22` and `2.02`, half the gap: compressing one
    middle layer alone costs `≤ 0.12` because the layers above compensate, and the first and
    last layers, which nothing compensates, carry `1.5` of the `2`. Per parameter, `O` is the
    most sensitive matrix (`1.1–1.6` nats for 0.5M) and `DOWN/UP/GATE` the least (`0.2–0.6` for
    2.1M each) — the sensitivity §11.1's Fisher is built to see and the residual does not
    (`O`'s residual, `0.83`, is `DOWN`'s).

    **(3) The decode is a co-adapted whole.** Giving back any single matrix role exact recovers
    at most `0.19` (`DOWN`); giving back `O` exact makes GN30 v2's decode *worse* by `0.68`, and
    layer 7 exact worse by `0.15–0.19`. The loop trained the phenotype on top of the decoded
    matrices for a hundred generations, so the decode's matrices are consistent with each other
    and inconsistent with the exact ones. Partial schemes that mix exact and decoded matrices
    cannot be read off this ablation; the embedding is the exception because it is the input,
    not a member of the co-adapted stack, and giving it back helps in both runs.

    **(4) The refit is not step-limited.** Continuing Algorithm 1's regression from GN50's best
    checkpoint (a fresh Adam at the run's rate, uniform sampling) for 50 / 100 / 200 / 400 / 800
    more steps moves the mean matrix residual `0.531 → 0.526 → 0.522 → 0.518 → 0.513 → 0.507`
    and the decode `7.29 → 7.45 → 7.58 → 7.39 → 7.41 → 7.40`: sixteen generations' worth of
    refit steps buy `0.024` of residual and cost `0.11` nats. The observation that raised the
    question — `mild`, with 21× GN50's genome, at residual `0.62` at generation 57 against
    GN50's `0.66` — therefore does not mean the refit is short of steps; it means a g-network's
    fit of a flat-spectrum matrix plateaus at the same place at any width, §P-D.44's structural
    point once more. The refit budget is not a lever, and no arm is spent on it.

    *Pre-registered from this (§11, written before any of it ran).* Three variants at GN50's
    `H = 50`, seed 0, CRN with GN50, run in this order *ahead of* the pre-registered stage 2 —
    the user's steer was that the training routine itself may need adjusting, and every transfer
    or meta reading is only as good as the loop that made its source:

    - **GN50-E** (§11.4, `--keep EMB`): the table copied through the decode as the paper copies
      its scalars, and **counted** — ratio `5.9×`, the class of `mild` (`5.0×`), with the budget
      placed where the ablation says the function is. Point prediction from the ablation: `5.5`
      at the best generation if the reallocation leaves the loop's dynamics unchanged; reading
      R-E1 against `mild`.
    - **GN50-F** (§11.1, `--fisher`): the refit's metric becomes the second-order estimate of
      the CE increase — entries sampled `∝ F̃ = 0.9 F/mean F + 0.1`, `F` the diagonal empirical
      Fisher on 32 *fitness* windows, never the test set (EWC's estimator, Kirkpatrick et al.
      2017 Eq. 3; the Laplace expansion at a trained point). Reading R-F1: `corr(r_F, birth) >
      +0.5` over generations 100–200, where the plain residual gave random signs.
    - **GN50-FS** (§11.1 + §11.2, `--select`): a refit is kept only if its decode is no worse on
      the fitness set, else genome and optimiser are restored — the `(1+1)` acceptance rule with
      the regression as the proposal, from generation 30. Readings R-S1 (no excursion to null
      after generation 60) and R-S2 (acceptance above 10 %).

    And the branch, fixed now: if a variant reads *works* (`< 6.0`), §10's routine is rerun with
    that arm's flags and transferred to `heldout`. The runner
    (`scratchpad/run_bottleneck_stage2b.py`, log `runs/bottleneck/stage2b.log`) replaces the
    stage 2 waiter, which had started nothing; stage 2's own thirteen steps follow unchanged.
    Cost: `~5.5` h of card before stage 2 begins, from `~21:30`. Neither variant touches the
    test set, and with the flags off the loop is Algorithm 1 bit for bit (I12: the sampler
    reproduces `randint` exactly).

    *Recorded against §P-D.48.* That entry called the matrices' `0.7–0.8` and the table's `0.01`
    "suspect in opposite directions"; both suspicions held, the second by four nats. Its
    interpretation of the walk stands, and (3) adds that the walk's decodes are internally
    consistent objects, which is why a run's best decode is a maximum of the walk and not a fit
    that a local improvement can lift.

50. **`mild` (5×) reached the best decode of any arm, 6.81 at generation 120, and then
    collapsed: one burst diverged to the uniform state and the loop fitted the dead phenotype.
    Behind it, a defect of mine: a quarter of the wide g-networks were dead from generation 1,
    so the "5×" arm regenerated twenty-one of its matrices as zeros in every generation. A
    LeakyReLU control (`mild-L`) is pre-registered; `mild` seed 1 is dropped.** *2026-09-07,
    20:50. `runs/bottleneck/mild-tinystories-s0-seed0/` (`H = 419`, `H₂ = 52`, genome
    2,607,895 parameters, ratio 4.99×, 200 generations, 1.65 h, 27.6 min in the hook);
    `scratchpad/dead_gnets.py`; spec §11.5; test I16.*

    *The trajectory.*

    | gen | ε | before | **birth** | after | shuffle | resid | dead g-nets |
    |---|---|---|---|---|---|---|---|
    | 1 | 0.95 | 3.39 | 9.45 | 3.44 | 9.02 | 0.794 | 66 |
    | 5 | 0.78 | 2.42 | **15.25** | 2.54 | — | 0.760 | 52 |
    | 20 | 0.37 | 2.36 | 11.39 | 4.05 | 9.007 | 0.679 | 23 |
    | 60 | 0.05 | 3.38 | 8.02 | 6.36 | 9.045 | 0.617 | 21 |
    | 80 | 0.018 | 3.42 | 7.09 | 6.56 | 9.009 | 0.574 | 20 |
    | 100 | 0.007 | 3.32 | 8.04 | 6.74 | 9.009 | 0.560 | 21 |
    | **120** | 0.0025 | 3.28 | **6.81** | 6.84 | 9.011 | 0.533 | 21 |
    | 121 | 0.0024 | **9.01** | 9.03 | 9.01 | — | 0.576 | 22 |
    | 125 | 0.0019 | 9.011 | 9.011 | 9.011 | — | 0.373 | 15 |
    | 129 | 0.0016 | 9.011 | 9.011 | 9.011 | — | 0.256 | 13 |
    | 134 | 0.0012 | 3.73 | 9.01 | 7.25 | — | — | — |
    | 139 | 0.0010 | 9.011 | 9.01 | 9.01 | — | — | — |
    | 160 | 0.0003 | 3.32 | 9.03 | 9.03 | 9.011 | 0.514 | 21 |
    | 200 | 0.0000 | 9.011 | 9.02 | 9.02 | 9.011 | 0.560 | 22 |

    Best decode `6.81` (generation 120; the next best `6.83, 6.85, 6.87` at 89, 87, 96), the
    best of any arm and still *between*. Plateau 100–150 `8.52`, 150–200 `9.04`; final test
    `5.77`; the final decode `9.018`. Per-role residuals at the best generation are **not lower
    than GN50's** with a genome 21× larger: `Q 0.66, K 0.69, V 0.63, O 0.82, GATE 0.83, UP
    0.62, DOWN 0.86, AGATE 0.94, OGATE 0.84` against GN50's `0.78, 0.77, 0.67, 0.83, 0.68,
    0.67, 0.83, 0.81, 0.69` — §P-D.49 (4)'s plateau, at 21× the width.

    *Defect 1: dead g-networks.* At the best checkpoint, 21 of the 88 matrix g-networks have
    residual `> 0.98`, and 20 of those have **no active unit in their second layer** on any code
    (`scratchpad/dead_gnets.py`: first-layer units 100 % active, second-layer 0 %): the network
    outputs its final bias, a constant, and the decoded matrix has `‖D‖/‖W‖` of `0.002–0.15` —
    `GATE_0, GATE_2, GATE_3, GATE_5, GATE_6, DOWN_1, DOWN_2, DOWN_5, DOWN_6, DOWN_7, UP_4, UP_6,
    Q_1, K_4, O_4, O_5, O_6, V_5, OGATE_2, OGATE_5, OGATE_6, AGATE_3`. They died at
    **generation 1** (66 keys above `0.98` after the first refit, 23 by generation 20, 21 for
    the rest of the run) and never recovered, as a ReLU layer with every unit inactive cannot:
    no gradient reaches it. Six more networks have 4–25 % of their second layer alive and
    residuals of `0.90–0.95`. **GN50 and GN30 v2 have none** (0 of 88 at every generation
    checked; GN30 v2 had `DOWN_5` at `> 0.98` for twenty generations and recovered). The
    difference is the width: at `H = 419` the second layer's fan-in is 419, its default
    initialisation is `±1/√419 = ±0.049`, and its pre-activations are small enough that Adam at
    `2·10⁻³` walks the biases past them inside the first fifty steps; at `H = 50` they are three
    times larger. The activation is my choice (§1: "the paper's unstated choice, taken as
    ReLU"), the width is the arm's, and the two together are the defect. Consequences: the
    `4.99×` is nominal — the genome's effective content is the 67 live networks plus 21 zero
    matrices; the phenotype was regenerated with a quarter of its matrices at zero in every
    generation from `ε ≈ 0` on and re-learned them from zero in every 1M-token burst, which it
    could (`before` `3.3` throughout: SwiGLU's gate at zero still passes a gradient); and the
    at-birth `6.81` was reached *with* twenty-one zero matrices, which says the live two-thirds
    carry more at `H = 419` than the whole does at `H = 50`.

    *Defect 2: a burst diverged to an absorbing state.* The burst after the best decode
    (generation 120, at birth `6.81`) went from a training CE of `4.57` (step 8400) to `8.54`
    (step 8600) and `9.00` (8800): the phenotype reached the **uniform distribution exactly**
    (`9.0109 = ln 8192`) and stayed there for nine generations, `before = birth = after =
    9.0109`, while the genome fitted the dead phenotype (residual `0.55 → 0.25`: near-zero
    matrices are easy to fit) and decoded null. Bursts escaped at generations 130 (`6.07`),
    133–138 (`3.5–4.6`) and 160 (`3.32`), and died again at 139 and 200; the regenerated decode
    was null every time, so each escape started from null. No burst of GN50, GN30 v1 or v2
    diverged (600 bursts, `before` always `3.3–3.8`); the decode's norms at generation 120 were
    ordinary (`max ‖D‖/‖W‖ = 1.03`, max row ratio `1.03`), so the trigger was not an outlier
    weight. What is specific to this arm is that every burst starts with twenty-one zero
    matrices under a constant Muon rate of `0.02` with no warm-up (§3.1). Whether that is the
    cause is what `mild-L` decides (R-L2).

    *What is pre-registered from this (§11.5, before it ran).* `--activation leaky` (LeakyReLU,
    slope 0.01, Maas et al. 2013) in both hidden layers of every g-network; the activation is
    recorded in the genome's summary and reloaded with it (I16: a dead ReLU layer decodes a
    constant and passes no gradient below it, the leaky one passes gradient). Arm **`mild-L`**
    (seed 0, everything else as `mild`) is the honest `5×` control for R-E1; readings **R-L1**
    (no matrix key above `0.98` after generation 20) and **R-L2** (no burst with `before > 8.5`).
    ReLU stays the default so the `H = 50` arms stay comparable with GN50 under CRN. `mild`
    seed 1 is dropped from the queue — a second seed of a configuration with a quarter of its
    genome dead has no reading — and `mild-L` takes its slot, after GN50-FS. `mild` stays on
    record as the arm it was.

    *Reading against §6, as it stands.* Three ratios have now run under Algorithm 1 as written
    (174×, 107×, 5× nominal): best decodes `7.09 / 7.67`, `7.29`, `6.81` — all *between*, the
    shuffle at null every time (wiring, not distribution), and every plateau followed by a walk
    or a collapse. `mild`'s lifetime reading is not takeable: `before` averages `5.08` over
    generations 100–200 only because half of those bursts were dead.

51. **The lifetime reading with its matched control: the constrained phenotype pays 1.6–1.9
    nats against the tabula-rasa network at the same training tokens, more than five hundred
    `σ_e`, and the decode's head start is worth zero tokens. The paper's "hot-started" claim
    (Fig. 3E, `≈ 0`) fails by that margin at every ratio.** *2026-09-07, 21:25.
    `runs/bottleneck/free-tinystories-s0-seed0/` (201M tokens, the arms' recipe, seed and
    window order, no genome; 1.07 h; final test `1.804`, fitness `1.827`); stage 1 complete,
    stage 2b began at 21:17 with GN50-E.*

    | arm | ratio | phenotype at 201M tokens | best per burst, gens 100–200 (`before` mean) | `free` at the same tokens | cost |
    |---|---|---|---|---|---|
    | `free` | — | **1.804** | — (curve `1.857 → 1.827` over 100–200M) | — | 0 |
    | GN30 v2 | 174× | 3.520 | 3.70 | 1.80–1.86 | **+1.72 / +1.85** |
    | GN50 | 107× | 3.617 | 3.44 | 1.80–1.86 | **+1.81 / +1.59** |
    | GN30 v1 (decay) | 174× | 5.93 (collapsed, §P-D.46) | 3.4 (100–160) | 1.85 | +1.6 on the plateau |
    | `mild` | 5× nominal | 5.77 (collapsed, §P-D.50) | 5.08 (half the bursts dead) | — | not takeable |

    §6's bar is `3σ_e = 0.009` nats (`σ_e = 0.0029`, `runs/anlage/noise-tinystories-s0/`); the
    measured cost is `1.6–1.9`, two hundred times the bar. It is the cost of living on the
    decodable set (§P-D.46's reading of Eq. 1's trade-off, now against the matched control
    rather than R1's mean), and it is the same at 107× and 174×: the phenotype that is
    regenerated every generation cannot get below `~3.4` in a burst, whatever the decode it
    starts from.

    *The head start is worth nothing.* Generation 1's burst is a free burst — `ε(1) = 0.95`
    has not acted yet, and the arms share `free`'s windows — so every arm's `before` at
    generation 1, `3.389`, is `free` at 1M tokens. From the decode at `7.3–7.7` a burst of 1M
    tokens reaches `3.3–3.7`; from a random initialisation the same burst reaches `3.39`.
    Starting `1.5` nats ahead of random ends level with random or behind it: the decode's
    wiring does not shorten gradient descent at all, which is the direct negation of "as
    though hot-started". (`free` at 10M tokens is `2.23`; the constrained phenotype never
    reaches that in any single burst.)

    *What this fixes in the readings.* §P-D.46 and §P-D.48 quoted the lifetime cost against
    R1 v3's mean `1.809` at 25M tokens; the matched number is `1.80–1.86` over the same
    generations, and the readings stand unchanged. The variant arms (§11) are measured against
    this same control.

52. **GN50-E reads WORKS. With the vocabulary table carried verbatim and the eight layers
    compressed 107×, the decode reaches 2.86 nats at birth — 85 % of the way from null to the
    free network — improving monotonically to the last generation, with no walk and no
    collapse, the residual aligned with the function again, the shuffle control worse than
    null, and the trained phenotype within 0.37 nats of the free network. One run; its second
    seed now runs first.** *2026-09-07, 22:55. `runs/bottleneck/GN50-E-tinystories-s0-seed0/`
    (`H = 50`, `H₂ = 10`, `--keep EMB`; genome 2,217,620 parameters of which 2,097,153 are the
    table, ratio 5.87×; 200 generations, 1.54 h, 23 min in the hook). Free control §P-D.51.*

    *The trajectory.*

    | gen | ε | before | **birth** | after | shuffle | resid | dead g-nets |
    |---|---|---|---|---|---|---|---|
    | 1 | 0.95 | 3.39 | 8.75 | 3.41 | 9.16 | 0.836 | — |
    | 5 | 0.78 | 2.42 | 14.28 | 2.43 | — | 0.785 | — |
    | 20 | 0.37 | 2.21 | 14.24 | 2.36 | 10.30 | 0.744 | — |
    | 40 | 0.14 | 2.23 | 15.03 | 3.01 | 10.81 | 0.727 | — |
    | 60 | 0.05 | 2.25 | 10.58 | 3.78 | 11.10 | 0.696 | 9 |
    | 75 | 0.024 | 2.25 | 5.99 | 3.76 | — | 0.667 | 1 |
    | 80 | 0.018 | 2.25 | 5.14 | 3.65 | 10.28 | 0.654 | 0 |
    | 100 | 0.007 | 2.25 | 3.60 | 3.36 | 9.76 | 0.624 | 0 |
    | 120 | 0.0025 | 2.23 | 3.22 | 3.18 | 9.62 | 0.600 | 0 |
    | 140 | 0.0009 | 2.22 | 3.11 | 3.10 | 9.71 | 0.582 | 0 |
    | 160 | 0.0003 | 2.21 | 3.00 | 3.00 | 9.70 | 0.566 | 0 |
    | 180 | 0.0001 | 2.20 | 2.93 | 2.93 | 9.28 | 0.555 | 0 |
    | **199** | 0.0000 | 2.19 | **2.86** | 2.86 | — | 0.547 | 0 |
    | 200 | 0.0000 | 2.19 | 2.87 | 2.87 | 9.44 | 0.548 | 0 |

    Final test `2.173`, fitness `2.200`; the final decode `2.855`, its shuffle `9.35`.

    *Readings, against §6 and §11.3.*

    **Works.** Below `6.0` from generation 75, below `4.0` from 92, below `3.0` from 157; best
    `2.863` at generation 199 and still falling (`2.93 → 2.86` over the last twenty). On the
    paper's own scale — the fraction of the gap from null (`9.01`) to the trained network
    (`1.80`, §P-D.51) that the decode recovers — this is `85 %`; every previous arm was at
    `18–30 %`.

    **Not a walk.** No generation above `8.95` after 100, no diverging burst, a monotone
    trajectory from generation 60 on, and the last eight decodes within `0.03` of each other
    (`2.86–2.89`): §P-D.47's two-nat spread was a property of the walking regime, not of the
    loop.

    **Wiring, in its strongest form.** The shuffle sits at `9.2–11.7` at all 21 checks — worse
    than uniform: a network whose rows are permuted is confidently wrong, which is what a
    decode that carries specific wiring looks like under the paper's Fig. 2D control.

    **The residual is aligned with the function again.** Over generations 100–200,
    `corr(resid, birth) = +0.98`; with the table compressed it was `−0.40 / +0.25 / +0.17`
    (§P-D.48). The matrices' residual `0.62 → 0.55` tracks the decode `3.60 → 2.86`. The
    misalignment of §P-D.47–48 was the table's: the one object the metric could not see was
    the one that decided the function. §11.1's Fisher arm is now a test of a *second-order*
    effect and is kept in the queue for that reason.

    **Lifetime.** `before` averages `2.22` over generations 100–200 and the final test is
    `2.173` against `free`'s `1.804` at the same tokens: `+0.37` nats, against `+1.7–1.9` in
    every other arm (§P-D.51) — forty times §6's bar still, so the constraint is not free, but
    the decodable set now contains a network within `0.4` nats of the unconstrained one. The
    regeneration jolt is `0.84` nats on average (`after − before`) and one burst recovers it.

    **Hot start, holding for the first time.** From the decode at `2.86` a 1M-token burst
    reaches `2.19`; `free` reaches `2.23` at **10M** tokens and `2.00` at 20M. The decode is
    worth about ten bursts of head start, where at every other arm it was worth zero
    (§P-D.51). The paper's "as though hot-started" holds here qualitatively.

    **R-E1.** `2.86` against `mild`'s `6.81` (the same ratio class, the whole network
    compressed): `3.9` nats, twice the spread bar; `mild-L` is pending. The ablation's point
    prediction (`5.5`) was pessimistic by `2.6` nats: it was taken on a phenotype trained
    *under* the compressed table, and the reallocation changes the loop — with the input
    fixed, the phenotype and the decode co-adapt toward a decodable set that contains a good
    network, which no ablation of the old phenotype could show. Nine g-networks were dead at
    generation 60 and all recovered by 78 (`H = 50`; §11.5's control is still owed for
    `H = 419`).

    *What it is and what it is not.* The eight layers — 11M parameters — decode from 120k
    g-network parameters, `107×`, to a network at `2.86`: the paper's method, working on the
    layers of a language model. The whole-network ratio, `5.9×`, is set by the table, which is
    `95 %` of this genome and is not compressed at all; the honest sentence is "the layers
    compress a hundredfold once the table is carried", not "the network compresses sixfold".
    The table needs a *content* code, not a coordinate code — the natural one is a rank-`r`
    factorisation refit by SVD each generation, `r(V + d)` parameters (rank 32: `0.27M`, a
    whole-network ratio near `30×`) — and that is the next arm, to be specified after the
    second seed and not tonight.

    *The queue, changed at 22:52 and why.* The first positive result in five models' worth of
    genomic-bottleneck attempts is one run; its replication outranks everything else, and the
    transfer and meta questions (§9, §10) are answerable only from a working source. The
    fixed-list runner was replaced by an editable queue (`scratchpad/run_bottleneck_queue.py`,
    `queue.json`, log `runs/bottleneck/queue.log`); GN50-F, fifteen minutes in, was stopped
    and requeued. Order now: **GN50-E seed 1**, GN50-F, GN50-FS, mild-L, then transfer
    GN50-E → tasks and → tinystories, meta-GN30-E, meta-free, the heldout transfers from both,
    GN50-E → heldout, and then every pre-registered stage-2 step from the near-null sources,
    unchanged, for the record. With the table carried, the transfer's `decode-all` and
    `copy-all` arms share the exact table and their contrast isolates the layers' decode; this
    is recorded here before those arms run.

53. **GN50-E replicates on a second seed to within 0.03 nats at every checkpoint: the
    reading is WORKS on two seeds.** *2026-09-08, 01:05.
    `runs/bottleneck/GN50-E-tinystories-s0-seed1/` (a different phenotype initialisation,
    genome initialisation and window order; the same pool, recipe and settings; 2.06 h — the
    card was shared).*

    | | seed 0 | seed 1 |
    |---|---|---|
    | first generation below 6.0 / 4.0 / 3.0 | 75 / 92 / 157 | 75 / 92 / 153 |
    | birth at 100 / 140 / 180 / 200 | 3.60 / 3.11 / 2.93 / 2.87 | 3.58 / 3.07 / 2.92 / 2.84 |
    | best at birth (generation) | 2.863 (199) | 2.836 (200) |
    | `before` mean 100–200; jolt | 2.218; 0.84 | 2.217; 0.82 |
    | final test; final decode; its shuffle | 2.173; 2.855; 9.35 | 2.166; 2.818; 9.47 |
    | `corr(resid, birth)` 100–200 | +0.98 | +0.98 |
    | diverging bursts; generations at null after 100 | none; none | none; none |
    | shuffle range over 21 checks | 9.16–11.7 | 9.21–11.2 |

    The two trajectories differ by less than the gap between any two consecutive checkpoints
    of either. Against §P-D.47 — two numerically-identical GN30 runs two nats apart — this is
    the other regime: with the table carried, the loop is a contraction, not a walk, and its
    outcome is a property of the configuration. Every reading of §P-D.52 stands on two seeds:
    works (`< 6.0` from generation 75, best `2.84–2.86`), lifetime `+0.36–0.37` against
    `free`, hot start worth about ten bursts, wiring under the shuffle control. Both runs were
    still improving at generation 200 (`−0.03` to `−0.04` per twenty generations); the paper's
    500 generations would be a 4-h run and is not queued tonight.

54. **GN50-F, killed from outside at generation 175, had already read WORKS with the table
    compressed: the Fisher metric alone takes the decode from 7.29 to 5.54 at 107×, aligns the
    residual with the function, stops the walk, and halves the phenotype's lifetime cost. The
    card then sat idle for seven and a half hours.** *2026-09-08, 11:30.
    `runs/bottleneck/GN50-F-tinystories-s0-seed0/log.txt` (175 generations of 200, `ε(175) =
    0.0002`, fully in regime; `best.json` at generation 173; no `loop.json`); the queue reordered
    and relaunched at 11:28.*

    *The trajectory, from the log.*

    | gen | before | **birth** | after | shuffle | resid | `r_F` |
    |---|---|---|---|---|---|---|
    | 1 | 3.39 | 8.91 | 3.44 | 9.02 | 0.854 | 0.825 |
    | 10 | 2.39 | 8.50 | 2.83 | 10.80 | 0.778 | 0.681 |
    | 20 | 2.35 | 8.15 | 3.89 | 10.23 | 0.716 | 0.592 |
    | 40 | 2.51 | 7.07 | 5.42 | 10.56 | 0.662 | 0.483 |
    | 60 | 2.58 | 6.56 | 5.54 | 10.74 | 0.612 | 0.392 |
    | 80 | 2.61 | 6.02 | 5.49 | 10.41 | 0.587 | 0.311 |
    | 100 | 2.60 | 5.88 | 5.69 | 9.75 | 0.562 | 0.272 |
    | 120 | 2.60 | 5.83 | 5.72 | 9.78 | 0.555 | 0.251 |
    | 140 | 2.60 | 5.60 | 5.55 | 9.35 | 0.556 | 0.236 |
    | 160 | 2.60 | 5.61 | 5.60 | 9.31 | 0.545 | 0.223 |
    | **173** | 2.60 | **5.54** | 5.53 | — | 0.538 | 0.210 |
    | 175 | 2.60 | 5.54 | 5.53 | — | 0.538 | 0.211 |

    *Readings, against §11.3.* **R-F1 met:** over generations 100–175, `corr(r_F, birth) =
    +0.74` (bar `+0.5`), and the plain residual is aligned too (`+0.67`) — once the refit is
    driven by the entries the function depends on, both metrics track it. **R-F2: works,** below
    `6.0` from generation 83, `5.54` at the best, `5.75` on the plateau, never above `6.0` after
    generation 100; against GN50's `7.29` that is `1.75` nats, just under §P-D.47's two-nat bar,
    but the *shape* is the contraction regime of §P-D.52–53 — monotone, no excursion, the
    shuffle at `9.3–10.8`, worse than null — and not the walk of §P-D.48, whose spread the bar
    was measured in. **Lifetime:** `before` averages `2.60` against `free`'s `1.83–1.86`: `+0.75`,
    against `+1.6–1.8` at every arm with the paper's metric; the regeneration jolt is `3.1`
    nats and a burst recovers it. With the table still compressed by coordinates (§P-D.49), the
    Fisher weighting puts the table's fit on the rows the probe's gradient reaches — the frequent
    tokens — which is where the CE is, and that alone is worth `1.75` nats at birth and a nat of
    lifetime. Two independent levers now each move the loop from *between* to *works*: the
    table's code (§P-D.52–53) and the refit's metric (this entry); their combination
    (`GN50-FE`, and `GN50-FR32` with the rank code) is pre-registered in §11.3 with reading
    R-C1 and queued.

    *The loss.* At 04:02:37 the training job exited with code `1073807364 = 0x40010004`
    (`DBG_TERMINATE_PROCESS`, a termination from outside; `queue.err` is empty, no Python
    error), while its runner survived, wrote `[FAIL]`, met the `PAUSE` flag set at 01:15 for
    the runner swap, and stopped. This session's own process was restarted at some point
    between 01:45 and 11:26; the kill and that restart coincide, as they did in §P-D.41. The
    card was idle from 04:03 to 11:28. Mitigation: the queue's jobs are now started in their
    own process group with no console window (`CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`),
    which is the documented way to keep console-control terminations from reaching them;
    whether it suffices is known only at the next session exit, and is recorded as untested.
    The partial run keeps its log and its best checkpoint (generation 173, `5.537`); the full
    run is requeued for `loop.json` and its last 25 generations.

    *The queue at 11:28, and why.* The E-based transfer and meta steps first (the user's
    question, from the working source), then `GN50-FE` (the combination both results point
    to), `R32`, `R16`, the `F` rerun, `FS`, `mild-L`, the pre-registered near-null-source
    steps, `R64`, `FR32`, the branch. Every step is idempotent and the order can be changed
    while a job runs (`scratchpad/queue.json`, re-read before each step).

55. **Transfer GN50-E → verifier tasks: the source is a net hindrance on this target for every
    whole-network arm, copy and decode alike; the decode hinders less than the copy — the sign
    of the paper's Fig. 3G, in the negative-transfer regime; the high-layer arms are neutral.
    The pre-registered bar is corrected to the random seeds' spread at the fraction read.**
    *2026-09-08, 12:20. `runs/bottleneck/xfer-tasks-from-GN50-E-tinystories-s0-seed0/`
    (source GN50-E at its best, generation 199, decode `2.86`; target: the default verifier
    families on the reserved ids, 5M tokens per arm, evaluated every 0.5M, three random seeds
    first; 44 min).*

    | arm | birth (target) | 25 % | 50 % | 100 % | source CE after the 5M target tokens |
    |---|---|---|---|---|---|
    | random s0 / s1 / s2 | 8.77 / 9.23 / 8.54 | 0.379 / 0.369 / 0.370 | 0.365 / 0.364 / 0.364 | 0.3609 / 0.3611 / 0.3609 | 9.12 |
    | copy-all | 16.18 | 0.700 | 0.397 | 0.3613 | 9.78 (from 2.19) |
    | decode-all | 8.83 | 0.641 | 0.388 | 0.3611 | 7.73 (from 2.86) |
    | copy-low | 9.17 | 0.908 | 0.401 | 0.3614 | 7.44 |
    | decode-low | 9.14 | 0.844 | 0.383 | 0.3614 | 7.52 |
    | copy-high | 8.57 | 0.370 | 0.365 | 0.3609 | 9.13 |
    | decode-high | 8.65 | 0.369 | 0.365 | 0.3611 | 9.54 |

    *The bar, corrected.* §9.3 takes `σ_e` from three random seeds; the script took it at the
    **final** CE (`0.0002`, bar `0.0005`), where every arm has converged to `0.361` and nothing
    is distinguishable. At 25 % of the budget the same three seeds spread twenty-five times
    wider (`3σ = 0.016`), at 50 % `0.002`. Every contrast at a fraction is read against the
    spread *at that fraction* (`scratchpad/read_transfer.py`); the correction can only fail a
    reading, never pass one, and it fails T3 below.

    *Readings.* **The source's net value on this target is negative.** `decode-all` is `0.26`
    worse than random at 25 %, `copy-all` `0.32` worse, the low-layer arms `0.46–0.53` worse;
    by 50 % the gaps are `0.02–0.04` (ten to twenty times the bar there) and by 100 % they are
    gone. A network trained on TinyStories and then on reserved-id symbolic tasks is slower
    than one started from random: the source's wiring has to be undone first. On this target the
    paper's question — does the decode transfer like the copy? — is asked between two
    hindrances. **T1 yes, trivially:** everything converges. **T2 yes:** `decode-low` beats
    `copy-low` by `0.064` at 25 % (four times the bar) and `decode-all` beats `copy-all` by
    `0.059`: the decoded network is *less* of a hindrance than the copied one — less to undo —
    which is the paper's mechanism (the decode is simpler and more general than the weights it
    was fitted to) in the one direction this target can show it. **T3 no:** `decode-high` is
    `0.011` better than random at 25 % against a bar of `0.016`; neutral, like `copy-high`.
    **Retention:** after 5M tokens of tasks, `copy-all`'s TinyStories CE went `2.19 → 9.78`
    (beyond null) and `decode-all`'s `2.86 → 7.73` — catastrophic forgetting in both, less in
    the decoded one, the same asymmetry as above. §P-D.41's lesson holds: every stream is
    recorded, so this is adaptation and retention, not one mistaken for the other.

    *What this target measures.* Symbolic tasks on 64 reserved ids reach `0.36` from scratch in
    5M tokens with no language needed, and the three random seeds' finals agree to `0.0002`: the
    target measures interference, not transfer, and a TinyStories source can only interfere.
    §9's informative targets are `heldout` (the GRID family and lengths never trained) and
    `tinystories` (the hot start on the source's own distribution), and `tinystories` is running.

56. **The hot start, measured on the source's own distribution: GN50-E's decode starts 1.1
    nats ahead of a random initialisation, stays ahead by 0.28 after 5M tokens (a random start
    never catches its 1M-token point within the budget), and trails the copied weights by a
    shrinking 0.02 at the end — the paper's Fig. 3E and Fig. 3G, holding on a language model
    to within a bar.** *2026-09-08, 12:35.
    `runs/bottleneck/xfer-tinystories-from-GN50-E-tinystories-s0-seed0/` (source GN50-E at
    generation 199; target: TinyStories, a different window pool of the same corpus, 5M tokens
    per arm evaluated every 0.5M, three random seeds first; 15 min).*

    | arm | birth | 0.5M | 1M | 2.5M | 5M |
    |---|---|---|---|---|---|
    | random (3 seeds) | 10.3 | 3.46–3.48 | 3.12–3.13 | 2.61–2.62 | 2.265–2.276 |
    | copy-all | 2.19 | 2.23 | 2.16 | 2.09 | **1.966** |
    | decode-all | 2.86 | 2.33 | 2.23 | 2.12 | **1.989** |
    | copy-low | 12.2 | 2.87 | 2.55 | 2.25 | 2.054 |
    | decode-low | 12.7 | 2.90 | 2.56 | 2.26 | 2.058 |
    | copy-high | 9.5 | 3.42 | 3.02 | 2.56 | 2.267 |
    | decode-high | 9.7 | 3.45 | 3.06 | 2.59 | 2.279 |

    Bars (three times the random seeds' spread): `0.030` at 0.5M, `0.015` at 1M, `0.010` at
    2.5M, `0.018` at 5M.

    *Readings.* **Against random:** `decode-all` is `1.14` nats ahead at 0.5M tokens, `0.89` at
    1M, `0.49` at 2.5M, `0.28` at 5M — thirty to sixty bars — and a random start never reaches
    the decode's 1M-token CE (`2.23`) inside the 5M budget. This is the hot start of §P-D.52
    measured the way §9 pre-registered it, and it is the reading that failed at every arm
    before the table was carried (§P-D.51: worth zero tokens). **Against the copy (T1):**
    `decode − copy` is `+0.09, +0.06, +0.03, +0.023` at 0.5M, 1M, 2.5M, 5M: the decoded network
    trails the exact weights by a margin that shrinks through the run and ends at `1.3` bars —
    T1 as pre-registered (`< 1` bar) fails by `0.005` nats, and the honest sentence is "within
    `0.02` nats of copying the weights after 5M tokens, and converging". The paper's own claim
    for transfer was "indistinguishable from copying" except in the lowest layers; this is that
    to within the resolution of this test. **Low layers (T2):** `decode-low` and `copy-low`
    are equal (`0.006` at 1M, within the bar) — both carry the same exact table, so their
    contrast isolates the two decoded layers, which transfer exactly as the copied ones do; the
    paper's "decode transfers *better* in layer 1" is not reproduced, and not contradicted
    either (the paper's layer 1 has pixel topology; ours is two transformer layers). **High
    layers (T3):** ahead of random early (`−0.065` at 1M, four bars) and level by 5M, as the
    copied high layers are.

    *What the head start is worth.* A random start needs more than 5M tokens to reach the
    decode's CE after 1M; from `free`'s curve (§P-D.51) `2.23` falls at about 10M tokens — the
    decode is worth about nine million tokens, ninety per cent of a 10M-token budget. That is
    the paper's hot-start claim holding quantitatively on this distribution.

    *Reading with §P-D.55.* The same decode hinders on the reserved-id symbolic tasks and helps
    massively on its own distribution: the genome carries the source's competence, and whether
    that transfers is a property of the target's relation to the source, as it is for the exact
    weights. `heldout` — new task families the meta routine never trained — is the target the
    meta-learning question (§10) is read on; `meta-GN30-E` is running.

57. **The meta routine's own record, on a shared task sequence: regenerating from the genome
    returns a better generalist than the network its last task left (−0.22 nats), but the
    unconstrained sequential control is better still on the mixed distribution (by 0.35) and
    far better on the held-out family at the end (6.62 against the decode's 8.12). Whether the
    genome *adapts faster* on the held-out family is §10.3's reading, in the transfers now
    running.** *2026-09-08, 14:20. `runs/bottleneck/meta-GN30-E-tinystories-s0-seed0/`
    (`H = 30`, table carried, ratio 5.99×, 120 tasks × 1M tokens, 0.79 h) and
    `.../meta-free-tinystories-s0-seed0/` (the same 121 tasks in the same order from the same
    pool — digests equal — no genome, 0.97 h).*

    CE on the fixed mixed-distribution evaluation set at each task boundary (the free control's
    phenotype; the genome arm's phenotype `before`, its decode at `birth`, the regenerated
    phenotype `after`), with the task each burst trained on:

    | gen | task trained | control | genome: before | birth | after |
    |---|---|---|---|---|---|
    | 1 | MQAR/8 | 3.32 | 3.50 | 6.16 | 3.43 |
    | 20 | PERM/6 | 3.78 | 3.55 | 12.25 | 3.44 |
    | 40 | PERM/8 | 3.79 | 3.51 | 5.27 | 3.92 |
    | 60 | MODCOUNT/12 | 3.09 | 3.79 | 4.33 | 3.25 |
    | 80 | DYCK/4 | 4.62 | 4.48 | 4.07 | 3.89 |
    | 100 | MQAR/8 | 4.41 | 4.45 | 3.50 | 3.54 |
    | 120 | MQAR/6 | 3.42 | 3.05 | 3.62 | 3.25 |
    | **mean, last half** | | **3.57** | **3.92** | 4.08 | **3.70** |
    | final (after task 121, MQAR/8) | | 3.17 | 2.87 | — | — |
    | held-out GRID/16 at the end | | **6.62** | 8.73 | **8.12** | — |

    Best decode `3.17` at generation 97; the shuffle at `9.01` at every check; no burst diverged.

    *Readings.* **(1) The decode is a generalist.** Over the last half, `after − before =
    −0.22`: regenerating from the genome undoes the specialisation the last task left and
    returns a network better on the mixture than the one just trained. That is the Reptile
    mechanism made visible in this loop — each refit fits a network trained on a *different*
    task on top of the previous decode, so the genome integrates across tasks. **(2) It costs
    what the constraint always costs.** The constrained phenotype is `0.35` behind the free
    sequential network on the mixture at the same tokens (§P-D.51's lifetime cost, `+0.37` in
    the single-task arm, reappears here), and the decode itself is `0.13` behind the free
    network's *last-task-specialised* state. The final-burst point (`2.87` against `3.17`) is
    one task's burst and goes the other way; it is not a reading. **(3) Nothing innate on the
    held-out family.** At the end the free network scores `6.62` on GRID/16 and the decode
    `8.12` (the constrained phenotype `8.73`): the free network carries more of what GRID needs
    at the moment the transfer starts. §10.3 asks a different question — from these starts,
    who adapts faster on GRID — and the two `heldout` transfers (decode-all of `meta-GN30-E`
    against three random seeds; copy-all of `meta-free` against the same) decide it, with
    bars taken at each fraction (§P-D.55). **(4) Sequential single-task training forgets in
    both arms:** the boundary CE swings by a nat with the last task (PARITY and DYCK bursts
    hurt the mixture most), and mixed training reaches `0.36` on the same families in 5M
    tokens (§P-D.55) where either sequential arm is at `3–4` after 121M. The routine's value,
    if any, is in adaptation and not in the mixture score, which is why §10.3 was written on
    the held-out transfer.

58. **The held-out transfers of §P-D.57, read (they finished after that entry was written):
    on the family the meta routine never trained, the free sequential network's copied
    weights are +1.33 nats worse than a random start at the end of the budget and the
    genome's decode is +0.007 — neutral.** *2026-09-10.
    `runs/bottleneck/xfer-heldout-from-meta-GN30-E-tinystories-s0-seed0/transfer.json`,
    `.../xfer-heldout-from-meta-free-tinystories-s0-seed0/transfer.json`.* Against random at
    25 / 50 / 100 %: meta-free `copy-all` `+1.695 / +1.434 / +1.328`; meta-GN30-E `decode-all`
    `+0.105 / +0.008 / +0.007`, its constrained phenotype `copy-all` `+1.001 / +0.040 / +0.006`.
    Nothing beats random on a family the source never saw (the paper's F-MNIST outcome); the
    copy carries 1.3 nats of damage and the decode none. The full reading, and the model
    built on it, are in `docs/palimpsest/00 §0.2–0.3` and `03 P-P.0`.
