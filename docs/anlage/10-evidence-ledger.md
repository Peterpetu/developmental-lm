# 10 — Evidence ledger

Every borrowed number used anywhere in this specification, with the exact claim relied upon,
the source, its status, and its known failure mode. Tags: **[PR]** peer-reviewed venue;
**[pre]** preprint / technical report / model card; **[READ]** primary source read directly
during design; **[survey]** obtained through the three literature surveys of 2 September
2026 (their reports are in the session record); **[repo]** measured in this repository.

## 10.1 The genomic-bottleneck lineage

**E-1 · Shuvaev, Lachi, Koulakov & Zador, PNAS 121(38):e2409160121 (2024)** [PR] [READ]
(publisher PDF, end to end; bioRxiv v1 2021 and v2 2022 via the deep-read).
*Relied upon:* the g-network is a 3-layer MLP `20 → H → {5,10} → 1` on two concatenated
10-bit labels (Gray for spatial coordinates, binary for unordered units, one-hot for
classes), one per layer; training is "intermittent": gradient training of the p-network,
**MSE regression** of each g-network onto the current weights, regeneration
`W_n = eps(n) W_{n-1} + (1 - eps(n)) W_hat`, `eps(n) = exp(-n/20)` over 500 generations;
end-to-end backprop through the decoder was tried and abandoned as inefficient; MNIST 94% at
birth at 322×, 79% at 1,038×; CIFAR-10 76% at 92× (trained 89%); Atari nearly perfect at
birth to ~3,500×; **compression did not change the learning trajectory** ("hot-started");
lower layers transfer better through the bottleneck (1.5× faster to 50% on SVHN with layer
1), upper layers worse; MNIST → F-MNIST fails; `H log N` scaling of the genome; proposed and
not done: one genome across layers, stochastic decoding, co-optimising learning rules.
*Used in:* §00.4 X2, §03, §05.2 (regression), §05.4 (the `eps` dial), §05.5 (birth share,
shuffle control), §02.6 (stochastic decoding), §01.6.
*Failure mode:* every result is a compression of an already-trained network on MNIST /
CIFAR / Atari; no sequence model; the g-network's activation is never stated; single runs
without error bars; the authors call the loop Lamarckian.

**E-2 · Zador, Nat. Commun. 10:3770 (2019)** [PR] [survey]. Genome ≈ 10⁹ bits vs ≥ 10¹⁵ bits
to specify cortical connectivity; the bottleneck as a regulariser; "supervised evolution".
*Used in:* §00 framing only.

**E-3 · Barabási et al., Nat. Commun. 14:2226 (2023)** [PR] [survey]: `W = X O X^T`, plain
form a poor compressor, spatial form width-independent, MAML-encoded models do not overfit.
**Mitchell & Cheney, Trends in Genetics 41(6) (2025)** [PR] [survey]: the genome as a
VAE-like generative model; robustness → cryptic variation → evolvability. **Hamidi et al.,
arXiv:2606.28380 (2026)** [pre] [survey]: g-net-generated reservoir weights need 2–6 SVD
components vs > 25 for direct training; constant genome size as modules are appended.
*Used in:* §03's bilinear interaction (`O_m`), §02.6's generative prior, §06.1.
*Failure mode:* MNIST / reservoir scale; GEM stores per-neuron vectors (`O(N)`), which E-1
argues a genome cannot; Hamidi is a preprint without accuracy tables.

## 10.2 Growth

**E-4 · Du et al., "Stacking Your Transformers", NeurIPS 2024 spotlight, arXiv:2405.15319**
[PR] [survey]: depth-wise stacking reaches the from-scratch loss at 194B of 300B tokens at 7B
(**54.6%** speedup); 410M/400B 53.1%, 410M/700B 31.0%; growth factor `g in [2,4]`; timing law
`log10 d = 0.88 log10 N + 163.27/log10 C - 5.74`. *Used in:* §06.1, §06.5. *Failure mode:*
the gain **decays** with training length; one-shot growth beat gradual in their setup.

**E-5 · Width and layer expansion 2024–2026** [survey]: HyperCloning (Apple, arXiv:2409.12903)
2.2–4× faster to baseline accuracy with early catastrophic forgetting [pre]; zero/one-layer
expansion (Bu, arXiv:2511.04981, ICML 2026) ~5× on GPT-2-style 7B [PR]; SPARKLING
(arXiv:2602.02472, ICML 2026) mid-stage 2× width growth with RMS-scale preservation and an
asymmetric optimiser reset, up to 35% [PR]; "When is warmstarting effective?"
(arXiv:2605.13405) 2× growth most reliable, gains largest at ≤ 20 tokens/parameter [pre];
Masked Structural Growth (ICLR 2024, arXiv:2305.02869) function-preserving masks [PR].
*Used in:* §06.2 step 5, §06.3, §06.6 B-COPY. *Failure mode:* gains depend on the budget
regime; MSG treats LayerNorm only (the RMSNorm form (3.4) is this repository's, verified).

**E-6 · Frankle, Dziugaite, Roy & Carbin, ICML 2020, arXiv:1912.05671** [PR] [repo ledger
C-2]: linear mode connectivity is established at 0–20% of training across settings.
*Used in:* §07.1 (why siblings merge). *Failure mode:* vision CNNs; no LM study.

## 10.3 Evolution strategies and populations

**E-7 · EGGROLL, Sarkar et al., arXiv:2511.16652 (v2 2026)** [pre] [survey]: low-rank
perturbations `E = A B^T / sqrt(r)`, aggregate update full-rank once `N r > min(m,n)`,
`O(1/r)` convergence to full-rank ES; int8 RNN pretrained to 3.40 bits/byte vs 3.58 for an
fp32 SGD transformer at ~180× the GPU-hours. **Qiu et al., arXiv:2509.24372** [pre]: ES
fine-tuning with `N = 30` beats GRPO on Countdown; ARC-AGI 0.2% → 29.5% on Qwen2.5-14B.
*Used in:* §07.3 (the low-rank form of mutation); §00.4 X4 (why ES is not the optimiser).
*Failure mode:* no ES pretraining of a transformer exists; 180× compute.

**E-17 · Merging** [survey]: model soups (Wortsman et al., ICML 2022) [PR]; task arithmetic /
TIES (Ilharco et al. 2023; Yadav et al., NeurIPS 2023) [PR]; Branch-Train-Merge
(arXiv:2208.03306) and BTX (arXiv:2403.07816) [pre]; DiLoCo (arXiv:2311.08105) [pre]; PMA
(arXiv:2505.12082) [pre]. No 2025–2026 work runs a population of LMs with diversity pressure
or crossover during pretraining; every working population-like method is merge-based.
*Used in:* §07.1, §07.4 (7.1). *Failure mode:* merges of divergent branches lose; (7.2)
bounds the loss.

**E-18 · Case-based selection** [survey] [repo ledger C-11, C-12]: Ding & Spector, ICLR 2022
(lexicase over `p = 4` whole networks, +0.25–0.68% CIFAR at ~4× compute) [PR]; La Cava et al.
ε-lexicase [PR]; Helmuth & Spector down-sampled lexicase [PR]. *Used in:* §07.4. *Failure
mode:* gains on vision are small and the compute cost was not matched; no LM precedent.

## 10.4 The backbone

**E-8 · Gated DeltaNet, Yang, Kautz & Hatamizadeh, ICLR 2025, arXiv:2412.06464** [PR]
[survey]: `S_t = S_{t-1} alpha_t (I - beta_t k_t k_t^T) + beta_t v_t k_t^T`; ablation at
400M/15B — removing the decay gate +3.52 ppl, L2 → L1+SiLU +2.83, output gate +1.77, short
conv +1.60, output norm +0.20; FineWeb-Edu 1.3B/100B Wiki ppl: Transformer++ 18.53, Mamba2
16.56, DeltaNet 17.71, GDN 16.42, GDN-H1 (GDN + SWA) **16.07**. *Used in:* §04.4 (conv and
output gate restored), §00.6. *Failure mode:* reference code is non-commercially licensed
and is not consulted; implementation is from the equations (SOMA (5.6)–(5.16), verified).

**E-9 · Hybrid ratio** [survey]: Qwen3-Next (model card: 3 GDN : 1 gated attention) [pre];
Kimi Linear (arXiv:2510.26692: 3:1 5.65 ppl vs 7:1 5.70) [pre]; GLM-5.3-Flash and
Qwen3.8-Flash-Next both 3:1 [pre]; systematic analysis of 72 hybrids (arXiv:2507.06457)
recommends 3:1–6:1 [pre]. *Used in:* §04.5. *Failure mode:* second-order effect; the ratio
is not searched here (SCREEN).

**E-10 · Muon** [survey] [repo ledger C-20]: Jordan 2024 (blog) 1.35× vs AdamW at 124M;
"Muon is Scalable" (arXiv:2502.16982) ≈ 52% of AdamW's FLOPs on a dense scaling law
399M–1.5B [pre]. *Used in:* §04.6. *Failure mode:* Newton–Schulz convergence at scale
(arXiv:2505.04005); not a concern below 1B.

**E-11 · Peri-LN (ICML 2025, arXiv:2502.02732)** [PR]; **QK-norm** (arXiv:2302.05442;
Wortsman et al. arXiv:2309.14322) [PR] [repo ledger C-25, C-26]. *Used in:* §04.3, §04.4.

**E-12 · Gated attention, Qwen, NeurIPS 2025 best paper, arXiv:2505.06708** [PR] [survey]:
a head-wise sigmoid gate after SDPA removes attention sinks (first-token mass 83% → 4%),
tolerates larger learning rates; deployed in Qwen3-Next. *Used in:* §04.4 (replaces GNOMON's
sink column; enables plain SDPA). *Failure mode:* measured at 1.7B/15B MoE scale; the gain at
our scale is measured by R3.

**E-13 · Schedules** [survey]: Hägele et al., NeurIPS 2024 spotlight (arXiv:2405.18392):
cooldown ≈ 20% matches cosine; `1 - sqrt` shape beats linear [PR]; "Straight to Zero"
(arXiv:2502.15938): linear decay to zero beats cosine-to-10% [pre]. *Used in:* §04.6.

**E-14 · Batch size** [survey]: Zhang et al., ICLR 2025 (arXiv:2410.21676): CBS scales with
data, weakly with model size [PR]; "CBS revisited" (arXiv:2505.23971): CBS ≈ 0 at init →
batch warm-up saves 43% of steps on OLMo-1B [pre]. *Used in:* §04.6's `B(n)`.

**E-15 · Softcap and z-loss**: Gemma 2 (arXiv:2408.00118) [pre]; PaLM (arXiv:2204.02311)
[pre]. *Used in:* (4.5). **E-16 · modded-nanogpt records** [non-archival] [survey]: the
124M speedrun's technique list (QK-norm, value embeddings, U-net skips, softcap, Muon, batch
and sequence schedules, retie/untie at 2/3), record 89 at 1.23 min on 8×H100. *Used in:*
§04.3–§04.6 as the recipe's source; value embeddings as `VE`. *Failure mode:* non-archival;
several tricks are metric-specific (long-sequence eval); each is re-measured here.

## 10.5 Theory

**E-19 · Description length**: Hinton & van Camp, COLT 1993 [PR]; Blier & Ollivier, NeurIPS
2018 (arXiv:1802.07044) [PR]; McAllester 1999 PAC-Bayes [PR]. *Used in:* §05.3 Proposition
1. *Failure mode:* the bound applies to the birth model, not to the adult after a lifetime
(§05.3 says so).

**E-23 · Fast weights and the delta rule**: Schlag, Irie & Schmidhuber, ICML 2021
(arXiv:2102.11174) [PR]; Yang et al. DeltaNet, NeurIPS 2024 [PR]; TTT layers (Sun et al.,
ICML 2025, arXiv:2407.04620) [PR]; TTT-E2E (arXiv:2512.23675) [pre]; Nested Learning /
HOPE (NeurIPS 2025, arXiv:2512.24695) [PR]. *Used in:* §05.1 (the token-level instance;
the continual-learning extension of §09.4). *Failure mode:* HOPE's own end-to-end gain is
small and its ablation table is internally inconsistent (repo ledger C-19).

## 10.6 Benchmarks and hardware

**E-20 · TinyStories** [survey]: Eldan & Li, arXiv:2305.07759 [pre] (GPT-4-graded, no
headline loss); llama2.c README: 15M 1.072, 42M 0.847, **110M 0.760 nats** with the Llama-2
32K tokenizer [non-archival]; tiny dense-vs-sparse study (arXiv:2605.13769): 14.9M dense
1.6545, 21.1M dense 1.5608 with a 30K SentencePiece vocab at 426M tokens [pre]. *Used in:*
G-BASE (in bits per byte, because tokenizers differ). *Failure mode:* no two of these share a
tokenizer; a search-engine claim of "1.11 nats at 160M" was not found in the cited paper.

**E-21 · The Pile** [survey]: Mamba paper Table 3 (Gu & Dao, COLM 2024, arXiv:2312.00752)
at 300B Pile tokens, NeoX tokenizer: Pythia-160M ppl 29.64 (3.39 nats), Mamba-130M 10.56
(2.36), Pythia-1.4B 7.51 (2.02), Mamba-1.4B 6.80 (1.92) [PR]. *Used in:* G-BASE (Pile),
context only at matched tokens. *Failure mode:* our budget is 100× smaller; comparison is by
matched-token interpolation of published curves where available, else context only.

**E-22 · Single-GPU throughput** [survey]: llm.c on an RTX 4080 Super ≈ 73.7k tok/s at 124M,
MFU 56.7% (primary page not retrievable — **unverified**); single-4090 speedrun port ≈ 160k
tok/s peak, 3.28 in ~90 min at 0.92B tokens [non-archival]; scaling by `6N`: ≈ 26k tok/s at
350M, ≈ 9k at 1B on this card; memory: AdamW ≈ 16 B/param, 1B does not fit. *Used in:* §04.8
targets (asserted by profiling, never assumed).

**E-25 · This repository's own measurements** [repo]: `sigma_e = 0.0100` nats under CRN at
d = 128 / L = 4 / 2,000 steps; operator axis 18.5 `sigma_e` and monotone; SOMA 0.04%; GNOMON
5.99 nats final; frequency-rank code +0.0999 nats. *Used in:* §00 throughout.

## 10.9 Evidence for the programme genome (§12)

Added after §11. Every row was re-verified against the primary source on 2026-09-02 (arXiv
abstract and HTML pages); the numbers below are the sources' own, with the section, table or
figure they come from.

**P-1 · Xie et al., "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining",
NeurIPS 2023, arXiv:2305.10429** [PR] [READ: abstract, §3–4, Fig. 2–4, 6, Tables 3–4].
*Relied upon:* two 280M proxy models (Group DRO against a reference), 8% of the 8B model's
FLOPs (§3.1, Fig. 2 caption); the 8B model trained on the proxy's domain weights reaches the
baseline's downstream accuracy in 75k of 200k steps, **2.6x** (§3.2), and improves average
one-shot accuracy by **6.5 points** (Fig. 2; "few-shot" in the abstract); log-perplexity
improves on **all 22** Pile domains, including the down-weighted ones (Fig. 4, App. Table 4).
*Used in:* §12.0, §12.4 (the mixture landscape is interior and transfers 30x).
*Failure mode:* a 1B proxy underperforms the 280M proxy and even the 1B baseline (§4, Fig. 6,
Table 3a; "Group DRO is worse at larger scales"); the gain is on a 22-domain corpus and says
nothing about a homogeneous one.

**P-2 · Liu et al., "RegMix: Data Mixture as Regression for Language Model Pre-training",
ICLR 2025, arXiv:2407.01492** [PR] [READ: abstract, §3–4, Tables 2, 4].
*Relied upon:* 512 models of 1M non-embedding parameters, 1B tokens each, on random
mixtures; a regressor (ridge and LightGBM; LightGBM Spearman 97.1% vs linear 88.0%, Table 2)
from mixture to loss; the extrapolated optimum applied to a 1B/25B-token model (1000x larger,
25x longer) is the best of 64 candidate mixtures; the proxies cost ~2% of one 1B model's
FLOPs (§1 fn. 3) and RegMix matches or beats DoReMi at 10% of DoReMi's cost (Table 4:
3.5e18 vs 3.7e19 FLOPs; 14-benchmark average 47.3 vs 46.8 DoReMi vs 45.1 human).
*Used in:* §12.5 (the programme regression is this map; the population is the proxy set).
*Failure mode:* downstream-benchmark gains of 0.5–2 points; the loss surrogate is fit on
Pile-CC validation loss and its transfer is measured, not derived.

**P-3 · Li et al., "Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language
Models", arXiv:2208.03306 (2022)** [arXiv] [READ: §4, Tables 1, 3, Fig. 7].
*Relied upon:* experts branched from a shared seed model (50% of updates) and trained on
domains; **posterior-weighted parameter averaging** beats the compute-matched dense
Transformer-LM at 350M/750M/1.3B on both train and eval domains (Table 3; 1.3B: 13.4 vs 14.2
train, 17.0 vs 18.4 eval PPL), uniform averaging is worse than dense at every scale, and the
ensemble is better still (1.3B eval 16.3); the 64-domain forest matches a dense LM trained
with 2.5x the compute (Fig. 7).
*Used in:* §12.3 line 8 (siblings on different data from one warm start, merged in weight
space), §07.4 (7.1)'s fitted coefficients rather than a uniform average.
*Failure mode:* at 125M the average loses on eval domains (28.5 vs 25.2); the merged model
is worse than the ensemble; the domains are given, not learned.

**P-4 · Jaderberg et al., "Population Based Training of Neural Networks", arXiv:1711.09846
(2017)** [arXiv] [READ: §4.2.2].
*Relied upon:* a population of 32 Transformers on WMT14 En-De with learning rate and dropouts
under exploit/explore improves newstest2012 BLEU 23.71 → 24.23 and test BLEU 22.30 → 22.65.
*Used in:* §12.0 (`eta`), §12.3 line 4.
*Failure mode:* no language-modelling perplexity result exists in the paper (an earlier
draft of this ledger's author's memory placed one there; it does not); gains are small and
on a tuned baseline.

**P-5 · Hu et al., "MiniCPM: Unveiling the Potential of Small Language Models with Scalable
Training Strategies", arXiv:2404.06395 (2024)** [arXiv] [READ: §3].
*Relied upon:* hyperparameters found on 0.009B–0.5B models under Tensor-Programs (`muP`)
parametrisation transfer to 1.2B/2.4B (`scale_depth 1.4, scale_emb 12, init_std 0.1,
lr 0.01`); the batch-size law `bs = 1.21e9 / L^6.24`.
*Used in:* §12.0 (`eta` in scale-free units is a lineage object).
*Failure mode:* hyperparameters only; MiniCPM's data mixture is not set by small models.

**P-6 · Du et al., "Stacking Your Transformers", NeurIPS 2024, arXiv:2405.15319** [PR]
[READ: Fig. 1, 3, 4, 6, Eq. 2]. *Relied upon:* G_stack at 7B reaches the scratch model's
300B-token loss with 194B tokens (**54.6%**, Fig. 1); 1.1B 49.1%, 3B 54.5%; the timing law
`log10 d = 0.88 log10 N + 163.27 / log10 C - 5.74` (Eq. 2, §4.2; `d` = tokens before
stacking, `N` non-embedding parameters, `C` total FLOPs; the `C` term is `b / log10 C`,
which corrects the plan's transcription), growth factor 4.
*Used in:* §06.5 (`omega`'s prior), §12.0.
*Failure mode:* depth stacking of a fully trained small model; width growth is not covered.

**P-7 · Samragh et al., "Scaling Smart: Accelerating Large Language Model Pre-training with
Small Model Initialization", arXiv:2409.12903 (2024)** [arXiv] [READ: §2, §3, App. A].
*Relied upon:* the function-preserving width expansion `W_D = [[W_S/2 + eta_1, W_S/2 -
eta_1], [W_S/2 + eta_2, W_S/2 - eta_2]]`, `b_D = [b_S; b_S]`, query scaling `sqrt(d_S/d_D)`;
convergence speed-up 2.2x–4x (Fig. 2; OPT-350M→1.3B, Pythia-410M→1.4B, OLMo-1B→2.9B);
symmetry-breaking noise has "minimal" benefit (§3.4).
*Used in:* §06.2 B-COPY, §12.7.
*Failure mode:* author-reported speed-ups at n = 2; R5 re-measures.

**P-8 · Kang et al., "AutoScale: Scale-Aware Data Mixing for Pre-training LLMs", COLM 2025,
arXiv:2407.20177** [PR] [READ: abstract, §3]. *Relied upon:* the mixture that is optimal at
small scale is not optimal at large scale; a direct data-optimisation plus extrapolation
gives GPT-2-Large 28% faster perplexity reduction and up to 38% over unweighted training.
*Used in:* §12.4 (K-MIX is evaluated on the adult), §12.5 (trust region).
*Failure mode:* the extrapolation is itself a fitted law.

**P-9 · Chen et al., "Aioli: A Unified Optimization Framework for Language Model Data
Mixing", ICLR 2025, arXiv:2411.05735** [PR] [READ: abstract, §5]. *Relied upon:* "no
existing method consistently outperforms a simple stratified sampling baseline"; Aioli beats
stratified on 6/6 datasets by 0.27 test-PPL points.
*Used in:* §12.9 (the honest expectation on TinyStories), G-LIVE-mu.
*Failure mode:* online, on one model; the gains are small.

**P-10 · the mixture-law family** [arXiv/PR] [survey]: Ye et al. "Data Mixing Laws" (ICLR
2025, arXiv:2403.16952: the default RedPajama mixture needs 48% more steps than the optimised
one at 1B/100B); Fan et al. "DoGE" (ICML 2024, arXiv:2310.15393: 684M avg PPL 15.81 vs 16.53
uniform from an 82M proxy); Ge et al. "BiMix" (arXiv:2405.14908); Diao et al. "CLIMB"
(NeurIPS 2025, arXiv:2504.13161: 350M proxies + LightGBM, 60.41 vs RegMix 59.37 vs random
57.93 at 1B/400B); Belenki et al. "MDE" (arXiv:2502.15950); Shukor et al. "Scaling Laws for
Optimal Data Mixtures" (arXiv:2507.09404, joint-law MRE 0.24–0.57%); Llama 3
(arXiv:2407.21783 §3.2.1: small models on candidate mixes predict the large model).
*Used in:* §12.0 (the principle that the mixture is the object small populations can set for
large models is replicated across eight groups).
*Failure mode:* every gain is on a multi-domain corpus.

**P-11 · Decoded transformer language models** [survey, 2026-09-02]: no published work decodes
a transformer LM's weight matrices from neuron identity codes and reports its loss. Falk et
al. (arXiv:2510.02096) generate a GPT-2-scale initialisation from a weight-space autoencoder
and find it "slightly above or at random guessing" at birth (§4.4); Hyper-Compression (Fan et
al., arXiv:2409.00592) reaches 2.6x on LLaMA-2-7B by an ergodic lookup, not a learned decoder.
*Used in:* §11 (the measurements of R4 and the probe are, to the author's knowledge, the first
quantitative ones for a language model and agree with the one adjacent data point).

**P-12 · Frankle et al., "Linear Mode Connectivity and the Lottery Ticket Hypothesis", ICML
2020, arXiv:1912.05671** [PR] [READ: Table 1]. *Relied upon:* linear mode connectivity to SGD
noise becomes stable at 1.5–3% of training for CIFAR-10 ResNet-20/VGG-16 and at 16–20% for
ImageNet ResNet-50/Inception-v3.
*Used in:* §07.4 (siblings from one warm start lie in one basin).
*Failure mode:* **no language model was tested**; the plan's "LMC is established within the
first few percent" is a vision result, and the merge's acceptance rule (7.2) is what protects
the design if it fails to transfer.

## 10.7 Claims deliberately not relied upon

| not relied upon | why |
|---|---|
| open-endedness / QD improving a language-modelling objective | no instance exists (SOMA §06.0; the 2026 survey finds none) |
| evolved or meta-learned plasticity in LMs | Backpropamine's ~1.8 ppl on a 4.8M LSTM (ICLR 2019) is the only number; the working descendant is the delta rule (E-23) |
| ES as a pretraining optimiser | 180× GPU-hours for a worse-than-Adam result (E-7) |
| MoE at this scale | negative under total-parameter matching below ~100M (arXiv:2605.13769); positive only when FLOP-bound at fixed memory, which a 16 GB card is not |
| small-model ARC results as evidence of reasoning | the TRM analysis (arXiv:2512.11847): puzzle-id conditioning gives 0% without it; gains are task conditioning and test-time compute (E-28) |
| heritability as evidence of evolvability | SCREEN: `h2 = 0.991` on a landscape a hill-climb solves in four flips |
| any frontier leaderboard figure | aggregator-sourced; not used anywhere |

## 10.8 Numbers flagged as insufficiently verified

**[NORMATIVE]** None may be the sole support for a decision; each is marked where used.

1. The RTX 4080 Super llm.c throughput (E-22): primary page not retrieved; §04.8 measures.
2. TinyStories loss references (E-20): tokenizer-incomparable; G-BASE compares in BPB
   against a run reproduced here with the same tokenizer.
3. HyperCloning's and the layer-expansion papers' speedups (E-5): author-reported; R5
   re-measures B-COPY against B-FLAT at this scale.
4. Gated attention's learning-rate tolerance (E-12): measured at 1.7B; R2 sweeps here.
5. The plan's transcription of G_stack's timing law had `b · log10 C`; the paper's Eq. 2
   has `b / log10 C` (P-6). The plan's "PBT on Penn Treebank" does not exist (P-4).
