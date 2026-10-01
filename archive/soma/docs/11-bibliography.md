# SOMA — Specification 11: Evidence Ledger

**Spec version 3.1** (audit-v1, audit-v2 and the 26 defects A-46...A-71 found by running the code, repaired). Normative for all claims. Every borrowed number used anywhere in this specification
appears here with: the exact claim relied upon, the source, its peer-review status,
whether it is author-reported or independently replicated, and its known failure mode.

**Status tags.** `[PR]` peer-reviewed venue · `[PP]` preprint · `[AR]` author-reported
only · `[IR]` independently replicated or audited · `[READ]` primary source read directly
during design · `[EXTRACT]` obtained by automated extraction, single pass, **not** to be
the sole support for a decision · `[UNVERIFIED]` could not be confirmed.

---

## 11.1 Claims the architecture depends on

### C-1 · Connection cost produces modularity, evolvability, and better performance
**Used in:** §01.6, §03.2, §08 A1, §10 K4.
**Claim relied upon:** minimising *summed squared connection length* alongside performance
yields modularity `Q = 0.42` vs `0.18` (`p = 8e-9`); performance `1.0` (perfect) vs `0.98`
(`p = 2e-5`); generations-to-adapt `3` vs `65` and `12` vs `222.5`
(`p = 3e-78`, `9e-120`); functional decomposition in `56%` vs `0%` of trials (`p = 4e-11`).
Modularity does **not** arise from modularly-varying goals alone.
**Source:** Clune, Mouret & Lipson, *The evolutionary origins of modularity*,
Proc. R. Soc. B 280:20122863 (2013). arXiv:1207.2743. **[PR] [READ]** (full text via PMC).
**Failure mode / limits:** evolved networks of order `1e2` nodes on a synthetic
retina task; no language-model or gradient-trained precedent. Four orders of magnitude
from our setting.
**Why we still use it:** it is the only result that explains *why* ANN topology is uniform
and dense, and its mechanism (a cost term) is trivially portable. A1 tests it here.

### C-2 · The first small fraction of training selects the basin, and the window widens with scale
**Used in:** §03.5, §07.5.
**Claim relied upon:** networks become stable to SGD noise (linear-mode-connected, error
barrier `< 2%`) at approximately `0%` (LeNet/MNIST), `1.5%` (VGG-16/CIFAR-10), `3%`
(ResNet-20/CIFAR-10), `16%` (Inception-v3/ImageNet), `20%` (ResNet-50/ImageNet) of
training. Pruned subnetworks match full accuracy **only when stable**.
**Source:** Frankle, Dziugaite, Roy & Carbin, *Linear Mode Connectivity and the Lottery
Ticket Hypothesis*, ICML 2020. arXiv:1912.05671v4. **[PR] [IR] [READ]**.
**Failure mode / limits:** vision CNNs only. **No critical-period study exists for
transformers or language models.** Our use of the window is therefore an extrapolation
across architecture family, and is marked as such wherever it appears.
**Corroboration:** critical periods appear even in deep *linear* networks (Kleinman,
Achille & Soatto, ICLR 2024) **[PR]**, so the phenomenon is a property of depth and
gradient dynamics, not of a particular nonlinearity — which materially strengthens the
extrapolation.

### C-3 · Diversity injection is necessary to maintain plasticity
**Used in:** §03.4, §05.7, §10 K8.
**Claim relied upon:** *"Plasticity is maintained indefinitely only by algorithms that
continually inject diversity into the network."* Binary-classification accuracy falls from
`89%` to `77%` — about the level of a linear network — by the 2000th task. Correlates:
dying units, growing weight magnitude, falling effective rank. Adam's plasticity loss is
described as catastrophic. Dropout is **worse** than plain backprop. L2 alone does not fix
it; shrink-and-perturb nearly does but is highly hyperparameter-sensitive.
Contribution utility: `y = (|h - h_bar| * sum_out|w|) / (sum_in|w|)`, `eta = 0.99`,
replacement rate `rho = 1e-4`, maturity threshold `100` (SGD) / `1000` (Adam); incoming
weights resampled, **outgoing set to zero**.
**Source:** Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood & Sutton, *Loss of plasticity
in deep continual learning*, Nature 632:768–774 (2024); preprint arXiv:2306.13812; earlier
arXiv:2108.06325. **[PR]** top venue, preprint **[READ]**.
**Failure mode / limits:** the utility measure is explicitly acknowledged by the authors as
**heuristic and not global** — it does not consider a unit's effect on the represented
function — and they flag recurrent networks as needing better ranking. Vision and control
domains only. **Unit conversion:** the cited replacement rate is `1e-4` **per step**,
while §03.5's `rho_0` is a fraction **per selection event**; at M1 `rho_0/T_0 = 5.3e-5` per
step, within a factor of two once converted. §03.5 states the conversion because K8 rests on
turnover volume.

### C-4 · Duplication is the best growth operator; the timing law; growth advantage decays
**Used in:** §03.2, §03.5, §08 A8, §10 R2.
**Claim relied upon:** across `{direct duplication, learned map, zero-init, masked random}`,
whole-module **duplication wins**. Timing law
`log10(d) = 0.88 log10(N) + 163.27 / log10(C) - 5.74` with `N` raw non-embedding
parameters, `C = 6ND` FLOPs, `d` raw tokens; growth factor `g = 4` (useful range 2–4).
`7e9` params / `3e11` tokens: same loss at `1.94e11` tokens = **54.6% speedup**.
**Source:** Du, Luo, Qiu, Huang et al., *Stacking Your Transformers*, NeurIPS 2024
Spotlight. arXiv:2405.15319. **[PR] [AR] [READ]**.
**Verification performed here:** the timing law was checked numerically against all four
rows of the source's own guideline table and reproduces them to printed precision.
**Failure mode / limits — load-bearing:** growth happens **once** in the standard setup;
gradual multi-stage stacking **loses** to one-shot. The measured speedup **decays**:
`53.1%` at `4e11` tokens vs `31.0%` at `7e11` at fixed model size. The claim that the
advantage persists past `8e12` tokens is an **extrapolation of a log fit**, contradicted in
direction by the source's own two measured points. Speedups other than the `7e9` figure
were read from figures, not tables **[EXTRACT]**, and are not used for decisions.

### C-5 · Function-preserving growth independent of initialisation
**Used in:** §03.3.
**Claim relied upon:** masking plus normalization statistics that exclude new units gives
strict function preservation **regardless of how new weights are initialised**.
`y' = c (*) sigma(W'x) = [y;0]`; `mu' = (x'.c)/sum(c)`; `var' = ((x'-mu')^2.c)/sum(c)`;
attention grows by masking the **value** path only; depth by
`x_n = c (*) Layer(x_{n-1}) + (1-c) (*) x_{n-1}` with `c` a **per-coordinate** mask;
annealing `phi = 5000` steps, all values above `500` stable.
**Source:** Yao, Zhang et al., *Masked Structural Growth*, ICLR 2024. arXiv:2305.02869.
**[PR] [AR] [READ]** (Eq. 8/9 verified directly).
**Failure mode / limits:** the paper treats **LayerNorm only**; RMSNorm is absent. Our
RMSNorm form (§03.3, eq. 3.4) is **[OURS]** and carries a mandatory test. Partial
de-risking: an equivalent masked operator is used in the C-4 codebase on an RMSNorm
pre-norm stack. A claim of use at `1.01e11` parameters exists but is **[UNVERIFIED]**.

### C-6 · ES on phenotype weights is not affordable; ES selects for canalization
**Used in:** §02.1, §02.6.
**Claims relied upon:** (a) low-rank perturbations `E_i = (1/sqrt(r)) A_i B_i^T` with
`M <- M + (alpha/N) sum_i E_i f(M + sigma E_i)`, error `O(1/r)`; (b) the `2^20`-population
run *"required around 180 times more GPU-hours than the backprop baseline"*, reaching
`3.40` bits/byte against `3.58` for a **plain-SGD** transformer control; (c) ES
*"optimizes the average reward of the entire population, thereby seeking parameters that
are robust to perturbation"*, and ES policies are significantly more perturbation-robust
than policy-gradient ones.
**Sources:** (a),(b) Sarkar, Fellows, Duque et al., *EGGROLL*, arXiv:2511.16652 v2.
**[PP] [AR] [READ]**. (c) Lehman, Chen, Clune & Stanley, GECCO 2018, arXiv:1712.06568.
**[PR] [READ]**.
**How used:** (b) is the *negative* result that forces the Weismann barrier — it is used
to rule an option **out**, which is the most robust use of a single-source number.
(a) sizes the cross-run ES. (c) is the positive justification for ES on the germline.
**Failure mode / limits:** `3.40` bits/byte is a poor byte-level LM and the baseline used
SGD, not Adam. This number must never be cited as evidence that ES is competitive with
backprop.

### C-7 · Genome size can be made independent of unit count
**Used in:** §01.2, §09.3.
**Claim relied upon:** `W = X Omega X^T` with per-neuron identity; the **spatial** variant
derives identity from distance to 2-D Gaussian gene fields at `3` parameters per gene per
field, giving `P = G(G + 3L)` — **independent of neuron count**. Reported: `>80%` MNIST at
`<2%` of parameters; multi-layer `>90%` at `<0.1%`; on CIFAR-10 the **encoded** network
beat the unencoded one (`52.5%` vs `50.6%`); modularity is *"learned early and persists"*.
**Source:** D. L. Barabási et al., *Complex computation from developmental priors*,
Nature Communications 14:2226 (2023). **[PR] [AR]**.
**Failure mode / limits:** **plain (non-spatial) `W = XOX^T` is a poor compressor** —
`93%` MNIST needs `9` genes ≈ `90%` of baseline parameters. All compression comes from the
spatial prior. Optimised by **backpropagation**, not evolution, despite the framing. The
Heaviside is dropped for the learning experiments. Never tested above `~1e6` parameters.

### C-8 · No hypernetwork has generated full weights above ~1.4e6 parameters
**Used in:** §02.2.2 (the reason the genome is a parts list), §09.3, §10 K9.
**Claim relied upon:** the largest general full-weight generation is of order `1.4e6`
target parameters; LLM-scale "hypernetworks" generate low-rank adapter deltas over a
conventionally trained base, and one 2026 study required a `~2.5e9`-parameter hypernetwork
to serve a `1.5e9`-parameter target.
**Sources:** Universal Hypernetworks, arXiv:2604.02215 **[PP]**; hypernetwork knowledge-
injection scaling laws, arXiv:2607.19604 **[PP]**. Both **[EXTRACT]**.
**Corroborating primary read:** Shuvaev, Lachi, Koulakov & Zador, PNAS 121(38) (2024);
bioRxiv 2021.03.16.435261v2 **[PR] [READ]** — the g-network maps a pair of binary neuron
tags to one scalar weight; MNIST `322x` compression at `94%` at-birth accuracy; but for
deeper networks *"if g-networks are initialized randomly, they produce p-nets that are far
from the optimal fixed point... impossible to train"*, requiring
`W = (1-eps(n))W_hat + eps(n)W_{n-1}`, `eps(n) = exp(-n/20)`. **This failure mode is read
from the lineage's flagship paper directly and is the strongest available warning.**

### C-9 · Correct hypernetwork initialisation is mandatory
**Used in:** §01.5, §02.5.
**Claim relied upon:** variance composes multiplicatively through a weight-generating
layer, picking up a factor equal to the generating layer's fan-in; classical
initialisation therefore overshoots. Without the corrected rule, training *"failed to
enable training even with very small learning rates"*; one variant showed a **~270-epoch**
dead zone. **Batch normalization corrects vanishing but not exploding decoded weights.**
**Source:** Chang, Flokas & Lipson, *Principled Weight Initialization for Hypernetworks*,
ICLR 2020; arXiv mirror 2312.08399 (mirror posted 2023, venue 2020; both verified). **[PR]**; formulas cross-validated against the
author's own published reference implementation.
**Failure mode / limits:** the paper reports curves, not tables — there are **no quotable
final-loss numbers**. The qualitative failure modes are well attested; the magnitudes are
not available.

### C-10 · Token-level excess-loss selection improves language modelling
**Used in:** §06.2.
**Claim relied upon:** `L_Delta(x_i) = L_theta(x_i) - L_ref(x_i)`; train on the top `k%`;
`k = 60%` (1.1B) / `70%` (7B). 7B: `GSM8K 42.9 -> 66.9`, `MATH 22.2 -> 31.0`; baseline
reached ~`10x` faster; `+6.8` points over 15 general benchmarks. Token dynamics:
`low->low 51%`, `high->low 26%`, `high->high 11%`, `low->high 12%`. Ratio sweep flat from
`0.9` to `0.6`, **below baseline at `0.5`**.
**Source:** Lin, Gou, Gong et al., *Not All Tokens Are What You Need* (Rho-1),
NeurIPS 2024 (best-paper runner-up). arXiv:2404.07965v2. **[PR] [AR] [READ]**.
**Failure mode / limits:** authors note loss on *unselected* tokens rises substantially,
and that very large models may develop this inductive bias anyway. Verified only to `7e9`
parameters and `<1e11` tokens. Ratio-sweep digits used here come from the paper's
self-reference table, which is genuine tabular data; the main-text ratio figure is a
**figure**, not a table.

### C-11 · Case-based selection preserves specialists; the exact operator
**Used in:** §04.
**Claims relied upon:** the lexicase filter; `eps` per case from the **median absolute
deviation** `lambda(e_t) = median_j |e_{t_j} - median_k e_{t_k}|`; **semi-dynamic**
ε-lexicase recommended as default; selected individuals are ε-Pareto-set boundaries;
down-sampling flat from `0.25` to `0.02`, erratic at `0.01`; wall-clock `82–120%` of
tournament selection.
**Sources:** La Cava, Spector & Danai (GECCO 2016, corrected arXiv:1905.13266); La Cava
et al., Evolutionary Computation 2019 (arXiv:1709.05394); Helmuth & Spector, Artificial
Life 27(3–4):183 (2021), read via arXiv:2106.06085. **[PR] [READ]**.
**Failure mode / limits — important:** the down-sampling paper **does not measure
diversity or specialist preservation at all**; the specialist evidence is in a *different*
paper (Helmuth, Pantridge & Spector, GECCO 2019, arXiv:1905.09372), of which only the
abstract was read. Crucially, the source's own experiments give evidence **against** the
"changing environment" explanation and against a generalisation explanation; the surviving
explanation is that down-sampling *"allows evolution to consider more programs"*. §04.6
states it that way. Documented hazard: **hyperselection**, where a single individual
passing all subsampled cases takes every parent slot.

### C-12 · Lexicase in neural network training already exists — the novelty narrowing
**Used in:** §04.1, §10 R10.
**Claim relied upon:** a population of `p = 4` **whole networks**, each gradient-trained on
`1/p` of the data per generation, with lexicase selecting the surviving network case by
case. Gains `+0.25%` to `+0.68%` on CIFAR-10/100, `-0.39%` on one architecture, at ~`4x`
compute. Momentum must be reset each generation. Authors' limitations: *"may not work
directly with architectures that have been extensively optimized"*; integration with
architecture search is future work.
**Source:** Ding & Spector, *Optimizing Neural Networks with Gradient Lexicase Selection*,
ICLR 2022. arXiv:2312.12606 (the arXiv posting postdates the ICLR 2022 venue; both verified). **[PR] [AR] [READ]**.
**How used:** as **prior art that narrows our claim**. SOMA's novelty is unit-level,
in-model, learnability-weighted selection — individuals are units, not networks.
**Search caveat:** an arXiv full-text sweep found no application of lexicase to
transformers or language models, and the canonical applications list omits language
modelling. This negative is **probably genuine but not exhaustively verified**; a citation-
graph sweep of forward citations should precede any published novelty claim.

### C-13 · Signed transitions unlock state tracking
**Used in:** §05.2.2, §08 A7, §10 K7.
**Claim relied upon:** diagonal `Diag(s(x)) -> Diag(2s(x)-1)`; generalized Householder
`I - phi vv^T -> I - 2 phi vv^T`. Theorem: positive-real-eigenvalue LRNNs cannot recognise
parity at arbitrary length. Measured: parity `0.017 -> 1.000` (DeltaNet), `0.000 -> 1.000`
(Mamba); modular arithmetic `0.314 -> 0.971`; `S_5` with swaps solved by **one layer**.
Implementation is `beta = 2*sigmoid(...)` with the write term scaled by 2.
**Source:** Grazzi, Siems, Zela, Franke, Hutter & Pontil, ICLR 2025. arXiv:2411.12537v5.
**[PR] [AR] [READ]**.
**Failure mode / limits — load-bearing:** *"not directly compatible with a numerical
technique used by some diagonal LRNNs... which rely on positive state-transition matrices
to compute cumulative products in log space"*; an early chunkwise test of one such model
*"showed degraded performance"*. Also: median results are much weaker than best-of-3
(`0.826 +/- 0.146` vs `0.971`), and the modification **degrades** several language metrics
for the diagonal model. At `1.3e9` the language results are mixed. §05.2.3 analyses why
we expect composition to work and mandates the test rather than assuming it.

### C-14 · The mixer, its ablation priorities, and the hybrid ratio
**Used in:** §05.1, §05.2.
**Claims relied upon:** gated delta rule
`S_t = S_{t-1}(alpha_t(I - beta_t k_t k_t^T)) + beta_t v_t k_t^T`; WY representation and
UT transform for the chunkwise form; ablation at 400M/15B — removing the `alpha` gate
costs `+3.52` ppl / `-2.14` acc (largest), L1-vs-L2 norm `~+2.8`, output gate `+1.77`,
short conv `+1.60`, output norm `+0.20`; head dim 128 optimal.
**Source:** Yang, Kautz & Hatamizadeh, *Gated Delta Networks*, ICLR 2025. arXiv:2412.06464.
**[PR] [AR]**, equations **[READ]** via a verified extraction.
**Licensing:** the accompanying reference code is under a **non-commercial** licence and
is **not** consulted or vendored; §05 is implemented from equations only.
**Hybrid ratio source:** a 72-model controlled study (36 at 340M/20B, 36 at 1.3B/100B)
finding LM loss nearly flat across linear:full ratios while **recall** improves sharply
below 3:1, recommending 3:1–6:1, and warning that *"superior standalone linear models do
not necessarily excel in hybrids"*. arXiv:2507.06457 **[PP] [AR]**.
**Gaps:** the source does not state layer counts, `d_model`, head counts, chunk size, or
the `alpha` parameterisation; those are our choices and are marked as such in §05.

### C-15 · Combinatorial addressing scales
**Used in:** §02.2.2, §05.4.
**Claim relied upon:** product-key lookup with `K1, K2` of shape `sqrt(N) x (n/2)`,
top-`k` per half combined by `argmax s1[i1] + s2[i2]`, giving `O(sqrt(N))`; scaled to
`1.28e11` memory parameters (`6.4e7` keys) over `1e12` tokens; `1.3e9` base:
NaturalQuestions `7.76 -> 13.68`, TriviaQA F1 `32.64 -> 42.89`; a **single shared
parameter pool** across (optimally 3) layers.
**Source:** Memory Layers at Scale (Meta), arXiv:2412.09764. **[PP] [AR] [READ]**.
**Failure mode / limits:** authors state substantial engineering is required for
production efficiency; naive PyTorch achieves `~400 GB/s` against `3 TB/s` for their
custom kernels. Our use is for *unit addressing*, not memory lookup, which is an
extrapolation of the mechanism to a new purpose.

### C-16 · Computable open-endedness
**Used in:** §06.4.
**Claim relied upon:** a system is open-ended iff its artifact sequence is novel and
learnable with respect to an observer; novelty
`for all t, for all T>t, exists T'>T: E[l(t,T')] > E[l(t,T)]`; learnability
`for all T, for all t<t'<T: E[l(t',T)] < E[l(t,T)]`; *"interestingness is represented in
our definition by the observer's choice of loss function"*.
**Source:** Hughes, Dennis, Parker-Holder et al., ICML 2024 position paper.
arXiv:2406.04268. **[PR] [READ]**.
**Failure mode / limits:** it is a **position paper with no experiments**, and provides
*"limited practical guidance"* on measurement. §06.4's instantiation for language models is
**[OURS]**.

### C-17 · Regret-prioritised archives beat populations of learners on compute
**Used in:** §06.3.3.
**Claim relied upon:** a single learner plus a regret-prioritised archive with an edit
operator reaches comparable environment complexity at `~2.07e9` environment steps against
`~4.14e12` — under `0.05%`.
**Source:** Parker-Holder, Jiang, Dennis et al., ACCEL, ICML 2022. arXiv:2203.01302.
**[PR] [AR]**.
**Failure mode / limits:** reinforcement learning only; **no language-model precedent**.
The advantage over simpler prioritised replay is small in one domain and large in another,
i.e. it scales with how much the environment space rewards compounding edits — unknown for
our synthetic task space.

### C-18 · Verifier-free self-improvement sharpens rather than expands
**Used in:** §06.3.1, §08.6 rule 1.
**Claim relied upon:** RL-from-verifiable-rewards models win at `pass@1` but the **base
model wins at large `k`** (`k = 128` on one benchmark, `256–1024` on others); the RL
model's solvable set is nearly a subset of the base model's; distillation, by contrast,
does raise `pass@k` at all `k`.
**Source:** Yue et al., NeurIPS 2025. arXiv:2504.13837. **[PR]**. Corroborated by the
support-constraint analysis in arXiv:2507.14843 **[PP]**.
**How used:** it is why every generator in §06.3 is verifier-backed, and why `pass@k` at
large `k` is a mandatory reporting rule.

### C-19 · Multi-rate update gating
**Used in:** §01.7.
**Claim relied upon:** `C^(l) = (max_i C^(i)) / f_l`; update accumulates across a chunk and
applies only at chunk boundaries, zero otherwise; *"The conventional Transformer block is a
special instance of this formulation, where k = 1 and frequency of update is zero"*;
tokens off the boundary are parallelisable.
**Source:** Behrouz, Razaviyayn, Zhong & Mirrokni, *Nested Learning*, NeurIPS 2025.
arXiv:2512.24695. **[PR] [AR] [READ]**.
**Failure mode / limits:** the framework's own end-to-end gain over the same group's
previous single-level model is small (order `0.2–0.6` perplexity), and its ablation table
shows one removed component with **better** perplexity than the full model while the
caption asserts all components contribute — an instance of the reporting failure §08.6
rule 6 guards against. We adopt the **gating primitive**, not the performance claim.

### C-20 · Optimizer
**Used in:** §05.7.
**Claim relied upon:** Muon — Newton–Schulz orthogonalisation of the momentum matrix,
matrix parameters only, LR scaled by `0.2 sqrt(max(m,n))`, Nesterov `0.95`, weight decay
`0.1`; roughly `2x` compute efficiency against AdamW at compute-optimal training; scaled to
`1.6e10`-parameter MoE over `5.7e12` tokens.
**Sources:** Jordan (2024) blog **[non-archival]**; Liu, Su et al., *Muon is Scalable*,
arXiv:2502.16982 **[PP] [AR]**.
**Failure mode / limits:** a known concern that Newton–Schulz convergence degrades as
singular values of random matrices shrink with scale (arXiv:2505.04005 **[PP]**). Not a
problem at our scale; flagged for §09.

### C-21 · Small-scale wins that die at scale
**Used in:** §08.6 rule 5, §10 R4.
**Claim relied upon:** an adaptive-computation architecture beats its baseline at
`1.35e8`–`7.3e8` but **loses at `1.7e9`** (`48.4` vs `48.9` average); authors state it
*"might also signal that the current architecture design ... is not suitable for scaling"*.
Its advantage is largest at small budgets and shrinks with compute. Additionally: a
**random router** reaches `2.9335` NLL / `39.1` acc against the best learned router's
`2.8667` / `40.1`. Its Pareto claim rests on **analytic parameter-counting FLOPs** that
omit normalisation, non-linearity, attention-kernel overhead and router cost.
**Source:** Bae, Kim, Bayat et al., *Mixture-of-Recursions*, arXiv:2507.10524.
**[PP] [AR]**.
**How used:** as a cautionary constraint. It is why per-token recursion depth is demoted to
an ablated option, why B3 (random-router control) is mandatory, and why §07.8 requires
measured wall-clock for every efficiency claim.

### C-22 · Evaluation failure modes to avoid
**Used in:** §08.6.
**Claims relied upon:** an audited small-model reasoning result fell from a claimed `41%`
to a verified `32%`; the hierarchical architecture was **not** the driver (a vanilla
transformer came within ~5 points); the outer refinement loop was; and a `puzzle_id`
embedding meant that **ablating task identity gave zero accuracy**. A replication of a
related model found most accuracy arrives at the **first** recursion step and that
`1000`-sample voting was worth `~+11` points.
**Sources:** independent audit by the benchmark organisers **[IR]**; replication study
arXiv:2512.11847 **[PP] [IR]**.
**How used:** §08.6 rules 2 and 3 are written directly from these failures.

---

### C-23 · Batch-size law
**Used in:** §05.8 (5.22).
**Claim:** optimal and critical batch size are power laws in total tokens and approximately
independent of model size; fitted `B_tok ~ 22.91 * D_tok^0.47` tokens.
**Source:** Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training,
NeurIPS 2025, arXiv:2505.13738. **[PR] [AR]**.
**Limits:** fitted on dense transformers; SOMA's per-step cost varies with `|U_live|`, so the
law is used as a *starting point* and the realised batch is recorded, not asserted.

### C-24 · Cross-layer sharing asymmetry
**Used in:** §05.5 (why units live at the Site, not the mixer).
**Claim:** sharing attention parameters across layers costs approximately nothing
(`+0.1` to `-0.7` on a benchmark average) while sharing feed-forward parameters is where the
damage concentrates; smaller sharing groups are always better.
**Source:** ALBERT, ICLR 2020. **[PR] [AR]**.
**Limits:** encoder-only, masked-LM objective, 2019-era scale.

### C-25 · Peri-LN
**Used in:** §05.1.
**Claim:** Pre-LN shows variance growth exponential in depth; Post-LN has stable variance but
degraded gradient flow; normalising both sublayer input and output balances the two, validated
to 3.2B.
**Source:** Peri-LN: Revisiting Normalization Layer in the Transformer Architecture,
ICML 2025, arXiv:2502.02732. **[PR] [AR]**.

### C-26 · QK-norm and attention entropy collapse
**Used in:** §05.3.
**Claim:** attention entropy decays exponentially in the spectral norm of the attention
logits; bounding `||q||, ||k||` prevents the collapse and permits a higher stable learning
rate.
**Source:** Stabilizing Transformer Training by Preventing Attention Entropy Collapse,
ICML 2023, arXiv:2303.06296. **[PR]**.
**Limits:** the bound `|s| <= sqrt(d_h)` requires a **gain-free** norm; §05.3 specifies this
explicitly, which spec v1 did not.

### C-27 · Attention sinks
**Used in:** §05.3 (5.17).
**Claim:** softmax forces a distribution, so a head with nothing to retrieve dumps mass on
early tokens; providing an explicit learned sink logit is better than letting one emerge.
Keeping a small number of initial tokens permanently enables stable streaming at 4M+ tokens.
**Source:** StreamingLLM, ICLR 2024, arXiv:2309.17453 **[PR] [IR]**; explicit learned per-head
sink as shipped in a 2025 open-weight model card **[AR]**.

### C-28 · ReLU-squared
**Used in:** §05.5 (5.19).
**Claim:** `ReLU^2` is the activation used by current small-scale speed-record recipes and is
cheaper than SwiGLU at comparable quality in that regime.
**Source:** modded-nanogpt record log **[non-archival] [AR]**.
**Limits:** non-archival and at a single scale. **[OPEN]** O-2 retains SwiGLU as a config
switch pending an M2 A/B at matched parameters. `gain_act` for `ReLU^2` is **measured**
(§02.5), not taken from any source.

### C-29 · Embedding tying threshold
**Used in:** §05.6.
**Claim:** tying input and output embeddings is preferred below roughly 100M parameters and
untying above.
**Source:** practitioner consensus across open model releases **[SEC]**.
**Limits:** secondary-source only; this is a low-stakes default, and M-nano/M1 both sit well
below the threshold so the choice is uncontested at our scale.

---

## 11.2 Claims we deliberately do **not** rely on

Recorded so that no one later mistakes their absence for an oversight.

| Not relied upon | Why |
|---|---|
| Open-endedness / QD improving a language-modelling objective | **no instance exists** (§06.0) |
| Temporally-unrolled developmental programs | tested against one-shot decoding and showed **no benefit** |
| Pure evolution as a competitive LM optimizer | `180x` GPU-hours for a worse-than-Adam result (C-6) |
| Sparse-topology-from-scratch as a source of capability | pruned topologies cannot be trained from scratch to matching accuracy |
| Learned optimizers | an independent evaluation found the flagship system requires problem-specific tuning, does not find better solutions, and is not faster than tuned Adam |
| "Edge of chaos" training-data complexity | one correlational preprint, frozen backbone, linear probe, **no perplexity result**, no replication; and the classic version of this claim has a famous failed replication |
| Self-organized criticality as a control knob | descriptive only; no controlled intervention improving an LM metric |
| Biological Hebbian / three-factor plasticity rules | effectively zero LM evidence; the working descendant is the delta rule (C-14) |

---

## 11.3 Numbers flagged as insufficiently verified

**[NORMATIVE]** None of the following may be the sole support for a design decision.
Where one currently is, the decision is marked `[OPEN]` in its section.

1. Growth speedups other than the `7e9`/`54.6%` figure (read from figures) — C-4.
2. The `1.01e11`-parameter masked-growth claim — C-5.
3. Hypernetwork ceiling figures from 2026 preprints (single extraction) — C-8.
4. Chang et al. magnitudes: the paper reports curves only; there are no numbers — C-9.
5. Specialist-preservation evidence: abstract only, and in a different paper from the
   down-sampling results — C-11.
6. The absence of lexicase in language-model training: three non-exhaustive searches —
   C-12.
**[NORMATIVE — the rule is now enforced, not merely asserted]** Spec v1 stated that where an
insufficiently-verified number is the sole support for a decision, the decision is marked
`[OPEN]` — and then violated it in at least three places. The markers are now placed:

| Decision | Sole support | Marker location |
|---|---|---|
| §03.3 masked-growth operator at scale | C-5's `1.01e11` claim, **[UNVERIFIED]** | `[OPEN]` in §03.3 |
| §04.5 Proposition 4.3, specialist preservation | C-11, abstract-only, from a *different* paper than the down-sampling results | `[OPEN]` in §04.5 |
| §09.3 hypernetwork-ceiling contrast | C-8 ceiling figures, single extraction | `[OPEN]` in §09.3 |
| §04.1 novelty narrowing | C-12's negative search result, three non-exhaustive searches | `[OPEN]` in §10 R10 |

7. Any frontier leaderboard figure: aggregator-sourced, mutually inconsistent in at least
   one case, unverifiable against vendor documentation. **Not used anywhere in this
   specification.**

---

## 11.4 Provenance of this ledger

Fourteen sources were read directly during design (marked **[READ]**). A further set was
read in depth by paper-reading processes that were instructed to transcribe equations
rather than paraphrase, to flag every unverified digit, and to report explicitly when a
number was absent from the source rather than reconstruct it. Where two extraction passes
disagreed, the disagreement is recorded above rather than resolved by preference.

Two design options were eliminated purely on the strength of primary-source reading —
full-weight genome decoding (C-8) and evolution strategies on phenotype weights (C-6) —
and one novelty claim was narrowed (C-12). Recording eliminations is as much a function of
this ledger as recording adoptions.
