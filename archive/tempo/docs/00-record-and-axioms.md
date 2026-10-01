# 00 — The record, and the axioms of TEMPO

**Status: record and constitution.** This document states what this repository has measured
(§0.1), what the genomic-bottleneck paper actually established on re-reading (§0.2), the
requirements of the brief of 2026-09-04 (§0.3), the axioms TEMPO is built on (§0.4), what is
reused from ANLAGE and why (§0.5), and the process rule (§0.6). Nothing in it is softened:
negative results are results (project rule: research integrity).

TEMPO (*tempo*: the rate at which a passage moves; a piece has more than one) is the fourth
language-model design in this repository, after SOMA (`main`), GNOMON (`gnomon`) and ANLAGE
(`anlage`). The three before it were built to a research-grade standard of conformance and
each was measured to be inert or worse at the level of the mechanism it existed to test. The
measurements are the constraints below.

## 0.1 What this repository has measured

Every number is re-measured from the artifact named; the registers are
`docs/anlage/RESULTS.md`, `docs/gnomon/00-postmortem.md`, `docs/anlage/11-weight-genome-measured.md`.

| fact | artifact | consequence for TEMPO |
|---|---|---|
| A trained 13M language model's weight matrices have flat spectra (effective rank 0.80–0.94 of full); a per-unit latent of 16 numbers reconstructs 7–20 % of a row; one regression of the adult into a 718k-real neural-kernel genome decodes to 8.53 nats against a null of 9.01 and a trained 1.857 | `runs/anlage/birth-tinystories-s0-dg64H4z16/{birth,diag}.json` | the genome never carries weight matrices; it carries **gains, offsets, switches, learning rates** — the vector roles regressed to `< 1 %` residual in the same run |
| Training the decode end to end collapses to the uniform predictor (9.011 nats) with every decoded matrix at rank 1 | `runs/anlage/manifold-tinystories-s0-dg64H4z16h256/manifold.json` | no gradient path from a per-step loss into the genome |
| The PNAS-style assimilated lifetime (refit every 100 steps, pull toward the decode) cost 0.145 nats and decoded to 13.07 | `runs/anlage/sleep-tinystories-s0-anchor-e100-r100-a0.01-dg64H4z16-seed0/sleep.json` | the weights are never pulled toward a compressed image of themselves |
| In a loop where the adult's weights can be copied, a genome that compresses them can only lose information (axiom A8 of ANLAGE §11.3) | `docs/anlage/11-weight-genome-measured.md` §11.3 | the fast weights persist for life; the genome is an **additional slow channel**, not a replacement |
| Four short sibling lifetimes merged lose to one long lifetime of the same compute by 0.115 nats; the loop's population cost 0.011 and 0.007 nats at two generations | `runs/anlage/pop2-tinystories-s0/pop.json`, `runs/anlage/gen{0,1}-*/generation.json` | **no population**; all compute goes to one continuing agent |
| The data-mixture programme was **indistinguishable from noise** on a homogeneous corpus (+0.002 [−0.002, +0.005]: a null result, not a small positive one); the learning-rate axis moved 0.09 nats in the same screen | `runs/anlage/mix-tinystories-s0/mix.json`, `runs/anlage/live-tinystories-s0/live.json` | the genome's leverage on *how to learn* is large; on *what data* it is small unless the data are heterogeneous, which is why the local blend (02 §2.9) is built to be heterogeneous |
| The inherited scale profile (norm gains, gate offsets, logit scale) is worse at the instant of birth (+0.97) and better after 2.5M tokens on every seed (−0.014 [−0.019, −0.010]) | `runs/anlage/init-tinystories-s0/init.json` | gain-space is where a compact heritable object helps; it is also the criticality knob (07) |
| Copy-growth equals training the larger stage from scratch at matched FLOPs (+0.003 [−0.007, +0.014]); units seeded from a unit prior are worse than copies by 0.020 [0.011, 0.030] | `runs/anlage/grow-tinystories-s0to1/grow.json` | growth is a **copy** operation; its timing is learned; its benefit is only claimed when measured at a budget that can use it |
| A decode inside the training step costs the throughput (GNOMON: 42.9k vs 69.0k tok/s in the probe) | `docs/gnomon/00-postmortem.md`, `manifold.json` | the genome is expressed **once per episode**, never per token (G-NOFREE-INVERSE) |
| Seed noise under common random numbers at TS stage 0: `σ_e = 0.0060` (v1 sets), `0.0029` (v3 sets) | `runs/anlage/noise-tinystories-s0/noise.json` | every claim is a paired contrast against `σ_e`, measured first |
| Two GPU jobs on this card oversubscribe memory and one died with `cudaErrorIllegalAddress` | `runs/anlage/mix-launch.log` (06:46) | one serial chain per card |

## 0.2 What the genomic-bottleneck paper established (re-read in full, 2026-09-03)

Shuvaev, Lachi, Koulakov & Zador, *PNAS* 121(38):e2409160121 (2024). The mechanism is not a
single compression. It is five hundred rounds of

```
   train the network a little ──► fit the genome to the weights (MSE) ──► regenerate:
   W_n = ε(n) W_{n−1} + (1 − ε(n)) Ŵ_{n−1},   ε(n) = exp(−n/20)
```

so that after ~60 rounds the network is re-expressed from the genome every round and merely
retrained on top (Algorithm 1; Methods, "intermittent training"; Eq. 3). What it buys, in the
paper's own words:

* innate performance at birth: 94 % on MNIST at 322x, 79 % at 1,038x; 76 % on CIFAR-10 at
  92x (trained 89 %); Atari near-perfect to ~3,500x;
* **no change in the learning trajectory**: "genomic compression did not affect the learning
  trajectory; the only speedup was due to the higher initial performance, as though the
  p-network was 'hot-started'" (Fig. 3E; Discussion);
* transfer **indistinguishable from copying the weights** (CIFAR-10 → SVHN, all layers), with
  two exceptions: the lowest layers alone (50 % in 3.2 vs 4.8 epochs, 1.5x) and an
  environment shift (1.5x gravity), where the compressed network had learned *simpler, more
  general* structure (Gabor-like filters, Fig. 2F; conventional gaits instead of reward
  hacks, Fig. 5D); MNIST → F-MNIST fails; upper layers transfer worse than random;
* "if g-networks are initialized randomly, they produce p-networks that are far from the
  optimal fixed point ... impossible to train" (Methods), which is the collapse the probe
  above reproduced.

The bottleneck's real product is a **simplicity bias that transfers**, obtained by repeatedly
forcing the phenotype onto a compressible set. Three consequences are the spine of TEMPO:

1. **The compression is a process, not a snapshot.** Its value appears only when it runs
   hundreds of times over a life. It is therefore a standard, continuous part of the model.
2. **What it compresses is not the connectome.** Matrices are incompressible per unit (§0.1);
   what the paper shows transferring is low-layer general structure, and what a compact object
   can carry here is *gain structure*: the quantities that set what a fixed network computes
   and how fast it learns.
3. **Copying is not a competitor once the model is lifelong.** There is no next generation to
   copy into; the fast weights persist; the genome is a second, slower memory whose only job is
   to make the fast memory's learning better.

The brief's backward axis has a precise counterpart: hippocampal **reverse replay** (Foster &
Wilson, *Nature* 440:680–683, 2006), in which a just-experienced sequence is replayed backwards
during rest and written into slow structure.

## 0.3 The brief (2026-09-04), decoded into requirements

| id | requirement | operational meaning | where it is decided |
|---|---|---|---|
| B1 | MODEL = AGENT, mathematically and in training | one system whose state evolves under a single perception stream and a single output stream; "training" is the agent's life | 01 Definition 1; A1 (one code path) |
| B2 | egocentric, one input space | every modality, including the agent's own emissions, enters as one stream of continuous vectors; the backbone never sees a modality-specific token table | 02 (2.1); I14; the action's closure is through the world (02 §2.6) |
| B3 | developmental growth by 2026 RL and latent morphology, not open-ended evolution | no population; the active shape is a latent variable (blocks switched on over life); growth decisions are learned | 05; **K-MORPH** |
| B4 | RL optimises the genomic optimisation, which optimises training performance | RL trains the rule that updates the compact latent programme; the reward is the improvement of the agent's own subsequent learning | 06; **K-GENE**, K-HORIZON |
| B5 | lifelong continuous genomic encoding translating between domains and modalities | the compact latent carries what transfers across tasks and modalities | 04; **K-TRANSFER**, **K-XFER** |
| B6 | many modalities; pattern training at the edge of chaos | a modality-agnostic interface; criticality measured, then maintained if it drifts | 02 §2.9, 07; **K-BLEND**, K-CRIT (conditional) |
| B7 | the bottleneck is a standard, continuous part of the model | the compression runs inside the lifetime, hundreds of times | 04 §4.5 (the cycle at every boundary) |
| B8 | action space = token space: one continuous output; the model is continuous, the genome is not | the model predicts the next continuous vector; decoders are downstream; the genome updates discretely and rarely | 03 §3.7, 04 §4.7; **G-ACT**, K-CONT |
| B9 | two timescales inside the model, and a third that works backwards over the past | a fast per-span process; a slow per-episode genomic process; a reverse-time consolidation that writes the genome | 01 §1.2, 06 §6.2; **K-REV** |
| B10 | not decorative; core; beneficial | every mechanism has a matched-compute control and a kill criterion | 10 §10.2 in full |
| B11 | 2026 SOTA capability on benchmarks and agentic use | the architecture must scale; its mechanisms must be load-bearing at the scale this card reaches; the scaling path is stated, not claimed | K-SOTA is the local claim; the benchmark path is stated in 11 §P-A.13 and **not** claimed |

No row of this table is satisfied by the specification: each is a hypothesis with a named
criterion, and the criterion's verdict — including NOT EVALUATED — is what the register
reports (10 §10.4).

Decisions taken with the user on 2026-09-04: the control environment is a pure-torch
point-mass (no installs); the first build's modality scope is the *generated* blend of
02 §2.9 taken as far as local generation allows, with the genome as the one scale-free object
across every stream; pretrained image/speech encoders are interfaces with tests and no data.

## 0.4 The axioms of TEMPO

**[NORMATIVE]** Every design decision in 01–09 traces to one of these.

| # | axiom | replaces |
|---|---|---|
| A1 | **Sequential, not adversarial.** The genome is written from the lifetime by a consolidation process and read into the next episode as gains. Gradient descent and the genome never compete for credit. | ANLAGE A1, kept |
| A2 | **The bottleneck is priced, not imposed.** The genome is the first part of a two-part code; its size (`d_g` reals) is a dial; the compression ratio is reported, never targeted. | ANLAGE A2, kept |
| A3 | **Full-capacity lifetime.** The phenotype is a free-weight hybrid trained by the strongest recipe this repository has measured; the hot path is the baseline's; nothing constrains the weights to a decodable set. The genome acts on *gains*, which are parameters of the same network, so "unconstrained" means: no term of any loss pulls `W` toward an image of itself, and the only object the genome writes is the gain vector `u` at an episode's start. | ANLAGE A3, restated for gains |
| A4 | **One agent, no population.** All compute goes to one continuing life; variation that a population would search is searched by the consolidation rule over the genome. | ANLAGE A4, reversed by measurement |
| A5 | **Development is growth by switching, timed by the slow process.** The network is provisioned larger than it is born; blocks switch on when the consolidation rule raises their switch; a switched-off block costs no compute; a newborn block is a copy. | ANLAGE A5, revised |
| A6 | **One consolidation primitive across timescales.** Per span, the delta rule's exact least-squares step; per step, the gradient; per episode, a proposal from the reversed episode inside a trust region. The three are *analogous*, not the same operator: only τ0 is literally a least-squares projection, τ1 is a first-order step, and τ2 is a sampled proposal accepted by a probe. The axiom is a design principle — each level writes a smaller, slower memory from a faster one — and is not used as a mathematical identity anywhere in 01–10. | ANLAGE A6, restated honestly |
| A7 | **Measure before machinery.** `σ_e` under CRN first; every mechanism must move the outcome by `≥ 3 σ_e` against its best conventional replacement at matched compute, or it is removed and reported. | ANLAGE A7, kept |
| A8 | **Heritable is what does not copy.** The genome carries what must span sizes, modalities, runs and machines without loss and what is cheaper to acquire once than to re-learn: gains, offsets, switches, learning rates, the modality mixture. What copies — the weights — is inherited by persisting. Operationally this is the requirement that the genome's per-episode move be small beside the fast system's, measured in the one space where both act (01 (1.6), recorded as `C2` every episode). | ANLAGE §11.5 A8, kept, with its measurement named |
| A9 | **The model is the agent.** There is one code path for the life; deployment is a life with `η_1` possibly zero; every modality is a stream; the agent perceives its own emissions. | new (B1, B2) |

## 0.5 What is reused from ANLAGE and the earlier lines, and why

| component | path | reuse |
|---|---|---|
| gated delta rule, chunkwise, verified to 1e-10 against the float64 oracle; streams state across calls | `gnomon/pheno/mixer_chunk.py::gated_delta_rule_chunk`, `anlage/ops.py::delta_mixer(state=)` | the τ0 memory (03 §3.4) |
| attention with GQA, gain-free QK-norm, RoPE, output gate; SwiGLU; RMS norm; softcap; chunked CE | `anlage/ops.py`, `gnomon/pheno/rmsnorm.py` | the backbone operators (03) |
| Muon (Newton–Schulz 5), AdamW groups, the warm-up/constant/decay-to-zero schedule, batch warm-up, micro-batch rule | `gnomon/optim/muon.py`, `anlage/recipe.py` | the τ1 optimiser and schedule (08) |
| window pools under CRN, classified pools with systematic rounding, evaluation sets v3, strata | `screen/crn.py`, `anlage/data.py` | the text stream and evaluation (02, 10) |
| copy-growth with HyperCloning split and symmetry-breaking noise; the growth ramp | `anlage/grow.py::grow(mode="copy")`, `Ramp` | births (05) |
| paired contrasts, BCa and `t` intervals, TOST, power | `screen/variance.py`, `gnomon/utils/stats.py`, `scripts/anlage_analyse.py::paired` | the protocol (10) |
| verifier tasks with exact rewards and the learnability-gated curriculum | `soma/openended/tasks.py`, `soma/openended/archive.py`, `soma/eval/regression.py` | the symbolic stream (02 §2.9.2) |
| mirrored ES over a compact object with an injected evaluator | `soma/germline/es.py::es_step` | a registered alternative estimator for the consolidation rule (06 §6.6) |
| seeds, shapes, reports | `gnomon/utils/{rng,shapes,report}.py` | everywhere |

Not carried over: the neural-kernel weight decoder (`anlage/genome.py` matrix roles), the
latent bank as a carrier of weights, the population and the merge, the programme's mixture
over data classes as a *separate* object (it becomes a genome role), the token embedding table
(TEMPO has no vocabulary in the backbone). They stay on their branches as the record.

## 0.6 The process rule

**[NORMATIVE]** No mechanism is built before the measurement that could falsify it is
scheduled, and no mechanism is kept after its kill criterion fires. The order of work is the
order of gates in 10 §10.3; the specification is written before the code; every acceptance row
in 01–09 is owned by a test or by an artifact check (`scripts/tempo_acceptance.py`); every
normative section is owned by code (`scripts/tempo_spec_trace.py`); every number in the
results register is re-measured from an artifact on disk.
