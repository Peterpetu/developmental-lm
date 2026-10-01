# 10 — Kill criteria and the protocol

**Status: normative.** Measurement rules (§10.1), the gates and kill criteria (§10.2), the
runs R0–R10 with budgets, seeds and artifacts (§10.3), the results register and manifests
(§10.4), and the one verdict rule (§10.5). Implementation: `scripts/tempo_protocol.py` (the
chain), `scripts/tempo_analyse.py` (every verdict), `scripts/tempo_register.py` (the register).

## 10.1 Measurement rules

**[NORMATIVE]**

* **Two fixed sets per stream.** The **evaluation set** reports a run's score; the disjoint
  **fitness set** feeds the reward of 06 §6.3, so that no criterion is read off the same
  windows the consolidation optimised. Text: TinyStories v3
  (`anlage/data.py::EvalSet.sample_documents`, `EVAL_VERSION = 3`), the union of both
  validation shards by document index mod 3 — residue 2 is the evaluation (test) set, residue 0
  the fitness set — as windows of `T + K` tokens scored on targets `K … T+K−1`, per-token NLL
  in nats, from a fresh state (08 §8.3). Every generated stream has its own sets from reserved
  seeds, `10^6 + 1000 i + j` (evaluation) and `10^6 + 1000 i + 500 + j` (fitness) for stream
  `i`. Reward-bearing streams are scored on both sets by **greedy rollouts** (mean actions,
  greedy decoding) with the stream's exact reward, not by the likelihood of their own samples.
* **Noise, per criterion.** Every bar is `3 σ` where `σ` is the standard deviation, over the
  **control arm's seeds**, of **the criterion's own metric** — not a single global `σ_e`. A
  criterion on the final text NLL uses the control arm's spread in final text NLL; a criterion
  on a per-stream reward uses that stream's spread in that reward. `σ_e` (R1's spread in final
  text NLL at the primordium) remains the reported noise floor of the text configuration and
  the bar of K-CONT, which is a contrast of two text scores. A criterion whose control arm has
  fewer than two seeds is NOT EVALUATED (G-NOISE).
* **Contrasts.** Paired by seed under CRN: arms of a contrast share the **order seed**, so
  they read the same windows in the same order, and differ in the model seed only as the arms'
  own initialisations require (08 §8.1). Reported as the paired mean difference with its
  `t`-interval at `α = 0.05` and `n`, always beside the bar. `n ≥ 4` where a contrast is a
  claim.
* **Compute matching.** Lives are compared **at equal cumulative FLOPs**: each record carries
  the fresh-state score at the end of every episode against `F_cum` (probes, rollouts and
  criticality probes included, 09 §9.1), and a contrast is read at the smallest final budget
  among the arms compared, by linear interpolation of the two curves. An arm that spends more
  compute per episode therefore gets fewer episodes in the comparison, not a free advantage.
  Only K-SOTA names wall-clock and only R1's table names tokens.
* **One chain, pausable.** Every GPU run on this card is one serial chain (00 §0.6); smoke
  tests run on CPU; the wall-clock of a run is recorded together with its step-time outliers,
  since the card is shared with the user's other use. Because it is shared, the chain can be
  stopped on request: `python scripts/tempo_protocol.py --pause` writes `runs/tempo/PAUSE`, the
  running life stops at its next step and checkpoints (08 §8.5), and the chain exits without
  marking anything complete; `--resume` clears the file and the chain continues, the paused run
  picking up inside the episode it was in. A paused record says `"paused": true` and
  `"complete": false`, so a pause can never be read as a finished run (§10.5).
* **No missing run is a pass.** A criterion whose run has not been executed, or whose record
  does not say `"complete": true`, is NOT EVALUATED, never passed.

## 10.2 Gates and kill criteria

**[NORMATIVE]** Gates (G-) are bugs or preconditions; kills (K-) are the design's claims.
Every bar below is `3 σ` in the sense of §10.1 and every verdict follows §10.5.

| id | fires when | consequence |
|---|---|---|
| G-NOISE | the control arm's `σ` for a criterion's metric is unmeasured (fewer than two complete seeds) | that criterion is NOT EVALUATED |
| G-STAT | the two-sided transient test of 01 (1.8) fails at a boundary | `N` doubled once per life, then flagged; recorded either way |
| G-IDENT | any row of 09 §9.6 fails | a bug; nothing runs until fixed |
| G-SPECNUM | any parameter count of the specification differs from the module tree | the specification is corrected to the tree and the correction recorded |
| G-REGRESS | the gain regression residual `R_e > 0.01` at any boundary (04 §4.5) | `r` doubled, the run restarted; recorded |
| G-NOFREE-INVERSE | any expression of `g`, regression, consolidation, JVP or SVD runs inside the per-step path (sentinels of 04 E11, 06 C12, 07 K6) | a bug |
| G-ACT | on the point-mass, the shared output through `D_act` with 02 (2.12) does not reach the mean return of a dedicated 2-layer MLP policy (`8 → 64 → 64 → 2`) trained with the same estimator, the same number of rollouts and the same horizon, by the bar of the MLP's own seed spread (3 seeds), on the reserved evaluation starts | the shared output space is not carrying actions; the interface is re-examined (FAST-style discrete actions, 11 §P-A.9) before any multimodal claim |
| **K-CONT** | the span agent has worse per-token test NLL than the token backbone at **equal training FLOPs** by the bar, at 25M span-agent tokens | `K = 2`, then `K = 1` (the token head of 02 §2.2); reported |
| **K-GENE** | TEMPO does not beat each of the arms (a) `γ ≡ 0`, (b) homeostatic only, (c) random `Δg` of norm `τ`, (d) free gains (04 §4.9) on the **test score at equal compute** over a 20-episode life | the RL rule is decorative; the genome reduces to the best arm; reported with (e) SVF, (f) ES and (g) GRPO beside it |
| **K-HORIZON** | TEMPO's per-episode rewards `r_e` are positive on average **and** its test score at equal compute is worse than arm (a)'s by the bar, in the sense of §10.5 | the reward window is lengthened to two episodes and R4 repeated once; if it fires again, K-GENE fires. Both conditions are required: a rule that is merely worse has not shown the short-horizon *bias*, which is winning per-episode rewards while losing the outcome. The reward series' two halves are recorded, so a rule that wins early and loses late is visible |
| **K-REV** | the reverse read does not beat the forward read with the same `ψ`, estimator, read-out rule (06 §6.2) and cost | the third axis is a direction, not a mechanism; reported |
| **K-SAL** | salience-ranked rehearsal does not beat uniform rehearsal of the same size | the buffer is drawn uniformly; the head stays a diagnostic |
| **K-MORPH** | RL-timed growth from the primordium does not beat (a) the fixed schedule, (b) the one-shot doubling by hypercloning, or (c) the flat adult, each at the **same FLOP budget** (05 §5.7) | growth stays scheduled or off; reported |
| **K-CRIT** | conditional on R3's trigger (07 §7.6): arm (b) (the `ζ` rule alone) does not beat arm (a) (`κ = 0`) | the rule removed, `ζ ≡ 0`. If R3 does not trigger: NOT NEEDED, with the profile |
| **K-TRANSFER** | after the shift to a held-out verifier family, the agent with its genome does not adapt faster than the same weights with the genome reset (08 §8.4) at 10 % of an episode, measured by the task stream's greedy reward | B5's within-modality claim fails; reported |
| **K-XFER** | the genome written by text episodes does not speed the learning of at least two of {control, vision, speech, patterns} against the reset at 10 % of an episode, each against the **reset arm's own** spread on that same early score | the genome does not translate between modalities; reported. A stream that was not run is NOT EVALUATED and is not counted as a lost pair; if fewer than four are evaluated the criterion is NOT EVALUATED |
| **K-BLEND** | the blended agent's score is **worse** than a stream-specific agent's on more than one stream, each at the same per-stream compute | interference dominates; the blend is narrowed and the reason recorded. Without the stream-specific arms the criterion is NOT EVALUATED, never passed |
| **K-SOTA** | the full agent is worse than the FLOP-matched token baseline trained on the same total training FLOPs at the adult budget | not SOTA-capable at this scale; reported with the gap. (ANLAGE's 1.5693 is a *different* budget and shape and is quoted as context, never as this criterion's bar) |

## 10.3 The runs

**[NORMATIVE]** Every run is one script writing `runs/tempo/<run>-<arm>[-<tag>]-<config>-s<seed>/`
with a manifest (§10.4) and a JSON of every recorded number; `scripts/tempo_analyse.py` prints
every verdict from those JSONs and nothing else. `N` is the screens' 250 steps at `B = 32`
(4.1M tokens per episode) unless a row says otherwise; the production `N` is 1000.

| run | what | configuration and budget | seeds | artifacts and verdicts |
|---|---|---|---|---|
| **R0 SPAN** | train the codec (02 (2.5)) on TinyStories; freeze the encoder; measure the ceiling | codec `d_c = 256`, 200M tokens (50M gave a ceiling of `0.0126`, above I1; the artifact is kept as `codec-ts-s0-tokens50M`), AdamW `3·10^{-4}` with cosine decay, batch 4096 spans | 1 per `K` | `codec.pt`, ceiling NLL (I1 `< 0.01`), noise tolerance (I2); decides whether `K = 4` proceeds |
| **R1 NOISE** | the span agent at the primordium, `γ ≡ 0`, one episode of 25M tokens; `σ_e`; throughput and memory; the latent-term ablation | TS-primordium, `N = 1525`, `B = 32`; arms `nolat` (`λ_lat = 0`, the registered loss) and `lat01` (`λ_lat = 0.1`) | 4 each | `σ_e`, mean test NLL, spans/s, tokens/s, peak memory (P12, N6); the ablation's contrast |
| **R2 CONT** | K-CONT at equal training FLOPs; then the fallbacks `K = 2` and `K = 1` (each with its own codec, R1 and baseline) | the span agent at the primordium against ANLAGE's `Phenotype` with value embeddings, its width chosen FLOP-nearest and its token budget scaled to equal training FLOPs | 4 each | the contrast with its interval; the `K` decision |
| **R3 STAT/CRIT** | a 20-episode life with `γ ≡ 0` and the rule off: G-STAT per episode; the `λ_l^{rel}`, `Λ` profile; the A8 ratio | primordium, `N = 250`, `B = 32` | 2 | the profiles (K7); whether K-CRIT is triggered; the `N` the screens use |
| **R4 GENE** | K-GENE, K-HORIZON, K-CRIT's arms: TEMPO with estimator 2 against (a)–(d), plus (e) SVF, (f) ES, (g) GRPO | primordium, 20 episodes at R3's `N`; compared at equal cumulative FLOPs | 4 for TEMPO and (a)–(d); 2 for (e)–(g) | the test score curve against `F_cum` per arm; the reward series and its halves; `‖g − g_0‖`, `R_e`, the probes' rewards including the null |
| **R5 REV/SAL** | K-REV and K-SAL against R4's TEMPO arm at the same order seeds | as R4's TEMPO arm with the read order flipped, and with uniform rehearsal | 4 each | the two contrasts |
| **R6 MORPH** | K-MORPH: TEMPO's growth against (a) fixed schedule, (b) one-shot doubling, (c) flat adult | every arm runs until `F_cum` reaches `F_budget = 40` primordium-episodes, so the arms differ in episodes and not in compute | 4 each | the test score at the budget; the switch trajectories; active parameters over the life |
| **R7 TRANSFER** | K-TRANSFER: from R4's life, one episode on text + the verifier tasks with a held-out family | the agent's `g` against the genome reset (08 §8.4); the first evaluation, a tenth of the way in | 4 each | the contrast on the task stream's greedy reward |
| **R8 BLEND** | the seven streams under the mixture role; K-BLEND; the per-stream noise for K-XFER; G-ACT | primordium, 20 episodes blended; seven stream-specific arms `only-<stream>` at the uniform per-stream share of the same budget | 4 blend, 4 per specific arm | per-stream scores and returns; the mixture trajectory |
| **G-ACT** | the MLP policy on the point-mass at the same rollouts and horizon, against the agent of R8 on the reserved starts | `8 → 64 → 64 → 2`, the same group-advantage estimator | 3 | the returns and the gap against the MLP's own spread |
| **R9 XFER** | K-XFER: `g` from the text life against the reset, on the blend | from R4's text life; one episode of the blend, the first evaluation | 4 each | four per-stream contrasts |
| **R10 SOTA** | K-SOTA at the adult budget against the FLOP-matched token baseline at the same total training FLOPs | the full agent (all mechanisms surviving R2–R9), estimator 1, 60 episodes | 2 each | test NLL and the gap |
| **Pile lineage** | R1, R2, R4, R7, R10 repeated at the Pile-adult shape once `data/pile_*` exists | Pile-primordium; budgets scaled by the throughput table | as above | the same register, a second table |

Order: R0 → R1 → R2 → R3 → R4 → R5 → R6 → R7 → R8 → R9 → R10 → Pile. Each run's verdict is
read before the next mechanism's code is trusted; a mechanism killed upstream runs downstream
only as its best surviving arm.

## 10.4 Manifests and the register

**[NORMATIVE]** Every run directory holds `manifest.json`: the script's arguments, the
configuration, the model and order seeds, the evaluation- and fitness-set versions with their
digests, the codec's SHA-256, the code digest of `tempo/*.py`, the digest of `scripts/tempo_*.py`,
`torch.__version__`, the device, the wall-clock, the step-time outliers (steps slower than
three times the median, since the card is shared), and the episodes actually completed.
`docs/tempo/RESULTS.md` is the register, rendered by `scripts/tempo_register.py` and never
edited by hand: one row per number with its run directory, seed, budget and the criterion it
serves; a verdict line per criterion; negative results kept with their artifacts.

## 10.5 The verdict rule

**[NORMATIVE]** One rule, applied by `scripts/tempo_analyse.py` to every criterion. Let `m` be
the paired mean of (mechanism − control) on the criterion's metric, `[lo, hi]` its `t`-interval
at `α = 0.05`, and `bar = 3 σ` with `σ` the control arm's spread on that metric (§10.1). Lower
is better for every metric (a reward enters as its negative).

For a **"not better" criterion** (it fires unless the mechanism beats the control):

```
   DID NOT FIRE   iff   m ≤ −bar   and   hi < 0            the mechanism is better by the bar, significantly
   FIRED          iff   m ≥ 0                              the mechanism is not better at all
   INCONCLUSIVE   otherwise                                 better, but not by the bar or not significantly
```

For a **"worse" criterion** (it fires when the mechanism is worse than the control by the bar,
e.g. K-CONT, K-BLEND, K-SOTA):

```
   FIRED          iff   m ≥ bar   and   lo > 0
   DID NOT FIRE   iff   hi < bar
   INCONCLUSIVE   otherwise
```

A criterion with fewer than two paired seeds, or with a missing or incomplete control or
mechanism arm, or whose bar is unmeasured, is **NOT EVALUATED** — never INCONCLUSIVE, which
says "run and unclear" about a contrast that was not run, and never a pass.

**One implementation.** Every criterion goes through one function that applies this rule; the
per-criterion prose of §10.2 says what the arms and the metric are, and nothing else. The
second adversarial round found this layer deciding criteria by hand and disagreeing with
itself: K-HORIZON fired on a positive point estimate with no bar and no interval; a missing
arm read as "did not fire"; incomplete runs counted as complete; a stream whose greedy reward
failed to record had its *loss in nats* substituted for that reward, and the change of units
was reported as a ten-nat regression; K-XFER counted a stream that was never run as a lost
pair (11 §P-C.37). `tests_tempo/unit/test_analyse.py` now feeds the rule synthetic records
whose right answer is known, including every boundary case of both forms. An INCONCLUSIVE criterion at `n = 4` is re-run at `n = 8`
once before it is recorded as inconclusive (the `2.3 σ` case of ANLAGE's K-INIT is the
precedent); it is never read as a pass. A mechanism survives only on DID NOT FIRE. The final
verdict of the local protocol is the list of surviving mechanisms with their measured margins,
and the statement of B11 as a path is repeated with those margins in place of claims.
