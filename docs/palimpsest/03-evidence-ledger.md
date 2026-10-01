# 03 — Evidence ledger

**Status: record.** Every finding, correction and reading, in the order obtained, with its
artifact. Entries `P-P.n`. Verdicts are those of `02-protocol.md`; the register
`RESULTS.md` is rendered by `scripts/palimpsest_analyse.py --markdown`.

0. **The record's unrecorded artifact: on a family the lineage never saw, the copied free
   network is a 1.3-nat hindrance and the genome-regenerated one is neutral.** *2026-09-10,
   read on this day; `runs/bottleneck/xfer-heldout-from-meta-GN30-E-tinystories-s0-seed0/transfer.json`
   and `.../xfer-heldout-from-meta-free-tinystories-s0-seed0/transfer.json`, both complete
   (`git` and `time_utc` in the files), written after TEMPO `§P-D.57` and never read into it.*
   Target: the held-out task family (GRID, and length 16) of the meta routine, 5M tokens per
   arm, three random seeds first, contrasts against random at 25 / 50 / 100 %:

   | source | arm | 25 % | 50 % | 100 % | at birth |
   |---|---|---|---|---|---|
   | meta-free (the free sequential network, final state) | copy-all | +1.695 | +1.434 | **+1.328** | −2.01 |
   | | copy-low | +8.38 | +7.02 | +1.71 | +0.54 |
   | | copy-high | +0.019 | +0.010 | +0.003 | +0.18 |
   | meta-GN30-E (generation 97, the best decode) | decode-all | +0.105 | +0.008 | **+0.007** | −0.94 |
   | | copy-all (its constrained phenotype) | +1.001 | +0.040 | +0.006 | −0.08 |
   | | decode-low | +1.299 | +1.262 | +0.169 | +0.60 |
   | | decode-high | +0.010 | +0.003 | +0.001 | −0.12 |

   *Reading.* Nothing beats a random start on a family the source never saw (the paper's
   MNIST → F-MNIST outcome). But the two whole-network arms separate by 1.3 nats at the end
   of the budget: the free sequential network's weights carry damage that 5M tokens do not
   undo, and the decode of the genome carries none. The genome's constrained phenotype
   (`copy-all` from meta-GN30-E) also recovers, to +0.006 — because it was regenerated from
   the genome 24 tasks before the end and trained on top, i.e. it too had been through the
   sieve. This is the cleanest measurement in the repository of what `00 §0.3.3` calls Exit 2,
   and it is recorded here because it is the observation the sixth model is built on. Also
   recorded as TEMPO `§P-D.58`.

1. **The sixth model is specified and its instruments are built and tested before any life
   runs.** *2026-09-10, 16:40–19:30.* `docs/palimpsest/00–02`; `palimpsest/{envs,expand,distil,
   life,analyse}.py`; `scripts/palimpsest_{life,analyse,build_code}.py`; `tests_palimpsest/`
   (31 tests: counts, Theorem 1 for every core in both new-unit modes, the KL closed form,
   dreams, distillation recovery, births, the byte rule, EWC, the life loop for every arm on
   the CPU with its reproducibility, the analyser on planted records). Two corrections found
   by the tests: (a) the cores' parameter counts are `5,388,417 / 3,218,561 / 2,136,385 /
   1,330,369` (ratios `2.41 / 4.04 / 6.09 / 9.78`), not the hand estimate in the first draft
   of `01 §1.3` — the value embeddings and the table do not shrink with the width; the
   table was rewritten from the test; (b) a core too shallow to hold an attention layer
   cannot copy one at depth expansion; the new layer takes the free initialisation (recorded
   in `expand.py`). The second environment: `data/py_8192`, 10,552 Python files, 79.1M train
   tokens, 3,000 held out (`runs/palimpsest/build_code.log`).

2. **Two defects found by the acceptance tests before any life ran, and one instrument.**
   *2026-09-10, 19:30–20:10.* (a) In the `fresh` new-unit mode of the expansion the kept
   units' output columns (`O`, `DOWN`) were still divided by the tiling factor while their
   duplicates' columns were zeroed, so every branch output was scaled by `1/k` — and the
   gain-free RMS norm that follows every branch **hid the scale**: the logits moved by only
   `0.009` at `k = 4` while the branch output was off by a factor of four (layer 0: `2.11`
   against `1.9·10^{-6}` in copy mode). T2's tolerance caught the residue; the trace found the
   cause. The kept columns now carry the undivided weights. The lesson is recorded because it
   is general to this backbone: under Peri-LN a per-branch scale error is invisible to any
   test that reads only the logits. (b) Dreams were reproducible only at equal batch size (the
   categorical draw consumed the generator in batch order); the sampler now draws one uniform
   per (sequence, position) up front and samples by inverse CDF in float64, so a dream is a
   function of the seed alone. (c) Dreaming without a cache cost `47 s` for `32k` tokens on the
   card (`smoke-sieve-c8`); `palimpsest/sample.py::Sampler` steps the phenotype one token at a
   time by the exact delta recurrence with a carried state, a three-token convolution history
   and a key/value cache with RoPE at the absolute position, and equals the full forward to
   `2·10^{-4}` on the CPU (T4b; the two dream paths give identical samples on one seed). The
   first wave was queued before the cache existed and runs at `T_dream = 128`, `N_δ = 250k`;
   every step reads the code at its start, so the sieve steps from the second onward use the
   cached sampler, which changes cost and nothing else.

3. **R1, the copy on the shift life (seed 0): the shift is severe. Eight million tokens of
   code take stories from 2.09 to 7.41 nats (the null is 9.01); eight million of stories take
   code from 1.48 to 7.31. The copy relearns each faster than from scratch, and forgets it
   again.** *2026-09-10, 11:41. `runs/palimpsest/shift-copy-s0/record.json` (464 s; 72k tok/s;
   8.4 GB).*

   | phase | env | at birth | 10 % | 25 % | 50 % | 100 % | the other env at 100 % |
   |---|---|---|---|---|---|---|---|
   | 1 | stories | 10.3 | 3.191 | 2.704 | 2.400 | **2.093** | code 8.94 |
   | 2 | code | 8.94 | 2.556 | 2.069 | 1.769 | **1.477** | stories 7.41 |
   | 3 | stories | 7.41 | 2.698 | 2.398 | 2.230 | **2.006** | code 7.31 |
   | 4 | code | 7.31 | 2.025 | 1.723 | 1.591 | **1.398** | stories 6.74 |

   Scores: `U_4 = 4.070` (stories 6.74, code 1.40), `Λ = 3.815`; retention `R_2(stories) =
   +5.32`, `R_3(code) = +5.83`, `R_4(stories) = +4.74`. Two readings. *The copy carries
   something:* stories at 10 % of phase 3 (2.70) is below phase 1's 3.19, and phase 3 ends
   0.09 below phase 1, so relearning is faster than learning — the damage is not total. *And
   it carries damage:* nothing of the previous environment survives the next at the level
   the union score reads (both environments end four to five nats above their best). Code is
   the easier environment for this tokenizer (1.40 at 8M against stories' 2.09; the source is
   repetitive at 0.51 tokens per character). The first-wave bar for any arm on this life is
   set by this record: an arm that keeps both environments near their best would score
   `U ≈ 1.75`, the copy scores `4.07`, and the difference is the room the sieve has.

4. **A pre-registered expectation, written before the rehearsal arms ran: on a two-environment
   alternation, pseudo-rehearsal from the uncompressed phenotype should do as well as the
   sieve's rehearsal; accumulation can only show where the life holds more environments
   than the copy can hold at once.** *2026-09-10, 12:05.* At the switch `k → k+1` the copy
   `W_k` has just been trained on `ε_k` and has forgotten `ε_{k−1}` (P-P.3: to 7.3–7.4), so its
   own samples are of `ε_k` alone; on `stories → code → stories → code` that is exactly the
   environment the next phase needs to keep, and `copy-dream` should score with `sieve-life`.
   The core accumulates (P4): `C_2` is distilled from `W_2` on code *and* from `C_1` on `C_1`'s
   dreams of stories, so its samples are of both. On `stories → code → tasks` the copy's
   rehearsal at the last switch carries code only and stories are gone; the core's carries
   both. The three-environment life (`shift3`) is therefore queued in the first wave, with the
   reading **K-DREAM-LIFE on `shift3`** as the one that decides whether accumulation through
   the compressed store is what pays: `sieve-life − copy-dream < −3σ` on `shift3` and not on
   `shift`. If the sieve loses on both, generative rehearsal is the mechanism and compression
   is not; if it wins on both, the compression is doing something beyond accumulation and
   the reason is to be found (the core's samples may be *more general* than the phenotype's,
   the paper's simplicity bias — to be measured by their entropy and by a classifier).

5. **R2's first point, `sieve-c8` on the shift life (seed 0, EOS-start dreams): the reborn
   phenotype forgets like the copy (union 4.36 against 4.07), but the core — six times
   smaller — scores 2.35 on the union, and the newborn knows both environments (2.86 / 2.10)
   where the copy at the same switch knows one (7.41 / 1.48). And a defect of the dream
   definition, found in the core's curve.** *2026-09-10, 11:57.
   `runs/palimpsest/shift-sieve-c8-eos-s0/record.json` (993 s; overhead 33 % of the phenotype's
   training FLOPs: 244 distillation steps of 4M tokens per consolidation at 97–149 s, the
   cached sampler's 250k dream tokens in 13 s).*

   | after phase | env trained | phenotype (stories / code) | core (stories / code) | newborn at the next birth |
   |---|---|---|---|---|
   | 1 | stories | 2.093 / 8.88 | **2.591** / 8.81 | 2.591 / 8.81 (= the core, Theorem 1 on the card) |
   | 2 | code | 7.687 / 1.506 | 2.856 / **2.096** | 2.856 / 2.096 |
   | 3 | stories | 2.041 / 8.27 | 2.388 / 5.827 | 2.388 / 5.827 |
   | 4 | code | 7.258 / 1.466 | 2.882 / 1.827 | — |

   Distillation: `D_data = 0.47, 0.62, 0.35, 0.39` nats (the core's KL to the phenotype on the
   current environment — the price of ratio 6 is about half a nat on stories at 4M
   distillation tokens); `D_dream = 0.14, 0.08, 0.06` (the new core matches the old one on the
   old one's samples).

   *Readings.* (1) **K-GENOME does not fire, by 1.7 nats at one seed**: the genome as a model
   (`G_4 = 2.354`) beats the copied phenotype on the union (`U_4 = 4.070`) — the compressed
   store carries the life better than the full network that lived it. (2) **The rebirth alone
   does not protect the phenotype**: born at 2.59 on stories, it is at 7.82 after 10 % of the
   code phase (the copy, born at 2.09, is at 6.31), so the expanded network forgets *faster*
   than the copy — the inherited function lives in tiled units that the new environment
   overwrites, and the fresh capacity is fully plastic. The copy also relearns faster (2.70 at
   10 % of phase 3 from 7.41) than the newborn improves (2.74 from 2.86): latent knowledge
   recovers quicker than explicit inherited knowledge is refined. Rehearsal in the lifetime
   (`sieve-life`) is what the phenotype's union needs; the frontier at `U_K` may well be
   monotone at the copy while `G_K` is not. (3) **The dream defect.** The core after phase 3
   has lost code (2.10 → 5.83) although its dream KL to the previous core was 0.08: the dreams
   were all stories. Definition 6 started every dream at `EOS`; a Python file averages 7,500
   tokens, so a 512-token window of code almost never contains a document start, the core's
   `P(· | EOS)` is learned from stories alone, and its dreams of "the past" contained none of
   the code it knew. Fixed: the start state is `mixed` (half `EOS`, half a uniformly random
   token) by default, the EOS start kept as the control (`01 §1.5`); this run is kept as
   `shift-sieve-c8-eos-s0` and the mixed-start run re-queued as `shift-sieve-c8-s0` before the
   rest of the frontier, so that every sieve arm of the wave shares one dream definition.

6. **R1 on the stationary double, the copy (seed 0): 2.093 → 1.951 → 1.896 → 1.863 over four
   8M-token phases of stories on four different pools; each phase's re-warmed schedule costs
   0.15–0.19 nats at its 10 % mark before the phase improves on the last.** *2026-09-10, 12:05.
   `runs/palimpsest/stat-copy-s0/record.json` (446 s).* `U_4 = 1.863` at 32M tokens, beside
   ANLAGE R1's 1.857 at 25M in one uninterrupted schedule: the per-phase restart is worth about
   the same as seven million tokens, and every arm of the stationary life pays it alike. This is
   the number K-STAT reads the sieve arms against: Proposition 1 predicts none below it.

7. **The mixed dream start repairs the core's memory: with half the dreams begun at a random
   token, the core keeps code through a stories consolidation (2.13 → 2.86 instead of 2.10 →
   5.83) at a cost of 0.05 on stories, and the genome's union score improves to 2.26; the
   reborn phenotype is unchanged in its forgetting (union 4.49).** *2026-09-10, 12:21.
   `runs/palimpsest/shift-sieve-c8-s0/record.json` (957 s; overhead 26 %).*

   | after phase | core, EOS start (P-P.5) | core, mixed start | newborn (mixed) |
   |---|---|---|---|
   | 1 stories | 2.591 / 8.81 | 2.589 / 9.09 | 2.589 / 9.10 |
   | 2 code | 2.856 / 2.096 | 2.795 / 2.134 | 2.795 / 2.134 |
   | 3 stories | 2.388 / 5.827 | 2.436 / **2.858** | 2.436 / 2.857 |
   | 4 code | 2.882 / 1.827 | 2.697 / 1.826 | — |

   `D_dream = 0.15, 0.11, 0.08`; `G_4 = 2.262` (copy `U_4 = 4.070`; the EOS core 2.354). The
   phenotype's own numbers are the copy's story again: born knowing both (2.44 / 2.86 at the
   last switch), it ends the code phase at 7.52 on stories. Two facts for the next readings:
   the core's cost of ratio 6 is about 0.5 nats on the environment just consolidated (2.59
   against 2.09; 2.13 against 1.50) and its drift over two further phases another 0.1–0.3;
   and dreams at one eighth of the distillation volume hold an environment to within 0.7 of
   its consolidated value across one phase in which the phenotype forgot it entirely.

8. **K-STAT reads MONOTONE at one seed: on the stationary life the sieve is worse than the
   copy at every phase and at the end (2.015 against 1.863), and its core is worse still
   (2.26), exactly as Proposition 1 predicts when there is nothing harmful to discard.**
   *2026-09-10, 12:37. `runs/palimpsest/stat-sieve-c8-s0/record.json` (952 s).*

   | phase | copy at 100 % | sieve: born | at 10 % | at 100 % | core after |
   |---|---|---|---|---|---|
   | 1 | 2.093 | — | 3.191 | 2.092 | 2.577 |
   | 2 | 1.951 | 2.578 | 2.757 | 2.038 | 2.412 |
   | 3 | 1.896 | 2.412 | 2.659 | 2.028 | 2.317 |
   | 4 | 1.863 | 2.318 | 2.620 | 2.015 | 2.260 |

   Each rebirth costs the phenotype about 0.5 nats (the core's ratio-6 price), of which the
   phase recovers most but not all: the sieve ends 0.15 behind the copy at equal tokens,
   with 26 % more compute. The core improves by 0.17, 0.10, 0.06 per consolidation and its
   KL to the phenotype falls 0.45 → 0.27 (the same environment, refit each time) — the
   compressed store accumulates on a stationary stream too, but slower than the phenotype
   learns, and never catches it. This is the theorem's half of the double dissociation
   (`02 §2.2`): under no shift, the copy is the argmin; whether the shift life gives an
   interior argmin on the phenotype's union (K-SIEVE) or only on the genome's (K-GENOME) is
   what the frontier and the rehearsal arms decide.

9. **`scratch` on the shift life (seed 0): union 4.737; the frontier's three points at one seed
   are copy 4.070 < sieve-c8 4.488 < scratch 4.737 on the phenotype's union — the rebirth
   carries something (0.25 better than a fresh start on the union, 0.8 nats at the 10 % marks)
   and less than the copy's latent knowledge (0.4 worse). K-SIEVE on `U_K` fires at this seed:
   the argmin is the copy.** *2026-09-10, 12:44. `runs/palimpsest/shift-scratch-s0/record.json`
   (456 s).* Scratch relearns each environment to the same level every time (2.089 / 2.097 on
   stories, 1.565 / 1.566 on code); its 10 % marks (3.19 stories, 2.69 code) against the
   copy's (2.70, 2.03) and the sieve's (2.75, 2.29–2.83) order the three starts by what they
   carry into a phase. The phenotype's union is decided by what the last phase destroys, and
   all three lose the other environment to 7.5–7.9; the sieve's rebirth does not change that.
   The pre-registered reading stands: the mechanism's value, if any, is in the genome
   (K-GENOME: 2.26 against 4.07) and in the rehearsal it makes possible (`sieve-life`, running).

10. **`sieve-life-c8` (seed 0): the phenotype that rehearses the core's dreams keeps both
    environments — union 2.323 against the copy's 4.070, the sieve's 4.488 and scratch's
    4.737; retention +1.1 / +2.3 / +1.1 nats where every other arm lost +5 to +7; 28 %
    overhead.** *2026-09-10, 13:10. `runs/palimpsest/shift-sieve-life-c8-s0/record.json`
    (1,563 s: three rehearsal buffers of 33,381 dreams × 128 tokens = 4.27M tokens each, the
    byte rule at ratio 6, 200 s apiece with the cached sampler).*

    | phase | env | born | 10 % | 50 % | 100 % | the other env at 100 % |
    |---|---|---|---|---|---|---|
    | 1 | stories | 10.3 | 3.191 | 2.400 | 2.093 | code 8.94 |
    | 2 | code | 2.59 / 9.10 | 2.894 | 1.947 | **1.534** | stories **3.159** (copy: 7.41) |
    | 3 | stories | 2.79 / 2.14 | 2.775 | 2.364 | **2.092** | code **3.829** (copy: 7.31) |
    | 4 | code | 2.45 / 2.89 | 2.340 | 1.737 | **1.482** | stories **3.164** (copy: 6.74) |

    Core after each phase: 2.59 / 9.09, 2.79 / 2.14, 2.45 / 2.89, 2.72 / 1.83 (`G_4 = 2.275`).
    *Readings.* The rehearsal costs the current environment little (code 1.534 against the
    copy's 1.477 in phase 2; stories 2.092 against 2.006 in phase 3) and holds the other
    within 1–2 nats of its best instead of losing it to the null. The phenotype's union
    (2.32) is now close to the core's (2.27): the genome and the network it rebuilds agree on
    what the life is. What this does **not** yet say: whether the compression is doing the
    work or the rehearsal alone would — `copy-dream` (the copied phenotype rehearsing its own
    pre-switch samples, no compression) and `replay` (stored data at the same byte size) are
    the two controls, both queued next; P-P.4 pre-registers that on a two-environment
    alternation `copy-dream` should match, and that the three-environment life is where the
    compressed store's accumulation can separate them.

11. **`copy-dream` (seed 0): the copied phenotype rehearsing its own pre-switch samples keeps
    both environments better than the sieve's rehearsal — union 2.125 against `sieve-life`'s
    2.323 — at half the overhead (13 % against 28 %) and with no store beyond the model
    itself. K-DREAM-LIFE against `copy-dream` fires at this seed (+0.20), as P-P.4
    pre-registered for a two-environment alternation.** *2026-09-10, 13:28.
    `runs/palimpsest/shift-copy-dream-s0/record.json` (1,044 s).*

    | phase | env | 10 % | 100 % | the other env at 100 % (copy-dream / sieve-life / copy) |
    |---|---|---|---|---|
    | 2 | code | 2.628 | 1.514 | stories **2.801** / 3.159 / 7.41 |
    | 3 | stories | 2.461 | 2.008 | code **2.493** / 3.829 / 7.31 |
    | 4 | code | 1.838 | 1.415 | stories **2.835** / 3.164 / 6.74 |

    Retention +0.71 / +0.98 / +0.83 (the sieve's +1.07 / +2.30 / +1.07; the copy's +5.3 / +5.8 /
    +4.7). *Reading.* Generative rehearsal is the mechanism that keeps a life, and the
    uncompressed network is the better source of it: its samples are more faithful than a
    six-times-smaller core's (the sieve's phenotype ends 0.33–1.34 behind on the environment
    being kept), and it accumulates too — a phenotype that rehearsed stories during the code
    phase still knows stories at 2.80 when it dreams for the next switch, so its dreams carry
    both environments as the core's do (P-P.4's argument for the three-environment life
    assumed the copy dreams only the last environment; a *rehearsed* copy does not). The
    compressed store's remaining claim is size: a 2.1M-parameter genome that regenerates the
    life within 0.2 nats of the best method here, where `copy-dream` needs nothing but the
    13M model that is already there. What K-DREAM-LIFE now asks of `shift3` is whether three
    environments separate the two rehearsals; and the byte-matched `replay` (stored data)
    decides whether generated rehearsal is even competitive with kept data.

12. **`sieve-c4` (the depth-halved core, ratio 4.0; seed 0): the phenotype's union is no
    better (4.569), the genome's is — `G_4 = 2.033` against `c8`'s 2.262: a larger core
    carries the life better (its KL to the phenotype 0.21–0.32 against `c8`'s 0.39–0.66), and
    the frontier on the genome's score runs toward the uncompressed end.** *2026-09-10, 13:41.
    `runs/palimpsest/shift-sieve-c4-s0/record.json` (778 s; overhead 29 %).* Core after each
    phase: 2.29 / 8.52, 2.50 / 1.78, 2.23 / 2.61, 2.43 / 1.64; newborns at those values. On the
    phenotype's union the order at one seed is now copy 4.07 < sieve-c8 4.49 < sieve-c4 4.57
    < scratch 4.74, K-SIEVE unchanged (fires).

13. **`replay` (stored data at the genome's byte size, 10 % of every batch; seed 0): union
    1.981, the best arm — retention +0.51 / +0.56 / +0.59 — with no overhead. K-REPLAY fires
    against every sieve arm at this seed: the sieve's rebirth is 2.5 nats behind, its
    rehearsal 0.34.** *2026-09-10, 13:48. `runs/palimpsest/shift-replay-s0/record.json`
    (457 s; buffer 4.27M tokens, filled 1.42M per phase, the byte rule at ratio 6).* The
    ranking on the two-environment shift life at one seed:

    | arm | what it keeps | union `U_4` | overhead |
    |---|---|---|---|
    | replay (stored data) | 4.27M tokens of the past | **1.981** | 0 |
    | copy-dream (its own samples) | nothing beyond the model | 2.125 | 13 % |
    | sieve-life (the core's samples) | a 2.1M-parameter core | 2.323 | 28 % |
    | copy | nothing | 4.070 | 0 |
    | sieve-c8 / sieve-c4 (rebirth only) | a core | 4.488 / 4.569 | 26–29 % |
    | scratch | nothing | 4.737 | 0 |

    *Reading.* Kept data beats generated rehearsal by 0.14, and generated rehearsal from the
    full network beats it from the compressed one by 0.20; each step down the table trades
    fidelity of the rehearsed past for what has to be stored — data, nothing, a small core.
    The compression buys storage and costs fidelity; under this criterion it is not
    load-bearing, and rehearsal is. The iid mixture (K-SOTA-C) will bound the whole table
    from below; the three-environment lives and the second seeds follow.

14. **The three-environment life, the copy (seed 0): stories → code → tasks leaves the copy
    at 8.73 / 8.48 / 0.36 — both natural-text environments at the null after 8M tokens of the
    reserved-alphabet tasks — union 5.859.** *2026-09-10, 13:54.
    `runs/palimpsest/shift3-copy-s0/record.json` (359 s).* The tasks are learned to 0.36 (the
    floor of `§P-D.55`) and cost the two languages everything; this is the life on which
    P-P.4's question is read (the rehearsal arms follow).

15. **Three environments, `sieve-life-c8` (seed 0): the phenotype keeps all three — 3.23 /
    3.94 / 0.36, union 2.510 against the copy's 5.859 — and the core holds the whole life
    at 2.85 / 2.93 / 0.37 (`G_3 = 2.047`).** *2026-09-10, 14:13.
    `runs/palimpsest/shift3-sieve-life-c8-s0/record.json` (1,135 s; overhead 27 %).* At the
    last switch the newborn knows stories and code (2.79 / 2.16) and nothing of the tasks
    (12.6); 8M tokens of tasks with the core's dreams at a tenth of every batch leave stories
    at 3.23 (born 2.79) and code at 3.94 (born 2.16) — code, consolidated once and two
    phases old, is the environment that fades most, as in the two-environment life. The
    tasks distil to `D = 0.005`: a 2.1M-parameter core represents the verifier stream as well
    as the phenotype does. Whether the uncompressed network's own samples do the same on
    three environments is `shift3-copy-dream`, running.

16. **Three environments, `copy-dream` (seed 0): the uncompressed network's own samples keep
    all three at least as well — 3.24 / 2.72 / 0.36, union 2.109 against the sieve's 2.510 —
    at 12 % overhead. P-P.4's pre-registered expectation is decided: the rehearsed copy
    accumulates too, and K-DREAM-LIFE against `copy-dream` fires on both lives at this
    seed.** *2026-09-10, 14:26. `runs/palimpsest/shift3-copy-dream-s0/record.json` (748 s).*
    The copy that rehearsed stories during the code phase still knows stories at 2.79 when
    it dreams for the tasks phase, so its buffer carries both languages, and the tasks phase
    leaves them at 3.24 / 2.72 where the core's dreams left them at 3.23 / 3.94. The one
    environment the sieve keeps *as well* is the one just consolidated (stories, 3.23 against
    3.24); the environment two phases old is where the core's lower fidelity shows (code,
    3.94 against 2.72; the core itself holds it at 2.93). *Reading, at one seed on two lives:*
    the mechanism that keeps a life across environments is rehearsal from the model's own
    samples — no data kept, no compression needed; the compressed store's dreams are a
    lower-fidelity source of the same thing. The frontier that remains open is the
    rehearsal's cost in the ratio: `sieve-life` at `c4` and `c2` are queued so that the
    curve `sieve-life(r) → copy-dream` as `r → 1` is measured rather than assumed.

17. **Three environments, `replay` (seed 0): 2.83 / 2.14 / 0.36, union 1.777 — the ranking of
    the two-environment life holds with three: replay 1.78 < copy-dream 2.11 < sieve-life
    2.51 < copy 5.86.** *2026-09-10, 14:32. `runs/palimpsest/shift3-replay-s0/record.json`
    (358 s).* Retention +0.50 / +0.74 / +0.62. The three ways of keeping a life stand 0.33
    and 0.40 nats apart at this seed, in the order of the fidelity of what they rehearse.

18. **Second seeds of the two rehearsal arms on the shift life: `copy-dream` 2.112 (2.125),
    `sieve-life-c8` 2.330 (2.323); paired `sieve-life − copy-dream = +0.208 [+0.084, +0.331]`
    at `n = 2`, the interval excluding zero. K-DREAM-LIFE against `copy-dream` fires with a
    bar.** *2026-09-10, 15:15. `runs/palimpsest/shift-copy-dream-s1`, `shift-sieve-life-c8-s1`
    (1,046 s, 1,558 s).* The seed-to-seed spread of both arms is 0.007–0.013 in the union;
    the copy's and replay's second seeds (running) give `σ_U` for the remaining criteria.

19. **The copy's second seed: 4.089 (4.070); `σ_U = 0.0134`, the bar `3σ_U = 0.040`. With the
    bar, the shift life's verdicts at this point:** *2026-09-10, 15:23.
    `runs/palimpsest/shift-copy-s1` (456 s).*

    | criterion | reading | numbers |
    |---|---|---|
    | K-SIEVE (the frontier on the phenotype's union) | **FIRES** | argmin is the copy: 4.08 < sieve-c8 4.49 < sieve-c4 4.57 < scratch 4.74 |
    | K-STAT | MONOTONE (n = 1) | copy 1.863 < sieve-c8 2.015 |
    | K-GENOME | **did not fire** | the core 2.26 (c8) / 2.03 (c4) against the copied phenotype 4.08 |
    | K-DREAM-LIFE against `sieve-c8` | ahead by 2.17 (n = 1) | rehearsal in the lifetime is what keeps the phenotype |
    | K-DREAM-LIFE against `copy-dream` | **FIRES** (n = 2) | +0.208 [+0.084, +0.331]: the uncompressed source is better |
    | `sieve-life` against `copy` | did not fire | −1.753 [−1.833, −1.672] (n = 2) |
    | `copy-dream` against `copy` | did not fire | −1.960 [−2.164, −1.756] (n = 2) |
    | K-REPLAY | FIRES (n = 1, replay's second seed running) | the sieve's rebirth 2.51 behind kept data; its rehearsal 0.34 behind |

    *The verdict of the model, as `02 §2.2` defines it, is `(K-SIEVE fires, K-STAT monotone)`:
    compression is not load-bearing on the phenotype's union even where the theorem allows
    it, and the theorem's half holds.* What the life measured beside the pre-registered pair:
    the genome — a network a sixth the size — carries the life 1.8 nats better than the
    copied phenotype does, and rehearsal from a model's own samples (compressed or not) is
    the mechanism that keeps a phenotype across environments, at 0.1–0.3 nats from kept data.

20. **Replay's second seed: 1.991 (1.981). Paired `sieve-life − replay = +0.341 [+0.316,
    +0.365]` at `n = 2`; K-REPLAY fires with a bar for the rehearsal arm as for the rebirth.**
    *2026-09-10, 15:30. `runs/palimpsest/shift-replay-s1` (457 s).* The two-seed ranking on the
    shift life is now replay 1.986 < copy-dream 2.119 < sieve-life 2.326 < copy 4.079, every
    gap at least seven times the bar.

21. **The rehearsal's cost in the ratio, at one seed: `sieve-life` at ratio 6.1 → 2.33, 4.0 →
    2.22, 2.4 → 2.19, and the uncompressed source (`copy-dream`, ratio 1) 2.12 — monotone
    toward the full network, as P-P.16 asked. And the ratio-2.4 core, as a model, scores
    1.991 on the union: level with kept-data replay (1.986) with nothing kept.**
    *2026-09-10, 16:33. `runs/palimpsest/shift-sieve-life-c4-s0` (1,331 s, overhead 33 %),
    `shift-sieve-life-c2-s0` (2,442 s, overhead 55 %: the byte rule sizes the rehearsal buffer
    with the core, 10.8M dream tokens per switch at ratio 2.4).*

    | source of the rehearsal | ratio | phenotype union | the genome's union `G_4` |
    |---|---|---|---|
    | copy-dream (the network itself) | 1 | 2.119 (n = 2) | — |
    | sieve-life-c2 | 2.4 | 2.193 | **1.991** |
    | sieve-life-c4 | 4.0 | 2.224 | 2.041 |
    | sieve-life-c8 | 6.1 | 2.326 (n = 2) | 2.269 (n = 2) |
    | replay (kept data) | — | 1.986 (n = 2) | — |

    Cores after each phase at ratio 2.4: 2.22 / 8.61, 2.42 / 1.73, 2.17 / 2.69, 2.39 / 1.59
    (`D_data = 0.15–0.25`). *Readings.* (1) The phenotype's rehearsal is best from the most
    faithful source, and every compression of the source costs it: the sieve is never the
    better rehearsal. (2) The genome itself is a different object: at ratio 2.4 the core
    holds the life at 1.99, level with a phenotype that replayed 4.27M stored tokens, because
    it is balanced (stories 2.39, code 1.59) where every phenotype ends specialised on its
    last environment (replay 2.57 / 1.39). A store that never trains on the current
    environment directly, only on the phenotype's function and its own past, is the best
    *union* model in the table below kept data — which is K-GENOME's finding made sharper:
    the genome is not what makes the phenotype better; it is, at a mild ratio, a better
    lifelong model than the phenotype, at a sixth of the phenotype's competence gap on the
    environment just left.

22. **The rebirth frontier is complete and monotone from the copy to scratch: copy 4.079
    (n = 2) < c2 4.317 < c8 4.488 < c4 4.569 < c16 4.703 < scratch 4.737 on the phenotype's
    union; K-SIEVE fires with the whole frontier. The genome's frontier runs the other way
    and is best at the mildest ratio: c2 1.980 < c4 2.033 < c8 2.262 < c16 2.337.**
    *2026-09-10, 17:02. `runs/palimpsest/shift-sieve-c{2,16}-s0` (970 s at 42 % overhead;
    772 s at 21 %).* Two details. The width-halved core at ratio 6.1 (`c8`) beats the
    depth-halved core at ratio 4.0 (`c4`) on the phenotype's union by 0.08 — six bars — though
    the deeper, narrower core is the better genome (2.03 against 2.26): what a rebirth carries
    into a phase and what a core carries as a model are different things. And every rebirth
    at every ratio forgets the other environment to 7.2–7.9, as the copy does to 6.7: the
    frontier's monotonicity is the theorem's, not a property of the cores.

23. **EWC at `λ_E = 10^3` (seed 0): union 3.358 — the quadratic penalty on the accumulated
    Fisher keeps a third of what the copy loses (stories 5.30 at the end against the copy's
    6.74; retention +4.1 / +3.0 / +3.3) and costs the current environment nothing (code 1.420),
    but it is 1.2–1.4 nats behind every rehearsal arm. K-EWC fires against no sieve arm's
    rebirth (the sieves are 1.0–1.3 worse than EWC on the phenotype's union) and does not
    fire against the sieve's rehearsal (`sieve-life-c8` 2.33).** *2026-09-10, 17:10.
    `runs/palimpsest/shift-ewc-1e3-s0` (497 s; `λ_E = 10^4` queued).*

24. **LwF at `λ_L = 1` (seed 0): union 3.189 — the KL to the frozen previous phenotype on the
    current environment's inputs keeps more of the past than EWC (stories 4.44 against 5.30;
    retention +2.3 / +2.7 / +2.3) and taxes the present hardest of any arm (code 1.941
    against the copy's 1.477): a teacher that does not know the new environment pulls
    against learning it. 25 % overhead (one teacher forward per training token). Behind every
    rehearsal arm by a nat; K-LWF does not fire against the sieve's rehearsal and fires
    against every rebirth.** *2026-09-10, 17:28. `runs/palimpsest/shift-lwf-s0` (1,031 s).*

25. **Shrink-and-perturb (`λ_s = 0.6, p_s = 0.1`; seed 0): union 4.612 — worse than the copy
    (4.079). The shrink toward the initialisation costs the birth (stories 2.09 → 4.12 at
    the switch, code 1.47 → 3.90) and buys no retention (+5.8 / +6.2 / +5.8, the copy's +5.3
    / +5.8 / +4.7): Ash & Adams's repair is for lost plasticity, and plasticity is not what
    this life loses — the copy relearns each environment faster than any other arm, it
    just forgets the other one. K-SP does not fire against any sieve arm (the sieves'
    rebirths are 0.04–0.30 better at one seed), which says only that both are bad.**
    *2026-09-10, 17:35. `runs/palimpsest/shift-shrink_perturb-s0` (458 s).*

26. **Truncation (`trunc-c8`: the adult's primordium block expanded, no distillation; seed
    0): born at 8.5 / 9.1 — the sub-network of a trained adult computes nothing usable, as
    ANLAGE's R8 births showed (10.3, 6.5) — union 4.885, below scratch (4.737). Against the
    same core's sieve (4.488) the distillation is worth 0.40 on the phenotype's union and
    6 nats at birth: K-GROW-C(trunc) does not fire.** *2026-09-10, 17:43.
    `runs/palimpsest/shift-trunc-c8-s0` (457 s).* The block relearns each environment from
    near-null every phase (3.32 at 10 % of phase 3 against the copy's 2.70), so what a sieve
    inherits is what the distillation put into the core, not the block it started from.

27. **The dream ablation (`sieve-nodream-c8`; seed 0): without the previous core's dreams the
    core forgets like the phenotype — 7.67 / 2.17 after the code phase, 2.44 / 7.45 after the
    next stories phase — and its union is 4.798 against 2.262 with dreams. K-DREAM does not
    fire for the genome (2.5 nats); on the phenotype's union the two rebirths differ by 0.08
    (inconclusive at the bar), because the rebirth does not protect the phenotype either
    way.** *2026-09-10, 17:57. `runs/palimpsest/shift-sieve-nodream-c8-s0` (822 s).* Dreaming is
    what makes the compressed store accumulate; distillation from the phenotype alone
    transmits the phenotype's forgetting into the core intact.
