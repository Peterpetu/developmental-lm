# 09 — Protocol, kill criteria, and what may be claimed

Everything in this document is fixed **before** the runs it governs, so that no
criterion can be chosen after seeing a result. Where a number here was measured
rather than chosen, the measurement is given.

## 9.1 The runs

**[NORMATIVE]**

| run | configuration | budget | the claim it tests |
|---|---|---|---|
| `G0-HERIT` | M-pico, 12 genomes x 3 seeds | 4 min | Q1: does the genome carry the function? (**done**, §9.5) |
| `HORIZON` | M-pico, 3 lifetime lengths x 12 x 3 | 20 min | how short a fitness lifetime still ranks genomes (**done**, §9.5) |
| `EVO` | M-pico, 3 run-seeds | §8.3 | **C1**: the generation curve |
| `EVO-RAND` | as `EVO`, `p_random = 1` | 1 seed | `KG-QD`: is MAP-Elites earning its place? |
| `EVO-NOARCH` | as `EVO`, `Gamma_a` frozen | 1 seed | is architecture search earning its place? |
| `FRONTIER` | M-nano, `d_z in {0,2,4,8,16,32,64}` x 3 seeds | 6 h | fixes the operating point (§06.5) (`scripts/gnomon_frontier.py`) |
| `TRANSFER` | pico elite decoded at nano and M1, zero-shot | 2 h | **H8**: the gnomon claim (§02.7) |
| `M1` | 512 x 12, TinyStories | 8 h | held-out CE, bits-per-byte, samples (`scripts/gnomon_m1.py`) |
| `BASE-FREE` | free-weight twin at matched **free-parameter** count | 3 h | the compression claim |
| `BASE-BIG` | free-weight twin at matched **decoded** count | 3 h | the honest upper bound |

**[NORMATIVE — the behavioural suite is inapplicable here, and is named rather
than dropped.]** This table's `M1` row read "behavioural suite" until v1.1,
inherited from SOMA along with §00.4's "verified and reused" list. SOMA's suite
(`soma/eval/regression.py`) grades PARITY, MODCOUNT, PERM, DYCK, MQAR and COPY,
serialised over a reserved symbol block: formal-language tasks **a model was
trained on**. GNOMON trains on TinyStories and nothing else — §05.2 removes
every auxiliary objective on purpose — so it has never seen those tasks and its
accuracy on them is chance by construction. The suite is therefore inapplicable
to this configuration rather than merely unrun, which §09.6 rule 4 requires be
said out loud. What a language model can be held to is reported instead:
held-out CE on the test split, bits-per-byte at a **measured** tokens-per-byte
ratio, and samples at a fixed decoding.

**[NORMATIVE]** Every reported number comes from the **test** split (§05.6),
which selection never sees. The fitness split is used for evolution and for
`h2`, and for nothing that is reported as a result.

## 9.2 Kill criteria

**[NORMATIVE]** Each states a condition, a test, and what happens when it fires.
"Fires" means the architecture or a claim is wrong, not that the run was
unlucky; every one that fires is reported.

| | condition | test | consequence |
|---|---|---|---|
| `KG-HERIT` | `h2 < 0.25`, or its 95% BCa lower bound `< 0.10` | §06.2's design | the constitution does not hold empirically; stop and redesign |
| `KG-FLAT` | no significant downward trend in `best` across generations | Mann-Kendall (exact null) over 3 run-seeds, BH-corrected | evolution is inert; C1 fails |
| `KG-QD` | MAP-Elites indistinguishable from random search | TOST on final `best` and `qd_score`, margin 0.05 nats, 3 vs 3 seeds | C4 is decorative |
| `KG-SCALE` | a transferred genome no better than a random genome at the target scale | paired lifetimes, 3 seeds, BCa on the difference | H8 is false |
| `KG-RANK` | decoded `erank/min(d)` of the archive's elites trends down | Mann-Kendall over generations | the decoder is degenerating |
| `KG-NOFREE` | any free weight tensor exists | §04.6, with its positive control | C1 is violated; a bug, fix it |
| `KG-SHARE` | `share_retrain < 0.50` | §06.3.1's estimator, at every certification point | the genome is not carrying the function; §00's failure mode has returned |

**[NORMATIVE — `G-ACCEPT`, the row-level net.]** `gnomon_spec_trace.py` checks
that every *section* carrying a [NORMATIVE] statement is referenced from the
tree. That net is coarse, and four rows went through it: §04.7's `P7`, §06.7's
`S4` and `S6`, and §08.6's `E3` were each specified as binding, each sat in a
section whose other rows *were* implemented, and none existed — so the section
was mapped and the requirement was not. `scripts/gnomon_acceptance.py` closes
that: every id in an acceptance table must have a `test_<id>_...` in the module
owning that specification file, ids being unique per file rather than globally.
It is not a hypothetical: on its first run it found two rows added minutes
earlier whose tests existed under other names. An exemption needs a stated
reason, as `spec_trace`'s does.

**[NORMATIVE — two of these were unevaluable until v1.1, and one was missing.]**
`KG-RANK` is a trend test over generations on the archive's elites' decoded
effective rank; `erank` was computed for every elite and written to no history
file, so the series the criterion is defined over did not exist and the
criterion could not be evaluated from a completed run at all. `KG-SHARE` was
stated in §01.6 with a threshold and omitted from this table, and the statistic
it names was never defined — §06.3.1 now defines it. Both are computed by
`scripts/gnomon_analyse.py`, which reports a criterion as **NOT EVALUABLE**
rather than as passing when the run did not record what it needs.

**[NORMATIVE]** `KG-HERIT` runs **before** the machinery it guards, on the smoke
run. That ordering is the process lesson of §00.3 and is not negotiable for
schedule reasons.

## 9.3 The registered prediction

**[NORMATIVE — recorded before `EVO` runs.]** §06.2 measured, at generation 0
on M-pico with `d_z = 16`, `N_life = 600`:

```
    h2      = 0.389      95% BCa [0.172, 0.620]
    sigma_P = 0.273 nats
```

The breeder's equation `R = i h2 sigma_P` with `i = 1.271` (truncation at the
top quarter) gives

```
    R = 0.135 nats per generation
```

**[NORMATIVE]** This is an **upper bound on the early response**, not a point
prediction. MAP-Elites is not truncation selection: parents are drawn uniformly
over filled niches, so its selection differential is strictly smaller than
global truncation's, and per-niche elitism spends part of the budget maintaining
diversity rather than pressing on quality. §9.6 reports the observed early slope
against this bound. A response *exceeding* it would indicate the fitness
estimate is correlated with something other than the genome — a leak, not a
success — and is treated as a defect to be diagnosed.

## 9.4 What may be claimed

**[NORMATIVE]**

* **May be claimed if measured:** the compression ratio; the generation curve
  and its significance; that MAP-Elites beats or does not beat random search;
  the `d_z` frontier; whether a genome evolved small transfers large; held-out
  CE against the two matched baselines.
* **May not be claimed:** that GNOMON is state of the art on TinyStories at any
  parameter count, unless `BASE-BIG` is actually beaten, which §01.10 records as
  unlikely; that compression accelerates learning (the PNAS paper found it does
  not — see below); any comparison to the paper's at-birth percentages, since
  theirs compress a network already trained on the target task.

**[NORMATIVE — a prediction taken from the paper, before GNOMON runs.]**

> genomic compression did not affect the learning trajectory; the only speedup
> was due to the higher initial performance, as though the p-network was
> "hot-started" by the g-network. Thus genomic compression, at least under these
> conditions, did not affect the learning rate.

If this transfers, evolution should move `L_birth` and the *level* of the curve
while leaving the late-phase slope unchanged. Recorded here so it cannot be
fitted afterwards.

## 9.5 Results recorded so far

**[NORMATIVE — measured, not projected. Every number re-measured before report.]**

**`G0-HERIT`** — M-pico, `d_z = 16`, `N_life = 600`, batch 32, 12 genomes x 3
seeds, 36 lifetimes in 237 s:

```
    h2 = 0.389   95% BCa [0.172, 0.620]     KG-HERIT: PASS
    V_G = 0.0290   V_E = 0.0455   sigma_P = 0.273
    best genome 6.071   worst 6.686   spread 0.615 nats
    between-genome sd 0.170   within-genome (seed) sd 0.213
    birth CE 8.683 (ln V = 8.318); birth_share at generation 0 = -0.18
```

The genome carries the function. SOMA's equivalent statistic was ~0.

**`HORIZON`** — how short a fitness lifetime can be, M-pico at batch 128,
12 genomes x 3 seeds at each length (§08.3 carries the full table):

```
    N_life  50   h2 0.402  [0.058, 0.699]  sigma_P 0.187   6.0 s   FAIL
    N_life 100   h2 0.310  [0.052, 0.677]  sigma_P 0.325  11.1 s   FAIL
    N_life 150   h2 0.398  [0.138, 0.726]  sigma_P 0.446  15.7 s   PASS
```

The point estimate is stable; what changes is `sigma_P`, which more than doubles
as the models actually separate, and with it the *evidence*. `N_life = 150` at
batch 128 is the same 2.5M-token budget as the `N_life = 600` at batch 32 above,
and reproduces its `h2` (0.398 against 0.389) in a third of the wall-clock.

## 9.6 Reporting rules

**[NORMATIVE]**

1. Every curve is held-out CE in nats on the **test** split, natural text, at the
   configuration's `T`; bits-per-byte alongside.
2. Structural events (a new elite, a change of best architecture) are annotated
   on the generation curve, never smoothed away.
3. Error bars are BCa bootstrap intervals over **run seeds**, and the number of
   seeds is always stated. Three is the minimum and is stated as a limitation
   wherever it is used.
4. Any arm that failed to run, or that was cut for budget, is listed explicitly
   with the reason. A missing arm is never silently omitted.
5. Every number in a final report is re-measured immediately before reporting,
   from the artifacts on disk, never quoted from an earlier message.
