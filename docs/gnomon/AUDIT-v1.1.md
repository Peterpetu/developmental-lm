# AUDIT v1.1 — is GNOMON implemented to its specification?

**Status: register, not specification.** This is the record of a full audit of
the v1.0 tree against `docs/gnomon/00..10`, and of what was done about each
finding. It is deliberately not a spec file: `scripts/gnomon_spec_trace.py`
matches `^\d{2}-`, and a register is evidence rather than a requirement.

The question asked was narrow: **is every normative statement implemented, and
is the tree ready to run §09.1's schedule?** The answer at v1.0 was no, in
seventeen respects, of which four would have made the remaining runs either
impossible or misleading.

Method: read all 2,097 lines of specification; read every module; cross-check
each acceptance-test table row against the suite; run the code.

## Summary

| | v1.0 | v1.1 |
|---|---|---|
| tests | 194 | **275** |
| planted defects (`G-MUTATE`) | 22 | **26** |
| spec sections mapped (`G-SPECMAP`) | 61 / 61 | 62 / 62 |
| acceptance rows with a test (`G-ACCEPT`) | *unchecked* | **80 / 80**, 1 exempt |
| configured constants checked (`G-SPECNUM`) | *no gate* | **37 tests** |
| source files not under version control | **4** | 0 |
| driver scripts that start | 14 / 15 | **20 / 20** |

Reproduce every row: `python -m pytest tests_gnomon -q`,
`python scripts/gnomon_spec_trace.py`, `python scripts/gnomon_acceptance.py`.

## A. Findings that blocked the remaining runs

### A-1 — four source modules were not in the repository

`.gitignore` carried an unanchored `data/`, written for the 6.4 GB of tokenised
corpora at the repository root. It also matched `gnomon/data/` and
`soma/data/`, which hold *source*: the token stream, the corpus writer and the
tokenizer. `gnomon.lifetime.runner` imports `..data.stream` at module scope, so
a fresh clone of the branch could not import the package at all.

**Resolution.** Pattern anchored to `/data/`; 1,349 lines recovered. A sweep for
any other `.py` hidden by an ignore rule found none.

### A-2 — §02.9's token code was normative, tested, and never used

§2.1's table and §2.9 make Gray-over-frequency-rank the specified coding for the
token axis, and §2.9 measures it at **+0.0999 nats, 95% BCa [+0.0288, +0.2647]**
against §2.4's bijection under plain AdamW — the larger of the two effects §10
found. `Decoder.set_token_rank` implemented it and T11–T13 tested it; nothing
but `scripts/gnomon_tokencode.py` ever called it. Every evolutionary run, every
baseline, every transfer and the completed `EVO` archive used the coding the
specification had already rejected.

**Resolution.** Installed by default from the training shard before the first
step, with the counts' digest in the run manifest. §2.4 remains reachable as the
control that measured the difference. Two further defects fell out:

* the count window was the first 20M tokens, leaving **243 of 4,096** types
  unseen; over all 471,987,251 it is **173**, which is the corpus's own property
  and which no window recovers. The full pass costs 1.6 s against a lifetime's
  15, so the bound bought nothing and cost a definition;
* `frequency_rank` used `torch.argsort` without `stable=True`. **508 of 4,096**
  types are tied, the largest group being those 173, and on this input the
  default sort *differs* from the stable one — so the code was a function of
  sort internals rather than of the corpus. Ties now break by ascending id.

### A-3 — the compression claim was overstated by 15%

SOMA carried a `G-SPECNUM` gate that recomputed every configured constant.
GNOMON had none, and the project's headline quantity was stated three times and
agreed with the code in none of them.

| quantity | §01.7 | §03.5 | §07.2 | measured |
|---|---|---|---|---|
| `\|phenotype\| : \|Gamma\|` at M1 | 11,200 | 10,830 | 10,324 | **9,458** |
| `\|Gamma_a\|` | — | ~550 | 796 | **796** |
| `\|Gamma\|` | — | ~5,033 | 5,279 | **5,279** (+27 = 5,306) |
| germline at M1 (§01.1) | ~4,850 | | | **5,306** |
| `n_c` (§02.2) | | 12 | | **12** (§02.2 said 10) |
| decoded params, pico/nano/M1 (§04.4) | 1.57 M / 3.4 M / 54.5 M | | | **1,528,969 / 4,060,883 / 49,926,705** |

None was caught because the only test that touched them asserted `ratio > 100`
and that the ratio improves with scale — both true of every wrong value as well
as of the right one.

**Resolution.** `tests_gnomon/unit/test_spec_numbers.py`: 32 tests asserting
each constant as an *equality* against a value recomputed from the module tree,
including the `H log N` derivative (`+2 d_h` per channel bit, `+d_h` per vocab
bit) at every `d_z` on §06.5's frontier. Every document corrected. The property
the claim was making is unchanged and now checked.

### A-4 — the acceptance tests certified a path the science did not take

§05.1 had two implementations. `lifetime()` was exercised by every acceptance
test L1–L9; `LifetimeRunner.run` produced every number the project has reported.
They had drifted: `lifetime()` ran plain `dpr` when asked for `dpr-frozen`,
could not express §10.6's pinned-coefficient probe at all, and would have gone
on decoding §2.4's token code after A-2's fix. A suite that certifies a path the
science does not take is §00.3's defect one level down.

**Resolution.** `lifetime()` is a wrapper; L1–L9 bind the runner. This exposed a
test that had become vacuous: L3 asserted that `lifetime()` cleared
`requires_grad` on the `Decoder` handed to it — a side effect of that
implementation, not §05.1's invariant — and the runner *copies* a genome's
values, so the assertion became trivially true. Restated over the tensors that
take part in a lifetime, with `G-MUTATE` M22 planting a germline write to show
it can still fail.

## B. Specified and unimplemented

| # | what | where |
|---|---|---|
| B-1 | `A-LAMARCK`, the arm every Darwinian result is measured against | §08.5; §01.7 and §05.1 both pointed at "§07.6", which is an acceptance-test table |
| B-2 | `KG-RANK` evaluable — `erank` was computed per elite and written to no history, so the series the criterion is defined over did not exist | §07.4, §09.2 |
| B-3 | `KG-SHARE` — stated with a threshold in §01.6, omitted from §09.2's table, and the statistic it names defined nowhere | §01.6, §06 |
| B-4 | `P7` / `G-INFER`, the inference cache | §04.7 |
| B-5 | `S4`, the probe reproduces SOMA's 0.024 | §06.7 |
| B-6 | `S6`, split hygiene | §06.7 |
| B-7 | `E3`, seed pairing | §08.6 |
| B-8 | `L8` — only the *stream* was resumable; the lifetime had no checkpoint, so §09.1's 8-hour M1 run could not survive an interruption | §05.7 |
| B-9 | the `FRONTIER` driver | §06.5, §09.1 |
| B-10 | the `M1` driver | §09.1 |
| B-11 | `gnomon/lifetime/graph.py` — 110 lines arguing CUDA-graph capture is the only acceleration surviving architecture search, imported by nothing | §07.1 |

All eleven are implemented. Three carry decisions worth recording:

**B-1.** Three definitional choices, now in §08.5: the germline step is taken
after the soma's, so both see the same `W` within a step; consecutive seeds are
consecutive lives of one lineage, with the consequence stated that the mean over
seeds is *not* an unbiased fitness estimate in this arm; and the step size is
`0.1x` the soma's and untuned, because an upper bound tuned against the arms it
bounds is not one.

**B-3.** §06.1 rules `share_retrain` out as the gate on the smoke run — at
generation 0 it is zero by construction — and that argument does not make it
unreportable once evolution has run. §06.3.1 defines it: a random genome given a
full lifetime, against the evolved genome given none.

**B-11.** Connecting it found the blocker: `F.embedding`'s backward sorts its
indices and calls `unique_by_key`, which synchronises, and a synchronising
kernel cannot be captured. The advanced-indexing form can be, and the two agree
to under one float32 ulp across six shapes (relative 0 to 6.8e-08). Measured
against eager at M-pico: step 1 identical, steps 2–12 at 1e-7, step 20 at 6e-3,
each backend bit-identical to itself, **1.60x** throughput, 1.8 s to bind per
architecture against `torch.compile`'s ~30. Opt-in; `A-LAMARCK` refuses under it.

## C. Recorded as a defect and not fixed

### C-1 — the learning rate (§10.6)

§10.6 states it plainly: "`lr` was `0.03` throughout, chosen from a sweep at
2,000 steps and then used at 300 and 600 [...] Those runs are re-baselined
before any of them is reported as a result." `M_PICO.lr` was still `3e-3` and
the completed `EVO` run used it.

Measured at the fitness horizon (M-pico, `N_life = 150`, batch 128, 9 rates x 6
seed-paired lifetimes, plain AdamW, §2.9's token code):

| `lr` | held-out CE | vs best | 95% BCa on the paired difference |
|---|---|---|---|
| 0.015 | 6.0245 | +0.4262 | [+0.3508, +0.4900] |
| 0.030 | 5.8353 | +0.2371 | [+0.1724, +0.3036] |
| 0.060 | 5.6867 | +0.0885 | [+0.0349, +0.1287] |
| 0.090 | 5.5985 | +0.0003 | [−0.0910, +0.0644] indistinguishable |
| **0.120** | **5.5982** | — | the best |
| 0.180 | 5.6444 | +0.0462 | [+0.0199, +0.0752] |
| 0.250 | 5.6713 | +0.0730 | [+0.0272, +0.1402] |
| 0.350 | 5.7737 | +0.1755 | [+0.1035, +0.2257] |
| 0.500 | 5.8591 | +0.2609 | [+0.1841, +0.3298] |

The curve **turns inside the grid** — 0.09 → 0.12 → 0.18 gives 5.5985 → 5.5982 →
5.6444 — which §10.6's own sweep could not establish, because its grid ended at
0.12 with 0.12 the best value and what it measured was therefore a bound.

And at M-pico's **own** horizon (`N_life = 600`, batch 32), which is the value
`M_PICO.lr` carries:

| `lr` | held-out CE | vs best | 95% BCa on the paired difference |
|---|---|---|---|
| 0.015 | 5.5871 | +0.2690 | [+0.2100, +0.3266] |
| 0.030 | 5.4016 | +0.0835 | [+0.0458, +0.1267] |
| **0.060** | **5.3181** | — | the best |
| 0.090 | 5.3503 | +0.0322 | [−0.0027, +0.0770] indistinguishable |
| 0.120 | 5.3212 | +0.0031 | [−0.1089, +0.0981] indistinguishable |
| 0.180 | 5.3808 | +0.0627 | [−0.0365, +0.1606] indistinguishable |
| 0.250 | 5.4269 | +0.1088 | [+0.0195, +0.1960] |
| 0.350 | 5.5780 | +0.2600 | [+0.1612, +0.3508] |
| 0.500 | 5.7109 | +0.3928 | [+0.3013, +0.4480] |

Turning inside the grid again — 0.03 → 0.06 → 0.09 gives 5.4016 → 5.3181 →
5.3503 — over a broad plateau. `3e-3`, the configured value, is off the bottom:
worse than 0.015, which is itself +0.269 nats behind.

**Resolution.** `M_PICO.lr = 0.06`, `gnomon_evolve.py --lr` defaults to 0.09,
each being the smallest of its own indistinguishable set — the conservative end
of a region the data cannot separate, which degrades more gracefully when the
horizon or the batch moves.

The larger change is structural. A rate is only meaningful together with the
horizon and batch it was measured at, so `PhenoConfig.lr_provenance` carries
that string, it names the artifact, it begins with `measured` or `NOT MEASURED`
so a reader can tell at a glance, and it lands in every run manifest.
`G-SPECNUM` asserts that a *measured* claim names its artifact and matches the
configuration's own `N_life` and `batch` — so "measured" cannot quietly come to
mean "measured somewhere else", which is exactly what the defect was.

`scripts/gnomon_lrsweep.py` refuses to call an edge value an optimum, and
`scripts/gnomon_setlr.py` reads the recommendation back out of the artifacts
rather than from a message (§09.6 rule 5).

**M-nano and M1 are marked `NOT MEASURED`, with the reason.** §06.5's `FRONTIER`
runs at M-nano, so running it at `3e-3` would repeat this defect at the very run
that fixes the operating point; the M-nano sweep is the last outstanding
measurement. At M1 a 9-point sweep is 54 full lifetimes at `N_life = 20000`
against §09.1's 8-hour budget for one, so whatever is used there is an
extrapolation and is reported as one. Both drivers print a warning when they run
at an unmeasured rate.

## D. Deviations left standing, with reasons

**D-1 — the behavioural suite is inapplicable, not merely unrun.** §09.1's `M1`
row read "behavioural suite" and §00.4 lists it under "verified and reused".
SOMA's suite grades PARITY, MODCOUNT, PERM, DYCK, MQAR and COPY: formal-language
tasks *a model was trained on*. GNOMON trains on TinyStories and nothing else,
§05.2 having removed every auxiliary objective on purpose, so its accuracy on
them is chance by construction. §09.1 now says so, per §09.6 rule 4.

**D-2 — the document mask is implemented, tested, and unused in training.**
`Gnomon.forward` accepts `doc_id` and §04.7's P4 asserts the mask leaks nothing,
but `TokenStream` samples packed windows that cross document boundaries and no
run supplies `doc_id`. This is the standard packed causal-LM stream and §05.2
does not require otherwise, so it is a deviation of the *data pipeline* from
what the operator supports rather than from a normative statement. Changing it
would be a research decision, not a completion, and is left to the author.

**D-3 — §00.1's ablation-A3 row is internally inconsistent.** It states CE
2.1250 and delta +0.0027; every other row satisfies `delta = CE − 2.1005`. The
prose, §0.1's third corroborating signal and the 0.04% figure all reconcile with
the delta. §00.1 records the discrepancy rather than carrying both silently.
Which is the typo is not established, and the conclusion does not turn on it: at
either value the evolutionary machinery contributed under 0.4% of the learning.

**D-4 — `M_NANO.lr` and `M1.lr` are measured only where they were measured.**
The sweep above is at M-pico. Any configuration run at an unmeasured rate
inherits C-1's defect, so the sweep is run per configuration and the horizon it
was measured at is stated with the value.

## E. Defects found by *running* the code, not by reading it

Recorded separately because §00.3's lesson is that a conformance suite cannot
find this class.

| | |
|---|---|
| E-1 | `scripts/gnomon_smoke.py` raised `TypeError` on every invocation — still passing v0.9's `use_eta=` to a `lifetime()` that had not accepted it since v1.0 |
| E-2 | a history row containing `NaN` is not valid JSON and does not compare equal to itself, so E4's determinism and E7's round-trip could not be asserted over one. Result files now go through `write_json`, which passes `allow_nan=False` |
| E-3 | `LifetimeRunner.__init__` called a method that touched graph state not yet constructed — found by the first smoke run after the change, not by 194 passing tests |
| E-4 | the regex that routed 14 scripts through `write_json` matched `pathlib.Path(out, "config.json")` as one argument group; `gnomon_evolve.py` then raised before its first lifetime |
| E-5 | `G-ACCEPT`, on its first run, found two acceptance rows added minutes earlier whose tests existed under other names |

## F. What is still to run

Implementation is complete; §09.1's schedule is not. Standing at v1.1:

| run | status |
|---|---|
| `G0-HERIT`, `HORIZON` | done (§09.5) |
| `EVO` | **1 of 3 run-seeds**, and that one under §2.4's token code and `lr = 0.003`, so it is superseded rather than partial |
| `EVO-RAND`, `EVO-NOARCH`, `A-NOX`, `A-1D`, `A-FIXSIG`, `A-ANNEAL`, `A-LAMARCK` | not run |
| `FRONTIER` | not run; driver exists |
| `TRANSFER` | not run; driver exists |
| `M1` | not run; driver exists |
| `BASE-FREE`, `BASE-BIG` | run, at the `lr` C-1 records as defective; to be re-baselined |

`KG-HERIT` has fired and passed. `KG-FLAT`, `KG-QD`, `KG-SCALE`, `KG-RANK` and
`KG-SHARE` are implemented and unevaluated, and `scripts/gnomon_analyse.py`
reports each as **NOT EVALUABLE** rather than as passing until the runs that
decide it exist.
