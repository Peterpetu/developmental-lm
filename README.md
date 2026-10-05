# developmental-lm — six attempts to make a compact genome load-bearing in a small language model, each killed by its own criterion

Between 29 August and 10 September 2026 I ran six pre-registered research programmes on one question: **can a compact genome — something far smaller than the weights — carry a useful part of what a language model knows?** Each programme wrote its kill criteria before its first run and derived its verdicts mechanically from the run artifacts. Every compact-genome mechanism lost to simply copying weights at matched compute. The one place compression turned out to be load-bearing was transfer to a task family the source had never seen, not performance on a stationary task.

This repository holds the six programmes, their pre-registered protocols, the mechanically rendered results registers, and the run artifacts behind every number below. It also contains a faithful replication of the PNAS 2024 genomic bottleneck on a 13M-parameter language model, with the catch stated up front.

## The five numbers

| # | Finding | Numbers | Artifact |
|---|---|---|---|
| 1 | **The genomic bottleneck at birth, replicated** (Shuvaev, Lachi, Koulakov & Zador, PNAS 2024, on a 13M TinyStories LM). Innate ability is real: a decoded newborn is far from random before it has seen a token. | birth CE **2.863** (seed 0, generation 199) and **2.836** (seed 1, generation 200) vs a null of 9.011 and a fully trained adult at 1.857 | `runs/bottleneck/GN50-E-tinystories-s0-seed{0,1}/best.json`; protocol `docs/bottleneck/00-replication.md` |
| 2 | **But the embedding table had to be exempt.** With it compressed too, the newborn is near-null and the lifetime cost is large. | GN50-E keeps 2.10M of its 2.22M "genome" parameters in the uncompressed vocabulary table: **5.9× overall, 107× on the compressed layers**. Fully compressed: GN30 birth 7.67 / lifetime 3.52; GN50 birth 7.29 / lifetime 3.62, against 1.86 free | `runs/bottleneck/*/loop.json.gz` |
| 3 | **Where compression is load-bearing: transfer.** On a task family neither source had seen, weights copied from a free-trained source hinder; weights from the bottleneck-trained source do not, whether copied or decoded. | copied free weights **+1.328 nats** worse than a random start at 5M tokens; from the bottleneck-trained source, copy **+0.006**, decode **+0.007**; one seed | `runs/bottleneck/xfer-heldout-from-meta-{free,GN30-E}-tinystories-s0-seed0/transfer.json` |
| 4 | **Under environment shift, copying beats every compressed rebirth.** The compression ratio has no interior optimum; the argmin is "don't compress". | copy **4.079** < sieve-c2 4.317 < c8 4.488 < c4 4.569 < c16 4.703 < scratch 4.737 (union CE after a three-environment life); rehearsal: replay 1.981, generative rehearsal from the uncompressed model 2.125, EWC 3.358, LwF 3.189 | `runs/palimpsest/*/record.json`; register `docs/palimpsest/RESULTS.md` |
| 5 | **A backprop-free hyperdimensional LM works, and loses.** Binding/bundling context codes, a delta-rule memory and a ridge readout, with a 20-dial genome. | **2.408** nats at 25M tokens vs a trigram at 3.168; a 13M transformer at the same tokens: 1.857 | `runs/spore/final-25M/life.json`, `runs/spore/counts-25M/`; register `docs/spore/RESULTS.md` |

Everything was run on one RTX 4080 SUPER, on TinyStories, with models of at most 78M parameters, over thirteen days. Several arms are single-seed; the registers mark which. Those are the limits of the evidence and they are stated wherever a number appears.

## The programmes, in order

**SOMA** (`soma/`, documents in `archive/soma/`) — a transformer carrying a routed population of "mortal units" selected by ε-lexicase, with a germline/soma split and growth and death. It trained, and the machinery was inert: removing selection, birth and death changed the final cross-entropy by 0.003 (`archive/soma/runs/m-nano-s20260828{,-A3}/m1_report.json`: 2.1223 vs 2.1250), and a plain baseline was better (2.0205, `m0-m-nano-s20260828/m0_report.json`). The postmortem (`docs/gnomon/00-postmortem.md`) measured this in week three; the measurement had been available in hour six. Its §0.1 randomise-in-place table was a probe whose output was not archived; the archived report numbers are the ones above.

**GNOMON** (`gnomon/`, `docs/gnomon/`) — every weight matrix decoded from one shared decoder whose 5,306 parameters are the genome, searched by MAP-Elites over an open-ended archive; the architecture is decoded too. The compression is real (49.9M decoded weights from 20.7 KB, improving with scale). The quality is not: the evolved elite reaches **5.235** nats at 2,000 steps where a free-weight model of the *same* architecture reaches **3.020** and a 75k-parameter free model reaches **4.295** (`runs/gnomon/{gen-elite-2000,base-big-2000,base-free-2000}.json`). Heritability of lifetime outcome across genomes was 0.39 [0.17, 0.62], measured on 600-step lifetimes that end at 6–7 nats (`runs/gnomon/herit-pico-dz16.json`). The flat-archive and no-architecture ablation arms were never run; the protocol says so.

**ANLAGE** (`anlage/`, `docs/anlage/`) — a weight genome regressed from a trained 13M adult, then a programme genome (data mixture, growth schedule, recipe) with a sibling population. Six kill criteria, all fired: a decoded newborn scored 8.534 against an adult at 1.857 and a null at 9.011 (`runs/anlage/birth-tinystories-s0-dg64H4z16/birth.json`); training through the decoder collapsed to the null; assimilation cost +0.145; populations, growth schedules and the outer loop each lost to one long lifetime. The best model in this repository is the plain copy-grown 78M lineage at **1.569** nats (`runs/anlage/gen1-tinystories-single-s0/generation.json`). Seed noise was measured first (σ_e 0.003–0.006), so every verdict has a bar.

**PALIMPSEST** (`palimpsest/`, `docs/palimpsest/`) — the sixth model, built to answer one question with the five before it as evidence: how could compression be load-bearing? It states a "copy theorem" (an informal proposition: at matched compute, copying weights dominates any lossy compression on a stationary task) and tests the exits under environment shift. Finding 4 above. The FLOP-matched copy control named in the protocol (`docs/palimpsest/02-protocol.md`, R6) was not run; most arms are one seed; the register was rendered before ledger entries P-P.23–27 and those entries live in `docs/palimpsest/03-evidence-ledger.md`. The positive result — pseudo-rehearsal from the model's own samples works almost as well as kept-data replay — is a re-finding of generative replay (Robins 1995; Shin et al. 2017).

**SPORE** (`spore/`, `docs/spore/`) — finding 5. Its own gates: the hyperdimensional primitives are load-bearing (ablating binding, memory or growth costs 0.04–0.12 nats against a seed noise of 0.0007), the scaling criterion against the transformer fires, and the evolved genome's gain at 1M tokens does not carry to 5M.

**TEMPO** (`tempo/`, documents in `archive/tempo/`) — a lifespan agent with a span codec, an RL-trained gain genome, latent morphology and criticality control. Its evidence ledger reports the gain genome at +0.013 [−0.057, +0.082] over a random control (n=2) and the span interface costing 6.9× what it saves; its rendered results register predates those runs and lists every criterion as not evaluated. It is archived with the ledger as the record; nothing from it is claimed.

**BOTTLENECK** (`bottleneck/`, `docs/bottleneck/00-replication.md`) — findings 1–3. The replication was pre-registered against the paper as printed, after three earlier designs in this repository had each departed from the paper's procedure in a way that explained their nulls.

**SCREEN** (`screen/`) — the mechanism-screening harness (common random numbers, seed-noise estimation, operator-landscape probes) the later programmes ran on. One artifact: `runs/screen/topology-L4-T128.json`.

## How the verdicts were produced

Each programme has a protocol document written before its runs (`docs/<name>/02-protocol.md` or equivalent) with named kill criteria (`K-BIRTH`, `K-SIEVE`, `K-SOTA`, …). An analyser script (`scripts/<name>_analyse.py`) reads the run artifacts and prints each criterion as PASS, FIRES or NOT EVALUABLE; the `RESULTS.md` registers are its output and are never edited by hand. Protocols that changed during a campaign carry dated amendments inside the file; where an amendment post-dates a result, the register says so. Seed noise is measured before any contrast is read, and contrasts are reported with the bar they had to clear.

## Known limitations

* One GPU, one dataset family (TinyStories, plus a Python-source environment in PALIMPSEST), thirteen days. The numbers are small-scale by design and are not claims about large models.
* Several decisive arms are n=1 or n=2; the registers mark seed counts, and intervals are given only where they exist.
* The PALIMPSEST FLOP-matched copy control and the GNOMON flat-archive / no-architecture ablations were specified and not run.
* The genomic-bottleneck replication's headline number depends on exempting the embedding table (finding 2); the fully compressed arms are reported beside it for that reason.
* `screen/landscape.py` carries an unfinished edit made after SCREEN's one artifact (`runs/screen/topology-L4-T128.json`) was produced; four of its landscape tests fail against that edit as exported. Every other suite passes: 396 tests across ANLAGE, PALIMPSEST and GNOMON (6 skipped, 2 slow end-to-end deselected), 104 across BOTTLENECK, SPORE and the rest of SCREEN.

## Reproduce

```bash
pip install torch numpy
python -m pytest tests_bottleneck tests_anlage tests_palimpsest tests_spore tests_gnomon tests_screen -q
python scripts/build_tokenizer.py && python scripts/prepare_data.py      # TinyStories caches (6.4 GB, not included)
python scripts/bottleneck_loop.py --help                                   # the replication loop; arms in docs/bottleneck/00-replication.md
python scripts/palimpsest_analyse.py --markdown                            # re-render a register from runs/
```

The run artifacts (every `.json` the registers cite, with checkpoints omitted) are in `runs/`; large `loop.json` records are gzipped.

## Layout

```
soma/                      the shared backbone every later programme imports (the 13M transformer, data pipeline, task families) — and SOMA's own, archived, programme
bottleneck/ anlage/ palimpsest/ spore/ gnomon/ screen/ tempo/   the programmes' code, each with its tests_<name>/ suite (tests/ is SOMA's)
docs/<name>/               protocol, evidence ledger and results register per programme
runs/<name>/               the artifacts the registers cite
scripts/                   training loops, analysers, data preparation, gates
archive/soma/              SOMA's specification, gates and run reports; archive/tempo/ holds TEMPO's documents and ledger
```

## License and citation

The code and any released weights are licensed under the Apache License 2.0 (`LICENSE`); the documentation, figures and data under CC BY 4.0. Both require attribution: a copy or derivative of the code keeps the `NOTICE` file, and any reuse of the text, figures or data credits the author and links to the source.

If you use this work, build on it, or report results obtained with it, cite it:

> Lahtinen, P. (2026). *developmental-lm: six pre-registered attempts to make a compact genome load-bearing in a small language model.* https://github.com/Peterpetu/developmental-lm

```bibtex
@misc{lahtinen2026developmental,
  author = {Lahtinen, Peter},
  title  = {developmental-lm: six pre-registered attempts to make a compact genome load-bearing in a small language model},
  year   = {2026},
  url    = {https://github.com/Peterpetu/developmental-lm}
}
```

`CITATION.cff` carries the same reference for GitHub's "Cite this repository" button and for reference managers.
