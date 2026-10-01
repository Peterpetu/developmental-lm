# TEMPO — a lifespan agent model with nested timescales and a genomic channel

*Tempo* (music): the rate at which a passage moves. A piece has more than one.

The fourth model in this repository, designed 2026-09-04 against the brief "lifespan agent
model evolution, learning, compressing and adapting" and against everything SOMA, GNOMON and
ANLAGE measured. One continuous agent: a single perception stream, a single continuous
output, a fast per-span state, slow weights, a compact genome of gains written by a
reverse-time consolidation that reinforcement learning trains on the agent's own learning
improvement, a latent morphology that grows in large moves, and a measured edge of chaos.
Every mechanism has a matched-compute control and a pre-registered kill criterion.

These documents
are its normative form and take precedence where they differ. Every difference, and every
correction the measurements and the adversarial review of 2026-09-04 forced, is in
`11-evidence-ledger.md` §P-C (items 1–31).

| document | contents |
|---|---|
| [00-record-and-axioms](00-record-and-axioms.md) | the measured record, the PNAS re-read, the brief decoded, axioms A1–A9, reuse, the process rule |
| [01-agent-and-dynamics](01-agent-and-dynamics.md) | Definition 1, the three timescales, Heuristic 1 (timescale separation) with what it does *not* claim and the two quantities measured in its place, G-STAT (two-sided), acceptance A1–A8 |
| [02-interface](02-interface.md) | the perception stream, the span codec, decoding, losses and gradient insulation, continuous heads, the policy-gradient term, time codes, the seven-stream local blend with every generator, acceptance I1–I14 |
| [03-backbone](03-backbone.md) | configurations, the residual block, the delta and attention operators, the FFN, the gain fold and active views, parameter and FLOP accounting, acceptance P1–P12 |
| [04-genome-and-expression](04-genome-and-expression.md) | the genome, the role registry with counts, the low-rank expression map and ranges, the episode cycle of the gains (the PNAS loop in gain space), zero per-token cost, the K-GENE arms, acceptance E1–E12 |
| [05-morphology](05-morphology.md) | switches and groups, skipping, births by copy, Theorems G1 and G2, the primordium, FLOP accounting, the K-MORPH controls, acceptance M1–M10 |
| [06-consolidation-and-rl](06-consolidation-and-rl.md) | window summaries, the reverse read, the reward, the genome step, estimators 1 and 2, the short-horizon defences, the ES alternative, salience and rehearsal, acceptance C1–C12 |
| [07-criticality](07-criticality.md) | the propagated tangent, the per-sub-block and relative Lyapunov exponents, the stack's exponent, the homeostatic rule on ζ, references, K-CRIT, acceptance K1–K8 |
| [08-life-and-inference](08-life-and-inference.md) | birth, the loop with every ordering, lanes, boundaries and group rows, inference as the same loop, the genome reset, the checkpoint, the record, acceptance L1–L12 |
| [09-accounting-and-numerics](09-accounting-and-numerics.md) | parameters and compute, per-token accounting, memory, bounds, dtypes, the identity tests, acceptance N1–N8 |
| [10-protocol](10-protocol.md) | measurement rules (two fixed sets per stream, a σ per criterion, equal-compute comparison), gates and kill criteria, runs R0–R10 with budgets and seeds, manifests and the register, the one verdict rule |
| [11-evidence-ledger](11-evidence-ledger.md) | the machine and repository inventory, the verified literature (P-A, P-B), the corrections forced on the design (P-C), and the open observations recorded before the runs that decide them (P-D) |
| RESULTS.md | the register of every measured number (written as the runs complete) |

Code: `tempo/`; tests: `tests_tempo/`; scripts: `scripts/tempo_*.py`; runs: `runs/tempo/`.

The chain is one serial job: `python scripts/tempo_protocol.py --through R10` (idempotent: a
run whose record says `"complete": true` is skipped). `scripts/tempo_analyse.py` prints every
verdict from the records and nothing else; `scripts/tempo_register.py` renders `RESULTS.md`.
