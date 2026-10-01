# PALIMPSEST — compression as a sieve: a lineage that consolidates into a small core and is reborn from it by growth

**Branch `tempo`, designed and built 2026-09-10** after reading every model document in this
repository from its beginning, for one question the user put: *how can the compression be
made load-bearing?* The sixth model. Every number below is re-measured from `runs/palimpsest/`
by `scripts/palimpsest_analyse.py`; every finding and "why" is in
[`03-evidence-ledger.md`](03-evidence-ledger.md).

## The answer in one paragraph

Five models put a compression of what a lineage had learned beside, under or around a
language model, and all six mechanisms lost at matched compute — by 0.007 to 4 nats, tracking
the tightness of the constraint. [`00 §0.3`](00-record-and-argument.md) shows why that had to
happen: on a stationary corpus with the previous weights available to copy, any genome fit
to those weights cannot beat copying them (the copy theorem), and the one exit the designs
hoped for — regularisation — is closed by the information count of a trained matrix. The
record's every *positive* reading of compression is of a different kind: the copied network
carried **damage** into a changed environment (it hindered, forgot, and on a family it never
saw was 1.3 nats worse than a random start) and the compressed one did not. **Compression is
load-bearing exactly when what it discards is harmful to what comes next**, and the only
place a language model's lifetime leaves harm behind is where the environment changes. So the
sixth model compresses in *function* space (a small core network written by distillation, not
a decoder of weights), decompresses by *development* (a function-preserving expansion, so the
phenotype is born computing the core and grows into the environment), lets the core *dream*
its past (generative rehearsal from the compressed store, no data kept), and is scored over a
*life* of environments against the copy and every conventional continual-learning control.
The claim is a double dissociation: an interior optimum in the compression ratio under a
shift of environments, and none without one.

## What was measured

*(rendered from the register as the runs complete — see [`RESULTS.md`](RESULTS.md))*

## Files

| file | contents |
|---|---|
| [`00-record-and-argument.md`](00-record-and-argument.md) | the record of five models read for the question; the copy theorem and its three exits; the creative-reasoning pass; axioms P1–P8 |
| [`01-model.md`](01-model.md) | environments and the life; the phenotype; the cores; the expansion (Theorem 1); distillation, dreams, consolidation; the arms; scores and accounting; acceptance tests T1–T11 |
| [`02-protocol.md`](02-protocol.md) | K-SIEVE and K-STAT (the double dissociation), the controls, the runs, reporting rules |
| [`03-evidence-ledger.md`](03-evidence-ledger.md) | P-P.0–: every finding and correction with its artifact |
| `palimpsest/` | `envs.py`, `expand.py`, `distil.py`, `sample.py`, `life.py`, `analyse.py` |
| `tests_palimpsest/` | 32 tests |
| `scripts/palimpsest_*.py` | `build_code` (the Python-source environment), `life` (one life), `analyse` (every verdict) |

## Running it

```
python -m pytest tests_palimpsest -q
python scripts/palimpsest_build_code.py
python scripts/palimpsest_life.py --name shift-copy-s0 --life shift --arm copy --seed 0
python scripts/palimpsest_life.py --name shift-sieve-c8-s0 --life shift --arm sieve --core c8 --seed 0
python scripts/palimpsest_analyse.py --markdown
```

A four-phase life of 8M tokens per phase is about ten minutes for the copy and about
fifteen for a sieve arm on the RTX 4080 SUPER.
