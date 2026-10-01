"""Read the `lr` sweeps and report the value each configuration should carry.

SPEC: 05 §5.4, SPEC: 09 §9.6 rule 5, SPEC: 10 §10.6

§09.6 rule 5: "Every number in a final report is re-measured immediately before
reporting, from the artifacts on disk, never quoted from an earlier message."
The configured `lr` is such a number -- §10.6 records that using one measured at
a different horizon is a defect -- so it is read out of `runs/gnomon/lrsweep-*`
rather than transcribed.

Reports rather than edits. Which value a configuration carries is the author's
decision; this makes it an informed one, and refuses to recommend a value whose
sweep put the optimum at the edge of its grid.

    python scripts/gnomon_setlr.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

files = sorted(glob.glob("runs/gnomon/lrsweep-*.json"))
if not files:
    sys.exit("no sweeps in runs/gnomon/; run scripts/gnomon_lrsweep.py first")

print(f"{'sweep':34s} {'N_life':>7s} {'batch':>6s} {'best lr':>8s} "
      f"{'CE':>8s} {'runner-up':>10s}  verdict")
rows = []
for f in files:
    d = json.loads(pathlib.Path(f).read_text())
    best = d["best_lr"]
    by_lr = {r["lr"]: r for r in d["rows"]}
    # The runner-up is only interesting when it is *indistinguishable* from the
    # best: a sweep whose top two overlap has located a plateau, not a point,
    # and saying so is more honest than reporting the argmin to three digits.
    ties = [r["lr"] for r in d["rows"]
            if r["lr"] != best and r["ci"][0] <= 0.0 <= r["ci"][1]]
    verdict = ("EDGE OF GRID -- a bound, not an optimum; extend --lrs"
               if d["edge_of_grid"] else
               f"plateau with {ties}" if ties else "a turning point")
    rows.append((d, best, ties, verdict))
    print(f"{pathlib.Path(f).stem[:34]:34s} {d['N_life']:7d} {d['batch']:6d} "
          f"{best:8.3f} {by_lr[best]['mean']:8.4f} "
          f"{str(ties)[:10]:>10s}  {verdict}")

print()
for d, best, ties, verdict in rows:
    if d["edge_of_grid"]:
        print(f"{d['config']} at N_life={d['N_life']}: NO RECOMMENDATION -- "
              f"{verdict}")
        continue
    # Where the top values are indistinguishable, take the *smallest* of them.
    # A learning rate at the low edge of a plateau is the conservative end of a
    # region the data cannot separate, and it degrades more gracefully if the
    # horizon or the batch moves.
    choice = min([best] + ties)
    note = (f" (smallest of the indistinguishable set {sorted([best] + ties)})"
            if ties else "")
    print(f"{d['config']:8s} N_life {d['N_life']:6d} batch {d['batch']:4d}  "
          f"-> lr = {choice:g}{note}")
    print(f"         provenance: measured at N_life={d['N_life']}, "
          f"batch={d['batch']}, rule={d['rule']}, token_code={d['token_code']}, "
          f"{d['genomes']}x{d['seeds']} seed-paired lifetimes")
