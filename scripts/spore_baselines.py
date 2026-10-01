"""The count-model nulls on the same stream and windows. SPEC: docs/spore/02-protocol.md §2.2.

    python scripts/spore_baselines.py --name counts-5M --budget 5000000
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from spore_common import RUNS, load_all, log_to, manifest, write_record  # noqa: E402

from spore.baselines import fit_count_models  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--budget", type=int, default=5_000_000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch-windows", type=int, default=32)
    args = ap.parse_args()
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    lin, corpus, ev_fit, ev_test = load_all()
    rec = fit_count_models(corpus.train, budget=args.budget, offset=args.offset, T=lin.T,
                           batch_windows=args.batch_windows, V=lin.vocab, ev_fit=ev_fit, ev_test=ev_test, log=log)
    rec.update(manifest({"run": args.name, "model": "counts", "budget": args.budget, "offset": args.offset,
                         "eval_test_digest": ev_test.digest, "eval_fit_digest": ev_fit.digest}))
    write_record(out / "counts.json", rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
