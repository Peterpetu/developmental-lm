"""Every verdict from the records on disk; `--markdown` renders docs/palimpsest/RESULTS.md.
SPEC: docs/palimpsest/02-protocol.md."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from palimpsest.analyse import load_records, report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    text = report(load_records())
    print(text)
    if args.markdown:
        md = REPO / "docs" / "palimpsest" / "RESULTS.md"
        md.write_text("# PALIMPSEST — the results register\n\nRendered by `python scripts/palimpsest_analyse.py --markdown` on "
                      + time.strftime("%Y-%m-%d %H:%M") + " from `runs/palimpsest/*/record.json`; every number carries its "
                      "artifact. Verdicts: `02-protocol.md`; readings and reasons: `03-evidence-ledger.md`. Never edit by hand.\n\n```\n"
                      + text + "\n```\n", encoding="utf-8")
        print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
