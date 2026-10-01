"""Every acceptance row of docs/tempo/01-09, and what owns it.

SPEC: 00 §0.6  ("every acceptance row in 01-09 is owned by a test or by an artifact check")

    python scripts/tempo_acceptance.py [--json runs/tempo/acceptance.json]

The specification's acceptance tables are the contract; this script reads them out of the
documents and asks, for every row, whether something in the repository actually checks it:

  * **a test** -- the row's id (`I5`, `P6`, `C10`, `L2`, ...) appears in a test function's
    name, its class's name, or its docstring, in `tests_tempo/`;
  * **an artifact check** -- the row is a property of a run's recorded numbers, listed in
    `ARTIFACT_ROWS` below and checked here against `runs/tempo/`;
  * **nothing** -- reported as UNOWNED, which is a failure of 00 §0.6, not a passing state.

It does not run the tests (that is `pytest tests_tempo`); it checks that the contract and the
suite have not drifted apart, which is the failure mode a green suite cannot detect.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from tempo_common import REPO, RUNS, write_json

if hasattr(sys.stdout, "reconfigure"):                 # the specification's tables carry Greek and box glyphs
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DOCS = REPO / "docs" / "tempo"
TESTS = REPO / "tests_tempo"
ROW_RE = re.compile(r"^\|\s*(?P<id>[A-Z]\d+[a-z]?)\s*\|\s*(?P<name>[^|]+?)\s*\|", re.M)
PREFIX_DOC = {"A": "01", "I": "02", "P": "03", "E": "04", "M": "05", "C": "06", "K": "07",
              "L": "08", "N": "09"}

ARTIFACT_ROWS = {
    "I1": ("the codec ceiling is < 0.01 nats/token", "codec"),
    "I2": ("the codec's noise gap is < 0.05 nats/token", "codec"),
    "N6": ("R1 records the peak memory per configuration and it is < 13 GB", "memory"),
    "L3": ("resume on CUDA agrees to 1e-5 over 10 steps", "cuda_resume"),
    "M5": ("the primordium's measured step time is < 60 % of the adult's", "step_time"),
    "N2": ("flops_per_span is within 5 % of the FLOP counter", "flops"),
}
"""Rows that a unit test cannot settle because they are properties of a run on this card."""


def rows_of(doc: pathlib.Path) -> list[tuple[str, str]]:
    text = doc.read_text(encoding="utf-8")
    out = []
    for m in ROW_RE.finditer(text):
        rid = m.group("id")
        if PREFIX_DOC.get(rid[0]) != doc.name[:2]:
            continue                                   # a cross-reference, not this table's row
        out.append((rid, m.group("name")))
    return out


def test_index() -> dict[str, list[str]]:
    """id -> the test classes and functions that name it.

    A row is owned when its id appears in a test's name (`test_a3_...`, `test_k2b_...`) or
    anywhere in that test's block (its docstring or an assertion's message). Blocks are cut at
    every `class`/`def` header, so an id in one test is not credited to its neighbour.
    """
    index: dict[str, list[str]] = {}
    for f in sorted(TESTS.rglob("test_*.py")):
        src = f.read_text(encoding="utf-8")
        for block in re.split(r"(?=^[ \t]*(?:class|def)\s)", src, flags=re.M):
            m = re.match(r"^[ \t]*(?:class|def)\s+(\w+)", block)
            if not m:
                continue
            name = m.group(1)
            ids = set(re.findall(r"\b([A-Za-z]\d+[a-z]?)\b", block))
            ids |= {r.upper() for r in re.findall(r"(?:^|_)([a-z]\d+[a-z]?)(?=_|$)", name)}
            ids |= set(re.findall(r"[A-Z]\d+[a-z]?", name))            # TestI9Insulation, TestK3K4K5
            for key in {i.upper() for i in ids}:
                if key[0] in PREFIX_DOC and re.fullmatch(r"[A-Z]\d+[A-Z]?", key):
                    index.setdefault(key, []).append(f"{f.relative_to(REPO)}::{name}")
    return index


def artifact_checks() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in sorted(RUNS.glob("codec-*-s*")):
        res = d / "result.json"
        if not res.exists() or d.name.endswith("-v1"):
            continue
        blob = json.loads(res.read_text(encoding="utf-8"))
        out.setdefault("I1", {})[d.name] = {"value": blob["ceiling_nll_per_token"], "ok": bool(blob["I1"])}
        out.setdefault("I2", {})[d.name] = {"value": blob["noise_gap_per_token"], "ok": bool(blob["I2"])}
    for d in sorted(RUNS.glob("R1-*-s*")):
        rec = d / "record.json"
        if not rec.exists() or d.name.endswith("-v1"):
            continue
        blob = json.loads(rec.read_text(encoding="utf-8"))
        if not blob.get("complete"):
            continue
        eps = blob.get("episodes", [])
        if eps:
            mem = eps[-1].get("peak_mem_gb")
            out.setdefault("N6", {})[d.name] = {"value": mem, "ok": bool(mem is not None and mem < 13.0)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    idx = test_index()
    arts = artifact_checks()
    report = {"rows": {}, "unowned": [], "artifacts": arts}
    lines = []
    for doc in sorted(DOCS.glob("0[1-9]-*.md")):
        rows = rows_of(doc)
        if not rows:
            continue
        lines.append(f"\n{doc.name}  ({len(rows)} rows)")
        for rid, name in rows:
            tests = idx.get(rid.upper(), [])
            art = ARTIFACT_ROWS.get(rid)
            checked = arts.get(rid)
            if art:
                if checked:
                    ok = all(v["ok"] for v in checked.values())
                    owner = f"artifact  {art[0]}: {'ok' if ok else 'FAILS'} in {len(checked)} run(s)"
                    kind = "artifact"
                else:
                    owner = f"artifact  {art[0]}: NOT MEASURED YET"
                    kind = "artifact-pending"
                if tests:
                    owner += f"  (named by {tests[0]})"
            elif tests:
                owner = f"test  {tests[0]}" + (f" (+{len(tests) - 1})" if len(tests) > 1 else "")
                kind = "test"
            else:
                owner = "UNOWNED"
                kind = "unowned"
                report["unowned"].append(f"{doc.name}:{rid}")
            report["rows"][rid] = {"doc": doc.name, "name": name, "kind": kind, "tests": tests}
            lines.append(f"  {rid:5s} {name[:58]:60s} {owner}")
    n = len(report["rows"])
    n_test = sum(1 for v in report["rows"].values() if v["kind"] == "test")
    n_art = sum(1 for v in report["rows"].values() if v["kind"].startswith("artifact"))
    lines.append(f"\n{n} acceptance rows: {n_test} owned by tests, {n_art} by artifact checks, "
                 f"{len(report['unowned'])} UNOWNED")
    if report["unowned"]:
        lines.append("UNOWNED (00 §0.6 requires an owner for every row): " + ", ".join(report["unowned"]))
    print("\n".join(lines))
    if args.json:
        write_json(pathlib.Path(args.json), report)
    raise SystemExit(1 if report["unowned"] else 0)


if __name__ == "__main__":
    main()
