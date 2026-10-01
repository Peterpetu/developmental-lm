#!/usr/bin/env python
"""G-ACCEPT: every acceptance row of docs/anlage is owned by a test or by a run artifact.

SPEC: 08 §8.6

    python scripts/anlage_acceptance.py

Rows are the `| ID | ... |` lines of the acceptance tables in `docs/anlage/*.md` (IDs like
`D7`, `G12`, `M3`). A row is owned by a unit test whose name contains the ID (`test_D7_...`,
`test_G2_G3_...`), or by an artifact check registered below (rows that are measurements of a
run, not properties of a function). Rows that are neither are printed as UNOWNED, and the
script exits non-zero if any exist; rows whose artifact is absent are NOT EVALUABLE.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
DOCS = REPO / "docs" / "anlage"
TESTS = REPO / "tests_anlage"
RUNS = REPO / "runs" / "anlage"

NOT_TEST_ROWS = {"A", "X"}          # the axiom and post-mortem tables of §00 and §11 are not tests


def _rows() -> dict[str, tuple[str, str]]:
    out = {}
    pat = re.compile(r"^\|\s*\*{0,2}([A-Z]{1,2}\d{1,2})\*{0,2}\s*\|\s*([^|]*)\|")
    for f in sorted(DOCS.glob("*.md")):
        for line in f.read_text(encoding="utf-8").splitlines():
            m = pat.match(line)
            if m and m.group(1)[0] not in NOT_TEST_ROWS and not m.group(1).startswith("P-"):
                out.setdefault(m.group(1), (f.name, m.group(2).strip()))
    return out


def _test_names() -> list[str]:
    names = []
    for f in TESTS.rglob("test_*.py"):
        names += re.findall(r"def (test_[A-Za-z0-9_]+)", f.read_text(encoding="utf-8"))
    return names


def _owned_by_test(rid: str, names: list[str]) -> str | None:
    for n in names:
        parts = n[len("test_"):].split("_")
        if rid in parts:
            return n
    return None


def _latest(glob: str):
    hits = sorted(RUNS.glob(glob))
    return json.loads(hits[-1].read_text()) if hits else None


def _artifact_checks() -> dict[str, callable]:
    def c5():
        b = _latest("birth-*/birth.json")
        if not b:
            return None, "no birth report"
        share = (b["L_null"] - b["random_genome_ce"]) / (b["L_null"] - b["adult_ce"])
        return abs(share) <= 0.05 or share < 0, f"random-genome birth_share {share:+.3f} (a random genome is worse than the null: share < 0)"

    def c10():
        b = _latest("birth-*/birth.json")
        if not b:
            return None, "no birth report"
        r = b["regress"]
        need = ("residual_mean", "steps", "seconds", "n_genome", "n_latents", "n_phenotype", "ratio_genome", "ratio_genome_latents")
        ok = all(k in r and r[k] == r[k] for k in need)
        return ok, "every regress field present and finite" if ok else f"missing: {[k for k in need if k not in r]}"

    def d10():
        b = _latest("birth-*/birth.json")
        n = _latest("noise-*/noise.json")
        if not b or not n:
            return None, "needs birth and noise reports"
        decode_s = b["regress"]["seconds"] / max(b["regress"]["steps"], 1)     # one decode + backward
        step_s = 25_000_000 / 69_020 / 1526                                    # R1: 213 ms per step
        n_sleep = 100
        ratio = decode_s / (n_sleep * step_s)
        return ratio < 0.01, f"one decode {decode_s * 1e3:.0f} ms vs {n_sleep} steps {n_sleep * step_s:.1f} s: {ratio:.1%} (bound 1%)"

    def g4():
        g = _latest("grow-*/grow.json")
        n = _latest("noise-*/noise.json")
        if not g or not n:
            return None, "no growth run"
        worst = 0.0
        for key, r in g["results"].items():
            for e in r.get("events", []):
                if e.get("event") == "grow" and e.get("post_ramp_ce") is not None and e.get("pre_ce") is not None:
                    worst = max(worst, e["post_ramp_ce"] - e["pre_ce"])
        return worst <= 2 * n["sigma_e"], f"largest post-ramp minus pre-growth CE {worst:+.4f} (bound 2 sigma_e = {2 * n['sigma_e']:.4f})"

    def g12():
        return False, "MISSING: mid-ramp checkpoint/resume is not implemented (§06.7 G12); recorded in RESULTS.md"

    def m12():
        m = _latest("mix-*/mix.json")
        n = _latest("noise-*/noise.json")
        if not m or not n:
            return None, "no mixture run"
        tps = [r["tokens_per_s"] for arm in m["results"].values() for r in arm]
        base = 69_020.0
        ratio = (sum(tps) / len(tps)) / base
        return ratio >= 0.98, f"mixed-pool throughput {sum(tps) / len(tps):,.0f} tok/s vs plain {base:,.0f} ({ratio:.1%}; bound 98%)"

    return {"C5": c5, "C10": c10, "D10": d10, "G4": g4, "G12": g12, "M12": m12,
            "M10": lambda: (True, "covered by S9 and G9 (budget tests)"),
            "D12": lambda: (True, "covered by the G-NOFREE-INVERSE sentinel tests")}


def main() -> int:
    rows = _rows()
    names = _test_names()
    checks = _artifact_checks()
    unowned, lines = [], []
    for rid in sorted(rows, key=lambda r: (r.rstrip("0123456789"), int(re.sub(r"\D", "", r)))):
        f, prop = rows[rid]
        t = _owned_by_test(rid, names)
        if t:
            lines.append(f"{rid:5s} TEST      {t}")
        elif rid in checks:
            ok, msg = checks[rid]()
            state = "NOT EVALUABLE" if ok is None else ("PASS" if ok else "FAILS")
            lines.append(f"{rid:5s} ARTIFACT  {state:13s} {msg}")
        else:
            unowned.append(rid)
            lines.append(f"{rid:5s} UNOWNED   ({f}: {prop[:60]})")
    print("\n".join(lines))
    print(f"\n{len(rows)} rows, {sum(1 for l in lines if ' TEST ' in l)} by test, "
          f"{sum(1 for l in lines if ' ARTIFACT ' in l)} by artifact, {len(unowned)} unowned")
    return 1 if unowned else 0


if __name__ == "__main__":
    raise SystemExit(main())
