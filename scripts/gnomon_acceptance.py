"""G-ACCEPT: every acceptance-test row in the specification has a test.

SPEC: 09 §9.2

`gnomon_spec_trace.py` checks that every **section** carrying a [NORMATIVE]
statement is referenced from somewhere in the tree. That is a coarse net, and
four rows slipped through it: §04.7's `P7` (the inference cache), §06.7's `S4`
(the probe reproduces SOMA's 0.024) and `S6` (split hygiene), and §08.6's `E3`
(seed pairing) were all specified as binding, all sat in sections whose *other*
rows were implemented, and none of them existed. A section-level check cannot
see that, because the section was mapped.

This is the row-level net. Each specification file carries tables of the form

    | # | property | test |
    |---|---|---|
    | T4 | spread (P2) | per-bit mean over the image lies in ... |

and each such id must appear in a test function named `test_<id>_...` in the
module that owns that specification file. Ids collide across files -- §02's `T1`
is a codomain check and §07's `T1` is a genome size -- so the mapping is by
file, not global.

    python scripts/gnomon_acceptance.py            # exit 1 if any row is unmapped
    python scripts/gnomon_acceptance.py --list     # also list the satisfied rows
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SPEC_DIR = REPO / "docs" / "gnomon"
TESTS = REPO / "tests_gnomon"

#: Which test module owns which specification file. A row's id is looked up
#: only there, because ids are unique within a file and not across the project.
OWNER = {
    "02-codes.md": ["unit/test_codes.py"],
    "03-decoder.md": ["unit/test_decoder.py"],
    "04-phenotype.md": ["unit/test_pheno.py"],
    "05-lifetime.md": ["unit/test_lifetime.py"],
    "06-share.md": ["unit/test_share.py"],
    "07-architecture.md": ["unit/test_arch.py"],
    "08-archive.md": ["unit/test_qd.py"],
    "10-plasticity.md": ["unit/test_plasticity.py"],
}

#: Rows whose evidence is a *run* rather than a test, with the run named. An
#: exemption without a reason is itself a defect (`spec_trace`'s rule).
EXEMPT: dict[str, str] = {
    "05-lifetime.md:L10": "a structural property of the module tree -- there is "
                          "one implementation of §5.1 -- asserted by "
                          "test_P7_a_training_decode_is_still_inside_the_loop "
                          "in test_pheno.py, which lives with the cache it "
                          "distinguishes itself from",
}

ROW_RE = re.compile(r"^\|\s*([A-Z]+\d+[a-z]?)\s*\|", re.M)
DEF_RE = re.compile(r"^def (test_([A-Za-z]+\d+[a-z]?)_\w*)", re.M)


def spec_rows() -> dict[str, list[str]]:
    """{spec file: [row id, ...]} for every acceptance-test table."""
    out: dict[str, list[str]] = {}
    for path in sorted(SPEC_DIR.glob("*.md")):
        ids = ROW_RE.findall(path.read_text(encoding="utf-8"))
        # Deduplicate while preserving order: a row may be restated.
        seen: list[str] = []
        for i in ids:
            if i not in seen:
                seen.append(i)
        if seen:
            out[path.name] = seen
    return out


def test_ids(rel_paths: list[str]) -> dict[str, list[str]]:
    """{row id: [test function name, ...]} defined in these modules."""
    found: dict[str, list[str]] = {}
    for rel in rel_paths:
        p = TESTS / rel
        if not p.exists():
            continue
        for fn, ident in DEF_RE.findall(p.read_text(encoding="utf-8")):
            found.setdefault(ident, []).append(fn)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list satisfied rows too")
    a = ap.parse_args()

    unmapped: list[tuple[str, str]] = []
    unowned: list[str] = []
    n_rows = n_ok = n_exempt = 0

    for spec, ids in spec_rows().items():
        owners = OWNER.get(spec)
        if owners is None:
            unowned.append(f"{spec}: {len(ids)} rows, no owning test module")
            continue
        have = test_ids(owners)
        for rid in ids:
            n_rows += 1
            key = f"{spec}:{rid}"
            if key in EXEMPT:
                n_exempt += 1
                if a.list:
                    print(f"  {key:34s} EXEMPT -- {EXEMPT[key]}")
                continue
            if rid in have:
                n_ok += 1
                if a.list:
                    print(f"  {key:34s} <- {', '.join(have[rid])}")
            else:
                unmapped.append((key, ", ".join(owners)))

    print(f"\nacceptance rows  : {n_rows}")
    print(f"exempt           : {n_exempt}")
    print(f"mapped           : {n_ok}")
    print(f"unmapped         : {len(unmapped)}")
    for key, owners in unmapped:
        print(f"  UNMAPPED {key}  -- expected `def test_{key.split(':')[1]}_...` "
              f"in {owners}")
    for line in unowned:
        print(f"  NO OWNER {line}")

    if unmapped or unowned:
        print("\nan acceptance row without a test is a requirement nobody checks")
        return 1
    print("\nevery acceptance row has a test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
