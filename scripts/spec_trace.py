#!/usr/bin/env python
"""Specification traceability checker.

Every section of ``docs/spec/*.md`` that contains at least one ``[NORMATIVE]``
statement must be referenced from code or tests by a marker of the form::

    # SPEC: 05 §5.2.5

Sections with no ``[NORMATIVE]`` content are exempt, as are sections listed in
``EXEMPT`` below (with a stated reason — an exemption without a reason is itself
a defect).

This is what turns "everything is implemented according to the spec" from a
claim into a check. It deliberately does *not* verify that the code is correct,
only that every binding statement has an owner; correctness is the job of the
test suite and the adversarial review.

Usage
-----
    python scripts/spec_trace.py                 # report, exit 1 if unmapped
    python scripts/spec_trace.py --list-mapped   # also list satisfied sections
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO / "docs" / "spec"
CODE_DIRS = [REPO / "soma", REPO / "tests", REPO / "scripts"]

# Section headings look like "## 5.2 The recurrent mixer" or "### 5.2.5 ...".
HEADING_RE = re.compile(r"^#{2,4}\s+(\d+(?:\.\d+)*)\s+(.*)$")
NORMATIVE_RE = re.compile(r"\[NORMATIVE")
# Spec file names begin with a two-digit number.
SPECFILE_RE = re.compile(r"^(\d{2})-")
# Markers in code: "SPEC: 05 §5.2.5", "SPEC: 05 5.2.5", "SPEC: 05§5.2.5"
MARKER_RE = re.compile(r"SPEC:\s*(\d{2})\s*(?:§|#|\s)\s*(\d+(?:\.\d+)*)")

# Sections whose normative content is not code-bearing. Each needs a reason.
EXEMPT: dict[str, str] = {
    "0.4": "notation conventions; enforced by 01.10 coverage, not by a module",
    "0.5": "marker vocabulary; a documentation convention",
    "9.1": "governs what may be claimed in prose, not code behaviour",
    "9.2": "governs what may be claimed in prose, not code behaviour",
    "9.3": "scaling argument; the measurable part is 08.6.1, which is mapped",
    "9.5": "scaling argument; the measurable part is profiling in 07.8",
    "9.6": "states unimplemented production requirements by design",
    "1.11": (
        "the symbol table: a glossary, normative about *names* rather than "
        "behaviour. Every symbol it lists is owned by the section that defines "
        "the quantity, and those sections are mapped; a marker here would point "
        "at the index rather than at the thing indexed"
    ),
    "8.8": (
        "states what is NOT built and NOT claimed below ~1e9 parameters. The "
        "only way to implement it is to not implement something, which no "
        "marker can witness; §09 carries the admissibility conditions"
    ),
    "10.2": (
        "the risk register: each entry names a risk and the section whose "
        "mechanism bears it, and those sections are mapped. R1-R8 are "
        "statements about what could go wrong, not procedures"
    ),
    "10.3": "epistemic caveats accompanying external presentation",
    "10.4": "a stated commitment about how results will be reported",
    "11.1": "evidence ledger; per-claim provenance, not code",
    "11.2": "evidence ledger",
    "11.3": "evidence ledger",
    "11.4": "evidence ledger",
}


def spec_sections() -> dict[str, tuple[str, str]]:
    """Return {key: (file, heading)} for sections containing [NORMATIVE]."""
    out: dict[str, tuple[str, str]] = {}
    for path in sorted(SPEC_DIR.glob("*.md")):
        m = SPECFILE_RE.match(path.name)
        if not m:
            continue  # AUDIT-v1.md and similar are not normative specs
        num = m.group(1)
        file_major = str(int(num))  # "05" -> "5"; headings are "5.2", not "05.5.2"
        current: str | None = None
        heading = ""
        buf: list[str] = []

        def flush() -> None:
            if current and any(NORMATIVE_RE.search(line) for line in buf):
                # Headings already carry the file number as their first
                # component ("5.2" in 05-backbone.md), so the section number IS
                # the key. Assert the invariant rather than assuming it.
                if current.split(".")[0] != file_major:
                    raise SystemExit(
                        f"{path.name}: heading '{current}' does not begin with "
                        f"file number {file_major}; numbering invariant broken"
                    )
                out[current] = (path.name, heading)

        for line in path.read_text(encoding="utf-8").splitlines():
            h = HEADING_RE.match(line)
            if h:
                flush()
                current, heading = h.group(1), h.group(2).strip()
                buf = []
            else:
                buf.append(line)
        flush()
    return out


def code_markers() -> dict[str, list[str]]:
    """Return {key: [files]} for every SPEC marker found in code."""
    found: dict[str, list[str]] = {}
    for root in CODE_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.resolve() == Path(__file__).resolve():
                continue  # the docstring example is not a real marker
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for fnum, sec in MARKER_RE.findall(text):
                if sec.split(".")[0] != str(int(fnum)):
                    raise SystemExit(
                        f"{path}: marker 'SPEC: {fnum} §{sec}' is inconsistent -- "
                        f"section must begin with {int(fnum)}"
                    )
                found.setdefault(sec, []).append(
                    str(path.relative_to(REPO)).replace("\\", "/")
                )
    return found


def _covered(key: str, markers: dict[str, list[str]]) -> bool:
    """A section is covered by an exact marker or by any marker on a subsection.

    Marking `05.5.2` also satisfies `05.5`, since implementing a subsection
    necessarily implements part of its parent.
    """
    if key in markers:
        return True
    return any(k.startswith(key + ".") for k in markers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-mapped", action="store_true")
    args = ap.parse_args()

    sections = spec_sections()
    markers = code_markers()

    unmapped: list[tuple[str, str, str]] = []
    mapped: list[str] = []
    for key, (fname, heading) in sorted(sections.items()):
        if key in EXEMPT:
            continue
        if _covered(key, markers):
            mapped.append(key)
        else:
            unmapped.append((key, fname, heading))

    total = len(sections)
    exempt_n = sum(1 for k in sections if k in EXEMPT)

    print(f"normative sections : {total}")
    print(f"exempt             : {exempt_n}")
    print(f"mapped             : {len(mapped)}")
    print(f"unmapped           : {len(unmapped)}")

    # Markers that point at a section which does not exist or is not normative:
    # these are stale references and are just as much a defect as an unmapped
    # section, because they create the appearance of coverage.
    stale = sorted(k for k in markers if k not in sections and k not in EXEMPT)
    if stale:
        print(f"\nSTALE MARKERS ({len(stale)}) -- point at a non-normative or absent section:")
        for k in stale:
            print(f"  SPEC: {k}  referenced from {', '.join(sorted(set(markers[k])))}")

    if args.list_mapped:
        print("\nMAPPED:")
        for k in mapped:
            print(f"  {k}  <- {', '.join(sorted(set(markers.get(k, ['(via subsection)']))))}")

    if unmapped:
        print(f"\nUNMAPPED ({len(unmapped)}):")
        for key, fname, heading in unmapped:
            print(f"  {key}  {fname}  {heading}")

    if unmapped or stale:
        return 1
    print("\nspec traceability complete")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
