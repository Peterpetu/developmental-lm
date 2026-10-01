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
SPEC_DIR = REPO / "docs" / "gnomon"
# Only GNOMON's own tree. `scripts/` holds SOMA-era scripts whose `SPEC: 05
# §5.7` markers refer to SOMA's §05, and both projects number their spec files
# 00-09 -- so including them would let a marker written for one specification
# silently satisfy a section of the other. Only `scripts/gnomon_*.py` is scanned.
CODE_DIRS = [REPO / "gnomon", REPO / "tests_gnomon"]
CODE_GLOBS = [(REPO / "scripts", "gnomon_*.py")]
# Section headings look like "## 5.2 The recurrent mixer" or "### 5.2.5 ...".
HEADING_RE = re.compile(r"^#{2,4}\s+(\d+(?:\.\d+)*)\s+(.*)$")
NORMATIVE_RE = re.compile(r"\[NORMATIVE")
# Spec file names begin with a two-digit number.
SPECFILE_RE = re.compile(r"^(\d{2})-")
# Markers in code: "SPEC: 05 §5.2.5", "SPEC: 05 5.2.5", "SPEC: 05§5.2.5"
# The left boundary matters: without it `SOMA-SPEC:` -- the provenance marker on
# the vendored modules, which points at SOMA's specification -- also matches, and
# since both projects number their spec files 00-09 a SOMA marker would silently
# satisfy a GNOMON section.
MARKER_RE = re.compile(r"(?<![A-Za-z-])SPEC:\s*(\d{2})\s*(?:§|#|\s)\s*(\d+(?:\.\d+)*)")
# Sections whose normative content is not code-bearing. Each needs a reason.
EXEMPT: dict[str, str] = {
    "0.1": "the post-mortem's measurements; a record, not a procedure",
    "0.2": "the post-mortem's root causes; each is answered by a mechanism in "
           "§01, and those sections are mapped",
    "0.3": "a process lesson about when a gate runs; §06 carries the gate",
    "0.4": "an inventory of what was vendored; the provenance headers on the "
           "vendored files are the witness",
    "1.7": "the barrier is a theorem about capacity, not a procedure: it is "
           "witnessed by |Gamma| : |phenotype| in §03.5, which is mapped",
    "1.8": "names the two timescales; each is implemented by the section that "
           "defines it (§05 and §08), and both are mapped",
    "1.9": "names the three novel mechanisms; each is implemented by §03.4, "
           "§04.5 and §08, all mapped",
    "1.10": "governs what may be claimed in prose, not code behaviour",
    "1.11": "the symbol table: a glossary, normative about *names*. Every "
            "symbol it lists is owned by the section defining the quantity",
    "2.7": "states the scale-freedom claim; the code that makes it true is "
           "§02.2 and §02.4, and the test is KG-SCALE in §09",
    "9.1": "the run schedule; scripts/gnomon_*.py are the runs themselves",
    "9.3": "a registered prediction, checked in §09.6's report",
    "9.4": "governs what may be claimed in prose, not code behaviour",
    "9.5": "recorded measurements; a record, not a procedure",
    "9.6": "reporting rules for prose",
    "10.5": "a prediction registered before the arms were run; section 10.5b records what was measured against it",
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
    paths = [p for root in CODE_DIRS if root.exists() for p in root.rglob("*.py")]
    paths += [p for root, pat in CODE_GLOBS if root.exists()
              for p in root.glob(pat)]
    if True:
        for path in paths:
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