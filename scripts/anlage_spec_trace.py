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
    python scripts/anlage_spec_trace.py                 # report, exit 1 if unmapped
    python scripts/spec_trace.py --list-mapped   # also list satisfied sections
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO / "docs" / "anlage"
# Only GNOMON's own tree. `scripts/` holds SOMA-era scripts whose `SPEC: 05
# §5.7` markers refer to SOMA's §05, and both projects number their spec files
# 00-09 -- so including them would let a marker written for one specification
# silently satisfy a section of the other. Only `scripts/gnomon_*.py` is scanned.
CODE_DIRS = [REPO / "anlage", REPO / "tests_anlage"]
CODE_GLOBS = [(REPO / "scripts", "anlage_*.py")]
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
    "0.3": "SCREEN's measurements; a record, not a procedure",
    "0.5": "the axioms; each is discharged by the section it names, and those are mapped",
    "0.6": "an inventory of what is reused; the provenance in each module's docstring is the witness",
    "0.7": "a process rule: scripts/anlage_noise.py is the instrument, and 08 §8.1 maps it",
    "1.1": "the parameter classes; owned by config, genome, latents and model, all mapped",
    "1.2": "names the four rates; each is implemented by the section that defines it",
    "1.3": "names the mechanisms discharging C1; each is owned by 05, 06 and 08",
    "1.4": "names the mechanisms discharging C2; each is owned by 03, 05, 07",
    "1.5": "generation zero is the baseline: a property of the run scripts, 08 §8.1",
    "1.6": "governs what may be claimed in prose",
    "1.7": "the barrier as a cost: 05 §5.3 carries the proposition",
    "1.8": "the symbol table: a glossary, normative about names",
    "2.7": "the scale-freedom claim; 03 §3.6 carries the count and K-SCALE the test",
    "3.8": "where the decode runs: enforced by lifetime.py, mapped under 04 §4.6",
    "4.9": "the baselines are arms of scripts/anlage_grow.py and anlage_pop.py (08 §8.1)",
    "5.1": "the definition; its three instances are owned by 04 §4.4 (mixer), 04 §4.6 (step) and 05 §5.2",
    "5.3": "a proposition with its proof; tested through C2 and C3",
    "5.7": "registered ablations; arms of later runs",
    "5.8": "a proposition; tested by D7 and C8",
    "6.1": "rationale",
    "6.4": "the ordering of regression before growth; scripts/anlage_grow.py implements it",
    "6.5": "the schedule; a configuration table consumed by the run scripts",
    "7.1": "rationale",
    "7.3": "mutation: population.py, mapped",
    "7.5": "the compute split; a run-script argument (scripts/anlage_pop.py)",
    "7.6": "the G-LIVE screen; reuses screen/variance.py, invoked by the run scripts",
    "8.1": "the run schedule; scripts/anlage_*.py are the runs",
    "8.3": "the criteria; scripts/anlage_analyse.py evaluates them",
    "8.4": "the dials; a rule about provenance carried in manifests",
    "8.5": "the manifest; scripts/anlage_common.py writes it",
    "8.6": "reporting rules for prose",
    "9.1": "governs what may be claimed",
    "9.5": "governs what may be claimed",
    "10.8": "a list of insufficiently verified numbers; governs prose",
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