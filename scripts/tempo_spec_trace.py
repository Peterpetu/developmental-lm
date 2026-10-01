"""The spec trace, in both directions.

  * every `SPEC:` tag in `tempo/*.py` names a section that exists in `docs/tempo/`, and every
    normative document is referenced by at least one module (the code points at the contract);
  * every pointer *between* the documents resolves: a section reference `NN §N.N`, an equation
    reference `(N.N)`, an acceptance row `I5`/`P12`, and a ledger item `11 §P-C.k` (the
    contract points at itself).

The second direction is the one that drifts silently. A correction that renames an equation --
splitting `(4.1)` into `(4.1a)` and `(4.1b)`, say -- leaves every earlier reference to `(4.1)`
pointing at nothing, and no test notices, because the documents are prose.

    python scripts/tempo_spec_trace.py [--strict]

Exit code 1 under `--strict` when anything points nowhere.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "tempo"
CODE = REPO / "tempo"

TAG_RE = re.compile(r"SPEC:\s*(\d\d)\s*(§[\d.]+(?:-[\d.]+)?|\([\d.]+\)|[A-Z]-[A-Z-]+|R\d+)?")
SECTION_RE = re.compile(r"^#{2,3}\s+(\d+(?:\.\d+)+)\b", re.M)
EQ_RE = re.compile(r"\((\d+\.\d+[a-z]?)\)")
ROW_RE = re.compile(r"^\|\s*([A-Z]\d+[a-z]?)\s*\|", re.M)
ROW_HOME = {"A": "01", "I": "02", "P": "03", "E": "04", "M": "05", "C": "06", "K": "07",
            "L": "08", "N": "09"}
EQ_HOME = {str(i): f"0{i}" for i in range(1, 10)}
EXTERNAL = ("SOMA", "ANLAGE", "GNOMON", "CALM", "PNAS", "Borkar", "et al.")
"""Equation numbers borrowed from another design's or another paper's specification are not
ours to resolve. The marker can sit on the previous line when a sentence wraps, so two lines
of context are examined."""
AXIOM_DOC = "00"
"""00 §0.4 numbers the *axioms* A1-A9. They share their letter with 01's acceptance rows and
are a different namespace, so acceptance-row references are not resolved inside 00."""


def cross_references(idx: dict) -> list[str]:
    """Every pointer from one document to another, and whether it resolves."""
    bad = []
    rows = {k: set(ROW_RE.findall(d["text"])) for k, d in idx.items()}
    for k, d in idx.items():
        text = d["text"]
        lines = text.splitlines()
        for line_no, line in enumerate(lines, 1):
            here = f"{d['file']}:{line_no}"
            context = line + (" " + lines[line_no - 2] if line_no >= 2 else "")
            external = any(e in context for e in EXTERNAL)
            for m in re.finditer(r"\b(\d{2})\s*§(\d+\.\d+)", line):
                doc, sec = m.group(1), m.group(2)
                if doc not in idx:
                    bad.append(f"{here}: document {doc} does not exist ({m.group(0)})")
                elif not any(s == sec or s.startswith(sec + ".") for s in idx[doc]["sections"]):
                    bad.append(f"{here}: {m.group(0)} -> no such section in {idx[doc]['file']}")
            if not external:
                for m in re.finditer(r"\((\d+\.\d+[a-z]?)\)", line):
                    eq, owner = m.group(1), EQ_HOME.get(m.group(1).split(".")[0])
                    if owner in idx and eq not in idx[owner]["equations"]:
                        bad.append(f"{here}: equation ({eq}) -> not defined in {idx[owner]['file']}")
            if k != AXIOM_DOC:
                for m in re.finditer(r"(?<![A-Za-z0-9_])([A-Z]\d+[a-z]?)(?![A-Za-z0-9_])", line):
                    rid = m.group(1)
                    owner = ROW_HOME.get(rid[0])
                    if owner in idx and rid not in rows.get(owner, ()):
                        bad.append(f"{here}: acceptance row {rid} -> not in {idx[owner]['file']}")
            for m in re.finditer(r"11\s*§P-([CD])\.(\d+)", line):
                sect = idx["11"]["text"].split(f"## P-{m.group(1)}")[-1] if f"## P-{m.group(1)}" in idx["11"]["text"] else idx["11"]["text"]
                if not re.search(rf"^{m.group(2)}\.\s+\*\*", sect, re.M):
                    bad.append(f"{here}: 11 §P-{m.group(1)}.{m.group(2)} -> no such ledger item")
    return bad


def doc_index() -> dict[str, dict]:
    idx = {}
    for f in sorted(DOCS.glob("[0-9][0-9]-*.md")):
        text = f.read_text(encoding="utf-8")
        idx[f.name[:2]] = {"file": f.name, "sections": set(SECTION_RE.findall(text)),
                           "equations": set(EQ_RE.findall(text)), "text": text}
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    idx = doc_index()
    missing, seen_docs = [], set()
    n_tags = 0
    for f in sorted(CODE.glob("*.py")):
        for line_no, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in TAG_RE.finditer(line):
                doc, ref = m.group(1), m.group(2)
                n_tags += 1
                if doc not in idx:
                    missing.append(f"{f.name}:{line_no}: document {doc} does not exist")
                    continue
                seen_docs.add(doc)
                if ref is None:
                    continue
                d = idx[doc]
                ref = ref.rstrip(".")
                if ref.startswith("§"):
                    sec = ref[1:].split("-")[0]
                    if sec not in d["sections"] and not any(s.startswith(sec + ".") or s == sec for s in d["sections"]):
                        missing.append(f"{f.name}:{line_no}: {doc} {ref} not found in {d['file']}")
                elif ref.startswith("("):
                    if ref[1:-1] not in d["equations"]:
                        missing.append(f"{f.name}:{line_no}: {doc} {ref} not found in {d['file']}")
                else:
                    if ref not in d["text"]:
                        missing.append(f"{f.name}:{line_no}: {doc} {ref} not found in {d['file']}")
    unreferenced = sorted(set(idx) - seen_docs - {"00", "10", "11"})
    print(f"{n_tags} SPEC tags over {len(list(CODE.glob('*.py')))} modules; "
          f"{len(missing)} dangling; documents unreferenced by code: {unreferenced or 'none'}")
    for m in missing:
        print("  " + m)
    xrefs = cross_references(idx)
    print(f"{len(xrefs)} dangling cross-references between the documents")
    for m in xrefs:
        print("  " + m)
    return 1 if (args.strict and (missing or unreferenced or xrefs)) else 0


if __name__ == "__main__":
    sys.exit(main())
