"""`G-NOSTUB`: no placeholder implementation survives anywhere in the package.

The project rules make this a hard rule: "A function signature with `pass`,
`TODO`, `raise NotImplementedError`, or equivalent is never a valid
deliverable."

This is an **absence** check, and an absence check that has never been shown to
fire proves nothing at all — it passes identically on a correct tree and on a
tree the checker cannot read. Every scan here is therefore paired with a
**planted positive control**: the same checker, run over a fixture that contains
exactly the construct being forbidden, must report it. The controls are what
make a green result evidence rather than an assumption.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path
from typing import List, Tuple

import pytest

REPO = Path(__file__).resolve().parents[2]
ROOTS = ("soma", "scripts")
MARKERS = ("TODO", "FIXME", "XXX:", "HACK", "placeholder implementation")


def _sources() -> List[Path]:
    out: List[Path] = []
    for root in ROOTS:
        out.extend(sorted((REPO / root).rglob("*.py")))
    return [p for p in out if "__pycache__" not in p.parts]


def _is_protocol(node: ast.ClassDef) -> bool:
    """A `typing.Protocol` body legitimately consists of `...`."""
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        if name == "Protocol":
            return True
    return False


def find_stubs(source: str, filename: str = "<test>") -> List[Tuple[int, str]]:
    """Every hollow function body in `source`, as `(lineno, reason)`.

    A body is hollow when, after any leading docstring is removed, it consists
    solely of `pass`, `...`, or `raise NotImplementedError`. Functions defined
    inside a `typing.Protocol` are exempt: `...` there **is** the declaration,
    not a missing one.
    """
    tree = ast.parse(source, filename=filename)
    protocol_bodies = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and _is_protocol(node)
        for child in node.body
    }
    found: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if id(node) in protocol_bodies:
            continue
        body = list(node.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(
            body[0].value, ast.Constant
        ) and isinstance(body[0].value.value, str):
            body = body[1:]
        if not body:
            found.append((node.lineno, f"{node.name}: docstring-only body"))
            continue
        if len(body) != 1:
            continue
        stmt = body[0]
        if isinstance(stmt, ast.Pass):
            found.append((node.lineno, f"{node.name}: body is `pass`"))
        elif (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        ):
            found.append((node.lineno, f"{node.name}: body is `...`"))
        elif isinstance(stmt, ast.Raise):
            exc = stmt.exc
            name = ""
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                name = exc.func.id
            elif isinstance(exc, ast.Name):
                name = exc.id
            if name == "NotImplementedError":
                found.append((node.lineno, f"{node.name}: raises NotImplementedError"))
    return found


def find_markers(source: str) -> List[Tuple[int, str]]:
    """Deferral markers in comments or strings, as `(lineno, text)`.

    Matched on the raw text rather than on comments alone, because a `TODO` in a
    docstring defers exactly as much work as one in a comment.
    """
    out: List[Tuple[int, str]] = []
    for i, line in enumerate(source.splitlines(), start=1):
        for m in MARKERS:
            if m in line:
                out.append((i, line.strip()[:120]))
                break
    return out


# ---------------------------------------------------------------------------
# Positive controls — run first, because they license everything below
# ---------------------------------------------------------------------------

_PLANTED = '''
"""A fixture that is deliberately defective."""

from typing import Protocol


class Legit(Protocol):
    def scores(self, x: int) -> float:
        ...


def hollow_pass(a, b):
    pass


def hollow_ellipsis(a):
    ...


def hollow_raise(a):
    """Has a docstring, and nothing else."""
    raise NotImplementedError


def hollow_docstring_only(a):
    """Nothing but this."""


def fine(a):
    return a + 1
'''


def test_the_stub_checker_detects_every_planted_stub() -> None:
    hits = {name.split(":")[0] for _, name in find_stubs(_PLANTED)}
    assert hits == {
        "hollow_pass",
        "hollow_ellipsis",
        "hollow_raise",
        "hollow_docstring_only",
    }, f"checker missed or over-reported: {sorted(hits)}"


def test_the_stub_checker_exempts_a_protocol_body() -> None:
    """`scores` inside a `Protocol` is a declaration, not a stub."""
    assert not any(n.startswith("scores") for _, n in find_stubs(_PLANTED))


def test_the_marker_checker_detects_a_planted_marker() -> None:
    planted = "x = 1  # TODO: finish this\ny = 2\n"
    hits = find_markers(planted)
    assert len(hits) == 1 and hits[0][0] == 1


# ---------------------------------------------------------------------------
# The absence claim itself
# ---------------------------------------------------------------------------


def test_no_stub_bodies_anywhere_in_the_package() -> None:
    offenders = []
    for path in _sources():
        src = io.open(path, encoding="utf-8").read()
        for lineno, reason in find_stubs(src, str(path)):
            offenders.append(f"{path.relative_to(REPO)}:{lineno} {reason}")
    assert not offenders, "stub bodies found:\n  " + "\n  ".join(offenders)


def test_no_deferral_markers_anywhere_in_the_package() -> None:
    offenders = []
    for path in _sources():
        src = io.open(path, encoding="utf-8").read()
        for lineno, text in find_markers(src):
            offenders.append(f"{path.relative_to(REPO)}:{lineno} {text}")
    assert not offenders, "deferral markers found:\n  " + "\n  ".join(offenders)


def test_the_scan_actually_reaches_the_package() -> None:
    """Without this the two assertions above pass on an empty file list."""
    files = _sources()
    assert len(files) >= 50, f"only {len(files)} source files scanned"
    assert any(p.name == "train.py" for p in files)
    assert any(p.name == "mixer_chunk.py" for p in files)
