"""The gate oracle's own control (`G0`).

Every other gate in `GATES.md` is an assertion of the form "this command exited
zero and printed `GATE_OK <name>`". That assertion is worth exactly as much as
the oracle's inability to print the token for a command that failed. This module
is the positive control for the oracle itself: if any test here fails, no result
in the ledger can be trusted, including the ones that passed.

The checks are deliberately behavioural — they run `scripts/gate.py` as a
subprocess and read its output — rather than importing `main()` and asserting on
a return code, because the ledger reads stdout, not return codes, and a defect
that prints the token while returning non-zero would be invisible to the
in-process form.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "scripts" / "gate.py"


def _run(*argv: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE), *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO),
    )


def test_a_succeeding_command_emits_the_token() -> None:
    r = _run("control_pass", "--", sys.executable, "-c", "raise SystemExit(0)")
    assert r.returncode == 0
    assert "GATE_OK control_pass" in r.stdout


@pytest.mark.parametrize("code", [1, 2, 7, 127])
def test_a_failing_command_can_never_emit_the_token(code: int) -> None:
    """The single property the whole ledger rests on."""
    r = _run("control_fail", "--", sys.executable, "-c", f"raise SystemExit({code})")
    assert r.returncode != 0
    assert "GATE_OK" not in r.stdout + r.stderr
    assert "GATE_FAIL control_fail" in r.stderr


def test_a_command_that_prints_the_token_itself_still_fails() -> None:
    """A command may print anything, including a forged success token.

    The oracle's contract is that *it* prints the token after reaping a zero
    exit. A child that prints `GATE_OK` and then fails must not produce a gate
    that a reader would count as met — so the failure token must also be
    present, and the exit status must be non-zero. This is the adversarial case:
    the ledger's reader sees combined output.
    """
    r = _run(
        "control_forged", "--", sys.executable, "-c",
        "print('GATE_OK control_forged'); raise SystemExit(3)",
    )
    assert r.returncode != 0
    assert "GATE_FAIL control_forged" in r.stderr


def test_a_missing_separator_fails_rather_than_succeeding_vacuously() -> None:
    r = _run("control_nosep")
    assert r.returncode != 0
    assert "GATE_OK" not in r.stdout


def test_an_empty_command_fails() -> None:
    r = _run("control_empty", "--")
    assert r.returncode != 0
    assert "GATE_OK" not in r.stdout


def test_a_timeout_fails_and_does_not_emit_the_token() -> None:
    r = _run(
        "control_timeout", "--timeout", "1", "--", sys.executable, "-c",
        "import time; time.sleep(30)",
        timeout=60.0,
    )
    assert r.returncode != 0
    assert "GATE_OK" not in r.stdout
    assert "timed out" in r.stderr


def test_a_missing_executable_fails() -> None:
    r = _run("control_missing", "--", "definitely-not-a-real-program-9f3a")
    assert r.returncode != 0
    assert "GATE_OK" not in r.stdout
