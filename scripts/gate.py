#!/usr/bin/env python
"""Gate oracle for the SOMA completion ledger.

Runs a command and prints ``GATE_OK <name>`` **only** when it exits zero. Any
non-zero exit, timeout, or exception prints ``GATE_FAIL <name>`` and exits 1, so
the success token can never be produced by a failing check.

The token is emitted *after* the child process has been reaped and its exit
status confirmed, which is what makes it a success-only marker in the sense the
ledger requires.

Usage
-----
    python scripts/gate.py <name> -- <command> [args ...]

Example
-------
    python scripts/gate.py mixer_equivalence -- python -m pytest tests/unit/test_mixer_equivalence.py -q
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

TOKEN_OK = "GATE_OK"
TOKEN_FAIL = "GATE_FAIL"


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``argv`` at the first bare ``--``.

    Returns (before, after). Raises SystemExit if the separator is absent, since
    a gate without a command would otherwise silently succeed.
    """
    if "--" not in argv:
        print(
            f"{TOKEN_FAIL} <unnamed>: no '--' separator; "
            "usage: gate.py <name> -- <command>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1 :]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    before, command = _split_argv(argv)

    parser = argparse.ArgumentParser(prog="gate.py", add_help=True)
    parser.add_argument("name", help="gate identifier, echoed in the success token")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="seconds before the command is killed and the gate fails",
    )
    args = parser.parse_args(before)

    if not command:
        print(f"{TOKEN_FAIL} {args.name}: empty command", file=sys.stderr)
        return 2

    start = time.monotonic()
    try:
        proc = subprocess.run(command, timeout=args.timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        print(
            f"{TOKEN_FAIL} {args.name}: timed out after {elapsed:.1f}s",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError as exc:
        print(f"{TOKEN_FAIL} {args.name}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"{TOKEN_FAIL} {args.name}: interrupted", file=sys.stderr)
        return 130

    elapsed = time.monotonic() - start
    if rc == 0:
        # Emitted only after a confirmed zero exit status.
        print(f"{TOKEN_OK} {args.name} ({elapsed:.1f}s)")
        return 0

    print(f"{TOKEN_FAIL} {args.name}: exit {rc} after {elapsed:.1f}s", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
