#!/usr/bin/env python
"""Build the second environment of PALIMPSEST: Python source from the local interpreter.

SPEC: docs/palimpsest/01-model.md §1.1 (environments)

    python scripts/palimpsest_build_code.py --out data/py_8192 --max-chars 150000000

Every `.py` file under the interpreter's `Lib` and `site-packages` is one document (a
distribution shift of the *language*, not of the tokenizer: the story tokenizer is
byte-level and round-trips source exactly, at 0.51 tokens per character against ~0.25
on prose). Files are de-duplicated by content hash, files above `--max-file` bytes are
skipped (generated tables, vendored data), the order is a seeded shuffle so the holdout
is not the alphabetical head, and the first `--holdout-docs` documents form the
validation split, excluded from train by index -- prepare_data.py's rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gnomon.data.corpus import tokenize_corpus  # noqa: E402
from gnomon.data.tokenizer import load_tokenizer  # noqa: E402


def collect(roots, max_file: int, seed: int) -> list[Path]:
    seen, files = set(), []
    for root in roots:
        for dp, _, fn in os.walk(root):
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = Path(dp) / f
                try:
                    s = p.stat().st_size
                except OSError:
                    continue
                if s < 200 or s > max_file:
                    continue
                files.append(p)
    rng = random.Random(seed)
    rng.shuffle(files)
    out = []
    for p in files:
        try:
            b = p.read_bytes()
        except OSError:
            continue
        h = hashlib.blake2b(b, digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(REPO / "data" / "tokenizer_8192.json"))
    ap.add_argument("--out", default=str(REPO / "data" / "py_8192"))
    ap.add_argument("--max-chars", type=int, default=150_000_000)
    ap.add_argument("--max-file", type=int, default=200_000)
    ap.add_argument("--holdout-docs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()
    lib = Path(os.__file__).parent
    roots = [lib / "site-packages", lib]
    t0 = time.time()
    files = collect(roots, args.max_file, args.seed)
    print(f"{len(files)} unique .py files under {roots} in {time.time() - t0:.0f}s")
    tok = load_tokenizer(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def read(p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    held = [read(p) for p in files[: args.holdout_docs]]
    held = [t for t in held if t]
    idx_val = tokenize_corpus(iter(held), tok, out, "validation", name="python:holdout", progress_every=0)
    print(f"validation {idx_val.as_dict()}")

    def train_docs():
        chars = 0
        for p in files[args.holdout_docs:]:
            t = read(p)
            if not t:
                continue
            chars += len(t)
            yield t
            if chars >= args.max_chars:
                return

    idx_tr = tokenize_corpus(train_docs(), tok, out, "train", name="python:train", progress_every=5000)
    print(f"train {idx_tr.as_dict()}")
    (out / "prepare.json").write_text(json.dumps({
        "tokenizer": args.tokenizer, "tokenizer_hash": tok.hash(), "dataset": "python-local",
        "roots": [str(r) for r in roots], "n_files": len(files), "holdout_docs": args.holdout_docs,
        "max_chars": args.max_chars, "max_file": args.max_file, "seed": args.seed,
        "train": idx_tr.as_dict(), "validation": idx_val.as_dict(),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
