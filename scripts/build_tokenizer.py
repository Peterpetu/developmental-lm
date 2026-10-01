#!/usr/bin/env python
"""Train the project tokenizer with the reserved control block.

SPEC: 06 §6.3.2
SPEC: 07 §7.4.2

    python scripts/build_tokenizer.py --vocab 8192 --out data/tokenizer_8192.json

**[NORMATIVE]** Token ids `0..63` are a reserved block that the tokenizer never
merges and never emits for natural text. The script verifies that on a corpus
sample before writing anything: a tokenizer that violates it would silently
change the structure of PARITY and DYCK and invalidate both the
length-generalisation claims of §08.2 and the expressivity argument of §05.2.2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from soma.data.tokenizer import (  # noqa: E402
    assert_reserved_block_unused,
    build_tokenizer,
)
from scripts._corpus_source import iter_corpus, corpus_choices  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="tinystories", choices=corpus_choices())
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max-docs", type=int, default=200_000,
                    help="documents used to fit the merges; the full corpus is "
                         "tokenised later by prepare_data.py")
    ap.add_argument("--cache", type=str, default=str(REPO / "data" / "raw"))
    ap.add_argument("--local-path", type=str, default=None,
                    help="required with --dataset local")
    args = ap.parse_args()

    print(f"fitting a {args.vocab}-token BPE on up to {args.max_docs:,} documents")
    docs = list(
        iter_corpus(args.dataset, args.split, args.cache, args.max_docs,
                    args.local_path)
    )
    if not docs:
        print("no documents found", file=sys.stderr)
        return 2
    print(f"  {len(docs):,} documents, {sum(len(d) for d in docs):,} characters")

    tok = build_tokenizer(docs, args.vocab)
    assert_reserved_block_unused(tok, docs[: min(2000, len(docs))])

    out = Path(args.out)
    tok.save(out)
    meta = {
        "vocab_size": tok.vocab_size,
        "tokenizer_hash": tok.hash(),
        "dataset": args.dataset,
        "split": args.split,
        "n_docs_fitted": len(docs),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  vocab {tok.vocab_size}  hash {tok.hash()}")
    print(f"  reserved block 0..63 verified unused on {min(2000, len(docs)):,} samples")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
