#!/usr/bin/env python
"""Tokenise a corpus into the memmapped stream with a document index.

SPEC: 06 §6.1.1
SPEC: 07 §7.4.2
SPEC: 08 §8.1

    python scripts/prepare_data.py --tokenizer data/tokenizer_8192.json \\
        --dataset tinystories --out data/tinystories_8192

Produces `<split>.tokens.u16`, `<split>.starts.i64`, `<split>.lengths.i64` and
`<split>.index.json` per split. Every document is stored contiguously with its
own `doc_id`, and windows are addressed by `(doc_id, position_in_doc)` — which
is what makes the `L_ref` cache invariant to §07.3's data-order jitter
(§06.1.1).

**Holdout.** §08.1 requires every reported curve to be held-out natural text. If
the source has no validation split, one is carved from the **head** of train and
those documents are **excluded** from train by index, so the two are disjoint by
construction rather than by assumption.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from soma.data.corpus import tokenize_corpus  # noqa: E402
from soma.data.tokenizer import (  # noqa: E402
    assert_reserved_block_unused,
    load_tokenizer,
)
from scripts._corpus_source import corpus_choices, iter_corpus  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="tinystories", choices=corpus_choices())
    ap.add_argument("--local-path", default=None)
    ap.add_argument("--cache", default=str(REPO / "data" / "raw"))
    ap.add_argument("--max-docs", type=int, default=None,
                    help="cap on TRAIN documents, after the holdout is removed")
    ap.add_argument("--holdout-docs", type=int, default=20_000)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"tokenizer vocab {tok.vocab_size}  hash {tok.hash()}")

    def source() -> Iterator[str]:
        return iter_corpus(args.dataset, "train", args.cache, None, args.local_path)

    # ---- validation: the first `holdout_docs` documents, excluded from train --
    t0 = time.time()
    print(f"holdout: first {args.holdout_docs:,} documents of train")
    held: List[str] = []
    for i, doc in enumerate(source()):
        if i >= args.holdout_docs:
            break
        held.append(doc)
    if not held:
        print("no documents read", file=sys.stderr)
        return 2

    sample: List[str] = []
    idx_val = tokenize_corpus(
        _sampling(iter(held), sample), tok, out, "validation",
        name=f"{args.dataset}:holdout", progress_every=0,
    )
    assert_reserved_block_unused(tok, sample[:500])
    print(f"  validation {idx_val.as_dict()}  in {time.time() - t0:.0f}s")

    # ---- train: everything after the holdout ---------------------------------
    t0 = time.time()
    print("train: documents after the holdout")
    idx_tr = tokenize_corpus(
        _skip(source(), args.holdout_docs, args.max_docs), tok, out, "train",
        name=f"{args.dataset}:train", progress_every=250_000,
    )
    print(f"  train {idx_tr.as_dict()}  in {time.time() - t0:.0f}s")

    (out / "prepare.json").write_text(
        json.dumps(
            {
                "tokenizer": str(args.tokenizer),
                "tokenizer_hash": tok.hash(),
                "dataset": args.dataset,
                "holdout_docs": args.holdout_docs,
                "holdout_rule": "train[:holdout_docs]; excluded from train by index",
                "train": idx_tr.as_dict(),
                "validation": idx_val.as_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


def _skip(stream: Iterable[str], n_skip: int, max_docs: Optional[int]) -> Iterator[str]:
    kept = 0
    for i, doc in enumerate(stream):
        if i < n_skip:
            continue
        yield doc
        kept += 1
        if max_docs is not None and kept >= max_docs:
            return


def _sampling(stream: Iterable[str], sink: List[str], n: int = 500) -> Iterator[str]:
    for doc in stream:
        if len(sink) < n:
            sink.append(doc)
        yield doc


if __name__ == "__main__":
    raise SystemExit(main())
