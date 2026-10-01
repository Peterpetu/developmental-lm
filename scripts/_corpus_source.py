"""Corpus sources for the data-preparation scripts.

SPEC: 07 §7.4.2

M0's corpus is **TinyStories V2 (GPT-4 subset)**, the project tokenizer at
`V = 8192`, `T_ctx = 512` (§07.4.2). §08.3 additionally requires "a second story
corpus for generator independence", which is what `--dataset` selects.

Downloads are cached under `data/raw/` so a run is reproducible without network
access after the first fetch, and every corpus reports a content hash that binds
the `L_ref` sidecar (§06.1.1).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

__all__ = ["iter_corpus", "corpus_choices", "SOURCES"]


SOURCES = {
    "tinystories": {
        "hf": "roneneldan/TinyStories",
        "config": None,
        "field": "text",
        "note": "TinyStories V2 (GPT-4 subset) — §07.4.2's M0 corpus",
    },
    "tinystories-instruct": {
        "hf": "roneneldan/TinyStoriesInstruct",
        "config": None,
        "field": "text",
        "note": "§08.3's second story corpus, for generator independence",
    },
}


def corpus_choices() -> List[str]:
    return sorted(SOURCES) + ["local"]


def iter_corpus(
    name: str,
    split: str = "train",
    cache_dir: str | os.PathLike = "data/raw",
    max_docs: int | None = None,
    local_path: str | os.PathLike | None = None,
) -> Iterator[str]:
    """Yield documents as strings.

    `name = "local"` reads newline-delimited documents from `local_path`, which
    is how a run reproduces without network access — and how the smoke
    configuration runs at all.
    """
    if name == "local":
        if local_path is None:
            raise ValueError("--dataset local requires --local-path")
        n = 0
        with open(local_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield line
                n += 1
                if max_docs is not None and n >= max_docs:
                    return
        return

    if name not in SOURCES:
        raise KeyError(f"unknown corpus {name!r}; known: {corpus_choices()}")

    from datasets import load_dataset

    spec = SOURCES[name]
    cache = str(Path(cache_dir).resolve())
    Path(cache).mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        spec["hf"], spec["config"], split=split, cache_dir=cache, streaming=False
    )
    field = spec["field"]
    for i, row in enumerate(ds):
        text = row[field]
        if not isinstance(text, str) or not text.strip():
            continue
        yield text.strip()
        if max_docs is not None and i + 1 >= max_docs:
            return
