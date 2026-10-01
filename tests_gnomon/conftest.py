"""Shared fixtures. SPEC: 05 §5.6.

The tokenised corpora are 6.4 GB and are deliberately not in the repository:
they are reproducible from `scripts/build_tokenizer.py` and
`scripts/prepare_data.py`, and `.gitignore` excludes them by design. A handful
of tests read them anyway, because what they assert *is* a property of the
corpus -- L9's split disjointness and S6's split hygiene cannot be checked
against a synthetic shard, since the thing being checked is how the real
holdout was divided.

Those tests **skip** when the corpus is absent rather than failing. A clean
clone that reports two red tests teaches a reader that red is normal here,
which is the habit that lets a real failure through; a skip with the command
that would fix it teaches them what to run.
"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def corpus_available(vocab: int = 4096) -> bool:
    root = REPO / "data" / f"ts_{vocab}"
    return (root / "validation.index.json").exists() and (
        root / "train.index.json").exists()


def requires_corpus(vocab: int = 4096):
    """Skip marker for a test that needs the real tokenised shards."""
    return pytest.mark.skipif(
        not corpus_available(vocab),
        reason=(f"data/ts_{vocab} is absent -- it is 6.4 GB, gitignored by "
                f"design, and rebuilt with `python scripts/build_tokenizer.py` "
                f"then `python scripts/prepare_data.py`"),
    )
