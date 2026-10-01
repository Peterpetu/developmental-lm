#!/usr/bin/env python
"""Precompute `L_ref` from the frozen `theta_m0`, for §06.2's token selection.

SPEC: 06 §6.1.1
SPEC: 06 §6.2

    python scripts/build_ref_cache.py --data data/ts_4096 \
        --m0-report runs/m0-m-nano-s20260828/m0_report.json

**What this produces and why it must exist.** §06.2 selects the top `k_tok`
fraction of natural tokens by **excess loss** `Delta_i = L_theta(x_i) -
L_ref(x_i)`, where `L_ref` comes from `theta_m0` — a *frozen, better* model.
That reference is the whole content of the mechanism: without it the criterion
degenerates to raw loss, which selects whatever the model currently finds hard,
including the tokens that are hard because they are unpredictable. The two
criteria are different mechanisms, and only one of them is §06.2.

`soma/train.py` accepts a `ReferenceCache` and falls back to raw loss when none
is supplied. The fallback is correct behaviour for M0's own run and for a smoke
test; it is **not** what an M1 run should be doing, and nothing in the pipeline
built the cache. This script closes that gap (audit A-72).

**Why precomputed.** `theta_m0` is frozen and the corpus is fixed, so `L_ref` is
a function of `(doc_id, position_in_doc)` alone. Computing it at training time
would be an extra full forward on every batch — `+25-33%` of step time, which
spec v1 mis-stated as "under 3%". Precomputing costs one pass over the corpus
and makes the training-time cost exactly zero.

**Why `(doc_id, position_in_doc)` and not stream position.** Attention is masked
at document boundaries (§05.4), so a token's conditioning depends only on its
own document and not on what was packed beside it. `corpus.windows(T_ctx)` *is*
the addressing scheme and §07.3's jitter permutes its rows and nothing else — so
the cached value survives every shuffle. Keyed by stream position it would not,
and every recorded hash would still match.

The sidecar is `float16`, 2 bytes per token: 944 MB for the 472M-token M-nano
corpus, 1.6 GB at M1. It is memory-mapped and read by index, never resident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from soma.config import get_config, m0_config  # noqa: E402
from soma.data import TokenizedCorpus  # noqa: E402
from soma.model.model import SomaModel  # noqa: E402
from soma.openended.references import RefCacheBinding, ReferenceCache  # noqa: E402


def _checkpoint_id(path: Path) -> str:
    """A content hash of the checkpoint, so the binding names a *file*.

    §06.1.1's binding includes `m0_checkpoint_id`. A run id would not do: two M0
    runs with the same id and different weights would silently share a cache.
    """
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--m0-report", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="overrides the path in the M0 report")
    ap.add_argument("--out", default=None,
                    help="sidecar path; defaults to <data>/<split>.lref.f16")
    ap.add_argument("--rows", type=int, default=256, help="windows per forward")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-windows", type=int, default=None,
                    help="debug only: stop after this many windows, leaving the "
                         "remainder of the sidecar at its fill value")
    args = ap.parse_args()

    report = json.loads(Path(args.m0_report).read_text(encoding="utf-8"))
    ckpt = Path(args.checkpoint or report["theta_m0_checkpoint"])
    if not ckpt.exists():
        print(f"no theta_m0 checkpoint at {ckpt}. §07.4.2 makes it one of M0's "
              f"three artifacts; run scripts/run_m0.py first.", file=sys.stderr)
        return 2

    cfg = m0_config(get_config(report["config"]))
    corpus = TokenizedCorpus(args.data, args.split)

    if corpus.index.tokenizer_hash != report["tokenizer_hash"]:
        print(f"tokenizer mismatch: corpus {corpus.index.tokenizer_hash} vs M0 "
              f"{report['tokenizer_hash']}. §06.1.1 refuses a cache computed "
              f"under a different tokenizer.", file=sys.stderr)
        return 2
    if corpus.index.vocab_size != cfg.V:
        print(f"vocabulary mismatch: {corpus.index.vocab_size} vs {cfg.V}",
              file=sys.stderr)
        return 2

    blob = torch.load(ckpt, map_location=args.device, weights_only=False)
    model = SomaModel(cfg, dense_sites=True).to(args.device)
    model.load_state_dict(blob["model"])
    model.eval()
    genome = blob.get("genome")

    binding = RefCacheBinding(
        tokenizer_hash=corpus.index.tokenizer_hash,
        corpus_hash=corpus.index.corpus_hash,
        m0_checkpoint_id=_checkpoint_id(ckpt),
        T_ctx=cfg.T_ctx,
    )
    out_path = Path(args.out or (Path(args.data) / f"{args.split}.lref.f16"))
    cache = ReferenceCache(out_path, binding)

    n_tokens = int(corpus.index.n_tokens)
    # `0.0` is the fill value for positions that carry no target: the final
    # token of every document, and any padding. Those positions are never
    # `scored`, so the value is never read — but a fill of `nan` would poison
    # `Delta_i` if that invariant were ever broken, and a fill of `inf` would
    # make them always selected. Zero fails safe in the direction of "no excess".
    values = np.zeros(n_tokens, dtype=np.float16)

    windows = corpus.windows(cfg.T_ctx)
    if args.limit_windows:
        windows = windows[: args.limit_windows]
    n_win = windows.shape[0]

    print(f"theta_m0 = {ckpt}")
    print(f"  binding {binding.digest()}")
    print(f"  {n_win:,} windows over {n_tokens:,} tokens, T_ctx = {cfg.T_ctx}")

    t0 = time.time()
    covered = 0
    for start in range(0, n_win, args.rows):
        w = windows[start : start + args.rows]
        toks, _docs, valid, flat = corpus.gather(w, cfg.T_ctx)

        # The same target/scored construction the training stream uses
        # (`soma/data/stream.py`): position `t` predicts token `t + 1`, the last
        # position of a row has no target, and a position is scored only if both
        # it and its target are real tokens. Any divergence here would produce a
        # cache whose values are aligned to different tokens than the ones the
        # training loop looks up — which nothing downstream could detect.
        targets = np.zeros_like(toks)
        targets[:, :-1] = toks[:, 1:]
        scored = valid.copy()
        scored[:, -1] = False
        sv = np.zeros_like(valid)
        sv[:, :-1] = valid[:, 1:]
        scored &= sv

        x = torch.from_numpy(toks).to(args.device)
        y = torch.from_numpy(targets).to(args.device)
        v = torch.from_numpy(valid).to(args.device)
        d = torch.from_numpy(_docs).to(args.device)

        logits = model(x, genome=genome, doc_id=d, valid=v,
                       want_metabolic=False).logits.float()
        logZ = torch.logsumexp(logits, dim=-1)
        gathered = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        loss = (logZ - gathered).cpu().numpy()

        m = scored & (flat >= 0)
        values[flat[m]] = loss[m].astype(np.float16)
        covered += int(m.sum())

        if start % (args.rows * 200) == 0:
            done = start + w.shape[0]
            rate = covered / max(time.time() - t0, 1e-9)
            print(f"  {done:,}/{n_win:,} windows  {covered:,} tokens  "
                  f"{rate:,.0f} tok/s", flush=True)

    cache.save(values)
    wall = time.time() - t0
    frac = covered / max(1, n_tokens)
    print(f"\n  wrote {out_path} ({values.nbytes / 2**20:.0f} MiB)")
    print(f"  covered {covered:,} / {n_tokens:,} tokens ({frac:.4f})")
    print(f"  {wall / 60:.1f} min at {covered / max(wall, 1e-9):,.0f} tok/s")
    print(f"  mean L_ref over covered tokens: "
          f"{float(values[values != 0].astype(np.float32).mean()):.4f} nats")

    # Coverage is a correctness statement, not a progress bar, and the expected
    # shortfall is exactly **one token per window** — not per document.
    #
    # `windows(T_ctx)` tiles each document contiguously with stride `T_ctx`, and
    # within a window position `t` predicts `t + 1`, so the window's final token
    # has no target *in that window* and is not the non-final position of any
    # other. It is therefore never scored — by this pass **or by the training
    # stream**, which builds `targets` and `scored` the same way from the same
    # window list. The cache is complete with respect to every lookup that can
    # ever happen; the uncovered ~`1/T_ctx` of tokens is a property of the
    # windowed packing, not of this script. Measured: 4,805,032 uncovered
    # against 4,805,032 windows, exactly.
    expected_gap = n_win
    gap = n_tokens - covered
    print(f"  uncovered {gap:,} against {n_win:,} windows "
          f"(one final token per window, ~1/T_ctx of the corpus)")
    if args.limit_windows is None and gap > expected_gap * 1.5:
        print(f"\nWARNING: {gap:,} tokens uncovered, expected about "
              f"{expected_gap:,} (one per window). The cache is written, but "
              f"the window enumeration and the training stream may disagree.",
              file=sys.stderr)

    corpus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
