#!/usr/bin/env python
"""Evaluate a finished run: held-out CE/BPB, the behavioural suite, `Q_route`,
`erank`, generation samples, and the kill criteria that a single run can decide.

SPEC: 08 §8.1
SPEC: 08 §8.2
SPEC: 08 §8.3
SPEC: 10 §10.1

    python scripts/eval_tinystories.py --run runs/m1-s20260828 --data data/ts_8192

**[NORMATIVE — §08.7's reporting discipline]** applies to everything printed
here: `pass_at_k` at large `k` and never `pass@1` alone; no task-identity
conditioning; measured wall-clock and memory with any efficiency claim;
components whose gain shrinks with compute flagged as suspect; negative results
at equal prominence; verification status carried; **inconclusive is a reportable
outcome**.

Generation quality (§08.3) is graded against a **frozen** versioned artifact —
60 prompts, a 5-point rubric with written anchors, fixed decoding, a pinned
grader, and a Krippendorff `alpha >= 0.6` agreement check before the rubric may
be used. This script emits the samples in that fixed format; it does not grade
them, because grading requires the pinned judge and §08.3 forbids changing the
grader between compared runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from soma.config import get_config  # noqa: E402
from soma.data import TokenizedCorpus, TrainingStream, load_tokenizer  # noqa: E402
from soma.eval.metrics import bits_per_byte, erank_probe, held_out_ce  # noqa: E402
from soma.eval.qroute import q_route  # noqa: E402
from soma.eval.regression import run_suite  # noqa: E402
from soma.germline import Genome  # noqa: E402
from soma.model.model import SomaModel  # noqa: E402
from soma.utils.logging import JsonlLogger  # noqa: E402
from soma.utils.stats import mann_kendall  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None, help="defaults to the run report's")
    ap.add_argument("--tokenizer", default=None, help="for decoded samples")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--suite-n-eval", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run = Path(args.run)
    report_path = run / "m1_report.json"
    meta = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    cfg_name = args.config or meta.get("config", "m1")
    cfg = get_config(cfg_name)
    out: dict = {"run": str(run), "config": cfg_name}

    train = TokenizedCorpus(args.data, "train")
    val = TokenizedCorpus(args.data, "validation")

    # ---- 1. the curve, from the run's own log (§08.1) ---------------------
    evals = JsonlLogger.read(run / "log.jsonl", "eval")
    if evals:
        final = evals[-1]
        out["held_out_ce_nats"] = final["ce"]
        out["held_out_ppl"] = float(np.exp(final["ce"]))
        n_tok = val.index.n_tokens
        n_bytes = _corpus_bytes(args.data, "validation", n_tok)
        out["bpb"] = bits_per_byte(final["ce"], n_tok, n_bytes)
        out["curve"] = [(e["step"], e["tokens"], e["ce"]) for e in evals]
        print(f"held-out CE {final['ce']:.4f} nats  PPL {out['held_out_ppl']:.3f}  "
              f"BPB {out['bpb']:.4f}")
    else:
        print("no eval records in the run log", file=sys.stderr)

    # ---- 2. structural annotations (§08.1) --------------------------------
    out["structural_events"] = meta.get("structural_events", [])
    print(f"structural events annotated on the curve: {len(out['structural_events'])}")

    # ---- 3. erank trend, K8 (§10.1) ---------------------------------------
    eranks = JsonlLogger.read(run / "log.jsonl", "erank")
    if len(eranks) >= 3:
        tail = [e["mean"] for e in eranks if e["step"] >= 0.5 * cfg.N_step]
        if len(tail) >= 3:
            mk = mann_kendall(tail, alternative="decreasing")
            fires = bool(mk.p_value < 0.05 and mk.slope < -1e-4)
            out["K8"] = {
                "p_value": mk.p_value, "slope": mk.slope, "method": mk.method,
                "n_points": len(tail), "fires": fires,
            }
            print(f"K8 erank trend: p={mk.p_value:.4f} slope={mk.slope:.3e} "
                  f"-> {'FIRES' if fires else 'does not fire'}")
        else:
            out["K8"] = {"skipped": f"only {len(tail)} points in the final 50%"}
    else:
        out["K8"] = {"skipped": f"only {len(eranks)} erank records"}

    # ---- 4. the model, for suite / Q_route / samples ----------------------
    ckpt = Path(args.checkpoint) if args.checkpoint else (run / "checkpoints" / "final.pt")
    model = genome = None
    if ckpt.exists():
        blob = torch.load(ckpt, map_location=args.device, weights_only=False)
        model = SomaModel(cfg).to(args.device)
        model.load_state_dict(blob["model"])
        genome = blob.get("genome")
        print(f"loaded checkpoint {ckpt}")
    else:
        print(f"no checkpoint at {ckpt}; suite, Q_route and samples are skipped")

    if model is not None and genome is not None:
        rng = np.random.default_rng(args.seed)

        # §08.2's behavioural suite -- scaled accuracy only.
        t0 = time.time()
        rows = run_suite(model, cfg, rng, genome=genome,
                         n_eval=args.suite_n_eval, device=args.device)
        out["suite"] = [r.as_dict() for r in rows]
        print(f"\nbehavioural suite ({time.time() - t0:.0f}s), scaled accuracy:")
        for r in rows:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.name:10s} n_len={str(r.n_len):>4s}  "
                  f"scaled={r.scaled:+.3f} (bar {r.pass_bar:.2f}, chance {r.acc_random:.3f})")

        # §08.1.1's Q_route on a fixed probe batch.
        probe = torch.from_numpy(
            train.gather(train.windows(cfg.T_ctx)[:16], cfg.T_ctx)[0]
        ).to(args.device)
        with torch.no_grad():
            fo = model(probe, genome=genome, want_metabolic=False)
        qr = q_route(
            [so.slots.reshape(-1, cfg.k_route) for so in fo.site_outputs],
            [so.r_ut.reshape(-1, cfg.k_route) for so in fo.site_outputs],
            [p.live for p in model.populations],
        )
        out["Q_route"] = qr
        print(f"\nQ_route (Barber bipartite modularity, BRIM): {qr['Q_route']:+.4f}")

        out["erank_final"] = erank_probe(model, probe, genome)
        print(f"erank per Site: {[round(v, 1) for v in out['erank_final']]}")

        # §08.3's samples, in the frozen decoding format.
        if args.tokenizer:
            tok = load_tokenizer(args.tokenizer)
            out["samples"] = _samples(model, genome, cfg, tok, args, rng)
            print(f"\n{len(out['samples'])} generation samples "
                  f"(temperature 0.8, top_p 0.95, max_new 200):")
            for s in out["samples"][:4]:
                print(f"  > {s['prompt']}\n    {s['completion'][:180]}")

    (run / "eval_report.json").write_text(json.dumps(out, indent=2, default=str),
                                          encoding="utf-8")
    print(f"\nwrote {run / 'eval_report.json'}")
    train.close()
    val.close()
    return 0


def _corpus_bytes(data: str, split: str, n_tokens: int) -> int:
    """Bytes of the held-out text, for `BPB`'s `tokens/bytes` ratio."""
    p = Path(data) / f"{split}.index.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    # Recorded at preparation time when available; otherwise estimated from the
    # measured ~4.2 characters per token of this tokenizer on this corpus, and
    # the estimate is reported as such.
    return int(meta.get("n_bytes", n_tokens * 4.2))


@torch.no_grad()
def _samples(model, genome, cfg, tok, args, rng) -> list:
    """§08.3's fixed decoding: temperature 0.8, top_p 0.95, max_new 200."""
    prompts = [
        "Once upon a time there was a little girl who",
        "Tom found a shiny red ball in the garden and",
        "The cat wanted to climb the tall tree but",
        "Lily and her brother went to the park to",
        "One day a small dog saw a big bone and",
        "The old man sat by the river and",
    ]
    model.eval()
    out = []
    for i in range(min(args.n_samples, len(prompts) * 2)):
        text = prompts[i % len(prompts)]
        ids = tok.encode(text)
        ctx = list(ids)
        g = torch.Generator(device=args.device).manual_seed(args.seed + i)
        for _ in range(200):
            x = torch.tensor([ctx[-cfg.T_ctx :]], dtype=torch.long, device=args.device)
            logits = model(x, genome=genome, want_metabolic=False).logits[0, -1].float()
            probs = torch.softmax(logits / 0.8, dim=-1)
            srt, idx = torch.sort(probs, descending=True)
            keep = (torch.cumsum(srt, 0) - srt) < 0.95
            srt = torch.where(keep, srt, torch.zeros_like(srt))
            srt = srt / srt.sum()
            nxt = int(idx[torch.multinomial(srt, 1, generator=g)])
            if nxt == 2:                    # EOS
                break
            ctx.append(nxt)
        out.append({
            "prompt": text,
            "completion": tok.decode(ctx[len(ids):]),
            "temperature": 0.8, "top_p": 0.95, "max_new": 200,
            "seed": args.seed + i,
        })
    model.train()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
