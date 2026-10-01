"""R0 SPAN: train the span codec on TinyStories, freeze it, measure its ceiling and noise
tolerance.

SPEC: 02 §2.2  (the codec, the loss (2.5), the ceiling)
SPEC: 10 §10.3 R0 (50M tokens, AdamW 3e-4; artifacts codec.pt, ceiling, I1/I2)

    python scripts/tempo_codec.py --config ts --tokens 50000000 --seed 0

Writes `runs/tempo/codec-<config>-s<seed>/{codec.pt, result.json, manifest.json, log.txt}`.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from tempo_common import REPO, config_by_name, eval_windows, load_corpus, log_to, manifest, run_dir, write_json
from tempo.codec import SpanCodec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ts")
    ap.add_argument("--tokens", type=int, default=50_000_000)
    ap.add_argument("--batch", type=int, default=4096, help="spans per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-eval", type=int, default=256)
    args = ap.parse_args()

    cfg = config_by_name(args.config)
    out = run_dir(f"codec-{args.config}-s{args.seed}")
    log = log_to(out / "log.txt")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    corpus = load_corpus()
    train = np.asarray(corpus.train.tokens)
    n_tok = int(train.shape[0])
    ev = eval_windows(corpus, cfg, n=args.n_eval, split="test")            # (n, T + K)
    ev_spans = torch.from_numpy(ev.reshape(-1, cfg.K)).to(args.device)
    # K = 1 is the token head of 02 §2.2 (no decoder layers); K >= 2 the two-layer within-span decoder
    n_dec_layers = 0 if cfg.K == 1 else None
    codec = SpanCodec(cfg.vocab, cfg.K, cfg.d_x, d_c=cfg.d_c, n_dec_layers=n_dec_layers).to(args.device)
    steps = max(1, args.tokens // (args.batch * cfg.K))
    opt = torch.optim.AdamW(codec.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    warm = max(1, steps // 50)
    log(f"codec {codec.n_params():,} params; {steps} steps of {args.batch} spans; device {args.device}")
    t0 = time.time()
    for step in range(steps):
        lr = args.lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        offs = torch.randint(0, n_tok - cfg.K, (args.batch,), generator=gen)
        idx = offs[:, None] + torch.arange(cfg.K)[None]
        spans = torch.from_numpy(train[idx.numpy()].astype(np.int64)).to(args.device)
        with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
            res = codec.loss(spans, generator=None)
        opt.zero_grad(set_to_none=True)
        res["loss"].backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            log(f"step {step:6d}  loss {float(res['loss']):.4f}  nll/token {float(res['nll_token']):.4f}  "
                f"kl {float(res['kl']):.2f}  lr {lr:.2e}  {time.time() - t0:.0f}s")
    codec.eval()
    with torch.no_grad():
        ceilings = []
        gaps = []
        for a in range(0, ev_spans.shape[0], 8192):
            s = ev_spans[a:a + 8192]
            ceilings.append(float(codec.ceiling(s)) * s.shape[0])
            x_bar, _, _ = codec.encode(s)
            clean, _ = codec.nll(x_bar, s)
            eps = torch.randn(x_bar.shape, generator=gen).to(x_bar.device)
            eps = eps / eps.norm(dim=-1, keepdim=True) * (codec.sigma_z * math.sqrt(codec.d_x))
            noisy, _ = codec.nll(x_bar + eps, s)
            gaps.append(float((noisy - clean).sum()) / cfg.K)
        ceiling = sum(ceilings) / ev_spans.shape[0]
        gap = sum(gaps) / ev_spans.shape[0]
    result = {"complete": True, "ceiling_nll_per_token": ceiling, "noise_gap_per_token": gap,
              "I1": ceiling < 0.01, "I2": gap < 0.05, "steps": steps, "batch_spans": args.batch,
              "tokens": steps * args.batch * cfg.K, "n_params": codec.n_params(),
              "seconds": time.time() - t0, "n_eval_spans": int(ev_spans.shape[0])}
    torch.save({"state": codec.state_dict(), "vocab": cfg.vocab, "K": cfg.K, "d_x": cfg.d_x, "d_c": cfg.d_c,
                "n_dec_layers": codec.n_dec_layers, "result": result}, out / "codec.pt")
    write_json(out / "result.json", result)
    write_json(out / "manifest.json", manifest({"args": vars(args), "config": cfg.as_dict(), "run": "R0"}))
    log(f"ceiling {ceiling:.5f} nats/token (I1 {result['I1']}); noise gap {gap:.5f} (I2 {result['I2']})")


if __name__ == "__main__":
    main()
