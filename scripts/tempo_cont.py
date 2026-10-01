"""R2 CONT: the token backbone at matched training FLOPs, scored on the same tokens as the span agent.

SPEC: 10 §10.2 K-CONT, §10.3 R2
SPEC: 02 §2.4  (per-token comparability: `L_txt / K` on tokens `K .. T + K - 1`)
SPEC: 09 §9.1  (training FLOPs = 3 x forward FLOPs; the frozen encoder is not run at lambda_lat = 0)

The span agent (R1's arm `nolat` at the primordium) and ANLAGE's token `Phenotype` (with
its value embeddings, the strongest form on this card) are compared at *equal training
FLOPs*: the forward FLOPs per token of each are measured with `torch.utils.flop_counter`
on the training path (the span agent: the input embedding, the backbone at the primordium,
the decoder's likelihood; the token model: its full forward), the token model's width is the
candidate whose FLOPs per token are nearest the span agent's, and its token budget is then
scaled by the exact ratio so that the two spend the same training FLOPs. Both are scored on
the v3 test windows of `T + K` tokens at the targets `K .. T + K - 1`. The token model reads
a fresh pool (no window repeats within the budget), as the span agent's lanes do.

    python scripts/tempo_cont.py --tokens 25000000 --seed 0 [--config tsk2] [--width 320]

Writes `runs/tempo/R2-token-<config>-s<seed>/{result.json, manifest.json, model.pt, log.txt}`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from tempo_common import (REPO, array_digest, config_by_name, eval_windows, file_sha256, load_corpus, log_to,
                          manifest, run_dir, scripts_digest, write_json)
from anlage.config import StageConfig
from anlage.data import EvalSet
from anlage.lifetime import per_token_loss, train
from anlage.model import Phenotype, save_phenotype
from anlage.recipe import TrainConfig, micro_batch_for
from tempo.backbone import Backbone
from tempo.codec import SpanCodec
from tempo.modalities import TextIn
from tempo.morphology import SwitchPattern

CANDIDATE_WIDTHS = (192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 1024)
"""The widths the token baseline may take. It must be wide enough that the FLOP-nearest choice
is INTERIOR for every shape compared: the adult-shape diagnostic of 11 §P-D.17 needs
`1.494e8` FLOPs/token and the old list's last element gave `1.225e8`, a ratio of 0.820, so the
baseline's shape was chosen under a constraint rather than freely. The total-FLOP match is exact
either way (the token budget compensates), so no verdict was invalidated -- but a clipped choice
must be visible, which is what `width_at_boundary` in the result is for (11 §P-D.19)."""
TRAIN_FACTOR = 3.0
"""SPEC: 09 §9.1 -- training FLOPs are three times the forward FLOPs (forward + two backward products)."""


def span_agent_flops_per_token(cfg, codec_path: str, *, adult: bool = False) -> dict:
    """Forward FLOPs per token of the span agent's training path: the span embedding, the
    backbone, the decoder's likelihood over the next span (the frozen encoder is not run at
    lambda_lat = 0, 11 §P-C.23).

    `adult` selects the shape the backbone is measured in, and it must match the shape the span
    run actually used: the baseline's WIDTH is chosen to sit nearest this number, so a baseline
    sized against the primordium is not FLOP-matched to an adult span agent, and nothing in the
    record would say so (11 §P-D.17)."""
    bb = Backbone(cfg)
    text_in = TextIn(cfg.vocab, cfg.K, cfg.d_x)
    blob = torch.load(codec_path, map_location="cpu", weights_only=False)
    n_dec = blob.get("n_dec_layers", 0 if blob["K"] == 1 else None)
    codec = SpanCodec(blob["vocab"], blob["K"], blob["d_x"], d_c=blob["d_c"], n_dec_layers=n_dec)
    toks = torch.randint(0, cfg.vocab, (cfg.T_s + 1, cfg.K))
    with FlopCounterMode(display=False) as fc:
        x = text_in(toks[:-1])
    f_in = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        y, _, _, _ = bb(x[None], pattern=SwitchPattern.adult(cfg) if adult
                        else SwitchPattern.primordium(cfg))
    f_bb = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        codec.nll(y[0], toks[1:])
    f_dec = fc.get_total_flops()
    return {"text_in": f_in / cfg.T, "backbone": f_bb / cfg.T, "decoder": f_dec / cfg.T,
            "total": (f_in + f_bb + f_dec) / cfg.T}


def token_model_flops_per_token(scfg: StageConfig, vocab: int, T: int) -> float:
    m = Phenotype(scfg, vocab, tied=True)
    w = torch.randint(0, vocab, (1, T + 1))
    with FlopCounterMode(display=False) as fc:
        m.loss(w)
    return fc.get_total_flops() / T


def stage_config(width: int, layers: int, *, use_ve: bool = True) -> StageConfig:
    """`use_ve=False` drops the value embeddings, one of the two features the baseline has and
    the span agent does not (11 §P-D.20). It changes the per-token cost, so the FLOP matching
    and the width selection follow it automatically."""
    return StageConfig(stage=0, d_model=width, L=layers, n_h=max(1, width // 64),
                       n_kv_attn=(2 if (width // 64) % 2 == 0 else 1), d_ff=4 * width, d_head=64, chunk_size=64,
                       use_ve=use_ve)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ts")
    ap.add_argument("--run", default="R2", help="R2 (K-CONT at R1's budget) or R10 (K-SOTA at the adult budget)")
    ap.add_argument("--tokens", type=int, default=25_000_000, help="the span agent's token budget (R1)")
    ap.add_argument("--tag", default="",
                    help="a name segment between the arm and the config, as tempo_life.py has. "
                         "Without it the adult baseline would be written to "
                         "R2-token-<config>-s<seed> and OVERWRITE the primordium baseline "
                         "already there (11 §P-D.17).")
    ap.add_argument("--no-ve", action="store_true",
                    help="drop the baseline's value embeddings (11 §P-D.20)")
    ap.add_argument("--no-tied", action="store_true",
                    help="drop the baseline's tied input/output embeddings (11 §P-D.20)")
    ap.add_argument("--adult", action="store_true",
                    help="the span arm ran at the adult shape (morphology=flat), so size the "
                         "baseline against the adult's per-token cost (11 §P-D.17)")
    ap.add_argument("--span-record", default=None,
                    help="the span agent's record.json. When given, the baseline is matched to the "
                         "compute that agent ACTUALLY charged itself (its F_episode over its own token "
                         "count) instead of an analytic estimate over an assumed budget (11 §P-D.2).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=0, help="force the token model's width (else FLOP-nearest)")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--codec", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    cfg = config_by_name(args.config)
    out = run_dir(f"{args.run}-token{('-' + args.tag) if args.tag else ''}-{args.config}-s{args.seed}")
    log = log_to(out / "log.txt")
    codec_path = args.codec or str(REPO / "runs" / "tempo" / f"codec-{args.config}-s0" / "codec.pt")
    span = span_agent_flops_per_token(cfg, codec_path, adult=args.adult)
    log(f"span agent forward FLOPs/token at the {'adult' if args.adult else 'primordium'}: "
        f"text_in {span['text_in']:.3e} backbone {span['backbone']:.3e} "
        f"decoder {span['decoder']:.3e} total {span['total']:.3e}")
    use_ve = not args.no_ve
    choices = {d: token_model_flops_per_token(stage_config(d, args.layers, use_ve=use_ve), cfg.vocab, cfg.T)
               for d in CANDIDATE_WIDTHS}
    width = args.width or min(choices, key=lambda d: abs(math_log_ratio(choices[d], span["total"])))
    scfg = stage_config(width, args.layers, use_ve=use_ve)
    ratio = choices[width] / span["total"]
    # (a) the span side: prefer the compute the agent CHARGED ITSELF over an analytic estimate
    # of an assumed budget. The two differ by ~1 % at the ts shape, always in the baseline's
    # favour, and a K-CONT gap small enough to be moved by 1 % must not rest on the estimate.
    F_span_analytic = TRAIN_FACTOR * span["total"] * args.tokens
    tokens_span, F_span, span_basis = args.tokens, F_span_analytic, "analytic"
    if args.span_record:
        rec = json.loads(pathlib.Path(args.span_record).read_text(encoding="utf-8"))
        if not rec.get("complete"):
            raise SystemExit(f"--span-record {args.span_record} is not a completed run")
        tokens_span = int(sum(st.get("tokens", 0) for st in rec["steps"]))
        F_span = float(rec["episodes"][0]["F_episode"])
        span_basis = "charged"
        log(f"span agent charged {F_span:.6e} FLOPs over {tokens_span:,} tokens "
            f"({F_span / max(tokens_span, 1):.4e}/token); the analytic estimate was "
            f"{F_span_analytic:.6e} over {args.tokens:,} ({100 * (F_span_analytic / F_span - 1):+.2f} %)")
    tokens_token = int(round(F_span / (TRAIN_FACTOR * choices[width])))
    F_tok = TRAIN_FACTOR * choices[width] * tokens_token
    log(f"token model width {width} (L={args.layers}, "
        f"{'value embeddings' if use_ve else 'no value embeddings'}, "
        f"{'tied' if not args.no_tied else 'untied'}): {choices[width]:.3e} FLOPs/token, "
        f"ratio {ratio:.3f}; token budget {tokens_token:,} against the span agent's {tokens_span:,} "
        f"({span_basis} basis; training FLOPs {F_tok:.3e} vs {F_span:.3e}); candidates {choices}")
    torch.manual_seed(args.seed)
    model = Phenotype(scfg, cfg.vocab, tied=not args.no_tied, seed=args.seed).to(args.device)
    corpus = load_corpus()
    n_pool = max(4096, -(-tokens_token // cfg.T) + 64)                  # fresh windows: no repeats within the budget
    pool = corpus.pool(n_windows=n_pool, T=cfg.T, pool_seed=20260904 + args.seed, device=args.device)
    ev_train = EvalSet.sample_documents([corpus.fitness, corpus.test], corpus.rank, n_windows=64, T=cfg.T,
                                        split="fitness", seed=20260903, mod=3, residue=0, device=args.device)
    tc = TrainConfig(tokens=tokens_token, batch_final=args.batch, T=cfg.T,
                     micro_batch=micro_batch_for(model.n_params(), args.batch))
    t0 = time.time()
    res = train(model, pool, ev_train, tc, order_seed=args.seed, eval_every_tokens=max(1, tokens_token // 5),
                compile_mode=None, log=log, device=args.device)
    # the comparable score: targets K .. T + K - 1 of the T + K token windows
    ev_np = eval_windows(corpus, cfg, n=args.n_eval, split="test")
    ev = torch.from_numpy(ev_np).to(args.device)
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for a in range(0, ev.shape[0], 16):
            w = ev[a:a + 16]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=w.is_cuda):
                l = per_token_loss(model, w)                              # (B, T + K - 1) targets 1 .. T+K-1
            l = l[:, cfg.K - 1:]                                          # targets K .. T+K-1
            tot += float(l.sum())
            cnt += l.numel()
    test_nll = tot / cnt
    # (b) the token side: `train()` rounds to whole batches, so what it consumed is not what was
    # requested. The verdict must be read against what was DELIVERED on both sides.
    tokens_delivered = int(res.as_dict().get("tokens", tokens_token))
    F_tok_delivered = TRAIN_FACTOR * choices[width] * tokens_delivered
    log(f"delivered: token {F_tok_delivered:.6e} over {tokens_delivered:,} tokens vs span {F_span:.6e}; "
        f"ratio {F_tok_delivered / F_span:.6f}")
    result = {"span_shape": "adult" if args.adult else "primordium",
              "use_ve": use_ve, "tied": not args.no_tied,
              # a selection at either end of CANDIDATE_WIDTHS was constrained, not free
              "width_at_boundary": width in (CANDIDATE_WIDTHS[0], CANDIDATE_WIDTHS[-1]),
              "width": width, "layers": args.layers, "n_params": model.n_params(),
              "n_params_nonemb": model.n_params_nonemb(), "flops_per_token_token_model": choices[width],
              "flops_per_token_span_agent": span, "flop_ratio_per_token": ratio,
              "tokens_span_agent": tokens_span, "tokens": tokens_token,
              "train_flops_token_model": F_tok, "train_flops_span_agent": F_span, "flop_ratio": F_tok / F_span,
              # the numbers a verdict is read against: both sides as actually spent
              "span_flops_basis": span_basis, "train_flops_span_agent_analytic": F_span_analytic,
              "tokens_delivered": tokens_delivered, "train_flops_token_model_delivered": F_tok_delivered,
              "flop_ratio_delivered": F_tok_delivered / F_span,
              "pool_windows": n_pool, "test_nll_targets_K_to_TK": test_nll, "train_result": res.as_dict(),
              "seconds": time.time() - t0, "complete": True}
    save_phenotype(model, out / "model.pt")
    write_json(out / "result.json", result)
    write_json(out / "manifest.json", manifest({"run": args.run, "arm": "token", "args": vars(args),
                                               "stage": scfg.__dict__, "corpus": corpus.manifest(),
                                               "codec_sha256": file_sha256(codec_path), "eval_digest": array_digest(ev_np),
                                               "scripts_digest": scripts_digest()}))
    log(f"token model test NLL on targets {cfg.K}..{cfg.T + cfg.K - 1}: {test_nll:.4f} "
        f"({res.tokens_per_s:.0f} tok/s, {res.peak_mem_gb:.2f} GB)")


def math_log_ratio(a: float, b: float) -> float:
    import math
    return math.log(a / b)


if __name__ == "__main__":
    main()
