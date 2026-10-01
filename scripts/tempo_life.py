"""The generic life runner: one arm of one run, one seed, one serial job.

SPEC: 10 §10.3  (R1 NOISE, R3 STAT/CRIT, R4 GENE, R5 REV, R6 MORPH, R7 TRANSFER, R8 BLEND, R9 XFER, R10 SOTA)
SPEC: 10 §10.4  (manifests: code and script digests, the evaluation sets' digests, the codec's hash)
SPEC: 08        (the life; the record)

    python scripts/tempo_life.py --run R1 --arm tempo --seed 0 --episodes 1 --N 250 --B 32
    python scripts/tempo_life.py --run R4 --arm a_gamma0 --seed 0 --episodes 20 --N 250 --B 32
    python scripts/tempo_life.py --run R8 --arm tempo --streams all --episodes 20 ...
    python scripts/tempo_life.py --run R8 --arm tempo --streams only-control ...      (K-BLEND's specific agents)
    python scripts/tempo_life.py --run R7 --arm tempo --load runs/tempo/R4-tempo-ts-s0/life.pt --reset-genome ...
    python scripts/tempo_life.py --run R6 --arm flat --F-budget 40 ...                 (K-MORPH at equal FLOPs)

Arms (LifeConfig presets, 04 §4.9, 05 §5.7, 06 §6.6-6.8, 10 §10.2):
    tempo      the full agent (estimator by --estimator; restem in the screens)
    a_gamma0   gamma = 0, no consolidation, primordium shape         (K-GENE (a))
    b_homeo    the zeta rule alone                                    (K-GENE (b))
    c_random   random dg of norm tau                                   (K-GENE (c))
    d_free     free gains carried over, no regression, no genome       (K-GENE (d))
    e_svf      TEMPO with singular-direction gains                     (K-GENE (e))
    f_es       TEMPO with mirrored ES on g                              (K-GENE (f))
    g_grpo     TEMPO with the GRPO variant of estimator 2               (K-GENE (g))
    forward    TEMPO with the forward read                              (K-REV)
    uniform    TEMPO with uniform rehearsal                             (K-SAL)
    schedule / oneshot / flat   the K-MORPH controls
    nohomeo    TEMPO with kappa = 0                                     (K-CRIT control)
    nolat      arm (a) with lambda_lat = 0                              (the text loss of 02 §2.2)
    lat01      arm (a) with lambda_lat = 0.1                            (the recorded ablation, 11 §P-C.23)

The window order is a function of `--order-seed` (default: the seed), so that arms at the
same seed read the same windows (common random numbers, 10 §10.1) while the model seed
sets the initialisation and the proposals.

Writes `runs/tempo/<run>-<arm>[-<tag>]-<config>-s<seed>/{record.json, manifest.json, life.pt, log.txt}`;
`record.json` carries `"complete": true` only when every requested episode finished.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import statistics
import time

import numpy as np
import torch

from tempo_common import (REPO, RUNS, array_digest, config_by_name, eval_windows, file_sha256, load_corpus, log_to,
                          manifest, run_dir, scripts_digest, step_time_outliers, write_json)
from tempo.codec import SpanCodec
from tempo.life import Life, LifeConfig, Paused
from tempo.modalities import STREAMS
from tempo.morphology import SwitchPattern, flops_per_window
from tempo.streams import build_streams

ARMS = {
    "tempo": {},
    "a_gamma0": dict(genome=False, free_gains=False, regenerate=False, homeostasis=False, estimator="none",
                     morphology="primordium", rehearsal="uniform"),
    "b_homeo": dict(genome=False, free_gains=False, regenerate=False, homeostasis=True, estimator="none",
                    morphology="primordium", rehearsal="uniform"),
    "c_random": dict(estimator="random"),
    "d_free": dict(genome=False, free_gains=True, regenerate=False, homeostasis=False, estimator="none",
                   morphology="primordium", rehearsal="uniform"),
    "e_svf": dict(svf=True),
    "f_es": dict(estimator="es"),
    "g_grpo": dict(estimator="grpo"),
    "forward": dict(reverse=False),
    "uniform": dict(rehearsal="uniform"),
    "schedule": dict(morphology="schedule"),
    "oneshot": dict(morphology="oneshot"),
    "flat": dict(morphology="flat"),
    "nohomeo": dict(homeostasis=False),
    "nolat": dict(genome=False, free_gains=False, regenerate=False, homeostasis=False, estimator="none",
                  morphology="primordium", rehearsal="uniform", lambda_lat=0.0),
    # 11 §P-D.17: `nolat` at the ADULT shape. Every K-CONT span run so far ran the primordium,
    # whose active proportions are d_ff/d = 2.0 and (n_heads*d_head)/d = 0.5 -- half the
    # conventional values, and half the FLOP-matched baseline's -- so the criterion measured the
    # span interface and the born-small shape together. This arm holds everything else fixed and
    # grows the shape up, so the two can be told apart.
    "nolat_flat": dict(genome=False, free_gains=False, regenerate=False, homeostasis=False,
                       estimator="none", morphology="flat", rehearsal="uniform", lambda_lat=0.0),
    "lat01": dict(genome=False, free_gains=False, regenerate=False, homeostasis=False, estimator="none",
                  morphology="primordium", rehearsal="uniform", lambda_lat=0.1),
}
STREAM_CHOICES = ["text", "tasks", "all"] + [f"only-{s}" for s in STREAMS]
PAUSE_EXIT = 42
"""The exit code a paused run uses, so the chain can tell a pause from a failure (08 §8.5)."""


def write_record(out, life, args, *, t_start: float, n_done: int, paused: bool, complete: bool = False) -> None:
    """The run's record. `complete` is true only when every requested episode finished (or the
    compute budget was reached); a paused record says so and is resumable (10 §10.5)."""
    write_json(out / "record.json", {
        **life.record, "wall_seconds": time.time() - t_start,
        "episodes_requested": args.episodes, "episodes_done": n_done,
        "paused": bool(paused), "complete": bool(complete),
        "resume": {"episode": life.episode, "step": life._resume_step, "in_episode": life._in_episode},
    })


def load_codec(path, cfg, device, *, fresh_decoder: bool = False, seed: int = 0) -> SpanCodec:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    codec = SpanCodec(blob["vocab"], blob["K"], blob["d_x"], d_c=blob["d_c"], n_dec_layers=blob.get("n_dec_layers"))
    codec.load_state_dict(blob["state"])
    if (blob["vocab"], blob["K"], blob["d_x"]) != (cfg.vocab, cfg.K, cfg.d_x):
        raise ValueError("the codec does not match the configuration")
    if fresh_decoder:
        # 11 §P-D.19: drop R0's fit of the decoder, keep the frozen encoder. The encoder still
        # defines `x_bar` and is still never executed at `lambda_lat = 0`; only the warm start
        # of the thing that turns `y_hat` into logits is removed.
        codec.reinit_decoder(seed=seed)
    return codec.to(device).eval()


def select_streams(streams: dict, choice: str) -> dict:
    if choice == "text":
        return {"text": streams["text"]}
    if choice == "tasks":
        return {"text": streams["text"], "task": streams["task"]}
    if choice == "all":
        return streams
    if choice.startswith("only-"):
        name = choice[5:]
        if name not in streams:
            raise ValueError(f"unknown stream {name}")
        return {name: streams[name]}
    raise ValueError(choice)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", default="tempo", choices=sorted(ARMS))
    ap.add_argument("--config", default="ts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order-seed", type=int, default=None, help="the window order (default: the seed)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--N", type=int, default=250)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--estimator", default=None, help="override the arm's estimator (reinforce|restem|...)")
    ap.add_argument("--streams", default="text", choices=STREAM_CHOICES,
                    help="text; tasks = text + the verifier tasks (R7); all = the seven streams (R8, R9); only-<stream>")
    ap.add_argument("--codec", default=None, help="R0's codec.pt (default runs/tempo/codec-<config>-s0/codec.pt)")
    ap.add_argument("--fresh-decoder", action="store_true",
                    help="re-initialise the span decoder instead of warm-starting it from R0. "
                         "R0 fits it to decode the ENCODER's latent x_bar; the life asks it to "
                         "decode the BACKBONE's y_hat, a different distribution. Diagnostic for "
                         "11 §P-D.19; needs no new baseline, since the FLOPs are unchanged.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sigma-e", type=float, default=0.003)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--n-fitness", type=int, default=64)
    ap.add_argument("--n-rollout-eval", type=int, default=16,
                    help="greedy rollouts per reward-bearing stream on each of the two fixed sets (10 §10.1)")
    ap.add_argument("--load", default=None, help="a life.pt to continue from (R7/R9)")
    ap.add_argument("--reset-genome", action="store_true", help="R7/R9: g <- g_0 and gains regenerated from it")
    ap.add_argument("--tag", default="")
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--n-probe-steps", type=int, default=100)
    ap.add_argument("--F-budget", type=float, default=0.0,
                    help="stop when the cumulative FLOPs (probes included) reach this many primordium-episodes (K-MORPH)")
    ap.add_argument("--no-autocast", action="store_true")
    ap.add_argument("--pause-file", default=None,
                    help="a path whose existence pauses the run at the next step (default runs/tempo/PAUSE)")
    ap.add_argument("--no-resume", action="store_true", help="ignore a paused checkpoint in the run directory")
    ap.add_argument("--pause-every", type=int, default=1,
                    help="how often to look for the pause file, in steps (1: every step)")
    args = ap.parse_args()

    cfg = config_by_name(args.config)
    name = f"{args.run}-{args.arm}{('-' + args.tag) if args.tag else ''}-{args.config}-s{args.seed}"
    out = run_dir(name)
    log = log_to(out / "log.txt")
    codec_path = args.codec or (RUNS / f"codec-{args.config}-s0" / "codec.pt")
    codec = load_codec(codec_path, cfg, args.device, fresh_decoder=args.fresh_decoder, seed=args.seed)
    corpus = load_corpus()
    train = np.asarray(corpus.train.tokens)
    ev = eval_windows(corpus, cfg, n=args.n_eval, split="test")
    fit = eval_windows(corpus, cfg, n=args.n_fitness, split="fitness")
    # the point-mass horizon tiles the window: 2 H is a multiple of T_s (02 §2.9.3), at least 64 ticks
    half = max(1, cfg.T_s // 2)
    horizon = half * max(1, -(-64 // half))
    streams = select_streams(build_streams(cfg, train, eval_tokens=ev, fitness_tokens=fit, seed=args.seed, G=8,
                                           control_horizon=horizon), args.streams)
    preset = dict(ARMS[args.arm])
    if args.estimator:
        preset["estimator"] = args.estimator
    lc = LifeConfig(N=args.N, B=args.B, micro_batch=args.micro_batch, sigma_e=args.sigma_e,
                    eval_every=args.eval_every, n_eval=args.n_eval, n_fitness=args.n_fitness,
                    n_rollout_eval=args.n_rollout_eval, device=args.device,
                    autocast=(args.device == "cuda" and not args.no_autocast), seed=args.seed,
                    order_seed=args.order_seed, G_proposals=args.G, n_probe_steps=args.n_probe_steps,
                    F_budget=args.F_budget, **preset)
    life = Life(cfg, lc, codec, streams)
    pause_file = pathlib.Path(args.pause_file) if args.pause_file else (RUNS / "PAUSE")
    resumed = False
    if not args.no_resume and (out / "life.pt").exists() and (out / "record.json").exists():
        prev = json.loads((out / "record.json").read_text(encoding="utf-8"))
        if prev.get("paused") and not prev.get("complete"):
            life.load(out / "life.pt")                      # the digest must match: same code, same run
            resumed = True
            log(f"resumed {name} at episode {life.episode}"
                + (f", step {life._resume_step}" if life._resume_step is not None else ""))
    if args.load and not resumed:
        life.load(args.load, allow_digest_mismatch=True)
        log(f"loaded {args.load} at episode {life.episode}")
        if args.reset_genome:
            life.reset_genome()
            log("genome reset to the prior; gains regenerated; consolidation memory cleared; pattern frozen")
    man = manifest({
        "run": args.run, "arm": args.arm, "args": vars(args), "config": cfg.as_dict(),
        "life": dataclasses.asdict(lc), "codec": str(codec_path), "codec_sha256": file_sha256(codec_path),
        "fresh_decoder": bool(args.fresh_decoder),
        "streams": sorted(streams), "eval_windows": int(ev.shape[0]), "eval_digest": array_digest(ev),
        "fitness_windows": int(fit.shape[0]), "fitness_digest": array_digest(fit), "corpus": corpus.manifest(),
        "scripts_digest": scripts_digest(), "F_ref": life.F_ref,
        "F_window_primordium": flops_per_window(cfg, SwitchPattern.primordium(cfg)),
    })
    write_json(out / "manifest.json", man)
    log(f"{name}: {args.episodes} episodes of N={args.N}, B={args.B} on {args.device}; arm {args.arm}; "
        f"streams {sorted(streams)}; order seed {life.order_seed}"
        + (f"; RESUMED at episode {life.episode}" if resumed else ""))
    if pause_file.exists():
        log(f"the pause file {pause_file} is present: nothing to do until it is removed")
        write_record(out, life, args, t_start=time.time(), n_done=life.episode, paused=True)
        raise SystemExit(PAUSE_EXIT)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    n_done = life.episode

    def should_pause(n: int) -> bool:
        return (n % max(1, args.pause_every) == 0) and pause_file.exists()

    while life.episode < args.episodes:
        t0 = time.time()
        try:
            rec = life.run_episode(log=log, should_pause=should_pause)
        except Paused as pause:
            life.save(out / "life.pt")
            write_record(out, life, args, t_start=t_start, n_done=n_done, paused=True)
            log(f"paused in episode {pause.episode} before step {pause.next_step}; "
                f"remove {pause_file} and re-run the same command to continue")
            raise SystemExit(PAUSE_EXIT)
        ev_text = rec["eval_final"].get("text")
        rec["episode_seconds"] = time.time() - t0
        rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        steps = [r for r in life.record["steps"][-life.N:] if "seconds" in r]
        med = statistics.median(r["seconds"] for r in steps) if steps else float("nan")
        tokens_per_step = statistics.median(r["tokens"] for r in steps) if steps else float("nan")
        rec["spans_per_s"] = (lc.B * cfg.T_s) / med if med else float("nan")
        rec["tokens_per_s"] = tokens_per_step / med if med else float("nan")
        rec["step_time_outliers"] = step_time_outliers([r["seconds"] for r in steps])
        log(f"episode {life.episode - 1}: L_bar {rec['L_bar']:.4f} L_check {rec['L_check']:.4f} "
            f"test {ev_text['fresh'] if ev_text else float('nan'):.4f} "
            f"reward {rec.get('reward', float('nan')):+.4f} Lambda {rec.get('Lambda', float('nan')):+.3f} "
            f"|g-g0| {rec.get('g_norm', 0.0):.2f} R {rec.get('regression_residual', float('nan')):.2e} "
            f"gstat {rec['gstat_ok']} F {rec['F_cum'] / life.F_ref:.2f} ref-episodes "
            f"{rec['episode_seconds']:.0f}s {rec['tokens_per_s']:.0f} tok/s mem {rec['peak_mem_gb']:.2f} GB")
        n_done = life.episode
        life.save(out / "life.pt")
        write_record(out, life, args, t_start=t_start, n_done=n_done, paused=False,
                     complete=bool(n_done >= args.episodes or life.budget_exhausted))
        if life.budget_exhausted:
            log(f"F budget of {args.F_budget} primordium-episodes reached after {n_done} episodes")
            break
    man["wall_seconds"] = time.time() - t_start
    man["episodes_done"] = n_done
    write_json(out / "manifest.json", man)
    log(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
