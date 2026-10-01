"""G-ACT: the dedicated MLP policy on the point-mass under the same estimator, against the
agent's shared continuous output, on the same reserved starts.

SPEC: 10 §10.2 G-ACT
SPEC: 02 §2.9.3 (the point-mass), §2.6 (the policy-gradient term)

    python scripts/tempo_gact.py --episodes 20 --N 250 --B 32 --seed 0 [--life runs/tempo/R8-tempo-ts-s0/life.pt]

The MLP sees the number of rollouts the agent's control lanes see in `--episodes` episodes
of the blend at the uniform share (`B / 7` lanes x `N` steps x one rollout group per
`2H / T_s` steps), with `G = 8` rollouts per start state and the DAPO-clipped
group-advantage loss. Both are scored by the mean-action return on the reserved
evaluation starts of the control stream (`n = 64`, the same start seeds for both). The
protocol runs three MLP seeds; the bar is three standard deviations of the MLP's return
over its seeds (10 §10.5). Writes `runs/tempo/GACT-mlp-<config>-s<seed>/result.json`.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from tempo_common import RUNS, config_by_name, log_to, manifest, run_dir, scripts_digest, write_json
from tempo.envs import G_ROLLOUTS, MLPPolicy, PointMass, grpo_loss, group_advantage
from tempo.modalities import STREAMS
from tempo.streams import _eval_seed


def train_mlp(env: PointMass, n_rollouts: int, seed: int, log, lr: float = 3e-4) -> tuple[MLPPolicy, list[float]]:
    torch.manual_seed(seed)
    pol = MLPPolicy()
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    curve = []
    n_groups = max(1, n_rollouts // G_ROLLOUTS)
    for k in range(n_groups):
        s0 = env.reset(1, gen).repeat(G_ROLLOUTS, 1)
        rets = []
        s = s0
        traj_s, traj_a = [], []
        for t in range(env.horizon):
            obs = env.observation(s, t)
            with torch.no_grad():
                a = pol.sample(obs, gen)
            traj_s.append(obs)
            traj_a.append(a)
            s, r = env.step(s, a, t)
            rets.append(r)
        R = torch.stack(rets, 1).sum(1)
        adv, keep = group_advantage(R)
        if keep:
            lp = torch.stack([pol.log_prob(o, a) for o, a in zip(traj_s, traj_a)], 1)   # (G, H)
            loss = grpo_loss(lp, lp.detach(), adv, torch.ones_like(lp))
            opt.zero_grad()
            loss.backward()
            opt.step()
            pol.project()
        curve.append(float(R.mean()))
        if k % 50 == 0:
            log(f"  group {k}/{n_groups} mean return {float(R.mean()):+.3f}")
    return pol, curve


@torch.no_grad()
def evaluate_policy(env: PointMass, act_fn, n: int = 64) -> float:
    """The mean return over the reserved evaluation starts (one start per seed, as the
    control stream's own evaluation draws them)."""
    total = 0.0
    for i in range(n):
        s = env.reset(1, torch.Generator().manual_seed(_eval_seed("control", i)))
        for t in range(env.horizon):
            a = act_fn(env.observation(s, t), t)
            s, r = env.step(s, a, t)
            total += float(r.sum())
    return total / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--N", type=int, default=250)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--life", default=None, help="a life.pt whose agent is scored on the same starts")
    ap.add_argument("--config", default="ts")
    args = ap.parse_args()
    cfg = config_by_name(args.config)
    out = run_dir(f"GACT-mlp-{args.config}-s{args.seed}")
    log = log_to(out / "log.txt")
    half = max(1, cfg.T_s // 2)
    horizon = half * max(1, -(-64 // half))                          # as tempo_life.py: 2H tiles T_s
    env = PointMass(horizon=horizon)
    lanes_control = max(1, args.B // len(STREAMS))
    steps_per_rollout = max(1, (2 * horizon) // cfg.T_s)
    n_rollouts = args.episodes * args.N * lanes_control * G_ROLLOUTS // steps_per_rollout
    log(f"MLP policy: {n_rollouts} rollouts (= {args.episodes} episodes x {args.N} steps x {lanes_control} lanes "
        f"x G={G_ROLLOUTS} / {steps_per_rollout} steps per rollout), horizon {horizon}")
    t0 = time.time()
    pol, curve = train_mlp(env, n_rollouts, args.seed, log)
    mlp_return = evaluate_policy(env, lambda o, t: pol.mean(o), n=args.n_eval)
    zero_return = evaluate_policy(env, lambda o, t: torch.zeros(o.shape[0], 2), n=args.n_eval)
    result = {"mlp_return": mlp_return, "zero_action_return": zero_return, "curve": curve,
              "rollouts": n_rollouts, "horizon": horizon, "seconds": time.time() - t0, "complete": True}
    if args.life:
        from tempo.life import Lane, Life, LifeConfig
        from tempo_life import load_codec
        from tempo.streams import build_streams
        from tempo_common import eval_windows, load_corpus
        codec = load_codec(RUNS / f"codec-{args.config}-s0" / "codec.pt", cfg, "cpu")
        corpus = load_corpus()
        streams = build_streams(cfg, np.asarray(corpus.train.tokens), eval_tokens=eval_windows(corpus, cfg, n=8),
                                fitness_tokens=eval_windows(corpus, cfg, n=8, split="fitness"), control_horizon=horizon)
        life = Life(cfg, LifeConfig(device="cpu", B=8, N=1, n_rollout_eval=args.n_eval), codec, streams)
        life.load(args.life, allow_digest_mismatch=True)
        lane = Lane(0, "control", {}, None, None, True)
        rewards = streams["control"].evaluate(life, lane, args.n_eval, fitness=False)
        result["agent_return"] = float(np.mean(rewards))
        result["agent_life"] = args.life
    write_json(out / "result.json", result)
    write_json(out / "manifest.json", manifest({"run": "G-ACT", "args": vars(args), "scripts_digest": scripts_digest()}))
    log(f"MLP return {mlp_return:+.3f} (zero-action {zero_return:+.3f})" +
        (f"; agent return {result['agent_return']:+.3f}" if args.life else ""))


if __name__ == "__main__":
    main()
