"""Streams: what a lane reads, window by window, from each of the seven sources.

SPEC: 08 §8.3   (lanes: contiguous cursors; declared boundaries; window-local attention)
SPEC: 02 §2.9   (the seven streams of the local blend and their terms)
SPEC: 02 §2.8   (wall-clock times, multi-rate interleaving)
SPEC: 06 §6.9   (windows are re-fetchable by identity for rehearsal)
SPEC: 10 §10.1  (fixed evaluation and fitness sets per stream from reserved seeds)

A `Window` is `T_s` input spans with, per modality, the raw inputs at their positions
and the raw *targets* (the next span's raw) at the positions that predict them, a
score mask, and, for a rollout row, the group and the reward. Passive streams (text,
patterns, interleaving) are functions of their cursor and continue across windows
(one declared boundary at the lane's first window and at a source restart, 08 §8.3);
interactive streams (tasks, control, vision queries, speech) act through the `Actor`
the life supplies, so that what the agent emitted is what the window contains (the
egocentric closure of 01 (1.4)).

Second pass after the adversarial review (11 §P-C.29): a supervised row is
teacher-forced (the truth is the input at the answer positions, as (2.6) requires
of a likelihood over the truth) and is separate from the `G` rollout rows, which
carry the emitted answer as input and the policy-gradient positions and no
supervised target; every row of a group carries the fast state it entered with
(`state_in`, the forked lane state under which its samples were drawn); every
window is re-fetchable exactly from its identity (the task parameters travel in
the identity, a pattern window is replayed from its generator state); the
pattern generators continue across a lane's windows; each interactive stream
evaluates by greedy rollouts on a reserved fitness set.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional, Protocol

import numpy as np
import torch

from soma.openended.tasks import BOS, EOS, PAD, SEP, TaskParams, generate
from .config import TempoConfig
from .modalities import MODALITY_INDEX, STREAMS
from . import synthetic as S
from .envs import G_ROLLOUTS, PointMass, SymbolicTaskStream, pad_to_spans

__all__ = ["Window", "RLGroup", "Actor", "Stream", "TextStream", "PatternStream", "TaskStream",
           "ControlStream", "VisionStream", "SpeechStream", "InterleaveStream", "build_streams",
           "EVAL_SEED_BASE", "FITNESS_OFFSET", "window_batch_targets", "scored_tokens",
           "REWARD_STREAMS", "TOKEN_MODALITIES", "ANSWER_BUDGET_SPANS"]
TOKEN_MODALITIES = ("text", "task")
"""The modalities whose scored positions are worth `K` tokens each; every other modality's
scored position is one span."""

EVAL_SEED_BASE = 10 ** 6
"""SPEC: 10 §10.1 -- reserved seeds `10^6 + 1000 i + j` for stream `i`'s evaluation set."""
FITNESS_OFFSET = 500
"""SPEC: 10 §10.1 -- the fitness set (the reward's windows, 06 §6.3) is disjoint from the
evaluation set: seeds `10^6 + 1000 i + 500 + j`; for text, the v3 rows with index `= 0 mod 4`
are the fitness set and the other rows the evaluation set."""
REWARD_STREAMS = ("task", "control", "vision", "speech")
ANSWER_BUDGET_SPANS = 6
"""SPEC: 02 §2.9.2 -- how many spans a task rollout may emit before it is cut off.

A **constant**, not a function of the instance's answer. Deriving the budget from
`len(inst.answer)` lets the truth's length decide how long the agent is allowed to speak,
which is a leak from the answer into the rollout that produces the reward (11 §P-C.35): an
instance with a long answer gets more attempts to terminate than one with a short answer, on
the same policy. Six spans covers the longest answer the reserved families generate at `K = 4`
with slack for the `EOS` span."""


@dataclasses.dataclass
class RLGroup:
    """One rollout of a group of `G`: positions the agent emitted at, what it emitted,
    the modality of the emission, the group's id and this rollout's reward."""

    group: int
    reward: float
    positions: np.ndarray                 # (n_e,) span positions of the emissions
    modality: str                         # "text" (tokens through the codec) or a head name
    samples: torch.Tensor                 # (n_e, K) tokens or (n_e, d_a) continuous samples


@dataclasses.dataclass
class Window:
    stream: str
    ident: tuple
    boundary: bool
    T_s: int
    t_w: np.ndarray                                              # (T_s,) seconds
    mod: np.ndarray                                              # (T_s,) modality index per input span
    inputs: dict[str, tuple[np.ndarray, torch.Tensor]]           # modality -> (positions, raw (n, ...))
    targets: dict[str, tuple[np.ndarray, torch.Tensor]]          # modality -> (positions t, raw_{t+1} (n, ...))
    score_mask: np.ndarray                                       # (T_s,) which target positions are scored
    rl: Optional[RLGroup] = None
    n_tokens: int = 0                                            # scored tokens (text) for per-token accounting
    state_in: Optional[dict] = None                              # the entering fast state of a group row
    rollout: int = 0                                             # 0: the row the lane continues from
    n_used: int = 0
    lambda_ref: Optional[float] = None


class Actor(Protocol):
    """What an interactive stream needs from the life (08 §8.4): perceive one span on a
    lane's live state and get `y_hat`; decode `y_hat` through a head; fork and restore the
    lane; read its live fast state."""

    def perceive(self, lane, raw: torch.Tensor, modality: str, t_w: float) -> torch.Tensor: ...
    def decode_text(self, y_hat: torch.Tensor, generator: torch.Generator, *, greedy: bool = False) -> torch.Tensor: ...
    def decode_head(self, y_hat: torch.Tensor, head: str, generator: torch.Generator, *, greedy: bool = False) -> torch.Tensor: ...
    def fork(self, lane): ...
    def restore(self, lane, snapshot) -> None: ...
    def state_of(self, lane) -> Optional[dict]: ...


class Stream(Protocol):
    name: str
    rate_hz: float

    def cursor(self, lane: int, seed: int) -> dict: ...
    def next_windows(self, cursor: dict, actor: Optional[Actor], lane, generator: torch.Generator) -> list[Window]: ...
    def fetch(self, ident: tuple, *, t0: float = 0.0) -> Window: ...
    """`t0` is the wall-clock second the window began. 02 (2.13) makes the time code a function
    of it, so a window re-fetched to be COMPARED with a live scoring must be given the `t0` it
    had; the fixed evaluation and fitness sets have no live counterpart and keep 0.0."""
    def eval_windows(self, n: int) -> list[Window]: ...
    def fitness_windows(self, n: int) -> list[Window]: ...


def _branches(actor, lane, n: int, name: str):
    """`n` scratch lanes forked from `lane`'s current state: the rollouts of one group all
    start from the state the lane is in (08 §8.3) and differ only in what they sample."""
    snapshot = actor.fork(lane)
    out = []
    for i in range(n):
        b = type(lane)(20_000 + i, name, {}, None, None, True)
        actor.restore(b, snapshot)
        out.append(b)
    return out


def _perceive_all(actor, lanes, raws, modality: str, t_w) -> torch.Tensor:
    """One batched backbone step over `lanes` when the actor offers one, else a loop.

    **Every row steps.** A branch that has finished is stepped with a pad span and its output
    ignored by the caller; it is never held back, because holding a row back makes the lanes'
    KV caches diverge and a batched call cannot then use any cache at all (11 §P-C.35). The
    two paths are the same computation, which `test_life.py` asserts on the *emissions*, not
    only on the rewards.
    """
    if hasattr(actor, "perceive_many"):
        return actor.perceive_many(lanes, raws, modality, t_w)
    return torch.stack([actor.perceive(ln, raws[i], modality, float(t_w[i])) for i, ln in enumerate(lanes)])


def _decode_text_all(actor, y, generator, *, greedy: bool) -> torch.Tensor:
    """`(n, K)` sampled or greedy spans from `y (n, d_x)`."""
    if hasattr(actor, "decode_text_many"):
        return actor.decode_text_many(y, generator, greedy=greedy)
    return torch.stack([actor.decode_text(y[i], generator, greedy=greedy) for i in range(y.shape[0])])


def _decode_head_all(actor, y, head: str, generator, *, greedy: bool) -> torch.Tensor:
    if hasattr(actor, "decode_head_many"):
        return actor.decode_head_many(y, head, generator, greedy=greedy)
    return torch.stack([actor.decode_head(y[i], head, generator, greedy=greedy) for i in range(y.shape[0])])


def _seed_of(cursor: dict) -> int:
    return (cursor["seed"] * 1_000_003 + cursor["lane"] * 7919 + cursor["k"]) % (1 << 31)


def _eval_seed(name: str, i: int, *, fitness: bool = False) -> int:
    return EVAL_SEED_BASE + 1000 * STREAMS.index(name) + (FITNESS_OFFSET if fitness else 0) + int(i)


# =============================================================================
# Text. SPEC: 02 §2.9.1, 08 §8.3
# =============================================================================

class TextStream:
    """Consecutive windows of `T + K` tokens per lane over a shard's flat token array;
    the lane's start offset is seeded; a window's inputs are spans `0..T_s-1` and its
    targets spans `1..T_s`. The cursor advances by `T` so the next window continues the
    text the fast state has read; the boundary is declared at the lane's first window and
    when the source wraps."""

    name = "text"
    rate_hz = 4.0

    def __init__(self, cfg: TempoConfig, tokens: np.ndarray, *, stream_name: str = "text",
                 eval_offsets: Optional[np.ndarray] = None, eval_tokens: Optional[np.ndarray] = None,
                 fitness_tokens: Optional[np.ndarray] = None):
        """`eval_offsets` index `tokens` (synthetic tests) or, with `eval_tokens (n, T + K)`
        given, the evaluation windows are those rows (the v3 test set over the validation
        shards, 10 §10.1) and `fitness_tokens` the rows of the v3 fitness set (document
        residue 0), the reward's windows. Without `fitness_tokens`, rows `i = 0 mod 4` of the
        offsets form the fitness set and the others the evaluation set."""
        self.cfg, self.tokens, self.name = cfg, tokens, stream_name
        self.n_tok = int(tokens.shape[0])
        self.span_tokens = cfg.T + cfg.K
        if self.n_tok < self.span_tokens + 1:
            raise ValueError("shard too small for one window")
        self.eval_offsets = eval_offsets
        self.eval_tokens = None if eval_tokens is None else np.asarray(eval_tokens, dtype=np.int64)
        self.fitness_tokens = None if fitness_tokens is None else np.asarray(fitness_tokens, dtype=np.int64)
        for rows in (self.eval_tokens, self.fitness_tokens):
            if rows is not None and rows.shape[1] != self.span_tokens:
                raise ValueError(f"fixed windows must have T + K = {self.span_tokens} tokens")

    def cursor(self, lane: int, seed: int) -> dict:
        rng = np.random.default_rng((int(seed) * 7919 + lane) % (1 << 63))
        return {"off": int(rng.integers(0, self.n_tok - self.span_tokens)), "fresh": True, "t": 0.0}

    def _window_at(self, off: int, *, boundary: bool, t0: float = 0.0, row: Optional[np.ndarray] = None,
                   tag: str = "eval") -> Window:
        cfg = self.cfg
        K, T_s = cfg.K, cfg.T_s
        src = self.tokens[off:off + self.span_tokens] if row is None else row
        toks = torch.from_numpy(np.asarray(src).astype(np.int64))
        spans = toks.view(T_s + 1, K)
        pos = np.arange(T_s)
        w = Window(self.name, (self.name, off) if row is None else (self.name, tag, off), boundary, T_s,
                   t0 + pos / self.rate_hz, np.full(T_s, MODALITY_INDEX[self.name], dtype=np.int64),
                   {self.name: (pos, spans[:-1])}, {self.name: (pos, spans[1:])},
                   np.ones(T_s, dtype=bool), n_tokens=T_s * K, n_used=T_s)
        return w

    def next_windows(self, cursor: dict, actor, lane, generator) -> list[Window]:
        off = cursor["off"]
        boundary = cursor["fresh"]
        if off + self.span_tokens > self.n_tok:
            off = 0
            boundary = True                                       # the source restarts (08 §8.3 (b))
        w = self._window_at(off, boundary=boundary, t0=cursor.get("t", 0.0))
        cursor["off"] = off + self.cfg.T
        cursor["fresh"] = False
        cursor["t"] = cursor.get("t", 0.0) + self.cfg.T_s / self.rate_hz
        return [w]

    def fetch(self, ident: tuple, *, t0: float = 0.0) -> Window:
        if len(ident) == 3 and ident[1] in ("eval", "fit"):
            rows = self.eval_tokens if ident[1] == "eval" else self.fitness_tokens
            return self._window_at(int(ident[2]), boundary=True, t0=t0, row=rows[int(ident[2])], tag=ident[1])
        return self._window_at(int(ident[1]), boundary=True, t0=t0)

    def _fixed(self, n: int, *, fitness: bool) -> list[Window]:
        if self.fitness_tokens is not None and self.eval_tokens is not None:
            rows = self.fitness_tokens if fitness else self.eval_tokens
            return [self._window_at(i, boundary=True, row=rows[i], tag="fit" if fitness else "eval")
                    for i in range(min(n, len(rows)))]
        if self.eval_tokens is not None:
            idx = [i for i in range(len(self.eval_tokens)) if (i % 4 == 0) == fitness][:n]
            return [self._window_at(i, boundary=True, row=self.eval_tokens[i]) for i in idx]
        if self.eval_offsets is None:
            raise ValueError("no evaluation windows")
        offs = [int(o) for i, o in enumerate(self.eval_offsets) if (i % 4 == 0) == fitness][:n]
        return [self._window_at(o, boundary=True) for o in offs]

    def eval_windows(self, n: int) -> list[Window]:
        return self._fixed(n, fitness=False)

    def fitness_windows(self, n: int) -> list[Window]:
        return self._fixed(n, fitness=True)


# =============================================================================
# Pattern streams. SPEC: 02 §2.9.6
# =============================================================================

def _lorenz_rk4(u: np.ndarray, dt: float = 0.01, sigma: float = 10.0, rho: float = 28.0,
                beta: float = 8.0 / 3.0) -> np.ndarray:
    k1 = S._lorenz_f(u, sigma, rho, beta)
    k2 = S._lorenz_f(u + 0.5 * dt * k1, sigma, rho, beta)
    k3 = S._lorenz_f(u + 0.5 * dt * k2, sigma, rho, beta)
    k4 = S._lorenz_f(u + dt * k3, sigma, rho, beta)
    return u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class PatternStream:
    """Logistic / Lorenz / ECA observations at 100 Hz, one observation per span; the
    target of span `t` is observation `t+1` under a Gaussian (or Bernoulli) head. A lane
    holds one generator (its kind by lane index) whose trajectory *continues* across the
    lane's windows: the last observation of window `k` is the first input of window `k+1`
    (the fast state is the memory of the trajectory, 08 §8.3). The boundary is declared
    at the lane's first window only. A window is identified by `(kind, seed, k)` and is
    replayed exactly from the generator's state at the window's start (cached when the
    window was generated, else by re-running the trajectory from its seed)."""

    name = "pattern"
    rate_hz = 100.0
    KINDS = ("logistic", "lorenz", "eca")
    LORENZ_SCALE = 20.0

    def __init__(self, cfg: TempoConfig, *, kind_cycle: tuple[str, ...] = KINDS):
        self.cfg, self.kinds = cfg, kind_cycle

    def cursor(self, lane: int, seed: int) -> dict:
        kind = self.kinds[int(lane) % len(self.kinds)]
        return {"seed": int(seed), "lane": int(lane), "k": 0, "t": 0.0, "kind": kind,
                "state": None, "obs": None}

    # ------------------------------------------------------------------ generators
    def _params(self, kind: str, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        if kind == "logistic":
            return {"r": round(float(rng.uniform(3.5, 4.0)), 6), "x0": float(rng.uniform(0.05, 0.95))}
        if kind == "lorenz":
            return {"x0": tuple(float(v) for v in rng.normal(0, 1, size=3) + np.array([1.0, 1.0, 20.0]))}
        if kind == "eca":
            return {"rule": int(rng.choice(S.ECA_RULES)), "row_seed": int(rng.integers(0, 2 ** 31))}
        raise ValueError(kind)

    def _start(self, kind: str, params: dict, n: int) -> np.ndarray:
        """The first `n` observations of the trajectory (after the generator's burn-in)."""
        if kind == "logistic":
            obs, _ = S.logistic_stream(params["r"], n, x0=params["x0"])
        elif kind == "lorenz":
            obs, _ = S.lorenz_stream(n, x0=params["x0"], burn=200, scale=self.LORENZ_SCALE)
        else:
            rows, _ = S.eca_stream(params["rule"], n, seed=params["row_seed"], trials=1, horizon=1)
            obs = rows.astype(np.float32)
        return np.asarray(obs, dtype=np.float32)

    def _continue(self, kind: str, params: dict, state: np.ndarray, n_new: int) -> np.ndarray:
        """`obs (n_new + 1, d)` with `obs[0] = state` and `n_new` further steps of the map."""
        out = np.empty((n_new + 1,) + tuple(state.shape), dtype=np.float32)
        out[0] = state
        if kind == "logistic":
            r = float(params["r"])
            hist = [float(v) for v in state]                         # (x_k, ..., x_{k-7})
            for i in range(1, n_new + 1):
                x = r * hist[0] * (1.0 - hist[0])
                hist = [x] + hist[:-1]
                out[i] = np.asarray(hist, dtype=np.float32)
        elif kind == "lorenz":
            u = np.asarray(state, dtype=np.float64) * self.LORENZ_SCALE
            for i in range(1, n_new + 1):
                u = _lorenz_rk4(u)
                out[i] = (u / self.LORENZ_SCALE).astype(np.float32)
        else:
            row = np.asarray(state, dtype=np.uint8)
            for i in range(1, n_new + 1):
                row = S.eca_step(row, int(params["rule"]))
                out[i] = row.astype(np.float32)
        return out

    def _observations(self, kind: str, seed: int, k: int) -> np.ndarray:
        """`T_s + 1` observations of window `k` of trajectory `(kind, seed)`, **always by the
        same route**: the generator's own run from its parameters, advanced `k` windows.

        The first form cached the state at each window's start and continued from it, but the
        cached state is the float32 *observation*, and re-deriving a chaotic map from a rounded
        state is not the same trajectory: on the logistic map at `lambda = ln 2` per step, 128
        steps amplified float32 rounding to O(1), so a rehearsed or probed window `k = 0` was a
        different window from the one whose loss was recorded (11 §P-C.38). The cache is gone;
        a window is a pure function of `(kind, seed, k)`, which is what 08 §8.3 requires of an
        identity, and it costs `k` window-lengths of a cheap generator to re-derive.
        """
        T_s = self.cfg.T_s
        params = self._params(kind, seed)
        obs = self._start(kind, params, T_s + 1)
        for _ in range(int(k)):
            obs = self._continue(kind, params, obs[-1], T_s)
        return obs

    def _window(self, kind: str, seed: int, k: int, obs: np.ndarray, *, boundary: bool, t0: float) -> Window:
        T_s = self.cfg.T_s
        params = self._params(kind, seed)
        lam = S.lambda_ref("logistic", r=params["r"]) if kind == "logistic" else \
            S.lambda_ref("lorenz") if kind == "lorenz" else S.lambda_ref("eca", rule=params["rule"])
        o = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        pos = np.arange(T_s)
        w = Window(self.name, (self.name, kind, int(seed), int(k)), boundary, T_s, t0 + pos / self.rate_hz,
                   np.full(T_s, MODALITY_INDEX[kind], dtype=np.int64),
                   {kind: (pos, o[:-1])}, {kind: (pos, o[1:])}, np.ones(T_s, dtype=bool), n_used=T_s,
                   lambda_ref=float(lam))
        return w

    def next_windows(self, cursor, actor, lane, generator) -> list[Window]:
        """The lane's trajectory continues across its windows. It is advanced by carrying the
        *live* float64 tail in the cursor, so the lane pays nothing to continue; the identity
        `(kind, seed, k)` re-derives the same window from scratch, and the two agree because
        `_continue` is the same map either way (asserted by a test)."""
        kind, k = cursor["kind"], cursor["k"]
        seed = (cursor["seed"] * 1_000_003 + cursor["lane"] * 7919) % (1 << 31)
        prev = cursor.get("obs")
        if prev is None:
            obs = self._start(kind, self._params(kind, seed), self.cfg.T_s + 1)
        else:
            obs = self._continue(kind, self._params(kind, seed), np.asarray(prev), self.cfg.T_s)
        w = self._window(kind, seed, k, obs, boundary=(k == 0), t0=cursor["t"])
        cursor["obs"] = obs[-1].copy()
        cursor["k"] = k + 1
        cursor["t"] += self.cfg.T_s / self.rate_hz
        return [w]

    def fetch(self, ident: tuple, *, t0: float = 0.0) -> Window:
        kind, seed, k = ident[1], int(ident[2]), int(ident[3])
        return self._window(kind, seed, k, self._observations(kind, seed, k), boundary=True, t0=t0)

    def _fixed(self, n: int, *, fitness: bool) -> list[Window]:
        return [self.fetch((self.name, self.kinds[i % len(self.kinds)], _eval_seed(self.name, i, fitness=fitness), 0))
                for i in range(n)]

    def eval_windows(self, n: int) -> list[Window]:
        return self._fixed(n, fitness=False)

    def fitness_windows(self, n: int) -> list[Window]:
        return self._fixed(n, fitness=True)


# =============================================================================
# Verifier tasks. SPEC: 02 §2.9.2
# =============================================================================

def _text_items(tokens: list[int], K: int) -> torch.Tensor:
    toks, _ = pad_to_spans(tokens, [True] * len(tokens), K)
    return torch.from_numpy(toks).view(-1, K)


def _decode_until_eos(actor, lane, y, generator, *, max_spans: int, modality: str, t: float, dt: float,
                      greedy: bool) -> list[int]:
    emitted: list[int] = []
    for j in range(max_spans):
        span = actor.decode_text(y, generator, greedy=greedy)
        emitted.extend(int(v) for v in span)
        if EOS in span.tolist():
            break
        if j + 1 < max_spans:
            y = actor.perceive(lane, span, modality, t + (j + 1) * dt)
    return emitted


def _candidate(emitted: list[int]) -> list[int]:
    return emitted[:emitted.index(EOS)] if EOS in emitted else list(emitted)


class TaskStream:
    """Prompt spans perceived; the answer emitted span by span by the agent (`G`
    rollouts from the forked lane state), verified exactly. Each draw yields one
    teacher-forced supervised row (prompt then the true answer as inputs; targets the next
    prompt span and the true answer spans; the lane continues from this row) and `G`
    rollout rows (prompt then the emitted answer; the policy-gradient positions and the
    reward; no supervised target). The parameters travel in the identity so that a
    window re-fetches exactly whatever the curriculum did since."""

    name = "task"
    rate_hz = 4.0

    def __init__(self, cfg: TempoConfig, *, seed: int = 0, G: int = G_ROLLOUTS):
        self.cfg, self.G = cfg, G
        self.tasks = SymbolicTaskStream(T_ctx=cfg.T, seed=seed)

    def cursor(self, lane: int, seed: int) -> dict:
        return {"seed": int(seed), "lane": int(lane), "k": 0, "t": 0.0}

    # ------------------------------------------------------------------ curriculum
    def seed_curriculum(self, solve_rate: Callable[[TaskParams, int], float]) -> None:
        self.tasks.seed_archive(solve_rate)

    def curriculum_step(self, solve_rate: Callable[[TaskParams, int], float], n: int = 1) -> dict:
        return self.tasks.archive_step(solve_rate, n)

    @staticmethod
    def _params_tuple(params: TaskParams) -> tuple:
        return tuple(dataclasses.astuple(params))

    @staticmethod
    def _params_from(t: tuple) -> TaskParams:
        return TaskParams(*t)

    def instance(self, seed: int, params: TaskParams):
        return generate(params, np.random.default_rng(int(seed)))

    # ------------------------------------------------------------------ layouts
    def _supervised(self, inst, t0: float, ident: tuple, boundary: bool, state_in) -> Window:
        cfg = self.cfg
        K, T_s = cfg.K, cfg.T_s
        prompt = _text_items(self.tasks.prompt_tokens(inst), K)[:T_s]                 # (n_p, K)
        truth = _text_items(list(inst.answer) + [EOS], K)                             # (n_a, K)
        n_p = prompt.shape[0]
        n_a = min(truth.shape[0], T_s - n_p)
        inputs = torch.full((T_s, K), PAD, dtype=torch.long)
        inputs[:n_p] = prompt
        inputs[n_p:n_p + n_a] = truth[:n_a]
        targets = torch.full((T_s, K), PAD, dtype=torch.long)
        targets[:n_p - 1] = prompt[1:]
        a0 = n_p - 1
        n_sup = min(truth.shape[0], T_s - a0)
        targets[a0:a0 + n_sup] = truth[:n_sup]
        score = np.zeros(T_s, dtype=bool)
        score[a0:a0 + n_sup] = True
        pos = np.arange(T_s)
        w = Window(self.name, ident, boundary, T_s, t0 + pos / self.rate_hz,
                   np.full(T_s, MODALITY_INDEX["task"], dtype=np.int64),
                   {"task": (pos, inputs)}, {"task": (pos[:a0 + n_sup], targets[:a0 + n_sup])},
                   score, n_tokens=int(n_sup * K), state_in=state_in, rollout=0, n_used=n_p + n_a)
        return w

    def _rollout_row(self, inst, emitted: list[int], t0: float, ident: tuple, boundary: bool, group: int,
                     reward: float, state_in, r: int) -> Window:
        cfg = self.cfg
        K, T_s = cfg.K, cfg.T_s
        prompt = _text_items(self.tasks.prompt_tokens(inst), K)[:T_s]
        n_p = prompt.shape[0]
        emit = _text_items(list(emitted), K) if emitted else prompt[:0]
        a0 = n_p - 1
        n_e = max(0, min(emit.shape[0], T_s - n_p))
        inputs = torch.full((T_s, K), PAD, dtype=torch.long)
        inputs[:n_p] = prompt
        inputs[n_p:n_p + n_e] = emit[:n_e]
        pos = np.arange(T_s)
        rl = RLGroup(group, reward, np.arange(a0, a0 + n_e), "text", emit[:n_e]) if n_e else None
        w = Window(self.name, ident, boundary, T_s, t0 + pos / self.rate_hz,
                   np.full(T_s, MODALITY_INDEX["task"], dtype=np.int64),
                   {"task": (pos, inputs)}, {}, np.zeros(T_s, dtype=bool), rl=rl, state_in=state_in,
                   rollout=r, n_used=n_p + n_e)
        return w

    # ------------------------------------------------------------------ acting
    def act_group(self, actor, lanes, insts, generator: torch.Generator, t: float, *, greedy: bool):
        """`n` branches in lockstep. Rows are partitioned by **prompt length** first: a row
        whose prompt is shorter than its neighbours' cannot be padded up to theirs, because
        the state it would then decode from is the state after its prompt *plus pad spans*,
        not the state after its own separator (11 §P-C.35). Within a partition every row steps
        every span, and a row that has emitted EOS is stepped with a pad whose output is
        discarded."""
        K, T_s = self.cfg.K, self.cfg.T_s
        n = len(lanes)
        prompts = [_text_items(self.tasks.prompt_tokens(i), K)[:T_s] for i in insts]
        by_len: dict[int, list[int]] = {}
        for i, p in enumerate(prompts):
            by_len.setdefault(int(p.shape[0]), []).append(i)
        emitted: list[list[int]] = [[] for _ in range(n)]
        for n_p, rows in sorted(by_len.items()):
            sub = [lanes[i] for i in rows]
            budget = max(1, min(T_s - n_p, ANSWER_BUDGET_SPANS))    # the same for every row (see the constant)
            y = None
            for j in range(n_p):
                y = _perceive_all(actor, sub, [prompts[i][j] for i in rows], "task",
                                  [t + j / self.rate_hz] * len(rows))
            done = [False] * len(rows)
            for j in range(budget):
                span = _decode_text_all(actor, y, generator, greedy=greedy)
                for b, i in enumerate(rows):
                    if done[b]:
                        continue
                    emitted[i].extend(int(v) for v in span[b])
                    if EOS in span[b].tolist():
                        done[b] = True
                if all(done) or j + 1 >= budget:
                    break
                # every row steps; a finished row's output is discarded, so that the rows of a
                # partition never fall out of cache phase (11 §P-C.35)
                y = _perceive_all(actor, sub, [span[b] for b in range(len(rows))], "task",
                                  [t + (n_p + j + 1) / self.rate_hz] * len(rows))
        return emitted, [float(self.tasks.reward(insts[i], emitted[i])) for i in range(n)]

    def _act(self, actor, lane, inst, generator: torch.Generator, t: float, *, greedy: bool) -> tuple[list[int], float]:
        em, rw = self.act_group(actor, [lane], [inst], generator, t, greedy=greedy)
        return em[0], rw[0]

    def next_windows(self, cursor, actor, lane, generator) -> list[Window]:
        """One prompt: the supervised row and `G` rollouts from a forked copy of the lane's
        state, returned together so that the group's advantage is formed in one step (02 (2.12))."""
        seed = _seed_of(cursor)
        params = self.tasks.sample_params(np.random.default_rng(seed))   # CRN: the order seed alone
        inst = self.instance(seed, params)
        cursor["k"] += 1
        ident = (self.name, seed, self._params_tuple(params))
        boundary = True                                             # each task is its own source
        t0 = cursor["t"]
        cursor["t"] += self.cfg.T_s / self.rate_hz
        if actor is None:
            return [self._supervised(inst, t0, ident, boundary, None)]
        lane.boundary_pending = True                                # each task is a new source
        branches = _branches(actor, lane, self.G, self.name)
        for b in branches:
            b.S, b.cache, b.boundary_pending = None, None, True
        g = torch.Generator().manual_seed((seed * 31) % (1 << 31))
        emitted, rewards = self.act_group(actor, branches, [inst] * self.G, g, t0, greedy=False)
        windows = [self._supervised(inst, t0, ident, boundary, None)]
        for r in range(self.G):
            windows.append(self._rollout_row(inst, emitted[r], t0, ident + (r + 1,), boundary, seed,
                                             rewards[r], None, r + 1))
        return windows

    def fetch(self, ident: tuple, *, t0: float = 0.0) -> Window:
        seed, params = int(ident[1]), self._params_from(tuple(ident[2]))
        return self._supervised(self.instance(seed, params), t0, (self.name, seed, tuple(ident[2])), True, None)

    def _fixed_params(self, i: int) -> TaskParams:
        from soma.openended.tasks import family_lower_bounds
        lbs = family_lower_bounds()
        return lbs[i % len(lbs)]

    def _fixed(self, n: int, *, fitness: bool) -> list[Window]:
        return [self.fetch((self.name, _eval_seed(self.name, i, fitness=fitness), self._params_tuple(self._fixed_params(i))))
                for i in range(n)]

    def eval_windows(self, n: int) -> list[Window]:
        return self._fixed(n, fitness=False)

    def fitness_windows(self, n: int) -> list[Window]:
        return self._fixed(n, fitness=True)

    def evaluate(self, actor, lane, n: int, *, fitness: bool = True) -> list[float]:
        """Greedy rewards on the reserved set from a fresh state (the fitness score of the
        reward-bearing streams, 06 §6.3; 10 §10.1), the `n` instances in lockstep."""
        seeds = [_eval_seed(self.name, i, fitness=fitness) for i in range(n)]
        insts = [self.instance(seeds[i], self._fixed_params(i)) for i in range(n)]
        lanes = [type(lane)(20_000 + i, self.name, {}, None, None, True) for i in range(n)]
        _, rewards = self.act_group(actor, lanes, insts, torch.Generator().manual_seed(seeds[0]), 0.0, greedy=True)
        return rewards


# =============================================================================
# Continuous control. SPEC: 02 §2.9.3
# =============================================================================

class ControlStream:
    """The point-mass: each tick perceives the proprioception span and the reward span;
    the agent's action at each proprioception span is sampled from the Gaussian head;
    `G` rollouts from the same start state and lane state form a group with the episode
    return as reward. A rollout of `H` ticks is `2H` spans, split into windows of `T_s`;
    each window carries the fast state the rollout entered it with. The lane continues
    from rollout 0."""

    name = "control"
    rate_hz = 20.0

    def __init__(self, cfg: TempoConfig, *, manoeuvre: bool = False, G: int = G_ROLLOUTS, horizon: int = 64):
        self.cfg, self.G = cfg, G
        self.env = PointMass(manoeuvre=manoeuvre, horizon=horizon)
        if 2 * horizon % cfg.T_s:
            raise ValueError("2 * horizon must be a multiple of T_s so rollouts tile windows")

    def cursor(self, lane: int, seed: int) -> dict:
        return {"seed": int(seed), "lane": int(lane), "k": 0, "t": 0.0, "pending": []}

    def _rollout_windows(self, actions: torch.Tensor, rewards: torch.Tensor, ident: tuple, group: int,
                         t0: float, obs: torch.Tensor, states: list, r: int) -> list[Window]:
        """`obs (H, 8)` perceived, `rewards (H,)`, `actions (H, 2)` -> windows of `T_s` spans:
        span `2i` = proprioception of tick `i`, span `2i+1` = the reward of tick `i`."""
        cfg = self.cfg
        H = actions.shape[0]
        T_s = cfg.T_s
        n_win = (2 * H) // T_s
        out = []
        ret = float(rewards.sum())
        for k in range(n_win):
            lo = k * T_s // 2
            hi = lo + T_s // 2
            pos = np.arange(T_s)
            prop_pos = pos[0::2]
            rew_pos = pos[1::2]
            mod = np.empty(T_s, dtype=np.int64)
            mod[0::2] = MODALITY_INDEX["prop"]
            mod[1::2] = MODALITY_INDEX["reward"]
            t_w = t0 + (lo + np.arange(T_s) / 2.0) / self.rate_hz
            rl = RLGroup(group, ret, prop_pos, "act", actions[lo:hi])
            w = Window(self.name, ident + (k,), k == 0, T_s, t_w, mod,
                       {"prop": (prop_pos, obs[lo:hi]), "reward": (rew_pos, rewards[lo:hi, None])},
                       {}, np.zeros(T_s, dtype=bool), rl=rl, state_in=states[k] if states else None,
                       rollout=r, n_used=T_s)
            out.append(w)
        return out

    def act_group(self, actor, lanes, s0: torch.Tensor, generator: torch.Generator, t0: float, *, greedy: bool):
        """`n` rollouts in lockstep from the start states `s0 (n, 8)`; returns
        `(obs (n, H, 8), actions (n, H, 2), rewards (n, H), states)` with `states[i][k]` the
        fast state rollout `i` entered window `k` with (08 §8.3)."""
        T_s = self.cfg.T_s
        n = len(lanes)
        s = s0
        t = t0
        obs_l, act_l, rew_l = [], [], []
        states = [[] for _ in range(n)]
        ticks_per_window = T_s // 2
        for tick in range(self.env.horizon):
            if tick % ticks_per_window == 0:
                for i, ln in enumerate(lanes):
                    states[i].append(actor.state_of(ln))
            o = self.env.observation(s, tick)
            y = _perceive_all(actor, lanes, o, "prop", [t] * n)
            a = _decode_head_all(actor, y, "act", generator, greedy=greedy)
            s, rw = self.env.step(s, a.to(s.dtype), tick)
            _perceive_all(actor, lanes, rw[:, None], "reward", [t + 0.5 / self.rate_hz] * n)
            obs_l.append(o)
            act_l.append(a)
            rew_l.append(rw)
            t += 1.0 / self.rate_hz
        return (torch.stack(obs_l, 1), torch.stack(act_l, 1), torch.stack(rew_l, 1), states)

    def _act(self, actor, lane, s0: torch.Tensor, generator: torch.Generator, t0: float, *, greedy: bool
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        obs, acts, rews, states = self.act_group(actor, [lane], s0, generator, t0, greedy=greedy)
        return obs[0], acts[0], rews[0], states[0]

    def next_windows(self, cursor, actor, lane, generator) -> list[Window]:
        """`G` rollouts from the same start and lane state; each rollout is `2H / T_s`
        windows: the `k`-th windows of all rollouts form one group and are returned
        together; later `k` are queued on the cursor for the lane's next steps."""
        if cursor["pending"]:
            return cursor["pending"].pop(0)
        seed = _seed_of(cursor)
        cursor["k"] += 1
        g_env = torch.Generator().manual_seed(seed)
        s0 = self.env.reset(1, g_env)
        ident = (self.name, seed)
        t0 = cursor["t"]
        cursor["t"] += self.env.horizon / self.rate_hz
        if actor is None:
            _, actions, rewards = self.env.rollout(lambda s, t: torch.zeros(1, 2), s0)
            obs = torch.stack([self.env.observation(s0, 0)[0]] * actions.shape[1])
            ws = self._rollout_windows(actions[0], rewards[0], ident, seed, t0, obs, [], 0)
            cursor["pending"] = [[w] for w in ws[1:]]
            return [ws[0]]
        # 08 §8.3 (d): a rollout group is a new source, so its branches start from a reset
        # state. The first form forked the lane and kept its state, which made the `boundary`
        # flag on every control window inert (11 §P-C.38)
        lane.boundary_pending = True
        branches = _branches(actor, lane, self.G, self.name)
        for b in branches:
            b.S, b.cache, b.boundary_pending = None, None, True
        g = torch.Generator().manual_seed((seed * 31) % (1 << 31))
        obs, actions, rewards, states = self.act_group(actor, branches, s0.repeat(self.G, 1), g, t0, greedy=False)
        per_rollout = [self._rollout_windows(actions[r], rewards[r], ident + (r,), seed, t0, obs[r], states[r], r)
                       for r in range(self.G)]
        n_k = len(per_rollout[0])
        groups = [[per_rollout[r][k] for r in range(len(per_rollout))] for k in range(n_k)]
        cursor["pending"] = groups[1:]
        return groups[0]

    def fetch(self, ident: tuple, *, t0: float = 0.0) -> Window:
        """The zero-action rollout of the start `seed` (a passive window: proprioception
        and rewards perceived, nothing scored); rehearsal of control windows re-lives the
        start, it does not replay the agent's old actions."""
        seed = int(ident[1])
        g_env = torch.Generator().manual_seed(seed)
        s0 = self.env.reset(1, g_env)
        states, actions, rewards = self.env.rollout(lambda s, t: torch.zeros(1, 2), s0)
        obs = torch.stack([self.env.observation(states[:, t], t)[0] for t in range(actions.shape[1])])
        k = int(ident[-1]) if len(ident) > 2 else 0
        return self._rollout_windows(actions[0], rewards[0], (self.name, seed), seed, t0, obs, [], 0)[k]

    def eval_windows(self, n: int) -> list[Window]:
        return [self.fetch((self.name, _eval_seed(self.name, i))) for i in range(n)]

    def fitness_windows(self, n: int) -> list[Window]:
        return [self.fetch((self.name, _eval_seed(self.name, i, fitness=True))) for i in range(n)]

    def eval_starts(self, n: int, *, fitness: bool = False) -> torch.Tensor:
        return torch.cat([self.env.reset(1, torch.Generator().manual_seed(_eval_seed(self.name, i, fitness=fitness)))
                          for i in range(n)])

    def evaluate(self, actor, lane, n: int, *, fitness: bool = True) -> list[float]:
        """Mean-action returns from a fresh state on the reserved starts. The `n` rollouts are
        independent and of equal length, so they are stepped in lockstep on `n` scratch lanes
        through one batched backbone call per tick (08 §8.3) when the actor offers one."""
        seeds = [_eval_seed(self.name, i, fitness=fitness) for i in range(n)]
        s = torch.cat([self.env.reset(1, torch.Generator().manual_seed(sd)) for sd in seeds])
        lanes = [type(lane)(20_000 + i, self.name, {}, None, None, True) for i in range(n)]
        _, _, rewards, _ = self.act_group(actor, lanes, s, torch.Generator().manual_seed(seeds[0]), 0.0, greedy=True)
        return [float(v) for v in rewards.sum(1)]


# =============================================================================
# Synthetic vision. SPEC: 02 §2.9.4
# =============================================================================

class VisionStream:
    """Frames at 5 Hz; every 4th frame is followed by a query (task spans) whose answer
    the agent emits (`G` rollouts); frame positions predict the next frame (2.10). The
    supervised row is teacher-forced (true answers as inputs); the rollout rows carry
    the emitted answers and the policy-gradient positions."""

    name = "vision"
    rate_hz = 5.0

    def __init__(self, cfg: TempoConfig, *, G: int = G_ROLLOUTS, frames_per_window: int = 16):
        self.cfg, self.G = cfg, G
        self.fpw = max(4, min(frames_per_window, cfg.T_s // 2))   # queries and answers need room

    def cursor(self, lane: int, seed: int) -> dict:
        return {"seed": int(seed), "lane": int(lane), "k": 0, "t": 0.0}

    def _scene_run(self, seed: int):
        stream = S.SceneStream(seed)
        return stream.next_frames(self.fpw + 1)

    def _layout(self, frames, t_w0, queries, answers_by_query: list[list[int]], ident, group, reward, t0,
                boundary, *, supervised: bool, state_in, r: int) -> Window:
        """Inputs: frame f_i; after frame i that carries a query, the query spans and then
        the answer spans (the truth when supervised, the emission otherwise). Targets
        (supervised row only): frame positions predict the next frame; the span before each
        answer span predicts the true answer. Rollout rows: the emitted spans' positions."""
        cfg = self.cfg
        K, T_s = cfg.K, cfg.T_s
        pos_i, mods, img_pos, img_raw, img_tpos, img_tgt = 0, [], [], [], [], []
        task_pos, task_raw, task_tpos, task_tgt, score, rl_pos, rl_samples = [], [], [], [], [], [], []
        t_list = []
        n_tokens = 0
        qi = 0
        for i in range(self.fpw):
            if pos_i >= T_s:
                break
            img_pos.append(pos_i)
            img_raw.append(torch.from_numpy(frames[i]))
            t_list.append(t0 + t_w0[i])
            mods.append(MODALITY_INDEX["image"])
            if qi < len(queries) and queries[qi][0] == i:
                _, q, a, kind = queries[qi]
                prompt = _text_items([BOS] + list(q) + [SEP], K)
                truth = _text_items(list(a) + [EOS], K)
                ans_tokens = list(a) + [EOS] if supervised else (answers_by_query[qi] if qi < len(answers_by_query) else [])
                ans = _text_items(ans_tokens, K) if ans_tokens else truth[:0]
                qi += 1
                pos_i += 1
                for j in range(prompt.shape[0]):
                    if pos_i >= T_s:
                        break
                    task_pos.append(pos_i)
                    task_raw.append(prompt[j])
                    mods.append(MODALITY_INDEX["task"])
                    t_list.append(t0 + t_w0[i] + (j + 1) * 0.05)
                    if supervised and j + 1 < prompt.shape[0]:
                        task_tpos.append(pos_i)
                        task_tgt.append(prompt[j + 1])
                    pos_i += 1
                a0 = pos_i - 1
                n_ans = ans.shape[0]
                for j in range(n_ans):
                    if pos_i >= T_s:
                        break
                    task_pos.append(pos_i)
                    task_raw.append(ans[j])
                    mods.append(MODALITY_INDEX["task"])
                    t_list.append(t0 + t_w0[i] + (prompt.shape[0] + j + 1) * 0.05)
                    pos_i += 1
                if supervised:
                    for j in range(min(truth.shape[0], T_s - a0)):
                        task_tpos.append(a0 + j)
                        task_tgt.append(truth[j])
                        score.append(a0 + j)
                        n_tokens += K
                else:
                    n_rl = max(0, min(n_ans, T_s - a0))
                    if n_rl:
                        rl_pos.extend(range(a0, a0 + n_rl))
                        rl_samples.append(ans[:n_rl])
                continue
            if supervised and i + 1 < self.fpw and pos_i + 1 < T_s:
                img_tpos.append(pos_i)
                img_tgt.append(torch.from_numpy(frames[i + 1]))
            pos_i += 1
        n_used = pos_i
        mod = np.full(T_s, MODALITY_INDEX["image"], dtype=np.int64)
        mod[:len(mods)] = np.asarray(mods, dtype=np.int64)
        t_w = np.zeros(T_s)
        t_w[:len(t_list)] = np.asarray(t_list)
        if len(t_list) < T_s:
            t_w[len(t_list):] = t_list[-1] + np.arange(1, T_s - len(t_list) + 1) * 0.05
        inputs = {}
        if img_pos:
            inputs["image"] = (np.asarray(img_pos), torch.stack(img_raw))
        if task_pos:
            inputs["task"] = (np.asarray(task_pos), torch.stack(task_raw))
        targets = {}
        if img_tpos:
            targets["image"] = (np.asarray(img_tpos), torch.stack(img_tgt))
        if task_tpos:
            targets["task"] = (np.asarray(task_tpos), torch.stack(task_tgt))
        sm = np.zeros(T_s, dtype=bool)
        if score:
            sm[np.asarray(score, dtype=np.int64)] = True
        for p in img_tpos:
            sm[p] = True
        rl = RLGroup(group, reward, np.asarray(rl_pos), "text", torch.cat(rl_samples)) if rl_pos else None
        if n_used < T_s:                                            # black frames fill the unused tail
            pad_pos = np.arange(n_used, T_s)
            pad_img = torch.zeros(len(pad_pos), 3, 32, 32)
            if "image" in inputs:
                inputs["image"] = (np.concatenate([inputs["image"][0], pad_pos]), torch.cat([inputs["image"][1], pad_img]))
            else:
                inputs["image"] = (pad_pos, pad_img)
        w = Window(self.name, ident, boundary, T_s, t_w, mod, inputs, targets, sm, rl=rl, n_tokens=n_tokens,
                   state_in=state_in, rollout=r, n_used=n_used)
        return w

    def act_group(self, actor, lanes, frames, t_w0, queries, generator, t0: float, *, greedy: bool):
        """`n` branches over the same scene in lockstep: every branch perceives the same
        frames and query spans and decodes its own answers."""
        K = self.cfg.K
        n = len(lanes)
        emitted_all: list[list[list[int]]] = [[] for _ in range(n)]
        correct = [0] * n
        total = 0
        qi = 0
        for i in range(self.fpw):
            fr = torch.from_numpy(frames[i])
            y = _perceive_all(actor, lanes, [fr] * n, "image", [t0 + t_w0[i]] * n)
            if qi < len(queries) and queries[qi][0] == i:
                _, q, a, kind = queries[qi]
                qi += 1
                total += 1
                prompt = _text_items([BOS] + list(q) + [SEP], K)
                for j in range(prompt.shape[0]):
                    y = _perceive_all(actor, lanes, [prompt[j]] * n, "task", [t0 + t_w0[i] + (j + 1) * 0.05] * n)
                em = [[] for _ in range(n)]
                done = [False] * n
                n_ans = 2
                for j in range(n_ans):
                    span = _decode_text_all(actor, y, generator, greedy=greedy)
                    for b in range(n):
                        if done[b]:
                            continue
                        em[b].extend(int(v) for v in span[b])
                        if EOS in span[b].tolist():
                            done[b] = True
                    # every emitted span is perceived, including the last: `_layout` places all
                    # of them as inputs, so the rollout's context must contain all of them
                    # (11 §P-C.35). Every row steps; a finished row's output is discarded.
                    y = _perceive_all(actor, lanes, [span[b] for b in range(n)], "task",
                                      [t0 + t_w0[i] + (prompt.shape[0] + j + 1) * 0.05] * n)
                    if all(done):
                        break
                for b in range(n):
                    emitted_all[b].append(em[b])
                    correct[b] += int(_candidate(em[b]) == list(a))
        return emitted_all, [c / max(total, 1) for c in correct]

    def _act(self, actor, lane, frames, t_w0, queries, generator, t0: float, *, greedy: bool):
        em, rw = self.act_group(actor, [lane], frames, t_w0, queries, generator, t0, greedy=greedy)
        return em[0], rw[0]

    def next_windows(self, cursor, actor, lane, generator) -> list[Window]:
        seed = _seed_of(cursor)
        cursor["k"] += 1
        frames, t_w0, queries = self._scene_run(seed)
        ident = (self.name, seed)
        t0 = cursor["t"]
        cursor["t"] += (self.fpw + 1) / self.rate_hz
        if actor is None:
            return [self._layout(frames, t_w0, queries, [], ident, seed, 0.0, t0, True, supervised=True, state_in=None, r=0)]
        lane.boundary_pending = True
        branches = _branches(actor, lane, self.G, self.name)
        for b in branches:
            b.S, b.cache, b.boundary_pending = None, None, True
        g = torch.Generator().manual_seed((seed * 31) % (1 << 31))
        emitted_all, rewards = self.act_group(actor, branches, frames, t_w0, queries, g, t0, greedy=False)
        windows = [self._layout(frames, t_w0, queries, [], ident, seed, 0.0, t0, True, supervised=True, state_in=None, r=0)]
        for r in range(self.G):
            windows.append(self._layout(frames, t_w0, queries, emitted_all[r], ident + (r + 1,), seed, rewards[r],
                                        t0, True, supervised=False, state_in=None, r=r + 1))
        return windows

    def fetch(self, ident: tuple, *, t0: float = 0.0) -> Window:
        frames, t_w0, queries = self._scene_run(int(ident[1]))
        return self._layout(frames, t_w0, queries, [], (self.name, int(ident[1])), int(ident[1]), 0.0, t0, True,
                            supervised=True, state_in=None, r=0)

    def eval_windows(self, n: int) -> list[Window]:
        return [self.fetch((self.name, _eval_seed(self.name, i))) for i in range(n)]

    def fitness_windows(self, n: int) -> list[Window]:
        return [self.fetch((self.name, _eval_seed(self.name, i, fitness=True))) for i in range(n)]

    def evaluate(self, actor, lane, n: int, *, fitness: bool = True) -> list[float]:
        """Each reserved scene is a different frame sequence, so the scenes are scored one at a
        time; the branches inside a scene are already batched (there is one per scene here)."""
        out = []
        for i in range(n):
            seed = _eval_seed(self.name, i, fitness=fitness)
            frames, t_w0, queries = self._scene_run(seed)
            lane.S, lane.cache, lane.boundary_pending = None, None, True
            _, r = self._act(actor, lane, frames, t_w0, queries, torch.Generator().manual_seed(seed), 0.0, greedy=True)
            out.append(r)
        return out


# =============================================================================
# Tone-coded speech. SPEC: 02 §2.9.5
# =============================================================================

class SpeechStream:
    """Transcription: frames perceived, the symbols emitted through the codec (task
    spans), exact reward. Production: the symbols perceived as task spans, the frames
    emitted through the `speech` Gaussian head, decoded by the nearest tone, exact
    reward. Alternating by window index. The supervised row is teacher-forced (the
    true symbols, or the true frames, as inputs); the rollout rows carry the emissions."""

    name = "speech"
    rate_hz = 62.5
    N_SYMBOLS_PER_UTTERANCE = 6

    def __init__(self, cfg: TempoConfig, *, G: int = G_ROLLOUTS):
        self.cfg, self.G = cfg, G

    def cursor(self, lane: int, seed: int) -> dict:
        return {"seed": int(seed), "lane": int(lane), "k": 0, "t": 0.0}

    @property
    def n_symbols(self) -> int:
        """Symbols per utterance: 6 when the window has room, fewer at small `T_s`
        (4 frames per symbol, then the separator and the emitted spans)."""
        return max(1, min(self.N_SYMBOLS_PER_UTTERANCE, (self.cfg.T_s - 4) // (S.FRAMES_PER_SYMBOL + 1)))

    def _utterance(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, S.N_TONES, size=self.n_symbols)

    def _symbols_tokens(self, symbols) -> list[int]:
        from soma.openended.tasks import sym
        return [sym(int(s)) for s in symbols]

    def _layout_transcription(self, symbols, answer_tokens: list[int], ident, group, reward, t0, boundary,
                              *, supervised: bool, state_in, r: int) -> Window:
        cfg = self.cfg
        K, T_s = cfg.K, cfg.T_s
        frames = torch.from_numpy(S.tone_frames(symbols))                          # (4n, 64)
        n_f = min(frames.shape[0], T_s - 2)
        frames = frames[:n_f]
        prompt = _text_items([SEP], K)
        truth = _text_items(self._symbols_tokens(symbols) + [EOS], K)
        ans = _text_items(answer_tokens, K) if answer_tokens else truth[:0]
        f_pos = list(range(n_f))
        t_pos = [n_f]
        e_pos = list(range(n_f + 1, min(T_s, n_f + 1 + ans.shape[0])))
        ans = ans[:len(e_pos)]
        n_used = min(n_f + 1 + ans.shape[0], T_s)
        inputs = {"speech": (np.asarray(f_pos), frames)}
        task_in_pos = t_pos + e_pos
        inputs["task"] = (np.asarray(task_in_pos), torch.cat([prompt, ans]))
        a0 = t_pos[0]
        n_sup = min(truth.shape[0], T_s - a0)
        targets = {}
        sm = np.zeros(T_s, dtype=bool)
        rl = None
        if supervised:
            if n_f > 1:
                targets["speech"] = (np.asarray(f_pos[:-1]), frames[1:])
                sm[:n_f - 1] = True
            targets["task"] = (np.arange(a0, a0 + n_sup), truth[:n_sup])
            sm[a0:a0 + n_sup] = True
        elif ans.shape[0]:
            rl = RLGroup(group, reward, np.arange(a0, a0 + ans.shape[0]), "text", ans)
        mod = np.full(T_s, MODALITY_INDEX["speech"], dtype=np.int64)
        for p in task_in_pos:
            mod[p] = MODALITY_INDEX["task"]
        t_w = t0 + np.arange(T_s) / self.rate_hz
        if n_used < T_s:
            pad_pos = np.arange(n_used, T_s)
            inputs["speech"] = (np.concatenate([inputs["speech"][0], pad_pos]), torch.cat([frames, torch.zeros(len(pad_pos), 64)]))
        w = Window(self.name, ident, boundary, T_s, t_w, mod, inputs, targets, sm, rl=rl,
                   n_tokens=int(n_sup * K) if supervised else 0, state_in=state_in, rollout=r, n_used=n_used)
        return w

    def _layout_production(self, symbols, frames_in: Optional[torch.Tensor], ident, group, reward, t0, boundary,
                           *, supervised: bool, state_in, r: int) -> Window:
        cfg = self.cfg
        K, T_s = cfg.K, cfg.T_s
        prompt = _text_items([BOS] + self._symbols_tokens(symbols) + [SEP], K)[:T_s - 1]
        truth = torch.from_numpy(S.tone_frames(symbols))                           # (4n, 64)
        n_p = prompt.shape[0]
        fr = truth if supervised else (frames_in if frames_in is not None else truth[:0])
        fr = fr[:T_s - n_p]
        n_e = fr.shape[0]
        pos_task = list(range(n_p))
        pos_sp = list(range(n_p, n_p + n_e))
        n_used = min(n_p + n_e, T_s)
        inputs = {"task": (np.asarray(pos_task), prompt)}
        if n_e:
            inputs["speech"] = (np.asarray(pos_sp), fr)
        a0 = n_p - 1
        n_sup = min(truth.shape[0], T_s - a0)
        targets = {}
        sm = np.zeros(T_s, dtype=bool)
        rl = None
        if supervised:
            targets["task"] = (np.arange(n_p - 1), prompt[1:])
            targets["speech"] = (np.arange(a0, a0 + n_sup), truth[:n_sup])
            sm[a0:a0 + n_sup] = True
        elif n_e:
            rl = RLGroup(group, reward, np.arange(a0, a0 + n_e), "speech", fr)
        mod = np.full(T_s, MODALITY_INDEX["speech"], dtype=np.int64)
        mod[:n_p] = MODALITY_INDEX["task"]
        t_w = t0 + np.arange(T_s) / self.rate_hz
        if n_used < T_s:
            pad_pos = np.arange(n_used, T_s)
            if "speech" in inputs:
                inputs["speech"] = (np.concatenate([inputs["speech"][0], pad_pos]), torch.cat([inputs["speech"][1], torch.zeros(len(pad_pos), 64)]))
            else:
                inputs["speech"] = (pad_pos, torch.zeros(len(pad_pos), 64))
        # the production row scores *frames*, not text: `n_tokens` counts scored text tokens
        # only, and claiming the prompt's tokens made the batch denominator wrong (11 §P-C.38)
        w = Window(self.name, ident, boundary, T_s, t_w, mod, inputs, targets, sm, rl=rl,
                   n_tokens=0, state_in=state_in, rollout=r, n_used=n_used)
        return w

    def act_group(self, actor, lanes, symbols_list, production: bool, generator, t0: float, *, greedy: bool):
        """`n` branches in lockstep. Every branch of a training group has the same utterance;
        the evaluation set gives each branch its own, which is why the raws are per branch."""
        K = self.cfg.K
        n = len(lanes)
        # rows are partitioned by the length of what they perceive before they emit: a shorter
        # utterance cannot be padded up to a longer one, because the state it would decode from
        # would be the state after its own frames *plus pad frames* (11 §P-C.35)
        if not production:
            frames = [torch.from_numpy(S.tone_frames(sy)) for sy in symbols_list]
            by_len: dict[int, list[int]] = {}
            for b, f in enumerate(frames):
                by_len.setdefault(int(f.shape[0]), []).append(b)
            em: list[list[int]] = [[] for _ in range(n)]
            sep = _text_items([SEP], K)[0]
            max_spans = self.n_symbols // K + 2
            for n_f, rows in sorted(by_len.items()):
                sub = [lanes[b] for b in rows]
                for i in range(n_f):
                    _perceive_all(actor, sub, [frames[b][i] for b in rows], "speech",
                                  [t0 + i / self.rate_hz] * len(rows))
                y = _perceive_all(actor, sub, [sep] * len(rows), "task", [t0 + n_f / self.rate_hz] * len(rows))
                done = [False] * len(rows)
                for j in range(max_spans):
                    span = _decode_text_all(actor, y, generator, greedy=greedy)
                    for i2, b in enumerate(rows):
                        if done[i2]:
                            continue
                        em[b].extend(int(v) for v in span[i2])
                        if EOS in span[i2].tolist():
                            done[i2] = True
                    if all(done) or j + 1 >= max_spans:
                        break
                    y = _perceive_all(actor, sub, [span[i2] for i2 in range(len(rows))], "task",
                                      [t0 + (n_f + j + 1) / self.rate_hz] * len(rows))
            return em, [float(_candidate(em[b]) == self._symbols_tokens(symbols_list[b])) for b in range(n)]
        prompts = [_text_items([BOS] + self._symbols_tokens(sy) + [SEP], K) for sy in symbols_list]
        by_len = {}
        for b, p in enumerate(prompts):
            by_len.setdefault(int(p.shape[0]), []).append(b)
        out: list[list] = [[] for _ in range(n)]
        for n_p, rows in sorted(by_len.items()):
            sub = [lanes[b] for b in rows]
            y = None
            for j in range(n_p):
                y = _perceive_all(actor, sub, [prompts[b][j] for b in rows], "task",
                                  [t0 + j / self.rate_hz] * len(rows))
            n_frames = max(S.FRAMES_PER_SYMBOL * len(symbols_list[b]) for b in rows)
            for i in range(n_frames):
                f = _decode_head_all(actor, y, "speech", generator, greedy=greedy)
                for i2, b in enumerate(rows):
                    out[b].append(f[i2])
                if i + 1 < n_frames:
                    y = _perceive_all(actor, sub, [f[i2] for i2 in range(len(rows))], "speech",
                                      [t0 + (n_p + i + 1) / self.rate_hz] * len(rows))
        emitted = [torch.stack(o) for o in out]
        rewards = [float(np.array_equal(S.decode_frames(emitted[b].detach().cpu().float().numpy()), symbols_list[b]))
                   for b in range(n)]
        return emitted, rewards

    def _act(self, actor, lane, symbols, production: bool, generator, t0: float, *, greedy: bool):
        em, rw = self.act_group(actor, [lane], [symbols], production, generator, t0, greedy=greedy)
        return em[0], rw[0]

    def _supervised(self, symbols, production: bool, ident, seed, t0, state_in) -> Window:
        """The teacher-forced row: the **truth** is the input at the answer positions, as
        08 §8.3 requires of a supervised row. Transcription passed an empty answer, so the
        truth was never an input and half its scored targets were predicted from a padded zero
        frame (11 §P-C.38)."""
        if production:
            return self._layout_production(symbols, None, ident, seed, 0.0, t0, True, supervised=True, state_in=state_in, r=0)
        truth = self._symbols_tokens(symbols) + [EOS]
        return self._layout_transcription(symbols, truth, ident, seed, 0.0, t0, True, supervised=True, state_in=state_in, r=0)

    def next_windows(self, cursor, actor, lane, generator) -> list[Window]:
        seed = _seed_of(cursor)
        production = cursor["k"] % 2 == 1
        cursor["k"] += 1
        symbols = self._utterance(seed)
        ident = (self.name, seed, "prod" if production else "trans")
        t0 = cursor["t"]
        cursor["t"] += self.cfg.T_s / self.rate_hz
        if actor is None:
            return [self._supervised(symbols, production, ident, seed, t0, None)]
        lane.boundary_pending = True
        branches = _branches(actor, lane, self.G, self.name)
        for b in branches:
            b.S, b.cache, b.boundary_pending = None, None, True
        g = torch.Generator().manual_seed((seed * 31) % (1 << 31))
        emitted, rewards = self.act_group(actor, branches, [symbols] * self.G, production, g, t0, greedy=False)
        windows = [self._supervised(symbols, production, ident, seed, t0, None)]
        for r in range(self.G):
            if production:
                w = self._layout_production(symbols, emitted[r], ident + (r + 1,), seed, rewards[r], t0, True,
                                            supervised=False, state_in=None, r=r + 1)
            else:
                w = self._layout_transcription(symbols, emitted[r], ident + (r + 1,), seed, rewards[r], t0, True,
                                               supervised=False, state_in=None, r=r + 1)
            windows.append(w)
        return windows

    def fetch(self, ident: tuple, *, t0: float = 0.0) -> Window:
        symbols = self._utterance(int(ident[1]))
        return self._supervised(symbols, ident[2] == "prod", tuple(ident[:3]), int(ident[1]), t0, None)

    def _fixed(self, n: int, *, fitness: bool) -> list[Window]:
        return [self.fetch((self.name, _eval_seed(self.name, i, fitness=fitness), "prod" if i % 2 else "trans")) for i in range(n)]

    def eval_windows(self, n: int) -> list[Window]:
        return self._fixed(n, fitness=False)

    def fitness_windows(self, n: int) -> list[Window]:
        return self._fixed(n, fitness=True)

    def evaluate(self, actor, lane, n: int, *, fitness: bool = True) -> list[float]:
        """Transcription and production are different span layouts, so the two halves of the
        set are scored as two lockstep groups."""
        out: list[float] = []
        for production in (False, True):
            idx = [i for i in range(n) if bool(i % 2) == production]
            if not idx:
                continue
            seeds = [_eval_seed(self.name, i, fitness=fitness) for i in idx]
            syms = [self._utterance(sd) for sd in seeds]
            lanes = [type(lane)(20_000 + i, self.name, {}, None, None, True) for i in idx]
            _, rw = self.act_group(actor, lanes, syms, production, torch.Generator().manual_seed(seeds[0]), 0.0,
                                   greedy=True)
            out.extend(rw)
        return out


# =============================================================================
# Multi-rate interleaving. SPEC: 02 §2.9.7
# =============================================================================

class InterleaveStream:
    """Tone speech frames (62.5 Hz, predicted under the speech head) interleaved by
    wall-clock with text spans (4 spans/s) from the text stream: 15-16 frames between
    consecutive text spans. Text positions predict the next text span; frame positions
    predict the next frame. The text continues across the lane's windows (the boundary is
    the text cursor's first window); each window's utterance is drawn from its seed."""

    name = "interleave"
    rate_hz = 62.5

    def __init__(self, cfg: TempoConfig, text: TextStream):
        self.cfg, self.text = cfg, text

    def cursor(self, lane: int, seed: int) -> dict:
        c = self.text.cursor(lane, seed)
        return {"text": c, "seed": int(seed), "lane": int(lane), "k": 0, "t": 0.0}

    @property
    def n_text(self) -> int:
        return max(2, self.cfg.T_s // 16)

    def _window(self, text_off: int, seed: int, t0: float, boundary: bool, ident: tuple) -> Window:
        cfg = self.cfg
        K, T_s = cfg.K, cfg.T_s
        rng = np.random.default_rng(seed)
        n_text = self.n_text
        toks = torch.from_numpy(np.asarray(self.text.tokens[text_off:text_off + (n_text + 1) * K]).astype(np.int64)).view(n_text + 1, K)
        symbols = rng.integers(0, S.N_TONES, size=(T_s // S.FRAMES_PER_SYMBOL) + 1)
        frames = torch.from_numpy(S.tone_frames(symbols))
        events = S.interleave([[(k / 4.0, ("text", k)) for k in range(n_text)],
                               [(i / 62.5, ("speech", i)) for i in range(frames.shape[0])]])
        events = events[:T_s + 1]
        mod = np.zeros(T_s, dtype=np.int64)
        t_w = np.zeros(T_s)
        tpos, traw, tt, ttgt, spos, sraw, st, stgt = [], [], [], [], [], [], [], []
        last_text = {}
        last_speech = {}
        for p, (t, si, payload) in enumerate(events[:T_s]):
            kind, k = payload
            t_w[p] = t0 + t
            if kind == "text":
                mod[p] = MODALITY_INDEX["text"]
                tpos.append(p)
                traw.append(toks[k])
                last_text[p] = k
            else:
                mod[p] = MODALITY_INDEX["speech"]
                spos.append(p)
                sraw.append(frames[k])
                last_speech[p] = k
        for p in tpos:                                              # the next span of the same modality
            k = last_text[p]
            if k + 1 < toks.shape[0]:
                tt.append(p)
                ttgt.append(toks[k + 1])
        for p in spos:
            k = last_speech[p]
            if k + 1 < frames.shape[0]:
                st.append(p)
                stgt.append(frames[k + 1])
        inputs = {"text": (np.asarray(tpos), torch.stack(traw)), "speech": (np.asarray(spos), torch.stack(sraw))}
        targets = {"text": (np.asarray(tt), torch.stack(ttgt)), "speech": (np.asarray(st), torch.stack(stgt))}
        sm = np.zeros(T_s, dtype=bool)
        sm[np.asarray(tt + st, dtype=np.int64)] = True
        w = Window(self.name, ident, boundary, T_s, t_w, mod, inputs, targets, sm, n_tokens=len(tt) * K, n_used=T_s)
        return w

    def next_windows(self, cursor, actor, lane, generator) -> list[Window]:
        c = cursor["text"]
        off = c["off"]
        boundary = c["fresh"]
        if off + (self.n_text + 1) * self.cfg.K > self.text.n_tok:
            off = 0
            boundary = True
        seed = _seed_of(cursor)
        w = self._window(off, seed, cursor["t"], boundary, (self.name, off, seed))
        c["off"] = off + self.n_text * self.cfg.K
        c["fresh"] = False
        cursor["k"] += 1
        cursor["t"] += self.cfg.T_s / self.rate_hz
        return [w]

    def fetch(self, ident: tuple, *, t0: float = 0.0) -> Window:
        return self._window(int(ident[1]), int(ident[2]), t0, True, tuple(ident[:3]))

    def _fixed(self, n: int, *, fitness: bool) -> list[Window]:
        base = _eval_seed(self.name, 0, fitness=fitness)
        rng = np.random.default_rng(base)
        return [self.fetch((self.name, int(rng.integers(0, self.text.n_tok - (self.n_text + 1) * self.cfg.K)), base + i))
                for i in range(n)]

    def eval_windows(self, n: int) -> list[Window]:
        return self._fixed(n, fitness=False)

    def fitness_windows(self, n: int) -> list[Window]:
        return self._fixed(n, fitness=True)


def build_streams(cfg: TempoConfig, tokens: np.ndarray, *, eval_offsets: Optional[np.ndarray] = None,
                  eval_tokens: Optional[np.ndarray] = None, fitness_tokens: Optional[np.ndarray] = None,
                  seed: int = 0, G: int = G_ROLLOUTS, control_horizon: int = 64) -> dict[str, object]:
    """The seven streams of 02 §2.9 over one text shard."""
    text = TextStream(cfg, tokens, eval_offsets=eval_offsets, eval_tokens=eval_tokens, fitness_tokens=fitness_tokens)
    return {
        "text": text,
        "task": TaskStream(cfg, seed=seed, G=G),
        "control": ControlStream(cfg, G=G, horizon=control_horizon),
        "vision": VisionStream(cfg, G=G),
        "speech": SpeechStream(cfg, G=G),
        "pattern": PatternStream(cfg),
        "interleave": InterleaveStream(cfg, text),
    }


def scored_tokens(w: Window, K: int) -> int:
    """The number of things `window_loss` will sum this window's likelihood over: a scored
    text or task position is `K` tokens, any other scored position is one span. This is the
    quantity 08 §8.2 S3 divides the batch's loss by, and it is computable before the forward
    (the loss must be scaled for gradient accumulation before any micro-batch runs)."""
    total = 0
    for m, (pos, _) in w.targets.items():
        if len(pos) == 0:
            continue
        scored = int(np.asarray([w.score_mask[p] for p in pos]).sum())
        total += scored * (K if m in TOKEN_MODALITIES else 1)
    return total


def window_batch_targets(windows: list[Window]) -> dict[str, tuple[np.ndarray, np.ndarray, torch.Tensor]]:
    """Gather targets across a batch: modality -> (batch index, position, raw)."""
    out: dict[str, list] = {}
    for b, w in enumerate(windows):
        for m, (pos, raw) in w.targets.items():
            if len(pos) == 0:
                continue
            out.setdefault(m, []).append((np.full(len(pos), b), pos, raw))
    return {m: (np.concatenate([a for a, _, _ in v]), np.concatenate([p for _, p, _ in v]),
                torch.cat([r for _, _, r in v])) for m, v in out.items()}
