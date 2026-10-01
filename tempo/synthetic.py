"""The generated streams of the local blend: scenes, tone-coded speech, chaotic
patterns with their Lyapunov exponents, and multi-rate interleaving.

SPEC: 02 §2.9.4  (synthetic vision: 32x32 scenes, transforms, exact queries)
SPEC: 02 §2.9.5  (tone-coded speech: symbols as tones, 64-bin frames at 62.5 Hz)
SPEC: 02 §2.9.6  (logistic map, Lorenz, elementary cellular automata; lambda_ref)
SPEC: 02 §2.9.7  (multi-rate interleaving by wall-clock)

Every generator is a deterministic function of its seed (I10) and depends on
numpy only. Tensors are produced by the callers.

Deviation from 02 §2.9.5, recorded (11 §P-C.16): the spec placed 56 semitone
tones from 220 Hz at a 4 kHz sample rate, which puts the upper symbols above
the Nyquist frequency and packs the lower ones inside one DFT bin. The tones
are placed on the frame's own DFT grid instead: sample rate 8 kHz, 128 samples
(16 ms, 62.5 Hz) per frame, `f_s = 62.5 (s + 4)` Hz for `s = 0..55` (250 Hz to
3687.5 Hz, below Nyquist), 64 log-magnitude bins `0..63` of the 128-point DFT.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Sequence

import numpy as np

from soma.openended.tasks import GRID_TRANSFORMS, N_SYMBOLS, sym

__all__ = [
    # scenes
    "SHAPES", "COLOURS", "GRID", "CELL", "IMG", "Scene", "random_scene", "render",
    "transform_scene", "QUERY_KINDS", "make_query", "solve_query", "query_tokens",
    "SceneStream", "SYM",
    # speech
    "SAMPLE_RATE", "FRAME", "FRAMES_PER_SYMBOL", "N_TONES", "tone_freq", "symbol_waveform",
    "frame_features", "tone_frames", "tone_bank", "decode_frames",
    # patterns
    "logistic_stream", "lorenz_stream", "eca_step", "eca_stream", "ECA_RULES", "WOLFRAM_CLASS",
    # interleaving
    "interleave",
]

# =============================================================================
# Synthetic vision. SPEC: 02 §2.9.4
# =============================================================================

SHAPES = ("circle", "square", "triangle")
COLOURS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
           (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))
GRID, CELL, IMG = 4, 8, 32
QUERY_KINDS = ("COUNT", "WHERE", "AFTER", "SAME")


class SYM:
    """The reserved-alphabet layout of the scene queries (56 symbols available).

    query kinds 0..3, colours 4..9, shapes 10..12, NONE 13, YES 14, NO 15,
    transforms 16..24, cells 25..40, counts 41..45 (0..4).
    """

    KIND = 0
    COLOUR = 4
    SHAPE = 10
    NONE = 13
    YES = 14
    NO = 15
    TRANSFORM = 16
    CELL = 25
    COUNT = 41
    END = COUNT + 5

    @staticmethod
    def kind(k: str) -> int:
        return sym(SYM.KIND + QUERY_KINDS.index(k))

    @staticmethod
    def colour(c: int) -> int:
        return sym(SYM.COLOUR + c)

    @staticmethod
    def shape(s: int) -> int:
        return sym(SYM.SHAPE + s)

    @staticmethod
    def transform(tf: str) -> int:
        return sym(SYM.TRANSFORM + GRID_TRANSFORMS.index(tf))

    @staticmethod
    def cell(i: int) -> int:
        return sym(SYM.CELL + i)

    @staticmethod
    def count(n: int) -> int:
        return sym(SYM.COUNT + n)


assert SYM.END <= N_SYMBOLS


@dataclasses.dataclass(frozen=True)
class Scene:
    """Up to four items `(shape, colour, row, col)`, at most one per cell."""

    items: tuple[tuple[int, int, int, int], ...]

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """`(shape_grid, colour_grid)` of shape `(GRID, GRID)`, `-1` where empty."""
        sg = -np.ones((GRID, GRID), dtype=np.int64)
        cg = -np.ones((GRID, GRID), dtype=np.int64)
        for s, c, r, k in self.items:
            sg[r, k], cg[r, k] = s, c
        return sg, cg

    @staticmethod
    def from_grid(sg: np.ndarray, cg: np.ndarray) -> "Scene":
        items = []
        for r in range(GRID):
            for k in range(GRID):
                if sg[r, k] >= 0:
                    items.append((int(sg[r, k]), int(cg[r, k]), r, k))
        return Scene(tuple(items))


def random_scene(rng: np.random.Generator) -> Scene:
    """1-4 shapes of random type and colour in distinct cells."""
    n = int(rng.integers(1, 5))
    cells = rng.choice(GRID * GRID, size=n, replace=False)
    items = []
    for c in cells:
        items.append((int(rng.integers(0, len(SHAPES))), int(rng.integers(0, len(COLOURS))),
                      int(c) // GRID, int(c) % GRID))
    return Scene(tuple(sorted(items, key=lambda it: (it[2], it[3]))))


def _shape_mask(shape: int) -> np.ndarray:
    """The `CELL x CELL` boolean mask of a shape drawn in its cell."""
    yy, xx = np.mgrid[0:CELL, 0:CELL].astype(np.float64)
    cy = cx = (CELL - 1) / 2.0
    if shape == 0:                                                   # circle, radius 3
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= 3.0 ** 2
    if shape == 1:                                                   # square 6x6
        return (yy >= 1) & (yy <= 6) & (xx >= 1) & (xx <= 6)
    if shape == 2:                                                   # upward triangle
        hw = (yy - 1.0) / 5.0 * 3.0
        return (yy >= 1) & (yy <= 6) & (np.abs(xx - cx) <= hw)
    raise ValueError(shape)


_MASKS = [_shape_mask(s) for s in range(len(SHAPES))]


def render(scene: Scene) -> np.ndarray:
    """`(3, 32, 32)` float32 in `[0, 1]`, black background."""
    img = np.zeros((3, IMG, IMG), dtype=np.float32)
    for s, c, r, k in scene.items:
        m = _MASKS[s]
        y0, x0 = r * CELL, k * CELL
        for ch in range(3):
            img[ch, y0:y0 + CELL, x0:x0 + CELL][m] = COLOURS[c][ch]
    return img


def transform_scene(scene: Scene, tf: str) -> Scene:
    """The nine `GRID_TRANSFORMS` names of `soma/openended/tasks.py` applied to the cell grid.
    The positional ones share SOMA's numpy semantics (`rot90` is `np.rot90`, flips, `translate`
    a roll by (1, 1), `tile` the identity as in SOMA); `recolour` is the scene's own rule, the
    colour successor modulo 6 on occupied cells (SOMA recolours a symbol grid including its
    background, which has no counterpart here)."""
    sg, cg = scene.grid()
    if tf == "identity" or tf == "tile":
        return Scene.from_grid(sg, cg)
    if tf in ("rot90", "rot180", "rot270"):
        k = {"rot90": 1, "rot180": 2, "rot270": 3}[tf]
        return Scene.from_grid(np.rot90(sg, k), np.rot90(cg, k))
    if tf == "flip_h":
        return Scene.from_grid(np.fliplr(sg), np.fliplr(cg))
    if tf == "flip_v":
        return Scene.from_grid(np.flipud(sg), np.flipud(cg))
    if tf == "translate":
        roll = lambda g: np.roll(np.roll(g, 1, axis=0), 1, axis=1)
        return Scene.from_grid(roll(sg), roll(cg))
    if tf == "recolour":
        cg2 = np.where(cg >= 0, (cg + 1) % len(COLOURS), -1)
        return Scene.from_grid(sg, cg2)
    raise ValueError(f"unknown transform {tf!r}")


def make_query(scene: Scene, rng: np.random.Generator) -> tuple[list[int], list[int], str]:
    """One query of a random kind with its exact answer: `(query, answer, kind)`.

    Queries are drawn so that the trivial answers do not dominate: with probability 1/2
    COUNT names a (colour, shape) present in the scene, AFTER names a cell occupied after
    the transform, WHERE names a shape present, and SAME draws two distinct cells of which
    at least one is occupied (a constant answer earned ~80 % exact reward before this).
    """
    kind = QUERY_KINDS[int(rng.integers(0, 4))]
    items = scene.items
    if kind == "COUNT":
        if items and rng.random() < 0.5:
            s, c, _, _ = items[int(rng.integers(0, len(items)))]
        else:
            c, s = int(rng.integers(0, 6)), int(rng.integers(0, 3))
        q = [SYM.kind(kind), SYM.colour(c), SYM.shape(s)]
    elif kind == "WHERE":
        s = items[int(rng.integers(0, len(items)))][0] if (items and rng.random() < 0.5) else int(rng.integers(0, 3))
        q = [SYM.kind(kind), SYM.shape(s)]
    elif kind == "AFTER":
        tf = GRID_TRANSFORMS[int(rng.integers(0, len(GRID_TRANSFORMS)))]
        after = transform_scene(scene, tf).items
        if after and rng.random() < 0.5:
            _, _, r, k = after[int(rng.integers(0, len(after)))]
            cell = r * GRID + k
        else:
            cell = int(rng.integers(0, GRID * GRID))
        q = [SYM.kind(kind), SYM.transform(tf), SYM.cell(cell)]
    else:
        occupied = [r * GRID + k for _, _, r, k in items]
        a = occupied[int(rng.integers(0, len(occupied)))] if (occupied and rng.random() < 0.5) else int(rng.integers(0, GRID * GRID))
        b = int(rng.integers(0, GRID * GRID - 1))
        b = b + 1 if b >= a else b                                             # distinct from a
        q = [SYM.kind(kind), SYM.cell(a), SYM.cell(b)]
    return q, solve_query(scene, q), kind


def solve_query(scene: Scene, query: Sequence[int]) -> list[int]:
    """The exact answer of a query (total on well-formed queries). SPEC: 02 §2.9.4.

    WHERE with several matches answers the first cell in row-major order; SAME on
    two empty cells is YES.
    """
    q = [int(t) for t in query]
    kind = QUERY_KINDS[q[0] - sym(SYM.KIND)]
    sg, cg = scene.grid()
    if kind == "COUNT":
        colour, shape = q[1] - sym(SYM.COLOUR), q[2] - sym(SYM.SHAPE)
        return [SYM.count(int(((sg == shape) & (cg == colour)).sum()))]
    if kind == "WHERE":
        shape = q[1] - sym(SYM.SHAPE)
        hits = np.argwhere(sg == shape)
        if len(hits) == 0:
            return [sym(SYM.NONE)]
        r, k = hits[0]
        return [SYM.cell(int(r) * GRID + int(k))]
    if kind == "AFTER":
        tf = GRID_TRANSFORMS[q[1] - sym(SYM.TRANSFORM)]
        cell = q[2] - sym(SYM.CELL)
        sg2, _ = transform_scene(scene, tf).grid()
        s = sg2[cell // GRID, cell % GRID]
        return [sym(SYM.NONE)] if s < 0 else [SYM.shape(int(s))]
    if kind == "SAME":
        a, b = q[1] - sym(SYM.CELL), q[2] - sym(SYM.CELL)
        same = (sg[a // GRID, a % GRID] == sg[b // GRID, b % GRID]
                and cg[a // GRID, a % GRID] == cg[b // GRID, b % GRID])
        return [sym(SYM.YES) if same else sym(SYM.NO)]
    raise ValueError(kind)


def query_tokens(query: Sequence[int], answer: Sequence[int]) -> tuple[list[int], list[bool]]:
    """`[BOS] + query + [SEP] + answer + [EOS]` with the answer-only loss mask."""
    from soma.openended.tasks import BOS, EOS, SEP
    toks = [BOS] + list(query) + [SEP] + list(answer) + [EOS]
    start = 1 + len(query) + 1
    mask = [i >= start for i in range(len(toks))]
    return toks, mask


class SceneStream:
    """Frames at 5 Hz; a scene persists 4 frames then undergoes one random transform;
    a query with its exact answer is emitted with every 4th frame. SPEC: 02 §2.9.4."""

    RATE_HZ = 5.0
    FRAMES_PER_SCENE = 4

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(int(seed))
        self.scene = random_scene(self.rng)
        self.t = 0

    def next_frames(self, n: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, list[int], list[int], str]]]:
        """`(frames (n, 3, 32, 32), t_w (n,), queries)`: `queries` holds one
        `(frame_index, query, answer, kind)` per scene, attached to the scene's last frame."""
        frames = np.zeros((n, 3, IMG, IMG), dtype=np.float32)
        t_w = np.zeros(n, dtype=np.float64)
        queries = []
        for i in range(n):
            frames[i] = render(self.scene)
            t_w[i] = self.t / self.RATE_HZ
            if self.t % self.FRAMES_PER_SCENE == self.FRAMES_PER_SCENE - 1:
                q, a, kind = make_query(self.scene, self.rng)
                queries.append((i, q, a, kind))
                tf = GRID_TRANSFORMS[int(self.rng.integers(0, len(GRID_TRANSFORMS)))]
                self.scene = transform_scene(self.scene, tf)
                if not self.scene.items:
                    self.scene = random_scene(self.rng)
            self.t += 1
        return frames, t_w, queries


# =============================================================================
# Tone-coded speech. SPEC: 02 §2.9.5 (with the recorded deviation)
# =============================================================================

SAMPLE_RATE = 8000
FRAME = 128                     # samples per frame: 16 ms, 62.5 Hz
FRAMES_PER_SYMBOL = 4
N_TONES = N_SYMBOLS             # 56
N_BINS = 64
BIN_HZ = SAMPLE_RATE / FRAME    # 62.5 Hz per DFT bin


def tone_freq(s: int) -> float:
    """`f_s = 62.5 (s + 4)` Hz, on the frame's DFT grid, `250 .. 3687.5` Hz."""
    if not 0 <= s < N_TONES:
        raise ValueError(s)
    return BIN_HZ * (s + 4)


def symbol_waveform(s: int) -> np.ndarray:
    """`w(n) = hann(n) sin(2 pi f_s n / SR)` over the symbol's `4 x 128` samples."""
    n = FRAMES_PER_SYMBOL * FRAME
    t = np.arange(n, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * (t + 0.5) / n)
    return (hann * np.sin(2.0 * np.pi * tone_freq(s) * t / SAMPLE_RATE)).astype(np.float64)


def frame_features(wave: np.ndarray) -> np.ndarray:
    """`log(1 + |DFT_128(frame)|)` bins `0..63` per frame -> `(n_frames, 64)` float32."""
    if len(wave) % FRAME:
        raise ValueError("waveform length must be a multiple of FRAME")
    frames = wave.reshape(-1, FRAME)
    mag = np.abs(np.fft.rfft(frames, n=FRAME, axis=1))[:, :N_BINS]
    return np.log1p(mag).astype(np.float32)


def tone_frames(symbols: Sequence[int]) -> np.ndarray:
    """The frames `(4 n, 64)` of a symbol sequence."""
    if len(symbols) == 0:
        return np.zeros((0, N_BINS), dtype=np.float32)
    wave = np.concatenate([symbol_waveform(int(s)) for s in symbols])
    return frame_features(wave)


_BANK: np.ndarray | None = None


def tone_bank() -> np.ndarray:
    """`(56, 4, 64)`: the frames of every symbol."""
    global _BANK
    if _BANK is None:
        _BANK = np.stack([tone_frames([s]) for s in range(N_TONES)])
    return _BANK


def decode_frames(feats: np.ndarray) -> np.ndarray:
    """Nearest-tone decoding of `(4 n, 64)` frames, block by block -> `(n,)` symbols."""
    feats = np.asarray(feats, dtype=np.float32)
    if feats.shape[0] % FRAMES_PER_SYMBOL or feats.shape[1] != N_BINS:
        raise ValueError("frames must be (4 n, 64)")
    blocks = feats.reshape(-1, FRAMES_PER_SYMBOL * N_BINS)                    # (n, 256)
    bank = tone_bank().reshape(N_TONES, -1)
    d = ((blocks[:, None, :] - bank[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


# =============================================================================
# Pattern streams at the edge of chaos. SPEC: 02 §2.9.6
# =============================================================================

def logistic_stream(r: float, n: int, *, x0: float = 0.3, burn: int = 100, lags: int = 8
                    ) -> tuple[np.ndarray, float]:
    """`x_{k+1} = r x_k (1 - x_k)`; observation `(x_k, ..., x_{k-7})`; `lambda_ref` is the
    mean of `log |r (1 - 2 x_k)|` over the emitted run (`ln 2` at `r = 4`)."""
    if not 0.0 < r <= 4.0 or not 0.0 < x0 < 1.0:
        raise ValueError("r in (0, 4], x0 in (0, 1)")
    x = float(x0)
    for _ in range(burn):
        x = r * x * (1.0 - x)
    xs = np.empty(n + lags, dtype=np.float64)
    hist = [x]
    for _ in range(lags - 1):
        x = r * x * (1.0 - x)
        hist.append(x)
    xs[:lags] = hist
    lam = 0.0
    for k in range(n):
        x = r * x * (1.0 - x)
        xs[lags + k] = x
    seq = xs[lags:]
    lam = float(np.mean(np.log(np.abs(r * (1.0 - 2.0 * seq)) + 1e-300)))
    obs = np.stack([xs[lags + k - np.arange(lags)] for k in range(n)]).astype(np.float32)
    return obs, lam


def _lorenz_f(u: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = u
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def _lorenz_J(u: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = u
    return np.array([[-sigma, sigma, 0.0], [rho - z, -1.0, -x], [y, x, -beta]])


def lorenz_stream(n: int, *, dt: float = 0.01, x0: Sequence[float] = (1.0, 1.0, 1.0),
                  burn: int = 1000, sigma: float = 10.0, rho: float = 28.0,
                  beta: float = 8.0 / 3.0, scale: float = 20.0) -> tuple[np.ndarray, float]:
    """RK4 at `dt`; observation `(x, y, z) / 20`; `lambda_ref` by the Benettin method on
    the variational equation (tangent vector renormalised every step), `~ 0.9056`."""
    u = np.array(x0, dtype=np.float64)
    d = np.array([1.0, 0.0, 0.0])
    log_sum = 0.0
    obs = np.empty((n, 3), dtype=np.float32)

    def rk4(u, d):
        k1, l1 = _lorenz_f(u, sigma, rho, beta), _lorenz_J(u, sigma, rho, beta) @ d
        k2, l2 = (_lorenz_f(u + 0.5 * dt * k1, sigma, rho, beta),
                  _lorenz_J(u + 0.5 * dt * k1, sigma, rho, beta) @ (d + 0.5 * dt * l1))
        k3, l3 = (_lorenz_f(u + 0.5 * dt * k2, sigma, rho, beta),
                  _lorenz_J(u + 0.5 * dt * k2, sigma, rho, beta) @ (d + 0.5 * dt * l2))
        k4, l4 = (_lorenz_f(u + dt * k3, sigma, rho, beta),
                  _lorenz_J(u + dt * k3, sigma, rho, beta) @ (d + dt * l3))
        return (u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4),
                d + dt / 6.0 * (l1 + 2 * l2 + 2 * l3 + l4))

    for _ in range(burn):
        u, d = rk4(u, d)
        d = d / np.linalg.norm(d)
    for k in range(n):
        u, d = rk4(u, d)
        nd = np.linalg.norm(d)
        log_sum += np.log(nd)
        d = d / nd
        obs[k] = u / scale
    return obs, float(log_sum / (n * dt))


ECA_RULES = (250, 4, 90, 30, 110, 184)
WOLFRAM_CLASS = {250: 1, 4: 2, 90: 3, 30: 3, 110: 4, 184: 2}

LAMBDA_REF_STEPS = 100_000
_LAMBDA_CACHE: dict = {}


def lambda_ref(kind: str, **params) -> float:
    """The reference Lyapunov exponent of a generator, computed over `LAMBDA_REF_STEPS`
    iterations (not over a window: a 128-step estimate is noise, 11 §P-C.25) and cached:
    `logistic(r=)`: the mean of `log|r(1-2x)|` (`ln 2` at `r = 4`); `lorenz()`: the Benettin
    exponent, `0.9056`; `eca(rule=)`: the damage-spreading rate of `eca_stream`."""
    key = (kind, tuple(sorted((k, round(float(v), 6)) for k, v in params.items())))
    if key in _LAMBDA_CACHE:
        return _LAMBDA_CACHE[key]
    if kind == "logistic":
        _, lam = logistic_stream(float(params["r"]), LAMBDA_REF_STEPS, x0=0.3, burn=1000)
    elif kind == "lorenz":
        _, lam = lorenz_stream(LAMBDA_REF_STEPS, burn=1000)
    elif kind == "eca":
        _, lam = eca_stream(int(params["rule"]), 2, seed=0, trials=64)
    else:
        raise ValueError(kind)
    _LAMBDA_CACHE[key] = float(lam)
    return float(lam)


def eca_step(row: np.ndarray, rule: int) -> np.ndarray:
    """One step of elementary cellular automaton `rule` with periodic boundary."""
    left, right = np.roll(row, 1), np.roll(row, -1)
    idx = (left << 2) | (row << 1) | right
    table = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    return table[idx]


def eca_stream(rule: int, n: int, *, width: int = 64, seed: int = 0, trials: int = 32,
               horizon: int | None = None) -> tuple[np.ndarray, float]:
    """`(rows (n, width) uint8, lambda_ref)` with `lambda_ref` the damage-spreading rate
    (Bagnoli, Rechtman & Ruffo 1992): two copies differing in one cell are run for
    `horizon = width / 2` steps and

        lambda = mean_trials  mean_{t = 1..horizon}  (1/t) log max(H_t, 1/width),

    `H_t` the Hamming distance at step `t`: the finite-time spreading rate averaged over
    the horizon (a single-time rate is degenerate for additive rules such as 90, whose
    damage set has `H_t = 2^popcount(t)` and cancels under the periodic boundary at
    `t = width/2`). Negative when damage dies (class 1), near zero when it persists without
    spreading (class 2), positive when it spreads (classes 3-4)."""
    if not 0 <= rule < 256:
        raise ValueError(rule)
    rng = np.random.default_rng(int(seed))
    row = rng.integers(0, 2, size=width, dtype=np.uint8)
    rows = np.empty((n, width), dtype=np.uint8)
    for k in range(n):
        rows[k] = row
        row = eca_step(row, rule)
    T = width // 2 if horizon is None else int(horizon)
    lams = []
    for _ in range(trials):
        a = rng.integers(0, 2, size=width, dtype=np.uint8)
        b = a.copy()
        b[int(rng.integers(0, width))] ^= 1
        acc = 0.0
        for t in range(1, T + 1):
            a, b = eca_step(a, rule), eca_step(b, rule)
            h = float((a != b).sum())
            acc += np.log(max(h, 1.0 / width)) / t
        lams.append(acc / T)
    return rows, float(np.mean(lams))


# =============================================================================
# Multi-rate interleaving. SPEC: 02 §2.9.7
# =============================================================================

def interleave(streams: Iterable[Sequence[tuple[float, object]]]) -> list[tuple[float, int, object]]:
    """Merge `(t_w, payload)` sequences by wall-clock; ties keep stream order, then
    arrival order (a stable sort). Returns `(t_w, stream_index, payload)` triples."""
    merged = []
    for si, s in enumerate(streams):
        for j, (t, payload) in enumerate(s):
            merged.append((float(t), si, j, payload))
    merged.sort(key=lambda e: (e[0], e[1], e[2]))
    return [(t, si, payload) for t, si, _, payload in merged]
