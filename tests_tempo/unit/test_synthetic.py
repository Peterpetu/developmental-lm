"""Acceptance row I10 of docs/tempo/02 on the generated streams: determinism, exact
answers against an independent solver, exact tone decoding, the reference Lyapunov
exponents, and the interleaver.

SPEC: 02 §2.9.4-2.9.7, §2.10
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from soma.openended.tasks import GRID_TRANSFORMS, sym
from tempo import synthetic as S


# ----------------------------------------------------------------- an independent scene reader
def _read_scene_from_pixels(img: np.ndarray) -> S.Scene:
    """Detect each cell's shape by pixel count and colour by the lit colour: independent
    of `Scene.grid` and of the masks' identities (counts 32 / 36 / 18)."""
    counts = {int(m.sum()): i for i, m in enumerate(S._MASKS)}
    assert len(counts) == 3
    items = []
    for r in range(S.GRID):
        for k in range(S.GRID):
            cell = img[:, r * S.CELL:(r + 1) * S.CELL, k * S.CELL:(k + 1) * S.CELL]
            lit = cell.sum(0) > 0
            n = int(lit.sum())
            if n == 0:
                continue
            col = tuple(float(v) for v in cell[:, lit][:, 0])
            colour = S.COLOURS.index(col)
            items.append((counts[n], colour, r, k))
    return S.Scene(tuple(items))


def _transform_image(img: np.ndarray, tf: str) -> np.ndarray:
    if tf in ("identity", "tile"):
        return img.copy()
    if tf in ("rot90", "rot180", "rot270"):
        k = {"rot90": 1, "rot180": 2, "rot270": 3}[tf]
        return np.rot90(img, k, axes=(1, 2)).copy()
    if tf == "flip_h":
        return img[:, :, ::-1].copy()
    if tf == "flip_v":
        return img[:, ::-1, :].copy()
    if tf == "translate":
        return np.roll(np.roll(img, S.CELL, axis=1), S.CELL, axis=2)
    if tf == "recolour":
        out = np.zeros_like(img)
        for r in range(S.GRID):
            for k in range(S.GRID):
                cell = img[:, r * S.CELL:(r + 1) * S.CELL, k * S.CELL:(k + 1) * S.CELL]
                lit = cell.sum(0) > 0
                if lit.any():
                    col = tuple(float(v) for v in cell[:, lit][:, 0])
                    nxt = S.COLOURS[(S.COLOURS.index(col) + 1) % 6]
                    for ch in range(3):
                        out[ch, r * S.CELL:(r + 1) * S.CELL, k * S.CELL:(k + 1) * S.CELL][lit] = nxt[ch]
        return out
    raise ValueError(tf)


def _solve_from_pixels(img: np.ndarray, query) -> list[int]:
    q = list(query)
    kind = S.QUERY_KINDS[q[0] - sym(S.SYM.KIND)]
    scene = _read_scene_from_pixels(img)
    if kind == "AFTER":
        tf = S.GRID_TRANSFORMS[q[1] - sym(S.SYM.TRANSFORM)] if hasattr(S, "GRID_TRANSFORMS") \
            else GRID_TRANSFORMS[q[1] - sym(S.SYM.TRANSFORM)]
        scene2 = _read_scene_from_pixels(_transform_image(img, tf))
        cell = q[2] - sym(S.SYM.CELL)
        hits = [it for it in scene2.items if it[2] * S.GRID + it[3] == cell]
        return [sym(S.SYM.NONE)] if not hits else [S.SYM.shape(hits[0][0])]
    if kind == "COUNT":
        colour, shape = q[1] - sym(S.SYM.COLOUR), q[2] - sym(S.SYM.SHAPE)
        return [S.SYM.count(sum(1 for it in scene.items if it[0] == shape and it[1] == colour))]
    if kind == "WHERE":
        shape = q[1] - sym(S.SYM.SHAPE)
        hits = sorted((it for it in scene.items if it[0] == shape), key=lambda it: (it[2], it[3]))
        return [sym(S.SYM.NONE)] if not hits else [S.SYM.cell(hits[0][2] * S.GRID + hits[0][3])]
    a, b = q[1] - sym(S.SYM.CELL), q[2] - sym(S.SYM.CELL)
    at = {it[2] * S.GRID + it[3]: (it[0], it[1]) for it in scene.items}
    same = at.get(a) == at.get(b)
    return [sym(S.SYM.YES) if same else sym(S.SYM.NO)]


class TestScenes:
    def test_deterministic_from_seed(self):
        a = S.SceneStream(7).next_frames(20)
        b = S.SceneStream(7).next_frames(20)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]

    def test_answers_equal_an_independent_pixel_solver(self):
        rng = np.random.default_rng(11)
        n_by_kind = {k: 0 for k in S.QUERY_KINDS}
        for _ in range(300):
            scene = S.random_scene(rng)
            img = S.render(scene)
            assert _read_scene_from_pixels(img) == scene
            q, a, kind = S.make_query(scene, rng)
            assert a == _solve_from_pixels(img, q), (kind, q, a)
            n_by_kind[kind] += 1
        assert min(n_by_kind.values()) > 20

    def test_transforms_match_numpy_semantics_on_the_cell_grid(self):
        rng = np.random.default_rng(3)
        scene = S.random_scene(rng)
        sg, cg = scene.grid()
        for tf in GRID_TRANSFORMS:
            sg2, cg2 = S.transform_scene(scene, tf).grid()
            img2 = _transform_image(S.render(scene), tf)
            assert _read_scene_from_pixels(img2).grid()[0].tolist() == sg2.tolist(), tf
            assert _read_scene_from_pixels(img2).grid()[1].tolist() == cg2.tolist(), tf

    def test_stream_rate_and_queries(self):
        frames, t_w, queries = S.SceneStream(1).next_frames(12)
        assert frames.shape == (12, 3, 32, 32) and frames.min() >= 0 and frames.max() <= 1
        assert np.allclose(np.diff(t_w), 1 / 5.0)
        assert [q[0] for q in queries] == [3, 7, 11]
        toks, mask = S.query_tokens(queries[0][1], queries[0][2])
        assert sum(mask) == len(queries[0][2]) + 1


class TestTones:
    def test_frequencies_below_nyquist_and_on_the_bin_grid(self):
        for s in range(S.N_TONES):
            f = S.tone_freq(s)
            assert f < S.SAMPLE_RATE / 2
            assert abs(f / S.BIN_HZ - round(f / S.BIN_HZ)) < 1e-9

    def test_decoding_is_exact_for_every_symbol_sequence(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            symbols = rng.integers(0, S.N_TONES, size=int(rng.integers(1, 12)))
            frames = S.tone_frames(symbols)
            assert frames.shape == (4 * len(symbols), 64)
            assert np.array_equal(S.decode_frames(frames), symbols)
        allsym = np.arange(S.N_TONES)
        assert np.array_equal(S.decode_frames(S.tone_frames(allsym)), allsym)

    def test_frame_rate(self):
        assert S.SAMPLE_RATE / S.FRAME == 62.5


class TestPatterns:
    def test_logistic_lyapunov_at_r_4_is_ln2(self):
        obs, lam = S.logistic_stream(4.0, 100_000, x0=0.31)
        assert obs.shape == (100_000, 8)
        assert abs(lam - math.log(2.0)) < 0.01, lam
        assert np.allclose(obs[1, 1:], obs[0, :-1])          # lag structure

    def test_logistic_periodic_regime_is_negative(self):
        _, lam = S.logistic_stream(3.5, 5_000)
        assert lam < 0

    def test_lorenz_lyapunov(self):
        obs, lam = S.lorenz_stream(100_000)
        assert obs.shape == (100_000, 3)
        assert abs(lam - 0.9056) < 0.015, lam

    def test_eca_classes_by_damage_spreading(self):
        lam = {rule: S.eca_stream(rule, 10, seed=1)[1] for rule in S.ECA_RULES}
        # classes 3-4 spread; class 1 dies; class 2 (rule 4: isolated cells, rule 184:
        # traffic particles) neither spreads nor grows: at or below zero, or small
        assert lam[30] > 0.02 and lam[90] > 0.02 and lam[110] > 0.02
        assert lam[250] < -0.02
        assert lam[4] <= 0.0 and abs(lam[184]) < 0.1
        assert max(lam[4], lam[184]) < min(lam[30], lam[90], lam[110])

    def test_eca_step_rule_90_is_xor_of_neighbours(self):
        row = np.random.default_rng(2).integers(0, 2, size=64).astype(np.uint8)
        nxt = S.eca_step(row, 90)
        assert np.array_equal(nxt, np.roll(row, 1) ^ np.roll(row, -1))

    def test_determinism(self):
        a = S.eca_stream(110, 50, seed=5)[0]
        b = S.eca_stream(110, 50, seed=5)[0]
        assert np.array_equal(a, b)


class TestInterleave:
    def test_merge_by_wall_clock_with_stable_ties(self):
        fast = [(k / 62.5, ("sp", k)) for k in range(32)]
        slow = [(k / 4.0, ("tx", k)) for k in range(3)]
        merged = S.interleave([slow, fast])
        assert [t for t, _, _ in merged] == sorted(t for t, _, _ in merged)
        assert merged[0][2] == ("tx", 0) and merged[1][2] == ("sp", 0)      # tie: stream order
        between = [p for t, si, p in merged if 0.0 < t < 0.25]
        assert len(between) == 15 and all(p[0] == "sp" for p in between)
