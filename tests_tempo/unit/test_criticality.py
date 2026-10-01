"""Acceptance rows K1-K5 of docs/tempo/07 on the criticality probe (K6-K8 need the life
loop and the run records).

SPEC: 07 §7.7
"""

from __future__ import annotations

import math

import pytest
import torch

from tempo.backbone import Backbone
from tempo.config import TINY
from tempo.criticality import exponent_of_map, needs_control, probe, stack_exponents, zeta_update
from tempo.morphology import SwitchPattern


class TestK1K2:
    @pytest.mark.parametrize("kind,l", [("mix", 0), ("mix", 1), ("ffn", 0)])
    def test_k1_jvp_equals_central_differences(self, kind, l):
        cfg = TINY
        bb = Backbone(cfg)
        pat = SwitchPattern.adult(cfg)
        H = torch.randn(1, cfg.T_s, cfg.d)
        S = None
        if cfg.op(l) == "delta":
            S = torch.randn(1, cfg.h, cfg.d_head, cfg.d_head) * 0.1

        def f(X):
            return bb.sub_block(kind, l, X, S=S, pattern=pat, gains=None, zeta=None)

        delta = torch.randn(1, cfg.T_s, cfg.d)
        _, Jd = torch.func.jvp(f, (H,), (delta,))
        s = 1e-3
        fd = (f(H + s * delta) - f(H - s * delta)) / (2 * s)
        assert (Jd - fd).norm() / fd.norm() < 1e-4

    def test_k2_known_exponents(self):
        H = torch.randn(1, 8, 32)
        g = torch.Generator().manual_seed(0)
        for c in (0.5, 1.0, 2.0):
            lam = exponent_of_map(lambda X: c * X, H, n_delta=4, generator=g)
            assert abs(lam - math.log(c)) < 1e-6
        Q, _ = torch.linalg.qr(torch.randn(32, 32))
        lam = exponent_of_map(lambda X: X @ Q, H, n_delta=4, generator=g)
        assert abs(lam) < 1e-5

    def test_k2b_propagated_tangent_telescopes(self):
        """07 (7.2): the per-block exponents of the propagated tangent sum exactly to the
        exponent of the composed stack for the same perturbation."""
        cfg = TINY
        bb = Backbone(cfg)
        pat = SwitchPattern.primordium(cfg)
        h0 = torch.randn(1, cfg.T_s, cfg.d)
        state = {l: torch.randn(1, cfg.h, cfg.d_head, cfg.d_head) * 0.1 for l in bb.init_state(1)}
        res = stack_exponents(bb, h0, state, pattern=pat, gains=None, zeta=None, n_delta=1,
                              generator=torch.Generator().manual_seed(7))
        delta = torch.randn(h0.shape, generator=torch.Generator().manual_seed(7))

        def f_total(H):
            h = H
            for kind, l in res["keys"]:
                h = bb.sub_block(kind, l, h, S=None if kind == "ffn" else state.get(l), pattern=pat, gains=None, zeta=None)
            return h

        _, Jd = torch.func.jvp(f_total, (h0,), (delta,))
        direct = 0.5 * float(torch.log(Jd.square().sum()) - torch.log(delta.square().sum()))
        assert abs(sum(res["abs"]) - direct) < 1e-4
        assert len(res["abs"]) == 2 * len(pat.active_layers()) and len(res["rel"]) == len(res["abs"])
        # the relative form subtracts the stream's own growth: the two differ block by block
        assert any(abs(a - r) > 1e-6 for a, r in zip(res["abs"], res["rel"]))


class TestK3K4K5:
    def test_k3_a_skipped_layer_has_exponent_zero_and_is_not_probed(self):
        cfg = TINY
        bb = Backbone(cfg)
        pat = SwitchPattern.primordium(cfg)
        l_off = cfg.layers_of_quarter(1)[0]
        H = torch.randn(1, cfg.T_s, cfg.d)
        pat_zero = SwitchPattern(cfg, tuple(0.0 if k == l_off else m for k, m in enumerate(pat.layer_m)),
                                 pat.head_m, pat.slice_m)
        lam = exponent_of_map(lambda X: bb.sub_block("ffn", l_off, X, S=None, pattern=pat_zero, gains=None, zeta=None), H)
        assert lam == 0.0
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        prof = probe(bb, x, pattern=pat, n_delta=2, generator=torch.Generator().manual_seed(0))
        assert set(prof["mix"]) == set(cfg.primordium_layers) and set(prof["ffn"]) == set(cfg.primordium_layers)
        assert abs(prof["Lambda"] - (sum(prof["mix"].values()) + sum(prof["ffn"].values()))) < 1e-5
        assert abs(prof["Lambda_abs"] - (sum(prof["mix_abs"].values()) + sum(prof["ffn_abs"].values()))) < 1e-5
        assert math.isfinite(prof["Lambda_se"])

    def test_k3b_probe_uses_the_entering_states(self):
        cfg = TINY
        bb = Backbone(cfg)
        pat = SwitchPattern.primordium(cfg)
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        states = [{l: torch.randn(1, cfg.h, cfg.d_head, cfg.d_head) for l in bb.init_state(1)} for _ in range(2)]
        p0 = probe(bb, x, pattern=pat, n_delta=1, generator=torch.Generator().manual_seed(0))
        p1 = probe(bb, x, states=states, pattern=pat, n_delta=1, generator=torch.Generator().manual_seed(0))
        assert p0["Lambda"] != p1["Lambda"]

    def test_k4_rule_direction_clip_and_clamp(self):
        z = torch.zeros(2 * 4)
        prof = {"mix": {0: 0.4, 1: -3.0}, "ffn": {0: 0.0, 2: 5.0}}
        z2 = zeta_update(z, prof, kappa=0.5, clip=1.0, clamp=2.0)
        assert abs(z2[0] + 0.2) < 1e-6 and abs(z2[2] - 0.5) < 1e-6 and z2[1] == 0.0 and abs(z2[5] + 0.5) < 1e-6
        assert z2[3] == 0.0 and z2[4] == 0.0 and z2[6] == 0.0 and z2[7] == 0.0
        z3 = zeta_update(torch.full((8,), 1.9), {"mix": {0: -1.0}, "ffn": {}})
        assert z3[0] == 2.0
        with pytest.raises(FloatingPointError):
            zeta_update(z, {"mix": {0: float("nan")}, "ffn": {}})
        assert needs_control([{"Lambda": 0.1}, {"Lambda": -0.3}, {"Lambda": 0.6}])
        assert not needs_control([{"Lambda": 0.1, "mix": {0: 0.2}}, {"Lambda": -0.49}])
        assert needs_control([{"Lambda": 0.0, "mix": {1: 0.7}, "ffn": {}}])

    def test_k5_probe_runs_in_fp32_under_autocast(self):
        cfg = TINY
        bb = Backbone(cfg)
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prof = probe(bb, x, pattern=SwitchPattern.primordium(cfg), n_delta=1,
                         generator=torch.Generator().manual_seed(0))
        assert all(math.isfinite(v) for v in prof["mix"].values())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with pytest.raises(TypeError):
                exponent_of_map(lambda X: (X.to(torch.bfloat16) * 2), torch.randn(1, 4, 8))

    def test_profile_of_a_scaled_network(self):
        """A network whose every branch is turned off is the identity: Lambda = 0 in both
        forms; turning branches up moves it."""
        cfg = TINY
        bb = Backbone(cfg)
        x = torch.randn(2, cfg.T_s, cfg.d_x)
        pat = SwitchPattern.primordium(cfg)
        zeta_off = torch.full((2 * cfg.L,), -20.0)
        prof0 = probe(bb, x, pattern=pat, zeta=zeta_off, n_delta=2, generator=torch.Generator().manual_seed(0))
        assert abs(prof0["Lambda"]) < 1e-6 and abs(prof0["Lambda_abs"]) < 1e-6
        prof1 = probe(bb, x, pattern=pat, zeta=torch.full((2 * cfg.L,), 2.0), n_delta=2,
                      generator=torch.Generator().manual_seed(0))
        assert prof1["Lambda"] != 0.0
