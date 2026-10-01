"""Acceptance rows I9, I12, I14 of docs/tempo/02 on the perception stream.

SPEC: 02 §2.10
"""

from __future__ import annotations

import math

import pytest
import torch

from tempo.codec import GaussianHead
from tempo.modalities import (
    MODALITIES, MODALITY_INDEX, STREAMS, FrozenFeatureEncoder, ImageDecoder, ImageEncoder,
    Insulation, LinearEncoder, Perception, SpeechEncoder, TextIn, TimeCode, insulate,
)


class TestI9Insulation:
    def test_kappa_zero_blocks_the_backbone_and_not_the_head(self):
        y = torch.randn(6, 8, requires_grad=True)
        head = GaussianHead(8, 3)
        a = torch.randn(6, 3)
        head.nll(insulate(y, 0.0), a).sum().backward()
        assert y.grad is None or torch.count_nonzero(y.grad) == 0
        assert head.W_a.weight.grad is not None and head.W_a.weight.grad.abs().sum() > 0

    def test_kappa_one_equals_uninsulated(self):
        head = GaussianHead(8, 3)
        a = torch.randn(6, 3)
        y1 = torch.randn(6, 8, requires_grad=True)
        head.nll(y1, a).sum().backward()
        y2 = y1.detach().clone().requires_grad_(True)
        head.nll(insulate(y2, 1.0), a).sum().backward()
        assert torch.allclose(y1.grad, y2.grad)
        y3 = y1.detach().clone().requires_grad_(True)
        head.nll(insulate(y3, 0.25), a).sum().backward()
        assert torch.allclose(y3.grad, 0.25 * y1.grad, atol=1e-6)

    def test_rule(self):
        """02 §2.4: zero until the head's loss improved by half its first value, then a linear
        ramp over the following episode, then one; well posed for negative losses."""
        ins = Insulation()
        ins.update("act", 0, 10.0)
        assert ins.value("act", 0.5, episode=1) == 0.0
        ins.update("act", 1, 8.0)                          # improvement 2 < 5: not yet
        assert ins.value("act", 0.9, episode=2) == 0.0
        ins.update("act", 2, 4.9)                          # improvement 5.1 >= 5: the ramp fills episode 3
        assert ins.value("act", 0.0, episode=3) == 0.0
        assert abs(ins.value("act", 0.25, episode=3) - 0.25) < 1e-12
        assert ins.value("act", 1.0, episode=3) == 1.0 and ins.value("act", 0.1, episode=4) == 1.0
        neg = Insulation()
        neg.update("sp", 0, -100.0)
        neg.update("sp", 1, -120.0)                        # improvement 20 < 50
        assert "sp" not in neg.ramp_episode
        neg.update("sp", 2, -160.0)                        # improvement 60 >= 50
        assert neg.ramp_episode["sp"] == 3
        d = ins.state_dict()
        ins2 = Insulation()
        ins2.load_state_dict(d)
        assert ins2.value("act", 0.5, episode=3) == 0.5

    def test_linear_encoders_keep_magnitude_under_unit_rms(self):
        """11 §P-C.26: a bias-free linear map followed by unit-RMS would erase |raw|."""
        p = Perception(16)
        enc = LinearEncoder(1, 16)
        r1, r2 = torch.tensor([[2.0]]), torch.tensor([[4.0]])
        mod, t = torch.tensor([3]), torch.zeros(1, dtype=torch.float64)
        assert not torch.allclose(p(enc(r1), mod, t), p(enc(r2), mod, t), atol=1e-4)


class TestI12TimeCodes:
    def test_distinguishes_one_speech_frame_and_is_periodic(self):
        tc = TimeCode(16)
        t = torch.tensor([10.0, 10.0 + 1 / 62.5, 10.0 + 128.0], dtype=torch.float64)
        f = tc.features(t)
        assert (f[0] - f[1]).abs().max() > 1e-3
        assert (f[0] - f[2]).abs().max() < 1e-6
        assert tc(t).shape == (3, 16)


class TestI14ShapesAndRms:
    def test_perception_unit_rms_and_shape(self):
        p = Perception(16)
        feat = torch.randn(5, 16) * 7
        mod = torch.tensor([MODALITY_INDEX["text"], 1, 2, 3, 4])
        t_w = torch.linspace(0, 1, 5, dtype=torch.float64)
        x = p(feat, mod, t_w)
        assert x.shape == (5, 16)
        base = x - p.mu(mod) - p.tau(t_w)
        assert torch.allclose(base.square().mean(-1), torch.ones(5), atol=1e-5)
        with pytest.raises(Exception):
            p(torch.randn(5, 15), mod, t_w)

    def test_every_encoder_returns_d_x(self):
        d_x = 16
        assert TextIn(64, 4, d_x, d_e=8)(torch.randint(0, 64, (3, 4))).shape == (3, d_x)
        assert ImageEncoder(d_x)(torch.rand(3, 3, 32, 32)).shape == (3, d_x)
        assert SpeechEncoder(d_x)(torch.rand(3, 64)).shape == (3, d_x)
        assert SpeechEncoder(d_x)(torch.rand(3, 5, 64)).shape == (3, 5, d_x)
        assert LinearEncoder(8, d_x)(torch.rand(3, 8)).shape == (3, d_x)
        assert FrozenFeatureEncoder(1024, d_x)(torch.rand(3, 1024)).shape == (3, d_x)
        dec = ImageDecoder(d_x)
        assert dec(torch.randn(3, d_x)).shape == (3, 3, 32, 32)
        assert dec.nll(torch.randn(3, d_x), torch.rand(3, 3, 32, 32)).shape == (3,)

    def test_speech_encoder_is_per_frame(self):
        """The acting path (one frame) and the training path (a window) perceive a frame
        identically (11 §P-C.26)."""
        enc = SpeechEncoder(16)
        f = torch.rand(2, 6, 64)
        per_window = enc(f)
        per_frame = torch.stack([enc(f[:, i]) for i in range(6)], 1)
        assert torch.allclose(per_window, per_frame)

    def test_shape_assertions_fire(self):
        with pytest.raises(Exception):
            ImageEncoder(16)(torch.rand(2, 3, 16, 16))
        with pytest.raises(Exception):
            TextIn(64, 4, 16, d_e=8)(torch.randint(0, 64, (3, 3)))
        with pytest.raises(Exception):
            LinearEncoder(8, 16)(torch.rand(3, 7))

    def test_registry(self):
        assert len(STREAMS) == 7
        assert {m.stream for m in MODALITIES} == set(STREAMS) - {"interleave"}
        assert MODALITIES[MODALITY_INDEX["speech"]].rate_hz == 62.5
        assert MODALITIES[MODALITY_INDEX["prop"]].d_raw == 8
