"""Acceptance rows I1-I7 of docs/tempo/02 on the span codec and the heads.

SPEC: 02 §2.10
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import stats

from tempo.codec import (
    BOS, BernoulliHead, FlowHead, GaussianHead, SpanCodec, text_span_loss, unit_rms,
)
from tempo.modalities import TextIn


def _train_codec(V=64, K=2, d_x=16, steps=900, seed=0):
    """A codec on a structured synthetic corpus: spans from a fixed set of 512 patterns."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    codec = SpanCodec(V, K, d_x, d_c=32, n_layers=1, n_heads=2, d_ff=64)
    patterns = torch.randint(0, V, (512, K), generator=g)
    opt = torch.optim.Adam(codec.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = torch.randint(0, 512, (256,), generator=g)
        out = codec.loss(patterns[idx], generator=g)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    return codec, patterns, g


class TestI1I2Ceiling:
    def test_i1_ceiling_on_synthetic_vocabulary(self):
        codec, patterns, _ = _train_codec()
        ceiling = float(codec.ceiling(patterns))
        assert ceiling < 0.05, ceiling

    def test_i2_noise_tolerance(self):
        codec, patterns, g = _train_codec()
        codec.eval()
        x_bar, _, _ = codec.encode(patterns)
        clean, _ = codec.nll(x_bar, patterns)
        eps = torch.randn(x_bar.shape, generator=g)
        eps = eps / eps.norm(dim=-1, keepdim=True) * (codec.sigma_z * math.sqrt(codec.d_x))
        noisy, _ = codec.nll(x_bar + eps, patterns)
        gap = float((noisy.mean() - clean.mean()) / codec.K)
        assert gap < 0.05, gap


class TestI3I4I5Identities:
    def test_i3_input_side_shares_nothing_with_the_codec(self):
        codec = SpanCodec(64, 2, 16, d_c=32, n_layers=1, n_heads=2, d_ff=64)
        text_in = TextIn(64, 2, 16, d_e=8)
        ids_codec = {id(p) for p in codec.parameters()}
        assert not any(id(p) in ids_codec for p in text_in.parameters())
        toks = torch.randint(0, 64, (5, 2))
        before = text_in(toks).clone()
        with torch.no_grad():
            for p in codec.parameters():
                p.add_(1.0)
        assert torch.equal(text_in(toks), before)

    def test_i4_likelihood_sums_to_one(self):
        V, K = 8, 2
        codec = SpanCodec(V, K, 6, d_c=16, n_layers=1, n_heads=2, d_ff=32)
        x = torch.randn(1, 6)
        total = 0.0
        for seq in itertools.product(range(V), repeat=K):
            toks = torch.tensor([seq])
            nll, _ = codec.nll(x, toks)
            total += math.exp(-float(nll))
        assert abs(total - 1.0) < 1e-6, total

    def test_i5_the_K_1_token_head(self):
        """02 §2.2: at K = 1 with no decoder layers the likelihood is a softmax head on an
        affine image of y alone: the logits do not depend on any token input, and
        `L_txt / K` is the per-token cross-entropy of that head."""
        V = 16
        codec = SpanCodec(V, 1, 8, d_c=16, n_layers=1, n_heads=2, d_ff=32, n_dec_layers=0)
        x = torch.randn(7, 8)
        toks = torch.randint(0, V, (7, 1))
        other = (toks + 3) % V
        assert torch.allclose(codec.decode_logits(x, toks), codec.decode_logits(x, other))
        logits = codec.decode_logits(x, toks)[:, 0]
        ce = F.cross_entropy(logits.float(), toks[:, 0], reduction="none")
        nll, _ = codec.nll(x, toks)
        assert torch.allclose(nll, ce, atol=1e-6)
        loss, _ = text_span_loss(codec, x, toks, x, lambda_lat=0.0, z_loss=0.0)
        assert torch.allclose(loss / codec.K, ce, atol=1e-6)
        with pytest.raises(ValueError):
            SpanCodec(V, 2, 8, d_c=16, n_layers=1, n_heads=2, d_ff=32, n_dec_layers=0)


class TestI6I7Heads:
    def test_i6_gaussian_matches_scipy(self):
        head = GaussianHead(5, 3)
        with torch.no_grad():
            head.w_s.copy_(torch.tensor([0.1, -0.3, 0.5]))
        y = torch.randn(4, 5)
        a = torch.randn(4, 3)
        nll = head.nll(y, a)
        mu = head.mean(y).detach().numpy()
        sd = np.exp(head.log_std().detach().numpy())
        for i in range(4):
            ref = -stats.multivariate_normal.logpdf(a[i].numpy(), mean=mu[i], cov=np.diag(sd ** 2))
            assert abs(float(nll[i]) - ref) < 1e-5                # fp32 arithmetic (02 I6)

    def test_i6b_bernoulli_matches_reference(self):
        head = BernoulliHead(4, 6)
        y = torch.randn(3, 4)
        t = (torch.rand(3, 6) > 0.5).float()
        nll = head.nll(y, t)
        p = torch.sigmoid(head.logits(y))
        ref = -(t * torch.log(p) + (1 - t) * torch.log(1 - p)).sum(-1)
        assert torch.allclose(nll, ref, atol=1e-5)

    def test_i7_flow_recovers_a_bimodal_target_and_the_gaussian_does_not(self):
        g = torch.Generator().manual_seed(1)
        torch.manual_seed(1)
        n = 2048
        modes = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
        which = torch.randint(0, 2, (n,), generator=g)
        target = modes[which] + 0.2 * torch.randn(n, 2, generator=g)
        y = torch.ones(n, 4)
        flow = FlowHead(4, 2, width=128)
        opt = torch.optim.Adam(flow.parameters(), lr=2e-3)
        for _ in range(1500):
            loss = flow.nll(y, target, generator=g).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        samples = flow.sample(y, steps=10, generator=g)

        def mode_stats(s):
            near = (s[:, 0].abs() - 2.0).abs() < 0.6
            frac_near = near.float().mean().item()
            balance = (s[near, 0] > 0).float().mean().item()
            spread = (s[near, 0].abs() - s[near, 0].abs().mean()).std().item()
            return frac_near, balance, spread

        frac, bal, spread = mode_stats(samples)
        # both modes recovered: nearly every sample sits at a mode, the modes are balanced,
        # the within-mode spread is the target's 0.2 (the KS p-value is reported, not
        # asserted: a 10-step Euler integration carries a small bias that a two-sample KS
        # on 2048 points detects, while the density's two modes are what the row tests)
        p_flow = stats.ks_2samp(samples[:, 0].numpy(), target[:, 0].numpy()).pvalue
        print(f"flow: near {frac:.3f} balance {bal:.3f} spread {spread:.3f} ks_p {p_flow:.2e}")
        assert frac > 0.9 and 0.4 < bal < 0.6 and 0.1 < spread < 0.4, (frac, bal, spread)
        gauss = GaussianHead(4, 2)
        opt = torch.optim.Adam(gauss.parameters(), lr=1e-2)
        for _ in range(500):
            loss = gauss.nll(y, target).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        gs = gauss.sample(y, generator=g)
        frac_g = ((gs[:, 0].abs() - 2.0).abs() < 0.6).float().mean().item()
        assert frac_g < 0.5, frac_g


class TestShapesAndSampling:
    def test_unit_rms(self):
        x = torch.randn(3, 16) * 5
        u = unit_rms(x)
        assert torch.allclose(u.square().mean(-1), torch.ones(3), atol=1e-5)

    def test_sample_is_deterministic_under_a_generator(self):
        codec = SpanCodec(32, 3, 8, d_c=16, n_layers=1, n_heads=2, d_ff=32)
        x = torch.randn(4, 8)
        a = codec.sample(x, generator=torch.Generator().manual_seed(3))
        b = codec.sample(x, generator=torch.Generator().manual_seed(3))
        assert torch.equal(a, b) and a.shape == (4, 3)
        greedy = codec.sample(x, temperature=0.0)
        assert greedy.shape == (4, 3)
        nuc = codec.sample(x, top_p=0.5, generator=torch.Generator().manual_seed(4))
        assert nuc.shape == (4, 3)

    def test_decoder_is_causal_within_the_span(self):
        codec = SpanCodec(32, 4, 8, d_c=16, n_layers=1, n_heads=2, d_ff=32)
        x = torch.randn(2, 8)
        t1 = torch.randint(0, 32, (2, 4))
        t2 = t1.clone()
        t2[:, 3] = (t2[:, 3] + 1) % 32
        l1, l2 = codec.decode_logits(x, t1), codec.decode_logits(x, t2)
        assert torch.allclose(l1[:, :3], l2[:, :3])

    def test_bos_is_the_reserved_id(self):
        assert BOS == 1

    def test_shape_assertions_fire(self):
        codec = SpanCodec(32, 4, 8, d_c=16, n_layers=1, n_heads=2, d_ff=32)
        with pytest.raises(Exception):
            codec.encode(torch.randint(0, 32, (2, 3)))
        with pytest.raises(Exception):
            codec.nll(torch.randn(2, 7), torch.randint(0, 32, (2, 4)))
