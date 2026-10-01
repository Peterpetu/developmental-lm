"""The span codec, the text likelihood, and the continuous heads.

SPEC: 02 §2.2  (the output codec: encoder E_txt, decoder D_txt, the codec loss)
SPEC: 02 §2.3  (decoding a span from a predicted latent)
SPEC: 02 §2.4  (the text loss (2.6), Gaussian (2.7), Bernoulli (2.8), flow (2.9))
SPEC: 02 §2.5  (the continuous heads)

The codec is trained once (R0) on token spans and frozen; the backbone's text
target is its unit-RMS latent and the backbone's text loss is a proper
likelihood through its within-span decoder. The heads are the decoders of the
continuous modalities; each exposes `nll` (a proper negative log-likelihood or,
for the flow head, the flow-matching loss) and `sample`.

Deviation from 02 (2.5), recorded (11 §P-C.15): the training latent fed to the
decoder is the reparameterised posterior sample `z = mu_z + s_z * eps_1`,
normalised, plus the fixed noise `sigma_z * eps_2` and the coordinate dropout;
the deterministic target `x_bar = mu_z / RMS(mu_z)` is what the backbone
predicts. With the noise only on the deterministic latent, `s_z` would enter the
loss through the KL alone and the KL would drive it to a constant: the posterior
scale must be *used* by the decoder for the VAE regulariser to be one.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from anlage.ops import SOFTCAP, Z_LOSS
from gnomon.utils.shapes import check_shape

__all__ = [
    "BOS", "unit_rms", "SpanCodec", "text_span_loss", "GaussianHead", "BernoulliHead",
    "FlowHead", "SIGMA_Z", "BETA_KL", "KL_FLOOR", "P_LATENT_DROP", "LAMBDA_LAT",
]

BOS = 1
"""The reserved BOS id of the tokenizer (soma/openended/tasks.py); `v_0` of (2.4)."""
SIGMA_Z = 0.3
"""SPEC: 02 (2.5) -- the decoder is trained under Gaussian noise of this scale."""
BETA_KL = 1e-3
"""SPEC: 02 (2.5) -- the weight of the KL regulariser."""
KL_FLOOR = 0.5
"""SPEC: 02 (2.5) -- the per-dimension floor of the KL (free bits)."""
P_LATENT_DROP = 0.15
"""SPEC: 02 (2.5) -- the fraction of latent coordinates zeroed during codec training."""
LAMBDA_LAT = 0.0
"""SPEC: 02 (2.6) -- the latent regression weight of the text loss: 0 (11 §P-C.23)."""
LAMBDA_LAT_ABLATION = 0.1
"""The plan's value, the recorded ablation (it cost 0.223 nats at 25M tokens)."""


def unit_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """`x / RMS(x)` over the last dimension, computed in fp32. SPEC: 02 (2.3)."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return out.to(x.dtype)


class _Block(nn.Module):
    """A pre-LN transformer block (GELU FFN); bidirectional or causal. SPEC: 02 (2.3)-(2.4)."""

    def __init__(self, d: int, n_heads: int, d_ff: int, *, causal: bool):
        super().__init__()
        if d % n_heads:
            raise ValueError("d must be a multiple of n_heads")
        self.d, self.n_heads, self.causal = d, n_heads, causal
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, K, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(N, K, 3, self.n_heads, d // self.n_heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                     # (N, H, K, dh)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = x + self.proj(a.transpose(1, 2).reshape(N, K, d))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class SpanCodec(nn.Module):
    """`E_txt` (2.3) and `D_txt` (2.4) over spans of `K` tokens. SPEC: 02 §2.2."""

    def __init__(self, vocab: int, K: int, d_x: int, *, d_c: int = 256, n_layers: int = 2,
                 n_heads: int = 4, d_ff: int = 1024, sigma_z: float = SIGMA_Z,
                 beta: float = BETA_KL, kl_floor: float = KL_FLOOR,
                 p_drop: float = P_LATENT_DROP, n_dec_layers: Optional[int] = None):
        """`n_dec_layers = 0` is the token head of the `K = 1` fallback (02 §2.2): the
        logits are `softcap(W_v LN(T''[BOS] + P' + W_c y))`, a softmax head on a
        `d_c`-wide affine image of `y` with no within-span recurrence."""
        super().__init__()
        if K < 1 or vocab < 2 or d_x < 1:
            raise ValueError("K >= 1, vocab >= 2, d_x >= 1")
        if n_dec_layers is None:
            n_dec_layers = n_layers
        if n_dec_layers == 0 and K != 1:
            raise ValueError("a decoder without layers is the K = 1 token head only")
        self.n_dec_layers = n_dec_layers
        self.vocab, self.K, self.d_x, self.d_c = vocab, K, d_x, d_c
        self.sigma_z, self.beta, self.kl_floor, self.p_drop = sigma_z, beta, kl_floor, p_drop
        # encoder (2.3)
        self.T_enc = nn.Embedding(vocab, d_c)
        self.P_enc = nn.Parameter(torch.zeros(K, d_c))
        self.enc = nn.ModuleList([_Block(d_c, n_heads, d_ff, causal=False) for _ in range(n_layers)])
        self.ln_enc = nn.LayerNorm(d_c)
        self.W_z = nn.Linear(d_c, d_x, bias=False)
        self.W_s = nn.Linear(d_c, d_x, bias=False)
        # decoder (2.4)
        self.T_dec = nn.Embedding(vocab, d_c)
        self.P_dec = nn.Parameter(torch.zeros(K, d_c))
        self.W_c = nn.Linear(d_x, d_c, bias=False)
        self.dec = nn.ModuleList([_Block(d_c, n_heads, d_ff, causal=True) for _ in range(n_dec_layers)])
        self.ln_dec = nn.LayerNorm(d_c)
        self.W_v = nn.Linear(d_c, vocab, bias=False)
        for emb in (self.T_enc, self.T_dec):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.P_enc, std=0.02)
        nn.init.normal_(self.P_dec, std=0.02)

    # ------------------------------------------------------------------ encoder
    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(x_bar, mu_z, log_s)`: the unit-RMS target latent and the posterior. SPEC: 02 (2.3)."""
        check_shape(tokens, [None, self.K], name="tokens")
        u = self.T_enc(tokens) + self.P_enc[None]
        for blk in self.enc:
            u = blk(u)
        pooled = self.ln_enc(u).mean(1)                                       # (N, d_c)
        mu = self.W_z(pooled)
        log_s = self.W_s(pooled).clamp(-8.0, 4.0)
        return unit_rms(mu.float()), mu, log_s                                # the target stays fp32 (09 §9.5)

    # ------------------------------------------------------------------ decoder
    def decode_logits(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Logits `(N, K, V)` of `p(v_i | x, v_{<i})` with teacher forcing. SPEC: 02 (2.4)."""
        N = tokens.shape[0]
        check_shape(tokens, [N, self.K], name="tokens")
        check_shape(x, [N, self.d_x], name="latent")
        prev = torch.cat([tokens.new_full((N, 1), BOS), tokens[:, :-1]], dim=1)
        c = self.T_dec(prev) + self.P_dec[None] + self.W_c(x)[:, None, :]
        for blk in self.dec:
            c = blk(c)
        z = self.W_v(self.ln_dec(c))
        return SOFTCAP * torch.tanh(z / SOFTCAP)

    def nll(self, x: torch.Tensor, tokens: torch.Tensor, *, z_loss: float = 0.0
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(-sum_i log p(v_i | x, v_<i)) (N,)` and the z-loss per row `(N,)`,
        `z_loss * sum_i lse_i^2` (the same per-token convention as the NLL, so the caller masks
        and reduces both alike). SPEC: 02 (2.4), (2.6)."""
        logits = self.decode_logits(x, tokens).float()
        lse = torch.logsumexp(logits, dim=-1)                                 # (N, K)
        picked = logits.gather(-1, tokens[..., None]).squeeze(-1)
        nll = (lse - picked).sum(1)
        zl = z_loss * (lse ** 2).sum(1) if z_loss > 0 else torch.zeros_like(nll)
        return nll, zl

    # --------------------------------------------------------------------- loss
    def loss(self, tokens: torch.Tensor, *, generator: Optional[torch.Generator] = None
             ) -> dict[str, torch.Tensor]:
        """The codec loss (2.5) on a batch of spans; keys `loss`, `nll_token`, `kl`."""
        x_bar, mu, log_s = self.encode(tokens)
        s = log_s.exp()
        eps1 = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        eps2 = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        z = unit_rms(mu + s * eps1) + self.sigma_z * eps2
        if self.p_drop > 0:
            keep = torch.rand(mu.shape, generator=generator, device=mu.device) >= self.p_drop
            z = z * keep.to(z.dtype)
        nll, _ = self.nll(z, tokens)
        kl = 0.5 * (mu.float().square() + s.float().square() - 2.0 * log_s.float() - 1.0)
        kl_dim = kl.mean(0)                                                    # (d_x,)
        kl_floored = torch.clamp(kl_dim, min=self.kl_floor).sum()
        total = nll.mean() + self.beta * kl_floored
        return {"loss": total, "nll_token": nll.mean() / self.K, "kl": kl_dim.sum(),
                "x_bar": x_bar}

    @torch.no_grad()
    def ceiling(self, tokens: torch.Tensor) -> torch.Tensor:
        """Per-token NLL on the true (noise-free) latents: the codec ceiling of I1."""
        x_bar, _, _ = self.encode(tokens)
        nll, _ = self.nll(x_bar, tokens)
        return nll.mean() / self.K

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, x: torch.Tensor, *, temperature: float = 1.0, top_p: float = 1.0,
               generator: Optional[torch.Generator] = None, forbid_reserved: int = 0) -> torch.Tensor:
        """Sample `v_1..v_K` autoregressively within the span from `x`. SPEC: 02 §2.3.

        `temperature = 0` is greedy decoding; `top_p < 1` is nucleus sampling;
        `forbid_reserved = n` masks the ids `0..n-1` (natural text never emits the reserved
        block, 02 §2.2; the verifier streams keep it as their alphabet). The draws use the
        supplied CPU generator whatever the tensor's device.
        """
        check_shape(x, [None, self.d_x], name="latent")
        N = x.shape[0]
        out = x.new_zeros((N, self.K), dtype=torch.long)
        for i in range(self.K):
            logits = self.decode_logits(x, out)[:, i].float()
            if forbid_reserved > 0:
                logits[:, :forbid_reserved] = float("-inf")
            if temperature <= 0:
                out[:, i] = logits.argmax(-1)
                continue
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sp, si = probs.sort(dim=-1, descending=True)
                cum = sp.cumsum(-1)
                drop = cum - sp > top_p
                sp = sp.masked_fill(drop, 0.0)
                sp = sp / sp.sum(-1, keepdim=True)
                pick = torch.multinomial(sp.cpu(), 1, generator=generator).to(x.device)
                out[:, i] = si.gather(-1, pick).squeeze(-1)
            else:
                out[:, i] = torch.multinomial(probs.cpu(), 1, generator=generator).to(x.device).squeeze(-1)
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encoder_parameters(self) -> list[nn.Parameter]:
        """`E_txt`: frozen after R0; it defines the target latent `x_bar`. SPEC: 02 §2.2."""
        mods = [self.T_enc, self.P_enc, self.enc, self.ln_enc, self.W_z, self.W_s]
        out = []
        for m in mods:
            out += [m] if isinstance(m, nn.Parameter) else list(m.parameters())
        return out

    @torch.no_grad()
    def reinit_decoder(self, seed: int = 0) -> None:
        """Re-initialise `D_txt` as if R0 had never run, keeping the frozen encoder.

        11 §P-D.19: R0 fits the decoder to decode `x_bar`, the encoder's latent on the unit-RMS
        sphere; the life then asks the same weights to decode `y_hat`, the backbone's
        prediction. That is a warm start on a different input distribution, and it puts a prior
        on the geometry the backbone must produce. A token model's unembed has no such prior.
        This makes the two comparable, and is a diagnostic rather than a design change.
        """
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        for emb in (self.T_dec,):
            emb.weight.copy_(torch.randn(emb.weight.shape, generator=g) * 0.02)
        self.P_dec.copy_(torch.randn(self.P_dec.shape, generator=g) * 0.02)
        for lin in (self.W_c, self.W_v):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * lin.weight.shape[1] ** -0.5)
            if lin.bias is not None:
                lin.bias.zero_()
        self.ln_dec.reset_parameters()
        for blk in self.dec:
            for p_ in blk.parameters():
                if p_.dim() >= 2:
                    p_.copy_(torch.randn(p_.shape, generator=g) * p_.shape[-1] ** -0.5)
                else:
                    p_.zero_()

    def decoder_parameters(self) -> list[nn.Parameter]:
        """`D_txt`: initialised by R0 and trained at tau_1 with the backbone as the conditional
        language model of the span given `y_hat` (11 §P-C.21). SPEC: 02 §2.2."""
        mods = [self.T_dec, self.P_dec, self.W_c, self.dec, self.ln_dec, self.W_v]
        out = []
        for m in mods:
            out += [m] if isinstance(m, nn.Parameter) else list(m.parameters())
        return out


def text_span_loss(codec: SpanCodec, y_hat: torch.Tensor, tokens: torch.Tensor,
                   x_bar: torch.Tensor, *, lambda_lat: float = LAMBDA_LAT,
                   z_loss: float = Z_LOSS) -> tuple[torch.Tensor, torch.Tensor]:
    """`L_txt` of (2.6) per span `(N,)` and the z-loss per span `(N,)`.

    `L_txt / K` is a per-token NLL of the same tokens a token model scores (I5).
    """
    nll, zl = codec.nll(y_hat, tokens, z_loss=z_loss)
    lat = (y_hat.float() - x_bar.float()).square().sum(-1)
    return nll + lambda_lat * lat, zl


# =============================================================================
# Continuous heads. SPEC: 02 §2.5
# =============================================================================

class GaussianHead(nn.Module):
    """`a ~ N(W_a y + b_a, diag exp(2 w_s))`. SPEC: 02 (2.7)."""

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, d_x: int, d_a: int):
        super().__init__()
        self.d_x, self.d_a = d_x, d_a
        self.W_a = nn.Linear(d_x, d_a)
        self.w_s = nn.Parameter(torch.zeros(d_a))

    def log_std(self) -> torch.Tensor:
        return self.w_s.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    def mean(self, y: torch.Tensor) -> torch.Tensor:
        check_shape(y, [None, self.d_x], name="y_hat")
        return self.W_a(y)

    def nll(self, y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """`-log N(a; mu, sigma^2)` per row `(N,)`, in fp32."""
        mu = self.mean(y).float()
        check_shape(a, [mu.shape[0], self.d_a], name="a")
        ls = self.log_std().float()
        z = (a.float() - mu) * torch.exp(-ls)
        return 0.5 * z.square().sum(-1) + ls.sum() + 0.5 * self.d_a * math.log(2.0 * math.pi)

    def log_prob(self, y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return -self.nll(y, a)

    def sample(self, y: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        mu = self.mean(y)
        xi = torch.randn(mu.shape, generator=generator, dtype=torch.float32).to(mu.device, mu.dtype)
        return mu + torch.exp(self.log_std()).to(mu.dtype) * xi

    @torch.no_grad()
    def project(self) -> None:
        """09 §9.4: `w_s` is a parameter bound, kept by projection after every optimiser step
        (a graph clamp alone has zero gradient outside the range and never re-enters it)."""
        self.w_s.clamp_(self.LOG_STD_MIN, self.LOG_STD_MAX)


class BernoulliHead(nn.Module):
    """`y_j ~ Bernoulli(sigmoid(softcap(W_b y)_j))`. SPEC: 02 (2.8)."""

    def __init__(self, d_x: int, d_b: int):
        super().__init__()
        self.d_x, self.d_b = d_x, d_b
        self.W_b = nn.Linear(d_x, d_b)

    def logits(self, y: torch.Tensor) -> torch.Tensor:
        check_shape(y, [None, self.d_x], name="y_hat")
        z = self.W_b(y)
        return SOFTCAP * torch.tanh(z / SOFTCAP)

    def nll(self, y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lg = self.logits(y).float()
        check_shape(target, [lg.shape[0], self.d_b], name="target")
        return F.binary_cross_entropy_with_logits(lg, target.float(), reduction="none").sum(-1)

    def sample(self, y: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        p = torch.sigmoid(self.logits(y).float())
        u = torch.rand(p.shape, generator=generator).to(p.device)
        return (u < p).to(y.dtype)


class FlowHead(nn.Module):
    """Rectified-flow head `v_phi(a_s, s, y)` with `a_s = (1-s) a + s eps`. SPEC: 02 (2.9), §2.5.

    `s = 1` is pure noise and `s = 0` the data, so `d a_s / d s = eps - a` is the
    regression target and sampling integrates from `s = 1` down to `s = 0` with
    `a <- a - (1/steps) v_phi(a, s, y)` (the sign follows from the parameterisation).
    `s` is drawn from `Beta(1.5, 1)` by inverse CDF `s = u^{1/1.5}` so that the
    draw follows the supplied generator.
    """

    N_FOURIER = 8

    def __init__(self, d_x: int, d_a: int, *, width: int = 512):
        super().__init__()
        self.d_x, self.d_a = d_x, d_a
        d_in = d_a + 2 * self.N_FOURIER + d_x
        self.net = nn.Sequential(nn.Linear(d_in, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, d_a))

    def _time_features(self, s: torch.Tensor) -> torch.Tensor:
        """`[sin(pi f s), cos(pi f s)]_{f=1..8}`: period 2, so `s = 0` and `s = 1` are distinct."""
        f = torch.arange(1, self.N_FOURIER + 1, device=s.device, dtype=s.dtype)
        ang = math.pi * s[:, None] * f[None]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def velocity(self, a_s: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        check_shape(y, [None, self.d_x], name="y_hat")
        check_shape(a_s, [y.shape[0], self.d_a], name="a_s")
        return self.net(torch.cat([a_s, self._time_features(s.to(a_s.dtype)), y], dim=-1))

    def nll(self, y: torch.Tensor, a: torch.Tensor, *, generator: Optional[torch.Generator] = None
            ) -> torch.Tensor:
        """The flow-matching loss (2.9) per row `(N,)` (not a likelihood; named `nll`
        so that every head exposes the same interface)."""
        check_shape(a, [y.shape[0], self.d_a], name="a")
        N = a.shape[0]
        u = torch.rand((N,), generator=generator, dtype=torch.float32).to(a.device)
        s = u.pow(1.0 / 1.5).to(a.dtype)
        eps = torch.randn(a.shape, generator=generator, dtype=torch.float32).to(a.device, a.dtype)
        a_s = (1.0 - s)[:, None] * a + s[:, None] * eps
        v = self.velocity(a_s, s, y)
        return (v.float() - (eps - a).float()).square().sum(-1)

    @torch.no_grad()
    def sample(self, y: torch.Tensor, *, steps: int = 10,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        check_shape(y, [None, self.d_x], name="y_hat")
        N = y.shape[0]
        a = torch.randn((N, self.d_a), generator=generator, dtype=torch.float32).to(y.device, y.dtype)
        for k in range(steps):
            s = torch.full((N,), 1.0 - k / steps, device=y.device, dtype=y.dtype)
            a = a - (1.0 / steps) * self.velocity(a, s, y)
        return a
