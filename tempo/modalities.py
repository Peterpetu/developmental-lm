"""The perception stream and the modality encoders and decoders.

SPEC: 02 §2.1  (x_t = E_m(raw) + mu_m + tau(t), unit-RMS encoder outputs)
SPEC: 02 §2.2  (the input-side span embedding E^in_txt, (2.2))
SPEC: 02 §2.4  (gradient insulation, (2.11))
SPEC: 02 §2.7  (pretrained-encoder forms: a projection of frozen features)
SPEC: 02 §2.8  (time codes, (2.13))
SPEC: 02 §2.9  (the encoders and decoders of the local blend)

Every encoder maps its raw form to `R^{d_x}`; `Perception` normalises the
result to unit RMS and adds the modality embedding and the time code, so the
backbone never sees a modality-specific object.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnomon.utils.shapes import check_shape
from .codec import unit_rms

__all__ = [
    "interface_flops",
    "STREAMS", "MODALITIES", "MODALITY_INDEX", "Modality", "TextIn", "TimeCode",
    "ModalityTable", "Perception", "ImageEncoder", "ImageDecoder", "SpeechEncoder",
    "LinearEncoder", "FrozenFeatureEncoder", "insulate", "Insulation", "N_TIME_FEATURES",
    "TIME_K_MIN", "TIME_K_MAX",
]

STREAMS = ("text", "task", "control", "vision", "speech", "pattern", "interleave")
"""SPEC: 02 §2.9 -- the seven streams the mixture role `mu` ranges over."""


@dataclasses.dataclass(frozen=True)
class Modality:
    """One entry of the perception stream: its raw dimensionality and rate."""

    name: str
    d_raw: int | None          # None for token modalities
    rate_hz: float             # spans per second of (simulated) wall-clock
    stream: str                # which of STREAMS carries it
    terms: tuple[str, ...]     # the loss terms of 02 §2.4 present on its spans


MODALITIES: tuple[Modality, ...] = (
    Modality("text", None, 4.0, "text", ("txt",)),
    Modality("task", None, 4.0, "task", ("txt", "rl")),
    Modality("prop", 8, 20.0, "control", ("rl",)),
    Modality("reward", 1, 20.0, "control", ()),
    Modality("image", 3 * 32 * 32, 5.0, "vision", ("img",)),
    Modality("speech", 64, 62.5, "speech", ("gau", "rl")),
    Modality("logistic", 8, 100.0, "pattern", ("gau",)),
    Modality("lorenz", 3, 100.0, "pattern", ("gau",)),
    Modality("eca", 64, 100.0, "pattern", ("ber",)),
)
"""SPEC: 02 §2.9 -- the modalities of the local blend and their rates."""

MODALITY_INDEX: dict[str, int] = {m.name: i for i, m in enumerate(MODALITIES)}

TIME_K_MIN, TIME_K_MAX = -6, 7
N_TIME_FEATURES = 2 * (TIME_K_MAX - TIME_K_MIN + 1)     # 28
"""SPEC: 02 (2.13) -- periods 2^k s for k = -6..7: 1/64 s to 128 s."""


class TextIn(nn.Module):
    """`E^in_txt(v_1..v_K) = W_x [T_emb[v_1]; ...; T_emb[v_K]]`. SPEC: 02 (2.2).

    Shares no parameter with the codec (I3): the input side is trained with the
    backbone at tau_1; the codec serves the output likelihood only.
    """

    def __init__(self, vocab: int, K: int, d_x: int, *, d_e: int = 128):
        super().__init__()
        self.vocab, self.K, self.d_x, self.d_e = vocab, K, d_x, d_e
        self.T_emb = nn.Embedding(vocab, d_e)
        self.W_x = nn.Linear(K * d_e, d_x, bias=False)
        nn.init.normal_(self.T_emb.weight, std=d_e ** -0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """`(..., K)` token ids -> `(..., d_x)`."""
        if tokens.shape[-1] != self.K:
            raise ValueError(f"expected spans of {self.K} tokens, got {tokens.shape[-1]}")
        e = self.T_emb(tokens)                                                # (..., K, d_e)
        return self.W_x(e.reshape(*tokens.shape[:-1], self.K * self.d_e))


class TimeCode(nn.Module):
    """`tau(t) = W_tau [sin(2 pi t / 2^k), cos(2 pi t / 2^k)]_{k=-6..7}`. SPEC: 02 (2.13)."""

    def __init__(self, d_x: int):
        super().__init__()
        self.d_x = d_x
        self.W_tau = nn.Linear(N_TIME_FEATURES, d_x, bias=False)
        nn.init.normal_(self.W_tau.weight, std=0.02)
        self.register_buffer("periods", torch.tensor(
            [2.0 ** k for k in range(TIME_K_MIN, TIME_K_MAX + 1)], dtype=torch.float64))

    def features(self, t_w: torch.Tensor) -> torch.Tensor:
        """`(N,)` seconds -> `(N, 28)` in fp32 (the angle is formed in float64)."""
        ang = 2.0 * math.pi * t_w.to(torch.float64)[:, None] / self.periods[None]
        return torch.cat([ang.sin(), ang.cos()], dim=-1).float()

    def forward(self, t_w: torch.Tensor) -> torch.Tensor:
        check_shape(t_w, [None], name="t_w")
        return self.W_tau(self.features(t_w).to(self.W_tau.weight.dtype))


class ModalityTable(nn.Module):
    """`mu_m in R^{d_x}`, one per modality, `N(0, 0.02^2)` at birth. SPEC: 02 §2.1."""

    def __init__(self, n_modalities: int, d_x: int):
        super().__init__()
        self.table = nn.Embedding(n_modalities, d_x)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, mod: torch.Tensor) -> torch.Tensor:
        return self.table(mod)


class Perception(nn.Module):
    """`x_t = E_m(raw_t)/RMS + mu_m + tau(t)`. SPEC: 02 (2.1)."""

    def __init__(self, d_x: int, n_modalities: int = len(MODALITIES)):
        super().__init__()
        self.d_x = d_x
        self.mu = ModalityTable(n_modalities, d_x)
        self.tau = TimeCode(d_x)

    def forward(self, feat: torch.Tensor, mod: torch.Tensor, t_w: torch.Tensor) -> torch.Tensor:
        """`feat (N, d_x)` encoder output, `mod (N,)` modality ids, `t_w (N,)` seconds."""
        check_shape(feat, [None, self.d_x], name="feat")
        N = feat.shape[0]
        check_shape(mod, [N], name="mod")
        check_shape(t_w, [N], name="t_w")
        return unit_rms(feat) + self.mu(mod) + self.tau(t_w)


# =============================================================================
# Encoders and decoders of the local blend. SPEC: 02 §2.9
# =============================================================================

class ImageEncoder(nn.Module):
    """Four stride-2 3x3 convolutions `3 -> 32 -> 64 -> 128 -> 256`, GELU, mean pool,
    `W_img: (d_x x 256)`. SPEC: 02 §2.9.4. Input `(N, 3, 32, 32)` in `[0, 1]`."""

    CHANNELS = (3, 32, 64, 128, 256)

    def __init__(self, d_x: int):
        super().__init__()
        self.d_x = d_x
        ch = self.CHANNELS
        self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1)
                                    for i in range(4)])
        self.W_img = nn.Linear(ch[-1], d_x, bias=False)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        check_shape(img, [None, 3, 32, 32], name="image")
        h = img
        for c in self.convs:
            h = F.gelu(c(h))
        return self.W_img(h.mean(dim=(2, 3)))


class ImageDecoder(nn.Module):
    """`W'_img: d_x -> 256 x 2 x 2`, then four stride-2 transposed convolutions
    `256 -> 128 -> 64 -> 32 -> 3`; the frame-prediction loss (2.10) is the squared
    error summed over pixels. SPEC: 02 §2.9.4."""

    CHANNELS = (256, 128, 64, 32, 3)

    def __init__(self, d_x: int):
        super().__init__()
        self.d_x = d_x
        ch = self.CHANNELS
        self.W_seed = nn.Linear(d_x, ch[0] * 2 * 2)
        self.deconvs = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i + 1], 4, stride=2, padding=1)
                                      for i in range(4)])

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        check_shape(y, [None, self.d_x], name="y_hat")
        h = self.W_seed(y).view(-1, self.CHANNELS[0], 2, 2)
        for i, d in enumerate(self.deconvs):
            h = d(h)
            if i < 3:
                h = F.gelu(h)
        return h                                                                # (N, 3, 32, 32)

    def nll(self, y: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """`||Dec(y) - frame||^2` per row `(N,)`. SPEC: 02 (2.10)."""
        pred = self.forward(y)
        check_shape(frame, list(pred.shape), name="frame")
        return (pred.float() - frame.float()).square().sum(dim=(1, 2, 3))


class SpeechEncoder(nn.Module):
    """A per-frame encoder `64 -> 128 -> 256 -> d_x` (GELU). SPEC: 02 §2.9.5.

    Per frame, not a convolution over consecutive frames: a temporal context inside the
    encoder made the acting path (one frame at a time) and the training path (a window)
    perceive the same frame differently (11 §P-C.26); the backbone carries the temporal
    context. Input `(N, 64)` or `(N, T_f, 64)`."""

    def __init__(self, d_x: int, d_frame: int = 64):
        super().__init__()
        self.d_x, self.d_frame = d_x, d_frame
        self.net = nn.Sequential(nn.Linear(d_frame, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU(),
                                 nn.Linear(256, d_x, bias=False))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[-1] != self.d_frame:
            raise ValueError(f"expected {self.d_frame}-bin frames, got {frames.shape[-1]}")
        return self.net(frames.to(self.net[0].weight.dtype))


class LinearEncoder(nn.Module):
    """`E_m(raw) = W raw + b`, `W: (d_x x d_raw)`: proprioception, reward, the pattern
    streams. SPEC: 02 §2.9.3, §2.9.6. The map is *affine*: under the unit-RMS
    normalisation of (2.1) a linear map would erase the magnitude of `raw` (a scalar
    reward would become its sign; 11 §P-C.26), whereas `raw -> (W raw + b)/RMS` is
    injective on a bounded domain for a generic `b`."""

    def __init__(self, d_raw: int, d_x: int):
        super().__init__()
        self.d_raw, self.d_x = d_raw, d_x
        self.W = nn.Linear(d_raw, d_x, bias=True)
        nn.init.normal_(self.W.bias, std=0.5)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.shape[-1] != self.d_raw:
            raise ValueError(f"expected raw dimension {self.d_raw}, got {raw.shape[-1]}")
        return self.W(raw.to(self.W.weight.dtype))


class FrozenFeatureEncoder(nn.Module):
    """`E_m(raw) = W_m * frozen(raw)`: the pretrained-encoder form (JEPA video, a
    neural speech codec). SPEC: 02 §2.7. The frozen model is a callable supplied by
    the caller; only the projection is a parameter. Not evaluable locally (10)."""

    def __init__(self, d_feat: int, d_x: int, frozen=None):
        super().__init__()
        self.d_feat, self.d_x = d_feat, d_x
        self.frozen = frozen
        self.W = nn.Linear(d_feat, d_x, bias=False)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = raw if self.frozen is None else self.frozen(raw)
        if feat.shape[-1] != self.d_feat:
            raise ValueError(f"expected features of width {self.d_feat}, got {feat.shape[-1]}")
        return self.W(feat.detach().to(self.W.weight.dtype))


# =============================================================================
# Gradient insulation. SPEC: 02 §2.4 (2.11)
# =============================================================================

def insulate(y_hat: torch.Tensor, kappa: float) -> torch.Tensor:
    """`kappa y + (1 - kappa) stopgrad(y)`: the head sees `y` in full; the backbone
    receives `kappa` of the head's gradient."""
    if not 0.0 <= kappa <= 1.0:
        raise ValueError("kappa in [0, 1]")
    if kappa >= 1.0:
        return y_hat
    if kappa <= 0.0:
        return y_hat.detach()
    return kappa * y_hat + (1.0 - kappa) * y_hat.detach()


@dataclasses.dataclass
class Insulation:
    """The tau_2 rule for `kappa_m` (02 §2.4): zero until the head's own loss has improved
    by half its first-episode value (`first - loss >= 0.5 |first|`, well posed for
    densities whose NLL can be negative), then a linear ramp from 0 to 1 over the
    following episode, then 1. `update(name, episode, loss)` is called once per boundary
    with the *head's own* mean loss of the episode just ended; `value(name, frac)` is the
    kappa at fraction `frac` of the current episode."""

    first_loss: dict[str, float] = dataclasses.field(default_factory=dict)
    ramp_episode: dict[str, int] = dataclasses.field(default_factory=dict)   # the episode during which kappa ramps

    def value(self, name: str, frac: float = 1.0, episode: Optional[int] = None) -> float:
        if name not in self.ramp_episode:
            return 0.0
        e_ramp = self.ramp_episode[name]
        if episode is None:
            return 1.0
        if episode < e_ramp:
            return 0.0
        if episode > e_ramp:
            return 1.0
        return float(min(1.0, max(0.0, frac)))

    def update(self, name: str, episode: int, loss: float) -> None:
        """After episode `episode` ended with the head's mean loss `loss`."""
        if name not in self.first_loss:
            self.first_loss[name] = float(loss)
            return
        if name in self.ramp_episode:
            return
        first = self.first_loss[name]
        if first - float(loss) >= 0.5 * abs(first):
            self.ramp_episode[name] = episode + 1                              # the ramp fills the next episode

    @property
    def kappa(self) -> dict[str, float]:
        """The kappa at the end of the last update, per head (for the record)."""
        return {n: (1.0 if n in self.ramp_episode else 0.0) for n in self.first_loss}

    def state_dict(self) -> dict:
        return {"first_loss": dict(self.first_loss), "ramp_episode": dict(self.ramp_episode)}

    def load_state_dict(self, d: dict) -> None:
        self.first_loss = dict(d["first_loss"])
        self.ramp_episode = dict(d.get("ramp_episode", {}))


# =============================================================================
# The interface's compute. SPEC: 05 §5.6, 09 §9.1
# =============================================================================

@torch.no_grad()
def interface_flops(agent, cfg, *, modalities=None) -> dict:
    """Forward FLOPs per span of each modality's interface, **measured** on the real modules.

    05 (5.2) counts the backbone. It does not count the interface, and for text the interface
    is not small: the span decoder alone is `44 %` of the span agent's forward cost at the
    TS shape (11 §P-C.34). Leaving it out of `F_episode` does not merely rescale the axis --
    it changes the *ratio* between two morphologies (adult over primordium is `3.93` on the
    backbone alone and `2.61` with the interface charged to both), and that ratio is exactly
    what an equal-compute comparison reads (10 §10.1).

    Returned per modality: `enc` (raw -> x, charged per input span) and `head` (y_hat ->
    likelihood, charged per scored target span). Measured with `torch.utils.flop_counter`,
    the same instrument `scripts/tempo_cont.py` uses for the token baseline, so that the two
    sides of K-CONT are counted by one method rather than by two hand-derived formulas that
    have to be trusted to agree.
    """
    from torch.utils.flop_counter import FlopCounterMode

    dev = next(agent.parameters()).device
    n = 8                                                    # a batch large enough to be representative
    out: dict[str, dict] = {}
    names = modalities or [m.name for m in MODALITIES]

    def count(fn) -> float:
        with FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops() / n

    for m in names:
        raw = _sample_raw(m, cfg, n, dev)
        if raw is None:
            continue
        y = torch.randn(n, cfg.d_x, device=dev)
        enc = count(lambda: agent.encode(m, raw))
        head = 0.0
        if m in ("text", "task"):
            head = count(lambda: agent.codec.nll(y, raw))
        elif m in agent.heads:
            head = count(lambda: agent.heads[m].nll(y, raw))
        out[m] = {"enc": float(enc), "head": float(head)}
    return out


def _sample_raw(m: str, cfg, n: int, device):
    """One batch of raw input of modality `m`, at the shapes the streams produce."""
    if m in ("text", "task"):
        return torch.randint(0, cfg.vocab, (n, cfg.K), device=device)
    if m == "image":
        return torch.rand(n, 3, 32, 32, device=device)
    if m == "speech":
        return torch.rand(n, 64, device=device)
    if m in ("prop", "reward", "logistic", "lorenz", "eca"):
        d = {"prop": 8, "reward": 1, "logistic": 8, "lorenz": 3, "eca": 64}[m]
        return torch.rand(n, d, device=device)
    return None
