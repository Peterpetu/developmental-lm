"""`Bauplan`: a discrete body plan, and the free-weight transformer it expresses.

The genome candidate S1 searches. Every field is discrete; none of them is a
weight, and no weight is a differentiable function of any of them. That is the
LAW OF DISJOINT SUBSTRATES, made structural rather than argued.

Extends `gnomon/pheno/plan.py:Arch`, which already carried per-layer
`(ops, n_h, d_ff, window)`, with the two things a screen needs and it lacked:
a **bank map** (which layers share weights) and **exact resource accounting**
(unique parameters, KV-cache bytes, forward FLOPs).

The accounting is load-bearing, not decoration. A screen that prices structure
selects on the price; if the price is wrong it selects for the wrong thing while
looking exactly like a working experiment. So `flops_per_token` is analytic,
written out term by term, and tested against hand-derived counts rather than
against itself.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnomon.pheno import ops

__all__ = ["Bauplan", "BodyModel", "GLOBAL"]

_OPS = ("attn", "delta")

#: An unambiguously global attention span. Deliberately *not* `T`: `ops.py:92`
#: masks only when `window < T`, so `window == T` is global too, and a menu
#: offering `T` as its "local" option would be silently inert -- the defect that
#: voided 240 of an earlier draft's 256 body plans.
GLOBAL = 1 << 30


@dataclasses.dataclass(frozen=True)
class Bauplan:
    """One body plan. Per-layer fields are tuples of length `L`."""

    ops: tuple[str, ...]
    window: tuple[int, ...]          # attention span; ignored by `delta` layers
    d_ff: tuple[int, ...]
    bank: tuple[int, ...]            # layers sharing a bank id share every tensor
    n_h: tuple[int, ...]
    d_model: int = 128
    n_kv: int = 2
    d_head: int = 32
    vocab: int = 4096
    chunk_size: int = 64

    def __post_init__(self) -> None:
        L = len(self.ops)
        for name in ("window", "d_ff", "bank", "n_h"):
            if len(getattr(self, name)) != L:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} entries, ops has {L}")
        bad = [o for o in self.ops if o not in _OPS]
        if bad:
            raise ValueError(f"unknown operator(s) {bad}; expected {_OPS}")
        if any(w < 1 for w in self.window):
            raise ValueError(f"window entries must be >= 1: {self.window}")
        if any(h % self.n_kv for h in self.n_h):
            raise ValueError(f"every n_h must be a multiple of n_kv={self.n_kv}")
        # Sharing a bank means sharing *tensors*, so the shapes must agree.
        # Rejected at construction rather than at expression time, because a
        # plan that cannot be expressed is not a plan, and a mutation operator
        # that can produce one is a defect in the operator.
        for b in set(self.bank):
            sig = {(self.ops[l], self.n_h[l], self.d_ff[l])
                   for l in range(L) if self.bank[l] == b}
            if len(sig) > 1:
                raise ValueError(
                    f"bank {b} shared by layers of differing shape: {sorted(sig)}")

    @property
    def L(self) -> int:
        return len(self.ops)

    @property
    def n_banks(self) -> int:
        return len(set(self.bank))

    @staticmethod
    def uniform(L: int = 4, op: str = "attn", *, d_model: int = 128,
                d_ff: int = 512, n_h: int = 4, window: int = GLOBAL,
                share: Optional[tuple[int, ...]] = None, **kw) -> "Bauplan":
        """The reference plan: every layer identical, no sharing."""
        return Bauplan(
            ops=(op,) * L, window=(window,) * L, d_ff=(d_ff,) * L,
            bank=share if share is not None else tuple(range(L)),
            n_h=(n_h,) * L, d_model=d_model, **kw)

    def replace(self, **kw) -> "Bauplan":
        return dataclasses.replace(self, **kw)

    # ------------------------------------------------------------ accounting

    def layer_params(self, l: int) -> int:
        """Parameters in one layer's own tensors, ignoring sharing."""
        d, h, f = self.d_model, self.n_h[l], self.d_ff[l]
        d_qk, d_kv = h * self.d_head, self.n_kv * self.d_head
        qkvo = d * d_qk + 2 * d * d_kv + d_qk * d
        gate = h if self.ops[l] == "attn" else 2 * h * d      # sink | alpha,beta
        ffn = 3 * d * f
        norms = 2 * d                                          # norm_mix, norm_ffn
        return qkvo + gate + ffn + norms

    def n_unique_params(self, *, include_embedding: bool = False) -> int:
        """Distinct trainable reals. Sharing reduces this; it never reduces FLOPs."""
        seen, total = set(), 0
        for l in range(self.L):
            if self.bank[l] in seen:
                continue
            seen.add(self.bank[l])
            total += self.layer_params(l)
        total += self.d_model                                  # norm_final
        if include_embedding:
            total += self.vocab * self.d_model                 # tied E
        return total

    def kv_cache_bytes(self, *, T: int, bytes_per_elem: int = 2) -> int:
        """Bytes of K/V state an autoregressive decode holds at length `T`.

        Attention layers hold `min(T, window)` positions of K and V per KV head.
        Delta layers hold a fixed-size recurrent state `(n_h, d_head, d_head)`
        instead -- the entire point of the operator at long context, and the
        reason this descriptor separates the two.
        """
        total = 0
        for l in range(self.L):
            if self.ops[l] == "attn":
                total += 2 * self.n_kv * self.d_head * min(T, self.window[l])
            else:
                total += self.n_h[l] * self.d_head * self.d_head
        return total * bytes_per_elem

    def flops_per_token(self, *, T: int) -> int:
        """Forward FLOPs per token, analytic, multiply-accumulate counted as 2.

        Written out term by term because "matched FLOPs" is what every ablation
        in the screen is matched on, and an implementation that silently counted
        *steps* instead would hand delta plans 3-4x more compute -- measured here
        at 16 ms/step for attention against 45-65 ms/step for delta.

        Sharing does **not** appear: `L` layer applications cost `L` layers'
        arithmetic whatever `n_banks` is. That asymmetry between this function
        and `n_unique_params` is the only reason a sharing axis can exist -- and
        it is also why matching both resources at once forces `n_banks == L`,
        which is how an earlier draft of this work algebraically deleted its own
        largest genome component without noticing.
        """
        d, V = self.d_model, self.vocab
        total = 0
        for l in range(self.L):
            h, f = self.n_h[l], self.d_ff[l]
            d_qk, d_kv = h * self.d_head, self.n_kv * self.d_head
            total += 2 * d * d_qk + 2 * 2 * d * d_kv + 2 * d_qk * d   # q,k,v,o
            if self.ops[l] == "attn":
                # Mean attended span over positions 0..T-1 under a causal mask
                # truncated to `window`:  sum_t min(t+1, w) / T.
                w = min(self.window[l], T)
                span = (w * (w + 1) / 2 + (T - w) * w) / T
                total += int(2 * 2 * h * self.d_head * span)     # scores + AV
            else:
                # Gated delta rule: per token the state (n_h, d_head, d_head) is
                # read, corrected and written -- three passes over the state.
                total += 2 * 3 * h * self.d_head * self.d_head
                total += 2 * 2 * h * d                            # alpha, beta
            total += 2 * 3 * d * f                                # SwiGLU
        total += 2 * d * V                                        # tied head
        return int(total)

    def descriptors(self, *, T: int) -> tuple[float, float]:
        """`(log10 KV-cache bytes, log10 unique params)`, without training."""
        return (math.log10(self.kv_cache_bytes(T=T)),
                math.log10(self.n_unique_params()))

    def summary(self) -> str:
        return " ".join(
            f"{o[0]}{h}/{f}/w{'inf' if w >= (1 << 20) else w}/b{b}"
            for o, h, f, w, b in
            zip(self.ops, self.n_h, self.d_ff, self.window, self.bank))


class BodyModel(nn.Module):
    """A `Bauplan` expressed as a free-weight transformer.

    Peri-LN with a `1/sqrt(2L)` branch scale and a tied head, matching
    `gnomon/pheno/model.py:Gnomon.forward` in structure so a screen number and a
    GNOMON-era number describe the same architecture. The head is a plain tied
    linear rather than GNOMON's cosine-with-temperature: that normalisation
    exists because *decoded* weights are structured in a way free weights are
    not, and importing it would import a fix for a problem this model lacks.

    Layers sharing a bank hold the **same** `nn.Parameter` objects -- identity,
    not equality. `tests_screen` asserts exactly that, because a bug giving each
    layer its own copy would turn every sharing measurement into a measurement
    of nothing while the parameter count still looked right.
    """

    def __init__(self, plan: Bauplan, *, seed: int = 0):
        super().__init__()
        self.plan = plan
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        d = plan.d_model

        def normal(*shape, std):
            return nn.Parameter(torch.randn(*shape, generator=g) * std)

        self.banks = nn.ModuleDict()
        for l in range(plan.L):
            key = str(plan.bank[l])
            if key in self.banks:
                continue
            h, f = plan.n_h[l], plan.d_ff[l]
            d_qk, d_kv = h * plan.d_head, plan.n_kv * plan.d_head
            p = nn.ParameterDict({
                "q": normal(d_qk, d, std=d ** -0.5),
                "k": normal(d_kv, d, std=d ** -0.5),
                "v": normal(d_kv, d, std=d ** -0.5),
                "o": normal(d, d_qk, std=d_qk ** -0.5),
                "gate": normal(f, d, std=d ** -0.5),
                "up": normal(f, d, std=d ** -0.5),
                "down": normal(d, f, std=f ** -0.5),
                "n_mix": nn.Parameter(torch.ones(d)),
                "n_ffn": nn.Parameter(torch.ones(d)),
            })
            if plan.ops[l] == "attn":
                p["sink"] = nn.Parameter(torch.zeros(h))
            else:
                p["ab"] = normal(2 * h, d, std=d ** -0.5)
            self.banks[key] = p

        self.E = normal(plan.vocab, d, std=0.02)
        self.n_final = nn.Parameter(torch.ones(d))

    def bank_of(self, l: int) -> nn.ParameterDict:
        return self.banks[str(self.plan.bank[l])]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        p = self.plan
        x = F.embedding(tokens, self.E)
        branch = 1.0 / (2.0 * p.L) ** 0.5
        for l in range(p.L):
            W = self.bank_of(l)
            h = ops.rms_norm(x, 1.0)
            if p.ops[l] == "attn":
                y = ops.attention(
                    h, W["q"], W["k"], W["v"], W["o"], W["sink"],
                    n_h=p.n_h[l], n_kv=p.n_kv, d_head=p.d_head,
                    window=p.window[l])
            else:
                y, _ = ops.delta_mixer(
                    h, W["q"], W["k"], W["v"], W["ab"], W["o"],
                    n_h=p.n_h[l], n_kv=p.n_kv, d_head=p.d_head,
                    chunk_size=p.chunk_size)
            x = x + ops.rms_norm(y, W["n_mix"]) * branch
            f = ops.swiglu(ops.rms_norm(x, 1.0), W["gate"], W["up"], W["down"])
            x = x + ops.rms_norm(f, W["n_ffn"]) * branch
        return F.linear(ops.rms_norm(x, self.n_final), self.E)

    def loss(self, window: torch.Tensor) -> torch.Tensor:
        """Next-token cross-entropy in nats over a `(B, T+1)` window."""
        logits = self(window[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            window[:, 1:].reshape(-1))

    def n_params(self, *, include_embedding: bool = True) -> int:
        """Counted by tensor *identity*, so a shared bank counts once."""
        seen, total = set(), 0
        for name, prm in self.named_parameters():
            if id(prm) in seen:
                continue
            seen.add(id(prm))
            if not include_embedding and name == "E":
                continue
            total += prm.numel()
        return total
