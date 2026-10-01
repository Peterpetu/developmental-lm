"""Sampling from a decoded phenotype, and the units a loss is reported in.

SPEC: 05 §5.2
SPEC: 09 §9.6

Held-out cross-entropy is the decision quantity, but it is not readable, so
every report carries the same number in three forms: nats per token (the
objective), bits per byte (comparable across tokenizers), and a sample of what
the model actually writes. The third is the only one that cannot be gamed by a
change of units, and it is the one that tells you whether a loss of 6.5 nats
means "learning slowly" or "producing noise".
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = ["generate", "bits_per_byte", "perplexity", "describe_loss"]


def bits_per_byte(ce_nats: float, n_tokens: int, n_bytes: int) -> float:
    """`BPB = (CE / ln 2) * (tokens / bytes)`.

    Vendored from SOMA `eval/metrics.py`. The token/byte ratio is what makes
    this comparable across tokenizers, which is why it is reported rather than
    perplexity: perplexity is tokenizer-dependent and is reporting-only.
    """
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")
    return (ce_nats / math.log(2.0)) * (n_tokens / n_bytes)


def perplexity(ce_nats: float) -> float:
    return math.exp(ce_nats)


def describe_loss(ce_nats: float, vocab: int, *, tokens_per_byte: float = 0.3
                  ) -> str:
    """One line putting a loss in context against the two anchors that matter."""
    return (f"{ce_nats:.4f} nats  |  ppl {perplexity(ce_nats):9.1f}  |  "
            f"bpb {bits_per_byte(ce_nats, 1, 1) * tokens_per_byte:.3f}  |  "
            f"ln V = {math.log(vocab):.3f}, so {math.log(vocab) - ce_nats:+.3f} "
            f"nats learned of a possible {math.log(vocab):.2f}")


@torch.no_grad()
def generate(model, tokenizer, *, prompt: str = "", max_new: int = 120,
             temperature: float = 0.8, top_p: float = 0.95,
             seed: int = 0, device: str = "cuda",
             weights: Optional[dict] = None) -> str:
    """Nucleus sampling from a decoded phenotype.

    The decode is done **once** and reused for every step (§03.7's inference
    path): `Z` does not move during generation, so `W` is constant, and this is
    the same caching that makes inference throughput equal to an ordinary
    transformer's.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    # SPEC: 03 §3.7, SPEC: 04 §4.7 P7 -- the named inference path, so
    # `G-INFER` tests the call this makes rather than a parallel one.
    W = model.inference_weights() if weights is None else weights
    ids = tokenizer.encode(prompt) if prompt else []
    if not ids:
        ids = [0]
    T = model.cfg.T
    out: list[int] = list(ids)

    for _ in range(max_new):
        ctx = torch.tensor([out[-T:]], dtype=torch.long, device=device)
        logits = model(ctx, weights=W)[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            srt, idx = torch.sort(probs, descending=True)
            keep = (torch.cumsum(srt, dim=-1) - srt) < top_p
            keep[0] = True
            srt = srt * keep
            srt = srt / srt.sum()
            nxt = int(idx[torch.multinomial(srt, 1, generator=g)])
        out.append(nxt)
    return tokenizer.decode(out)
