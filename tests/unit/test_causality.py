"""Causality on the **production** forward path, by perturbation.

SPEC: 05 §5.3
SPEC: 05 §5.4
SPEC: 06 §6.1.1

Gate `G-CAUSAL`. Every masking test in the suite called
`Attention.attention_weights(...)` — a **diagnostic copy** of the masking logic —
and never `Attention.forward`. The depthwise causal convolution had no causality
test at all. Audit A-83 measured what that permits: breaking only `forward`'s
causal mask and changing the conv's left-pad from `(k-1, 0)` to `(k-2, 1)` leaks
0.34 and 1.76 of perturbation into earlier positions, moves a logit by 3.87, and
**all 415 tests still pass**.

The failure mode is the worst kind: the loss goes *down*, nothing raises, and the
result is a model that has been reading its own answers.

The method here is a perturbation probe rather than an inspection of a mask
tensor. It makes no assumption about how masking is implemented — only that
changing a token at position `j` must not change any output at position `i < j`,
which is what causality *means* and is the property that has to survive any
refactor of how it is achieved.
"""

from __future__ import annotations

import pytest
import torch

from soma.config import M_NANO
from soma.germline import Genome
from soma.model.model import SomaModel
from soma.utils.rng import RngBundle

SPLIT = 16
LENGTH = 32


def _leak(fn, x: torch.Tensor, split: int = SPLIT) -> float:
    """Max change at positions `< split` when positions `>= split` are replaced.

    Returns the infinity norm of the difference on the causal prefix. Exactly
    zero is the only acceptable answer for a float-deterministic function.
    """
    with torch.no_grad():
        a = fn(x)
        y = x.clone()
        if y.dtype.is_floating_point:
            y[:, split:] = torch.randn_like(y[:, split:])
        else:
            y[:, split:] = torch.randint(
                0, int(y.max()) + 1, y[:, split:].shape, dtype=y.dtype
            )
        b = fn(y)
        return float((a[:, :split] - b[:, :split]).abs().max())


@pytest.fixture(scope="module")
def model_and_genome():
    rngs = RngBundle.create(9, device="cpu")
    genome = Genome.initialise(M_NANO, generator=torch.Generator().manual_seed(9))
    rngs.seed_global("init")
    model = SomaModel(M_NANO)
    model.seed_populations(genome, rngs.np("init"), rngs.pt("init"))
    model.eval()
    return model, genome


def test_attention_forward_is_causal(model_and_genome) -> None:
    """Not `attention_weights` — the path the model actually runs."""
    model, _ = model_and_genome
    attn = next(m for m in model.modules() if type(m).__name__ == "Attention")
    x = torch.randn(2, LENGTH, M_NANO.d_model)
    assert _leak(lambda t: attn(t), x) == 0.0


def test_the_short_convolution_is_causal(model_and_genome) -> None:
    """§05.2.4's depthwise conv is a left-pad away from leaking the next `k-1`
    tokens, and had no test of any kind."""
    model, _ = model_and_genome
    conv = next(m for m in model.modules()
                if type(m).__name__ == "CausalDepthwiseConv1d")
    x = torch.randn(2, LENGTH, conv.conv.in_channels)
    assert _leak(lambda t: conv(t), x) == 0.0


def test_the_mixer_is_causal(model_and_genome) -> None:
    """The recurrence carries state forward only; the chunkwise form must not
    let a later chunk influence an earlier one."""
    model, _ = model_and_genome
    mixer = next(m for m in model.modules() if type(m).__name__ == "Mixer")
    x = torch.randn(2, LENGTH, M_NANO.d_model)
    assert _leak(lambda t: mixer(t)[0] if isinstance(mixer(t), tuple) else mixer(t),
                 x) == 0.0


def test_the_site_is_position_wise(model_and_genome) -> None:
    """A Site routes per token and must not mix across positions at all — not
    merely not-from-the-future. The router's per-token widening makes this worth
    asserting: `w_t` is chosen from a batch-level statistic, and a version that
    chose it from the batch **max** would couple every token to every other."""
    model, genome = model_and_genome
    site = model.sites[0]
    pop = model.populations[0]
    x = torch.randn(2, LENGTH, M_NANO.d_model)

    def run(t):
        return site(t, field=genome.field_, p_u=pop.p_u, g_u=pop.g_u,
                    m_u=pop.m_u, active=pop.active,
                    n_active=int(pop.active.sum())).y

    assert _leak(run, x) == 0.0


def test_the_whole_model_is_causal(model_and_genome) -> None:
    """End to end, on token ids: the composition of every masked component."""
    model, genome = model_and_genome
    tokens = torch.randint(0, M_NANO.V, (2, LENGTH),
                           generator=torch.Generator().manual_seed(3))

    def run(t):
        return model(t, genome=genome, want_metabolic=False).logits

    assert _leak(run, tokens) == 0.0


def test_the_probe_can_detect_a_leak(model_and_genome) -> None:
    """The positive control, and the reason to trust the zeros above.

    A deliberately **acausal** function of the same shape must register a
    non-zero leak. Without this, `_leak` returning 0.0 could mean "causal" or
    could mean the probe never perturbed anything.
    """
    x = torch.randn(2, LENGTH, 8)

    def acausal(t):
        # a right-shift: every position reads the one after it
        out = torch.zeros_like(t)
        out[:, :-1] = t[:, 1:]
        return out

    assert _leak(acausal, x) > 0.0

    def causal(t):
        out = torch.zeros_like(t)
        out[:, 1:] = t[:, :-1]
        return out

    assert _leak(causal, x) == 0.0


def test_attention_does_not_cross_document_boundaries(model_and_genome) -> None:
    """§05.4's document mask, on `forward`.

    §06.1.1's `L_ref` cache is bound to `doc_boundary_policy: "masked"`: every
    cached per-token loss is valid **only** if a token's conditioning depends on
    its own document alone. A leak here silently invalidates the whole sidecar
    while every recorded hash still matches.
    """
    model, _ = model_and_genome
    attn = next(m for m in model.modules() if type(m).__name__ == "Attention")
    x = torch.randn(1, LENGTH, M_NANO.d_model)
    doc = torch.zeros(1, LENGTH, dtype=torch.long)
    doc[:, SPLIT:] = 1                      # two documents packed in one row

    with torch.no_grad():
        a = attn(x, doc_id=doc)
        y = x.clone()
        y[:, :SPLIT] = torch.randn_like(y[:, :SPLIT])   # perturb document 0
        b = attn(y, doc_id=doc)
    # Document 1 must be unaffected by anything in document 0.
    assert float((a[:, SPLIT:] - b[:, SPLIT:]).abs().max()) == 0.0
