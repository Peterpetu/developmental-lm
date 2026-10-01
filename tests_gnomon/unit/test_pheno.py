"""SPEC: 04 §4.7 -- the acceptance tests P1..P7 for the phenotype."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from gnomon.config import (
    GenomeShape, M_PICO, M_NANO, PhenoConfig, ROLE_EMB, ROLE_NORM_MIX, ROLE_Q,
    ROLE_SINK,
)
from gnomon.pheno import ops
from gnomon.pheno.mixer_chunk import gated_delta_rule_chunk
from gnomon.pheno.mixer_ref import gated_delta_rule_reference
from gnomon.pheno.model import Gnomon
from gnomon.pheno.plan import Arch
from gnomon.pheno.twin import FreeTwin

SHAPE = GenomeShape()
SMALL = PhenoConfig(name="small", d_model=64, L=2, n_h=2, n_kv=1, d_head=32,
                    d_ff=128, vocab=128, T=64, N_life=8, chunk_size=32,
                    window=32, batch=2)


def _tokens(cfg=SMALL, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab, (b, cfg.T + 1), generator=g)


# ---------------------------------------------------------------------- P1

def test_P1_twin_equivalence():
    """Decoded weights placed in a free-weight model give identical logits.

    This is what makes the compression claim a claim about *this* model.
    """
    m = Gnomon(SMALL, SHAPE, genome_seed=0, soma_seed=1).double()
    twin = FreeTwin(SMALL, seed=9).double().load_decoded(m.decode())
    tok = _tokens()[:, :-1]
    torch.testing.assert_close(m(tok), twin(tok), rtol=0, atol=1e-12)


def test_P1_twin_is_trainable_by_ordinary_means():
    """The baseline of §09 must actually be a working free-weight transformer."""
    twin = FreeTwin(SMALL, seed=3)
    opt = torch.optim.AdamW(twin.parameters(), lr=3e-3)
    tok = _tokens(b=4)
    first = float(twin.loss(tok))
    for _ in range(25):
        loss = twin.loss(tok)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert float(twin.loss(tok)) < first - 0.2, "the twin did not train"


def test_P1_twin_exposes_only_its_own_weights():
    """The host `Gnomon`'s decoder and soma must not leak into the baseline."""
    twin = FreeTwin(SMALL, seed=0)
    names = {n for n, _ in twin.named_parameters()}
    assert all(n.startswith("W.") for n in names)
    assert twin.n_params() == sum(
        math.prod(SMALL.role_shape(r)) for r, _ in twin._host.decode_keys())
    ids = {id(p) for p in twin.parameters()}
    assert not (ids & {id(p) for p in twin._host.decoder.parameters()})
    assert not (ids & {id(p) for p in twin._host.soma.parameters()})


# ---------------------------------------------------------------------- P2

def test_P2_the_chunkwise_mixer_still_matches_its_oracle():
    """SPEC: 04 §4.1. The vendored numerics, re-verified after the refactor.

    SOMA's (5.15) as first derived dropped the `g^r/g^i` factor and would have
    shipped a subtly wrong mixer that still trained. The oracle is the reason
    that was caught, and it is vendored with the code it checks.
    """
    torch.manual_seed(0)
    B, H, T, dk, dv = 2, 3, 96, 16, 16
    q = torch.randn(B, H, T, dk, dtype=torch.float64)
    k = torch.randn(B, H, T, dk, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, dv, dtype=torch.float64)
    alpha = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))
    beta = 2 * torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))
    for cs in (16, 32):
        a, sa = gated_delta_rule_chunk(q, k, v, alpha, beta, chunk_size=cs)
        b, sb = gated_delta_rule_reference(q, k, v, alpha, beta)
        torch.testing.assert_close(a, b, rtol=0, atol=1e-10)
        torch.testing.assert_close(sa, sb, rtol=0, atol=1e-10)


def test_P2_delta_mixer_functional_form_matches_the_reference_path():
    torch.manual_seed(1)
    cfg = SMALL
    x = torch.randn(2, cfg.T, cfg.d_model, dtype=torch.float64)
    Ws = [torch.randn(*s, dtype=torch.float64) * 0.1 for s in
          ((cfg.d_qk, cfg.d_model), (cfg.d_kv, cfg.d_model),
           (cfg.d_kv, cfg.d_model), (2 * cfg.n_h, cfg.d_model),
           (cfg.d_model, cfg.d_qk))]
    kw = dict(n_h=cfg.n_h, n_kv=cfg.n_kv, d_head=cfg.d_head,
              chunk_size=cfg.chunk_size)
    a, _ = ops.delta_mixer(x, *Ws, **kw)
    b, _ = ops.delta_mixer(x, *Ws, use_reference=True, **kw)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------- P3

def test_P3_every_decoded_role_has_its_specified_shape():
    for cfg in (SMALL, M_PICO, M_NANO):
        m = Gnomon(cfg, SHAPE)
        for (role, layer), v in m.decode().items():
            assert v.shape == cfg.role_shape(role), (cfg.name, role, layer)


# ---------------------------------------------------------------------- P4

def test_P4_no_future_token_leaks():
    """A token's logits must be invariant to every token after it."""
    m = Gnomon(SMALL, SHAPE, genome_seed=0, soma_seed=1).double()
    W = m.decode()
    tok = _tokens()[:, :-1]
    base = m(tok, weights=W)
    alt = tok.clone()
    alt[:, -1] = (alt[:, -1] + 7) % SMALL.vocab
    torch.testing.assert_close(base[:, :-1], m(alt, weights=W)[:, :-1],
                               rtol=0, atol=1e-11)


def test_P4_attention_sink_lets_a_head_decline_to_attend():
    """SPEC: 04 §4.1. `sum_j a[t,j] < 1` is the property the sink exists for."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64, dtype=torch.float64)
    W = [torch.randn(64, 64, dtype=torch.float64) * 0.1 for _ in range(4)]
    sink = torch.full((2,), 3.0, dtype=torch.float64)
    a = ops.attention(x, W[0], W[1], W[2], W[3], sink, n_h=2, n_kv=2,
                      d_head=32, return_weights=True)
    assert (a.sum(-1) < 1.0).all()
    zero = ops.attention(x, W[0], W[1], W[2], W[3],
                         torch.full((2,), -30.0, dtype=torch.float64),
                         n_h=2, n_kv=2, d_head=32, return_weights=True)
    torch.testing.assert_close(zero.sum(-1), torch.ones_like(zero.sum(-1)),
                               rtol=0, atol=1e-8)


def test_P4_sliding_window_masks_what_it_says():
    torch.manual_seed(0)
    x = torch.randn(1, 32, 64, dtype=torch.float64)
    W = [torch.randn(64, 64, dtype=torch.float64) * 0.1 for _ in range(4)]
    sink = torch.full((2,), -30.0, dtype=torch.float64)
    a = ops.attention(x, W[0], W[1], W[2], W[3], sink, n_h=2, n_kv=2,
                      d_head=32, window=8, return_weights=True)
    idx = torch.arange(32)
    outside = (idx[:, None] - idx[None, :]) >= 8
    assert float(a[0, :, outside].abs().max()) < 1e-12


def test_P4_document_mask_isolates_documents():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 64, dtype=torch.float64)
    W = [torch.randn(64, 64, dtype=torch.float64) * 0.1 for _ in range(4)]
    sink = torch.full((2,), -30.0, dtype=torch.float64)
    doc = torch.tensor([[0] * 8 + [1] * 8])
    a = ops.attention(x, W[0], W[1], W[2], W[3], sink, n_h=2, n_kv=2,
                      d_head=32, doc_id=doc, return_weights=True)
    assert float(a[0, :, 8:, :8].abs().max()) < 1e-12


# ---------------------------------------------------------------------- P5

def test_P5_G_NOFREE_the_model_holds_only_Gamma_and_Z():
    """SPEC: 04 §4.6. The absence gate."""
    m = Gnomon(M_PICO, SHAPE)
    got = {n for n, _ in m.named_parameters()}
    want = ({f"decoder.{n}" for n, _ in m.decoder.named_parameters()}
            | {f"soma.{n}" for n, _ in m.soma.named_parameters()})
    assert got == want, got ^ want
    assert len(m.soma_parameters()) == 3
    n_free = sum(p.numel() for p in m.parameters())
    assert n_free == m.n_genome() + m.n_soma()
    # No buffer anywhere has the shape of a decoded weight.
    decoded_shapes = {tuple(v.shape) for v in m.decode().values()}
    for name, b in m.named_buffers():
        assert tuple(b.shape) not in decoded_shapes, name


def test_P5_positive_control_a_planted_W_shared_fails_the_gate():
    """SPEC: 04 §4.6. An absence check must be shown to fire.

    The planted parameter is exactly SOMA's `W_shared`: a dense projection on
    the residual path, which carried 6.98 of that model's 6.76 nats of learning
    while its evolutionary population carried 0.16.
    """
    m = Gnomon(M_PICO, SHAPE)
    m.W_shared = nn.Parameter(torch.zeros(M_PICO.d_model, M_PICO.d_model))
    got = {n for n, _ in m.named_parameters()}
    want = ({f"decoder.{n}" for n, _ in m.decoder.named_parameters()}
            | {f"soma.{n}" for n, _ in m.soma.named_parameters()})
    assert got != want, "the positive control did not fire"
    assert sum(p.numel() for p in m.parameters()) != m.n_genome() + m.n_soma()


def test_P5_operators_own_no_parameters():
    """SPEC: 04 §4.1. A module that owned weights would be a W_shared in waiting."""
    import inspect

    for _, obj in inspect.getmembers(ops):
        assert not isinstance(obj, (nn.Parameter, nn.Module))
    assert all(callable(getattr(ops, n)) for n in
               ("attention", "delta_mixer", "swiglu", "rms_norm"))


# ---------------------------------------------------------------------- P6

def test_P6_determinism():
    m = Gnomon(SMALL, SHAPE, genome_seed=0, soma_seed=1)
    tok = _tokens()[:, :-1]
    assert torch.equal(m(tok), m(tok))


def test_forward_is_finite_at_every_configuration():
    for cfg in (SMALL, M_PICO, M_NANO):
        for gs in range(3):
            m = Gnomon(cfg, SHAPE, genome_seed=gs, soma_seed=gs + 40)
            tok = torch.randint(0, cfg.vocab, (2, min(cfg.T, 128) + 1))
            L = float(m.loss(tok))
            assert math.isfinite(L), (cfg.name, gs)
            # Init loss must sit near ln V: a random genome should be ignorant,
            # not confidently wrong (§04.3's cosine logits exist for this).
            assert math.log(cfg.vocab) - 0.5 < L < math.log(cfg.vocab) + 1.5, (
                f"{cfg.name} seed {gs}: init loss {L:.3f} vs ln V "
                f"{math.log(cfg.vocab):.3f}"
            )


def test_gradient_reaches_the_genome_and_the_soma_through_the_forward_pass():
    """The end-to-end version of D3: `dL/dGamma` must exist through the model."""
    m = Gnomon(SMALL, SHAPE, genome_seed=0, soma_seed=1)
    m.loss(_tokens(b=2)).backward()
    for name, p in m.decoder.named_parameters():
        assert p.grad is not None and p.grad.abs().max() > 0, f"Gamma.{name}"
    for name, p in m.soma.named_parameters():
        assert p.grad is not None and p.grad.abs().max() > 0, f"Z.{name}"


def test_arch_selects_which_roles_are_decoded():
    """An attention layer never materialises DELTA_BETA, and vice versa."""
    from gnomon.config import ROLE_DELTA_BETA
    a = Gnomon(SMALL, SHAPE, arch=Arch.uniform(SMALL, ("attn",) * SMALL.L))
    d = Gnomon(SMALL, SHAPE, arch=Arch.uniform(SMALL, ("delta",) * SMALL.L))
    assert not any(r == ROLE_DELTA_BETA for r, _ in a.decode_keys())
    assert not any(r == ROLE_SINK for r, _ in d.decode_keys())
    assert all((ROLE_Q, l) in dict.fromkeys(a.decode_keys())
               for l in range(SMALL.L))


def test_arch_length_must_match_the_configuration():
    with pytest.raises(ValueError, match="arch has"):
        Gnomon(SMALL, SHAPE, arch=Arch.uniform(SMALL, ("attn",) * (SMALL.L + 1)))


def test_random_weights_have_the_innate_scale_and_no_genome_content():
    """SPEC: 05 §5.5. `W_rand` is the anchor the annealing starts from."""
    m = Gnomon(SMALL, SHAPE)
    a = m.random_weights(seed=1)
    b = m.random_weights(seed=1)
    c = m.random_weights(seed=2)
    for k in a:
        assert a[k].shape == m.decode()[k].shape
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    assert not torch.equal(a[(ROLE_Q, 0)], c[(ROLE_Q, 0)])


# ---------------------------------------------------------------------- P7

def test_P7_the_inference_cache_is_bit_identical_to_decoding_per_step():
    """SPEC: 04 §4.7 P7, SPEC: 03 §3.7. `G-INFER`, the identity half.

    The production path of §01: at inference `Z` does not move, so `W` is
    constant and the decode is a one-off. "Identical", not "close" -- the cache
    is the *same* tensors, so anything but equality means the decode is not a
    pure function of `(Gamma, Z)` and D10's determinism is wrong somewhere.
    """
    m = Gnomon(SMALL, SHAPE, genome_seed=0, soma_seed=1).eval()
    tok = torch.randint(0, SMALL.vocab, (3, SMALL.T), generator=
                        torch.Generator().manual_seed(0))
    W = m.inference_weights()
    with torch.no_grad():
        cached = [m(tok, weights=W) for _ in range(3)]
        per_step = [m(tok) for _ in range(3)]
    for a in cached[1:]:
        assert torch.equal(cached[0], a), "the cache is not stable"
    for a, b in zip(cached, per_step):
        assert torch.equal(a, b), "the cache does not match a fresh decode"


def test_P7_the_inference_cache_is_measurably_faster():
    """SPEC: 04 §4.7 P7. `G-INFER`, the throughput half.

    Without this half the cache could be a no-op that re-decoded anyway and P7
    would still pass. The margin is deliberately loose -- this asserts that the
    decode is not being repeated, not a specific speedup -- because a wall-clock
    threshold tight enough to be interesting is one that fails on a loaded
    machine, and a flaky gate is worse than a loose one.
    """
    import time

    m = Gnomon(SMALL, SHAPE, genome_seed=0, soma_seed=1).eval()
    tok = torch.randint(0, SMALL.vocab, (3, SMALL.T))
    W = m.inference_weights()
    n = 12
    with torch.no_grad():
        for _ in range(3):                       # warm
            m(tok, weights=W); m(tok)
        t0 = time.perf_counter()
        for _ in range(n):
            m(tok, weights=W)
        t_cached = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            m(tok)
        t_decode = time.perf_counter() - t0
    assert t_cached < t_decode, (t_cached, t_decode)


def test_P7_the_cached_weights_carry_no_graph():
    """SPEC: 03 §3.7. A generation loop must not accumulate one.

    An inference path that kept the decode graph alive would grow without bound
    over a long sample, which is a memory leak wearing the costume of a cache.
    """
    m = Gnomon(SMALL, SHAPE, genome_seed=0, soma_seed=1)
    for k, v in m.inference_weights().items():
        assert not v.requires_grad and v.grad_fn is None, k


def test_P7_a_training_decode_is_still_inside_the_loop():
    """SPEC: 05 §5.1 line 5. The cache must not be reachable from training.

    §03.7's inference cache and §05.1's per-step decode are opposites, and the
    reason they are both right is that `Z` moves in one and not the other. A
    reader who took the cache for the general rule would reinstate M10.
    """
    import inspect

    from gnomon.lifetime.runner import LifetimeRunner

    src = inspect.getsource(LifetimeRunner._build_step)
    assert "inference_weights" not in src, (
        "the training step is using the inference cache; `Z` moves every step "
        "and the cache would be stale (SPEC: 05 §5.1 line 5)"
    )
    assert "m.decode()" in src


def test_P5_G_NOFREE_the_optimizers_cover_Z_and_in_the_darwinian_arms_nothing_else():
    """SPEC: 04 §4.6, second paragraph. The other half of the absence gate.

    "a free tensor that exists but is not optimized is not a bypass, and a
    tensor that is optimized but was not declared is". The first half of
    `G-NOFREE` walks the *model* and finds exactly `Gamma` union `Z`. That can
    be true while a third tensor is being stepped -- SOMA's `W_shared` did not
    have to be declared anywhere to carry 6.98 of that model's 6.76 nats, it
    only had to move -- so the gate has to walk the optimizers too.

    Compared by **identity**, in both directions: an extra tensor in a
    parameter group is a bypass, and a missing one is a declaration that has
    drifted from what the lifetime actually optimises. `G-MUTATE` M25 plants
    the second, which is the one a reader would not think to look for.
    """
    import dataclasses

    import numpy as np

    from gnomon.data.stream import Shard
    from gnomon.genome.decoder import Decoder
    from gnomon.lifetime.runner import LifetimeRunner

    cfg = dataclasses.replace(SMALL, N_life=2, batch=4)
    toks = np.random.default_rng(0).integers(0, cfg.vocab, 20000).astype(np.uint16)
    sh = Shard(toks, np.arange(0, 20000, 100, dtype=np.int64),
               np.full(200, 100, dtype=np.int64), {"n_tokens": 20000})
    r = LifetimeRunner(cfg, SHAPE, train_shard=sh, eval_shard=sh, device="cpu",
                       compile_mode=None, n_eval_batches=1)
    r.run(Decoder(SHAPE, seed=0), seed=1, rule="dpr")

    optimised = {id(p) for p in r.last_optimized}
    declared = {id(p) for p in r.model.soma_parameters()}
    genome = {id(p) for p in r.model.genome_parameters()}
    assert optimised == declared, (
        f"optimised-but-not-declared: {len(optimised - declared)}, "
        f"declared-but-not-optimised: {len(declared - optimised)}"
    )
    assert not (optimised & genome), "a Darwinian lifetime optimises `Gamma`"
    assert len(declared) == 3, "Z is z, zeta and y -- §04.5 lists no fourth"


def test_P5_G_NOFREE_the_lamarckian_arm_declares_Gamma_as_well():
    """SPEC: 04 §4.6, SPEC: 08 §8.5. The one arm where `Gamma` may be stepped.

    "in the Darwinian arms, nothing else" is a statement about the Darwinian
    arms. `A-LAMARCK` optimises `Gamma` on purpose, and the gate must say so
    rather than either failing on it or being silently relaxed for everything.
    """
    import dataclasses

    import numpy as np

    from gnomon.data.stream import Shard
    from gnomon.genome.decoder import Decoder
    from gnomon.lifetime.runner import LifetimeRunner

    cfg = dataclasses.replace(SMALL, N_life=2, batch=4)
    toks = np.random.default_rng(0).integers(0, cfg.vocab, 20000).astype(np.uint16)
    sh = Shard(toks, np.arange(0, 20000, 100, dtype=np.int64),
               np.full(200, 100, dtype=np.int64), {"n_tokens": 20000})
    r = LifetimeRunner(cfg, SHAPE, train_shard=sh, eval_shard=sh, device="cpu",
                       compile_mode=None, n_eval_batches=1)
    r.run(Decoder(SHAPE, seed=0), seed=1, rule="dpr", lamarck=True)

    optimised = {id(p) for p in r.last_optimized}
    declared = ({id(p) for p in r.model.soma_parameters()}
                | {id(p) for p in r.model.genome_parameters()})
    assert optimised == declared
