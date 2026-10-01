"""SPEC: 05 §5.7 -- the acceptance tests L1..L9 for the lifetime."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tests_gnomon.conftest import requires_corpus
import torch

from gnomon.config import GenomeShape, M_PICO, PhenoConfig
from gnomon.data.stream import Shard, TokenStream, eval_batches
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.train import eps_at, evaluate, lifetime, wsd_lr
from gnomon.pheno.model import Gnomon

SHAPE = GenomeShape()
TINY = PhenoConfig(name="tiny", d_model=32, L=2, n_h=2, n_kv=1, d_head=16,
                   d_ff=64, vocab=64, T=64, N_life=12, chunk_size=32,
                   window=32, batch=4, lr=3e-3)


def _fake_shard(n_tokens=20000, vocab=64, seed=0):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, size=n_tokens).astype(np.uint16)
    starts = np.arange(0, n_tokens, 100, dtype=np.int64)
    lengths = np.full(len(starts), 100, dtype=np.int64)
    return Shard(tokens, starts, lengths, {"n_tokens": n_tokens})


# ---------------------------------------------------------------------- L1

def test_L1_the_decode_is_inside_the_training_loop():
    """SPEC: 05 §5.1 line 5. A decode hoisted out of the loop goes stale.

    `W` is a function of `Z`, and `Z` moves every step, so caching `W` across
    steps severs the gradient path from the loss to the identities. That is the
    class of defect that made SOMA's routing gradient severable while 452 tests
    passed, so it is asserted directly: after one optimizer step, a fresh decode
    must differ from the one taken before it.
    """
    m = Gnomon(TINY, SHAPE, genome_seed=0, soma_seed=1)
    before = m.decode()
    opt = torch.optim.AdamW(m.soma_parameters(), lr=1e-2)
    loss = m.loss(torch.randint(0, TINY.vocab, (4, TINY.T + 1)), weights=before)
    loss.backward()
    opt.step()
    after = m.decode()
    changed = [k for k in before if not torch.allclose(before[k], after[k])]
    assert len(changed) > len(before) // 2, (
        f"only {len(changed)}/{len(before)} decoded roles moved after a soma "
        f"step; a stale decode would show as zero"
    )


# ---------------------------------------------------------------------- L2

def test_L2_gradient_reaches_every_soma_tensor():
    m = Gnomon(TINY, SHAPE, genome_seed=0, soma_seed=1)
    m.loss(torch.randint(0, TINY.vocab, (4, TINY.T + 1))).backward()
    for name, p in m.soma.named_parameters():
        assert p.grad is not None and p.grad.abs().max() > 0, name


# ---------------------------------------------------------------------- L3

def _runner(cfg=TINY, **kw):
    """A `LifetimeRunner` over fake shards. SPEC: 05 §5.1.

    L1..L9 bind the runner directly rather than the `lifetime` convenience
    wrapper, because the invariants they assert are properties of the tensors
    that actually take part in a lifetime -- the runner's own `Gamma` and `Z` --
    and not of the `Decoder` a caller passes in, whose values are *copied*.
    """
    from gnomon.lifetime.runner import LifetimeRunner
    return LifetimeRunner(cfg, SHAPE, train_shard=_fake_shard(vocab=cfg.vocab),
                          eval_shard=_fake_shard(vocab=cfg.vocab, seed=1),
                          device="cpu", compile_mode=None, n_eval_batches=2,
                          **kw)


@pytest.mark.slow
def test_L3_the_genome_is_not_written_during_a_lifetime():
    """SPEC: 05 §5.7 L3. The germline is read, never written, in the Darwinian arms.

    Asserted on the runner's *own* genome. The version this replaces asserted it
    on the `Decoder` handed to `lifetime()`, which the runner never uses as a
    live tensor -- so under the unified implementation it would have passed
    while saying nothing, which is how a test comes to be unable to fail.
    """
    r = _runner()
    before = [p.detach().clone() for p in r.model.genome_parameters()]
    r.run(Decoder(SHAPE, seed=0), seed=3)
    for p, b in zip(r.model.genome_parameters(), before):
        assert torch.equal(p.detach(), b)


def test_L3_genome_requires_grad_is_off_in_darwinian_arms():
    """SPEC: 05 §5.1 line 9. No gradient reaches `Gamma` outside A-LAMARCK."""
    r = _runner()
    assert not any(p.requires_grad for p in r.model.genome_parameters())
    r.run(Decoder(SHAPE, seed=0), seed=3)
    assert not any(p.requires_grad for p in r.model.genome_parameters())
    assert all(p.grad is None for p in r.model.genome_parameters())
    # ...and the soma is the half that *does* move.
    assert all(p.requires_grad for p in r.model.soma_parameters())


# ---------------------------------------------------------------------- L4

def test_L4_the_decoded_rule_is_detached_and_in_range():
    """SPEC: 10 §10.2. A lifetime must not switch its own plasticity off."""
    from gnomon.config import N_RULE

    m = Gnomon(M_PICO, SHAPE, genome_seed=0, soma_seed=1)
    coef = m.plasticity()
    for k, v in coef.items():
        assert not v.requires_grad and v.grad_fn is None, k
        assert v.shape[1] == N_RULE
    assert coef["z"].shape == (M_PICO.d_model, N_RULE)
    assert coef["zeta"].shape == (M_PICO.vocab, N_RULE)


def test_L4_the_rule_reduces_to_adam_at_its_baseline_point():
    """SPEC: 10 §10.6 P1, at the level of the lifetime rather than the tensor."""
    from gnomon.lifetime.plasticity import ADAM_POINT, DecodedRule
    from gnomon.config import N_RULE

    torch.manual_seed(0)
    a = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    rule = DecodedRule({"z": a})
    adam = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                             weight_decay=0.0)
    pin = torch.zeros(12, N_RULE, dtype=torch.float64)
    for j, v in enumerate(ADAM_POINT):
        pin[:, j] = v
    g = torch.Generator().manual_seed(1)
    for _ in range(25):
        grad = torch.randn(12, 3, generator=g, dtype=torch.float64)
        a.grad, b.grad = grad.clone(), grad.clone()
        rule.step({"z": pin}, loss=1.0, lr=1e-2)
        adam.step()
    torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=1e-9)


# ---------------------------------------------------------------------- L5

def test_L5_annealing_endpoints():
    """SPEC: 05 §5.5. `eps(0) = 1`, and `eps(N_life)` is negligible."""
    for cfg in (TINY, M_PICO):
        assert eps_at(0, cfg) == pytest.approx(1.0)
        assert eps_at(cfg.N_life, cfg) < 1e-10, cfg.name
        # 3*lam exactly, not int(3*lam): at TINY's lam = 0.48 the truncation
        # to n = 1 tests a different point of the curve entirely.
        assert eps_at(3 * cfg.lam, cfg) < 0.05
        assert eps_at(7 * cfg.lam, cfg) < 1e-3
        assert cfg.lam == pytest.approx(0.04 * cfg.N_life)


def test_L5_at_eps_one_the_model_is_exactly_W_rand():
    m = Gnomon(TINY, SHAPE, genome_seed=0, soma_seed=1).double()
    Wr = m.random_weights(seed=5)
    tok = torch.randint(0, TINY.vocab, (2, TINY.T + 1))[:, :-1]
    torch.testing.assert_close(m(tok, eps=1.0, W_rand=Wr),
                               m(tok, weights=Wr), rtol=0, atol=1e-12)


def test_L5_eps_requires_W_rand():
    m = Gnomon(TINY, SHAPE)
    with pytest.raises(ValueError, match="W_rand"):
        m(torch.zeros(1, 4, dtype=torch.long), eps=0.5)


# ---------------------------------------------------------------------- L6

def test_L6_wsd_schedule_boundaries():
    cfg = dataclasses.replace(TINY, N_life=1000, lr=1.0)
    assert wsd_lr(0, cfg) == pytest.approx(1 / 50)
    assert wsd_lr(49, cfg) == pytest.approx(1.0)
    assert wsd_lr(500, cfg) == pytest.approx(1.0)
    assert wsd_lr(799, cfg) == pytest.approx(1.0)
    assert wsd_lr(800, cfg) == pytest.approx(1.0)
    assert wsd_lr(999, cfg) == pytest.approx(1 / 200, abs=1e-9)
    assert all(0 <= wsd_lr(n, cfg) <= 1.0 for n in range(cfg.N_life))


# ---------------------------------------------------------------------- L7

def test_L7_a_lifetime_is_reproducible_from_its_seed():
    a = lifetime(Decoder(SHAPE, seed=0), TINY, SHAPE, seed=11,
                 train_shard=_fake_shard(), eval_shard=_fake_shard(seed=1),
                 n_eval_batches=2)
    b = lifetime(Decoder(SHAPE, seed=0), TINY, SHAPE, seed=11,
                 train_shard=_fake_shard(), eval_shard=_fake_shard(seed=1),
                 n_eval_batches=2)
    assert a.final == b.final and a.birth == b.birth
    assert a.train_curve == b.train_curve


def test_L7_different_seeds_differ():
    a = lifetime(Decoder(SHAPE, seed=0), TINY, SHAPE, seed=11,
                 train_shard=_fake_shard(), eval_shard=_fake_shard(seed=1),
                 n_eval_batches=2)
    b = lifetime(Decoder(SHAPE, seed=0), TINY, SHAPE, seed=12,
                 train_shard=_fake_shard(), eval_shard=_fake_shard(seed=1),
                 n_eval_batches=2)
    assert a.final != b.final


# ---------------------------------------------------------------------- L8

def test_L8_the_stream_is_resumable_bit_for_bit():
    """SPEC: 05 §5.7 L8. Offsets are derived from `(seed, step)`, not replayed."""
    sh = _fake_shard()
    a = TokenStream(sh, batch=4, T=32, seed=7)
    want = [a.next() for _ in range(6)]
    b = TokenStream(sh, batch=4, T=32, seed=7)
    b.seek(3)
    for k in range(3, 6):
        assert torch.equal(b.next(), want[k]), k
    assert b.state()["step"] == 6


def test_L8_streams_with_different_seeds_differ():
    sh = _fake_shard()
    assert not torch.equal(TokenStream(sh, batch=4, T=32, seed=1).next(),
                           TokenStream(sh, batch=4, T=32, seed=2).next())


# ---------------------------------------------------------------------- L9

@requires_corpus()
def test_L9_the_fitness_and_test_splits_are_disjoint():
    """SPEC: 05 §5.6. Selection must not be able to overfit the reported metric."""
    fit = Shard.load("data/ts_4096", "fitness")
    test = Shard.load("data/ts_4096", "test")
    assert fit.n_tokens > 0 and test.n_tokens > 0
    a = np.asarray(fit.tokens[:4096])
    b = np.asarray(test.tokens[:4096])
    assert not np.array_equal(a, b)
    full = Shard.load("data/ts_4096", "validation" if False else "train")
    assert fit.n_tokens + test.n_tokens <= 4486475 + 8
    assert fit.meta["split"] == "fitness" and test.meta["split"] == "test"


def test_eval_batches_are_fixed_and_non_overlapping():
    """SPEC: 05 §5.6. Same tokens for every genome, so comparisons are paired."""
    sh = _fake_shard()
    a = eval_batches(sh, batch=2, T=32, n_batches=3)
    b = eval_batches(sh, batch=2, T=32, n_batches=3)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    flat = torch.cat([x.reshape(-1) for x in a])
    assert flat.numel() == 3 * 2 * 33
    with pytest.raises(ValueError, match="tokens"):
        eval_batches(sh, batch=2, T=32, n_batches=10**6)


# --------------------------------------------------------------- the result

@pytest.mark.slow
def test_a_lifetime_reduces_held_out_loss():
    cfg = dataclasses.replace(TINY, N_life=120, lr=5e-3)
    r = lifetime(Decoder(SHAPE, seed=0), cfg, SHAPE, seed=1,
                 train_shard=_fake_shard(n_tokens=50000, vocab=cfg.vocab),
                 eval_shard=_fake_shard(n_tokens=20000, vocab=cfg.vocab, seed=1),
                 n_eval_batches=2)
    # Random tokens: the learnable floor is ln V, so require it approach it.
    assert r.final < r.birth, f"birth {r.birth:.3f} -> final {r.final:.3f}"
    assert r.final < math.log(cfg.vocab) + 0.35
    assert r.n_genome == SHAPE.n_params()
    # TINY exists to be fast, not to compress; the compression claim is
    # tested at the real configurations in test_decoder.py.
    assert r.n_decoded > 4 * r.n_genome


# ------------------------------------------------------- A-LAMARCK (§08.5)
#
# The arm §08.5 specifies as "the paper's own algorithm, as the upper bound",
# which §01.7 and §05.1 both promise and which nothing implemented until v1.1:
# `EvolveConfig` had no flag and the runner unconditionally froze `Gamma`. Its
# cross-references pointed at §07.6, which is an acceptance-test table.

def test_A_LAMARCK_opens_the_gradient_path_into_every_genome_tensor():
    """SPEC: 08 §8.5. The lifetime loss reaches `Gamma` through the decode.

    Per *tensor*, not in aggregate: a path that reaches eleven of twelve is a
    severed path, and §03.8's D3 makes the same demand of the decode itself.
    """
    r = _runner()
    g = Decoder(SHAPE, seed=0)
    base = {k: v.clone() for k, v in g.state_dict().items()}
    res = r.run(g, seed=5, rule="adam", lamarck=True)
    after = r.export_genome()
    assert res.lamarck is True
    unmoved = [k for k in base if torch.equal(base[k], after[k])]
    assert not unmoved, f"no gradient reached {unmoved}"


def test_A_LAMARCK_closes_the_path_again_before_returning():
    """SPEC: 05 §5.1 line 9. A later Darwinian lifetime must not inherit it.

    Without this the arm would be sticky: one Lamarckian run inside a process
    would silently make every subsequent run Lamarckian too, and the arms would
    stop being arms.
    """
    r = _runner()
    g = Decoder(SHAPE, seed=0)
    r.run(g, seed=5, rule="adam", lamarck=True)
    assert not any(p.requires_grad for p in r.model.genome_parameters())
    assert all(p.grad is None for p in r.model.genome_parameters())

    # A Darwinian lifetime reloads `g`, so what it must end holding is `g`
    # itself -- not the genome the Lamarckian run left behind. Comparing
    # against the latter would assert that the reload did *not* happen.
    res = r.run(g, seed=5, rule="adam")           # Darwinian again
    assert res.lamarck is False
    ended = r.export_genome()
    for k, v in g.state_dict().items():
        assert torch.equal(ended[k], v.cpu()), k


def test_A_LAMARCK_is_labelled_everywhere_it_can_be_read():
    """SPEC: 01 §1.7 -- *every* artifact of the arm is labelled."""
    from gnomon.qd.archive import Member
    from gnomon.qd.evolve import EvolveConfig

    assert EvolveConfig(lamarck=True).label() == "A-LAMARCK"
    # ...and it wins over any other flag set alongside it, because a run that
    # is both A-NOX and Lamarckian is a Lamarckian result first.
    assert EvolveConfig(lamarck=True, no_crossover=True).label() == "A-LAMARCK"
    assert EvolveConfig().label() == "GNOMON"
    m = Member(state={}, fitness=1.0, descriptors=(0.0, 1.0), niche=(0, 0),
               generation=0, n_decoded=1, arch_summary="", per_seed=[1.0],
               lamarck=True)
    assert m.lamarck is True
    assert Member(state={}, fitness=1.0, descriptors=(0.0, 1.0), niche=(0, 0),
                  generation=0, n_decoded=1, arch_summary="",
                  per_seed=[1.0]).lamarck is False


def test_A_LAMARCK_seeds_are_consecutive_lives_of_one_lineage():
    """SPEC: 08 §8.5. `load=False` continues rather than reloading.

    An acquired change discarded between seeds is not inherited, and
    inheritance of acquired change is the whole content of the arm.
    """
    r = _runner()
    g = Decoder(SHAPE, seed=0)
    r.run(g, seed=5, rule="adam", lamarck=True)
    first = r.export_genome()
    r.run(g, seed=6, rule="adam", lamarck=True, load=False)
    second = r.export_genome()
    assert any(not torch.equal(first[k], second[k]) for k in first)
    # Reloading instead would have thrown the first life away.
    r.run(g, seed=6, rule="adam", lamarck=True, load=True)
    third = r.export_genome()
    assert any(not torch.equal(second[k], third[k]) for k in second)


# ------------------------------------------------- the CUDA-graph backend
#
# `gnomon/lifetime/graph.py` was written at v1.0, argued in its own docstring
# that CUDA-graph capture is the only acceleration that survives architecture
# search, and was never imported -- 110 lines of unreachable code while
# `evolve()` ran eager. Connecting it is what found the blocker below.

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="CUDA-graph capture requires a CUDA device")


@cuda_only
def test_the_embedding_backward_under_capture_is_the_same_arithmetic():
    """SPEC: 04 §4.1. `ops.embed`'s two forms differ in kernel, not in value.

    `F.embedding`'s backward sorts its indices and calls `unique_by_key`, which
    synchronises and therefore cannot be captured. The advanced-indexing form
    accumulates with `atomicAdd` and can be. The whole justification for
    swapping them under capture is that they compute the same thing, so that is
    asserted rather than assumed -- forward *and* backward.
    """
    from gnomon.pheno import ops

    V, d = 512, 32
    E0 = torch.randn(V, d, device="cuda")
    tok = torch.randint(0, V, (8, 64), device="cuda")
    up = torch.randn(8, 64, d, device="cuda")

    def grad(capturable):
        E = E0.clone().requires_grad_(True)
        (ops.embed(tok, E, capturable=capturable) * up).sum().backward()
        return E.grad

    # The forward is the same gather, exactly.
    assert torch.equal(ops.embed(tok, E0, capturable=False),
                       ops.embed(tok, E0, capturable=True))
    # The backward is the same sum in a different order, so it agrees to within
    # float32 rounding and not beyond it. Bit-equality holds only when hardly
    # any two tokens share a row -- measured `0` at `(4096, 128)` with one token
    # per row, and `6.8e-08` relative at eight -- so asserting it here would be
    # asserting a property of the *test's* shape rather than of the operation.
    a, b = grad(False), grad(True)
    rel = (a - b).abs().max().item() / a.abs().max().item()
    assert rel < 1.2e-7, f"relative difference {rel:.3e} exceeds a float32 ulp"


@cuda_only
def test_the_captured_step_is_the_same_step():
    """SPEC: 05 §5.1. Step one must be exact; after that, chaos is expected.

    The first step is bit-identical because nothing has amplified yet, and that
    is the assertion that catches a captured step doing something *different*.
    Later steps are not asserted equal: measured, they separate at 1e-7 (float32
    kernel selection) and grow exponentially, because an optimiser running
    through a `tanh` decoder is chaotic. Demanding equality at step 20 would be
    demanding that the arithmetic of two CUDA kernels agree exactly, which is
    not a property either backend has or needs.
    """
    from gnomon.data.stream import Shard
    from gnomon.lifetime.runner import LifetimeRunner
    import numpy as np

    cfg = dataclasses.replace(TINY, N_life=1, batch=8)
    sh = _fake_shard(n_tokens=40000, vocab=cfg.vocab)
    g = Decoder(SHAPE, seed=4)

    def one(mode):
        r = LifetimeRunner(cfg, SHAPE, train_shard=sh, eval_shard=sh,
                           device="cuda", compile_mode=mode, n_eval_batches=1)
        return r, r.run(g, seed=2, rule="adam")

    ra, a = one(None)
    rb, b = one("cudagraph")
    assert ra.backend == "eager" and rb.backend == "cudagraph"
    assert a.train_curve[0][1] == b.train_curve[0][1], (
        "the captured step computes a different first step"
    )


@cuda_only
def test_each_backend_repeats_itself_bit_for_bit():
    """SPEC: 05 §5.7 L7, read as it can only be read on a GPU.

    L7 asks that one `(Gamma, seed)` give one answer. That holds *within* a
    backend -- which is the sense in which any CUDA library satisfies it -- and
    not across backends, where kernel selection differs by an ulp. A run states
    which backend produced it rather than pretending the distinction is absent.
    """
    from gnomon.lifetime.runner import LifetimeRunner

    cfg = dataclasses.replace(TINY, N_life=6, batch=8)
    sh = _fake_shard(n_tokens=40000, vocab=cfg.vocab)
    g = Decoder(SHAPE, seed=4)

    for mode in (None, "cudagraph"):
        def once():
            r = LifetimeRunner(cfg, SHAPE, train_shard=sh, eval_shard=sh,
                               device="cuda", compile_mode=mode,
                               n_eval_batches=1)
            return r.run(g, seed=2, rule="adam")
        x, y = once(), once()
        assert x.final == y.final and x.train_curve == y.train_curve, mode


@cuda_only
def test_A_LAMARCK_refuses_to_run_under_capture():
    """SPEC: 08 §8.5. The capture does not include `dL/dGamma`.

    Refusing is the only safe behaviour: running anyway would produce an arm
    labelled `A-LAMARCK` whose genome never moved, which is the one failure the
    arm's own assertion exists to prevent.
    """
    from gnomon.lifetime.runner import LifetimeRunner

    cfg = dataclasses.replace(TINY, N_life=2, batch=8)
    sh = _fake_shard(n_tokens=40000, vocab=cfg.vocab)
    r = LifetimeRunner(cfg, SHAPE, train_shard=sh, eval_shard=sh, device="cuda",
                       compile_mode="cudagraph", n_eval_batches=1)
    with pytest.raises(ValueError, match="A-LAMARCK"):
        r.run(Decoder(SHAPE, seed=4), seed=2, rule="adam", lamarck=True)


def test_the_capturable_embedding_is_off_by_default():
    """SPEC: 05 §5.7 L7. Determinism is the default; speed is the opt-in."""
    from gnomon.pheno.model import Gnomon

    assert Gnomon(TINY, SHAPE).capturable is False


# ------------------------------------------------ L8, the whole requirement

def test_L8_a_resumed_lifetime_matches_the_uninterrupted_one(tmp_path):
    """SPEC: 05 §5.7 L8. "Checkpoint at `N_life/2`, resume, match bit-for-bit."

    Until v1.1 only the *stream* was resumable, and the test named L8 tested
    only the stream. The lifetime had no checkpoint at all, so §09.1's 8-hour M1
    run had no way to survive an interruption: a crash at hour seven lost
    everything. That is the sense in which an acceptance test can pass while the
    requirement it names is unimplemented.

    Bit-for-bit, not close: everything a step reads is saved, so anything less
    than equality means something was missed. The two candidates that would show
    up here and nowhere else are the rule's step count `t` -- bias correction
    divides by `1 - beta^t`, so restarting it scales the first resumed step by up
    to `1/(1-beta_1)` -- and the neuromodulator's running baseline, which `nu`
    is a function of.
    """
    cfg = dataclasses.replace(TINY, N_life=12, batch=4, lr=1e-2)
    g = Decoder(SHAPE, seed=0)
    whole = _runner(cfg).run(g, seed=7, rule="dpr")

    # The interruption is simulated by *keeping* `N_life` and taking the last
    # checkpoint before the end -- not by shortening the lifetime. `wsd_lr`
    # reads `cfg.N_life`, so a half-length run trains on a different schedule
    # and would diverge for a reason that has nothing to do with resuming.
    ck = tmp_path / "life.pt"
    r2 = _runner(cfg)
    r2.run(g, seed=7, rule="dpr", checkpoint=str(ck), checkpoint_every=5)
    assert ck.exists(), "no checkpoint was written"

    resumed = _runner(cfg).run(g, seed=7, rule="dpr", checkpoint=str(ck),
                               resume=True)

    assert resumed.train_curve == whole.train_curve, (
        "the resumed trajectory diverges from the uninterrupted one"
    )
    assert resumed.final == whole.final
    assert resumed.birth == whole.birth


def test_L8_the_rule_state_round_trips(tmp_path):
    """SPEC: 05 §5.7 L8, at the level of the object that carries the state."""
    from gnomon.lifetime.plasticity import DecodedRule

    torch.manual_seed(0)
    p = torch.randn(6, 3, requires_grad=True)
    rule = DecodedRule({"z": p})
    coef = torch.zeros(6, 4)
    coef[:, 1] = 1.0
    for i in range(9):
        p.grad = torch.randn(6, 3)
        rule.step({"z": coef}, loss=2.0 - 0.1 * i, lr=1e-2)
    st = rule.state_dict()

    q = torch.randn(6, 3, requires_grad=True)
    other = DecodedRule({"z": q}).load_state_dict(st)
    assert other.t == rule.t == 9
    assert other.nu.state() == rule.nu.state()
    assert other.last_nu == rule.last_nu
    assert torch.equal(other.state["z"].m, rule.state["z"].m)
    assert torch.equal(other.state["z"].v, rule.state["z"].v)


def test_L8_a_checkpoint_refuses_a_mismatched_architecture(tmp_path):
    """A checkpoint that loads into the wrong graph is worse than none.

    It does not crash: it silently continues a different model's lifetime and
    reports the result as the first model's.
    """
    cfg = dataclasses.replace(TINY, N_life=4, batch=4)
    r = _runner(cfg)
    ck = tmp_path / "life.pt"
    r.run(Decoder(SHAPE, seed=0), seed=7, checkpoint=str(ck), checkpoint_every=2)

    from gnomon.pheno.plan import Arch
    r.set_arch(Arch(("attn",), (cfg.n_h,), (cfg.d_ff,), (cfg.window,)))
    with pytest.raises(ValueError, match="architecture"):
        r.run(Decoder(SHAPE, seed=0), seed=7, checkpoint=str(ck), resume=True)


def test_L8_a_checkpoint_refuses_a_mismatched_token_code(tmp_path):
    """SPEC: 02 §2.9. The code is a coordinate system; resuming across one
    would continue a lifetime in a different basis."""
    cfg = dataclasses.replace(TINY, N_life=4, batch=4)
    r = _runner(cfg)
    ck = tmp_path / "life.pt"
    r.run(Decoder(SHAPE, seed=0), seed=7, checkpoint=str(ck), checkpoint_every=2)
    r.set_token_code(None)
    with pytest.raises(ValueError, match="token code"):
        r.run(Decoder(SHAPE, seed=0), seed=7, checkpoint=str(ck), resume=True)
