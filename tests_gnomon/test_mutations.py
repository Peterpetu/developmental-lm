"""G-MUTATE: a catalogue of planted defects the suite must catch.

SPEC: 09 §9.2

The second of the two things SOMA's adversarial review proved were missing.
Agent B found **eight** mutations that reinstated already-registered defects
while 452 tests passed; every one of them was a test that could not fail. Here
mutation testing is a gate rather than a discovery: each entry plants a specific
defect and names the test that must reject it, and *this* file fails if any
planted defect survives.

Each mutation is applied with `monkeypatch`, so it is undone whatever happens,
and the designated test is called directly rather than through a nested pytest
run -- the assertion being exercised is then the real one from the real suite,
not a copy of it that could drift away from it.
"""

from __future__ import annotations

import math

import pytest
import torch

from gnomon.config import GenomeShape, M1, M_PICO
from gnomon.genome import codes as C
from gnomon.genome import decoder as D
from gnomon.pheno.model import Gnomon
from gnomon.qd.archive import Archive

import tests_gnomon.unit.test_arch as t_arch
import tests_gnomon.unit.test_codes as t_codes
import tests_gnomon.unit.test_decoder as t_dec
import tests_gnomon.unit.test_lifetime as t_life
import tests_gnomon.unit.test_pheno as t_pheno
import tests_gnomon.unit.test_plasticity as t_plast
import tests_gnomon.unit.test_qd as t_qd


def _must_fail(fn, *args, label=""):
    """Call `fn`; the planted defect survives if it does not raise."""
    try:
        fn(*args)
    except (AssertionError, ValueError, RuntimeError, TypeError,
            FloatingPointError):
        return
    pytest.fail(f"the planted defect {label!r} SURVIVED {fn.__name__}; that "
                f"test cannot fail and is not evidence of anything")


# ------------------------------------------------------- the decoder itself

def test_M1_severed_decoder_gradient(monkeypatch):
    """SOMA shipped a routing path whose gradient could be severed silently."""
    orig = D.Decoder.decode_many

    def severed(self, *a, **kw):
        return {k: v.detach() for k, v in orig(self, *a, **kw).items()}

    monkeypatch.setattr(D.Decoder, "decode_many", severed)
    _must_fail(t_pheno.test_gradient_reaches_the_genome_and_the_soma_through_the_forward_pass,
               label="severed decoder gradient")


def test_M2_dropped_u_ctx_term(monkeypatch):
    """Losing the context term makes every layer and role decode alike."""
    orig = D.Decoder._readout
    monkeypatch.setattr(D.Decoder, "_readout",
                        lambda self, ur, uc, ux=0.0: orig(self, ur, uc, 0.0))
    _must_fail(t_dec.test_D1_matches_the_entrywise_oracle, 0,
               label="dropped u_ctx")


def test_M3_one_hidden_layer(monkeypatch):
    """The degeneracy §03.1 exists to prevent: rank that does not grow with width."""
    def one_layer(self, u_row, u_col, u_ctx=0.0):
        h1 = torch.tanh(torch.exp(self.log_gain)
                        * (u_row.unsqueeze(1) + u_col.unsqueeze(0) + u_ctx)
                        + self.c1)
        return h1 @ self.w_out + self.o_0

    monkeypatch.setattr(D.Decoder, "_readout", one_layer)
    _must_fail(t_dec.test_D5b_decoded_rank_scales_with_width,
               label="one hidden layer")


def test_M4_unit_gain(monkeypatch):
    """`log_gain = 0` puts tanh in its linear regime; §03.6 measured the cost."""
    orig = D.Decoder.__init__

    def init(self, shape, *, seed=0):
        orig(self, shape, seed=seed)
        with torch.no_grad():
            self.log_gain.zero_()

    monkeypatch.setattr(D.Decoder, "__init__", init)
    _must_fail(t_dec.test_D5a_decoded_rank_clears_the_absolute_floor, M1,
               label="unit gain")


def test_M5_no_per_matrix_normalisation(monkeypatch):
    """Without it a mutant's decoded scale depends on `Gamma` and can explode."""
    orig = D.Decoder.decode_many

    def unnormalised(self, requests, soma, cfg, *, shapes=None, chunk_elems=None):
        out = orig(self, requests, soma, cfg, shapes=shapes,
                   chunk_elems=chunk_elems)
        return {k: v * 3.0 for k, v in out.items()}

    monkeypatch.setattr(D.Decoder, "decode_many", unnormalised)
    _must_fail(t_dec.test_D7_the_grouped_decode_has_the_same_scale,
               label="no per-matrix normalisation")


def test_M6_inner_axes_borrow_channel_identities(monkeypatch):
    """The collision this module actually had: tiling the channel table."""
    orig = D.Decoder._axis

    def borrow(self, kind, n, soma, cache, *, column):
        if kind == "inner":
            t = cache["chan"]
            reps = -(-n // t.shape[0])
            b = t.repeat(reps, 1)[:n]
            z = torch.zeros(n, self.shape.d_z, dtype=b.dtype, device=b.device)
            if column:
                b = C.column_codes(b)
            return torch.cat([b, z], dim=-1)
        return orig(self, kind, n, soma, cache, column=column)

    monkeypatch.setattr(D.Decoder, "_axis", borrow)
    _must_fail(t_dec.test_inner_axes_do_not_collide, label="tiled inner identities")


# --------------------------------------------------------------- the codes

def test_M7_zero_one_code_polarity(monkeypatch):
    """`{0,1}` makes low-index identities systematically smaller in norm."""
    def zero_one(idx, n_bits):
        shifts = torch.arange(n_bits, dtype=torch.int64, device=idx.device)
        return ((idx.unsqueeze(-1) >> shifts) & 1).to(torch.float32)

    monkeypatch.setattr(C, "bits_to_pm1", zero_one)
    _must_fail(t_codes.test_T1_codomain_is_pm1, label="{0,1} code polarity")


def test_M8_prefix_coding(monkeypatch):
    """The naive construction §02.3 rejects: three bit positions never vary."""
    monkeypatch.setattr(C, "seeded_bijection",
                        lambda n_bits, seed: torch.arange(1 << n_bits,
                                                          dtype=torch.int64))
    _must_fail(t_codes.test_T5_no_bit_position_is_constant, label="prefix coding")


def test_M9_absolute_layer_index(monkeypatch):
    """Absolute depth makes cross-`L` transfer extrapolation, not interpolation."""
    def absolute(L, shape, device="cpu"):
        idx = torch.arange(L, dtype=torch.int64, device=device)
        return C.bits_to_pm1(idx, shape.n_L)

    monkeypatch.setattr(C, "layer_codes", absolute)
    _must_fail(t_codes.test_T8_depth_is_relative_not_absolute,
               label="absolute layer index")


# ------------------------------------------------------------ the lifetime

def test_M10_stale_decode(monkeypatch):
    """A decode hoisted out of the training loop severs `dL/dZ` silently."""
    cache: dict = {}
    orig = Gnomon.decode

    def stale(self):
        if "W" not in cache:
            cache["W"] = {k: v.detach() for k, v in orig(self).items()}
        return cache["W"]

    monkeypatch.setattr(Gnomon, "decode", stale)
    _must_fail(t_life.test_L1_the_decode_is_inside_the_training_loop,
               label="stale decode")


def test_M11_rule_coefficients_attached_to_the_graph(monkeypatch):
    """A lifetime could then minimise loss by switching its plasticity off."""
    from gnomon.lifetime.plasticity import coefficients_from

    def attached(self, W=None):
        W = self.decode() if W is None else W
        chan = torch.stack([W[(13, l)] for l in range(self.arch.L)]).mean(0)
        return {"z": coefficients_from(chan),
                "zeta": coefficients_from(W[(14, -1)])}

    monkeypatch.setattr(Gnomon, "plasticity", attached)
    _must_fail(t_plast.test_P3_nothing_in_the_rule_is_differentiable,
               label="rule coefficients attached")


def test_M12_the_rule_collapses_to_a_fixed_optimiser(monkeypatch):
    """SPEC: 10 §10.1. The defect §10 exists to remove, planted back.

    A rule pinned to `ADAM_POINT` whatever the genome says is v0.9's design with
    the ornament removed: the lifetime is then stock deep learning and the
    genome has no say in how anything changes.
    """
    from gnomon.config import N_RULE
    from gnomon.lifetime.plasticity import ADAM_POINT

    def pinned(self, W=None):
        out = {}
        for k, p in (("z", self.soma.z), ("zeta", self.soma.zeta)):
            c = torch.zeros(p.shape[0], N_RULE)
            for j, v in enumerate(ADAM_POINT):
                c[:, j] = v
            out[k] = c
        return out

    monkeypatch.setattr(Gnomon, "plasticity", pinned)
    _must_fail(t_plast.test_P2_the_coefficients_are_decoded_from_the_genome,
               label="the rule ignores the genome")


def test_M12b_self_adaptation_applied_after_use(monkeypatch):
    """SPEC: 10 §10.4 P6. Using the old strength severs selection from it."""
    from gnomon.genome import arch as A

    orig = A.mutate

    def after(modules, *, sigma_rel=0.15, p_mut=0.25, generator=None,
              strategy=None):
        orig(modules, sigma_rel=sigma_rel, p_mut=p_mut, generator=generator,
             strategy=None)                       # mutate with the OLD strength

    monkeypatch.setattr(A, "mutate", after)
    monkeypatch.setattr(t_plast, "mutate", after)
    _must_fail(t_plast.test_P6_strengths_are_mutated_before_they_are_used,
               label="self-adaptation applied after use")


def test_M13_annealing_never_starts_from_W_rand(monkeypatch):
    """`eps(0)` must be exactly 1, or the developmental phase does nothing."""
    bad = lambda n, cfg: 0.5 * math.exp(-n / cfg.lam)
    # Both bindings: the test module imported `eps_at` by value at import time,
    # so patching only the defining module would plant nothing at all.
    monkeypatch.setattr("gnomon.lifetime.train.eps_at", bad)
    monkeypatch.setattr(t_life, "eps_at", bad)
    _must_fail(t_life.test_L5_annealing_endpoints, label="eps(0) != 1")


# ------------------------------------------------------- the free-weight gate

def test_M14_a_planted_W_shared(monkeypatch):
    """SOMA's own defect: a dense projection on the residual path."""
    orig = Gnomon.__init__

    def init(self, cfg, shape, **kw):
        orig(self, cfg, shape, **kw)
        self.W_shared = torch.nn.Parameter(torch.zeros(cfg.d_model, cfg.d_model))

    monkeypatch.setattr(Gnomon, "__init__", init)
    _must_fail(t_pheno.test_P5_G_NOFREE_the_model_holds_only_Gamma_and_Z,
               label="planted W_shared")


# ---------------------------------------------------------- the architecture

def test_M15_the_architecture_head_reads_the_soma(monkeypatch):
    """C3 requires the architecture to be innate."""
    from gnomon.genome import arch as A

    orig = A.ArchHead.decode_arch

    def reads_soma(self, cfg):
        soma = None                      # the name alone is what T2 forbids
        return orig(self, cfg)

    monkeypatch.setattr(A.ArchHead, "decode_arch", reads_soma)
    _must_fail(t_arch.test_T2_the_architecture_is_innate, label="arch reads soma")


def test_M16_mutation_never_changes_the_graph(monkeypatch):
    """Then it is not architecture search, whatever the spec says."""
    from gnomon.genome import arch as A

    orig = A.mutate

    def only_weights(modules, **kw):
        orig([m for m in modules if not isinstance(m, A.ArchHead)], **kw)

    monkeypatch.setattr(A, "mutate", only_weights)
    monkeypatch.setattr(t_arch, "mutate", only_weights)
    _must_fail(t_arch.test_T7_mutation_changes_the_graph,
               label="mutation spares the architecture head")


def test_M17_absolute_mutation_sigma(monkeypatch):
    """Destroys a shrunken tensor and leaves a grown one untouched."""
    from gnomon.genome import arch as A

    def absolute(modules, *, sigma_rel=0.15, p_mut=0.25, generator=None):
        with torch.no_grad():
            for mod in modules:
                for p in mod.parameters():
                    if p.numel():
                        p.add_(torch.randn(p.shape, generator=generator) * sigma_rel)

    monkeypatch.setattr(A, "mutate", absolute)
    monkeypatch.setattr(t_arch, "mutate", absolute)
    _must_fail(t_arch.test_T10_mutation_is_scale_relative,
               label="absolute mutation sigma")


# ---------------------------------------------------------------- the archive

def test_M18_non_elitist_insertion(monkeypatch):
    """A worse genome displacing an elite makes `qd_score` non-monotone."""
    def always(self, member):
        self.n_attempts += 1
        self.cells[member.niche] = member
        self.n_inserts += 1
        return True

    monkeypatch.setattr(Archive, "insert", always)
    _must_fail(t_qd.test_E1_elitism_a_worse_genome_never_displaces_an_elite,
               label="non-elitist insertion")
    _must_fail(t_qd.test_E1_qd_score_is_non_decreasing_over_a_run,
               label="non-elitist insertion")


def test_M19_coverage_counts_attempts(monkeypatch):
    monkeypatch.setattr(Archive, "coverage",
                        lambda self: self.n_attempts / self.n_niches)
    _must_fail(t_qd.test_E8_coverage_counts_filled_niches_not_attempts,
               label="coverage counts attempts")


def test_M20_parents_sampled_by_fitness_not_by_niche(monkeypatch):
    """Greedy parent choice removes the mechanism that keeps stepping stones."""
    def greedy(self, k, generator):
        elites = sorted(self.cells.values(), key=lambda m: m.fitness)
        return [elites[0]] * k

    monkeypatch.setattr(Archive, "sample_parents", greedy)
    _must_fail(t_qd.test_parents_are_sampled_uniformly_over_filled_niches,
               label="greedy parent sampling")


def test_M21_the_budget_is_not_exact(monkeypatch):
    from gnomon.qd import mapelites as ME

    orig = ME.map_elites

    def short(archive, **kw):
        kw["G_gen"] = max(0, kw["G_gen"] - 1)
        try:
            return orig(archive, **kw)
        except AssertionError:
            raise AssertionError("budget assertion fired, as it must")

    monkeypatch.setattr(ME, "map_elites", short)
    monkeypatch.setattr(t_qd, "map_elites", short)
    _must_fail(t_qd.test_E9_the_budget_is_exact, label="short budget")


# -------------------------------------------------- the Weismann barrier

def test_M22_the_lifetime_writes_the_germline(monkeypatch):
    """SPEC: 05 §5.1 line 9. Lamarck by accident rather than by declaration.

    Added in v1.1, because L3 had been asserting a *side effect* of the
    duplicate lifetime implementation -- that `lifetime()` cleared
    `requires_grad` on the `Decoder` it was handed -- rather than that the
    germline is unwritten. Under one implementation that assertion became
    vacuously true, so the catalogue must show the restated one can still fail.

    The plant is the honest form of the defect: a lifetime that nudges `Gamma`
    by a fraction of the soma's own step. §01.7's barrier is a theorem about
    capacity and this does not violate it -- but §08's fitness would then be
    measuring a genome no generation ever selected, and every artifact of the
    real Lamarckian arm is required to be labelled precisely so that this
    cannot happen silently.
    """
    from gnomon.lifetime.plasticity import DecodedRule

    orig = DecodedRule.step
    leaked: dict = {}

    def leaky(self, coeffs, loss, lr):
        orig(self, coeffs, loss, lr)
        for p in leaked.get("gamma", []):
            p.add_(torch.randn_like(p), alpha=1e-6)

    import gnomon.lifetime.runner as R
    orig_run = R.LifetimeRunner.run

    def run(self, genome, **kw):
        leaked["gamma"] = list(self.model.genome_parameters())
        return orig_run(self, genome, **kw)

    monkeypatch.setattr(DecodedRule, "step", leaky)
    monkeypatch.setattr(R.LifetimeRunner, "run", run)
    _must_fail(t_life.test_L3_the_genome_is_not_written_during_a_lifetime,
               label="the lifetime writes the germline")


def test_M23_the_lamarckian_path_is_severed(monkeypatch):
    """SPEC: 08 §8.5. An "upper bound" that bounds nothing.

    `A-LAMARCK` is reported as the ceiling the Darwinian arms are measured
    against, so a severed gradient into `Gamma` does not produce a visibly
    broken run -- it produces a *plausible* one, numerically identical to the
    Darwinian arm and labelled as the bound. That is §00's RC1 in the
    instrument rather than in the model, and it is the reason the arm asserts
    that `Gamma` moved rather than trusting that it did.

    The plant detaches the decoded weights, which is the natural way to break
    it: everything still runs, the loss still falls, and the genome never
    hears about it.
    """
    from gnomon.pheno.model import Gnomon as G

    orig = G.decode
    monkeypatch.setattr(G, "decode",
                        lambda self: {k: v.detach() for k, v in orig(self).items()})
    _must_fail(
        t_life.test_A_LAMARCK_opens_the_gradient_path_into_every_genome_tensor,
        label="detached decode in A-LAMARCK")


def test_M24_the_lamarckian_arm_leaks_into_the_next_lifetime(monkeypatch):
    """SPEC: 05 §5.1 line 9. Stickiness would stop the arms being arms.

    One Lamarckian lifetime inside a process would make every later one
    Lamarckian too, so a Darwinian control run in the same process would
    silently be a second Lamarckian run -- and the control is the whole point.
    """
    import gnomon.lifetime.runner as R

    orig = R.LifetimeRunner.run

    def sticky(self, genome, **kw):
        res = orig(self, genome, **kw)
        if kw.get("lamarck"):
            for prm in self.model.genome_parameters():
                prm.requires_grad_(True)          # left open
        return res

    monkeypatch.setattr(R.LifetimeRunner, "run", sticky)
    _must_fail(t_life.test_A_LAMARCK_closes_the_path_again_before_returning,
               label="Lamarckian path left open")


def test_M25_the_declared_soma_drifts_from_what_is_optimised(monkeypatch):
    """SPEC: 04 §4.6, second paragraph. The bypass the model check cannot see.

    §04.6's first half walks the model and finds exactly `Gamma` union `Z`.
    That stays true while the *declaration* and the *optimisers* disagree, and
    a lifetime is defined by what moves. SOMA's `W_shared` did not have to be
    declared anywhere to carry 6.98 of that model's 6.76 nats.

    The plant drops `y` from the declared soma. Nothing raises: the model still
    holds the same tensors, the loss still falls, `y` is still optimised -- and
    it is now optimised without being declared, which is the half of the
    sentence that has no other test.
    """
    from gnomon.pheno.model import Gnomon as G

    orig = G.soma_parameters
    monkeypatch.setattr(G, "soma_parameters",
                        lambda self: [p for p in orig(self)
                                      if p is not self.soma.y])
    _must_fail(
        t_pheno.test_P5_G_NOFREE_the_optimizers_cover_Z_and_in_the_darwinian_arms_nothing_else,
        label="the declared soma omits `y`")


def test_the_catalogue_is_complete():
    """Every mutation in this module is a test; none was left defined but unused."""
    import inspect
    import sys

    mod = sys.modules[__name__]
    planted = [n for n, _ in inspect.getmembers(mod, inspect.isfunction)
               if n.startswith("test_M")]
    assert len(planted) >= 26, planted
    nums = sorted({int("".join(c for c in n.split("_")[1][1:] if c.isdigit()))
                   for n in planted})
    assert nums == list(range(1, len(nums) + 1)), f"gaps in the catalogue: {nums}"
