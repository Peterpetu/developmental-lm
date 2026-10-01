"""§08.4's baselines and §08.6's ablations: every switch changes something.

SPEC: 08 §8.4
SPEC: 08 §8.6

Gate `G-ABLATION`. §08.6 states that *"every novel component carries exactly one
ablation and one kill criterion"*. v3.0 could reach four of them: A2, A3, A12 and
B1. **A7, A9, A10, B2 and B3 were fully specified and had no switch anywhere**
(audit A-80), so §05.11's falsifiers for the router and the context gate, §01.8's
falsifiers for the Moment regime, and kill criteria K7 and K10 could not be
evaluated at all.

Each test below asserts that flipping the switch **changes the model's output**.
An ablation flag that is wired but inert is worse than a missing one: the arm
runs, the report names it, and the comparison is against the same model.
"""

from __future__ import annotations

import pytest
import torch

from soma.config import M_NANO
from soma.germline import Genome
from soma.model.model import SomaModel
from soma.utils.rng import RngBundle


@pytest.fixture(scope="module")
def built():
    rngs = RngBundle.create(5, device="cpu")
    genome = Genome.initialise(M_NANO, generator=torch.Generator().manual_seed(5))
    rngs.seed_global("init")
    model = SomaModel(M_NANO)
    model.seed_populations(genome, rngs.np("init"), rngs.pt("init"))
    model.eval()
    tokens = torch.randint(0, M_NANO.V, (2, 32), generator=torch.Generator().manual_seed(1))
    return model, genome, tokens


def _logits(model, genome, tokens):
    with torch.no_grad():
        return model(tokens, genome=genome, want_metabolic=False).logits.clone()


def test_a7_unsigned_beta_changes_the_mixer(built) -> None:
    """A7: `beta_mix = sigmoid`, range `[0,1]`. §05.2.2's signed transitions need
    `beta > 1` to reach a negative eigenvalue, so A7 removes exactly the
    mechanism K7 is adjudicated against."""
    model, genome, tokens = built
    before = _logits(model, genome, tokens)
    flipped = 0
    for m in model.modules():
        if hasattr(m, "beta_unsigned"):
            m.beta_unsigned = True
            flipped += 1
    assert flipped > 0, "no module carries the A7 switch"
    after = _logits(model, genome, tokens)
    for m in model.modules():
        if hasattr(m, "beta_unsigned"):
            m.beta_unsigned = False
    assert not torch.allclose(before, after), "A7 did not change the forward"


def test_a7_bounds_beta_below_one(built) -> None:
    """The property, not just a difference: under A7 no `beta` exceeds 1, so no
    transition eigenvalue can be negative."""
    model, _, tokens = built
    mixer = next(m for m in model.modules() if hasattr(m, "beta_unsigned"))
    x = torch.randn(2, 16, M_NANO.d_model)
    with torch.no_grad():
        mixer.beta_unsigned = False
        _, _, _, _, beta_full = mixer.features(x)
        mixer.beta_unsigned = True
        _, _, _, _, beta_abl = mixer.features(x)
        mixer.beta_unsigned = False
    assert float(beta_full.max()) > 1.0, "the full model never reaches beta > 1 here"
    assert float(beta_abl.max()) <= 1.0


def test_a10_constant_context_gate_changes_the_site(built) -> None:
    """A10: `psi == 1`. A unit's expression stops depending on the token."""
    model, genome, tokens = built
    before = _logits(model, genome, tokens)
    for m in model.modules():
        if hasattr(m, "psi_off"):
            m.psi_off = True
    after = _logits(model, genome, tokens)
    site = model.sites[0]
    with torch.no_grad():
        psi = site.psi(torch.randn(2, 8, M_NANO.d_model))
    for m in model.modules():
        if hasattr(m, "psi_off"):
            m.psi_off = False
    assert not torch.allclose(before, after), "A10 did not change the forward"
    assert torch.equal(psi, torch.ones_like(psi)), "psi is not identically 1"


def test_b3_random_router_freezes_and_goes_uniform(built) -> None:
    """B3: routing fixed at initialisation, `r_ut = 1/k_route`.

    Both halves are asserted. Uniform weights alone would still be a *learned*
    router with its softmax replaced; frozen parameters alone would still weight
    by an arbitrary random scale.
    """
    model, genome, tokens = built
    before = _logits(model, genome, tokens)
    router = model.sites[0].router
    trainable_before = [p.requires_grad for p in router.parameters()]
    assert any(trainable_before), "the router had no trainable parameters to begin with"

    for site in model.sites:
        site.router.freeze_for_b3()
    after = _logits(model, genome, tokens)

    assert not any(p.requires_grad for p in router.parameters())
    assert not torch.allclose(before, after), "B3 did not change the forward"

    with torch.no_grad():
        out = model(tokens, genome=genome, want_metabolic=False)
    r = out.site_outputs[0].r_ut
    live = out.site_outputs[0].slots >= 0
    n = live.sum(-1, keepdim=True).clamp_min(1)
    expected = live.to(r.dtype) / n.to(r.dtype)
    assert torch.allclose(r, expected, atol=1e-6), "r_ut is not uniform under B3"

    for site in model.sites:
        site.router.random_router = False
        for p in site.router.parameters():
            p.requires_grad_(True)


def test_the_lifetime_config_exposes_every_specified_arm() -> None:
    """§08.4 and §08.6 name ten arms that are not scalar pins. Each must be
    reachable, or the ablation table is a list of experiments nobody can run."""
    from soma.train import LifetimeConfig

    fields = set(LifetimeConfig.__dataclass_fields__)
    for name in ("use_mean_utility", "enable_selection", "beta_unsigned",
                 "freeze_body_plan", "psi_off", "pre_gate", "dense_sites",
                 "frozen_field", "random_router"):
        assert name in fields, f"{name} is not a LifetimeConfig field"


def test_the_driver_maps_every_arm_to_exactly_one_switch() -> None:
    """The CLI's arm table, read from the driver itself so it cannot drift."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "scripts"
           / "train_tinystories.py").read_text(encoding="utf-8")
    block = re.search(r"arm = \{(.*?)\}\.get\(args\.arm", src, re.S)
    assert block, "the driver's arm table is no longer parseable"
    entries = re.findall(r'"(A\d+|B\d+)":\s*dict\((\w+)=', block.group(1))
    got = dict(entries)
    assert got == {
        "A2": "use_mean_utility",
        "A3": "enable_selection",
        "A7": "beta_unsigned",
        "A9": "freeze_body_plan",
        "A10": "psi_off",
        "A12": "pre_gate",
        "B1": "dense_sites",
        "B2": "frozen_field",
        "B3": "random_router",
    }, got


def test_the_scalar_ablations_are_reachable_by_pin() -> None:
    """A1a, A1b, A5, A6 and A11 change a constant, not the forward, and §02.2.3
    reaches them through `ablation.pin` — which bypasses the link so an endpoint
    value like `lambda_wire = 0` is attainable at all."""
    cfg = M_NANO.with_pins(lambda_wire=0.0, k_tok=1.0, f_oe=0.0, p_rec=0.0)
    assert cfg.policy.lambda_wire == 0.0
    assert cfg.policy.k_tok == 1.0
    assert cfg.policy.f_oe == 0.0
    assert cfg.policy.p_rec == 0.0
    # A1b is on `U_cap`, not on `lambda_size`, because `L_size` has identically
    # zero gradient and an ablation setting it to zero would be a provable no-op
    # (audit A-31).
    assert M_NANO.with_pins(U_cap=64).policy.U_cap == 64
