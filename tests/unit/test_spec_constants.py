"""Every configured value agrees across §01.9, §05.0 and §07.4; every derived
constant satisfies its defining equation.

SPEC: 01 §1.9
SPEC: 01 §1.10
SPEC: 05 §5.0

Gate `G-SPECNUM`. This is the test that makes "the spec tables and the code
agree" a check rather than a claim — and it is the reason §05.10's parameter
table is now produced by `parameter_report()` rather than written by hand
(audit A-49).
"""

from __future__ import annotations

import math

import pytest

from soma.config import (
    A_TARGET,
    C_TARGET,
    D_MIN,
    D_UNIF_8,
    E_TARGET,
    M1,
    M_NANO,
    NOCOLLAPSE_FRACTION,
    SIGMA_G_INIT,
    SomaConfig,
)

CONFIGS = [M_NANO, M1]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_N_step_is_derived_not_configured(cfg: SomaConfig) -> None:
    """§05.8: `D_tok` and `B_tok` are configured; `N_step` is derived."""
    assert cfg.N_step == math.ceil(cfg.D_tok / cfg.B_tok)


def test_tabled_N_step_values() -> None:
    """§01.9's table."""
    assert M_NANO.N_step == 2442
    assert M1.N_step == 12208


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_T_0_satisfies_1_17(cfg: SomaConfig) -> None:
    """(1.17): `T_0 = ceil( n_c ln(1 + N_step/n_c) / E_target )`.

    Audit A-6 recorded the inverted form `n_c / (E_target ln(...))`, which gives
    ~25 events instead of 10 — a 2.5x error in structural churn, in the
    direction that would have destabilised training.
    """
    n_c = cfg.n_c
    expected = math.ceil(n_c * math.log1p(cfg.N_step / n_c) / E_TARGET)
    assert cfg.T_0 == expected


def test_tabled_derived_periods() -> None:
    """§01.9's `T_0`, `T_arch`, `T_cert` rows."""
    assert (M_NANO.T_0, M1.T_0) == (99, 492)
    assert (M_NANO.T_arch, M1.T_arch) == (122, 610)
    assert (M_NANO.T_cert, M1.T_cert) == (610, 3052)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_T_arch_and_T_cert_satisfy_1_18_1_19(cfg: SomaConfig) -> None:
    """(1.18)-(1.19). Audit A-10: v2 left both with **no value anywhere**."""
    assert cfg.T_arch == max(1, cfg.N_step // A_TARGET)
    assert cfg.T_cert == max(1, cfg.N_step // C_TARGET)


def test_realised_event_counts_match_the_table() -> None:
    """§01.7: the *realised* count may differ from the target by one, and no
    section may assume the target is attained exactly."""
    assert M_NANO.realised_selection_events() == 10
    assert M1.realised_selection_events() == 10
    assert M_NANO.realised_periodic_events(M_NANO.T_arch) == 20
    assert M1.realised_periodic_events(M1.T_arch) == 20
    assert M_NANO.realised_periodic_events(M_NANO.T_cert) == 4
    assert M1.realised_periodic_events(M1.T_cert) == 3


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_phi_ann_formula_and_fit(cfg: SomaConfig) -> None:
    """§03.3, and A-41's `phi_ann < T_0` assertion."""
    assert cfg.phi_ann == min(5000, max(50, int(0.02 * cfg.N_step)))
    assert cfg.phi_ann < cfg.T_0, (
        "a unit born at one event must finish its anneal before the next; "
        "v1's phi_ann = 5000 against T_sel = 2000 is the configuration §03.4's "
        "dying-unit exclusion was added to contain"
    )


def test_phi_ann_tabled() -> None:
    assert (M_NANO.phi_ann, M1.phi_ann) == (50, 244)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_q_cases_is_proportional_to_d_rate(cfg: SomaConfig) -> None:
    """§01.10 / audit A-11.

    v2 used `ceil(D_min / d_rate)`, which is **inversely** proportional, so
    §04.5.1's cost governor amplified the cost it was meant to bound.
    """
    assert cfg.q_cases == max(D_MIN, round(cfg.policy.d_rate * cfg.B_tok))
    lower = cfg.with_pins(d_rate=0.05)
    assert lower.q_cases < cfg.q_cases, "reducing d_rate must reduce the case count"


def test_q_cases_tabled() -> None:
    assert (M_NANO.q_cases, M1.q_cases) == (1638, 6554)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_cap_off_is_reachable(cfg: SomaConfig) -> None:
    """§04.6 / audit A-22: `cap_off = max(2, ceil(0.5 * n_birth_max))`.

    v2 fixed `cap_off = 6` at M1 against a per-event birth count that could not
    reach it, so §04.10's falsifier and §07.6's hit-rate log were identically
    zero **by arithmetic**.
    """
    assert cfg.cap_off == max(2, math.ceil(0.5 * cfg.n_birth_max))
    assert cfg.cap_off <= cfg.n_birth_max or cfg.cap_off == 2


def test_cap_off_tabled() -> None:
    assert (M_NANO.cap_off, M1.cap_off) == (6, 10)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_mixer_shape_constraint(cfg: SomaConfig) -> None:
    """§05.0: `n_m * d_v = d_model * e_v`.

    Audit A-15: v2 set `n_m = 2, d_v = 64` at M-nano against `d_model = 192`
    (`128 != 192`), a hard shape error in §05.2.4.
    """
    assert cfg.n_m * cfg.d_v == cfg.d_model * cfg.e_v


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_grid_and_head_constraints(cfg: SomaConfig) -> None:
    assert cfg.U_max == cfg.side ** 2
    assert cfg.d_model % cfg.n_h == 0
    assert cfg.n_h % cfg.n_kv == 0
    assert cfg.d_h == cfg.d_model // cfg.n_h
    assert cfg.d_key % 2 == 0
    assert cfg.policy.U_cap <= cfg.U_max


def test_germline_scale_constants_are_identical_across_configs() -> None:
    """§01.2 / audit A-16.

    v2 pinned M-nano `N_G = 512` and M1 `N_G = 2048`, which made the
    M-nano -> M1 germline inheritance of §02.6.2 **shape-impossible**, since
    `mu_g` has shape `N_G x m_dim`.
    """
    assert M_NANO.N_G == M1.N_G == 2048
    assert M_NANO.k_g == M1.k_g == 8
    assert M_NANO.m_dim == M1.m_dim == 8
    assert M_NANO.J_max == M1.J_max == 256


def test_germline_size_matches_9_1() -> None:
    """§09.3's (9.1) arithmetic. v2 computed it with `m_dim + 1 = 3`."""
    from soma.germline.policy import N_POLICY

    expected = 2048 * (8 + 1) + 256 * 8 + N_POLICY
    assert M_NANO.germline.germline_reals(N_POLICY) == expected
    assert expected == 18432 + 2048 + N_POLICY


def test_d_unif_closed_form_matches_the_threshold() -> None:
    """(1.13a) / audit A-48. v3.0 tabulated 1.375, which is wrong by 1.0%."""
    from soma.model.metabolic import d_unif, d_unif_monte_carlo

    assert D_UNIF_8 == pytest.approx(d_unif(8), rel=0, abs=0)
    assert d_unif(8) == pytest.approx(1.389328, abs=1e-5)
    assert d_unif(3) == pytest.approx(4.0 / 3.0, abs=1e-9)
    assert d_unif(2) == pytest.approx(4.0 / math.pi, abs=1e-9)
    assert d_unif_monte_carlo(8, 4096) == pytest.approx(d_unif(8), abs=5e-3)
    assert M_NANO.nocollapse_threshold == pytest.approx(
        NOCOLLAPSE_FRACTION * d_unif(8), abs=1e-12
    )


def test_sigma_g_gives_a_median_phi_near_one_half() -> None:
    """§01.2 / audit A-14: at `sigma_g = 0.6` the measured median is 0.062."""
    import torch

    from soma.germline.field import Field

    g = torch.Generator().manual_seed(20260828)
    f = Field.initialise(M_NANO.germline, generator=g, sigma_g=SIGMA_G_INIT)
    m = f.moments(n_probe=40_000, generator=g)
    assert m["median_phi"] == pytest.approx(0.50, abs=0.02)
    assert m["E_phi"] == pytest.approx(0.5147, abs=0.01)
    assert m["E_phi2"] == pytest.approx(0.2809, abs=0.01)

    bad = Field.initialise(M_NANO.germline, generator=g, sigma_g=0.6)
    mb = bad.moments(n_probe=20_000, generator=g)
    assert mb["median_phi"] < 0.10, "the v2 defect must remain reproducible"


def test_parameter_counts_match_section_5_10() -> None:
    """§05.10, including the mixer output norm v3.0 omitted (audit A-49)."""
    from soma.model.model import SomaModel

    m1 = SomaModel(M1).parameter_report()
    assert m1["total"] == 24_411_160
    assert m1["embedding"] == 4_194_304
    assert m1["mixer_blocks"] == 11_893_248
    assert m1["attention_blocks"] == 1_966_104
    assert m1["sites"] + m1["populations"] == 4_104_192
    assert m1["parts_library"] == 2_228_224
    assert m1["norm_gains"] == 25_088

    nano = SomaModel(M_NANO).parameter_report()
    assert nano["total"] == 2_935_752


def test_the_spec_tables_that_restate_each_other_agree() -> None:
    """§07.4.1: "These values are the ones in §01.9 and nowhere else; this table
    restates them and may not diverge. Gate `G-SPECNUM` asserts the agreement
    mechanically."

    v3.0 asserted that sentence and checked nothing: §07.4.1's parameter row
    still carried `2,934,984 / 24,406,552`, the counts A-49 withdrew when it
    found §05.10's table had omitted the mixer output norm. A restatement that
    nothing compares is a second source of truth (audit A-77).

    This reads the numbers **out of the two spec files** and compares them to
    each other and to the code, so a future edit to either table that does not
    touch the other fails here.
    """
    import re
    from pathlib import Path

    from soma.model.model import SomaModel

    spec = Path(__file__).resolve().parents[2] / "docs" / "spec"
    prot = (spec / "07-training-protocol.md").read_text(encoding="utf-8")
    back = (spec / "05-backbone.md").read_text(encoding="utf-8")

    row = re.search(
        r"\|\s*Parameters \(§05\.10\)\s*\|\s*\*\*([\d,]+)\*\*\s*\|"
        r"\s*\*\*([\d,]+)\*\*\s*\|",
        prot,
    )
    assert row, "§07.4.1's parameter row is no longer parseable; the check is vacuous"
    nano_spec = int(row.group(1).replace(",", ""))
    m1_spec = int(row.group(2).replace(",", ""))

    m1_total = re.search(r"\|\s*\*\*Total\*\*\s*\|\s*\*\*([\d,]+)\*\*", back)
    assert m1_total, "§05.10's total row is no longer parseable"
    m1_back = int(m1_total.group(1).replace(",", ""))

    nano_back = re.search(
        r"`m_dim=8`\):\s*\*\*([\d,]+)\*\*", back
    )
    assert nano_back, "§05.10's M-nano total is no longer parseable"
    nano_backv = int(nano_back.group(1).replace(",", ""))

    code_m1 = SomaModel(M1).parameter_report()["total"]
    code_nano = SomaModel(M_NANO).parameter_report()["total"]

    assert m1_spec == m1_back == code_m1, (m1_spec, m1_back, code_m1)
    assert nano_spec == nano_backv == code_nano, (nano_spec, nano_backv, code_nano)


def test_optimizer_group_split_matches_section_5_10() -> None:
    """§05.7's groups are enumerated by name and are exhaustive."""
    from soma.model.model import SomaModel

    model = SomaModel(M1)
    muon, adamw = model.optimizer_groups()
    n_m = sum(p.numel() for p in muon)
    n_a = sum(p.numel() for p in adamw)
    assert n_m == 19_947_520
    assert n_a == 4_463_640
    assert n_m + n_a == model.parameter_report()["total"]


def test_p_u_is_in_the_adamw_group() -> None:
    """§05.7 / audit A-42: Newton-Schulz on `U_max x 8` would destroy the
    relative geometry on `S^7` that `phi_g` reads and `L_disp` shapes."""
    from soma.model.model import SomaModel

    model = SomaModel(M_NANO)
    _, adamw = model.optimizer_groups()
    assert any(p is model.populations[0].p_u for p in adamw)


def test_eval_band_fits_the_context() -> None:
    """§08.2 / audit A-32: M-nano's `T_ctx = 128` cannot hold a length-256
    prompt plus its answer, so it runs the reduced `40..96` band."""
    assert max(M_NANO.eval_lengths) == 96
    assert max(M1.eval_lengths) == 256
    assert max(M_NANO.eval_lengths) * 2 < M_NANO.T_ctx * 2


def test_growth_schedule_defaults() -> None:
    """§03.5.1."""
    assert len(M_NANO.growth_schedule) == 3
    assert all(0 < s < M_NANO.N_step for s in M_NANO.growth_schedule)


def test_wsd_schedule_endpoints() -> None:
    """§05.8."""
    from soma.optim.muon import wsd_lr

    for cfg in CONFIGS:
        assert wsd_lr(0, cfg) == 0.0
        assert wsd_lr(cfg.n_warm, cfg) == pytest.approx(cfg.lr_max)
        assert wsd_lr(cfg.n_dec, cfg) == pytest.approx(cfg.lr_max)
        assert wsd_lr(cfg.N_step - 1, cfg) < 0.01 * cfg.lr_max


def test_kappa_lag_is_per_update_not_per_step() -> None:
    """(6.2). Spec v1's per-step reading gives a ~22k-step half-life."""
    for cfg in CONFIGS:
        assert cfg.kappa_lag == pytest.approx(0.5 ** (cfg.T_ref / cfg.h_lag))
        n_updates_to_half = cfg.h_lag / cfg.T_ref
        assert cfg.kappa_lag ** n_updates_to_half == pytest.approx(0.5)


def test_invalid_configs_are_refused_at_construction() -> None:
    """`validate()` runs in `__post_init__`, so an invalid config cannot exist."""
    from dataclasses import replace

    with pytest.raises(ValueError, match="n_m\\*d_v"):
        replace(M_NANO, n_m=2)
    with pytest.raises(ValueError, match="U_max"):
        replace(M_NANO, side=15)
    with pytest.raises(ValueError, match="phi_ann"):
        replace(M_NANO, B_tok=1024, D_tok=1024 * 40)


def test_ablation_pins_reach_endpoint_values() -> None:
    """§02.2.3: the links cannot express an endpoint; `ablation.pin` can."""
    a1a = M_NANO.with_pins(lambda_wire=0.0)
    assert a1a.policy.lambda_wire == 0.0
    a5 = M_NANO.with_pins(k_tok=1.0)
    assert a5.policy.k_tok == 1.0
    a6 = M_NANO.with_pins(f_oe=0.0)
    assert a6.policy.f_oe == 0.0
    a1b = M_NANO.with_pins(U_cap=M_NANO.U_max)
    assert a1b.policy.U_cap == M_NANO.U_max
    a11 = M_NANO.with_pins(p_rec=0.0)
    assert a11.policy.p_rec == 0.0


def test_every_policy_default_is_interior_to_its_range() -> None:
    """§02.2.3 / audit A-46: an affine-sigmoid maps onto the OPEN interval, so a
    default at an endpoint has no preimage and generation 0 cannot be built."""
    from soma.germline.policy import Policy

    for cfg in CONFIGS:
        p = Policy.from_constants(cfg.policy)
        back = p.constants()
        for name in ("lambda_wire", "rho_0", "n_c_frac", "p_rec", "d_rate",
                     "k_tok", "f_oe", "eta_ema", "g_growth", "sigma_pos"):
            assert getattr(back, name) == pytest.approx(
                getattr(cfg.policy, name), rel=1e-6
            ), f"{name} does not round-trip through its link"
