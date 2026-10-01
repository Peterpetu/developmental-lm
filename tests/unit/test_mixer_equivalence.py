"""The chunkwise mixer against the naive-recurrence oracle.

SPEC: 05 §5.2.5
SPEC: 05 §5.2.6

Gates `G-MIXER`, `G-MIXER-UNGATED`, `G-UTCOND`.

**[NORMATIVE]** `tests/unit/test_mixer_equivalence.py` asserts agreement to
`1e-5` (`float32`) and `1e-10` (`float64`) for `C_chunk` in `{16,32,64}` and
`T_ctx` in `{1,31,32,33,63,64,65,512}`.

The `alpha == 1` case is pinned separately (`G-MIXER-UNGATED`) because the v1
error — omitting `Gam_decay` from (5.16) — is correct **exactly** in that limit.
A test that only exercised the ungated case would have passed against the wrong
equation, which is the failure mode K6 exists to prevent.
"""

from __future__ import annotations

import pytest
import torch

from soma.model.mixer_chunk import (
    BETA_MAX,
    gated_delta_rule_chunk,
    ut_condition_number,
)
from soma.model.mixer_ref import gated_delta_rule_reference, transition_eigenvalues

import soma.model.mixer_chunk as _mc

_mc.CHECK_FINITE = True
"""The per-solve finiteness assertions are a synchronisation each, so the
training loop leaves them off (§07.4.3 checks finiteness once per step
instead). The tests turn them on: here the cost is irrelevant and a
localised message is worth more."""

CHUNKS = [16, 32, 64]
LENGTHS = [1, 31, 32, 33, 63, 64, 65, 512]


def _inputs(B, H, T, d_k, d_v, dtype, *, alpha_one=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, d_k, generator=g, dtype=dtype)
    k = torch.randn(B, H, T, d_k, generator=g, dtype=dtype)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, T, d_v, generator=g, dtype=dtype)
    if alpha_one:
        a = torch.ones(B, H, T, dtype=dtype)
    else:
        a = torch.sigmoid(torch.randn(B, H, T, generator=g, dtype=dtype))
    b = (2 * torch.sigmoid(torch.randn(B, H, T, generator=g, dtype=dtype))).clamp(
        max=BETA_MAX
    )
    return q, k, v, a, b


@pytest.mark.parametrize("C", CHUNKS)
@pytest.mark.parametrize("T", LENGTHS)
def test_float64_agreement_to_1e_10(C: int, T: int) -> None:
    """§05.2.6's `float64` bar. `C_chunk` need not divide `T_ctx`."""
    q, k, v, a, b = _inputs(2, 3, T, 16, 16, torch.float64, seed=T * 100 + C)
    S0 = torch.randn(2, 3, 16, 16, dtype=torch.float64) * 0.1
    o_ref, s_ref = gated_delta_rule_reference(q, k, v, a, b, initial_state=S0)
    o_ch, s_ch = gated_delta_rule_chunk(
        q, k, v, a, b, chunk_size=C, initial_state=S0
    )
    scale = float(o_ref.abs().max()) + 1e-12
    assert float((o_ref - o_ch).abs().max()) / scale < 1e-10
    assert float((s_ref - s_ch).abs().max()) / (float(s_ref.abs().max()) + 1e-12) < 1e-10


@pytest.mark.parametrize("C", [32, 64])
@pytest.mark.parametrize("T", [33, 65, 512])
def test_float32_agreement_to_1e_5(C: int, T: int) -> None:
    q, k, v, a, b = _inputs(2, 4, T, 64, 64, torch.float32, seed=T + C)
    o_ref, _ = gated_delta_rule_reference(q, k, v, a, b)
    o_ch, _ = gated_delta_rule_chunk(q, k, v, a, b, chunk_size=C)
    scale = float(o_ref.abs().max()) + 1e-8
    assert float((o_ref - o_ch).abs().max()) / scale < 1e-5


def test_solve_dtype_is_a_floor_not_an_exact_dtype() -> None:
    """§05.9 / audit A-47.

    v3.0 required `float32` "regardless of ambient dtype", which downcasts a
    `float64` call and makes §05.2.6's own `1e-10` test unsatisfiable — the
    realised error is then `~2e-7`, the `float32` epsilon, and `G-MIXER` could
    never pass on a correct implementation.
    """
    q, k, v, a, b = _inputs(1, 2, 64, 16, 16, torch.float64, seed=7)
    o_ref, _ = gated_delta_rule_reference(q, k, v, a, b)
    o_ch, _ = gated_delta_rule_chunk(q, k, v, a, b, chunk_size=32)
    assert float((o_ref - o_ch).abs().max()) < 1e-12, (
        "float64 precision is not being preserved through the triangular solve"
    )


def test_padding_and_validity_mask() -> None:
    """§05.2.5: sequences are padded to a multiple of `C_chunk` and pad
    positions are masked."""
    q, k, v, a, b = _inputs(2, 2, 50, 16, 16, torch.float64, seed=3)
    valid = torch.ones(2, 2, 50, dtype=torch.bool)
    valid[:, :, 40:] = False
    o_ref, s_ref = gated_delta_rule_reference(q, k, v, a, b, valid=valid)
    o_ch, s_ch = gated_delta_rule_chunk(q, k, v, a, b, chunk_size=32, valid=valid)
    assert float((o_ref - o_ch).abs().max()) < 1e-10
    assert float(o_ch[:, :, 40:].abs().max()) == 0.0, "pad positions must emit zero"


def test_state_carries_across_calls() -> None:
    """Splitting a sequence and threading the state must equal one call."""
    q, k, v, a, b = _inputs(1, 2, 96, 16, 16, torch.float64, seed=9)
    o_full, s_full = gated_delta_rule_chunk(q, k, v, a, b, chunk_size=32)
    o1, s1 = gated_delta_rule_chunk(
        q[:, :, :64], k[:, :, :64], v[:, :, :64], a[:, :, :64], b[:, :, :64],
        chunk_size=32,
    )
    o2, s2 = gated_delta_rule_chunk(
        q[:, :, 64:], k[:, :, 64:], v[:, :, 64:], a[:, :, 64:], b[:, :, 64:],
        chunk_size=32, initial_state=s1,
    )
    assert float((torch.cat([o1, o2], dim=2) - o_full).abs().max()) < 1e-10
    assert float((s2 - s_full).abs().max()) < 1e-10


# ---------------------------------------------------------------------------
# G-MIXER-UNGATED: pin the v1 error
# ---------------------------------------------------------------------------


def _v1_chunk(q, k, v, a, b, C):
    """Spec v1's (5.16), **without** `Gam_decay`. Deliberately wrong.

    Reproduced here so the difference is measured rather than asserted. Keeping
    the wrong form in the test suite is what makes `G-MIXER-UNGATED` a real
    control: it demonstrates the error is detectable, and that it is detectable
    **only** when `alpha != 1`.
    """
    import soma.model.mixer_chunk as mc

    B, H, T, dk = q.shape
    dv = v.shape[-1]
    q_p, pad = mc._pad_to_multiple(q, C, 2)
    k_p, _ = mc._pad_to_multiple(k, C, 2)
    v_p, _ = mc._pad_to_multiple(v, C, 2)
    a_p, _ = mc._pad_to_multiple(a, C, 2, 1.0)
    b_p, _ = mc._pad_to_multiple(b, C, 2, 0.0)
    Tp = q_p.shape[2]
    N = Tp // C
    dt = torch.promote_types(q.dtype, torch.float32)
    q_c = q_p.reshape(B, H, N, C, dk).to(dt)
    k_c = k_p.reshape(B, H, N, C, dk).to(dt)
    v_c = v_p.reshape(B, H, N, C, dv).to(dt)
    a_c = a_p.reshape(B, H, N, C).to(dt)
    b_c = b_p.reshape(B, H, N, C).to(dt).clamp(0, mc.BETA_MAX)
    log_g = torch.cumsum(torch.log(a_c.clamp_min(1e-30)), -1)
    g = torch.exp(log_g)
    causal = torch.tril(torch.ones(C, C, dtype=torch.bool))
    gam = torch.where(
        causal,
        torch.exp((log_g.unsqueeze(-1) - log_g.unsqueeze(-2)).clamp(max=0.0)),
        torch.zeros(1, dtype=dt),
    )
    kkt = k_c @ k_c.transpose(-1, -2)
    eye = torch.eye(C, dtype=dt)
    sl = torch.tril(torch.ones(C, C, dtype=dt), -1)
    W = torch.linalg.solve_triangular(
        eye + (b_c.unsqueeze(-1) * kkt) * sl, b_c.unsqueeze(-1) * k_c,
        upper=False, unitriangular=True,
    )
    U = torch.linalg.solve_triangular(
        eye + (b_c.unsqueeze(-1) * (gam * kkt)) * sl, b_c.unsqueeze(-1) * v_c,
        upper=False, unitriangular=True,
    )
    gl = g[..., -1:]
    Qa = q_c * g.unsqueeze(-1)
    Wa = W * g.unsqueeze(-1)
    Ka = k_c * (gl / g).unsqueeze(-1)
    S = torch.zeros(B, H, dv, dk, dtype=dt)
    attn_v1 = (q_c @ k_c.transpose(-1, -2)) * causal.to(dt)  # <-- no Gam_decay
    outs = []
    for c in range(N):
        ST = S.transpose(-1, -2)
        inner = U[:, :, c] - Wa[:, :, c] @ ST
        outs.append(Qa[:, :, c] @ ST + attn_v1[:, :, c] @ inner)
        S = S * gl[:, :, c].unsqueeze(-1) + inner.transpose(-1, -2) @ Ka[:, :, c]
    O = torch.stack(outs, 2).reshape(B, H, Tp, dv)
    return O[:, :, :T, :]


def test_v1_form_is_correct_only_in_the_ungated_limit() -> None:
    """`G-MIXER-UNGATED`.

    The v1 form yields `sum_i (k^iT q^r) u~^i` — missing `g_r/g_i` — and
    `- sum_i (k^iT q^r) g_i (S w^i)^T`, carrying `g_i` where `g_r` is required.
    """
    q, k, v, a, b = _inputs(1, 2, 64, 16, 16, torch.float64, seed=5)
    ref_gated, _ = gated_delta_rule_reference(q, k, v, a, b)
    v3_gated, _ = gated_delta_rule_chunk(q, k, v, a, b, chunk_size=32)
    v1_gated = _v1_chunk(q, k, v, a, b, 32)

    assert float((v3_gated - ref_gated).abs().max()) < 1e-10
    err_v1 = float((v1_gated.double() - ref_gated).abs().max())
    assert err_v1 > 1e-3, (
        f"the v1 error must be detectable under gating; measured {err_v1:.3e}. "
        f"If this assertion fails the test cannot distinguish the two equations "
        f"and K6 has no teeth."
    )

    qo, ko, vo, ao, bo = _inputs(1, 2, 64, 16, 16, torch.float64, alpha_one=True, seed=5)
    ref_un, _ = gated_delta_rule_reference(qo, ko, vo, ao, bo)
    v3_un, _ = gated_delta_rule_chunk(qo, ko, vo, ao, bo, chunk_size=32)
    v1_un = _v1_chunk(qo, ko, vo, ao, bo, 32)
    assert float((v3_un - ref_un).abs().max()) < 1e-10
    assert float((v1_un.double() - ref_un).abs().max()) < 1e-4, (
        "in the ungated limit the two forms must coincide; that they do is the "
        "reason a test run only at alpha == 1 would have passed against the "
        "wrong equation"
    )


# ---------------------------------------------------------------------------
# G-UTCOND
# ---------------------------------------------------------------------------


def test_gradient_is_finite_under_sustained_decay() -> None:
    """(5.14) `K-> = (g_C/g_i) k^i` / audit A-65.

    A decay gate that sits at `alpha ~ 0.2` for a whole chunk drives
    `g_i = prod alpha` to `~1e-45` at `C = 64`. The value is representable in
    float32; `g_i^2` is not. Forming `K->` as the quotient `g_C / g_i` therefore
    has derivative `-g_C / g_i^2 = NaN` **while the forward pass is still
    finite** — which is exactly how it presented in a lifetime: a loss of `4.04`
    and a non-finite gradient reaching only `w_alpha`.

    The threshold matters and is asserted: this must be finite at a decay
    strength where the *quotient* form is not, or the test is measuring the
    easy regime.
    """
    B, H, T, d_k, d_v, C = 1, 1, 64, 16, 16, 64
    q, k, v, _, b_ = _inputs(B, H, T, d_k, d_v, torch.float32, seed=7)
    # alpha ~ 0.22 sustained: sum_j log alpha_j = -96 over the chunk.
    a_raw = torch.full((B, H, T), -1.5, dtype=torch.float32, requires_grad=True)
    a = a_raw.exp()

    O, S = gated_delta_rule_chunk(q, k, v, a, b_, chunk_size=C)
    assert torch.isfinite(O).all(), "forward already non-finite; wrong regime"
    (O.square().sum() + S.square().sum()).backward()
    assert torch.isfinite(a_raw.grad).all(), (
        "gradient through the decay gate is not finite"
    )

    # Positive control: the quotient form, run on the identical inputs, is the
    # thing that fails. Without this the assertion above would pass even if the
    # regime were too mild to distinguish the two forms.
    la = torch.full((1, 1, 1, C), -1.5, dtype=torch.float32, requires_grad=True)
    log_g = torch.cumsum(la, dim=-1)
    gg = torch.exp(log_g)
    quot = gg[..., -1:] / gg
    quot.square().sum().backward()
    assert not torch.isfinite(la.grad).all(), (
        "the quotient form no longer overflows at this decay strength, so the "
        "test above has stopped distinguishing the two forms"
    )


@pytest.mark.parametrize("C", CHUNKS)
def test_ut_conditioning_at_the_beta_clamp(C: int) -> None:
    """§05.2.3: `cond(I + strictlower(diag(beta) K K^T)) < 1e6` at
    `beta = 2 - 1e-3`."""
    g = torch.Generator().manual_seed(C)
    k = torch.randn(8, C, 64, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    beta = torch.full((8, C), BETA_MAX)
    cond = ut_condition_number(beta, k)
    assert float(cond.max()) < 1e6


def test_beta_is_clamped_below_two() -> None:
    """Unbounded `beta` with `||k|| = 1` admits inverses whose entries grow with
    chunk length."""
    assert BETA_MAX < 2.0
    q, k, v, a, b = _inputs(1, 1, 32, 8, 8, torch.float64, seed=1)
    b_huge = torch.full_like(b, 5.0)
    o, _ = gated_delta_rule_chunk(q, k, v, a, b_huge, chunk_size=32)
    assert torch.isfinite(o).all(), "the clamp must keep the solve conditioned"


# ---------------------------------------------------------------------------
# (5.5)
# ---------------------------------------------------------------------------


def test_eigenvalues_match_5_5() -> None:
    """(5.5): `alpha (1 - beta)` along `k`, `alpha` orthogonal to it.

    An earlier draft omitted `alpha` and claimed `1 - beta` and `1`, so `-1`
    looked attainable. It is not: a Householder reflection is approached only as
    `alpha -> 1`.
    """
    alpha = torch.tensor([0.9999])
    beta = torch.tensor([BETA_MAX])
    along, orth = transition_eigenvalues(alpha, beta)
    assert float(along) == pytest.approx(0.9999 * (1.0 - BETA_MAX))
    assert float(orth) == pytest.approx(0.9999)
    assert -float(alpha) < float(along) < float(alpha)
    # §05.2.2's persistence argument: 0.9999^256 = 0.975.
    assert (0.9999 ** 256) == pytest.approx(0.975, abs=1e-3)


def test_negative_eigenvalue_is_reachable_and_flips_state() -> None:
    """Proposition 5.1's premise: the transition must admit a negative
    eigenvalue, or `(11)*` is not recognisable at arbitrary length."""
    d = 4
    k = torch.zeros(1, 1, 2, d, dtype=torch.float64)
    k[..., 0] = 1.0
    v = torch.zeros(1, 1, 2, d, dtype=torch.float64)
    q = torch.zeros(1, 1, 2, d, dtype=torch.float64)
    q[..., 0] = 1.0
    a = torch.ones(1, 1, 2, dtype=torch.float64)
    b = torch.full((1, 1, 2), BETA_MAX, dtype=torch.float64)
    S0 = torch.zeros(1, 1, d, d, dtype=torch.float64)
    S0[0, 0, 0, 0] = 1.0
    o, s = gated_delta_rule_reference(q, k, v, a, b, initial_state=S0)
    assert float(o[0, 0, 0, 0]) < 0.0, "one step must flip the sign along k"
    assert float(o[0, 0, 1, 0]) > 0.0, "two steps must restore it"
