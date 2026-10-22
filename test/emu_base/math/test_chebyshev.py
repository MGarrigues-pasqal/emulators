import cmath
import math
from unittest.mock import Mock

import pytest
import torch

from test.utils_testing import (
    chebyshev_coefficients_oracle,
    chebyshev_rounding_bound,
    matrix_exp_error,
    oracle_coefficient_error,
    tails,
    truncation_index,
)

dtype = torch.complex128
EPS = torch.finfo(torch.float64).eps
# Largest per-coefficient error of the Miller recurrence, measured against mpmath for
# alpha <= 30000 when the recurrence was first validated.
MILLER_ERROR = 4e-16


def checked_truncation_index(alpha: float, tolerance: float) -> int:
    """
    K from the oracle coefficients, refusing (as a test error) cases where the oracle's
    own error could move K: the tails on both sides of K must be away from `tolerance`
    by more than the summed oracle error.
    """
    coefs = chebyshev_coefficients_oracle(alpha)
    K = truncation_index(coefs, tolerance)
    margin = len(coefs) * (oracle_coefficient_error(alpha) + MILLER_ERROR)
    t = tails(coefs)
    assert t[K] < tolerance - margin, "test case too close to the truncation threshold"
    assert K == 0 or t[K - 1] > tolerance + margin, "test case too close to threshold"
    return K


@pytest.mark.parametrize(
    "alpha", [0.0, 1e-10, 3e-6, 0.5, 1.0, 7.3, 50.0, 300.0, -7.3, -300.0]
)
@pytest.mark.parametrize("tolerance", [1e-4, 1e-8])
def test_exp_chebyshev_coefficients_match_oracle(alpha: float, tolerance: float) -> None:
    """Values, signs of (-i)^k, factor 2 on k >= 1, conjugation for alpha < 0, minimal K."""
    expected = chebyshev_coefficients_oracle(alpha)
    K = checked_truncation_index(alpha, tolerance)

    from emu_base.math.chebyshev import exp_chebyshev_coefficients

    coefs = exp_chebyshev_coefficients(alpha, tolerance)
    assert all(isinstance(c, complex) for c in coefs)
    assert len(coefs) == K + 1  # smallest K with sum_{k>K} |c_k| <= tolerance
    # Budget per coefficient: oracle error (FFT, phase rounding of |alpha| eps) plus the
    # Miller recurrence error. Both tolerances explicit; atol = 2x the budget. A sign or
    # factor-2 bug changes some c_k by |J_k(alpha)| ~ 0.1, far above.
    atol = 2 * (oracle_coefficient_error(alpha) + MILLER_ERROR)
    torch.testing.assert_close(
        torch.tensor(coefs, dtype=dtype), expected[: K + 1], rtol=0.0, atol=atol
    )


@pytest.mark.parametrize("alpha", [0.5, 7.3, 50.0, -50.0])
@pytest.mark.parametrize("tolerance", [1e-4, 1e-10])
def test_exp_chebyshev_coefficients_truncation_bound(alpha: float, tolerance: float) -> None:
    """sum_k c_k T_k(x) is within `tolerance` of exp(-i alpha x) on [-1, 1], ends included."""
    from emu_base.math.chebyshev import exp_chebyshev_coefficients

    coefs = torch.tensor(exp_chebyshev_coefficients(alpha, tolerance), dtype=dtype)
    x = torch.linspace(-1.0, 1.0, 2001, dtype=torch.float64)
    k = torch.arange(len(coefs), dtype=torch.float64)
    series = (coefs * torch.cos(k * torch.arccos(x)[:, None])).sum(dim=1)
    expected = torch.exp(-1j * alpha * x)
    # Budget: truncation <= tolerance since |T_k| <= 1 (the contract of the coefficients);
    # T_k(x) = cos(k arccos x) with |arccos error| <= pi eps, so |error| <= (2 pi k + 2) eps
    # per term; summation of K + 1 terms <= (K + 1) eps sum |c_k|; exp(-i alpha x):
    # (|alpha| + 2) eps. atol = 2x the budget.
    a = coefs.abs()
    rounding = float((a * (2 * math.pi * k + 2)).sum() + len(a) * a.sum()) * EPS
    budget = tolerance + rounding + (abs(alpha) + 2) * EPS
    torch.testing.assert_close(series, expected, rtol=0.0, atol=2 * budget)


@pytest.mark.parametrize("alpha", [math.nan, math.inf, -math.inf])
def test_exp_chebyshev_coefficients_rejects_non_finite(alpha: float) -> None:
    from emu_base.math.chebyshev import exp_chebyshev_coefficients

    with pytest.raises(ValueError):
        exp_chebyshev_coefficients(alpha, 1e-10)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("alpha", [1e-10, 3e-6, 0.5, 7.3, 50.0, -7.3])
@pytest.mark.parametrize("tolerance", [1e-6, 1e-9])
@pytest.mark.parametrize("widening", [0.0, 0.3])
def test_chebyshev_exp_matches_matrix_exp(
    alpha: float, tolerance: float, widening: float
) -> None:
    """
    phase * sum_k c_k T_k(op / 2) v against matrix_exp, for op = 2 (h - c) / r.
    widening = 0: bounds at the extreme eigenvalues (worst case for the rounding of the
    recurrence); widening > 0: loose, off-centre bounds as given by Gershgorin.
    alpha = 1e-10 gives K = 0 (op never applied), alpha = 3e-6 gives K = 1.
    """
    torch.manual_seed(1337)
    dim = 16
    x = torch.randn(dim, dim, dtype=dtype)
    # spectrum away from 0, so that the phase exp(-i dt c) is not trivial
    h = (x + x.mH) / 2 + 3.0 * torch.eye(dim, dtype=dtype)
    eigs = torch.linalg.eigvalsh(h)
    # eigvalsh is backward stable (|error| <~ dim eps ||h||): widen by 4x that so that
    # the spectrum of op / 2 stays inside [-1, 1], where |T_k| <= 1.
    slack = 4 * dim * EPS * eigs.abs().max().item()
    width = (eigs[-1] - eigs[0]).item()
    lower = eigs[0].item() - slack - widening * width
    upper = eigs[-1].item() + slack + 0.7 * widening * width
    c, r = (upper + lower) / 2, (upper - lower) / 2
    dt = alpha / r
    m = 2 * (h - c * torch.eye(dim, dtype=dtype)) / r
    op = Mock(side_effect=lambda y: m @ y)

    v = torch.randn(dim, dtype=dtype)
    v *= 3.7 / v.norm()  # not normalised
    v_copy = v.clone()
    expected = torch.linalg.matrix_exp(-1j * dt * h) @ v

    K = checked_truncation_index(alpha, tolerance)
    coefs = chebyshev_coefficients_oracle(alpha)[: K + 1]
    # Budget (2-norm), proportional to ||v||: truncation <= tolerance; rounding of the
    # recurrence with a local error per term <= eps (dim * max row sum of |m| + 3)
    # (dense product of dim terms, one subtraction, one scaling); matrix_exp estimate.
    # atol = 3x the budget (an elementwise error is at most the 2-norm error).
    local = EPS * (dim * m.abs().sum(dim=1).max().item() + 3)
    budget = v.norm().item() * (
        tolerance + chebyshev_rounding_bound(coefs, local) + matrix_exp_error(h, dt)
    )

    from emu_base.math.chebyshev import chebyshev_exp

    out = chebyshev_exp(op, v, alpha, cmath.exp(-1j * dt * c), tolerance)

    assert op.call_count == K  # one application of op per term T_1 ... T_K
    assert torch.equal(v, v_copy)  # input not written
    assert out.data_ptr() != v.data_ptr()  # a new tensor, never v itself
    torch.testing.assert_close(out, expected, rtol=0.0, atol=3 * budget)
