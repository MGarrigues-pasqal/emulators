import cmath
import math
from decimal import Decimal
from unittest.mock import Mock

import pytest
import torch

from test.utils_testing import (
    chebyshev_coefficients_exact,
    chebyshev_rounding_bound,
    matrix_exp_error,
)

dtype = torch.complex128
EPS = torch.finfo(torch.float64).eps
# Largest |J_k| error of the Miller recurrence, measured against mpmath (30 digits) for
# 100 <= alpha <= 30000, and against chebyshev_coefficients_exact for the alpha below.
MILLER_ERROR = 4e-16
# The code sums the tail in double precision from Bessel values with a small relative
# error; cases whose exact tail is this close (relative) to the tolerance are refused.
TAIL_MARGIN = Decimal("1e-6")


def checked_truncation_index(alpha: float, tolerance: float) -> int:
    """
    Smallest K with sum_{k>K} |c_k| <= tolerance, from the exact tails, refusing (as a
    test error) cases where rounding in the code under test could legitimately move K.
    """
    _, tails = chebyshev_coefficients_exact(alpha)
    tol = Decimal(tolerance)
    K = next(k for k, tail in enumerate(tails) if tail <= tol)
    assert tails[K] < tol * (1 - TAIL_MARGIN), "test case too close to the threshold"
    assert K == 0 or tails[K - 1] > tol * (1 + TAIL_MARGIN), "test case too close"
    return K


@pytest.mark.parametrize(
    "alpha", [0.0, 1e-10, 3e-6, 0.5, 1.0, 7.3, 50.0, 300.0, -7.3, -300.0]
)
@pytest.mark.parametrize("tolerance", [1e-4, 1e-8, 1e-12, 1e-14])
def test_exp_chebyshev_coefficients_match_oracle(alpha: float, tolerance: float) -> None:
    """
    Values, signs of (-i)^k, factor 2 on k >= 1, conjugation for alpha < 0, minimal K,
    including the series tolerances used by the steps (krylov_tolerance / 100).
    """
    expected, _ = chebyshev_coefficients_exact(alpha)
    K = checked_truncation_index(alpha, tolerance)

    from emu_base.math.chebyshev import exp_chebyshev_coefficients

    coefs = exp_chebyshev_coefficients(alpha, tolerance)
    assert all(isinstance(c, complex) for c in coefs)
    assert len(coefs) == K + 1  # smallest K with sum_{k>K} |c_k| <= tolerance
    # Budget per coefficient: Miller error on J_k, doubled for c_k, plus the rounding of
    # the exact oracle to double (eps |c_k| / 2 <= eps). Both tolerances explicit;
    # atol = 2x the budget. A sign or factor-2 bug changes some c_k by ~0.1.
    atol = 2 * (2 * MILLER_ERROR + EPS)
    torch.testing.assert_close(
        torch.tensor(coefs, dtype=dtype), expected[: K + 1], rtol=0.0, atol=atol
    )


@pytest.mark.parametrize("alpha", [0.5, 7.3, 50.0, -50.0])
@pytest.mark.parametrize("tolerance", [1e-4, 1e-10])
def test_exp_chebyshev_coefficients_truncation_bound(
    alpha: float, tolerance: float
) -> None:
    """sum_k c_k T_k(x) is within tail(K) <= tolerance of exp(-i alpha x) on [-1, 1]."""
    _, tails = chebyshev_coefficients_exact(alpha)
    K = checked_truncation_index(alpha, tolerance)

    from emu_base.math.chebyshev import exp_chebyshev_coefficients

    coefs = torch.tensor(exp_chebyshev_coefficients(alpha, tolerance), dtype=dtype)
    x = torch.linspace(-1.0, 1.0, 2001, dtype=torch.float64)
    k = torch.arange(len(coefs), dtype=torch.float64)
    series = (coefs * torch.cos(k * torch.arccos(x)[:, None])).sum(dim=1)
    expected = torch.exp(-1j * alpha * x)
    # Truncation: <= tail(K) (exact tails, |T_k| <= 1), not scaled. Rounding: coefficient
    # errors (len (2 MILLER_ERROR + eps)); T_k(x) = cos(k arccos x) with |arccos error|
    # <= pi eps, so (2 pi k + 2) eps |c_k| per term; summation (K + 1) eps sum |c_k|;
    # exp(-i alpha x): (|alpha| + 2) eps. atol = tail(K) + 2x the rounding.
    a = coefs.abs()
    rounding = (
        float((a * (2 * math.pi * k + 2)).sum() + len(a) * a.sum()) * EPS
        + len(a) * (2 * MILLER_ERROR + EPS)
        + (abs(alpha) + 2) * EPS
    )
    atol = float(tails[K]) + 2 * rounding
    torch.testing.assert_close(series, expected, rtol=0.0, atol=atol)


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
    phase * sum_k c_k T_k(op / 2) v, for op = 2 (h - c) / r, against the same truncated
    series built from the exact coefficients in the eigenbasis of op / 2 (any missing,
    extra or wrong term shows at the scale |c_K| >= ~tolerance), and against matrix_exp.
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
    phase = cmath.exp(-1j * dt * c)
    m = 2 * (h - c * torch.eye(dim, dtype=dtype)) / r
    op = Mock(side_effect=lambda y: m @ y)

    v = torch.randn(dim, dtype=dtype)
    v *= 3.7 / v.norm()  # not normalised
    v_copy = v.clone()
    K = checked_truncation_index(alpha, tolerance)
    exact, tails = chebyshev_coefficients_exact(alpha)
    coefs = exact[: K + 1]
    lam, vecs = torch.linalg.eigh(m / 2)
    theta = torch.arccos(lam.clamp(-1.0, 1.0))
    k = torch.arange(K + 1, dtype=torch.float64)
    p = (coefs * torch.cos(k * theta[:, None])).sum(dim=1)
    series = phase * (vecs @ (p * (vecs.mH @ v)))
    expected = torch.linalg.matrix_exp(-1j * dt * h) @ v

    # Budget (2-norm, per unit of ||v||), rounding only:
    # - recurrence: local error per term <= eps (dim * max row sum of |m| + 3) (dense
    #   product of dim terms, one subtraction, one scaling), chebyshev_rounding_bound;
    # - coefficients: (K + 1) (2 MILLER_ERROR + eps), as in the coefficient test;
    # - reference series (estimate): eigh backward error dim eps ||m / 2|| moves p(m / 2)
    #   by <= max|p'| ~ |alpha| + 1 times it; cos(k arccos): (2 pi k + 2) eps |c_k|;
    #   products with the eigenvectors: 3 dim eps.
    # atol = 3x the budget; truncation (tail(K) <= tolerance) only against matrix_exp.
    local = EPS * (dim * m.abs().sum(dim=1).max().item() + 3)
    a = coefs.abs()
    rounding = (
        chebyshev_rounding_bound(coefs, local)
        + (K + 1) * (2 * MILLER_ERROR + EPS)
        + (abs(alpha) + 1) * dim * EPS * lam.abs().max().item()
        + float((a * (2 * math.pi * k + 2)).sum()) * EPS
        + 3 * dim * EPS
    )
    norm = v.norm().item()

    from emu_base.math.chebyshev import chebyshev_exp

    out = chebyshev_exp(op, v, alpha, phase, tolerance)

    assert op.call_count == K  # one application of op per term T_1 ... T_K
    assert torch.equal(v, v_copy)  # input not written
    assert out.data_ptr() != v.data_ptr()  # a new tensor, never v itself
    torch.testing.assert_close(
        torch.linalg.vector_norm(out - series),
        torch.zeros((), dtype=torch.float64),
        rtol=0.0,
        atol=3 * norm * rounding,
    )
    end_to_end = float(tails[K]) + 3 * (rounding + matrix_exp_error(h, dt))
    torch.testing.assert_close(
        torch.linalg.vector_norm(out - expected),
        torch.zeros((), dtype=torch.float64),
        rtol=0.0,
        atol=norm * end_to_end,
    )
