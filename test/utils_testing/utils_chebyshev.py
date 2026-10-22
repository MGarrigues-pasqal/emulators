"""
Independent oracle and error budgets for the Chebyshev expansion of exp(-i alpha x).

exp(-i alpha x) = sum_k c_k T_k(x) on [-1, 1], with c_0 = J_0(alpha) and
c_k = 2 (-i)^k J_k(alpha) (Jacobi-Anger). The oracle takes the c_k from the FFT of
exp(-i alpha cos(theta)), independently of the Bessel recurrence of the code under test.
"""

import math

import torch

EPS = torch.finfo(torch.float64).eps
# exp(-i dt H) is split into sub-steps of alpha <= MAX_ALPHA (see emu_sv.time_evolution.torch)
MAX_ALPHA = 2000.0


def _significant_length(alpha: float) -> int:
    """Number of c_k above 1e-30: J_k(alpha) decays super-exponentially past the
    turning point k = alpha (Airy regime of width alpha**(1/3))."""
    a = abs(alpha)
    return int(a + 12 * a ** (1 / 3) + 60) + 1


def chebyshev_coefficients_oracle(alpha: float) -> torch.Tensor:
    """
    c_0, c_1, ... of exp(-i alpha x), complex128, from the FFT of
    f(theta) = exp(-i alpha cos theta) = sum_k (-i)^k J_k(alpha) e^{i k theta}.
    The FFT on M points returns sum_{m = k mod M} (-i)^m J_m(alpha); with
    M >= 4 * _significant_length(alpha), the aliased terms are below 1e-30.
    """
    length = _significant_length(alpha)
    size = 2 ** math.ceil(math.log2(4 * length))
    theta = 2 * math.pi * torch.arange(size, dtype=torch.float64) / size
    a = torch.fft.fft(torch.exp(-1j * alpha * torch.cos(theta))) / size
    coefs = a[:length].clone()
    coefs[1:] *= 2
    return coefs


def oracle_coefficient_error(alpha: float) -> float:
    """
    Bound on |c_k(oracle) - c_k| for each k:
    - each sample: |alpha| * |fl(cos theta) - cos theta| <= 16 |alpha| eps
      (theta rounded twice, cos once), exp: 2 eps;
    - FFT (Higham, Accuracy and Stability, § 24.1): <= 5 log2(M) eps in 2-norm,
      relative to ||f|| / sqrt(M) = 1 (|f| = 1, Parseval);
    - c_k = 2 a_k doubles it.
    """
    size = 2 ** math.ceil(math.log2(4 * _significant_length(alpha)))
    return 2 * (16 * abs(alpha) + 2 + 5 * math.log2(size)) * EPS


def tails(coefs: torch.Tensor) -> torch.Tensor:
    """tails[k] = sum_{j > k} |c_j|"""
    a = coefs.abs()
    suffix = torch.flip(torch.cumsum(torch.flip(a, [0]), 0), [0])
    return torch.cat((suffix[1:], torch.zeros(1, dtype=a.dtype)))


def truncation_index(coefs: torch.Tensor, tolerance: float) -> int:
    """Smallest K with sum_{k > K} |c_k| <= tolerance."""
    return int(torch.nonzero(tails(coefs) <= tolerance)[0])


def chebyshev_rounding_bound(coefs: torch.Tensor, local_error: float) -> float:
    """
    First-order bound on the rounding error of phase * sum_k c_k T_k(Ht) v, computed with
    the recurrence T_k = 2 Ht T_{k-1} - T_{k-2}, for ||v|| = 1 and the spectrum of Ht
    in [-1, 1]:
    - a local error delta_j at term j propagates as e_k = sum_{j<=k} U_{k-j}(Ht) delta_j,
      with ||U_m(Ht)|| <= m + 1 (Chebyshev polynomials of the second kind), so the
      error of the sum is <= sum_j ||delta_j|| sum_{k>=j} |c_k| (k - j + 1);
    - accumulation into the sum: (K + 1) eps sum_k |c_k|.
    `local_error` bounds ||delta_j|| (one application of the operator, one subtraction).
    This is a worst-case bound, growing as alpha**2.5 eps.
    """
    a = coefs.abs()
    k = torch.arange(len(a), dtype=torch.float64)
    s0 = torch.flip(torch.cumsum(torch.flip(a, [0]), 0), [0])
    s1 = torch.flip(torch.cumsum(torch.flip(a * (k + 1), [0]), 0), [0])
    beta = s1 - k * s0  # sum_{k' >= j} |c_k'| (k' - j + 1)
    return float(local_error * beta.sum() + len(a) * EPS * a.sum())


def matrix_exp_error(h: torch.Tensor, dt: float) -> float:
    """
    Estimate (not a proven bound) of ||matrix_exp(-i dt h) v - exp(-i dt h) v|| for a
    unit vector v and a Hermitian h: scaling and squaring has a backward error of order
    eps (Higham 2005), amplified by the norm of the argument and summed over dim terms.
    """
    norm = torch.linalg.eigvalsh(h).abs().max().item() * abs(dt)
    return h.shape[0] * EPS * max(1.0, norm)


def chebyshev_step_budget(
    h: torch.Tensor, omegas: torch.Tensor, dt: float, tolerance: float
) -> float:
    """
    Budget on ||chebyshev step - matrix_exp(-i dt h) v|| for a unit vector v, where h is
    the dense Rydberg Hamiltonian built from `omegas` (raw Rabi frequencies) and
    `tolerance` is the truncation tolerance of the whole step.
    Spectral bounds as in the code under test, from independent inputs: the dense
    diagonal and R = sum_j |Omega_j| / 2 (Gershgorin). Sub-steps of alpha <= MAX_ALPHA,
    each with tolerance / m. The operator 2 (H - c) / r applied by the torch path sums
    N + 1 terms per row with absolute row sums <= 2: local error <= (2 (N + 1) + 3) eps.
    """
    nqubits = len(omegas)
    d = h.diagonal().real
    radius = omegas.abs().sum().item() / 2
    r = (d.max() - d.min()).item() / 2 + radius
    alpha = r * dt
    m = max(1, math.ceil(abs(alpha) / MAX_ALPHA))
    coefs = chebyshev_coefficients_oracle(alpha / m)
    K = truncation_index(coefs, tolerance / m)
    local = (2 * (nqubits + 1) + 3) * EPS
    per_substep = tolerance / m + chebyshev_rounding_bound(coefs[: K + 1], local)
    return m * per_substep + matrix_exp_error(h, dt)
