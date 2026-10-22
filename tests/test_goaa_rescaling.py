"""The GOAA rescaling contract.

After ``_build_rotated_lcu`` the LCU 1-norm C satisfies
``(2*ell + 1) * arcsin(1/C) == pi/2``, which is the condition for oblivious
amplitude amplification to succeed deterministically (``p_anc0 == 1``) and the
synthesized circuit to be exact. The contract is verified directly on
synthetic one-term operators over a range of initial norms ``C_0``, without a
full circuit synthesis, and the lower bound on q that restricts which norm
windows occur is checked by random sampling of canonical boundaries.
"""

import numpy as np
import pytest

from TreeMPU.MPU import CircuitDecomposition as CD


def _achieved_angle(C_0: float) -> tuple[float, int, float]:
    """Rescale a synthetic operator of 1-norm ``C_0``; return the angle, ell and C."""
    ell, phis = CD._compute_rotation_params(C_0)
    # One LCU term of weight C_0 reproduces the 1-norm the rescaling sees;
    # _build_rotated_lcu only ever reads the coefficient magnitudes.
    coef, unit, C = CD._build_rotated_lcu([C_0], [np.eye(2, dtype=complex)], C_0, phis)
    return (2 * ell + 1) * np.arcsin(1.0 / C), ell, C


@pytest.mark.parametrize("ell_target", [1, 2, 3, 4, 5])
def test_rescaling_is_exact_when_c0_is_already_representable(ell_target):
    """A norm on a GOAA staircase target needs no rescaling and gives angle pi/2."""
    C_0 = 1.0 / np.sin(np.pi / (4 * ell_target + 2))
    angle, ell, _ = _achieved_angle(C_0)
    assert ell == ell_target
    assert angle == pytest.approx(np.pi / 2, abs=1e-9)


@pytest.mark.parametrize("C_0", [2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0])
def test_rescaling_is_exact_where_sqrt2_is_not_binding(C_0):
    """The rescaling is exact for C_0 above about 2.29, where sqrt(2) is not binding."""
    angle, _, _ = _achieved_angle(C_0)
    assert angle == pytest.approx(np.pi / 2, abs=1e-9)


@pytest.mark.parametrize("C_0", [1.0001, 1.05, 1.2, 1.41, 2.0001, 2.05, 2.1, 2.2, 2.28])
def test_rescaling_is_exact_where_sqrt2_is_binding(C_0):
    """The rescaling is exact where the required factor exceeds sqrt(2)."""
    angle, _, _ = _achieved_angle(C_0)
    assert angle == pytest.approx(np.pi / 2, abs=1e-9)


@pytest.mark.parametrize("C_0", [1.05, 1.2, 2.05, 2.2])
def test_extra_rotation_qubits_allocated_when_factor_exceeds_sqrt2(C_0):
    """Where one rotation qubit cannot reach the target, more than one is allocated."""
    # In these bands the required factor C'(ell)/C_0 exceeds sqrt(2) (consecutive
    # targets differ by 2.000 for ell = 0 -> 1 and 1.618 for 1 -> 2), so a single
    # rotation qubit cannot reach the target; ceil(2 log2(factor)) qubits can.
    ell, phis = CD._compute_rotation_params(C_0)
    C_prime = 1.0 / np.sin(np.pi / (4 * ell + 2))
    assert C_prime > C_0 * np.sqrt(2)  # one qubit cannot get there
    assert len(phis) >= 2  # so more than one is allocated


def test_rotation_qubit_count_stays_at_one_when_sqrt2_suffices():
    """A single rotation qubit is used whenever a factor of at most sqrt(2) suffices."""
    for C_0 in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 12.0]:
        _, phis = CD._compute_rotation_params(C_0)
        assert len(phis) == 1


def test_q_lower_bound_makes_the_first_band_unoccupied():
    """Canonical boundaries satisfy q >= D, so (1, sqrt(2)] is empty for D > 1."""
    # Canonical form gives L_k = 1, R_k = diag(s) with sum_i s_i^2 = 1, so
    # q_k = sqrt(sum_i 1/s_i^2) >= D_k by Cauchy-Schwarz.
    rng = np.random.default_rng(0)
    for D in (1, 2, 3, 4, 5):
        for _ in range(2000):
            s2 = rng.dirichlet(np.ones(D) * rng.uniform(0.5, 50))
            assert np.sqrt(np.sum(1.0 / s2)) >= D - 1e-9
