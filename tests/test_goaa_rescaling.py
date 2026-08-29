"""
The GOAA rescaling contract: after ``_build_rotated_lcu``, the LCU 1-norm C
must satisfy ``(2*ell + 1) * arcsin(1/C) == pi/2`` exactly, since that is the
condition under which oblivious amplitude amplification succeeds
deterministically (``p_anc0 == 1``) and the synthesized circuit is exact.

These tests exercise that contract directly, rather than through a full
circuit synthesis, so a violation is attributable to the rescaling itself.
"""

import numpy as np
import pytest

from TreeMPU.MPU import CircuitDecomposition as CD


def _achieved_angle(C_0: float) -> tuple[float, int, float]:
    """Run the real rescaling path on a synthetic operator with 1-norm C_0."""
    ell, phis = CD._compute_rotation_params(C_0)
    # One LCU term of weight C_0 reproduces the 1-norm the rescaling sees;
    # _build_rotated_lcu only ever reads the coefficient magnitudes.
    coef, unit, C = CD._build_rotated_lcu([C_0], [np.eye(2, dtype=complex)], C_0, phis)
    return (2 * ell + 1) * np.arcsin(1.0 / C), ell, C


@pytest.mark.parametrize("ell_target", [1, 2, 3, 4, 5])
def test_rescaling_is_exact_when_c0_is_already_representable(ell_target):
    """
    C_0 = 1/sin(pi/(4l+2)) needs no rescaling at all (phi == 0). This is the
    case every pre-existing fixture happens to land on -- cz_interaction_mpu
    has C_0 == 2.0 exactly, which is the ell=1 point.
    """
    C_0 = 1.0 / np.sin(np.pi / (4 * ell_target + 2))
    angle, ell, _ = _achieved_angle(C_0)
    assert ell == ell_target
    assert angle == pytest.approx(np.pi / 2, abs=1e-9)


@pytest.mark.parametrize("C_0", [2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0])
def test_rescaling_is_exact_outside_the_saturating_bands(C_0):
    """Above C_0 ~ 2.29 the sqrt(2) amplification ceiling is never binding."""
    angle, _, _ = _achieved_angle(C_0)
    assert angle == pytest.approx(np.pi / 2, abs=1e-9)


@pytest.mark.parametrize("C_0", [1.0001, 1.05, 1.2, 1.41, 2.0001, 2.05, 2.1, 2.2, 2.28])
def test_rescaling_is_exact_inside_the_formerly_saturating_bands(C_0):
    """
    These are the windows a single rotation qubit could not reach: the
    required factor C'(ell)/C_0 there exceeds sqrt(2) (the ratio between
    consecutive targets is 2.000 for ell=0->1 and 1.618 for 1->2, both above
    sqrt(2)), so the old code clipped the arcsin argument and left GOAA
    inexact. Allocating ceil(2*log2(factor)) rotation qubits reaches them
    exactly.
    """
    angle, _, _ = _achieved_angle(C_0)
    assert angle == pytest.approx(np.pi / 2, abs=1e-9)


@pytest.mark.parametrize("C_0", [1.05, 1.2, 2.05, 2.2])
def test_widened_ansatz_is_what_makes_those_bands_reachable(C_0):
    """
    Pin the mechanism: in these bands one qubit provably cannot reach the
    target, and the code must respond by widening the ansatz rather than by
    saturating phi.
    """
    ell, phis = CD._compute_rotation_params(C_0)
    C_prime = 1.0 / np.sin(np.pi / (4 * ell + 2))
    assert C_prime > C_0 * np.sqrt(2)  # one qubit cannot get there
    assert len(phis) >= 2  # so more than one is allocated


def test_rotation_qubit_count_stays_at_one_when_sqrt2_suffices():
    """The common case must not pay for extra ancillas it does not need."""
    for C_0 in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 12.0]:
        _, phis = CD._compute_rotation_params(C_0)
        assert len(phis) == 1


def test_q_lower_bound_makes_the_first_band_unoccupied():
    """
    Canonical form gives L_k = 1, R_k = diag(s) with sum_i s_i^2 = 1, so
    q_k = sqrt(sum_i 1/s_i^2) >= D_k by Cauchy-Schwarz. The window
    (1, sqrt(2)] therefore needs D_k = 1, where q is exactly 1 and no
    rescaling is required -- only the (2, 2.288] window is ever occupied,
    and it needs bond dimension exactly 2.
    """
    rng = np.random.default_rng(0)
    for D in (1, 2, 3, 4, 5):
        for _ in range(2000):
            s2 = rng.dirichlet(np.ones(D) * rng.uniform(0.5, 50))
            assert np.sqrt(np.sum(1.0 / s2)) >= D - 1e-9
