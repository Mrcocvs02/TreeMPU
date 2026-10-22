"""Shared MPU fixtures: explicit unitary MPUs and cached LCU data."""

import numpy as np
import pytest

import TreeMPU
from TreeMPU import UniformMPU


@pytest.fixture(scope="module")
def identity_mpu():
    """Return the identity MPU, U_N = I, with bond dimension D = 1."""
    d, D = 2, 1
    A = np.einsum("ij,ab->ijab", np.eye(d), np.eye(D))
    l_in = np.ones(D)
    r_in = np.ones(D)
    return d, D, A, l_in, r_in


@pytest.fixture(scope="module")
def semisimple_v_mpu():
    """Return a semi-simple MPU with a V-gate action on product states.

    The tuple ``(d, D, A, l_in, r_in)`` holds the blocked bulk tensor and the
    boundary vectors. The minimal D = 5 open-boundary representation is
    blocked with L = 2 sites so that Assumption 1 holds (``d**(2L) = 16 >= 5``).
    """
    d = 2
    D_base = 5
    theta = np.pi / 4
    V = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    A_base = {
        (0, 0): np.diag([1, 1, 0, 0, 0]),
        (1, 1): np.diag([1, 0, 1, 0, 0]),
        (0, 1): np.diag([0, 0, 0, 1, 0]),
        (1, 0): np.diag([0, 0, 0, 0, 1]),
    }

    L = 2
    d_block = d**L  # d=4 after blocking
    A_blocked = np.zeros((d_block, d_block, D_base, D_base), dtype=complex)

    for idx_out in range(d_block):
        for idx_in in range(d_block):
            # Collective index to bits of the original sites, most significant first.
            j = [(idx_out >> k) & 1 for k in range(L)][::-1]
            i = [(idx_in >> k) & 1 for k in range(L)][::-1]

            res = A_base.get((j[0], i[0]), np.zeros((D_base, D_base)))
            for s in range(1, L):
                res = res @ A_base.get((j[s], i[s]), np.zeros((D_base, D_base)))

            A_blocked[idx_out, idx_in] = res

    # Rank-1 boundaries: r_in closes the virtual paths, l_in encodes the operator b.
    l_in = np.array([1, V[0, 0] - 1, V[1, 1] - 1, V[0, 1], V[1, 0]], dtype=complex)
    r_in = np.array([1, 1, 1, 1, 1], dtype=complex)

    return d_block, D_base, A_blocked, l_in, r_in


@pytest.fixture(scope="module")
def local_unitary_mpu():
    """Return an MPU applying a local Pauli-X gate on every site (D = 1)."""
    d, D = 2, 1
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    A = np.einsum("ij,ab->ijab", X, np.eye(D))
    l_in = np.ones(D)
    r_in = np.ones(D)
    return d, D, A, l_in, r_in


@pytest.fixture(scope="module")
def cz_interaction_mpu():
    """Return the CZ-like interacting MPU with its asymmetric boundary vectors."""
    d, D = 2, 2

    A_00 = np.array([[1, 0], [0, 0]], dtype=float)
    A_01 = np.array([[0, 0], [0, 0]], dtype=float)
    A_10 = np.array([[0, 0], [0, 0]], dtype=float)
    A_11 = np.array([[1, 0], [0, 1]], dtype=float)

    # Shape becomes (2, 2, 2, 2)
    A = np.array([[A_00, A_01], [A_10, A_11]], dtype=float)

    l_in = np.array([1.0, 1.0], dtype=float)
    r_in = np.array([1.0, -2.0], dtype=float)

    return d, D, A, l_in, r_in


@pytest.fixture(scope="module")
def random_complex_mpo():
    """Return a random dense complex MPO tensor with random boundary vectors."""
    d, D = 3, 2
    np.random.seed(42)
    A = np.random.randn(d, d, D, D) + 1j * np.random.randn(d, d, D, D)
    l_in = np.random.randn(D) + 1j * np.random.randn(D)
    r_in = np.random.randn(D) + 1j * np.random.randn(D)
    return d, D, A, l_in, r_in


@pytest.fixture(scope="module")
def q_unif_result(request):
    """Return an MPU fixture (named by ``request.param``) with its optimized q.

    The tuple is ``(d, D, A, l_in, r_in, sigma, tau, q_val)`` from
    ``optimize_q_unif`` with the global NumPy seed fixed to 0.
    """
    d, D, A, l_in, r_in = request.getfixturevalue(request.param)
    np.random.seed(0)
    sigma, tau, q_val = TreeMPU.utils.optimize_q_unif(A, l_in, r_in)
    return d, D, A, l_in, r_in, sigma, tau, q_val


@pytest.fixture(scope="module")
def identity_lcu_data(identity_mpu):
    """Return the two-term LCU data of the merging operator of the identity MPU.

    The tuple is ``(D, coeffs, units, C, mpu)`` for a ``UniformMPU`` with N = 4.
    """
    d, D, A, l_in, r_in = identity_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    coeffs, units, C = mpu._build_lcu_data()
    return D, coeffs, units, C, mpu


@pytest.fixture(scope="module")
def cz_lcu_data(cz_interaction_mpu):
    """Return the two-term LCU data of the merging operator of the CZ-like MPU.

    The tuple is ``(D, coeffs, units, C, mpu)`` for a ``UniformMPU`` with N = 4.
    """
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    coeffs, units, C = mpu._build_lcu_data()
    return D, coeffs, units, C, mpu


@pytest.fixture(scope="module")
def semisimple_lcu_data(semisimple_v_mpu):
    """Return the two-term LCU data of the semi-simple V-gate MPU.

    The tuple is ``(D, coeffs, units, C, mpu)`` for a ``UniformMPU`` with N = 4.
    """
    d, D, A, l_in, r_in = semisimple_v_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    coeffs, units, C = mpu._build_lcu_data()
    return D, coeffs, units, C, mpu
