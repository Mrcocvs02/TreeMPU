import pytest
import numpy as np
from TreeMPU import UniformMPU
import TreeMPU


@pytest.fixture(scope="module")
def identity_mpu():
    """Generates an exact identity MPU tensor with minimal canonical bond dimension (D=1)."""
    d, D = 2, 1
    A = np.einsum("ij,ab->ijab", np.eye(d), np.eye(D)) / np.sqrt(d)
    l_in = np.ones(D)
    r_in = np.ones(D)
    return d, D, A, l_in, r_in


@pytest.fixture(scope="module")
def semisimple_v_mpu():
    """
    Generates a semi-simple MPU (Example 14) with V-gate action on product states.
    Uses minimal D=5 OBC representation and L=2 blocking to satisfy Assumption 1.
    """
    # 1. Base parameters (qubit, original bond dim D=5) [cite: 420-434]
    d = 2
    D_base = 5
    theta = np.pi / 4
    V = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # 2. Bulk tensor definition (D=5) [cite: 421-424]
    A_base = {
        (0, 0): np.diag([1, 1, 0, 0, 0]),
        (1, 1): np.diag([1, 0, 1, 0, 0]),
        (0, 1): np.diag([0, 0, 0, 1, 0]),
        (1, 0): np.diag([0, 0, 0, 0, 1]),
    }

    # 3. Blocking L=2 for Assumption 1 (d^2L = 16 >= 5)
    L = 2
    d_block = d**L  # d=4 after blocking
    A_blocked = np.zeros((d_block, d_block, D_base, D_base), dtype=complex)

    for idx_out in range(d_block):
        for idx_in in range(d_block):
            # Map collective indices back to the original sites
            j = [(idx_out >> k) & 1 for k in range(L)][::-1]
            i = [(idx_in >> k) & 1 for k in range(L)][::-1]

            # Contraction: A[j1,i1] @ A[j0,i0]
            res = A_base.get((j[0], i[0]), np.zeros((D_base, D_base)))
            for s in range(1, L):
                res = res @ A_base.get((j[s], i[s]), np.zeros((D_base, D_base)))

            A_blocked[idx_out, idx_in] = res

    # 4. Minimal boundary vectors (rank 1)  [cite: 144, 2201-2205]
    # r_in closes the virtual paths, l_in encodes the operator b [cite: 431]
    l_in = np.array([1, V[0, 0] - 1, V[1, 1] - 1, V[0, 1], V[1, 0]], dtype=complex)
    r_in = np.array([1, 1, 1, 1, 1], dtype=complex)

    return d_block, D_base, A_blocked, l_in, r_in


@pytest.fixture(scope="module")
def local_unitary_mpu():
    """Generates an MPU applying a local Pauli-X gate globally (D=1)."""
    d, D = 2, 1
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    A = np.einsum("ij,ab->ijab", X, np.eye(D))
    l_in = np.ones(D)
    r_in = np.ones(D)
    return d, D, A, l_in, r_in


@pytest.fixture(scope="module")
def cz_interaction_mpu():
    """Generates the CZ-like interacting MPU with specific boundary vectors."""
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
    """Generates a random dense complex MPO tensor."""
    d, D = 3, 2
    np.random.seed(42)
    A = np.random.randn(d, d, D, D) + 1j * np.random.randn(d, d, D, D)
    l_in = np.random.randn(D) + 1j * np.random.randn(D)
    r_in = np.random.randn(D) + 1j * np.random.randn(D)
    return d, D, A, l_in, r_in


@pytest.fixture(scope="module")
def q_unif_result(request):
    d, D, A, l_in, r_in = request.getfixturevalue(request.param)
    np.random.seed(0)
    sigma, tau, q_val = TreeMPU.utils.optimize_q_unif(A, l_in, r_in)
    return d, D, A, l_in, r_in, sigma, tau, q_val


@pytest.fixture(scope="module")
def identity_lcu_data(identity_mpu):
    d, D, A, l_in, r_in = identity_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    coeffs, units, C = mpu._build_lcu_data()
    return D, coeffs, units, C, mpu


@pytest.fixture(scope="module")
def cz_lcu_data(cz_interaction_mpu):
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    coeffs, units, C = mpu._build_lcu_data()
    return D, coeffs, units, C, mpu


@pytest.fixture(scope="module")
def semisimple_lcu_data(semisimple_v_mpu):
    d, D, A, l_in, r_in = semisimple_v_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    coeffs, units, C = mpu._build_lcu_data()
    return D, coeffs, units, C, mpu
