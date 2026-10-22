"""Tests of ``UniformMPU``: construction, boundary operators, q and LCU data.

The tests verify constructor validation (unitarity and injectivity checks,
quimb encapsulation with the ``A[p_in, p_out, bond_L, bond_R]`` convention),
the boundary operators from tensor contractions, the conditioning number
``q_unif`` (closed form for the identity MPU), and the LCU decomposition of
the merging operator, which is compared with independently reconstructed
operators and the unitarity of the B and W_ctrl blocks.
"""

from unittest.mock import patch

import numpy as np
import pytest
import quimb.tensor as qtn

from TreeMPU.MPU import UniformMPU

ALL_LCU_DATA = ["identity_lcu_data", "cz_lcu_data", "semisimple_lcu_data"]


# --- UniformMPU Constructor and Validation Logic ---
@patch("TreeMPU.MPU.check_mpo_unitarity")
def test_uniformmpu_fails_unitarity(mock_unitarity, identity_mpu):
    """A failed unitarity check aborts construction."""
    mock_unitarity.side_effect = ValueError("Unitarity strictly lost")
    _, _, A_bulk, l_in, r_in = identity_mpu

    with pytest.raises(ValueError, match="Unitarity strictly lost"):
        UniformMPU(A=A_bulk, l_vec=l_in, r_vec=r_in, N=4)


@patch("TreeMPU.MPU.check_assumption_1")
@patch("TreeMPU.MPU.check_mpo_unitarity")
def test_uniformmpu_fails_injectivity(mock_unitarity, mock_injectivity, cz_interaction_mpu):
    """A failed Assumption 1 check aborts construction."""
    mock_unitarity.return_value = None
    mock_injectivity.side_effect = ValueError("Assumption 1 failed")

    _, _, A_bulk, l_in, r_in = cz_interaction_mpu

    with pytest.raises(ValueError, match="Assumption 1 failed"):
        UniformMPU(A=A_bulk, l_vec=l_in, r_vec=r_in, N=4)


def test_uniformmpu_quimb_conversion(identity_mpu):
    """The bulk tensor and boundaries become quimb tensors with the expected inds.

    The tags and index names of the wrapped tensors are checked as well.
    """
    _, bond_dim, A_bulk, l_in, r_in = identity_mpu
    num_sites = 4

    mpu = UniformMPU(A=A_bulk, l_vec=l_in, r_vec=r_in, N=num_sites)

    assert isinstance(mpu.A, qtn.Tensor)
    assert "A" in mpu.A.tags
    # Package convention A[p_in, p_out, bond_L, bond_R] (see utils.get_mpo_site_tensors).
    assert mpu.A.inds == ("p_in", "p_out", "bond_0", "bond_N")

    assert mpu._D == bond_dim
    assert mpu._N == num_sites

    assert "l" in mpu.l_vec.tags
    assert mpu.l_vec.inds == ("bond_0",)
    assert "r" in mpu.r_vec.tags
    assert mpu.r_vec.inds == (f"bond_{num_sites}",)


@pytest.mark.parametrize(
    "mpu_fixture_name",
    [
        "identity_mpu",
        "local_unitary_mpu",
        "cz_interaction_mpu",
        "semisimple_v_mpu",  # Critical case D=5
    ],
)
def test_uniformmpu_initialization_success(mpu_fixture_name, request):
    """Every valid MPU fixture passes the unitarity and injectivity checks, q >= 1."""
    d_phys, D_bond, A_bulk, l_in, r_in = request.getfixturevalue(mpu_fixture_name)
    num_sites = 4

    mpu = UniformMPU(A=A_bulk, l_vec=l_in, r_vec=r_in, N=num_sites)

    assert mpu._D == D_bond
    assert mpu._d == d_phys
    assert mpu.q_unif >= 1.0 - 1e-12


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_get_merging_operator_shape(lcu_fixture, request):
    """The merging operator has shape (D^2, D^2)."""
    D, _, _, _, mpu = request.getfixturevalue(lcu_fixture)
    M = mpu._get_merging_operator()
    assert M.shape == (D**2, D**2)


def test_uniformmpu_non_unitary_bulk_tensor_is_rejected():
    """A non-unitary bulk tensor is rejected."""
    D = 2  # Bond dimension
    d = 2  # Physical dimension

    # Shape follows the package convention A[p_in, p_out, bond_L, bond_R].
    A = np.zeros((d, d, D, D), dtype=complex)

    # Identity-like structure ...
    A[0, 0, 0, 0] = 1.0
    A[0, 0, 1, 1] = 1.0
    A[1, 1, 0, 0] = 1.0
    A[1, 1, 1, 1] = 1.0

    # ... with one entry perturbed so that the isometry condition fails.
    A[0, 0, 0, 0] += 1.0

    # The boundary vectors are valid; only the bulk tensor violates unitarity.
    l_in = np.array([1.0, 0.0])
    r_in = np.array([1.0, 0.0])

    with pytest.raises(ValueError, match="Unitarity strictly lost"):
        UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)


# --- UniformMPU Internal Contractions & Boundary Operators ---
@pytest.mark.parametrize(
    "mpu_fixture_name", ["identity_mpu", "cz_interaction_mpu", "semisimple_v_mpu"]
)
def test_compute_boundary_operators_real_contraction(mpu_fixture_name, request):
    """The boundary operators from the contraction are non-zero (D, D) matrices."""
    _, bond_dim, bulk_data, l_in, r_in = request.getfixturevalue(mpu_fixture_name)
    num_sites = 4

    mpu = UniformMPU(bulk_data, l_in, r_in, N=num_sites)
    L2_mat, R2_mat = mpu._compute_boundary_operators()

    # The boundary operators act on the bond space.
    assert L2_mat.shape == (bond_dim, bond_dim)
    assert R2_mat.shape == (bond_dim, bond_dim)

    # A vanishing operator would make the merging kernel undefined.
    assert np.linalg.norm(L2_mat) > 0
    assert np.linalg.norm(R2_mat) > 0


# --- UniformMPU q_unif Calculation & Numerical Stability ---


def test_q_unif_dimension_mismatch(identity_mpu):
    """Boundary operators of incompatible dimensions raise ValueError in q."""
    d_phys, _, bulk_tensor, l_in, r_in = identity_mpu

    mpu = UniformMPU(bulk_tensor, l_in, r_in, N=2)

    # Bypass the constructor, which only produces consistent operators.
    mpu.L_inv = np.eye(2)
    mpu.R_inv = np.eye(3)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        mpu._compute_q_unif()


def test_q_unif_identity_case(identity_mpu):
    """q_unif equals 1 for the identity MPU with D = 1."""
    d_phys, bond_dim, bulk_tensor, l_in, r_in = identity_mpu

    # Patch the checks and the contraction so that only _compute_q_unif is tested.
    with (
        patch("TreeMPU.MPU.check_mpo_unitarity", return_value=True),
        patch("TreeMPU.MPU.check_assumption_1", return_value=(None, None)),
        patch(
            "TreeMPU.MPU.UniformMPU._compute_boundary_operators",
            return_value=(np.eye(bond_dim) / bond_dim, np.eye(bond_dim) / bond_dim),
        ),
    ):
        mpu = UniformMPU(bulk_tensor, l_in, r_in, N=4)

    # Canonical boundary operators of the D = 1 identity MPU.
    mpu.L = np.array([[1.0]])
    mpu.R = np.array([[1.0]])

    expected_q = 1.0
    assert mpu._compute_q_unif() == pytest.approx(expected_q)


# --- LCU Decomposition ---


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_C_positive(lcu_fixture, request):
    """The LCU 1-norm must be strictly positive."""
    D, coeffs, units, C, mpu = request.getfixturevalue(lcu_fixture)
    assert C > 0


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_unitaries_are_unitary(lcu_fixture, request):
    """Each unitary in the LCU list must be unitary."""
    D, coeffs, units, C, mpu = request.getfixturevalue(lcu_fixture)

    for i, W in enumerate(units):
        err = np.linalg.norm(W @ W.conj().T - np.eye(W.shape[0]))
        assert err < 1e-10, f"Non-unitary W[{i}] detected: {err:.2e}"


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_terms_are_unique(lcu_fixture, request):
    """Ensure no duplicate unitaries in LCU list."""
    D, coeffs, units, C, mpu = request.getfixturevalue(lcu_fixture)

    seen = set()
    for i, W in enumerate(units):
        key = tuple(W.ravel().round(10))  # Rounding absorbs float noise in the key
        assert key not in seen, f"Duplicate unitary found at index {i}"
        seen.add(key)


@pytest.mark.parametrize("lcu_fixture", ALL_LCU_DATA)
def test_lcu_reconstruction_from_terms(lcu_fixture, request):
    """Reconstruct merging operator from LCU terms and verify correctness."""
    D, _, _, _, mpu = request.getfixturevalue(lcu_fixture)
    coeffs, units, C = mpu._build_lcu_data()

    M_recon = np.zeros((D**2, D**2), dtype=complex)
    for c, W in zip(coeffs, units):
        M_recon += c * W

    M_ref = mpu._get_merging_operator().reshape(D**2, D**2)
    assert np.allclose(M_recon, M_ref, atol=1e-8), (
        f"Reconstruction error: {np.max(np.abs(M_recon - M_ref)):.2e}"
    )


def test_uniformmpu_lcu_caching(cz_interaction_mpu):
    """Multiple calls to _build_lcu_data return the same cached result."""
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    res1 = mpu._build_lcu_data()
    res2 = mpu._build_lcu_data()

    assert res1 is res2

    for c1, c2 in zip(res1[0], res2[0]):
        assert c1 == pytest.approx(c2, abs=1e-12)
    for W1, W2 in zip(res1[1], res2[1]):
        assert np.allclose(W1, W2, atol=1e-12)


# --- Unitary Merging Components ---
def test_build_merging_unitary_B_prepares_correct_state(identity_lcu_data):
    """B|0> equals (1/sqrt(C)) sum_i sqrt(c_i) |i>."""
    D, coeffs, units, C, mpu = identity_lcu_data
    mpu._build_merging_unitary()

    B, _ = mpu._merging_unitary_cache
    K = len(coeffs)
    dim_ancilla = B.shape[0]

    state_0 = np.zeros(dim_ancilla, dtype=complex)
    state_0[0] = 1.0
    prepared = B @ state_0

    expected = np.zeros(dim_ancilla, dtype=complex)
    expected[:K] = np.sqrt(coeffs) / np.sqrt(C)

    assert np.allclose(prepared, expected, atol=1e-10)


def test_build_merging_unitary_B_is_unitary(cz_lcu_data):
    """B must be a unitary matrix."""
    D, coeffs, units, C, mpu = cz_lcu_data
    mpu._build_merging_unitary()

    B, _ = mpu._merging_unitary_cache
    err = np.linalg.norm(B @ B.conj().T - np.eye(B.shape[0]))
    assert err < 1e-10


def test_build_merging_unitary_W_ctrl_is_unitary(cz_lcu_data):
    """W_ctrl must be a unitary matrix."""
    D, coeffs, units, C, mpu = cz_lcu_data
    mpu._build_merging_unitary()

    _, W_ctrl = mpu._merging_unitary_cache
    err = np.linalg.norm(W_ctrl @ W_ctrl.conj().T - np.eye(W_ctrl.shape[0]))
    assert err < 1e-10


def test_build_merging_unitary_B_first_col_is_unit_vector(cz_lcu_data):
    """The first column of B has unit norm."""
    D, coeffs, units, C, mpu = cz_lcu_data
    mpu._build_merging_unitary()

    B, _ = mpu._merging_unitary_cache
    first_col = B[:, 0]
    norm = np.linalg.norm(first_col)

    assert norm == pytest.approx(1.0, abs=1e-10), (
        f"First column of B has norm {norm:.6f}, expected 1.0. "
        f"Check normalization: use sqrt(c_i)/sqrt(C), not sqrt(c_i)/C."
    )
