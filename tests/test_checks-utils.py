import pytest  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from TreeMPU.checks import (
    check_mpo_unitarity,
    check_assumption_1,
    verify_lcu,
    verify_merging_unitary,
)
from TreeMPU.utils import optimize_q_unif, get_merging_operator
from TreeMPU.MPU import UniformMPU


# ============================================================================
# SECTION 1: MPO UNITARITY CHECKS
# ============================================================================


@pytest.mark.parametrize("N", [1, 5, 10])
@pytest.mark.parametrize(
    "mpu_fixture",
    ["identity_mpu", "local_unitary_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
)
def test_check_mpo_unitarity_valid(mpu_fixture, N, request):
    """
    Evaluates unitarity on physically sound MPU tensors.
    Locked to N_max=1 until the trace metric in check_mpo_unitarity is fixed.
    """
    _, _, A, l_in, r_in = request.getfixturevalue(mpu_fixture)

    check_mpo_unitarity(N_max=N, A_np=A, l_in=l_in, r_in=r_in, tol=1e-6)


@pytest.mark.parametrize("N", [1, 5, 10])
def test_check_mpo_unitarity_failure(random_complex_mpo, N):
    """
    A strictly random MPO violates unitarity and must raise a ValueError.
    """
    _, _, A, l_in, r_in = random_complex_mpo

    with pytest.raises(ValueError, match="Unitarity strictly lost"):
        check_mpo_unitarity(N_max=N, A_np=A, l_in=l_in, r_in=r_in, tol=1e-6)


# ============================================================================
# SECTION 2: ASSUMPTION 1 (INJECTIVITY)
# ============================================================================


@pytest.mark.parametrize(
    "mpu_fixture",
    ["identity_mpu", "local_unitary_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
)
def test_check_assumption_1_valid(mpu_fixture, request):
    """
    Valid MPUs map the bond space injectively to the physical space.
    """
    _, D, A, l_in, r_in = request.getfixturevalue(mpu_fixture)

    s_left, s_right = check_assumption_1(A, l_in, r_in)

    assert np.sum(s_left > 1e-12) == D
    assert np.sum(s_right > 1e-12) == D


def test_check_assumption_1_deficient_rank():
    """
    Forces Assumption 1 to fail (d^2 < D), which must raise a ValueError.
    """
    d, D = 2, 5
    np.random.seed(42)
    A = np.random.randn(d, d, D, D)
    l_in = np.random.randn(D)
    r_in = np.random.randn(D)

    with pytest.raises(ValueError, match="Assumption 1 failed"):
        check_assumption_1(A, l_in, r_in)


def test_check_assumption_1_singular():
    """
    A degenerate tensor (zeros) must fail injectivity and raise a ValueError.
    """
    d, D = 2, 2
    A = np.zeros((d, d, D, D))
    l_in = np.ones(D)
    r_in = np.ones(D)

    with pytest.raises(ValueError, match="Assumption 1 failed"):
        check_assumption_1(A, l_in, r_in)


# ============================================================================
# SECTION 3: q_unif optimization
# ============================================================================
@pytest.mark.parametrize(
    "q_unif_result",
    ["identity_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
    indirect=True,
)
def test_optimize_q_unif_density_matrices_valid(q_unif_result):
    d, D, A, l_in, r_in, sigma, tau, q_val = q_unif_result
    for label, rho in [("sigma", sigma), ("tau", tau)]:
        assert np.allclose(rho, rho.conj().T, atol=1e-8), f"{label} not Hermitian"
        assert np.real(np.trace(rho)) == pytest.approx(1.0, abs=1e-6)
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals > -1e-8), f"{label} negative eigenvalue: {eigvals}"


# ---------------------------------------------------------------------
# 5.2: Returned density matrices have correct shape
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "q_unif_result",
    ["identity_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
    indirect=True,
)
def test_optimize_q_unif_output_shapes(q_unif_result):
    """
    Verifies that sigma and tau have shape (d, d) matching the physical dimension.
    """
    d, D, A, l_in, r_in, sigma, tau, q_val = q_unif_result

    assert sigma.shape == (d, d), f"Expected sigma shape {(d, d)}, got {sigma.shape}"
    assert tau.shape == (d, d), f"Expected tau shape {(d, d)}, got {tau.shape}"


# ---------------------------------------------------------------------
# 5.3: Optimized q_val is a finite positive real number
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "q_unif_result",
    ["identity_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
    indirect=True,
)
def test_optimize_q_unif_returns_finite_positive(q_unif_result):
    d, D, A, l_in, r_in, sigma, tau, q_val = q_unif_result
    assert isinstance(q_val, (float, np.floating))
    assert not np.isnan(q_val)
    assert not np.isinf(q_val)
    assert q_val > 0.0


# ---------------------------------------------------------------------
# 5.4: Identity MPU gives known analytical q_unif
# ---------------------------------------------------------------------
def test_optimize_q_unif_identity_analytical(identity_mpu):
    """
    For the identity MPU, the optimal q_unif has a known value.
    The optimizer must recover it.
    """
    d, D, A, l_in, r_in = identity_mpu

    # Compute locally
    np.random.seed(0)
    _, _, q_val = optimize_q_unif(A, l_in, r_in)

    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    expected_q = mpu.q_unif

    assert q_val == pytest.approx(expected_q, abs=1e-4)


# ---------------------------------------------------------------------
# 5.5: Optimizer result <= constructor result (rank-1 boundary)
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "q_unif_result",
    ["identity_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
    indirect=True,
)
def test_optimize_q_unif_leq_constructor(q_unif_result):
    d, D, A, l_in, r_in, sigma, tau, q_optimized = q_unif_result
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)
    assert q_optimized <= mpu.q_unif + 1e-4


# ---------------------------------------------------------------------
# 5.6: Self-consistency – recompute q from returned density matrices
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "q_unif_result",
    ["identity_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
    indirect=True,
)
def test_optimize_q_unif_self_consistency(q_unif_result):
    """
    Reconstruct L2 and R2 from the returned sigma/tau using the same
    einsum contractions, then manually compute q and compare to res.fun.
    """
    d, D, A, l_in, r_in, sigma, tau, q_val = q_unif_result
    eps_reg = 1e-8

    T = A
    L_in = np.outer(l_in, l_in.conj())
    R_in = np.outer(r_in, r_in.conj())

    # Reconstruct boundary operators from returned density matrices
    L2 = np.einsum("ij, oipq, pm, ojmn -> qn", sigma, T, L_in, T.conj())
    R2 = np.einsum("ij, oipq, qn, ojmn -> pm", tau, T, R_in, T.conj())

    # Regularized pseudoinverse via eigendecomposition (same as in loss)
    def reg_inv(M):
        eigvals, eigvecs = np.linalg.eigh(M)
        inv_diag = np.diag(1.0 / (np.maximum(eigvals, 0.0) + eps_reg))
        return eigvecs @ inv_diag @ eigvecs.conj().T

    inv_L = reg_inv(L2)
    inv_R = reg_inv(R2)

    q_recomputed = np.sqrt(np.abs(np.real(np.trace(inv_R @ inv_L.T))))

    assert q_recomputed == pytest.approx(
        q_val, abs=1e-4
    ), f"Recomputed q ({q_recomputed}) != returned q ({q_val})"


# ---------------------------------------------------------------------
# 5.7: Determinism under fixed random seed
# ---------------------------------------------------------------------
def test_optimize_q_unif_deterministic(cz_interaction_mpu):
    """
    With the same random seed, two calls must return identical results.
    """
    d, D, A, l_in, r_in = cz_interaction_mpu

    np.random.seed(42)
    sigma_1, tau_1, q_1 = optimize_q_unif(A, l_in, r_in)

    np.random.seed(42)
    sigma_2, tau_2, q_2 = optimize_q_unif(A, l_in, r_in)

    assert q_1 == pytest.approx(q_2, abs=1e-12)
    assert np.allclose(sigma_1, sigma_2, atol=1e-12)
    assert np.allclose(tau_1, tau_2, atol=1e-12)


# ---------------------------------------------------------------------
# 5.8: Ill-conditioned (semisimple D=5) does not crash
# ---------------------------------------------------------------------
@pytest.mark.parametrize("eps_reg", [1e-6, 1e-8, 1e-12])
def test_optimize_q_unif_semisimple_no_crash(eps_reg, semisimple_v_mpu):
    """
    The D=5 semisimple case may be ill-conditioned; verify no NaN/Inf.
    """
    d, D, A, l_in, r_in = semisimple_v_mpu
    np.random.seed(8)

    _, _, q_val = optimize_q_unif(A, l_in, r_in, eps_reg=eps_reg)

    assert not np.isnan(q_val)
    assert not np.isinf(q_val)
    assert q_val > 0.0


# ---------------------------------------------------------------------
# 6: Merging operator
# ---------------------------------------------------------------------
def test_merging_operator_kernel_value(identity_mpu):
    """
    For identity MPU (D=1), M[0,0,:,:] must equal R_inv.T @ L_inv exactly.
    """
    d, D, A, l_in, r_in = identity_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    M = mpu._get_merging_operator()
    expected_kernel = mpu.R_inv.T @ mpu.L_inv

    assert np.allclose(M[0, :], expected_kernel.flatten(), atol=1e-12)


def test_merging_operator_only_00_nonzero(cz_interaction_mpu):
    """
    By definition, only the M[0,0,:,:] block carries the kernel.
    All other slices must be exactly zero.
    """
    d, D, A, l_in, r_in = cz_interaction_mpu
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    M = mpu._get_merging_operator()

    for j in range(D**2):
        if j == 0:
            continue
        assert np.allclose(M[j, :], 0.0, atol=1e-15), f"M[{j},:] is nonzero"


@pytest.mark.parametrize(
    "mpu_fixture",
    ["identity_mpu", "cz_interaction_mpu", "semisimple_v_mpu"],
)
def test_merging_operator_complex_dtype(mpu_fixture, request):
    """
    Output must always be complex regardless of input dtype.
    """
    d, D, A, l_in, r_in = request.getfixturevalue(mpu_fixture)
    mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=4)

    M = mpu._get_merging_operator()
    assert np.iscomplexobj(M)


def test_merging_operator_nonsquare_raises():
    L_inv = np.ones((2, 3))
    R_inv = np.ones((2, 3))
    with pytest.raises(ValueError, match="square matrices"):
        get_merging_operator(L_inv, R_inv)


def test_merging_operator_dimension_mismatch_raises():
    L_inv = np.eye(2)
    R_inv = np.eye(3)
    with pytest.raises(ValueError, match="same dimension"):
        get_merging_operator(L_inv, R_inv)


# ---------------------------------------------------------------------
# 7: LCU verification
# ---------------------------------------------------------------------
def test_verify_lcu_passes_valid_decomposition(identity_lcu_data):
    """Valid LCU must pass all internal checks without raising."""
    D, coeffs, units, C, mpu = identity_lcu_data
    M_ref = mpu._get_merging_operator().reshape(D**2, D**2)
    # Should not raise
    verify_lcu(M_ref, coeffs, units)


def test_verify_lcu_fails_non_positive_coefficient(identity_lcu_data):
    """A negative coefficient must be rejected."""
    D, coeffs, units, C, mpu = identity_lcu_data
    M_ref = mpu._get_merging_operator().reshape(D**2, D**2)
    bad_coeffs = [-abs(coeffs[0])] + list(coeffs[1:])
    with pytest.raises(ValueError, match="not real positive"):
        verify_lcu(M_ref, bad_coeffs, units)


def test_verify_lcu_fails_non_unitary_matrix(identity_lcu_data):
    """A non-unitary matrix in the list must be rejected."""
    D, coeffs, units, C, mpu = identity_lcu_data
    M_ref = mpu._get_merging_operator().reshape(D**2, D**2)
    bad_units = [u.copy() for u in units]
    bad_units[0] = bad_units[0] * 2.0  # breaks unitarity
    with pytest.raises(ValueError, match="W @ W"):
        verify_lcu(M_ref, coeffs, bad_units)


def test_verify_lcu_fails_wrong_reconstruction(identity_lcu_data):
    """Correct coefficients and unitaries but wrong target M must be rejected."""
    D, coeffs, units, C, mpu = identity_lcu_data
    wrong_M = np.eye(D**2, dtype=complex) * 99.0
    with pytest.raises(ValueError, match="does not reconstruct"):
        verify_lcu(wrong_M, coeffs, units)


# ---------------------------------------------------------------------
# 8: Verify Merging Unitary
# ---------------------------------------------------------------------
def test_verify_merging_unitary_passes(identity_lcu_data):
    """
    For a valid LCU, the merging unitary must correctly embed M/C
    in the |0> ancilla subspace without raising.
    """
    D, coeffs, units, C, mpu = identity_lcu_data
    mpu._build_merging_unitary()

    B, W_ctrl = mpu._merging_unitary_cache
    M = mpu._get_merging_operator().reshape(D**2, D**2)
    dim_ancilla = B.shape[0]
    # Should not raise
    verify_merging_unitary(
        B=B,
        W_ctrl=W_ctrl,
        M_operator=M,
        C=C,
        dim_system=D**2,
        dim_ancilla=dim_ancilla,
    )


def test_verify_merging_unitary_fails_wrong_C(cz_lcu_data):
    """Passing a wrong normalization constant C must break post-selection."""
    D, coeffs, units, C, mpu = cz_lcu_data
    mpu._build_merging_unitary()

    B, W_ctrl = mpu._merging_unitary_cache
    M = mpu._get_merging_operator().reshape(D**2, D**2)
    dim_ancilla = B.shape[0]
    with pytest.raises(ValueError, match="Merging unitary post-selection failed."):
        verify_merging_unitary(
            B=B,
            W_ctrl=W_ctrl,
            M_operator=M,
            C=C * 2.0,  # wrong C
            dim_system=D**2,
            dim_ancilla=dim_ancilla,
        )
