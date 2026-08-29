import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from TreeMPU import UniformMPU
from TreeMPU.utils import optimize_boundary_operators


def _isometry_residual(L, R, A, k=1, l_vec=None, r_vec=None):
    """
    Independent re-check of (V_n^[L,R])^dagger V_n^[L,R] = I, n=1..k, plus
    the boundary isometries when l_vec/r_vec are given.

    Uses the *same* index convention as create_local_isometries
    (R_tn = Tensor(R, inds=["ext_R", "bond_1"]), i.e. R acts as
    R[output, bond_R] -- R's transpose relative to the more "natural"
    L[output, bond_L] pattern L already follows), so this is a faithful
    stand-in for what the production tree-merge circuit actually needs, not
    just an internally-consistent alternative convention.
    """
    d = A.shape[0]
    Rt = R.T
    total = 0.0
    LAR = np.einsum("ab,ijbc,cd->ijad", L, A, Rt)
    VdV = np.einsum("ijab,pjab->ip", LAR.conj(), LAR)
    total += np.linalg.norm(VdV - np.eye(d)) ** 2
    if k >= 2:
        # n=2: chain two copies of A before dressing with L, R.
        AA = np.einsum("ijac,klcb->ikjlab", A, A)  # (i,k,j,l,a,b)
        AA = AA.reshape(d * d, d * d, A.shape[2], A.shape[3])
        LAR2 = np.einsum("ab,pqbc,cd->pqad", L, AA, Rt)
        VdV2 = np.einsum("pqab,rqab->pr", LAR2.conj(), LAR2)
        total += np.linalg.norm(VdV2 - np.eye(d * d)) ** 2

    if l_vec is not None:
        Vl = np.einsum("b,ijbc,cd->ijd", l_vec, A, Rt)
        VldVl = np.einsum("ijd,pjd->ip", Vl.conj(), Vl)
        total += np.linalg.norm(VldVl - np.eye(d)) ** 2

    if r_vec is not None:
        Vr = np.einsum("ab,ijbc,c->ija", L, A, r_vec)
        VrdVr = np.einsum("ija,pja->ip", Vr.conj(), Vr)
        total += np.linalg.norm(VrdVr - np.eye(d)) ** 2

    return float(np.sqrt(total))


@pytest.fixture(scope="module")
def cz_tensor():
    A_00 = np.array([[1, 0], [0, 0]], dtype=complex)
    A_01 = np.zeros((2, 2), dtype=complex)
    A_10 = np.zeros((2, 2), dtype=complex)
    A_11 = np.array([[1, 0], [0, 1]], dtype=complex)
    A = np.array([[A_00, A_01], [A_10, A_11]], dtype=complex)
    l_in = np.array([1.0, 1.0], dtype=complex)
    r_in = np.array([1.0, -2.0], dtype=complex)
    return A, l_in, r_in


def test_optimizer_matches_or_beats_canonical(cz_tensor):
    A, l_in, r_in = cz_tensor
    umpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=2)

    L_opt, R_opt, q_opt, iso_res = optimize_boundary_operators(
        A, k=1, l_vec=l_in, r_vec=r_in, n_restarts=5, lam_iso=1e6, seed=0
    )

    assert iso_res < 1e-3
    assert q_opt <= umpu.q_unif + 1e-3
    assert q_opt >= 1.0 - 1e-6

    # Returned (L_opt, R_opt) must independently satisfy the bulk *and*
    # boundary isometry conditions the optimizer targeted.
    assert _isometry_residual(L_opt, R_opt, A, k=1, l_vec=l_in, r_vec=r_in) < 1e-3


def test_optimizer_k2_also_satisfies_length_2_isometry(cz_tensor):
    A, l_in, r_in = cz_tensor
    L_opt, R_opt, q_opt, iso_res = optimize_boundary_operators(
        A, k=2, l_vec=l_in, r_vec=r_in, n_restarts=5, lam_iso=1e6, seed=1
    )
    assert iso_res < 1e-3
    assert _isometry_residual(L_opt, R_opt, A, k=2, l_vec=l_in, r_vec=r_in) < 1e-3


def test_gauge_optimizer_never_worse_than_hpd_only(cz_tensor):
    """
    Gauge optimization explores a strictly larger feasible set (thesis
    section "Gauge optimization and the q_unif orbit": taking G=I recovers
    the HPD-only case exactly), so q_gauge can never exceed q_hpd. It is
    NOT guaranteed to do strictly better for every tensor -- for this
    particular cz_interaction_mpu fixture, both land at q=2.0 (verified
    independently across several seeds/restart counts/lam_iso up to 1e8:
    2.0 appears to be a hard floor for this specific tensor, gauge or not).
    This only checks the "never worse" guarantee; it is not evidence that
    gauge helps in general (it does for other tensors, e.g. the notebook's
    subspace_unitary_rotation case).
    """
    A, l_in, r_in = cz_tensor

    _, _, q_hpd, iso_hpd = optimize_boundary_operators(
        A, k=1, gauge=False, l_vec=l_in, r_vec=r_in, n_restarts=5, lam_iso=1e6, seed=0
    )
    L_g, R_g, q_gauge, iso_gauge = optimize_boundary_operators(
        A, k=1, gauge=True, l_vec=l_in, r_vec=r_in, n_restarts=5, lam_iso=1e6, seed=0
    )

    assert iso_gauge < 1e-3
    assert q_gauge <= q_hpd + 1e-3
    assert q_gauge >= 1.0 - 1e-6

    # The returned pair is already the effective (generally non-HPD)
    # boundary completion -- check it directly, not via a separate gauge.
    assert _isometry_residual(L_g, R_g, A, k=1, l_vec=l_in, r_vec=r_in) < 1e-3


def test_uniformmpu_synthesize_uses_optimized_boundaries_correctly(cz_tensor):
    """
    End-to-end regression test for the UniformMPU(boundary_opt_k=...,
    boundary_opt_gauge=True) integration: this is exactly the path that
    exposed the R-vs-R.T convention mismatch between
    optimize_boundary_operators and create_local_isometries (gauge-optimized
    boundaries are the least likely to be accidentally symmetric, which is
    what let that mismatch hide behind passing tests for real/HPD-only
    cases). synthesize() must still produce a circuit implementing the
    exact target unitary (CZ, for this fixture) with deterministic (p_anc0
    == 1) GOAA success.
    """
    A, l_in, r_in = cz_tensor
    mpu = UniformMPU(
        A=A,
        l_vec=l_in,
        r_vec=r_in,
        N=2,
        boundary_opt_k=1,
        boundary_opt_gauge=True,
        boundary_opt_n_restarts=8,
        boundary_opt_seed=0,
    )
    assert mpu.boundary_source == "optimized_gauge"

    qc, site_sys_regs, _, _, _ = mpu.synthesize()
    sys_indices = [qc.find_bit(q).index for reg in site_sys_regs for q in reg]
    n_sys = len(sys_indices)
    dim = 2**n_sys

    U_ref = np.diag([1, 1, 1, -1]).astype(complex)
    cols = []
    for basis_idx in range(dim):
        bits = format(basis_idx, f"0{n_sys}b")
        qc_test = qc.copy_empty_like()
        for i, idx in enumerate(sys_indices):
            if bits[-(i + 1)] == "1":
                qc_test.x(idx)
        qc_test.compose(qc, inplace=True)

        sv = Statevector.from_instruction(qc_test)
        psi_sys = np.zeros(dim, dtype=complex)
        for s in range(dim):
            full_idx = 0
            for i, q in enumerate(sys_indices):
                if (s >> i) & 1:
                    full_idx |= 1 << q
            psi_sys[s] = sv.data[full_idx]
        assert np.sum(np.abs(psi_sys) ** 2) == pytest.approx(1.0, abs=1e-4)
        cols.append(psi_sys)

    U_rec = np.column_stack(cols)
    idx = np.unravel_index(np.argmax(np.abs(U_ref)), U_ref.shape)
    phase = U_ref[idx] / U_rec[idx]
    phase /= np.abs(phase)
    np.testing.assert_allclose(U_rec * phase, U_ref, atol=1e-3)


def test_optimizer_degenerate_D1_can_fall_below_q_one():
    """
    For D=1 the isometry constraint alone pins L*R (hence q) to a unique
    value; for the identity MPU that value is 1/sqrt(d) < 1. This documents
    the known limitation (see optimize_boundary_operators' docstring):
    callers must check q_opt >= 1 before trusting the result, since q < 1
    is not usable by the GOAA-based circuit synthesis (arcsin(1/C)
    requires C >= 1).
    """
    d, D = 2, 1
    A = np.einsum("ij,ab->ijab", np.eye(d), np.eye(D)) / np.sqrt(d)

    _, _, q_opt, iso_res = optimize_boundary_operators(
        A, k=1, n_restarts=3, lam_iso=1e6, seed=0
    )

    assert q_opt < 1.0
    assert q_opt == pytest.approx(1.0 / np.sqrt(d), abs=0.1)
