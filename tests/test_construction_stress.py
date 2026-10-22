"""Construction-correctness tests of the synthesized circuits.

Every full-circuit test compares the synthesized circuit, simulated with
``qiskit.quantum_info.Statevector``, against a dense unitary contracted
directly from the raw input tensors with ``quimb``; intermediate proxies
(isometry residual, GOAA success flag, ``boundary_source`` label) are never
used on their own, because a self-reported residual can differ from the true
``||V^dagger V - I||`` and a symmetric boundary cannot expose an ``R`` versus
``R.T`` convention mismatch. Two tiers are used. Tier 1 compares the full
circuit with the dense reference at N = 2, 3 (one or two tree layers); circuit
size grows with N, so larger N is too slow to simulate routinely. Tier 2
checks the local isometry defect by direct tensor contraction and the tree
merge plan at larger N (4, 5, 7, including odd N), which exercises the same
multi-layer indexing without simulating a circuit. Unoptimized random
boundaries are excluded because they need many LCU terms and are slow to
simulate.
"""

import numpy as np
import pytest
import quimb.tensor as qtn
from qiskit.quantum_info import Statevector

from mpu_generators import generate_random_mpu
from TreeMPU import GeneralMPU, UniformMPU

# --- Shared helpers ---


def _dense_unitary_uniform(A, l_vec, r_vec, N):
    """Contract ``N`` copies of one tensor with the boundaries into a dense unitary."""
    tn = qtn.Tensor(l_vec, inds=["bond_0"])
    for k in range(N):
        A_tn = qtn.Tensor(A, inds=[f"p_in_{k}", f"p_out_{k}", f"bond_{k}", f"bond_{k + 1}"])
        tn = tn @ A_tn
    tn = tn @ qtn.Tensor(r_vec, inds=[f"bond_{N}"])
    out_inds = [f"p_out_{k}" for k in reversed(range(N))]
    in_inds = [f"p_in_{k}" for k in reversed(range(N))]
    return tn.to_dense(out_inds, in_inds)


def _dense_unitary_general(tensors, l_vec, r_vec):
    """Contract a list of site tensors with the boundaries into a dense unitary."""
    N = len(tensors)
    tn = qtn.Tensor(l_vec, inds=["bond_0"])
    for k, A_k in enumerate(tensors):
        A_tn = qtn.Tensor(A_k, inds=[f"p_in_{k}", f"p_out_{k}", f"bond_{k}", f"bond_{k + 1}"])
        tn = tn @ A_tn
    tn = tn @ qtn.Tensor(r_vec, inds=[f"bond_{N}"])
    out_inds = [f"p_out_{k}" for k in reversed(range(N))]
    in_inds = [f"p_in_{k}" for k in reversed(range(N))]
    return tn.to_dense(out_inds, in_inds)


def _assert_synthesize_matches_reference(mpu, U_ref, atol=1e-6):
    """Assert that the circuit equals ``U_ref`` up to a phase with exact GOAA success."""
    qc, site_sys_regs, _, _, _ = mpu.synthesize()
    sys_indices = [qc.find_bit(q).index for reg in site_sys_regs for q in reg]
    n_sys = len(sys_indices)
    dim = 2**n_sys

    cols = []
    p_anc0 = np.zeros(dim)
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
        p_anc0[basis_idx] = float(np.sum(np.abs(psi_sys) ** 2))
        cols.append(psi_sys)

    U_rec = np.column_stack(cols)
    assert np.all(p_anc0 > 1 - 1e-6), f"GOAA did not succeed exactly: {p_anc0}"

    idx = np.unravel_index(np.argmax(np.abs(U_ref)), U_ref.shape)
    phase = U_ref[idx] / U_rec[idx]
    phase /= np.abs(phase)
    np.testing.assert_allclose(U_rec * phase, U_ref, atol=atol)


def _isometry_defect(V_tn):
    """Return ``||V^dagger V - I||`` of a local isometry tensor by direct contraction."""
    in_inds = sorted(i for i in V_tn.inds if i.startswith("p_in"))
    out_inds = [i for i in V_tn.inds if i not in in_inds]
    mat = V_tn.to_dense(out_inds, in_inds)
    return float(np.linalg.norm(mat.conj().T @ mat - np.eye(mat.shape[1])))


def _twist(A, W):
    """Conjugate the physical output leg of a bulk tensor by ``W``.

    Site-dependent twists of one tensor give a chain with a known unitary.
    """
    return np.einsum("jJ,iJab->ijab", W, A)


def _random_unitary(d, rng):
    """Return a Haar-distributed unitary of dimension ``d``."""
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Q, R = np.linalg.qr(X)
    return Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))


# --- tier 1: exact Mode I boundary, full circuit vs dense reference (N <= 3) ---


@pytest.mark.parametrize(
    "D,d,N,seed",
    [
        (2, 2, 2, 0),
        (2, 2, 3, 0),  # multi-layer (2 tree layers)
    ],
)
def test_uniformmpu_mode1_exact_synthesizes_correctly(D, d, N, seed):
    """The Mode I exact-boundary circuit of a uniform MPU equals the dense unitary."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_blocks=2,
        d=d,
        bond_dim=D,
        mode="branching",
        uniform=True,
        blocking="auto",
        seed=seed,
    )
    A = tensors[0]
    U_ref = _dense_unitary_uniform(A, l_vec, r_vec, N)

    mpu = UniformMPU(A=A, l_vec=l_vec, r_vec=r_vec, N=N, boundary_mode1_exact=True)
    assert mpu.boundary_source == "mode1_exact"
    assert abs(mpu.q_unif - D) < 1e-8

    _assert_synthesize_matches_reference(mpu, U_ref)


# --- limitation: D = 2 exact boundary with d not a power of 2 ---
# For D = 2 the exact Mode I boundary has q = D = 2, which lies on the ell = 1
# GOAA target (rotation angle phi = 0). With a physical register padded to a
# power of 2 (d not a power of 2) the GOAA circuit does not reach p_anc0 = 1,
# although the local isometries are exact. The test is a strict xfail so that
# any change in this behaviour is flagged.


@pytest.mark.xfail(
    reason="Unsupported: with q = D = 2 (rotation angle phi = 0, ell = 1) and a "
    "physical register padded because d is not a power of 2, the GOAA rotation "
    "circuit does not reach exact success (p_anc0 < 1) although the local "
    "isometries are exact.",
    strict=True,
)
@pytest.mark.parametrize(
    "d,seed",
    [
        (3, 1),
        (5, 5),
    ],
)
def test_uniformmpu_mode1_exact_D2_nonpow2_d_goaa_not_exact(d, seed):
    """D = 2 exact boundaries with d not a power of 2 give inexact GOAA success."""
    D, N = 2, 2
    tensors, l_vec, r_vec = generate_random_mpu(
        n_blocks=2,
        d=d,
        bond_dim=D,
        mode="branching",
        uniform=True,
        blocking="auto",
        seed=seed,
    )
    A = tensors[0]
    U_ref = _dense_unitary_uniform(A, l_vec, r_vec, N)

    mpu = UniformMPU(A=A, l_vec=l_vec, r_vec=r_vec, N=N, boundary_mode1_exact=True)
    assert mpu.boundary_source == "mode1_exact"

    _assert_synthesize_matches_reference(mpu, U_ref)


# --- tier 2: cheap exact-isometry check at larger N and D ---


@pytest.mark.parametrize(
    "D,d,N,seed",
    [
        (2, 2, 4, 2),  # two simultaneous same-layer merges
        (2, 2, 5, 6),  # odd, 3 tree layers
        (2, 2, 7, 8),  # odd, beyond the minimal case
        (2, 3, 2, 1),  # d not a power of 2, D > 1 (padded physical register)
        (2, 5, 2, 5),  # d=5, also not a power of 2
        (3, 2, 6, 9),
        (3, 3, 2, 7),  # d not a power of 2 combined with D=3
        (4, 2, 3, 4),
    ],
)
def test_uniformmpu_mode1_exact_isometries_exact_at_larger_N(D, d, N, seed):
    """Exact Mode I boundaries give local isometries with machine-precision defect."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_blocks=2,
        d=d,
        bond_dim=D,
        mode="branching",
        uniform=True,
        blocking="auto",
        seed=seed,
    )
    A = tensors[0]
    mpu = UniformMPU(A=A, l_vec=l_vec, r_vec=r_vec, N=N, boundary_mode1_exact=True)
    assert mpu.boundary_source == "mode1_exact"
    assert abs(mpu.q_unif - D) < 1e-8

    V_l, V_b, V_r = mpu.create_local_isometries()
    for k, V_tn in enumerate((V_l, V_b, V_r)):
        assert _isometry_defect(V_tn) < 1e-8, f"site {k} tensor is not an isometry"


# --- gauge-optimized (non-Hermitian, asymmetric L/R) boundary at a multi-layer N ---
# Only an asymmetric boundary distinguishes the R and R.T conventions.


@pytest.fixture(scope="module")
def cz_tensor():
    """Return a CZ-like bulk tensor with an asymmetric boundary pair."""
    A_00 = np.array([[1, 0], [0, 0]], dtype=complex)
    A_01 = np.zeros((2, 2), dtype=complex)
    A_10 = np.zeros((2, 2), dtype=complex)
    A_11 = np.array([[1, 0], [0, 1]], dtype=complex)
    A = np.array([[A_00, A_01], [A_10, A_11]], dtype=complex)
    l_in = np.array([1.0, 1.0], dtype=complex)
    r_in = np.array([1.0, -2.0], dtype=complex)
    return A, l_in, r_in


def test_uniformmpu_gauge_optimized_synthesizes_correctly(cz_tensor):
    """A gauge-optimized, non-Hermitian boundary pair gives a correct circuit.

    The synthesized circuit equals the dense unitary.
    """
    # N is capped at 2 because circuit simulation cost grows quickly with N.
    N = 2
    A, l_in, r_in = cz_tensor
    U_ref = _dense_unitary_uniform(A, l_in, r_in, N)

    mpu = UniformMPU(
        A=A,
        l_vec=l_in,
        r_vec=r_in,
        N=N,
        boundary_opt_k=1,
        boundary_opt_gauge=True,
        boundary_opt_n_restarts=8,
        boundary_opt_seed=0,
    )
    assert mpu.boundary_source == "optimized_gauge"
    # A gauge-dressed (L, R) is generically not Hermitian; the asymmetry is what
    # distinguishes the R and R.T conventions.
    assert (
        np.linalg.norm(mpu.L - mpu.L.conj().T) > 1e-6
        or np.linalg.norm(mpu.R - mpu.R.conj().T) > 1e-6
    )

    _assert_synthesize_matches_reference(mpu, U_ref)


# --- non-uniform chains: a different random twist at every site ---


@pytest.fixture(scope="module")
def random_twisted_chain():
    """Return a 7-site chain of distinct tensors and its boundaries.

    The tensors are independent twists of one CZ-like tensor.
    """
    A_00 = np.array([[1, 0], [0, 0]], dtype=complex)
    A_01 = np.zeros((2, 2), dtype=complex)
    A_10 = np.zeros((2, 2), dtype=complex)
    A_11 = np.array([[1, 0], [0, 1]], dtype=complex)
    A = np.array([[A_00, A_01], [A_10, A_11]], dtype=complex)
    l_in = np.array([1.0, 1.0], dtype=complex)
    r_in = np.array([1.0, -2.0], dtype=complex)

    rng = np.random.default_rng(11)
    tensors = [A] + [_twist(A, _random_unitary(2, rng)) for _ in range(6)]
    return tensors, l_in, r_in


def test_generalmpu_sigma_tau_synthesizes_correctly(random_twisted_chain):
    """The sigma/tau-boundary circuit of a non-uniform MPU equals the dense unitary."""
    N = 2
    tensors, l_in, r_in = random_twisted_chain
    tensors = tensors[:N]
    U_ref = _dense_unitary_general(tensors, l_in, r_in)

    mpu = GeneralMPU(tensors, l_in, r_in, boundary_sigma_tau=True)
    _assert_synthesize_matches_reference(mpu, U_ref)


def test_generalmpu_boundary_opt_p_synthesizes_correctly(random_twisted_chain):
    """The optimized-boundary circuit of a non-uniform MPU equals the dense unitary."""
    N = 2
    tensors, l_in, r_in = random_twisted_chain
    tensors = tensors[:N]
    U_ref = _dense_unitary_general(tensors, l_in, r_in)

    mpu = GeneralMPU(tensors, l_in, r_in, boundary_opt_p=0, boundary_opt_n_restarts=4)
    _assert_synthesize_matches_reference(mpu, U_ref)


@pytest.mark.parametrize("N", [4, 5, 7])
def test_generalmpu_sigma_tau_isometries_exact_at_larger_N(random_twisted_chain, N):
    """The sigma/tau site tensors are isometries at larger N."""
    tensors, l_in, r_in = random_twisted_chain
    tensors = tensors[:N]

    mpu = GeneralMPU(tensors, l_in, r_in, boundary_sigma_tau=True)
    for k, V_tn in enumerate(mpu._site_tensors):
        assert _isometry_defect(V_tn) < 1e-6, f"site {k} tensor is not an isometry"


def test_generalmpu_tree_merge_plan_matches_expected_structure_odd_N():
    """The tree-merge plan for odd N carries the leftover segment forward as expected."""
    # The analytic resource estimator consumes the same plan, so N = 7, 9 are
    # also covered through the estimator tests.
    assert GeneralMPU._tree_merge_plan(7) == [[0, 2, 4], [1, 5], [3]]
    assert GeneralMPU._tree_merge_plan(9) == [[0, 2, 4, 6], [1, 5], [3], [7]]
