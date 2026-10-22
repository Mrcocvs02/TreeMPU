"""Tests of ``TreeMPU.GeneralMPU``, the site-dependent MPU synthesizer.

Genuinely non-uniform chains are built by twisting the output leg of the CZ
bulk tensor with local unitaries, so the reference unitary is known by
construction and is contracted densely with ``quimb`` independently of the
code under test. The tests verify input validation, the per-site isometry
property (thesis Lemma 1), agreement of the boundary operators with
``UniformMPU`` at the chain ends, end-to-end equality of the synthesized
circuit (``Statevector`` simulation) with the dense reference, and the
boundary optimizers (partial optimization and sigma/tau), which must never be
worse than the canonical construction and must keep every site an isometry.
"""

import numpy as np
import pytest
import quimb.tensor as qtn
from qiskit.quantum_info import Statevector

from TreeMPU import GeneralMPU, UniformMPU

# --- Helpers ---


def _twist(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Conjugate the physical output leg of a bulk tensor by the local unitary ``W``."""
    return np.einsum("jJ,iJab->ijab", W, A)


def _dense_unitary_from_tensors(tensors: list, l_vec: np.ndarray, r_vec: np.ndarray) -> np.ndarray:
    """Return the dense unitary contracted from the site tensors and boundaries."""
    N = len(tensors)
    tn = qtn.Tensor(l_vec, inds=["bond_0"])
    for k, A_k in enumerate(tensors):
        A_tn = qtn.Tensor(A_k, inds=[f"p_in_{k}", f"p_out_{k}", f"bond_{k}", f"bond_{k + 1}"])
        tn = tn @ A_tn
    tn = tn @ qtn.Tensor(r_vec, inds=[f"bond_{N}"])
    # Reversed site order: quimb's to_dense treats the first-listed index as
    # the most significant (kron-style), whereas Qiskit's little-endian
    # Statevector treats the first-declared qubit (site 0) as least
    # significant. Reversing aligns the reference matrix with the circuit.
    out_inds = [f"p_out_{k}" for k in reversed(range(N))]
    in_inds = [f"p_in_{k}" for k in reversed(range(N))]
    return tn.to_dense(out_inds, in_inds)


def _isometry_defect(V_tn: qtn.Tensor) -> float:
    """Return ``||V^dagger V - I||`` for a local (possibly boundary) isometry tensor."""
    in_inds = sorted(i for i in V_tn.inds if i.startswith("p_in"))
    out_inds = sorted(i for i in V_tn.inds if i.startswith("p_out")) + [
        t for t in ("ext_L", "ext_R") if t in V_tn.inds
    ]
    mat = V_tn.to_dense(out_inds, in_inds)
    return float(np.linalg.norm(mat.conj().T @ mat - np.eye(mat.shape[1])))


def _reconstruct_action(qc, site_sys_regs, all_anc_qubits):
    """Return the circuit's system action on the ancilla-|0> slice and p_anc0.

    p_anc0 is the vector of success probabilities, one per input basis state.
    """
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
            # every anc_indices bit is left at 0 in full_idx
            psi_sys[s] = sv.data[full_idx]

        p_anc0[basis_idx] = float(np.sum(np.abs(psi_sys) ** 2))
        cols.append(psi_sys)

    return np.column_stack(cols), p_anc0


def _assert_circuit_matches_reference(mpu, U_ref: np.ndarray, atol: float = 1e-6):
    """Assert that the circuit equals ``U_ref`` up to a phase with exact GOAA success."""
    qc, site_sys_regs, _, _, all_anc_qubits = mpu.synthesize()
    U_rec, p_anc0 = _reconstruct_action(qc, site_sys_regs, all_anc_qubits)

    assert np.all(p_anc0 > 1 - 1e-6), f"GOAA did not succeed exactly: {p_anc0}"

    # Fix the (input-independent) global phase using the reference's largest
    # entry, then compare the full reconstructed action.
    idx = np.unravel_index(np.argmax(np.abs(U_ref)), U_ref.shape)
    phase = U_ref[idx] / U_rec[idx]
    phase /= np.abs(phase)
    np.testing.assert_allclose(U_rec * phase, U_ref, atol=atol)


# --- Fixtures: a non-uniform tensor family that is unitary by construction ---


@pytest.fixture(scope="module")
def cz_tensor():
    """Return the D = 2 CZ bulk tensor with boundaries.

    The tensor is the one behind the ``cz_interaction_mpu`` fixture of conftest.
    """
    A_00 = np.array([[1, 0], [0, 0]], dtype=complex)
    A_01 = np.zeros((2, 2), dtype=complex)
    A_10 = np.zeros((2, 2), dtype=complex)
    A_11 = np.array([[1, 0], [0, 1]], dtype=complex)
    A = np.array([[A_00, A_01], [A_10, A_11]], dtype=complex)
    l_in = np.array([1.0, 1.0], dtype=complex)
    r_in = np.array([1.0, -2.0], dtype=complex)
    return A, l_in, r_in


@pytest.fixture(scope="module")
def twisted_2site(cz_tensor):
    """Return a 2-site non-uniform chain (X twist on site 1) and its dense unitary."""
    A, l_in, r_in = cz_tensor
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    tensors = [A, _twist(A, X)]
    U_ref = _dense_unitary_from_tensors(tensors, l_in, r_in)
    return tensors, l_in, r_in, U_ref


@pytest.fixture(scope="module")
def twisted_3site(cz_tensor):
    """Return a 3-site non-uniform chain (X and H twists) and its dense unitary."""
    A, l_in, r_in = cz_tensor
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    tensors = [A, _twist(A, X), _twist(A, H)]
    U_ref = _dense_unitary_from_tensors(tensors, l_in, r_in)
    return tensors, l_in, r_in, U_ref


# --- Construction validation ---


def test_generalmpu_rejects_bond_dimension_mismatch(cz_tensor):
    """A mismatch between neighbouring bond dimensions is rejected."""
    A, l_in, r_in = cz_tensor
    # The left bond (3) mismatches the neighbour's right bond (2) while the right bond
    # still matches r_vec, which isolates the internal-bond check.
    A_wrong = np.zeros((2, 2, 3, 2), dtype=complex)
    A_wrong[:, :, :2, :] = A

    with pytest.raises(ValueError, match="Bond dimension mismatch"):
        GeneralMPU([A, A_wrong], l_in, r_in)


def test_generalmpu_rejects_left_boundary_dimension_mismatch(cz_tensor):
    """A left boundary vector of the wrong length is rejected."""
    A, l_in, r_in = cz_tensor
    with pytest.raises(ValueError, match="l_vec has dimension"):
        GeneralMPU([A, A], np.array([1.0, 1.0, 1.0]), r_in)


def test_generalmpu_rejects_right_boundary_dimension_mismatch(cz_tensor):
    """A right boundary vector of the wrong length is rejected."""
    A, l_in, r_in = cz_tensor
    with pytest.raises(ValueError, match="r_vec has dimension"):
        GeneralMPU([A, A], l_in, np.array([1.0, 1.0, 1.0]))


def test_generalmpu_rejects_non_unitary_chain(twisted_2site):
    """A perturbed, non-unitary chain is rejected."""
    tensors, l_in, r_in, _ = twisted_2site
    A0, A1 = tensors
    A1_broken = A1.copy()
    A1_broken[0, 0, 0, 0] += 0.1

    with pytest.raises(ValueError, match="Unitarity strictly lost"):
        GeneralMPU([A0, A1_broken], l_in, r_in)


def test_generalmpu_rejects_rank_deficient_bond(twisted_2site):
    """A redundant bond is rejected as given and reduced by canonicalization."""
    # Padding the shared bond with an unused dimension preserves unitarity but
    # makes the bond rank deficient; with canonicalize=True it is reduced back to D = 2.
    tensors, l_in, r_in, _ = twisted_2site
    A0, A1 = tensors

    A0_pad = np.zeros((2, 2, 2, 3), dtype=complex)
    A0_pad[:, :, :, :2] = A0
    A1_pad = np.zeros((2, 2, 3, 2), dtype=complex)
    A1_pad[:, :, :2, :] = A1

    with pytest.raises(ValueError, match="Minimal bond dimension"):
        GeneralMPU([A0_pad, A1_pad], l_in, r_in, canonicalize=False)

    mpu = GeneralMPU([A0_pad, A1_pad], l_in, r_in)  # canonicalize=True default
    assert mpu._bond_dims == [2]


# --- Per-site isometry property (Lemma 1) and boundary-operator consistency ---


@pytest.mark.parametrize("fixture_name", ["twisted_2site", "twisted_3site"])
def test_generalmpu_site_tensors_are_isometries(fixture_name, request):
    """Every site tensor is an isometry (thesis Lemma 1)."""
    tensors, l_in, r_in, _ = request.getfixturevalue(fixture_name)
    mpu = GeneralMPU(tensors, l_in, r_in)

    for k, V_tn in enumerate(mpu._site_tensors):
        assert _isometry_defect(V_tn) < 1e-8, f"site {k} tensor is not an isometry"


def test_generalmpu_boundary_L_R_match_uniform_at_edges(cz_tensor):
    """The boundary operators at the chain ends equal those of UniformMPU."""
    # Both constructions use a single-site maximally mixed support at the chain
    # boundary, so they agree there; interior bonds may differ. canonicalize=False
    # keeps the as-given gauge, which UniformMPU also uses.
    A, l_in, r_in = cz_tensor
    N = 4
    umpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=N)
    gmpu = GeneralMPU([A] * N, l_in, r_in, canonicalize=False)

    np.testing.assert_allclose(gmpu.L[0], umpu.L, atol=1e-10)
    np.testing.assert_allclose(gmpu.R[N - 2], umpu.R, atol=1e-10)


# --- End-to-end circuit correctness ---


def test_generalmpu_synthesize_matches_reference_2site(twisted_2site):
    """The synthesized 2-site circuit equals the dense unitary."""
    tensors, l_in, r_in, U_ref = twisted_2site
    mpu = GeneralMPU(tensors, l_in, r_in)
    _assert_circuit_matches_reference(mpu, U_ref)


def test_generalmpu_synthesize_builds_3site_circuit(twisted_3site):
    """The 3-site circuit has one merge bundle per bond with consistent ancillas."""
    # Statevector simulation of the full 3-site circuit is too slow for a routine test; the
    # shared merge/GOAA code path is simulated at 2 sites above, so only the assembly of
    # one bundle per bond is checked here.
    tensors, l_in, r_in, _ = twisted_3site
    mpu = GeneralMPU(tensors, l_in, r_in)

    assert mpu._tree_merge_plan(3) == [[0], [1]]

    qc, site_sys_regs, site_bond_regs, anc_regs, all_anc_qubits = mpu.synthesize()

    assert qc.num_qubits == sum(r.size for r in site_sys_regs) + len(all_anc_qubits)
    assert len(anc_regs) == 2
    # Each layer uses its own bond's bundle; the ancilla widths of different bonds may
    # differ but must match their own bundle.
    for layer, layer_regs in enumerate(anc_regs):
        (lcu_r, rot_r) = layer_regs[0]
        bundle = mpu._merge_bundle(layer)
        assert lcu_r.size == bundle["n_lcu"]
        assert rot_r.size == bundle["n_rot"]


# --- Partial optimization (power-of-two boundary lengths) ---


def test_generalmpu_boundary_opt_p_synthesizes_correctly(twisted_2site):
    """Partial boundary optimization records a valid source.

    The synthesized circuit still equals the dense unitary.
    """
    # The optimizer need not beat the canonical boundary for this tensor, so only a valid
    # source is asserted; the circuit must reproduce the target in either case.
    tensors, l_in, r_in, U_ref = twisted_2site
    mpu = GeneralMPU(tensors, l_in, r_in, boundary_opt_p=0, boundary_opt_n_restarts=3)

    assert mpu.boundary_source[0] in ("canonical", "optimized", "optimized_gauge")
    assert mpu.q >= 1.0 - 1e-6

    _assert_circuit_matches_reference(mpu, U_ref)


def test_generalmpu_boundary_opt_p_can_improve_on_canonical(twisted_2site):
    """Partial optimization with more restarts is no worse than canonical."""
    tensors, l_in, r_in, U_ref = twisted_2site
    mpu_canon = GeneralMPU(tensors, l_in, r_in)
    mpu_opt = GeneralMPU(tensors, l_in, r_in, boundary_opt_p=0, boundary_opt_n_restarts=8)

    assert mpu_opt.q <= mpu_canon.q + 1e-6
    _assert_circuit_matches_reference(mpu_opt, U_ref)


def test_generalmpu_boundary_opt_p_none_matches_canonical(twisted_2site):
    """boundary_opt_p=None reproduces the canonical construction."""
    tensors, l_in, r_in, _ = twisted_2site
    mpu_canon = GeneralMPU(tensors, l_in, r_in)
    mpu_default = GeneralMPU(tensors, l_in, r_in, boundary_opt_p=None)

    assert mpu_default.boundary_source == ["canonical"]
    np.testing.assert_allclose(mpu_default.L[0], mpu_canon.L[0])
    np.testing.assert_allclose(mpu_default.R[0], mpu_canon.R[0])


# --- sigma/tau-optimized chain baseline ---


@pytest.fixture(scope="module")
def twisted_5site(cz_tensor):
    """Return a 5-site non-uniform chain, longer than the optimization window."""
    A, l_in, r_in = cz_tensor
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    rng = np.random.default_rng(7)

    def _random_unitary(d, rng):
        """Return a Haar-distributed ``d x d`` unitary drawn from ``rng``."""
        Xr = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        Q, R = np.linalg.qr(Xr)
        return Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))

    W1 = _random_unitary(2, rng)
    W2 = _random_unitary(2, rng)
    tensors = [A, _twist(A, X), _twist(A, H), _twist(A, W1), _twist(A, W2)]
    return tensors, l_in, r_in


def test_generalmpu_boundary_sigma_tau_false_matches_canonical(twisted_2site):
    """boundary_sigma_tau=False reproduces the canonical construction."""
    tensors, l_in, r_in, _ = twisted_2site
    mpu_canon = GeneralMPU(tensors, l_in, r_in)
    mpu_default = GeneralMPU(tensors, l_in, r_in, boundary_sigma_tau=False)

    assert mpu_default.boundary_source == mpu_canon.boundary_source == ["canonical"]
    np.testing.assert_allclose(mpu_default.L[0], mpu_canon.L[0])
    np.testing.assert_allclose(mpu_default.R[0], mpu_canon.R[0])


def test_generalmpu_boundary_sigma_tau_never_worse_than_canonical(twisted_5site):
    """The sigma/tau boundary never has a larger q than the canonical one."""
    tensors, l_in, r_in = twisted_5site
    mpu_canon = GeneralMPU(tensors, l_in, r_in)
    mpu_st = GeneralMPU(tensors, l_in, r_in, boundary_sigma_tau=True)

    assert mpu_st.q <= mpu_canon.q + 1e-9
    assert any(src == "sigma_tau_optimized" for src in mpu_st.boundary_source)


def test_generalmpu_boundary_sigma_tau_max_sites_windowing_is_monotonic(twisted_5site):
    """A larger sigma/tau window never gives a larger q."""
    tensors, l_in, r_in = twisted_5site
    mpu_naive = GeneralMPU(tensors, l_in, r_in)
    mpu_w1 = GeneralMPU(
        tensors, l_in, r_in, boundary_sigma_tau=True, boundary_sigma_tau_max_sites=1
    )
    mpu_w2 = GeneralMPU(
        tensors, l_in, r_in, boundary_sigma_tau=True, boundary_sigma_tau_max_sites=2
    )
    mpu_full = GeneralMPU(tensors, l_in, r_in, boundary_sigma_tau=True)

    assert mpu_w1.q <= mpu_naive.q + 1e-6
    assert mpu_w2.q <= mpu_w1.q + 1e-6
    assert mpu_full.q <= mpu_w2.q + 1e-6


def test_generalmpu_boundary_sigma_tau_site_tensors_are_isometries(twisted_5site):
    """The sigma/tau site tensors are isometries to machine precision."""
    tensors, l_in, r_in = twisted_5site
    mpu = GeneralMPU(tensors, l_in, r_in, boundary_sigma_tau=True)

    for k, V_tn in enumerate(mpu._site_tensors):
        assert _isometry_defect(V_tn) < 1e-8, f"site {k} tensor is not an isometry"


def test_generalmpu_boundary_sigma_tau_synthesizes_correctly(twisted_2site):
    """The sigma/tau circuit equals the dense unitary."""
    tensors, l_in, r_in, U_ref = twisted_2site
    mpu = GeneralMPU(tensors, l_in, r_in, boundary_sigma_tau=True)
    _assert_circuit_matches_reference(mpu, U_ref)


def test_generalmpu_boundary_sigma_tau_composes_with_boundary_opt_p(twisted_5site):
    """Combining sigma/tau with partial optimization is no worse than sigma/tau alone."""
    tensors, l_in, r_in = twisted_5site
    mpu_st = GeneralMPU(tensors, l_in, r_in, boundary_sigma_tau=True)
    mpu_both = GeneralMPU(
        tensors,
        l_in,
        r_in,
        boundary_sigma_tau=True,
        boundary_opt_p=0,
        boundary_opt_n_restarts=4,
    )

    assert mpu_both.q <= mpu_st.q + 1e-6


@pytest.mark.slow
def test_boundary_opt_p_never_accepts_a_non_isometric_site():
    """Optimized boundary windows are accepted only if the interior site is isometric."""
    import warnings

    from mpu_generators import MPUGenerator
    from TreeMPU.checks import (
        dense_unitary_from_tensors,
        phase_aligned_error,
        reconstruct_circuit_unitary,
    )

    t, l, r = MPUGenerator(seed=2).fusion(d=2, bond_dim=2, n_sites=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = GeneralMPU(t, l, r, boundary_opt_p=1, boundary_opt_n_restarts=3)
        qc, sys_regs, *_ = g.synthesize()
    U, leak = reconstruct_circuit_unitary(qc, sys_regs)
    assert phase_aligned_error(U, dense_unitary_from_tensors(t, l, r)) < 1e-10
    assert leak < 1e-10


def test_chain_sigma_tau_uses_the_input_leg():
    """The sigma/tau optimizer acts on the input leg of the site tensors.

    A chain containing a cyclic shift therefore keeps isometric sites.
    """
    from mpu_generators import build_family

    d, chi = 2, 4
    S = np.zeros((d, d, chi, chi), complex)
    for a in range(d):
        for i in range(d):
            for o in range(d):
                S[i, o, a * d + i, a * d + o] = 1
    lS = np.zeros(chi, complex)
    for s in range(d):
        lS[s * d + s] = 1
    t, lb, rb = build_family("fusion_nested_flag", d=2, bond_dim=2, seed=0)
    A = np.einsum("imab,mocd->ioacbd", t[0], S).reshape(2, 2, 8, 8)
    mpu = GeneralMPU(
        tensors_A=[A] * 4, l_vec=np.kron(lb, lS), r_vec=np.kron(rb, lS), boundary_sigma_tau=True
    )
    assert "sigma_tau_optimized" in mpu.boundary_source
    for k in range(4):
        V = mpu._site_isometry_tensor(k)
        p_in = [i for i in V.inds if i.startswith("p_in")]
        M = V.to_dense([i for i in V.inds if i not in p_in], p_in)
        assert np.linalg.norm(M.conj().T @ M - np.eye(M.shape[1])) < 1e-10
    mpu.synthesize()
