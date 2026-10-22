"""Tests of the closed-form Mode I boundary ``mode1_exact_boundary`` (q = D).

The closed-form operators are compared with their values ``L = 1/sqrt(D)``,
``R = 1``, and the isometry conditions of the returned pair are re-checked by
``_isometry_residual``, a direct tensor contraction independent of the
function's self-reported residual. Invalid boundary vectors and tensors that
violate (RR) are verified to raise, the ``UniformMPU`` integration is checked
for the local isometries and the fallback warning, and the synthesized circuit
is compared with the dense unitary contracted from the tensors.
"""

import numpy as np
import pytest
import quimb.tensor as qtn
from qiskit.quantum_info import Statevector

from mpu_generators import generate_random_mpu
from TreeMPU import UniformMPU
from TreeMPU.utils import mode1_exact_boundary


def _isometry_residual(L, R, A, l_vec, r_vec):
    """Return the combined bulk and boundary isometry residual of ``(L, R)``."""
    d = A.shape[0]
    Rt = R.T
    LAR = np.einsum("ab,ijbc,cd->ijad", L, A, Rt)
    VdV = np.einsum("ijab,pjab->ip", LAR.conj(), LAR)
    total = np.linalg.norm(VdV - np.eye(d)) ** 2

    Vl = np.einsum("b,ijbc,cd->ijd", l_vec, A, Rt)
    VldVl = np.einsum("ijd,pjd->ip", Vl.conj(), Vl)
    total += np.linalg.norm(VldVl - np.eye(d)) ** 2

    Vr = np.einsum("ab,ijbc,c->ija", L, A, r_vec)
    VrdVr = np.einsum("ija,pja->ip", Vr.conj(), Vr)
    total += np.linalg.norm(VrdVr - np.eye(d)) ** 2

    return float(np.sqrt(total))


@pytest.fixture(scope="module", params=[(2, 2), (3, 2), (4, 3)])
def mode1_uniform_tensor(request):
    """Return a uniform Mode I tensor with boundaries for several (D, d)."""
    D, d = request.param
    tensors, l_vec, r_vec = generate_random_mpu(
        n_blocks=2,
        d=d,
        bond_dim=D,
        mode="branching",
        uniform=True,
        blocking="auto",
        seed=D * 10 + d,
    )
    return tensors[0], l_vec, r_vec, D, d


def test_mode1_exact_boundary_reaches_q_equals_D_exactly(mode1_uniform_tensor):
    """The closed-form boundary has q = D and satisfies the isometry conditions."""
    A, l_vec, r_vec, D, d = mode1_uniform_tensor
    L, R, q, iso_residual = mode1_exact_boundary(A, l_vec, r_vec)

    assert q == float(D)
    assert iso_residual < 1e-8
    # Independent of the function's self-reported residual.
    assert _isometry_residual(L, R, A, l_vec, r_vec) < 1e-8


def test_mode1_exact_boundary_matches_closed_form_values(mode1_uniform_tensor):
    """The returned operators are L = 1/sqrt(D) and R = 1."""
    A, l_vec, r_vec, D, d = mode1_uniform_tensor
    L, R, _q, _iso = mode1_exact_boundary(A, l_vec, r_vec)
    np.testing.assert_allclose(L, np.eye(D) / np.sqrt(D), atol=1e-12)
    np.testing.assert_allclose(R, np.eye(D), atol=1e-12)


def test_mode1_exact_boundary_rejects_wrong_l_vec(mode1_uniform_tensor):
    """A left boundary that is not a single basis vector is rejected."""
    A, _l_vec, r_vec, D, _d = mode1_uniform_tensor
    bad_l_vec = np.ones(D, dtype=complex) / np.sqrt(D)  # not a single e_v
    with pytest.raises(ValueError, match="single standard basis vector"):
        mode1_exact_boundary(A, bad_l_vec, r_vec)


def test_mode1_exact_boundary_rejects_wrong_r_vec(mode1_uniform_tensor):
    """A right boundary that is not the all-ones vector is rejected."""
    A, l_vec, _r_vec, D, _d = mode1_uniform_tensor
    bad_r_vec = np.zeros(D, dtype=complex)
    bad_r_vec[0] = 1.0  # not the all-ones vector
    with pytest.raises(ValueError, match="all-ones"):
        mode1_exact_boundary(A, l_vec, bad_r_vec)


def test_mode1_exact_boundary_rejects_non_RR_tensor():
    """A tensor that does not satisfy (RR) is rejected by the isometry check."""
    D, d = 3, 2
    rng = np.random.default_rng(0)
    X = rng.standard_normal((d, d, D, D)) + 1j * rng.standard_normal((d, d, D, D))
    l_vec = np.zeros(D, dtype=complex)
    l_vec[0] = 1.0
    r_vec = np.ones(D, dtype=complex)
    with pytest.raises(ValueError, match="row resolution condition"):
        mode1_exact_boundary(X, l_vec, r_vec)


def test_uniformmpu_boundary_mode1_exact_integration(mode1_uniform_tensor):
    """UniformMPU adopts the exact boundary and its local isometries are exact."""
    A, l_vec, r_vec, D, d = mode1_uniform_tensor
    mpu = UniformMPU(A=A, l_vec=l_vec, r_vec=r_vec, N=3, boundary_mode1_exact=True)

    assert mpu.boundary_source == "mode1_exact"
    assert mpu.mode1_exact is True
    assert abs(mpu.q_unif - D) < 1e-8

    # Check the tensors that create_local_isometries() actually builds.
    V_l, V_b, V_r = mpu.create_local_isometries()
    for k, V_tn in enumerate((V_l, V_b, V_r)):
        in_inds = sorted(i for i in V_tn.inds if i.startswith("p_in"))
        out_inds = [i for i in V_tn.inds if i not in in_inds]
        mat = V_tn.to_dense(out_inds, in_inds)
        defect = np.linalg.norm(mat.conj().T @ mat - np.eye(mat.shape[1]))
        assert defect < 1e-8, f"site {k} tensor is not an exact isometry"


def test_uniformmpu_boundary_mode1_exact_beats_canonical(mode1_uniform_tensor):
    """The exact boundary has q no larger than the canonical one."""
    A, l_vec, r_vec, D, d = mode1_uniform_tensor
    mpu_default = UniformMPU(A=A, l_vec=l_vec, r_vec=r_vec, N=3)
    mpu_exact = UniformMPU(A=A, l_vec=l_vec, r_vec=r_vec, N=3, boundary_mode1_exact=True)
    assert mpu_exact.q_unif <= mpu_default.q_unif + 1e-9


def test_uniformmpu_boundary_mode1_exact_false_matches_canonical(mode1_uniform_tensor):
    """boundary_mode1_exact=False reproduces the canonical construction."""
    A, l_vec, r_vec, D, d = mode1_uniform_tensor
    mpu_canon = UniformMPU(A=A, l_vec=l_vec, r_vec=r_vec, N=3)
    mpu_default = UniformMPU(A=A, l_vec=l_vec, r_vec=r_vec, N=3, boundary_mode1_exact=False)
    assert mpu_default.boundary_source == mpu_canon.boundary_source
    np.testing.assert_allclose(mpu_default.L, mpu_canon.L)
    np.testing.assert_allclose(mpu_default.R, mpu_canon.R)


def test_uniformmpu_boundary_mode1_exact_warns_and_falls_back_on_mismatch():
    """A tensor without Mode I structure warns and uses the canonical boundary."""
    A_00 = np.array([[1, 0], [0, 0]], dtype=complex)
    A_01 = np.zeros((2, 2), dtype=complex)
    A_10 = np.zeros((2, 2), dtype=complex)
    A_11 = np.array([[1, 0], [0, 1]], dtype=complex)
    A = np.array([[A_00, A_01], [A_10, A_11]], dtype=complex)
    l_in = np.array([1.0, 1.0], dtype=complex)  # not a single e_v
    r_in = np.array([1.0, -2.0], dtype=complex)  # not all-ones

    with pytest.warns(RuntimeWarning, match="boundary_mode1_exact"):
        mpu = UniformMPU(A=A, l_vec=l_in, r_vec=r_in, N=2, boundary_mode1_exact=True)

    assert mpu.mode1_exact is False
    assert mpu.boundary_source != "mode1_exact"


def test_uniformmpu_boundary_mode1_exact_synthesizes_correctly():
    """The exact-boundary circuit equals the dense unitary with exact GOAA success."""
    D, d, N = 2, 2, 2
    tensors, l_vec, r_vec = generate_random_mpu(
        n_blocks=2,
        d=d,
        bond_dim=D,
        mode="branching",
        uniform=True,
        blocking="auto",
        seed=3,
    )
    A = tensors[0]

    tn = qtn.Tensor(l_vec, inds=["bond_0"])
    for k in range(N):
        A_tn = qtn.Tensor(A, inds=[f"p_in_{k}", f"p_out_{k}", f"bond_{k}", f"bond_{k + 1}"])
        tn = tn @ A_tn
    tn = tn @ qtn.Tensor(r_vec, inds=[f"bond_{N}"])
    out_inds = [f"p_out_{k}" for k in reversed(range(N))]
    in_inds = [f"p_in_{k}" for k in reversed(range(N))]
    U_ref = tn.to_dense(out_inds, in_inds)

    mpu = UniformMPU(A=A, l_vec=l_vec, r_vec=r_vec, N=N, boundary_mode1_exact=True)
    assert mpu.boundary_source == "mode1_exact"

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
    np.testing.assert_allclose(U_rec * phase, U_ref, atol=1e-6)
