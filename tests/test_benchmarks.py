"""Benchmark tests: known reference values of the method.

* MCZ: q = 2, ell = 1, and the documented circuit size (README quickstart).
* Exactness: synthesized circuits reproduce U_N (reconstruction).
* Scaling law: depth(2N)/depth(N) -> 2*ell + 1 for the analytic recursion.
* GOAA thresholds: C'(ell) = csc(pi/(4 ell + 2)).
"""

import numpy as np
import pytest

from TreeMPU import (
    GeneralMPU,
    UniformMPU,
    dense_unitary_from_tensors,
    phase_aligned_error,
    reconstruct_circuit_unitary,
)
from TreeMPU.MPU import CircuitDecomposition


def _mcz():
    A = np.zeros((2, 2, 2, 2), dtype=complex)
    A[0, 0] = np.diag([1, 0])
    A[1, 1] = np.eye(2)
    return A, np.array([1, 1], dtype=complex), np.array([1, -2], dtype=complex)


def test_mcz_is_the_multicontrolled_z():
    A, l, r = _mcz()
    U = dense_unitary_from_tensors([A] * 3, l, r)
    np.testing.assert_allclose(U, np.diag([1] * 7 + [-1]), atol=1e-14)


def test_mcz_reference_values():
    A, l, r = _mcz()
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=8, light=True)
    assert mpu.q_unif == pytest.approx(2.0, abs=1e-12)
    assert mpu.ell == 1
    qc, *_ = mpu.synthesize()
    assert (qc.num_qubits, qc.depth()) == (36, 134)  # composite gates (README quickstart)


@pytest.mark.parametrize("ell", [1, 2, 3, 4])
def test_goaa_targets(ell):
    C = 1 / np.sin(np.pi / (4 * ell + 2))
    got_ell, phis = CircuitDecomposition._compute_rotation_params(C)
    assert got_ell == ell
    assert all(abs(p) < 1e-12 for p in phis)  # exactly on target: no rescaling


@pytest.mark.slow
def test_mcz_circuit_is_exact_at_N4():
    A, l, r = _mcz()
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=4)
    qc, sys_regs, *_ = mpu.synthesize()
    U_rec, leak = reconstruct_circuit_unitary(qc, sys_regs)
    assert phase_aligned_error(U_rec, dense_unitary_from_tensors([A] * 4, l, r)) < 1e-12
    assert leak < 1e-12


@pytest.mark.slow
@pytest.mark.parametrize(
    "options,ell", [(dict(boundary_sigma_tau=False), 2), (dict(boundary_opt_k=1), 1)]
)
def test_depth_scaling_law(options, ell):
    from mpu_generators import build_family

    t, l, r = build_family("branching_uniform", d=2, bond_dim=2, seed=0)
    mpu = UniformMPU(A=t[0], l_vec=l, r_vec=r, N=8, light=True, **options)
    assert mpu.ell == ell
    params = mpu.synthesis_cost_params()
    d1, d2 = (UniformMPU.analytic_synthesis_stats(params, N)["depth"] for N in (2**10, 2**11))
    # exact circuit depths; the ratio approaches 2 ell + 1 from above
    # (3.011 for ell = 1, 5.0001 for ell = 2 at this N)
    assert d2 / d1 == pytest.approx(2 * ell + 1, rel=5e-3)


@pytest.mark.slow
def test_reconstruction_restricts_to_the_physical_space_for_d3():
    """For d = 3 each site register has 2 qubits; the padding state is not physical."""
    from mpu_generators import UniformMPUGenerator

    t, l, r = UniformMPUGenerator(0).fusion(d=3, bond_dim=3, n_sites=2, skeleton="cyclic")
    mpu = UniformMPU(A=t[0], l_vec=l, r_vec=r, N=2)
    qc, sys_regs, *_ = mpu.synthesize()
    U_rec, leak = reconstruct_circuit_unitary(qc, sys_regs, d=3)
    assert U_rec.shape == (9, 9)
    assert phase_aligned_error(U_rec, dense_unitary_from_tensors(t, l, r)) < 1e-10
    assert leak < 1e-10


def test_pooled_ancillas_trade_qubits_for_depth():
    """Pooling never raises the qubit count and never lowers the depth."""
    A, l, r = _mcz()
    params = UniformMPU(A=A, l_vec=l, r_vec=r, N=4, light=True).synthesis_cost_params()
    for N in (4, 16, 64):
        plain = UniformMPU.analytic_synthesis_stats(params, N)
        for kw in (dict(pool_ancillas=True), dict(pool_levels=1), dict(pool_levels=2)):
            pooled = UniformMPU.analytic_synthesis_stats(params, N, **kw)
            assert pooled["depth"] >= plain["depth"]
            assert pooled["num_qubits"] <= plain["num_qubits"]


@pytest.mark.parametrize("N", [2, 3, 5, 8])
@pytest.mark.parametrize("pool", [dict(), dict(pool_ancillas=True), dict(pool_levels=1)])
def test_estimates_equal_the_built_circuit(N, pool):
    """estimate_synthesis_stats, analytic_synthesis_stats and the flattened
    circuit agree exactly on depth, block count and qubits."""
    from TreeMPU.MPU import CircuitDecomposition as CD

    A, l, r = _mcz()
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=N, light=True)
    qc, *_ = mpu.synthesize(**pool)
    blocks, cache = [], {}
    for ci in qc.data:
        qs = [qc.find_bit(q).index for q in ci.qubits]
        blocks += [tuple(qs[i] for i in b) for b in CD._leaf_blocks(ci.operation, cache)]
    layer: dict = {}
    for b in blocks:
        t = max(layer.get(q, 0) for q in b) + 1
        layer.update({q: t for q in b})
    built = dict(depth=max(layer.values()), gate_count=len(blocks), num_qubits=qc.num_qubits)
    assert mpu.estimate_synthesis_stats(**pool) == built
    params = UniformMPU(A=A, l_vec=l, r_vec=r, N=4, light=True).synthesis_cost_params()
    assert UniformMPU.analytic_synthesis_stats(params, N, **pool) == built


def test_pooled_circuit_is_exact():
    A, l, r = _mcz()
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=4, light=True)
    qc, sys_regs, *_ = mpu.synthesize(pool_ancillas=True)
    assert qc.num_qubits == 14
    U_rec, leak = reconstruct_circuit_unitary(qc, sys_regs)
    assert phase_aligned_error(U_rec, dense_unitary_from_tensors([A] * 4, l, r)) < 1e-12
    assert leak < 1e-12


def test_pool_ancillas_uses_one_register_per_layer():
    A, l, r = _mcz()
    params = UniformMPU(A=A, l_vec=l, r_vec=r, N=4, light=True).synthesis_cost_params()
    for N, qubits in ((4, 14), (8, 28), (16, 54)):
        pooled = UniformMPU.analytic_synthesis_stats(params, N, pool_ancillas=True)
        assert pooled["num_qubits"] == qubits
        assert pooled == UniformMPU.analytic_synthesis_stats(params, N, pool_levels=N)


def _complex_gauge(A, l, r, seed=5):
    """The same MPU in a random complex bond basis: A -> G^-1 A G."""
    D = A.shape[2]
    rng = np.random.default_rng(seed)
    G = np.eye(D) + 0.5 * (rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))
    G_inv = np.linalg.inv(G)
    return np.einsum("ac,ijce,eb->ijab", G_inv, A, G), G.T @ l, G_inv @ r


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("general", [False, True])
def test_canonical_synthesis_is_exact_in_a_complex_bond_gauge(general):
    """Canonical boundaries of a complex (non-real) tensor: the environments
    must be built from the conjugate of the tensor on the contracted leg."""
    from mpu_generators import build_family

    t, l, r = build_family("branching_uniform", d=2, bond_dim=2, seed=3)
    A, l_g, r_g = _complex_gauge(t[0], l, r)
    N = 3
    U = dense_unitary_from_tensors([t[0]] * N, l, r)
    np.testing.assert_allclose(dense_unitary_from_tensors([A] * N, l_g, r_g), U, atol=1e-12)
    if general:
        mpu = GeneralMPU(tensors_A=[A] * N, l_vec=l_g, r_vec=r_g, canonicalize=False)
    else:
        mpu = UniformMPU(A=A, l_vec=l_g, r_vec=r_g, N=N, boundary_sigma_tau=False)
    qc, sys_regs, *_ = mpu.synthesize()
    U_rec, leak = reconstruct_circuit_unitary(qc, sys_regs)
    assert phase_aligned_error(U_rec, U) < 1e-10
    assert leak < 1e-10


@pytest.mark.slow
def test_site_dependent_boundary_goes_below_chi():
    """Documented example: a GeneralMPU boundary with q < chi on one bond."""
    from mpu_generators import MPUGenerator

    t, l, r = MPUGenerator(seed=1).fusion(d=2, bond_dim=2, n_sites=4)
    mpu = GeneralMPU(tensors_A=t, l_vec=l, r_vec=r, boundary_opt_p=1, boundary_opt_n_restarts=3)
    assert min(mpu._q_bonds) == pytest.approx(1.66, abs=0.01)
    qc, sys_regs, *_ = mpu.synthesize()
    U_rec, leak = reconstruct_circuit_unitary(qc, sys_regs)
    assert phase_aligned_error(U_rec, dense_unitary_from_tensors(t, l, r)) < 1e-8
    assert leak < 1e-8


def test_mps_columns_match_the_state_vector_reconstruction():
    from mpu_generators import build_family
    from TreeMPU import mps_circuit_columns

    t, l, r = build_family("branching_uniform", d=2, bond_dim=2, seed=0)
    mpu = UniformMPU(A=t[0], l_vec=l, r_vec=r, N=3, boundary_opt_k=1)
    qc, sys_regs, *_ = mpu.synthesize()
    U_rec, _ = reconstruct_circuit_unitary(qc, sys_regs)
    cols = mps_circuit_columns(qc, sys_regs, range(8))
    C = np.stack([cols[x] for x in range(8)], axis=1)
    assert phase_aligned_error(C, U_rec) < 1e-10
    assert phase_aligned_error(C, dense_unitary_from_tensors([t[0]] * 3, l, r)) < 1e-10
