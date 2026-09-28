"""Benchmark tests: known reference values of the method.

* MCZ: q = 2, ell = 1, and the documented circuit size (README quickstart).
* Exactness: synthesized circuits reproduce U_N (reconstruction).
* Scaling law: depth(2N)/depth(N) -> 2*ell + 1 for the analytic recursion.
* GOAA thresholds: C'(ell) = csc(pi/(4 ell + 2)).
"""

import numpy as np
import pytest

from TreeMPU import (
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
    d1, d2 = (UniformMPU.analytic_synthesis_stats(params, N)["depth"] for N in (2**13, 2**14))
    assert d2 / d1 == pytest.approx(2 * ell + 1, rel=2e-3)


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
    """Pooling never lowers the depth and never raises the qubit count."""
    A, l, r = _mcz()
    params = UniformMPU(A=A, l_vec=l, r_vec=r, N=4, light=True).synthesis_cost_params()
    for N in (4, 16, 64):
        plain = UniformMPU.analytic_synthesis_stats(params, N)
        for kw in (dict(pool_ancillas=True), dict(pool_levels=1), dict(pool_levels=2)):
            pooled = UniformMPU.analytic_synthesis_stats(params, N, **kw)
            assert pooled["depth"] >= plain["depth"]
            assert pooled["num_qubits"] <= plain["num_qubits"]
            assert pooled["gate_count"] == plain["gate_count"]
