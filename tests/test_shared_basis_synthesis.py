"""Synthesis of the shared-basis automata and of Mode II cyclic with ``bond_dim > d``.

The generators of these families are tested in the ``mpu_generators``
repository; here the circuits of :class:`TreeMPU.UniformMPU` are compared
with the dense unitary contracted from the tensors (blocked so that
Assumption 1 holds).
"""

import numpy as np
import pytest

from mpu_generators import UniformMPUGenerator
from TreeMPU import UniformMPU
from TreeMPU.checks import (
    dense_unitary_from_tensors,
    phase_aligned_error,
    reconstruct_circuit_unitary,
)
from TreeMPU.utils import block_mpu_tensors


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_cyclic_bond_dim_above_d_circuit_is_exact(seed):
    """Mode II cyclic with bond dimension 3 on qubits: circuit equals the dense unitary.

    Seeds 0 and 7 have a merging row with entries of order 1e-8 on which
    Qiskit's ``StatePreparation`` is off by O(1); the closed-form ``W_ctrl``
    replaces it by a dense completion (``_CLOSED_V_STATE_TOL``).
    """
    tensors, l, r = UniformMPUGenerator(seed=seed).fusion(
        d=2, bond_dim=3, n_blocks=2, skeleton="cyclic", blocking="auto"
    )
    mpu = UniformMPU(A=tensors[0], l_vec=l, r_vec=r, N=2)
    qc, sys_regs, *_ = mpu.synthesize()
    U, leak = reconstruct_circuit_unitary(qc, sys_regs)
    assert phase_aligned_error(U, dense_unitary_from_tensors(tensors, l, r)) < 1e-10
    assert leak < 1e-10


@pytest.mark.slow
def test_shared_basis_ring_circuit_is_exact():
    """Shared-basis ring with D = 3 on qubits: closed-form Mode I boundary, exact circuit."""
    A, l, r = UniformMPUGenerator(seed=0).shared_basis_automaton(D=3, d=2)
    blocked = block_mpu_tensors([A] * 2, 2)
    mpu = UniformMPU(A=blocked[0], l_vec=l, r_vec=r, N=2, boundary_mode1_exact=True)
    assert mpu.boundary_source == "mode1_exact"
    assert np.isclose(mpu.q_unif, 3.0)
    qc, sys_regs, *_ = mpu.synthesize()
    U, leak = reconstruct_circuit_unitary(qc, sys_regs)
    reference = dense_unitary_from_tensors([blocked[0]] * 2, l, r)
    assert phase_aligned_error(U, reference) < 1e-10
    assert leak < 1e-10


def test_state_preparation_error_detects_a_wrong_gate_and_accepts_a_right_one():
    """``_state_preparation_error`` is zero up to a global phase, O(1) otherwise."""
    from qiskit.circuit.library import StatePreparation, UnitaryGate

    from TreeMPU.MPU import CircuitDecomposition

    rng = np.random.default_rng(0)
    v = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    v /= np.linalg.norm(v)
    good = StatePreparation(v).definition.to_gate()
    assert CircuitDecomposition._state_preparation_error(good, v) < 1e-12
    assert CircuitDecomposition._state_preparation_error(good, np.exp(0.7j) * v) < 1e-12
    wrong = UnitaryGate(np.eye(8))
    assert CircuitDecomposition._state_preparation_error(wrong, v) > 0.5


@pytest.mark.parametrize("n", [3, 4])
def test_dense_state_completion_gate_prepares_the_target_exactly(n):
    """The dense completion has ``target`` as its first column, also for tiny entries."""
    from TreeMPU.MPU import CircuitDecomposition

    rng = np.random.default_rng(n)
    v = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    v[rng.random(2**n) < 0.4] *= 1e-9  # the regime in which StatePreparation fails
    v /= np.linalg.norm(v)
    gate = CircuitDecomposition._dense_state_completion_gate(v, "V")
    assert CircuitDecomposition._state_preparation_error(gate, v) < 1e-12
