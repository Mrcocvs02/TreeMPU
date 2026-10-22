"""Invariant tests for the optimized circuit-synthesis pipeline.

Each test targets an invariant that the optimizations rely on and that a
generic coverage test would not exercise: the padded system register
(physical dimension ``d`` not a power of two together with bond dimension
``D > 1``), the number of columns the circuit depends on, the agreement of
the analytic cost model with the circuit actually built, the isometry
condition on the installed boundary pair, and the fallback routes that
production inputs do not take. References are independent of the
implementation under test: dense matrices contracted from the tensors,
explicit statevector simulation, or closed-form operators. Cases are small
to keep the file cheap.
"""

from __future__ import annotations

import numpy as np
import pytest

from TreeMPU.MPU import (
    CircuitDecomposition as CD,
)
from TreeMPU.MPU import (
    UniformMPU,
)

# --- helpers ---


def cz_mpu():
    """Return the CZ-interaction MPU tensor with boundaries, a valid MPU at D = 2."""
    A_00 = np.array([[1, 0], [0, 0]], dtype=complex)
    A_11 = np.array([[1, 0], [0, 1]], dtype=complex)
    Z = np.zeros((2, 2), dtype=complex)
    A = np.array([[A_00, Z], [Z, A_11]], dtype=complex)
    l = np.array([1.0, 1.0], dtype=complex)
    r = np.array([1.0, -2.0], dtype=complex)
    return A, l, r


def padded_mpu(d: int, D: int):
    """Return a generated MPU tensor and boundaries with both ``d`` and ``D`` above 1."""
    from mpu_generators import generate_random_mpu

    tensors, l, r = generate_random_mpu(n_sites=3, d=d, bond_dim=D, mode="branching", seed=0)
    return np.asarray(tensors[0]), np.asarray(l), np.asarray(r)


def identity_mpu(d: int, D: int = 1):
    """Return the identity MPO at physical dimension ``d``, unitary by construction."""
    A = np.zeros((d, d, D, D), dtype=complex)
    for i in range(d):
        A[i, i, 0, 0] = 1.0
    l = np.zeros(D, dtype=complex)
    r = np.zeros(D, dtype=complex)
    l[0] = r[0] = 1.0
    return A, l, r


def goaa_success_on_physical_inputs(mpu, qc, sys_regs) -> float:
    """Return the worst-case ancilla-|0> weight over the physical product basis."""
    from qiskit.quantum_info import Statevector

    # Only physical inputs are fed: with d not a power of two the register has
    # basis states that no use of the MPU produces, and they measure another quantity.
    d = mpu._d
    sys_idx = [qc.find_bit(q).index for reg in sys_regs for q in reg]
    n_sys = len(sys_idx)
    worst = 1.0
    for a in range(d):
        for b in range(d):
            psi = np.zeros(1 << qc.num_qubits, dtype=complex)
            idx = 0
            for bit, q in enumerate(sys_regs[0]):
                if (a >> bit) & 1:
                    idx |= 1 << qc.find_bit(q).index
            for bit, q in enumerate(sys_regs[1]):
                if (b >> bit) & 1:
                    idx |= 1 << qc.find_bit(q).index
            psi[idx] = 1.0
            sv = Statevector(psi).evolve(qc).data
            p = 0.0
            for s in range(1 << n_sys):
                full = 0
                for i, q in enumerate(sys_idx):
                    if (s >> i) & 1:
                        full |= 1 << q
                p += abs(sv[full]) ** 2
            worst = min(worst, p)
    return worst


# --- the padded system register ---


@pytest.mark.parametrize("d", [3, 5])
def test_goaa_exact_with_padded_system_register(d):
    """GOAA succeeds with certainty for d not a power of two."""
    A, l, r = padded_mpu(d, 2)
    np.random.seed(0)
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=2, light=True)
    qc, sys_regs, _, _, _ = mpu.synthesize()
    p = goaa_success_on_physical_inputs(mpu, qc, sys_regs)
    assert p > 1 - 1e-8, f"GOAA not exact at d={d}: p={p}"


def test_row_padding_places_the_block_in_the_padded_multi_index():
    """The dilated isometry matrix agrees with the tensor at the padded strides.

    It is zero in the padded slots.
    """
    A, l, r = padded_mpu(3, 2)  # D > 1 is needed for a padded axis with entries to its right
    np.random.seed(0)
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=3, light=True)

    V = mpu._site_isometry_tensor(1)
    nb = mpu._bond_qubit_count(0)
    nsys = int(np.ceil(np.log2(mpu._d)))
    M = CD._build_isometry_unitary(
        V,
        label="V",
        D_sys_pad=2**nsys,
        D_bond_pad_L=(2**nb) if "ext_L" in V.inds else None,
        D_bond_pad_R=(2**nb) if "ext_R" in V.inds else None,
        tol=mpu._isometry_tol(),
    )

    out_inds = [t for t in ["ext_L", "ext_R"] if t in V.inds] + sorted(
        i for i in V.inds if i.startswith("p_out")
    )
    in_inds = sorted(i for i in V.inds if i.startswith("p_in"))
    raw = V.to_dense(out_inds, in_inds)
    raw_dims = [V.ind_size(i) for i in out_inds]
    pad_dims = [(2**nb) if i.startswith("ext") else (2**nsys) for i in out_inds]

    got = M[:, : raw.shape[1]].reshape(tuple(pad_dims) + (raw.shape[1],))
    want = raw.reshape(tuple(raw_dims) + (raw.shape[1],))
    sl = tuple(slice(0, n) for n in raw_dims)
    assert np.max(np.abs(got[sl] - want)) < 1e-12, (
        "the physical block is not at its padded multi-index"
    )
    # and nothing may have leaked into the padded slots
    mask = np.ones(tuple(pad_dims), dtype=bool)
    mask[sl] = False
    assert np.max(np.abs(got[mask])) < 1e-12, "the dilation put amplitude in the padded slots"


# --- how many columns the circuit actually depends on ---


def test_site_gate_reproduces_exactly_the_columns_it_promises():
    """The dispatched site gate matches the isometry on its first d_pad columns."""
    from qiskit.quantum_info import Operator

    for d in (2, 4):
        A, l, r = identity_mpu(d, 1)
        np.random.seed(0)
        mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=3, light=True)
        nb = mpu._bond_qubit_count(0)
        nsys = int(np.ceil(np.log2(d)))
        V = mpu._site_isometry_tensor(1)
        V_mat = CD._build_isometry_unitary(
            V,
            label="V",
            D_sys_pad=2**nsys,
            D_bond_pad_L=(2**nb) if "ext_L" in V.inds else None,
            D_bond_pad_R=(2**nb) if "ext_R" in V.inds else None,
            tol=mpu._isometry_tol(),
        )
        d_pad = 2**nsys
        gate = CD._site_isometry_gate(V_mat, d_pad, "V")
        got = np.asarray(Operator(gate).data)[:, :d_pad]
        err = float(np.max(np.abs(got - V_mat[:, :d_pad])))
        assert err < 1e-8, f"d={d}: dispatcher gate differs by {err}"


def test_dispatcher_rejects_a_candidate_that_does_not_match():
    """The dispatcher falls back to a gate reproducing a generic unitary's columns."""
    from qiskit.quantum_info import Operator

    rng = np.random.default_rng(0)
    z = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    Q, _ = np.linalg.qr(z)
    gate = CD._site_isometry_gate(Q, 4, "V")
    got = np.asarray(Operator(gate).data)[:, :4]
    assert np.max(np.abs(got - Q[:, :4])) < 1e-8


# --- The analytic model must not drift from the real construction ---


@pytest.mark.parametrize("N", [3, 4, 5])
def test_analytic_model_matches_the_built_map(N):
    """The analytic cost model equals the built circuit's statistics, odd N too."""
    A, l, r = cz_mpu()
    np.random.seed(0)
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=N, light=True)
    params = mpu.synthesis_cost_params()
    built = mpu.estimate_synthesis_stats()
    analytic = CD.analytic_synthesis_stats(params, N)
    for k in ("depth", "gate_count", "num_qubits"):
        assert built[k] == analytic[k], (
            f"N={N}: {k} {built[k]} (estimate) != {analytic[k]} (analytic)"
        )


# --- Invariants the optimizations rely on ---


def test_q_is_never_below_the_bond_dimension():
    """The conditioning number q is at least D for the canonical uniform boundaries."""
    cases = [(2, 2, cz_mpu())]
    cases += [(d, 1, identity_mpu(d, 1)) for d in (3, 4)]
    for d, D, (A, l, r) in cases:
        np.random.seed(0)
        mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=3, light=True)
        assert mpu.q_unif >= D - 1e-9, f"d={d}, D={D}: q={mpu.q_unif} is below the floor D={D}"


def test_content_cache_cannot_confuse_two_different_matrices():
    """The leaf cache key separates different matrices and joins identical ones."""
    from qiskit.circuit.library import UnitaryGate

    rng = np.random.default_rng(0)
    z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    Q, _ = np.linalg.qr(z)
    g1 = UnitaryGate(Q, check_input=False)
    g2 = UnitaryGate(Q.copy(), check_input=False)
    z2 = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    Q2, _ = np.linalg.qr(z2)
    g3 = UnitaryGate(Q2, check_input=False)

    k1, k2, k3 = CD._content_key(g1), CD._content_key(g2), CD._content_key(g3)
    assert k1 is not None and k1 == k2, "same matrix, different key"
    assert k1 != k3, "different matrices collided on the same key"

    cache: dict = {}
    s1 = CD._leaf_stats(g1, cache)
    s2 = CD._leaf_stats(g2, cache)
    assert s1 == s2, "cache returned different stats for identical content"


def test_construction_is_reproducible_under_a_fixed_seed():
    """Two objects built under the same NumPy seed have identical leaf costs."""
    A, l, r = cz_mpu()
    leaves = []
    for _ in range(2):
        np.random.seed(7)
        mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=3, light=True)
        leaves.append(mpu.synthesis_cost_params()["site"]["bulk"])
    assert leaves[0] == leaves[1], f"same seed gave different leaves: {leaves}"


@pytest.mark.parametrize("n_qubits,n_dirty", [(4, 1), (5, 2), (6, 3)])
def test_borrowed_ancilla_reflection_returns_the_ancillas_untouched(n_qubits, n_dirty):
    """The reflection about |0> with borrowed ancillas equals the reference operator.

    The borrowed ancillas are left unchanged.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    gate = CD._make_reflection_zero(n_qubits, n_dirty)
    total = n_qubits + n_dirty
    qc = QuantumCircuit(total)
    qc.append(gate, range(total))
    got = np.asarray(Operator(qc).data)

    ref0 = -np.eye(2**n_qubits, dtype=complex)
    ref0[0, 0] = 1.0
    want = np.kron(np.eye(2**n_dirty, dtype=complex), ref0)
    assert np.max(np.abs(got - want)) < 1e-9, (
        f"n={n_qubits}, dirty={n_dirty}: reflection is not "
        f"(2|0><0| - I) (x) I on the borrowed qubits"
    )


# --- fallback routes ---


def test_per_term_fallback_builds_the_same_mpu_as_the_closed_form():
    """The per-term fallback for the controlled W implements the same MPU."""
    from qiskit.quantum_info import Statevector

    import TreeMPU.MPU as MOD

    A, l, r = cz_mpu()

    def action(disable_closed_form: bool):
        """Return the 2-site system action of the circuit as a dense matrix.

        With ``disable_closed_form`` the closed-form controlled-W gates are
        replaced by the per-term fallback.
        """
        orig = MOD.CircuitDecomposition._build_closed_wctrl_gates
        if disable_closed_form:
            MOD.CircuitDecomposition._build_closed_wctrl_gates = staticmethod(lambda *a, **k: None)
        try:
            np.random.seed(0)
            mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=2, light=True)
            qc, sys_regs, _, _, _ = mpu.synthesize()
        finally:
            MOD.CircuitDecomposition._build_closed_wctrl_gates = orig

        d = mpu._d
        sys_idx = [qc.find_bit(q).index for reg in sys_regs for q in reg]
        n_sys = len(sys_idx)
        cols = []
        for a_ in range(d):
            for b_ in range(d):
                psi = np.zeros(1 << qc.num_qubits, dtype=complex)
                idx = 0
                for bit, q in enumerate(sys_regs[0]):
                    if (a_ >> bit) & 1:
                        idx |= 1 << qc.find_bit(q).index
                for bit, q in enumerate(sys_regs[1]):
                    if (b_ >> bit) & 1:
                        idx |= 1 << qc.find_bit(q).index
                psi[idx] = 1.0
                sv = Statevector(psi).evolve(qc).data
                out = np.zeros(1 << n_sys, dtype=complex)
                for s in range(1 << n_sys):
                    full = 0
                    for i, q in enumerate(sys_idx):
                        if (s >> i) & 1:
                            full |= 1 << q
                    out[s] = sv[full]
                cols.append(out)
        return np.array(cols).T

    # The closed form is disabled by making it return None, its own signal for
    # "structure absent, fall back". The two circuits must implement the same MPU;
    # they may differ off the block, since two block-encodings of one operator
    # have different unitary completions.
    closed = action(False)
    fallback = action(True)
    # global phase is not observable
    k = np.unravel_index(np.argmax(np.abs(closed)), closed.shape)
    ph = closed[k] / fallback[k]
    ph /= abs(ph)
    err = float(np.max(np.abs(fallback * ph - closed)))
    assert err < 1e-8, f"the per-term fallback implements a different MPU: err={err}"


# --- Odd N beyond the smallest case ---


@pytest.mark.parametrize("N", [7, 9])
def test_analytic_model_on_larger_odd_lengths(N):
    """The analytic cost model equals the built circuit for larger odd N."""
    A, l, r = cz_mpu()
    np.random.seed(0)
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=N, light=True)
    params = mpu.synthesis_cost_params()
    built = mpu.estimate_synthesis_stats()
    analytic = CD.analytic_synthesis_stats(params, N)
    for key in ("depth", "gate_count", "num_qubits"):
        assert built[key] == analytic[key], (
            f"N={N}: {key} {built[key]} (estimate) != {analytic[key]} (analytic)"
        )


def test_boundary_operators_satisfy_the_isometry_condition():
    """The installed boundary pair makes every site map an isometry."""
    A, l, r = cz_mpu()
    np.random.seed(0)
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=3, light=True)
    for k in range(mpu._N):
        V = mpu._site_isometry_tensor(k)
        out_inds = [t for t in ["ext_L", "ext_R"] if t in V.inds] + sorted(
            i for i in V.inds if i.startswith("p_out")
        )
        in_inds = sorted(i for i in V.inds if i.startswith("p_in"))
        mat = V.to_dense(out_inds, in_inds)
        gram = mat.conj().T @ mat
        err = float(np.max(np.abs(gram - np.eye(gram.shape[0]))))
        assert err < 1e-8, f"site {k}: isometry defect {err}"
