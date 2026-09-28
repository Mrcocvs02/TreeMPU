"""Adversarial tests for the optimised pipeline.

Every test here targets an invariant that a real defect violated, or that
one of the optimisations silently assumes. The selection is not generic
"more coverage": each case is chosen because something already went wrong
there, or because the surrounding code would not notice if it did.

What the existing suite could not catch, and why:

* every other test uses ``d = 2``, so nothing exercised a padded system
  register -- which is exactly where a bug once hid a circuit that did not
  implement the MPU at all (GOAA success 0.37 instead of 1);
* the scaling study runs through the analytic estimator, which never
  builds a circuit, so a synthesis bug cannot surface there;
* the dispatcher's fidelity guard validates whatever domain it is handed,
  so a wrong ``n_in`` makes it check the wrong thing and pass.

Cases are kept small on purpose: the whole file has to stay affordable
inside a suite that already runs for minutes.
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

# =====================================================================
# helpers
# =====================================================================


def cz_mpu():
    """A genuinely D=2 MPU: the CZ-interaction tensor the suite already uses.

    Needed because the identity MPO is only a valid MPU at D=1 -- at D=2
    its boundary map has rank 1 and ``check_mpo_unitarity`` rejects it for
    failing Assumption 1. That rejection is correct, and using the
    identity at D>1 would be testing an invalid input.
    """
    A_00 = np.array([[1, 0], [0, 0]], dtype=complex)
    A_11 = np.array([[1, 0], [0, 1]], dtype=complex)
    Z = np.zeros((2, 2), dtype=complex)
    A = np.array([[A_00, Z], [Z, A_11]], dtype=complex)
    l = np.array([1.0, 1.0], dtype=complex)
    r = np.array([1.0, -2.0], dtype=complex)
    return A, l, r


def padded_mpu(d: int, D: int):
    """A valid MPU with BOTH d and D genuinely > 1, from the package generator.

    Needed because ``identity_mpu`` at D=1 does NOT exercise the row
    padding: with raw dims (1, 1, d) and padded (1, 1, d_pad), the only
    padded axis is the innermost one, and appending zeros at the end is
    then exactly the same as placing the block -- so the old, broken code
    passes. This was caught by a mutation test: reintroducing the bug left
    the D=1 tests green.

    The bug only shows when a padded axis has something to its right, i.e.
    d not a power of two together with D > 1.
    """
    from mpu_generators import generate_random_mpu

    tensors, l, r = generate_random_mpu(n_sites=3, d=d, bond_dim=D, mode="branching", seed=0)
    return np.asarray(tensors[0]), np.asarray(l), np.asarray(r)


def identity_mpu(d: int, D: int = 1):
    """The identity MPO at physical dimension ``d``: unitary by construction.

    Used instead of a random tensor wherever the test is about plumbing
    rather than about a particular MPU, so that a failure can never be
    blamed on the generator or on an unlucky seed.
    """
    A = np.zeros((d, d, D, D), dtype=complex)
    for i in range(d):
        A[i, i, 0, 0] = 1.0
    l = np.zeros(D, dtype=complex)
    r = np.zeros(D, dtype=complex)
    l[0] = r[0] = 1.0
    return A, l, r


def goaa_success_on_physical_inputs(mpu, qc, sys_regs) -> float:
    """Worst-case ancilla-|0> weight over the PHYSICAL product basis.

    Restricted to physical inputs deliberately. With ``d`` not a power of
    two the register has basis states no real use of the MPU produces, and
    feeding them measures something else -- that confound is what made the
    first diagnosis of that bug point at the wrong quantity.
    """
    from qiskit.quantum_info import Statevector

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


# =====================================================================
# the padded system register
# =====================================================================


@pytest.mark.parametrize("d", [3, 5])
def test_goaa_exact_with_padded_system_register(d):
    """GOAA must succeed with certainty when d is not a power of two.

    This is the regression test for the padded-register bug. The row padding in
    ``_build_isometry_unitary`` used to append zeros instead of placing the
    block in the padded multi-index, and the two agree only when every
    dimension is already a power of two. Measured failure before the fix:
    0.37 (d=3), 0.59 (d=5), with leakage 1.26 out of the physical output
    subspace -- so not a post-selection cost but a wrong action.
    """
    A, l, r = padded_mpu(d, 2)
    np.random.seed(0)
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=2, light=True)
    qc, sys_regs, _, _, _ = mpu.synthesize()
    p = goaa_success_on_physical_inputs(mpu, qc, sys_regs)
    assert p > 1 - 1e-8, f"GOAA not exact at d={d}: p={p}"


def test_row_padding_places_the_block_in_the_padded_multi_index():
    """The dilated matrix must agree with the tensor at PADDED strides.

    A unit test on the exact line that broke, rather than only on the
    symptom. ``mat``'s row index runs with the tensor's strides (D, D*d);
    the register reads it with (D_pad, D_pad*d_pad). Appending zeros is
    the same thing only when they coincide, so this checks a case where
    they do not: d = 3 -> d_pad = 4.
    """
    A, l, r = padded_mpu(3, 2)  # D>1 is essential, see padded_mpu
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


# =====================================================================
# how many columns the circuit actually depends on
# =====================================================================


def test_site_gate_reproduces_exactly_the_columns_it_promises():
    """The dispatcher's chosen gate must match V on its first d_pad columns.

    The circuit depends on exactly ``d_pad`` columns:
    scrambling from ``d_pad`` on leaves the MPU invariant to 1e-15, and
    one column earlier breaks it. The dispatcher is therefore allowed to
    differ from ``V_mat`` beyond that -- but never before it, and the
    guard that enforces this is the only thing standing between a cheaper
    circuit and a silently wrong one (this branch has been observed to return a
    confidently wrong circuit).
    """
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
    """The fidelity guard must actually reject, not merely exist.

    A past bug was a guard handed the wrong domain: it passed while the
    circuit was wrong. So the guard is exercised directly here by handing
    the dispatcher a matrix whose promised columns cannot be reproduced by
    a cheaper route, and checking the dense fallback is returned rather
    than a mismatching gate.
    """
    from qiskit.quantum_info import Operator

    rng = np.random.default_rng(0)
    z = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    Q, _ = np.linalg.qr(z)
    gate = CD._site_isometry_gate(Q, 4, "V")
    got = np.asarray(Operator(gate).data)[:, :4]
    assert np.max(np.abs(got - Q[:, :4])) < 1e-8


# =====================================================================
# The analytic model must not drift from the real construction
# =====================================================================


@pytest.mark.parametrize("N", [3, 4, 5])
def test_analytic_model_matches_the_built_map(N):
    """``analytic_synthesis_stats`` is only useful while it is exact.

    The whole scaling study is computed from it and never simulates a
    circuit, so a drift here would be invisible everywhere else. Odd N is
    included deliberately: it takes the carry branch of the merge tree,
    which the even cases never reach.
    """
    A, l, r = cz_mpu()
    np.random.seed(0)
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=N, light=True)
    params = mpu.synthesis_cost_params()
    built = mpu.estimate_synthesis_stats()
    analytic = CD.analytic_synthesis_stats(params, N)
    for k in ("depth", "gate_count", "num_qubits"):
        assert built[k] == analytic[k], (
            f"N={N}: {k} {built[k]} (mappa) != {analytic[k]} (analitico)"
        )


# =====================================================================
# Invariants the optimisations rely on
# =====================================================================


def test_q_is_never_below_the_bond_dimension():
    """``q >= D`` for the uniform canonical boundaries built here.

    Not a general bound (only q >= 1 is proved; site-dependent windowed
    boundaries can go below D), but it holds for these canonical cases, so a
    violation here would signal a wrong q or boundary.
    """
    cases = [(2, 2, cz_mpu())]
    cases += [(d, 1, identity_mpu(d, 1)) for d in (3, 4)]
    for d, D, (A, l, r) in cases:
        np.random.seed(0)
        mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=3, light=True)
        assert mpu.q_unif >= D - 1e-9, f"d={d}, D={D}: q={mpu.q_unif} is below the floor D={D}"


def test_content_cache_cannot_confuse_two_different_matrices():
    """The implementation keys the leaf cache on matrix CONTENT.

    A collision would silently attribute one gate's cost to another, and
    the symptom would be wrong numbers rather than an error -- the hardest
    kind to notice. This checks the key separates matrices that differ,
    and joins matrices that do not.
    """
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
    """``UniformMPU.__init__`` calls an UNSEEDED optimiser.

    Two objects built from the same ingredients are therefore different
    MPUs unless the seed is fixed -- measured once as a cnot leaf worth
    57, 58 or 59 gates depending on the construction, which produced a
    false "the cache changes the numbers" conclusion. Any comparison
    between two separately built MPUs is invalid without this.
    """
    A, l, r = cz_mpu()
    leaves = []
    for _ in range(2):
        np.random.seed(7)
        mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=3, light=True)
        leaves.append(mpu.synthesis_cost_params()["site"]["bulk"])
    assert leaves[0] == leaves[1], f"same seed gave different leaves: {leaves}"


@pytest.mark.parametrize("n_qubits,n_dirty", [(4, 1), (5, 2), (6, 3)])
def test_borrowed_ancilla_reflection_returns_the_ancillas_untouched(n_qubits, n_dirty):
    """The implementation borrows qubits that are merely idle, not free.

    They belong to the rest of the circuit, so ``Ref0`` must return them
    bit-for-bit -- not just unentangled but unchanged. If it did not, the
    14x saving would be bought with silent corruption elsewhere.
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


# =====================================================================
# the fallback routes are never exercised in production
# =====================================================================


def test_per_term_fallback_builds_the_same_mpu_as_the_closed_form():
    """The fallback for ``W_ctrl`` is live code that never runs.

    Instrumenting ``_build_closed_wctrl_gates`` across 6 families x
    N in {4,8}: 12 calls, 12 closed form, **zero** fallbacks. So the
    per-term and dense routes are a safety net that has never been
    exercised -- and a safety net that only deploys on inputs outside the
    tested families, i.e. exactly when nothing else is known to work.

    The closed form is disabled here by making it return ``None``, which
    is its own documented signal for "structure absent, fall back". Both
    circuits must implement the same MPU; they are free to differ off the
    block, since two valid block-encodings of the same operator have
    different unitary completions (the criterion used by the construction).
    """
    from qiskit.quantum_info import Statevector

    import TreeMPU.MPU as MOD

    A, l, r = cz_mpu()

    def action(disable_closed_form: bool):
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

    closed = action(False)
    fallback = action(True)
    # global phase is not observable
    k = np.unravel_index(np.argmax(np.abs(closed)), closed.shape)
    ph = closed[k] / fallback[k]
    ph /= abs(ph)
    err = float(np.max(np.abs(fallback * ph - closed)))
    assert err < 1e-8, f"the per-term fallback implements a different MPU: err={err}"


# =====================================================================
# Odd N beyond the smallest case
# =====================================================================


@pytest.mark.parametrize("N", [7, 9])
def test_analytic_model_on_larger_odd_lengths(N):
    """The carry branch of the merge tree, past the smallest odd case.

    N = 3 and 5 already exercise a carry, but only with one leftover
    segment at the bottom layer. Larger odd N carries at several layers,
    and the recursion tracks forward and inverse costs separately there --
    which is where an off-by-one would show up rather than at N = 3.
    """
    A, l, r = cz_mpu()
    np.random.seed(0)
    mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=N, light=True)
    params = mpu.synthesis_cost_params()
    built = mpu.estimate_synthesis_stats()
    analytic = CD.analytic_synthesis_stats(params, N)
    for key in ("depth", "gate_count", "num_qubits"):
        assert built[key] == analytic[key], (
            f"N={N}: {key} {built[key]} (mappa) != {analytic[key]} (analitico)"
        )


def test_boundary_operators_satisfy_the_isometry_condition():
    """``(L, R)`` must make the local map an isometry -- what AF varies.

    The construction moves the boundary pair around the null space of two linear
    systems and gains ell 3->2 by it, but the move is only legitimate
    while the isometry condition holds; AF found infeasible points with
    ``q < D`` and isometry defect up to 9.2, and the whole method rests on
    rejecting those. This pins the invariant for the pair the pipeline
    installs by default, so a regression in the boundary machinery cannot
    pass unnoticed.
    """
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
