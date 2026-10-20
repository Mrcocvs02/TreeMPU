r"""Exact quantum-circuit synthesis of open-boundary Matrix Product Unitaries.

Implements the tree decomposition of Styliaris, Trivedi and Cirac
(PRL 135, 260602) and its extensions developed in the thesis (chapters
"MPU decomposition" and "Analytical results"):

1. the chain is cut into single-site isometries
   :math:`V_n = L\,A\,R` using boundary operators :math:`L, R`;
2. neighbouring blocks are merged pairwise along a binary tree; each merge
   applies a block encoding (LCU) of the rank-one merging operator
   :math:`M` with 1-norm :math:`q=\|R^{-\top}L^{-1}\|_F`;
3. :math:`\ell` rounds of generalized oblivious amplitude amplification
   (GOAA) make each merge exact, with :math:`\ell` the smallest integer with
   :math:`q\le\csc(\pi/(4\ell+2))`; the depth then scales as
   :math:`N^{\log_2(2\ell+1)}`.

Classes
-------
:class:`UniformMPU`
    Translation-invariant MPU (one bulk tensor), several boundary strategies.
:class:`GeneralMPU`
    Site-dependent MPU (one tensor per site), per-bond boundaries.
:class:`CircuitDecomposition`
    Shared tree/GOAA engine and resource estimators.

Conventions
-----------
Site tensors ``A[p_in, p_out, bond_L, bond_R]``; boundary operators
``L[ext, bond]``, ``R[ext, bond]``; Qiskit little-endian qubit order with
site 0 as the least significant system register. The synthesized circuit
equals :math:`U_N` up to a global phase.

References
----------
Unqualified numbers in this module (Eq. A33, Eq. 26, Lemma 1, Definition 3,
Theorem 1', ...) refer to G. Styliaris, R. Trivedi, J. I. Cirac, "Quantum
circuits for matrix-product unitaries", PRL 135, 260602 (2025) and its
supplemental material; results of the thesis are cited by LaTeX label
(e.g. ``thm:schmidt_bound``).
"""

from __future__ import annotations

import hashlib
import warnings
from abc import ABC

import numpy as np
import quimb.tensor as qtn
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import (
    CRZGate,
    Isometry,
    RZGate,
    StatePreparation,
    UnitaryGate,
)
from qiskit.quantum_info import Operator
from qiskit.synthesis import (
    qs_decomposition,
    synth_mcx_1_dirty_kg24,
    synth_mcx_n_dirty_i15,
)
from scipy.linalg import qr

from .checks import (
    check_assumption_1,
    check_chain_unitarity,
    check_full_rank,
    check_mpo_unitarity,
    verify_lcu,
    verify_merging_unitary,
)
from .utils import (
    compute_chain_boundary_operators,
    compute_lcu_pad_indices,
    conditioning_number,
    dilate_isometry_to_unitary,
    get_merging_operator,
    get_mpo_site_tensors,
    matrix_sqrt_hermitian,
    mode1_exact_boundary,
    optimize_boundary_operators,
    optimize_chain_sigma_tau,
    optimize_q_unif,
    pad_block_diagonal_operator,
    solve_qD_boundary,
    to_canonical_form,
)

#: Relative tolerance within which the LCU 1-norm C is identified with a GOAA
#: target C'(ell) = 1/sin(pi/(4 ell + 2)) (1, 2, 3.236, ...). Exact boundaries
#: give C either within ~1e-15 of a target or far from it; iterative boundary
#: solvers (qD_general, optimized, optimized_gauge) converge to ~1e-8 and are
#: accepted at an isometry tolerance of 1e-4, so identifying C with the target
#: introduces an amplification error below the precision of the boundary itself.
GOAA_TARGET_RTOL = 1e-7


#: Isometry residual ||V^dag V - I||_F a site may have after a partially
#: optimized boundary is accepted in GeneralMPU (equal to the tolerance the
#: synthesis applies to optimized boundaries, see _isometry_tol).
_OPT_P_SITE_TOL = 1e-4


# Largest (LCU ancilla + rotation + both bond registers) width for which
# _build_U_block_encoding assembles W_ctrl as one dense UnitaryGate instead of
# attaching each LCU term with Gate.control(). 12 qubits is a 4096x4096
# complex matrix (~268 MB), comfortably built and synthesized once, while the
# per-term route at that width would first produce ~4^w gates per term and
# then control-convert every one of them.
DENSE_BLOCK_ENCODING_MAX_QUBITS = 12


class CircuitDecomposition(ABC):  # noqa: B024 (hooks raise NotImplementedError)
    r"""Tree decomposition of an MPU into a quantum circuit (abstract base class).

    Holds the machinery shared by :class:`UniformMPU` and :class:`GeneralMPU`:
    LCU decomposition and block encoding of the merging operator, GOAA
    parameters (:meth:`_compute_rotation_params`), the tree schedule, circuit
    assembly (:meth:`synthesize`) and resource estimation
    (:meth:`estimate_synthesis_stats`, :meth:`synthesis_cost_params`,
    :meth:`analytic_synthesis_stats`). Subclasses provide the site isometries
    and the per-bond merge data.
    """

    def __init__(self, N: int, d: int, l_vec: np.ndarray, r_vec: np.ndarray):
        self._N = N
        self._d = d
        self.l_vec = qtn.Tensor(l_vec, inds=["bond_0"], tags=["l"])
        self.r_vec = qtn.Tensor(r_vec, inds=[f"bond_{N}"], tags=["r"])

    @staticmethod
    def _compute_lcu(M: np.ndarray) -> tuple[list, list, float]:
        """
        Optimal two-term LCU: ``M = (sigma_1/2) (W_+ + W_-)``.

        With ``M = U Sigma V^dag`` and ``Sigma' = Sigma / sigma_1`` (every
        ``sigma'_j <= 1`` by construction), set

            W_pm = U (Sigma' +- i sqrt(I - Sigma'^2)) V^dag.

        Each diagonal entry has modulus ``sigma'^2 + (1 - sigma'^2) = 1``, so
        ``W_pm`` are unitary, and their average is exactly ``U Sigma' V^dag``,
        giving ``sigma_1 (W_+ + W_-)/2 = M`` -- no approximation, and no
        condition on rank, structure or dimension of ``M``.

        Optimality
        ----------
        ``C = sum_i c_i`` is what sets the GOAA cost. For ANY exact LCU
        ``M = sum_i c_i W_i`` with unitary ``W_i``, the triangle inequality
        gives ``||M||_op <= sum_i c_i ||W_i||_op = C``, so ``C >= ||M||_op``
        is an absolute lower bound over all decompositions. This one attains
        it with equality (``C = sigma_1 = ||M||_op``), so none can do better.

        On the merging operators this package builds, though, the gain is NOT
        in ``C``: ``M = |00><1|(R^-1 (x) L^-1)`` is rank one, so a Fourier
        (inverse-DFT) decomposition would also give ``D^2`` coefficients of
        modulus ``sigma_1 / D^2`` summing to exactly ``sigma_1``.

        The decisive gain is the TERM COUNT: ``K`` drops from ``D^2`` to 2,
        so ``n_lcu = ceil(log2 K)`` drops to 1 and ``_build_U_block_encoding``
        makes 2 rather than ``D^2`` calls to ``Gate.control()``. Each such
        call deep-copies the controlled gate's definition, which for
        :math:`D^2` terms exceeds several GB of memory already at D = 5-14.
        """
        U, S, Vh = np.linalg.svd(M, full_matrices=True)
        S = np.real(np.asarray(S).ravel())
        sigma_1 = float(S[0])

        if sigma_1 < 1e-14:
            raise ValueError("Merging operator is numerically zero.")

        S_norm = S / sigma_1
        # sigma'_j = 1 gives sqrt(0); floating point can push it a hair above
        # 1, so clip rather than let sqrt see a negative argument.
        root = np.sqrt(np.clip(1.0 - S_norm**2, 0.0, None))

        # Degenerate case: if every sigma'_j == 1 then M/sigma_1 is already
        # unitary, root vanishes and W_+ == W_- == U V^dag. Emitting the same
        # unitary twice would be correct but wasteful (and leaves a redundant
        # LCU ancilla); collapse to the exact one-term decomposition instead.
        if float(np.max(root)) < 1e-12:
            return [sigma_1], [U @ Vh], sigma_1

        W_plus = U @ np.diag(S_norm + 1j * root) @ Vh
        W_minus = U @ np.diag(S_norm - 1j * root) @ Vh

        coefficients = [sigma_1 / 2.0, sigma_1 / 2.0]
        unitaries = [W_plus, W_minus]

        return coefficients, unitaries, np.sum(coefficients)

    @staticmethod
    def _compute_merging_unitary(coefficients, unitaries, C, build_w_ctrl=True):
        """
        Builds the two components of the merging unitary U = B† W_ctrl B.

        Given an LCU decomposition of the merging operator M = sum_i c_i W_i,
        post-selecting the ancilla on |0> after applying U implements M on any
        state in the image of the isometry V.

        ``build_w_ctrl=False`` returns ``None`` for ``W_ctrl`` and builds only
        ``B``. The circuit path never needs the dense ``W_ctrl`` (its size is
        (ancilla x system)^2, hundreds of GB already at bond dimension ~14);
        only the debug check ``_verify_merging_unitary`` does.

        Parameters
        ----------
        coefficients : array-like, shape (K,)
            Positive real coefficients c_i in M = sum_i c_i W_i.
        unitaries : list of np.ndarray, each shape (dim_system, dim_system)
            Unitary matrices W_i in the LCU decomposition.
        C: float, sum of all coefficients

        Returns
        -------
        B : np.ndarray, shape (dim_ancilla, dim_ancilla)
            Ancilla state-preparation unitary, B|0> = C^(-1/2) sum_i sqrt(c_i) |i>.
        W_ctrl : np.ndarray, shape (dim_system * dim_ancilla, dim_system * dim_ancilla)
            Controlled unitary W_ctrl = sum_i (W_i)_S ⊗ |i><i|_A.
        """
        K = len(coefficients)
        dim_ancilla = 2 ** int(np.ceil(np.log2(K)))
        dim_system = int(unitaries[0].shape[0])

        # ------------------------------------------------------------------
        # Build B: unitary that prepares the ancilla state (Eq. A33)
        #   B|0>_A = C^(-1/2) * sum_i sqrt(c_i) |i>_A  (normalized: sum_i c_i = C)
        # ------------------------------------------------------------------
        first_col = np.zeros(dim_ancilla, dtype=complex)
        first_col[:K] = np.sqrt(coefficients) / np.sqrt(C)

        M = np.eye(dim_ancilla, dtype=complex)
        M[:, 0] = first_col
        Q, R = qr(M)

        diag_R = np.diag(R)
        phases = np.where(np.abs(diag_R) > 1e-14, diag_R / np.abs(diag_R), 1.0)
        B = Q @ np.diag(phases)

        # ------------------------------------------------------------------
        # Build W_ctrl: controlled unitary (Eq. A34)
        #   W_ctrl = sum_i (W_i)_S ⊗ |i><i|_A
        # For i >= K (unused ancilla states), identity is applied on the system.
        # ------------------------------------------------------------------
        if not build_w_ctrl:
            return B, None

        dim_total = dim_system * dim_ancilla
        W_ctrl = np.zeros((dim_total, dim_total), dtype=complex)
        for i in range(dim_ancilla):
            proj = np.zeros((dim_ancilla, dim_ancilla), dtype=complex)
            proj[i, i] = 1.0
            W_i = (
                np.array(unitaries[i], dtype=complex)
                if i < K
                else np.eye(dim_system, dtype=complex)
            )
            W_ctrl += np.kron(proj, W_i)

        return B, W_ctrl

    @staticmethod
    def _compute_rotation_params(C: float) -> tuple[int, list[float]]:
        r"""
        Choose ``ell`` and the rotation angles that make GOAA exact, i.e. that
        satisfy Eq. (A47) ``(2*ell + 1) * theta = pi/2`` with ``sin(theta) =
        1/C'`` after the rescaling of Eq. (A48).

        The admissible targets are discrete, ``C'(ell) = 1/sin(pi/(4*ell+2))``
        = 1, 2, 3.236, 4.494, ..., and the rescaling can only *increase* the
        LCU 1-norm (``|cos phi| + |sin phi| >= 1``), so from a given ``C`` one
        must reach the next target above it. A single rotation qubit as
        written in Eq. (A48) multiplies the norm by at most ``sqrt(2)``, which
        is enough only when the required factor is at most that. It is not
        always: the ratios between consecutive targets are 2.000, 1.618,
        1.389, 1.281, ..., so the ``ell=1 -> 2`` step (1.618) exceeds
        ``sqrt(2)`` and the corresponding window ``C in (2, 2.288]`` is
        unreachable with one qubit. That window is genuinely occupied --
        canonical form gives ``q_k >= D_k`` (Cauchy-Schwarz on Eq. 26 with
        ``sum_i s_i^2 = 1``), so bond dimension 2 lands there routinely, while
        the ``ell=0 -> 1`` window ``C in (1, sqrt(2)]`` needs ``D_k = 1``,
        where ``q = 1`` exactly and no rescaling is required at all.

        Using ``n`` rotation qubits instead of one multiplies the norm by
        ``prod_j (|cos phi_j| + |sin phi_j|) <= 2^(n/2)``, so taking
        ``n = ceil(2*log2(f))`` for the required factor ``f = C'(ell)/C`` and
        splitting it evenly, ``|cos phi_j| + |sin phi_j| = f^(1/n)``, makes
        every target reachable exactly (a single qubit with the arcsin
        argument clipped to 1 would leave GOAA inexact).

        Returns ``(ell, phis)``; ``len(phis)`` is the number of rotation
        qubits, which is 1 in the common case.
        """
        # Smallest ell whose target C'(ell) reaches C. C is snapped to a target
        # within relative GOAA_TARGET_RTOL, so that values an iterative
        # boundary solver returns as 2 +- 1e-8 do not flip ell or add a
        # rotation qubit with a ~1e-8 angle (see GOAA_TARGET_RTOL).
        ell = 0
        while C > (1.0 / np.sin(np.pi / (4 * ell + 2))) * (1.0 + GOAA_TARGET_RTOL):
            ell += 1

        C_prime = 1.0 / np.sin(np.pi / (4 * ell + 2))
        factor = C_prime / C
        if factor < 1.0 + GOAA_TARGET_RTOL:
            factor = 1.0

        # Smallest number of rotation qubits whose reach 2^(n/2) covers factor.
        n_rot = 1 if factor <= np.sqrt(2) + 1e-12 else int(np.ceil(2.0 * np.log2(factor) - 1e-12))
        per_qubit = factor ** (1.0 / n_rot)
        # cos(phi) + sin(phi) = sqrt(2) sin(phi + pi/4) = per_qubit
        phi = float(np.arcsin(np.clip(per_qubit / np.sqrt(2), -1.0, 1.0)) - np.pi / 4)

        return ell, [phi] * n_rot

    @staticmethod
    def _closed_branch_structure(coef_0: list, phis: list, coef_ref: list) -> list | None:
        """
        Replay ``_build_rotated_lcu``'s bookkeeping to recover, per rotation
        angle, which of the two branches ``[cos(phi) Id, sin(phi) iZ]``
        actually survived its ``mag < 1e-14`` filter.

        Returns ``levels`` (one list of surviving branch indices per angle),
        or ``None`` if the surviving set is not the same for every term at
        some level -- in which case the LCU list is not a clean product over
        (sign) x (levels) and the closed form's index arithmetic would be
        wrong. That cannot happen for the angles ``_compute_rotation_params``
        produces (the filter compares ``|c_i p_j|`` against an absolute
        threshold, and the ``c_i`` are all equal at the first level and stay
        proportional afterwards), but it is checked rather than assumed:
        getting it wrong is silent, and the resulting circuit would be wrong
        only on the families where it matters most.

        ``coef_ref`` is ``goaa_data["coef"]``; the reconstructed coefficient
        list is compared against it elementwise, which is what actually
        pins down the term ORDER (and hence the lcu bit assignment) at
        runtime instead of trusting the derivation.
        """
        coef_p = list(coef_0)
        levels = []
        for phi in phis:
            p = [np.cos(phi), np.sin(phi)]
            surv = None
            new_coef = []
            for c_i in coef_p:
                keep = [j for j, p_j in enumerate(p) if abs(c_i * p_j) >= 1e-14]
                if surv is None:
                    surv = keep
                elif surv != keep:
                    return None
                new_coef.extend(abs(c_i * p[j]) for j in keep)
            if not surv:
                return None
            levels.append(surv)
            coef_p = new_coef

        if len(coef_p) != len(coef_ref):
            return None
        if not np.allclose(coef_p, coef_ref, rtol=1e-9, atol=1e-12):
            return None
        return levels

    @staticmethod
    def _build_closed_wctrl_gates(
        goaa_data: dict,
        D_bond: int,
        D_bond_pad: int,
        n_rot: int,
    ) -> dict | None:
        r"""
        Closed-form circuit for ``W_ctrl``, replacing the per-term and dense
        routes entirely. Returns ``None`` when the required structure is
        absent, so the caller falls back to those existing routes.

        Derivation
        ----------
        The merging operator this package builds is
        ``M = |0...0><1| (R^-1 (x) L^-1)``, hence **rank one with output ket
        exactly** ``|0...0>`` -- verified on every family tested (mcz, cnot,
        diag, subspace, random_d2D2, random_d2D3), and structural, not
        accidental. Writing its SVD as ``M = sigma_1 |e0><v|`` (the gauge
        freedom in the left singular vector is fixed by taking ``U = I``,
        always legal precisely because the output ket is ``e0``), the optimal
        two-term dilation of ``_compute_lcu`` collapses to

            W_pm = D_pm V^dag,   D_pm = Pi_0 +- i (1 - Pi_0),

        with ``Pi_0 = |e0><e0|`` and ``V^dag`` ANY unitary whose first row is
        ``v``. Both are unitary and ``sigma_1 (W_+ + W_-)/2 = sigma_1 Pi_0
        V^dag = M`` exactly, since ``(D_+ + D_-)/2 = Pi_0``.

        Controlling on the sign qubit therefore needs only ONE ``V``:

            sum_s |s><s| (x) W_s = D_ctrl . (1_A (x) V^dag),
            D_ctrl = 1_A (x) Pi_0 + i Z_A (x) (1 - Pi_0),

        which is diagonal in the computational basis (eigenvalue 1 whenever
        the system register is ``|0...0>``, ``i (-1)^a`` otherwise).

        Rotation extension (the K>2 case)
        ---------------------------------
        ``_build_rotated_lcu`` extends every term by
        ``[cos(phi) Id, sin(phi) iZ]`` per angle, the inner loop being the
        fastest-varying index. The whole controlled operator therefore
        factorizes across registers:

            W_ctrl = (sum_s |s><s| (x) W_s)_{lcu_sign, bond}
                     (x)_h (sum_j |j><j| (x) sigma_h^(j) V_j)_{lcu_h, rot_h}

        and each rotation factor is, with ``eps_h = sign(cos phi_h)
        sign(sin phi_h)``,

            G_h = sign(cos phi_h) . [ |0><0| (x) Id + |1><1| (x) eps_h iZ ]
                = sign(cos phi_h) . CRZ(-eps_h pi)

        i.e. ONE controlled rotation plus a global sign -- no dense synthesis
        at all. When only one branch survives the coefficient filter the
        factor is unconditional: identity (a global sign) for ``j=0``, or
        ``RZ(-pi)`` on the rotation qubit for ``j=1``.

        Bit assignment (validated numerically, not assumed)
        --------------------------------------------------
        The term index ``t`` is built sign-first, then one bit per surviving
        level in ``phis`` order, so the sign is ``t``'s MOST significant bit.
        Qiskit's ``UnitaryGate`` on ``list(dummy_lcu)`` makes ``lcu[0]`` the
        LEAST significant bit of ``t`` (same convention the dense route's
        ``kron(W_i, proj)`` relies on), hence

            sign      -> lcu[n_lcu - 1]
            level a   -> lcu[n_lcu - 2 - a]      (a = 0, 1, ... in phis order)
            phis[h]   -> rot[n_rot - 1 - h]

        the last from ``unit[t] = kron(W_s, f_1, ..., f_n)`` with ``f_n``
        least significant against ``term_targets``' rot-first order.
        Validated end to end against the trusted route (block error 3.9e-16
        mcz, 2.3e-16 cnot, 3.3e-15 diag K=4, 1.9e-16 random_d2D2 K=8/n_rot=2),
        and guarded at runtime by ``_closed_branch_structure``'s elementwise
        comparison of the reconstructed coefficient list against
        ``goaa_data["coef"]``.

        Padding
        -------
        ``M_bond``'s raw axes are ``(ext_R, ext_L)`` in C-order (see
        ``compute_lcu_pad_indices``); the padded bond space keeps that order
        with ``D_bond -> D_bond_pad``, so the physical block is placed with
        ``np.ix_(pad_idx, pad_idx)``. ``pad_idx[0] == 0``, so the output ket
        stays ``e0`` and the gauge fix survives padding. Note the resulting
        block acts as ZERO on the padded-away subspace where the per-term
        route leaves identity there; both are valid, the pad subspace is
        never populated by the tree's isometries, and the block-encoding
        condition on the physical subspace is what GOAA needs.

        Returns
        -------
        dict or None
            ``V_dag_gate``/``V_gate``: the shared state-preparation unitary
            and its inverse. ``D_ctrl_gate``/``D_ctrl_dag_gate``: the
            diagonal sign gate, carrying the accumulated global phase so it
            stays inside a leaf gate (the flat schedules have no place for a
            circuit-level ``global_phase``). ``sign_pos``: index into the lcu
            register. ``phase_ctrl``: ``(lcu_pos, rot_pos, eps)`` per active
            level. ``phase_fixed``: ``rot_pos`` per level where only the
            ``iZ`` branch survived.

        Why StatePreparation for V
        --------------------------
        ``V^dag`` is only constrained on its first row, so it can be built as
        the inverse of a state-preparation circuit for ``v.conj()``
        (Moettoenen et al. 2004, O(D) gates) rather than by handing a full
        dense unitary to ``qs_decomposition`` (O(D^2)). Measured leaf-block
        ratio against ``qs_decomposition`` on the same dimension: 6.6x at
        D=8, 11.5x at D=16, 21.6x at D=32 -- the ratio grows with D, which is
        the regime that made the larger families unsynthesizable.
        """
        K = int(goaa_data["K"])
        n_lcu = int(np.ceil(np.log2(K)))
        if n_lcu < 1:
            return None

        phis = list(goaa_data["phis"])
        levels = CircuitDecomposition._closed_branch_structure(
            goaa_data["coef_0"], phis, goaa_data["coef"]
        )
        if levels is None or len(phis) != n_rot:
            return None

        active = [h for h, s in enumerate(levels) if len(s) == 2]
        if K != 2 * 2 ** len(active) or n_lcu != 1 + len(active):
            return None

        coef_0 = goaa_data["coef_0"]
        unit_0 = goaa_data["unit_0"]
        if len(coef_0) != 2:
            return None
        M_bond = coef_0[0] * unit_0[0] + coef_0[1] * unit_0[1]

        D_full_pad = D_bond_pad * D_bond_pad
        if M_bond.shape[0] != D_bond * D_bond:
            return None
        if D_bond_pad != D_bond:
            pad_idx = compute_lcu_pad_indices(D_bond, D_bond_pad, 1)
            M_pad = np.zeros((D_full_pad, D_full_pad), dtype=complex)
            M_pad[np.ix_(pad_idx, pad_idx)] = M_bond
            M_bond = M_pad

        U_svd, S_svd, Vh_svd = np.linalg.svd(M_bond)
        if float(S_svd[0]) < 1e-13:
            return None
        # Gauge fix requires the output ket to be e0; if M is not rank one
        # with that ket, |U[0,0]| < 1 and the closed form does not apply.
        if abs(abs(U_svd[0, 0]) - 1.0) > 1e-8:
            return None
        v_row = Vh_svd[0, :] * (U_svd[0, 0] / abs(U_svd[0, 0]))
        v_row = v_row / np.linalg.norm(v_row)

        n_sys = int(np.log2(D_full_pad))
        V_gate = StatePreparation(v_row.conj()).definition.to_gate(label="V")
        V_dag_gate = V_gate.inverse()
        V_dag_gate.label = "V_dag"

        # Global sign of each rotation factor, folded into D_ctrl below.
        global_phase = 0.0
        phase_ctrl, phase_fixed = [], []
        for a, h in enumerate(active):
            phi = phis[h]
            s_cos = 1.0 if np.cos(phi) >= 0 else -1.0
            s_sin = 1.0 if np.sin(phi) >= 0 else -1.0
            phase_ctrl.append((n_lcu - 2 - a, n_rot - 1 - h, s_cos * s_sin))
            if s_cos < 0:
                global_phase += np.pi
        for h, surv in enumerate(levels):
            if len(surv) == 2:
                continue
            phi, j = phis[h], surv[0]
            sign = (np.cos(phi) if j == 0 else np.sin(phi)) >= 0
            if j == 1:
                phase_fixed.append(n_rot - 1 - h)
            if not sign:
                global_phase += np.pi

        def _d_ctrl(sign: float) -> Gate:
            qc = QuantumCircuit(n_sys + 1, name=f"Dctrl{'p' if sign > 0 else 'm'}")
            anc = n_sys
            qc.rz(-sign * np.pi, anc)
            qc.x(range(n_sys))
            qc.append(RZGate(sign * np.pi).control(n_sys), list(range(n_sys)) + [anc])
            qc.x(range(n_sys))
            qc.global_phase = sign * global_phase
            return qc.to_gate()

        return {
            "V_dag_gate": V_dag_gate,
            "V_gate": V_gate,
            "D_ctrl_gate": _d_ctrl(+1.0),
            "D_ctrl_dag_gate": _d_ctrl(-1.0),
            "sign_pos": n_lcu - 1,
            "phase_ctrl": phase_ctrl,
            "phase_fixed": phase_fixed,
        }

    @staticmethod
    def _build_rotated_lcu(
        coef: list, unit: list, C: float, phis: list[float]
    ) -> tuple[list, list, float]:
        """
        Extend M to M' = M ⊗ (cos(phi_j)*I + i*sin(phi_j)*Z) for each angle in
        ``phis``, one factor per rotation qubit. Makes the Grover exponent
        ``ell`` an integer (Eq. A48, generalized to several qubits so that any
        required norm factor up to ``2**(len(phis)/2)`` is reachable).
        """
        Id = np.eye(2, dtype=complex)
        Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        coef_p, unit_p = list(coef), list(unit)
        for phi in phis:
            phase_unitaries = [Id, 1j * Z]
            phase_coeffs = [np.cos(phi), np.sin(phi)]
            new_coef, new_unit = [], []
            for c_i, W_i in zip(coef_p, unit_p):
                for p_j, V_j in zip(phase_coeffs, phase_unitaries):
                    raw = c_i * p_j
                    mag = np.abs(raw)
                    if mag < 1e-14:
                        continue
                    new_coef.append(float(mag))
                    new_unit.append(np.kron(W_i, (raw / mag) * V_j))
            coef_p, unit_p = new_coef, new_unit

        return coef_p, unit_p, sum(coef_p)

    @staticmethod
    def _make_reflection_zero(n_qubits: int, n_dirty: int = 0) -> Gate:
        """
        Implements 2|0...0><0...0| - I on ``n_qubits``, optionally using
        ``n_dirty`` BORROWED ancillas.

        The circuit is locally transpiled to ['u', 'cx', 'id'] to make the
        MCX decomposition reproducible.

        Borrowed ancillas
        -----------------
        ``n_dirty > 0`` returns a gate on ``n_qubits + n_dirty`` qubits whose
        LAST ``n_dirty`` qubits are *dirty* ancillas: they need not be in
        |0>, they may hold anything, and the construction returns them
        exactly as it found them. So they cost NO space -- the caller routes
        qubits that are already present and idle at that point of the
        schedule (see ``_idle_local_indices``), rather than allocating new
        ones, which would widen every downstream operator.

        ``synth_mcx_n_dirty_i15`` (Iten et al. 2015) wants ``n_ctrl - 2``
        ancillas and is the better of the two; ``synth_mcx_1_dirty_kg24``
        (Khattar-Gidney 2024) wants exactly one and is always affordable
        (the minimum idle count measured anywhere in the tree is 2).
        """

        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")

        n_ctrl = n_qubits - 1
        qc = QuantumCircuit(n_qubits + n_dirty, name=f"Ref0_{n_qubits}")

        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)

        if n_qubits > 1:
            mcx = None
            if n_dirty > 0:
                # Prefer the n-dirty construction when enough borrowed
                # qubits were routed; fall back to the 1-dirty one, which
                # only ever needs a single qubit.
                if n_dirty >= max(n_ctrl - 2, 1):
                    mcx = synth_mcx_n_dirty_i15(n_ctrl)
                elif n_ctrl >= 3:
                    mcx = synth_mcx_1_dirty_kg24(n_ctrl)
            if mcx is not None:
                # synth_* return circuits on (controls | target | ancillas),
                # so the ancilla slots are whatever follows the target.
                n_anc_used = mcx.num_qubits - n_ctrl - 1
                qc.compose(
                    mcx,
                    list(range(n_ctrl))
                    + [n_qubits - 1]
                    + list(range(n_qubits, n_qubits + n_anc_used)),
                    inplace=True,
                )
            else:
                qc.mcx(list(range(n_ctrl)), n_qubits - 1)
        else:
            qc.z(0)

        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))

        # Converts I - 2|0><0| into 2|0><0| - I.
        qc.global_phase = np.pi

        qc = transpile(
            qc,
            basis_gates=["u", "cx", "id"],
            optimization_level=1,
        )

        return qc.to_gate(label=f"Ref0_{n_qubits}")

    @staticmethod
    def _idle_local_indices(n_total: int, used: list[int]) -> list[int]:
        """Local indices of ``range(n_total)`` not touched by ``used``.

        These are the qubits that a reflection can borrow as dirty ancillas:
        the reflection is a single instruction in the schedule, so anything
        outside its own support is idle exactly while it runs, and the
        borrowed-ancilla construction hands them back unchanged.

        Measured availability in the merge tree (mcz and diag, N up to 16):
        2 to 74 idle qubits during Ref0, never zero -- so the one-dirty
        construction is always affordable.
        """
        busy = set(used)
        return [i for i in range(n_total) if i not in busy]

    @staticmethod
    def _borrow_for_reflection(n_total: int, reflect_loc: list[int]) -> tuple[int, list[int]]:
        """``(n_dirty, borrowed_loc)`` for a reflection on ``reflect_loc``.

        Lends the largest useful amount to ``synth_mcx_n_dirty_i15``
        (``n_ctrl - 2``) among the idle qubits available; if that is not
        enough, ``_make_reflection_zero`` falls back on its own to the
        single-dirty-ancilla construction.
        No allocation: these are qubits already present and idle at that
        point in the schedule.
        """
        n_ctrl = len(reflect_loc) - 1
        if n_ctrl < 3:
            return 0, []
        idle = CircuitDecomposition._idle_local_indices(n_total, reflect_loc)
        n_dirty = min(len(idle), max(n_ctrl - 2, 1))
        return n_dirty, idle[:n_dirty]

    @staticmethod
    def _qubit_union(a: list, b: list) -> list:
        """``a + b`` without repeated qubits (order kept). Children share
        history-ancilla qubits when their layer's ancillas are pooled."""
        seen = set(a)
        return a + [q for q in b if q not in seen]

    @staticmethod
    def _canonical_segment_qubits(seg: dict) -> list:
        r"""
        Return the canonical external qubit order of a tree segment.

        Segment order
        -------------
        ``sys | b_ct_R | b_ct_L | b_spectator | history_ancillas``

        * ``sys``: physical qubits of the segment's sites;
        * ``b_ct_R``, ``b_ct_L``: bond registers at the segment's right/left
          edge that will be *contracted* (merged) at the next tree level;
        * ``b_spectator``: open bond registers not touched by the next merge;
        * ``history_ancillas``: LCU and rotation ancillas of the merges
          already performed inside the segment (all in :math:`|0\rangle`
          on the success branch).

        Notes
        -----
        This is bookkeeping order for composed tree gates, not necessarily the
        tensor-axis order used to build local dense matrices.
        """
        return (
            seg["sys"]
            + seg["b_ct_R"]
            + seg["b_ct_L"]
            + seg["b_spectator"]
            + seg["history_ancillas"]
        )

    @staticmethod
    def _build_isometry_unitary(
        V_tn: qtn.Tensor,
        label: str,
        D_sys_pad: int,
        D_bond_pad_L: int | None = None,
        D_bond_pad_R: int | None = None,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """
        Convert a local quimb isometry tensor into a square unitary matrix.

        Parameters
        ----------
        V_tn:
            Quimb tensor representing a local isometry. It may contain physical
            input legs ``p_in_*``, physical output legs ``p_out_*``, and external
            virtual legs ``ext_L`` and/or ``ext_R``.
        label:
            Human-readable label used in error messages.
        D_sys_pad:
            Padded physical dimension, usually a power of two.
        D_bond_pad_L, D_bond_pad_R:
            Padded dimension of the ``ext_L``/``ext_R`` legs, usually a power
            of two. Required exactly when the corresponding leg is present
            (a uniform-bulk MPU has the same value for both; a general MPU's
            left/right bonds may differ since they belong to different cuts).
        tol:
            Isometry-acceptance tolerance forwarded to
            :func:`dilate_isometry_to_unitary`. Boundaries from
            ``optimize_boundary_operators`` only satisfy the isometry
            condition up to the optimizer's penalty-method precision, not
            machine epsilon, so this is wider than the default there for
            optimized (non-canonical) sources.

        Returns
        -------
        np.ndarray
            Square unitary dilation of the isometry.

        Notes
        -----
        The local Qiskit gate is appended to qubits ordered as

            ``sys | ext_R | ext_L``.

        Since Qiskit matrices are little-endian with respect to the target list,
        the dense tensor axes must be ordered as

            ``ext_L | ext_R | sys``.

        Boundary sites only have one of ``ext_L`` or ``ext_R``; this rule still
        gives the correct matrix-axis order.
        """
        sys_inds = sorted(i for i in V_tn.inds if i.startswith("p_out"))
        in_inds = sorted(i for i in V_tn.inds if i.startswith("p_in"))

        # Important: bond first (MSB), sys last (LSB)
        bond_inds = [t for t in ["ext_L", "ext_R"] if t in V_tn.inds]
        bond_pads = {"ext_L": D_bond_pad_L, "ext_R": D_bond_pad_R}

        out_inds = bond_inds + sys_inds

        if not sys_inds:
            raise ValueError(f"{label}: no p_out_* legs found.")
        if not in_inds:
            raise ValueError(f"{label}: no p_in_* legs found.")

        mat = V_tn.to_dense(out_inds, in_inds)

        expected_rows = D_sys_pad
        for b in bond_inds:
            pad = bond_pads[b]
            if pad is None:
                raise ValueError(f"{label}: missing padding for leg '{b}'.")
            expected_rows *= pad

        pad_rows = expected_rows - mat.shape[0]
        if pad_rows < 0:
            raise ValueError(
                f"{label}: dense isometry has {mat.shape[0]} rows, "
                f"but the padded output dimension is {expected_rows}."
            )
        raw_dims = [V_tn.ind_size(i) for i in out_inds]
        pad_dims = [bond_pads[i] if i in bond_pads else D_sys_pad for i in out_inds]
        big = np.zeros(tuple(pad_dims) + (mat.shape[1],), dtype=complex)
        big[tuple(slice(0, r) for r in raw_dims)] = mat.reshape(tuple(raw_dims) + (mat.shape[1],))
        mat_pad = big.reshape(expected_rows, mat.shape[1])

        return dilate_isometry_to_unitary(mat_pad, tol=tol)

    @staticmethod
    def _tree_merge_plan(N: int) -> list[list[int]]:
        """
        Return, for each tree layer, the list of bond indices at which a
        pair-merge occurs.

        A "bond index" ``b`` (0-indexed) refers to the internal MPU bond
        between original sites ``b`` and ``b + 1``. Odd leftover segments are
        carried upward unchanged.

        Parameters
        ----------
        N:
            Number of initial sites.

        Returns
        -------
        list[list[int]]
            ``plan[layer][pair_idx]`` is the bond index merged by the
            ``pair_idx``-th pair at that layer.
        """
        if N <= 0:
            raise ValueError("N must be positive.")

        segments = [(k, k) for k in range(N)]
        plan: list[list[int]] = []

        while len(segments) > 1:
            layer_bonds = []
            new_segments = []

            for i in range(0, len(segments) - 1, 2):
                lo_L, hi_L = segments[i]
                lo_R, hi_R = segments[i + 1]
                layer_bonds.append(hi_L)
                new_segments.append((lo_L, hi_R))

            if len(segments) % 2 == 1:
                new_segments.append(segments[-1])

            plan.append(layer_bonds)
            segments = new_segments

        return plan

    @staticmethod
    def _compute_goaa_bundle(M: np.ndarray) -> dict:
        """
        Build the full LCU + GOAA bundle for a single merging operator ``M``.

        This is the data-only orchestration shared by every merge point,
        whether it is the single system-size-independent operator of a
        uniform-bulk MPU or one of the N-1 site-dependent operators of a
        general MPU.
        """
        coef_0, unit_0, C_0 = CircuitDecomposition._compute_lcu(M)
        ell, phis = CircuitDecomposition._compute_rotation_params(C_0)
        coef, unit, C = CircuitDecomposition._build_rotated_lcu(
            coef_0,
            unit_0,
            C_0,
            phis,
        )
        B_mat, W_ctrl_mat = CircuitDecomposition._compute_merging_unitary(
            coef,
            unit,
            C,
            build_w_ctrl=False,
        )

        return {
            "coef_0": coef_0,
            "unit_0": unit_0,
            "C_0": C_0,
            "coef": coef,
            "unit": unit,
            "C": float(C),
            "ell": int(ell),
            "phis": [float(p) for p in phis],
            "n_rot": len(phis),
            "B_mat": B_mat,
            "W_ctrl_mat": W_ctrl_mat,
            "K": len(unit),
        }

    @staticmethod
    def _build_U_block_encoding(
        goaa_data: dict,
        D_bond: int,
        n_bond_per_site: int,
    ) -> tuple[QuantumCircuit, Gate, Gate, list, list, int, int]:
        """
        Build the block-encoding gate ``U = B† W_ctrl B`` for a merge point,
        together with its dagger, in both circuit/Gate form and "schedule"
        form (see Notes).

        Parameters
        ----------
        goaa_data:
            Bundle produced by :meth:`_compute_goaa_bundle`.
        D_bond:
            Unpadded bond dimension at this merge point.
        n_bond_per_site:
            Number of qubits used to encode one virtual bond.

        Returns
        -------
        U_circ:
            Qiskit circuit implementing ``U``.
        U_gate:
            Gate version of ``U_circ``. Its external signature is
            ``lcu | rot | contracted_bonds``.
        U_dagger_gate:
            Gate implementing ``U†``, built directly (see Notes) rather than
            via ``U_gate.inverse()``.
        u_schedule, u_dagger_schedule:
            The same content as ``U_circ``/the dagger circuit, but as a flat
            ``list[tuple[Gate, tuple[int, ...]]]`` of (leaf gate, local qubit
            indices) pairs over the same ``lcu | rot | contracted_bonds``
            signature, for :meth:`synthesize` (see its docstring).
        n_lcu:
            Number of LCU ancilla qubits.
        n_rot:
            Number of rotation ancilla qubits (``len(phis)`` from
            :meth:`_compute_rotation_params`; usually 1).

        Notes
        -----
        ``W_ctrl = sum_i |i><i|_lcu ⊗ W_i`` is deliberately never assembled as
        one dense operator on the combined (LCU ancilla) x (bond, rotation)
        space: that space grows as the *product* of the ancilla and system
        dimensions, and every one of Qiskit's dense-matrix routes to a
        controlled/padded gate (``UnitaryGate`` construction, ``.control()``
        on a ``UnitaryGate``) re-materializes a matrix of that combined size,
        which becomes prohibitive well before realistic bond dimensions.
        Instead, each term ``W_i`` is decomposed into elementary 1-/2-qubit
        gates individually (:func:`qiskit.synthesis.qs_decomposition`, cost
        polynomial in the *unpadded per-term* dimension only) and attached
        to the circuit controlled on its LCU index. No object in this
        construction ever scales with the ancilla x system product.

        For the same reason, ``U_dagger_gate`` is built directly here rather
        than via ``U_gate.inverse()``: qiskit's generic ``Gate.inverse()`` on
        a composite gate recurses into every one of the K controlled
        sub-gates it contains, and doing that through a gate that was itself
        produced by ``qs_decomposition`` + ``.control()`` is not cheap -- for
        realistic K (tens of LCU terms) it was observed to take upwards of
        ten seconds and multiple GB, dwarfing the cost of building U itself,
        for a single merge point. Since ``W_ctrl`` is block-diagonal (each
        term controlled on a distinct, orthogonal LCU basis state),
        ``W_ctrl† = sum_i |i><i| ⊗ W_i†`` termwise, and ``U† = B† W_ctrl† B``
        (``B`` and ``B†`` swap places, each term is independently
        conjugate-transposed and re-decomposed) -- exactly the same
        per-term, polynomial-cost construction as the forward direction,
        just with ``W_i.conj().T`` in place of ``W_i``.
        """
        B_mat = goaa_data["B_mat"]
        unit = goaa_data["unit"]
        K = goaa_data["K"]

        # Number of rotation qubits is chosen by _compute_rotation_params so
        # that the GOAA target C'(ell) is exactly reachable; it is 1 unless the
        # required norm factor exceeds sqrt(2).
        n_rot = int(goaa_data["n_rot"])
        n_lcu = int(np.ceil(np.log2(K)))

        D_bond_pad = 2**n_bond_per_site

        # B is a STATE PREPARATION, not a generic unitary: it is built by QR
        # from a single constrained first column sqrt(c_i)/sqrt(C) (real and
        # non-negative), so every other column is arbitrary completion --
        # exactly the situation of V^dag in _build_closed_wctrl_gates, and it
        # gets the same treatment. StatePreparation (Moettoenen et al. 2004)
        # only pays for the constrained column.
        #
        # As with W_ctrl's closed form, this is a DIFFERENT unitary
        # completion than the QR one, so U changes outside the encoded
        # block. That is legitimate for the same reason: GOAA only needs
        # <0|U|0> = M/C, which depends on B through B|0> alone.
        #
        # But it is NOT unconditionally cheaper, so the two are BUILT AND
        # COMPARED rather than one being adopted on principle. B is at most
        # dim_ancilla x dim_ancilla with n_lcu <= 3, so synthesizing both and
        # counting is free. Measured (leaf blocks, B + B_dag):
        #
        #   n_lcu <= 2   1 + 1  ->  1 + 1     no difference
        #   n_lcu == 3  59 + 57 ->  3 + 3     110 blocks per merge
        #
        # and end-to-end the unconditional version made cnot WORSE (15864 vs
        # 15702 gates at N=8).
        B_gate = UnitaryGate(B_mat, label="B", check_input=False)
        B_dag_gate = UnitaryGate(B_mat.conj().T, label="B_dag", check_input=False)
        try:
            _prep = StatePreparation(np.asarray(B_mat)[:, 0]).definition.to_gate(label="B")
            _prep_dag = _prep.inverse()
            _prep_dag.label = "B_dag"
            _cache: dict = {}
            _cost_dense = (
                CircuitDecomposition._leaf_stats(B_gate, _cache)[1]
                + CircuitDecomposition._leaf_stats(B_dag_gate, _cache)[1]
            )
            _cost_prep = (
                CircuitDecomposition._leaf_stats(_prep, _cache)[1]
                + CircuitDecomposition._leaf_stats(_prep_dag, _cache)[1]
            )
            if _cost_prep < _cost_dense:
                B_gate, B_dag_gate = _prep, _prep_dag
        except Exception:
            # StatePreparation rejects columns it cannot normalize; the dense
            # unitary is always valid, so keep it rather than fail.
            pass

        dummy_lcu = QuantumRegister(n_lcu, "d_lcu")
        dummy_rot = QuantumRegister(n_rot, "d_rot")
        dummy_bc = QuantumRegister(2 * n_bond_per_site, "d_bc")

        U_circ = QuantumCircuit(dummy_lcu, dummy_rot, dummy_bc, name="U")
        U_dagger_circ = QuantumCircuit(dummy_lcu, dummy_rot, dummy_bc, name="U_dag")

        U_circ.append(B_gate, list(dummy_lcu))
        U_dagger_circ.append(B_gate, list(dummy_lcu))

        # Per-term target order matches the (rot | ext_L | ext_R) bit layout
        # that pad_block_diagonal_operator/compute_lcu_pad_indices assumes
        # (the LCU control is applied separately below via .control()).
        bc_ext_R = list(dummy_bc[:n_bond_per_site])
        bc_ext_L = list(dummy_bc[n_bond_per_site:])
        term_targets = list(dummy_rot) + bc_ext_L + bc_ext_R

        # Integer-index mirror of the same signature, for the schedule form.
        # Must match term_targets' actual (rot | ext_L | ext_R) order above,
        # not dummy_bc's own (ext_R | ext_L) register layout.
        lcu_idx = list(range(n_lcu))
        rot_idx = list(range(n_lcu, n_lcu + n_rot))
        bc_idx = list(range(n_lcu + n_rot, n_lcu + n_rot + 2 * n_bond_per_site))
        bc_ext_R_idx = bc_idx[:n_bond_per_site]
        bc_ext_L_idx = bc_idx[n_bond_per_site:]
        term_targets_idx = rot_idx + bc_ext_L_idx + bc_ext_R_idx

        u_schedule = [(B_gate, tuple(lcu_idx))]
        u_dagger_schedule = [(B_gate, tuple(lcu_idx))]

        # --- Adaptive route for W_ctrl -------------------------------------
        # The per-term route below attaches each LCU term with
        # `Gate.control(n_lcu)`. That call does NOT wrap an opaque gate: it
        # walks the term's already-synthesized decomposition and turns every
        # one of its gates into a multi-controlled one (profiled:
        # _add_control -> mcrz -> _mcsu2_real_diagonal). So its cost scales
        # with the term's own gate count, which for a w-qubit term is
        # ~4^w from qs_decomposition. Measured consequences: 1352 cx vs 236
        # for the same U on mcz (5.7x), and outright failure to build at all
        # for lee_yang (9-qubit terms, ~262k gates each, then controlled).
        #
        # When the whole (ancilla x system) register is small enough to hold
        # a dense matrix, assembling W_ctrl = sum_i |i><i| (x) W_i once and
        # handing that single UnitaryGate to Qiskit is strictly better: one
        # synthesis, no control-conversion pass. Above the threshold the
        # dense matrix itself becomes the problem (it grows as the *product*
        # of the two dimensions), so the per-term route is kept for that
        # regime -- which is exactly what it was written for.
        # --- Closed-form route for W_ctrl (see _build_closed_wctrl_gates)
        # Replaces BOTH the dense and per-term routes below: one state-prep
        # circuit for V (O(D_bond_pad) gates, vs O(D_bond_pad^2) from
        # qs_decomposition on a full dense unitary), one multi-controlled
        # phase gate, and one CRZ per surviving rotation branch -- instead
        # of synthesizing all K terms and then converting each to a
        # controlled gate. Falls through to the existing routes whenever the
        # required structure is absent (M not rank one with output ket e0,
        # or an LCU term list that is not a clean product over sign x
        # levels); the helper returns None and nothing here changes.
        #
        # Measured leaf blocks / depth for U, closed vs per-term:
        #   mcz, cnot (K=2)            146 -> 16,  depth 99 -> 11   (9.1x)
        #
        # Validation criterion: the test is NOT that Operator(U_gate) equals
        # the per-term route's. Two valid block-encodings of the same M/C
        # legitimately differ OUTSIDE the encoded block (they are different
        # unitary completions of the same block). What has to hold is
        # <0|_{lcu,rot} U |0>_{lcu,rot} = SWAP_LR(M) e^{i sum phi} / C, with
        # U unitary -- GOAA needs nothing more. Both factors in that target
        # are easy to miss: the block comes out with ext_L and ext_R
        # exchanged (dummy_bc lists ext_R on the LOW qubits while M_bond's
        # raw axes are (ext_R, ext_L) C-order, see compute_lcu_pad_indices),
        # and each rotation factor contributes e^{i phi} on its rot=0 entry.
        # Both are invisible on mcz/cnot, whose M is nonzero only where
        # i_R == i_L and whose phi is ~1e-16.
        #
        # n_rot can be >= 1 even when K == 2: _compute_rotation_params can
        # return phi ~ -1e-16, whose sin branch _build_rotated_lcu then drops
        # (mag < 1e-14), leaving an identity factor on the rotation qubit.
        closed = CircuitDecomposition._build_closed_wctrl_gates(
            goaa_data, D_bond, D_bond_pad, n_rot
        )
        if closed is not None:
            V_dag_gate = closed["V_dag_gate"]
            V_gate = closed["V_gate"]
            D_ctrl_gate = closed["D_ctrl_gate"]
            D_ctrl_dag_gate = closed["D_ctrl_dag_gate"]
            sign_q = dummy_lcu[closed["sign_pos"]]
            sign_i = lcu_idx[closed["sign_pos"]]
            bc_targets = bc_ext_L + bc_ext_R
            bc_idx_full = bc_ext_L_idx + bc_ext_R_idx

            # Rotation factors: one CRZ(-eps pi) per level where both
            # branches survived, one unconditional RZ(-pi) (= iZ) per level
            # where only the sin branch did. They act on (lcu_h, rot_h)
            # only, so they commute with the D_ctrl/V part and their order
            # relative to it is free.
            rot_fwd, rot_bwd = [], []
            for lcu_pos, rot_pos, eps in closed["phase_ctrl"]:
                pair = [dummy_lcu[lcu_pos], dummy_rot[rot_pos]]
                pair_i = (lcu_idx[lcu_pos], rot_idx[rot_pos])
                rot_fwd.append((CRZGate(-eps * np.pi), pair, pair_i))
                rot_bwd.append((CRZGate(eps * np.pi), pair, pair_i))
            for rot_pos in closed["phase_fixed"]:
                targ = [dummy_rot[rot_pos]]
                targ_i = (rot_idx[rot_pos],)
                rot_fwd.append((RZGate(-np.pi), targ, targ_i))
                rot_bwd.append((RZGate(np.pi), targ, targ_i))

            # W_ctrl = D_ctrl . (1_lcu (x) V^dagger) (x) rotation factors:
            # apply V^dagger to the bond register, THEN D_ctrl (jointly on
            # bond + sign qubit). No second V application: W_ctrl is NOT a
            # V-sandwich, it is D_ctrl acting after a single V^dagger.
            U_circ.append(V_dag_gate, bc_targets)
            U_circ.append(D_ctrl_gate, bc_targets + [sign_q])
            u_schedule += [
                (V_dag_gate, tuple(bc_idx_full)),
                (D_ctrl_gate, tuple(bc_idx_full + [sign_i])),
            ]
            for g, targ, targ_i in rot_fwd:
                U_circ.append(g, targ)
                u_schedule.append((g, targ_i))
            U_circ.append(B_dag_gate, list(dummy_lcu))
            u_schedule.append((B_dag_gate, tuple(lcu_idx)))

            # U^dagger = B^dagger W_ctrl^dagger B (B_gate already applied at
            # the top of both circuits): W_ctrl^dagger = V . D_ctrl^dagger.
            # D_ctrl is diagonal but NOT Hermitian (eigenvalues 1 and
            # i*(-1)^a), so D_ctrl^dagger != D_ctrl -- built directly via
            # RZ(+pi) instead of RZ(-pi) globally (D_ctrl_dag_gate), which
            # also carries the conjugated global phase.
            for g, targ, targ_i in rot_bwd:
                U_dagger_circ.append(g, targ)
                u_dagger_schedule.append((g, targ_i))
            U_dagger_circ.append(D_ctrl_dag_gate, bc_targets + [sign_q])
            U_dagger_circ.append(V_gate, bc_targets)
            U_dagger_circ.append(B_dag_gate, list(dummy_lcu))
            u_dagger_schedule += [
                (D_ctrl_dag_gate, tuple(bc_idx_full + [sign_i])),
                (V_gate, tuple(bc_idx_full)),
                (B_dag_gate, tuple(lcu_idx)),
            ]

            return (
                U_circ,
                U_circ.to_gate(),
                U_dagger_circ.to_gate(),
                u_schedule,
                u_dagger_schedule,
                n_lcu,
                n_rot,
            )

        w_total = n_lcu + n_rot + 2 * n_bond_per_site
        use_dense = w_total <= DENSE_BLOCK_ENCODING_MAX_QUBITS

        if use_dense:
            padded = [
                pad_block_diagonal_operator(W_i, D_bond, D_bond_pad, 2**n_rot) for W_i in unit
            ]
            dim_term = padded[0].shape[0]
            if n_lcu == 0:
                W_ctrl_full = padded[0]
            else:
                dim_anc = 2**n_lcu
                W_ctrl_full = np.zeros((dim_term * dim_anc, dim_term * dim_anc), dtype=complex)
                for i in range(dim_anc):
                    proj = np.zeros((dim_anc, dim_anc), dtype=complex)
                    proj[i, i] = 1.0
                    W_i = padded[i] if i < len(padded) else np.eye(dim_term, dtype=complex)
                    # Qiskit is little-endian over the append() qubit list and
                    # the LCU register is listed first, so the ancilla factor
                    # is the LEAST significant one -- hence kron(term, proj),
                    # not kron(proj, term). Verified against the per-term
                    # route by comparing Operator(U_gate) on every family.
                    W_ctrl_full += np.kron(W_i, proj)

            W_ctrl_gate = UnitaryGate(W_ctrl_full, label="W_ctrl", check_input=False)
            W_ctrl_dag_gate = UnitaryGate(
                W_ctrl_full.conj().T, label="W_ctrl_dag", check_input=False
            )

            dense_targets = list(dummy_lcu) + term_targets
            dense_idx = tuple(lcu_idx + term_targets_idx)
            U_circ.append(W_ctrl_gate, dense_targets)
            U_dagger_circ.append(W_ctrl_dag_gate, dense_targets)
            u_schedule.append((W_ctrl_gate, dense_idx))
            u_dagger_schedule.append((W_ctrl_dag_gate, dense_idx))

            U_circ.append(B_dag_gate, list(dummy_lcu))
            U_dagger_circ.append(B_dag_gate, list(dummy_lcu))
            u_schedule.append((B_dag_gate, tuple(lcu_idx)))
            u_dagger_schedule.append((B_dag_gate, tuple(lcu_idx)))

            return (
                U_circ,
                U_circ.to_gate(),
                U_dagger_circ.to_gate(),
                u_schedule,
                u_dagger_schedule,
                n_lcu,
                n_rot,
            )

        for i, W_i_raw in enumerate(unit):
            W_i_padded = pad_block_diagonal_operator(W_i_raw, D_bond, D_bond_pad, 2**n_rot)
            term_gate = qs_decomposition(W_i_padded).to_gate(label=f"W_{i}")
            controlled_term = term_gate.control(n_lcu, ctrl_state=i)
            U_circ.append(controlled_term, list(dummy_lcu) + term_targets)
            u_schedule.append((controlled_term, tuple(lcu_idx + term_targets_idx)))

            term_dag_gate = qs_decomposition(W_i_padded.conj().T).to_gate(label=f"W_{i}_dag")
            controlled_term_dag = term_dag_gate.control(n_lcu, ctrl_state=i)
            U_dagger_circ.append(controlled_term_dag, list(dummy_lcu) + term_targets)
            u_dagger_schedule.append((controlled_term_dag, tuple(lcu_idx + term_targets_idx)))

        U_circ.append(B_dag_gate, list(dummy_lcu))
        U_dagger_circ.append(B_dag_gate, list(dummy_lcu))
        u_schedule.append((B_dag_gate, tuple(lcu_idx)))
        u_dagger_schedule.append((B_dag_gate, tuple(lcu_idx)))

        return (
            U_circ,
            U_circ.to_gate(),
            U_dagger_circ.to_gate(),
            u_schedule,
            u_dagger_schedule,
            n_lcu,
            n_rot,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    # Subclasses expose their (possibly site-dependent) data through these
    # four methods. Everything below -- segment construction, tree merging,
    # and circuit assembly -- is shared between uniform-bulk and general
    # MPUs and only ever calls into these hooks, never into subclass
    # internals directly.

    def _site_isometry_tensor(self, k: int) -> qtn.Tensor:
        """Return the dressed local isometry tensor V_k for site k (0-indexed)."""
        raise NotImplementedError

    def _bond_qubit_count(self, bond_index: int) -> int:
        """Number of qubits used to encode internal bond ``bond_index``."""
        raise NotImplementedError

    def _bond_dimension(self, bond_index: int) -> int:
        """Unpadded dimension of internal bond ``bond_index``."""
        raise NotImplementedError

    def _merge_bundle(self, bond_index: int) -> dict:
        """
        Return the precomputed merge data for internal bond ``bond_index``:
        a dict with keys ``n_bond_per_site``, ``n_lcu``, ``n_rot``, ``ell``,
        ``U_gate``, ``R_Phi_gate``.
        """
        raise NotImplementedError

    def _isometry_tol(self) -> float:
        """
        Acceptance tolerance for the per-site isometry check
        (``dilate_isometry_to_unitary``). Defaults to machine-precision-tight
        (matching the canonical sigma/tau construction); subclasses whose
        local isometries can come from a numerical optimizer override this
        to a looser value for that case.
        """
        return 1e-10

    # ------------------------------------------------------------------
    # Tree-circuit engine (flat schedule-based; see synthesize's docstring
    # for why it is built this way).
    # ------------------------------------------------------------------

    @staticmethod
    def _remap_schedule(
        schedule: list[tuple[Gate, tuple[int, ...]]], mapping: list[int]
    ) -> list[tuple[Gate, tuple[int, ...]]]:
        """Re-index every entry's local qubits through ``mapping`` (old index
        -> new index), leaving the gate objects themselves untouched."""
        return [(gate, tuple(mapping[q] for q in qs)) for gate, qs in schedule]

    # ------------------------------------------------------------------
    # Segment programs. A segment's content is a list of entries
    #   ("g", gate, local_qubits)            one leaf gate
    #   ("s", child, inverse, local_map)     a whole child segment
    # in program order. Running a segment backwards (its inverse) replays
    # the entries in reverse order, inverting each leaf and flipping each
    # child call. synthesize() expands the program into the circuit;
    # estimate_synthesis_stats() schedules the very same program, so the
    # two agree exactly by construction.
    # ------------------------------------------------------------------

    @staticmethod
    def _remap_program(prog: list, mapping: list[int]) -> list:
        out = []
        for e in prog:
            if e[0] == "g":
                out.append(("g", e[1], tuple(mapping[q] for q in e[2])))
            else:
                out.append(("s", e[1], e[2], tuple(mapping[q] for q in e[3])))
        return out

    def _program_signature(self, prog: list) -> int:
        """Small integer identifying a program up to structure: segments with
        the same leaf gates (by content) on the same local qubits share it,
        so the uniform bulk of a chain is scheduled once per tree layer."""
        table = self.__dict__.setdefault("_sig_table", {})
        key = []
        for e in prog:
            if e[0] == "g":
                ckey = CircuitDecomposition._content_key(e[1])
                key.append(("g", ckey if ckey is not None else id(e[1]), e[2]))
            else:
                key.append(("s", e[1]["sig"], e[2], e[3]))
        return table.setdefault(tuple(key), len(table))

    def _reflection_gate(self, n_qubits: int, n_dirty: int) -> Gate:
        """``_make_reflection_zero`` cached per size (one object per shape)."""
        cache = self.__dict__.setdefault("_ref0_gates", {})
        key = (n_qubits, n_dirty)
        if key not in cache:
            cache[key] = CircuitDecomposition._make_reflection_zero(n_qubits, n_dirty=n_dirty)
        return cache[key]

    def _leaf_inverse(self, gate: Gate) -> Gate:
        """``gate.inverse()``, one object per leaf gate."""
        cache = self.__dict__.setdefault("_inv_gates", {})
        hit = cache.get(id(gate))
        if hit is None:
            hit = cache[id(gate)] = (gate, gate.inverse())
        return hit[1]

    def _expand_program(self, seg: dict, inverse: bool, qmap: list, out: list) -> None:
        """Append ``(gate, qubits)`` for every leaf of ``seg`` (run backwards if ``inverse``)."""
        for e in reversed(seg["prog"]) if inverse else seg["prog"]:
            if e[0] == "g":
                gate = self._leaf_inverse(e[1]) if inverse else e[1]
                out.append((gate, [qmap[i] for i in e[2]]))
            else:
                self._expand_program(e[1], e[2] != inverse, [qmap[i] for i in e[3]], out)

    def _program_depth(self, seg, inverse, start, memo, cache):
        """Exact as-soon-as-possible schedule of ``seg`` over <=2-qubit leaf blocks.

        ``start[i]`` is the layer at which local qubit ``i`` becomes free;
        returns the layers after the segment. This is the greedy per-qubit
        layering of the fully expanded circuit, computed without expanding
        it: the result only depends on the segment's structure and on
        ``start`` up to a common shift, which is memoized.
        """
        base = min(start) if start else 0
        key = (seg["sig"], inverse, tuple(t - base for t in start))
        hit = memo.get(key)
        if hit is not None:
            return [t + base for t in hit]
        f = list(start)
        for e in reversed(seg["prog"]) if inverse else seg["prog"]:
            if e[0] == "g":
                gate = self._leaf_inverse(e[1]) if inverse else e[1]
                for blk in CircuitDecomposition._leaf_blocks(gate, cache):
                    qs = [e[2][i] for i in blk]
                    t = max(f[q] for q in qs) + 1
                    for q in qs:
                        f[q] = t
            else:
                lmap = e[3]
                sub = self._program_depth(e[1], e[2] != inverse, [f[i] for i in lmap], memo, cache)
                for i, t in zip(lmap, sub, strict=True):
                    f[i] = t
        memo[key] = [t - base for t in f]
        return f

    def _program_count(self, seg, inverse, memo, cache) -> int:
        """Number of <=2-qubit leaf blocks of ``seg`` (memoized per structure)."""
        key = (seg["sig"], inverse)
        hit = memo.get(key)
        if hit is not None:
            return hit
        n = 0
        for e in seg["prog"]:
            if e[0] == "g":
                gate = self._leaf_inverse(e[1]) if inverse else e[1]
                n += len(CircuitDecomposition._leaf_blocks(gate, cache))
            else:
                n += self._program_count(e[1], e[2] != inverse, memo, cache)
        memo[key] = n
        return n

    def _make_segment_list(
        self,
        D_sys_pad: int,
        n_sys_per_site: int,
        site_sys_regs: list[QuantumRegister],
        site_bond_regs: list[QuantumRegister],
    ) -> list[dict]:
        """
        Build the initial per-site segment list for the tree-merge synthesis.

        Each segment carries a flat ``"schedule"`` (list of ``(leaf gate,
        local qubit indices)`` pairs) and a scalar ``"global_phase"``,
        materialized into an actual circuit only once, at the very end --
        see :meth:`synthesize`'s docstring for why.
        """
        iso_cache: dict = {}
        N = self._N
        segments: list[dict] = []

        for k in range(N):
            sys_q = list(site_sys_regs[k])
            bond_q = list(site_bond_regs[k])

            if len(sys_q) != n_sys_per_site:
                raise ValueError(
                    f"Site {k}: expected {n_sys_per_site} system qubits, got {len(sys_q)}."
                )

            n_right_k = self._bond_qubit_count(k) if k < N - 1 else 0
            n_left_k = self._bond_qubit_count(k - 1) if k > 0 else 0
            leaf_gate, has_L, has_R = self._site_leaf(k, D_sys_pad, n_left_k, n_right_k, iso_cache)

            if has_R and has_L:
                b_ct_R = bond_q[:n_right_k]
                b_ct_L = bond_q[n_right_k : n_right_k + n_left_k]
            elif has_R:
                b_ct_R = bond_q[:]
                b_ct_L = []
            elif has_L:
                b_ct_R = []
                b_ct_L = bond_q[:]
            else:
                raise ValueError(f"Site {k}: local isometry has neither ext_R nor ext_L.")

            n_q = len(sys_q) + len(bond_q)

            seg = {
                "sys": sys_q,
                "b_ct_R": b_ct_R,
                "b_ct_L": b_ct_L,
                "b_spectator": [],
                "history_ancillas": [],
                "prog": [("g", leaf_gate, tuple(range(n_q)))],
                "global_phase": 0.0,
            }
            seg["sig"] = self._program_signature(seg["prog"])
            seg["v_qubits"] = self._canonical_segment_qubits(seg)

            if len(set(seg["v_qubits"])) != len(seg["v_qubits"]):
                raise ValueError(f"Duplicate qubits in initial segment {k}.")

            segments.append(seg)

        return segments

    def _site_leaf(self, k, D_sys_pad, n_left_k, n_right_k, iso_cache):
        """``(gate, has_L, has_R)``: the leaf gate of site ``k``'s isometry."""
        V_tn = self._site_isometry_tensor(k)
        has_R = "ext_R" in V_tn.inds
        has_L = "ext_L" in V_tn.inds
        V_mat = CircuitDecomposition._build_isometry_unitary(
            V_tn,
            label=f"V_site_{k}",
            D_sys_pad=D_sys_pad,
            D_bond_pad_L=2**n_left_k if has_L else None,
            D_bond_pad_R=2**n_right_k if has_R else None,
            tol=self._isometry_tol(),
        )
        return (
            CircuitDecomposition._site_isometry_gate(V_mat, D_sys_pad, f"V{k}", iso_cache),
            has_L,
            has_R,
        )

    def _merge_layer(
        self,
        segments: list[dict],
        layer_anc_pairs: list[tuple[QuantumRegister, QuantumRegister]],
        layer_bonds: list[int],
        layer: int = 0,
    ) -> list[dict]:
        """
        Merge one layer of the tree: bond contraction, canonical reordering,
        and the pairwise GOAA merge, composing schedules by list
        concatenation and index remapping instead of building nested qiskit
        ``Gate`` objects -- see :meth:`synthesize`'s docstring for why. In
        particular, ``R_Psi = V_pair Ref_0 V_pair†`` never calls
        ``V_pair_gate.inverse()``: it reverses ``pair_schedule`` and inverts
        each (leaf-level, hence cheap) entry individually, which is what
        keeps this cheap no matter how large the tree above it has already
        grown.
        """
        new_segments: list[dict] = []
        pair_idx = 0
        n_seg = len(segments)

        for i in range(0, n_seg - 1, 2):
            seg_L = segments[i]
            seg_R = segments[i + 1]

            if seg_L["v_qubits"] != self._canonical_segment_qubits(seg_L):
                raise ValueError(f"Layer {layer}, pair {pair_idx}: left segment is not canonical.")

            if seg_R["v_qubits"] != self._canonical_segment_qubits(seg_R):
                raise ValueError(f"Layer {layer}, pair {pair_idx}: right segment is not canonical.")

            bond = layer_bonds[pair_idx]
            bundle = self._merge_bundle(bond)

            n_bond_per_site = bundle["n_bond_per_site"]
            n_lcu = bundle["n_lcu"]
            n_rot = bundle["n_rot"]
            ell = bundle["ell"]
            u_schedule = bundle["u_schedule"]
            u_dagger_schedule = bundle["u_dagger_schedule"]
            R_Phi_gate = bundle["R_Phi_gate"]

            # a pooled register may be larger than this merge needs: use its first qubits
            lcu_q = list(layer_anc_pairs[pair_idx][0])[:n_lcu]
            rot_q = list(layer_anc_pairs[pair_idx][1])[:n_rot]

            if len(lcu_q) != n_lcu or len(rot_q) != n_rot:
                raise ValueError(f"Layer {layer}, pair {pair_idx}: ancilla register size mismatch.")

            b_ct_pair = seg_L["b_ct_R"] + seg_R["b_ct_L"]

            if len(b_ct_pair) != 2 * n_bond_per_site:
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: contracted bond has "
                    f"{len(b_ct_pair)} qubits, expected {2 * n_bond_per_site}."
                )

            b_spectator_pair = seg_L["b_spectator"] + seg_R["b_spectator"]
            sys_pair = seg_L["sys"] + seg_R["sys"]

            v_qubits_pair = (
                seg_L["sys"]
                + seg_R["sys"]
                + seg_L["b_ct_R"]
                + seg_R["b_ct_L"]
                + seg_L["b_ct_L"]
                + seg_R["b_ct_R"]
                + seg_L["b_spectator"]
                + seg_R["b_spectator"]
                + self._qubit_union(seg_L["history_ancillas"], seg_R["history_ancillas"])
            )

            if len(set(v_qubits_pair)) != len(v_qubits_pair):
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: duplicate qubits in raw pair support."
                )

            q2local = {q: j for j, q in enumerate(v_qubits_pair)}

            # Called only within this iteration, so capturing the loop variables is intended.
            def to_local(qlist: list, context: str) -> list[int]:  # noqa: B023
                missing = [q for q in qlist if q not in q2local]  # noqa: B023
                if missing:
                    raise ValueError(
                        f"Layer {layer}, pair {pair_idx}, {context}: "  # noqa: B023
                        f"{len(missing)} qubits are not present in v_qubits_pair."
                    )
                return [q2local[q] for q in qlist]  # noqa: B023

            left_local = to_local(seg_L["v_qubits"], "left child")
            right_local = to_local(seg_R["v_qubits"], "right child")
            b_ct_local = to_local(b_ct_pair, "contracted bonds")

            n_v = len(v_qubits_pair)
            n_vm = n_v + n_lcu + n_rot

            # (no separate v_loc needed here: pair_schedule's own indices are
            # already 0..n_v-1 by construction, unlike synthesize()'s Gate
            # form which needs v_loc explicitly to .append(V_pair_gate, ...))
            lcu_loc = list(range(n_v, n_v + n_lcu))
            rot_loc = list(range(n_v + n_lcu, n_vm))

            u_signature_loc = lcu_loc + rot_loc + b_ct_local

            if len(u_signature_loc) != bundle["U_gate"].num_qubits:
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: U signature length "
                    f"{len(u_signature_loc)} does not match U_gate.num_qubits "
                    f"{bundle['U_gate'].num_qubits}."
                )

            if len(set(u_signature_loc)) != len(u_signature_loc):
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: duplicate indices in U signature."
                )

            # Children are referenced, not copied: V_pair is "left, then
            # right" and V_pair^-1 is "right^-1, then left^-1", where a
            # segment's inverse replays its program backwards with every
            # leaf inverted (see _expand_program).
            pair_prog = [
                ("s", seg_L, False, tuple(left_local)),
                ("s", seg_R, False, tuple(right_local)),
            ]
            pair_inv_prog = [
                ("s", seg_R, True, tuple(right_local)),
                ("s", seg_L, True, tuple(left_local)),
            ]

            b_op_for_rpsi = seg_L["b_ct_L"] + seg_R["b_ct_R"] + b_spectator_pair
            b_op_local = to_local(b_op_for_rpsi, "open/spectator bonds")
            reflect_loc = lcu_loc + rot_loc + b_ct_local + b_op_local

            # R_Psi = V_pair Ref_0 V_pair^-1
            n_dirty, borrowed = self._borrow_for_reflection(n_vm, reflect_loc)
            ref0_gate = self._reflection_gate(len(reflect_loc), n_dirty)
            rpsi_prog = (
                pair_inv_prog + [("g", ref0_gate, tuple(reflect_loc + borrowed))] + pair_prog
            )

            u_mapped = [("g", g, qs) for g, qs in self._remap_schedule(u_schedule, u_signature_loc)]
            u_dag_mapped = [
                ("g", g, qs) for g, qs in self._remap_schedule(u_dagger_schedule, u_signature_loc)
            ]

            # G = - U R_Psi U^dagger R_Phi; the global_phase += pi it
            # applies is tracked as a scalar below instead of on a
            # QuantumCircuit.
            g_prog = (
                [("g", R_Phi_gate, tuple(lcu_loc + rot_loc))] + u_dag_mapped + rpsi_prog + u_mapped
            )

            vm_prog = pair_prog + u_mapped + g_prog * int(ell)
            vm_phase = seg_L["global_phase"] + seg_R["global_phase"] + int(ell) * np.pi

            new_seg = {
                "sys": sys_pair,
                "b_ct_R": seg_R["b_ct_R"],
                "b_ct_L": seg_L["b_ct_L"],
                "b_spectator": b_spectator_pair + b_ct_pair,
                "history_ancillas": (
                    self._qubit_union(seg_L["history_ancillas"], seg_R["history_ancillas"])
                    + lcu_q
                    + rot_q
                ),
            }

            canonical = self._canonical_segment_qubits(new_seg)
            new_seg["v_qubits"] = canonical

            if len(set(canonical)) != len(canonical):
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: duplicate qubits in canonical segment."
                )

            raw_order_global = v_qubits_pair + lcu_q + rot_q

            if len(raw_order_global) != len(canonical):
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: raw/canonical size mismatch "
                    f"{len(raw_order_global)} != {len(canonical)}."
                )

            canonical_index = {q: j for j, q in enumerate(canonical)}
            raw_targets_in_canonical = [canonical_index[q] for q in raw_order_global]

            new_seg["prog"] = self._remap_program(vm_prog, raw_targets_in_canonical)
            new_seg["global_phase"] = vm_phase
            new_seg["sig"] = self._program_signature(new_seg["prog"])

            new_segments.append(new_seg)
            pair_idx += 1

        if n_seg % 2 == 1:
            new_segments.append(segments[-1])

        return new_segments

    # ------------------------------------------------------------------
    # Stats-only estimator -- computes (depth, gate_count) without ever
    # materializing a schedule or circuit. See estimate_synthesis_stats's
    # docstring for why this exists alongside synthesize.
    # ------------------------------------------------------------------

    @staticmethod
    def _site_isometry_gate(
        V_mat: np.ndarray, n_in: int, label: str, cache: dict | None = None
    ) -> Gate:
        """The cheapest correct gate for a local site isometry.

        Two candidates are built, their elementary-gate cost is measured, and
        the cheaper one is kept (the content cache makes this a one-off cost
        per distinct matrix):

        1. ``UnitaryGate`` over the full dilated unitary (equivalent to
           Qiskit's ``qs_decomposition`` with its best flags);
        2. ``Isometry`` (Iten et al.) over the ``n_in`` physical columns only,
           so the free completion is never synthesized.

        Qiskit's ``Isometry`` fails on many of these matrices because of an
        exact algebraic degeneracy in its column elimination (not size,
        tolerance or sparsity; raising its ``epsilon`` returns a wrong
        circuit). A random single-qubit unitary on every wire breaks the
        degeneracy and costs one gate per wire to undo. Typical gains on bulk
        isometries (two-qubit blocks, dense -> isometry route):

            subspace   44522 -> 17845 blocks (2.49x), 16812 -> 8802 CX
            rand_d2D3  19076 ->  9263 (2.06x)
            rand_d4D4   4676 ->  2303 (2.03x)
            diag        2690 ->  2274 (1.18x)
            cnot          55 ->    59 (0.93x)   <- dense wins
            mcz           28 ->    59 (0.47x)   <- dense wins

        so the rule is "small ``n_in/dim`` favours the isometry", but it is
        applied by measurement, not by that heuristic.

        The isometry candidate is accepted only if it reproduces the physical
        columns to machine precision; otherwise the dense gate is used.
        """
        V_mat = np.asarray(V_mat)
        dim = V_mat.shape[0]
        dense = UnitaryGate(V_mat, label=label, check_input=False)
        cache = {} if cache is None else cache
        if n_in <= 0 or n_in >= dim:
            return dense

        try:
            n_out = int(round(np.log2(dim)))
            k = 1 << int(np.ceil(np.log2(n_in)))
            if k >= dim:
                return dense
            cols = V_mat[:, :k]

            # Deterministic preconditioner: seeded from the matrix content,
            # so the choice is reproducible across runs and across the two
            # places that build the same site (the estimator and the
            # synthesiser must not disagree).
            seed = int(hashlib.sha1(np.ascontiguousarray(V_mat).tobytes()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            us = []
            P = np.array([[1.0]], dtype=complex)
            for _ in range(n_out):
                z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
                u, _ = np.linalg.qr(z)
                us.append(u)
                P = np.kron(P, u)

            qc = QuantumCircuit(n_out)
            qc.append(Isometry(P @ cols, 0, 0), range(n_out))
            # qiskit's qubit 0 is the LEAST significant index, so the
            # tensor factors are undone in reverse wire order.
            for j, u in enumerate(us):
                qc.append(UnitaryGate(u.conj().T, check_input=False), [n_out - 1 - j])
            qc = transpile(qc, basis_gates=["u", "cx", "id"], optimization_level=1)
            iso_gate = qc.to_gate(label=label)

            got = np.asarray(Operator(qc).data)[:, :n_in]
            err = float(np.max(np.abs(got - V_mat[:, :n_in])))
            if err > 1e-8:
                return dense

            c_dense = CircuitDecomposition._leaf_stats(dense, cache)[1]
            c_iso = CircuitDecomposition._leaf_stats(iso_gate, cache)[1]
            return iso_gate if c_iso < c_dense else dense
        except Exception:
            return dense

    @staticmethod
    def _content_key(gate: Gate) -> tuple | None:
        """
        A cache key derived from a ``UnitaryGate``'s *matrix*, or ``None``
        for anything else.

        Why this exists: a chain of N sites builds N separate
        ``UnitaryGate`` objects for its local isometries, but those matrices
        take only **three** distinct values (left boundary, bulk, right
        boundary) -- verified on mcz and diag at N = 4, 8, 16. Keyed on
        ``id`` alone, every bulk site is a cache miss, and each miss pays a
        full ``qs_decomposition`` inside ``.definition``. Measured cost of
        that redundancy, counting ``UnitaryGate._define`` calls:

            diag      N=4/8/16 ->  4 /  8 / 16 calls,   2.3 /  3.8 /  8.3 s
            subspace  N=4/8/16 ->  4 /  8 / 16 calls, 126.4 / 388.1 / 886.2 s

        i.e. exactly N calls where 3 would do, and on subspace that is
        ~50 s of pure recomputation per redundant site. Hashing the matrix
        (~1 MB at dim 256) is microseconds against that.

        The key includes ``num_qubits`` and the dtype/shape so that two
        different arrays can never collide on a digest alone, and the
        digest is of the exact bytes -- no tolerance. Two gates with
        bit-identical matrices decompose identically (the label plays no
        part in synthesis), so sharing blocks between them is exact, not an
        approximation.
        """
        if not isinstance(gate, UnitaryGate):
            return None
        try:
            mat = np.ascontiguousarray(gate.params[0])
        except Exception:
            return None
        return (
            "mat",
            gate.num_qubits,
            mat.shape,
            mat.dtype.str,
            hashlib.sha1(mat.tobytes()).hexdigest(),
        )

    @staticmethod
    def _leaf_blocks(gate: Gate, cache: dict) -> list[tuple[int, ...]]:
        """
        Flatten `gate` into its own <=2-qubit leaf blocks (each a tuple of
        local qubit indices into `gate`'s own num_qubits), cached by object
        identity -- same stopping rule as synthesize's leaves (a gate
        this atomic is never re-wrapped), so this recursion is only ever a
        handful of levels deep and its cost never scales with N.

        The basis is therefore "arbitrary 1- and 2-qubit unitaries", with
        no qiskit ``transpile``/``decompose`` pass involved. Every consumer
        of these blocks must use this same stopping rule, or the counts
        are not comparable.

        This is the ONE recursive qiskit-``.definition`` walk paid per
        distinct gate. Both a gate's own (depth, count) and its inverse's
        are derived from this same flat block list
        without ever calling ``gate.inverse()``: inverting a sequence of
        operations reverses their order and inverts each individually
        (``(ABC)^-1 = C^-1 B^-1 A^-1``), and at <=2-qubit granularity a
        block's own inverse still occupies exactly the same qubits and
        still counts as exactly one block, so "reverse this flat list" is
        an *exact* leaf-level realization of "fully flatten gate.inverse()"
        -- not an approximation, and it avoids a second ``.definition`` walk
        of ``gate.inverse()``.

        The cache stores ``id(gate) -> (gate, blocks)`` rather than just
        the blocks: many gates measured here are temporaries (the sub-gates
        of a ``.definition``, which qiskit rebuilds on each access), and
        CPython recycles the address of a freed object -- so caching on
        ``id`` alone silently returns one gate's blocks for an unrelated
        later one. Keeping a reference to the key object in the value
        makes the id unique for as long as the entry lives. Without this
        the measured gate count is not even deterministic: it depends on
        when the garbage collector happens to reuse an address.

        The ``num_qubits != len(local_qs)`` branch below is a defensive
        invariant check only; qiskit guarantees the two are equal, and it
        has not been observed to fire.
        """
        if isinstance(gate, _BlockLeaf):
            return gate.blocks
        key = id(gate)
        hit = cache.get(key)
        if hit is not None:
            return hit[1]
        # Content key first: `.definition` below is the expensive call, and
        # for a site isometry it is pure recomputation on every bulk site
        # (see _content_key). Checking it *before* touching `.definition`
        # is the whole point -- an id miss must not be allowed to trigger
        # the synthesis.
        ckey = CircuitDecomposition._content_key(gate)
        if ckey is not None:
            hit = cache.get(ckey)
            if hit is not None:
                cache[key] = (gate, hit[1])
                return hit[1]
        if gate.num_qubits <= 2:
            blocks = [tuple(range(gate.num_qubits))]
        else:
            try:
                defn = gate.definition
            except Exception as exc:
                # `.definition` synthesizes the gate lazily. For some shapes
                # (e.g. the padded d = chi = 8 isometry of fusion_group_S3)
                # Qiskit's Isometry -> UniformlyControlledGate decomposition
                # fails its own internal unitarity check, although the
                # isometry itself is exact to ~1e-15. The gate is then costed
                # as a single opaque block, which UNDER-counts its depth and
                # gate count: the resulting estimate is a lower bound.
                warnings.warn(
                    f"Qiskit could not decompose gate {gate.name!r} on {gate.num_qubits} "
                    f"qubits ({type(exc).__name__}); it is counted as one block, so "
                    "resource estimates involving it are lower bounds.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                defn = None
            if defn is None:
                blocks = [tuple(range(gate.num_qubits))]
            else:
                blocks = []
                q2idx = {q: i for i, q in enumerate(defn.qubits)}
                for ci in defn.data:
                    local_qs = tuple(q2idx[q] for q in ci.qubits)
                    if len(local_qs) <= 2:
                        blocks.append(local_qs)
                        continue
                    if ci.operation.num_qubits != len(local_qs):
                        blocks.append(local_qs)
                        continue
                    for blk in CircuitDecomposition._leaf_blocks(ci.operation, cache):
                        blocks.append(tuple(local_qs[i] for i in blk))
        cache[key] = (gate, blocks)
        if ckey is not None:
            cache[ckey] = (gate, blocks)
        return blocks

    @staticmethod
    def _blocks_stats(blocks: list[tuple[int, ...]]) -> tuple[int, int]:
        """(depth, gate_count) of a flat leaf-block list via exact per-qubit
        greedy scheduling, in the order given -- see _leaf_blocks."""
        last_layer: dict[int, int] = {}
        for qs in blocks:
            start = max((last_layer.get(q, 0) for q in qs), default=0)
            for q in qs:
                last_layer[q] = start + 1
        return (max(last_layer.values()) if last_layer else 0, len(blocks))

    @staticmethod
    def _leaf_stats(gate: Gate, cache: dict) -> tuple[int, int]:
        """(depth, gate_count) of `gate` at <=2-qubit block granularity --
        forward direction only. See _leaf_blocks for the recursion."""
        return CircuitDecomposition._blocks_stats(CircuitDecomposition._leaf_blocks(gate, cache))

    @staticmethod
    def _schedule_stats(
        schedule: list[tuple[Gate, tuple[int, ...]]], cache: dict
    ) -> tuple[int, int]:
        """(depth, gate_count) of a flat schedule (as built by
        _build_U_block_encoding's u_schedule/u_dagger_schedule), via exact
        per-qubit greedy scheduling over each entry's own leaf stats."""
        last_layer: dict[int, int] = {}
        total_count = 0
        for gate, qs in schedule:
            d, c = CircuitDecomposition._leaf_stats(gate, cache)
            start = max((last_layer.get(q, 0) for q in qs), default=0)
            end = start + d
            for q in qs:
                last_layer[q] = end
            total_count += c
        depth = max(last_layer.values()) if last_layer else 0
        return depth, total_count

    def estimate_synthesis_stats(
        self, pool_ancillas: bool = False, pool_levels: int | None = None
    ) -> dict:
        """Exact resources of :meth:`synthesize`'s circuit without building it.

        The tree is built exactly as :meth:`synthesize` builds it (same
        registers, leaf gates and segment programs, see
        :meth:`_build_tree`), but instead of expanding the root program into
        a circuit, it is scheduled directly: the greedy as-soon-as-possible
        layering of the <=2-qubit leaf blocks (:meth:`_program_depth`) and
        their number (:meth:`_program_count`). A segment's schedule depends
        only on its structure and on when each of its qubits becomes free
        (up to a common shift), so identical subtrees (the uniform bulk)
        and repeated invocations (the ``2 ell + 1`` replays of every child)
        are scheduled once and memoized. The flattened circuit, whose size
        grows like ``N^(1 + log2(2 ell + 1))``, is never built.

        Parameters
        ----------
        pool_ancillas, pool_levels
            As in :meth:`synthesize`.

        Returns
        -------
        dict
            ``depth`` and ``gate_count`` in <=2-qubit leaf blocks (arbitrary
            one- and two-qubit unitaries, the stopping rule of
            ``_leaf_blocks``; not transpiled ``["u", "cx"]`` gates) and
            ``num_qubits``. All three equal those of the circuit
            :meth:`synthesize` returns, flattened with the same rule.
        """
        tree = self._build_tree(pool_ancillas, pool_levels)
        master = tree["master"]
        cache: dict = {}
        n_local = len(master["v_qubits"])
        f = self._program_depth(master, False, [0] * n_local, {}, cache)
        num_qubits = sum(len(r) for r in tree["all_regs"])
        return dict(
            depth=max(f, default=0),
            gate_count=self._program_count(master, False, {}, cache),
            num_qubits=num_qubits,
        )

    # ------------------------------------------------------------------
    # Analytic estimator -- same recursion as estimate_synthesis_stats, but
    # as a pure function of a handful of measured constants. See
    # synthesis_cost_params / analytic_synthesis_stats.
    # ------------------------------------------------------------------

    def synthesis_cost_params(self, cache: dict | None = None) -> dict:
        """The N-independent data :meth:`analytic_synthesis_stats` needs.

        For a uniform chain every size-independent ingredient of the
        circuit is fixed: the three site isometries (left boundary, bulk,
        right boundary), the bond's U and U-dagger schedules, R_Phi and the
        register sizes. They are extracted once, as their <=2-qubit leaf
        blocks in both directions (a leaf and its ``inverse()`` are
        synthesized independently by Qiskit and generally differ), so that
        the circuit of *any* N can be scheduled from them exactly. The
        result is plain lists and integers (JSON round trips allowed).

        Requires ``N >= 3`` (a shorter chain has no bulk site) and a chain
        whose bonds all share one merge bundle (checked).
        """
        N = self._N
        if N < 3:
            raise ValueError(
                f"synthesis_cost_params needs N >= 3 to see a bulk site "
                f"(got N={N}); the parameters it returns are N-independent, "
                f"so extract them from a small chain and predict any N."
            )
        cache = {} if cache is None else cache

        nb = self._bond_qubit_count(0)
        bundle = self._merge_bundle(0)
        for b in range(1, N - 1):
            if self._bond_qubit_count(b) != nb or self._merge_bundle(b) is not bundle:
                raise ValueError(
                    "The analytic estimator assumes a uniform chain (one "
                    f"merge bundle for every bond); bond {b} differs."
                )

        def blocks(gate):
            fwd = CircuitDecomposition._leaf_blocks(gate, cache)
            inv = CircuitDecomposition._leaf_blocks(self._leaf_inverse(gate), cache)
            return [[list(x) for x in fwd], [list(x) for x in inv]]

        n_sys_per_site = int(np.ceil(np.log2(self._d)))
        D_sys_pad = 2**n_sys_per_site
        iso_cache: dict = {}
        site = {}
        for kind, k in (("left", 0), ("bulk", 1), ("right", N - 1)):
            n_left = nb if k > 0 else 0
            n_right = nb if k < N - 1 else 0
            gate, _, _ = self._site_leaf(k, D_sys_pad, n_left, n_right, iso_cache)
            site[kind] = blocks(gate)

        return dict(
            format=2,
            d=int(self._d),
            n_bond_per_site=nb,
            n_lcu=bundle["n_lcu"],
            n_rot=bundle["n_rot"],
            ell=int(bundle["ell"]),
            u_width=bundle["U_gate"].num_qubits,
            site=site,
            u=[[*blocks(g), list(qs)] for g, qs in bundle["u_schedule"]],
            u_dag=[[*blocks(g), list(qs)] for g, qs in bundle["u_dagger_schedule"]],
            R_Phi=blocks(bundle["R_Phi_gate"]),
        )

    @staticmethod
    def analytic_synthesis_stats(
        params: dict,
        N: int,
        pool_ancillas: bool = False,
        pool_levels: int | None = None,
        borrowed_ancillas: bool = True,
    ) -> dict:
        """:meth:`estimate_synthesis_stats` for any N, from :meth:`synthesis_cost_params`.

        The tree of an N-site chain is rebuilt from the N-independent leaf
        data (site isometries, U / U-dagger schedules, R_Phi) with the same
        builder as :meth:`synthesize` and scheduled exactly, so the result
        equals the flattened circuit that :meth:`synthesize` would return
        for that N. Only the reflections, whose width grows with N, are
        synthesized on the fly (one per distinct width, cheap even at
        thousands of qubits).

        The recursion behind the scaling laws: each tree layer replays its
        children ``2 ell + 1`` times (``V_pair`` once and ``R_Psi =
        V_pair Ref_0 V_pair^-1`` in each of the ``ell`` Grover iterates),
        hence depth ~ ``N^log2(2 ell + 1)`` and gate count ~
        ``N^(1 + log2(2 ell + 1))``.

        Parameters
        ----------
        params : dict
            Output of :meth:`synthesis_cost_params` (JSON round trips allowed).
        N : int
            Number of sites (``>= 2``).
        pool_ancillas, pool_levels
            As in :meth:`synthesize`: one lcu/rotation register per tree
            layer (or for the lowest ``pool_levels`` layers). Siblings then
            share qubits and serialize; fewer idle qubits can also change
            the reflections' borrowed-ancilla construction.
        borrowed_ancillas : bool
            ``False`` prices every reflection as a plain multi-controlled X
            without borrowed ancillas, for comparison (not what
            :meth:`synthesize` builds).

        Returns
        -------
        dict
            ``{"depth", "gate_count", "num_qubits"}``; depth and gate count
            in units of <=2-qubit leaf blocks.
        """
        if params.get("format") != 2:
            raise ValueError("params come from an older synthesis_cost_params; regenerate them.")
        if N < 2:
            raise ValueError("N must be >= 2.")
        tree = _ParamsTree(params, N, borrowed_ancillas)
        return tree.estimate_synthesis_stats(pool_ancillas, pool_levels)

    def synthesize(
        self,
        unitary_label: str | None = None,
        pool_ancillas: bool = False,
        pool_levels: int | None = None,
    ) -> tuple[
        QuantumCircuit,
        list[QuantumRegister],
        list[QuantumRegister],
        list[list[tuple[QuantumRegister, QuantumRegister]]],
        list,
    ]:
        r"""Synthesize the MPU into a Qiskit circuit (tree decomposition with GOAA).

        The chain is cut into single-site isometries, which are merged
        pairwise along a binary tree; every merge applies the block encoding
        of the merging operator and :math:`\ell` rounds of generalized
        oblivious amplitude amplification (thesis ``MPU_Decomposition``;
        Styliaris et al., PRL 135, 260602). The circuit is assembled from a
        flat list of leaf gates (one ``UnitaryGate``/isometry per entry,
        never re-wrapped), which keeps construction time proportional to
        the number of leaves.

        Parameters
        ----------
        unitary_label : str, optional
            Unused; kept for backward compatibility.
        pool_ancillas : bool
            Recycle ancillas: every merge of a tree layer uses the same lcu
            and rotation register (one pair per layer instead of one per
            merge). Valid because an exact merge returns its ancillas to
            :math:`|0\rangle` for every input; sibling subtrees then act on
            shared qubits and run one after the other, trading depth for
            qubits (see :meth:`analytic_synthesis_stats`).
        pool_levels : int, optional
            Pool only the lowest ``pool_levels`` layers.

        Returns
        -------
        qc_global : qiskit.QuantumCircuit
            Circuit implementing :math:`U_N` on the system registers, with
            every other qubit starting and ending in :math:`|0\rangle`, up to
            a global phase.
        site_sys_regs : list of QuantumRegister
            System register of each site (site 0 first), :math:`\lceil\log_2 d\rceil` qubits each.
        site_bond_regs : list of QuantumRegister
            Bond registers.
        anc_regs : list of list of (QuantumRegister, QuantumRegister)
            Per tree layer, the ``(lcu, rotation)`` ancilla registers of each merge.
        all_anc_qubits : list
            Every non-system qubit (for post-selection).

        Notes
        -----
        Gate counts and depth of the returned circuit refer to *composite*
        gates (multi-qubit ``UnitaryGate``/isometry objects). For
        elementary-gate resources transpile first, e.g.
        ``qiskit.transpile(qc, basis_gates=["u", "cx"])``, or use
        :meth:`estimate_synthesis_stats` / :meth:`synthesis_cost_params`.
        Without pooling, ancilla registers are allocated per merge (no
        recycling).
        """
        del unitary_label
        tree = self._build_tree(pool_ancillas, pool_levels)
        master_segment = tree["master"]
        qc_global = QuantumCircuit(*tree["all_regs"], name="mpu_circuit_lean")
        leaves: list = []
        self._expand_program(master_segment, False, list(master_segment["v_qubits"]), leaves)
        for gate, qubits in leaves:
            qc_global.append(gate, qubits)
        qc_global.global_phase += master_segment["global_phase"]

        site_sys_regs, site_bond_regs, anc_regs = (
            tree["site_sys_regs"],
            tree["site_bond_regs"],
            tree["anc_regs"],
        )
        all_anc_qubits = []

        for b_reg in site_bond_regs:
            all_anc_qubits.extend(list(b_reg))

        for layer_regs in anc_regs:
            for lcu_r, rot_r in dict.fromkeys(layer_regs):
                all_anc_qubits.extend(list(lcu_r))
                all_anc_qubits.extend(list(rot_r))

        return qc_global, site_sys_regs, site_bond_regs, anc_regs, all_anc_qubits

    def _build_tree(self, pool_ancillas: bool = False, pool_levels: int | None = None) -> dict:
        """Registers and the root segment program shared by :meth:`synthesize`
        and :meth:`estimate_synthesis_stats`."""
        N = self._N

        n_sys_per_site = int(np.ceil(np.log2(self._d)))
        D_sys_pad = 2**n_sys_per_site

        site_sys_regs = [QuantumRegister(n_sys_per_site, name=f"sys_{k}") for k in range(N)]

        def n_bond_qubits_right(k: int) -> int:
            return self._bond_qubit_count(k) if k < N - 1 else 0

        def n_bond_qubits_left(k: int) -> int:
            return self._bond_qubit_count(k - 1) if k > 0 else 0

        site_bond_regs = [
            QuantumRegister(
                n_bond_qubits_right(k) + n_bond_qubits_left(k),
                name=f"bond_{k}",
            )
            for k in range(N)
        ]

        plan = self._tree_merge_plan(N)

        anc_regs: list[list[tuple[QuantumRegister, QuantumRegister]]] = []

        if pool_ancillas:
            pool_levels = len(plan)

        for layer, layer_bonds in enumerate(plan):
            layer_regs = []
            bundles = [self._merge_bundle(bond) for bond in layer_bonds]

            if pool_levels and layer < pool_levels:
                lcu_r = QuantumRegister(max(b["n_lcu"] for b in bundles), name=f"lcu_l{layer}")
                rot_r = QuantumRegister(max(b["n_rot"] for b in bundles), name=f"rot_l{layer}")
                layer_regs = [(lcu_r, rot_r)] * len(bundles)
            else:
                for p, bundle in enumerate(bundles):
                    lcu_r = QuantumRegister(bundle["n_lcu"], name=f"lcu_l{layer}_p{p}")
                    rot_r = QuantumRegister(bundle["n_rot"], name=f"rot_l{layer}_p{p}")
                    layer_regs.append((lcu_r, rot_r))

            anc_regs.append(layer_regs)

        all_regs = []

        for k in range(N):
            all_regs.append(site_sys_regs[k])
            all_regs.append(site_bond_regs[k])

        for layer_regs in anc_regs:
            for lcu_r, rot_r in dict.fromkeys(layer_regs):
                all_regs.append(lcu_r)
                all_regs.append(rot_r)

        segments = self._make_segment_list(
            D_sys_pad=D_sys_pad,
            n_sys_per_site=n_sys_per_site,
            site_sys_regs=site_sys_regs,
            site_bond_regs=site_bond_regs,
        )

        for layer, layer_anc_pairs in enumerate(anc_regs):
            segments = self._merge_layer(
                segments=segments,
                layer_anc_pairs=layer_anc_pairs,
                layer_bonds=plan[layer],
                layer=layer,
            )

        if len(segments) != 1:
            raise RuntimeError(f"Tree synthesis ended with {len(segments)} segments instead of 1.")

        master_segment = segments[0]

        if master_segment["v_qubits"] != self._canonical_segment_qubits(master_segment):
            raise RuntimeError("Final master segment violates canonical ordering.")

        return dict(
            master=master_segment,
            all_regs=all_regs,
            site_sys_regs=site_sys_regs,
            site_bond_regs=site_bond_regs,
            anc_regs=anc_regs,
        )


class GeneralMPU(CircuitDecomposition):
    r"""Site-dependent MPU defined by one tensor per site.

    Every quantity that is size independent for :class:`UniformMPU`
    (boundary operators, merging operator, conditioning number, GOAA data) is
    indexed by the bond it belongs to and computed once, with one left and one
    right sweep over the chain. By default the chain is first brought to
    canonical form (minimal bond dimensions, :math:`L_k=I`,
    :math:`R_k=\mathrm{diag}(s)`), which removes the need for Assumption 1.

    Attributes
    ----------
    q : float
        Largest per-bond conditioning number, :math:`\max_k q_k` (thesis
        ``eq:q_nonuniform``).
    boundary_source : list of str
        Boundary strategy per bond (``"canonical"``, ``"sigma_tau_optimized"``,
        ``"optimized"``).
    tensors : list of ndarray
        The (canonicalized) site tensors actually used.
    """

    def __init__(
        self,
        tensors_A: list[np.ndarray],
        l_vec: np.ndarray,
        r_vec: np.ndarray,
        tol: float = 1e-12,
        canonicalize: bool = True,
        canonical_tol: float = 1e-10,
        boundary_sigma_tau: bool = False,
        boundary_sigma_tau_max_sites: int | None = None,
        boundary_sigma_tau_n_restarts: int = 4,
        boundary_sigma_tau_seed: int = 0,
        boundary_opt_p: int | None = None,
        boundary_opt_gauge: bool = False,
        boundary_opt_n_restarts: int = 10,
        boundary_opt_lam_iso: float = 1e6,
        boundary_opt_seed: int = 0,
    ):
        r"""Build the per-bond boundary data of a site-dependent MPU.

        Parameters
        ----------
        tensors_A : list of ndarray
            Site tensors ``A_k[p_in, p_out, bond_L, bond_R]`` (same ``d``), ``N >= 2``.
        l_vec, r_vec : ndarray
            Boundary vectors (contracted into the first / last bond).
        tol : float
            Tolerance of the structural checks.
        canonicalize : bool
            Bring the chain to canonical form first (default). Bond dimensions may
            then be smaller than the input ones.
        canonical_tol : float
            Relative singular-value threshold of the canonicalization.
        boundary_sigma_tau : bool
            Optimize a density matrix per site near each chain end
            (:func:`~TreeMPU.utils.optimize_chain_sigma_tau`); a bond keeps the
            result only if its :math:`q_k` decreases.
        boundary_sigma_tau_max_sites : int, optional
            Number of sites nearest each end with a free density matrix
            (``None``: all).
        boundary_sigma_tau_n_restarts, boundary_sigma_tau_seed
            Passed to :func:`~TreeMPU.utils.optimize_chain_sigma_tau`.
        boundary_opt_p : int, optional
            Partial optimization: for windows of up to :math:`2^p` sites that reach
            a chain end, replace the boundary operator of the bond by a directly
            optimized one (:func:`~TreeMPU.utils.optimize_boundary_operators`); kept
            only if :math:`q_k` decreases with a small isometry residual.
        boundary_opt_gauge : bool
            Not supported together with ``boundary_opt_p`` (raises).
        boundary_opt_n_restarts, boundary_opt_lam_iso, boundary_opt_seed
            Passed to :func:`~TreeMPU.utils.optimize_boundary_operators`.

        Raises
        ------
        ValueError
            Malformed tensors, inconsistent bond dimensions, failed normalization
            check, or rank-deficient boundary operators (with ``canonicalize=False``).
        NotImplementedError
            If ``N < 2``.
        """
        N = len(tensors_A)

        if N < 2:
            raise NotImplementedError("GeneralMPU.synthesize currently requires N >= 2.")

        # --- 1. Theoretical Validations (Fail-Fast Logic) ---

        d = tensors_A[0].shape[0]

        for k, A_k in enumerate(tensors_A):
            if A_k.ndim != 4:
                raise ValueError(
                    f"Tensor A_{k} must have four axes "
                    f"(p_in, p_out, bond_L, bond_R). Got shape {A_k.shape}."
                )
            if A_k.shape[0] != d or A_k.shape[1] != d:
                raise ValueError(
                    f"Tensor A_{k} has physical dimension {A_k.shape[:2]}, "
                    f"expected the uniform d={d} shared by every site."
                )

        if tensors_A[0].shape[2] != len(l_vec):
            raise ValueError(
                f"l_vec has dimension {len(l_vec)}, but A_0's left bond "
                f"dimension is {tensors_A[0].shape[2]}."
            )
        if tensors_A[-1].shape[3] != len(r_vec):
            raise ValueError(
                f"r_vec has dimension {len(r_vec)}, but A_{{{N - 1}}}'s right "
                f"bond dimension is {tensors_A[-1].shape[3]}."
            )
        for k in range(N - 1):
            if tensors_A[k].shape[3] != tensors_A[k + 1].shape[2]:
                raise ValueError(
                    f"Bond dimension mismatch between site {k} (right bond "
                    f"{tensors_A[k].shape[3]}) and site {k + 1} (left bond "
                    f"{tensors_A[k + 1].shape[2]})."
                )

        # --- 2. State Encapsulation ---

        self.tensors = [np.asarray(A, dtype=complex) for A in tensors_A]
        self.tol = tol

        # Unlike the uniform-bulk case, Assumption 1 is NOT needed here. The
        # paper is explicit that for nonuniform MPUs "the tensors L_j, R_k
        # retain the properties (a)-(c) from the uniform case, even without
        # the need for Assumption 1", because one may "bring, without loss of
        # generality, the MPU into a canonical form" -- successive SVDs over
        # the Choi state's paired physical index, after which the bond
        # dimension at every cut is minimal and Eq. (26) gives L_k = 1,
        # R_k = diag(s_k) with all s_k strictly positive.
        #
        # Rank deficiency is therefore a property of the *representation*,
        # not of the MPU: it means the caller's bond dimension carries
        # redundancy. Canonicalizing removes it, rather than rejecting the
        # chain (which is what UniformMPU must do instead, since there the
        # bulk tensor has to stay a single repeated tensor -- see the paper's
        # footnote 2 on Theorem 1' -- so blocking, not canonical form, is the
        # remedy on that side).
        if canonicalize:
            self.tensors, l_vec, r_vec = to_canonical_form(
                self.tensors, l_vec, r_vec, tol=canonical_tol
            )
            N = len(self.tensors)
            if N < 2:
                raise ValueError(
                    "Canonical form collapsed the chain below 2 tensors; the "
                    "MPU is trivial at this length."
                )
        self.canonicalized = bool(canonicalize)

        super().__init__(N=N, d=d, l_vec=l_vec, r_vec=r_vec)

        # --- 3. Boundary operators: one left sweep + one right sweep (O(N)) ---

        L2_list, R2_list, residual = compute_chain_boundary_operators(self.tensors, l_vec, r_vec)
        check_chain_unitarity(residual, tol=1e-6)

        self.boundary_source = ["canonical"] * (N - 1)

        # --- 3.1. Optional sigma/tau-optimized baseline (see
        # boundary_sigma_tau docstring above). Every candidate is a valid
        # density matrix by construction, so there is no isometry
        # constraint to fail here -- only whether it actually beats the
        # naive I/d pair, decided independently per bond (L and R together). Wrapped in try/except like
        # UniformMPU's own boundary_sigma_tau: a JAX optimizer that fails to
        # converge should fall back to the always-valid I/d pair, not abort
        # construction.
        if boundary_sigma_tau:
            try:
                L2_st, R2_st, _ = optimize_chain_sigma_tau(
                    self.tensors,
                    l_vec,
                    r_vec,
                    max_free_sites=boundary_sigma_tau_max_sites,
                    n_restarts=boundary_sigma_tau_n_restarts,
                    seed=boundary_sigma_tau_seed,
                )

                def _q_from_squares(L2, R2):
                    L2_inv = np.linalg.inv(L2)
                    R2_inv = np.linalg.inv(R2)
                    return float(np.real(np.sqrt(np.trace(R2_inv @ L2_inv.T))))

                for b in range(N - 1):
                    if np.all(np.isfinite(L2_st[b])) and np.all(np.isfinite(R2_st[b])):
                        q_st = _q_from_squares(L2_st[b], R2_st[b])
                        q_naive = _q_from_squares(L2_list[b], R2_list[b])
                        if q_st < q_naive - 1e-9:
                            L2_list[b], R2_list[b] = L2_st[b], R2_st[b]
                            self.boundary_source[b] = "sigma_tau_optimized"
            except Exception as exc:
                # A failed optimization falls back to the I/d baseline, but is
                # reported so that genuine errors are not hidden.
                warnings.warn(
                    f"sigma/tau chain optimization failed ({type(exc).__name__}: {exc}); "
                    "using the I/d boundaries.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._bond_dims = [L2_list[b].shape[0] for b in range(N - 1)]

        self.L: list[np.ndarray] = []
        self.R: list[np.ndarray] = []
        self.L_inv: list[np.ndarray] = []
        self.R_inv: list[np.ndarray] = []

        for b in range(N - 1):
            check_full_rank(L2_list[b], label=f"L^2 at bond {b}", tol=tol)
            check_full_rank(R2_list[b], label=f"R^2 at bond {b}", tol=tol)

            L_b = matrix_sqrt_hermitian(L2_list[b])
            R_b = matrix_sqrt_hermitian(R2_list[b])

            self.L.append(L_b)
            self.R.append(R_b)
            self.L_inv.append(np.linalg.inv(L_b))
            self.R_inv.append(np.linalg.inv(R_b))

        # --- 3.5. Optional "partial optimization" at power-of-two window
        # lengths (thesis section "Partial Optimization"). See the
        # boundary_opt_p docstring above for exactly which bonds this can
        # affect and why: only ones whose window already reaches a true
        # chain boundary (l_vec or r_vec) within 2**boundary_opt_p sites.
        # self.boundary_source was already initialized above (and possibly
        # partly set to "sigma_tau_optimized" by boundary_sigma_tau); this
        # step only ever overwrites individual bonds it actually improves.

        if boundary_opt_p is not None:
            if boundary_opt_gauge:
                # Every boundary_opt_p window is anchored to l_vec or r_vec,
                # i.e. (with canonicalize=True, the default) has a trivial
                # (dimension-1) bond on the anchored side and the chain's own
                # bond dimension on the other -- optimize_boundary_operators
                # rejects gauge=True there since the same unitary G cannot
                # dress two differently-sized bonds (L@G^dagger, G@R). This
                # is not an occasional failure to retry around: it holds for
                # every window boundary_opt_p ever attempts, so raise early
                # and clearly rather than let the first attempt crash deep
                # inside the optimizer.
                raise ValueError(
                    "boundary_opt_gauge=True cannot be combined with "
                    "boundary_opt_p: every window it optimizes is anchored "
                    "to the chain's own l_vec/r_vec, giving it a trivial "
                    "bond on that side (after canonicalization) that a "
                    "shared gauge transformation cannot act on together "
                    "with the other, non-trivial side."
                )
            max_window = 2**boundary_opt_p

            def _q_from(L_b, R_b):
                L2_inv = np.linalg.inv(L_b) @ np.linalg.inv(L_b)
                R2_inv = np.linalg.inv(R_b) @ np.linalg.inv(R_b)
                return float(np.real(np.sqrt(np.trace(R2_inv @ L2_inv.T))))

            def _site_is_isometric(k: int) -> bool:
                # Site k uses L[k-1] (or l_vec) on the left and R[k] (or r_vec)
                # on the right. Windows optimized from the two chain ends are
                # independent, so a site where they meet must be re-checked.
                A = self.tensors[k]
                if k == 0:
                    V = np.einsum("a,ijab,yb->jyi", l_vec, A, self.R[0])
                elif k == N - 1:
                    V = np.einsum("xa,ijab,b->xji", self.L[N - 2], A, r_vec)
                else:
                    V = np.einsum("xa,ijab,yb->xjyi", self.L[k - 1], A, self.R[k])
                V = V.reshape(-1, A.shape[0])
                return np.linalg.norm(V.conj().T @ V - np.eye(A.shape[0])) < _OPT_P_SITE_TOL

            for b in range(N - 1):
                w_left = b + 1
                if w_left <= max_window:
                    _, R_opt, q_opt, iso_res = optimize_boundary_operators(
                        self.tensors[: b + 1],
                        gauge=boundary_opt_gauge,
                        l_vec=l_vec,
                        n_restarts=boundary_opt_n_restarts,
                        lam_iso=boundary_opt_lam_iso,
                        seed=boundary_opt_seed,
                    )
                    if (
                        q_opt >= 1.0 - 1e-6
                        and iso_res < 1e-3
                        and _q_from(self.L[b], R_opt) <= self._compute_q_bond(b) + 1e-9
                    ):
                        R_old = self.R[b]
                        self.R[b] = R_opt
                        if _site_is_isometric(b):
                            self.R_inv[b] = np.linalg.inv(R_opt)
                            self.boundary_source[b] = (
                                "optimized_gauge" if boundary_opt_gauge else "optimized"
                            )
                        else:
                            self.R[b] = R_old

                w_right = N - 1 - b
                if w_right <= max_window:
                    L_opt, _, q_opt, iso_res = optimize_boundary_operators(
                        self.tensors[b + 1 :],
                        gauge=boundary_opt_gauge,
                        r_vec=r_vec,
                        n_restarts=boundary_opt_n_restarts,
                        lam_iso=boundary_opt_lam_iso,
                        seed=boundary_opt_seed,
                    )
                    if (
                        q_opt >= 1.0 - 1e-6
                        and iso_res < 1e-3
                        and _q_from(L_opt, self.R[b]) <= self._compute_q_bond(b) + 1e-9
                    ):
                        L_old = self.L[b]
                        self.L[b] = L_opt
                        if _site_is_isometric(b + 1):
                            self.L_inv[b] = np.linalg.inv(L_opt)
                            self.boundary_source[b] = (
                                "optimized_gauge" if boundary_opt_gauge else "optimized"
                            )
                        else:
                            self.L[b] = L_old

        # --- 4. Per-bond merging operators & MPU conditioning number ---

        self.MergingOperators = [
            get_merging_operator(self.L_inv[b], self.R_inv[b]) for b in range(N - 1)
        ]
        self._q_bonds = [self._compute_q_bond(b) for b in range(N - 1)]
        self.q = float(max(self._q_bonds))

        # --- 5. Eager per-bond LCU / GOAA data ---

        self._goaa_bonds = [
            CircuitDecomposition._compute_goaa_bundle(self.MergingOperators[b])
            for b in range(N - 1)
        ]

        # --- 6. Eager per-site / per-bond circuit-synthesis data ---

        self._n_sys_per_site = int(np.ceil(np.log2(d)))
        self._bond_qubits = [int(np.ceil(np.log2(self._bond_dims[b]))) for b in range(N - 1)]

        self._bond_bundles = []
        for b in range(N - 1):
            n_bond_per_site = self._bond_qubits[b]
            _, U_gate, U_dagger_gate, u_schedule, u_dagger_schedule, n_lcu, n_rot = (
                CircuitDecomposition._build_U_block_encoding(
                    self._goaa_bonds[b],
                    D_bond=self._bond_dims[b],
                    n_bond_per_site=n_bond_per_site,
                )
            )
            self._bond_bundles.append(
                {
                    "n_bond_per_site": n_bond_per_site,
                    "n_lcu": n_lcu,
                    "n_rot": n_rot,
                    "ell": int(self._goaa_bonds[b]["ell"]),
                    "U_gate": U_gate,
                    "U_dagger_gate": U_dagger_gate,
                    "u_schedule": u_schedule,
                    "u_dagger_schedule": u_dagger_schedule,
                    "R_Phi_gate": CircuitDecomposition._make_reflection_zero(n_lcu + n_rot),
                }
            )

        self._site_tensors = self.create_local_isometries()

    def _compute_q_bond(self, b: int) -> float:
        r"""
        Site-dependent MPU conditioning number at bond b:
        $q_b = \sqrt{\text{Tr}(R_{inv,b}^2 (L_{inv,b}^2)^T)}$ (Eq. 17).
        """
        L2_inv = self.L_inv[b] @ self.L_inv[b]
        R2_inv = self.R_inv[b] @ self.R_inv[b]
        return float(np.real(np.sqrt(np.trace(R2_inv @ L2_inv.T))))

    def create_local_isometries(self) -> list[qtn.Tensor]:
        """
        Construct the N dressed local isometry tensors V_0, ..., V_{N-1}.

        Mirrors :meth:`UniformMPU.create_local_isometries`, generalized to
        site-dependent tensors and per-bond capping operators: site k is
        dressed on its left with ``L[k - 1]`` (or ``l_vec`` if k == 0) and on
        its right with ``R[k]`` (or ``r_vec`` if k == N - 1).
        """
        N = self._N
        site_tensors: list[qtn.Tensor] = []

        for k in range(N):
            site_label = k + 1
            A_tn, _ = get_mpo_site_tensors(site_label, self.tensors[k])

            if k > 0:
                L_tn = qtn.Tensor(self.L[k - 1], inds=["ext_L", f"bond_{k}"], tags=["L"])
                A_tn = L_tn @ A_tn

            if k < N - 1:
                R_tn = qtn.Tensor(self.R[k], inds=["ext_R", f"bond_{k + 1}"], tags=["R"])
                A_tn = A_tn @ R_tn

            if k == 0:
                A_tn = self.l_vec @ A_tn
            if k == N - 1:
                A_tn = A_tn @ self.r_vec

            site_tensors.append(A_tn)

        return site_tensors

    def _get_merging_operator(self, bond_index: int) -> np.ndarray:
        return self.MergingOperators[bond_index]

    def _build_lcu_data(self, bond_index: int) -> tuple[list, list, float]:
        bundle = self._goaa_bonds[bond_index]
        return bundle["coef_0"], bundle["unit_0"], bundle["C_0"]

    def _build_goaa_data(self, bond_index: int) -> dict:
        return self._goaa_bonds[bond_index]

    def _site_isometry_tensor(self, k: int) -> qtn.Tensor:
        return self._site_tensors[k]

    def _bond_qubit_count(self, bond_index: int) -> int:
        return self._bond_qubits[bond_index]

    def _bond_dimension(self, bond_index: int) -> int:
        return self._bond_dims[bond_index]

    def _merge_bundle(self, bond_index: int) -> dict:
        return self._bond_bundles[bond_index]

    def _isometry_tol(self) -> float:
        if all(src == "canonical" for src in self.boundary_source):
            return 1e-10
        # At least one bond's L/R came from optimize_boundary_operators
        # (penalty-method precision) or from optimize_chain_sigma_tau, whose
        # environments put sigma/tau on the output leg (see its docstring),
        # a case not covered by the input-leg exactness argument.
        return 1e-4


class UniformMPU(CircuitDecomposition):
    r"""Translation-invariant MPU defined by one bulk tensor and two boundary vectors.

    :math:`U_N=\sum \ell_{v_0} r_{v_N}\,A[v_0,v_1]\otimes\cdots\otimes A[v_{N-1},v_N]`.
    Exact unitarity is checked at construction (:func:`~TreeMPU.checks.check_mpo_unitarity`),
    a boundary pair :math:`(L,R)` is chosen, and the circuit is produced by
    :meth:`synthesize`.

    Boundary strategies (``boundary_source`` after construction), in priority order:

    ``"mode1_exact"``
        Closed form reaching :math:`q=\chi`, for Mode I tensors with
        single-vertex readout (``boundary_mode1_exact=True``).
    ``"qD_general"``
        :func:`~TreeMPU.utils.solve_qD_boundary`, reaching :math:`q=\chi` for
        general bulk tensors (``boundary_opt_k``).
    ``"optimized"``, ``"optimized_gauge"``
        :func:`~TreeMPU.utils.optimize_boundary_operators` (``boundary_opt_k``,
        optionally ``boundary_opt_gauge``).
    ``"canonical_sigma_tau"``
        Canonical boundaries with optimized :math:`\sigma,\tau`
        (``boundary_sigma_tau=True``, default), used only if it lowers :math:`q`.
    ``"canonical"``
        Canonical boundaries with :math:`\sigma=\tau=I/d`
        (Styliaris et al.; requires Assumption 1).

    Attributes
    ----------
    q_unif : float
        Conditioning number of the chosen boundary (LCU 1-norm of the merging
        operator).
    ell : int
        Number of GOAA rounds per merge.
    boundary_source : str
        Which strategy produced ``L``, ``R``.
    L, R, L_inv, R_inv : ndarray
        Boundary operators and inverses.
    unitarity_checked_up_to : int
        Largest chain length at which exact unitarity was checked.
    unitarity_certified_all_N : bool
        Whether that suffices for every :math:`N` (lengths up to
        :math:`\chi^2+1` checked).
    """

    def __init__(
        self,
        A: np.ndarray,
        l_vec: np.ndarray,
        r_vec: np.ndarray,
        N: int,
        tol: float = 1e-12,
        debug: bool = False,
        light: bool | int = False,
        boundary_sigma_tau: bool = True,
        boundary_mode1_exact: bool = False,
        boundary_opt_k: int | None = None,
        boundary_opt_gauge: bool = False,
        boundary_opt_n_restarts: int = 10,
        boundary_opt_lam_iso: float = 1e6,
        boundary_opt_seed: int = 0,
    ):
        r"""Check the MPU, choose a boundary pair and precompute the merge data.

        Parameters
        ----------
        A : ndarray
            Bulk tensor ``A[p_in, p_out, bond_L, bond_R]``, shape ``(d, d, chi, chi)``.
        l_vec, r_vec : ndarray
            Boundary vectors.
        N : int
            Number of sites (``N >= 2``).
        tol : float
            Tolerance of the unitarity and structural checks.
        debug : bool
            Also verify the LCU decomposition and the block encoding numerically.
        light : bool or int
            Lengths covered by the exact unitarity check: ``False`` all
            :math:`1..N`; ``True`` :math:`1..\min(N,\chi^2+1)`, which by the thesis
            finite-size cutoff theorem certifies every :math:`N` at a cost
            independent of :math:`N`; an integer :math:`k`: :math:`1..\min(N,k)`.
            The check costs ~0.4 s at 32 sites and ~30 s / 1.5 GB at 256.
        boundary_sigma_tau : bool
            Try canonical boundaries with optimized :math:`\sigma,\tau`
            (:func:`~TreeMPU.utils.optimize_q_unif`); adopted only if :math:`q`
            decreases.
        boundary_mode1_exact : bool
            Try :func:`~TreeMPU.utils.mode1_exact_boundary` (Mode I tensors with
            :math:`\ell=e_v`, :math:`r=\sum_v e_v`; reaches :math:`q=\chi`). Falls
            back with a ``RuntimeWarning`` if the structure is absent.
        boundary_opt_k : int, optional
            Try :func:`~TreeMPU.utils.solve_qD_boundary` and, if it fails,
            :func:`~TreeMPU.utils.optimize_boundary_operators` with the isometry
            constraint imposed at block lengths :math:`1..k`. Every candidate is
            re-validated with an exact isometry check before being accepted.
        boundary_opt_gauge : bool
            Also optimize a unitary gauge (strictly larger feasible set); skips
            ``solve_qD_boundary``, which returns HPD boundaries only.
        boundary_opt_n_restarts, boundary_opt_lam_iso, boundary_opt_seed
            Passed to :func:`~TreeMPU.utils.optimize_boundary_operators`.

        Raises
        ------
        ValueError
            If the MPU is not unitary, or no admissible boundary could be built
            (Assumption 1 fails and no free construction succeeded).
        NotImplementedError
            If ``N < 2``.

        Notes
        -----
        Assumption 1 (injective length-1 boundary maps) is required only by the
        canonical constructions; when a free construction is requested its failure
        is a warning. :math:`q=\chi` from ``mode1_exact`` / ``qD_general`` is
        taken as final (the optimizers are skipped). This is a heuristic, not a
        bound: by Cauchy-Schwarz :math:`q\,\|LR^\top\|_F\ge\chi`, and no uniform
        boundary tested had :math:`q<\chi`, but site-dependent windowed boundaries
        (``GeneralMPU(boundary_opt_p=...)``) do reach :math:`q<\chi` (the thesis
        proves :math:`q\ge 1`). Boundaries from iterative solvers are accepted at
        isometry tolerance ``1e-4`` and each site isometry is projected onto the
        nearest exact one during synthesis; use
        :func:`~TreeMPU.checks.block_isometry_residuals` to quantify the residual.
        """
        if N < 2:
            raise NotImplementedError("UniformMPU.synthesize currently requires N >= 2.")

        # --- 1. Theoretical Validations (Fail-Fast Logic) ---

        # Check global unitary property via local left-canonical isometry
        # condition. `light` caps the length walked here -- see the docstring
        # for why this is the constructor's dominant cost and what capping it
        # gives up.
        chi = A.shape[2]
        if light is False:
            n_check = N
        else:
            n_check = min(N, chi**2 + 1 if light is True else int(light))
            if n_check < 1:
                raise ValueError(f"light must cap the unitarity check at >= 1 site, got {light}.")
        self.unitarity_checked_up_to = n_check
        check_mpo_unitarity(n_check, A, l_in=l_vec, r_in=r_vec, tol=tol)
        #: True if exact unitarity is certified for *every* chain length
        #: (all lengths up to chi**2 + 1 were checked; thesis finite-size
        #: cutoff theorem), not only up to ``unitarity_checked_up_to``.
        self.unitarity_certified_all_N = n_check >= chi**2 + 1

        # Assumption 1 (injectivity of the boundary maps) is needed ONLY to
        # guarantee the canonical (sigma/tau or plain I/d) construction below
        # produces invertible L, R -- see the docstring's note on this. It is
        # NOT needed by mode1_exact_boundary, solve_qD_boundary, or
        # optimize_boundary_operators: each independently guarantees its own
        # invertibility, regardless of whether the raw boundary maps are full
        # rank. So this is not an unconditional prerequisite -- only a
        # warning when a free construction was actually requested, since
        # Assumption 1 might not be needed at all; a hard failure otherwise,
        # since the (sole) canonical construction would be unusable.
        try:
            check_assumption_1(A, l_vec, r_vec, tol=1e-12)
            assumption_1_ok = True
        except ValueError as exc:
            assumption_1_ok = False
            if boundary_mode1_exact or boundary_opt_k is not None:
                warnings.warn(
                    f"Assumption 1 not satisfied ({exc}); the canonical "
                    "boundary construction is unavailable as a fallback -- "
                    "proceeding with only the requested free "
                    "construction(s) (boundary_mode1_exact/boundary_opt_k); "
                    "construction will fail if none of them succeed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                raise

        # --- 2. State Encapsulation ---

        # Metadata and Dimensions
        self.tol = tol
        self._D = A.shape[2]  # Bond dimension
        self._d = A.shape[0]

        # Convert raw numpy arrays into quimb Tensors for the graph
        self.A = qtn.Tensor(
            data=A,
            inds=("p_in", "p_out", "bond_0", "bond_N"),
            tags=["A"],
        )

        super().__init__(N=N, d=self._d, l_vec=l_vec, r_vec=r_vec)

        # --- 3. Derived Quantum Properties ---
        self.mode1_exact = False
        self.qD_general = False
        self.qD_diagnostics = None
        self.q_baseline = None
        self.boundary_opt_result = None

        if assumption_1_ok:
            # Baseline: sigma/tau optimized, NOT the plain I/d pair. The
            # maximally mixed choice is just one arbitrary feasible point;
            # optimizing the two density matrices stays entirely inside the
            # canonical construction (so the isometry holds at every length
            # by construction, with no constraint to enforce) and can lower q.
            # The result is kept only if q actually decreases.
            self.L_id, self.R_id = self._compute_boundary_operators()
            self.sigma_opt = self.tau_opt = None
            if boundary_sigma_tau:
                try:
                    sg, tu, _ = optimize_q_unif(
                        A, l_vec, r_vec, rng=np.random.default_rng(boundary_opt_seed)
                    )
                    L_st, R_st = self._compute_boundary_operators(sigma=sg, tau=tu)
                    if (
                        np.all(np.isfinite(L_st))
                        and np.all(np.isfinite(R_st))
                        and np.linalg.matrix_rank(L_st) == self._D
                        and np.linalg.matrix_rank(R_st) == self._D
                        # adopt only a genuine improvement, not rounding noise
                        and conditioning_number(L_st, R_st)
                        < conditioning_number(self.L_id, self.R_id) * (1.0 - 1e-9)
                    ):
                        self.L_canonical, self.R_canonical = L_st, R_st
                        self.sigma_opt, self.tau_opt = sg, tu
                    else:
                        self.L_canonical, self.R_canonical = self.L_id, self.R_id
                except Exception as exc:
                    # Fall back to I/d, but report the failure.
                    warnings.warn(
                        f"sigma/tau optimization failed ({type(exc).__name__}: {exc}); "
                        "using the I/d boundaries.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self.L_canonical, self.R_canonical = self.L_id, self.R_id
            else:
                self.L_canonical, self.R_canonical = self.L_id, self.R_id

            self.boundary_source = (
                "canonical_sigma_tau" if self.sigma_opt is not None else "canonical"
            )
        else:
            # No canonical fallback available -- overwritten below if
            # boundary_mode1_exact/boundary_opt_k succeeds; the final check
            # before self.L_inv catches the case where nothing did.
            self.L_id = self.R_id = None
            self.sigma_opt = self.tau_opt = None
            self.L_canonical, self.R_canonical = None, None
            self.boundary_source = None

        if boundary_mode1_exact:
            try:
                L_m1, R_m1, _q_m1, _iso_res_m1 = mode1_exact_boundary(A, l_vec, r_vec)
                self.L_canonical, self.R_canonical = L_m1, R_m1
                self.mode1_exact = True
                self.boundary_source = "mode1_exact"
            except ValueError as exc:
                warnings.warn(
                    f"boundary_mode1_exact=True but the closed form does not "
                    f"apply to this tensor/boundary ({exc}); falling back to "
                    "the sigma/tau baseline (or naive I/d)"
                    + (
                        "."
                        if assumption_1_ok
                        else ", which is itself unavailable since Assumption "
                        "1 also failed -- construction will fail unless "
                        "boundary_opt_k succeeds."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.L, self.R = self.L_canonical, self.R_canonical

        if boundary_opt_k is not None:
            # Try the general q=D floor construction first, checked at the
            # same support lengths requested for the optimizer below (see
            # docstring). Skipped entirely if boundary_mode1_exact already
            # got q=D -- nothing to gain, only search cost to lose. Also
            # skipped when boundary_opt_gauge=True: that explicitly asks for
            # the strictly larger non-HPD feasible set gauge exploration
            # covers, which solve_qD_boundary -- HPD square roots only --
            # cannot provide, so it must not preempt that search.
            qD_success = False
            if not self.mode1_exact and not boundary_opt_gauge:
                try:
                    L_qD, R_qD, _q_qD, diag_qD = solve_qD_boundary(
                        A,
                        l_vec,
                        r_vec,
                        n_restarts=boundary_opt_n_restarts,
                        seed=boundary_opt_seed,
                        k_bulk=boundary_opt_k,
                    )
                    # solve_qD_boundary's own residual diagnostics are a
                    # necessary but not sufficient gate: the fast bond-space-
                    # recursion estimate they rely on (same machinery
                    # optimize_boundary_operators uses) has been measured to
                    # disagree with the true isometry defect by up to nine
                    # orders of magnitude on some tensors, so the candidate
                    # gets the same independent, EXACT re-check
                    # (_validate_boundary_pair) as the optimizer candidates
                    # below before being trusted.
                    if (
                        diag_qD["cond1_residual"] < 1e-6
                        and diag_qD["cond2_residual"] < 1e-6
                        and diag_qD["bulk_defect_out_of_sample"] < 1e-6
                        and self._validate_boundary_pair(L_qD, R_qD)
                    ):
                        self.L_canonical, self.R_canonical = L_qD, R_qD
                        self.qD_general = True
                        self.qD_diagnostics = diag_qD
                        self.boundary_source = "qD_general"
                        qD_success = True
                except ValueError:
                    pass

            self.L, self.R = self.L_canonical, self.R_canonical

            if not qD_success and not self.mode1_exact:
                # Compared against the sigma/tau baseline (already
                # optimized) when one is available, not against I/d: the
                # free optimizer is adopted only if it beats what is
                # already available for free. When Assumption 1 failed and
                # nothing has succeeded yet, there is no canonical baseline
                # to beat (or warm-start from) -- adopt the optimizer's
                # result whenever it is merely feasible, and only a cold
                # start is attempted.
                if self.L_canonical is not None:
                    L2_inv = np.linalg.inv(self.L_canonical) @ np.linalg.inv(self.L_canonical)
                    R2_inv = np.linalg.inv(self.R_canonical) @ np.linalg.inv(self.R_canonical)
                    q_canonical = float(np.sqrt(np.real(np.trace(R2_inv @ L2_inv.T))))
                    starts = (
                        ("cold", None),
                        ("warm", (self.L_canonical, self.R_canonical)),
                    )
                else:
                    q_canonical = None
                    starts = (("cold", None),)
                self.q_baseline = q_canonical

                # Try both a cold start (identity, then random restarts -- the
                # optimizer's own default) and a warm start from the canonical
                # sigma/tau pair. Neither dominates: a warm start can converge
                # far faster and more reliably on tensors where the
                # unconstrained landscape is hard to navigate, a cold start
                # explores more of it. Both are run and the best VALIDATED
                # candidate is kept, mirroring the proxy/warm-started pattern
                # that worked for the q=D floor search.
                candidates = []
                for tag, warm in starts:
                    L_c, R_c, q_c, iso_c = optimize_boundary_operators(
                        A,
                        k=boundary_opt_k,
                        gauge=boundary_opt_gauge,
                        l_vec=l_vec,
                        r_vec=r_vec,
                        n_restarts=boundary_opt_n_restarts,
                        lam_iso=boundary_opt_lam_iso,
                        seed=boundary_opt_seed,
                        warm_start=warm,
                    )
                    candidates.append((tag, L_c, R_c, q_c, iso_c))

                self.boundary_opt_result = {
                    "candidates": {
                        tag: {"L": L_c, "R": R_c, "q": q_c, "iso_residual": iso_c}
                        for tag, L_c, R_c, q_c, iso_c in candidates
                    },
                    "q_baseline": q_canonical,
                }

                # q < 1 cannot be used by GOAA (arcsin(1/C) undefined for C < 1;
                # see optimize_boundary_operators' docstring); a large
                # SELF-REPORTED isometry residual already rules a candidate out.
                # Even a candidate that passes both of those still gets an
                # independent, EXACT re-check (_validate_boundary_pair) before
                # being trusted -- optimize_boundary_operators' own iso_residual
                # comes from the fast bond-space-recursion estimate, which has
                # been measured to disagree with the true isometry defect by up
                # to nine orders of magnitude on some tensors. A candidate that
                # looks fine by its own report but fails the real check is a
                # genuine reliability problem worth surfacing loudly -- distinct
                # from the ordinary, silent fallback used when the optimizer
                # honestly does not beat canonical.
                best = None
                saw_self_reported_ok_but_invalid = False
                for tag, L_c, R_c, q_c, iso_c in candidates:
                    feasible = q_c >= 1.0 - 1e-6 and iso_c < 1e-3
                    # 1e-6: the optimizer converges only to ~1e-9..1e-8 in q,
                    # so a candidate equal to canonical within that noise
                    # must not be rejected (same scale as `feasible`).
                    beats_canonical = q_canonical is None or q_c <= q_canonical + 1e-6
                    if not (feasible and beats_canonical):
                        continue
                    if self._validate_boundary_pair(L_c, R_c):
                        if best is None or q_c < best[3]:
                            best = (tag, L_c, R_c, q_c)
                    else:
                        saw_self_reported_ok_but_invalid = True

                self.boundary_opt_result["chosen"] = best[0] if best else None

                if saw_self_reported_ok_but_invalid:
                    warnings.warn(
                        "optimize_boundary_operators reported a low isometry "
                        "residual for at least one candidate boundary pair, "
                        "but an independent exact check (||V^dagger V - I||_F "
                        "on the real dense local isometries) rejected it. "
                        "Falling back to the certainly-unitary canonical "
                        "sigma/tau boundary for anything not otherwise "
                        "validated. This is the same bond-space-recursion-"
                        "vs-exact-check discrepancy documented for the q=D "
                        "floor search; if it recurs often on real tensors, "
                        "_bondspace_isometry_defect itself needs revisiting.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                if best is not None:
                    _, L_opt, R_opt, q_opt_chosen = best
                    self.L, self.R = L_opt, R_opt
                    self.boundary_source = "optimized_gauge" if boundary_opt_gauge else "optimized"

        if self.L is None or self.R is None:
            raise ValueError(
                "Assumption 1 is not satisfied for this tensor/boundary, and "
                "no free construction (boundary_mode1_exact/boundary_opt_k) "
                "succeeded either -- no valid boundary L, R could be "
                "constructed. Either fix the tensor/boundary, or pass "
                "boundary_opt_k (optionally with boundary_mode1_exact=True) "
                "to try a construction that does not require Assumption 1."
            )
        self.L_inv = np.linalg.inv(self.L)
        self.R_inv = np.linalg.inv(self.R)

        self.MergingOperator = self._get_merging_operator()
        self.q_unif = self._compute_q_unif()

        self._lcu_cache = None
        self._goaa_cache = None
        self._merging_unitary_cache = None
        self.ell = None
        self.C = None

        # --- 4. Eager precomputation of circuit-synthesis data ---
        # Everything below depends only on quantities already fixed above
        # (system size independent), so it is computed once here rather than
        # lazily/repeatedly inside synthesize().
        self._V_l, self._V_b, self._V_r = self.create_local_isometries()

        self._n_sys_per_site = int(np.ceil(np.log2(self._d)))
        self._n_bond_per_site = int(np.ceil(np.log2(self._D)))

        goaa_data = self._build_goaa_data()
        _, U_gate, U_dagger_gate, u_schedule, u_dagger_schedule, n_lcu, n_rot = (
            CircuitDecomposition._build_U_block_encoding(
                goaa_data,
                D_bond=self._D,
                n_bond_per_site=self._n_bond_per_site,
            )
        )
        self._uniform_merge_bundle = {
            "n_bond_per_site": self._n_bond_per_site,
            "n_lcu": n_lcu,
            "n_rot": n_rot,
            "ell": int(goaa_data["ell"]),
            "U_gate": U_gate,
            "U_dagger_gate": U_dagger_gate,
            "u_schedule": u_schedule,
            "u_dagger_schedule": u_dagger_schedule,
            "R_Phi_gate": CircuitDecomposition._make_reflection_zero(n_lcu + n_rot),
        }

        if debug:
            self._verify_lcu()
            self._verify_merging_unitary()

    def _validate_boundary_pair(
        self, L_cand: np.ndarray, R_cand: np.ndarray, tol: float = 1e-6
    ) -> bool:
        """
        Exact isometry check for a candidate ``(L, R)`` pair, independent of
        the bond-space estimate ``optimize_boundary_operators`` uses
        internally (``utils._bondspace_isometry_defect``), which can disagree
        with the true ``||V^dagger V - I||_F`` on some tensors.

        Reuses ``_build_isometry_unitary`` -- the exact same check
        ``synthesize()`` would eventually hit while
        building the real circuit -- so there is only one implementation
        of "is this actually an isometry" in the codebase, run here
        proactively instead of discovering a bad boundary late, deep
        inside gate construction (or, worse, never, if a caller only ever
        inspects ``q_unif``/``ell`` without synthesizing).
        """
        L_saved, R_saved = self.L, self.R
        self.L, self.R = L_cand, R_cand
        try:
            D_sys_pad = 2 ** int(np.ceil(np.log2(self._d)))
            D_bond_pad = 2 ** int(np.ceil(np.log2(self._D)))
            for V_tn in self.create_local_isometries():
                has_L = "ext_L" in V_tn.inds
                has_R = "ext_R" in V_tn.inds
                CircuitDecomposition._build_isometry_unitary(
                    V_tn,
                    label="boundary_opt_k validation",
                    D_sys_pad=D_sys_pad,
                    D_bond_pad_L=D_bond_pad if has_L else None,
                    D_bond_pad_R=D_bond_pad if has_R else None,
                    tol=tol,
                )
            return True
        except ValueError:
            return False
        finally:
            self.L, self.R = L_saved, R_saved

    def _site_isometry_tensor(self, k: int) -> qtn.Tensor:
        if k == 0:
            return self._V_l
        if k == self._N - 1:
            return self._V_r
        return self._V_b

    def _bond_qubit_count(self, bond_index: int) -> int:
        return self._n_bond_per_site

    def _bond_dimension(self, bond_index: int) -> int:
        return self._D

    def _merge_bundle(self, bond_index: int) -> dict:
        return self._uniform_merge_bundle

    def _isometry_tol(self) -> float:
        # Exact constructions: canonical boundaries are isometric for ANY
        # density matrices sigma, tau (the optimizer only chooses them), and
        # mode1_exact is closed form. "optimized"/"optimized_gauge" satisfy
        # the isometry only to penalty-method precision, and "qD_general"
        # accepts a 1e-6 solver residual, so they get the looser tolerance.
        if self.boundary_source in ("canonical", "canonical_sigma_tau", "mode1_exact"):
            return 1e-10
        # Dilation still snaps optimizer outputs to the nearest exact
        # isometry (see dilate_isometry_to_unitary), so a looser acceptance
        # tolerance here only affects which optimizer outputs are usable at
        # all, not the exactness of what synthesize() ultimately uses.
        return 1e-4

    def create_local_isometries(self) -> tuple[qtn.Tensor, qtn.Tensor, qtn.Tensor]:
        """
        Construct local MPU isometries for left-boundary, bulk, and right-boundary sites.

        The local tensor convention is

            A[p_in, p_out, bond_L, bond_R].

        Returns
        -------
        V_l:
            Left-boundary isometry with external leg ``ext_R``.
        V_b:
            Bulk isometry with external legs ``ext_L`` and ``ext_R``.
        V_r:
            Right-boundary isometry with external leg ``ext_L``.
        """
        A_data = self.A.data

        A1_tn, _ = get_mpo_site_tensors(1, A_data)
        A2_tn, _ = get_mpo_site_tensors(2, A_data)

        L_tn = qtn.Tensor(self.L, inds=["ext_L", "bond_1"], tags=["L"])
        R_tn = qtn.Tensor(self.R, inds=["ext_R", "bond_1"], tags=["R"])

        # Left boundary: l -- A -- R
        V_l = self.l_vec @ A1_tn @ R_tn

        # Bulk: L -- A -- R
        V_b = L_tn @ A2_tn @ R_tn.reindex({"bond_1": "bond_2"})

        # Right boundary: L -- A -- r
        V_r = L_tn @ A2_tn @ self.r_vec.reindex({f"bond_{self._N}": "bond_2"})

        return V_l, V_b, V_r

    def _compute_boundary_operators(
        self,
        sigma: np.ndarray | None = None,
        tau: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts the L2 and R2 boundary operators as matrices via TN contractions.
        L2 acts on the first bond index (bond_0), R2 on the last (bond_{N}).

        ``sigma``/``tau`` are the density matrices closing the physical legs of
        the first/last site. The textbook canonical choice is the maximally
        mixed state ``I/d`` (used when these are ``None``), but that is an
        arbitrary point in the feasible set, not an optimum: ``sigma`` and
        ``tau`` are free density matrices, and choosing them well already
        lowers ``q_unif`` without leaving the canonical construction (e.g. q
        53.67 -> 35.56 on a diagonal-phase MPU, ell 42 -> 28).
        :func:`utils.optimize_q_unif` finds them;
        ``UniformMPU.__init__`` uses it by default (``boundary_sigma_tau``),
        so the fallback when the *free* optimizer is unavailable or unhelpful
        is this optimized pair rather than the plain ``I/d`` one.
        """
        sigma = np.eye(self._d) / self._d if sigma is None else np.asarray(sigma)
        tau = np.eye(self._d) / self._d if tau is None else np.asarray(tau)

        # --- 1. Left Boundary Contraction (L2) ---
        l_vec_conj = self.l_vec.H.reindex({"bond_0": "bond_0_dag"}).retag({"l": "l*"})
        A_1, A_1_adj = get_mpo_site_tensors(1, self.A.data)

        L2 = A_1 & self.l_vec & l_vec_conj & A_1_adj
        L2 &= qtn.Tensor(sigma, inds=["p_in_1", "p_out_f_1"], tags=["σ"])

        # --- 2. Right Boundary Contraction (R2) ---
        r_vec_conj = self.r_vec.H.reindex({f"bond_{self._N}": f"bond_{self._N}_dag"}).retag(
            {"r": "r*"}
        )
        A_N, A_N_adj = get_mpo_site_tensors(self._N, self.A.data)

        R2 = A_N & self.r_vec & r_vec_conj & A_N_adj
        R2 &= qtn.Tensor(
            tau,
            inds=[f"p_in_{self._N}", f"p_out_f_{self._N}"],
            tags=["τ"],
        )

        # --- 3. Matrix Extraction & Dimensionality Checks ---
        L2_matrix = (L2 ^ ...).data
        R2_matrix = (R2 ^ ...).data

        trace_L2 = np.real(np.trace(L2_matrix))
        trace_R2 = np.real(np.trace(R2_matrix))

        if trace_L2 < 1e-15 or trace_R2 < 1e-15:
            raise ValueError(
                f"Unphysical non-positive trace detected "
                f"(L:{trace_L2:.2e}, R:{trace_R2:.2e}). "
                "Check MPO tensors and boundary vectors."
            )

        validation = [
            ("L2 Rank", L2_matrix.ndim == 2),
            ("R2 Rank", R2_matrix.ndim == 2),
            ("L2 Shape", L2_matrix.shape == (self._D, self._D)),
            ("R2 Shape", R2_matrix.shape == (self._D, self._D)),
        ]
        failed_checks = [label for label, passed in validation if not passed]
        if failed_checks:
            raise ValueError(
                f"Boundary operator contraction failed: {', '.join(failed_checks)}. "
                "Verify index naming logic in get_mpo_site_tensors."
            )

        # --- 4. Square root via stable hermitian decomposition ---
        # The site isometries contract the SECOND axis of L[ext, bond] and
        # R[ext, bond] with the bond, so the environment computed above is the
        # complex conjugate of the Gram matrix they need (equal for real
        # environments, wrong for complex ones).
        L = matrix_sqrt_hermitian(L2_matrix.conj())
        R = matrix_sqrt_hermitian(R2_matrix.conj())

        return L, R

    def _compute_q_unif(self) -> float:
        r"""Conditioning number :math:`q=\|R^{-\top}L^{-1}\|_F` of the chosen boundary.

        Equal to the LCU 1-norm :math:`C_0` of the merging operator used by
        the circuit, for every boundary source (see
        :func:`~TreeMPU.utils.conditioning_number`); for HPD boundaries it is
        :math:`\sqrt{\operatorname{tr}[R^{-2}(L^{-2})^\top]}` (thesis
        ``eq:q_unif_LR``).
        """
        if self.L_inv.shape != self.R_inv.shape:
            raise ValueError(f"Dimension mismatch: L2 {self.L_inv.shape} vs R2 {self.R_inv.shape}")
        return float(np.linalg.norm(self.R_inv.T @ self.L_inv))

    def _get_merging_operator(self) -> np.ndarray:
        # Call the parent method using the instance attributes
        return get_merging_operator(self.L_inv, self.R_inv)

    def _build_lcu_data(self):
        """
        Uniform case: the merge operator is the same at every bond.
        Computed once, cached, and reused for every merge in the tree.
        Uses the two-term SVD-based LCU decomposition (``_compute_lcu``).
        """
        if self._lcu_cache is None:
            coeffs, units, C = CircuitDecomposition._compute_lcu(self.MergingOperator)
            self._lcu_cache = (coeffs, units, C)
        return self._lcu_cache

    def _build_merging_unitary(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build and cache the (B, W_ctrl) block-encoding pair for the original
        (non-rotated) LCU decomposition of the merging operator, i.e. the
        raw ``U = B† W_ctrl B`` embedding of ``M`` before the extra
        rotation qubit that makes the GOAA exponent ell an integer.
        """
        if self._merging_unitary_cache is None:
            coef_0, unit_0, C_0 = self._build_lcu_data()
            self._merging_unitary_cache = CircuitDecomposition._compute_merging_unitary(
                coef_0, unit_0, C_0
            )
        return self._merging_unitary_cache

    def _verify_lcu(self) -> None:
        """Debug-only: check the original LCU decomposition reconstructs M."""
        D = self._D
        coef_0, unit_0, _ = self._build_lcu_data()
        verify_lcu(self.MergingOperator.reshape(D**2, D**2), coef_0, unit_0)

    def _verify_merging_unitary(self) -> None:
        """Debug-only: check (B, W_ctrl) correctly embeds M / C_0."""
        D = self._D
        _, _, C_0 = self._build_lcu_data()
        B, W_ctrl = self._build_merging_unitary()
        verify_merging_unitary(
            B=B,
            W_ctrl=W_ctrl,
            M_operator=self.MergingOperator.reshape(D**2, D**2),
            C=C_0,
            dim_system=D**2,
            dim_ancilla=B.shape[0],
        )

    def _build_goaa_data(self) -> dict:
        """Build (once) and cache the LCU + GOAA bundle of the merging operator.

        See :meth:`CircuitDecomposition._compute_goaa_bundle`; also sets
        ``self.C`` (rotated LCU norm) and ``self.ell``.
        """
        if getattr(self, "_goaa_cache", None) is None:
            self._goaa_cache = CircuitDecomposition._compute_goaa_bundle(self.MergingOperator)
            self.C = self._goaa_cache["C"]
            self.ell = self._goaa_cache["ell"]
        return self._goaa_cache


class _BlockLeaf:
    """A leaf gate known only through its <=2-qubit blocks (and its inverse's)."""

    def __init__(self, blocks, inv_blocks, num_qubits: int, name: str = "leaf"):
        self.blocks = [tuple(b) for b in blocks]
        self._inv_blocks = [tuple(b) for b in inv_blocks]
        self.num_qubits = num_qubits
        self.name = name

    def inverse(self) -> _BlockLeaf:
        return _BlockLeaf(self._inv_blocks, self.blocks, self.num_qubits, self.name + "_dg")


class _ParamsTree(CircuitDecomposition):
    """The synthesis tree of an N-site uniform chain rebuilt from
    :meth:`CircuitDecomposition.synthesis_cost_params` (see
    :meth:`CircuitDecomposition.analytic_synthesis_stats`)."""

    def __init__(self, params: dict, N: int, borrowed_ancillas: bool = True):
        self._N = N
        self._d = int(params["d"])
        self._nb = int(params["n_bond_per_site"])
        self._borrowed = borrowed_ancillas

        def leaf(entry, name):
            fwd, inv = entry[0], entry[1]
            width = 1 + max((q for blk in fwd for q in blk), default=0)
            return _BlockLeaf(fwd, inv, width, name)

        n_sys = int(np.ceil(np.log2(self._d)))
        self._site = {k: leaf(v, f"V_{k}") for k, v in params["site"].items()}
        for kind, n_bonds in (("left", 1), ("bulk", 2), ("right", 1)):
            self._site[kind].num_qubits = n_sys + n_bonds * self._nb
        self._bundle = dict(
            n_bond_per_site=self._nb,
            n_lcu=int(params["n_lcu"]),
            n_rot=int(params["n_rot"]),
            ell=int(params["ell"]),
            U_gate=_BlockLeaf([], [], int(params["u_width"]), "U"),
            u_schedule=[(leaf(e, "u"), tuple(e[2])) for e in params["u"]],
            u_dagger_schedule=[(leaf(e, "u_dg"), tuple(e[2])) for e in params["u_dag"]],
            R_Phi_gate=leaf(params["R_Phi"], "R_Phi"),
        )
        self._bundle["R_Phi_gate"].num_qubits = self._bundle["n_lcu"] + self._bundle["n_rot"]

    def _bond_qubit_count(self, bond_index: int) -> int:
        return self._nb

    def _merge_bundle(self, bond_index: int) -> dict:
        return self._bundle

    def _site_leaf(self, k, D_sys_pad, n_left_k, n_right_k, iso_cache):
        kind = "left" if k == 0 else ("right" if k == self._N - 1 else "bulk")
        return self._site[kind], k > 0, k < self._N - 1

    def _borrow_for_reflection(self, n_total, reflect_loc):
        if not self._borrowed:
            return 0, []
        return CircuitDecomposition._borrow_for_reflection(n_total, reflect_loc)
