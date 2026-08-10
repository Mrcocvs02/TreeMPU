from __future__ import annotations

from abc import ABC

import hashlib
import warnings

import numpy as np
import quimb.tensor as qtn
from scipy.linalg import qr
from typing import List

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator
from qiskit.circuit.library import (
    CRZGate,
    Isometry,
    RZGate,
    StatePreparation,
    UnitaryGate,
)
from qiskit.synthesis import (
    qs_decomposition,
    synth_mcx_1_dirty_kg24,
    synth_mcx_n_dirty_i15,
)

from .checks import (
    check_assumption_1,
    check_mpo_unitarity,
    check_chain_unitarity,
    check_full_rank,
    verify_lcu,
    verify_merging_unitary,
)
from .utils import (
    get_mpo_site_tensors,
    get_merging_operator,
    dilate_isometry_to_unitary,
    pad_block_diagonal_operator,
    compute_lcu_pad_indices,
    matrix_sqrt_hermitian,
    compute_chain_boundary_operators,
    optimize_boundary_operators,
    optimize_q_unif,
    optimize_chain_sigma_tau,
    to_canonical_form,
)


# Largest (LCU ancilla + rotation + both bond registers) width for which
# _build_U_block_encoding assembles W_ctrl as one dense UnitaryGate instead of
# attaching each LCU term with Gate.control(). 12 qubits is a 4096x4096
# complex matrix (~268 MB), comfortably built and synthesized once, while the
# per-term route at that width would first produce ~4^w gates per term and
# then control-convert every one of them.
DENSE_BLOCK_ENCODING_MAX_QUBITS = 12


class CircuitDecomposition(ABC):
    """
    Abstract Base Class for Matrix-Product Unitaries (MPUs).
    Enforces a strict API for compiling the MPU into a quantum circuit.
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
        in ``C``: ``M = |00><1|(R^-1 (x) L^-1)`` is rank one, so the previous
        inverse-DFT construction already produced ``D^2`` coefficients all of
        modulus ``sigma_1 / D^2``, summing to exactly ``sigma_1`` -- also
        optimal. Measured: mcz C = 2.0000 both ways, diag_phase(d=4)
        C = 53.6749 both ways, identical ``ell``. (For a generic full-rank
        ``M`` they do differ, ``||M||_op < ||M||_1``, measured up to 14x at
        dimension 32; that case does not arise here.)

        The decisive gain is the TERM COUNT: ``K`` drops from ``D^2`` to 2,
        so ``n_lcu = ceil(log2 K)`` drops to 1 and ``_build_U_block_encoding``
        makes 2 rather than ``D^2`` calls to ``Gate.control()``. Each such
        call deep-copies the controlled gate's definition, and that is what
        made the larger families unsynthesizable at all: profiled blowing
        past 3.5 GB inside ``.control()`` for subspace_rotation (D=5, 25
        terms) and lee_yang (D=14, 196 terms), both reduced to 2 terms here.
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

        ``build_w_ctrl=False`` skips the dense ``W_ctrl`` construction below
        (returning ``None`` in its place) and only builds ``B``. The actual
        circuit-building path, ``_build_U_block_encoding``, never touches
        ``W_ctrl`` -- by design, per its own docstring, it attaches each LCU
        term individually via ``qs_decomposition`` + ``.control()`` specifically
        to avoid ever materializing a matrix on the combined (ancilla) x
        (system) space. The only caller that needs the real dense ``W_ctrl``
        is the debug-only ``_verify_merging_unitary``, which gets it from its
        own independent, non-rotated call to this function (see
        ``_build_merging_unitary``) -- so the ``_compute_goaa_bundle`` /
        ``_build_goaa_data`` eager-precomputation call sites, which stash
        their ``W_ctrl`` in a cache nothing ever reads, should pass
        ``build_w_ctrl=False``. Skipping it matters in practice: dim_total
        (ancilla x system) grows with the bond dimension D (roughly D^2 x
        the rotation-qubit doubling), so the dense (dim_total x dim_total)
        matrix this builds can reach hundreds of GB for bond dimensions as
        modest as D~14, well before the actually-used, per-term construction
        becomes a problem.

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
            Ancilla state-preparation unitary satisfying B|0> = (1/C) sum_i sqrt(c_i) |i>.
        W_ctrl : np.ndarray, shape (dim_system * dim_ancilla, dim_system * dim_ancilla)
            Controlled unitary W_ctrl = sum_i (W_i)_S ⊗ |i><i|_A.
        """
        K = len(coefficients)
        dim_ancilla = 2 ** int(np.ceil(np.log2(K)))
        dim_system = int(unitaries[0].shape[0])

        # ------------------------------------------------------------------
        # Build B: unitary that prepares the ancilla state (Eq. A33)
        #   B|0>_A = (1/C) * sum_i sqrt(c_i) |i>_A
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
        #   W_ctrl = sum_i (W_i)_S ⊗ |i><i|_A
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
        every target reachable exactly. The old single-qubit code instead
        clipped the arcsin argument to 1, silently saturating ``phi`` at 45
        degrees and leaving GOAA inexact (``p_anc0 < 1``).

        Returns ``(ell, phis)``; ``len(phis)`` is the number of rotation
        qubits, which is 1 in the common case.
        """
        theta = np.arcsin(np.clip(1.0 / C, -1.0, 1.0))
        ell = int(np.ceil(np.pi / (4 * theta) - 0.5 - 1e-9))
        ell = max(ell, 0)

        C_prime = 1.0 / np.sin(np.pi / (4 * ell + 2))
        factor = max(C_prime / C, 1.0)

        # Smallest number of rotation qubits whose reach 2^(n/2) covers factor.
        n_rot = (
            1
            if factor <= np.sqrt(2) + 1e-12
            else int(np.ceil(2.0 * np.log2(factor) - 1e-12))
        )
        per_qubit = factor ** (1.0 / n_rot)
        # cos(phi) + sin(phi) = sqrt(2) sin(phi + pi/4) = per_qubit
        phi = float(np.arcsin(np.clip(per_qubit / np.sqrt(2), -1.0, 1.0)) - np.pi / 4)

        return ell, [phi] * n_rot

    @staticmethod
    def _closed_branch_structure(
        coef_0: list, phis: list, coef_ref: list
    ) -> list | None:
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
    def _borrow_for_reflection(
        n_total: int, reflect_loc: list[int]
    ) -> tuple[int, list[int]]:
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
    def _canonical_segment_qubits(seg: dict) -> list:
        """
        Return the canonical external qubit order of a tree segment.

        Segment order
        -------------
        ``sys | b_ct_R | b_ct_L | b_spectator | anc_storiche``

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
            + seg["anc_storiche"]
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
        big[tuple(slice(0, r) for r in raw_dims)] = mat.reshape(
            tuple(raw_dims) + (mat.shape[1],)
        )
        mat_pad = big.reshape(expected_rows, mat.shape[1])

        return dilate_isometry_to_unitary(mat_pad, tol=tol)

    @staticmethod
    def _tree_pair_counts(N: int) -> list[int]:
        """
        Return the number of binary merge pairs at each tree layer.

        Odd leftover segments are carried upward unchanged.

        Parameters
        ----------
        N:
            Number of initial sites.

        Returns
        -------
        list[int]
            Number of merge pairs per layer.
        """
        if N <= 0:
            raise ValueError("N must be positive.")

        counts = []
        n_segments = N

        while n_segments > 1:
            counts.append(n_segments // 2)
            n_segments = (n_segments + 1) // 2

        return counts

    @staticmethod
    def _tree_merge_plan(N: int) -> list[list[int]]:
        """
        Return, for each tree layer, the list of bond indices at which a
        pair-merge occurs.

        A "bond index" ``b`` (0-indexed) refers to the internal MPU bond
        between original sites ``b`` and ``b + 1``. Odd leftover segments are
        carried upward unchanged, mirroring :meth:`_tree_pair_counts`.

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
            Number of rotation ancilla qubits. Currently always 1.

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
        # 15702 gates at N=8), which is what motivated measuring instead of
        # assuming.
        B_gate = UnitaryGate(B_mat, label="B", check_input=False)
        B_dag_gate = UnitaryGate(B_mat.conj().T, label="B_dag", check_input=False)
        try:
            _prep = StatePreparation(np.asarray(B_mat)[:, 0]).definition.to_gate(
                label="B"
            )
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
        # See the module-level report for the K>2 families.
        #
        # Validation criterion, worth stating because getting it wrong cost
        # a debugging session: the test is NOT that Operator(U_gate) equals
        # the per-term route's. Two valid block-encodings of the same M/C
        # legitimately differ OUTSIDE the encoded block (they are different
        # unitary completions of the same block), and comparing the full
        # matrices reported a spurious 1.225 error. What has to hold is
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
                pad_block_diagonal_operator(W_i, D_bond, D_bond_pad, 2**n_rot)
                for W_i in unit
            ]
            dim_term = padded[0].shape[0]
            if n_lcu == 0:
                W_ctrl_full = padded[0]
            else:
                dim_anc = 2**n_lcu
                W_ctrl_full = np.zeros(
                    (dim_term * dim_anc, dim_term * dim_anc), dtype=complex
                )
                for i in range(dim_anc):
                    proj = np.zeros((dim_anc, dim_anc), dtype=complex)
                    proj[i, i] = 1.0
                    W_i = (
                        padded[i]
                        if i < len(padded)
                        else np.eye(dim_term, dtype=complex)
                    )
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
            W_i_padded = pad_block_diagonal_operator(
                W_i_raw, D_bond, D_bond_pad, 2**n_rot
            )
            term_gate = qs_decomposition(W_i_padded).to_gate(label=f"W_{i}")
            controlled_term = term_gate.control(n_lcu, ctrl_state=i)
            U_circ.append(controlled_term, list(dummy_lcu) + term_targets)
            u_schedule.append((controlled_term, tuple(lcu_idx + term_targets_idx)))

            term_dag_gate = qs_decomposition(W_i_padded.conj().T).to_gate(
                label=f"W_{i}_dag"
            )
            controlled_term_dag = term_dag_gate.control(n_lcu, ctrl_state=i)
            U_dagger_circ.append(controlled_term_dag, list(dummy_lcu) + term_targets)
            u_dagger_schedule.append(
                (controlled_term_dag, tuple(lcu_idx + term_targets_idx))
            )

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
            V_tn = self._site_isometry_tensor(k)

            sys_q = list(site_sys_regs[k])
            bond_q = list(site_bond_regs[k])

            if len(sys_q) != n_sys_per_site:
                raise ValueError(
                    f"Site {k}: expected {n_sys_per_site} system qubits, "
                    f"got {len(sys_q)}."
                )

            has_R = "ext_R" in V_tn.inds
            has_L = "ext_L" in V_tn.inds

            n_right_k = self._bond_qubit_count(k) if k < N - 1 else 0
            n_left_k = self._bond_qubit_count(k - 1) if k > 0 else 0

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
                raise ValueError(
                    f"Site {k}: local isometry has neither ext_R nor ext_L."
                )

            D_bond_pad_L = 2**n_left_k if has_L else None
            D_bond_pad_R = 2**n_right_k if has_R else None

            V_mat = CircuitDecomposition._build_isometry_unitary(
                V_tn,
                label=f"V_site_{k}",
                D_sys_pad=D_sys_pad,
                D_bond_pad_L=D_bond_pad_L,
                D_bond_pad_R=D_bond_pad_R,
                tol=self._isometry_tol(),
            )

            n_q = len(sys_q) + len(bond_q)
            leaf_gate = CircuitDecomposition._site_isometry_gate(
                V_mat, D_sys_pad, f"V{k}", iso_cache
            )

            seg = {
                "sys": sys_q,
                "b_ct_R": b_ct_R,
                "b_ct_L": b_ct_L,
                "b_spectator": [],
                "anc_storiche": [],
                "schedule": [(leaf_gate, tuple(range(n_q)))],
                "global_phase": 0.0,
            }
            seg["v_qubits"] = self._canonical_segment_qubits(seg)

            if len(set(map(id, seg["v_qubits"]))) != len(seg["v_qubits"]):
                raise ValueError(f"Duplicate qubits in initial segment {k}.")

            segments.append(seg)

        return segments

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
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: left segment is not canonical."
                )

            if seg_R["v_qubits"] != self._canonical_segment_qubits(seg_R):
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: right segment is not canonical."
                )

            bond = layer_bonds[pair_idx]
            bundle = self._merge_bundle(bond)

            n_bond_per_site = bundle["n_bond_per_site"]
            n_lcu = bundle["n_lcu"]
            n_rot = bundle["n_rot"]
            ell = bundle["ell"]
            u_schedule = bundle["u_schedule"]
            u_dagger_schedule = bundle["u_dagger_schedule"]
            R_Phi_gate = bundle["R_Phi_gate"]

            lcu_q = list(layer_anc_pairs[pair_idx][0])
            rot_q = list(layer_anc_pairs[pair_idx][1])

            if len(lcu_q) != n_lcu or len(rot_q) != n_rot:
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: ancilla register size mismatch."
                )

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
                + seg_L["anc_storiche"]
                + seg_R["anc_storiche"]
            )

            if len(set(map(id, v_qubits_pair))) != len(v_qubits_pair):
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: duplicate qubits in raw pair support."
                )

            q2local = {id(q): j for j, q in enumerate(v_qubits_pair)}

            def to_local(qlist: list, context: str) -> list[int]:
                missing = [q for q in qlist if id(q) not in q2local]
                if missing:
                    raise ValueError(
                        f"Layer {layer}, pair {pair_idx}, {context}: "
                        f"{len(missing)} qubits are not present in v_qubits_pair."
                    )
                return [q2local[id(q)] for q in qlist]

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

            pair_schedule = self._remap_schedule(
                seg_L["schedule"], left_local
            ) + self._remap_schedule(seg_R["schedule"], right_local)

            b_op_for_rpsi = seg_L["b_ct_L"] + seg_R["b_ct_R"] + b_spectator_pair
            b_op_local = to_local(b_op_for_rpsi, "open/spectator bonds")
            reflect_loc = lcu_loc + rot_loc + b_ct_local + b_op_local

            # R_Psi = V_pair Ref_0 V_pair^-1: invert pair_schedule by
            # reversing it and inverting each (leaf-level, hence cheap) gate.
            pair_inv_schedule = [(g.inverse(), qs) for g, qs in reversed(pair_schedule)]
            n_dirty, borrowed = CircuitDecomposition._borrow_for_reflection(
                n_vm, reflect_loc
            )
            ref0_gate = CircuitDecomposition._make_reflection_zero(
                len(reflect_loc), n_dirty=n_dirty
            )
            rpsi_schedule = (
                pair_inv_schedule
                + [(ref0_gate, tuple(reflect_loc + borrowed))]
                + pair_schedule
            )

            u_mapped = self._remap_schedule(u_schedule, u_signature_loc)
            u_dag_mapped = self._remap_schedule(u_dagger_schedule, u_signature_loc)

            # G = - U R_Psi U^dagger R_Phi; the global_phase += pi it
            # applies is tracked as a scalar below instead of on a
            # QuantumCircuit.
            g_schedule = (
                [(R_Phi_gate, tuple(lcu_loc + rot_loc))]
                + u_dag_mapped
                + rpsi_schedule
                + u_mapped
            )

            vm_schedule = pair_schedule + u_mapped + g_schedule * int(ell)
            vm_phase = seg_L["global_phase"] + seg_R["global_phase"] + int(ell) * np.pi

            new_seg = {
                "sys": sys_pair,
                "b_ct_R": seg_R["b_ct_R"],
                "b_ct_L": seg_L["b_ct_L"],
                "b_spectator": b_spectator_pair + b_ct_pair,
                "anc_storiche": (
                    seg_L["anc_storiche"] + seg_R["anc_storiche"] + lcu_q + rot_q
                ),
            }

            canonical = self._canonical_segment_qubits(new_seg)
            new_seg["v_qubits"] = canonical

            if len(set(map(id, canonical))) != len(canonical):
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: duplicate qubits in canonical segment."
                )

            raw_order_global = v_qubits_pair + lcu_q + rot_q

            if len(raw_order_global) != len(canonical):
                raise ValueError(
                    f"Layer {layer}, pair {pair_idx}: raw/canonical size mismatch "
                    f"{len(raw_order_global)} != {len(canonical)}."
                )

            canonical_index = {id(q): j for j, q in enumerate(canonical)}
            raw_targets_in_canonical = [
                canonical_index[id(q)] for q in raw_order_global
            ]

            new_seg["schedule"] = self._remap_schedule(
                vm_schedule, raw_targets_in_canonical
            )
            new_seg["global_phase"] = vm_phase

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

        Point T's dispatcher, for the one element where the choice is not
        already settled. Two candidates are BUILT and MEASURED, and the
        cheaper kept -- the same pattern as ``B`` in point S, and for the
        same reason: the right answer depends on the matrix, and measuring
        costs one synthesis that the content cache then pays for once.

        Candidate 1 -- ``UnitaryGate`` over the full dilated matrix. Point R
        established this is exactly ``qsd(opt_a1=True, opt_a2=False)`` and
        that no ``qs_decomposition`` flag combination beats it, so there is
        nothing further to try on this branch.

        Candidate 2 -- ``Isometry`` (Iten et al.) over the ``n_in``
        physical columns only, so the free completion is never synthesised.
        This is the branch point Q could not use: qiskit's ``Isometry``
        raises on these matrices, and point Y2 found why. It is not size
        (a random dense 5->8 isometry synthesises fine), not the tolerance
        (raising ``epsilon`` yields a silently WRONG circuit, error 1.80),
        and not sparsity as such -- densifying with ``H^{(x)n}`` reaches
        100% density and still fails. It is an exact algebraic degeneracy
        in Iten's column elimination, and mixing in 1e-6 of a random dense
        isometry is enough to break it.

        Hence the preconditioner: one RANDOM 2x2 unitary per wire. Random
        enough to break the degeneracy, and a tensor product of
        single-qubit gates, so undoing it costs one gate per wire --
        16 gates against 17845 on subspace. Measured on the bulk isometries:

            subspace   44522 -> 17845 blocks (2.49x), 16812 -> 8802 CX
            rand_d2D3  19076 ->  9263 (2.06x)
            rand_d4D4   4676 ->  2303 (2.03x)
            diag        2690 ->  2274 (1.18x)
            cnot          55 ->    59 (0.93x)   <- dense wins
            mcz           28 ->    59 (0.47x)   <- dense wins

        so the rule is "small ``n_in/dim`` favours the isometry", but it is
        applied by measurement, not by that heuristic.

        The fidelity check on the physical columns is MANDATORY, not
        defensive: point Q shows this branch is capable of returning a
        confidently wrong circuit. A candidate that does not reproduce the
        physical columns is discarded, never repaired.
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
            seed = int(
                hashlib.sha1(np.ascontiguousarray(V_mat).tobytes()).hexdigest()[:8], 16
            )
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
        are derived from this same flat block list (see ``_leaf_stats_pair``)
        without ever calling ``gate.inverse()``: inverting a sequence of
        operations reverses their order and inverts each individually
        (``(ABC)^-1 = C^-1 B^-1 A^-1``), and at <=2-qubit granularity a
        block's own inverse still occupies exactly the same qubits and
        still counts as exactly one block, so "reverse this flat list" is
        an *exact* leaf-level realization of "fully flatten gate.inverse()"
        -- not an approximation. The first version of this recursion called
        ``gate.inverse()`` and then re-walked the freshly built inverse
        gate's own ``.definition`` from scratch, which cost roughly as much
        again as the forward walk (measured: ~3.4s forward-only vs ~8-9s
        forward+inverse per bond on a d=2/D=2 random MPU) for no benefit.

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
        if gate.num_qubits <= 2 or gate.definition is None:
            blocks = [tuple(range(gate.num_qubits))]
        else:
            blocks = []
            defn = gate.definition
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
        return CircuitDecomposition._blocks_stats(
            CircuitDecomposition._leaf_blocks(gate, cache)
        )

    @staticmethod
    def _leaf_stats_pair(
        gate: Gate, cache: dict
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        ``(stats of gate, stats of gate's inverse)``.

        For an ordinary composite ``Gate``/``Instruction``, ``.inverse()``
        reduces to "reverse ``self.definition``'s instructions and invert
        each individually" -- which, fully flattened, is *exactly* reversing
        ``_leaf_blocks(gate, cache)`` (see that method's docstring), so the
        inverse's stats are derived from the one forward walk with no
        second recursion and no ``gate.inverse()`` call at all.

        A few gate classes encountered in this codebase override
        ``.inverse()`` to bypass ``.definition`` reversal entirely --
        ``UnitaryGate`` (the V_i site isometries, some LCU padding pieces)
        goes through ``.adjoint()``, returning a *new* ``UnitaryGate`` over
        the conjugate-transposed matrix, independently re-synthesized by
        ``qs_decomposition`` when its own ``.definition`` is later built;
        ``MCPhaseGate``/``MCXGate`` (found nested inside
        ``qs_decomposition``'s own output) return ``MCPhaseGate(-lambda,
        ...)`` / a fresh self-inverse ``MCXGate`` -- structurally the
        *same* decomposition as the forward gate, not a reversed one.
        Whether a gate uses the default mechanism at all could be checked
        generically via ``type(gate).inverse is Instruction.inverse`` (true
        for plain ``Gate`` instances such as ``QuantumCircuit.to_gate()``
        output, false for all of the above) -- falling back to the slow,
        exact path (build the real ``.inverse()``, walk it fresh) only for
        the classes that fail this check.

        That was tried on a d=2/D=2 random MPU at N=4: computing the
        ``MCPhaseGate``/``MCXGate`` cases exactly moved the reported depth
        by all of 0.009% (9271815 -> 9272615) while giving back essentially
        the entire speedup this function exists for (28.9s -> 58.4s per
        bond, matching the pre-optimization cost) -- because these two
        classes are common inside a qs_decomposition tree, and building
        their ``.definition`` fresh is exactly the expensive walk being
        avoided. Given depth is already a documented, much larger
        approximation from the tree-layer aggregate composition in
        ``_merge_layer_stats`` (~20% at N=4 in the same test), chasing an
        extra 0.01% here at 2x the cost is not worth it. So the fast path
        (reverse the forward blocks) is used unconditionally, including for
        these overriding classes -- gate_count is unaffected either way
        (all of them preserve the forward gate's own leaf count; only
        depth gets this small, quantified extra slack). ``UnitaryGate``
        specifically was also checked in isolation and made no measurable
        difference in the cases tested here (site isometries are <=2
        qubits at d=2, so they never even reach this branch), but the same
        reasoning would apply if it ever did.

        This is unrelated to, and does not contradict, the fact that
        ``u_schedule`` and ``u_dagger_schedule`` (two *independently*
        qs_decomposition-synthesized circuits for U and U†) have been
        measured with different gate counts (by a factor of ~2): that
        asymmetry is about comparing two separately-built decompositions of
        different matrices (W_i vs W_i.conj().T) -- the same underlying
        phenomenon as the ``UnitaryGate`` case above, in fact -- whereas
        this function computes the structural inverse of one specific
        already-built gate, which the reversal identity pins down exactly
        for gate count regardless of which override (if any) applies.
        """
        blocks = CircuitDecomposition._leaf_blocks(gate, cache)
        fwd = CircuitDecomposition._blocks_stats(blocks)
        inv = CircuitDecomposition._blocks_stats(list(reversed(blocks)))
        return fwd, inv

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

    @staticmethod
    def _schedule_stats_pair(
        schedule: list[tuple[Gate, tuple[int, ...]]], cache: dict
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        ``(forward stats, inverse stats)`` of a flat schedule, where the
        "inverse" is the schedule reversed with every leaf individually
        inverted -- exactly what _merge_layer builds when it needs the
        adjoint of an already-assembled block. Since leaf adjoints have
        their own (different) cost, this is measured, not assumed.
        """
        fwd_last: dict[int, int] = {}
        inv_last: dict[int, int] = {}
        fwd_count = 0
        inv_count = 0
        stats = [
            (CircuitDecomposition._leaf_stats_pair(gate, cache), qs)
            for gate, qs in schedule
        ]
        for (f, _), qs in stats:
            start = max((fwd_last.get(q, 0) for q in qs), default=0)
            for q in qs:
                fwd_last[q] = start + f[0]
            fwd_count += f[1]
        for (_, inv), qs in reversed(stats):
            start = max((inv_last.get(q, 0) for q in qs), default=0)
            for q in qs:
                inv_last[q] = start + inv[0]
            inv_count += inv[1]
        return (
            (max(fwd_last.values()) if fwd_last else 0, fwd_count),
            (max(inv_last.values()) if inv_last else 0, inv_count),
        )

    @staticmethod
    def _stat_seq(
        components: list[tuple[int, int, "list[int] | tuple[int, ...]"]],
    ) -> tuple[int, int]:
        """
        Compose several (depth, count, qubits) components -- in program
        order, each treated as one opaque block of known depth/count -- into
        the (depth, count) of their sequence, via exact per-qubit greedy
        scheduling: components on genuinely disjoint qubits can overlap
        (run in parallel, so total depth is only their max, not their sum),
        while components sharing any qubit are forced sequential.

        This, not naive addition, is what makes composing a merge tree
        exact: e.g. a pair merge's two children act on disjoint qubits (so
        the pair's depth is max(depth_L, depth_R), not depth_L + depth_R),
        while R_Psi's reflection genuinely does touch every qubit at its
        level and so forces full sequential composition there.
        """
        last_layer: dict[int, int] = {}
        total_count = 0
        for depth, count, qubits in components:
            qubits = list(qubits)
            start = max((last_layer.get(q, 0) for q in qubits), default=0)
            end = start + depth
            for q in qubits:
                last_layer[q] = end
            total_count += count
        depth = max(last_layer.values()) if last_layer else 0
        return depth, total_count

    def _bond_u_stats(
        self, bundle: dict, cache: dict
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """
        ``(u, u_inv, u_dagger, u_dagger_inv)`` for one bond, each a
        ``(depth, count)`` in that bond's own u_schedule/u_dagger_schedule
        local numbering. Computed once per bond and cached on the bundle --
        this is the only place that ever walks a bond's LCU terms, and it
        costs O(K), never O(N).

        Note ``u_dagger`` (built directly from each LCU term's conjugate
        transpose) and ``u_inv`` (the u_schedule reversed with each leaf
        inverted) are the *same operator* but not the same circuit: qiskit
        synthesizes them independently, with materially different costs.
        _merge_layer_stats uses whichever one is actually emitted at each
        position.
        """
        if "_u_stats" not in bundle:
            bundle["_u_stats"], bundle["_u_inv_stats"] = (
                CircuitDecomposition._schedule_stats_pair(bundle["u_schedule"], cache)
            )
            bundle["_u_dagger_stats"], bundle["_u_dagger_inv_stats"] = (
                CircuitDecomposition._schedule_stats_pair(
                    bundle["u_dagger_schedule"], cache
                )
            )
        return (
            bundle["_u_stats"],
            bundle["_u_inv_stats"],
            bundle["_u_dagger_stats"],
            bundle["_u_dagger_inv_stats"],
        )

    def _make_segment_list_stats(
        self,
        n_sys_per_site: int,
        D_sys_pad: int,
        cache: dict,
    ) -> tuple[list[dict], int]:
        """
        Stats-only counterpart of _make_segment_list: identical index
        bookkeeping (sys/b_ct_R/b_ct_L/canonical v_qubits), but using plain
        integers throughout instead of qiskit Qubit objects (no actual
        circuit is ever built here), and each segment carries a "stat"
        ``(depth, count)`` instead of a Gate/schedule. A segment's own stat
        is always exact over its *entire* v_qubits range (every merge step
        below composes with a component that spans the full local width, so
        this invariant holds all the way up the tree -- see
        estimate_synthesis_stats's docstring).
        """
        iso_cache: dict = {}
        N = self._N
        segments: list[dict] = []
        next_qubit = 0

        for k in range(N):
            V_tn = self._site_isometry_tensor(k)

            has_R = "ext_R" in V_tn.inds
            has_L = "ext_L" in V_tn.inds

            n_right_k = self._bond_qubit_count(k) if k < N - 1 else 0
            n_left_k = self._bond_qubit_count(k - 1) if k > 0 else 0

            sys_q = list(range(next_qubit, next_qubit + n_sys_per_site))
            next_qubit += n_sys_per_site
            bond_q = list(range(next_qubit, next_qubit + n_right_k + n_left_k))
            next_qubit += n_right_k + n_left_k

            if has_R and has_L:
                b_ct_R = bond_q[:n_right_k]
                b_ct_L = bond_q[n_right_k:]
            elif has_R:
                b_ct_R, b_ct_L = bond_q[:], []
            elif has_L:
                b_ct_R, b_ct_L = [], bond_q[:]
            else:
                raise ValueError(
                    f"Site {k}: local isometry has neither ext_R nor ext_L."
                )

            D_bond_pad_L = 2**n_left_k if has_L else None
            D_bond_pad_R = 2**n_right_k if has_R else None

            V_mat = CircuitDecomposition._build_isometry_unitary(
                V_tn,
                label=f"V_site_{k}",
                D_sys_pad=D_sys_pad,
                D_bond_pad_L=D_bond_pad_L,
                D_bond_pad_R=D_bond_pad_R,
                tol=self._isometry_tol(),
            )
            leaf_gate = CircuitDecomposition._site_isometry_gate(
                V_mat, D_sys_pad, f"V{k}", iso_cache
            )
            stat, stat_inv = CircuitDecomposition._leaf_stats_pair(leaf_gate, cache)

            seg = {
                "sys": sys_q,
                "b_ct_R": b_ct_R,
                "b_ct_L": b_ct_L,
                "b_spectator": [],
                "anc_storiche": [],
                "stat": stat,
                "stat_inv": stat_inv,
            }
            seg["v_qubits"] = self._canonical_segment_qubits(seg)
            segments.append(seg)

        return segments, next_qubit

    def _merge_layer_stats(
        self,
        segments: list[dict],
        layer_bonds: list[int],
        cache: dict,
        next_qubit: int,
    ) -> tuple[list[dict], int]:
        """
        Stats-only counterpart of _merge_layer: identical index
        bookkeeping, but every gate-composition step is replaced by an
        O(1) aggregate composition on (depth, count) pairs via _stat_seq,
        never touching the underlying leaf gate count.

        Each segment carries two aggregates, ``stat`` and ``stat_inv``,
        because inverting a block is *not* free: _merge_layer builds
        V_pair^-1 by reversing the schedule and inverting each leaf, and a
        qiskit ``UnitaryGate`` adjoint is re-synthesized from scratch with
        its own (measurably different) cost. Propagating both directions
        keeps the recursion O(1) per merge while staying exact on counts.
        """
        new_segments: list[dict] = []
        pair_idx = 0
        n_seg = len(segments)

        for i in range(0, n_seg - 1, 2):
            seg_L = segments[i]
            seg_R = segments[i + 1]

            bond = layer_bonds[pair_idx]
            bundle = self._merge_bundle(bond)

            # n_bond_per_site = bundle["n_bond_per_site"]
            n_lcu = bundle["n_lcu"]
            n_rot = bundle["n_rot"]
            ell = bundle["ell"]

            lcu_q = list(range(next_qubit, next_qubit + n_lcu))
            next_qubit += n_lcu
            rot_q = list(range(next_qubit, next_qubit + n_rot))
            next_qubit += n_rot

            b_ct_pair = seg_L["b_ct_R"] + seg_R["b_ct_L"]
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
                + seg_L["anc_storiche"]
                + seg_R["anc_storiche"]
            )

            q2local = {q: j for j, q in enumerate(v_qubits_pair)}
            left_local = [q2local[q] for q in seg_L["v_qubits"]]
            right_local = [q2local[q] for q in seg_R["v_qubits"]]
            b_ct_local = [q2local[q] for q in b_ct_pair]

            n_v = len(v_qubits_pair)
            n_vm = n_v + n_lcu + n_rot

            lcu_loc = list(range(n_v, n_v + n_lcu))
            rot_loc = list(range(n_v + n_lcu, n_vm))
            u_signature_loc = lcu_loc + rot_loc + b_ct_local

            b_op_for_rpsi = seg_L["b_ct_L"] + seg_R["b_ct_R"] + b_spectator_pair
            b_op_local = [q2local[q] for q in b_op_for_rpsi]
            reflect_loc = lcu_loc + rot_loc + b_ct_local + b_op_local

            u_st, u_inv_st, u_dag_st, u_dag_inv_st = self._bond_u_stats(bundle, cache)
            R_Phi_st, R_Phi_inv_st = CircuitDecomposition._leaf_stats_pair(
                bundle["R_Phi_gate"], cache
            )
            _n_dirty, _ = CircuitDecomposition._borrow_for_reflection(n_vm, reflect_loc)
            ref0_gate = CircuitDecomposition._make_reflection_zero(
                len(reflect_loc), n_dirty=_n_dirty
            )
            ref0_st, ref0_inv_st = CircuitDecomposition._leaf_stats_pair(
                ref0_gate, cache
            )

            seq = CircuitDecomposition._stat_seq
            v_loc = list(range(n_v))
            vm_loc = list(range(n_vm))
            phi_loc = lcu_loc + rot_loc

            # Every block below is tracked in both directions: `_fwd` as this
            # construction emits it, `_inv` as the *next* layer up will
            # emit it inside its own V_pair^-1 (schedule reversed, each leaf
            # individually inverted). The two genuinely differ -- see
            # _leaf_stats_pair -- so the inverse cannot be reused for the
            # forward cost, which is the assumption this used to make.
            pair_fwd = seq(
                [
                    (*seg_L["stat"], left_local),
                    (*seg_R["stat"], right_local),
                ]
            )
            pair_inv = seq(
                [
                    (*seg_R["stat_inv"], right_local),
                    (*seg_L["stat_inv"], left_local),
                ]
            )

            # R_Psi = V_pair Ref_0 V_pair^-1 (applied right-to-left: V_pair^-1
            # first, then Ref_0, then V_pair -- same convention as
            # _merge_layer). Its own inverse replays the same
            # V_pair^-1-first / V_pair-last order, only Ref_0 itself flips
            # to its inverse in between.
            rpsi_fwd = seq(
                [
                    (*pair_inv, v_loc),
                    (*ref0_st, reflect_loc),
                    (*pair_fwd, v_loc),
                ]
            )
            rpsi_inv = seq(
                [
                    (*pair_inv, v_loc),
                    (*ref0_inv_st, reflect_loc),
                    (*pair_fwd, v_loc),
                ]
            )

            # G = - U R_Psi U^dagger R_Phi
            g_fwd = seq(
                [
                    (*R_Phi_st, phi_loc),
                    (*u_dag_st, u_signature_loc),
                    (*rpsi_fwd, vm_loc),
                    (*u_st, u_signature_loc),
                ]
            )
            g_inv = seq(
                [
                    (*u_inv_st, u_signature_loc),
                    (*rpsi_inv, vm_loc),
                    (*u_dag_inv_st, u_signature_loc),
                    (*R_Phi_inv_st, phi_loc),
                ]
            )

            vm_fwd = seq(
                [(*pair_fwd, v_loc), (*u_st, u_signature_loc)]
                + [(*g_fwd, vm_loc)] * int(ell)
            )
            vm_inv = seq(
                [(*g_inv, vm_loc)] * int(ell)
                + [(*u_inv_st, u_signature_loc), (*pair_inv, v_loc)]
            )

            new_seg = {
                "sys": sys_pair,
                "b_ct_R": seg_R["b_ct_R"],
                "b_ct_L": seg_L["b_ct_L"],
                "b_spectator": b_spectator_pair + b_ct_pair,
                "anc_storiche": (
                    seg_L["anc_storiche"] + seg_R["anc_storiche"] + lcu_q + rot_q
                ),
                "stat": vm_fwd,
                "stat_inv": vm_inv,
            }
            new_seg["v_qubits"] = self._canonical_segment_qubits(new_seg)

            new_segments.append(new_seg)
            pair_idx += 1

        if n_seg % 2 == 1:
            new_segments.append(segments[-1])

        return new_segments, next_qubit

    def estimate_synthesis_stats(self) -> dict:
        """
        Estimate the fully-decomposed circuit's (depth, gate_count,
        num_qubits) in O(N) time, without ever materializing the flat
        instruction schedule that synthesize builds (let alone an
        actual QuantumCircuit) -- computed instead as a small aggregate
        ``(depth, count)`` pair per tree segment, composed from its
        children's aggregates in O(1) per merge via ``_stat_seq``.

        Why this exists alongside synthesize()
        -----------------------------------------------------------------
        The true (fully flattened) gate count of this construction can be
        astronomically large: R_Psi = V_pair Ref0 V_pair^-1 replays the
        *entire* lower-layer content twice (forward and inverse) every time
        it's used, and it's used ``ell`` times per tree layer -- so the
        total size compounds *multiplicatively* across the O(log N) tree
        layers (matching the paper's own claimed depth scaling
        ``N^(1+log2 q)``: for q > 1 this is genuinely superlinear, not a
        synthesize bug). Materializing that flat sequence just to
        count it, as synthesize does, therefore costs as much as the
        true (possibly enormous) gate count itself.

        The fix is the same idea as synthesize's own flat schedule,
        applied one level higher: instead of a flat list of leaf gates, keep
        only a ``(depth, count)`` summary per tree segment. ``Seq`` (two
        segments merging), ``Repeat`` (``ell`` copies of the Grover
        iterate), and inversion (used for ``V_pair^-1``) all compose via
        O(1) arithmetic on these summaries -- inversion doesn't even need
        recomputing, since inverting changes neither depth nor gate count.
        The one exact subtlety is that composing two summaries is *not*
        naive addition: components on disjoint qubits (e.g. a pair merge's
        two children) can run in parallel (depth = max, not sum), so
        ``_stat_seq`` does genuine per-qubit greedy scheduling -- just on a
        handful of aggregates per merge, never on the underlying leaves.

        Net effect: this scales to any N cheaply (each of the O(N) tree
        nodes costs O(1)), even when the *reported* depth/gate_count
        themselves are enormous, which is exactly the regime where
        synthesize() becomes intractable to run at all.

        Accuracy: gate_count and num_qubits are exact; depth is a rigorous
        upper bound whose looseness grows with the number of tree layers
        -----------------------------------------------------------------
        ``gate_count`` is a pure sum over leaves, so coarsening a subtree
        into one ``(depth, count)`` aggregate loses no counting information
        -- verified exact (to the gate) against a full flat-schedule
        measurement on both a small trusted case (N=2 UniformMPU) and a
        larger random MPU (N=4, d=2/D=2), in both cases after fixing two
        bugs this cross-check exposed: (1) inverting a block is not free --
        a ``UnitaryGate`` adjoint is re-synthesized by qiskit, not derived,
        and can cost a different gate count than the forward gate, so every
        aggregate here is tracked in both directions (``stat``/``stat_inv``,
        see ``_merge_layer_stats``); (2) the measurement scripts' own
        ``id(gate)``-keyed caches must keep the gate object alive alongside
        its cached stats, or CPython recycling a freed gate's address
        silently returns a stale, unrelated entry.

        ``depth`` has no such guarantee: treating a composed subtree as one
        opaque block (as ``_stat_seq`` does when using it as a component in
        a further composition) implicitly forces every qubit in that
        block's footprint to stay "busy" for the block's *own* worst-case
        depth, even if some of those qubits actually finished earlier and
        had slack an exact per-qubit schedule could have used. That slack
        is real but invisible once a block has been coarsened, and it's
        lost again at every layer built on top of it, so the overestimate
        compounds with the number of merge layers (measured: +1.7% at N=2,
        one merge layer; +22% at N=4, two merge layers, on the same random
        d=2/D=2 MPU). Recovering exact depth would mean carrying a full
        per-qubit finish-time profile through every merge instead of one
        scalar -- exactly the O(N)-breaking cost this method exists to
        avoid. Treat ``depth`` as a safe upper bound on true circuit depth,
        not a tight estimate, when N spans many tree layers.

        Returns
        -------
        dict with keys ``depth``, ``gate_count``, ``num_qubits``.
        """
        N = self._N
        n_sys_per_site = int(np.ceil(np.log2(self._d)))
        D_sys_pad = 2**n_sys_per_site

        cache: dict = {}
        segments, next_qubit = self._make_segment_list_stats(
            n_sys_per_site, D_sys_pad, cache
        )

        plan = self._tree_merge_plan(N)
        for layer_bonds in plan:
            segments, next_qubit = self._merge_layer_stats(
                segments, layer_bonds, cache, next_qubit
            )

        if len(segments) != 1:
            raise RuntimeError(
                f"Tree synthesis ended with {len(segments)} segments instead of 1."
            )

        depth, gate_count = segments[0]["stat"]
        return dict(depth=depth, gate_count=gate_count, num_qubits=next_qubit)

    # ------------------------------------------------------------------
    # Analytic estimator -- same recursion as estimate_synthesis_stats, but
    # as a pure function of a handful of measured constants. See
    # synthesis_cost_params / analytic_synthesis_stats.
    # ------------------------------------------------------------------

    @staticmethod
    def _ref0_stats_analytic(
        n_qubits: int, n_dirty: int, cache: dict
    ) -> tuple[int, int]:
        """
        ``(depth, gate_count)`` of ``_make_reflection_zero(n, n_dirty)``
        WITHOUT building it, for the two borrowed-ancilla branches.

        Why a closed form is mandatory here rather than a nicety: the
        analytic estimator's whole point is reaching N far beyond what the
        map-building estimator can do, and the top tree layer's reflection
        spans essentially the whole register -- at N = 2^16 that is O(10^5)
        qubits, where instantiating a qiskit circuit is simply not possible.
        Every other parameter of the model is N-independent; this is the
        only one that grows, so it is the only one that must be a formula.

        The formulas are not fitted, they are the exact measured values,
        checked against a real synthesis at EVERY n from 8 to 80 and spot
        checked up to 256:

            i15  (n_dirty >= max(n_ctrl-2, 1)):
                depth = 16n - 29                      count = 18n - 25
            kg24 (0 < n_dirty below that):
                depth = 20n - 37 (n even)             count = 30n - 75
                        20n - 41 (n odd)

        Below n = 8 the affine form does not hold (the transpiler's
        peephole pass still bites), so those few n fall through to an
        actual synthesis -- cheap precisely because n is small.

        **The kg24 depth is affine in n only at fixed PARITY**, and that
        cost four hours: the first measurement sampled n as 4,5,6,7,8,10,
        12,16,20,... -- past 8 only even values -- so "20n - 37" fitted
        every sampled point perfectly and was wrong by exactly 4 on every
        odd n. It first showed up as a 200-gate depth discrepancy against
        the map estimator that grew by (2*ell+1) per tree layer, i.e.
        looking exactly like a broken recursion rather than a mis-sampled
        constant. When fitting a synthesis cost, sample consecutive sizes.

        The ``n_dirty = 0`` branch is deliberately NOT modelled: it is the
        O(n^2) plain-MCX path, and ``_borrow_for_reflection`` only ever
        returns 0 dirty ancillas when ``n_ctrl < 3``, i.e. n <= 3. It
        therefore never occurs at large n and is left to synthesis.

        Branch selection replicates ``_make_reflection_zero``'s own
        conditions exactly rather than approximating them -- picking the
        wrong branch would silently mis-cost by 1.5x.
        """
        n_ctrl = n_qubits - 1
        if n_dirty > 0 and n_qubits >= 8:
            if n_dirty >= max(n_ctrl - 2, 1):
                return (16 * n_qubits - 29, 18 * n_qubits - 25)
            if n_ctrl >= 3:
                return (
                    20 * n_qubits - (37 if n_qubits % 2 == 0 else 41),
                    30 * n_qubits - 75,
                )
        key = ("ref0", n_qubits, n_dirty)
        hit = cache.get(key)
        if hit is not None:
            return hit[1]
        gate = CircuitDecomposition._make_reflection_zero(n_qubits, n_dirty=n_dirty)
        stat = CircuitDecomposition._leaf_stats(gate, cache)
        cache[key] = (gate, stat)
        return stat

    def synthesis_cost_params(self, cache: dict | None = None) -> dict:
        """
        The N-independent constants the analytic estimator needs.

        Everything expensive in ``estimate_synthesis_stats`` lives here and
        is paid ONCE: the three distinct site isometries (left boundary,
        bulk, right boundary), the bond's U / U-dagger schedules, and
        R_Phi. Extracting them requires ``N >= 3`` (a chain shorter than
        that has no bulk site to measure) but the resulting dict predicts
        any N.

        Only valid for a chain whose bonds are all the same -- every bond
        must return the same merge bundle and the same qubit count. That is
        checked, not assumed: a non-uniform chain would need one parameter
        set per bond and the O(log N) collapse would not apply.
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

        n_sys_per_site = int(np.ceil(np.log2(self._d)))
        D_sys_pad = 2**n_sys_per_site

        site: dict[str, tuple] = {}
        site_inv: dict[str, tuple] = {}
        for kind, k in (("left", 0), ("bulk", 1), ("right", N - 1)):
            V_tn = self._site_isometry_tensor(k)
            has_R = "ext_R" in V_tn.inds
            has_L = "ext_L" in V_tn.inds
            V_mat = CircuitDecomposition._build_isometry_unitary(
                V_tn,
                label=f"V_site_{k}",
                D_sys_pad=D_sys_pad,
                D_bond_pad_L=(2**nb) if has_L else None,
                D_bond_pad_R=(2**nb) if has_R else None,
                tol=self._isometry_tol(),
            )
            # NB: the number of physical columns uses the bond DIMENSION,
            # not its number of qubits. Using `nb` (the qubit count) here
            # gave n_in = 12 instead of 20 on subspace, and the fidelity
            # guard only compared those 12: physical columns 12..19 stayed
            # free and the circuit was SILENTLY wrong.
            gate = CircuitDecomposition._site_isometry_gate(
                V_mat, D_sys_pad, f"V{k}", cache
            )
            site[kind], site_inv[kind] = CircuitDecomposition._leaf_stats_pair(
                gate, cache
            )

        u, u_inv, u_dag, u_dag_inv = self._bond_u_stats(bundle, cache)
        R_Phi, R_Phi_inv = CircuitDecomposition._leaf_stats_pair(
            bundle["R_Phi_gate"], cache
        )

        return dict(
            n_sys_per_site=n_sys_per_site,
            n_bond_per_site=nb,
            n_lcu=bundle["n_lcu"],
            n_rot=bundle["n_rot"],
            ell=int(bundle["ell"]),
            site=site,
            site_inv=site_inv,
            u=u,
            u_inv=u_inv,
            u_dag=u_dag,
            u_dag_inv=u_dag_inv,
            R_Phi=R_Phi,
            R_Phi_inv=R_Phi_inv,
        )

    @staticmethod
    def analytic_synthesis_stats(
        params: dict,
        N: int,
        pool_ancillas: bool = False,
        pool_levels: int | None = None,
        borrowed_ancillas: bool = True,
    ) -> dict:
        """
        ``estimate_synthesis_stats`` as a pure function of ``params`` and N.

        This is the *same* recursion as ``_merge_layer_stats``, with the
        per-segment qubit LISTS replaced by their lengths. That replacement
        is exact, not a simplification: every overlap the greedy scheduler
        resolves in ``_stat_seq`` is fixed by the construction's layout, so
        each composition collapses to arithmetic (derived below), and the
        only quantity that ever depended on the actual indices --
        ``_borrow_for_reflection``'s idle count -- is just
        ``n_vm - n_reflect``, since ``reflect_loc``'s entries are distinct
        and all lie inside the segment.

        Per merge, writing ``d``/``c`` for depth/count:

            pair     = (max(d_L, d_R),          c_L + c_R)      [disjoint]
            rpsi     = d_pair_inv + d_ref0 + d_pair
            g        = d_RPhi + d_Udag + d_rpsi + d_U
            vm       = d_pair + d_U + ell * d_g
            vm_inv   = ell * d_g_inv + d_Uinv + d_pair_inv

        The children are the only ``max`` in the whole recursion (they sit
        on disjoint qubits); everything else overlaps on the bond-carrier
        qubits and therefore serialises. **The ``(2l+1)`` that produces the
        observed ``depth ~ N^log2(2l+1)`` law is visible right there**: each
        ``g`` replays ``rpsi``, which replays the child twice, so a layer
        multiplies the child's depth by ``1 + 2*ell = 2l+1`` and its count
        by ``2*(2l+1)`` -- hence the gate exponent being the depth exponent
        plus one. The law is the solution of this recursion, not a fit to
        the data.

        Cost: O(N) trivial arithmetic with a memoised merge, against
        ``estimate_synthesis_stats``'s O(N) *syntheses*. On subspace the
        latter is ~50 s per site.

        Segments are represented as
        ``(n_sys, n_bctR, n_bctL, n_spec, n_anc, stat, stat_inv)`` --
        mirroring ``_merge_layer_stats``'s five qubit groups exactly, in
        the same order, because ``b_spectator`` and ``anc_storiche``
        accumulate differently and both feed the reflection's width.
        """
        if N < 1:
            raise ValueError("N must be positive.")
        nsys = params["n_sys_per_site"]
        nb = params["n_bond_per_site"]
        n_lcu, n_rot, ell = params["n_lcu"], params["n_rot"], params["ell"]

        # Segment states are memo keys, so every (depth, count) inside them
        # must be hashable. A params dict that has been through JSON comes
        # back with lists instead of tuples, which fails only once the
        # first memo lookup happens -- deep inside the recursion, with an
        # error that says nothing about serialisation.
        def pair(x):
            return (int(x[0]), int(x[1]))

        u, u_inv = pair(params["u"]), pair(params["u_inv"])
        u_dag, u_dag_inv = pair(params["u_dag"]), pair(params["u_dag_inv"])
        rphi, rphi_inv = pair(params["R_Phi"]), pair(params["R_Phi_inv"])
        site = {k: pair(v) for k, v in params["site"].items()}
        site_inv = {k: pair(v) for k, v in params["site_inv"].items()}

        cache: dict = {}
        memo: dict = {}

        def merge(L, R):
            key = (L, R)
            hit = memo.get(key)
            if hit is not None:
                return hit

            (sL, bRL, bLL, spL, anL, stL, stiL, lvlL) = L
            (sR, bRR, bLR, spR, anR, stR, stiR, lvlR) = R

            n_v = (sL + bRL + bLL + spL + anL) + (sR + bRR + bLR + spR + anR)
            n_vm = n_v + n_lcu + n_rot
            n_bct_pair = bRL + bLR
            n_bop = bLL + bRR + spL + spR
            n_reflect = n_lcu + n_rot + n_bct_pair + n_bop

            n_ctrl = n_reflect - 1
            if n_ctrl < 3 or not borrowed_ancillas:
                # `borrowed_ancillas=False` reproduces the pre-point-U
                # behaviour (plain MCX, O(n^2)); it exists so the baseline
                # can be priced with the same recursion as the optimised
                # version, rather than compared against a remembered number.
                n_dirty = 0
            else:
                n_dirty = min(n_vm - n_reflect, max(n_ctrl - 2, 1))
            ref0 = CircuitDecomposition._ref0_stats_analytic(n_reflect, n_dirty, cache)

            # The two children sit on disjoint qubits, so they overlap and
            # the pair's depth is their max -- this is the ONLY place in the
            # recursion where anything runs in parallel, and it is what
            # makes depth grow like N^log2(2l+1) instead of like the gate
            # count.
            #
            # ``pool_ancillas`` gives that up deliberately: if every merge
            # draws its lcu/rot register from one shared pool, two sibling
            # subtrees no longer act on disjoint qubits and must serialise.
            # The ancillas genuinely can be shared -- they return to |0...0>
            # at machine precision after each merge (measured: p = 1.0000,
            # purity 1.0, everything else 1e-31) -- so this is a real
            # option, not an approximation. It is priced here rather than
            # assumed cheap.
            # ``pool_levels`` pools only the lowest merge layers. That is
            # where nearly all the registers are (layer 0 alone holds N/2 of
            # the N-1 merges) and where depth is cheapest, so the frontier
            # is far better than the all-or-nothing version: the qubit
            # saving is roughly geometric in the number of pooled levels
            # while the depth cost is a constant factor, not a factor of N.
            lvl = max(lvlL, lvlR)
            share = pool_ancillas or (pool_levels is not None and lvl < pool_levels)
            if share:
                d_pair = stL[0] + stR[0]
                d_pairi = stiL[0] + stiR[0]
            else:
                d_pair = max(stL[0], stR[0])
                d_pairi = max(stiL[0], stiR[0])
            c_pair = stL[1] + stR[1]
            c_pairi = stiL[1] + stiR[1]

            d_rpsi = d_pairi + ref0[0] + d_pair
            c_rpsi = c_pairi + ref0[1] + c_pair

            d_g = rphi[0] + u_dag[0] + d_rpsi + u[0]
            c_g = rphi[1] + u_dag[1] + c_rpsi + u[1]
            d_gi = u_inv[0] + d_rpsi + u_dag_inv[0] + rphi_inv[0]
            c_gi = u_inv[1] + c_rpsi + u_dag_inv[1] + rphi_inv[1]

            vm = (d_pair + u[0] + ell * d_g, c_pair + u[1] + ell * c_g)
            vmi = (ell * d_gi + u_inv[0] + d_pairi, ell * c_gi + u_inv[1] + c_pairi)

            out = (
                sL + sR,
                bRR,
                bLL,
                spL + spR + n_bct_pair,
                anL + anR + n_lcu + n_rot,
                vm,
                vmi,
                lvl + 1,
            )
            memo[key] = out
            return out

        segs = []
        for k in range(N):
            kind = "left" if k == 0 else ("right" if k == N - 1 else "bulk")
            segs.append(
                (
                    nsys,
                    nb if k < N - 1 else 0,
                    nb if k > 0 else 0,
                    0,
                    0,
                    site[kind],
                    site_inv[kind],
                    0,
                )
            )

        while len(segs) > 1:
            new = [merge(segs[i], segs[i + 1]) for i in range(0, len(segs) - 1, 2)]
            if len(segs) % 2 == 1:
                new.append(segs[-1])
            segs = new

        depth, gate_count = segs[0][5]
        # N sites' system qubits, one bond register per side of every
        # internal bond, and the lcu+rot ancillas: one register per merge
        # (N-1 of them) as allocated today, or a single shared register when
        # pooled.
        if pool_ancillas:
            n_anc_regs = 1
        elif pool_levels:
            # Merges at layer h number ceil(N / 2^(h+1)); pooling the lowest
            # ``pool_levels`` of them collapses each such layer to one
            # register, leaving the rest one-per-merge.
            n_anc_regs, rem = 0, N
            h = 0
            while rem > 1:
                m = rem // 2
                n_anc_regs += 1 if h < pool_levels else m
                rem = m + (rem % 2)
                h += 1
            n_anc_regs = max(n_anc_regs, 1)
        else:
            n_anc_regs = N - 1
        return dict(
            depth=depth,
            gate_count=gate_count,
            num_qubits=(N * nsys + 2 * (N - 1) * nb + n_anc_regs * (n_lcu + n_rot)),
        )

    def synthesize(
        self,
        unitary_label: str | None = None,
    ) -> tuple[
        QuantumCircuit,
        list[QuantumRegister],
        list[QuantumRegister],
        list[list[tuple[QuantumRegister, QuantumRegister]]],
        list,
    ]:
        """
        Synthesize the MPU into a Qiskit quantum circuit.

        Builds the circuit via a flat Python-level instruction "schedule" --
        a list of ``(leaf gate, local qubit indices)`` tuples composed by
        list concatenation and index remapping -- materializing the actual
        ``QuantumCircuit`` only once, at the very end.

        Why it is built this way
        -------------------------
        An earlier implementation built each tree layer's merged segment as
        an actual qiskit ``Gate`` (via ``.to_gate()`` on a circuit containing
        the previous layer's already-composite Gates, with ``ell``
        repetitions of the Grover iterate among them). Empirically,
        ``.to_gate()``'s cost on such a circuit scales with the *total*
        nested size of everything inside it, not just that circuit's own
        instruction count: wrapping a merge containing the Grover iterate
        repeated ``ell`` times was observed to take seconds and multiple GB
        even for a small (N=2, D=4) chain, well before any actual N-scaling
        could even begin -- and, for a merging operator with enough LCU
        terms (e.g. a generic, non-fine-tuned random MPU), simulating the
        resulting single opaque top-level instruction could fail to
        terminate in any practical time at all, even at N=2. Every entry
        ever stored in the schedule here, by contrast, is a genuine leaf
        (built by exactly one ``qs_decomposition``/``UnitaryGate`` call,
        never re-wrapped into a bigger Gate), so e.g. inverting
        ``pair_schedule`` for R_Psi's ``V_pair^-1`` stays proportional to the
        schedule's own length no matter how large the tree above it grows.
        This flat-schedule construction replaced the ``.to_gate()``-based one
        entirely once cross-checked against it on small chains (same gates,
        same global phase, same qubit registers/ordering).

        Parameters
        ----------
        unitary_label:
            Optional label reserved for external diagnostics. This
            implementation does not write diagnostics to disk.

        Returns
        -------
        qc_global:
            Quantum circuit implementing the MPU decomposition.
        site_sys_regs:
            Physical system registers, one per site.
        site_bond_regs:
            Bond/virtual ancilla registers, one per site.
        anc_regs:
            Tree ancilla registers. Each layer contains pairs
            ``(lcu_register, rot_register)``.
        all_anc_qubits:
            Flat list of all non-system qubits, useful for postselection or
            tests.
        """
        del unitary_label

        N = self._N

        n_sys_per_site = int(np.ceil(np.log2(self._d)))
        D_sys_pad = 2**n_sys_per_site

        site_sys_regs = [
            QuantumRegister(n_sys_per_site, name=f"sys_{k}") for k in range(N)
        ]

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

        for layer, layer_bonds in enumerate(plan):
            layer_regs = []

            for p, bond in enumerate(layer_bonds):
                bundle = self._merge_bundle(bond)
                lcu_r = QuantumRegister(bundle["n_lcu"], name=f"lcu_l{layer}_p{p}")
                rot_r = QuantumRegister(bundle["n_rot"], name=f"rot_l{layer}_p{p}")
                layer_regs.append((lcu_r, rot_r))

            anc_regs.append(layer_regs)

        all_regs = []

        for k in range(N):
            all_regs.append(site_sys_regs[k])
            all_regs.append(site_bond_regs[k])

        for layer_regs in anc_regs:
            for lcu_r, rot_r in layer_regs:
                all_regs.append(lcu_r)
                all_regs.append(rot_r)

        qc_global = QuantumCircuit(*all_regs, name="mpu_circuit_lean")

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
            raise RuntimeError(
                f"Tree synthesis ended with {len(segments)} segments instead of 1."
            )

        master_segment = segments[0]

        if master_segment["v_qubits"] != self._canonical_segment_qubits(master_segment):
            raise RuntimeError("Final master segment violates canonical ordering.")

        for gate, local_qubits in master_segment["schedule"]:
            qc_global.append(
                gate, [master_segment["v_qubits"][i] for i in local_qubits]
            )
        qc_global.global_phase += master_segment["global_phase"]

        all_anc_qubits = []

        for b_reg in site_bond_regs:
            all_anc_qubits.extend(list(b_reg))

        for layer_regs in anc_regs:
            for lcu_r, rot_r in layer_regs:
                all_anc_qubits.extend(list(lcu_r))
                all_anc_qubits.extend(list(rot_r))

        return qc_global, site_sys_regs, site_bond_regs, anc_regs, all_anc_qubits

    def exact_synthesis_stats(self) -> dict:
        """
        Compute the fully-decomposed circuit's TRUE (depth, gate_count,
        num_qubits) -- exact, unlike estimate_synthesis_stats()'s cheap but
        approximate upper bound on depth -- by building the same flat
        instruction "map" synthesize() builds (via
        _make_segment_list/_merge_layer, unchanged), then reading
        (depth, gate_count) directly off that schedule with an exact
        per-qubit greedy scan, instead of materializing a QuantumCircuit
        and measuring that.

        Why this is exact where estimate_synthesis_stats() is not
        -----------------------------------------------------------------
        estimate_synthesis_stats() coarsens each tree segment into a single
        scalar ``(depth, count)`` aggregate the moment it's built, so once
        two segments are merged, any per-qubit slack inside either one
        (some qubit finishing before the segment's own worst case) is gone
        for good -- every later composition sees only the coarsened
        summary. This method never coarsens: ``_merge_layer`` builds
        the actual flat sequence of leaf-level ``(gate, local_qubits)``
        entries exactly as synthesize() would, and the scan below
        walks that sequence with a genuine per-qubit greedy schedule (the
        same methodology as ``_blocks_stats``/``cached_unroll``), so every
        leaf's true effect on every qubit's finish time is accounted for
        individually. It was confirmed empirically (see
        estimate_synthesis_stats's docstring) that a cheap, reusable,
        O(1)-per-merge closed form for this does not exist in general
        (a bond-level gate's response to a non-uniform external start does
        not reduce to "wait for the max, then add a fixed per-qubit
        offset" -- verified false by direct test); this method sidesteps
        that obstruction entirely by paying for the real answer instead of
        approximating it.

        Cost: polynomial in N, not O(N)
        -----------------------------------------------------------------
        ``master_segment["schedule"]`` is itself already the same "map"
        synthesize() builds (a flat, but *compact*, list of ``(gate,
        local_qubit_tuple)`` pairs assembled via list concatenation and
        ``ell``-repetition -- e.g. 1134 top-level entries at N=4 on a
        d=2/D=2 random MPU, not yet the true 24.5M leaf count) via
        ``_merge_layer``, unchanged; this does include real
        ``gate.inverse()`` calls to build R_Psi's ``V_pair^-1`` (measured:
        212 of them at that same N=4, still cheap, ~7s total), so this
        method does not avoid qiskit's own Gate machinery entirely, only
        the *final* ``QuantumCircuit`` materialization step. The dominant
        cost, measured directly, is the scan itself: walking each of those
        ~1000 top-level entries' own cached ``_leaf_blocks`` down to the
        true ~24.5M leaves and updating a per-qubit finish-time dict for
        each one (~44s of the ~46s total at N=4) -- this part is
        irreducible, since an exact answer requires touching every true
        leaf once, and matches (to within measurement noise) the cost of
        synthesize() followed by a separate flatten-and-measure pass.

        Memory: the ~2.7GB peak RSS at N=4 is real, and mostly comes from
        ``_leaf_blocks``'s keepalive cache retaining every gate it ever
        touches (and hence each one's own qiskit-internal ``.definition``,
        a full nested ``QuantumCircuit``) for the scan's whole duration --
        confirmed by a variant that discarded a gate's ``._definition``
        right after extracting its blocks, which cut peak RSS by ~40%
        (2.7GB -> 1.6GB) with identical depth/gate_count. That variant is
        NOT used here, though: it also tried dropping the keepalive
        reference to the gate itself (reasoning that
        ``master_segment["schedule"]`` already keeps every relevant gate
        alive), and that crashed with an ``IndexError`` from a stale
        ``id()`` collision -- proof that a gate's own ``.definition`` does
        not reliably keep its sub-instructions alive on its own; the cache
        entry for *every* level of the recursion is what does that (see
        ``_leaf_blocks``'s docstring). A safe version of the memory
        optimization is possible in principle (clear a gate's own
        ``._definition`` after use while still keeping the gate object
        itself cached), but only for gates that are provably local to this
        one call -- the bond-level gates in ``bundle[...]`` are cached and
        reused across other calls (including by ``synthesize()``,
        which needs their ``.definition`` to stay valid to build an actual
        circuit), so clearing those would risk silently corrupting later
        use. Not implemented here pending a safe way to distinguish the
        two cases.

        Net effect: this does not change which N are reachable in
        principle (still polynomial in the true schedule size, matching
        the paper's own ``N^(1+log2 q)`` depth-scaling claim for q>1), it
        only pushes the practical ceiling further out than
        synthesize() + full-circuit measurement did.

        Returns
        -------
        dict with keys ``depth``, ``gate_count``, ``num_qubits``.
        """
        N = self._N

        n_sys_per_site = int(np.ceil(np.log2(self._d)))
        D_sys_pad = 2**n_sys_per_site

        site_sys_regs = [
            QuantumRegister(n_sys_per_site, name=f"sys_{k}") for k in range(N)
        ]

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

        for layer, layer_bonds in enumerate(plan):
            layer_regs = []
            for p, bond in enumerate(layer_bonds):
                bundle = self._merge_bundle(bond)
                lcu_r = QuantumRegister(bundle["n_lcu"], name=f"lcu_l{layer}_p{p}")
                rot_r = QuantumRegister(bundle["n_rot"], name=f"rot_l{layer}_p{p}")
                layer_regs.append((lcu_r, rot_r))
            anc_regs.append(layer_regs)

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
            raise RuntimeError(
                f"Tree synthesis ended with {len(segments)} segments instead of 1."
            )

        master_segment = segments[0]
        num_qubits = len(master_segment["v_qubits"])

        cache: dict = {}
        last_layer: dict[int, int] = {}
        total_count = 0
        for gate, local_qubits in master_segment["schedule"]:
            for leaf_qs in CircuitDecomposition._leaf_blocks(gate, cache):
                global_qs = tuple(local_qubits[i] for i in leaf_qs)
                start = max((last_layer.get(q, 0) for q in global_qs), default=0)
                end = start + 1
                for q in global_qs:
                    last_layer[q] = end
                total_count += 1

        depth = max(last_layer.values()) if last_layer else 0
        return dict(depth=depth, gate_count=total_count, num_qubits=num_qubits)


class GeneralMPU(CircuitDecomposition):
    """
    Encapsulates a heterogeneous (site-dependent) Matrix-Product Unitary.
    Defined by a strictly ordered sequence of local bulk tensors A_1, ..., A_N.

    Unlike :class:`UniformMPU`, there is no translation invariance to exploit:
    every quantity that was system-size-independent there (boundary
    operators, merging operator, MPU conditioning number, LCU/GOAA data,
    block-encoding gate) is here indexed by the internal bond it belongs to.
    All of it is computed exactly once, in ``__init__``, via a single left
    sweep and a single right sweep over the chain (Lemma 1 of Styliaris et
    al.), rather than recomputed per-bond from scratch or lazily during
    synthesis.
    """

    def __init__(
        self,
        tensors_A: List[np.ndarray],
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
        """
        Args:
            tensors_A: List of N site-dependent bulk tensors [A_0, ..., A_{N-1}],
                each of shape (d, d, D_left_k, D_right_k), sharing the same
                physical dimension d.
            l_vec: Left boundary vector, contracted into A_0's left leg.
            r_vec: Right boundary vector, contracted into A_{N-1}'s right leg.
            tol: Numerical tolerance for structural checks.
            canonicalize: Bring the chain into the MPU canonical form of the
                paper's Definition 3 before doing anything else (default).
                This is what makes Assumption 1 unnecessary for nonuniform
                MPUs: the paper states the site-dependent L_j, R_k keep their
                properties "even without the need for Assumption 1" once the
                MPU is in canonical form, where Eq. (26) gives L_k = 1 and
                R_k = diag(Schmidt values) > 0. It also replaces the caller's
                bond dimensions by the minimal ones, so ``self.tensors`` and
                ``self._bond_dims`` may be smaller than what was passed in.
                Set False only to inspect a specific representation as given;
                a rank-deficient one will then be rejected, as before.
            canonical_tol: Relative singular-value threshold used by the
                canonicalization to decide which bond directions are null.
            boundary_sigma_tau: If True, before attempting boundary_opt_p
                (below), lift the naive maximally-mixed I/d chain sweep to
                utils.optimize_chain_sigma_tau -- the GeneralMPU analogue of
                UniformMPU's boundary_sigma_tau baseline, generalizing
                utils.optimize_q_unif's single-tensor sigma/tau optimization
                to a free, per-site density matrix at every site within
                boundary_sigma_tau_max_sites of each chain boundary (thesis
                section "Partial Optimization"). Every free sigma/tau is a
                density matrix by construction, so unlike boundary_opt_p
                there is no isometry constraint that can fail: this can only
                ever match or improve each bond's q_bond over the naive I/d
                baseline, per bond (kept independently for L and R), never
                make it worse. This becomes the new baseline that
                boundary_opt_p compares against and falls back to, so the
                effective fallback order is free optimization (boundary_opt_p)
                -> optimized sigma/tau -> naive I/d.
            boundary_sigma_tau_max_sites: Caps how many sites nearest each
                boundary get a free sigma/tau (``None`` = every site).
                Because the sweep that builds every bond's L2/R2 already
                costs O(N) regardless, this only bounds the *parameter
                count* being optimized -- useful to keep the optimization
                itself tractable on long chains.
            boundary_sigma_tau_n_restarts, boundary_sigma_tau_seed: Passed
                through to utils.optimize_chain_sigma_tau.
            boundary_opt_p: If given, attempt "partial optimization"
                (thesis section "Partial Optimization") at every
                power-of-two window length up to ``2**boundary_opt_p``: for
                bond b, whenever the ``b+1`` sites to its left already reach
                the chain's own left boundary, R[b] is replaced by a
                directly-optimized boundary tensor (passing that exact
                window of site-dependent tensors straight to
                utils.optimize_boundary_operators, which evaluates the
                window's isometry constraint via bond-space-only
                transfer-matrix recursions -- cost linear in the window
                length, never exponential, so no window-size cap is needed
                beyond ``2**boundary_opt_p`` itself); symmetrically for L[b]
                using the sites to its right and r_vec. Bonds whose window
                does not reach a chain boundary within ``2**boundary_opt_p``
                sites are left at their (possibly sigma/tau-optimized, see
                boundary_sigma_tau above) baseline value -- unlike the
                uniform-bulk case, there is no cheap way to anchor an
                *interior* window without recursively depending on a lower
                level's own optimized boundary, which this does not attempt.
                A replacement is only kept if it improves q_bond and reaches
                q_bond >= 1 with a small isometry residual (same guards as
                UniformMPU's boundary_opt_k), so this can only ever reduce
                (never increase) self.q.
            boundary_opt_gauge:
                Must be False whenever boundary_opt_p is given (raises
                otherwise): every window boundary_opt_p optimizes is anchored
                to l_vec or r_vec, hence (with canonicalize=True) has a
                trivial bond on that side, and gauge requires both ends to
                share one bond dimension.
            boundary_opt_n_restarts, boundary_opt_lam_iso, boundary_opt_seed:
                Passed through to optimize_boundary_operators.
        """
        N = len(tensors_A)

        if N < 2:
            raise NotImplementedError(
                "GeneralMPU.synthesize currently requires N >= 2."
            )

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

        L2_list, R2_list, residual = compute_chain_boundary_operators(
            self.tensors, l_vec, r_vec
        )
        check_chain_unitarity(residual, tol=1e-6)

        self.boundary_source = ["canonical"] * (N - 1)

        # --- 3.1. Optional sigma/tau-optimized baseline (see
        # boundary_sigma_tau docstring above). Every candidate is a valid
        # density matrix by construction, so there is no isometry
        # constraint to fail here -- only whether it actually beats the
        # naive I/d pair, decided independently per bond (and per L vs R,
        # since the two sweeps are independent). Wrapped in try/except like
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
            except Exception:
                # optimize_chain_sigma_tau is a JAX optimization: if it
                # fails to converge or hits a numerical issue, fall back to
                # the naive I/d pair rather than aborting construction.
                pass

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
                        self.R[b] = R_opt
                        self.R_inv[b] = np.linalg.inv(R_opt)
                        self.boundary_source[b] = (
                            "optimized_gauge" if boundary_opt_gauge else "optimized"
                        )

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
                        self.L[b] = L_opt
                        self.L_inv[b] = np.linalg.inv(L_opt)
                        self.boundary_source[b] = (
                            "optimized_gauge" if boundary_opt_gauge else "optimized"
                        )

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
        self._bond_qubits = [
            int(np.ceil(np.log2(self._bond_dims[b]))) for b in range(N - 1)
        ]

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
                    "R_Phi_gate": CircuitDecomposition._make_reflection_zero(
                        n_lcu + n_rot
                    ),
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
                L_tn = qtn.Tensor(
                    self.L[k - 1], inds=["ext_L", f"bond_{k}"], tags=["L"]
                )
                A_tn = L_tn @ A_tn

            if k < N - 1:
                R_tn = qtn.Tensor(
                    self.R[k], inds=["ext_R", f"bond_{k + 1}"], tags=["R"]
                )
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
        # See UniformMPU._isometry_tol: at least one bond's L/R came from
        # optimize_boundary_operators, which only satisfies the isometry up
        # to penalty-method precision, not machine epsilon.
        return 1e-4


class UniformMPU(CircuitDecomposition):
    """
    Encapsulates a translationally invariant Matrix-Product Unitary.
    Defined by a single repeating bulk tensor A.
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
        boundary_opt_k: int | None = None,
        boundary_opt_gauge: bool = False,
        boundary_opt_n_restarts: int = 10,
        boundary_opt_lam_iso: float = 1e6,
        boundary_opt_seed: int = 0,
    ):
        """
        Args:
            A: The bulk numpy array (d, d, D, D).
            l_vec: Left boundary vector.
            r_vec: Right boundary vector.
            N: Total number of physical sites.
            tol: Numerical tolerance for structural checks.
            debug: if True gives info and sanity checks.
            light: Controls how far the O(N)-length unitarity check runs.
                ``check_mpo_unitarity`` walks n = 1..N, growing and
                re-contracting two tensor networks at every step, so its cost
                grows far faster than the chain itself: profiled at 0.4s/9MB
                for N=32 but 32s/1.5GB for N=256 (~99.6% of the whole
                constructor), and it exhausts several GB before N=512. That
                makes it, not the synthesis or any downstream estimator, the
                binding constraint on reachable N.

                For a translationally invariant MPU the check is also largely
                redundant: unitarity follows from a *local* isometry condition
                on the single bulk tensor plus the boundary vectors, so
                violations show up at short lengths -- walking all the way to
                N re-verifies the same algebraic identity with progressively
                larger contractions.

                - ``False`` (default): check all lengths 1..N. Unchanged
                  behaviour; keep it whenever N is small enough to afford it.
                - ``True``: check lengths 1..min(N, 8).
                - ``int k``: check lengths 1..min(N, k), to pick the cutoff
                  explicitly.

                This only ever *shortens* the check; it is never skipped
                entirely, so a tensor that is not a valid MPU is still
                rejected. What is given up is the guarantee that unitarity was
                machine-verified at the exact length being synthesized.
            boundary_opt_k: If given, lift the canonical sigma/tau boundary
                construction and instead search for invertible L, R directly
                minimizing the MPU conditioning number q, subject to the
                isometry constraint at lengths 1..boundary_opt_k
                (utils.optimize_boundary_operators; thesis sections
                "Lifting the Canonical Injectivity Constraint" and "Partial
                Optimization"). Every downstream quantity (MergingOperator,
                q_unif, LCU/GOAA data, the synthesized circuit) is then
                built from the optimized L, R instead of the canonical ones.
                Silently falls back to the canonical construction if the
                optimizer cannot reach q >= 1 (see optimize_boundary_operators'
                docstring for when this happens) or fails to beat it.
            boundary_opt_gauge: Additionally optimize a unitary gauge
                transformation (thesis section "Gauge optimization and the
                q_unif orbit"); explores a strictly larger feasible set.
            boundary_opt_n_restarts, boundary_opt_lam_iso, boundary_opt_seed:
                Passed through to optimize_boundary_operators.
        """
        if N < 2:
            raise NotImplementedError(
                "UniformMPU.synthesize currently requires N >= 2."
            )

        # --- 1. Theoretical Validations (Fail-Fast Logic) ---

        # Check global unitary property via local left-canonical isometry
        # condition. `light` caps the length walked here -- see the docstring
        # for why this is the constructor's dominant cost and what capping it
        # gives up.
        if light is False:
            n_check = N
        else:
            n_check = min(N, 8 if light is True else int(light))
            if n_check < 1:
                raise ValueError(
                    f"light must cap the unitarity check at >= 1 site, got {light}."
                )
        self.unitarity_checked_up_to = n_check
        check_mpo_unitarity(n_check, A, l_in=l_vec, r_in=r_vec, tol=tol)

        # Check injectivity / minimal bond dimension at the boundaries
        s_left, s_right = check_assumption_1(A, l_vec, r_vec, tol=1e-12)

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
        # Baseline: sigma/tau optimized, NOT the plain I/d pair. The maximally
        # mixed choice is just one arbitrary feasible point; optimizing the two
        # density matrices stays entirely inside the canonical construction
        # (so the isometry holds at every length by construction, with no
        # constraint to enforce) yet already lowers q -- measured 53.67 ->
        # 35.56 on diagonal-phase (ell 42 -> 28) and 10.0 -> 9.0 on
        # subspace-rotation. Everything downstream therefore falls back to
        # this, never to I/d.
        self.L_id, self.R_id = self._compute_boundary_operators()
        self.sigma_opt = self.tau_opt = None
        if boundary_sigma_tau:
            try:
                sg, tu, _ = optimize_q_unif(A, l_vec, r_vec)
                L_st, R_st = self._compute_boundary_operators(sigma=sg, tau=tu)
                if (
                    np.all(np.isfinite(L_st))
                    and np.all(np.isfinite(R_st))
                    and np.linalg.matrix_rank(L_st) == self._D
                    and np.linalg.matrix_rank(R_st) == self._D
                ):
                    self.L_canonical, self.R_canonical = L_st, R_st
                    self.sigma_opt, self.tau_opt = sg, tu
                else:
                    self.L_canonical, self.R_canonical = self.L_id, self.R_id
            except Exception:
                # optimize_q_unif is a JAX optimization: if it fails to
                # converge, fall back to I/d rather than aborting construction.
                self.L_canonical, self.R_canonical = self.L_id, self.R_id
        else:
            self.L_canonical, self.R_canonical = self.L_id, self.R_id

        self.L, self.R = self.L_canonical, self.R_canonical
        self.boundary_source = (
            "canonical_sigma_tau" if self.sigma_opt is not None else "canonical"
        )

        if boundary_opt_k is not None:
            # Compared against the sigma/tau baseline (already optimized),
            # not against I/d: the free optimizer is adopted only if it
            # beats what is already available for free.
            L2_inv = np.linalg.inv(self.L_canonical) @ np.linalg.inv(self.L_canonical)
            R2_inv = np.linalg.inv(self.R_canonical) @ np.linalg.inv(self.R_canonical)
            q_canonical = float(np.sqrt(np.real(np.trace(R2_inv @ L2_inv.T))))
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
            for tag, warm in (
                ("cold", None),
                ("warm", (self.L_canonical, self.R_canonical)),
            ):
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
                if not (
                    q_c >= 1.0 - 1e-6 and iso_c < 1e-3 and q_c <= q_canonical + 1e-9
                ):
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
                _, L_opt, R_opt, _ = best
                self.L, self.R = L_opt, R_opt
                self.boundary_source = (
                    "optimized_gauge" if boundary_opt_gauge else "optimized"
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
        Real, EXACT isometry check for a candidate ``(L, R)`` pair -- not
        the fast bond-space-recursion estimate ``optimize_boundary_
        operators`` reports internally (``utils._bondspace_isometry_
        defect``), which has been measured, on the closely related q=D
        floor search, to disagree with the true ``||V^dagger V - I||_F``
        by up to nine orders of magnitude on some tensors (e.g. subspace:
        8.6e-11 reported vs 0.92 actual). A candidate that looks fine by
        its own optimizer-reported residual is therefore not yet
        trustworthy on its own.

        Reuses ``_build_isometry_unitary`` -- the exact same check
        ``synthesize()``/``synthesize()`` would eventually hit while
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
        if self.boundary_source == "canonical":
            return 1e-10
        # Boundaries from optimize_boundary_operators only satisfy the
        # isometry condition up to penalty-method precision; dilation still
        # snaps them to the nearest exact isometry (see
        # dilate_isometry_to_unitary), so a looser acceptance tolerance here
        # only affects which optimizer outputs are usable at all, not the
        # exactness of what synthesize() ultimately uses.
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
        lowers ``q_unif`` without leaving the canonical construction at all
        (measured: q 53.67 -> 35.56 on diagonal-phase, ell 42 -> 28; 10.0 ->
        9.0 on subspace-rotation). :func:`utils.optimize_q_unif` finds them;
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
        r_vec_conj = self.r_vec.H.reindex(
            {f"bond_{self._N}": f"bond_{self._N}_dag"}
        ).retag({"r": "r*"})
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
        L = matrix_sqrt_hermitian(L2_matrix)
        R = matrix_sqrt_hermitian(R2_matrix)

        return L, R

    def _compute_q_unif(self) -> float:
        r"""
        Calculates the entangling power $q_{unif}$ for a Translationally Invariant (TI) MPU.
        Formula: $q_{unif} = \sqrt{\text{Tr}(L_2^{-1}) \text{Tr}(R_2^{-1})}$
        """
        if self.L_inv.shape != self.R_inv.shape:
            raise ValueError(
                f"Dimension mismatch: L2 {self.L_inv.shape} vs R2 {self.R_inv.shape}"
            )

        # trace(L@L) >= 0 only holds for Hermitian positive-definite L, R
        # (true of the canonical construction, and of the HPD-only lifted
        # optimizer). The gauge-optimized boundary pair is generally *not*
        # Hermitian, so this sanity check does not apply to it -- validity
        # there is instead established by the optimizer's own isometry
        # residual and q >= 1 checks (see __init__).
        if getattr(self, "boundary_source", "canonical") != "optimized_gauge" and (
            np.real(np.trace(self.L @ self.L)) < 0
            or np.real(np.trace(self.R @ self.R)) < 0
        ):
            raise ValueError(
                "Unphysical negative trace detected in boundary operators."
            )

        L2_inv = self.L_inv @ self.L_inv
        R2_inv = self.R_inv @ self.R_inv

        q_unif = np.sqrt(np.trace(R2_inv @ L2_inv.T))

        return float(np.real(q_unif))

    def _get_merging_operator(self) -> np.ndarray:
        # Call the parent method using the instance attributes
        return get_merging_operator(self.L_inv, self.R_inv)

    def _get_rotation_params(self) -> tuple[int, list[float]]:
        """
        Compute ell and the rotation angles for the uniform merging operator.
        Wrapper around the abstract _compute_rotation_angle using cached C.
        """
        _, _, C = self._build_lcu_data()
        return CircuitDecomposition._compute_rotation_params(C)

    def _build_lcu_data(self):
        """
        Uniform case: the merge operator is the same at every bond.
        Computed once, cached, and reused for every merge in the tree.
        Uses the optimal SVD-based LCU decomposition with root-of-unity phases.
        """
        if self._lcu_cache is None:
            coeffs, units, C = CircuitDecomposition._compute_lcu(self.MergingOperator)
            self._lcu_cache = (coeffs, units, C)
        return self._lcu_cache

    def _build_merging_unitary(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build and cache the (B, W_ctrl) block-encoding pair for the original
        (non-rotated) LCU decomposition of the merging operator, i.e. the
        raw ``U = B W_ctrl B†``-style embedding of ``M`` before the extra
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
        """
        Build and cache all uniform GOAA data.

        Returns
        -------
        dict
            Dictionary containing the original LCU, rotated LCU, normalization
            constants, GOAA parameters, and block-encoding matrices.
        """
        if getattr(self, "_goaa_cache", None) is not None:
            return self._goaa_cache

        coef_0, unit_0, C_0 = self._build_lcu_data()

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

        self._goaa_cache = {
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

        self.C = float(C)
        self.ell = int(ell)

        return self._goaa_cache
