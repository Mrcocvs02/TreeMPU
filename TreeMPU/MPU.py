r"""Exact quantum-circuit synthesis of open-boundary Matrix Product Unitaries.

The module compiles an open-boundary MPU

.. math::

    U_N = \sum_{v_0,\dots,v_N} \ell_{v_0}\, r_{v_N}\,
          A[v_0,v_1]\otimes A[v_1,v_2]\otimes\cdots\otimes A[v_{N-1},v_N]

into an exact Qiskit circuit following the tree decomposition of Styliaris,
Trivedi and Cirac [PRL 135, 260602 (2025)], with the extensions of the thesis
(chapters "MPU decomposition" and "Analytical results"):

1. Boundary operators :math:`L, R` turn every site block
   :math:`V=L\,A\,R` into an isometry.
2. Neighbouring blocks are merged pairwise along a binary tree. Each merge
   applies a block encoding (two-term LCU) of the rank-one merging operator
   :math:`M` with 1-norm :math:`q=\|R^{-\top}L^{-1}\|_F`
   (thesis ``eq:merging_operator``, ``lem:M_rank_one``).
3. :math:`\ell` rounds of generalized oblivious amplitude amplification
   (GOAA) make each merge exact, with :math:`\ell` the smallest integer such
   that :math:`q\le\csc(\pi/(4\ell+2))` after lifting :math:`q` onto the
   target with rotation qubits (``rem:exact_amplification_angle``). The depth
   scales as :math:`N^{\log_2(2\ell+1)}` (``thm:uniform_depth``).

Classes
-------
:class:`UniformMPU`
    Translation-invariant MPU (one bulk tensor) with several boundary
    strategies.
:class:`GeneralMPU`
    Site-dependent MPU (one tensor per site) with per-bond boundaries and
    :math:`q=\max_k q_k` (``thm:nonuniform_depth``).
:class:`CircuitDecomposition`
    Abstract base class: LCU/GOAA engine, tree schedule, circuit assembly and
    exact resource estimators.

Conventions
-----------
==================== =====================================================
Site tensor          ``A[p_in, p_out, bond_L, bond_R]`` (first physical
                     index is the input)
Boundary operators   ``L[ext, bond]``, ``R[ext, bond]``; merging kernel
                     :math:`R^{-\top}L^{-1}`
Conditioning number  :math:`q=\|R^{-\top}L^{-1}\|_F`
Site isometry        maps the physical **input** leg to (output, external
                     bonds)
Qubit order          Qiskit little endian; site 0 is the least significant
                     system register
Circuit              equals :math:`U_N` up to a global phase; ancillas start
                     and end in :math:`|0\rangle`
==================== =====================================================

The stored boundary operators ``L[ext, bond]`` and ``R[ext, bond]`` are the
transposes of the matrices :math:`L` and :math:`R` in the thesis product
:math:`V=L\,A\,R`: the site isometries contract the second axis of both with
the bond index, and the merging kernel :math:`R^{-\top}L^{-1}` and the
conditioning number :math:`q` above are formed from the stored arrays. The
boundary strategy of a chain is reported in ``boundary_source``; its
vocabulary differs between the classes (see the class docstrings), for
example ``"canonical_sigma_tau"``
(:class:`UniformMPU`) and ``"sigma_tau_optimized"`` (:class:`GeneralMPU`)
name the same strategy.

Examples
--------
>>> from mpu_generators import build_family  # doctest: +SKIP
>>> from TreeMPU import UniformMPU
>>> tensors, l_vec, r_vec = build_family(  # doctest: +SKIP
...     "branching_uniform", d=2, bond_dim=2, seed=0
... )
>>> mpu = UniformMPU(tensors[0], l_vec, r_vec, N=4, light=True)  # doctest: +SKIP
>>> qc, sys_regs, bond_regs, anc_regs, ancillas = mpu.synthesize()  # doctest: +SKIP
>>> params = mpu.synthesis_cost_params()  # doctest: +SKIP
>>> UniformMPU.analytic_synthesis_stats(params, N=2**16)  # doctest: +SKIP

References
----------
Equation, lemma and definition numbers without a qualifier (Eq. A33, Eq. 26,
Lemma 1, Definition 3, Theorem 1') refer to G. Styliaris, R. Trivedi and
J. I. Cirac, "Quantum circuits for matrix-product unitaries", PRL 135, 260602
(2025) and its supplemental material. Results of the thesis (M. Vassallo,
*Beyond-QCA MPUs: Generation, Certification, and Circuit Synthesis*) are
cited by LaTeX label (e.g. ``thm:schmidt_bound``); see
``docs/theory_code_map.md``.
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
#: give C within ~1e-15 of a target or far from it; iterative solvers
#: (qD_general, optimized, optimized_gauge) converge to ~1e-8 and are accepted
#: at isometry tolerance 1e-4, so the identification stays below the precision
#: of the boundary itself.
GOAA_TARGET_RTOL = 1e-7


#: Isometry residual ||V^dag V - I||_F a site may have after a partially
#: optimized boundary is accepted in GeneralMPU; equals the tolerance applied
#: to optimized boundaries (see ``_isometry_tol``).
_OPT_P_SITE_TOL = 1e-4

#: Largest isometry residual of a merge-tree segment of a ``GeneralMPU`` with
#: optimized boundaries before a ``RuntimeWarning`` is issued.
_SEGMENT_ISOMETRY_WARN_TOL = 1e-6


#: Largest width (LCU ancilla + rotation + both bond registers, in qubits) for
#: which ``_build_U_block_encoding`` assembles W_ctrl as one dense UnitaryGate
#: instead of attaching each LCU term with ``Gate.control()``. 12 qubits is a
#: 4096 x 4096 complex matrix (~268 MB); beyond that the per-term route is
#: used.
DENSE_BLOCK_ENCODING_MAX_QUBITS = 12


class CircuitDecomposition(ABC):  # noqa: B024 (hooks raise NotImplementedError)
    r"""Tree decomposition of an MPU into a quantum circuit (abstract base class).

    Holds the machinery shared by :class:`UniformMPU` and :class:`GeneralMPU`:
    the LCU decomposition and block encoding of the merging operator, the GOAA
    parameters (:meth:`_compute_rotation_params`), the tree schedule, circuit
    assembly (:meth:`synthesize`) and exact resource estimation
    (:meth:`estimate_synthesis_stats`, :meth:`synthesis_cost_params`,
    :meth:`analytic_synthesis_stats`). Subclasses provide the site isometries
    and the per-bond merge data through the hooks ``_site_isometry_tensor``,
    ``_bond_qubit_count``, ``_bond_dimension``, ``_merge_bundle`` and
    ``_isometry_tol``.

    Parameters
    ----------
    N : int
        Number of sites.
    d : int
        Local (physical) dimension.
    l_vec : ndarray, shape (D_0,)
        Left boundary vector, attached to index ``bond_0``.
    r_vec : ndarray, shape (D_N,)
        Right boundary vector, attached to index ``bond_{N}``.

    Attributes
    ----------
    l_vec, r_vec : quimb.tensor.Tensor
        Boundary vectors as one-index tensors tagged ``"l"`` and ``"r"``.

    Notes
    -----
    Segment programs. The content of a tree segment is a list of entries in
    program order: ``("g", gate, local_qubits)`` is one leaf gate and
    ``("s", child, inverse, local_map)`` is a whole child segment. The inverse
    of a segment replays its entries in reverse order, inverting each leaf and
    flipping each child call. :meth:`synthesize` expands the program into the
    circuit and :meth:`estimate_synthesis_stats` schedules the same program,
    so the two agree by construction.
    """

    def __init__(self, N: int, d: int, l_vec: np.ndarray, r_vec: np.ndarray):
        self._N = N
        self._d = d
        self.l_vec = qtn.Tensor(l_vec, inds=["bond_0"], tags=["l"])
        self.r_vec = qtn.Tensor(r_vec, inds=[f"bond_{N}"], tags=["r"])

    @staticmethod
    def _compute_lcu(M: np.ndarray) -> tuple[list, list, float]:
        r"""Return the optimal two-term LCU of a merging operator.

        With ``M = U Sigma V^dag`` and ``Sigma' = Sigma / sigma_1`` (all
        ``sigma'_j <= 1``), the unitaries

        .. math::

            W_\pm = U\,(\Sigma' \pm i\sqrt{I-\Sigma'^2})\,V^\dagger

        are unitary (each diagonal entry has modulus 1) and satisfy
        ``M = (sigma_1/2)(W_+ + W_-)`` exactly, for any ``M``.

        Parameters
        ----------
        M : ndarray, shape (n, n)
            Merging operator.

        Returns
        -------
        coefficients : list of float
            ``[sigma_1/2, sigma_1/2]``, or ``[sigma_1]`` when ``M/sigma_1``
            is already unitary.
        unitaries : list of ndarray
            ``[W_plus, W_minus]``, or ``[U V^dag]`` in the unitary case.
        C : float
            LCU 1-norm ``sum(coefficients)``, equal to ``sigma_1``.

        Raises
        ------
        ValueError
            If the largest singular value is below ``1e-14``.

        Notes
        -----
        The 1-norm ``C`` sets the GOAA cost. Any exact LCU
        ``M = sum_i c_i W_i`` has ``C >= ||M||_op`` (triangle inequality), and
        this one attains the bound. For the rank-one merging operators built
        here a Fourier decomposition would give the same ``C``; the benefit
        is the term count: ``K = 2`` (one LCU qubit) instead of ``D^2``, so
        ``_build_U_block_encoding`` issues two ``Gate.control()`` calls
        instead of ``D^2``, each of which copies the controlled gate.
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
        r"""Build the state preparation ``B`` (and optionally ``W_ctrl``) of a merge.

        For an LCU ``M = sum_i c_i W_i`` the merging unitary is
        ``U = B^dag W_ctrl B``; projecting the ancilla on ``|0>`` after ``U``
        implements ``M / C`` on the image of the isometry.

        Parameters
        ----------
        coefficients : array-like, shape (K,)
            Positive real coefficients ``c_i``.
        unitaries : list of ndarray
            ``K`` unitaries ``W_i``, each of shape ``(dim_system, dim_system)``.
        C : float
            Sum of the coefficients.
        build_w_ctrl : bool, default True
            If False, return ``None`` for ``W_ctrl``. The circuit path never
            needs the dense ``W_ctrl`` (its size is
            ``(ancilla x system)^2``); only the debug check
            ``_verify_merging_unitary`` does.

        Returns
        -------
        B : ndarray, shape (dim_ancilla, dim_ancilla)
            Unitary with ``B|0> = C^(-1/2) sum_i sqrt(c_i) |i>`` (Eq. A33);
            ``dim_ancilla = 2**ceil(log2 K)``.
        W_ctrl : ndarray or None, shape (dim_system*dim_ancilla,) * 2
            ``sum_i (W_i)_S (x) |i><i|_A`` (Eq. A34), with the identity on
            the system for unused ancilla states ``i >= K``.
        """
        K = len(coefficients)
        dim_ancilla = 2 ** int(np.ceil(np.log2(K)))
        dim_system = int(unitaries[0].shape[0])

        # --- B: ancilla state preparation (Eq. A33) ---
        first_col = np.zeros(dim_ancilla, dtype=complex)
        first_col[:K] = np.sqrt(coefficients) / np.sqrt(C)

        M = np.eye(dim_ancilla, dtype=complex)
        M[:, 0] = first_col
        Q, R = qr(M)

        diag_R = np.diag(R)
        phases = np.where(np.abs(diag_R) > 1e-14, diag_R / np.abs(diag_R), 1.0)
        B = Q @ np.diag(phases)

        # --- W_ctrl: controlled unitary (Eq. A34) ---
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
        r"""Return ``ell`` and the rotation angles that make GOAA exact.

        GOAA is exact when ``(2 ell + 1) theta = pi/2`` with
        ``sin(theta) = 1/C'`` (Eq. A47) after the rescaling of Eq. (A48). The
        admissible norms are the discrete targets
        ``C'(ell) = 1/sin(pi/(4 ell + 2)) = 1, 2, 3.236, 4.494, ...``, and the
        rescaling can only increase the LCU 1-norm
        (``|cos phi| + |sin phi| >= 1``).

        Parameters
        ----------
        C : float
            LCU 1-norm of the merging operator (``q`` for the operators built
            here).

        Returns
        -------
        ell : int
            Smallest integer with ``C <= C'(ell)`` (up to the relative
            tolerance ``GOAA_TARGET_RTOL``).
        phis : list of float
            One angle per rotation qubit; ``len(phis)`` is the number of
            rotation qubits (1 in the common case).

        Notes
        -----
        One rotation qubit multiplies the norm by at most ``sqrt(2)``. The
        ratios between consecutive targets are 2.000, 1.618, 1.389, ..., so
        the step ``ell = 1 -> 2`` (window ``C in (2, 2.288]``) is not
        reachable with one qubit; this window is occupied, because canonical
        form gives ``q_k >= D_k``. With ``n`` qubits the norm is multiplied
        by ``prod_j (|cos phi_j| + |sin phi_j|) <= 2^(n/2)``; choosing
        ``n = ceil(2 log2 f)`` for the required factor ``f = C'(ell)/C`` and
        splitting it evenly makes every target reachable exactly.
        """
        # C is snapped to a target within GOAA_TARGET_RTOL so that values an
        # iterative solver returns as 2 +- 1e-8 neither flip ell nor add a
        # rotation qubit with a ~1e-8 angle.
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
        """Recover which rotation branches survive the coefficient filter.

        Replays the bookkeeping of ``_build_rotated_lcu`` and returns, per
        rotation angle, the list of surviving branch indices of
        ``[cos(phi) Id, sin(phi) iZ]`` (those with ``|c_i p_j| >= 1e-14``).

        Parameters
        ----------
        coef_0 : list of float
            Coefficients of the unrotated LCU.
        phis : list of float
            Rotation angles.
        coef_ref : list of float
            ``goaa_data["coef"]``; the reconstructed coefficient list is
            compared with it elementwise, which pins down the term order.

        Returns
        -------
        list of list of int, or None
            One list of surviving branch indices per angle, or ``None`` if the
            surviving set differs between terms at some level or the
            reconstructed coefficients do not match ``coef_ref``. The closed
            form of ``_build_closed_wctrl_gates`` is valid only in the
            non-``None`` case; this is checked at run time.
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
        r"""Build the closed-form circuit pieces of ``W_ctrl``.

        Returns ``None`` when the required structure is absent, so that the
        caller falls back to the per-term or dense route.

        Parameters
        ----------
        goaa_data : dict
            Output of ``_compute_goaa_bundle`` (keys ``K``, ``phis``,
            ``coef``, ``coef_0``, ``unit_0``).
        D_bond : int
            Physical bond dimension.
        D_bond_pad : int
            Padded (power-of-two) bond dimension.
        n_rot : int
            Number of rotation qubits.

        Returns
        -------
        dict or None
            ``V_dag_gate`` / ``V_gate``: the shared state-preparation unitary
            and its inverse. ``D_ctrl_gate`` / ``D_ctrl_dag_gate``: the
            diagonal sign gate and its inverse, carrying the accumulated
            global phase. ``sign_pos``: index of the sign qubit in the LCU
            register. ``phase_ctrl``: ``(lcu_pos, rot_pos, eps)`` per active
            level. ``phase_fixed``: ``rot_pos`` per level where only the
            ``iZ`` branch survives.

        Notes
        -----
        The merging operator is ``M = |0..0><1|(R^-1 (x) L^-1)``: rank one
        with output ket ``|0..0>`` (``eq:merging_operator``). Writing
        ``M = sigma_1 |e0><v|`` with ``U = I``, the two-term LCU of
        ``_compute_lcu`` becomes

        .. math::

            W_\pm = D_\pm V^\dagger,\qquad
            D_\pm = \Pi_0 \pm i(1-\Pi_0),

        with ``Pi_0 = |e0><e0|`` and ``V^dag`` any unitary whose first row is
        ``v``. Controlling on the sign qubit then needs a single ``V``:

        .. math::

            \sum_s |s\rangle\langle s|\otimes W_s
            = D_{\rm ctrl}\,(1_A\otimes V^\dagger),\qquad
            D_{\rm ctrl} = 1_A\otimes\Pi_0 + iZ_A\otimes(1-\Pi_0),

        and ``D_ctrl`` is diagonal in the computational basis.

        Rotation extension. ``_build_rotated_lcu`` extends every term by
        ``[cos(phi) Id, sin(phi) iZ]`` per angle (inner loop fastest), so
        ``W_ctrl`` factorizes across registers. Each rotation factor is
        ``sign(cos phi) [ |0><0| (x) Id + |1><1| (x) eps iZ ]``, i.e. one
        controlled ``RZ`` plus a global sign, with
        ``eps = sign(cos phi) sign(sin phi)``. When only one branch survives
        the coefficient filter the factor is unconditional.

        Bit assignment. The term index is built sign-first, then one bit per
        surviving level in ``phis`` order; ``lcu[0]`` is the least
        significant bit (Qiskit convention), hence ``sign -> lcu[n_lcu-1]``,
        level ``a -> lcu[n_lcu-2-a]`` and ``phis[h] -> rot[n_rot-1-h]``. The
        assignment is validated numerically against the per-term route and
        guarded at run time by ``_closed_branch_structure``.

        Padding. The raw axes of the bond block are ``(ext_R, ext_L)`` in
        C order (``compute_lcu_pad_indices``); the physical block is placed in
        the padded space with ``np.ix_(pad_idx, pad_idx)`` and
        ``pad_idx[0] == 0``, so the output ket stays ``e0``. The block acts as
        zero on the padded-away subspace (the per-term route leaves the
        identity there). That subspace is never populated by the tree
        isometries, and only the block-encoding condition on the physical
        subspace is needed.

        ``V^dag`` is constrained only on its first row, so it is built as the
        inverse of a state preparation of ``v.conj()`` (Moettoenen et al.
        2004, ``O(D)`` gates) instead of a dense ``qs_decomposition``
        (``O(D^2)``).
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
        try:
            V_gate = StatePreparation(v_row.conj()).definition.to_gate(label="V")
        except Exception as exc:
            # Qiskit's isometry decomposition can reject a nearly degenerate row
            # (large conditioning number) with an internal unitarity error.
            warnings.warn(
                f"Closed-form W_ctrl unavailable ({type(exc).__name__}: {exc}); "
                "using the per-term/dense block encoding for this bond.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
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
        """Extend the LCU by one rotation factor per angle in ``phis``.

        Returns the LCU of ``M (x) (cos(phi_j) I + i sin(phi_j) Z)`` for each
        angle, as ``(coef, unit, C)``. This makes the Grover exponent ``ell``
        an integer (Eq. A48, generalized to several qubits so that any norm
        factor up to ``2**(len(phis)/2)`` is reachable). Terms with
        coefficient below ``1e-14`` are dropped.

        Parameters
        ----------
        coef, unit : list
            Coefficients (non-negative floats) and unitary matrices of the
            LCU of ``M``.
        C : float
            LCU 1-norm of ``M`` (the sum of ``coef``); unused, the norm is
            recomputed from the new coefficients.
        phis : list of float
            One rotation angle per extra qubit.

        Returns
        -------
        coef, unit : list
            Coefficients and unitaries of the extended LCU; each unitary acts
            on one extra qubit per angle.
        C : float
            Sum of the new coefficients.
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
        r"""Return the reflection ``2|0..0><0..0| - I`` as a gate.

        Parameters
        ----------
        n_qubits : int
            Number of qubits the reflection acts on (positive).
        n_dirty : int, default 0
            Number of borrowed (dirty) ancillas. If positive, the gate acts
            on ``n_qubits + n_dirty`` qubits and the last ``n_dirty`` of them
            are the borrowed ancillas.

        Returns
        -------
        qiskit.circuit.Gate
            Gate labelled ``Ref0_{n_qubits}``, transpiled to
            ``['u', 'cx', 'id']`` so that the multi-controlled-X
            decomposition is reproducible.

        Raises
        ------
        ValueError
            If ``n_qubits`` is not positive.

        Notes
        -----
        Dirty ancillas may hold any state and are returned unchanged, so they
        cost no space: the caller routes qubits that are idle at that point
        of the schedule (see ``_idle_local_indices``). The multi-controlled X
        uses ``synth_mcx_n_dirty_i15`` (Iten et al. 2015, needs
        ``n_ctrl - 2`` ancillas) when enough are lent, else
        ``synth_mcx_1_dirty_kg24`` (Khattar and Gidney 2024, one ancilla)
        for ``n_ctrl >= 3``, else a plain ``mcx``.
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
                # Prefer the n-dirty construction when enough qubits were
                # lent; otherwise use the one-dirty one.
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
        """Return the indices of ``range(n_total)`` not touched by ``used``.

        The reflection is a single instruction in the schedule, so every
        qubit outside its support is idle while it runs and can be borrowed
        as a dirty ancilla (it is returned unchanged).

        Parameters
        ----------
        n_total : int
            Number of local qubits.
        used : list of int
            Local qubits in the support of the instruction.

        Returns
        -------
        list of int
            Idle local qubit indices, in increasing order.
        """
        busy = set(used)
        return [i for i in range(n_total) if i not in busy]

    @staticmethod
    def _borrow_for_reflection(n_total: int, reflect_loc: list[int]) -> tuple[int, list[int]]:
        """Return ``(n_dirty, borrowed_loc)`` for a reflection on ``reflect_loc``.

        Lends up to ``n_ctrl - 2`` idle qubits (the amount wanted by
        ``synth_mcx_n_dirty_i15``), at least one when any is idle; if fewer
        are available, ``_make_reflection_zero`` falls back to the
        one-dirty-ancilla construction. Nothing is allocated: the lent qubits
        are already present in the schedule. Returns ``(0, [])`` when the
        reflection has fewer than three controls.

        Parameters
        ----------
        n_total : int
            Number of local qubits.
        reflect_loc : list of int
            Local qubits of the reflection; the last one is the target.

        Returns
        -------
        n_dirty : int
            Number of borrowed qubits.
        borrowed_loc : list of int
            Their local indices.
        """
        n_ctrl = len(reflect_loc) - 1
        if n_ctrl < 3:
            return 0, []
        idle = CircuitDecomposition._idle_local_indices(n_total, reflect_loc)
        n_dirty = min(len(idle), max(n_ctrl - 2, 1))
        return n_dirty, idle[:n_dirty]

    @staticmethod
    def _qubit_union(a: list, b: list) -> list:
        """Return ``a + b`` without repeated qubits, keeping the order.

        Children share history-ancilla qubits when their layer's ancillas are
        pooled.

        Parameters
        ----------
        a, b : list
            Qubit lists.

        Returns
        -------
        list
            ``a`` followed by the elements of ``b`` not in ``a``.
        """
        seen = set(a)
        return a + [q for q in b if q not in seen]

    @staticmethod
    def _canonical_segment_qubits(seg: dict) -> list:
        r"""Return the canonical external qubit order of a tree segment.

        The order is
        ``sys | b_ct_R | b_ct_L | b_spectator | history_ancillas``:

        * ``sys``: physical qubits of the segment's sites;
        * ``b_ct_R``, ``b_ct_L``: bond registers at the segment's right/left
          edge that are *contracted* (merged) at the next tree level;
        * ``b_spectator``: open bond registers not touched by the next merge;
        * ``history_ancillas``: LCU and rotation ancillas of the merges
          already performed inside the segment (all in :math:`|0\rangle`
          on the success branch).

        Parameters
        ----------
        seg : dict
            Segment with the keys above, each a list of qubits.

        Returns
        -------
        list
            Concatenation of the five lists.

        Notes
        -----
        This is the bookkeeping order of composed tree gates, not the tensor
        axis order used to build local dense matrices.
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
        r"""Convert a local isometry tensor into a square unitary matrix.

        Rows are ordered ``ext_L | ext_R | sys`` (most to least significant)
        and columns by the sorted ``p_in_*`` legs. Qiskit matrices are little
        endian with respect to the target list, so the gate is appended to
        qubits ordered ``sys | ext_R | ext_L``. Boundary sites have only one
        of ``ext_L``/``ext_R``; the rule is unchanged.

        Parameters
        ----------
        V_tn : quimb.tensor.Tensor
            Local isometry with physical input legs ``p_in_*``, physical
            output legs ``p_out_*`` and external legs ``ext_L`` and/or
            ``ext_R`` (the isometry maps the input legs to the output and
            external legs).
        label : str
            Label used in error messages.
        D_sys_pad : int
            Padded physical dimension (a power of two).
        D_bond_pad_L, D_bond_pad_R : int or None
            Padded dimensions of the ``ext_L``/``ext_R`` legs (powers of two).
            Required exactly when the corresponding leg is present. A uniform
            bulk has the same value for both; in a general MPU they may
            differ because they belong to different cuts.
        tol : float, default 1e-10
            Isometry-acceptance tolerance forwarded to
            :func:`dilate_isometry_to_unitary`. Optimized boundaries satisfy
            the isometry condition only to the optimizer precision, so
            callers pass a wider value for them (``_isometry_tol``).

        Returns
        -------
        ndarray
            Square unitary dilation of the zero-padded isometry.

        Raises
        ------
        ValueError
            If the tensor has no ``p_out_*`` or no ``p_in_*`` legs, a padding
            is missing for a present external leg, the padded output
            dimension is smaller than the dense one, or (raised by
            :func:`~TreeMPU.utils.dilate_isometry_to_unitary`) the padded
            matrix is not an isometry within ``tol``. The last condition is
            the failure mode of a bad boundary pair and is what
            :meth:`UniformMPU._validate_boundary_pair` relies on.
        """
        sys_inds = sorted(i for i in V_tn.inds if i.startswith("p_out"))
        in_inds = sorted(i for i in V_tn.inds if i.startswith("p_in"))

        # Bond legs first (most significant), system last (least significant).
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
        """Return, for each tree layer, the bond indices at which pairs merge.

        Bond ``b`` (0-indexed) is the internal MPU bond between sites ``b``
        and ``b + 1``. Segments are paired left to right; an odd leftover
        segment is carried up unchanged.

        Parameters
        ----------
        N : int
            Number of sites (positive).

        Returns
        -------
        list of list of int
            ``plan[layer][pair_idx]`` is the bond index merged by the
            ``pair_idx``-th pair at that layer.

        Raises
        ------
        ValueError
            If ``N`` is not positive.
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
        """Return the LCU and GOAA data of one merging operator.

        Data-only orchestration shared by every merge point: the single
        operator of a uniform MPU or each of the ``N - 1`` operators of a
        general MPU.

        Parameters
        ----------
        M : ndarray, shape (n, n)
            Merging operator.

        Returns
        -------
        dict
            ``coef_0``, ``unit_0``, ``C_0``: LCU before the rotation
            extension; ``coef``, ``unit``, ``C``, ``K``: LCU after it (``K``
            terms); ``ell``: number of GOAA rounds; ``phis``, ``n_rot``:
            rotation angles and their count; ``B_mat``: ancilla state
            preparation; ``W_ctrl_mat``: always ``None`` (the dense
            controlled unitary is not built here).
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
        """Build the block encoding ``U = B^dag W_ctrl B`` of a merge and its dagger.

        Parameters
        ----------
        goaa_data : dict
            Bundle produced by :meth:`_compute_goaa_bundle`.
        D_bond : int
            Unpadded bond dimension at this merge point.
        n_bond_per_site : int
            Number of qubits encoding one virtual bond.

        Returns
        -------
        U_circ : qiskit.QuantumCircuit
            Circuit implementing ``U`` on registers ``lcu | rot | bonds``.
        U_gate : qiskit.circuit.Gate
            Gate version of ``U_circ``; signature ``lcu | rot |
            contracted_bonds``.
        U_dagger_gate : qiskit.circuit.Gate
            Gate implementing ``U^dag``, built directly (not by
            ``U_gate.inverse()``).
        u_schedule, u_dagger_schedule : list of (Gate, tuple of int)
            The same content as flat lists of (leaf gate, local qubit
            indices) over the same signature, consumed by
            :meth:`synthesize`.
        n_lcu : int
            Number of LCU ancilla qubits.
        n_rot : int
            Number of rotation ancilla qubits.

        Notes
        -----
        Only the block ``<0|_{lcu,rot} U |0>_{lcu,rot} = M/C`` (up to the
        rotation phases and the ``ext_L``/``ext_R`` exchange of
        ``compute_lcu_pad_indices``) matters for GOAA; two valid encodings may
        differ outside it.

        Three routes build ``W_ctrl`` (chosen in this order):

        1. Closed form (:meth:`_build_closed_wctrl_gates`): one state
           preparation, one multi-controlled phase gate and one controlled
           ``RZ`` per rotation branch. Valid when ``M`` is rank one with
           output ket ``e0``, which holds for every operator built here.
        2. Dense, when ``lcu + rot + 2 * bond`` qubits do not exceed
           ``DENSE_BLOCK_ENCODING_MAX_QUBITS``: ``W_ctrl`` is assembled once
           and handed to Qiskit as one ``UnitaryGate``.
        3. Per term otherwise: each ``W_i`` is decomposed with
           ``qs_decomposition`` and controlled on its LCU index. This avoids
           any matrix of size (ancilla x system)^2, but
           ``Gate.control()`` converts every gate of the decomposition, so
           its cost grows with the term's gate count.

        ``U^dag`` is built directly because ``W_ctrl`` is block diagonal,
        ``W_ctrl^dag = sum_i |i><i| (x) W_i^dag``, so that
        ``U^dag = B^dag W_ctrl^dag B`` follows by the same construction as
        ``U``; ``Gate.inverse()`` on the composite gate would recurse into
        every controlled sub-gate.

        The state preparation ``B`` is constrained only on ``B|0>``. A
        ``StatePreparation`` completion (Moettoenen et al. 2004) is compared
        with the dense QR completion by leaf-block count and the cheaper one
        is used; ``B`` has at most three qubits, so building both is cheap.
        """
        B_mat = goaa_data["B_mat"]
        unit = goaa_data["unit"]
        K = goaa_data["K"]

        # Number of rotation qubits: 1 unless the required norm factor
        # exceeds sqrt(2) (see _compute_rotation_params).
        n_rot = int(goaa_data["n_rot"])
        n_lcu = int(np.ceil(np.log2(K)))

        D_bond_pad = 2**n_bond_per_site

        # B only has to satisfy B|0> = sum_i sqrt(c_i/C) |i>, so a
        # state-preparation completion is a valid alternative to the QR one.
        # The cheaper of the two (leaf-block count) is kept; see Notes.
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

        # Per-term target order is (rot | ext_L | ext_R), the bit layout
        # assumed by pad_block_diagonal_operator / compute_lcu_pad_indices.
        bc_ext_R = list(dummy_bc[:n_bond_per_site])
        bc_ext_L = list(dummy_bc[n_bond_per_site:])
        term_targets = list(dummy_rot) + bc_ext_L + bc_ext_R

        # Integer-index mirror of the signature, for the schedule form; it
        # follows term_targets' order, not dummy_bc's (ext_R | ext_L) layout.
        lcu_idx = list(range(n_lcu))
        rot_idx = list(range(n_lcu, n_lcu + n_rot))
        bc_idx = list(range(n_lcu + n_rot, n_lcu + n_rot + 2 * n_bond_per_site))
        bc_ext_R_idx = bc_idx[:n_bond_per_site]
        bc_ext_L_idx = bc_idx[n_bond_per_site:]
        term_targets_idx = rot_idx + bc_ext_L_idx + bc_ext_R_idx

        u_schedule = [(B_gate, tuple(lcu_idx))]
        u_dagger_schedule = [(B_gate, tuple(lcu_idx))]

        # --- Closed-form route for W_ctrl (see Notes) ---
        # n_rot can exceed 1 even when K == 2: a rotation angle ~1e-16 has its
        # sin branch dropped by the 1e-14 filter in _build_rotated_lcu, leaving
        # an identity factor on the rotation qubit.
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

            # One CRZ(-eps pi) per level where both branches survived, one
            # unconditional RZ(-pi) (= iZ) per level where only the sin branch
            # did. They act on (lcu_h, rot_h) only, so their order relative to
            # the D_ctrl/V part is free.
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
            # V^dagger on the bonds, then D_ctrl jointly on bonds + sign qubit.
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

            # U^dagger = B^dagger W_ctrl^dagger B with
            # W_ctrl^dagger = V . D_ctrl^dagger. D_ctrl is diagonal but not
            # Hermitian (eigenvalues 1 and i (-1)^a), so D_ctrl_dag_gate uses
            # RZ(+pi) and carries the conjugated global phase.
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
                    # Qiskit is little endian over the append() list and the
                    # LCU register comes first, so the ancilla factor is the
                    # least significant one: kron(term, proj).
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

    # --- hooks ---
    # Subclasses provide their (possibly site-dependent) data through the
    # hooks below; segment construction, tree merging and circuit assembly
    # only call these hooks.

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
        """Return the precomputed merge data of internal bond ``bond_index``.

        Returns
        -------
        dict
            Keys ``n_bond_per_site``, ``n_lcu``, ``n_rot``, ``ell``,
            ``U_gate``, ``u_schedule``, ``u_dagger_schedule``, ``R_Phi_gate``.
        """
        raise NotImplementedError

    def _isometry_tol(self) -> float:
        """Return the acceptance tolerance of the per-site isometry check.

        The tolerance is forwarded to ``dilate_isometry_to_unitary``. The
        default is tight (canonical sigma/tau boundaries); subclasses whose
        isometries can come from a numerical optimizer override it.

        Returns
        -------
        float
            Isometry tolerance.
        """
        return 1e-10

    # --- tree-circuit engine (flat, schedule-based; see ``synthesize``) ---

    @staticmethod
    def _remap_schedule(
        schedule: list[tuple[Gate, tuple[int, ...]]], mapping: list[int]
    ) -> list[tuple[Gate, tuple[int, ...]]]:
        """Re-index the local qubits of every entry through ``mapping``.

        ``mapping`` maps old to new local index; gate objects are untouched.

        Parameters
        ----------
        schedule : list of (Gate, tuple of int)
            Leaf gates with their local qubits.
        mapping : list of int
            ``mapping[old] = new``.

        Returns
        -------
        list of (Gate, tuple of int)
            The re-indexed schedule.
        """
        return [(gate, tuple(mapping[q] for q in qs)) for gate, qs in schedule]

    # --- segment programs (data layout: see the class Notes) ---

    @staticmethod
    def _remap_program(prog: list, mapping: list[int]) -> list:
        """Re-index the local qubits of every entry of a segment program."""
        out = []
        for e in prog:
            if e[0] == "g":
                out.append(("g", e[1], tuple(mapping[q] for q in e[2])))
            else:
                out.append(("s", e[1], e[2], tuple(mapping[q] for q in e[3])))
        return out

    def _program_signature(self, prog: list) -> int:
        """Return a small integer identifying a program up to structure.

        Segments with the same leaf gates (by content) on the same local
        qubits share it, so the uniform bulk of a chain is scheduled once per
        tree layer.

        Parameters
        ----------
        prog : list
            Segment program (see the class Notes).

        Returns
        -------
        int
            Index into a per-instance table of distinct program structures.
        """
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
        """Append ``(gate, qubits)`` of every leaf of ``seg`` to ``out``.

        Runs the program backwards, with inverted leaves, if ``inverse``.
        ``qmap`` maps the segment's local qubit indices to global ones.
        """
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

        Parameters
        ----------
        seg : dict
            Segment with its program under ``"prog"`` and a structure id
            under ``"sig"``.
        inverse : bool
            Schedule the inverse of the segment.
        start : list of int
            Layer at which each local qubit of ``seg`` becomes free.
        memo : dict
            Memo of segment schedules, keyed by structure and shifted start.
        cache : dict
            Leaf-block cache.

        Returns
        -------
        list of int
            Layer at which each local qubit is free after the segment.
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
        """Return the number of <=2-qubit leaf blocks of ``seg``.

        Parameters
        ----------
        seg : dict
            Segment with its program under ``"prog"`` and a structure id
            under ``"sig"``.
        inverse : bool
            Count the blocks of the inverse of the segment.
        memo : dict
            Memo keyed by structure, shared across calls.
        cache : dict
            Leaf-block cache.

        Returns
        -------
        int
            Number of leaf blocks.
        """
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
        """Build the initial per-site segment list of the tree merge.

        Each segment carries a program (see the segment-program comment
        above) and a scalar ``"global_phase"``; circuits are materialized
        only once, at the end of :meth:`synthesize`.

        Parameters
        ----------
        D_sys_pad : int
            Padded physical dimension of one site.
        n_sys_per_site : int
            Number of system qubits per site.
        site_sys_regs, site_bond_regs : list of QuantumRegister
            Per-site system and bond registers.

        Returns
        -------
        list of dict
            One segment per site with keys ``sys``, ``b_ct_R``, ``b_ct_L``,
            ``b_spectator``, ``history_ancillas``, ``prog``,
            ``global_phase``, ``sig`` and ``v_qubits``.

        Raises
        ------
        ValueError
            If a register has the wrong size, a site isometry has no external
            leg, or qubits are duplicated.
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
        """Merge one tree layer: contract bonds and apply the GOAA merge.

        For each pair of adjacent segments the merge is
        ``V_pair`` (both children), ``U``, then ``ell`` rounds of
        ``G = -U R_Psi U^dag R_Phi`` with ``R_Psi = V_pair Ref_0 V_pair^-1``.
        Schedules are composed by list concatenation and index remapping, not
        by nesting Qiskit gates; ``V_pair^-1`` is the reversed program with
        every leaf inverted, so no large ``Gate.inverse()`` is taken.

        Parameters
        ----------
        segments : list of dict
            Segments of the previous layer (canonical qubit order).
        layer_anc_pairs : list of (QuantumRegister, QuantumRegister)
            LCU and rotation register of each pair. A pooled register may be
            larger than needed; its first qubits are used.
        layer_bonds : list of int
            Bond index merged by each pair (see :meth:`_tree_merge_plan`).
        layer : int, default 0
            Layer index, used in error messages.

        Returns
        -------
        list of dict
            Merged segments; an odd leftover segment is carried up unchanged.

        Raises
        ------
        ValueError
            If a segment is not in canonical order, register sizes do not
            match the merge bundle, or qubits are duplicated.

        Notes
        -----
        The reflection ``Ref_0`` acts only on the current merge's registers
        (LCU, rotation and all bond registers of the pair), not on the
        history ancillas of the children. This equals the thesis's projector
        on all auxiliaries because every encoding built here shares the row
        ``e_0`` (property BE*) and every tree block is an isometry. The
        equivalence follows by induction on the tree given those two
        properties (not stated as a labelled thesis result) and is validated
        numerically against the dense unitary; it does not hold for a generic
        block encoding.
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

            # pair_schedule indices are 0..n_v-1 by construction.
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

    # --- site-isometry gates and exact leaf-block statistics ---

    @staticmethod
    def _site_isometry_gate(
        V_mat: np.ndarray, n_in: int, label: str, cache: dict | None = None
    ) -> Gate:
        """Return the cheaper of two gates implementing a site isometry.

        Candidates: a ``UnitaryGate`` over the full dilated unitary, and a
        Qiskit ``Isometry`` (Iten et al.) over the ``n_in`` physical columns
        only, so the free completion is never synthesized. Their elementary
        cost is measured and the smaller one is kept (the content cache makes
        this a one-off cost per distinct matrix). Which wins depends on the
        matrix (the isometry route helps when ``n_in`` is much smaller than
        ``dim``, the dense one for small blocks), so the choice is made by
        measurement, not by a rule.

        Parameters
        ----------
        V_mat : ndarray, shape (dim, dim)
            Dilated unitary; its first ``n_in`` columns are the isometry.
        n_in : int
            Number of physical input columns.
        label : str
            Gate label.
        cache : dict, optional
            Leaf-block cache shared with the estimators.

        Returns
        -------
        qiskit.circuit.Gate
            The dense gate, or the isometry gate when it reproduces the
            physical columns to ``1e-8`` and has fewer leaf blocks.

        Notes
        -----
        Qiskit's ``Isometry`` can fail on matrices with an exact algebraic
        degeneracy in its column elimination. A random single-qubit unitary
        on every wire, seeded from the matrix content so that estimator and
        synthesizer build the same gate, removes the degeneracy and is undone
        by one gate per wire.
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

            # Preconditioner seeded from the matrix content: reproducible,
            # and identical in the estimator and the synthesizer.
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
            # Qiskit's qubit 0 is the least significant index, so the tensor
            # factors are undone in reverse wire order.
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
        """Return a cache key derived from a ``UnitaryGate``'s matrix.

        Returns ``None`` for any other gate. A chain of ``N`` sites builds
        ``N`` separate ``UnitaryGate`` objects, but their matrices take only
        three distinct values (left boundary, bulk, right boundary). Keyed on
        ``id`` alone every bulk site would trigger its own
        ``qs_decomposition`` inside ``.definition``; hashing the matrix is
        negligible in comparison.

        The key contains ``num_qubits``, shape, dtype and a digest of the
        exact bytes (no tolerance). Gates with bit-identical matrices
        decompose identically (the label plays no part), so sharing blocks
        between them is exact.

        Parameters
        ----------
        gate : qiskit.circuit.Gate
            Gate to key.

        Returns
        -------
        tuple or None
            Hashable key, or ``None`` for a gate that is not a
            ``UnitaryGate``.
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
        """Flatten ``gate`` into its <=2-qubit leaf blocks.

        Parameters
        ----------
        gate : qiskit.circuit.Gate
            Gate to flatten (a ``_BlockLeaf`` returns its stored blocks).
        cache : dict
            Memo keyed by ``id(gate)`` and by content key; shared between
            calls.

        Returns
        -------
        list of tuple of int
            Leaf blocks, each a tuple of local qubit indices into the gate's
            own ``num_qubits``.

        Warns
        -----
        RuntimeWarning
            If Qiskit cannot synthesize the gate's definition; the gate is
            then counted as one block and estimates are lower bounds.

        Notes
        -----
        The basis is "arbitrary one- and two-qubit unitaries" with no
        ``transpile`` pass; every consumer of these blocks must use this same
        stopping rule for counts to be comparable. This is the one recursive
        ``.definition`` walk paid per distinct gate. The blocks of a gate's
        inverse are the reversed list: at this granularity each block's
        inverse occupies the same qubits and counts as one block, so
        reversing is an exact leaf-level flattening of ``gate.inverse()``
        without a second ``.definition`` walk.

        The cache stores ``id(gate) -> (gate, blocks)``. Keeping a reference
        to the gate prevents CPython from recycling its address for an
        unrelated temporary (sub-gates of a ``.definition`` are rebuilt on
        each access), which would return the wrong blocks.
        """
        if isinstance(gate, _BlockLeaf):
            return gate.blocks
        key = id(gate)
        hit = cache.get(key)
        if hit is not None:
            return hit[1]
        # Content key first: an id miss must not trigger the synthesis in
        # `.definition` for a site isometry already seen (see _content_key).
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
                # `.definition` synthesizes lazily and Qiskit's internal
                # unitarity check can reject a gate whose matrix is an exact
                # isometry up to rounding. The gate is then one opaque block,
                # which under-counts: the estimate is a lower bound.
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
        """Return ``(depth, gate_count)`` of a flat leaf-block list.

        Exact per-qubit greedy scheduling in the order given.

        Parameters
        ----------
        blocks : list of tuple of int
            Local qubits of each leaf block, in program order.

        Returns
        -------
        depth, gate_count : int
            Number of layers and number of blocks.
        """
        last_layer: dict[int, int] = {}
        for qs in blocks:
            start = max((last_layer.get(q, 0) for q in qs), default=0)
            for q in qs:
                last_layer[q] = start + 1
        return (max(last_layer.values()) if last_layer else 0, len(blocks))

    @staticmethod
    def _leaf_stats(gate: Gate, cache: dict) -> tuple[int, int]:
        """Return ``(depth, gate_count)`` of ``gate`` (forward direction).

        Counted at <=2-qubit block granularity; see :meth:`_leaf_blocks`.

        Parameters
        ----------
        gate : qiskit.circuit.Gate
            Leaf gate.
        cache : dict
            Leaf-block cache shared across calls.

        Returns
        -------
        depth, gate_count : int
            Number of layers and number of blocks.
        """
        return CircuitDecomposition._blocks_stats(CircuitDecomposition._leaf_blocks(gate, cache))

    @staticmethod
    def _schedule_stats(
        schedule: list[tuple[Gate, tuple[int, ...]]], cache: dict
    ) -> tuple[int, int]:
        """Return ``(depth, gate_count)`` of a flat schedule.

        The schedule is a list of (leaf gate, local qubits) pairs, as built
        by ``_build_U_block_encoding``; per-qubit greedy scheduling over each
        entry's own leaf statistics is exact.

        Parameters
        ----------
        schedule : list of (Gate, tuple of int)
            Leaf gates with the local qubits they act on.
        cache : dict
            Leaf-block cache shared across calls.

        Returns
        -------
        depth, gate_count : int
            Number of layers and number of blocks.
        """
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
        """Return the exact resources of the circuit of :meth:`synthesize`.

        The tree is built exactly as :meth:`synthesize` builds it (same
        registers, leaf gates and segment programs, :meth:`_build_tree`), but
        the root program is scheduled directly instead of expanded into a
        circuit: greedy as-soon-as-possible layering of the <=2-qubit leaf
        blocks (:meth:`_program_depth`) and their number
        (:meth:`_program_count`). A segment's schedule depends only on its
        structure and on when its qubits become free (up to a common shift),
        so identical subtrees (the uniform bulk) and the ``2 ell + 1``
        replays of every child are scheduled once and memoized. The
        flattened circuit, whose size grows like
        ``N^(1 + log2(2 ell + 1))``, is never built.

        Parameters
        ----------
        pool_ancillas : bool, default False
            As in :meth:`synthesize`.
        pool_levels : int, optional
            As in :meth:`synthesize`.

        Returns
        -------
        dict
            ``depth`` and ``gate_count`` in <=2-qubit leaf blocks (arbitrary
            one- and two-qubit unitaries, the stopping rule of
            ``_leaf_blocks``; not transpiled ``["u", "cx"]`` gates), and
            ``num_qubits``. The three values are exact: they equal those of
            the circuit :meth:`synthesize` returns, flattened with the same
            rule.

        Warns
        -----
        RuntimeWarning
            If Qiskit cannot decompose a leaf gate; the estimate is then a
            lower bound (see ``_leaf_blocks``).
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

    # --- analytic estimator ---
    # The same recursion as estimate_synthesis_stats, as a pure function of
    # N-independent leaf data (synthesis_cost_params,
    # analytic_synthesis_stats).

    def synthesis_cost_params(self, cache: dict | None = None) -> dict:
        """Extract the N-independent data needed by :meth:`analytic_synthesis_stats`.

        For a uniform chain every size-independent ingredient of the circuit
        is fixed: the three site isometries (left boundary, bulk, right
        boundary), the U and U-dagger schedules of the bond, ``R_Phi`` and
        the register sizes. They are stored as their <=2-qubit leaf blocks in
        both directions (a leaf and its ``inverse()`` are synthesized
        independently by Qiskit and generally differ), so that the circuit of
        any ``N`` can be scheduled from them exactly.

        Parameters
        ----------
        cache : dict, optional
            Leaf-block cache (see ``_leaf_blocks``) to reuse across calls.

        Returns
        -------
        dict
            Plain lists and integers (JSON round trips are allowed) with the
            keys ``format``, ``d``, ``n_bond_per_site``, ``n_lcu``,
            ``n_rot``, ``ell``, ``u_width``, ``site``, ``u``, ``u_dag`` and
            ``R_Phi``. A gate is stored as ``[fwd, inv]``: the lists of its
            <=2-qubit leaf blocks (each a list of local qubit indices) and
            those of its inverse. ``site`` maps ``"left"``, ``"bulk"`` and
            ``"right"`` to such a pair; ``u`` and ``u_dag`` are the U and
            U-dagger schedules as lists of ``[fwd, inv, qubits]``, with
            ``qubits`` the local qubits the gate acts on; ``R_Phi`` is the
            pair of the zero-state reflection of the merge registers.

        Raises
        ------
        ValueError
            If ``N < 3`` (no bulk site) or if the bonds do not all share one
            merge bundle (not a uniform chain).
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
        """Return :meth:`estimate_synthesis_stats` for any ``N`` from stored leaf data.

        The tree of an ``N``-site chain is rebuilt from the N-independent
        leaf data (site isometries, U / U-dagger schedules, ``R_Phi``) with
        the same builder as :meth:`synthesize` and scheduled exactly, so the
        result is exact: it equals the flattened circuit (<=2-qubit leaf
        blocks) that :meth:`synthesize` would return for that ``N``. Only the
        reflections, whose width grows with ``N``, are synthesized on the fly
        (one per distinct width).

        Each tree layer replays its children ``2 ell + 1`` times (``V_pair``
        once and ``R_Psi = V_pair Ref_0 V_pair^-1`` in each of the ``ell``
        Grover iterates), hence depth ~ ``N^log2(2 ell + 1)`` and gate count
        ~ ``N^(1 + log2(2 ell + 1))``.

        Parameters
        ----------
        params : dict
            Output of :meth:`synthesis_cost_params` (JSON round trips are
            allowed).
        N : int
            Number of sites (``>= 2``).
        pool_ancillas : bool, default False
            As in :meth:`synthesize`: one lcu/rotation register per tree
            layer. Siblings then share qubits and serialize; fewer idle
            qubits can also change the borrowed-ancilla construction of the
            reflections.
        pool_levels : int, optional
            Pool only the lowest ``pool_levels`` layers.
        borrowed_ancillas : bool, default True
            If False, every reflection is priced as a plain multi-controlled
            X without borrowed ancillas, for comparison (not what
            :meth:`synthesize` builds).

        Returns
        -------
        dict
            ``{"depth", "gate_count", "num_qubits"}``; depth and gate count
            in units of <=2-qubit leaf blocks.

        Raises
        ------
        ValueError
            If ``params`` has an unsupported ``format`` or ``N < 2``.
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
        oblivious amplitude amplification (thesis chapter "MPU
        decomposition", ``sec:tree_merging``; Styliaris et al., PRL 135,
        260602). The circuit is assembled from a flat list of leaf gates (one
        ``UnitaryGate``/isometry per entry, never re-wrapped), so the
        construction time is proportional to the number of leaves.

        Parameters
        ----------
        unitary_label : str, optional
            Ignored; accepted for interface compatibility.
        pool_ancillas : bool, default False
            Recycle ancillas per tree layer: every merge of a layer uses the
            same lcu and rotation register (one pair per layer instead of one
            per merge). Sibling subtrees then act on shared qubits and run
            one after the other, trading depth for qubits. Registers of
            *different* layers are never shared, because a parent's
            reflection replays its children while the parent's own ancillas
            are active. Overrides ``pool_levels``.
        pool_levels : int, optional
            Pool only the lowest ``pool_levels`` layers (default: none).

        Returns
        -------
        qc_global : qiskit.QuantumCircuit
            Circuit implementing :math:`U_N` on the system registers, with
            every other qubit starting and ending in :math:`|0\rangle`, up to
            a global phase.
        site_sys_regs : list of QuantumRegister
            System register of each site (site 0 first),
            :math:`\lceil\log_2 d\rceil` qubits each.
        site_bond_regs : list of QuantumRegister
            Bond register of each site (right and left bond qubits).
        anc_regs : list of list of (QuantumRegister, QuantumRegister)
            Per tree layer, the ``(lcu, rotation)`` ancilla registers of each
            merge (the same pair repeated when pooled).
        all_anc_qubits : list
            Every non-system qubit (bond and ancilla qubits), for
            post-selection.

        Notes
        -----
        Pooling is exact under two conditions: every tree block is an
        isometry, and the block encoding of each merging operator maps its
        ancilla ``|0>`` input and output through the same row ``e_0``
        (property BE*). Both hold for every encoding built here (the second
        is false for a generic block encoding). Under them the reflection on
        the current merge's registers only is equivalent to the thesis's
        projector on all auxiliaries. The argument is an induction on the
        tree (see :meth:`_merge_layer`, not a labelled thesis result) and is
        validated numerically; for optimized boundaries the deviation is of
        the order of the isometry residual.

        Gate counts and depth of the returned circuit refer to *composite*
        gates (multi-qubit ``UnitaryGate``/isometry objects). For elementary
        resources transpile first, e.g.
        ``qiskit.transpile(qc, basis_gates=["u", "cx"])``, or use
        :meth:`estimate_synthesis_stats` / :meth:`synthesis_cost_params`.
        Without pooling, ancilla registers are allocated per merge.
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
        """Build the registers and the root segment program.

        Shared by :meth:`synthesize` and :meth:`estimate_synthesis_stats`.

        Parameters
        ----------
        pool_ancillas : bool, default False
            Use one lcu/rotation register per layer for all layers.
        pool_levels : int, optional
            Use one register per layer for the lowest ``pool_levels`` layers.

        Returns
        -------
        dict
            ``master`` (root segment), ``all_regs``, ``site_sys_regs``,
            ``site_bond_regs`` and ``anc_regs``.

        Raises
        ------
        RuntimeError
            If the tree does not end in a single canonical segment.
        """
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
    indexed by the bond it belongs to and computed once, with one left and
    one right sweep over the chain. By default the chain is first brought to
    canonical form (minimal bond dimensions, :math:`L_k=I`,
    :math:`R_k=\mathrm{diag}(s)`), which removes the need for Assumption 1
    (thesis ``sec:nonuniform_mpus``). The depth is governed by
    :math:`q=\max_k q_k` (``thm:nonuniform_depth``).

    Parameters
    ----------
    tensors_A : list of ndarray
        Site tensors ``A_k[p_in, p_out, bond_L, bond_R]`` with a common
        ``d``; at least two.
    l_vec, r_vec : ndarray
        Boundary vectors, contracted into the first / last bond.
    tol : float, default 1e-12
        Tolerance of the full-rank checks of the boundary operators.
    canonicalize : bool, default True
        Bring the chain to canonical form first. Bond dimensions may then be
        smaller than the input ones.
    canonical_tol : float, default 1e-10
        Relative singular-value threshold of the canonicalization.
    boundary_sigma_tau : bool, default False
        Optimize a density matrix per site near each chain end
        (:func:`~TreeMPU.utils.optimize_chain_sigma_tau`); a bond keeps the
        result only if its :math:`q_k` decreases. A failed optimization falls
        back to :math:`\sigma=\tau=I/d` with a ``RuntimeWarning``.
    boundary_sigma_tau_max_sites : int, optional
        Number of sites nearest each end with a free density matrix
        (``None``: all).
    boundary_sigma_tau_n_restarts : int, default 4
        Passed to :func:`~TreeMPU.utils.optimize_chain_sigma_tau`.
    boundary_sigma_tau_seed : int, default 0
        Passed to :func:`~TreeMPU.utils.optimize_chain_sigma_tau`.
    boundary_opt_p : int, optional
        Partial optimization (thesis section "Partial optimization"): for
        windows of up to :math:`2^p` sites that reach a chain end, replace the
        boundary operator of the bond by a directly optimized one
        (:func:`~TreeMPU.utils.optimize_boundary_operators`). A replacement is
        kept only if :math:`q_k` does not increase, the optimizer's isometry
        residual is small and the affected site passes an isometry check.
    boundary_opt_gauge : bool, default False
        Reserved for unitary-gauge optimization, which is not supported per
        window. ``True`` together with ``boundary_opt_p`` raises
        ``ValueError``; without ``boundary_opt_p`` the flag is not read and
        has no effect.
    boundary_opt_n_restarts : int, default 10
        Passed to :func:`~TreeMPU.utils.optimize_boundary_operators`.
    boundary_opt_lam_iso : float, default 1e6
        Isometry penalty weight, passed to
        :func:`~TreeMPU.utils.optimize_boundary_operators`.
    boundary_opt_seed : int, default 0
        Passed to :func:`~TreeMPU.utils.optimize_boundary_operators`.

    Attributes
    ----------
    q : float
        Largest per-bond conditioning number, :math:`\max_k q_k` (thesis
        ``eq:q_nonuniform``). The individual :math:`q_k` are held in the
        private list ``_q_bonds``, an implementation detail; a public
        per-bond value is
        ``conditioning_number(mpu.L[k], mpu.R[k])``
        (:func:`~TreeMPU.utils.conditioning_number`).
    tol : float
        Tolerance of the full-rank checks of the boundary operators.
    boundary_source : list of str
        Boundary strategy per bond: ``"canonical"``,
        ``"sigma_tau_optimized"`` (``boundary_sigma_tau``) or ``"optimized"``
        (``boundary_opt_p``). The names differ from those of
        :class:`UniformMPU` (``"canonical_sigma_tau"`` there).
    tensors : list of ndarray
        The (canonicalized) site tensors actually used.
    canonicalized : bool
        Whether the chain was brought to canonical form.
    L, R, L_inv, R_inv : list of ndarray
        Per-bond boundary operators ``L[ext, bond]``, ``R[ext, bond]`` and
        their inverses.
    MergingOperators : list of ndarray
        Per-bond merging operators with kernel :math:`R^{-\top}L^{-1}`.
    l_vec, r_vec : quimb.tensor.Tensor
        Boundary vectors (after canonicalization), see
        :class:`CircuitDecomposition`.

    Raises
    ------
    ValueError
        Malformed tensors, inconsistent bond dimensions, failed normalization
        check, canonical form collapsing below two tensors, rank-deficient
        boundary operators (with ``canonicalize=False``), or
        ``boundary_opt_gauge=True`` together with ``boundary_opt_p``.
    NotImplementedError
        If there are fewer than two tensors.

    Warns
    -----
    RuntimeWarning
        If the sigma/tau chain optimization fails.

    Notes
    -----
    ``boundary_source`` never contains ``"optimized_gauge"``: gauge dressing
    would need one unitary acting on both sides of a window, but every window
    is anchored at a chain end and has a trivial bond there, so the gauge
    option is rejected together with ``boundary_opt_p``.

    Partial optimization anchors its windows at a chain end only. Boundaries
    it accepts are validated on a single-site isometry check, so isometry of
    longer blocks is not verified at run time. Site-dependent windowed
    boundaries can reach :math:`q_k<\chi`; :math:`q\ge\chi` is a target of
    the uniform strategies, not a bound (see ``docs/userguide.md``,
    section 6).
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
        N = len(tensors_A)

        if N < 2:
            raise NotImplementedError("GeneralMPU.synthesize currently requires N >= 2.")

        # --- input validation ---

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

        # --- state ---

        self.tensors = [np.asarray(A, dtype=complex) for A in tensors_A]
        self.tol = tol

        # Assumption 1 is not needed here: the canonical form (minimal bond
        # dimension at every cut, L_k = 1, R_k = diag(s_k) with s_k > 0,
        # Eq. 26) removes rank deficiency, which is a property of the
        # representation, not of the MPU. UniformMPU cannot canonicalize
        # because its bulk must remain one repeated tensor (footnote 2 on
        # Theorem 1').
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

        # --- 3.1. Optional sigma/tau-optimized baseline ---
        # Every candidate is a valid density-matrix pair, so there is no
        # isometry constraint to fail; a bond keeps it only if it beats the
        # I/d pair. An optimizer that fails falls back to I/d (with a
        # warning) rather than aborting construction.
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

        # --- 3.5. Optional partial optimization (thesis "Partial
        # Optimization") ---
        # Only bonds whose window reaches a chain boundary (l_vec or r_vec)
        # within 2**boundary_opt_p sites are affected; improved bonds overwrite
        # boundary_source individually.

        if boundary_opt_p is not None:
            if boundary_opt_gauge:
                # Every window is anchored to l_vec or r_vec, so it has a
                # trivial bond on the anchored side and the chain's bond
                # dimension on the other; one unitary gauge cannot dress both
                # (L G^dagger, G R).
                raise ValueError(
                    "boundary_opt_gauge=True cannot be combined with "
                    "boundary_opt_p: every window it optimizes is anchored "
                    "to the chain's own l_vec/r_vec, giving it a trivial "
                    "bond on that side (after canonicalization) that a "
                    "shared gauge transformation cannot act on together "
                    "with the other, non-trivial side."
                )
            max_window = 2**boundary_opt_p
            # boundary_opt_gauge is False below (rejected above); the gauge
            # branches are kept for when gauge dressing is supported per window.

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

        if "optimized" in self.boundary_source:
            worst = max(self.segment_isometry_residuals().values(), default=0.0)
            if worst > _SEGMENT_ISOMETRY_WARN_TOL:
                warnings.warn(
                    "boundary_opt_p accepted windows whose merged tree segments are not "
                    f"exact isometries (largest residual {worst:.2e}); the circuit is "
                    "exact only up to this residual. Use boundary_opt_p=None or "
                    "the canonical/sigma_tau boundaries for an exact synthesis.",
                    RuntimeWarning,
                    stacklevel=2,
                )

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
        r"""Return the conditioning number :math:`q_b` of bond ``b``.

        :math:`q_b=\sqrt{\mathrm{Tr}(R_b^{-2}(L_b^{-2})^T)}` (Eq. 17), equal
        to :math:`\|R_b^{-\top}L_b^{-1}\|_F`.

        Parameters
        ----------
        b : int
            Bond index, ``0 <= b < N - 1``.

        Returns
        -------
        float
            :math:`q_b`, computed from the current ``L_inv[b]``, ``R_inv[b]``.
        """
        L2_inv = self.L_inv[b] @ self.L_inv[b]
        R2_inv = self.R_inv[b] @ self.R_inv[b]
        return float(np.real(np.sqrt(np.trace(R2_inv @ L2_inv.T))))

    def create_local_isometries(self) -> list[qtn.Tensor]:
        """Return the dressed local isometry tensors ``V_0, ..., V_{N-1}``.

        Mirrors :meth:`UniformMPU.create_local_isometries` for site-dependent
        tensors and per-bond operators: site ``k`` is dressed on its left
        with ``L[k - 1]`` (or ``l_vec`` if ``k == 0``) and on its right with
        ``R[k]`` (or ``r_vec`` if ``k == N - 1``). Each isometry maps the
        physical input leg ``p_in_*`` to the output leg ``p_out_*`` and the
        external bonds ``ext_L``/``ext_R``.

        Returns
        -------
        list of quimb.tensor.Tensor
            One tensor per site.
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
        """Return the merging operator of bond ``bond_index``.

        Per-bond accessor mirroring :class:`UniformMPU`; not used by the
        synthesis, which reads the precomputed per-bond bundles.
        """
        return self.MergingOperators[bond_index]

    def _build_lcu_data(self, bond_index: int) -> tuple[list, list, float]:
        """Return ``(coefficients, unitaries, 1-norm)`` of the LCU of a bond.

        Per-bond accessor mirroring :class:`UniformMPU`; not used by the
        synthesis.
        """
        bundle = self._goaa_bonds[bond_index]
        return bundle["coef_0"], bundle["unit_0"], bundle["C_0"]

    def _build_goaa_data(self, bond_index: int) -> dict:
        """Return the stored GOAA bundle of bond ``bond_index``.

        Per-bond accessor mirroring :class:`UniformMPU`; not used by the
        synthesis.
        """
        return self._goaa_bonds[bond_index]

    def _tree_segments(self) -> list[tuple[int, int]]:
        """Return the site ranges ``(first, last)`` of every segment of the merge tree."""
        level = [(k, k) for k in range(self._N)]
        out = list(level)
        while len(level) > 1:
            merged = [(level[i][0], level[i + 1][1]) for i in range(0, len(level) - 1, 2)]
            if len(level) % 2:
                merged.append(level[-1])
            out.extend(seg for seg in merged if seg not in out)
            level = merged
        return out

    def segment_isometry_residuals(self, max_dim: int = 2**10) -> dict[tuple[int, int], float]:
        r"""Return the isometry residual of every merge-tree segment.

        For the segment of sites ``first..last`` the dressed block is
        :math:`V=L_{\rm first-1}A_{\rm first}\cdots A_{\rm last}R_{\rm last}`
        (with :math:`\ell` or :math:`r` at the chain ends), a map from the
        physical input legs of the segment to (outputs, external bonds). The
        tree synthesis is exact only if every such block is an isometry.

        Parameters
        ----------
        max_dim : int
            Skip segments whose input dimension :math:`d^{n}` exceeds this
            (the check builds dense blocks).

        Returns
        -------
        dict
            ``{(first, last): ||V^dagger V - 1||_F}`` for the checked segments.
        """
        d = self._d
        residuals = {}
        for first, last in self._tree_segments():
            if d ** (last - first + 1) > max_dim:
                continue
            X = self.tensors[first]
            for k in range(first + 1, last + 1):
                X = np.einsum("IJac,ijcb->IiJjab", X, self.tensors[k])
                X = X.reshape(X.shape[0] * X.shape[1], X.shape[2] * X.shape[3], *X.shape[4:])
            left = self.l_vec.data[None, :] if first == 0 else self.L[first - 1]
            right = self.r_vec.data[None, :] if last == self._N - 1 else self.R[last]
            V = np.einsum("xa,IJab,yb->xJyI", left, X, right).reshape(-1, X.shape[0])
            residuals[(first, last)] = float(np.linalg.norm(V.conj().T @ V - np.eye(V.shape[1])))
        return residuals

    def _site_isometry_tensor(self, k: int) -> qtn.Tensor:
        return self._site_tensors[k]

    def _bond_qubit_count(self, bond_index: int) -> int:
        return self._bond_qubits[bond_index]

    def _bond_dimension(self, bond_index: int) -> int:
        return self._bond_dims[bond_index]

    def _merge_bundle(self, bond_index: int) -> dict:
        return self._bond_bundles[bond_index]

    def _isometry_tol(self) -> float:
        """Return the isometry tolerance (tight unless a bond was optimized)."""
        # Environment-based boundaries (canonical, and the sigma/tau optimum
        # with density matrices on the input leg) are exactly isometric.
        if all(src in ("canonical", "sigma_tau_optimized") for src in self.boundary_source):
            return 1e-10
        # At least one bond comes from optimize_boundary_operators, which
        # enforces the isometry only to penalty-method precision.
        return 1e-4


class UniformMPU(CircuitDecomposition):
    r"""Translation-invariant MPU defined by one bulk tensor and two boundary vectors.

    .. math::

        U_N=\sum \ell_{v_0} r_{v_N}\,A[v_0,v_1]\otimes\cdots\otimes
        A[v_{N-1},v_N]

    (thesis ``def:uniform_mpu``). Exact unitarity is checked at construction
    (:func:`~TreeMPU.checks.check_mpo_unitarity`), a boundary pair
    :math:`(L,R)` is chosen among the strategies below, and the circuit is
    produced by :meth:`synthesize`.

    Boundary strategies, in priority order (value of ``boundary_source``, the
    option that selects it in parentheses):

    - ``"mode1_exact"``: closed form with :math:`q=\chi` for Mode I tensors
      with single-vertex readout (``boundary_mode1_exact``).
    - ``"qD_general"``: :func:`~TreeMPU.utils.solve_qD_boundary`, aiming at
      :math:`q=\chi` for general bulk tensors (``boundary_opt_k``).
    - ``"optimized"``, ``"optimized_gauge"``:
      :func:`~TreeMPU.utils.optimize_boundary_operators` (``boundary_opt_k``,
      with ``boundary_opt_gauge`` for the gauge variant).
    - ``"canonical_sigma_tau"``: canonical boundaries with optimized
      :math:`\sigma,\tau` (``boundary_sigma_tau``, default), used only if it
      lowers :math:`q` relative to ``"canonical"``.
    - ``"canonical"``: canonical boundaries with :math:`\sigma=\tau=I/d`
      (Styliaris et al.; requires Assumption 1).

    :class:`GeneralMPU` names the sigma/tau strategy ``"sigma_tau_optimized"``.

    Parameters
    ----------
    A : ndarray, shape (d, d, chi, chi)
        Bulk tensor ``A[p_in, p_out, bond_L, bond_R]``.
    l_vec, r_vec : ndarray, shape (chi,)
        Boundary vectors.
    N : int
        Number of sites (``N >= 2``).
    tol : float, default 1e-12
        Tolerance of the unitarity check
        (:func:`~TreeMPU.checks.check_mpo_unitarity`).
    debug : bool, default False
        Also verify the LCU decomposition and the block encoding numerically.
    light : bool or int, default False
        Lengths covered by the exact unitarity check: ``False`` all
        :math:`1..N`; ``True`` :math:`1..\min(N,\chi^2+1)`, which by the
        thesis finite-size cutoff theorem certifies every :math:`N` at a
        cost independent of :math:`N`; an integer :math:`k`:
        :math:`1..\min(N,k)`. The full check is the dominant cost of the
        constructor, and its time and memory grow quickly with :math:`N`.
    boundary_sigma_tau : bool, default True
        Try canonical boundaries with optimized :math:`\sigma,\tau`
        (:func:`~TreeMPU.utils.optimize_q_unif`); adopted only if :math:`q`
        decreases.
    boundary_mode1_exact : bool, default False
        Try :func:`~TreeMPU.utils.mode1_exact_boundary` (Mode I tensors with
        :math:`\ell=e_v`, :math:`r=\sum_v e_v`; :math:`q=\chi`). Falls back
        with a ``RuntimeWarning`` if the structure is absent.
    boundary_opt_k : int, optional
        Try :func:`~TreeMPU.utils.solve_qD_boundary` and, if it fails,
        :func:`~TreeMPU.utils.optimize_boundary_operators` with the isometry
        constraint imposed at block lengths :math:`1..k`. Every candidate is
        re-validated with an exact isometry check before being accepted.
    boundary_opt_gauge : bool, default False
        Also optimize a unitary gauge (strictly larger feasible set); skips
        ``solve_qD_boundary``, which returns HPD boundaries only.
    boundary_opt_n_restarts : int, default 10
        Passed to the optimizers.
    boundary_opt_lam_iso : float, default 1e6
        Isometry penalty weight, passed to
        :func:`~TreeMPU.utils.optimize_boundary_operators`.
    boundary_opt_seed : int, default 0
        Random seed of the optimizers.

    Attributes
    ----------
    A : quimb.tensor.Tensor
        Bulk tensor with indices ``("p_in", "p_out", "bond_0", "bond_N")``.
    l_vec, r_vec : quimb.tensor.Tensor
        Boundary vectors, see :class:`CircuitDecomposition`.
    tol : float
        Tolerance of the unitarity check.
    unitarity_checked_up_to : int
        Largest chain length at which exact unitarity was checked.
    unitarity_certified_all_N : bool
        Whether that suffices for every :math:`N` (lengths up to
        :math:`\chi^2+1` checked).
    boundary_source : str
        Strategy that produced ``L``, ``R`` (table above).
    L, R, L_inv, R_inv : ndarray, shape (chi, chi)
        Chosen boundary operators ``L[ext, bond]``, ``R[ext, bond]`` and their
        inverses.
    MergingOperator : ndarray, shape (chi**2, chi**2)
        Rank-one merging operator with kernel :math:`R^{-\top}L^{-1}`
        (:func:`~TreeMPU.utils.get_merging_operator`).
    q_unif : float
        Conditioning number :math:`q` of the chosen boundary (LCU 1-norm of
        the merging operator).
    ell : int
        Number of GOAA rounds per merge.
    C : float
        LCU 1-norm of the merging operator after lifting :math:`q` onto the
        GOAA target with rotation qubits.
    mode1_exact : bool
        True if the boundary comes from the ``mode1_exact`` closed form
        (``boundary_mode1_exact=True`` and the structure applies).
    qD_general : bool
        True if the boundary comes from ``solve_qD_boundary``.
    qD_diagnostics : dict or None
        Diagnostics returned by :func:`~TreeMPU.utils.solve_qD_boundary`
        (keys ``cond1_residual``, ``cond2_residual``, ``bulk_defect_at_fit``,
        ``bulk_defect_out_of_sample``, ``tr_L2_R2T``, ``L2_min_eigenvalue``,
        ``null_dim_L2_family``); ``None`` unless ``qD_general`` is True.
    q_baseline : float or None
        :math:`q` of the canonical pair that the optimizer result is compared
        against; ``None`` unless ``optimize_boundary_operators`` ran and a
        canonical pair exists.
    boundary_opt_result : dict or None
        Record of the ``optimize_boundary_operators`` run, ``None`` if it did
        not run. Keys: ``"candidates"`` (dict from the start tag ``"cold"`` /
        ``"warm"`` to a dict with ``"L"``, ``"R"``, ``"q"``,
        ``"iso_residual"``; ``"warm"`` is absent without a canonical
        baseline), ``"q_baseline"`` and ``"chosen"`` (tag of the adopted
        candidate, or ``None`` if none was accepted). This is the way to
        inspect a failed optimization.
    L_id, R_id : ndarray or None
        Canonical pair with :math:`\sigma=\tau=I/d`; ``None`` if Assumption 1
        fails.
    sigma_opt, tau_opt : ndarray or None, shape (d, d)
        Optimized density matrices of the ``"canonical_sigma_tau"`` pair;
        ``None`` if that pair was not adopted.
    L_canonical, R_canonical : ndarray or None
        Starting pair of the boundary selection: the ``sigma_opt``/``tau_opt``
        pair if adopted, else ``L_id``, ``R_id``; replaced by the
        ``mode1_exact`` or ``qD_general`` pair when that strategy succeeds.
        ``None`` if Assumption 1 fails and no free construction applies.

    Raises
    ------
    ValueError
        If the MPU is not unitary, ``light`` caps the check below one site,
        or no admissible boundary could be built (Assumption 1 fails and no
        free construction succeeded).
    NotImplementedError
        If ``N < 2``.

    Warns
    -----
    RuntimeWarning
        If Assumption 1 fails while a free construction was requested, if
        ``mode1_exact`` does not apply, if the sigma/tau optimization fails,
        or if an optimizer candidate passes its own isometry report but fails
        the exact check.

    Notes
    -----
    Assumption 1 (injective length-1 boundary maps) is required only by the
    canonical constructions; when a free construction is requested its
    failure is a warning. A boundary with :math:`q=\chi` from ``mode1_exact``
    or ``qD_general`` is taken as final (the optimizers are skipped).

    Conditioning number. :math:`q\ge1` is proved for admissible boundaries.
    By Cauchy-Schwarz :math:`q\,\|LR^\top\|_F\ge\chi`, so :math:`q\ge\chi` is
    proved whenever :math:`\|LR^\top\|_F\le1`, which covers the
    environment-based boundaries (see ``docs/userguide.md``, section 6). It
    is not a bound in general: :math:`q=\chi` is a target of ``mode1_exact``
    and ``qD_general``, and site-dependent windowed boundaries
    (``GeneralMPU(boundary_opt_p=...)``) reach :math:`q<\chi`.

    Candidate selection. An ``optimize_boundary_operators`` candidate is
    usable only if :math:`q\ge1` (:math:`q<1` is unusable by GOAA, since
    :math:`\arcsin(1/C)` is undefined), its self-reported isometry residual is
    small and it does not exceed the canonical :math:`q` (within the
    optimizer's convergence noise). It then also needs the exact check of
    ``_validate_boundary_pair``, because the optimizer's residual comes from a
    bond-space estimate. A candidate that passes the self-reported test but
    fails the exact check triggers a ``RuntimeWarning``; an optimizer that
    simply does not beat the canonical pair falls back silently. The best
    surviving candidate (smallest :math:`q`) is adopted.

    Validation of iterative boundaries. Boundaries from iterative solvers are
    validated with an exact isometry check on the length-1 blocks (tolerance
    ``1e-6``), accepted at isometry tolerance ``1e-4`` during synthesis and
    snapped to the nearest exact isometry; isometry of longer blocks is
    validated numerically only. Use
    :func:`~TreeMPU.checks.block_isometry_residuals` to quantify the
    residual.
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
        if N < 2:
            raise NotImplementedError("UniformMPU.synthesize currently requires N >= 2.")

        # --- input validation and unitarity check ---
        # `light` caps the number of lengths checked: the check is the
        # constructor's dominant cost (see the `light` parameter).
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

        # Assumption 1 is needed only to make the canonical (sigma/tau or I/d)
        # construction produce invertible L, R. mode1_exact_boundary,
        # solve_qD_boundary and optimize_boundary_operators each guarantee
        # their own invertibility. Hence a failure is only a warning when one
        # of these free constructions was requested, and an error otherwise.
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

        # --- state ---

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

        # --- boundary operators ---
        self.mode1_exact = False
        self.qD_general = False
        self.qD_diagnostics = None
        self.q_baseline = None
        self.boundary_opt_result = None

        if assumption_1_ok:
            # Baseline: the sigma/tau-optimized canonical pair, not plain I/d.
            # I/d is one arbitrary feasible point; optimizing the density
            # matrices stays inside the canonical construction (isometry at
            # every length, nothing to enforce) and can lower q. Kept only if
            # q decreases.
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
            # No canonical fallback: overwritten below if boundary_mode1_exact
            # or boundary_opt_k succeeds, else rejected by the check before
            # self.L_inv.
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
            # General q = chi construction first, validated at the support
            # lengths requested for the optimizer. Skipped if mode1_exact
            # already reached q = chi, and when boundary_opt_gauge=True, which
            # asks for the larger non-HPD feasible set that solve_qD_boundary
            # (HPD square roots only) cannot provide.
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
                    # The solver's own residuals are necessary but not
                    # sufficient: they use a fast bond-space estimate that can
                    # disagree with the true isometry defect, so the candidate
                    # also gets the exact check (_validate_boundary_pair).
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
                # Compared against the sigma/tau baseline when available (not
                # against I/d): the optimizer is adopted only if it beats what
                # is free. Without a canonical baseline (Assumption 1 failed)
                # a merely feasible result is adopted, from a cold start only.
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

                # Cold start (identity, then random restarts) and warm start
                # from the canonical sigma/tau pair are both run, and the best
                # validated candidate is kept: neither dominates in general.
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

                # Selection rule: see "Candidate selection" in the class Notes.
                best = None
                saw_self_reported_ok_but_invalid = False
                for tag, L_c, R_c, q_c, iso_c in candidates:
                    feasible = q_c >= 1.0 - 1e-6 and iso_c < 1e-3
                    # 1e-6: a candidate equal to canonical within the optimizer's
                    # convergence noise in q is kept.
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

        # --- eager precomputation of circuit-synthesis data ---
        # Everything below is independent of N, so it is computed once here
        # rather than inside synthesize().
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
        """Return whether ``(L_cand, R_cand)`` gives exact local isometries.

        Builds every site isometry with the candidate pair through
        ``_build_isometry_unitary``, the same check that ``synthesize`` runs,
        so that a bad boundary is detected at construction instead of during
        gate construction (or never, if only ``q_unif``/``ell`` are used).
        This check is independent of the bond-space estimate used inside
        ``optimize_boundary_operators``
        (``utils._bondspace_isometry_defect``), which can disagree with the
        true ``||V^dagger V - I||_F``.

        Parameters
        ----------
        L_cand, R_cand : ndarray
            Candidate boundary operators ``L[ext, bond]``, ``R[ext, bond]``.
        tol : float, default 1e-6
            Isometry-acceptance tolerance.

        Returns
        -------
        bool
            True if all site isometries (left boundary, bulk, right
            boundary) pass; ``self.L`` and ``self.R`` are restored either way.
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
        # Canonical boundaries are isometric for any density matrices
        # sigma, tau, and mode1_exact is closed form. "optimized",
        # "optimized_gauge" satisfy the isometry only to penalty-method
        # precision and "qD_general" accepts a 1e-6 solver residual, so they
        # get the looser tolerance.
        if self.boundary_source in ("canonical", "canonical_sigma_tau", "mode1_exact"):
            return 1e-10
        # The dilation snaps optimizer outputs to the nearest exact isometry
        # (see dilate_isometry_to_unitary); the looser tolerance only decides
        # which optimizer outputs are usable, not the exactness of the
        # circuit.
        return 1e-4

    def create_local_isometries(self) -> tuple[qtn.Tensor, qtn.Tensor, qtn.Tensor]:
        """Return the local isometries of the left, bulk and right sites.

        Each isometry maps the physical input leg ``p_in_*`` to the output leg
        ``p_out_*`` and the external bonds; the bulk tensor has the layout
        ``A[p_in, p_out, bond_L, bond_R]``.

        Returns
        -------
        V_l : quimb.tensor.Tensor
            Left-boundary isometry ``l A R`` with external leg ``ext_R``.
        V_b : quimb.tensor.Tensor
            Bulk isometry ``L A R`` with external legs ``ext_L`` and
            ``ext_R``.
        V_r : quimb.tensor.Tensor
            Right-boundary isometry ``L A r`` with external leg ``ext_L``.
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
        r"""Return the boundary operators :math:`L, R` from the environments.

        :math:`L^2` and :math:`R^2` are obtained by contracting the first and
        last site with its conjugate, the boundary vector and a density
        matrix on the input leg (output traced), as in thesis
        ``eq:partial_trace_unitarity``. The stored ``L`` and ``R`` are the
        Hermitian square roots of the *complex conjugates* of these contracted
        matrices :math:`L^2`, :math:`R^2`: the site isometries contract the
        second axis of ``L[ext, bond]`` and ``R[ext, bond]`` with the bond, so
        they need the conjugate of the environment Gram matrix (the two
        coincide for real environments).

        Parameters
        ----------
        sigma, tau : ndarray, shape (d, d), optional
            Density matrices closing the physical legs of the first / last
            site. ``None`` selects the maximally mixed state ``I/d``. They
            are free parameters; :func:`~TreeMPU.utils.optimize_q_unif`
            chooses them to lower ``q_unif`` without leaving the canonical
            construction.

        Returns
        -------
        L, R : ndarray, shape (chi, chi)
            Boundary operators ``L[ext, bond]``, ``R[ext, bond]``.

        Raises
        ------
        ValueError
            If a trace is non-positive or a contraction has the wrong shape.
        """
        sigma = np.eye(self._d) / self._d if sigma is None else np.asarray(sigma)
        tau = np.eye(self._d) / self._d if tau is None else np.asarray(tau)

        # --- left boundary contraction (L2) ---
        l_vec_conj = self.l_vec.H.reindex({"bond_0": "bond_0_dag"}).retag({"l": "l*"})
        A_1, A_1_adj = get_mpo_site_tensors(1, self.A.data)

        L2 = A_1 & self.l_vec & l_vec_conj & A_1_adj
        L2 &= qtn.Tensor(sigma, inds=["p_in_1", "p_out_f_1"], tags=["σ"])

        # --- right boundary contraction (R2) ---
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

        # --- matrix extraction and shape checks ---
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

        # --- square roots (stable Hermitian decomposition) ---
        # Conjugate, see the docstring.
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

        Returns
        -------
        float
            :math:`q` of the current ``L_inv``, ``R_inv``.

        Raises
        ------
        ValueError
            If ``L_inv`` and ``R_inv`` differ in shape.
        """
        if self.L_inv.shape != self.R_inv.shape:
            raise ValueError(f"Dimension mismatch: L2 {self.L_inv.shape} vs R2 {self.R_inv.shape}")
        return float(np.linalg.norm(self.R_inv.T @ self.L_inv))

    def _get_merging_operator(self) -> np.ndarray:
        """Return the merging operator of the chosen boundary pair."""
        return get_merging_operator(self.L_inv, self.R_inv)

    def _build_lcu_data(self):
        """Return the cached two-term LCU of the merging operator.

        The merging operator is the same at every bond; the LCU
        (``_compute_lcu``) is computed once and reused by every merge.

        Returns
        -------
        coeffs, units, C
            Coefficients, unitaries and 1-norm of the LCU, as returned by
            :meth:`CircuitDecomposition._compute_lcu`.
        """
        if self._lcu_cache is None:
            coeffs, units, C = CircuitDecomposition._compute_lcu(self.MergingOperator)
            self._lcu_cache = (coeffs, units, C)
        return self._lcu_cache

    def _build_merging_unitary(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the cached ``(B, W_ctrl)`` pair of the unrotated LCU.

        This is the raw ``U = B^dag W_ctrl B`` embedding of ``M``, before the
        rotation qubits that make the GOAA exponent ``ell`` an integer. The
        dense ``W_ctrl`` is built only for the ``debug`` verification.

        Returns
        -------
        B, W_ctrl : ndarray
            Output of :meth:`CircuitDecomposition._compute_merging_unitary`
            for the cached LCU.
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

        Returns
        -------
        dict
            The bundle returned by
            :meth:`CircuitDecomposition._compute_goaa_bundle`.
        """
        if getattr(self, "_goaa_cache", None) is None:
            self._goaa_cache = CircuitDecomposition._compute_goaa_bundle(self.MergingOperator)
            self.C = self._goaa_cache["C"]
            self.ell = self._goaa_cache["ell"]
        return self._goaa_cache


class _BlockLeaf:
    """A leaf gate known only through the <=2-qubit blocks of itself and its inverse.

    Parameters
    ----------
    blocks, inv_blocks : list of tuple of int
        Leaf blocks of the gate and of its inverse (local qubit indices).
    num_qubits : int
        Width of the gate.
    name : str, default "leaf"
        Label.
    """

    def __init__(self, blocks, inv_blocks, num_qubits: int, name: str = "leaf"):
        self.blocks = [tuple(b) for b in blocks]
        self._inv_blocks = [tuple(b) for b in inv_blocks]
        self.num_qubits = num_qubits
        self.name = name

    def inverse(self) -> _BlockLeaf:
        return _BlockLeaf(self._inv_blocks, self.blocks, self.num_qubits, self.name + "_dg")


class _ParamsTree(CircuitDecomposition):
    """The synthesis tree of an ``N``-site uniform chain from stored leaf data.

    Rebuilt from the output of :meth:`CircuitDecomposition.synthesis_cost_params`
    (see :meth:`CircuitDecomposition.analytic_synthesis_stats`); only the
    estimators are meaningful, no real gates exist.

    Parameters
    ----------
    params : dict
        Output of ``synthesis_cost_params``.
    N : int
        Number of sites.
    borrowed_ancillas : bool, default True
        If False, reflections are priced without borrowed ancillas.
    """

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
