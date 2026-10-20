r"""Verification routines for MPUs and their circuit decomposition.

Structural checks (used at construction)
    :func:`check_mpo_unitarity` (exact unitarity for lengths ``1..N_max``),
    :func:`check_assumption_1`, :func:`check_chain_unitarity`
    (normalization only), :func:`check_full_rank`.
Block-encoding checks (``debug=True``)
    :func:`verify_lcu`, :func:`verify_merging_unitary`.
Dense references (small systems)
    :func:`dense_unitary_from_tensors`, :func:`reconstruct_circuit_unitary`,
    :func:`phase_aligned_error`, :func:`block_isometry_residuals`.
"""

import logging

import numpy as np  # pyright: ignore[reportMissingImports]
from quimb import tensor as qtn  # type: ignore

from .utils import get_mpo_site_tensors

logger = logging.getLogger(__name__)


#: Floor for the (linear) normalization check of :func:`check_mpo_unitarity`:
#: contraction rounding gives |1 - Tr(rho)/d^N| ~ 1e-12 for complex tensors.
NORMALIZATION_ATOL = 1e-9


def check_mpo_unitarity(
    N_max: int, A_np: np.ndarray, l_in: np.ndarray, r_in: np.ndarray, tol: float = 1e-6
) -> None:
    r"""Check exact unitarity of a translation-invariant MPO for :math:`N=1,\dots,N_{\max}`.

    For every length :math:`N` two conditions on :math:`\rho=U_NU_N^\dagger`
    are checked (thesis ``prop:cauchy_schwarz_unitarity``):

    * spectral flatness, :math:`\mathcal P_N=(\operatorname{Tr}\rho)^2/(d^N\operatorname{Tr}\rho^2)=1`,
      which holds iff :math:`U_N` is unitary up to a scale factor;
    * normalization, :math:`\operatorname{Tr}\rho = d^N`, which fixes that
      factor to 1.

    Together they are equivalent to :math:`U_NU_N^\dagger=I`. By the thesis
    finite-size cutoff theorem (``sec:finite-size-cutoff``), passing for all
    :math:`N\le\chi^2+1` certifies unitarity for every :math:`N`.

    Parameters
    ----------
    N_max : int
        Largest chain length checked.
    A_np : ndarray
        Bulk tensor ``A[p_in, p_out, bond_L, bond_R]``.
    l_in, r_in : ndarray
        Boundary vectors.
    tol : float
        Tolerance on :math:`|1-\mathcal P_N|`. This is quadratic in the
        deviation from unitarity (:math:`1-\mathcal P_N\approx
        \|\rho-\bar\rho\|_F^2/(d^N\bar\rho^2)`), so e.g. ``tol=1e-12``
        detects relative deviations of order :math:`10^{-6}`. The
        normalization :math:`|1-\operatorname{Tr}\rho/d^N|`, which is linear,
        is checked against ``max(tol, NORMALIZATION_ATOL)`` so that rounding
        errors of the contraction do not reject unitary chains.

    Raises
    ------
    ValueError
        At the first :math:`N` where either condition fails.

    Notes
    -----
    Each step contracts growing networks for :math:`\rho` and :math:`\rho^2`;
    the cost grows quickly with :math:`N` (see ``UniformMPU(light=...)``).
    """
    l_vec = qtn.Tensor(l_in.copy(), inds=["bond_0"], tags=["l"])
    r_vec = r_in.copy()

    l_vec_conj = l_vec.H.reindex({"bond_0": "bond_0_dag"}).retag({"l": "l*"})

    single_store = l_vec & l_vec_conj
    double_store = (
        l_vec
        & l_vec_conj
        & l_vec.reindex({"bond_0": f"bond_{N_max + 1}"})
        & l_vec_conj.reindex({"bond_0_dag": f"bond_{N_max + 1}_dag"})
    )

    r_vec_tensor = qtn.Tensor(r_vec, inds=["bond_0"], tags=["r"])
    r_vec_conj = r_vec_tensor.H.reindex({"bond_0": "bond_0_dag"}).retag({"r": "r*"})

    for N in range(1, N_max + 1):
        r_loc = r_vec_tensor.reindex({"bond_0": f"bond_{N}"})
        r_loc_dag = r_vec_conj.reindex({"bond_0_dag": f"bond_{N}_dag"})

        A_k, A_k_dag = get_mpo_site_tensors(N, A_np)
        single_store &= A_k & A_k_dag.reindex({f"p_out_f_{N}": f"p_in_{N}"})

        A_k_double, A_k_dag_double = get_mpo_site_tensors(N + N_max + 1, A_np)
        double_store &= (
            A_k
            & A_k_dag.reindex({f"p_out_f_{N}": f"p_middle_{N}"})
            & A_k_double.reindex({f"p_in_{N + N_max + 1}": f"p_middle_{N}"})
            & A_k_dag_double.reindex({f"p_out_f_{N + N_max + 1}": f"p_in_{N}"})
        )

        UU_dag = single_store.copy() & r_loc & r_loc_dag
        trace_rho = UU_dag.contract()
        numerator = trace_rho**2

        UU_double = (
            double_store
            & r_loc
            & r_loc_dag
            & r_vec_tensor.reindex({"bond_0": f"bond_{N + N_max + 1}"})
            & r_vec_conj.reindex({"bond_0_dag": f"bond_{N + N_max + 1}_dag"})
        )

        denominator = (A_np.shape[0] ** N) * UU_double.contract()

        check_val = (numerator / denominator).real if denominator != 0 else np.nan
        norm_err = abs(1.0 - trace_rho / A_np.shape[0] ** N)

        error_k = abs(1.0 - float(check_val))
        logger.debug(f"Site {N:03d} | Unitarity Check: {check_val:.10f} | Error: {error_k:.2e}")

        if error_k > tol or np.isnan(error_k):
            raise ValueError(
                f"Unitarity strictly lost at N={N}. Error {error_k:.2e} exceeds tolerance {tol}."
            )
        if norm_err > max(tol, NORMALIZATION_ATOL):
            raise ValueError(
                f"Unitarity lost at N={N}: U U^dag is proportional to the identity but "
                f"Tr(U U^dag)/d^N = {trace_rho / A_np.shape[0] ** N:.6g} != 1."
            )

    logger.debug(f"MPO Unitarity verified up to N_max={N_max}.")


def check_assumption_1(
    A: np.ndarray, l_vec: np.ndarray, r_vec: np.ndarray, tol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    r"""Check Assumption 1: the length-1 boundary maps are injective.

    The maps :math:`\mathbb C^\chi\to\mathbb C^{d^2}` given by
    :math:`\ell^\top A` and :math:`A\,r` must have rank :math:`\chi` (thesis,
    Assumption 1); this makes the canonical boundary operators invertible.

    Returns
    -------
    s_left, s_right : ndarray
        Singular values of the two maps.

    Raises
    ------
    ValueError
        If either map has rank below :math:`\chi` (threshold ``tol``).
    """
    D = A.shape[2]

    left_map = np.tensordot(l_vec, A, axes=(0, 2))
    left_matrix = left_map.reshape(-1, D)
    s_left = np.linalg.svd(left_matrix, compute_uv=False)

    right_map = np.tensordot(A, r_vec, axes=(3, 0))
    right_matrix = right_map.reshape(-1, D)
    s_right = np.linalg.svd(right_matrix, compute_uv=False)

    rank_left = np.sum(s_left > tol)
    rank_right = np.sum(s_right > tol)

    if rank_left < D:
        raise ValueError(
            f"Assumption 1 failed: Left boundary map is not injective. Rank {rank_left} < D ({D})."
        )
    if rank_right < D:
        raise ValueError(
            f"Assumption 1 failed: Right boundary map is not injective. Rank {rank_right} < D ({D})."
        )

    logger.debug("Assumption 1 (Injectivity) verified on both boundaries.")
    return s_left, s_right


def check_chain_unitarity(residual: complex, tol: float = 1e-6) -> None:
    r"""Check the normalization :math:`\operatorname{Tr}(U^\dagger U)/d^N = 1` of a chain.

    ``residual`` is the third output of
    :func:`~TreeMPU.utils.compute_chain_boundary_operators`. This is a
    *necessary* condition for unitarity (e.g. :math:`\mathrm{diag}(\sqrt2,0)`
    passes it); it guards against gross errors in a site-dependent chain,
    whose unitarity must be established otherwise (by construction, or with
    :func:`dense_unitary_from_tensors` at small :math:`N`).

    Raises
    ------
    ValueError
        If :math:`|1-\text{residual}|` exceeds ``tol``.
    """
    error = abs(1.0 - complex(residual))
    logger.debug(f"Chain unitarity residual: {residual} | Error: {error:.2e}")

    if error > tol or np.isnan(error):
        raise ValueError(
            f"Unitarity strictly lost. Residual {residual} deviates from 1 "
            f"by {error:.2e}, exceeding tolerance {tol}."
        )


def check_full_rank(M: np.ndarray, label: str, tol: float = 1e-10) -> np.ndarray:
    """
    Verifies that a boundary operator (L^2 or R^2, at a given bond) is full
    rank, i.e. that the bond dimension at that cut cannot be further
    compressed. Raises ValueError if the matrix is rank-deficient.
    Returns the singular values for diagnostics.
    """
    D = M.shape[0]
    s = np.linalg.svd(M, compute_uv=False)
    rank = int(np.sum(s > tol))

    if rank < D:
        raise ValueError(
            f"Minimal bond dimension assumption failed at {label}: rank {rank} < D ({D})."
        )

    return s


def verify_lcu(
    M_original: np.ndarray, coefficients: list, unitaries: list, tol: float = 1e-6
) -> None:
    """
    Verifies the validity of an LCU decomposition.
    Raises ValueError if mathematical or dimensional constraints are violated.
    """
    M_original = np.asarray(M_original, dtype=complex)

    if M_original.ndim != 2 or M_original.shape[0] != M_original.shape[1]:
        raise ValueError(f"M_original must be square. Got shape: {M_original.shape}")

    dim = M_original.shape[0]
    M_reconstructed = np.zeros_like(M_original, dtype=complex)
    I_exact = np.eye(dim, dtype=complex)

    for idx, (c, W) in enumerate(zip(coefficients, unitaries, strict=True)):
        # Tolerance check for imaginary part and positivity
        if abs(np.imag(c)) > tol or np.real(c) < -tol:
            raise ValueError(f"Coefficient [{idx}] is not real positive (within tol): {c}")

        W = np.asarray(W, dtype=complex)
        if W.shape != (dim, dim):
            raise ValueError(
                f"Unitary [{idx}] has incorrect shape. Expected {(dim, dim)}, got {W.shape}."
            )

        # Unitarity check
        if not np.allclose(W @ W.conj().T, I_exact, atol=tol):
            raise ValueError(f"Matrix [{idx}] violates W @ W† = I.")

        M_reconstructed += np.real(c) * W

    # Reconstruction check
    if not np.allclose(M_reconstructed, M_original, atol=tol):
        error_norm = np.linalg.norm(M_reconstructed - M_original, ord="fro")
        raise ValueError(f"LCU does not reconstruct M. Frobenius error: {error_norm:.4e}")

    logger.debug(
        "LCU verification passed: real/positive coefficients, valid unitaries, exact reconstruction."
    )


def verify_merging_unitary(
    B: np.ndarray,
    W_ctrl: np.ndarray,
    M_operator: np.ndarray,
    C: float,
    dim_system: int,
    dim_ancilla: int,
    tol: float = 1e-10,
) -> None:
    """
    Verifies that the block encoding correctly embeds M / C in the |0>_A subspace.
    Convention: Ancilla ⊗ System  (W_ctrl = kron(proj_i, W_i))
    """
    # B acts on ancilla, system is untouched → B ⊗ I_system
    B_full = np.kron(B, np.eye(dim_system, dtype=complex))
    B_full_dag = np.kron(B.conj().T, np.eye(dim_system, dtype=complex))

    U_total = B_full_dag @ W_ctrl @ B_full

    # Post-select ancilla on |0> : P_0 = |0><0|_A ⊗ I_system
    proj_0_ancilla = np.zeros((dim_ancilla, dim_ancilla), dtype=complex)
    proj_0_ancilla[0, 0] = 1.0
    P_0 = np.kron(proj_0_ancilla, np.eye(dim_system, dtype=complex))

    U_post_selected = P_0 @ U_total @ P_0
    M_target_embedded = np.kron(proj_0_ancilla, M_operator / C)

    if not np.allclose(U_post_selected, M_target_embedded, atol=tol):
        error = np.linalg.norm(U_post_selected - M_target_embedded, ord="fro")
        raise ValueError(f"Merging unitary post-selection failed. Frobenius error: {error:.4e}")

    logger.debug("Merging unitary verification passed: post-selection yields M / C.")


# ---------------------------------------------------------------------------
# Dense reference checks (small systems only)
# ---------------------------------------------------------------------------

#: Largest number of qubits for which :func:`reconstruct_circuit_unitary`
#: will run a state-vector simulation (2**24 complex amplitudes = 256 MiB).
MAX_RECONSTRUCTION_QUBITS = 24


def dense_unitary_from_tensors(tensors, l_vec: np.ndarray, r_vec: np.ndarray) -> np.ndarray:
    r"""Contract an open-boundary MPU into a dense :math:`d^N \times d^N` matrix.

    Parameters
    ----------
    tensors : list of ndarray
        Site tensors ``A[p_in, p_out, bond_L, bond_R]``, one per site.
    l_vec, r_vec : ndarray
        Boundary vectors contracted into the first left / last right bond.

    Returns
    -------
    ndarray
        ``U[out, in]`` with site 0 as the *least* significant digit, i.e. the
        same little-endian ordering Qiskit uses for the system registers
        returned by :meth:`CircuitDecomposition.synthesize`.

    Notes
    -----
    Cost and memory grow as :math:`d^{2N}`; intended for validation at small N.
    """
    tensors = list(tensors)
    n = len(tensors)
    d = tensors[0].shape[0]
    dim = d**n
    if dim > 2**12:
        raise ValueError(f"d**N = {dim} is too large for a dense reference (limit 4096).")
    tn = qtn.Tensor(np.asarray(l_vec), inds=["bond_0"])
    for k, A_k in enumerate(tensors):
        tn = tn @ qtn.Tensor(A_k, inds=[f"p_in_{k}", f"p_out_{k}", f"bond_{k}", f"bond_{k + 1}"])
    tn = tn @ qtn.Tensor(np.asarray(r_vec), inds=[f"bond_{n}"])
    out_inds = [f"p_out_{k}" for k in reversed(range(n))]
    in_inds = [f"p_in_{k}" for k in reversed(range(n))]
    return tn.to_dense(out_inds, in_inds)


def reconstruct_circuit_unitary(
    circuit, system_registers, d: int | None = None, max_qubits: int = MAX_RECONSTRUCTION_QUBITS
):
    r"""Recover the operator a synthesized circuit applies to the system.

    Every physical computational-basis input of the system registers is
    prepared with all ancilla and bond qubits in :math:`|0\rangle`, the
    circuit is simulated exactly, and the output amplitudes on physical
    system states with all other qubits in :math:`|0\rangle` are collected
    column by column. For an exact synthesis the result equals :math:`U_N`
    up to a global phase and the leakage is zero.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Output of :meth:`CircuitDecomposition.synthesize`.
    system_registers : list of qiskit.QuantumRegister
        Per-site system registers, site 0 first (second return value of
        ``synthesize``), each with :math:`\lceil\log_2 d\rceil` qubits.
    d : int, optional
        Local physical dimension. Only register values ``0..d-1`` are
        physical; the others are padding. Defaults to ``2**n`` for ``n``
        qubits per register.
    max_qubits : int
        Refuse to simulate wider circuits (the cost is :math:`2^{n}` memory
        per column). Checked before any allocation.

    Returns
    -------
    U_rec : ndarray
        Post-selected operator on the physical space, ``U_rec[out, in]``,
        of size :math:`d^N`, site 0 least significant (as in
        :func:`dense_unitary_from_tensors`).
    leakage : float
        Largest probability, over physical basis inputs, of ending outside
        the physical, all-ancillas-zero subspace (0 for an exact synthesis).
    """
    from qiskit.quantum_info import Statevector

    if circuit.num_qubits > max_qubits:
        raise ValueError(
            f"circuit has {circuit.num_qubits} qubits > max_qubits={max_qubits}; "
            "state-vector reconstruction would need too much memory."
        )
    regs = [[circuit.find_bit(q).index for q in reg] for reg in system_registers]
    n_per = len(regs[0])
    d = 2**n_per if d is None else int(d)
    if d > 2**n_per:
        raise ValueError(f"d={d} does not fit in registers of {n_per} qubits.")
    n_sites = len(regs)
    dim = d**n_sites

    def full_index(s: int) -> int:
        """Qubit bit pattern of physical basis state s = sum_k p_k d**k."""
        idx = 0
        for k in range(n_sites):
            p = (s // d**k) % d
            for i, q in enumerate(regs[k]):
                if (p >> i) & 1:
                    idx |= 1 << q
        return idx

    targets = [full_index(s) for s in range(dim)]
    cols = []
    leakage = 0.0
    for s_in in range(dim):
        prep = circuit.copy_empty_like()
        for q in range(circuit.num_qubits):
            if (targets[s_in] >> q) & 1:
                prep.x(q)
        prep.compose(circuit, inplace=True)
        amps = Statevector.from_instruction(prep).data
        col = amps[targets]
        leakage = max(leakage, 1.0 - float(np.sum(np.abs(col) ** 2)))
        cols.append(col)
    return np.column_stack(cols), leakage


def mps_circuit_columns(
    circuit,
    system_registers,
    inputs,
    d: int | None = None,
    max_bond: int | None = None,
    cutoff: float = 1e-12,
) -> dict[int, np.ndarray]:
    r"""Columns of a synthesized circuit's operator, simulated as a matrix product state.

    The tensor-network counterpart of :func:`reconstruct_circuit_unitary` for
    circuits too wide for a state vector. For each physical basis input
    ``x`` (ancilla and bond qubits in :math:`|0\rangle`) the circuit, lowered
    to ``u``/``cx`` gates, is applied to the product state with
    ``quimb.tensor.CircuitMPS``; the non-system qubits are then projected on
    :math:`|0\rangle` and the :math:`d^N` system amplitudes read out. For an
    exact synthesis this is column ``x`` of :math:`U_N` (up to one global
    phase) with unit norm.

    The simulation is exact up to the SVD truncation ``cutoff``; with
    ``max_bond`` it becomes a controlled approximation (the norm deficit and
    the column errors then measure its accuracy). Cost grows with the
    circuit's entanglement and gate count, not with ``2**n_qubits``: e.g. a
    36-qubit, N = 8 circuit takes about two minutes per column.

    Parameters
    ----------
    circuit, system_registers, d
        As in :func:`reconstruct_circuit_unitary`.
    inputs : iterable of int
        Physical basis inputs ``x = sum_k p_k d**k`` (site 0 least significant).
    max_bond : int, optional
        Bond-dimension cap of the MPS (default: none).
    cutoff : float
        Relative singular-value cutoff of every gate application.

    Returns
    -------
    dict
        ``{x: amplitudes}``, each of length :math:`d^N` (site 0 least significant).
        Compare with :func:`phase_aligned_error` on the stacked columns.
    """
    from qiskit import transpile

    low = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    n = low.num_qubits
    regs = [[low.find_bit(q).index for q in reg] for reg in system_registers]
    n_per = len(regs[0])
    d = 2**n_per if d is None else int(d)
    n_sites = len(regs)
    sys_q = [q for reg in regs for q in reg]
    anc = sorted(set(range(n)) - set(sys_q))
    gates = []
    for ci in low.data:
        if ci.operation.name == "barrier":
            continue
        qs = [low.find_bit(q).index for q in ci.qubits]
        gates.append(
            (None if ci.operation.name == "cx" else np.asarray(ci.operation.to_matrix()), qs)
        )

    def digits(x):
        bits = [0] * n
        for k in range(n_sites):
            p = (x // d**k) % d
            for i, q in enumerate(regs[k]):
                bits[q] = (p >> i) & 1
        return bits

    out_index = [tuple(digits(y)[q] for q in sys_q) for y in range(d**n_sites)]
    cols = {}
    for x in inputs:
        circ = qtn.CircuitMPS(
            n,
            psi0=qtn.MPS_computational_state(digits(int(x))),
            gate_opts=dict(max_bond=max_bond, cutoff=cutoff),
        )
        for g, qs in gates:
            if g is None:
                circ.apply_gate("CX", *qs)
            else:
                circ.apply_gate_raw(g, qs)
        psi = circ.psi
        t = psi.isel({psi.site_ind(q): 0 for q in anc})
        amps = np.asarray(t.contract(output_inds=[psi.site_ind(q) for q in sys_q]).data)
        amps = amps.reshape([2] * len(sys_q))
        cols[int(x)] = np.exp(1j * low.global_phase) * np.array([amps[i] for i in out_index])
    return cols


def phase_aligned_error(U_rec: np.ndarray, U_ref: np.ndarray) -> float:
    r"""Normalized distance :math:`\min_\varphi \|U_{rec}-e^{i\varphi}U_{ref}\|_F/\sqrt{D}`."""
    overlap = np.trace(U_ref.conj().T @ U_rec)
    phase = 1.0 if abs(overlap) < 1e-14 else overlap / abs(overlap)
    return float(np.linalg.norm(U_rec - phase * U_ref) / np.sqrt(U_ref.shape[0]))


def block_isometry_residuals(
    A: np.ndarray,
    l_vec: np.ndarray,
    r_vec: np.ndarray,
    L: np.ndarray,
    R: np.ndarray,
    lengths=(1, 2, 4),
    max_dim: int = 2**10,
) -> dict[int, tuple[float, float, float]]:
    r"""Isometry residuals of the blocks a boundary pair ``(L, R)`` defines.

    For each block length :math:`n` returns
    :math:`\|V^\dagger V - I\|_F` for the left-boundary block
    :math:`\ell A^n R`, the bulk block :math:`L A^n R` and the right-boundary
    block :math:`L A^n r`, each viewed as a map from the :math:`n` input legs
    to (output legs, external bond legs). Exact boundaries give residuals at
    machine precision for every :math:`n`; boundaries from a numerical
    optimizer satisfy the constraint only to the optimizer's precision, and
    the synthesis snaps each site isometry onto the nearest exact one
    (:func:`~TreeMPU.utils.dilate_isometry_to_unitary`), so these residuals
    bound the resulting deviation of the circuit from :math:`U_N`.

    Parameters
    ----------
    A : ndarray
        Bulk tensor ``A[p_in, p_out, bond_L, bond_R]``.
    l_vec, r_vec : ndarray
        Boundary vectors.
    L, R : ndarray
        Boundary operators ``L[ext, bond]``, ``R[ext, bond]`` (e.g.
        ``mpu.L``, ``mpu.R`` of a :class:`~TreeMPU.MPU.UniformMPU`).
    lengths : iterable of int
        Block lengths :math:`n`; lengths with :math:`d^n >` ``max_dim`` are
        skipped (checked before any allocation).
    max_dim : int
        Largest input dimension :math:`d^n` evaluated.

    Returns
    -------
    dict
        ``{n: (left, bulk, right)}``.
    """
    d = A.shape[0]
    out: dict[int, tuple[float, float, float]] = {}
    for n in lengths:
        if d**n > max_dim:
            continue
        T = A
        for _ in range(n - 1):
            T = np.einsum("ijab,klbc->ikjlac", T, A).reshape(
                T.shape[0] * d, T.shape[1] * d, T.shape[2], A.shape[3]
            )
        dim_in = T.shape[0]
        eye = np.eye(dim_in)
        blocks = (
            np.einsum("a,ijab,yb->jyi", l_vec, T, R).reshape(-1, dim_in),
            np.einsum("xa,ijab,yb->xjyi", L, T, R).reshape(-1, dim_in),
            np.einsum("xa,ijab,b->xji", L, T, r_vec).reshape(-1, dim_in),
        )
        out[n] = tuple(float(np.linalg.norm(V.conj().T @ V - eye)) for V in blocks)
    return out
