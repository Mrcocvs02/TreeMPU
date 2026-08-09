import numpy as np  # pyright: ignore[reportMissingImports]
from typing import Tuple
from quimb import tensor as qtn  # type: ignore
import logging
from .utils import get_mpo_site_tensors

logger = logging.getLogger(__name__)


def check_mpo_unitarity(
    N_max: int, A_np: np.ndarray, l_in: np.ndarray, r_in: np.ndarray, tol: float = 1e-6
) -> None:
    """
    Iteratively verifies the unitarity of an MPO chain by checking the ratio
    between Tr(rho)^2 and Tr(rho^2). For a unitary evolution, this ratio must be 1.0.
    Raises ValueError if unitarity is violated beyond the tolerance.
    """
    l_vec = qtn.Tensor(l_in.copy(), inds=["bond_0"], tags=["l"])
    r_vec = r_in.copy()

    l_vec_conj = l_vec.H.reindex({"bond_0": "bond_0_dag"}).retag({"l": "l*"})

    single_store = l_vec & l_vec_conj
    double_store = (
        l_vec
        & l_vec_conj
        & l_vec.reindex({"bond_0": f"bond_{N_max+1}"})
        & l_vec_conj.reindex({"bond_0_dag": f"bond_{N_max+1}_dag"})
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
            & A_k_double.reindex({f"p_in_{N+N_max+1}": f"p_middle_{N}"})
            & A_k_dag_double.reindex({f"p_out_f_{N+N_max+1}": f"p_in_{N}"})
        )

        UU_dag = single_store.copy() & r_loc & r_loc_dag
        numerator = UU_dag.contract() ** 2

        UU_double = (
            double_store
            & r_loc
            & r_loc_dag
            & r_vec_tensor.reindex({"bond_0": f"bond_{N+N_max+1}"})
            & r_vec_conj.reindex({"bond_0_dag": f"bond_{N+N_max+1}_dag"})
        )

        denominator = (A_np.shape[0] ** N) * UU_double.contract()

        try:
            check_val = (numerator / denominator).real
        except (ZeroDivisionError, RuntimeWarning):
            check_val = np.nan

        error_k = abs(1.0 - float(check_val))
        logger.debug(
            f"Site {N:03d} | Unitarity Check: {check_val:.10f} | Error: {error_k:.2e}"
        )

        if error_k > tol or np.isnan(error_k):
            raise ValueError(
                f"Unitarity strictly lost at N={N}. Error {error_k:.2e} exceeds tolerance {tol}."
            )

    logger.debug(f"MPO Unitarity verified up to N_max={N_max}.")


def check_assumption_1(
    A: np.ndarray, l_vec: np.ndarray, r_vec: np.ndarray, tol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verifies Assumption 1 for a uniform MPU.
    Checks that the map from the bond space to the physical space is injective.
    Raises ValueError if the rank is strictly less than the bond dimension D.
    Returns singular values for diagnostics.
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
    """
    Verifies the global unitarity residual produced by
    ``utils.compute_chain_boundary_operators`` for a nonuniform MPU chain.

    The residual is the fully-closed contraction of the chain (boundary
    vectors included) and must equal exactly 1 for a genuinely unitary MPU.
    Raises ValueError if unitarity is violated beyond the tolerance.
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
            f"Minimal bond dimension assumption failed at {label}: "
            f"rank {rank} < D ({D})."
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

    for idx, (c, W) in enumerate(zip(coefficients, unitaries)):
        # Tolerance check for imaginary part and positivity
        if abs(np.imag(c)) > tol or np.real(c) < -tol:
            raise ValueError(
                f"Coefficient [{idx}] is not real positive (within tol): {c}"
            )

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
        raise ValueError(
            f"LCU does not reconstruct M. Frobenius error: {error_norm:.4e}"
        )

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
        raise ValueError(
            f"Merging unitary post-selection failed. " f"Frobenius error: {error:.4e}"
        )

    logger.debug("Merging unitary verification passed: post-selection yields M / C.")
