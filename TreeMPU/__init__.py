r"""TreeMPU: exact quantum-circuit synthesis of Matrix Product Unitaries.

TreeMPU compiles an open-boundary Matrix Product Unitary (MPU) into an exact
circuit with the tree decomposition of Styliaris, Trivedi and Cirac
[PRL 135, 260602 (2025)] and the extensions of the accompanying thesis.

Quickstart
----------
>>> from mpu_generators import build_family  # doctest: +SKIP
>>> from TreeMPU import UniformMPU  # doctest: +SKIP
>>> tensors, l_vec, r_vec = build_family(
...     "branching_uniform", d=2, bond_dim=2, seed=0
... )  # doctest: +SKIP
>>> mpu = UniformMPU(tensors[0], l_vec, r_vec, N=4, light=True, boundary_opt_k=1)  # doctest: +SKIP
>>> qc, sys_regs, bond_regs, anc_regs, ancillas = mpu.synthesize()  # doctest: +SKIP

Conventions
-----------
Site tensors are ``A[p_in, p_out, bond_L, bond_R]``; boundary operators are
``L[ext, bond]`` and ``R[ext, bond]`` (stored transposed with respect to the
thesis product ``L A R``) with merging kernel :math:`R^{-\top}L^{-1}` and
conditioning number :math:`q=\|R^{-\top}L^{-1}\|_F`.
Site 0 is the least significant digit in dense matrices and in the Qiskit
register order. Importing the package enables ``jax_enable_x64``.

Public API
----------
Classes
    :class:`UniformMPU`, :class:`GeneralMPU`, :class:`CircuitDecomposition`.
Checks (:mod:`TreeMPU.checks`)
    :func:`check_mpo_unitarity`, :func:`check_assumption_1`,
    :func:`check_chain_unitarity`, :func:`check_full_rank`,
    :func:`verify_lcu`, :func:`verify_merging_unitary`,
    :func:`dense_unitary_from_tensors`, :func:`reconstruct_circuit_unitary`,
    :func:`mps_circuit_columns`, :func:`phase_aligned_error`,
    :func:`block_isometry_residuals`.
Boundary construction and tensor utilities (:mod:`TreeMPU.utils`)
    :func:`get_mpo_site_tensors`, :func:`optimize_q_unif`,
    :func:`matrix_sqrt_hermitian`, :func:`get_merging_operator`,
    :func:`conditioning_number`, :func:`compute_chain_boundary_operators`,
    :func:`optimize_boundary_operators`, :func:`optimize_chain_sigma_tau`,
    :func:`solve_qD_boundary`, :func:`mode1_exact_boundary`,
    :func:`to_canonical_form`, :func:`dilate_isometry_to_unitary`,
    :func:`block_mpu_tensors`, :func:`satisfies_assumption_1`.
Modules
    ``MPU``, ``utils``, ``checks``; ``__version__``.
"""

from . import MPU, checks, utils
from .checks import (
    block_isometry_residuals,
    check_assumption_1,
    check_chain_unitarity,
    check_full_rank,
    check_mpo_unitarity,
    dense_unitary_from_tensors,
    mps_circuit_columns,
    phase_aligned_error,
    reconstruct_circuit_unitary,
    verify_lcu,
    verify_merging_unitary,
)
from .MPU import CircuitDecomposition, GeneralMPU, UniformMPU
from .utils import (
    block_mpu_tensors,
    compute_chain_boundary_operators,
    conditioning_number,
    dilate_isometry_to_unitary,
    get_merging_operator,
    get_mpo_site_tensors,
    matrix_sqrt_hermitian,
    mode1_exact_boundary,
    optimize_boundary_operators,
    optimize_chain_sigma_tau,
    optimize_q_unif,
    satisfies_assumption_1,
    solve_qD_boundary,
    to_canonical_form,
)

__version__ = "0.3.0"

__all__ = [
    "UniformMPU",
    "GeneralMPU",
    "CircuitDecomposition",
    "check_mpo_unitarity",
    "check_assumption_1",
    "check_chain_unitarity",
    "check_full_rank",
    "verify_lcu",
    "verify_merging_unitary",
    "dense_unitary_from_tensors",
    "reconstruct_circuit_unitary",
    "mps_circuit_columns",
    "phase_aligned_error",
    "block_isometry_residuals",
    "get_mpo_site_tensors",
    "optimize_q_unif",
    "matrix_sqrt_hermitian",
    "get_merging_operator",
    "conditioning_number",
    "compute_chain_boundary_operators",
    "optimize_boundary_operators",
    "optimize_chain_sigma_tau",
    "solve_qD_boundary",
    "mode1_exact_boundary",
    "to_canonical_form",
    "dilate_isometry_to_unitary",
    "block_mpu_tensors",
    "satisfies_assumption_1",
    "utils",
    "checks",
    "MPU",
    "__version__",
]
