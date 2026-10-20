"""TreeMPU: exact quantum-circuit synthesis of Matrix Product Unitaries.

Main entry points
-----------------
:class:`UniformMPU`, :class:`GeneralMPU`
    Build an MPU, choose boundary operators, and synthesize its circuit.
:mod:`TreeMPU.utils`
    Boundary constructions, conditioning number, tensor utilities.
:mod:`TreeMPU.checks`
    Unitarity checks and dense reference tools.
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

__version__ = "0.2.0"

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
