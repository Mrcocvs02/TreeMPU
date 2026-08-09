"""
MPU_Decomposition: A theoretical and numerical framework for
Matrix-Product Unitaries and their quantum circuit synthesis.
"""

from . import utils
from . import checks
from . import MPU

from .MPU import UniformMPU, GeneralMPU, CircuitDecomposition
from .checks import (
    check_mpo_unitarity,
    check_assumption_1,
    check_chain_unitarity,
    check_full_rank,
    verify_lcu,
    verify_merging_unitary,
)
from .utils import (
    get_mpo_site_tensors,
    optimize_q_unif,
    matrix_sqrt_hermitian,
    get_merging_operator,
    compute_chain_boundary_operators,
    optimize_boundary_operators,
)

__all__ = [
    "UniformMPU",
    "GeneralMPU",
    "CircuitDecomposition",
    "check_mpo_unitarity",
    "get_mpo_site_tensors",
    "optimize_q_unif",
    "check_assumption_1",
    "check_chain_unitarity",
    "check_full_rank",
    "utils",
    "checks",
    "matrix_sqrt_hermitian",
    "verify_lcu",
    "get_merging_operator",
    "verify_merging_unitary",
    "compute_chain_boundary_operators",
    "optimize_boundary_operators",
    "MPU",
]
