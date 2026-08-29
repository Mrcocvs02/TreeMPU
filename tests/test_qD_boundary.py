"""
Tests for utils.solve_qD_boundary (the general q=D floor construction --
see its docstring for the 5-step derivation, method specified by the user,
implemented and independently validated here) and its integration into
UniformMPU(boundary_opt_k=...).

Every claim is checked at least two independent ways, matching this
codebase's established discipline: (1) the function's own residual
diagnostics, (2) a from-scratch dense reconstruction of the length-1
boundary isometries V_1^[l,R]/V_1^[L,r], built independently here (not
reusing solve_qD_boundary's internal Gamma-tensor machinery), and (3) the
bond-space isometry defect at several *out-of-sample* prefix lengths (n
beyond what the fit's own k_bulk was scored on), reusing the codebase's
own already-validated _isometry_transfer_tensor/_bondspace_isometry_defect
(the same machinery optimize_boundary_operators relies on).
"""

import numpy as np
import jax.numpy as jnp
import pytest

from TreeMPU import UniformMPU
from TreeMPU.utils import (
    solve_qD_boundary,
    _isometry_transfer_tensor,
    _bondspace_isometry_defect,
)
from mpu_generators import UniformMPUGenerator


def _dense_V1_lR(A, l_vec, R):
    """V = A @ R^T -- matches UniformMPU.create_local_isometries exactly
    (R_tn = Tensor(R, inds=["ext_R","bond_1"]): R's SECOND axis contracts,
    not the first -- see solve_qD_boundary's docstring note on this)."""
    d, D = A.shape[0], A.shape[2]
    A0 = np.tensordot(l_vec, A, axes=(0, 2))  # (d,d,D) axes (i,j,c)
    V = np.einsum("ijc,ec->ije", A0, R)
    V_mat = np.transpose(V, (1, 2, 0)).reshape(d * D, d)
    return V_mat.conj().T @ V_mat - np.eye(d)


def _dense_V1_Lr(A, r_vec, L):
    """W = A @ L^T -- same transposed convention as above, for L."""
    d, D = A.shape[0], A.shape[2]
    A0r = np.tensordot(A, r_vec, axes=(3, 0))  # (d,d,D) axes (i,j,a)
    W = np.einsum("ija,ea->ije", A0r, L)
    W_mat = np.transpose(W, (1, 2, 0)).reshape(d * D, d)
    return W_mat.conj().T @ W_mat - np.eye(d)


def _bulk_defect_out_of_sample(A, L2, R2, d, k=6):
    G = _isometry_transfer_tensor(A)
    G_list = [jnp.array(G)] * k
    d_list = [d] * k
    return float(
        _bondspace_isometry_defect(
            G_list, d_list, jnp.array(L2), jnp.array(R2), all_prefixes=True
        )
    )


@pytest.fixture(scope="module")
def mode1_D3_tensor():
    """
    Mode I, single-vertex readout, D=3 -- the case that exposed the missing
    bulk-isometry constraint (Step 3 alone: best defect found was 0.066
    after 500 restarts; with the Step 4b bulk residual, machine precision
    on the first restart). Also mode1_exact_boundary-eligible, so this
    exercises solve_qD_boundary as a standalone function, not the
    UniformMPU integration (that needs a tensor mode1_exact_boundary
    can't already claim -- see mode2_cyclic_D3_tensor below).

    blocking="auto" (capped at max_blocking=4, NOT the default 8 -- see
    CLAUDE.md: block_mpu_tensors allocates a (d**L, d**L, D, D) array per
    attempt, exponential in L, and the default max_blocking=8 can reach
    several GB for a tensor that never satisfies Assumption 1) is required:
    the unblocked tensor is rank-deficient at the boundary and UniformMPU's
    constructor rejects it before solve_qD_boundary ever runs.
    """
    gen = UniformMPUGenerator(seed=0, verify=True)
    tensors, l_vec, r_vec = gen.branching(
        d=2, bond_dim=3, n_blocks=2, blocking="auto", max_blocking=4
    )
    return tensors[0], l_vec, r_vec


@pytest.fixture(scope="module")
def mode2_cyclic_D3_tensor():
    """
    Mode II / cyclic, d=D=3 -- NOT mode1_exact_boundary-eligible (l_vec has
    3 nonzero entries, not a single standard basis vector), so this is what
    exercises solve_qD_boundary's actual generality and the new UniformMPU
    integration path (boundary_source == "qD_general") end to end.

    nested_flag was tried first and rejected: every seed checked (0-9, both
    D=2 and D=3) fails Assumption 1 even at blocking up to L=4 -- a
    structural property of that skeleton's boundary, not bad luck. cyclic
    satisfies Assumption 1 at L=1 (no blocking needed) for every seed
    checked.
    """
    gen = UniformMPUGenerator(seed=0, verify=True)
    tensors, l_vec, r_vec = gen.fusion(
        d=3, bond_dim=3, n_blocks=2, skeleton="cyclic", blocking="auto", max_blocking=4
    )
    return tensors[0], l_vec, r_vec


def test_solve_qD_boundary_reaches_q_equals_D_and_is_a_true_isometry(mode1_D3_tensor):
    A, l_vec, r_vec = mode1_D3_tensor
    d, D = A.shape[0], A.shape[2]
    L, R, q, diag = solve_qD_boundary(A, l_vec, r_vec, n_restarts=25, seed=0, k_bulk=3)

    assert q == float(D)
    assert diag["cond1_residual"] < 1e-6
    assert diag["cond2_residual"] < 1e-6
    assert diag["bulk_defect_out_of_sample"] < 1e-6
    assert diag["L2_min_eigenvalue"] > 0

    # Independent re-derivations, not the function's own self-reported values.
    L2 = L.conj().T @ L
    R2 = R.conj().T @ R
    assert np.linalg.norm(_dense_V1_lR(A, l_vec, R)) < 1e-6
    assert np.linalg.norm(_dense_V1_Lr(A, r_vec, L)) < 1e-6
    assert _bulk_defect_out_of_sample(A, L2, R2, d, k=6) < 1e-6


def test_solve_qD_boundary_generalizes_beyond_mode1_exact(mode2_cyclic_D3_tensor):
    """
    The whole point of this construction: it applies to boundaries
    mode1_exact_boundary structurally cannot (l_vec here is not a single
    e_v), and still reaches the same provable q=D floor.
    """
    A, l_vec, r_vec = mode2_cyclic_D3_tensor
    d, D = A.shape[0], A.shape[2]
    # Confirm this tensor genuinely isn't mode1_exact_boundary-eligible:
    # a single-vertex l_vec has exactly one nonzero entry, this one doesn't.
    assert np.sum(np.abs(l_vec) > 1e-9) != 1

    L, R, q, diag = solve_qD_boundary(A, l_vec, r_vec, n_restarts=25, seed=0, k_bulk=3)

    assert q == float(D)
    assert diag["cond1_residual"] < 1e-6
    assert diag["cond2_residual"] < 1e-6
    assert diag["bulk_defect_out_of_sample"] < 1e-6

    L2 = L.conj().T @ L
    R2 = R.conj().T @ R
    assert np.linalg.norm(_dense_V1_lR(A, l_vec, R)) < 1e-6
    assert np.linalg.norm(_dense_V1_Lr(A, r_vec, L)) < 1e-6
    assert _bulk_defect_out_of_sample(A, L2, R2, d, k=6) < 1e-6


def test_uniformmpu_boundary_opt_k_prefers_qD_general_when_available(
    mode2_cyclic_D3_tensor,
):
    """
    End-to-end integration: UniformMPU(boundary_opt_k=...) must try
    solve_qD_boundary first and adopt it (boundary_source == "qD_general"),
    skipping optimize_boundary_operators entirely, for a tensor
    mode1_exact_boundary cannot handle.
    """
    A, l_vec, r_vec = mode2_cyclic_D3_tensor
    mpu = UniformMPU(
        A=A,
        l_vec=l_vec,
        r_vec=r_vec,
        N=2,
        boundary_opt_k=3,
        boundary_opt_n_restarts=25,
        boundary_opt_seed=0,
    )
    assert mpu.boundary_source == "qD_general"
    assert mpu.qD_general is True
    assert mpu.boundary_opt_result is None  # optimizer never ran
    assert mpu.qD_diagnostics["cond1_residual"] < 1e-6
    assert mpu.qD_diagnostics["bulk_defect_out_of_sample"] < 1e-6

    # L, R actually adopted must themselves satisfy the same independent checks.
    L2 = mpu.L.conj().T @ mpu.L
    R2 = mpu.R.conj().T @ mpu.R
    assert np.linalg.norm(_dense_V1_lR(A, l_vec, mpu.R)) < 1e-6
    assert np.linalg.norm(_dense_V1_Lr(A, r_vec, mpu.L)) < 1e-6
    assert _bulk_defect_out_of_sample(A, L2, R2, A.shape[0], k=6) < 1e-6


def test_uniformmpu_boundary_opt_gauge_skips_qD_general(mode2_cyclic_D3_tensor):
    """
    solve_qD_boundary only ever returns HPD L, R; boundary_opt_gauge=True is
    an explicit request for the strictly larger non-HPD feasible set gauge
    exploration covers, so qD_general must NOT preempt it even though (per
    the test above) it would otherwise succeed for this exact tensor.
    """
    A, l_vec, r_vec = mode2_cyclic_D3_tensor
    mpu = UniformMPU(
        A=A,
        l_vec=l_vec,
        r_vec=r_vec,
        N=2,
        boundary_opt_k=3,
        boundary_opt_gauge=True,
        boundary_opt_n_restarts=5,
        boundary_opt_seed=0,
    )
    assert mpu.qD_general is False
    assert mpu.boundary_source != "qD_general"


def test_uniformmpu_boundary_mode1_exact_skips_qD_general(mode1_D3_tensor):
    """
    When boundary_mode1_exact already reached q=D for free, trying
    solve_qD_boundary afterward would only cost JAX search time for no
    possible gain (q=D is already the proven floor) -- must be skipped.
    """
    A, l_vec, r_vec = mode1_D3_tensor
    mpu = UniformMPU(
        A=A,
        l_vec=l_vec,
        r_vec=r_vec,
        N=2,
        boundary_mode1_exact=True,
        boundary_opt_k=3,
    )
    assert mpu.mode1_exact is True
    assert mpu.qD_general is False
    assert mpu.boundary_source == "mode1_exact"
