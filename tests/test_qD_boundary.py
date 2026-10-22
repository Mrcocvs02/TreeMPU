"""Tests of ``utils.solve_qD_boundary`` and its ``UniformMPU`` integration.

The solver is the general q = D boundary construction.

Each result is checked against independent references: the function's residual
diagnostics, dense length-1 boundary isometries ``V_1^[l,R]`` and ``V_1^[L,r]``
rebuilt here without the solver's internal tensors, and the bond-space
isometry defect at prefix lengths beyond those used by the fit. The solver is
verified on a Mode I tensor and on a Mode II cyclic tensor that
``mode1_exact_boundary`` cannot handle, the precedence among the boundary
options of ``UniformMPU`` is verified, and robustness under a change of bond
basis is verified through the conditioning number and block isometry
residuals.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from mpu_generators import UniformMPUGenerator
from TreeMPU import UniformMPU
from TreeMPU.utils import (
    _bondspace_isometry_defect,
    _isometry_transfer_tensor,
    solve_qD_boundary,
)


def _dense_V1_lR(A, l_vec, R):
    """Return ``V^dagger V - 1`` of the dense length-1 isometry from ``l_vec``, ``R``."""
    d, D = A.shape[0], A.shape[2]
    A0 = np.tensordot(l_vec, A, axes=(0, 2))  # (d,d,D) axes (i,j,c)
    V = np.einsum("ijc,ec->ije", A0, R)
    V_mat = np.transpose(V, (1, 2, 0)).reshape(d * D, d)
    return V_mat.conj().T @ V_mat - np.eye(d)


def _dense_V1_Lr(A, r_vec, L):
    """Return ``W^dagger W - 1`` of the dense length-1 isometry from ``L``, ``r_vec``."""
    d, D = A.shape[0], A.shape[2]
    A0r = np.tensordot(A, r_vec, axes=(3, 0))  # (d,d,D) axes (i,j,a)
    W = np.einsum("ija,ea->ije", A0r, L)
    W_mat = np.transpose(W, (1, 2, 0)).reshape(d * D, d)
    return W_mat.conj().T @ W_mat - np.eye(d)


def _bulk_defect_out_of_sample(A, L2, R2, d, k=6):
    """Return the bond-space isometry defect over all prefixes up to length ``k``."""
    G = _isometry_transfer_tensor(A)
    G_list = [jnp.array(G)] * k
    d_list = [d] * k
    return float(
        _bondspace_isometry_defect(G_list, d_list, jnp.array(L2), jnp.array(R2), all_prefixes=True)
    )


@pytest.fixture(scope="module")
def mode1_D3_tensor():
    """Return a Mode I D = 3 tensor with single-vertex readout (closed form applies)."""
    gen = UniformMPUGenerator(seed=0, verify=True)
    tensors, l_vec, r_vec = gen.branching(
        d=2, bond_dim=3, n_blocks=2, blocking="auto", max_blocking=4
    )
    return tensors[0], l_vec, r_vec


@pytest.fixture(scope="module")
def mode2_cyclic_D3_tensor():
    """Return a Mode II cyclic d = D = 3 tensor with a non-basis left boundary."""
    gen = UniformMPUGenerator(seed=0, verify=True)
    tensors, l_vec, r_vec = gen.fusion(
        d=3, bond_dim=3, n_blocks=2, skeleton="cyclic", blocking="auto", max_blocking=4
    )
    return tensors[0], l_vec, r_vec


def test_solve_qD_boundary_reaches_q_equals_D_and_is_a_true_isometry(mode1_D3_tensor):
    """The solver reaches q = D with exact boundary and bulk isometries (Mode I)."""
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
    """The solver reaches q = D where the Mode I closed form does not apply."""
    A, l_vec, r_vec = mode2_cyclic_D3_tensor
    d, D = A.shape[0], A.shape[2]
    # A single-vertex l_vec has exactly one non-zero entry, so this tensor is not
    # eligible for mode1_exact_boundary.
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
    """UniformMPU adopts the q = D solver before the optimizer when it succeeds."""
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
    assert mpu.boundary_opt_result is None  # the optimizer was not run
    assert mpu.qD_diagnostics["cond1_residual"] < 1e-6
    assert mpu.qD_diagnostics["bulk_defect_out_of_sample"] < 1e-6

    # L, R actually adopted must themselves satisfy the same independent checks.
    L2 = mpu.L.conj().T @ mpu.L
    R2 = mpu.R.conj().T @ mpu.R
    assert np.linalg.norm(_dense_V1_lR(A, l_vec, mpu.R)) < 1e-6
    assert np.linalg.norm(_dense_V1_Lr(A, r_vec, mpu.L)) < 1e-6
    assert _bulk_defect_out_of_sample(A, L2, R2, A.shape[0], k=6) < 1e-6


def test_uniformmpu_boundary_opt_gauge_skips_qD_general(mode2_cyclic_D3_tensor):
    """boundary_opt_gauge=True takes precedence over the (HPD) q = D solver."""
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
    """The q = D solver is skipped when the Mode I exact boundary already has q = D."""
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


@pytest.mark.parametrize("kind", ["real", "complex"])
def test_solve_qD_boundary_is_robust_to_a_change_of_bond_basis(kind):
    """The solver reaches q = D with an isometric pair after a change of bond basis.

    The basis change is real or complex.
    """
    from mpu_generators import build_family
    from TreeMPU import block_isometry_residuals, conditioning_number

    t, l, r = build_family("branching_uniform", d=2, bond_dim=2, seed=0)
    D = 2
    rng = np.random.default_rng(10)
    G = np.eye(D) + 0.5 * (
        rng.normal(size=(D, D)) + (1j * rng.normal(size=(D, D)) if kind == "complex" else 0)
    )
    G_inv = np.linalg.inv(G)
    A = np.einsum("ac,ijce,eb->ijab", G_inv, t[0], G)
    lv, rv = G.T @ l, G_inv @ r
    L, R, q, _ = solve_qD_boundary(A, lv, rv)
    assert conditioning_number(L, R) == pytest.approx(D, abs=1e-8)
    res = block_isometry_residuals(A, lv, rv, L, R, lengths=(1, 2, 4))
    assert max(max(v) for v in res.values()) < 1e-6
