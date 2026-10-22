"""Tests of ``generate_random_mpu`` and the chain utilities in ``TreeMPU.utils``.

Generated chains are verified to be unitary at every length against
``_dense_chain``, a dense contraction written independently of the library, and
via the O(N) boundary residual. Further tests cover blocking (shapes,
unitarity, automatic selection of the smallest blocking that satisfies
Assumption 1), translation invariance of the uniform flag, rejection of
invalid arguments, and the paper's canonical form (eq. 26): full rank,
unitarity, ``L_k`` proportional to the identity and ``R_k`` diagonal and
positive.
"""

import numpy as np
import pytest

from mpu_generators import generate_random_mpu
from TreeMPU.utils import (
    block_mpu_tensors,
    compute_chain_boundary_operators,
    satisfies_assumption_1,
    to_canonical_form,
)


def _dense_chain(tensors, l_vec, r_vec):
    """Contract the chain into a dense matrix indexed ``[in, out]``.

    The tensors follow the ``A[p_in, p_out, bond_L, bond_R]`` convention and
    ``l_vec`` meets ``tensors[0]``. Site 0 is the most significant digit of
    the dense indices, the opposite of the library's dense order; this is
    harmless for the unitarity checks made here.
    """
    M = np.einsum("ijab->aijb", tensors[0])
    for A in tensors[1:]:
        M = np.einsum("aIJc,klcb->aIkJlb", M, A)
        s = M.shape
        M = M.reshape(s[0], s[1] * s[2], s[3] * s[4], s[5])
    return np.einsum("a,aIJb,b->IJ", l_vec, M, r_vec)


@pytest.mark.parametrize("mode", ["branching", "fusion", "grafted"])
def test_generated_mpu_is_exactly_unitary(mode):
    """Every mode yields a chain that is unitary to machine precision."""
    tensors, l_vec, r_vec = generate_random_mpu(n_sites=3, d=2, bond_dim=2, mode=mode, seed=0)
    U = _dense_chain(tensors, l_vec, r_vec)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-10)


@pytest.mark.parametrize("mode", ["branching", "fusion", "grafted"])
def test_generated_mpu_unitary_at_every_length(mode):
    """Truncations of a generated chain are unitary at every length N."""
    # One fixed rule must work for every N, so the truncated chains are checked too.
    tensors, l_vec, r_vec = generate_random_mpu(n_sites=4, d=2, bond_dim=2, mode=mode, seed=1)
    for n in range(1, len(tensors) + 1):
        U = _dense_chain(tensors[:n], l_vec, r_vec)
        np.testing.assert_allclose(
            U.conj().T @ U,
            np.eye(U.shape[0]),
            atol=1e-10,
            err_msg=f"{mode} lost unitarity at N={n}",
        )


def test_blocking_preserves_unitarity_and_shapes():
    """Blocking raises d, keeps the bond dimension and preserves unitarity."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=4, d=2, bond_dim=3, mode="branching", blocking=2, seed=2
    )
    assert len(tensors) == 2
    assert tensors[0].shape == (4, 4, 3, 3)

    U = _dense_chain(tensors, l_vec, r_vec)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-10)


def test_block_mpu_tensors_matches_unblocked_contraction():
    """Blocking is a pure regrouping and leaves the dense chain operator unchanged."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=4, d=2, bond_dim=2, mode="fusion", blocking=1, seed=3
    )
    np.testing.assert_allclose(
        _dense_chain(block_mpu_tensors(tensors, 2), l_vec, r_vec),
        _dense_chain(tensors, l_vec, r_vec),
        atol=1e-12,
    )


def test_uniform_flag_gives_translation_invariant_bulk():
    """The uniform flag repeats one bulk tensor."""
    tensors, _, _ = generate_random_mpu(
        n_sites=4, d=2, bond_dim=2, mode="branching", uniform=True, seed=4
    )
    for A in tensors[1:]:
        np.testing.assert_allclose(A, tensors[0])


def test_chain_boundary_residual_is_one():
    """The O(N) boundary-operator unitarity residual equals 1."""
    tensors, l_vec, r_vec = generate_random_mpu(n_sites=5, d=2, bond_dim=2, mode="grafted", seed=5)
    _, _, residual = compute_chain_boundary_operators(tensors, l_vec, r_vec)
    assert abs(1.0 - complex(residual)) < 1e-10


@pytest.mark.parametrize(
    "mode,d,bond_dim",
    [
        ("branching", 2, 3),  # out-degree capped at d, so needs more blocking still
        ("branching", 3, 5),
    ],
)
def test_auto_blocking_reaches_assumption_1(mode, d, bond_dim):
    """blocking="auto" returns chains satisfying Assumption 1 at the requested D."""
    # The automatic choice verifies Assumption 1 rather than predicting it from a
    # d**(2L) >= D count: the reachable set of the sampled digraph limits the rank.
    tensors, l_vec, r_vec = generate_random_mpu(
        n_blocks=2, d=d, bond_dim=bond_dim, mode=mode, blocking="auto", seed=0
    )
    assert satisfies_assumption_1(tensors, l_vec, r_vec)
    assert len(tensors) == 2
    assert tensors[0].shape[2] == bond_dim


def test_auto_blocking_reports_failure_rather_than_returning_bad_chain():
    """blocking="auto" raises when max_blocking is too small.

    A chain that violates Assumption 1 is never returned instead.
    """
    # Mode I caps the out-degree at d, so the reachable set grows slowly and a large D
    # needs a large blocking factor.
    with pytest.raises(ValueError, match="No blocking"):
        generate_random_mpu(
            n_blocks=2,
            d=2,
            bond_dim=8,
            mode="branching",
            blocking="auto",
            seed=0,
            max_blocking=4,
        )


@pytest.mark.parametrize("bond_dim", [4, 5])
def test_per_site_digraphs_unlock_blocking_for_branching(bond_dim):
    """Resampling the digraph per site makes larger bond dimensions reachable.

    The per-site digraph is the one of thesis ``thm:mode1-unitarity``.
    """
    # A single fixed digraph saturates the reachable set after a few steps, so blocking
    # alone would not give full rank; per-site resampling keeps it growing.
    tensors, l_vec, r_vec = generate_random_mpu(
        n_blocks=2,
        d=2,
        bond_dim=bond_dim,
        mode="branching",
        blocking="auto",
        seed=0,
        max_blocking=8,
    )
    assert satisfies_assumption_1(tensors, l_vec, r_vec)


def test_uniform_branching_reuses_one_digraph():
    """A translation-invariant Mode I chain uses the same digraph at every site."""
    tensors, _, _ = generate_random_mpu(
        n_sites=4, d=2, bond_dim=3, mode="branching", uniform=True, seed=0
    )
    support = [np.abs(A).sum(axis=(0, 1)) > 1e-12 for A in tensors]
    for s in support[1:]:
        np.testing.assert_array_equal(s, support[0])


def test_n_blocks_controls_returned_chain_length():
    """n_blocks sets the number of returned tensors."""
    # bond_dim == d, so blocking=1 is admissible and the blocking search is not involved.
    for nb in (2, 3, 5):
        tensors, _, _ = generate_random_mpu(
            n_blocks=nb, d=2, bond_dim=2, mode="branching", blocking="auto", seed=0
        )
        assert len(tensors) == nb


def test_satisfies_assumption_1_detects_a_rank_deficient_chain():
    """Assumption 1 is reported as violated for an unblocked rank-deficient chain."""
    # Unblocked D = 4 at d = 2 cannot be full rank at the first cut.
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=2, d=2, bond_dim=4, mode="branching", blocking=1, seed=0
    )
    assert not satisfies_assumption_1(tensors, l_vec, r_vec)


@pytest.mark.parametrize("mode,d,bond_dim", [("fusion", 2, 4), ("fusion", 3, 5)])
def test_fusion_rejects_bond_dim_exceeding_d(mode, d, bond_dim):
    """Mode II with bond_dim > d is rejected up front."""
    # The character matrix of Z_bond_dim on C^d has rank min(d, bond_dim); extra channels
    # would silently collapse under canonical form.
    with pytest.raises(ValueError, match="needs d >="):
        generate_random_mpu(n_sites=2, d=d, bond_dim=bond_dim, mode=mode, blocking=1)


@pytest.mark.parametrize("mode,d,bond_dim", [("fusion", 4, 4), ("fusion", 5, 3), ("grafted", 4, 8)])
def test_fusion_and_grafted_reach_exact_bond_dim_when_d_suffices(mode, d, bond_dim):
    """The requested bond dimension is preserved by canonicalization when d suffices."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=2, d=d, bond_dim=bond_dim, mode=mode, blocking=1, seed=0
    )
    canon_tensors, _, _ = to_canonical_form(tensors, l_vec, r_vec)
    assert canon_tensors[0].shape[3] == bond_dim


def test_generator_rejects_inconsistent_arguments():
    """Inconsistent generator arguments raise ValueError."""
    with pytest.raises(ValueError):  # odd bond_dim has no grafted factorization
        generate_random_mpu(n_sites=2, d=2, bond_dim=3, mode="grafted")
    with pytest.raises(ValueError):  # blocking must divide the chain
        generate_random_mpu(n_sites=3, d=2, bond_dim=2, blocking=2)
    with pytest.raises(ValueError):  # GeneralMPU needs at least 2 tensors
        generate_random_mpu(n_sites=2, d=2, bond_dim=2, blocking=2)
    with pytest.raises(ValueError):
        generate_random_mpu(n_sites=2, d=2, bond_dim=2, mode="nonexistent")


@pytest.mark.parametrize(
    "mode,d,bond_dim",
    [
        ("branching", 2, 4),
        ("branching", 2, 6),
        ("branching", 3, 5),
        ("branching", 3, 6),
        ("branching", 4, 6),
        ("branching", 5, 8),
    ],
)
def test_to_canonical_form_reaches_full_rank_and_stays_unitary(mode, d, bond_dim):
    """Canonical form satisfies Assumption 1 for redundant bonds and stays unitary."""
    # Only "branching" is parametrized: fusion and grafted with insufficient d are
    # rejected at generation time, so no redundant instance of them can be built.
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=3, d=d, bond_dim=bond_dim, mode=mode, blocking=1, seed=0
    )
    assert not satisfies_assumption_1(tensors, l_vec, r_vec)

    canon_tensors, canon_l, canon_r = to_canonical_form(tensors, l_vec, r_vec)
    assert satisfies_assumption_1(canon_tensors, canon_l, canon_r)

    _, _, residual = compute_chain_boundary_operators(canon_tensors, canon_l, canon_r)
    assert abs(1.0 - complex(residual)) < 1e-8


def test_to_canonical_form_gives_paper_eq26_structure():
    """In canonical form L_k is proportional to 1 and R_k is positive diagonal.

    This is the boundary structure of the paper's canonical form (eq. 26).
    """
    # Only proportionality of L_k to the identity is asserted: the boundary-operator
    # sweep divides by d at each step, which adds a scalar d**(-depth/2) to L_k.
    from TreeMPU.utils import (
        compute_chain_boundary_operators as _cbo,
    )
    from TreeMPU.utils import (
        matrix_sqrt_hermitian,
    )

    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=3, d=2, bond_dim=4, mode="branching", blocking=1, seed=0
    )
    ct, cl, cr = to_canonical_form(tensors, l_vec, r_vec)
    L2_list, R2_list, _ = _cbo(ct, cl, cr)
    for b in range(len(ct) - 1):
        L = matrix_sqrt_hermitian(L2_list[b])
        R2 = R2_list[b]
        Dk = L.shape[0]
        scale = np.trace(L).real / Dk
        np.testing.assert_allclose(L, scale * np.eye(Dk), atol=1e-8)
        assert scale > 1e-10
        off_diag = R2 - np.diag(np.diag(R2))
        assert np.linalg.norm(off_diag) < 1e-8
        assert np.all(np.diag(R2).real > 1e-10)
