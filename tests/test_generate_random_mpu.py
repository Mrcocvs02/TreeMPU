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
    """
    Contract the whole chain densely, in this package's
    ``A[p_in, p_out, bond_L, bond_R]`` convention with sites running left to
    right and ``l_vec`` meeting ``tensors[0]``.

    Only ever called on deliberately tiny chains in this file: it is the one
    check that actually pins down the index/site-order conversion the
    generator performs (the reference's product runs A_N ... A_1 with the
    opposite index layout), which no amount of contracting the *boundary*
    operators can catch -- those trace over both physical legs symmetrically
    and so are blind to exactly the transpose that would be wrong.
    """
    M = np.einsum("ijab->aijb", tensors[0])
    for A in tensors[1:]:
        M = np.einsum("aIJc,klcb->aIkJlb", M, A)
        s = M.shape
        M = M.reshape(s[0], s[1] * s[2], s[3] * s[4], s[5])
    return np.einsum("a,aIJb,b->IJ", l_vec, M, r_vec)


@pytest.mark.parametrize("mode", ["branching", "fusion", "grafted"])
def test_generated_mpu_is_exactly_unitary(mode):
    """Every mode's defining algebraic condition must yield an exactly unitary chain."""
    tensors, l_vec, r_vec = generate_random_mpu(n_sites=3, d=2, bond_dim=2, mode=mode, seed=0)
    U = _dense_chain(tensors, l_vec, r_vec)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-10)


@pytest.mark.parametrize("mode", ["branching", "fusion", "grafted"])
def test_generated_mpu_unitary_at_every_length(mode):
    """
    The whole point of the certified generators is that one fixed rule works
    for *every* N, not just the sampled one -- so truncating the same chain
    must stay unitary rather than only the full length doing so.
    """
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
    """Blocking raises the physical dimension without touching the bond dimension."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=4, d=2, bond_dim=3, mode="branching", blocking=2, seed=2
    )
    assert len(tensors) == 2
    assert tensors[0].shape == (4, 4, 3, 3)

    U = _dense_chain(tensors, l_vec, r_vec)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-10)


def test_block_mpu_tensors_matches_unblocked_contraction():
    """Blocking is a pure regrouping: the dense chain operator must not change."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=4, d=2, bond_dim=2, mode="fusion", blocking=1, seed=3
    )
    np.testing.assert_allclose(
        _dense_chain(block_mpu_tensors(tensors, 2), l_vec, r_vec),
        _dense_chain(tensors, l_vec, r_vec),
        atol=1e-12,
    )


def test_uniform_flag_gives_translation_invariant_bulk():
    tensors, _, _ = generate_random_mpu(
        n_sites=4, d=2, bond_dim=2, mode="branching", uniform=True, seed=4
    )
    for A in tensors[1:]:
        np.testing.assert_allclose(A, tensors[0])


def test_chain_boundary_residual_is_one():
    """The O(N) unitarity residual the package itself checks."""
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
    """
    blocking="auto" must *verify*, not predict: these fail the
    minimal-bond-dimension check at the blocking a pure d**(2L) >= D count
    would suggest, because the sampled digraph's reachable set -- not the
    dimension count -- is what limits the environment's rank. (fusion/grafted
    are not parametrized here: for them, insufficient d is a hard, N-independent
    representation-span limit -- see test_fusion_rejects_bond_dim_exceeding_d --
    so generate_random_mpu rejects it upfront, before blocking is even tried;
    blocking cannot rescue it since it only raises the *physical* dimension of
    already-redundant channels, never the number of independent ones.)
    """
    tensors, l_vec, r_vec = generate_random_mpu(
        n_blocks=2, d=d, bond_dim=bond_dim, mode=mode, blocking="auto", seed=0
    )
    assert satisfies_assumption_1(tensors, l_vec, r_vec)
    assert len(tensors) == 2
    assert tensors[0].shape[2] == bond_dim


def test_auto_blocking_reports_failure_rather_than_returning_bad_chain():
    """
    branching at d=2 caps every row's out-degree at 2 (that is what (RR) can
    resolve), so the reachable set -- hence the boundary rank -- grows only
    a couple of states per blocked site. A large D therefore needs a large
    blocking factor, and when max_blocking is too small to get there the
    generator must say so rather than hand back a chain GeneralMPU would
    reject. (D=8 typically needs L in the 5-8 range; 4 is not enough.)
    """
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
    """
    The Mode I theorem (thesis thm:mode1-unitarity) allows the digraph to vary with the site, and the generator
    uses that whenever the chain is non-uniform. It is not cosmetic: with a
    single fixed digraph the reachable set from the readout vertex
    saturates after a few steps (measured: D=4 stalls at ~3.5 of 4 states
    from L=4 on), so blocking stops buying any rank and these cases were
    unreachable at *every* blocking factor. Resampling per site keeps the
    set growing, which is what makes them admissible at all.
    """
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
    """A translation-invariant chain cannot vary its digraph, by definition."""
    tensors, _, _ = generate_random_mpu(
        n_sites=4, d=2, bond_dim=3, mode="branching", uniform=True, seed=0
    )
    support = [np.abs(A).sum(axis=(0, 1)) > 1e-12 for A in tensors]
    for s in support[1:]:
        np.testing.assert_array_equal(s, support[0])


def test_n_blocks_controls_returned_chain_length():
    # bond_dim == d here so blocking=1 is trivially admissible (the digraph
    # reachability search this file exercises elsewhere is orthogonal to
    # what this test checks: that n_blocks controls the returned length).
    for nb in (2, 3, 5):
        tensors, _, _ = generate_random_mpu(
            n_blocks=nb, d=2, bond_dim=2, mode="branching", blocking="auto", seed=0
        )
        assert len(tensors) == nb


def test_satisfies_assumption_1_detects_a_rank_deficient_chain():
    """Unblocked D=4 at d=2 cannot be full rank at the first cut (d**2 = 4 < ...)."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=2, d=2, bond_dim=4, mode="branching", blocking=1, seed=0
    )
    assert not satisfies_assumption_1(tensors, l_vec, r_vec)


@pytest.mark.parametrize("mode,d,bond_dim", [("fusion", 2, 4), ("fusion", 3, 5)])
def test_fusion_rejects_bond_dim_exceeding_d(mode, d, bond_dim):
    """
    Mode II's group-convolution channels are built inside a d-dimensional
    representation (characters of Z_bond_dim on C^d), so at most d of them
    are ever linearly independent -- the d x bond_dim character matrix
    V[k,g] = omega^{kg} has rank min(d, bond_dim) (a genuine DFT submatrix
    whenever d <= bond_dim). Asking for more channels than d silently
    collapses under canonical form rather than raising, so the generator
    must reject it upfront instead.
    """
    with pytest.raises(ValueError, match="needs d >="):
        generate_random_mpu(n_sites=2, d=d, bond_dim=bond_dim, mode=mode, blocking=1)


@pytest.mark.parametrize("mode,d,bond_dim", [("fusion", 4, 4), ("fusion", 5, 3), ("grafted", 4, 8)])
def test_fusion_and_grafted_reach_exact_bond_dim_when_d_suffices(mode, d, bond_dim):
    """The bond_dim parameter is the actual bond dimension, not a channel
    count that canonical form might later shrink -- verified by canonicalizing
    and checking nothing was dropped."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=2, d=d, bond_dim=bond_dim, mode=mode, blocking=1, seed=0
    )
    canon_tensors, _, _ = to_canonical_form(tensors, l_vec, r_vec)
    assert canon_tensors[0].shape[3] == bond_dim


def test_generator_rejects_inconsistent_arguments():
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
    """
    The paper's claim for the nonuniform case: canonical form makes
    Assumption 1 unnecessary, regardless of the caller's (possibly
    redundant) bond dimension. Every case here failed satisfies_assumption_1
    unblocked before to_canonical_form existed. ("branching" only: "fusion"/
    "grafted" with insufficient d are now rejected at generation time -- see
    test_fusion_rejects_bond_dim_exceeding_d -- so there is no way to build
    a redundant instance of those two modes through the public API any more.)
    """
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=3, d=d, bond_dim=bond_dim, mode=mode, blocking=1, seed=0
    )
    assert not satisfies_assumption_1(tensors, l_vec, r_vec)

    canon_tensors, canon_l, canon_r = to_canonical_form(tensors, l_vec, r_vec)
    assert satisfies_assumption_1(canon_tensors, canon_l, canon_r)

    _, _, residual = compute_chain_boundary_operators(canon_tensors, canon_l, canon_r)
    assert abs(1.0 - complex(residual)) < 1e-8


def test_to_canonical_form_gives_paper_eq26_structure():
    """
    Eq. (26): in canonical form L_k is proportional to the identity and R_k
    is diagonal with strictly positive entries (the Choi-state Schmidt
    values) -- checked directly, not just via the rank/unitarity proxies
    used elsewhere.

    Only *proportionality* to the identity is asserted, not L_k == 1 exactly:
    compute_chain_boundary_operators divides by d at every sweep step (its
    own normalization, chosen so the chain's global unitarity residual comes
    out to exactly 1), so L_k picks up a d**(-depth/2) scalar the paper's raw
    Choi-MPS SVD convention does not carry. That prefactor is a convention
    mismatch between the two normalizations, not a violation of Eq. (26)'s
    actual claim, which is about L_k's *direction* (proportional to identity),
    not its overall scale.
    """
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
