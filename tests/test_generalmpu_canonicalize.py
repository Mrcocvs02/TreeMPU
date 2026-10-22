"""``GeneralMPU(canonicalize=...)`` integration.

For non-uniform MPUs the route of the paper is the canonical form rather than
Assumption 1. The tests verify that chains rejected with ``canonicalize=False``
are accepted by the default canonicalizing constructor with a reduced bond
dimension, that canonicalization leaves an already minimal chain unchanged,
and that the incompatible option pair ``boundary_opt_gauge`` with
``boundary_opt_p`` is rejected at construction.
"""

import numpy as np
import pytest

from mpu_generators import generate_random_mpu
from TreeMPU.MPU import GeneralMPU


@pytest.mark.parametrize(
    "mode,d,bond_dim,n_sites",
    # n_sites = 2 (a single bond) keeps eager circuit synthesis cheap; each case
    # canonicalizes down to an effective bond dimension of at most 2.
    [("branching", 2, 4, 2), ("branching", 2, 3, 2), ("branching", 3, 4, 2)],
)
def test_canonicalize_true_accepts_what_false_rejects(mode, d, bond_dim, n_sites):
    """Canonicalization accepts and reduces chains that canonicalize=False rejects."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=n_sites, d=d, bond_dim=bond_dim, mode=mode, blocking=1, seed=0
    )
    with pytest.raises(ValueError):
        GeneralMPU(tensors, l_vec, r_vec, canonicalize=False)

    mpu = GeneralMPU(tensors, l_vec, r_vec)  # canonicalize=True is the default
    assert mpu.canonicalized
    assert all(D <= bond_dim for D in mpu._bond_dims)
    assert np.isfinite(mpu.q)


def test_canonicalize_is_a_noop_on_an_already_minimal_chain():
    """Canonicalization leaves the bond dimensions and q of a minimal chain unchanged."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=2, d=2, bond_dim=2, mode="fusion", blocking=1, seed=0
    )
    mpu_off = GeneralMPU(tensors, l_vec, r_vec, canonicalize=False)
    mpu_on = GeneralMPU(tensors, l_vec, r_vec, canonicalize=True)
    assert mpu_off._bond_dims == mpu_on._bond_dims == [2]
    assert mpu_off.q == pytest.approx(mpu_on.q, rel=1e-6)


def test_boundary_opt_gauge_with_boundary_opt_p_raises_early():
    """boundary_opt_gauge together with boundary_opt_p is rejected at construction."""
    # Every boundary_opt_p window is anchored at l_vec or r_vec, so it has a trivial
    # bond on that side, while gauge=True needs one shared bond dimension at both ends.
    tensors, l_vec, r_vec = generate_random_mpu(
        n_blocks=2, d=2, bond_dim=3, mode="branching", blocking="auto", seed=0
    )
    with pytest.raises(ValueError, match="boundary_opt_gauge"):
        GeneralMPU(
            tensors,
            l_vec,
            r_vec,
            boundary_opt_p=1,
            boundary_opt_gauge=True,
        )
