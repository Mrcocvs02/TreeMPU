"""
GeneralMPU(canonicalize=...) integration: the paper's route for nonuniform
MPUs is canonical form, not Assumption 1 -- these chains would previously be
rejected by GeneralMPU(canonicalize=False) outright.
"""

import numpy as np
import pytest

from mpu_generators import generate_random_mpu
from TreeMPU.MPU import GeneralMPU


@pytest.mark.parametrize(
    "mode,d,bond_dim,n_sites",
    # n_sites=2 (a single bond) keeps eager circuit synthesis cheap; the
    # point here is the canonicalize behavior, not stress-testing D. Each
    # canonicalizes down to D_eff=2 (checked below), matching the D=2
    # ceiling this machine's memory budget allows for full circuit synthesis.
    [("branching", 2, 4, 2), ("branching", 2, 3, 2), ("branching", 3, 4, 2)],
)
def test_canonicalize_true_accepts_what_false_rejects(mode, d, bond_dim, n_sites):
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
    """A chain sampled at d >= bond_dim (fusion) is already minimal; canonical
    form should not shrink it further, and canonicalize=False must accept it."""
    tensors, l_vec, r_vec = generate_random_mpu(
        n_sites=2, d=2, bond_dim=2, mode="fusion", blocking=1, seed=0
    )
    mpu_off = GeneralMPU(tensors, l_vec, r_vec, canonicalize=False)
    mpu_on = GeneralMPU(tensors, l_vec, r_vec, canonicalize=True)
    assert mpu_off._bond_dims == mpu_on._bond_dims == [2]
    assert mpu_off.q == pytest.approx(mpu_on.q, rel=1e-6)


def test_boundary_opt_gauge_with_boundary_opt_p_raises_early():
    """
    Every boundary_opt_p window is anchored to l_vec or r_vec, so with
    canonicalize=True (the default) it always has a trivial bond on that
    side -- gauge=True can never apply there (it needs one shared bond
    dimension for both ends). This must be rejected immediately, not left to
    crash on the first window optimize_boundary_operators actually attempts.
    """
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
