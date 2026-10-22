# TreeMPU/utils.py
r"""Boundary operators, conditioning number and tensor utilities for TreeMPU.

Contents
--------
Boundary constructions
    :func:`optimize_q_unif` (canonical boundaries with optimized
    :math:`\sigma,\tau`), :func:`mode1_exact_boundary`,
    :func:`solve_qD_boundary`, :func:`optimize_boundary_operators`,
    :func:`compute_chain_boundary_operators`, :func:`optimize_chain_sigma_tau`.
Conditioning and merging
    :func:`conditioning_number`, :func:`get_merging_operator`.
Tensor utilities
    :func:`get_mpo_site_tensors`, :func:`block_mpu_tensors`,
    :func:`to_canonical_form`, :func:`matrix_sqrt_hermitian`,
    :func:`dilate_isometry_to_unitary`.

Conventions
-----------
Site tensors are ``A[p_in, p_out, bond_L, bond_R]``. Boundary operators are
stored as ``L[ext, bond]`` and ``R[ext, bond]`` (transposed with respect to
the thesis product ``L A R``), where ``ext`` is the index of the bond
register in the circuit; the merging kernel between two blocks is
:math:`K = R^{-\top}L^{-1}` and :math:`q=\|K\|_F`. The bond dimension is
:math:`\chi` in prose and ``D`` in code.

Importing this module enables 64-bit precision in JAX
(``jax_enable_x64``); the optimizers select the CPU platform
(``jax_platform_name="cpu"``), which applies process-wide.

References
----------
Unqualified numbers in this module (Eq. A33, Eq. 26, Lemma 1, Definition 3,
Theorem 1', ...) refer to G. Styliaris, R. Trivedi, J. I. Cirac, "Quantum
circuits for matrix-product unitaries", PRL 135, 260602 (2025) and its
supplemental material; results of the thesis are cited by LaTeX label
(e.g. ``thm:schmidt_bound``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np  # type: ignore
import quimb.tensor as qtn  # type: ignore
import scipy.linalg as la  # type: ignore
from jax.scipy.linalg import expm as jax_expm
from scipy.optimize import least_squares, minimize

jax.config.update("jax_enable_x64", True)


def optimize_q_unif(T_blocked, l_vec, r_vec, eps_reg=1e-8, rng=None):
    r"""Optimize the density matrices of the canonical boundary.

    The canonical boundary operators are
    :math:`L^2=\operatorname{Tr}_{\rm out}[(\sigma\otimes\ell\ell^\dagger)\,A^\dagger A]`
    and analogously :math:`R^2` with :math:`\tau` (thesis, canonical
    boundaries). The density matrices act on the *input* physical leg and
    the output leg is traced. Every choice gives admissible (isometric)
    boundaries; the function minimizes :math:`q=\|R^{-\top}L^{-1}\|_F` over
    them with L-BFGS-B and JAX gradients.

    Parameters
    ----------
    T_blocked : ndarray
        Bulk tensor ``A[p_in, p_out, bond_L, bond_R]``.
    l_vec, r_vec : ndarray
        Boundary vectors of shape ``(bond,)``.
    eps_reg : float, optional
        Tikhonov term added to :math:`L^2, R^2` before inversion
        (default ``1e-8``).
    rng : numpy.random.Generator, optional
        Source of the small random perturbation of the starting point. If
        omitted, the global NumPy RNG (``numpy.random.randn``) is used.

    Returns
    -------
    sigma, tau : jax.Array
        Optimized density matrices of shape ``(d, d)`` with unit trace.
    q : float
        Value of :math:`q` at the optimum, including the regularization.

    Notes
    -----
    The start is :math:`I/d` plus a perturbation of scale ``1e-3``; pass
    ``rng`` (as :class:`~TreeMPU.MPU.UniformMPU` does) for reproducible
    results. The optimum is a local one.
    """
    dim = T_blocked.shape[0]
    n_params = 2 * dim**2

    def vec_to_rho(p):
        X = p[: dim**2].reshape((dim, dim)) + 1j * p[dim**2 :].reshape((dim, dim))
        rho = X @ X.conj().T
        return rho / jnp.trace(rho).real

    @jax.jit
    def loss_and_grad(params, T, L_in, R_in):
        def loss_fn(p):
            sigma = vec_to_rho(p[:n_params])
            tau = vec_to_rho(p[n_params:])

            # L^2, R^2 with sigma/tau on the INPUT physical leg and the
            # output leg traced, as in UniformMPU._compute_boundary_operators.
            L2 = jnp.einsum("ij, iopq, pm, jomn -> qn", sigma, T, L_in, T.conj())
            R2 = jnp.einsum("ij, iopq, qn, jomn -> pm", tau, T, R_in, T.conj())

            D = L2.shape[0]
            I_reg = eps_reg * jnp.eye(D)

            # Direct inversion: stable under autodiff
            inv_L = jnp.linalg.inv(L2 + I_reg)
            inv_R = jnp.linalg.inv(R2 + I_reg)

            # Optimized trace computation: tr(A @ B.T) = sum(A * B)
            trace_val = jnp.sum(inv_R * inv_L)

            return jnp.sqrt(jnp.abs(jnp.real(trace_val)))

        return jax.value_and_grad(loss_fn)(params)

    T_j = jnp.asarray(T_blocked)
    L_in_j = jnp.outer(l_vec, l_vec.conj())
    R_in_j = jnp.outer(r_vec, r_vec.conj())

    def scipy_wrapper(p):
        val, grad = loss_and_grad(p, T_j, L_in_j, R_in_j)
        # Scipy requires native float64
        return np.asarray(val, dtype=np.float64), np.asarray(grad, dtype=np.float64)

    # Symmetric initialization + noise
    noise = np.random.randn(2 * n_params) if rng is None else rng.standard_normal(2 * n_params)
    x0 = noise * 0.001
    eye_flat = np.eye(dim).flatten() / np.sqrt(dim)
    x0[: dim**2] += eye_flat
    x0[n_params : n_params + dim**2] += eye_flat

    res = minimize(
        scipy_wrapper,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
    )

    return vec_to_rho(res.x[:n_params]), vec_to_rho(res.x[n_params:]), res.fun


def mode1_exact_boundary(
    A: np.ndarray,
    l_vec: np.ndarray,
    r_vec: np.ndarray,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    r"""Return the closed-form boundary of a Mode I tensor with :math:`q=\chi`.

    Applies to a Mode I ("branching") ``UniformMPU`` with single-vertex
    readout, ``l_vec`` :math:`=e_v` and ``r_vec`` :math:`=\sum_u e_u`, the
    boundary used by ``mpu_generators.MPUGenerator.branching``. No
    optimization is needed. The row-resolution condition (RR) that
    ``mpu_generators.base._row_resolution_blocks`` certifies for Mode I
    tensors,

    .. math::

        \sum_u A[v,u]^\dagger A[v,u] = \mathbb 1_d, \qquad
        A[v,u]^\dagger A[v,u'] = 0 \quad (u\neq u'),

    is the isometry certificate for this boundary up to one scalar:

    - ``l_vec`` :math:`=e_v`, :math:`R=\mathbb 1_\chi`: the length-1
      isometry is row :math:`v` of (RR), so :math:`V^\dagger V=\mathbb 1_d`.
    - ``r_vec`` :math:`=\sum_u e_u`, :math:`L=\mathbb 1_\chi`: summing (RR)
      over all :math:`\chi` branches gives
      :math:`V^\dagger V=\chi\,\mathbb 1_d`, which the rescaling
      :math:`L=\mathbb 1_\chi/\sqrt\chi` corrects.

    With :math:`L^2=\mathbb 1_\chi/\chi` and :math:`R^2=\mathbb 1_\chi` one
    has :math:`\operatorname{tr}(L^2(R^2)^T)=1` and
    :math:`R^2=((L^2)^T)^{-1}/\chi`, which is the equality case of the bound
    discussed in :func:`solve_qD_boundary`, so :math:`q=\chi`.

    Parameters
    ----------
    A : ndarray
        Bulk tensor ``A[p_in, p_out, bond_L, bond_R]`` of shape
        ``(d, d, D, D)``.
    l_vec, r_vec : ndarray
        Boundary vectors of shape ``(D,)``. ``l_vec`` must equal a standard
        basis vector and ``r_vec`` the all-ones vector, each within ``tol``.
    tol : float, optional
        Tolerance of the structural checks on ``l_vec`` and ``r_vec`` and of
        the final isometry check (default ``1e-8``).

    Returns
    -------
    L : ndarray
        ``eye(D) / sqrt(D)``, shape ``(D, D)``.
    R : ndarray
        ``eye(D)``, shape ``(D, D)``.
    q : float
        ``float(D)``.
    iso_residual : float
        ``max(||V_L^dagger V_L - I_d||, ||V_R^dagger V_R - I_d||)``
        (Frobenius norm), computed by contracting the given ``A``.

    Raises
    ------
    ValueError
        If ``A`` or the boundary vectors have the wrong shape, if ``l_vec``
        or ``r_vec`` does not have the required structure, or if
        ``iso_residual > tol`` (``A`` does not satisfy (RR) for this
        boundary).

    Notes
    -----
    The closed form is not a general boundary construction: it depends on
    the single-vertex readout and on ``A`` satisfying (RR). The latter holds
    for tensors sampled by the ``mpu_generators`` Mode I generators but not
    for an arbitrary tensor with the same boundary vectors, hence the exact
    isometry check on the given tensor.
    """
    d = A.shape[0]
    D = A.shape[2]
    if A.shape != (d, d, D, D):
        raise ValueError(f"A must have shape (d, d, D, D); got {A.shape}.")

    l_vec = np.asarray(l_vec, dtype=complex)
    r_vec = np.asarray(r_vec, dtype=complex)
    if l_vec.shape != (D,) or r_vec.shape != (D,):
        raise ValueError(
            f"l_vec/r_vec must have shape (D,)=({D},); got {l_vec.shape}/{r_vec.shape}."
        )

    v = int(np.argmax(np.abs(l_vec)))
    e_v = np.zeros(D, dtype=complex)
    e_v[v] = 1.0
    if np.linalg.norm(l_vec - e_v) > tol:
        raise ValueError(
            "mode1_exact_boundary requires l_vec to be a single standard "
            f"basis vector e_v (single-vertex readout); got {l_vec}, "
            f"nearest candidate e_{v}."
        )
    ones_D = np.ones(D, dtype=complex)
    if np.linalg.norm(r_vec - ones_D) > tol:
        raise ValueError(
            "mode1_exact_boundary requires r_vec to be the all-ones "
            f"vector (sum over every branch); got {r_vec}."
        )

    L = np.eye(D, dtype=complex) / np.sqrt(D)
    R = np.eye(D, dtype=complex)

    # Exact isometry check on the given tensor: the row-resolution algebra
    # above is a derivation, not a proof that this particular A satisfies it.
    V_L = np.transpose(A[:, :, v, :], (1, 2, 0)).reshape(d * D, d)
    defect_L = float(np.linalg.norm(V_L.conj().T @ V_L - np.eye(d)))

    r_sum = A.sum(axis=3) / np.sqrt(D)  # (p_in, p_out, bond_L)
    V_R = np.transpose(r_sum, (1, 2, 0)).reshape(d * D, d)
    defect_R = float(np.linalg.norm(V_R.conj().T @ V_R - np.eye(d)))

    iso_residual = max(defect_L, defect_R)
    if iso_residual > tol:
        raise ValueError(
            f"mode1_exact_boundary: l_vec/r_vec match the required "
            f"single-vertex-readout structure, but A does not satisfy the "
            f"row resolution condition (RR) this closed form depends on "
            f"(exact isometry defect {iso_residual:.3e} > tol {tol:.1e}). "
            "This tensor was not sampled from a certified Mode I "
            "generator (or was altered afterward); use optimize_q_unif or "
            "the naive I/d construction instead."
        )

    return L, R, float(D), iso_residual


def _hermitian_basis(D: int) -> list[np.ndarray]:
    """Return a real basis of the ``D x D`` Hermitian matrices.

    The basis consists of the ``D`` diagonal projectors followed by a
    symmetric and an antisymmetric (``+-i``) pair for each off-diagonal index
    pair ``i < j``.

    Parameters
    ----------
    D : int
        Matrix dimension.

    Returns
    -------
    list of ndarray
        ``D**2`` complex ``(D, D)`` matrices, linearly independent over the
        reals.
    """
    basis = []
    for i in range(D):
        E = np.zeros((D, D), dtype=complex)
        E[i, i] = 1.0
        basis.append(E)
    for i in range(D):
        for j in range(i + 1, D):
            E = np.zeros((D, D), dtype=complex)
            E[i, j] = 1.0
            E[j, i] = 1.0
            basis.append(E)
            F = np.zeros((D, D), dtype=complex)
            F[i, j] = 1j
            F[j, i] = -1j
            basis.append(F)
    return basis


def _affine_hermitian_solve(
    images: list[np.ndarray], target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve ``sum_k theta_k images[k] = target`` for real ``theta``.

    ``images[k]`` is the image, a ``d x d`` matrix, of the ``k``-th vector of
    :func:`_hermitian_basis` under a real-linear map. Real and imaginary
    parts are stacked into one real system so that ``theta`` is real and the
    reconstructed matrix is Hermitian. The system is solved by least squares.

    Parameters
    ----------
    images : list of ndarray
        ``K`` complex ``(d, d)`` matrices.
    target : ndarray
        Right-hand side, shape ``(d, d)``.

    Returns
    -------
    theta0 : ndarray
        Particular (minimum-norm) solution.
    null_basis : ndarray
        Orthonormal real basis (columns) of the homogeneous solutions.
    residual : float
        Norm of the residual of ``theta0``.
    """
    d = target.shape[0]
    K = len(images)
    M = np.zeros((2 * d * d, K), dtype=float)
    for k, img in enumerate(images):
        flat = img.flatten()
        M[: d * d, k] = flat.real
        M[d * d :, k] = flat.imag
    b = np.zeros(2 * d * d, dtype=float)
    b[: d * d] = target.flatten().real
    b[d * d :] = target.flatten().imag
    theta0, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
    residual = float(np.linalg.norm(M @ theta0 - b))
    null_basis = la.null_space(M)
    return theta0, null_basis, residual


#: Largest Frobenius residual of the length-1 isometry conditions accepted by
#: :func:`solve_qD_boundary`; every pair is re-validated exactly by ``UniformMPU``.
_QD_RESIDUAL_TOL = 1e-6
#: Largest squared bulk isometry defect accepted by :func:`solve_qD_boundary`.
_QD_BULK_DEFECT_TOL = 1e-8
#: Boundary residual or bulk defect above which :func:`solve_qD_boundary`
#: re-solves once in a whitened bond basis.
_QD_REFINE_TOL = 1e-9


def solve_qD_boundary(
    A: np.ndarray,
    l_vec: np.ndarray,
    r_vec: np.ndarray,
    tol: float = 1e-8,
    n_restarts: int = 30,
    seed: int = 0,
    k_bulk: int = 3,
    _whitened: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    r"""Return boundary operators with :math:`q=\chi` for a bulk tensor.

    The construction is a short fit inside an affine family of Hermitian
    matrices. With ``L2 = L^dagger L`` and ``R2 = R^dagger R`` (Gram matrices
    without conjugated coefficients, see Notes), it proceeds as follows.

    1. Fix the ansatz ``R2 = (L2^T)^{-1} / D``, which gives
       :math:`\operatorname{tr}(L_2 R_2^T)=1` and hence :math:`q=\chi` for
       any positive-definite ``L2``.
    2. Solve the linear length-1 isometry condition of the right boundary
       leg for ``L2`` by real-stacked least squares and a null-space
       computation over the real vector space of Hermitian matrices. This
       gives an affine family ``L2 = L2_0 + sum_k psi_k N_k``.
    3. If the family has free parameters, locate a positive-definite centre
       (maximum of the smallest eigenvalue; ``R2`` diverges at
       ``det L2 = 0``, so the fit must start on the positive-definite side)
       and run ``n_restarts`` least-squares fits from starts inside the
       positive-definite region. The residual contains the length-1
       condition of the left boundary leg for ``R2 = (L2^T)^{-1} / D``, the
       bulk isometry defect at prefix lengths ``1, ..., k_bulk`` and a
       penalty on small eigenvalues of ``L2``. The best restart is polished.
    4. If the residuals exceed ``_QD_REFINE_TOL``, solve once more in the
       bond basis where the first solution is the identity (whitening,
       ``W = L2^{1/2}``) and map the result back. If the whitened solve
       raises, the original pair is kept.
    5. Accept the pair only if the boundary residual is at most
       ``_QD_RESIDUAL_TOL`` and the bulk defect at most
       ``_QD_BULK_DEFECT_TOL``; otherwise raise. Every accepted pair is
       re-validated by an exact isometry check in ``UniformMPU``.

    Parameters
    ----------
    A : ndarray
        Bulk tensor ``A[p_in, p_out, bond_L, bond_R]`` of shape
        ``(d, d, D, D)``.
    l_vec, r_vec : ndarray
        Boundary vectors of shape ``(D,)``. No structure is required, unlike
        :func:`mode1_exact_boundary`.
    tol : float, optional
        Smallest admissible eigenvalue of the positive-definite ``L2``
        (default ``1e-8``).
    n_restarts : int, optional
        Number of random restarts of the least-squares fit; used only if the
        ``L2`` family has free parameters (default 30).
    seed : int, optional
        Seed of the restarts (default 0); the result is deterministic.
    k_bulk : int, optional
        Largest prefix length at which the bulk isometry defect enters the
        fit (default 3). The returned diagnostics report the defect at
        ``k_bulk + 1`` as an out-of-sample check.
    _whitened : bool, optional
        Internal flag marking the recursive call in the whitened basis; not
        part of the public interface (default ``False``).

    Returns
    -------
    L, R : ndarray
        Hermitian positive-definite square roots of ``L2`` and ``R2``
        (via :func:`matrix_sqrt_hermitian`), of shape ``(D, D)``, usable as
        ``UniformMPU.L`` and ``UniformMPU.R``.
    q : float
        ``float(D)``, the value fixed by the ansatz; it is attained by a
        valid boundary only if the diagnostics below are small.
    diagnostics : dict
        ``cond1_residual`` and ``cond2_residual`` (Frobenius residuals of
        the two length-1 conditions at the returned pair),
        ``bulk_defect_at_fit`` (squared defect summed over prefix lengths
        ``1..k_bulk``), ``bulk_defect_out_of_sample`` (same at
        ``1..k_bulk+1``), ``tr_L2_R2T`` (equal to 1 by the ansatz),
        ``L2_min_eigenvalue`` and ``null_dim_L2_family`` (number of free
        parameters of the fit). After a whitened re-solve the dictionary
        is that of the inner call plus ``whitened = True``.

    Raises
    ------
    ValueError
        If ``A`` does not have shape ``(d, d, D, D)``, if the affine family
        of ``L2`` has no positive-definite member, if the best ``L2`` is not
        positive definite within ``tol``, or if the residual or bulk defect
        exceeds the acceptance limits (``_QD_RESIDUAL_TOL``,
        ``_QD_BULK_DEFECT_TOL``). The construction then does not apply to
        this tensor and boundary.

    Notes
    -----
    **Proved.** For positive-definite ``L2``, ``R2`` set
    :math:`M=L_2^{1/2}R_2^TL_2^{1/2}`. Then
    :math:`\operatorname{tr}M=\operatorname{tr}(L_2R_2^T)` and
    :math:`\operatorname{tr}M^{-1}=q^2`. The Cauchy-Schwarz inequality
    :math:`\operatorname{tr}(M)\operatorname{tr}(M^{-1})\ge\chi^2` gives
    :math:`q\ge\chi/\sqrt{\operatorname{tr}(L_2R_2^T)}`, with equality iff
    :math:`M\propto\mathbb 1`. The ansatz of step 1 saturates it. The bound
    implies :math:`q\ge\chi` only when
    :math:`\operatorname{tr}(L_2R_2^T)\le 1`; in general only :math:`q\ge1`
    is proved (thesis), and site-dependent windowed boundaries reach
    :math:`q<\chi`.

    **Validated numerically only.** That the length-1 conditions together
    with the bulk defect at lengths ``1..k_bulk`` imply the isometry of
    longer blocks, and that a compatible ``L2`` exists. Existence is
    plausible because a certified MPU already satisfies
    :math:`U_N^\dagger U_N=\mathbb 1` for every :math:`N`, but it is not
    proved. Inspect ``diagnostics`` rather than assuming success.

    **Conventions.** The two length-1 conditions are linear in ``R2`` and
    ``L2`` separately:

    .. math::

        \sum_{c,c'} (R_2)_{cc'} B[c]^\dagger B[c'] = \mathbb 1_d, \qquad
        \sum_{a,a'} (L_2)_{aa'} C[a]^\dagger C[a'] = \mathbb 1_d,

    with :math:`B[c]=\sum_a l_a A[:,:,a,c]` and
    :math:`C[a]=\sum_c A[:,:,a,c]\,r_c`; the physical index contracted is
    the output leg. ``L2`` and ``R2`` appear unconjugated because
    ``UniformMPU.create_local_isometries`` contracts the second axis of both
    ``L[ext, bond]`` and ``R[ext, bond]`` with the bulk bond index. The bulk
    defect is evaluated by a bond-space recursion on the leg-swapped tensor
    (the site isometry maps the physical input leg to (output, external
    bonds)), and the right Gram matrix enters it transposed. These
    conventions are invisible for real symmetric ``L2``, ``R2`` (the natural
    bond basis of the generator families), so complex boundary vectors or a
    complex bond basis are needed to test them. The bulk defect is part of
    the fit because the length-1 conditions alone leave longer blocks
    unconstrained once the ``L2`` family has free parameters; ``L2`` and
    ``R2`` dress the repeated bulk tensor directly, not through ``l_vec`` and
    ``r_vec``.
    """
    d = A.shape[0]
    D = A.shape[2]
    if A.shape != (d, d, D, D):
        raise ValueError(f"A must have shape (d, d, D, D); got {A.shape}.")
    l_vec = np.asarray(l_vec, dtype=complex).reshape(D)
    r_vec = np.asarray(r_vec, dtype=complex).reshape(D)

    B = np.einsum("a,ijac->cij", l_vec, A)  # (D, d, d): B[c] = sum_a l_a A[:,:,a,c]
    C = np.einsum("ijac,c->aij", A, r_vec)  # (D, d, d): C[a] = sum_c A[:,:,a,c] r_c

    # Contract over the OUTPUT physical leg (j) so that the free indices
    # (i, k) are the INPUT leg: the isometry acts from p_in, consistent with
    # the V_L/V_R construction in mode1_exact_boundary.
    Gamma_B = np.einsum("cij,dkj->cdik", B.conj(), B)  # (D,D,d,d)
    Gamma_C = np.einsum("aij,bkj->abik", C.conj(), C)  # (D,D,d,d)

    I_d = np.eye(d, dtype=complex)
    herm_basis = _hermitian_basis(D)

    def matrix_of_theta(theta: np.ndarray) -> np.ndarray:
        """Return the Hermitian matrix with coordinates ``theta``."""
        M = np.zeros((D, D), dtype=complex)
        for k, Nk in enumerate(herm_basis):
            M += theta[k] * Nk
        return M

    def cond1_residual_matrix(R2: np.ndarray) -> np.ndarray:
        """Return the residual of the left-boundary length-1 condition."""
        # R2 enters unconjugated, exactly as L2 does in the second condition.
        return np.einsum("cd,cdjk->jk", R2, Gamma_B) - I_d

    # Bulk isometry residual on the leg-swapped tensor (see Notes, Conventions).
    G_bulk = jnp.array(_isometry_transfer_tensor(A.transpose(1, 0, 2, 3)))

    def bulk_defect(L2: np.ndarray, R2: np.ndarray, k: int) -> float:
        """Return the squared bulk defect summed over prefixes ``1..k``."""
        defect = _bondspace_isometry_defect(
            [G_bulk] * k, [d] * k, jnp.array(L2), jnp.array(R2.T), all_prefixes=True
        )
        return float(jnp.real(defect))

    # Step 3, second condition: defines the L2 family.
    images_L = [np.einsum("ab,abjk->jk", Nk, Gamma_C) for Nk in herm_basis]
    theta0_L, null_L, _ = _affine_hermitian_solve(images_L, I_d)

    K_null = null_L.shape[1]
    if K_null == 0:
        theta_star = theta0_L
    else:
        # Positive definiteness of L2 is part of the problem: the affine
        # family also contains indefinite solutions of the linear conditions,
        # and which one an unconstrained fit reaches depends on the bond
        # basis. Eigenvalues of L2 below ``pd_floor`` (a fixed fraction of the
        # particular solution's norm per unit of bond dimension) are
        # penalized in the residual vector.
        scale = float(np.linalg.norm(matrix_of_theta(theta0_L))) or 1.0
        pd_floor = 1e-3 * scale / D

        def objective(psi: np.ndarray) -> np.ndarray:
            theta = theta0_L + null_L @ psi
            L2_trial = matrix_of_theta(theta)
            eig = np.linalg.eigvalsh(L2_trial)
            pd_pen = np.maximum(pd_floor - eig, 0.0) / pd_floor
            try:
                R2_trial = np.linalg.inv(L2_trial.T) / D
            except np.linalg.LinAlgError:
                return np.full(2 * d * d + 1 + D, 1e6)
            flat = cond1_residual_matrix(R2_trial).flatten()
            bdef = bulk_defect(L2_trial, R2_trial, k_bulk)
            bdef = bdef if bdef > 0 else 0.0
            return np.concatenate([flat.real, flat.imag, [np.sqrt(bdef)], pd_pen])

        # Start inside the positive-definite part of the family. R2 =
        # (L2^T)^-1 / D diverges at det L2 = 0, a barrier the fit does not
        # cross, so a start on the indefinite side never reaches a valid
        # point. The most positive point maximizes the concave function
        # lambda_min(L2(psi)); Nelder-Mead from several starts finds it.
        def neg_lam_min(psi: np.ndarray) -> float:
            return -float(np.linalg.eigvalsh(matrix_of_theta(theta0_L + null_L @ psi)).min())

        rng = np.random.default_rng(seed)
        center, best_lam = np.zeros(K_null), -neg_lam_min(np.zeros(K_null))
        for k in range(4):
            start = rng.standard_normal(K_null) * scale * (0 if k == 0 else 1.0)
            res = minimize(
                neg_lam_min,
                start,
                method="Nelder-Mead",
                options=dict(xatol=1e-10 * scale, fatol=1e-12 * scale, maxiter=4000),
            )
            if -res.fun > best_lam:
                center, best_lam = res.x, -res.fun
        if best_lam <= tol:
            raise ValueError(
                "solve_qD_boundary: the affine family of L2 solving the right-boundary "
                f"condition has no positive-definite member (max lambda_min = {best_lam:.3e})."
            )
        spread = np.linalg.norm(center) + scale

        best_key, best_psi = None, None
        spreads = (0.0, 0.1, 0.5)
        for k in range(n_restarts):
            psi0 = center + rng.standard_normal(K_null) * spreads[k % len(spreads)] * spread
            while neg_lam_min(psi0) >= -tol:  # pull back inside the PD region
                psi0 = center + 0.5 * (psi0 - center)
            sol = least_squares(objective, psi0, method="trf", max_nfev=200)
            cost = float(np.sum(sol.fun**2))
            pd = np.linalg.eigvalsh(matrix_of_theta(theta0_L + null_L @ sol.x)).min() > tol
            key = (not pd, cost)
            if best_key is None or key < best_key:
                best_key, best_psi = key, sol.x
        # Polish the best restart to machine precision.
        sol = least_squares(
            objective, best_psi, method="trf", max_nfev=2000, ftol=1e-15, xtol=1e-15, gtol=1e-15
        )
        if np.sum(sol.fun**2) <= best_key[1] and neg_lam_min(sol.x) < -tol:
            best_psi = sol.x
        theta_star = theta0_L + null_L @ best_psi

    L2 = matrix_of_theta(theta_star)
    eigvals_L2 = np.linalg.eigvalsh(L2)
    if eigvals_L2.min() <= tol:
        raise ValueError(
            f"solve_qD_boundary: best L2 found is not positive-definite "
            f"(min eigenvalue {eigvals_L2.min():.3e} <= tol {tol:.1e}); the "
            "q=D ansatz requires L2 > 0. This tensor/boundary may not admit "
            "a solution in this family (the existence of a q = D pair is not guaranteed)."
        )
    R2 = np.linalg.inv(L2.T) / D

    cond2_residual = float(np.linalg.norm(np.einsum("ab,abjk->jk", L2, Gamma_C) - I_d))
    cond1_residual = float(np.linalg.norm(cond1_residual_matrix(R2)))
    q_trace = float(np.real(np.trace(L2 @ R2.T)))
    bulk_defect_at_fit = bulk_defect(L2, R2, k_bulk)
    bulk_defect_out_of_sample = bulk_defect(L2, R2, k_bulk + 1)

    # The bond-space defect is a squared Frobenius norm and is compared
    # against the tolerances in its own (squared) units.
    cond_worst = max(cond1_residual, cond2_residual)
    bulk = max(bulk_defect_out_of_sample, 0.0)
    if (cond_worst > _QD_REFINE_TOL or bulk > _QD_REFINE_TOL) and not _whitened:
        # Both the isometry defect and the fit are covariant under the bond-basis
        # change A -> G^-1 A G (L2 -> G^dag L2 G, R2 -> conj(G) R2 G^T, q
        # unchanged) but not equally well conditioned: a skewed basis leaves the
        # fit stuck at a residual of about 1e-6. In the basis G = L2^{-1/2} the
        # approximate solution is close to the identity, as in the natural basis of
        # the generator families, so the problem is solved once more there.
        W = matrix_sqrt_hermitian(L2)
        Gw = np.linalg.inv(W)
        A_w = np.einsum("ac,ijce,eb->ijab", W, A, Gw)
        try:
            L_w, R_w, _, diag_w = solve_qD_boundary(
                A_w, Gw.T @ l_vec, W @ r_vec, tol, n_restarts, seed, k_bulk, _whitened=True
            )
        except ValueError:
            pass
        else:
            P = W @ (L_w.conj().T @ L_w) @ W
            Q = Gw.conj() @ (R_w.conj().T @ R_w) @ Gw.T
            diag_w["whitened"] = True
            return matrix_sqrt_hermitian(P), matrix_sqrt_hermitian(Q), float(D), diag_w
    if cond_worst > _QD_RESIDUAL_TOL or bulk > _QD_BULK_DEFECT_TOL:
        raise ValueError(
            "solve_qD_boundary: the best positive-definite L2 leaves boundary residual "
            f"{cond_worst:.2e} (limit {_QD_RESIDUAL_TOL:.0e}) and bulk defect {bulk:.2e} "
            f"(limit {_QD_BULK_DEFECT_TOL:.0e}); no valid q=D pair found."
        )

    L = matrix_sqrt_hermitian(L2)
    R = matrix_sqrt_hermitian(R2)

    diagnostics = {
        "cond1_residual": cond1_residual,
        "cond2_residual": cond2_residual,
        "bulk_defect_at_fit": bulk_defect_at_fit,
        "bulk_defect_out_of_sample": bulk_defect_out_of_sample,
        "tr_L2_R2T": q_trace,
        "L2_min_eigenvalue": float(eigvals_L2.min()),
        "null_dim_L2_family": K_null,
    }
    return L, R, float(D), diagnostics


def _isometry_transfer_tensor(A: np.ndarray) -> np.ndarray:
    r"""Return the per-site transfer tensor of the bond-space isometry defect.

    .. math::

        G[j,j',a,c,d,b] = \sum_i A[i,j,c,a]^* A[i,j',d,b]

    The first physical leg :math:`i` of the tensor passed in is contracted;
    the second leg (:math:`j, j'`) stays free together with two copies of
    the doubled bond space. The site isometry maps the physical *input* leg
    to (output, external bonds), so callers pass ``A.transpose(1, 0, 2, 3)``:
    the output leg is then contracted and the input legs stay free, as in
    :math:`V^\dagger V`.

    Parameters
    ----------
    A : ndarray
        Tensor ``A[i, j, bond_L, bond_R]`` of shape ``(d, d, D_L, D_R)``.

    Returns
    -------
    ndarray
        Complex array of shape ``(d, d, D_R, D_L, D_L, D_R)``.

    Notes
    -----
    In :func:`_bondspace_isometry_defect`, ``left_mat[c, c']`` multiplies the
    conjugated copy's ``c`` (pass :math:`L^\dagger L`) and ``right_mat`` enters
    transposed (pass :math:`(R^\dagger R)^T`). The left and right bond
    dimensions of ``A`` are independent: the outer legs ``(a, b)`` have size
    ``D_R`` and the inner legs ``(c, d)`` have size ``D_L``. This allows the
    rectangular first and last sites produced by :func:`to_canonical_form`.
    """
    d, _, D_L, D_R = A.shape
    G = np.zeros((d, d, D_R, D_L, D_L, D_R), dtype=complex)
    for i in range(d):
        for j in range(d):
            for jp in range(d):
                G[j, jp] += np.einsum("ac,db->acdb", A[i, j].conj().T, A[i, jp])
    return G


def _bondspace_isometry_defect(G_list, d_list, left_mat, right_mat, all_prefixes: bool):
    r"""Return the squared isometry defect of a window of sites in bond space.

    The defect is :math:`\sum_n \|X_n - I\|_F^2` with
    :math:`X_n = V_n^\dagger V_n`, where :math:`V_n` is the isometry of the
    first :math:`n` sites dressed with ``left_mat`` on the left and
    ``right_mat`` on the right. The sum runs over every prefix length if
    ``all_prefixes`` is true, otherwise over the full window only.

    Parameters
    ----------
    G_list : list of jax.Array
        Per-site transfer tensors from :func:`_isometry_transfer_tensor`.
    d_list : list of int
        Physical dimension of each site.
    left_mat : jax.Array
        Left Gram matrix ``L^dagger L``.
    right_mat : jax.Array
        Transposed right Gram matrix ``(R^dagger R)^T``.
    all_prefixes : bool
        Include every prefix length ``n = 1, ..., len(G_list)``.

    Returns
    -------
    jax.Array
        Real scalar, the summed squared Frobenius defect.

    Notes
    -----
    The cost is linear in ``len(G_list)`` and polynomial in the bond
    dimension; no object of size :math:`\prod_k d_k` is formed.
    :math:`\operatorname{Tr}X_n` uses a single-copy ``(D, D)`` environment.
    :math:`\operatorname{Tr}X_n^2=\|X_n\|_F^2` uses a joint two-copy
    ``(D, D, D, D)`` recursion in which the second copy carries
    :math:`\overline{G_k}` at the same site-local output indices as the first
    copy, which reproduces the Hermiticity of :math:`X_n`. The defect is
    evaluated as :math:`\operatorname{Tr}X^2 - 2\operatorname{Tr}X + \dim`
    and therefore has a cancellation floor of about ``1e-12``.
    """
    env = left_mat
    S = jnp.einsum("ab,cd->abcd", left_mat, left_mat.conj())

    total_dim = 1
    total_defect = jnp.zeros((), dtype=left_mat.real.dtype)
    n_sites = len(G_list)
    for idx, (G_k, d_k) in enumerate(zip(G_list, d_list)):
        env = jnp.einsum("jjAabB,ab->AB", G_k, env)

        T1 = jnp.einsum("jqAabB,abcd->jqABcd", G_k, S)
        S = jnp.einsum("jqABcd,jqCcdD->ABCD", T1, G_k.conj())

        total_dim *= d_k
        if all_prefixes or idx == n_sites - 1:
            tr_X = jnp.trace(env @ right_mat)
            tr_X2 = jnp.einsum("abcd,ba,cd->", S, right_mat, right_mat)
            total_defect = total_defect + jnp.real(tr_X2) - 2.0 * jnp.real(tr_X) + total_dim

    return total_defect


def optimize_boundary_operators(
    A: np.ndarray | list[np.ndarray],
    k: int = 1,
    gauge: bool = False,
    l_vec: np.ndarray | None = None,
    r_vec: np.ndarray | None = None,
    n_restarts: int = 10,
    lam_iso: float = 1e6,
    maxiter: int = 2000,
    seed: int = 0,
    adaptive: bool = True,
    feas_tol: float = 1e-6,
    warm_start: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    r"""Optimize boundary operators :math:`L, R` to minimize :math:`q`.

    Minimizes the conditioning number

    .. math:: q(L,R) = \sqrt{\operatorname{tr}[R^{-2}(L^{-2})^T]}

    subject to the isometry constraint :math:`(V_n^{[L,R]})^\dagger
    V_n^{[L,R]} = I_{d^n}` for :math:`n = 1, \dots, k` (Styliaris et al.,
    end matter, Lemma 5). The operators are parametrized as
    :math:`L, R = \exp(H)` with Hermitian :math:`H`, which keeps them
    Hermitian positive definite (HPD) for any real parameter vector. The
    constraint is imposed as a quadratic penalty, with a bond-space
    transfer-matrix evaluation of the defect (see
    :func:`_bondspace_isometry_defect`); the penalty weight follows an
    adaptive schedule over ``n_restarts`` L-BFGS-B restarts (see Notes).

    Parameters
    ----------
    A : ndarray or list of ndarray
        Either one bulk tensor ``A[p_in, p_out, bond_L, bond_R]`` of shape
        ``(d, d, D, D)``, whose isometry constraint is imposed at every
        prefix length ``1, ..., k`` of its ``k``-fold repetition, or a list
        of ``m`` (possibly distinct) tensors with consistent bonds forming
        one window of a non-uniform chain. For a list, ``k`` is ignored and
        the constraint is imposed at the full window length ``m`` only.
    k : int, optional
        Largest support length of the constraint for a single tensor
        (default 1).
    gauge : bool, optional
        If true, a unitary gauge :math:`G=\exp(iH_G)` is optimized jointly
        and the cost and constraint are evaluated on
        :math:`(\Lambda,\Gamma)=(LG^{-1},GR)`, which is returned. This pair
        is generally not HPD, so the feasible set is larger (thesis, section
        "Gauge optimization and the :math:`q_{\rm unif}` orbit"). Requires
        equal left and right bond dimensions of the window (default false).
    l_vec, r_vec : ndarray, optional
        Boundary vectors of the chain. If given, the length-1 boundary
        isometries :math:`V^{[l,R]}_1 = \ell A R` and
        :math:`V^{[L,r]}_1 = L A r` are constrained as well, as the circuit
        synthesis requires. Without them a pair can satisfy the bulk
        constraint and still fail at the chain ends.
    n_restarts : int, optional
        Number of L-BFGS-B restarts (the landscape is non-convex; default 10).
    lam_iso : float, optional
        Initial penalty weight of the isometry violation (default ``1e6``).
    maxiter : int, optional
        Maximum L-BFGS-B iterations per restart (default 2000).
    seed : int, optional
        Seed of the restart perturbations (default 0); deterministic.
    adaptive : bool, optional
        If true (default), the penalty weight and the perturbation radius
        change between restarts (see Notes). If false, ``lam_iso`` is fixed
        and candidates are ranked by raw cost.
    feas_tol : float, optional
        Isometry residual (Frobenius norm) below which a restart counts as
        feasible for the adaptive schedule and the candidate selection
        (default ``1e-6``). Independent of the tolerance a caller applies to
        the returned ``iso_residual``.
    warm_start : tuple of ndarray, optional
        HPD pair ``(L0, R0)``, for instance the canonical boundary, from which
        restart 0 starts instead of the identity. It is converted to the
        ``exp(H)`` parametrization with the matrix logarithm and must be
        positive definite. Cold and warm starts explore different regions;
        a single call runs one schedule only.

    Returns
    -------
    L_opt : ndarray
        Optimized left boundary, shape ``(D_L, D_L)`` with ``D_L`` the left
        bond of the first tensor; gauge-dressed if ``gauge`` is true.
    R_opt : ndarray
        Optimized right boundary, shape ``(D_R, D_R)`` with ``D_R`` the right
        bond of the last tensor. ``D_L`` and ``D_R`` differ for windows at
        the end of a canonicalized chain, where the outer bond has dimension 1.
    q_opt : float
        ``conditioning_number(L_opt, R_opt)`` if ``D_L == D_R``. Otherwise
        :math:`q` is not defined (it needs both operators on one bond) and the
        value is the proxy ``sqrt(tr[R^-2])``, to be used only as search
        guidance; ``GeneralMPU`` computes the true same-bond ``q_b`` itself.
    iso_residual : float
        ``sqrt(sum_n ||V_n^dagger V_n - I||_F^2 + boundary defects)`` over
        the bulk lengths and the two boundary blocks.

    Raises
    ------
    ValueError
        If ``gauge`` is true and the window has ``D_L != D_R``.

    Notes
    -----
    **Adaptive schedule.** An infeasible restart (``iso > feas_tol``)
    multiplies the weight by 3 and shrinks the perturbation radius. A
    feasible restart that does not improve the best ``q`` halves the weight
    and enlarges the radius. An improving feasible restart changes nothing.
    The weight stays within a factor ``1e4`` of ``lam_iso``. The best
    candidate then receives one final polish at the largest weight.

    **Reading the result.** Because the constraint is a penalty,
    ``iso_residual`` must be inspected before ``q_opt`` is trusted.
    :math:`q\ge1` is not implied by the isometry constraint (for a bulk
    tensor with ``D = 1`` the constraint pins :math:`q` to a value that can
    be below 1) while the GOAA rotation angle :math:`\arcsin(1/q)` needs
    :math:`q\ge1`. :math:`q<1` is softly penalized, and callers must check
    ``q_opt >= 1`` (with tolerance) and fall back to the canonical
    construction otherwise.

    **Cost and conventions.** The cost uses the Frobenius form
    :math:`\|\Gamma^{-\top}\Lambda^{-1}\|_F` on the returned (dressed)
    pair, which equals the formula above for HPD pairs and is the LCU 1-norm
    of the merging operator built from the pair in general. The isometry
    defect uses the transposed right Gram matrix, so ``L_opt`` and ``R_opt``
    can be used as ``UniformMPU.L`` and ``UniformMPU.R`` without further
    transposition. Which isometries hold beyond the enforced lengths is
    validated numerically only.
    """
    jax.config.update("jax_platform_name", "cpu")
    fdtype, cdtype, np_fdtype = jnp.float64, jnp.complex128, np.float64

    # A single tensor is checked at every prefix length 1..k of its k-fold
    # repetition (UniformMPU); a list is a window of distinct tensors checked
    # at its full length only (GeneralMPU). The defect is evaluated by
    # bond-space recursions and never materializes a d^n x d^n object.
    check_all_prefixes = isinstance(A, np.ndarray)
    tensors_list = [A] * k if check_all_prefixes else list(A)
    # L dresses the window's own left bond and R its right bond; they differ
    # when to_canonical_form has absorbed the chain's boundary vectors,
    # leaving a dimension-1 outer bond at a chain end.
    D_L = tensors_list[0].shape[2]
    D_R = tensors_list[-1].shape[3]
    if gauge and D_L != D_R:
        raise ValueError(
            f"gauge=True requires a common bond dimension on both ends "
            f"(the same unitary G dresses both L and R via L@G^dagger, G@R), "
            f"but this window has D_L={D_L} != D_R={D_R}."
        )
    n_par_list = [D_L * D_L, D_R * D_R] + ([D_L * D_L] if gauge else [])

    # Site isometries map the physical INPUT leg to (output, external bonds):
    # transfer tensors of the leg-swapped tensors (see solve_qD_boundary).
    G_list = [
        jnp.array(_isometry_transfer_tensor(Ak.transpose(1, 0, 2, 3)), dtype=cdtype)
        for Ak in tensors_list
    ]
    d_list = [Ak.shape[0] for Ak in tensors_list]

    # l_vec/r_vec anchor the true chain boundary, so their isometry check
    # spans the tensors passed in: one site for a single repeated tensor,
    # the whole window for a list.
    boundary_G_list = G_list[:1] if check_all_prefixes else G_list
    boundary_d_list = d_list[:1] if check_all_prefixes else d_list

    # Cumulative offsets into the packed real parameter vector: each matrix
    # occupies 2*n_par_list[idx] reals (real part, then imaginary part).
    _offsets = np.concatenate(([0], np.cumsum([2 * n for n in n_par_list])))
    _dims = [D_L, D_R] + ([D_L] if gauge else [])  # matching side length per matrix

    def _herm_from_real(Yr, Yi):
        return (Yr + Yr.T) / 2.0 + 1j * (Yi - Yi.T) / 2.0

    def pack(mats: list[np.ndarray]) -> np.ndarray:
        p = np.empty(int(_offsets[-1]), dtype=np_fdtype)
        for idx, M in enumerate(mats):
            n = n_par_list[idx]
            s = _offsets[idx]
            p[s : s + n] = np.real(M).ravel()
            p[s + n : s + 2 * n] = np.imag(M).ravel()
        return p

    def unpack(p) -> list:
        p = p.astype(fdtype)
        out = []
        for idx in range(len(n_par_list)):
            n = n_par_list[idx]
            s = _offsets[idx]
            dim = _dims[idx]
            Yr = p[s : s + n].reshape(dim, dim)
            Yi = p[s + n : s + 2 * n].reshape(dim, dim)
            out.append(_herm_from_real(Yr, Yi).astype(cdtype))
        return out

    def _isometry_defect(L_iso, R_iso):
        """Return ``sum_n ||(L A^n R)^dagger (L A^n R) - I_{d^n}||_F^2``."""
        # create_local_isometries contracts the second axis of R with the
        # bond index, so R acts through its transpose relative to the
        # transfer tensors (which follow L's convention). Using R_iso.T makes
        # the returned L_opt, R_opt usable as UniformMPU.L / .R directly.
        R_pkg = R_iso.T
        LdagL = L_iso.conj().T @ L_iso
        RRdag = R_pkg @ R_pkg.conj().T
        return _bondspace_isometry_defect(G_list, d_list, LdagL, RRdag, check_all_prefixes)

    l_vec_jax = None if l_vec is None else jnp.array(l_vec, dtype=cdtype)
    r_vec_jax = None if r_vec is None else jnp.array(r_vec, dtype=cdtype)

    def _boundary_isometry_defect(L_iso, R_iso):
        r"""Return the squared isometry defect of the two boundary blocks.

        The defect is ``||V^dagger V - I||_F^2`` summed over the left block
        ``V^[l,R]`` and the right block ``V^[L,r]``, each term present only
        if the corresponding boundary vector is given. The blocks span the
        same tensors as ``boundary_G_list``: a single site for a uniform
        tensor, the whole window for a list. Longer support is produced by
        the tree-merge recursion, not by this optimizer.
        """
        defect = jnp.zeros((), dtype=fdtype)

        if l_vec_jax is not None:
            # R_iso.T: see the matching note in _isometry_defect.
            R_pkg = R_iso.T
            RRdag = R_pkg @ R_pkg.conj().T
            # left_mat[c, c'] multiplies conj(A[.., c, ..]) A[.., c', ..]:
            # conj(l_c) l_c', like L^dag L.
            l_outer = jnp.outer(l_vec_jax.conj(), l_vec_jax)
            defect = defect + _bondspace_isometry_defect(
                boundary_G_list, boundary_d_list, l_outer, RRdag, False
            )

        if r_vec_jax is not None:
            LdagL = L_iso.conj().T @ L_iso
            r_outer = jnp.outer(r_vec_jax, r_vec_jax.conj())
            defect = defect + _bondspace_isometry_defect(
                boundary_G_list, boundary_d_list, LdagL, r_outer, False
            )

        return defect

    def _cost_and_diagnostics(p, lam):
        mats = unpack(p)
        L = jax_expm(mats[0])
        R = jax_expm(mats[1])

        if gauge:
            Gmat = jax_expm(1j * mats[2])
            L_iso = L @ Gmat.conj().T
            R_iso = Gmat @ R
        else:
            L_iso, R_iso = L, R

        if D_L == D_R:
            # q^2 = ||R^{-T} L^{-1}||_F^2 of the pair the circuit is built from
            # (the dressed pair when gauge=True); equals tr[R^-2 (L^-2)^T] for HPD.
            K = jnp.linalg.inv(R_iso).T @ jnp.linalg.inv(L_iso)
            q2 = jnp.real(jnp.sum(K * K.conj())).astype(fdtype)
        else:
            Ri = jnp.linalg.inv(R)
            # tr[R^-2 (L^-2)^T] presumes L and R act on the same bond. When the
            # outer bonds differ (D_L = 1 for a window anchored at a
            # canonicalized chain end) no product combining them is defined,
            # so this scalar falls back to the conditioning of R alone. The
            # accept/reject decision uses the same-bond q_b computed by
            # GeneralMPU._compute_q_bond, not this proxy.
            q2 = jnp.real(jnp.trace(Ri @ Ri)).astype(fdtype)

        iso = _isometry_defect(L_iso, R_iso) + _boundary_isometry_defect(L_iso, R_iso)

        # q < 1 is reachable by the isometry constraint alone (for D = 1 it
        # is the only solution) but the GOAA rotation angle arcsin(1/q) is
        # undefined there. The soft penalty keeps the optimizer out of that
        # region unless the isometry term leaves no choice; callers must still
        # check q_opt >= 1 - tol (see Notes).
        q1_violation = jnp.clip(1.0 - jnp.sqrt(jnp.abs(q2)), 0.0, None) ** 2

        cost = q2 + lam * iso + lam * q1_violation
        # q2 above uses the Frobenius form, valid for the (generally non-HPD)
        # dressed pair; the HPD-only trace formula must not be applied to it.
        return cost, jnp.sqrt(jnp.abs(q2)), jnp.sqrt(jnp.abs(iso)), L, R, L_iso, R_iso

    # The penalty weight lam is a traced argument so that the adaptive
    # schedule can change it between restarts without recompilation; only
    # shapes and dtypes trigger a retrace.
    _jit_full = jax.jit(_cost_and_diagnostics)
    _jit_cost_grad = jax.jit(jax.value_and_grad(lambda p, lam: _cost_and_diagnostics(p, lam)[0]))

    def cost_and_grad_np(p, lam):
        val, g = _jit_cost_grad(jnp.array(p, dtype=fdtype), jnp.asarray(lam, dtype=fdtype))
        return float(val), np.array(g, dtype=np_fdtype)

    p_warm = np.zeros(int(_offsets[-1]), dtype=np_fdtype)
    _jit_full(jnp.array(p_warm, dtype=fdtype), jnp.asarray(lam_iso, dtype=fdtype))
    _jit_cost_grad(jnp.array(p_warm, dtype=fdtype), jnp.asarray(lam_iso, dtype=fdtype))

    mats0_warm = None
    if warm_start is not None:
        L0, R0 = warm_start
        # L, R = exp(H) with Hermitian H, so the matrix logarithm of an HPD
        # matrix places restart 0 exactly at (L0, R0). pack()/unpack()
        # symmetrize, which discards the floating-point asymmetry of logm.
        H0_L = la.logm(np.asarray(L0, dtype=complex))
        H0_R = la.logm(np.asarray(R0, dtype=complex))
        mats0_warm = [H0_L, H0_R] + ([np.zeros((D_L, D_L), dtype=complex)] if gauge else [])

    rng = np.random.default_rng(seed)

    # Bounds the adaptive schedule can wander within, relative to the
    # caller-supplied lam_iso (which remains the schedule's starting point).
    lam_lo = max(float(lam_iso) / 1e4, 1.0)
    lam_hi = float(lam_iso) * 1e4
    cur_lam = float(lam_iso)
    cur_scale_mult = 1.0

    best_feasible_q, best_feasible_p = np.inf, None
    best_any_cost, best_any_p = np.inf, None

    for restart in range(n_restarts):
        if restart == 0 and mats0_warm is not None:
            mats0 = mats0_warm
        elif restart == 0:
            mats0 = [np.zeros((dim, dim), dtype=complex) for dim in _dims]
        else:
            scale = (0.1 + 0.4 * (restart / n_restarts)) * cur_scale_mult
            mats0 = []
            for dim in _dims:
                Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
                mats0.append((Z + Z.conj().T) * scale / 2.0)

        p0 = pack(mats0)
        lam_used = cur_lam if adaptive else float(lam_iso)
        res = minimize(
            lambda p, lam=lam_used: cost_and_grad_np(p, lam),
            p0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": 1e-15,
                "gtol": 1e-10,
                "maxcor": 50,
            },
        )

        if res.fun < best_any_cost:
            best_any_cost = res.fun
            best_any_p = res.x.copy()

        if adaptive:
            _, q_i, iso_i, _, _, _, _ = _jit_full(jnp.array(res.x, dtype=fdtype), lam_used)
            q_i, iso_i = float(q_i), float(iso_i)
            is_feasible = iso_i < feas_tol
            improved = is_feasible and q_i < best_feasible_q - 1e-9

            if is_feasible and (best_feasible_p is None or q_i < best_feasible_q):
                best_feasible_q = q_i
                best_feasible_p = res.x.copy()

            if restart < n_restarts - 1:
                if not is_feasible:
                    # Infeasible: tighten the penalty, take smaller steps.
                    cur_lam = min(cur_lam * 3.0, lam_hi)
                    cur_scale_mult = max(cur_scale_mult * 0.7, 0.2)
                elif not improved:
                    # Feasible but not better than the best q so far: the
                    # penalty is likely limiting q, so loosen it and widen
                    # the perturbation.
                    cur_lam = max(cur_lam / 2.0, lam_lo)
                    cur_scale_mult = min(cur_scale_mult * 1.5, 5.0)
                # else: feasible and improved -> stay in this regime.

    best_p = best_feasible_p if best_feasible_p is not None else best_any_p

    if adaptive:
        # The schedule may end on a loosened penalty. Finish with one
        # high-penalty refinement from the best point so that the returned
        # candidate is as feasible as a fixed-penalty run would be; keep it
        # only if it stays feasible (or nothing feasible existed).
        polish_lam = lam_hi
        res_polish = minimize(
            lambda p: cost_and_grad_np(p, polish_lam),
            best_p,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": 1e-15,
                "gtol": 1e-10,
                "maxcor": 50,
            },
        )
        _, _, iso_polish, _, _, _, _ = _jit_full(jnp.array(res_polish.x, dtype=fdtype), polish_lam)
        if float(iso_polish) < feas_tol or best_feasible_p is None:
            best_p = res_polish.x

    _, _, _, L_hpd_jax, R_hpd_jax, L_opt_jax, R_opt_jax = _jit_full(
        jnp.array(best_p, dtype=fdtype), jnp.asarray(lam_iso, dtype=fdtype)
    )
    L_opt = np.array(L_opt_jax)
    R_opt = np.array(R_opt_jax)

    # q of the returned pair (the gauge-dressed one when gauge=True), i.e. the
    # LCU 1-norm of the merging operator a circuit built from it uses. For
    # D_L != D_R only the proxy sqrt(tr R^-2) is available (see Notes).
    if D_L == D_R:
        q_opt = conditioning_number(L_opt, R_opt)
    else:
        Ri_f = np.linalg.inv(np.array(R_hpd_jax))
        q_opt = float(np.sqrt(max(float(np.real(np.trace(Ri_f @ Ri_f))), 0.0)))  # see note above

    iso_residual = float(
        np.sqrt(
            np.abs(
                _isometry_defect(L_opt_jax, R_opt_jax)
                + _boundary_isometry_defect(L_opt_jax, R_opt_jax)
            )
        )
    )

    return L_opt, R_opt, q_opt, iso_residual


def get_mpo_site_tensors(k: int, A_np: np.ndarray) -> tuple[qtn.Tensor, qtn.Tensor]:
    """Return ket and bra tensors for site ``k`` of a uniform MPU.

    Parameters
    ----------
    k : int
        Site label used to name the indices.
    A_np : ndarray
        Local tensor ``A[p_in, p_out, bond_L, bond_R]`` of shape
        ``(d_in, d_out, D_L, D_R)``.

    Returns
    -------
    A_k : quimb.tensor.Tensor
        Ket tensor with indices ``p_in_k, p_out_k, bond_{k-1}, bond_k``.
    A_k_dag : quimb.tensor.Tensor
        Conjugate tensor with indices ``p_out_f_k, p_out_k, bond_{k-1}_dag,
        bond_k_dag``.

    Raises
    ------
    ValueError
        If ``A_np`` does not have four axes.

    Notes
    -----
    ``A_k_dag`` shares the label ``p_out_k`` with the ket, so contracting the
    two traces the output leg and leaves the ket and bra input legs
    (``p_in_k`` and ``p_out_f_k``) free, as in ``U^dagger U``-type networks.
    """
    if A_np.ndim != 4:
        raise ValueError(
            f"A_np must have four axes: (p_in, p_out, bond_L, bond_R). Got shape {A_np.shape}."
        )

    A_k = qtn.Tensor(
        A_np,
        inds=[
            f"p_in_{k}",
            f"p_out_{k}",
            f"bond_{k - 1}",
            f"bond_{k}",
        ],
        tags=[f"A_{k}"],
    )

    A_k_dag = qtn.Tensor(
        np.conj(A_np),
        inds=[
            f"p_out_f_{k}",
            f"p_out_{k}",
            f"bond_{k - 1}_dag",
            f"bond_{k}_dag",
        ],
        tags=[f"A_{k}_dag"],
    )

    return A_k, A_k_dag


def matrix_sqrt_hermitian(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return the square root of a Hermitian positive-semidefinite matrix.

    Parameters
    ----------
    M : ndarray
        Hermitian matrix of shape ``(n, n)``.
    tol : float, optional
        Eigenvalues down to ``-tol`` are treated as numerical noise and set
        to zero (default ``1e-12``).

    Returns
    -------
    ndarray
        The Hermitian positive-semidefinite square root, shape ``(n, n)``.

    Raises
    ------
    ValueError
        If an eigenvalue is below ``-tol``.
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    if np.any(eigvals < -tol):
        raise ValueError(
            f"Matrix is not positive-semidefinite: min eigenvalue = {eigvals.min():.2e}"
        )
    eigvals = np.maximum(eigvals, 0.0)  # remove rounding-level negative eigenvalues
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def conditioning_number(L: np.ndarray, R: np.ndarray) -> float:
    r"""Return the conditioning number :math:`q(L,R)=\|R^{-\top}L^{-1}\|_F`.

    This is the LCU 1-norm :math:`\|M\|_1` of the rank-one merging operator
    built from ``(L, R)`` (see :func:`get_merging_operator`), and hence the
    quantity that fixes the number :math:`\ell` of amplitude-amplification
    rounds. For Hermitian positive-definite :math:`L, R` it equals
    :math:`\sqrt{\operatorname{tr}[R^{-2}(L^{-2})^\top]}` (thesis
    ``eq:q_unif_LR``); the Frobenius form holds for any invertible pair,
    including gauge-dressed ones.

    Parameters
    ----------
    L, R : ndarray
        Invertible boundary operators ``L[ext, bond]``, ``R[ext, bond]``
        of the same square shape.

    Returns
    -------
    float
        :math:`q`; it is at least 1 for admissible boundaries (thesis).
    """
    return float(np.linalg.norm(np.linalg.inv(R).T @ np.linalg.inv(L)))


def get_merging_operator(L_inv: np.ndarray, R_inv: np.ndarray) -> np.ndarray:
    r"""Return the merging operator of a boundary pair as a dense matrix.

    The operator is :math:`M=|00\rangle\langle I|(R^{-1}\otimes L^{-1})`
    (thesis ``eq:merging_operator``). Merging two adjacent
    blocks inserts :math:`K=R^{-\top}L^{-1}` between the right boundary
    operator of the left block and the left boundary operator of the right
    block, restoring the original bond contraction. As a map on the two
    external bond registers it has rank one; only the ``(0, 0)`` output
    block is nonzero.

    Parameters
    ----------
    L_inv, R_inv : ndarray
        Inverses of the boundary operators ``L[ext, bond]`` and
        ``R[ext, bond]``, square and of the same dimension :math:`\chi`.

    Returns
    -------
    ndarray
        Complex :math:`\chi^2\times\chi^2` matrix with axes ``(ext_R, ext_L)``
        whose only nonzero row is ``vec(K)``, so that
        :math:`\|M\|_1=\|K\|_F=q`.

    Raises
    ------
    ValueError
        If the inputs are not square or have different dimensions.
    """
    # Input checks.
    if L_inv.shape[0] != L_inv.shape[1] or R_inv.shape[0] != R_inv.shape[1]:
        raise ValueError("L_inv and R_inv must be square matrices.")

    if L_inv.shape != R_inv.shape:
        raise ValueError("L_inv and R_inv must have the same dimension.")

    # Only the (0, 0) block is filled: M = |00><I| (R^-1 x L^-1).
    D = L_inv.shape[0]
    M = np.zeros((D, D, D, D), dtype=complex)
    M[0, 0, :, :] = R_inv.T @ L_inv  # kernel, (D, D)
    return M.reshape(D**2, D**2)


def dilate_isometry_to_unitary(V_mat: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Dilate an isometry to a unitary.

    The complementary columns are an orthonormal basis of the null space of
    ``V^dagger`` (``scipy.linalg.null_space``).

    Parameters
    ----------
    V_mat : ndarray
        Matrix of shape ``(m, n)`` with ``m >= n`` within ``tol`` (Frobenius
        norm of ``V^dagger V - I_n``) of an isometry.
    tol : float, optional
        Acceptance tolerance on ``||V^dagger V - I_n||_F`` (default
        ``1e-10``). Canonical boundaries satisfy it to machine precision;
        boundaries from a numerical optimizer
        (:func:`optimize_boundary_operators`) generally do not, so the
        default is not machine epsilon.

    Returns
    -------
    ndarray
        Unitary of shape ``(m, m)``. Its first ``n`` columns are the exact
        isometry closest to ``V_mat`` in Frobenius norm (polar factor
        ``U @ Vh`` of the SVD), not ``V_mat`` itself: inputs within ``tol``
        are snapped onto the isometry manifold so that the residual does not
        accumulate in the tree-merge recursion.

    Raises
    ------
    ValueError
        If ``m < n`` or ``V_mat`` is farther than ``tol`` from an isometry.
    """
    m, n = V_mat.shape

    if m < n:
        raise ValueError(f"V_mat must have at least as many rows as columns. Got ({m}, {n}).")

    defect = np.linalg.norm(V_mat.conj().T @ V_mat - np.eye(n))
    if defect > tol:
        raise ValueError(
            f"V_mat is not a valid isometry: ||V†V - I||_F = {defect:.3e} > tol={tol:.1e}."
        )

    # Snap to the nearest exact isometry (Procrustes / polar decomposition),
    # so the returned unitary is exact regardless of how close V_mat was.
    U_svd, _, Vh_svd = la.svd(V_mat, full_matrices=False)
    V_mat = U_svd @ Vh_svd

    if m == n:
        return V_mat.copy()

    # Orthogonal complement = null space of V†, computed via SVD
    tail = la.null_space(V_mat.conj().T)  # shape (m, m-n)
    V_tilde = np.hstack((V_mat, tail))  # shape (m, m)

    return V_tilde


def compute_lcu_pad_indices(D_bond: int, D_bond_pad: int, dim_rot: int = 2) -> np.ndarray:
    """Return the map from unpadded to padded bond-register indices.

    The raw space has axes ``(ext_R, ext_L, rot)`` of sizes
    ``(D_bond, D_bond, dim_rot)`` in C order (``rot`` fastest). The padded
    space has the same axis order with ``ext_L`` padded to ``D_bond_pad``
    (Qiskit little-endian register layout), so the flat index of
    ``(i_R, i_L, rot)`` is::

        idx_pad = rot + i_L * dim_rot + i_R * dim_rot * D_bond_pad

    Parameters
    ----------
    D_bond : int
        Unpadded bond dimension.
    D_bond_pad : int
        Padded bond dimension, at least ``D_bond`` (typically the next power
        of 2).
    dim_rot : int, optional
        Dimension of the rotation register: 2 for one rotation qubit, 1 when
        none is used (closed-form route). Default 2.

    Returns
    -------
    ndarray
        Integer array of shape ``(D_bond * D_bond * dim_rot,)``; entry ``k``
        is the padded flat index of the ``k``-th raw basis vector.

    Raises
    ------
    ValueError
        If ``D_bond_pad < D_bond``.
    """
    if D_bond_pad < D_bond:
        raise ValueError(f"D_bond_pad ({D_bond_pad}) must be >= D_bond ({D_bond}).")

    i_R = np.arange(D_bond)[:, None, None]
    i_L = np.arange(D_bond)[None, :, None]
    rot = np.arange(dim_rot)[None, None, :]

    pad_indices = (rot + i_L * dim_rot + i_R * dim_rot * D_bond_pad).ravel()

    return pad_indices.astype(np.intp, copy=False)


def pad_block_diagonal_operator(
    W_ctrl: np.ndarray,
    D_bond: int,
    D_bond_pad: int,
    dim_rot: int = 2,
) -> np.ndarray:
    """Embed a block-diagonal controlled operator into the padded bond space.

    Parameters
    ----------
    W_ctrl : ndarray
        Controlled unitary of shape ``(dim_lcu_A * dim_W_raw,) * 2`` with
        ``dim_W_raw = D_bond**2 * dim_rot``, built on the unpadded space as
        ``dim_lcu_A`` diagonal blocks.
    D_bond : int
        Unpadded bond dimension.
    D_bond_pad : int
        Padded bond dimension (next power of 2 at least ``D_bond``).
    dim_rot : int, optional
        Dimension of the rotation register (default 2).

    Returns
    -------
    ndarray
        Complex matrix of shape ``(dim_lcu_A * dim_W_pad,) * 2`` with
        ``dim_W_pad = D_bond_pad**2 * dim_rot``. Each block is placed at the
        indices given by :func:`compute_lcu_pad_indices`; all other basis
        states are left fixed (identity).

    Raises
    ------
    ValueError
        If the number of rows of ``W_ctrl`` is not a multiple of
        ``dim_W_raw``.
    """
    dim_W_raw = (D_bond**2) * dim_rot
    dim_W_pad = (D_bond_pad**2) * dim_rot

    if W_ctrl.shape[0] % dim_W_raw != 0:
        raise ValueError(
            f"W_ctrl rows ({W_ctrl.shape[0]}) not divisible by dim_W_raw ({dim_W_raw}). "
            f"Check D_bond={D_bond}, dim_rot={dim_rot}."
        )

    dim_lcu_A = W_ctrl.shape[0] // dim_W_raw
    pad_indices = compute_lcu_pad_indices(D_bond, D_bond_pad, dim_rot)

    W_ctrl_full = np.eye(dim_lcu_A * dim_W_pad, dtype=complex)

    for k in range(dim_lcu_A):
        offset_raw = k * dim_W_raw
        offset_pad = k * dim_W_pad

        raw_block = W_ctrl[
            offset_raw : offset_raw + dim_W_raw,
            offset_raw : offset_raw + dim_W_raw,
        ]

        target_indices = offset_pad + pad_indices
        W_ctrl_full[np.ix_(target_indices, target_indices)] = raw_block

    return W_ctrl_full


def block_mpu_tensors(tensors: list[np.ndarray], block_size: int) -> list[np.ndarray]:
    r"""Contract every ``block_size`` consecutive tensors into one.

    The effective physical dimension becomes ``d**block_size`` while the bond
    dimension is unchanged. Input and output tensors both use
    ``A[p_in, p_out, bond_L, bond_R]``.

    Parameters
    ----------
    tensors : list of ndarray
        Site tensors; the length must be divisible by ``block_size``.
    block_size : int
        Number of sites per block, at least 1 (1 returns complex copies).

    Returns
    -------
    list of ndarray
        ``len(tensors) // block_size`` blocked tensors of dtype complex.

    Raises
    ------
    ValueError
        If ``block_size < 1`` or the chain length is not divisible by it.

    Notes
    -----
    Physical legs are flattened with the *first* site of each block most
    significant. This is the opposite of the dense and Qiskit convention used
    elsewhere (site 0 least significant): the blocked chain describes the
    same unitary only after reversing the order of the sites inside each
    block. To compare ``dense_chain(block_mpu_tensors(A, b))`` with
    ``dense_chain(A)``, permute the ``b`` digits of each block.

    Blocking can restore full-rank boundary operators. The left environment
    at the first cut is a Gram matrix of only ``d**2`` vectors, so a chain
    with :math:`\chi > d^2` is rank deficient there, whereas a block of ``m``
    sites supplies ``d**(2m)`` of them. This matters for the canonical
    ``UniformMPU`` boundaries (``check_assumption_1``) and for
    ``GeneralMPU(canonicalize=False)``; the default ``GeneralMPU``
    canonicalizes and does not need it.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}.")
    if block_size == 1:
        return [np.asarray(A, dtype=complex) for A in tensors]
    if len(tensors) % block_size:
        raise ValueError(
            f"Chain length {len(tensors)} is not divisible by block_size {block_size}."
        )

    blocked = []
    for start in range(0, len(tensors), block_size):
        A = np.asarray(tensors[start], dtype=complex)
        for B in tensors[start + 1 : start + block_size]:
            A = np.einsum("ijac,klcb->ikjlab", A, np.asarray(B, dtype=complex))
            s = A.shape
            A = A.reshape(s[0] * s[1], s[2] * s[3], s[4], s[5])
        blocked.append(A)
    return blocked


def to_canonical_form(
    tensors: list[np.ndarray],
    l_vec: np.ndarray,
    r_vec: np.ndarray,
    tol: float = 1e-10,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    r"""Bring a chain into the canonical form (Definition 3).

    The canonical form (Definition 3 of Styliaris et al.) is the MPS
    canonical form of the Choi state of the MPU, obtained by successive SVDs over the paired physical index
    :math:`(i,j)`. It makes Assumption 1 unnecessary for non-uniform chains:
    the bond dimension at every cut is minimal, and maximally mixed
    :math:`\sigma,\tau` give

    .. math::

        L_k = \mathbb 1, \qquad R_k = \mathrm{diag}(s_{k,1}, \dots, s_{k,D_k}),

    with :math:`s_{k,i}>0` the Choi-state Schmidt values (Eq. 26 / A8 of the
    reference). Rank deficiency of a raw chain therefore reflects a
    non-minimal bond dimension, which canonicalization removes.

    Parameters
    ----------
    tensors : list of ndarray
        Site tensors ``A[p_in, p_out, bond_L, bond_R]``; at least two.
    l_vec, r_vec : ndarray
        Boundary vectors, absorbed into the first and last tensor.
    tol : float, optional
        Relative threshold (to ``max(1, s_max)``) below which a singular
        value counts as zero and its bond direction is dropped (default
        ``1e-10``).

    Returns
    -------
    tensors : list of ndarray
        Canonical chain with minimal bond dimensions and trivial
        (dimension-1) outer legs.
    l_vec, r_vec : ndarray
        The trivial boundary vectors ``[1]``.

    Raises
    ------
    ValueError
        If fewer than two tensors are given.

    Notes
    -----
    A right-to-left sweep runs first and a left-to-right sweep second. One
    sweep alone orthonormalizes the environment on its own side but leaves
    the rank seen from the other side untouched, so the minimal dimension of
    a cut, the smaller of the two, is reached only after both sweeps. The
    sweep that runs last determines the surviving gauge, so the left sweep
    runs second to obtain :math:`L_k=\mathbb 1`.
    """
    T = [np.asarray(A, dtype=complex) for A in tensors]
    if len(T) < 2:
        raise ValueError("to_canonical_form requires at least 2 tensors.")

    T[0] = np.tensordot(np.asarray(l_vec, dtype=complex), T[0], axes=(0, 2))[:, :, None, :]
    T[-1] = np.tensordot(T[-1], np.asarray(r_vec, dtype=complex), axes=(3, 0))[:, :, :, None]
    N = len(T)

    for k in range(N - 1, 0, -1):  # right -> left (first)
        di, dj, DL, DR = T[k].shape
        U, S, Vh = np.linalg.svd(
            T[k].transpose(2, 0, 1, 3).reshape(DL, di * dj * DR), full_matrices=False
        )
        keep = S > tol * max(1.0, float(S.max()) if S.size else 1.0)
        U, S, Vh = U[:, keep], S[keep], Vh[keep]
        T[k] = Vh.reshape(-1, di, dj, DR).transpose(1, 2, 0, 3)
        T[k - 1] = np.einsum("ijax,xy->ijay", T[k - 1], U @ np.diag(S))

    for k in range(N - 1):  # left -> right (last: its gauge is the one kept)
        di, dj, DL, DR = T[k].shape
        U, S, Vh = np.linalg.svd(
            T[k].transpose(2, 0, 1, 3).reshape(DL * di * dj, DR), full_matrices=False
        )
        keep = S > tol * max(1.0, float(S.max()) if S.size else 1.0)
        U, S, Vh = U[:, keep], S[keep], Vh[keep]
        T[k] = U.reshape(DL, di, dj, -1).transpose(1, 2, 0, 3)
        T[k + 1] = np.einsum("xy,ijyb->ijxb", np.diag(S) @ Vh, T[k + 1])

    one = np.ones(1, dtype=complex)
    return T, one, one


def satisfies_assumption_1(
    tensors: list[np.ndarray],
    l_vec: np.ndarray,
    r_vec: np.ndarray,
    tol: float = 1e-12,
) -> bool:
    """Return whether every cut of a chain has full-rank boundary operators.

    This is the condition :class:`~TreeMPU.MPU.GeneralMPU` needs with
    ``canonicalize=False`` and that the ``blocking="auto"`` option of
    ``mpu_generators`` searches for. The operators are computed by
    :func:`compute_chain_boundary_operators`.

    Parameters
    ----------
    tensors : list of ndarray
        Site tensors ``A[p_in, p_out, bond_L, bond_R]``.
    l_vec, r_vec : ndarray
        Boundary vectors.
    tol : float, optional
        Smallest singular value that counts as nonzero (default ``1e-12``).

    Returns
    -------
    bool
        False if the operators cannot be built (fewer than two tensors) or
        any ``L2`` or ``R2`` has a singular value at most ``tol``.
    """
    try:
        L2_list, R2_list, _ = compute_chain_boundary_operators(tensors, l_vec, r_vec)
    except ValueError:
        return False
    for L2, R2 in zip(L2_list, R2_list):
        if np.linalg.svd(L2, compute_uv=False).min() <= tol:
            return False
        if np.linalg.svd(R2, compute_uv=False).min() <= tol:
            return False
    return True


def compute_chain_boundary_operators(
    tensors: list[np.ndarray], l_vec: np.ndarray, r_vec: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], complex]:
    r"""Return the left and right environment operators of every bond.

    ``tensors[k]`` has shape ``(d, d, D_left_k, D_right_k)`` for
    ``k = 0, ..., N-1``, with ``D_left_0 == len(l_vec)`` and
    ``D_right_{N-1} == len(r_vec)``; all sites share the physical dimension
    ``d``. Each physical input leg is closed with :math:`I/d`
    and the output leg is traced, in two sweeps of :math:`O(N)` contractions
    (Eq. 22-23 and Lemma 1 of Styliaris et al.).

    Parameters
    ----------
    tensors : list of ndarray
        Site tensors ``A[p_in, p_out, bond_L, bond_R]``.
    l_vec, r_vec : ndarray
        Boundary vectors of shape ``(D_left_0,)`` and ``(D_right_{N-1},)``.

    Returns
    -------
    L2_list : list of ndarray
        ``L2_list[b]`` is :math:`L_{b+1}^2`, shape
        ``(D_right_b, D_right_b)``: the left cap of a segment starting at
        site ``b + 1``, for ``b = 0, ..., N-2``.
    R2_list : list of ndarray
        ``R2_list[b]`` is :math:`R_b^2`, shape ``(D_right_b, D_right_b)``:
        the right cap of a segment ending at site ``b``, for
        ``b = 0, ..., N-2``.
    unitarity_residual : complex
        :math:`\operatorname{Tr}(U^\dagger U)/d^N`. It equals 1 for every
        unitary chain, but a value of 1 does not imply unitarity (a
        normalization check only; see
        :func:`~TreeMPU.checks.check_chain_unitarity`).

    Raises
    ------
    ValueError
        If the chain has fewer than two tensors.
    """
    N = len(tensors)
    if N < 2:
        raise ValueError("compute_chain_boundary_operators requires N >= 2.")

    d = tensors[0].shape[0]

    # --- Left sweep: env_L after site k is L_{k+1}^2 (bond k) ---
    L2_list: list[np.ndarray] = [None] * (N - 1)  # type: ignore[list-item]

    A0 = np.tensordot(l_vec, tensors[0], axes=(0, 2))  # (d, d, D_right_0)
    # Environments in the convention of L[ext, bond] / R[ext, bond] (the
    # conjugated factor comes first); see UniformMPU._compute_boundary_operators.
    env_L = np.einsum("ijc,ijd->cd", A0.conj(), A0) / d
    L2_list[0] = env_L

    for k in range(1, N - 1):
        Ak = tensors[k]  # (d, d, D_left_k, D_right_k)
        env_L = np.einsum("ijac,ab,ijbd->cd", Ak.conj(), env_L, Ak) / d
        L2_list[k] = env_L

    # --- Right sweep: env_R after site k is R_{k-1}^2 (bond k-1) ---
    R2_list: list[np.ndarray] = [None] * (N - 1)  # type: ignore[list-item]

    A_last = np.tensordot(tensors[N - 1], r_vec, axes=(3, 0))  # (d, d, D_left_{N-1})
    env_R = np.einsum("ijm,ijn->mn", A_last.conj(), A_last) / d
    R2_list[N - 2] = env_R

    for k in range(N - 2, 0, -1):
        Ak = tensors[k]  # (d, d, D_left_k, D_right_k)
        env_R = np.einsum("ijac,cd,ijbd->ab", Ak.conj(), env_R, Ak) / d
        R2_list[k - 1] = env_R

    # --- Global closure: continue the left sweep through the last site ---
    unitarity_residual = np.einsum("ijm,mn,ijn->", A_last.conj(), L2_list[N - 2], A_last) / d

    return L2_list, R2_list, complex(unitarity_residual)


def optimize_chain_sigma_tau(
    tensors: list[np.ndarray],
    l_vec: np.ndarray,
    r_vec: np.ndarray,
    max_free_sites: int | None = None,
    eps_reg: float = 1e-8,
    n_restarts: int = 4,
    maxiter: int = 2000,
    beta: float = 50.0,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    r"""Optimize the per-site density matrices of a chain boundary.

    Generalizes :func:`optimize_q_unif` from one repeated tensor to a
    (possibly non-uniform) chain; it is the ``GeneralMPU`` analogue of the
    ``boundary_sigma_tau`` option of ``UniformMPU`` (thesis, section
    "Partial optimization").

    :func:`compute_chain_boundary_operators` closes the physical legs of each
    site with :math:`I/d`. Here each site within a window of
    ``max_free_sites`` sites nearest the corresponding chain end carries a
    free density matrix, :math:`\sigma_k` in the left sweep and
    :math:`\tau_k` in the right sweep, parametrized as
    :math:`XX^\dagger/\operatorname{tr}(XX^\dagger)`. Sites outside the
    window keep :math:`I/d`. All free matrices are optimized jointly with
    L-BFGS-B against a log-sum-exp smoothing of :math:`\max_b q_b` over all
    bonds, because the maximum determines the circuit depth. The sweep costs
    :math:`O(N)` regardless of the window size; since the sweeps propagate a
    running environment, a window at one end can improve every bond.

    Parameters
    ----------
    tensors : list of ndarray
        Site tensors ``A[p_in, p_out, bond_L, bond_R]``, as in
        :func:`compute_chain_boundary_operators`.
    l_vec, r_vec : ndarray
        Boundary vectors.
    max_free_sites : int, optional
        Number of sites nearest each chain end whose density matrix is free;
        ``None`` (default) frees every site, 0 returns the :math:`I/d`
        baseline without optimization. Values are clipped to ``[0, N-1]``.
    eps_reg : float, optional
        Tikhonov term added before each inversion in the loss (default
        ``1e-8``).
    n_restarts : int, optional
        Number of L-BFGS-B restarts (default 4); the restart with the smallest
        true :math:`\max_b q_b` is kept.
    maxiter : int, optional
        Maximum L-BFGS-B iterations per restart (default 2000).
    beta : float, optional
        Sharpness of the log-sum-exp smoothing (default 50); larger values
        approach the true maximum with a harder landscape.
    seed : int, optional
        Seed of the restart perturbations (default 0); deterministic.

    Returns
    -------
    L2_list, R2_list : list of ndarray
        Same shapes and meaning as the first two outputs of
        :func:`compute_chain_boundary_operators`, built from the optimized
        density matrices and conjugated to the ``L[ext, bond]`` convention.
    q_achieved : float
        True (not smoothed) :math:`\max_b q_b` of the best restart, computed
        with the ``eps_reg`` Tikhonov term of the loss. Compare
        it with the per-bond :math:`q_b` of the :math:`I/d` baseline before
        adopting the result: a lower maximum does not imply that every bond
        improved.

    Raises
    ------
    ValueError
        If the chain has fewer than two tensors.

    Notes
    -----
    The density matrices act on the physical *input* leg with the output leg
    traced. Every outcome is then an admissible boundary and the isometry
    property holds at every block length, because a per-bond mix of
    input-leg environments of disjoint regions is an environment pair; no
    isometry constraint is imposed in the optimization and the worst case is
    no improvement over :math:`I/d`. The output-leg variant is equivalent only under a
    no-signalling condition on :math:`U^\dagger`, which fails for example for
    the cyclic shift, and is not used.
    """
    jax.config.update("jax_platform_name", "cpu")
    fdtype, cdtype, np_fdtype = jnp.float64, jnp.complex128, np.float64

    N = len(tensors)
    if N < 2:
        raise ValueError("optimize_chain_sigma_tau requires N >= 2.")
    d = tensors[0].shape[0]

    w_L = (N - 1) if max_free_sites is None else max(0, min(max_free_sites, N - 1))
    w_R = (N - 1) if max_free_sites is None else max(0, min(max_free_sites, N - 1))

    T_j = [jnp.asarray(Ak, dtype=cdtype) for Ak in tensors]
    l_vec_j = jnp.asarray(l_vec, dtype=cdtype)
    r_vec_j = jnp.asarray(r_vec, dtype=cdtype)
    I_over_d = jnp.eye(d, dtype=cdtype) / d

    n_par = 2 * d * d  # one density matrix packs into this many reals
    sigma_span = w_L * n_par
    n_total = sigma_span + w_R * n_par

    def vec_to_rho(p_block):
        X = p_block[: d * d].reshape((d, d)) + 1j * p_block[d * d :].reshape((d, d))
        rho = X @ X.conj().T
        return rho / jnp.trace(rho).real

    def _step_L(sigma, T, env):
        # sigma acts on the INPUT physical leg (T[i, o] = A[p_in, p_out]) with
        # the output traced, as in optimize_q_unif: the site isometries map
        # p_in -> (p_out, ext). env replaces the rank-1 l_vec seed after the
        # first site.
        return jnp.einsum("ij,iopq,pm,jomn->qn", sigma, T, env, T.conj())

    def _step_R(tau, T, env):
        # tau on the input physical leg, as in _step_L.
        return jnp.einsum("ij,iopq,qn,jomn->pm", tau, T, env, T.conj())

    def _forward(p):
        sigmas = [vec_to_rho(p[k * n_par : (k + 1) * n_par]) for k in range(w_L)]
        taus = [
            vec_to_rho(p[sigma_span + k * n_par : sigma_span + (k + 1) * n_par]) for k in range(w_R)
        ]

        env = jnp.outer(l_vec_j, l_vec_j.conj())
        L2 = []
        for k in range(N - 1):
            sigma_k = sigmas[k] if k < w_L else I_over_d
            env = _step_L(sigma_k, T_j[k], env)
            L2.append(env)

        env = jnp.outer(r_vec_j, r_vec_j.conj())
        R2 = [None] * (N - 1)
        for k in range(N - 1, 0, -1):
            idx_from_right = (N - 1) - k
            tau_k = taus[idx_from_right] if idx_from_right < w_R else I_over_d
            env = _step_R(tau_k, T_j[k], env)
            R2[k - 1] = env

        q_bonds = []
        for b in range(N - 1):
            D_b = L2[b].shape[0]
            I_reg = eps_reg * jnp.eye(D_b, dtype=cdtype)
            L2_inv = jnp.linalg.inv(L2[b] + I_reg)
            R2_inv = jnp.linalg.inv(R2[b] + I_reg)
            q2 = jnp.real(jnp.trace(R2_inv @ L2_inv.T))
            q_bonds.append(jnp.sqrt(jnp.clip(q2, 0.0, None)))
        q_bonds = jnp.stack(q_bonds)

        loss = jax.scipy.special.logsumexp(beta * q_bonds) / beta
        return loss, q_bonds, L2, R2

    def _loss_only(p):
        loss, _, _, _ = _forward(p)
        return loss

    _jit_value_and_grad = jax.jit(jax.value_and_grad(_loss_only))
    _jit_forward = jax.jit(_forward)

    if n_total == 0:
        # No free parameters (max_free_sites=0): the result is the I/d
        # baseline, so the optimizer is skipped (scipy rejects an empty vector).
        _, q_bonds_j, L2_j, R2_j = _jit_forward(jnp.zeros(0, dtype=fdtype))
        # conjugated to the L[ext, bond] convention of compute_chain_boundary_operators
        return (
            [np.asarray(M).conj() for M in L2_j],
            [np.asarray(M).conj() for M in R2_j],
            float(jnp.max(q_bonds_j)),
        )

    def scipy_wrapper(p_np):
        val, grad = _jit_value_and_grad(jnp.asarray(p_np, dtype=fdtype))
        return np.asarray(val, dtype=np_fdtype), np.asarray(grad, dtype=np_fdtype)

    rng = np.random.default_rng(seed)
    eye_flat = np.eye(d).flatten() / np.sqrt(d)

    best = None
    for _ in range(max(1, n_restarts)):
        x0 = rng.standard_normal(n_total) * 0.01
        for k in range(w_L):
            x0[k * n_par : k * n_par + d * d] += eye_flat
        for k in range(w_R):
            base = sigma_span + k * n_par
            x0[base : base + d * d] += eye_flat

        res = minimize(
            scipy_wrapper,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10},
        )

        _, q_bonds_j, L2_j, R2_j = _jit_forward(jnp.asarray(res.x, dtype=fdtype))
        q_true = float(jnp.max(q_bonds_j))
        if best is None or q_true < best[0]:
            best = (q_true, L2_j, R2_j)

    q_achieved, L2_best, R2_best = best
    # conjugated to the L[ext, bond] convention of compute_chain_boundary_operators
    return (
        [np.asarray(M).conj() for M in L2_best],
        [np.asarray(M).conj() for M in R2_best],
        q_achieved,
    )
