# mpu_decomposition/utils.py
from __future__ import annotations
import numpy as np  # type: ignore
import quimb.tensor as qtn  # type: ignore
import scipy.linalg as la  # type: ignore
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as jax_expm
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


def optimize_q_unif(T_blocked, l_vec, r_vec, eps_reg=1e-8):
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

            # Contractions for L^2 and R^2
            L2 = jnp.einsum("ij, oipq, pm, ojmn -> qn", sigma, T, L_in, T.conj())
            R2 = jnp.einsum("ij, oipq, qn, ojmn -> pm", tau, T, R_in, T.conj())

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
    x0 = np.random.randn(2 * n_params) * 0.001
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


def _isometry_transfer_tensor(A: np.ndarray) -> np.ndarray:
    r"""
    Build the per-site transfer tensor used to evaluate isometry defects at
    any support length without ever enumerating $d^n$ physical paths
    explicitly (see func:`_bondspace_isometry_defect`).

    $$G[j,j',a,c,d,b] = \sum_i A[i,j,c,a]^* A[i,j',d,b]$$

    i.e. the physical *input* leg $i$ is already contracted here (it plays
    the role of the maximally-mixed "outside" support in Lemma 1), leaving
    the physical *output* legs $j, j'$ free together with two copies of the
    doubled bond space ($(a,c)$ and $(d,b)$) that $L^2, R^2$ act on.

    A's left and right bond dimensions are not linked: $(a,b)$ (the
    outer, forward-propagating legs, see func:`_bondspace_isometry_defect`)
    have size $D_R$ (A's own right bond) and $(c,d)$ (the inner legs,
    contracted against the running state) have size $D_L$. This matters in
    practice func:`to_canonical_form` absorbs the chain's boundary vectors,
    making the first/last site's *outer* bond trivially 1-dimensional while
    its *inner* bond is not, i.e. genuinely rectangular, assuming a single
    shared bond dimension here would raise on exactly that (now the default)
    case.
    """
    d, _, D_L, D_R = A.shape
    G = np.zeros((d, d, D_R, D_L, D_L, D_R), dtype=complex)
    for i in range(d):
        for j in range(d):
            for jp in range(d):
                G[j, jp] += np.einsum("ac,db->acdb", A[i, j].conj().T, A[i, jp])
    return G


def _bondspace_isometry_defect(G_list, d_list, left_mat, right_mat, all_prefixes: bool):
    r"""
    Frobenius-norm-squared isometry defect $\sum_n \|X_n - I_{d_1 \cdots
    d_n}\|_F^2$ ($X_n = V_n^\dagger V_n$, $n$ ranging over every prefix
    length if ``all_prefixes`` else only the full chain), for $V_n$ built by
    dressing sites $1,\dots,n$ (per-site transfer tensors in ``G_list``,
    from func:`_isometry_transfer_tensor`) with ``left_mat`` on the far
    left and ``right_mat`` on the far right.

    Computed *entirely* via bond-space (size-$D$) transfer-matrix
    recursions: cost is linear in ``len(G_list)`` and polynomial in $D$,
    and never materializes an object of size $\prod d_k$ (replaces an
    earlier implementation that built the $(d^n, d^n)$ Gram matrix
    explicitly -- fine for $n=1$ but exponential for $n>1$ or long chains,
    and the earlier ``dense_chain_tensor`` helper that merged a whole
    window into one dense tensor before checking it, exponential in the
    window length. Both are eliminated by this function.).

    - $\operatorname{Tr}[X_n]$: a single-copy $(D,D)$-sized environment
      recursion (trace over each site's shared output index).
    - $\operatorname{Tr}[X_n^2] = \|X_n\|_F^2$ (since $X_n$ is Hermitian,
      being a Gram matrix): a joint two-copy $(D,D,D,D)$-sized state
      recursion. Copy 2 uses $\overline{G_k}$ at the *same* site-local
      output indices as copy 1 -- NOT $G_k$ with those indices swapped
      (the two differ in general; only the conjugate relation reproduces
      $X_n$'s own Hermiticity, verified against brute-force enumeration on
      small cases before this replaced the exponential implementation).
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
            total_defect = (
                total_defect + jnp.real(tr_X2) - 2.0 * jnp.real(tr_X) + total_dim
            )

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
    r"""
    Lift the canonical $\sigma,\tau$-derived boundary construction and
    directly optimize invertible boundary tensors $L, R$ minimizing the MPU
    conditioning number

    $$q(L,R) = \sqrt{\operatorname{tr}[R^{-2}(L^{-2})^T]}$$

    subject to the isometry constraint $(V_n^{[L,R]})^\dagger V_n^{[L,R]} =
    I_{d^n}$ for every $n = 1, \dots, k$ (Styliaris et al., end matter Lemma
    5). $L, R$ are parametrized as $\exp(H)$ for a
    Hermitian $H$, which keeps them Hermitian positive-definite (HPD) by
    construction for any unconstrained real parameter vector. ``q`` is
    always evaluated on this underlying HPD pair, even when ``gauge=True``
    (see below).

    If ``gauge=True``, an additional unitary gauge $G = \exp(iH_G)$ is
    optimized jointly: $q$ is still evaluated on the HPD pair $(L,R)$, but
    the isometry constraint is evaluated on the gauge-dressed pair
    $(\Lambda,\Gamma) = (LG^{-1}, GR)$. Since $\Lambda,\Gamma$ are
    generically *not* HPD, this explores a strictly larger feasible set of
    invertible boundary completions than the ``gauge=False`` case (thesis
    section "Gauge optimization and the $q_{\mathrm{unif}}$ orbit"), and the
    returned $(L,R)$ are $(\Lambda,\Gamma)$ directly, i.e. already the
    effective boundary tensors to use, ``q_opt`` is nonetheless computed
    from the underlying HPD pair, not from $(\Lambda,\Gamma)$ (which are
    generally not HPD, making the HPD-only $q$ formula meaningless if
    applied to them directly).

    The isometry constraint is enforced as a quadratic penalty (not a hard
    constraint), so ``iso_residual`` should always be inspected before
    trusting ``q_opt`` -- this is a standard, but only approximately
    feasible, penalty-method tradeoff.

    ``q >= 1`` is *not* implied by the isometry constraint alone (unlike the
    canonical sigma/tau construction, where it holds automatically): q
    upper-bounds the LCU 1-norm of the merging operator, and the GOAA
    rotation angle is ``arcsin(1/C)``, undefined for ``C < 1``. This is
    softly penalized during optimization, but for some tensors (e.g. any
    D=1 bulk tensor, where the isometry constraint alone pins q to a unique
    value) the unconstrained isometry-satisfying optimum genuinely has
    q < 1. Callers MUST check ``q_opt >= 1`` (with tolerance) before
    adopting the result, falling back to the canonical construction
    otherwise.

    Parameters
    ----------
    A:
        Either a single bulk MPU tensor, shape ``(d, d, D, D)`` with axes
        ``(p_in, p_out, bond_L, bond_R)`` -- in which case the isometry
        constraint is checked at *every* prefix length ``1, ..., k`` of the
        ``k``-fold repetition of this same tensor (the "partial
        optimization" length; ``k`` need not span the full chain) -- or a
        list of ``m`` (possibly distinct) such tensors sharing consistent
        bond dimensions, representing one contiguous window of a
        non-uniform chain treated as a single unit: in that case ``k`` is
        ignored and the constraint is checked only at the window's full
        length ``m`` (this is what ``GeneralMPU``'s per-bond partial
        optimization passes -- a window of the chain's own distinct
        tensors, never merged into one dense tensor).
    k:
        Enforce the isometry constraint for support lengths ``1, ..., k``
        when A is a single tensor (ignored when A is a list; see
        above).
    gauge:
        If True, additionally optimize a unitary gauge transformation.
    l_vec, r_vec:
        Boundary vectors of the MPU. If given, the length-1 *boundary*
        isometries ``V^{[l,R]}_1 = l A R`` and ``V^{[L,r]}_1 = L A r`` are
        also constrained (Lemma 5's boundary case), which is what a
        tree-merge circuit synthesis actually needs from the first/last
        site's dressed tensor -- the bulk constraint above says nothing
        about how ``L, R`` interact with the chain's own ``l, r``. Without
        these, an optimized ``(L, R)`` can satisfy the bulk isometry
        perfectly while still being unusable at the chain's boundary
        (this is not merely a theoretical concern: it is exactly what
        happens for the gauge-optimized solution on ``cz_interaction_mpu``).
    n_restarts:
        Number of L-BFGS-B random restarts (the landscape is non-convex).
    lam_iso:
        Initial penalty weight for isometry violation (see ``adaptive``).
    maxiter:
        Maximum L-BFGS-B iterations per restart.
    seed:
        Base seed for restart initialization (deterministic).
    adaptive:
        A fixed ``lam_iso`` forces an awkward trade-off: too small and
        restarts converge to a low q that does not actually satisfy the
        isometry constraint; too large and the penalty term dominates the
        landscape, trapping restarts in shallow local optima that satisfy
        the constraint without ever exploring lower q. If True (default),
        ``lam_iso``/the restart perturbation radius are instead adjusted
        *between* restarts based on the previous restart's own outcome:
        infeasible (``iso > feas_tol``) tightens the penalty and shrinks
        the next perturbation (be more conservative, closer to a feasible
        point); feasible but no better than the best q found so far loosens
        the penalty and widens the perturbation (be more exploratory, since
        the constraint is not what is holding q back). A restart that is
        both feasible and improves on the best-known q leaves both
        unchanged. Regardless of where the schedule ends up, the best
        candidate found is always given one final high-``lam_iso`` polish
        step before being returned, so the returned ``iso_residual`` is
        never worse than what a fixed ``lam_iso`` run would have given.
        Set to False to reproduce the old fixed-``lam_iso``, best-by-raw-cost
        behavior exactly.
    feas_tol:
        Isometry residual (Frobenius norm) below which a restart's result is
        classified as "feasible" for the purposes of the adaptive schedule
        above and the best-candidate selection. This is independent of (and
        normally tighter than) whatever tolerance a caller later applies to
        the returned ``iso_residual``.
    warm_start:
        Optional ``(L0, R0)`` Hermitian positive-definite pair (typically
        the canonical sigma/tau boundary) to start restart 0 from, instead
        of the identity (``L = R = exp(0) = I``). Neither dominates the
        other in general: a cold start explores more of the landscape
        across ``n_restarts``, a warm start from a point already known to
        satisfy the isometry constraint can converge far faster and more
        reliably on tensors where the unconstrained landscape is hard to
        navigate (one measured case: warm-started convergence in ~3s where
        a cold run had not finished after 20 minutes). Callers wanting the
        best of both should call this function twice (``warm_start=None``
        and ``warm_start=(L0, R0)``) and keep whichever validated result is
        better -- this function itself always runs only one schedule.
        ``L0``/``R0`` are converted to the ``exp(H)`` parametrization via
        the matrix logarithm; must be genuinely HPD (as the canonical
        construction is) or the log is not real-diagonalizable-Hermitian
        and the warm start will not be exact.

    Returns
    -------
    L_opt, R_opt:
        Optimized boundary tensors (already gauge-dressed if ``gauge=True``,
        which requires the window's left and right bond dimensions to
        agree). ``L_opt`` has shape ``(D_L, D_L)`` (the window's own left
        bond) and ``R_opt`` has shape ``(D_R, D_R)`` (its own right bond);
        these differ whenever the window sits at a canonicalized chain's
        boundary, where the outer leg is absorbed to dimension 1.
    q_opt:
        Achieved MPU conditioning number ``q(L_opt, R_opt)`` when
        ``D_L == D_R``. Otherwise the paper's ``q(L,R) = sqrt(tr[R^-2(L^-2)^T])``
        is not defined at all (it presumes L, R on the same bond), so this
        falls back to ``sqrt(tr[R^-2])`` -- treat this value only as a
        search-guidance proxy in that case, not as a true MPU conditioning
        number; callers such as ``GeneralMPU`` compute the real, same-bond
        ``q_b`` separately, from the pairing that actually matters.
    iso_residual:
        Achieved ``sum_n ||V_n^dagger V_n - I||_F`` (Frobenius norm, summed
        over ``n = 1, ..., k``) -- inspect this against your tolerance
        before trusting ``q_opt``.
    """
    jax.config.update("jax_platform_name", "cpu")
    fdtype, cdtype, np_fdtype = jnp.float64, jnp.complex128, np.float64

    # A single tensor is checked at every prefix length 1..k of its k-fold
    # repetition (UniformMPU's usage); a list is a window of the chain's own
    # distinct tensors, checked only at its full length (GeneralMPU's usage
    # -- see the docstring). Either way, per-site transfer tensors are built
    # once and the isometry defect is evaluated via bond-space-only
    # transfer-matrix recursions (_bondspace_isometry_defect), never by
    # materializing a (prod(d_k), prod(d_k))-sized object.
    check_all_prefixes = isinstance(A, np.ndarray)
    tensors_list = [A] * k if check_all_prefixes else list(A)
    # L dresses the window's own left bond, R its own right bond -- these
    # need not agree: to_canonical_form absorbs the chain's boundary vectors,
    # so a window anchored at the true left/right edge has a trivial
    # (dimension-1) OUTER bond on that side while its INNER bond is whatever
    # the chain's own internal bond dimension is.
    D_L = tensors_list[0].shape[2]
    D_R = tensors_list[-1].shape[3]
    if gauge and D_L != D_R:
        raise ValueError(
            f"gauge=True requires a common bond dimension on both ends "
            f"(the same unitary G dresses both L and R via L@G^dagger, G@R), "
            f"but this window has D_L={D_L} != D_R={D_R}."
        )
    n_par_list = [D_L * D_L, D_R * D_R] + ([D_L * D_L] if gauge else [])

    G_list = [
        jnp.array(_isometry_transfer_tensor(Ak), dtype=cdtype) for Ak in tensors_list
    ]
    d_list = [Ak.shape[0] for Ak in tensors_list]

    # l_vec/r_vec anchor the *true* chain boundary, so their isometry check
    # always spans the tensor(s) actually passed in: just the single tensor
    # (length 1) when A is a single repeated tensor, or the whole window
    # (its full length) when A is an explicit list.
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
        """sum_n ||(L_iso A^n R_iso)^dagger (L_iso A^n R_iso) - I_{d^n}||_F^2"""
        # create_local_isometries dresses A's bond_R leg as R[output, bond_R]
        # (R_tn = Tensor(R, inds=["ext_R", "bond_1"])) -- i.e. R acts via its
        # *transpose* relative to how the transfer tensors below are set up
        # (which mirror L's convention, L[output, bond_L]). Using R_iso.T
        # here, not R_iso, is what makes the returned L_opt, R_opt directly
        # usable as UniformMPU.L / .R without a further transpose downstream.
        R_pkg = R_iso.T
        LdagL = L_iso.conj().T @ L_iso
        RRdag = R_pkg @ R_pkg.conj().T
        return _bondspace_isometry_defect(
            G_list, d_list, LdagL, RRdag, check_all_prefixes
        )

    l_vec_jax = None if l_vec is None else jnp.array(l_vec, dtype=cdtype)
    r_vec_jax = None if r_vec is None else jnp.array(r_vec, dtype=cdtype)

    def _boundary_isometry_defect(L_iso, R_iso):
        """
        ||V^[l,R]^dagger V^[l,R] - I||_F^2 + ||V^[L,r]^dagger V^[L,r] - I||_F^2

        Spans the same tensor(s) as boundary_G_list/boundary_d_list (see
        above): create_local_isometries only ever needs the boundary tensor
        actually being optimized (a single site for UniformMPU, or the
        whole window for GeneralMPU's per-bond partial optimization) to be
        an isometry once dressed with l_vec/r_vec -- the tree-merge
        recursion, not this optimizer, is what grows that to longer support.
        """
        defect = jnp.zeros((), dtype=fdtype)

        if l_vec_jax is not None:
            # R_iso.T: see the matching note in _isometry_defect.
            R_pkg = R_iso.T
            RRdag = R_pkg @ R_pkg.conj().T
            l_outer = jnp.outer(l_vec_jax, l_vec_jax.conj())
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

    # n_free = 3 if gauge else 2

    def _cost_and_diagnostics(p, lam):
        mats = unpack(p)
        L = jax_expm(mats[0])
        R = jax_expm(mats[1])

        Li = jnp.linalg.inv(L)
        Ri = jnp.linalg.inv(R)
        if D_L == D_R:
            q2 = jnp.real(jnp.trace((Ri @ Ri) @ (Li @ Li).T)).astype(fdtype)
        else:
            # q(L,R) = tr[R^-2 (L^-2)^T] presumes L, R act on the SAME bond
            # (matching the paper's q_k, always evaluated at one shared cut)
            # -- true whenever this window's own outer bonds agree (the
            # UniformMPU case, and any GeneralMPU window not anchored at a
            # canonicalized chain's trivial boundary). When they do not
            # (D_L=1 here, from to_canonical_form absorbing l_vec/r_vec),
            # L and (L^-2)^T live on a different-dimensional space than R,
            # so no matrix product/trace combining them is even defined.
            # This "L" still matters for the bulk isometry check below; it
            # is only this *scalar proxy* that must fall back to R's own
            # conditioning alone -- the actual accept/reject decision the
            # caller makes uses the correctly-paired, same-bond q_b
            # (GeneralMPU._compute_q_bond), never this internal one.
            q2 = jnp.real(jnp.trace(Ri @ Ri)).astype(fdtype)

        if gauge:
            Gmat = jax_expm(1j * mats[2])
            L_iso = L @ Gmat.conj().T
            R_iso = Gmat @ R
        else:
            L_iso, R_iso = L, R

        iso = _isometry_defect(L_iso, R_iso) + _boundary_isometry_defect(L_iso, R_iso)

        # q < 1 is mathematically reachable by the bare isometry constraint
        # (e.g. it is the *only* solution for D=1), but breaks the calling
        # circuit synthesis: q upper-bounds the LCU 1-norm C of the merging
        # operator, and GOAA's rotation angle is arcsin(1/C), undefined for
        # C < 1. Softly penalize dipping below 1 so the optimizer trades
        # off into that region only when the isometry term leaves no other
        # choice (as for D=1) -- callers must still check q_opt >= 1 - tol
        # before trusting the result; see the docstring.
        q1_violation = jnp.clip(1.0 - jnp.sqrt(jnp.abs(q2)), 0.0, None) ** 2

        cost = q2 + lam * iso + lam * q1_violation
        # L, R (the underlying HPD pair) are returned alongside L_iso, R_iso
        # (the effective, gauge-dressed pair to actually use): q must always
        # be read off L, R -- L_iso, R_iso are generally not HPD when
        # gauge=True, and applying the HPD-only q formula to them directly
        # is meaningless (this was a real bug: an earlier version recomputed
        # the returned q_opt from L_iso, R_iso, which for several notebook
        # test tensors produced nonsensical values, including an exact 0.0
        # for lee_yang_mpu, because L_iso^dagger L_iso is not HPD so its
        # "square" trace is not even guaranteed real/nonnegative).
        return cost, jnp.sqrt(jnp.abs(q2)), jnp.sqrt(jnp.abs(iso)), L, R, L_iso, R_iso

    # lam is a traced argument (not a Python-float closure constant) so the
    # adaptive schedule below can vary it every restart without forcing a
    # recompilation each time -- only the *shape/dtype* of the arguments
    # (fixed here) triggers a retrace, not the runtime value of a traced arg.
    _jit_full = jax.jit(_cost_and_diagnostics)
    _jit_cost_grad = jax.jit(
        jax.value_and_grad(lambda p, lam: _cost_and_diagnostics(p, lam)[0])
    )

    def cost_and_grad_np(p, lam):
        val, g = _jit_cost_grad(
            jnp.array(p, dtype=fdtype), jnp.asarray(lam, dtype=fdtype)
        )
        return float(val), np.array(g, dtype=np_fdtype)

    p_warm = np.zeros(int(_offsets[-1]), dtype=np_fdtype)
    _jit_full(jnp.array(p_warm, dtype=fdtype), jnp.asarray(lam_iso, dtype=fdtype))
    _jit_cost_grad(jnp.array(p_warm, dtype=fdtype), jnp.asarray(lam_iso, dtype=fdtype))

    mats0_warm = None
    if warm_start is not None:
        L0, R0 = warm_start
        # L, R are parametrized as exp(H) for Hermitian H; the matrix log
        # of an HPD matrix is exactly such an H, so this places restart 0
        # exactly at (L0, R0) rather than merely near it. logm's result is
        # Hermitian only up to floating point for a numerically-HPD input;
        # pack()/unpack() symmetrize anyway (_herm_from_real), so the tiny
        # residual asymmetry is discarded on the first round trip and does
        # not need to be cleaned up here.
        H0_L = la.logm(np.asarray(L0, dtype=complex))
        H0_R = la.logm(np.asarray(R0, dtype=complex))
        mats0_warm = [H0_L, H0_R] + (
            [np.zeros((D_L, D_L), dtype=complex)] if gauge else []
        )

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
                Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal(
                    (dim, dim)
                )
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
            _, q_i, iso_i, _, _, _, _ = _jit_full(
                jnp.array(res.x, dtype=fdtype), lam_used
            )
            q_i, iso_i = float(q_i), float(iso_i)
            is_feasible = iso_i < feas_tol
            improved = is_feasible and q_i < best_feasible_q - 1e-9

            if is_feasible and (best_feasible_p is None or q_i < best_feasible_q):
                best_feasible_q = q_i
                best_feasible_p = res.x.copy()

            if restart < n_restarts - 1:
                if not is_feasible:
                    # Isometry not satisfied tightly enough -> push harder
                    # and take smaller, more conservative steps next time.
                    cur_lam = min(cur_lam * 3.0, lam_hi)
                    cur_scale_mult = max(cur_scale_mult * 0.7, 0.2)
                elif not improved:
                    # Feasible but stuck at (or above) the best q found so
                    # far -> the penalty is likely crowding out exploration
                    # of lower-q directions; loosen it and widen the search.
                    cur_lam = max(cur_lam / 2.0, lam_lo)
                    cur_scale_mult = min(cur_scale_mult * 1.5, 5.0)
                # else: feasible and improved -> stay in this regime.

    best_p = best_feasible_p if best_feasible_p is not None else best_any_p

    if adaptive:
        # The schedule above may end on a loosened lam_iso for exploration
        # purposes; the returned candidate must satisfy the caller's own
        # tolerance regardless of where the schedule happened to land, so
        # always finish with one dedicated high-penalty refinement from the
        # best point found. Only accept it if it stays feasible (or nothing
        # feasible existed to lose) -- otherwise keep the pre-polish point.
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
        _, _, iso_polish, _, _, _, _ = _jit_full(
            jnp.array(res_polish.x, dtype=fdtype), polish_lam
        )
        if float(iso_polish) < feas_tol or best_feasible_p is None:
            best_p = res_polish.x

    _, _, _, L_hpd_jax, R_hpd_jax, L_opt_jax, R_opt_jax = _jit_full(
        jnp.array(best_p, dtype=fdtype), jnp.asarray(lam_iso, dtype=fdtype)
    )
    L_opt = np.array(L_opt_jax)
    R_opt = np.array(R_opt_jax)

    # q from the underlying HPD pair (L_hpd, R_hpd), NOT from L_opt, R_opt --
    # see the note in _cost_and_diagnostics on why using the gauge-dressed
    # pair here would be meaningless.
    Li_f = np.linalg.inv(np.array(L_hpd_jax))
    Ri_f = np.linalg.inv(np.array(R_hpd_jax))
    if D_L == D_R:
        q_opt = float(np.real(np.trace((Ri_f @ Ri_f) @ (Li_f @ Li_f).T)))
    else:
        q_opt = float(np.real(np.trace(Ri_f @ Ri_f)))  # see the matching note above
    q_opt = float(np.sqrt(max(q_opt, 0.0)))

    iso_residual = float(
        np.sqrt(
            np.abs(
                _isometry_defect(L_opt_jax, R_opt_jax)
                + _boundary_isometry_defect(L_opt_jax, R_opt_jax)
            )
        )
    )

    return L_opt, R_opt, q_opt, iso_residual


# ======================================================================
# Random MPU generation
#
# Follows "Certified Generators of Matrix Product Unitaries: A Minimal,
# Self-Contained Account". That reference writes an MPU as
#
#     U_N = l^T A_N ... A_1 r,     A_n[a, b] in M_d(C),
#
# and certifies U_N^dagger U_N = 1 for *every* N by imposing an exact
# algebraic condition on the tensors rather than by sampling them freely
# (unconstrained Haar sampling provably fails: the boundary-compressed
# block satisfies E[Phi^dagger Phi] = 1_d / D, a strict contraction for
# D > 1). Three such conditions are implemented here, one per ``mode``.
#
# Two conventions differ from that reference and are converted here, once,
# in _pdf_to_package_tensor / the site-order reversal:
#
#   * index order -- the reference's A[bond_L, bond_R, p_out, p_in] versus
#     this package's A[p_in, p_out, bond_L, bond_R];
#   * site order -- the reference's matrix product runs A_N ... A_1, so
#     its A_1 is the *rightmost* site, whereas this package indexes sites
#     left to right with ``l_vec`` meeting ``tensors[0]``.
# ======================================================================


def get_mpo_site_tensors(k: int, A_np: np.ndarray) -> tuple[qtn.Tensor, qtn.Tensor]:
    """
    Return ket and bra tensors for site ``k`` of a uniform MPU.

    The package-wide tensor convention is

        A_np[p_in, p_out, bond_L, bond_R].

    Parameters
    ----------
    k:
        Site label. Physical and bond indices are named using this value.
    A_np:
        Local MPU tensor with shape ``(d_in, d_out, D_L, D_R)``.

    Returns
    -------
    A_k:
        Ket tensor with indices
        ``p_in_k, p_out_k, bond_{k-1}, bond_k``.
    A_k_dag:
        Conjugate tensor used in boundary/unitarity contractions.

    Notes
    -----
    The first physical axis of ``A_k_dag`` corresponds to the conjugate of
    ``p_in_k`` and is named ``p_out_f_k`` because in ``U U†``-type contractions
    it plays the final output/input-comparison role.
    """
    if A_np.ndim != 4:
        raise ValueError(
            "A_np must have four axes: "
            "(p_in, p_out, bond_L, bond_R). "
            f"Got shape {A_np.shape}."
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
    """Compute M^{1/2} for a Hermitian positive-semidefinite matrix."""
    eigvals, eigvecs = np.linalg.eigh(M)
    if np.any(eigvals < -tol):
        raise ValueError(
            f"Matrix is not positive-semidefinite: min eigenvalue = {eigvals.min():.2e}"
        )
    eigvals = np.maximum(
        eigvals, 0.0
    )  # to erase small negative eigenvalues due to numerical errors
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def get_merging_operator(L_inv: np.ndarray, R_inv: np.ndarray) -> np.ndarray:
    """
    Computes the merging operator M from the boundary operators L2 and R2.
    This operator is crucial for the synthesis of the MPU into a quantum circuit.
    The merging operator encapsulates the entanglement structure at the boundaries
    and is derived from the inverses of L2 and R2.
    """
    # Validate input dimensions
    if L_inv.shape[0] != L_inv.shape[1] or R_inv.shape[0] != R_inv.shape[1]:
        raise ValueError("L2 and R2 must be square matrices.")

    if L_inv.shape != R_inv.shape:
        raise ValueError("L2 and R2 must be of the same dimension.")

    # Compute the merging operator M = |0,0⟩⟨I|(R⁻¹ ⊗ L⁻¹)
    try:
        D = L_inv.shape[0]

        M = np.zeros((D, D, D, D), dtype=complex)

        kernel = R_inv.T @ L_inv  # (D, D)
        M[0, 0, :, :] = kernel

        return M.reshape(D**2, D**2)
    except la.LinAlgError as e:
        raise ValueError(f"Failed to compute merging operator: {e}")


def dilate_isometry_to_unitary(V_mat: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Dilates an isometry V (shape: m x n, m >= n) to a unitary (shape: m x m).

    The complementary columns are an orthonormal basis for the null space of V†,
    computed via SVD (scipy.linalg.null_space).

    Parameters
    ----------
    V_mat : np.ndarray, shape (m, n), m >= n
        A matrix within ``tol`` (Frobenius norm) of satisfying V†V = I_n.
    tol:
        Acceptance tolerance on ``||V†V - I_n||_F``. The canonical
        (sigma/tau-derived) construction satisfies this to machine
        precision; a boundary pair from a numerical optimizer
        (utils.optimize_boundary_operators) generally does not, so this is
        deliberately not hardcoded to machine epsilon.

    Returns
    -------
    V_tilde : np.ndarray, shape (m, m)
        Unitary dilation. ``V_tilde[:, :n]`` is the *exact* isometry closest
        to ``V_mat`` in Frobenius norm (via the polar decomposition
        ``V_mat = U S Vh -> U @ Vh``), not ``V_mat`` itself -- inputs within
        ``tol`` are snapped onto the isometry manifold exactly, so that
        numerical drift from an upstream optimizer never compounds through
        the tree-merge recursion that consumes this dilation.

    Raises
    ------
    ValueError
        If m < n or V is farther than ``tol`` from being an isometry.
    """
    m, n = V_mat.shape

    if m < n:
        raise ValueError(
            f"V_mat must have at least as many rows as columns. Got ({m}, {n})."
        )

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


def compute_lcu_pad_indices(
    D_bond: int, D_bond_pad: int, dim_rot: int = 2
) -> np.ndarray:
    """
    Computes the index map from the raw (unpadded) bond+rot space to the
    padded bond+rot space used by Qiskit's little-endian qubit ordering.

    The raw space has axes (ext_R, ext_L, rot) with sizes
    (D_bond, D_bond, dim_rot) in C-order (ext_R slowest, rot fastest).

    The padded space has the same axis order but D_bond is replaced by
    D_bond_pad, so the flat index of (i_R, i_L, rot) in the padded space is:
        idx_pad = i_L + i_R * D_bond_pad + rot * D_bond_pad**2

    This reflects Qiskit's convention:
        - ext_L  -> LSB  (stride 1)
        - ext_R  -> mid  (stride D_bond_pad)
        - rot    -> MSB  (stride D_bond_pad**2)

    Parameters
    ----------
    D_bond : int
        Physical (unpadded) bond dimension.
    D_bond_pad : int
        Padded bond dimension (must be >= D_bond, typically next power of 2).
    dim_rot : int, optional
        Dimension of the rotation register. Default is 2 (for a single qubit rotation).
        Rarely 1 as would imply an integer ell parameter by chance.

    Returns
    -------
    pad_indices : np.ndarray, shape (D_bond * D_bond * dim_rot,), dtype=np.intp
        pad_indices[k] is the flat index in the padded space corresponding
        to the k-th basis vector in the raw space (C-order enumeration).

    Raises
    ------
    ValueError
        If D_bond_pad < D_bond.
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
    """
    Embeds the raw 2D block-diagonal W_ctrl into the padded Hilbert space
    required by Qiskit's qubit register layout.

    Parameters
    ----------
    W_ctrl : np.ndarray, shape (dim_lcu_A * dim_W_raw, dim_lcu_A * dim_W_raw)
        The raw controlled-unitary built in the unpadded space.
    D_bond : int
        Physical (unpadded) bond dimension.
    D_bond_pad : int
        Padded bond dimension (next power of 2 >= D_bond).
    dim_rot : int, optional
        Dimension of the rotation ancilla register. Default is 2.

    Returns
    -------
    W_ctrl_full : np.ndarray, shape (dim_lcu_A * dim_W_pad, dim_lcu_A * dim_W_pad)
        Padded controlled-unitary ready for UnitaryGate instantiation.

    Raises
    ------
    ValueError
        If W_ctrl shape is inconsistent with the given dimensions.
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


def to_canonical_form(
    tensors: list[np.ndarray],
    l_vec: np.ndarray,
    r_vec: np.ndarray,
    tol: float = 1e-10,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    r"""
    Bring a (nonuniform) MPU into the canonical form of Definition 3 of
    Styliaris et al., i.e. the usual MPS canonical form of its Choi state,
    obtained by successive SVDs over the paired physical index $(i,j)$.

    This is what makes Assumption 1 unnecessary in the nonuniform case. The
    paper is explicit: the site-dependent $L_j, R_k$ keep their isometry
    properties *"even without the need for Assumption 1"*, because "bringing,
    without loss of generality, the MPU into a canonical form" makes the bond
    dimension at every cut minimal, and then choosing $\rho, \tau$ maximally
    mixed gives

    .. math::
        L_k = 1, \qquad R_k = \mathrm{diag}(s_{k,1}, \dots, s_{k,D_k}),

    with the $s_{k,i}$ the strictly positive Choi-state Schmidt values (paper
    Eq. 26 / A8). Rank deficiency in a raw chain is therefore a statement
    about its *representation*, not about the MPU: it means the given bond
    dimension is not minimal, and canonicalizing removes the redundancy
    rather than requiring the caller to work around it.

    Both sweeps are needed, and their order is not free. One sweep alone
    makes the environment on *its* side proportional to the identity but
    leaves the rank seen from the other side untouched, so the minimal
    dimension at a cut -- the smaller of the two -- is only reached after
    sweeping back (verified: a chain whose left sweep stops at bonds [2, 3]
    with a rank-deficient ``R`` reduces to [2, 2] and full rank once the
    other sweep runs). The sweep that runs *last* is the one whose gauge
    survives, so to land on Eq. (26)'s ``L_k = 1`` the left sweep must come
    second -- hence right-to-left first, then left-to-right.

    The boundary vectors are absorbed, so the returned chain has trivial
    (dimension-1) outer legs and its own ``l_vec``/``r_vec`` are ``[1]``.

    Parameters
    ----------
    tensors, l_vec, r_vec:
        Chain in this package's ``(p_in, p_out, bond_L, bond_R)`` convention.
    tol:
        Relative threshold below which a singular value counts as zero and
        its bond direction is dropped.

    Returns
    -------
    tensors, l_vec, r_vec:
        Canonical chain with minimal bond dimensions, and the matching
        trivial boundary vectors.
    """
    T = [np.asarray(A, dtype=complex) for A in tensors]
    if len(T) < 2:
        raise ValueError("to_canonical_form requires at least 2 tensors.")

    T[0] = np.tensordot(np.asarray(l_vec, dtype=complex), T[0], axes=(0, 2))[
        :, :, None, :
    ]
    T[-1] = np.tensordot(T[-1], np.asarray(r_vec, dtype=complex), axes=(3, 0))[
        :, :, :, None
    ]
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


def compute_chain_boundary_operators(
    tensors: list[np.ndarray], l_vec: np.ndarray, r_vec: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], complex]:
    """
    Incrementally builds the left- and right-boundary (environment) operators
    for every internal bond of a (possibly nonuniform) MPU chain.

    ``tensors[k]`` has shape ``(d, d, D_left_k, D_right_k)`` for
    ``k = 0, ..., N-1``, with ``D_left_0 == len(l_vec)`` and
    ``D_right_{N-1} == len(r_vec)``. Each site is assumed to share the same
    physical dimension ``d``.

    This generalizes the single-site maximally-mixed-state contraction used
    for uniform-bulk MPUs (Eq. 22-23 of Styliaris et al.) to an incremental
    left/right sweep over the whole chain (Lemma 1 of the same reference),
    each sweep costing O(N) tensor contractions rather than recomputing every
    cut from scratch.

    Returns
    -------
    L2_list:
        ``L2_list[b]`` is $L_{b+1}^2$ (shape ``(D_right_b, D_right_b)``), the
        left-cap operator for a segment starting at site ``b + 1``, for
        ``b = 0, ..., N-2``.
    R2_list:
        ``R2_list[b]`` is $R_b^2$ (shape ``(D_right_b, D_right_b)``), the
        right-cap operator for a segment ending at site ``b``, for
        ``b = 0, ..., N-2``.
    unitarity_residual:
        Scalar that must equal 1 (within tolerance) iff the chain, together
        with ``l_vec`` and ``r_vec``, defines a unitary MPO.
    """
    N = len(tensors)
    if N < 2:
        raise ValueError("compute_chain_boundary_operators requires N >= 2.")

    d = tensors[0].shape[0]

    # --- Left sweep: env_L after site k is L_{k+1}^2 (bond k) ---
    L2_list: list[np.ndarray] = [None] * (N - 1)  # type: ignore[list-item]

    A0 = np.tensordot(l_vec, tensors[0], axes=(0, 2))  # (d, d, D_right_0)
    env_L = np.einsum("ijc,ijd->cd", A0, A0.conj()) / d
    L2_list[0] = env_L

    for k in range(1, N - 1):
        Ak = tensors[k]  # (d, d, D_left_k, D_right_k)
        env_L = np.einsum("ijac,ab,ijbd->cd", Ak, env_L, Ak.conj()) / d
        L2_list[k] = env_L

    # --- Right sweep: env_R after site k is R_{k-1}^2 (bond k-1) ---
    R2_list: list[np.ndarray] = [None] * (N - 1)  # type: ignore[list-item]

    A_last = np.tensordot(tensors[N - 1], r_vec, axes=(3, 0))  # (d, d, D_left_{N-1})
    env_R = np.einsum("ijm,ijn->mn", A_last, A_last.conj()) / d
    R2_list[N - 2] = env_R

    for k in range(N - 2, 0, -1):
        Ak = tensors[k]  # (d, d, D_left_k, D_right_k)
        env_R = np.einsum("ijac,cd,ijbd->ab", Ak, env_R, Ak.conj()) / d
        R2_list[k - 1] = env_R

    # --- Global closure: continue the left sweep through the last site ---
    unitarity_residual = (
        np.einsum("ijm,mn,ijn->", A_last, L2_list[N - 2], A_last.conj()) / d
    )

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
    r"""
    Generalize :func:`optimize_q_unif` from a single repeated bulk tensor to
    a (possibly nonuniform) chain -- the ``GeneralMPU`` analogue of
    ``UniformMPU``'s ``boundary_sigma_tau`` baseline (thesis section
    "Partial Optimization").

    :func:`compute_chain_boundary_operators` builds every bond's $L^2$/$R^2$
    by an incremental left/right sweep that closes each site's physical legs
    with the maximally mixed state $I/d$ (Lemma 1) -- one arbitrary,
    feasible point, not an optimum. Exactly as ``optimize_q_unif`` replaces
    that single fixed choice with a free density matrix for a single
    repeated tensor, this replaces it with a free, per-site density matrix
    $\sigma_k$ (left sweep) / $\tau_k$ (right sweep) for every site within a
    bounded window nearest the corresponding chain boundary; sites beyond
    the window keep using $I/d$, exactly like the unoptimized construction.
    The sweep itself always costs $O(N)$ regardless of the window size --
    only the *parameter count* being optimized is capped by
    ``max_free_sites`` (``None`` means every site is free).

    Because each $\sigma_k$/$\tau_k$ is always a valid density matrix by
    construction (parametrized as $XX^\dagger/\operatorname{tr}(XX^\dagger)$,
    exactly as in ``optimize_q_unif``), the isometry property holds
    automatically for *any* optimization outcome -- unlike
    ``optimize_boundary_operators``, there is no isometry constraint to
    violate here and hence nothing to validate post-hoc: the worst this can
    do is fail to improve on the $I/d$ baseline, never produce an invalid
    pair.

    Since the left and right sweeps carry a running environment matrix
    across all $N-1$ bonds, optimizing the window nearest one boundary can
    improve every bond's $L^2$/$R^2$, not just the ones inside the window --
    the accumulated sweep state carries the improvement forward through the
    remaining, fixed-$I/d$ sites. All free sites (both windows) are
    optimized jointly against a single objective: a smooth (log-sum-exp)
    approximation of $\max_b q_b$ over every bond, since that maximum, not
    any individual bond, is what ``self.q`` and the circuit depth actually
    depend on.

    Parameters
    ----------
    tensors, l_vec, r_vec:
        Same as :func:`compute_chain_boundary_operators`.
    max_free_sites:
        Number of sites nearest each boundary whose density matrix is a
        free parameter (``None`` = every site, i.e. the whole chain).
    eps_reg:
        Tikhonov regularization added before each matrix inversion inside
        the loss (matches ``optimize_q_unif``).
    n_restarts:
        Number of L-BFGS-B random restarts; the restart with the best true
        $\max_b q_b$ is kept.
    maxiter:
        Maximum L-BFGS-B iterations per restart.
    beta:
        Sharpness of the log-sum-exp approximation to $\max_b q_b$ used as
        the optimization objective (larger = closer to the true max, but a
        harder landscape).
    seed:
        Base seed for restart initialization (deterministic).

    Returns
    -------
    L2_list, R2_list:
        Same shapes/semantics as :func:`compute_chain_boundary_operators`'s
        return values, built from the optimized $\sigma_k$/$\tau_k$.
    q_achieved:
        The true (hard) $\max_b q_b$ reached by the best restart -- compare
        this against the $I/d$ baseline's own per-bond $q_b$ before
        adopting, since a lower max does not guarantee every individual
        bond improved.
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
        # Mirrors optimize_q_unif's L2 step exactly, with env replacing the
        # rank-1 l_vec seed once the sweep has moved past the first site.
        return jnp.einsum("ij,oipq,pm,ojmn->qn", sigma, T, env, T.conj())

    def _step_R(tau, T, env):
        # Mirrors optimize_q_unif's R2 step exactly.
        return jnp.einsum("ij,oipq,qn,ojmn->pm", tau, T, env, T.conj())

    def _forward(p):
        sigmas = [vec_to_rho(p[k * n_par : (k + 1) * n_par]) for k in range(w_L)]
        taus = [
            vec_to_rho(p[sigma_span + k * n_par : sigma_span + (k + 1) * n_par])
            for k in range(w_R)
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
        # No free parameters at all (max_free_sites=0): the "optimized"
        # result is exactly the I/d baseline. Skip the optimizer entirely
        # rather than handing scipy an empty parameter vector.
        _, q_bonds_j, L2_j, R2_j = _jit_forward(jnp.zeros(0, dtype=fdtype))
        return (
            [np.asarray(M) for M in L2_j],
            [np.asarray(M) for M in R2_j],
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
    return (
        [np.asarray(M) for M in L2_best],
        [np.asarray(M) for M in R2_best],
        q_achieved,
    )
