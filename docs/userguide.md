# TreeMPU user guide

TreeMPU compiles an open-boundary **Matrix Product Unitary** (MPU) into an
**exact** quantum circuit, following the tree decomposition of Styliaris,
Trivedi and Cirac (PRL **135**, 260602 (2025)) with the extensions developed
in the thesis *Beyond-QCA MPUs: Generation, Certification, and Circuit
Synthesis* (M. Vassallo, University of Bologna, 2026). Its companion package
`mpu_generators` samples certified random MPUs to feed it.

## 1. Why

MPUs are the tensor-network description of many-body unitaries with a
bounded bond dimension $\chi$. They include quantum cellular automata (QCA)
and, beyond them, unitaries with unbounded causal cones. A generic MPU has no
known finite-depth circuit. The tree decomposition gives an exact circuit
for **any** MPU whose depth scales polynomially in $N$, with an exponent set
by a single conditioning number $q$. TreeMPU implements this construction
end to end: certification, boundary optimization, circuit assembly in
Qiskit, verification, and resource estimation up to $N\sim10^5$ without
building circuits.

## 2. Installation

```bash
git clone --recurse-submodules https://github.com/Mrcocvs02/TreeMPU.git
cd TreeMPU
pip install -r requirements.txt
pip install -e . -e ./mpu_generators
pytest -m "not slow"                 # ~2 min; full suite ~10 min
```

Python ≥ 3.10; dependencies: NumPy, SciPy, quimb ≥ 1.8, JAX (CPU, 64-bit), Qiskit ≥ 2.1.
Importing TreeMPU enables `jax_enable_x64`.

## 3. The method in one page

1. **Boundary operators.** For a bulk tensor $A$ and boundary vectors
   $\ell,r$, choose invertible $L,R$ such that every site block
   $V=L\,A\,R$ (and the edge blocks $\ell A R$, $L A r$) is an **isometry**.
   The stored operators `L[ext, bond]`, `R[ext, bond]` are transposed with
   respect to the thesis product $L\,A\,R$ (the second array axis of both
   contracts the bond). The canonical choice (thesis, canonical boundaries) is
   $L^2=\operatorname{Tr}_{\rm out}[(\sigma\otimes\ell\ell^\dagger)A^\dagger A]$
   with a density matrix $\sigma$ on the input leg (default $\sigma=I/d$),
   and similarly for $R$.
2. **Merging.** Adjacent blocks are merged by inserting
   $K=R^{-\top}L^{-1}$ on the two bond registers. The merging operator
   $M$ is rank one with LCU 1-norm
   $$q=\|R^{-\top}L^{-1}\|_F \quad(\text{`conditioning_number`}).$$
3. **Amplitude amplification.** The block encoding of $M$ succeeds with
   amplitude $1/q$; $\ell$ rounds of oblivious amplitude amplification make
   it exact, with $\ell$ the smallest integer such that
   $q\le C'(\ell)=\csc\frac{\pi}{4\ell+2}$ ($C'=1,2,3.236,4.494,\dots$).
   Extra rotation qubits lift $q$ exactly onto $C'(\ell)$.
4. **Tree.** Blocks are merged along a binary tree; each layer replays its
   children $2\ell+1$ times, so
   $$\text{depth}(N)\propto N^{\log_2(2\ell+1)},\qquad \text{gates}(N)\propto N^{\log_2(2\ell+1)+1}.$$

Everything therefore hinges on making $q$ small: $1<q\le2$ gives $\ell=1$
(depth exponent $\log_2 3\approx1.58$).

## 4. Workflow

```python
from mpu_generators import build_family
from TreeMPU import (
    UniformMPU,
    reconstruct_circuit_unitary,
    dense_unitary_from_tensors,
    phase_aligned_error,
)

tensors, l_vec, r_vec = build_family("branching_uniform", d=2, bond_dim=2, seed=0)
A = tensors[0]
mpu = UniformMPU(A, l_vec, r_vec, N=4, light=True, boundary_opt_k=1)
print(mpu.q_unif, mpu.ell, mpu.boundary_source)  # ~2.0 1 qD_general

qc, sys_regs, bond_regs, anc_regs, ancillas = mpu.synthesize()  # Qiskit circuit
# verification at small N (state-vector simulation, <= 24 qubits; here 16)
U_rec, leakage = reconstruct_circuit_unitary(qc, sys_regs, d=2)
err = phase_aligned_error(U_rec, dense_unitary_from_tensors([A] * 4, l_vec, r_vec))
# resources at large N without building circuits
params = mpu.synthesis_cost_params()
UniformMPU.analytic_synthesis_stats(params, N=2**16)
```

For site-dependent chains use `GeneralMPU(tensors, l_vec, r_vec)`: it
canonicalizes the chain and uses $q=\max_k q_k$ over bonds. Note the
default difference: `boundary_sigma_tau` is `True` for `UniformMPU` but
`False` for `GeneralMPU` (canonical $\sigma=\tau=I/d$ unless requested).

### Boundary strategies (`UniformMPU`)

| option | `boundary_source` | notes |
|---|---|---|
| `boundary_mode1_exact=True` | `mode1_exact` | closed form, $q=\chi$; Mode I tensors with $\ell=e_v$, $r=\sum_v e_v$ |
| `boundary_opt_k=k` | `qD_general` | solver reaching $q=\chi$ on the generator families, in any bond basis (may fail to converge; then the optimizer or the canonical boundary is used) |
| `boundary_opt_k=k` (fallback) | `optimized` | direct optimization over HPD $L,R$ |
| `boundary_opt_k=k, boundary_opt_gauge=True` | `optimized_gauge` | also a unitary gauge |
| default | `canonical_sigma_tau` | optimized $\sigma,\tau$, used only if $q$ decreases |
| `boundary_sigma_tau=False` | `canonical` | $\sigma=\tau=I/d$ (Styliaris et al.; needs Assumption 1) |

`GeneralMPU` reports one source per bond in `boundary_source` with its own
vocabulary: `canonical`, `sigma_tau_optimized` (the analogue of
`canonical_sigma_tau`) and `optimized`. `boundary_opt_gauge` is not
supported there (it raises together with `boundary_opt_p` and has no effect
without it).

Every non-canonical candidate is re-validated with an exact isometry
check. Iterative boundaries are accepted at isometry tolerance $10^{-4}$;
`block_isometry_residuals` reports the actual residuals (typically
$10^{-10}$–$10^{-7}$).

## 5. Conventions

| object | convention |
|---|---|
| site tensor | `A[p_in, p_out, bond_L, bond_R]`; block `A[:, :, v, u]` = transpose of the operator $A[v,u]$ |
| MPU | $U_N=\sum \ell_{v_0}r_{v_N}A[v_0,v_1]\otimes\cdots$, boundaries contracted without conjugation |
| boundary operators | `L[ext, bond]`, `R[ext, bond]`; merging kernel $R^{-\top}L^{-1}$ |
| qubit order | Qiskit little endian; site 0 = least significant system register |
| dense matrices | `dense_unitary_from_tensors` returns `U[out, in]` in the same order |
| circuit | equals $U_N$ up to a global phase; ancillas start and end in $\lvert0\rangle$ |
| isometry orientation | the site isometry maps the physical **input** leg to (output, external bonds), $V:p_{\rm in}\to(p_{\rm out},\text{ext})$; environments ($\sigma$, $\tau$) are density matrices on the input leg with the output traced. The opposite orientation is equivalent only under a no-signalling condition on $U^\dagger$ and fails, e.g., for the cyclic shift |

## 6. What is proved and what is validated

| statement | status |
|---|---|
| exactness of the tree decomposition with GOAA (given exact isometries) | proved (Styliaris et al.; thesis) |
| unitarity certificate: exact unitarity for $N\le\chi^2+1$ ⇒ all $N$ (translation invariant) | proved (thesis, finite-size cutoff theorem); `light=True` |
| $q\ge1$ for admissible boundaries | proved |
| $q\ge\chi$ for uniform pairs | **proved only under extra hypotheses**: $\lVert LR^\top\rVert_F\le1$ (holds when one side is an affine combination of input-leg environments: canonical, $\sigma/\tau$, gauge of the MPU; or the transfer matrix is invertible; or a diagonal-bond Mode II tensor). False without Assumption 1 (counterexample with $q\to1$ at $\chi=2$); open for optimized Mode I pairs at $d=2$ (numerically $\lVert LR^\top\rVert_F=1$) |
| $q\ge\chi$ in general | **false in general**: no uniform boundary tested went below $\chi$, but site-dependent windowed boundaries can reach $q<\chi$ (observed for `boundary_opt_p=1` on the middle bond of a 4-site chain, whose two windows both end on the chain boundary; e.g. 1.66 at $\chi=2$ with an exact circuit, see `test_site_dependent_boundary_goes_below_chi`); $q=\chi$ is a target of `mode1_exact`/`qD_general`, not a bound |
| isometry of longer blocks (length 1 ⇒ all lengths) | proved for environment pairs and their affine hull (canonical, $\sigma/\tau$, multi-site, gauge of the MPU); for optimized/qD pairs only validated numerically (block lengths up to 8, natural and complex bond basis) or by a one-time $n=2$ kernel-inclusion check. `GeneralMPU` accepts optimized windows on a site check, so all-length validity is not verified at run time |
| reflection on the current merge's registers only (and ancilla recycling) | proved by induction on the tree (the argument is to be included in the thesis appendix) for every $N$ and $\ell$ by induction on the tree, *given* that every tree block is an isometry and that each merging operator's block encoding shares the row $e_0$ (true for every encoding built here; false for a generic block encoding, where the thesis's reflection on all auxiliaries is needed). Error is $O(\text{isometry residual})$ for optimized boundaries. Also validated by exact reconstruction ($\ell=2$) and MPS simulation (N = 8) |
| chain $\sigma/\tau$ optimization (`GeneralMPU`) | exact at every length; the density matrices sit on the physical input leg (output traced). Putting them on the output leg would be exact only for no-signalling MPUs, and fails, e.g., for the cyclic shift |
| depth law $N^{\log_2(2\ell+1)}$ | exact solution of the implemented recursion |

## 7. Limits and cost

* **Memory.** `dense_unitary_from_tensors` and `reconstruct_circuit_unitary`
  are exponential ($d^{2N}$ entries, $2^{n_{\rm qubits}}$ amplitudes); the
  latter refuses circuits wider than 24 qubits, checked before allocating.
  Wider circuits can be verified column by column with
  `mps_circuit_columns` (MPS simulation; cost set by entanglement and gate
  count: minutes per column at 36 qubits).
* **Unitarity check.** `check_mpo_unitarity` costs about 0.4 s at 32 sites and
  about 30 s / 1.5 GB at 256 on a laptop; use `light=True`.
* **Qubit count.** By default ancillas are allocated per merge;
  `synthesize(pool_ancillas=True)` (or `pool_levels=k`) recycles them with
  one register per tree layer, trading depth for qubits. The estimators
  take the same options and stay exact (the one exception is
  `analytic_synthesis_stats(..., borrowed_ancillas=False)`, which prices the
  reflections without borrowed ancillas for comparison and is not what
  `synthesize` builds).
* **Composite gates.** `qc.depth()` counts multi-qubit composite gates;
  transpile to `["u", "cx"]` or use the estimators for elementary counts.
  If Qiskit cannot decompose a gate, the estimate is a lower bound and a
  warning is emitted.
* **Open boundaries only.** Periodic MPUs are not supported.

## 8. Further material

* `notebooks/`: six executed tutorials (basics → end to end).
* [theory_code_map.md](theory_code_map.md): thesis/paper labels → functions.
* [thesis_errata.md](thesis_errata.md): where the thesis and this
  library differ (the library is authoritative).
* `tests/`: unit tests and benchmarks of exactness (reconstruction) and of
  the scaling law.
* `benchmarks/`: `run_benchmarks.py` and its output `RESULTS.md` (exactness of the
  circuits, exact scaling up to N = 1024, comparison with generic unitary synthesis).
