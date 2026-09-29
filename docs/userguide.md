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

Python ≥ 3.10; dependencies: NumPy, SciPy, quimb, JAX (CPU, 64-bit), Qiskit ≥ 1.3.
Importing TreeMPU enables `jax_enable_x64`.

## 3. The method in one page

1. **Boundary operators.** For a bulk tensor $A$ and boundary vectors
   $\ell,r$, choose invertible $L,R$ such that every site block
   $V=L\,A\,R$ (and the edge blocks $\ell A R$, $L A r$) is an **isometry**.
   The canonical choice (thesis, canonical boundaries) is
   $L^2=\operatorname{Tr}_{\rm out}[(\sigma\otimes\ell\ell^\dagger)A^\dagger A]$
   with a density matrix $\sigma$ on the input leg (default $\sigma=I/d$),
   and similarly for $R$.
2. **Merging.** Adjacent blocks are merged by inserting
   $K=R^{-\top}L^{-1}$ on the two bond registers. The merging operator
   $M$ is rank one with LCU 1-norm
   $$q=\|R^{-\top}L^{-1}\|_F \quad(\text{`conditioning\_number`}).$$
3. **Amplitude amplification.** The block encoding of $M$ succeeds with
   amplitude $1/q$; $\ell$ rounds of oblivious amplitude amplification make
   it exact, with $\ell$ the smallest integer such that
   $q\le C'(\ell)=\csc\frac{\pi}{4\ell+2}$ ($C'=1,2,3.236,4.494,\dots$).
   Extra rotation qubits lift $q$ exactly onto $C'(\ell)$.
4. **Tree.** Blocks are merged along a binary tree; each layer replays its
   children $2\ell+1$ times, so
   $$\text{depth}(N)\propto N^{\log_2(2\ell+1)},\qquad \text{gates}(N)\propto N^{\log_2(2\ell+1)+1}.$$

Everything therefore hinges on making $q$ small: $q\le2$ gives $\ell=1$
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
canonicalizes the chain and uses $q=\max_k q_k$ over bonds.

### Boundary strategies (`UniformMPU`)

| option | `boundary_source` | notes |
|---|---|---|
| `boundary_mode1_exact=True` | `mode1_exact` | closed form, $q=\chi$; Mode I tensors with $\ell=e_v$, $r=\sum_v e_v$ |
| `boundary_opt_k=k` | `qD_general` | solver reaching $q=\chi$ on the generator families (not gauge covariant; may fail and fall back) |
| `boundary_opt_k=k` (fallback) | `optimized` | direct optimization over HPD $L,R$ |
| `boundary_opt_k=k, boundary_opt_gauge=True` | `optimized_gauge` | also a unitary gauge |
| default | `canonical_sigma_tau` | optimized $\sigma,\tau$, used only if $q$ decreases |
| `boundary_sigma_tau=False` | `canonical` | $\sigma=\tau=I/d$ (Styliaris et al.; needs Assumption 1) |

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

## 6. What is proved and what is validated

| statement | status |
|---|---|
| exactness of the tree decomposition with GOAA (given exact isometries) | proved (Styliaris et al.; thesis) |
| unitarity certificate: exact unitarity for $N\le\chi^2+1$ ⇒ all $N$ (translation invariant) | proved (thesis, finite-size cutoff theorem); `light=True` |
| $q\ge1$ for admissible boundaries | proved |
| $q\ge\chi$ | **false in general**: no uniform boundary tested went below $\chi$, but site-dependent windowed boundaries reach $q<\chi$ (e.g. 1.66 at $\chi=2$, exact circuit, see `test_site_dependent_boundary_goes_below_chi`); $q=\chi$ is a target of `mode1_exact`/`qD_general`, not a bound |
| isometry of longer blocks for iterative boundaries | validated numerically (block lengths up to 8) |
| reflection on the current merge's registers only | validated by exact reconstruction (incl. $\ell=2$); no written proof |
| depth law $N^{\log_2(2\ell+1)}$ | exact solution of the implemented recursion |

## 7. Limits and cost

* **Memory.** `dense_unitary_from_tensors` and `reconstruct_circuit_unitary`
  are exponential ($d^{2N}$ entries, $2^{n_{\rm qubits}}$ amplitudes); the
  latter refuses circuits wider than 24 qubits, checked before allocating.
* **Unitarity check.** `check_mpo_unitarity` costs ~0.4 s at 32 sites and
  ~30 s / 1.5 GB at 256; use `light=True`.
* **Qubit count.** Ancillas are allocated per merge (no recycling), so
  `synthesize()` uses more qubits than the thesis minimum; the analytic
  estimator can model one register per tree layer (`pool_ancillas=True`).
* **Composite gates.** `qc.depth()` counts multi-qubit composite gates;
  transpile to `["u", "cx"]` or use the estimators for elementary counts.
  If Qiskit cannot decompose a gate, the estimate is a lower bound and a
  warning is emitted.
* **Open boundaries only.** Periodic MPUs are not supported.

## 8. Further material

* `notebooks/`: six executed tutorials (basics → end to end).
* [theory_code_map.md](theory_code_map.md): thesis/paper labels → functions.
* `tests/`: unit tests and benchmarks of exactness (reconstruction) and of
  the scaling law.
