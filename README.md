# TreeMPU

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2508.08160-b31b1b.svg)](https://arxiv.org/abs/2508.08160)

Exact quantum circuit synthesis for Matrix Product Unitaries (MPUs), implementing
the block-encoding + generalized oblivious amplitude amplification (GOAA)
construction of:

> G. Styliaris, R. Trivedi, J. I. Cirac, "Quantum Circuits for
> Matrix-Product Unitaries", *Phys. Rev. Lett.* **135**, 260602 (2025).
> [arXiv:2508.08160](https://arxiv.org/abs/2508.08160),
> [doi:10.1103/yshb-hmml](https://doi.org/10.1103/yshb-hmml).

**Statement of need.** Matrix Product Unitaries describe many-body
unitaries of bounded bond dimension, including quantum cellular automata
and unitaries beyond them. The tree decomposition above gives an exact
circuit for any MPU, but no implementation was available. Given an MPU's
tensors, TreeMPU certifies its unitarity for every system size, chooses
boundary operators that minimize the circuit cost, builds a standard
`qiskit.QuantumCircuit` that reproduces the operator exactly (up to a
global phase), verifies it, and predicts resources up to $N\sim10^5$
without building circuits. Uniform (translation-invariant) and
site-dependent MPUs are supported.

The derivations and numerical results behind the implementation are in the
author's master's thesis, *Beyond-QCA MPUs: Generation, Certification, and
Circuit Synthesis* (M. Vassallo, University of Bologna, 2026), available
from [AMS Laurea](https://amslaurea.unibo.it/) once published.

**Documentation:** [user guide](docs/userguide.md) ·
[tutorial notebooks](notebooks/) · [theory → code map](docs/theory_code_map.md)

## Contents

- [Key features](#key-features)
- [Background](#background)
- [Install](#install)
- [Quickstart](#quickstart)
- [Usage guide](#usage-guide)
- [Advanced examples](#advanced-examples)
- [Performance notes](#performance-notes)
- [Testing](#testing)
- [Related package: mpu_generators](#related-package-mpu_generators)
- [Status](#status)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Key features

- Exact circuit synthesis for MPUs via block-encoding + GOAA, returned as
  a standard `qiskit.QuantumCircuit`.
- `UniformMPU` (translationally invariant, one repeating bulk tensor) and
  `GeneralMPU` (fully heterogeneous, site-dependent tensors), sharing one
  synthesis engine (`CircuitDecomposition`).
- A family of boundary-operator constructions, from the paper's canonical
  choice up to a closed-form-plus-fit construction that reaches the
  conditioning number $q = D$ (no uniform boundary tested went lower;
  $q = D$ is not a bound: only $q \ge 1$ is proved).
- Exact unitarity certificate for every $N$ (checks up to $\chi^2+1$, thesis
  finite-size cutoff theorem).
- Exact cost estimators (`estimate_synthesis_stats`, `analytic_synthesis_stats`)
  that give the qubit count, gate count and depth (in units of ≤2-qubit
  blocks) of the circuit `synthesize()` builds, without building it, for
  any N (checked out to $N = 2^{14}$, 131 065 qubits, in minutes).
- Built-in structural/numerical verification at every stage (`checks`
  module): chain unitarity, boundary full rank, LCU reconstruction,
  block-encoding correctness.
- Canonicalization of general (non-uniform) chains to minimal bond
  dimension, plus a `block_mpu_tensors` helper for coarse-graining chains
  that don't yet satisfy the construction's structural assumptions.

## Background

A **Matrix Product Unitary (MPU)** is an $N$-site operator written as a
matrix product operator — one small tensor
$A[p_\text{in}, p_\text{out}, \chi_L, \chi_R]$ per site, contracted along
a virtual **bond** of dimension $D$, and capped at the two ends of the
chain by boundary vectors $\vec l, \vec r \in \mathbb{C}^D$ — that
happens to be *exactly* unitary as an operator on all $N$ physical sites,
not just approximately so. Global unitarity is a nontrivial,
chain-length-independent constraint on $(A, \vec l, \vec r)$, and TreeMPU
checks it explicitly (`check_mpo_unitarity` for the uniform, single-tensor
case; `compute_chain_boundary_operators`/`check_chain_unitarity` for a
fully general chain) rather than assuming it.

Turning that tensor description into a circuit needs **boundary (capping)
operators** $L, R$ — invertible matrices, one per internal bond
(Hermitian positive definite for the canonical constructions) — built by contracting the bulk tensor against a chosen
density matrix that closes the physical leg at the far end of the chain.
Dressed by $L$ and $R$, every local tensor becomes an honest isometry,
i.e. something a quantum circuit can implement directly. The paper's
canonical choice takes that density matrix to be maximally mixed
($I/d$); TreeMPU additionally exposes several strictly-better
constructions (see the [boundary strategies table](#boundary-construction-strategies)
below), all validated by an independent, exact isometry-residual check
before being trusted.

The **conditioning number**
$q(L,R) = \lVert R^{-\mathsf T}L^{-1}\rVert_F$ (equal to
$\sqrt{\operatorname{tr}[R^{-2}(L^{-2})^{\mathsf T}]}$ for Hermitian positive-definite
$L, R$) controls everything downstream: it *is* the LCU 1-norm $C$ of the
rank-one *merging operator* that block-encoding + GOAA has to implement at
every internal node of the synthesis tree, and it fixes the number of GOAA
rounds
$$
\ell = \min\Big\{k \in \mathbb{N} : q \le \csc\frac{\pi}{4k+2}\Big\},
$$
which is what actually sets the circuit cost,
$\operatorname{depth}(N) \propto N^{\log_2(2\ell+1)}$ and
$\operatorname{gates}(N) \propto N^{1+\log_2(2\ell+1)}$ — the exact
solution of the implemented recursion (the exact circuit depth ratio is
5.00001 per doubling at $N = 2^{14}$ for $\ell = 2$). The thesis proves
$q \ge 1$. By Cauchy–Schwarz, $q\,\lVert LR^\top\rVert_F \ge D$, and no
uniform boundary went below $q = D$ (270 validated optimized boundaries);
site-dependent windowed boundaries can: in our sweeps only at N = 4 with
`boundary_opt_p=1`, on the middle bond, whose two optimization windows both
end on the chain's boundary vectors (none at N = 5, nor at N = 8 with
`p = 1, 2`); e.g. $q = 1.66$ at $D = 2$, still with an exact circuit;
reproduced by
`tests/test_benchmarks.py::test_site_dependent_boundary_goes_below_chi`), so $q = D$ is a target, not a bound.
`solve_qD_boundary` gives a closed-form-plus-small-fit construction that
reaches $q = D$ on the generator families, also after a change of bond
basis (natural 21/21, real gauge 20/21, complex gauge 19/21 tested cases),
generalizing the fully closed-form `mode1_exact_boundary` available only
for a special ("Mode I") tensor family. When it does not converge it
raises, and `UniformMPU` falls back to the optimizer or to the exact
canonical boundary; every pair is validated exactly before use. Crossing one of the $\ell(q)$ staircase
thresholds above by reaching $q = D$ can move the circuit's cost by many
orders of magnitude — the gain sits in the scaling *exponent*, not a
constant factor, and blocking (coarse-graining sites together) leaves
$q = D$ unchanged, since it changes the physical dimension, not the
bond dimension.

Once boundaries are fixed, the **merging operator** is decomposed as a
linear combination of two unitaries (LCU, chosen optimally so its
1-norm matches $\lVert M \rVert_{\text{op}}$) and embedded into a larger
unitary via **block-encoding**; post-selecting an ancilla register on
$\lvert 0 \rangle$ implements the merging operator up to a known
normalization, and **GOAA** amplifies that success probability toward 1
using $\ell$ repeated Grover-like rounds. The whole $N$-site chain is
then assembled by merging site pairs in a binary tree with $\log_2 N$
layers; each layer replays its children $2\ell+1$ times, which gives the
depth $\propto N^{\log_2(2\ell+1)}$ and gate count $\propto N^{1+\log_2(2\ell+1)}$
above.

## Install

Requires Python ≥ 3.10. Not on PyPI; install from the repository:

```bash
git clone --recurse-submodules https://github.com/Mrcocvs02/TreeMPU.git
cd TreeMPU
pip install -r requirements.txt              # numpy, scipy, quimb, jax, qiskit (+ matplotlib, jupyter for the notebooks)
pip install -e . -e ./mpu_generators         # TreeMPU and the mpu_generators submodule
```

Extras: `pip install -e ".[dev]"` (pytest, ruff, mypy) and
`pip install -e ".[notebooks]"`. [`mpu_generators`](mpu_generators/) is a
git submodule (its own repository,
[MPU_Generation](https://github.com/Mrcocvs02/MPU_Generation)); it depends
on TreeMPU, so install TreeMPU first when installing it on its own.
Importing TreeMPU enables 64-bit precision in JAX; the optimizers run on
the CPU.

## Quickstart

```python
import numpy as np
from TreeMPU import UniformMPU

# The 2-site "MCZ" MPU: A[p_in, p_out, bond_L, bond_R], d=2, D=2.
A = np.zeros((2, 2, 2, 2), dtype=complex)
A[1, 1] = np.eye(2)
A[0, 0] = np.array([[1, 0], [0, 0]], dtype=complex)
l_vec = np.array([1, 1], dtype=complex)
r_vec = np.array([1, -2], dtype=complex)

mpu = UniformMPU(A=A, l_vec=l_vec, r_vec=r_vec, N=8)
circuit, sys_regs, bond_regs, ancilla_regs, all_ancillas = mpu.synthesize()

print(f"q = {mpu.q_unif:.4f}, GOAA rounds ell = {mpu.ell}")
print(f"{circuit.num_qubits} qubits, depth {circuit.depth()}")
# -> q = 2.0000, GOAA rounds ell = 1
# -> 36 qubits, depth 134   (composite gates)

from qiskit import transpile

flat = transpile(circuit, basis_gates=["u", "cx"], optimization_level=1)
print(flat.depth(), sum(flat.count_ops().values()))
# -> 2018 10086   (elementary gates; Qiskit 2.4, may vary slightly between versions)
```

## Usage guide

### Tensor convention

Every bulk tensor uses `A[p_in, p_out, bond_L, bond_R]` — physical input
leg, physical output leg, then the two virtual bond legs. `UniformMPU`
takes one such tensor and repeats it `N` times; `GeneralMPU` takes an
explicit list, one (possibly different) tensor per site.

### `UniformMPU` vs `GeneralMPU`

| | `UniformMPU` | `GeneralMPU` |
|---|---|---|
| Chain | single repeating tensor `A`, `N` sites | explicit list of `N` tensors, one per site |
| Constructor | `UniformMPU(A, l_vec, r_vec, N, ...)` | `GeneralMPU(tensors_A, l_vec, r_vec, ...)` |
| Boundary/cost computation | $O(1)$ (translation invariance) | $O(N)$, one left sweep + one right sweep (Styliaris et al., Lemma 1) |
| Structural requirement | `check_assumption_1` (length-1 boundary maps injective), unless a self-certifying boundary (`boundary_mode1_exact`/`boundary_opt_k`) is used | none by default — `canonicalize=True` brings the chain to minimal bond dimension first, making the assumption unnecessary |
| Conditioning number | `mpu.q_unif` (single scalar) | `mpu.q` (max over the per-bond `mpu._q_bonds`) |

### Boundary construction strategies

`UniformMPU` tries boundary constructions in priority order and records
the winner in `mpu.boundary_source`:

| `boundary_source` | Enabled by | What it does |
|---|---|---|
| `mode1_exact` | `boundary_mode1_exact=True` | Closed-form, zero-optimization; reaches $q=D$ exactly, but only for "Mode I" tensors with single-vertex readout ($\vec l = e_v$) |
| `qD_general` | `boundary_opt_k=<int>` | `solve_qD_boundary`: closed-form-plus-small-fit; reaches $q=D$ on the generator families (natural bond basis); raises otherwise, and the optimizer or the canonical boundary is used |
| `optimized` / `optimized_gauge` | `boundary_opt_k=<int>` (fallback), optionally + `boundary_opt_gauge=True` | `optimize_boundary_operators`: direct JAX optimization over invertible $L,R$ (optionally a further unitary gauge $G$), minimizing $q$ subject to isometry at prefix lengths $1..k$ |
| `canonical_sigma_tau` (default) | `boundary_sigma_tau=True` (default) | `optimize_q_unif`: JAX optimization over the canonical $\sigma,\tau$ density matrices |
| `canonical` | `boundary_sigma_tau=False` | The paper's baseline: maximally-mixed $\sigma=\tau=I/d$ |

Worth knowing: for `optimize_boundary_operators`, $q\ge1$ is **not**
automatic the way it is for the canonical construction — GOAA needs
$C\ge1$ — so every candidate is independently re-validated
(`_validate_boundary_pair`) before being trusted, and callers should
check `q_opt >= 1` themselves before relying on a result.

`GeneralMPU` exposes the per-bond analogues: `boundary_sigma_tau` (a
per-bond `optimize_chain_sigma_tau`) and `boundary_opt_p` ("partial
optimization": directly optimize a bond's boundary wherever a
power-of-two window reaches a true chain edge).

### Cost estimation before synthesis

```python
mpu.estimate_synthesis_stats()  # {"depth", "gate_count", "num_qubits"}, no circuit built
params = mpu.synthesis_cost_params()  # N-independent leaf data (JSON-serializable)
UniformMPU.analytic_synthesis_stats(params, N=2**12)  # the same numbers for any N
mpu.synthesize(pool_ancillas=True)  # recycle ancillas: one register per tree layer
```

Both estimators are **exact**: they build the same tree of leaf gates and
sub-circuit calls as `synthesize()` and schedule it (greedy per-qubit
layering, memoized on identical subtrees) without expanding it, so depth,
gate count and qubit count equal those of the flattened circuit
`synthesize()` returns, with or without `pool_ancillas`/`pool_levels`.
`analytic_synthesis_stats` rebuilds that tree for any N from the
N-independent data of `synthesis_cost_params` (N = 1024 in ~10 s).
Counts are in **≤2-qubit blocks** (arbitrary one- and two-qubit
unitaries), not transpiled `["u", "cx"]` gates: for the quickstart MCZ at
N=8 that is 10842 blocks, `transpile` gives 10086 gates. If Qiskit cannot
decompose a gate, a warning is emitted and the numbers are lower bounds.

Ancilla recycling (`pool_ancillas=True`, or `pool_levels=k` for the lowest
k layers) lets all merges of a tree layer share one lcu/rotation register:
an exact merge returns its ancillas to $|0\rangle$, so sibling subtrees can
reuse them one after the other. It saves qubits (MCZ, N = 16: 76 → 54) at
the price of depth, since siblings no longer run in parallel.

### Verification helpers (`TreeMPU.checks`)

| Function | Validates |
|---|---|
| `check_mpo_unitarity(N_max, A, l, r)` | Exact unitarity ($U U^\dagger = I$) of a uniform chain for lengths 1..`N_max`; `N_max = chi**2 + 1` certifies every length |
| `check_assumption_1(A, l, r)` | Injectivity (full rank $D$) of the length-1 boundary maps |
| `check_chain_unitarity(residual)` | Normalization $\mathrm{Tr}(U^\dagger U)/d^N = 1$ of a site-dependent chain (necessary, not sufficient) |
| `dense_unitary_from_tensors`, `reconstruct_circuit_unitary`, `phase_aligned_error` | Dense reference and exact circuit verification (small systems) |
| `mps_circuit_columns(qc, sys_regs, inputs)` | Selected columns of the circuit's operator by MPS simulation (quimb): circuit verification beyond the state-vector limit (36 qubits in ~2 min per column) |
| `block_isometry_residuals` | Isometry residuals of a boundary pair's blocks |
| `check_full_rank(M, label)` | A boundary operator is full rank (bond dimension is minimal) at a cut |
| `verify_lcu(M, coeffs, unitaries)` | LCU coefficients are real/positive, each term is unitary, and they reconstruct $M$ |
| `verify_merging_unitary(...)` | Post-selecting the block-encoding unitary on $\lvert 0\rangle$ reproduces $M/C$ |

Pass `debug=True` to `UniformMPU` to run `verify_lcu`/`verify_merging_unitary`
at construction time on the (unrotated, dense) LCU block encoding of the
merging operator. They do not check the gates the circuit actually uses;
use `reconstruct_circuit_unitary` (small circuits) or `mps_circuit_columns`
(wide circuits) for that.

## Advanced examples

### Reaching an exact target via an optimized, gauge-lifted boundary

Using the MCZ tensor from the quickstart (`A`, `l_vec`, `r_vec` above):

```python
from TreeMPU import UniformMPU

mpu = UniformMPU(
    A=A,
    l_vec=l_vec,
    r_vec=r_vec,
    N=2,
    boundary_opt_k=1,
    boundary_opt_gauge=True,
    boundary_opt_n_restarts=8,
    boundary_opt_seed=0,
)
assert mpu.boundary_source == "optimized_gauge"
qc, site_sys_regs, _, _, _ = mpu.synthesize()
# Reproduces diag([1, 1, 1, -1]) exactly.
```

This runs a JAX optimization with restarts (about 10 s on a laptop CPU);
it is not meant for a hot inner loop.

### `GeneralMPU` with genuinely site-dependent tensors

```python
import numpy as np
from TreeMPU import GeneralMPU


def _twist(A, W):
    """Apply a local unitary W after the site (multiplies the output leg by W)."""
    return np.einsum("jJ,iJab->ijab", W, A)


X = np.array([[0, 1], [1, 0]], dtype=complex)
tensors = [A, _twist(A, X)]  # A = the MCZ tensor above
mpu = GeneralMPU(tensors, l_vec, r_vec)
qc, site_sys_regs, site_bond_regs, anc_regs, all_ancillas = mpu.synthesize()
```

### Reaching $q = D$ from a randomly generated MPU (bridges to `mpu_generators`)

```python
from TreeMPU import UniformMPU
from mpu_generators import UniformMPUGenerator

gen = UniformMPUGenerator(seed=0, verify=True)
tensors, l_vec, r_vec = gen.fusion(
    d=3, bond_dim=3, n_blocks=2, skeleton="cyclic", blocking="auto", max_blocking=4
)
A = tensors[0]

mpu = UniformMPU(
    A=A,
    l_vec=l_vec,
    r_vec=r_vec,
    N=2,
    boundary_opt_k=3,
    boundary_opt_n_restarts=25,
    boundary_opt_seed=0,
)
assert mpu.boundary_source == "qD_general"  # reaches q = D, here D = 3
```

## Performance notes

- `light` (default `False`): caps the exact unitarity check performed at
  construction time — `False` checks lengths up to `N`, `True` up to
  $\min(N, \chi^2+1)$ (enough to certify every $N$), an `int k` up to `k`.
  The check costs 0.4 s / 9 MB at 32 sites and 32 s / 1.5 GB at 256, so
  `light=True` is cheap for small $\chi$ but expensive for large $\chi$
  (e.g. 197 sites for $\chi = 14$).
- `DENSE_BLOCK_ENCODING_MAX_QUBITS = 12` (module constant in `MPU.py`):
  the largest total width for which the block-encoding's controlled
  unitary is assembled as one dense matrix rather than gate by gate;
  above it, LCU terms are attached individually via `Gate.control()`.
- `block_mpu_tensors(tensors, block_size)` coarse-grains `block_size`
  consecutive sites into one, raising the physical dimension to
  `d**block_size` while keeping the bond dimension fixed — needed to
  satisfy `check_assumption_1` when the bond dimension exceeds what a
  single site's boundary map can resolve (e.g. tensors produced by
  `mpu_generators`, where $D$ can exceed $d^2$). **Caution**: the
  resulting physical dimension grows doubly exponentially in the
  blocking exponent — always predict the resulting tensor size
  analytically before calling this inside a swept loop.

## Testing

```bash
pytest -m "not slow"     # ~2 min
pytest                   # full suite, ~10 min (JAX optimizations, circuit reconstructions)
```

`pyproject.toml` sets `pythonpath = [".", "mpu_generators"]`, so the
suite runs correctly from the repo root without a separate install of
either package. Test files (`tests/`):

- `test_benchmarks.py` — reference values (MCZ: q = 2, ell = 1, circuit
  size), exact reconstruction, and the scaling law depth(2N)/depth(N) → 2ell+1.
- `test_adversarial.py`, `test_construction_stress.py`,
  `test_generalmpu_canonicalize.py`, `test_mode1_exact_boundary.py`,
  `test_generate_random_mpu.py` — padded registers, fallback routes,
  canonicalization, Mode I boundaries, generator integration.

- `test_MPU.py`, `test_GeneralMPU.py` — core synthesis correctness for
  both classes.
- `test_checks-utils.py` — the `checks` module plus boundary/LCU
  utilities.
- `test_goaa_rescaling.py` — GOAA rotation-angle and round-count (`ell`)
  correctness.
- `test_optimize_boundary_operators.py` — boundary optimizer
  correctness, including the gauge-lifted case and the degenerate
  $D=1$ case.
- `test_qD_boundary.py` — the general $q=D$ construction
  (`solve_qD_boundary`) and its `UniformMPU` integration.

## Related package: `mpu_generators`

[`mpu_generators`](mpu_generators/) (a git submodule here, developed as
its own package at [MPU_Generation](https://github.com/Mrcocvs02/MPU_Generation))
generates random MPU tensors that are unitary by construction (certified
algebraically for Modes I and II),
suitable as direct input to TreeMPU — see its own
[README](mpu_generators/README.md). The dependency between the two
packages is one-way: `mpu_generators` imports from `TreeMPU` (boundary
sweeps, canonicalization, blocking); `TreeMPU` never imports from it.

## Status

Alpha (0.2.0).

- `UniformMPU` — fully implemented: circuit synthesis, asymptotic and
  exact cost estimation, and the full family of boundary strategies
  above, including the `qD_general` construction reaching $q = D$.
- `GeneralMPU` — fully implemented: boundary operators via Lemma-1 chain
  sweeps, canonicalization, local isometries, circuit synthesis, and
  per-bond `boundary_sigma_tau`/windowed `boundary_opt_p` optimization.
- A few large or structurally adversarial configurations (very high
  bond dimension after certain fusion-type generative constructions)
  remain open cases where full dense reconstruction is currently
  intractable rather than merely slow; see the relevant module
  docstrings for the specific isometry-tolerance caveats involved.

## Citation

If you use this code, please cite the paper this implementation is based
on:

```bibtex
@article{Styliaris2025,
  title   = {Quantum Circuits for Matrix-Product Unitaries},
  author  = {Styliaris, Georgios and Trivedi, Rahul and Cirac, J. Ignacio},
  journal = {Physical Review Letters},
  volume  = {135},
  pages   = {260602},
  year    = {2025},
  doi     = {10.1103/yshb-hmml},
  eprint  = {2508.08160},
  archivePrefix = {arXiv}
}
```

and, once available, the author's thesis:

```bibtex
@mastersthesis{Vassallo2026,
  title  = {Beyond-QCA MPUs: Generation, Certification, and Circuit Synthesis},
  author = {Vassallo, Marco},
  school = {Alma Mater Studiorum -- Universit{\`a} di Bologna},
  year   = {2026}
}
```

## Contributing

Run `ruff check` and `pytest -m "not slow"` before submitting changes.
Issues and PRs are welcome; this is
thesis-adjacent research code, so expect the API to still move.

## License

MIT — see [LICENSE](LICENSE).
