# TreeMPU

Quantum circuit synthesis for Matrix Product Unitaries (MPUs), implementing
the block-encoding + generalized oblivious amplitude amplification (GOAA)
construction of:

> G. Styliaris, R. Trivedi, J. I. Cirac, "Quantum Circuits for
> Matrix-Product Unitaries", *Phys. Rev. Lett.* **135**, 260602 (2025).
> [arXiv:2508.08160](https://arxiv.org/abs/2508.08160),
> [doi:10.1103/yshb-hmml](https://doi.org/10.1103/yshb-hmml).

This is, to our knowledge, the first working implementation of that
algorithm.

The full derivations, proofs, and numerical results behind this
implementation are written up in the author's undergraduate thesis (Marco
Vassallo, Alma Mater Studiorum — Università di Bologna, expected graduation
September 2026), which will be downloadable from
[AMS Laurea](https://amslaurea.unibo.it/) by searching "Marco Vassallo"
once published.

## Install

```bash
git clone https://github.com/Mrcocvs02/MPU_Decomposition.git
cd MPU_Decomposition
pip install -e .
```

Not yet on PyPI.

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
```

## Status

Pre-alpha, under active development.

- `UniformMPU` (translationally invariant MPUs, single repeating bulk
  tensor): fully implemented, including circuit synthesis, asymptotic cost
  estimation, and an optional sigma/tau-optimized boundary baseline
  (`boundary_sigma_tau`).
- `GeneralMPU` (heterogeneous, site-dependent MPUs): boundary operators,
  GOAA data, local isometries and circuit synthesis (`synthesize()`) are all
  implemented, including an optional per-bond sigma/tau-optimized boundary
  baseline (`boundary_sigma_tau`) and windowed "partial optimization"
  (`boundary_opt_p`).

## License

MIT — see [LICENSE](LICENSE).
