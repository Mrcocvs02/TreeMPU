# mpu_decomposition

Quantum circuit synthesis for Matrix Product Unitaries (MPUs), implementing
the block-encoding + generalized oblivious amplitude amplification (GOAA)
construction of Styliaris et al. This is, to our knowledge, the first
working implementation of that algorithm.

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
from mpu_decomposition import UniformMPU

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
  tensor): fully implemented, including circuit synthesis and asymptotic
  cost estimation.
- `GeneralMPU` (heterogeneous, site-dependent MPUs): boundary operators,
  GOAA data and local isometries are computed per bond in the constructor;
  `synthesize()` is intentionally not yet enabled pending end-to-end
  verification against a genuinely non-uniform chain.

## License

MIT — see [LICENSE](LICENSE).
