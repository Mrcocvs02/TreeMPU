# Benchmark results

Produced by `python benchmarks/run_benchmarks.py` (2026-11-01): Python 3.13.11, NumPy 2.4.2,
SciPy 1.17.1, Qiskit 2.5.2, quimb 1.14.0, JAX 0.11.0 (CPU), 8-core laptop CPU (Intel Core Ultra 5 226V).
Times are indicative. The N=4 site-dependent chain (19 qubits, ell = 2) was started but its dense
reconstruction did not finish in 12 minutes, so the N=3 chain is reported.

## Exactness of the synthesized circuits

| MPU | q | ell | qubits | error | leakage | time (s) |
|---|---|---|---|---|---|---|
| MCZ, N=4 | 2.000 | 1 | 16 | 1.3e-14 | 2.7e-14 | 49 |
| branching_uniform (d=2, chi=2), N=3 | 2.064 | 2 | 17 | 1.1e-14 | 3.0e-14 | 224 |
| fusion_cyclic (d=2, chi=2), N=3 | 2.699 | 2 | 13 | 1.1e-14 | 2.8e-14 | 4 |
| fusion_nested_flag (d=2, chi=2), N=3 | 2.201 | 2 | 17 | 1.5e-14 | 3.9e-14 | 212 |
| grafted (d=2, chi=2), N=3 | 2.086 | 2 | 17 | 1.0e-14 | 2.4e-14 | 1913 |
| shared-basis ring (chi=3), 2 sites per block, N=2 blocks | 3.000 | 2 | 11 | 2.2e-14 | 4.8e-14 | 4 |
| cyclic Z_3 on qubits (chi=3 > d), seed 0, N=2 blocks | 7.581 | 6 | 11 | 5.6e-14 | 1.2e-13 | 11 |
| cyclic Z_3 on qubits (chi=3 > d), seed 7, N=2 blocks | 3.812 | 3 | 11 | 2.8e-14 | 5.7e-14 | 6 |
| site-dependent Mode II (GeneralMPU), N=3 | 2.699 | 2 | 13 | 4.1e-15 | 1.8e-15 | 3 |

## Scaling with the chain length

Counts are exact, in blocks acting on at most two qubits. `N` counts the tensors of the chain. Ratio = value(N)/value(N/2); the prediction for the depth is `2*ell + 1`.

### MCZ (q=2.00, ell=1)

| N | qubits | depth | blocks | depth ratio | block ratio |
|---|---|---|---|---|---|
| 4 | 16 | 526 | 1374 | - | - |
| 8 | 36 | 2071 | 10842 | 3.94 | 7.89 |
| 16 | 76 | 6915 | 72618 | 3.34 | 6.70 |
| 32 | 156 | 22075 | 457890 | 3.19 | 6.31 |
| 64 | 316 | 68587 | 2812794 | 3.11 | 6.14 |
| 128 | 636 | 210146 | 17070882 | 3.06 | 6.07 |
| 256 | 1276 | 638903 | 103003098 | 3.04 | 6.03 |
| 512 | 2556 | 1933350 | 619742850 | 3.03 | 6.02 |
| 1024 | 5116 | 5833059 | 3723611514 | 3.02 | 6.01 |

`ell` = 1, so the depth ratio tends to 3.

### Mode II cyclic, chi=2 (q=2.70, ell=2)

| N | qubits | depth | blocks | depth ratio | block ratio |
|---|---|---|---|---|---|
| 4 | 19 | 1952 | 5155 | - | - |
| 8 | 43 | 11114 | 62343 | 5.69 | 12.09 |
| 16 | 91 | 58262 | 675599 | 5.24 | 10.84 |
| 32 | 187 | 294450 | 7013311 | 5.05 | 10.38 |
| 64 | 379 | 1477964 | 71412735 | 5.02 | 10.18 |
| 128 | 763 | 7399612 | 720511583 | 5.01 | 10.09 |
| 256 | 1531 | 37015307 | 7237009279 | 5.00 | 10.04 |
| 512 | 3067 | 185111844 | 72529504671 | 5.00 | 10.02 |
| 1024 | 6139 | 925625468 | 726091995455 | 5.00 | 10.01 |

`ell` = 2, so the depth ratio tends to 5.

### Mode I chi=3, 2 sites per block (q=3, ell=2)

| N | qubits | depth | blocks | depth ratio | block ratio |
|---|---|---|---|---|---|
| 4 | 29 | 15516 | 43386 | - | - |
| 8 | 65 | 80932 | 541682 | 5.22 | 12.49 |
| 16 | 137 | 413038 | 5969194 | 5.10 | 11.02 |
| 32 | 281 | 2071914 | 62445818 | 5.02 | 10.46 |
| 64 | 569 | 10371952 | 638212666 | 5.01 | 10.22 |
| 128 | 1145 | 51884296 | 6450870362 | 5.00 | 10.11 |
| 256 | 2297 | 259467648 | 64852365754 | 5.00 | 10.05 |
| 512 | 4601 | 1297420120 | 650241856538 | 5.00 | 10.03 |
| 1024 | 9209 | 6487254032 | 6511009338106 | 5.00 | 10.01 |

`ell` = 2, so the depth ratio tends to 5.

## Baseline: generic unitary synthesis

CNOT counts after transpilation to `["u", "cx"]`. `qs_decomposition` receives the dense unitary and uses no MPU structure (it does exploit a diagonal matrix, which is why MCZ is cheap for it). The generic MPU is a Mode I tensor (`branching_uniform`, d=2, chi=2, seed 0) with `boundary_opt_k=1` (q=2, ell=1); TreeMPU's cost depends only on (d, chi, ell), so MCZ gives the same TreeMPU numbers.

| N | QSD, generic MPU | QSD, MCZ | TreeMPU (any chi=2, ell=1 MPU) | TreeMPU qubits |
|---|---|---|---|---|
| 2 | 2 | 1 | 42 | 6 |
| 3 | 18 | 6 | 290 | 11 |
| 4 | 94 | 18 | 534 | 16 |
| 5 | 423 | 50 | 2074 | 21 |
| 6 | 1781 | 130 | 2318 | 26 |
| 7 | 7317 | 322 | 3354 | 31 |
| 8 | 29655 | 770 | 4102 | 36 |
| 9 | 119383 | 1794 | - | 41 |

