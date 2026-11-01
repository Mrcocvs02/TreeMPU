"""Benchmarks of TreeMPU: exactness, resource scaling, and a generic-synthesis baseline.

Usage: ``python benchmarks/run_benchmarks.py [exact|scaling|baseline|all] > benchmarks/RESULTS.md``

* ``exact``: the synthesized circuit is compared with the dense unitary contracted
  from the tensors, for the generator families, the multi-controlled-Z MPU and a
  site-dependent chain (``GeneralMPU``). Reported: conditioning number ``q``, GOAA
  rounds ``ell``, qubits, error of the circuit up to a global phase and leakage out
  of the ancilla-zero subspace.
* ``scaling``: exact depth / block count / qubit number of the circuit for growing
  chain length (``estimate_synthesis_stats`` for small ``N``,
  ``analytic_synthesis_stats`` beyond), with the ratio per doubling of ``N`` and the
  limit ``2*ell + 1`` that the analysis predicts.
* ``baseline``: the MCZ unitary ``diag(1, ..., 1, -1)`` on ``N`` qubits synthesized by
  Qiskit's generic unitary synthesis (``qs_decomposition``, no MPU structure used)
  versus TreeMPU, counted in CNOT gates after transpilation to ``["u", "cx"]``.
  MCZ has far cheaper hand-made circuits; it is used because its dense unitary is
  known in closed form, not as a claim of superiority.
"""

import os
import sys
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from qiskit import transpile  # noqa: E402
from qiskit.synthesis import qs_decomposition  # noqa: E402

from mpu_generators import MPUGenerator, UniformMPUGenerator, build_family  # noqa: E402
from TreeMPU import (  # noqa: E402
    GeneralMPU,
    UniformMPU,
    dense_unitary_from_tensors,
    phase_aligned_error,
    reconstruct_circuit_unitary,
)
from TreeMPU.MPU import CircuitDecomposition  # noqa: E402
from TreeMPU.utils import block_mpu_tensors  # noqa: E402


def mcz():
    """Return the MCZ bulk tensor and boundary vectors (U = diag(1, ..., 1, -1))."""
    A = np.zeros((2, 2, 2, 2), dtype=complex)
    A[0, 0] = np.diag([1, 0])
    A[1, 1] = np.eye(2)
    return A, np.array([1, 1], dtype=complex), np.array([1, -2], dtype=complex)


def exactness_cases():
    """Yield ``(label, build)``; ``build()`` returns ``(mpu, tensors, l, r)`` (``tensors`` for the dense reference)."""

    def uniform(A, l, r, n, **kw):
        return UniformMPU(A=A, l_vec=l, r_vec=r, N=n, light=True, **kw), [A] * n, l, r

    A, l, r = mcz()
    yield "MCZ, N=4", lambda: uniform(A, l, r, 4)
    for fam in ("branching_uniform", "fusion_cyclic", "fusion_nested_flag", "grafted"):
        t, l2, r2 = build_family(fam, d=2, bond_dim=2, seed=0)
        yield f"{fam} (d=2, chi=2), N=3", lambda t=t, l2=l2, r2=r2: uniform(t[0], l2, r2, 3)
    A3, l3, r3 = UniformMPUGenerator(seed=0).shared_basis_automaton(D=3, d=2)
    Ab = block_mpu_tensors([A3] * 2, 2)[0]
    yield (
        "shared-basis ring (chi=3), 2 sites per block, N=2 blocks",
        lambda: uniform(Ab, l3, r3, 2, boundary_mode1_exact=True),
    )
    for seed in (0, 7):
        t, l4, r4 = UniformMPUGenerator(seed=seed).fusion(
            d=2, bond_dim=3, n_blocks=2, skeleton="cyclic", blocking="auto"
        )
        yield (
            f"cyclic Z_3 on qubits (chi=3 > d), seed {seed}, N=2 blocks",
            lambda t=t, l4=l4, r4=r4: uniform(t[0], l4, r4, 2),
        )

    def general():
        t, lg, rg = MPUGenerator(seed=0).fusion(d=2, bond_dim=2, n_sites=3)
        return GeneralMPU(t, lg, rg), t, lg, rg

    yield "site-dependent Mode II (GeneralMPU), N=3", general


def run_exact(only=None):
    """Print the exactness table (``only``: restrict to the cases whose label contains it, no header)."""
    if only is None:
        print("## Exactness of the synthesized circuits\n", flush=True)
        print("| MPU | q | ell | qubits | error | leakage | time (s) |")
        print("|---|---|---|---|---|---|---|")
    for label, build in exactness_cases():
        if only is not None and only not in label:
            continue
        t0 = time.time()
        mpu, tensors, l, r = build()
        qc, sys_regs, *_ = mpu.synthesize()
        try:
            U, leak = reconstruct_circuit_unitary(qc, sys_regs)
        except ValueError:  # more than 24 qubits: no state-vector reconstruction
            print(
                f"| {label} | - | - | {qc.num_qubits} | not reconstructible (> 24 qubits) | - | - |",
                flush=True,
            )
            continue
        err = phase_aligned_error(U, dense_unitary_from_tensors(tensors, l, r))
        q = getattr(mpu, "q_unif", None) or getattr(mpu, "q", float("nan"))
        ell = CircuitDecomposition._compute_rotation_params(q)[0]
        print(
            f"| {label} | {q:.3f} | {ell} | {qc.num_qubits} "
            f"| {err:.1e} | {leak:.1e} | {time.time() - t0:.0f} |",
            flush=True,
        )
    print()


def scaling_families():
    """Return ``{label: (A, l, r)}`` with different numbers of GOAA rounds."""
    A, l, r = mcz()
    out = {"MCZ (q=2.00, ell=1)": (A, l, r, {})}
    t, l2, r2 = build_family("fusion_cyclic", d=2, bond_dim=2, seed=0)
    out["Mode II cyclic, chi=2 (q=2.70, ell=2)"] = (t[0], l2, r2, {})
    A3, l3, r3 = UniformMPUGenerator(seed=0).shared_basis_automaton(D=3, d=2)
    out["Mode I chi=3, 2 sites per block (q=3, ell=2)"] = (
        block_mpu_tensors([A3] * 2, 2)[0],
        l3,
        r3,
        {"boundary_mode1_exact": True},
    )
    return out


def run_scaling():
    """Print depth, block count and qubits for growing N, with the doubling ratios."""
    print("## Scaling with the chain length\n")
    print(
        "Counts are exact, in blocks acting on at most two qubits. `N` counts the tensors "
        "of the chain. Ratio = value(N)/value(N/2); the predicted limits are "
        "`2*ell + 1` for the depth and `2*(2*ell + 1)` for the block count.\n"
    )
    for label, (A, l, r, kw) in scaling_families().items():
        base = UniformMPU(A=A, l_vec=l, r_vec=r, N=4, light=True, **kw)
        params = base.synthesis_cost_params()
        print(f"### {label}\n")
        print("| N | qubits | depth | blocks | depth ratio | block ratio |")
        print("|---|---|---|---|---|---|")
        prev = None
        for k in range(2, 11):
            N = 2**k
            st = UniformMPU.analytic_synthesis_stats(params, N)
            row = f"| {N} | {st['num_qubits']} | {st['depth']} | {st['gate_count']} |"
            if prev:
                row += f" {st['depth'] / prev['depth']:.2f} | {st['gate_count'] / prev['gate_count']:.2f} |"
            else:
                row += " - | - |"
            print(row, flush=True)
            prev = st
        print(
            f"\n`ell` = {base.ell}, so the depth ratio tends to {2 * base.ell + 1} and the block ratio to {2 * (2 * base.ell + 1)}.\n"
        )


def run_baseline():
    """Print CNOT counts of generic unitary synthesis versus TreeMPU."""
    print("## Baseline: generic unitary synthesis\n")
    print(
        'CNOT counts after transpilation to `["u", "cx"]`. `qs_decomposition` receives the '
        "dense unitary and uses no MPU structure (it does exploit a diagonal matrix, which is why "
        "MCZ is cheap for it). The generic MPU is a Mode I tensor (`branching_uniform`, d=2, "
        "chi=2, seed 0) with `boundary_opt_k=1` (q=2, ell=1); TreeMPU's cost depends only on "
        "(d, chi, ell), so MCZ gives the same TreeMPU numbers.\n"
    )
    print("| N | QSD, generic MPU | QSD, MCZ | TreeMPU (any chi=2, ell=1 MPU) | TreeMPU qubits |")
    print("|---|---|---|---|---|")
    t, l, r = build_family("branching_uniform", d=2, bond_dim=2, seed=0)
    A = t[0]
    for N in range(2, 10):
        U = dense_unitary_from_tensors([A] * N, l, r)
        cx_gen = (
            transpile(qs_decomposition(U), basis_gates=["u", "cx"], optimization_level=0)
            .count_ops()
            .get("cx", 0)
        )
        Um = np.diag([1.0] * (2**N - 1) + [-1.0]).astype(complex)
        cx_mcz = (
            transpile(qs_decomposition(Um), basis_gates=["u", "cx"], optimization_level=0)
            .count_ops()
            .get("cx", 0)
        )
        mpu = UniformMPU(A=A, l_vec=l, r_vec=r, N=N, light=True, boundary_opt_k=1)
        st = mpu.estimate_synthesis_stats()
        cx_tree = "-"
        if N <= 8:
            qc, *_ = mpu.synthesize()
            cx_tree = (
                transpile(qc, basis_gates=["u", "cx"], optimization_level=1)
                .count_ops()
                .get("cx", 0)
            )
        print(f"| {N} | {cx_gen} | {cx_mcz} | {cx_tree} | {st['num_qubits']} |", flush=True)
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "exact_case":  # a single case of the exactness table, by label substring
        run_exact(only=sys.argv[2])
    if which in ("exact", "all"):
        run_exact()
    if which in ("scaling", "all"):
        run_scaling()
    if which in ("baseline", "all"):
        run_baseline()
