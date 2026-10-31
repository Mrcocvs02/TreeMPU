# Changelog — TreeMPU

## 0.3.0 (2026-10-22)

Second correctness pass (complex tensors, estimators, boundary solvers),
real ancilla recycling, exact resource estimates, and documentation to the
standard of an open-source physics library. All generator outputs are
unchanged (bit for bit); circuit outputs change only where listed under
*Fixed*.

### Added
- `synthesize(pool_ancillas=..., pool_levels=...)`: ancilla recycling in the
  built circuit, one lcu/rotation register per tree layer (merges of one layer
  run sequentially on shared qubits; registers of different layers cannot be
  shared). The estimators take the same options.
- `checks.mps_circuit_columns`: columns of a circuit's operator by MPS
  simulation (quimb), for circuits beyond the state-vector limit.
- `GeneralMPU.segment_isometry_residuals()`: isometry residual of every
  merge-tree segment; `GeneralMPU` warns when windows accepted by
  `boundary_opt_p` leave a segment residual above 1e-6.
- `docs/thesis_errata.md`: differences between this library and the published
  thesis; `CITATION.cff`; this changelog; continuous integration (ruff and the
  fast test suite).
- `mpu_generators` 0.3.0: hand-picked and random digraphs and rank partitions
  for Mode I (`digraph`, `out_degree`, `ranks`); `coherent_in_n=False` for an
  independent family per chain length.

### Changed
- `estimate_synthesis_stats` and `analytic_synthesis_stats` are now **exact**:
  depth, <=2-qubit block count and qubit number equal those of the circuit
  `synthesize` builds (depth was off by 1-11 %). `synthesis_cost_params`
  returns a new format (leaf blocks of every N-independent gate and of its
  inverse); parameters from 0.2.0 are rejected with an explicit error.
  `analytic_synthesis_stats` rebuilds the tree: about ten seconds at N = 1024.
- `synthesize` expands segment programs (leaf gates and calls to child
  segments) shared with the estimators; qubit bookkeeping is keyed by value.
- `optimize_chain_sigma_tau` places the free density matrices on the physical
  input leg (output traced), as `optimize_q_unif` and the canonical
  construction do; the output-leg variant was exact only for no-signalling
  MPUs. `GeneralMPU` accepts these boundaries at the exact tolerance.
- `solve_qD_boundary` starts inside the positive-definite part of its solution
  family, refines once in a whitened bond basis and raises instead of
  returning a pair that fails its own conditions.
- `GeneralMPU(boundary_opt_gauge=True)` is documented as unsupported (it
  raises together with `boundary_opt_p`).

### Fixed
- Canonical boundary environments conjugated the wrong copy of a complex
  tensor (`UniformMPU`, `compute_chain_boundary_operators`,
  `optimize_chain_sigma_tau`): circuits were wrong in a complex bond basis.
- The bond-space isometry term of `solve_qD_boundary` and
  `optimize_boundary_operators` used the wrong Gram-matrix convention and
  the output leg; both failed after a change of bond basis.
- `GeneralMPU` with `boundary_sigma_tau=True` could adopt boundaries whose
  merge-tree segments were not isometric, giving an inexact circuit without
  an error.
- `pool_ancillas` of the estimator modelled one shared register instead of
  one per layer.
- The closed-form block encoding falls back (with a warning) to the dense
  route when Qiskit's state preparation rejects a nearly degenerate row
  (bonds with q of order 1e2).
- **Inexact circuits from Qiskit's `StatePreparation`.** The closed-form
  block encoding prepares the merging row with `StatePreparation`, which is
  accurate to rounding on generic rows but returned a wrong state (distance
  up to 1, and 1e-9 to 1e-10 in milder cases) for rows with entries of order
  1e-8 to 1e-10, with no error or warning. Seen for blocked Mode II `cyclic`
  tensors with `bond_dim > d` (circuit error 0.5 and 1.0 for 2 of 10 seeds,
  1e-9 for most others) and, at the 1e-9 level, for padded Mode I bonds
  (`d=2, D=3`, `d=3, D=3`); the families with `bond_dim <= d` and unpadded
  bonds tested here were accurate to 1e-15. The prepared vector is now
  simulated (up to 12 qubits) and, if it is off by more than 1e-11, replaced
  by a dense unitary completion (`qs_decomposition`), which is exact; no
  other change in the closed form was needed (verified by substituting an
  exact `V`). Circuits for the affected bonds change; all others are
  unchanged.
- Normalization check of `check_mpo_unitarity`: absolute floor for complex
  rounding (`NORMALIZATION_ATOL`).
- Declared dependency minimums were wrong: `synth_mcx_1_dirty_kg24` only
  exists from qiskit 2.1, and quimb < 1.8 does not import with current scipy.
  Now `qiskit>=2.1`, `quimb>=1.8` (tested with qiskit 2.1.2, quimb 1.8.0).

### Removed
- Private helpers of the old rectangle-composition estimator (`_stat_seq`,
  `_merge_layer_stats`, `_make_segment_list_stats`, `_leaf_stats_pair`,
  `_schedule_stats_pair`, `_bond_u_stats`) and the fitted reflection-cost
  formula `_ref0_stats_analytic`.

### Documentation
- Numpydoc docstrings and why-not-what comments throughout; conventions
  (input-leg isometry, transposed `L`/`R` storage, merging kernel, qubit
  order) stated in every module docstring; status of each claim stated as
  proved, proved under hypotheses, or validated numerically (user guide,
  section 6); known limitations and roadmap in the README; AI usage
  disclosure.

## 0.2.0 (2026-09-27)

Correctness audit against the thesis ("MPU decomposition", "Analytical
results"). Every change that alters numerical output is listed under
*Changed results*; all other outputs are bit-for-bit identical to 0.1.0
(checked by `tests/golden/`).

### Fixed (changed results)
- **σ/τ boundary optimization** (`optimize_q_unif`): σ and τ were contracted
  on the *output* physical leg while the boundary operators built from them
  use the *input* leg (thesis, canonical boundaries). The optimizer therefore
  minimized a different function (e.g. reported q = 2.000, actual 2.218), and
  the default `UniformMPU` path could end with a *larger* q than plain I/d
  (3.44 → 4.00, 12.9 → 19.9 on tested instances). The contraction is fixed
  and the σ/τ result is adopted only if q decreases.
- **Conditioning number**: `q` is now defined once,
  `conditioning_number(L, R) = ‖R⁻ᵀL⁻¹‖_F`, the LCU 1-norm of the merging
  operator the circuit uses (identical to √tr[R⁻²(L⁻²)ᵀ] for Hermitian
  positive-definite boundaries). The gauge optimizer minimizes and reports
  this quantity for the gauge-dressed pair it returns; previously it reported
  the q of the undressed pair (differences ~1e-7).
- **GOAA target tolerance** (`GOAA_TARGET_RTOL = 1e-7`): a merge norm within
  1e-7 of a target C′(ℓ) = csc(π/(4ℓ+2)) is identified with it, so solver
  noise (q = 2 ± 1e-8) no longer changes ℓ or adds a rotation qubit with a
  ~1e-8 angle. All affected circuits reconstruct U_N exactly
  (errors 1e-13–1e-16, N = 2, 4); none uses more gates than before.
- **Unitarity check** (`check_mpo_unitarity`): also requires
  Tr(UU†) = d^N. The spectral-flatness ratio alone accepts U = c·(unitary)
  for any c > 0 (e.g. d^(−N/2)·I, which a test fixture relied on).

### Changed
- `UniformMPU(light=True)` checks lengths 1…min(N, χ²+1) instead of 1…8;
  by the thesis finite-size cutoff theorem this certifies every N. New
  attribute `unitarity_certified_all_N`.
- Resource estimates warn when Qiskit cannot decompose a gate (the estimate
  is then a lower bound).

### Added
- `conditioning_number`, `checks.block_isometry_residuals`,
  `checks.dense_unitary_from_tensors`, `checks.reconstruct_circuit_unitary`,
  `checks.phase_aligned_error`; `__version__`; more names exported from the
  package (`solve_qD_boundary`, `mode1_exact_boundary`, `to_canonical_form`,
  `optimize_chain_sigma_tau`, `dilate_isometry_to_unitary`).
- Tests: golden regression fixtures, audit-claim tests, block-isometry tests.

### Removed
- `CircuitDecomposition.exact_synthesis_stats` (unused, untested; use
  `estimate_synthesis_stats` or `synthesis_cost_params`) and dead private
  helpers.

### Documentation
- Numpydoc docstrings for the public API; conventions (index order, L/R
  storage, merging kernel, qubit order, global phase) stated once in the
  module docstrings; development-history notes removed from docstrings;
  the claim "q ≥ χ is a proven floor" corrected (it relies on
  ‖LRᵀ‖_F = 1, observed for all tested admissible pairs but not proved).

## 0.1.0
Initial release (as `mpu_decomposition`, renamed `TreeMPU`).
