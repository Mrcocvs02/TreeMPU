# Theory → code map

Where each construction of the thesis (*Beyond-QCA MPUs: Generation,
Certification, and Circuit Synthesis*, chapters "MPU decomposition",
"Analytical results", "Generating MPUs") and of the tensor-first sampling
paper ("Tensor-First Sampling of Exact Matrix-Product Unitaries") is
implemented. Labels are the LaTeX `\label`s of the sources. **Status**:
*implemented*, *partial* (see note), *not implemented*.

## TreeMPU — circuit synthesis (thesis ch. "MPU decomposition")

| Theory | Label | Code | Status |
|---|---|---|---|
| Uniform MPU | `def:uniform_mpu`, `eq:uniform_mpu` | `UniformMPU` | implemented |
| Canonical boundaries (σ on the input leg, output traced) | `eq:partial_trace_unitarity`, `eq:L_R_algebraic` | `UniformMPU._compute_boundary_operators`, `utils.optimize_q_unif` (σ, τ optimized) | implemented |
| Local isometries $V=LAR$ | `thm:local_isometry`, `eq:isometry_definition` | `UniformMPU.create_local_isometries`, `CircuitDecomposition._build_isometry_unitary` | implemented |
| Assumption 1 (full rank) | `ass:full_rank` | `checks.check_assumption_1` (injective length-1 boundary maps, needed by the canonical `UniformMPU` boundaries); `utils.satisfies_assumption_1` checks the related condition of full-rank boundary operators at every cut (needed by `GeneralMPU(canonicalize=False)`) | implemented |
| Merging operator $M=\lvert00\rangle\langle I\rvert(R^{-1}\otimes L^{-1})$ | `eq:merging_operator`, `lem:merging_identity` | `utils.get_merging_operator` (kernel $R^{-\top}L^{-1}$) | implemented |
| LCU decomposition, state preparation $B$, controlled unitary | `lem:lcu_decomposition`, `eq:state_preparation`, `eq:controlled_unitary`, `eq:lcu_unitary` | `CircuitDecomposition._compute_lcu`, `_compute_merging_unitary`, `_build_U_block_encoding` (closed-form $W_{\rm ctrl}$ route not described in the thesis) | implemented |
| Reflections, subspace amplitude amplification | `lem:reflections`, `eq:R_phi`, `eq:R_psi`, `thm:subspace_aa` | `_make_reflection_zero`, `_merge_layer` | implemented |
| Exact amplification angle, $\ell$, rotation qubits | `rem:exact_amplification_angle`, `eq:modified_C`, `rem:amplification_gap`, `prop:multi_ancilla_amplification` | `CircuitDecomposition._compute_rotation_params`, `_build_rotated_lcu`; target tolerance `GOAA_TARGET_RTOL` | implemented |
| $M$ rank one, $\lVert M\rVert_1=q$ | `lem:M_rank_one`, `eq:M_trace_norm`, `eq:q_unif_LR` | `utils.conditioning_number`, `UniformMPU.q_unif` | implemented |
| Conditioning number, optimized | `def:conditioning_number_v2`, `eq:q_unif_optimized` | `utils.optimize_boundary_operators`, `utils.solve_qD_boundary`, `utils.mode1_exact_boundary` | implemented |
| $\ell$ bound, merging depth | `eq:ell_bound`, `eq:merging_depth_explicit` | `_compute_rotation_params`, `analytic_synthesis_stats` | implemented |
| Depth theorem, recursion $T_k\le(2\ell+1)T_{k-1}$ | `thm:uniform_depth`, `eq:depth_recursion_v2`, `rem:exact_recursion_bound` | `CircuitDecomposition.analytic_synthesis_stats`, `estimate_synthesis_stats`, `synthesis_cost_params` | implemented |
| Tree merging, ancilla management | `sec:tree_merging` | `_tree_merge_plan`, `synthesize` | partial: ancillas are not recycled in the built circuit (only in the analytic estimator, `pool_ancillas=True`) |
| Non-uniform MPUs, $q_k$, $q=\max_k q_k$ | `eq:nonuniform_isometry`, `eq:qk_definition`, `eq:q_nonuniform`, `thm:nonuniform_depth` | `GeneralMPU` | implemented |
| Canonical form $L_k=\mathbb 1$, $R_k=\mathrm{diag}(s)$ | `sec:nonuniform_mpus` | `utils.to_canonical_form` (`GeneralMPU(canonicalize=True)`) | implemented |
| Schmidt bound $q\le\sqrt\chi/s_{\min}$ | `thm:schmidt_bound`, `eq:schmidt_bound` | computable from `to_canonical_form`; no dedicated function | partial |

## TreeMPU — thesis ch. "Analytical results"

| Theory | Label | Code | Status |
|---|---|---|---|
| Spectral-flatness unitarity test $\mathcal P_N$ | `eq:PN_definition`, `prop:cauchy_schwarz_unitarity` | `checks.check_mpo_unitarity` (+ normalization $\mathrm{Tr}\,\rho=d^N$) | implemented |
| Finite-size cutoff $\chi^2+1$ | section "Finite-size cutoff" | `UniformMPU(light=True)`, attribute `unitarity_certified_all_N` | implemented |
| Blocking | section "Blocking" | `utils.block_mpu_tensors` | implemented; the blocking cost model $R_N(m)$ is not |
| Lifting the injectivity constraint | section "Lifting…" | `utils.optimize_boundary_operators` (HPD $e^{H}$ parametrization) | implemented |
| Gauge optimization, $q$ orbit | section "Gauge optimization…" | `optimize_boundary_operators(gauge=True)` (unitary gauge only; thesis allows $GL(\chi)$) | partial |
| Partial optimization (level-dependent $L_p$, $R_p$) | section "Partial optimization" | `GeneralMPU(boundary_opt_p=…)` (windows anchored at a chain end only) | partial |
| Staircase compilation, windowed compression | `sec:staircase-compilation`, `sec:windowed-compression` | — | not implemented (external scripts) |

## mpu_generators — thesis ch. "Generating MPUs" and the paper

| Theory | Label | Code | Status |
|---|---|---|---|
| Mode I, row resolution | `thm:mode1-unitarity`, `eq:mode1-condition`; paper `thm:main1` | `MPUGenerator.branching`, `base._row_resolution_blocks`, `_check_row_resolution` | implemented (balanced rank partition, fixed out-degree) |
| Combined / cascaded readout | `cor:mode1-readout`, `rem:cascaded-readout` | `branching(readout=…)`, `_combined_readout_sites` | implemented |
| Row duplication | `rem:row-duplication` | `UniformMPUGenerator.degenerate_rows` | implemented |
| Translation-invariant specialization | `rem:TI-specialization` | `UniformMPUGenerator` | implemented |
| CNOT staircase witness | `ex:cnot-prefix-rr`; paper χ=2 witness | `UniformMPUGenerator.group_automaton("Zn", 2)` | implemented |
| Mode II, fusion closure + boundary | `eq:mode2-closure`, `thm:mode2-unitarity`, `eq:mode2-boundary`; paper `eq:fusion-closure` | `MPUGenerator.fusion` | implemented |
| Group convolution, Peter–Weyl boundary | `ex:group-convolution`; paper `sec:peterweyl` | `fusion(skeleton="group")`, `_group_fourier_boundary` | partial: abelian groups and $S_3$ only |
| Nested flags, matrix units | table of skeletons; paper `sec:mode2` | `fusion(skeleton="nested_flag" / "matrix_units")` | implemented |
| Dressing invariance | `prop:mode2-dressing-invariance` | per-site Haar dressing in `fusion` | implemented (no undressed option) |
| Mode II-B, MPO product tensor | `eq:mpo-product-tensor` | `skeletons.mpo_product_tensor` | implemented |
| Bond-resolved intertwiner | `eq:intertwiner` | `skeletons.solve_intertwiner`, `verify_intertwiner` | partial: null-complement and boundary-compatibility conditions not checked |
| Lee–Yang worked instance | `sec:lee-yang-worked` | `skeletons.lee_yang_skeleton(alpha, beta)` | implemented |
| Grafted composition | `thm:grafted-factorization`, `rem:grafted-right-boundary` | `MPUGenerator.grafted` (cyclic skeleton) | implemented |
| Operator product of chains | paper, composition remark | `mixing.stack` | implemented |
| Mode II causal multipliers, $S(\ell)$, Mode I four-copy transfer, $\chi=2$ measure-zero theorem | paper `thm:mode2-causal`, `thm:genericity`, `prop:four-copy`, `thm:chi2` | — (analysis scripts outside the libraries) | not implemented |
