# Differences between this library and the published thesis

The thesis *Beyond-QCA MPUs: generation, certification, and circuit synthesis*
(M. Vassallo, University of Bologna, 2026;
<https://amslaurea.unibo.it/id/eprint/39893>) was published before the code
audit that led to version 0.3.0. This page lists the places where the
published text and the code or documentation of this repository disagree.
**Where they differ, the code and the documentation of this repository are
authoritative.** Line numbers refer to the LaTeX sources of the thesis, not
to the PDF; labels are the LaTeX labels also cited in the docstrings.

## 1. Statements that need correction

| # | Where | Published statement | Correct statement |
|---|---|---|---|
| 1 | `MPU_Decomposition.tex`, remark *Ancilla management* (l. 966-968) | Merge ancillas "can be recycled after a merging layer is completed"; the number of simultaneously active auxiliary systems is proportional to the active blocks of the layer. | Registers of *different* layers cannot be shared: a parent's reflection $R_\Psi=\tilde V\,\mathrm{Ref}_0\,\tilde V^\dagger$ replays its children while the parent's own ancillas are in use. Merges of the *same* layer can share one lcu/rotation register if they run one after the other (`synthesize(pool_ancillas=True)`), at the price of depth. The ancilla count is then $\sum_{\rm layers}(n_{\rm lcu}+n_{\rm rot})=O(\log N)$ registers; without sharing it is one register per merge ($N-1$). |
| 2 | `Numerical_results.tex`, resource counting (l. 1184) | "Ancilla registers are allocated one per merging tree layer", qubit count $n_{\rm total}=n_{\rm system}+\sum_{\rm layers}n_{\rm ancilla}$. | The qubit counts in the tables were produced by the estimator without any pooling option, i.e. one register per merge ($N-1$ registers). The per-layer count of the text is what `pool_ancillas=True` builds (e.g. MCZ, $N=16$: 76 qubits per merge, 54 per layer). The numbers should be read as per-merge counts, or recomputed with pooling (which also changes depth, because siblings no longer run in parallel). |
| 3 | `Generating_MPUs.tex`, Mode I sampler (l. 331-339) | $A[v,u_i]=W_1P_iW_2^\dagger$ with coordinate projectors, described as equivalent to the implemented sampler. | The generators sample $A[v,u_i]=V\,P_i\,W^\dagger$ with Haar $V,W$ and $P_i$ the projector on the $i$-th block of columns of $W$. Both satisfy row resolution for every draw, but the laws differ (checked numerically by comparing the two samplers). All numerical results of the thesis refer to the implemented sampler. |
| 4 | `Generating_MPUs.tex`, nested-flag skeleton (l. 640-645, 717-741) and the MCZ paragraph | Increasing-rank labels, closure $\min(u,v)$, identity last. | The code and the tensor-first paper use rank $P_a=d-a$, closure $\max(u,v)$, identity first. The two conventions are related by the relabelling $a\mapsto m-a$ and are equivalent if applied consistently to ranks, boundary coefficients, fusion table and accessibility indices. |
| 5 | `Generating_MPUs.tex`, grafted composition and the `cyclic` skeleton | Stated for the regular-representation skeleton. | The code uses the `cyclic` skeleton ($\mathbb Z_M$ characters). It coincides with the regular representation (group convolution) only when the number of channels $M$ divides $d$; otherwise $\operatorname{Tr}y_g^\dagger y_h\ne\delta_{gh}\,d$ and the genericity identity $S(\ell)=d^2(1-\sum_g|\ell_g|^4)$ does not apply. |
| 6 | `MPU_Decomposition.tex`, canonical boundaries (l. 109, 152-235) | $\sigma,\tau$ are written on "the physical leg" of the operator-valued tensor. | The density matrices sit on the physical **input** leg and the output leg is traced; the local isometry maps $p_{\rm in}\to(p_{\rm out},\text{ext})$. Environments on the output leg are equivalent only under a no-signalling condition on $U^\dagger$ and fail, e.g., for the cyclic shift. |
| 7 | `Analytical_results.tex`, gauge orbit and general formula for $q$ (l. 1394-1450) | Formulas written without specifying which tensor copy is conjugated. | For complex tensors the conjugation matters: the site isometries contract the second axis of $L$ and $R$, so the environments are the Hermitian square roots of the *conjugated* contracted $L^2$, $R^2$. With the unconjugated formulas $q$ is not invariant under a complex bond gauge (0.198 versus 0.449 in a test); the library uses the conjugated form. |
| 8 | `MPU_Decomposition.tex`, reflection projector $P_S$ (l. 798-803) | $P_S$ projects on all auxiliary inputs, including the ancillas of earlier merges. | This is correct. The implementation reflects only the current merge's registers (LCU, rotation, bonds), which is equivalent because every encoding built here shares the row $e_0$ (property BE\*) and every tree block is an isometry. For a generic block encoding of the same merging operator the narrow reflection is *not* exact. An inductive argument is given in the user guide, section 6; it is not a labelled result of the thesis. |
| 9 | `thm:local_isometry` and the partial-optimization discussion | Isometry of the local blocks. | The theorem covers the blocks $\ell AR$, $LAR$, $LAr$ of length 1 and, by the environment argument, every length for pairs built from environments (canonical, $\sigma/\tau$, gauge of the MPU). For optimized or qD boundaries, longer blocks are isometric only to solver precision and this is validated numerically, not proved; `GeneralMPU` warns when a merge-tree segment of an optimized chain has a residual above $10^{-6}$ (`segment_isometry_residuals`). |

The thesis proves $q\ge1$ for admissible boundaries and does not claim
$q\ge\chi$; statements that $q=\chi$ is a target (not a bound) appear only in
this repository's documentation.

## 2. Numerical results computed with the pre-audit code

The σ/τ optimizer (`optimize_q_unif`) contracted $\sigma,\tau$ on the output
leg while the boundary operators built from them use the input leg. It
therefore minimized a different function, and the default `UniformMPU` could
end with a larger $q$ than the plain $I/d$ boundary. The following results
were computed with it and may change when regenerated with version 0.3.0:

* `tab:num-dec-t4-blocking-optimization` and its discussion
  (`Numerical_results.tex`, around l. 1435-1455): the `canonical_sigma_tau`
  column ($q_{\sigma\tau}$ from 4.12 to 156) and the conclusion that it is
  "not a reliable way to remove the conditioning blow-up". With the fixed
  optimizer σ/τ can only improve on $I/d$; the conclusion is most likely an
  artifact of the bug.
* `tab:num-dec-conditioning` (column $q^{\rm density}_{\rm opt}$) and
  `tab:num-dec-ablation` (density arm): values may decrease.
* `tab:num-dec-reconstruction`: rows labelled `canonical_sigma_tau` stay
  exact; the selected boundary, hence $\ell$ and the resources, can change.
* Resources of gauge-optimized boundaries with $q$ at $2\pm10^{-8}$: $\ell$
  may drop from 2 to 1 (a solver-noise tolerance, `GOAA_TARGET_RTOL`, now
  identifies such values with the target).
* Qubit counts: see item 2 above.

All generator-based numbers (tensors, boundaries, certificates) are
bit-for-bit unchanged: the audited generators reproduce the pre-audit outputs
exactly for every family and seed tested.
