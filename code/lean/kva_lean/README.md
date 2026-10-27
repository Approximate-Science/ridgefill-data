# KVA lemmas in Lean 4 + Mathlib

Formalisation of eleven lemmas (plus a shared helper module) for a paper on KVA, a training-free prefill
accelerator for long prompts. All proofs compile with **zero `sorry`/`admit`**,
and every main theorem depends only on the standard axioms
`[propext, Classical.choice, Quot.sound]` (see `#print axioms` output below).

## Files

| Lemma | File | Main Lean theorems |
|---|---|---|
| 1. GDN prefix-error contraction | `KvaLean/GdnContraction.lean` | `frob_gdn_step_le`, `gdn_prefix_contraction` (helpers: `frob`, `frob_sq`, `frob_smul`, `gdn_entry`, `frob_gdn_step_sq`) |
| 2. Speedup monotonicity and bound | `KvaLean/Speedup.lean` | `kva_speedup_mono_S`, `kva_speedup_mono_T`, `kva_speedup_le_bound`, `kva_speedup_eq_bound_iff`, `kva_speedup_ge_one` (+ projector-cost: `kva_speedupP_le`, `kva_speedupP_ge_one`) |
| 3. Running sums = batch ridge | `KvaLean/RidgeSums.lean` | `ridge_two_batch`, `ridge_posDef`, `ridge_isUnit_det`, `ridge_K_batch`, `gram_weighted`, `gram_discount` (helpers: `gram_fromRows`, `target_fromRows`, `gram_stack`, `target_stack`, `stackBatches`) |
| 4. Uniform logit shift | `KvaLean/SoftmaxShift.lean` | `softmax_cond_on_unchanged`, `softmax_cond_off_unchanged`, `softmaxMass_shift_mono`, `softmaxMass_shift_strict` |
| 5. Attention received bounds influence | `KvaLean/AttnSensitivity.lean` | `attn_value_exact`, `attn_value_total`, `attn_key_exact`, `attn_key_weight_le`, `attn_key_bound`, `attn_joint_bound`, `attn_joint_total`, `attn_zero_attention_no_influence` (softmax helpers: `softmaxW_shift_delta`, `softmaxW_shift_abs_le`, `softmaxW_shift_tv`, `softmaxW_shift_on_le`, `abs_shiftFactor_le`, `shift_core_ineq`, `attnWeight_pos`) |
| 6. Impact ≤ error × sensitivity | `KvaLean/ImpactBound.lean` | `impact_le`, `firstOrder_le`, `impact_keep_exact`, `topSet_compl_sum_le`, `impact_keep_top` |
| 7. Cross terms of the joint change | `KvaLean/CrossTerms.lean` | `quad_joint_eq_general`, `quad_joint_eq`, `perToken_gap`, `quad_joint_eq_needs_symm`, `crossTerm_abs_le`, `crossTerm_abs_le_sq`, `crossTerm_abs_le_frob` (helpers: `sum_split_at`, `sum_sum_split`, `quadF_add_sub`, `dotProduct_mulVec_symm`, `vnorm_mulVec_le_frob`) |
| 8. GDN correction with exact rows (rho) | `KvaLean/StCorrection.lean` | `glaState_closed`, `st_correction_exact`, `st_scale_unique`, `st_unscaled_overcorrects`, `rhoE1_empty`, `rhoE1_all`, `rhoE1_le_geom`, `rhoE1_fast_forget_numeric`, `gdn_tail_exact_bound` (notes: `notes/StCorrection.md`) |
| 9. QK-norm erases key-size errors | `KvaLean/QkNormInvariance.lean` | `rmsNorm_smul_pos`, `qk_attnOut_rescale`, `rmsNorm_eq_dir`, `rmsNorm_dir_lipschitz_max`, `qk_attn_dir_bound`, `rmsNormEps_sub_le`, `vnorm_rmsNorm_eq`, `abs_vnorm_rmsNorm_ratio_sub_one_le` (notes: `notes/QkNormInvariance.md`) |
| 10. Speedup with exact rows | `KvaLean/SpeedupExactRows.lean` | `kva_speedupR_anti_R`, `kva_speedupR_anti_premium`, `kva_speedupR_ge_one_iff`, `kva_speedupR_ge_one_iff_le_breakEven`, `kva_speedupR_ge_one_iff_calib`, `kva_speedupR_lt_one_of_large`, `kva_speedupR_ge_one_of_unit_premium` (notes: `notes/SpeedupExactRows.md`) |
| 11. Count-matched random control | `KvaLean/RandomControl.lean` | `randomGain_eq`, `skill_eq`, `skill_pairwise`, `unmatched_control_bias`, `skill_top_max`, `sum_skill_kSubsets`, `sum_skill_relabel`, `exists_ge_avgSet` (notes: `notes/RandomControl.md`) |
| shared helper | `KvaLean/VecNorm.lean` | `vnorm` (Euclidean norm as a plain sum), `abs_dotProduct_le` (Cauchy–Schwarz), `vnorm_smul`, `vnorm_add_le`, `vnorm_sub_le`, `vnorm_sum_le` |

`KvaLean.lean` imports all modules.

## Theorems in plain math

### Lemma 1 — GDN prefix-error contraction
- `frob A = √(∑ i, ∑ j, (A i j)²)` (Frobenius norm; Mathlib has no
  Frobenius norm on matrices, so it is defined directly — this is the
  standard definition).
- Key step (`frob_gdn_step_le`): for unit `k` (`∑ j, (k j)² = 1`) and
  `β ∈ [0, 2]`, `‖A·(I − βkkᵀ)‖_F ≤ ‖A‖_F`.
  Proof by the identity `‖B‖² = ‖A‖² − β(2−β)·‖Ak‖²` (`frob_gdn_step_sq`).
- Main (`gdn_prefix_contraction`): two trajectories with the same inputs
  `(α, β, k, v)` satisfy
  `‖S T − S′ T‖_F ≤ (∏_{t<T} α t)·‖S 0 − S′ 0‖_F`.
- Deviations: `β` is proved for the **stronger** range `[0,2]` (task asks
  `[0,1]`, notes `[0,2]` suffices); the product is over `range T`
  (`t = 0..T−1`), which equals `∏_{t=1}^{T}` by reindexing; states are
  `Matrix m n ℝ` with `[Fintype m] [Fintype n] [DecidableEq n]`
  (`DecidableEq` is needed to write the matrix `1`); per-step hypotheses
  `α t ∈ [0,1]`, `β t ∈ [0,2]`, `k t` unit.

### Lemma 2 — speedup monotonicity and bound
With `C_exact = L·N·c`, `C_kva = S·N·c + (L−S)·(T·c + (N−T)·f)`,
`speedup = C_exact / C_kva`:
- (a) `kva_speedup_mono_S`: non-increasing in `S`;
  `kva_speedup_mono_T`: non-increasing in `T`.
- (b) `kva_speedup_le_bound`: `speedup ≤ L/S`;
  `kva_speedup_eq_bound_iff`: equality iff `S = L ∨ (T = 0 ∧ f = 0)`.
- (c) `kva_speedup_ge_one`: `speedup ≥ 1`.
- Projector cost `p ≥ 0` per skipped token (`kvaCostKvaP`, `kvaSpeedupP`):
  `kva_speedupP_le` (projector only hurts) and `kva_speedupP_ge_one`
  (needs `p ≤ (L−S)·(c−f)`).
- Deviations: all quantities are **reals** (the paper's naturals cast to
  `ℝ`); everything is proved under `0 ≤ f ≤ c`, which is **weaker** than
  the paper's `0 ≤ f < c` (so the results are slightly stronger).
  Standing hypotheses: `1 ≤ S ≤ L`, `0 ≤ T ≤ N`, `0 < N`, `0 < c`.

### Lemma 3 — running sums equal the batch ridge solution
- Two batches (`ridge_two_batch`, via `gram_fromRows`, `target_fromRows`
  from Mathlib's block-matrix API):
  `(XᵀX + λI)⁻¹XᵀY = (X₁ᵀX₁ + X₂ᵀX₂ + λI)⁻¹(X₁ᵀY₁ + X₂ᵀY₂)`
  with `X = fromRows X₁ X₂`, `Y = fromRows Y₁ Y₂`.
- `K` batches (`ridge_K_batch`): same identity with batches stacked over
  the sigma type `Σ k, m k` (`stackBatches`) and `∑ k` running sums.
- Forgetting factor (`gram_weighted`, `gram_discount`): scaling batch `k`
  by `√(w k)` turns the plain Gram sum into the `w`-weighted one; with
  `w k = γ^(K−k)` this is discounted least squares.
- Invertibility (`ridge_posDef`, `ridge_isUnit_det`): `XᵀX` is PSD
  (from `Matrix.posSemidef_conjTranspose_mul_self`, using `ᴴ = ᵀ` over
  `ℝ`), so `XᵀX + λ•1` is positive definite for `λ > 0` and its
  determinant is a unit.
- Deviations: `λI` is written `lam • 1`; the stacked-vs-sums **equality
  itself needs no invertibility** (pure rewrite of the Gram/target
  splits) — invertibility is proved as a separate theorem, which is what
  makes the inverses meaningful.

### Lemma 4 — uniform shift on a subset of logits
Softmax over a finite type, shift `z′ i = z i + c` on `P`, unchanged off `P`:
- (a) `softmax_cond_on_unchanged`: `exp(z′ i)/∑_{j∈P} exp(z′ j)` unchanged
  for `i ∈ P`; (b) `softmax_cond_off_unchanged`: same over `Pᶜ`.
- (c) `softmaxMass_shift_mono`: total mass on `P` is non-decreasing in `c`;
  `softmaxMass_shift_strict`: strictly increasing if `P` and `Pᶜ` are
  both non-empty.
- Deviations: the partition is `P` vs its complement `Pᶜ` (equivalent to
  a `P`/`Q` partition); the monotone/strict theorems need `[Nonempty ι]`
  so the softmax denominator is positive (if `ι` is empty all masses are
  `0/0 = 0` and the statements hold trivially, but the proof goes through
  cross-multiplication).

### Notation for Lemmas 5–7 (`VecNorm.lean`)
Vectors are plain functions `Fin d → ℝ`. Inner product is Mathlib's
`dotProduct`, `⟪u, v⟫ = u ⬝ᵥ v = ∑ k, u k * v k`; the norm is defined directly
as `‖u‖ = vnorm u = √(∑ k, (u k)²)`. `VecNorm.lean` proves Cauchy–Schwarz
`|⟪u, v⟫| ≤ ‖u‖‖v‖`, `‖c•u‖ = |c|‖u‖` and the triangle inequality (two
vectors and finite sums).

### Lemma 5 — the attention a token receives bounds its influence
Explains empirical findings **(ii)** (impact is ranked by sensitivity, not by
error size) and **(iii)** (the attention a token receives predicts its
impact): a skipped token's key/value error reaches the outputs only through
the attention weights `a i j0` it receives.

Setup: queries `q i` (`i : Fin m`), keys `k j`, values `v j` (`j : Fin n`),
scores `s i j = c·⟪q i, k j⟫`, weights `a i j = exp(s i j)/∑_l exp(s i l)`
(`softmaxW`, `attnWeight`), outputs `o i = ∑ j, a i j • v j` (`attnOut`).
Perturbations are `Function.update v j0 (v j0 + δ)` and
`Function.update k j0 (k j0 + δ)`; a key perturbation shifts only the logit
`s i j0`, by `ε i = c·⟪q i, δ⟫` (`attnScore_update_on/off`).
- (a) values, exact: `attn_value_exact`: `o' i − o i = a i j0 • δ`;
  `attn_value_total`: `∑ i, ‖o' i − o i‖ = (∑ i, a i j0)·‖δ‖`.
- (b) keys. Single-logit shift of a softmax row, exact
  (`softmaxW_shift_delta`): `a' j − a j = ρ·(𝟙[j = j0] − a j)` with
  `ρ = exp(z j0)(exp ε − 1)/Z'` (`Z'` the new denominator), and
  `|ρ| ≤ a j0·(exp|ε| − 1)` (`abs_shiftFactor_le`). Consequences:
  - `attn_key_weight_le`: `|a' i j − a i j| ≤ a i j0·(exp|ε i| − 1)` for every `j`;
  - `softmaxW_shift_tv`: `∑ j, |a' j − a j| ≤ 2·a j0·(1 − a j0)·(exp|ε| − 1)`;
  - `attn_key_exact`: `o' i − o i = ρ i • (v j0 − o i)` (the output moves
    along the line to `v j0`);
  - `attn_key_bound`: if `‖v j‖ ≤ M` for all `j`,
    `‖o' i − o i‖ ≤ 2·M·a i j0·(exp|ε i| − 1)`.
- (c) key and value together:
  - `attn_joint_bound`: `‖Δo i‖ ≤ a i j0·(exp|ε i|·‖δv‖ + 2M·(exp|ε i| − 1))`;
  - `attn_joint_total`: if `|ε i| ≤ E` for all `i`,
    `∑ i, ‖Δo i‖ ≤ (∑ i, a i j0)·(exp E·‖δv‖ + 2M·(exp E − 1))`, i.e. the
    total influence is linear in the attention received;
  - `attn_zero_attention_no_influence`: if `a i j0 = 0` for all `i`, no output
    changes.
- Deviations / notes:
  - **(c) as literally phrased is vacuous**: softmax weights are strictly
    positive (`attnWeight_pos`), so `a i j0 = 0` never holds for exact,
    unmasked softmax. The theorem is proved as stated, and the non-vacuous
    content is `attn_joint_total` (influence → 0 linearly as received
    attention → 0). A zero weight would need a mask, which is not modelled.
  - `c > 0` is never used; every result holds for any real `c`.
  - `M` is any bound with `‖v j‖ ≤ M` for all `j` (e.g. `max_j ‖v j‖`),
    rather than the max itself.
  - The key bound is stated with the Euclidean norm. `SoftmaxShift.lean` is
    not reused: the exact single-logit formula `softmaxW_shift_delta` gives
    both the per-weight and the output bounds directly.

### Lemma 6 — impact ≤ error × sensitivity, and concentration
Explains empirical findings **(i)+(ii)**: with errors spread evenly
(`‖δ j‖ ≤ e`) the impact of a token set is bounded by `e` times its share of
the sensitivity mass `∑ ‖g j‖`, so concentrated sensitivity gives
concentrated impact.
- (a) `impact_le`: `|⟪g, δ⟫| ≤ ‖g‖·‖δ‖`; `firstOrder_le`:
  `|∑ j, ⟪g j, δ j⟫| ≤ ∑ j, ‖g j‖·‖δ j‖`.
- (b) `impact_keep_exact`: if `δ j = 0` for `j ∈ S` and `‖δ j‖ ≤ e` for all
  `j`, then `|∑ j, ⟪g j, δ j⟫| ≤ e·∑_{j∉S} ‖g j‖`.
- (b, choice of `S`) `topSet_compl_sum_le`: if `w ≥ 0` and `S` holds the
  largest weights (a threshold `τ` with `w j ≥ τ` on `S`, `w j ≤ τ` off `S`),
  then `∑_{j∉S} w j ≤ ∑_{j∉T} w j` for every `T` with `#T ≤ #S`.
  `impact_keep_top` combines the two with `w j = ‖g j‖`: keeping the top
  gradient-norm set exact gives the smallest bound of this form among all
  sets of the same or smaller size.
- Deviations: none in (a)/(b). "Largest gradient norms" is formalised by the
  threshold condition, which also covers ties.

### Lemma 7 — cross terms: why per-token sums miss the joint change
Explains empirical finding **(iv)**: per-token impacts do not add up to the
joint loss change; the gap is exactly the pairwise interaction terms. (The
lemma gives the identity and a size bound; it does not prove the sign or the
measured ~2.5× factor.)

`f x = ½⟪x, H x⟫ + ⟪b, x⟫` (`quadF`) on `Fin N → ℝ`, pieces `δ j`
(`j : Fin n`), cross term `crossTerm H δ = ∑_{j<k} ⟪δ j, H δ k⟫`
(written `∑ j, ∑ k ∈ Ioi j`).
- `quad_joint_eq` (needs `H.IsSymm`):
  `f(x + ∑ j, δ j) − f x = ∑ j, (f(x + δ j) − f x) + ∑_{j<k} ⟪δ j, H δ k⟫`.
- `perToken_gap`: joint change − per-token sum = cross term.
- `quad_joint_eq_general` (any `H`): same identity with the symmetrised cross
  term `½∑_{j<k} (⟪δ j, H δ k⟫ + ⟪δ k, H δ j⟫)`.
- `crossTerm_abs_le`: if `‖H u‖ ≤ L‖u‖` for all `u`,
  `|cross| ≤ L·∑_{j<k} ‖δ j‖‖δ k‖`; `crossTerm_abs_le_sq`: the same as
  `(L/2)·((∑ j, ‖δ j‖)² − ∑ j, ‖δ j‖²) = (L/2)·∑_{j≠k} ‖δ j‖‖δ k‖`;
  `crossTerm_abs_le_frob`: with `L = ‖H‖_F` (`vnorm_mulVec_le_frob`, `frob`
  from Lemma 1).
- Deviations:
  - **Added hypothesis `H` symmetric** in the `∑_{j<k} ⟪δ j, H δ k⟫` form.
    Without it the identity is false; `quad_joint_eq_needs_symm` is a proved
    counterexample (`H = !![0,1;0,0]`, `δ = e₀, e₁`: true gap `½`, stated
    cross term `1`). A Hessian is symmetric, so this costs nothing; the
    general-`H` identity is also proved.
  - The pieces `δ j` need not have disjoint supports (tokens as blocks are a
    special case); the identity holds for any decomposition.
  - The cross-term bound is a factor 2 **sharper** than the requested
    `‖H‖·∑_{j≠k}‖δ j‖‖δ k‖` (it sums over `j < k` only). The operator norm is
    a hypothesis `‖H u‖ ≤ L‖u‖`; the Frobenius norm is one valid `L`.
  - No general C² Taylor bound: only the quadratic identity is proved.

## Toolchain

- Lean `4.34.1` (commit `5045d0056413266e57c625dcd7c365b10e377c52`),
  Lake `5.0.0-src+5045d00`, elan `4.2.4`.
- Mathlib `v4.34.1` (manifest pins `d13f23b723b8a846827a245b89c10fc7d3f11612`
  plus its transitive deps in `lake-manifest.json`).
- `lean-toolchain` says `leanprover/lean4:v4.34.1`.

## How to build

Elan lives inside the workspace at `../.elan` (i.e. `<workspace>/.elan`),
so every command needs:

```sh
cd kva_lean
export ELAN_HOME="$PWD/../.elan"
export PATH="$ELAN_HOME/bin:$PATH"
```

First-time setup (already done here; `lake new kva_lean math` fetched the
cache automatically, and `lake exe cache get` confirms it):

```sh
lake exe cache get   # expect: "No files to download" / "Already decompressed 8908 file(s)"
lake build           # must succeed with no errors and no warnings
```

`grep -rn "sorry\|admit" KvaLean/` returns nothing.

## Axiom check (`#print axioms`, from the build log)

Every theorem below reports exactly `[propext, Classical.choice, Quot.sound]`
— in particular no `sorryAx`:

- `frob_gdn_step_le`, `gdn_prefix_contraction`
- `kva_speedup_mono_S`, `kva_speedup_mono_T`, `kva_speedup_le_bound`,
  `kva_speedup_eq_bound_iff`, `kva_speedup_ge_one`
- `ridge_two_batch`, `ridge_posDef`, `ridge_isUnit_det`, `ridge_K_batch`,
  `gram_weighted`, `gram_discount`
- `softmax_cond_on_unchanged`, `softmax_cond_off_unchanged`,
  `softmaxMass_shift_mono`, `softmaxMass_shift_strict`
- `softmaxW_shift_delta`, `softmaxW_shift_abs_le`, `softmaxW_shift_tv`,
  `attn_value_exact`, `attn_value_total`, `attn_key_exact`,
  `attn_key_weight_le`, `attn_key_bound`, `attn_joint_bound`,
  `attn_joint_total`, `attn_zero_attention_no_influence`
- `impact_le`, `firstOrder_le`, `impact_keep_exact`, `topSet_compl_sum_le`,
  `impact_keep_top`
- `quad_joint_eq_general`, `quad_joint_eq`, `perToken_gap`,
  `quad_joint_eq_needs_symm`, `crossTerm_abs_le`, `crossTerm_abs_le_sq`,
  `crossTerm_abs_le_frob`

Lemmas 5–7 (2026-10-02, `lake build` from `lean/kva_lean` with
a local `ELAN_HOME`): 3116 jobs,
no errors, no warnings; elaboration time per new file (`lake env lean`,
CPU): `VecNorm` 1.4 s, `AttnSensitivity` 2.4 s, `ImpactBound` 1.3 s,
`CrossTerms` 1.4 s. All 23 `#print axioms` lines in the three new files
report `[propext, Classical.choice, Quot.sound]`, e.g.:

```text
info: KvaLean/AttnSensitivity.lean:...: 'attn_joint_total' depends on axioms: [propext, Classical.choice, Quot.sound]
info: KvaLean/ImpactBound.lean:...: 'impact_keep_exact' depends on axioms: [propext, Classical.choice, Quot.sound]
info: KvaLean/CrossTerms.lean:...: 'quad_joint_eq' depends on axioms: [propext, Classical.choice, Quot.sound]
Build completed successfully (3116 jobs).
```

Build log tail for Lemmas 1–4 (earlier full build, `touch` + `lake build`
forcing a rebuild of those four modules):

```text
info: ... 'gram_weighted' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ... 'gram_discount' depends on axioms: [propext, Classical.choice, Quot.sound]
ℹ [3106/3108] Replayed KvaLean.SoftmaxShift
info: ... 'softmax_cond_on_unchanged' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ... 'softmax_cond_off_unchanged' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ... 'softmaxMass_shift_mono' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ... 'softmaxMass_shift_strict' depends on axioms: [propext, Classical.choice, Quot.sound]
Build completed successfully (3108 jobs).
```

with zero warnings and zero errors; `grep -rn "sorry\|admit" KvaLean/`
is empty.

### Lemmas 8-11 (2026-10-04)
Added after the Flash-Next run; each file's plain-math statement, the measurement it backs, its assumptions and what it doesn't cover are in `notes/<File>.md`. All four build with `lake build` (3,120 jobs) with no `sorry` and only `propext`, `Classical.choice`, `Quot.sound`.
