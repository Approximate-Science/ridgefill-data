import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Data.Matrix.ColumnRowPartitioned
import Mathlib.Algebra.Order.Star.Real
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# Running sums equal the batch ridge solution

For stacked data `X = [X₁; X₂]`, `Y = [Y₁; Y₂]` and `λ > 0`:
`(XᵀX + λI)⁻¹XᵀY = (X₁ᵀX₁ + X₂ᵀX₂ + λI)⁻¹(X₁ᵀY₁ + X₂ᵀY₂)`.

Generalised to `K` batches stacked over a sigma type, and to a forgetting
factor `γ ∈ (0,1]` via discounted sums (weighted least squares).
Invertibility follows from `λ > 0` and positive semidefiniteness.
-/

open Matrix

/-- The Gram matrix of vertically stacked data splits into a sum. -/
theorem gram_fromRows {m₁ m₂ n : Type*} [Fintype m₁] [Fintype m₂]
    (X₁ : Matrix m₁ n ℝ) (X₂ : Matrix m₂ n ℝ) :
    (fromRows X₁ X₂)ᵀ * (fromRows X₁ X₂) = X₁ᵀ * X₁ + X₂ᵀ * X₂ := by
  rw [transpose_fromRows, fromCols_mul_fromRows]

/-- The target matrix of vertically stacked data splits into a sum. -/
theorem target_fromRows {m₁ m₂ n p : Type*} [Fintype m₁] [Fintype m₂]
    (X₁ : Matrix m₁ n ℝ) (X₂ : Matrix m₂ n ℝ)
    (Y₁ : Matrix m₁ p ℝ) (Y₂ : Matrix m₂ p ℝ) :
    (fromRows X₁ X₂)ᵀ * (fromRows Y₁ Y₂) = X₁ᵀ * Y₁ + X₂ᵀ * Y₂ := by
  rw [transpose_fromRows, fromCols_mul_fromRows]

/-- Two-batch ridge identity: solving on stacked data equals solving on running sums. -/
theorem ridge_two_batch {m₁ m₂ n p : Type*} [Fintype m₁] [Fintype m₂]
    [Fintype n] [DecidableEq n]
    (X₁ : Matrix m₁ n ℝ) (X₂ : Matrix m₂ n ℝ)
    (Y₁ : Matrix m₁ p ℝ) (Y₂ : Matrix m₂ p ℝ) (lam : ℝ) :
    ((fromRows X₁ X₂)ᵀ * (fromRows X₁ X₂) + lam • 1)⁻¹
        * ((fromRows X₁ X₂)ᵀ * (fromRows Y₁ Y₂))
      = (X₁ᵀ * X₁ + X₂ᵀ * X₂ + lam • 1)⁻¹ * (X₁ᵀ * Y₁ + X₂ᵀ * Y₂) := by
  rw [gram_fromRows, target_fromRows]

/-- `XᵀX` is positive semidefinite (over `ℝ`, `ᴴ = ᵀ`). -/
theorem posSemidef_transpose_mul_self_real {m n : Type*} [Fintype m] [Finite n]
    (X : Matrix m n ℝ) : (Xᵀ * X).PosSemidef := by
  have h := posSemidef_conjTranspose_mul_self X
  rwa [conjTranspose_eq_transpose_of_trivial] at h

/-- The regularised normal matrix `XᵀX + λI` is positive definite for `λ > 0`. -/
theorem ridge_posDef {m n : Type*} [Fintype m] [Finite n] [DecidableEq n]
    (X : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam) :
    (Xᵀ * X + lam • (1 : Matrix n n ℝ)).PosDef :=
  Matrix.PosDef.posSemidef_add (posSemidef_transpose_mul_self_real X)
    (Matrix.PosDef.one.smul hlam)

/-- Hence its determinant is a unit (the normal equations are solvable). -/
theorem ridge_isUnit_det {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]
    (X : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam) :
    IsUnit (Xᵀ * X + lam • (1 : Matrix n n ℝ)).det := by
  rw [isUnit_iff_ne_zero]
  exact ne_of_gt (ridge_posDef X hlam).det_pos

/-- Vertically stack `K` batches into one matrix over the sigma type. -/
noncomputable def stackBatches {K : ℕ} {m : Fin K → Type*} {n : Type*}
    (Xf : ∀ k : Fin K, Matrix (m k) n ℝ) : Matrix (Σ k, m k) n ℝ :=
  Matrix.of fun i j => Xf i.1 i.2 j

/-- The Gram matrix of `K` stacked batches is the sum of per-batch Grams. -/
theorem gram_stack {K : ℕ} {m : Fin K → Type*} [∀ k, Fintype (m k)]
    {n : Type*} (Xf : ∀ k : Fin K, Matrix (m k) n ℝ) :
    (stackBatches Xf)ᵀ * (stackBatches Xf)
      = ∑ k, ((Xf k)ᵀ * (Xf k)) := by
  ext a b
  simp only [Matrix.mul_apply, Matrix.transpose_apply, stackBatches,
    Matrix.of_apply, Matrix.sum_apply]
  rw [Fintype.sum_sigma]

/-- The target matrix of `K` stacked batches is the sum of per-batch targets. -/
theorem target_stack {K : ℕ} {m : Fin K → Type*} [∀ k, Fintype (m k)]
    {n p : Type*} (Xf : ∀ k : Fin K, Matrix (m k) n ℝ)
    (Yf : ∀ k : Fin K, Matrix (m k) p ℝ) :
    (stackBatches Xf)ᵀ * (stackBatches Yf)
      = ∑ k, ((Xf k)ᵀ * (Yf k)) := by
  ext a b
  simp only [Matrix.mul_apply, Matrix.transpose_apply, stackBatches,
    Matrix.of_apply, Matrix.sum_apply]
  rw [Fintype.sum_sigma]

/-- `K`-batch ridge identity: stacked solve equals running-sums solve. -/
theorem ridge_K_batch {K : ℕ} {m : Fin K → Type*} [∀ k, Fintype (m k)]
    {n p : Type*} [Fintype n] [DecidableEq n]
    (Xf : ∀ k : Fin K, Matrix (m k) n ℝ)
    (Yf : ∀ k : Fin K, Matrix (m k) p ℝ) (lam : ℝ) :
    ((stackBatches Xf)ᵀ * (stackBatches Xf) + lam • 1)⁻¹
        * ((stackBatches Xf)ᵀ * (stackBatches Yf))
      = (∑ k, ((Xf k)ᵀ * (Xf k)) + lam • 1)⁻¹
        * (∑ k, ((Xf k)ᵀ * (Yf k))) := by
  rw [gram_stack, target_stack]

/-- Scaling rows by `√w` turns a plain Gram sum into a `w`-weighted one. -/
theorem gram_weighted {K : ℕ} {m : Fin K → Type*} [∀ k, Fintype (m k)]
    {n : Type*} (Xf : ∀ k : Fin K, Matrix (m k) n ℝ)
    (w : Fin K → ℝ) (hw : ∀ k, 0 ≤ w k) :
    (stackBatches fun k => Real.sqrt (w k) • Xf k)ᵀ
        * (stackBatches fun k => Real.sqrt (w k) • Xf k)
      = ∑ k, (w k) • (((Xf k)ᵀ * (Xf k))) := by
  rw [gram_stack]
  apply Finset.sum_congr rfl
  intro k _
  have hcc : Real.sqrt (w k) * Real.sqrt (w k) = w k :=
    Real.mul_self_sqrt (hw k)
  ext a b
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.smul_apply, smul_eq_mul,
    Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  linear_combination (Xf k j a * Xf k j b) * hcc

/-- Discounted sums with forgetting factor `γ`: row weights `γ ^ (K − k)`.
This is weighted least squares on the streamed batches. -/
theorem gram_discount {K : ℕ} {m : Fin K → Type*} [∀ k, Fintype (m k)]
    {n : Type*} (Xf : ∀ k : Fin K, Matrix (m k) n ℝ)
    (gam : ℝ) (hg : 0 ≤ gam) :
    (stackBatches fun k => Real.sqrt (gam ^ (K - k.val)) • Xf k)ᵀ
        * (stackBatches fun k => Real.sqrt (gam ^ (K - k.val)) • Xf k)
      = ∑ k, ((gam ^ (K - k.val)) • (((Xf k)ᵀ * (Xf k)))) :=
  gram_weighted Xf _ (fun _ => pow_nonneg hg _)

#print axioms ridge_two_batch
#print axioms ridge_posDef
#print axioms ridge_isUnit_det
#print axioms ridge_K_batch
#print axioms gram_weighted
#print axioms gram_discount
