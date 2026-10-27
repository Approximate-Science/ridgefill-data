import Mathlib.Analysis.Real.Sqrt
import Mathlib.Data.Matrix.Mul
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# Euclidean norm on `Fin d → ℝ`, written as a plain finite sum

Shared helper for `AttnSensitivity`, `ImpactBound` and `CrossTerms`.
Vectors are plain functions `Fin d → ℝ`; the inner product is Mathlib's
`dotProduct` (`u ⬝ᵥ v = ∑ k, u k * v k`) and the norm is defined directly as
`vnorm u = √(∑ k, (u k)²)`. Proved here: Cauchy–Schwarz, homogeneity and the
triangle inequality (for two vectors and for finite sums).
-/

open Finset

variable {d : ℕ}

/-- Euclidean norm: `vnorm u = √(∑ k, (u k)²)`. -/
noncomputable def vnorm (u : Fin d → ℝ) : ℝ :=
  Real.sqrt (∑ k, (u k) ^ 2)

theorem vnorm_nonneg (u : Fin d → ℝ) : 0 ≤ vnorm u :=
  Real.sqrt_nonneg _

theorem vnorm_sq (u : Fin d → ℝ) : (vnorm u) ^ 2 = ∑ k, (u k) ^ 2 :=
  Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _)

theorem vnorm_zero : vnorm (0 : Fin d → ℝ) = 0 := by
  simp [vnorm]

/-- Cauchy–Schwarz: `|⟪u, v⟫| ≤ ‖u‖·‖v‖`. -/
theorem abs_dotProduct_le (u v : Fin d → ℝ) : |u ⬝ᵥ v| ≤ vnorm u * vnorm v := by
  unfold vnorm
  rw [← Real.sqrt_mul (Finset.sum_nonneg fun _ _ => sq_nonneg _)]
  apply Real.abs_le_sqrt
  exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _

theorem vnorm_smul (c : ℝ) (u : Fin d → ℝ) : vnorm (c • u) = |c| * vnorm u := by
  unfold vnorm
  have h : ∑ k, ((c • u) k) ^ 2 = c ^ 2 * ∑ k, (u k) ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k _
    simp only [Pi.smul_apply, smul_eq_mul]
    ring
  rw [h, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq_eq_abs]

/-- Triangle inequality for two vectors. -/
theorem vnorm_add_le (u v : Fin d → ℝ) : vnorm (u + v) ≤ vnorm u + vnorm v := by
  have hcs := abs_dotProduct_le u v
  have hle : u ⬝ᵥ v ≤ vnorm u * vnorm v := le_trans (le_abs_self _) hcs
  have hsq : (vnorm (u + v)) ^ 2 = (vnorm u) ^ 2 + 2 * (u ⬝ᵥ v) + (vnorm v) ^ 2 := by
    rw [vnorm_sq, vnorm_sq, vnorm_sq]
    simp only [dotProduct, Pi.add_apply, Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro k _
    ring
  have h0 := vnorm_nonneg (u + v)
  have h1 := vnorm_nonneg u
  have h2 := vnorm_nonneg v
  nlinarith

theorem vnorm_sub_le (u v : Fin d → ℝ) : vnorm (u - v) ≤ vnorm u + vnorm v := by
  have h := vnorm_add_le u ((-1 : ℝ) • v)
  rw [vnorm_smul, abs_neg, abs_one, one_mul, neg_one_smul, ← sub_eq_add_neg] at h
  exact h

/-- Triangle inequality for finite sums. -/
theorem vnorm_sum_le {ι : Type*} (s : Finset ι) (f : ι → Fin d → ℝ) :
    vnorm (∑ j ∈ s, f j) ≤ ∑ j ∈ s, vnorm (f j) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [vnorm_zero]
  | insert a s ha ih =>
    rw [Finset.sum_insert ha, Finset.sum_insert ha]
    exact le_trans (vnorm_add_le _ _) (by linarith)
