import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# Uniform shift on a subset of logits

Softmax over a finite index set `ι`, partitioned into `P` and `Pᶜ`.
Adding the same constant `c` to every logit in `P`:

- (a) the distribution conditioned on `P` is unchanged;
- (b) the distribution conditioned on `Pᶜ` is unchanged;
- (c) the total softmax mass on `P` is non-decreasing in `c`
      (strictly increasing if both `P` and `Pᶜ` are non-empty).
-/

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Softmax mass of a set `s`: `∑_{i ∈ s} exp (z i) / ∑_i exp (z i)`. -/
noncomputable def softmaxMass (z : ι → ℝ) (s : Finset ι) : ℝ :=
  (∑ i ∈ s, Real.exp (z i)) / (∑ i, Real.exp (z i))

/-- Logits with the constant `c` added on `P`. -/
noncomputable def shiftLogits (z : ι → ℝ) (P : Finset ι) (c : ℝ) : ι → ℝ :=
  fun i => if i ∈ P then z i + c else z i

omit [Fintype ι] in
theorem shiftLogits_on (z : ι → ℝ) (P : Finset ι) (c : ℝ) (i : ι) (hi : i ∈ P) :
    shiftLogits z P c i = z i + c := by
  simp [shiftLogits, hi]

omit [Fintype ι] in
theorem shiftLogits_off (z : ι → ℝ) (P : Finset ι) (c : ℝ) (i : ι) (hi : i ∉ P) :
    shiftLogits z P c i = z i := by
  simp [shiftLogits, hi]

-- The shifted exponentials over `P` factor as `exp c` times the unshifted ones.
-- (No `Fintype`: all sums are over the finite set `P`.)
omit [Fintype ι] in
theorem sum_shift_on (z : ι → ℝ) (P : Finset ι) (c : ℝ) :
    ∑ i ∈ P, Real.exp (shiftLogits z P c i)
      = Real.exp c * ∑ i ∈ P, Real.exp (z i) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  rw [shiftLogits_on _ _ _ _ hi, Real.exp_add, mul_comm]

/-- The shifted exponentials off `P` are unchanged. -/
theorem sum_shift_off (z : ι → ℝ) (P : Finset ι) (c : ℝ) :
    ∑ i ∈ Pᶜ, Real.exp (shiftLogits z P c i) = ∑ i ∈ Pᶜ, Real.exp (z i) := by
  apply Finset.sum_congr rfl
  intro i hi
  rw [shiftLogits_off _ _ _ _ (Finset.mem_compl.mp hi)]

/-- The total shifted mass splits as `exp c * A + B`. -/
theorem sum_shift_total (z : ι → ℝ) (P : Finset ι) (c : ℝ) :
    ∑ i, Real.exp (shiftLogits z P c i)
      = Real.exp c * (∑ i ∈ P, Real.exp (z i)) + ∑ i ∈ Pᶜ, Real.exp (z i) := by
  have h : ∑ i, Real.exp (shiftLogits z P c i)
      = (∑ i ∈ P, Real.exp (shiftLogits z P c i))
        + (∑ i ∈ Pᶜ, Real.exp (shiftLogits z P c i)) := by
    rw [← Finset.sum_union (disjoint_compl_right), Finset.union_compl]
  rw [h, sum_shift_on, sum_shift_off]

-- (a) The distribution conditioned on `P` is unchanged by the shift.
omit [Fintype ι] in
theorem softmax_cond_on_unchanged (z : ι → ℝ) (P : Finset ι) (c : ℝ) (i : ι)
    (hi : i ∈ P) :
    Real.exp (shiftLogits z P c i) / (∑ j ∈ P, Real.exp (shiftLogits z P c j))
      = Real.exp (z i) / (∑ j ∈ P, Real.exp (z j)) := by
  rw [sum_shift_on, shiftLogits_on _ _ _ _ hi, Real.exp_add,
    mul_comm (Real.exp (z i)) (Real.exp c)]
  exact mul_div_mul_left _ _ (Real.exp_ne_zero c)

/-- (b) The distribution conditioned on `Pᶜ` is unchanged by the shift. -/
theorem softmax_cond_off_unchanged (z : ι → ℝ) (P : Finset ι) (c : ℝ) (i : ι)
    (hi : i ∈ Pᶜ) :
    Real.exp (shiftLogits z P c i) / (∑ j ∈ Pᶜ, Real.exp (shiftLogits z P c j))
      = Real.exp (z i) / (∑ j ∈ Pᶜ, Real.exp (z j)) := by
  rw [sum_shift_off, shiftLogits_off _ _ _ _ (Finset.mem_compl.mp hi)]

/-- (c) The total softmax mass on `P` is non-decreasing in the shift `c`. -/
theorem softmaxMass_shift_mono [Nonempty ι] (z : ι → ℝ) (P : Finset ι)
    (c₁ c₂ : ℝ) (h12 : c₁ ≤ c₂) :
    softmaxMass (shiftLogits z P c₁) P ≤ softmaxMass (shiftLogits z P c₂) P := by
  have hA0 : 0 ≤ ∑ i ∈ P, Real.exp (z i) :=
    Finset.sum_nonneg (fun i _ => (Real.exp_pos _).le)
  have hB0 : 0 ≤ ∑ i ∈ Pᶜ, Real.exp (z i) :=
    Finset.sum_nonneg (fun i _ => (Real.exp_pos _).le)
  have hD1 : 0 < ∑ i, Real.exp (shiftLogits z P c₁ i) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) Finset.univ_nonempty
  have hD2 : 0 < ∑ i, Real.exp (shiftLogits z P c₂ i) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) Finset.univ_nonempty
  have e12 : Real.exp c₁ ≤ Real.exp c₂ := Real.exp_le_exp.mpr h12
  have key : Real.exp c₁ * (∑ i ∈ P, Real.exp (z i)) * (∑ i ∈ Pᶜ, Real.exp (z i))
      ≤ Real.exp c₂ * (∑ i ∈ P, Real.exp (z i)) * (∑ i ∈ Pᶜ, Real.exp (z i)) := by
    have h := mul_le_mul_of_nonneg_right e12 (mul_nonneg hA0 hB0)
    linarith
  unfold softmaxMass
  rw [div_le_div_iff₀ hD1 hD2, sum_shift_on, sum_shift_on,
    sum_shift_total, sum_shift_total]
  linear_combination key

/-- (c, strict) The mass on `P` is strictly increasing when both sides are non-empty. -/
theorem softmaxMass_shift_strict [Nonempty ι] (z : ι → ℝ) (P : Finset ι)
    (c₁ c₂ : ℝ) (h12 : c₁ < c₂) (hP : P.Nonempty) (hQ : Pᶜ.Nonempty) :
    softmaxMass (shiftLogits z P c₁) P < softmaxMass (shiftLogits z P c₂) P := by
  have hA0 : 0 < ∑ i ∈ P, Real.exp (z i) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) hP
  have hB0 : 0 < ∑ i ∈ Pᶜ, Real.exp (z i) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) hQ
  have hD1 : 0 < ∑ i, Real.exp (shiftLogits z P c₁ i) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) Finset.univ_nonempty
  have hD2 : 0 < ∑ i, Real.exp (shiftLogits z P c₂ i) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) Finset.univ_nonempty
  have e12 : Real.exp c₁ < Real.exp c₂ := Real.exp_lt_exp.mpr h12
  have key : Real.exp c₁ * (∑ i ∈ P, Real.exp (z i)) * (∑ i ∈ Pᶜ, Real.exp (z i))
      < Real.exp c₂ * (∑ i ∈ P, Real.exp (z i)) * (∑ i ∈ Pᶜ, Real.exp (z i)) := by
    have h := mul_lt_mul_of_pos_right e12 (mul_pos hA0 hB0)
    linarith
  unfold softmaxMass
  rw [div_lt_div_iff₀ hD1 hD2, sum_shift_on, sum_shift_on,
    sum_shift_total, sum_shift_total]
  linear_combination key

#print axioms softmax_cond_on_unchanged
#print axioms softmax_cond_off_unchanged
#print axioms softmaxMass_shift_mono
#print axioms softmaxMass_shift_strict
