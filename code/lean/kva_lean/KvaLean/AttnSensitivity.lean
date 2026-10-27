import KvaLean.VecNorm
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# The attention a token receives bounds its influence

Single-head softmax attention: queries `q i` (`i : Fin m`), keys `k j` and
values `v j` (`j : Fin n`), all in `Fin d → ℝ`.
Scores `s i j = c * ⟪q i, k j⟫`, weights `a i j = softmax_j (s i ·)`,
outputs `o i = ∑ j, a i j • v j`.

- (a) Values, exact: perturbing only `v j0` by `δ` changes each `o i` by
  exactly `a i j0 • δ`, so `∑ i, ‖Δo i‖ = (∑ i, a i j0) * ‖δ‖`.
- (b) Keys: perturbing only `k j0` by `δ` shifts the single logit `s i j0`
  by `ε i = c * ⟪q i, δ⟫`. Then exactly `a' i j - a i j = ρ i * (𝟙[j = j0] - a i j)`
  and `o' i - o i = ρ i • (v j0 - o i)` with `|ρ i| ≤ a i j0 * (exp |ε i| - 1)`.
  Hence `|a' i j - a i j| ≤ a i j0 * (exp |ε i| - 1)` and
  `‖Δo i‖ ≤ 2 * M * a i j0 * (exp |ε i| - 1)` when `‖v j‖ ≤ M` for all `j`.
- (c) Both together: `‖Δo i‖ ≤ a i j0 * (exp |ε i| * ‖δv‖ + 2 * M * (exp |ε i| - 1))`,
  so the total output change is linear in the attention `∑ i, a i j0` that
  token `j0` receives, and is zero if that attention is zero.
  Softmax weights are strictly positive (`attnWeight_pos`), so the literal
  zero-attention case cannot occur; the linear bound is the real content.

The temperature `c` may be any real number; `c > 0` is never used.
-/

open Finset

variable {m n d : ℕ}

/-! ## Softmax of one row of logits -/

/-- Softmax weight of entry `j` of a row of logits `z`. -/
noncomputable def softmaxW (z : Fin n → ℝ) (j : Fin n) : ℝ :=
  Real.exp (z j) / ∑ l, Real.exp (z l)

theorem sumExp_pos (z : Fin n → ℝ) (j0 : Fin n) : 0 < ∑ l, Real.exp (z l) :=
  Finset.sum_pos (fun l _ => Real.exp_pos (z l)) ⟨j0, Finset.mem_univ _⟩

theorem softmaxW_pos (z : Fin n → ℝ) (j : Fin n) : 0 < softmaxW z j :=
  div_pos (Real.exp_pos _) (sumExp_pos z j)

theorem softmaxW_sum (z : Fin n → ℝ) (j0 : Fin n) : ∑ j, softmaxW z j = 1 := by
  unfold softmaxW
  rw [← Finset.sum_div]
  exact div_self (sumExp_pos z j0).ne'

theorem softmaxW_le_one (z : Fin n → ℝ) (j : Fin n) : softmaxW z j ≤ 1 := by
  unfold softmaxW
  rw [div_le_one (sumExp_pos z j)]
  exact Finset.single_le_sum (fun l _ => (Real.exp_pos (z l)).le) (Finset.mem_univ j)

/-- Shifting the single logit `j0` by `ε` multiplies its exponential by `exp ε`:
the denominator becomes `Z + exp (z j0) * (exp ε - 1)`. -/
theorem sumExp_shift (z z' : Fin n → ℝ) (j0 : Fin n) (ε : ℝ)
    (hoff : ∀ j, j ≠ j0 → z' j = z j) (hon : z' j0 = z j0 + ε) :
    ∑ l, Real.exp (z' l)
      = ∑ l, Real.exp (z l) + Real.exp (z j0) * (Real.exp ε - 1) := by
  have h : ∑ l, Real.exp (z' l) - ∑ l, Real.exp (z l)
      = Real.exp (z' j0) - Real.exp (z j0) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_eq_single j0
    · intro j _ hj
      rw [hoff j hj, sub_self]
    · intro h
      exact absurd (Finset.mem_univ j0) h
  rw [hon, Real.exp_add] at h
  linarith

/-- The shift factor `ρ = exp (z j0) * (exp ε - 1) / Z'` (`Z'` the new denominator). -/
noncomputable def shiftFactor (z' : Fin n → ℝ) (z0 ε : ℝ) : ℝ :=
  Real.exp z0 * (Real.exp ε - 1) / ∑ l, Real.exp (z' l)

/-- Exact change of every softmax weight under a single-logit shift:
`a' j - a j = ρ * (𝟙[j = j0] - a j)`. -/
theorem softmaxW_shift_delta (z z' : Fin n → ℝ) (j0 : Fin n) (ε : ℝ)
    (hoff : ∀ j, j ≠ j0 → z' j = z j) (hon : z' j0 = z j0 + ε) (j : Fin n) :
    softmaxW z' j - softmaxW z j
      = shiftFactor z' (z j0) ε * ((if j = j0 then 1 else 0) - softmaxW z j) := by
  have hZ := sumExp_pos z j0
  have hZ' := sumExp_pos z' j0
  have hsum := sumExp_shift z z' j0 ε hoff hon
  unfold softmaxW shiftFactor
  by_cases hj : j = j0
  · subst hj
    rw [ite_eq_left rfl, hon, Real.exp_add]
    field_simp
    rw [hsum]
    ring
  · rw [ite_eq_right hj, hoff j hj]
    field_simp
    rw [hsum]
    ring

/-- The core scalar inequality `Z * |exp ε - 1| ≤ Z' * (exp |ε| - 1)`,
where `Z' = Z + E0 * (exp ε - 1)` and `0 ≤ E0 ≤ Z`. -/
theorem shift_core_ineq (Z E0 ε : ℝ) (hE0 : 0 ≤ E0) (hE0Z : E0 ≤ Z) :
    Z * |Real.exp ε - 1| ≤ (Z + E0 * (Real.exp ε - 1)) * (Real.exp |ε| - 1) := by
  rcases le_or_gt 0 ε with hε | hε
  · have hu : 1 ≤ Real.exp ε := Real.one_le_exp hε
    rw [abs_of_nonneg hε, abs_of_nonneg (by linarith)]
    nlinarith [mul_nonneg hE0 (mul_nonneg (sub_nonneg.mpr hu) (sub_nonneg.mpr hu))]
  · have hu : Real.exp ε < 1 := Real.exp_lt_one_iff.mpr hε
    have hu0 : 0 < Real.exp ε := Real.exp_pos ε
    have hw0 : 0 < Real.exp (-ε) := Real.exp_pos (-ε)
    have huw : Real.exp ε * Real.exp (-ε) = 1 := by
      rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
    rw [abs_of_neg hε, abs_of_neg (by linarith)]
    have key : (Z + E0 * (Real.exp ε - 1)) * (Real.exp (-ε) - 1)
          - Z * -(Real.exp ε - 1)
        = (1 - Real.exp ε) ^ 2 * Real.exp (-ε) * (Z - E0) := by
      linear_combination (Z * (2 - Real.exp ε) - E0 * (1 - Real.exp ε)) * huw
    have hnn : 0 ≤ (1 - Real.exp ε) ^ 2 * Real.exp (-ε) * (Z - E0) :=
      mul_nonneg (mul_nonneg (sq_nonneg _) hw0.le) (by linarith)
    linarith

/-- The shift factor is at most `a j0 * (exp |ε| - 1)` in absolute value. -/
theorem abs_shiftFactor_le (z z' : Fin n → ℝ) (j0 : Fin n) (ε : ℝ)
    (hoff : ∀ j, j ≠ j0 → z' j = z j) (hon : z' j0 = z j0 + ε) :
    |shiftFactor z' (z j0) ε| ≤ softmaxW z j0 * (Real.exp |ε| - 1) := by
  have hZ := sumExp_pos z j0
  have hZ' := sumExp_pos z' j0
  have hsum := sumExp_shift z z' j0 ε hoff hon
  have hE0 := Real.exp_pos (z j0)
  have hE0Z : Real.exp (z j0) ≤ ∑ l, Real.exp (z l) :=
    Finset.single_le_sum (fun l _ => (Real.exp_pos (z l)).le) (Finset.mem_univ j0)
  have core := shift_core_ineq (∑ l, Real.exp (z l)) (Real.exp (z j0)) ε hE0.le hE0Z
  rw [← hsum] at core
  unfold shiftFactor softmaxW
  rw [abs_div, abs_mul, abs_of_pos hE0, abs_of_pos hZ', div_le_iff₀ hZ',
    div_mul_eq_mul_div, div_mul_eq_mul_div, le_div_iff₀ hZ]
  have := mul_le_mul_of_nonneg_left core hE0.le
  linarith

/-- Per-weight bound: every weight moves by at most `a j0 * (exp |ε| - 1)`. -/
theorem softmaxW_shift_abs_le (z z' : Fin n → ℝ) (j0 : Fin n) (ε : ℝ)
    (hoff : ∀ j, j ≠ j0 → z' j = z j) (hon : z' j0 = z j0 + ε) (j : Fin n) :
    |softmaxW z' j - softmaxW z j| ≤ softmaxW z j0 * (Real.exp |ε| - 1) := by
  rw [softmaxW_shift_delta z z' j0 ε hoff hon j, abs_mul]
  have hρ := abs_shiftFactor_le z z' j0 ε hoff hon
  have h1 : |(if j = j0 then (1 : ℝ) else 0) - softmaxW z j| ≤ 1 := by
    have hp := softmaxW_pos z j
    have hl := softmaxW_le_one z j
    split_ifs <;> rw [abs_le] <;> constructor <;> linarith
  calc |shiftFactor z' (z j0) ε| * |(if j = j0 then (1 : ℝ) else 0) - softmaxW z j|
      ≤ |shiftFactor z' (z j0) ε| * 1 :=
        mul_le_mul_of_nonneg_left h1 (abs_nonneg _)
    _ ≤ softmaxW z j0 * (Real.exp |ε| - 1) := by rw [mul_one]; exact hρ

/-- Total variation: `∑ j, |a' j - a j| = 2 * (1 - a j0) * |ρ|`, hence
`∑ j, |a' j - a j| ≤ 2 * a j0 * (1 - a j0) * (exp |ε| - 1)`. -/
theorem softmaxW_shift_tv (z z' : Fin n → ℝ) (j0 : Fin n) (ε : ℝ)
    (hoff : ∀ j, j ≠ j0 → z' j = z j) (hon : z' j0 = z j0 + ε) :
    ∑ j, |softmaxW z' j - softmaxW z j|
      ≤ 2 * softmaxW z j0 * (1 - softmaxW z j0) * (Real.exp |ε| - 1) := by
  set ρ := shiftFactor z' (z j0) ε
  have hterm : ∀ j, |softmaxW z' j - softmaxW z j|
      = |ρ| * softmaxW z j + (if j = j0 then |ρ| * (1 - 2 * softmaxW z j0) else 0) := by
    intro j
    rw [softmaxW_shift_delta z z' j0 ε hoff hon j, abs_mul]
    by_cases hj : j = j0
    · subst hj
      rw [ite_eq_left rfl, ite_eq_left rfl,
        abs_of_nonneg (a := 1 - softmaxW z j) (by linarith [softmaxW_le_one z j])]
      ring
    · rw [ite_eq_right hj, ite_eq_right hj, zero_sub, abs_neg, abs_of_pos (softmaxW_pos z j)]
      ring
  rw [Finset.sum_congr rfl (fun j _ => hterm j), Finset.sum_add_distrib, ← Finset.mul_sum,
    softmaxW_sum z j0, Finset.sum_ite_eq' Finset.univ j0, ite_eq_left (Finset.mem_univ _)]
  have hρ := abs_shiftFactor_le z z' j0 ε hoff hon
  have ha1 : 0 ≤ 1 - softmaxW z j0 := by linarith [softmaxW_le_one z j0]
  have := mul_le_mul_of_nonneg_left hρ ha1
  nlinarith

/-- The shifted weight of `j0` grows by at most the factor `exp |ε|`. -/
theorem softmaxW_shift_on_le (z z' : Fin n → ℝ) (j0 : Fin n) (ε : ℝ)
    (hoff : ∀ j, j ≠ j0 → z' j = z j) (hon : z' j0 = z j0 + ε) :
    softmaxW z' j0 ≤ softmaxW z j0 * Real.exp |ε| := by
  have hd := softmaxW_shift_delta z z' j0 ε hoff hon j0
  rw [ite_eq_left rfl] at hd
  have hρ := le_trans (le_abs_self _) (abs_shiftFactor_le z z' j0 ε hoff hon)
  have ha0 := (softmaxW_pos z j0).le
  have ha1 : 0 ≤ 1 - softmaxW z j0 := by linarith [softmaxW_le_one z j0]
  have hX : 0 ≤ Real.exp |ε| - 1 := by linarith [Real.one_le_exp (abs_nonneg ε)]
  nlinarith [mul_le_mul_of_nonneg_left hρ ha1, mul_nonneg ha0 (mul_nonneg hX ha0)]

/-! ## Single-head attention -/

/-- Scores `s i j = c * ⟪q i, k j⟫`. -/
def attnScore (c : ℝ) (q : Fin m → Fin d → ℝ) (k : Fin n → Fin d → ℝ)
    (i : Fin m) (j : Fin n) : ℝ :=
  c * (q i ⬝ᵥ k j)

/-- Weights `a i j = softmax_j (s i ·)`. -/
noncomputable def attnWeight (c : ℝ) (q : Fin m → Fin d → ℝ) (k : Fin n → Fin d → ℝ)
    (i : Fin m) (j : Fin n) : ℝ :=
  softmaxW (attnScore c q k i) j

/-- Outputs `o i = ∑ j, a i j • v j`. -/
noncomputable def attnOut (c : ℝ) (q : Fin m → Fin d → ℝ) (k v : Fin n → Fin d → ℝ)
    (i : Fin m) : Fin d → ℝ :=
  ∑ j, attnWeight c q k i j • v j

/-- Every attention weight is strictly positive. -/
theorem attnWeight_pos (c : ℝ) (q : Fin m → Fin d → ℝ) (k : Fin n → Fin d → ℝ)
    (i : Fin m) (j : Fin n) : 0 < attnWeight c q k i j :=
  softmaxW_pos _ _

/-- Perturbing key `j0` by `δ` shifts only the logit `j0`, by `ε i = c * ⟪q i, δ⟫`. -/
theorem attnScore_update_off (c : ℝ) (q : Fin m → Fin d → ℝ) (k : Fin n → Fin d → ℝ)
    (j0 : Fin n) (δ : Fin d → ℝ) (i : Fin m) (j : Fin n) (hj : j ≠ j0) :
    attnScore c q (Function.update k j0 (k j0 + δ)) i j = attnScore c q k i j := by
  simp [attnScore, Function.update_of_ne hj]

theorem attnScore_update_on (c : ℝ) (q : Fin m → Fin d → ℝ) (k : Fin n → Fin d → ℝ)
    (j0 : Fin n) (δ : Fin d → ℝ) (i : Fin m) :
    attnScore c q (Function.update k j0 (k j0 + δ)) i j0
      = attnScore c q k i j0 + c * (q i ⬝ᵥ δ) := by
  simp [attnScore, dotProduct_add, mul_add]

/-! ### (a) Values: exact -/

/-- (a) Perturbing only `v j0` by `δ` changes `o i` by exactly `a i j0 • δ`. -/
theorem attn_value_exact (c : ℝ) (q : Fin m → Fin d → ℝ) (k v : Fin n → Fin d → ℝ)
    (j0 : Fin n) (δ : Fin d → ℝ) (i : Fin m) :
    attnOut c q k (Function.update v j0 (v j0 + δ)) i - attnOut c q k v i
      = attnWeight c q k i j0 • δ := by
  unfold attnOut
  rw [← Finset.sum_sub_distrib]
  rw [Finset.sum_eq_single j0]
  · simp
  · intro j _ hj
    simp [Function.update_of_ne hj]
  · intro h
    exact absurd (Finset.mem_univ j0) h

/-- (a) Summed over queries: `∑ i, ‖Δo i‖ = (∑ i, a i j0) * ‖δ‖`
(impact through values = attention received × value error). -/
theorem attn_value_total (c : ℝ) (q : Fin m → Fin d → ℝ) (k v : Fin n → Fin d → ℝ)
    (j0 : Fin n) (δ : Fin d → ℝ) :
    ∑ i, vnorm (attnOut c q k (Function.update v j0 (v j0 + δ)) i - attnOut c q k v i)
      = (∑ i, attnWeight c q k i j0) * vnorm δ := by
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i _
  rw [attn_value_exact, vnorm_smul, abs_of_pos (attnWeight_pos c q k i j0)]

/-! ### (b) Keys: exact direction and bound -/

/-- (b) Perturbing only `k j0` by `δ` moves `o i` along `v j0 - o i`:
`o' i - o i = ρ i • (v j0 - o i)`, with `ρ i` the shift factor of
`ε i = c * ⟪q i, δ⟫`. -/
theorem attn_key_exact (c : ℝ) (q : Fin m → Fin d → ℝ) (k v : Fin n → Fin d → ℝ)
    (j0 : Fin n) (δ : Fin d → ℝ) (i : Fin m) :
    attnOut c q (Function.update k j0 (k j0 + δ)) v i - attnOut c q k v i
      = shiftFactor (attnScore c q (Function.update k j0 (k j0 + δ)) i)
          (attnScore c q k i j0) (c * (q i ⬝ᵥ δ))
        • (v j0 - attnOut c q k v i) := by
  set z := attnScore c q k i
  set z' := attnScore c q (Function.update k j0 (k j0 + δ)) i
  have hoff : ∀ j, j ≠ j0 → z' j = z j := fun j hj => attnScore_update_off c q k j0 δ i j hj
  have hon : z' j0 = z j0 + c * (q i ⬝ᵥ δ) := attnScore_update_on c q k j0 δ i
  have hterm : ∀ j, softmaxW z' j • v j - softmaxW z j • v j
      = shiftFactor z' (z j0) (c * (q i ⬝ᵥ δ))
          • ((if j = j0 then v j0 else 0) - softmaxW z j • v j) := by
    intro j
    rw [← sub_smul, softmaxW_shift_delta z z' j0 _ hoff hon j, mul_smul, sub_smul]
    by_cases hj : j = j0
    · subst hj
      rw [ite_eq_left rfl, ite_eq_left rfl, one_smul]
    · rw [ite_eq_right hj, ite_eq_right hj, zero_smul]
  unfold attnOut attnWeight
  rw [← Finset.sum_sub_distrib, Finset.sum_congr rfl (fun j _ => hterm j), ← Finset.smul_sum,
    Finset.sum_sub_distrib, Finset.sum_ite_eq' Finset.univ j0, ite_eq_left (Finset.mem_univ _)]

/-- (b) Per-weight bound: `|a' i j - a i j| ≤ a i j0 * (exp |ε i| - 1)` for every `j`. -/
theorem attn_key_weight_le (c : ℝ) (q : Fin m → Fin d → ℝ) (k : Fin n → Fin d → ℝ)
    (j0 : Fin n) (δ : Fin d → ℝ) (i : Fin m) (j : Fin n) :
    |attnWeight c q (Function.update k j0 (k j0 + δ)) i j - attnWeight c q k i j|
      ≤ attnWeight c q k i j0 * (Real.exp |c * (q i ⬝ᵥ δ)| - 1) :=
  softmaxW_shift_abs_le _ _ j0 _ (fun j hj => attnScore_update_off c q k j0 δ i j hj)
    (attnScore_update_on c q k j0 δ i) j

/-- `‖v j0 - o i‖ ≤ 2 * M` when every value has norm at most `M`. -/
theorem vnorm_value_sub_out_le (c : ℝ) (q : Fin m → Fin d → ℝ) (k v : Fin n → Fin d → ℝ)
    (j0 : Fin n) (i : Fin m) (M : ℝ) (hM : ∀ j, vnorm (v j) ≤ M) :
    vnorm (v j0 - attnOut c q k v i) ≤ 2 * M := by
  have ho : vnorm (attnOut c q k v i) ≤ M := by
    unfold attnOut
    calc vnorm (∑ j, attnWeight c q k i j • v j)
        ≤ ∑ j, vnorm (attnWeight c q k i j • v j) := vnorm_sum_le _ _
      _ = ∑ j, attnWeight c q k i j * vnorm (v j) := by
          apply Finset.sum_congr rfl
          intro j _
          rw [vnorm_smul, abs_of_pos (attnWeight_pos c q k i j)]
      _ ≤ ∑ j, attnWeight c q k i j * M :=
          Finset.sum_le_sum fun j _ =>
            mul_le_mul_of_nonneg_left (hM j) (attnWeight_pos c q k i j).le
      _ = M := by
          rw [← Finset.sum_mul]
          unfold attnWeight
          rw [softmaxW_sum _ j0, one_mul]
  linarith [vnorm_sub_le (v j0) (attnOut c q k v i), hM j0]

/-- (b) Output bound: `‖Δo i‖ ≤ 2 * M * a i j0 * (exp |ε i| - 1)`
with `ε i = c * ⟪q i, δ⟫` and `‖v j‖ ≤ M` for all `j`. -/
theorem attn_key_bound (c : ℝ) (q : Fin m → Fin d → ℝ) (k v : Fin n → Fin d → ℝ)
    (j0 : Fin n) (δ : Fin d → ℝ) (i : Fin m) (M : ℝ) (hM : ∀ j, vnorm (v j) ≤ M) :
    vnorm (attnOut c q (Function.update k j0 (k j0 + δ)) v i - attnOut c q k v i)
      ≤ 2 * M * attnWeight c q k i j0 * (Real.exp |c * (q i ⬝ᵥ δ)| - 1) := by
  rw [attn_key_exact, vnorm_smul]
  have hρ := abs_shiftFactor_le (attnScore c q k i)
    (attnScore c q (Function.update k j0 (k j0 + δ)) i) j0 (c * (q i ⬝ᵥ δ))
    (fun j hj => attnScore_update_off c q k j0 δ i j hj) (attnScore_update_on c q k j0 δ i)
  have hv := vnorm_value_sub_out_le c q k v j0 i M hM
  have ha := (attnWeight_pos c q k i j0).le
  have hX : 0 ≤ Real.exp |c * (q i ⬝ᵥ δ)| - 1 := by
    linarith [Real.one_le_exp (abs_nonneg (c * (q i ⬝ᵥ δ)))]
  unfold attnWeight
  calc _ ≤ softmaxW (attnScore c q k i) j0 * (Real.exp |c * (q i ⬝ᵥ δ)| - 1)
        * vnorm (v j0 - attnOut c q k v i) :=
        mul_le_mul_of_nonneg_right hρ (vnorm_nonneg _)
    _ ≤ softmaxW (attnScore c q k i) j0 * (Real.exp |c * (q i ⬝ᵥ δ)| - 1) * (2 * M) :=
        mul_le_mul_of_nonneg_left hv (mul_nonneg ha hX)
    _ = _ := by ring

/-! ### (c) Key and value together; zero attention means zero influence -/

/-- (c) Perturbing both `k j0` by `δk` and `v j0` by `δv`:
`‖Δo i‖ ≤ a i j0 * (exp |ε i| * ‖δv‖ + 2 * M * (exp |ε i| - 1))`. -/
theorem attn_joint_bound (c : ℝ) (q : Fin m → Fin d → ℝ) (k v : Fin n → Fin d → ℝ)
    (j0 : Fin n) (δk δv : Fin d → ℝ) (i : Fin m) (M : ℝ) (hM : ∀ j, vnorm (v j) ≤ M) :
    vnorm (attnOut c q (Function.update k j0 (k j0 + δk)) (Function.update v j0 (v j0 + δv)) i
        - attnOut c q k v i)
      ≤ attnWeight c q k i j0 *
          (Real.exp |c * (q i ⬝ᵥ δk)| * vnorm δv + 2 * M * (Real.exp |c * (q i ⬝ᵥ δk)| - 1)) := by
  set k' := Function.update k j0 (k j0 + δk)
  have hsplit : attnOut c q k' (Function.update v j0 (v j0 + δv)) i - attnOut c q k v i
      = (attnOut c q k' (Function.update v j0 (v j0 + δv)) i - attnOut c q k' v i)
        + (attnOut c q k' v i - attnOut c q k v i) := by abel
  rw [hsplit]
  refine le_trans (vnorm_add_le _ _) ?_
  rw [attn_value_exact, vnorm_smul, abs_of_pos (attnWeight_pos c q k' i j0)]
  have hval : attnWeight c q k' i j0
      ≤ attnWeight c q k i j0 * Real.exp |c * (q i ⬝ᵥ δk)| :=
    softmaxW_shift_on_le _ _ j0 _ (fun j hj => attnScore_update_off c q k j0 δk i j hj)
      (attnScore_update_on c q k j0 δk i)
  have hkey := attn_key_bound c q k v j0 δk i M hM
  have := mul_le_mul_of_nonneg_right hval (vnorm_nonneg δv)
  nlinarith

/-- (c) Summed over queries: if `|ε i| ≤ E` for every query, the total output change is
at most `(∑ i, a i j0) * (exp E * ‖δv‖ + 2 * M * (exp E - 1))`, linear in the attention
token `j0` receives. -/
theorem attn_joint_total (c : ℝ) (q : Fin m → Fin d → ℝ) (k v : Fin n → Fin d → ℝ)
    (j0 : Fin n) (δk δv : Fin d → ℝ) (M E : ℝ) (hM : ∀ j, vnorm (v j) ≤ M)
    (hE : ∀ i, |c * (q i ⬝ᵥ δk)| ≤ E) :
    ∑ i, vnorm (attnOut c q (Function.update k j0 (k j0 + δk))
        (Function.update v j0 (v j0 + δv)) i - attnOut c q k v i)
      ≤ (∑ i, attnWeight c q k i j0) * (Real.exp E * vnorm δv + 2 * M * (Real.exp E - 1)) := by
  rw [Finset.sum_mul]
  apply Finset.sum_le_sum
  intro i _
  refine le_trans (attn_joint_bound c q k v j0 δk δv i M hM) ?_
  apply mul_le_mul_of_nonneg_left _ (attnWeight_pos c q k i j0).le
  have hexp : Real.exp |c * (q i ⬝ᵥ δk)| ≤ Real.exp E := Real.exp_le_exp.mpr (hE i)
  have hM0 : 0 ≤ M := le_trans (vnorm_nonneg _) (hM j0)
  nlinarith [vnorm_nonneg δv]

/-- (c) Corollary: a token that receives zero attention from every query cannot change any
output through its key or value. (Vacuous for exact softmax, where `attnWeight_pos` shows
every weight is positive; the content is the linear bound `attn_joint_total`.) -/
theorem attn_zero_attention_no_influence (c : ℝ) (q : Fin m → Fin d → ℝ)
    (k v : Fin n → Fin d → ℝ) (j0 : Fin n) (δk δv : Fin d → ℝ)
    (h0 : ∀ i, attnWeight c q k i j0 = 0) (i : Fin m) :
    attnOut c q (Function.update k j0 (k j0 + δk)) (Function.update v j0 (v j0 + δv)) i
      = attnOut c q k v i := by
  have hM : ∀ j, vnorm (v j) ≤ ∑ l, vnorm (v l) := fun j =>
    Finset.single_le_sum (fun l _ => vnorm_nonneg (v l)) (Finset.mem_univ j)
  have h := attn_joint_bound c q k v j0 δk δv i _ hM
  rw [h0 i, zero_mul] at h
  have h2 := le_antisymm h (vnorm_nonneg _)
  have hsq := vnorm_sq (attnOut c q (Function.update k j0 (k j0 + δk))
    (Function.update v j0 (v j0 + δv)) i - attnOut c q k v i)
  rw [h2] at hsq
  funext t
  have ht : (attnOut c q (Function.update k j0 (k j0 + δk))
      (Function.update v j0 (v j0 + δv)) i - attnOut c q k v i) t ^ 2 = 0 := by
    have hz := (Finset.sum_eq_zero_iff_of_nonneg (fun s _ => sq_nonneg _)).mp
      (by rw [← hsq]; ring) t (Finset.mem_univ t)
    exact hz
  have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp ht
  rw [Pi.sub_apply] at this
  linarith

#print axioms softmaxW_shift_delta
#print axioms softmaxW_shift_abs_le
#print axioms softmaxW_shift_tv
#print axioms attn_value_exact
#print axioms attn_value_total
#print axioms attn_key_exact
#print axioms attn_key_weight_le
#print axioms attn_key_bound
#print axioms attn_joint_bound
#print axioms attn_joint_total
#print axioms attn_zero_attention_no_influence
