import KvaLean.VecNorm
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# Impact ≤ error × sensitivity, and concentration

Tokens `j : Fin n`, each with a loss gradient `g j` and an approximation error
`δ j` (vectors in `Fin d → ℝ`). The first-order loss change is
`∑ j, ⟪g j, δ j⟫`.

- (a) `|⟪g j, δ j⟫| ≤ ‖g j‖ * ‖δ j‖` (Cauchy–Schwarz), and
  `|∑ j, ⟪g j, δ j⟫| ≤ ∑ j, ‖g j‖ * ‖δ j‖`.
- (b) If every error is bounded, `‖δ j‖ ≤ e`, and the tokens in `S` are kept
  exact (`δ j = 0` on `S`), the first-order change is at most
  `e * ∑_{j ∉ S} ‖g j‖`.
- (b, choice of `S`) Among sets of at most `#S` tokens, the set `S` holding the
  largest gradient norms leaves the smallest outside sensitivity mass
  `∑_{j ∉ S} ‖g j‖`, so it gives the smallest bound in (b).

A flat error (`‖δ j‖ ≤ e` for all `j`) plus a concentrated sensitivity
(`∑_{j ∉ S} ‖g j‖` small for a small `S`) therefore gives a concentrated impact.
-/

open Finset

variable {n d : ℕ}

/-- (a) Per-token impact is at most error times sensitivity. -/
theorem impact_le (g δ : Fin d → ℝ) : |g ⬝ᵥ δ| ≤ vnorm g * vnorm δ :=
  abs_dotProduct_le g δ

/-- (a) The total first-order change is at most `∑ j, ‖g j‖ * ‖δ j‖`. -/
theorem firstOrder_le (g δ : Fin n → Fin d → ℝ) :
    |∑ j, g j ⬝ᵥ δ j| ≤ ∑ j, vnorm (g j) * vnorm (δ j) :=
  le_trans (Finset.abs_sum_le_sum_abs _ _)
    (Finset.sum_le_sum fun j _ => impact_le (g j) (δ j))

/-- (b) Keeping the tokens of `S` exact (`δ j = 0` on `S`) with every error bounded by `e`
leaves a first-order change of at most `e * ∑_{j ∉ S} ‖g j‖`. -/
theorem impact_keep_exact (g δ : Fin n → Fin d → ℝ) (S : Finset (Fin n)) (e : ℝ)
    (hS : ∀ j ∈ S, δ j = 0) (he : ∀ j, vnorm (δ j) ≤ e) :
    |∑ j, g j ⬝ᵥ δ j| ≤ e * ∑ j ∈ Sᶜ, vnorm (g j) := by
  have hsplit : ∑ j, g j ⬝ᵥ δ j = ∑ j ∈ Sᶜ, g j ⬝ᵥ δ j := by
    rw [← Finset.sum_add_sum_compl S, Finset.sum_eq_zero (fun j hj => by rw [hS j hj,
      dotProduct_zero]), zero_add]
  rw [hsplit, Finset.mul_sum]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum fun j _ => ?_)
  calc |g j ⬝ᵥ δ j| ≤ vnorm (g j) * vnorm (δ j) := impact_le _ _
    _ ≤ vnorm (g j) * e := mul_le_mul_of_nonneg_left (he j) (vnorm_nonneg _)
    _ = e * vnorm (g j) := mul_comm _ _

/-- (b, choice of `S`) If `S` holds the largest weights (some threshold `τ` has
`w j ≥ τ` on `S` and `w j ≤ τ` off `S`) and `w ≥ 0`, then any `T` with `#T ≤ #S`
leaves at least as much weight outside: `∑_{j ∉ S} w j ≤ ∑_{j ∉ T} w j`. -/
theorem topSet_compl_sum_le (w : Fin n → ℝ) (hw : ∀ j, 0 ≤ w j) (S T : Finset (Fin n))
    (τ : ℝ) (hin : ∀ j ∈ S, τ ≤ w j) (hout : ∀ j ∉ S, w j ≤ τ) (hcard : T.card ≤ S.card) :
    ∑ j ∈ Sᶜ, w j ≤ ∑ j ∈ Tᶜ, w j := by
  -- Replace `τ` by `max τ 0` so that the threshold is non-negative.
  set τ' := max τ 0
  have hin' : ∀ j ∈ S, τ' ≤ w j := fun j hj => max_le (hin j hj) (hw j)
  have hout' : ∀ j ∉ S, w j ≤ τ' := fun j hj => le_trans (hout j hj) (le_max_left _ _)
  have hτ' : 0 ≤ τ' := le_max_right _ _
  -- It suffices that `T` carries no more weight than `S`.
  have hS := Finset.sum_add_sum_compl S w
  have hT := Finset.sum_add_sum_compl T w
  suffices h : ∑ j ∈ T, w j ≤ ∑ j ∈ S, w j by linarith
  have hS2 := Finset.sum_sdiff (f := w) (Finset.inter_subset_left : S ∩ T ⊆ S)
  have hT2 := Finset.sum_sdiff (f := w) (Finset.inter_subset_left : T ∩ S ⊆ T)
  rw [Finset.inter_comm] at hT2
  have hc1 := Finset.card_sdiff_add_card_inter S T
  have hc2 := Finset.card_sdiff_add_card_inter T S
  rw [Finset.inter_comm] at hc2
  have hcd : (T \ (S ∩ T)).card ≤ (S \ (S ∩ T)).card := by
    rw [Finset.sdiff_inter_self_right, Finset.sdiff_inter_self_left]
    omega
  have hlo : ∑ j ∈ T \ (S ∩ T), w j ≤ (T \ (S ∩ T)).card • τ' :=
    Finset.sum_le_card_nsmul _ _ _ fun j hj => hout' j (by
      rw [Finset.mem_sdiff, Finset.mem_inter] at hj
      exact fun hjS => hj.2 ⟨hjS, hj.1⟩)
  have hhi : (S \ (S ∩ T)).card • τ' ≤ ∑ j ∈ S \ (S ∩ T), w j :=
    Finset.card_nsmul_le_sum _ _ _ fun j hj => hin' j (Finset.mem_sdiff.mp hj).1
  have hmid : (T \ (S ∩ T)).card • τ' ≤ (S \ (S ∩ T)).card • τ' := by
    rw [nsmul_eq_mul, nsmul_eq_mul]
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast hcd) hτ'
  linarith

/-- (b, combined) Keeping exact a top-sensitivity set `S` (largest `‖g j‖`) gives a bound
`e * ∑_{j ∉ S} ‖g j‖` that is no larger than the bound for any other set `T` with
`#T ≤ #S`. -/
theorem impact_keep_top (g δ : Fin n → Fin d → ℝ) (S T : Finset (Fin n)) (e τ : ℝ)
    (he0 : 0 ≤ e) (hS : ∀ j ∈ S, δ j = 0) (he : ∀ j, vnorm (δ j) ≤ e)
    (hin : ∀ j ∈ S, τ ≤ vnorm (g j)) (hout : ∀ j ∉ S, vnorm (g j) ≤ τ)
    (hcard : T.card ≤ S.card) :
    |∑ j, g j ⬝ᵥ δ j| ≤ e * ∑ j ∈ Sᶜ, vnorm (g j)
      ∧ e * ∑ j ∈ Sᶜ, vnorm (g j) ≤ e * ∑ j ∈ Tᶜ, vnorm (g j) :=
  ⟨impact_keep_exact g δ S e hS he,
    mul_le_mul_of_nonneg_left
      (topSet_compl_sum_le (fun j => vnorm (g j)) (fun _ => vnorm_nonneg _) S T τ hin hout hcard)
      he0⟩

#print axioms impact_le
#print axioms firstOrder_le
#print axioms impact_keep_exact
#print axioms topSet_compl_sum_le
#print axioms impact_keep_top
