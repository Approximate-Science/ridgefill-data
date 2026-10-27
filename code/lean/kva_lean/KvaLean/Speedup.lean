import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# KVA prefill speedup: monotonicity and bounds

Prompt of `N` tokens, `L`-layer model, split `S`, exact tail `T`,
per-token per-layer full cost `c > 0`, per-token per-skipped-layer fill cost `f`.

- `C_exact = L·N·c`
- `C_kva = S·N·c + (L−S)·(T·c + (N−T)·f)`
- `speedup = C_exact / C_kva`

All quantities are modelled as reals (the paper's naturals cast to `ℝ`).
We prove the statements under `0 ≤ f ≤ c`, which is *weaker* (hence stronger results)
than the paper's `0 ≤ f < c`.
-/

/-- Exact (full-prefill) cost. -/
noncomputable def kvaCostExact (L N c : ℝ) : ℝ := L * N * c

/-- KVA cost with split `S` and exact tail `T`. -/
noncomputable def kvaCostKva (L N S T c f : ℝ) : ℝ :=
  S * N * c + (L - S) * (T * c + (N - T) * f)

/-- Speedup = exact cost / KVA cost. -/
noncomputable def kvaSpeedup (L N S T c f : ℝ) : ℝ :=
  kvaCostExact L N c / kvaCostKva L N S T c f

/-- KVA cost with an extra projector cost `p ≥ 0` per skipped token. -/
noncomputable def kvaCostKvaP (L N S T c f p : ℝ) : ℝ :=
  kvaCostKva L N S T c f + (N - T) * p

/-- Speedup with projector cost. -/
noncomputable def kvaSpeedupP (L N S T c f p : ℝ) : ℝ :=
  kvaCostExact L N c / kvaCostKvaP L N S T c f p

theorem kvaCostKva_pos (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T) (hTN : T ≤ N)
    (hN : 0 < N) (hc : 0 < c) (hf0 : 0 ≤ f) :
    0 < kvaCostKva L N S T c f := by
  have hS : 0 < S := lt_of_lt_of_le one_pos hS1
  have h1 : 0 < S * N * c := mul_pos (mul_pos hS hN) hc
  have h2 : 0 ≤ (L - S) * (T * c + (N - T) * f) := by
    apply mul_nonneg (sub_nonneg.mpr hSL)
    apply add_nonneg (mul_nonneg hT0 hc.le)
    exact mul_nonneg (sub_nonneg.mpr hTN) hf0
  unfold kvaCostKva; linarith

/-- (a) Speedup is non-increasing in the split `S`. -/
theorem kva_speedup_mono_S (hS1 : 1 ≤ S₁) (h12 : S₁ ≤ S₂) (h2L : S₂ ≤ L)
    (hT0 : 0 ≤ T) (hTN : T ≤ N) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hfc : f ≤ c) :
    kvaSpeedup L N S₂ T c f ≤ kvaSpeedup L N S₁ T c f := by
  have hD1 : 0 < kvaCostKva L N S₁ T c f :=
    kvaCostKva_pos hS1 (le_trans h12 h2L) hT0 hTN hN hc hf0
  have hD2 : 0 < kvaCostKva L N S₂ T c f :=
    kvaCostKva_pos (le_trans hS1 h12) h2L hT0 hTN hN hc hf0
  have hdiff : kvaCostKva L N S₁ T c f ≤ kvaCostKva L N S₂ T c f := by
    have h1 : 0 ≤ S₂ - S₁ := sub_nonneg.mpr h12
    have h2 : 0 ≤ (N - T) * (c - f) :=
      mul_nonneg (sub_nonneg.mpr hTN) (sub_nonneg.mpr hfc)
    have heq : kvaCostKva L N S₂ T c f - kvaCostKva L N S₁ T c f
        = (S₂ - S₁) * ((N - T) * (c - f)) := by
      unfold kvaCostKva; ring
    linarith [mul_nonneg h1 h2]
  have hL : 0 < L := lt_of_lt_of_le (lt_of_lt_of_le one_pos hS1) (le_trans h12 h2L)
  have hnum : 0 ≤ kvaCostExact L N c := by
    unfold kvaCostExact; positivity
  unfold kvaSpeedup
  exact div_le_div_of_nonneg_left hnum hD1 hdiff

/-- (a) Speedup is non-increasing in the exact tail `T`. -/
theorem kva_speedup_mono_T (hS1 : 1 ≤ S) (hSL : S ≤ L)
    (hT0 : 0 ≤ T₁) (h12 : T₁ ≤ T₂) (h2N : T₂ ≤ N)
    (hN : 0 < N) (hc : 0 < c) (hf0 : 0 ≤ f) (hfc : f ≤ c) :
    kvaSpeedup L N S T₂ c f ≤ kvaSpeedup L N S T₁ c f := by
  have hD1 : 0 < kvaCostKva L N S T₁ c f :=
    kvaCostKva_pos hS1 hSL hT0 (le_trans h12 h2N) hN hc hf0
  have hD2 : 0 < kvaCostKva L N S T₂ c f :=
    kvaCostKva_pos hS1 hSL (le_trans hT0 h12) h2N hN hc hf0
  have hdiff : kvaCostKva L N S T₁ c f ≤ kvaCostKva L N S T₂ c f := by
    have h1 : 0 ≤ T₂ - T₁ := sub_nonneg.mpr h12
    have h2 : 0 ≤ (L - S) * (c - f) :=
      mul_nonneg (sub_nonneg.mpr hSL) (sub_nonneg.mpr hfc)
    have heq : kvaCostKva L N S T₂ c f - kvaCostKva L N S T₁ c f
        = (T₂ - T₁) * ((L - S) * (c - f)) := by
      unfold kvaCostKva; ring
    have := mul_nonneg h1 h2
    linarith
  have hnum : 0 ≤ kvaCostExact L N c := by
    have hL : 0 < L := lt_of_lt_of_le (lt_of_lt_of_le one_pos hS1) hSL
    unfold kvaCostExact; positivity
  unfold kvaSpeedup
  exact div_le_div_of_nonneg_left hnum hD1 hdiff

/-- (b) Upper bound `speedup ≤ L / S`. -/
theorem kva_speedup_le_bound (hS1 : 1 ≤ S) (hSL : S ≤ L)
    (hT0 : 0 ≤ T) (hTN : T ≤ N) (hN : 0 < N) (hc : 0 < c) (hf0 : 0 ≤ f) :
    kvaSpeedup L N S T c f ≤ L / S := by
  have hS : 0 < S := lt_of_lt_of_le one_pos hS1
  have hD : 0 < kvaCostKva L N S T c f :=
    kvaCostKva_pos hS1 hSL hT0 hTN hN hc hf0
  have hL : 0 < L := lt_of_lt_of_le hS hSL
  have hgap : 0 ≤ L * ((L - S) * (T * c + (N - T) * f)) := by
    apply mul_nonneg hL.le
    apply mul_nonneg (sub_nonneg.mpr hSL)
    apply add_nonneg (mul_nonneg hT0 hc.le)
    exact mul_nonneg (sub_nonneg.mpr hTN) hf0
  unfold kvaSpeedup kvaCostExact
  rw [div_le_div_iff₀ hD hS]
  unfold kvaCostKva
  nlinarith

/-- (b) Equality case: `speedup = L / S` iff `S = L` or (`T = 0` and `f = 0`). -/
theorem kva_speedup_eq_bound_iff (hS1 : 1 ≤ S) (hSL : S ≤ L)
    (hT0 : 0 ≤ T) (hTN : T ≤ N) (hN : 0 < N) (hc : 0 < c) (hf0 : 0 ≤ f) :
    kvaSpeedup L N S T c f = L / S ↔ (S = L ∨ (T = 0 ∧ f = 0)) := by
  have hS : 0 < S := lt_of_lt_of_le one_pos hS1
  have hD : 0 < kvaCostKva L N S T c f :=
    kvaCostKva_pos hS1 hSL hT0 hTN hN hc hf0
  have hL : 0 < L := lt_of_lt_of_le hS hSL
  have hXnn : 0 ≤ T * c + (N - T) * f :=
    add_nonneg (mul_nonneg hT0 hc.le) (mul_nonneg (sub_nonneg.mpr hTN) hf0)
  constructor
  · intro h
    unfold kvaSpeedup kvaCostExact at h
    rw [div_eq_div_iff (ne_of_gt hD) (ne_of_gt hS)] at h
    -- h : L * N * c * S = L * costKva
    have hL' : L * ((L - S) * (T * c + (N - T) * f)) = 0 := by
      unfold kvaCostKva at h; nlinarith [h]
    have := (mul_eq_zero.mp hL')
    rcases this with hL0 | hprod
    · exact absurd hL0 (ne_of_gt hL)
    · rcases mul_eq_zero.mp hprod with hLS | hX
      · left; linarith
      · right
        have hTc : T * c = 0 := by
          have := (add_eq_zero_iff_of_nonneg (mul_nonneg hT0 hc.le)
            (mul_nonneg (sub_nonneg.mpr hTN) hf0)).mp hX
          exact this.1
        have hNf : (N - T) * f = 0 := by
          have := (add_eq_zero_iff_of_nonneg (mul_nonneg hT0 hc.le)
            (mul_nonneg (sub_nonneg.mpr hTN) hf0)).mp hX
          exact this.2
        have hT : T = 0 := by
          rcases mul_eq_zero.mp hTc with h | h
          · exact h
          · exact absurd h (ne_of_gt hc)
        have hNsub : N - T ≠ 0 := by
          rw [hT, sub_zero]; exact ne_of_gt hN
        have hf : f = 0 := by
          rcases mul_eq_zero.mp hNf with h | h
          · exact absurd h hNsub
          · exact h
        exact ⟨hT, hf⟩
  · intro h
    rcases h with hSL' | ⟨hT, hf⟩
    · -- S = L: both sides equal 1
      subst hSL'
      unfold kvaSpeedup kvaCostExact kvaCostKva
      have hD' : S * N * c + (S - S) * (T * c + (N - T) * f) = S * N * c := by ring
      rw [hD']
      rw [div_self (ne_of_gt (mul_pos (mul_pos hS hN) hc))]
      rw [div_self (ne_of_gt hS)]
    · -- T = 0, f = 0: KVA cost equals S * N * c
      subst hT; subst hf
      unfold kvaSpeedup kvaCostExact kvaCostKva
      have hD' : S * N * c + (L - S) * (0 * c + (N - 0) * 0) = S * N * c := by ring
      rw [hD']
      have hSNc : S * N * c ≠ 0 := ne_of_gt (mul_pos (mul_pos hS hN) hc)
      field_simp

/-- (c) Speedup is at least 1. -/
theorem kva_speedup_ge_one (hS1 : 1 ≤ S) (hSL : S ≤ L)
    (hT0 : 0 ≤ T) (hTN : T ≤ N) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hfc : f ≤ c) :
    1 ≤ kvaSpeedup L N S T c f := by
  have hD : 0 < kvaCostKva L N S T c f :=
    kvaCostKva_pos hS1 hSL hT0 hTN hN hc hf0
  have hL : 0 < L := lt_of_lt_of_le (lt_of_lt_of_le one_pos hS1) hSL
  have hgap : kvaCostKva L N S T c f ≤ kvaCostExact L N c := by
    have h1 : 0 ≤ (L - S) * ((N - T) * (c - f)) :=
      mul_nonneg (sub_nonneg.mpr hSL)
        (mul_nonneg (sub_nonneg.mpr hTN) (sub_nonneg.mpr hfc))
    have heq : kvaCostExact L N c - kvaCostKva L N S T c f
        = (L - S) * ((N - T) * (c - f)) := by
      unfold kvaCostExact kvaCostKva; ring
    linarith
  unfold kvaSpeedup
  rw [le_div_iff₀ hD]
  linarith [hgap]

/-- KVA cost with projector cost is positive. -/
theorem kvaCostKvaP_pos (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T) (hTN : T ≤ N)
    (hN : 0 < N) (hc : 0 < c) (hf0 : 0 ≤ f) (hp : 0 ≤ p) :
    0 < kvaCostKvaP L N S T c f p := by
  have hD : 0 < kvaCostKva L N S T c f :=
    kvaCostKva_pos hS1 hSL hT0 hTN hN hc hf0
  have hle : kvaCostKva L N S T c f ≤ kvaCostKvaP L N S T c f p := by
    unfold kvaCostKvaP
    have : 0 ≤ (N - T) * p := mul_nonneg (sub_nonneg.mpr hTN) hp
    linarith
  linarith

/-- With projector cost `p`, speedup only decreases. -/
theorem kva_speedupP_le (hS1 : 1 ≤ S) (hSL : S ≤ L)
    (hT0 : 0 ≤ T) (hTN : T ≤ N) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) :
    kvaSpeedupP L N S T c f p ≤ kvaSpeedup L N S T c f := by
  have hD : 0 < kvaCostKva L N S T c f :=
    kvaCostKva_pos hS1 hSL hT0 hTN hN hc hf0
  have hnum : 0 ≤ kvaCostExact L N c := by
    have hL : 0 < L := lt_of_lt_of_le (lt_of_lt_of_le one_pos hS1) hSL
    unfold kvaCostExact; positivity
  have hle : kvaCostKva L N S T c f ≤ kvaCostKvaP L N S T c f p := by
    unfold kvaCostKvaP
    have : 0 ≤ (N - T) * p := mul_nonneg (sub_nonneg.mpr hTN) hp
    linarith
  have hP : 0 < kvaCostKvaP L N S T c f p := lt_of_lt_of_le hD hle
  unfold kvaSpeedupP kvaSpeedup
  exact div_le_div_of_nonneg_left hnum hD hle

/-- With projector cost, speedup is still ≥ 1 provided the projector is cheap
enough: `p ≤ (L - S) * (c - f)`. (Note `f + p ≤ c` plus `1 ≤ L - S` implies this.) -/
theorem kva_speedupP_ge_one (hS1 : 1 ≤ S) (hSL : S ≤ L)
    (hT0 : 0 ≤ T) (hTN : T ≤ N) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (_hfc : f ≤ c) (hp : 0 ≤ p) (hcheap : p ≤ (L - S) * (c - f)) :
    1 ≤ kvaSpeedupP L N S T c f p := by
  have hP : 0 < kvaCostKvaP L N S T c f p :=
    kvaCostKvaP_pos hS1 hSL hT0 hTN hN hc hf0 hp
  have hgap : kvaCostKvaP L N S T c f p ≤ kvaCostExact L N c := by
    have h1 : 0 ≤ (N - T) * (((L - S) * (c - f)) - p) :=
      mul_nonneg (sub_nonneg.mpr hTN) (sub_nonneg.mpr hcheap)
    have heq : kvaCostExact L N c - kvaCostKvaP L N S T c f p
        = (N - T) * (((L - S) * (c - f)) - p) := by
      unfold kvaCostExact kvaCostKvaP kvaCostKva; ring
    linarith
  unfold kvaSpeedupP
  rw [le_div_iff₀ hP]
  linarith [hgap]

#print axioms kva_speedup_mono_S
#print axioms kva_speedup_mono_T
#print axioms kva_speedup_le_bound
#print axioms kva_speedup_eq_bound_iff
#print axioms kva_speedup_ge_one
