import KvaLean.Speedup

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# KVA prefill speedup with exact prefix rows at a per-row premium

Extends `KvaLean/Speedup.lean` (same symbols: prompt `N`, layers `L`, split `S`,
exact tail `T`, per-token per-layer exact cost `c > 0`, per-token per-skipped-layer
fill cost `f`, per-skipped-token projector cost `p`).

New ingredient: `R` of the `N − T` prefix rows are kept exact in the late layers, and each
such row costs `κ·c` per late layer, i.e. `κ` times a normal exact token's late-layer cost
(`κ ≥ 1` in practice; on the measured engine these rows run the full MoE with experts in
host RAM). `κ` is the brief's "c"; `c` is already the per-layer cost in `Speedup.lean`.
The other `N − T − R` prefix rows pay fill and projector as before.

- `C_R = S·N·c + (L−S)·(T·c + R·κ·c + (N−T−R)·f) + (N−T−R)·p`
- `speedup_R = C_exact / C_R`

Per-row quantities used in the break-even statements:
- saving of one approximated prefix row vs exact: `s = (L−S)·(c−f) − p`
  (`kvaRowSaving`; `0 ≤ s` is exactly the "cheap projector" hypothesis of
  `kva_speedupP_ge_one`);
- premium of one exact prefix row vs a normal exact row: `e = (κ−1)·(L−S)·c`
  (`kvaRowPremium`).

Then `C_exact − C_R = (N−T−R)·s − R·e`, so KVA with exact rows is at least as fast as exact
iff `R·e ≤ (N−T−R)·s`, and (when `s + e > 0`) iff `R ≤ (N−T)·s/(s+e)` (`kvaBreakEvenR`).

All quantities are reals, as in `Speedup.lean`. The premium `κ` is a model input; its value
on real hardware is an empirical measurement and is not derived here.
-/

variable {L N S T c f p R κ : ℝ}

/-- KVA cost with `R` exact prefix rows at late-layer premium `κ`. -/
noncomputable def kvaCostKvaR (L N S T c f p R κ : ℝ) : ℝ :=
  S * N * c + (L - S) * (T * c + R * (κ * c) + (N - T - R) * f) + (N - T - R) * p

/-- Speedup with `R` exact prefix rows at premium `κ`. -/
noncomputable def kvaSpeedupR (L N S T c f p R κ : ℝ) : ℝ :=
  kvaCostExact L N c / kvaCostKvaR L N S T c f p R κ

/-- Late-layer saving of one approximated prefix row versus exact: `(L−S)·(c−f) − p`. -/
noncomputable def kvaRowSaving (L S c f p : ℝ) : ℝ := (L - S) * (c - f) - p

/-- Extra late-layer cost of one exact row at premium `κ` versus a normal exact row:
`(κ−1)·(L−S)·c`. -/
noncomputable def kvaRowPremium (L S c κ : ℝ) : ℝ := (κ - 1) * ((L - S) * c)

/-- Break-even number of exact prefix rows: `(N−T)·s / (s + e)`. -/
noncomputable def kvaBreakEvenR (L N S T c f p κ : ℝ) : ℝ :=
  (N - T) * kvaRowSaving L S c f p / (kvaRowSaving L S c f p + kvaRowPremium L S c κ)

/-! ## Consistency with `Speedup.lean` -/

/-- With no exact rows the cost is the existing projector-cost model `kvaCostKvaP`. -/
theorem kvaCostKvaR_zero : kvaCostKvaR L N S T c f p 0 κ = kvaCostKvaP L N S T c f p := by
  unfold kvaCostKvaR kvaCostKvaP kvaCostKva; ring

/-- With no exact rows the speedup is the existing `kvaSpeedupP`. -/
theorem kvaSpeedupR_zero : kvaSpeedupR L N S T c f p 0 κ = kvaSpeedupP L N S T c f p := by
  unfold kvaSpeedupR kvaSpeedupP; rw [kvaCostKvaR_zero]

/-- At premium `κ = 1` and no projector cost, `R` exact prefix rows are the same as an
exact tail of `T + R`. -/
theorem kvaCostKvaR_unit_premium :
    kvaCostKvaR L N S T c f 0 R 1 = kvaCostKva L N S (T + R) c f := by
  unfold kvaCostKvaR kvaCostKva; ring

/-- An exact row minus an approximated row, per row: `s + e = (L−S)·(κ·c − f) − p`. -/
theorem kvaRowSaving_add_premium :
    kvaRowSaving L S c f p + kvaRowPremium L S c κ = (L - S) * (κ * c - f) - p := by
  unfold kvaRowSaving kvaRowPremium; ring

/-- The cost is affine in `R`: each exact row adds `s + e` over the no-exact-row cost. -/
theorem kvaCostKvaR_eq_add :
    kvaCostKvaR L N S T c f p R κ
      = kvaCostKvaP L N S T c f p + R * ((L - S) * (κ * c - f) - p) := by
  unfold kvaCostKvaR kvaCostKvaP kvaCostKva; ring

/-- The cost gap to exact prefill: `C_exact − C_R = (N−T−R)·s − R·e`. -/
theorem kvaCostExact_sub_kvaCostKvaR :
    kvaCostExact L N c - kvaCostKvaR L N S T c f p R κ
      = (N - T - R) * kvaRowSaving L S c f p - R * kvaRowPremium L S c κ := by
  unfold kvaCostExact kvaCostKvaR kvaRowSaving kvaRowPremium; ring

/-- The cost with exact rows is positive. -/
theorem kvaCostKvaR_pos (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T)
    (hR0 : 0 ≤ R) (hRN : R ≤ N - T) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ : 0 ≤ κ) :
    0 < kvaCostKvaR L N S T c f p R κ := by
  have hS : 0 < S := lt_of_lt_of_le one_pos hS1
  have h1 : 0 < S * N * c := mul_pos (mul_pos hS hN) hc
  have hrest : 0 ≤ N - T - R := by linarith
  have h2 : 0 ≤ (L - S) * (T * c + R * (κ * c) + (N - T - R) * f) := by
    apply mul_nonneg (sub_nonneg.mpr hSL)
    apply add_nonneg (add_nonneg (mul_nonneg hT0 hc.le) _) (mul_nonneg hrest hf0)
    exact mul_nonneg hR0 (mul_nonneg hκ hc.le)
  have h3 : 0 ≤ (N - T - R) * p := mul_nonneg hrest hp
  unfold kvaCostKvaR; linarith

theorem kvaCostExact_nonneg (hS1 : 1 ≤ S) (hSL : S ≤ L) (hN : 0 < N) (hc : 0 < c) :
    0 ≤ kvaCostExact L N c := by
  have hL : 0 < L := lt_of_lt_of_le (lt_of_lt_of_le one_pos hS1) hSL
  unfold kvaCostExact; positivity

/-! ## 1. Monotonicity -/

/-- (1a) Speedup is non-increasing in the number `R` of exact prefix rows, provided an exact
row costs at least as much as an approximated one: `p ≤ (L−S)·(κ·c − f)`. -/
theorem kva_speedupR_anti_R (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T)
    (hR0 : 0 ≤ R₁) (h12 : R₁ ≤ R₂) (h2N : R₂ ≤ N - T) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ : 0 ≤ κ) (hrow : p ≤ (L - S) * (κ * c - f)) :
    kvaSpeedupR L N S T c f p R₂ κ ≤ kvaSpeedupR L N S T c f p R₁ κ := by
  have hD1 : 0 < kvaCostKvaR L N S T c f p R₁ κ :=
    kvaCostKvaR_pos hS1 hSL hT0 hR0 (le_trans h12 h2N) hN hc hf0 hp hκ
  have hdiff : kvaCostKvaR L N S T c f p R₁ κ ≤ kvaCostKvaR L N S T c f p R₂ κ := by
    have heq : kvaCostKvaR L N S T c f p R₂ κ - kvaCostKvaR L N S T c f p R₁ κ
        = (R₂ - R₁) * ((L - S) * (κ * c - f) - p) := by
      unfold kvaCostKvaR; ring
    have := mul_nonneg (sub_nonneg.mpr h12) (sub_nonneg.mpr hrow)
    linarith
  unfold kvaSpeedupR
  exact div_le_div_of_nonneg_left (kvaCostExact_nonneg hS1 hSL hN hc) hD1 hdiff

/-- (1a, practical form) The row condition of `kva_speedupR_anti_R` holds whenever `κ ≥ 1`
and the projector is cheap in the sense of `kva_speedupP_ge_one`: `p ≤ (L−S)·(c−f)`. -/
theorem kva_speedupR_anti_R_of_premium (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T)
    (hR0 : 0 ≤ R₁) (h12 : R₁ ≤ R₂) (h2N : R₂ ≤ N - T) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ1 : 1 ≤ κ) (hcheap : p ≤ (L - S) * (c - f)) :
    kvaSpeedupR L N S T c f p R₂ κ ≤ kvaSpeedupR L N S T c f p R₁ κ := by
  have hrow : p ≤ (L - S) * (κ * c - f) := by
    have : 0 ≤ (L - S) * ((κ - 1) * c) :=
      mul_nonneg (sub_nonneg.mpr hSL) (mul_nonneg (sub_nonneg.mpr hκ1) hc.le)
    have heq : (L - S) * (κ * c - f) = (L - S) * (c - f) + (L - S) * ((κ - 1) * c) := by ring
    linarith
  exact kva_speedupR_anti_R hS1 hSL hT0 hR0 h12 h2N hN hc hf0 hp
    (le_trans zero_le_one hκ1) hrow

/-- (1b) Speedup is non-increasing in the per-row premium `κ`. -/
theorem kva_speedupR_anti_premium (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T)
    (hR0 : 0 ≤ R) (hRN : R ≤ N - T) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ0 : 0 ≤ κ₁) (h12 : κ₁ ≤ κ₂) :
    kvaSpeedupR L N S T c f p R κ₂ ≤ kvaSpeedupR L N S T c f p R κ₁ := by
  have hD1 : 0 < kvaCostKvaR L N S T c f p R κ₁ :=
    kvaCostKvaR_pos hS1 hSL hT0 hR0 hRN hN hc hf0 hp hκ0
  have hdiff : kvaCostKvaR L N S T c f p R κ₁ ≤ kvaCostKvaR L N S T c f p R κ₂ := by
    have heq : kvaCostKvaR L N S T c f p R κ₂ - kvaCostKvaR L N S T c f p R κ₁
        = (κ₂ - κ₁) * ((L - S) * (R * c)) := by
      unfold kvaCostKvaR; ring
    have := mul_nonneg (sub_nonneg.mpr h12)
      (mul_nonneg (sub_nonneg.mpr hSL) (mul_nonneg hR0 hc.le))
    linarith
  unfold kvaSpeedupR
  exact div_le_div_of_nonneg_left (kvaCostExact_nonneg hS1 hSL hN hc) hD1 hdiff

/-! ## 2. Exact break-even -/

/-- (2) Exact break-even. KVA with `R` exact rows at premium `κ` is at least as fast as exact
prefill iff the exact rows' premium fits inside the net saving of the approximated rows:
`R·(κ−1)·((L−S)·c) ≤ (N−T−R)·((L−S)·(c−f) − p)`. -/
theorem kva_speedupR_ge_one_iff (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T)
    (hR0 : 0 ≤ R) (hRN : R ≤ N - T) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ : 0 ≤ κ) :
    1 ≤ kvaSpeedupR L N S T c f p R κ ↔
      R * ((κ - 1) * ((L - S) * c)) ≤ (N - T - R) * ((L - S) * (c - f) - p) := by
  have hD : 0 < kvaCostKvaR L N S T c f p R κ :=
    kvaCostKvaR_pos hS1 hSL hT0 hR0 hRN hN hc hf0 hp hκ
  have heq := kvaCostExact_sub_kvaCostKvaR (L := L) (N := N) (S := S) (T := T) (c := c)
    (f := f) (p := p) (R := R) (κ := κ)
  unfold kvaRowSaving kvaRowPremium at heq
  unfold kvaSpeedupR
  rw [le_div_iff₀ hD, one_mul]
  constructor <;> intro h <;> linarith

/-- (2, total-cost form) Equivalently: the exact rows' whole late-layer cost
`κ·R·((L−S)·c)` must fit inside the late-layer budget that exact prefill spends on all
`N − T` prefix rows, minus what the approximated rows still cost (fill + projector).
(Only the premium `κ − 1` counts against the saving, because exact prefill also pays
`(L−S)·c` for each of those `R` rows.) -/
theorem kva_speedupR_ge_one_iff_total (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T)
    (hR0 : 0 ≤ R) (hRN : R ≤ N - T) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ : 0 ≤ κ) :
    1 ≤ kvaSpeedupR L N S T c f p R κ ↔
      κ * R * ((L - S) * c) ≤ (N - T) * ((L - S) * c) - (N - T - R) * ((L - S) * f + p) := by
  rw [kva_speedupR_ge_one_iff hS1 hSL hT0 hR0 hRN hN hc hf0 hp hκ]
  constructor <;> intro h <;> linarith

/-- (2, strict) KVA with exact rows is strictly slower than exact prefill iff
`(N−T−R)·((L−S)·(c−f) − p) < R·(κ−1)·((L−S)·c)`. -/
theorem kva_speedupR_lt_one_iff (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T)
    (hR0 : 0 ≤ R) (hRN : R ≤ N - T) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ : 0 ≤ κ) :
    kvaSpeedupR L N S T c f p R κ < 1 ↔
      (N - T - R) * ((L - S) * (c - f) - p) < R * ((κ - 1) * ((L - S) * c)) := by
  rw [← not_le, kva_speedupR_ge_one_iff hS1 hSL hT0 hR0 hRN hN hc hf0 hp hκ, not_le]

/-- (2, threshold form) If an exact row costs strictly more than an approximated one
(`s + e > 0`), KVA with exact rows is at least as fast as exact iff
`R ≤ (N−T)·s/(s+e)` (`kvaBreakEvenR`). -/
theorem kva_speedupR_ge_one_iff_le_breakEven (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T)
    (hR0 : 0 ≤ R) (hRN : R ≤ N - T) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ : 0 ≤ κ)
    (hgap : 0 < kvaRowSaving L S c f p + kvaRowPremium L S c κ) :
    1 ≤ kvaSpeedupR L N S T c f p R κ ↔ R ≤ kvaBreakEvenR L N S T c f p κ := by
  rw [kva_speedupR_ge_one_iff hS1 hSL hT0 hR0 hRN hN hc hf0 hp hκ]
  unfold kvaBreakEvenR
  rw [le_div_iff₀ hgap]
  unfold kvaRowSaving kvaRowPremium
  constructor <;> intro h <;> linarith

/-- (2, threshold form, strict) Under `s + e > 0`: slower than exact iff `R > (N−T)·s/(s+e)`. -/
theorem kva_speedupR_lt_one_iff_breakEven_lt (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T)
    (hR0 : 0 ≤ R) (hRN : R ≤ N - T) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ : 0 ≤ κ)
    (hgap : 0 < kvaRowSaving L S c f p + kvaRowPremium L S c κ) :
    kvaSpeedupR L N S T c f p R κ < 1 ↔ kvaBreakEvenR L N S T c f p κ < R := by
  rw [← not_le, kva_speedupR_ge_one_iff_le_breakEven hS1 hSL hT0 hR0 hRN hN hc hf0 hp hκ hgap,
    not_le]

/-- (2, two-point calibration) The cost is affine in `R`, so the break-even can be read off
three costs: exact `C_exact`, no exact rows `C_0`, and `R₁ > 0` exact rows `C_{R₁}`.
KVA with `R` exact rows is at least as fast as exact iff
`R·(C_{R₁} − C_0) ≤ R₁·(C_exact − C_0)`. This is the form the measured TTFTs plug into. -/
theorem kva_speedupR_ge_one_iff_calib (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T)
    (hR0 : 0 ≤ R) (hRN : R ≤ N - T) (hR₁ : 0 < R₁) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ : 0 ≤ κ) :
    1 ≤ kvaSpeedupR L N S T c f p R κ ↔
      R * (kvaCostKvaR L N S T c f p R₁ κ - kvaCostKvaR L N S T c f p 0 κ)
        ≤ R₁ * (kvaCostExact L N c - kvaCostKvaR L N S T c f p 0 κ) := by
  have hD : 0 < kvaCostKvaR L N S T c f p R κ :=
    kvaCostKvaR_pos hS1 hSL hT0 hR0 hRN hN hc hf0 hp hκ
  set d : ℝ := (L - S) * (κ * c - f) - p with hd
  have hR' : kvaCostKvaR L N S T c f p R κ = kvaCostKvaP L N S T c f p + R * d :=
    kvaCostKvaR_eq_add
  have h1 : kvaCostKvaR L N S T c f p R₁ κ = kvaCostKvaP L N S T c f p + R₁ * d :=
    kvaCostKvaR_eq_add
  have h0 : kvaCostKvaR L N S T c f p 0 κ = kvaCostKvaP L N S T c f p := kvaCostKvaR_zero
  rw [h1, h0]
  unfold kvaSpeedupR
  rw [le_div_iff₀ hD, one_mul, hR']
  set E := kvaCostExact L N c
  set P := kvaCostKvaP L N S T c f p
  have hring : R * (P + R₁ * d - P) - R₁ * (E - P) = R₁ * (P + R * d - E) := by ring
  constructor
  · intro h
    have : R₁ * (P + R * d - E) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos hR₁.le (by linarith)
    linarith
  · intro h
    have hle : R₁ * (P + R * d - E) ≤ R₁ * 0 := by rw [mul_zero]; linarith
    have := le_of_mul_le_mul_left hle hR₁
    linarith

/-! ## 3. Corollaries -/

/-- The break-even row count lies in `[0, N − T)` when `κ > 1`, `S < L`, `T < N` and the
projector is cheap (`p ≤ (L−S)·(c−f)`): so there is always a range of `R` where KVA loses. -/
theorem kvaBreakEvenR_mem (hSL : S < L) (hTN : T < N) (hc : 0 < c)
    (hκ1 : 1 < κ) (hcheap : p ≤ (L - S) * (c - f)) :
    0 ≤ kvaBreakEvenR L N S T c f p κ ∧ kvaBreakEvenR L N S T c f p κ < N - T := by
  have hs : 0 ≤ kvaRowSaving L S c f p := by unfold kvaRowSaving; linarith
  have he : 0 < kvaRowPremium L S c κ := by
    unfold kvaRowPremium
    exact mul_pos (sub_pos.mpr hκ1) (mul_pos (sub_pos.mpr hSL) hc)
  have hNT : 0 < N - T := sub_pos.mpr hTN
  have hgap : 0 < kvaRowSaving L S c f p + kvaRowPremium L S c κ := by linarith
  unfold kvaBreakEvenR
  refine ⟨div_nonneg (mul_nonneg hNT.le hs) hgap.le, ?_⟩
  rw [div_lt_iff₀ hgap]
  nlinarith [mul_pos hNT he]

/-- (3) With premium `κ > 1`, once `R` exceeds
`(N−T)·((L−S)·(c−f) − p) / ((L−S)·(c−f) − p + (κ−1)·(L−S)·c)`
KVA with exact rows is strictly slower than exact prefill. -/
theorem kva_speedupR_lt_one_of_large (hS1 : 1 ≤ S) (hSL : S < L) (hT0 : 0 ≤ T)
    (hTN : T ≤ N) (hRN : R ≤ N - T) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ1 : 1 < κ) (hcheap : p ≤ (L - S) * (c - f))
    (hR : (N - T) * ((L - S) * (c - f) - p)
        / ((L - S) * (c - f) - p + (κ - 1) * ((L - S) * c)) < R) :
    kvaSpeedupR L N S T c f p R κ < 1 := by
  have hs : 0 ≤ kvaRowSaving L S c f p := by unfold kvaRowSaving; linarith
  have he : 0 < kvaRowPremium L S c κ := by
    unfold kvaRowPremium
    exact mul_pos (sub_pos.mpr hκ1) (mul_pos (sub_pos.mpr hSL) hc)
  have hgap : 0 < kvaRowSaving L S c f p + kvaRowPremium L S c κ := by linarith
  have hBE : kvaBreakEvenR L N S T c f p κ < R := by
    unfold kvaBreakEvenR kvaRowSaving kvaRowPremium; exact hR
  have hBE0 : 0 ≤ kvaBreakEvenR L N S T c f p κ := by
    unfold kvaBreakEvenR
    exact div_nonneg (mul_nonneg (sub_nonneg.mpr hTN) hs) hgap.le
  have hR0 : 0 ≤ R := le_of_lt (lt_of_le_of_lt hBE0 hBE)
  exact (kva_speedupR_lt_one_iff_breakEven_lt hS1 hSL.le hT0 hR0 hRN hN hc hf0 hp
    (le_trans zero_le_one hκ1.le) hgap).mpr hBE

/-- (3, extreme case) Keeping every prefix row exact (`R = N − T`) at premium `κ > 1` is
strictly slower than exact prefill whenever there is a prefix (`T < N`) and a late part
(`S < L`). -/
theorem kva_speedupR_all_exact_lt_one (hS1 : 1 ≤ S) (hSL : S < L) (hT0 : 0 ≤ T)
    (hTN : T < N) (hc : 0 < c) (hf0 : 0 ≤ f) (hp : 0 ≤ p) (hκ1 : 1 < κ) :
    kvaSpeedupR L N S T c f p (N - T) κ < 1 := by
  have hN : 0 < N := lt_of_le_of_lt hT0 hTN
  rw [kva_speedupR_lt_one_iff hS1 hSL.le hT0 (sub_nonneg.mpr hTN.le) le_rfl hN hc hf0 hp
    (le_trans zero_le_one hκ1.le)]
  have : 0 < (N - T) * ((κ - 1) * ((L - S) * c)) :=
    mul_pos (sub_pos.mpr hTN) (mul_pos (sub_pos.mpr hκ1) (mul_pos (sub_pos.mpr hSL) hc))
  have h0 : (N - T - (N - T)) * ((L - S) * (c - f) - p) = 0 := by ring
  linarith

/-- (3, no premium) With `κ = 1` and no projector cost, exact prefix rows never make KVA
slower than exact prefill, for any `0 ≤ R ≤ N − T`. (They are just a longer exact tail.) -/
theorem kva_speedupR_ge_one_of_unit_premium (hS1 : 1 ≤ S) (hSL : S ≤ L) (hT0 : 0 ≤ T)
    (hR0 : 0 ≤ R) (hRN : R ≤ N - T) (hN : 0 < N) (hc : 0 < c)
    (hf0 : 0 ≤ f) (hfc : f ≤ c) :
    1 ≤ kvaSpeedupR L N S T c f 0 R 1 := by
  have h : kvaSpeedupR L N S T c f 0 R 1 = kvaSpeedup L N S (T + R) c f := by
    unfold kvaSpeedupR kvaSpeedup; rw [kvaCostKvaR_unit_premium]
  rw [h]
  exact kva_speedup_ge_one hS1 hSL (add_nonneg hT0 hR0) (by linarith) hN hc hf0 hfc

#print axioms kvaCostKvaR_zero
#print axioms kvaCostKvaR_unit_premium
#print axioms kvaCostExact_sub_kvaCostKvaR
#print axioms kvaCostKvaR_pos
#print axioms kva_speedupR_anti_R
#print axioms kva_speedupR_anti_R_of_premium
#print axioms kva_speedupR_anti_premium
#print axioms kva_speedupR_ge_one_iff
#print axioms kva_speedupR_ge_one_iff_total
#print axioms kva_speedupR_lt_one_iff
#print axioms kva_speedupR_ge_one_iff_le_breakEven
#print axioms kva_speedupR_lt_one_iff_breakEven_lt
#print axioms kva_speedupR_ge_one_iff_calib
#print axioms kvaBreakEvenR_mem
#print axioms kva_speedupR_lt_one_of_large
#print axioms kva_speedupR_all_exact_lt_one
#print axioms kva_speedupR_ge_one_of_unit_premium
