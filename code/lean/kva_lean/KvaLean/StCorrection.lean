import KvaLean.GdnContraction
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# GDN terminal-state correction scaled by `ρ` (engine change E1)

Decay-gated linear recurrence with scalar gates, rows `t = 1..T`:
`S 0 = 0`, `S t = g t • S (t-1) + u t`, states in any real vector space `V`
(e.g. `Fin d → ℝ` or the GDN state `Matrix m n ℝ`).

- (1) Closed form: `S T = ∑_{j=1..T} w j • u j` with `w j = ∏_{k=j+1..T} g k`.
- (2) Approximated rows `A` get inputs `u j + e j`. The terminal-state error is
  `∑_{j∈A} w j • e j`. With `e j = μ + ε j` on `A` and the all-rows correction
  `C = (∑_j w j) • μ`, subtracting `ρ • C` with
  `ρ = (∑_{j∈A} w j) / ∑_j w j` leaves exactly `∑_{j∈A} w j • ε j`;
  `ρ` is the only scale that does this (when `μ ≠ 0`), the unscaled `C`
  over-corrects by `(∑_{j∉A} w j) • μ`, and `A = ∅` / `A = all` give `ρ = 0` / `ρ = 1`.
- (3) Fast forgetting: if `0 ≤ g k ≤ γ` and the last `m` rows are exact, each
  `w j` with `j ∈ A` is at most `γ^m` times its weight at time `T - m`; with
  `γ < 1`, `ρ ≤ γ^m / (1 - γ)` for every `T` (`γ = 0.4`, `m = 16`: `ρ < 10⁻⁶`).
- (4) Full delta rule (with the erase term, `GdnContraction.lean`): only a bound.
  If two GDN runs get identical inputs on the last `m` steps and `α ≤ γ ≤ 1`
  there, their terminal states differ by at most `γ^m` times their difference
  `m` steps earlier.
-/

open Finset

section Linear

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Decay-gated linear recurrence, rows `1..T`: `S 0 = 0`, `S (t+1) = g (t+1) • S t + u (t+1)`. -/
def glaState (g : ℕ → ℝ) (u : ℕ → V) : ℕ → V
  | 0 => 0
  | t + 1 => g (t + 1) • glaState g u t + u (t + 1)

/-- Decay weight of row `j` at time `T`: `w j = ∏_{k=j+1..T} g k` (product of later gates). -/
noncomputable def decayW (g : ℕ → ℝ) (T j : ℕ) : ℝ :=
  ∏ k ∈ Ioc j T, g k

/-- Total decay weight of all rows `1..T`: `W = ∑_{j=1..T} w j`. -/
noncomputable def wsum (g : ℕ → ℝ) (T : ℕ) : ℝ :=
  ∑ j ∈ Ioc 0 T, decayW g T j

/-- E1 scale: decay-weighted share of approximated rows, `ρ = (∑_{j∈A} w j) / W`. -/
noncomputable def rhoE1 (g : ℕ → ℝ) (T : ℕ) (A : Finset ℕ) : ℝ :=
  (∑ j ∈ A, decayW g T j) / wsum g T

/-- Inputs with the rows in `A` approximated: `u j + e j` on `A`, `u j` elsewhere. -/
def approxIn (A : Finset ℕ) (u e : ℕ → V) (j : ℕ) : V :=
  if j ∈ A then u j + e j else u j

/-- The latest row has weight `1` (empty product). -/
theorem decayW_self (g : ℕ → ℝ) (T : ℕ) : decayW g T T = 1 := by
  simp [decayW]

theorem decayW_nonneg (g : ℕ → ℝ) (hg : ∀ k, 0 ≤ g k) (T j : ℕ) : 0 ≤ decayW g T j :=
  Finset.prod_nonneg fun k _ => hg k

/-- One more step multiplies every earlier row's weight by the new gate. -/
theorem decayW_succ (g : ℕ → ℝ) {T j : ℕ} (hj : j ≤ T) :
    decayW g (T + 1) j = decayW g T j * g (T + 1) := by
  simp only [decayW]
  exact Finset.prod_Ioc_succ_top hj g

/-- (1) Closed form: `S T = ∑_{j=1..T} w j • u j`. -/
theorem glaState_closed (g : ℕ → ℝ) (u : ℕ → V) (T : ℕ) :
    glaState g u T = ∑ j ∈ Ioc 0 T, decayW g T j • u j := by
  induction T with
  | zero => simp [glaState]
  | succ T ih =>
    rw [glaState, ih, Finset.sum_Ioc_succ_top (Nat.zero_le T), decayW_self, one_smul,
      Finset.smul_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro j hj
    rw [decayW_succ g (Finset.mem_Ioc.mp hj).2, smul_smul, mul_comm]

/-- (2) Terminal-state error of approximating the rows in `A`: `∑_{j∈A} w j • e j`. -/
theorem st_error_eq (g : ℕ → ℝ) (u e : ℕ → V) {T : ℕ} {A : Finset ℕ}
    (hA : A ⊆ Ioc 0 T) :
    glaState g (approxIn A u e) T - glaState g u T = ∑ j ∈ A, decayW g T j • e j := by
  rw [glaState_closed, glaState_closed, ← Finset.sum_sub_distrib]
  have hterm : ∀ j, decayW g T j • approxIn A u e j - decayW g T j • u j
      = decayW g T j • (if j ∈ A then e j else 0) := by
    intro j
    rw [← smul_sub]
    congr 1
    by_cases h : j ∈ A <;> simp [approxIn, h]
  simp_rw [hterm]
  symm
  calc ∑ j ∈ A, decayW g T j • e j
      = ∑ j ∈ A, decayW g T j • (if j ∈ A then e j else 0) :=
        Finset.sum_congr rfl fun j hj => by simp [hj]
    _ = _ := Finset.sum_subset hA fun j _ hj => by simp [hj]

/-- (2) Error after subtracting any multiple `c • C` of the all-rows correction
`C = W • μ`, when `e j = μ + ε j` on `A`: zero-mean part plus `(∑_{j∈A} w j - c W) • μ`. -/
theorem st_error_any_scale (g : ℕ → ℝ) (u e ε : ℕ → V) (μ : V) (c : ℝ) {T : ℕ}
    {A : Finset ℕ} (hA : A ⊆ Ioc 0 T) (he : ∀ j ∈ A, e j = μ + ε j) :
    (glaState g (approxIn A u e) T - c • (wsum g T • μ)) - glaState g u T
      = ∑ j ∈ A, decayW g T j • ε j
        + ((∑ j ∈ A, decayW g T j) - c * wsum g T) • μ := by
  have herr := st_error_eq g u e hA
  have hsplit : ∑ j ∈ A, decayW g T j • e j
      = (∑ j ∈ A, decayW g T j) • μ + ∑ j ∈ A, decayW g T j • ε j := by
    rw [Finset.sum_smul, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j hj => by rw [he j hj, smul_add]
  rw [sub_right_comm, herr, hsplit, smul_smul, sub_smul]
  abel

/-- (2) Main: with `W ≠ 0`, subtracting `ρ • C` leaves exactly the zero-mean part
`∑_{j∈A} w j • ε j`. -/
theorem st_correction_exact (g : ℕ → ℝ) (u e ε : ℕ → V) (μ : V) {T : ℕ}
    {A : Finset ℕ} (hA : A ⊆ Ioc 0 T) (he : ∀ j ∈ A, e j = μ + ε j)
    (hW : wsum g T ≠ 0) :
    (glaState g (approxIn A u e) T - rhoE1 g T A • (wsum g T • μ)) - glaState g u T
      = ∑ j ∈ A, decayW g T j • ε j := by
  rw [st_error_any_scale g u e ε μ _ hA he, rhoE1, div_mul_cancel₀ _ hW, sub_self,
    zero_smul, add_zero]

/-- (2) Uniqueness: if `μ ≠ 0` and `W ≠ 0`, `ρ` is the only scale `c` for which
subtracting `c • C` leaves exactly the zero-mean part. -/
theorem st_scale_unique (g : ℕ → ℝ) (u e ε : ℕ → V) (μ : V) (c : ℝ) {T : ℕ}
    {A : Finset ℕ} (hA : A ⊆ Ioc 0 T) (he : ∀ j ∈ A, e j = μ + ε j)
    (hW : wsum g T ≠ 0) (hμ : μ ≠ 0)
    (hc : (glaState g (approxIn A u e) T - c • (wsum g T • μ)) - glaState g u T
      = ∑ j ∈ A, decayW g T j • ε j) :
    c = rhoE1 g T A := by
  rw [st_error_any_scale g u e ε μ c hA he, add_eq_left, smul_eq_zero] at hc
  rcases hc with hc | hc
  · rw [rhoE1, eq_div_iff hW]
    linarith
  · exact absurd hc hμ

/-- (2) The unscaled correction (original `+st`, `c = 1`) over-corrects when some rows
are exact: the residual mean part is `-(∑_{j∉A} w j) • μ`. -/
theorem st_unscaled_overcorrects (g : ℕ → ℝ) (u e ε : ℕ → V) (μ : V) {T : ℕ}
    {A : Finset ℕ} (hA : A ⊆ Ioc 0 T) (he : ∀ j ∈ A, e j = μ + ε j) :
    (glaState g (approxIn A u e) T - wsum g T • μ) - glaState g u T
      = ∑ j ∈ A, decayW g T j • ε j - (∑ j ∈ Ioc 0 T \ A, decayW g T j) • μ := by
  have h := st_error_any_scale g u e ε μ 1 hA he
  rw [one_smul, one_mul] at h
  rw [h, wsum, ← Finset.sum_sdiff hA]
  have : (∑ j ∈ A, decayW g T j)
      - ((∑ j ∈ Ioc 0 T \ A, decayW g T j) + ∑ j ∈ A, decayW g T j)
      = -(∑ j ∈ Ioc 0 T \ A, decayW g T j) := by ring
  rw [this, neg_smul, sub_eq_add_neg]

/-- (2) With every row approximated, the terminal-state error is `C + ∑_j w j • ε j`:
`C = W • μ` is exactly the mean part that the original `+st` correction removes. -/
theorem st_full_error_decomp (g : ℕ → ℝ) (u e ε : ℕ → V) (μ : V) (T : ℕ)
    (he : ∀ j ∈ Ioc 0 T, e j = μ + ε j) :
    glaState g (approxIn (Ioc 0 T) u e) T - glaState g u T
      = wsum g T • μ + ∑ j ∈ Ioc 0 T, decayW g T j • ε j := by
  have h := st_error_any_scale g u e ε μ 0 (subset_refl _) he
  rw [zero_smul, sub_zero, zero_mul, sub_zero] at h
  rw [h, add_comm, wsum]

/-- (2) Endpoint `A = ∅`: `ρ = 0`. -/
theorem rhoE1_empty (g : ℕ → ℝ) (T : ℕ) : rhoE1 g T ∅ = 0 := by
  simp [rhoE1]

/-- (2) Endpoint `A = ∅`: no correction is applied and the terminal state is exact. -/
theorem st_empty_exact (g : ℕ → ℝ) (u e : ℕ → V) (μ : V) (T : ℕ) :
    glaState g (approxIn ∅ u e) T - rhoE1 g T ∅ • (wsum g T • μ) = glaState g u T := by
  have hin : approxIn (∅ : Finset ℕ) u e = u := by
    funext j
    simp [approxIn]
  rw [hin, rhoE1_empty, zero_smul, sub_zero]

/-- (2) Endpoint `A = all rows`: `ρ = 1`, i.e. E1 reduces to the original `+st`. -/
theorem rhoE1_all (g : ℕ → ℝ) (T : ℕ) (hW : wsum g T ≠ 0) : rhoE1 g T (Ioc 0 T) = 1 := by
  rw [rhoE1, ← wsum, div_self hW]

/-- With non-negative gates and `T ≥ 1`, `W ≥ 1` (the latest row has weight `1`),
so `W ≠ 0` holds automatically for real GDN gates. -/
theorem wsum_ge_one (g : ℕ → ℝ) (hg : ∀ k, 0 ≤ g k) {T : ℕ} (hT : 1 ≤ T) :
    1 ≤ wsum g T := by
  have hmem : T ∈ Ioc 0 T := Finset.mem_Ioc.mpr ⟨by omega, le_refl T⟩
  have h := Finset.single_le_sum (fun j _ => decayW_nonneg g hg T j) hmem
  rw [decayW_self] at h
  exact h

/-- `0 ≤ ρ` for non-negative gates. -/
theorem rhoE1_nonneg (g : ℕ → ℝ) (hg : ∀ k, 0 ≤ g k) (T : ℕ) (A : Finset ℕ) :
    0 ≤ rhoE1 g T A :=
  div_nonneg (Finset.sum_nonneg fun j _ => decayW_nonneg g hg T j)
    (Finset.sum_nonneg fun j _ => decayW_nonneg g hg T j)

/-- `ρ ≤ 1` for non-negative gates. -/
theorem rhoE1_le_one (g : ℕ → ℝ) (hg : ∀ k, 0 ≤ g k) {T : ℕ} {A : Finset ℕ}
    (hA : A ⊆ Ioc 0 T) : rhoE1 g T A ≤ 1 := by
  have hle : ∑ j ∈ A, decayW g T j ≤ wsum g T :=
    Finset.sum_le_sum_of_subset_of_nonneg hA fun j _ _ => decayW_nonneg g hg T j
  have hW0 : 0 ≤ wsum g T := Finset.sum_nonneg fun j _ => decayW_nonneg g hg T j
  rcases hW0.eq_or_lt with h0 | hpos
  · rw [rhoE1, ← h0, div_zero]
    exact zero_le_one
  · exact (div_le_one hpos).mpr hle

end Linear

section FastForgetting

/-- (3) Per-row bound: if `0 ≤ g k ≤ γ` and row `j ≤ T - m`, then
`w_T j ≤ γ^m · w_{T-m} j`. -/
theorem decayW_tail_le (g : ℕ → ℝ) (γ : ℝ) (hg : ∀ k, 0 ≤ g k ∧ g k ≤ γ) {T m j : ℕ}
    (hm : m ≤ T) (hj : j ≤ T - m) :
    decayW g T j ≤ γ ^ m * decayW g (T - m) j := by
  have hsplit : decayW g T j = decayW g (T - m) j * ∏ k ∈ Ioc (T - m) T, g k := by
    simp only [decayW]
    exact (Finset.prod_Ioc_consecutive g hj (Nat.sub_le T m)).symm
  have htail : ∏ k ∈ Ioc (T - m) T, g k ≤ γ ^ m := by
    calc ∏ k ∈ Ioc (T - m) T, g k ≤ ∏ _k ∈ Ioc (T - m) T, γ :=
          Finset.prod_le_prod₀ (fun k _ => (hg k).1) (fun k _ => (hg k).2)
      _ = γ ^ m := by rw [Finset.prod_const, Nat.card_Ioc]; congr 1; omega
  rw [hsplit, mul_comm (γ ^ m)]
  exact mul_le_mul_of_nonneg_left htail (decayW_nonneg g (fun k => (hg k).1) _ _)

/-- (3) Summed over the approximated rows (all at or before `T - m`):
`∑_{j∈A} w_T j ≤ γ^m · ∑_{j∈A} w_{T-m} j`. -/
theorem wsumA_tail_le (g : ℕ → ℝ) (γ : ℝ) (hg : ∀ k, 0 ≤ g k ∧ g k ≤ γ) {T m : ℕ}
    {A : Finset ℕ} (hm : m ≤ T) (hA : A ⊆ Ioc 0 (T - m)) :
    ∑ j ∈ A, decayW g T j ≤ γ ^ m * ∑ j ∈ A, decayW g (T - m) j := by
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun j hj =>
    decayW_tail_le g γ hg hm (Finset.mem_Ioc.mp (hA hj)).2

/-- (3) `ρ ≤ γ^m · (∑_{j∈A} w_{T-m} j) / W` when the last `m` rows are exact. -/
theorem rhoE1_tail_le (g : ℕ → ℝ) (γ : ℝ) (hg : ∀ k, 0 ≤ g k ∧ g k ≤ γ) {T m : ℕ}
    {A : Finset ℕ} (hm : m ≤ T) (hA : A ⊆ Ioc 0 (T - m)) :
    rhoE1 g T A ≤ γ ^ m * (∑ j ∈ A, decayW g (T - m) j) / wsum g T := by
  rw [rhoE1, mul_div_assoc]
  have hW : 0 ≤ wsum g T :=
    Finset.sum_nonneg fun j _ => decayW_nonneg g (fun k => (hg k).1) T j
  rw [← mul_div_assoc]
  exact div_le_div_of_nonneg_right (wsumA_tail_le g γ hg hm hA) hW

/-- The total weight is the recurrence driven by constant input `1`. -/
theorem wsum_eq_glaState_one (g : ℕ → ℝ) (T : ℕ) :
    wsum g T = glaState g (fun _ => (1 : ℝ)) T := by
  rw [glaState_closed, wsum]
  simp

/-- With `0 ≤ g k ≤ γ < 1`, the total weight is at most `1 / (1 - γ)` for every `T`. -/
theorem wsum_le_geom (g : ℕ → ℝ) (γ : ℝ) (hg : ∀ k, 0 ≤ g k ∧ g k ≤ γ) (hγ : γ < 1)
    (T : ℕ) : wsum g T ≤ 1 / (1 - γ) := by
  rw [wsum_eq_glaState_one]
  have h1 : 0 < 1 - γ := by linarith
  have key : ∀ t, 0 ≤ glaState g (fun _ => (1 : ℝ)) t
      ∧ glaState g (fun _ => (1 : ℝ)) t ≤ 1 / (1 - γ) := by
    intro t
    induction t with
    | zero => exact ⟨le_refl 0, by simp only [glaState]; positivity⟩
    | succ t ih =>
      obtain ⟨h0, hle⟩ := ih
      obtain ⟨hg0, hgγ⟩ := hg (t + 1)
      simp only [glaState, smul_eq_mul]
      refine ⟨by positivity, ?_⟩
      rw [le_div_iff₀ h1] at hle ⊢
      nlinarith [mul_le_mul hgγ (le_refl (glaState g (fun _ => (1 : ℝ)) t)) h0
        (le_trans hg0 hgγ)]
  exact (key T).2

/-- (3) T-independent bound: if `0 ≤ g k ≤ γ < 1`, `m ≤ T` and every approximated row
is at or before `T - m` (the last `m` rows exact), then `ρ ≤ γ^m / (1 - γ)`. -/
theorem rhoE1_le_geom (g : ℕ → ℝ) (γ : ℝ) (hg : ∀ k, 0 ≤ g k ∧ g k ≤ γ) (hγ : γ < 1)
    {T m : ℕ} {A : Finset ℕ} (hm : m ≤ T) (hA : A ⊆ Ioc 0 (T - m)) :
    rhoE1 g T A ≤ γ ^ m / (1 - γ) := by
  have hg0 : ∀ k, 0 ≤ g k := fun k => (hg k).1
  have hγ0 : 0 ≤ γ := le_trans (hg 0).1 (hg 0).2
  have hpos : 0 < 1 - γ := by linarith
  have hrhs : 0 ≤ γ ^ m / (1 - γ) := div_nonneg (pow_nonneg hγ0 m) hpos.le
  rcases Nat.eq_zero_or_pos T with hT | hT
  · -- `T = 0`: no rows, `A = ∅`.
    subst hT
    have hA0 : A = ∅ := Finset.subset_empty.mp (by simpa using hA)
    rw [hA0, rhoE1_empty]
    exact hrhs
  · have hW1 := wsum_ge_one g hg0 hT
    have hnum0 : 0 ≤ ∑ j ∈ A, decayW g T j :=
      Finset.sum_nonneg fun j _ => decayW_nonneg g hg0 T j
    -- `ρ ≤ ∑_{j∈A} w_T j` since `W ≥ 1`.
    have h1 : rhoE1 g T A ≤ ∑ j ∈ A, decayW g T j := by
      rw [rhoE1]
      exact div_le_self hnum0 hW1
    -- `∑_{j∈A} w_{T-m} j ≤ W_{T-m} ≤ 1/(1-γ)`.
    have h2 : ∑ j ∈ A, decayW g (T - m) j ≤ 1 / (1 - γ) :=
      le_trans (Finset.sum_le_sum_of_subset_of_nonneg hA
        fun j _ _ => decayW_nonneg g hg0 (T - m) j) (wsum_le_geom g γ hg hγ (T - m))
    calc rhoE1 g T A ≤ ∑ j ∈ A, decayW g T j := h1
      _ ≤ γ ^ m * ∑ j ∈ A, decayW g (T - m) j := wsumA_tail_le g γ hg hm hA
      _ ≤ γ ^ m * (1 / (1 - γ)) := mul_le_mul_of_nonneg_left h2 (pow_nonneg hγ0 m)
      _ = γ ^ m / (1 - γ) := by ring

/-- (3) Numerical case from the GPU smoke check: gates `≤ 0.4` and the 16 rows just
before the tail exact give `ρ < 10⁻⁶` (bound `0.4^16 / 0.6 ≈ 7.2·10⁻⁷`), for any `T`. -/
theorem rhoE1_fast_forget_numeric (g : ℕ → ℝ) (hg : ∀ k, 0 ≤ g k ∧ g k ≤ 0.4)
    {T : ℕ} {A : Finset ℕ} (hm : 16 ≤ T) (hA : A ⊆ Ioc 0 (T - 16)) :
    rhoE1 g T A < 1 / 1000000 := by
  have h := rhoE1_le_geom g 0.4 hg (by norm_num) hm hA
  have hnum : (0.4 : ℝ) ^ 16 / (1 - 0.4) < 1 / 1000000 := by norm_num
  exact lt_of_le_of_lt h hnum

end FastForgetting

section DeltaRule

open Matrix

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- (4) Full GDN step with the erase term (`GdnContraction.lean` convention, 0-indexed
steps): two runs that receive identical inputs `(α, β, k, v)` on the `L` steps
`n0, …, n0 + L - 1`, with `0 ≤ α t ≤ γ ≤ 1`, `β t ∈ [0, 2]` and unit `k t` there,
end at most `γ^L` times their state difference at step `n0` apart. Any earlier
difference (approximated rows changing `α, β, k` or `v`) is allowed. -/
theorem gdn_tail_exact_bound (S S' : ℕ → Matrix m n ℝ) (alp bet : ℕ → ℝ)
    (kk : ℕ → (n → ℝ)) (vv : ℕ → (m → ℝ)) (γ : ℝ) (hγ1 : γ ≤ 1) (n0 L : ℕ)
    (hk : ∀ t, n0 ≤ t → t < n0 + L → ∑ j, ((kk t) j) ^ 2 = 1)
    (ha : ∀ t, n0 ≤ t → t < n0 + L → 0 ≤ alp t ∧ alp t ≤ γ)
    (hb : ∀ t, n0 ≤ t → t < n0 + L → 0 ≤ bet t ∧ bet t ≤ 2)
    (hS : ∀ t, n0 ≤ t → t < n0 + L → S (t + 1)
      = alp t • (S t * ((1 : Matrix n n ℝ) - bet t • vecMulVec (kk t) (kk t)))
        + bet t • vecMulVec (vv t) (kk t))
    (hS' : ∀ t, n0 ≤ t → t < n0 + L → S' (t + 1)
      = alp t • (S' t * ((1 : Matrix n n ℝ) - bet t • vecMulVec (kk t) (kk t)))
        + bet t • vecMulVec (vv t) (kk t)) :
    frob (S (n0 + L) - S' (n0 + L)) ≤ γ ^ L * frob (S n0 - S' n0) := by
  have hc := gdn_prefix_contraction (T := L) (fun t => S (n0 + t)) (fun t => S' (n0 + t))
    (fun t => alp (n0 + t)) (fun t => bet (n0 + t)) (fun t => kk (n0 + t))
    (fun t => vv (n0 + t))
    (fun t ht => hk (n0 + t) (by omega) (by omega))
    (fun t ht => ⟨(ha (n0 + t) (by omega) (by omega)).1,
      le_trans (ha (n0 + t) (by omega) (by omega)).2 hγ1⟩)
    (fun t ht => hb (n0 + t) (by omega) (by omega))
    (fun t ht => hS (n0 + t) (by omega) (by omega))
    (fun t ht => hS' (n0 + t) (by omega) (by omega))
  simp only [add_zero] at hc
  have hprod : ∏ t ∈ range L, alp (n0 + t) ≤ γ ^ L := by
    calc ∏ t ∈ range L, alp (n0 + t) ≤ ∏ _t ∈ range L, γ :=
          Finset.prod_le_prod₀ (fun t ht => (ha (n0 + t) (by omega)
            (by have := Finset.mem_range.mp ht; omega)).1)
            (fun t ht => (ha (n0 + t) (by omega)
              (by have := Finset.mem_range.mp ht; omega)).2)
      _ = γ ^ L := by rw [Finset.prod_const, Finset.card_range]
  exact le_trans hc (mul_le_mul_of_nonneg_right hprod (frob_nonneg _))

end DeltaRule

#print axioms glaState_closed
#print axioms st_error_eq
#print axioms st_error_any_scale
#print axioms st_correction_exact
#print axioms st_scale_unique
#print axioms st_unscaled_overcorrects
#print axioms st_full_error_decomp
#print axioms rhoE1_empty
#print axioms st_empty_exact
#print axioms rhoE1_all
#print axioms wsum_ge_one
#print axioms rhoE1_nonneg
#print axioms rhoE1_le_one
#print axioms decayW_tail_le
#print axioms wsumA_tail_le
#print axioms rhoE1_tail_le
#print axioms wsum_le_geom
#print axioms rhoE1_le_geom
#print axioms rhoE1_fast_forget_numeric
#print axioms gdn_tail_exact_bound
