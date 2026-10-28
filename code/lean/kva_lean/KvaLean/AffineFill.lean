import KvaLean.QkNormInvariance
import Mathlib.LinearAlgebra.AffineSpace.AffineMap
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# Exact affine fills and the non-affinity of RMS normalization

An affine fill `A x + b` agrees with a target on every input exactly when the
target preserves every parametrized line, including extrapolation. Mathlib's
`AffineMap.apply_lineMap` gives the forward implication. For the converse we
construct the linear part `f x - f 0` from the line condition.

For positive dimension and `ε > 0`, RMS normalization violates this condition
at `0, e₀, 2e₀`. A learned elementwise gain preserves the violation whenever
its zeroth coordinate is nonzero. `no_affine_fill_matches_rmsnorm` packages
both conclusions for Table H.1 of the RidgeFill paper.

These statements concern equality on the whole real input space. They do not
assert non-affinity of an arbitrary composition containing RMS normalization,
nor impose a computational lower bound or a condition on a finite prompt set.
-/

section AffineFill

variable {V W : Type*} [AddCommGroup V] [Module ℝ V]
  [AddCommGroup W] [Module ℝ W]

/-- A target has an exact affine fill iff it preserves every parametrized line. -/
theorem affine_fill_iff_line_condition (f : V → W) :
    (∃ A : V →ₗ[ℝ] W, ∃ b : W, ∀ x, f x = A x + b) ↔
      (∀ x y : V, ∀ t : ℝ,
        f (x + t • (y - x)) = f x + t • (f y - f x)) := by
  constructor
  · rintro ⟨A, b, hf⟩ x y t
    let F : V →ᵃ[ℝ] W := A.toAffineMap + AffineMap.const ℝ V b
    have hF (v : V) : F v = f v := (hf v).symm
    simpa only [AffineMap.lineMap_apply_module', add_comm, hF] using
      F.apply_lineMap x y t
  · intro h
    have hsmul (t : ℝ) (x : V) :
        f (t • x) - f 0 = t • (f x - f 0) := by
      have ht := h 0 x t
      simp only [sub_zero, zero_add] at ht
      rw [ht, add_sub_cancel_left]
    have hadd (x y : V) :
        f (x + y) - f 0 = (f x - f 0) + (f y - f 0) := by
      have hpoint : (2 : ℝ) • (x + (1 / 2 : ℝ) • (y - x)) = x + y := by
        module
      rw [← hpoint, hsmul, h x y (1 / 2 : ℝ)]
      module
    let A : V →ₗ[ℝ] W :=
      { toFun := fun x => f x - f 0
        map_add' := hadd
        map_smul' := hsmul }
    exact ⟨A, f 0, fun x => (sub_add_cancel (f x) (f 0)).symm⟩

/-- One violation of the line condition rules out every exact affine fill. -/
theorem no_affine_fill_of_line_violation (f : V → W) (x y : V) (t : ℝ)
    (h : f (x + t • (y - x)) ≠ f x + t • (f y - f x)) :
    ¬ ∃ A : V →ₗ[ℝ] W, ∃ b : W, ∀ v, f v = A v + b := by
  intro hf
  exact h ((affine_fill_iff_line_condition f).mp hf x y t)

end AffineFill

/-- RMS normalization without gain, with the stabilizer inside the square root. -/
noncomputable def rmsNormalize {n : ℕ} (ε : ℝ) (x : Fin n → ℝ) : Fin n → ℝ :=
  (Real.sqrt ((∑ i, x i ^ 2) / n + ε))⁻¹ • x

/-- RMS normalization sends zero to zero, for every stabilizer. -/
theorem rmsNormalize_zero {n : ℕ} (ε : ℝ) :
    rmsNormalize ε (0 : Fin n → ℝ) = 0 := by
  simp [rmsNormalize]

/-- The existing learned-gain RMSNorm is elementwise gain after `rmsNormalize`. -/
theorem rmsNormEps_eq_gain_rmsNormalize {n : ℕ} (ε : ℝ) (g x : Fin n → ℝ) :
    rmsNormEps ε g x = fun i => g i * rmsNormalize ε x i := by
  ext i
  simp only [rmsNormEps, rmsNormalize, rmsEps, meanSq, Pi.smul_apply,
    smul_eq_mul, div_eq_mul_inv]
  ring

/-- On a coordinate vector of size `c`, the sum of squares is `c²`. -/
theorem sum_sq_single {n : ℕ} (i : Fin n) (c : ℝ) :
    ∑ j, (Pi.single i c j) ^ 2 = c ^ 2 := by
  classical
  simp [Pi.single_apply]

/-- Doubling a coordinate vector violates homogeneity in its zeroth coordinate. -/
theorem rmsNormalize_two_smul_ne_at_zero {n : ℕ} (hn : 1 ≤ n) {ε : ℝ} (hε : 0 < ε) :
    let e : Fin n → ℝ := Pi.single ⟨0, hn⟩ 1
    rmsNormalize ε ((2 : ℝ) • e) ⟨0, hn⟩ ≠ 2 * rmsNormalize ε e ⟨0, hn⟩ := by
  let i : Fin n := ⟨0, hn⟩
  let e : Fin n → ℝ := Pi.single i 1
  change rmsNormalize ε ((2 : ℝ) • e) i ≠ 2 * rmsNormalize ε e i
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have he : ∑ j, e j ^ 2 = 1 := by
    simpa only [e, one_pow] using sum_sq_single i 1
  have htwo : ∑ j, (((2 : ℝ) • e) j) ^ 2 = 4 := by
    have hv : (2 : ℝ) • e = Pi.single i 2 := by
      ext j
      simp [e, Pi.single_apply, apply_ite]
    rw [hv, sum_sq_single]
    norm_num
  intro hi
  simp only [rmsNormalize] at hi
  rw [htwo, he] at hi
  simp only [Pi.smul_apply, smul_eq_mul, e, Pi.single_eq_same] at hi
  have hinv : (Real.sqrt (4 / n + ε))⁻¹ = (Real.sqrt (1 / n + ε))⁻¹ := by
    linarith
  have hsqrt := inv_injective hinv
  have harg : (4 : ℝ) / n + ε = 1 / n + ε :=
    (Real.sqrt_inj (by positivity) (by positivity)).mp hsqrt
  have hlt : (1 : ℝ) / n < 4 / n := div_lt_div_of_pos_right (by norm_num) hnR
  linarith

/-- Doubling a coordinate vector does not double its RMS-normalized value. -/
theorem rmsNormalize_two_smul_ne {n : ℕ} (hn : 1 ≤ n) {ε : ℝ} (hε : 0 < ε) :
    let e : Fin n → ℝ := Pi.single ⟨0, hn⟩ 1
    rmsNormalize ε ((2 : ℝ) • e) ≠ (2 : ℝ) • rmsNormalize ε e := by
  dsimp only
  intro h
  exact rmsNormalize_two_smul_ne_at_zero hn hε (congrFun h ⟨0, hn⟩)

/-- RMS normalization with positive stabilizer is not an affine map. -/
theorem rmsNormalize_not_affine {n : ℕ} (hn : 1 ≤ n) {ε : ℝ} (hε : 0 < ε) :
    ¬ ∃ A : (Fin n → ℝ) →ₗ[ℝ] (Fin n → ℝ), ∃ b,
      ∀ x, rmsNormalize ε x = A x + b := by
  let e : Fin n → ℝ := Pi.single ⟨0, hn⟩ 1
  apply no_affine_fill_of_line_violation (rmsNormalize ε) 0 e 2
  simpa only [sub_zero, zero_add, rmsNormalize_zero] using
    rmsNormalize_two_smul_ne hn hε

/-- A gain nonzero at coordinate zero preserves the RMSNorm line violation. -/
theorem rmsNormEps_two_smul_ne {n : ℕ} (hn : 1 ≤ n) {ε : ℝ} (hε : 0 < ε)
    (g : Fin n → ℝ) (hg : g ⟨0, hn⟩ ≠ 0) :
    let e : Fin n → ℝ := Pi.single ⟨0, hn⟩ 1
    rmsNormEps ε g ((2 : ℝ) • e) ≠ (2 : ℝ) • rmsNormEps ε g e := by
  dsimp only
  intro h
  have hi := congrFun h ⟨0, hn⟩
  simp only [rmsNormEps_eq_gain_rmsNormalize, Pi.smul_apply, smul_eq_mul] at hi
  rw [mul_left_comm (2 : ℝ)] at hi
  exact rmsNormalize_two_smul_ne_at_zero hn hε ((mul_left_cancel₀ hg) hi)

/-- Learned-gain RMSNorm is not affine if the zeroth gain is nonzero. -/
theorem rmsNormEps_not_affine {n : ℕ} (hn : 1 ≤ n) {ε : ℝ} (hε : 0 < ε)
    (g : Fin n → ℝ) (hg : g ⟨0, hn⟩ ≠ 0) :
    ¬ ∃ A : (Fin n → ℝ) →ₗ[ℝ] (Fin n → ℝ), ∃ b,
      ∀ x, rmsNormEps ε g x = A x + b := by
  let e : Fin n → ℝ := Pi.single ⟨0, hn⟩ 1
  have hzero : rmsNormEps ε g 0 = 0 := by
    ext i
    simp [rmsNormEps]
  apply no_affine_fill_of_line_violation (rmsNormEps ε g) 0 e 2
  simpa only [hzero, sub_zero, zero_add] using
    rmsNormEps_two_smul_ne hn hε g hg

/-- Table H.1: no affine fill matches RMSNorm on all inputs, without gain or
with any learned gain whose zeroth coordinate is nonzero. -/
theorem no_affine_fill_matches_rmsnorm {n : ℕ} (hn : 1 ≤ n) {ε : ℝ} (hε : 0 < ε) :
    (¬ ∃ A : (Fin n → ℝ) →ₗ[ℝ] (Fin n → ℝ), ∃ b,
      ∀ x, rmsNormalize ε x = A x + b) ∧
    (∀ g : Fin n → ℝ, g ⟨0, hn⟩ ≠ 0 →
      ¬ ∃ A : (Fin n → ℝ) →ₗ[ℝ] (Fin n → ℝ), ∃ b,
        ∀ x, rmsNormEps ε g x = A x + b) := by
  exact ⟨rmsNormalize_not_affine hn hε, rmsNormEps_not_affine hn hε⟩

#print axioms affine_fill_iff_line_condition
#print axioms no_affine_fill_of_line_violation
#print axioms rmsNormalize_not_affine
#print axioms rmsNormEps_not_affine
#print axioms no_affine_fill_matches_rmsnorm

