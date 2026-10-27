import Mathlib.LinearAlgebra.Matrix.Defs
import Mathlib.Data.Matrix.Mul
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# GDN prefix-error contraction

Gated delta-rule state update (one step):
`S₊ = α • (S * (1 - β • vecMulVec k k)) + β • vecMulVec v k`,
with decay gate `α ∈ [0,1]`, `β ∈ [0,1]` and unit vector `k` (`‖k‖₂ = 1`).

Two trajectories driven by the same inputs from different initial states satisfy
`‖S T - S' T‖ ≤ (∏_{t < T} α t) * ‖S 0 - S' 0‖`.

The norm is the Frobenius norm, defined directly as `√(∑ i, ∑ j, (A i j)^2)`
(Mathlib has no Frobenius norm on matrices; this is the standard definition).
The beta range is proved in the stronger form `β ∈ [0, 2]`: the key step is
`‖B‖² = ‖A‖² - β(2-β)‖Ak‖² ≤ ‖A‖²`.
-/

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n]

/-- Frobenius norm on real matrices (squared-sum square root). -/
noncomputable def frob (A : Matrix m n ℝ) : ℝ :=
  Real.sqrt (∑ i, ∑ j, (A i j) ^ 2)

theorem frob_nonneg (A : Matrix m n ℝ) : 0 ≤ frob A :=
  Real.sqrt_nonneg _

theorem frob_sq (A : Matrix m n ℝ) :
    (frob A) ^ 2 = ∑ i, ∑ j, (A i j) ^ 2 := by
  simp only [frob]
  exact Real.sq_sqrt (Finset.sum_nonneg fun _ _ =>
    Finset.sum_nonneg fun _ _ => sq_nonneg _)

theorem frob_smul (c : ℝ) (A : Matrix m n ℝ) :
    frob (c • A) = |c| * frob A := by
  simp only [frob]
  have hsum : ∑ i, ∑ j, ((c • A) i j) ^ 2
      = c ^ 2 * ∑ i, ∑ j, (A i j) ^ 2 := by
    simp only [Matrix.smul_apply, smul_eq_mul]
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [hsum, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq_eq_abs]

-- Entrywise form of one GDN multiplication step.
omit [Fintype m] in
theorem gdn_entry [DecidableEq n] (A : Matrix m n ℝ) (k : n → ℝ) (beta : ℝ) (i : m)
    (j : n) :
    (A * ((1 : Matrix n n ℝ) - beta • vecMulVec k k)) i j
      = A i j - beta * (∑ l, A i l * k l) * k j := by
  have hM : (A * (beta • vecMulVec k k)) i j
      = beta * (∑ l, A i l * k l) * k j := by
    simp only [Matrix.mul_apply, Matrix.smul_apply, smul_eq_mul,
      Matrix.vecMulVec_apply]
    trans ∑ l, (beta * (A i l * k l) * k j)
    · exact Finset.sum_congr rfl (fun l _ => by ring)
    · rw [← Finset.sum_mul, ← Finset.mul_sum]
  have hAM : A * ((1 : Matrix n n ℝ) - beta • vecMulVec k k)
      = A - A * (beta • vecMulVec k k) := by
    rw [Matrix.mul_sub, Matrix.mul_one]
  rw [hAM, Matrix.sub_apply, hM]

/-- Squared-norm identity: one step removes `β(2-β)‖Ak‖²`. -/
theorem frob_gdn_step_sq [DecidableEq n] (A : Matrix m n ℝ) (k : n → ℝ) (beta : ℝ)
    (hk : ∑ j, (k j) ^ 2 = 1) :
    (frob (A * ((1 : Matrix n n ℝ) - beta • vecMulVec k k))) ^ 2
      = (frob A) ^ 2
        - beta * (2 - beta) * (∑ i, (∑ l, A i l * k l) ^ 2) := by
  rw [frob_sq, frob_sq]
  have row : ∀ i, ∑ j, ((A * ((1 : Matrix n n ℝ) - beta • vecMulVec k k)) i j) ^ 2
      = (∑ j, (A i j) ^ 2)
        - beta * (2 - beta) * (∑ l, A i l * k l) ^ 2 := by
    intro i
    set y := ∑ l, A i l * k l with hy
    have e1 : ∀ j ∈ Finset.univ,
        ((A * ((1 : Matrix n n ℝ) - beta • vecMulVec k k)) i j) ^ 2
          = (A i j) ^ 2 - (2 * beta * y) * (A i j * k j)
            + ((beta * y) ^ 2) * (k j) ^ 2 := by
      intro j _
      rw [gdn_entry]
      ring_nf
      rw [hy]
      ring
    have hsum : ∑ j, ((A * ((1 : Matrix n n ℝ) - beta • vecMulVec k k)) i j) ^ 2
        = ∑ j, ((A i j) ^ 2 - (2 * beta * y) * (A i j * k j)
            + ((beta * y) ^ 2) * (k j) ^ 2) :=
      Finset.sum_congr rfl e1
    rw [hsum, Finset.sum_add_distrib, Finset.sum_sub_distrib]
    have m1 : ∑ j, (2 * beta * y) * (A i j * k j)
        = (2 * beta * y) * y := by
      rw [← Finset.mul_sum, hy]
    have m2 : ∑ j, ((beta * y) ^ 2) * (k j) ^ 2 = ((beta * y) ^ 2) * 1 := by
      rw [← Finset.mul_sum, hk]
    rw [m1, m2]
    ring
  calc ∑ i, ∑ j, ((A * ((1 : Matrix n n ℝ) - beta • vecMulVec k k)) i j) ^ 2
      = ∑ i, ((∑ j, (A i j) ^ 2)
          - beta * (2 - beta) * (∑ l, A i l * k l) ^ 2) :=
        Finset.sum_congr rfl (fun i _ => row i)
    _ = _ := by rw [Finset.sum_sub_distrib, ← Finset.mul_sum]

/-- Key step: right-multiplying by `(I - βkkᵀ)` does not increase the norm,
for unit `k` and `β ∈ [0, 2]`. -/
theorem frob_gdn_step_le [DecidableEq n] (A : Matrix m n ℝ) (k : n → ℝ) (beta : ℝ)
    (hk : ∑ j, (k j) ^ 2 = 1) (hb0 : 0 ≤ beta) (hb2 : beta ≤ 2) :
    frob (A * ((1 : Matrix n n ℝ) - beta • vecMulVec k k)) ≤ frob A := by
  have hnn : 0 ≤ beta * (2 - beta) * (∑ i, (∑ l, A i l * k l) ^ 2) :=
    mul_nonneg (mul_nonneg hb0 (by linarith))
      (Finset.sum_nonneg fun _ _ => sq_nonneg _)
  have hid := frob_gdn_step_sq A k beta hk
  simp only [frob]
  apply Real.sqrt_le_sqrt
  have hB := frob_sq (A * ((1 : Matrix n n ℝ) - beta • vecMulVec k k))
  have hA := frob_sq A
  linarith

/-- GDN prefix-error contraction over `T` steps. -/
theorem gdn_prefix_contraction [DecidableEq n] {T : ℕ}
    (S S' : ℕ → Matrix m n ℝ) (alp : ℕ → ℝ) (bet : ℕ → ℝ)
    (kk : ℕ → (n → ℝ)) (vv : ℕ → (m → ℝ))
    (hk : ∀ t, t < T → ∑ j, ((kk t) j) ^ 2 = 1)
    (ha : ∀ t, t < T → 0 ≤ alp t ∧ alp t ≤ 1)
    (hb : ∀ t, t < T → 0 ≤ bet t ∧ bet t ≤ 2)
    (hS : ∀ t, t < T → S (t + 1)
      = alp t • (S t * ((1 : Matrix n n ℝ) - bet t • vecMulVec (kk t) (kk t)))
        + bet t • vecMulVec (vv t) (kk t))
    (hS' : ∀ t, t < T → S' (t + 1)
      = alp t • (S' t * ((1 : Matrix n n ℝ) - bet t • vecMulVec (kk t) (kk t)))
        + bet t • vecMulVec (vv t) (kk t)) :
    frob (S T - S' T)
      ≤ (∏ t ∈ Finset.range T, alp t) * frob (S 0 - S' 0) := by
  induction T with
  | zero => simp
  | succ T ih =>
    have sub : ∀ t, t < T →
        (∑ j, ((kk t) j) ^ 2 = 1) ∧ (0 ≤ alp t ∧ alp t ≤ 1)
          ∧ (0 ≤ bet t ∧ bet t ≤ 2)
          ∧ (S (t + 1) = alp t • (S t * ((1 : Matrix n n ℝ) - bet t • vecMulVec (kk t) (kk t)))
              + bet t • vecMulVec (vv t) (kk t))
          ∧ (S' (t + 1) = alp t • (S' t * ((1 : Matrix n n ℝ) - bet t • vecMulVec (kk t) (kk t)))
              + bet t • vecMulVec (vv t) (kk t)) := by
      intro t ht
      exact ⟨hk t (by omega), ha t (by omega), hb t (by omega),
        hS t (by omega), hS' t (by omega)⟩
    have ih' := ih (fun t ht => (sub t ht).1)
      (fun t ht => (sub t ht).2.1) (fun t ht => (sub t ht).2.2.1)
      (fun t ht => (sub t ht).2.2.2.1) (fun t ht => (sub t ht).2.2.2.2)
    have hTT : T < T + 1 := by omega
    have haT := ha T hTT
    have hbT := hb T hTT
    have hkT := hk T hTT
    have hD : S (T + 1) - S' (T + 1)
        = alp T • ((S T - S' T)
          * ((1 : Matrix n n ℝ) - bet T • vecMulVec (kk T) (kk T))) := by
      rw [hS T hTT, hS' T hTT, add_sub_add_right_eq_sub, ← smul_sub,
        ← Matrix.sub_mul]
    have hstep := frob_gdn_step_le (S T - S' T) (kk T) (bet T) hkT
      hbT.1 hbT.2
    have hcomb : frob ((S T - S' T) * ((1 : Matrix n n ℝ) - bet T • vecMulVec (kk T) (kk T)))
        ≤ (∏ t ∈ Finset.range T, alp t) * frob (S 0 - S' 0) :=
      le_trans hstep ih'
    rw [hD, frob_smul, abs_of_nonneg haT.1, Finset.prod_range_succ]
    calc alp T * frob ((S T - S' T) * ((1 : Matrix n n ℝ) - bet T • vecMulVec (kk T) (kk T)))
        ≤ alp T * ((∏ t ∈ Finset.range T, alp t) * frob (S 0 - S' 0)) :=
          mul_le_mul_of_nonneg_left hcomb haT.1
      _ = (∏ t ∈ Finset.range T, alp t) * alp T * frob (S 0 - S' 0) := by
          ring

#print axioms frob_gdn_step_le
#print axioms gdn_prefix_contraction
