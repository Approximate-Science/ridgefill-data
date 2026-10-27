import KvaLean.VecNorm
import KvaLean.GdnContraction
import Mathlib.Order.Interval.Finset.Fin
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# Why per-token sums miss the joint change: cross terms

Quadratic `f x = ½ ⟪x, H x⟫ + ⟪b, x⟫` on `Fin N → ℝ`, and a perturbation split
into per-token pieces `δ j` (`j : Fin n`). With `H` symmetric:

  `f (x + ∑ j, δ j) - f x = ∑ j, (f (x + δ j) - f x) + ∑_{j < k} ⟪δ j, H δ k⟫`.

So the sum of single-token changes misses exactly the cross terms.

- `quad_joint_eq_general`: the identity for any `H`, with the symmetrised cross
  term `½ ∑_{j<k} (⟪δ j, H δ k⟫ + ⟪δ k, H δ j⟫)`.
- `quad_joint_eq`: the identity above, for symmetric `H` (`H.IsSymm`).
  Symmetry is needed: for non-symmetric `H` the stated cross term is wrong
  (`quad_joint_eq_needs_symm`); only its symmetrised version is correct.
  A Hessian is always symmetric.
- `perToken_gap`: joint change minus the per-token sum equals the cross term.
- `crossTerm_abs_le`, `crossTerm_abs_le_sq`, `crossTerm_abs_le_frob`: if
  `‖H u‖ ≤ L ‖u‖` for all `u`, then
  `|cross| ≤ L ∑_{j<k} ‖δ j‖ ‖δ k‖ = (L/2) ((∑ j, ‖δ j‖)² - ∑ j, ‖δ j‖²)`,
  i.e. `(L/2) ∑_{j≠k} ‖δ j‖ ‖δ k‖`; the Frobenius norm of `H` is such an `L`.

The pieces `δ j` need not have disjoint supports: the identity holds for any
decomposition of the perturbation, and per-token blocks are a special case.
-/

open Finset Matrix

variable {n N : ℕ}

/-! ## Pair sums over `j < k` -/

/-- A row sum splits at `j` into the parts below, at and above `j`. -/
theorem sum_split_at (F : Fin n → ℝ) (j : Fin n) :
    ∑ k, F k = ∑ k ∈ Iio j, F k + F j + ∑ k ∈ Ioi j, F k := by
  have h : ∀ k, F k = (if k < j then F k else 0) + (if k = j then F k else 0)
      + (if j < k then F k else 0) := by
    intro k
    rcases lt_trichotomy k j with h | h | h
    · simp [h, h.ne, not_lt.mpr h.le]
    · simp [h]
    · simp [h, h.ne', not_lt.mpr h.le]
  rw [Finset.sum_congr rfl (fun k _ => h k), Finset.sum_add_distrib, Finset.sum_add_distrib,
    Finset.sum_ite_eq' Finset.univ j, ite_eq_left (Finset.mem_univ _), ← Finset.sum_filter,
    ← Finset.sum_filter, Finset.filter_gt_eq_Iio, Finset.filter_lt_eq_Ioi]

/-- A double sum is the diagonal plus each unordered pair `j < k` counted both ways. -/
theorem sum_sum_split (F : Fin n → Fin n → ℝ) :
    ∑ j, ∑ k, F j k = ∑ j, F j j + ∑ j, ∑ k ∈ Ioi j, (F j k + F k j) := by
  have hswap : ∑ j, ∑ k ∈ Iio j, F j k = ∑ k, ∑ j ∈ Ioi k, F j k :=
    Finset.sum_comm' (fun j k => by simp)
  rw [Finset.sum_congr rfl (fun j _ => sum_split_at (F j) j), Finset.sum_add_distrib,
    Finset.sum_add_distrib, hswap]
  simp only [Finset.sum_add_distrib]
  ring

/-! ## The quadratic and its cross terms -/

/-- Quadratic `f x = ½ ⟪x, H x⟫ + ⟪b, x⟫`. -/
noncomputable def quadF (H : Matrix (Fin N) (Fin N) ℝ) (b x : Fin N → ℝ) : ℝ :=
  (1 / 2) * (x ⬝ᵥ (H *ᵥ x)) + b ⬝ᵥ x

/-- Cross term `∑_{j < k} ⟪δ j, H δ k⟫`. -/
def crossTerm (H : Matrix (Fin N) (Fin N) ℝ) (δ : Fin n → Fin N → ℝ) : ℝ :=
  ∑ j, ∑ k ∈ Ioi j, δ j ⬝ᵥ (H *ᵥ δ k)

theorem quadF_add_sub (H : Matrix (Fin N) (Fin N) ℝ) (b x y : Fin N → ℝ) :
    quadF H b (x + y) - quadF H b x
      = (1 / 2) * (x ⬝ᵥ (H *ᵥ y) + y ⬝ᵥ (H *ᵥ x) + y ⬝ᵥ (H *ᵥ y)) + b ⬝ᵥ y := by
  unfold quadF
  rw [mulVec_add, dotProduct_add, add_dotProduct, add_dotProduct, dotProduct_add]
  ring

/-- For symmetric `H`, `⟪u, H w⟫ = ⟪w, H u⟫`. -/
theorem dotProduct_mulVec_symm (H : Matrix (Fin N) (Fin N) ℝ) (hH : H.IsSymm)
    (u w : Fin N → ℝ) : u ⬝ᵥ (H *ᵥ w) = w ⬝ᵥ (H *ᵥ u) := by
  rw [dotProduct_mulVec, ← mulVec_transpose, hH.eq, dotProduct_comm]

/-- Joint change = sum of single-token changes + symmetrised cross terms (any `H`). -/
theorem quad_joint_eq_general (H : Matrix (Fin N) (Fin N) ℝ) (b x : Fin N → ℝ)
    (δ : Fin n → Fin N → ℝ) :
    quadF H b (x + ∑ j, δ j) - quadF H b x
      = ∑ j, (quadF H b (x + δ j) - quadF H b x)
        + (1 / 2) * ∑ j, ∑ k ∈ Ioi j, (δ j ⬝ᵥ (H *ᵥ δ k) + δ k ⬝ᵥ (H *ᵥ δ j)) := by
  have h1 : x ⬝ᵥ (H *ᵥ ∑ j, δ j) = ∑ j, x ⬝ᵥ (H *ᵥ δ j) := by
    rw [mulVec_sum, dotProduct_sum]
  have h2 : (∑ j, δ j) ⬝ᵥ (H *ᵥ x) = ∑ j, δ j ⬝ᵥ (H *ᵥ x) := sum_dotProduct _ _ _
  have h3 : b ⬝ᵥ ∑ j, δ j = ∑ j, b ⬝ᵥ δ j := dotProduct_sum _ _ _
  have h4 : (∑ j, δ j) ⬝ᵥ (H *ᵥ ∑ j, δ j)
      = ∑ j, δ j ⬝ᵥ (H *ᵥ δ j)
        + ∑ j, ∑ k ∈ Ioi j, (δ j ⬝ᵥ (H *ᵥ δ k) + δ k ⬝ᵥ (H *ᵥ δ j)) := by
    rw [mulVec_sum, sum_dotProduct]
    simp_rw [dotProduct_sum]
    exact sum_sum_split (fun j k => δ j ⬝ᵥ (H *ᵥ δ k))
  have h5 : ∑ j, (quadF H b (x + δ j) - quadF H b x)
      = (1 / 2) * (∑ j, x ⬝ᵥ (H *ᵥ δ j) + ∑ j, δ j ⬝ᵥ (H *ᵥ x)
          + ∑ j, δ j ⬝ᵥ (H *ᵥ δ j)) + ∑ j, b ⬝ᵥ δ j := by
    simp_rw [quadF_add_sub]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_add_distrib,
      Finset.sum_add_distrib]
  rw [quadF_add_sub, h1, h2, h3, h4, h5]
  ring

/-- Joint change = sum of single-token changes + `∑_{j<k} ⟪δ j, H δ k⟫` (symmetric `H`). -/
theorem quad_joint_eq (H : Matrix (Fin N) (Fin N) ℝ) (hH : H.IsSymm) (b x : Fin N → ℝ)
    (δ : Fin n → Fin N → ℝ) :
    quadF H b (x + ∑ j, δ j) - quadF H b x
      = ∑ j, (quadF H b (x + δ j) - quadF H b x) + crossTerm H δ := by
  rw [quad_joint_eq_general, crossTerm, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k _
  rw [dotProduct_mulVec_symm H hH (δ k) (δ j)]
  ring

/-- The sum of single-token changes misses exactly the cross terms. -/
theorem perToken_gap (H : Matrix (Fin N) (Fin N) ℝ) (hH : H.IsSymm) (b x : Fin N → ℝ)
    (δ : Fin n → Fin N → ℝ) :
    (quadF H b (x + ∑ j, δ j) - quadF H b x) - ∑ j, (quadF H b (x + δ j) - quadF H b x)
      = crossTerm H δ := by
  rw [quad_joint_eq H hH]
  ring

/-- Symmetry is needed in `quad_joint_eq`: for `H = !![0, 1; 0, 0]` (not symmetric),
`δ 0 = e₀`, `δ 1 = e₁`, `b = x = 0`, the true gap is `½` but `crossTerm` is `1`. -/
theorem quad_joint_eq_needs_symm :
    let H : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 0, 0]
    let δ : Fin 2 → Fin 2 → ℝ := ![![1, 0], ![0, 1]]
    (quadF H 0 (0 + ∑ j, δ j) - quadF H 0 0) - ∑ j, (quadF H 0 (0 + δ j) - quadF H 0 0)
      ≠ crossTerm H δ := by
  intro H δ
  simp [H, δ, quadF, crossTerm, Fin.sum_univ_two, dotProduct, Matrix.mulVec]

/-! ## Size of the cross terms -/

/-- If `‖H u‖ ≤ L ‖u‖` for all `u`, then `|cross| ≤ L ∑_{j<k} ‖δ j‖ ‖δ k‖`. -/
theorem crossTerm_abs_le (H : Matrix (Fin N) (Fin N) ℝ) (δ : Fin n → Fin N → ℝ) (L : ℝ)
    (hL : ∀ u, vnorm (H *ᵥ u) ≤ L * vnorm u) :
    |crossTerm H δ| ≤ L * ∑ j, ∑ k ∈ Ioi j, vnorm (δ j) * vnorm (δ k) := by
  unfold crossTerm
  rw [Finset.mul_sum]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum fun j _ => ?_)
  rw [Finset.mul_sum]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum fun k _ => ?_)
  calc |δ j ⬝ᵥ (H *ᵥ δ k)| ≤ vnorm (δ j) * vnorm (H *ᵥ δ k) := abs_dotProduct_le _ _
    _ ≤ vnorm (δ j) * (L * vnorm (δ k)) :=
        mul_le_mul_of_nonneg_left (hL _) (vnorm_nonneg _)
    _ = L * (vnorm (δ j) * vnorm (δ k)) := by ring

/-- Same bound in closed form: `|cross| ≤ (L/2) ((∑ j, ‖δ j‖)² - ∑ j, ‖δ j‖²)`,
which is `(L/2) ∑_{j≠k} ‖δ j‖ ‖δ k‖`. -/
theorem crossTerm_abs_le_sq (H : Matrix (Fin N) (Fin N) ℝ) (δ : Fin n → Fin N → ℝ) (L : ℝ)
    (hL : ∀ u, vnorm (H *ᵥ u) ≤ L * vnorm u) :
    |crossTerm H δ| ≤ (L / 2) * ((∑ j, vnorm (δ j)) ^ 2 - ∑ j, vnorm (δ j) ^ 2) := by
  have hsplit := sum_sum_split (fun j k => vnorm (δ j) * vnorm (δ k))
  have hsq : (∑ j, vnorm (δ j)) ^ 2 = ∑ j, ∑ k, vnorm (δ j) * vnorm (δ k) := by
    rw [sq, Finset.sum_mul_sum]
  have hpair : ∑ j, ∑ k ∈ Ioi j, (vnorm (δ j) * vnorm (δ k) + vnorm (δ k) * vnorm (δ j))
      = 2 * ∑ j, ∑ k ∈ Ioi j, vnorm (δ j) * vnorm (δ k) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k _
    ring
  have hdiag : ∑ j, vnorm (δ j) * vnorm (δ j) = ∑ j, vnorm (δ j) ^ 2 :=
    Finset.sum_congr rfl fun j _ => (sq _).symm
  have h := crossTerm_abs_le H δ L hL
  rw [hsq, hsplit, hpair, hdiag]
  linarith

/-- The Frobenius norm bounds the operator norm: `‖H u‖ ≤ ‖H‖_F ‖u‖`. -/
theorem vnorm_mulVec_le_frob (H : Matrix (Fin N) (Fin N) ℝ) (u : Fin N → ℝ) :
    vnorm (H *ᵥ u) ≤ frob H * vnorm u := by
  apply le_of_pow_le_pow_left₀ two_ne_zero (mul_nonneg (frob_nonneg H) (vnorm_nonneg u))
  rw [mul_pow, vnorm_sq, vnorm_sq, frob_sq, Finset.sum_mul]
  exact Finset.sum_le_sum fun r _ => Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (H r) u

/-- Cross-term bound with the Frobenius norm of `H`. -/
theorem crossTerm_abs_le_frob (H : Matrix (Fin N) (Fin N) ℝ) (δ : Fin n → Fin N → ℝ) :
    |crossTerm H δ| ≤ (frob H / 2) * ((∑ j, vnorm (δ j)) ^ 2 - ∑ j, vnorm (δ j) ^ 2) :=
  crossTerm_abs_le_sq H δ (frob H) (vnorm_mulVec_le_frob H)

#print axioms quad_joint_eq_general
#print axioms quad_joint_eq
#print axioms perToken_gap
#print axioms quad_joint_eq_needs_symm
#print axioms crossTerm_abs_le
#print axioms crossTerm_abs_le_sq
#print axioms crossTerm_abs_le_frob
