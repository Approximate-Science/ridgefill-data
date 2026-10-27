import KvaLean.ImpactBound
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# A count-matched random control measures selection skill, not row count

KVA keeps some prompt-prefix rows exact. A selection rule picks a set `S` of `k` of the `n`
prefix rows; it is judged against a count-matched random control, `k` rows drawn uniformly
from all `k`-subsets.

Rows are `j : Fin n`, each with a real *impact* `I j` (the benefit of keeping row `j`
exact; larger is better). Under the additive (first-order) model the gain of a set is
`gain I S = ∑_{j ∈ S} I j`. The control's expected gain is the average of `gain I R` over
`R ∈ powersetCard k univ` (`randomGain`), and the *skill* of `S` is
`skill I S = gain I S - randomGain I #S`.

1. Expected random gain (`card_kSubsets`, `card_kSubsets_mem`, `sum_gain_kSubsets`,
   `choose_ratio`, `randomGain_eq`): the `k`-subsets number `choose n k`; each row lies in
   `choose (n-1) (k-1)` of them; so `∑_R gain I R = choose (n-1) (k-1) * ∑_j I j` and
   `randomGain I k = (k / n) * ∑_j I j`.
2. Selection skill:
   - `skill_eq`, `skill_eq_sum_sub_mean`: `skill I S = ∑_{j∈S} I j - (#S/n) ∑_j I j
     = ∑_{j∈S} (I j - mean I)`.
   - `skill_pairwise`: `n * skill I S = ∑_{j ∈ S} ∑_{j' ∉ S} (I j - I j')`, a pairwise ranking
     score over (kept, dropped) pairs.
   - `skill_add_const`, `skill_const`: a uniform per-row benefit cancels exactly; with equal
     impacts every set of every size has skill `0`.
   - `unmatched_control_bias`: against a control of a different size `k'` the difference picks
     up the pure count term `(#S - k') * mean I`.
   - `skill_top_nonneg`, `skill_top_max`, `skill_bottom_nonpos`: the top-`k` set has skill
     `≥ 0` and the largest skill among sets of its size; the bottom-`k` set has skill `≤ 0`.
   - `skill_pos_exists_pair`, `skill_neg_exists_pair`: positive (negative) skill certifies a
     kept row with strictly larger (smaller) impact than some dropped row.
   - `sum_skill_kSubsets`, `sum_skill_relabel`: a selection independent of the impacts has
     zero skill in expectation, both for a uniformly random `S` and for a fixed `S` with the
     impacts relabelled by a uniformly random permutation.
   - Model-free versions for any set function `G` (no additivity): `sum_sub_avgSet`,
     `exists_ge_avgSet`.
3. First-order model (`randomGain_linear`, `randomGain_add`, `randomGain_full`,
   `randomGain_proportional`): the control's expected gain is linear in `k`, proportional to
   the count, and reaches the full gain `∑_j I j` at `k = n`.
4. Link to `ImpactBound.lean`: `skill_top_max` is proved from `topSet_compl_sum_le` (after a
   constant shift that makes the weights non-negative).
-/

open Finset

variable {n : ℕ}

/-! ## Definitions -/

/-- Gain of keeping the rows of `S` exact, under the additive model. -/
def gain (I : Fin n → ℝ) (S : Finset (Fin n)) : ℝ := ∑ j ∈ S, I j

/-- Average of a set function `G` over the `k`-subsets of `Fin n` (uniform random `k` rows).
If there are no `k`-subsets (`k > n`) this is `0 / 0 = 0`. -/
noncomputable def avgSet (G : Finset (Fin n) → ℝ) (k : ℕ) : ℝ :=
  (∑ R ∈ powersetCard k (univ : Finset (Fin n)), G R) / (n.choose k : ℝ)

/-- Expected gain of the count-matched random control with `k` rows. -/
noncomputable def randomGain (I : Fin n → ℝ) (k : ℕ) : ℝ := avgSet (gain I) k

/-- Selection skill: gain of `S` minus the expected gain of `#S` uniformly random rows. -/
noncomputable def skill (I : Fin n → ℝ) (S : Finset (Fin n)) : ℝ :=
  gain I S - randomGain I S.card

/-! ## 1. Expected random gain -/

/-- There are `choose n k` subsets of size `k`. -/
theorem card_kSubsets (k : ℕ) :
    (powersetCard k (univ : Finset (Fin n))).card = n.choose k := by
  simp

/-- For `k ≥ 1`, each row lies in exactly `choose (n-1) (k-1)` of the `k`-subsets. -/
theorem card_kSubsets_mem {k : ℕ} (hk : 0 < k) (j : Fin n) :
    ((powersetCard k (univ : Finset (Fin n))).filter (fun R => j ∈ R)).card
      = (n - 1).choose (k - 1) := by
  have h := card_filter_powersetCard_subset ({j} : Finset (Fin n)) univ k
    (subset_univ _) (by simp; omega)
  simpa using h

/-- Double counting: the gains of all `k`-subsets add up to `choose (n-1) (k-1) * ∑_j I j`. -/
theorem sum_gain_kSubsets (I : Fin n → ℝ) {k : ℕ} (hk : 0 < k) :
    ∑ R ∈ powersetCard k (univ : Finset (Fin n)), gain I R
      = ((n - 1).choose (k - 1) : ℝ) * ∑ j, I j := by
  unfold gain
  calc ∑ R ∈ powersetCard k (univ : Finset (Fin n)), ∑ j ∈ R, I j
      = ∑ R ∈ powersetCard k (univ : Finset (Fin n)), ∑ j, if j ∈ R then I j else 0 := by
        refine Finset.sum_congr rfl fun R _ => ?_
        rw [← Finset.sum_filter]
        congr 1
        ext j
        simp
    _ = ∑ j, ∑ R ∈ powersetCard k (univ : Finset (Fin n)), if j ∈ R then I j else 0 :=
        Finset.sum_comm
    _ = ∑ j, ((n - 1).choose (k - 1) : ℝ) * I j := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [← Finset.sum_filter, Finset.sum_const, card_kSubsets_mem hk j, nsmul_eq_mul]
    _ = ((n - 1).choose (k - 1) : ℝ) * ∑ j, I j := by rw [Finset.mul_sum]

/-- The ratio of the two counts: `choose (n-1) (k-1) / choose n k = k / n`. -/
theorem choose_ratio (hn : 0 < n) {k : ℕ} (hk : 0 < k) (hkn : k ≤ n) :
    ((n - 1).choose (k - 1) : ℝ) / (n.choose k : ℝ) = (k : ℝ) / n := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  obtain ⟨l, rfl⟩ : ∃ l, k = l + 1 := ⟨k - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  have hid : ((m + 1 : ℕ) : ℝ) * (m.choose l : ℝ)
      = ((m + 1).choose (l + 1) : ℝ) * ((l + 1 : ℕ) : ℝ) := by
    exact_mod_cast Nat.add_one_mul_choose_eq m l
  have hc : ((m + 1).choose (l + 1) : ℝ) ≠ 0 := by
    exact_mod_cast (Nat.choose_pos hkn).ne'
  have hm : ((m + 1 : ℕ) : ℝ) ≠ 0 := by positivity
  rw [div_eq_div_iff hc hm]
  linarith

/-- **Expected random gain.** Uniformly random `k` rows (`k ≤ n`) gain on average
`(k / n) * ∑_j I j`. -/
theorem randomGain_eq (I : Fin n → ℝ) (hn : 0 < n) {k : ℕ} (hkn : k ≤ n) :
    randomGain I k = (k / n : ℝ) * ∑ j, I j := by
  unfold randomGain avgSet
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · simp [gain]
  · rw [sum_gain_kSubsets I hk, mul_div_right_comm, choose_ratio hn hk hkn]

/-! ## 2. Selection skill -/

/-- Skill is the gain of `S` minus `#S / n` of the total impact. -/
theorem skill_eq (I : Fin n → ℝ) (hn : 0 < n) (S : Finset (Fin n)) :
    skill I S = gain I S - (S.card / n : ℝ) * ∑ j, I j := by
  rw [skill, randomGain_eq I hn (by simpa using S.card_le_univ)]

/-- Skill is the summed excess of the selected rows over the mean impact. -/
theorem skill_eq_sum_sub_mean (I : Fin n → ℝ) (hn : 0 < n) (S : Finset (Fin n)) :
    skill I S = ∑ j ∈ S, (I j - (∑ i, I i) / n) := by
  rw [skill_eq I hn, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, gain]
  ring

/-- **Pairwise ranking form.** `n * skill I S` is the sum, over every kept row `j ∈ S` and
every dropped row `j' ∉ S`, of `I j - I j'`. -/
theorem skill_pairwise (I : Fin n → ℝ) (hn : 0 < n) (S : Finset (Fin n)) :
    (n : ℝ) * skill I S = ∑ j ∈ S, ∑ j' ∈ Sᶜ, (I j - I j') := by
  have htot := Finset.sum_add_sum_compl S I
  have hcard : (Sᶜ.card : ℝ) = n - S.card := by
    rw [Finset.card_compl, Fintype.card_fin, Nat.cast_sub (by simpa using S.card_le_univ)]
  have hn' : (n : ℝ) ≠ 0 := by positivity
  simp only [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul]
  rw [← Finset.mul_sum, skill_eq I hn, gain, hcard, ← htot]
  field_simp
  ring

/-- A uniform per-row benefit `c` cancels exactly: skill depends only on how impact differs
between rows, not on the number of rows kept. -/
theorem skill_add_const (I : Fin n → ℝ) (c : ℝ) (S : Finset (Fin n)) :
    skill (fun j => I j + c) S = skill I S := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · have : S = ∅ := Finset.eq_empty_of_isEmpty S
    subst this
    simp [skill, randomGain, avgSet, gain]
  · rw [skill_eq_sum_sub_mean _ hn, skill_eq_sum_sub_mean _ hn, Finset.sum_add_distrib,
      Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    have hn' : (n : ℝ) ≠ 0 := by positivity
    refine Finset.sum_congr rfl fun j _ => ?_
    field_simp
    ring

/-- With equal impacts on all rows, every selection of every size has skill `0`. -/
theorem skill_const (c : ℝ) (S : Finset (Fin n)) : skill (fun _ => c) S = 0 := by
  have h := skill_add_const (fun _ : Fin n => (0 : ℝ)) c S
  simp only [zero_add] at h
  rw [h]
  simp [skill, randomGain, avgSet, gain]

/-- **Count-matching matters.** Compared with a random control of a different size `k'`,
the difference is the skill plus a pure count term `(#S - k') * mean I`. -/
theorem unmatched_control_bias (I : Fin n → ℝ) (hn : 0 < n) (S : Finset (Fin n)) {k' : ℕ}
    (hk' : k' ≤ n) :
    gain I S - randomGain I k' = skill I S + ((S.card : ℝ) - k') * ((∑ j, I j) / n) := by
  rw [skill_eq I hn, randomGain_eq I hn hk']
  ring

/-- The top-`k` set (every kept impact `≥ τ ≥` every dropped impact) has skill `≥ 0`. -/
theorem skill_top_nonneg (I : Fin n → ℝ) (hn : 0 < n) (S : Finset (Fin n)) (τ : ℝ)
    (hin : ∀ j ∈ S, τ ≤ I j) (hout : ∀ j ∉ S, I j ≤ τ) : 0 ≤ skill I S := by
  have h : 0 ≤ (n : ℝ) * skill I S := by
    rw [skill_pairwise I hn]
    refine Finset.sum_nonneg fun j hj => Finset.sum_nonneg fun j' hj' => ?_
    have := hout j' (Finset.mem_compl.mp hj')
    have := hin j hj
    linarith
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  by_contra hc
  push Not at hc
  linarith [mul_neg_of_pos_of_neg hn' hc]

/-- The top-`k` set has the largest skill among all sets of the same size. Proved from
`topSet_compl_sum_le` (`ImpactBound.lean`) after shifting the impacts to be non-negative. -/
theorem skill_top_max (I : Fin n → ℝ) (S T : Finset (Fin n)) (τ : ℝ)
    (hin : ∀ j ∈ S, τ ≤ I j) (hout : ∀ j ∉ S, I j ≤ τ) (hcard : T.card = S.card) :
    skill I T ≤ skill I S := by
  set m := ∑ j, |I j|
  have hw : ∀ j, 0 ≤ I j + m := fun j => by
    have h1 : |I j| ≤ m := Finset.single_le_sum (f := fun j => |I j|)
      (fun i _ => abs_nonneg _) (Finset.mem_univ j)
    have h2 := neg_abs_le (I j)
    linarith
  have h := topSet_compl_sum_le (fun j => I j + m) hw S T (τ + m)
    (fun j hj => by linarith [hin j hj]) (fun j hj => by linarith [hout j hj])
    hcard.le
  simp only [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, Finset.card_compl,
    hcard] at h
  have hS := Finset.sum_add_sum_compl S I
  have hT := Finset.sum_add_sum_compl T I
  rw [skill, skill, hcard, gain, gain]
  linarith

/-- The bottom-`k` set (every kept impact `≤ τ ≤` every dropped impact) has skill `≤ 0`. -/
theorem skill_bottom_nonpos (I : Fin n → ℝ) (hn : 0 < n) (S : Finset (Fin n)) (τ : ℝ)
    (hin : ∀ j ∈ S, I j ≤ τ) (hout : ∀ j ∉ S, τ ≤ I j) : skill I S ≤ 0 := by
  have h : (n : ℝ) * skill I S ≤ 0 := by
    rw [skill_pairwise I hn]
    refine Finset.sum_nonpos fun j hj => Finset.sum_nonpos fun j' hj' => ?_
    have := hout j' (Finset.mem_compl.mp hj')
    have := hin j hj
    linarith
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  by_contra hc
  push Not at hc
  linarith [mul_pos hn' hc]

/-- Positive skill certifies a correctly ranked pair: some kept row has strictly larger
impact than some dropped row. -/
theorem skill_pos_exists_pair (I : Fin n → ℝ) (hn : 0 < n) (S : Finset (Fin n))
    (h : 0 < skill I S) : ∃ j ∈ S, ∃ j' ∉ S, I j' < I j := by
  by_contra hc
  push Not at hc
  have : (n : ℝ) * skill I S ≤ 0 := by
    rw [skill_pairwise I hn]
    refine Finset.sum_nonpos fun j hj => Finset.sum_nonpos fun j' hj' => ?_
    have := hc j hj j' (Finset.mem_compl.mp hj')
    linarith
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  linarith [mul_pos hn' h]

/-- Negative skill (worse than the random control) certifies a misranked pair: some kept row
has strictly smaller impact than some dropped row. -/
theorem skill_neg_exists_pair (I : Fin n → ℝ) (hn : 0 < n) (S : Finset (Fin n))
    (h : skill I S < 0) : ∃ j ∈ S, ∃ j' ∉ S, I j < I j' := by
  by_contra hc
  push Not at hc
  have : 0 ≤ (n : ℝ) * skill I S := by
    rw [skill_pairwise I hn]
    refine Finset.sum_nonneg fun j hj => Finset.sum_nonneg fun j' hj' => ?_
    have := hc j hj j' (Finset.mem_compl.mp hj')
    linarith
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  linarith [mul_neg_of_pos_of_neg hn' h]

/-! ### Model-free versions (any set function, no additivity) -/

/-- For any set function `G`, deviations from the random-control average sum to zero over
the `k`-subsets: a uniformly random selection has zero skill in expectation. -/
theorem sum_sub_avgSet (G : Finset (Fin n) → ℝ) (k : ℕ) :
    ∑ S ∈ powersetCard k (univ : Finset (Fin n)), (G S - avgSet G k) = 0 := by
  rw [Finset.sum_sub_distrib, Finset.sum_const, card_kSubsets, nsmul_eq_mul, avgSet]
  rcases eq_or_ne (n.choose k : ℝ) 0 with h | h
  · have hk : n < k := by
      by_contra hk
      push Not at hk
      exact (Nat.choose_pos hk).ne' (by exact_mod_cast h)
    rw [Finset.powersetCard_eq_empty.mpr (by simpa using hk)]
    simp
  · field_simp
    ring

/-- For any set function `G` and `k ≤ n`, some `k`-set does at least as well as the random
control's average: the best `k`-set has skill `≥ 0` even without additivity. -/
theorem exists_ge_avgSet (G : Finset (Fin n) → ℝ) {k : ℕ} (hkn : k ≤ n) :
    ∃ S ∈ powersetCard k (univ : Finset (Fin n)), avgSet G k ≤ G S := by
  have hne : (powersetCard k (univ : Finset (Fin n))).Nonempty :=
    Finset.powersetCard_nonempty.mpr (by simpa using hkn)
  have h := sum_sub_avgSet G k
  by_contra hc
  push Not at hc
  have : ∑ S ∈ powersetCard k (univ : Finset (Fin n)), (G S - avgSet G k) < 0 :=
    Finset.sum_neg (fun S hS => by linarith [hc S hS]) hne
  linarith

/-- Additive case of `sum_sub_avgSet`: summed over all `k`-subsets, skill is `0`. -/
theorem sum_skill_kSubsets (I : Fin n → ℝ) (k : ℕ) :
    ∑ S ∈ powersetCard k (univ : Finset (Fin n)), skill I S = 0 := by
  rw [← sum_sub_avgSet (gain I) k]
  refine Finset.sum_congr rfl fun S hS => ?_
  rw [skill, randomGain, (Finset.mem_powersetCard.mp hS).2]

/-- Over all relabellings `σ`, the `σ`-sum of `I (σ j)` does not depend on `j`. -/
theorem sum_perm_apply (I : Fin n → ℝ) (j : Fin n) :
    (n : ℝ) * ∑ σ : Equiv.Perm (Fin n), I (σ j) = (n.factorial : ℝ) * ∑ i, I i := by
  have hj : ∀ j' : Fin n, ∑ σ : Equiv.Perm (Fin n), I (σ j)
      = ∑ σ : Equiv.Perm (Fin n), I (σ j') := fun j' =>
    Fintype.sum_equiv (Equiv.mulRight (Equiv.swap j j')) _ _ (fun σ => by
      simp [Equiv.Perm.mul_apply, Equiv.swap_apply_right])
  calc (n : ℝ) * ∑ σ : Equiv.Perm (Fin n), I (σ j)
      = ∑ j' : Fin n, ∑ σ : Equiv.Perm (Fin n), I (σ j') := by
        rw [Finset.sum_congr rfl fun j' _ => (hj j').symm, Finset.sum_const, Finset.card_univ,
          Fintype.card_fin, nsmul_eq_mul]
    _ = ∑ σ : Equiv.Perm (Fin n), ∑ j' : Fin n, I (σ j') := Finset.sum_comm
    _ = ∑ σ : Equiv.Perm (Fin n), ∑ i, I i :=
        Finset.sum_congr rfl fun σ _ => Equiv.sum_comp σ I
    _ = (n.factorial : ℝ) * ∑ i, I i := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_perm, Fintype.card_fin,
          nsmul_eq_mul]

/-- **Selection independent of the impacts.** For a fixed set `S`, averaging over all
relabellings of the impacts (`I ∘ σ`, `σ` a uniformly random permutation of the rows) gives
zero skill. -/
theorem sum_skill_relabel (I : Fin n → ℝ) (hn : 0 < n) (S : Finset (Fin n)) :
    ∑ σ : Equiv.Perm (Fin n), skill (fun j => I (σ j)) S = 0 := by
  have hn' : (n : ℝ) ≠ 0 := by positivity
  have hmean : ∀ σ : Equiv.Perm (Fin n), (∑ i, I (σ i)) = ∑ i, I i := fun σ =>
    Equiv.sum_comp σ I
  simp only [skill_eq_sum_sub_mean _ hn, hmean]
  rw [Finset.sum_comm]
  refine Finset.sum_eq_zero fun j _ => ?_
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_perm,
    Fintype.card_fin, nsmul_eq_mul]
  have h := sum_perm_apply I j
  field_simp
  linarith

/-! ## 3. First-order model: the control's gain is linear in the count -/

/-- The control's expected gain is `k` times the mean impact. -/
theorem randomGain_linear (I : Fin n → ℝ) (hn : 0 < n) {k : ℕ} (hkn : k ≤ n) :
    randomGain I k = k * ((∑ j, I j) / n) := by
  rw [randomGain_eq I hn hkn]
  ring

/-- Additive in the count: `randomGain (k₁ + k₂) = randomGain k₁ + randomGain k₂`. -/
theorem randomGain_add (I : Fin n → ℝ) (hn : 0 < n) {k₁ k₂ : ℕ} (h : k₁ + k₂ ≤ n) :
    randomGain I (k₁ + k₂) = randomGain I k₁ + randomGain I k₂ := by
  rw [randomGain_linear I hn h, randomGain_linear I hn (by omega),
    randomGain_linear I hn (by omega)]
  push_cast
  ring

/-- Keeping all `n` rows exact gives the full gain `∑_j I j`. -/
theorem randomGain_full (I : Fin n → ℝ) (hn : 0 < n) : randomGain I n = ∑ j, I j := by
  rw [randomGain_eq I hn le_rfl]
  have hn' : (n : ℝ) ≠ 0 := by positivity
  rw [div_self hn', one_mul]

/-- Proportional to the count: `randomGain k = (k / n) * randomGain n`. -/
theorem randomGain_proportional (I : Fin n → ℝ) (hn : 0 < n) {k : ℕ} (hkn : k ≤ n) :
    randomGain I k = (k / n : ℝ) * randomGain I n := by
  rw [randomGain_full I hn, randomGain_eq I hn hkn]

#print axioms card_kSubsets
#print axioms card_kSubsets_mem
#print axioms sum_gain_kSubsets
#print axioms choose_ratio
#print axioms randomGain_eq
#print axioms skill_eq
#print axioms skill_eq_sum_sub_mean
#print axioms skill_pairwise
#print axioms skill_add_const
#print axioms skill_const
#print axioms unmatched_control_bias
#print axioms skill_top_nonneg
#print axioms skill_top_max
#print axioms skill_bottom_nonpos
#print axioms skill_pos_exists_pair
#print axioms skill_neg_exists_pair
#print axioms sum_sub_avgSet
#print axioms exists_ge_avgSet
#print axioms sum_skill_kSubsets
#print axioms sum_perm_apply
#print axioms sum_skill_relabel
#print axioms randomGain_linear
#print axioms randomGain_add
#print axioms randomGain_full
#print axioms randomGain_proportional
