import KvaLean.AttnSensitivity

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# Context dependence rules out exact per-row fills

A fill `φ (x i)` sees only row `i` of a prompt. Two prompts with the same
row there but different target outputs therefore rule out every exact
per-row fill, without any linearity assumption on the fill or target.

For causal single-head softmax attention with a residual, such a pair exists
at row `1` of length-two prompts for every prescribed row state `z`, whenever
the linear value map `V` is nonzero. Compare `(z, z)` with `(a, z)`, choosing
`V a ≠ V z`. The output difference is the strictly positive weight of row
`0` in the second prompt times `V a - V z`. Query and key maps and the real
score scale are unrestricted. An injective subsequent per-row map preserves
the obstruction, including an MLP block when that whole block is injective.

This models one head, with no separate output projection (a linear output
projection can be absorbed into `V` if the resulting value map is nonzero).
It does not establish the claim for arbitrary multi-head compositions or
noninjective MLP blocks, which can erase context dependence. The prompts
range over all real row states; a restricted prompt family must contain a
witness pair to inherit the obstruction. The result concerns exactness on
every prompt, not average error, approximation quality, or a runtime lower
bound, and does not by itself prove the paper's full speed-versus-loss claim.
-/

section PerRowFill

variable {A B : Type*} {N : ℕ}

/-- Equal input rows with unequal target outputs rule out any exact per-row fill. -/
theorem no_per_row_fill_of_context_witness
    (T : (Fin N → A) → Fin N → B) (x x' : Fin N → A) (i : Fin N)
    (hrow : x i = x' i) (hout : T x i ≠ T x' i) :
    ¬ ∃ φ : A → B, ∀ x i, φ (x i) = T x i := by
  rintro ⟨φ, hφ⟩
  apply hout
  rw [← hφ x i, hrow, hφ x' i]

end PerRowFill

section CausalAttention

variable {d N : ℕ}

/-- Softmax over exactly the causal prefix `0, …, i`, using `attnScore`.
The prefix is indexed by `Fin (i.val + 1)` and embedded in the full prompt. -/
noncomputable def causalAttnWeight (c : ℝ)
    (Q K : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ)) (x : Fin N → Fin d → ℝ)
    (i : Fin N) (j : Fin (i.val + 1)) : ℝ :=
  softmaxW (attnScore c (fun l => Q (x l))
    (fun l => K (x (l.castLE (Nat.succ_le_of_lt i.isLt)))) i) j

/-- Causal single-head attention with residual: `x i + ∑ j ≤ i, w i j • V (x j)`.
Summing over the prefix also excludes future rows from the denominator. -/
noncomputable def causalAttnResidual (c : ℝ)
    (Q K V : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ)) (x : Fin N → Fin d → ℝ)
    (i : Fin N) : Fin d → ℝ :=
  x i + ∑ j : Fin (i.val + 1), causalAttnWeight c Q K x i j •
    V (x (j.castLE (Nat.succ_le_of_lt i.isLt)))

/-- Every weight within the causal prefix is strictly positive. -/
theorem causalAttnWeight_pos (c : ℝ)
    (Q K : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ)) (x : Fin N → Fin d → ℝ)
    (i : Fin N) (j : Fin (i.val + 1)) : 0 < causalAttnWeight c Q K x i j :=
  softmaxW_pos _ _

/-- The weights within the causal prefix sum to one. -/
theorem causalAttnWeight_sum (c : ℝ)
    (Q K : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ)) (x : Fin N → Fin d → ℝ)
    (i : Fin N) : ∑ j, causalAttnWeight c Q K x i j = 1 :=
  softmaxW_sum _ ⟨0, Nat.zero_lt_succ _⟩

/-- On a constant prompt, attention returns its constant value, plus the residual. -/
theorem causalAttnResidual_const (c : ℝ)
    (Q K V : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ)) (z : Fin d → ℝ) (i : Fin N) :
    causalAttnResidual c Q K V (fun _ => z) i = z + V z := by
  simp only [causalAttnResidual, ← Finset.sum_smul, causalAttnWeight_sum, one_smul]

/-- At the second row, the causal prefix consists of both rows. -/
theorem causalAttnResidual_two (c : ℝ)
    (Q K V : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ)) (a z : Fin d → ℝ) :
    causalAttnResidual c Q K V ![a, z] 1 = z +
      (causalAttnWeight c Q K ![a, z] 1 0 • V a +
        causalAttnWeight c Q K ![a, z] 1 1 • V z) := by
  simp [causalAttnResidual, Fin.sum_univ_two]

/-- Changing `(z, z)` to `(a, z)` changes row `1` by the new positive
weight of row `0` times `V a - V z`, even though the keys also change. -/
theorem causalAttnResidual_pair_sub_const (c : ℝ)
    (Q K V : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ)) (a z : Fin d → ℝ) :
    causalAttnResidual c Q K V ![a, z] 1 -
        causalAttnResidual c Q K V (fun _ : Fin 2 => z) 1 =
      causalAttnWeight c Q K ![a, z] 1 0 • (V a - V z) := by
  have hw : causalAttnWeight c Q K ![a, z] 1 0 +
      causalAttnWeight c Q K ![a, z] 1 1 = 1 := by
    have hsum := causalAttnWeight_sum c Q K ![a, z] 1
    change (∑ j : Fin 2, causalAttnWeight c Q K ![a, z] 1 j) = 1 at hsum
    simpa only [Fin.sum_univ_two] using hsum
  have hv : causalAttnWeight c Q K ![a, z] 1 0 • V z +
      causalAttnWeight c Q K ![a, z] 1 1 • V z = V z := by
    rw [← add_smul, hw, one_smul]
  rw [causalAttnResidual_two, causalAttnResidual_const, smul_sub]
  calc
    _ = causalAttnWeight c Q K ![a, z] 1 0 • V a +
        causalAttnWeight c Q K ![a, z] 1 1 • V z -
        (causalAttnWeight c Q K ![a, z] 1 0 • V z +
          causalAttnWeight c Q K ![a, z] 1 1 • V z) := by rw [hv]; abel
    _ = _ := by abel

/-- Every row state has two length-two contexts with that state at row `1`
and different residual attention outputs there, provided `V ≠ 0`. -/
theorem causal_attention_context_dependent (c : ℝ)
    (Q K V : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ)) (hV : V ≠ 0) (z : Fin d → ℝ) :
    ∃ x x' : Fin 2 → Fin d → ℝ, x 1 = z ∧ x' 1 = z ∧
      causalAttnResidual c Q K V x 1 ≠ causalAttnResidual c Q K V x' 1 := by
  have hu : ∃ u, V u ≠ 0 := by
    by_contra! h
    exact hV (LinearMap.ext h)
  obtain ⟨u, hu⟩ := hu
  have ha : V (z + u) ≠ V z := by
    simpa only [map_add, ne_eq, add_eq_left] using hu
  refine ⟨(fun _ => z), ![z + u, z], rfl, rfl, ?_⟩
  intro heq
  have hn : causalAttnWeight c Q K ![z + u, z] 1 0 • (V (z + u) - V z) ≠ 0 :=
    smul_ne_zero (causalAttnWeight_pos c Q K ![z + u, z] 1 0).ne' (sub_ne_zero.mpr ha)
  apply hn
  rw [← causalAttnResidual_pair_sub_const, heq, sub_self]

/-- No function of a row's own state matches this attention sublayer on every
length-two prompt, for any query/key maps and any real score scale, if `V ≠ 0`. -/
theorem no_per_row_fill_matches_attention (c : ℝ)
    (Q K V : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ)) (hV : V ≠ 0) :
    ¬ ∃ φ : (Fin d → ℝ) → (Fin d → ℝ),
      ∀ (x : Fin 2 → Fin d → ℝ) (i : Fin 2),
        φ (x i) = causalAttnResidual c Q K V x i := by
  obtain ⟨x, x', hx, hx', hout⟩ := causal_attention_context_dependent c Q K V hV 0
  exact no_per_row_fill_of_context_witness (causalAttnResidual c Q K V)
    x x' 1 (hx.trans hx'.symm) hout

/-- An injective subsequent per-row map, such as an injective MLP block,
preserves the impossibility of exact per-row fills. -/
theorem no_per_row_fill_matches_attention_of_injective {W : Type*} (c : ℝ)
    (Q K V : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ)) (hV : V ≠ 0)
    (h : (Fin d → ℝ) → W) (hh : Function.Injective h) :
    ¬ ∃ φ : (Fin d → ℝ) → W,
      ∀ (x : Fin 2 → Fin d → ℝ) (i : Fin 2),
        φ (x i) = h (causalAttnResidual c Q K V x i) := by
  obtain ⟨x, x', hx, hx', hout⟩ := causal_attention_context_dependent c Q K V hV 0
  exact no_per_row_fill_of_context_witness (fun x i => h (causalAttnResidual c Q K V x i))
    x x' 1 (hx.trans hx'.symm) (fun heq => hout (hh heq))

end CausalAttention

#print axioms no_per_row_fill_of_context_witness
#print axioms causalAttnResidual_pair_sub_const
#print axioms causal_attention_context_dependent
#print axioms no_per_row_fill_matches_attention
#print axioms no_per_row_fill_matches_attention_of_injective
