import KvaLean.VecNorm
import KvaLean.AttnSensitivity
import Mathlib.Tactic

-- Standalone research formalisation (not Mathlib itself):
-- Mathlib's file-header style does not apply.
set_option linter.style.header false

/-!
# QK-norm: key-size errors are invisible, only key-direction errors matter

Models with QK-norm (Qwen3.5, Qwen3-Next style) apply RMSNorm with a learned
per-dimension weight `w` to every key before attention:
`rmsNorm w x = w ⊙ x / rms x` with `rms x = √(mean_i (x i)²)` (here `eps = 0`).

1. Scale invariance (`rmsNorm_smul_pos`, `rmsNorm_smul_neg`):
   `rmsNorm w (c • x) = rmsNorm w x` for `c > 0` and `= -rmsNorm w x` for `c < 0`.
2. Attention (`logit_rmsNorm_smul_pos`, `qk_attnScore_rescale`,
   `qk_attnWeight_rescale`, `qk_attnOut_rescale`): if a projector error only
   rescales each filled key input (`x̂ j = c j • x j`, `c j > 0`, a different
   factor per key allowed), every logit, every softmax weight and every
   attention output is exactly unchanged. Any per-position map applied after
   the norm (e.g. RoPE) is allowed.
3. Direction only (`rmsNorm_eq_dir`, `rmsNorm_eq_iff_unitDir_eq`,
   `rmsNorm_dir_lipschitz`, `rmsNorm_dir_lipschitz_max`, `rmsNorm_dir_lower`):
   `rmsNorm w x = √d • (w ⊙ x/‖x‖)`, so
   `‖rmsNorm w x̂ - rmsNorm w x‖ ≤ (max_i |w i|) · √d · ‖x̂/‖x̂‖ - x/‖x‖‖`,
   with a matching lower bound `min_i |w i| · √d · ‖Δdir‖`. Linked to Lemma 5
   (`qk_attn_dir_bound`): the attention-output change from one filled key is
   bounded by its direction error alone. `unitDir_sub_le_rescaled`: the
   direction error is at most `2 ‖c • x̂ - x‖ / ‖x‖` for the best `c > 0`.
   Post-norm key size (`vnorm_rmsNorm_eq`, `vnorm_rmsNorm_uniform`,
   `abs_vnorm_rmsNorm_sub_le`, `abs_vnorm_rmsNorm_sub_le_spread`,
   `vnorm_rescale_sub`): `‖rmsNorm w x‖ = √d ‖w ⊙ x/‖x‖‖`, so the size gap
   between a filled and a true normalised key is a function of direction only,
   `≤ √d (S / wmin) θ` with `S` the spread of the squared gains (`0` for uniform
   gains). Rescaling a normalised key to another's size moves it by exactly
   that gap; the size ratio is within `(S / wmin²) θ` of `1`
   (`abs_vnorm_rmsNorm_ratio_sub_one_le`). L2 normalisation is the case
   `w ≡ 1/√d` (`rmsNorm_const_eq_unitDir`).
4. `eps > 0` (`rmsNormEps_eq_smul`, `rmsNormEps_sub_le`,
   `rmsNormEps_smul_pos_le`, `rmsNormEps_smul_pos_le_ratio`): RMSNorm with
   `eps` is the exact one times a factor `γ ∈ [0, 1]` with
   `1 - γ ≤ eps / (mean x² + eps)`. Hence
   `‖Δ‖ ≤ W √d (‖Δdir‖ + max (eps/(mean x̂² + eps)) (eps/(mean x² + eps)))`,
   and a pure rescale costs at most `W √d · max (eps/mean x̂²) (eps/mean x²)`.

Conventions: vectors are `Fin d → ℝ`, `‖·‖ = vnorm` from `VecNorm.lean`,
`x/‖x‖ = unitDir x`. Lean's `a / 0 = 0` makes `rmsNorm w 0 = 0` and
`unitDir 0 = 0`; theorems that need `x ≠ 0` say so, the others hold for all `x`.
-/

open Finset

variable {m n d : ℕ}

/-! ## Definitions -/

/-- Mean square `meanSq x = (∑ i, (x i)²) / d`. -/
noncomputable def meanSq (x : Fin d → ℝ) : ℝ :=
  (∑ i, (x i) ^ 2) / d

/-- Root mean square with `eps = 0`: `rms x = √(meanSq x)`. -/
noncomputable def rms (x : Fin d → ℝ) : ℝ :=
  Real.sqrt (meanSq x)

/-- RMSNorm with learned per-dimension weight `w` and `eps = 0`:
`rmsNorm w x i = w i * x i / rms x`. -/
noncomputable def rmsNorm (w x : Fin d → ℝ) : Fin d → ℝ :=
  fun i => w i * x i / rms x

/-- Unit direction `x / ‖x‖` (`0` at `x = 0`, by `0⁻¹ = 0`). -/
noncomputable def unitDir (x : Fin d → ℝ) : Fin d → ℝ :=
  (vnorm x)⁻¹ • x

/-- RMS with `eps`: `rmsEps ε x = √(meanSq x + ε)`. -/
noncomputable def rmsEps (ε : ℝ) (x : Fin d → ℝ) : ℝ :=
  Real.sqrt (meanSq x + ε)

/-- RMSNorm with `eps`: `rmsNormEps ε w x i = w i * x i / rmsEps ε x`. -/
noncomputable def rmsNormEps (ε : ℝ) (w x : Fin d → ℝ) : Fin d → ℝ :=
  fun i => w i * x i / rmsEps ε x

/-- QK-normed keys: `K j = post j (rmsNorm w (k j))`, where `post j` is any map
applied after the norm (e.g. RoPE at position `j`; `fun _ => id` for none). -/
noncomputable def qkKeys (w : Fin d → ℝ) (post : Fin n → (Fin d → ℝ) → Fin d → ℝ)
    (k : Fin n → Fin d → ℝ) : Fin n → Fin d → ℝ :=
  fun j => post j (rmsNorm w (k j))

/-! ## Basic facts -/

theorem meanSq_nonneg (x : Fin d → ℝ) : 0 ≤ meanSq x :=
  div_nonneg (Finset.sum_nonneg fun _ _ => sq_nonneg _) (Nat.cast_nonneg d)

theorem meanSq_smul (c : ℝ) (x : Fin d → ℝ) : meanSq (c • x) = c ^ 2 * meanSq x := by
  have h : ∑ i, ((c • x) i) ^ 2 = c ^ 2 * ∑ i, (x i) ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    simp only [Pi.smul_apply, smul_eq_mul]
    ring
  unfold meanSq
  rw [h, mul_div_assoc]

theorem rms_smul (c : ℝ) (x : Fin d → ℝ) : rms (c • x) = |c| * rms x := by
  unfold rms
  rw [meanSq_smul, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq_eq_abs]

/-- `rms x = ‖x‖ / √d`. -/
theorem rms_eq (x : Fin d → ℝ) : rms x = vnorm x / Real.sqrt d := by
  unfold rms meanSq vnorm
  exact Real.sqrt_div' _ (Nat.cast_nonneg d)

theorem vnorm_pos_of_ne_zero {x : Fin d → ℝ} (hx : x ≠ 0) : 0 < vnorm x := by
  rcases (vnorm_nonneg x).lt_or_eq with h | h
  · exact h
  · exfalso
    apply hx
    have hs : ∑ k, (x k) ^ 2 = 0 := by rw [← vnorm_sq, ← h]; ring
    rw [Finset.sum_eq_zero_iff_of_nonneg (fun _ _ => sq_nonneg _)] at hs
    funext k
    simpa using hs k (Finset.mem_univ k)

theorem rms_pos {x : Fin d → ℝ} (hx : x ≠ 0) : 0 < rms x := by
  obtain ⟨i, _⟩ := Function.ne_iff.mp hx
  rw [rms_eq]
  exact div_pos (vnorm_pos_of_ne_zero hx) (Real.sqrt_pos.mpr (Nat.cast_pos.mpr i.pos))

theorem meanSq_pos {x : Fin d → ℝ} (hx : x ≠ 0) : 0 < meanSq x :=
  Real.sqrt_pos.mp (rms_pos hx)

theorem vnorm_unitDir {x : Fin d → ℝ} (hx : x ≠ 0) : vnorm (unitDir x) = 1 := by
  have h := vnorm_pos_of_ne_zero hx
  rw [unitDir, vnorm_smul, abs_inv, abs_of_pos h, inv_mul_cancel₀ h.ne']

theorem unitDir_smul_pos (x : Fin d → ℝ) {c : ℝ} (hc : 0 < c) :
    unitDir (c • x) = unitDir x := by
  unfold unitDir
  rw [vnorm_smul, abs_of_pos hc, smul_smul, mul_inv, mul_comm c⁻¹, mul_assoc,
    inv_mul_cancel₀ hc.ne', mul_one]

/-- Weighted norm, upper: `‖w ⊙ v‖ ≤ W ‖v‖` if `|w i| ≤ W` for all `i`. -/
theorem vnorm_mul_le (w v : Fin d → ℝ) (W : ℝ) (hW : ∀ i, |w i| ≤ W) :
    vnorm (fun i => w i * v i) ≤ W * vnorm v := by
  rcases isEmpty_or_nonempty (Fin d) with hd | ⟨⟨i0⟩⟩
  · simp [vnorm]
  · have hW0 : 0 ≤ W := le_trans (abs_nonneg _) (hW i0)
    unfold vnorm
    rw [← Real.sqrt_sq hW0, ← Real.sqrt_mul (sq_nonneg W)]
    apply Real.sqrt_le_sqrt
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    have h1 : (w i) ^ 2 ≤ W ^ 2 := sq_le_sq' (abs_le.mp (hW i)).1 (abs_le.mp (hW i)).2
    calc (w i * v i) ^ 2 = (w i) ^ 2 * (v i) ^ 2 := by ring
      _ ≤ W ^ 2 * (v i) ^ 2 := mul_le_mul_of_nonneg_right h1 (sq_nonneg _)

/-- Weighted norm, lower: `wmin ‖v‖ ≤ ‖w ⊙ v‖` if `0 ≤ wmin ≤ |w i|` for all `i`. -/
theorem le_vnorm_mul (w v : Fin d → ℝ) (wmin : ℝ) (h0 : 0 ≤ wmin)
    (hw : ∀ i, wmin ≤ |w i|) :
    wmin * vnorm v ≤ vnorm (fun i => w i * v i) := by
  unfold vnorm
  rw [← Real.sqrt_sq h0, ← Real.sqrt_mul (sq_nonneg wmin)]
  apply Real.sqrt_le_sqrt
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  have h1 : wmin ^ 2 ≤ (w i) ^ 2 := by
    rw [← sq_abs (w i), sq, sq]
    exact mul_self_le_mul_self h0 (hw i)
  calc wmin ^ 2 * (v i) ^ 2 ≤ (w i) ^ 2 * (v i) ^ 2 :=
        mul_le_mul_of_nonneg_right h1 (sq_nonneg _)
    _ = (w i * v i) ^ 2 := by ring

/-! ## 1. Scale invariance (`eps = 0`) -/

/-- (1) `rmsNorm w (c • x) = rmsNorm w x` for every `c > 0`. -/
theorem rmsNorm_smul_pos (w x : Fin d → ℝ) {c : ℝ} (hc : 0 < c) :
    rmsNorm w (c • x) = rmsNorm w x := by
  funext i
  simp only [rmsNorm, rms_smul, abs_of_pos hc, Pi.smul_apply, smul_eq_mul]
  rw [mul_left_comm, mul_div_mul_left _ _ hc.ne']

/-- (1) `rmsNorm w (c • x) = -rmsNorm w x` for every `c < 0`. -/
theorem rmsNorm_smul_neg (w x : Fin d → ℝ) {c : ℝ} (hc : c < 0) :
    rmsNorm w (c • x) = -rmsNorm w x := by
  funext i
  simp only [rmsNorm, rms_smul, abs_of_neg hc, Pi.smul_apply, smul_eq_mul, Pi.neg_apply]
  rw [show w i * (c * x i) = -c * -(w i * x i) by ring,
    mul_div_mul_left _ _ (neg_ne_zero.mpr hc.ne), neg_div]

/-! ## 2. Attention: rescaled key inputs change nothing -/

/-- (2) One logit: `⟪q, rmsNorm w (c • x)⟫ = ⟪q, rmsNorm w x⟫` for `c > 0`. -/
theorem logit_rmsNorm_smul_pos (q w x : Fin d → ℝ) {c : ℝ} (hc : 0 < c) :
    q ⬝ᵥ rmsNorm w (c • x) = q ⬝ᵥ rmsNorm w x := by
  rw [rmsNorm_smul_pos w x hc]

/-- Rescaling each key input by its own factor `c j > 0` leaves the QK-normed keys
unchanged (with any post-norm map such as RoPE). -/
theorem qkKeys_rescale (w : Fin d → ℝ) (post : Fin n → (Fin d → ℝ) → Fin d → ℝ)
    (k : Fin n → Fin d → ℝ) (c : Fin n → ℝ) (hc : ∀ j, 0 < c j) :
    qkKeys w post (fun j => c j • k j) = qkKeys w post k := by
  funext j
  simp only [qkKeys]
  rw [rmsNorm_smul_pos w (k j) (hc j)]

/-- (2) Every logit `s i j = t * ⟪q i, K j⟫` is unchanged. -/
theorem qk_attnScore_rescale (t : ℝ) (q : Fin m → Fin d → ℝ) (w : Fin d → ℝ)
    (post : Fin n → (Fin d → ℝ) → Fin d → ℝ) (k : Fin n → Fin d → ℝ) (c : Fin n → ℝ)
    (hc : ∀ j, 0 < c j) :
    attnScore t q (qkKeys w post (fun j => c j • k j)) = attnScore t q (qkKeys w post k) := by
  rw [qkKeys_rescale w post k c hc]

/-- (2) Every softmax weight is unchanged. -/
theorem qk_attnWeight_rescale (t : ℝ) (q : Fin m → Fin d → ℝ) (w : Fin d → ℝ)
    (post : Fin n → (Fin d → ℝ) → Fin d → ℝ) (k : Fin n → Fin d → ℝ) (c : Fin n → ℝ)
    (hc : ∀ j, 0 < c j) :
    attnWeight t q (qkKeys w post (fun j => c j • k j)) = attnWeight t q (qkKeys w post k) := by
  rw [qkKeys_rescale w post k c hc]

/-- (2) Every attention output is unchanged (values `v` untouched). -/
theorem qk_attnOut_rescale (t : ℝ) (q : Fin m → Fin d → ℝ) (w : Fin d → ℝ)
    (post : Fin n → (Fin d → ℝ) → Fin d → ℝ) (k v : Fin n → Fin d → ℝ) (c : Fin n → ℝ)
    (hc : ∀ j, 0 < c j) :
    attnOut t q (qkKeys w post (fun j => c j • k j)) v = attnOut t q (qkKeys w post k) v := by
  rw [qkKeys_rescale w post k c hc]

/-! ## 3. Only the direction matters -/

/-- (3) `rmsNorm w x = √d • (w ⊙ x/‖x‖)`: it sees `x` only through `x/‖x‖`. -/
theorem rmsNorm_eq_dir (w x : Fin d → ℝ) :
    rmsNorm w x = Real.sqrt d • fun i => w i * unitDir x i := by
  funext i
  simp only [rmsNorm, rms_eq, unitDir, Pi.smul_apply, smul_eq_mul]
  rw [div_div_eq_mul_div]
  ring

/-- (3) Equal directions give equal normalised keys. -/
theorem rmsNorm_eq_of_unitDir_eq (w : Fin d → ℝ) {x xh : Fin d → ℝ}
    (h : unitDir xh = unitDir x) : rmsNorm w xh = rmsNorm w x := by
  rw [rmsNorm_eq_dir, rmsNorm_eq_dir, h]

/-- (3) If no weight is zero, the normalised key determines the direction:
`rmsNorm w x̂ = rmsNorm w x ↔ x̂/‖x̂‖ = x/‖x‖`. -/
theorem rmsNorm_eq_iff_unitDir_eq (w : Fin d → ℝ) (hw : ∀ i, w i ≠ 0) (x xh : Fin d → ℝ) :
    rmsNorm w xh = rmsNorm w x ↔ unitDir xh = unitDir x := by
  refine ⟨fun h => ?_, rmsNorm_eq_of_unitDir_eq w⟩
  funext i
  have hi := congrFun h i
  rw [rmsNorm_eq_dir, rmsNorm_eq_dir] at hi
  simp only [Pi.smul_apply, smul_eq_mul] at hi
  have hd : Real.sqrt d ≠ 0 := (Real.sqrt_pos.mpr (Nat.cast_pos.mpr i.pos)).ne'
  exact mul_left_cancel₀ (hw i) (mul_left_cancel₀ hd hi)

/-- (3) Exact error: `rmsNorm w x̂ - rmsNorm w x = √d • (w ⊙ (x̂/‖x̂‖ - x/‖x‖))`. -/
theorem rmsNorm_sub_eq (w xh x : Fin d → ℝ) :
    rmsNorm w xh - rmsNorm w x
      = Real.sqrt d • fun i => w i * (unitDir xh - unitDir x) i := by
  rw [rmsNorm_eq_dir, rmsNorm_eq_dir, ← smul_sub]
  congr 1
  funext i
  simp only [Pi.sub_apply]
  ring

/-- (3) Lipschitz bound in the direction error:
`‖rmsNorm w x̂ - rmsNorm w x‖ ≤ W · √d · ‖x̂/‖x̂‖ - x/‖x‖‖` whenever `|w i| ≤ W`. -/
theorem rmsNorm_dir_lipschitz (w : Fin d → ℝ) (W : ℝ) (hW : ∀ i, |w i| ≤ W)
    (xh x : Fin d → ℝ) :
    vnorm (rmsNorm w xh - rmsNorm w x) ≤ W * Real.sqrt d * vnorm (unitDir xh - unitDir x) := by
  rw [rmsNorm_sub_eq, vnorm_smul, abs_of_nonneg (Real.sqrt_nonneg _)]
  have h := vnorm_mul_le w (unitDir xh - unitDir x) W hW
  calc Real.sqrt d * vnorm (fun i => w i * (unitDir xh - unitDir x) i)
      ≤ Real.sqrt d * (W * vnorm (unitDir xh - unitDir x)) :=
        mul_le_mul_of_nonneg_left h (Real.sqrt_nonneg _)
    _ = _ := by ring

/-- (3) The same bound with `W = max_i |w i|` (needs `d > 0` for the max). -/
theorem rmsNorm_dir_lipschitz_max (hd : 0 < d) (w xh x : Fin d → ℝ) :
    vnorm (rmsNorm w xh - rmsNorm w x)
      ≤ (Finset.univ.sup' ⟨⟨0, hd⟩, Finset.mem_univ _⟩ fun i => |w i|) * Real.sqrt d
          * vnorm (unitDir xh - unitDir x) :=
  rmsNorm_dir_lipschitz w _ (fun i => Finset.le_sup' (fun i => |w i|) (Finset.mem_univ i)) xh x

/-- (3) Matching lower bound: if `0 ≤ wmin ≤ |w i|` for all `i`,
`wmin · √d · ‖Δdir‖ ≤ ‖rmsNorm w x̂ - rmsNorm w x‖`; direction errors are not damped. -/
theorem rmsNorm_dir_lower (w : Fin d → ℝ) (wmin : ℝ) (h0 : 0 ≤ wmin)
    (hw : ∀ i, wmin ≤ |w i|) (xh x : Fin d → ℝ) :
    wmin * Real.sqrt d * vnorm (unitDir xh - unitDir x) ≤ vnorm (rmsNorm w xh - rmsNorm w x) := by
  rw [rmsNorm_sub_eq, vnorm_smul, abs_of_nonneg (Real.sqrt_nonneg _)]
  have h := le_vnorm_mul w (unitDir xh - unitDir x) wmin h0 hw
  calc wmin * Real.sqrt d * vnorm (unitDir xh - unitDir x)
      = Real.sqrt d * (wmin * vnorm (unitDir xh - unitDir x)) := by ring
    _ ≤ _ := mul_le_mul_of_nonneg_left h (Real.sqrt_nonneg _)

/-- Size of a normalised key: `‖rmsNorm w x‖ ≤ W · √d` (equality `√d` for `w = 1`). -/
theorem vnorm_rmsNorm_le (w : Fin d → ℝ) (W : ℝ) (hW : ∀ i, |w i| ≤ W) {x : Fin d → ℝ}
    (hx : x ≠ 0) : vnorm (rmsNorm w x) ≤ W * Real.sqrt d := by
  rw [rmsNorm_eq_dir, vnorm_smul, abs_of_nonneg (Real.sqrt_nonneg _)]
  have h := vnorm_mul_le w (unitDir x) W hW
  rw [vnorm_unitDir hx, mul_one] at h
  calc Real.sqrt d * vnorm (fun i => w i * unitDir x i) ≤ Real.sqrt d * W :=
        mul_le_mul_of_nonneg_left h (Real.sqrt_nonneg _)
    _ = _ := by ring

/-- Reverse triangle inequality `|‖u‖ - ‖v‖| ≤ ‖u - v‖`. -/
theorem abs_vnorm_sub_vnorm_le (u v : Fin d → ℝ) : |vnorm u - vnorm v| ≤ vnorm (u - v) := by
  have h1 := vnorm_add_le (u - v) v
  rw [sub_add_cancel] at h1
  have h2 := vnorm_sub_le u (u - v)
  rw [sub_sub_cancel] at h2
  exact abs_le.mpr ⟨by linarith, by linarith⟩

/-! ### Post-norm key size (what the knorm / kdir arms change) -/

/-- The post-norm key size depends only on the direction:
`‖rmsNorm w x‖ = √d · ‖w ⊙ x/‖x‖‖`. -/
theorem vnorm_rmsNorm_eq (w x : Fin d → ℝ) :
    vnorm (rmsNorm w x) = Real.sqrt d * vnorm (fun i => w i * unitDir x i) := by
  rw [rmsNorm_eq_dir, vnorm_smul, abs_of_nonneg (Real.sqrt_nonneg _)]

/-- Uniform gains `|w i| = W`: every nonzero key has post-norm size exactly `W √d`. -/
theorem vnorm_rmsNorm_uniform (w : Fin d → ℝ) (W : ℝ) (hw : ∀ i, |w i| = W)
    {x : Fin d → ℝ} (hx : x ≠ 0) : vnorm (rmsNorm w x) = W * Real.sqrt d := by
  obtain ⟨i, _⟩ := Function.ne_iff.mp hx
  have hW0 : 0 ≤ W := hw i ▸ abs_nonneg _
  apply le_antisymm (vnorm_rmsNorm_le w W (fun i => (hw i).le) hx)
  have h := le_vnorm_mul w (unitDir x) W hW0 (fun i => (hw i).ge)
  rw [vnorm_unitDir hx, mul_one] at h
  rw [vnorm_rmsNorm_eq]
  calc W * Real.sqrt d = Real.sqrt d * W := by ring
    _ ≤ _ := mul_le_mul_of_nonneg_left h (Real.sqrt_nonneg _)

/-- Post-norm size gap, crude: `|‖K̂‖ - ‖K‖| ≤ W √d θ` (`θ` the direction error). -/
theorem abs_vnorm_rmsNorm_sub_le (w : Fin d → ℝ) (W : ℝ) (hW : ∀ i, |w i| ≤ W)
    (xh x : Fin d → ℝ) :
    |vnorm (rmsNorm w xh) - vnorm (rmsNorm w x)|
      ≤ W * Real.sqrt d * vnorm (unitDir xh - unitDir x) :=
  le_trans (abs_vnorm_sub_vnorm_le _ _) (rmsNorm_dir_lipschitz w W hW xh x)

/-- Post-norm size gap, sharp: if `0 < wmin ≤ |w i|` and `|w i² - c2| ≤ S` for all `i`
(`S` = spread of the squared gains around any centre `c2`), then
`|‖K̂‖ - ‖K‖| ≤ √d · (S / wmin) · θ`. It is `0` for uniform gains, whatever `θ`. -/
theorem abs_vnorm_rmsNorm_sub_le_spread (w : Fin d → ℝ) (wmin c2 S : ℝ) (h0 : 0 < wmin)
    (hw : ∀ i, wmin ≤ |w i|) (hS : ∀ i, |w i ^ 2 - c2| ≤ S) {x xh : Fin d → ℝ}
    (hx : x ≠ 0) (hxh : xh ≠ 0) :
    |vnorm (rmsNorm w xh) - vnorm (rmsNorm w x)|
      ≤ Real.sqrt d * (S / wmin) * vnorm (unitDir xh - unitDir x) := by
  obtain ⟨i0, _⟩ := Function.ne_iff.mp hx
  have hS0 : 0 ≤ S := le_trans (abs_nonneg _) (hS i0)
  set u := unitDir x
  set uh := unitDir xh
  have hu1 : vnorm u = 1 := vnorm_unitDir hx
  have huh1 : vnorm uh = 1 := vnorm_unitDir hxh
  have hu : ∑ i, (u i) ^ 2 = 1 := by rw [← vnorm_sq, hu1]; norm_num
  have huh : ∑ i, (uh i) ^ 2 = 1 := by rw [← vnorm_sq, huh1]; norm_num
  set A := vnorm (fun i => w i * uh i)
  set B := vnorm (fun i => w i * u i)
  have hA : wmin ≤ A := by
    have := le_vnorm_mul w uh wmin h0.le hw
    rwa [huh1, mul_one] at this
  have hB : wmin ≤ B := by
    have := le_vnorm_mul w u wmin h0.le hw
    rwa [hu1, mul_one] at this
  have hdiff : A ^ 2 - B ^ 2 = ∑ i, (w i ^ 2 - c2) * ((uh i - u i) * (uh i + u i)) := by
    have h : ∑ i, (w i ^ 2 - c2) * ((uh i - u i) * (uh i + u i))
        = ∑ i, (w i * uh i) ^ 2 - ∑ i, (w i * u i) ^ 2
          - c2 * (∑ i, (uh i) ^ 2 - ∑ i, (u i) ^ 2) := by
      rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, Finset.mul_sum,
        ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro i _
      ring
    rw [h, hu, huh, vnorm_sq, vnorm_sq]
    ring
  have hplus : vnorm (uh + u) ≤ 2 := by
    have := vnorm_add_le uh u
    rw [hu1, huh1] at this
    linarith
  have hbound : |A ^ 2 - B ^ 2| ≤ S * (vnorm (uh - u) * 2) := by
    rw [hdiff]
    calc |∑ i, (w i ^ 2 - c2) * ((uh i - u i) * (uh i + u i))|
        ≤ ∑ i, |(w i ^ 2 - c2) * ((uh i - u i) * (uh i + u i))| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i, S * (|uh i - u i| * |uh i + u i|) := by
          apply Finset.sum_le_sum
          intro i _
          rw [abs_mul, abs_mul]
          exact mul_le_mul_of_nonneg_right (hS i) (by positivity)
      _ = S * ((fun i => |uh i - u i|) ⬝ᵥ (fun i => |uh i + u i|)) := by
          rw [dotProduct, Finset.mul_sum]
      _ ≤ S * (vnorm (fun i => |uh i - u i|) * vnorm (fun i => |uh i + u i|)) :=
          mul_le_mul_of_nonneg_left (le_trans (le_abs_self _) (abs_dotProduct_le _ _)) hS0
      _ = S * (vnorm (uh - u) * vnorm (uh + u)) := by
          simp only [vnorm, sq_abs, Pi.sub_apply, Pi.add_apply]
      _ ≤ S * (vnorm (uh - u) * 2) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hplus (vnorm_nonneg _)) hS0
  have hAB : |A - B| * (A + B) = |A ^ 2 - B ^ 2| := by
    rw [sq_sub_sq, abs_mul, abs_of_pos (by linarith : 0 < A + B), mul_comm]
  have hgap : |A - B| ≤ S / wmin * vnorm (uh - u) := by
    rw [div_mul_eq_mul_div, le_div_iff₀ h0]
    have h1 : |A - B| * (2 * wmin) ≤ |A - B| * (A + B) :=
      mul_le_mul_of_nonneg_left (by linarith) (abs_nonneg _)
    nlinarith
  rw [vnorm_rmsNorm_eq, vnorm_rmsNorm_eq, ← mul_sub, abs_mul,
    abs_of_nonneg (Real.sqrt_nonneg _), mul_assoc]
  exact mul_le_mul_of_nonneg_left hgap (Real.sqrt_nonneg _)

/-- Rescaling a nonzero vector `K` to size `t` moves it by exactly `|t - ‖K‖|`.
With `K = K̂` (filled) and `t = ‖K‖` (true) this is the knorm arm's move; with
`K` the true key and `t = ‖K̂‖` it is the kdir arm's remaining error. Both equal the
post-norm size gap, bounded above by the direction error. -/
theorem vnorm_rescale_sub (K : Fin d → ℝ) (hK : K ≠ 0) (t : ℝ) :
    vnorm ((t / vnorm K) • K - K) = |t - vnorm K| := by
  have hA := vnorm_pos_of_ne_zero hK
  rw [show (t / vnorm K) • K - K = (t / vnorm K - 1) • K by rw [sub_smul, one_smul],
    vnorm_smul, show t / vnorm K - 1 = (t - vnorm K) / vnorm K by field_simp, abs_div,
    abs_of_pos hA, div_mul_cancel₀ _ hA.ne']

/-- Size ratio of normalised keys (the knorm / ktail scale factor): with `0 < wmin ≤ |w i|`
and `|w i² - c2| ≤ S`, `|‖K‖ / ‖K̂‖ - 1| ≤ (S / wmin²) · θ`, `θ` the direction error. -/
theorem abs_vnorm_rmsNorm_ratio_sub_one_le (w : Fin d → ℝ) (wmin c2 S : ℝ) (h0 : 0 < wmin)
    (hw : ∀ i, wmin ≤ |w i|) (hS : ∀ i, |w i ^ 2 - c2| ≤ S) {x xh : Fin d → ℝ}
    (hx : x ≠ 0) (hxh : xh ≠ 0) :
    |vnorm (rmsNorm w x) / vnorm (rmsNorm w xh) - 1|
      ≤ S / wmin ^ 2 * vnorm (unitDir xh - unitDir x) := by
  obtain ⟨i0, _⟩ := Function.ne_iff.mp hxh
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr (Nat.cast_pos.mpr i0.pos)
  have hlow : Real.sqrt d * wmin ≤ vnorm (rmsNorm w xh) := by
    rw [vnorm_rmsNorm_eq]
    have := le_vnorm_mul w (unitDir xh) wmin h0.le hw
    rw [vnorm_unitDir hxh, mul_one] at this
    exact mul_le_mul_of_nonneg_left this hsd.le
  have hA : 0 < vnorm (rmsNorm w xh) := lt_of_lt_of_le (mul_pos hsd h0) hlow
  have hgap := abs_vnorm_rmsNorm_sub_le_spread w wmin c2 S h0 hw hS hx hxh
  rw [abs_sub_comm] at hgap
  rw [div_sub_one hA.ne', abs_div, abs_of_pos hA, div_le_iff₀ hA]
  have hθ := vnorm_nonneg (unitDir xh - unitDir x)
  have hS0 : 0 ≤ S := le_trans (abs_nonneg _) (hS i0)
  calc |vnorm (rmsNorm w x) - vnorm (rmsNorm w xh)|
      ≤ Real.sqrt d * (S / wmin) * vnorm (unitDir xh - unitDir x) := hgap
    _ = S / wmin ^ 2 * vnorm (unitDir xh - unitDir x) * (Real.sqrt d * wmin) := by
        field_simp
    _ ≤ S / wmin ^ 2 * vnorm (unitDir xh - unitDir x) * vnorm (rmsNorm w xh) :=
        mul_le_mul_of_nonneg_left hlow (by positivity)

/-- L2 normalisation is the special case `w ≡ 1/√d`: `rmsNorm (1/√d) x = x/‖x‖`.
So everything above (with `W = wmin = 1/√d`, `S = 0`) also covers L2-normalised
keys, as in Gated DeltaNet layers. -/
theorem rmsNorm_const_eq_unitDir (x : Fin d → ℝ) :
    rmsNorm (fun _ => 1 / Real.sqrt d) x = unitDir x := by
  funext i
  have hsd : Real.sqrt d ≠ 0 := (Real.sqrt_pos.mpr (Nat.cast_pos.mpr i.pos)).ne'
  rw [rmsNorm_eq_dir]
  simp only [Pi.smul_apply, smul_eq_mul]
  field_simp

/-! ### Direction error vs plain error -/

/-- Direction error vs relative error: `‖x̂/‖x̂‖ - x/‖x‖‖ ≤ 2 ‖x̂ - x‖ / ‖x‖`. -/
theorem unitDir_sub_le {x xh : Fin d → ℝ} (hx : x ≠ 0) (hxh : xh ≠ 0) :
    vnorm (unitDir xh - unitDir x) ≤ 2 * vnorm (xh - x) / vnorm x := by
  have ha := vnorm_pos_of_ne_zero hxh
  have hb := vnorm_pos_of_ne_zero hx
  set a := vnorm xh
  set b := vnorm x
  have hsplit : unitDir xh - unitDir x = b⁻¹ • (xh - x) + (a⁻¹ - b⁻¹) • xh := by
    unfold unitDir
    funext i
    simp only [Pi.add_apply, Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
    ring
  have hrev : |a - b| ≤ vnorm (xh - x) := abs_vnorm_sub_vnorm_le xh x
  have hterm : |a⁻¹ - b⁻¹| * a = |b - a| / b := by
    rw [inv_sub_inv ha.ne' hb.ne', abs_div, abs_mul, abs_of_pos ha, abs_of_pos hb]
    field_simp
  rw [hsplit]
  calc vnorm (b⁻¹ • (xh - x) + (a⁻¹ - b⁻¹) • xh)
      ≤ vnorm (b⁻¹ • (xh - x)) + vnorm ((a⁻¹ - b⁻¹) • xh) := vnorm_add_le _ _
    _ = vnorm (xh - x) / b + |b - a| / b := by
        rw [vnorm_smul, vnorm_smul, abs_inv, abs_of_pos hb, hterm, inv_mul_eq_div]
    _ ≤ vnorm (xh - x) / b + vnorm (xh - x) / b := by
        rw [abs_sub_comm] at hrev
        gcongr
    _ = 2 * vnorm (xh - x) / b := by ring

/-- Direction error vs the error left after the best rescaling: for every `c > 0`,
`‖x̂/‖x̂‖ - x/‖x‖‖ ≤ 2 ‖c • x̂ - x‖ / ‖x‖`. The size part of the error drops out. -/
theorem unitDir_sub_le_rescaled {x xh : Fin d → ℝ} (hx : x ≠ 0) (hxh : xh ≠ 0)
    {c : ℝ} (hc : 0 < c) :
    vnorm (unitDir xh - unitDir x) ≤ 2 * vnorm (c • xh - x) / vnorm x := by
  have hcx : c • xh ≠ 0 := smul_ne_zero hc.ne' hxh
  have h := unitDir_sub_le hx hcx
  rwa [unitDir_smul_pos xh hc] at h

/-- (3) + Lemma 5: replacing one key input `k j0` by a filled `x̂` changes each attention
output by at most `2 M a i j0 (exp (|t| ‖q i‖ W √d θ) - 1)`, `θ = ‖x̂/‖x̂‖ - k j0/‖k j0‖‖`
the direction error alone. `post j0` (e.g. RoPE) must be 1-Lipschitz, `‖v j‖ ≤ M`. -/
theorem qk_attn_dir_bound (t : ℝ) (q : Fin m → Fin d → ℝ) (w : Fin d → ℝ) (W : ℝ)
    (hW : ∀ i, |w i| ≤ W) (post : Fin n → (Fin d → ℝ) → Fin d → ℝ) (j0 : Fin n)
    (hpost : ∀ u u' : Fin d → ℝ, vnorm (post j0 u - post j0 u') ≤ vnorm (u - u'))
    (k v : Fin n → Fin d → ℝ) (xh : Fin d → ℝ) (i : Fin m) (M : ℝ)
    (hM : ∀ j, vnorm (v j) ≤ M) :
    vnorm (attnOut t q (qkKeys w post (Function.update k j0 xh)) v i
        - attnOut t q (qkKeys w post k) v i)
      ≤ 2 * M * attnWeight t q (qkKeys w post k) i j0
          * (Real.exp (|t| * vnorm (q i) * (W * Real.sqrt d
              * vnorm (unitDir xh - unitDir (k j0)))) - 1) := by
  set K := qkKeys w post k
  set δ := post j0 (rmsNorm w xh) - post j0 (rmsNorm w (k j0))
  have hK : qkKeys w post (Function.update k j0 xh) = Function.update K j0 (K j0 + δ) := by
    funext j
    by_cases hj : j = j0
    · subst hj
      simp [K, δ, qkKeys]
    · simp [K, qkKeys, Function.update_of_ne hj]
  rw [hK]
  refine le_trans (attn_key_bound t q K v j0 δ i M hM) ?_
  have hM0 : 0 ≤ M := le_trans (vnorm_nonneg _) (hM j0)
  have ha := (attnWeight_pos t q K i j0).le
  have hδ : vnorm δ ≤ W * Real.sqrt d * vnorm (unitDir xh - unitDir (k j0)) :=
    le_trans (hpost _ _) (rmsNorm_dir_lipschitz w W hW xh (k j0))
  have hε : |t * (q i ⬝ᵥ δ)| ≤ |t| * vnorm (q i) * (W * Real.sqrt d
      * vnorm (unitDir xh - unitDir (k j0))) := by
    rw [abs_mul, mul_assoc]
    apply mul_le_mul_of_nonneg_left _ (abs_nonneg t)
    exact le_trans (abs_dotProduct_le _ _)
      (mul_le_mul_of_nonneg_left hδ (vnorm_nonneg _))
  have hexp := Real.exp_le_exp.mpr hε
  have h2 : 0 ≤ 2 * M * attnWeight t q K i j0 := by positivity
  exact mul_le_mul_of_nonneg_left (by linarith) h2

/-! ## 4. `eps > 0`: approximate invariance -/

theorem rmsEps_pos (ε : ℝ) (hε : 0 ≤ ε) {x : Fin d → ℝ} (hx : x ≠ 0) : 0 < rmsEps ε x :=
  Real.sqrt_pos.mpr (by linarith [meanSq_pos hx])

/-- (4) With `eps`, RMSNorm is the exact one times `γ = rms x / rmsEps ε x`. -/
theorem rmsNormEps_eq_smul (ε : ℝ) (hε : 0 ≤ ε) (w : Fin d → ℝ) {x : Fin d → ℝ}
    (hx : x ≠ 0) : rmsNormEps ε w x = (rms x / rmsEps ε x) • rmsNorm w x := by
  have hr := rms_pos hx
  have hR := rmsEps_pos ε hε hx
  funext i
  simp only [rmsNormEps, rmsNorm, Pi.smul_apply, smul_eq_mul]
  field_simp

/-- (4) The factor `γ` lies in `[0, 1]` ... -/
theorem epsFactor_mem (ε : ℝ) (hε : 0 ≤ ε) {x : Fin d → ℝ} (hx : x ≠ 0) :
    0 ≤ rms x / rmsEps ε x ∧ rms x / rmsEps ε x ≤ 1 := by
  have hR := rmsEps_pos ε hε hx
  refine ⟨div_nonneg (Real.sqrt_nonneg _) hR.le, ?_⟩
  rw [div_le_one hR]
  exact Real.sqrt_le_sqrt (by linarith)

/-- (4) ... and `1 - γ ≤ eps / (mean x² + eps)`. -/
theorem one_sub_epsFactor_le (ε : ℝ) (hε : 0 ≤ ε) {x : Fin d → ℝ} (hx : x ≠ 0) :
    1 - rms x / rmsEps ε x ≤ ε / (meanSq x + ε) := by
  have hm := meanSq_pos hx
  have hR := rmsEps_pos ε hε hx
  set a := rms x
  set b := rmsEps ε x
  have ha0 : 0 ≤ a := Real.sqrt_nonneg _
  have ha2 : a ^ 2 = meanSq x := Real.sq_sqrt (meanSq_nonneg x)
  have hb2 : b ^ 2 = meanSq x + ε := Real.sq_sqrt (by linarith)
  have hab : a ≤ b := Real.sqrt_le_sqrt (by linarith)
  have hb2pos : 0 < b ^ 2 := by positivity
  have hlhs : 1 - a / b = (b ^ 2 - a * b) / b ^ 2 := by field_simp
  have hrhs : ε / (meanSq x + ε) = (b ^ 2 - a ^ 2) / b ^ 2 := by
    rw [← hb2, ha2, hb2]
    ring
  rw [hlhs, hrhs, div_le_div_iff_of_pos_right hb2pos]
  nlinarith [mul_nonneg ha0 (sub_nonneg.mpr hab)]

/-- (4) General error with `eps`: for `x, x̂ ≠ 0`, `|w i| ≤ W`,
`‖Δ‖ ≤ W √d (‖Δdir‖ + max (eps/(mean x̂² + eps)) (eps/(mean x² + eps)))`.
The size of `x̂` enters only through the `eps` term. -/
theorem rmsNormEps_sub_le (ε : ℝ) (hε : 0 ≤ ε) (w : Fin d → ℝ) (W : ℝ)
    (hW : ∀ i, |w i| ≤ W) {x xh : Fin d → ℝ} (hx : x ≠ 0) (hxh : xh ≠ 0) :
    vnorm (rmsNormEps ε w xh - rmsNormEps ε w x)
      ≤ W * Real.sqrt d * (vnorm (unitDir xh - unitDir x)
          + max (ε / (meanSq xh + ε)) (ε / (meanSq x + ε))) := by
  set gh := rms xh / rmsEps ε xh
  set g := rms x / rmsEps ε x
  have hgh := epsFactor_mem ε hε hxh
  have hg := epsFactor_mem ε hε hx
  have hgh1 := one_sub_epsFactor_le ε hε hxh
  have hg1 := one_sub_epsFactor_le ε hε hx
  have hsplit : rmsNormEps ε w xh - rmsNormEps ε w x
      = gh • (rmsNorm w xh - rmsNorm w x) + (gh - g) • rmsNorm w x := by
    rw [rmsNormEps_eq_smul ε hε w hxh, rmsNormEps_eq_smul ε hε w hx, smul_sub, sub_smul]
    abel
  have hA := rmsNorm_dir_lipschitz w W hW xh x
  have hr := vnorm_rmsNorm_le w W hW hx
  have hdiff : |gh - g| ≤ max (ε / (meanSq xh + ε)) (ε / (meanSq x + ε)) := by
    rw [abs_le]
    constructor
    · linarith [le_max_left (ε / (meanSq xh + ε)) (ε / (meanSq x + ε))]
    · linarith [le_max_right (ε / (meanSq xh + ε)) (ε / (meanSq x + ε))]
  set A := vnorm (rmsNorm w xh - rmsNorm w x)
  set Mx := max (ε / (meanSq xh + ε)) (ε / (meanSq x + ε))
  have hA0 : 0 ≤ A := vnorm_nonneg _
  have hW0 : 0 ≤ W * Real.sqrt d := le_trans (vnorm_nonneg _) hr
  rw [hsplit]
  calc vnorm (gh • (rmsNorm w xh - rmsNorm w x) + (gh - g) • rmsNorm w x)
      ≤ vnorm (gh • (rmsNorm w xh - rmsNorm w x)) + vnorm ((gh - g) • rmsNorm w x) :=
        vnorm_add_le _ _
    _ = |gh| * A + |gh - g| * vnorm (rmsNorm w x) := by rw [vnorm_smul, vnorm_smul]
    _ ≤ 1 * A + Mx * (W * Real.sqrt d) := by
        rw [abs_of_nonneg hgh.1]
        have hMx : 0 ≤ Mx := le_trans (abs_nonneg _) hdiff
        exact add_le_add (mul_le_mul_of_nonneg_right hgh.2 hA0)
          (mul_le_mul hdiff hr (vnorm_nonneg _) hMx)
    _ ≤ W * Real.sqrt d * vnorm (unitDir xh - unitDir x) + Mx * (W * Real.sqrt d) := by
        linarith
    _ = _ := by ring

/-- (4) Pure rescale with `eps` (`x̂ = c • x`, `c > 0`):
`‖Δ‖ ≤ W √d · max (eps/(c² mean x² + eps)) (eps/(mean x² + eps))`. -/
theorem rmsNormEps_smul_pos_le (ε : ℝ) (hε : 0 ≤ ε) (w : Fin d → ℝ) (W : ℝ)
    (hW : ∀ i, |w i| ≤ W) {x : Fin d → ℝ} (hx : x ≠ 0) {c : ℝ} (hc : 0 < c) :
    vnorm (rmsNormEps ε w (c • x) - rmsNormEps ε w x)
      ≤ W * Real.sqrt d * max (ε / (c ^ 2 * meanSq x + ε)) (ε / (meanSq x + ε)) := by
  have h := rmsNormEps_sub_le ε hε w W hW hx (smul_ne_zero hc.ne' hx)
  rwa [unitDir_smul_pos x hc, sub_self, vnorm_zero, zero_add, meanSq_smul] at h

/-- (4) The same, as ratios: the rescale error is at most
`W √d · max (eps / mean x̂²) (eps / mean x²)` with `x̂ = c • x`, so it vanishes
as `eps / mean x² → 0` (for fixed `c`). -/
theorem rmsNormEps_smul_pos_le_ratio (ε : ℝ) (hε : 0 ≤ ε) (w : Fin d → ℝ) (W : ℝ)
    (hW : ∀ i, |w i| ≤ W) {x : Fin d → ℝ} (hx : x ≠ 0) {c : ℝ} (hc : 0 < c) :
    vnorm (rmsNormEps ε w (c • x) - rmsNormEps ε w x)
      ≤ W * Real.sqrt d * max (ε / meanSq (c • x)) (ε / meanSq x) := by
  have hm := meanSq_pos hx
  have hcm : 0 < meanSq (c • x) := meanSq_pos (smul_ne_zero hc.ne' hx)
  have h := rmsNormEps_smul_pos_le ε hε w W hW hx hc
  have hW0 : 0 ≤ W * Real.sqrt d :=
    le_trans (vnorm_nonneg _) (vnorm_rmsNorm_le w W hW hx)
  refine le_trans h (mul_le_mul_of_nonneg_left ?_ hW0)
  rw [meanSq_smul] at hcm ⊢
  apply max_le_max
  · exact div_le_div_of_nonneg_left hε hcm (by linarith)
  · exact div_le_div_of_nonneg_left hε hm (by linarith)

/-- (4) Logit form: `|⟪q, rmsNormEps (c • x)⟫ - ⟪q, rmsNormEps x⟫|
≤ ‖q‖ · W √d · max (eps/(c² mean x² + eps)) (eps/(mean x² + eps))`. -/
theorem logit_rmsNormEps_smul_pos_le (ε : ℝ) (hε : 0 ≤ ε) (q w : Fin d → ℝ) (W : ℝ)
    (hW : ∀ i, |w i| ≤ W) {x : Fin d → ℝ} (hx : x ≠ 0) {c : ℝ} (hc : 0 < c) :
    |q ⬝ᵥ rmsNormEps ε w (c • x) - q ⬝ᵥ rmsNormEps ε w x|
      ≤ vnorm q * (W * Real.sqrt d
          * max (ε / (c ^ 2 * meanSq x + ε)) (ε / (meanSq x + ε))) := by
  rw [← dotProduct_sub]
  exact le_trans (abs_dotProduct_le _ _)
    (mul_le_mul_of_nonneg_left (rmsNormEps_smul_pos_le ε hε w W hW hx hc) (vnorm_nonneg _))

/-! ## Axiom check -/

#print axioms rmsNorm_smul_pos
#print axioms rmsNorm_smul_neg
#print axioms logit_rmsNorm_smul_pos
#print axioms qkKeys_rescale
#print axioms qk_attnScore_rescale
#print axioms qk_attnWeight_rescale
#print axioms qk_attnOut_rescale
#print axioms rmsNorm_eq_dir
#print axioms rmsNorm_eq_of_unitDir_eq
#print axioms rmsNorm_eq_iff_unitDir_eq
#print axioms rmsNorm_sub_eq
#print axioms rmsNorm_dir_lipschitz
#print axioms rmsNorm_dir_lipschitz_max
#print axioms rmsNorm_dir_lower
#print axioms vnorm_rmsNorm_eq
#print axioms vnorm_rmsNorm_uniform
#print axioms abs_vnorm_rmsNorm_sub_le
#print axioms abs_vnorm_rmsNorm_sub_le_spread
#print axioms vnorm_rescale_sub
#print axioms abs_vnorm_rmsNorm_ratio_sub_one_le
#print axioms rmsNorm_const_eq_unitDir
#print axioms unitDir_sub_le
#print axioms unitDir_sub_le_rescaled
#print axioms qk_attn_dir_bound
#print axioms rmsNormEps_eq_smul
#print axioms epsFactor_mem
#print axioms one_sub_epsFactor_le
#print axioms rmsNormEps_sub_le
#print axioms rmsNormEps_smul_pos_le
#print axioms rmsNormEps_smul_pos_le_ratio
#print axioms logit_rmsNormEps_smul_pos_le
