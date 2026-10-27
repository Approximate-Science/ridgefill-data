# QK-norm makes key-size errors irrelevant (notes for `KvaLean/QkNormInvariance.lean`)

File: `lean/kva_lean/KvaLean/QkNormInvariance.lean`
(imports `VecNorm`, `AttnSensitivity`). 45 theorems, no `sorry`/`admit`. All 31 `#print axioms`
lines show `[propext, Classical.choice, Quot.sound]`. Checked with `lake env lean` (about 2 s wall, 3.4 GB RSS).
Not added to `KvaLean.lean` or the README; the coordinator does that.

## Setup

- Vectors are `Fin d → ℝ`. The norm is `vnorm` from `VecNorm.lean`. The direction is
  `unitDir x = x / ‖x‖`, and Lean's `0⁻¹ = 0` makes `unitDir 0 = 0`.
- `meanSq x = (∑ x_i²)/d`, `rms x = √(meanSq x)`, `rmsNorm w x = w ⊙ x / rms x`. This is eps = 0.
- `rmsEps ε x = √(meanSq x + ε)`, `rmsNormEps ε w x = w ⊙ x / rmsEps ε x`.
- `w` is the effective per-dimension gain. Qwen3.5 stores it zero-centred and applies `(1 + w)`: see
  `kva/ops.py` `rms_norm_zc` and transformers `Qwen3_5RMSNorm`. Here `w` means that `(1 + w)`.
- `qkKeys w post k j = post j (rmsNorm w (k j))`. `post j` is any map applied after the norm, such as RoPE
  at position `j`.
- Attention (`attnScore`, `attnWeight`, `attnOut`) is reused from Lemma 5 (`AttnSensitivity.lean`).

**Where the theorems apply in KVA.** The projector predicts the layer input. That input goes through the
layer's input RMSNorm, then `k_proj`, then (on Qwen3.5) `k_norm`, then RoPE. Sources: the `scripts/fit_kvfold.py`
docstring and `kva/model.py` `reads`. So the filled key does pass through QK-norm, which is the theorems'
assumption. The KN1 rescale arms (`kva/kva.py` `rescale_kv`) act on the **k-normed, pre-RoPE** key.

## Theorems in plain words

### 1. Scale invariance (eps = 0)

- `rmsNorm_smul_pos`: multiplying the key input by any `c > 0` leaves the normalised key unchanged.
- `rmsNorm_smul_neg`: for `c < 0`, the normalised key flips sign.
- Neither needs `x ≠ 0`. At `x = 0` both sides are `0`.

### 2. Attention

- `logit_rmsNorm_smul_pos`: a single logit `q · rmsNorm(c x)` is unchanged for `c > 0`.
- `qkKeys_rescale`, `qk_attnScore_rescale`, `qk_attnWeight_rescale`, `qk_attnOut_rescale`: suppose a projector
  error only rescales each filled key input, with **its own** factor `c_j > 0` per key. Then every logit, every
  softmax weight and every attention output is exactly unchanged, for all queries.
- Any per-position map after the norm is allowed (RoPE, the `1/√d` temperature).

### 3. Only the direction matters (eps = 0)

- `rmsNorm_eq_dir`: `rmsNorm w x = √d · (w ⊙ x/‖x‖)`. The normalised key sees `x` only through its direction.
- `rmsNorm_eq_of_unitDir_eq`: if two inputs have the same direction, they give the same normalised key.
- `rmsNorm_eq_iff_unitDir_eq`: when every gain is nonzero, the converse also holds. Equal normalised keys
  mean equal directions.
- `rmsNorm_sub_eq`: the exact error is `rmsNorm w x̂ − rmsNorm w x = √d · w ⊙ (x̂/‖x̂‖ − x/‖x‖)`.
- **Requested bound.** `rmsNorm_dir_lipschitz` (any `W ≥ |w_i|`) and `rmsNorm_dir_lipschitz_max`
  (`W = max_i |w_i|`):

  `‖rmsNorm w x̂ − rmsNorm w x‖ ≤ max_i|w_i| · √d · ‖x̂/‖x̂‖ − x/‖x‖‖`
- `rmsNorm_dir_lower`: a matching lower bound, `min_i|w_i| · √d · ‖Δdir‖ ≤ ‖Δ‖`. Direction errors are not
  damped. They pass through at a gain between `min|w|·√d` and `max|w|·√d`.
- `qk_attn_dir_bound` (this result plus Lemma 5): replace key `j0` by a filled input `x̂`. Then each attention
  output moves by at most

  `2 M a_{i,j0} (exp(|t| ‖q_i‖ W √d θ) − 1)`

  where `θ` is the direction error, `M ≥ ‖v_j‖` and `a_{i,j0}` is the attention that key `j0` receives.
  `post j0` must be 1-Lipschitz, which RoPE is because it is a rotation. A pure rescale gives `θ = 0`, so the
  bound is `0`.
- `unitDir_sub_le`: the direction error is at most `2‖x̂ − x‖/‖x‖`.
- `unitDir_sub_le_rescaled`: for every `c > 0`, the direction error is at most `2‖c x̂ − x‖/‖x‖`. So only the
  error left after the best rescale can count.

### Post-norm key size (explains the knorm, ktail and kdir arms)

- `vnorm_rmsNorm_eq`: `‖rmsNorm w x‖ = √d ‖w ⊙ x/‖x‖‖`. After the norm, a key's size is a function of its
  direction alone.
- `vnorm_rmsNorm_uniform`: if all gains are equal (`|w_i| = W`), every normalised key has size exactly `W√d`.
- `abs_vnorm_rmsNorm_sub_le`: the size gap between the filled and true normalised keys is at most `W√d θ`.
- `abs_vnorm_rmsNorm_sub_le_spread`: a sharper bound for the same gap.
  - Assume `0 < wmin ≤ |w_i|`, and `|w_i² − c2| ≤ S` for some centre `c2`.
  - Then the gap is at most `√d (S/wmin) θ`.
  - `S` is the spread of the squared gains, so the gap is `0` for uniform gains whatever the direction error.
- `abs_vnorm_rmsNorm_ratio_sub_one_le`: under the same assumptions, the size ratio `‖K‖/‖K̂‖` is within
  `(S/wmin²) θ` of 1. This ratio is the per-row knorm factor, and ktail's scale is a mean of these ratios.
- `vnorm_rescale_sub`: rescaling a nonzero vector `K` to size `t` moves it by exactly `|t − ‖K‖|`. Applied to
  the arms:
  - **knorm** moves the filled key by exactly the size gap.
  - **kdir** (true direction at the filled size) leaves the key off the true key by exactly the same gap.
- `rmsNorm_const_eq_unitDir`: L2 normalisation is the case `w ≡ 1/√d`. So items 1–3 also cover L2-normalised
  keys, such as the Gated DeltaNet layers. Qwen3.5 calls these with `use_qk_l2norm_in_kernel=True` in
  transformers.

### 4. eps > 0

- `rmsNormEps_eq_smul`: RMSNorm with eps equals the exact RMSNorm times a factor
  `γ = rms x / rmsEps ε x`.
- `epsFactor_mem`, `one_sub_epsFactor_le`: `γ` lies in `[0, 1]`, and `1 − γ ≤ ε/(mean x² + ε)`.
- `rmsNormEps_sub_le`: for any `x, x̂ ≠ 0`,

  `‖Δ‖ ≤ W√d (θ + max(ε/(mean x̂² + ε), ε/(mean x² + ε)))`.

  The size of `x̂` enters only through the eps term.
- `rmsNormEps_smul_pos_le`, `rmsNormEps_smul_pos_le_ratio`: a pure rescale `x̂ = c x` (`c > 0`) costs at most
  `W√d · max(ε/mean x̂², ε/mean x²)`. This tends to 0 as `ε/mean x² → 0` with `c` fixed.
- `logit_rmsNormEps_smul_pos_le`: the same bound for one logit, multiplied by `‖q‖`.
- The transformers default for Qwen3.5 `rms_norm_eps` is 1e-6; the checkpoint config was not checked here.
  Key mean squares well above 1e-6 make this term negligible. A key whose mean square is comparable to eps
  gets shrunk, so its size starts to matter.

## What it backs (KN1, paired grade NLL vs kva; numbers as given in the brief)

KN1 numbers are quoted from the coordinator's brief. Take the engine, protocol and set labels from the stamped
KN1 record before quoting them in the paper.

**Llama-3.1-8B (no QK-norm).** The key size reaches the logits linearly: rescaling key `j` by `c` multiplies
its logits by `c`. Lemma 5 then bounds the effect through `δ = (c − 1)k`. Size fixes can therefore help, and
they do:

| Arm | Loss removed (S12 / S16) |
|---|---|
| knorm (oracle) | 38% / 73% |
| ktail (deployable) | 32% / 54% |

There is no invariance theorem for Llama. The contrast is the point.

**Qwen3.5-9B (k_norm with gains `1 + w`).**

- Items 1 and 2: the pre-norm size of a filled key is invisible.
- Post-norm, the size is a function of direction (`vnorm_rmsNorm_eq`).
- The knorm arm moves a key by exactly the post-norm size gap (`vnorm_rescale_sub`). That gap is at most
  `√d (S/wmin) θ`, so knorm cannot fix anything a direction fix would not.
- ktail's per-head scale is a mean of ratios each within `(S/wmin²) θ` of 1.
- Prediction: ≈ no effect. Measured: knorm +0.0012 / +0.0003, ktail −0.0001.
- The true direction (kdir) removed 28% / 20% and restored needles to 56/56. This is consistent with item 3:
  the key error is a direction error. After kdir, the key-side residual is only the post-norm size gap (the
  same small quantity as above).
- The theorems therefore say that the remaining loss is not a key-size effect. It must come from things they do
  not cover, such as values, GDN layers or other paths.

**Side consequence of the same theorem.** Both models apply input RMSNorm before `k_proj`/`v_proj`. So a pure
rescale of a predicted hidden state is invisible to K and V in **both** models (eps = 0). The per-head key-size
error that knorm fixes on Llama is therefore created by hidden-state *direction* errors passing through
`k_proj`. On Qwen3.5, `k_norm` then removes that per-head size as well.

## Assumptions and what this does NOT cover

- Items 1–3 are exact only for **eps = 0**. With eps > 0 the invariance is approximate, with the explicit
  bound in item 4.
- The theorems hold only where the error enters **before** the norm. A rescale applied *after* the norm (as in
  the knorm and ktail arms) is not invisible. The post-norm size lemmas handle that case by showing the size
  gap is small and tied to direction.
- Values are not normalised. Value errors are untouched. They are covered by Lemma 5 (a), not here.
- The bounds hold row by row, and a different rescale factor per key is fine. Query errors are not modelled;
  the query is any fixed vector.
- bf16 rounding and the fp32 casts in `rms_norm_zc` are not modelled.
- `qk_attn_dir_bound` handles one filled key at a time. Several filled keys can be combined with Lemma 6 or 7,
  which is not done here.
- `S` and `wmin` (the spread and minimum of the Qwen3.5 k_norm gains) are hypotheses. Their measured values
  are not used or claimed here. The gain statistics are in `results/arch-contrasts/weights.py`, which was not
  read for numbers.
- Nothing here predicts the size of the Llama gains (38–73%); it only explains why the same arm does nothing on
  Qwen3.5.
- No claim about Qwen3.8-Flash-Next beyond "it applies RMSNorm with a per-dimension gain to each key" (from the
  brief). If so, everything above applies with that gain as `w`.
