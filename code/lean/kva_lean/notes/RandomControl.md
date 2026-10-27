# RandomControl.lean: a count-matched random control measures selection skill, not row count

File: `lean/kva_lean/KvaLean/RandomControl.lean`
(imports `KvaLean.ImpactBound`). Check: `nice -n 10 lake env lean KvaLean/RandomControl.lean`,
about 1.5 s wall and 3.4 GB RSS. No errors, no warnings, no `sorry`/`admit`. All 25 `#print axioms` lines report
`[propext, Classical.choice, Quot.sound]`.

## Setup

- Rows `j : Fin n`. Each row has a real impact `I j`: the benefit of keeping that row exact (larger is better).
  Impacts may be negative.
- `gain I S = ∑_{j∈S} I j` is the additive (first-order) gain of keeping the set `S` exact.
- `avgSet G k` is the average of any set function `G` over the `k`-subsets (`powersetCard k univ`).
  This is the uniformly random control.
- `randomGain I k = avgSet (gain I) k` is the control's expected gain.
- `skill I S = gain I S - randomGain I #S` is rule minus count-matched control.

Sign convention: the measured differences are loss changes (negative is better), so the measured skill is minus
the loss difference.

## Theorems in plain words

### 1. Expected random gain

- `card_kSubsets`: there are `choose n k` subsets of size `k`.
- `card_kSubsets_mem` (needs `k ≥ 1`): each row lies in `choose (n-1) (k-1)` of them.
- `sum_gain_kSubsets` (needs `k ≥ 1`): summed over all `k`-subsets, the gains equal
  `choose (n-1) (k-1) · ∑_j I j`. The proof is by double counting.
- `choose_ratio` (needs `0 < k ≤ n`): `choose (n-1) (k-1) / choose n k = k/n`.
- `randomGain_eq` (needs `0 < n` and `k ≤ n`; includes `k = 0`): random `k` rows gain on average
  `(k/n) · ∑_j I j`.

### 2. Selection skill

- `skill_eq`: `skill = ∑_{j∈S} I j − (#S/n)·∑_j I j`.
- `skill_eq_sum_sub_mean`: `skill = ∑_{j∈S} (I j − mean I)`. Only a kept row's excess over the average row counts.
- `skill_pairwise`: `n · skill = ∑_{j∈S} ∑_{j'∉S} (I j − I j')`.
  - Skill is literally a pairwise ranking score over every (kept, dropped) pair.
  - Equivalently, `skill = (k(n−k)/n) · (mean impact kept − mean impact dropped)`.
- `skill_add_const`: adding the same benefit `c` to every row leaves skill unchanged.
  The "more exact rows help" part cancels exactly.
- `skill_const`: if every row has the same impact, every selection of every size has skill 0.
- `unmatched_control_bias`: against a control of a different size `k'`, rule minus control = skill
  + `(#S − k')·mean I`. A size mismatch adds a pure count term. That term is why the control must be count-matched.
- `skill_top_nonneg`: the top-`k` set has skill ≥ 0. "Top-`k`" means a threshold `τ` with `I ≥ τ` on the kept rows
  and `I ≤ τ` on the dropped rows, so ties are allowed.
- `skill_top_max`: the top-`k` set has the largest skill among all sets of its size. It needs no sign assumption on `I`.
- `skill_bottom_nonpos`: the bottom-`k` set has skill ≤ 0.
- `skill_pos_exists_pair`: positive skill implies some kept row has strictly larger impact than some dropped row.
- `skill_neg_exists_pair`: negative skill (worse than random) implies some kept row has strictly smaller impact than
  some dropped row.
- Selection independent of the impacts has zero expected skill:
  - `sum_skill_kSubsets`: for `S` uniform over the `k`-subsets, the skills sum to 0 over all of them.
  - `sum_skill_relabel` (with helper `sum_perm_apply`): for a fixed `S` and impacts relabelled by a uniformly random
    permutation `σ` (`I ∘ σ`), the skills sum to 0 over all `σ`. This is the "rule carries no information about which
    row has which impact" case.
- Model-free versions, for any set function `G` with no additivity:
  - `sum_sub_avgSet`: deviations from the control average sum to 0 over the `k`-subsets.
  - `exists_ge_avgSet` (needs `k ≤ n`): some `k`-set is at least the control average, so the best `k`-set never loses
    to its random control.

### 3. First-order model: the control's gain is linear in the count

- `randomGain_linear`: `randomGain k = k · mean I`.
- `randomGain_add`: `randomGain (k₁+k₂) = randomGain k₁ + randomGain k₂`.
- `randomGain_full`: keeping all `n` rows gives the full gain `∑_j I j`.
- `randomGain_proportional`: `randomGain k = (k/n) · randomGain n`.

### 4. Link to `ImpactBound.lean`

`skill_top_max` is proved from `topSet_compl_sum_le`.
- The impacts are shifted by `∑|I j|` to make them non-negative, which that lemma needs.
- The shift cancels because the two sets have the same size.
- With `I j = e·‖g j‖`, this is Lemma 6's top-gradient-norm choice: it maximises the skill of the bound.

## What it backs (numbers as given in the brief: Flash-Next, quick-tier ppl, paired)

The protocol version and set were not given in the brief. The coordinator should check these against the stamped
tables before quoting them.

- **Class protection 25% + st** beat its count-matched random + st by −0.0092 nats [−0.0168, −0.0018].
  - That is skill ≈ +0.0092, with a CI that excludes 0.
  - By `skill_pairwise` and `skill_pos_exists_pair`, in the additive model the protected rows carry a higher mean
    impact than the unprotected ones.
  - By `skill_add_const` and `unmatched_control_bias`, this is not explained by keeping 25% of rows. The count-matched
    control already gets that benefit.
  - "+ st" is in both arms. In the additive model a shared component cancels in the difference.
- **R15 regex rule** was +0.0140 [+0.0060, +0.0225] worse than its random control.
  - That is skill ≈ −0.0140, with a CI that excludes 0.
  - By `skill_neg_exists_pair` and `skill_pairwise`, its kept rows have a lower mean impact than its dropped rows.
    It ranks worse than chance, not merely no better. The bottom-`k` set is the extreme case (`skill_bottom_nonpos`).
  - Its `k` was not in the brief, so the per-row gap is not computed.
- **Random 8.8% exact rows alone** improved ppl by about −0.021 nats over no exact rows.
  This is an observation under the first-order model, not a proof.
  - `randomGain_proportional` gives total `∑ I ≈ 0.021 / 0.088 ≈ 0.239` nats, about 0.0024 nats per percentage point
    of rows.
  - This predicts random 17.6% ≈ −0.042, random 25% ≈ −0.060, and all rows exact ≈ −0.239. The last is the full
    KVA-vs-exact ppl gap at this configuration.
  - Measured departures from this straight line measure non-additivity, i.e. the cross terms of Lemma 7.
  - Only one point (8.8%) is given, so linearity itself is untested here.
  - Illustration only, mixing a no-st 8.8% control with the "+ st" 25% comparison: with total ≈ 0.239, a skill of
    0.0092 at k/n = 0.25 means:
    - the protected rows' mean impact is ≈ 1.15× the average row's;
    - 25% of rows carry ≈ 29% of the gain (0.060 + 0.009 of 0.239);
    - kept-minus-dropped mean gap ≈ 0.21× the average row impact.
    This assumes additivity and that "st" does not change per-row impacts.

## Assumptions and what it does NOT cover

- **Additivity.**
  - The control-gain formula, linearity in `k`, the mean-excess and pairwise forms, and the top/bottom-`k` results
    all assume `gain(S) = ∑_{j∈S} I j`.
  - Real gains have interaction terms. Lemma 7 (`CrossTerms.lean`) shows per-token sums miss the joint change.
  - Only `sum_sub_avgSet`, `exists_ge_avgSet` and `sum_skill_kSubsets`-style "random has zero mean skill" hold for any
    set function.
- **Fixed impacts.** `I` is deterministic. The only randomness is the control's draw, or the relabelling `σ`.
- **No statistics.** The measured control uses a finite number of random draws on a finite eval set. Its sampling
  noise, the paired CIs and the bootstrap are not formalised. The theorems concern the exact expectation.
- **"Independent of I"** is formalised two ways: as a uniformly random `S`, and as a fixed `S` under exchangeable
  (uniformly permuted) impacts. A real rule (regex, class) is a fixed function of the content. Whether its choice
  correlates with impact is exactly what skill measures. Nothing here says which rule is best, or that class protection
  approximates the top-`k` set.
- **Edge cases.**
  - `randomGain_eq` needs `k ≤ n` (for `k > n` there are no `k`-subsets and `avgSet` is `0/0 = 0`) and `0 < n`.
  - `card_kSubsets_mem` and `sum_gain_kSubsets` need `k ≥ 1`. For `k = 0`, `choose (n-1) (k-1)` is `choose (n-1) 0 = 1`
    in ℕ subtraction, which is not the count.
