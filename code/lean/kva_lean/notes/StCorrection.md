# StCorrection.lean — the +st state correction scaled by rho (engine change E1)

File: `projects/research/kva/lean/kva_lean/KvaLean/StCorrection.lean` (377 lines, imports `GdnContraction`).
Check: `nice -n 10 lake env lean KvaLean/StCorrection.lean` → no errors, no warnings, about 2 s.
No `sorry`/`admit`; all 20 `#print axioms` lines report `[propext, Classical.choice, Quot.sound]`.

## Bottom line

For a decay-gated linear recurrence (scalar gate per step, no erase term), scaling the all-rows correction C by
rho = (decay-weighted share of approximated rows) removes the mean part of the fill error **exactly**, for any
choice of exact rows. rho is the only scale that does this. The unscaled correction over-corrects by the exact
rows' weight times the mean error. With fast-forgetting gates (≤ 0.4) and the last 16 rows exact, rho < 1e-6 at
any prompt length. For the full delta rule (with the erase term) only a norm bound is proved, not the exact
correction.

## Setup

- Rows t = 1..T. State `S 0 = 0`, `S t = g t • S (t-1) + u t` (`glaState`). States live in any real vector space
  (covers `Fin d → ℝ` and the GDN state matrix). Gates `g t` are scalars (one per head per token, like GDN's alpha).
- Weight of row j at time T: `w j = ∏_{k=j+1..T} g k` (`decayW`). Total `W = ∑_{j=1..T} w j` (`wsum`).
- Approximated rows `A ⊆ {1..T}` get input `u j + e j`, the rest `u j` (`approxIn`).
- `rho = (∑_{j∈A} w j) / W` (`rhoE1`). Correction `C = W • μ`.

## Theorems

**(1) Closed form**
- `glaState_closed`: `S T = ∑_{j=1..T} w j • u j`.
- Helpers: `decayW_self` (newest row has weight 1), `decayW_nonneg`, `decayW_succ` (one more step multiplies
  every older weight by the new gate).

**(2) Error and correction**
- `st_error_eq`: terminal-state error from approximating A is exactly `∑_{j∈A} w j • e j`.
- `st_error_any_scale`: if `e j = μ + ε j` on A, subtracting any multiple `c • C` leaves
  `∑_{j∈A} w j • ε j + (∑_{j∈A} w j − c·W) • μ`.
- `st_correction_exact` (the main result): with `W ≠ 0`, subtracting `rho • C` leaves exactly
  `∑_{j∈A} w j • ε j`. Only the zero-mean part remains.
- `st_scale_unique`: if `μ ≠ 0` and `W ≠ 0`, rho is the only scale with that property.
- `st_unscaled_overcorrects`: applying the full C (the original +st) when some rows are exact leaves
  `∑_{j∈A} w j • ε j − (∑_{j exact} w j) • μ`, i.e. it over-corrects by the exact rows' share. This is why E1
  is needed.
- `st_full_error_decomp`: with every row approximated, the error is `C + ∑_j w j • ε j`, so C is exactly the mean
  part of the all-approximated error (what +st was fitted on).
- Endpoints: `rhoE1_empty` (A = ∅ gives rho = 0), `st_empty_exact` (then no correction is applied and the state is
  exact), `rhoE1_all` (A = all rows gives rho = 1, the original +st).
- Range: `rhoE1_nonneg`, `rhoE1_le_one` (0 ≤ rho ≤ 1 for non-negative gates). `wsum_ge_one`: with non-negative
  gates and T ≥ 1, W ≥ 1, so the `W ≠ 0` hypothesis always holds for real GDN gates.

**(3) Fast forgetting** (gates `0 ≤ g k ≤ γ`, last m rows exact, i.e. `A ⊆ {1..T−m}`)
- `decayW_tail_le`: each approximated row's weight `w_T j ≤ γ^m · w_{T−m} j`.
- `wsumA_tail_le`: `∑_{j∈A} w_T j ≤ γ^m · ∑_{j∈A} w_{T−m} j`.
- `rhoE1_tail_le`: `rho ≤ γ^m · (∑_{j∈A} w_{T−m} j) / W` (the form asked for in the brief).
- `wsum_le_geom`: with γ < 1, `W ≤ 1/(1−γ)` at every T.
- `rhoE1_le_geom`: with γ < 1, **`rho ≤ γ^m / (1−γ)` for every prompt length T** (uses W ≥ 1 and the bound above).
- `rhoE1_fast_forget_numeric`: gates ≤ 0.4 and the 16 rows before the tail exact give **rho < 1e-6**.

Numbers: 0.4^16 = 4.3e-7; the full bound is 0.4^16 / 0.6 = 7.2e-7. That is consistent with the GPU smoke check's
rho_min = 0 on fast-forgetting heads: the theorem guarantees rho is below 1e-6, which reads as 0 at any
printed precision coarser than that, and real gates below 0.4 make it smaller still. The theorem does not say rho
is exactly 0. It is strictly positive whenever A is non-empty and the gates are positive. So the smoke-check value
is correct behaviour, not a bug.

**(4) Full delta rule, bound only** (reuses `gdn_prefix_contraction` from `GdnContraction.lean`)
- `gdn_tail_exact_bound`: two full GDN runs, erase term included, that get identical inputs (alpha, beta, k, v) on
  the last L steps, with `0 ≤ alpha ≤ γ ≤ 1`, `beta ∈ [0,2]` and unit k there, end at most `γ^L` times their
  state difference at the start of those L steps apart (Frobenius norm). Earlier steps may differ in anything,
  including alpha, beta and k. So if the last 16 rows are exact on a head with gates ≤ 0.4, the whole terminal-state
  error from approximated rows is at most 4.3e-7 times the error at that point. In that regime both the error and
  the E1 correction vanish together.

## What it backs

- **E1 is the right scale.** It explains why random 8.8% exact rows + st still help on FN4 (Flash-Next, quick
  tier, 9 docs: −0.0145 nats [−0.0243, −0.0059] vs the same rows without st). With exact rows present, rho-scaled
  C removes the mean error of the rows that are still approximated, and no other scale does. The unscaled C would
  push the exact rows' share of μ back into the state. The theorem does not predict the size of the gain. That
  depends on how large μ is relative to ε, which is empirical.
- **rho_min = 0 in the smoke check is expected**: `rhoE1_fast_forget_numeric`.

## Assumptions and what it does NOT cover

- **Linear, decay-only recurrence.** The exact statements (1)–(3) cover `S t = g t • S (t−1) + u t` only. With
  GDN's erase term `(I − beta k kᵀ)`, the transition is a matrix that depends on the row. The mean part then
  does not factor as `(∑ w j) • μ`, and rho-scaling is not exact. Only the bound in (4) is proved for that case.
- **Same gates in both runs.** (1)–(3) assume approximating a row changes only its input u j, not its gate. In
  the real model, an approximated row's alpha (and beta, k) also change. (4) allows this, but only as a bound.
- **C uses the current prompt's W.** The theorem takes `C = W • μ` with this prompt's gates. In the engine, C is
  fitted on calibration prompts, so it is an average over their gates. The mismatch between that average and this
  prompt's W is not modelled.
- **One shared mean μ.** Every approximated row is assumed to have the same mean error μ (a position-independent
  mean). Position- or content-dependent mean error is not covered.
- **"Zero-mean" is interpretation.** Lean proves an algebraic identity for any split `e j = μ + ε j`. That the
  remaining `∑ w j • ε j` is small or averages out is a statistical assumption, not part of the proof.
- (3) needs `0 ≤ g k ≤ γ` for **all** k (the weaker "for k in the last m steps" would also work for
  `decayW_tail_le`; the global form keeps statements simple). `rhoE1_le_geom` needs γ < 1. (4) needs γ ≤ 1 because
  `gdn_prefix_contraction` requires alpha ≤ 1.
- Indexing: (1)–(3) use rows 1..T (`Finset.Ioc 0 T`), matching the brief. (4) uses `GdnContraction.lean`'s 0-indexed
  step convention.
