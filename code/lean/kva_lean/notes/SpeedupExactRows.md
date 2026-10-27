# SpeedupExactRows.lean: KVA speedup when some prefix rows stay exact

File: `lean/kva_lean/KvaLean/SpeedupExactRows.lean` (imports `KvaLean.Speedup`).
Status (2026-10-04): compiles with `lake env lean` in about 2 s, no warnings, no `sorry`/`admit`.
All 17 `#print axioms` lines report `[propext, Classical.choice, Quot.sound]`.
Not added to `KvaLean.lean` or the README (the coordinator integrates).

## The model

This file uses the symbols of `Speedup.lean`:

- `N` is the prompt length, `L` the number of layers, `S` the split, `T` the exact tail;
- `c` is the per-token per-layer exact cost, `f` the fill cost per skipped token per late layer, and `p` the projector cost per skipped token.

**The new ingredient.** `R` of the `N − T` prefix rows stay exact in the late layers. Each one costs `κ·c` per late
layer, `κ` times a normal exact token's late-layer cost. **`κ` is the brief's "c"**; the letter `c` was already taken
by `Speedup.lean`.

```
C_R = S·N·c + (L−S)·(T·c + R·κ·c + (N−T−R)·f) + (N−T−R)·p        (kvaCostKvaR)
speedup_R = L·N·c / C_R                                           (kvaSpeedupR)
```

**Assumptions built into the model:**

- The early layers are unchanged: every token pays `S·c` there.
- Exact rows pay neither fill nor projector.
- The cost is linear in `R`.

**Two per-row quantities** carry every result:

- **Saving of an approximated row** versus exact: `s = (L−S)·(c−f) − p`.
  - `s ≥ 0` is exactly the existing "cheap projector" condition of `kva_speedupP_ge_one`.
- **Premium of an exact row** versus a normal exact row: `e = (κ−1)·(L−S)·c`.

The key identity (`kvaCostExact_sub_kvaCostKvaR`):

```
C_exact − C_R = (N−T−R)·s − R·e
```

## Theorems in plain words

**Consistency with the existing file**

- `kvaCostKvaR_zero`, `kvaSpeedupR_zero`: with `R = 0` you get back the existing projector-cost model
  (`kvaCostKvaP`, `kvaSpeedupP`).
- `kvaCostKvaR_unit_premium`: with `κ = 1` and `p = 0`, keeping `R` prefix rows exact is identical to a longer exact
  tail `T + R`.
- `kvaCostKvaR_eq_add`: the cost is affine in `R`. Each exact row adds `(L−S)·(κc − f) − p = s + e`
  (`kvaRowSaving_add_premium`).
- `kvaCostKvaR_pos`: the cost is positive under the standing hypotheses.

**1. Monotonicity**

- `kva_speedupR_anti_R`: the speedup is non-increasing in `R`, **provided** an exact row costs at least as much as an
  approximated one: `p ≤ (L−S)·(κc − f)`.
  - This hypothesis is needed. If the projector costs more than an exact row's late layers, keeping rows exact makes
    prefill faster, and the theorem would be false.
- `kva_speedupR_anti_R_of_premium`: that condition holds automatically when `κ ≥ 1` and the projector is cheap
  (`p ≤ (L−S)(c−f)`, the same condition the existing file uses).
- `kva_speedupR_anti_premium`: the speedup is non-increasing in the premium `κ`. This needs no extra condition.

**2. Exact break-even**

- `kva_speedupR_ge_one_iff`: KVA with exact rows is at least as fast as exact **iff**
  `R·(κ−1)·(L−S)·c ≤ (N−T−R)·((L−S)·(c−f) − p)`.
  - The rows' premium has to fit inside the net saving of the rows that are still approximated.
- `kva_speedupR_ge_one_iff_total`: the same condition in the brief's phrasing:
  `κ·R·(L−S)·c ≤ (N−T)·(L−S)·c − (N−T−R)·((L−S)·f + p)`.
  - The exact rows' whole late-layer cost has to fit inside what exact prefill spends on the late layers of all
    `N − T` prefix rows, minus what the approximated rows still cost.
  - **Correction to the brief's wording.** "c·R·(per-row cost) ≤ savings on the approximated rows − fill/projector"
    is right only if "savings" counts all `N − T` prefix rows. Counted over the approximated rows alone, it is
    off by `R·(L−S)·c`. Only the premium `κ − 1` competes with the saving, because exact prefill also pays for those
    `R` rows.
- `kva_speedupR_lt_one_iff`: the strict version (slower iff `>`).
- `kva_speedupR_ge_one_iff_le_breakEven` / `kva_speedupR_lt_one_iff_breakEven_lt`: as a threshold. When
  `s + e > 0`, KVA is at least as fast iff
  ```
  R ≤ R* = (N−T)·s / (s+e)
         = (N−T)·((L−S)(c−f) − p) / ((L−S)(c−f) − p + (κ−1)(L−S)c)        (kvaBreakEvenR)
  ```
  - The break-even fraction of prefix rows, `R*/(N−T) = s/(s+e)`, does not depend on `N` or `T`. It depends only on
    the per-row costs.
- `kva_speedupR_ge_one_iff_calib` (two-point calibration): because the cost is affine in `R`, the break-even can be
  read off three costs. These are exact (`C_exact`), no exact rows (`C_0`), and `R₁ > 0` exact rows (`C_{R₁}`).
  KVA is at least as fast iff `R·(C_{R₁} − C_0) ≤ R₁·(C_exact − C_0)`. This is the form the measurements plug into.

**3. Corollaries**

- `kva_speedupR_lt_one_of_large`: if `κ > 1`, `S < L`, `T ≤ N` and the projector is cheap, then every `R > R*`
  (formula above, written out in the statement) gives a speedup strictly below 1.
- `kvaBreakEvenR_mem`: under the same conditions with `T < N`, `0 ≤ R* < N − T`. So the losing range is never empty.
- `kva_speedupR_all_exact_lt_one`: keeping every prefix row exact (`R = N − T`) at `κ > 1` is strictly slower than
  exact.
- `kva_speedupR_ge_one_of_unit_premium`: with `κ = 1` and `p = 0` (and `f ≤ c`), the speedup is at least 1 for every
  `0 ≤ R ≤ N − T`. It is proved by reduction to the existing `kva_speedup_ge_one` with tail `T + R`.

## Plugging in the measured numbers

**The measurements.** These are as given in the brief: Flash-Next, TTFT, split 24 of 48, T = 2048. The brief does
not give the engine build or protocol version, and these numbers have **not** been re-checked against stamped
results.

| prompt | exact `C_exact` | KVA, no exact rows `C_0` | KVA, 5.6% rows exact `C_{R₁}` |
|---|---|---|---|
| 9K  | 9.91 s  | 6.44 s  | 7.69 s  |
| 16K | 16.21 s | 9.63 s  | 12.17 s |
| 32K | 32.75 s | 19.14 s | 23.40 s |

**Break-even fraction (calibration form).** `ρ* = 5.6% · (C_exact − C_0)/(C_{R₁} − C_0)`. This needs only the model's
linearity in `R`; it does not need `κ`, `f`, `p` or the cost per layer separately.

- 9K: 5.6% · 3.47/1.25 = **15.5%**
- 16K: 5.6% · 6.58/2.54 = **14.5%**
- 32K: 5.6% · 13.61/4.26 = **17.9%**

**Prediction at 22.4% rows exact (100% class protection, = 4 × 5.6%).** `C = C_0 + 4·(C_{R₁} − C_0)`:

| prompt | predicted KVA TTFT | exact | predicted speedup |
|---|---|---|---|
| 9K  | 11.44 s | 9.91 s  | 0.87 (15% slower) |
| 16K | 19.79 s | 16.21 s | 0.82 (22% slower) |
| 32K | 36.18 s | 32.75 s | 0.90 (10% slower) |

22.4% is above the break-even of 14.5-18% at all three lengths. So the model predicts that full class protection is
**slower than exact**, by about 10-22%. This was **not measured**.

**What κ these numbers imply.** This step needs one more assumption: the per-token per-layer cost is the same for
all layers and tokens. Then the late-layer exact cost of the prefix rows is `≈ C_exact · (24/48) · (N−T)/N`, taking
9K = 9216 (9000 changes nothing material).

- **Approximated rows.** They still cost `a ≈ 7-11%` of an exact row's late layers (`(L−S)f + p` in units of
  `(L−S)c`).
- **Premium relative to a normal exact token's late-layer cost (the model's `κ`):** about 5.9 / 6.5 / 5.1 at
  9K / 16K / 32K.
- **Premium relative to a whole exact token (all 48 layers):** about 2.9 / 3.2 / 2.5. This is the brief's
  "c ≈ 2.5-3".
  - The brief's figure therefore matches the whole-token normalisation, not the late-layer one in its own
    definition. In the model's terms, `κ ≈ 5-6.5`.
  - The 22.4% prediction above does not depend on which normalisation you pick.
- **Consistency check:** `ρ* = (1 − a)/(κ − a)` gives 0.90/5.80 = 15.5% at 9K, the same as the calibration form.

## What is proved and what is not

**Proved** (for any real parameters satisfying the stated hypotheses):

- the cost formula's algebra;
- monotonicity in `R` and `κ`;
- the exact break-even condition and the threshold `R*`;
- the corollaries.

**Not proved: empirical inputs.**

- **The premium `κ` is an empirical input, not a theorem.** Its value (≈ 5-6.5 against late-layer cost, ≈ 2.5-3
  against a whole token) comes from one measured config per prompt length. Nothing in Lean derives it.
- **Linearity in `R` is a modelling assumption.**
  - The calibration uses two points (`R = 0` and 5.6%), so it has zero degrees of freedom. Linearity has not been
    tested at a third point.
  - Going to 22.4% is a 4× extrapolation.
  - On this engine the exact rows run the full MoE with experts in host RAM. A fixed per-pass expert-transfer cost
    would make the cost sub-linear in `R`, and 22.4% could then be cheaper than predicted. Contention could make it
    super-linear.
  - One measured TTFT at 22.4% would settle this.
- **"Measured TTFT = model cost" is assumed.** The model has no fixed overheads that change with `R`, no attention
  cost that grows quadratically with `N`, and no overlap or scheduling effects.
- **Exact rows are assumed to skip the projector.** If the engine runs the projector on every prefix row anyway, each
  exact row also pays `p`. That makes `s + e` larger and lowers the break-even slightly.

**Not covered at all:**

- quality: why rows are protected, and what protection buys;
- the early layers (assumed identical in both arms);
- memory and VRAM;
- any engine-specific cost structure beyond the per-row premium.
