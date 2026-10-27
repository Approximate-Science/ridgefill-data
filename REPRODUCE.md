# Checking the load-bearing numbers

A second reader's checklist for the numbers that carry the paper, in the order to do them. Each check names the
files, what to recompute and what counts as a pass. The scripts in `recompute/` do each computation; this list is for
a reader who wants to redo it independently. Stop and write it down the moment a number does not reproduce.

Python 3.10+ with numpy is enough for everything below. Bootstraps pair items (RULER, LongBench) or documents (KL,
loss); the seeds and resample counts are those of the paper.

## 1. The gate (LB6)

- `results/qwen38-27b/results/checks/1792971012492247646/`: CHECK-A compares the fork with RidgeFill off against
  upstream llama.cpp 1242be66f on 7 comparisons (8K, 16K, 32K, two LongBench prompts, scorer loss on 2 documents);
  CHECK-B compares E3 resident with streamed at 16K and 65K. Each record must say identical and name both builds.
  The logit dumps themselves are not released; their sha256 are in `provenance/external.tsv`.
- `results/qwen38-27b/results/RUN-MANIFEST.json` and `provenance/inputs/qwen38-27b/LOCK.json`: `server_runtime_diff_empty`
  is true, and the archive hashes match.
- Pass: 7 + 2 identical; builds and hashes match. Script: `recompute/e3_checks.py`.

## 2. 27B time to first token (LB2)

- `results/qwen38-27b/results/timing/{16384,32768,65536,120000}/{stock,resident,streamed}/result.json`: median of the
  three timed requests' prompt times; ratios stock / resident and stock / streamed.
- Expected 1.249 / 1.192 (16K), 1.302 / 1.267 (32K), 1.307 / 1.300 (64K), 1.281 / 1.293 (120K); stock medians 12.966,
  27.488, 60.495, 128.100 s. Stock and E3 of one length ran on the same GPU (`gpu` and the `GPU_UUID-n` labels).
- Pass: every ratio within 0.001. Script: `recompute/e3_timing.py`.

## 3. 27B RULER (LB3)

- `results/qwen38-27b/results/ruler/{stock,e3}/`: 13 tasks × 20 items at 16,384 and 32,768 tokens; each record holds
  the score, the answer budget and the hash of its prompt. Item ids, tasks and budgets: `provenance/inputs/qwen38-27b/ruler-index.json`.
- Bootstrap: 10,000 resamples, seed 1, items resampled within task, mean of the 13 task means. Expected Δ −1.19
  [−3.13, +0.79] and −1.42 [−3.21, +0.35]; overall 85.69 / 84.51 and 87.85 / 86.42.
- Re-scoring the generations needs the RULER answers, which are not released; they are rebuilt with the RULER
  generator at the commit named in `provenance/inputs/test-set-locked/MANIFEST.tsv`.
- Pass: task means exact; intervals within ±0.05. Script: `recompute/e3_ruler.py`.

## 4. 27B LongBench v2 (LB4)

- `results/qwen38-27b/longbench/{stock,e3}/*.json` (503 each). `code/qwen38-27b-kit/longbench/report_lb.py` on that
  folder reproduces `REPORT.txt` byte for byte. Gold answers are hashed; each record keeps `judge`.
- Expected 38.57 vs 37.18, Δ −1.39 [−4.97, +2.19]; only stock correct 48, only E3 41; sign test p = 0.525.
- Pass: overall and group means exact; interval within ±0.1. Scripts: `recompute/lb_report.py`, `recompute/lb_groups.py`.

## 5. 27B post-prompt agreement (LB5)

- `results/qwen38-27b/PER-DOC-KL.json`, `results/qwen38-27b/results/kld/{16384,32768}/`: 100 documents per length,
  512 scored positions each.
- Expected 16K: mean KL 0.0894 [0.0399, 0.1556], median 0.0040, p99 0.965, top-1 94.64 %. 32K: 0.2656
  [0.0599, 0.6017], 0.0042, 12.77, 93.60 %.
- Pass: medians and top-1 exact; means within 1e-4. Script: `recompute/e3_kl.py`.

## 6. Reference study: fitted fill against identity fill (LB8, LB9)

- `results/reference/{llama-3.1-8b-instruct,qwen3-8b,qwen3.5-9b}-ruler/` and `*-decode/`. Filled-region subset: samples
  whose every scored needle lies before the 4,096-token tail (216 Llama, 138 Qwen3-8B, 219 Qwen3.5).
- Paired BCa bootstrap, 5,000 resamples within task. Expected +39.7 [+32.6, +46.5] Llama, +49.6 [+43.3, +56.2]
  Qwen3-8B, +3.3 [−1.3, +7.8] Qwen3.5. Decode KL 0.124 vs 0.334, 0.012 vs 0.021, 0.013 vs 0.032.
- Pass: arm means exact; intervals within ±0.5. Scripts: `recompute/lb8_fill_vs_identity.py`, `recompute/lb9_decode.py`.

## 7. Flash-Next (LB7)

- `results/flashnext/final-det-20261010/results/{exact,ridgefill,graded}/`: speed medians and ratios 1.50 / 1.54 / 1.60;
  perplexity gap +2.65 % [−0.02 %, +4.93 %] over 90 documents (per-document ΔNLL averaged within domain); RULER 99.8
  vs 100.0. Quality mode against plain RidgeFill: −1.02 % [−2.87 %, +0.80 %].
- Scripts: `recompute/fn_speed.py`, `recompute/fn_quality.py`, `recompute/fnq.py`, `recompute/fn_determinism.py`.

## 8. SwiftKV and POP (LB10, LB11, R3, R5)

- `results/reference/llama-3.1-8b-instruct-nll/`, `*-ruler-deep/`, `llama-3.1-8b-instruct-nll-fresh/`. Expected
  +0.010 [−0.004, +0.025]; −0.024 [−0.037, −0.015]; NI lower bounds −2.18 and −2.15 (margin −3); −0.268
  [−0.336, −0.200]. Key tokens (R3): +0.229 [−0.020, +0.485] and −0.224 [−0.415, −0.057].
- Compute against POP (R5, §2.5): `code/kva/compute.py` with `recompute/cfg/`. Scripts: `recompute/lb10_swiftkv.py`,
  `recompute/lb11_pop.py`, `recompute/r3_keytok.py`, `recompute/r5_compute_vs_pop.py`.

Write back, for each check, PASS or the number you got next to the expected one and the file it came from.
