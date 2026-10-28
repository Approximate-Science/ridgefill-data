# From the paper to the files

Each row is one claim of the paper: the files that hold its evidence, the script that recomputes it, and how many
values `derived/claims.csv` checks for it. `bash recompute/run_all.sh` prints every value as `id<TAB>value`;
`recompute/gate.py` compares them with `derived/claims.csv`, which also gives each value's expected string.

Abbreviations: `R/` = `results/reference/`, `F/` = `results/flashnext/final-det-20261010/results/`,
`Q/` = `results/qwen38-27b/`, `D/` = `derived/`, `P/` = `provenance/`.

## The method

| Claim | What the paper says | Files | Script | Values |
|---|---|---|---|---|
| LB1a | The maps are closed-form ridge solutions from one capture pass; rows, documents and λ per map | `code/kva/ridge.py`, `code/scripts/fit_sweep.py`; `P/fits/{reference,qwen38-27b,flashnext}/`; each `R/*/cell.json` (map sha256); `P/external.tsv` | `recompute/check_fit_records.py` | 30 |
| LB1b | Every prompt token keeps a cache entry in every layer | the fork's fill path (commit b0439f7bb, `P/external.tsv`); `Q/results/checks/1792971012492247646/CHECK-B/` | read the code | – |

## Qwen3.8-27B in llama.cpp (§4)

| Claim | What the paper says | Files | Script | Values |
|---|---|---|---|---|
| LB2 | TTFT speedup, warm, same GPU, median of three; VRAM over stock | `Q/results/timing/{16384,32768,65536,120000}/{stock,resident,streamed}/` | `recompute/e3_timing.py` | 36 |
| LB3 | RULER 13 tasks × 20 items at 16K and 32K, paired bootstrap | `Q/results/ruler/{stock,e3}/`, `P/inputs/qwen38-27b/ruler-index.json` | `recompute/e3_ruler.py` | 88 |
| LB4 | LongBench v2, 503 items, official CoT protocol; groups | `Q/longbench/{stock,e3}/`, `Q/longbench/REPORT.txt`, `P/inputs/qwen38-27b/longbench-index.json` | `recompute/lb_report.py` (reproduces REPORT.txt byte for byte), `recompute/lb_groups.py`; `code/qwen38-27b-kit/longbench/report_lb.py` | 82 |
| LB5 | Post-prompt agreement and KL, 100 documents per length | `Q/PER-DOC-KL.json`, `Q/results/kld/`, `Q/results/kld-work/` | `recompute/e3_kl.py` | 22 |
| LB6 | Gates: fork with the fill off is byte-identical to upstream; resident = streamed | `Q/results/checks/1792971012492247646/`, `Q/results/RUN-MANIFEST.json`, `P/inputs/qwen38-27b/LOCK.json`; logit dumps by sha256 in `P/external.tsv` | `recompute/e3_checks.py` | 8 |
| S1 | KL by copyable share | `Q/PER-DOC-KL.json` | `recompute/e3_kl.py` | 18 |
| S2 | BF16 reference on 25 documents | `Q/bf16/`, `Q/BF16-COMPARISON.json`, `Q/BF16-SELECTION.json` | `recompute/e3_bf16.py` | 9 |
| S3 | Multi-user runs | `Q/results/mix/`, `Q/results/MIX-HARNESS-AMENDMENT.json` | `recompute/e3_mix.py` | 21 |
| S4 | LongBench workload prefill speedup (SUPPORTING) | `Q/longbench/{stock,e3}/` | `recompute/lb_report.py` | 1 |

## Qwen3.8-Flash-Next in vLLM (§5)

| Claim | What the paper says | Files | Script | Values |
|---|---|---|---|---|
| LB7a | TTFT ratio of medians at 9K, 16K, 32K | `F/{exact,ridgefill}/speed/` | `recompute/fn_speed.py` | 3 |
| LB7b | 64K request time, 25 samples | `F/{exact,ridgefill}/ruler-a/` | `recompute/fn_speed.py --ruler64k` | 4 |
| LB7c | Perplexity gap and RULER | `F/{exact,ridgefill}/ppl-a/`, `F/*/ruler-a/` | `recompute/fn_quality.py` | 10 |
| LB7d | Quality mode against plain RidgeFill | `F/{graded,ridgefill}/ppl-a/` | `recompute/fnq.py` | 8 |
| S5 | Quality mode | `F/graded/` | `recompute/fn_quality.py`, `recompute/fn_speed.py` | 10 |
| S6 | Determinism | `F/exact/{ppl-a,ppl-b}/`, `F/graded/control/`, `results/flashnext/final-det-20261010/{exact-ppl-a-vs-b.txt,bitwise_ab.py}` | `recompute/fn_determinism.py` | 4 |

## Reference study (§6)

| Claim | What the paper says | Files | Script | Values |
|---|---|---|---|---|
| LB8 | Fitted fill against identity fill on filled-region retrieval | `R/*-ruler/` | `recompute/lb8_fill_vs_identity.py` | 18 |
| LB9 | Decode agreement and KL | `R/*-decode/` | `recompute/lb9_decode.py` | 18 |
| LB10 | Against SwiftKV at its split 16 | `R/llama-3.1-8b-instruct-nll/` | `recompute/lb10_swiftkv.py` | 8 |
| LB11 | Against POP at 5/8 depth; fresh documents | `R/*-ruler-deep/`, `R/llama-3.1-8b-instruct-nll-fresh/` | `recompute/lb11_pop.py` | 24 |
| R1, R2 | Filled-region RULER at 8K–32K, every arm | `R/llama-3.1-8b-instruct-ruler/samples/ruler/`, `R/qwen3.5-9b-ruler/samples/ruler/` | `recompute/ruler1.py` | 19 |
| R3 | SwiftKV comparison on key tokens | `R/llama-3.1-8b-instruct-nll/samples/longppl/` | `recompute/r3_keytok.py` | 22 |
| R4 | Qwen3-8B rows against RidgeFill alone, per sample | `R/qwen3-8b-ruler/samples/ruler/` | `recompute/r4_counts.py` | 7 |
| R5 | Qwen3-8B at 5/8 against POP, multiply-accumulates | `recompute/cfg/{llama-3.1-8b,qwen3-8b}.json` | `recompute/r5_compute_vs_pop.py` | 8 |
| R6 | RidgeFill worse than no fill, by length | `R/llama-3.1-8b-instruct-nll/samples/longppl/`, `D/per-document-nll.csv` | `recompute/r6_nofill_by_length.py` | 11 |
| S7 | Cost at the primary split | `R/*-nll/`, `R/*-nll-causal/`, `D/per-document-nll.csv` | `recompute/s7_primary_cost.py` | 18 |
| S8 | Depth dependence on Llama | `R/*-nll/`, `R/llama-3.1-8b-instruct-nll-fresh/` | `recompute/s8_depth.py` | 14 |
| S9 | Full-depth rows and their controls | `R/*-ruler/` | `recompute/s9_rows.py` | 22 |
| S10 | Splices | `R/*-nll-causal/`, `R/*-ruler/` | `recompute/s10_splices.py` | 25 |
| S11 | LongBench v1 on Llama | `R/llama-3.1-8b-instruct-longbench/` (stored scores; gold answers hashed) | `recompute/s11_longbench.py` | 11 |
| S12 | Second question on a once-filled cache | `R/*-secondq/` | `recompute/s12_turn2.py` | 6 |

## Development measurements (Appendix I)

Status DEVELOPMENT, not FINAL: these back §1, §3 and Appendix I, and no number of §4 to §6 rests on them.
`DEV/` = `results/development/`; its `README.md` names the models and builds.

| Claim | What the paper says | Files | Script | Values |
|---|---|---|---|---|
| D1 | Corrections from approximation theory (Table I.1), the key-mean shift end to end, the fill error's correlation along the prompt, spread exact rows against a tail | `DEV/closures/`, `DEV/whiteness/` | `recompute/dev_history.py` | 33 |
| D2 | Needles in the filled region and exact digit rows on the development checkpoint; digit and attention rows on the reference models (Table I.2) | `DEV/needles-dev/`, `DEV/needles-reference/` | `recompute/dev_history.py` | 41 |
| D3 | Fill error by token class; concentration of the first-order cost and its check | `DEV/token-classes/` | `recompute/dev_history.py` | 8 |
| D4 | Decode-time check on Qwen3.8-27B: full-scan overhead, periodic-check misses | `DEV/certificate-27b/` | `recompute/dev_history.py` | 4 |
| D5 | Needle suites passed where RULER multi-key failed on Qwen3.8-27B | `DEV/needles-vs-ruler-27b/` | `recompute/dev_history.py` | 6 |

## Values of the paper's text and appendices (release 1.3)

Each printed result of the paper that the claims above do not cover. The scripts read the same raw files.

| Claim | What the paper says | Script | Values |
|---|---|---|---|
| M | §1–3: shares, calibration and fit-selection numbers of the method sections | `recompute/` m_fit_selection.py, m_prefix_shares.py, m_reference_population.py | 20 |
| E | §4: 27B row shares and compute, timing offsets, per-item diagnostics, multi-user details | `recompute/` e_timing_shares.py, e_item_diagnostics.py, e_mix_users.py | 47 |
| F | §5, §7–9: engine, calibration and reference-study limits quoted in the text | `recompute/` f_engine_limits.py, f_calibration_limits.py, f_reference_limits.py | 22 |
| T | §6: split curve, arm, fresh-document and RULER values not covered by S7–S12 | `recompute/` t_nll.py, t_ruler.py, t_fresh.py | 127 |
| AB | Appendices A and B: every arm of Tables A.1–A.3 and every contrast of Table B.1 | `recompute/` appendix_a_arms.py, appendix_ab_contrasts.py | 153 |
| C | Appendix C: filled-region RULER counts and arm means by task and length | `recompute/` c_ruler_by_task_length.py | 259 |
| Z | Appendices D–I: second-question details, release provenance counts and other appendix values | `recompute/` z_second_question.py, z_release_provenance.py, z_misc_counts.py | 50 |

## Facts in §9 and the appendices

| Fact | Files | Check |
|---|---|---|
| Calibration rows, documents, λ per split | `P/fits/`, `P/inputs/calibration-docs.tsv` | `recompute/check_fit_records.py` |
| Overlap of calibration text with the test sets | `P/checks/calib-vs-longbench-v2.json`, `P/inputs/calibration-exclude.json` | `tools/overlap_lbv2.py` (needs the texts) |
| Run ledger | `P/ledger/` | `tools/verify_ledger.py` |
| Formal statements | `code/lean/kva_lean/` | `lake build` in that folder |
| Figure 10 | `Q/results/timing/` | `recompute/fig_speedup_gap.py` |

Not claimed (and so not here): an identity-fill arm on Qwen3.8-27B, a matched comparison with token-dropping methods,
any memory saving, and measured TTFT on the three reference models.
