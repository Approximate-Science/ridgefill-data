# E3 vs stock — A100 CUDA scorecard

Status: COMPLETE. All new measurements use pure builds of unmodified upstream 1242be66f and the committed E3 scorer source in RUN-MANIFEST.json. Separate instrumented servers are used only for correctness gates; CHECK-A ties them to the pure builds with identical 128-token outputs and top-10 probabilities on two prompts per arm.

RULER deviations: owner instruct format, study lengths, thinking off. Official answer budgets are unchanged. LongBench uses official CoT with thinking off. KL is KL(stock || E3), over the 512 document tokens after the prompt; bootstrap units are documents. dNLL is E3 minus the engine exact forward, tied to upstream by CHECK-A.

RULER CIs resample paired items within task and average the 13 task means (10,000 resamples, seed 1). KL median/p99 and top-1 agreement pool the equal-length per-position arrays. Timing shows median [min,max] of three distinct warm requests; cold and warm-up are excluded. Mix means average two fresh replicates; maximum ITL is the maximum over both.

| RULER length/task | n | Stock % | E3 % | Delta pp [95% CI] | Length stops S/E |
|---|---:|---:|---:|---:|---:|
| 16384/overall | 260 | 85.69 | 84.51 | -1.19 [-3.13, +0.79] | —/— |
| 16384/cwe | 20 | 64.00 | 54.00 | -10.00 [-32.00, +12.00] | 8/12 |
| 16384/fwe | 20 | 100.00 | 98.33 | -1.67 [-5.00, +0.00] | 0/0 |
| 16384/niah_multikey_1 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 16384/niah_multikey_2 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 16384/niah_multikey_3 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 16384/niah_multiquery | 20 | 100.00 | 98.75 | -1.25 [-3.75, +0.00] | 0/0 |
| 16384/niah_multivalue | 20 | 100.00 | 97.50 | -2.50 [-6.25, +0.00] | 0/0 |
| 16384/niah_single_1 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 16384/niah_single_2 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 16384/niah_single_3 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 16384/qa_1 | 20 | 90.00 | 85.00 | -5.00 [-15.00, +0.00] | 0/0 |
| 16384/qa_2 | 20 | 60.00 | 65.00 | +5.00 [+0.00, +15.00] | 0/0 |
| 16384/vt | 20 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 20/20 |
| 32768/overall | 260 | 87.85 | 86.42 | -1.42 [-3.21, +0.35] | —/— |
| 32768/cwe | 20 | 82.00 | 68.50 | -13.50 [-36.00, +9.00] | 4/7 |
| 32768/fwe | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 32768/niah_multikey_1 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 32768/niah_multikey_2 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 32768/niah_multikey_3 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 32768/niah_multiquery | 20 | 100.00 | 98.75 | -1.25 [-3.75, +0.00] | 0/0 |
| 32768/niah_multivalue | 20 | 100.00 | 96.25 | -3.75 [-10.00, +0.00] | 0/0 |
| 32768/niah_single_1 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 32768/niah_single_2 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 32768/niah_single_3 | 20 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 32768/qa_1 | 20 | 85.00 | 85.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 32768/qa_2 | 20 | 75.00 | 75.00 | +0.00 [+0.00, +0.00] | 0/0 |
| 32768/vt | 20 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 20/20 |

Official vt remains a 30-token-budget cell; if both arms stop in the preamble it carries no retrieval information.

| Prompt tokens | Docs | Mean KL [95% CI] | Median KL | p99 KL | Top-1 agreement | dNLL [95% CI] |
|---:|---:|---:|---:|---:|---:|---:|
| 16384 | 100 | 0.0894135 [0.0398684, 0.155597] | 0.00403776 | 0.965263 | 94.6367% | -0.0329524 [-0.0874699, +0.00588599] |
| 32768 | 100 | 0.26558 [0.0599408, 0.601741] | 0.00423008 | 12.7717 | 93.5957% | -0.214175 [-0.562869, -0.00249486] |

| LongBench group | n | Stock % | E3 % | Delta pp |
|---|---:|---:|---:|---:|
| overall | 503 | 38.57 | 37.18 | -1.39 |
| domain: Code Repository Understanding | 50 | 34.00 | 34.00 | +0.00 |
| domain: Long In-context Learning | 81 | 38.27 | 30.86 | -7.41 |
| domain: Long Structured Data Understanding | 33 | 36.36 | 36.36 | +0.00 |
| domain: Long-dialogue History Understanding | 39 | 48.72 | 41.03 | -7.69 |
| domain: Multi-Document QA | 125 | 30.40 | 34.40 | +4.00 |
| domain: Single-Document QA | 175 | 44.00 | 42.29 | -1.71 |
| difficulty: easy | 192 | 42.71 | 40.62 | -2.08 |
| difficulty: hard | 311 | 36.01 | 35.05 | -0.96 |
| length: long | 108 | 26.85 | 25.93 | -0.93 |
| length: medium | 215 | 38.14 | 36.74 | -1.40 |
| length: short | 180 | 46.11 | 44.44 | -1.67 |

LongBench paired delta 95% CI: [-5.17, +2.19] pp. Extraction/cap diagnostics: `{"e3": {"no_answer": 181, "pass1_length_stops": 302, "pass2_length_stops": 181}, "stock": {"no_answer": 189, "pass1_length_stops": 301, "pass2_length_stops": 189}}`.
LongBench workload prefill speedup: 1.180×; arms ran on different physical GPUs in the completed benchmark.

| Tokens | Arm | Warm prompt s [min,max] | Client TTFT s [min,max] | Peak GiB | Delta stock MiB | Speedup |
|---:|---|---:|---:|---:|---:|---:|
| 16384 | stock | 12.966 [12.930,13.027] | 12.976 [12.940,13.035] | 17.207 | +0 | 1.000× |
| 16384 | resident | 10.380 [10.372,10.454] | 10.388 [10.381,10.463] | 18.762 | +1592 | 1.249× |
| 16384 | streamed | 10.881 [10.785,10.907] | 10.890 [10.795,10.915] | 17.469 | +268 | 1.192× |
| 32768 | stock | 27.488 [27.474,27.512] | 27.502 [27.489,27.527] | 18.223 | +0 | 1.000× |
| 32768 | resident | 21.112 [21.074,21.154] | 21.128 [21.090,21.170] | 19.982 | +1802 | 1.302× |
| 32768 | streamed | 21.689 [21.665,21.713] | 21.709 [21.680,21.729] | 18.689 | +478 | 1.267× |
| 65536 | stock | 60.495 [60.486,60.495] | 60.522 [60.513,60.522] | 20.254 | +0 | 1.000× |
| 65536 | resident | 46.284 [46.059,46.286] | 46.311 [46.087,46.313] | 22.471 | +2270 | 1.307× |
| 65536 | streamed | 46.551 [46.543,46.785] | 46.578 [46.571,46.814] | 21.176 | +944 | 1.300× |
| 120000 | stock | 128.100 [128.096,128.128] | 128.163 [128.144,128.174] | 23.635 | +0 | 1.000× |
| 120000 | resident | 100.014 [99.882,100.322] | 100.065 [99.933,100.388] | 26.578 | +3014 | 1.281× |
| 120000 | streamed | 99.056 [98.892,99.281] | 99.105 [98.959,99.332] | 25.326 | +1732 | 1.293× |

| Mix | Arm | Injector TTFT s | Other tok/s | ITL mean/p99/max ms | Injector tok/s | Peak GiB |
|---|---|---:|---:|---:|---:|---:|
| separate-c2 | stock | 13.476 | 19.910 | 49.379/1592.177/1703.403 | 37.408 | 18.418 |
| separate-c2 | e3-streamed | 11.337 | 21.871 | 44.904/858.324/1310.746 | 37.491 | 18.664 |
| separate-c8 | stock | 13.930 | 88.878 | 69.835/584.683/1821.095 | 16.284 | 25.672 |
| separate-c8 | e3-streamed | 11.862 | 91.981 | 67.459/654.340/1809.552 | 16.388 | 25.916 |
| unified-c1 | stock | 13.063 | — | —/—/— | 42.915 | 17.209 |
| unified-c1 | e3-streamed | 11.109 | — | —/—/— | 42.829 | 17.455 |
| unified-c2 | stock | 13.532 | 19.917 | 49.331/1605.655/1707.842 | 38.226 | 18.436 |
| unified-c2 | e3-streamed | 11.696 | 21.597 | 45.497/862.995/1317.065 | 38.336 | 18.682 |
| unified-c4 | stock | 14.184 | 60.064 | 46.471/148.028/1785.047 | 26.626 | 20.887 |
| unified-c4 | e3-streamed | 12.215 | 62.964 | 44.199/668.950/1323.738 | 26.794 | 21.148 |
| unified-c8 | stock | 15.322 | 86.903 | 71.577/621.739/1919.579 | 16.331 | 25.789 |
| unified-c8 | e3-streamed | 13.532 | 89.476 | 69.347/774.912/1876.143 | 16.456 | 26.090 |
