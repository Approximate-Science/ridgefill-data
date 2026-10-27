# E3 vs stock CUDA paper measurement kit

## Identity-fill extension (CPU prepared, GPU opt-in)

`identity` uses the **pinned pure E3 source and a mandatory reproduction-gated rebuild**, with the same server/scorer options, 4096 exact tail, S1, lexical net, special tokens, qattn, vtail and trim. Its projector differs. The packed `models/identity-projector-qwen3.8-27b-i8-S16.gguf` is derived from the real i8 projector; the `i8` in its filename identifies that source, **its 12 required attention maps are exact F16 identities**. The 36 GDN maps are omitted to invoke native `x = h_S`; the mandatory additive state-correction tensors are zero. Every GGUF metadata value and the lexical/class row-score tensor are preserved. `validation/identity-projector.json` and `models.sha256` bind the artifact.

Two limits prevent claiming a strictly matched-row, matched-speed ablation. **S1 selection depends on the filled KV cache and therefore on the projector**; the CPU comparison records actual row overlap. Freezing E3's selected rows would require a different protocol or engine support. Also, attention maps cannot be omitted: vtail asserts they exist, and the loader requires them. Exact Q8 identity weights still round activations in the CPU matmul (native maximum error .00390625); F16 identities preserve E3's saved F16 `h_S` exactly. Keep `--h-type f16`. Actual speed must be measured. The unchanged loader logs native identity for 36 GDN layers and `12 maps of type f16` for the exact attention identities; it cannot print “identity” for all 48 layers or “0 maps” without a source change. No learned fill maps remain. Full CPU outputs and limitations are in NOTES.md.

The original box and its binaries/source trees no longer exist. On a fresh two- or four-A100 box, transfer this kit and restore only the completed `results-20261025/results` directory. `setup.sh --identity-only` follows the normal build/download path: pure upstream stock and pure E3 from the pinned archives, CUDA sm_80, stock tokenizer/exact helper and E3 scorer. REPRODUCTION uses pure binaries, so this mode skips the two instrumented builds; normal setup still builds them for CHECK-A/B. Preflight verifies every pure source file against its pinned archive and build manifest, preserves the original data checks, hashes all 1,254 reused result/provenance files, and binds the **new** binary/library hashes in `IDENTITY-MANIFEST.json`. It retains the old binary hashes for provenance without requiring equality. GPU model/memory must match the earlier run; N may be 2 or 4. The requested `gpu_count=N` and all UUIDs are recorded in the identity manifest. `--gpus N` is passed through setup, preflight, REPRODUCTION and every sharded step; default N is 4. The old run manifest, measurements, tokenizer gate and scorecard remain intact.

Exact command sequence on a fresh CUDA development box, after transferring the kit and completed results to these paths:

```bash
cd /workspace/kit
set -o pipefail
bash setup.sh --identity-only --gpus 2 /workspace/kit/results-20261025/results 2>&1 | tee setup-identity.log
bash run_all.sh --identity-only --gpus 2 /workspace/kit/results-20261025/results 2>&1 | tee run-identity.log
```

Before any identity measurement, `run_all.sh` must pass **REPRODUCTION** on N disjoint GPU shards. The parent checks idle GPUs before launch; concurrent reproduction workers may use GPUs 0..N-1. The frozen `data/reproduction-plan.json` selects the first item in cwe / niah_multikey_1 / niah_multivalue / qa_1 at each length (eight items total), replayed with both the new stock and E3 servers. All 16 response texts must be byte-identical to their stored arm, with identical requests and finish reasons. The gate also reruns four distinct stored KL documents: the first two ids at 16K, then the first two unused ids at 32K using the new E3 scorer: regenerate off-mode stock distributions, then E3 with --kl-against. Every scientific scorer field is compared exactly, including NLL, KL arrays/statistics, logprobs, logsumexp, selected rows and vtail scales. Only build metadata and wall time are excluded. There is no tolerance or configuration fallback. A mismatch or engine error writes `identity-reproduction/FAIL.json` and stops; owner review is required. Success writes an atomic, hash-bound `identity-reproduction/PASS.json`, also checked by the individual identity runners and final report. Saved gate items resume without repeating valid forwards. Full commands, outputs and provenance remain under `identity-reproduction/`.

After REPRODUCTION, the default identity-only run includes **all measurements, in order: RULER → KL → TTFT → LongBench**, followed by the report. This adds 520 RULER records under `ruler/identity`, 200 KL records under `kld-identity`, 16 new timing results under `timing-identity/L/ARM`, 503 records under `longbench/identity`, and `SCORECARD-IDENTITY.{md,json}`. RULER request bodies are identical to E3's. KL regenerates stock per document with `--save-logprobs`, checks its target logprobs against the original stock result, runs **identity only** with `--kl-against`, and deletes the reference immediately (also on failure). The full E3 quality run is reused; only the reproduction samples are rerun. At most N float32 references coexist (~1 GB on two GPUs, ~2 GB on four). TTFT reruns **stock, E3 resident, E3 streamed and identity resident** at each of 16384/32768/65536/120000, with cold + full warmup + three timed prompts, all on physical GPU 0 while GPUs 1..N-1 are idle. Every arm uses a fresh server. A shared `timing-identity/SESSION.json` binds the new results to one identity manifest and GPU UUID; old `timing/` results are preserved. The identity scorecard uses only this new same-box timing table and pairs identity with the new stock/E3 resident prompts. The original CUDA gates remain attached to the baseline; REPRODUCTION ties the new pure builds to it. Mix is reused. Reports add paired identity−stock and identity−E3 CIs; KL-to-stock differences are distinguished from a direct E3-vs-identity KL, which is unavailable without E3 distributions.

`longbench_identity.py` is the final measurement step and reuses the unchanged vendored `cloud-longbench/run_lb.py` two-pass request/scoring logic, official CoT templates, seeds, 1024/128 answer budgets, the exact frozen 503 items prepared at `max_len=120000`, and server `-c 121101`. Completed stock/E3 LongBench scores are reused. The former `--longbench-identity` flag remains accepted for existing command sequences; LongBench now runs with or without it. A complete identity report requires all 503 identity LongBench results.

`identity_run.py` writes atomic, fsynced completion markers only after the corresponding command exits successfully and every expected result is present and readable:

- `identity-steps/01-ruler.complete.json`: 520 items.
- `identity-steps/02-kl.complete.json`: 200 documents.
- `identity-steps/03-ttft.complete.json`: 16 arm/length results plus their shared session record (17 files).
- `identity-steps/04-longbench.complete.json`: 503 items.
- `identity-steps/05-report.complete.json`: both final scorecard files.

Each marker binds the identity manifest and SHA256 of every completed result. Rerunning `run_all.sh` verifies the saved REPRODUCTION gate and measurement markers and skips finished steps; a missing marker resumes through the existing per-item checkpoints. Ctrl-C or SIGTERM stops the active child and its workers; a box shutdown leaves already fsynced item files intact. Copy the results directory home, including `IDENTITY-MANIFEST.json`, `identity-reproduction/`, `identity-steps/`, item files and provenance/logs, before releasing the box. If a shutdown leaves temporary `reference.logprobs*` files in KL attempts, those payloads need not be transferred: unfinished KL pairs regenerate them. A partial transfer must include every file named in a copied completion marker. Preserve the **newly rebuilt** build/source artifacts for future identity resume; after the identity manifest is created, its binary hashes remain immutable within that run. Setup requires fresh source directories; resume an existing build with `run_all.sh`, not by rerunning setup.

An interrupted RULER or LongBench step skips completed items on resume; an unfinished LongBench item repeats its two passes. KL repeats only unfinished document pairs. TTFT repeats only an unfinished arm/length's cold/warm-up/three-prompt protocol. To inspect a partial scorecard on CPU, run `python3 report.py --identity-only --allow-partial --out RESULTS`; it stays explicitly incomplete until all required measurements finish.

CPU reproduction (no GPU, no engine source changes):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/llama.cpp-ridgefill/gguf-py \
  python3 cloud-kit/identity_projector.py --source /path/to/original-projector.gguf
python3 cloud-kit/identity_cpu.py --fork /path/to/llama.cpp-ridgefill \
  --model /path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf
python3 -m unittest discover -s cloud-kit/tests -v
```

The GGUF script requires the fork's gguf-py and NumPy only when rebuilding/verifying the artifact. Normal identity setup uses its SHA256 and requires no added Python dependencies. The saved native operator and hidden-state checks link the unchanged fork libraries; their source and build command are packed.

| Identity-only step | Active A100 time | N=2 wall | N=4 wall |
|---|---:|---:|---:|
| Fresh setup | no measured inference | 15–30 min plus download | 15–30 min plus download |
| REPRODUCTION, N shards | ≈0.20h (N=2), ≈0.23h (N=4) | ≈6 min | ≈4 min |
| RULER, 520 identity requests | 2.6h | 1.3h | 0.65h |
| KL, 200 stock-save + identity pairs | 3.6h | 1.8h | 0.9h |
| Same-box TTFT, four arms × four lengths | 1.2–1.27h | 72–76 min | 72–76 min |
| LongBench, 503 identity items, last | 14h | 7h | 3.5h |

For **N=2**, plan **11.4–11.5 box-hours**, **21.6–21.7 active A100 GPU-hours**, or **22.8–23.0 allocated GPU-hours** including GPU 1 idle during TTFT. Add setup/download/copy time. For N=4, plan roughly 6.3–6.4 box-hours; the same-box timing step remains serial. The old $2.51/hour quote was for four GPUs and is not assumed to price the new two-GPU box.

TTFT estimate uses the completed per-arm request sums: stock 1145.62s, resident 887.60s, streamed 890.76s, identity estimated as resident 887.60s, plus 16 fresh loads ×30s =4291.58s (71.5min). The audit's more conservative 58-minute original three-arm estimate plus 18 minutes identity gives 76min. Reproduction uses stored sample forwards (304.38s RULER +290.57s KL), plus 2N server loads; at N=2 that is ≈715 GPU-s across two shards, about six minutes wall. Rates remain estimates, not measured identity GPU results. The RULER 1.5× rate-stop remains active; checkpoints preserve finished items if the box must be stopped.

The legacy full stock/E3 workflow below describes the original four-GPU protocol. The identity-only sequence above supports N=2 or N=4.

Run on **one 4× NVIDIA A100 40 GB box** (PCIe or SXM4; all four identical, the model is recorded), CUDA sm_80, with the kit at `/workspace/kit`. All GPU jobs are opt-in shell commands below. Preparing this kit and its CPU gate requires no GPU.

Stock is **unmodified upstream 1242be66f**, without a projector. E3 is **b70b0c5b4847d51f02421ce5895e873d8930efe2**, b0439f7bb plus exactly one committed scorer extension. The server and runtime (`src`, `ggml`, `common`, `tools/server`) are identical to b0439f7bb; the empty runtime diff is verified and recorded in `sources/LOCK.json`. Each arm has two independent source/build trees:

| Variant | Source / binaries | Use |
|---|---|---|
| Pure | `src/pure/{stock,e3}` / `build/pure/{stock,e3}/bin` | Every RULER, KL, timing and mix measurement; scorer NLL gates and tokenization |
| Instrumented | `src/instrumented/{stock,e3}` / `build/instrumented/{stock,e3}/bin` | Only `checks.py`; adds the identical raw float32 logit observation hook |

Pure source files must match every byte of the pinned source archive, with no added source files. Instrumented sources must match the archive plus exactly the expected observation hook. `build_manifest.py` verifies this after compilation and binds hashes of every executable and shared library. Runners reject the wrong variant, arm or changed artifact, and preflight records all four build manifests. Each server's library directory is pinned to its own variant. The independent `exact-score` helper links **only the pure stock library**. An inherited observation environment variable is cleared before launching any process.

The source pack is finalized and ready. Its committed source archives and hashes are pinned in `sources/LOCK.json`. To reproduce the E3 archive locally:

```bash
python3 cloud-kit/finalize_sources.py --repo PATH_TO_FORK --commit b70b0c5b4847d51f02421ce5895e873d8930efe2
```

That CPU-only command requires exactly one descendant commit of b0439f7bb, allows only scorer/tests/docs changes (including the worker's NOTES-A.md), verifies an empty server/runtime diff, creates `sources/e3.tar.gz` with `git archive`, and writes `sources/LOCK.json`. Do not archive an uncommitted worktree or replace the scorer build on the box. The scorer takes one document per invocation; the KL runner already invokes it separately for each document and arm.

`FILES.txt` is the complete transfer file list. Root `SHA256SUMS` covers every listed file except itself, including the two source archives, packed projector, all data, harness, tests, and CPU validation evidence. `data/SHA256SUMS` separately covers every packed data file. Local build scratch and generated outputs are excluded. After any deliberate kit edit, regenerate the transfer list with `python3 cloud-kit/release_manifest.py` and verify from the kit directory with `sha256sum -c SHA256SUMS`.

## Commands on the rented box

Use a CUDA development image with `nvcc` installed, at least 64 GB host RAM and 100 GB free disk. Transfer the entire kit directory (including the packed projector) to `/workspace/kit`. Transfer/renting is outside these scripts.

```bash
cd /workspace/kit
bash setup.sh 2>&1 | tee setup.log
bash run_all.sh /workspace/kit/results 2>&1 | tee run-all.log
```

`setup.sh` verifies packed data/source hashes, builds both pure and instrumented variants of each arm with identical CUDA architecture 80 settings, verifies each source/build manifest, builds the pure-stock exact-forward helper, downloads only the model from `unsloth/Qwen3.8-27B-GGUF` at revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`, verifies model/projector SHA256, runs CPU tests and the saved-response gate, and compares CPU-tokenizer probes with the stock server's `/tokenize`. It requires a fresh `src/` directory. Install/build/download failures stop setup.

`run_all.sh` verifies the four identical GPU models, binary/library/model/data hashes and immutable run manifest. It then executes, in order:

1. CHECK-A and CHECK-B on GPU 0. Every mismatch or engine error stops the run. CHECK-A first ties each pure measurement server to its instrumented counterpart: stock and active E3 resident must generate exactly the same 128 tokens and identical per-position top-10 probabilities on the two frozen 16K/32K cold prompts, with temperature 0, `n_probs=10` and `post_sampling_probs=false`. The observation hook is enabled during the instrumented comparison. CHECK-A then compares raw dense logits on 8K/16K/32K prompts and two actual LongBench prompts near 32K/120K, using only instrumented servers for stock, fork without projector, and projector-loaded fork with RidgeFill off per request. Finally, the pure E3 scorer in off mode must match all 512 target logprobs/NLL from the helper linked to pure upstream on two documents. CHECK-B compares instrumented resident/streamed 128-token outputs and raw logits, plus pure-scorer target logprobs/NLL, at 16K/65K.
2. RULER on four disjoint GPU shards, followed by KL on four shards. Both arms of an item use the same physical GPU. Worker failure terminates all four workers and their server process groups.
3. TIMING-WARM, then the fixed-arrival concurrency mix, on GPU 0. GPUs 1–3 must have no compute processes. Each arm uses a fresh server.
4. `report.py` writes `SCORECARD.md` and `SCORECARD.json`. Missing measurements or gates make the default report fail.

Individual commands, useful for resuming after an inspected failure:

```bash
python3 preflight.py --out /workspace/kit/results
python3 checks.py --gpu 0 --out /workspace/kit/results
python3 drive.py ruler_replay.py /workspace/kit/results
python3 drive.py kld_run.py /workspace/kit/results
python3 timing.py --gpu 0 --out /workspace/kit/results
python3 mix.py --gpu 0 --out /workspace/kit/results
python3 report.py --out /workspace/kit/results
```

Completed item JSON files are atomically renamed and fsynced; reruns skip them. Interrupted KL pairs rerun both forwards, and interrupted timing/mix arms restart on a fresh server. Intermediate engine outputs live in separate attempt folders. Run manifests reject changes to binaries, libraries, model, projector, hardware, harness or data. Keep the full results directory, including attempts, server logs and provenance. A rate above 1.5× the audit estimate in the first 10% of any RULER shard writes `RATE-STOP.json` and stops all workers. Report the revised cost to the owner before removing that marker and resuming. Gate failures require the owner's decision; no configuration fallback is automatic.

## Frozen protocols and data

- RULER: exactly **13 tasks × 20 items × 2 lengths = 520 items**, original instruct messages, temperature 0, top_k 1, seed 0, cache off and thinking off. Budgets: niah 128, vt 30, cwe 120, fwe 50, qa 32. Stock and E3 canonical request bodies are asserted equal after removing only RidgeFill fields. Both servers use `-c 33792`. The original validate-harness scorer is vendored unchanged. `ruler-saved.jsonl` is for the CPU regression gate only; it never enters the new CUDA scores.
- KL: first 100 eligible LongBench ids at each L∈{16384,32768}; context text only, extracted verbatim between the frozen prompt's `<text>` and final `</text>` delimiters. No question, instruction, chat template or added BOS enters a window. Tokenization uses the pinned model vocabulary, with `parse_special=true` (the server and `llama-tokenize` defaults). Each packed token-id array is the first **L+512** document tokens; targets are zero-based positions **L through L+511**. The selector endpoint is `known` (L), so selection excludes targets. The scorer runs `--mode off --save-logprobs TMP` then E3 with `--kl-against TMP`; it deletes TMP after each pair, including on failure. All E3 settings match the frozen `cloud-longbench/run_lb.py` config; the projector is resident. Output reports KL(stock || E3), pooled median/p99, top-1 agreement, dNLL and document-bootstrap CIs. The native distribution file is treated as the scorer's opaque format.
- KL temporary storage: the dtype-tagged payload is **float32**, about **0.5 GB per document** at `--scored 512` (512 × 248320 × 4 = 508559360 payload bytes, plus header). The runner deletes it immediately after E3 comparison, also cleaning up on failure. The four-GPU driver runs one worker per GPU, with document pairs processed serially: at most four saved references, about **2 GB total**. Do not launch additional KL workers on occupied GPUs.
- Timing: saved 16K/32K prompts are copied exactly. For 65K/120K, concatenate LongBench **document token arrays in ascending item id, with no separators**, take the first L tokens, and apply the original TIMING-WARM seed-0 shuffle: shuffle 512-token prefix blocks, keep the final 4096 tokens fixed, and require five distinct leading tokens. The final partial prefix block at 120000 (192 tokens) participates in the same shuffle. This extension is necessary because 120000 is not a multiple of 512 and the frozen LongBench documents individually contain fewer than 120000 tokens. Source ids and hashes are packed alongside the prompts. Each fresh arm runs cold, one full-length warm-up, then three timed distinct prompts, one generated token per request. Responses must show zero prompt-cache reuse; server log and response prompt-eval times must agree within .02 ms. Client TTFT is measured from the first token-bearing SSE event. Resident/streamed outputs must agree.
- Mix: exact saved token arrays, arrivals, request lengths, generation counts and arm/repetition order from the six final specs. Unified KV C1/C2/C4/C8; separate KV C2/C8; two fresh runs per arm per spec. E3 uses the **streamed** projector; background requests use RidgeFill off. No baseline reuse. Stream counts, complete prompt evaluation, arrival lateness and background overlap are validated by the ported original harness logic.
- Every server/scorer process records `nvidia-smi -q`, full command, binary/shared-library hashes, model/projector hashes and packed-data manifest hash. Per-process VRAM is sampled every .25 s with `nvidia-smi --query-compute-apps`; GPU index is pinned by `CUDA_VISIBLE_DEVICES`.
- CIs: 10,000 seeded bootstrap resamples. RULER resamples paired items within each task and averages task means; KL and dNLL resample documents. The scorecard includes per-task RULER CIs and stop counts, and the already completed 503-item LongBench result with domain/difficulty/length splits and missing-answer counts. Only new CUDA outputs populate RULER/KL/timing/mix tables.

Every packed file is listed in `data/SHA256SUMS`; `data/PROVENANCE.json` records tokenizer/model/input hashes and extraction rules. Repacking uses explicit local paths:

```bash
python3 cloud-kit/pack_data.py --validate PATH_TO_VALIDATE --items cloud-longbench/items.jsonl \
  --audit-rows audit-scratch/ruler_rows.json --tokenizer PATH_TO_CPU_LLAMA_TOKENIZE --model PATH_TO_MODEL
python3 cloud-kit/ruler_replay.py --cpu-gate
python3 -m unittest discover -s cloud-kit/tests -v
python3 -m py_compile cloud-kit/*.py cloud-kit/vendor/*.py cloud-kit/tests/*.py
```

## Estimated GPU time

These are planning estimates from AUDIT.md §4, using measured A100 prefill rates (stock/E3 ms per token: ~16K .809/.682; ~28K .838/.693; ~58K .916/.773; ~116K 1.065/.904) and 36.3 decode tokens/s. KL distribution I/O has not been measured on this box.

| Step | GPU time | Placement / expected wall |
|---|---:|---|
| Build and model download | no inference | about 30–60 min (four builds) |
| CHECK-A | about 18 min | GPU 0; audit 13 min plus about 5 min for pure/instrumented comparisons |
| CHECK-B | 5 min | GPU 0 |
| RULER | 5.7 A100-h | four GPUs, about 1.4 h |
| Revised 100+100-document KL | about 2.5 A100-h | four GPUs, about .6 h |
| TIMING-WARM | 58 min | GPU 0, others idle |
| Mix | 25 min | GPU 0, others idle |
| Report | CPU only | seconds to minutes |

KL estimate: 100×(16.0+13.9 s) + 100×(32.1+27.3 s) ≈ 8930 GPU-s, using the audit's scorer overhead plus prefill rates. The audit baseline was about 9.9 A100-h / 4.2 h elapsed. The added pure/instrumented comparisons cost approximately 4 model loads × 30 s + 2 × (13.3 + 27.5 + 11.2 + 22.7) s prefill + 8 × 128/36.3 s decode ≈ 5 min. Revised total ≈ **10.0 A100-h / 4.3–4.8 h elapsed**, including the extra builds; approximately $11–12 at the audit's $2.512/box-hour rate. A failed gate stops before quality runs. GPU measurements, CUDA builds and on-box time estimates are not validated by this CPU-only preparation.

Local validation scratch under `.cpu-build-check/`, `__pycache__/` and `cpu-preview/` is not needed on the box and may be excluded from transfer. The scratch contains the CPU-only build and its local test inputs; none of its scripts run in the on-box command sequence.
