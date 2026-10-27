# C5 run plan: Flash-Next final tier (2026-10-08)

Claim C5 (Dylan, 2026-10-08), both graded, one-sided non-inferiority vs exact on the engine tier:
perplexity no more than +1% (per-document paired novel-token NLL, upper bound <= log 1.01) AND RULER filled-region
accuracy no worse than -2 points (per-row paired, `finals/stats.py` `noninferiority` with the exact-binomial floor);
speed: prefill speedup vs exact at 16K >= 1.25x (TTFT at 9K / 16K / 32K, 5 timed repeats after 1 warm-up, median).
Graded arm: RidgeFill + GDN correction + class56 at 25% (`class56:0.25`).

## Pins
- Engine `tcclaviger/vllm:29.08.10`, image id sha256:809657ddb1533db10a0c08267c3ee6fde180b839e379713414efa121415914a3,
  RidgeFill built in (`vllm/v1/ced.py`, `vllm/models/qwen4_exp/amd/ced.py`); no overlay. **Not the protocol's pinned image
  (protocol.json engine_image = 29.07.16, sha256:2703275702ab...): a protocol change, Dylan's call.**
- Model `$C5_MODEL_DIR` (tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ
  at 6fe3e7464c18).
- Projector: `TheCacheIsALie/RidgeFill-Projector-Qwen3.8-Flash-Next-i8` at 0b6f2ce017fe912198c697bcc813a1578bc67ce1
  (manifest sha256 795363fa...), converted by `c5/convert.py` to `projector-tcc/` (SHA256SUMS there; checks in
  `checks/engine_check.json`). `final` map: verbatim from the release's own bf16 source (`kva-big-s24.safetensors`,
  sha256 c9db6064... = the manifest's `fit.proj_sha256`), pending the coordinator's OK; never read by a scored number
  (approximated rows' logits are not scored; no MTP).
- Tail T = 2048 (below). Split 24 (the release's). `DYLUHN_KVA_EXACT_TAIL=1` with prefix caching off: the tail is exactly
  N - 2048 rows, no block rounding (the engine's planner, `CedPolicy.plan`); the real tail of every request is read
  back from the engine log (`c5/annotate.py`, `results/<arm>/tails.json`).

## Tail: T = 2048
- The engine's default and the release's declared default (`ridgefill.json` tail.default 2048); the release's quality
  numbers were measured at T 2048.
- RULER filled-region population (needle ends at or before N - T, the 4 NIAH tasks, locked set, Flash-Next tokenizer,
  raw format): 8K-32K gives 234 / 195 / 139 eligible rows at T 1024 / 2048 / 4096. At 4096 the population (139) is
  below the 149 rows the exact-binomial floor needs for a -2 point margin even with zero misses, so T 4096 cannot
  support C5's RULER margin. At 2048, adding niah_single_1 at 64K (25 rows, 24 eligible, declared here before any run)
  gives **219 eligible rows** (>= 200).
- Speed: at 16K, T 2048 approximates 87.5% of the prompt (T 4096: 75%), the setting the 1.25x criterion is tested at.
- Not on the protocol's battery grid (tails 1024, 4096); C5 is an engine FINAL cell with its own declared tail.

## Data (locked set, `datasets/test-set-locked`, manifest checked on every load; no text is opened by hand)
- Perplexity: every locked long-prompt document of nominal length <= 32K (the server's window is set for RULER's 64K
  rows, but prompt-logprob scoring is kept to the 10-04-proven lengths): 90 documents, 6 domains x 15 (8K/16K/32K x 30;
  15 PG-19 prose among them); 1.77 M prompt tokens. Scored: the 512-token continuation, novel tokens primary
  (`kva.data.novel_mask`), all tokens reported. The 36 documents at 64K/128K are listed as skipped.
- RULER: `kva.locked.ruler_prompts`, raw format, niah_single_1 / niah_multikey_1 (32-token cap, protocol 1.1.2) and
  niah_multivalue / niah_multiquery (128) at 8K/16K/32K (300 rows, 5.65 M tokens) + niah_single_1 at 64K (25 rows,
  1.63 M tokens); official scorer (`kva.locked.score`); greedy, eos from the checkpoint, no penalties, the engine's
  degenerate-loop abort off (`--degen-max-period 0`, it is on by default in 29.08.10 and would cut greedy outputs).
- Speed: prefixes of one locked 32K document (the first in path order: `long-prompts/chat_32768_0.txt`) at 9,216 /
  16,384 / 32,768 tokens. On every RidgeFill boot each round runs the arm and the same server's exact path
  (`dyluhn_kva=0`) back to back, alternating which goes first: arms interleaved per round on one boot (protocol
  section 2 Speed v1.1); the exact boot times exact alone as the cross-check.

## Arms (one boot each; RidgeFill settings are server-wide)
| arm | server env | steps | paper name |
|---|---|---|---|
| exact | no DYLUHN_KVA | ppl a, ppl b (same-boot rerun), RULER 8-32K, RULER 64K, RULER 8K+16K rerun, speed | exact, exact-rerun |
| graded | ST=ridgefill-s24-st, XROWS=class56:0.25 | ppl, RULER 8-32K, RULER 64K, speed, positive control | ridgefill+corr+prot25 |
| ridgefill-corr | ST=ridgefill-s24-st, XROWS=0 | ppl, speed | ridgefill+corr |
| ridgefill | ST=0, XROWS=0 | ppl, speed | ridgefill |
| prot10 / prot50 / prot100 | ST=..., XROWS=class56:0.1 / 0.5 / 1.0 | ppl, speed | knob curve |
| extras (budget-gated) | ridgefill boot again | RULER 8-32K + 64K | reported comparison |
Common: DYLUHN_KVA_TAIL=2048, EXACT_TAIL=1, PROJECTOR_IMPL=triton (the engine default), ST_ALPHA=1.0. Every RidgeFill
request sends `vllm_xargs {dyluhn_kva: 1, dyluhn_kva_tail: 2048}`: the engine skips RidgeFill on prompt-logprob requests
and on prompts under 8,192 tokens unless asked (`CedPolicy.plan`), and 8K RULER rows are under 8,192 tokens.
- **Identity (T4.5): not built.** On Flash-Next the identity fill is each layer's own hyper-connection read of the
  boundary (`qfn/engine.py` hc_identity: per-stream RMSNorm + sigmoid gate, nonlinear); the engine's `layer.L` is a
  fixed linear map, and it has no noproj mode. Reported to the coordinator; no substitute unless chosen.
- **Positive control (all rows exact):** not expressible (class56 keeps class tokens only; a tail >= N disables the
  plan). Substitute run on the graded boot: 6 documents with `dyluhn_kva=0` (no row approximated, projector, correction
  and selector loaded); must match the exact boot within the exact-vs-exact noise floor. It checks that the loaded
  RidgeFill machinery leaves the exact path alone; it does not exercise the fill.
- **Random-25%-rows control: no engine rule.** Proposal (not decided): (a) not run on the engine tier, said in the paper,
  the reference tier's count-matched random controls standing for the question; or (b) a labelled substitute,
  "class-matched, rarity-shuffled": the engine's class56 rule with a unigram table whose values are a seeded permutation
  over the eligible ids (`c5/randctl.py`, built: `projector-randctl/freq-shuffled-20261008.safetensors`, sha256
  f92564c6...). It keeps the same count of the same classes, chosen by a random order of token identities instead of by
  rarity; it is not position-random and not drawn from all rows. Cost if chosen: one boot, ppl + RULER, about 2.4 h.

## Time estimate (from the 10-04 run: exact ~1,000 tok/s; RidgeFill ~1,430 tok/s at 9K, ~1,700 at 16K-32K; boots
healthy in 190-220 s then, plus the PLE table rebuild 29.08.10 does on every boot (no --ple-cache-reuse), unknown;
exact-row arms assumed slower: graded ~1,400, prot10 ~1,550, prot50 ~1,250, prot100 ~1,100 tok/s; prompt-logprob
passes +10%; decoding ~25 tok/s. Re-plan from the smoke test's measured rates before MODE run.)
| arm | boot+warm+stop | ppl | RULER | speed | other | total |
|---|---|---|---|---|---|---|
| exact | 10 | 32 + 32 (rerun) | 110 (8-32K) + 30 (64K) + 52 (8K+16K rerun) | 6 | | **272 min** |
| graded | 10 | 23 | 83 + 20 | 10 | 2 (control) | **148 min** |
| ridgefill-corr | 10 | 19 | - | 9 | | **38 min** |
| ridgefill | 10 | 19 | - | 9 | | **38 min** |
| prot10 | 10 | 21 | - | 10 | | **41 min** |
| prot50 | 10 | 26 | - | 10 | | **46 min** |
| prot100 | 10 | 30 | - | 10 | | **50 min** |
| **mandatory total** | | | | | | **633 min = 10.6 h** |
| extras: ridgefill RULER | 10 | | 100 | | | 110 min |
Over the ~10 h budget by ~35 min at these rates. Declared cuts, in this order, applied at the re-plan if the smoke's
rates do not bring the total under 10 h: (1) extras not run; (2) the knob arms' perplexity on the 60 documents <= 16K
(`C5_KNOB_PPL_MAXLEN=16384`, saves ~35 min); (3) prot10 and prot50 not run (the knob curve keeps 25 and 100). RULER
rows are never cut; exact and graded run every row.

## Order and gates
1. CPU (done): conversion, engine round-trip (`c5/engine_check.py` in the image), tests (`tests/test_c5.py`).
2. Coordinator: the display moves off the second card; coordinator registers the FINAL record
   (`c5/REGISTRATION-DRAFT.md`) and the protocol change for 29.08.10 if Dylan agrees; the kit is committed.
3. `c5/run_c5.sh smoke` (~20 min incl. boot): graded boot, non-test filler prompts; reports determinism (the same
   request twice, both paths: bitwise or not), the rates, and that a 65,408-token request fits. If the boot fails at
   max-model-len 66560: `C5_MAX_LEN=40960` drops the 64K RULER rows (eligible 195: above the 149 floor, below the
   200 target).
4. Re-plan from smoke.json (apply the cuts above if needed), then `c5/run_c5.sh run` (exact, graded, ridgefill-corr,
   ridgefill, prot10, prot50, prot100), then `c5/run_c5.sh extras` if time is left.
- Stops: the watchdog kills the run at the first MES / SMU / ring / reset / page-fault line (`tools/kcheck.sh`) and
  writes `safety/KERNEL-ALERT`; every boot refuses while it exists, while another container runs, with kernel fault
  lines since the last boot, with a dirty tree, a projector differing from its SHA256SUMS, MemAvailable < 61 GiB, or
  < 60 GiB free for the PLE table. The whole run holds the machine's GPU lock file (`C5_GPU_LOCK`).
- Determinism (protocol.json engine_tier.final_requires the deterministic top-k fix): not checkable on the CPU (the QSA
  selection is a compiled op). Read from the smoke (same request twice) and from exact ppl a vs b; if they differ,
  the run is pre-fix by the protocol's rule and its numbers carry `engine_tier.prefix_label` and support no claim.

## Settings carried from the 10-04 run (`kva-flashnext-iterate/final`, `qfn/server.py`)
eager TP2, `--gpu-memory-utilization 0.72`, `--max-num-seqs 1`, `--max-num-batched-tokens 2048`, `--kv-cache-dtype
fp8`, `--no-enable-prefix-caching`, `--language-model-only`, expert offload 35 GiB, PLE on NVMe with a 4 GiB row cache,
`R4D_AR_QUANT=0` (exact all-reduce), `CLAV_ATTN_AUTOTUNE=0`, SDMA-era env (GPU_MAX_HW_QUEUES=1, HSA_* as before).
- FIX-1 (idle warm-up of the prompt-logprob path): the overlay's RPC is gone; a client-side analogue (one
  prompt-logprob request and one generation on non-test ids, both paths, on the idle server) runs after every boot.
- FIX-2 (THP off in the engine processes): `prctl(PR_SET_THP_DISABLE)` in a two-line entrypoint wrapper before the
  image's own entrypoint (inherited across fork and exec); no engine code touched.
- FIX-3 (capture fsync / cache drop): no captures here; n/a.
- FIX-4: container 108g hard / 100G soft (memory.high), expert 35 GiB + PLE 4 GiB, util 0.72, the model and projector
  purged from the page cache before boot and kept out until healthy.
- Not carried: `--ple-cache-reuse` (removed in 29.08.10: the PLE table is rebuilt on every boot, into
  `ridgefill-c5-20261008/pleoffload`, ~48 GB).
