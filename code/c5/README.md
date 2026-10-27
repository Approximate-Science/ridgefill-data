# c5/: the C5 run kit (Flash-Next final tier on tcclaviger's vLLM 29.08.10)

Plan, arms, tail, data, time estimate: `PLAN.md`. Draft FINAL record: `REGISTRATION-DRAFT.md` (not registered).
Work directory: `$C5_DIR` (required; with `C5_MODEL_DIR`, `C5_GPU_LOCK` and `C5_WATCH_PCI`: `run_c5.sh` header).

| file | what |
|---|---|
| `convert.py` | release (int8 maps, correction, rowsel) -> the engine's format in `projector-tcc/` (checked twice, read back, SHA256SUMS) |
| `engine_check.py` | runs in the image on the CPU: his `resolve_bundle`, `Projector`, `StCorrection` (TP1, TP2 ranks), `XrowsSelector` against the release, bit for bit; report `checks/engine_check.json` |
| `randctl.py` | the PROPOSED random-rows substitute table (class-matched, rarity-shuffled); not wired into an arm |
| `client.py` | one arm's tests against a booted server: ppl (prompt logprobs), RULER (NIAH), speed (TTFT, interleaved with the same boot's exact path), positive control; one JSON + .sha256 per sample, resumable |
| `smoke.py` | the smoke test on the graded boot, non-test filler prompts only: determinism, rates, 64K fit |
| `annotate.py` | the real tail and exact-row count of every request from the engine log -> `results/<arm>/tails.json` |
| `run_c5.sh` | the driver: `smoke`, `run [ARM...]`, `extras`, `status`; one boot per arm, watchdog, GPU lock, preflight |
| `watchdog.sh` | kernel-fault watchdog (tools/kcheck.sh + page faults + VRAM), stops the run at the first fault |

Commands (GPU; from a committed checkout of the kit):
```
c5/run_c5.sh smoke          # ~20 min; then re-plan from $C5_DIR/smoke/smoke.json
c5/run_c5.sh run            # mandatory arms, resumable (re-run the same command after a stop)
c5/run_c5.sh extras         # budget-gated
c5/run_c5.sh status
```
CPU checks: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_c5.py`; the engine round trip:
```
docker run --rm --network none -e HIP_VISIBLE_DEVICES= -v $PWD:/kit:ro,z -v $C5_DIR/projector-tcc:/tcc:ro,z \
  -v $C5_DIR/projector-src:/src:ro,z -v <model>:/model:ro,z -v <data/qfn/proj>:/finalsrc:ro,z -v $C5_DIR/checks:/out:z \
  --entrypoint python tcclaviger/vllm:29.08.10 -I /kit/c5/engine_check.py /out/engine_check.json
```

## What the finals tables need (second priority; not built)
`finals/tables.py` reads one `flashnext` cell (FINAL record with declared_share 25, margins {ppl_ratio_upper_pct,
ruler_points_lower, ttft_min_speedup, ttft_length}, split_layer) whose samples follow the finals sample-record contract
(one JSON per unit under `<cell>/samples/`, sidecar, protocol stamp, tier `engine-final`, every record holding `exact`,
split key "-"). A converter from `results/` would write:
- `pg19` (the perplexity set; name kept by tables.py although the documents are all six domains): per document
  `arms.{exact, exact-rerun, ridgefill, ridgefill+corr, ridgefill+corr+prot{10,25,50,100}}["-"].novel_nll`, plus
  `rows_exact_pct` for the protected arms (from `tails.json`: 100 x xrows / tail_start), domain, doc, tokens.
- `ruler`: per row `arms.{exact, ridgefill+corr+prot25, exact-rerun (8K+16K rows), ridgefill (extras)}["-"].score`,
  task, length, `needle_filled` (T 2048).
- `ttft`: units `<length>#round<r>` with `ttft_s` per arm; exact from the same boot's exact path (the speed files'
  `exact-switch` runs) so `ttft_speedup` pairs within a boot.
- `agree` (decode agreement, table column 6): NOT measured by this kit (not in the C5 brief); the column stays
  missing unless the layout drops it or a decode-agreement step is added.
- Rows the layout has that this run cannot fill: identity (engine mode), ridgefill+corr+random25, ridgefill+corr+allrows
  (positive control): tables.py would report them MISSING; they need "not run" handling (like registration.CUT) or the
  substitutes once decided.
