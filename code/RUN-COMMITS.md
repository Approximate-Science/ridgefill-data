# Run commits

`code/` is the reference implementation at tag `ridgefill-v1.0` (commit `114de6bf8211d41ae70b47a29ade5e737f4d40e7`),
exported without history over the paths in `release/PUBLIC-PATHS.txt` of that commit. The runs themselves used the
earlier commits below. Each cell's `host.json` records its commit; each map's safetensors metadata and fit record
record the fit commit.

| Commit | Date | What it produced |
|---|---|---|
| `42028dfc3e04cd1b46bcf231d37b35f200b8bd64` | 2026-10-07 | the first loss runs, decode agreement and second-question runs on all three models, and Qwen3-8B's RULER: cells `llama-3.1-8b-instruct-{nll,decode,secondq}`, `qwen3.5-9b-{nll,decode,secondq}`, `qwen3-8b-{nll,decode,secondq,ruler}` |
| `253f845ae1684a62da3fa70c3c3804aa6c21f120` | 2026-10-07 | Llama-3.1-8B's RULER and the second loss runs on Llama-3.1-8B and Qwen3.5-9B: `llama-3.1-8b-instruct-ruler`, `llama-3.1-8b-instruct-nll-causal`, `qwen3.5-9b-nll-causal` |
| `21ccd24402c29deeeb80de9a5a8828a6439cc016` | 2026-10-07 | Qwen3.5-9B's RULER, Qwen3-8B's second loss run and its 5/8-depth comparison with POP: `qwen3.5-9b-ruler`, `qwen3-8b-nll-causal`, `qwen3-8b-ruler-deep` |
| `019ae056229362276e0af4db62d1b2ba771f8860` | 2026-10-08 | Llama-3.1-8B's LongBench v1, the fresh documents and Llama-3.1-8B's 5/8-depth comparison with POP: `llama-3.1-8b-instruct-longbench`, `llama-3.1-8b-instruct-nll-fresh`, `llama-3.1-8b-instruct-ruler-deep` |
| `d1164746aa21b3f339db6d8bd2919601db450d7f` | 2026-10-10 | the Flash-Next run kit (`code/c5/`) that ran `results/flashnext/final-det-20261010/` |
| `4cb057d8a2d099be0e3819be7acca6a688847cbb` | 2026-10-06 | the Llama-3.1-8B and Qwen3.5-9B map fits (all four splits) |
| `cba2d910cb02b6b0c086eb1115b98b556aad5651` | 2026-10-04 | the Qwen3-8B map fits (splits 9, 13, 18, 22) |

## Rebuilding a run's code

`run-commit-diffs/<commit>.diff` is `git diff <commit> ridgefill-v1.0` over the exported paths. Reverse-applying it
to a copy of `code/` gives that commit's version of those files:

```bash
cp -r code /tmp/code-42028dfc && cd /tmp/code-42028dfc
git apply -R ../path/to/code/run-commit-diffs/42028dfc.diff
```

The old commits held a few literals that the public tag removed (machine paths, device and GPU ids, host names,
internal document names, push targets). On removed lines of each diff those literals are replaced by
`<REDACTED:RULE>`, so the rebuilt files carry the placeholder where the run had the literal. Six comment lines that
quoted chat instructions (in `c5/kfilter.sh`, `c5/run_c5.sh` and `kva/select.py`) are reworded in the export and in
the diffs, so the rebuilt files carry the reworded comments too. Where a placeholder
sits in code rather than in a string or comment, the rebuilt file does not parse until a value is put back: this is
the case for d1164746's `c5/run_c5.sh` and `tests/test_tail_round.py`. Nothing else differs.
Files outside the exported paths (datasets, results, development scripts) are not part of the diffs.
