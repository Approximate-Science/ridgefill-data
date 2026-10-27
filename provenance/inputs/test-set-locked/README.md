# KVA paper: LOCKED test set

**Frozen 2026-10-01.** The manifest hash is in `FROZEN.json`. No model has been run on this data.

## Rules
- **Tuning happens on the dev set only.** Dev set: the 23 graded prompts and the K20 NIAH set, in the HF repo `prompts/`.
- **This set runs once per final configuration** and is never used to choose settings.
- **Any new calibration or fitting data must be checked against this set first** (50-gram overlap on decoded text, as in `finalize.py`).

## Contents (126 long prompts, 32 RULER files, 8 LongBench tasks; ~0.5 GB with the generator code and raw data)
**`long-prompts/`:** 6 domains × 21 documents.
- Per domain: 5 at 8K, 5 at 16K, 5 at 32K, 4 at 64K and 2 at ~128K tokens (Qwen tokenizer).
- Each file is prompt + continuation. The **continuation is the last 512 tokens under the Qwen3.8-Flash-Next tokenizer**.
- The boundary is stored as a character offset in `long-prompts/SPLITS.tsv`, so every model splits at the same text point. That file also gives the prompt/continuation token counts for all three tokenizers.

| Domain (file prefix) | Source | Licence |
|---|---|---|
| prose | PG-19 test (Project Gutenberg) | public domain (US); PG-19 Apache-2.0 |
| papers | arXiv full text | per-paper arXiv licences (mostly no redistribution) |
| chat | UltraChat train_sft, multi-turn | MIT |
| legal | EU legislation text via the LexGLUE loader | CC-BY-4.0 / EUR-Lex reuse with attribution |
| encyclopedia (file prefix `tech_doc`) | English Wikipedia 20231101.en (the builder mislabelled it as tech docs) | CC-BY-SA-4.0 |
| source_code | GitHub: bootstrap, django, react, rust-lang/regex | MIT / BSD-3 / MIT / MIT-or-Apache-2.0 |

**`ruler/`:** NVIDIA RULER, commit in `ruler_commit.txt`. 25 samples per task per length, at 8K/16K/32K/64K, text level.
- Tasks: niah_single_1, niah_multikey_1, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_1.
- **qa_2 was skipped:** its HotpotQA download host timed out.

**`longbench/`:** THUDM/LongBench (`data.zip`, main). 50 rows each.
- Tasks: narrativeqa, qasper, hotpotqa, 2wikimqa, gov_report, trec, lcc, repobench-p.
- Score with the official LongBench metrics for each task.

## Integrity
- `MANIFEST.tsv`: sha256, bytes, domain/task, source, licence and token counts for every file. `FROZEN.json`: the manifest hash, generator commits, dataset revisions and the amendment log.
- **Dev overlap:** 0 shared 50-grams with the dev set and the calibration data (decoded text).
  - The builder rejected 39 candidates.
  - The final check found 12 LongBench code rows (lcc 4, repobench-p 8) overlapping the projector/GDN-correction calibration data. They were replaced with clean rows (see `FROZEN.json`).

## Known gaps
- `build.py` regenerates RULER, LongBench and the manifest. The scripts that sourced the long prompts were not kept, so the frozen files plus their hashes are the reference.
- The `tech_doc` files are encyclopedia articles, not technical documentation. A real technical-docs domain is missing.
- arXiv texts carry per-paper licences: fine for private research; check before any redistribution.
