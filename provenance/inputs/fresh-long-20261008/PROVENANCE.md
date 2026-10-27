# fresh-long-20261008: provenance

Built by `scripts/build_fresh_long.py` (seed 20261008; `--check` rebuilds every file byte for byte). A fresh long-document NLL set for confirming post-hoc findings on data that neither shaped them nor overlaps anything already used. It mirrors `datasets/test-set-locked/long-prompts/` (same six domains and sources, same construction and split rule) with new documents: one per domain per nominal length 8192, 16384, 32768, 65536, 128000 = 30 documents. No model has been run on it.

## Sources (Hugging Face datasets, pinned revisions)

| domain | HF dataset @ revision | shard(s) read | source label (MANIFEST source_url) | licence |
|---|---|---|---|---|
| chat | `HuggingFaceH4/ultrachat_200k` @ `8049631c405ae6576f93f445c6b8166f76f5505a` | `data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet` | UltraChat (train_sft), multi-turn dialogues concatenated | MIT (UltraChat) |
| encyclopedia | `wikimedia/wikipedia` @ `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | `20231101.en/train-00021-of-00041.parquet` | English Wikipedia 20231101.en (wikimedia/wikipedia) | CC-BY-SA-4.0 |
| legal | `coastalcph/lex_glue` @ `c23fdff1a6bf74e0e1a71cb86f1e781d37da888c` | `eurlex/train-00000-of-00001.parquet` | EU legislation text via the LexGLUE loader (eurlex, train) | LexGLUE CC-BY-4.0; EUR-Lex reuse with attribution |
| papers | `ccdv/arxiv-summarization` @ `240aaf1a969b3f8cd0ade6986bfad0cd730ee288` | `document/train-00000-of-00015.parquet` | arXiv full text (HF ccdv/arxiv-summarization, document, train) | Per-paper arXiv licences; most are arXiv non-exclusive (no redistribution) |
| prose | `emozilla/pg19-test` @ `c5e39bf32e33f9111323aa68d7d9000d22722035` | `data/test-00000-of-00001-29a571947c0b5ccc.parquet` | PG-19 test split (Project Gutenberg books; HF mirror emozilla/pg19-test) | Public domain (US) texts; PG-19 dataset Apache-2.0 |
| source_code | `codeparrot/github-code-clean` @ `c48d40f9e70f0196f8236901ee35807f7d6c44c0` | `data/train-00311-of-00880.parquet` | GitHub files (Python / JavaScript / Rust / CSS; MIT, BSD-3-Clause, Apache-2.0) via HF codeparrot/github-code-clean (substitute for the locked set's four repositories) | Per-file repository licence: MIT / BSD-3-Clause / Apache-2.0 (dataset Apache-2.0) |

Shard choice: `random.Random("20261008:<domain>:shard").randrange(<number of shards>)` for the multi-shard sources (chat, encyclopedia, papers, source_code); the others are read whole.

Differences from the locked set's sources, stated plainly:
- **prose**: PG-19's test split through the HF parquet mirror `emozilla/pg19-test` (`deepmind/pg19` is a loading script that downloads from Google storage).
- **papers**: the HF arXiv full-text dump `ccdv/arxiv-summarization` (config `document`, field `article`); the locked set's README names only 'arXiv full text (HF arxiv dump, train)'.
- **source_code (substitute)**: the locked set drew on four GitHub repositories (twbs/bootstrap, django/django, facebook/react, rust-lang/regex), which are not an HF dataset and could not be fetched under this build's network rule (HF datasets only). The substitute is GitHub files of the same languages and licences from `codeparrot/github-code-clean`: Python, JavaScript, Rust and CSS files under MIT, BSD-3-Clause or Apache-2.0, grouped by repository (repositories in seeded order, files in path order), each file headed by a `==> <repo>/<path> <==` line.
- **chat**: each UltraChat dialogue is written as `User: ...` / `Assistant: ...` turns separated by a blank line (the locked builder's exact dialogue format was not kept).
- **encyclopedia**: the file prefix is `encyclopedia_` (the locked set's `tech_doc_` prefix was a mislabel).

## Construction (as the locked long prompts)

- Units of one domain are joined with a blank line until the text holds nominal + 512 tokens under the Qwen3.8-Flash-Next tokenizer; the text is cut at the start of token nominal + 512 (the last unit is used in part; its rest is discarded, never reused).
- The continuation is the last 512 Flash-Next tokens, stored as a character offset (`long-prompts/SPLITS.tsv` split_char; `datasets/test-set-locked/finalize.py`'s rule), so every tokenizer splits at the same text point. SPLITS.tsv has finalize.py's columns (llama-3.1-8b, qwen3.5-9b, qwen3.8-flash-next) plus qwen3-8b; MANIFEST.tsv has the locked manifest's columns plus qwen3-8b.
- Tokenizers (tokenizer.json sha256): llama-3.1-8b `6b9e4e7fb171f92f`; qwen3.5-9b `5f9e4d4901a92b99`; qwen3.8-flash-next `0997f410c57a1f4e`; qwen3-8b `aeb13307a71acd8f`.
- Texts: CRLF/CR to LF and NUL removed (the harness reads the files with Path.read_text).

## Selection rule (seed 20261008)

Per domain, the rows of the shard in the order of `random.Random("20261008:<domain>").shuffle` (source_code: repositories in that order), keeping rows of at least chat 2000, encyclopedia 8000, legal 2000, papers 10000, prose 0, source_code 1000 characters; the first chat 1500, encyclopedia 1500, legal 3000, papers 600, prose 100, source_code 6000 such rows are the candidate pool (`fetch`; each pool's sha256 is pinned in the builder and in FROZEN.json). Units are taken in pool order for 8192, 16384, 32768, 65536, 128000; a unit is rejected and the next taken if:
1. its sha256 (raw or normalized text) equals a reference text's (or a whole locked file's), or
2. it shares a run of >= 50 words of consecutive shared normalized-word 13-grams (NFKC, casefold, whitespace split; `kva.impact_corpus.grams` at n = 13) with a reference, or with a unit already accepted for this set.

References (fingerprints only; no locked text is printed or written):
- `locked/long-prompts`, `locked/ruler`, `locked/longbench`: every frozen file of `datasets/test-set-locked/MANIFEST.tsv` (verified against FROZEN.json first), JSONL per string value;
- `calibration`: the T0.2 calibration universe from T0.2's inputs (`checks/t02/overlap.py` check_inputs; 1109 texts: `kva.data.calib_docs` train / held / sterm, the FN-CAP2 chat copies of `qfn.steps.capture_plan`, the impact corpus (sha256-checked), every text of the FN-BIGCAP plans PLAN-b0..b4; T0.2's own build_docs stops on its PLAN-chatold cross-check since the exclusion list changed the chat-ask order, so the texts are read directly) plus every text of the private plan-train / plan-held / plan-sterm unfiltered (the calibration-exclude list's documents included; `results/calib-docs-*.json` name subsets of plan-train);
- `dev`: the private plan-grade.json, `datasets/corpus/*.txt` and `datasets/niah-general-dev`.

Each assembled file was checked again as a whole against every reference (a hit would stop the build).

## Overlap report

Controls (in memory): positive test-set-locked/long-prompts/chat_128000_0.txt words 1000-1199 (in memory): flagged (200-word run); negative (600 random synthetic words, seed 20261008): not flagged.

Reference sizes: locked/long-prompts 126 texts, 3,072,412 distinct 13-grams; locked/ruler 4,720 texts, 2,774,090 distinct 13-grams; locked/longbench 4,935 texts, 2,073,874 distinct 13-grams; calibration 1,274 texts, 3,015,762 distinct 13-grams; dev 447 texts, 767,770 distinct 13-grams.

Per document: shared 13-grams / longest shared run in words, against each reference group (rejection at >= 50); no sha256 hit anywhere.

| document | units | locked/long-prompts | locked/ruler | locked/longbench | calibration | dev |
|---|---|---|---|---|---|---|
| chat_8192_0.txt | 5 | 3 / 15 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| chat_16384_0.txt | 15 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| chat_32768_0.txt | 30 | 4 / 14 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| chat_65536_0.txt | 50 | 13 / 17 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| chat_128000_0.txt | 108 | 9 / 17 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| encyclopedia_8192_0.txt | 4 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| encyclopedia_16384_0.txt | 7 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| encyclopedia_32768_0.txt | 5 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| encyclopedia_65536_0.txt | 7 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| encyclopedia_128000_0.txt | 38 | 0 / 0 | 0 / 0 | 24 / 24 | 0 / 0 | 0 / 0 |
| legal_8192_0.txt | 3 | 158 / 45 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| legal_16384_0.txt | 16 | 900 / 49 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| legal_32768_0.txt | 29 | 1406 / 47 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| legal_65536_0.txt | 38 | 1726 / 49 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| legal_128000_0.txt | 49 | 2638 / 49 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| papers_8192_0.txt | 2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| papers_16384_0.txt | 3 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| papers_32768_0.txt | 4 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| papers_65536_0.txt | 7 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| papers_128000_0.txt | 12 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| prose_8192_0.txt | 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| prose_16384_0.txt | 1 | 10 / 22 | 0 / 0 | 7 / 19 | 0 / 0 | 0 / 0 |
| prose_32768_0.txt | 1 | 3 / 15 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| prose_65536_0.txt | 2 | 9 / 21 | 0 / 0 | 7 / 19 | 0 / 0 | 0 / 0 |
| prose_128000_0.txt | 2 | 10 / 22 | 0 / 0 | 7 / 19 | 0 / 0 | 0 / 0 |
| source_code_8192_0.txt | 6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| source_code_16384_0.txt | 8 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| source_code_32768_0.txt | 9 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| source_code_65536_0.txt | 31 | 3 / 15 | 0 / 0 | 0 / 0 | 9 / 21 | 0 / 0 |
| source_code_128000_0.txt | 57 | 0 / 0 | 0 / 0 | 1 / 13 | 69 / 40 | 41 / 24 |

Largest overlap per reference group (all below the rejection run): locked/long-prompts: 49-word run, 2638 shared 13-grams (legal_128000_0.txt); locked/ruler: none; locked/longbench: 24-word run, 24 shared 13-grams (encyclopedia_128000_0.txt); calibration: 40-word run, 69 shared 13-grams (source_code_128000_0.txt); dev: 24-word run, 41 shared 13-grams (source_code_128000_0.txt). The legal documents share EUR-Lex formulae with the locked legal documents (thousands of shared 13-grams, runs just under the 50-word rule): a confirmation on the legal domain is the least independent of the locked set.

Rejected candidate units (taken in pool order, skipped, the next one used):

- chat: 0
- encyclopedia: 0
- legal: 39 (a unit already in this set: 11; locked/long-prompts: 28)
- papers: 0
- prose: 3 (locked/long-prompts: 3)
- source_code: 13 (a unit already in this set: 1; calibration: 12)

The source ids of every unit of every file are in `long-prompts/SOURCES.tsv` (chat: UltraChat prompt_id; encyclopedia: `wiki:<id>`; legal: `eurlex-train:<row>`; papers: `arxiv-summarization:<shard>:<row>`; prose: `pg19-test:<file>`; source_code: `github:<repo>/<path>`); the full numbers (and every rejected unit's id and reason) in `OVERLAP.json`.

## Licences and redistribution

- Per-domain licences are in MANIFEST.tsv (licence column) and the table above.
- **arXiv**: per-paper licences, most the arXiv non-exclusive licence (no redistribution). The locked set keeps its arXiv texts in this private repository (committed) and the public release excludes the locked texts (release/SCRUB-REPORT.md: builders and hashes only). This set does the same: the papers texts are committed here for private research only and must not ship in any public artifact; the builder and the hashes rebuild them from the pinned HF revision.
- Wikipedia text is CC-BY-SA-4.0 (attribution and share-alike apply to any redistribution); UltraChat MIT; LexGLUE CC-BY-4.0 with EUR-Lex reuse terms; PG-19 texts public domain in the US; the code files carry their repositories' MIT / BSD-3-Clause / Apache-2.0 licences.

## Rebuild

```bash
# 1. candidate pools (needs pyarrow and network to huggingface.co; any python with pyarrow):
FRESH_LONG_CACHE=<dir> python3 -P -E scripts/build_fresh_long.py fetch
# 2. the set (repo venv; the T0.2 inputs are required):
FRESH_LONG_CACHE=<dir> KVA_DATA=<private data> QFN_BIGCAP_PLANS=<FN-BIGCAP plans> \
  PYTHONPATH=. .venv/bin/python scripts/build_fresh_long.py build     # or --check
```
