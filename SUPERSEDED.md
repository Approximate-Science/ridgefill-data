# What the paper replaces

This file names earlier measurements and files that the paper does not use, so that a reader who meets them
elsewhere knows which numbers replace them. It holds no data.

## Public records

- **The technical note** (Zenodo, DOI 10.5281/zenodo.23179168; GitHub `Dyluhn/ridgefill`, 2026-10-05). Its headline
  speed figures for Qwen3.8-Flash-Next were measured mid-ramp against a slow stock run, and they are superseded. The paper's Flash-Next numbers (§5) come from `results/flashnext/final-det-20261010/`: exact and
  RidgeFill on the same server start, alternating, ratio of medians.
- **Map files with the paper's file names in other repositories.** Files named like
  `<model>-sweep-s<split>.safetensors` in other repositories, including the gated research datasets under the same
  Hugging Face organization, are earlier fits with different hashes. The paper's maps are exactly the files whose
  sha256 is listed in `provenance/external.tsv` and recorded in each cell's `cell.json`.

## Runs left out of this repository

- **Abandoned reference cells.** Ten reference-study runs stopped before they finished; `results/reference/ABANDONED.md`
  lists them with the reason and a digest of their files.
- **Qwen3.8-27B on AMD R9700.** The earlier E1/E2/E3 measurements on R9700 cards (Vulkan) are superseded by the
  four-A100 run in `results/qwen38-27b/`. One R9700 timing series was measured on a degraded host before a crash.
- **Qwen3.8-27B multi-user attempt.** One unified-C8 attempt failed the harness's arrival check because request files
  were written after the schedule clock started. It is left out; `results/qwen38-27b/results/MIX-HARNESS-AMENDMENT.json`
  records the harness fix and the rerun scope.
- **Earlier answer budgets and protocols.** RULER at non-official answer budgets, LongBench v2 at a 4,096-token cap,
  and custom needle suites are not used.
