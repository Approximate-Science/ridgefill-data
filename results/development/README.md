# Development measurements (Appendix I)

These records come from the method's development, before the evaluation design of the paper's §3 was fixed. They
back statements in §1 and §3 and the numbers of Appendix I. They are not evidence for §4 to §6: `MANIFEST.tsv` gives
every file here the status DEVELOPMENT, and the sets are small. `recompute/dev_history.py` recomputes every number of
Appendix I from these files (claims D1 to D5 in `CLAIMS.md`).

| Folder | Appendix | Model | What it holds |
|---|---|---|---|
| `closures/` | I.1, Table I.1 | development checkpoint | Per-prompt, per-layer attention-output error of the plain fill and eight corrections (`k3k7_raw.json`), the original summary (`k3k7_summary.json`), the corrections in the form the engine can apply (`k3k7x.json`), and the end-to-end loss with and without the key-mean shift (`k15.json`) |
| `whiteness/` | I.1 | development checkpoint | Correlation of the fill error along the prompt on three held-out documents (`diag_residual.log`); exact rows spread through the prompt against an exact tail of the same size (`results_e4.json`) |
| `needles-dev/` | I.2 | development checkpoint | 60 needle prompts, every arm's answer (`k20/`, summary `k20.json`); digit rows exact against the fill and a longer tail, corrected answer format (`k28_niah.json`, summary only) |
| `needles-reference/` | I.2, Table I.2 | Llama-3.1-8B, Qwen3.5-9B | Numeric multi-key needles (`niah-bypass-mk4-*`) and text needles (`niahg-bypass-*`), every arm's answer per prompt; the original summaries |
| `token-classes/` | I.3 | Llama-3.1-8B, Qwen3.5-9B | Fill error by token class at split 12 (`{llama,qwen}-s12.json`); first-order impact per document (`te2-concentration.csv`) and its check against the measured gain (`te2-*-s12-t1024.json`) |
| `certificate-27b/` | I.4 | Qwen3.8-27B | Decode-time check: timing with and without the full per-row scan (`b1-job11`, `b1-job14-timing`), miss rates of the periodic check (`b1-job12`, `b1-job13`, `b1-job14-diag`) |
| `needles-vs-ruler-27b/` | I.5 | Qwen3.8-27B | The configuration of 2026-10-22 against stock: both needle suites and RULER at 16K and 32K, summaries and both arms' answers |

**The development checkpoint.** `closures/`, `whiteness/` and `needles-dev/` used our own
IQ4_XS quantization of the community build `orcarouter/Qwen3.8-Flash-Next-Uncensored`, served in our own development
engine on two R9700 GPUs at split 16. It is not the checkpoint of §5, which is `tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ` in vLLM.

**Qwen3.8-27B.** `certificate-27b/` and `needles-vs-ruler-27b/` ran on development builds of the llama.cpp fork
(Vulkan, AMD R9700 GPUs), not the A100 runs of §4.

**What was changed.** Host names, GPU identifiers, PCI addresses and home paths in the records' provenance fields
were replaced by `tools/strip_prompts.py` (rows in `provenance/STRIPPED.tsv`); an internal project name was
replaced the same way where it appears. Nothing else was edited. `provenance/ORIGINS.tsv` says where each folder came from.
