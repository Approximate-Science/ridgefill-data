# Scan allowlist

`tools/scrub_check.py` fails on every hit that no row below names. Each row names a path glob (`*` also matches
`/`), the rule it exempts, and the reason. Exempted hits are still listed in the scan report, and a row that matches
nothing fails the scan. Rows were written and checked by hand for release v1.0 (2026-10-27), each after reading
every hit it covers.

| path glob | rule | reason |
|---|---|---|
| results/reference/* | PCI_ID | PCI bus addresses of the rented cloud GPUs (RTX 4090, L40S, RTX 6000 Ada, A100) in each run's provenance block; not the authors' machines and not serial numbers or UUIDs. |
| results/qwen38-27b/* | PCI_ID | PCI bus addresses of the four rented A100 GPUs in the 27B run's provenance and process records; not serial numbers or UUIDs (those were replaced by GPU_UUID-n labels). |
| results/qwen38-27b/longbench/*/66ec0f7f821e116aacb19dde.json | RADIANCE | The model's answer to a LongBench v2 question about 3D style transfer uses the term "radiance field"; unrelated to any software of that name. |
| results/qwen38-27b/longbench/stock/66fa2734bb02136c067c627a.json | EMAIL | The model's answer writes an example URL with the placeholder credentials user:token; no address or credential. |
| results/reference/llama-3.1-8b-instruct-longbench/work/results/longbench-llama-3.1-8b-instruct-longbench/lb-lcc-38.json | EMAIL | A Python decorator (@pytest.mark) after an escaped newline in generated code; not an address. |
