# engine-fix: the one-file determinism fix for the serving engine

The engine run of the paper used `tcclaviger/vllm:29.08.10` (digest `sha256:809657ddb1533db10a0c08267c3ee6fde180b839e379713414efa121415914a3`;
vLLM 0.29.0.dev0 at upstream commit 2bdbbc80 plus the fork's patches) with one file replaced:

| file | sha256 |
|---|---|
| the fixed `qsa.py`, mounted over `vllm/models/qwen4_exp/amd/ops/qsa.py` (not shipped: rebuild it as below) | `eaf0dd1975799f4c8897cbea62791de44fd28322758fb9891cbe7ff0ef52db99` |
| the image's original `qsa.py` | `942aa353ae33b9d475122b7b318d1fea6be05a7c94d6acd59b4c9b409656de2d` |
| `SHA256SUMS` (the overlay's checksum list) | `37bd46ff34c4e3ab940aa50e912a4e63416ddde29c3122f0d74ae3347d63baac` |

`patch-qsa-v3.diff` is the diff against the original; `PATCHES.md` explains the three changes (P1 canonical
selection order, P2 deterministic block top-k above 2048 columns, P3 NaN guard). The file carries the vLLM project's
Apache-2.0 SPDX header.

Rebuild the fixed file from the image's original, check both hashes, and apply it as a read-only bind mount, as the
run kit does:

    docker run --rm --entrypoint cat tcclaviger/vllm:29.08.10 \
        /opt/vllm/lib/python3.14/site-packages/vllm/models/qwen4_exp/amd/ops/qsa.py > qsa.py
    sha256sum qsa.py                                 # 942aa353... (the original)
    patch qsa.py engine-fix/patch-qsa-v3.diff        # outside a git checkout: git apply -p6 engine-fix/patch-qsa-v3.diff
    sha256sum qsa.py                                 # eaf0dd19... (the fixed file)
    docker run ... -v $PWD/qsa.py:/opt/vllm/lib/python3.14/site-packages/vllm/models/qwen4_exp/amd/ops/qsa.py:ro ...

With the fix, exact's two perplexity passes on one server start are bitwise identical on all 90 documents; the fix costs
5-7% of decode throughput and does not change TTFT, since every arm ran with it.
