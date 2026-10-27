# Patches (overlay files mounted read-only over the image's originals)

In the determinism work the patched files lived in an `overlay/` directory (checksums: `overlay/SHA256SUMS`, here
`SHA256SUMS`), each mounted as `-v overlay/<path>:/opt/vllm/lib/python3.14/site-packages/vllm/<path>:ro`; the
originals came from the image (`engine-src/vllm/`). This release ships the diff only: `README.md` rebuilds the fixed
file from the image's original.

## P1 + P2 + P3: `models/qwen4_exp/amd/ops/qsa.py`

Versions of the file:
| version | sha256 | used by | content |
|---|---|---|---|
| v1 | (not kept) | D3 `d3-sortfix` | P1 only. Equals v2 run with `DET_QSA_SELECT_MAX_COLS=-1` (the P2 branch is skipped) |
| v2 | c0a7ae3b... | D4, v1-v4 boots | P1 + P2 |
| v3 | eaf0dd19... (current, `overlay/SHA256SUMS`) | g1, g3 boots | P1 + P2 + P3 |

- **P1 (D3): canonical order of the fused select output.** Right after libr4d `attn_sparse_topk_expand`, each row
  of the selection is sorted ascending with -1 padding last (`_canonical_selection_order_`). Env `DET_QSA_SORT=0`
  turns it off. For <= 2048 columns (context <= 8192 tokens) the small kernel already emits ascending: positions
  <= 8192 stayed bitwise equal to the unpatched D1 values, so P1 is a no-op there.
- **P2 (D4): deterministic block top-k above 2048 columns.** When the fused select would run with more than
  `DET_QSA_SELECT_MAX_COLS` (default 2048) columns, the block top-k is computed in torch instead:
  - the kth largest visible score, from `torch.topk` values only (the value is deterministic);
  - every block above it, plus ties at the kth value taken lowest index first (cumsum rank);
  - emitted ascending, -1 padded;
  - expanded to token indices by the engine's own Triton `expand_qsa_block_indices_cuda` (the existing fallback);
  - `sel_len` set to the live count.

  `DET_QSA_SELECT_MAX_COLS=-1` keeps the fused select everywhere (= P1 only). Decode passes no `max_seq_len`, so it
  scores the full table width (~16.6K columns) and always takes P2.
- **P3 (after D7, code review): NaN guard.** A visible NaN score is excluded from selection. Without the guard,
  `topk` would rank it first and poison the kth value. For finite scores it is a bitwise no-op (checked by g1).
- **CPU unit test** against a (score desc, index asc) reference: 300 random cases with heavy exact ties, NaN
  garbage past the visible prefix, 1% NaN inside it, and rows > 512. All equal (the test script is not part of this
  release).

Diff of v3 against the original:

```diff
--- engine-src/vllm/models/qwen4_exp/amd/ops/qsa.py	2026-10-04 21:23:10.000000000 -0500
+++ overlay/models/qwen4_exp/amd/ops/qsa.py	2026-10-10 08:41:57.492799507 -0500
@@ -15,6 +15,51 @@
 
 _LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
 _TOPK_WORKSPACE_BYTES = 1024 * 1024
+# DET overlay (2026-10-10, ridgefill-determinism D3): the libr4d fused select returns the right set of indices but in a
+# run-dependent order once a row sees more than 2048 compressed blocks (context > 8192 tokens); the split-tile sparse
+# attention sums in selection order, so identical requests differ there. Sort each row ascending, -1 padding last.
+_DET_QSA_SORT = os.environ.get("DET_QSA_SORT", "1") != "0"
+_DET_INT32_MAX = 2**31 - 1
+
+
+# D4: the fused select also breaks exact score ties (repeated text gives identical compressed keys) run-dependently
+# above 2048 columns. Above DET_QSA_SELECT_MAX_COLS columns the block top-k is done here instead, ties broken by
+# (score desc, block index asc), then expanded by the Triton kernel. -1 keeps the fused select everywhere.
+_DET_QSA_SELECT_MAX_COLS = int(os.environ.get("DET_QSA_SELECT_MAX_COLS", "2048"))
+_DET_SELECT_ROWS = 512
+
+
+def _canonical_selection_order_(selection: torch.Tensor) -> None:
+    keys = selection.masked_fill(selection < 0, _DET_INT32_MAX)
+    keys = torch.sort(keys, dim=1).values
+    selection.copy_(keys.masked_fill_(keys == _DET_INT32_MAX, -1))
+
+
+def _deterministic_block_topk(
+    logits: torch.Tensor, visible_blocks: torch.Tensor, block_topk: int
+) -> torch.Tensor:
+    """[rows, block_topk] int32 block ids: the top block_topk visible columns by score, exact ties to the lower
+    index, emitted ascending, -1 padded (rows with fewer visible blocks keep them all)."""
+    rows, columns = logits.shape
+    k = min(block_topk, columns)
+    out = torch.full((rows, block_topk), -1, dtype=torch.int32, device=logits.device)
+    cols = torch.arange(columns, device=logits.device)
+    for r0 in range(0, rows, _DET_SELECT_ROWS):
+        r1 = min(r0 + _DET_SELECT_ROWS, rows)
+        # a NaN score is never selected (it would rank first in topk and poison the kth value)
+        valid = (cols[None, :] < visible_blocks[r0:r1, None]) & ~torch.isnan(logits[r0:r1])
+        scores = logits[r0:r1].masked_fill(~valid, float("-inf"))
+        kth = torch.topk(scores, k, dim=1, sorted=False).values.amin(dim=1, keepdim=True)
+        above = scores > kth
+        tied = valid & (scores == kth)
+        need = k - above.sum(dim=1, keepdim=True)
+        take = above | (tied & (torch.cumsum(tied, dim=1) <= need))
+        slot = torch.where(take, torch.cumsum(take, dim=1) - 1, k)
+        dst = torch.full((r1 - r0, k + 1), -1, dtype=torch.int32, device=logits.device)
+        dst.scatter_(1, slot, cols.to(torch.int32).expand(r1 - r0, -1))
+        out[r0:r1, :k] = dst[:, :k]
+    return out
+
 
 _CLAV_QSA_SCORE: bool | None = None
 
@@ -1239,6 +1284,20 @@
             compress_ratio,
             num_columns=columns,
         )
+        if use_r4d_select and 0 <= _DET_QSA_SELECT_MAX_COLS < columns:
+            blocks = _deterministic_block_topk(logits, visible_blocks, block_topk)
+            expand_qsa_block_indices_cuda(
+                blocks,
+                query_positions[row_slice],
+                sequence_lengths,
+                token_to_req[row_slice],
+                compress_ratio,
+                token_topk,
+                out[row_slice],
+            )
+            if sel_len is not None:
+                sel_len[row_slice].copy_((out[row_slice] >= 0).sum(dim=1))
+            continue
         if use_r4d_select and logits.stride(1) == 1:
             from vllm.model_executor.kernels.r4d_lib import import_r4d
 
@@ -1259,6 +1318,8 @@
                 out_chunk.stride(0),
                 torch.cuda.current_stream().cuda_stream,
             )
+            if _DET_QSA_SORT:
+                _canonical_selection_order_(out_chunk)
             continue
         if sel_len is not None:
             # Unfused chunk: the full padded width is the always-safe hint (the
```
