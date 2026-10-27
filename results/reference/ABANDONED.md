# Abandoned reference cells (not released)

Ten runs of the reference study were stopped before they finished and replaced by the cells in this folder. Their
partial records are not released and no number in the paper comes from them. Each is listed with the reason and a
digest of its files, so a copy can be matched later: the digest is the sha256 of the `sha256sum` listing of every file
in the run's folder (paths relative to the folder, sorted bytewise, dot-files left out).

| Run (folder name) | Reason | Files | Result records | Registration | Tree digest (sha256) |
|---|---|---|---|---|---|
| llama-3.1-8b-instruct-nll, 2026-10-07T05:19Z | older code | 9 | 1 | FINAL-LLAMA-NLL | 830d3ef9d16d57c0e07c110bbe6469cacfe3cff04822d4cd206e07cf0a0208bd |
| llama-3.1-8b-instruct-ruler, 2026-10-07T23:09Z | older code (protocol 1.1.3 generation cap) | 36 | 28 | FINAL-LLAMA-RULER-2 | 799044f3101cfe8caae1e1e3f1b7311c7cb4cf351fe1b40eb4963edd1ee80df0 |
| qwen3.5-9b-nll, 2026-10-07T05:19Z | older code | 9 | 1 | FINAL-QWEN35-NLL | 79895e5599699f0fbb789091d947b0b0731c06b7374acabdb0dc8cda21835cbf |
| qwen3.5-9b-nll-causal, 2026-10-08T00:09Z | rented host lost mid-run | 30 | 22 | FINAL-QWEN35-NLL-CAUSAL | 2c9f33a359cd6d720021cbbad424bf4ef6ed7746f40b0173071b90cf35da12fd |
| qwen3.5-9b-ruler, 2026-10-07T23:09Z | older code (protocol 1.1.3 generation cap) | 19 | 11 | FINAL-QWEN35-RULER-2 | 894a0de5922ccbc50a9aa146ae004f04dec36c6c06ed53a2258b09f0f3bacc35 |
| qwen3.5-9b-ruler, 2026-10-08T04:24Z | rented host expired mid-run | 106 | 98 | FINAL-QWEN35-RULER-2 | a8c7131a6d9c94eb3ada682f03007ca190b6ae5710fdf182f4566aa805f9cf2e |
| qwen3.5-9b-secondq, 2026-10-07T05:19Z | older code | 9 | 1 | FINAL-QWEN35-SECONDQ | 58f99215a5669cd3d368685e6465734b50d2a4b0247de3fe5af500f8d7d4831a |
| qwen3-8b-nll, 2026-10-07T06:08Z | older code; replaced by the re-registered cell (FINAL-QWEN3-8B-NLL-2) | 90 | 82 | FINAL-QWEN3-8B-NLL-2 | 25ffa819cf89bb1d2b3eb62306d6c122844288d9d68bf666a312ed42fd34e35a |
| qwen3-8b-ruler, 2026-10-07T06:08Z | older code | 54 | 46 | FINAL-QWEN3-8B-RULER | 22856a3292efffca15829e84b37304041a01f2fbd97cad4d2597d017fb5ed92b |
| qwen3-8b-secondq, 2026-10-07T06:08Z | older code | 16 | 8 | FINAL-QWEN3-8B-SECONDQ | 68fa2df26a96ff70e327d6a748a819d54388a37ca3b7d6335e941ff58951a41e |

"Older code" means the run had started on a code commit that was then replaced when arms or protocol rules were
added; the released cell of the same name reran every item on the new commit. The ledger records each change
(`provenance/ledger/`).
