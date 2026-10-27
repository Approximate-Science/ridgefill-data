# finals/: the declared cells of the reference-model finals on a rented NVIDIA box

0. Cells are generated: `finals/make_cells.py` writes `finals/cells/*.json` (16: Llama-3.1-8B-Instruct, Qwen3.5-9B,
   Qwen3-8B x ruler / longbench / nll / decode / secondq, plus the Llama speed cell), to the scope in its docstring.
   `--check` proves the committed ones are the generated ones. Never edit a cell by hand; a change after registration
   needs a new FINAL- id. Each projector file is pinned by its sha256 from the clean fit report
   (`results/fit-sweep-<model>-s<S>.json` whose rows are the clean count); until then the cell is unpinned and
   cannot be registered or run.
1. Build the image on the tagged harness: `finals/build.sh protocol-v1.1` (context is an allowlist: no models, no
   locked texts, no results; the committed calibration inputs, the derived-set manifests and the commit ride along;
   `--check` verifies everything offline).
2. On the box: one volume at `/data`; the locked set at `/data/locked-set` (KVA_LOCKED_DIR, with `MANIFEST.tsv` +
   `FROZEN.json`), and beside it any set a cell names in its optional `dataset` field (`/data/fresh-long-20261008`,
   longppl only: `datasets/fresh-long-20261008/PROVENANCE.md`); Qwen3-8B's clean projectors copied to `/data/proj`. First job, the clean refits:
   `docker run --gpus all -v /data:/data -e KVA_GPU_NAME=A100 -e HF_TOKEN=... --entrypoint /opt/kva/finals/refit_clean.sh kva-finals:protocol-v1.1`
   (Llama + Qwen3.5 at S8/12/16/20 into `/data/proj`); copy their fit reports (`/data/refit-work/results/fit-sweep-*.json`)
   into the repo's `results/` and rerun `make_cells.py`: the Llama and Qwen3.5 cells are now pinned.
3. Register: `finals/register.sh` PRINTS one `labbook hypothesis --dir ledger/paper FINAL-<MODEL>-<KIND> ...` per
   cell (claim, prediction and kill from OUTLINE.md, the statistics rules, and the pins: the sha256 of the cell and
   of every harness config generated from it). Review, then run them from the repo root, all from one tree (the
   pins are of the bytes that tree generates). Unpinned cells and the speed cell print a named refusal instead.
   A registered cell is never rewritten (make_cells.py refuses: RegisteredCellChanged); a cell changed before any
   run gets a new id (finals/lib/registration.py SUPERSEDES: FINAL-QWEN3-8B-NLL -> FINAL-QWEN3-8B-NLL-2), and
   register.sh also prints the `labbook record` of the supersession, after which the old id is refused.
   Commit, re-tag, rebuild the image: its ledger must hold the records, because the harness checks every config's
   FINAL- id against the image's own ledger (kva/protocol.py); KVA_LEDGER reaches only runner.py's check.
   Then one container per cell, one card per cell:
   `docker run -d --gpus '"device=0"' -v /data:/data -e KVA_GPU_NAME=A100 -e HF_TOKEN=... kva-finals:protocol-v1.1 finals/cells/llama-3.1-8b-instruct-ruler.json`
4. Before any GPU work a cell refuses by name: a FINAL- record missing or pinning other bytes (ConfigNotRegistered),
   an unpinned or stale projector (UnpinnedProjector; StaleProjector: not the pinned bytes, or metadata rows / docs
   not the clean calibration list's), a derived-set manifest that is not the pinned one, a GPU that is not the named
   one (kva/gpu.py), a changed locked set.
5. The paper's runs pushed each finished cell to two private destinations (scripts not in the public code). The
   public code has no push step: results stay in `$FINALS_RESULTS` and the runner says so.
6. A dead card: rent another box, mount the same `/data`, rerun with `--new-host` (old results kept as
   `.abandoned-<utc>`, the cell restarts whole, never spliced across hosts).
7. Results land in `$FINALS_RESULTS/<cell>/` (`cell.json`, `harness-<config>.json`, `host.json`, `canary.json`,
   `work/results/<set>-<config>/<prompt>.json` with `.sha256` sidecars); `.done` appears when the cell is finished
   (in the paper's runs, only after both pushes succeeded).
8. Back home: `python finals/verify_finals.py RESULTS --ledger ledger/paper/ledger.jsonl --require-done`: every
   file against its sidecar, cell.json and every harness config against the FINAL- record's pins; non-zero exit
   and no table on any mismatch.
9. Convert: `python finals/convert.py RESULTS --ledger ledger/paper/ledger.jsonl` turns each cell's harness records
   into the sample records the tables read (`samples/`, stamped by `samples/CONVERTED.json`; verify first, official
   scorers re-run, refusals by name; docstring). Then `python finals/tables.py RESULTS`. `finals/align_audit.py`
   lists the table cells the declared cells cannot fill (`finals/align-gaps.json`, a reviewed list). While the
   Flash-Next tier has not run, `--pending flashnext` writes every reference table with each Flash-Next value as
   `\pending` (refused once a flashnext cell, results directory or FINAL- record of it exists; tables.py docstring).
   On the public copy of the results (no ledger, no dotfiles, no projector files):
   `python finals/tables.py RESULTS --no-ledger --out DIR` writes every table it can compute and lists the cells that
   need the ledger or the projectors (`--proj-dir` adds the projector values); `verify_finals.py RESULTS --no-ledger`
   checks the sample records against their sidecars.

Env (named errors when missing): `KVA_GPU_NAME` (the rented card's name, e.g. A100; the image sets
`KVA_GPU_VENDOR=nvidia`), `KVA_LOCKED_DIR`, `HF_TOKEN` (Llama is gated),
`KVA_PROJ_DIR` (`/data/proj`), `KVA_MODEL_DIR` (`/data/models`), `KVA_DATA` (`/data`: refit sums and projectors),
`KVA_REPO` (`/opt/kva`), `FINALS_RESULTS` (default `/data/results/finals`), `KVA_LEDGER` (default
`$KVA_REPO/ledger/paper/ledger.jsonl`; read by runner.py only, the harness reads the image's ledger: build after
registering), `KVA_PYTHON`.

`KVA_FINALS_MOCK=1` swaps the reference harness for `finals/mock/harness.py` (deterministic fakes, CPU); nothing
from a mock run is evidence. The one named refusal left: `TODO-HARNESS-VLLM-TTFT` (T3.1 speed cell; kvabackend.py).
