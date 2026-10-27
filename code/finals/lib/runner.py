#!/usr/bin/env python3
"""Run one declared finals cell end to end, unattended, on a rented box (finals/run_final.sh).

Order (every step fails closed and refuses by name - BRIEF-COMMON gate rule 3):
  1. validate the cell against finals/cell.schema.json (+ cross-field rules)
  2. refuse registration 'FINAL-TBD' by name; the FINAL- record must exist in the ledger (read-only) and pin the
     sha256 of this cell and of every harness config generated from it (finals/lib/registration.py)
  3. check the mounted locked set: MANIFEST.tsv's sha256 == the cell's dataset_manifest_sha256
     == FROZEN.json (the frozen-set rule, PLAN.md 'Locking and backups'), and every source file
     against its manifest row. A cell with 'dataset' (e.g. fresh-long-20261008, longppl only) reads that
     set instead, from the directory of that name beside KVA_LOCKED_DIR, with the same checks
  4. host fingerprint: refuse to resume a cell on a different host without --new-host (which
     restarts the cell, moving the old results to .abandoned-<utc>)
  5. regenerate the host canary (exact arm on 512 tokens, stored with GPU name and driver)
  6. run the samples: all arms of a sample before the next sample (the pairing unit), each written
     atomically (tmp, fsync, sha256 sidecar, read back, rename); finished samples are skipped
     (idempotent resume)
  7. push.sh to BOTH destinations; only its success marks the cell done (.done). Without finals/push.sh (the
     public code) the push is skipped with a warning and the results stay local
Mock mode (KVA_FINALS_MOCK=1) runs the mock harness instead of the reference one; nothing from a
mock run is evidence.

Env: KVA_LOCKED_DIR (required; the mounted locked set), KVA_LEDGER (default $KVA_REPO/ledger/paper/
ledger.jsonl), FINALS_RESULTS (default $KVA_REPO/results/finals), KVA_REPO (the harness repo),
KVA_FINALS_MOCK=1, KVA_MODEL_DIR + KVA_PROJ_DIR (kva mode), HF_TOKEN (model pull), KVA_MOCK_HOST /
KVA_MOCK_GPU / KVA_MOCK_DRIVER (mock host identity for tests).
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import importlib.util

FINALS = Path(__file__).resolve().parents[1]
LIB = FINALS / "lib"
sys.path.insert(0, str(LIB))
import atomic                                                # noqa: E402
import cellcheck                                             # noqa: E402
import host_canary                                           # noqa: E402


def _mock_harness():
    """Load finals/mock/harness.py by path ('mock' is a stdlib-shadowing name: no path import)."""
    spec = importlib.util.spec_from_file_location("finals_mock_harness", FINALS / "mock" / "harness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import kvabackend                                            # noqa: E402
import registration                                          # noqa: E402


def refuse(msg):
    print(f"refusing: {msg}", file=sys.stderr)
    sys.exit(2)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def repo():
    r = os.environ.get("KVA_REPO") or str(FINALS.parent)
    if not (Path(r) / "scripts" / "evaluate.py").exists():
        refuse(f"KVA_REPO {r!r} has no scripts/evaluate.py: not the harness repo")
    return Path(r)


def load_locked(cell):
    """(manifest, dir) with every manifest row's file hash-checked; named refusals throughout. A cell with 'dataset'
    reads that frozen set (beside the locked set: $KVA_LOCKED_DIR/../<dataset>) instead, under the same checks."""
    d = os.environ.get("KVA_LOCKED_DIR")
    if not d:
        refuse("KVA_LOCKED_DIR is not set: mount the locked set and name it (models come from HF, "
               "locked-set texts never ride inside the image)")
    d = Path(d)
    name = cell.get("dataset")
    what = "the locked set" if name is None else f"dataset {name}"
    if name is not None:
        d = d.parent / name
        if not d.is_dir():
            refuse(f"dataset {name}: {d} does not exist (it is uploaded beside the locked set: provision.sh)")
    if not (d / "MANIFEST.tsv").is_file():
        refuse(f"{'KVA_LOCKED_DIR ' if name is None else ''}{d} has no MANIFEST.tsv: this is not {what}")
    man_sha = atomic.sha256_file(d / "MANIFEST.tsv")
    if man_sha != cell["dataset_manifest_sha256"]:
        refuse(f"{what} changed: {d}/MANIFEST.tsv hashes to {man_sha}, the cell declares "
               f"{cell['dataset_manifest_sha256']} (PLAN.md: locked sets are frozen; a change needs a "
               "protocol amendment, never a quiet rerun)")
    frozen = json.loads((d / "FROZEN.json").read_text()) if (d / "FROZEN.json").is_file() else {}
    if frozen.get("manifest_sha256") != man_sha:
        refuse(f"{d}/FROZEN.json manifest_sha256 {frozen.get('manifest_sha256')} does not match "
               f"{d}/MANIFEST.tsv ({man_sha}): {what} is inconsistent")
    manifest = {}
    for row in csv.DictReader((d / "MANIFEST.tsv").open(), delimiter="\t"):
        f = d / row["path"]
        if not f.is_file():
            refuse(f"{'locked-set' if name is None else name} file {row['path']} (from MANIFEST.tsv) is missing in {d}")
        if f.suffix == ".jsonl":
            with f.open() as fh:
                rows = sum(1 for line in fh if line.strip())
        else:
            rows = 1
        manifest[row["path"]] = {"sha256": row["sha256"], "rows": rows}
    return manifest, d


PREFIX = {"ruler": "ruler/", "multiturn": "ruler/", "longbench": "longbench/", "longppl": "long-prompts/",
          "decode": "long-prompts/", "ttft": "long-prompts/"}


def check_sources(cell, manifest, d):
    """Hash every source file the cell's runs read against its manifest row (hashing locked texts is
    allowed; nothing else ever opens them)."""
    prefixes = tuple({PREFIX[r["set"]] for r in cell["runs"]})
    for rel, m in manifest.items():
        if not rel.startswith(prefixes):
            continue
        got = atomic.sha256_file(d / rel)
        if got != m["sha256"]:
            what = "locked-set" if cell.get("dataset") is None else cell["dataset"]
            refuse(f"{what} file {rel} hashes to {got}, MANIFEST.tsv says {m['sha256']}: "
                   f"{'the locked set' if cell.get('dataset') is None else 'the set'} changed since it was frozen")


def sample_ids(cell, manifest):
    ids = _mock_harness().sample_ids(cell, manifest)
    if not ids:
        refuse(f"no {cell['kind']} rows in the locked MANIFEST.tsv for cell {cell['name']}")
    return ids


def check_ledger(cell, ledger):
    """The FINAL- record must exist and pin exactly this cell and the configs generated from it."""
    if not ledger.is_file():
        refuse(f"the ledger {ledger} does not exist (KVA_LEDGER): the FINAL- registration can not be checked")
    try:
        problems = registration.check_record(cell, ledger)
    except kvabackend.HarnessGap as e:
        refuse(str(e))
    if problems:
        refuse("\n  - ".join(problems))


def ensure_models(cell, mock):
    """Pull the cell's model(s) from HF by id at the pinned commit; named errors when missing."""
    if mock:
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        refuse("huggingface_hub is not installed: the model pull can not run")
    root = os.environ.get("KVA_MODEL_DIR")
    if not root:
        refuse("KVA_MODEL_DIR is not set: name where the pulled HF models live (never inside the image)")
    swift = any(r.get("model") == "swiftkv" for r in cell["runs"])
    for m in [cell["model"]] + ([cell["swiftkv_model"]] if swift else []):
        dest = Path(root) / m["name"]
        try:
            snapshot_download(repo_id=m["id"], revision=m["commit"], local_dir=str(dest),
                              token=os.environ.get("HF_TOKEN"))
        except Exception as e:                                   # noqa: BLE001 - named below, not swallowed
            refuse(f"model pull failed for {m['id']} at {m['commit']} into {dest}: {type(e).__name__}: {e}")
        print(f"model {m['id']} at {m['commit']} -> {dest}", flush=True)


def run(args):
    mock = os.environ.get("KVA_FINALS_MOCK") == "1"
    cell_path = Path(args.cell)
    cell = json.loads(cell_path.read_text())
    schema = json.loads(Path(args.schema).read_text())

    # 1. schema + cross-field rules
    errs = cellcheck.validate(schema, cell, cell_path.name)
    serrs, warns = cellcheck.semantic(cell)
    if errs or serrs:
        refuse("the cell config is invalid against " + str(args.schema) + ":\n  - " + "\n  - ".join(errs + serrs))
    for w in warns:
        log(f"WARNING: {w}")

    # 2. registration
    if cell["registration"] == "FINAL-TBD":
        refuse(f"cell {cell['name']} carries registration FINAL-TBD: no FINAL- ledger record exists yet "
               "(write it with labbook hypothesis --dir ledger/paper, then put its id in the cell)")
    check_ledger(cell, Path(os.environ.get("KVA_LEDGER", repo() / "ledger" / "paper" / "ledger.jsonl")))

    # 3. locked set
    manifest, locked_dir = load_locked(cell)
    check_sources(cell, manifest, locked_dir)

    # 4. results dir + host fingerprint
    results_root = Path(os.environ.get("FINALS_RESULTS", repo() / "results" / "finals"))
    cell_dir = results_root / cell["name"]
    info = host_canary.host_info(mock)
    fp = host_canary.fingerprint(info)
    if (cell_dir / "host.json").exists():
        stored = json.loads((cell_dir / "host.json").read_text())
        if stored.get("fingerprint") != fp:
            if not args.new_host:
                refuse(f"cell {cell['name']} was started on host {stored['host']['host']} "
                       f"({stored['host']['gpu_name']}, driver {stored['host']['driver']}) and this is "
                       f"{info['host']} ({info['gpu_name']}, driver {info['driver']}): a cell does not "
                       "resume on a different host. Pass --new-host to restart the cell here (the old "
                       "results are kept as .abandoned-<utc>)")
            gone = cell_dir.with_name(cell_dir.name + ".abandoned-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
            shutil.move(str(cell_dir), str(gone))
            log(f"--new-host: previous results moved to {gone}; the cell restarts on this host")
    cell_dir.mkdir(parents=True, exist_ok=True)
    if not (cell_dir / "host.json").exists():
        atomic.write_result(cell_dir / "host.json", {"fingerprint": fp, "host": info})
    # the cell's declaration is frozen with its first results
    if (cell_dir / "cell.json").exists():
        if json.loads((cell_dir / "cell.json").read_text()) != cell:
            refuse(f"cell.json in {cell_dir} differs from {cell_path}: a declared cell does not change "
                   "after results exist (write a new FINAL- record and a new cell instead)")
    else:
        atomic.write_result(cell_dir / "cell.json", cell)

    # 5. models + canary (the canary stores the GPU name and driver with the reference)
    ensure_models(cell, mock)
    canary_path, arms = host_canary.run(cell, cell_dir, mock)
    canary_sha = atomic.sha256_file(canary_path)
    log(f"canary ok ({', '.join(sorted(arms))}) -> {canary_path}")

    # 6. samples: pairing unit = one sample with all its arms, atomic, idempotent
    ids = sample_ids(cell, manifest)
    samples_dir = cell_dir / "samples"
    todo = []
    for sid in ids:
        p = samples_dir / (sid.replace("/", "__") + ".json")
        if atomic.finished(p):
            continue
        todo.append((sid, p))
    log(f"{len(ids)} samples, {len(ids) - len(todo)} finished, {len(todo)} to run")

    if mock:
        MH = _mock_harness()
        for cfg, _ in kvabackend.harness_configs(cell):       # the configs the FINAL record pins, as the kva mode writes them
            atomic.write_result(cell_dir / f"harness-{cfg['name']}.json", cfg)
        for sid, p in todo:
            atomic.write_result(p, MH.run_sample(cell, sid, manifest, canary_sha))
            log(f"sample {sid} done ({sum(len(r['arms']) for r in cell['runs'])} run arms) -> {p.name}")
    else:
        python = os.environ.get("KVA_PYTHON") or sys.executable
        n = kvabackend.run_cell(cell, cell_dir, python)
        log(f"harness wrote {n} records; records live under {cell_dir / 'work'}")

    # 7. push to BOTH destinations; only then is the cell done. The public code has no push.sh (the push targets
    # were private): there the results stay local, and the log says so.
    if (FINALS / "push.sh").exists():
        rc = os.system(f"'{FINALS / 'push.sh'}' '{cell_dir}'")
        if rc != 0:
            refuse(f"push.sh failed (exit {rc}): the cell is NOT marked done; results stay in {cell_dir} "
                   "and the repush cron will retry (PLAN.md: both destinations must succeed)")
    else:
        log(f"WARNING: no finals/push.sh in this checkout: the results are not pushed anywhere and stay in {cell_dir}")
    tree = atomic.treehash(cell_dir)
    atomic.write_bytes(cell_dir / ".done",
                         json.dumps({"cell": cell["name"], "registration": cell["registration"],
                                     "tree_sha256": tree, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                              time.gmtime())},
                                    indent=1).encode())
    log(f"cell {cell['name']} done: {len(ids)} samples, tree_sha256 {tree}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--new-host", action="store_true")
    ap.add_argument("--schema", default=str(FINALS / "cell.schema.json"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()