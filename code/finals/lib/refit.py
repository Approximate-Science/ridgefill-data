#!/usr/bin/env python3
"""The clean refit on the rented box (PLAN.md correction 2026-10-04: the Llama and Qwen3.5 refits are the first job on
the rented box): the fit_sweep.py lines of jobs/paper-refit-clean.jobs (the single source of the models and splits),
run in the finals image with each model pulled from HF at its pinned commit (finals/make_cells.py MODELS) into
KVA_MODEL_DIR, the projectors written to KVA_DATA/proj (= KVA_PROJ_DIR in the image) and the sums to KVA_DATA/sums.
The calibration list is the committed one (datasets/graded-and-calibration-sets minus datasets/calibration-exclude.json;
its clean rows equal the private copy's: 834,541 Llama, 839,652 Qwen3.5, checked 2026-10-04). After each fit every
projector of the line is checked as the cells check it (finals/lib/kvabackend.py check_projector: metadata rows = the
clean count, calib_exclude_sha256 = the committed list's); a mismatch exits non-zero by name.
--dry-run prints the plan (models, commits, commands, checks) without pulling or fitting (the CPU test).
Env: KVA_REPO, KVA_DATA, KVA_MODEL_DIR (required), KVA_GPU_VENDOR/KVA_GPU_NAME (kva/gpu.py), HF_TOKEN (Llama is gated).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

FINALS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINALS / "lib"))
sys.path.insert(0, str(FINALS))


def docs_count(repo, model, layer, sha256):
    """The calibration document count behind the projector with this sha256: its fit report's 'docs' (fit_sweep.py
    records the count since e86e7a1; a list counts its names) or, for a fit that predates the field (the Qwen3-8B
    sweep, cba2d91), the docs list results/calib-docs-<model>-sweep.json took from the fit's own capture state
    (scripts/calib_docs_of_sums.py), which must name this sha256. None when neither records it."""
    for path, field in ((Path(repo) / "results" / f"fit-sweep-{model}-s{layer}.json", "projector_sha256"),
                        (Path(repo) / "results" / f"calib-docs-{model}-sweep.json", "projectors")):
        if not path.is_file():
            continue
        rep = json.loads(path.read_text())
        pinned = rep.get(field)
        if sha256 in ((pinned or {}).values() if isinstance(pinned, dict) else [pinned]) and "docs" in rep:
            return rep["docs"] if isinstance(rep["docs"], int) else len(rep["docs"])
    return None


def plan(repo):
    """[(job name, model name, splits)] from jobs/paper-refit-clean.jobs."""
    out = []
    for line in (repo / "jobs" / "paper-refit-clean.jobs").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, script, model_path, splits = line.split()
            if script != "scripts/fit_sweep.py":
                sys.exit(f"refit: {name} runs {script}, not scripts/fit_sweep.py")
            out.append((name, Path(model_path).name, [int(s) for s in splits.split(",")]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="refit just this model (a name in jobs/paper-refit-clean.jobs), one box per model")
    args = ap.parse_args()
    missing = [v for v in ("KVA_REPO", "KVA_DATA", "KVA_MODEL_DIR") if not os.environ.get(v)]
    if missing:
        sys.exit(f"refit: refusing: {', '.join(missing)} not set")
    repo, data, models = (Path(os.environ[v]) for v in ("KVA_REPO", "KVA_DATA", "KVA_MODEL_DIR"))
    import make_cells
    import kvabackend
    excl = hashlib.sha256((repo / "datasets" / "calibration-exclude.json").read_bytes()).hexdigest()
    steps = plan(repo)
    if args.only:
        steps = [x for x in steps if x[1] == args.only]
        if not steps:
            sys.exit(f"refit: refusing: --only {args.only} is not a model in jobs/paper-refit-clean.jobs")
    for name, model, splits in steps:
        m = make_cells.MODELS[model]
        prov = dict(calib_rows=make_cells.CALIB_ROWS[model], calib_docs=make_cells.CALIB_DOCS, calib_exclude_sha256=excl,
                    fitted_after=make_cells.FITTED_AFTER)
        cmd = [sys.executable, str(repo / "scripts" / "fit_sweep.py"), str(models / model), ",".join(map(str, splits))]
        print(f"refit {name}: {m['id']} at {m['commit']} -> {models / model}; {' '.join(cmd)}; then check "
              f"{len(splits)} projectors in {data / 'proj'} for rows {prov['calib_rows']}", flush=True)
        if args.dry_run:
            continue
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=m["id"], revision=m["commit"], local_dir=str(models / model),
                          token=os.environ.get("HF_TOKEN"))
        rc = subprocess.run(cmd, env=dict(os.environ, KVA_WORK=str(data / "refit-work"))).returncode
        if rc:
            sys.exit(f"refit {name}: fit_sweep.py exited {rc}")
        for s in splits:
            kvabackend.check_projector(data / "proj" / f"{model}-sweep-s{s}.safetensors", prov)
        print(f"refit {name}: {len(splits)} clean projectors checked", flush=True)


if __name__ == "__main__":
    main()
