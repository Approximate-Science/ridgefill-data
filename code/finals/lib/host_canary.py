#!/usr/bin/env python3
"""The host canary for a finals cell: regenerated on every start of run_final.sh.

Per PLAN.md (per-host canary) and kva/canary.py's contract: the exact arm (and the plain noproj arm)
on the fixed 512-token prompt (datasets/corpus/prose.txt) must reproduce its stored logits hash
before any result is written. The finals reference is stored inside the cell's results dir together
with the GPU name and driver, so a resumed cell can only continue on hardware that reproduces it.

Mock mode (KVA_FINALS_MOCK=1): a deterministic fake (a hash of cell, host info and arm), so the CPU
pipeline test exercises the store/compare logic. Real mode reuses the repo's kva.canary run(),
kva.model.load and kva.gpu.setup() (the NVIDIA path: KVA_GPU_VENDOR=nvidia, KVA_GPU_NAME); it needs CUDA
(a rented NVIDIA box) and is NOT testable on a CPU.

Used by finals/lib/runner.py. CLI: host_canary.py --cell CELL.json --results DIR [--mock]
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

FINALS = Path(__file__).resolve().parents[1]


class CanarySetupError(RuntimeError):
    """The canary can not run on this host: no CUDA, no model, no driver info. Never skip it."""


class CanaryMismatch(RuntimeError):
    """The canary's logits differ from the stored reference: host RAM corruption or a changed engine
    (kva/canary.py CanaryMismatch). Void the cell and rerun; never patch results."""


def host_info(mock):
    """{host, machine, gpu_name, driver, torch, commit}: the fingerprint's hardware half."""
    if mock:
        return {"host": os.environ.get("KVA_MOCK_HOST", socket.gethostname()),
                "machine": platform.machine(),
                "gpu_name": os.environ.get("KVA_MOCK_GPU", "mock-gpu"),
                "driver": os.environ.get("KVA_MOCK_DRIVER", "mock-driver"),
                "torch": "mock", "commit": "mock"}
    import torch
    if not torch.cuda.is_available():
        raise CanarySetupError("no CUDA device visible: the finals canary needs the rented GPU "
                               "(mock mode is KVA_FINALS_MOCK=1)")
    p = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError) as e:
        raise CanarySetupError(f"nvidia-smi gave no driver version: {e}") from e
    repo = os.environ.get("KVA_REPO")
    if not repo:
        raise CanarySetupError("KVA_REPO is not set: the canary records the harness commit")
    sys.path.insert(0, repo)
    from kva import prov                                     # noqa: E402  (needs KVA_REPO on the path)
    return {"host": socket.gethostname(), "machine": platform.machine(),
            "gpu_name": p.name, "driver": driver, "torch": torch.__version__,
            "commit": prov.COMMIT_AT_START}


def fingerprint(info):
    """One id for a host's hardware+engine state; run_final.sh refuses a resume when it changes."""
    h = hashlib.sha256()
    for k in ("host", "machine", "gpu_name", "driver", "torch", "commit"):
        h.update(f"{k}={info[k]}\n".encode())
    return h.hexdigest()


def _compute_mock(cell, info):
    arms = {}
    for arm in ("exact", "noproj"):
        seed = f"mock|{cell['name']}|{cell['model']['name']}|{info['gpu_name']}|{info['driver']}|{arm}"
        arms[arm] = {"logits_sha256": hashlib.sha256(seed.encode()).hexdigest()}
    return arms


def _compute_real(cell, info):
    """The repo's canary (kva/canary.py): exact + noproj on the fixed 512-token prompt, on the device and with the
    deterministic flags kva/gpu.py sets up (KVA_GPU_VENDOR=nvidia + KVA_GPU_NAME on the rented box, as evaluate.py)."""
    sys.path.insert(0, os.environ["KVA_REPO"])
    from kva import canary as K, data, gpu, model as M
    model_dir = Path(os.environ.get("KVA_MODEL_DIR", "")) / cell["model"]["name"]
    if not model_dir.is_dir():
        raise CanarySetupError(f"model dir {model_dir} does not exist (KVA_MODEL_DIR / "
                               f"{cell['model']['name']}): pull the model before the canary runs")
    device, _ = gpu.setup()
    tok = data.tokenizer(str(model_dir))
    m = M.load(model_dir, device=device)
    try:
        return K.run(m, K.prompt_ids(tok))
    finally:
        # release the model before the runner starts evaluate.py: PyTorch's caching allocator would otherwise keep
        # its memory reserved in this process (a second copy of the weights; Qwen3.5-9B OOMs a 40 GB card, 2026-10-07)
        import gc
        import torch
        del m
        gc.collect()
        torch.cuda.empty_cache()


def run(cell, results_dir, mock):
    """Regenerate the canary; store it on the first start of this cell on this host, compare after.

    Returns (canary_path, arms) so the runner can link the sample records to it.
    """
    info = host_info(mock)
    arms = _compute_mock(cell, info) if mock else _compute_real(cell, info)
    rec = {"model": cell["model"]["name"], "engine": "reference", "host": info,
           "fingerprint": fingerprint(info), "arms": arms,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = Path(results_dir) / "canary.json"
    if path.exists():
        stored = json.loads(path.read_text())
        got = {a: v["logits_sha256"] for a, v in arms.items()}
        want = {a: v["logits_sha256"] for a, v in stored.get("arms", {}).items()}
        if got != want:
            diff = [a for a in set(got) | set(want) if got.get(a) != want.get(a)]
            raise CanaryMismatch(
                f"canary mismatch ({path.name}): arms {diff} differ from the stored reference "
                f"(stored host {stored['host']['host']}/{stored['host']['gpu_name']}/driver "
                f"{stored['host']['driver']}, now {info['host']}/{info['gpu_name']}/{info['driver']}). "
                f"Host RAM corruption or a changed engine: void the cell and rerun (PLAN.md contingencies)")
    else:
        sys.path.insert(0, str(FINALS / "lib"))
        from atomic import write_result
        write_result(path, rec)
    return path, arms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    p, a = run(json.loads(Path(args.cell).read_text()), args.results, args.mock)
    print("canary ok:", p, " ".join(sorted(a)))