"""Job-start canary against host RAM corruption (2026-10-03: non-ECC RAM flips bits; weights pass through host
memory at load and the embedding table stays there).

One fixed prompt (BOS + the first tokens of the dev corpus' prose file, N in all) runs through the exact model and
one plain KVA arm; the logits of every row an arm computes exactly (exact: all N, noproj: its tail) are hashed and
compared with the reference stored per model, engine, protocol version and device (plus TORCH_BLAS_PREFER_HIPBLASLT
when set: it changes the GEMM kernels) in results/canary/. The KVA arm is noproj (identity fill at the middle layer,
exact tail TAIL): the KVA code path without a projector, whose bytes Projector.load checks by sha256.

Per host (paper T0.4, 2026-10-04): the key also carries a hash of the host's facts (host_facts: GPU name and arch, GPU
driver, torch and its CUDA/HIP runtime), recorded in the reference file ("host") and in every variant's env. A new
host (a cloud card, another driver or torch) therefore regenerates its own reference on its first run instead of being
held to bitwise agreement with another host's logits; within a host nothing changes. The in-kernel amdgpu driver is
identified by the kernel release, so a kernel update also starts a new reference (its first run stores, and says so).
Legacy references (keyed without the host hash, written before this change) stay in use on the host that wrote them:
they are adopted when every stored variant was recorded with this process's torch and HIP versions (the only host facts
they hold; the device arch is in their name). They exist only for protocol <= 1.0.2, which the paper's finals (v1.1)
never read.

A new protocol version on a host that already has a reference (protocol v1.1, 2026-10-04; checks/phase0-NOTES.md 5.2).
The canary does not depend on the protocol version (the prompt, arms and engine are the same; on the development machine the
Llama v1.0.1 and v1.0.2 references are bitwise identical), but the key carries it, so a version bump used to leave the
new key empty and its first run stored whatever it computed, unchecked. Now a key with no reference is seeded from
the newest earlier protocol version's reference of the same model, engine, device and BLAS switch on the same host
(earlier_reference: the same host key, or a legacy file adoptable here): the run is checked against it exactly as
against its own (bitwise, or within TOL for VARIANT_MODELS), a mismatch raises CanaryMismatch and stores nothing, and
only a match writes the new key's file (its variants copied, "seeded_from" naming the source). Only a key with no
earlier reference on this host at all (a new host, model, device or engine) stores its first run unchecked, and its
log line says so.

The first run on a key with nothing to check against stores its result as the reference; a later mismatch raises
CanaryMismatch before the job writes any result. Bitwise, except for models with known numeric variants
(VARIANT_MODELS: Qwen3.5's per-process GDN variant, ledger CD4, fixed per process in about 1 of 3-4 processes): they
also pass when no scored log-prob of either arm moves by more than TOL from a stored variant. Mode 'accept' adds the
current result as a variant; 'reset' replaces every stored variant (an intended engine change).
"""
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path

import torch

from kva import data, prov
from kva.kva import Arm, check_finite, finish, run_lower, score_tokens

N = 512               # prompt tokens
TAIL = 128            # the noproj arm's exact tail
# max |delta log-prob| over the scored canary tokens for VARIANT_MODELS: CD4's variant drift is far below it, a 64-byte
# run of random bits gives NaN (NonFiniteLogits) or far more
TOL = 0.02
VARIANT_MODELS = ("qwen3.5-9b",)
ENGINE = "reference"  # evaluate.py runs kva/model.py (protocol.json's engine "reference")
SOURCE = data.DATASETS / "corpus" / "prose.txt"


class CanaryMismatch(RuntimeError):
    """The canary's logits differ bitwise from every stored variant: host RAM corruption (weights, buffers) or a
    changed engine (torch, ROCm, kernels). scripts/chain.sh relaunches the job once; a repeat is likely real."""


def tolerance(model):
    """TOL for a model with known numeric variants, None (bitwise) for the others."""
    return TOL if model in VARIANT_MODELS else None


def prompt_ids(tok):
    return (data.bos_ids(tok) + data.encode(tok, SOURCE.read_text()[:N * 16]))[:N]


def driver_version():
    """The GPU kernel driver: NVIDIA's from /proc/driver/nvidia/version, amdgpu's module version (or, for the in-kernel
    driver, the running kernel's release); 'none' on a host without either."""
    nvidia = Path("/proc/driver/nvidia/version")
    if nvidia.exists():
        return nvidia.read_text().splitlines()[0].strip()
    amdgpu = Path("/sys/module/amdgpu")
    if amdgpu.exists():
        v = amdgpu / "version"
        version = v.read_text().strip() if v.exists() else ""
        return f"amdgpu {version}" if version else f"amdgpu in-kernel {platform.release()}"
    return "none"


def host_facts(gpu_info):
    """What makes this host's logits its own (gpu.setup()'s gpu_info; a CPU run has no name or arch)."""
    return dict(gpu=gpu_info.get("name", "cpu"), arch=str(gpu_info.get("arch", "cpu")).split(":")[0],
                driver=gpu_info.get("driver") or driver_version(), torch=torch.__version__,
                runtime=f"hip {torch.version.hip}" if torch.version.hip else
                f"cuda {torch.version.cuda}" if torch.version.cuda else "cpu")


def host_key(facts):
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()[:12]


def _legacy_adoptable(path):
    """A pre-host-key reference written on this host: every variant has this process's torch and HIP versions."""
    try:
        variants = json.loads(Path(path).read_text())["variants"]
    except (OSError, ValueError, KeyError):
        return False
    return bool(variants) and all(v["env"].get("torch") == torch.__version__
                                  and v["env"].get("hip") == torch.version.hip for v in variants)


def reference_path(results, model, protocol, gpu_info):
    """results/canary/<model>-<engine>-v<protocol>-<arch>[-hipblaslt<x>]-host<key>.json, or the same name without
    -host<key> when that legacy file exists, this host wrote it (_legacy_adoptable) and the keyed one does not."""
    device = str(gpu_info.get("arch", "cpu")).split(":")[0]
    blas = os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT")
    name = f"{model}-{ENGINE}-v{protocol}-{device}" + (f"-hipblaslt{blas}" if blas is not None else "")
    keyed = Path(results) / "canary" / f"{name}-host{host_key(host_facts(gpu_info))}.json"
    legacy = Path(results) / "canary" / f"{name}.json"
    return legacy if not keyed.exists() and legacy.exists() and _legacy_adoptable(legacy) else keyed


_NAME = re.compile(rf"^(?P<head>.+-{ENGINE})-v(?P<ver>\d+(?:\.\d+)*)-(?P<rest>.+?)(?P<host>-host[0-9a-f]{{12}})?\.json$")


def earlier_reference(path):
    """The reference an empty key is seeded from: among the files next to path for the same model, engine, device and
    BLAS switch under an EARLIER numeric protocol version, written on this host (the same -host<key>, or a legacy file
    _legacy_adoptable here), the newest version (a keyed file before a legacy one); None if there is none."""
    path = Path(path)
    me = _NAME.match(path.name)
    if me is None or not path.parent.is_dir():
        return None
    key = tuple(int(x) for x in me["ver"].split("."))
    found = []
    for f in path.parent.iterdir():
        g = _NAME.match(f.name)
        if g is None or (g["head"], g["rest"]) != (me["head"], me["rest"]):
            continue
        ver = tuple(int(x) for x in g["ver"].split("."))
        if ver < key and (g["host"] == me["host"] or (g["host"] is None and _legacy_adoptable(f))):
            found.append((ver, g["host"] is not None, f.name, f))
    return max(found)[3] if found else None


@torch.no_grad()
def run(m, ids):
    """{arm: {logits_sha256, logprobs}} on prompt ids [N]: exact over all N rows (N - 1 scored tokens), noproj over
    its exact tail (TAIL - 1)."""
    ids = torch.tensor(ids)
    split = m.n_layers // 2
    hs, lower = run_lower(m, ids, {split})
    out = {}
    for arm, keep in ((Arm("exact", split=split), N), (Arm("noproj", split=split, tail=TAIL, fill="identity"), TAIL)):
        logits = check_finite(finish(m, ids, hs[split], lower, arm, keep, keep_cache=False)[0], f"canary {arm.name}")
        out[arm.name] = dict(logits_sha256=hashlib.sha256(logits.float().cpu().numpy().tobytes()).hexdigest(),
                             logprobs=[float(x) for x in score_tokens(logits, ids, keep - 1)])
        del logits
    return out


def _delta(variant, got, arm):
    return max(abs(x - y) for x, y in zip(variant[arm]["logprobs"], got[arm]["logprobs"]))


def _mismatch(path, ref, got, ids_sha, env, tol):
    """The error text: which arms differ, by how much (max |delta log-prob| vs the nearest variant), and what to do."""
    near = min(ref["variants"], key=lambda v: max(_delta(v, got, a) for a in got))
    arms = ", ".join(f"{a} {_delta(near, got, a):.3g}" for a in got)
    old = near["env"]
    return (f"canary mismatch ({path}): the logits differ " + ("bitwise" if tol is None else f"by more than {tol}")
            + f" from all {len(ref['variants'])} stored variant(s); max |delta log-prob| over the scored tokens vs the "
            f"nearest: {arms}"
            + ("; the canary prompt itself changed (tokenizer or corpus)" if ref["prompt_sha256"] != ids_sha else "")
            + f". Stored: torch {old['torch']} hip {old['hip']} card {old['card']} gpu {old.get('gpu', '?')} driver "
            f"{old.get('driver', '?')} commit {old['commit'][:8]}; now: torch {env['torch']} hip {env['hip']} card "
            f"{env['card']} gpu {env['gpu']} driver {env['driver']} commit {env['commit'][:8]}. "
            f"Host RAM corruption or a changed engine. Rerun with --canary reset after an intended engine change, "
            f"--canary accept for a known per-process variant (Qwen3.5 GDN, ledger CD4).")


def check(m, tok, path, gpu_info, mode="check", tol=None):
    """Run the canary and compare it with the variants stored at path (see the module doc for the modes); tol
    (tolerance()): the max |delta log-prob| a result that differs bitwise from every variant may still have.
    Returns the log line: which variant matched, bitwise or within tol."""
    path = Path(path)
    ids = prompt_ids(tok)
    ids_sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    got = run(m, ids)
    facts = host_facts(gpu_info)
    env = dict(torch=torch.__version__, hip=torch.version.hip, card=gpu_info.get("pci", "cpu"), gpu=facts["gpu"],
               driver=facts["driver"], runtime=facts["runtime"], commit=prov.COMMIT_AT_START,
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    ref = json.loads(path.read_text()) if path.exists() else None
    seed = earlier_reference(path) if ref is None and mode != "reset" else None
    if seed is not None:                     # an empty key on a host with an earlier version's reference: check it
        ref = json.loads(seed.read_text())
    where = path.name if seed is None else f"{seed.name} (an earlier protocol version on this host)"
    variants = [] if ref is None or mode == "reset" else ref["variants"]

    def store(note):
        prov.write_json(path, dict(model=Path(m.path).name, engine=ENGINE,
                                   source=str(SOURCE.relative_to(data.DATASETS.parent)), n=N, split=m.n_layers // 2,
                                   tail=TAIL, prompt_sha256=ids_sha, host=facts, variants=variants,
                                   **({"seeded_from": seed.name} if seed is not None else {})))
        return note + (f"; seeded {path.name} from it" if seed is not None else "")
    match = [i for i, v in enumerate(variants) if all(v[a]["logits_sha256"] == got[a]["logits_sha256"] for a in got)]
    if match:
        note = f"canary: bitwise match with variant {match[0] + 1} of {len(variants)} in {where}"
        return store(note) if seed is not None else note
    if variants and mode == "check":
        d, i = min((max(_delta(v, got, a) for a in got), i) for i, v in enumerate(variants))
        if tol is not None and d <= tol and ref["prompt_sha256"] == ids_sha:
            note = (f"canary: tolerance match with variant {i + 1} of {len(variants)} in {where} "
                    f"(max |delta log-prob| {d:.3g} <= {tol})")
            return store(note) if seed is not None else note
        raise CanaryMismatch(_mismatch(seed or path, ref, got, ids_sha, env, tol))
    variants.append(dict(got, env=env))
    first = "first run: no reference for this model, engine and device on this host under any protocol version"
    why = first if mode == "check" and seed is None else mode
    return store(f"canary: stored variant {len(variants)} in {path} ({why})")
