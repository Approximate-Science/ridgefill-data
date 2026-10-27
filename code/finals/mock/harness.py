#!/usr/bin/env python3
"""The mock harness: deterministic fake results so the whole finals pipeline (validate -> canary ->
run -> atomic write -> push -> verify) runs on CPU in under a minute.

It never opens sample texts: every metric is a function of sha256(cell | sample id | arm | split), so
two runs produce byte-identical records (the oracle in test_pipeline.sh relies on that). It stands in
for the reference harness only; it is NOT evidence and never touches a model (nothing from a mock run
is a result of the paper - PLAN.md Phase 2 dry-run rule).

Imported by finals/lib/runner.py (mock backend). Sample ids come from the locked-set MANIFEST.tsv:
  ruler, longbench, secondq <manifest path>#<row index, 3 digits> (secondq: niah_multikey_1 files)
  nll, decode               <manifest path> (one sample per long-prompts file)
  speed                     <manifest path>#rep<0..4> (OUTLINE.md T3.1: 5 repeats)
Cell v2: every run's arms, keyed <run id>/<arm>.
"""
import hashlib
import json


def _h(*parts):
    """A stable integer seed from the parts (the fake metric generator)."""
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


def _metrics(set_name, seed):
    """The set's fake metric block: fixed shape, values from the seed."""
    r4 = lambda n, lo, hi: round(lo + (seed % n) / n * (hi - lo), 4)          # noqa: E731
    if set_name == "ttft":
        return {"ttft_s": r4(997, 4.0, 40.0), "peak_gib": r4(31, 2.0, 24.0)}
    if set_name == "longppl":
        return {"nll": r4(997, 1.8, 3.6), "novel_nll": r4(313, 1.5, 4.0), "key_token_nll": r4(503, 1.2, 5.0)}
    if set_name == "decode":
        return {"kl_mean": r4(997, 0.0, 0.5), "frac_identical": r4(313, 0.2, 1.0)}
    if set_name == "multiturn":
        return {"turns": [{"turn": 1, "correct": bool(seed % 2)}, {"turn": 2, "correct": bool(seed % 3)}]}
    return {"score": r4(20003, 0.0, 1.0)}                                     # ruler, longbench


def sample_ids(cell, manifest):
    """Sample ids for a cell kind from the locked-set manifest ({path: {sha256, rows}})."""
    kind = cell["kind"]
    paths = sorted(p for p in manifest if p.startswith(("ruler/", "longbench/", "long-prompts/")))
    if kind.startswith("ruler") or kind == "longbench":            # ruler, ruler-deep (v1.1.3), longbench
        prefix = "ruler/" if kind.startswith("ruler") else "longbench/"
        return [f"{p}#{i:03d}" for p in paths if p.startswith(prefix) for i in range(manifest[p]["rows"])]
    if kind == "secondq":
        return [f"{p}#{i:03d}" for p in paths if p.startswith("ruler/niah_multikey_1_") for i in range(manifest[p]["rows"])]
    if kind in ("nll", "nll-causal", "nll-fresh", "decode"):
        return [p for p in paths if p.startswith("long-prompts/")]
    if kind == "speed":
        return [f"{p}#rep{r}" for p in paths if p.startswith("long-prompts/") for r in range(5)]
    raise ValueError(f"unknown cell kind {kind!r}")


def source_sha(sample_id, manifest):
    """The manifest sha256 of a sample's source file (before the '#...')."""
    rel = sample_id.split("#", 1)[0]
    if rel not in manifest:
        raise KeyError(f"sample {sample_id}: source file {rel!r} is not in the locked MANIFEST.tsv")
    return manifest[rel]["sha256"]


def run_sample(cell, sample_id, manifest, canary_sha256):
    """The whole pairing unit at once: every arm of every run of the cell, one record.

    Byte-identical for the same inputs (no clock, no random): the record is the oracle's fixed point.
    """
    arms = {}
    for run in cell["runs"]:
        for arm in run["arms"]:
            seed = _h(cell["name"], sample_id, run["id"], arm)
            arms[f"{run['id']}/{arm}"] = _metrics(run["set"], seed)
            if arm.endswith("+rows"):
                arms[f"{run['id']}/{arm}"]["n_rows_exact"] = seed % 101            # ~10% row budget
    rec = {
        "cell": cell["name"],
        "kind": cell["kind"],
        "sample": sample_id,
        "source_sha256": source_sha(sample_id, manifest),
        "canary_sha256": canary_sha256,
        "deterministic": True,
        "code_tag": cell["code_tag"],
        "arms": arms,
    }
    return json.loads(json.dumps(rec, sort_keys=True))        # key order fixed = byte-identical
