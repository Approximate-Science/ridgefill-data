"""LB6: the Qwen3.8-27B gates (results/qwen38-27b/results/checks/<run>/). CHECK-A: the fork with RidgeFill off (no
maps, and maps loaded with the mode off) against upstream: comparisons whose logits are byte-identical (prompts at
8K / 16K / 32K, two LongBench prompts, scorer NLL on two documents), with the two builds named by their server binary
and its sha256 in RUN-MANIFEST.json. CHECK-B: E3 resident against streamed, identical 128-token output, logits and
scorer NLL. Pure (measurement) against instrumented (gate) builds: identical 128-token outputs and top-10
probabilities. RUN-MANIFEST: the runtime diff between scorer and server commits is empty and the source archives'
hashes equal provenance/inputs/qwen38-27b/LOCK.json's."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, parse, say                 # noqa: E402

args = parse(__doc__)
Q = args.root / "results" / "qwen38-27b" / "results"
run = load(Q / "checks" / "PASS.json")["provenance"]
C = Q / run
manifest = load(Q / "RUN-MANIFEST.json")
a = load(C / "CHECK-A" / "result.json")
ident_a = [c["name"] for c in a["comparisons"] if c.get("byte_identical") or c.get("all512_logprobs_identical")]
builds = set()
for server in ("stock", "fork-off", "loaded-off"):
    prov = load(C / "CHECK-A" / server / "provenance.json")["binaries"]
    binary = load(C / "CHECK-A" / server / "command.json")["command"][0]
    rel = binary.split("/kit/", 1)[-1]
    if manifest["binaries"].get(rel) != prov[binary]:
        raise SystemExit(f"CHECK-A {server}: binary {rel} is not the build RUN-MANIFEST.json names")
    builds.add(rel)
say(f"CHECK-A identical: {len(ident_a)} of {len(a['comparisons'])} ({', '.join(ident_a)}); builds {sorted(builds)}")
claim("LB6.checkA.identical", len(ident_a))
claim("LB6.checkA.comparisons", len(a["comparisons"]))
named = {"build/instrumented/stock/bin/llama-server", "build/instrumented/e3/bin/llama-server"}
claim("LB6.checkA.builds", len(builds & named))
b = load(C / "CHECK-B" / "result.json")
ident_b = [c["length"] for c in b["comparisons"]
           if c["output128_identical"] and c["logits128_byte_identical"] and c["all512_logprobs_identical"]]
say(f"CHECK-B identical at {ident_b}")
claim("LB6.checkB.identical", len(ident_b))
claim("LB6.checkB.lengths", "/".join(str(x) for x in ident_b))
p = a["pure_vs_instrumented"]
ident_p = [x for x in p if x["tokens_identical"] and x["topk_probabilities_identical"]]
say(f"pure vs instrumented identical: {len(ident_p)} of {len(p)} (arms {sorted({x['arm'] for x in p})})")
claim("LB6.pure_vs_instrumented.identical", len(ident_p))
src = manifest["sources"]
lock = load(args.root / "provenance" / "inputs" / "qwen38-27b" / "LOCK.json")
claim("LB6.server_runtime_diff_empty", "yes" if src["server_runtime_diff_empty"] is True else "no")
claim("LB6.archives_match_lock", "yes" if src["archives"] == lock["archives"] else "no")
