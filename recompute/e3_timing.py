"""LB2: Qwen3.8-27B time to first token, warm, one A100 (results/qwen38-27b/results/timing/<length>/<arm>/). Per length
and arm: the server's prompt time of the three timed requests after a cold and a warm-up request, median of three;
speedup = stock median / arm median (resident: maps in VRAM; streamed: maps from host memory); peak VRAM over stock
(MiB, the sampler's peak for the server process). Checks: the stored medians equal the recomputed ones; stock and both
RidgeFill arms ran on the same physical GPU (the GPU label in the VRAM samples and the GPU index); the server command
lines are identical once the RidgeFill flags and the binary are removed."""
import sys
from pathlib import Path
from statistics import median

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, rows, say   # noqa: E402

args = parse(__doc__)
Q = args.root / "results" / "qwen38-27b" / "results"
LENGTHS = {n: str(n) for n in (16384, 32768, 65536, 120000)}


def server(length, arm):
    r = load(Q / "timing" / str(length) / arm / "result.json")
    folder = need(Q / r["provenance"])
    timed = [x for x in r["rows"] if x["phase"].startswith("timed-")]
    if len(r["rows"]) != 5 or len(timed) != 3:
        raise SystemExit(f"{length} {arm}: {len(r['rows'])} requests, {len(timed)} timed; expected 5 and 3")
    med = median(x["prompt_ms"] / 1000 for x in timed)
    if abs(med - r["warm_prompt_s"]["median"]) > 1e-9:
        raise SystemExit(f"{length} {arm}: recomputed median {med} differs from the stored "
                         f"{r['warm_prompt_s']['median']}")
    labels = {p["uuid"] for v in rows(folder / "vram.jsonl") for p in v["processes"]}
    peak = max([p["used_mib"] for v in rows(folder / "vram.jsonl") for p in v["processes"]] or [0])
    if peak != r["peak_mib"] or load(folder / "exit.json")["peak_mib"] != peak:
        raise SystemExit(f"{length} {arm}: peak VRAM {peak} MiB in vram.jsonl, {r['peak_mib']} in result.json")
    cmd = load(folder / "command.json")["command"][1:]
    flags, skip = [], False
    for i, a in enumerate(cmd):
        if skip:
            skip = False
            continue
        if a.startswith("--ridgefill"):
            skip = i + 1 < len(cmd) and not cmd[i + 1].startswith("-")
            continue
        flags.append(a)
    gpu = (load(folder / "provenance.json")["gpu"],
           load(folder / "command.json")["environment"]["CUDA_VISIBLE_DEVICES"])
    return dict(median=med, peak=peak, labels=labels, gpu=gpu, flags=flags)


for length, key in LENGTHS.items():
    s = {arm: server(length, arm) for arm in ("stock", "resident", "streamed")}
    claim(f"LB2.stock_s.{key}", num(s["stock"]["median"], 2))
    for arm in ("resident", "streamed"):
        x = s[arm]
        say(f"{length} {arm}: {x['median']:.3f} s vs stock {s['stock']['median']:.3f} s, speedup "
            f"{s['stock']['median'] / x['median']:.4f}x, VRAM {x['peak'] - s['stock']['peak']:+d} MiB")
        claim(f"LB2.{arm}_s.{key}", num(x["median"], 2))
        claim(f"LB2.{arm}.{key}", num(s["stock"]["median"] / x["median"], 3))
        claim(f"LB2.vram_mib.{arm}.{key}", f"{x['peak'] - s['stock']['peak']:+d}")
    same = len(set().union(*(x["labels"] for x in s.values()))) == 1 and len({x["gpu"] for x in s.values()}) == 1
    flags = all(x["flags"] == s["stock"]["flags"] for x in s.values())
    say(f"{length}: GPU labels {sorted(set().union(*(x['labels'] for x in s.values())))}, GPU index "
        f"{sorted({x['gpu'] for x in s.values()})}; non-RidgeFill flags identical: {flags}")
    claim(f"LB2.same_gpu.{key}", "yes" if same else "no")
    claim(f"LB2.flags_identical.{key}", "yes" if flags else "no")
