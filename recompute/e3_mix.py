"""S3: Qwen3.8-27B serving several users (results/qwen38-27b/results/mix/<schedule>/<arm>-c<C>-<rep>/): one 16K
request arrives while C other users decode, stock against E3 streamed, one GPU, two repetitions per cell; each value
is the mean of the two repetitions (cloud-kit report.py mix_report): the new request's TTFT, the other users'
throughput and their inter-token latency (mean, p99), and the change in the other users' throughput. Engine errors:
the request errors each run recorded (errors.json) plus server exits with a nonzero code."""
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say, sgn  # noqa: E402

SCHEDULES = ("unified-c1", "unified-c2", "unified-c4", "unified-c8", "separate-c2", "separate-c8")
args = parse(__doc__)
M = args.root / "results" / "qwen38-27b" / "results" / "mix"
gains, errors, runs = [], 0, 0
for sched in SCHEDULES:
    by_arm = {}
    for f in sorted(need(M / sched).glob("*/result.json")):
        r = load(f)
        by_arm.setdefault(r["arm"], []).append(r)
        server = need(M.parent / r["provenance"])
        errors += len(load(server / "errors.json")) + (load(server / "exit.json")["rc"] != 0)
        runs += 1
    v = {}
    for arm, part in by_arm.items():
        if len(part) != 2:
            raise SystemExit(f"{sched} {arm}: {len(part)} repetitions, expected 2")
        m = [p["metrics"] for p in part]
        v[arm] = dict(ttft=statistics.mean(x["injector_ttft_s"] for x in m),
                      tps=statistics.mean(x["other_aggregate_tps"] for x in m) if m[0]["other_aggregate_tps"] else None,
                      p99=statistics.mean(x["other_itl_ms"]["p99"] for x in m) if m[0]["other_itl_ms"] else None)
    s, e = v["stock"], v["e3-streamed"]
    key = sched.replace("unified-", "unified.").replace("separate-", "separate.")
    say(f"{sched}: new-request TTFT {s['ttft']:.4f} / {e['ttft']:.4f} s"
        + (f", others {s['tps']:.3f} / {e['tps']:.3f} tok/s ({100 * (e['tps'] / s['tps'] - 1):+.2f} %), p99 ITL "
           f"{s['p99']:.2f} / {e['p99']:.2f} ms" if s["tps"] else ""))
    claim(f"S3.{key}.ttft.stock", num(s["ttft"], 2))
    claim(f"S3.{key}.ttft.e3", num(e["ttft"], 2))
    if s["tps"]:
        gains.append(100 * (e["tps"] / s["tps"] - 1))
        if sched.startswith("unified"):
            claim(f"S3.{key}.p99_ms.stock", num(s["p99"], 0))
            claim(f"S3.{key}.p99_ms.e3", num(e["p99"], 0))
say(f"other users' throughput change {min(gains):+.2f} % to {max(gains):+.2f} %; engine errors {errors} in {runs} runs")
claim("S3.others_tps_gain_pct.min", sgn(min(gains), 1))
claim("S3.others_tps_gain_pct.max", sgn(max(gains), 1))
claim("S3.engine_errors", errors)
