"""Paper section 4.8, Table 14: other users' throughput and inter-token latency in five mixed-serving schedules.
Recompute each repetition from the background requests' saved token timestamps and sent times in
results/qwen38-27b/results/mix/<schedule>/<arm>-*/result.json, following code/qwen38-27b-kit/mix.py and common.py:
throughput counts every delivered token over last token minus first request sent; ITLs pool adjacent-token gaps
over background users within the repetition; p99 is the ceil(0.99*n)-th ordered gap. Follow report.py mix_report():
equal-weight arithmetic means of the two repetitions' throughput, mean ITL and p99 ITL, but the maximum over both
repetitions' maximum ITL. No bootstrap. Print the Table 14 cells requested by group E (the three other p99 cells
already have sources). Human-readable diagnostics go to stderr; stdout contains only claim lines.
"""
import math
import statistics
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say       # noqa: E402

SCHEDULES = ("unified-c2", "unified-c4", "unified-c8", "separate-c2", "separate-c8")
P99_ARMS = {
    "unified-c2": ("stock",),
    "unified-c4": (),
    "unified-c8": ("stock", "e3-streamed"),
    "separate-c2": ("stock", "e3-streamed"),
    "separate-c8": ("stock", "e3-streamed"),
}


def comma_num(value, nd):
    """common.num's fixed precision with the paper's thousands separator."""
    whole, dot, fraction = num(value, nd).partition(".")
    return f"{int(whole):,}{dot}{fraction}"


def metrics(result):
    """Port the background-user portion of mix.summarize, using token timestamps rather than saved summaries."""
    background = [r for r in result["requests"] if r["role"] == "background"]
    if len(background) != result["concurrency"] - 1 or not background:
        raise SystemExit(f"{result['name']}: incomplete background request population")
    for r in background:
        times = r["token_times"]
        if len(times) != r["predict"] or times != sorted(times) or len(times) < 2:
            raise SystemExit(f"{result['name']} {r['id']}: incomplete or nonmonotone token timestamps")
    gaps = sorted(1000 * (b - a) for r in background for a, b in zip(r["token_times"], r["token_times"][1:]))
    # The original distribution() uses statistics.mean on unsorted gaps; statistics.mean is order-independent.
    values = dict(
        tps=sum(r["predict"] for r in background)
        / (max(r["token_times"][-1] for r in background) - min(r["sent"] for r in background)),
        itl_mean_ms=statistics.mean(gaps),
        itl_p99_ms=gaps[math.ceil(.99 * len(gaps)) - 1],
        itl_max_ms=gaps[-1],
    )
    saved = result["metrics"]
    checks = dict(tps=saved["other_aggregate_tps"], itl_mean_ms=saved["other_itl_ms"]["mean"],
                  itl_p99_ms=saved["other_itl_ms"]["p99"], itl_max_ms=saved["other_itl_ms"]["maximum"])
    for key, value in values.items():
        if not math.isclose(value, checks[key], rel_tol=1e-12, abs_tol=1e-12):
            raise SystemExit(f"{result['name']}: raw {key} {value} differs from stored summary {checks[key]}")
    return values


args = parse(__doc__)
M = args.root / "results" / "qwen38-27b" / "results" / "mix"
for schedule in SCHEDULES:
    paths = sorted(need(M / schedule).glob("*/result.json"))
    by_arm = {}
    for path in paths:
        result = load(path)
        by_arm.setdefault(result["arm"], []).append(result)
    if set(by_arm) != {"stock", "e3-streamed"}:
        raise SystemExit(f"{schedule}: missing or unexpected serving arm")
    for arm in ("stock", "e3-streamed"):
        part = by_arm[arm]
        if sorted(r["repetition"] for r in part) != [0, 1]:
            raise SystemExit(f"{schedule} {arm}: expected repetitions 0 and 1 exactly once")
        repetitions = [metrics(r) for r in part]
        values = {key: statistics.mean(r[key] for r in repetitions)
                  for key in ("tps", "itl_mean_ms", "itl_p99_ms")}
        values["itl_max_ms"] = max(r["itl_max_ms"] for r in repetitions)
        with redirect_stdout(sys.stderr):
            say(f"{schedule} {arm}: others {values['tps']:.9f} tok/s, "
                f"ITL mean/p99/max {values['itl_mean_ms']:.9f}/{values['itl_p99_ms']:.9f}/"
                f"{values['itl_max_ms']:.9f} ms")
        key = "E.mix." + schedule.replace("-", ".") + "." + arm
        claim(key + ".other_tps", num(values["tps"], 2))
        claim(key + ".other_itl_mean_ms", num(values["itl_mean_ms"], 2))
        if arm in P99_ARMS[schedule]:
            claim(key + ".other_itl_p99_ms", comma_num(values["itl_p99_ms"], 1))
        claim(key + ".other_itl_max_ms", comma_num(values["itl_max_ms"], 1))
