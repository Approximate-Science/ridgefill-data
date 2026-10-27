"""LB7a, S5 (speed): Qwen3.8-Flash-Next time to first token, one warm-up and five timed requests per length, each arm
alternating with the exact path on the same server start; speedup = median exact TTFT / median arm TTFT. Also the
spread of the exact path's median across the three server starts (exact's own, RidgeFill's, quality mode's).
--ruler64k (LB7b, S5): median request time (wall_s) over the 25 single-needle RULER samples at 64K, each arm on its
own server start, and exact's median over the arm's."""
import sys
from pathlib import Path
from statistics import median

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say      # noqa: E402

ARMS = {"ridgefill": "ridgefill", "quality": "graded"}       # claim key -> result folder
args = parse(__doc__, lambda p: p.add_argument("--ruler64k", action="store_true"))
F = args.root / "results" / "flashnext" / "final-det-20261010" / "results"


def timed(runs):
    xs = [r["ttft_s"] for r in runs if not r.get("warmup")]
    if len(xs) != 5 or len(runs) != 6:
        raise SystemExit(f"expected one warm-up and five timed requests, got {len(runs)} runs")
    return xs


if not args.ruler64k:
    spread = 0.0
    for L, key in ((9216, "9k"), (16384, "16k"), (32768, "32k")):
        exact_paths = [median(timed(load(F / "exact" / "speed" / f"speed-{L}.json")["runs"]["arm"]))]
        for arm, folder in ARMS.items():
            d = load(F / folder / "speed" / f"speed-{L}.json")
            ex, ar = median(timed(d["runs"]["exact-switch"])), median(timed(d["runs"]["arm"]))
            exact_paths.append(ex)
            say(f"{arm} {L}: exact path {ex:.3f} s / arm {ar:.3f} s = {ex / ar:.4f}x")
            claim(f"{'LB7a' if arm == 'ridgefill' else 'S5'}.{arm}.ttft_speedup.{key}", num(ex / ar, 2))
        spread = max(spread, max(exact_paths) / min(exact_paths) - 1)
        say(f"{L}: exact-path medians across the three starts {', '.join(f'{x:.3f}' for x in exact_paths)} s")
    say(f"largest spread of the exact path across starts: {100 * spread:.2f} %")
    claim("LB7b.exact_spread_pct", num(100 * spread, 1))
else:
    def wall(folder):
        files = sorted(need(F / folder / "ruler-a").glob("ruler-niah_single_1-65536-*.json"))
        if len(files) != 25:
            raise SystemExit(f"{folder}: {len(files)} 64K RULER records, expected 25")
        return median(load(f)["wall_s"] for f in files)
    exact = wall("exact")
    say(f"exact 64K median request time {exact:.3f} s")
    claim("LB7b.exact.wall_s", num(exact, 1))
    for arm, folder in ARMS.items():
        w = wall(folder)
        say(f"{arm} 64K median request time {w:.3f} s, speedup {exact / w:.4f}x")
        tag = "LB7b" if arm == "ridgefill" else "S5"
        claim(f"{tag}.{arm}.wall_s", num(w, 1))
        claim(f"{tag}.{arm}.speedup_64k", num(exact / w, 2))
