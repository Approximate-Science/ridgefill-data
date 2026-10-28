"""Sections 5.1 and 9: Flash-Next host-resident experts per GPU and the 27B exact-prefix-row share range.

Extract the expert count (excluding empty slots) from exact's raw expert offload plan v2 boot log. For the 27B,
follow recompute/fig_speedup_gap.py: median n_exact_prefix over the three timed resident requests at each length,
divided by prefix length (prompt length minus the exact tail); report the extrema across the four timing lengths.
Warm-up requests are excluded. Human diagnostics go to stderr; stdout contains only claims.
"""
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from statistics import median

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say       # noqa: E402

args = parse(__doc__)


def describe(*parts):
    with redirect_stdout(sys.stderr):
        say(*parts)


boot = need(args.root / "results" / "flashnext" / "final-det-20261010" / "results" / "exact"
            / "boot" / "server-boot.log")
log = re.sub(r"\x1b\[[0-9;]*m", "", boot.read_text(encoding="utf-8"))
counts = re.findall(r"host rows \(per rank\)\s+[\d.]+ GiB\s+(\d+) experts \+ \d+ holes", log)
if len(counts) != 1:
    raise SystemExit(f"{boot}: expected one host-expert count from the v2 plan, got {len(counts)}")
experts = int(counts[0])
describe(f"Flash-Next expert offload plan v2: {experts:,} host-resident experts per rank/GPU")
claim("F.flashnext.host_experts_per_gpu", f"{experts:,}")

timing = need(args.root / "results" / "qwen38-27b" / "results" / "timing")
shares = []
for length in (16384, 32768, 65536, 120000):
    path = timing / str(length) / "resident" / "result.json"
    rows = [r for r in load(path)["rows"] if r["phase"].startswith("timed-")]
    if len(rows) != 3:
        raise SystemExit(f"{path}: expected three timed requests, got {len(rows)}")
    metrics = [r["response"]["timings"]["ridgefill"] for r in rows]
    prefixes = {r["n_prefix"] for r in metrics}
    if len(prefixes) != 1 or any(r["n_prompt"] != length or r["n_prefix"] != length - r["n_exact_tail"]
                                 for r in metrics):
        raise SystemExit(f"{path}: inconsistent prompt/prefix/tail lengths")
    prefix = prefixes.pop()
    share = 100 * median(r["n_exact_prefix"] for r in metrics) / prefix
    shares.append(share)
    describe(f"27B {length}: median exact prefix rows / prefix = {share:.9f} %")
claim("F.qwen38.exact_prefix_share_min", num(min(shares), 0))
claim("F.qwen38.exact_prefix_share_max", num(max(shares), 0) + " %")
