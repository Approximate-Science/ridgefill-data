"""Paper sections 2.4 and 2.5: realized exact-prefix shares, filled prompt shares, and skipped analytic MAC shares.

Flash-Next quality mode: min/max of 100*xrows/tail_start over graded/tails.json's approximated ppl-a documents
(exclude control, speed, and RULER). Qwen3.8-27B: per length, median over the three timed resident requests, excluding
cold and warmup, as recompute/fig_speedup_gap.py; selector share is 100*n_exact_prefix/n_prefix, filled share is
100*n_filled/n_prompt, and analytic MAC savings is 100*mac_share_skipped, including the recorded probe, maps and
value-rescale cost. The timing selector range is the min/max of the per-length medians across the four lengths.
All percentages round to whole percent as printed; no bootstrap. Claim lines go to stdout, evidence to stderr.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path
from statistics import median

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, num, parse, say             # noqa: E402

args = parse(__doc__)
tails = load(args.root / "results" / "flashnext" / "final-det-20261010" / "results" / "graded" / "tails.json")
protected = []
for name, record in sorted(tails["samples"].items()):
    if not name.startswith("ppl-a/"):
        continue
    if not record["approximated"] or record["tail_start"] <= 0:
        raise SystemExit(f"{name}: missing approximated prefix in quality-mode perplexity run")
    protected.append((100.0 * record["xrows"] / record["tail_start"], name))
if not protected:
    raise SystemExit("missing input: no graded ppl-a prefix-row records")
low, high = min(protected), max(protected)
with redirect_stdout(sys.stderr):
    say(f"Flash-Next quality mode, {len(protected)} documents: prefix protected share "
        f"{low[0]:.12g} % ({low[1]}) to {high[0]:.12g} % ({high[1]})")
claim("M.flashnext.protected_prefix_pct.min", num(low[0], 0))
claim("M.flashnext.protected_prefix_pct.max", num(high[0], 0) + " %")

timing = args.root / "results" / "qwen38-27b" / "results" / "timing"
prefix_shares = []
for length in (16384, 32768, 65536, 120000):
    result = load(timing / str(length) / "resident" / "result.json")
    timed = [r for r in result["rows"] if r["phase"].startswith("timed-")]
    if len(timed) != 3:
        raise SystemExit(f"{length} resident: expected three timed requests, found {len(timed)}")
    rows = [r["response"]["timings"]["ridgefill"] for r in timed]
    if any(r["n_prompt"] != length or r["n_prefix"] <= 0 or
           r["n_filled"] + r["n_exact_prefix"] != r["n_prefix"] or
           r["n_prefix"] + r["n_exact_tail"] != r["n_prompt"] for r in rows):
        raise SystemExit(f"{length} resident: inconsistent prompt/prefix/filled/exact row counts")
    prefix = median(100.0 * r["n_exact_prefix"] / r["n_prefix"] for r in rows)
    filled = median(100.0 * r["n_filled"] / r["n_prompt"] for r in rows)
    skipped = median(100.0 * r["mac_share_skipped"] for r in rows)
    prefix_shares.append(prefix)
    with redirect_stdout(sys.stderr):
        say(f"27B resident {length}: median exact prefix {prefix:.12g} %, "
            f"filled prompt {filled:.12g} %, skipped analytic MACs {skipped:.12g} %")
    if length in (16384, 120000):
        claim(f"M.qwen38_27b.resident.{length}.filled_prompt_pct", num(filled, 0) + " %")
        claim(f"M.qwen38_27b.resident.{length}.mac_skipped_pct", num(skipped, 0) + " %")
claim("M.qwen38_27b.resident.exact_prefix_pct.min", num(min(prefix_shares), 0))
claim("M.qwen38_27b.resident.exact_prefix_pct.max", num(max(prefix_shares), 0) + " %")
