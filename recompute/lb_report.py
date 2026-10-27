"""LB4 (overall) and S4: Qwen3.8-27B LongBench v2, official pred.py CoT protocol at full length, 503 items, from the
per-item records (results/qwen38-27b/longbench/{stock,e3}/<id>.json; the stored judge field, the gold answers are not
needed). Re-implements the benchmark report (code/qwen38-27b-kit/longbench/report_lb.py) line for line and checks that
its output equals results/qwen38-27b/longbench/REPORT.txt byte for byte. The paired interval is that report's
bootstrap: random.Random(20261025), rng.choices over the item differences in sorted-id order, 10,000 resamples,
boot[250] / boot[9750]; the sign test is exact on the discordant items. S4 is the workload prefill speedup (sum of the
pass-1 prompt times, stock over E3)."""
import math
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say, sgn  # noqa: E402

args = parse(__doc__)
out = args.root / "results" / "qwen38-27b" / "longbench"
arms = {arm: {p.stem: load(p) for p in sorted(need(out / arm).glob("*.json"))} for arm in ("stock", "e3")}
ids = sorted(set(arms["stock"]) & set(arms["e3"]))
lines = [f'paired items {len(ids)} (stock {len(arms["stock"])}, e3 {len(arms["e3"])})\n']


def accuracy(rows):
    return 100 * sum(r["judge"] for r in rows) / len(rows) if rows else float("nan")


groups = [("overall", lambda r: True)]
groups += [(g, lambda r, g=g: r["difficulty"] == g) for g in ("easy", "hard")]
groups += [(g, lambda r, g=g: r["length"] == g) for g in ("short", "medium", "long")]
lines.append("| Group | n | Stock % | E3 % | Delta pp |\n|---|---:|---:|---:|---:|")
for name, keep in groups:
    sel = [i for i in ids if keep(arms["stock"][i])]
    s, e = accuracy([arms["stock"][i] for i in sel]), accuracy([arms["e3"][i] for i in sel])
    lines.append(f"| {name} | {len(sel)} | {s:.1f} | {e:.1f} | {e - s:+.1f} |")
diff = [int(arms["e3"][i]["judge"]) - int(arms["stock"][i]["judge"]) for i in ids]
rng = random.Random(20261025)
boot = sorted(100 * sum(rng.choices(diff, k=len(diff))) / len(diff) for _ in range(10000))
only_s, only_e = diff.count(-1), diff.count(1)
n = only_s + only_e
p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_s, only_e) + 1)) / 2 ** n) if n else 1.0
lines.append(f"\npaired E3 - stock: {100 * sum(diff) / len(diff):+.2f} pp, 95% bootstrap CI "
             f"[{boot[250]:+.2f}, {boot[9750]:+.2f}] (10,000 resamples)")
lines.append(f"only stock correct {only_s}, only E3 correct {only_e}, exact sign test p = {p:.3f}")
diag = {}
for arm in ("stock", "e3"):
    rows = [arms[arm][i] for i in ids]
    diag[arm] = (sum(r["pred"] is None for r in rows), sum(r["pass1_finish"] == "length" for r in rows))
    lines.append(f'{arm}: no extracted answer {diag[arm][0]}, pass-1 length stops {diag[arm][1]}, pass-2 length stops '
                 f'{sum(r["pass2_finish"] == "length" for r in rows)}, '
                 f'empty pass 1 {sum(r["empty_pass1"] for r in rows)}')
prefill = [(arms["stock"][i]["pass1_timings"]["prompt_ms"], arms["e3"][i]["pass1_timings"]["prompt_ms"],
            arms["stock"][i]["n_prompt_tokens"]) for i in ids
           if arms["stock"][i].get("pass1_timings") and arms["e3"][i].get("pass1_timings")]
speedup = None
BANDS = (("all", 0, 10 ** 9), ("<32K", 0, 32768), ("32-100K", 32768, 100000), (">=100K", 100000, 10 ** 9))
for name, low, high in BANDS:
    sel = [(s, e) for s, e, t in prefill if low <= t < high]
    if sel:
        sp = sum(s for s, _ in sel) / sum(e for _, e in sel)
        speedup = sp if name == "all" else speedup
        lines.append(f"prefill {name}: n {len(sel)}, stock {sum(s for s, _ in sel) / 1000:.0f} s total, "
                     f"E3 {sum(e for _, e in sel) / 1000:.0f} s total, speedup {sp:.3f}x")
text = "\n".join(lines) + "\n"
for line in text.splitlines():
    say(line)
same = text == need(out / "REPORT.txt").read_text()
say("REPORT.txt reproduced byte for byte" if same else "REPORT.txt DIFFERS from the recomputed report")
claim("LB4.report_identical", "yes" if same else "no")
claim("LB4.n", len(ids))
claim("LB4.stock", num(accuracy([arms["stock"][i] for i in ids]), 2))
claim("LB4.e3", num(accuracy([arms["e3"][i] for i in ids]), 2))
claim("LB4.delta", sgn(100 * sum(diff) / len(diff), 2))
claim("LB4.ci_lo", sgn(boot[250], 2))
claim("LB4.ci_hi", sgn(boot[9750], 2))
claim("LB4.only_stock", only_s)
claim("LB4.only_e3", only_e)
claim("LB4.sign_test_p", f"{p:.3f}")
for arm in ("stock", "e3"):
    claim(f"LB4.no_answer.{arm}", diag[arm][0])
    claim(f"LB4.pass1_cap.{arm}", diag[arm][1])
claim("S4.workload_prefill_speedup", f"{speedup:.3f}")
say(f"S4: items whose two arms ran on the same GPU index: "
    f"{sum(arms['stock'][i]['gpu'] == arms['e3'][i]['gpu'] for i in ids)} of {len(ids)}")
