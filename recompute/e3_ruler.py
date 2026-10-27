"""LB3: Qwen3.8-27B RULER, 13 tasks x 20 items at 16,384 and 32,768 tokens, stock against E3, from the stored per-item
scores (results/qwen38-27b/results/ruler/{stock,e3}/<id>.json; the kit scored each answer with the vendored official
scorer when it ran). Item list and order: provenance/inputs/qwen38-27b/ruler-index.json (ids, task, length,
max_tokens). The bootstrap is the kit's (cloud-kit common.bootstrap, report.py / ruler_replay.summarize): paired item
differences resampled within task with random.Random(1), 10,000 resamples, the mean of the 13 task means, percentile
interval bs[250] / bs[9750]. Checks: max_tokens equals the official budget per task, the prompts (messages sha256) are
identical across arms, and how many answers stopped at the length cap."""
import math
import random
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, num, parse, say, sgn       # noqa: E402

BUDGET = dict(niah=128, vt=30, cwe=120, fwe=50, qa=32)


def budget(task):
    return BUDGET["niah" if task.startswith("niah") else "qa" if task.startswith("qa") else task]


def bootstrap(groups, repeats=10000, seed=1):
    rng = random.Random(seed)
    bs = sorted(sum(sum(rng.choice(g) for _ in g) / len(g) for g in groups) / len(groups) for _ in range(repeats))
    return (statistics.mean(statistics.mean(g) for g in groups),
            bs[int(.025 * repeats)], bs[min(repeats - 1, int(.975 * repeats))])


args = parse(__doc__)
Q = args.root / "results" / "qwen38-27b" / "results" / "ruler"
index = load(args.root / "provenance" / "inputs" / "qwen38-27b" / "ruler-index.json")
recs = []
ok_budget = ok_prompt = True
for item in index:
    pair = {a: load(Q / a / (item["id"] + ".json")) for a in ("stock", "e3")}
    for a, r in pair.items():
        if r["id"] != item["id"] or r["task"] != item["task"] or r["length"] != item["length"]:
            raise SystemExit(f"{a}/{item['id']}.json is not the indexed item")
        ok_budget &= r["request"]["max_tokens"] == item["max_tokens"] == budget(item["task"])
    ok_prompt &= pair["stock"]["request"]["messages"]["sha256"] == pair["e3"]["request"]["messages"]["sha256"]
    recs.append(dict(item, **{a: r["score"] for a, r in pair.items()},
                     **{a + "_finish": r["finish_reason"] for a, r in pair.items()}))
claim("LB3.n_items", len(recs))
claim("LB3.max_tokens_official", "yes" if ok_budget else "no")
claim("LB3.prompts_identical", "yes" if ok_prompt else "no")
niah_min = math.inf
for length in (16384, 32768):
    groups, per = [], {}
    tasks = sorted({r["task"] for r in recs})
    for task in tasks:
        part = [r for r in recs if r["length"] == length and r["task"] == task]
        if len(part) != 20:
            raise SystemExit(f"{length} {task}: {len(part)} items, expected 20")
        groups.append([r["e3"] - r["stock"] for r in part])
        per[task] = part
        means = {a: statistics.mean(r[a] for r in part) for a in ("stock", "e3")}
        stops = {a: sum(r[a + "_finish"] == "length" for r in part) for a in ("stock", "e3")}
        d, lo, hi = bootstrap([groups[-1]])
        say(f"{length} {task:16s} stock {means['stock']:7.3f} e3 {means['e3']:7.3f} delta {d:+.2f} "
            f"[{lo:+.2f}, {hi:+.2f}] length stops {stops['stock']}/{stops['e3']}")
        for a in ("stock", "e3"):
            claim(f"LB3.{length}.{task}.{a}", num(means[a], 2))
        if task.startswith("niah"):
            niah_min = min(niah_min, means["stock"], means["e3"])
        if task == "cwe":
            claim(f"LB3.{length}.cwe.delta", sgn(d, 1))
            claim(f"LB3.{length}.cwe.ci_lo", sgn(lo, 1))
            claim(f"LB3.{length}.cwe.ci_hi", sgn(hi, 1))
            for a in ("stock", "e3"):
                claim(f"LB3.{length}.cwe.length_stops.{a}", stops[a])
        if task == "vt":
            for a in ("stock", "e3"):
                claim(f"LB3.{length}.vt.length_stops.{a}", stops[a])
    d, lo, hi = bootstrap(groups)
    say(f"{length} overall: delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    claim(f"LB3.{length}.delta", sgn(d, 2))
    claim(f"LB3.{length}.ci_lo", sgn(lo, 2))
    claim(f"LB3.{length}.ci_hi", sgn(hi, 2))
    for a in ("stock", "e3"):
        m, lo, hi = bootstrap([[r[a] for r in per[t]] for t in tasks])
        say(f"{length} {a}: {m:.4f} [{lo:.4f}, {hi:.4f}]")
        claim(f"LB3.{length}.{a}", num(m, 2))
        claim(f"LB3.{length}.{a}.ci_lo", num(lo, 2))
        claim(f"LB3.{length}.{a}.ci_hi", num(hi, 2))
claim("LB3.niah_min", num(niah_min, 2))
