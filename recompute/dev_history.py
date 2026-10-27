"""D1-D5: the development measurements of the paper's Appendix I (results/development/, status DEVELOPMENT).

D1 corrections from approximation theory and the fill error's correlation along the prompt; D2 needles in the filled
region and exact digit rows; D3 fill error by token class and how concentrated its cost is; D4 the decode-time check
on Qwen3.8-27B; D5 the needle suites against RULER on Qwen3.8-27B. Intervals are percentile bootstraps over the units
named in each section (10,000 resamples, seed 1, units in sorted order)."""
import random
import re
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, rows, say, sgn   # noqa: E402

args = parse(__doc__)
D = args.root / "results" / "development"


def boot(stat, n, repeats=10000, seed=1):
    """Percentile 95 % interval of stat(indices) over resamples of n units."""
    rng = random.Random(seed)
    bs = sorted(stat([rng.randrange(n) for _ in range(n)]) for _ in range(repeats))
    return bs[int(.025 * repeats)], bs[min(repeats - 1, int(.975 * repeats))]


def ci(lo, hi, nd):
    return f"[{sgn(lo, nd)}, {sgn(hi, nd)}]"


# D1. Closure corrections, scored as the relative change in attention-output error against the plain fill, averaged
# over the late attention layers, on the 23 development prompts (ratio of means; prompts are the bootstrap unit).
raw = load(D / "closures" / "k3k7_raw.json")["grade"]
prompts = sorted(raw)


def per_prompt(fix):
    return [statistics.mean(raw[p]["layers"][L][fix]["attn_rel"] for L in sorted(raw[p]["layers"], key=int))
            for p in prompts]


base = per_prompt("base")
for key, fix in (("jensen", "jensen+"), ("jensen_reversed", "jensen-"), ("jensen_per_row", "lev+"),
                 ("jensen_cross", "jensen+xc"), ("cross", "xc"), ("value_map", "valcal"),
                 ("temperature", "temp"), ("key_mean", "kmean")):
    v = per_prompt(fix)
    rel = lambda idx, v=v: sum(v[i] for i in idx) / sum(base[i] for i in idx) - 1   # noqa: E731
    lo, hi = boot(rel, len(v))
    better = sum(a < b for a, b in zip(v, base))
    say(f"D1 {fix}: {100 * rel(range(len(v))):+.1f} % [{100 * lo:+.1f}, {100 * hi:+.1f}], {better}/{len(v)} better")
    claim(f"D1.{key}.change", sgn(100 * rel(range(len(v))), 1))
    claim(f"D1.{key}.ci", ci(100 * lo, 100 * hi, 1))
    claim(f"D1.{key}.better", f"{better} of {len(v)}")

# The key-mean shift end to end: loss on novel tokens, with minus without the shift, both with the terminal-state
# correction at split 16 and tail 2048, on the same 23 prompts.
k15 = sorted(load(D / "closures" / "k15.json")["rows"], key=lambda r: r["name"])
d = [r["16/2048+st+km"] - r["16/2048+st"] for r in k15]
lo, hi = boot(lambda idx: statistics.mean(d[i] for i in idx), len(d))
claim("D1.key_mean_loss.n", len(d))
claim("D1.key_mean_loss.delta", sgn(statistics.mean(d), 4))
claim("D1.key_mean_loss.ci", ci(lo, hi, 4))

# Correlation of the fill error between a row and the row 1 or 8 positions earlier, mean over layers, per document.
log = need(D / "whiteness" / "diag_residual.log").read_text()
lag = {k: [float(x) for x in re.findall(rf"r\[t\]~r\[t-{k}\]: .*mean ([+-][0-9.]+)", log)] for k in (1, 8)}
if len(lag[1]) != 3 or len(lag[8]) != 3:
    raise SystemExit("diag_residual.log: expected three documents")
claim("D1.lag1.range", f"{num(min(lag[1]), 2)} to {num(max(lag[1]), 2)}")
claim("D1.lag8.max", num(max(lag[8]), 2))
# Exact rows spread through the prompt against the same number of exact rows in the tail.
rel = load(D / "whiteness" / "results_e4.json")["cross_fold"]["rel"]
cells = sorted(k[len("burst"):] for k in rel if k.startswith("burst"))
tail_wins = sum(rel["tail" + c] < rel["burst" + c] for c in cells)
claim("D1.spread.cells", len(cells))
claim("D1.spread.tail_wins", tail_wins)
claim("D1.spread.at25.spread", num(rel["burst0.25_c256"], 2))
claim("D1.spread.at25.tail", num(rel["tail0.25_c256"], 2))

# D2. Needles in the filled region on the development checkpoint (60 prompts, split 16, tails 2048 and 4096).
recs = [load(p) for p in sorted((D / "needles-dev" / "k20").glob("*.json"))]
wrong, kinds = 0, {"last": 0, "other": 0}
for arm in ("16/2048+st", "16/4096+st"):
    filled = correct = noanswer = 0
    for r in recs:
        run = next(x for x in r["runs"] if x["mode"] == arm)
        if not r["needle_end"] <= run["approx"]:
            continue
        filled += 1
        nums = re.findall(r"\d{7}", run["text"])
        if run["correct"]:
            correct += 1
        elif not nums:
            noanswer += 1
        else:
            wrong += 1
            diff = [i for i in range(7) if nums[0][i] != r["value"][i]]
            kinds["last" if diff == [6] else "other"] += 1
    t = arm.split("/")[1].split("+")[0]
    claim(f"D2.dev.t{t}.filled", filled)
    claim(f"D2.dev.t{t}.correct", correct)
    claim(f"D2.dev.t{t}.no_number", noanswer)
    tail = [next(x for x in r["runs"] if x["mode"] == arm) for r in recs]
    tail = [(x, r) for x, r in zip(tail, recs) if not r["needle_end"] <= x["approx"]]
    claim(f"D2.dev.t{t}.tail", f"{sum(x['correct'] for x, _ in tail)} of {len(tail)}")
claim("D2.dev.wrong", wrong)
claim("D2.dev.wrong_last_digit_only", kinds["last"])
exact = [next(x for x in r["runs"] if x["mode"] == "off") for r in recs]
claim("D2.dev.exact.wrong", sum(1 for x in exact if not x["correct"] and re.findall(r"\d{7}", x["text"])))
claim("D2.dev.exact.no_number", sum(1 for x in exact if not re.findall(r"\d{7}", x["text"])))
k28 = load(D / "needles-dev" / "k28_niah.json")
n = k28["n_projected"]
for arm, key in (("base", "fill"), ("D", "digits"), ("tailD", "tail")):
    claim(f"D2.digits.{key}", f"{round(k28['arms'][arm]['acc_projected'] * n)} of {n}")
share = [k28["budgets"][dom]["D"]["pct_prompt"] for dom in sorted(k28["budgets"])]
claim("D2.digits.share", f"{num(min(share), 1)} to {num(max(share), 1)}")

# Reference models: exact digit rows and exact attention-selected rows against the fill, on numeric multi-key needles
# and on non-numeric needles; units are (prompt, setting) pairs whose needle lies in the filled region.
R = D / "needles-reference"
for model in ("llama", "qwen"):
    for kind, folder in (("numeric", f"niah-bypass-mk4-{model}"), ("text", f"niahg-bypass-{model}")):
        pairs = {"digits": [], "qattn5": []}
        for path in sorted((R / folder).glob("*.json")):
            r = load(path)["runs"]
            for cfg in ("s12-t1024", "s16-t1024", "s16-t4096"):
                fill = r[f"kva-{cfg}"]
                if not fill["needle_projected"]:
                    continue
                for arm in pairs:
                    pairs[arm].append(int(r[f"kva-{cfg}-{arm}"]["correct"]) - int(fill["correct"]))
        for arm, v in pairs.items():                         # in points of retrieval
            lo, hi = boot(lambda idx, v=v: statistics.mean(v[i] for i in idx), len(v))
            claim(f"D2.ref.{model}.{kind}.{arm}.n", len(v))
            claim(f"D2.ref.{model}.{kind}.{arm}.delta", sgn(100 * statistics.mean(v), 0))
            claim(f"D2.ref.{model}.{kind}.{arm}.ci", ci(100 * lo, 100 * hi, 0))
# The plain fill on hexadecimal identifiers, Llama-3.1-8B at split 12, tail 1024.
hex_ok = [load(p)["runs"]["kva-s12-t1024"]["correct"] for p in sorted((R / "niahg-bypass-llama").glob("ng-hexid-*.json"))]
claim("D2.ref.llama.hexid.s12", f"{sum(hex_ok)} of {len(hex_ok)}")

# D3. Fill error by token class (split 12, 23 development prompts): the largest class mean over the smallest, the
# 23 special tokens left out; and the share of first-order impact held by each document's top 5 % of tokens.
for model in ("llama", "qwen"):
    by = {c: v for c, v in load(D / "token-classes" / f"{model}-s12.json")["by_class"].items() if c != "special"}
    err = [v["proj"] for v in by.values()]
    claim(f"D3.{model}.classes", len(by))
    claim(f"D3.{model}.ratio", num(max(err) / min(err), 2))
conc = [line.split(",") for line in need(D / "token-classes" / "te2-concentration.csv").read_text().splitlines()]
head, conc = conc[0], conc[1:]
for model, tag in (("A", "llama"), ("B", "qwen")):
    top5 = [float(r[head.index("top5")]) for r in conc if r[0] == model]
    claim(f"D3.{tag}.top5", f"{num(100 * statistics.mean(top5), 0)} %")
    lin = load(D / "token-classes" / f"te2-{tag}-s12-t1024.json")["summary"]["linearisation"]
    claim(f"D3.{tag}.first_order_over_measured", num(sum(lin["sum_impact"]) / sum(lin["d_attn"]), 1))

# D4. The decode-time check on Qwen3.8-27B: the full per-row scan's decoding overhead (median of three rounds, two
# runs) and the per-token miss rate of the periodic checks (controls left out).
C = D / "certificate-27b"
for length in ("16384", "32768"):
    over = [load(C / job / "summary.json")[length]["overhead_pct"]["full"] for job in ("b1-job11", "b1-job14-timing")]
    claim(f"D4.full_scan.{length}", f"{num(min(over), 1)} to {num(max(over), 1)}")
rates = []
for job in ("b1-job12", "b1-job14-diag"):
    for name, row in load(C / job / "summary.json")["table"].items():
        if "control" not in name and "k1" not in name.split("-"):
            rates.append(row["miss_rate_per_token"])
claim("D4.periodic.configs", len(rates))
claim("D4.periodic.miss", f"{num(100 * min(rates), 0)} to {num(100 * max(rates), 0)}")

# D5. Qwen3.8-27B, the configuration of 2026-10-22 against stock: the two needle suites and RULER at 16K.
E = D / "needles-vs-ruler-27b"
correct = {arm: sum(load(E / s / "summary.json")["correct"][arm] for s in ("needles-fresh", "needles-tuning"))
           for arm in ("stock", "winner")}
total = sum(load(E / s / "summary.json")["n"] for s in ("needles-fresh", "needles-tuning"))
claim("D5.needles.stock", f"{correct['stock']} of {total}")
claim("D5.needles.ridgefill", f"{correct['winner']} of {total}")
ruler = load(E / "ruler16" / "summary.json")["official"]
for task in ("niah_multikey_2", "niah_multikey_3"):
    n = ruler["stock"]["by_task"][task]["count"]
    for arm, key in (("stock", "stock"), ("winner", "ridgefill")):
        claim(f"D5.{task}.{key}", f"{round(ruler[arm]['by_task'][task]['score'] * n / 100)} of {n}")
