"""LB4 (groups): Qwen3.8-27B LongBench v2 by difficulty, length and domain: accuracy per arm and the paired E3 - stock
difference with the benchmark report's bootstrap applied to each group (a fresh random.Random(20261025) per group,
rng.choices over the group's item differences in sorted-id order, 10,000 resamples, boot[250] / boot[9750]). Domains
come from provenance/inputs/qwen38-27b/longbench-index.json (id, domain, difficulty, length; the item records carry no
domain). Also the scorecard's own overall interval (cloud-kit common.bootstrap: random.Random(1), items in the
index's order), which the paper cites next to the report's."""
import random
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say, sgn  # noqa: E402

DOMAINS = {"Single-Document QA": "single_doc_qa", "Multi-Document QA": "multi_doc_qa",
           "Long In-context Learning": "long_icl", "Long-dialogue History Understanding": "long_dialogue",
           "Code Repository Understanding": "code_repo", "Long Structured Data Understanding": "structured"}
args = parse(__doc__)
out = args.root / "results" / "qwen38-27b" / "longbench"
index = load(args.root / "provenance" / "inputs" / "qwen38-27b" / "longbench-index.json")
arms = {arm: {p.stem: load(p) for p in sorted(need(out / arm).glob("*.json"))} for arm in ("stock", "e3")}
ids = sorted(set(arms["stock"]) & set(arms["e3"]))
meta = {x["id"]: x for x in index}
if sorted(meta) != ids:
    raise SystemExit("longbench-index.json does not list the items of results/qwen38-27b/longbench")
for i in ids:
    if (arms["stock"][i]["difficulty"], arms["stock"][i]["length"]) != (meta[i]["difficulty"], meta[i]["length"]):
        raise SystemExit(f"{i}: the index's difficulty / length differ from the item record's")
diff = {i: int(arms["e3"][i]["judge"]) - int(arms["stock"][i]["judge"]) for i in ids}
groups = [(g, [i for i in ids if meta[i]["difficulty"] == g]) for g in ("easy", "hard")]
groups += [(g, [i for i in ids if meta[i]["length"] == g]) for g in ("short", "medium", "long")]
groups += [(key, [i for i in ids if meta[i]["domain"] == d]) for d, key in DOMAINS.items()]
for key, sel in groups:
    d = [diff[i] for i in sel]
    rng = random.Random(20261025)
    boot = sorted(100 * sum(rng.choices(d, k=len(d))) / len(d) for _ in range(10000))
    s = 100 * sum(arms["stock"][i]["judge"] for i in sel) / len(sel)
    e = 100 * sum(arms["e3"][i]["judge"] for i in sel) / len(sel)
    say(f"{key}: n {len(sel)}, stock {s:.2f}, E3 {e:.2f}, delta {e - s:+.2f} [{boot[250]:+.2f}, {boot[9750]:+.2f}]")
    claim(f"LB4.{key}.n", len(sel))
    claim(f"LB4.{key}.stock", num(s, 2))
    claim(f"LB4.{key}.e3", num(e, 2))
    claim(f"LB4.{key}.delta", sgn(e - s, 2))
    claim(f"LB4.{key}.ci_lo", sgn(boot[250], 2))
    claim(f"LB4.{key}.ci_hi", sgn(boot[9750], 2))
d = [100 * diff[x["id"]] for x in index]
rng = random.Random(1)
bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
say(f"scorecard bootstrap (seed 1, index order): {statistics.mean(d):+.2f} [{bs[250]:+.2f}, {bs[9750]:+.2f}]")
claim("LB4.scorecard.ci_lo", sgn(bs[250], 2))
claim("LB4.scorecard.ci_hi", sgn(bs[9750], 2))
