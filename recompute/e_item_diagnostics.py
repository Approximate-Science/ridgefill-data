"""Paper sections 4.4 and 4.7: LongBench v2 pairs run on different GPU indices and the extreme 32K code item's
copyable share. Pair LongBench's per-item records by the sorted intersection of stock and E3 filenames, as in
recompute/lb_report.py, then count unequal gpu fields. Read the code item's recorded copyable share from
results/qwen38-27b/PER-DOC-KL.json, selected uniquely by (L, id); no withheld benchmark text or gold is needed.
Human-readable diagnostics go to stderr; stdout contains only claim lines.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say       # noqa: E402

args = parse(__doc__)
Q = args.root / "results" / "qwen38-27b"
arms = {arm: {p.stem: load(p) for p in sorted(need(Q / "longbench" / arm).glob("*.json"))}
        for arm in ("stock", "e3")}
ids = sorted(set(arms["stock"]) & set(arms["e3"]))
if not ids:
    raise SystemExit("no paired LongBench items")
different = sum(arms["stock"][i]["gpu"] != arms["e3"][i]["gpu"] for i in ids)
with redirect_stdout(sys.stderr):
    say(f"LongBench: {different} of {len(ids)} paired items ran on different GPU indices")
claim("E.longbench.pairs_different_gpu", different)

length, item_id = 32768, "66ecf139821e116aacb1e0e1"
selected = [r for r in load(Q / "PER-DOC-KL.json") if r["L"] == length and r["id"] == item_id]
if len(selected) != 1:
    raise SystemExit(f"PER-DOC-KL.json: expected one ({length}, {item_id}) record, got {len(selected)}")
copyable = selected[0]["copyable"]
if not 0 <= copyable <= 1:
    raise SystemExit(f"{item_id}: invalid copyable share {copyable}")
with redirect_stdout(sys.stderr):
    say(f"{length} {item_id}: recorded copyable share {copyable}")
claim(f"E.kl.{length}.{item_id}.copyable", num(copyable, 2))
