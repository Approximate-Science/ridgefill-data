"""Z: Appendix D, Table D.1 and the second-needle selector overlap.

Re-score turn 2 from the original multiturn outputs with data.niah_correct's
first standalone seven-digit rule (think blocks removed), and with RULER's
substring rule. Means weight samples equally; filled-region means filter the
arm's turn-2 needle_projected flag. Coverage intersects the first question's
half-open selected-row intervals with the second needle, then takes the median
over filled-region samples. The paired Qwen3 contrast uses reftree/bca, exactly
as finals/tables.py: task-stratified, unit-weighted 95% BCa, 5,000 resamples,
seed 20261004, sorted sample ids, half-weight bootstrap ties.
"""
import re
import statistics
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, claims_ci, load, need, num, parse, say   # noqa: E402
from reftree import MODELS, Tree                            # noqa: E402

args = parse(__doc__)
t = Tree(args.root, [m + "-secondq" for m in MODELS.values()])


def strict(turn):
    text = re.sub(r"<think>.*?(</think>|$)", "", turn["text"], flags=re.S)
    match = re.search(r"(?<!\d)\d{7}(?!\d)", text)
    return bool(match and match.group(0) == turn["value"])


def selected_overlap(intervals, start, end):
    """Count the union's intersection; reject overlapping or unsorted intervals."""
    last, total = 0, 0
    for a, b in intervals:
        if not last <= a < b:
            raise ValueError("selected rows must be sorted disjoint half-open intervals")
        total += max(0, min(b, end) - max(a, start))
        last = b
    return total


for short, model in MODELS.items():
    directory = need(args.root / "results" / "reference" / (model + "-secondq") / "work" / "results")
    paths = sorted(directory.glob("multiturn-*/*.json"))
    if not paths:
        raise SystemExit(f"missing input: no multiturn outputs under {directory}")
    raw = [load(path) for path in paths]
    units = {r["source"] + "#" + f"{r['row']:02d}": r for r in raw}
    records = t.records("battery", model, "multiturn")
    if len(units) != len(raw) or set(units) != set(records):
        raise ValueError(f"{model}: original outputs and sample records do not pair one-to-one")

    # Find the actual primary split suffix, rather than assuming the same layer
    # number on the models with different layer counts.
    runs = raw[0]["runs"]
    fill_name, = [a for a in runs if re.fullmatch(r"ridgefill-s\d+", a)]
    suffix = fill_name.removeprefix("ridgefill")
    names = {"ridgefill": fill_name, "identity_rows": "identity+rows" + suffix,
             "ridgefill_rows": "ridgefill+rows" + suffix}
    normalized = {"ridgefill": "kva", "identity_rows": "noproj+qattn10",
                  "ridgefill_rows": "kva+qattn10"}
    for key, arm in names.items():
        turns = [r["runs"][arm]["turns"][1] for r in raw]
        for unit, r in units.items():
            turn = r["runs"][arm]["turns"][1]
            score = strict(turn)
            if score != turn["correct"] or score != bool(records[unit]["arms"][normalized[key]]["3/8"]["turn2"]):
                raise ValueError(f"{model} {unit} {arm}: strict score disagrees with stored score")
        filled = [turn for turn in turns if turn["needle_projected"]]
        if key != "ridgefill_rows":
            claim(f"Z.turn2.{short}.{key}.strict", num(100 * statistics.mean(map(strict, turns)), 1))
        if key != "identity_rows":
            claim(f"Z.turn2.{short}.{key}.filled.strict", num(100 * statistics.mean(map(strict, filled)), 1))
        if short == "llama" and key == "ridgefill":
            claim("Z.turn2.llama.ridgefill.substring",
                  num(100 * statistics.mean(v["value"].lower() in v["text"].lower() for v in turns), 1))

    tail_counts = [sum(not r["runs"][a]["turns"][1]["needle_projected"] for r in raw)
                   for a in names.values()]
    if len(set(tail_counts)) != 1:
        raise ValueError(f"{model}: filled arms disagree on the exact-tail population")
    claim(f"Z.turn2.{short}.tail_needles", tail_counts[0])
    coverage = []
    for r in raw:
        run = r["runs"][names["ridgefill_rows"]]
        turn = run["turns"][1]
        if not turn["needle_projected"]:
            continue
        start, end = turn["needle_start"], turn["needle_end"]
        covered = selected_overlap(run["rows"], start, end)
        if covered != turn["needle_rows_exact"] or not start < end:
            raise ValueError(f"{model} {r['name']}: invalid second-needle row coverage")
        coverage.append(covered / (end - start))
    claim(f"Z.turn2.{short}.needle_coverage.median", num(100 * statistics.median(coverage), 0) + " %")
    with redirect_stdout(sys.stderr):
        say(f"{model}: {len(raw)} samples, {tail_counts[0]} second needles in the tail; "
            f"median selected-row coverage {100 * statistics.median(coverage):.8f}% "
            f"over {len(coverage)} filled-region second needles")

c = t.contrast("battery", MODELS["qwen3"], "turn2", ("kva+qattn10", "3/8"), ("exact", "-"))
claims_ci("Z.turn2.qwen3.rows_minus_exact", c["est"], c["lo"], c["hi"], 1)
with redirect_stdout(sys.stderr):
    say(f"Qwen3 rows minus exact: {c['n']} paired samples; 95% BCa [{c['lo']}, {c['hi']}]")
