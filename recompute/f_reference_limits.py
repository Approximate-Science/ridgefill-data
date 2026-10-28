"""Section 8: reference-model retrieval, key-token loss, the worst Llama document, and second-needle row coverage.

Use finals/tables.py's sample-weighted filled-region RULER means and document-weighted paired NLL gaps. Primary
rows-arm NLL reads the superseding nll-causal cells, with key-token documents selected by exact's n_key > 0.
The half-depth follow-up is present in the ordinary Llama ruler cell (8K-32K), not its ruler-deep cell.
For second questions, intersect the first question's half-open selected-row ranges with turn 2's needle span;
take the median token fraction over needles wholly before the exact tail. All reported values are descriptive
point statistics, so no bootstrap is required. Human diagnostics go to stderr; stdout contains only claims.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path
from statistics import median

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say, sgn  # noqa: E402
from reftree import MODELS, Tree                            # noqa: E402

args = parse(__doc__)
P = "3/8"
t = Tree(args.root, [m + "-ruler" for m in MODELS.values()]
         + [MODELS[s] + "-nll-causal" for s in ("llama", "qwen35")])


def describe(*parts):
    with redirect_stdout(sys.stderr):
        say(*parts)


def retrieval(short, suffix, arm, split, low, high):
    model = MODELS[short]
    where = lambda r: low <= r["length"] <= high
    values = t.values("battery", model, "ruler_filled", arm, split, where)
    estimate = t.mean("battery", model, "ruler_filled", arm, split, where=where)
    describe(f"{model} {arm}@{split}, {low}-{high}: filled-region mean {estimate:.8f}, n={len(values)}")
    claim(f"F.{short}.{suffix}", num(estimate, 1))


for short in ("llama", "qwen35"):
    retrieval(short, "rows.filled_8k_64k", "kva+qattn10", P, 8192, 65536)
    model = MODELS[short]
    _, units, arm, exact = t.paired("battery", model, "nll_key", ("kva+qattn10", P), ("exact", "-"))
    gap = float((arm - exact).mean())
    describe(f"{model} rows - exact key-token NLL: {gap:.10f} nats, n={len(units)} documents")
    claim(f"F.{short}.rows.key_nll_gap", num(gap, 2))

doc = "source_code_128000_0"
_, units, arm, exact = t.paired("battery", MODELS["llama"], "nll_novel",
                              ("kva+qattn10", P), ("exact", "-"), where=lambda r: r["doc"] == doc)
if len(units) != 1:
    raise SystemExit(f"expected one locked document named {doc}, got {len(units)}")
gap = float((arm - exact)[0])
describe(f"Llama {doc}, rows - exact novel-token NLL: {gap:.10f} nats")
claim("F.llama.source_code_128000_0.rows.novel_nll_gap", sgn(gap, 3))

retrieval("qwen3", "rows.filled_8k_32k", "kva+qattn10", P, 8192, 32768)
retrieval("qwen3", "random.filled_8k_32k", "kva+random", P, 8192, 32768)
retrieval("llama", "half_depth.filled_8k_32k", "kva", "1/2", 8192, 32768)
for length in (8192, 65536):
    retrieval("llama", f"primary.filled_{length}", "kva", P, length, length)

for short in ("llama", "qwen35"):
    model = MODELS[short]
    cell = need(args.root / "results" / "reference" / (model + "-secondq"))
    files = sorted(need(cell / "work" / "results" / ("multiturn-" + model + "-secondq")).glob("ruler2k-*.json"))
    if not files:
        raise SystemExit(f"missing input: no second-question records under {cell}")
    coverage = []
    for path in files:
        record = load(path)
        runs = [run for name, run in record["runs"].items() if name.startswith("ridgefill+rows-")]
        if len(runs) != 1:
            raise SystemExit(f"{path}: expected one RidgeFill + rows run, got {len(runs)}")
        run = runs[0]
        turn = run["turns"][1]
        start, end = turn["needle_start"], turn["needle_end"]
        if not 0 <= start < end or turn["turn"] != 2:
            raise SystemExit(f"{path}: invalid second needle")
        filled = end <= run["P"]
        if filled != turn["needle_projected"]:
            raise SystemExit(f"{path}: inconsistent second-needle filled-region flag")
        if not filled:
            continue
        covered, previous = 0, 0
        for left, right in run["rows"]:
            if not previous <= left < right <= run["P"]:
                raise SystemExit(f"{path}: selected rows must be disjoint half-open prefix ranges")
            covered += max(0, min(end, right) - max(start, left))
            previous = right
        coverage.append(covered / (end - start))
    if not coverage:
        raise SystemExit(f"{cell}: no second needles in the filled region")
    estimate = 100 * median(coverage)
    describe(f"{model} second-needle coverage: median {estimate:.8f} %, n={len(coverage)} of {len(files)} records")
    claim(f"F.{short}.turn2.rows.needle_coverage_median", num(estimate, 0) + " %")
