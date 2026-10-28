"""Appendix A: the previously unclaimed primary-split arm means and document-name overlap.

Read the reference cells' per-sample JSON, with the finals' estimands: pooled RULER
sample means (filled rows where specified), document-weighted arm-minus-exact NLL,
prompt-weighted decode agreement/KL, and sample-weighted second-question accuracy.
Tree routes the rows NLL arms to their corrected nll-causal records, including that
cell's exact baseline; key NLL uses only documents whose exact mask has key tokens.
The question-first means read ruler-qfirst, not question-blind ruler arms. Count
the intersection of distinct fresh/locked Llama document names directly from JSON.
All human-readable diagnostics go to stderr; stdout contains only claim lines.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, num, parse, say, sgn              # noqa: E402
from reftree import METRICS, MODELS, Tree                    # noqa: E402

# Stats.metric in code/finals/tables.py also declares this all-item readout.
METRICS["qfirst_all"] = ("ruler-qfirst", "score", 100.0, "accuracy", None)

# Each token is metric[:population]. An arm's population is the records that
# carry it: Qwen3 controls and question-first arms have only the needle tasks.
MEANS = {
    "llama": [
        ("exact", "exact", "-", "ruler_filled:8k_64k"),
        ("nofill", "skip", "3/8", "nll_novel nll_key agree kl"),
        ("identity", "noproj", "3/8", "nll_key agree"),
        ("pop", "pop-native", "pop", "nll_novel nll_key"),
        ("ridgefill", "kva", "3/8", "nll_key agree turn2 ruler_filled:8k ruler_filled:64k"),
        ("ridgefill_rows", "kva+qattn10", "3/8", "ruler_filled:8k_64k nll_novel nll_key"),
        ("identity_rows", "noproj+qattn10", "3/8", "ruler_filled:8k_64k nll_novel nll_key turn2"),
        ("ridgefill_random", "kva+random", "3/8", "ruler_filled:8k_32k nll_novel nll_key"),
        ("ridgefill_longer_tail", "kva+tailmatch", "3/8", "ruler_filled:8k_32k nll_novel nll_key"),
        ("nofill_longer_tail", "skip+tailbudget", "3/8", "nll_novel nll_key"),
        ("identity_qblind", "noproj+qblind10", "3/8", "ruler_filled:8k_32k"),
        ("ridgefill_qblind", "kva+qblind10", "3/8", "ruler_filled:8k_32k"),
    ],
    "qwen35": [
        ("exact", "exact", "-", "ruler_filled:8k_64k"),
        ("nofill", "skip", "3/8", "ruler_filled:8k_32k nll_novel nll_key agree kl"),
        ("identity", "noproj", "3/8", "nll_novel nll_key agree"),
        ("pop", "pop-native", "pop", "nll_novel nll_key"),
        ("ridgefill", "kva", "3/8", "nll_novel nll_key agree turn2"),
        ("ridgefill_rows", "kva+qattn10", "3/8", "ruler_filled:8k_64k nll_novel nll_key"),
        ("identity_rows", "noproj+qattn10", "3/8", "ruler_filled:8k_64k nll_novel nll_key turn2"),
        ("ridgefill_random", "kva+random", "3/8", "ruler_filled:8k_32k nll_novel nll_key"),
        ("ridgefill_longer_tail", "kva+tailmatch", "3/8", "ruler_filled:8k_32k nll_novel nll_key"),
        ("nofill_longer_tail", "skip+tailbudget", "3/8", "ruler_filled:8k_32k nll_novel nll_key"),
        ("identity_qblind", "noproj+qblind10", "3/8", "ruler_filled:8k_32k"),
        ("ridgefill_qblind", "kva+qblind10", "3/8", "ruler_filled:8k_32k"),
    ],
    "qwen3": [
        ("exact", "exact", "-", "ruler_all:8k_32k"),
        ("nofill", "skip", "3/8", "ruler_filled:8k_32k ruler_all:8k_32k nll_novel nll_key agree kl"),
        ("identity", "noproj", "3/8", "ruler_all:8k_32k nll_novel nll_key agree"),
        ("pop", "pop-native", "pop", "ruler_all:8k_32k nll_novel nll_key"),
        ("ridgefill", "kva", "3/8", "ruler_all:8k_32k nll_novel nll_key agree turn2"),
        ("ridgefill_rows", "kva+qattn10", "3/8", "ruler_filled:8k_32k ruler_all:8k_32k nll_novel nll_key qfirst_filled:8k_32k qfirst_all:8k_32k"),
        ("identity_rows", "noproj+qattn10", "3/8", "ruler_filled:8k_32k ruler_all:8k_32k nll_novel nll_key turn2 qfirst_filled:8k_32k qfirst_all:8k_32k"),
        ("ridgefill_random", "kva+random", "3/8", "ruler_filled:8k_32k ruler_all:8k_32k nll_novel nll_key"),
        ("ridgefill_longer_tail", "kva+tailmatch", "3/8", "ruler_filled:8k_32k ruler_all:8k_32k nll_novel nll_key"),
        ("nofill_longer_tail", "skip+tailbudget", "3/8", "ruler_filled:8k_32k ruler_all:8k_32k nll_novel nll_key"),
    ],
}
POPULATIONS = {
    "8k_32k": lambda r: 8192 <= r["length"] <= 32768,
    "8k_64k": lambda r: 8192 <= r["length"] <= 65536,
    "8k": lambda r: r["length"] == 8192,
    "64k": lambda r: r["length"] == 65536,
}

args = parse(__doc__)
t = Tree(args.root, [m + suffix for m in MODELS.values()
                     for suffix in ("-ruler", "-nll", "-nll-causal", "-decode", "-secondq")]
         + [MODELS["llama"] + "-nll-fresh"])

for short, specs in MEANS.items():
    model = MODELS[short]
    for key, arm, split, metrics in specs:
        for spec in metrics.split():
            metric, _, population = spec.partition(":")
            where = POPULATIONS.get(population)
            delta = metric.startswith("nll_")
            value = t.mean("battery", model, metric, arm, split, vs_exact=delta, where=where)
            routed = t.route(metric, (arm, split))
            n = len(t.values("battery", model, routed, arm, split, where=where))
            if delta:
                formatted = sgn(value, 3)
            else:
                formatted = num(value, 3 if metric == "kl" else 1)
                if metric == "agree":
                    formatted += " %"
            cid = ".".join(p for p in ("AB", short, metric, population, key) if p)
            with redirect_stdout(sys.stderr):
                say(f"{cid}: {routed}, {arm}@{split}, n={n}, mean={value:.9f}")
            claim(cid, formatted)

locked = {r["doc"] for r in t.records("battery", MODELS["llama"], "longppl").values()}
fresh_scope = "dataset:" + t.cells[MODELS["llama"] + "-nll-fresh"]["dataset"]
fresh = {r["doc"] for r in t.records(fresh_scope, MODELS["llama"], "longppl").values()}
with redirect_stdout(sys.stderr):
    say(f"Llama distinct document names: locked={len(locked)}, fresh={len(fresh)}, overlap={len(locked & fresh)}")
claim("AB.llama.documents.fresh_locked_overlap", len(locked & fresh))
