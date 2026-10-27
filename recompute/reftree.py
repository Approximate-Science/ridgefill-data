"""The reference cells' sample records and the finals tables' contrasts, ported from code/finals/tables.py (release
branch: Tree.load, Stats.values / mean / contrast / noninferiority / ratio) without the ledger, verification and
table rendering. A pairing unit is a record's "sample" within (scope, model, set); a contrast pairs the units that
carry both arms; strata are RULER rows within task, documents within domain; the estimand weighting and the
supersession of the rows / tail-scale NLL arms by the nll-causal cells are the tables' own.
"""
from pathlib import Path

import numpy as np

import bca as S
from common import load, need

PRIMARY = "3/8"
SUPERSEDED = ("noproj+qattn10", "kva+qattn10", "kva+ktail", "kva+vtail")      # nll-causal re-runs these at 3/8
STRATA_KEY = {"ruler": "task", "ruler-deep": "task", "ruler-qfirst": "task", "multiturn": "task",
              "longbench": "task", "longppl": "domain", "longppl-causal": "domain", "decode": None}
WEIGHTING = {"accuracy": "unit", "nll": "unit", "agreement": "unit", "kl": "unit", "longbench": "stratum"}
FILLED = lambda r: bool(r.get("needle_filled"))                                       # noqa: E731
METRICS = {  # name -> (set, record metric, scale, estimand, population filter)
    "ruler_all": ("ruler", "score", 100.0, "accuracy", None),
    "ruler_filled": ("ruler", "score", 100.0, "accuracy", FILLED),
    "deep_filled": ("ruler-deep", "score", 100.0, "accuracy", FILLED),
    "qfirst_filled": ("ruler-qfirst", "score", 100.0, "accuracy", FILLED),
    "longbench": ("longbench", "score", 100.0, "longbench", None),
    "nll_novel": ("longppl", "novel_nll", 1.0, "nll", None),
    "nll_key": ("longppl", "key_nll", 1.0, "nll", None),
    "nll_novel_causal": ("longppl-causal", "novel_nll", 1.0, "nll", None),
    "nll_key_causal": ("longppl-causal", "key_nll", 1.0, "nll", None),
    "agree": ("decode", "top1_agree", 100.0, "agreement", None),
    "kl": ("decode", "kl_mean", 1.0, "kl", None),
    "turn2": ("multiturn", "turn2", 100.0, "accuracy", None),
}
SUBSETS = {"64k": lambda r: r.get("length") == 65536, "le32k": lambda r: r.get("length", 0) <= 32768}


def scope_of(cell):
    return "dataset:" + cell["dataset"] if cell.get("dataset") else "battery"


class Tree:
    """The sample records of the named reference cells (results/reference/<cell>/samples/**/*.json)."""

    def __init__(self, root, cells):
        self.data, self.cells = {}, {}
        for name in cells:
            d = need(Path(root) / "results" / "reference" / name)
            cell = load(d / "cell.json")
            self.cells[name] = cell
            scope, model = scope_of(cell), cell["model"]["name"]
            files = sorted(f for f in (d / "samples").rglob("*.json") if f.name != "CONVERTED.json")
            if not files:
                raise SystemExit(f"missing input: no sample records under {d / 'samples'}")
            for f in files:
                r = load(f)
                if "skipped" in r:
                    continue
                set_ = r["set"]
                if cell["kind"] == "nll-causal":
                    set_ = "longppl-causal"
                elif cell["kind"] == "ruler-deep":
                    set_ = {"ruler": "ruler-deep"}[set_]
                slot = self.data.setdefault((scope, model, set_), {})
                if r["sample"] in slot:
                    raise SystemExit(f"{name}: unit {r['sample']} of {model} {set_} is filled by two cells")
                slot[r["sample"]] = r

    def records(self, scope, model, set_):
        recs = self.data.get((scope, model, set_))
        if not recs:
            raise SystemExit(f"no {set_} records for {model} in scope {scope}: is its cell loaded?")
        return recs

    def values(self, scope, model, name, arm, split, where=None):
        """{unit: value} of one arm at one split over its population (the units that carry it)."""
        set_, key, scale, _, own = METRICS[name]
        if name.startswith("nll_key"):                     # documents with key tokens (the exact model's mask)
            own = lambda r: (r["arms"].get("exact", {}).get("-") or {}).get("n_key", 0) > 0     # noqa: E731
        out = {}
        for unit, r in self.records(scope, model, set_).items():
            if (own and not own(r)) or (where and not where(r)):
                continue
            m = r["arms"].get(arm, {}).get(split)
            if m is None:
                continue
            if m.get(key) is None:
                raise SystemExit(f"{model} {set_} {unit}: arm {arm}@{split} has no {key}")
            out[unit] = float(m[key]) * scale
        if not out:
            raise SystemExit(f"{model} {set_}: no unit carries {arm}@{split} with {key}")
        return out

    @staticmethod
    def route(name, *arm_splits):
        """An NLL metric that involves a superseded arm at the primary split reads the nll-causal cell's records."""
        if name in ("nll_novel", "nll_key") and any(a in SUPERSEDED and s == PRIMARY for a, s in arm_splits):
            return name + "_causal"
        return name

    def _strata(self, scope, model, name, units):
        set_ = METRICS[name][0]
        key = STRATA_KEY.get(set_)
        if key is None:
            return None
        recs = self.records(scope, model, set_)
        return [recs[u].get(key) for u in units]

    def mean(self, scope, model, name, arm, split, vs_exact=False, where=None):
        """Estimand mean of an arm (vs_exact: of arm - exact, paired) over its population."""
        name = self.route(name, (arm, split))
        v = self.values(scope, model, name, arm, split, where)
        if vs_exact:
            e = self.values(scope, model, name, "exact", "-", where)
            units = sorted(set(v) & set(e))
            x = np.array([v[u] - e[u] for u in units])
        else:
            units = sorted(v)
            x = np.array([v[u] for u in units])
        if WEIGHTING[METRICS[name][3]] == "stratum":
            labs = self._strata(scope, model, name, units)
            return float(np.mean([x[[i for i, lab in enumerate(labs) if lab == g]].mean() for g in sorted(set(labs))]))
        return float(x.mean())

    def paired(self, scope, model, name, a, b, where=None):
        """(units, xa, xb) of the units that carry both (arm, split) a and b, units sorted."""
        name = self.route(name, a, b)
        va = self.values(scope, model, name, *a, where=where)
        vb = self.values(scope, model, name, *b, where=where)
        units = sorted(set(va) & set(vb))
        if not units:
            raise SystemExit(f"{model} {name}: no unit carries both {a} and {b}")
        return name, units, np.array([va[u] for u in units]), np.array([vb[u] for u in units])

    def contrast(self, scope, model, name, a, b, where=None):
        """Paired a - b: dict(n, est, lo, hi) with the BCa interval of the declared estimand."""
        name, units, xa, xb = self.paired(scope, model, name, a, b, where)
        r = S.bca(xa - xb, strata=self._strata(scope, model, name, units),
                  weighting=WEIGHTING[METRICS[name][3]])
        return dict(n=len(units), est=r.estimate, lo=r.lo, hi=r.hi, deltas=xa - xb, units=units)

    def noninferiority_lower(self, scope, model, name, a, b):
        """(n, estimate, one-sided 95 % lower bound) of a - b with the binomial-floor guard (RULER rows)."""
        name, units, xa, xb = self.paired(scope, model, name, a, b)
        est, lo, _ = S.noninferiority_lower(xa - xb, strata=self._strata(scope, model, name, units),
                                            weighting=WEIGHTING[METRICS[name][3]])
        return len(units), est, lo

    def ratio(self, scope, model, name, cols, stat, where=None):
        """BCa of stat(means of the paired arm columns); returns (n, estimate, lo, hi), lo/hi None when the
        statistic is not finite on some resample (the tables print the estimate alone then)."""
        name = self.route(name, *cols)
        vs = [self.values(scope, model, name, *c, where=where) for c in cols]
        units = sorted(set.intersection(*(set(v) for v in vs)))
        x = np.array([[v[u] for v in vs] for u in units])
        try:
            r = S.bca(x, stat=stat, strata=self._strata(scope, model, name, units),
                      weighting=WEIGHTING[METRICS[name][3]])
        except S.IntervalUndefined as e:
            return len(units), e.estimate, None, None
        return len(units), r.estimate, r.lo, r.hi


MODELS = {"llama": "llama-3.1-8b-instruct", "qwen35": "qwen3.5-9b", "qwen3": "qwen3-8b"}   # claim-id short names
