#!/usr/bin/env python3
"""A synthetic, verifiable finals results tree for the table generator (finals/tables.py). NOTHING here is evidence.

It writes a small pseudo-repo that mirrors the real layout, so the whole path mock results -> verify_finals.py ->
tables.py -> paper/tex/check_numbers.py runs on CPU in seconds:
  ROOT/finals-results/<cell>/   cell.json, host.json, canary.json, samples/<set>/<id>.json (+ .sha256 sidecars,
                                atomic.write_result), .pushed (both receipts), .done   -- what verify_finals reads
  ROOT/ledger/paper/ledger.jsonl  FINAL-MOCK-* hypothesis records (each 'cell' = the cell declaration) and the
                                pre-final records the generator reads (S16 probe values, file pins,
                                negative-result values)
  ROOT/proj/*.safetensors       header-only projector files (metadata lam / rows / docs; no tensors)
  ROOT/results/...              a v1.0 battery JSON (grade per-prompt deltas) and two timing JSONs, pinned by sha256
  ROOT/paper/LEVERS-PAPER.md    a curated lever list (the Markdown table the generator reads), pinned by sha256

The records follow the finals sample-record contract that tables.py reads (its docstring is the definition): one
record per (cell, set, unit) with every arm the unit ran, arms[arm][split-key] = metrics, and its tier. The arms follow
the cells' layout where tables.py depends on it (finals/make_cells.py): the splices in the NLL cells (longppl) and,
kexact / vexact only, on the RULER cells' NIAH rows; the question-first set (T1.4) runs exact and the two rows arms;
otherwise the mock keeps the full design of OUTLINE T1 on every row (finals/align_audit.py checks the tables against
the cells' own layout). Effects are PLANTED so the
pipeline's positive and negative controls have known answers (the defaults below; build(..., overrides) changes
them): on Llama RidgeFill beats identity (RULER filled and NLL), on Qwen3.5 the two fills tie (identical latent
rates, independent noise), exact keys carry Llama's loss and values Qwen3.5's, and on Flash-Next the quality mode
is within the perplexity criterion and RULER margin of exact. Values come from a numpy Generator seeded by sha256(cell | set | unit), so two builds are
byte-identical.

CLI: tables_tree.py DEST   -> writes the tree, prints its root
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FINALS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINALS / "lib"))
import atomic                                                # noqa: E402
from registration import pins_text as registration_pins_text  # noqa: E402

MANIFEST_SHA = "f" * 64                                      # a fake locked-manifest hash (never the real one)
FRACS = ("1/4", "3/8", "1/2", "5/8")
DEPTH_FACTOR = {"1/4": 1.6, "3/8": 1.0, "1/2": 0.7, "5/8": 0.45}     # loss scales down as the split moves deeper
RULER_TASKS = ("niah_single_1", "niah_multikey_1", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1")
RULER_LENGTHS = (8192, 16384, 32768, 65536)
LB_TASKS = ("narrativeqa", "qasper", "hotpotqa", "gov_report", "multi_news", "trec", "lcc", "repobench-p")
DOMAINS = ("prose", "code", "papers", "chat", "legal", "web")
MODELS = {
    "llama-3.1-8b-instruct": dict(id="meta-llama/Llama-3.1-8B-Instruct", commit="0" * 40, proj="llama-final-s12",
                                  lam=0.03, rows=1000001, max_len=None, swiftkv=True),
    "qwen3.5-9b": dict(id="Qwen/Qwen3.5-9B", commit="1" * 40, proj="qwen35-final-s12", lam=0.1, rows=1000002,
                       max_len=None, swiftkv=False),
    "qwen3-8b": dict(id="Qwen/Qwen3-8B", commit="2" * 40, proj="qwen3-final-s13", lam=0.03, rows=1000003,
                     max_len=32768, swiftkv=False),
}
FN_MODEL = "qwen3.8-flash-next"

# planted latent rates (RULER filled-region needle probability; the tail region is 0.92 for every arm)
RULER_P = {
    "llama-3.1-8b-instruct": {"exact": 0.92, "nofill": 0.25, "identity": 0.45, "pop-native": 0.40, "ridgefill": 0.70,
                              "identity+rows": 0.78, "ridgefill+rows": 0.88, "kva+random": 0.71,
                              "kva+tailmatch": 0.72, "skip+tailbudget": 0.30, "noproj+qblind10": 0.50,
                              "kva+qblind10": 0.74, "swiftkv": 0.80, "swiftkv-native": 0.66,
                              "kva+kexact": 0.88, "kva+vexact": 0.73, "kvexact": 0.92, "kva+ktail": 0.80,
                              "kva+vtail": 0.71},
    "qwen3.5-9b": {"exact": 0.92, "nofill": 0.30, "identity": 0.55, "pop-native": 0.50, "ridgefill": 0.55,
                   "identity+rows": 0.86, "ridgefill+rows": 0.86, "kva+random": 0.57, "kva+tailmatch": 0.58,
                   "skip+tailbudget": 0.33, "noproj+qblind10": 0.60, "kva+qblind10": 0.60,
                   "kva+kexact": 0.90, "kva+vexact": 0.62, "kvexact": 0.92, "kva+ktail": 0.56, "kva+vtail": 0.60},
    "qwen3-8b": {"exact": 0.92, "nofill": 0.30, "identity": 0.52, "pop-native": 0.47, "ridgefill": 0.56,
                 "identity+rows": 0.84, "ridgefill+rows": 0.85, "kva+random": 0.57, "kva+tailmatch": 0.58,
                 "skip+tailbudget": 0.33, "noproj+qblind10": 0.58, "kva+qblind10": 0.60,
                 "kva+kexact": 0.89, "kva+vexact": 0.61, "kvexact": 0.92, "kva+ktail": 0.58, "kva+vtail": 0.60},
}
# planted mean NLL delta vs exact (nats) at the primary split; other splits scale by DEPTH_FACTOR
NLL_MU = {
    "llama-3.1-8b-instruct": {"nofill": 0.30, "identity": 0.14, "pop-native": 0.10, "ridgefill": 0.07,
                              "identity+rows": 0.11, "ridgefill+rows": 0.05, "kva+random": 0.068,
                              "kva+tailmatch": 0.066, "skip+tailbudget": 0.27, "swiftkv": 0.04,
                              "swiftkv-native": 0.09, "kva+kexact": 0.02, "kva+vexact": 0.062, "kvexact": 0.0,
                              "kva+ktail": 0.04, "kva+vtail": 0.065},
    "qwen3.5-9b": {"nofill": 0.20, "identity": 0.06, "pop-native": 0.05, "ridgefill": 0.06, "identity+rows": 0.045,
                   "ridgefill+rows": 0.045, "kva+random": 0.058, "kva+tailmatch": 0.057, "skip+tailbudget": 0.18,
                   "kva+kexact": 0.052, "kva+vexact": 0.015, "kvexact": 0.0, "kva+ktail": 0.059, "kva+vtail": 0.05},
    "qwen3-8b": {"nofill": 0.22, "identity": 0.07, "pop-native": 0.06, "ridgefill": 0.065, "identity+rows": 0.05,
                 "ridgefill+rows": 0.048, "kva+random": 0.063, "kva+tailmatch": 0.062, "skip+tailbudget": 0.20,
                 "kva+kexact": 0.055, "kva+vexact": 0.02, "kvexact": 0.0, "kva+ktail": 0.064, "kva+vtail": 0.055},
}
for _m, _mu in NLL_MU.items():                       # the identity controls of the nll-causal cells (T1.5 count-matched)
    _mu.update({"noproj+random": _mu["identity"] - 0.004, "noproj+tailmatch": _mu["identity"] - 0.006})
# protocol 1.1.3: the 1.1.1 NLL cells' rows / ktail / vtail arms read the scored continuation, which LOWERED their NLL;
# the nll cell carries this planted leak on those arms, the nll-causal cell (the correction) the true NLL_MU
LEAK = {"identity+rows": -0.02, "ridgefill+rows": -0.015, "kva+ktail": -0.01, "kva+vtail": -0.008}
AGREE = {"nofill": 0.55, "identity": 0.72, "ridgefill": 0.84}
SPEEDUP = {"nofill": 2.0, "identity": 1.62, "ridgefill": 1.58, "pop-native": 1.35}   # at 3/8; deeper splits less
FN = {   # Flash-Next, the C5 tier as run: (mean dNLL vs exact, RULER filled rate, TTFT speedup)
    "exact": (0.0, 0.97, 1.0), "exact-rerun": (0.0, 0.97, 1.0),
    "ridgefill+corr+prot25": (0.012, 0.97, 1.36), "ridgefill": (0.025, 0.96, 1.55),
}
FN_SPEED_ARMS = ("ridgefill+corr+prot25", "ridgefill")      # each timed against its own boot's exact path
FN_DATA = dict(longppl=dict(docs=18), ruler=dict(rows=None, long=dict(rows=None)),
               ttft=dict(lengths=[9216, 16384, 32768], warmup=1, timed=5))


def _rng(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def _r(x, nd=6):
    return float(round(float(x), nd))


# ----------------------------------------------------------------------------------------------- cells
def _cell(name, kind, model, arms, registration, **extra):
    m = MODELS.get(model)
    c = {"name": name, "kind": kind,
         "model": {"id": m["id"] if m else "Qwen/Qwen3.8-Flash-Next", "name": model,
                   "commit": m["commit"] if m else "3" * 40},
         "split": "3/8", "tail": 4096, "arms": arms, "selector": "qattn10" if any("rows" in a for a in arms) else "none",
         "dataset_manifest_sha256": MANIFEST_SHA, "deterministic": True, "registration": registration,
         "code_tag": "protocol-v1.1", "note": "MOCK cell for finals/tables.py (finals/mock/tables_tree.py)"}
    if m:                                             # cell v2's projector block (finals/cell.schema.json)
        c["projector"] = {"provenance": {"calib_rows": m["rows"], "calib_docs": 98, "calib_exclude_sha256": "0" * 64,
                                         "fitted_after": "0000000"},
                          "files": [{"layer": int(m["proj"].rsplit("s", 1)[1]), "file": f"{m['proj']}.safetensors"}]}
    c.update(extra)
    return c


BATTERY_ARMS = ["exact", "nofill", "identity", "pop-native", "ridgefill", "identity+rows", "ridgefill+rows",
                "kva+random", "kva+tailmatch", "skip+tailbudget"]
SPLICES = ["kva+kexact", "kva+vexact", "kvexact", "kva+ktail", "kva+vtail"]
RULER_SPLICES = ("kva+kexact", "kva+vexact")                 # on the NIAH rows only, as the RULER cells run them
NIAH = RULER_TASKS[:4]
QBLIND_P = {"identity+rows": "noproj+qblind10", "ridgefill+rows": "kva+qblind10"}   # planted question-first rates
MARGINS = {"ppl_ratio_upper_pct": 1.0, "ruler_points_lower": -2.0}   # every reference cell's, as finals/make_cells.py
CAUSAL_ARMS = ["exact", "identity", "ridgefill", "identity+rows", "ridgefill+rows", "noproj+random", "kva+random",
               "noproj+tailmatch", "kva+tailmatch", "kva+ktail", "kva+vtail"]   # finals/make_cells.py CAUSAL
QBLIND = ["noproj+qblind10", "kva+qblind10"]
CORE = ["exact", "identity", "ridgefill", "identity+rows", "ridgefill+rows"]
# protocol 1.1.3: Llama's and Qwen3.5's RULER cells are trimmed to the NIAH tasks (finals/make_cells.py runs_of): the
# declared runs, which tables.py reads (the arms a unit carries follow them here); Qwen3-8B keeps the full mock design
TRIMMED = ("llama-3.1-8b-instruct", "qwen3.5-9b")


def trimmed_runs(model):
    l32 = [8192, 16384, 32768]
    out = [dict(id="ruler-niah", set="ruler", split="3/8", role="primary", tasks=list(NIAH), lengths=l32,
                arms=BATTERY_ARMS + QBLIND)]
    if model.startswith("llama"):
        out.append(dict(id="ruler-niah-splices", set="ruler", split="3/8", role="mechanism", tasks=list(NIAH),
                        lengths=l32, arms=["exact"] + list(RULER_SPLICES)))
        out.append(dict(id="ruler-niah-1-2", set="ruler", split="1/2", role="appendix", tasks=list(NIAH),
                        lengths=l32, arms=CORE))
    out.append(dict(id="ruler-64k", set="ruler", split="3/8", role="primary", tasks=list(NIAH), lengths=[65536],
                    arms=CORE))
    return out


# the 2026-10-08 follow-ups (finals/make_cells.py): the matched-depth RULER cells (Llama, Qwen3-8B), Llama's LongBench
# back (FINAL-LLAMA-LONGBENCH-2: the primary arms only, no SwiftKV run; the Qwen LongBench cells stay cut) and the
# fresh-data NLL cell (its own frozen set: records carry dataset and tier = that name)
DEEP_MODELS = ("llama-3.1-8b-instruct", "qwen3-8b")
LB_ARMS = ["exact", "nofill", "identity", "pop-native", "ridgefill", "identity+rows", "ridgefill+rows"]
FRESH_DATASET = "fresh-long-mock"
LAYER = {"llama-3.1-8b-instruct": {"1/2": 16, "5/8": 20}, "qwen3-8b": {"5/8": 22}}   # protocol.json splits


def _followup_projector(c, model, fracs):
    m = MODELS[model]
    c["projector"]["files"] = [{"layer": LAYER[model][f], "file": f"{m['proj'].rsplit('-s', 1)[0]}-s{LAYER[model][f]}"
                                                                    ".safetensors"} for f in fracs]
    return c


def cells():
    out = []
    for model, m in MODELS.items():
        short = model.split("-")[0].replace(".", "")
        arms = BATTERY_ARMS + (["swiftkv", "swiftkv-native"] if m["swiftkv"] else [])
        caps = {"max_len": m["max_len"]} if m["max_len"] else {}
        if model in TRIMMED:
            runs = trimmed_runs(model)
            out.append(_cell(f"mock-{short}-ruler", "ruler", model, sorted({a for r in runs for a in r["arms"]},
                                                                           key=(BATTERY_ARMS + QBLIND + list(RULER_SPLICES)).index),
                             f"FINAL-MOCK-{short.upper()}-RULER-2", sets=["ruler", "multiturn"], runs=runs, **caps,
                             margins=dict(MARGINS)))
        else:
            out.append(_cell(f"mock-{short}-ruler", "ruler", model, arms + list(RULER_SPLICES),
                             f"FINAL-MOCK-{short.upper()}-RULER", sets=["ruler", "ruler-qfirst", "multiturn"], **caps,
                             margins=dict(MARGINS)))
        # LongBench: cut for the Qwen models (registration.CUT); Llama's is back (FINAL-LLAMA-LONGBENCH-2)
        if model == "llama-3.1-8b-instruct":
            out.append(_cell(f"mock-{short}-longbench", "longbench", model, LB_ARMS, f"FINAL-MOCK-{short.upper()}-LB-2",
                             sets=["longbench"], margins=dict(MARGINS)))
        if model in DEEP_MODELS:
            runs = [dict(id="ruler-niah-deep", set="ruler", split="5/8", role="primary", tasks=list(NIAH),
                         lengths=[8192, 16384, 32768], arms=["exact", "identity", "ridgefill", "pop-native"])]
            out.append(_followup_projector(_cell(
                f"mock-{short}-ruler-deep", "ruler-deep", model, runs[0]["arms"], f"FINAL-MOCK-{short.upper()}-RULER-DEEP",
                sets=["ruler"], runs=runs, split="5/8", **caps, margins=dict(MARGINS)), model, ["5/8"]))
        if model == "llama-3.1-8b-instruct":
            runs = [dict(id="nll-fresh", set="longppl", split="1/2", role="primary",
                         arms=["exact", "identity", "ridgefill", "pop-native"]),
                    dict(id="nll-fresh-5-8", set="longppl", split="5/8", role="curve", arms=["exact", "identity", "ridgefill"])]
            out.append(_followup_projector(_cell(
                f"mock-{short}-nll-fresh", "nll-fresh", model, runs[0]["arms"], f"FINAL-MOCK-{short.upper()}-NLL-FRESH",
                sets=["longppl"], runs=runs, split="1/2", dataset=FRESH_DATASET, margins=dict(MARGINS)), model,
                ["1/2", "5/8"]))
        out.append(_cell(f"mock-{short}-nll-causal", "nll-causal", model, CAUSAL_ARMS,
                         f"FINAL-MOCK-{short.upper()}-NLL-CAUSAL", sets=["longppl"], **caps, margins=dict(MARGINS)))
        out.append(_cell(f"mock-{short}-nll", "nll", model, arms + SPLICES, f"FINAL-MOCK-{short.upper()}-NLL",
                         sets=["longppl", "decode"], split_curve=list(FRACS), appendix_split="1/2", **caps, margins=dict(MARGINS)))
        if model == "llama-3.1-8b-instruct":
            out.append(_cell(f"mock-{short}-speed", "speed-controls", model,
                             ["exact", "nofill", "identity", "pop-native", "ridgefill"],
                             f"FINAL-MOCK-{short.upper()}-SPEED", sets=["ttft"], split_curve=list(FRACS)))
    out.append(_cell("mock-flashnext-final", "flashnext", FN_MODEL, list(FN), "FINAL-MOCK-FN",
                     sets=["longppl", "ruler", "ttft"], split="-", split_layer=24, tail=2048, selector="class",
                     declared_share=25, criteria={"ppl_gap_upper_pct": 2.5}, data=copy.deepcopy(FN_DATA),
                     margins={"ppl_ratio_upper_pct": 1.0, "ruler_points_lower": -2.0, "ttft_min_speedup": 1.25,
                              "ttft_length": 16384}))
    return out


def _fn_data(cell, fixture):
    """The flashnext cell's declared counts at this fixture size (finals/make_cells.py engine_cells' 'data')."""
    cell["data"]["ruler"]["rows"] = 4 * 3 * fixture["fn_ruler"]
    cell["data"]["ruler"]["long"]["rows"] = fixture["fn_ruler"]
    cell["data"]["longppl"]["docs"] = len(DOMAINS) * fixture["fn_docs"]


# ----------------------------------------------------------------------------------------------- units per set
def units(cell, set_, fixture):
    if set_ == "ruler" and cell["kind"] == "flashnext":     # NIAH 8K-32K and niah_single_1 at 64K
        return [dict(sample=f"ruler/{t}_{L}.jsonl#{i:02d}", task=t, length=L, needle_filled=i % 4 != 0)
                for t in NIAH for L in (8192, 16384, 32768) for i in range(fixture["fn_ruler"])] + \
               [dict(sample=f"ruler/niah_single_1_65536.jsonl#{i:02d}", task="niah_single_1", length=65536,
                     needle_filled=i % 4 != 0) for i in range(fixture["fn_ruler"])]
    if set_ == "longppl" and cell["kind"] == "flashnext":   # the locked long-prompt documents <= 32K
        return [dict(sample=f"long-prompts/{d}_{8192 * 2 ** (k % 3)}_{k}.txt", doc=f"{d}_{k}", domain=d,
                     length=8192 * 2 ** (k % 3), tokens=8192 * 2 ** (k % 3) + 512)
                for d in DOMAINS for k in range(fixture["fn_docs"])]
    if set_ == "ttft" and cell["kind"] == "flashnext":
        return [dict(sample=f"{a}/{L}#round{r}", length=L, round=r, boot=a) for a in FN_SPEED_ARMS
                for L in (9216, 16384, 32768) for r in range(5)]
    if set_ in ("ruler", "ruler-qfirst", "multiturn"):
        n = fixture["ruler"] if set_ == "ruler" else fixture["qfirst"]
        if cell.get("runs") and set_ == "ruler":      # trimmed to 4 tasks: twice the rows keep the pooled population
            n *= 2
        lengths = RULER_LENGTHS if set_ != "ruler-qfirst" else (16384,)
        if cell.get("runs") and set_ == "ruler":
            lengths = sorted({n for r in cell["runs"] for n in r["lengths"]})
        out = []
        tasks = NIAH if cell.get("runs") and set_ == "ruler" else RULER_TASKS
        for t in tasks:
            for L in lengths:
                for i in range(n):
                    u = dict(sample=f"ruler/{t}_{L}.jsonl#{i:03d}", task=t, length=L,
                             needle_filled=(i % 5) != 4)
                    if cell.get("max_len") and L > cell["max_len"]:
                        u["skipped"] = f"over max_len {cell['max_len']} (no YaRN)"
                    out.append(u)
        if set_ == "multiturn":
            out = [dict(u, sample=u["sample"] + "+turn2") for u in out if u["length"] == 16384]
        return out
    if set_ == "longbench":
        return [dict(sample=f"longbench/{t}.jsonl#{i:03d}", task=t) for t in LB_TASKS for i in range(fixture["lb"])]
    if set_ in ("longppl", "decode"):
        docs = []
        for di, d in enumerate(DOMAINS):
            for k in range(fixture["docs"]):
                tokens = 8192 + ((di * 7 + k * 13) % 15) * 8192
                root = cell.get("dataset", "long-prompts")          # another frozen set: its own documents
                u = dict(sample=f"{root}/{d}_{k}.txt", doc=f"{d}_{k}", domain=d, tokens=tokens)
                if cell.get("max_len") and tokens > cell["max_len"]:
                    u["skipped"] = f"over max_len {cell['max_len']} (no YaRN)"
                docs.append(u)
        if set_ == "decode":
            docs = [u for u in docs if "skipped" not in u and u["tokens"] <= 32768][:fixture["decode"]]
        return docs
    if set_ == "ttft":
        lengths = (8192, 16384, 32768, 65536)
        return [dict(sample=f"ttft/{L}#round{r}", length=L, round=r) for L in lengths for r in range(5)]
    raise ValueError(set_)


def arm_splits(cell, arm):
    if cell["kind"] == "flashnext":
        return ["-"]
    if arm == "exact":
        return ["-"]
    if arm == "pop-native":
        return ["pop"]
    if arm in ("swiftkv", "swiftkv-native"):
        return ["1/2"]
    if arm in ("nofill", "identity", "ridgefill") and cell.get("split_curve"):
        return list(cell["split_curve"])
    return ["3/8"]


# ----------------------------------------------------------------------------------------------- metrics
def metrics(cell, set_, u, arm, split, eff):
    model = cell["model"]["name"]
    g = _rng(cell["name"], set_, u["sample"], "unit")             # shared across arms: the pairing correlation
    shared = g.random(), g.normal(), g.normal()
    a = _rng(cell["name"], set_, u["sample"], arm, split)
    if cell["kind"] == "flashnext":
        mu, p, speed = eff["fn"][arm]
        if set_ == "longppl":                                      # a deterministic engine: exact-rerun == exact
            base = 2.4 + 0.4 * shared[1]
            d = 0.0 if arm.startswith("exact") else mu + 0.004 * (0.6 * shared[2] + 0.8 * a.normal())
            return {"novel_nll": _r(base + d), "n_novel": 490}
        if set_ == "ruler":
            pp = p if u["needle_filled"] else 0.99
            ex = 1.3 * u["length"] / 1024 * (1 + 0.01 * shared[1])
            return {"score": float((0.8 * shared[0] + 0.2 * a.random()) < pp),
                    "wall_s": _r(ex / speed * (1 + 0.01 * a.normal()), 4)}
        if set_ == "ttft":
            ex = 0.6 * u["length"] / 1024 * (1 + 0.01 * shared[1])
            return {"ttft_s": _r(ex / speed * (1 + 0.01 * a.normal()), 4)}
    if set_ in ("ruler", "ruler-qfirst", "multiturn"):
        pmap = eff["ruler"][model]
        pp = pmap[QBLIND_P.get(arm, arm) if set_ == "ruler-qfirst" else arm] if u["needle_filled"] else 0.92
        key = "turn2" if set_ == "multiturn" else "score"
        return {key: float((0.8 * shared[0] + 0.2 * a.random()) < pp)}
    if set_ == "longbench":
        base = 0.35 + 0.3 * shared[0]
        mu = 0.0 if arm == "exact" else 0.4 * eff["nll"][model].get(arm, 0.1)
        return {"score": _r(min(1.0, max(0.0, base - mu + 0.03 * a.normal())))}
    if set_ == "longppl":
        base = 2.2 + 0.3 * shared[1]
        if arm == "exact":
            return {"novel_nll": _r(base), "key_nll": _r(base + 1.5), "n_novel": 400, "n_key": 30}
        mu = eff["nll"][model][arm] * (DEPTH_FACTOR[split] if split in DEPTH_FACTOR else 1.0)
        if cell["kind"] == "nll" and split == "3/8":
            mu += LEAK.get(arm, 0.0)
        d = mu + 0.03 * (0.7 * shared[2] + 0.7 * a.normal())
        return {"novel_nll": _r(base + d), "key_nll": _r(base + 1.5 + 2 * d), "n_novel": 400, "n_key": 30}
    if set_ == "decode":
        if arm == "exact":
            return {}
        ag = AGREE[arm] * (1.0 if split == "3/8" else 0.95)
        return {"top1_agree": _r(min(1.0, ag + 0.04 * a.normal())), "kl_mean": _r(max(0.0, (1 - ag) * 0.6 + 0.02 * a.normal()))}
    if set_ == "ttft":
        ex = 0.5 * u["length"] / 1024 * (1 + 0.01 * shared[1])
        if arm == "exact":
            sp = 1.0
        else:
            sp = SPEEDUP[arm] if split in ("3/8", "pop") else 1 + (SPEEDUP[arm] - 1) * {"1/4": 1.3, "1/2": 0.7, "5/8": 0.45}[split]
        return {"ttft_s": _r(ex / sp * (1 + 0.01 * a.normal()), 4)}
    raise ValueError(set_)


def record(cell, set_, u, canary_sha, eff):
    rec = {"cell": cell["name"], "kind": cell["kind"], "set": set_, "sample": u["sample"],
           "source_sha256": hashlib.sha256(u["sample"].encode()).hexdigest(), "canary_sha256": canary_sha,
           "deterministic": True, "code_tag": cell["code_tag"],
           "protocol": eff["cell_protocol"].get(cell["name"], eff["protocol"]), "mock": True,
           "tier": "engine-final" if cell["kind"] == "flashnext" else cell.get("dataset", "locked-final")}
    if cell.get("dataset"):
        rec["dataset"] = cell["dataset"]                          # finals/convert.py record_tier
    rec.update({k: v for k, v in u.items() if k != "sample"})
    if "skipped" not in u:
        rec["arms"] = {arm: {s: metrics(cell, set_, u, arm, s, eff) for s in _splits(cell, set_, arm, u)}
                       for arm in cell["arms"] if _fn_carries(cell, set_, arm, u) and (
                           cell["kind"] == "flashnext" or _in_set(set_, arm, u) and _in_runs(cell, set_, arm, u))}
    return rec


def _fn_carries(cell, set_, arm, u):
    """The Flash-Next arms a unit carries: every arm on a document, all but exact-rerun on a RULER row, a TTFT round its
    own arm and that boot's exact path."""
    if cell["kind"] != "flashnext":
        return True
    if set_ == "ruler":
        return arm != "exact-rerun"
    if set_ == "ttft":
        return arm in ("exact", u["boot"])
    return True


def _in_set(set_, arm, u):
    """Which arms a unit carries (decode: the fills; multiturn: T1.8's arms; question-first: exact and the rows arms;
    the splices: every NLL document, the NIAH rows of RULER (kexact, vexact), nowhere else)."""
    if set_ == "decode":
        return arm in ("exact", "nofill", "identity", "ridgefill")
    if set_ == "multiturn":
        return arm in ("exact", "ridgefill", "identity+rows", "ridgefill+rows")
    if set_ == "ruler-qfirst":
        return arm in ("exact", "identity+rows", "ridgefill+rows")
    if arm in SPLICES:
        return set_ == "longppl" or (set_ == "ruler" and arm in RULER_SPLICES and u["task"] in NIAH
                                     and u["length"] <= 32768)
    return True


def _covers(r, set_, u):
    return (r.get("set", set_) == set_ and ("tasks" not in r or u.get("task") in r["tasks"])
            and ("lengths" not in r or u.get("length") in r["lengths"]))


def _splits(cell, set_, arm, u):
    """The split keys of an arm on a unit: a declared-runs cell's runs that cover the unit (exact: "-")."""
    if not cell.get("runs") or set_ not in {r.get("set") for r in cell["runs"]} or arm in ("exact", "pop-native"):
        return arm_splits(cell, arm)
    return sorted({r["split"] for r in cell["runs"] if arm in r["arms"] and _covers(r, set_, u)})


def _in_runs(cell, set_, arm, u):
    """A cell with declared runs: the unit carries the arms of the runs that cover it."""
    if not cell.get("runs") or set_ not in {r.get("set") for r in cell["runs"]}:
        return True
    return any(arm in r["arms"] and _covers(r, set_, u) for r in cell["runs"])


# ----------------------------------------------------------------------------------------------- pre-final sources
def _safetensors_header(meta):
    h = json.dumps({"__metadata__": {k: str(v) for k, v in meta.items()}}).encode()
    return len(h).to_bytes(8, "little") + h


def _ledger_line(prev_lines, rec):
    rec = dict(rec, seq=len(prev_lines) + 1, prev=hashlib.sha256((prev_lines[-1] if prev_lines else "").encode()).hexdigest(),
               utc="2026-10-04T00:00:00+00:00")
    prev_lines.append(json.dumps(rec, sort_keys=True))


def build(dest, overrides=None, fixture=None):
    """Write the tree under dest; returns dest. overrides: {'fn': {arm: tuple}, 'ruler': {...}, 'nll': {...},
    'drop_cells': [...], 'protocol': '1.1.0', 'cell_protocol': {cell name: version}, 'ledger': [record, ...],
    'posthoc_record': False} for controls (default protocol: the repo's protocol.json version; 'ledger': extra records
    appended to the ledger; 'posthoc_record': False leaves out FINAL-LLAMA-RULER-ROWS)."""
    dest = Path(dest)
    ov = overrides or {}
    fx = dict(ruler=5, qfirst=4, lb=6, docs=4, decode=8, fn_ruler=16, fn_docs=10)
    fx.update(fixture or {})
    eff = {"ruler": copy.deepcopy(RULER_P), "nll": copy.deepcopy(NLL_MU), "fn": copy.deepcopy(FN),
           "protocol": ov.get("protocol", json.loads((FINALS.parent / "protocol.json").read_text())["version"]),
           "cell_protocol": ov.get("cell_protocol", {})}
    for k in ("ruler", "nll", "fn"):
        for key, val in ov.get(k, {}).items():
            if k == "fn":
                eff["fn"][key] = val
            else:
                eff[k][key[0]][key[1]] = val
    res = dest / "finals-results"
    proj = dest / "proj"
    proj.mkdir(parents=True, exist_ok=True)
    lines = []
    all_cells = cells()
    for model, m in MODELS.items():
        p = proj / f"{m['proj']}.safetensors"
        p.write_bytes(_safetensors_header({"model": model, "lam": m["lam"], "rows": m["rows"], "docs": 98,
                                           "split": m["proj"].rsplit("s", 1)[1], "note": "MOCK projector header"}))
        for c in all_cells:
            if c["model"]["name"] != model:
                continue
            for f in c["projector"]["files"]:
                q = proj / f["file"]
                if not q.is_file():                                  # a follow-up's projector at its own split
                    q.write_bytes(_safetensors_header({"model": model, "lam": m["lam"], "rows": m["rows"], "docs": 98,
                                                       "split": f["layer"], "note": "MOCK projector header"}))
                f["sha256"] = atomic.sha256_file(q)
    for c in all_cells:
        if c["name"] in ov.get("drop_cells", ()):
            continue
        if c["kind"] == "flashnext":
            _fn_data(c, fx)
        d = res / c["name"]
        canary = {"arms": {"exact": {"logits_sha256": hashlib.sha256(c["name"].encode()).hexdigest()}},
                  "host": {"host": "mock"}}
        atomic.write_result(d / "canary.json", canary)
        canary_sha = atomic.sha256_file(d / "canary.json")
        atomic.write_result(d / "cell.json", c)
        atomic.write_result(d / "host.json", {"fingerprint": "mock", "host": {"host": "mock", "gpu_name": "NVIDIA MOCK-GPU"}})
        for set_ in c["sets"]:
            for u in units(c, set_, fx):
                safe = u["sample"].replace("/", "__").replace("#", "_")
                atomic.write_result(d / "samples" / set_ / f"{safe}.json", record(c, set_, u, canary_sha, eff))
        if c["kind"] == "flashnext":                  # finals/convert_c5.py's engine.json: the determinism read
            n = c["data"]["longppl"]["docs"]
            atomic.write_result(d / "engine.json", dict(
                cell=c["name"], image="mock/vllm:0", protocol=eff["protocol"], overlay={},
                determinism=dict(docs=n, bitwise_equal=ov.get("fn_bitwise_equal", n)),
                control=dict(docs=2, bitwise_equal=2)))
        tree = atomic.treehash(d)
        atomic.write_bytes(d / ".pushed", json.dumps({"ssh": {"tree": tree}, "hf": {"tree": tree}}).encode())
        atomic.write_bytes(d / ".done", tree.encode())
        # a FINAL- record as finals/register.sh writes one: free text whose claim pins the cell's sha256 (labbook
        # stores text only); these mock cells have no harness configs (no kvabackend run), so only the cell is pinned
        pin = registration_pins_text({"cell": {c["name"]: atomic.sha256_file(d / "cell.json")}, "config": {}})
        _ledger_line(lines, {"type": "hypothesis", "id": c["registration"], "claim": "MOCK registration | " + pin,
                             "predict": "mock", "kill": "mock"})
    # pre-final sources ---------------------------------------------------------------------------
    (dest / "results" / "tables").mkdir(parents=True, exist_ok=True)
    prompts = [f"p{i:02d}" for i in range(23)]
    g = _rng("battery")
    e12 = {p: _r(0.4 + 0.2 * g.random()) for p in prompts}
    e16 = {p: _r(e12[p] + 0.19 + 0.12 * g.normal()) for p in prompts}
    battery = {"protocol": "1.0.2", "grade": [
        {"model": "llama-3.1-8b-instruct", "set": "grade", "format": "raw", "split": s, "tail": 4096, "arm": "noproj",
         "per_prompt": pp} for s, pp in ((12, e12), (16, e16))]}
    bpath = dest / "results" / "tables" / "battery-v1.0.json"
    bpath.write_text(json.dumps(battery, indent=1) + "\n")
    for model, (analytic, measured) in (("llama", (1.71, 1.61)), ("qwen", (1.63, 1.42))):
        (dest / "results" / f"timing-sweep-{model}.json").write_text(json.dumps({"runs": {
            "exact": {"split": None, "tail": None, "analytic_speedup": 1.0, "measured_speedup": 1.0},
            "kva-s12-t4096": {"split": 12, "tail": 4096, "analytic_speedup": analytic, "measured_speedup": measured}}}))
    (dest / "paper").mkdir(exist_ok=True)
    (dest / "paper" / "LEVERS-PAPER.md").write_text(
        "# MOCK lever list\n\nProse the reader skips | even with a pipe.\n\n"
        "| # | Lever | What it is | Where tested | Status | Evidence |\n|---|---|---|---|---|---|\n"
        "| 1 | Split depth | MOCK-WHAT | all | MOCK outcome (dev, v1.0.2) | MOCK-1 |\n"
        "| 2 | Exact tail length & block rounding | MOCK-WHAT | Flash-Next engine, run noise about +-0.01 nats/doc | "
        "MOCK outcome_with_specials 50% (dev, v1.0.2) | MOCK-2 |\n")
    rel = "../../"
    _ledger_line(lines, {"type": "record", "label": "battery-regenerated-20261004", "values": [0],
                         "sources": {rel + "results/tables/battery-v1.0.json": atomic.sha256_file(bpath)},
                         "note": "MOCK pin"})
    _ledger_line(lines, {"type": "record", "label": "timing-pin", "values": [0], "note": "MOCK pin",
                         "sources": {rel + f"results/timing-sweep-{m}.json": atomic.sha256_file(dest / "results" / f"timing-sweep-{m}.json")
                                     for m in ("llama", "qwen")}})
    _ledger_line(lines, {"type": "record", "label": "levers-pin", "values": [0], "note": "MOCK pin",
                         "sources": {rel + "paper/LEVERS-PAPER.md": atomic.sha256_file(dest / "paper" / "LEVERS-PAPER.md")}})
    _ledger_line(lines, {"type": "record", "label": "s16probe-results", "note": "MOCK",
                         "values": [0.1905, 0.1897, -0.0278, 0.1882, 0.2102, 0.2049, -0.0018, 0.2002,
                                    -0.0016, -0.0227, -0.0063, 0.0718]})
    for lab, vals in (("neg-datasize", [0.001, -0.004, 0.006]),
                      ("neg-q1b", [-0.0000022, -0.0011, 0.00092]),
                      ("neg-wf1", [-0.054, -0.125, 0.0]),
                      ("neg-fq3", [0.022, -0.001, 0.044])):
        _ledger_line(lines, {"type": "record", "label": lab, "values": vals, "note": "MOCK"})
    if ov.get("posthoc_record", True):                  # the analysis-only follow-up the tables read (C3_POSTHOC)
        _ledger_line(lines, {"type": "hypothesis", "id": "FINAL-LLAMA-RULER-ROWS", "claim": "MOCK analysis-only follow-up",
                             "predict": "MOCK"})
    for rec in ov.get("ledger", ()):
        _ledger_line(lines, rec)
    led = dest / "ledger" / "paper"
    led.mkdir(parents=True, exist_ok=True)
    (led / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    return dest


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "mock-tables-tree"))
