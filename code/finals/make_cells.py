#!/usr/bin/env python3
"""Write (or --check) finals/cells/*.json: the declared cells of the reference-model finals (OUTLINE.md T1, T2, T3.1)
under protocol 1.1, from the tables below. Deterministic; --check fails when a committed cell differs from what this
script writes (the cells are generated, never edited by hand: a change goes here, then into a NEW FINAL- record).

Scope (the coordinator's 2026-10-04 evening scope; exactly this, nothing more). Primary split 3/8 (S12, S12, S13),
tail 4096. NIAH = the 4 NIAH tasks; L32 = 8K/16K/32K.
  ruler     primary arms (exact, nofill, identity, pop-native, ridgefill, identity+rows, ridgefill+rows) on all 8 tasks
            at L32; at 64K exact, identity, ridgefill and both rows arms (not Qwen3-8B); the T1.5 controls (random rows
            and tail matches for both fills, nofill + tail budget) on the NIAH subset at L32, in the same config as
            those rows' primary arms (one pairing unit); the T2.1 RULER splices ridgefill+kexact and ridgefill+vexact
            on the NIAH subset at L32, same config (OUTLINE C4: "exact keys alone restore retrieval on both"; the
            coordinator's alignment scope, 2026-10-04); appendix 1/2 (identity, ridgefill, both rows) at L32, Llama
            only; T1.4 question-first set (NIAH, L32): exact, identity+rows, ridgefill+rows; SwiftKV (Llama): as
            released and maps + exact tail, L32. No split curve on RULER.
  nll       the split curve 1/4, 3/8, 1/2, 5/8 for nofill, identity, ridgefill; pop-native (the split-curve table's POP
            native row); at 3/8 the rows arms identity+rows and ridgefill+rows with their random-rows and tail-match
            controls and nofill+tailbudget (T1.5 on NLL), the splices kexact, vexact, kvexact and ktail, vtail; on Llama
            SwiftKV as released and its maps + exact tail at S16 (R6, its own config on the SwiftKV checkpoint);
            key-token NLL (tail 4096, threshold 2.0); 64K-128K documents for Llama and Qwen3.5 only (the coordinator's
            decisions of 2026-10-04 night). Qwen3-8B's NLL cell was registered (FINAL-QWEN3-8B-NLL) before these arms;
            before any run it is re-registered as FINAL-QWEN3-8B-NLL-2 (registration.SUPERSEDES)
  longbench primary arms at 3/8 (+ SwiftKV as released on Llama)
  decode    T1.7: 20 documents (kva.derived.pick_decode_docs), exact, nofill, identity, ridgefill
  secondq   T1.8: the 100 two-key rows (Qwen3-8B 75: <= 32K), exact, ridgefill, identity+rows, ridgefill+rows
  speed     T3.1 (Llama only): refused by name (TODO-HARNESS-VLLM-TTFT) until the engine path exists
Each RULER config is a partition of the rows (NIAH L32 / other tasks L32 / 64K / question-first / SwiftKV), so a row's
arms live in one record. Qwen3-8B runs at <= 32K (protocol.json max_len; no YaRN): caps and max_len 32768, the files
over it in 'skipped', by name.
Constants measured once (their tests recompute them where the inputs exist: tests/test_finals_wiring.py,
tests/test_refit_clean.py):
  CALIB_ROWS / CALIB_DOCS  the clean calibration list (kva.data.calib_ids(tok, 'train') after cba2d91): rows per
              tokenizer, 71 documents; a projector whose metadata disagrees is refused as stale (kvabackend)
  TAILBUDGET  b of nofill+tailbudget, per cell kind: the fill's modelled cost (kva/compute.py tail_budget at S, T 4096)
              as a share of N, the maximum over the lengths the control runs at (RULER 8K-32K; NLL the documents'
              nominal lengths, 8K-128K, Qwen3-8B 8K-32K), rounded up to 3 decimals
Projector sha256: read from the clean fit report results/fit-sweep-<model>-s<S>.json (its rows must be the clean
count, and the fit's calibration document count CALIB_DOCS: the report's 'docs', or for the Qwen3-8B sweep, which
predates that field, results/calib-docs-qwen3-8b-sweep.json, the list its capture state recorded); absent until the
clean refit lands, which leaves the file unpinned (cellcheck warns; register.sh and run_final.sh refuse the cell by
name).
A registered cell never changes: before writing (and in --check) every cell whose FINAL- id has a ledger record must
generate exactly the cell and configs that record pins (registration.check_record), else RegisteredCellChanged by name
(a change after registration needs a new FINAL- id).
The engine tier (Flash-Next, C5) is not a harness cell: engine_cells() writes its one cell to finals/engine/, generated
from FINAL-FLASHNEXT-C5's declared values and checked against them (registration.check_engine_record); FINAL ids of
registration.ENGINE and registration.CANCELLED need no cell in cells/.
Usage: make_cells.py [--check] [--ledger L]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

FINALS = Path(__file__).resolve().parent
REPO = FINALS.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FINALS / "lib"))
from kva import derived  # noqa: E402
import refit  # noqa: E402
import registration  # noqa: E402
from registration import SUPERSEDES, canonical, registration_id  # noqa: E402

FITTED_AFTER = "cba2d91"
CALIB_ROWS = {"llama-3.1-8b-instruct": 834541, "qwen3.5-9b": 839652, "qwen3-8b": 845043}
CALIB_DOCS = 71
TAILBUDGET = {"ruler": {"llama-3.1-8b-instruct": 0.186, "qwen3.5-9b": 0.203, "qwen3-8b": 0.203},
              "nll": {"llama-3.1-8b-instruct": 0.186, "qwen3.5-9b": 0.211, "qwen3-8b": 0.203}}
MARGINS = {"ppl_ratio_upper_pct": 1.0, "ruler_points_lower": -2.0}       # TOST: perplexity +1% (0.01 relative), RULER -2
MODELS = {
    "llama-3.1-8b-instruct": dict(id="unsloth/Llama-3.1-8B-Instruct", commit="4699cc75b550f9c6f3173fb80f4703b62d946aa5",
                                  layers=dict(zip(("1/4", "3/8", "1/2", "5/8"), (8, 12, 16, 20))),
                                  caps=dict(ruler_max_len=65536, longbench_max_len=31500)),
    "qwen3.5-9b": dict(id="Qwen/Qwen3.5-9B", commit="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                       layers=dict(zip(("1/4", "3/8", "1/2", "5/8"), (8, 12, 16, 20))),
                       caps=dict(ruler_max_len=65536, longbench_max_len=31500)),
    "qwen3-8b": dict(id="Qwen/Qwen3-8B", commit="b968826d9c46dd6066d109eabc6255188de91218",
                     layers=dict(zip(("1/4", "3/8", "1/2", "5/8"), (9, 13, 18, 22))),
                     caps=dict(ruler_max_len=32768, longbench_max_len=31500, longppl_max_len=32768), max_len=32768),
}
SWIFTKV = dict(id="Snowflake/Llama-3.1-SwiftKV-8B-Instruct", name="llama-3.1-swiftkv-8b-instruct",
               commit="22a50b8c022dd8f3c6800ae1c96b54d46d89b840")   # HF API checked 2026-10-04: this id, this commit
NIAH = list(derived.QFIRST_TASKS)
OTHER = ["vt", "cwe", "fwe", "qa_1"]
L32 = [8192, 16384, 32768]
PRIMARY = ["exact", "nofill", "identity", "pop-native", "ridgefill", "identity+rows", "ridgefill+rows"]
CORE = ["exact", "identity", "ridgefill", "identity+rows", "ridgefill+rows"]
CONTROLS = ["identity+random", "ridgefill+random", "identity+tailmatch", "ridgefill+tailmatch", "nofill+tailbudget"]
CURVE = ["exact", "nofill", "identity", "ridgefill"]
SPLICES = ["ridgefill+kexact", "ridgefill+vexact", "kvexact", "ridgefill+ktail", "ridgefill+vtail"]
RULER_SPLICES = ["ridgefill+kexact", "ridgefill+vexact"]          # T2.1 on retrieval (C4), NIAH subset only
QBLIND = ["identity+qblind", "ridgefill+qblind"]                  # qblind10: a cache built before the question exists
# Protocol 1.1.3 (Dylan, 2026-10-07): the RULER cells that had not run are trimmed to the NIAH tasks (they supersede
# their registrations, registration.SUPERSEDES), LongBench is cut (registration.CUT), and each model gets an
# nll-causal cell re-running the NLL arms whose 1.1.1 evaluation read the scored continuation.
TRIMMED_RULER = ("llama-3.1-8b-instruct", "qwen3.5-9b")
CAUSAL = ["exact", "identity", "ridgefill", "identity+rows", "ridgefill+rows", "identity+random", "ridgefill+random",
          "identity+tailmatch", "ridgefill+tailmatch", "ridgefill+ktail", "ridgefill+vtail"]
NOTES = {
    "ruler": "T1.1 (8 tasks, 25 rows per task and length, official scorer, filled-region needles primary) at S{s}; T1.5 "
             "controls and T1.4 question-first on the NIAH subset at 8K-32K; 64K rows for the core arms",
    "longbench": "T1.2 (secondary): 8 tasks x 50 rows, official metrics, middle truncation 31500, primary split only",
    "nll": "T1.3 (126 documents, novel + key-token NLL per length bin) with the split curve; T2.1 splices and T2.2 "
           "ktail/vtail at S{s}",
    "decode": "T1.7: 512 greedy tokens after the 20 documents in 'decode.only'",
    "secondq": "T1.8: the locked two-key RULER set (datasets/derived/ruler-twokey.json)",
    "speed": "T3.1 (Llama, measured TTFT at 8K-64K, all four splits): refused by name until the engine path exists "
             "(TODO-HARNESS-VLLM-TTFT); scripts/ttft.py is the runner",
    "ruler-trimmed": "T1.1 trimmed to the NIAH tasks (filled-region needles primary) at S{s}, T1.5 controls and the "
                     "question-blind rows pair; protocol 1.1.2 generation cap",
    "nll-fresh": "follow-up confirmation (2026-10-08) on the fresh frozen long-document set fresh-long-20261008: "
                 "RidgeFill vs identity at S{s} and 5/8, and vs POP as published",
    "ruler-deep": "follow-up (2026-10-08): NIAH filled-region retrieval at the 5/8 split S{s} against POP as published "
                  "(its own, shallower split): matched or deeper skipping; protocol 1.1.2 generation cap",
    "nll-causal": "protocol 1.1.3 correction: the NLL arms whose 1.1.1 evaluation read the scored continuation, re-run "
                  "at S{s} with plain fills and controls alongside",
}


def run(id_, set_, split, role, arms, **kw):
    return dict(id=id_, set=set_, split=split, role=role, arms=list(arms), **kw)


def runs_of(model, kind):
    llama, small = model.startswith("llama"), model == "qwen3-8b"
    if kind == "ruler" and model in TRIMMED_RULER:
        out = [run("ruler-niah", "ruler", "3/8", "primary", PRIMARY + CONTROLS + QBLIND, tasks=NIAH, lengths=L32)]
        if llama:
            out += [run("ruler-niah-splices", "ruler", "3/8", "mechanism", ["exact"] + RULER_SPLICES, tasks=NIAH,
                        lengths=L32),
                    run("ruler-niah-1-2", "ruler", "1/2", "appendix", CORE, tasks=NIAH, lengths=L32)]
        # 64K NIAH rows on both: the pooled filled-region population must reach the non-inferiority floor's n >= 149
        # (finals/stats.py noninferiority: -2 points needs 149 eligible rows; 8K-32K alone gives about 138)
        out.append(run("ruler-64k", "ruler", "3/8", "primary", CORE, tasks=NIAH, lengths=[65536]))
        return out
    if kind == "nll-causal":
        return [run("nll", "longppl", "3/8", "primary", CAUSAL)]
    if kind == "nll-fresh":                        # Dylan 2026-10-08: confirm the post-hoc S16/S20 reads on fresh documents
        return [run("nll-fresh", "longppl", "1/2", "primary", ["exact", "identity", "ridgefill", "pop-native"]),
                run("nll-fresh-5-8", "longppl", "5/8", "curve", ["exact", "identity", "ridgefill"])]
    if kind == "ruler-deep":                       # Dylan 2026-10-08: RULER at matched/deeper skipping vs POP as published
        return [run("ruler-niah-deep", "ruler", "5/8", "primary", ["exact", "identity", "ridgefill", "pop-native"],
                    tasks=NIAH, lengths=L32)]
    if kind == "ruler":
        out = [run("ruler-niah", "ruler", "3/8", "primary", PRIMARY + CONTROLS, tasks=NIAH, lengths=L32),
               run("ruler-niah-splices", "ruler", "3/8", "mechanism", ["exact"] + RULER_SPLICES, tasks=NIAH, lengths=L32)]
        if llama:
            out.append(run("ruler-niah-1-2", "ruler", "1/2", "appendix", CORE, tasks=NIAH, lengths=L32))
        out.append(run("ruler-other", "ruler", "3/8", "primary", PRIMARY, tasks=OTHER, lengths=L32))
        if llama:
            out.append(run("ruler-other-1-2", "ruler", "1/2", "appendix", CORE, tasks=OTHER, lengths=L32))
        if not small:
            out.append(run("ruler-64k", "ruler", "3/8", "primary", CORE, lengths=[65536]))
        out.append(run("ruler-qfirst", "ruler", "3/8", "primary", ["exact", "identity+rows", "ridgefill+rows"],
                       variant="qfirst", tasks=NIAH, lengths=L32))
        if llama:
            out.append(run("ruler-swiftkv", "ruler", "1/2", "swiftkv", ["exact", "swiftkv-native", "swiftkv"],
                           model="swiftkv", lengths=L32))
        return out
    if kind == "longbench":                        # re-added for Llama only on 2026-10-08, primary arms, no SwiftKV run
        return [run("longbench", "longbench", "3/8", "primary", PRIMARY)]
    if kind == "nll":
        rows = ["identity+rows", "ridgefill+rows"]
        return [run("nll", "longppl", "3/8", "primary", CURVE + ["pop-native"] + rows + CONTROLS + SPLICES)] + [
            run(f"nll-{f.replace('/', '-')}", "longppl", f, "curve", CURVE) for f in ("1/4", "1/2", "5/8")] + (
            [run("nll-swiftkv", "longppl", "1/2", "swiftkv", ["exact", "swiftkv-native", "swiftkv"], model="swiftkv")]
            if llama else [])
    if kind == "decode":
        return [run("decode", "decode", "3/8", "primary", CURVE)]
    if kind == "secondq":
        return [run("secondq", "multiturn", "3/8", "primary", ["exact", "ridgefill", "identity+rows", "ridgefill+rows"])]
    if kind == "speed":
        return [run(f"ttft-{f.replace('/', '-')}", "ttft", f, "primary" if f == "3/8" else "curve",
                    PRIMARY[:5] if f == "3/8" else CURVE) for f in ("1/4", "3/8", "1/2", "5/8")]
    raise ValueError(kind)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_paths():
    rows = (REPO / "datasets" / "test-set-locked" / "MANIFEST.tsv").read_text().splitlines()[1:]
    return [r.split("\t")[0] for r in rows if r.strip()]


def nominal(path):
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return int(stem.rsplit("_", 2)[-2]) if path.startswith("long-prompts/") else int(stem.rsplit("_", 1)[-1])


def skipped(model, runs):
    """The locked files of the cell's sets that no run reads because of the model's max_len, by name (sorted)."""
    cap = MODELS[model].get("max_len")
    if not cap:
        return []
    paths, sets = manifest_paths(), {r["set"] for r in runs}
    out = set()
    if sets & {"ruler"}:
        out |= {p for p in paths if p.startswith("ruler/") and nominal(p) > cap}
    if "multiturn" in sets:
        out |= {p for p in derived.sources("twokey") if nominal(p) > cap}
    if sets & {"longppl", "decode"}:
        out |= {p for p in paths if p.startswith("long-prompts/") and p.endswith(".txt") and nominal(p) > cap}
    return sorted(out)


def clean_sha256(model, layer):
    """The projector sha256 from its clean fit report, or None until the clean refit lands: the report's rows must be
    the clean count and the fit's calibration document count (refit.docs_count: the report's 'docs', or the list the
    fit recorded, results/calib-docs-<model>-sweep.json) the declared CALIB_DOCS, so the pin carries both."""
    path = REPO / "results" / f"fit-sweep-{model}-s{layer}.json"
    if not path.is_file():
        return None
    rep = json.loads(path.read_text())
    sha = rep.get("projector_sha256")
    if rep.get("rows") != CALIB_ROWS[model] or refit.docs_count(REPO, model, layer, sha) != CALIB_DOCS:
        return None
    return sha


def projector_block(model, runs):
    m = MODELS[model]
    layers = sorted({m["layers"][r["split"]] for r in runs if r.get("model") != "swiftkv"
                     and any(a.startswith("ridgefill") or a == "kvexact" for a in r["arms"])})
    files = []
    for layer in layers:
        f = {"layer": layer, "file": f"{model}-sweep-s{layer}.safetensors"}
        sha = clean_sha256(model, layer)
        if sha:
            f["sha256"] = sha
        files.append(f)
    return dict(provenance=dict(calib_rows=CALIB_ROWS[model], calib_docs=CALIB_DOCS, fitted_after=FITTED_AFTER,
                                calib_exclude_sha256=sha_file(REPO / "datasets" / "calibration-exclude.json")),
                files=files)


def registration_of(model, kind):
    """The cell's FINAL- id: FINAL-<MODEL>-<KIND>, or its successor when that id was superseded before any run."""
    rid = registration_id(model, kind)
    return SUPERSEDES.get(rid, rid)


def cell(model, kind):
    m = MODELS[model]
    runs = runs_of(model, kind)
    frozen = json.loads((REPO / "datasets" / "test-set-locked" / "FROZEN.json").read_text())["manifest_sha256"]
    v113 = kind in ("nll-causal", "ruler-deep", "longbench", "nll-fresh") or (kind == "ruler" and model in TRIMMED_RULER)
    note = NOTES["ruler-trimmed" if kind == "ruler" and v113 else kind]
    caps = dict(m["caps"])
    if kind in ("ruler", "ruler-deep") and v113:     # protocol 1.1.2's generation cap, the protocol's own values
        caps["ruler_max_new"] = dict(json.loads((REPO / "protocol.json").read_text())["ruler_max_new"]["tasks"])
    c = dict(name=f"{model}-{kind}", kind=kind, model=dict(id=m["id"], name=model, commit=m["commit"]), split="3/8",
             tail=4096, runs=runs, caps=caps, margins=dict(MARGINS), dataset_manifest_sha256=frozen,
             deterministic=True, registration=registration_of(model, kind),
             code_tag="protocol-v1.1.3" if v113 else "protocol-v1.1", note=note.format(s=m["layers"]["3/8"]))
    if kind == "ruler-deep":                         # its one run is at 5/8: the cell's primary split
        c["split"], c["note"] = "5/8", NOTES[kind].format(s=m["layers"]["5/8"])
    if kind == "nll-fresh":                          # the fresh frozen set (datasets/fresh-long-20261008), not the locked one
        c["split"], c["note"] = "1/2", NOTES[kind].format(s=m["layers"]["1/2"])
        c["dataset"] = "fresh-long-20261008"
        c["dataset_manifest_sha256"] = json.loads((REPO / "datasets" / "fresh-long-20261008" / "FROZEN.json").read_text())["manifest_sha256"]
    if "max_len" in m:
        c["max_len"] = m["max_len"]
    if any(r.get("model") == "swiftkv" for r in runs):
        c["swiftkv_model"] = dict(SWIFTKV)
    arms = {a for r in runs for a in r["arms"]}
    if any(a.startswith("ridgefill") or a == "kvexact" for a in arms):
        c["projector"] = projector_block(model, runs)
    if "nofill+tailbudget" in arms:
        span = "8K-32K" if kind == "ruler" or "max_len" in m else "8K-128K"
        what = "the controls' lengths" if kind == "ruler" else "the documents' nominal lengths"
        c["tailbudget"] = dict(b=TAILBUDGET[kind][model], basis="kva/compute.py tail_budget at the primary split, tail "
                               f"4096: the fill's modelled cost as extra tail rows / N, max over {what} ({span}), "
                               "rounded up (finals/make_cells.py)")
    sets = {r["set"] for r in runs}
    if "longppl" in sets:
        c["keytoken"] = dict(tail=4096, threshold=2.0)
    if "decode" in sets:
        c["decode"] = {"from": "longppl", "steps": 512, "only": derived.pick_decode_docs(manifest_paths())}
    if "multiturn" in sets:
        c["secondq"] = {"from": "ruler", "turns": 2}
    pins = {f"ruler-{w}": sha_file(derived.manifest_path(w)) for w in derived.SETS
            if (w == "twokey" and "multiturn" in sets) or (w == "qfirst" and any(r.get("variant") == "qfirst" for r in runs))}
    if pins:
        c["derived"] = pins
    skip = skipped(model, runs)
    if skip:
        c["skipped"] = skip
    return c


def cells():
    kinds = ("ruler", "nll", "decode", "secondq", "nll-causal", "ruler-deep", "longbench", "speed")
    kinds = kinds + ("nll-fresh",)
    keep = {"speed": ("llama-3.1-8b-instruct",), "longbench": ("llama-3.1-8b-instruct",),   # LongBench: cut elsewhere
            "nll-fresh": ("llama-3.1-8b-instruct",),
            "ruler-deep": ("llama-3.1-8b-instruct", "qwen3-8b")}                           # (registration.CUT)
    return [cell(m, k) for m in MODELS for k in kinds if m in keep.get(k, (m,))]


# The engine tier (Flash-Next, claim C5; registration.ENGINE): run by the C5 kit (c5-kit, run_c5.sh MODE final, run
# final-det-20261010) on a local machine, one server boot per arm, not by the finals harness. Its cell is generated here
# from FINAL-FLASHNEXT-C5's declared values (registration.check_engine_record checks them against the record) and the
# amendments that changed the run before it started (10: the deterministic engine build of protocol 1.1.6, instruct-
# format RULER, the fresh run) or the reading after it (11: the perplexity criterion = the measured gap to exact). Arms:
# the paper's names; 'from' = the kit's arm directory and pass. Written to finals/engine/ (not cells/: no harness runs
# it), read by finals/convert_c5.py and verify_finals.py.
ENGINE_DIR = FINALS / "engine"
FN_DATA = dict(longppl=dict(docs=90, domains=6, max_nominal=32768, scored=512),
               ruler=dict(tasks=NIAH, lengths=L32, rows=300, long=dict(task="niah_single_1", length=65536, rows=25)),
               ttft=dict(lengths=[9216, 16384, 32768], warmup=1, timed=5, prompt="long-prompts/chat_32768_0.txt"),
               control=dict(docs=6, what="the first 6 documents of the perplexity set on quality mode's exact path"))


def engine_cells():
    p = json.loads((REPO / "protocol.json").read_text())
    img = p["engine_tier"]["c5_image"]
    frozen = json.loads((REPO / "datasets" / "test-set-locked" / "FROZEN.json").read_text())["manifest_sha256"]
    return [dict(
        name="qwen3.8-flash-next-c5", kind="flashnext", driver="c5-kit", registration="FINAL-FLASHNEXT-C5",
        model=dict(id="tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ", name="qwen3.8-flash-next", commit="6fe3e7464c18"),
        protocol="1.1.6", tier="engine-final", code_tag="c5-final-det", sets=["longppl", "ruler", "ttft"],
        split="-", split_layer=24, tail=2048, declared_share=25,
        margins=dict(ppl_ratio_upper_pct=1.0, ruler_points_lower=-2.0, ttft_min_speedup=1.25, ttft_length=16384),
        criteria=dict(ppl_gap_upper_pct=2.5),
        amendments={"amend-FINAL-FLASHNEXT-C5-10": "deterministic engine build of protocol 1.1.6",
                    "amend-FINAL-FLASHNEXT-C5-11": "is at most log(1.025) (+2.5%)",
                    "det-read-FINAL-FLASHNEXT-C5": "90/90 bitwise-equal"},
        engine=dict(image=img["image"], id=img["id"], overlay=dict(img["overlay"]["files"]),
                    overlay_sums_sha256=img["overlay"]["sha256sums_sha256"], kit="c5-kit d1164746aa21b3f339db6d8bd2919601db450d7f",
                    run="final-det-20261010"),
        arms={"exact": dict(dir="exact", ppl="ppl-a", ruler="ruler-a", policy=None),
              "exact-rerun": dict(dir="exact", ppl="ppl-b", policy=None),
              "ridgefill+corr+prot25": dict(dir="graded", ppl="ppl-a", ruler="ruler-a", speed="speed", control="control",
                                            policy="CED on (quality)"),
              "ridgefill": dict(dir="ridgefill", ppl="ppl-a", ruler="ruler-a", speed="speed", policy="CED on (plain)")},
        data=FN_DATA, dataset_manifest_sha256=frozen, deterministic=True,
        note="C5, the Flash-Next engine tier as run (protocol 1.1.6): exact (with its same-boot second perplexity pass), "
             "quality mode (RidgeFill + GDN terminal-state correction + class56 exact rows at 25%) and plain RidgeFill; "
             "perplexity on the 90 locked long-prompt documents <= 32K, RULER NIAH 8K-32K + niah_single_1 64K "
             "(instruct format), TTFT against the same boot's exact path")]


def registered_changes(cs, ledger):
    """[problem] for every cell whose FINAL- record exists in the ledger but pins other bytes (cell or configs), and for
    every FINAL- id in the ledger that no cell carries unless registration.SUPERSEDES names its successor (a registered
    cell is never dropped or renamed silently); an analysis-only id (registration.ANALYSIS) needs the cell it reads."""
    os.environ.setdefault("KVA_REPO", str(REPO))
    out = []
    for c in cs:
        if c["kind"] != "speed" and registration.record(c["registration"], ledger) is not None:
            out += [f"RegisteredCellChanged: {c['name']}: {p}" for p in registration.check_record(c, ledger)]
    carried = {c["registration"] for c in cs}
    for line in Path(ledger).read_text().splitlines():
        rid = json.loads(line).get("id", "") if json.loads(line).get("type") == "hypothesis" else ""
        if rid in registration.ANALYSIS:
            if registration.ANALYSIS[rid] not in carried:
                out.append(f"AnalysisCellMissing: {rid} is an analysis of {registration.ANALYSIS[rid]}, which no cell "
                           "carries")
            continue
        if rid in registration.ENGINE or rid in registration.CANCELLED:
            continue                                   # the engine tier: engine_cells(), checked in main()
        if rid.startswith("FINAL-") and rid not in carried and SUPERSEDES.get(rid) not in carried \
                and rid not in registration.CUT:
            out.append(f"RegisteredCellDropped: {rid} is in the ledger, no cell carries it and registration.SUPERSEDES "
                       "names no successor that one does")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ledger", default=str(REPO / "ledger" / "paper" / "ledger.jsonl"))
    args = ap.parse_args()
    out = FINALS / "cells"
    cs = cells()
    changed = registered_changes(cs, args.ledger)
    es = engine_cells()
    changed += [f"RegisteredCellChanged: {p}" for c in es for p in registration.check_engine_record(c, args.ledger)]
    changed += [f"EngineCellMissing: {rid} has no engine cell" for rid, name in registration.ENGINE.items()
                if name not in {c["name"] for c in es}]
    if changed:
        print("make_cells: a registered cell would change (a change after registration needs a new FINAL- id):\n  "
              + "\n  ".join(changed), file=sys.stderr)
        sys.exit(1)
    want = {f"{c['name']}.json": canonical(c) for c in cs}
    have = {p.name: p.read_bytes() for p in out.glob("*.json")}
    ewant = {f"{c['name']}.json": canonical(c) for c in es}
    ehave = {p.name: p.read_bytes() for p in ENGINE_DIR.glob("*.json")}
    if args.check:
        bad = sorted(n for n in set(want) | set(have) if want.get(n) != have.get(n))
        ebad = sorted(n for n in set(ewant) | set(ehave) if ewant.get(n) != ehave.get(n))
        print(f"make_cells: {len(want)} cells, " + (f"DIFFER: {', '.join(bad)}" if bad else "all identical"))
        print(f"make_cells: {len(ewant)} engine cell(s), " + (f"DIFFER: {', '.join(ebad)}" if ebad else "all identical"))
        sys.exit(1 if bad or ebad else 0)
    for n in set(have) - set(want):
        (out / n).unlink()
    for n, b in want.items():
        (out / n).write_bytes(b)
    ENGINE_DIR.mkdir(exist_ok=True)
    for n in set(ehave) - set(ewant):
        (ENGINE_DIR / n).unlink()
    for n, b in ewant.items():
        (ENGINE_DIR / n).write_bytes(b)
    print(f"make_cells: wrote {len(want)} cells to {out} and {len(ewant)} engine cell(s) to {ENGINE_DIR}")


if __name__ == "__main__":
    main()
