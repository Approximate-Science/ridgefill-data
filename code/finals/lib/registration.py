#!/usr/bin/env python3
"""FINAL- registrations of the finals cells: the labbook command per cell, and the pins a run is checked against.

A cell's FINAL- record is a labbook hypothesis (ledger/paper; written by Dylan from finals/register.sh's printout):
claim, prediction and kill condition from OUTLINE.md's claims table (C1-C5, R6) and tests, with the statistics rules
of PROTOCOL.md section 6 (v1.1), and, at the end of the claim, the PINS: the sha256 of the cell (its canonical bytes,
as finals/lib/atomic.write_result writes cell.json) and of every harness config finals/lib/kvabackend.py generates
from it (the bytes evaluate.py reads). labbook stores free text only, so the pins are text in a fixed form,
"pins: cell <name> sha256=<hex>; config <name> sha256=<hex>; ...", parsed back by pins_of(). run_final.sh refuses a
cell whose record is missing or whose pins differ (check_record); verify_finals.py checks the results against them.

Operationalisations marked "(declared here)" are this file's choices where OUTLINE.md names a test but no threshold;
Dylan reviews them before running the commands.
CLI: registration.py CELL.json...   prints one labbook command per cell (or a named refusal as a comment)
"""
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

FINALS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINALS / "lib"))
import kvabackend                                            # noqa: E402

PIN = re.compile(r"\b(cell|config) (\S+) sha256=([0-9a-f]{64})")
STATS = ("Statistics (PROTOCOL.md section 6, v1.1.3): paired per-sample deltas; BCa 95% intervals from 5,000 resamples of "
         "prompts (rows within task on RULER and LongBench), each sample's arms kept together; estimand {estimand}; one "
         "primary contrast per claim (protocol.json contrasts: (a) ridgefill - identity, (b) ridgefill+rows - "
         "identity+rows), its minimum detectable effect computed and printed with the result; Holm correction over "
         "every other contrast of the claim; one-sided non-inferiority tests at the declared margins (perplexity "
         "+1%, RULER -2 points; a degenerate interval never passes, a bounded score's bound is never looser than the "
         "exact binomial floor: finals/stats.py noninferiority); RULER per-length curves are descriptive (only "
         "pooled contrasts are claimable, on the actual eligible rows).")


def gen_cap(cell):
    """The registration's sentence for a cell's v1.1.2 RULER generation cap (caps.ruler_max_new), else ''."""
    caps = cell.get("caps", {}).get("ruler_max_new")
    if not caps:
        return ""
    named = ", ".join(f"{t} {n}" for t, n in sorted(caps.items()))
    return (f" Generation budget (protocol 1.1.2): {named} tokens; every other task at RULER's tokens_to_generate "
            f"(truncating FINAL-QWEN3-8B-RULER's 3,556 NIAH arm-rows to these budgets changed 1 row and no arm's score).")


ESTIMAND = {"ruler": "sample-weighted accuracy (filled-region needles primary, all needles reported)",
            "longbench": "sample-weighted official task score", "nll": "document-weighted NLL (novel tokens primary, "
            "key tokens at threshold 2.0 nats and all tokens reported, per length bin)",
            "decode": "document-weighted mean per-step KL(exact || arm) and forced top-1 agreement",
            "secondq": "sample-weighted turn-2 accuracy (turn 1 reported)"}
ESTIMAND["nll-causal"] = ESTIMAND["nll"]
ESTIMAND["ruler-deep"] = ESTIMAND["ruler"]
ESTIMAND["nll-fresh"] = ESTIMAND["nll"]
SHORT = {"llama-3.1-8b-instruct": "LLAMA", "qwen3.5-9b": "QWEN35", "qwen3-8b": "QWEN3-8B"}
# A registered FINAL- id whose cell changed BEFORE any run is superseded by a new id (never edited in place): the old
# record stays in the ledger, a 'superseded-<old id>' record says why, and a cell under the old id is refused
# (check_record, verify_finals). make_cells.py gives the regenerated cell the new id.
SUPERSEDES = {"FINAL-QWEN3-8B-NLL": "FINAL-QWEN3-8B-NLL-2",
              "FINAL-LLAMA-RULER": "FINAL-LLAMA-RULER-2", "FINAL-QWEN35-RULER": "FINAL-QWEN35-RULER-2",
              "FINAL-LLAMA-LONGBENCH": "FINAL-LLAMA-LONGBENCH-2"}
_TRIM = ("trimmed before any run (Dylan, 2026-10-07, after the scope audit reviews/2026-10-07-scope-audit-fable.md): "
         "the NIAH tasks only")
SUPERSEDE_REASON = {"FINAL-QWEN3-8B-NLL": "arms added before any run (pop-native; identity+rows and ridgefill+rows "
                    "with their random-rows and tail-match controls; nofill+tailbudget: the coordinator's decisions of "
                    "2026-10-04 night); no result existed",
                    "FINAL-LLAMA-RULER": _TRIM + " (the other four tasks, their 1/2-split run, the question-first "
                    "variant and SwiftKV RULER cut; 64K kept on the NIAH rows; the 1/2-split NIAH run kept and read as a "
                    "labelled follow-up); protocol 1.1.2's generation cap; the question-blind rows pair (qblind10) "
                    "added. The predictions and kills are the registered ones, minus clauses about the cut runs; no "
                    "result existed",
                    "FINAL-QWEN35-RULER": _TRIM + " (the other four tasks, the splices and the question-first "
                    "variant cut; 64K kept on the NIAH rows, so the pooled population reaches the non-inferiority "
                    "floor's 149 rows); protocol 1.1.2's generation cap; the question-blind rows pair "
                    "(qblind10) added. The predictions and kills are the registered ones, minus clauses about the cut "
                    "runs; no result existed",
                    "FINAL-LLAMA-LONGBENCH": "cut before any run on 2026-10-07 (record cut-FINAL-LLAMA-LONGBENCH) and "
                    "re-added by Dylan on 2026-10-08 with the primary arms only (no SwiftKV run): the one realistic "
                    "long-document task the 2026-10-07 reviews asked for; the prediction and kill are the registered ones; "
                    "no result existed"}
# A registered FINAL- id cut before any run (no successor): the record stays in the ledger, a 'cut-<id>' record says
# why, make_cells.py writes no cell for it and the tables print the cell's columns as not run.
CUT = {f"FINAL-{m}-LONGBENCH": "cut before any run (Dylan, 2026-10-07): T1.2 is secondary with kill 'None', about "
       "$16 of cloud time for one column; the paper reports LongBench as not run" for m in ("QWEN35", "QWEN3-8B")}
# FINAL-LLAMA-LONGBENCH was cut on 2026-10-07 too and re-added on 2026-10-08 as FINAL-LLAMA-LONGBENCH-2 (SUPERSEDES)
# An analysis-only FINAL- id: a registered test on an existing cell's data, with no cell of its own (no new run). The
# value is the FINAL- id of the cell it reads; make_cells.py writes no cell for it and refuses it unless that cell is
# carried; finals/tables.py reports it where it reads the data. FINAL-LLAMA-RULER-ROWS: Llama's rows-versus-exact
# non-inferiority (claim 4, OUTLINE.md revision 4), declared 2026-10-08 after a readout of the data: post hoc.
ANALYSIS = {"FINAL-LLAMA-RULER-ROWS": "FINAL-LLAMA-RULER-2"}
# The engine tier (Flash-Next, claim C5): run by the C5 kit on a local machine, not by the finals harness. Its FINAL-
# record pins the kit, the projector bundle and protocol.json, not a cell, so the cell is generated from the record's
# declared values (finals/make_cells.py engine_cells(), written to finals/engine/) and checked against them here
# (check_engine_record); finals/convert_c5.py turns the kit's results into the cell's sample records. FINAL id -> cell.
ENGINE = {"FINAL-FLASHNEXT-C5": "qwen3.8-flash-next-c5"}
# A registered FINAL- id that never ran and has no cell (no successor, no cut record): kept in the ledger, refused as
# a cell, and never read by the tables. FINAL-FLASHNEXT-C5-CHAT: a held-out labelled-chat row for C5, registered
# 2026-10-10 and cancelled before any run (Dylan).
CANCELLED = {"FINAL-FLASHNEXT-C5-CHAT": "registered 2026-10-10 and cancelled before any run (Dylan): no cell, no data"}
ENGINE_DECLARED = re.compile(r"Declared: declared_share = (\d+), margins \{([^}]*)\}, split_layer = (\d+), "
                             r"tail = (\d+)\.")


def engine_declared(text):
    """The values an engine-tier FINAL- record declares ('Declared: declared_share = .., margins {..}, split_layer = ..,
    tail = ..'), or None."""
    m = ENGINE_DECLARED.search(text or "")
    if not m:
        return None
    margins = {}
    for part in m.group(2).split(","):
        k, _, v = part.partition(":")
        margins[k.strip()] = float(v) if "." in v else int(v)
    return dict(declared_share=int(m.group(1)), margins=margins, split_layer=int(m.group(3)), tail=int(m.group(4)))


def labelled(label, ledger):
    """The ledger's record with this label (the last one), or None."""
    hit = None
    for line in Path(ledger).read_text().splitlines():
        rec = json.loads(line)
        if rec.get("label") == label:
            hit = rec
    return hit


def check_engine_record(cell, ledger):
    """[] if an engine-tier cell carries exactly what its FINAL- record declares (share, margins, split layer, tail), every
    amendment it names is in the ledger and says what the cell takes from it, and its protocol version was adopted by a
    protocol-change record; else the problems, by name."""
    rid = cell.get("registration")
    if ENGINE.get(rid) != cell.get("name"):
        return [f"EngineCellUnknown: {cell.get('name')} / {rid} is not an engine-tier cell (registration.ENGINE)"]
    rec = record(rid, ledger)
    if rec is None:
        return [f"registration {rid} is not a hypothesis id in {ledger}"]
    declared = engine_declared(rec.get("claim"))
    if declared is None:
        return [f"{rid}: the record declares no 'Declared: declared_share = .., margins {{..}}, split_layer = .., "
                "tail = ..' values"]
    problems = [f"EngineCellChanged: {cell['name']} {k} = {cell.get(k)!r}, {rid} declares {v!r}"
                for k, v in declared.items() if cell.get(k) != v]
    for label, must in cell.get("amendments", {}).items():
        a = labelled(label, ledger)
        if a is None:
            problems.append(f"{cell['name']}: amendment {label} is not in {ledger}")
        elif must not in (a.get("note") or ""):
            problems.append(f"{cell['name']}: amendment {label} does not say {must!r}")
    v = str(cell.get("protocol"))
    change = labelled(f"protocol-change-{v}", ledger) or {}
    if not re.search(rf"protocol-change \S+ -> {re.escape(v)}(?![\d.])", change.get("note") or ""):
        problems.append(f"{cell['name']}: protocol {v} has no 'protocol-change-{v}' record adopting it")
    return problems


def registration_id(model, kind):
    return f"FINAL-{SHORT[model]}-{kind.upper()}"


def current_id(model, kind):
    rid = registration_id(model, kind)
    return SUPERSEDES.get(rid, rid)


def superseded_label(rid):
    return f"superseded-{rid}"


def superseded(registration, ledger):
    """The ledger's 'superseded-<id>' record for this registration, or None."""
    for line in Path(ledger).read_text().splitlines():
        rec = json.loads(line)
        if rec.get("type") == "record" and rec.get("label") == superseded_label(registration):
            return rec
    return None


def canonical(obj):
    return json.dumps(obj, indent=1, sort_keys=True).encode() + b"\n"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pins(cell):
    """{'cell': {name: sha}, 'config': {name: sha}} for a cell (named refusal for a run with no harness path)."""
    return {"cell": {cell["name"]: sha(canonical(cell))},
            "config": {cfg["name"]: sha(kvabackend.config_bytes(cfg)) for cfg, _ in kvabackend.harness_configs(cell)}}


def pins_text(p):
    return "pins: " + "; ".join(f"{k} {n} sha256={h}" for k in ("cell", "config") for n, h in p[k].items())


def pins_of(text):
    got = {"cell": {}, "config": {}}
    for k, name, h in PIN.findall(text or ""):
        got[k][name] = h
    return got


def _layers(cell):
    PR, p = kvabackend._protocol()
    m = p["models"][cell["model"]["name"]]
    return {r["split"]: kvabackend.layer_of(m, r["split"]) for r in cell["runs"] if r.get("model") != "swiftkv"}, m


def texts(cell):
    """(claim, predict, kill) for a cell, by kind and model (OUTLINE.md claims and tests)."""
    kind, model = cell["kind"], cell["model"]["name"]
    L, m = _layers(cell)
    if kind == "ruler-deep":
        return ruler_deep(cell, L, m)
    if kind == "nll-fresh":
        return nll_fresh(cell, L, m)
    s, tb = L["3/8"], cell.get("tailbudget", {}).get("b")
    curve = "/".join(f"S{L[f]}" for f in ("1/4", "3/8", "1/2", "5/8") if f in L)
    llama, hybrid = model.startswith("llama"), m["kind"] == "hybrid"
    mg = cell["margins"]
    stats = STATS.format(estimand=ESTIMAND.get(kind, "")).replace("+1%", f"+{mg['ppl_ratio_upper_pct']:g}%").replace(
        "-2 points", f"{mg['ruler_points_lower']:g} points")
    cap = (" Runs at <= 32K only (no YaRN): the files over the cap are skipped by name (the cell's 'skipped')."
           if cell.get("max_len") == 32768 else "")
    if kind == "ruler" and cell["code_tag"] != "protocol-v1.1":
        return ruler_trimmed(cell, L, s, tb, llama, mg, stats)
    if kind == "nll-causal":
        return nll_causal(cell, s, llama, hybrid, model, stats, cap)
    if kind == "ruler":
        common = (f"RULER (T1.1): 8 tasks x 25 rows per task and length, official scorer, primary split 3/8 = S{s}, tail "
                  f"4096: exact, nofill, identity (POP's fill at our split and tail), pop-native, ridgefill, identity+rows "
                  f"and ridgefill+rows (qattn10) at 8K-32K" + ("" if cap else "; at 64K exact, identity, ridgefill and "
                  "both rows arms") + f". On the NIAH subset (4 NIAH tasks, 8K-32K): the T1.5 controls (random rows at "
                  f"the rows arm's count, that count spent on a longer tail, the fill's modelled budget spent on a longer "
                  f"tail: nofill+tailbudget, b = {tb}), the T2.1 splices ridgefill+kexact and ridgefill+vexact (C4: exact "
                  f"keys alone restore retrieval; exact values are its control) and T1.4 (question before the document, "
                  f"the keys only before it: identity+rows and ridgefill+rows, whose selector then reads the document "
                  f"only; datasets/derived/ruler-qfirst.json). Filled-region needles: NIAH rows whose every scored "
                  f"needle ends at or before N - {cell['tail']} (kva.locked.needle_spans; the other tasks have no needle)."
                  + (f" Appendix: identity, ridgefill and both rows arms at the "
                  f"1/2 split S{L['1/2']} (8K-32K). R6: SwiftKV as released and its maps + exact tail at S16 (8K-32K)."
                  if llama else "") + cap + gen_cap(cell))
        if llama:
            return ("C2 (OUTLINE.md): on a dense model the fitted map is what makes the fill usable: RidgeFill beats the "
                    "identity fill and the no-fill control on RULER needles in the filled region. " + common,
                    f"Primary (a): ridgefill - identity on the pooled filled-region accuracy at S{s} > 0 (BCa lower bound "
                    "> 0). Secondary (Holm): ridgefill - nofill > 0; ridgefill > nofill+tailbudget; rows arms > random "
                    "rows and > their tail match (NIAH subset); (b) ridgefill+rows - identity+rows reported; SwiftKV "
                    "reported (R6, not a test). C4 (NIAH subset): ridgefill+kexact closes ridgefill's filled-region gap "
                    "to exact (BCa interval of the closed share reaches 100% and excludes 0), ridgefill+vexact does "
                    "not. " + stats,
                    "C2 fails if (a)'s BCa lower bound is <= 0 (C2 needs this AND the NLL primary of FINAL-LLAMA-NLL), or "
                    "if identity at a later split matches RidgeFill at equal measured TTFT (T3.1, NLL curve). Then stop, "
                    "audit T0.1 again, and Dylan decides (PLAN.md contingencies).")
        if hybrid:
            return ("C3 (OUTLINE.md): on a QK-normed hybrid the map does not help and exact rows do: fitted and identity "
                    "fills tie, and question-attended exact rows (10%) restore RULER retrieval. " + common,
                    f"(a) ridgefill - identity at S{s}: the BCa 95% interval of the pooled filled-region accuracy contains "
                    "0 (the wash, reported with its MDE; not an equivalence claim). (b) ridgefill+rows - identity+rows "
                    "reported. Rows restore retrieval (declared here): identity+rows and ridgefill+rows minus exact >= "
                    f"{mg['ruler_points_lower']:g} points by TOST on the pooled filled-region accuracy, and rows > random "
                    "rows at the same count (NIAH subset). The question-first rows are reported as scope. C4 (NIAH "
                    "subset): ridgefill+kexact closes ridgefill's filled-region gap to exact (BCa interval of the closed "
                    "share reaches 100% and excludes 0). " + stats,
                    "C3 fails if the rows arms do not restore retrieval (the TOST fails, or rows <= random rows): the hybrid "
                    "story shrinks to identity fill + tail only (PLAN.md contingencies). A question-first collapse is a "
                    "scope statement in the paper, not a kill, provided it is stated.")
        return (f"C4's QK-norm test with T1 (OUTLINE.md T2.4): Qwen3-8B is dense with QK-norm and no GDN; if QK-norm, not "
                f"the hybrid design, is why the map does not help on Qwen3.5, Qwen3-8B behaves like Qwen3.5 here: fitted "
                f"and identity fills tie at S{s} (3/8 of 36, boundary below) and rows restore retrieval. " + common,
                f"ridgefill - identity at S{s}: the BCa 95% interval of the pooled filled-region accuracy contains 0; "
                f"identity+rows and ridgefill+rows minus exact >= {mg['ruler_points_lower']:g} points by TOST (declared "
                "here); (a) and (b) are the primaries, Holm on the rest; ridgefill+kexact closes ridgefill's "
                "filled-region gap to exact on the NIAH subset (C4, as on the reference models). " + stats,
                f"Qwen3-8B behaves like Llama: ridgefill - identity BCa lower bound > 0 at S{s} (read with the splice "
                f"ordering of {current_id(model, 'nll')}): QK-norm alone does not explain the difference, and the paper "
                "says so.")
    if kind == "longbench":
        return (f"T1.2 LongBench (secondary, downstream sanity; no claim rests on it): 8 frozen tasks x 50 rows, official "
                f"per-task metrics, middle truncation to {cell['caps']['longbench_max_len']} tokens; the primary arms at "
                f"S{s}" + ("; SwiftKV as released at S16 (R6)." if any(r.get("model") == "swiftkv" for r in cell["runs"])
                           else "."),
                "ridgefill - identity has the sign of the same model's RULER primary contrast (Holm-adjusted, reported "
                "with its MDE); nothing from this cell is primary. " + stats,
                "None (secondary): a sign opposite to the RULER primary is reported next to it, never hidden.")
    if kind == "nll":
        common = (f"T1.3 long-document NLL on the locked set (126 documents, 8K-128K), arm minus exact on the "
                  f"continuation's novel tokens (no 8-token-id window seen earlier, kva.data.novel_mask) and key tokens "
                  f"(LongPPL-style, threshold {cell['keytoken']['threshold']:g} nats, kva/keytok.py, tail-only reference "
                  f"T = {cell['keytoken']['tail']}), per length bin; the split curve {curve} for nofill, identity, "
                  f"ridgefill (C1)" + (f"; pop-native (POP's own split S{m['pop_split']}, the continuation as its tail: "
                  f"the split-curve table's POP native row)" if any("pop-native" in r["arms"] for r in cell["runs"])
                  else "") + (f"; at S{s} the rows arms identity+rows and ridgefill+rows (qattn10), their random-rows and "
                  f"tail-match controls and nofill+tailbudget (b = {tb}, the fill's modelled cost over these documents' "
                  f"lengths): T1.5 at equal cost on NLL" if tb is not None else "")
                  + f"; at S{s} the splices ridgefill+kexact, ridgefill+vexact, kvexact and the rescales "
                  f"ridgefill+ktail, ridgefill+vtail (T2.1, T2.2; vtail is appendix)."
                  + (" R6: SwiftKV as released and its maps + exact tail at S16 on the SwiftKV checkpoint (its own "
                     "config), against Llama's exact; the converter requires the checkpoint's own exact to equal "
                     "Llama's on every document (SK1: identical base weights)." if llama else "")
                  + (" Documents over 32K are skipped by name." if cap else ""))
        if llama:
            return ("C2 (NLL half), C1 and C4 on the dense model. " + common,
                    f"Primary (a): ridgefill - identity on the document-weighted novel-token NLL at S{s} < 0 (BCa upper "
                    "bound < 0). C1: ridgefill - identity < 0 at every curve split and ridgefill's NLL rises as the split "
                    "moves earlier, no reversal outside the CIs. C4 (interventional): ridgefill+kexact removes more of "
                    "ridgefill's NLL excess over exact than ridgefill+vexact (BCa on the difference of removed shares "
                    "> 0); ridgefill+ktail removes >= 50% of that excess (declared here; BCa lower bound of the removed "
                    "share > 0). Equal cost (Holm): ridgefill - nofill+tailbudget < 0 on the novel-token NLL. R6 is "
                    "reported, not tested (SwiftKV's distilled fill is expected to beat ridgefill at S16). " + stats,
                    "C2 fails if (a)'s BCa upper bound is >= 0 (both halves are required, with FINAL-LLAMA-RULER). C1 "
                    "fails on a reversal outside the CI or a split where ridgefill does not beat identity. C4 fails on "
                    "Llama if the splice ordering flips (vexact removes more than kexact) or ktail removes < 50%.")
        if hybrid:
            return ("C3 (NLL) and C4 on the hybrid (kvexact on a hybrid = attention oracle with the GDN layers projected: "
                    "the GDN-only error). " + common,
                    f"ridgefill - identity at S{s}: the BCa interval of the document-weighted novel-token NLL contains 0 "
                    "(reported with its MDE). C4: ridgefill+vexact removes more of ridgefill's NLL excess than "
                    "ridgefill+kexact (values carry the loss on Qwen3.5); ktail is a demonstration (QK-norm undoes a "
                    "pre-norm key scale), not a test. " + stats,
                    "C4 fails if the splice ordering flips on Qwen3.5 (kexact removes more of the excess than vexact).")
        return ("C4's QK-norm test (OUTLINE.md T2.4) on Qwen3-8B (dense, QK-norm, no GDN). " + common,
                f"Qwen3-8B behaves like Qwen3.5: ridgefill+vexact removes more of ridgefill's NLL excess than "
                f"ridgefill+kexact; ridgefill+ktail removes < 50%; ridgefill - identity at S{s}: the interval contains 0. "
                + stats,
                "Qwen3-8B behaves like Llama (kexact removes more than vexact, or ktail removes >= 50%, or ridgefill - "
                "identity BCa upper bound < 0): C4's QK-norm clause fails; the splice result on the two reference "
                "models stands as measured.")
    if kind == "decode":
        claim = (f"T1.7 decode-side effect: 512 greedy tokens after each of 20 locked long-document prompts (picked by "
                 f"kva.derived.pick_decode_docs: manifest order under seed 20261004, every length bin <= 32K), exact vs "
                 f"ridgefill, identity and nofill at S{s}: forced per-step KL(exact || arm) and top-1 agreement, free "
                 f"first divergence and identical share; the filled cache is attended for the whole generation.")
        if llama:
            return ("Supports C2. " + claim,
                    "ridgefill has a lower mean per-step KL and a higher forced top-1 agreement than identity and nofill "
                    "(BCa over the 20 documents; Holm across the readouts). " + stats,
                    "ridgefill's KL is not below identity's (the interval of the difference contains 0 or favours "
                    "identity): the decode side does not support C2; the paper limits C2 to prefill quality.")
        return (("C3's wash on the decode side. " if hybrid else "C4's QK-norm test on the decode side. ") + claim,
                "ridgefill and identity tie (the interval of the KL difference contains 0); both agree with exact more "
                "than nofill. " + stats,
                "None for the claim (supporting evidence): a ridgefill advantage outside the CI is reported as "
                + ("contradicting the wash." if hybrid else f"Qwen3-8B behaving like Llama (read with "
                                                             f"{current_id(model, 'nll')})."))
    if kind == "secondq":
        return (f"T1.8 second question on a cached prefix (the multi-turn scope of C3; the intro keeps the agentic "
                f"framing only if it holds): the locked two-key RULER set (niah_multikey_1 rows, "
                f"datasets/derived/ruler-twokey.json pinned in the cell); the prefix is filled once with rows chosen for "
                f"question 1 (qattn10), question 2 asks another key of the row against the same cache; arms exact, "
                f"ridgefill, identity+rows, ridgefill+rows at S{s}, tail 4096.{cap}",
                "Turn 2: ridgefill+rows - identity+rows > 0 on the pooled turn-2 accuracy " +
                ("(the dense model)" if llama else "(on this QK-normed model the interval may contain 0, as C3's wash)") +
                "; rows chosen for question 1 do not help question 2: ridgefill+rows - ridgefill on turn 2 has an "
                "interval containing 0. " + stats,
                "Turn 2 collapses (declared here: ridgefill+rows' turn-2 accuracy is below half of exact's turn-2 "
                "accuracy): the intro drops the agentic framing (OUTLINE.md T1.8).")
    raise kvabackend.HarnessGap(f"no registration text for kind {kind!r}")


def ruler_trimmed(cell, L, s, tb, llama, mg, stats):
    """The trimmed RULER cells (FINAL-LLAMA-RULER-2, FINAL-QWEN35-RULER-2): the registered predictions and kills of
    the cells they supersede, minus clauses about the cut runs, plus the follow-ups declared on 2026-10-07."""
    ids = sorted(r["id"] for r in cell["runs"])
    want = sorted(["ruler-niah", "ruler-niah-splices", "ruler-niah-1-2", "ruler-64k"] if llama else ["ruler-niah", "ruler-64k"])
    if ids != want:
        raise kvabackend.HarnessGap(f"no registration text for trimmed RULER runs {ids} (want {want})")
    common = (f"RULER (T1.1, trimmed before any run): the 4 NIAH tasks x 25 rows per task and length, official scorer, "
              f"primary split 3/8 = S{s}, tail 4096. At 8K-32K: exact, nofill, identity (POP's fill at our split and "
              f"tail), pop-native, ridgefill, identity+rows and ridgefill+rows (qattn10), the T1.5 controls (random rows "
              f"at the rows arm's count, that count spent on a longer tail, nofill+tailbudget, b = {tb}) and the "
              f"question-blind rows pair identity+qblind and ridgefill+qblind (qblind10: the selector reads the 64 rows "
              f"before P and never the question; the question-first variant of T1.4 is a question-position test, not "
              f"question-blind); at 64K on the NIAH rows exact, identity, ridgefill and both rows arms (pooled: the "
              f"non-inferiority floor needs >= 149 eligible rows)" + (f"; the T2.1 splices ridgefill+kexact and "
              f"ridgefill+vexact; at the 1/2 split S{L['1/2']} (8K-32K) identity, ridgefill and both rows arms" if llama
              else "") + f". Filled-region needles: NIAH rows whose every scored needle "
              f"ends at or before N - {cell['tail']} (kva.locked.needle_spans)." + gen_cap(cell) + " Not run (cut "
              "2026-10-07): the other four RULER tasks" + (", their 1/2-split run, the question-first variant and SwiftKV"
              " RULER." if llama else ", the splices and the question-first variant."))
    blind = (" The question-blind rows pair is reported next to the question-attended rows (scope: a cache built "
             "before the question exists).")
    if llama:
        return ("C2 (OUTLINE.md): on a dense model the fitted map is what makes the fill usable: RidgeFill beats the "
                "identity fill and the no-fill control on RULER needles in the filled region. " + common,
                f"Primary (a): ridgefill - identity on the pooled filled-region accuracy at S{s} > 0 (BCa lower bound "
                "> 0). Secondary (Holm): ridgefill - nofill > 0; ridgefill > nofill+tailbudget; rows arms > random "
                "rows and > their tail match; (b) ridgefill+rows - identity+rows reported. C4: ridgefill+kexact closes "
                "ridgefill's filled-region gap to exact (BCa interval of the closed share reaches 100% and excludes 0), "
                "ridgefill+vexact does not." + blind + f" Follow-up, declared 2026-10-07 after FINAL-LLAMA-NLL showed "
                f"the S{s} fill failing at >= 64K and winning at S{L['1/2']} (labelled post hoc in the paper): "
                f"ridgefill - identity > 0 at S{L['1/2']} (8K-32K); the 64K contrast at S{s} is reported on its own. "
                + stats,
                "C2 fails if (a)'s BCa lower bound is <= 0 (C2 needs this AND the NLL primary of FINAL-LLAMA-NLL, which "
                "failed as registered on 2026-10-07: C2 as registered has failed; this cell decides its retrieval half "
                "and the follow-up).")
    return ("C3 (OUTLINE.md): on a QK-normed hybrid the map does not help and exact rows do: fitted and identity fills "
            "tie, and question-attended exact rows (10%) restore RULER retrieval. " + common,
            f"(a) ridgefill - identity at S{s}: the BCa 95% interval of the pooled filled-region accuracy contains 0 "
            "(the wash, reported with its MDE; not an equivalence claim). (b) ridgefill+rows - identity+rows reported. "
            "Rows restore retrieval (declared): identity+rows and ridgefill+rows minus exact >= "
            f"{mg['ruler_points_lower']:g} points by a one-sided non-inferiority test on the pooled filled-region "
            "accuracy (finals/stats.py noninferiority with the exact-binomial floor), and rows > random rows at the "
            "same count." + blind + " " + stats,
            "C3 fails if the rows arms do not restore retrieval (the non-inferiority test fails, or rows <= random "
            "rows): the hybrid story shrinks to identity fill + tail only (PLAN.md contingencies).")


def ruler_deep(cell, L, m):
    """The matched-depth RULER follow-up (Dylan, 2026-10-08): RidgeFill at 5/8 against POP as published."""
    s, n, pop = L["5/8"], m["layers"], m["pop_split"]
    mg = cell["margins"]
    stats = STATS.format(estimand=ESTIMAND["ruler-deep"]).replace("+1%", f"+{mg['ppl_ratio_upper_pct']:g}%").replace(
        "-2 points", f"{mg['ruler_points_lower']:g} points")
    claim = (f"Matched-depth retrieval against POP as published (a follow-up declared 2026-10-08 after "
             f"{current_id(cell['model']['name'], 'ruler')}, labelled post hoc in the paper): on the 4 NIAH tasks at "
             f"8K-32K (25 rows per task and length, official scorer, filled-region needles primary: every scored needle "
             f"ends at or before N - {cell['tail']}), RidgeFill and the identity fill at the 5/8 split S{s} (skipping "
             f"{n - s} of {n} layers for the filled rows, tail {cell['tail']}) against pop-native, POP at its own split "
             f"S{pop} (skipping {n - pop}): RidgeFill skips MORE layers than POP." + gen_cap(cell))
    predict = (f"ridgefill at S{s} minus pop-native on the pooled filled-region accuracy: one-sided non-inferiority at a "
               f"-3 point margin (declared here: with about 138 eligible rows the exact-binomial floor of finals/stats.py "
               f"noninferiority is about -2.15 points, so the protocol's -2 is unattainable by design at this n). "
               f"ridgefill - identity at S{s} > 0 (BCa lower bound > 0). " + stats)
    kill = ("the non-inferiority bound of ridgefill - pop-native is below -3 points: at deeper skipping RidgeFill does "
            "not match POP's retrieval, and the paper says so (the matched-depth claim stays perplexity-only).")
    return claim, predict, kill


def nll_fresh(cell, L, m):
    """The fresh-document confirmation (Dylan, 2026-10-08) of the post-hoc Llama reads at S16/S20 and vs POP."""
    s16, s20, n, pop = L["1/2"], L["5/8"], m["layers"], m["pop_split"]
    mg = cell["margins"]
    stats = STATS.format(estimand=ESTIMAND["nll-fresh"]).replace("+1%", f"+{mg['ppl_ratio_upper_pct']:g}%").replace(
        "-2 points", f"{mg['ruler_points_lower']:g} points")
    claim = (f"Confirmation on fresh data (declared 2026-10-08 before any run on this set) of follow-ups read after "
             f"FINAL-LLAMA-NLL: the frozen set {cell['dataset']} (30 long documents, 6 domains x 8K/16K/32K/64K/128K, "
             f"selected under seed 20261008 from the locked set's sources with no exact or 50-token overlap with the "
             f"locked set or the calibration corpus; manifest {cell['dataset_manifest_sha256'][:16]}), novel-token and "
             f"key-token NLL as {current_id(cell['model']['name'], 'nll')}. Arms: exact, identity and ridgefill at "
             f"S{s16} and S{s20} (skipping {n - s16} and {n - s20} of {n} layers), pop-native (POP at its own split "
             f"S{pop}, skipping {n - pop}).")
    predict = (f"(a) ridgefill - identity < 0 at S{s16} and at S{s20} on the document-weighted novel-token NLL (BCa upper "
               f"bound < 0 for each; Holm over the two). (b) ridgefill at S{s20} - pop-native < 0 (BCa upper bound < 0): "
               f"RidgeFill skipping more layers than POP loses less NLL. " + stats)
    kill = (f"(a) fails (a BCa upper bound >= 0 at S{s16} or S{s20}): the post-hoc S16/S20 reads are not confirmed and the "
            f"paper reports them as exploratory only. (b) fails: the claim against POP as published is not confirmed on "
            f"fresh data and is withdrawn.")
    return claim, predict, kill


def nll_causal(cell, s, llama, hybrid, model, stats, cap):
    """The 1.1.3 correction cells: the NLL arms whose 1.1.1 evaluation read the scored continuation, re-run."""
    orig = current_id(model, "nll")
    claim = (f"Correction cell (protocol 1.1.3) for {orig}: re-runs at S{s} the adaptive arms whose 1.1.1 evaluation "
             f"read the scored continuation (identity+rows and ridgefill+rows: the qattn window ended inside it; "
             f"ridgefill+ktail and ridgefill+vtail: the tail scale averaged over it), now with the window ending before "
             f"the continuation and the tail scale excluding it (scripts/evaluate.py, kva.kva.exclude_scored; found by "
             f"the 2026-10-07 reviews), on the same locked long-document set{' (<= 32K)' if cap else ''}, novel-token "
             f"and key-token NLL as {orig}. exact, identity, ridgefill and the count-matched random-rows and tail-match "
             f"controls run alongside, so every contrast pairs arms of one host. The 1.1.1 numbers of the four arms are "
             f"superseded: kept in the ledger and reported only as the size of the leak.")
    control = (f" Positive control: exact, identity and ridgefill (no adaptive decision) reproduce {orig}'s "
               "document NLLs within 1e-3 nats.")
    if llama:
        return (claim,
                "As registered in FINAL-LLAMA-NLL, re-tested: ridgefill+ktail removes >= 50% of ridgefill's NLL excess "
                "over exact (C4; BCa lower bound of the removed share > 0); rows arms beat their random-rows and "
                "tail-match controls (Holm); ridgefill+rows - identity+rows reported." + control + " " + stats,
                "ridgefill+ktail removes < 50% of the excess: C4's ktail clause fails on Llama (as registered). Rows "
                "arms not beating random rows: the rows advantage on NLL was the leak, and the paper says so.")
    if hybrid:
        return (claim,
                "ktail is a demonstration on Qwen3.5 (QK-norm undoes a pre-norm key scale), not a test; rows arms vs "
                "their random-rows and tail-match controls and ridgefill+rows - identity+rows reported (Holm)."
                + control + " " + stats,
                "None for the claims (rows restoring retrieval is RULER's test, FINAL-QWEN35-RULER-2): rows not beating "
                "random rows on NLL is reported.")
    return (claim,
            "As registered in FINAL-QWEN3-8B-NLL-2, re-tested: ridgefill+ktail removes < 50% of ridgefill's NLL excess "
            "(Qwen3-8B behaves like Qwen3.5 on the key scale); rows arms vs controls reported (Holm)." + control + " "
            + stats,
            "ridgefill+ktail removes >= 50%: Qwen3-8B behaves like Llama on the key scale (C4's QK-norm clause, read "
            "with FINAL-QWEN3-8B-NLL-2 and FINAL-QWEN3-8B-RULER).")


def command(cell):
    """The labbook command that registers a cell (a list of argv strings)."""
    pinned = pins_text(pins(cell))                 # first: a run with no harness path refuses by name here
    unpinned = [f["file"] for f in cell.get("projector", {}).get("files", []) if "sha256" not in f]
    if unpinned:
        raise kvabackend.HarnessGap(f"UnpinnedProjector: {', '.join(unpinned)} have no declared sha256 yet (the clean "
                                    "refit first: jobs/paper-refit-clean.jobs, then make_cells.py reads the fit reports)")
    claim, predict, kill = texts(cell)
    head = (f"[{cell['registration']}] cell {cell['name']} ({cell['kind']}, {cell['model']['id']} at "
            f"{cell['model']['commit'][:12]}), code tag {cell['code_tag']}, locked manifest "
            f"{cell['dataset_manifest_sha256'][:16]}, deterministic. ")
    old = next((o for o, n in SUPERSEDES.items() if n == cell["registration"]), None)
    if old:
        head += f"Supersedes {old}: {SUPERSEDE_REASON[old]}. "
    return ["labbook", "hypothesis", "--dir", "ledger/paper", cell["registration"],
            "--claim", head + claim + " | " + pinned, "--predict", predict, "--kill", kill]


def record(registration, ledger):
    for line in Path(ledger).read_text().splitlines():
        rec = json.loads(line)
        if rec.get("type") == "hypothesis" and rec.get("id") == registration:
            return rec
    return None


def supersede_command(old, cell, ledger=None):
    """The labbook command (run from the repo root) that records why `old` is superseded by the cell's id; it pins the
    regenerated cell file (sources resolve from ledger/paper, as tables.py reads them)."""
    prev = record(old, ledger) if ledger and Path(ledger).is_file() else None
    was = pins_of(prev.get("claim")) if prev else {"cell": {}}
    pin = "".join(f" {old} pinned cell {n} sha256={h}." for n, h in was["cell"].items())
    note = (f"{old} is superseded by {cell['registration']}: {SUPERSEDE_REASON[old]}.{pin} "
            f"{cell['registration']} pins the regenerated cell (this record's source). A cell registered {old} never "
            "runs: finals/lib/registration.py check_record and finals/verify_finals.py refuse it by this record.")
    argv = ["labbook", "record", "--dir", ".", superseded_label(old), "--values", "0",
            "--source", f"../../finals/cells/{cell['name']}.json", "--note", note]
    return "(cd ledger/paper && " + " ".join(shlex.quote(a) for a in argv) + ")"


def check_record(cell, ledger):
    """[] if the ledger's FINAL- record pins exactly this cell and its configs and its id is not superseded, else the
    problems, by name."""
    want = pins(cell)                              # first: a run with no harness path refuses by name here
    if superseded(cell["registration"], ledger):
        return [f"Superseded: {cell['registration']} is superseded ({superseded_label(cell['registration'])} in "
                f"{ledger}): run its successor {SUPERSEDES.get(cell['registration'], '?')}"]
    rec = record(cell["registration"], ledger)
    if rec is None:
        return [f"registration {cell['registration']} is not a hypothesis id in {ledger}: register the cell first "
                f"(finals/register.sh prints the command)"]
    got = pins_of(rec.get("claim"))
    problems = [f"ConfigNotRegistered: {k} {n} sha256 {h[:16]}... is not what {cell['registration']} pins "
                f"({got[k].get(n, 'nothing')[:16]})" for k in ("cell", "config") for n, h in want[k].items()
                if got[k].get(n) != h]
    problems += [f"ConfigNotRegistered: {cell['registration']} pins {k} {n}, which this cell does not generate"
                 for k in ("cell", "config") for n in got[k] if n not in want[k]]
    return problems


def main(paths):
    for p in paths:
        cell = json.loads(Path(p).read_text())
        try:
            argv = command(cell)
        except kvabackend.HarnessGap as e:
            print(f"# {cell['registration']}: not registrable yet: {e}\n")
            continue
        for k in ("cell", "config"):
            for n, h in pins(cell)[k].items():
                print(f"# {cell['registration']} pins {k} {n} sha256 {h}")
        print(" ".join(shlex.quote(a) for a in argv) + "\n")
        for old in (o for o, n in SUPERSEDES.items() if n == cell["registration"]):
            print(f"# {cell['registration']} supersedes {old}: after the command above, record why (one line, from "
                  "the repo root):")
            print(supersede_command(old, cell, FINALS.parent / "ledger" / "paper" / "ledger.jsonl") + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
