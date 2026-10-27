#!/usr/bin/env python3
"""The finals table generator: the ONLY path by which a number reaches the RidgeFill paper (PLAN.md "Locking and
backups"; PROTOCOL.md section 6 v1.1 and section 10.5). It reads a VERIFIED finals results tree and the ledger
(read-only), computes every table cell with finals/stats.py, and writes paper/tex/tables/*.tex in the skeleton's
layouts, paper/tex/numbers.tex's result macros, and tables/cells.json (every cell with its value, evidence tier,
protocol version and sources). Nothing is typed by hand; nothing is written unless everything is there.

Order (each step fails closed and refuses by name; exit 1, nothing written):
  1. verify   finals/verify_finals.py's checks on every cell, with --require-done: sidecars, FINAL- ledger records,
              canary, both push receipts. A cell whose cell.json differs from its FINAL- record in ANY field is
              refused too (verify compares the declared fields only).
  2. load     sample records (contract below); a cell with reference-harness results (work/results) must hold their
              current conversion (finals/convert.py: samples/ and its CONVERTED.json stamp, verify_finals.stamp_problems);
              arm names normalised through protocol.json arm_aliases; every record's protocol must be the current
              protocol's major.minor (finals run under it only) and its tier the scope's (locked-final for the reference
              cells, engine-final for Flash-Next); every record holds the exact arm; a (scope, model, set, arm, split,
              unit) filled by two cells is refused (no best-of-k).
  3. compute  every table cell the layout asks for (\\TBDcell slots of TEMPLATES); a cell whose data is absent is
              collected as MISSING, never guessed; pre-final numbers (S16 probe, cost model, negative-result rows,
              calibration, lambda, levers) come only from ledger records or files whose sha256 a ledger record pins.
  4. refuse   if anything is missing: print the missing list (grouped by reason) and write nothing.
  5. write    each file atomically. A tree registered FINAL-MOCK-* is refused when --out is the repo's paper/tex.

Finals sample-record contract (finals/convert.py writes it from the reference harness, finals/convert_c5.py from the
C5 kit for the engine tier; finals/mock/tables_tree.py writes it for the mock tree): one JSON file per (cell, set,
unit) under <cell>/samples/, with sidecar, holding
  set        ruler | ruler-qfirst (T1.4 question first) | multiturn (T1.8) | longbench | longppl | decode | ttft
  sample     the unit id (RULER row, document, prompt, or ttft "<length>#round<r>"; engine tier
             "<arm>/<length>#round<r>"); protocol (the stamp version);
             tier (locked-final | engine-final)
  task, length, needle_filled (ruler sets, NIAH rows: every scored needle ends at or before N - T, T the cell's
             declared tail; rows without a needle have no filled-region readout), domain, doc, tokens
  skipped    reason (declared cap, e.g. Qwen3-8B over 32K): no arms; counted, never imputed
  arms       {arm: {split-key: metrics}}; split keys "1/4" "3/8" "1/2" "5/8" (fractions of depth), "pop" (POP's own
             split), "-" (exact and every Flash-Next arm). Metrics: score (0-1) | turn2 (0/1) | novel_nll, key_nll,
             n_key (nats) | top1_agree (0-1), kl_mean | ttft_s | wall_s (engine-tier RULER request time). Every
             record holds exact.
Populations: a cell's configs partition its rows and run different arms on different partitions (finals/make_cells.py:
the controls and the splices on the NIAH subset, 64K for the core arms, ...), so an arm's population is the units that
carry it, and a contrast or ratio pairs the units that carry every arm in it. A unit never lacks an arm its config ran
(finals/convert.py refuses an incomplete record); an arm with a unit but no value for the metric is MISSING.
A cell's scope is "flashnext" for the engine table and "battery" for every other kind: the mechanism table reads the
splices from the NLL cells (longppl) and the RULER cells' NIAH rows, like the batteries. Primary split = the cell's
declared split (3/8).
Reference-model arm names are protocol kinds or their aliases (protocol.json arm_aliases). Flash-Next arm names (one
"flashnext" cell, the C5 tier as run: finals/engine/qwen3.8-flash-next-c5.json, generated from FINAL-FLASHNEXT-C5's
declared values, declared_share = 25, margins {ppl_ratio_upper_pct, ruler_points_lower, ttft_min_speedup, ttft_length},
split_layer, tail, and criteria.ppl_gap_upper_pct of amendment 11): exact, exact-rerun (exact's same-boot second
perplexity pass), ridgefill+corr+prot25 (quality mode), ridgefill (RidgeFill alone). Its sets: longppl (the locked
long-prompt documents <= 32K), ruler (NIAH 8K-32K and niah_single_1 64K), ttft (each arm's rounds against its own boot's
exact path). Its verify path is verify_finals.check_engine_cell (the record pins the kit, not the cell), and its
engine.json carries the determinism read the table's note prints. C5 (DEV-17): perplexity = the measured gap to
exact (paired novel-token NLL difference, document-weighted, BCa within domain) at most log(1 + criterion); RULER =
non-inferiority of the filled-region rows at the registered lower margin with the exact-binomial floor; speed = TTFT
speedup at the registered length at least the registered minimum.
A pairing unit lives in one cell: a unit whose arms come from two cells is refused (never splice). The one declared
exception is protocol 1.1.3's supersession (SUPERSESSION): the nll-causal cells' longppl records load as their own set
(longppl-causal), and every NLL value involving a superseded arm is computed inside that cell. Cut or trimmed parts of
the design (LongBench; the Llama and Qwen3.5 RULER cells' non-NIAH, question-first, SwiftKV and splice runs) print "--"
with a "not run (cut ...)" note, from registration.CUT and the RULER cells' declared runs. The 2026-10-08 follow-ups:
ruler-deep cells load as their own set (FOLLOWUP_SETS: the RULER cell's units, never a second fill of them); a cell on
another frozen set (nll-fresh, 'dataset') is its own scope (dataset:<name>, tier = the set's name: never pooled with the
locked set); both report in their own panels, outside every primary, Holm and MDE family.

Statistics: finals/stats.py (BCa 5,000 resamples, seed and estimands declared there). Strata: RULER and LongBench
rows within task, documents within domain. Holm: one family per model table (all its "Holm" rows; at least as
conservative as the protocol's per-claim families, which are subsets). MDE: primary contrasts and the MDE table.
Non-inferiority (stats.noninferiority; the declared margins are one-sided, so not equivalence): Flash-Next C5 (RULER
lower margin on the per-row deltas with the bounded-score ceiling guard; perplexity is read as the measured gap) and
C3's rows arms minus exact on the reference RULER cells whose FINAL- record declares it (C3_NONINF), margin from the cell.

Ratio statistics (gap closed, loss removed, carried by: Stats.ratio): when the statistic is not finite on some bootstrap
resample (stats.IntervalUndefined: the denominator, e.g. RidgeFill's RULER gap to exact, is zero there), no interval is
printed: the cell shows the point estimate and "--" in place of the interval (a verdict read from the interval prints
"--" with the estimate), and the table carries a note naming why (the gap in discordant rows, or RidgeFill's NLL loss);
a non-finite point estimate is "--" too. No resample is dropped or imputed.
Protocol versions: every cell's records carry one version (the stamp), of the current major.minor; the cells may run
under different patch versions of it only when each is a version the protocol adopted (a ledger protocol-change record
"protocol-change A -> B", or the current protocol.json version): anything else is refused by name. \\protocolversion is
protocol.json's version (the rules the tables are computed under), \\protocolversionsrun lists the versions the cells
ran under, and the configs table names each cell's.
Pending tiers (--pending flashnext): the Flash-Next tier has not run. Every Flash-Next table cell and \\resFN* macro
renders \\pending (defined in main.tex), the Flash-Next table and the MDE table carry the note "Flash-Next tier not yet
run", and cells.json logs each such cell as status pending (never computed). Refused when a flashnext cell, any results
directory named for it, or a FINAL- ledger record of the Flash-Next tier exists (no real data is hidden behind the flag).

--no-ledger (the public copy of a results tree, without the ledger and, when --proj-dir is not given, without the
projector files): verification is finals/verify_finals.py --no-ledger (cell.json and every sample record against
their sidecars and the conversion stamp; no FINAL- record and no .done marker is required, and the raw harness records
under work/results, whose gold answers may be stripped, are not read: the tables read only the per-sample scores in
samples/); the patch versions of the records are not checked against the ledger's protocol-change records (every
record must still carry the current major.minor). A cell that needs the ledger, a ledger-pinned file or a projector
file is not computed: it prints \\TBDcell, and the run writes everything else and lists those cells with their reasons
(stdout and cells.json "missing") instead of refusing. Without numbers.tex in --out, a numbers.tex with every computed
macro is written.

Usage: tables.py RESULTS [--repo ROOT] [--ledger L | --no-ledger] [--protocol P] [--out paper/tex] [--proj-dir D]
       [--levers F] [--battery-v10 F] [--timing F ...] [--pending flashnext]
Exit: 0 written; 1 refused (verify failure, missing cells, statistics refusal); 2 usage / missing input.
"""
import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

FINALS = Path(__file__).resolve().parent
REPO = FINALS.parent
sys.path.insert(0, str(FINALS / "lib"))
sys.path.insert(0, str(FINALS))
import atomic                                                # noqa: E402
import kvabackend                                            # noqa: E402
import refit                                                 # noqa: E402
import registration                                          # noqa: E402
import stats as S                                            # noqa: E402

TBD = r"\TBDcell"
PRIMARY = "3/8"
FRACS = ("1/4", "3/8", "1/2", "5/8")
REF_MODELS = {"dense": "llama-3.1-8b-instruct", "hybrid": "qwen3.5-9b", "qkn": "qwen3-8b"}
FN_MODEL = "qwen3.8-flash-next"
SHORT = {"llama-3.1-8b-instruct": r"\llamashort", "qwen3.5-9b": r"\qwenhybridshort", "qwen3-8b": r"\qwendense",
         FN_MODEL: r"\flashnext"}
FN_TTFT_LENS = (9216, 16384, 32768)                          # Flash-Next "9K / 16K / 32K"


class Refused(Exception):
    """A named refusal of the whole run: nothing is written."""


TIER_OF_SCOPE = {"battery": "locked-final", "flashnext": "engine-final"}
# The one declared double fill (protocol 1.1.3, paper/PROTOCOL.md change log; FINAL-*-NLL-CAUSAL): the 1.1.1 NLL
# evaluation of these arms read the scored continuation (the qattn window, the ktail / vtail scale), so each model's
# "by" cell re-runs them on the same documents with its own exact, fills and controls ("with"). Its records load under
# their own set (as_set), never next to the "of" cell's: every NLL value or contrast that involves a superseded arm at
# the split is computed entirely inside the "by" cell (one pairing unit, one host: Stats.route); the "of" cell's values
# of those arms are read only as the size of the leak (LEAK_MACROS). Any other unit filled by two cells is still
# refused (Tree.load: no best-of-k). finals/align_audit.py reads the same constant.
SUPERSESSION = dict(of="nll", by="nll-causal", set="longppl", as_set="longppl-causal", split="3/8",
                    arms=("noproj+qattn10", "kva+qattn10", "kva+ktail", "kva+vtail"),
                    with_=("exact", "noproj", "kva", "noproj+random", "kva+random", "noproj+tailmatch", "kva+tailmatch"))


def scope_of(cell):
    """flashnext (the engine table), dataset:<name> (a cell on another frozen set: never pooled with the locked set's
    results; its records carry tier = that name, finals/convert.py record_tier), else battery."""
    if cell["kind"] == "flashnext":
        return "flashnext"
    return DATASET_SCOPE + cell["dataset"] if cell.get("dataset") else "battery"


DATASET_SCOPE = "dataset:"


def tier_of_scope(scope):
    return scope[len(DATASET_SCOPE):] if scope.startswith(DATASET_SCOPE) else TIER_OF_SCOPE[scope]


# Follow-up cells (declared 2026-10-08) that run arms of another cell on the same units: their records are their own
# population (own set key; every value and contrast computed inside the cell), never a second fill of the other cell's
# units. kind -> {record set: set key}. Any other double fill is still refused.
FOLLOWUP_SETS = {"ruler-deep": {"ruler": "ruler-deep"}}


class Missing(Exception):
    """One required table cell has no data (collected; the run refuses at the end with the full list)."""


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================================ formatting
def _num(x, nd):
    s = f"{x:.{nd}f}"
    return "0." + "0" * nd if nd and float(s) == 0 else ("0" if not nd and float(s) == 0 else s)


def fsigned(x, nd):
    """Signed delta: $+0.046$ / $-0.046$ / $0.000$."""
    s = _num(abs(x), nd)
    if float(s) == 0:
        return f"${s}$"
    return f"${'+' if x > 0 else '-'}{s}$"


def fplain(x, nd):
    return _num(x, nd) if x >= 0 else f"${'-'}{_num(-x, nd)}$"


def fpct(x, nd=1, signed=False):
    return (fsigned(x, nd)[:-1] + r"\%$") if signed else f"{fplain(x, nd)}\\%"


def fci(lo, hi, nd, pct=False, signed=True):
    if pct:
        f = (lambda v: fpct(v, nd, signed=True)) if signed else (lambda v: fpct(v, nd))
    else:
        f = (lambda v: fsigned(v, nd)) if signed else (lambda v: fplain(v, nd))
    return f"[{f(lo)}, {f(hi)}]"


def tex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("$", r"\$").replace("{", r"\{").replace("}", r"\}"))


def frac_float(key):
    return float(Fraction(key))


def mm(version):
    return ".".join(str(version).split(".")[:2])


# ============================================================================================ loading
class Tree:
    """The verified results tree, indexed for the tables."""

    def __init__(self, results, ledger_path, protocol_path):
        """ledger_path None: --no-ledger (module doc)."""
        self.results = Path(results)
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.protocol = json.loads(Path(protocol_path).read_text())
        self.aliases = self.protocol.get("arm_aliases", {})
        self.ledger = ([json.loads(line) for line in self.ledger_path.read_text().splitlines() if line.strip()]
                       if self.ledger_path is not None else [])
        self.cells = {}                                       # name -> declared cell (== FINAL record's cell)
        self.cell_dirs = {}                                   # name -> its results dir
        self.vf = _load_by_path("finals_verify", FINALS / "verify_finals.py")
        self.data = {}                                        # (scope, model, set) -> {unit: record}
        self.skipped = {}                                     # (scope, model, set) -> {unit: reason}
        self.versions = set()
        self.cell_versions = {}                               # cell name -> {protocol version of its records}

    # ---- step 1: verify
    def verify(self):
        vf = self.vf
        cell_dirs = sorted(d for d in self.results.iterdir() if d.is_dir() and not d.name.startswith(".")
                           and (d / "cell.json").exists()) if self.results.is_dir() else []
        if not cell_dirs:
            raise Refused(f"VERIFY FAILED: {self.results} holds no finals cell (a cell dir has cell.json)")
        problems = []
        for d in cell_dirs:
            p, info = vf.check_cell(d, self.ledger_path)
            problems += p
            if info and not info["done"] and self.ledger_path is not None:
                problems.append(f"{d.name}: no .done marker (the tables require --require-done)")
        if problems:
            raise Refused("VERIFY FAILED (finals/verify_finals.py): no table is built from unverified results:\n  "
                          + "\n  ".join(problems))
        return cell_dirs

    def final_record(self, registration):
        for rec in self.ledger:
            if rec.get("type") == "hypothesis" and rec.get("id") == registration:
                return rec
        return None

    def canon(self, arm, scope):
        if scope == "flashnext":
            return arm
        kind = self.aliases.get(arm, arm)
        known = set(self.protocol.get("arms", [])) | set(self.protocol.get("outside_battery", [])) | {"exact"}
        if kind not in known:
            raise Refused(f"arm {arm!r} is no protocol kind or alias of protocol {self.protocol['version']}")
        return kind

    # ---- step 2: load
    def load(self):
        cur = mm(self.protocol["version"])
        for d in self.verify():
            cell = json.loads((d / "cell.json").read_text())
            if self.ledger_path is not None:                  # --no-ledger: verify checked cell.json's sidecar
                rec = self.final_record(cell["registration"])
                # the FINAL- record pins cell.json's sha256 (finals/lib/registration.py; labbook stores text, so the
                # pin, not a copy of the cell, is the declaration's anchor); verify checked it, this repeats it here
                pinned = registration.pins_of(rec.get("claim"))["cell"].get(cell["name"])
                if cell.get("driver") == "c5-kit":            # the engine tier: its record pins the kit, not the cell;
                    pinned = atomic.sha256_file(d / "cell.json")   # verify checked it against the declared values
                if pinned != atomic.sha256_file(d / "cell.json"):
                    raise Refused(f"{d.name}: cell.json is not the cell its FINAL- record {cell['registration']} pins "
                                  "(the declaration made before the run is the only one the tables read)")
            if (d / "work" / "results").is_dir():
                problems = self.vf.stamp_problems(d, raw=self.ledger_path is not None)
                if problems:
                    raise Refused("UNCONVERTED HARNESS RESULTS: the tables read finals sample records, which "
                                  "finals/convert.py writes from work/results: " + "; ".join(problems))
            self.cells[cell["name"]] = cell
            self.cell_dirs[cell["name"]] = d
            scope = scope_of(cell)
            model = cell["model"]["name"]
            for f in sorted((d / "samples").rglob("*.json")):
                if f.name == self.vf.STAMP:
                    continue
                r = json.loads(f.read_text())
                for k in ("set", "sample", "protocol", "tier"):
                    if k not in r:
                        raise Refused(f"{d.name}/{f.relative_to(d)}: record has no {k!r} (sample-record contract)")
                if mm(r["protocol"]) != cur:
                    raise Refused(f"{d.name}/{f.name}: protocol {r['protocol']} is not the current protocol "
                                  f"{self.protocol['version']}: finals run under the current protocol only, and "
                                  "results of two eras never share a table (PROTOCOL.md section 9)")
                if r["tier"] != tier_of_scope(scope) or r.get("dataset") != cell.get("dataset"):
                    raise Refused(f"{d.name}/{f.relative_to(d)}: evidence tier {r['tier']!r} (dataset "
                                  f"{r.get('dataset')!r}) in a {scope} table, which takes {tier_of_scope(scope)} only "
                                  "(a dev-set record is never a finals number; another frozen set's records are never "
                                  "pooled with the locked set's)")
                if "skipped" not in r and "exact" not in r.get("arms", {}):
                    raise Refused(f"{d.name}/{f.relative_to(d)}: no exact arm: every pairing unit holds the exact "
                                  "model's run (sample-record contract)")
                self.versions.add(r["protocol"])
                self.cell_versions.setdefault(cell["name"], set()).add(r["protocol"])
                set_ = r["set"]
                if cell["kind"] == SUPERSESSION["by"]:
                    set_ = self.superseding(cell, d, f, r)
                    r = dict(r, set=set_)
                elif cell["kind"] in FOLLOWUP_SETS:
                    own = FOLLOWUP_SETS[cell["kind"]]
                    if set_ not in own:
                        raise Refused(f"{d.name}/{f.relative_to(d)}: a {cell['kind']} cell records set {set_!r}; it "
                                      f"runs {sorted(own)} only")
                    set_ = own[set_]
                    r = dict(r, set=set_)
                key = (scope, model, set_)
                if "skipped" in r:
                    self.skipped.setdefault(key, {})[r["sample"]] = (r["skipped"], r)
                    continue
                arms = {self.canon(arm, scope): by_split for arm, by_split in r["arms"].items()}
                slot = self.data.setdefault(key, {})
                if r["sample"] in slot:
                    other = slot[r["sample"]]
                    twice = sorted(f"{a}@{s}" for a, sp in arms.items() for s in sp
                                   if a != "exact" and s in other["arms"].get(a, {}))
                    raise Refused((f"{scope} {model} {r['set']} {r['sample']}: {twice} is filled by two cells "
                                   f"({other['_cell']}, {cell['name']}): no best-of-k; " if twice else "")
                                  + f"pairing unit {r['sample']} ({model} {r['set']}) is split across cells "
                                  f"{other['_cell']} and {cell['name']}: every arm of a unit runs in one cell; never "
                                  "splice (PLAN.md contingencies; finals/convert.py merges a cell's configs into one "
                                  "record)")
                slot[r["sample"]] = dict(r, arms=arms, _cell=cell["name"])
        if not self.versions:
            raise Refused("the results tree holds no sample record")
        self.check_versions()
        return self

    def adopted_versions(self):
        """The protocol versions the protocol adopted: every 'protocol-change A -> B' ledger record's B, and the current
        protocol.json version."""
        out = {self.protocol["version"]}
        for rec in self.ledger:
            for text in (rec.get("note"), rec.get("claim")):
                out |= set(re.findall(r"protocol-change \S+ -> (\d+(?:\.\d+)*)", text or ""))
        return out

    def check_versions(self):
        """A declared mix only: one version per cell, each adopted by the protocol (module doc, Protocol versions)."""
        for name, vs in sorted(self.cell_versions.items()):
            if len(vs) != 1:
                raise Refused(f"{name}: its records carry protocol versions {sorted(vs)}: a cell runs under one "
                              "version (finals/convert.py refuses a mixed cell)")
        if self.ledger_path is None:
            return                                            # --no-ledger: no protocol-change records to check against
        adopted = self.adopted_versions()
        bad = {n: v for n, vs in self.cell_versions.items() for v in vs if v not in adopted}
        if bad:
            raise Refused(f"protocol versions {sorted(set(bad.values()))} (cells {sorted(bad)}) are not versions the "
                          f"protocol adopted (ledger protocol-change records and protocol.json {self.protocol['version']}"
                          f": {sorted(adopted)}): an undeclared mix of versions never shares a table")

    def cell_version(self, name):
        return next(iter(self.cell_versions[name])) if name in self.cell_versions else None

    def superseding(self, cell, d, f, r):
        """The set key of a record of the declared superseding cell (SUPERSESSION); anything it was not declared to
        re-run is refused."""
        sp = SUPERSESSION
        if r["set"] != sp["set"]:
            raise Refused(f"{d.name}/{f.relative_to(d)}: a {sp['by']} cell records set {r['set']!r}; it supersedes "
                          f"{sp['of']} arms on {sp['set']} only (protocol 1.1.3)")
        for arm, by_split in r.get("arms", {}).items():
            kind = self.canon(arm, "battery")
            bad = [s for s in by_split if not ((kind in sp["arms"] and s == sp["split"]) or
                                               (kind in sp["with_"] and s in (sp["split"], "-")))]
            if kind not in sp["arms"] + sp["with_"] or bad:
                raise Refused(f"{d.name}/{f.relative_to(d)}: arm {arm}@{bad or sorted(by_split)} is not part of the "
                              f"declared supersession (superseded {list(sp['arms'])} at {sp['split']}, alongside "
                              f"{list(sp['with_'])}): no best-of-k")
        return sp["as_set"]

    def is_mock(self):
        return any(c["registration"].startswith("FINAL-MOCK") for c in self.cells.values())

    def cells_of(self, model, scope=None):
        return [c for c in self.cells.values() if c["model"]["name"] == model and scope in (None, scope_of(c))]

    # ---- access
    def records(self, scope, model, set_):
        recs = self.data.get((scope, model, set_))
        if not recs:
            raise Missing(f"no {set_} records for {model} in the {scope} cells")
        return recs

    def values(self, scope, model, set_, arm, split, metric, where=None, scale=1.0):
        """{unit: value} of one arm over its population (the units that carry it at this split: module doc,
        Populations); a unit of the population without the metric is MISSING."""
        recs = self.records(scope, model, set_)
        out, lacking = {}, []
        for unit, r in recs.items():
            if where is not None and not where(r):
                continue
            m = r["arms"].get(arm, {}).get(split)
            if m is None:
                continue                                      # not run on this unit by the declared configs
            if metric not in m or m[metric] is None:
                lacking.append(unit)
                continue
            out[unit] = float(m[metric]) * scale
        if lacking and out:
            raise Missing(f"{model} {set_}: arm {arm} split {split} metric {metric} lacks {len(lacking)} of "
                          f"{len(lacking) + len(out)} units (e.g. {lacking[0]})")
        if not out:
            raise Missing(f"{model} {set_} ({scope}): no unit has arm {arm} at split {split} with {metric}"
                          + (" in the population" if where else ""))
        return out

    def ledger_by_label(self, label, type_=None):
        hits = [r for r in self.ledger if (r.get("label") == label or r.get("id") == label)
                and (type_ is None or r.get("type") == type_)]
        if not hits:
            raise Missing(f"no ledger record {label!r}" + (f" of type {type_}" if type_ else ""))
        return hits[-1]

    def require_record(self, label, type_):
        """A declaration the code names must exist in the ledger; --no-ledger: not checked (module doc)."""
        if self.ledger_path is not None:
            self.ledger_by_label(label, type_)

    def pinned(self, repo, rel):
        """A repo file whose sha256 a ledger record pins (sources are relative to the ledger's directory)."""
        p = Path(repo) / rel
        if not p.is_file():
            raise Missing(f"{rel} does not exist under {repo}")
        digest = atomic.sha256_file(p)
        for r in self.ledger:
            for src, h in (r.get("sources") or {}).items():
                if (self.ledger_path.parent / src).resolve() == p.resolve() and h == digest:
                    return p
        raise Missing(f"{rel} (sha256 {digest[:12]}) is pinned by no ledger record: append one with "
                      f"`labbook record --source` before its numbers are read")


# ============================================================================================ statistics helpers
STRATA_KEY = {"ruler": "task", "ruler-deep": "task", "ruler-qfirst": "task", "multiturn": "task", "longbench": "task", "longppl": "domain",
              "longppl-causal": "domain", "decode": None}
SUBSETS = {"64k": lambda r: r.get("length") == 65536}
CAUSAL_OF = {"nll_novel": "nll_novel_causal", "nll_key": "nll_key_causal"}
RULER_UNIT_RANGE = (-100.0, 100.0)    # one RULER row's paired score difference in points (score 0-1 x 100): the bound
                                      # of the non-inferiority ceiling guard (finals/stats.py noninferiority)


class Stats:
    """Cached contrasts over the tree (one BCa per distinct contrast; the declared seed for every one)."""

    def __init__(self, tree):
        self.t = tree
        self.cache = {}

    def metric(self, name):
        """name -> (set, metric key, scale, estimand, population filter)."""
        filled = lambda r: bool(r.get("needle_filled"))                         # noqa: E731
        return {
            "ruler_all": ("ruler", "score", 100.0, "accuracy", None),
            "ruler_filled": ("ruler", "score", 100.0, "accuracy", filled),
            "deep_filled": ("ruler-deep", "score", 100.0, "accuracy", filled),
            "qfirst_all": ("ruler-qfirst", "score", 100.0, "accuracy", None),
            "qfirst_filled": ("ruler-qfirst", "score", 100.0, "accuracy", filled),
            "longbench": ("longbench", "score", 100.0, "longbench", None),
            "nll_novel": ("longppl", "novel_nll", 1.0, "nll", None),
            "nll_key": ("longppl", "key_nll", 1.0, "nll", lambda r: True),
            "nll_novel_causal": (SUPERSESSION["as_set"], "novel_nll", 1.0, "nll", None),
            "nll_key_causal": (SUPERSESSION["as_set"], "key_nll", 1.0, "nll", lambda r: True),
            "agree": ("decode", "top1_agree", 100.0, "agreement", None),
            "kl": ("decode", "kl_mean", 1.0, "kl", None),
            "turn2": ("multiturn", "turn2", 100.0, "accuracy", None),
        }[name]

    def vals(self, scope, model, name, arm, split, where=None):
        set_, key, scale, _, own = self.metric(name)
        if name in ("nll_key", "nll_key_causal"):   # documents with key tokens only (the key mask is the exact model's)
            own = lambda r: (r["arms"].get("exact", {}).get("-") or {}).get("n_key", 0) > 0     # noqa: E731
        both = own if where is None else where if own is None else (lambda r: own(r) and where(r))
        return self.t.values(scope, model, set_, arm, split, key, where=both, scale=scale)

    @staticmethod
    def route(name, *arm_splits):
        """The metric a value or contrast reads: an NLL metric involving a superseded arm at the split (SUPERSESSION)
        reads the superseding cell's records, all of its arms, the exact one included."""
        sp = SUPERSESSION
        if name in CAUSAL_OF and any(a in sp["arms"] and s == sp["split"] for a, s in arm_splits):
            return CAUSAL_OF[name]
        return name

    def mean(self, scope, model, name, arm, split, vs_exact=False, where=None, route=True):
        """Descriptive estimand mean of an arm (or of arm - exact) over its population (where: a further filter).
        route=False reads the metric as named (the superseded values: LEAK_MACROS only)."""
        if route:
            name = self.route(name, (arm, split))
        v = self.vals(scope, model, name, arm, split, where)
        if vs_exact:
            e = self.vals(scope, model, name, "exact", "-", where)
            units, a, b = S.paired(*common(v, e, f"{model} {name} {arm}@{split} - exact"),
                                   what=f"{model} {name} {arm}@{split} - exact")
            x, units_ = a - b, units
        else:
            units_, x = sorted(v), np.array([v[u] for u in sorted(v)])
        set_, _, _, estimand, _ = self.metric(name)
        if S.weighting_of(estimand) == "stratum":
            recs = self.t.records(scope, model, set_)
            labs = [recs[u].get(STRATA_KEY[set_]) for u in units_]
            return float(np.mean([x[[i for i, l in enumerate(labs) if l == g]].mean() for g in sorted(set(labs))]))
        return float(x.mean())

    def contrast(self, scope, model, name, a, b, *, name_b=None, subset=None):
        """Paired A - B (arm, split) contrast on metric `name` (B's metric name_b for a cross-set contrast); subset:
        a SUBSETS key restricting the population (e.g. "64k").

        Returns dict(n, est, lo, hi, p, sd, mde, deltas)."""
        if name_b is None:
            name = self.route(name, a, b)
        key = (scope, model, name, a, b, name_b) + ((subset,) if subset else ())
        if key in self.cache:
            return self.cache[key]
        where = SUBSETS[subset] if subset else None
        va = self.vals(scope, model, name, *a, where=where)
        vb = self.vals(scope, model, name_b or name, *b, where=where)
        # the units that carry both arms (a cross-set contrast: one RULER row under two prompt layouts, filled in both)
        units, xa, xb = S.paired(*common(va, vb, f"{model} {name} {a} - {name_b or name} {b}"),
                                 what=f"{model} {name}: {a} - {b}")
        set_, _, _, estimand, _ = self.metric(name)
        recs = self.t.records(scope, model, set_)
        strata = [recs[u].get(STRATA_KEY.get(set_)) for u in units] if STRATA_KEY.get(set_) else None
        d = xa - xb
        r = S.bca(d, strata=strata, weighting=S.weighting_of(estimand))
        sd, n, m = S.mde_paired(d)
        out = dict(n=n, est=r.estimate, lo=r.lo, hi=r.hi, p=S.pvalue(r), sd=sd, mde=m, deltas=d, units=units)
        self.cache[key] = out
        return out

    def noninferiority(self, scope, model, name, a, b, *, lower=None, upper=None, unit_range=None):
        """Non-inferiority of the paired A - B contrast against declared one-sided margins (finals/stats.py
        noninferiority), on the contrast's units with its strata and estimand weighting."""
        name = self.route(name, a, b)
        key = ("noninferiority", scope, model, name, a, b, lower, upper, unit_range)
        if key not in self.cache:
            c = self.contrast(scope, model, name, a, b)
            set_, _, _, estimand, _ = self.metric(name)
            recs = self.t.records(scope, model, set_)
            strata = [recs[u].get(STRATA_KEY[set_]) for u in c["units"]] if STRATA_KEY.get(set_) else None
            self.cache[key] = S.noninferiority(c["deltas"], lower=lower, upper=upper, unit_range=unit_range,
                                               strata=strata, weighting=S.weighting_of(estimand))
        return self.cache[key]

    def ratio(self, scope, model, name, cols, stat, level=S.LEVEL, why=None):
        """BCa of stat(means of the given arm columns), units paired across columns. When the statistic is not finite
        on some resample (stats.IntervalUndefined), an UndefinedRatio: the estimate on the data, no interval, and the
        note why(x) names the reason (x: the (n, k) paired columns)."""
        name = self.route(name, *cols)
        vs = [self.vals(scope, model, name, *c) for c in cols]
        units = sorted(set.intersection(*(set(v) for v in vs)))
        if not units:
            raise Missing(f"{model} {name}: no unit carries every arm of {cols}")
        x = np.array([[v[u] for v in vs] for u in units])
        set_, _, _, estimand, _ = self.metric(name)
        recs = self.t.records(scope, model, set_)
        strata = [recs[u].get(STRATA_KEY.get(set_)) for u in units] if STRATA_KEY.get(set_) else None
        try:
            return S.bca(x, stat=stat, strata=strata, weighting=S.weighting_of(estimand), level=level)
        except S.IntervalUndefined as e:
            if why is None:
                raise
            return UndefinedRatio(e.estimate, len(units), f"{why(x)} (not finite on {e.n_bad} of {e.n_resamples} "
                                                          "resamples)")

    def ttft_speedup(self, scope, model, arm, split, length):
        """Median over rounds of exact / median over rounds of the arm, at one prompt length, over the rounds that carry
        the arm (its population; on the engine tier each arm's rounds carry its own boot's exact path)."""
        recs = self.t.records(scope, model, "ttft")
        ex, ar = [], []
        for r in recs.values():
            if r.get("length") != length or split not in r["arms"].get(arm, {}):
                continue
            e = r["arms"].get("exact", {}).get("-", {}).get("ttft_s")
            a = r["arms"].get(arm, {}).get(split, {}).get("ttft_s")
            if e is None or a is None:
                raise Missing(f"{model} ttft {length}: round {r['sample']} lacks exact or {arm}@{split}")
            ex.append(e)
            ar.append(a)
        if not ex:
            raise Missing(f"{model} ttft: no round at length {length} for {arm}@{split}")
        return statistics.median(ex) / statistics.median(ar)


class UndefinedRatio:
    """A ratio statistic whose bootstrap interval does not exist (stats.IntervalUndefined): the estimate on the data
    (possibly non-finite), never an interval (lo / hi are None, so no caller can read one), and the note."""

    def __init__(self, estimate, n, note):
        self.estimate, self.n, self.note, self.lo, self.hi = estimate, n, note, None, None


class Noted:
    """A cell value that carries a table note (Run.cell): value None prints "--"."""

    def __init__(self, value, note):
        self.value, self.note = value, note


def r_est(r):
    """The point estimate of a ratio statistic ("--" with the note when it is not finite; the note also when only the
    interval is undefined, so a printed estimate never hides that)."""
    if isinstance(r, UndefinedRatio):
        return Noted(r.estimate if math.isfinite(r.estimate) else None, r.note)
    return r.estimate


def r_bound(r, k):
    """One end ("lo" / "hi") of a ratio's interval: "--" with the note when the interval is undefined."""
    return Noted(None, r.note) if isinstance(r, UndefinedRatio) else getattr(r, k)


def r_ci(r):
    return Noted(None, r.note) if isinstance(r, UndefinedRatio) else (r.lo, r.hi)


def common(va, vb, what):
    """Both {unit: value} maps restricted to the units that carry both arms (module doc, Populations)."""
    both = set(va) & set(vb)
    if not both:
        raise Missing(f"{what}: no unit carries both arms")
    return {u: va[u] for u in both}, {u: vb[u] for u in both}


def share_verdict(lo, hi, est):
    """Restoration verdict from a share's CI (percent): consistent with full / none / neither / both."""
    tag = f"({fpct(est)})"
    if lo <= 0 and hi >= 100:
        return f"inconclusive {tag}"
    if hi >= 100:
        return f"yes {tag}"
    if lo <= 0:
        return f"no {tag}"
    return f"partial {tag}"


# ============================================================================================ the run context
class Run:
    def __init__(self, tree, st, repo, args):
        self.t, self.st, self.repo, self.args = tree, st, Path(repo), args
        self.missing = []                                     # (where, reason)
        self.cellslog = []                                    # cells.json entries
        self.macros = {}
        self.macro_src = {}
        self.table = "numbers.tex"
        self.notes = {}                                       # table -> [note] (Noted values), in order, once each
        self.flushed = {}                                     # table -> notes already printed under a panel
        self.pending = set(getattr(args, "pending", None) or ())   # tiers declared not yet run (--pending)
        self.last_note = None

    def pend(self, where, sources=""):
        """A cell of a tier declared pending (--pending): the \\pending marker, logged as pending, never computed."""
        self.cellslog.append({"table": self.table, "cell": where, "value": PENDING, "tier": "pending",
                              "protocol": "pending", "status": "pending", "sources": sources})
        return PENDING

    def note(self, text):
        lst = self.notes.setdefault(self.table, [])
        if text not in lst:
            lst.append(text)

    def flush_notes(self, table):
        """The notes of `table` not yet printed, as one footnote line (under the panel that collected them)."""
        done = self.flushed.setdefault(table, [])
        new = [n for n in self.notes.get(table, []) if n not in done]
        done += new
        # notes with the same reason are merged into one ("A and B: reason"; review 2026-10-11)
        groups = {}
        for n in new:
            who, sep, why = n.partition(": interval undefined: ")
            groups.setdefault(sep + why if sep else n, []).append(who if sep else None)
        merged = [(and_list([w for w in whos if w]) + why) if any(whos) else why for why, whos in groups.items()]
        return [rf"{{\footnotesize ``--'' in place of an interval: {'; '.join(merged)}.\par}} % RATIO-UNDEFINED"] if new else []

    def cell(self, where, fn, fmt, tier, sources=""):
        """Compute one table value; a Missing or a statistics refusal is collected, never guessed. A Noted value
        prints "--" (value None) or its value, and its note goes to the table (numbers.tex: the macro's gen note)."""
        self.last_note = None
        try:
            v = fn()
        except Missing as e:
            self.missing.append((where, str(e)))
            return TBD
        except S.StatsError as e:
            self.missing.append((where, f"statistics refused: {e}"))
            return TBD
        except KeyError as e:
            self.missing.append((where, f"a record lacks the field {e} (sample-record contract)"))
            return TBD
        entry = {}
        if isinstance(v, Noted):
            v, self.last_note = v.value, v.note
            self.note(self.last_note)
            entry["note"] = self.last_note
        s = "--" if v is None else fmt(v)
        self.cellslog.append({"table": self.table, "cell": where, "value": s, "tier": tier,
                              "protocol": tier_protocol(self, tier), "sources": sources, **entry})
        return s

    def macro(self, name, fn, fmt, tier, sources=""):
        self.macros[name] = self.cell(f"numbers.tex \\{name}", fn, fmt, tier, sources)
        self.macro_src[name] = f"{tier}{'; ' + sources if sources else ''}" + (f"; {self.last_note}" if self.last_note
                                                                               else "")

    def macro_pending(self, name, tier):
        """A pending macro; a constant (\\cst...) keeps the \\xspace its prose relies on."""
        self.macros[name] = self.pend(f"numbers.tex \\{name}", tier) + (r"\xspace" if name.startswith("cst") else "")
        self.macro_src[name] = f"pending: {PENDING_NOTE[tier]}"


PENDING = r"\pending"
PENDING_TIERS = ("flashnext",)
PENDING_NOTE = {"flashnext": "Flash-Next tier not yet run"}


def tier_protocol(run, tier):
    if tier in ("locked-final", "engine-final", "declared") or tier in {c.get("dataset") for c in run.t.cells.values()}:
        return ",".join(sorted(run.t.versions))
    return tier.split(":", 1)[1] if ":" in tier else ""


# ============================================================================================ battery tables
BAT_A = [  # (row label in the template, arm kind, split key or "P" for the primary split)
    ("Exact", "exact", "-"), ("No fill", "skip", "P"), ("Identity", "noproj", "P"), ("POP native", "pop-native", "pop"),
    (r"\method", "kva", "P"), ("Identity + rows", "noproj+qattn10", "P"), (r"\method + rows", "kva+qattn10", "P"),
    (r"\method + random rows", "kva+random", "P"), (r"\method + longer tail", "kva+tailmatch", "P"),
    ("No fill + longer tail", "skip+tailbudget", "P"), ("Identity + rows, q-blind", "noproj+qattn10", "P"),
    (r"\method + rows, q-blind", "kva+qattn10", "P"), ("SwiftKV maps + exact tail", "swiftkv", "1/2"),
    ("SwiftKV native", "swiftkv-native", "1/2"),
]
# T1.4 ("q-blind" rows in the templates, printed "question-first (position)": DEV-10, a question-POSITION test): the
# cells run the question-attention rows arms (qattn10: the selector reads the last 64 rows)
# on the question-first prompts (set ruler-qfirst, kva/derived.py), where those rows hold no key; the protocol kinds
# noproj+qblind10 / kva+qblind10 (a prefix window on question-last prompts) are not what runs. These rows read set
# ruler-qfirst.
QFIRST_ROWS = ("Identity + rows, q-blind", r"\method + rows, q-blind")
# DEV-8 (the coordinator's decisions, 2026-10-04 night): cells the finals do not run print "--" and the table carries the
# one-line \notRunNote (numbers.tex): every TTFT cell (the engine timing path is a pending decision; the generator builds
# without it) and LongBench for the controls and SwiftKV maps (LongBench is secondary; not run).
LONGBENCH_NOT_RUN = ("kva+random", "kva+tailmatch", "skip+tailbudget", "swiftkv")
NOT_RUN_NOTE = r"{\footnotesize\notRunNote\par}"
NOT_RUN_TABLES = ("battery-dense", "battery-hybrid", "battery-dense-qkn")    # splitcurve: no "--" left after DEV-22
BAT_COLS = ["RULER all", "RULER filled", "LongBench avg", "dNLL novel", "dNLL key", "Decode agree", "Decode KL",
            "Turn-2 RULER", "TTFT speedup"]

P, R12 = PRIMARY, "1/2"
M_, ID_, NF_ = r"\method", "identity", "no fill"
LBL = {"ruler_filled": "RULER filled", "nll_novel": r"$\Delta$NLL novel", "nll_key": r"$\Delta$NLL key",
       "longbench": "LongBench avg", "agree": "Decode agree", "turn2": "Turn-2 RULER", "qfirst_filled": "RULER filled"}
# panel B rows, in the template's order: (contrast label, metric label) as printed, then A arm, A split, B arm,
# B split, metric, B's metric for a cross-set contrast. The labels are checked against the template at render time.
CONTRASTS = {
    "dense": [
        (rf"{M_} $-$ identity", "ruler_filled", "kva", P, "noproj", P, "ruler_filled", None),
        (rf"{M_} $-$ identity", "nll_novel", "kva", P, "noproj", P, "nll_novel", None),
        (rf"{M_} + rows $-$ identity + rows", "ruler_filled", "kva+qattn10", P, "noproj+qattn10", P, "ruler_filled", None),
        (rf"{M_} $-$ no fill", "ruler_filled", "kva", P, "skip", P, "ruler_filled", None),
        (rf"{M_} $-$ no fill", "nll_novel", "kva", P, "skip", P, "nll_novel", None),
        (rf"{M_} $-$ no fill + longer tail", "nll_novel", "kva", P, "skip+tailbudget", P, "nll_novel", None),
        (rf"{M_} $-$ identity", "nll_key", "kva", P, "noproj", P, "nll_key", None),
        (rf"{M_} $-$ identity", "longbench", "kva", P, "noproj", P, "longbench", None),
        (rf"{M_} $-$ identity", "agree", "kva", P, "noproj", P, "agree", None),
        (rf"{M_} + rows $-$ {M_} + random rows", "ruler_filled", "kva+qattn10", P, "kva+random", P, "ruler_filled", None),
        (rf"{M_} + rows $-$ {M_} + longer tail", "ruler_filled", "kva+qattn10", P, "kva+tailmatch", P, "ruler_filled", None),
        (rf"{M_} + rows $-$ identity + rows", "turn2", "kva+qattn10", P, "noproj+qattn10", P, "turn2", None),
        (rf"{M_} + rows $-$ identity + rows, q-blind", "qfirst_filled", "kva+qattn10", P, "noproj+qattn10", P,
         "qfirst_filled", None),
        (rf"{M_} $-$ POP native", "nll_novel", "kva", P, "pop-native", "pop", "nll_novel", None),
        (rf"SwiftKV maps + exact tail $-$ {M_}", "nll_novel", "swiftkv", R12, "kva", R12, "nll_novel", None),
        (rf"SwiftKV native $-$ {M_}", "nll_novel", "swiftkv-native", R12, "kva", R12, "nll_novel", None),
    ],
    "other": [
        (rf"{M_} $-$ identity", "ruler_filled", "kva", P, "noproj", P, "ruler_filled", None),
        (rf"{M_} $-$ identity", "nll_novel", "kva", P, "noproj", P, "nll_novel", None),
        (rf"{M_} + rows $-$ identity + rows", "ruler_filled", "kva+qattn10", P, "noproj+qattn10", P, "ruler_filled", None),
        ("identity + rows $-$ exact", "ruler_filled", "noproj+qattn10", P, "exact", "-", "ruler_filled", None),
        (rf"{M_} + rows $-$ exact", "ruler_filled", "kva+qattn10", P, "exact", "-", "ruler_filled", None),
        (rf"{M_} + rows $-$ {M_} + random rows", "ruler_filled", "kva+qattn10", P, "kva+random", P, "ruler_filled", None),
        (rf"{M_} + rows $-$ {M_} + longer tail", "ruler_filled", "kva+qattn10", P, "kva+tailmatch", P, "ruler_filled", None),
        (rf"{M_} $-$ no fill", "nll_novel", "kva", P, "skip", P, "nll_novel", None),
        (rf"{M_} $-$ identity", "nll_key", "kva", P, "noproj", P, "nll_key", None),
        (rf"{M_} $-$ identity", "longbench", "kva", P, "noproj", P, "longbench", None),
        (rf"{M_} $-$ identity", "agree", "kva", P, "noproj", P, "agree", None),
        (rf"{M_} + rows $-$ identity + rows", "turn2", "kva+qattn10", P, "noproj+qattn10", P, "turn2", None),
        ("identity + rows, q-blind $-$ identity + rows", "qfirst_filled", "noproj+qattn10", P, "noproj+qattn10", P,
         "qfirst_filled", "ruler_filled"),
        (rf"{M_} $-$ POP native", "nll_novel", "kva", P, "pop-native", "pop", "nll_novel", None),
    ],
}
# C3's rows-versus-exact non-inferiority (DEV-9): declared in FINAL-QWEN35-RULER and FINAL-QWEN3-8B-RULER ("identity+rows
# and ridgefill+rows minus exact >= <ruler_points_lower> points ... on the pooled filled-region accuracy"); not declared
# for Llama (FINAL-LLAMA-RULER, FINAL-LLAMA-RULER-2). model -> (table, numbers.tex macro prefix).
C3_NONINF = {"qwen3.5-9b": ("battery-hybrid", "resQwen"), "qwen3-8b": ("battery-dense-qkn", "resQwenDense"),
             "llama-3.1-8b-instruct": ("battery-dense", "resLlama")}
# Llama's rows-versus-exact test is an analysis-only follow-up on the RULER-2 cell, registered 2026-10-08 AFTER an
# unregistered readout of the same data (claim 4 narrowed in OUTLINE.md revision 4): the panel and its macros carry the
# post hoc label, and the record must exist in the ledger (refused by name otherwise). model -> ledger ID.
C3_POSTHOC = {"llama-3.1-8b-instruct": "FINAL-LLAMA-RULER-ROWS"}
C3_ROWS = [("identity + rows $-$ exact", "noproj+qattn10", "IdRows"), (rf"{M_} + rows $-$ exact", "kva+qattn10", "FillRows")]
# DEV-23: "binomial floor" for both cases stats.noninferiority uses it (no sample worse: the exact bound; some sample
# worse: the floor replaces a less conservative BCa bound); the captions say which is which (NONINF_RULE)
METHOD_LABEL = {"bca": "BCa", "exact-binomial-floor": "binomial floor"}
METRIC_ND = {"ruler_filled": 1, "qfirst_filled": 1, "longbench": 1, "turn2": 1, "agree": 1, "nll_novel": 3,
             "nll_key": 3}


# Protocol 1.1.3 (Dylan, 2026-10-07): LongBench is cut (registration.CUT: no cell), and the Llama and Qwen3.5 RULER cells
# are trimmed to the NIAH tasks (FINAL-LLAMA-RULER-2, FINAL-QWEN35-RULER-2: no other task, question-first or SwiftKV RULER
# run, Qwen3.5 no RULER splices; the question-blind pair qblind10 added). What a model's cells do not run prints "--",
# and the table carries a "not run (cut ...)" note (DEV-11): a column never silently changes its population.
CUT_DATE = "2026-10-07"
ALL_RULER_TASKS = ("niah_single_1", "niah_multikey_1", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1")
QBLIND = ("noproj+qblind10", "kva+qblind10")
NOT_RUN_WHAT = {"longbench": "LongBench", "ruler_all": "RULER on the other four tasks",
                "qfirst": "the question-first (position) arms", "swiftkv": "SwiftKV on RULER",
                "splices": "the RULER splices (exact keys / values)", "qblind": "the question-blind rows"}


def ruler_declared(run, model):
    """What the model's RULER cell declares: {tasks, qfirst, swiftkv, arms (protocol kinds)} from its runs (a cell
    without runs, the mock's full design: every task, its sets and arms). None without a RULER cell."""
    cs = [c for c in run.t.cells_of(model, "battery") if c["kind"] == "ruler"]
    if len(cs) != 1:
        return None
    c = cs[0]
    if "runs" not in c:
        arms = {run.t.canon(a, "battery") for a in c.get("arms", [])}
        return dict(tasks=set(ALL_RULER_TASKS), qfirst="ruler-qfirst" in c.get("sets", ["ruler-qfirst"]),
                    swiftkv=bool({"swiftkv", "swiftkv-native"} & arms), arms=arms)
    rs = c["runs"]
    plain = [r for r in rs if r.get("set", "ruler") == "ruler" and not r.get("variant") and r.get("role") != "swiftkv"]
    return dict(tasks={t for r in plain for t in (r.get("tasks") or ALL_RULER_TASKS)},
                qfirst=any(r.get("variant") == "qfirst" for r in rs),
                swiftkv=any(r.get("role") == "swiftkv" for r in rs),
                arms={run.t.canon(kvabackend.PAPER.get(a, a), "battery") for r in rs for a in r["arms"]})   # paper names


def not_run(run, model, what):
    """The NOT_RUN_WHAT label when the model's declared cells do not run `what` (protocol 1.1.3's cuts), else None."""
    if what == "longbench":
        rid = registration.registration_id(model, "longbench")
        if rid not in registration.CUT:
            return None
        if any(c["kind"] == "longbench" for c in run.t.cells_of(model)):
            raise Refused(f"{model}: a LongBench cell is in the tree but {rid} is cut (registration.CUT): a cut cell "
                          "never runs")
        return NOT_RUN_WHAT[what]
    d = ruler_declared(run, model)
    if d is None:
        return None
    ran = {"ruler_all": set(ALL_RULER_TASKS) <= d["tasks"], "qfirst": d["qfirst"], "swiftkv": d["swiftkv"],
           "splices": {"kva+kexact", "kva+vexact"} <= d["arms"], "qblind": set(QBLIND) <= d["arms"]}[what]
    return None if ran else NOT_RUN_WHAT[what]


def longbench_arms(run, model):
    """The arm kinds the model's LongBench cell declares (FINAL-LLAMA-LONGBENCH-2: the primary arms only, no SwiftKV
    run); every arm when there is no LongBench cell (then not_run or a missing cell decides)."""
    cs = [c for c in run.t.cells_of(model, "battery") if c["kind"] == "longbench"]
    if len(cs) != 1:
        return set(PAPER_ARMS)
    c = cs[0]
    names = [a for r in c["runs"] for a in r["arms"]] if "runs" in c else c.get("arms", [])
    return {run.t.canon(kvabackend.PAPER.get(a, a), "battery") for a in names}


PAPER_ARMS = [a for _, a, _ in BAT_A]


def longbench_note(run, model):
    """DEV-11: the panel-A arms a model's LongBench cell does not run (beyond DEV-8's controls), by name."""
    if not_run(run, model, "longbench"):
        return None
    gone = [lab for lab, a, _ in BAT_A if a not in LONGBENCH_NOT_RUN and a not in longbench_arms(run, model)
            and not (lab in QFIRST_ROWS)]
    cs = [c for c in run.t.cells_of(model, "battery") if c["kind"] == "longbench"]
    if not gone or not cs:
        return None
    return (rf"{{\footnotesize Not run on LongBench: {', '.join(gone)} (that run covers the primary arms "
            r"only).\par} % DEV-11")


def cut_note(run, model, whats):
    items = [w for w in (not_run(run, model, x) for x in whats) if w]
    return (rf"{{\footnotesize Not run: {'; '.join(items)}.\par}} % DEV-11") if items else None


def battery_panel_a(run, model, row, col):
    label, arm, split = BAT_A[row]
    split = PRIMARY if split == "P" else split
    st, sc = run.st, "battery"
    name = BAT_COLS[col]
    where = f"battery {model} A '{label}' {name}"
    src = f"{model} {arm}@{split}"
    if name in ("RULER all", "RULER filled"):
        if (name == "RULER all" and not_run(run, model, "ruler_all")) or (label in QFIRST_ROWS and not_run(
                run, model, "qfirst")) or (arm.startswith("swiftkv") and not_run(run, model, "swiftkv")):
            return "--"                                       # DEV-11
        m = ("qfirst_" if label in QFIRST_ROWS else "ruler_") + ("all" if name == "RULER all" else "filled")
        return run.cell(where, lambda: st.mean(sc, model, m, arm, split), lambda v: fplain(v, 1), "locked-final", src)
    if name == "LongBench avg":
        if arm in LONGBENCH_NOT_RUN or not_run(run, model, "longbench") or arm not in longbench_arms(run, model):
            return "--"                                       # DEV-8
        return run.cell(where, lambda: st.mean(sc, model, "longbench", arm, split), lambda v: fplain(v, 1),
                        "locked-final", src)
    if name in ("dNLL novel", "dNLL key"):
        m = "nll_novel" if name == "dNLL novel" else "nll_key"
        return run.cell(where, lambda: st.mean(sc, model, m, arm, split, vs_exact=True), lambda v: fsigned(v, 3),
                        "locked-final", src)
    if name == "Decode agree":
        return run.cell(where, lambda: st.mean(sc, model, "agree", arm, split), lambda v: fpct(v), "locked-final", src)
    if name == "Decode KL":
        return run.cell(where, lambda: st.mean(sc, model, "kl", arm, split), lambda v: fplain(v, 3), "locked-final", src)
    if name == "Turn-2 RULER":
        return run.cell(where, lambda: st.mean(sc, model, "turn2", arm, split), lambda v: fplain(v, 1),
                        "locked-final", src)
    if name == "TTFT speedup":
        return "--"                                           # DEV-8
    raise AssertionError(name)


def battery_panel_b(run, model, which, row, cells):
    label, mlabel, a, sa, b, sb, metric, metric_b = CONTRASTS[which][row]
    layout(cells[:2] == [label, LBL[mlabel]], f"battery {model} panel B row {row + 1}", cells[:2], [label, LBL[mlabel]])
    st = run.st
    where = f"battery {model} B row {row + 1} ({a}@{sa} - {b}@{sb}, {metric})"
    src = f"{model} {metric}: {a}@{sa} - {b}@{sb}"
    nd = METRIC_ND[metric]
    get = lambda: st.contrast("battery", model, metric, (a, sa), (b, sb), name_b=metric_b)      # noqa: E731
    out = list(cells)
    if contrast_not_run(run, model, metric):
        out[2:5] = ["--"] * 3                                 # DEV-11
        return out
    out[2] = run.cell(where + " n", lambda: get()["n"], str, "locked-final", src)
    out[3] = run.cell(where + " delta", lambda: get()["est"], lambda v: fsigned(v, nd), "locked-final", src)
    out[4] = run.cell(where + " CI", lambda: (get()["lo"], get()["hi"]), lambda v: fci(*v, nd), "locked-final", src)
    if out[5] == TBD:
        out[5] = run.cell(where + " MDE", lambda: get()["mde"], lambda v: fplain(v, nd), "locked-final", src)
    if a == "swiftkv":
        out[-1] = "ablation"                                  # DEV-13: not the R6 comparison (swiftkv-native is)
    elif a == "swiftkv-native":
        pass                                                  # DEV-13: the SwiftKV comparison keeps the template's role
    return out


def contrast_not_run(run, model, metric):
    return (metric == "longbench" and not_run(run, model, "longbench")) or (
        metric.startswith("qfirst") and not_run(run, model, "qfirst"))


# FINAL-LLAMA-RULER-2 declares two readouts next to the pooled primary, outside every primary and Holm family (DEV-14):
# the 64K contrasts at S12 reported on their own (64K NIAH rows only), and the follow-up at the 1/2 split (S16, run
# ruler-niah-1-2, 8K-32K NIAH), declared 2026-10-07 after FINAL-LLAMA-NLL (post hoc).
# (label, A, B, subset, role, numbers.tex macro stem)
LLAMA_ON_ITS_OWN = [
    (rf"{M_} $-$ identity, 64K (reported on its own)", ("kva", P), ("noproj", P), "64k", "reported", "SixtyFourKDiff"),
    ("identity + rows $-$ exact, 64K (reported on its own)", ("noproj+qattn10", P), ("exact", "-"), "64k", "reported",
     "SixtyFourKIdRowsVsExact"),
    (rf"{M_} + rows $-$ exact, 64K (reported on its own)", ("kva+qattn10", P), ("exact", "-"), "64k", "reported",
     "SixtyFourKFillRowsVsExact"),
    (rf"{M_} $-$ identity at the 1/2 split, follow-up, declared 2026-10-07 (post hoc)",
     ("kva", R12), ("noproj", R12), None, "post hoc", "FollowupDiff"),
]


def llama_own_rows(run):
    model, st, lines = "llama-3.1-8b-instruct", run.st, []
    for label, a, b, subset, role, _ in LLAMA_ON_ITS_OWN:
        w = f"battery {model} B ({a[0]}@{a[1]} - {b[0]}@{b[1]}, ruler_filled{', ' + subset if subset else ''})"
        src = f"{model} ruler_filled: {a[0]}@{a[1]} - {b[0]}@{b[1]}" + (f" ({subset} rows)" if subset else "")
        get = lambda a=a, b=b, subset=subset: st.contrast("battery", model, "ruler_filled", a, b, subset=subset)  # noqa: E731
        cells = [label, LBL["ruler_filled"], run.cell(w + " n", lambda get=get: get()["n"], str, "locked-final", src),
                 run.cell(w + " delta", lambda get=get: get()["est"], lambda v: fsigned(v, 1), "locked-final", src),
                 run.cell(w + " CI", lambda get=get: _ci(get()), lambda v: fci(*v, 1), "locked-final", src), "--", role]
        lines.append(" & ".join(cells) + r" \\ % DEV-14")
    return lines


QBLIND_ROWS = [("Identity + rows, question-blind", "noproj+qblind10", "identity + rows", "noproj+qattn10", "Id"),
               (r"\method + rows, question-blind", "kva+qblind10", r"\method + rows", "kva+qattn10", "Fill")]
# the question-blind rows (qblind10, FINAL-LLAMA-RULER-2 / FINAL-QWEN35-RULER-2: "reported next to the question-attended
# rows (scope)"): numbers.tex macro prefix per model that declares them
QBLIND_MACROS = {"llama-3.1-8b-instruct": "resLlama", "qwen3.5-9b": "resQwen"}


def qblind_rows(run, model, panel):
    """DEV-12: the question-blind pair's rows, after the question-first rows (panel A: filled-region accuracy) and
    after the question-first contrast (panel B: blind minus the question-attended rows), for a model that runs it."""
    if not_run(run, model, "qblind"):
        return []
    st, lines = run.st, []
    for label, arm, rlabel, rarm, _ in QBLIND_ROWS:
        src = f"{model} ruler_filled: {arm}@{P}"
        if panel == 0:
            v = run.cell(f"battery {model} A '{label}' RULER filled", lambda arm=arm: st.mean(
                "battery", model, "ruler_filled", arm, P), lambda v: fplain(v, 1), "locked-final", src)
            cells = [label, "--", v] + ["--"] * 7
        else:
            w = f"battery {model} B ({arm}@{P} - {rarm}@{P}, ruler_filled)"
            get = lambda arm=arm, rarm=rarm: st.contrast("battery", model, "ruler_filled", (arm, P), (rarm, P))  # noqa: E731
            cells = [f"{label[0].lower()}{label[1:]} $-$ {rlabel}", LBL["ruler_filled"],
                     run.cell(w + " n", lambda get=get: get()["n"], str, "locked-final", src),
                     run.cell(w + " delta", lambda get=get: get()["est"], lambda v: fsigned(v, 1), "locked-final", src),
                     run.cell(w + " CI", lambda get=get: _ci(get()), lambda v: fci(*v, 1), "locked-final", src),
                     "--", "scope"]
        lines.append(" & ".join(cells) + r" \\ % DEV-12")
    return lines


# DEV-15: matched-depth retrieval vs POP as published (FINAL-LLAMA-RULER-DEEP, FINAL-QWEN3-8B-RULER-DEEP; follow-up
# declared 2026-10-08, post hoc): the ruler-deep cell's own rows (set ruler-deep), outside every primary / Holm / MDE
# family. The non-inferiority margin is the one those records REGISTER in their predictions ("one-sided
# non-inferiority at a -3 point margin": at about 138 eligible rows the binomial floor makes -2 unattainable), not the
# cells' generic margins.ruler_points_lower; tests/test_finals_tables.py checks it against the real ledger.
DEEP = {"llama-3.1-8b-instruct": "resLlama", "qwen3-8b": "resQwenDense"}
DEEP_SPLIT = "5/8"
RULER_DEEP_MARGIN = -3.0
DEEP_ARMS = [("exact", "-", "Exact"), ("noproj", DEEP_SPLIT, "Id"), ("kva", DEEP_SPLIT, "Fill"), ("pop-native", "pop", "Pop")]
DEEP_LABEL = {"exact": "exact", "noproj": "identity at the 5/8 split", "kva": r"\method at the 5/8 split",
              "pop-native": "POP native (its own split)"}


def deep_cell(run, model):
    cs = [c for c in run.t.cells_of(model, "battery") if c["kind"] == "ruler-deep"]
    return cs[0] if len(cs) == 1 else None


def deep_noninf(run, model):
    return run.st.noninferiority("battery", model, "deep_filled", ("kva", DEEP_SPLIT), ("pop-native", "pop"),
                                 lower=RULER_DEEP_MARGIN, unit_range=RULER_UNIT_RANGE)


def deep_vs_identity(run, model):
    return run.st.contrast("battery", model, "deep_filled", ("kva", DEEP_SPLIT), ("noproj", DEEP_SPLIT))


def deep_panel(run, model):
    if model not in DEEP or deep_cell(run, model) is None:
        return []
    st, src, lf = run.st, f"{model} ruler-deep filled", "locked-final"
    w = f"battery {model} matched-depth follow-up"
    lines = [r"\medskip", r"\begin{tabular}{@{}lrrrll@{}}", r"\toprule",
             r"Matched depth vs POP as published (follow-up, declared 2026-10-08, post hoc) & $n$ & Value & "
             r"Bound / 95\% CI & Method & Verdict \\", r"\midrule"]
    for arm, split, _ in DEEP_ARMS:
        get = lambda arm=arm, split=split: st.vals("battery", model, "deep_filled", arm, split)  # noqa: E731
        lines.append(" & ".join([f"{DEEP_LABEL[arm]}, RULER filled",
                                 run.cell(f"{w} {arm} n", lambda get=get: len(get()), str, lf, src),
                                 run.cell(f"{w} {arm} mean", lambda get=get: float(np.mean(list(get().values()))),
                                          lambda v: fplain(v, 1), lf, src), "--", "--", "--"]) + r" \\")
    ni = lambda: deep_noninf(run, model)                                     # noqa: E731
    lines.append(" & ".join([rf"\method at 5/8 $-$ POP native, non-inferiority $\ge$ {fsigned(RULER_DEEP_MARGIN, 1)}",
                             run.cell(w + " NI n", lambda: ni().n, str, lf, src),
                             run.cell(w + " NI delta", lambda: ni().estimate, lambda v: fsigned(v, 1), lf, src),
                             run.cell(w + " NI bound", lambda: ni().lo, lambda v: fsigned(v, 2), lf, src),
                             run.cell(w + " NI method", lambda: METHOD_LABEL[ni().method], str, lf, src),
                             run.cell(w + " NI verdict", lambda: noninf_verdict(ni()), str, lf, src)]) + r" \\")
    c = lambda: deep_vs_identity(run, model)                                 # noqa: E731
    lines.append(" & ".join([r"\method $-$ identity at 5/8",
                             run.cell(w + " fill-id n", lambda: c()["n"], str, lf, src),
                             run.cell(w + " fill-id delta", lambda: c()["est"], lambda v: fsigned(v, 1), lf, src),
                             run.cell(w + " fill-id CI", lambda: _ci(c()), lambda v: fci(*v, 1), lf, src), "BCa",
                             run.cell(w + " fill-id verdict", lambda: "met" if c()["lo"] > 0 else "not met", str, lf,
                                      src)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return [ln + " % DEV-15" for ln in lines]


# DEV-16: the fresh-data confirmation (FINAL-LLAMA-NLL-FRESH, declared 2026-10-08 before any run on its set): the
# nll-fresh cell's own scope (dataset:<name>: never pooled with the locked set), novel-token NLL; (a) ridgefill -
# identity at 1/2 and at 5/8 with Holm over the two, (b) ridgefill at 5/8 - pop-native; key-token NLL reported.
FRESH = {"llama-3.1-8b-instruct": "resLlama"}
FRESH_ROWS = [  # (label, metric, A, B, macro stem, role: holm / test / reported)
    (r"\method $-$ identity at the 1/2 split", "nll_novel", ("kva", "1/2"), ("noproj", "1/2"), "FreshHalf", "holm"),
    (r"\method $-$ identity at the 5/8 split", "nll_novel", ("kva", "5/8"), ("noproj", "5/8"), "FreshFiveEighths", "holm"),
    (r"\method at 5/8 $-$ POP native", "nll_novel", ("kva", "5/8"), ("pop-native", "pop"), "FreshPop", "test"),
    (r"\method $-$ identity at the 1/2 split", "nll_key", ("kva", "1/2"), ("noproj", "1/2"), "FreshKeyHalf", "reported"),
    (r"\method $-$ identity at the 5/8 split", "nll_key", ("kva", "5/8"), ("noproj", "5/8"), "FreshKeyFiveEighths",
     "reported"),
]


def fresh_scope(run, model):
    cs = [c for c in run.t.cells_of(model) if c["kind"] == "nll-fresh"]
    return (scope_of(cs[0]), cs[0]["dataset"]) if len(cs) == 1 else (None, None)


def fresh_holm(run, model):
    scope, _ = fresh_scope(run, model)
    ps = {stem: run.st.contrast(scope, model, met, a, b)["p"] for _, met, a, b, stem, role in FRESH_ROWS if role == "holm"}
    adj = S.holm(ps)
    return {k: (v, ps[k] <= 1.0 / S.N_RESAMPLES) for k, v in adj.items()}


def fresh_panel(run, model):
    scope, dataset = fresh_scope(run, model)
    if model not in FRESH or scope is None:
        return []
    st, tier = run.st, dataset
    lines = [r"\medskip", r"\begin{tabular}{@{}llrrcll@{}}", r"\toprule",
             r"Fresh-data confirmation (fresh documents; follow-up, declared 2026-10-08) & Metric & $n$ & $\Delta$ & 95\% CI & "
             r"Holm & Verdict \\",
             r"\midrule"]
    for label, met, a, b, stem, role in FRESH_ROWS:
        w = f"fresh {model} {a[0]}@{a[1]} - {b[0]}@{b[1]} {met}"
        src = f"{dataset} {model} {met}: {a[0]}@{a[1]} - {b[0]}@{b[1]}"
        get = lambda met=met, a=a, b=b: st.contrast(scope, model, met, a, b)  # noqa: E731
        holm = (run.cell(w + " Holm", lambda stem=stem: fresh_holm(run, model)[stem], lambda v: fmt_p(*v), tier, src)
                if role == "holm" else "--")
        verdict = (run.cell(w + " verdict", lambda get=get: "met" if get()["hi"] < 0 else "not met", str, tier, src)
                   if role != "reported" else "reported")
        lines.append(" & ".join([label, LBL[met], run.cell(w + " n", lambda get=get: get()["n"], str, tier, src),
                                 run.cell(w + " delta", lambda get=get: get()["est"], lambda v: fsigned(v, 3), tier, src),
                                 run.cell(w + " CI", lambda get=get: _ci(get()), lambda v: fci(*v, 3), tier, src),
                                 holm, verdict]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return [ln + " % DEV-16" for ln in lines]


def ruler_margin(run, model):
    """The RULER non-inferiority margin (points) the model's RULER cell declares (FINAL- record: margins)."""
    ms = {(c.get("margins") or {}).get("ruler_points_lower") for c in run.t.cells_of(model, "battery")
          if c["kind"] == "ruler"}
    if len(ms) != 1 or None in ms:
        raise Missing(f"{model}: the RULER cell declares no single margins.ruler_points_lower (found {sorted(map(str, ms))})")
    return float(ms.pop())


def c3_noninf(run, model, arm):
    """C3: rows arm minus exact on the pooled filled-region RULER accuracy, per-row paired deltas in points, against the
    cell's declared lower margin, with the bounded-score ceiling guard (n: the rows that carry both arms, filled)."""
    return run.st.noninferiority("battery", model, "ruler_filled", (arm, P), ("exact", "-"),
                                 lower=ruler_margin(run, model), unit_range=RULER_UNIT_RANGE)


def noninf_verdict(t):
    if t.passed:
        return "met"
    return "not met (degenerate)" if t.reason == S.DEGENERATE else "not met"


def holm_roles(run, model, which, rows):
    """Holm-adjusted p for the table's family (rows whose Role cell is 'Holm')."""
    ps = {}
    for i in rows:
        a, sa, b, sb, metric, metric_b = CONTRASTS[which][i][2:]
        if contrast_not_run(run, model, metric):
            continue                                          # not run: outside the family (DEV-11)
        try:
            ps[i] = run.st.contrast("battery", model, metric, (a, sa), (b, sb), name_b=metric_b)["p"]
        except (Missing, S.StatsError):
            pass                                              # already collected by the panel cell
    adj = S.holm({str(k): v for k, v in ps.items()}) if ps else {}
    return {k: (v, ps[int(k)] <= 1.0 / S.N_RESAMPLES) for k, v in adj.items()}


def fmt_p(p, floored=False):
    """Holm-adjusted p; floored: the raw p sat at the bootstrap's resolution floor (1 / n_resamples), so the
    printed value is an upper bound."""
    if p < 0.001:
        return r"$p<0.001$"
    return f"$p\\le {p:.3f}$" if floored else f"$p={p:.3f}$"


# ============================================================================================ the table fillers
class Filler:
    """Fills one template: cell(panel, row, cells) -> cells; rows(panel) -> [lines] for longtables;
    caption() -> the next caption value."""

    def __init__(self, run):
        self.run = run
        self.caps = []

    def caption(self):
        if not self.caps:
            raise AssertionError("more caption slots than values")
        return self.caps.pop(0)


class BatteryFiller(Filler):
    def __init__(self, run, key):
        super().__init__(run)
        self.model = REF_MODELS[key]
        self.which = "dense" if key == "dense" else "other"
        t = run.t
        cells = [c for c in t.cells_of(self.model, "battery") if c["kind"] not in FOLLOWUP_SETS]   # follow-ups: own split
        if not cells:
            run.missing.append((f"battery {self.model}", "no battery cell for this model in the tree"))
        decl = cells[0] if cells else {}
        splits = {c["split"] for c in cells}
        if len(splits) > 1 or (splits and splits != {PRIMARY}):
            run.missing.append((f"battery {self.model}", f"the battery cells declare splits {sorted(splits)}; the "
                                f"tables are built at the declared primary split {PRIMARY} (OUTLINE: 3/8 of depth)"))
        layer = lambda f: split_layer(run, self.model, f)                         # noqa: E731
        tail = decl.get("tail", TBD)
        if key == "qkn":
            cap = decl.get("max_len")
            nll_sk = run.t.skipped.get(("battery", self.model, "longppl"), {})
            nll_all = len(nll_sk) + len(run.t.data.get(("battery", self.model, "longppl"), {}))
            rl = sorted({r.get("length") for _, r in run.t.skipped.get(("battery", self.model, "ruler"), {}).values()})
            self.caps = [run.cell("battery-dense-qkn caption split", lambda: layer(decl["split"]), str, "declared"),
                         decl.get("split", TBD), str(tail),
                         run.cell("caption cap", lambda: _need(cap, "Qwen3-8B max_len"), lambda v: f"{v:,}".replace(",", "{,}"), "declared"),
                         run.cell("caption skipped RULER length", lambda: _need(rl, "skipped RULER lengths"),
                                  lambda v: ", ".join(f"{x // 1024}K" for x in v), "locked-final"),
                         run.cell("caption skipped docs", lambda: _need(len(nll_sk), "skipped documents"), str, "locked-final"),
                         run.cell("caption docs", lambda: _need(nll_all, "documents"), str, "locked-final")]
        else:
            self.caps = [run.cell(f"battery {key} caption split", lambda: layer(decl["split"]), str, "declared"),
                         decl.get("split", TBD), str(tail)]
        self.holm = None

    def cell(self, panel, row, cells):
        out = list(cells)
        if panel == 0:
            layout(cells[0] == BAT_A[row][0], f"battery {self.model} panel A row {row + 1}", cells[0], BAT_A[row][0])
            for c in range(1, len(cells)):
                if cells[c] == TBD:
                    out[c] = battery_panel_a(self.run, self.model, row, c - 1)
            return out
        out = battery_panel_b(self.run, self.model, self.which, row, cells)
        if cells[-1] == "Holm":
            if self.holm is None:
                rows = [i for i, c in enumerate(CONTRASTS[self.which]) if self.role(i) == "Holm"]
                self.holm = holm_roles(self.run, self.model, self.which, rows)
            if str(row) in self.holm:
                out[-1] = f"Holm, {fmt_p(*self.holm[str(row)])}"
        return out

    def role(self, i):
        return ROLE_COLUMN[self.which][i]

    QFIRST_CONTRAST = {"dense": rf"{M_} + rows $-$ identity + rows, q-blind",
                       "other": "identity + rows, q-blind $-$ identity + rows"}

    def after_row(self, panel, label):
        if (panel, label) in ((0, QFIRST_ROWS[-1]), (1, self.QFIRST_CONTRAST[self.which])):
            return qblind_rows(self.run, self.model, panel)
        if self.which == "dense" and panel == 1 and label == CONTRASTS["dense"][-1][0]:
            return llama_own_rows(self.run)                    # DEV-14, after the last declared contrast
        return []

    def after_panel(self, panel):
        if panel == 1:
            return deep_panel(self.run, self.model) + fresh_panel(self.run, self.model)    # DEV-15, DEV-16
        note = cut_note(self.run, self.model, ("longbench", "ruler_all", "qfirst")
                        + (("swiftkv",) if self.which == "dense" else ()))
        lb = longbench_note(self.run, self.model)
        return [x for x in (ruler_population_note(self.run, self.model), note, lb) if x]

    def noninf_panel(self):
        """DEV-9: the C3 non-inferiority panel (rows arms minus exact), under panel B of the tables whose FINAL- record
        declares it (C3_NONINF). Every line carries its DEV marker."""
        run, model = self.run, self.model
        head = r"Non-inferiority vs exact"
        if model in C3_POSTHOC:
            run.t.require_record(C3_POSTHOC[model], "hypothesis")
            head = r"Non-inferiority vs exact (follow-up, declared 2026-10-08, post hoc)"
        lines = [r"\medskip", r"\begin{tabular}{@{}llrrrrll@{}}", r"\toprule",
                 head + r" & Metric & $n$ & $\Delta$ & Margin & Bound & Method & Verdict \\",
                 r"\midrule"]
        for label, arm, _ in C3_ROWS:
            w = f"battery {model} C3 non-inferiority {arm}@{P} - exact"
            src = f"{model} ruler_filled: {arm}@{P} - exact@-"
            get = lambda arm=arm: c3_noninf(run, model, arm)                     # noqa: E731
            cells = [label, LBL["ruler_filled"],
                     run.cell(w + " n", lambda: get().n, str, "locked-final", src),
                     run.cell(w + " delta", lambda: get().estimate, lambda v: fsigned(v, 1), "locked-final", src),
                     run.cell(w + " margin", lambda: ruler_margin(run, model), lambda v: r"$\ge$ " + fsigned(v, 1),
                              "declared", f"{model} RULER cell margins.ruler_points_lower"),
                     run.cell(w + " bound", lambda: get().lo, lambda v: fsigned(v, 2), "locked-final", src),
                     run.cell(w + " method", lambda: METHOD_LABEL[get().method], str, "locked-final", src),
                     run.cell(w + " verdict", lambda: noninf_verdict(get()), str, "locked-final", src)]
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return [ln + " % DEV-9" for ln in lines]


def ruler_population_note(run, model):
    """DEV-24 (review 2026-10-11): the panel-A arms do not all run on the same RULER samples (the controls and the
    question-blind rows ran at 8K-32K only), so the note gives each group of arms its filled-region n and lengths."""
    arms = [(lab, a, PRIMARY if s_ == "P" else s_) for lab, a, s_ in BAT_A if lab not in QFIRST_ROWS]
    arms += [(lab, a, PRIMARY) for lab, a, *_ in QBLIND_ROWS]
    try:
        recs = run.t.records("battery", model, "ruler")
    except Missing:
        return None
    groups = {}
    for lab, arm, split in arms:
        try:
            v = run.st.vals("battery", model, "ruler_filled", arm, split)
        except Missing:
            continue                                          # not run (its cells print "--")
        lens = sorted({recs[u]["length"] for u in v})
        groups.setdefault((len(v), lens[0], lens[-1]), []).append(lab[0].lower() + lab[1:] if lab[1:2].islower() else lab)
    if len(groups) < 2:
        return None
    parts = [rf"{n} samples at {lo // 1024}K--{hi // 1024}K for {and_list(labs)}" for (n, lo, hi), labs in
             sorted(groups.items(), key=lambda kv: -kv[0][0])]
    run.cellslog.append({"table": run.table, "cell": f"battery {model} RULER populations", "value": "; ".join(parts),
                         "tier": "locked-final", "protocol": tier_protocol(run, "locked-final"),
                         "sources": f"{model} ruler_filled per arm"})
    return rf"{{\footnotesize RULER filled: {'; '.join(parts)}.\par}} % DEV-24"


def pending_note(tier):
    return rf"{{\footnotesize {PENDING_NOTE[tier]}: every \pending{{}} cell is filled when it has run.\par}} % PENDING"


def layout(ok, where, got, want):
    """The template and the generator's row specs must describe the same rows (a drift refuses the run)."""
    if not ok:
        raise Refused(f"LAYOUT DRIFT at {where}: the template has {got!r}, the generator's spec {want!r}")


def _need(v, what):
    if v in (None, [], ""):
        raise Missing(f"{what} not declared / not found")
    return v


def split_layer(run, model, frac):
    p = run.t.protocol["models"].get(model)
    if p is None:
        raise Missing(f"{model} is not in protocol.json")
    if frac == "pop":
        return _need(p.get("pop_split"), f"{model} pop_split")
    depths = run.t.protocol["depths"]
    f = frac_float(frac)
    if f not in depths:
        raise Missing(f"depth {frac} not in protocol.json depths {depths}")
    return p["splits"][depths.index(f)]


class SplitFiller(Filler):
    ORDER = [("llama-3.1-8b-instruct", f) for f in FRACS + ("pop",)] + \
            [("qwen3.5-9b", f) for f in FRACS + ("pop",)] + [("qwen3-8b", f) for f in FRACS + ("pop",)]

    def __init__(self, run):
        super().__init__(run)
        tails = {c["tail"] for c in run.t.cells.values() if c["kind"] in ("nll", "nll-curve-splices")}
        self.caps = [run.cell("splitcurve caption tail", lambda: _single(tails, "nll cell tails"), str, "declared")]

    def cell(self, panel, row, cells):
        model, frac = self.ORDER[row]
        run, st, out = self.run, self.run.st, list(cells)
        layout(cells[0] == ("POP native" if frac == "pop" else frac), f"splitcurve row {row + 1}", cells[0], frac)
        w = f"splitcurve {model} {frac}"
        out[1] = run.cell(w + " S", lambda: split_layer(run, model, frac), str, "declared")
        if frac == "pop":
            out[3] = run.cell(w + " POP native dNLL", lambda: st.mean("battery", model, "nll_novel", "pop-native", "pop", vs_exact=True),
                              lambda v: fsigned(v, 3), "locked-final", f"{model} pop-native")
            out[8] = "--"                                     # DEV-8: TTFT
            return out
        for c, arm in ((2, "kva"), (3, "noproj"), (4, "skip")):
            out[c] = run.cell(f"{w} dNLL {arm}", lambda arm=arm: st.mean("battery", model, "nll_novel", arm, frac, vs_exact=True),
                              lambda v: fsigned(v, 3), "locked-final", f"{model} {arm}@{frac}")
        out[5] = run.cell(w + " kva-noproj CI", lambda: _ci(st.contrast("battery", model, "nll_novel", ("kva", frac), ("noproj", frac))),
                          lambda v: fci(*v, 3), "locked-final", f"{model} kva - noproj @{frac}")
        if cells[6] == TBD:
            nxt = FRACS[FRACS.index(frac) + 1]
            out[6] = run.cell(w + " next-this CI", lambda: _ci(st.contrast("battery", model, "nll_novel", ("kva", nxt), ("kva", frac))),
                              lambda v: fci(*v, 3), "locked-final", f"{model} kva@{nxt} - kva@{frac}")
        out[7:10] = ["--"] * 3                                # DEV-8: TTFT
        return out


def _ci(c):
    return c["lo"], c["hi"]


def _single(vals, what):
    vals = set(vals)
    if len(vals) != 1:
        raise Missing(f"{what}: expected one declared value, found {sorted(vals) or 'none'}")
    return vals.pop()


MECH_ARMS = [("kva", None), ("kva+kexact", "keys"), ("kva+vexact", "values"), ("kvexact", "both"), ("kva+ktail", "ktail")]
MECH_MODELS = ("llama-3.1-8b-instruct", "qwen3.5-9b", "qwen3-8b")
MECH_LABEL = {"kva": r"\method", "kva+kexact": "+ exact keys", "kva+vexact": "+ exact values",
              "kvexact": "+ exact keys and values", "kva+ktail": "+ key scale (tail)"}
ROWS_LABEL = {"kva+qattn10": r"\method + rows"}


def loss_share(run, model, arm):
    """Share (%) of RidgeFill's NLL loss vs exact that the splice removes: 100 (1 - mean d_arm / mean d_kva), over the
    documents of the NLL cell (the splices run there, finals/make_cells.py)."""
    st = run.st
    return lambda: st.ratio("battery", model, "nll_novel", [(arm, P), ("kva", P), ("exact", "-")],
                            lambda m: 100.0 * (1.0 - (m[..., 0] - m[..., 2]) / (m[..., 1] - m[..., 2])),
                            why=lambda x: nll_loss_why(mech_who(model, arm), x[:, 1], x[:, 2]))


def mech_who(model, arm):
    return f"{SHORT[model]} {MECH_LABEL.get(arm, ROWS_LABEL.get(arm, tex_escape(arm)))}"


def nll_loss_why(who, base, exact):
    """The reason a share of RidgeFill's NLL loss has no interval: that loss's mean is zero on some resample."""
    return (rf"{who}: interval undefined: \method's NLL loss to exact (mean {fsigned(float(np.mean(base - exact)), 3)} nats "
            f"over {len(base)} documents) is zero on some resample")


def ruler_gap_why(who, base, exact):
    """The reason a share of the RULER gap closed has no interval: the gap to exact, in discordant rows."""
    k = int(np.count_nonzero(base != exact))
    return (rf"{who}: interval undefined: the gap to exact is {k} samples (the samples where exact and \method "
            f"differ, of {len(base)}), so some resamples have no gap")


def mech_rows(r):
    """The mechanism RULER subset: the rows the splices ran on (the RULER cells' NIAH rows at 8K-32K)."""
    return P in r["arms"].get("kva+kexact", {})


# DEV-7: the declared cells run the RULER splices kexact and vexact only (on the NIAH subset; the coordinator's
# alignment scope, 2026-10-04: OUTLINE C4's "exact keys alone restore retrieval" needs exact keys, and exact values are
# its control); kvexact and ktail run on the NLL documents only. Their RULER cells print "--" (not run by the declared
# design), as the template's convention says; every other mechanism cell is computed.
MECH_NOT_ON_RULER = ("kvexact", "kva+ktail")


def gap_closed(run, scope, model, arm, base="kva"):
    """Share (%) of base's filled-region RULER gap to exact that the arm closes."""
    st = run.st
    return lambda: st.ratio(scope, model, "ruler_filled", [(arm, P), (base, P), ("exact", "-")],
                            lambda m: 100.0 * (m[..., 0] - m[..., 1]) / (m[..., 2] - m[..., 1]),
                            why=lambda x: ruler_gap_why(mech_who(model, arm), x[:, 1], x[:, 2]))


class MechFiller(Filler):
    def cell(self, panel, row, cells):
        run, st, out = self.run, self.run.st, list(cells)
        if panel == 0:
            model, (arm, _) = MECH_MODELS[row // 5], MECH_ARMS[row % 5]
            layout(cells[0] == SHORT[model] and cells[1] == MECH_LABEL[arm], f"mechanism row {row + 1}", cells[:2],
                   [SHORT[model], MECH_LABEL[arm]])
            w = f"mechanism {model} {arm}"
            out[2] = run.cell(w + " dNLL", lambda: st.mean("battery", model, "nll_novel", arm, P, vs_exact=True),
                              lambda v: fsigned(v, 3), "locked-final", f"{model} nll {arm}")
            if arm in MECH_NOT_ON_RULER:
                out[5] = out[6] = "--"                       # DEV-7
            elif not_run(run, model, "splices"):
                out[5] = out[6] = "--"                       # DEV-11: no splice rows, no mechanism RULER subset
            else:
                out[5] = run.cell(w + " RULER filled", lambda: st.mean("battery", model, "ruler_filled", arm, P,
                                                                       where=mech_rows),
                                  lambda v: fplain(v, 1), "locked-final", f"{model} ruler NIAH subset {arm}")
            if arm != "kva":
                out[3] = run.cell(w + " loss removed", lambda: r_est(loss_share(run, model, arm)()),
                                  lambda v: fpct(v), "locked-final", f"{model} mechanism {arm} share")
                out[4] = run.cell(w + " loss removed CI", lambda: r_ci(loss_share(run, model, arm)()),
                                  lambda v: fci(*v, 1, pct=True, signed=False), "locked-final",
                                  f"{model} mechanism {arm} share")
                if arm not in MECH_NOT_ON_RULER and not not_run(run, model, "splices"):
                    out[6] = run.cell(w + " gap closed", lambda: r_est(gap_closed(run, "battery", model, arm)()),
                                      lambda v: fpct(v), "locked-final", f"{model} ruler NIAH subset {arm} gap")
            return out
        model = MECH_MODELS[row]
        layout(cells[0] == SHORT[model], f"mechanism panel B row {row + 1}", cells[0], SHORT[model])
        w = f"mechanism panel B {model}"
        out[3] = run.cell(w + " carried by", lambda: carried_by(run, model), str, "locked-final", f"{model} kexact vs vexact")
        out[4] = "--" if not_run(run, model, "splices") else run.cell(
            w + " exact keys restore", lambda: _verdict(gap_closed(run, "battery", model, "kva+kexact")()), str,
            "locked-final", f"{model} kexact RULER gap")
        if cells[5] in (TBD, "demonstration"):              # DEV-24: measured on the QK-normed models too
            out[5] = run.cell(w + " key scale", lambda: _verdict(loss_share(run, model, "kva+ktail")()), str,
                              "locked-final", f"{model} ktail share")
        out[6] = run.cell(w + " rows restore", lambda: _verdict(gap_closed(run, "battery", model, "kva+qattn10")()), str,
                          "locked-final", f"{model} rows RULER gap")
        return out


    def after_panel(self, panel):
        cut = [SHORT[m] for m in MECH_MODELS if panel == 0 and not_run(self.run, m, "splices")]
        out = [rf"{{\footnotesize Not run: {NOT_RUN_WHAT['splices']} on {', '.join(cut)}.\par}}"
               " % DEV-11"] if cut else []
        if panel == 0:                                        # DEV-24: the size of the mechanism RULER subset
            run, parts = self.run, []
            for m in MECH_MODELS:
                if not_run(run, m, "splices"):
                    continue
                n = run.cell(f"mechanism RULER subset n {m}", lambda m=m: len(run.st.vals(
                    "battery", m, "ruler_filled", "kva+kexact", P)), str, "locked-final", f"{m} ruler NIAH subset")
                parts.append(f"{n} on {SHORT[m]}")
            if parts:
                out.append(rf"{{\footnotesize RULER filled: the filled-region samples at 8K--32K on which the splices "
                           rf"ran ({and_list(parts)}).\par}} % DEV-24")
        return out


def _verdict(r):
    """A restoration verdict read from the share's interval; without an interval, "--" with the estimate."""
    if isinstance(r, UndefinedRatio):
        return Noted(f"-- ({fpct(r.estimate)})" if math.isfinite(r.estimate) else None, r.note)
    return share_verdict(r.lo, r.hi, r.estimate)


def carried_by(run, model):
    """keys / values from the CI of (share removed by exact keys - share removed by exact values)."""
    r = run.st.ratio("battery", model, "nll_novel", [("kva+kexact", P), ("kva+vexact", P), ("kva", P), ("exact", "-")],
                     lambda m: (m[..., 1] - m[..., 0]) / (m[..., 2] - m[..., 3]),
                     why=lambda x: nll_loss_why(f"{SHORT[model]} carried by (exact keys vs exact values)", x[:, 2],
                                                x[:, 3]))
    if isinstance(r, UndefinedRatio):
        return Noted(None, r.note)
    # DEV-24 (review 2026-10-11): "neither" when neither exact half alone removes a share whose interval lies above zero
    shares = [loss_share(run, model, a)() for a in ("kva+kexact", "kva+vexact")]
    if not any(not isinstance(x, UndefinedRatio) and x.lo > 0 for x in shares):
        return "neither"
    return "keys" if r.lo > 0 else "values" if r.hi < 0 else "undecided"


# ---------------------------------------------------------------------------------------------- Flash-Next
# The C5 tier as run (protocol 1.1.6; FINAL-FLASHNEXT-C5 with amendments 10 and 11; DEV-17: this table replaces the
# skeleton's three-panel layout, whose identity, random-rows, knob, all-rows, correction-only and decode-agreement
# cells were not run). Panel A: one row per arm; panel B: C5's criteria for quality mode. Arms are the engine cell's
# (finals/engine/qwen3.8-flash-next-c5.json); the perplexity unit is a locked long-prompt document (set longppl,
# novel-token NLL over the 512 scored tokens), the RULER unit a row, the TTFT unit a timed round of one arm's boot
# against that boot's exact path.
FN_ROWS = [("Exact", "exact"), ("Quality mode", "ridgefill+corr+prot{share}"), (r"\method", "ridgefill")]
FN_C_LABEL = ["Perplexity gap to exact", r"RULER filled $-$ exact", "TTFT speedup at 16K"]
FN_NOTE = (r"\par\smallskip{{\footnotesize Engine: \texttt{{{image}}} with the one-file determinism fix (protocol {protocol}). "
           r"Exact's two perplexity passes on one server start are bitwise identical on {det} of {docs} documents; "
           r"quality mode's server start reproduces exact's values bitwise on {ctl} of {ctl_n} documents served with "
           r"\method switched off.\par}} % DEV-17")


class FlashFiller(Filler):
    def __init__(self, run):
        super().__init__(run)
        self.pending = "flashnext" in run.pending
        cs = run.t.cells_of(FN_MODEL, "flashnext")
        self.decl = cs[0] if len(cs) == 1 else None
        if self.decl is None and not self.pending:
            run.missing.append(("flashnext", f"expected one flashnext cell for {FN_MODEL}, found {len(cs)}"))
        share = (self.decl or {}).get("declared_share")
        if self.decl is not None and share != 25:
            run.missing.append(("flashnext", f"the layout's quality mode is class protection at 25% but the FINAL record "
                                f"declares {share!r}"))
        self.share = 25
        if self.pending:
            self.caps = [run.pend(f"flashnext caption {i}") for i in range(8)]
            return
        q = self.arm("ridgefill+corr+prot{share}")
        c = lambda w, fn, fmt=str, tier="engine-final": run.cell(f"flashnext caption {w}", fn, fmt, tier)  # noqa: E731
        lvl = f"{int(S.LEVEL * 100)}\\%"
        self.caps = [c("share", lambda: _need(share, "declared_share"), lambda v: f"{v}\\%", "declared"),
                     c("docs", lambda: self.d_nll(q)["n"]), lvl,
                     c("RULER rows", lambda: len(run.t.records("flashnext", FN_MODEL, "ruler"))),
                     c("RULER filled rows", lambda: self.ruler_d(q)["n"]),
                     c("tail", lambda: _need((self.decl or {}).get("tail"), "the cell's tail"), str, "declared"),
                     c("rounds", lambda: _need(((self.decl or {}).get("data") or {}).get("ttft", {}).get("timed"),
                                               "the cell's timed TTFT rounds"), str, "declared"),
                     f"{100 * (1 - S.ALPHA):g}\\%"]

    def arm(self, a):
        return a.format(share=self.share)

    def margin(self, k):
        m = (self.decl or {}).get("margins") or {}
        if k not in m:
            raise Missing(f"flashnext FINAL record declares no margin {k!r}")
        return m[k]

    def criterion(self, k):
        m = (self.decl or {}).get("criteria") or {}
        if k not in m:
            raise Missing(f"the flashnext cell carries no criterion {k!r} (amendment 11)")
        return m[k]

    def lengths(self):
        got = tuple(((self.decl or {}).get("data") or {}).get("ttft", {}).get("lengths") or ())
        if got != FN_TTFT_LENS:
            raise Missing(f"the flashnext cell declares TTFT lengths {got}; the layout's columns are {FN_TTFT_LENS}")
        return got

    def d_nll(self, arm, ref="exact"):
        return self.run.st.contrast("flashnext", FN_MODEL, "nll_novel", (arm, "-"), (ref, "-"))

    def gap(self, arm, k="est"):
        """Perplexity gap to exact in percent: 100 (exp(mean paired novel-token NLL difference) - 1)."""
        return 100.0 * math.expm1(self.d_nll(arm)[k])

    def ruler_d(self, arm):
        return self.run.st.contrast("flashnext", FN_MODEL, "ruler_filled", (arm, "-"), ("exact", "-"))

    def speedup(self, arm, length):
        return self.run.st.ttft_speedup("flashnext", FN_MODEL, arm, "-", length)

    def ppl_met(self, arm):
        """C5 (a), amendment 11: the measured gap, est of the paired NLL difference, at most log(1 + margin)."""
        return self.d_nll(arm)["est"] <= math.log1p(self.criterion("ppl_gap_upper_pct") / 100.0)

    def ruler_noninf(self, arm):
        """C5 (b): RULER non-inferiority vs exact: per-row filled-region score deltas in points, bounded by
        RULER_UNIT_RANGE (the ceiling guard: BCa or the exact binomial floor, whichever is more conservative)."""
        return self.run.st.noninferiority("flashnext", FN_MODEL, "ruler_filled", (arm, "-"), ("exact", "-"),
                                          lower=self.margin("ruler_points_lower"), unit_range=RULER_UNIT_RANGE)

    def speed_met(self, arm):
        """C5 (c): TTFT speedup at the registered length at least the registered minimum."""
        return self.speedup(arm, self.margin("ttft_length")) >= self.margin("ttft_min_speedup")

    def long_speedup(self, arm):
        """The 64K RULER rows' request time: exact's median wall time over the arm's (prefill of the prompt and a short
        answer, one request per row)."""
        long = lambda r: r.get("length") == 65536                                  # noqa: E731
        ex = self.run.t.values("flashnext", FN_MODEL, "ruler", "exact", "-", "wall_s", where=long)
        ar = self.run.t.values("flashnext", FN_MODEL, "ruler", arm, "-", "wall_s", where=long)
        if set(ex) != set(ar):
            raise Missing(f"flashnext 64K rows: exact and {arm} ran different rows")
        return statistics.median(ex.values()) / statistics.median(ar.values())

    def engine(self):
        """engine.json of the converted cell (finals/convert_c5.py): the determinism read and the exact-path control."""
        d = self.run.t.cell_dirs.get((self.decl or {}).get("name"))
        p = d / "engine.json" if d else None
        if p is None or not p.is_file():
            raise Missing("the flashnext cell has no engine.json (finals/convert_c5.py)")
        e = json.loads(p.read_text())
        det = e.get("determinism") or {}
        if not det.get("docs") or det.get("bitwise_equal") != det.get("docs"):
            raise Missing(f"determinism read {det.get('bitwise_equal')}/{det.get('docs')}: not every document bitwise equal")
        return e

    def cell(self, panel, row, cells):
        run, st, out = self.run, self.run.st, list(cells)
        want = FN_ROWS[row][0] if panel == 0 else FN_C_LABEL[row]
        layout(cells[0] == want, f"flashnext panel {panel} row {row + 1}", cells[0], want)
        if self.pending:                                      # --pending flashnext: every value slot, never computed
            return [c.replace(TBD, run.pend(f"flashnext panel {panel} row {row + 1} col {i}")) if TBD in c else c
                    for i, c in enumerate(out)]
        if panel == 0:
            arm = self.arm(FN_ROWS[row][1])
            w = f"flashnext A {arm}"
            if arm != "exact":
                out[1] = run.cell(w + " ppl gap", lambda: self.gap(arm), lambda v: fpct(v, 2, True), "engine-final", arm)
                out[2] = run.cell(w + " ppl gap CI", lambda: (self.gap(arm, "lo"), self.gap(arm, "hi")),
                                  lambda v: fci(*v, 2, pct=True), "engine-final", arm)
            out[3] = run.cell(w + " RULER all", lambda: st.mean("flashnext", FN_MODEL, "ruler_all", arm, "-"),
                              lambda v: fplain(v, 1), "engine-final", arm)
            out[4] = run.cell(w + " RULER filled", lambda: st.mean("flashnext", FN_MODEL, "ruler_filled", arm, "-"),
                              lambda v: fplain(v, 1), "engine-final", arm)
            if arm != "exact":
                for c, i in zip((5, 6, 7), range(3)):
                    out[c] = run.cell(f"{w} TTFT {FN_TTFT_LENS[i]}", lambda i=i: self.speedup(arm, self.lengths()[i]),
                                      lambda v: fplain(v, 2), "engine-final", arm)
                out[8] = run.cell(f"{w} 64K request", lambda: self.long_speedup(arm), lambda v: fplain(v, 2),
                                  "engine-final", f"{arm}: 64K RULER rows, median wall_s")
            return out
        q = self.arm("ridgefill+corr+prot{share}")
        w = f"flashnext C5 {['ppl', 'ruler', 'speed'][row]}"
        verdict = lambda ok: "met" if ok else "not met"                            # noqa: E731
        if row == 0:
            out[1] = out[1].replace(TBD, run.cell(w + " margin", lambda: self.criterion("ppl_gap_upper_pct"),
                                                  lambda v: fpct(v, 1, signed=True), "declared"))
            out[2] = run.cell(w + " estimate", lambda: self.gap(q), lambda v: fpct(v, 2, True), "engine-final", q)
            out[4] = run.cell(w + " verdict", lambda: verdict(self.ppl_met(q)), str, "engine-final", q)
        elif row == 1:
            out[1] = out[1].replace(TBD, run.cell(w + " margin", lambda: self.margin("ruler_points_lower"),
                                                  lambda v: fsigned(v, 1), "declared"))
            out[2] = run.cell(w + " estimate", lambda: self.ruler_d(q)["est"], lambda v: fsigned(v, 1), "engine-final", q)
            out[3] = run.cell(w + " bound", lambda: self.ruler_noninf(q).lo, lambda v: fsigned(v, 2), "engine-final", q)
            out[4] = run.cell(w + " verdict", lambda: self.verdict(self.ruler_noninf(q)), str, "engine-final", q)
        else:
            def sp16():
                if self.margin("ttft_length") != 16384:
                    raise Missing(f"the registered TTFT length is {self.margin('ttft_length')}; the row says 16K")
                return self.speedup(q, 16384)
            out[1] = out[1].replace(TBD, run.cell(w + " margin", lambda: self.margin("ttft_min_speedup"),
                                                  lambda v: f"{fplain(v, 2)}\\speedx", "declared"))
            out[2] = run.cell(w + " estimate", sp16, lambda v: f"{fplain(v, 2)}\\speedx", "engine-final", q)
            out[4] = run.cell(w + " verdict", lambda: verdict(self.speed_met(q)), str, "engine-final", q)
        return out

    def after_panel(self, panel):
        if self.pending:
            return [pending_note("flashnext")] if panel == 1 else []
        if panel != 1:
            return []
        e = self.run.cell("flashnext note: engine and determinism read", self.engine, lambda v: "ok", "engine-final")
        if e != "ok":
            return []
        e = self.engine()
        ctl = e.get("control") or {}
        return [FN_NOTE.format(image=tex_escape(e["image"]), protocol=e["protocol"],
                               det=e["determinism"]["bitwise_equal"], docs=e["determinism"]["docs"],
                               ctl=ctl.get("bitwise_equal", "--"), ctl_n=ctl.get("docs", "--"))]

    @staticmethod
    def verdict(t):
        """The non-inferiority verdict; a degenerate interval never passes (stats.noninferiority)."""
        if not t.passed:
            return "not met (degenerate)" if t.reason == S.DEGENERATE else "not met"
        return "met"


# ---------------------------------------------------------------------------------------------- negative, MDE
# Rows 3-6 of tables/negative.tex: development evidence, read from the ledger (DECLARED here; the set and protocol
# labels are declarations about each record's provenance, cited in the comment, to be confirmed by Dylan). Every
# record holds [estimate, BCa lo, BCa hi] computed with finals/stats.py from the per-sample files its sources pin, so
# no row of the table carries another interval method (row 4 read labbook's percentile comparison, seq 303, until
# 2026-10-04).
ENGINE_PREFIX = None        # set label = protocol.json engine_tier.prefix_label (v1.1.1: pre-fix engine numbers)
NEGATIVE_PREFINAL = [
    # Flash-Next calibration data size (2026-10-04 engine runs, before the deterministic top-k fix, protocol 1.0.2 era)
    dict(row=2, kind="record", label="neg-datasize", set=ENGINE_PREFIX, protocol="v1.0.2", nd=3),
    # Q1b (P4 sweep): +st - no st, Qwen3.5 S12 T4096, graded set helps16, 2026-10-01 (before v1.0)
    dict(row=3, kind="record", label="neg-q1b", protocol="pre-protocol", nd=3),
    # WF1 (mk4 needles; FQ3/WF1 configs ran under 1.0.1): needs a record [est, lo, hi]
    dict(row=4, kind="record", label="neg-wf1", protocol="v1.0.1", nd=3),
    # FQ3 (refresh19 vs equal-MAC qattn rows, grade; hypothesis says protocol 1.0.1): needs a record [est, lo, hi]
    dict(row=5, kind="record", label="neg-fq3", protocol="v1.0.1", nd=3),
]


def prefinal_values(run, spec):
    rec = run.t.ledger_by_label(spec["label"], "record")
    v = rec.get("values") or []
    if len(v) != 3:
        raise Missing(f"ledger record {spec['label']!r} must hold values [estimate, lo, hi], has {v}")
    return tuple(float(x) for x in v)


NEG_LABEL = [r"Hybrid wash: \method ties identity", "SwiftKV's trained maps beat the ridge fit",
             "Calibration data size is flat", "GDN terminal-state correction does nothing",
             "Weighted projector fit loses needles", "Refresh layer loses to rows at equal compute"]


class NegFiller(Filler):
    def cell(self, panel, row, cells):
        run, st, out = self.run, self.run.st, list(cells)
        layout(cells[0] == NEG_LABEL[row], f"negative row {row + 1}", cells[0], NEG_LABEL[row])
        w = f"negative row {row + 1}"
        if row < 2:
            model, arms = ("qwen3.5-9b", (("kva", P), ("noproj", P))) if row == 0 else \
                ("llama-3.1-8b-instruct", (("swiftkv", R12), ("kva", R12)))
            out[2] = out[2].replace(r"v\TBDcell", "v" + "/".join(locked_versions(run.t)))      # DEV-25
            c = lambda: st.contrast("battery", model, "nll_novel", *arms)      # noqa: E731
            out[4] = run.cell(w + " estimate", lambda: c()["est"], lambda v: fsigned(v, 3), "locked-final", model)
            out[5] = run.cell(w + " CI", lambda: _ci(c()), lambda v: fci(*v, 3), "locked-final", model)
            return out
        spec = next(s for s in NEGATIVE_PREFINAL if s["row"] == row)
        tier = f"dev:{spec['protocol']}"
        if out[2] == TBD:
            set_ = engine_prefix_label(run) if spec["set"] is ENGINE_PREFIX else spec["set"]
            out[2] = f"{set_}, {spec['protocol']}"
        else:
            out[2] = out[2].replace(TBD, spec["protocol"])
        out[4] = run.cell(w + " estimate", lambda: prefinal_values(run, spec)[0], lambda v: fsigned(v, spec["nd"]),
                          tier, spec["label"])
        out[5] = run.cell(w + " CI", lambda: prefinal_values(run, spec)[1:], lambda v: fci(*v, spec["nd"]), tier,
                          spec["label"])
        if out[6] == TBD:
            out[6] = run.cell(w + " evidence", lambda: run.t.ledger_by_label(spec["label"]) and neg_evidence(row),
                              str, tier, spec["label"])                      # DEV-25: the lever that holds the row
        return out


def tex_label(s):
    """Declared label text for a table cell: TeX-escaped, '+-' as a plus-minus sign."""
    return tex_escape(s).replace("+-", r"$\pm$")


def engine_prefix_label(run):
    """protocol.json's label for pre-fix engine numbers. TeX never hyphenates a word that touches an explicit hyphen,
    and "deterministic-" alone is wider than the negative table's "Set, protocol" column, so every hyphen is set
    between a penalty and a zero skip: the line may break after it and the words on both sides can hyphenate."""
    tier = run.t.protocol.get("engine_tier") or {}
    label = tex_label(_need(tier.get("prefix_label"), "protocol.json engine_tier.prefix_label (protocol 1.1.1)"))
    return label.replace("-", r"\nobreak\hspace{0pt}-\hspace{0pt}")


MDE_ROWS = [(m, c) for m in MECH_MODELS for c in range(3)] + [("fn", "ppl"), ("fn", "ruler"), ("cell", None)]


class MdeFiller(Filler):
    def __init__(self, run):
        super().__init__(run)
        self.caps = [f"{int(S.POWER * 100)}\\% power, two-sided {int(S.ALPHA * 100)}\\%"]

    def cell(self, panel, row, cells):
        run, st, out = self.run, self.run.st, list(cells)
        model, c = MDE_ROWS[row]
        want = SHORT[FN_MODEL] if model == "fn" else SHORT["llama-3.1-8b-instruct"] if model == "cell" else SHORT[model]
        layout(cells[0] == want, f"mde row {row + 1}", cells[0], want)
        w = f"mde row {row + 1}"
        if model == "fn" and "flashnext" in run.pending:
            out[3:6] = [run.pend(f"{w} {k}") for k in ("n", "SD", "MDE")]
            return out
        if model == "fn":
            if c == "ppl":
                get = lambda: _ppl_pct_mde(st, run.fillers["flashnext"].arm("ridgefill+corr+prot{share}"))  # noqa: E731
                nd = 2
            else:
                q = run.fillers["flashnext"].arm("ridgefill+corr+prot{share}")
                get = lambda: st.contrast("flashnext", FN_MODEL, "ruler_filled", (q, "-"), ("exact", "-"))  # noqa: E731
                nd = 1
        elif model == "cell":
            get, nd = (lambda: _single_cell_mde(run)), 1
        else:
            a, sa, b, sb, metric, _ = CONTRASTS["dense" if model.startswith("llama") else "other"][c][2:]
            get = lambda: st.contrast("battery", model, metric, (a, sa), (b, sb))   # noqa: E731
            nd = METRIC_ND[metric]
        out[3] = run.cell(w + " n", lambda: get()["n"], str, "locked-final")
        out[4] = run.cell(w + " SD", lambda: get()["sd"], lambda v: fplain(v, nd), "locked-final")
        out[5] = run.cell(w + " MDE", lambda: get()["mde"], lambda v: fplain(v, nd), "locked-final")
        return out


    def after_panel(self, panel):
        return [pending_note("flashnext")] if "flashnext" in self.run.pending else []


def _ppl_pct_mde(st, q):
    """Quality mode vs exact as a per-document perplexity ratio change in percent: 100 (exp(d) - 1)."""
    c = st.contrast("flashnext", FN_MODEL, "nll_novel", (q, "-"), ("exact", "-"))
    d = 100.0 * np.expm1(c["deltas"])
    sd, n, m = S.mde_paired(d)
    return dict(n=n, sd=sd, mde=m)


def _single_cell_mde(run):
    """Median over Llama's (task, length) RULER cells of the per-cell paired SD of RidgeFill - identity (filled
    needles), with the median per-cell n: what one cell can detect (descriptive only, PROTOCOL.md section 6)."""
    model = "llama-3.1-8b-instruct"
    c = run.st.contrast("battery", model, "ruler_filled", ("kva", P), ("noproj", P))
    recs = run.t.records("battery", model, "ruler")
    groups = {}
    for u, d in zip(c["units"], c["deltas"]):
        groups.setdefault((recs[u]["task"], recs[u]["length"]), []).append(d)
    sds = [np.std(v, ddof=1) for v in groups.values() if len(v) >= 2]
    ns = [len(v) for v in groups.values() if len(v) >= 2]
    if not sds:
        raise Missing("no RULER cell with two filled needles")
    sd, n = float(np.median(sds)), int(np.median(ns))
    return dict(n=n, sd=sd, mde=S.mde(sd, n))


# ---------------------------------------------------------------------------------------------- longtables
PS_NLL = [("kva", P), ("noproj", P), ("skip", P), ("pop-native", "pop"), ("kva+qattn10", P), ("noproj+qattn10", P)]
PS_RULER = [("exact", "-"), ("kva", P), ("noproj", P), ("skip", P), ("kva+qattn10", P), ("noproj+qattn10", P)]


class PerSampleFiller(Filler):
    """Per-unit appendix rows. An arm a unit does not carry (its config did not run it: module doc, Populations) prints
    "--", the template's mark for a cell the declared design does not run. RULER rows: the (task, length) cells with
    filled-region needles (NIAH rows; a row with no needle has no filled-region readout)."""

    def rows(self, panel):
        run, lines = self.run, []
        for model in MECH_MODELS:
            try:
                if panel == 0:
                    recs = run.t.records("battery", model, "longppl")
                    half = []
                    for u in sorted(recs, key=lambda u: doc_key(recs[u].get("doc", u))):
                        vals = []
                        for a, s in PS_NLL:      # a superseded arm: its own cell's arm - exact (SUPERSESSION)
                            src = recs if run.st.route("nll_novel", (a, s)) == "nll_novel" else \
                                run.t.records("battery", model, SUPERSESSION["as_set"])
                            arms = src.get(u, {}).get("arms", {})
                            vals.append(fsigned(arms[a][s]["novel_nll"] - arms["exact"]["-"]["novel_nll"], 3)
                                        if s in arms.get(a, {}) else "--")
                        r = recs[u]
                        half.append(" & ".join([doc_label(r.get("doc", u))] + vals))
                    # DEV-25: two documents per line under one group row per model (the table is 342 documents long)
                    lines.append(r"\multicolumn{14}{@{}l}{\textit{" + SHORT[model] + r"}} \\")
                    k = (len(half) + 1) // 2
                    for i in range(k):
                        right = half[k + i] if k + i < len(half) else " & ".join([""] * 7)
                        lines.append(half[i] + " & " + right + r" \\")
                else:
                    recs = [r for r in run.t.records("battery", model, "ruler").values() if r.get("needle_filled")]
                    if not recs:
                        raise Missing(f"{model} RULER: no filled-region needle")
                    for task, L in sorted({(r["task"], r["length"]) for r in recs}):
                        sub = [r for r in recs if (r["task"], r["length"]) == (task, L)]
                        vals = [str(len(sub))]                    # DEV-25: the cell's filled-region samples
                        for a, s in PS_RULER:
                            got = [r["arms"][a][s]["score"] for r in sub if s in r["arms"].get(a, {})]
                            vals.append(fplain(100.0 * np.mean(got), 1) if got else "--")
                        lines.append(" & ".join([SHORT[model], tex_escape(task), f"{L // 1024}K"] + vals) + r" \\")
            except (Missing, KeyError) as e:
                run.missing.append((f"persample panel {panel} {model}", f"{e}"))
        run.cellslog.append({"cell": f"persample panel {panel}", "value": f"{len(lines)} rows", "tier": "locked-final",
                             "protocol": tier_protocol(run, "locked-final"), "sources": "battery longppl/ruler"})
        return lines or [TBD]


# DEV-25: wording fixes of the curated lever list (paper/LEVERS-PAPER.md is pinned by the ledger, so they are applied
# here): lever 30's reason contradicted the measured key-scale shares (review 1 of 2026-10-11, item 3)
LEVER_WORDING = [("none on Qwen3.5, whose key norm undoes it", "none on Qwen3.5, where the key norm leaves little scale "
                  "error")]


def lever_wording(text):
    for old, new in LEVER_WORDING:
        text = text.replace(old, new)
    return text


DOC_DOMAIN = {"tech_doc": "encyc.", "source_code": "code"}     # file prefix -> printed domain (persample caption)


def doc_key(doc):
    """A locked document's file stem <domain>_<nominal length>_<index> as a sort key: domain, length, index (another
    stem sorts by itself)."""
    m = re.fullmatch(r"(.+)_(\d+)_(\d+)", str(doc))
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else (str(doc), 0, 0)


def doc_label(doc):
    dom, length, idx = doc_key(doc)
    if not length:
        return tex_escape(str(doc))
    k = length // 1000 if length % 1000 == 0 else length // 1024          # 128000 -> 128K, 8192 -> 8K
    return f"{tex_escape(DOC_DOMAIN.get(dom, dom))} {k}K-{idx}"


class LeversFiller(Filler):
    def rows(self, panel):
        run = self.run
        path = Path(run.args.levers) if run.args.levers else run.repo / "paper" / "LEVERS-PAPER.md"
        try:
            rel = path.resolve().relative_to(run.repo.resolve())
            p = run.t.pinned(run.repo, rel)
        except (Missing, ValueError) as e:
            run.missing.append(("levers", f"TODO-TABLES-LEVERS-SOURCE: the curated, ledger-pinned lever list "
                                f"paper/LEVERS-PAPER.md is needed: {e}. TRIAGE.md's own table is not usable as is: "
                                "it carries numbers of the excluded history era (FN-H) the stop list bans"))
            return [TBD]
        # the Markdown table of paper/LEVERS-PAPER.md: # | Lever | What it is | Where tested | Status | Evidence;
        # every other line (prose, the header row, the |---| rule) is skipped; "What it is" stays in that file
        cells = [[x.strip() for x in ln.strip().strip("|").split("|")] for ln in p.read_text().splitlines()
                 if ln.startswith("|") and not ln.startswith("|-")]
        rows = [c for c in cells if c and c[0].isdigit()]
        bad = [c for c in rows if len(c) != 6]
        if not rows or bad:
            run.missing.append(("levers", f"{rel}: {len(bad)} lever row(s) without the 6 columns # | Lever | What it "
                                f"is | Where tested | Status | Evidence" if rows else f"{rel} holds no lever row"))
            return [TBD]
        run.cellslog.append({"cell": "levers", "value": f"{len(rows)} rows", "tier": "dev:curated",
                             "protocol": "as each row states", "sources": str(rel)})
        # DEV-25: the pre-fix engine label (protocol.json engine_tier.prefix_label, on many rows) is printed as the key
        # "E" that the caption defines; the Evidence cell wraps (DEV-6)
        eng = "Flash-Next " + _need((run.t.protocol.get("engine_tier") or {}).get("prefix_label"),
                                    "protocol.json engine_tier.prefix_label")
        short = lambda x: tex_label(lever_wording(x.replace(eng + " (v1.0.2)", "E").replace(eng, "E")))   # noqa: E731
        return [" & ".join([*(short(r[i]) for i in (0, 1, 3, 4)), wrap_cell(short(r[5]), "2.2cm")]) + r" \\"
                for r in rows]


class ConfigsFiller(Filler):
    def rows(self, panel):
        run, lines = self.run, []
        for name in sorted(run.t.cells):
            c = run.t.cells[name]
            model = c["model"]["name"]
            w = f"configs {name}"
            S_ = run.cell(w + " S", lambda: split_layer(run, model, c["split"]) if c["split"] != "-" else
                          _need(c.get("split_layer"), f"{name} split_layer"), str, "declared")
            lam = "--"
            if c.get("projector"):
                lam = run.cell(w + " lambda", lambda: projector_meta(run, c)["lam"], lambda v: tex_escape(v), "declared",
                               primary_projector(run, c).get("file", "?"))
            sets = ", ".join(c.get("sets", [c["kind"]]))
            lines.append(" & ".join([wrap_cell(tex_escape(name), CONFIGS_WRAP[0]), SHORT.get(model, tex_escape(model)),
                                     wrap_cell(tex_escape(sets), CONFIGS_WRAP[1]), str(S_), str(c["tail"]), lam,
                                     c["dataset_manifest_sha256"][:12],
                                     wrap_cell(tex_escape(c["registration"]), CONFIGS_WRAP[2])]) + r" \\")
            extra = f"protocol {run.t.cell_version(name)}; code tag {tex_escape(c['code_tag'])}"
            if c.get("projector") and primary_projector(run, c).get("sha256"):
                extra += f"; projector sha256 {primary_projector(run, c)['sha256'][:12]}"
            lines.append(r"\multicolumn{8}{@{}p{0.9\linewidth}@{}}{\quad\footnotesize " + extra + r"} \\")
        run.cellslog.append({"cell": "configs", "value": f"{len(run.t.cells)} configs", "tier": "declared",
                             "protocol": tier_protocol(run, "declared"), "sources": "FINAL- ledger records"})
        return lines


# DEV-6: configs.tex keeps the skeleton's eight one-line columns, which cannot hold real config names, set lists and
# FINAL- record IDs side by side (the mock-filled build ran 73 pt over the text width; real cell names are longer). Those
# three cells are set as top-aligned paragraphs of a fixed width that break at hyphens and spaces; nothing else changes.
CONFIGS_WRAP = ("2.9cm", "2.2cm", "2.9cm")                  # Config, Set, Ledger record


def wrap_cell(text, width):
    return r"\parbox[t]{" + width + r"}{\raggedright\hspace{0pt}" + text + r"\strut}"


_proj_cache = {}


def primary_projector(run, cell):
    """The cell's projector file entry at its primary split (cell v2: projector.files lists one per layer)."""
    files = (cell.get("projector") or {}).get("files") or []
    layer = split_layer(run, cell["model"]["name"], cell["split"]) if cell.get("split") not in (None, "-") else None
    hit = [f for f in files if f.get("layer") == layer]
    return hit[0] if hit else {}


def projector_meta(run, cell):
    """The safetensors header metadata of a cell's declared projector (at its primary split), after checking its
    declared sha256."""
    pj = primary_projector(run, cell)
    if not pj:
        raise Missing(f"{cell['registration']} declares no projector file at its primary split")
    d = run.args.proj_dir
    if not d:
        raise Missing("--proj-dir / KVA_PROJ_DIR is not set: lambda and calibration counts come from the projectors")
    p = Path(d) / pj["file"]
    if p in _proj_cache:
        return _proj_cache[p]
    if not p.is_file():
        raise Missing(f"projector {p} does not exist")
    if not pj.get("sha256"):
        raise Missing(f"{cell['registration']} declares no projector sha256: the projector is not pinned")
    if atomic.sha256_file(p) != pj["sha256"]:
        raise Refused(f"projector {p} does not match the sha256 {cell['registration']} declares")
    with open(p, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        meta = json.loads(f.read(n)).get("__metadata__") or {}
    _proj_cache[p] = meta
    return meta


# ============================================================================================ numbers.tex macros
def fill_macros(run):
    st, m = run.st, run.macro
    L, Q, Q3 = MECH_MODELS
    fn = run.fillers["flashnext"]
    lf, ef = "locked-final", "engine-final"
    q = fn.arm("ridgefill+corr+prot{share}")
    pct = lambda x: 100.0 * math.expm1(x)                                      # noqa: E731
    sp2 = lambda v: fplain(v, 2)                                               # noqa: E731
    # Flash-Next (the contract with the prose: AI-Work/ridgefill-paper-20261004/finish-20261010/FN-MACROS.md)
    if "flashnext" in run.pending:
        m = lambda nm, *_a, **_k: run.macro_pending(nm, "flashnext")        # noqa: E731
    gap = lambda arm, k: (lambda: fn.gap(arm, k))                             # noqa: E731
    pct2 = lambda v: fpct(v, 2, True)                                          # noqa: E731
    cst = lambda f: (lambda v: f(v) + r"\xspace")                             # noqa: E731
    decl = lambda *ks: (lambda: _need(_dig(fn.decl or {}, ks), "the flashnext cell's " + ".".join(ks)))  # noqa: E731
    m("cstFNDocs", lambda: _fn_count(run, fn, "longppl", None, ("data", "longppl", "docs")), cst(str), "declared")
    m("cstFNRulerSamples", lambda: (_fn_count(run, fn, "ruler", lambda r: r["length"] <= 32768, ("data", "ruler", "rows")),
                                    _fn_count(run, fn, "ruler", lambda r: r["length"] == 65536,
                                              ("data", "ruler", "long", "rows"))),
      cst(lambda v: f"{v[0]} + {v[1]}"), "declared")
    m("cstFNProtectShare", decl("declared_share"), cst(lambda v: f"{v}\\%"), "declared")
    m("cstFNPplMargin", decl("criteria", "ppl_gap_upper_pct"), cst(lambda v: fpct(v, 1, signed=True)), "declared",
      "amend-FINAL-FLASHNEXT-C5-11")
    m("cstFNRulerMargin", decl("margins", "ruler_points_lower"),
      cst(lambda v: f"${'-' if v < 0 else '+'}{abs(v):g}$ points"), "declared")
    m("cstFNTTFTLengths", fn.lengths, cst(lambda v: and_list([f"{n // 1024}K" for n in v])), "declared")
    for nm, arm in (("resFNQualityPplGap", q), ("resFNSpeedPplCost", "ridgefill")):
        for suf, k in (("", "est"), ("Lo", "lo"), ("Hi", "hi")):
            m(nm + suf, gap(arm, k), pct2, ef, f"{arm} - exact, nll_novel")
    m("resFNQualityRulerLower", lambda: fn.ruler_noninf(q).lo, lambda v: fsigned(v, 2), ef, q)   # bounds: 2 decimals
    for nm, arm in (("resFNQualityRuler", q), ("resFNExactRuler", "exact"), ("resFNSpeedRuler", "ridgefill")):
        m(nm, lambda arm=arm: st.mean("flashnext", FN_MODEL, "ruler_all", arm, "-"), lambda v: fplain(v, 1), ef, arm)
    for nm, arm, f in (("resFNQualitySpeedupLo", q, min), ("resFNQualitySpeedupHi", q, max),
                       ("resFNSpeedSpeedupLo", "ridgefill", min), ("resFNSpeedSpeedupHi", "ridgefill", max)):
        m(nm, lambda arm=arm, f=f: f(fn.speedup(arm, n) for n in fn.lengths()), sp2, ef, arm)
    m("resFNQualitySpeedupSixteen", lambda: fn.speedup(q, fn.margin("ttft_length")), sp2, ef, q)
    m("resFNQualityLongSpeedup", lambda: fn.long_speedup(q), sp2, ef, f"{q}: 64K RULER rows, median wall_s")
    m("resFNDeterminismDocs", lambda: fn.engine()["determinism"]["bitwise_equal"], str, ef, "engine.json")
    m("resFNDocs", lambda: fn.d_nll(q)["n"], str, ef)
    m("resFNRulerN", lambda: len(run.t.records("flashnext", FN_MODEL, "ruler")), str, ef)
    m("cstFNLayers", lambda: _need((run.t.protocol.get("models", {}).get(FN_MODEL) or {}).get("layers"),
                                   f"protocol.json models.{FN_MODEL}.layers"), cst(str), "declared", "protocol.json")
    m("cstFNSplit", decl("split_layer"), cst(str), "declared")
    m("cstFNTail", decl("tail"), cst(str), "declared")
    m = run.macro
    # the prose's readouts of the reference tables (2026-10-10): each the same cell the battery tables print
    for nm, met in (("resQwenDenseRulerDiff", "ruler_filled"),):
        for suf, k in (("", "est"), ("Lo", "lo"), ("Hi", "hi")):
            m(nm + suf, lambda k=k: st.contrast("battery", Q3, "ruler_filled", ("kva", P), ("noproj", P))[k],
              lambda v: fsigned(v, 1), lf, f"{Q3} ruler_filled: kva@{P} - noproj@{P}")
    for nm, mod, arm, s_ in (("resQwenDenseFillRuler", Q3, "kva", P), ("resQwenDenseIdRuler", Q3, "noproj", P),
                             ("resQwenDenseExactRuler", Q3, "exact", "-"), ("resQwenDenseRowsRuler", Q3, "kva+qattn10", P),
                             ("resQwenDenseRandomRowsRuler", Q3, "kva+random", P), ("resLlamaRowsRuler", L, "kva+qattn10", P),
                             ("resLlamaExactRuler", L, "exact", "-")):
        m(nm, lambda mod=mod, arm=arm, s_=s_: st.mean("battery", mod, "ruler_filled", arm, s_), lambda v: fplain(v, 1), lf,
          f"{mod} ruler_filled {arm}@{s_}")
    for nm, arm, s_ in (("resLlamaLongBenchFill", "kva", P), ("resLlamaLongBenchPop", "pop-native", "pop"),
                        ("resLlamaLongBenchExact", "exact", "-")):
        m(nm, lambda arm=arm, s_=s_: st.mean("battery", L, "longbench", arm, s_), lambda v: fplain(v, 1), lf,
          f"{L} longbench {arm}@{s_}")
    for nm, mod in (("resLlamaKl", L), ("resQwenKl", Q), ("resQwenDenseKl", Q3)):
        for suf, arm in (("Fill", "kva"), ("Id", "noproj")):
            m(nm + suf, lambda mod=mod, arm=arm: st.mean("battery", mod, "kl", arm, P), lambda v: fplain(v, 3), lf,
              f"{mod} decode kl_mean {arm}@{P}")
    m("resQwenTurnTwoRidge", lambda: st.mean("battery", Q, "turn2", "kva+qattn10", P), lambda v: fplain(v, 1), lf,
      f"{Q} turn2 kva+qattn10@{P}")
    margin_pts = lambda v: f"${'-' if v < 0 else '+'}{abs(v):g}$ points" + r"\xspace"           # noqa: E731
    m("cstRowsMargin", lambda: _single([c["margins"]["ruler_points_lower"] for mod in C3_NONINF
                                        for c in run.t.cells_of(mod, "battery") if c["kind"] == "ruler"],
                                       "the RULER cells' declared ruler_points_lower"), margin_pts, "declared",
      "the RULER cells' margins (FINAL- records)")
    m("cstDeepMargin", lambda: RULER_DEEP_MARGIN, margin_pts, "declared", "the ruler-deep registrations' margin")
    m("refHardware", lambda: ref_hardware(run), str, "declared", "the reference cells' host.json gpu_name")
    review_macros(run, fn, q)
    # dense model
    c = lambda mod, met, a, b: st.contrast("battery", mod, met, a, b)          # noqa: E731
    c_sub = lambda mod, a, b, subset: st.contrast("battery", mod, "ruler_filled", a, b, subset=subset)  # noqa: E731
    for nm, met in (("resLlamaRulerDiff", "ruler_filled"), ("resLlamaNllDiff", "nll_novel"),
                    ("resQwenRulerDiff", "ruler_filled"), ("resQwenNllDiff", "nll_novel")):
        mod = L if nm.startswith("resLlama") else Q
        nd = METRIC_ND[met]
        m(nm, lambda mod=mod, met=met: c(mod, met, ("kva", P), ("noproj", P))["est"], lambda v, nd=nd: fsigned(v, nd), lf)
        m(nm + "Lo", lambda mod=mod, met=met: c(mod, met, ("kva", P), ("noproj", P))["lo"], lambda v, nd=nd: fsigned(v, nd), lf)
        m(nm + "Hi", lambda mod=mod, met=met: c(mod, met, ("kva", P), ("noproj", P))["hi"], lambda v, nd=nd: fsigned(v, nd), lf)
    m("resLlamaRulerVsNofill", lambda: c(L, "ruler_filled", ("kva", P), ("skip", P))["est"], lambda v: fsigned(v, 1), lf)
    m("resLlamaNllVsNofill", lambda: c(L, "nll_novel", ("kva", P), ("skip", P))["est"], lambda v: fsigned(v, 3), lf)
    m("resLlamaPopNativeNll", lambda: st.mean("battery", L, "nll_novel", "pop-native", "pop", vs_exact=True),
      lambda v: fsigned(v, 3), lf)
    m("resLlamaPopNativeRuler", lambda: st.mean("battery", L, "ruler_filled", "pop-native", "pop"), lambda v: fplain(v, 1), lf)
    for nm, arm in (("resLlamaRowsGain", "kva+qattn10"), ("resLlamaRandomRowsGain", "kva+random"),
                    ("resLlamaLongerTailGain", "kva+tailmatch")):
        m(nm, lambda arm=arm: c(L, "ruler_filled", (arm, P), ("kva", P))["est"], lambda v: fsigned(v, 1), lf)
    for nm, arm in (("resLlamaAgreeRidge", "kva"), ("resLlamaAgreeIdentity", "noproj"), ("resLlamaAgreeNofill", "skip")):
        m(nm, lambda arm=arm: st.mean("battery", L, "agree", arm, P), fpct, lf)
    for nm, arm, s in (("resLlamaTurnTwoRidge", "kva+qattn10", P), ("resLlamaTurnTwoIdentity", "noproj+qattn10", P),
                       ("resLlamaTurnTwoExact", "exact", "-")):
        m(nm, lambda arm=arm, s=s: st.mean("battery", L, "turn2", arm, s), lambda v: fplain(v, 1), lf)
    for suf, k in (("", "est"), ("Lo", "lo"), ("Hi", "hi")):
        m("resSwiftMapsMinusRidge" + suf, lambda k=k: c(L, "nll_novel", ("swiftkv", R12), ("kva", R12))[k],
          lambda v: fsigned(v, 3), lf)
    m("resSwiftNativeNll", lambda: st.mean("battery", L, "nll_novel", "swiftkv-native", R12, vs_exact=True),
      lambda v: fsigned(v, 3), lf)
    for suf, k in (("", "est"), ("Lo", "lo"), ("Hi", "hi")):           # R6: SwiftKV as released - RidgeFill at S16
        m("resSwiftNativeMinusRidge" + suf, lambda k=k: c(L, "nll_novel", ("swiftkv-native", R12), ("kva", R12))[k],
          lambda v: fsigned(v, 3), lf, f"{L} nll_novel: swiftkv-native@{R12} - kva@{R12}")
    # hybrid model
    m("resQwenNllMDE", lambda: c(Q, "nll_novel", ("kva", P), ("noproj", P))["mde"], lambda v: fplain(v, 3), lf)
    m("resQwenRulerMDE", lambda: c(Q, "ruler_filled", ("kva", P), ("noproj", P))["mde"], lambda v: fplain(v, 1), lf)
    for nm, mod in (("resLlama", L), ("resQwenDense", Q3)):
        m(nm + "RulerMDE", lambda mod=mod: c(mod, "ruler_filled", ("kva", P), ("noproj", P))["mde"], lambda v: fplain(v, 1), lf)
    for nm, mod in (("resLlama", L), ("resQwen", Q), ("resQwenDense", Q3)):          # the pooled filled-region pairs
        m(nm + "RulerPairs", lambda mod=mod: c(mod, "ruler_filled", ("kva", P), ("noproj", P))["n"], str, lf)
    for nm, arm, s, met in (("resQwenRowsRuler", "noproj+qattn10", P, "ruler_filled"), ("resQwenExactRuler", "exact", "-", "ruler_filled"),
                            ("resQwenFillRuler", "kva", P, "ruler_filled"), ("resQwenRandomRowsRuler", "kva+random", P, "ruler_filled"),
                            # protocol 1.1.3: the question-blind rows are qblind10 (the T1.4 question-first runs were a
                            # question-position test, and Qwen3.5's are cut); NUMBERS_SRC restates the src note
                            ("resQwenBlindRowsRuler", "noproj+qblind10", P, "ruler_filled")):
        m(nm, lambda arm=arm, s=s, met=met: st.mean("battery", Q, met, arm, s), lambda v: fplain(v, 1), lf)
    # matched depth vs POP as published (DEV-15) and the fresh-data confirmation (DEV-16): follow-ups, own families
    for mod, pre in DEEP.items():
        if deep_cell(run, mod) is None:
            continue
        src = f"{mod} ruler-deep filled"
        for arm, split, tag in DEEP_ARMS:
            m(f"{pre}Deep{tag}Ruler", lambda mod=mod, arm=arm, split=split: float(np.mean(list(
                st.vals("battery", mod, "deep_filled", arm, split).values()))), lambda v: fplain(v, 1), lf, src)
        ni = lambda mod=mod: deep_noninf(run, mod)                                 # noqa: E731
        m(f"{pre}DeepFillMinusPop", lambda ni=ni: ni().estimate, lambda v: fsigned(v, 1), lf, src)
        m(f"{pre}DeepFillMinusPopBound", lambda ni=ni: ni().lo, lambda v: fsigned(v, 2), lf, src)
        m(f"{pre}DeepFillMinusPopN", lambda ni=ni: ni().n, str, lf, src)
        m(f"{pre}DeepFillMinusPopVerdict", lambda ni=ni: noninf_verdict(ni()), str, lf, src)
        for suf, k in (("", "est"), ("Lo", "lo"), ("Hi", "hi")):
            m(f"{pre}DeepFillMinusId{suf}", lambda mod=mod, k=k: deep_vs_identity(run, mod)[k], lambda v: fsigned(v, 1),
              lf, src)
    for mod, pre in FRESH.items():
        scope, dataset = fresh_scope(run, mod)
        if scope is None:
            continue
        for _, met, a, b, stem, role in FRESH_ROWS:
            src = f"{dataset} {mod} {met}: {a[0]}@{a[1]} - {b[0]}@{b[1]}"
            for suf, k in (("", "est"), ("Lo", "lo"), ("Hi", "hi")):
                m(f"{pre}{stem}{suf}", lambda met=met, a=a, b=b, k=k, scope=scope, mod=mod: st.contrast(
                    scope, mod, met, a, b)[k], lambda v: fsigned(v, 3), dataset, src)
            if role == "holm":
                m(f"{pre}{stem}Holm", lambda stem=stem, mod=mod: fresh_holm(run, mod)[stem], lambda v: fmt_p(*v),
                  dataset, src)
        m(f"{pre}FreshDocs", lambda scope=scope, mod=mod: st.contrast(scope, mod, "nll_novel", ("kva", "1/2"),
                                                                     ("noproj", "1/2"))["n"], str, dataset)
    # Llama RULER on its own (DEV-14): the 64K contrasts and the 1/2-split follow-up, outside every family
    for _, a, b, subset, _, stem in LLAMA_ON_ITS_OWN:
        src = f"{L} ruler_filled: {a[0]}@{a[1]} - {b[0]}@{b[1]}" + (f" ({subset} rows)" if subset else "")
        get = lambda a=a, b=b, subset=subset: c_sub(L, a, b, subset)                # noqa: E731
        for suf, k in (("", "est"), ("Lo", "lo"), ("Hi", "hi")):
            m(f"resLlamaRuler{stem}{suf}", lambda get=get, k=k: get()[k], lambda v: fsigned(v, 1), lf, src)
        m(f"resLlamaRuler{stem}N", lambda get=get: get()["n"], str, lf, src)
    # the question-blind pair (DEV-12): filled-region accuracy and blind minus question-attended rows
    for mod, pre in QBLIND_MACROS.items():
        for _, arm, _, rarm, tag in QBLIND_ROWS:
            src = f"{mod} ruler_filled: {arm}@{P} - {rarm}@{P}"
            m(f"{pre}{tag}BlindRuler", lambda mod=mod, arm=arm: st.mean("battery", mod, "ruler_filled", arm, P),
              lambda v: fplain(v, 1), lf, src)
            for suf, k in (("", "est"), ("Lo", "lo"), ("Hi", "hi")):
                m(f"{pre}{tag}BlindMinusRows{suf}", lambda mod=mod, arm=arm, rarm=rarm, k=k: c(
                    mod, "ruler_filled", (arm, P), (rarm, P))[k], lambda v: fsigned(v, 1), lf, src)
    # superseded 1.1.1 NLL values (SUPERSESSION): the size of the leak only, never a claim
    for mod, pre in LEAK_MODELS.items():
        for arm, tag in LEAK_ARMS:
            orig = lambda mod=mod, arm=arm: st.mean("battery", mod, "nll_novel", arm, P, vs_exact=True, route=False)  # noqa: E731
            fixed = lambda mod=mod, arm=arm: st.mean("battery", mod, "nll_novel", arm, P, vs_exact=True)  # noqa: E731
            src = f"SUPERSEDED: {mod} {arm}@{P} dNLL novel, {SUPERSESSION['of']} cell minus {SUPERSESSION['by']} cell"
            m(f"{pre}Leak{tag}", lambda o=orig, f=fixed: o() - f(), lambda v: fsigned(v, 3), lf, src)
            m(f"{pre}Leak{tag}Orig", orig, lambda v: fsigned(v, 3), lf, src)
            m(f"{pre}Leak{tag}Fixed", fixed, lambda v: fsigned(v, 3), lf, src)
    # C3: the rows arms against exact, non-inferiority (DEV-9; the models whose FINAL- record declares it)
    for mod, (_, pre) in C3_NONINF.items():
        for _, arm, tag in C3_ROWS:
            get = lambda mod=mod, arm=arm: c3_noninf(run, mod, arm)                 # noqa: E731
            nm, src = f"{pre}{tag}VsExact", f"{mod} ruler_filled: {arm}@{P} - exact"
            if mod in C3_POSTHOC:
                run.t.require_record(C3_POSTHOC[mod], "hypothesis")
                src += f" (follow-up {C3_POSTHOC[mod]}, declared 2026-10-08 after a readout: post hoc)"
            m(nm, lambda get=get: get().estimate, lambda v: fsigned(v, 1), lf, src)
            m(nm + "Bound", lambda get=get: get().lo, lambda v: fsigned(v, 2), lf, src)
            m(nm + "N", lambda get=get: get().n, str, lf, src)
            m(nm + "Verdict", lambda get=get: noninf_verdict(get()), str, lf, src)
    # mechanism
    for nm, mod, arm in (("resLlamaKexactShare", L, "kva+kexact"), ("resLlamaVexactShare", L, "kva+vexact"),
                         ("resQwenKexactShare", Q, "kva+kexact"), ("resQwenVexactShare", Q, "kva+vexact")):
        m(nm, lambda mod=mod, arm=arm: r_est(loss_share(run, mod, arm)()), fpct, lf)
    for nm, mod, arm, where in (("resLlamaKexactRuler", L, "kva+kexact", None), ("resQwenKexactRuler", Q, "kva+kexact", None),
                                ("resLlamaMechExactRuler", L, "exact", mech_rows), ("resQwenMechExactRuler", Q, "exact", mech_rows)):
        if not_run(run, mod, "splices"):
            run.macros[nm], run.macro_src[nm] = "--", f"not run (cut {CUT_DATE}): {NOT_RUN_WHAT['splices']}"
            continue
        m(nm, lambda mod=mod, arm=arm, where=where: st.mean("battery", mod, "ruler_filled", arm,
                                                           "-" if arm == "exact" else P, where=where),
          lambda v: fplain(v, 1), lf)
    m("resLlamaKtailShare", lambda: r_est(loss_share(run, L, "kva+ktail")()), fpct, lf)
    for suf, k in (("Lo", "lo"), ("Hi", "hi")):
        m("resLlamaKtailShare" + suf, lambda k=k: r_bound(loss_share(run, L, "kva+ktail")(), k), fpct, lf)
    # S16 probe (dev set, protocol 1.0.2): ledger values + the v1.0 battery pinned by the ledger
    probe = lambda: run.t.ledger_by_label("s16probe-results")["values"]       # noqa: E731
    share = lambda i, j: 100.0 * (probe()[i] - probe()[j]) / probe()[i]       # noqa: E731
    tier = "dev:v1.0.2"
    m("resProbeKeyShareLo", lambda: min(share(0, 2), share(4, 6)), lambda v: fpct(v, 0), tier, "s16probe-results")
    m("resProbeKeyShareHi", lambda: max(share(0, 2), share(4, 6)), lambda v: fpct(v, 0), tier, "s16probe-results")
    m("resProbeValShareLo", lambda: min(share(0, 3), share(4, 7)), lambda v: fpct(v, 1), tier, "s16probe-results")
    m("resProbeValShareHi", lambda: max(share(0, 3), share(4, 7)), lambda v: fpct(v, 1), tier, "s16probe-results")
    m("resProbeSinkShareLo", lambda: min(share(0, 1), share(4, 5)), lambda v: fpct(v, 1), tier, "s16probe-results")
    m("resProbeSinkShareHi", lambda: max(share(0, 1), share(4, 5)), lambda v: fpct(v, 1), tier, "s16probe-results")
    m("resProbeVexactExcursion", lambda: probe()[11], lambda v: fsigned(v, 3), tier, "s16probe-results")
    ex = lambda: probe_excess(run)                                             # noqa: E731
    m("resProbeExcess", lambda: ex()["est"], lambda v: fsigned(v, 3), tier, "battery-v1.0.json")
    m("resProbeExcessLo", lambda: ex()["lo"], lambda v: fsigned(v, 3), tier, "battery-v1.0.json")
    m("resProbeExcessHi", lambda: ex()["hi"], lambda v: fsigned(v, 3), tier, "battery-v1.0.json")
    m("resProbeExcessPrompts", lambda: ex()["worse"], str, tier, "battery-v1.0.json")
    m("resProbeExcessTotal", lambda: ex()["n"], str, tier, "battery-v1.0.json")
    # cost model (reference PyTorch path, pre-protocol timing files pinned by the ledger)
    m("resCostOverLo", lambda: min(cost_over(run)), lambda v: fpct(v, 0), "dev:pre-protocol", "timing-sweep-*.json")
    m("resCostOverHi", lambda: max(cost_over(run)), lambda v: fpct(v, 0), "dev:pre-protocol", "timing-sweep-*.json")
    # calibration (the final projectors' headers, checked against the FINAL- records' declared sha256)
    m("resCalibDocs", lambda: _single([calib_docs(run, cc) for cc in final_projector_cells(run)], "calibration docs"),
      str, "declared", "projector headers; fit records")
    for nm, mod in (("resCalibRowsLlama", L), ("resCalibRowsQwen", Q), ("resCalibRowsQwenDense", Q3)):
        m(nm, lambda mod=mod: int(_single([projector_meta(run, cc)["rows"] for cc in final_projector_cells(run, mod)],
                                          f"{mod} calibration rows")),
          lambda v: f"{v:,}".replace(",", "{,}"), "declared", "projector headers")
    m("protocolversion", lambda: run.t.protocol["version"], str, "declared", "protocol.json version")
    m("protocolversionsrun", lambda: and_list(sorted(run.t.versions, key=version_key)), str, "declared",
      "the protocol stamps of the cells' records (one per cell; the configs table names each cell's)")
    m("cstKeyTokenThreshold", lambda: _need(run.t.protocol.get("keytoken_threshold"), "protocol.json keytoken_threshold"),
      lambda v: fplain(float(v), 1), "declared", "protocol.json")


def review_macros(run, fn, q):
    """Macros the 2026-10-11 reviews asked for: comparisons over the same samples, the declared primary contrasts the
    prose cites, the samples behind a non-inferiority bound, the engine's 64K request times and the engine build.
    Each is a cell the tables print or the contrast behind one."""
    st, lf, ef = run.st, "locked-final", "engine-final"
    run.new_notes = getattr(run, "new_notes", {})

    def m(name, fn_, fmt, tier, sources=""):                # numbers.tex gains each once (rewrite_numbers' `new`)
        run.new_notes[name] = f"generated (reviews of 2026-10-11): {sources}"
        if "flashnext" in run.pending and (name.startswith("resFN") or name.startswith("cstFN")):
            return run.macro_pending(name, "flashnext")
        run.macro(name, fn_, fmt, tier, sources)
    L, Q, Q3 = MECH_MODELS
    c = lambda mod, met, a, b: st.contrast("battery", mod, met, a, b)          # noqa: E731
    ci3 = (("", "est"), ("Lo", "lo"), ("Hi", "hi"))
    # the controls' population (8K-32K): RidgeFill and RidgeFill + rows on the samples where the random-rows control ran
    for pre, mod in (("resLlama", L), ("resQwen", Q)):
        ctl = lambda r: P in r["arms"].get("kva+random", {})                    # noqa: E731
        src = f"{mod} ruler_filled on the units of kva+random@{P}"
        m(pre + "ControlN", lambda mod=mod: c(mod, "ruler_filled", ("kva+qattn10", P), ("kva+random", P))["n"], str, lf, src)
        for nm, arm in (("FillRulerCtl", "kva"), ("RowsRulerCtl", "kva+qattn10")):
            m(pre + nm, lambda mod=mod, arm=arm: st.mean("battery", mod, "ruler_filled", arm, P, where=ctl),
              lambda v: fplain(v, 1), lf, src)
        for nm, b in (("RowsVsRandom", "kva+random"), ("RowsVsLongerTail", "kva+tailmatch")):
            for suf, k in ci3:
                m(pre + nm + suf, lambda mod=mod, b=b, k=k: c(mod, "ruler_filled", ("kva+qattn10", P), (b, P))[k],
                  lambda v: fsigned(v, 1), lf, f"{mod} ruler_filled: kva+qattn10@{P} - {b}@{P}")
    for suf, k in ci3:
        m("resQwenDenseRowsVsRandom" + suf, lambda k=k: c(Q3, "ruler_filled", ("kva+qattn10", P), ("kva+random", P))[k],
          lambda v: fsigned(v, 1), lf, f"{Q3} ruler_filled: kva+qattn10@{P} - kva+random@{P}")
    # POP native's RULER population on Llama (8K-32K): RidgeFill on the same samples
    pop_u = lambda r: "pop" in r["arms"].get("pop-native", {})                 # noqa: E731
    m("resLlamaPopNativeRulerN", lambda: len(st.vals("battery", L, "ruler_filled", "pop-native", "pop")), str, lf,
      f"{L} ruler_filled pop-native@pop: n")
    m("resLlamaFillRulerOnPop", lambda: st.mean("battery", L, "ruler_filled", "kva", P, where=pop_u),
      lambda v: fplain(v, 1), lf, f"{L} ruler_filled kva@{P} on the units of pop-native@pop")
    for pre, mod in (("resLlama", L), ("resQwen", Q)):      # the lengths of the controls' population
        def ctl_lengths(mod=mod):
            recs = run.t.records("battery", mod, "ruler")
            lens = sorted({recs[u]["length"] for u in st.vals("battery", mod, "ruler_filled", "kva+random", P)})
            return f"{lens[0] // 1024}K--{lens[-1] // 1024}K" if len(lens) > 1 else f"{lens[0] // 1024}K"
        m(pre + "ControlLengths", ctl_lengths, str, lf, f"{mod} ruler_filled kva+random@{P}: sample lengths")
    m("cstFNClaimLength", lambda: fn.margin("ttft_length"), lambda v: f"{v // 1024}K\\xspace", "declared",
      "the flashnext cell's margins.ttft_length (FINAL-FLASHNEXT-C5)")
    m("resLlamaRulerVsNofillN", lambda: c(L, "ruler_filled", ("kva", P), ("skip", P))["n"], str, lf,
      f"{L} ruler_filled: kva@{P} - skip@{P}")
    # the declared primary contrast of the rows arms (RidgeFill + rows - identity + rows), every model
    for pre, mod in (("resLlama", L), ("resQwen", Q), ("resQwenDense", Q3)):
        src = f"{mod} ruler_filled: kva+qattn10@{P} - noproj+qattn10@{P}"
        get = lambda mod=mod: c(mod, "ruler_filled", ("kva+qattn10", P), ("noproj+qattn10", P))   # noqa: E731
        for suf, k in ci3 + (("MDE", "mde"),):
            m(pre + "RowsPrimary" + suf, lambda get=get, k=k: get()[k],
              (lambda v: fplain(v, 1)) if k == "mde" else (lambda v: fsigned(v, 1)), lf, src)
        # greedy agreement, RidgeFill - identity (percentage points)
        for suf, k in ci3:
            m(pre + "AgreeDiff" + suf, lambda mod=mod, k=k: c(mod, "agree", ("kva", P), ("noproj", P))[k],
              lambda v: fsigned(v, 1), lf, f"{mod} agree: kva@{P} - noproj@{P}")
    for suf, k in ci3 + (("MDE", "mde"),):
        m("resQwenDenseNllDiff" + suf, lambda k=k: c(Q3, "nll_novel", ("kva", P), ("noproj", P))[k],
          (lambda v: fplain(v, 3)) if k == "mde" else (lambda v: fsigned(v, 3)), lf, f"{Q3} nll_novel: kva@{P} - noproj@{P}")
    # the key-scale correction on the QK-normed models (mechanism panel A)
    for pre, mod in (("resQwen", Q), ("resQwenDense", Q3)):
        m(pre + "KtailShare", lambda mod=mod: r_est(loss_share(run, mod, "kva+ktail")()), fpct, lf, f"{mod} ktail share")
        for suf, k in (("Lo", "lo"), ("Hi", "hi")):
            m(pre + "KtailShare" + suf, lambda mod=mod, k=k: r_bound(loss_share(run, mod, "kva+ktail")(), k), fpct, lf,
              f"{mod} ktail share")
    m("resLlamaTurnTwoFill", lambda: st.mean("battery", L, "turn2", "kva", P), lambda v: fplain(v, 1), lf,
      f"{L} turn2 kva@{P}")
    m("resSingleCellN", lambda: _single_cell_mde(run)["n"], str, lf, "mde table, single RULER cell: median per-cell n")
    # the samples behind each rows-versus-exact bound: how many scored below / above exact
    for mod, (_, pre) in C3_NONINF.items():
        get = lambda mod=mod: c(mod, "ruler_filled", ("kva+qattn10", P), ("exact", "-"))["deltas"]   # noqa: E731
        src = f"{mod} ruler_filled: kva+qattn10@{P} - exact, per-sample deltas"
        m(pre + "FillRowsVsExactWorse", lambda get=get: int((get() < 0).sum()), str, lf, src)
        m(pre + "FillRowsVsExactBetter", lambda get=get: int((get() > 0).sum()), str, lf, src)
    # the engine: quality mode's filled-region samples below exact, and the 64K request times
    fd = lambda: fn.ruler_d(q)                                                  # noqa: E731
    m("resFNQualityRulerFilledN", lambda: fd()["n"], str, ef, q)
    m("resFNQualityRulerWorse", lambda: int((fd()["deltas"] < 0).sum()), str, ef, q)
    worse = lambda arm: [u for u, d in zip(fd()["units"], fd()["deltas"]) if d < 0]     # noqa: E731
    for nm, arm in (("resFNQualityRulerWorseScore", q), ("resFNQualityRulerWorseExactScore", "exact")):
        m(nm, lambda arm=arm: float(np.mean([st.vals("flashnext", FN_MODEL, "ruler_filled", arm, "-")[u]
                                             for u in worse(arm)])), lambda v: fplain(v, 1), ef, q)
    long = lambda r: r.get("length") == 65536                                  # noqa: E731
    for nm, arm in (("resFNLongExactSec", "exact"), ("resFNLongQualitySec", q), ("resFNLongSpeedSec", "ridgefill")):
        m(nm, lambda arm=arm: statistics.median(run.t.values("flashnext", FN_MODEL, "ruler", arm, "-", "wall_s",
                                                             where=long).values()),
          lambda v: fplain(v, 1), ef, f"{arm}: 64K RULER rows, median wall_s")
    m("resFNSpeedLongSpeedup", lambda: fn.long_speedup("ridgefill"), lambda v: fplain(v, 2), ef,
      "ridgefill: 64K RULER rows, median wall_s")
    # the engine build (protocol.json engine_tier.c5_image: the image, its digest and the one-file fix)
    img = lambda: _need((run.t.protocol.get("engine_tier") or {}).get("c5_image"), "protocol.json engine_tier.c5_image")  # noqa: E731
    m("engineImage", lambda: _need(img().get("image"), "c5_image.image"), lambda v: rf"\texttt{{{tex_escape(v)}}}\xspace",
      "declared", "protocol.json engine_tier.c5_image.image")
    m("engineDigest", lambda: _need(img().get("id"), "c5_image.id"), lambda v: rf"\texttt{{{tex_escape(v[:19])}}}\xspace",
      "declared", "protocol.json engine_tier.c5_image.id (sha256, first twelve hex digits)")
    m("engineFixSha", lambda: _single(((img().get("overlay") or {}).get("files") or {}).values(), "the overlay's file"),
      lambda v: rf"\texttt{{{v[:12]}}}\xspace", "declared",
      "protocol.json engine_tier.c5_image.overlay.files (sha256, first twelve hex digits)")


def ref_hardware(run):
    """The GPUs the reference-model finals ran on: each locked-set or frozen-set cell's host.json gpu_name (one GPU per
    cell), as text. The cells record no cloud provider."""
    names = set()
    for n, c in run.t.cells.items():
        if scope_of(c) == "flashnext":
            continue
        h = run.t.cell_dirs[n] / "host.json"
        g = (json.loads(h.read_text()).get("host") or {}).get("gpu_name") if h.is_file() else None
        if not g:
            raise Missing(f"{n}: host.json names no gpu_name")
        names.add(g)
    short = sorted(g.replace("NVIDIA ", "", 1) for g in names)
    return "NVIDIA " + and_list(short) + " GPUs, one per cell"


def _dig(d, keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _fn_count(run, fn, set_, where, declared):
    """The number of a Flash-Next set's units (where: a filter), which must be the count the engine cell declares."""
    recs = run.t.records("flashnext", FN_MODEL, set_)
    n = sum(1 for r in recs.values() if where is None or where(r))
    want = _dig(fn.decl or {}, declared)
    if n != want:
        raise Missing(f"flashnext {set_}: {n} units, the cell declares {want} ({'.'.join(declared)})")
    return n


def version_key(v):
    return tuple(int(x) for x in str(v).split("."))


def and_list(items):
    items = [str(x) for x in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def calib_docs(run, cell):
    """The calibration document count of a cell's final projector: its header's 'docs'; for a projector fitted before
    the header carried it (the Qwen3-8B sweep), the count its fit recorded, found by the pinned sha256
    (finals/lib/refit.py docs_count: the fit report's 'docs' or results/calib-docs-<model>-sweep.json)."""
    meta = projector_meta(run, cell)
    if "docs" in meta:
        return int(meta["docs"])
    pj = primary_projector(run, cell)
    n = refit.docs_count(run.repo, cell["model"]["name"], pj["layer"], pj["sha256"])
    if n is None:
        raise Missing(f"{pj['file']}: no calibration document count in its header, its fit report or "
                      f"results/calib-docs-{cell['model']['name']}-sweep.json (scripts/calib_docs_of_sums.py)")
    return n


def final_projector_cells(run, model=None):
    cs = [c for c in run.t.cells.values() if c.get("projector") and c["model"]["name"] in MECH_MODELS
          and c.get("split") == PRIMARY and (model is None or c["model"]["name"] == model)]   # the primary projectors
    if not cs:
        raise Missing(f"no FINAL- cell declares a projector{' for ' + model if model else ''}")
    return cs


def probe_excess(run):
    """Identity fill S16 - S12 on Llama's graded dev set at T4096 (S16PROBE's premise), per prompt from the v1.0
    battery file the ledger pins; BCa as everywhere else (the hypothesis text quoted a percentile interval)."""
    p = run.t.pinned(run.repo, Path(run.args.battery_v10 or "results/tables/battery-v1.0.json"))
    rows = {(r["model"], r.get("format"), r["split"], r["tail"], r["arm"]): r for r in json.loads(p.read_text())["grade"]}
    k = lambda s: ("llama-3.1-8b-instruct", "raw", s, 4096, "noproj")          # noqa: E731
    if k(16) not in rows or k(12) not in rows:
        raise Missing("the v1.0 battery has no Llama grade noproj rows at S12 and S16, T4096")
    units, a, b = S.paired(rows[k(16)]["per_prompt"], rows[k(12)]["per_prompt"], what="S16 - S12 identity fill")
    d = a - b
    r = S.bca(d)
    return dict(est=r.estimate, lo=r.lo, hi=r.hi, worse=int((d > 0).sum()), n=len(d))


def cost_over(run):
    """Analytic / measured speedup - 1 (percent) at the primary split (S12, T4096) per reference model."""
    out = []
    for rel in run.args.timing or ["results/timing-sweep-llama.json", "results/timing-sweep-qwen.json"]:
        p = run.t.pinned(run.repo, Path(rel))
        runs = json.loads(p.read_text())["runs"]
        hit = [v for v in runs.values() if v.get("split") == 12 and v.get("tail") == 4096]
        if len(hit) != 1:
            raise Missing(f"{rel}: expected one S12 T4096 run, found {len(hit)}")
        out.append(100.0 * (hit[0]["analytic_speedup"] / hit[0]["measured_speedup"] - 1.0))
    return out


# ============================================================================================ rendering
ROLE_COLUMN = {}


def _roles():
    for which, name in (("dense", "battery-dense"), ("other", "battery-hybrid")):
        rows = template_rows(name, 1)
        ROLE_COLUMN[which] = [r[-1] for r in rows]


def template_rows(name, panel):
    """The data rows (split cells) of one panel of a template."""
    out = []

    class Rec(Filler):
        def __init__(self):
            self.caps = []

        def cell(self, p, r, cells):
            if p == panel:
                out.append(cells)
            return cells

        def caption(self):
            return TBD

        def rows(self, p):
            return []
    render(name, Rec(), structural=False)
    return out


def render(name, filler, structural=True):
    """The table file: the template with its slots filled by `filler` (None: the template itself)."""
    text = TEMPLATES[name]
    if filler is None:
        return text
    out, panel, in_body, row, longt, head = [], -1, False, 0, False, []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith(r"\begin{tabular}") or s.startswith(r"\begin{longtable}"):
            panel, in_body, row, longt = panel + 1, False, 0, s.startswith(r"\begin{longtable}")
            head = []
            if structural and (name, panel) in SPACING:
                out.append(SPACING[(name, panel)] + " % DEV-5")
            if structural and not longt:                      # DEV-18: the panel is set in a box first
                out.extend(FIT_OPEN)
            if structural and name == "mechanism" and panel == 0:
                out.append("% DEV-7: RULER cells of kvexact and ktail are -- (the declared cells run only kexact and "
                           "vexact on RULER; finals/tables.py MECH_NOT_ON_RULER)")
        elif longt and not in_body and not s.startswith(r"\caption") and s not in (r"\endhead", r"\bottomrule",
                                                                                     r"\endfoot"):
            head.append(line)                                 # \toprule, the header row, \midrule
        elif s == r"\midrule" and not longt:
            if not in_body and structural and name == "negative" and panel == 0:
                out.append(line)
                out.append(ERA_BANNER.replace("VERSIONS", ", ".join(locked_versions(filler.run.t))) + " % DEV-3")
                in_body = True
                continue
            in_body = True
        elif s == r"\endfoot" and longt:
            in_body = True
            out.append(line)
            continue
        elif s == r"\endhead" and longt and structural:
            # DEV-4: the caption and label only on the first page; later pages repeat the header (longtable idiom)
            out.append(r"\endfirsthead % DEV-4")
            out.append(r"\caption[]{(continued)}\\ % DEV-4")
            out.extend(h + " % DEV-4" for h in head)
            out.append(line)                                  # the template's own \endhead
            continue
        elif s.startswith(r"\bottomrule") and not longt:
            in_body = False
        elif s.startswith(r"\end{longtable}"):
            in_body = False
            if structural and (name, panel) in SPACING and SPACING[(name, panel)].startswith(r"\begingroup"):
                out.append(line)
                out.append(r"\endgroup % DEV-5")
                continue
        if in_body and "&" in s and s.endswith("\\\\"):
            if longt:
                out.extend(filler.rows(panel))
                continue
            cells = [c.strip() for c in s[:-2].split("&")]
            out.append(" & ".join(filler.cell(panel, row, cells)) + r" \\")
            if structural and hasattr(filler, "after_row"):
                out.extend(filler.after_row(panel, cells[0]))   # DEV-12 rows (they carry their marker)
            row += 1
            continue
        if TBD in line and not s.startswith("%"):
            line = re.sub(r"\\TBDcell(\{\})?", lambda _m: filler.caption(), line)
        if structural and s == r"\end{tabular}":
            out.append(line)
            out.extend(FIT_CLOSE)                             # DEV-18: scaled to the text width only when wider
            line = None
            extra = [NOT_RUN_NOTE + " % DEV-8"] if panel == 0 and name in NOT_RUN_TABLES else []   # DEV-8: under "--"
            extra += filler.after_panel(panel) if hasattr(filler, "after_panel") else []         # DEV-11: cut notes
            extra += filler.run.flush_notes(name)             # ratio cells without an interval (RATIO-UNDEFINED)
            if panel == 1 and name in C3_TABLES:
                extra += filler.noninf_panel()                # DEV-9: C3's non-inferiority under the contrasts
            out.extend(fit_panels(extra))
            continue
        if structural and name == "battery-dense" and s.startswith(r"\caption{") and "SwiftKV rows" in s:
            line = line.replace("(its base weights are bit-identical).",
                                "(its base weights are bit-identical), at SwiftKV's own split "
                                + filler.swiftkv_split + r" (paired per document, BCa); SwiftKV's maps with \method's "
                                r"exact tail are an ablation, not the comparison.")
        out.append(line)
    if structural and name in WRAP_FIRST:
        out = wrap_first_cells(out, *WRAP_FIRST[name])
    if structural and name in SPLIT_BEFORE:
        out = split_floats(out, name, SPLIT_BEFORE[name])
    if structural and not longt:
        out = fit_height(out)
    if structural:
        out = [dev_wording(name, ln) for ln in out]
    if structural and name in DEAD_TABLES:
        out = drop_dead(out)
    return "\n".join(out)


# DEV-26 (review 2026-10-11): a battery table drops what its model's cells did not run at all: a panel-A column whose
# every cell is "--" (RULER "all" on Llama and Qwen3.5, LongBench on the Qwen tables), a panel-A row whose every value is
# "--" (the question-first rows where they were cut) and a contrast row with no n. The "Not run" notes still name them.
# A comment line records what was dropped, so the layout test can drop the same from the template (apply_dead).
DEAD_TABLES = ("battery-dense", "battery-hybrid", "battery-dense-qkn")
PANEL_A_SPEC = r"\begin{tabular}{@{}lrrrrrrrr@{}}"           # panel A after DEV-22
PANEL_A_GROUPS = [("RULER", (1, 2)), ("LongB.", (3,)), (r"$\Delta$NLL", (4, 5)), ("Decode", (6, 7)), ("Turn 2", (8,))]
DEAD_NOTE = "% DEV-26 dropped: "
WRAPPED_CELL = re.compile(r"^\\parbox\[t\]\{[^}]*\}\{\\raggedright\\hspace\{0pt\}(.*?)\\strut\}$")


def _cells(line):
    body = line.split(" % DEV-")[0]
    if "&" not in body or not body.rstrip().endswith("\\\\") or body.lstrip().startswith(("%", "\\multicolumn")):
        return None
    return [c.strip() for c in CELL_SPLIT.split(body.rstrip()[:-2])]


def _row_key(cells):
    first = WRAPPED_CELL.sub(lambda m: m.group(1), cells[0])
    return first + " & " + cells[1] if len(cells) == 7 else first


def _panels(lines):
    """(start, end) line spans of the panels: from a \\begin{tabular} to its \\end{tabular}."""
    spans, start = [], None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(r"\begin{tabular}"):
            start = i
        elif ln.lstrip().startswith(r"\end{tabular}") and start is not None:
            spans.append((start, i))
            start = None
    return spans


def drop_dead(lines):
    cols, rows = set(), []
    for k, (a, b) in enumerate(_panels(lines)):
        body = [c for c in (_cells(ln) for ln in lines[a:b]) if c]
        if lines[a].strip().startswith(PANEL_A_SPEC):
            mid = next(i for i in range(a, b) if lines[i].strip() == r"\midrule")
            data = [c for c in (_cells(ln) for ln in lines[mid:b]) if c]
            cols = {j for j in range(1, 9) if all(c[j] == "--" for c in data)}
            rows += [_row_key(c) for c in data if all(x == "--" for x in c[1:])]
        else:
            rows += [_row_key(c) for c in body if len(c) == 7 and c[2] == "--" and c[3] == "--" and c[0] != "Contrast"]
    if not cols and not rows:
        return lines
    out = apply_dead(lines, sorted(cols), rows)
    return out + [DEAD_NOTE + "cols=" + ",".join(map(str, sorted(cols))) + "; rows=" + " | ".join(rows)]


def apply_dead(lines, cols, rows):
    """Drop panel-A columns `cols` (indices of the DEV-22 layout) and the rows whose key is in `rows` (panel A: the arm;
    a contrast panel: contrast and metric), rebuilding panel A's spec and two header lines."""
    rows, out, in_a = set(rows), [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith(PANEL_A_SPEC):
            in_a = True
            if cols:
                ln = ln.replace(PANEL_A_SPEC, r"\begin{tabular}{@{}l" + "r" * (8 - len(cols)) + "@{}}")
        elif s.startswith(r"\end{tabular}"):
            in_a = False
        c = _cells(ln)
        if c is not None and _row_key(c) in rows:
            continue
        if in_a and cols:
            keep = [(lab, [j for j in js if j not in cols]) for lab, js in PANEL_A_GROUPS]
            keep = [(lab, js) for lab, js in keep if js]
            if s.startswith(r"& \multicolumn{2}{c}{RULER}"):
                ln = " & " + " & ".join(lab if len(js) == 1 else rf"\multicolumn{{{len(js)}}}{{c}}{{{lab}}}"
                                        for lab, js in keep) + r" \\"
            elif s.startswith(r"\cmidrule(lr){2-3}"):
                pos, parts = 2, []
                for i, (lab, js) in enumerate(keep):
                    parts.append(rf"\cmidrule({'l' if i == len(keep) - 1 else 'lr'}){{{pos}-{pos + len(js) - 1}}}")
                    pos += len(js)
                ln = "".join(parts)
            elif c is not None and len(c) == 9:
                body, sep, note = ln.partition(" % DEV-")
                cells = CELL_SPLIT.split(body.rstrip()[:-2])
                ln = "&".join(x for j, x in enumerate(cells) if j not in cols).rstrip() + " \\\\" + sep + note
        out.append(ln)
    return out


# DEV-18: with real values several panels are wider than the text (battery-dense panel B by about 245 pt, built
# 2026-10-10); each tabular is set in a box and scaled down to \linewidth only when it is wider (graphicx \resizebox,
# which the paper loads); a panel that fits is set as before. Longtables cannot be boxed: SPACING narrows them.
FIT_OPEN = [r"\ifdefined\genfitbox\else\newsavebox{\genfitbox}\fi% DEV-18", r"\sbox{\genfitbox}{% DEV-18"]
FIT_CLOSE = [r"}% DEV-18", r"\ifdim\wd\genfitbox>\linewidth\resizebox{\linewidth}{!}{\usebox{\genfitbox}}"
             r"\else\usebox{\genfitbox}\fi\par% DEV-18"]

# DEV-19: battery-dense with real values is taller than the page (by 153 pt after DEV-18, pdflatex 2026-10-10). Every
# table float's panels (from its first tabular to its caption) are set in one box, the caption and label in another;
# when both together are taller than \textheight the panels are scaled down to the height left beside the caption,
# otherwise they are set as before.
TALL_OPEN = [r"\ifdefined\genfitall\else\newsavebox{\genfitall}\newsavebox{\genfitcap}\fi% DEV-19",
             r"\setbox\genfitall\vbox{\hsize\linewidth% DEV-19"]
TALL_CAPTION = [r"}% DEV-19", r"\setbox\genfitcap\vbox{\hsize\linewidth% DEV-19"]
TALL_CLOSE = [r"}% DEV-19",
              r"\ifdim\dimexpr\ht\genfitall+\dp\genfitall+\ht\genfitcap+\dp\genfitcap\relax>\textheight"
              r"\leavevmode\resizebox*{!}{\dimexpr\textheight-\ht\genfitcap-\dp\genfitcap-\baselineskip\relax}"
              r"{\copy\genfitall}\par\else\copy\genfitall\fi\copy\genfitcap% DEV-19"]


def fit_height(lines):
    """DEV-19 on the float that holds the caption (after a DEV-21 split: its first part, DEV-27)."""
    cap = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].startswith(r"\caption{")), None)
    start = max((i for i, ln in enumerate(lines) if ln == SPLIT_BEGIN and (cap is None or i < cap)), default=0)
    first = next((i for i, ln in enumerate(lines) if ln == FIT_OPEN[0] and i >= start), None)
    if first is None or cap is None or cap < first or not lines[cap + 1].startswith(r"\label{tab:"):
        return lines
    while first > 0 and lines[first - 1].endswith("% DEV-5"):        # the panel's spacing goes inside the box too
        first -= 1
    return lines[:first] + TALL_OPEN + lines[first:cap] + TALL_CAPTION + lines[cap:cap + 2] + TALL_CLOSE + \
        lines[cap + 2:]

# DEV-20: what makes the battery tables wide is their first column (arm names, contrast names, the follow-up panels'
# titles: battery-dense panel B was 718 pt against a 470 pt text width). In those tables a first cell longer than its
# column's width is set as a top-aligned paragraph of that width (wrap_cell, as DEV-6); shorter cells are unchanged.
# Widths: panel A (arms), then every later panel.
WRAP_FIRST = {"battery-dense": ("4.6cm", "4.6cm"), "battery-hybrid": ("4.6cm", "4.6cm"),
              "battery-dense-qkn": ("4.6cm", "4.6cm")}      # panel A: 4.6cm since DEV-22/26 narrowed it
CHARS_PER_CM = 6.0                                    # \footnotesize Latin Modern: about 6 characters per cm


def visible_len(cell):
    t = cell.replace(r"\method", "RidgeFill").replace(r"\llamashort", "Llama-3.1-8B")
    t = re.sub(r"\\[A-Za-z]+\*?", "", t)
    return len(re.sub(r"[{}$]", "", t).strip())


def wrap_first_cells(lines, first_width, rest_width):
    out, boxed, panel = [], False, -1
    for ln in lines:
        if ln == FIT_OPEN[1]:
            boxed, panel = True, panel + 1
        elif ln == FIT_CLOSE[0]:
            boxed = False
        body = ln.split(" % DEV-")[0] if " % DEV-" in ln else ln
        if boxed and "&" in body and body.rstrip().endswith("\\\\") and not body.lstrip().startswith(("%", "\\multicolumn")):
            width = first_width if panel == 0 else rest_width
            head, sep, tail = ln.partition(" & ")
            if visible_len(head) > float(width[:-2]) * CHARS_PER_CM:
                ln = wrap_cell(head, width) + sep + tail
        out.append(ln)
    return out

# DEV-21: with real values a battery table is taller than a page even with DEV-20 (battery-dense: 1,196 pt of panels and
# a 157 pt caption against a 650 pt text height). It is set as consecutive floats of the same table, split before the
# panels named here (0 = the arms panel, 1 = the declared contrasts, 2 = the first follow-up panel the filler appends).
# Each earlier part ends with "Table N, part k of n" (a \ref to the table's own label, so no second caption or anchor);
# the caption and the label stay under the last part, so every \ref to the table is unchanged.
# The first part keeps the template's [p]: it must fill at least \floatpagefraction (half a page), so the two smaller
# tables split after their contrasts.
SPLIT_BEFORE = {"battery-dense": (1, 2), "battery-hybrid": (2,), "battery-dense-qkn": (2,)}
SPLIT_BEGIN = r"\begin{table}[tp] % DEV-21"            # a part may fill a float page of its own


def split_floats(lines, name, before):
    starts = [i for i, ln in enumerate(lines) if ln == FIT_OPEN[0]]
    label = next((re.search(r"\\label\{([^}]*)\}", ln).group(1) for ln in lines if ln.startswith(r"\label{tab:")), None)
    if label is None or max(before) >= len(starts):
        raise Refused(f"DEV-21: {name} has {len(starts)} panels and label {label!r}; the split is before {before}")
    tmpl = TEMPLATES[name].split("\n")
    b = tmpl.index(next(ln for ln in tmpl if ln.startswith(r"\begin{table}")))
    pre = tmpl[b + 1:]
    pre = [ln for ln in pre[:next(i for i, ln in enumerate(pre) if ln.startswith(r"\begin{tabular}"))]
           if not ln.startswith("%")]                     # the float's own settings (\centering, font, spacing)
    n = len(before) + 1
    cuts = []
    for k in before:
        i = starts[k]
        while i > 0 and lines[i - 1].endswith("% DEV-5"):
            i -= 1
        cuts.append(i)
    # DEV-27 (review 2026-10-11): the caption and label go under the first part, so the reader meets the caption with
    # the table's first panel; each later part ends with "Table N (continued), part k of n"
    cap = next(i for i, ln in enumerate(lines) if ln.startswith(r"\caption{"))
    caption = lines[cap:cap + 2]
    lines = lines[:cap] + lines[cap + 2:]
    out, prev = [], 0
    for part, i in enumerate(cuts, 1):
        end = caption if part == 1 else [rf"\par\smallskip{{\small Table~\ref{{{label}}} (continued), part {part} of "
                                         rf"{n}.\par}} % DEV-21"]
        out += lines[prev:i] + end + [r"\end{table} % DEV-21", SPLIT_BEGIN] + \
            [ln + " % DEV-21" for ln in pre if ln.strip()]
        prev = i
    last = max(i for i, ln in enumerate(lines) if ln == r"\end{table}")
    return out + lines[prev:last] + [rf"\par\smallskip{{\small Table~\ref{{{label}}} (continued), part {n} of "
                                     rf"{n}.\par}} % DEV-21"] + lines[last:]


def fit_panels(lines):
    """DEV-18 for the panels a filler appends whole (C3 non-inferiority, matched depth, fresh data): each tabular in
    them is boxed and scaled like the template's own."""
    out = []
    for ln in lines:
        if ln.lstrip().startswith(r"\begin{tabular}"):
            out.extend(FIT_OPEN)
        out.append(ln)
        if ln.lstrip().startswith(r"\end{tabular}"):
            out.extend(FIT_CLOSE)
    return out


SWIFT_ABLATION = r"ablation: SwiftKV maps + \method's exact tail"
SWIFT_NATIVE = "SwiftKV as released"
C3_TABLES = {table for table, _ in C3_NONINF.values()}
_LEVEL1 = f"{100 * (1 - S.ALPHA):g}\\%"
# DEV-23 (review 2026-10-11): the bound as stats.noninferiority computes it: the more conservative of the one-sided BCa
# bound and the zero-regression binomial floor, which alone decides (and is an exact bound) when no sample is worse.
NONINF_RULE = (rf"$\Delta$ and its one-sided {_LEVEL1} lower bound: the more conservative of the BCa bound and the binomial"
               rf" floor $-100\,(1-{S.ALPHA:g}^{{1/n}})$ points (Method names the one that decides); when no sample is"
               r" worse than the reference, the floor alone is used and is an exact bound.")
_C3_CAPTION = r" Last panel: non-inferiority of the rows arms against exact (the margin declared before the run): " + NONINF_RULE
_C3_CAPTION_LLAMA = _C3_CAPTION.replace("(the margin declared before the run)",
                                        "(follow-up, declared 2026-10-08 after a readout of the data: post hoc; the "
                                        "margin is the RULER cell's)")
# \paperdata{paper text}{data text} and \paperonly{text}: main.tex prints the first, data.tex (the results-data companion,
# which holds data and plain descriptions only) the second (DEV-25)
_NEG_LABELS = [  # (template label, paper label, data label)
    (r"Hybrid wash: \method ties identity", r"\method ties identity on NLL", r"\method vs identity, hybrid"),
    ("SwiftKV's trained maps beat the ridge fit", "SwiftKV's trained maps beat the ridge fit", "SwiftKV maps vs ridge fit"),
    ("Calibration data size is flat", "Calibration data size is flat", "Calibration data size"),
    ("GDN terminal-state correction does nothing", "GDN terminal-state correction shows no gain",
     "GDN terminal-state correction"),
    ("Weighted projector fit loses needles", "Weighted projector fit loses needles", "Weighted projector fit"),
    ("Refresh layer loses to rows at equal compute", "Refresh layer loses to rows at equal compute",
     "Refresh layer vs rows"),
]
# the lever of the paper's lever table (paper/LEVERS-PAPER.md) that holds each development row (rows 3-6)
NEG_EVIDENCE = {2: 16, 3: 39, 4: 23, 5: 38}


def neg_evidence(row):
    n = NEG_EVIDENCE[row]
    return rf"\paperdata{{Table~\ref{{tab:levers}}, lever {n}}}{{the paper's lever {n}}}"


# DEV-10: wording of the generated tables (the templates are the skeleton's bytes). The declared criteria are one-sided
# margins, so the tests are non-inferiority tests, not equivalence (TOST); the T1.4 rows put the question before the
# document, which the document's hidden states still reach through causal attention: a question-POSITION test, not a
# question-blind one. Labels only; the data keys (QFIRST_ROWS, qfirst_*) are unchanged. Applied to every generated line.
# DEV-22 to DEV-26 (the two reviews of 2026-10-11): no TTFT on the reference models (columns dropped, drop_ttft), no
# internal ids or verdict prose in captions, "sample" for a RULER item, data-document variants (\paperdata).
WORDING = {
    "battery-dense": [(", q-blind", ", question-first (position)"),
                      # the comparison is SwiftKV as released vs RidgeFill at S16; SwiftKV's maps + our exact tail is an
                      # ablation (its role cell: battery_panel_b)
                      ("SwiftKV maps + exact tail", SWIFT_ABLATION), ("SwiftKV native", SWIFT_NATIVE),
                      (r"$^\ast$: unequal cost (labelled, never a claim).", r"$^\ast$: a comparison at unequal cost."),
                      ("Top: one row per arm.", "First panel: one table row per arm."),
                      ("Bottom: declared contrasts;", "Second panel: declared contrasts;"),
                      (r"SwiftKV rows are measured against \llama's exact (its base weights are bit-identical), at "
                       r"SwiftKV's own split ", r"The SwiftKV arms are measured against \llama's exact\paperonly{ (its base "
                       r"weights are bit-identical)}. The SwiftKV comparison is SwiftKV as released, which has no exact "
                       r"tail, minus \method, which also runs its exact tail at full depth, both at SwiftKV's own split "),
                      (r"exact tail are an ablation, not the comparison.", r"exact tail are an ablation."),
                      ("MDE: minimum detectable effect.}", "MDE: minimum detectable effect. $p\\le$: the uncorrected $p$ "
                       "was at the bootstrap's smallest value." + _C3_CAPTION_LLAMA + "}")],
    "negative": [("SwiftKV maps + exact tail $-$", SWIFT_ABLATION + " $-$"),
                 (r"\begin{tabular}{@{}p{3.1cm}lp{1.5cm}p{3.3cm}rcl@{}}",
                  r"\begin{tabular}{@{}p{3.0cm}lp{2.4cm}p{3.1cm}rcp{1.5cm}@{}}"),
                 ("Finding & Model & Set, protocol", r"\paperdata{Finding}{Comparison} & Model & Set, protocol"),
                 (" & both & ", r" & \llamashort{} + \qwenhybridshort & "),
                 (r" & Q1b, P4 \\", " & " + neg_evidence(3) + r" \\"), (r" & WF1 \\", " & " + neg_evidence(4) + r" \\"),
                 (r" & FQ3 \\", " & " + neg_evidence(5) + r" \\"),
                 (r"\caption{Negative results. Rows", r"\caption{\paperdata{Negative results.}{Development-set and pre-fix "
                  r"comparisons.} Rows"),
                 ("Evidence names the table or the ledger record.}", r"Evidence names the table or the lever that holds "
                  r"the row; ``\llamashort{} + \qwenhybridshort'': both models pooled.}")]
                + [(old + " & ", rf"\paperdata{{{paper}}}{{{data}}} & ") for old, paper, data in _NEG_LABELS],
    "battery-dense-qkn": [(", q-blind", ", question-first (position)"),
                          ("long documents are skipped.}", "long documents are skipped. RULER ``all'': every scored "
                           "sample of all eight RULER tasks." + _C3_CAPTION + "}")],
    "battery-hybrid": [(", q-blind", ", question-first (position)"),
                       (r"same layout as Table~\ref{tab:battery-dense}. TTFT is measured on \llama only.}",
                        r"same layout as Table~\ref{tab:battery-dense}." + _C3_CAPTION + "}")],
    "splitcurve": [(r" with measured TTFT on \llamashort.", "."),
                   ("; a reversal is an interval above zero.", "; an interval above zero is a significant rise.")],
    "mechanism": [(r" & demonstration & ", r" & \TBDcell & "),
                  (r"Bottom: the per-architecture repair table; on a QK-normed model the key-scale correction cannot act "
                   r"(it is a demonstration, not a test).}",
                   r"Bottom: one row per model. Loss carried by: keys or values when the interval of the difference of "
                   r"their shares excludes zero, neither when neither share's interval lies above zero. Restore and key "
                   r"scale: yes, partial or no from the share's interval (the share in parentheses).}")],
    "mde": [("then the engine's equivalence contrasts", "then the engine's non-inferiority contrasts"),
            (r" & ppl ratio & ", r" & ppl gap (\%) & "),
            ("from the paired standard deviation. The last row shows why single RULER cells are descriptive only.}",
             r"from the paired standard deviation (in the metric's unit). The last row is one task at one length on "
             r"\llamashort, at the median per-cell number of filled-region samples\paperonly{: single RULER cells are "
             r"descriptive only}.}")],
    "persample": [(r"\begin{longtable}{@{}llrrrrrrr@{}}", r"\begin{longtable}{@{}lrrrrrr@{\hspace{10pt}}lrrrrrr@{}}"),
                  (r"Model & Document & Tokens & \method & identity & no fill & POP nat. & \method+rows & id.+rows \\",
                   r"Doc. & RF & id. & nf & POP & RF+r & id.+r & Doc. & RF & id. & nf & POP & RF+r & id.+r \\"),
                  (r"\caption{Per-document NLL delta against exact (novel tokens, nats per token) at the primary split "
                   r"and tail.}",
                   r"\caption{Per-document NLL delta against exact (novel tokens, nats per token) at the primary split "
                   r"and tail, two documents per line, grouped by model and sorted by domain and length. Doc.: domain, "
                   r"nominal length and index of the document (encyc.: the encyclopedia domain, files tech\_doc\_*; code: "
                   r"source\_code\_*); the documents' token counts are in the released result files. RF: \method; id.: identity; nf: no "
                   r"fill; POP: POP native; +r: + full-depth rows, from the second run (causal scoring) against that "
                   r"run's own exact.}"),
                  (r"\caption{Per-cell RULER accuracy (filled-region needles, points) at the primary split and tail.}",
                   r"\caption{Per-cell RULER accuracy (filled-region samples, points) at the primary split and tail. A "
                   r"cell with no filled-region sample is omitted; $n$: the cell's filled-region samples.}"),
                  (r"\begin{longtable}{@{}lllrrrrrr@{}}", r"\begin{longtable}{@{}lllrrrrrrr@{}}"),
                  (r"Model & Task & Length & exact &", r"Model & Task & Length & $n$ & exact &")],
    "levers": [(r"\begin{longtable}{@{}rp{4.2cm}p{2.6cm}p{5.2cm}l@{}}",
                r"\begin{longtable}{@{}rp{3.2cm}p{3.0cm}p{6.4cm}l@{}}"),
               ("Every development lever, one line each. Development-set evidence only; none is used for a claim.}",
                r"Every development lever, one line each. Development-set evidence only; none is used for a claim. "
                r"E: \method in the \flashnext engine before its determinism fix, run noise about $\pm$0.01 nats per "
                r"document (protocol v1.0.2). Evidence: ledger records and experiment cards of the released "
                r"repository.}")],
}
NUMBERS_SRC = {"resSwiftMapsMinusRidge": "battery-dense panel B, ABLATION row (SwiftKV maps + RidgeFill's exact tail - "
                                          "RidgeFill at S16, dNLL novel; not the R6 comparison: resSwiftNativeMinusRidge)",
               "resSwiftMapsMinusRidgeLo": "same ablation contrast, BCa lower bound",
               "resSwiftMapsMinusRidgeHi": "same ablation contrast, BCa upper bound",
               "resSwiftNativeNll": "battery-dense panel A, SwiftKV as released (swiftkv-native at S16), dNLL novel vs exact",
"resQwenBlindRowsRuler": "battery-hybrid question-blind rows (DEV-12), identity+qblind (noproj+qblind10), "
                                         "RULER filled (protocol 1.1.3; before it this macro read the question-first "
                                         "T1.4 runs, a question-position test)"}
NUMBERS_WORDING = [("(TOST)", "(non-inferiority)"), ("(question-blind)", "(question-first, position)")]


def dev_wording(name, line):
    for old, new in WORDING.get(name, ()):
        line = line.replace(old, new)
    if name in TTFT_DROP:
        line = drop_ttft(name, line)
    return line


# DEV-22 (Revision 3, C2'; review 2026-10-11): TTFT is measured on the engine only, so the reference tables' TTFT columns
# (every cell "--") are dropped: the battery tables' last column, the split curve's last three. A pure function of one
# line, so a template line and its generated line drop the same cells (the layout test applies dev_wording to both).
TTFT_DROP = {"battery-dense": 1, "battery-hybrid": 1, "battery-dense-qkn": 1, "splitcurve": 3}
TTFT_FULL = 10                                                # the template's column count of those panels
_BAT_TTFT = [(r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\begin{tabular}{@{}lrrrrrrrr@{}}"),
             (r" & Turn 2 & TTFT \\", r" & Turn 2 \\"), (r"\cmidrule(lr){9-9}\cmidrule(l){10-10}", r"\cmidrule(l){9-9}")]
TTFT_HEAD = {"battery-dense": _BAT_TTFT, "battery-hybrid": _BAT_TTFT, "battery-dense-qkn": _BAT_TTFT,
             "splitcurve": [(r"\begin{tabular}{@{}lrrrrccrrr@{}}", r"\begin{tabular}{@{}lrrrrcc@{}}"),
                            (r" & \multicolumn{3}{c}{TTFT speedup (16K)} \\", r" \\"),
                            (r"\cmidrule(lr){7-7}\cmidrule(l){8-10}", r"\cmidrule(l){7-7}"),
                            (r"\multicolumn{10}{@{}l}", r"\multicolumn{7}{@{}l}")]}
CELL_SPLIT = re.compile(r"(?<!\\)&")


def drop_ttft(name, line):
    for old, new in TTFT_HEAD[name]:
        line = line.replace(old, new)
    body, sep, note = line.partition(" % DEV-")
    if "&" not in body or not body.rstrip().endswith("\\\\") or body.lstrip().startswith("%"):
        return line
    cells = CELL_SPLIT.split(body.rstrip()[:-2])
    if len(cells) != TTFT_FULL:
        return line
    return "&".join(cells[:-TTFT_DROP[name]]).rstrip() + " \\\\" + sep + note


# DEV-5: the skeleton sized these panels with "TBD" placeholders; with real-width values (signed intervals, config
# names, hashes) they overflow the text width (measured with pdflatex on the mock-filled paper: battery panel B 18-23 pt,
# mechanism 26 pt, configs 260 pt, negative 55 pt). Only spacing and font size change; columns, rows and headers do not.
# negative: still 14 pt over at \footnotesize with 2 pt column gaps (its three p{} columns are fixed by the skeleton),
# so it is set in \scriptsize (2026-10-04 re-measure; the narrower p{} widths are a skeleton change, see tex2-NOTES).
SPACING = {
    ("battery-dense", 1): r"\setlength{\tabcolsep}{2pt}",
    ("battery-dense-qkn", 1): r"\setlength{\tabcolsep}{2pt}",
    ("battery-hybrid", 1): r"\setlength{\tabcolsep}{2pt}",
    ("mechanism", 0): r"\setlength{\tabcolsep}{2pt}\footnotesize",
    ("negative", 0): r"\setlength{\tabcolsep}{2pt}\scriptsize",
    ("configs", 0): r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}",
    ("levers", 0): r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}",       # DEV-25: compact (review 2026-10-11)
    # the NLL longtable is 10 pt over the text width at the skeleton's 4pt column gaps (pdflatex, 2026-10-10)
    ("persample", 0): r"\begingroup\scriptsize\setlength{\tabcolsep}{1.5pt}",   # DEV-25: two documents per line
}

ERA_BANNER = (r"\multicolumn{7}{@{}p{0.95\linewidth}@{}}{\textit{Table rows 1--2 are locked finals under protocol "
              r"VERSIONS; table rows 3--6 are development evidence of earlier eras, each labelled; no contrast or "
              r"interval crosses eras; change log in Appendix~\ref{app:protocol}.}} \\")


def locked_versions(tree):
    """The protocol versions of the locked-set finals cells (the battery scope; the engine tier's are its own)."""
    return sorted({v for n, c in tree.cells.items() if scope_of(c) == "battery" for v in tree.cell_versions.get(n, ())},
                  key=version_key)


def stamp(name, run, header):
    """Line 1 of a generated table: what generated it, from what, at which evidence tiers (cells.json: per cell)."""
    tiers = sorted({c["tier"] for c in run.cellslog if c.get("table") == name})
    return (f"% tables/{name}.tex -- GENERATED by finals/tables.py from results tree {header['tree'][:16]}, "
            f"ledger {header['ledger'][:16]}, protocol {header['protocol']} ({header['mock']}); evidence tiers: "
            f"{', '.join(tiers)}; never edit by hand.")


# ============================================================================================ main
def compute(args):
    """Every table, numbers.tex and the missing list, nothing written: (run, texts, numbers). run.missing lists every
    required cell without data (finals/align_audit.py reads it on a tree laid out as the declared cells)."""
    tree = Tree(args.results, args.ledger, args.protocol).load()
    repo = Path(args.repo)
    out_dir = Path(args.out)
    check_pending(tree, getattr(args, "pending", None) or (), out_dir)
    run = Run(tree, Stats(tree), repo, args)
    _roles()
    makers = {   # in this order: mde and the numbers read the flashnext filler
        "battery-dense": lambda: BatteryFiller(run, "dense"), "battery-dense-qkn": lambda: BatteryFiller(run, "qkn"),
        "battery-hybrid": lambda: BatteryFiller(run, "hybrid"), "splitcurve": lambda: SplitFiller(run),
        "mechanism": lambda: MechFiller(run), "flashnext": lambda: FlashFiller(run), "negative": lambda: NegFiller(run),
        "mde": lambda: MdeFiller(run), "persample": lambda: PerSampleFiller(run), "levers": lambda: LeversFiller(run),
        "configs": lambda: ConfigsFiller(run),
    }
    run.fillers, texts = {}, {}
    for name, make in makers.items():
        run.table = name                                      # every cell (captions too) is logged under its table
        f = run.fillers[name] = make()
        if name == "battery-dense":
            f.swiftkv_split = str(split_layer_or_tbd(run, "llama-3.1-8b-instruct", "1/2"))
        texts[name] = render(name, f)
    run.table = "numbers.tex"
    fill_macros(run)
    if not (out_dir / "numbers.tex").is_file():               # --no-ledger without the paper's skeleton (main())
        src = lambda n: run.macro_src.get(n, "").replace("%", "pct")             # noqa: E731
        return run, texts, "".join(f"\\newcommand{{\\{n}}}{{{v}}}%{GEN}{src(n)}\n" for n, v in sorted(run.macros.items()))
    numbers = (out_dir / "numbers.tex").read_text()
    new_numbers, unknown = rewrite_numbers(numbers, run.macros, run.macro_src,
                                           {**NEW_MACROS, **getattr(run, "new_notes", {})})
    for u in unknown:
        run.missing.append((f"numbers.tex \\{u}", "a result macro the generator does not compute (add it to "
                            "fill_macros, never type it)"))
    return run, texts, new_numbers


FN_NAMES = ("flashnext", "flash-next", "flash_next", FN_MODEL)


def check_pending(tree, pending, out_dir):
    """--pending TIER only while the tier has not run: refused when any of its data or its FINAL- record exists, and
    when the paper does not define the marker."""
    for tier in pending:
        if tier not in PENDING_TIERS:
            raise Refused(f"--pending {tier}: only {list(PENDING_TIERS)} can be pending")
        found = [f"cell {c['name']}" for c in tree.cells.values() if scope_of(c) == "flashnext" or
                 c["model"]["name"] == FN_MODEL]
        found += [f"results directory {d.name}" for d in sorted(tree.results.iterdir())
                  if any(k in d.name.lower() for k in FN_NAMES)]
        found += [f"ledger record {r.get('id') or r.get('label')}" for r in tree.ledger
                  if str(r.get("id") or r.get("label") or "").startswith("FINAL-")
                  and any(k in json.dumps(r).lower() for k in FN_NAMES + ("flash",))]
        if found:
            raise Refused(f"--pending {tier} refused: the tier has data or a FINAL- record ({'; '.join(found)}); a "
                          "pending marker never stands in for real results")
        main_tex = out_dir / "main.tex"
        if not main_tex.is_file() or not re.search(r"\\(re)?newcommand\{?\\pending\}?", main_tex.read_text()):
            raise Refused(f"--pending {tier}: {main_tex} does not define \\pending (the marker every pending cell "
                          "prints)")


def build(args):
    out_dir = Path(args.out)
    run, texts, new_numbers = compute(args)
    tree = run.t
    if tree.is_mock() and out_dir.resolve() == (REPO / "paper" / "tex").resolve():
        raise Refused("a FINAL-MOCK tree never writes the paper's own tables (paper/tex); use --out on a copy")
    if run.missing and tree.ledger_path is not None:
        raise Refused("\n".join([f"TABLES REFUSED: {len(run.missing)} required cell(s) missing; nothing written."]
                                + missing_lines(run.missing)))
    if run.missing:
        print("\n".join([f"tables: --no-ledger: {len(run.missing)} cell(s) not computed (\\TBDcell), by reason:"]
                        + missing_lines(run.missing)))
    header = {"tree": atomic.treehash(tree.results),
              "ledger": atomic.sha256_file(tree.ledger_path) if tree.ledger_path is not None else "no-ledger",
              "protocol": ",".join(sorted(tree.versions)), "mock": "MOCK, not evidence" if tree.is_mock() else "finals"}
    if run.pending:
        header["pending"] = sorted(run.pending)
    for name, text in texts.items():
        lines = text.split("\n")
        lines[0] = stamp(name, run, header)
        atomic.write_bytes(out_dir / "tables" / f"{name}.tex", "\n".join(lines).encode())
    atomic.write_bytes(out_dir / "numbers.tex", new_numbers.encode())
    missing = {"missing": [{"cell": w, "reason": why} for w, why in run.missing]} if run.missing else {}
    atomic.write_bytes(out_dir / "tables" / "cells.json",
                       (json.dumps({"generator": "finals/tables.py", **header, "seed": S.SEED,
                                    "n_resamples": S.N_RESAMPLES, "cells": run.cellslog, **missing}, indent=1)
                        + "\n").encode())
    left = [ln.split("}")[0].split("{\\")[1] for ln in new_numbers.splitlines()
            if ln.startswith(r"\newcommand") and r"\TBD{" in ln]
    return run, left


def missing_lines(missing):
    """The missing cells grouped by reason, largest group first."""
    groups = {}
    for where, why in missing:
        groups.setdefault(why, []).append(where)
    out = []
    for why, wheres in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        out.append(f"  [{len(wheres)}] {why}")
        out.extend(f"        {w}" for w in wheres)
    return out


def split_layer_or_tbd(run, model, frac):
    try:
        return split_layer(run, model, frac)
    except Missing as e:
        run.missing.append(("battery-dense caption SwiftKV split", str(e)))
        return TBD


LEAK_MODELS = {"llama-3.1-8b-instruct": "resLlama", "qwen3.5-9b": "resQwen", "qwen3-8b": "resQwenDense"}
LEAK_ARMS = [("noproj+qattn10", "IdRows"), ("kva+qattn10", "FillRows"), ("kva+ktail", "Ktail"), ("kva+vtail", "Vtail")]
assert [a for a, _ in LEAK_ARMS] == list(SUPERSESSION["arms"])


def _c3_macro_notes():
    out = {}
    for mod, pre in QBLIND_MACROS.items():
        for _, arm, _, rarm, tag in QBLIND_ROWS:
            what = f"{mod} question-blind rows (DEV-12), {arm}"
            out[f"{pre}{tag}BlindRuler"] = f"{what}, RULER filled (points)"
            out[f"{pre}{tag}BlindMinusRows"] = f"{what} - {rarm} (question-attended), RULER filled, Delta (scope)"
            out[f"{pre}{tag}BlindMinusRowsLo"] = "same, BCa lower bound"
            out[f"{pre}{tag}BlindMinusRowsHi"] = "same, BCa upper bound"
    for label, a, b, subset, role, stem in LLAMA_ON_ITS_OWN:
        what = (f"battery-dense DEV-14 row, Llama RULER filled {a[0]}@{a[1]} - {b[0]}@{b[1]}"
                + (" on the 64K NIAH rows only (reported on its own)" if subset else
                   " at the 1/2 split (follow-up, declared 2026-10-07 after FINAL-LLAMA-NLL, post hoc)")
                + "; outside every primary and Holm family")
        out[f"resLlamaRuler{stem}"] = what + ": Delta (points)"
        out[f"resLlamaRuler{stem}Lo"] = "same, BCa lower bound"
        out[f"resLlamaRuler{stem}Hi"] = "same, BCa upper bound"
        out[f"resLlamaRuler{stem}N"] = "same, n (the rows that carry both arms)"
    for mod, pre in DEEP.items():
        what = (f"{mod} ruler-deep cell (DEV-15): matched depth vs POP as published, NIAH 8K-32K filled region; "
                "follow-up declared 2026-10-08 (post hoc), outside every primary / Holm / MDE family")
        for arm, split, tag in DEEP_ARMS:
            out[f"{pre}Deep{tag}Ruler"] = f"{what}: {arm}@{split} RULER filled (points)"
        out[f"{pre}DeepFillMinusPop"] = f"{what}: kva@5/8 - pop-native Delta (points)"
        out[f"{pre}DeepFillMinusPopBound"] = (f"{what}: one-sided 95% lower bound of kva@5/8 - pop-native at the "
                                              f"registered {RULER_DEEP_MARGIN:g} point margin (BCa or the exact "
                                              "binomial floor, whichever is more conservative)")
        out[f"{pre}DeepFillMinusPopN"] = f"{what}: n of kva@5/8 - pop-native"
        out[f"{pre}DeepFillMinusPopVerdict"] = f"{what}: non-inferiority met / not met at {RULER_DEEP_MARGIN:g} points"
        out[f"{pre}DeepFillMinusId"] = f"{what}: kva - noproj at 5/8, Delta (points)"
        out[f"{pre}DeepFillMinusIdLo"] = "same, BCa lower bound"
        out[f"{pre}DeepFillMinusIdHi"] = "same, BCa upper bound"
    for mod, pre in FRESH.items():
        what = (f"{mod} nll-fresh cell (DEV-16): fresh-data confirmation, its own frozen set (never pooled with the "
                "locked set)")
        for _, met, a, b, stem, role in FRESH_ROWS:
            out[f"{pre}{stem}"] = f"{what}: {a[0]}@{a[1]} - {b[0]}@{b[1]}, {met}, Delta (nats)"
            out[f"{pre}{stem}Lo"] = "same, BCa lower bound"
            out[f"{pre}{stem}Hi"] = "same, BCa upper bound"
            if role == "holm":
                out[f"{pre}{stem}Holm"] = "same, Holm-adjusted p over the two ridgefill - identity contrasts"
        out[f"{pre}FreshDocs"] = f"{what}: n (documents)"
    for nm, what in (("resSwiftNativeMinusRidge", "battery-dense panel B, R6 row: SwiftKV as released (swiftkv-native) - "
                                                  "RidgeFill at S16, dNLL novel, paired per document"),
                     ("resSwiftNativeMinusRidgeLo", "same R6 contrast, BCa lower bound"),
                     ("resSwiftNativeMinusRidgeHi", "same R6 contrast, BCa upper bound")):
        out[nm] = what
    for mod, pre in LEAK_MODELS.items():
        for arm, tag in LEAK_ARMS:
            what = (f"SUPERSEDED (protocol 1.1.3: read the scored continuation; size of the leak, never a claim): {mod} "
                    f"{arm} dNLL novel vs exact")
            out[f"{pre}Leak{tag}"] = (f"{what}, original ({SUPERSESSION['of']} cell, 1.1.1 evaluation) minus corrected "
                                      f"({SUPERSESSION['by']} cell), each vs its own cell's exact")
            out[f"{pre}Leak{tag}Orig"] = f"{what}, original ({SUPERSESSION['of']} cell)"
            out[f"{pre}Leak{tag}Fixed"] = f"{what}, corrected ({SUPERSESSION['by']} cell)"
    for mod, (table, pre) in C3_NONINF.items():
        for _, arm, tag in C3_ROWS:
            what = f"{table} non-inferiority panel (C3), {arm} - exact, RULER filled" + (
                f" (follow-up {C3_POSTHOC[mod]}, declared 2026-10-08, post hoc)" if mod in C3_POSTHOC else "")
            out[f"{pre}{tag}VsExact"] = f"{what}: Delta in points"
            out[f"{pre}{tag}VsExactBound"] = (f"{what}: one-sided 95% lower bound (BCa, or the exact binomial floor "
                                              "when that is more conservative)")
            out[f"{pre}{tag}VsExactN"] = f"{what}: n (the filled-region rows that carry both arms)"
            out[f"{pre}{tag}VsExactVerdict"] = f"{what}: met / not met against the RULER cell's declared margin"
    return out


# result macros the skeleton's numbers.tex does not define yet: the generator adds each once, after the anchor, with its
# src note; later runs replace its value like any other
NEW_MACROS = _c3_macro_notes()
NEW_MACRO_ANCHOR = "resQwenBlindRowsRuler"
SECTION_C = "% ===================================================================================================== C. results"


GEN = " | gen: "


def rewrite_numbers(text, values, sources=None, new=None):
    """numbers.tex with every generated macro's value replaced (one definition per line; the src note kept, with
    NUMBERS_WORDING applied, and a ' | gen: <tier; source>' suffix naming what this generator read, replaced on every
    run). new: {name: src note} of generated macros numbers.tex may not define yet; each one missing is added after the
    NEW_MACRO_ANCHOR line (else at the end).
    Returns (text, result macros of section C that still hold \\TBD and that the generator does not compute)."""
    out, unknown, in_c, defined, anchor = [], [], False, set(), None
    for line in text.split("\n"):
        if line.startswith(SECTION_C):
            in_c = True
        mt = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{", line)
        if mt:
            name = mt.group(1)
            defined.add(name)
            if name == NEW_MACRO_ANCHOR:
                anchor = len(out)
            i, depth = mt.end(), 1
            while depth:
                depth += {"{": 1, "}": -1}.get(line[i], 0)
                i += 1
            if name in values:
                rest = line[i - 1:]
                if GEN in rest:
                    rest = rest[:rest.index(GEN)]
                for old, wording in NUMBERS_WORDING:
                    rest = rest.replace(old, wording)
                if name in NUMBERS_SRC:
                    rest = re.sub(r"% src: .*", lambda _m: "% src: " + NUMBERS_SRC[name], rest, count=1)
                gen = (GEN + sources[name].replace("%", "pct")) if sources and name in sources else ""
                line = line[:mt.end()] + values[name] + rest + gen
            elif in_c and r"\TBD{" in line:
                unknown.append(name)
        out.append(line)
    add = [n for n in (new or {}) if n in values and n not in defined]
    if add:
        gen = lambda n: (GEN + sources[n].replace("%", "pct")) if sources and n in sources else ""   # noqa: E731
        at = len(out) if anchor is None else anchor + 1
        out[at:at] = [f"\\newcommand{{\\{n}}}{{{values[n]}}}% src: {new[n]}{gen(n)}" for n in add]
    return "\n".join(out), unknown


def main(argv=None):
    import os
    ap = argparse.ArgumentParser(description="finals table generator (the only path from results to the paper)")
    ap.add_argument("results", help="the verified finals results tree (one dir per cell)")
    ap.add_argument("--repo", default=str(REPO), help="repo root for pinned files (default: this repo)")
    ap.add_argument("--ledger", default=None, help="default <repo>/ledger/paper/ledger.jsonl (read-only)")
    ap.add_argument("--no-ledger", action="store_true",
                    help="a public copy of the results tree without the ledger (module doc): lists what it cannot "
                         "compute instead of refusing")
    ap.add_argument("--protocol", default=str(REPO / "protocol.json"))
    ap.add_argument("--out", default=str(REPO / "paper" / "tex"), help="the paper/tex directory to write into")
    ap.add_argument("--proj-dir", default=os.environ.get("KVA_PROJ_DIR"), help="final projectors (env KVA_PROJ_DIR)")
    ap.add_argument("--levers", default=None, help="curated lever list (default <repo>/paper/LEVERS-PAPER.md)")
    ap.add_argument("--battery-v10", default=None, help="v1.0 battery JSON (default results/tables/battery-v1.0.json)")
    ap.add_argument("--timing", nargs="*", default=None, help="timing JSONs (default results/timing-sweep-*.json)")
    ap.add_argument("--pending", action="append", choices=PENDING_TIERS, default=[],
                    help="a tier that has not run: its cells print \\pending (refused if it has data)")
    args = ap.parse_args(argv)
    if args.no_ledger and args.ledger:
        print("tables: --ledger and --no-ledger exclude each other", file=sys.stderr)
        return 2
    args.ledger = None if args.no_ledger else (args.ledger or str(Path(args.repo) / "ledger" / "paper" / "ledger.jsonl"))
    need = [(args.protocol, "protocol.json")] + ([] if args.no_ledger else
                                                 [(args.ledger, "ledger"), (Path(args.out) / "numbers.tex", "numbers.tex")])
    for p, what in need:
        if not Path(p).is_file():
            print(f"tables: {what} {p} does not exist" + (" (--no-ledger for a public copy of the tree)"
                                                          if what == "ledger" else ""), file=sys.stderr)
            return 2
    try:
        run, left = build(args)
    except Refused as e:
        print(str(e), file=sys.stderr)
        return 1
    pend = [c for c in run.cellslog if c.get("status") == "pending"]
    print(f"tables: wrote {len(TEMPLATES)} tables, {len(run.macros)} numbers.tex macros and tables/cells.json "
          f"({len(run.cellslog) - len(pend)} cells computed, {len(pend)} pending) into {args.out}")
    if pend:
        print(f"tables: PENDING (not computed, printed as \\pending): {len(pend)} cells and macros of "
              f"{', '.join(sorted(run.pending))} ({'; '.join(PENDING_NOTE[t] for t in sorted(run.pending))})")
    und = [c for c in run.cellslog if c.get("note")]
    if und:
        print(f"tables: {len(und)} ratio cell(s) without an interval (\"--\" with a table note)")
    if left:
        print(f"tables: numbers.tex placeholders outside the generator's scope (Dylan's decisions): {', '.join(left)}")
    return 0



# ======================================================================================= layout templates
# Verbatim copies of the skeleton's placeholder tables (paper/tex/tables/*.tex at commit 9afda4f), except the ones
# REPLACED_TEMPLATES names. The generator
# fills their \TBDcell slots; template mode (render(name, None)) reproduces each file byte for byte
# (tests/test_finals_tables.py pins that against git). Declared deviations, applied only when values are filled:
# DEV-1 flashnext knob panel (PLAN 'LaTeX skeleton DONE': the positive control is an all-prefix-rows arm, 100% of
# matches is a curve point), DEV-2 the battery-dense caption names the SwiftKV split, DEV-3 the era banner row of
# negative.tex (PROTOCOL.md section 9), DEV-4 longtables get \endfirsthead and a "(continued)" head, so a table
# that spans pages defines its label once, DEV-5 spacing and font size of the panels that overflow (SPACING), DEV-6
# the configs rows' long cells and the levers rows' Evidence cell wrap (wrap_cell), DEV-7 and DEV-8 (see MECH_NOT_ON_RULER
# and NOT_RUN_NOTE), DEV-9 the C3 non-inferiority panel under panel B of battery-hybrid and battery-dense-qkn (C3_NONINF)
# and a caption sentence for it, DEV-10 wording (WORDING: non-inferiority, not equivalence/TOST; "question-first
# (position)", not "q-blind"), DEV-11 protocol 1.1.3's cuts ("--" and a "Not run (cut ...)" note), DEV-12 the
# question-blind rows (qblind10), DEV-13 R6: SwiftKV as released vs RidgeFill at S16, SwiftKV's maps + RidgeFill's exact
# tail labelled an ablation (labels, role cells, caption), DEV-14 Llama's 64K-on-its-own and 1/2-split follow-up rows,
# DEV-15 the matched-depth panel (ruler-deep), DEV-16 the fresh-data confirmation panel (nll-fresh), DEV-17 the
# Flash-Next table is the C5 tier as run (2026-10-10; REPLACED_TEMPLATES): the skeleton's layout had arms, a knob panel
# and contrasts that were not run on the engine.
REPLACED_TEMPLATES = {"flashnext": "DEV-17: the C5 tier as run, 2026-10-10 (FINAL-FLASHNEXT-C5, amendments 10 and 11)"}
TEMPLATES = {
    "battery-dense": r"""% tables/battery-dense.tex -- GENERATED TABLE (placeholder written 2026-10-04; no result yet).
% The table generator overwrites this whole file from verified finals; never edit numbers here by hand.
% Every \TBDcell is a value the generator writes; "--" is a cell the declared design does not run.
% Number formats (all tables): signed deltas with an explicit sign; NLL in nats per token, 3 decimals; RULER and
% LongBench in points (0-100), 1 decimal; shares and agreement in percent, 1 decimal; TTFT speedups exact/arm,
% 2 decimals with \speedx; intervals as [lo, hi] from the BCa bootstrap unless the header says otherwise.
% Layout contract, panel A (one row per arm, rows in this order):
%   Arm | RULER all | RULER filled | LongBench avg | dNLL novel | dNLL key | Decode agree | Decode KL | Turn-2 RULER | TTFT speedup
%   RULER all / filled: mean points over the pooled cells (filled = needles at depth < N - T; the primary metric)
%   LongBench avg: mean of the eight task means, official metrics (secondary)
%   dNLL novel / key: arm minus exact, document-weighted mean over the long-document set (key = LongPPL-style key tokens)
%   Decode agree / KL: greedy-token agreement (%) and mean per-step KL(exact || arm) over the decode steps (T1.7)
%   Turn-2 RULER: second question on the cached prefix, accuracy in points (T1.8)
%   TTFT speedup: measured exact TTFT / arm TTFT at 16K, median over repeats (T3.1)
% Layout contract, panel B (one row per declared contrast, rows in this order):
%   Contrast | Metric | n (pairs) | Delta | 95% CI (BCa) | MDE | Role (primary / Holm = secondary, Holm-corrected / exploratory / result;
%   a trailing * marks a comparison at unequal cost, which the protocol requires to be labelled)
% Model: Llama-3.1-8B-Instruct, split 12, tail 4096 (caption values are generated). Rows 13-14 are Llama only.
\begin{table}[p]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{RULER} & LongB. & \multicolumn{2}{c}{$\Delta$NLL} & \multicolumn{2}{c}{Decode} & Turn 2 & TTFT \\
\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-9}\cmidrule(l){10-10}
Arm & all & filled & avg & novel & key & agree & KL & RULER & speedup \\
\midrule
Exact & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & \TBDcell & \TBDcell \\
No fill & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & \TBDcell \\
Identity & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & \TBDcell \\
POP native & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & \TBDcell \\
\method & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
Identity + rows & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & \TBDcell & -- \\
\method + rows & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & \TBDcell & -- \\
\method + random rows & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
\method + longer tail & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
No fill + longer tail & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
Identity + rows, q-blind & \TBDcell & \TBDcell & -- & -- & -- & -- & -- & -- & -- \\
\method + rows, q-blind & \TBDcell & \TBDcell & -- & -- & -- & -- & -- & -- & -- \\
SwiftKV maps + exact tail & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
SwiftKV native & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
\bottomrule
\end{tabular}

\medskip
\footnotesize
\begin{tabular}{@{}llrrcrl@{}}
\toprule
Contrast & Metric & $n$ & $\Delta$ & 95\% CI & MDE & Role \\
\midrule
\method $-$ identity & RULER filled & \TBDcell & \TBDcell & \TBDcell & \TBDcell & primary \\
\method $-$ identity & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & \TBDcell & primary \\
\method + rows $-$ identity + rows & RULER filled & \TBDcell & \TBDcell & \TBDcell & \TBDcell & primary \\
\method $-$ no fill & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ no fill & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ no fill + longer tail & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ identity & $\Delta$NLL key & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ identity & LongBench avg & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ identity & Decode agree & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ \method + random rows & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ \method + longer tail & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ identity + rows & Turn-2 RULER & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ identity + rows, q-blind & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & exploratory \\
\method $-$ POP native & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & -- & exploratory$^\ast$ \\
SwiftKV maps + exact tail $-$ \method & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & -- & result \\
SwiftKV native $-$ \method & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & -- & result$^\ast$ \\
\bottomrule
\end{tabular}
\caption{Locked-set battery on \llama, split \TBDcell{} (\TBDcell{} of depth), tail \TBDcell{}. Top: one row per arm. Identity is POP's fill at our split and tail; POP native is POP's own split and boundary handling. Bottom: declared contrasts; $\Delta$ is the first arm minus the second; Holm: secondary, Holm-corrected; $^\ast$: unequal cost (labelled, never a claim). SwiftKV rows are measured against \llama's exact (its base weights are bit-identical). MDE: minimum detectable effect.}
\label{tab:battery-dense}
\end{table}
""",
    "battery-dense-qkn": r"""% tables/battery-dense-qkn.tex -- GENERATED TABLE (placeholder written 2026-10-04; no result yet).
% The table generator overwrites this whole file from verified finals; never edit numbers here by hand.
% Every \TBDcell is a value the generator writes; "--" is a cell the declared design does not run.
% Number formats (all tables): signed deltas with an explicit sign; NLL in nats per token, 3 decimals; RULER and
% LongBench in points (0-100), 1 decimal; shares and agreement in percent, 1 decimal; TTFT speedups exact/arm,
% 2 decimals with \speedx; intervals as [lo, hi] from the BCa bootstrap unless the header says otherwise.
% Layout contract, panel A (one row per arm, rows in this order):
%   Arm | RULER all | RULER filled | LongBench avg | dNLL novel | dNLL key | Decode agree | Decode KL | Turn-2 RULER | TTFT speedup
%   RULER all / filled: mean points over the pooled cells (filled = needles at depth < N - T; the primary metric)
%   LongBench avg: mean of the eight task means, official metrics (secondary)
%   dNLL novel / key: arm minus exact, document-weighted mean over the long-document set (key = LongPPL-style key tokens)
%   Decode agree / KL: greedy-token agreement (%) and mean per-step KL(exact || arm) over the decode steps (T1.7)
%   Turn-2 RULER: second question on the cached prefix, accuracy in points (T1.8)
%   TTFT speedup: measured exact TTFT / arm TTFT at 16K, median over repeats (T3.1)
% Layout contract, panel B (one row per declared contrast, rows in this order):
%   Contrast | Metric | n (pairs) | Delta | 95% CI (BCa) | MDE | Role (primary / Holm = secondary, Holm-corrected / exploratory / result;
%   a trailing * marks a comparison at unequal cost, which the protocol requires to be labelled)
% Model: Qwen3-8B, split 13, tail 4096, capped at 32K (RULER 64K cell and 36 of 126 documents skipped).
\begin{table}[p]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{RULER} & LongB. & \multicolumn{2}{c}{$\Delta$NLL} & \multicolumn{2}{c}{Decode} & Turn 2 & TTFT \\
\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-9}\cmidrule(l){10-10}
Arm & all & filled & avg & novel & key & agree & KL & RULER & speedup \\
\midrule
Exact & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & \TBDcell & -- \\
No fill & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- \\
Identity & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- \\
POP native & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
\method & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- \\
Identity + rows & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & \TBDcell & -- \\
\method + rows & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & \TBDcell & -- \\
\method + random rows & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
\method + longer tail & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
No fill + longer tail & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
Identity + rows, q-blind & \TBDcell & \TBDcell & -- & -- & -- & -- & -- & -- & -- \\
\method + rows, q-blind & \TBDcell & \TBDcell & -- & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}

\medskip
\footnotesize
\begin{tabular}{@{}llrrcrl@{}}
\toprule
Contrast & Metric & $n$ & $\Delta$ & 95\% CI & MDE & Role \\
\midrule
\method $-$ identity & RULER filled & \TBDcell & \TBDcell & \TBDcell & \TBDcell & primary \\
\method $-$ identity & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & \TBDcell & primary \\
\method + rows $-$ identity + rows & RULER filled & \TBDcell & \TBDcell & \TBDcell & \TBDcell & primary \\
identity + rows $-$ exact & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ exact & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ \method + random rows & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ \method + longer tail & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ no fill & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ identity & $\Delta$NLL key & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ identity & LongBench avg & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ identity & Decode agree & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ identity + rows & Turn-2 RULER & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
identity + rows, q-blind $-$ identity + rows & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & exploratory \\
\method $-$ POP native & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & -- & exploratory$^\ast$ \\
\bottomrule
\end{tabular}
\caption{Locked-set battery on \qwendense (dense, QK-norm), split \TBDcell{} (\TBDcell{} of depth), tail \TBDcell{}; same layout as Table~\ref{tab:battery-dense}. Run at no more than \TBDcell{} tokens without YaRN: the \TBDcell{} RULER cell and \TBDcell{} of the \TBDcell{} long documents are skipped.}
\label{tab:battery-dense-qkn}
\end{table}
""",
    "battery-hybrid": r"""% tables/battery-hybrid.tex -- GENERATED TABLE (placeholder written 2026-10-04; no result yet).
% The table generator overwrites this whole file from verified finals; never edit numbers here by hand.
% Every \TBDcell is a value the generator writes; "--" is a cell the declared design does not run.
% Number formats (all tables): signed deltas with an explicit sign; NLL in nats per token, 3 decimals; RULER and
% LongBench in points (0-100), 1 decimal; shares and agreement in percent, 1 decimal; TTFT speedups exact/arm,
% 2 decimals with \speedx; intervals as [lo, hi] from the BCa bootstrap unless the header says otherwise.
% Layout contract, panel A (one row per arm, rows in this order):
%   Arm | RULER all | RULER filled | LongBench avg | dNLL novel | dNLL key | Decode agree | Decode KL | Turn-2 RULER | TTFT speedup
%   RULER all / filled: mean points over the pooled cells (filled = needles at depth < N - T; the primary metric)
%   LongBench avg: mean of the eight task means, official metrics (secondary)
%   dNLL novel / key: arm minus exact, document-weighted mean over the long-document set (key = LongPPL-style key tokens)
%   Decode agree / KL: greedy-token agreement (%) and mean per-step KL(exact || arm) over the decode steps (T1.7)
%   Turn-2 RULER: second question on the cached prefix, accuracy in points (T1.8)
%   TTFT speedup: measured exact TTFT / arm TTFT at 16K, median over repeats (T3.1)
% Layout contract, panel B (one row per declared contrast, rows in this order):
%   Contrast | Metric | n (pairs) | Delta | 95% CI (BCa) | MDE | Role (primary / Holm = secondary, Holm-corrected / exploratory / result;
%   a trailing * marks a comparison at unequal cost, which the protocol requires to be labelled)
% Model: Qwen3.5-9B, split 12, tail 4096. No SwiftKV rows; TTFT column is '--' (T3.1 runs on Llama only).
\begin{table}[p]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{RULER} & LongB. & \multicolumn{2}{c}{$\Delta$NLL} & \multicolumn{2}{c}{Decode} & Turn 2 & TTFT \\
\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-9}\cmidrule(l){10-10}
Arm & all & filled & avg & novel & key & agree & KL & RULER & speedup \\
\midrule
Exact & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & \TBDcell & -- \\
No fill & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- \\
Identity & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- \\
POP native & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
\method & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- \\
Identity + rows & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & \TBDcell & -- \\
\method + rows & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & \TBDcell & -- \\
\method + random rows & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
\method + longer tail & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
No fill + longer tail & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
Identity + rows, q-blind & \TBDcell & \TBDcell & -- & -- & -- & -- & -- & -- & -- \\
\method + rows, q-blind & \TBDcell & \TBDcell & -- & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}

\medskip
\footnotesize
\begin{tabular}{@{}llrrcrl@{}}
\toprule
Contrast & Metric & $n$ & $\Delta$ & 95\% CI & MDE & Role \\
\midrule
\method $-$ identity & RULER filled & \TBDcell & \TBDcell & \TBDcell & \TBDcell & primary \\
\method $-$ identity & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & \TBDcell & primary \\
\method + rows $-$ identity + rows & RULER filled & \TBDcell & \TBDcell & \TBDcell & \TBDcell & primary \\
identity + rows $-$ exact & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ exact & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ \method + random rows & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ \method + longer tail & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ no fill & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ identity & $\Delta$NLL key & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ identity & LongBench avg & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method $-$ identity & Decode agree & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
\method + rows $-$ identity + rows & Turn-2 RULER & \TBDcell & \TBDcell & \TBDcell & -- & Holm \\
identity + rows, q-blind $-$ identity + rows & RULER filled & \TBDcell & \TBDcell & \TBDcell & -- & exploratory \\
\method $-$ POP native & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell & -- & exploratory$^\ast$ \\
\bottomrule
\end{tabular}
\caption{Locked-set battery on \qwenhybrid, split \TBDcell{} (\TBDcell{} of depth), tail \TBDcell{}; same layout as Table~\ref{tab:battery-dense}. TTFT is measured on \llama only.}
\label{tab:battery-hybrid}
\end{table}
""",
    "splitcurve": r"""% tables/splitcurve.tex -- GENERATED TABLE (placeholder written 2026-10-04; no result yet).
% The table generator overwrites this whole file from verified finals; never edit numbers here by hand.
% Every \TBDcell is a value the generator writes; "--" is a cell the declared design does not run.
% Number formats (all tables): signed deltas with an explicit sign; NLL in nats per token, 3 decimals; RULER and
% LongBench in points (0-100), 1 decimal; shares and agreement in percent, 1 decimal; TTFT speedups exact/arm,
% 2 decimals with \speedx; intervals as [lo, hi] from the BCa bootstrap unless the header says otherwise.
% Layout contract: per model an italic group row, then one row per depth (1/4, 3/8, 1/2, 5/8), then one POP native row.
%   Depth | S | dNLL RidgeFill | dNLL identity | dNLL no fill | RidgeFill - identity [CI] | RidgeFill(next deeper split) - RidgeFill(this) [CI] | TTFT RidgeFill | TTFT identity | TTFT no fill
%   The reversal column is '--' at the deepest split. TTFT columns are Llama only (T3.1 measures all four splits); '--' elsewhere.
%   The POP native row puts its NLL delta in the identity column and its TTFT in the identity TTFT column.
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrrrccrrr@{}}
\toprule
 & & \multicolumn{3}{c}{$\Delta$NLL novel} & \method $-$ identity & next split $-$ this & \multicolumn{3}{c}{TTFT speedup (16K)} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}\cmidrule(l){8-10}
Depth & $S$ & \method & identity & no fill & 95\% CI & 95\% CI & \method & identity & no fill \\
\midrule
\multicolumn{10}{@{}l}{\textit{\llamashort}} \\
1/4 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
3/8 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
1/2 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
5/8 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & \TBDcell & \TBDcell & \TBDcell \\
POP native & \TBDcell & -- & \TBDcell & -- & -- & -- & -- & \TBDcell & -- \\
\midrule
\multicolumn{10}{@{}l}{\textit{\qwenhybridshort}} \\
1/4 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- \\
3/8 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- \\
1/2 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- \\
5/8 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
POP native & \TBDcell & -- & \TBDcell & -- & -- & -- & -- & -- & -- \\
\midrule
\multicolumn{10}{@{}l}{\textit{\qwendense}} \\
1/4 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- \\
3/8 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- \\
1/2 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- \\
5/8 & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & -- & -- & -- & -- \\
POP native & \TBDcell & -- & \TBDcell & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\caption{Split curve on the locked long-document set (tail \TBDcell{}) with measured TTFT on \llamashort. $S$ is the split layer. ``Next split $-$ this'' is \method's change from this depth to the next deeper one; a reversal is an interval above zero. The POP native row uses POP's split and a one-token tail, so its identity columns hold its fill.}
\label{tab:splitcurve}
\end{table}
""",
    "mechanism": r"""% tables/mechanism.tex -- GENERATED TABLE (placeholder written 2026-10-04; no result yet).
% The table generator overwrites this whole file from verified finals; never edit numbers here by hand.
% Every \TBDcell is a value the generator writes; "--" is a cell the declared design does not run.
% Number formats (all tables): signed deltas with an explicit sign; NLL in nats per token, 3 decimals; RULER and
% LongBench in points (0-100), 1 decimal; shares and agreement in percent, 1 decimal; TTFT speedups exact/arm,
% 2 decimals with \speedx; intervals as [lo, hi] from the BCa bootstrap unless the header says otherwise.
% Layout contract, panel A: rows (model x arm) in this order; arms RidgeFill, +exact keys (kexact), +exact values (vexact),
%   +exact keys and values (kvexact), +key scale measured on the tail (ktail). The value-scale correction (vtail) and the
%   calibration-domain shift are appendix material and not in this table.
%   Model | Arm | dNLL novel | loss removed (% of RidgeFill's NLL loss vs exact) | 95% CI of that share | RULER filled | gap closed (% of RidgeFill's filled-region RULER gap)
% Layout contract, panel B: one row per model; QK-norm and GDN columns are model facts; the others are read from panel A and
%   the battery tables ("keys" / "values"; "yes" / "no" / "partial" with the share in parentheses).
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrcrr@{}}
\toprule
Model & Arm & $\Delta$NLL novel & loss removed & 95\% CI & RULER filled & gap closed \\
\midrule
\llamashort & \method & \TBDcell & -- & -- & \TBDcell & -- \\
\llamashort & + exact keys & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\llamashort & + exact values & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\llamashort & + exact keys and values & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\llamashort & + key scale (tail) & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\midrule
\qwenhybridshort & \method & \TBDcell & -- & -- & \TBDcell & -- \\
\qwenhybridshort & + exact keys & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\qwenhybridshort & + exact values & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\qwenhybridshort & + exact keys and values & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\qwenhybridshort & + key scale (tail) & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\midrule
\qwendense & \method & \TBDcell & -- & -- & \TBDcell & -- \\
\qwendense & + exact keys & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\qwendense & + exact values & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\qwendense & + exact keys and values & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\qwendense & + key scale (tail) & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\bottomrule
\end{tabular}

\medskip

\begin{tabular}{@{}lcccccc@{}}
\toprule
Model & QK-norm & GDN layers & loss carried by & exact keys restore & key scale & rows restore \\
\midrule
\llamashort & no & no & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\qwenhybridshort & yes & yes & \TBDcell & \TBDcell & demonstration & \TBDcell \\
\qwendense & yes & no & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\bottomrule
\end{tabular}
\caption{Mechanism. Top: oracle splices on the locked long-document set and the RULER subset, at the primary split and tail; ``loss removed'' is the share of \method's NLL loss against exact that the splice removes, ``gap closed'' the share of its filled-region RULER gap. Bottom: the per-architecture repair table; on a QK-normed model the key-scale correction cannot act (it is a demonstration, not a test).}
\label{tab:mechanism}
\end{table}
""",
    "flashnext": r"""% tables/flashnext.tex -- GENERATED TABLE (layout of 2026-10-10: the C5 tier as run; DEV-17).
% The table generator overwrites this whole file from verified finals; never edit numbers here by hand.
% Every \TBDcell is a value the generator writes; "--" is a cell the declared design does not run.
% Number formats (all tables): signed deltas with an explicit sign; NLL in nats per token, 3 decimals; RULER and
% LongBench in points (0-100), 1 decimal; shares and agreement in percent, 1 decimal; TTFT speedups exact/arm,
% 2 decimals with \speedx; intervals as [lo, hi] from the BCa bootstrap unless the header says otherwise.
% Layout contract, panel A (arms in this order): exact; quality mode (RidgeFill + GDN terminal-state correction + class
%   protection at the declared share, ridgefill+corr+prot25); RidgeFill alone (ridgefill).
%   Arm | ppl gap to exact, 100 (exp(dNLL) - 1) in % | its 95% CI (BCa, documents within domain) | RULER all | RULER
%   filled | TTFT speedup 9K | 16K | 32K (exact TTFT on the arm's own boot / the arm's, median of the timed rounds)
% Layout contract, panel A, last column: 64K request speedup (exact's median wall time of the 64K RULER rows over the
%   arm's; each arm on its own boot; DEV-24, review 2026-10-11)
% Layout contract, panel B (C5's criteria, quality mode): Criterion | margin | estimate | one-sided bound | verdict
%   (perplexity: the measured gap, ledger amendment 11; RULER: one-sided lower bound, BCa or the exact binomial floor;
%   speed: TTFT speedup at the registered length). Under panel B: the engine build and the determinism read.
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{Perplexity vs exact} & \multicolumn{2}{c}{RULER} & \multicolumn{3}{c}{TTFT speedup} & 64K \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(l){9-9}
Arm & gap & 95\% CI & all & filled & 9K & 16K & 32K & request \\
\midrule
Exact & -- & -- & \TBDcell & \TBDcell & -- & -- & -- & -- \\
Quality mode & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\method & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\bottomrule
\end{tabular}

\medskip

\begin{tabular}{@{}llrrl@{}}
\toprule
Criterion (quality mode) & Margin & Estimate & Bound & Verdict \\
\midrule
Perplexity gap to exact & $\le$ \TBDcell & \TBDcell & -- & \TBDcell \\
RULER filled $-$ exact & bound $\ge$ \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
TTFT speedup at 16K & $\ge$ \TBDcell & \TBDcell & -- & \TBDcell \\
\bottomrule
\end{tabular}
\caption{\flashnext in the serving engine. Quality mode: \method with the GDN terminal-state correction and class protection at \TBDcell{} of class matches; \method: the fill alone. Perplexity gap: $\exp(\Delta\mathrm{NLL})-1$ against exact on the same \TBDcell{} locked documents (novel tokens; per-document paired differences, document-weighted; BCa \TBDcell{} interval, documents resampled within domain). RULER: accuracy in points over all \TBDcell{} samples and over the \TBDcell{} filled-region samples (every scored needle lies in the prefix, before the exact tail of \TBDcell{} tokens). TTFT speedup: exact TTFT on the same server start over the arm's, median of \TBDcell{} rounds. 64K request: exact's median request time on the 64K RULER samples (prefill of the prompt and a short answer) over the arm's, each on its own server start. Bottom: the criteria declared for quality mode before the run; the RULER bound is one-sided \TBDcell{}, the more conservative of the BCa bound and the binomial floor $-100\,(1-0.05^{1/n})$ points, which alone is used when no sample is worse than exact.}
\label{tab:flashnext}
\end{table}
""",
    "negative": r"""% tables/negative.tex -- GENERATED TABLE (placeholder written 2026-10-04; no result yet).
% The table generator overwrites this whole file from verified finals; never edit numbers here by hand.
% Every \TBDcell is a value the generator writes; "--" is a cell the declared design does not run.
% Number formats (all tables): signed deltas with an explicit sign; NLL in nats per token, 3 decimals; RULER and
% LongBench in points (0-100), 1 decimal; shares and agreement in percent, 1 decimal; TTFT speedups exact/arm,
% 2 decimals with \speedx; intervals as [lo, hi] from the BCa bootstrap unless the header says otherwise.
% Layout contract: one row per finding, in OUTLINE 6.5's order.
%   Finding | Model | Set, protocol (e.g. "locked, v1.1", "dev, v1.0.1", "engine tier, v1.0.2") | Contrast | Estimate | 95% CI | Evidence
%   Rows 3-6 are pre-final evidence: the generator reads their ledger records (Q1b, P4, WF1, FQ3; the Flash-Next data-size
%   run of 2026-10-04) and must print their own set and protocol version, never the finals'. The Evidence cells are
%   table references or ledger IDs, not numbers.
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}p{3.1cm}lp{1.5cm}p{3.3cm}rcl@{}}
\toprule
Finding & Model & Set, protocol & Contrast & Estimate & 95\% CI & Evidence \\
\midrule
Hybrid wash: \method ties identity & \qwenhybridshort & locked, v\TBDcell & \method $-$ identity, $\Delta$NLL novel & \TBDcell & \TBDcell & Table~\ref{tab:battery-hybrid} \\
SwiftKV's trained maps beat the ridge fit & \llamashort & locked, v\TBDcell & SwiftKV maps + exact tail $-$ \method, $\Delta$NLL novel & \TBDcell & \TBDcell & Table~\ref{tab:battery-dense} \\
Calibration data size is flat & \flashnext & \TBDcell & three times the data $-$ base fit, $\Delta$NLL & \TBDcell & \TBDcell & \TBDcell \\
GDN terminal-state correction does nothing & \qwenhybridshort & dev, \TBDcell & + correction $-$ none, $\Delta$NLL & \TBDcell & \TBDcell & Q1b, P4 \\
Weighted projector fit loses needles & both & dev, \TBDcell & weighted $-$ plain fit, needles & \TBDcell & \TBDcell & WF1 \\
Refresh layer loses to rows at equal compute & both & dev, \TBDcell & refresh $-$ rows at equal compute, $\Delta$NLL & \TBDcell & \TBDcell & FQ3 \\
\bottomrule
\end{tabular}
\caption{Negative results. Rows from development sets or earlier protocol versions are labelled in the ``Set, protocol'' column and are never compared with locked-set numbers; Evidence names the table or the ledger record.}
\label{tab:negative}
\end{table}
""",
    "mde": r"""% tables/mde.tex -- GENERATED TABLE (placeholder written 2026-10-04; no result yet).
% The table generator overwrites this whole file from verified finals; never edit numbers here by hand.
% Every \TBDcell is a value the generator writes; "--" is a cell the declared design does not run.
% Number formats (all tables): signed deltas with an explicit sign; NLL in nats per token, 3 decimals; RULER and
% LongBench in points (0-100), 1 decimal; shares and agreement in percent, 1 decimal; TTFT speedups exact/arm,
% 2 decimals with \speedx; intervals as [lo, hi] from the BCa bootstrap unless the header says otherwise.
% Layout contract: one row per primary contrast per model (two contrasts, three rows: RULER and NLL for (a), RULER for (b)),
%   then the engine's equivalence contrasts, then one illustrative single-cell RULER row.
%   Model | Primary contrast | Metric | n (pairs) | SD of paired delta | MDE (power and level as declared in protocol v1.1; the caption states them)
\begin{table}[t]
\centering
\footnotesize
\begin{tabular}{@{}lllrrr@{}}
\toprule
Model & Primary contrast & Metric & $n$ & SD of paired $\Delta$ & MDE \\
\midrule
\llamashort & \method $-$ identity & RULER filled & \TBDcell & \TBDcell & \TBDcell \\
\llamashort & \method $-$ identity & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell \\
\llamashort & \method + rows $-$ identity + rows & RULER filled & \TBDcell & \TBDcell & \TBDcell \\
\qwenhybridshort & \method $-$ identity & RULER filled & \TBDcell & \TBDcell & \TBDcell \\
\qwenhybridshort & \method $-$ identity & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell \\
\qwenhybridshort & \method + rows $-$ identity + rows & RULER filled & \TBDcell & \TBDcell & \TBDcell \\
\qwendense & \method $-$ identity & RULER filled & \TBDcell & \TBDcell & \TBDcell \\
\qwendense & \method $-$ identity & $\Delta$NLL novel & \TBDcell & \TBDcell & \TBDcell \\
\qwendense & \method + rows $-$ identity + rows & RULER filled & \TBDcell & \TBDcell & \TBDcell \\
\flashnext & Quality mode $-$ exact & ppl ratio & \TBDcell & \TBDcell & \TBDcell \\
\flashnext & Quality mode $-$ exact & RULER filled & \TBDcell & \TBDcell & \TBDcell \\
\llamashort & single RULER cell & accuracy & \TBDcell & \TBDcell & \TBDcell \\
\bottomrule
\end{tabular}
\caption{Minimum detectable effect of every primary contrast at the declared power and level (\TBDcell{}), from the paired standard deviation. The last row shows why single RULER cells are descriptive only.}
\label{tab:mde}
\end{table}
""",
    "persample": r"""% tables/persample.tex -- GENERATED TABLE (placeholder written 2026-10-04; no result yet).
% The table generator overwrites this whole file from verified finals; never edit numbers here by hand.
% Every \TBDcell is a value the generator writes; "--" is a cell the declared design does not run.
% Number formats (all tables): signed deltas with an explicit sign; NLL in nats per token, 3 decimals; RULER and
% LongBench in points (0-100), 1 decimal; shares and agreement in percent, 1 decimal; TTFT speedups exact/arm,
% 2 decimals with \speedx; intervals as [lo, hi] from the BCa bootstrap unless the header says otherwise.
% Layout contract, NLL longtable: one row per (model, locked long document) in manifest order; Document is the file stem
%   (never any text), Tokens the prompt length under that model's tokenizer; Qwen3-8B rows only for documents <= 32K.
%   Model | Document | Tokens | RidgeFill | identity | no fill | POP native | RidgeFill + rows | identity + rows
% Layout contract, RULER longtable: one row per (model, task, length); filled-region accuracy in points.
%   Model | Task | Length | exact | RidgeFill | identity | no fill | RidgeFill + rows | identity + rows
% Per-sample RULER and LongBench outcomes go in the released result files, not in the PDF.
\begingroup\footnotesize\setlength{\tabcolsep}{4pt}
\begin{longtable}{@{}llrrrrrrr@{}}
\caption{Per-document NLL delta against exact (novel tokens, nats per token) at the primary split and tail.}\label{tab:persample-nll}\\
\toprule
Model & Document & Tokens & \method & identity & no fill & POP nat. & \method+rows & id.+rows \\
\midrule
\endhead
\bottomrule
\endfoot
\TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\end{longtable}

\begin{longtable}{@{}lllrrrrrr@{}}
\caption{Per-cell RULER accuracy (filled-region needles, points) at the primary split and tail.}\label{tab:persample-ruler}\\
\toprule
Model & Task & Length & exact & \method & identity & no fill & \method+rows & id.+rows \\
\midrule
\endhead
\bottomrule
\endfoot
\TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\end{longtable}
\endgroup
""",
    "levers": r"""% tables/levers.tex -- GENERATED TABLE (placeholder written 2026-10-04; no result yet).
% The table generator overwrites this whole file from verified finals; never edit numbers here by hand.
% Every \TBDcell is a value the generator writes; "--" is a cell the declared design does not run.
% Number formats (all tables): signed deltas with an explicit sign; NLL in nats per token, 3 decimals; RULER and
% LongBench in points (0-100), 1 decimal; shares and agreement in percent, 1 decimal; TTFT speedups exact/arm,
% 2 decimals with \speedx; intervals as [lo, hi] from the BCa bootstrap unless the header says otherwise.
% Layout contract: one row per lever of docs/flash-next/run-20261004/levers/TRIAGE.md (70 levers) plus the reference-model
%   levers of paper/GAINS.md, in the triage's numbering; Outcome is one line with its set and protocol version; Evidence is
%   ledger IDs or a results path.
\begin{longtable}{@{}rp{4.2cm}p{2.6cm}p{5.2cm}l@{}}
\caption{Every development lever, one line each. Development-set evidence only; none is used for a claim.}\label{tab:levers}\\
\toprule
\# & Lever & Where tested & Outcome & Evidence \\
\midrule
\endhead
\bottomrule
\endfoot
\TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\end{longtable}
""",
    "configs": r"""% tables/configs.tex -- GENERATED TABLE (placeholder written 2026-10-04; no result yet).
% The table generator overwrites this whole file from verified finals; never edit numbers here by hand.
% Every \TBDcell is a value the generator writes; "--" is a cell the declared design does not run.
% Number formats (all tables): signed deltas with an explicit sign; NLL in nats per token, 3 decimals; RULER and
% LongBench in points (0-100), 1 decimal; shares and agreement in percent, 1 decimal; TTFT speedups exact/arm,
% 2 decimals with \speedx; intervals as [lo, hi] from the BCa bootstrap unless the header says otherwise.
% Layout contract: one row per FINAL- ledger record; Config is the config file stem; lambda* the chosen ridge penalty of
%   that split's projector; Manifest sha256 the first 12 hex digits of the dataset manifest hash; Ledger record the FINAL-
%   record ID. The code tag and the projector file hashes go in a second line per config when the generator has them.
\begin{longtable}{@{}lllrrlll@{}}
\caption{Final configurations, declared in the ledger before each run.}\label{tab:configs}\\
\toprule
Config & Model & Set & $S$ & $T$ & $\lambda^\star$ & Manifest sha256 & Ledger record \\
\midrule
\endhead
\bottomrule
\endfoot
\TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell & \TBDcell \\
\end{longtable}
""",
}


if __name__ == "__main__":
    sys.exit(main())
