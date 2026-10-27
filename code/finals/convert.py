#!/usr/bin/env python3
"""Convert a finished cell's reference-harness results into the finals sample records finals/tables.py reads (its
docstring is the contract). The collect pass between a cell the box finished and a paper table; read-only on the
harness results, the cell, the ledger and protocol.json.

Input, per cell dir (finals/lib/runner.py writes it): cell.json, host.json, canary.json, harness-<config>.json (the
configs the FINAL- record pins), work/results/<set>-<config>/<prompt>.json as scripts/evaluate.py writes them, each
with its .sha256 sidecar.
Order (each step fails closed and refuses by name; nothing is written unless every record of the cell converts):
  1. verify   finals/verify_finals.py's checks on the cell (every sidecar, the FINAL- record's pins of cell.json and
              of every harness config, the canary); a results dir that is not '<evaluate set>-<pinned config>', or
              a pinned config without results, is refused.
  2. arms     every config arm -> (protocol kind, split key): kind by kva.protocol.arm_kind on the arm's spec AND by
              its paper name (finals/lib/kvabackend.py PAPER, '<paper name>-s<layer>'); the two must agree. Split keys:
              the layer's fraction of depth for the cell's model (protocol.json splits/depths: "1/4" "3/8" "1/2"
              "5/8"), "pop" for pop-native (at the model's pop_split), "-" for exact. Arm names in the records are
              protocol kinds (ridgefill -> kva, identity -> noproj, ...: protocol.json arm_aliases and the v1.1 kinds),
              the names tables.py resolves.
  3. records  every record must hold exactly its config's arms plus exact, its stamp (the protocol version the
              harness validated it under, kva.protocol.validate) must be of the current protocol's major.minor, and its
              config must be the pinned one (name and arm names). Metrics come from the official scorers, re-run here
              on what the harness stored and compared with the harness's own value (two instruments; a disagreement is
              refused):
                ruler, qfirst   score = kva.metrics.ruler_row(metric, pred, refs); needle_filled = every scored needle
                                (the record's 'needles', kva.locked.needle_spans) ends at or before N - T, T the cell's
                                declared tail (NIAH rows; the other tasks have no needle and no filled-region readout)
                longbench       score = kva.metrics.longbench_row(task, pred, refs, all_classes)
                longppl, grade  novel_nll = kva.locked.nll over the novel mask; key_nll + n_key over the key-token mask
                                (kva.keytok.key_mask at the cell's declared threshold, which must equal protocol.json's;
                                the tail-only reference must be at the cell's keytoken tail); all_nll; checked against
                                kva.keytok.readout's deltas and (longppl) the harness's own score
                decode          top1_agree = the forced top-1 agreement, kl_mean (checked against the per-step KL list),
                                first_divergence, frac_identical (kva/agree.py)
                multiturn       turn1, turn2 = each turn's correct (kva.data.niah_correct re-run on the stored text)
  4. merge    one record per (set, pairing unit): the arms of every config that ran the unit (a cell's RULER configs
              partition its rows; the SwiftKV configs run the same rows on the SwiftKV checkpoint, whose own exact is
              kept as exact_swiftkv, never as the arm; on NLL it must equal the cell's exact, SK1, since the SwiftKV
              arms' key-token mask comes from it). An arm at one split from two configs is refused.
  5. skipped  one record per file the cell declares 'skipped' (its max_len cap), never imputed; a file the harness
              skipped that the cell does not declare is refused.
  6. write    samples/<set>/<unit>.json with sidecars (atomic, finals/lib/atomic.py), files of an earlier conversion that
              this one does not produce removed, then samples/CONVERTED.json last: the treehash of work/results and the
              sha256 of every sample file. Deterministic: a rerun on the same results writes the same bytes
              (idempotent). tables.py refuses a cell with harness results and no current stamp.
Sets: evaluate set -> sample set: ruler -> ruler (ruler_variant qfirst -> ruler-qfirst), longbench, longppl, decode,
multiturn as named; grade (the dev set, never in a finals cell) converts under its own name with tier "dev", so a test can
convert it and tables.py refuses it. Units: RULER-derived and LongBench rows '<locked file>#<row>' (one RULER row keeps
its id under both prompt layouts, so the question-first contrast pairs it); documents '<locked file>'; grade 'grade/<name>'.
Every record carries protocol (the stamp's version), tier (locked-final; dev for grade), cell and configs. A cell with
'dataset' (another frozen set, e.g. fresh-long-20261008) is not the locked set: its records carry dataset and tier = that
dataset's name (record_tier), never locked-final, so tables.py keeps them apart from every locked-set result.

Usage: convert.py RESULTS [--cell NAME ...] [--ledger L]
Exit: 0 converted (or nothing to convert); 1 refused; 2 usage.
"""
import argparse
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

FINALS = Path(__file__).resolve().parent
REPO = FINALS.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FINALS / "lib"))
import atomic                                                # noqa: E402
import kvabackend                                            # noqa: E402

EVAL_SETS = ("ruler", "longbench", "longppl", "decode", "multiturn", "grade")
TIER = {"grade": "dev"}                                      # every other set: locked-final


def record_tier(cell, set_=None, decode_from=None):
    """{tier, [dataset]} of a converted record: dev for grade, the frozen set's name for a cell with 'dataset',
    else locked-final."""
    if "dev" in (TIER.get(set_), TIER.get(decode_from)):
        return {"tier": "dev"}
    if cell.get("dataset"):
        return {"tier": cell["dataset"], "dataset": cell["dataset"]}
    return {"tier": "locked-final"}
TOL = 1e-9                                                   # two instruments on the same stored numbers


class ConvertRefused(Exception):
    """A named refusal: the cell is not converted and nothing of it is written."""


def _verify_module():
    spec = importlib.util.spec_from_file_location("finals_verify", FINALS / "verify_finals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VF = _verify_module()


def mm(version):
    return ".".join(str(version).split(".")[:2])


def safe(sample):
    return sample.replace("/", "__").replace("#", "_")


# ---------------------------------------------------------------------------------------------- 1. verify, configs
def verify(cell_dir, ledger):
    """verify_finals.check_cell, less the currency of the conversion stamp this run rewrites (a cell resumed after an
    earlier conversion is converted again; a tampered sample file is still refused)."""
    stale = set(VF.stamp_problems(cell_dir)) if (cell_dir / "samples" / VF.STAMP).is_file() else set()
    problems = [x for x in VF.check_cell(cell_dir, ledger)[0] if x not in stale]
    if problems:
        raise ConvertRefused("finals/verify_finals.py:\n  " + "\n  ".join(problems))


def harness_runs(cell_dir):
    """[(evaluate set, config, results dir)] for every pinned config (harness-<name>.json) of the cell (the harness's
    own canary references, work/results/canary, are not results)."""
    configs = {f.name[len("harness-"):-len(".json")]: json.loads(f.read_text()) for f in cell_dir.glob("harness-*.json")}
    out, seen = [], set()
    for d in sorted(p for p in (cell_dir / "work" / "results").iterdir() if p.is_dir() and p.name != "canary"):
        set_, _, name = d.name.partition("-")
        if set_ not in EVAL_SETS or name not in configs:
            raise ConvertRefused(f"{cell_dir.name}: work/results/{d.name} is not '<evaluate set>-<pinned config>' "
                                 f"(sets {EVAL_SETS}, pinned configs {sorted(configs)})")
        out.append((set_, configs[name], d))
        seen.add(name)
    if set(configs) - seen:
        raise ConvertRefused(f"{cell_dir.name}: pinned configs without results: {sorted(set(configs) - seen)}")
    return out


# ---------------------------------------------------------------------------------------------- 2. arms
def split_key(cell, kind, layer, p):
    m = p["models"][cell["model"]["name"]]
    if kind == "pop-native":
        if layer != m.get("pop_split"):
            raise ConvertRefused(f"pop-native at layer {layer}, not {cell['model']['name']}'s pop_split {m.get('pop_split')}")
        return "pop"
    if layer not in m["splits"]:
        raise ConvertRefused(f"layer {layer} is no split of {cell['model']['name']} (protocol.json {m['splits']})")
    f = Fraction(p["depths"][m["splits"].index(layer)]).limit_denominator(64)
    return f"{f.numerator}/{f.denominator}"


def arm_keys(cfg, cell, p):
    """{config arm name: (protocol kind, split key)}; exact -> ('exact', '-')."""
    from kva import protocol as PR
    out = {"exact": ("exact", "-")}
    for spec in cfg["arms"]:
        kind = PR.arm_kind(spec, p)
        paper, _, layer = spec["name"].rpartition("-s")
        if kind is None or kvabackend.PAPER.get(paper) != kind or layer != str(spec.get("split")):
            raise ConvertRefused(f"{cfg['name']}: arm {spec['name']!r}: its spec is protocol kind {kind!r}, its name "
                                 f"says {kvabackend.PAPER.get(paper)!r} at layer {layer!r}: the two must agree")
        out[spec["name"]] = (kind, split_key(cell, kind, spec["split"], p))
    if len(set(out.values())) != len(out):
        raise ConvertRefused(f"{cfg['name']}: two arms map to one (kind, split): {sorted(out.values())}")
    return out


# ---------------------------------------------------------------------------------------------- 3. metrics
def _agree(what, a, b, tol=TOL):
    if a is None or b is None or abs(a - b) > tol:
        raise ConvertRefused(f"{what}: the official scorer gives {a!r}, the harness stored {b!r} (two instruments "
                             "disagree)")


def gen_scores(rec, set_):
    from kva import metrics as MT
    out = {}
    for name, run in rec["runs"].items():
        if run.get("metric") != rec["metric"]:
            raise ConvertRefused(f"{rec['name']} {name}: metric {run.get('metric')!r}, the row's is {rec['metric']!r}")
        s = (MT.ruler_row(rec["metric"], run["pred"], rec["refs"]) if set_ == "ruler"
             else MT.longbench_row(rec["task"], run["pred"], rec["refs"], rec["all_classes"]))
        _agree(f"{rec['name']} {name} score", s, run["score"], 0.0)
        out[name] = {"score": s}
    return out


def nll_scores(rec, set_, cell, p):
    from kva import keytok, locked
    kt = cell.get("keytoken")
    if not kt or kt.get("threshold") != p.get("keytoken_threshold"):
        raise ConvertRefused(f"{cell['name']}: the key-token readout needs the cell's declared keytoken threshold, equal "
                             f"to protocol.json's ({p.get('keytoken_threshold')}); the cell declares {kt}")
    ref = rec.get("tail_only") or {}
    if ref.get("T") != kt["tail"]:
        raise ConvertRefused(f"{rec['name']}: tail-only reference at T {ref.get('T')}, the cell declares {kt['tail']}")
    novel = rec["novel"]
    if not any(novel):
        raise ConvertRefused(f"{rec['name']}: no novel token in the scored window")
    ex = rec["runs"]["exact"]["logprobs"]
    key = keytok.key_mask(ex, ref["logprobs"], kt["threshold"])
    n_key = sum(key)
    out = {}
    for name, run in rec["runs"].items():
        lp = run["logprobs"]
        if len(lp) != len(novel):
            raise ConvertRefused(f"{rec['name']} {name}: {len(lp)} log-probs for {len(novel)} scored tokens")
        out[name] = dict(novel_nll=locked.nll(lp, novel), key_nll=locked.nll(lp, key) if n_key else None,
                         n_key=n_key, n_novel=sum(novel), all_nll=locked.nll(lp))
        if set_ == "longppl":
            _agree(f"{rec['name']} {name} all-token NLL", out[name]["all_nll"], run["score"])
    for name, r in keytok.readout(rec, kt["threshold"]).items():
        _agree(f"{rec['name']} {name} novel delta", out[name]["novel_nll"] - out["exact"]["novel_nll"], r["novel"]["delta"])
        if n_key:
            _agree(f"{rec['name']} {name} key delta", out[name]["key_nll"] - out["exact"]["key_nll"], r["key"]["delta"])
    return out


def decode_scores(rec):
    out = {"exact": {}}
    for name, run in rec["runs"].items():
        if name == "exact":
            continue
        kl = run["kl"]
        _agree(f"{rec['name']} {name} kl_mean", sum(kl) / len(kl), run["kl_mean"], 1e-6)  # 8-decimal rounding
        out[name] = dict(top1_agree=run["forced_top1_agree"], kl_mean=run["kl_mean"],
                         first_divergence=run["first_divergence"], frac_identical=run["frac_identical"])
    return out


def turn_scores(rec, cell):
    from kva import data
    want = list(range(1, cell["secondq"]["turns"] + 1))
    out = {}
    for name, run in rec["runs"].items():
        turns = run["turns"]
        if [t["turn"] for t in turns] != want:
            raise ConvertRefused(f"{rec['name']} {name}: turns {[t['turn'] for t in turns]}, the cell declares {want}")
        for t in turns:
            _agree(f"{rec['name']} {name} turn {t['turn']} correct", float(data.niah_correct(t["text"], t["value"])),
                   float(t["correct"]), 0.0)
        out[name] = {f"turn{t['turn']}": float(t["correct"]) for t in turns}
    return out


# ---------------------------------------------------------------------------------------------- 3. units
def unit(rec, set_, cell):
    """(sample set, unit id, unit fields) of one harness record."""
    if set_ == "grade" or rec.get("decode_from") == "grade":
        return set_, f"grade/{rec['name']}", dict(doc=rec["name"], tokens=rec["n"])
    if "source" not in rec:
        raise ConvertRefused(f"{rec['name']}: a {set_} record with no locked source file (a dev prompt set): no finals "
                             "sample")
    if set_ in ("longppl", "decode"):
        src = rec["source"]
        return set_, src, dict(doc=src.rsplit("/", 1)[-1].rsplit(".", 1)[0], domain=rec["task"], tokens=rec["n"],
                               length=rec["length"])
    sid = f"{rec['source']}#{rec['row']:02d}"
    if set_ == "longbench":
        return "longbench", sid, dict(task=rec["task"], length=rec["length"])
    fields = dict(task=rec["task"], length=rec["length"], tokens=rec["n"])
    if set_ == "multiturn":
        return "multiturn", sid, fields
    if "needles" in rec:
        fields["needle_filled"] = all(e <= rec["n"] - cell["tail"] for _, e in rec["needles"])
    elif rec.get("ruler_task", "niah") == "niah":
        raise ConvertRefused(f"{rec['name']}: a NIAH row without 'needles' (the scored needles' token spans, "
                             "kva.locked.needle_spans): no filled-region readout can be made")
    return ("ruler-qfirst" if rec.get("variant") == "qfirst" else "ruler"), sid, fields


def stamp_of(rec, cfg):
    """The protocol version every run of this record was validated under (its provenance entries)."""
    versions = set()
    for prov in rec["provenance"]:
        c = prov.get("config") or {}
        if (c.get("config") or {}).get("name") != cfg["name"] or \
                sorted(a["name"] for a in c["config"]["arms"]) != sorted(a["name"] for a in cfg["arms"]):
            raise ConvertRefused(f"{rec['name']}: run under config {(c.get('config') or {}).get('name')!r} with other "
                                 f"arms than the pinned {cfg['name']}")
        if not c.get("protocol"):
            raise ConvertRefused(f"NoProtocolStamp: {rec['name']} was run without a protocol stamp (a dry run): no "
                                 "finals sample")
        versions.add(c["protocol"]["version"])
    if len(versions) != 1:
        raise ConvertRefused(f"{rec['name']}: runs under protocol versions {sorted(versions)}")
    return versions.pop()


def convert_record(rec, set_, cfg, keys, cell, p):
    version = stamp_of(rec, cfg)
    if mm(version) != mm(p["version"]):
        raise ConvertRefused(f"{rec['name']}: protocol {version} is not the current protocol {p['version']}")
    if set(rec["runs"]) != set(keys):
        raise ConvertRefused(f"{rec['name']}: arms {sorted(rec['runs'])} are not its config's {sorted(keys)} (an "
                             "unfinished or foreign record)")
    metrics = (gen_scores(rec, set_) if set_ in ("ruler", "longbench") else nll_scores(rec, set_, cell, p)
               if set_ in ("longppl", "grade") else decode_scores(rec) if set_ == "decode" else turn_scores(rec, cell))
    out_set, sid, fields = unit(rec, set_, cell)
    swift = cfg["model"].rstrip("/").rsplit("/", 1)[-1] != cell["model"]["name"]
    arms, extra = {}, {}
    for name, m in metrics.items():
        kind, split = keys[name]
        if swift and kind == "exact":
            extra["exact_swiftkv"] = m            # the SwiftKV checkpoint's own exact (T1.6), never the cell's exact
        else:
            arms.setdefault(kind, {})[split] = m
    return dict(set=out_set, sample=sid, protocol=version, **record_tier(cell, set_, rec.get("decode_from")), cell=cell["name"],
                configs=[cfg["name"]], **fields, **extra, arms=arms), rec["provenance"][-1]["config"].get("skipped") or []


def merge(records):
    """One record per (set, unit): the arms of every config that ran it; unit fields must agree."""
    out = {}
    for r in records:
        key = (r["set"], r["sample"])
        if key not in out:
            out[key] = r
            continue
        m = out[key]
        for k, v in r.items():
            if k not in ("arms", "configs") and m.get(k, v) != v:
                raise ConvertRefused(f"{key}: field {k} is {m[k]!r} in {m['configs']} and {v!r} in {r['configs']}")
            m.setdefault(k, v)
        for kind, by_split in r["arms"].items():
            for split, metrics in by_split.items():
                if split in m["arms"].get(kind, {}):
                    raise ConvertRefused(f"{key}: arm {kind} at split {split} from two configs "
                                         f"({m['configs']}, {r['configs']})")
                m["arms"].setdefault(kind, {})[split] = metrics
        m["configs"] = sorted(m["configs"] + r["configs"])
    return [out[k] for k in sorted(out)]


def check_swiftkv_exact(records):
    """SK1: the SwiftKV checkpoint's base weights are Llama's, so its own exact must equal the cell's exact on every NLL
    document; the SwiftKV arms' key-token mask comes from its own exact, and their NLL is read against Llama's."""
    for r in records:
        if r["set"] not in ("longppl", "grade") or "exact_swiftkv" not in r:
            continue
        a, b = r["exact_swiftkv"], r["arms"].get("exact", {}).get("-")
        same = b is not None and a["n_key"] == b["n_key"] and all(
            (a[k] is None) == (b[k] is None) and (a[k] is None or abs(a[k] - b[k]) <= TOL)
            for k in ("novel_nll", "key_nll", "all_nll"))
        if not same:
            raise ConvertRefused(f"{r['sample']}: the SwiftKV checkpoint's own exact differs from the cell's exact "
                                 f"({a} vs {b}): its base weights are the model's (SK1), so the R6 arms would be read "
                                 "against another model and another key-token mask")


def skipped_records(cell, harness_skipped, version):
    declared = cell.get("skipped", [])
    extra = sorted(set(harness_skipped) - set(declared))
    if extra:
        raise ConvertRefused(f"{cell['name']}: the harness skipped {extra}, which the cell does not declare")
    sets = {r["set"] for r in cell.get("runs", [])}
    out = []
    for path in declared:
        group = ("ruler", "multiturn") if path.startswith("ruler/") else ("longppl", "decode")
        set_ = next((s for s in group if s in sets), None)
        if set_ is None:
            raise ConvertRefused(f"{cell['name']}: declares {path} skipped but runs no set that reads it")
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        length = int(stem.rsplit("_", 2)[-2]) if path.startswith("long-prompts/") else int(stem.rsplit("_", 1)[-1])
        out.append(dict(set=set_, sample=path, protocol=version, **record_tier(cell), cell=cell["name"], length=length,
                        skipped=f"over max_len {cell.get('max_len')} (declared in the cell; no YaRN)"))
    return out


# ---------------------------------------------------------------------------------------------- the cell
def convert_cell(cell_dir, ledger, protocol_path=REPO / "protocol.json"):
    """Convert one cell dir; returns {set: records written}. Nothing is written before every record converts."""
    verify(cell_dir, ledger)
    cell = json.loads((cell_dir / "cell.json").read_text())
    p = json.loads(Path(protocol_path).read_text())
    records, harness_skipped, versions = [], set(), set()
    for set_, cfg, d in harness_runs(cell_dir):
        keys = arm_keys(cfg, cell, p)
        files = sorted(d.glob("*.json"))
        if not files:
            raise ConvertRefused(f"{cell_dir.name}: {d.name} holds no record")
        for f in files:
            rec, skipped = convert_record(json.loads(f.read_text()), set_, cfg, keys, cell, p)
            records.append(rec)
            versions.add(rec["protocol"])
            harness_skipped |= set(skipped)
    if len(versions) != 1:
        raise ConvertRefused(f"{cell_dir.name}: records under protocol versions {sorted(versions)}")
    records = merge(records)
    check_swiftkv_exact(records)
    records += skipped_records(cell, harness_skipped, versions.pop())
    return write(cell_dir, records)


def write(cell_dir, records):
    out = cell_dir / "samples"
    want = {f"{r['set']}/{safe(r['sample'])}.json": r for r in records}
    if len(want) != len(records):
        raise ConvertRefused(f"{cell_dir.name}: two units map to one file name")
    digests = {rel: atomic.write_result(out / rel, r) for rel, r in sorted(want.items())}
    for f in sorted(out.rglob("*.json")):
        rel = f.relative_to(out).as_posix()
        if rel not in want and rel != VF.STAMP:
            f.unlink()
            Path(f"{f}.sha256").unlink(missing_ok=True)
    atomic.write_result(out / VF.STAMP, dict(
        converter="finals/convert.py", converter_sha256=atomic.sha256_file(Path(__file__)), cell=cell_dir.name,
        work_results_treehash=atomic.treehash(cell_dir / "work" / "results"), samples=digests))
    counts = {}
    for r in records:
        counts[r["set"]] = counts.get(r["set"], 0) + 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="the finals results tree (one dir per cell)")
    ap.add_argument("--cell", action="append", default=[], help="convert only these cells (default: every cell with "
                    "harness results)")
    ap.add_argument("--ledger", default=str(REPO / "ledger" / "paper" / "ledger.jsonl"))
    args = ap.parse_args(argv)
    root = Path(args.results)
    if not root.is_dir() or not Path(args.ledger).is_file():
        print(f"convert: {root} is not a directory or ledger {args.ledger} does not exist", file=sys.stderr)
        return 2
    cells = sorted(d for d in root.iterdir() if d.is_dir() and (d / "cell.json").is_file()
                   and (not args.cell or d.name in args.cell))
    if args.cell and {d.name for d in cells} != set(args.cell):
        print(f"convert: no cell dir for {sorted(set(args.cell) - {d.name for d in cells})} in {root}", file=sys.stderr)
        return 2
    for d in cells:
        if not (d / "work" / "results").is_dir():
            print(f"convert: {d.name}: no harness results (work/results): nothing to convert")
            continue
        try:
            counts = convert_cell(d, Path(args.ledger))
        except ConvertRefused as e:
            print(f"CONVERT REFUSED: {d.name}: {e}", file=sys.stderr)
            return 1
        print(f"convert: {d.name}: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
