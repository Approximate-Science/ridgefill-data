#!/usr/bin/env python3
"""Convert the C5 kit's results (the Flash-Next engine tier, FINAL-FLASHNEXT-C5) into the finals sample records
finals/tables.py reads: the one "flashnext" cell. Read-only on the kit's results, the ledger and protocol.json; writes
one cell dir under DEST. The engine-tier counterpart of finals/convert.py (which converts the reference harness).

Input: the kit's results dir of one run (run_c5.sh MODE final: results/<arm dir>/{ppl-a, ppl-b, ruler-a, speed,
control, boot, tails.json}), arm dirs and passes as the engine cell declares them (finals/engine/<cell>.json, generated
by finals/make_cells.py engine_cells(); its 'arms' map the paper's arm names to them). Other passes in the arm dirs
(ppl-a-chat-labelled-*, ppl-a-fresh-long-*) are not part of the cell and are never read.
Order (each step fails closed and refuses by name; nothing is written unless every record converts):
  1. cell      the engine cell must be the generated one (finals/engine/) and carry exactly what its FINAL- record
               declares (registration.check_engine_record); its protocol must be protocol.json's version.
  2. files     every file read is checked against its .sha256 sidecar ('<hex>  <name>'); each pass's _set.json lists
               exactly the samples present (no missing, no extra); tails.json reports no problem, every RidgeFill-arm
               sample was served on the RidgeFill path with the cell's tail and split (approximated, tail read back
               and matched), every exact sample on the exact path.
  3. boot      each arm's boot/overlay-check.txt shows the determinism overlay file with protocol.json's sha256
               (engine_tier.c5_image.overlay), and boot/ced-policy.txt the arm's declared CED policy (none for exact).
  4. records   perplexity (set longppl, unit = the locked file): novel_nll = kva.locked.nll over the stored novel mask,
               checked against the kit's own nll_novel (two instruments); the prompt (ids_sha256, n, source_sha256)
               must be the same in every arm and the source the locked manifest's file at its Flash-Next token count.
               RULER (set ruler, unit '<locked file>#<row>'): score = kva.metrics.ruler_row(metric, pred, refs),
               checked against the kit's; needle_filled = every needle ends at or before n - T (T the cell's tail),
               checked against the RidgeFill arms' stored value; wall_s (the request's wall time) kept for the 64K
               request-time readout. TTFT (set ttft, unit '<arm>/<length>#round<r>'): each RidgeFill arm's speed file
               holds its own timed rounds and the same boot's exact path, interleaved per round; the warm-up round is
               dropped, the medians are checked against the kit's.
  5. engine    engine.json: the determinism read (exact's same-boot perplexity passes a and b, bitwise equal stored
               log-prob lists per document) and the machinery control (quality mode's boot on its exact path against
               exact's pass a), with the overlay and the engine image.
  6. write     DEST/<cell>/cell.json (the engine cell's bytes), engine.json, samples/<set>/<unit>.json with sidecars,
               then samples/CONVERTED.json: converter sha256, the source dir and the sha256 of every source file read,
               every sample file's sha256 (verify_finals.py check_engine_cell re-checks all of them). Deterministic.
Usage: convert_c5.py C5_RESULTS DEST [--cell finals/engine/<cell>.json] [--ledger L] [--protocol P]
Exit: 0 converted; 1 refused; 2 usage.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

FINALS = Path(__file__).resolve().parent
REPO = FINALS.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FINALS / "lib"))
import atomic                                                # noqa: E402
import registration                                          # noqa: E402

STAMP = "CONVERTED.json"
ENGINE = "engine.json"
ENGINE_DIR = FINALS / "engine"                               # the generated engine cells (finals/make_cells.py)
TOL = 1e-9


class ConvertRefused(Exception):
    """A named refusal: nothing of the cell is written."""


def safe(sample):
    return sample.replace("/", "__").replace("#", "_")


class Source:
    """The kit's results dir; every file read is checked against its sidecar and remembered for the stamp."""

    def __init__(self, root):
        self.root = Path(root)
        self.read = {}

    def json(self, rel, sidecar=True):
        p = self.root / rel
        if not p.is_file():
            raise ConvertRefused(f"{rel}: missing in {self.root}")
        digest = atomic.sha256_file(p)
        if sidecar:
            sc = Path(f"{p}.sha256")
            got = sc.read_text().split()[0] if sc.is_file() and sc.read_text().split() else None
            if got != digest:
                raise ConvertRefused(f"{rel}: does not match its .sha256 sidecar ({'none' if got is None else got[:12]} vs "
                                     f"{digest[:12]}): tampered, truncated or half-written")
        self.read[rel] = digest
        return json.loads(p.read_text())

    def text(self, rel):
        p = self.root / rel
        if not p.is_file():
            raise ConvertRefused(f"{rel}: missing in {self.root}")
        self.read[rel] = atomic.sha256_file(p)
        return p.read_text(errors="replace")

    def samples(self, rel_dir, key, expected=None, prefix=False):
        """{name: record} of one pass: exactly the samples its _set.json lists under key; a pass the kit ran in several
        invocations into one dir (RULER 8K-32K, then 64K: _set.json lists the last one) must hold exactly `expected`,
        the names the cell declares, its _set.json's among them; prefix=True: the kit ran the first len(expected) of the
        names its _set.json lists (the machinery control)."""
        listed = self.json(f"{rel_dir}/_set.json")[key]
        have = sorted(p.name[:-len(".json")] for p in (self.root / rel_dir).glob("*.json") if p.name != "_set.json")
        want = sorted(expected) if expected is not None else sorted(listed)
        if want != have or not (listed[:len(want)] == list(expected) if prefix else set(listed) <= set(have)):
            extra, missing = sorted(set(have) - set(want)), sorted((set(want) | set(listed)) - set(have))
            raise ConvertRefused(f"{rel_dir}: samples differ from the declared set (missing {missing[:3]}, extra {extra[:3]})")
        out = {}
        for name in want:
            r = self.json(f"{rel_dir}/{name}.json")
            if r.get("name") != name:
                raise ConvertRefused(f"{rel_dir}/{name}.json: record name {r.get('name')!r}")
            out[name] = r
        return out


# ---------------------------------------------------------------------------------------------- 1. the cell
def load_cell(cell_path, ledger, protocol):
    raw = Path(cell_path).read_bytes()
    cell = json.loads(raw)
    gen = ENGINE_DIR / f"{cell.get('name')}.json"
    if not gen.is_file() or gen.read_bytes() != raw:
        raise ConvertRefused(f"{cell_path}: not the generated engine cell {gen} (finals/make_cells.py engine_cells)")
    problems = registration.check_engine_record(cell, ledger)
    if problems:
        raise ConvertRefused("; ".join(problems))
    if cell["protocol"] != protocol["version"]:
        raise ConvertRefused(f"{cell['name']}: protocol {cell['protocol']} is not protocol.json's {protocol['version']}")
    return cell, raw


# ---------------------------------------------------------------------------------------------- 2-3. files, boot
def check_tails(src, cell, arm, spec, names_by_pass):
    tails = src.json(f"{spec['dir']}/tails.json")
    if tails.get("problems"):
        raise ConvertRefused(f"{spec['dir']}/tails.json reports problems: {tails['problems'][:3]}")
    ridge = arm != "exact" and arm != "exact-rerun"
    for pass_, names in names_by_pass.items():
        for n in names:
            t = tails["samples"].get(f"{pass_}/{n}")
            if t is None:
                raise ConvertRefused(f"{spec['dir']}/tails.json: no entry for {pass_}/{n}")
            if ridge and not (t.get("approximated") and t.get("match") and t.get("tail") == cell["tail"]
                              and t.get("split") == cell["split_layer"]):
                raise ConvertRefused(f"{spec['dir']} {pass_}/{n}: not served on the RidgeFill path with tail "
                                     f"{cell['tail']} at split {cell['split_layer']} ({t})")
            if not ridge and t.get("approximated") is not False:
                raise ConvertRefused(f"{spec['dir']} {pass_}/{n}: an exact-arm sample served approximated ({t})")


def check_boot(src, cell, spec):
    check = src.text(f"{spec['dir']}/boot/overlay-check.txt")
    for path, digest in cell["engine"]["overlay"].items():
        if not any(ln.startswith("OK") and path in ln and digest in ln for ln in check.splitlines()):
            raise ConvertRefused(f"{spec['dir']}/boot/overlay-check.txt: no 'OK' line for the overlay {path} at {digest[:12]}")
    policy = src.text(f"{spec['dir']}/boot/ced-policy.txt")
    on = [ln for ln in policy.splitlines() if "CED on (" in ln]
    if spec["policy"] is None and on:
        raise ConvertRefused(f"{spec['dir']}: an exact boot with RidgeFill on ({on[0][-120:]})")
    if spec["policy"] is not None and not any(spec["policy"] in ln for ln in on):
        raise ConvertRefused(f"{spec['dir']}/boot/ced-policy.txt: no '{spec['policy']}' line")
    if spec["policy"] == "CED on (quality)" and not any(f"-s{cell['declared_share']}" in ln for ln in on):
        raise ConvertRefused(f"{spec['dir']}: quality mode's exact rows are not at the declared share "
                             f"{cell['declared_share']}%")


def manifest():
    rows = (REPO / "datasets" / "test-set-locked" / "MANIFEST.tsv").read_text().splitlines()
    head = rows[0].split("\t")
    return {r.split("\t")[0]: dict(zip(head, r.split("\t"))) for r in rows[1:] if r.strip()}


# ---------------------------------------------------------------------------------------------- 4. records
def ppl_metrics(r, cell):
    from kva import locked
    lp, novel = r["logprobs"], r["novel"]
    if len(lp) != len(novel) or len(lp) != cell["data"]["longppl"]["scored"] or r.get("cont_len") != len(lp):
        raise ConvertRefused(f"{r['name']}: {len(lp)} log-probs, {len(novel)} mask entries, cont_len {r.get('cont_len')}; "
                             f"the cell scores {cell['data']['longppl']['scored']}")
    if sum(novel) != r["novel_count"] or not any(novel):
        raise ConvertRefused(f"{r['name']}: novel mask counts {sum(novel)}, the record says {r['novel_count']}")
    v = locked.nll(lp, novel)
    if abs(v - r["nll_novel"]) > TOL:
        raise ConvertRefused(f"{r['name']}: novel-token NLL {v!r} from the log-probs, the kit stored {r['nll_novel']!r} "
                             "(two instruments disagree)")
    return dict(novel_nll=v, n_novel=sum(novel), all_nll=locked.nll(lp))


def ruler_metrics(r, cell, ridge):
    from kva import metrics as MT
    s = MT.ruler_row(r["metric"], r["pred"], r["refs"])
    if s != r["score"]:
        raise ConvertRefused(f"{r['name']}: the official scorer gives {s!r}, the kit stored {r['score']!r}")
    filled = all(e <= r["n"] - cell["tail"] for _, e in r["needles"])
    if ridge and filled != r["needle_filled"]:
        raise ConvertRefused(f"{r['name']}: needle_filled {filled} at tail {cell['tail']}, the kit stored "
                             f"{r['needle_filled']}")
    return dict(score=s, wall_s=float(r["wall_s"])), filled


def same_prompt(unit, have, r, fields):
    for k in fields:
        if k in have and have[k] != r[k]:
            raise ConvertRefused(f"{unit}: {k} differs across arms ({have[k]!r} vs {r[k]!r}): not one pairing unit")
        have[k] = r[k]


def ppl_records(src, cell, man):
    out = {}
    for arm, spec in cell["arms"].items():
        recs = src.samples(f"{spec['dir']}/{spec['ppl']}", "docs")
        if len(recs) != cell["data"]["longppl"]["docs"]:
            raise ConvertRefused(f"{arm}: {len(recs)} perplexity documents, the cell declares {cell['data']['longppl']['docs']}")
        check_tails(src, cell, arm, spec, {spec["ppl"]: recs})
        for name, r in recs.items():
            unit = r["source"]
            m = man.get(unit)
            if m is None or m["sha256"] != r["source_sha256"] or int(m["qwen3.8-flash-next"]) != r["n"]:
                raise ConvertRefused(f"{name}: {unit} is not the locked manifest's file at its Flash-Next token count")
            if r["length"] > cell["data"]["longppl"]["max_nominal"]:
                raise ConvertRefused(f"{name}: nominal length {r['length']} over the cell's {cell['data']['longppl']['max_nominal']}")
            rec = out.setdefault(unit, dict(set="longppl", sample=unit, doc=unit.rsplit("/", 1)[-1].rsplit(".", 1)[0],
                                            domain=r["task"], length=r["length"], tokens=r["n"], arms={}, _p={}))
            same_prompt(unit, rec["_p"], r, ("ids_sha256", "source_sha256", "n", "task", "length"))
            rec["arms"][arm] = {"-": ppl_metrics(r, cell)}
            rec.setdefault("_lp", {})[arm] = r["logprobs"]
    return out


def ruler_records(src, cell):
    d = cell["data"]["ruler"]
    want = len(d["tasks"]) * len(d["lengths"]) * 25
    if d["rows"] != want:
        raise ConvertRefused(f"the cell declares {d['rows']} RULER rows at 8K-32K, its tasks x lengths x 25 are {want}")
    names = [f"ruler-{t}-{L}-{i:02d}" for t in d["tasks"] for L in d["lengths"] for i in range(25)] + \
            [f"ruler-{d['long']['task']}-{d['long']['length']}-{i:02d}" for i in range(d["long"]["rows"])]
    out = {}
    for arm, spec in cell["arms"].items():
        if "ruler" not in spec:
            continue
        recs = src.samples(f"{spec['dir']}/{spec['ruler']}", "rows", names)
        if len(recs) != d["rows"] + d["long"]["rows"]:
            raise ConvertRefused(f"{arm}: {len(recs)} RULER rows, the cell declares {d['rows']} + {d['long']['rows']}")
        check_tails(src, cell, arm, spec, {spec["ruler"]: recs})
        for name, r in recs.items():
            ok = (r["task"] in d["tasks"] and r["length"] in d["lengths"]) or \
                 (r["task"] == d["long"]["task"] and r["length"] == d["long"]["length"])
            if not ok or r.get("prefix") != "instruct":
                raise ConvertRefused(f"{name}: task {r['task']} at {r['length']} in format {r.get('prefix')!r} is not a "
                                     "declared row (instruct format)")
            unit = f"{r['source']}#{r['row']:02d}"
            m, filled = ruler_metrics(r, cell, arm != "exact")
            rec = out.setdefault(unit, dict(set="ruler", sample=unit, task=r["task"], length=r["length"], tokens=r["n"],
                                            needle_filled=filled, arms={}, _p={}))
            same_prompt(unit, rec["_p"], r, ("ids_sha256", "source_sha256", "n", "refs", "needles"))
            rec["arms"][arm] = {"-": m}
    return out


def ttft_records(src, cell):
    d = cell["data"]["ttft"]
    out, prompt = {}, {}
    for arm, spec in cell["arms"].items():
        if "speed" not in spec:
            continue
        for L in d["lengths"]:
            r = src.json(f"{spec['dir']}/{spec['speed']}/speed-{L}.json")
            if r["length"] != L or r["n"] != L or r["source"] != d["prompt"] or r.get("tail") != cell["tail"]:
                raise ConvertRefused(f"{spec['dir']} speed {L}: length {r['length']}, n {r['n']}, source {r['source']}, "
                                     f"tail {r.get('tail')}: not the declared prompt prefix at the cell's tail")
            same_prompt(f"ttft {L}", prompt.setdefault(L, {}), r, ("ids_sha256",))
            runs = {k: r["runs"][k] for k in ("arm", "exact-switch")}
            for k, rs in runs.items():
                warm = [x for x in rs if x.get("warmup")]
                if len(warm) != d["warmup"] or not rs[0].get("warmup") or len(rs) - len(warm) != d["timed"]:
                    raise ConvertRefused(f"{spec['dir']} speed {L} {k}: {len(warm)} warm-up and {len(rs) - len(warm)} "
                                         f"timed rounds, the cell declares {d['warmup']} + {d['timed']}")
            arm_t = [x["ttft_s"] for x in runs["arm"] if not x.get("warmup")]
            ex_t = [x["ttft_s"] for x in runs["exact-switch"] if not x.get("warmup")]
            for what, xs, stored in (("arm", arm_t, r["arm_ttft_median_s"]),
                                     ("exact-switch", ex_t, r["exact-switch_ttft_median_s"])):
                if abs(statistics.median(xs) - stored) > TOL:
                    raise ConvertRefused(f"{spec['dir']} speed {L}: median {what} TTFT {statistics.median(xs)!r}, the "
                                         f"kit stored {stored!r}")
            for i, (a, e) in enumerate(zip(arm_t, ex_t)):
                unit = f"{arm}/{L}#round{i}"
                out[unit] = dict(set="ttft", sample=unit, length=L, round=i, boot=spec["dir"],
                                 arms={"exact": {"-": dict(ttft_s=e)}, arm: {"-": dict(ttft_s=a)}})
    return out


# ---------------------------------------------------------------------------------------------- 5. engine
def engine_read(src, cell, ppl):
    ex, rerun = cell["arms"]["exact"], cell["arms"]["exact-rerun"]
    if ex["dir"] != rerun["dir"]:
        raise ConvertRefused("exact and exact-rerun are not the same boot (arm dir)")
    same = sorted(u for u, r in ppl.items() if r["_lp"].get("exact") is not None
                  and r["_lp"]["exact"] == r["_lp"].get("exact-rerun"))
    ctrl_spec = next(((a, s) for a, s in cell["arms"].items() if "control" in s), None)
    control = None
    if ctrl_spec:
        arm, spec = ctrl_spec
        listed = src.json(f"{spec['dir']}/{spec['control']}/_set.json")["docs"]
        recs = src.samples(f"{spec['dir']}/{spec['control']}", "docs", listed[:cell["data"]["control"]["docs"]],
                           prefix=True)
        tails = src.json(f"{spec['dir']}/tails.json")["samples"]
        bad = [n for n in recs if (tails.get(f"{spec['control']}/{n}") or {}).get("approximated") is not False]
        if bad:
            raise ConvertRefused(f"{spec['dir']} control: {bad[:3]} not served on the exact path")
        eq = sorted(r["source"] for r in recs.values() if r["source"] in ppl
                    and r["logprobs"] == ppl[r["source"]]["_lp"]["exact"])
        control = dict(arm=arm, boot=spec["dir"], docs=len(recs), bitwise_equal=len(eq), units=eq)
    return dict(cell=cell["name"], image=cell["engine"]["image"], image_id=cell["engine"]["id"],
                overlay=cell["engine"]["overlay"], protocol=cell["protocol"],
                determinism=dict(arms=["exact", "exact-rerun"], boot=ex["dir"], docs=len(ppl), bitwise_equal=len(same),
                                 what="stored prompt log-prob lists of the same-boot perplexity passes, per document"),
                control=control)


# ---------------------------------------------------------------------------------------------- the cell
def convert(c5_results, dest, cell_path, ledger, protocol_path=REPO / "protocol.json"):
    protocol = json.loads(Path(protocol_path).read_text())
    cell, raw = load_cell(cell_path, ledger, protocol)
    src = Source(c5_results)
    for spec in cell["arms"].values():
        check_boot(src, cell, spec)
    ppl = ppl_records(src, cell, manifest())
    ruler = ruler_records(src, cell)
    ttft = ttft_records(src, cell)
    engine = engine_read(src, cell, ppl)
    stamp = dict(protocol=cell["protocol"], tier=cell["tier"], cell=cell["name"])
    records = []
    for r in list(ppl.values()) + list(ruler.values()) + list(ttft.values()):
        if "exact" not in r["arms"]:
            raise ConvertRefused(f"{r['set']} {r['sample']}: no exact arm")
        records.append({**{k: v for k, v in r.items() if not k.startswith("_")}, **stamp})
    out = Path(dest) / cell["name"]
    want = {f"{r['set']}/{safe(r['sample'])}.json": r for r in records}
    if len(want) != len(records):
        raise ConvertRefused("two units map to one file name")
    atomic.write_bytes(out / "cell.json", raw)
    atomic.write_result(out / ENGINE, engine)
    digests = {rel: atomic.write_result(out / "samples" / rel, r) for rel, r in sorted(want.items())}
    for f in sorted((out / "samples").rglob("*.json")):
        rel = f.relative_to(out / "samples").as_posix()
        if rel not in want and rel != STAMP:
            f.unlink()
            Path(f"{f}.sha256").unlink(missing_ok=True)
    atomic.write_result(out / "samples" / STAMP, dict(
        converter="finals/convert_c5.py", converter_sha256=atomic.sha256_file(Path(__file__)), cell=cell["name"],
        source=str(Path(c5_results).resolve()), source_files=dict(sorted(src.read.items())),
        engine_sha256=atomic.sha256_file(out / ENGINE), samples=digests))
    counts = {}
    for r in records:
        counts[r["set"]] = counts.get(r["set"], 0) + 1
    return counts, engine


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("c5_results", help="the C5 kit's results dir of the run (results/<arm dir>/...)")
    ap.add_argument("dest", help="the finals results tree to write the cell dir into")
    ap.add_argument("--cell", default=str(ENGINE_DIR / "qwen3.8-flash-next-c5.json"))
    ap.add_argument("--ledger", default=str(REPO / "ledger" / "paper" / "ledger.jsonl"))
    ap.add_argument("--protocol", default=str(REPO / "protocol.json"))
    args = ap.parse_args(argv)
    if not Path(args.c5_results).is_dir() or not Path(args.ledger).is_file() or not Path(args.cell).is_file():
        print("convert_c5: C5_RESULTS, --cell and --ledger must exist", file=sys.stderr)
        return 2
    try:
        counts, engine = convert(args.c5_results, args.dest, args.cell, args.ledger, args.protocol)
    except ConvertRefused as e:
        print(f"CONVERT REFUSED: {e}", file=sys.stderr)
        return 1
    det, ctl = engine["determinism"], engine["control"]
    print(f"convert_c5: {engine['cell']}: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
          + f"; determinism {det['bitwise_equal']}/{det['docs']} documents bitwise equal"
          + (f"; control {ctl['bitwise_equal']}/{ctl['docs']}" if ctl else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
