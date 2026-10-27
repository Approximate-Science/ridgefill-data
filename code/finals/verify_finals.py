#!/usr/bin/env python3
"""Verify a finals results tree: every result file against its sha256 sidecar, every cell against its FINAL- ledger
record. Read-only on the ledger (ledger/paper/ledger.jsonl). Any mismatch: a non-zero exit, errors on stderr and NO
table (BRIEF-COMMON rule 5/6: no table without verified numbers).

A FINAL- record is a labbook hypothesis whose claim ends with the pins finals/register.sh put there
(finals/lib/registration.py): the sha256 of the cell and of every harness config generated from it. Checked: the
results' cell.json hashes to the pinned cell sha256; every pinned config is in the cell dir (harness-<name>.json)
with its pinned sha256, and no harness config is there that the record does not pin. A cell run by another driver
(the Flash-Next tier) has no harness configs and its record pins only the cell. The engine tier as run (a cell with
driver 'c5-kit', registration.ENGINE) is checked by check_engine_cell instead: its record pins the kit, not the cell.

Also checked (fail closed): a .done marker without a .pushed marker with BOTH receipts is a lie;
a sample record's canary_sha256 must match the cell's canary.json; samples/ converted from the harness results
(finals/convert.py) must be the conversion of the work/results there now (its stamp, stamp_problems).

--no-ledger (the public copy of a results tree: no ledger, no dotfiles, and the raw harness records under work/results
may have fields stripped, e.g. LongBench's gold answers): no FINAL- record check; cell.json is checked against its
sidecar instead; the sample records (samples/, what the tables read) are checked against their sidecars and the
conversion stamp; work/results is not checked (its files are covered by the public tree's own manifest).
"""
import argparse
import json
import sys
from pathlib import Path

FINALS = Path(__file__).resolve().parent
sys.path.insert(0, str(FINALS / "lib"))
import atomic                                                # noqa: E402
import registration                                          # noqa: E402

RESULT_DIRS = ("samples", "work/results")
STAMP = "CONVERTED.json"                                     # samples/ written by finals/convert.py from work/results
ENGINE_DIR = FINALS / "engine"                               # the generated engine-tier cells (finals/make_cells.py)


def fail(problems):
    for p in problems:
        print(f"VERIFY FAIL: {p}", file=sys.stderr)
    sys.exit(1)


def stamp_problems(cell_dir, raw=True):
    """[] if samples/ is the current conversion of work/results (finals/convert.py's stamp): the stamp exists, names the
    treehash of work/results as it is now (raw=False: not checked, module doc --no-ledger), and lists exactly the
    sample files there with their sha256."""
    stamp = cell_dir / "samples" / STAMP
    if not stamp.is_file():
        return [f"{cell_dir.name}: harness results (work/results) with no conversion stamp samples/{STAMP}: run "
                "finals/convert.py first"]
    s = json.loads(stamp.read_text())
    if raw and s.get("work_results_treehash") != atomic.treehash(cell_dir / "work" / "results"):
        return [f"{cell_dir.name}: samples/ converts other harness results than work/results holds now (stale): rerun "
                "finals/convert.py"]
    have = {f.relative_to(cell_dir / "samples").as_posix(): atomic.sha256_file(f)
            for f in (cell_dir / "samples").rglob("*.json") if f.name != STAMP}
    listed = s.get("samples") or {}
    bad = sorted(k for k in set(have) | set(listed) if have.get(k) != listed.get(k))
    return [f"{cell_dir.name}: samples/ differs from its conversion stamp ({len(bad)} file(s), e.g. {bad[:3]})"] if bad else []


def check_engine_cell(cell_dir, cell, ledger_path):
    """(problems, info) for an engine-tier cell dir (finals/convert_c5.py wrote it): cell.json is the generated engine
    cell (finals/engine/, finals/make_cells.py engine_cells) and carries what its FINAL- record declares
    (registration.check_engine_record); samples/ and engine.json are what the conversion stamp lists, every source file
    of the kit it names still has the sha256 read then, and the determinism read covers every document
    (engine_tier.final_requires: otherwise the tier is not claimed). No canary and no push: the tier ran on the
    local machine, not on a rented host."""
    name = cell_dir.name
    problems = ([f"{name}: {p}" for p in registration.check_engine_record(cell, ledger_path)]
                if ledger_path is not None else [])
    gen = ENGINE_DIR / f"{cell.get('name')}.json"
    if not gen.is_file() or gen.read_bytes() != (cell_dir / "cell.json").read_bytes():
        problems.append(f"{name}: cell.json is not the generated engine cell {gen}")
    stamp = cell_dir / "samples" / STAMP
    if not stamp.is_file() or not atomic.finished(stamp):
        return problems + [f"{name}: no conversion stamp samples/{STAMP} (with its sidecar): run finals/convert_c5.py"], None
    s = json.loads(stamp.read_text())
    have = {f.relative_to(cell_dir / "samples").as_posix(): atomic.sha256_file(f)
            for f in (cell_dir / "samples").rglob("*.json") if f.name != STAMP}
    bad = sorted(k for k in set(have) | set(s.get("samples") or {}) if have.get(k) != (s.get("samples") or {}).get(k))
    if bad:
        problems.append(f"{name}: samples/ differs from its conversion stamp ({len(bad)} file(s), e.g. {bad[:3]})")
    bad = [f for f in (cell_dir / "samples").rglob("*.json") if not atomic.finished(f)]
    if bad:
        problems.append(f"{name}: {len(bad)} sample file(s) do not match their .sha256 sidecar (e.g. {bad[0].name})")
    engine = cell_dir / "engine.json"
    if not engine.is_file() or not atomic.finished(engine) or atomic.sha256_file(engine) != s.get("engine_sha256"):
        problems.append(f"{name}: engine.json is missing or not the one the conversion stamp names")
    else:
        det = json.loads(engine.read_text()).get("determinism") or {}
        if not det.get("docs") or det.get("bitwise_equal") != det.get("docs"):
            problems.append(f"{name}: determinism read {det.get('bitwise_equal')}/{det.get('docs')} documents bitwise "
                            "equal: the engine tier is not deterministic (protocol engine_tier.final_requires)")
    src = Path(s.get("source") or "")
    if ledger_path is None:
        pass                                                  # --no-ledger: the kit's raw results are not re-checked
    elif not src.is_dir():
        problems.append(f"{name}: the kit's results {src} (the conversion's source) are not reachable to re-check")
    else:
        changed = [r for r, h in sorted((s.get("source_files") or {}).items())
                   if not (src / r).is_file() or atomic.sha256_file(src / r) != h]
        if changed or not s.get("source_files"):
            problems.append(f"{name}: {len(changed)} source file(s) of the kit changed since the conversion "
                            f"(e.g. {changed[:3]}): rerun finals/convert_c5.py")
    info = {"cell": cell.get("name"), "registration": cell.get("registration"), "kind": cell.get("kind"),
            "results": len(have), "arms": len(cell.get("arms", {})), "done": not problems, "host": "local"}
    return problems, info


def check_cell(cell_dir, ledger_path):
    """(problems, info) for one cell dir; info feeds the table printed only when all is well. ledger_path None: the
    --no-ledger checks (module doc)."""
    problems = []
    cell_path = cell_dir / "cell.json"
    if not cell_path.is_file():
        return [f"{cell_dir.name}: no cell.json: not a finals results dir"], None
    cell = json.loads(cell_path.read_text())
    if cell.get("driver") == "c5-kit":
        return check_engine_cell(cell_dir, cell, ledger_path)
    rec = registration.record(cell.get("registration", ""), ledger_path) if ledger_path is not None else None
    if ledger_path is None:
        if not atomic.finished(cell_path):
            problems.append(f"{cell_dir.name}: cell.json does not match its .sha256 sidecar (or has none)")
    elif rec is None:
        problems.append(f"{cell_dir.name}: registration {cell.get('registration')!r} has no "
                         f"hypothesis record in {ledger_path} (FINAL- records are written before the GPU "
                         "is touched, PLAN.md Phase 1)")
    elif registration.superseded(cell.get("registration", ""), ledger_path):
        problems.append(f"{cell_dir.name}: registration {cell['registration']} is superseded "
                        f"({registration.superseded_label(cell['registration'])}): its cell never runs; results under it "
                        "are not finals")
    else:
        pinned = registration.pins_of(rec.get("claim"))
        got = atomic.sha256_file(cell_path)
        if pinned["cell"].get(cell.get("name")) != got:
            problems.append(f"{cell_dir.name}: cell.json (sha256 {got[:16]}) differs from the FINAL- ledger record "
                            f"{cell['registration']} (pins {pinned['cell'].get(cell.get('name'), 'no cell')[:16]})")
        if not pinned["config"] and list(cell_dir.glob("harness-*.json")):
            problems.append(f"{cell_dir.name}: the FINAL- ledger record {cell['registration']} pins no harness config, "
                            "but the cell dir holds some")
        for name, want in pinned["config"].items():
            f = cell_dir / f"harness-{name}.json"
            have = atomic.sha256_file(f) if f.is_file() else None
            if have != want:
                problems.append(f"{cell_dir.name}: harness config {name} differs from the FINAL- ledger record "
                                f"({'missing' if have is None else have[:16]} vs pinned {want[:16]})")
        extra = sorted(f.name for f in cell_dir.glob("harness-*.json")
                       if f.name[len("harness-"):-len(".json")] not in pinned["config"])
        if extra:
            problems.append(f"{cell_dir.name}: harness configs the FINAL- record does not pin: {extra}")
    canary = cell_dir / "canary.json"
    canary_sha = None
    if not canary.is_file():
        problems.append(f"{cell_dir.name}: no canary.json: the host canary never ran")
    else:
        canary_sha = atomic.sha256_file(canary)
        arms = json.loads(canary.read_text()).get("arms", {})
        if not arms.get("exact", {}).get("logits_sha256"):
            problems.append(f"{cell_dir.name}: canary.json has no exact-arm logits_sha256")
    n_results = 0
    for sub in (RESULT_DIRS if ledger_path is not None else ("samples",)):
        d = cell_dir / sub
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.json")):
            if f.name.endswith(".sha256") or f.parent.name == "canary":   # the harness's own canary refs
                continue
            n_results += 1
            if not atomic.finished(f):
                problems.append(f"{cell_dir.name}: {f.relative_to(cell_dir)} does not match its "
                                ".sha256 sidecar (or has none): tampered, truncated or half-written")
            rec = json.loads(f.read_text())
            if isinstance(rec, dict) and rec.get("canary_sha256") and rec["canary_sha256"] != canary_sha:
                problems.append(f"{cell_dir.name}: {f.name} was run against canary "
                                f"{rec['canary_sha256'][:12]}, but the cell's canary.json is "
                                f"{(canary_sha or '')[:12]}")
    if n_results == 0:
        problems.append(f"{cell_dir.name}: no result files under {' or '.join(RESULT_DIRS)}")
    if (cell_dir / "work" / "results").is_dir() and (cell_dir / "samples" / STAMP).is_file():
        problems += stamp_problems(cell_dir, raw=ledger_path is not None)
    if (cell_dir / ".done").exists() and (FINALS / "push.sh").exists():   # a checkout without push.sh never pushes
        pushed = cell_dir / ".pushed"
        if not pushed.is_file():
            problems.append(f"{cell_dir.name}: .done exists but .pushed does not: the cell was marked "
                            "done without a successful push to both destinations")
        else:
            p = json.loads(pushed.read_text())
            if "ssh" not in p or "hf" not in p:
                problems.append(f"{cell_dir.name}: .pushed does not carry both receipts (ssh, hf)")
    info = {"cell": cell.get("name"), "registration": cell.get("registration"),
            "kind": cell.get("kind"), "results": n_results,
            "arms": len(cell.get("arms", [])), "done": (cell_dir / ".done").exists(),
            "host": json.loads((cell_dir / "host.json").read_text())["host"]["host"]
            if (cell_dir / "host.json").is_file() else "?"}
    return problems, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="the finals results tree (one dir per cell)")
    ap.add_argument("--ledger", default=str(FINALS.parent / "ledger" / "paper" / "ledger.jsonl"))
    ap.add_argument("--require-done", action="store_true",
                    help="a cell without a .done marker is a failure too")
    ap.add_argument("--no-ledger", action="store_true",
                    help="the public copy of a results tree: no FINAL- record checks (module doc)")
    args = ap.parse_args()
    root = Path(args.results)
    if not root.is_dir():
        fail([f"{root} is not a directory"])
    ledger = None if args.no_ledger else Path(args.ledger)
    if ledger is not None and not ledger.is_file():
        fail([f"ledger {ledger} does not exist (--ledger, or --no-ledger for a public copy of the tree)"])

    all_problems, infos = [], []
    cells = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
                   and (d / "cell.json").exists())
    if not cells:
        fail([f"{root} contains no cell results dirs (a cell dir has cell.json)"])
    for d in cells:
        problems, info = check_cell(d, ledger)
        if args.require_done and info and not info["done"]:
            problems.append(f"{d.name}: no .done marker (--require-done)")
        all_problems += problems
        if info:
            infos.append(info)
        else:
            fail(problems)                                            # no cell.json: not a cell dir
    if all_problems:
        fail(all_problems)

    # every check passed: the table (and only then)
    print(f"{'cell':<28} {'registration':<22} {'kind':<18} {'results':>7} {'arms':>4} {'done':>5} host")
    for i in infos:
        print(f"{i['cell']:<28} {i['registration']:<22} {i['kind']:<18} {i['results']:>7} "
              f"{i['arms']:>4} {str(i['done']):>5} {i['host']}")
    print(f"verify_finals: {len(infos)} cell(s) OK against {ledger or 'their sidecars (--no-ledger)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())