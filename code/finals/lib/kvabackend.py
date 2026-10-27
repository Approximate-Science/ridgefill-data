#!/usr/bin/env python3
"""The kva backend: turn a cell into reference-harness runs (scripts/evaluate.py) under protocol 1.1.

harness_configs(cell) builds one evaluate.py config per (set, variant, model) group of the cell's runs (cell v2,
finals/cell.schema.json): a prompt's record then holds all its arms at every split with one exact. Every arm is
kva.protocol.template(kind, ...) for the arm's protocol kind (PAPER: the paper's arm names; the five of protocol.json
arm_aliases plus the v1.1 kinds), so every arm is a protocol 1.1 row by construction, and the CPU test
(tests/test_finals_wiring.py) validates every generated config with kva.protocol.validate under a ledger holding the
cell's FINAL- id. Only the names, the paths and the declared values change in the template's spec: names are
<paper name>-s<layer>; projectors are $KVA_PROJ_DIR/<model>-sweep-s<layer>.safetensors (the protocol's sweep naming);
matched controls (+random, +tailmatch) name their rows arm of the same run; nofill+tailbudget takes the cell's b.
Splits: a fraction of the model's layers on the boundary below (protocol.json depths_note): Llama and Qwen3.5 1/4, 3/8,
1/2, 5/8 = 8, 12, 16, 20; Qwen3-8B 9, 13, 18, 22; pop-native at the model's pop_split; SwiftKV at 16. The exact arm
runs in every config (evaluate.py) and is the shared reference of the run's pairing unit: one record per prompt with
every arm, written atomically by kva/prov.write_json.

run_cell(cell, cell_dir, python) refuses by name before any GPU work: a projector that is missing or STALE (its
metadata rows are not the clean calibration list's row count the cell declares, or its calib_exclude_sha256 differs:
fitted before the T0.2 exclusion list, cba2d91), a derived-set manifest (datasets/derived) whose sha256 is not the
cell's, a missing env var. Then per run: the config is written (canonical bytes: their sha256 is what the FINAL record
pins, finals/lib/registration.py), evaluate.py runs once, and every record it wrote gets a .sha256 sidecar.

Named refusals that remain (hardware or engine paths that do not exist; never a fallback):
  TODO-HARNESS-VLLM-TTFT   the speed cells (T3.1): scripts/ttft.py exists, but only exact has an engine path; the
                           Llama/Qwen RidgeFill path in tcclaviger's vLLM is not built (checks/harness-NOTES.md sizes it)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

FINALS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINALS / "lib"))
import atomic                                                # noqa: E402

PAPER = {"nofill": "skip", "identity": "noproj", "ridgefill": "kva", "pop-native": "pop-native",
         "identity+rows": "noproj+qattn10", "ridgefill+rows": "kva+qattn10",
         "identity+qblind": "noproj+qblind10", "ridgefill+qblind": "kva+qblind10",
         "identity+random": "noproj+random", "ridgefill+random": "kva+random",
         "identity+tailmatch": "noproj+tailmatch", "ridgefill+tailmatch": "kva+tailmatch",
         "nofill+tailbudget": "skip+tailbudget",
         "ridgefill+kexact": "kva+kexact", "ridgefill+vexact": "kva+vexact", "kvexact": "kvexact",
         "ridgefill+ktail": "kva+ktail", "ridgefill+vtail": "kva+vtail",
         "swiftkv": "swiftkv", "swiftkv-native": "swiftkv-native"}
ROWS_OF = {"noproj": "identity+rows", "kva": "ridgefill+rows"}        # a matched control's rows arm, by fill
EVALUATE_SETS = ("ruler", "longbench", "longppl", "decode", "multiturn")


class HarnessGap(NotImplementedError):
    """A named missing piece: the cell can not run until it exists. No approximation."""


class StaleProjector(RuntimeError):
    """A projector that is not the clean refit the cell declares (finals/lib/kvabackend.py doc)."""


def repo():
    r = os.environ.get("KVA_REPO")
    if not r:
        raise HarnessGap("KVA_REPO is not set: the kva backend needs the harness repo root")
    if r not in sys.path:
        sys.path.insert(0, r)
    return Path(r)


def _protocol():
    repo()
    from kva import protocol as PR
    return PR, PR.load(repo() / "protocol.json")


def layer_of(model_entry, frac):
    """A split fraction on the boundary below (protocol.json depths_note): floor(num x layers / den)."""
    num, den = (int(x) for x in frac.split("/"))
    return num * model_entry["layers"] // den


def _model_of(cell, run):
    return cell["swiftkv_model"]["name"] if run.get("model") == "swiftkv" else cell["model"]["name"]


def arm_spec(PR, p, cell, run, paper, layer):
    """One evaluate.py arm spec for a paper arm name at this layer: kva.protocol.template() with names, paths and the
    cell's declared values filled in."""
    kind = PAPER.get(paper)
    if kind is None:
        raise HarnessGap(f"unknown arm {paper!r} (finals/lib/kvabackend.py PAPER)")
    model = _model_of(cell, run)
    m = p["models"][model]
    split = m["pop_split"] if kind == "pop-native" else layer
    tail = None if kind in PR.TAILLESS else cell["tail"]
    spec = PR.template(kind, model, split, tail, p)
    spec["name"] = f"{paper}-s{split}"
    if "projector" in spec:
        declared = {f["layer"]: f["file"] for f in cell.get("projector", {}).get("files", [])}
        if declared.get(split) != spec["projector"]:
            raise HarnessGap(f"cell {cell['name']}: arm {paper} at layer {split} reads {spec['projector']}, which the "
                             f"cell's projector.files does not declare ({declared.get(split)})")
        spec["projector"] = f"$KVA_PROJ_DIR/{spec['projector']}"
    sel = spec.get("select", {})
    if "match" in sel:
        sel["match"] = f"{ROWS_OF[kind.split('+')[0]]}-s{split}"
    if kind == "skip+tailbudget":
        sel["budget"] = cell["tailbudget"]["b"]
    return spec


def _group_key(run):
    return (run["set"], run.get("variant"), run.get("model"), tuple(run.get("tasks", ())),
            tuple(run.get("lengths", ())))


def harness_config(cell, runs):
    """(config, evaluate.py set) of the cell's runs that share a set, variant, model and row selection (tasks,
    lengths): ONE evaluate.py config, so a prompt's record holds every arm of every split of those runs and their one
    exact (OUTLINE.md: the pairing unit is one sample with all its arms; evaluate.py shares the lower layers per prompt
    across the splits). The groups of a set partition its rows (make_cells.py), so no unit is in two configs.
    Declarative; validated under protocol 1.1 by the CPU test. Named after the group's first run."""
    run = runs[0]
    if run["set"] == "ttft":
        raise HarnessGap("TODO-HARNESS-VLLM-TTFT: the speed cells are T3.1's interleaved vLLM TTFT runs; "
                         "scripts/ttft.py has an engine path for exact only (no RidgeFill path in the engine yet: "
                         "checks/harness-NOTES.md)")
    PR, p = _protocol()
    model = _model_of(cell, run)
    m = p["models"][model]
    arms, parts = [], []
    for r in runs:
        layer = 16 if r.get("model") == "swiftkv" else layer_of(m, r["split"])
        if layer not in m["splits"]:
            raise HarnessGap(f"split {r['split']} of {model} is layer {layer}, not one of protocol.json's {m['splits']}")
        arms += [arm_spec(PR, p, cell, r, a, layer) for a in r["arms"] if a != "exact"]
        parts.append(f"{r['id']} ({r['role']}, split {r['split']} = layer {layer})")
    cfg = {
        "name": f"{cell['model']['name']}-{run['id']}",
        "protocol": cell["code_tag"].removeprefix("protocol-v"),
        "registration": [cell["registration"]],
        "model": f"$KVA_MODEL_DIR/{model}",
        "tokenizer": f"$KVA_MODEL_DIR/{cell['model']['name']}",
        "niah_format": m.get("niah_format"),
        "note": f"finals cell {cell['name']}, set {run['set']}, runs {', '.join(parts)}; generated by "
                f"finals/lib/kvabackend.py from the cell; exact runs as always",
        "arms": arms,
        **cell.get("caps", {}),
    }
    if run["set"] == "longppl":
        cfg["keytoken_tail"] = cell["keytoken"]["tail"]
    if "dataset" in cell:                  # another frozen set instead of the locked one (kva/locked.py DATASETS)
        cfg.update(dataset=cell["dataset"], dataset_manifest_sha256=cell["dataset_manifest_sha256"])
    if run["set"] == "decode":
        cfg.update(decode_from=cell["decode"]["from"], decode_steps=cell["decode"]["steps"],
                   only=",".join(cell["decode"]["only"]))
    if run["set"] == "multiturn":
        cfg.update(multiturn_from=cell["secondq"]["from"], multiturn_turns=cell["secondq"]["turns"])
    if run.get("variant"):
        cfg["ruler_variant"] = run["variant"]
    if run.get("tasks"):
        cfg["ruler_tasks"] = list(run["tasks"])
    if run.get("lengths"):
        cfg["ruler_lengths"] = list(run["lengths"])
    names = [a["name"] for a in arms]
    if len(set(names)) != len(names):
        raise HarnessGap(f"cell {cell['name']}: an arm appears twice in {cfg['name']}: {names}")
    return cfg, run["set"]


def harness_configs(cell):
    """[(config, set)]: one config per (set, variant, model, tasks, lengths) group of the cell's runs, in first-run
    order (a named refusal for a group with no harness path)."""
    if cell["kind"] == "flashnext":
        raise HarnessGap("TODO-FLASHNEXT-DRIVER: the flashnext cell runs on the engine's own driver (qfn), not on the "
                         "reference harness; its configs and pins come from that driver")
    groups = {}
    for r in cell["runs"]:
        groups.setdefault(_group_key(r), []).append(r)
    return [harness_config(cell, runs) for runs in groups.values()]


def config_bytes(cfg):
    """The bytes run_cell writes and evaluate.py reads (atomic.write_result's format): their sha256 is pinned."""
    return json.dumps(cfg, indent=1, sort_keys=True).encode() + b"\n"


# ---- run-time checks (before any GPU work)
def projector_files(cfg):
    return sorted({a["projector"] for a in cfg["arms"] if "projector" in a})


def check_projector(path, provenance, sha256=None):
    """StaleProjector unless the file exists, is the bytes the cell pins (sha256, when given) and its metadata says it
    is the clean refit the cell declares (rows; docs and calib_exclude_sha256 when it records them)."""
    from kva.kva import Projector
    if not Path(path).is_file():
        raise StaleProjector(f"projector {path} does not exist (KVA_PROJ_DIR): the clean refit is mounted there")
    if sha256 is not None and atomic.sha256_file(path) != sha256:
        raise StaleProjector(f"StaleProjector: {Path(path).name} is not the file the cell pins (sha256 {sha256[:16]})")
    meta = Projector.meta_of(path)
    if "docs" in meta and meta["docs"] != str(provenance["calib_docs"]):
        raise StaleProjector(f"StaleProjector: {Path(path).name} was fitted on {meta['docs']} calibration documents, "
                             f"the clean list has {provenance['calib_docs']}")
    if meta.get("rows") != str(provenance["calib_rows"]):
        raise StaleProjector(f"StaleProjector: {Path(path).name} was fitted on {meta.get('rows')} calibration rows, "
                             f"the clean list (after {provenance['fitted_after']}, datasets/calibration-exclude.json) "
                             f"has {provenance['calib_rows']}: a pre-exclusion projector; refit it (jobs/paper-refit-clean.jobs)")
    got = meta.get("calib_exclude_sha256")
    if got is not None and got != provenance["calib_exclude_sha256"]:
        raise StaleProjector(f"StaleProjector: {Path(path).name} was fitted with exclusion list {got[:12]}, the cell "
                             f"declares {provenance['calib_exclude_sha256'][:12]}")


def check_derived(cell):
    """The derived-set manifests the cell's runs read must be the bytes the cell (and its FINAL record) pins."""
    for name, want in cell.get("derived", {}).items():
        path = repo() / "datasets" / "derived" / f"{name}.json"
        got = atomic.sha256_file(path) if path.is_file() else None
        if got != want:
            raise HarnessGap(f"DerivedManifestMismatch: {path} hashes to {got}, the cell pins {want} "
                             f"(scripts/build_derived.py --check)")


def preflight(cell, configs):
    for v in ("KVA_MODEL_DIR", "KVA_PROJ_DIR"):
        if not os.environ.get(v):
            raise HarnessGap(f"{v} is not set: the harness configs read models and projectors from it")
    check_derived(cell)
    pinned = {f["file"]: f.get("sha256") for f in cell.get("projector", {}).get("files", [])}
    for cfg, _ in configs:
        for f in projector_files(cfg):
            name = Path(f).name
            if not pinned.get(name):
                raise StaleProjector(f"UnpinnedProjector: cell {cell['name']} declares no sha256 for {name}: the clean "
                                     "refit's sha256 is declared first (finals/make_cells.py reads the clean fit report)")
            check_projector(os.path.expandvars(f), cell["projector"]["provenance"], pinned[name])


def run_cell(cell, cell_dir, python):
    """Preflight, then each run: write its config, run the harness once, sidecar every record it wrote."""
    configs = harness_configs(cell)
    preflight(cell, configs)
    n = 0
    for cfg, set_name in configs:
        cfg_path = cell_dir / f"harness-{cfg['name']}.json"
        atomic.write_result(cfg_path, cfg)
        cmd = [python, str(repo() / "scripts" / "evaluate.py"), str(cfg_path), set_name]
        print(f"harness: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, env=dict(os.environ, KVA_WORK=str(cell_dir / "work"))).returncode
        if rc != 0:
            raise HarnessGap(f"the reference harness failed (exit {rc}) for {cfg['name']}: see its log above; "
                             "the cell is not done and no result is trusted")
        out_dir = cell_dir / "work" / "results" / f"{set_name}-{cfg['name']}"
        records = sorted(out_dir.glob("*.json"))
        if not records:
            raise HarnessGap(f"the harness exited 0 but wrote no records in {out_dir}")
        for r in records:
            atomic.refresh_sidecar(r)
        print(f"harness: {len(records)} records sidecared in {out_dir}", flush=True)
        n += len(records)
    return n
