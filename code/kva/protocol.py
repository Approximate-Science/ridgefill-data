"""The research protocol as code (protocol.json; rules in paper/PROTOCOL.md).

Every evaluation config is checked here before a GPU is touched. A config MUST name the protocol version,
the ledger hypotheses it serves (registered before the run), and the model; every arm is a protocol row (a
protocol arm kind at one of the model's protocol cells) or the config carries an "exploratory" reason. The
returned stamp goes into every result's provenance, so tables can refuse to mix protocol versions or
exploratory arms with battery rows.

Eras (protocol v1.1.0, 2026-10-04). A config is checked under the protocol of its own major.minor: protocol.json
when that is the current minor, else the superseded protocol kept byte for byte in protocol-archive/ (the last
patch of each superseded minor, so the stamp's sha256 is the one that era's results already carry). A config with
no version, or with a version that is neither, is refused by name. Locked-set finals run only under the current
protocol. A frozen protocol ("phase": "frozen", v1.1 draft D.9) refuses every run without a FINAL- registration,
whatever era its config names (frozen_gate; the q27 and qfn drivers and scripts/gdn_memory.py call it too).

Arm kinds. arm_kind(spec) names a config arm spec's kind by its structure (KINDS). Given a protocol it is None unless
that protocol lists the kind in its "arms", so a v1.1 kind is no kind of v1.0 and v1.0 configs classify exactly as
they did under v1.0 (one exception, deliberate: SwiftKV as released, fill swiftkv_native, was filed under the v1.0
kind swiftkv; it is now its own kind swiftkv-native, which v1.0 does not have; no v1.0 record has one).
A protocol row (protocol_row) is a kind of the config's era at one of the model's cells: the split x tail grid, with
each kind's own rules (pop-native at the model's pop_split with the scored rows as its tail; swiftkv and
swiftkv-native on the baseline model only, swiftkv-native without a tail; +st on hybrids only; qblind at the blind
split only; kvexact as fill oracle on dense models and as attention-oracle + GDN-proj on hybrids). Matched controls
(+random, +tailmatch) are protocol rows only when select.match names a rows arm of the same config with the same
fill, split and tail. A battery row (battery_row) is a protocol row whose kind is not in protocol.json's
"outside_battery" (the mechanism and control arms): scripts/battery.py tabulates battery rows and lists the other
protocol rows by name, never in its tables.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from kva.paths import REPO

PROTOCOL_FILE = REPO / "protocol.json"
ARCHIVE = REPO / "protocol-archive"          # protocol-<version>.json: superseded protocols, byte for byte
BATTERY_KEYS = {"name", "split", "tail", "fill", "projector", "sets", "select", "st", "reuse_fill"}
ARM_KEYS = BATTERY_KEYS | {"tail_from", "exact_part", "fill_by_type", "rescale", "rescale_skip_last"}    # v1.1 kinds
QATTN5 = {"kind": "qattn", "budget": 0.05}
QBLIND5 = {"kind": "qattn", "budget": 0.05, "window": "prefix"}
QATTN10 = {"kind": "qattn", "budget": 0.1}
QBLIND10 = {"kind": "qattn", "budget": 0.1, "window": "prefix"}
ROWS = {"qattn5": QATTN5, "qblind5": QBLIND5, "qattn10": QATTN10, "qblind10": QBLIND10}   # kind suffix -> select
MATCHED = {"random": "random", "tail_match": "tailmatch"}        # matched controls: select kind -> kind suffix
BASE = {"identity": "noproj", "proj": "kva", "none": "skip", "swiftkv": "swiftkv"}
FILL = {v: k for k, v in BASE.items()}
SWEEP = re.compile(r"-sweep-s\d+\.safetensors$")
KVEXACT_HYBRID = {"attention": "oracle", "gdn": "proj"}
TAILLESS = ("pop-native", "swiftkv-native")                       # kinds whose tail is not a protocol tail
KINDS = ("noproj", "kva", "skip", "swiftkv", "noproj+qattn5", "kva+qattn5", "noproj+qblind5", "kva+qblind5", "kva+st",
         "noproj+qattn10", "kva+qattn10", "noproj+qblind10", "kva+qblind10", "pop-native", "swiftkv-native",
         "kva+kexact", "kva+vexact", "kvexact", "kva+ktail", "kva+vtail", "noproj+random", "kva+random",
         "noproj+tailmatch", "kva+tailmatch", "skip+tailbudget")  # every kind arm_kind can name
LOCKED_SETS = {"ruler": "ruler", "longbench": "longbench", "longppl": "long-prompts"}   # evaluate.py set -> protocol.json
CAPS = {"ruler": "ruler_max_len", "longbench": "longbench_max_len", "longppl": "longppl_max_len"}
FINAL = "FINAL-"
PHASES = ("development", "frozen")


class ProtocolError(SystemExit):
    pass


def load(path=PROTOCOL_FILE):
    raw = Path(path).read_bytes()
    p = json.loads(raw)
    p["sha256"] = hashlib.sha256(raw).hexdigest()
    return p


def minor(version):
    """'1.0' from '1.0' or '1.0.3': results compare only within one major.minor."""
    return ".".join(str(version).split(".")[:2])


def _version_key(v):
    return tuple(int(x) for x in v.split("."))


def archived(archive=ARCHIVE):
    """{major.minor: path} of the superseded protocols in protocol-archive/ (the highest patch of each minor)."""
    best = {}
    for f in Path(archive).glob("protocol-*.json"):
        v = f.name[len("protocol-"):-len(".json")]
        if re.fullmatch(r"\d+\.\d+\.\d+", v) and (minor(v) not in best or _version_key(v) > best[minor(v)][0]):
            best[minor(v)] = (_version_key(v), f)
    return {k: f for k, (_, f) in best.items()}


def era(version, current=None):
    """The protocol a config stamped `version` is checked under: the current protocol for its major.minor, else the
    archived protocol of that major.minor; None for any other version."""
    cur = current or load()
    if minor(version) == minor(cur["version"]):
        return cur
    path = archived().get(minor(version))
    return load(path) if path else None


def phase(p):
    """'development' or 'frozen' (a protocol without the key is in development); anything else is refused."""
    ph = p.get("phase", "development")
    if ph not in PHASES:
        raise ProtocolError(f"protocol {p.get('version')} phase {ph!r} is not one of {PHASES}")
    return ph


def frozen_gate(regs, p=None):
    """[error] for a run without a FINAL- registration under a frozen protocol (draft D.9), else []."""
    p = p or load()
    if phase(p) == "frozen" and not any(str(r).startswith(FINAL) for r in regs or ()):
        return [f"protocol {p['version']} is frozen: every paper-model run needs a declared final's {FINAL}... "
                f"registration (unfreezing is a protocol change)"]
    return []


# ---- arm kinds and cells
def _only(spec, keys):
    return not set(spec) - set(keys) - {"name", "sets"}


def _diagnostic(spec, base):
    """kva + a K/V splice (exact_part k | v) or a tail-measured rescale (ktail | vtail), nothing else on the arm."""
    if base != "kva" or not _only(spec, ("split", "tail", "fill", "projector", "exact_part", "rescale",
                                         "rescale_skip_last")):
        return None
    part, rescale, skip = spec.get("exact_part"), spec.get("rescale"), spec.get("rescale_skip_last")
    if part in ("k", "v") and rescale is None and skip is None:
        return f"kva+{part}exact"
    if part is None and rescale in ("ktail", "vtail") and (skip is None or (type(skip) is int and skip >= 0)):
        return f"kva+{rescale}"
    return None


def _tail_budget(sel):
    """select start_recent with start 0: no exact rows, round(budget x N) more exact tail rows."""
    b = sel.get("budget")
    return (set(sel) == {"kind", "start", "budget"} and sel["kind"] == "start_recent" and type(sel["start"]) is int
            and sel["start"] == 0 and type(b) in (int, float) and 0 < b < 1)


def _kind(spec):
    if set(spec) - ARM_KEYS:
        return None
    fill, sel = spec.get("fill", "exact"), spec.get("select")
    if fill == "swiftkv_native":
        return "swiftkv-native" if _only(spec, ("split", "fill")) else None
    if "tail_from" in spec:
        ok = spec["tail_from"] == "continuation" and fill == "identity" and _only(spec, ("split", "fill", "tail_from"))
        return "pop-native" if ok else None
    if fill == "oracle":
        return "kvexact" if _only(spec, ("split", "tail", "fill")) else None
    if fill == "by_type":
        ok = (spec.get("fill_by_type") == KVEXACT_HYBRID and SWEEP.search(str(spec.get("projector", "")))
              and _only(spec, ("split", "tail", "fill", "fill_by_type", "projector")))
        return "kvexact" if ok else None
    base = BASE.get(fill)
    if base is None or "fill_by_type" in spec:
        return None
    if base == "kva" and not SWEEP.search(str(spec.get("projector", ""))):
        return None
    if {"exact_part", "rescale", "rescale_skip_last"} & set(spec):
        return _diagnostic(spec, base)
    if spec.get("st"):
        return "kva+st" if base == "kva" and not sel else None
    if not sel:
        return base
    if base == "skip":
        return "skip+tailbudget" if _tail_budget(sel) else None
    if base not in ("kva", "noproj"):
        return None
    for suffix, rows in ROWS.items():
        if sel == rows:
            return f"{base}+{suffix}"
    if set(sel) == {"kind", "match"} and sel["kind"] in MATCHED and isinstance(sel["match"], str):
        return f"{base}+{MATCHED[sel['kind']]}"
    return None


def arm_kind(spec, p=None):
    """The protocol arm kind of a config arm spec, or None if it is not one (extra fields, a non-standard
    projector, another selector or budget). With a protocol p: None also for a kind p does not list in "arms"."""
    kind = _kind(spec)
    return None if kind is None or (p is not None and kind not in p["arms"]) else kind


def template(kind, model, split, tail, p):
    """A config arm spec of this kind at (split, tail) (arm_kind's inverse): what the battery asks protocol_row about."""
    name = f"{kind}-s{split}" + ("" if tail is None else f"-t{tail}")
    proj = f"{model}-sweep-s{split}.safetensors"
    if kind == "pop-native":
        return dict(name=name, split=split, fill="identity", tail_from="continuation")
    if kind == "swiftkv-native":
        return dict(name=name, split=split, fill="swiftkv_native")
    if kind == "kvexact":
        if p["models"][model]["kind"] == "dense":
            return dict(name=name, split=split, tail=tail, fill="oracle")
        return dict(name=name, split=split, tail=tail, fill="by_type", fill_by_type=dict(KVEXACT_HYBRID), projector=proj)
    base, _, extra = kind.partition("+")
    spec = dict(name=name, split=split, tail=tail, fill=FILL[base])
    if base == "kva":
        spec["projector"] = proj
    if extra in ROWS:
        spec["select"] = dict(ROWS[extra])
    elif extra in ("random", "tailmatch"):
        spec["select"] = dict(kind="random" if extra == "random" else "tail_match", match=f"{base}+qattn10")
    elif extra == "st":
        spec["st"] = "st.pt"
    elif extra in ("kexact", "vexact"):
        spec["exact_part"] = extra[0]
    elif extra in ("ktail", "vtail"):
        spec["rescale"] = extra
    elif extra == "tailbudget":
        spec["select"] = dict(kind="start_recent", start=0, budget=0.1)
    return spec


def protocol_row(spec, model, p):
    """True if this arm is a protocol row for this model under p: a kind of p at one of the model's cells."""
    kind = arm_kind(spec, p)
    m = p["models"].get(model)
    if kind is None or m is None:
        return False
    split = spec.get("split")
    if kind == "swiftkv":
        return bool(m.get("baseline_only"))
    if kind == "swiftkv-native":
        return bool(m.get("baseline_only")) and split in m["splits"]
    if m.get("baseline_only"):
        return False
    if kind == "pop-native":
        return split == m.get("pop_split")
    if split not in m["splits"]:
        return False
    if kind == "kva+st" and m["kind"] == "dense":
        return False
    if "+qblind" in kind and split != p["blind_splits"].get(model):
        return False
    if kind == "kvexact" and (spec.get("fill") == "oracle") != (m["kind"] == "dense"):
        return False
    return spec.get("tail") in p["tails"]


def battery_row(spec, model, p):
    """True if this arm is a battery row for this model: a protocol row whose kind the battery tabulates."""
    return protocol_row(spec, model, p) and arm_kind(spec, p) not in p.get("outside_battery", ())


def battery_cells(model, p):
    """(split, tail, kind) of every battery row of a model, in table order: the split x tail grid with the kinds in
    protocol.json order, then pop-native at its split and swiftkv-native at each split (tail None)."""
    m = p["models"][model]
    kinds = [k for k in p["arms"] if k != "exact"]
    grid = [(s, t, k) for s in m["splits"] for t in p["tails"] for k in kinds if k not in TAILLESS]
    own = [(m.get("pop_split"), None, "pop-native")] + [(s, None, "swiftkv-native") for s in m["splits"]]
    return [(s, t, k) for s, t, k in grid + own
            if k in p["arms"] and s is not None and battery_row(template(k, model, s, t, p), model, p)]


def _row_in(spec, by_name, model, p):
    """protocol_row, plus a matched control's target: a rows arm of this config with the same fill, split, tail."""
    if not protocol_row(spec, model, p):
        return False
    kind = arm_kind(spec, p)
    if not kind.endswith(("+random", "+tailmatch")):
        return True
    base, target = kind.split("+")[0], by_name.get(spec["select"]["match"])
    return (target is not None and arm_kind(target, p) in {f"{base}+{r}" for r in ROWS}
            and (target.get("split"), target.get("tail")) == (spec.get("split"), spec.get("tail")))


# ---- configs
def registered(ledger_path):
    ids = set()
    for line in Path(ledger_path).read_text().splitlines():
        rec = json.loads(line)
        if rec.get("type") == "hypothesis":
            ids.add(rec["id"])
    return ids


def validate(cfg, p=None, ledger_path=None, env=os.environ, set_name=None):
    """Check a run config against the protocol of its era for one evaluation set; return the provenance stamp.
    Raises ProtocolError (exit) with every problem listed. The locked sets run only for declared finals
    (PROTOCOL.md section 4 step 5): a registration ID starting with FINAL-, under the current protocol."""
    cur = p or load()
    head = "protocol check failed (paper/PROTOCOL.md):\n  - "
    version = cfg.get("protocol")
    if not str(version or "").strip():
        raise ProtocolError(head + f"config has no 'protocol' version: name the protocol it runs under (current "
                                   f"{cur['version']}; archived {sorted(archived())})")
    ledger_path = ledger_path or REPO / cur["ledger"]
    errors = []
    p = era(version, cur)
    if p is None:
        errors.append(f"config protocol {version!r} is not the current protocol {cur['version']} nor an archived "
                      f"one {sorted(archived())}")
        p = cur                                     # list every other problem against the current protocol
    model_path = str(cfg.get("model", ""))
    model = Path(model_path).name
    if any(b in model_path.lower() for b in p["banned_model_substrings"]):
        errors.append(f"banned model {model_path} (no fine-tuned checkpoint of a banned family: "
                      "protocol.json banned_model_substrings)")
    elif model not in p["models"]:
        errors.append(f"model {model!r} is not in protocol.json models (protocol {p['version']})")
    regs = cfg.get("registration") or []
    if not regs:
        errors.append("no 'registration': list the ledger hypothesis IDs this run serves (register them first)")
    else:
        missing = sorted(set(regs) - registered(ledger_path))
        if missing:
            errors.append(f"registration IDs not in the ledger yet: {missing}")
    errors += frozen_gate(regs, cur)
    if set_name in LOCKED_SETS:
        if LOCKED_SETS[set_name] not in p["sets"]["locked"]:
            errors.append(f"set {set_name!r} is not a locked set of protocol.json")
        if not any(str(r).startswith(FINAL) for r in regs):
            errors.append(f"set {set_name!r} is the locked test set: finals only, so list the declared final's "
                          f"{FINAL}... ledger ID in 'registration' (PROTOCOL.md section 4 step 5)")
        if p is not cur:
            errors.append(f"set {set_name!r} is the locked test set: finals run only under the current protocol "
                          f"{cur['version']}, not {version!r}")
        if "ruler_max_new" in cfg and (set_name != "ruler" or cfg["ruler_max_new"] != p.get("ruler_max_new", {}).get("tasks")):
            errors.append(f"ruler_max_new {cfg['ruler_max_new']!r}: only the ruler set takes a generation cap, and only "
                          f"protocol.json's ruler_max_new.tasks {p.get('ruler_max_new', {}).get('tasks')!r} (v1.1.2)")
        cap = p["models"].get(model, {}).get("max_len")
        got = cfg.get(CAPS[set_name])
        if cap and not (type(got) is int and 0 < got <= cap):
            errors.append(f"{model} runs at <= {cap} tokens (protocol.json max_len): set {CAPS[set_name]} to at "
                          f"most {cap}, got {got!r}")
    if env.get("KVA_DETERMINISTIC", "1") != "1":
        errors.append("KVA_DETERMINISTIC must be 1 for every protocol run")
    # drop_bos (SINK1) changes the prompts themselves, so none of its arms is a protocol row
    arms = cfg.get("arms", [])
    by_name = {a.get("name"): a for a in arms}
    off = [a["name"] for a in arms
           if model not in p["models"] or not _row_in(a, by_name, model, p) or cfg.get("drop_bos")]
    if off and not str(cfg.get("exploratory", "")).strip():
        errors.append(f"arms outside the protocol (neither battery rows nor protocol rows outside the battery) {off}: "
                      f"add an 'exploratory' reason or fix them")
    if errors:
        raise ProtocolError(head + "\n  - ".join(errors))
    # a config that states an exploratory reason is exploratory as a whole, even if every arm is a protocol row
    return dict(version=p["version"], sha256=p["sha256"], registration=list(regs),
                battery=not off and not str(cfg.get("exploratory", "")).strip(), exploratory_arms=off, exploratory=cfg.get("exploratory") or None)
