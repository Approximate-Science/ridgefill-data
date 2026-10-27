"""finals/convert.py: a finished cell's reference-harness results -> the finals sample records finals/tables.py reads.

Real harness output, not fakes: make_tiny.py's tiny Llama (real tokenizer) runs scripts/evaluate.py through the kva
backend (finals/lib/kvabackend.run_cell, the box's own path) for four cells laid out as finals/make_cells.py lays them
out: RULER (primary arms, the T1.5 controls, the T2.1 RULER splices, question-first), NLL (the split curve,
pop-native, the splices, key tokens), decode, second question. The locked set is SYNTHETIC (made-up needles, keys and
prose in the locked layout; nothing here is or derives from a locked text); grade (the dev set) gives the dev-tier
record. evaluate.py validates a config's registration against the repo's own ledger, so the test cells borrow the
four FINAL- ids it holds; their pins live in the test tree's ledger only.
Checks:
- round trip: the converted cells join the mock tree (finals/mock/tables_tree.py, the mock for the rest) ->
  verify_finals -> tables.py builds every table -> check_numbers.py passes, and the real units are among the units the
  tables read (positive control);
- oracles simpler than the converter on the same records: needle_filled by decoding the filled prefix, the RULER score
  by substring match, novel and key-token NLL by hand, turn correctness by a 7-digit search;
- refusals by name: a tampered harness file; a harness file changed and re-sidecared after conversion (stale stamp, in
  verify_finals and tables); a cell with harness results and no conversion; a dev-tier (grade) record in tables; a
  stored score the official scorer does not reproduce; a record without one of its config's arms;
- idempotent: a second conversion writes the same bytes."""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kva import derived as D
from kva import locked as L
from kva import protocol as PR

ROOT = Path(__file__).resolve().parents[1]
FINALS = ROOT / "finals"
sys.path.insert(0, str(FINALS / "lib"))
import atomic                                                # noqa: E402
import kvabackend as KB                                      # noqa: E402
import registration as REG                                   # noqa: E402

MODEL = "llama-3.1-8b-instruct"
TOKDIR = ROOT / "data" / "models" / MODEL
NIAH = list(D.QFIRST_TASKS)
HEAD1 = ("A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you "
         "about the number afterwards.")
HEADN = ("Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you "
         "about the numbers afterwards.")
KEYS = ["zorbly-quill", "plinth-vortex", "marbled snorkel", "tepid-gazebo"]
FILLER = 150                                                 # ~2K tokens: N - 1024 (the RULER cell's tail) leaves a
ROWS = 1                                                     # filled region; asked = needle (task + row) % 4 varies
PROJ_META = dict(rows=1000001, docs=98, lam=0.03)            # the mock tree's Llama projector header (one declared value)
REAL = {"ruler": "FINAL-QWEN3-8B-LONGBENCH", "nll": "FINAL-QWEN3-8B-NLL", "decode": "FINAL-QWEN3-8B-DECODE",
        "secondq": "FINAL-QWEN3-8B-SECONDQ"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CV = _load("finals_convert", FINALS / "convert.py")
MOCK = _load("finals_mock_tables_tree", FINALS / "mock" / "tables_tree.py")


# ---------------------------------------------------------------------------------------------- the synthetic locked set
def needle(key, value):
    return f"One of the special magic numbers for {key} is: {value}."


def niah_row(index, asked, head, filler):
    values = [str(1000000 + 1111111 * (index % 7) + 37 * j) for j in range(4)]
    sents = [f"Filler line {j} says the grey owl {index} counts pebbles." for j in range(filler)]
    for j, (k, v) in enumerate(zip(KEYS, values)):
        sents.insert((j + 1) * filler // 5 + j, needle(k, v))
    key = KEYS[asked]
    return dict(index=index, input=f"{head}\n{' '.join(sents)}\n{D.QUESTION.format(key=key)}", outputs=[values[asked]],
                answer_prefix=D.PREFIX.format(key=key), length=0, length_w_model_temp=0, token_position_answer=0)


def prose(k):
    words = ["delta", "harbor", "copper", "lantern", "meadow", "quarry", "saddle", "tundra", "violet", "wicker"]
    sents = []
    for j in range(350):                                     # ~6K tokens: N > the key-token tail 4096 + the tail budget
        w = [words[(j * 3 + i * 7 + k) % len(words)] for i in range(6)]
        sents.append(f"The {w[0]} {w[1]} kept {j % 13} {w[2]} near the {w[3]} {w[4]} by the {w[5]}.")
    return " ".join(sents)


def build_locked(root):
    """ruler/<NIAH task>_<L>.jsonl (8K: long rows the cells run; longer: short rows the derived builders read only),
    long-prompts/prose_8192_<k>.txt with SPLITS.tsv; MANIFEST.tsv and FROZEN.json as the locked set has them."""
    rows = ["path\tsha256\tbytes\tdomain/task\tsource_url\tlicence\n"]

    def add(rel, body, domain):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body)
        rows.append(f"{rel}\t{hashlib.sha256(body.encode()).hexdigest()}\t{len(body)}\t{domain}\tsynthetic\tnone\n")
    for t, task in enumerate(NIAH):
        head = HEADN if task in ("niah_multivalue", "niah_multiquery") else HEAD1
        for L_ in L.RULER_LENGTHS:
            n, filler = (ROWS, FILLER) if L_ == 8192 else (1, 40)
            body = "".join(json.dumps(niah_row(10 * t + i, asked=(t + i) % 4, head=head, filler=filler)) + "\n"
                           for i in range(n))
            add(f"ruler/{task}_{L_}.jsonl", body, f"ruler/{task}")
    for task in L.LONGBENCH_PROMPT:                          # one short made-up row per LongBench task
        row = dict(dataset=task, _id=f"syn-{task}", length=120, answers=["violet harbor"],
                   all_classes=["Entity", "Location"] if task == "trec" else None,
                   context=" ".join(f"Line {j}: the violet harbor keeps {j} lanterns." for j in range(40)),
                   input="Where are the lanterns kept?")
        add(f"longbench/{task}.jsonl", json.dumps(row) + "\n", f"longbench/{task}")
    splits = ["path\tsplit_char\n"]
    for k in range(2):
        text = prose(k)
        add(f"long-prompts/prose_8192_{k}.txt", text, "prose/synthetic")
        splits.append(f"long-prompts/prose_8192_{k}.txt\t{len(text) - 1400}\n")
    (root / "long-prompts" / "SPLITS.tsv").write_text("".join(splits))
    (root / "MANIFEST.tsv").write_text("".join(rows))
    (root / "FROZEN.json").write_text(json.dumps(
        dict(manifest_sha256=hashlib.sha256((root / "MANIFEST.tsv").read_bytes()).hexdigest())))


# ---------------------------------------------------------------------------------------------- cells as make_cells lays them out
def run(id_, set_, split, role, arms, **kw):
    return dict(id=id_, set=set_, split=split, role=role, arms=list(arms), **kw)


PRIMARY = ["exact", "nofill", "identity", "pop-native", "ridgefill", "identity+rows", "ridgefill+rows"]
CONTROLS = ["identity+random", "ridgefill+random", "identity+tailmatch", "ridgefill+tailmatch", "nofill+tailbudget"]
CURVE = ["exact", "nofill", "identity", "ridgefill"]
SPLICES = ["ridgefill+kexact", "ridgefill+vexact", "kvexact", "ridgefill+ktail", "ridgefill+vtail"]


def cells_of_the_test(pins, frozen):
    files = lambda layers: [dict(layer=s, file=f"{MODEL}-sweep-s{s}.safetensors", sha256=pins[s]) for s in layers]  # noqa: E731
    base = dict(model=dict(id="meta-llama/Llama-3.1-8B-Instruct", name=MODEL, commit="0" * 40), split="3/8", tail=4096,
                caps=dict(ruler_max_len=8192, longbench_max_len=31500), max_len=8192,
                margins=dict(ppl_ratio_upper_pct=1.0, ruler_points_lower=-2.0), dataset_manifest_sha256=frozen,
                deterministic=True, code_tag="protocol-v1.1", note="TEST cell (tests/test_finals_convert.py)")
    prov = dict(calib_rows=PROJ_META["rows"], calib_docs=PROJ_META["docs"], calib_exclude_sha256="0" * 64,
                fitted_after="cba2d91")
    sel = dict(tasks=NIAH, lengths=[8192])
    return {                                                 # RULER and second question at tail 1024 (short prompts)
        "ruler": dict(base, name="real-llama-ruler", kind="ruler", registration=REAL["ruler"], tail=1024,
                      runs=[run("ruler-niah", "ruler", "3/8", "primary", PRIMARY + CONTROLS, **sel),
                            run("ruler-niah-splices", "ruler", "3/8", "mechanism", ["exact", "ridgefill+kexact",
                                                                                    "ridgefill+vexact"], **sel),
                            run("ruler-qfirst", "ruler", "3/8", "primary", ["exact", "identity+rows", "ridgefill+rows"],
                                variant="qfirst", **sel),
                            run("ruler-swiftkv", "ruler", "1/2", "swiftkv", ["exact", "swiftkv-native", "swiftkv"],
                                model="swiftkv", **sel)],
                      swiftkv_model=dict(id="Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
                                         name="llama-3.1-swiftkv-8b-instruct", commit="1" * 40),
                      projector=dict(provenance=prov, files=files([12])),
                      tailbudget=dict(b=0.186, basis="test")),
        "nll": dict(base, name="real-llama-nll", kind="nll", registration=REAL["nll"],
                    runs=[run("nll", "longppl", "3/8", "primary", CURVE + ["pop-native", "identity+rows", "ridgefill+rows"]
                              + CONTROLS + SPLICES)]
                    + [run(f"nll-{f.replace('/', '-')}", "longppl", f, "curve", CURVE) for f in ("1/4", "1/2", "5/8")]
                    + [run("nll-swiftkv", "longppl", "1/2", "swiftkv", ["exact", "swiftkv-native", "swiftkv"],
                           model="swiftkv")],
                    projector=dict(provenance=prov, files=files([8, 12, 16, 20])), tailbudget=dict(b=0.186, basis="test"),
                    swiftkv_model=dict(id="Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
                                       name="llama-3.1-swiftkv-8b-instruct", commit="1" * 40),
                    keytoken=dict(tail=4096, threshold=2.0)),
        "decode": dict(base, name="real-llama-decode", kind="decode", registration=REAL["decode"],
                       runs=[run("decode", "decode", "3/8", "primary", CURVE)],
                       projector=dict(provenance=prov, files=files([12])),
                       decode={"from": "longppl", "steps": 6, "only": ["lp-prose_8192_0", "lp-prose_8192_1"]}),
        "secondq": dict(base, name="real-llama-secondq", kind="secondq", registration=REAL["secondq"], tail=1024,
                        runs=[run("secondq", "multiturn", "3/8", "primary",
                                  ["exact", "ridgefill", "identity+rows", "ridgefill+rows"])],
                        projector=dict(provenance=prov, files=files([12])), secondq={"from": "ruler", "turns": 2},
                        skipped=[f"ruler/niah_multikey_1_{n}.jsonl" for n in (16384, 32768, 65536)]),
    }


def finish_cell(cell, cell_dir, ledger):
    """What runner.py does around the harness: cell.json, host.json, canary.json; then the FINAL- record (pins) in the
    test ledger, the conversion, and both push receipts with .done."""
    atomic.write_result(cell_dir / "host.json", {"fingerprint": "test", "host": {"host": "test", "gpu": "cpu"}})
    atomic.write_result(cell_dir / "canary.json", {"arms": {"exact": {"logits_sha256": "0" * 64}}})
    lines = ledger.read_text().splitlines()
    lines.append(json.dumps({"type": "hypothesis", "id": cell["registration"], "seq": len(lines) + 1,
                             "claim": "TEST | " + REG.pins_text(REG.pins(cell)), "predict": "test", "kill": "test"}))
    ledger.write_text("\n".join(lines) + "\n")


def mark_done(cell_dir):
    tree = atomic.treehash(cell_dir)
    atomic.write_bytes(cell_dir / ".pushed", json.dumps({"ssh": {"tree": tree}, "hf": {"tree": tree}}).encode())
    atomic.write_bytes(cell_dir / ".done", tree.encode())


def _env(tmp):
    return dict(KVA_REPO=str(ROOT), KVA_MODEL_DIR=str(tmp / "models"), KVA_PROJ_DIR=str(tmp / "proj"),
                KVA_LOCKED_DIR=str(tmp / "locked"), KVA_DERIVED_DIR=str(tmp / "derived"), KVA_CPU_DRYRUN="1",
                KVA_BULK=str(tmp / "bulk"), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4")


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    """The mock tree plus four converted cells of real harness output (module doc)."""
    if not (TOKDIR / "tokenizer.json").exists():
        pytest.skip("needs the local Llama-3.1 tokenizer (scripts/make_tiny.py copies it)")
    tmp = tmp_path_factory.mktemp("convert")
    with pytest.MonkeyPatch.context() as mp:
        for k, v in _env(tmp).items():
            mp.setenv(k, v)
        mp.delenv("KVA_DETERMINISTIC", raising=False)
        build_locked(tmp / "locked")
        mp.setattr(L, "LOCKED", tmp / "locked")
        mp.setattr(D, "DIR", tmp / "derived")
        D.DIR.mkdir()
        for which in D.SETS:
            D.manifest_path(which).write_text(D.dumps(D.build_manifest(which)))
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tiny.py"), str(tmp / "models")], check=True,
                       capture_output=True)
        from kva.kva import Projector
        g = torch.Generator().manual_seed(0)
        (tmp / "proj").mkdir()
        pins = {}
        for s in (8, 12, 16, 20):
            f = tmp / "proj" / f"{MODEL}-sweep-s{s}.safetensors"
            Projector(s, {i: torch.eye(64) + 0.02 * torch.randn(64, 64, generator=g) for i in range(s, 32)},
                      {i: 0.02 * torch.randn(64, generator=g) for i in range(s, 32)}).save(f, dict(PROJ_META, kind="test"))
            pins[s] = atomic.sha256_file(f)
        MOCK.build(tmp / "tree")
        shutil.copy(tmp / "proj" / f"{MODEL}-sweep-s12.safetensors", tmp / "tree" / "proj")   # tables read its header
        ledger = tmp / "tree" / "ledger" / "paper" / "ledger.jsonl"
        cells = cells_of_the_test(pins, json.loads((tmp / "locked" / "FROZEN.json").read_text())["manifest_sha256"])
        dirs = {}
        for kind, cell in cells.items():
            d = dirs[kind] = tmp / "tree" / "finals-results" / cell["name"]
            atomic.write_result(d / "cell.json", cell)
            KB.run_cell(cell, d, sys.executable)
            finish_cell(cell, d, ledger)
            CV.convert_cell(d, ledger)
            mark_done(d)
    shutil.copytree(ROOT / "paper" / "tex", tmp / "tree" / "paper" / "tex",
                    ignore=shutil.ignore_patterns("main.pdf", "__pycache__"))
    shutil.copy(ROOT / "protocol.json", tmp / "tree" / "protocol.json")
    return dict(tmp=tmp, tree=tmp / "tree", ledger=ledger, cells=cells, dirs=dirs)


def samples(d, set_):
    return {r["sample"]: r for r in (json.loads(f.read_text()) for f in sorted((d / "samples" / set_).glob("*.json")))}


def harness(d, name):
    rdir = d / "work" / "results" / name
    return {(f"{r['source']}#{r['row']:02d}" if "row" in r else r["source"]): r
            for r in (json.loads(f.read_text()) for f in sorted(rdir.glob("*.json")))}


def run_tables(tree):
    return subprocess.run([sys.executable, str(FINALS / "tables.py"), str(tree / "finals-results"), "--repo", str(tree),
                           "--out", str(tree / "paper" / "tex"), "--proj-dir", str(tree / "proj")],
                          capture_output=True, text=True)


# ---------------------------------------------------------------------------------------------- the round trip
def test_round_trip_real_harness_output_builds_the_tables(real):
    tree = real["tree"]
    v = subprocess.run([sys.executable, str(FINALS / "verify_finals.py"), str(tree / "finals-results"), "--ledger",
                        str(real["ledger"]), "--require-done"], capture_output=True, text=True)
    assert v.returncode == 0, v.stderr
    assert all(c["name"] in v.stdout for c in real["cells"].values())
    r = run_tables(tree)
    assert r.returncode == 0, r.stderr[-4000:]
    tex = tree / "paper" / "tex"
    c = subprocess.run([sys.executable, str(tex / "check_numbers.py"), "--root", str(tex), "--stray-only",
                        "--protocol", str(tree / "protocol.json")], capture_output=True, text=True)
    assert c.returncode == 0 and "stray     pass: 0 typed number(s)" in c.stdout, c.stdout + c.stderr
    # positive control: the tables read the real units, every set the four cells ran
    sys.path.insert(0, str(FINALS))
    import tables as T
    t = T.Tree(tree / "finals-results", real["ledger"], tree / "protocol.json").load()
    for kind, set_ in (("ruler", "ruler"), ("ruler", "ruler-qfirst"), ("nll", "longppl"), ("decode", "decode"),
                       ("secondq", "multiturn")):
        recs = samples(real["dirs"][kind], set_)
        got = {u for u, r in recs.items() if "skipped" not in r}
        held = set(t.data[("battery", MODEL, set_)])
        assert got and got <= held, (set_, sorted(got - held))
        assert {u for u, r in recs.items() if "skipped" in r} <= set(t.skipped.get(("battery", MODEL, set_), {}))
    rows = t.data[("battery", MODEL, "ruler")]
    real_rows = [u for u in samples(real["dirs"]["ruler"], "ruler")]
    assert all({"kva+kexact", "kva+vexact", "kva+random", "skip+tailbudget"} <= set(rows[u]["arms"]) for u in real_rows)


def test_converted_records_carry_kinds_splits_tier_and_protocol(real):
    p = PR.load()
    rul = samples(real["dirs"]["ruler"], "ruler")
    assert len(rul) == len(NIAH) * ROWS
    r = next(iter(rul.values()))
    assert set(r["arms"]) == {"exact", "skip", "noproj", "pop-native", "kva", "noproj+qattn10", "kva+qattn10",
                              "noproj+random", "kva+random", "noproj+tailmatch", "kva+tailmatch", "skip+tailbudget",
                              "kva+kexact", "kva+vexact", "swiftkv", "swiftkv-native"}
    assert r["arms"]["pop-native"].keys() == {"pop"} and r["arms"]["kva"].keys() == {"3/8"}
    assert r["arms"]["exact"].keys() == {"-"} and r["arms"]["swiftkv"].keys() == r["arms"]["swiftkv-native"].keys() == {"1/2"}
    assert (r["tier"], PR.minor(r["protocol"])) == ("locked-final", PR.minor(p["version"]))
    assert r["configs"] == ["llama-3.1-8b-instruct-ruler-niah", "llama-3.1-8b-instruct-ruler-swiftkv"]
    assert r["cell"] == "real-llama-ruler" and set(r["exact_swiftkv"]) == {"score"}   # SwiftKV's own exact, apart
    q = samples(real["dirs"]["ruler"], "ruler-qfirst")
    assert set(q) == set(rul)                                # one RULER row keeps its id under both layouts
    assert all(set(x["arms"]) == {"exact", "noproj+qattn10", "kva+qattn10"} for x in q.values())
    nll = samples(real["dirs"]["nll"], "longppl")
    d = next(iter(nll.values()))
    assert d["arms"]["kva"].keys() == {"1/4", "3/8", "1/2", "5/8"} and d["arms"]["kva+ktail"].keys() == {"3/8"}
    assert d["domain"] == "prose" and d["doc"].startswith("prose_8192_")
    assert {"noproj+qattn10", "kva+qattn10", "kva+random", "kva+tailmatch", "skip+tailbudget"} <= set(d["arms"])
    assert d["arms"]["swiftkv"].keys() == d["arms"]["swiftkv-native"].keys() == {"1/2"}            # R6 on NLL
    assert d["exact_swiftkv"] == d["arms"]["exact"]["-"]                  # SK1: the checkpoint's exact is Llama's
    assert d["configs"] == ["llama-3.1-8b-instruct-nll", "llama-3.1-8b-instruct-nll-swiftkv"]
    mt = samples(real["dirs"]["secondq"], "multiturn")
    skipped = [x for x in mt.values() if "skipped" in x]
    assert sorted(x["sample"] for x in skipped) == real["cells"]["secondq"]["skipped"]
    assert all(set(x["arms"]) == {"exact", "kva", "noproj+qattn10", "kva+qattn10"} for x in mt.values() if "arms" in x)


def test_needle_filled_against_a_decoded_prefix_oracle(real):
    """Oracle without offsets: a row is filled iff its scored needle's sentence is inside the decoded first N - T ids."""
    from kva import data
    tok = data.tokenizer(TOKDIR)
    fmt = PR.load()["models"][MODEL]["niah_format"]
    got = samples(real["dirs"]["ruler"], "ruler")
    seen = set()
    for task in NIAH:
        for i, line in enumerate((real["tmp"] / "locked" / "ruler" / f"{task}_8192.jsonl").read_text().splitlines()):
            row = json.loads(line)
            ids = L.wrap(tok, fmt, row["input"], row["answer_prefix"])
            prefix = tok.decode(ids[:len(ids) - real["cells"]["ruler"]["tail"]])
            key = KEYS[[needle(k, v) in row["input"] for k, v in zip(KEYS, [row["outputs"][0]] * 4)].index(True)]
            want = needle(key, row["outputs"][0]) in prefix
            assert got[f"ruler/{task}_8192.jsonl#{i:02d}"]["needle_filled"] is want, (task, i)
            seen.add(want)
    assert seen == {True, False}                             # both outcomes occur: the oracle can tell them apart


def test_scores_and_nll_against_hand_oracles(real):
    h = harness(real["dirs"]["ruler"], f"ruler-{MODEL}-ruler-niah")
    rul = samples(real["dirs"]["ruler"], "ruler")
    for sid, rec in h.items():
        for name, run_ in rec["runs"].items():
            pred = run_["pred"].lower()
            want = sum(ref.lower() in pred for ref in rec["refs"]) / len(rec["refs"])
            kind = "exact" if name == "exact" else PR.arm_kind(next(
                a for a in json.loads(next(real["dirs"]["ruler"].glob("harness-*niah.json")).read_text())["arms"]
                if a["name"] == name), PR.load())
            split = next(iter(rul[sid]["arms"][kind]))
            assert rul[sid]["arms"][kind][split]["score"] == want
    lp = harness(real["dirs"]["nll"], f"longppl-{MODEL}-nll")
    nll = samples(real["dirs"]["nll"], "longppl")
    for sid, rec in lp.items():
        ex, tail = rec["runs"]["exact"]["logprobs"], rec["tail_only"]["logprobs"]
        key = [f - t > 2.0 for f, t in zip(ex, tail)]
        for name, run_ in rec["runs"].items():
            x = run_["logprobs"]
            nov = [-a for a, m in zip(x, rec["novel"]) if m]
            kk = [-a for a, m in zip(x, key) if m]
            out = [v for by in nll[sid]["arms"].values() for v in by.values()
                   if v["all_nll"] == pytest.approx(-sum(x) / len(x), abs=0)]
            assert out and out[0]["novel_nll"] == pytest.approx(sum(nov) / len(nov), abs=1e-12)
            assert out[0]["n_key"] == sum(key)
            assert out[0]["key_nll"] == (pytest.approx(sum(kk) / len(kk), abs=1e-12) if kk else None)
    mt = harness(real["dirs"]["secondq"], f"multiturn-{MODEL}-secondq")
    got = samples(real["dirs"]["secondq"], "multiturn")
    for sid, rec in mt.items():
        for run_ in rec["runs"].values():
            for t in run_["turns"]:
                text = re.sub(r"<think>.*?(</think>|$)", "", t["text"], flags=re.S)
                m = re.search(r"(?<!\d)\d{7}(?!\d)", text)
                assert t["correct"] == bool(m and m.group(0) == t["value"])
        assert all(set(v["-" if k == "exact" else "3/8"]) == {"turn1", "turn2"} for k, v in got[sid]["arms"].items())
    dec = samples(real["dirs"]["decode"], "decode")
    for rec in harness(real["dirs"]["decode"], f"decode-{MODEL}-decode").values():
        for name, run_ in rec["runs"].items():
            if name != "exact":
                kind = "skip" if name.startswith("nofill") else "noproj" if name.startswith("identity") else "kva"
                assert dec[rec["source"]]["arms"][kind]["3/8"]["top1_agree"] == run_["forced_top1_agree"]


def test_conversion_is_idempotent(real):
    d = real["dirs"]["nll"]
    before = {f.relative_to(d).as_posix(): f.read_bytes() for f in sorted((d / "samples").rglob("*")) if f.is_file()}
    CV.convert_cell(d, real["ledger"])
    after = {f.relative_to(d).as_posix(): f.read_bytes() for f in sorted((d / "samples").rglob("*")) if f.is_file()}
    assert len(after) > 2
    assert after == before


# ---------------------------------------------------------------------------------------------- refusals
def _copy_cell(real, kind, tmp_path):
    d = tmp_path / real["dirs"][kind].name
    shutil.copytree(real["dirs"][kind], d)
    return d


def test_a_tampered_harness_file_is_refused_by_name(real, tmp_path):
    d = _copy_cell(real, "ruler", tmp_path)
    f = sorted((d / "work" / "results").glob("ruler-*niah/*.json"))[0]
    f.write_text(f.read_text().replace('"score": 0.0', '"score": 1.0', 1))
    with pytest.raises(CV.ConvertRefused) as e:
        CV.convert_cell(d, real["ledger"])
    assert f"{f.relative_to(d)} does not match its .sha256 sidecar" in str(e.value)


def test_a_harness_file_changed_after_conversion_makes_the_samples_stale(real, tmp_path):
    vf = CV.VF
    d = _copy_cell(real, "decode", tmp_path)
    assert vf.stamp_problems(d) == []
    f = sorted((d / "work" / "results").glob("decode-*/*.json"))[0]
    rec = json.loads(f.read_text())
    rec["n"] += 1
    f.write_text(json.dumps(rec))
    atomic.refresh_sidecar(f)                               # a consistent edit: only the stamp can tell
    assert "stale" in vf.stamp_problems(d)[0]
    problems, _ = vf.check_cell(d, real["ledger"])
    assert any("stale" in p for p in problems)
    CV.convert_cell(d, real["ledger"])                      # a resumed cell is converted again: current once more
    assert vf.stamp_problems(d) == [] and vf.check_cell(d, real["ledger"])[0] == []


def test_tables_refuse_an_unconverted_cell_and_a_dev_tier_record(real, tmp_path):
    tree = tmp_path / "tree"
    shutil.copytree(real["tree"], tree, symlinks=True)
    d = tree / "finals-results" / "real-llama-decode"
    shutil.rmtree(d / "samples")
    mark_done(d)
    r = run_tables(tree)
    assert r.returncode == 1 and "UNCONVERTED HARNESS RESULTS" in r.stderr and "no conversion stamp" in r.stderr
    CV.convert_cell(d, real["ledger"].parent.parent.parent / "tree" / "ledger" / "paper" / "ledger.jsonl"
                    if False else tree / "ledger" / "paper" / "ledger.jsonl")
    f = next((d / "samples" / "decode").glob("*.json"))
    rec = json.loads(f.read_text())
    rec["tier"] = "dev"
    atomic.write_result(f, rec)
    stamp = json.loads((d / "samples" / "CONVERTED.json").read_text())
    stamp["samples"][f.relative_to(d / "samples").as_posix()] = atomic.sha256_file(f)
    atomic.write_result(d / "samples" / "CONVERTED.json", stamp)
    mark_done(d)
    r = run_tables(tree)
    assert r.returncode == 1 and "evidence tier 'dev'" in r.stderr, r.stderr[-2000:]


def _doctor(d, glob, change):
    f = sorted((d / "work" / "results").glob(glob))[0]
    rec = json.loads(f.read_text())
    change(rec)
    f.write_text(json.dumps(rec))
    atomic.refresh_sidecar(f)
    return f


def test_a_score_the_official_scorer_does_not_reproduce_is_refused(real, tmp_path):
    d = _copy_cell(real, "ruler", tmp_path)

    def flip(rec):
        run_ = rec["runs"]["exact"]
        run_["score"] = 1.0 - run_["score"]
    _doctor(d, "ruler-*niah/*.json", flip)
    with pytest.raises(CV.ConvertRefused, match="two instruments disagree"):
        CV.convert_cell(d, real["ledger"])


def test_a_record_without_one_of_its_configs_arms_is_refused(real, tmp_path):
    d = _copy_cell(real, "nll", tmp_path)
    _doctor(d, "longppl-*/*.json", lambda rec: rec["runs"].pop("kvexact-s12"))
    with pytest.raises(CV.ConvertRefused, match="are not its config's"):
        CV.convert_cell(d, real["ledger"])


def test_grade_converts_as_a_dev_tier_record(real, tmp_path):
    """The dev set through the same path: evaluate.py grade (key tokens) -> convert -> set grade, tier dev, its NLL
    readout by the official functions (the dev tier is what tables.py refuses, test above)."""
    tmp = real["tmp"]
    p = PR.load()
    arms = []
    for kind, paper in (("skip", "nofill"), ("noproj", "identity"), ("kva", "ridgefill")):
        spec = PR.template(kind, MODEL, 12, 4096, p)
        spec["name"] = f"{paper}-s12"
        if "projector" in spec:
            spec["projector"] = str(tmp / "proj" / spec["projector"])
        arms.append(spec)
    cfg = dict(name="real-llama-grade", protocol="1.1", registration=[REAL["nll"]], model=str(tmp / "models" / MODEL),
               keytoken_tail=4096, arms=arms)
    d = tmp_path / "real-llama-grade"
    atomic.write_result(d / f"harness-{cfg['name']}.json", cfg)
    env = dict(os.environ, **_env(tmp), KVA_WORK=str(d / "work"))
    env.pop("KVA_DETERMINISTIC", None)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), str(d / f"harness-{cfg['name']}.json"),
                          "grade", "--only", "rust6-7k"], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    for f in (d / "work" / "results").rglob("*.json"):
        if f.parent.name != "canary":
            atomic.refresh_sidecar(f)
    cell = dict(name="real-llama-grade", kind="nll", registration="FINAL-TEST-GRADE", model=dict(name=MODEL),
                tail=4096, keytoken=dict(tail=4096, threshold=2.0), runs=[])
    atomic.write_result(d / "cell.json", cell)
    atomic.write_result(d / "canary.json", {"arms": {"exact": {"logits_sha256": "0" * 64}}})
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"type": "hypothesis", "id": "FINAL-TEST-GRADE", "claim": "TEST | " + REG.pins_text(
        {"cell": {cell["name"]: atomic.sha256_file(d / "cell.json")},
         "config": {cfg["name"]: atomic.sha256_file(d / f"harness-{cfg['name']}.json")}})}) + "\n")
    assert CV.convert_cell(d, ledger) == {"grade": 1}
    [rec] = samples(d, "grade").values()
    assert rec["tier"] == "dev" and rec["sample"] == "grade/rust6-7k"
    assert set(rec["arms"]) == {"exact", "skip", "noproj", "kva"}
    assert rec["arms"]["exact"]["-"]["novel_nll"] > 0 and rec["arms"]["kva"]["3/8"]["n_key"] >= 0


def test_longbench_converts_with_the_official_metric(real, tmp_path):
    """LongBench through the same path (a qa_f1 task and trec, the classification task scored on the first line):
    evaluate.py longbench -> convert -> set longbench, one unit per row, score = kva.metrics.longbench_row."""
    from kva import metrics as MT
    tmp = real["tmp"]
    p = PR.load()
    arms = []
    for kind, paper in (("noproj", "identity"), ("kva", "ridgefill")):
        spec = PR.template(kind, MODEL, 12, 1024, p)
        spec["name"] = f"{paper}-s12"
        if "projector" in spec:
            spec["projector"] = str(tmp / "proj" / spec["projector"])
        arms.append(spec)
    cfg = dict(name="real-llama-lb", protocol="1.1", registration=[REAL["ruler"]], model=str(tmp / "models" / MODEL),
               niah_format="chat", longbench_max_len=31500, arms=arms)
    d = tmp_path / "real-llama-lb"
    atomic.write_result(d / f"harness-{cfg['name']}.json", cfg)
    env = dict(os.environ, **_env(tmp), KVA_WORK=str(d / "work"))
    env.pop("KVA_DETERMINISTIC", None)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), str(d / f"harness-{cfg['name']}.json"),
                          "longbench", "--only", "lb-hotpotqa-00,lb-trec-00"], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    for f in (d / "work" / "results").rglob("*.json"):
        if f.parent.name != "canary":
            atomic.refresh_sidecar(f)
    cell = dict(name="real-llama-lb", kind="longbench", registration="FINAL-TEST-LB", model=dict(name=MODEL), tail=1024,
                runs=[])
    atomic.write_result(d / "cell.json", cell)
    atomic.write_result(d / "canary.json", {"arms": {"exact": {"logits_sha256": "0" * 64}}})
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"type": "hypothesis", "id": "FINAL-TEST-LB", "claim": "TEST | " + REG.pins_text(
        {"cell": {cell["name"]: atomic.sha256_file(d / "cell.json")},
         "config": {cfg["name"]: atomic.sha256_file(d / f"harness-{cfg['name']}.json")}})}) + "\n")
    assert CV.convert_cell(d, ledger) == {"longbench": 2}
    got = samples(d, "longbench")
    assert set(got) == {"longbench/hotpotqa.jsonl#00", "longbench/trec.jsonl#00"}
    for sid, rec in harness(d, "longbench-real-llama-lb").items():
        r = got[sid]
        assert (r["task"], r["length"], r["tier"]) == (rec["task"], 120, "locked-final")
        assert set(r["arms"]) == {"exact", "noproj", "kva"}
        for name, run_ in rec["runs"].items():
            kind = {"exact": "exact", "identity-s12": "noproj", "ridgefill-s12": "kva"}[name]
            split = "-" if kind == "exact" else "3/8"
            assert r["arms"][kind][split]["score"] == MT.longbench_row(rec["task"], run_["pred"], ["violet harbor"],
                                                                        rec["all_classes"])


def test_a_swiftkv_exact_that_is_not_the_models_is_refused(real, tmp_path):
    """The R6 arms on NLL are read against Llama's exact with the SwiftKV config's key-token mask, so its own exact must
    equal Llama's (SK1); a consistent edit of that exact (log-probs and the stored score together) is refused."""
    from kva import locked
    d = _copy_cell(real, "nll", tmp_path)

    def shift(rec):
        ex = rec["runs"]["exact"]
        ex["logprobs"] = [x - 0.01 for x in ex["logprobs"]]
        ex["score"] = locked.nll(ex["logprobs"])
    _doctor(d, "longppl-*-nll-swiftkv/*.json", shift)
    with pytest.raises(CV.ConvertRefused, match="SwiftKV checkpoint's own exact differs"):
        CV.convert_cell(d, real["ledger"])


def test_a_cell_on_another_frozen_set_is_labelled_by_it():
    """A cell with 'dataset' (fresh-long-20261008, FINAL-LLAMA-NLL-FRESH) is not the locked set: its records (and its
    skip records) carry dataset and tier = that name, never locked-final; grade stays dev; every other cell
    locked-final."""
    fresh = json.loads((FINALS / "cells" / "llama-3.1-8b-instruct-nll-fresh.json").read_text())
    locked = json.loads((FINALS / "cells" / "llama-3.1-8b-instruct-nll.json").read_text())
    assert fresh["dataset"] == "fresh-long-20261008"
    assert CV.record_tier(fresh, "longppl") == {"tier": "fresh-long-20261008", "dataset": "fresh-long-20261008"}
    assert CV.record_tier(locked, "longppl") == {"tier": "locked-final"}
    assert CV.record_tier(locked, "grade") == {"tier": "dev"}
    skipped = CV.skipped_records(dict(fresh, skipped=["long-prompts/chat_128000_0.txt"], max_len=32768), [], "1.1.3")
    assert [(r["tier"], r["dataset"]) for r in skipped] == [("fresh-long-20261008", "fresh-long-20261008")]
