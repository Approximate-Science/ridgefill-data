"""The prompt sets derived from locked RULER rows (kva/derived.py, scripts/build_derived.py), on SYNTHETIC rows in
RULER's template (made-up keys, values and filler; nothing here is or derives from a locked text):
- two-key (T1.8): turn 1 is the row, turn 2 another of its keys chosen by the seed, in RULER's one-key wording;
  malformed rows are refused by name;
- question first (T1.4): only the question line moves (instruction, question, document; answer prefix unchanged);
- the manifests are deterministic and carry no text (positive control: the synthetic keys ARE in the rows);
- run-time prompts are checked sample by sample against the manifest (a changed row, a changed manifest entry, a
  missing sample: refused by name), needle token spans come from character offsets, caps skip files by name;
- the multiturn machinery takes the built turns and the turn-2 text; T1.7's document pick;
- end to end through scripts/evaluate.py (multiturn_from ruler; ruler_variant qfirst) on make_tiny.py's models."""
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from kva import derived as D
from kva import locked as L
from kva import multiturn as MT
from kva.kva import Projector

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "models"
HEAD1 = ("A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you "
         "about the number afterwards.")
HEADN = ("Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you "
         "about the numbers afterwards.")
KEYS = ["zorbly-quill", "plinth-vortex", "marbled snorkel", "tepid-gazebo"]


def needle(key, value):
    return f"One of the special magic numbers for {key} is: {value}."


def synthetic_row(index, keys=KEYS, values=None, asked=1, filler=60, head=HEAD1, question=None, prefix=None):
    """A RULER niah row's shape with made-up content: needles at 4 depths of a filler haystack."""
    values = values or [str(1000000 + 1111111 * (index % 7) + 37 * j) for j in range(len(keys))]
    sents = [f"Filler line {j} says the grey owl {index} counts pebbles." for j in range(filler)]
    for j, (k, v) in enumerate(zip(keys, values)):
        sents.insert((j + 1) * filler // (len(keys) + 1) + j, needle(k, v))
    key = keys[asked]
    q = question or D.QUESTION.format(key=key)
    return dict(index=index, input=f"{head}\n{' '.join(sents)}\n{q}", outputs=[values[asked]],
                answer_prefix=prefix or D.PREFIX.format(key=key), length=0, length_w_model_temp=0,
                token_position_answer=0)


class Enc(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class CharTok:
    """One token per character, with offsets; chat template <u> ... </u><a>; BOS 1 on raw prompts."""
    bos_token_id = 1

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        ids = ([1] if add_special_tokens else []) + [ord(c) for c in text]
        enc = Enc(input_ids=ids)
        if return_offsets_mapping:
            enc["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return enc

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if i > 1)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "<u>" + msgs[0]["content"] + "</u><a>"


def _text(ids):
    return "".join(map(chr, ids))


@pytest.fixture
def fake_locked(tmp_path, monkeypatch):
    """A locked-set directory in the real layout (MANIFEST.tsv, FROZEN.json, ruler/<task>_<L>.jsonl) holding
    synthetic rows: 2 per file, for every file the derived sets read."""
    root = tmp_path / "locked"
    rows = ["path\tsha256\tbytes\tdomain/task\tsource_url\tlicence\n"]
    for task in D.QFIRST_TASKS:
        for n, length in enumerate(L.RULER_LENGTHS):
            rel = f"ruler/{task}_{length}.jsonl"
            head = HEADN if task in ("niah_multivalue", "niah_multiquery") else HEAD1
            body = "".join(json.dumps(synthetic_row(10 * n + i, asked=(n + i) % 4, head=head)) + "\n" for i in range(2))
            (root / "ruler").mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(body)
            rows.append(f"{rel}\t{hashlib.sha256(body.encode()).hexdigest()}\t{len(body)}\truler/{task}\tsynthetic\tnone\n")
    (root / "MANIFEST.tsv").write_text("".join(rows))
    (root / "FROZEN.json").write_text(json.dumps(
        dict(manifest_sha256=hashlib.sha256((root / "MANIFEST.tsv").read_bytes()).hexdigest())))
    monkeypatch.setattr(L, "LOCKED", root)
    monkeypatch.setattr(D, "DIR", tmp_path / "derived")
    return root


def _write_manifests():
    D.DIR.mkdir(parents=True, exist_ok=True)
    for which in D.SETS:
        D.manifest_path(which).write_text(D.dumps(D.build_manifest(which)))


# ---- two-key (T1.8)
def test_twokey_asks_another_key_of_the_row_chosen_by_the_seed():
    row = synthetic_row(3, asked=1)
    built = D.twokey(row, "ruler2k-8192-03")
    assert built == D.twokey(row, "ruler2k-8192-03")                      # deterministic
    first, second = built["turns"]
    assert (first["key"], first["value"]) == (KEYS[1], row["outputs"][0]) and built["turn1_needle"] == 1
    assert second["key"] in KEYS and second["key"] != first["key"] and built["turn2_needle"] == KEYS.index(second["key"])
    for t in built["turns"]:
        s, e = t["chars"]
        assert row["input"][s:e] == needle(t["key"], t["value"])
    assert built["ask"] == dict(question=D.QUESTION.format(key=second["key"]), reply=D.PREFIX.format(key=second["key"]))
    assert (built["user"], built["reply"]) == (row["input"], row["answer_prefix"])
    picks = {D.twokey(row, f"ruler2k-8192-{i:02d}")["turn2_needle"] for i in range(40)}
    assert picks == {0, 2, 3}                                            # the seed spreads over every other key


@pytest.mark.parametrize("kw,match", [
    (dict(keys=KEYS[:3]), "4 needles"),
    (dict(keys=[KEYS[0], KEYS[0], KEYS[2], KEYS[3]]), "distinct keys"),
    (dict(values=["123456", "2222222", "3333333", "4444444"]), "7-digit"),
    (dict(question="What is the special magic number for zorbly-quill in the text?"), "template"),
    (dict(prefix=" The answer is"), "template"),
])
def test_twokey_refuses_malformed_rows_by_name(kw, match):
    with pytest.raises(D.DerivedRowError, match=match):
        D.twokey(synthetic_row(0, **kw), "ruler2k-8192-00")
    row = synthetic_row(0)
    with pytest.raises(D.DerivedRowError, match="outputs is not exactly one needle"):
        D.twokey(dict(row, outputs=["9999999"]), "ruler2k-8192-00")


# ---- question first (T1.4)
def test_question_first_puts_the_keys_only_before_the_document():
    row = synthetic_row(5, head=HEADN)
    head, rest = row["input"].split("\n", 1)
    context, question = rest.rsplit("\n", 1)
    built = D.question_first(row, "rulerqf-niah_multivalue-8192-05")
    assert built == dict(user=f"{head}\n{question}\n{context}",
                         reply=" The special magic number mentioned in the provided text is")
    assert sorted(built["user"]) == sorted(row["input"])                 # the same characters, rearranged
    key = KEYS[1]
    assert key in row["answer_prefix"] and key not in built["reply"]     # + control: RULER's prefix restates the key
    assert built["user"].index(key) < built["user"].index("Filler line 0")   # every mention of the asked key ...
    assert built["user"].count(f"for {key} mentioned") == 1              # ... is the question, before the document
    multi = dict(row, input=row["input"].replace(D.QUESTION.format(key=key),
                                                 "What are all the special magic numbers for a, b and c mentioned in the provided text?"),
                 answer_prefix=" The special magic numbers for a, b and c mentioned in the provided text are")
    assert D.question_first(multi, "x")["reply"] == " The special magic numbers mentioned in the provided text are"
    with pytest.raises(D.DerivedRowError, match="instruction"):
        D.question_first(dict(row, input="Read this.\n" + row["input"]), "x")
    with pytest.raises(D.DerivedRowError, match="not RULER's NIAH question"):
        D.question_first(dict(row, input=row["input"] + "\nAnswer now."), "x")
    with pytest.raises(D.DerivedRowError, match="exactly once"):
        D.question_first(dict(row, answer_prefix=" The answer is"), "x")


# ---- manifests
def test_manifests_are_deterministic_and_hold_no_text(fake_locked):
    a = {w: D.dumps(D.build_manifest(w)) for w in D.SETS}
    b = {w: D.dumps(D.build_manifest(w)) for w in D.SETS}
    assert a == b
    two, qf = (json.loads(a[w]) for w in ("twokey", "qfirst"))
    assert two["count"] == 8 and qf["count"] == 32 and two["seed"] == qf["seed"] == 20261004
    assert [s["id"] for s in two["samples"][:2]] == ["ruler2k-8192-00", "ruler2k-8192-01"]
    assert qf["samples"][-1]["id"] == "rulerqf-niah_multiquery-65536-01"
    rows = "".join((fake_locked / s).read_text() for s in D.sources("qfirst"))
    for secret in KEYS + ["Filler line", "grey owl", synthetic_row(0)["outputs"][0]]:
        assert secret in rows and all(secret not in text for text in a.values())     # + control, - control
    assert "\\n" not in a["twokey"].split('"samples"')[1]                             # no text fields at all


def test_build_derived_check_detects_a_stale_manifest(fake_locked, tmp_path):
    env = dict(os.environ, KVA_LOCKED_DIR=str(fake_locked), KVA_DERIVED_DIR=str(tmp_path / "derived"))
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts" / "build_derived.py"), *a], env=env,  # noqa: E731
                                    capture_output=True, text=True)
    assert run().returncode == 0
    ok = run("--check")
    assert ok.returncode == 0 and "8 samples, identical" in ok.stdout and "32 samples, identical" in ok.stdout
    path = tmp_path / "derived" / "ruler-twokey.json"
    path.write_text(path.read_text().replace('"turn2_needle": ', '"turn2_needle": 1', 1))
    bad = run("--check")
    assert bad.returncode == 1 and "ruler-twokey.json: 8 samples, DIFFERS" in bad.stdout


# ---- run-time prompts
@pytest.mark.parametrize("fmt", ["chat", "raw"])
def test_twokey_prompts_carry_needle_spans_and_turn2_text(fake_locked, fmt):
    _write_manifests()
    tok = CharTok()
    ps, skipped = D.prompts("twokey", tok, fmt, L.manifest())
    assert skipped == [] and [p["name"] for p in ps][:3] == ["ruler2k-8192-00", "ruler2k-8192-01", "ruler2k-16384-00"]
    for p in ps:
        assert p["set"] == "multiturn" and p["max_new"] == 128 and len(p["turn_plan"]) == 2
        for t in p["turn_plan"]:
            assert _text(p["ids"][t["needle_start"]:t["needle_end"]]) == needle(t["key"], t["value"])
        assert p["asks"][0]["question"] == D.QUESTION.format(key=p["turn_plan"][1]["key"])
        assert (p["key"], p["needle_end"]) == (p["turn_plan"][0]["key"], p["turn_plan"][0]["needle_end"])
        assert _text(p["ids"]).endswith(D.PREFIX.format(key=p["key"]))
        assert _text(p["ids"]).startswith("<u>" if fmt == "chat" else "\x01")
    ps8, skipped8 = D.prompts("twokey", tok, fmt, L.manifest(), max_len=16384)
    assert skipped8 == ["ruler/niah_multikey_1_32768.jsonl", "ruler/niah_multikey_1_65536.jsonl"] and len(ps8) == 4


def test_qfirst_prompts_are_ruler_records_with_the_question_first(fake_locked):
    _write_manifests()
    ps, _ = D.prompts("qfirst", CharTok(), "chat", L.manifest())
    assert len(ps) == 32 and {p["task"] for p in ps} == set(D.QFIRST_TASKS)
    p = ps[0]
    assert (p["set"], p["metric"], p["variant"], p["ruler_task"]) == ("ruler", "string_match_all", "qfirst", "niah")
    text = _text(p["ids"])
    assert text.index("What is the special magic number") < text.index("Filler line 0")
    assert text.endswith("</u><a> The special magic number mentioned in the provided text is")   # no key at the end
    # the locked set's own ruler entry point routes the variant here; an unknown variant is refused
    cfg = dict(ruler_variant="qfirst", niah_format="chat", ruler_max_len=8192)
    via, skipped = L.prompts("ruler", CharTok(), cfg, "no-such-model", dict(models={}))
    assert [q["name"] for q in via] == [q["name"] for q in ps if q["length"] == 8192] and len(skipped) == 12
    with pytest.raises(SystemExit, match="only RULER variant is 'qfirst'"):
        L.prompts("ruler", CharTok(), dict(cfg, ruler_variant="blind"), "no-such-model", dict(models={}))


def test_a_declared_ruler_selection_loads_only_its_files(fake_locked):
    """ruler_tasks / ruler_lengths (the finals' NIAH subset at 8K-32K): only those files are read; the cap is
    separate; an unknown task is refused by name."""
    _write_manifests()
    tok, man = CharTok(), L.manifest()
    sel = dict(niah_format="raw", ruler_tasks=["niah_single_1", "niah_multikey_1"], ruler_lengths=[8192, 32768])
    ps, skipped = L.prompts("ruler", tok, sel, "m", dict(models={}))
    assert skipped == [] and sorted({(p["task"], p["length"]) for p in ps}) == [
        ("niah_multikey_1", 8192), ("niah_multikey_1", 32768), ("niah_single_1", 8192), ("niah_single_1", 32768)]
    ps, skipped = L.prompts("ruler", tok, dict(sel, ruler_max_len=16384), "m", dict(models={}))
    assert {p["length"] for p in ps} == {8192} and skipped == ["ruler/niah_single_1_32768.jsonl",
                                                                "ruler/niah_multikey_1_32768.jsonl"]
    qf, _ = L.prompts("ruler", tok, dict(sel, ruler_variant="qfirst"), "m", dict(models={}))
    assert {(p["task"], p["length"]) for p in qf} == {(p["task"], p["length"]) for p in
                                                       L.prompts("ruler", tok, sel, "m", dict(models={}))[0]}
    with pytest.raises(SystemExit, match="not RULER's tasks"):
        L.prompts("ruler", tok, dict(sel, ruler_tasks=["niah_single_9"]), "m", dict(models={}))


def test_a_changed_row_or_manifest_entry_is_refused_by_name(fake_locked):
    _write_manifests()
    tok, man = CharTok(), L.manifest()
    path = D.manifest_path("twokey")
    good = path.read_text()
    m = json.loads(good)
    m["samples"][3]["text_sha256"] = "0" * 64
    path.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="DerivedSetMismatch: ruler2k-16384-01: text_sha256 differ"):
        D.prompts("twokey", tok, "raw", man)
    m = json.loads(good)
    m["samples"].pop(0)
    path.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="ruler2k-8192-00 .* is not in"):
        D.prompts("twokey", tok, "raw", man)
    m = json.loads(good)
    m["samples"].append(dict(m["samples"][0], id="ruler2k-8192-07"))
    path.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="lists samples the locked rows do not have"):
        D.prompts("twokey", tok, "raw", man)
    m = json.loads(good)
    m["seed"] = 1
    path.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="another locked set, seed or source list"):
        D.prompts("twokey", tok, "raw", man)
    path.write_text(good)
    rel = "ruler/niah_multikey_1_8192.jsonl"                 # a changed locked row: the MANIFEST check catches it first
    (fake_locked / rel).write_text((fake_locked / rel).read_text().replace("pebbles", "marbles", 1))
    with pytest.raises(SystemExit, match="does not match its MANIFEST.tsv sha256"):
        D.prompts("twokey", tok, "raw", L.manifest())


def test_token_span_refuses_ids_that_are_not_the_texts_encoding():
    tok = CharTok()
    ids = [1] + [ord(c) for c in "abc needle xyz"]
    assert D.token_span(tok, "abc needle xyz", 4, 10, ids, 1) == (5, 11)
    with pytest.raises(SystemExit, match="not the text's encoding"):
        D.token_span(tok, "abc needle xyz", 4, 10, ids, 0)


# ---- the multiturn machinery takes the built turns
@pytest.mark.parametrize("fmt", ["chat", "raw"])
def test_multiturn_reads_the_built_turns_and_turn2_text(fake_locked, fmt):
    _write_manifests()
    tok = CharTok()
    p = D.prompts("twokey", tok, fmt, L.manifest())[0][0]
    assert MT.plan(tok, p, 2) == p["turn_plan"]
    with pytest.raises(ValueError, match="has 2"):
        MT.plan(tok, p, 3)
    pieces = ((tok("</a>", False)["input_ids"], tok("<u2>", False)["input_ids"]) if fmt == "chat" else ([], []))
    ask = _text(MT.ask_text_ids(tok, fmt, p["asks"][0], pieces))
    key2 = p["turn_plan"][1]["key"]
    want = (f"</a>{D.QUESTION.format(key=key2)}<u2>{D.PREFIX.format(key=key2)}" if fmt == "chat"
            else f"\n{D.QUESTION.format(key=key2)}{D.PREFIX.format(key=key2)}")
    assert ask == want


# ---- T1.7's documents
def test_decode_documents_are_seeded_stratified_and_capped():
    paths = [f"long-prompts/{d}_{L_}_{i}.txt" for d in ("prose", "code", "chat") for L_ in (8192, 16384, 32768, 65536)
             for i in range(3)] + ["long-prompts/SPLITS.tsv", "ruler/vt_8192.jsonl"]
    got = D.pick_decode_docs(paths, n=7)
    assert got == D.pick_decode_docs(paths, n=7) and len(set(got)) == 7
    lengths = [int(g.rsplit("_", 2)[-2]) for g in got]
    assert max(lengths) <= 32768 and lengths[:3] == [8192, 16384, 32768]   # round-robin over the bins
    assert D.pick_decode_docs(paths, n=7, seed=1) != got                    # the seed matters
    with pytest.raises(ValueError, match="only 27"):
        D.pick_decode_docs(paths, n=28)


def test_decode_documents_of_the_locked_manifest():
    """The 20 documents the decode cells name (locked MANIFEST.tsv paths only; no text is read)."""
    if not (ROOT / "datasets" / "test-set-locked" / "MANIFEST.tsv").is_file():
        pytest.skip("needs datasets/test-set-locked/MANIFEST.tsv (not in the public code)")
    paths = [line.split("\t")[0] for line in (ROOT / "datasets" / "test-set-locked" / "MANIFEST.tsv").read_text()
             .splitlines()[1:]]
    got = D.pick_decode_docs(paths)
    assert len(got) == 20 and {int(g.rsplit("_", 2)[-2]) for g in got} == {8192, 16384, 32768}


# ---- end to end through scripts/evaluate.py (CPU dry run, no protocol stamp) on make_tiny.py's models
def _env(tmp, locked_dir):
    env = dict(os.environ, KVA_CPU_DRYRUN="1", KVA_WORK=str(tmp / "work"), KVA_BULK=str(tmp / "bulk"),
               HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HIP_VISIBLE_DEVICES="-1",
               ROCR_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4", KVA_LOCKED_DIR=str(locked_dir),
               KVA_DERIVED_DIR=str(D.DIR))
    env.pop("KVA_DETERMINISTIC", None)
    return env


def test_twokey_and_qfirst_end_to_end(fake_locked, tmp_path):
    if not (MODELS / "llama-3.1-8b-instruct" / "tokenizer.json").exists():
        pytest.skip("needs the local Llama-3.1 tokenizer")
    _write_manifests()
    env = _env(tmp_path, fake_locked)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tiny.py"), str(tmp_path / "models")], env=env,
                   check=True, capture_output=True)
    m = tmp_path / "models" / "llama-3.1-8b-instruct"
    g = torch.Generator().manual_seed(0)
    proj = tmp_path / "llama-sweep-s12.safetensors"
    Projector(12, {i: torch.eye(64) + 0.02 * torch.randn(64, 64, generator=g) for i in range(12, 32)},
              {i: 0.02 * torch.randn(64, generator=g) for i in range(12, 32)}).save(proj, dict(kind="test"))
    rows10 = dict(kind="qattn", budget=0.1)
    arms = [dict(name="ridgefill", split=12, tail=256, fill="proj", projector=str(proj)),
            dict(name="ridgefill+rows", split=12, tail=256, fill="proj", projector=str(proj), select=rows10),
            dict(name="identity+rows", split=12, tail=256, fill="identity", select=rows10)]
    runs = {}
    for which, extra, only in (("multiturn", dict(multiturn_from="ruler", multiturn_turns=2), "ruler2k-8192-01"),
                               ("ruler", dict(ruler_variant="qfirst"), "rulerqf-niah_single_1-8192-00")):
        cfg = dict(name=f"derived-{which}", model=str(m), niah_format="chat", ruler_max_len=8192, arms=arms, **extra)
        path = tmp_path / f"{which}.json"
        path.write_text(json.dumps(cfg))
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), str(path), which, "--only", only],
                             env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-3000:]
        runs[which] = json.loads((tmp_path / "work" / "results" / f"{which}-derived-{which}" / f"{only}.json").read_text())
    rec = runs["multiturn"]
    plan = D.prompts("twokey", CharTok(), "chat", L.manifest())[0][1]["turn_plan"]
    assert [t["key"] for t in rec["turns"]] == [t["key"] for t in plan] and "asks" not in rec
    assert rec["derived"] == "ruler-twokey" and rec["provenance"][-1]["config"]["multiturn_from"] == "ruler"
    assert rec["provenance"][-1]["config"]["skipped"] == [f"ruler/niah_multikey_1_{n}.jsonl" for n in (16384, 32768, 65536)]
    for name, run in rec["runs"].items():
        assert [t["turn"] for t in run["turns"]] == [1, 2] and all(t["key"] for t in run["turns"])
        assert all(t["needle_projected"] == (t["needle_end"] <= run["P"]) for t in run["turns"])
    assert rec["runs"]["ridgefill+rows"]["n_rows"] > 0 and "select" not in rec["runs"]["ridgefill"]
    q = runs["ruler"]
    assert q["variant"] == "qfirst" and q["set"] == "ruler" and q["provenance"][-1]["config"]["ruler_variant"] == "qfirst"
    assert sorted(q["runs"]) == ["exact", "identity+rows", "ridgefill", "ridgefill+rows"]
    assert all(r["metric"] == "string_match_all" and 0 <= r["score"] <= 1 for r in q["runs"].values())


def test_qfirst_honours_the_v112_generation_cap(fake_locked):
    _write_manifests()
    ps, _ = D.prompts("qfirst", CharTok(), "chat", L.manifest(), max_new=dict(niah_single_1=32, niah_multikey_1=32))
    steps = {}
    for p in ps:
        steps.setdefault(p["task"], set()).add(p["max_new"])
    assert steps["niah_single_1"] == steps["niah_multikey_1"] == {32}
    assert steps["niah_multivalue"] == steps["niah_multiquery"] == {D.NIAH_STEPS}
