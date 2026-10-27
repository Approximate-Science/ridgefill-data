"""Prompt sets derived from LOCKED RULER rows (paper T1.8 and T1.4; protocol v1.1, PROTOCOL.md section 2 and open
item 6). Nothing here prints, stores or copies a locked text: the committed manifests (datasets/derived/*.json, built
by scripts/build_derived.py) hold sample ids, row numbers, needle ordinals and sha256 hashes only; the prompts are
rebuilt from the locked rows at run time (kva/locked.py: datasets/test-set-locked or KVA_LOCKED_DIR) and every sample
is checked against its manifest entry before anything runs.

twokey (T1.8, the second question on a cached prefix; evaluate.py set multiturn with "multiturn_from": "ruler").
  Rows: every row of ruler/niah_multikey_1_<L>.jsonl: RULER's four keyed needles "One of the special magic numbers
  for KEY is: VALUE.", one of them asked. Turn 1 is the row as T1.1 runs it (input, then answer_prefix). Turn 2 asks
  one of the three other keys, chosen by random.Random(f"{SEED}:{sample id}") from the others in document order, in
  RULER's own wording (its niah template for one key: QUESTION and PREFIX below). A row is refused by name unless it
  has exactly 4 needles with 4 distinct keys and 4 distinct 7-digit values, its outputs is exactly one needle's
  value, and its question line and answer prefix are the template's for that needle's key.
qfirst (T1.4, question before the document; evaluate.py set ruler with "ruler_variant": "qfirst").
  Rows: every row of the four RULER NIAH tasks (QFIRST_TASKS, the retrieval subset C3 is about). RULER's input is
  instruction \\n context \\n question; the variant is instruction \\n question \\n context, and RULER's answer prefix
  loses " for <query>" (" The special magic number(s) mentioned in the provided text is/are"), so the keys appear
  only BEFORE the document and a question-attention selector (identity+rows, ridgefill+rows: the last 64 rows) reads
  the document's end and the neutral prefix only: T1.4's "selector reads the document only" (2026-10-04, after the
  T1.4 arms were set to the qattn10 rows; a prefix that restated the key would let the selector see it). A row is
  refused by name unless its first line is RULER's NIAH instruction, its last line RULER's NIAH question
  ("What is / are all the special magic number(s) for <query> mentioned in the provided text?") and its answer prefix
  carries " for <query>" exactly once.
Sample ids: ruler2k-<L>-<row:02d> and rulerqf-<task>-<L>-<row:02d>. A manifest sample records the locked file, its
MANIFEST.tsv sha256, the row number, RULER's index, the nominal length, row_sha256 (the row's JSON line) and
text_sha256 (the built texts); two-key adds turn1_needle / turn2_needle (the asked needles' ordinals among the row's
needles in document order). Needle token positions are computed per tokenizer at run time from character offsets
(token_span), never stored in a manifest; question-first prompts carry the scored needles' token spans as 'needles'
(kva.locked.needle_spans), so a result record says which needles sit in the filled region.
"""
import hashlib
import json
import os
import random
import re
from pathlib import Path

from kva import data, locked

SEED = 20261004
DIR = Path(os.environ["KVA_DERIVED_DIR"]) if os.environ.get("KVA_DERIVED_DIR") else data.DATASETS / "derived"
NEEDLE = re.compile(r"One of the special magic numbers for ([^\n.:]+?) is: (\d+)\.")
QUESTION = "What is the special magic number for {key} mentioned in the provided text?"
PREFIX = " The special magic number for {key} mentioned in the provided text is"
QUERY = re.compile(r"What (?:is|are all) the special magic numbers? for (.+) mentioned in the provided text\?")
INSTRUCTION = re.compile(r"(Some|A) special magic numbers? (are|is) hidden within the following text\. Make sure to "
                         r"memorize it\. I will quiz you about the numbers? afterwards\.")
TWOKEY_TASK = "niah_multikey_1"
QFIRST_TASKS = ("niah_single_1", "niah_multikey_1", "niah_multivalue", "niah_multiquery")
NIAH_STEPS = 128                         # RULER's tokens_to_generate for niah (kva/locked.py RULER_TASKS)
SETS = ("twokey", "qfirst")
RULES = {
    "twokey": "T1.8: every row of ruler/niah_multikey_1_<L>.jsonl; turn 1 = the row (input, answer_prefix); turn 2 asks "
              "another of its 4 keys, random.Random(f'{seed}:{id}').choice over the 3 others in document order, in "
              "RULER's one-key niah wording (QUESTION, PREFIX in kva/derived.py)",
    "qfirst": "T1.4: every row of the 4 RULER NIAH tasks; input instruction/context/question becomes "
              "instruction/question/context; answer_prefix without ' for <query>' (the keys only before the document)",
}


class DerivedRowError(ValueError):
    """A locked row does not have the structure the derived set needs: the builder refuses it by name."""


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()


def manifest_path(which):
    return DIR / f"ruler-{which}.json"


def sources(which):
    """The locked files a derived set reads, in manifest order."""
    tasks = (TWOKEY_TASK,) if which == "twokey" else QFIRST_TASKS
    return [f"ruler/{t}_{L}.jsonl" for t in tasks for L in locked.RULER_LENGTHS]


def sample_id(which, rel, row):
    task, L = rel[len("ruler/"):-len(".jsonl")].rsplit("_", 1)
    return f"ruler2k-{L}-{row:02d}" if which == "twokey" else f"rulerqf-{task}-{L}-{row:02d}"


def needles(text):
    """[(key, value, char start, char end)] of RULER's numeric needles, in document order."""
    return [(m.group(1), m.group(2), m.start(), m.end()) for m in NEEDLE.finditer(text)]


def twokey(row, sid):
    """Turn 1 = the row; turn 2 = another key of the same row (module doc)."""
    ns = needles(row["input"])
    keys, values = [n[0] for n in ns], [n[1] for n in ns]
    if len(ns) != 4 or len(set(keys)) != 4 or len(set(values)) != 4 or not all(re.fullmatch(r"\d{7}", v) for v in values):
        raise DerivedRowError(f"{sid}: needs 4 needles with distinct keys and distinct 7-digit values, has {len(ns)} "
                              f"needles, {len(set(keys))} keys, {len(set(values))} values")
    asked = [j for j, v in enumerate(values) if [v] == row["outputs"]]
    if len(asked) != 1:
        raise DerivedRowError(f"{sid}: outputs is not exactly one needle's value ({len(asked)} matches)")
    t = asked[0]
    if row["input"].rsplit("\n", 1)[-1] != QUESTION.format(key=keys[t]) or row["answer_prefix"] != PREFIX.format(key=keys[t]):
        raise DerivedRowError(f"{sid}: the question line or answer prefix is not RULER's one-key niah template for the "
                              f"asked needle")
    u = random.Random(f"{SEED}:{sid}").choice([j for j in range(4) if j != t])
    turns = [dict(key=keys[j], value=values[j], chars=(ns[j][2], ns[j][3])) for j in (t, u)]
    return dict(user=row["input"], reply=row["answer_prefix"], turns=turns, turn1_needle=t, turn2_needle=u,
                ask=dict(question=QUESTION.format(key=keys[u]), reply=PREFIX.format(key=keys[u])))


def question_first(row, sid):
    """instruction \\n question \\n context, and the answer prefix without ' for <query>' (module doc)."""
    head, _, rest = row["input"].partition("\n")
    context, _, question = rest.rpartition("\n")
    if not INSTRUCTION.fullmatch(head):
        raise DerivedRowError(f"{sid}: the first line is not RULER's NIAH instruction")
    q = QUERY.fullmatch(question)
    if q is None or not context:
        raise DerivedRowError(f"{sid}: the last line is not RULER's NIAH question after a document")
    reply = row["answer_prefix"].replace(f" for {q.group(1)}", "")
    if row["answer_prefix"].count(f" for {q.group(1)}") != 1 or q.group(1) in reply:
        raise DerivedRowError(f"{sid}: the answer prefix does not carry the question's keys exactly once")
    return dict(user=f"{head}\n{question}\n{context}", reply=reply)


def build_one(which, row, sid):
    return twokey(row, sid) if which == "twokey" else question_first(row, sid)


def text_sha256(which, built):
    parts = [built["user"], built["reply"]]
    if which == "twokey":
        parts += [built["ask"]["question"], built["ask"]["reply"]]
    return _sha(parts)


def _lines(rel):
    return [line for line in (locked.LOCKED / rel).read_text().splitlines() if line.strip()]


def _frozen_sha():
    return json.loads((locked.LOCKED / "FROZEN.json").read_text())["manifest_sha256"]


def build_manifest(which):
    """The manifest of a derived set from the locked rows (scripts/build_derived.py writes it). No text inside."""
    if which not in SETS:
        raise ValueError(f"unknown derived set {which!r}: one of {SETS}")
    man = locked.manifest()
    samples = []
    for rel in sources(which):
        sha = locked.source(rel, man)
        for i, line in enumerate(_lines(rel)):
            row, sid = json.loads(line), sample_id(which, rel, i)
            built = build_one(which, row, sid)
            s = dict(id=sid, source=rel, source_sha256=sha, row=i, index=row["index"],
                     length=int(rel[:-len(".jsonl")].rsplit("_", 1)[1]),
                     row_sha256=hashlib.sha256(line.encode()).hexdigest(), text_sha256=text_sha256(which, built))
            if which == "twokey":
                s.update(turn1_needle=built["turn1_needle"], turn2_needle=built["turn2_needle"])
            samples.append(s)
    return dict(set=f"ruler-{which}", version=1, seed=SEED, builder="kva/derived.py (scripts/build_derived.py)",
                rule=RULES[which],
                locked_manifest_sha256=_frozen_sha(), sources=sources(which), count=len(samples), samples=samples)


def dumps(manifest):
    return json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"


def token_spans(tok, text, chars, ids, base):
    """[[first, last + 1)] of the tokens of `text` (encoded alone, as kva.locked.wrap does, placed at ids[base:]) that
    overlap each character span [start, end) of `chars`; one encoding with offsets for all of them; refuses if that
    encoding is not the one in ids."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    got = list(enc["input_ids"])
    if ids[base:base + len(got)] != got:
        raise SystemExit("DerivedSetMismatch: the prompt's token ids are not the text's encoding (tokenizer offsets)")
    out = []
    for start, end in chars:
        hit = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < end and e > start]
        if not hit:
            raise SystemExit("DerivedSetMismatch: a needle maps to no token")
        out.append((base + hit[0], base + hit[-1] + 1))
    return out


def token_span(tok, text, start, end, ids, base):
    """token_spans for one character span."""
    return token_spans(tok, text, [(start, end)], ids, base)[0]


def _check(which, sid, s, rel, sha, line, built):
    if s is None:
        raise SystemExit(f"DerivedSetMismatch: {sid} ({rel}) is not in {manifest_path(which)}")
    got = dict(source=rel, source_sha256=sha, row_sha256=hashlib.sha256(line.encode()).hexdigest(),
               text_sha256=text_sha256(which, built))
    if which == "twokey":
        got.update(turn1_needle=built["turn1_needle"], turn2_needle=built["turn2_needle"])
    bad = [k for k, v in got.items() if s.get(k) != v]
    if bad:
        raise SystemExit(f"DerivedSetMismatch: {sid}: {', '.join(bad)} differ from {manifest_path(which)} (the locked "
                         f"row or the builder changed; the manifest is pinned in the FINAL record)")


def prompts(which, tok, fmt, man, max_len=None, tasks=None, lengths=None, max_new=None):
    """(prompts, skipped locked files) of a derived set, every sample checked against its committed manifest.
    tasks / lengths: a declared selection of its files (kva.locked.ruler_selection); files outside it are not read."""
    path = manifest_path(which)
    if not path.is_file():
        raise SystemExit(f"DerivedSetMismatch: {path} does not exist (scripts/build_derived.py builds it)")
    m = json.loads(path.read_text())
    if m.get("locked_manifest_sha256") != _frozen_sha() or m.get("seed") != SEED or m.get("sources") != sources(which):
        raise SystemExit(f"DerivedSetMismatch: {path} was built from another locked set, seed or source list")
    by_id = {s["id"]: s for s in m["samples"]}
    out, skipped, seen = [], [], set()
    tasks, lengths = locked.ruler_selection(tasks, lengths)
    caps = locked.check_max_new(max_new)             # v1.1.2 ruler_max_new (qfirst rows are RULER NIAH rows)
    chosen = [rel for rel in sources(which)
              if rel[len("ruler/"):-len(".jsonl")].rsplit("_", 1)[0] in tasks and int(rel[:-len(".jsonl")].rsplit("_", 1)[1]) in lengths]
    for rel in chosen:
        L = int(rel[:-len(".jsonl")].rsplit("_", 1)[1])
        if max_len and L > max_len:
            skipped.append(rel)
            continue
        sha = locked.source(rel, man)
        task = rel[len("ruler/"):-len(".jsonl")].rsplit("_", 1)[0]
        for i, line in enumerate(_lines(rel)):
            row, sid = json.loads(line), sample_id(which, rel, i)
            try:
                built = build_one(which, row, sid)
            except DerivedRowError as e:
                raise SystemExit(f"DerivedSetMismatch: {e}")
            _check(which, sid, by_id.get(sid), rel, sha, line, built)
            seen.add(sid)
            ids = locked.wrap(tok, fmt, built["user"], built["reply"])
            q = dict(name=sid, task=task, ruler_task="niah", length=L, source=rel, source_sha256=sha, row=i,
                     index=row["index"], max_new=caps.get(task, NIAH_STEPS), format=fmt, derived=f"ruler-{which}",
                     derived_sha256=by_id[sid]["text_sha256"], ids=ids)
            if which == "qfirst":
                q.update(set="ruler", metric="string_match_all", refs=row["outputs"], variant="qfirst",
                         needles=locked.needle_spans(tok, fmt, built["user"], built["reply"], ids, row["outputs"], sid))
            else:
                # where the encoded text sits in ids (kva.locked.wrap): chat after the template head, raw after the BOS
                if fmt == "chat":
                    head, text = len(data.chat_pieces(tok)[0]), built["user"]
                else:
                    head, text = len(data.bos_ids(tok)), built["user"] + built["reply"]
                plan = []
                for t in built["turns"]:
                    s, e = token_span(tok, text, *t["chars"], ids, head)
                    plan.append(dict(key=t["key"], value=t["value"], needle_start=s, needle_end=e))
                q.update(set="multiturn", turn_plan=plan, asks=[built["ask"]], key=plan[0]["key"], value=plan[0]["value"],
                         needle_start=plan[0]["needle_start"], needle_end=plan[0]["needle_end"])
            out.append(q)
    extra = sorted(s["id"] for s in m["samples"] if s["source"] in chosen and s["source"] not in skipped
                   and s["id"] not in seen)
    if extra:
        raise SystemExit(f"DerivedSetMismatch: {path} lists samples the locked rows do not have: {extra[:5]}")
    return out, skipped


def pick_decode_docs(paths, n=20, max_len=32768, seed=SEED):
    """T1.7's documents (PROTOCOL.md section 2: manifest order under seed 20261004, each length bin <= max_len
    represented): the long-prompt files of the locked MANIFEST.tsv (in its order) up to max_len tokens, shuffled by
    random.Random(seed), then taken round-robin over the length bins (ascending) until n. Returns longppl prompt
    names (lp-<stem>), in pick order."""
    docs = [p for p in paths if p.startswith("long-prompts/") and p.endswith(".txt")]
    length = {p: int(p.rsplit("/", 1)[-1][:-len(".txt")].rsplit("_", 2)[-2]) for p in docs}
    docs = [p for p in docs if length[p] <= max_len]
    random.Random(seed).shuffle(docs)
    bins = {L: [p for p in docs if length[p] == L] for L in sorted(set(length[p] for p in docs))}
    picked = []
    while len(picked) < n and any(bins.values()):
        for L in bins:
            if bins[L] and len(picked) < n:
                picked.append(bins[L].pop(0))
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} long-prompt documents <= {max_len} tokens, {n} wanted")
    return ["lp-" + p.rsplit("/", 1)[-1][:-len(".txt")] for p in picked]
