#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Calibration documents vs evaluation sets: shared word runs and whole-text SHA-256 (CPU only, standard library).

Adapted from the kva repository's T0.2 check (checks/t02/overlap.py), with the same normalisation so the results
are comparable: a text is split on whitespace (str.split(); case, punctuation and digits are kept), an n-shingle is
n consecutive words joined by one space, and JSON / JSONL inputs are decoded with every string value counted as one
text (.txt files whole). Shingles are compared by hash; the longest shared run of a reported pair is then measured
word for word. Whole-text SHA-256 (UTF-8) of every calibration document is compared with the SHA-256 of every
evaluation text and of every evaluation file's bytes.

Nothing is printed or written but document ids, roles, counts, lengths and hashes: no calibration or evaluation
text, no source names, no paths beyond the file names given on the command line (relative to the input).

Calibration side (texts are not released; given as arguments):
  --plans DIR          plan-train.json ('calib' entries), plan-held.json, plan-sterm.json
  --exclude FILE       calibration-exclude.json: its 'train' names are left out of every map fit
  --fn-plans DIR       PLAN-b0.json and PLAN-b1.json, the two batches of the Flash-Next fit
  --chat-frame FILE    JSON {user_open, assistant_open, asks}: the capture's chat frame for the chat copies
  --chat-check FILE    optional: a batch plan whose 'chatold' items hold stored chat copies; each must equal the
                       copy rebuilt here (checks the rebuild rule)
  --fit-record G=FILE, --held-record G=FILE, --sterm-record G=FILE, --table-record G=FILE   (repeatable)
                       fit records naming the documents each user G (reference, qwen38-27b, flashnext) used:
                       fit-sweep / calib-docs JSON ('docs': names or count), 'held' lists, sterm records
                       (config docs or prompts), token-frequency tables (safetensors metadata provenance docs)
  --extra-calib R=DIR  optional: every DIR/*.txt as a calibration document with role R (no plans needed)
Roles: fit (plan-train 'calib' minus the exclusion list), flashnext-fit (PLAN-b1 'big'), chat-copy (PLAN-b0
'chattrain', PLAN-b1 'chatbig', and the chat copies of the lambda documents), lambda (plan-held, >= 1024 tokens,
minus the exclusion list and one withheld held-out document), sterm (plan-sterm, all), excluded-token-table (the excluded 'train'
names, which the token-frequency tables still count).
Chat copies are rebuilt as the capture built them: user_open + text + blank line + asks[i % len(asks)] +
assistant_open, where i is the document's index in the capture order (plan-train 'calib' in file order, then
the lambda documents).

Evaluation side (repeatable, NAME=PATH, PATH a file or a directory of .txt / .json / .jsonl):
  --eval NAME=PATH     texts; a JSONL record is one item, named by its '_id' or 'id' field (else its row)
  --eval-ids NAME=PATH token-id prompts: every JSON object with an integer 'ids' list is one item, decoded with
                       --gguf (the vocabulary of the model that made the ids; control tokens rendered as text);
                       --decode-probes FILE (JSONL {text, ids}) must decode exactly, and items naming a source
                       item of --decode-source NAME must share >= 90 % of their 50-shingles with it

Controls (in memory, never written): per evaluation set, the first 200 words of its first item of >= 200 words
must be flagged on that item (positive shingle control) and that item's whole text must collide by SHA-256
(positive hash control); 600 seeded random words must not be flagged anywhere (negative control).

Exit codes: 0 no calibration document shares a 50-word run or a whole-text hash with an evaluation item;
1 at least one does (every pair is in --out); 2 missing input or integrity error; 3 a control failed.

Usage (paths are examples):
  python3 -I tools/overlap_lbv2.py --plans $PLANS --exclude calibration-exclude.json --fn-plans $FN_PLANS \\
    --chat-frame $FRAME --fit-record reference=calib-docs-qwen3-8b-sweep.json ... \\
    --eval lbv2=items.jsonl --eval kit-ruler=data/ruler.jsonl --eval-ids kit-kl=data/kld --gguf $MODEL_GGUF \\
    --out calib-vs-longbench-v2.json --docs-tsv calibration-docs.tsv
"""
import argparse
import hashlib
import json
import random
import re
import struct
import sys
import time
from pathlib import Path

NS = (50, 13)
EXTS = (".txt", ".json", ".jsonl")
MAX_COMBOS = 5_000_000                       # anchor pairs examined per (document, text) for the longest run
# One held-out document is never used (the reference implementation's held-out exclusion); it is matched by the
# sha256 of its name, so the name itself is not published.
HELD_EXCLUDE_NAME_SHA256 = {"84f57b5fa277a7ed4af11992762af87605d6b02b1d71b0589743ac4a29d9c73d"}
DOC_ID = re.compile(r"^(chat(train|big|held)/)?(trainc\d+|heldc\d+|held\d-(rust|code)|sterm\d+|"
                    r"(code|prose|rust|techdocs|web)-\d{3})$")
LBV2_ID = re.compile(r"^[0-9a-f]{24}$")
ROLES = ("fit", "flashnext-fit", "chat-copy", "lambda", "sterm", "excluded-token-table")
GROUPS = ("reference", "qwen38-27b", "flashnext")
CONTROL = "__control__"


class CheckError(Exception):
    exit_code = 2


class ControlError(CheckError):
    exit_code = 3


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_text(s):
    return sha_bytes(s.encode("utf-8"))


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def gram_list(words, n):
    """Hash of every n-word shingle, in position order (t02: hash(' '.join(w[i:i + n])))."""
    return [hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)]


def strings(obj):
    """Every string value of a JSON value, each on its own (t02 / qfn.bigcap_overlap.strings)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CheckError(f"missing input: {Path(path).name}") from None


def kv(spec, what):
    if "=" not in spec:
        raise CheckError(f"{what} wants NAME=PATH, got {spec!r}")
    k, v = spec.split("=", 1)
    return k, Path(v)


# ---- calibration side --------------------------------------------------------------------------------------
def record_names(path, kind):
    """(names or None, count) a fit record gives for one kind of use."""
    if kind == "table":
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n)).get("__metadata__", {})
        names = json.loads(meta.get("provenance", "{}")).get("config", {}).get("docs")
        if not isinstance(names, list):
            raise CheckError(f"{Path(path).name}: no provenance docs list in the table metadata")
        return names, len(names)
    obj = load_json(path)
    cfg = obj.get("provenance", {}).get("config", {}) if isinstance(obj.get("provenance"), dict) else {}
    if kind == "fit":
        docs = obj.get("docs")
        if isinstance(docs, list):
            return docs, len(docs)
        if isinstance(docs, int):
            return None, docs
        raise CheckError(f"{Path(path).name}: no 'docs' list or count")
    if kind == "held":
        held = cfg.get("held", obj.get("held"))
        if not isinstance(held, list):
            raise CheckError(f"{Path(path).name}: no 'held' list")
        out = []
        for h in held:
            parts = str(h).rstrip("/").split("/")
            out.append(f"{parts[-2]}/{parts[-1]}" if len(parts) > 1 and parts[-2].startswith("chat") else parts[-1])
        return out, len(out)
    if kind == "sterm":
        docs = cfg.get("docs") or [p["name"] for p in obj.get("prompts", [])]
        if not docs:
            raise CheckError(f"{Path(path).name}: no sterm docs or prompts")
        return docs, len(docs)
    raise ValueError(kind)


def build_calibration(a, inputs):
    """[{doc_id, role, used_by, text}] and the checks made while building it."""
    docs, checks = [], {}

    def add(doc_id, role, text):
        if not DOC_ID.match(doc_id) and role in ROLES:
            raise CheckError(f"document id {doc_id!r} is not an opaque fit-record name")
        if any(d["doc_id"] == doc_id for d in docs):
            raise CheckError(f"duplicate calibration document id {doc_id}")
        docs.append(dict(doc_id=doc_id, role=role, used_by=set(), text=text))

    if a.plans:
        for need in (a.exclude, a.fn_plans, a.chat_frame):
            if need is None:
                raise CheckError("--plans needs --exclude, --fn-plans and --chat-frame")
        plan = {}
        for k in ("train", "held", "sterm"):
            p = a.plans / f"plan-{k}.json"
            plan[k] = load_json(p)
            inputs[f"calibration/plan-{k}.json"] = sha_file(p)
        excl = load_json(a.exclude)
        inputs["calibration/" + a.exclude.name] = sha_file(a.exclude)
        frame = load_json(a.chat_frame)
        inputs["calibration/chat-frame (private)"] = sha_file(a.chat_frame)
        b = {}
        for k in (0, 1):
            p = a.fn_plans / f"PLAN-b{k}.json"
            b[k] = load_json(p)
            inputs[f"calibration/PLAN-b{k}.json"] = sha_file(p)

        train_all = [x for x in plan["train"] if x["kind"] == "calib"]
        ex = set(excl["train"])
        fit = [x for x in train_all if x["name"] not in ex]
        excluded = [x for x in train_all if x["name"] in ex]
        held = [x for x in plan["held"] if x["tokens"] >= 1024 and x["name"] not in set(excl["held"])
                and hashlib.sha256(x["name"].encode()).hexdigest() not in HELD_EXCLUDE_NAME_SHA256]
        if ex - {x["name"] for x in train_all}:
            raise CheckError("an excluded train name is not a plan-train 'calib' document")
        order = [x["name"] for x in train_all] + [x["name"] for x in held]       # the capture order
        if len(set(order)) != len(order):
            raise CheckError("capture order has repeated names")
        idx = {n: i for i, n in enumerate(order)}
        text_of = {x["name"]: x["text"] for x in train_all + held}

        def chat(name):
            i = idx[name]
            return frame["user_open"] + text_of[name] + "\n\n" + frame["asks"][i % len(frame["asks"])] + \
                frame["assistant_open"]

        for x in fit:
            add(x["name"], "fit", x["text"])
        for x in excluded:
            add(x["name"], "excluded-token-table", x["text"])
        for x in held:
            add(x["name"], "lambda", x["text"])
        for x in plan["sterm"]:
            add(x["name"], "sterm", x["text"])

        fit_names = {x["name"] for x in fit}
        b0_train = [d["name"] for d in b[0] if d["kind"] == "train"]
        b0_chat = [d["name"] for d in b[0] if d["kind"] == "chattrain"]
        if set(b0_train) != fit_names or len(b0_train) != len(fit_names):
            raise CheckError("PLAN-b0 'train' names differ from the fit documents")
        if not set(b0_chat) <= fit_names or {d["kind"] for d in b[0]} - {"train", "chattrain"}:
            raise CheckError("PLAN-b0 holds chat copies of non-fit documents or unknown kinds")
        for n in b0_chat:
            add(f"chattrain/{n}", "chat-copy", chat(n))
        big = [d for d in b[1] if d["kind"] == "big"]
        chatbig = [d for d in b[1] if d["kind"] == "chatbig"]
        if len(big) + len(chatbig) != len(b[1]):
            raise CheckError("PLAN-b1 holds kinds other than big / chatbig")
        for d in big:
            if sha_text(d["text"]) != d["sha256"]:
                raise CheckError(f"PLAN-b1 {d['name']}: text differs from its sha256 field")
            add(d["name"], "flashnext-fit", d["text"])
        for d in chatbig:
            add(f"chatbig/{d['name']}", "chat-copy", d["text"])
        for n in [x["name"] for x in held]:
            add(f"chatheld/{n}", "chat-copy", chat(n))
        by_id = {d["doc_id"]: d for d in docs}
        fn_raw = set(b0_train) | {d2["name"] for d2 in big}
        for d in docs:                                   # the Flash-Next fit is defined by its batch plans
            if d["doc_id"] in fn_raw or d["doc_id"].startswith(("chattrain/", "chatbig/")):
                d["used_by"].add("flashnext")
        checks["flashnext_batches"] = dict(raw=len(b0_train) + len(big), chat=len(b0_chat) + len(chatbig))

        if a.chat_check:
            stored = [d for d in load_json(a.chat_check) if d.get("kind") == "chatold"]
            inputs["calibration/" + a.chat_check.name] = sha_file(a.chat_check)
            bad = [d["name"] for d in stored if d["name"] not in idx or d["text"] != chat(d["name"])]
            if bad or not stored:
                raise CheckError(f"chat-copy rebuild differs from {len(bad)} stored copies (of {len(stored)})")
            checks["chat_copy_rebuild"] = dict(stored_copies=len(stored), equal=len(stored))

        recs = []
        for kind, specs in (("fit", a.fit_record), ("held", a.held_record), ("sterm", a.sterm_record),
                            ("table", a.table_record)):
            for spec in specs:
                g, p = kv(spec, f"--{kind}-record")
                if g not in GROUPS:
                    raise CheckError(f"unknown user {g}; one of {GROUPS}")
                names, count = record_names(p, kind)
                inputs[f"records/{kind}/{g}/{p.name}"] = sha_file(p)
                recs.append(dict(kind=kind, group=g, file=p.name, count=count, named=names is not None))
                if kind == "fit":
                    if count != len(fit) or (names is not None and set(names) != fit_names):
                        raise CheckError(f"{p.name}: fit record lists {count} documents, not the {len(fit)} fit documents")
                    for n in fit_names:
                        by_id[n]["used_by"].add(g)
                    continue
                for n in names:
                    if n not in by_id:
                        raise CheckError(f"{p.name}: names {n}, which is not a calibration document")
                    by_id[n]["used_by"].add(g)
                if kind == "table":
                    if set(names) != {x["name"] for x in train_all}:
                        raise CheckError(f"{p.name}: the table's documents are not the plan-train 'calib' set")
        checks["records"] = recs
        empty = [d["doc_id"] for d in docs if not d["used_by"]]
        if empty:
            raise CheckError(f"{len(empty)} calibration documents have no user in the given records")

    for spec in a.extra_calib:
        role, folder = kv(spec, "--extra-calib")
        files = sorted(folder.glob("*.txt"))
        if not files:
            raise CheckError(f"--extra-calib {role}: no .txt documents")
        for p in files:
            add(p.stem, role, p.read_text(encoding="utf-8"))
        inputs[f"calibration/{role}/*.txt"] = sha_bytes("".join(f"{sha_file(p)}  {p.name}\n" for p in files).encode())
    if not docs:
        raise CheckError("no calibration documents: give --plans or --extra-calib")
    return docs, checks


# ---- token-id decoding (GGUF vocabulary) -------------------------------------------------------------------
def _gpt2_byte_of():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for x in range(256):
        if x not in bs:
            bs.append(x)
            cs.append(256 + n)
            n += 1
    return {chr(c): x for x, c in zip(bs, cs)}


def gguf_vocab(path):
    """(token piece bytes list, vocab fingerprint): tokenizer.ggml.tokens / token_type of a GGUF file."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    want, got = {"tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.model"}, {}
    with open(path, "rb") as f:
        rd = f.read

        def string():
            return rd(struct.unpack("<Q", rd(8))[0]).decode("utf-8")

        def value(t, keep):
            if t == 8:
                return string()
            if t == 9:
                et, n = struct.unpack("<IQ", rd(12))
                if et == 8:
                    vals = [string() for _ in range(n)]
                    return vals if keep else None
                raw = rd(sizes[et] * n)
                return list(struct.unpack(f"<{n}{'i' if et == 5 else 'B' if et in (0, 7) else 'I'}", raw)) \
                    if keep and et in (0, 4, 5, 7) else None
            return rd(sizes[t])

        if rd(4) != b"GGUF":
            raise CheckError(f"{Path(path).name} is not a GGUF file")
        _version, _tensors, n_kv = struct.unpack("<IQQ", rd(20))
        for _ in range(n_kv):
            key = string()
            t = struct.unpack("<I", rd(4))[0]
            v = value(t, key in want)
            if key in want:
                got[key] = v
            if len(got) == len(want):
                break
    if set(got) != want or got["tokenizer.ggml.model"] != "gpt2":
        raise CheckError(f"{Path(path).name}: no byte-level BPE vocabulary")
    byte_of = _gpt2_byte_of()
    pieces = []
    for tok, typ in zip(got["tokenizer.ggml.tokens"], got["tokenizer.ggml.token_type"]):
        if typ == 1:                                   # normal: byte-level characters back to bytes
            pieces.append(bytes(byte_of[c] for c in tok) if all(c in byte_of for c in tok) else tok.encode())
        else:                                          # control / user-defined: the token text itself
            pieces.append(tok.encode("utf-8"))
    finger = sha_bytes(b"".join(t.encode("utf-8") + b"\0" for t in got["tokenizer.ggml.tokens"]))
    return pieces, finger


def decode(pieces, ids):
    return b"".join(pieces[i] for i in ids).decode("utf-8", errors="replace")


# ---- evaluation side ---------------------------------------------------------------------------------------
def files_of(path):
    if path.is_dir():
        return path, sorted((p for p in path.rglob("*") if p.is_file() and p.suffix in EXTS), key=lambda p: p.as_posix())
    if path.is_file():
        return path.parent, [path]
    raise CheckError(f"missing evaluation input: {path.name}")


def text_items(path):
    """(item id, [texts], file rel) per item of a text evaluation input."""
    root, files = files_of(path)
    for f in files:
        rel = f.relative_to(root).as_posix()
        body = f.read_bytes().decode("utf-8", errors="replace")
        pre = rel + ":" if path.is_dir() else ""
        if f.suffix == ".txt":
            yield rel, [body], rel
        elif f.suffix == ".jsonl":
            for row, line in enumerate(body.split("\n")):      # not splitlines(): JSON strings may hold U+2028
                if line.strip():
                    rec = json.loads(line)
                    iid = (rec.get("_id") or rec.get("id")) if isinstance(rec, dict) else None
                    yield pre + str(iid if iid is not None else row), list(strings(rec)), rel
        else:
            yield rel, list(strings(json.loads(body))), rel


def id_items(path, pieces):
    """(item id, [decoded text], file rel, source item, n tokens) per 'ids' array of a token-id input."""
    root, files = files_of(path)

    def walk(o, jp):
        if isinstance(o, dict):
            if isinstance(o.get("ids"), list) and o["ids"] and all(isinstance(x, int) for x in o["ids"]):
                yield jp, o
            for k, v in o.items():
                if k != "ids":
                    yield from walk(v, f"{jp}.{k}" if jp else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from walk(v, f"{jp}[{i}]")

    for f in files:
        rel = f.relative_to(root).as_posix()
        for jp, o in walk(json.loads(f.read_text(encoding="utf-8")), ""):
            src = next((str(o[k]) for k in ("id", "name") if LBV2_ID.match(str(o.get(k, "")))), None)
            yield (rel + (f"#{jp}" if jp else "")), [decode(pieces, o["ids"])], rel, src, len(o["ids"])


# ---- the check -----------------------------------------------------------------------------------------------
def longest_run(cw, cpos, ew, epos_of):
    """Longest common word run of a calibration text (cw, 13-gram positions cpos) and an evaluation text (ew,
    shared 13-gram positions epos_of); every run of >= 13 words contains a shared 13-shingle anchor."""
    best, at, combos, capped = 0, None, 0, False
    for h, eps in epos_of.items():
        for i in cpos.get(h, ()):
            for j in eps:
                combos += 1
                if combos > MAX_COMBOS:
                    capped = True
                    break
                if i and j and cw[i - 1] == ew[j - 1]:
                    continue                           # not the start of a maximal run
                k = 0
                while i + k < len(cw) and j + k < len(ew) and cw[i + k] == ew[j + k]:
                    k += 1
                if k > best:
                    best, at = k, (i, j)
    return best, at, capped


def run(a):
    t0 = time.time()
    inputs = {"tool/" + Path(__file__).name: sha_file(Path(__file__))}
    docs, checks = build_calibration(a, inputs)
    log(f"{len(docs)} calibration documents: " +
        ", ".join(f"{r} {sum(d['role'] == r for d in docs)}" for r in sorted({d['role'] for d in docs})))

    pieces = None
    if a.eval_ids:
        if not a.gguf:
            raise CheckError("--eval-ids needs --gguf")
        pieces, finger = gguf_vocab(a.gguf)
        inputs["tokenizer/" + a.gguf.name] = sha_file(a.gguf)
        checks["gguf_vocab_fingerprint"] = dict(rule="sha256 of every token string followed by one NUL byte",
                                                sha256=finger, tokens=len(pieces))
        if a.decode_probes:
            probes = [json.loads(x) for x in a.decode_probes.read_text(encoding="utf-8").splitlines() if x.strip()]
            inputs["tokenizer/" + a.decode_probes.name] = sha_file(a.decode_probes)
            ok = sum(decode(pieces, p["ids"]) == p["text"] for p in probes)
            if ok != len(probes) or not probes:
                raise ControlError(f"decode control: {ok} of {len(probes)} probes decode exactly")
            checks["decode_probes"] = dict(probes=len(probes), exact=ok)

    sets = [(n, p, "text") for n, p in map(lambda s: kv(s, "--eval"), a.eval)] + \
           [(n, p, "ids") for n, p in map(lambda s: kv(s, "--eval-ids"), a.eval_ids)]

    def items_of(path, kind):
        if kind == "text":
            for iid, texts, rel in text_items(path):
                yield iid, texts, rel, None, None
        else:
            yield from id_items(path, pieces)

    # controls: planted copies of each set's first item of >= 200 words, and seeded random words
    rng = random.Random(20261004)
    vocab = [f"zq{rng.randrange(10 ** 6):06d}w{chr(rng.randrange(97, 123))}" for _ in range(64)]
    ctrl = [dict(doc_id=f"{CONTROL}neg", role="control", used_by=set(), text=" ".join(rng.choice(vocab) for _ in range(600)),
                 kind="neg", set=None, item=None)]
    for name, path, kind in sets:
        for iid, texts, *_ in items_of(path, kind):
            t = max(texts, key=lambda s: len(s.split()))
            if len(t.split()) >= 200:
                ctrl.append(dict(doc_id=f"{CONTROL}pos-shingle/{name}", role="control", used_by=set(),
                                 text=" ".join(t.split()[:200]), kind="pos-shingle", set=name, item=iid))
                ctrl.append(dict(doc_id=f"{CONTROL}pos-sha/{name}", role="control", used_by=set(), text=t,
                                 kind="pos-sha", set=name, item=iid))
                break
        else:
            raise CheckError(f"evaluation set {name}: no item of >= 200 words for the positive controls")
    alld = docs + ctrl

    owner = {n: {} for n in NS}
    words, pos13 = [], {}
    for k, d in enumerate(alld):
        w = d["text"].split()
        words.append(w)
        for n in NS:
            o = owner[n]
            for h in set(gram_list(w, n)):
                o.setdefault(h, []).append(k)
    doc_sha = {}
    for k, d in enumerate(alld):
        doc_sha.setdefault(sha_text(d["text"]), []).append(k)
    log(f"calibration shingles: {len(owner[50]):,} distinct 50-word, {len(owner[13]):,} distinct 13-word")

    def cpos_of(k):
        if k not in pos13:
            p = {}
            for i, h in enumerate(gram_list(words[k], 13)):
                p.setdefault(h, []).append(i)
            pos13[k] = p
        return pos13[k]

    pairs = {}            # (doc index, set, item) -> record
    set_stats = {}
    for name, path, kind in sets:
        st = dict(kind="token ids decoded" if kind == "ids" else "text", items=0, texts=0, words=0,
                  tokens=0 if kind == "ids" else None, shingles={str(n): 0 for n in NS}, files={})
        sources = {}
        root, files = files_of(path)
        for f in files:
            rel = f.relative_to(root).as_posix()
            st["files"][rel] = fsha = sha_file(f)
            for k in doc_sha.get(fsha, ()) if kind == "text" else ():
                pairs.setdefault((k, name, f"file:{rel}"), dict(s50=set(), s13=set(), sha=False, best=0, at=None,
                                                                capped=False, src=None))["sha"] = True
        for iid, texts, rel, src, ntok in items_of(path, kind):
            st["items"] += 1
            if ntok is not None:
                st["tokens"] += ntok
                if src:
                    sources[iid] = src
            for ti, t in enumerate(texts):
                st["texts"] += 1
                ew = t.split()
                st["words"] += len(ew)
                g = {n: gram_list(ew, n) for n in NS}
                hit_docs = {}
                for n in NS:
                    st["shingles"][str(n)] += len(g[n])
                    for h in set(g[n]) & owner[n].keys():
                        for k in owner[n][h]:
                            hit_docs.setdefault(k, {50: set(), 13: set()})[n].add(h)
                for k in doc_sha.get(sha_text(t), ()):
                    pairs.setdefault((k, name, iid), dict(s50=set(), s13=set(), sha=False, best=0, at=None,
                                                          capped=False, src=src))["sha"] = True
                if not hit_docs:
                    continue
                epos_all = {}
                need = set().union(*(v[13] for v in hit_docs.values()))
                for j, h in enumerate(g[13]):
                    if h in need:
                        epos_all.setdefault(h, []).append(j)
                for k, s in hit_docs.items():
                    r = pairs.setdefault((k, name, iid), dict(s50=set(), s13=set(), sha=False, best=0, at=None,
                                                              capped=False, src=src))
                    r["s50"] |= s[50]
                    r["s13"] |= s[13]
                    best, at, capped = longest_run(words[k], cpos_of(k), ew, {h: epos_all[h] for h in s[13]})
                    r["capped"] |= capped
                    if best > r["best"]:
                        r["best"], r["at"] = best, (at[0], ti, at[1])
        if sources:
            st["source_items"] = sources
        set_stats[name] = st
        log(f"{name}: {st['items']} items, {st['words']:,} words scanned ({time.time() - t0:.0f}s)")

    # decode control: decoded prompts that name a source item must contain its text
    if a.decode_source and pieces is not None:
        src_name = a.decode_source
        src_path = next((p for n, p, k in sets if n == src_name and k == "text"), None)
        if src_path is None:
            raise CheckError(f"--decode-source {src_name} is not a text evaluation set")
        want = {}
        for name, path, kind in sets:
            if kind == "ids":
                for iid, texts, _rel, src, _n in items_of(path, kind):
                    if src:
                        want.setdefault(src, []).append((name, iid, set(gram_list(texts[0].split(), 50))))
        found, fracs = 0, []
        for iid, texts, *_ in items_of(src_path, "text"):
            if iid in want:
                g = set().union(*(set(gram_list(t.split(), 50)) for t in texts))
                for _name, _iid, wg in want[iid]:
                    f = len(wg & g) / max(1, len(wg))
                    fracs.append(f)
                    found += f >= 0.9
        n_want = sum(len(v) for v in want.values())
        if found != n_want or not n_want:
            raise ControlError(f"decode control: {found} of {n_want} decoded prompts contain >= 90 % of their "
                               f"50-shingles in their source item")
        checks["decode_source_control"] = dict(source_set=src_name, prompts=n_want, passed=found,
                                               min_fraction_of_50_shingles_in_source=round(min(fracs), 5))

    # controls
    cidx = {d["doc_id"]: k for k, d in enumerate(alld) if d["role"] == "control"}
    cres = {}
    for d in ctrl:
        k = cidx[d["doc_id"]]
        mine = {key: r for key, r in pairs.items() if key[0] == k}
        if d["kind"] == "neg":
            if mine:
                raise ControlError(f"negative control flagged in {len(mine)} items")
            cres["negative"] = dict(words=600, seed=20261004, flagged=False)
            continue
        own = mine.get((k, d["set"], d["item"]))
        if d["kind"] == "pos-shingle":
            if not own or not own["s50"] or not own["s13"]:
                raise ControlError(f"positive shingle control for {d['set']} not flagged on its own item")
            cres.setdefault(d["set"], {})["positive_shingle"] = dict(
                item=d["item"], words=200, shared_50=len(own["s50"]), shared_13=len(own["s13"]),
                longest_run_words=own["best"], flagged=True)
        else:
            if not own or not own["sha"]:
                raise ControlError(f"positive sha256 control for {d['set']} not flagged on its own item")
            cres.setdefault(d["set"], {})["positive_sha256"] = dict(item=d["item"], flagged=True)
    log("controls: positive shingle and sha256 flagged on their own items in every set; negative not flagged")

    # report
    out_pairs, detail, summary = [], [], {}
    for (k, name, iid), r in sorted(pairs.items(), key=lambda x: (x[0][1], alld[x[0][0]]["doc_id"], x[0][2])):
        d = alld[k]
        if d["role"] == "control":
            continue
        row = dict(calib_doc=d["doc_id"], role=d["role"], used_by=",".join(g for g in GROUPS if g in d["used_by"]),
                   eval_set=name, item=iid, longest_run_words=r["best"], shared_50=len(r["s50"]),
                   shared_13=len(r["s13"]), sha256_match=r["sha"])
        if r["src"] and r["src"] != iid:
            row["source_item"] = r["src"]
        if r["capped"]:
            row["longest_run_lower_bound"] = True
        detail.append(dict(row, at=r["at"]))
        if r["s50"] or r["sha"]:
            out_pairs.append(row)
        for n in NS:
            s = r[f"s{n}"]
            if not s and not (n == 50 and r["sha"]):
                continue
            e = summary.setdefault(name, {}).setdefault(str(n), {}).setdefault(
                d["role"], dict(docs=set(), items=set(), pairs=0, distinct_shingles=set(), longest_run_words=0))
            e["docs"].add(d["doc_id"])
            e["items"].add(iid)
            e["pairs"] += 1
            e["distinct_shingles"] |= s
            e["longest_run_words"] = max(e["longest_run_words"], r["best"])
    for name in summary:
        for n in summary[name]:
            for role, e in summary[name][n].items():
                summary[name][n][role] = dict(docs_with_overlap=len(e["docs"]), items_with_overlap=len(e["items"]),
                                              pairs=e["pairs"], distinct_shared_shingles=len(e["distinct_shingles"]),
                                              longest_run_words=e["longest_run_words"])
    roles = sorted({d["role"] for d in docs})
    for spec in a.tokenizer:
        name, p = kv(spec, "--tokenizer")
        inputs[f"tokenizer/{name}/{p.name}"] = sha_file(p)
    rep = dict(
        check="calibration documents vs evaluation sets: shared word runs and whole-text sha256",
        parameters=dict(
            shingle_words=list(NS),
            normalisation="str.split() on whitespace; case, punctuation and digits kept; shingle = n words joined "
                          "by one space (same as kva checks/t02/overlap.py); JSON/JSONL: every string value is one "
                          "text, a record is one item; .txt whole; token-id prompts decoded with the GGUF vocabulary",
            shingle_match="hash of the joined shingle; the longest shared run is measured word for word from "
                          "13-shingle anchors (exact for runs of >= 13 words)",
            sha256="UTF-8 SHA-256 of each calibration text vs each evaluation text (string value or decoded "
                   "prompt) and each evaluation file's bytes",
            shared_counts="distinct shingles shared by the calibration document and the item (all its texts)",
            seconds=round(time.time() - t0)),
        inputs=inputs,
        calibration=dict(documents=len(docs), by_role={r: sum(d["role"] == r for d in docs) for r in roles},
                         used_by={r: {g: sum(d["role"] == r and g in d["used_by"] for d in docs) for g in GROUPS}
                                  for r in roles}, checks=checks),
        evaluation=set_stats,
        controls=cres,
        overlaps_50_or_hash=out_pairs,
        summary=summary,
        verdict=dict(pairs_with_50_word_run_or_hash=len(out_pairs),
                     calibration_docs=sorted({p["calib_doc"] for p in out_pairs}),
                     by_role={r: len({p["calib_doc"] for p in out_pairs if p["role"] == r}) for r in roles}),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    if a.detail:
        a.detail.write_text(json.dumps(detail, indent=0) + "\n", encoding="utf-8")
    if a.docs_tsv:
        write_tsv(a, docs)
    log(f"wrote {a.out.name}: {len(out_pairs)} pairs with a 50-word run or a whole-text hash")
    return 1 if out_pairs else 0


def write_tsv(a, docs):
    toks = []
    for spec in a.tokenizer:
        name, p = kv(spec, "--tokenizer")
        from tokenizers import Tokenizer                     # only needed for token counts
        tk = Tokenizer.from_file(str(p))
        toks.append((name, sha_file(p), [len(e.ids) for e in tk.encode_batch([d["text"] for d in docs],
                                                                             add_special_tokens=False)]))
    head = [
        "# RidgeFill calibration documents: one row per document (chat copies included). The texts and their sources",
        "# are not released; documents are identified by their names in the fit records and by sha256 of their text.",
        "# sha256: SHA-256 of the document text encoded as UTF-8, exactly as stored in the calibration plan and passed",
        "#   to the tokenizer (chat copies: the copy as captured, chat frame included); chars: Unicode code points.",
        "# role: fit = map fit document of every reference map, the Qwen3.8-27B map and the Flash-Next map;",
        "#   flashnext-fit = Flash-Next map only; chat-copy = Flash-Next chat-formatted copy (chattrain/, chatbig/ fit;",
        "#   chatheld/ lambda choice); lambda = held-out lambda-choice document; sterm = terminal-state correction prompt;",
        "#   excluded-token-table = excluded from every map fit, still counted by the token-frequency table.",
        "# used_by: the users whose fit records name the document (reference = Llama-3.1-8B, Qwen3.5-9B, Qwen3-8B",
        "#   maps; qwen38-27b; flashnext); for excluded-token-table rows, the users of the token-frequency table.",
    ]
    for name, tsha, _ in toks:
        head.append(f"# tokens_{name}: token count of the text, no BOS, not truncated (tokenizer.json sha256 {tsha}).")
    head += [f"# {note}" for note in a.tsv_note]
    head.append("# Generated by tools/overlap_lbv2.py; overlap results in provenance/checks/calib-vs-longbench-v2.json.")
    cols = ["doc_id", "role", "used_by", "chars", "sha256"] + [f"tokens_{n}" for n, _, _ in toks]
    order = {r: i for i, r in enumerate(ROLES)}
    lines = head + ["\t".join(cols)]
    for i in sorted(range(len(docs)), key=lambda i: (order.get(docs[i]["role"], 99), docs[i]["doc_id"])):
        d = docs[i]
        lines.append("\t".join([d["doc_id"], d["role"], ",".join(g for g in GROUPS if g in d["used_by"]),
                                str(len(d["text"])), sha_text(d["text"])] + [str(t[2][i]) for t in toks]))
    a.docs_tsv.parent.mkdir(parents=True, exist_ok=True)
    a.docs_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plans", type=Path)
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--fn-plans", type=Path)
    ap.add_argument("--chat-frame", type=Path)
    ap.add_argument("--chat-check", type=Path)
    for r in ("fit", "held", "sterm", "table"):
        ap.add_argument(f"--{r}-record", action="append", default=[], metavar="USER=FILE")
    ap.add_argument("--extra-calib", action="append", default=[], metavar="ROLE=DIR")
    ap.add_argument("--eval", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--eval-ids", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--gguf", type=Path)
    ap.add_argument("--decode-probes", type=Path)
    ap.add_argument("--decode-source", metavar="NAME")
    ap.add_argument("--tokenizer", action="append", default=[], metavar="NAME=TOKENIZER_JSON")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs-tsv", type=Path)
    ap.add_argument("--tsv-note", action="append", default=[], metavar="TEXT", help="extra header line for --docs-tsv")
    ap.add_argument("--detail", type=Path, help="private per-pair detail with word offsets (do not publish)")
    a = ap.parse_args(argv)
    if not a.eval and not a.eval_ids:
        ap.error("give at least one --eval or --eval-ids")
    try:
        return run(a)
    except CheckError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
