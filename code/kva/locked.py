"""The locked test set (datasets/test-set-locked; PROTOCOL.md sections 2 and 10.2): prompts and scores for
evaluate.py's ruler, longbench and longppl sets. Finals only: kva/protocol.py refuses these sets unless the
config carries a FINAL- registration (PROTOCOL.md section 4 step 5).

ruler      ruler/<task>_<length>.jsonl, 25 rows each. Prompt = RULER's input then its answer_prefix (RULER
           call_api.py: input + answer_prefix): raw, BOS + the joined text; chat, input as the user turn and
           answer_prefix opening the assistant reply. Greedy up to RULER's tokens_to_generate per task (niah
           128, vt 30, cwe 120, fwe 50, qa 32); score = RULER's postprocess_pred, then string_match_all
           (niah, vt, cwe, fwe) or string_match_part (qa). Config ruler_max_len caps the nominal length.
           NIAH rows also record 'needles': the token spans of the needles the row scores (needle_spans), for the
           filled-region readout; the other tasks have no needle (no filled-region readout).
longbench  longbench/<task>.jsonl, 50 rows each. LongBench v1 pred.py: dataset2prompt template, middle
           truncation to the config's longbench_max_len (required: half the tokens from each end, decoded and
           joined), then the chat template except on trec, lcc and repobench-p; greedy up to dataset2maxlen;
           score = the official metric of the task (kva/metrics.py).
longppl    long-prompts/*.txt. Prompt = text[:split_char] (BOS first), continuation = text[split_char:]
           (SPLITS.tsv: the same text point for every tokenizer), each encoded alone and joined as ids;
           score = NLL of every continuation token, arm minus exact, as the grade. novel (copy-mask, as the
           grade's) is recorded as a labelled extra. Config longppl_max_len caps the nominal length.
Where: datasets/test-set-locked in this repo, or KVA_LOCKED_DIR when set (the finals image mounts the locked set
there and never bakes it in: finals/README.md); the MANIFEST / FROZEN checks below run on whichever it is.
Other sets (config 'dataset', absent = the locked set): a set of DATASETS replaces the locked set for longppl only
(datasets/fresh-long-20261008: new long documents built as the locked long prompts, scripts/build_fresh_long.py). It
lives beside the locked set (datasets/<name>, or <name> next to KVA_LOCKED_DIR), and the config must pin its
MANIFEST.tsv sha256 ('dataset_manifest_sha256'), checked with FROZEN.json and every file; any mismatch refuses by name.
Variants (v1.1): config ruler_variant "qfirst" is the question-before-document prompt of T1.4 on the NIAH subset
(kva/derived.py, checked against datasets/derived/ruler-qfirst.json). Selection (v1.1 finals): config ruler_tasks and
ruler_lengths run a declared subset of RULER's files (ruler_selection); the cap is separate (ruler_max_len).
Format (raw | chat): protocol.json niah_format of the model, else the config's. Every record carries
the task, the nominal length and the source file's sha256 from MANIFEST.tsv, checked against the file
(and MANIFEST.tsv against FROZEN.json) before anything runs. Rows over a cap are listed as skipped.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

from kva import data
from kva import metrics as MT

LOCKED = Path(os.environ["KVA_LOCKED_DIR"]) if os.environ.get("KVA_LOCKED_DIR") else data.DATASETS / "test-set-locked"
SETS = ("ruler", "longbench", "longppl")
DATASETS = {"fresh-long-20261008": ("longppl",)}      # config 'dataset' -> the sets it holds
GEN_SETS = ("ruler", "longbench")

# RULER scripts/data/synthetic/constants.py (tokens_to_generate) and scripts/eval/synthetic/constants.py
RULER_TASKS = {
    "niah_single_1": ("niah", 128, "string_match_all"),
    "niah_multikey_1": ("niah", 128, "string_match_all"),
    "niah_multivalue": ("niah", 128, "string_match_all"),
    "niah_multiquery": ("niah", 128, "string_match_all"),
    "vt": ("variable_tracking", 30, "string_match_all"),
    "cwe": ("common_words_extraction", 120, "string_match_all"),
    "fwe": ("freq_words_extraction", 50, "string_match_all"),
    "qa_1": ("qa", 32, "string_match_part"),
}
RULER_LENGTHS = (8192, 16384, 32768, 65536)


def check_max_new(max_new):
    """A config's ruler_max_new (v1.1.2): {task: tokens} that LOWERS RULER's tokens_to_generate for the named tasks
    (protocol.json ruler_max_new; kva.protocol.validate pins the values). Greedy decoding is causal, so a capped run
    is the first `tokens` of the uncapped one. Never raises a budget, never names a non-RULER task."""
    if max_new is None:
        return {}
    bad = {t: v for t, v in dict(max_new).items()
           if t not in RULER_TASKS or type(v) is not int or not 0 < v <= RULER_TASKS[t][1]}
    if bad:
        raise SystemExit(f"ruler_max_new {bad}: each cap names a RULER task and lowers its tokens_to_generate "
                         f"(1..RULER's own value {({t: RULER_TASKS[t][1] for t in bad if t in RULER_TASKS})})")
    return dict(max_new)

# LongBench v1 config/dataset2prompt.json and dataset2maxlen.json (main at 2e00731f), verbatim
LONGBENCH_PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}
LONGBENCH_MAXLEN = {"narrativeqa": 128, "qasper": 128, "hotpotqa": 32, "2wikimqa": 32, "gov_report": 512, "trec": 64,
                    "lcc": 64, "repobench-p": 64}
LONGBENCH_NO_CHAT = ("trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p")   # pred.py: no chat template


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_dir(name):
    """The directory of a non-locked set (config 'dataset'): beside the locked set. Unknown names are refused."""
    if name not in DATASETS:
        raise SystemExit(f"dataset {name!r} is not a known set {sorted(DATASETS)} (absent = the locked set)")
    return (Path(os.environ["KVA_LOCKED_DIR"]).parent / name if os.environ.get("KVA_LOCKED_DIR")
            else data.DATASETS / name)


def manifest(root=None, pin=None, name="the locked set"):
    """{relative path: dict(sha256, domain)} from MANIFEST.tsv, after checking it against FROZEN.json (and, for a
    non-locked set, against the config's pinned sha256)."""
    root = LOCKED if root is None else root
    path = root / "MANIFEST.tsv"
    if not path.is_file() or not (root / "FROZEN.json").is_file():
        raise SystemExit(f"{name}: {root} has no MANIFEST.tsv / FROZEN.json: not the set")
    frozen = json.loads((root / "FROZEN.json").read_text())["manifest_sha256"]
    got = _sha256(path)
    if got != frozen:
        raise SystemExit(f"{path} does not match FROZEN.json manifest_sha256: {name} changed")
    if pin is not None and got != pin:
        raise SystemExit(f"{name}: {path} hashes to {got}, the config pins {pin}: not the frozen set the run declares")
    rows = csv.DictReader(path.open(), delimiter="\t")
    return {r["path"]: dict(sha256=r["sha256"], domain=r["domain/task"].split("/")[0]) for r in rows}


def source(rel, man, root=None):
    """The manifest sha256 of a locked file, after checking the file still has it."""
    root = LOCKED if root is None else root
    if rel not in man:
        raise SystemExit(f"{rel} is not in the {'locked ' if root == LOCKED else ''}MANIFEST.tsv of {root}")
    if _sha256(root / rel) != man[rel]["sha256"]:
        raise SystemExit(f"{root / rel} does not match its MANIFEST.tsv sha256: the set changed")
    return man[rel]["sha256"]


def fmt_of(cfg, model, p):
    """The prompt format: protocol.json's niah_format for the model, else the config's; never both disagreeing."""
    want = p["models"].get(model, {}).get("niah_format")
    got = cfg.get("niah_format")
    if want and got and want != got:
        raise SystemExit(f"config niah_format {got!r} differs from protocol.json's {want!r} for {model}")
    if not (want or got):
        raise SystemExit(f"no niah_format for {model} in protocol.json or the config")
    return want or got


def wrap(tok, fmt, user, reply=""):
    """Ids of a prompt: raw, BOS + user + reply as one text; chat, user as the user turn of the model's
    chat template (thinking off, data.chat_pieces) and reply opening the assistant turn."""
    if fmt == "raw":
        return data.bos_ids(tok) + data.encode(tok, user + reply)
    if fmt == "chat":
        head, reply_open = data.chat_pieces(tok)
        return head + data.encode(tok, user) + reply_open + (data.encode(tok, reply) if reply else [])
    raise ValueError(f"unknown format {fmt!r}")


def _rows(rel):
    return [json.loads(line) for line in (LOCKED / rel).read_text().splitlines() if line.strip()]


class NeedleSpanError(ValueError):
    """A NIAH row whose scored values are not each found exactly once among its needles: no filled-region readout
    can be made for it, so the set refuses to load (never a guessed position)."""


def needle_spans(tok, fmt, user, reply, ids, refs, name):
    """[[start, end)] token spans, in document order, of the needles a NIAH row scores: RULER's numeric needles
    ("One of the special magic numbers for KEY is: VALUE.", kva.derived.needles) whose VALUE is one of refs, each ref
    found exactly once. The row's text is placed in ids as wrap() places it (chat: user after the template head;
    raw: user + reply after the BOS). Results carry them so the finals readout can say whether every scored needle
    lies in the filled region (paper T1.1: depth < N - T)."""
    from kva import derived
    hits = [(v, s, e) for _, v, s, e in derived.needles(user) if v in refs]
    if sorted(v for v, _, _ in hits) != sorted(refs):
        raise NeedleSpanError(f"{name}: its {len(refs)} scored values are not each found exactly once among its needles "
                              f"({len(hits)} needle(s) carry one)")
    if fmt == "chat":
        text, base = user, len(data.chat_pieces(tok)[0])
    else:
        text, base = user + reply, len(data.bos_ids(tok))
    return [list(span) for span in derived.token_spans(tok, text, [(s, e) for _, s, e in hits], ids, base)]


def ruler_selection(tasks=None, lengths=None):
    """(tasks, lengths) a config runs (config ruler_tasks / ruler_lengths: a declared subset, e.g. the finals' NIAH
    subset at 8K-32K); None = all. Unknown names are refused. Files outside the selection are not loaded at all (a
    selection, not a cap: they are not 'skipped')."""
    tasks = list(RULER_TASKS) if tasks is None else list(tasks)
    lengths = list(RULER_LENGTHS) if lengths is None else list(lengths)
    bad = [t for t in tasks if t not in RULER_TASKS] + [L for L in lengths if L not in RULER_LENGTHS]
    if bad or not tasks or not lengths:
        raise SystemExit(f"ruler_tasks / ruler_lengths: {bad or 'empty'} not RULER's tasks {list(RULER_TASKS)} / "
                         f"lengths {list(RULER_LENGTHS)}")
    return tasks, lengths


def ruler_prompts(tok, fmt, man, max_len=None, tasks=None, lengths=None, max_new=None):
    out, skipped = [], []
    tasks, lengths = ruler_selection(tasks, lengths)
    caps = check_max_new(max_new)
    for task, (kind, steps, metric) in RULER_TASKS.items():
        steps = caps.get(task, steps)
        for L in RULER_LENGTHS:
            if task not in tasks or L not in lengths:
                continue
            rel = f"ruler/{task}_{L}.jsonl"
            if max_len and L > max_len:
                skipped.append(rel)
                continue
            sha = source(rel, man)
            for i, r in enumerate(_rows(rel)):
                name = f"ruler-{task}-{L}-{i:02d}"
                ids = wrap(tok, fmt, r["input"], r["answer_prefix"])
                q = dict(name=name, set="ruler", task=task, ruler_task=kind, length=L, source=rel, source_sha256=sha,
                         row=i, index=r["index"], max_new=steps, metric=metric, refs=r["outputs"], format=fmt, ids=ids)
                if kind == "niah":
                    q["needles"] = needle_spans(tok, fmt, r["input"], r["answer_prefix"], ids, r["outputs"], name)
                out.append(q)
    return out, skipped


def middle_truncate(tok, prompt, max_len):
    """LongBench pred.py: if the prompt (tokenized with the tokenizer's default special tokens) is longer than
    max_len, keep the first and last int(max_len / 2) tokens, each half decoded without special tokens, and
    join the two texts."""
    ids = tok(prompt).input_ids
    if len(ids) <= max_len:
        return prompt, False
    half = int(max_len / 2)
    return tok.decode(ids[:half], skip_special_tokens=True) + tok.decode(ids[-half:], skip_special_tokens=True), True


def longbench_prompts(tok, fmt, man, max_len):
    if not max_len:
        raise SystemExit("longbench needs the config's longbench_max_len (LongBench's middle-truncation length)")
    out = []
    for task, template in LONGBENCH_PROMPT.items():
        rel = f"longbench/{task}.jsonl"
        sha = source(rel, man)
        for i, r in enumerate(_rows(rel)):
            if r["dataset"] != task:
                raise SystemExit(f"{rel} row {i} is from {r['dataset']!r}")
            prompt, cut = middle_truncate(tok, template.format(**r), max_len)
            ids = wrap(tok, "raw" if task in LONGBENCH_NO_CHAT else fmt, prompt)
            out.append(dict(name=f"lb-{task}-{i:02d}", set="longbench", task=task, length=r["length"], source=rel,
                            source_sha256=sha, row=i, id=r["_id"], max_new=LONGBENCH_MAXLEN[task],
                            metric=MT.LONGBENCH_METRIC[task].__name__, refs=r["answers"],
                            all_classes=r["all_classes"], truncated=cut,
                            format="raw" if task in LONGBENCH_NO_CHAT else fmt, ids=ids))
    return out, []


def splits_column(model, columns):
    """The SPLITS.tsv token-count column of this model's tokenizer (its name prefixes the model's), or None."""
    keys = [c[len("prompt_tokens_"):] for c in columns if c.startswith("prompt_tokens_")]
    hits = [k for k in keys if model.startswith(k)]
    return max(hits, key=len) if hits else None


def longppl_docs(tok, man, model, max_len=None, root=None):
    root = LOCKED if root is None else root
    rows = list(csv.DictReader((root / "long-prompts" / "SPLITS.tsv").open(), delimiter="\t"))
    col = splits_column(model, rows[0].keys()) if rows else None
    bos = data.bos_ids(tok)
    out, skipped = [], []
    for r in sorted(rows, key=lambda r: r["path"]):
        rel = r["path"]
        stem = rel.rsplit("/", 1)[-1][:-len(".txt")]
        L = int(stem.rsplit("_", 2)[-2])
        if max_len and L > max_len:
            skipped.append(rel)
            continue
        sha = source(rel, man, root)
        text = (root / rel).read_text()
        s = int(r["split_char"])
        head, cont = data.encode(tok, text[:s]), data.encode(tok, text[s:])
        if col and (len(head), len(cont)) != (int(r[f"prompt_tokens_{col}"]), int(r[f"continuation_tokens_{col}"])):
            raise SystemExit(f"{rel}: {len(head)} + {len(cont)} tokens, SPLITS.tsv has {r[f'prompt_tokens_{col}']} + "
                             f"{r[f'continuation_tokens_{col}']} for {col}: not the tokenizer the split was counted with")
        ids = bos + head + cont
        novel = data.novel_mask(ids, len(cont))
        out.append(dict(name=f"lp-{stem}", set="longppl", task=man[rel]["domain"], length=L, source=rel,
                        source_sha256=sha, split_char=s, cont_len=len(cont), novel=novel, format="raw", ids=ids))
    return out, skipped


def prompts(which, tok, cfg, model, p):
    """(prompts, skipped source files) of a locked set for this model (config 'dataset': another frozen set)."""
    name = cfg.get("dataset")
    if name is not None:
        root = dataset_dir(name)
        if which not in DATASETS[name]:
            raise SystemExit(f"dataset {name!r} holds the {'/'.join(DATASETS[name])} set only, not {which!r}")
        if not cfg.get("dataset_manifest_sha256"):
            raise SystemExit(f"dataset {name!r}: the config pins no dataset_manifest_sha256 (its MANIFEST.tsv)")
        man = manifest(root, cfg["dataset_manifest_sha256"], name)
        return longppl_docs(tok, man, model, cfg.get("longppl_max_len"), root)
    man = manifest()
    if which == "ruler":
        variant = cfg.get("ruler_variant")
        sel = dict(tasks=cfg.get("ruler_tasks"), lengths=cfg.get("ruler_lengths"), max_new=cfg.get("ruler_max_new"))
        if variant == "qfirst":                    # T1.4: question before the document (kva/derived.py)
            from kva import derived
            return derived.prompts("qfirst", tok, fmt_of(cfg, model, p), man, cfg.get("ruler_max_len"), **sel)
        if variant is not None:
            raise SystemExit(f"ruler_variant {variant!r}: the only RULER variant is 'qfirst' (T1.4)")
        return ruler_prompts(tok, fmt_of(cfg, model, p), man, cfg.get("ruler_max_len"), **sel)
    if which == "longbench":
        return longbench_prompts(tok, fmt_of(cfg, model, p), man, cfg.get("longbench_max_len"))
    if which == "longppl":
        return longppl_docs(tok, man, model, cfg.get("longppl_max_len"))
    raise ValueError(which)


def tail_of(arm, q):
    """An arm's exact tail on a longppl document: the continuation + 1 for tail_from 'continuation' (pop-native)."""
    n = len(q["ids"])
    return q["cont_len"] + 1 if arm.tail_from == "continuation" else n - arm.prefix_len(n)


def check_tails(prompts, arms):
    """longppl: every scored row (the last prompt token on) must be exact in every arm: tail > continuation. SwiftKV
    native is the released model with no exact tail by definition (its scored rows run its own late layers)."""
    bad = [(a.name, q["name"], tail_of(a, q), q["cont_len"]) for q in prompts for a in arms
           if a.fill != "swiftkv_native" and tail_of(a, q) < q["cont_len"] + 1]
    if bad:
        raise SystemExit("longppl: the continuation must lie inside every arm's exact tail (tail >= continuation + 1); "
                         f"(arm, document, tail, continuation): {bad[:5]}{' ...' if len(bad) > 5 else ''}")


def score(q, text):
    """Per arm of a generation set: the prediction as scored, its 0-1 row score and the official metric."""
    if q["set"] == "ruler":
        pred = MT.postprocess_pred(text)
        return dict(pred=pred, metric=q["metric"], score=MT.ruler_row(q["metric"], pred, q["refs"]))
    return dict(pred=text, metric=q["metric"], score=MT.longbench_row(q["task"], text, q["refs"], q["all_classes"]))


def nll(logprobs, mask=None):
    xs = [-x for i, x in enumerate(logprobs) if mask is None or mask[i]]
    return sum(xs) / len(xs)
