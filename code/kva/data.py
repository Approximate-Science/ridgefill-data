"""Prompt sets, re-tokenised per model.

grade   the 23 graded prompts of the kva-offline programme (plan-grade.json): the same text,
        scored on the last 512 prompt tokens, novel tokens primary (not inside an 8-token span
        that occurs earlier in the prompt); the short-context reference is text[short_start:].
        Format raw (the text) or chat (the scored tokens as the assistant's reply; grade_prompts).
niah    the K20 generator (kva-offline harness gpu_offline.niah_prompts) with this model's
        tokenizer: the same RNG draws, haystack slices, keys, values and depths.
calib   projector fit: plan-train.json's 'calib' documents only (the corpus slices are left out:
        some overlap the NIAH haystack regions); lambda selection: plan-held.json (docs >= 1024
        tokens, minus one documentation text that overlaps the eval prompts: HELD_EXCLUDE_SHA256); terminal-state fit:
        plan-sterm.json.
Llama prompts start with <|begin_of_text|>; Qwen3.5 has no BOS. Prompts are capped at MAX_LEN
tokens by dropping tokens from the left (after the BOS), which keeps the scored tokens. grade_prompts(drop_bos=True)
(SINK1, off-distribution) drops that BOS afterwards.
"""
import hashlib
import json
import os
import random
import re
from pathlib import Path

DATASETS = Path(__file__).resolve().parents[1] / "datasets"
# plan-*.json and swap_summary.json. The shared copies are redacted (docs domain, rust7-32k withheld); the
# original files, when present locally in data/inputs-private, are used instead (same names, sha256 in SHA256SUMS).
_PRIVATE = Path(os.environ.get("KVA_DATA", DATASETS.parent / "data")) / "inputs-private"
INPUTS = _PRIVATE if (_PRIVATE / "plan-grade.json").exists() else DATASETS / "graded-and-calibration-sets"
MAX_LEN = 32768
SCORED = 512
COPY_SPAN = 8


def tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_path))


def bos_ids(tok):
    """[BOS] for tokenizers whose default encoding adds one (Llama), else []."""
    ids = tok("a").input_ids
    return ids[:1] if tok.bos_token_id is not None and ids and ids[0] == tok.bos_token_id else []


def encode(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def cap(ids, bos, max_len=MAX_LEN):
    if len(ids) <= max_len:
        return ids
    return bos + ids[len(ids) - (max_len - len(bos)):]


def copy_mask(ids, span=COPY_SPAN):
    """True where the token lies in a span of `span` tokens that also occurs verbatim starting
    earlier (kva-offline copy_tokens.copy_mask)."""
    first = {}
    mask = [False] * len(ids)
    for s in range(len(ids) - span + 1):
        gram = tuple(ids[s:s + span])
        if gram in first:
            for i in range(s, s + span):
                mask[i] = True
        else:
            first[gram] = s
    return mask


def novel_mask(ids, scored, span=COPY_SPAN):
    """The novel-token mask of the last `scored` ids, exactly as the grade and the long-prompt NLL score it (paper
    T1.3's "novel tokens"): a scored token is NOT novel if it lies inside any window of `span` (8) consecutive token
    ids that also occurs verbatim, as ids, starting at an earlier position of `ids` (copy_mask: the window's first
    occurrence is not marked; overlapping repeated windows mark the union of their tokens); every other scored token
    is novel. It is a token-id rule, not a string rule: a token whose text occurs earlier is still novel unless a
    whole 8-token window around it repeats, and the same text tokenised differently does not repeat. Callers pass the
    ids the rule runs over: grade_prompts the prompt without its first id (ids[1:]: the BOS on Llama, a text or
    chat-header token on Qwen3.5), locked.longppl_docs the whole prompt."""
    return [not m for m in copy_mask(ids, span)[-scored:]]


CHAT_ASK = "\n\nContinue the text above from exactly where it stops."   # the Flash-Next chat grade's (K23)


def chat_pieces(tok):
    """Ids of the model's chat template around one user message: (user open, assistant open), the
    assistant open ending where the reply starts. Thinking off where the template has the switch
    (Qwen3.5: an empty think block, as the Flash-Next chat grade); Llama-3.1 ignores it."""
    mark = "\x00DOC\x00"
    rendered = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    pre, post = rendered.split(mark)
    return encode(tok, pre), encode(tok, post)


def grade_prompts(tok, fmt="raw", drop_bos=False):
    """fmt 'raw': the document text (after BOS). fmt 'chat' (the Flash-Next chat grade, K23 with fix
    b4170e2): the document minus its last SCORED tokens plus CHAT_ASK as the user message, then the
    assistant header, then those SCORED tokens as the assistant's reply, so the scored tokens are
    the reply (never user-turn text: there the model predicts end-of-turn at every token). The pieces
    are joined as ids, so the reply has the raw grade's document tokens. Capping drops document
    tokens from the left of the user message.
    drop_bos (SINK1: is Llama's BOS attention sink what makes its keys matter?): the same prompts and short
    references without their first token, which must be the BOS; scored tokens and novel mask unchanged."""
    bos = bos_ids(tok)
    if fmt == "raw":
        head = bos

        def wrap(text):
            return bos + encode(tok, text)
    elif fmt == "chat":
        head, reply_open = chat_pieces(tok)
        ask = encode(tok, CHAT_ASK)

        def wrap(text):
            doc = encode(tok, text)
            return head + doc[:-SCORED] + ask + reply_open + doc[-SCORED:]
    else:
        raise ValueError(f"unknown grade format {fmt!r}")
    plan = json.loads((INPUTS / "plan-grade.json").read_text())
    helps = {r["name"]: r["helps"] for r in json.loads((INPUTS / "swap_summary.json").read_text())["rows"]}
    out = []
    for p in plan:
        full = wrap(p["text"])
        ids = cap(full, head)
        short = wrap(p["text"][p["short_start"]:])
        novel = novel_mask(ids[1:], SCORED)
        rec = dict(name=p["name"], kind=p["kind"], ids=ids, full_len=len(full), truncated=len(full) > len(ids),
                   short_ids=short, short_tail_match=short[-SCORED:] == ids[-SCORED:], novel=novel,
                   helps16=helps[p["name"]], format=fmt)
        if drop_bos:
            if not bos or ids[0] != bos[0] or short[0] != bos[0]:
                raise ValueError(f"drop_bos: prompt {p['name']} does not start with a BOS")
            rec.update(ids=ids[1:], short_ids=short[1:], bos_dropped=True)
        out.append(rec)
    return out


# ---- K20 NIAH (generator logic of kva-offline gpu_offline.niah_prompts, this model's tokenizer)
NIAH_LENGTHS = (8704, 16384, 32768)
NIAH_DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
NIAH_SOURCES = ("prose", "docs", "code", "rust")
NIAH_KEYS = ["amber-falcon", "quiet-harbor", "silver-lantern", "crimson-meadow", "paper-comet", "iron-willow",
             "velvet-canyon", "frozen-orchid", "hollow-summit", "golden-reef", "misty-anchor", "rapid-thistle"]


def corpus():
    # docs.txt (private documentation text) was redacted from the shared datasets; it is read from the local data folder
    local = Path(os.environ.get("KVA_DATA", DATASETS.parent / "data")) / "corpus"
    files = sorted((DATASETS / "corpus").glob("*.txt")) + sorted(local.glob("*.txt"))
    return {p.stem: p.read_text() for p in files}


def k20_cells(text_of):
    """The K20 generator's RNG draws: per (length, source) cell the haystack's character range in the
    corpus and, per depth, (depth, key, value)."""
    rng = random.Random(20260930)
    cells = []
    for L in NIAH_LENGTHS:
        for si, src in enumerate(NIAH_SOURCES):
            start = max(0, len(text_of[src]) - L * 7 - si * 1000 - rng.randrange(20000))
            needles = [(d, rng.choice(NIAH_KEYS), str(rng.randrange(1_000_000, 10_000_000))) for d in NIAH_DEPTHS]
            cells.append(dict(length=L, source=src, char_start=start, char_end=start + L * 7, needles=needles))
    return cells


def niah_question(fmt, key):
    """The K20 question about key, as it ends the prompt (chat: before the assistant header)."""
    if fmt == "chat":
        return (f"\n\nWhat is the special magic number for {key} mentioned in the text above? "
                f"Answer with the 7-digit number only.")
    return (f"\n\nWhat is the special magic number for {key} mentioned in the text above? "
            f"The special magic number for {key} is")


def niah_needle(key, value):
    """The needle sentence (its ids are spliced into the haystack as encoded alone)."""
    return f"\nThe special magic number for {key} is {value}.\n"


def niah_prompts(tok, fmt="raw", variant=None):
    """fmt 'raw': the K20 prompt (haystack, needle, question ending in 'The special magic number
    for KEY is'). fmt 'chat' (instruct models without a raw-completion habit, e.g. Llama-3.1-
    Instruct): the same haystack, needle and depth as the user turn of the model's chat template,
    the question ending in a short answer-format instruction, then the assistant header. The RNG
    draws (haystack start, key, value) are the same in both formats.
    variant 'mk4': the same prompts plus 3 distractor needles (same sentence, other keys and values) at
    3 of the other 4 depths, drawn from a per-prompt RNG so the target needle, haystack and question are
    unchanged; needle_start/needle_end point at the target needle."""
    text_of = corpus()
    if fmt == "chat":
        mark = "\x00NIAH\x00"
        rendered = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                           add_generation_prompt=True)
        pre, post = rendered.split(mark)
        head, tail_ids = encode(tok, pre), encode(tok, post)
    else:
        head, tail_ids = bos_ids(tok), []
    prompts = []
    for c in k20_cells(text_of):
        L, src, start = c["length"], c["source"], c["char_start"]
        hay = encode(tok, text_of[src][start:c["char_end"]])
        if len(hay) < L:
            raise RuntimeError(f"NIAH haystack {src}: {len(hay)} tokens < {L}")
        for d, key, value in c["needles"]:
            name = f"niah-{L}-{src}-{int(d * 100)}"
            needles = [(d, key, value, True)]
            if variant == "mk4":
                r2 = random.Random(f"{name}-mk4")
                others = r2.sample([k for k in NIAH_KEYS if k != key], 3)
                depths = r2.sample([x for x in NIAH_DEPTHS if x != d], 3)
                needles += [(dd, kk, str(r2.randrange(1_000_000, 10_000_000)), False) for dd, kk in zip(depths, others)]
            elif variant is not None:
                raise ValueError(f"unknown NIAH variant {variant!r}")
            coded = [(dd, encode(tok, niah_needle(kk, vv)), tgt) for dd, kk, vv, tgt in needles]
            q = encode(tok, niah_question(fmt, key))
            body = hay[: L - sum(len(n) for _, n, _ in coded) - len(q)]
            ids, prev = list(head), 0
            for dd, n, tgt in sorted(coded, key=lambda x: x[0]):
                at = int(dd * len(body))
                ids += body[prev:at]
                if tgt:
                    needle_start, needle_end = len(ids), len(ids) + len(n)
                ids += n
                prev = at
            ids += body[prev:] + q + tail_ids
            rec = dict(name=name, length=L, source=src, depth=d, key=key, value=value, format=fmt,
                       char_start=start, char_end=start + L * 7, needle_start=needle_start,
                       needle_end=needle_end, ids=ids)
            if variant:
                rec["variant"] = variant
                rec["distractors"] = [dict(depth=dd, key=kk, value=vv) for dd, kk, vv, tgt in needles if not tgt]
            prompts.append(rec)
    return prompts


def niah_correct(text, value):
    """Exact match of the first standalone 7-digit number after removing any think block."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    m = re.search(r"(?<!\d)\d{7}(?!\d)", text)
    return bool(m and m.group(0) == value)


def eos_ids(model_path, tok):
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for name in ("generation_config.json", "config.json"):
        f = Path(model_path) / name
        if f.exists():
            c = json.loads(f.read_text())
            e = c.get("eos_token_id", c.get("text_config", {}).get("eos_token_id"))
            ids |= set(e) if isinstance(e, list) else ({e} if e is not None else set())
    return ids


# ---- calibration sets
# held documents left out of lambda selection, by the sha256 of their name (the name itself is not public): one
# documentation text that shares ~4.4K token 50-grams with the grade/NIAH prompts
HELD_EXCLUDE_SHA256 = {"84f57b5fa277a7ed4af11992762af87605d6b02b1d71b0589743ac4a29d9c73d"}


def name_sha256(name):
    return hashlib.sha256(name.encode()).hexdigest()


CALIB_EXCLUDE_FILE = DATASETS / "calibration-exclude.json"


def calib_exclude(which):
    """Calibration documents excluded from the fit lists since protocol 1.1 (2026-10-04, T0.2): the names listed in
    datasets/calibration-exclude.json for 'train' / 'held' / 'sterm'. They share 50-word shingles with the raw
    LongBench pool copy inside the locked-set directory (the frozen manifest itself is disjoint). The file is the
    single source of truth; a missing file is an error, not an empty list."""
    if not CALIB_EXCLUDE_FILE.exists():
        raise FileNotFoundError(f"{CALIB_EXCLUDE_FILE}: the calibration exclusion list is required (protocol 1.1)")
    return set(json.loads(CALIB_EXCLUDE_FILE.read_text()).get(which, []))


def calib_docs(which):
    if which == "train":
        ex = calib_exclude("train")
        return [dict(name=x["name"], text=x["text"]) for x in json.loads((INPUTS / "plan-train.json").read_text())
                if x["kind"] == "calib" and x["name"] not in ex]
    if which == "held":
        ex = calib_exclude("held")
        return [dict(name=x["name"], text=x["text"]) for x in json.loads((INPUTS / "plan-held.json").read_text())
                if x["tokens"] >= 1024 and x["name"] not in ex and name_sha256(x["name"]) not in HELD_EXCLUDE_SHA256]
    if which == "sterm":
        ex = calib_exclude("sterm")
        return [dict(name=x["name"], text=x["text"]) for x in json.loads((INPUTS / "plan-sterm.json").read_text())
                if x["name"] not in ex]
    raise ValueError(which)


def calib_ids(tok, which):
    bos = bos_ids(tok)
    docs = calib_docs(which)
    if os.environ.get("KVA_LIMIT_DOCS"):              # dry runs only
        docs = docs[:int(os.environ["KVA_LIMIT_DOCS"])]
    return [dict(name=d["name"], ids=cap(bos + encode(tok, d["text"]), bos)) for d in docs]


# ---- L4: code-domain adaptation prompts (corpus slices disjoint from every eval prompt)
ADAPT_CHARS = 56000
ADAPT_STARTS = {   # chars; outside the grade slices (located in the corpus) and the NIAH tail regions
    "code": [140000, 184000, 375000, 432000, 590000, 650000, 710000, 770000, 830000, 890000],
    "rust": [100000, 350000, 900000, 1100000, 1600000, 2000000, 2400000, 2800000, 3200000, 3600000,
             4400000, 4800000, 5200000, 5600000],
}
GRAM = 50


def _grams(ids):
    return {tuple(ids[i:i + GRAM]) for i in range(len(ids) - GRAM + 1)}


def adapt_docs(tok):
    """The L4 adaptation prompts: ADAPT_CHARS-char slices of code.txt / rust.txt at ADAPT_STARTS,
    kept only if they share no token 50-gram with any grade or NIAH prompt (the corpus repeats
    text internally, so position alone is not enough). Returns (docs, dropped names)."""
    text_of = corpus()
    bos = bos_ids(tok)
    plan = json.loads((INPUTS / "plan-grade.json").read_text())
    taken = [(p["kind"], text_of[p["kind"]].find(p["text"]), len(p["text"])) for p in plan]
    eval_grams = set()
    for p in grade_prompts(tok):
        eval_grams |= _grams(p["ids"])
    for p in niah_prompts(tok):
        eval_grams |= _grams(p["ids"])
    docs, dropped = [], []
    for src, starts in ADAPT_STARTS.items():
        n = len(text_of[src])
        for s in starts:
            e = s + ADAPT_CHARS
            assert e < n - 260000, (src, s)                         # NIAH haystacks live in the last ~252K chars
            assert all(k != src or e <= a or s >= a + ln for k, a, ln in taken), (src, s)
            ids = bos + encode(tok, text_of[src][s:e])
            name = f"adapt-{src}-{s}"
            if _grams(ids) & eval_grams:
                dropped.append(name)
                continue
            docs.append(dict(name=name, source=f"corpus/{src}.txt[{s}:{e}]", ids=ids))
    return docs, dropped
