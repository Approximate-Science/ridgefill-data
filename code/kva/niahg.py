"""Non-numeric NIAH (the R15-general dev set, datasets/niah-general-dev) re-rendered for this model.

plan.json stores Flash-Next token ids; the prompts are rebuilt as TEXT and re-tokenised, with the
geometry of r15/niah_general.build_prompt: the cell's K20 haystack, body = hay[: L - len(needle) -
len(question)] tokens, the needle at int(depth * len(body)). The haystack is data.niah_prompts' (the
K20 RNG replayed on the local corpus), so both NIAH sets share it. With the Flash-Next tokenizer and
the chat-v2 wrapper this rebuild reproduces plan.json's ids exactly for all 300 prompts
(haystacks.json's rust char offsets are 270 higher than the replay's, the text is the same).
Formats:
  chat  the plan's question (ending in an answer-format instruction) in the model's chat template
        (as data.niah_prompts 'chat');
  raw   the question's first sentence, then the needle's lead-in up to the value ("... The special
        password for KEY is"), as K20's raw question.
Scoring (r15/score.py, per type): the reply is cut at the first special token / think block and
its first non-empty line taken, a restated lead-in dropped, answer and value normalised per type.
chat: exact match. raw (a completion runs on past the answer): the normalised answer must START
with the normalised value at a word boundary (hexid: the answer up to its first space, since its
normalisation drops spaces and would join the run-on text to the value).
"""
import json
import re
import unicodedata

from . import data

DIR = data.DATASETS / "niah-general-dev"
TYPES = ("name", "phrase", "hexid", "ident", "sentence")


def prompts(tok, fmt="raw", types=TYPES):
    plan = [p for p in json.loads((DIR / "plan.json").read_text()) if p["type"] in types]
    text_of = data.corpus()
    cells = {(c["length"], c["source"]): c for c in data.k20_cells(text_of)}
    if fmt == "chat":
        mark = "\x00NIAH\x00"
        pre, post = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                            add_generation_prompt=True).split(mark)
        head, tail_ids = data.encode(tok, pre), data.encode(tok, post)
    else:
        head, tail_ids = data.bos_ids(tok), []
    hay_ids = {}
    out = []
    for p in plan:
        L, src, value, needle_text = p["length"], p["source"], p["value"], p["needle_text"]
        c = cells[(L, src)]
        cell = f"{L}-{src}"
        if cell not in hay_ids:
            hay_ids[cell] = data.encode(tok, text_of[src][c["char_start"]:c["char_end"]])
            if len(hay_ids[cell]) < L:
                raise RuntimeError(f"niahg haystack {cell}: {len(hay_ids[cell])} tokens < {L}")
        lead = needle_text[:needle_text.index(value)].strip()
        q_text = p["question"] if fmt == "chat" else p["question"].split("?")[0] + "? " + lead
        enc = tok(needle_text, add_special_tokens=False, return_offsets_mapping=True)
        needle = enc["input_ids"]
        c0 = needle_text.index(value)
        c1 = c0 + len(value)
        vt = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < c1 and e > c0]
        q = data.encode(tok, q_text)
        body = hay_ids[cell][: L - len(needle) - len(q)]
        at = int(p["depth"] * len(body))
        ids = head + body[:at] + needle + body[at:] + q + tail_ids
        ns = len(head) + at
        out.append(dict(name=p["name"], type=p["type"], subtype=p["subtype"], length=L, source=src, depth=p["depth"],
                        key=p["key"], value=value, format=fmt, char_start=c["char_start"], char_end=c["char_end"],
                        needle_start=ns, needle_end=ns + len(needle), value_start=ns + vt[0], value_end=ns + vt[-1] + 1, ids=ids))
    return out


# ---- scoring (r15/score.py)
THINK_CLOSED = re.compile(r"<think>.*?</think>", re.S)
THINK_OPEN = re.compile(r"<think>.*", re.S)
SPECIAL_TXT = re.compile(r"<\|[a-z_]+\|>")
LEAD = re.compile(r"^(?:answer\s*:\s*)?(?:the\s+)?(?:special\s+)?(?:contact person|meeting place|password|identifier|"
                  r"function name|quote|magic number)(?:\s+for\s+[\w-]+)?(?:\s+mentioned in the text above)?\s+"
                  r"(?:is|was)\s*:?\s*", re.I)
QUOTES = "\"'`“”‘’«»*"
DASHES = {"–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-"}
WORD_CHARS = re.compile(r"[\w'-]")


def reply_text(text):
    """The reply before the first end-of-turn token, think blocks and other special tokens removed."""
    t = text.split("<|im_end|>")[0].split("<|endoftext|>")[0].split("<|eot_id|>")[0]
    t = THINK_OPEN.sub(" ", THINK_CLOSED.sub(" ", t))
    return SPECIAL_TXT.sub(" ", t)


def extract(text):
    """The answer: first non-empty line of the reply, lead-in and wrapping quotes/markdown dropped."""
    lines = [ln.strip() for ln in reply_text(text).splitlines() if ln.strip()]
    if not lines:
        return ""
    a = lines[0].lstrip("-•> ").strip()
    a = a.strip(QUOTES).strip()
    a = LEAD.sub("", a)
    return a.strip().strip(QUOTES).strip()


def normalize(s, typ):
    s = unicodedata.normalize("NFKC", s)
    for k, v in DASHES.items():
        s = s.replace(k, v)
    if typ == "ident":
        s = s.strip().strip(QUOTES).strip()
        s = re.sub(r"\(\)$", "", s)
        return s.rstrip(".,;:!?").strip(QUOTES)
    if typ == "hexid":
        s = s.casefold().strip().strip(QUOTES)
        return re.sub(r"\s+", "", s).rstrip(".,;:!?").strip(QUOTES)
    s = s.casefold()
    s = "".join(c if c.isalnum() or c in " -'" or c.isspace() else " " for c in s)
    s = re.sub(r"(?<![a-z0-9])['-]|['-](?![a-z0-9])", " ", s)      # keep only inner hyphens/apostrophes
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a, b, cap=3):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(text, value, typ, fmt):
    """dict(correct, form, answer). Miss forms: none (empty), truncated (a strict prefix of the
    value), near (edit distance <= 2 characters; raw: over the answer's first len(value) chars), wrong."""
    ans = extract(text)
    cut = re.split(r"\s", ans, maxsplit=1)[0] if (fmt != "chat" and typ == "hexid") else ans
    na, nv = normalize(cut, typ), normalize(value, typ)
    if fmt == "chat":
        ok = na == nv
    else:
        ok = na.startswith(nv) and not WORD_CHARS.match(na[len(nv):len(nv) + 1])
        na = na[:len(nv)]
    if ok:
        form = "ok"
    elif not na:
        form = "none"
    elif nv.startswith(na):
        form = "truncated"
    elif edit_distance(na, nv) <= 2:
        form = "near"
    else:
        form = "wrong"
    return dict(correct=ok, form=form, answer=ans)
