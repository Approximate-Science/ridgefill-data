"""Non-numeric NIAH (kva/niahg.py): prompt geometry on a synthetic plan (character tokenizer), the
per-type scoring rules, and (when the local data is present) the rebuild reproducing plan.json's
Flash-Next token ids exactly."""
import json
from pathlib import Path

import pytest

from kva import data, niahg


class Enc(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class CharTok:
    """One token per character; chat template = <u> ... </u><a>."""
    bos_token_id = None

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = Enc(input_ids=[ord(c) for c in text])
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "<u>" + msgs[0]["content"] + "</u><a>"


def _plan(tmp_path, monkeypatch):
    text = {s: "".join(chr(97 + (i * 7 + k) % 26) for i in range(400000)) for k, s in enumerate(data.NIAH_SOURCES)}
    monkeypatch.setattr(data, "corpus", lambda: text)
    plan = []
    for t, (needle, q, value) in {
        "name": ("\nThe special contact person for {key} is {value}.\n",
                 "\n\nWho is the special contact person for {key} mentioned in the text above? Answer with the full name only.",
                 "Ingrid Haldane"),
        "sentence": ("\nThe special quote for {key} is: \"{value}\"\n",
                     "\n\nWhat is the special quote for {key} mentioned in the text above? Answer with the quote only, word for word.",
                     "The tiny otter sang.")}.items():
        for L, src, d in ((8704, "prose", 0.3), (16384, "code", 0.9)):
            key = "amber-falcon"
            plan.append(dict(name=f"ng-{t}-{L}-{src}-{int(d * 100)}", type=t, subtype=t, length=L, source=src, depth=d,
                             key=key, value=value, needle_text=needle.format(key=key, value=value),
                             question=q.format(key=key)))
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(niahg, "DIR", tmp_path)
    return plan, text


@pytest.mark.parametrize("fmt", ["raw", "chat"])
def test_prompt_geometry(fmt, tmp_path, monkeypatch):
    plan, text = _plan(tmp_path, monkeypatch)
    ps = niahg.prompts(CharTok(), fmt)
    assert [p["name"] for p in ps] == [p["name"] for p in plan]
    cells = {(c["length"], c["source"]): c for c in data.k20_cells(text)}
    for p, src in zip(ps, plan):
        s = "".join(map(chr, p["ids"]))
        assert s[p["needle_start"]:p["needle_end"]] == src["needle_text"]
        assert s[p["value_start"]:p["value_end"]] == src["value"]
        c = cells[(p["length"], p["source"])]
        assert p["char_start"] == c["char_start"]
        head = "<u>" if fmt == "chat" else ""
        body_start = text[p["source"]][c["char_start"]:]
        assert s.startswith(head + body_start[:p["needle_start"] - len(head)])
        if fmt == "chat":
            assert s.endswith(src["question"] + "</u><a>") and len(s) == p["length"] + len("<u></u><a>")
        else:
            lead = src["needle_text"].strip().split(src["value"])[0].strip()
            assert s.endswith("mentioned in the text above? " + lead) and len(s) == p["length"]


def test_score_chat_exact_per_type():
    S = niahg.score
    assert S("Ingrid Haldane<|eot_id|>", "Ingrid Haldane", "name", "chat")["correct"]
    assert S("The special contact person for amber-falcon is **Ingrid Haldane**.", "Ingrid Haldane", "name", "chat")["correct"]
    assert not S("Ingrid Haldane and more", "Ingrid Haldane", "name", "chat")["correct"]
    assert S("purple elephant sunrise", "purple elephant sunrise", "phrase", "chat")["correct"]
    assert S("`3F2A-1b`.", "3f2a-1b", "hexid", "chat")["correct"]
    assert S("`fetchStaleLedger()`", "fetchStaleLedger", "ident", "chat")["correct"]
    assert not S("fetchstaleledger", "fetchStaleLedger", "ident", "chat")["correct"]
    assert S('"The tiny otter sang."', "The tiny otter sang.", "sentence", "chat")["correct"]
    assert S("<think>x</think>Ingrid Haldane", "Ingrid Haldane", "name", "chat")["correct"]
    assert S("", "Ingrid Haldane", "name", "chat")["form"] == "none"
    assert S("Ingrid", "Ingrid Haldane", "name", "chat")["form"] == "truncated"
    assert S("Ingrid Haldone", "Ingrid Haldane", "name", "chat")["form"] == "near"


def test_score_raw_prefix_at_word_boundary():
    S = niahg.score
    assert S(" Ingrid Haldane. The text goes on", "Ingrid Haldane", "name", "raw")["correct"]
    assert not S(" Ingrid Haldane-Smith.", "Ingrid Haldane", "name", "raw")["correct"]
    assert not S(" Ingrid Haldanes.", "Ingrid Haldane", "name", "raw")["correct"]
    assert S(" fetch_stale_ledger.\n\nNext", "fetch_stale_ledger", "ident", "raw")["correct"]
    assert not S(" fetch_stale_ledger_v", "fetch_stale_ledger", "ident", "raw")["correct"]
    assert S(' The tiny otter sang."\nMore', "The tiny otter sang.", "sentence", "raw")["correct"]
    assert S(" 3f2a-1b. Then", "3f2a-1b", "hexid", "raw")["correct"]
    r = S(" Ingrid Holdane. More", "Ingrid Haldane", "name", "raw")
    assert not r["correct"] and r["form"] == "near"


def test_score_raw_hexid_keeps_separator():
    """Raw hexid: normalize drops spaces, which turned "VALUE and more" into "valueand" and failed the word
    boundary. The space after the value now ends it; chat scoring is unchanged."""
    S, v = niahg.score, "a0aca968-4704-49a0-b6a3-c6016881cf8f"
    assert S(f" {v} and more", v, "hexid", "raw")["correct"]
    assert S(f" `{v}`, then", v.upper(), "hexid", "raw")["correct"]
    assert not S(f" {v}X and more", v, "hexid", "raw")["correct"]
    r = S(f" {v[:-1]} and more", v, "hexid", "raw")
    assert not r["correct"] and r["form"] == "truncated"
    assert S(f"{v[:8]} {v[8:]}", v, "hexid", "chat")["correct"]           # chat: inner spaces still dropped
    assert not S(f"{v} and more", v, "hexid", "chat")["correct"]


FLASH = data.DATASETS.parent / "data" / "models" / "tcc-qwen38-flash-next-mxfp4"


@pytest.mark.skipif(not (FLASH / "tokenizer.json").exists() or "docs" not in data.corpus(),
                    reason="needs the local Flash-Next tokenizer and docs corpus")
def test_rebuild_reproduces_plan_ids():
    tok = data.tokenizer(FLASH)
    open_, close = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    class ChatV2:
        def apply_chat_template(self, msgs, **kw):
            return open_ + msgs[0]["content"] + close

        def __call__(self, *a, **kw):
            return tok(*a, **kw)

        def __getattr__(self, k):
            return getattr(tok, k)
    mine = {p["name"]: p for p in niahg.prompts(ChatV2(), "chat")}
    plan = [p for p in json.loads((niahg.DIR / "plan.json").read_text()) if p["type"] in niahg.TYPES]
    assert len(plan) == len(mine) == 300
    for p in plan:
        m = mine[p["name"]]
        assert m["ids"] == p["ids"], p["name"]
        assert all(m[k] == p[k] for k in ("needle_start", "needle_end", "value_start", "value_end")), p["name"]
