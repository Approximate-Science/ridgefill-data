"""The multi-key NIAH variant (mk4) keeps the single-needle prompt's target, haystack and length,
adds 3 distractor needles, and points needle_start/needle_end at the target needle."""
import types

import pytest

from kva import data


class CharTokenizer:
    """One token per character; no BOS (like Qwen's raw path)."""
    bos_token_id = None

    def __call__(self, text, add_special_tokens=True):
        return types.SimpleNamespace(input_ids=[ord(c) for c in text])


@pytest.fixture
def prompts(monkeypatch):
    text = "".join(chr(97 + (i * 7) % 26) for i in range(400_000))
    monkeypatch.setattr(data, "corpus", lambda: {s: text for s in data.NIAH_SOURCES})
    tok = CharTokenizer()
    return data.niah_prompts(tok, "raw"), data.niah_prompts(tok, "raw", "mk4")


def decode(ids):
    return "".join(map(chr, ids))


def test_mk4_keeps_target_key_value_and_length(prompts):
    plain, mk4 = prompts
    assert [(p["name"], p["key"], p["value"], len(p["ids"])) for p in plain] == \
           [(p["name"], p["key"], p["value"], len(p["ids"])) for p in mk4]


def test_mk4_target_span_is_the_target_needle(prompts):
    for p in prompts[1]:
        span = decode(p["ids"][p["needle_start"]:p["needle_end"]])
        assert span == f"\nThe special magic number for {p['key']} is {p['value']}.\n"


def test_mk4_has_three_distinct_distractors_in_the_prompt(prompts):
    for p in prompts[1]:
        keys = [x["key"] for x in p["distractors"]]
        assert len(keys) == 3 and len(set(keys)) == 3 and p["key"] not in keys
        text = decode(p["ids"])
        for x in p["distractors"]:
            assert f"The special magic number for {x['key']} is {x['value']}." in text


def test_plain_variant_has_no_distractors(prompts):
    plain = prompts[0][0]
    assert "distractors" not in plain and decode(plain["ids"]).count("The special magic number for") == 2


def test_unknown_variant_fails_loud(monkeypatch):
    monkeypatch.setattr(data, "corpus", lambda: {s: "x" * 400_000 for s in data.NIAH_SOURCES})
    with pytest.raises(ValueError, match="unknown NIAH variant"):
        data.niah_prompts(CharTokenizer(), "raw", "mk9")
