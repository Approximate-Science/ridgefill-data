"""Multi-turn set and deferred repair (kva/multiturn.py), fp32 CPU on the tiny random models:
- the turn decode is kva.greedy's, bitwise;
- a later turn prefilled on a carried exact cache equals one exact forward over the whole history;
- repair@1 (every cached row: prefix, turn-1 tail with its question, answer) gives the late-layer cache (attention
  K/V and GDN states) of an exact forward over that history, and exact later turns;
- repair@0 is no repair; repair@r < 1 makes K/V rows < R exact and leaves every later row (rest of the prefix,
  tail, answer) stale, the GDN states as they were and the forked-from cache untouched;
- repair-q of rows S equals the prefix cache finish(exact_rows=S) gives (bitwise), of every prefix row the exact
  prefix; rows outside S, the tail and the GDN states stay stale;
- follow-up chat turns render as the model's chat template; distractor needle spans;
- later turns free each turn-1 K/V buffer they regrow (the Llama 32K OOM);
- the draft configs are refused by the v1.0 protocol; an end-to-end CPU run of the evaluator's multiturn set."""
import json
import os
import subprocess
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch

from kva import data
from kva import model as M
from kva import multiturn as MT
from kva import protocol
from kva.kva import Arm, Projector, finish, greedy, run_lower

ROOT = Path(__file__).resolve().parents[1]
SPLIT, TAIL, N = 3, 60, 200
TOL = dict(atol=1e-4, rtol=1e-4)


def _ids(n, seed=0):
    return torch.randint(0, 300, (n,), generator=torch.Generator().manual_seed(seed))


def _proj(m, split, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = m.hidden
    return Projector(split, {i: torch.eye(d) + 0.1 * torch.randn(d, d, generator=g) for i in range(split, m.n_layers)},
                     {i: 0.1 * torch.randn(d, generator=g) for i in range(split, m.n_layers)})


@pytest.fixture(params=["llama", "qwen"])
def tiny(request, tiny_llama, tiny_qwen):
    return M.load((tiny_llama if request.param == "llama" else tiny_qwen)[0], dtype=torch.float32)


def _turn1(m, ids, arm, steps=6):
    """Turn 1 of an arm: (cache, history ids in the cache, layer-split inputs of those rows, P, unfed token)."""
    hs, lower = run_lower(m, ids, {arm.split})
    logits, cache, P = finish(m, ids, hs[arm.split], lower, arm, 1)
    gen, _, hb = MT.decode(m, cache, logits, ids.shape[0], steps, (), arm.split)
    hist = torch.cat((ids, torch.tensor(gen[:-1])))
    return cache, hist, torch.cat((hs[arm.split], hb)), P, gen[-1]


def _kva(m):
    return Arm("kva", split=SPLIT, tail=TAIL, fill="proj", projector=_proj(m, SPLIT))


def _close(a, b):
    torch.testing.assert_close(a, b, **TOL)


def _late_equal(m, c, ref, split):
    for i in range(split, m.n_layers):
        if m.layer_types[i] == "attn":
            _close(c[i]["k"], ref[i]["k"])
            _close(c[i]["v"], ref[i]["v"])
        else:
            _close(c[i]["rec"], ref[i]["rec"])
            _close(c[i]["conv"], ref[i]["conv"])


def test_decode_is_greedy_bitwise(tiny):
    m = tiny
    ids = _ids(N, 1)
    arm = _kva(m)
    hs, lower = run_lower(m, ids, {SPLIT})
    outs = []
    for fn in ("greedy", "decode"):
        logits, cache, _ = finish(m, ids, hs[SPLIT], lower, arm, 1)
        outs.append(greedy(m, ids, cache, logits, 8, arm)[0] if fn == "greedy"
                    else MT.decode(m, cache, logits, N, 8)[0])
    assert outs[0] == outs[1] and len(outs[0]) == 8


def test_carried_turn_equals_exact_history(tiny):
    m = tiny
    ids = _ids(N, 2)
    cache, hist, _, P, last = _turn1(m, ids, Arm("exact", split=SPLIT))
    assert P == 0
    new = [last] + _ids(9, 3).tolist()
    lg, _ = MT.extend(m, cache, new, hist.shape[0])
    full = torch.cat((hist, torch.tensor(new)))
    ref, ref_cache = m.forward_exact(full, keep=torch.tensor([full.shape[0] - 1]))
    _close(lg, ref)
    _late_equal(m, cache, ref_cache, 0)


def test_full_repair_is_exact(tiny):
    m = tiny
    ids = _ids(N, 4)
    cache, hist, hb, P, last = _turn1(m, ids, _kva(m))
    H = hist.shape[0]
    _, ref_cache = m.forward_exact(hist)
    with pytest.raises(AssertionError):                     # the approximated cache is not exact
        _late_equal(m, cache, ref_cache, SPLIT)
    R = MT.repair_rows(1.0, P, H)
    assert P == N - TAIL and R == H > N                    # the turn-1 tail and answer rows are repaired too
    c = [dict(x) for x in cache]
    MT.repair(m, c, hb, SPLIT, R)
    _late_equal(m, c, ref_cache, SPLIT)
    new = [last] + _ids(7, 5).tolist()
    lg, _ = MT.extend(m, c, new, H)
    full = torch.cat((hist, torch.tensor(new)))
    ref, _ = m.forward_exact(full, keep=torch.tensor([full.shape[0] - 1]))
    _close(lg, ref)


def test_zero_repair_is_no_repair(tiny):
    m = tiny
    cache, hist, hb, P, last = _turn1(m, _ids(N, 6), _kva(m))
    assert MT.repair_rows(0.0, P, hist.shape[0]) == 0
    c = [dict(x) for x in cache]
    MT.repair(m, c, hb, SPLIT, 0)
    assert all(a.keys() == b.keys() and all(a[k] is b[k] for k in a) for a, b in zip(c, cache))
    new = [last] + _ids(5, 7).tolist()
    lg0, _ = MT.extend(m, c, new, hist.shape[0])
    lg1, _ = MT.extend(m, [dict(x) for x in cache], new, hist.shape[0])
    assert torch.equal(lg0, lg1)


def test_partial_repair_touches_rows_below_r_only(tiny):
    m = tiny
    cache, hist, hb, P, _ = _turn1(m, _ids(N, 8), _kva(m))
    H = hist.shape[0]
    R = MT.repair_rows(0.5, P, H)
    assert R == round(0.5 * P) < P
    # the valid views only: a K/V buffer's spare rows are uninitialised (may hold NaN)
    before = [{k: c[k].clone() for k in ("k", "v", "rec", "conv") if k in c} for c in cache]
    c = [dict(x) for x in cache]
    MT.repair(m, c, hb, SPLIT, R)
    _, ref_cache = m.forward_exact(hist)
    for i in range(m.n_layers):
        for k, v in before[i].items():                                     # the forked-from cache is untouched
            assert torch.equal(cache[i][k], v), (i, k)
        if i < SPLIT:
            assert c[i] == cache[i]
        elif m.layer_types[i] == "attn":
            for k in ("k", "v"):
                assert c[i][k].shape == cache[i][k].shape
                assert torch.equal(c[i][k][:, R:], cache[i][k][:, R:])                     # stale from R on
                _close(c[i][k][:, :R], ref_cache[i][k][:, :R])
                assert not torch.allclose(cache[i][k][:, :R], ref_cache[i][k][:, :R], **TOL)
                # the tail and answer: stale above the first late layer (there they read only their exact
                # layer-split inputs)
                assert torch.allclose(c[i][k][:, P:], ref_cache[i][k][:, P:], **TOL) == (i == SPLIT)
        else:                                                              # GDN state: repaired only at R = H
            assert c[i]["rec"] is cache[i]["rec"] and c[i]["conv"] is cache[i]["conv"]


def test_repair_rows():
    assert [MT.repair_rows(r, 100, 180) for r in (0, 0.25, 0.5, 0.999, 1.0)] == [0, 25, 50, 100, 180]
    assert MT.repair_rows(1.0, 0, 50) == 0


def test_repair_q_is_the_bypass_computation_on_the_prefix(tiny):
    m = tiny
    ids = _ids(N, 9)
    arm = _kva(m)
    cache, hist, hb, P, _ = _turn1(m, ids, arm)
    picked = [0, 5, 6, 30, 31, 32, 90, P - 1]
    c = [dict(x) for x in cache]
    MT.repair_selected(m, c, arm, hb[:P], picked)
    hs, lower = run_lower(m, ids, {SPLIT})
    _, by_rows, _ = finish(m, ids, hs[SPLIT], lower, arm, 1, exact_rows=picked)
    other = [j for j in range(P) if j not in picked]
    for i in range(SPLIT, m.n_layers):
        if m.layer_types[i] == "attn":
            for k in ("k", "v"):
                assert torch.equal(c[i][k][:, picked], by_rows[i][k][:, picked])
                assert torch.equal(c[i][k][:, other], cache[i][k][:, other])
                assert torch.equal(c[i][k][:, P:], cache[i][k][:, P:])            # tail and answer: stale
            assert not torch.equal(c[i]["k"][:, picked], cache[i]["k"][:, picked])
        else:
            assert c[i]["rec"] is cache[i]["rec"]
    full = [dict(x) for x in cache]
    MT.repair_selected(m, full, arm, hb[:P], list(range(P)))
    _, ref_cache = m.forward_exact(hist)
    for i in range(SPLIT, m.n_layers):
        if m.layer_types[i] == "attn":
            _close(full[i]["k"][:, :P], ref_cache[i]["k"][:, :P])


def test_extend_hook_that_does_nothing_is_bitwise(tiny):
    m = tiny
    ids = _ids(N, 10)
    out, seen = [], []
    for hook in (None, lambda low: seen.append(sorted(low))):
        cache, hist, _, _, last = _turn1(m, ids, _kva(m))
        out.append(MT.extend(m, cache, [last, 5, 6, 7], hist.shape[0], SPLIT, hook)[0])
    assert torch.equal(out[0], out[1]) and seen == [list(range(SPLIT))]


class CharTokenizer:
    bos_token_id = None

    def __call__(self, text, add_special_tokens=True):
        return types.SimpleNamespace(input_ids=[ord(c) for c in text])

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr, ids))


def test_plan_finds_distractor_needles(monkeypatch):
    text = "".join(chr(97 + (i * 7) % 26) for i in range(400_000))
    monkeypatch.setattr(data, "corpus", lambda: {s: text for s in data.NIAH_SOURCES})
    tok = CharTokenizer()
    for p in data.niah_prompts(tok, "raw", "mk4")[::7]:
        turns = MT.plan(tok, p)
        assert [t["key"] for t in turns] == [p["key"]] + [d["key"] for d in p["distractors"]]
        for t in turns:
            span = "".join(map(chr, p["ids"][t["needle_start"]:t["needle_end"]]))
            assert span == data.niah_needle(t["key"], t["value"])
    with pytest.raises(ValueError, match="mk4"):
        MT.plan(tok, data.niah_prompts(tok, "raw")[0])


def _short_mk4(monkeypatch):
    """A 1000-row raw mk4 prompt: its ~125-row follow-up questions outgrow every K/V buffer's spare rows."""
    text = "".join(chr(97 + (i * 7) % 26) for i in range(20_000))
    monkeypatch.setattr(data, "corpus", lambda: {s: text for s in data.NIAH_SOURCES})
    monkeypatch.setattr(data, "NIAH_LENGTHS", (1000,))
    tok = CharTokenizer()
    return tok, data.niah_prompts(tok, "raw", "mk4")[0]


def test_later_turns_free_the_turn1_buffers_they_regrow(tiny, monkeypatch):
    """Llama MT1 OOM'd on every 32K prompt (10-03): later turns outgrow a K/V buffer's spare rows (KV_RESERVE)
    and regrow it, and the fork's source still held turn 1's late-layer buffers (+2.5 GiB at 32K). Once regrown,
    each must be freed; 'alive' is that bug, 'kept' would mean this prompt never regrew a buffer."""
    m = tiny
    tok, p = _short_mk4(monkeypatch)
    seen, real = [], MT.later_turns

    def spy(model, tok, cache, *args, **kw):
        bufs = {i: weakref.ref(cache[i]["kbuf"]) for i in range(SPLIT, m.n_layers) if m.layer_types[i] == "attn"}
        out = real(model, tok, cache, *args, **kw)
        seen.append({i: "kept" if b() is cache[i]["kbuf"] else "alive" if b() is not None else "freed"
                     for i, b in bufs.items()})
        return out
    monkeypatch.setattr(MT, "later_turns", spy)
    rec = {"runs": {}}
    MT.run_prompt(m, tok, p, [_kva(m)], rec, "raw", (), 8)
    late = [i for i in range(SPLIT, m.n_layers) if m.layer_types[i] == "attn"]
    assert list(rec["runs"]) == ["exact", "kva"]
    assert seen == [dict.fromkeys(late, "freed")] * 2


def test_projector_released_before_later_turns(tiny, monkeypatch):
    """Only turn 1 fills from the projector, so its device copy (Projector.release; 0.6 GiB on Llama) is dropped
    before the later turns, as evaluate.py drops it for the arms that do not use it."""
    m = tiny
    tok, p = _short_mk4(monkeypatch)
    events, real_turns, real_release = [], MT.later_turns, Projector.release

    def turns(*args, **kw):
        events.append("later turns")
        return real_turns(*args, **kw)

    def release(self):
        events.append("release")
        real_release(self)
    monkeypatch.setattr(MT, "later_turns", turns)
    monkeypatch.setattr(Projector, "release", release)
    MT.run_prompt(m, tok, p, [_kva(m)], {"runs": {}}, "raw", (), 8)
    assert events == ["later turns", "release", "later turns"]          # exact, then kva


@pytest.mark.parametrize("name", ["llama-3.1-8b-instruct", "qwen3.5-9b"])
def test_followup_turn_renders_as_chat_template(name):
    path = ROOT / "data" / "models" / name
    if not (path / "tokenizer.json").exists():
        pytest.skip("needs the local tokenizer")
    tok = data.tokenizer(path)
    pieces = MT.followup_pieces(tok, "chat")
    q1, q2 = data.niah_question("chat", "amber-falcon"), data.niah_question("chat", "paper-comet")
    msgs = [{"role": "user", "content": "Some text." + q1}]
    turn1 = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    answer = data.encode(tok, "1234567")
    ids = data.encode(tok, turn1) + answer + MT.ask_ids(tok, "chat", "paper-comet", pieces)
    ref = tok.apply_chat_template(msgs + [{"role": "assistant", "content": "1234567"},
                                          {"role": "user", "content": q2.lstrip()}], tokenize=False,
                                  add_generation_prompt=True)
    got = tok.decode(ids, skip_special_tokens=False)
    # from the answer on (templates may render a past reply's opener differently, e.g. Qwen's think block)
    assert got[got.index("1234567"):] == ref[ref.index("1234567"):]
    assert MT.followup_pieces(tok, "raw") == ([], [])


def test_configs_runnable_under_v10_with_followup_arms_only():
    """Repair arms were dropped by Dylan (10-02); the follow-up test runs under v1.0 as registered exploratory MT1."""
    if not (ROOT / "configs" / "multiturn-llama.json").is_file():
        pytest.skip("needs configs/multiturn-*.json (not in the public code)")
    p = protocol.load()
    for short in ("llama", "qwen"):
        cfg = json.loads((ROOT / "configs" / f"multiturn-{short}.json").read_text())
        assert cfg["protocol"] == "1.0" and cfg["registration"] == ["MT1"] and cfg["niah_variant"] == "mk4"
        stamp = protocol.validate(cfg, p, env={}, set_name="multiturn")
        assert not stamp["battery"]
        MT.check_arms(cfg["arms"], "multiturn")
        assert not any("repair" in a for a in cfg["arms"])
        assert any(a["fill"] == "none" and a.get("select") for a in cfg["arms"])        # skip+qattn5 (FastKV-style)


# ---- end to end: scripts/evaluate.py multiturn on make_tiny.py's models (real tokenizers)
NAMES = {"llama": "llama-3.1-8b-instruct", "qwen": "qwen3.5-9b"}
PROMPT = "niah-8704-prose-10"


def _env(tmp):
    env = dict(os.environ, KVA_CPU_DRYRUN="1", KVA_WORK=str(tmp / "work"), KVA_BULK=str(tmp / "bulk"),
               HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HIP_VISIBLE_DEVICES="-1",
               ROCR_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4")
    env.pop("KVA_DETERMINISTIC", None)
    return env


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    if not all((ROOT / "data" / "models" / m / "tokenizer.json").exists() for m in NAMES.values()):
        pytest.skip("needs the local Llama-3.1 and Qwen3.5 tokenizers")
    tmp = tmp_path_factory.mktemp("multiturn")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tiny.py"), str(tmp / "models")], env=_env(tmp),
                   check=True, capture_output=True)
    return tmp


@pytest.mark.parametrize("short", ["llama", "qwen"])
def test_multiturn_end_to_end(tiny_models, short):
    """The draft config with only the T1024 arms, its projector a random one and no protocol stamp (a CPU code
    path run: the draft's protocol and registration are refused until v1.1 is adopted)."""
    tmp = tiny_models
    m = NAMES[short]
    g = torch.Generator().manual_seed(0)
    proj = tmp / f"{m}-s12.safetensors"
    Projector(12, {i: torch.eye(64) + 0.02 * torch.randn(64, 64, generator=g) for i in range(12, 32)},
              {i: 0.02 * torch.randn(64, generator=g) for i in range(12, 32)}).save(proj, dict(kind="test"))
    cfg = json.loads((ROOT / "configs" / f"multiturn-{short}.json").read_text())
    for k in ("protocol", "registration", "exploratory"):
        cfg.pop(k)
    cfg.update(name=f"mt-{short}", model=str(tmp / "models" / m))
    cfg["arms"] = [dict(a, projector=str(proj)) if "projector" in a else a for a in cfg["arms"] if a["tail"] == 1024]
    base = dict(split=12, tail=1024, fill="proj", projector=str(proj))     # parked repair code stays tested
    cfg["arms"] += [dict(base, name="kva-s12-t1024-repair25", repair=0.25), dict(base, name="kva-s12-t1024-repair50", repair=0.5),
                    dict(base, name="kva-s12-t1024-repair100", repair=1.0),
                    dict(base, name="kva-s12-t1024-repairq25", repair=0.25, repair_by="question"),
                    dict(base, name="kva-s12-t1024-repairq50", repair=0.5, repair_by="question"),
                    dict(base, name="kva-s12-t1024-qattn5-repair100", select=dict(kind="qattn", budget=0.05), repair=1.0)]
    path = tmp / f"mt-{short}.json"
    path.write_text(json.dumps(cfg))
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), str(path), "multiturn",
                          "--only", PROMPT], env=_env(tmp), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    rec = json.loads((tmp / "work" / "results" / f"multiturn-mt-{short}" / f"{PROMPT}.json").read_text())
    assert sorted(rec["runs"]) == sorted(["exact"] + [a["name"] for a in cfg["arms"]])
    assert [t["key"] for t in rec["turns"]] == [rec["key"]] + [d["key"] for d in rec["distractors"]]
    for name, run in rec["runs"].items():
        assert [t["turn"] for t in run["turns"]] == [1, 2, 3, 4]
        assert all(isinstance(t["correct"], bool) and t["finish_reason"] in ("stop", "length") for t in run["turns"])
        assert all(t["needle_projected"] == (t["needle_end"] <= run["P"]) for t in run["turns"])
    kva = rec["runs"]["kva-s12-t1024"]
    for name, run in rec["runs"].items():
        if "repair" in run:
            assert run["turns"][0] == dict(kva["turns"][0], needle_repaired=False) or "select" in run
            assert run["repair"]["row_layers"] == run["repair"]["rows"] * 20 and run["repair"]["macs"] > 0
            assert all(isinstance(t["needle_repaired"], bool) for t in run["turns"])
    full = rec["runs"]["kva-s12-t1024-repair100"]["repair"]
    assert full["rows"] > kva["P"] and rec["runs"]["kva-s12-t1024-repair50"]["repair"]["rows"] == round(kva["P"] / 2)
    q = rec["runs"]["kva-s12-t1024-repairq25"]["repair"]
    assert q["by"] == "question" and q["rows"] == round(kva["P"] / 4) and q["hs_bytes"] == 64 * kva["P"] * 2
    assert full["hs_bytes"] == 64 * full["rows"] * 2
    assert rec["runs"]["skip-s12-t1024-qattn5"]["n_rows"] == rec["runs"]["kva-s12-t1024-qattn5"]["n_rows"] > 0


def test_second_question_reuses_the_turn1_cache_without_refilling(tiny, monkeypatch):
    """Paper T1.8 (n_turns 2): per prompt one lower pass and one selection (rows chosen for question 1); each arm's
    late fill calls (fill, fill_kv, layer_rows: one per late layer) happen in its turn 1, and none while question 2
    (the first distractor key) runs on the carried cache. The count is its own control: a refill before or during
    turn 2 would add (layers - split) calls for that arm."""
    m = tiny
    tok, p = _short_mk4(monkeypatch)
    calls, lowers, chosen, at_turn2 = [], [], [], []
    for name in ("fill", "fill_kv", "layer_rows"):
        real = getattr(m, name)

        def spy(*a, _real=real, _name=name, **k):
            calls.append(_name)
            return _real(*a, **k)
        monkeypatch.setattr(m, name, spy)
    real_lower, real_choose, real_turns = MT.run_lower, MT.SEL.choose, MT.later_turns

    def lower_spy(*a, **k):
        lowers.append(1)
        return real_lower(*a, **k)

    def choose_spy(*a, **k):
        chosen.append(1)
        return real_choose(*a, **k)

    def turns_spy(*a, **k):
        at_turn2.append(len(calls))
        out = real_turns(*a, **k)
        assert len(calls) == at_turn2[-1], "a late layer was refilled during the second question"
        return out
    monkeypatch.setattr(MT, "run_lower", lower_spy)
    monkeypatch.setattr(MT.SEL, "choose", choose_spy)
    monkeypatch.setattr(MT, "later_turns", turns_spy)
    q10 = dict(kind="qattn", budget=0.10)
    split = 4                                   # qattn reads the exact attention layers below the split (layer 3)
    proj = _proj(m, split)
    spec = dict(split=split, tail=TAIL)
    arms = [Arm("ridgefill", fill="proj", projector=proj, meta=dict(spec, name="ridgefill", fill="proj"), **spec),
            Arm("ridgefill-rows10", fill="proj", projector=proj,
                meta=dict(spec, name="ridgefill-rows10", fill="proj", select=q10), **spec),
            Arm("identity-rows10", fill="identity", meta=dict(spec, name="identity-rows10", fill="identity", select=q10),
                **spec)]
    rec = {"runs": {}}
    MT.run_prompt(m, tok, p, arms, rec, "raw", (), 8, None, 2)
    late = m.n_layers - split
    assert lowers == [1] and chosen == [1]
    assert sorted(calls) == sorted(["fill"] * late + ["layer_rows"] * 2 * late)
    # groups run one after another (exact, then each arm): each fills its late layers once, then asks question 2
    assert at_turn2 == [0, late, 2 * late, 3 * late]
    assert [t["key"] for t in rec["turns"]] == [p["key"], p["distractors"][0]["key"]]
    for name, run in rec["runs"].items():
        assert [t["turn"] for t in run["turns"]] == [1, 2]
        t1, t2 = run["turns"]
        assert t2["ctx"] == p["ids"].__len__() + len(t1["gen"]) - 1          # turn 2 extends turn 1's cache
    rows = rec["runs"]["ridgefill-rows10"]["rows"]
    assert rows == rec["runs"]["identity-rows10"]["rows"] and rec["runs"]["ridgefill-rows10"]["n_rows"] > 0
    assert "needle_rows_exact" in rec["runs"]["ridgefill-rows10"]["turns"][1]


def test_turn_count_is_bounded():
    with pytest.raises(ValueError, match="2..4"):
        MT.plan(None, dict(variant="mk4"), 1)
    with pytest.raises(SystemExit, match="not supported on the multiturn set"):
        MT.check_arms([dict(name="pop", split=22, fill="identity", tail_from="continuation")], "multiturn")
