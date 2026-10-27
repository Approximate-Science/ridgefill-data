"""The protocol as code (kva/protocol.py): arm classification, battery rows, and the config check that
runs before every evaluation."""
import json

import pytest

from kva import protocol as PR

P = PR.load()
LLAMA, QWEN = "llama-3.1-8b-instruct", "qwen3.5-9b"


def _proj(model, s):
    return f"/data/proj/{model}-sweep-s{s}.safetensors"


def _kva(model, s, t, **kw):
    return dict(name=f"kva-s{s}-t{t}", split=s, tail=t, fill="proj", projector=_proj(model, s), **kw)


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("\n".join(json.dumps(dict(type="hypothesis", id=i)) for i in ("BL1", "QB1", "FINAL-L1")) + "\n")
    return path


def _cfg(model, arms, **kw):
    return dict(name="t", model=f"/data/models/{model}", protocol=P["version"], registration=["BL1"], arms=arms, **kw)


def test_arm_kinds():
    q5 = dict(kind="qattn", budget=0.05)
    assert PR.arm_kind(_kva(LLAMA, 8, 1024)) == "kva"
    assert PR.arm_kind(_kva(LLAMA, 8, 1024, select=q5)) == "kva+qattn5"
    assert PR.arm_kind(_kva(LLAMA, 8, 1024, select=dict(q5, window="prefix"))) == "kva+qblind5"
    assert PR.arm_kind(dict(name="n", split=8, tail=1024, fill="identity", select=q5)) == "noproj+qattn5"
    assert PR.arm_kind(dict(name="s", split=8, tail=1024, fill="none")) == "skip"
    assert PR.arm_kind(_kva(QWEN, 12, 4096, st="/x.pt", reuse_fill="kva-s12-t4096")) == "kva+st"


def test_non_battery_arms_are_not_kinds():
    if not PR.ARCHIVE.is_dir():
        pytest.skip("needs protocol-archive/ (the v1.0 protocol; not in the public code)")
    p10 = PR.era("1.0")                       # qattn10 rows and the oracle splice became kinds in v1.1, not in v1.0
    assert PR.arm_kind(_kva(LLAMA, 8, 1024, select=dict(kind="digits"))) is None
    assert PR.arm_kind(_kva(LLAMA, 8, 1024, select=dict(kind="qattn", budget=0.1)), p10) is None
    assert PR.arm_kind(_kva(LLAMA, 8, 1024, fill_by_type=dict(attention="proj", gdn="identity"))) is None
    assert PR.arm_kind(dict(_kva(LLAMA, 8, 1024), projector="/p/llama-lr256-s8.safetensors")) is None
    assert PR.arm_kind(dict(name="o", split=8, tail=1024, fill="oracle"), p10) is None


def test_battery_rows_respect_model_splits_tails_and_kinds():
    assert PR.battery_row(_kva(LLAMA, 12, 4096), LLAMA, P)
    assert not PR.battery_row(_kva(LLAMA, 11, 4096), LLAMA, P)              # not a protocol split
    assert not PR.battery_row(_kva(LLAMA, 12, 2048), LLAMA, P)              # not a protocol tail
    assert not PR.battery_row(_kva(LLAMA, 12, 4096, st="/x.pt"), LLAMA, P)  # +st on a dense model
    blind = dict(kind="qattn", budget=0.05, window="prefix")
    assert PR.battery_row(_kva(QWEN, 12, 1024, select=blind), QWEN, P)
    assert not PR.battery_row(_kva(QWEN, 16, 1024, select=blind), QWEN, P)  # blind only at the 3/8 split


def test_validate_battery_config_stamps_battery(ledger):
    stamp = PR.validate(_cfg(LLAMA, [_kva(LLAMA, 8, 1024)]), P, ledger, env={})
    assert stamp["battery"] and stamp["version"] == P["version"] and stamp["sha256"] == P["sha256"]


def test_validate_rejects_missing_or_unregistered_ids(ledger):
    with pytest.raises(PR.ProtocolError, match="no 'registration'"):
        PR.validate(dict(_cfg(LLAMA, [_kva(LLAMA, 8, 1024)]), registration=[]), P, ledger, env={})
    with pytest.raises(PR.ProtocolError, match="not in the ledger"):
        PR.validate(dict(_cfg(LLAMA, [_kva(LLAMA, 8, 1024)]), registration=["ZZ9"]), P, ledger, env={})


def test_validate_off_battery_needs_exploratory_reason(ledger):
    arms = [_kva(LLAMA, 8, 2048)]
    with pytest.raises(PR.ProtocolError, match="outside the battery"):
        PR.validate(_cfg(LLAMA, arms), P, ledger, env={})
    stamp = PR.validate(_cfg(LLAMA, arms, exploratory="tail curve"), P, ledger, env={})
    assert not stamp["battery"] and stamp["exploratory_arms"] == ["kva-s8-t2048"]


def test_validate_rejects_banned_model_wrong_version_and_nondeterminism(ledger):
    with pytest.raises(PR.ProtocolError, match="banned model"):
        PR.validate(_cfg("test-banned-finetune-censored", []), P, ledger, env={})
    with pytest.raises(PR.ProtocolError, match="not the current protocol"):
        PR.validate(dict(_cfg(LLAMA, []), protocol="0.9"), P, ledger, env={})
    with pytest.raises(PR.ProtocolError, match="KVA_DETERMINISTIC"):
        PR.validate(_cfg(LLAMA, []), P, ledger, env={"KVA_DETERMINISTIC": "0"})


@pytest.mark.parametrize("which", sorted(PR.LOCKED_SETS))
def test_locked_sets_need_a_final_registration(ledger, which):
    cfg = _cfg(LLAMA, [_kva(LLAMA, 12, 4096)])
    with pytest.raises(PR.ProtocolError, match="finals only"):
        PR.validate(cfg, P, ledger, env={}, set_name=which)
    stamp = PR.validate(dict(cfg, registration=["FINAL-L1", "BL1"]), P, ledger, env={}, set_name=which)
    assert stamp["battery"] and stamp["registration"] == ["FINAL-L1", "BL1"]
    with pytest.raises(PR.ProtocolError, match="not in the ledger"):        # FINAL- IDs are registered first too
        PR.validate(dict(cfg, registration=["FINAL-ZZ"]), P, ledger, env={}, set_name=which)
    PR.validate(cfg, P, ledger, env={}, set_name="grade")                  # dev sets: any registered ID


def test_locked_sets_are_protocol_locked_sets():
    assert sorted(PR.LOCKED_SETS.values()) == sorted(P["sets"]["locked"])


def test_protocol_splits_fit_each_model():
    for name, m in P["models"].items():
        assert all(0 < s < m["layers"] for s in m["splits"]), name
    for name, s in P["blind_splits"].items():
        assert s in P["models"][name]["splits"], name


def test_exploratory_reason_makes_the_whole_config_exploratory(ledger):
    stamp = PR.validate(_cfg(LLAMA, [_kva(LLAMA, 8, 1024)], exploratory="instrument check"), P, ledger, env={})
    assert not stamp["battery"] and stamp["exploratory_arms"] == []
