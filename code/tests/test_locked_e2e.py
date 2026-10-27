"""End to end on CPU: scripts/evaluate.py on the locked sets with the tiny random models of scripts/make_tiny.py
(real tokenizers), one row per set, every arm kind; and the protocol's refusal of a locked set without a
FINAL- registration. Needs the local Llama-3.1 and Qwen3.5 tokenizers."""
import json
import os
import subprocess
import sys

import pytest
import torch

from kva import locked as L
from kva.kva import Projector

ROOT = L.LOCKED.parents[1]
MODELS = ROOT / "data" / "models"
ROWS = {"ruler": "ruler-niah_single_1-8192-00", "longbench": "lb-hotpotqa-07", "longppl": "lp-prose_8192_0"}
NAMES = {"llama": "llama-3.1-8b-instruct", "qwen": "qwen3.5-9b"}


def _env(tmp):
    env = dict(os.environ, KVA_CPU_DRYRUN="1", KVA_WORK=str(tmp / "work"), KVA_BULK=str(tmp / "bulk"),
               HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HIP_VISIBLE_DEVICES="-1",
               ROCR_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4")
    env.pop("KVA_DETERMINISTIC", None)
    return env


def _run(cfg_path, which, tmp, only):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), str(cfg_path), which, "--only", only],
                          env=_env(tmp), capture_output=True, text=True)


def _proj(path, split, layers, g):
    W = {i: torch.eye(64) + 0.02 * torch.randn(64, 64, generator=g) for i in range(split, layers)}
    B = {i: 0.02 * torch.randn(64, generator=g) for i in range(split, layers)}
    Projector(split, W, B).save(path, dict(kind="test"))
    return str(path)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    if not all((MODELS / m / "tokenizer.json").exists() for m in NAMES.values()):
        pytest.skip("needs the local Llama-3.1 and Qwen3.5 tokenizers")
    tmp = tmp_path_factory.mktemp("locked")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tiny.py"), str(tmp / "models")], env=_env(tmp),
                   check=True, capture_output=True)
    g = torch.Generator().manual_seed(0)
    q5 = dict(kind="qattn", budget=0.05)
    cfgs = {}
    for short, m in NAMES.items():
        p12 = _proj(tmp / f"{m}-s12.safetensors", 12, 32, g)
        arms = [dict(name="kva", split=12, tail=1024, fill="proj", projector=p12),
                dict(name="noproj", split=12, tail=1024, fill="identity"),
                dict(name="skip", split=12, tail=1024, fill="none"),
                dict(name="kva-qattn5", split=12, tail=1024, fill="proj", projector=p12, select=q5),
                dict(name="noproj-qblind5", split=12, tail=1024, fill="identity", select=dict(q5, window="prefix")),
                dict(name="kva-ktail", split=12, tail=1024, fill="proj", projector=p12, rescale="ktail")]
        if short == "qwen":
            p16 = _proj(tmp / f"{m}-s16.safetensors", 16, 32, g)
            st = tmp / "st.pt"
            torch.save({i: 1e-3 * torch.randn(4, 16, 16, generator=g) for i in range(16, 32) if i % 4 != 3}, st)
            arms += [dict(name="kva16", split=16, tail=1024, fill="proj", projector=p16),
                     dict(name="kva16-st", split=16, tail=1024, fill="proj", projector=p16, st=str(st), reuse_fill="kva16")]
        cfg = dict(name=f"e2e-{short}", model=str(tmp / "models" / m), tokenizer=str(MODELS / m),
                   longbench_max_len=31500, longppl_max_len=8192, ruler_max_len=8192, arms=arms)
        cfgs[short] = tmp / f"{short}.json"
        cfgs[short].write_text(json.dumps(cfg))
    return tmp, cfgs


@pytest.mark.parametrize("short,which", [("llama", "ruler"), ("llama", "longbench"), ("llama", "longppl"),
                                         ("qwen", "longppl")])
def test_locked_set_end_to_end(tiny, short, which):
    tmp, cfgs = tiny
    out = _run(cfgs[short], which, tmp, ROWS[which])
    assert out.returncode == 0, out.stderr[-3000:]
    rec = json.loads((tmp / "work" / "results" / f"{which}-e2e-{short}" / f"{ROWS[which]}.json").read_text())
    arms = json.loads(cfgs[short].read_text())["arms"]
    assert sorted(rec["runs"]) == sorted(["exact"] + [a["name"] for a in arms])
    man = L.manifest()
    assert rec["source_sha256"] == man[rec["source"]]["sha256"] and rec["task"]
    assert which == "longbench" or rec["length"] == 8192
    assert rec["provenance"][-1]["config"]["set"] == which
    n = rec["n"]
    for name, run in rec["runs"].items():
        assert run["tail"] == (n if name == "exact" else 1024)
        if which == "longppl":
            assert len(run["logprobs"]) == rec["cont_len"] and run["metric"] == "nll"
            assert run["score"] == pytest.approx(-sum(run["logprobs"]) / rec["cont_len"])
        else:
            assert run["metric"] == rec["metric"] and 0 <= run["score"] <= 1 and isinstance(run["pred"], str)
            assert len(run["gen"]) <= rec["max_new"]
    # v1.1.3: on longppl the scored continuation is off limits to the qattn window and the tail scale
    scored = rec["cont_len"] if which == "longppl" else 0
    assert rec["runs"]["kva-qattn5"]["window_end"] == n - scored
    assert rec["runs"]["kva-ktail"]["rescale_skip_last"] == scored
    assert rec["runs"]["noproj-qblind5"]["window_end"] == n - 1024
    assert rec["runs"]["noproj-qblind5"]["n_rows"] == round(0.05 * n)
    assert rec["runs"]["noproj-qblind5"]["rows"][-1][1] <= n - 1024
    if short == "qwen":
        assert rec["runs"]["kva16-st"]["st"] and rec["runs"]["kva16-st"]["score"] != rec["runs"]["kva16"]["score"]


def test_locked_set_refused_without_final_registration(tmp_path):
    if not (ROOT / "ledger" / "paper" / "ledger.jsonl").is_file():
        pytest.skip("needs the ledger (ledger/paper/ledger.jsonl, not in the public code)")
    cfg = dict(name="t", protocol="1.0", registration=["BL1"], model=str(MODELS / NAMES["llama"]),
               arms=[dict(name="noproj-s12-t4096", split=12, tail=4096, fill="identity")])
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    out = _run(path, "ruler", tmp_path, "x")
    assert out.returncode != 0 and "finals only" in out.stderr + out.stdout
