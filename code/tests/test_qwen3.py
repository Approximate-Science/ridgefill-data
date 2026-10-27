"""Dense Qwen3 (Qwen3-8B's architecture; kva/model.py Qwen3), fp32 on the CPU, tiny random 4-layer model:
- model_class resolves qwen3 to Qwen3 (llama, llama_swiftkv, qwen3_5 unchanged) and refuses an unknown model_type, a
  scaled RoPE, attention bias and a sliding window by name (UnsupportedArchitecture), never Llama's layer silently;
- the forward equals transformers' Qwen3ForCausalLM, with the config in transformers 5's layout and in the released
  checkpoint's 4.51 layout (rope_theta at the top, rope_scaling null);
- positive control: the same weights read as Llama (no q/k norm) do NOT match, so the comparison resolves the norms;
- prefill then cached decode equals one full forward;
- a fill writes the cache the exact layer writes on the same inputs (k_norm included), bitwise;
- capture (scripts/fit_sweep.py's accumulators and hook over Model.forward_exact) gives the sums a naive
  recomputation gives; the ridge fit from them is a Projector that finish() runs (full tail: exact, bitwise);
- the paper configs for Qwen3-8B: model-commits entry, POP-native split, fit splits and the jobs file agree."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import torch

from kva import model as M
from kva.kva import Arm, Projector, finish, run_lower
from kva.ridge import RidgeFit

ROOT = Path(__file__).resolve().parents[1]
TOL = dict(atol=2e-4, rtol=2e-4)


def _ids(n, seed=0):
    return torch.randint(0, 300, (n,), generator=torch.Generator().manual_seed(seed))


def _fit_sweep():
    spec = importlib.util.spec_from_file_location("fit_sweep", ROOT / "scripts" / "fit_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _with_config(src, tmp_path, **changes):
    dst = tmp_path / "m"
    shutil.copytree(src, dst)
    (dst / "SHA256SUMS").unlink(missing_ok=True)            # kva.integrity rebuilds it for the changed config
    cfg = json.loads((dst / "config.json").read_text())
    for k, v in changes.items():
        if v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    (dst / "config.json").write_text(json.dumps(cfg))
    return dst


def test_model_class_resolves_and_refuses(tiny_qwen3, tiny_llama, tiny_qwen, tmp_path):
    assert M.model_class(tiny_qwen3[0]) is M.Qwen3
    assert M.model_class(tiny_llama[0]) is M.Llama and M.model_class(tiny_qwen[0]) is M.Qwen35
    other = _with_config(tiny_qwen3[0], tmp_path, model_type="qwen4_exp")
    with pytest.raises(M.UnsupportedArchitecture, match="model_type 'qwen4_exp'"):
        M.load(other, dtype=torch.float32)


@pytest.mark.parametrize("change,match", [
    (dict(rope_parameters=dict(rope_type="yarn", rope_theta=1e6, factor=4.0, original_max_position_embeddings=32768)),
     "rope_type 'yarn'"),
    (dict(attention_bias=True), "attention_bias True"),
    (dict(use_sliding_window=True), "use_sliding_window True")])
def test_qwen3_refuses_what_it_does_not_implement(tiny_qwen3, tmp_path, change, match):
    with pytest.raises(M.UnsupportedArchitecture, match=match):
        M.load(_with_config(tiny_qwen3[0], tmp_path, **change), dtype=torch.float32)


def _hf_logits(hf, ids):
    with torch.no_grad():
        return hf(ids[None]).logits[0].float()


def test_qwen3_matches_hf(tiny_qwen3):
    path, hf = tiny_qwen3
    m = M.load(path, dtype=torch.float32)
    assert isinstance(m, M.Qwen3) and m.n_layers == 4 and m.layer_types == ["attn"] * 4
    ids = _ids(150)
    mine, _ = m.forward_exact(ids, keep=torch.arange(150))
    torch.testing.assert_close(mine, _hf_logits(hf, ids), **TOL)


def test_qwen3_released_config_layout_matches_hf(tiny_qwen3, tmp_path):
    """Qwen3-8B's config.json (transformers 4.51): rope_theta at the top level, rope_scaling null."""
    path, hf = tiny_qwen3
    old = _with_config(path, tmp_path, rope_parameters=None, rope_scaling=None, rope_theta=1000000.0)
    m = M.load(old, dtype=torch.float32)
    ids = _ids(120, 1)
    mine, _ = m.forward_exact(ids, keep=torch.arange(120))
    torch.testing.assert_close(mine, _hf_logits(hf, ids), **TOL)


def test_positive_control_llama_layer_misses_the_qk_norms(tiny_qwen3):
    path, hf = tiny_qwen3
    m = M.Llama(path, dtype=torch.float32)                  # the pre-Qwen3 dispatch: Llama's layer, no q/k norm
    ids = _ids(150)
    mine, _ = m.forward_exact(ids, keep=torch.arange(150))
    assert (mine - _hf_logits(hf, ids)).abs().max() > 1e-2


def test_qwen3_cached_decode_matches_full(tiny_qwen3):
    m = M.load(tiny_qwen3[0], dtype=torch.float32)
    ids = _ids(140, 3)
    full, _ = m.forward_exact(ids, keep=torch.arange(140))
    _, cache = m.forward_exact(ids[:130])
    rows = []
    for t in range(130, 140):
        h = m.embed(ids[t:t + 1])
        for i in range(m.n_layers):
            h = m.layer(i, h, torch.tensor([t]), cache)
        rows.append(m.logits(h))
    torch.testing.assert_close(torch.cat(rows), full[130:], atol=1e-4, rtol=1e-4)


def test_qwen3_fill_writes_the_exact_layers_cache(tiny_qwen3):
    m = M.load(tiny_qwen3[0], dtype=torch.float32)
    x = torch.randn(50, m.hidden, generator=torch.Generator().manual_seed(4))
    pos = torch.arange(50)
    for i in range(m.n_layers):
        a, b = m.new_cache(), m.new_cache()
        m.fill(i, x, pos, a)
        m.layer(i, x, pos, b)
        assert torch.equal(a[i]["k"], b[i]["k"]) and torch.equal(a[i]["v"], b[i]["v"])
    k = m.reads(1, x)["k"]                                   # reads() keys are k-normed: unit RMS per head * k_norm
    unit = (k / m.W[1]["k_norm"]).pow(2).mean(-1)
    torch.testing.assert_close(unit, torch.ones_like(unit), atol=1e-3, rtol=1e-3)


def test_qwen3_capture_fit_and_fill(tiny_qwen3):
    """One capture over two documents with fit_sweep's hook, checked against a naive recomputation, then the ridge
    projector of split 2 through finish(): it runs, differs from the identity fill, and with the whole prompt as the
    tail is exact bitwise."""
    fs = _fit_sweep()
    m = M.load(tiny_qwen3[0], dtype=torch.float32)
    L, H, splits = m.n_layers, m.hidden, [1, 2]
    docs = [_ids(300, 10), _ids(260, 11)]
    acc = fs.accumulators(splits, L, H)
    xs = {}
    hook = fs.capture_hook(acc, xs)
    inputs = []
    with torch.no_grad():
        for d in docs:
            xs.clear()
            got = {}
            m.forward_exact(d, hook=lambda i, h: (hook(i, h), got.__setitem__(i, h)))
            inputs.append(got)
            acc["n"] += d.shape[0]
    assert acc["n"] == 560
    cx = inputs[0][2].float().mean(0)                        # the naive sums of split 2 (first document's shift)
    xc = torch.cat([g[2].float() for g in inputs]) - cx
    torch.testing.assert_close(acc["sxx"][2], (xc.T @ xc).double(), atol=1e-6, rtol=1e-6)
    cy = inputs[0][3].float().mean(0)
    yc = torch.cat([g[3].float() for g in inputs]) - cy
    torch.testing.assert_close(acc["sxy"][(2, 3)], (xc.T @ yc).double(), atol=1e-6, rtol=1e-6)
    fit = RidgeFit(fs.split_state(acc, 2, L, H))
    W, B = {}, {}
    for i in range(2, L):
        W[i], B[i] = fit.layer(i, (0.01,))[0.01]
    proj = Projector(2, {i: w.float() for i, w in W.items()}, {i: b.float() for i, b in B.items()})
    ids = _ids(200, 12)
    hs, lower = run_lower(m, ids, {2})
    exact, _ = m.forward_exact(ids, keep=torch.arange(160, 200))
    kva, _, P = finish(m, ids, hs[2], lower, Arm("kva", split=2, tail=60, fill="proj", projector=proj), 40)
    ident, _, _ = finish(m, ids, hs[2], lower, Arm("noproj", split=2, tail=60, fill="identity"), 40)
    full, _, P0 = finish(m, ids, hs[2], lower, Arm("kva", split=2, tail=200, fill="proj", projector=proj), 40)
    assert P == 140 and P0 == 0 and torch.equal(full, exact)
    assert torch.isfinite(kva).all() and (kva - ident).abs().max() > 1e-4 and (kva - exact).abs().max() > 1e-6


# ---- the paper configs for Qwen3-8B (no weights needed)
def test_qwen3_paper_configs_agree():
    if not (ROOT / "jobs" / "paper-qwen3-8b.jobs").is_file():
        pytest.skip("needs jobs/paper-qwen3-8b.jobs (not in the public code)")
    commits = (ROOT / "configs" / "model-commits.txt").read_text().splitlines()
    line = [x for x in commits if x.startswith("qwen3-8b ")]
    assert line == ["qwen3-8b commit b968826d9c46dd6066d109eabc6255188de91218"]
    from kva.kva import pop_split
    assert pop_split(36) == 24 and pop_split(32) == 22
    jobs = (ROOT / "jobs" / "paper-qwen3-8b.jobs").read_text().splitlines()
    fit = [x.split() for x in jobs if x.strip() and not x.startswith("#") and "fit_sweep.py" in x]
    assert len(fit) == 1 and fit[0][2] == "data/models/qwen3-8b" and fit[0][3] == "9,13,18,22,24"   # cba2d91: + 9, 22 (curve)
    cfg = json.loads((ROOT / "configs" / "paper" / "pop-native-qwen3-8b.json").read_text())
    [pop] = [a for a in cfg["arms"] if a["name"] == "pop-native"]
    assert cfg["model"].endswith("/qwen3-8b") and pop["split"] == 24 and pop["tail_from"] == "continuation"
    assert {a["split"] for a in cfg["arms"] if a is not pop} == {13}
