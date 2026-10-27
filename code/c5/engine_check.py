"""Round-trip check of the converted projector through tcclaviger's own loaders, on the CPU, inside his image.

Runs in tcclaviger/vllm:29.08.10 with no GPU: `import vllm` probes the ROCm device, so his two files are loaded
unmodified by path (vllm/v1/ced.py as vllm.v1.ced, vllm/models/qwen4_exp/amd/ced.py as its package's ced) with
stand-ins for the few vllm names they import at module level (logger, TP rank / all-reduce, forward context,
the GDN layer-name encoder), none of which the loaders below call. Then, with the run's environment:
  resolve_bundle      the bundle the engine resolves: projector file, correction rank files, alpha, xrows, freq
  projector_taps      split 24, taps (24,)
  Projector (torch)   every layer.L weight / bias equals bf16(codes * scale) / bf16(bias) recomputed here from the
                      release files, bit for bit; final equals the source file's final
  StCorrection        for TP 2 rank 0 and 1 (and TP 1), each layer's slice equals the release's st.L heads
                      [r * 24, (r + 1) * 24) bit for bit, alpha 1
  XrowsSelector       his class mask (from the tokenizer) selects exactly the release's finite-score ids, and his
                      rows() on seeded random prompts equals the release's own ranking (rarest share by score, ties by
                      position) at shares 0.10 / 0.25 / 0.50 / 1.0
Writes a JSON report (argv[1], default stdout) and exits non-zero on the first failure.

docker run --rm --network none -e HIP_VISIBLE_DEVICES= -v KIT:/kit:ro -v TCC:/tcc:ro -v SRC:/src:ro -v MODEL:/model:ro
  -v FINALSRC:/finalsrc:ro --entrypoint python tcclaviger/vllm:29.08.10 -I /kit/c5/engine_check.py /out/report.json
"""
import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path

import torch

VLLM = Path("/opt/vllm/lib/python3.14/site-packages/vllm")
TCC, SRC, MODEL = Path("/tcc"), Path("/src"), Path("/model")
FINALSRC = Path(os.environ.get("C5_FINAL_SOURCE", "/finalsrc/kva-big-s24.safetensors"))
SPLIT, LAYERS, HEADS = 24, 48, 48


class _Log(logging.LoggerAdapter):
    def info_once(self, msg, *a, **k):
        self.info(msg, *a)

    def warning_once(self, msg, *a, **k):
        self.warning(msg, *a)

    def process(self, msg, kwargs):
        kwargs.pop("scope", None)
        return msg, kwargs


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def load_his():
    """(vllm.v1.ced, the model-side ced) loaded from the image's files, unmodified."""
    _stub("vllm")
    _stub("vllm.logger", init_logger=lambda n: _Log(logging.getLogger(n), {}))
    _stub("vllm.distributed", get_tensor_model_parallel_rank=lambda: 0,
          tensor_model_parallel_all_reduce=lambda x: x)
    _stub("vllm.forward_context", get_forward_context=lambda: None)
    for p in ("vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.mamba",
              "vllm.model_executor.layers.mamba.gdn"):
        _stub(p)
    _stub("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn", _encode_layer_name=lambda s: s)
    _stub("vllm.v1")
    for p in ("vllm.models", "vllm.models.qwen4_exp", "vllm.models.qwen4_exp.amd"):
        _stub(p)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    eng = load("vllm.v1.ced", VLLM / "v1" / "ced.py")
    model = load("vllm.models.qwen4_exp.amd.ced", VLLM / "models" / "qwen4_exp" / "amd" / "ced.py")
    return eng, model


def release_tensors(name):
    from safetensors.torch import load_file
    return load_file(str(SRC / name))


def reference_rows(score, ids, P, share):
    """The release's own class56 rule on its score table: eligible = finite score, the rarest share (highest score),
    ties by position, ascending positions."""
    t = torch.as_tensor(ids[:P], dtype=torch.long)
    s = score[t]
    m = torch.isfinite(s).nonzero().flatten()
    if m.numel() == 0:
        return ()
    k = round(share * m.numel())
    if k >= m.numel():
        return tuple(m.tolist())
    if k <= 0:
        return ()
    order = torch.argsort(s[m], descending=True, stable=True)[:k]
    return tuple(sorted(m[order].tolist()))


def main(out=None):
    report, fail = {}, []

    def check(name, ok, detail=""):
        report[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            fail.append(name)

    eng, his = load_his()
    proj = TCC / "ridgefill-s24.safetensors"
    os.environ.update(DYLUHN_KVA="1", DYLUHN_KVA_PROJECTOR=str(proj), DYLUHN_KVA_ST="ridgefill-s24-st",
                      DYLUHN_KVA_XROWS="class56:0.25")
    b = eng.resolve_bundle(str(proj))
    check("bundle", b.projector == str(proj) and b.st_base == "ridgefill-s24-st" and b.st_alpha == 1.0
          and [Path(p).name for p in b.st_files] == ["ridgefill-s24-st.rank0.pt", "ridgefill-s24-st.rank1.pt"]
          and b.xrows == {"rule": "class56", "share": 0.25, "classes": ("cap", "mixed", "piece")}
          and Path(b.freq).name == "freq.safetensors" and b.kind == "quality", repr(b))
    os.environ["DYLUHN_KVA_ST"] = "0"
    os.environ["DYLUHN_KVA_XROWS"] = "0"
    b0 = eng.resolve_bundle(str(proj))
    check("bundle_plain", b0.kind == "plain" and not b0.st_files and b0.xrows is None, repr(b0))
    check("taps", eng.projector_taps(str(proj), LAYERS) == (SPLIT,), str(eng.projector_taps(str(proj), LAYERS)))

    cfg = types.SimpleNamespace(num_hidden_layers=LAYERS, hc_count=4, hidden_size=2560, linear_num_value_heads=HEADS)
    P = his.Projector(str(proj), torch.device("cpu"), cfg, impl="torch")
    check("projector_split", P.split == SPLIT and P.names == [f"layer.{i}" for i in range(SPLIT, LAYERS)] + ["final"])
    bad = []
    for L in range(SPLIT, LAYERS):
        t = release_tensors(f"proj8.L{L}.safetensors")
        c, s, bias = t[f"proj.{L}.codes"], t[f"proj.{L}.scale"], t[f"proj.{L}.bias"]
        w = (c.float().view(2560, 80, 128) * s.float()[:, :, None]).view(2560, 10240).to(torch.bfloat16)
        if not (torch.equal(P.weight[f"layer.{L}"], w) and torch.equal(P.bias[f"layer.{L}"], bias)):
            bad.append(L)
    check("projector_maps_bitwise", not bad, f"layers differing: {bad}")
    if FINALSRC.is_file():
        from safetensors import safe_open
        with safe_open(str(FINALSRC), "pt") as f:
            fin = f.get_tensor("final")
        check("projector_final_source", torch.equal(P.weight["final"], fin[:, :-1]) and
              torch.equal(P.bias["final"], fin[:, -1]))
    else:
        report["projector_final_source"] = {"ok": None, "detail": f"{FINALSRC} not mounted"}

    os.environ["DYLUHN_KVA_ST"] = "ridgefill-s24-st"
    bst = eng.resolve_bundle(str(proj))
    st = release_tensors("correction.safetensors")
    for tp in (1, 2):
        for rank in range(tp):
            C = his.StCorrection(bst, cfg, torch.device("cpu"), rank, tp)
            local = HEADS // tp
            bad = [L for L in C.correction if not torch.equal(
                C.correction[L], st[f"st.{L}"][rank * local:(rank + 1) * local])]
            layers = sorted(C.correction)
            check(f"correction_tp{tp}_rank{rank}", not bad and C.alpha == 1.0 and
                  layers == sorted(int(k.split(".")[1]) for k in st), f"layers {layers}, differing {bad}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    score = release_tensors("rowsel.safetensors")["score"]
    for share in (0.10, 0.25, 0.50, 1.0):
        sel = eng.XrowsSelector(eng.parse_xrows(f"class56:{share}"), str(TCC / "freq.safetensors"), tok)
        if share == 0.25:
            elig = torch.isin(sel.cls, sel.want)
            fin = torch.isfinite(score[:sel.vocab])
            diff = (elig != fin).nonzero().flatten()
            check("class_mask_equal", diff.numel() == 0 and int(torch.isfinite(score[sel.vocab:]).sum()) == 0,
                  f"engine {int(elig.sum())} eligible, release {int(fin.sum())}; differing ids {diff[:20].tolist()}")
        g = torch.Generator().manual_seed(20261008)
        mism = 0
        for trial in range(40):
            n = int(torch.randint(600, 6000, (1,), generator=g))
            ids = torch.randint(0, sel.vocab, (n,), generator=g).tolist()
            # repeats exercise the stable tie-break by position
            ids += ids[: n // 4]
            P_ = len(ids) - 512
            if sel.rows(ids, P_) != reference_rows(score, ids, P_, share):
                mism += 1
        check(f"xrows_rows_share{share:g}", mism == 0, f"{mism} of 40 prompts differ")
    report["_summary"] = {"failed": fail, "image_vllm": str(VLLM)}
    text = json.dumps(report, indent=1)
    if out:
        Path(out).write_text(text + "\n")
    print(text)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
