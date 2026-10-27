"""Convert the public RidgeFill release (TheCacheIsALie/RidgeFill-Projector-Qwen3.8-Flash-Next-i8 at 0b6f2ce0) into
the format of tcclaviger's engine (vllm/v1/ced.py + vllm/models/qwen4_exp/amd/ced.py in tcclaviger/vllm:29.08.10).

Release (ridgefill.json, split 24):
  proj8.L<L>.safetensors   proj.L.codes int8 [2560, 10240], proj.L.scale bf16 [2560, 80], proj.L.bias bf16 [2560]
                           (i8_row128: row i's 128-column group g is codes[i, 128g:128g+128] * scale[i, g])
  correction.safetensors   st.L f32 [48, 128, 128] for the 18 late GDN layers: the mean terminal-state error,
                           all heads, [heads, V, K]
  rowsel.safetensors       score f32 [248320]: -log unigram frequency on the class56 tokens, -inf elsewhere
Engine (one directory):
  ridgefill-s24.safetensors    layer.L [2560, 10241] bf16 (map | bias), L = 24..47, and final [10240, 10241] bf16
                               (boundary -> final_multi_hidden: read only for the approximated rows' logits and the
                               MTP drafter's hidden; the release ships no final, see --final)
  ridgefill-s24-st.rank<r>.pt  {"sum": {L: fp32 [heads/tp, V, K]}, "count": {L: 1}} per fit rank, as
                               tools/ced/fit_st.py writes them; StCorrection concatenates the files' heads in rank
                               order, divides by count and keeps its rank's slice, so sum / 1 is st.L exactly
  freq.safetensors             logfreq f32 [248320] = -score (bitwise; XrowsSelector ranks by -logfreq), 0.0 where
                               the release masks the id (-inf: not a class56 token, never read; c5/engine_check.py
                               checks the engine's own class mask selects exactly the release's finite ids)
  SHA256SUMS, CONVERSION.json
Maps: codes * scale in fp32 (exact: an 8-bit code times an 8-bit-mantissa bf16 fits fp32's 24 bits), the bias
appended in fp32, then one cast to bf16 (the engine's Projector casts every map to bf16 at load; this cast is the
only rounding, and the engine would apply the same one to the fp32 map).
Inputs are read from disk with O_DIRECT and checked against ridgefill.json / SHA256SUMS before use; the conversion
runs twice from two reads and must agree byte for byte, and every output is read back from disk (O_DIRECT) and
compared (the host's RAM flips bits: kva/integrity.py).

Usage: python -m c5.convert --src <projector-src> --out <projector-tcc> [--final source --final-source <bf16 file>]
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load as st_load, save as st_save

from kva.integrity import read_direct

RELEASE = "TheCacheIsALie/RidgeFill-Projector-Qwen3.8-Flash-Next-i8"
REVISION = "0b6f2ce017fe912198c697bcc813a1578bc67ce1"
MANIFEST_SHA256 = "795363fa153f0e7e16454af6acabd644709e703d84de0dd9535565226bb51faa"
SPLIT, LAYERS, H, HC, GROUP = 24, 48, 2560, 4, 128
WIDTH = HC * H
GDN_HEADS, TP = 48, 2
STEM = f"ridgefill-s{SPLIT}"
ST_BASE = f"{STEM}-st"


class ConversionError(RuntimeError):
    pass


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def read_checked(path, want):
    b = read_direct(str(path))
    got = sha256(b)
    if got != want:
        raise ConversionError(f"{path}: sha256 {got}, expected {want}")
    return b


def load_release(src, manifest_sha256=MANIFEST_SHA256):
    """(manifest, {name: bytes}) of every manifest file, each checked against ridgefill.json and SHA256SUMS."""
    src = Path(src)
    mb = read_direct(str(src / "ridgefill.json"))
    if sha256(mb) != manifest_sha256:
        raise ConversionError(f"{src}/ridgefill.json: sha256 {sha256(mb)}, expected {manifest_sha256}")
    man = json.loads(mb)
    sums = {}
    if (src / "SHA256SUMS").is_file():
        for line in read_direct(str(src / "SHA256SUMS")).decode().splitlines():
            if line.strip():
                h, name = line.split(None, 1)
                sums[name.lstrip("*")] = h
    files = {}
    for name, h in man["files"].items():
        if name in sums and sums[name] != h:
            raise ConversionError(f"{name}: SHA256SUMS {sums[name]} != ridgefill.json {h}")
        files[name] = read_checked(src / name, h)
    if man["split"] != SPLIT or man["projector"]["layout"] != "i8_row128" or man["stream_width"] != WIDTH:
        raise ConversionError(f"release geometry {man['split']} {man['projector']['layout']} {man['stream_width']}")
    return man, files


def dequant(codes, scale, bias):
    """[rows, width + 1] fp32: codes * their row group's scale, then the bias column."""
    rows, width = codes.shape
    if codes.dtype != torch.int8 or width % GROUP or scale.shape != (rows, width // GROUP) or bias.shape != (rows,):
        raise ConversionError(f"i8_row128 shapes {tuple(codes.shape)} {tuple(scale.shape)} {tuple(bias.shape)}")
    w = codes.to(torch.float32).view(rows, width // GROUP, GROUP) * scale.to(torch.float32)[:, :, None]
    return torch.cat([w.view(rows, width), bias.to(torch.float32)[:, None]], 1)


def maps(man, files):
    out = {}
    for L in range(SPLIT, LAYERS):
        t = st_load(files[man["projector"]["files"][str(L)]])
        m = dequant(t[f"proj.{L}.codes"], t[f"proj.{L}.scale"], t[f"proj.{L}.bias"])
        if m.shape != (H, WIDTH + 1):
            raise ConversionError(f"layer {L}: {tuple(m.shape)}")
        out[f"layer.{L}"] = m.to(torch.bfloat16).contiguous()
    return out


def final_map(kind, source, source_sha):
    if kind == "zero":
        return torch.zeros(WIDTH, WIDTH + 1, dtype=torch.bfloat16)
    f = st_load(read_checked(source, source_sha))["final"]
    if f.shape != (WIDTH, WIDTH + 1) or f.dtype != torch.bfloat16:
        raise ConversionError(f"{source}: final {tuple(f.shape)} {f.dtype}")
    return f.contiguous()


def corrections(man, files):
    """[{"sum": {L: fp32 [heads/TP, V, K]}, "count": {L: 1}} per rank], int layer keys (fit_st.py's)."""
    t = st_load(files["correction.safetensors"])
    layers = [int(x) for x in man["correction"]["layers"]]
    if sorted(int(k.split(".")[1]) for k in t) != sorted(layers):
        raise ConversionError(f"correction keys {sorted(t)} vs manifest layers {layers}")
    local = GDN_HEADS // TP
    ranks = []
    for r in range(TP):
        s = {}
        for L in layers:
            c = t[f"st.{L}"]
            if c.shape != (GDN_HEADS, 128, 128) or c.dtype != torch.float32:
                raise ConversionError(f"st.{L}: {tuple(c.shape)} {c.dtype}")
            s[L] = c[r * local:(r + 1) * local].clone().contiguous()
        ranks.append({"sum": s, "count": {L: 1 for L in layers}})
    return ranks


def freq(files):
    """(logfreq [V], eligible ids): -score where finite, 0.0 where the release masks the id."""
    sc = st_load(files["rowsel.safetensors"])["score"]
    if sc.dtype != torch.float32 or sc.ndim != 1 or torch.isnan(sc).any() or torch.isposinf(sc).any():
        raise ConversionError("rowsel score: expected f32 [V], finite or -inf")
    masked = torch.isneginf(sc)
    return torch.where(masked, torch.zeros_like(sc), -sc).contiguous(), int((~masked).sum())


def one_key(meta):
    """safetensors writes __metadata__ in hash order (several keys give different bytes run to run): one sorted JSON
    string under one key keeps the file byte-reproducible. The engine reads no metadata key of ours (ced_st, ced_freq,
    ced_xrows and taps stay absent: the arms name the correction and the table by environment)."""
    return {"c5": json.dumps(meta, sort_keys=True)}


def pt_bytes(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def build(src, final_kind, final_source=None, final_sha=None, manifest_sha256=MANIFEST_SHA256):
    """{file name: bytes} of the engine directory (deterministic: the same inputs give the same bytes)."""
    man, files = load_release(src, manifest_sha256)
    tensors = maps(man, files)
    tensors["final"] = final_map(final_kind, final_source, final_sha)
    meta = {"format": "tcc-ced (vllm/models/qwen4_exp/amd/ced.py Projector)", "release": RELEASE,
            "revision": REVISION, "release_manifest_sha256": manifest_sha256,
            "maps": "layer.L = bf16(fp32(codes) * fp32(scale) | fp32(bias)) from proj8.L<L>.safetensors",
            "final": (f"verbatim from {Path(final_source).name} (sha256 {final_sha}, the release's fit.proj_sha256)"
                      if final_kind == "source" else "zero map (the release ships no final)")}
    out = {f"{STEM}.safetensors": st_save(tensors, metadata=one_key(meta))}
    for r, obj in enumerate(corrections(man, files)):
        out[f"{ST_BASE}.rank{r}.pt"] = pt_bytes(obj)
    logfreq, eligible = freq(files)
    out["freq.safetensors"] = st_save({"logfreq": logfreq}, metadata=one_key({
        "source": f"{RELEASE}@{REVISION} rowsel.safetensors score", "rule": "logfreq = -score; 0.0 where score = -inf",
        "eligible": str(eligible)}))
    report = {"release": RELEASE, "revision": REVISION, "release_manifest_sha256": manifest_sha256,
              "split": SPLIT, "tp": TP, "final": final_kind,
              "final_source": str(final_source) if final_kind == "source" else None,
              "final_source_sha256": final_sha if final_kind == "source" else None,
              "freq_eligible_ids": eligible, "inputs": {n: man["files"][n] for n in sorted(man["files"])},
              "outputs": {n: sha256(b) for n, b in sorted(out.items())}}
    out["CONVERSION.json"] = (json.dumps(report, indent=1) + "\n").encode()
    return out


def write(out_dir, blobs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, b in blobs.items():
        tmp = out_dir / f".{name}.tmp"
        with open(tmp, "wb") as f:
            f.write(b)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_dir / name)
    (out_dir / "SHA256SUMS").write_text("".join(f"{sha256(b)}  {n}\n" for n, b in sorted(blobs.items())))


def convert(src, out, final_kind="source", final_source=None, manifest_sha256=MANIFEST_SHA256):
    man = json.loads(read_direct(str(Path(src) / "ridgefill.json")))
    final_sha = man["fit"]["proj_sha256"]
    first = build(src, final_kind, final_source, final_sha, manifest_sha256)
    second = build(src, final_kind, final_source, final_sha, manifest_sha256)   # a second read: RAM faults differ
    bad = [n for n in first if sha256(first[n]) != sha256(second[n])]
    if bad:
        raise ConversionError(f"two conversions disagree on {bad}: host RAM fault, run again")
    write(out, first)
    back = {n: sha256(read_direct(str(Path(out) / n))) for n in first}
    bad = [n for n in first if back[n] != sha256(first[n])]
    if bad:
        raise ConversionError(f"read-back differs on {bad}")
    return back


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--final", choices=("source", "zero"), default="source")
    ap.add_argument("--final-source", default="")
    a = ap.parse_args(argv)
    if a.final == "source" and not a.final_source:
        ap.error("--final source needs --final-source (the bf16 file whose sha256 is the release's fit.proj_sha256)")
    for name, h in sorted(convert(a.src, a.out, a.final, a.final_source).items()):
        print(h, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
