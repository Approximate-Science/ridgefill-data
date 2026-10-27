#!/usr/bin/env bash
# Run on the rented 4x A100 PCIe 40 GB CUDA box, after the committed source pack is finalized.
set -euo pipefail
cd "$(dirname "$0")"
IDENTITY_ONLY=0
GPU_COUNT=4
while [ "$#" -gt 0 ]; do
  case "$1" in
    --identity-only) IDENTITY_ONLY=1; shift ;;
    --gpus) GPU_COUNT="${2:?--gpus requires 2 or 4}"; shift 2 ;;
    *) break ;;
  esac
done
case "$GPU_COUNT" in 2|4) ;; *) echo "--gpus must be 2 or 4" >&2; exit 1 ;; esac
IDENTITY_OUT="$PWD/results-20261025/results"
if [ "$IDENTITY_ONLY" -eq 1 ] && [ "$#" -eq 1 ]; then IDENTITY_OUT="$1"; shift; fi
if [ "$#" -ne 0 ]; then echo "Usage: setup.sh [--identity-only] [--gpus N] [RESULTS]" >&2; exit 1; fi
sha256sum -c SHA256SUMS > kit-verify.log
if [ ! -s sources/LOCK.json ] || [ ! -s sources/e3.tar.gz ]; then
  echo "Missing committed E3 source package: run finalize_sources.py locally with the scorer commit before transfer." >&2
  exit 1
fi
python3 - <<'PY'
import json,hashlib
from pathlib import Path
lock=json.loads(Path('sources/LOCK.json').read_text())
for name,digest in lock['archives'].items():
 p=Path('sources')/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,p
assert lock['e3_commit'] != lock['e3_base'], 'Missing committed scorer extension'
PY
(cd data && sha256sum -c SHA256SUMS > ../data-verify.log)
apt-get update -qq
apt-get install -y -qq cmake build-essential python3 python3-pip curl git > apt.log
command -v nvcc
nvidia-smi -q > setup-nvidia-smi-q.txt
for arm in stock e3; do
  archive=sources/e3.tar.gz
  if [ "$arm" = stock ]; then archive=sources/stock-1242be66f.tar.gz; fi
  commit=$(python3 -c "import json; x=json.load(open('sources/LOCK.json')); print(x['stock_commit'] if '$arm'=='stock' else x['e3_commit'])")
  variants=(pure)
  # REPRODUCTION uses pure builds only. The normal CHECK-A/B workflow also
  # needs instrumented servers; avoid those two builds for identity-only setup.
  if [ "$IDENTITY_ONLY" -eq 0 ]; then variants+=(instrumented); fi
  for variant in "${variants[@]}"; do
    source_dir="src/$variant/$arm"
    build_dir="build/$variant/$arm"
    log="build-$variant-$arm.log"
    if [ -d "$source_dir" ]; then
      echo "$source_dir already exists; verify/remove it before rebuilding" >&2
      exit 1
    fi
    mkdir -p "$source_dir"
    tar -xzf "$archive" -C "$source_dir"
    if [ "$variant" = instrumented ]; then
      python3 instrument_sources.py "$source_dir"
    fi
    cmake -S "$source_dir" -B "$build_dir" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_COMMIT="$commit" > "$log" 2>&1
    targets=(llama-server)
    if [ "$variant" = pure ]; then
      targets+=(llama-tokenize)
      if [ "$arm" = e3 ]; then targets+=(llama-ridgefill-score); fi
    fi
    cmake --build "$build_dir" -j"$(nproc)" --target "${targets[@]}" >> "$log" 2>&1
    if [ "$variant" = pure ] && [ "$arm" = stock ]; then
      # Exact-forward helper links only the pure upstream library.
      c++ -std=c++17 -O2 exact_score.cpp -I "$source_dir/include" -I "$source_dir/ggml/include" -I "$source_dir/vendor" \
        -L"$build_dir/bin" -Wl,-rpath,'$ORIGIN' -lllama -lggml -lggml-base -o "$build_dir/bin/exact-score" > build-exact-score.log 2>&1
    fi
    python3 build_manifest.py --source "$source_dir" --archive "$archive" --build "$build_dir" --arm "$arm" --variant "$variant"
  done
done
mkdir -p models
if [ ! -s models/Qwen3.8-27B-UD-Q4_K_XL.gguf ]; then
  python3 -m pip install -q --break-system-packages huggingface_hub
  hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf \
    --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 --local-dir models > hf.log 2>&1
fi
sha256sum -c models.sha256
build/pure/e3/bin/llama-ridgefill-score --help > scorer-help.txt
python3 - <<'PY'
from pathlib import Path
text=Path('scorer-help.txt').read_text()
assert '--save-logprobs' in text and '--kl-against' in text, 'Required scorer extension missing'
PY
python3 -m unittest discover -s tests -v
python3 ruler_replay.py --cpu-gate > cpu-gate-box.txt
if [ "$IDENTITY_ONLY" -eq 1 ]; then
  python3 preflight.py --identity-only --gpus "$GPU_COUNT" --out "$IDENTITY_OUT" --tokenizer-port 18299 --verify-tokenization
  echo "Fresh identity builds verified. run_all.sh --identity-only must pass REPRODUCTION before any identity measurement."
else
  python3 preflight.py --gpus "$GPU_COUNT" --tokenizer-port 18299 --verify-tokenization
fi
