#!/usr/bin/env bash
# Start one LongBench worker per GPU: GPUs 0-1 begin with stock, 2-3 with E3; each switches arms when its queue is empty.
# Usage: drive.sh OUT [extra run_lb.py args, e.g. --limit 1]
set -euo pipefail
W=/workspace/lb
OUT=${1:?out dir}; shift
mkdir -p $OUT
for gpu in 0 1 2 3; do
  arm=stock; [ $gpu -ge 2 ] && arm=e3
  nohup python3 -u $W/run_lb.py --gpu $gpu --port $((18100 + gpu)) --arm $arm --switch --out $OUT \
    --model $W/models/Qwen3.8-27B-UD-Q4_K_XL.gguf --projector $W/models/ridgefill-projector-qwen3.8-27b-i8-S16.gguf \
    --stock-bin $W/build/stock-1242be66f/bin/llama-server --e3-bin $W/build/e3-a2d3dea0b/bin/llama-server "$@" \
    > $OUT/worker-gpu$gpu.log 2>&1 &
  echo "gpu$gpu $arm pid $!"
done
