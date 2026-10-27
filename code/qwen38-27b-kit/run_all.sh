#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
IDENTITY_ONLY=0
LONGBENCH_IDENTITY=0
GPU_COUNT=4
while [ "$#" -gt 0 ]; do
  case "$1" in
    --identity-only) IDENTITY_ONLY=1; shift ;;
    --longbench-identity) LONGBENCH_IDENTITY=1; shift ;;
    --gpus) GPU_COUNT="${2:?--gpus requires 2 or 4}"; shift 2 ;;
    *) break ;;
  esac
done
case "$GPU_COUNT" in 2|4) ;; *) echo "--gpus must be 2 or 4" >&2; exit 1 ;; esac
OUT=${1:-$PWD/results}
if [ "$#" -gt 1 ]; then echo "Usage: run_all.sh [--identity-only] [--gpus N] [RESULTS]" >&2; exit 1; fi
mkdir -p "$OUT"
if [ -f "$OUT/RATE-STOP.json" ]; then
  echo "A rate stop is pending. Report the revised cost to the owner before resuming." >&2
  exit 1
fi
if [ "$IDENTITY_ONLY" -eq 1 ]; then
  python3 preflight.py --gpus "$GPU_COUNT" --identity-only --out "$OUT"
  python3 reproduction.py --gpus "$GPU_COUNT" --identity-only --gpu 0 --out "$OUT"
  # Includes LongBench by default, last after RULER, KL and TTFT. The legacy
  # --longbench-identity flag is accepted for existing command sequences.
  exec python3 identity_run.py --gpus "$GPU_COUNT" --out "$OUT"
fi
if [ "$LONGBENCH_IDENTITY" -eq 1 ]; then echo "--longbench-identity requires --identity-only" >&2; exit 1; fi
python3 preflight.py --gpus "$GPU_COUNT" --out "$OUT"
python3 checks.py --gpus "$GPU_COUNT" --gpu 0 --out "$OUT"
python3 drive.py ruler_replay.py "$OUT" --gpus "$GPU_COUNT"
python3 drive.py kld_run.py "$OUT" --gpus "$GPU_COUNT"
python3 timing.py --gpus "$GPU_COUNT" --gpu 0 --out "$OUT"
python3 mix.py --gpus "$GPU_COUNT" --gpu 0 --out "$OUT"
python3 report.py --gpus "$GPU_COUNT" --out "$OUT"
