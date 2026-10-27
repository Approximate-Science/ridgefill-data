#!/bin/bash
# refit_clean.sh [--dry-run]: the clean Llama + Qwen3.5 refits on the rented box, before any cell (finals/lib/refit.py:
# jobs/paper-refit-clean.jobs' fit_sweep lines, models pulled at their pinned commits, every projector checked against
# the clean calibration row count the cells require). In the image: docker run --gpus all -v /data:/data
# -e KVA_GPU_NAME=A100 -e HF_TOKEN=... --entrypoint /opt/kva/finals/refit_clean.sh kva-finals:protocol-v1.1
set -euo pipefail
FINALS=$(cd "$(dirname "$0")" && pwd)
export KVA_REPO=${KVA_REPO:-$(cd "$FINALS/.." && pwd)}
PY=${KVA_PYTHON:-python3}
if [ -x "$KVA_REPO/.venv/bin/python" ]; then PY="$KVA_REPO/.venv/bin/python"; fi
exec "$PY" "$FINALS/lib/refit.py" "$@"
