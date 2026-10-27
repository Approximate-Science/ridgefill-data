#!/bin/bash
# run_final.sh CELL.json [--new-host]: run one declared finals cell end to end on this box.
#
# All logic (schema validation, FINAL-TBD refusal, ledger check, locked-set manifest check, host
# fingerprint, canary, pairing-unit sample loop with atomic writes + sha256 sidecars, push to both
# destinations, .done) is in finals/lib/runner.py - this wrapper only resolves the python and the
# repo and refuses by name when they are missing. Idempotent: finished samples are skipped, so a
# dead card means: rent another box, mount the same locked set and results dir, pass --new-host.
#
# Env (all named, no defaults baked in for host-specific things - BRIEF-COMMON rule 8):
#   KVA_REPO        the harness repo (default: the parent of this finals/ dir)
#   KVA_PYTHON      python with the repo's pinned deps (default: $KVA_REPO/.venv/bin/python, then python3)
#   KVA_LOCKED_DIR  the mounted locked set (required; named error if unset)
#   KVA_LEDGER      the ledger (default: $KVA_REPO/ledger/paper/ledger.jsonl)
#   FINALS_RESULTS  results root (default: $KVA_REPO/results/finals)
#   KVA_FINALS_MOCK=1  run the mock harness instead (CPU pipeline test; never evidence)
set -euo pipefail

FINALS=$(cd "$(dirname "$0")" && pwd)
REPO=${KVA_REPO:-$(cd "$FINALS/.." && pwd)}

if [ -n "${KVA_PYTHON:-}" ]; then
  PY=$KVA_PYTHON
elif [ -x "$REPO/.venv/bin/python" ]; then
  PY=$REPO/.venv/bin/python
elif command -v python3 >/dev/null; then
  PY=python3
else
  echo "refusing: no python found (set KVA_PYTHON to the repo venv's python)" >&2
  exit 2
fi
export KVA_REPO=$REPO

exec "$PY" "$FINALS/lib/runner.py" "$@"