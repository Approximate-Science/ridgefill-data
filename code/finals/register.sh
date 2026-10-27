#!/bin/bash
# register.sh [CELL.json ...]: PRINT (never run) the FINAL- registration of every finals cell: one
#   labbook hypothesis --dir ledger/paper FINAL-<MODEL>-<KIND> --claim ... --predict ... --kill ...
# per cell (default: every cell in finals/cells), preceded by comment lines naming the sha256 of the cell and of
# every harness config the record pins. Claim, prediction and kill text come from OUTLINE.md's claims table and
# tests and PROTOCOL.md section 6 (finals/lib/registration.py, which marks its own operationalisations "declared
# here"). A cell with no harness path (the T3.1 speed cell) prints a named refusal as a comment instead.
# Run the printed commands from the repo root, after review; then run_final.sh accepts the cell (it re-derives the
# configs and refuses any whose sha256 differs from the pins). A cell changed after registration needs a new id.
set -euo pipefail
FINALS=$(cd "$(dirname "$0")" && pwd)
REPO=${KVA_REPO:-$(cd "$FINALS/.." && pwd)}
PY=${KVA_PYTHON:-python3}
if [ -x "$REPO/.venv/bin/python" ]; then PY="$REPO/.venv/bin/python"; fi
export KVA_REPO=$REPO
if [ $# -eq 0 ]; then set -- "$FINALS"/cells/*.json; fi
echo "# FINAL- registrations ($(date -u +%Y-%m-%dT%H:%M:%SZ), code $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)); review, then run from $REPO"
exec "$PY" "$FINALS/lib/registration.py" "$@"
