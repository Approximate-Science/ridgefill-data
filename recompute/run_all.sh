#!/usr/bin/env bash
# Run every recompute script from the repository root. Claim lines ("<id><TAB><value>") go to stdout, everything else
# to stderr; a failing script fails the run. Usage: bash recompute/run_all.sh > recompute.out
# PYTHON selects the interpreter (default python3; numpy is needed by the BCa scripts).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
run() {
  local out
  echo "== recompute/$*" >&2
  out="$("$PY" -I -B "recompute/$@")"
  printf '%s\n' "$out" | awk -F'\t' 'NF == 2 { print; next } { print > "/dev/stderr" }'
}
run check_fit_records.py
run e3_timing.py
run e3_ruler.py
run lb_report.py
run lb_groups.py
run e3_kl.py
run e3_checks.py
run e3_bf16.py
run e3_mix.py
run fn_speed.py
run fn_speed.py --ruler64k
run fn_quality.py
run fnq.py
run fn_determinism.py
run ruler1.py
run r3_keytok.py
run r4_counts.py
run r5_compute_vs_pop.py
run r6_nofill_by_length.py
run lb8_fill_vs_identity.py
run lb9_decode.py
run lb10_swiftkv.py
run lb11_pop.py
run s7_primary_cost.py
run s8_depth.py
run s9_rows.py
run s10_splices.py
run s11_longbench.py
run s12_turn2.py
run dev_history.py
