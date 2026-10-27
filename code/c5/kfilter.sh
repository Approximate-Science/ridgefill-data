#!/bin/bash
# c5/kfilter.sh SINCE -> exit 1 only on UNRECOVERABLE amdgpu faults since SINCE (the run stops only on faults it
# cannot recover from): a failed GPU reset or recovery, a device lost or lost from the bus, or
# an SMU hang (09-30: the SMU hung, the reset failed, only a reboot helped). Everything else tools/kcheck.sh reports
# (MES timeouts, ring timeouts, successful resets, the overdrive warning) is recoverable: logged by the watchdog, never
# a stop. Prints the unrecoverable lines. KCHECK overrides the checker (tests).
KCHECK=${KCHECK:-$(cd "$(dirname "$0")/.." && pwd)/tools/kcheck.sh}
UNREC='reset failed|reset.*(fail|timed out)|recover(y)? failed|failed to recover|device lost|lost from bus|SMU is in hanged state|SMU.*hang'
out=$("$KCHECK" "$1") && exit 0
bad=$(printf '%s\n' "$out" | grep -iE "$UNREC") || exit 0
printf '%s\n' "$bad"
exit 1
