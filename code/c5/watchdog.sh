#!/bin/bash
# c5/watchdog.sh SINCE DRIVER_PID CONTAINER SAFETY_DIR: qfn/watchdog.sh for the C5 run. Since 2026-10-08 18:4x
# (Dylan) it stops ONLY on unrecoverable faults (c5/kfilter.sh); MES/ring timeouts, page faults and VRAM are logged. Every 5 s: tools/kcheck.sh
# (amdgpu MES / SMU / ring timeout / GPU reset / device lost since SINCE) and amdgpu page / memory-access faults; VRAM
# per R9700 (log above 27 GiB, stop above 31 GiB); every 60 s the host's early signs (page migrations, compaction
# stalls, KFD evicted time, the container's cgroup). On a hit: write SAFETY_DIR/KERNEL-ALERT, kill the driver, save
# the container log, stop and remove the container, save the kernel log since SINCE, exit 1. run_c5.sh refuses to
# boot while any KERNEL-ALERT exists. Exit 0 once the driver is gone and the container is not running.
REPO=$(cd "$(dirname "$0")/.." && pwd)
SINCE="$1"; DRIVER="$2"; C="$3"; D="$4"
KCHECK=$REPO/c5/kfilter.sh          # tools/kcheck.sh, riding through isolated MES TLB timeouts (c5/kfilter.sh)
DOCKER=${DOCKER:-docker}
# C5_WATCH_PCI: the PCI addresses of the cards the run uses, space-separated (as lspci -D prints them)
WATCH=${C5_WATCH_PCI:?set C5_WATCH_PCI to the PCI addresses of the cards the run uses}
WARN=$((27 * 1024 * 1024 * 1024)); GUARD=$((31 * 1024 * 1024 * 1024))
mkdir -p "$D"
CARDS=""
for c in /sys/class/drm/card[0-9]; do
  case " $WATCH " in *" $(basename "$(readlink -f "$c/device")") "*) CARDS="$CARDS $c/device/mem_info_vram_used";; esac
done
echo "watchdog pid $$ since $SINCE driver $DRIVER container $C cards$CARDS started $(date -Is)" >> $D/watchdog.log
peak=0; tick=0; last_cs=""; last_tol=0
signs() {
  local mig ev cg free cs
  mig=$(awk '/^pgmigrate_success/{m=$2} /^compact_stall/{c=$2} /^thp_fault_alloc/{t=$2} END{print "pgmigrate=" m " compact_stall=" c " thp_fault=" t}' /proc/vmstat)
  ev=$(cat /sys/class/kfd/kfd/proc/*/stats_*/evicted_ms 2>/dev/null | awk '{s+=$1} END{print s+0}')
  cg=$(find /sys/fs/cgroup/user.slice -maxdepth 6 -type d -name "docker-*.scope" 2>/dev/null | head -1)
  free=$(awk '/^MemFree:/{printf "%.1f", $2/2^20}' /proc/meminfo)
  echo "$(date -Is) signs $mig kfd_evicted_ms=$ev cg_current_gib=$([ -n "$cg" ] && awk '{printf "%.1f", $1/2^30}' $cg/memory.current 2>/dev/null || echo -) memfree_gib=$free" >> $D/watchdog.log
  cs=${mig##*compact_stall=}; cs=${cs%% *}
  [ -n "$last_cs" ] && [ "$cs" != "$last_cs" ] && echo "$(date -Is) EARLY SIGN: compact_stall $last_cs -> $cs" >> $D/watchdog.log
  last_cs=$cs
}
while true; do
  reason=""
  tick=$((tick + 1)); [ $((tick % 12)) -eq 1 ] && signs
  for f in $CARDS; do
    used=$(cat "$f" 2>/dev/null || echo 0)
    [ "$used" -gt "$peak" ] && peak=$used && echo "$(date -Is) VRAM peak $((peak / 1048576)) MiB ($f)" >> $D/watchdog.log
    [ "$used" -gt "$GUARD" ] && echo "$(date -Is) VRAM above 31 GiB on $f: $((used / 1048576)) MiB (logged, not a stop)" >> $D/watchdog.log
  done
  if ! out=$($KCHECK "$SINCE"); then reason="kernel${reason:++$reason}"; fi
  tol=$($REPO/tools/kcheck.sh "$SINCE" | grep -c .)
  [ "$tol" -gt "$last_tol" ] && { echo "$(date -Is) RECOVERABLE amdgpu lines since $SINCE: $tol (logged, not a stop):"; $REPO/tools/kcheck.sh "$SINCE" | tail -n $((tol - last_tol)); } >> $D/watchdog.log
  last_tol=$tol
  pfn=$(journalctl -k --since "$SINCE" -o short-iso | grep -cE "amdgpu.*(page fault|Memory access fault|PROTECTION_FAULT)")
  [ "$pfn" -gt "${last_pf:-0}" ] && echo "$(date -Is) amdgpu page-fault lines since $SINCE: $pfn (recoverable: logged, not a stop)" >> $D/watchdog.log
  last_pf=$pfn
  if [ -n "$reason" ]; then
    { echo "KERNEL ALERT ($reason) at $(date -Is) (since $SINCE)"; echo "$out"; } > $D/KERNEL-ALERT
    kill "$DRIVER" 2>/dev/null
    $DOCKER logs "$C" > $D/alert-$C.log 2>&1
    $DOCKER stop -t 10 "$C"; $DOCKER rm "$C"
    journalctl -k --since "$SINCE" -o short-iso > $D/alert-kernel.log
    echo "fired ($reason) $(date -Is)" >> $D/watchdog.log
    exit 1
  fi
  if ! kill -0 "$DRIVER" 2>/dev/null && [ "$($DOCKER inspect -f '{{.State.Running}}' $C 2>/dev/null)" != true ]; then
    echo "driver gone, container not running: clean exit $(date -Is); VRAM peak $((peak / 1048576)) MiB" >> $D/watchdog.log
    exit 0
  fi
  sleep 5
done
