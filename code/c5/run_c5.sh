#!/bin/bash
# c5/run_c5.sh MODE [ARM ...]: the C5 Flash-Next final run (c5/PLAN.md). One server boot per arm (RidgeFill settings
# are server-wide), each: preflight -> page-cache purge (FIX-4) -> docker run (eager TP2, FIX-2/4 settings) -> watchdog
# -> healthy -> CED policy line checked against the arm -> client-side warm-up (FIX-1 analogue) -> the arm's steps
# (c5.client) -> stop -> per-request real tails from the server log (c5.annotate). Resumable: a finished step leaves
# results/<arm>/.done-<step>; a finished sample its JSON + .sha256 (c5.client skips it).
#   MODE smoke   boot the graded arm, run c5.smoke (no locked data), stop. The coordinator reads smoke.json and
#                re-plans from its rates before MODE run.
#   MODE run     the mandatory arms in PLAN order (or the ARMs named), then stop.
#   MODE extras  the budget-gated extras (ridgefill on RULER), each only if it fits before
#                C5_END (epoch seconds; default: the run's start + 10 h, written by MODE run to state/end_epoch).
#   MODE final   the final on the deterministic engine: FINAL steps, FINALORDER (or the ARMs named); needs
#                C5_ENGINE_OVERLAY and C5_RUN.
#   MODE status  what is done.
# Hard rules (brief): holds the GPU lock file (C5_GPU_LOCK) for the whole run; refuses with another
# container running, a KERNEL-ALERT anywhere, kernel fault lines since the last boot, a dirty worktree, a projector file
# that differs from its SHA256SUMS, or too little host RAM; stops at the first fault (watchdog).
# C5_ENGINE_OVERLAY=DIR (opt-in; unset = the unpatched engine): every file DIR/SHA256SUMS lists (paths relative to the
# vllm package) is mounted read-only over the image's own; preflight checks the sums, the boot records them, and after
# healthy the container's copies must match them (check_overlay).
set -u
KIT=$(cd "$(dirname "$0")/.." && pwd)
# Machine-specific paths come from the environment (named errors when unset): C5_DIR (the work directory: projector,
# results, logs), C5_MODEL_DIR (the Flash-Next checkpoint), C5_GPU_LOCK (a lock file shared by every GPU job on the
# machine), C5_WATCH_PCI (the cards' PCI addresses, for watchdog.sh); optional: C5_PY (default the kit's .venv python),
# DOCKER (default docker from PATH), C5_CACHE (default $C5_DIR/cache/tcc), C5_EXTRA_ALERTS (more KERNEL-ALERT files
# that must not exist, space-separated).
C5=${C5_DIR:?set C5_DIR to the C5 work directory}
PY=${C5_PY:-$KIT/.venv/bin/python}
DOCKER=${DOCKER:-docker}
IMAGE=tcclaviger/vllm:29.08.10
IMAGE_ID=sha256:809657ddb1533db10a0c08267c3ee6fde180b839e379713414efa121415914a3
OVERLAY=${C5_ENGINE_OVERLAY:-}; VLLM_PKG=/opt/vllm/lib/python3.14/site-packages/vllm
MODEL=${C5_MODEL_DIR:?set C5_MODEL_DIR to the Flash-Next checkpoint directory}
PROJ=$C5/projector-tcc
CACHE=${C5_CACHE:-$C5/cache/tcc}
PLE=$C5/pleoffload
NAME=c5-flashnext; PORT=8079; URL=http://127.0.0.1:$PORT
TAIL=2048
MAX_LEN=${C5_MAX_LEN:-66560}          # the longest RULER row is 65,408 tokens + 32 generated; 40960 drops the 64K rows
# C5_RUN=NAME: results, logs and state under $C5/NAME (a fresh run beside the earlier ones; unset = $C5 itself).
# safety/ stays shared: a KERNEL-ALERT from any run refuses every boot.
RUNDIR=$C5${C5_RUN:+/$C5_RUN}
OUT=$RUNDIR/results; LOGS=$RUNDIR/logs; SAFE=$C5/safety; STATE=$RUNDIR/state
GPU_LOCK=${C5_GPU_LOCK:?set C5_GPU_LOCK to the GPU lock file every GPU job on this machine holds}
: "${C5_WATCH_PCI:?set C5_WATCH_PCI to the PCI addresses of the cards the run uses (watchdog.sh)}"; export C5_WATCH_PCI
MODE=${1:-status}; shift || true
mkdir -p $OUT $LOGS $SAFE $STATE $PLE $CACHE
log() { echo "$(date -Is) $*" | tee -a $LOGS/driver.log; }
export PYTHONPATH=$KIT HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4

# ---- arms: server environment (beyond the common one) and steps; c5/PLAN.md has the reasons
KVA_ENV="DYLUHN_KVA=1 DYLUHN_KVA_PROJECTOR=/c5proj/ridgefill-s24.safetensors DYLUHN_KVA_TAIL=$TAIL DYLUHN_KVA_EXACT_TAIL=1 DYLUHN_KVA_PROJECTOR_IMPL=triton DYLUHN_KVA_ST_ALPHA=1.0"
declare -A ENV STEPS KIND
ENV[exact]="";                                                    KIND[exact]=none
ENV[graded]="$KVA_ENV DYLUHN_KVA_ST=ridgefill-s24-st DYLUHN_KVA_XROWS=class56:0.25"; KIND[graded]=quality
ENV[ridgefill-corr]="$KVA_ENV DYLUHN_KVA_ST=ridgefill-s24-st DYLUHN_KVA_XROWS=0";  KIND[ridgefill-corr]=st
ENV[ridgefill]="$KVA_ENV DYLUHN_KVA_ST=0 DYLUHN_KVA_XROWS=0";                         KIND[ridgefill]=plain
ENV[prot10]="$KVA_ENV DYLUHN_KVA_ST=ridgefill-s24-st DYLUHN_KVA_XROWS=class56:0.1";   KIND[prot10]=quality
ENV[prot50]="$KVA_ENV DYLUHN_KVA_ST=ridgefill-s24-st DYLUHN_KVA_XROWS=class56:0.5";   KIND[prot50]=quality
ENV[prot100]="$KVA_ENV DYLUHN_KVA_ST=ridgefill-s24-st DYLUHN_KVA_XROWS=class56:1.0";  KIND[prot100]=quality
# diagnostic (2026-10-09, after the graded arm lost needles that plain RidgeFill kept): which add-on breaks it
ENV[diag-quality]="${ENV[graded]}";                                                 KIND[diag-quality]=quality
ENV[diag-rows]="$KVA_ENV DYLUHN_KVA_ST=0 DYLUHN_KVA_XROWS=class56:0.25";               KIND[diag-rows]=quality
ENV[diag-corr]="${ENV[ridgefill-corr]}";                                            KIND[diag-corr]=st
# correction-only failed (8% on niah_single_1): the shipped correction with its V/K axes swapped (stT) and with the
# two TP halves swapped (stS), files beside the pinned ones in projector-tcc (SHA256SUMS untouched)
# chat-format RULER check (2026-10-09): raw prompts are invalid for this thinking model; exact vs quality on chat prompts
ENV[diagc-exact]="";                                                                 KIND[diagc-exact]=none
ENV[diagc-quality]="${ENV[graded]}";                                                 KIND[diagc-quality]=quality
# H10 (2026-10-09): exact, chat multiquery, answer prefix without its leading space (the colon stop is the same with no
# projector at all; every RULER prefix opens the reply with a space-prefixed token the model never writes there)
ENV[diagns-exact]="";                                                                KIND[diagns-exact]=none
# H11 (2026-10-09): the same prompts with no answer prefix at all (H10 cut exact's stops 12 -> 3, not to 0)
ENV[diagnp-exact]="";                                                                KIND[diagnp-exact]=none
# H12 (2026-10-09, Dylan's design, replaces H11): the question asked as an instruction with the answer format, no prefix
ENV[diagin-exact]="";                                                                KIND[diagin-exact]=none
ENV[diag-corrT]="$KVA_ENV DYLUHN_KVA_ST=ridgefill-s24-stT DYLUHN_KVA_XROWS=0";        KIND[diag-corrT]=st
ENV[diag-corrS]="$KVA_ENV DYLUHN_KVA_ST=ridgefill-s24-stS DYLUHN_KVA_XROWS=0";        KIND[diag-corrS]=st
# refit of the correction on 29.08.10 itself (the engine's tools/ced st_fit_driver.py + fit_st.py, the release's
# 13 sterm prompts from datasets/graded-and-calibration-sets/plan-sterm.json), then the two arms that use it
ENV[stfit]="$KVA_ENV DYLUHN_KVA_ST=0 DYLUHN_KVA_XROWS=0 DYLUHN_KVA_ST_FIT_DIR=/cache/stfit-2908/dump"; KIND[stfit]=plain
ENV[diag-corrR]="$KVA_ENV DYLUHN_KVA_ST=ridgefill-s24-st2908 DYLUHN_KVA_XROWS=0";     KIND[diag-corrR]=st
ENV[diag-qualityR]="$KVA_ENV DYLUHN_KVA_ST=ridgefill-s24-st2908 DYLUHN_KVA_XROWS=class56:0.25"; KIND[diag-qualityR]=quality
# steps: set:pass. ruler = the 4 NIAH tasks at 8K-32K; ruler64 = niah_single_1 at 64K (declared, PLAN); rulerlow = the
# 4 NIAH tasks at 8K+16K (the same-boot exact rerun, PLAN); control = the graded boot's exact-path docs
STEPS[exact]="ppl:a ppl:b ruler:a ruler64:a rulerlow:b speed:a rulerchat:c ruler64chat:c"
STEPS[graded]="ppl:a ruler:a ruler64:a speed:a control:a rulerchat:c ruler64chat:c"
STEPS[ridgefill-corr]="ppl:a speed:a"
STEPS[ridgefill]="ppl:a speed:a rulerchat:c ruler64chat:c"
STEPS[prot10]="ppl:a speed:a"
STEPS[prot50]="ppl:a speed:a"
STEPS[prot100]="ppl:a speed:a"
STEPS[diagns-exact]="rulermqns:n"; STEPS[diagnp-exact]="rulermqnp:p"; STEPS[diagin-exact]="rulermqin:i"; STEPS[diagc-exact]="rulerchatdiag:c"; STEPS[diagc-quality]="rulerchatdiag:c"
STEPS[stfit]="stfit:a"; STEPS[diag-corrR]="rulerdiag:a"; STEPS[diag-qualityR]="rulerdiag:a"
STEPS[diag-corrT]="rulerquick:a"; STEPS[diag-corrS]="rulerquick:a"
STEPS[diag-quality]="rulerdiag:a"; STEPS[diag-rows]="rulerdiag:a"; STEPS[diag-corr]="rulerdiag:a"
ORDER="exact graded ridgefill-corr ridgefill prot10 prot50 prot100"
# the fixed-format rerun (MODE fix, 2026-10-09: the perplexity and RULER tests rerun in full with the fixed
# prompting). Perplexity first, every arm that measured it: the locked set again (pass f; exact
# twice, f and g, for its noise floor) plus the labelled / unlabelled dev chat docs (chat-labelled-20261009) and
# fresh-long's chat docs <= 32K. Then chat RULER in the instruct format (H12) on the arms that ran chat RULER.
declare -A FIXPPL FIXRULER
FIXPPL[exact]="ppl:f pplchat:f pplfresh:f ppl:g pplchat:g pplfresh:g"
for a in graded ridgefill ridgefill-corr prot100; do FIXPPL[$a]="ppl:f pplchat:f pplfresh:f"; done
for a in exact graded ridgefill; do FIXRULER[$a]="rulerinst:i ruler64inst:i"; done
FIXPPLORDER="exact graded ridgefill ridgefill-corr prot100"; FIXRULERORDER="exact graded ridgefill"
# the final on the deterministic engine (MODE final, 2026-10-10): the 29.08.10 image plus the determinism overlay
# (C5_ENGINE_OVERLAY, required) into a fresh C5_RUN (required). Claim arms first; exact's same-boot ppl:b right after
# ppl:a is the protocol's determinism read on the locked set; RULER in the instruct format (H12) on exact, graded and
# ridgefill; the dev-chat and fresh-chat perplexity on every arm with chat in its plan; speed on every arm.
declare -A FINAL
FINAL[exact]="ppl:a ppl:b pplchat:a pplfresh:a rulerinst:a ruler64inst:a speed:a"
FINAL[graded]="ppl:a pplchat:a pplfresh:a rulerinst:a ruler64inst:a speed:a control:a"
FINAL[ridgefill]="ppl:a pplchat:a pplfresh:a rulerinst:a ruler64inst:a speed:a"
FINAL[ridgefill-corr]="ppl:a pplchat:a pplfresh:a speed:a"
FINAL[prot100]="ppl:a pplchat:a pplfresh:a speed:a"
FINAL[prot10]="ppl:a speed:a"; FINAL[prot50]="ppl:a speed:a"
FINALORDER="exact graded ridgefill ridgefill-corr prot100 prot10 prot50"
# extras (MODE extras), budget-gated: arm, steps, minutes the plan estimates
EXTRAS=("ridgefill|ruler:a ruler64:a|110")

preflight() {
  local p=""
  for a in $SAFE/KERNEL-ALERT ${C5_EXTRA_ALERTS:-}; do
    [ -e "$a" ] && p="$p; $a exists"
  done
  local running; running=$($DOCKER ps --format '{{.Names}} {{.Image}}')
  [ -n "$running" ] && p="$p; containers running: $running"
  if git -C "$KIT" rev-parse HEAD >/dev/null 2>&1; then
    [ -z "$(git -C "$KIT" status --porcelain -- . ':(exclude)results' ':(exclude)ledger')" ] || p="$p; uncommitted changes in $KIT"
    git -C "$KIT" rev-parse HEAD > $STATE/code_commit
  else
    p="$p; $KIT is not a git checkout (run from a committed checkout of the kit)"
  fi
  (cd $PROJ && sha256sum --quiet -c SHA256SUMS) >/dev/null 2>&1 || p="$p; $PROJ does not match its SHA256SUMS"
  [ -z "$OVERLAY" ] || (cd "$OVERLAY" && sha256sum --quiet --strict -c SHA256SUMS) >/dev/null 2>&1 ||
    p="$p; engine overlay $OVERLAY: no such dir, no SHA256SUMS, or a file does not match it"
  [ "$($DOCKER image inspect -f '{{.Id}}' $IMAGE 2>/dev/null)" = "$IMAGE_ID" ] || p="$p; $IMAGE is not $IMAGE_ID"
  local avail; avail=$(awk '/^MemAvailable:/{printf "%d", $2/2^20}' /proc/meminfo)
  [ "$avail" -ge 61 ] || p="$p; MemAvailable ${avail} GiB < 61 (FIX-4: 35 experts + 4 PLE + 22)"
  local free; free=$(df -BG --output=avail $PLE | tail -1 | tr -dc 0-9)
  [ "$free" -ge 60 ] || p="$p; ${free} GiB free for the PLE table < 60"
  local since; since=$(cat $STATE/last_since 2>/dev/null || date -d '-30 min' '+%Y-%m-%d %H:%M:%S')
  $KIT/c5/kfilter.sh "$since" >/dev/null || p="$p; kernel fault lines since $since (c5/kfilter.sh)"
  [ -z "$p" ] || { log "PREFLIGHT REFUSED${p}"; return 1; }
}

purge() { find "$@" -type f -size +1M -exec dd if={} iflag=nocache count=0 status=none \; 2>/dev/null; }

boot() { # arm
  local arm=$1 since args e
  since="$(date '+%Y-%m-%d %H:%M:%S')"; echo "$since" > $STATE/last_since
  purge "$MODEL" "$PROJ"                                         # FIX-4: start with our big files out of the page cache
  args=(run -d --name $NAME --label c5=flashnext --network host --ipc host --shm-size 32g -v /sys:/sys:ro
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable
        --device /dev/kfd --device /dev/dri --group-add 105 --group-add 39 --ulimit memlock=-1:-1
        --memory 108g --memory-swap 108g
        -v $KIT/c5/shim:/c5shim:ro -v $MODEL:/app/models:ro -v $PROJ:/c5proj:ro -v $PLE:/app/pleoffload -v $CACHE:/cache -v $CACHE/tunableop:/tunableop)
  for e in OMP_NUM_THREADS=8 PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_RECORD_UNTUNED=0 \
           VLLM_ROCM_USE_AITER=0 GPU_MAX_HW_QUEUES=1 HSA_ENABLE_INTERRUPT=1 HSA_ENABLE_MWAITX=1 \
           TRITON_CACHE_DIR=/cache/triton VLLM_CACHE_ROOT=/cache/vllm TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
           ROCR_VISIBLE_DEVICES=0,1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1 \
           CLAV_ATTN_AUTOTUNE=0 R4D_AR_QUANT=0 PYTHONPATH=/c5shim CLAV_PLE_I6PROJ=0 ${ENV[$arm]}; do
    args+=(-e "$e")
  done
  mkdir -p $OUT/$arm/boot
  if [ -n "$OVERLAY" ]; then # C5_ENGINE_OVERLAY: mount what SHA256SUMS lists, as copied into this boot's provenance
    cp "$OVERLAY/SHA256SUMS" $OUT/$arm/boot/overlay-SHA256SUMS || { log "no $OVERLAY/SHA256SUMS"; return 1; }
    printf 'dir %s\nsha256 %s  SHA256SUMS\n' "$OVERLAY" "$(sha256sum < $OUT/$arm/boot/overlay-SHA256SUMS | cut -c1-64)" \
      > $OUT/$arm/boot/overlay.txt
    while read -r h f; do f=${f#\*}; args+=(-v "$OVERLAY/$f:$VLLM_PKG/$f:ro"); done < $OUT/$arm/boot/overlay-SHA256SUMS
    log "OVERLAY $(tr '\n' ' ' < $OUT/$arm/boot/overlay.txt)($(wc -l < $OUT/$arm/boot/overlay-SHA256SUMS) files)"
  fi
  # FIX-5: c5/shim/sitecustomize.py (PYTHONPATH above) restores PleNvmeStore.reuse, read only by a log line that
  # 29.08.10 left behind; without it every --ple-nvme-offload boot dies (see the shim's header)
  # PLE n-gram table: --nvme-engram-direct (2026-10-08: the n-gram table streams from the SSD): rows go
  # O_DIRECT from the table file on the SSD to a small shared ring the GPU gathers from; no RAM row cache (the
  # 4 GiB cache of the first boots is gone). Same mode on every arm.
  # FIX-6: CLAV_PLE_I6PROJ=0 (the engine's own switch, vllm/models/qwen4_exp/amd/ops/ple_proj.py): 29.08.10's fused
  # int6 PLE projection reshapes the padded NVMe rows without trimming them (shape '[38912, 130]' is invalid for input
  # of size 5242880, smoke boot 2026-10-08 17:04); the unfused path trims them (ple/core.py dequant_rows). Prefill
  # uses torch.matmul on both paths; only decode's fused kernel is lost. Same setting on every arm.
  # FIX-2: THP off for every engine process (PR_SET_THP_DISABLE survives fork and exec), then the image's entrypoint
  args+=(--entrypoint /opt/vllm/bin/python3 $IMAGE -c
         "import ctypes,os,sys; ctypes.CDLL(None).prctl(41,1,0,0,0); os.execv('/app/tools/image_entrypoint.sh',['/app/tools/image_entrypoint.sh']+sys.argv[1:])"
         /app/models --served-model-name qwen3.8-flash-next --host 127.0.0.1 --port $PORT
         --tensor-parallel-size 2 --max-model-len $MAX_LEN --max-num-seqs 1 --max-num-batched-tokens 2048
         --enable-chunked-prefill --kv-cache-dtype fp8 --gpu-memory-utilization 0.72 --language-model-only
         --enforce-eager --no-enable-prefix-caching --enable-expert-offload --expert-offload-mem 35
         --ple-nvme-offload --ple-nvme-dir /app/pleoffload --nvme-engram-direct --degen-max-period 0)
  printf '%s\n' "$DOCKER" "${args[@]}" > $OUT/$arm/boot/docker-run-args.txt
  log "BOOT $arm (since $since)"
  $DOCKER "${args[@]}" > $OUT/$arm/boot/container-id || { log "docker run failed"; return 1; }
  # FIX-4: the cgroup soft limit (rootless docker: user-owned), so the container reclaims its own cache first
  local cid cg; cid=$(cat $OUT/$arm/boot/container-id)
  cg=$(find /sys/fs/cgroup/user.slice -maxdepth 8 -type d -name "docker-$cid.scope" 2>/dev/null | head -1)
  [ -n "$cg" ] && echo 100G > $cg/memory.high 2>/dev/null && log "memory.high $(cat $cg/memory.high)" || log "memory.high not set"
  nohup $KIT/c5/watchdog.sh "$since" $$ $NAME $SAFE >/dev/null 2>&1 9>&- &
  echo $! > $STATE/watchdog_pid
  # FIX-4 purge loop until healthy; it must not outlive the server or hold gpu.lock (fd 9): a failed boot on
  # 2026-10-08 left it running for 40 min holding the lock
  ( for i in $(seq 1 240); do sleep 10; curl -sf -m 2 $URL/health >/dev/null && exit 0
      [ "$($DOCKER inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ] || exit 0; purge "$MODEL"; done ) 9>&- &
  echo $! > $STATE/purge_pid
  local t0=$SECONDS
  until curl -sf -m 2 $URL/health >/dev/null; do
    [ "$($DOCKER inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ] || { log "container died during boot"; return 1; }
    [ $((SECONDS - t0)) -gt 3600 ] && { log "not healthy after 60 min"; return 1; }
    sleep 10
  done
  log "HEALTHY $arm after $((SECONDS - t0)) s"
  echo $((SECONDS - t0)) > $OUT/$arm/boot/healthy_after_s
  [ "$($DOCKER inspect -f '{{.Image}}' $NAME)" = "$IMAGE_ID" ] || { log "container image is not $IMAGE_ID"; return 1; }
  check_overlay $arm || return 1
  $DOCKER logs $NAME > $OUT/$arm/boot/server-boot.log 2>&1
  check_policy $arm || return 1
  warm $arm
}

check_overlay() { # arm: in the running container, the engine's vllm package must be $VLLM_PKG, each overlaid file
                  # there must hash to what this boot recorded, and no unchecked-hash .pyc may shadow it (Python would
                  # import that without reading the source)
  [ -n "$OVERLAY" ] || return 0
  local b=$OUT/$1/boot
  $DOCKER exec -i $NAME /opt/vllm/bin/python3 -I - $VLLM_PKG "$(cat $b/overlay-SHA256SUMS)" > $b/overlay-check.txt 2>&1 <<'EOF'
import hashlib, importlib.util, os, sys
root, sums = sys.argv[1], sys.argv[2].splitlines()
sys.pycache_prefix = os.environ.get("PYTHONPYCACHEPREFIX") or None    # as the engine sees it (-I ignores it)
pkg = list(importlib.util.find_spec("vllm").submodule_search_locations)   # locates vllm without importing it
bad = [os.path.realpath(p) for p in pkg] != [os.path.realpath(root)]
print("OK  " if not bad else "FAIL", "vllm package", pkg)
for line in sums:
    want, rel = line.split(None, 1)
    f = os.path.join(root, rel.removeprefix("*"))
    got = hashlib.sha256(open(f, "rb").read()).hexdigest()
    pyc = importlib.util.cache_from_source(f) if f.endswith(".py") else ""
    shadow = os.path.isfile(pyc) and (open(pyc, "rb").read(8)[4] & 3) == 1   # PEP 552 flags: hash-based, unchecked
    ok = got == want and not shadow
    bad |= not ok
    print("OK  " if ok else "FAIL", f, got, "(an unchecked-hash .pyc shadows it)" if shadow else "")
sys.exit(1 if bad or not sums else 0)
EOF
  local rc=$?
  [ $rc -eq 0 ] || { log "OVERLAY CHECK FAILED in the container (rc $rc, $b/overlay-check.txt)"; return 1; }
  log "OVERLAY ok in the container: $(wc -l < $b/overlay-SHA256SUMS) files"
}

check_policy() { # the engine's own "CED on (...)" line must say what the arm asks for; exact must have none
  local arm=$1 line; line=$(grep -a "CED on (" $OUT/$arm/boot/server-boot.log | tail -1)
  echo "$line" > $OUT/$arm/boot/ced-policy.txt
  grep -a -E "CED (projector|correction)|THP|expert offload|Expert offload|PLE" $OUT/$arm/boot/server-boot.log | tail -40 >> $OUT/$arm/boot/ced-policy.txt
  if [ "${KIND[$arm]}" = none ]; then
    [ -z "$line" ] && return 0; log "exact boot shows a CED policy: $line"; return 1
  fi
  local want="CED on (${KIND[$arm]}): projector /c5proj/ridgefill-s24.safetensors split 24 taps (24,), exact tail $TAIL (unrounded)"
  case "$line" in *"$want"*) ;; *) log "CED policy mismatch for $arm: '$line' (want '$want')"; return 1;; esac
  case "$arm" in
    ridgefill) case "$line" in *"correction none, exact rows none"*) ;; *) log "ridgefill: correction or rows on"; return 1;; esac;;
    diag-rows) case "$line" in *"correction none, exact rows class56-cap+mixed+piece-s25 ranked by /c5proj/freq.safetensors"*) ;; *) log "policy: $line"; return 1;; esac;;
    stfit) case "$line" in *"correction none, exact rows none"*) ;; *) log "stfit: correction or rows on"; return 1;; esac;;
    diag-corrR) case "$line" in *"correction ridgefill-s24-st2908 x 1.0 (2 rank files), exact rows none"*) ;; *) log "policy: $line"; return 1;; esac;;
    diag-qualityR) case "$line" in *"correction ridgefill-s24-st2908 x 1.0 (2 rank files), exact rows class56-cap+mixed+piece-s25 ranked by /c5proj/freq.safetensors"*) ;; *) log "policy: $line"; return 1;; esac;;
    diag-corrT|diag-corrS) case "$line" in *"correction ridgefill-s24-st${arm#diag-corr} x 1.0 (2 rank files), exact rows none"*) ;; *) log "policy: $line"; return 1;; esac;;
    ridgefill-corr|diag-corr) case "$line" in *"correction ridgefill-s24-st x 1.0 (2 rank files), exact rows none"*) ;; *) log "policy: $line"; return 1;; esac;;
    *) local share=${ENV[$arm]##*class56:}
       local lab; lab=$(awk -v s="$share" 'BEGIN{printf "s%g", s * 100}')
       case "$line" in *"correction ridgefill-s24-st x 1.0 (2 rank files), exact rows class56-cap+mixed+piece-$lab ranked by /c5proj/freq.safetensors"*) ;;
         *) log "policy: $line (want class56 $lab)"; return 1;; esac;;
  esac
  log "POLICY ok: $line"
}

warm() { # FIX-1 analogue (the overlay's idle warm-up RPC is not available without the overlay): on the idle server,
         # one prompt_logprobs request and one generation on non-test ids, both paths; results discarded
  $PY - "$URL" "${KIND[$1]}" >> $LOGS/$1.log 2>&1 <<'EOF'
import sys
from c5.client import Client
from c5.smoke import filler_ids
from c5.client import tokenizer
url, kind = sys.argv[1], sys.argv[2]
cl = Client(url, 0 if kind == "none" else 1, 2048)
ids = filler_ids(tokenizer(), 10240)
for kva in ((None,) if kind == "none" else (0, 1)):
    cl.post(cl.body(ids, 1, f"c5-warm-lp-{kva}", kva=kva, prompt_logprobs=0))
    cl.post(cl.body(ids, 8, f"c5-warm-gen-{kva}", kva=kva))
print("warm-up done", flush=True)
EOF
}

stop_server() { # arm
  local arm=$1
  local b=$OUT/$arm/boot logs=() f
  # A second boot of the same arm must not overwrite the first boot's log: on 2026-10-09 that lost graded's ppl,
  # raw-RULER and control request lines, so its tails.json could not be rebuilt. Keep every boot's log; annotate reads all.
  [ -f $b/server.log ] && mv $b/server.log $b/server-$(date -r $b/server.log +%Y%m%dT%H%M%S).log
  $DOCKER logs $NAME > $b/server.log 2>&1
  $DOCKER stop -t 30 $NAME >/dev/null 2>&1; $DOCKER rm $NAME >/dev/null 2>&1
  for f in $b/server-2*.log $b/server.log; do [ -f "$f" ] && logs+=(--log "$f"); done
  $PY -m c5.annotate "${logs[@]}" --results $OUT/$arm >> $LOGS/$arm.log 2>&1 || log "annotate failed for $arm"
  sleep 15
  local pp; pp=$(cat $STATE/purge_pid 2>/dev/null); [ -n "$pp" ] && kill $pp 2>/dev/null
  local w; w=$(cat $STATE/watchdog_pid 2>/dev/null); [ -n "$w" ] && kill -0 $w 2>/dev/null && { sleep 10; kill $w 2>/dev/null; }
  log "STOPPED $arm"
}

run_step() { # arm step:pass [deadline]
  local arm=$1 step=${2%%:*} pass=${2##*:} dl=${3:-0} kva=1 extra=()
  [ "${KIND[$arm]}" = none ] && kva=0
  local mark=$OUT/$arm/.done-$step-$pass
  [ -e $mark ] && { log "skip $arm $step:$pass (done)"; return 0; }
  case $step in
    ppl) extra=(--sets ppl)
         case $arm in prot*) extra+=(--ppl-max-len ${C5_KNOB_PPL_MAXLEN:-32768});; esac;;   # PLAN: the knob arms' cut
    ruler) extra=(--sets ruler --ruler-lengths 8192,16384,32768);;
    ruler64) extra=(--sets ruler --ruler-lengths 65536 --ruler-tasks niah_single_1);;
    rulerlow) extra=(--sets ruler --ruler-lengths 8192,16384);;
    # amendment 9 (2026-10-09): the C5 RULER readout in chat format (thinking off); raw-format ruler/ruler64 above are
    # kept as measured and reported as invalid-format
    rulerchat) extra=(--sets ruler --ruler-lengths 8192,16384,32768 --ruler-format chat);;
    ruler64chat) extra=(--sets ruler --ruler-lengths 65536 --ruler-tasks niah_single_1 --ruler-format chat);;
    rulerchatdiag) extra=(--sets ruler --ruler-lengths 8192 --ruler-tasks niah_single_1,niah_multikey_1 --ruler-format chat);;
    rulermqns) extra=(--sets ruler --ruler-lengths 8192,16384,32768 --ruler-tasks niah_multiquery --ruler-format chat
                      --ruler-prefix lstrip);;
    rulermqnp) extra=(--sets ruler --ruler-lengths 8192,16384,32768 --ruler-tasks niah_multiquery --ruler-format chat
                      --ruler-prefix none --limit 61);;
    rulermqin) extra=(--sets ruler --ruler-lengths 8192,16384,32768 --ruler-tasks niah_multiquery --ruler-format chat
                      --ruler-prefix instruct);;
    pplchat) extra=(--sets ppl --ppl-dataset chat-labelled-20261009);;
    pplfresh) extra=(--sets ppl --ppl-dataset fresh-long-20261008 --ppl-domains chat);;
    rulerinst) extra=(--sets ruler --ruler-lengths 8192,16384,32768 --ruler-format chat --ruler-prefix instruct);;
    ruler64inst) extra=(--sets ruler --ruler-lengths 65536 --ruler-tasks niah_single_1 --ruler-format chat
                        --ruler-prefix instruct);;
    rulerquick) extra=(--sets ruler --ruler-lengths 8192 --ruler-tasks niah_single_1);;
    rulerdiag) extra=(--sets ruler --ruler-lengths 8192,16384 --ruler-tasks niah_single_1);;
    speed) extra=(--sets speed --speed-runs 5);;
    control) extra=(--sets control --control-docs 6);;
  esac
  if [ $step = stfit ]; then   # on the running server: both passes per prompt, then the fit, then the files beside the projector
    log "START $arm $step:$pass"
    local F=/cache/stfit-2908
    $DOCKER exec $NAME /opt/vllm/bin/python3 /app/tools/ced/st_fit_driver.py --url http://127.0.0.1:$PORT \
        --model qwen3.8-flash-next --books-dir $F/books --n 13 --tokens 10000 --chunk 2048 --min-tail 512 >> $LOGS/$arm.log 2>&1 &&
      $DOCKER exec $NAME /opt/vllm/bin/python3 /app/tools/ced/fit_st.py $F/dump --out $F/ridgefill-s24-st2908 --min-pairs 20 >> $LOGS/$arm.log 2>&1 &&
      cp $CACHE/stfit-2908/ridgefill-s24-st2908.rank0.pt $CACHE/stfit-2908/ridgefill-s24-st2908.rank1.pt $PROJ/
    local rc=$?
    log "END $arm $step:$pass rc=$rc"
    [ $rc -eq 0 ] && touch $mark
    return $rc
  fi
  log "START $arm $step:$pass"
  nice -n 10 $PY -m c5.client --arm $arm --kva $kva --tail $TAIL --base-url $URL --out $OUT --pass $pass \
    --deadline $dl "${extra[@]}" >> $LOGS/$arm.log 2>&1
  local rc=$?
  log "END $arm $step:$pass rc=$rc"
  [ $rc -eq 0 ] && touch $mark
  return $rc
}

run_arm() { # arm steps... (deadline in DL)
  local arm=$1; shift
  preflight || return 1
  if ! boot $arm; then stop_server $arm; return 1; fi
  local s rc=0
  for s in "$@"; do run_step $arm $s ${DL:-0} || { rc=$?; break; }; done
  stop_server $arm
  [ -e $SAFE/KERNEL-ALERT ] && { log "KERNEL-ALERT: stopping the run"; return 1; }
  return $rc
}

run_arms() { # steps-map arm...: each arm until its steps are done (finished samples are skipped on a retry); a server
             # that dies of a recoverable fault is booted again; only KERNEL-ALERT or a 3rd failure in a row stops
  local -n map=$1; local name=$1; shift
  local a s todo ok attempt
  for a in "$@"; do
    todo=""; for s in ${map[$a]}; do [ -e $OUT/$a/.done-${s%%:*}-${s##*:} ] || todo="$todo $s"; done
    [ -z "$todo" ] && { log "arm $a done ($name)"; continue; }
    ok=0
    for attempt in 1 2 3; do
      run_arm $a $todo && { ok=1; break; }
      [ -e $SAFE/KERNEL-ALERT ] && { log "STOP: unrecoverable fault during $a"; return 1; }
      log "RETRY: arm $a failed (attempt $attempt), booting again"
      todo=""; for s in ${map[$a]}; do [ -e $OUT/$a/.done-${s%%:*}-${s##*:} ] || todo="$todo $s"; done
    done
    [ $ok = 1 ] || { log "STOP: arm $a failed 3 times"; return 1; }
  done
}

exec 9>$GPU_LOCK
case $MODE in
  status)
    for a in $ORDER; do echo "$a: $(ls -a $OUT/$a 2>/dev/null | grep '^.done' | tr '\n' ' ')"; done; exit 0;;
  smoke)
    flock -n 9 || { echo "gpu.lock is held"; exit 1; }
    preflight || exit 1
    trap 'stop_server graded' EXIT
    boot graded || exit 1
    nice -n 10 $PY -m c5.smoke --base-url $URL --tail $TAIL --out $RUNDIR/smoke $([ $MAX_LEN -lt 66560 ] && echo --no-64k) \
      2>&1 | tee -a $LOGS/smoke.log
    exit ${PIPESTATUS[0]};;
  run)
    flock -n 9 || { echo "gpu.lock is held"; exit 1; }
    [ -e $STATE/end_epoch ] || echo $(( $(date +%s) + 10 * 3600 )) > $STATE/end_epoch
    arms=${*:-$ORDER}
    for a in $arms; do
      [ -n "${STEPS[$a]+x}" ] || { log "unknown arm $a"; exit 2; }
      todo=""; for s in ${STEPS[$a]}; do [ -e $OUT/$a/.done-${s%%:*}-${s##*:} ] || todo="$todo $s"; done
      [ -z "$todo" ] && { log "arm $a done"; continue; }
      # a server that dies of a recoverable fault is booted again and the arm resumes (finished samples are skipped);
      # only an unrecoverable fault (KERNEL-ALERT, c5/kfilter.sh) or a 3rd failure in a row stops the run
      ok=0
      for attempt in 1 2 3; do
        run_arm $a $todo && { ok=1; break; }
        [ -e $SAFE/KERNEL-ALERT ] && { log "STOP: unrecoverable fault during $a"; exit 1; }
        log "RETRY: arm $a failed (attempt $attempt), booting again"
        todo=""; for s in ${STEPS[$a]}; do [ -e $OUT/$a/.done-${s%%:*}-${s##*:} ] || todo="$todo $s"; done
      done
      [ $ok = 1 ] || { log "STOP: arm $a failed 3 times"; exit 1; }
    done
    log "MANDATORY ARMS DONE"; exit 0;;
  fix)
    flock -n 9 || { echo "gpu.lock is held"; exit 1; }
    run_arms FIXPPL $FIXPPLORDER || exit 1
    log "FIX PERPLEXITY DONE"
    run_arms FIXRULER $FIXRULERORDER || exit 1
    log "FIX RULER DONE"; exit 0;;
  final)
    [ -n "$OVERLAY" ] && [ -n "${C5_RUN:-}" ] || { echo "final needs C5_ENGINE_OVERLAY (the fixed engine) and C5_RUN (a fresh run dir)"; exit 2; }
    flock -n 9 || { echo "gpu.lock is held"; exit 1; }
    log "FINAL start: run $RUNDIR, overlay $OVERLAY, code $(git -C "$KIT" rev-parse HEAD)"
    run_arms FINAL ${*:-$FINALORDER} || exit 1
    log "FINAL DONE"; exit 0;;
  extras)
    flock -n 9 || { echo "gpu.lock is held"; exit 1; }
    END=${C5_END:-$(cat $STATE/end_epoch)}
    for x in "${EXTRAS[@]}"; do
      IFS='|' read -r a st est <<< "$x"
      left=$(( (END - $(date +%s)) / 60 ))
      [ $left -ge $est ] || { log "SKIP extra $a ($st): ${left} min left < ${est}"; continue; }
      DL=$END run_arm $a $st || { log "STOP: extra $a failed"; exit 1; }
    done
    exit 0;;
  *) echo "usage: $0 smoke|run [ARM...]|fix|final [ARM...]|extras|status"; exit 2;;
esac
