#!/bin/bash
# build.sh CODE_TAG [IMAGE] [--check]: build the finals image from the harness at CODE_TAG.
#
# The context is assembled from an explicit allowlist (git archive of CODE_TAG limited to the harness
# files, plus this finals/ dir), so results/, data/, and datasets/test-set-locked (whose texts are
# committed in the source repo) can never ride inside the image. --check stops after assembling the
# context and running lib/check_dockerfile.py against it (no docker daemon needed: the offline check
# test_pipeline.sh runs); without --check the image is built with ${DOCKER:-docker} (or podman).
#
# IMAGE defaults to kva-finals:<CODE_TAG>. TORCH_VERSION overrides the pinned CUDA wheel.
set -euo pipefail

FINALS=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$FINALS/.." && pwd)

die() { echo "build.sh: refusing: $*" >&2; exit 2; }
[ $# -ge 1 ] || die "usage: build.sh CODE_TAG [IMAGE] [--check]  (CODE_TAG = the harness tag, e.g. protocol-v1.1)"
TAG=$1
IMAGE=${2:-kva-finals:$TAG}
CHECK=0
[ "${3:-}" = "--check" ] && CHECK=1
DOCKER=${DOCKER:-docker}
command -v "$DOCKER" >/dev/null || DOCKER=$(command -v podman || true)
[ -n "$DOCKER" ] || die "no docker or podman found (DOCKER=... to name one)"

cd "$REPO"
git rev-parse --is-inside-work-tree >/dev/null || die "$REPO is not a git repo: the image is built from a tagged commit"

# the tag must exist for a real build; --check (structure only) may use HEAD
REV=$TAG
if ! git rev-parse -q --verify "$TAG^{commit}" >/dev/null; then
  if [ "$CHECK" = 1 ]; then
    echo "build.sh: WARNING: tag $TAG does not exist yet (Phase 1 tags the harness); --check uses HEAD" >&2
    REV=HEAD
  else
    die "git tag $TAG not found: the image is built from the tagged harness commit (tag it first)"
  fi
fi

CTX=$(mktemp -d /tmp/kva-finals-ctx.XXXXXX)
trap 'rm -rf "$CTX"' EXIT
# the allowlist: the reference harness and its inputs; NOTHING else (see finals/Dockerfile)
git archive "$REV" kva scripts configs tests requirements.txt protocol.json paper/PROTOCOL.md \
  datasets/corpus datasets/derived datasets/graded-and-calibration-sets datasets/calibration-exclude.json \
  README.md ledger/paper/ledger.jsonl jobs/paper-refit-clean.jobs | tar -x -C "$CTX"
git rev-parse "$REV" > "$CTX/CODE_COMMIT"      # kva/prov.py reads it where there is no .git (the image)
rsync -a --exclude '__pycache__' "$FINALS/" "$CTX/finals/"

# fail closed: nothing big or private may be in the context (models, locked texts, results, data)
BAD=$(find "$CTX" -type f \( -name '*.safetensors' -o -size +20M \) -printf '%P\n')
[ -z "$BAD" ] || die "the build context contains file(s) it must not: $BAD"
[ ! -e "$CTX/datasets/test-set-locked" ] || die "datasets/test-set-locked is in the context (locked texts are mounted, never baked)"
[ ! -e "$CTX/datasets/fresh-long-20261008" ] || die "datasets/fresh-long-20261008 is in the context (its texts are mounted, never baked)"
[ ! -e "$CTX/results" ] || die "results/ is in the context"
[ ! -e "$CTX/data" ] || die "data/ is in the context"

"$FINALS/lib/check_dockerfile.py" "$FINALS/Dockerfile" "$CTX"
echo "build.sh: context OK ($(du -sh "$CTX" | cut -f1)) at $REV"

if [ "$CHECK" = 1 ]; then
  echo "build.sh: --check passed (no image built)"
  exit 0
fi
"$DOCKER" build -f "$FINALS/Dockerfile" -t "$IMAGE" --build-arg TORCH_VERSION="${TORCH_VERSION:-2.11.0}" "$CTX"
echo "build.sh: built $IMAGE at $TAG"