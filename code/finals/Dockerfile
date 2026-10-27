# The finals image: the repo's reference harness + the official scorers, pinned deps; NO models and
# NO locked-set texts baked in (models are pulled from HF at start, the locked set and the projectors
# are mounted; see finals/README.md). Built ONLY by finals/build.sh, which assembles the context from
# an explicit allowlist at the harness's code tag (protocol-v1.1) so that no results/, no data/ and no
# datasets/test-set-locked ever ride along.
FROM python:3.12-slim

# the repo's pinned CPU-side deps (requirements.txt) + one pinned torch (the CUDA wheel; the repo's
# requirements.txt leaves torch to the platform - scripts/setup_env.sh - so it is pinned here)
ARG TORCH_VERSION=2.11.0

# push.sh needs rsync + ssh; verify/jq + the repush cron need jq and cron
RUN apt-get update && apt-get install -y --no-install-recommends \
      rsync openssh-client jq cron ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    KVA_DETERMINISTIC=1 \
    KVA_GPU_VENDOR=nvidia \
    KVA_DATA=/data \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    KVA_REPO=/opt/kva \
    KVA_MODEL_DIR=/data/models \
    KVA_PROJ_DIR=/data/proj \
    KVA_LOCKED_DIR=/data/locked-set \
    FINALS_RESULTS=/data/results/finals \
    HF_HOME=/data/hf

WORKDIR /opt/kva
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==${TORCH_VERSION}

# the harness (kva/, scripts/, configs/, tests/, the canary corpus, the derived-set manifests (ids and hashes
# only), the committed calibration inputs + exclusion list and the refit job (the clean refit runs on the box first:
# finals/refit_clean.sh), the protocol, the ledger, the commit it was built from) and this package; nothing else
# under datasets/ is copied: the locked
# set is mounted at /data/locked-set (KVA_LOCKED_DIR), models land in /data/models
COPY kva/ kva/
COPY scripts/ scripts/
COPY configs/ configs/
COPY tests/ tests/
COPY datasets/corpus/ datasets/corpus/
COPY datasets/derived/ datasets/derived/
COPY datasets/graded-and-calibration-sets/ datasets/graded-and-calibration-sets/
COPY datasets/calibration-exclude.json datasets/calibration-exclude.json
COPY jobs/paper-refit-clean.jobs jobs/paper-refit-clean.jobs
COPY CODE_COMMIT CODE_COMMIT
COPY protocol.json protocol.json
COPY paper/PROTOCOL.md paper/PROTOCOL.md
COPY ledger/paper/ledger.jsonl ledger/paper/ledger.jsonl
COPY README.md README.md
COPY finals/ finals/

# the locked set, projectors, models, HF cache and results live on the mounted volume
VOLUME /data

ENTRYPOINT ["/opt/kva/finals/run_final.sh"]
CMD ["--help"]