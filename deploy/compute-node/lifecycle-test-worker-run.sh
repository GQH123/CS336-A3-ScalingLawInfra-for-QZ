#!/usr/bin/env bash
set -Eeuo pipefail

# Inspectively test the real compute-node worker path with a packaged lifecycle
# dataset.
#
# Required environment:
#   LIFECYCLE_DATASET_DIR=/path/to/lifecycle_test_dataset
#
# Useful optional environment:
#   LIFECYCLE_WORK_DIR=/path/to/output-work-dir
#   LIFECYCLE_SKIP_WORKER_RUN=1
#   LIFECYCLE_TOKENIZED_INDEX_VALIDATION_MODE=metadata|full
#   LIFECYCLE_TRAINER=course_trainer.worker:train
#   LIFECYCLE_EVENT_SINK=jsonl|http|both
#   LIFECYCLE_CALLBACK_URL=http://<control-node>/internal/provider-events
#   LIFECYCLE_CALLBACK_TOKEN_ENV=SCALING_CALLBACK_TOKEN
#   LIFECYCLE_REQUESTED_RUNTIME_SECONDS=300
#
# Default behavior runs python -m scaling_backend.worker.run with the real
# course_trainer.worker:train callable. Set LIFECYCLE_SKIP_WORKER_RUN=1 when you
# only want to inspect lifecycle indexes and write the frozen worker manifest.

log() {
  printf '[lifecycle-compute] %s\n' "$*"
}

fail() {
  printf '[lifecycle-compute] ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

for package_dir in \
  "$REPO_ROOT/source/backend" \
  "$REPO_ROOT/source/training" \
  "$REPO_ROOT/source/data" \
  "$REPO_ROOT/refs/cs336-assignment3-scaling"
do
  if [[ -d "$package_dir" ]]; then
    export PYTHONPATH="$package_dir:${PYTHONPATH:-}"
  fi
done

DATASET_DIR="${LIFECYCLE_DATASET_DIR:-}"
[[ -n "$DATASET_DIR" ]] || fail "LIFECYCLE_DATASET_DIR is required"

if [[ -z "${LIFECYCLE_WORK_DIR:-}" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  LIFECYCLE_WORK_DIR="$DATASET_DIR/test-runs/compute-$timestamp"
fi

RUN_ID="${LIFECYCLE_EXPERIMENT_ID:-exp-lifecycle-000001}"
STUDENT_ID="${LIFECYCLE_STUDENT_ID:-staff-lifecycle}"
CALLBACK_URL="${LIFECYCLE_CALLBACK_URL:-http://127.0.0.1/internal/provider-events}"
CALLBACK_TOKEN_ENV="${LIFECYCLE_CALLBACK_TOKEN_ENV:-SCALING_CALLBACK_TOKEN}"
TRAINER="${LIFECYCLE_TRAINER:-course_trainer.worker:train}"
HEARTBEAT_INTERVAL="${LIFECYCLE_HEARTBEAT_INTERVAL_SECONDS:-0}"
VALIDATION_MODE="${LIFECYCLE_TOKENIZED_INDEX_VALIDATION_MODE:-metadata}"
EVENT_SINK="${LIFECYCLE_EVENT_SINK:-jsonl}"
EVENT_LOG_PATH="${LIFECYCLE_EVENT_LOG_PATH:-}"

if [[ -z "${!CALLBACK_TOKEN_ENV:-}" ]]; then
  export "$CALLBACK_TOKEN_ENV"="dev-lifecycle-callback-token"
fi

args=(
  -m scaling_backend.lifecycle_checks
  compute
  --dataset-dir "$DATASET_DIR"
  --work-dir "$LIFECYCLE_WORK_DIR"
  --run-id "$RUN_ID"
  --student-id "$STUDENT_ID"
  --callback-url "$CALLBACK_URL"
  --callback-token-env "$CALLBACK_TOKEN_ENV"
  --trainer "$TRAINER"
  --heartbeat-interval-seconds "$HEARTBEAT_INTERVAL"
  --tokenized-index-validation-mode "$VALIDATION_MODE"
  --event-sink "$EVENT_SINK"
)

if [[ -n "$EVENT_LOG_PATH" ]]; then
  args+=(--event-log-path "$EVENT_LOG_PATH")
fi

if [[ -n "${LIFECYCLE_REQUESTED_RUNTIME_SECONDS:-}" ]]; then
  args+=(--requested-runtime-seconds "$LIFECYCLE_REQUESTED_RUNTIME_SECONDS")
fi

case "${LIFECYCLE_SKIP_WORKER_RUN:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--skip-worker-run)
    ;;
esac

log "dataset: $DATASET_DIR"
log "work dir: $LIFECYCLE_WORK_DIR"
log "validation mode: $VALIDATION_MODE"
log "event sink: $EVENT_SINK"
if [[ "${LIFECYCLE_SKIP_WORKER_RUN:-0}" == "1" ]]; then
  log "worker run: skipped"
else
  log "worker run: enabled with trainer $TRAINER"
fi

exec "$PYTHON_BIN" "${args[@]}"
