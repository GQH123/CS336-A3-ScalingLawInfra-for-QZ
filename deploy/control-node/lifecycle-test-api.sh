#!/usr/bin/env bash
set -Eeuo pipefail

# Inspectively test control-node API lifecycle wiring with a packaged lifecycle
# dataset.
#
# Required environment:
#   LIFECYCLE_DATASET_DIR=/path/to/lifecycle_test_dataset
#
# Remote API mode, the default, also requires:
#   SCALING_API_BASE_URL=http://127.0.0.1:8000
#   SCALING_API_KEY=<one student key>
#
# Useful optional environment:
#   LIFECYCLE_API_MODE=remote-api|local-service
#   LIFECYCLE_CONTROL_RUNTIME_DIR=/path/to/assignment-3-control-runtime
#   SCALING_LOCAL_MANIFEST_DIR=/path/to/frozen-manifests
#   LIFECYCLE_API_UNIQUE_MODEL_SEED=1  # default; set 0 to reuse duplicate configs
#   LIFECYCLE_REQUESTED_RUNTIME_SECONDS=300
#
# local-service mode does not contact QZ or a running API. It builds the API
# service in-process with the fake provider and verifies that lifecycle runtime
# overrides freeze into the worker manifest.

log() {
  printf '[lifecycle-control] %s\n' "$*"
}

fail() {
  printf '[lifecycle-control] ERROR: %s\n' "$*" >&2
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

if [[ -d "$REPO_ROOT/source/backend" ]]; then
  export PYTHONPATH="$REPO_ROOT/source/backend:${PYTHONPATH:-}"
fi

DATASET_DIR="${LIFECYCLE_DATASET_DIR:-}"
[[ -n "$DATASET_DIR" ]] || fail "LIFECYCLE_DATASET_DIR is required"

RUNTIME_DIR="${LIFECYCLE_CONTROL_RUNTIME_DIR:-${CONTROL_RUNTIME_DIR:-}}"
if [[ -z "$RUNTIME_DIR" && -f "$SCRIPT_DIR/api-runtime.json" ]]; then
  RUNTIME_DIR="$SCRIPT_DIR"
fi

if [[ -n "$RUNTIME_DIR" ]]; then
  if [[ -z "${SCALING_API_CONFIG:-}" && -f "$RUNTIME_DIR/api-runtime.json" ]]; then
    export SCALING_API_CONFIG="$RUNTIME_DIR/api-runtime.json"
  fi
  if [[ -z "${SCALING_LOCAL_MANIFEST_DIR:-}" ]]; then
    export SCALING_LOCAL_MANIFEST_DIR="$RUNTIME_DIR/frozen-manifests"
  fi
fi

if [[ -f "$DATASET_DIR/api-runtime-overrides.env" ]]; then
  # shellcheck disable=SC1090
  source "$DATASET_DIR/api-runtime-overrides.env"
fi

MODE="${LIFECYCLE_API_MODE:-remote-api}"
args=(
  -m scaling_backend.lifecycle_checks
  control
  --dataset-dir "$DATASET_DIR"
  --mode "$MODE"
  --api-base-url "${SCALING_API_BASE_URL:-}"
  --api-key "${SCALING_API_KEY:-}"
  --local-manifest-dir "${SCALING_LOCAL_MANIFEST_DIR:-}"
  --student-id "${LIFECYCLE_STUDENT_ID:-staff-lifecycle}"
  --request-timeout-seconds "${LIFECYCLE_REQUEST_TIMEOUT_SECONDS:-30}"
)

if [[ -n "${LIFECYCLE_REQUESTED_RUNTIME_SECONDS:-}" ]]; then
  args+=(--requested-runtime-seconds "$LIFECYCLE_REQUESTED_RUNTIME_SECONDS")
fi

case "${LIFECYCLE_API_UNIQUE_MODEL_SEED:-1}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--unique-model-seed)
    ;;
esac

case "${LIFECYCLE_API_POLL_ONCE:-1}" in
  0|false|FALSE|no|NO|off|OFF)
    args+=(--no-poll-once)
    ;;
esac

log "dataset: $DATASET_DIR"
log "mode: $MODE"
if [[ "$MODE" == "remote-api" ]]; then
  log "api base url: ${SCALING_API_BASE_URL:-<unset>}"
  log "local manifest dir: ${SCALING_LOCAL_MANIFEST_DIR:-<unset>}"
fi

exec "$PYTHON_BIN" "${args[@]}"
