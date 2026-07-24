#!/usr/bin/env bash
set -Eeuo pipefail

# Launch the control-node API from a manually assembled runtime directory.
#
# Required before launch:
#   SCALING_CALLBACK_URL should be reachable from QZ worker containers, for
#   example through the WebSocket port-forward URL for this control node.
#   Shared-storage JSONL worker events remain available only when explicitly
#   configured with SCALING_WORKER_EVENT_IMPORT_DIR and QZ_WORKER_EVENT_LOG_DIR.

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SCALING_API_CONFIG="${SCALING_API_CONFIG:-$runtime_dir/api-runtime.json}"
export SCALING_STUDENT_KEYS_CSV="${SCALING_STUDENT_KEYS_CSV:-$runtime_dir/student_api_keys.csv}"
export SCALING_LOCAL_MANIFEST_DIR="${SCALING_LOCAL_MANIFEST_DIR:-$runtime_dir/frozen-manifests}"
export SCALING_STATE_SNAPSHOT_PATH="${SCALING_STATE_SNAPSHOT_PATH:-$runtime_dir/state-snapshot.json}"

config_value() {
  python - "$SCALING_API_CONFIG" "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
section = config.get(sys.argv[2]) or {}
print(str(section.get(sys.argv[3]) or ""))
PY
}

if [[ -z "${SCALING_CALLBACK_URL:-}" ]]; then
  export SCALING_CALLBACK_URL="$(config_value api callback_url)"
fi

if [[ -z "${SCALING_WORKER_EVENT_IMPORT_DIR:-}" ]]; then
  SCALING_WORKER_EVENT_IMPORT_DIR="$(config_value api worker_event_import_dir)"
  if [[ -n "$SCALING_WORKER_EVENT_IMPORT_DIR" ]]; then
    export SCALING_WORKER_EVENT_IMPORT_DIR
  fi
fi

if [[ -z "${QZ_WORKER_EVENT_LOG_DIR:-}" ]]; then
  QZ_WORKER_EVENT_LOG_DIR="$(config_value qz worker_event_log_dir)"
  if [[ -n "$QZ_WORKER_EVENT_LOG_DIR" ]]; then
    export QZ_WORKER_EVENT_LOG_DIR
  elif [[ -n "${SCALING_WORKER_EVENT_IMPORT_DIR:-}" ]]; then
    export QZ_WORKER_EVENT_LOG_DIR="$SCALING_WORKER_EVENT_IMPORT_DIR"
  fi
fi

case "$SCALING_CALLBACK_URL" in
  http://api.internal/*|https://api.internal/*|http://backend.internal/*|https://backend.internal/*)
    if [[ -z "${SCALING_WORKER_EVENT_IMPORT_DIR:-}" || -z "${QZ_WORKER_EVENT_LOG_DIR:-}" ]]; then
      cat >&2 <<'EOF'
ERROR: SCALING_CALLBACK_URL still points at an internal placeholder host.
QZ worker containers must be able to resolve and reach the callback URL unless
shared-storage worker event JSONL is configured.

Set a reachable callback URL, for example:
  export SCALING_CALLBACK_URL="http://<control-node-ip-or-forward-host>:8000/internal/provider-events"

Or set both shared-storage event directories:
  export SCALING_WORKER_EVENT_IMPORT_DIR="/shared/course/worker-events"
  export QZ_WORKER_EVENT_LOG_DIR="/shared/course/worker-events"
EOF
      exit 2
    fi
    ;;
esac

python -m scaling_backend.preflight \
  --config "$SCALING_API_CONFIG" \
  --create-dirs \
  --probe-provider \
  --output-json "$runtime_dir/preflight.json"

uvicorn scaling_backend.asgi:app --host 0.0.0.0 --port "${SCALING_API_PORT:-8000}"
