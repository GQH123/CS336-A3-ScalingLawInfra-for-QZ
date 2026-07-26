#!/usr/bin/env bash
set -Eeuo pipefail

# Control-node admin operations for the scaling-laws assignment backend.
#
# Online commands call the running admin API through scaling_backend.admin_cli.
# Offline commands edit local control-node state files and should be run only
# after the backend has been stopped.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python3 or python is required." >&2
  exit 2
fi

if [[ -n "${SCALING_BACKEND_SOURCE_DIR:-}" && -d "$SCALING_BACKEND_SOURCE_DIR/scaling_backend" ]]; then
  export PYTHONPATH="$SCALING_BACKEND_SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
elif [[ -d "$REPO_ROOT/source/backend/scaling_backend" ]]; then
  export PYTHONPATH="$REPO_ROOT/source/backend${PYTHONPATH:+:$PYTHONPATH}"
fi

RUNTIME_DIR="${SCALING_CONTROL_RUNTIME_DIR:-${CONTROL_RUNTIME_DIR:-$SCRIPT_DIR}}"
export SCALING_API_CONFIG="${SCALING_API_CONFIG:-$RUNTIME_DIR/api-runtime.json}"

usage() {
  cat <<'EOF'
Usage:
  admin-ops.sh <command> [args...]

Online admin API commands:
  queue
  experiments [--student-id STUDENT] [--status STATUS]
  student-budget STUDENT
  experiment EXPERIMENT_ID
  cancel EXPERIMENT_ID --reason REASON --actor ACTOR
  poll EXPERIMENT_ID
  mark-system-failure EXPERIMENT_ID --failure-reason REASON --reason REASON --actor ACTOR [--refund-seconds N]
  poll-active
  import-worker-events
  adjust-budget STUDENT --seconds N --reason REASON --actor ACTOR [--experiment-id ID]
  freeze-finals --reason REASON --actor ACTOR
  launch-finals --max-runtime-seconds N --reason REASON --actor ACTOR
  poll-final FINAL_RUN_ID
  mark-final-system-failure FINAL_RUN_ID --failure-reason REASON --reason REASON --actor ACTOR
  cancel-final FINAL_RUN_ID --reason REASON --actor ACTOR
  exports [OUTPUT_DIR]
  cancel-active [--reason REASON] [--actor ACTOR]
  prepare-reset [--export-dir DIR] [--reason REASON] [--actor ACTOR]
  seed-student-keys --roster-csv ROSTER --output-csv OUT [--overwrite]

Offline file commands:
  backup-state [OUTPUT_DIR]
  reset-state --confirm [--allow-running] [--keep-worker-events] [--keep-manifests] [--no-empty-snapshot]

Environment:
  SCALING_API_CONFIG             Defaults to $CONTROL_RUNTIME_DIR/api-runtime.json.
  CONTROL_RUNTIME_DIR            Defaults to this script's directory.
  SCALING_ADMIN_BASE_URL         Defaults to http://127.0.0.1:${SCALING_API_PORT:-8000}.
  SCALING_ADMIN_API_TOKEN        Defaults to api.admin_api_token in SCALING_API_CONFIG.
  SCALING_STATE_SNAPSHOT_PATH    Defaults to api.state_snapshot_path or state-snapshot.json.
  SCALING_LOCAL_MANIFEST_DIR     Defaults to api.local_manifest_dir or frozen-manifests.
  SCALING_WORKER_EVENT_IMPORT_DIR Defaults to api.worker_event_import_dir when configured.

Reset workflow:
  1. Run: admin-ops.sh prepare-reset --reason "course reset" --actor STAFF
  2. Stop the backend.
  3. Run: admin-ops.sh reset-state --confirm
  4. Restart the backend.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

config_value() {
  local section="$1"
  local key="$2"
  "$PYTHON_BIN" - "$SCALING_API_CONFIG" "$section" "$key" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
section = sys.argv[2]
key = sys.argv[3]

try:
    config = json.loads(path.read_text(encoding="utf-8"))
except FileNotFoundError:
    sys.exit(0)

value = (config.get(section) or {}).get(key, "")
if value is None:
    value = ""
if isinstance(value, (str, int, float, bool)):
    print(str(value))
else:
    print(json.dumps(value, sort_keys=True))
PY
}

timestamp() {
  date -u +%Y%m%dT%H%M%SZ
}

api_config_value="${SCALING_ADMIN_API_TOKEN:-}"
if [[ -z "$api_config_value" ]]; then
  api_config_value="$(config_value api admin_api_token)"
fi
export SCALING_ADMIN_API_TOKEN="$api_config_value"
export SCALING_ADMIN_BASE_URL="${SCALING_ADMIN_BASE_URL:-http://127.0.0.1:${SCALING_API_PORT:-8000}}"

if [[ -z "${SCALING_STATE_SNAPSHOT_PATH:-}" ]]; then
  snapshot_config="$(config_value api state_snapshot_path)"
  export SCALING_STATE_SNAPSHOT_PATH="${snapshot_config:-$RUNTIME_DIR/state-snapshot.json}"
fi

if [[ -z "${SCALING_LOCAL_MANIFEST_DIR:-}" ]]; then
  manifest_config="$(config_value api local_manifest_dir)"
  export SCALING_LOCAL_MANIFEST_DIR="${manifest_config:-$RUNTIME_DIR/frozen-manifests}"
fi

if [[ -z "${SCALING_WORKER_EVENT_IMPORT_DIR:-}" ]]; then
  worker_event_config="$(config_value api worker_event_import_dir)"
  if [[ -n "$worker_event_config" ]]; then
    export SCALING_WORKER_EVENT_IMPORT_DIR="$worker_event_config"
  fi
fi

admin_cli() {
  "$PYTHON_BIN" -m scaling_backend.admin_cli \
    --base-url "$SCALING_ADMIN_BASE_URL" \
    --admin-token "$SCALING_ADMIN_API_TOKEN" \
    "$@"
}

api_reachable() {
  "$PYTHON_BIN" - "$SCALING_ADMIN_BASE_URL" "$SCALING_ADMIN_API_TOKEN" <<'PY'
from __future__ import annotations

import sys
import urllib.error
import urllib.request

base_url = sys.argv[1].rstrip("/")
token = sys.argv[2] if len(sys.argv) > 2 else ""
request = urllib.request.Request(base_url + "/admin/queue")
request.add_header("Authorization", f"Bearer {token}")

try:
    with urllib.request.urlopen(request, timeout=2.0):
        sys.exit(0)
except urllib.error.HTTPError:
    # A 401/403 still means the API is running; reset should not continue.
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

exports_command() {
  if [[ "$#" -eq 0 ]]; then
    local output_dir="$RUNTIME_DIR/exports/admin-export-$(timestamp)"
    admin_cli exports --output-dir "$output_dir"
  elif [[ "$#" -eq 1 && "$1" != -* ]]; then
    admin_cli exports --output-dir "$1"
  else
    admin_cli exports "$@"
  fi
}

cancel_active_command() {
  local reason="admin maintenance"
  local actor="${USER:-admin}"

  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --reason)
        [[ "$#" -ge 2 ]] || die "--reason requires a value"
        reason="$2"
        shift 2
        ;;
      --actor)
        [[ "$#" -ge 2 ]] || die "--actor requires a value"
        actor="$2"
        shift 2
        ;;
      -h|--help)
        cat <<'EOF'
Usage:
  admin-ops.sh cancel-active [--reason REASON] [--actor ACTOR]

Cancels every active exploratory experiment and hidden final run visible in the
admin queue. This requires a running backend and a valid admin token.
EOF
        return 0
        ;;
      *)
        die "unknown cancel-active option: $1"
        ;;
    esac
  done

  "$PYTHON_BIN" - "$SCALING_ADMIN_BASE_URL" "$SCALING_ADMIN_API_TOKEN" "$reason" "$actor" <<'PY'
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

base_url = sys.argv[1].rstrip("/")
admin_token = sys.argv[2]
reason = sys.argv[3]
actor = sys.argv[4]

headers = {
    "Authorization": f"Bearer {admin_token}",
    "Content-Type": "application/json",
}


def request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            detail = json.loads(body)
        except ValueError:
            detail = {"message": body}
        raise RuntimeError(
            json.dumps(
                {
                    "method": method,
                    "path": path,
                    "status_code": exc.code,
                    **detail,
                },
                sort_keys=True,
            )
        ) from exc
    if not body:
        return {}
    return json.loads(body)


active_statuses = ("submitted", "queued", "running")
experiments: dict[str, dict[str, Any]] = {}
for status in active_statuses:
    body = request(
        "GET",
        "/admin/experiments?" + urllib.parse.urlencode({"status": status}),
    )
    for row in body.get("experiments", []):
        experiment_id = str(row.get("experiment_id", ""))
        if experiment_id:
            experiments[experiment_id] = row

queue = request("GET", "/admin/queue")
final_runs = {
    str(row.get("final_run_id", "")): row
    for row in queue.get("final", [])
    if row.get("final_run_id")
}

cancelled_experiments: list[str] = []
cancelled_final_runs: list[str] = []
failures: list[dict[str, str]] = []

for experiment_id in sorted(experiments):
    try:
        request(
            "POST",
            f"/admin/experiments/{urllib.parse.quote(experiment_id, safe='')}/cancel",
            {"reason": reason, "actor": actor},
        )
        cancelled_experiments.append(experiment_id)
    except RuntimeError as exc:
        failures.append({"id": experiment_id, "error": str(exc)})

for final_run_id in sorted(final_runs):
    try:
        request(
            "POST",
            f"/admin/final-runs/{urllib.parse.quote(final_run_id, safe='')}/cancel",
            {"reason": reason, "actor": actor},
        )
        cancelled_final_runs.append(final_run_id)
    except RuntimeError as exc:
        failures.append({"id": final_run_id, "error": str(exc)})

summary = {
    "cancelled_experiments": cancelled_experiments,
    "cancelled_final_runs": cancelled_final_runs,
    "experiment_count": len(cancelled_experiments),
    "final_run_count": len(cancelled_final_runs),
    "failures": failures,
}
print(json.dumps(summary, sort_keys=True))
sys.exit(1 if failures else 0)
PY
}

prepare_reset_command() {
  local reason="course reset"
  local actor="${USER:-admin}"
  local export_dir="$RUNTIME_DIR/exports/pre-reset-$(timestamp)"

  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --reason)
        [[ "$#" -ge 2 ]] || die "--reason requires a value"
        reason="$2"
        shift 2
        ;;
      --actor)
        [[ "$#" -ge 2 ]] || die "--actor requires a value"
        actor="$2"
        shift 2
        ;;
      --export-dir)
        [[ "$#" -ge 2 ]] || die "--export-dir requires a value"
        export_dir="$2"
        shift 2
        ;;
      -h|--help)
        cat <<'EOF'
Usage:
  admin-ops.sh prepare-reset [--export-dir DIR] [--reason REASON] [--actor ACTOR]

Exports course records and cancels active exploratory/final jobs through the
running admin API. After this succeeds, stop the backend and run:
  admin-ops.sh reset-state --confirm
EOF
        return 0
        ;;
      *)
        die "unknown prepare-reset option: $1"
        ;;
    esac
  done

  echo "exporting course records to $export_dir"
  admin_cli exports --output-dir "$export_dir"
  echo "cancelling active runs"
  cancel_active_command --reason "$reason" --actor "$actor"
  echo "prepare-reset complete; stop the backend, then run: $0 reset-state --confirm"
}

backup_state_command() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  admin-ops.sh backup-state [OUTPUT_DIR]

Copies the state snapshot, frozen manifests, and worker-event import directory
into OUTPUT_DIR. When OUTPUT_DIR is omitted, a timestamped directory under the
control runtime backups directory is used.
EOF
    return 0
  fi

  local backup_dir="${1:-$RUNTIME_DIR/backups/course-state-$(timestamp)}"
  [[ "$#" -le 1 ]] || die "backup-state accepts at most one output directory"

  mkdir -p "$backup_dir"
  if [[ -f "$SCALING_STATE_SNAPSHOT_PATH" ]]; then
    cp -p "$SCALING_STATE_SNAPSHOT_PATH" "$backup_dir/state-snapshot.json"
  fi
  if [[ -d "$SCALING_LOCAL_MANIFEST_DIR" ]]; then
    cp -a "$SCALING_LOCAL_MANIFEST_DIR" "$backup_dir/frozen-manifests"
  fi
  if [[ -n "${SCALING_WORKER_EVENT_IMPORT_DIR:-}" && -d "$SCALING_WORKER_EVENT_IMPORT_DIR" ]]; then
    cp -a "$SCALING_WORKER_EVENT_IMPORT_DIR" "$backup_dir/worker-events"
  fi

  echo "backup complete: $backup_dir"
}

write_empty_snapshot() {
  local snapshot_path="$1"
  "$PYTHON_BIN" - "$snapshot_path" <<'PY'
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "snapshot_version": 1,
    "experiments": [],
    "budgets": {},
    "final_submissions": {},
    "frozen_final_submissions": {},
    "final_submissions_frozen": False,
    "final_runs_launched": False,
    "final_runs": [],
    "budget_adjustments": [],
    "admin_actions": [],
    "experiment_events": [],
    "worker_events": [],
    "dispatcher_records": [],
    "counters": {
        "next_experiment_number": 1,
        "next_final_submission_number": 1,
        "next_final_run_number": 1,
        "next_budget_adjustment_number": 1,
        "next_admin_action_number": 1,
    },
}
handle_fd, tmp_name = tempfile.mkstemp(
    prefix=f"{path.name}.",
    suffix=".tmp",
    dir=str(path.parent),
)
try:
    with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_name, path)
except Exception:
    try:
        os.unlink(tmp_name)
    except OSError:
        pass
    raise
PY
}

archive_matching_files() {
  local directory="$1"
  local pattern="$2"
  local label="$3"
  local stamp="$4"

  [[ -n "$directory" && -d "$directory" ]] || return 0

  local files=()
  shopt -s nullglob
  files=("$directory"/$pattern)
  shopt -u nullglob

  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "no $label files to archive in $directory"
    return 0
  fi

  local archive_dir="${directory}.archive-${stamp}"
  mkdir -p "$archive_dir"
  mv "${files[@]}" "$archive_dir"/
  echo "archived $label files: $archive_dir"
}

reset_state_command() {
  local confirmed=0
  local allow_running=0
  local keep_worker_events=0
  local keep_manifests=0
  local no_empty_snapshot=0

  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --confirm)
        confirmed=1
        shift
        ;;
      --allow-running)
        allow_running=1
        shift
        ;;
      --keep-worker-events)
        keep_worker_events=1
        shift
        ;;
      --keep-manifests)
        keep_manifests=1
        shift
        ;;
      --no-empty-snapshot)
        no_empty_snapshot=1
        shift
        ;;
      -h|--help)
        cat <<'EOF'
Usage:
  admin-ops.sh reset-state --confirm [--allow-running] [--keep-worker-events] [--keep-manifests] [--no-empty-snapshot]

Archives the current state snapshot, worker-event JSONL imports, and frozen
manifest JSON files, then writes a clean empty snapshot. Run this after the
backend is stopped unless you deliberately pass --allow-running.
EOF
        return 0
        ;;
      *)
        die "unknown reset-state option: $1"
        ;;
    esac
  done

  if [[ "$confirmed" -ne 1 ]]; then
    echo "ERROR: reset-state requires --confirm." >&2
    return 2
  fi

  if [[ "$allow_running" -ne 1 ]] && api_reachable; then
    cat >&2 <<EOF
ERROR: API appears reachable at $SCALING_ADMIN_BASE_URL.
Stop the backend before reset-state, or pass --allow-running if you have already drained traffic and accept the risk.
EOF
    return 2
  fi

  local stamp
  stamp="$(timestamp)-$$"

  if [[ -f "$SCALING_STATE_SNAPSHOT_PATH" ]]; then
    local snapshot_archive="${SCALING_STATE_SNAPSHOT_PATH}.pre-reset-${stamp}"
    mkdir -p "$(dirname "$snapshot_archive")"
    mv "$SCALING_STATE_SNAPSHOT_PATH" "$snapshot_archive"
    echo "archived state snapshot: $snapshot_archive"
  else
    echo "no state snapshot found at $SCALING_STATE_SNAPSHOT_PATH"
  fi

  if [[ "$no_empty_snapshot" -ne 1 ]]; then
    write_empty_snapshot "$SCALING_STATE_SNAPSHOT_PATH"
    echo "wrote empty state snapshot: $SCALING_STATE_SNAPSHOT_PATH"
  fi

  if [[ "$keep_worker_events" -ne 1 && -n "${SCALING_WORKER_EVENT_IMPORT_DIR:-}" ]]; then
    archive_matching_files "$SCALING_WORKER_EVENT_IMPORT_DIR" "*.jsonl" "worker-event" "$stamp"
    mkdir -p "$SCALING_WORKER_EVENT_IMPORT_DIR"
  fi

  if [[ "$keep_manifests" -ne 1 ]]; then
    archive_matching_files "$SCALING_LOCAL_MANIFEST_DIR" "*.json" "manifest" "$stamp"
    mkdir -p "$SCALING_LOCAL_MANIFEST_DIR"
  fi

  echo "reset complete"
}

if [[ "$#" -eq 0 ]]; then
  usage
  exit 0
fi

command="$1"
shift

case "$command" in
  -h|--help|help)
    usage
    ;;
  queue|experiments|student-budget|experiment|cancel|poll|mark-system-failure|poll-active|import-worker-events|adjust-budget|freeze-finals|launch-finals|poll-final|mark-final-system-failure|cancel-final|seed-student-keys)
    admin_cli "$command" "$@"
    ;;
  exports)
    exports_command "$@"
    ;;
  cancel-active)
    cancel_active_command "$@"
    ;;
  prepare-reset)
    prepare_reset_command "$@"
    ;;
  backup-state)
    backup_state_command "$@"
    ;;
  reset-state)
    reset_state_command "$@"
    ;;
  *)
    die "unknown command: $command"
    ;;
esac
