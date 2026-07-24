#!/usr/bin/env bash
set -Eeuo pipefail

# Smoke-test the control-node public API by submitting a small exploratory run.
#
# Required environment:
#   SCALING_API_BASE_URL  Public/control API base URL, e.g. http://127.0.0.1:8000
#   SCALING_API_KEY       One student API key from student_api_keys.csv
#
# Useful optional environment:
#   SMOKE_REQUESTED_RUNTIME_SECONDS=300
#   SMOKE_TRAIN_TOKENS=4096
#   SMOKE_LEARNING_RATE=3e-4
#   SMOKE_NUM_HIDDEN_LAYERS=2
#   SMOKE_HIDDEN_SIZE=128
#   SMOKE_MODEL_SEED=$(date +%s)
#   SMOKE_CONFIG_FILE=/path/to/config-or-submit-payload.json
#   SMOKE_CONFIG_JSON='{"model": {...}, "training": {...}}'
#   SMOKE_WAIT_FOR_TERMINAL=0
#   SMOKE_MAX_POLLS=60
#   SMOKE_POLL_INTERVAL_SECONDS=10
#
# The default config is fixed, so rerunning this script may return
# 409 duplicate_config. That is treated as success: the script reuses the
# existing experiment_id and polls it. Set SMOKE_MODEL_SEED to a new value, or
# pass SMOKE_CONFIG_JSON/SMOKE_CONFIG_FILE, when you intentionally want a fresh
# experiment.

log() {
  printf '[control-smoke] %s\n' "$*"
}

fail() {
  printf '[control-smoke] ERROR: %s\n' "$*" >&2
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

[[ -n "${SCALING_API_BASE_URL:-}" ]] || fail "SCALING_API_BASE_URL is required"
[[ -n "${SCALING_API_KEY:-}" ]] || fail "SCALING_API_KEY is required"

log "api base url: ${SCALING_API_BASE_URL%/}"
log "python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
log "python version: $("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])')"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import requests


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "system_failed"}
FAILURE_STATUSES = {"failed", "cancelled", "system_failed"}


def log(message: str) -> None:
    print(f"[control-smoke] {message}", flush=True)


def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc
    return value


def float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number, got {raw!r}") from exc
    return value


def api_request(
    method: str,
    path: str,
    *,
    payload: Mapping[str, Any] | None = None,
    accept_duplicate: bool = False,
) -> tuple[int, dict[str, Any]]:
    response = requests.request(
        method,
        f"{BASE_URL}{path}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"raw_response": response.text}

    if response.status_code == 409 and accept_duplicate:
        if body.get("error") == "duplicate_config" and body.get("experiment_id"):
            return response.status_code, body

    if response.status_code < 200 or response.status_code >= 300:
        print(json.dumps(body, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)

    if not isinstance(body, dict):
        raise SystemExit(f"{method} {path} did not return a JSON object: {body!r}")
    return response.status_code, body


def load_config_and_runtime() -> tuple[dict[str, Any], int]:
    runtime = int_env("SMOKE_REQUESTED_RUNTIME_SECONDS", 300)
    config_file = os.environ.get("SMOKE_CONFIG_FILE", "").strip()
    config_json = os.environ.get("SMOKE_CONFIG_JSON", "").strip()
    if config_file and config_json:
        raise SystemExit("Set only one of SMOKE_CONFIG_FILE or SMOKE_CONFIG_JSON")

    loaded: Any | None = None
    if config_file:
        loaded = json.loads(Path(config_file).read_text(encoding="utf-8"))
    elif config_json:
        loaded = json.loads(config_json)

    if loaded is not None:
        if not isinstance(loaded, dict):
            raise SystemExit("Smoke config must be a JSON object")
        if "config" in loaded:
            config = loaded["config"]
            runtime = int(loaded.get("requested_runtime_seconds", runtime))
        else:
            config = loaded
        if not isinstance(config, dict):
            raise SystemExit("Smoke config payload must contain a JSON object config")
        return config, runtime

    training = {
        "train_tokens": int_env("SMOKE_TRAIN_TOKENS", 4096),
        "learning_rate": float_env("SMOKE_LEARNING_RATE", 3e-4),
    }
    model_seed = os.environ.get("SMOKE_MODEL_SEED", "").strip()
    if model_seed:
        training["model_seed"] = int(model_seed)

    return {
        "model": {
            "num_hidden_layers": int_env("SMOKE_NUM_HIDDEN_LAYERS", 2),
            "hidden_size": int_env("SMOKE_HIDDEN_SIZE", 128),
        },
        "training": training,
    }, runtime


BASE_URL = os.environ["SCALING_API_BASE_URL"].rstrip("/")
API_KEY = os.environ["SCALING_API_KEY"]
REQUEST_TIMEOUT_SECONDS = float_env("SMOKE_REQUEST_TIMEOUT_SECONDS", 30.0)
POLL_INTERVAL_SECONDS = float_env("SMOKE_POLL_INTERVAL_SECONDS", 10.0)
MAX_POLLS = int_env("SMOKE_MAX_POLLS", 60)
WAIT_FOR_TERMINAL = parse_bool_env("SMOKE_WAIT_FOR_TERMINAL", default=False)

if MAX_POLLS <= 0:
    raise SystemExit("SMOKE_MAX_POLLS must be positive")
if POLL_INTERVAL_SECONDS < 0:
    raise SystemExit("SMOKE_POLL_INTERVAL_SECONDS must be non-negative")

config, requested_runtime_seconds = load_config_and_runtime()
if requested_runtime_seconds <= 0:
    raise SystemExit("SMOKE_REQUESTED_RUNTIME_SECONDS must be positive")

log("checking budget")
_, budget_before = api_request("GET", "/budget")

submit_payload = {
    "config": config,
    "requested_runtime_seconds": requested_runtime_seconds,
}

log("submitting exploratory run")
submit_status, submit_result = api_request(
    "POST",
    "/submit",
    payload=submit_payload,
    accept_duplicate=True,
)

duplicate = submit_status == 409
experiment_id = str(submit_result.get("experiment_id", "")).strip()
if not experiment_id:
    raise SystemExit(f"submit response did not include experiment_id: {submit_result}")

if duplicate:
    log(f"duplicate config; reusing existing experiment_id={experiment_id}")
else:
    log(f"submitted experiment_id={experiment_id}")

poll_history: list[dict[str, Any]] = []

def poll_once() -> dict[str, Any]:
    _, experiment = api_request("GET", f"/experiment/{experiment_id}")
    status = str(experiment.get("status", ""))
    log(f"status: {status}")
    poll_history.append(
        {
            "status": status,
            "validation_count": len(experiment.get("validation_losses") or []),
            "final_validation_loss": experiment.get("final_validation_loss"),
        }
    )
    return experiment


experiment = poll_once()
if WAIT_FOR_TERMINAL:
    for _poll_number in range(1, MAX_POLLS):
        if str(experiment.get("status", "")) in TERMINAL_STATUSES:
            break
        time.sleep(POLL_INTERVAL_SECONDS)
        experiment = poll_once()

status = str(experiment.get("status", ""))

summary = {
    "api_base_url": BASE_URL,
    "budget_before": budget_before,
    "duplicate_config": duplicate,
    "experiment_id": experiment_id,
    "requested_runtime_seconds": requested_runtime_seconds,
    "config": config,
    "submit_result": submit_result,
    "latest_experiment": experiment,
    "poll_history": poll_history,
}

print(json.dumps(summary, indent=2, sort_keys=True))

if status in FAILURE_STATUSES:
    raise SystemExit(f"experiment reached failure status: {status}")

if WAIT_FOR_TERMINAL and status not in TERMINAL_STATUSES:
    raise SystemExit(
        f"experiment did not reach a terminal status after {MAX_POLLS} polls; "
        f"latest status={status!r}"
    )

if WAIT_FOR_TERMINAL and status != "completed":
    raise SystemExit(f"experiment terminal status was not completed: {status}")

log("control-node API smoke test passed")
PY
