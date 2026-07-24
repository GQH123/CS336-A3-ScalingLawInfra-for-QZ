#!/usr/bin/env bash
set -Eeuo pipefail

# Smoke-test scaling_backend.worker.run inside a compute-node worker environment.
#
# This script is intentionally self-contained:
# - starts a local callback receiver
# - creates tiny tokenized train/validation indexes
# - creates a frozen worker manifest
# - creates a tiny smoke trainer
# - runs python -m scaling_backend.worker.run
# - verifies callback events and terminal completion
#
# It does not run the real CS336/JAX trainer. Its purpose is to validate the
# worker entrypoint, manifest loading, tokenized-index preflight, trainer loading,
# callback posting, and worker result normalization.

log() {
  printf '[worker-smoke] %s\n' "$*"
}

fail() {
  printf '[worker-smoke] ERROR: %s\n' "$*" >&2
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

# If this script is run from the source checkout, make local packages importable.
# If it is copied into a built image, installed packages should already work and
# these paths will be harmless if absent.
for package_dir in \
  "$REPO_ROOT/source/backend" \
  "$REPO_ROOT/source/data" \
  "$REPO_ROOT/source/training" \
  "$REPO_ROOT/refs/cs336-assignment3-scaling"
do
  if [[ -d "$package_dir" ]]; then
    export PYTHONPATH="$package_dir:${PYTHONPATH:-}"
  fi
done

WORK_DIR="${SMOKE_WORK_DIR:-}"
if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$(mktemp -d /tmp/scaling-worker-smoke.XXXXXX)"
else
  mkdir -p "$WORK_DIR"
fi
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

CALLBACK_HOST="${SMOKE_CALLBACK_HOST:-127.0.0.1}"
CALLBACK_PORT="${SMOKE_CALLBACK_PORT:-0}"
CALLBACK_TOKEN_ENV="${SMOKE_CALLBACK_TOKEN_ENV:-SCALING_CALLBACK_TOKEN}"
TOKENIZED_INDEX_VALIDATION_MODE="${SMOKE_TOKENIZED_INDEX_VALIDATION_MODE:-metadata}"
HEARTBEAT_INTERVAL_SECONDS="${SMOKE_HEARTBEAT_INTERVAL_SECONDS:-0}"
RUN_ID="${SMOKE_EXPERIMENT_ID:-exp-smoke-000001}"

export "$CALLBACK_TOKEN_ENV"="${!CALLBACK_TOKEN_ENV:-dev-smoke-callback-token}"

CALLBACK_PID=""
cleanup() {
  local status=$?
  if [[ -n "$CALLBACK_PID" ]] && kill -0 "$CALLBACK_PID" >/dev/null 2>&1; then
    kill "$CALLBACK_PID" >/dev/null 2>&1 || true
    wait "$CALLBACK_PID" >/dev/null 2>&1 || true
  fi
  if [[ "${SMOKE_KEEP_WORK_DIR:-0}" != "1" ]]; then
    rm -rf "$WORK_DIR"
  else
    log "kept work directory: $WORK_DIR"
  fi
  exit "$status"
}
trap cleanup EXIT

log "work directory: $WORK_DIR"
log "python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
log "python version: $($PYTHON_BIN -c 'import sys; print(sys.version.split()[0])')"

"$PYTHON_BIN" - "${SMOKE_REQUIRE_REAL_TRAINER_IMPORTS:-0}" <<'PY'
import importlib
import sys

required = [
    "scaling_backend.worker.run",
    "course_trainer.worker",
]
for module_name in required:
    importlib.import_module(module_name)
    print(f"[worker-smoke] import ok: {module_name}")

optional = [
    "cs336_scaling",
    "jax",
    "equinox",
]
missing = []
for module_name in optional:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append((module_name, type(exc).__name__, str(exc)))
    else:
        print(f"[worker-smoke] optional import ok: {module_name}")

if missing:
    for module_name, exc_type, message in missing:
        print(
            f"[worker-smoke] warning: optional import failed: "
            f"{module_name}: {exc_type}: {message}",
            file=sys.stderr,
        )
    if sys.argv[1] == "1":
        raise SystemExit(1)
PY

cat > "$WORK_DIR/callback_receiver.py" <<'PY'
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import sys


event_log = Path(sys.argv[1])
ready_file = Path(sys.argv[2])
host = sys.argv[3]
port = int(sys.argv[4])


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = raw

        event = {
            "path": self.path,
            "authorization": self.headers.get("authorization", ""),
            "body": body,
        }
        with event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()

        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}\n')

    def log_message(self, *_args):
        return


server = HTTPServer((host, port), Handler)
ready_file.write_text(
    json.dumps({"host": host, "port": server.server_address[1]}, sort_keys=True)
    + "\n",
    encoding="utf-8",
)
server.serve_forever()
PY

EVENT_LOG="$WORK_DIR/callback-events.jsonl"
READY_FILE="$WORK_DIR/callback-ready.json"
CALLBACK_STDOUT="$WORK_DIR/callback-stdout.log"
CALLBACK_STDERR="$WORK_DIR/callback-stderr.log"

"$PYTHON_BIN" "$WORK_DIR/callback_receiver.py" \
  "$EVENT_LOG" \
  "$READY_FILE" \
  "$CALLBACK_HOST" \
  "$CALLBACK_PORT" \
  >"$CALLBACK_STDOUT" 2>"$CALLBACK_STDERR" &
CALLBACK_PID=$!

for _ in $(seq 1 100); do
  if [[ -s "$READY_FILE" ]]; then
    break
  fi
  if ! kill -0 "$CALLBACK_PID" >/dev/null 2>&1; then
    cat "$CALLBACK_STDERR" >&2 || true
    fail "callback receiver exited before becoming ready"
  fi
  sleep 0.05
done

[[ -s "$READY_FILE" ]] || fail "callback receiver did not become ready"

BOUND_PORT="$("$PYTHON_BIN" - "$READY_FILE" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["port"])
PY
)"
CALLBACK_URL="http://$CALLBACK_HOST:$BOUND_PORT/internal/provider-events"
log "callback receiver: $CALLBACK_URL"

MANIFEST_URI="$("$PYTHON_BIN" - "$WORK_DIR" "$RUN_ID" "$CALLBACK_URL" <<'PY'
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys


root = Path(sys.argv[1])
run_id = sys.argv[2]
callback_url = sys.argv[3]


def write_index(directory: Path, tokens: list[int]) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    payload = b"".join(int(token).to_bytes(4, "little") for token in tokens)
    shard = directory / "tokens-000000.bin"
    shard.write_bytes(payload)

    index = {
        "schema_version": 1,
        "token_dtype": "uint32",
        "byte_order": "little",
        "shard_format": "flat_binary_uint32_le",
        "total_tokens": len(tokens),
        "source_token_counts": {"smoke": len(tokens)},
        "shards": [
            {
                "path": shard.name,
                "tokens": len(tokens),
                "dtype": "uint32",
                "byte_order": "little",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_token_counts": {"smoke": len(tokens)},
            }
        ],
    }
    index_path = directory / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index_path.resolve().as_uri()


train_index_uri = write_index(root / "tokenized" / "train", list(range(1, 65)))
validation_index_uri = write_index(
    root / "tokenized" / "validation",
    list(range(101, 133)),
)

manifest_path = root / "manifest.json"
manifest = {
    "manifest_version": 1,
    "run_kind": "exploratory",
    "experiment_id": run_id,
    "student_id": "student-smoke",
    "manifest_uri": manifest_path.resolve().as_uri(),
    "callback_url": callback_url,
    "callback_base_url": callback_url,
    "model_config": {
        "attention_bias": False,
        "head_dim": 128,
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_attention_heads": 1,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "vocab_size": 32000,
    },
    "training_config": {
        "train_tokens": 1024,
        "sequence_length": 128,
        "train_batch_size": 1,
        "validation_batch_size": 1,
        "num_evals": 2,
        "model_seed": 0,
        "learning_rate": 3e-4,
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "warmup_fraction": 0.0,
        "final_lr_fraction": 0.0,
        "gradient_clip_norm": 1.0,
    },
    "data_config": {
        "train_tokens": 1024,
        "tokenized_index_uri": train_index_uri,
    },
    "validation_config": {
        "eval_manifest_id": "exploratory-eval-v0",
        "validation_tokens_per_eval": 128,
        "validation_batches_per_eval": 1,
        "tokenized_index_uri": validation_index_uri,
    },
    "runtime_config": {
        "reserved_runtime_seconds": 60,
        "max_runtime_seconds": 60,
    },
    "max_runtime_seconds": 60,
    "code_version": "smoke-test",
}
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

(root / "smoke_trainer.py").write_text(
    '''\
def train(manifest, emit_event):
    losses = [4.2, 3.9]
    for step, loss in enumerate(losses, start=1):
        emit_event({
            "event_type": "validation",
            "step": step,
            "loss": loss,
        })

    return {
        "validation_losses": losses,
        "final_validation_loss": losses[-1],
        "actual_runtime_seconds": 1,
    }
''',
    encoding="utf-8",
)

print(manifest_path.resolve().as_uri())
PY
)"
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"

log "manifest: $MANIFEST_URI"
log "running scaling_backend.worker.run"

set +e
"$PYTHON_BIN" -m scaling_backend.worker.run \
  --manifest-uri "$MANIFEST_URI" \
  --callback-url "$CALLBACK_URL" \
  --callback-token-env "$CALLBACK_TOKEN_ENV" \
  --trainer smoke_trainer:train \
  --heartbeat-interval-seconds "$HEARTBEAT_INTERVAL_SECONDS" \
  --tokenized-index-validation-mode "$TOKENIZED_INDEX_VALIDATION_MODE"
WORKER_EXIT=$?
set -e

log "worker exit code: $WORKER_EXIT"
if [[ "$WORKER_EXIT" -ne 0 ]]; then
  log "callback events so far:"
  cat "$EVENT_LOG" >&2 || true
  fail "worker smoke test failed"
fi

"$PYTHON_BIN" - "$EVENT_LOG" "$RUN_ID" "${!CALLBACK_TOKEN_ENV}" <<'PY'
from __future__ import annotations

from pathlib import Path
import json
import sys


event_log = Path(sys.argv[1])
run_id = sys.argv[2]
callback_token = sys.argv[3]

events = [
    json.loads(line)
    for line in event_log.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
bodies = [event["body"] for event in events]
event_types = [body.get("event_type") for body in bodies]

required = ["worker_started", "validation", "worker_completed"]
missing = [event_type for event_type in required if event_type not in event_types]
if missing:
    raise SystemExit(f"missing callback event(s): {missing}; got {event_types}")

for event in events:
    expected_auth = f"Bearer {callback_token}"
    if event["authorization"] != expected_auth:
        raise SystemExit(
            "unexpected callback Authorization header: "
            f"{event['authorization']!r}; expected {expected_auth!r}"
        )
    body = event["body"]
    if body.get("experiment_id") != run_id:
        raise SystemExit(f"event has wrong experiment_id: {body}")

validation_losses = [
    body["loss"]
    for body in bodies
    if body.get("event_type") == "validation"
]
if validation_losses != [4.2, 3.9]:
    raise SystemExit(f"unexpected validation event losses: {validation_losses}")

completed = [
    body for body in bodies if body.get("event_type") == "worker_completed"
]
if len(completed) != 1:
    raise SystemExit(f"expected exactly one worker_completed event, got {len(completed)}")

terminal = completed[0]
if terminal.get("validation_losses") != [4.2, 3.9]:
    raise SystemExit(f"bad terminal validation_losses: {terminal}")
if terminal.get("final_validation_loss") != 3.9:
    raise SystemExit(f"bad terminal final_validation_loss: {terminal}")
if terminal.get("actual_runtime_seconds") != 1:
    raise SystemExit(f"bad terminal actual_runtime_seconds: {terminal}")

print(
    json.dumps(
        {
            "events": event_types,
            "terminal": terminal,
        },
        indent=2,
        sort_keys=True,
    )
)
PY

log "smoke test passed"
