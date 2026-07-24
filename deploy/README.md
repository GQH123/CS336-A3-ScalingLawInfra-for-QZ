# Manual Container Deployment

This project has two deployment roles:

- **Control node**: runs the student/staff API backend, quota accounting,
  manifest freezing, QZ/provider submission, callbacks, admin workflow, and
  exports.
- **Compute node**: runs the worker entrypoint, Stanford-style training backend,
  model/trainer code, tokenized data access, validation, and training logs.

The control node submits jobs to the cluster. It does not mount or read the
100B-scale training dataset. Dataset paths stored in API config are
worker-visible references that are copied into frozen manifests and interpreted
inside the compute-node worker image.

## Packages

Control node:

```bash
python -m pip install --upgrade pip
python -m pip install ./source/backend
```

Compute node image:

```bash
python -m pip install --upgrade pip
python -m pip install -c deploy/compute-node/constraints.txt './refs/cs336-assignment3-scaling[server]'
python -m pip install ./source/backend ./source/data ./source/training
```

The Stanford `[server]` extra installs `jax[cuda13]==0.9.1` on Linux. Use that
default on CUDA 13-capable nodes, for example H200 images with a sufficiently
new NVIDIA driver. On nodes whose `nvidia-smi` reports CUDA 12.x, such as many
RTX 4090 deployments, do not install the `[server]` extra because it pulls the
CUDA 13 JAX plugin and can fail with `cudaErrorInsufficientDriver`.
`flash-hog 0.5.0` also declares `jax[cuda13]` in its wheel metadata, so install
it after the CUDA 12 dependency pass with dependency resolution disabled. The
CUDA 12 requirements file intentionally keeps runtime transitive packages
such as `chex==0.1.91` explicit because those later `--no-deps` installs will
not resolve them.

```bash
python -m pip install --upgrade pip
python -m pip install -c deploy/compute-node/constraints.txt -r deploy/compute-node/requirements-cuda12.txt
python -m pip install --no-deps flash-hog==0.5.0
python -m pip install --no-deps ./refs/cs336-assignment3-scaling
python -m pip install ./source/backend ./source/data ./source/training
```

If a CUDA 12 node previously installed the CUDA 13 JAX plugin, or if a smoke
check prints `PJRT_Api already exists for device type cuda`, remove all JAX CUDA
plugin packages before reinstalling the CUDA 12 dependency set. Having both
`jax-cuda12-*` and `jax-cuda13-*` in the same environment can make JAX log a
plugin configuration error and silently fall back to CPU:

```bash
python -m pip uninstall -y \
  jax jaxlib \
  jax-cuda13-plugin jax-cuda13-pjrt \
  jax-cuda12-plugin jax-cuda12-pjrt
python -m pip install -c deploy/compute-node/constraints.txt -r deploy/compute-node/requirements-cuda12.txt
python -m pip install --no-deps flash-hog==0.5.0
python -m pip install --no-deps ./refs/cs336-assignment3-scaling
```

Before reinstalling, this command should print no `jax-cuda*` packages. After
reinstalling for a CUDA 12 node, it should print `jax-cuda12-*` packages and no
`jax-cuda13-*` packages:

```bash
python -m pip list | grep -E '^(jax|jaxlib|jax-cuda)'
```

Validate the compute image before submitting course traffic:

```bash
python - <<'PY'
import importlib.util

installed_plugins = [
    plugin
    for plugin in ("xla_cuda12", "xla_cuda13")
    if importlib.util.find_spec(f"jax_plugins.{plugin}") is not None
]
if len(installed_plugins) > 1:
    raise SystemExit(
        "Multiple JAX CUDA plugins are installed: "
        + ", ".join(installed_plugins)
        + ". Clean the environment and reinstall exactly one CUDA plugin."
    )

import jax
from furu import Furu

devices = jax.devices()
cuda_devices = [
    device for device in devices
    if getattr(device, "platform", "").lower() in {"cuda", "gpu"}
    or type(device).__name__.lower().startswith("cudadevice")
]

print("jax", jax.__version__)
print("devices", devices)
print("device_count", jax.device_count())
print("Furu", Furu)

if not cuda_devices:
    raise SystemExit(
        "JAX did not expose CUDA devices. Do not proceed: clean duplicate "
        "jax-cuda12/jax-cuda13 plugins or install the JAX CUDA runtime matching "
        "this node's NVIDIA driver."
    )
PY
```

The `source/training` package provides the configured callable
`course_trainer.worker:train`. It is a thin adapter around the Stanford CS336
training backend from `refs/cs336-assignment3-scaling`, plus a loader for the
mounted tokenized-index data contract in `deploy/compute-node/data-contract.md`.

For source-level debugging without rebuilding the compute-node image:

```bash
export PYTHONPATH=/mnt/course-dev/current/source/backend:/mnt/course-dev/current/source/data:/mnt/course-dev/current/source/training:/mnt/course-dev/current/refs/cs336-assignment3-scaling:${PYTHONPATH:-}
```

Use this overlay only for staff-controlled debug runs.

## Control Node

Copy and fill the API config:

```bash
cp deploy/control-node/api-runtime.sample.json /secure/course/api-runtime.json
export SCALING_API_CONFIG=/secure/course/api-runtime.json
```

Environment variables with the same runtime names override the config file. This
keeps secrets out of JSON when needed:

```bash
python -m scaling_backend.generate_api_tokens
```

This prints shell exports such as:

```bash
export SCALING_INTERNAL_CALLBACK_TOKEN="<random worker callback bearer token>"
export SCALING_ADMIN_API_TOKEN="<different random staff admin bearer token>"
```

For a temporary shell session, evaluate the generated exports directly:

```bash
eval "$(python -m scaling_backend.generate_api_tokens)"
```

For a persistent service, write those generated values into the private
systemd environment file or secret store. The QZ browser session cookie is not a
random API token; set it separately after logging in to QZ manually and solving
any CAPTCHA challenge:

```bash
export QZ_COOKIE="<document.cookie from a logged-in qz.sii.edu.cn session>"
# Optional fallback only when QZ_COOKIE is absent.
export QZ_USERNAME="$QZ_USERNAME"
export QZ_PASSWORD_ENCRYPTED="$QZ_PASSWORD_ENCRYPTED"
```

Run preflight on the control node:

```bash
python -m scaling_backend.preflight \
  --config /secure/course/api-runtime.json \
  --create-dirs \
  --output-json /secure/course/preflight.json
```

Set `api.provider_poll_interval_seconds` in
`deploy/control-node/api-runtime.sample.json`, or export
`SCALING_PROVIDER_POLL_INTERVAL_SECONDS`, to enable the built-in control-node
poll loop. A value such as `30` makes the API poll active QZ jobs, refresh
state snapshots, and close jobs that fail after the worker process dies before
it can call back. Use `0` only when an external scheduler calls
`admin_cli poll-active`.

For QZ cookie or credential connectivity without submitting a training task:

```bash
python -m scaling_backend.preflight \
  --config /secure/course/api-runtime.json \
  --create-dirs \
  --probe-provider \
  --output-json /secure/course/preflight-provider.json
```

Start the API:

```bash
cp deploy/control-node/launch.sh /secure/course/launch.sh
cd /secure/course
SCALING_CALLBACK_URL="http://<websocket-forward-host>:8000/internal/provider-events" \
./launch.sh
```

The launcher is runtime-directory relative: by default it reads
`api-runtime.json`, `student_api_keys.csv`, and `frozen-manifests/` next to
itself. Set `SCALING_CALLBACK_URL` to the control-node URL reachable from QZ
worker containers, such as the WebSocket port-forward URL. Shared-storage JSONL
worker events are a fallback: enable them only by explicitly setting both
`SCALING_WORKER_EVENT_IMPORT_DIR` and `QZ_WORKER_EVENT_LOG_DIR`.

## Compute Node

Copy `deploy/compute-node/worker-runtime.sample.json` into the worker image or a
mounted staff config path if the trainer reads local worker settings:

```bash
cp deploy/compute-node/worker-runtime.sample.json /mnt/course-config/worker-runtime.json
export SCALING_WORKER_CONFIG=/mnt/course-config/worker-runtime.json
```

The bundled worker entrypoint receives the frozen manifest from the control node:

```bash
bash -lc 'export SCALING_CALLBACK_TOKEN=<worker-callback-token> && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate fnlps_a3_compute && exec python -m scaling_backend.worker.run --manifest-uri <frozen-manifest-uri> --callback-url <websocket-forward-callback-url> --callback-token-env SCALING_CALLBACK_TOKEN --trainer course_trainer.worker:train --heartbeat-interval-seconds 60 --tokenized-index-validation-mode metadata'
```

The control-node QZ adapter generates this wrapper from runtime settings. It
keeps HTTP callback mode by default and exports the callback token before
starting the worker with `exec`. If `QZ_WORKER_EVENT_LOG_DIR` is explicitly
configured, the adapter instead adds `--event-sink jsonl --event-log-path ...`;
in that fallback mode the callback token is not exported into the QZ command
because the worker does not call the callback endpoint.

The heartbeat flag emits periodic worker callbacks only while the Python worker
process is alive. Native crashes, platform kills, and segfaults cannot be
reported by that thread after the process exits, so QZ status/log polling on the
control node is the terminal-state backstop.

`metadata` mode validates `index.json`, safe shard paths, token counts, local
file sizes, and SHA-256 metadata format without reading full shards. Use `full`
only for offline audits or small rehearsals where reading every shard is
intentional.

### Interactive Compute-Node GPU Debug

When the interactive shell already runs on a GPU compute node, use a local debug
API with the `fake` provider to create a real backend experiment and frozen
manifest without submitting a QZ job. Then execute that manifest on the same
node with `python -m scaling_backend.worker.run`. This gives a full
submit-manifest-callback loop while avoiding image rebuilds during debugging.

Use a separate runtime directory and do not point this debug API at the
production state snapshot:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fnlps_a3_compute

export REPO=/path/to/assignment-3
export DEBUG_ROOT=/tmp/a3-local-gpu-debug
export API_PORT=18000
mkdir -p "$DEBUG_ROOT/frozen-manifests"
printf 'student_id,api_key\nlocal-debug,local-debug-key\n' > "$DEBUG_ROOT/student_api_keys.csv"

export PYTHONPATH="$REPO/source/backend:$REPO/source/data:$REPO/source/training:$REPO/refs/cs336-assignment3-scaling:${PYTHONPATH:-}"
export SCALING_PROVIDER=fake
export SCALING_STUDENT_KEYS_CSV="$DEBUG_ROOT/student_api_keys.csv"
export SCALING_INTERNAL_CALLBACK_TOKEN="local-debug-callback-token"
export SCALING_ADMIN_API_TOKEN="local-debug-admin-token"
export SCALING_CALLBACK_URL="http://127.0.0.1:${API_PORT}/internal/provider-events"
export SCALING_MANIFEST_BASE_URI="file://$DEBUG_ROOT/frozen-manifests"
export SCALING_LOCAL_MANIFEST_DIR="$DEBUG_ROOT/frozen-manifests"
export SCALING_STATE_SNAPSHOT_PATH="$DEBUG_ROOT/state-snapshot.json"
export SCALING_TOTAL_BUDGET_SECONDS=43200
export SCALING_VALIDATION_TOKENS_PER_EVAL=4096
export SCALING_WORKER_GPU_COUNT=2
export SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE=metadata

export SCALING_TOKENIZED_TRAIN_INDEX_URI="file:///mnt/course-data/tokenized/train/index.json"
export SCALING_TOKENIZED_VALIDATION_INDEX_URI="file:///mnt/course-data/tokenized/validation/index.json"

uvicorn scaling_backend.asgi:app --host 127.0.0.1 --port "$API_PORT"
```

In another shell on the same compute node, submit a small experiment to the
debug API. Varying `model_seed` avoids duplicate-config rejection:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fnlps_a3_compute

export REPO=/path/to/assignment-3
export DEBUG_ROOT=/tmp/a3-local-gpu-debug
export API_PORT=18000
export PYTHONPATH="$REPO/source/backend:$REPO/source/data:$REPO/source/training:$REPO/refs/cs336-assignment3-scaling:${PYTHONPATH:-}"

EXPERIMENT_ID="$(
python - <<'PY'
import os
import json
import sys
import time
import requests

api = f"http://127.0.0.1:{os.environ['API_PORT']}"
config = {
    "model": {
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 1,
        "dtype": "bfloat16",
    },
    "training": {
        "train_tokens": 4096,
        "sequence_length": 1024,
        "train_batch_size": 2,
        "validation_batch_size": 2,
        "num_evals": 1,
        "learning_rate": 3e-4,
        "model_seed": int(time.time()) % 2_000_000_000,
    },
}
response = requests.post(
    f"{api}/submit",
    headers={"Authorization": "Bearer local-debug-key"},
    json={"config": config, "requested_runtime_seconds": 300},
    timeout=30,
)
try:
    body = response.json()
except ValueError:
    body = {"raw_response": response.text}
if response.status_code < 200 or response.status_code >= 300:
    print(json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
    raise SystemExit(1)
print(body["experiment_id"])
PY
)"
export MANIFEST_PATH="$DEBUG_ROOT/frozen-manifests/${EXPERIMENT_ID}.json"
python -m json.tool "$MANIFEST_PATH" | sed -n '1,120p'
```

Run the submitted manifest on the local GPU:

```bash
export RUN_DIR="$DEBUG_ROOT/runs/$EXPERIMENT_ID"
mkdir -p "$RUN_DIR"
export SCALING_CALLBACK_TOKEN="local-debug-callback-token"

set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -m scaling_backend.worker.run \
  --manifest-uri "file://$MANIFEST_PATH" \
  --callback-url "http://127.0.0.1:${API_PORT}/internal/provider-events" \
  --callback-token-env SCALING_CALLBACK_TOKEN \
  --trainer course_trainer.worker:train \
  --heartbeat-interval-seconds 60 \
  --tokenized-index-validation-mode metadata \
  2>&1 | tee "$RUN_DIR/worker.log"
echo "worker exit: ${PIPESTATUS[0]}"
```

Inspect the backend result after the worker exits:

```bash
curl -sS \
  -H "Authorization: Bearer local-debug-key" \
  "http://127.0.0.1:${API_PORT}/experiment/${EXPERIMENT_ID}" \
  | python -m json.tool
```

Do not run this loop against the production API while the production provider is
`qz_distributed`; otherwise `/submit` will also create a QZ job, and the manual
worker and QZ worker can race on the same `experiment_id`.

## Lifecycle Dataset Rehearsal

After packaging a small lifecycle dataset with
`python -m scaling_data.lifecycle_dataset`, use the generated package to rehearse
both deployment roles before sending normal student traffic.

On the control node, load the lifecycle runtime overrides before launching or
restarting the API:

```bash
export LIFECYCLE_DATASET_DIR=/path/to/lifecycle_test_dataset
source "$LIFECYCLE_DATASET_DIR/api-runtime-overrides.env"
./launch.sh
```

Run an offline control-node check without contacting QZ or a running API:

```bash
export LIFECYCLE_DATASET_DIR=/path/to/lifecycle_test_dataset
export LIFECYCLE_API_MODE=local-service
export SCALING_CALLBACK_URL=https://backend/internal/provider-events
export SCALING_MANIFEST_BASE_URI=memory://manifests
deploy/control-node/lifecycle-test-api.sh
```

Run the remote API check against a live control-node API. This submits the
package's recommended tiny exploratory config, then verifies that the frozen
manifest contains the lifecycle train and validation index URIs. The script
defaults `LIFECYCLE_API_UNIQUE_MODEL_SEED=1` so it checks the current API
runtime instead of reusing an older duplicate submission:

```bash
export LIFECYCLE_DATASET_DIR=/path/to/lifecycle_test_dataset
export LIFECYCLE_CONTROL_RUNTIME_DIR=/path/to/assignment-3-control-runtime
export SCALING_API_BASE_URL=http://127.0.0.1:8000
export SCALING_API_KEY=<one-student-api-key>
deploy/control-node/lifecycle-test-api.sh
```

On a compute node or inside the worker image, run the lifecycle worker check.
The default runs the real `course_trainer.worker:train` path through
`python -m scaling_backend.worker.run` with JSONL events saved under the
lifecycle package:

```bash
export LIFECYCLE_DATASET_DIR=/path/to/lifecycle_test_dataset
deploy/compute-node/lifecycle-test-worker-run.sh
```

For an offline artifact audit that avoids launching the trainer, set:

```bash
export LIFECYCLE_SKIP_WORKER_RUN=1
export LIFECYCLE_TOKENIZED_INDEX_VALIDATION_MODE=full
deploy/compute-node/lifecycle-test-worker-run.sh
```

The lifecycle dataset uses the same validation shards for exploratory and final
paths. Use it only to prove API/provider/worker lifecycle behavior; it is not a
grading dataset.

## Data Contract

See `deploy/compute-node/data-contract.md` for the tokenized index schema and
standard compute-node-visible paths.
