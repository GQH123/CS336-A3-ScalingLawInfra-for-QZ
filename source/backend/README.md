# Scaling Backend

This directory contains the first native backend modules for the summer scaling
law assignment. The current implementation focuses on the provider boundary and
the QZ distributed-training adapter.

The adapter does not import or execute `refs/qzcli_tool`. That repository remains
reference material for the platform's cookie-authenticated request headers,
optional CAS login fallback, train-job payload shape, and operational behavior.

## Fake Provider

`scaling_backend.providers.fake.FakeProviderAdapter` is the deterministic local
provider for unit tests and submit-to-result smoke runs. It records submitted
manifests, returns stable `fake-job-000001`-style job IDs, can simulate a fixed
provider-status sequence, supports cancellation, and returns local artifact
pointers. It performs no network calls and starts no external processes.

## QZ Distributed Provider

`scaling_backend.providers.qz_distributed` submits one QZ distributed-training
job per frozen backend manifest. The QZ job runs a staff-built Docker image that
already contains the training worker and all runtime dependencies.

The compute image should install the API worker package, data utilities, the
course trainer adapter, and the Stanford reference backend:

```bash
python -m pip install -c deploy/compute-node/constraints.txt './refs/cs336-assignment3-scaling[server]'
python -m pip install ./source/backend ./source/data ./source/training
```

The Stanford `[server]` extra installs `jax[cuda13]==0.9.1` on Linux. For CUDA
12 driver environments, install the CUDA 12 dependency file first, then install
`flash-hog` and the Stanford package without dependencies so their CUDA 13
metadata is not resolved. The CUDA 12 requirements file explicitly includes
runtime transitive packages such as `chex==0.1.91` because the later
`--no-deps` installs will not resolve them:

```bash
python -m pip install -c deploy/compute-node/constraints.txt -r deploy/compute-node/requirements-cuda12.txt
python -m pip install --no-deps flash-hog==0.5.0
python -m pip install --no-deps ./refs/cs336-assignment3-scaling
python -m pip install ./source/backend ./source/data ./source/training
```

`source/training` provides `course_trainer.worker:train`, which adapts frozen
backend manifests and the mounted flat-binary tokenized data contract to the
local Stanford CS336 model, optimizer, and training loop modules.

Required staff configuration:

```bash
export QZ_API_BASE_URL="https://qz.sii.edu.cn"
export QZ_COOKIE_FILE="/secure/course/qz.cookie"
# Optional startup override or initial seed. When set, QZ_COOKIE wins over QZ_COOKIE_FILE.
export QZ_COOKIE="<document.cookie from a manually logged-in QZ browser session>"
export QZ_SESSION_HEARTBEAT_INTERVAL_SECONDS="900"
# Optional fallback only. CAPTCHA-protected deployments should rely on QZ_COOKIE_FILE.
export QZ_USERNAME="253108120093"
export QZ_PASSWORD_ENCRYPTED="<encrypted CAS payload>"
export QZ_WORKSPACE_ID="ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
export QZ_PROJECT_ID="project-c67c548f-f02c-453b-ba5b-8745db6886e7"
export QZ_COMPUTE_GROUP_ID="lcg-a91ad10b-415d-4abd-8170-828a2feae5d2"
export QZ_SPEC_ID="4dd0e854-e2a4-4253-95e6-64c13f0b5117"
export QZ_SPEC_GPU_TYPE="H200"
export QZ_SPEC_GPU_COUNT="1"
export QZ_SPEC_CPU_COUNT="15"
export QZ_SPEC_MEMORY_GB="200"
export QZ_IMAGE="registry.example.com/course/scaling-worker:latest"
export QZ_IMAGE_TYPE="SOURCE_PRIVATE"
export QZ_PRIORITY="10"
export QZ_SHM_GI="1200"
export QZ_CALLBACK_TOKEN_ENV="SCALING_CALLBACK_TOKEN"
export QZ_WORKER_TRAINER="course_trainer.worker:train"
export QZ_WORKER_HEARTBEAT_INTERVAL_SECONDS="60"
export QZ_WORKER_CONDA_ENV="fnlps_a3_compute"
export QZ_WORKER_CONDA_INIT="/opt/anaconda3/etc/profile.d/conda.sh"
```

These values are staff-only deployment settings. Student-facing APIs should never
expose QZ workspace IDs, compute group IDs, spec IDs, cookies, or raw provider
errors. QZ numeric settings for GPU count, CPU count, memory, priority,
shared-memory size, request timeout, login timeout, and login retry count must
be positive integers; malformed values are rejected during backend startup.
At startup the QZ adapter uses `QZ_COOKIE` first. If that variable is absent, it
loads the stripped cookie string from `QZ_COOKIE_FILE`; if neither source has a
cookie, it falls back to the legacy CAS credential flow. Every QZ response
`Set-Cookie` header and every successful CAS login refreshes the in-memory
cookie and writes it back to `QZ_COOKIE_FILE` with `0600` permissions when the
file path is configured. For persistent services, prefer leaving `QZ_COOKIE`
unset after the file is seeded so a stale environment value cannot mask the
refreshed file on restart.
When `QZ_SESSION_HEARTBEAT_INTERVAL_SECONDS` is greater than `0`, the API starts
a QZ-only background heartbeat that periodically calls the same read-only
distributed-training job-list endpoint used by preflight with `page_size=1`.
The heartbeat runs once when the API starts, then once per configured interval.
It does not submit, cancel, or mutate training jobs. Successful and failed
heartbeat attempts are logged through the uvicorn console logger, so a normal
`./launch.sh` session should show heartbeat activity. Any `Set-Cookie` header on
the heartbeat response follows the same `QZ_COOKIE_FILE` persistence path. Use
`0` to disable this API-side session heartbeat.
The QZ adapter keeps HTTP callback mode by default. It exports
`SCALING_INTERNAL_CALLBACK_TOKEN` into the worker command under
`QZ_CALLBACK_TOKEN_ENV` because the QZ create payload has no separate
environment-variable field. If staff explicitly configure
`QZ_WORKER_EVENT_LOG_DIR`, the adapter starts workers with `--event-sink jsonl`
and writes one shared-storage JSONL event file per run; in that fallback mode it
does not export the callback token into the QZ command.

## Slurm Provider

`scaling_backend.providers.slurm.SlurmProviderAdapter` is the alternate cluster
provider for staff testing or deployments where Slurm is the execution system.
It writes one `sbatch` script per frozen manifest, launches it with `sbatch
--parsable`, polls with `squeue` and then `sacct`, cancels with `scancel`, and
returns local `file://` pointers for stdout, stderr, and the generated script.

Example staff configuration:

```bash
export SCALING_PROVIDER="slurm"
export SLURM_WORK_DIR="/secure/course/slurm-runs"
export SLURM_PARTITION="gpu"
export SLURM_ACCOUNT="course"
export SLURM_GPUS_PER_TASK="1"
export SLURM_CPUS_PER_TASK="16"
export SLURM_MEMORY_GB="128"
export SLURM_TIME_LIMIT_MINUTES="90"
export SLURM_CALLBACK_TOKEN_ENV="SCALING_CALLBACK_TOKEN"
export SLURM_WORKER_TRAINER="course_trainer.worker:train"
export SLURM_WORKER_HEARTBEAT_INTERVAL_SECONDS="60"
export SLURM_PYTHON="python"
```

Slurm numeric resource settings must be positive integers and are validated at
backend startup. `SLURM_WORKER_HEARTBEAT_INTERVAL_SECONDS` is optional and uses
`0` to disable automatic worker heartbeats. Unit tests use an injected fake
command runner and do not require a live Slurm installation.

## Runtime App Factory

`scaling_backend.runtime.build_app_from_env(...)` builds the version-one FastAPI
app from staff-controlled environment variables. `scaling_backend.asgi:app` is
the import target for ASGI servers.

Required runtime settings:

```bash
export SCALING_STUDENT_KEYS_CSV="/secure/course/student_api_keys.csv"
eval "$(python -m scaling_backend.generate_api_tokens)"
export SCALING_CALLBACK_URL="http://<websocket-forward-host>:8000/internal/provider-events"
# Shared-filesystem manifest path:
export SCALING_MANIFEST_BASE_URI="file:///secure/course/frozen-manifests"
export SCALING_LOCAL_MANIFEST_DIR="/secure/course/frozen-manifests"
```

For HTTP-published manifests, use `SCALING_PUBLISHED_MANIFEST_DIR` plus
`SCALING_PUBLISHED_MANIFEST_BASE_URI` instead of `SCALING_LOCAL_MANIFEST_DIR`.
For shared-storage worker-event fallback, set both
`SCALING_WORKER_EVENT_IMPORT_DIR` and `QZ_WORKER_EVENT_LOG_DIR`.

`python -m scaling_backend.generate_api_tokens` prints random
`SCALING_INTERNAL_CALLBACK_TOKEN` and `SCALING_ADMIN_API_TOKEN` shell exports.
Use `--format json` when feeding a secret-management step instead of a shell.

The student key CSV is staff-private and must have this header:

```csv
student_id,api_key
student-1,student-1-secret-api-key
student-2,student-2-secret-api-key
```

Staff can generate this file from a roster CSV with a `student_id` column:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli \
  seed-student-keys \
  --roster-csv /secure/course/roster.csv \
  --output-csv /secure/course/student_api_keys.csv
```

`seed-student-keys` is a local operation and does not need
`SCALING_ADMIN_BASE_URL` or `SCALING_ADMIN_API_TOKEN`. It refuses to overwrite an
existing key CSV unless `--overwrite` is supplied and prints only the output path
and student count, not the generated secrets.

Optional runtime settings:

```bash
export SCALING_API_CONFIG="/secure/course/api-runtime.json"
export SCALING_PROVIDER="qz_distributed"       # qz_distributed, slurm, or fake
export SCALING_TOTAL_BUDGET_SECONDS="43200"   # 12 hours by default
export SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT="0"  # 0 means unlimited
export SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL="0"       # 0 means unlimited
export SCALING_CODE_VERSION="course-backend-v0"
export SCALING_DATA_MANIFEST_ID="exploratory-train-v0"
export SCALING_EVAL_MANIFEST_ID="exploratory-eval-v0"
export SCALING_FINAL_DATA_MANIFEST_ID="final-train-v0"
export SCALING_FINAL_EVAL_MANIFEST_ID="final-eval-v0"
export SCALING_VALIDATION_TOKENS_PER_EVAL="262144"
export SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL="524288"
export SCALING_TOKENIZED_TRAIN_INDEX_URI="file:///mnt/course-data/tokenized/train/index.json"
export SCALING_TOKENIZED_VALIDATION_INDEX_URI="file:///mnt/course-data/tokenized/validation/index.json"
export SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI="file:///mnt/course-data/tokenized/train/index.json"
export SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI="file:///mnt/course-data/tokenized/final-validation/index.json"
export SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE="metadata"
export SCALING_STATE_SNAPSHOT_PATH="/secure/course/state-snapshot.json"
```

`SCALING_API_CONFIG` points to the control-node API config. The API/control node
owns student auth, quotas, manifest freezing, provider submission, callbacks,
admin workflow, and exports. Values from explicit environment variables override
the config file, which lets staff keep secrets outside JSON. `SCALING_COURSE_CONFIG`
is accepted only as a compatibility alias for older local scripts.

`SCALING_DATA_MANIFEST_ID` and `SCALING_EVAL_MANIFEST_ID` are used for
exploratory submissions. `SCALING_FINAL_DATA_MANIFEST_ID` and
`SCALING_FINAL_EVAL_MANIFEST_ID` are used only when staff launch frozen final
runs. The final evaluation manifest should identify the larger held-out
leaderboard split and remain staff-private.
`SCALING_VALIDATION_TOKENS_PER_EVAL` and
`SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL` are staff-controlled evaluation
sample sizes, not student parameters. Accepted configs must make each value
divisible by `training.sequence_length * training.validation_batch_size`; this
prevents the worker from silently rounding validation batches.
The `SCALING_TOKENIZED_*_INDEX_URI` settings are optional private worker inputs.
When set, they are copied only into frozen provider manifests as
`data_config.tokenized_index_uri` or `validation_config.tokenized_index_uri`;
they are not returned in student-facing resolved configs. These are
compute-node-visible references; the API preflight validates their URI syntax
but does not require the tokenized data to exist on the control node.
`SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE` controls worker startup checks.
Use `metadata` for production mounted datasets; it validates `index.json`,
relative shard paths, token counts, local file sizes, and SHA-256 metadata
format without reading full shards. Use `full` for offline artifact audits where
reading and hashing every shard is intentional. The compute-node worker image
owns those data checks.

`SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT` and
`SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL` are optional fairness and
cluster-pressure guards for exploratory runs. The default `0` preserves
unlimited immediate exploratory launches. A positive per-student value limits
how many provider-dispatched exploratory jobs a single student may have active
at once; a positive global value limits provider-dispatched exploratory jobs
across all students. Excess accepted submissions are kept in the local
exploratory queue, reserve budget, and are launched by `poll-active` or terminal
worker/provider events once capacity frees. Completed, failed, cancelled, and
lost runs do not count against launch capacity.

`SCALING_PROVIDER_POLL_INTERVAL_SECONDS` enables the built-in control-node
background poller when set to a positive integer. The poller calls the same
active-run provider status path as `POST /admin/poll-active`, saves
`SCALING_STATE_SNAPSHOT_PATH` after each poll when snapshots are configured, and
lets the backend close provider-terminal jobs even when the worker process died
before it could send a callback. The default `0` disables the background loop so
staff can use an external scheduler or manual `poll-active` during local tests.

`SCALING_STATE_SNAPSHOT_PATH` is an optional lightweight local durability bridge.
When set and the file exists, the runtime factory loads the saved service state
at startup. When the FastAPI app is built through
`scaling_backend.runtime.build_app_from_env(...)`, successful mutating requests
also write a fresh snapshot after the request completes. This preserves
experiments, budgets, final submissions, final runs, worker events, and admin
adjustments across process restarts. Snapshot writes use a same-directory
temporary file and atomic replacement so an interrupted write does not clobber
the previous valid snapshot. When the provider supports `restore_job`, active
exploratory and final-run provider jobs are restored from the frozen manifests
saved in the snapshot so staff poll/cancel operations keep working after a
restart. This is not a replacement for the planned
database-backed production store.

Before starting a staff deployment, run the local preflight checker:

```bash
PYTHONPATH=source/backend python -m scaling_backend.preflight \
  --config /secure/course/api-runtime.json \
  --create-dirs \
  --output-json /secure/course/preflight.json
```

The preflight validates the student-key CSV, required runtime settings, callback
and manifest URI schemes, distinct staff secrets, manifest storage, snapshot
path, budget settings, manifest ID separation, tokenized-index URI schemes, and
provider-specific environment variables. It does not start FastAPI, submit
provider jobs, fetch manifests, or read large tokenized shards. Worker startup
uses metadata validation by default; run full shard hashing only as an explicit
offline audit.

To verify the configured provider without launching training, add
`--probe-provider`:

```bash
PYTHONPATH=source/backend python -m scaling_backend.preflight \
  --config /secure/course/api-runtime.json \
  --create-dirs \
  --probe-provider \
  --output-json /secure/course/preflight-provider.json
```

This probe remains non-submitting. When `QZ_COOKIE` is configured, the QZ path
uses it for the read-only distributed-training job-list endpoint with
`page_size=1`; otherwise it tries `QZ_COOKIE_FILE`, then performs the legacy CAS
login check only if neither cookie source is available. The Slurm path runs
`sinfo --version` and `sbatch --version` only.

For a local staff rehearsal with no provider network calls:

```bash
export SCALING_PROVIDER="fake"
export SCALING_STUDENT_KEYS_CSV="/tmp/scaling-student-keys.csv"
export SCALING_INTERNAL_CALLBACK_TOKEN="dev-internal"
export SCALING_ADMIN_API_TOKEN="dev-admin"
export SCALING_CALLBACK_URL="http://127.0.0.1:8000/internal/provider-events"
export SCALING_MANIFEST_BASE_URI="memory://manifests"
export SCALING_LOCAL_MANIFEST_DIR="/tmp/scaling-manifests"
uvicorn scaling_backend.asgi:app --host 127.0.0.1 --port 8000
```

When `SCALING_LOCAL_MANIFEST_DIR` is set, every accepted exploratory run and
staff-launched final run writes a canonical frozen manifest JSON file before
provider submission, and the provider receives that manifest's `file://` URI.
This is suitable for local rehearsals and shared-volume Docker deployments.
Manifest files are written through same-directory temporary files and atomic
replacement so an interrupted write does not corrupt an existing frozen
manifest.

For QZ deployments without a shared filesystem, set
`SCALING_PUBLISHED_MANIFEST_DIR` and `SCALING_PUBLISHED_MANIFEST_BASE_URI`
together. The backend writes canonical JSON under the local directory and passes
`<base-uri>/<run-id>.json` to the provider. Deployment tooling is responsible for
syncing or serving that directory so the worker image can read the returned URI.
The bundled worker can load `http://` and `https://` manifest URIs directly;
object-store schemes such as `s3://` and `gs://` require deployment-specific
worker support or an internal HTTP(S) serving layer.
Published manifest settings take precedence over `SCALING_LOCAL_MANIFEST_DIR`
when both are present.

## Dispatcher Boundary

`scaling_backend.dispatcher.ExperimentDispatcher` is the first dispatcher-facing
integration point. It records an in-memory experiment record, calls the provider
adapter, tracks provider job IDs, polls provider status, and records cancellation
events. It is intentionally small and database-free; the production scheduler can
replace the in-memory store while keeping the same provider contract.

The dispatcher expects frozen manifests with at least:

```json
{
  "manifest_version": 1,
  "run_kind": "exploratory",
  "experiment_id": "exp-0001",
  "student_id": "student-0001",
  "manifest_uri": "file:///secure/course/frozen-manifests/exp-0001.json",
  "callback_url": "http://<websocket-forward-host>:8000/internal/provider-events",
  "callback_base_url": "http://<websocket-forward-host>:8000/internal/provider-events",
  "model_config": {},
  "training_config": {},
  "resolved_config": {},
  "code_version": "course-backend-v0",
  "data_manifest_id": "exploratory-train-v0",
  "eval_manifest_id": "exploratory-eval-v0",
  "data_config": {"train_tokens": 1000000},
  "max_runtime_seconds": 3600,
  "runtime_config": {
    "reserved_runtime_seconds": 3600,
    "max_runtime_seconds": 3600
  },
  "created_at": "2026-07-16T15:00:00Z"
}
```

`callback_url` and `runtime_config.reserved_runtime_seconds` are kept as
compatibility aliases for the current worker and adapter path. New code should
prefer the top-level `callback_base_url` and `max_runtime_seconds` fields.

## Service Boundary

`scaling_backend.service.ExperimentService` is the first submit-to-result control
plane. It is intentionally in-memory, but it covers the core version-one policy
shape:

- stable same-student config hashing
- hard submit-time config validation and resolved metadata snapshots
- duplicate same-student config detection without a second budget charge
- separate experiments for identical configs from different students
- per-student budget reservation and charging
- frozen provider manifest construction
- worker callback ingestion for validation, completion, and failure events
- student-scoped result lookup
- student-scoped experiment listing
- mutable per-student final submissions with prediction interval validation
- in-memory final-submission freeze and final-run launch workflow
- in-memory instructor/admin inspection, cancellation, budget adjustment, and
  export records

This layer is the natural backing object for the first FastAPI routes:

```text
POST /submit              -> ExperimentService.submit(...)
GET /budget               -> ExperimentService.get_budget(...)
GET /experiments          -> ExperimentService.list_results(...)
GET /experiment/{id}      -> ExperimentService.get_result(...)
POST /final_submission    -> ExperimentService.set_final_submission(...)
GET /final_submission     -> ExperimentService.get_final_submission(...)
freeze final submissions  -> ExperimentService.freeze_final_submissions(...)
launch final runs         -> ExperimentService.launch_final_runs(...)
list all experiments      -> ExperimentService.admin_list_experiments(...)
inspect experiment detail -> ExperimentService.admin_get_experiment(...)
inspect queue             -> ExperimentService.admin_queue_snapshot(...)
poll provider status      -> ExperimentService.admin_poll_experiment(...)
cancel experiment         -> ExperimentService.admin_cancel_experiment(...)
budget adjustment         -> ExperimentService.admin_apply_budget_adjustment(...)
export course records     -> ExperimentService.admin_export_course_records(...)
POST /internal/provider-events
                         -> ExperimentService.record_worker_event(...)
```

The next persistence step is replacing the in-memory dictionaries with database
tables while preserving the service behavior covered by tests.

Current budget accounting semantics:

- rejected configs charge no budget and do not allocate experiment IDs
- duplicate same-student configs charge no additional budget
- queued and submitted jobs reserve their requested runtime
- completed jobs charge reported runtime clipped to `[1, reserved_runtime]`
- timeout failures charge the full reserved runtime
- other worker failures charge reported runtime clipped to
  `[1, reserved_runtime]`
- provider polling can update submitted/queued/running/failure visibility, but
  provider success without a worker completion callback is marked `lost` rather
  than converted into a completed result
- QZ provider polling reads both the job-detail endpoint and the worker log
  endpoint; log evidence such as nonzero `processExitCode` can mark a stale
  provider `RUNNING` detail as failed
- worker-reported losses must be finite before they are recorded
- internal worker callbacks must use a known `event_type`; unknown event names
  return `invalid_worker_event` and do not mutate experiment state

Current config validation semantics:

- `config.model` and `config.training` must be JSON objects
- `model.num_hidden_layers` and `model.hidden_size` are required positive
  integers
- `model.head_dim`, `model.intermediate_size`,
  `model.num_attention_heads`, `model.num_key_value_heads`, and
  `model.vocab_size` are positive integers when supplied, with conservative
  defaults for omitted optional fields
- `model.vocab_size` defaults to `50432` and must be at least `50432` for the
  current `EleutherAI/gpt-neox-20b` tokenized dataset; smaller vocabularies can
  make token IDs exceed the embedding table and are rejected before launch
- if `model.head_dim` is supplied, `model.hidden_size` must equal
  `model.num_attention_heads * model.head_dim`; otherwise `model.hidden_size`
  must be divisible by `model.num_attention_heads`
- `model.num_attention_heads` must be divisible by
  `model.num_key_value_heads`, and the current Stanford reference runtime
  requires `model.num_key_value_heads == model.num_attention_heads`
- `model.head_dim` must be `<= 128` and a multiple of `8` for the GPU
  attention backend
- `model.intermediate_size` must be greater than or equal to
  `model.hidden_size`
- `model.attention_bias` and `model.tie_word_embeddings` must be booleans when
  supplied
- `model.rms_norm_eps` must be positive finite, and `model.rope_theta` must be
  a positive integer
- `model.dtype` is restricted to `float32` or `bfloat16`
- `training.train_tokens` is required, positive, and at most `500000000000`
- `training.sequence_length`, `training.train_batch_size`,
  `training.validation_batch_size`, and `training.num_evals` are positive
  integers when supplied
- `training.train_tokens` must be divisible by
  `training.sequence_length * training.train_batch_size`, so accepted configs
  never train on a silently rounded-down token count
- total optimizer steps must be divisible by `training.num_evals`
- staff-configured validation tokens per evaluation must be divisible by
  `training.sequence_length * training.validation_batch_size`
- when the staff runtime declares `SCALING_WORKER_GPU_COUNT > 1`, both
  `training.train_batch_size` and `training.validation_batch_size` must be
  divisible by that worker GPU count because the Stanford trainer shards the
  batch axis across the FSDP mesh
- when `SCALING_WORKER_GPU_COUNT > 1`, `model.hidden_size` and
  `model.intermediate_size` must be divisible by that worker GPU count because
  Stanford FSDP parameter sharding partitions those axes; when
  `model.tie_word_embeddings` is false, `model.vocab_size` must also be
  divisible by the worker GPU count because the untied output head is sharded on
  its vocabulary axis
- `training.model_seed` is a non-negative integer and controls model
  initialization only
- `training.learning_rate` is required, positive, and finite
- `training.optimizer` is restricted to `adamw` or `sgd`, and
  `training.lr_schedule` is restricted to `cosine` or `constant`
- `training.weight_decay` must be non-negative and finite
- `training.adam_beta1` and `training.adam_beta2` must be finite fractions in
  `[0, 1)`
- `training.adam_epsilon` and `training.gradient_clip_norm` must be positive and
  finite
- `training.warmup_fraction` must be finite and in `[0, 1)`;
  `training.final_lr_fraction` must be finite and in `[0, 1]`
- accepted configs get a `resolved_config` snapshot containing parameter-count
  estimates, tokens per optimizer step, total optimizer steps, eval cadence,
  validation tokens and batches per eval, estimated training FLOPs, code
  version, data/eval manifest IDs, runtime cap, provider manifest URI, and
  resource warnings

## Public API Boundary

`scaling_backend.api.create_app(...)` builds the current FastAPI app around an
`ExperimentService` instance. It deliberately keeps authentication and callback
authorization simple for version one:

- student requests use `Authorization: Bearer <student API key>`
- staff config maps API keys to student IDs outside the request body
- worker callbacks use a separate internal bearer token
- staff/admin routes use a separate admin bearer token
- duplicate same-student submissions return `409` with the existing
  `experiment_id`
- cross-student result access returns `404` instead of revealing ownership

The public routes are:

```text
GET /budget
POST /submit
GET /experiments
GET /experiment/{experiment_id}
POST /final_submission
GET /final_submission
```

The internal worker route is:

```text
POST /internal/provider-events
```

The current staff/admin routes are:

```text
GET /admin/queue
POST /admin/poll-active
POST /admin/import-worker-events
GET /admin/experiments?student_id=<optional>&status=<optional>
GET /admin/students/{student_id}/budget
GET /admin/experiments/{experiment_id}
POST /admin/experiments/{experiment_id}/poll
POST /admin/experiments/{experiment_id}/cancel
POST /admin/experiments/{experiment_id}/system-failure
POST /admin/budget-adjustments
POST /admin/final-submissions/freeze
POST /admin/final-runs/launch
POST /admin/final-runs/{final_run_id}/poll
POST /admin/final-runs/{final_run_id}/system-failure
POST /admin/final-runs/{final_run_id}/cancel
GET /admin/exports
```

Student-facing routes should not expose QZ provider IDs, QZ payloads, raw QZ
errors, cookies, or staff deployment settings.
Worker callback `failure_type` values are normalized before they reach public
student result APIs. Known student-caused typed reasons such as `timeout`,
`numerical`, `resource`, `provider_rejected`, and `unknown_student_caused` pass
through as `failed` results. Worker-reported system reasons such as
`infrastructure_error`, including tokenized-index artifact failures, become
`system_failed`, release the active exploratory reservation without charging the
student, and keep raw type/message detail in staff-only `staff_failure_detail`.
When a worker-reported student failure, or a provider log tail from a job that
ended before sending a terminal callback, contains traceback-like compute-node
text, the student result may also include `failure_detail`. This field is a
sanitized public diagnostic intended for actionable runtime errors; it is not a
mirror of staff-only `staff_failure_detail`, and infrastructure/provider-secret
messages remain hidden.
Common raw worker exception names such as `FloatingPointError`, `TimeoutError`,
`MemoryError`, and malformed-result `WorkerResultError` are mapped to stable
typed categories.

`POST /admin/experiments/{experiment_id}/poll` refreshes the provider status for
an exploratory run and records the normalized provider status in the dispatcher
event ledger. It does not fabricate validation losses. If the provider reports
success before the worker has sent `worker_completed`, the experiment becomes
`lost` with `failure_reason = "missing_worker_completed_callback"` for staff
review.
`POST /admin/poll-active` applies the same polling semantics to every currently
active exploratory experiment and hidden final run. Terminal rows include a
`provider_artifacts` object with provider log/detail pointers. In production,
prefer the built-in `SCALING_PROVIDER_POLL_INTERVAL_SECONDS` loop or an external
scheduler that calls this endpoint regularly.
`POST /admin/experiments/{experiment_id}/system-failure` marks an exploratory
run as `system_failed` with a typed system failure reason such as
`provider_outage`, releases any active reservation, optionally records a refund
as a negative budget adjustment, and writes an admin action for audit.
Completed and terminal failed exploratory results include `used_runtime_seconds`,
`completed_at`, and `failed_at`. `used_runtime_seconds` is the billable runtime
after applying the course clipping and timeout policy, so timeouts report the
full reserved runtime while numerical/resource failures report clipped runtime.

`POST /submit` returns the accepted experiment ID, reservation, status,
top-level `resource_warnings`, and a student-facing `resolved_config` metadata
snapshot with private data/eval manifest IDs and provider manifest URIs redacted.
The full resolved snapshot is still stored in the frozen provider manifest so
the worker and later grading exports can reconstruct the accepted run without
reinterpreting mutable defaults. Frozen manifests use `manifest_version = 1`,
explicit `run_kind`, `code_version`, private data/eval manifest IDs,
`created_at`, and top-level `max_runtime_seconds` fields. Student result
payloads repeat `resource_warnings` as first-class metadata so risky accepted
configurations remain visible after the initial submit response.

Public API errors use stable JSON bodies:

```json
{
  "error": "invalid_config",
  "message": "model.hidden_size must be divisible by model.num_attention_heads"
}
```

Recoverable duplicate submissions return `409` with the same shape plus the
same-student `experiment_id`. Current public error codes include
`invalid_api_key`, `invalid_config`, `duplicate_config`, `experiment_not_found`,
`invalid_final_submission`,
`final_submission_not_found`, `invalid_callback_token`, `invalid_admin_token`,
`invalid_worker_event`, and `invalid_admin_operation`.

## Final Run Boundary

Final submissions are mutable until `ExperimentService.freeze_final_submissions`
is called by staff. Freezing copies the latest per-student submission into an
immutable in-memory snapshot with `frozen_at`, `frozen_by`, and `freeze_reason`.
The API validates final training configs before storing them; invalid configs
return `invalid_final_submission` and cannot be frozen. After freezing, further
final-submission edits are rejected.

`ExperimentService.launch_final_runs(...)` submits one provider job per frozen
submission. The launch is idempotent: repeated calls return the existing final
run records and do not submit duplicate provider jobs. Final-run manifests use
`manifest_version = 1` and `run_kind = "final"`, carry the `final_run_id`,
`final_submission_id`, prediction interval, resolved config, final train
manifest ID, final eval manifest ID, `created_at`, and top-level
`max_runtime_seconds`. They also include worker-facing `model_config`,
`data_config`, and `runtime_config` fields so the same worker entrypoint can
execute exploratory and final runs.
Workers report final-run validation, completion, and failure events through the
same internal provider callback endpoint using `final_run_id` instead of
`experiment_id`. Final-run callbacks update the hidden-eval record with
`actual_final_validation_loss`, `validation_losses`, status, and failure reason
for later grading exports. Final runs are staff-controlled and do not consume
student exploratory budget.

`POST /admin/final-runs/{final_run_id}/poll` refreshes the provider status for a
hidden final run. As with exploratory polling, provider success without a
`worker_completed` callback is marked `lost` with
`failure_reason = "missing_worker_completed_callback"` and does not create an
actual validation loss.
`POST /admin/final-runs/{final_run_id}/cancel` stops the provider job for an
active hidden final run and marks it `cancelled` with
`failure_reason = "admin_intervention"`. It does not alter exploratory student
budget.
`POST /admin/final-runs/{final_run_id}/system-failure` marks a hidden final run
as `system_failed` with a typed system reason such as `infrastructure_error`,
clears any actual hidden-eval loss, preserves staff-only detail, and writes an
admin action for audit. Final-run system-failure overrides do not adjust
exploratory student budget.
Worker-reported hidden-run system failures use the same `system_failed` status
and are marked for staff review rather than automatic worst-score treatment in
grading exports.

## Admin And Export Boundary

The first admin surface lives at the service layer so it can back either scripts
or instructor-only routes later. Current supported operations are:

- list all experiments across students with optional student/status filters,
  status, losses, config hash, and resolved metadata
- inspect a student's budget by student ID
- inspect experiment detail, including staff-only submitted config, worker event
  ledger, and provider log/detail artifact pointers
- inspect active exploratory and final queues separately, including per-student
  active counts
- cancel an experiment through the configured provider adapter
- mark instructor-reviewed system failures and optional refunds through an
  audited admin path
- record audited budget adjustments as separate records instead of mutating
  experiment rows silently
- export in-memory course records for experiments, budget snapshots, budget
  adjustments, frozen final submissions, and final runs, including prediction
  intervals and actual hidden-eval losses once final-run callbacks arrive
- write JSONL/CSV export artifacts through
  `ExperimentService.admin_write_course_exports(...)`

Budget adjustments accept positive or negative seconds. Negative adjustments are
refunds; positive adjustments are manual charges or penalties. Each adjustment
records the actor, reason, timestamp, target student, optional experiment ID,
and charged-budget values before and after the adjustment.
The current export shape is dictionary-of-record-lists and is intended as the
source for JSONL/CSV admin scripts. The current file writer emits:

```text
experiments.jsonl
experiments.csv
experiment_events.jsonl
worker_events.jsonl
budget_snapshots.jsonl
budget_snapshots.csv
budget_adjustments.jsonl
admin_actions.jsonl
final_submissions.jsonl
final_submissions.csv
final_runs.jsonl
final_runs.csv
grading.csv
```

JSONL files preserve complete nested records. CSV files provide stable summary
columns for spreadsheet inspection and grading pipelines, including exploratory
provider identity/status fields, failure reasons, and staff-only
`staff_failure_detail` diagnostics where applicable. `budget_snapshots.csv`
captures each known student's total, reserved, charged, and remaining seconds at
export time, while `budget_adjustments.jsonl` preserves the separate manual
adjustment ledger. Staff JSONL records preserve nested `provider_artifacts`;
`experiments.csv` and `final_runs.csv` expose `provider_logs_uri` and
`provider_detail_uri` for direct provider-log and job-detail lookup during
incident review.
Final-run JSONL records and `grading.csv` include prediction metrics when an
actual hidden final validation loss is available: `prediction_absolute_error`,
`prediction_interval_covered`, `prediction_interval_width`,
`prediction_interval_miss_distance`, `prediction_interval_score`, and the
lower-is-better `prediction_quality_penalty`. The interval score uses the
student handout policy for an 80% central prediction interval:
`width + 10 * miss_distance`; the combined penalty is
`0.5 * absolute_error + 0.5 * interval_score`. The fields remain empty/null for
pending, failed, or system-failed final runs without an actual loss. The
grading CSV records `final_run_grading_outcome` and
`final_run_score_policy` so student-caused final failures are marked for
worst-score treatment and system failures are held for staff review. It also
reserves methodology/writeup columns for the external report workflow:
`methodology_report_score`, `methodology_report_notes`, `analysis_code_score`,
`analysis_code_notes`, and `writeup_artifact_uri`.

The first operator-facing script surface is the HTTP-backed admin CLI:

```bash
export SCALING_ADMIN_BASE_URL="https://backend.example"
export SCALING_ADMIN_API_TOKEN="<staff-admin-token>"

python -m scaling_backend.admin_cli seed-student-keys --roster-csv /secure/course/roster.csv --output-csv /secure/course/student_api_keys.csv
python -m scaling_backend.admin_cli queue
python -m scaling_backend.admin_cli poll-active
python -m scaling_backend.admin_cli experiments --student-id student-1 --status running
python -m scaling_backend.admin_cli student-budget student-1
python -m scaling_backend.admin_cli experiment exp-000001
python -m scaling_backend.admin_cli poll exp-000001
python -m scaling_backend.admin_cli cancel exp-000001 --reason "quota audit" --actor staff
python -m scaling_backend.admin_cli mark-system-failure exp-000001 --failure-reason provider_outage --staff-failure-detail "QZ outage INC-42" --refund-seconds 120 --reason "confirmed provider outage" --actor staff
python -m scaling_backend.admin_cli adjust-budget student-1 --seconds -120 --reason "provider outage refund" --actor staff --experiment-id exp-000001
python -m scaling_backend.admin_cli freeze-finals --reason deadline --actor staff
python -m scaling_backend.admin_cli launch-finals --max-runtime-seconds 7200 --reason deadline --actor staff
python -m scaling_backend.admin_cli poll-final final-run-000001
python -m scaling_backend.admin_cli mark-final-system-failure final-run-000001 --failure-reason infrastructure_error --staff-failure-detail "hidden eval outage INC-55" --reason "confirmed hidden eval outage" --actor staff
python -m scaling_backend.admin_cli cancel-final final-run-000001 --reason "staff dry-run cleanup" --actor staff
python -m scaling_backend.admin_cli exports
python -m scaling_backend.admin_cli exports --output-dir /secure/course/exports/$(date +%F)
```

When shared-storage worker-event fallback is configured, add:

```bash
python -m scaling_backend.admin_cli import-worker-events
```

The CLI deliberately talks to the running FastAPI admin routes instead of
constructing its own service instance. That keeps runtime state, authentication,
and audit records centralized in the backend process.
The `queue` output separates exploratory and final-run queues. Exploratory rows
include `queue_position` for raw FIFO order and `fair_share_rank` for the
round-robin staff scheduling view across students; `fair_share_order` exposes
that fair-share ordering directly. Final-run rows have their own
`queue_position` because hidden final evaluation remains separate from
exploratory sharing.
By default `exports` prints the raw `/admin/exports` JSON payload. With
`--output-dir`, it writes the same JSONL/CSV export bundle listed above and
prints the generated path map.

For an offline course-readiness rehearsal with no live QZ calls:

```bash
PYTHONPATH=source/backend python -m scaling_backend.rehearsal \
  --output-dir /tmp/scaling-rehearsal
```

The rehearsal uses the fake provider, seeds two deterministic students in code,
submits exploratory runs, executes worker callbacks, freezes final submissions,
launches final runs, writes export artifacts under
`/tmp/scaling-rehearsal/exports`, and writes
`/tmp/scaling-rehearsal/rehearsal-report.json`. The report includes
`acceptance_checks`, queue snapshots before and after final-run launch, export
paths, and `/tmp/scaling-rehearsal/operations.log`. This is a local staff smoke
test for the admin/export workflow; it does not validate live QZ credentials or
the production trainer image. See `docs/instructor-operations-guide.md` for the
full staff workflow and readiness checklist.

For one launch-readiness gate that combines preflight, SOT milestone evidence,
release audit, and local rehearsal:

```bash
PYTHONPATH=source/backend python -m scaling_backend.readiness \
  --create-dirs \
  --rehearsal-output-dir /tmp/scaling-readiness-rehearsal \
  --output-json /secure/course/readiness.json
```

The readiness gate returns one JSON report with `preflight`, `sot_status`,
`release_audit`, and `rehearsal` sections. It exits non-zero if preflight fails,
any SOT milestone is missing implementation evidence, release audit has failed
checks, or any rehearsal acceptance check is false. Add `--probe-provider` in
the target provider environment to include the explicit non-submitting QZ/Slurm
connectivity probe.

For a static release-package audit before distributing student/staff docs:

```bash
PYTHONPATH=source/backend python -m scaling_backend.release_audit \
  --output-json /tmp/scaling-release-audit.json
```

The audit checks quickstarts, the workflow notebook, staff operations docs,
provider adapter guide, preflight/SOT-status/rehearsal tooling, stale public API
terms, and LaTeX source styling. It warns, rather than fails, when native PDF
tools such as `latexmk` or `xelatex` are not installed.

## Worker Boundary

`python -m scaling_backend.worker.run` is the command shape the QZ distributed
job should execute inside the staff-built Docker image:

```bash
python -m scaling_backend.worker.run \
  --manifest-uri /path/to/frozen-manifest.json \
  --callback-url http://<websocket-forward-host>:8000/internal/provider-events \
  --callback-token-env SCALING_CALLBACK_TOKEN \
  --trainer course_trainer.worker:train \
  --heartbeat-interval-seconds 60 \
  --tokenized-index-validation-mode metadata
```

The worker module loads and validates the manifest, emits started, heartbeat,
validation, completed, and failed event payloads, and loads the training function
from a `module:function` trainer spec. By default, worker events are posted to
`--callback-url` using `--callback-token-env`. With explicit `--event-sink jsonl`,
the same payloads are appended to `--event-log-path` as newline-delimited JSON;
the admin endpoint `POST /admin/import-worker-events` imports them from
`SCALING_WORKER_EVENT_IMPORT_DIR` and deduplicates replayed lines by `event_id`.
Staff can pass `--trainer` directly or set `SCALING_WORKER_TRAINER` inside the worker image. The QZ adapter reads
`QZ_WORKER_TRAINER` and appends the corresponding `--trainer` flag to the
distributed-task command. When `QZ_WORKER_CONDA_ENV` is configured, the adapter
wraps the worker command with `bash -lc`, sources `QZ_WORKER_CONDA_INIT` when
configured, activates that conda environment, and then `exec`s the Python
worker. In HTTP mode the wrapper also exports the callback token before worker
startup. Staff can enable wrapper-level periodic heartbeats
with `--heartbeat-interval-seconds`,
`SCALING_WORKER_HEARTBEAT_INTERVAL_SECONDS`,
`QZ_WORKER_HEARTBEAT_INTERVAL_SECONDS`, or
`SLURM_WORKER_HEARTBEAT_INTERVAL_SECONDS`; `0` disables the automatic heartbeat
loop. Heartbeat is a liveness signal only while the Python worker process is
alive; if the process is killed by the platform, segfaults, or exits through
native code before Python regains control, it cannot emit a terminal callback.
Provider polling and provider logs are the backstop for those cases. Exploratory
manifests identify callbacks with `experiment_id`; final-run manifests identify
callbacks with `final_run_id`.
Worker-facing `model_config` and `training_config` sections are copied from the
accepted `resolved_config`, not re-derived from raw student input, so omitted
defaults and normalized choices are frozen before provider dispatch.
`model_config` uses the same Stanford canonical model fields documented in
`docs/scaling-laws-api-quickstart.md` for `config.model`, including the resolved
`head_dim`. It is not a second schema layer and does not accept legacy model
field names.
The manifest also includes `validation_config`, which freezes the staff-owned
eval manifest ID, validation tokens per evaluation, and exact validation batch
count derived from the accepted `validation_batch_size`.
When staff include `data_config.tokenized_index_uri` or
`validation_config.tokenized_index_uri`, the worker preflights the referenced
tokenized `index.json` before importing the trainer loop. The preflight accepts
local paths, `file://`, `http://`, and `https://` URIs, requires safe relative
`uint32` shard paths with positive token counts and SHA-256 metadata, verifies
the declared total token count when present, and supports two validation modes.
`metadata` mode avoids full shard reads and is the production default for mounted
100B-scale reservoirs. `full` mode additionally reads and hashes every shard,
and should be reserved for offline audits or small rehearsals. Production
deployments should point these fields at the run-scoped train/eval indexes the
worker will actually consume, not at a larger reservoir index that is unrelated
to the submitted run.
The worker accepts both exploratory and final `manifest_version = 1` manifests.
It currently requires `student_id`, `model_config`, `data_config`,
`validation_config`, and `runtime_config`, and chooses `experiment_id` or
`final_run_id` as the callback identity. The trainer should read the canonical
top-level `max_runtime_seconds` when available; `runtime_config.max_runtime_seconds`
and `runtime_config.reserved_runtime_seconds` are present for compatibility.
The bundled worker also starts a local runtime watchdog around trainer
execution using the same canonical limit. On Unix-like main-thread execution,
the watchdog raises `TimeoutError`, emits `worker_failed`, and lets the backend
apply the stable `timeout` failure policy. Provider-level wall-clock limits
should remain configured as the hard outer guard.
Worker callbacks are stored in the staff event ledger and exported as
`worker_events.jsonl`. The legacy `experiment_events.jsonl` export remains as
an exploratory-only compatibility stream; student-facing result routes expose
only status, losses, and failure category. The worker can load inline JSON,
local paths, and
`file://`, `http://`, and `https://` manifest URIs. It rejects unsupported remote
schemes such as `s3://` and `gs://` with `WorkerManifestError` unless the worker
image adds its own object-store support. The configured trainer is responsible
for running the course training loop exactly as specified by the frozen manifest
and returning finite non-empty `validation_losses`, finite
`final_validation_loss` equal to the last validation loss, and positive
`actual_runtime_seconds`. If the trainer
returns malformed or non-finite results, the worker emits `worker_failed` with
`failure_type = "WorkerResultError"` instead of sending an invalid
`worker_completed` callback. Tokenized-index preflight failures are normalized
by the backend as `infrastructure_error`, because they indicate staff data
artifact or runtime-storage problems rather than student hyperparameter choices.
`CourseTrainerConfigError` failures from `course_trainer.worker:train` are
normalized as `invalid_runtime_config`, because they indicate a frozen student
configuration that the Stanford reference backend cannot execute exactly. The
backend also treats legacy worker `ValueError` reports that match known JAX FSDP
sharding-divisibility failures as `invalid_runtime_config` instead of
`unknown_student_caused`, so older manifests still surface the actionable
student-side cause.

Run the provider tests with:

```bash
PYTHONPATH=source/backend pytest -q source/backend/tests/providers
```

Run all current backend tests with:

```bash
PYTHONPATH=source/backend pytest -q source/backend/tests
```
