# Provider Adapter Implementation Guide

Date: 2026-07-16

Related SoT: `docs/scaling-assignment-sot.md`

Status: Draft for implementation handoff

This guide defines the provider abstraction needed to replace the original
Modal-specific dispatch path with course-controlled execution backends. It is
written so the course team can implement adapters for the internal GPU
submission API and Slurm without changing the public assignment API.

## 1. Purpose

The backend owns assignment policy:

- authentication
- budget accounting
- queue selection
- experiment status
- final submission freeze
- grading exports

The provider adapter owns infrastructure translation:

- submit a training job to a concrete execution system
- poll or receive provider status
- cancel a job when requested
- return log/artifact pointers
- map provider failures into backend status categories

The adapter must not make assignment policy decisions. It should not decide
whether a student is charged, whether a final run is valid, or whether a refund
is issued. It reports facts and normalized status; the backend applies policy.

## 2. Providers to Support

Version one should support three provider types:

- `fake`: local/dev provider for tests and submit-to-result smoke runs
- `slurm`: adapter for cluster testing where available
- `qz_distributed`: deployment target for course-controlled accelerator jobs
  submitted through the QZ distributed-training API

Only one provider needs to be active in a deployment. The dispatcher should be
configured with the selected provider.

The cloned `refs/qzcli_tool/` project is reference material only. The course
backend should not import it or shell out to it in production. The native
`qz_distributed` adapter should port only the platform logic we need:
browser-compatible cookie requests, `Set-Cookie` refresh handling, optional CAS
authentication fallback, distributed train-job payload construction, status
polling, cancellation, and log retrieval.

## 3. Core Concepts

### Backend Experiment ID

The backend experiment ID is the canonical assignment identifier. It is scoped
by the backend database and should be included in every provider manifest and
log path.

### Provider Job ID

The provider job ID is the concrete execution-system identifier returned by the
adapter after job submission. It may be a Slurm job ID, internal API task ID, or
local fake ID.

### Frozen Job Manifest

The frozen manifest is the immutable payload passed to a worker. It contains
all information needed to run the training job exactly as accepted by the
backend.

### Worker Event Channel

Workers report events and terminal status either through backend internal HTTP
endpoints or through shared-storage JSONL files imported by the control node.
Provider adapters should not infer validation losses from logs when the worker
event channel is working.

## 4. Provider Interface

The backend dispatcher needs four operations.

### Submit Job

Input: frozen job manifest.

Output:

- provider job ID
- provider name
- provider submission timestamp
- optional provider metadata safe for staff logs

Required behavior:

- submit exactly one job for one backend experiment
- return a stable provider job ID if the provider accepted the job
- raise a provider-submission error if the provider rejects the job before it
  starts
- never expose other students' jobs or identifiers

### Get Job Status

Input: provider job ID.

Output:

- normalized provider status
- raw provider status for staff logs
- timestamps if available
- failure message for staff logs if available
- artifact/log pointers if available

Normalized provider statuses:

- `submitted`
- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`
- `lost`
- `unknown`

### Cancel Job

Input:

- provider job ID
- cancellation reason
- actor or system component requesting cancellation

Output:

- cancellation accepted flag
- normalized provider status after cancellation if available
- raw provider response for staff logs

Cancellation should be best-effort. The backend remains responsible for
deciding the final experiment status and budget effect.

### Get Artifacts

Input: provider job ID.

Output:

- staff log URI or path
- worker output URI or path
- provider console URI if available
- artifact metadata safe for staff records

Public student APIs should not expose raw provider console links unless they are
explicitly sanitized.

## 5. Frozen Job Manifest Schema

Every provider should receive a `manifest_version = 1` JSON manifest. The first
implementation emits these fields:

- `manifest_version`: currently `1`
- `run_kind`
  - `exploratory`
  - `final`
- `experiment_id`, for exploratory runs
- `final_run_id`, for final runs where applicable
- `final_submission_id`, for final runs where applicable
- `student_id`
- `model_config`
- `training_config`
- `resolved_config`
- `config_hash`
- `code_version`
- `data_manifest_id`
- `eval_manifest_id`
- `data_config`
- `validation_config`
- `max_runtime_seconds`
- `runtime_config`
- `callback_base_url`
- `created_at`

`model_config` is the resolved copy of the public `config.model` object defined
in `docs/scaling-laws-api-quickstart.md`. It uses the Stanford canonical model field names
(`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `head_dim`, and
related fields), with defaults filled and normalized before provider dispatch.
It is not a separate provider schema, and legacy model field names are not valid
in frozen manifests.

The current worker and QZ adapter also receive compatibility aliases:

- `callback_url`, equivalent to `callback_base_url`
- `runtime_config.reserved_runtime_seconds`, equivalent to
  `max_runtime_seconds`
- `runtime_config.max_runtime_seconds`, equivalent to `max_runtime_seconds`

Deployment-specific fields such as `container_image`, `callback_auth_secret_ref`,
`output_uri`, and `log_uri` may be added later by the provider or scheduler
layer, but version one does not require them in the service-built manifest.

The manifest may contain private manifest IDs, but it should not expose exact
public-source dataset names, mixture weights, or final evaluation split details
to student-facing outputs.
For worker images that consume pre-tokenized data directly, staff may include
`data_config.tokenized_index_uri` and
`validation_config.tokenized_index_uri`. These URIs should point to the
run-scoped train and evaluation `index.json` files in the audited tokenized
format produced by `source/data`. They may be local paths, `file://`, `http://`,
or `https://` URIs visible from the compute-node worker container. The
control-node API passes these as opaque manifest references and should not mount
or validate the dataset itself. The bundled worker
preflights these indexes before trainer execution, verifies safe relative
`uint32` shard entries, declared token counts, SHA-256 metadata format, optional
local file sizes, and `total_tokens` consistency. Production worker commands
should use `--tokenized-index-validation-mode metadata` to avoid reading the full
mounted reservoir at startup. `full` mode additionally hashes every shard and is
intended for offline audits or small rehearsals. The worker emits
`TokenizedIndexError` on validation failure.
Runtime deployments populate these fields with the private
`SCALING_TOKENIZED_TRAIN_INDEX_URI`,
`SCALING_TOKENIZED_VALIDATION_INDEX_URI`,
`SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI`, and
`SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI` environment settings when those
settings are present.
`validation_config` freezes the staff-controlled evaluation manifest ID,
validation-token count per evaluation, and exact validation batch count derived
from the accepted student `validation_batch_size`; adapters and trainers should
not recompute or round these values.

## 6. Worker Event Contract

Workers should report events to the backend through HTTP callbacks or the
shared-storage JSONL import path.

Accepted backend `event_type` values:

- `worker_started`
- `heartbeat`
- `validation`
- `worker_completed`
- `worker_failed`

Unknown event names are rejected as invalid worker events and must not be
treated as successful callbacks.

Heartbeats are advisory liveness callbacks from the worker process, not a
durable crash detector. They stop when the Python process is killed by the
platform, segfaults in native code, or exits before the exception handler can
send `worker_failed`. The provider adapter must therefore expose a reliable
`get_status` path, and production deployments should run active-job polling in
addition to accepting worker callbacks.

Recommended event fields:

- backend experiment ID
- provider job ID
- worker run ID if distinct
- timestamp
- elapsed seconds
- validation losses when applicable
- failure category when applicable
- staff-only failure detail when applicable

Terminal completion payload:

- used runtime seconds
- validation-loss sequence
- final status
- artifact/log pointers

Terminal failure payload:

- used runtime seconds
- public failure category
- partial validation losses, if any
- staff-only failure detail
- artifact/log pointers

## 7. Failure Mapping

Provider adapters should map concrete provider states into backend categories.

Examples:

- provider queue rejection before launch: `provider_rejected`
- worker process exits due non-finite loss: `numerical`
- worker process exits due runtime timeout: `timeout`
- worker process exits due out-of-memory: `resource`
- provider node lost or platform outage: `infrastructure_error`
- admin cancels job: `admin_intervention`
- provider cannot find job after accepted submission: `worker_lost` or
  `unknown_system`, depending on evidence

The adapter should preserve raw provider details in staff logs. Public APIs
should expose only stable categories and sanitized messages.

## 8. Timeout Policy

The backend stores `max_runtime_seconds` and passes it in the manifest.

The bundled worker enforces a local runtime watchdog around trainer execution
using the canonical top-level `max_runtime_seconds` value when possible. On
Unix-like main-thread execution, the watchdog raises `TimeoutError` and emits a
`worker_failed` callback. The provider should also enforce a wall-clock limit as
the hard outer guard.

For QZ, status polling reads the distributed-training detail endpoint and the
worker log endpoint. The adapter keeps a bounded log tail in staff-only provider
metadata and treats explicit process-exit evidence, such as nonzero
`processExitCode` or `exit code`, as provider failure even if the detail endpoint
has not yet advanced from `RUNNING`. Terminal active-poll rows include provider
artifact pointers so staff can inspect the raw QZ logs.

If timeout is reached:

- worker reports `timeout` when it can
- provider timeout without worker callback is mapped to `timeout` if evidence
  shows the job exceeded its allowed runtime
- ambiguous provider termination should be mapped to a staff-review category

Timeouts are generally student-caused and charged as the full reserved runtime.

## 9. No Checkpoint/Resume

Version one does not support checkpoint/resume. Provider adapters should not
resume partially completed training jobs. A job either completes from scratch,
fails, is cancelled, or is marked for staff review.

This protects scaling-law measurements from recovery artifacts.

## 10. Slurm Adapter Notes

The implemented Slurm adapter translates each frozen manifest into:

- job script
- manifest URI passed to the worker entrypoint
- stdout/stderr paths
- time limit
- accelerator resource request
- working directory

Current Slurm behavior:

- write an `sbatch` script to `SLURM_WORK_DIR/scripts`
- submit with `sbatch --parsable`
- store Slurm job ID as provider job ID
- poll with `squeue`, falling back to `sacct` for terminal jobs
- cancel with `scancel`
- map Slurm states to normalized provider statuses
- collect stdout/stderr/script `file://` pointers as artifacts
- configure resource requests through positive integer staff env vars:
  `SLURM_GPUS_PER_TASK`, `SLURM_CPUS_PER_TASK`, `SLURM_MEMORY_GB`, and
  `SLURM_TIME_LIMIT_MINUTES`
- pass `SLURM_WORKER_HEARTBEAT_INTERVAL_SECONDS` to the worker command when
  staff want wrapper-level periodic heartbeat callbacks

The Slurm adapter is useful for testing even if the internal GPU API is the
main deployment target.

## 11. QZ Distributed Training Adapter Notes

The `qz_distributed` adapter runs in the control-node API backend and should
translate each frozen manifest into one QZ distributed-training task. The task
starts on compute nodes from a staff-built Docker image that already contains
the training worker, Stanford-style training backend, data loader, and all
integration-tested runtime dependencies. The task command should execute that
worker against the immutable manifest accepted by the API backend.

The first-version compute image should install both the Stanford reference
backend and the local trainer adapter package:

```bash
python -m pip install -c deploy/compute-node/constraints.txt './refs/cs336-assignment3-scaling[server]'
python -m pip install ./source/backend ./source/data ./source/training
```

The `[server]` extra is the CUDA 13 path. For compute nodes whose installed
NVIDIA driver supports only CUDA 12.x, install
`deploy/compute-node/requirements-cuda12.txt`, then install `flash-hog` and the
Stanford package with `--no-deps` so the CUDA 13 JAX plugin is not pulled into
the worker image. The CUDA 12 requirements file keeps runtime transitive
packages such as `chex==0.1.91` explicit because those `--no-deps` installs will
not resolve them.

`source/training` provides `course_trainer.worker:train`. That callable converts
frozen backend manifests into Stanford-style training configuration, reads the
mounted flat-binary `source/data` tokenized indexes lazily, and invokes the
local CS336 model, optimizer, and training loop modules when the Stanford
dependencies are installed in the compute image.

Expected responsibilities:

- submit one distributed-training task per experiment or final run
- pass the manifest URI or manifest path to the prebuilt worker image
- pass the worker trainer callable as `module:function` when configured through
  `QZ_WORKER_TRAINER`
- pass `QZ_WORKER_HEARTBEAT_INTERVAL_SECONDS` to the worker command when staff
  want wrapper-level periodic heartbeat callbacks
- request a staff-configured QZ workspace, project, compute group, and spec
- pass either an HTTP callback channel or a shared-storage JSONL event channel
  to the worker command; HTTP mode exports `SCALING_INTERNAL_CALLBACK_TOKEN` as
  `QZ_CALLBACK_TOKEN_ENV`, while JSONL mode writes to `QZ_WORKER_EVENT_LOG_DIR`
  and does not export the callback token
- capture the QZ train-job ID
- poll train-job detail and list endpoints
- cancel train jobs on admin request
- collect train-job logs/artifact pointers
- map QZ task failures into backend categories

The adapter should not expose QZ workspace IDs, compute-group IDs, spec IDs,
raw provider errors, or internal API schema details to students.

For deployments where QZ workers cannot see the backend filesystem, the backend
can publish frozen manifests through `SCALING_PUBLISHED_MANIFEST_DIR` plus
`SCALING_PUBLISHED_MANIFEST_BASE_URI`. In that mode, each accepted run writes a
canonical JSON manifest under the local staff-controlled directory, while the QZ
payload receives the worker-readable public URI `<base-uri>/<run-id>.json`.
Course deployment tooling must make the local directory available at that URI.
The bundled worker can fetch `http://` and `https://` manifest URIs directly.
Object-store schemes such as `s3://` and `gs://` require deployment-specific
worker support or an internal authenticated HTTP(S) endpoint that serves the
same manifest objects. Shared-volume rehearsals can instead use
`SCALING_LOCAL_MANIFEST_DIR`, which returns `file://` URIs.

When QZ workers can reach the control node through a WebSocket port-forward URL,
prefer the default HTTP callback channel. QZ deployments can still avoid
worker-to-control-node HTTP callbacks when the control node and compute nodes
share storage. Set `SCALING_WORKER_EVENT_IMPORT_DIR` on the control node and
`QZ_WORKER_EVENT_LOG_DIR` for the QZ worker path. The generated worker command
then receives `--event-sink jsonl` and
`--event-log-path <QZ_WORKER_EVENT_LOG_DIR>/<run-id>.jsonl`. Staff can call
`POST /admin/import-worker-events` to import those JSONL lines through the same
`service.record_worker_event(...)` path used by HTTP callbacks. Worker-generated
events include an `event_id`; replaying the same JSONL file is idempotent.

The generated QZ command is intentionally self-contained. When callback-token
injection and conda activation are configured, it runs as a `bash -lc` wrapper
that sources the conda initialization script, activates `QZ_WORKER_CONDA_ENV`,
and finally `exec`s `python -m scaling_backend.worker.run`. In HTTP callback
mode, the wrapper first exports the internal callback token under
`QZ_CALLBACK_TOKEN_ENV`; the token is not modeled as a separate QZ create-payload
field. In shared JSONL mode, no callback token export is added to the QZ command
because the worker does not call the callback endpoint.

The staff-built worker image must include the trainer callable referenced by
`QZ_WORKER_TRAINER`, for example `course_trainer.worker:train`. The backend
worker imports that callable dynamically and calls it as
`train(manifest, emit_event)`. The trainer must run exactly the frozen manifest
without hidden config mutation, use the frozen `validation_config`, consume the
same tokenized indexes preflighted by the worker when `tokenized_index_uri`
fields are present, emit optional validation events through `emit_event`, and
return finite non-empty `validation_losses`, finite
`final_validation_loss` equal to the last validation loss, and positive
`actual_runtime_seconds`. If the trainer
returns malformed or non-finite results, the worker emits `worker_failed` with
`failure_type = "WorkerResultError"` instead of sending an invalid completed
callback. If tokenized-index preflight fails, the worker emits
`failure_type = "TokenizedIndexError"` and the backend normalizes that failure
as `infrastructure_error`. If the trainer adapter rejects an unsupported frozen
configuration before training, it raises `CourseTrainerConfigError`; the backend
normalizes that failure as `invalid_runtime_config`. If the local runtime watchdog fires, the worker emits
`failure_type = "TimeoutError"` and the backend normalizes that failure as
`timeout`.

The initial deployment should use explicit staff-configured resource presets
rather than dynamic adapter-side resource selection. Dynamic placement is a
scheduler concern and can be added later once the backend has named resource
classes and fairness rules.

Before using QZ for a cohort, staff should run:

```bash
PYTHONPATH=source/backend python -m scaling_backend.preflight \
  --config /secure/course/api-runtime.json \
  --create-dirs \
  --probe-provider \
  --output-json /secure/course/preflight-provider.json
```

The QZ provider probe is non-submitting. If `QZ_COOKIE` is configured, it
validates that cookie by calling the read-only distributed-training job-list
endpoint with `page_size=1`. If `QZ_COOKIE` is absent, it reads
`QZ_COOKIE_FILE` and validates that cookie the same way. If neither cookie
source is available, it falls back to the legacy CAS login check with
`QZ_API_BASE_URL`, `QZ_USERNAME`, and `QZ_PASSWORD_ENCRYPTED`. The probe must not
call the distributed-training submission endpoint or allocate an instance.

When `QZ_SESSION_HEARTBEAT_INTERVAL_SECONDS` is greater than `0`, the API uses
the same read-only QZ provider probe as a session heartbeat. It runs once when
the API starts, then once per configured interval. It keeps the browser session
active without creating or mutating jobs, logs heartbeat success/failure through
the uvicorn console logger, and refreshes the configured `QZ_COOKIE_FILE` from
any `Set-Cookie` header returned by QZ. `0` disables the heartbeat.

## 12. Fake Provider for Tests

The fake provider should support local integration tests without real
accelerators.

It should be able to simulate:

- accepted job
- queued/running/completed status sequence
- validation-loss events
- timeout failure
- numerical failure
- provider rejection
- system failure
- cancellation

The fake provider is required for deterministic backend tests and CI.

## 13. Provider Acceptance Checklist

An adapter is ready for version-one use when:

- it can submit a smoke job from the backend dispatcher
- provider job ID is persisted in experiment status or event records
- status polling works
- cancellation works or fails with a clear staff-visible reason
- logs/artifact pointers are stored
- provider statuses map to backend statuses
- public APIs do not leak provider internals
- same-student and cross-student privacy tests still pass
- failure cases can be simulated or reproduced in a staging environment

## 14. Information Needed From the Course Team

To implement the internal GPU API adapter, the backend team needs:

- task submission endpoint or client library
- authentication method
- required request schema
- accelerator resource request fields
- job status endpoint or callback model
- cancellation endpoint
- log/artifact retrieval method
- timeout behavior
- common provider error payloads
- whether the provider supports environment variables, mounted files, or object
  storage manifests
- limits on job duration, output size, and concurrent jobs

Once these details are supplied, the adapter can be implemented behind the
interface described here without changing the public student API.
