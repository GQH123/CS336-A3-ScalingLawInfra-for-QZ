# Scaling Laws Backend Instructor Operations Guide

Status: Draft for staff rehearsal

Related documents:

- `docs/scaling-assignment-sot.md`
- `docs/provider-adapter-guide.md`
- `docs/scaling-laws-api-quickstart.md`
- `source/backend/README.md`

This guide gives course staff the version-one operational workflow for the
scaling-laws assignment backend. It focuses on the API-first implementation in
`source/backend`, the admin CLI, local rehearsal, final-run launch, exports,
and handling secrets in logs.

The deployment has two packages. The control node installs `source/backend` for
API, quota, provider, callback, and admin logic. The compute-node image installs
the worker, data utilities, local trainer adapter, and Stanford reference
backend:

```bash
python -m pip install -c deploy/compute-node/constraints.txt './refs/cs336-assignment3-scaling[server]'
python -m pip install ./source/backend ./source/data ./source/training
```

The Stanford `[server]` extra installs the CUDA 13 JAX runtime on Linux. Use it
only when the compute-node driver supports CUDA 13. If `nvidia-smi` reports CUDA
12.x, install the CUDA 12 dependency set first, then install `flash-hog` and the
Stanford package without dependencies. The CUDA 12 requirements file keeps
runtime transitive packages such as `chex==0.1.91` explicit because those
`--no-deps` installs will not resolve them:

```bash
python -m pip install -c deploy/compute-node/constraints.txt -r deploy/compute-node/requirements-cuda12.txt
python -m pip install --no-deps flash-hog==0.5.0
python -m pip install --no-deps ./refs/cs336-assignment3-scaling
python -m pip install ./source/backend ./source/data ./source/training
```

`source/training` provides `course_trainer.worker:train`, the callable used by
QZ/Slurm worker commands. It adapts frozen API manifests and mounted
flat-binary tokenized indexes to the local Stanford CS336 training loop.

## 1. Staff-Private Inputs

Before running the backend for students, prepare these staff-private values:

- student API key CSV with header `student_id,api_key`
- internal worker callback bearer token
- staff admin bearer token
- callback URL reachable from worker containers, or shared worker-event storage
  visible from both the control node and worker containers
- manifest base URI or local manifest directory
- selected provider name: `fake`, `slurm`, or `qz_distributed`

Never publish API keys, callback tokens, exact final evaluation manifests, or
provider console URLs in student-facing documents.

Example student-key CSV:

```csv
student_id,api_key
student-1,student-1-secret-api-key
student-2,student-2-secret-api-key
```

To generate this file from a roster CSV, prepare a staff-private roster with a
`student_id` column and run the local seed command:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli \
  seed-student-keys \
  --roster-csv /secure/course/roster.csv \
  --output-csv /secure/course/student_api_keys.csv
```

The `seed-student-keys` command is offline and does not require
`SCALING_ADMIN_BASE_URL` or `SCALING_ADMIN_API_TOKEN`. It writes only
`student_id,api_key`, refuses to overwrite an existing output file unless
`--overwrite` is provided, and prints a summary without revealing API keys.
Store the generated CSV as a secret and point `SCALING_STUDENT_KEYS_CSV` to that
path before starting the backend.

## 2. Backend Environment

Minimum environment for a staff-controlled deployment:

```bash
export SCALING_API_CONFIG="/secure/course/api-runtime.json"
export SCALING_STUDENT_KEYS_CSV="/secure/course/student_api_keys.csv"
export SCALING_INTERNAL_CALLBACK_TOKEN="<worker callback bearer token>"
export SCALING_ADMIN_API_TOKEN="<staff admin bearer token>"
export SCALING_CALLBACK_URL="http://<websocket-forward-host>:8000/internal/provider-events"
export SCALING_MANIFEST_BASE_URI="file:///secure/course/frozen-manifests"
export SCALING_LOCAL_MANIFEST_DIR="/secure/course/frozen-manifests"
export SCALING_PROVIDER="qz_distributed"
```

Recommended additional settings:

```bash
export SCALING_TOTAL_BUDGET_SECONDS="43200"
export SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT="1"
export SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL="8"
export SCALING_CODE_VERSION="course-backend-v0"
export SCALING_DATA_MANIFEST_ID="exploratory-train-v0"
export SCALING_EVAL_MANIFEST_ID="exploratory-eval-v0"
export SCALING_FINAL_DATA_MANIFEST_ID="final-train-v0"
export SCALING_FINAL_EVAL_MANIFEST_ID="final-eval-v0-hidden"
export SCALING_VALIDATION_TOKENS_PER_EVAL="262144"
export SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL="524288"
export SCALING_TOKENIZED_TRAIN_INDEX_URI="file:///mnt/course-data/tokenized/train/index.json"
export SCALING_TOKENIZED_VALIDATION_INDEX_URI="file:///mnt/course-data/tokenized/validation/index.json"
export SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI="file:///mnt/course-data/tokenized/train/index.json"
export SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI="file:///mnt/course-data/tokenized/final-validation/index.json"
export SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE="metadata"
export SCALING_STATE_SNAPSHOT_PATH="/secure/course/state-snapshot.json"
```

`SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT` and
`SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL` bound provider-dispatched exploratory
jobs. Accepted submissions beyond those limits stay queued, reserve budget, and
are launched by `poll-active` or terminal worker/provider events after capacity
frees.

Set `SCALING_PROVIDER_POLL_INTERVAL_SECONDS` to a positive value, for example
`30`, on the control node to enable the built-in active-job poll loop. This loop
uses the same provider status path as the admin `poll-active` command and saves
the configured state snapshot after each poll. Keep the value at `0` only for
local testing or when an external scheduler is responsible for calling
`poll-active`.

The validation-token settings are staff-private evaluation sample sizes. They
must be divisible by each submitted config's
`training.sequence_length * training.validation_batch_size`; otherwise the
backend rejects the config before provider dispatch instead of silently rounding
validation batches.
If a worker runs on multiple visible GPUs, set `SCALING_WORKER_GPU_COUNT` to that
count. The backend will then reject submitted `training.train_batch_size` and
`training.validation_batch_size` values that are not divisible by the worker GPU
count, matching the Stanford trainer's FSDP batch-axis sharding requirement.
It also rejects model parameter axes that the Stanford FSDP mesh would partition
unevenly: `model.hidden_size` and `model.intermediate_size` must be divisible by
the worker GPU count, and untied-output-head configs
(`model.tie_word_embeddings = false`) must also use a `model.vocab_size`
divisible by the worker GPU count. This catches JAX sharding-divisibility
failures before provider dispatch and gives students an actionable config
error instead of an opaque worker stack trace.
The `SCALING_TOKENIZED_*_INDEX_URI` settings are optional worker-readable
private data artifact pointers. When configured, they are written only into
frozen provider manifests and are used by the worker tokenized-index preflight;
they are not exposed through student result APIs. These paths are interpreted in
the compute-node worker image; the control-node API treats them as opaque
worker-visible URIs and does not require the data mount locally.
The production worker mode should be
`SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE=metadata` so mounted 100B-scale
data is checked by index schema, safe relative paths, token counts, local file
sizes, and SHA-256 metadata format without reading every shard. Use `full` only
for offline audits or small rehearsals.

For a shared-volume deployment, also set:

```bash
export SCALING_LOCAL_MANIFEST_DIR="/secure/course/frozen-manifests"
```

If the HTTP callback URL is unavailable and you need the shared-storage
worker-event fallback, also set both:

```bash
export SCALING_WORKER_EVENT_IMPORT_DIR="/secure/course/worker-events"
export QZ_WORKER_EVENT_LOG_DIR="/secure/course/worker-events"
```

For HTTP-published manifests, set both:

```bash
export SCALING_PUBLISHED_MANIFEST_DIR="/secure/course/frozen-manifests"
export SCALING_PUBLISHED_MANIFEST_BASE_URI="https://manifests.example.edu/scaling"
```

Run the deployment preflight before starting the backend:

```bash
PYTHONPATH=source/backend python -m scaling_backend.preflight \
  --config /secure/course/api-runtime.json \
  --create-dirs \
  --output-json /secure/course/preflight.json
```

The preflight does not start FastAPI and does not submit provider jobs. It
checks required runtime variables, callback and manifest URI schemes, student-key
CSV readability, distinct non-placeholder staff secrets, manifest storage,
snapshot parent directories, budget settings, manifest ID separation,
tokenized-index URI schemes, worker validation mode, shared worker-event storage,
and provider-specific environment variables. It resolves `SCALING_API_CONFIG`
before checking, so staff can validate the same JSON file used by the
control-node API container.
It does not fetch manifests or read large tokenized shards. Worker startup should
use `SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE=metadata` for production
mounted reservoirs and reserve `full` mode for offline audits. The command exits
non-zero if any check fails.

For a live infrastructure readiness check, add the explicit provider probe:

```bash
PYTHONPATH=source/backend python -m scaling_backend.preflight \
  --create-dirs \
  --probe-provider \
  --output-json /secure/course/preflight-provider.json
```

The provider probe is intentionally non-submitting. For `qz_distributed`, it
validates `QZ_COOKIE` first by calling the read-only distributed-training
job-list endpoint with `page_size=1`; if no cookie is configured, it falls back
to CAS login with `QZ_USERNAME`, `QZ_PASSWORD_ENCRYPTED`, and `QZ_API_BASE_URL`.
It does not create a distributed-training task. For `slurm`, it runs only
`sinfo --version` and `sbatch --version`; it does not call `sbatch` with a job
script.

Start the backend:

```bash
PYTHONPATH=source/backend uvicorn scaling_backend.asgi:app \
  --host 0.0.0.0 \
  --port 8000
```

## 3. Admin CLI Setup

The admin CLI talks to the running FastAPI admin routes and uses the same
runtime state and audit log as the backend.

```bash
export SCALING_ADMIN_BASE_URL="https://control.example.edu"
export SCALING_ADMIN_API_TOKEN="<staff admin bearer token>"
```

Run:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli queue
```

The command should return current exploratory and final-run queues. A 401
response means the admin token does not match `SCALING_ADMIN_API_TOKEN`.
Exploratory rows include `queue_position` for FIFO order and `fair_share_rank`
for the round-robin staff scheduling view across students. The separate
`fair_share_order` list shows the same fair-share ordering directly. Final-run
rows have their own `queue_position` because final evaluation is launched from
the frozen final-submission batch and remains separate from exploratory sharing.

## 4. Course-Day Operations

Inspect active queues:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli queue
PYTHONPATH=source/backend python -m scaling_backend.admin_cli poll-active
```

Use `queue_position` to inspect raw FIFO order and `fair_share_rank` /
`fair_share_order` to check whether one student's repeated exploratory runs are
being interleaved fairly with other students' active runs. `poll-active` updates
the provider state for currently active exploratory and final-run items. If an
item becomes terminal during polling, the response includes `provider_artifacts`
with provider log/detail pointers. With QZ, the status read also inspects worker
logs and can close a stale `RUNNING` detail when logs show a nonzero process
exit.

If shared-storage worker-event fallback is configured, also ingest worker JSONL:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli import-worker-events
```

Inspect experiments:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli experiments
PYTHONPATH=source/backend python -m scaling_backend.admin_cli experiments \
  --student-id student-1
PYTHONPATH=source/backend python -m scaling_backend.admin_cli experiment exp-000001
```

Inspect a student's budget:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli \
  student-budget student-1
```

Cancel an exploratory run:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli \
  cancel exp-000001 \
  --reason "student requested cancellation" \
  --actor staff
```

Cancellation is intended for active provider jobs. If an exploratory run has
already reached a terminal status, the cancel command returns the current run
record without dispatching a provider cancellation or changing the recorded
result.

Mark an instructor-reviewed system failure:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli \
  mark-system-failure exp-000001 \
  --failure-reason provider_outage \
  --staff-failure-detail "QZ outage incident INC-42" \
  --refund-seconds 120 \
  --reason "confirmed provider outage" \
  --actor staff
```

Apply a manual budget adjustment:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli \
  adjust-budget student-1 \
  --seconds -120 \
  --reason "provider outage refund" \
  --actor staff \
  --experiment-id exp-000001
```

All override commands require `--actor` and `--reason`; these are written to
the admin audit export.

## 5. Final-Run Workflow

After the student deadline, freeze the latest valid final submission for each
student:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli \
  freeze-finals \
  --reason deadline \
  --actor staff
```

This command closes the final-submission window even if it freezes zero
submissions. Late student submissions are rejected after this point. Invalid
final training configs are rejected when students submit them, so the freeze
step only sees stored submissions that passed API validation.

Launch the hidden final-run batch from the frozen submissions:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli \
  launch-finals \
  --max-runtime-seconds 7200 \
  --reason deadline \
  --actor staff
```

Poll active final jobs:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli poll-active
PYTHONPATH=source/backend python -m scaling_backend.admin_cli \
  poll-final final-run-000001
```

Cancel or mark a final run only for staff-reviewed cases:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli \
  cancel-final final-run-000001 \
  --reason "staff dry-run cleanup" \
  --actor staff

PYTHONPATH=source/backend python -m scaling_backend.admin_cli \
  mark-final-system-failure final-run-000001 \
  --failure-reason infrastructure_error \
  --staff-failure-detail "hidden eval outage INC-55" \
  --reason "confirmed hidden eval outage" \
  --actor staff
```

## 6. Export and Grading Bundle

Print the raw export JSON:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli exports
```

Write the standard JSONL/CSV bundle:

```bash
PYTHONPATH=source/backend python -m scaling_backend.admin_cli exports \
  --output-dir "/secure/course/exports/$(date +%F)"
```

Expected files:

- `experiments.jsonl`
- `experiments.csv`
- `experiment_events.jsonl`
- `worker_events.jsonl`
- `budget_snapshots.jsonl`
- `budget_snapshots.csv`
- `budget_adjustments.jsonl`
- `admin_actions.jsonl`
- `final_submissions.jsonl`
- `final_submissions.csv`
- `final_runs.jsonl`
- `final_runs.csv`
- `grading.csv`

Use `grading.csv` for the final leaderboard/grading pass. It includes hidden
final-run loss fields, prediction interval fields, and prediction scoring
columns: `prediction_absolute_error`, `prediction_interval_covered`,
`prediction_interval_width`, `prediction_interval_miss_distance`,
`prediction_interval_score`, and the lower-is-better
`prediction_quality_penalty`. The interval score follows the student handout
rule for an 80% central prediction interval:
`width + 10 * miss_distance`; the combined penalty is
`0.5 * absolute_error + 0.5 * interval_score`. The export also includes
methodology/writeup columns that staff can fill or join from the
report-submission workflow:
`methodology_report_score`, `methodology_report_notes`, `analysis_code_score`,
`analysis_code_notes`, and `writeup_artifact_uri`. It also includes
`final_run_grading_outcome` and `final_run_score_policy`: student-caused final
failures are marked for worst-score treatment, while system failures are marked
for staff review before scoring. Keep the JSONL files for audit and post-course
analysis. `worker_events.jsonl` is the unified exploratory plus final-run worker
callback ledger; `experiment_events.jsonl` is retained as an exploratory-only
compatibility export. `budget_snapshots.csv` is the point-in-time per-student
quota view at export time; `budget_adjustments.jsonl` remains the audited ledger
of manual refunds, charges, and penalties. Staff experiment detail and export
records include provider artifact pointers: JSONL records preserve the nested
`provider_artifacts` object, while `experiments.csv` and `final_runs.csv` expose
`provider_logs_uri` and `provider_detail_uri` for fast incident review.
Worker-reported infrastructure failures, including tokenized-index artifact
failures, are exported as `system_failed` with staff-only detail. Exploratory
worker system failures release the active reservation without charging the
student; hidden final-run worker system failures are marked for staff review in
`grading.csv`.

## 7. Local Staff Rehearsal

Before opening the assignment to students, run the deterministic local
rehearsal:

```bash
rm -rf /tmp/scaling-rehearsal
PYTHONPATH=source/backend python -m scaling_backend.rehearsal \
  --output-dir /tmp/scaling-rehearsal
```

The rehearsal uses the fake provider and does not call live QZ or Slurm. It:

- seeds two deterministic students in code
- submits two exploratory runs
- executes worker callbacks through the backend service
- freezes final submissions
- launches two hidden final runs
- writes the full export bundle
- writes an operations log
- scans rehearsal outputs for the local callback secret

Inspect:

```bash
python -m json.tool /tmp/scaling-rehearsal/rehearsal-report.json
cat /tmp/scaling-rehearsal/operations.log
ls -1 /tmp/scaling-rehearsal/exports
```

The report should contain:

```json
{
  "acceptance_checks": {
    "seeded_students": true,
    "queue_inspected": true,
    "final_submissions_frozen": true,
    "final_batch_launched": true,
    "grading_export_written": true,
    "operation_log_written": true,
    "secret_scan_passed": true
  }
}
```

If any acceptance check is false, fix the local workflow before using the
backend for a student cohort.

## 8. Release Package Audit

Before distributing the handout, quickstart, and notebook, run the static
release audit:

```bash
PYTHONPATH=source/backend python -m scaling_backend.release_audit \
  --output-json /tmp/scaling-release-audit.json
```

The release audit checks that:

- student quickstarts exist and use the implemented bearer-auth API contract
- the student notebook is valid, unexecuted, and covers submit/poll/analyze/final submission
- staff operations docs and provider adapter guide are present
- preflight, SOT milestone status, and local rehearsal tools are present with regression tests
- release-facing docs do not contain stale API fields from earlier drafts
- LaTeX sources use the configured `minted`/`monokai` code-block style

If `latexmk` or `xelatex` are unavailable, the audit reports a warning rather
than a failure. Run native PDF compilation in a TeX environment before final
publication.

## 9. Launch Readiness Gate

Before opening the backend to students, run the combined readiness gate:

```bash
PYTHONPATH=source/backend python -m scaling_backend.readiness \
  --create-dirs \
  --rehearsal-output-dir /tmp/scaling-readiness-rehearsal \
  --output-json /secure/course/readiness.json
```

This command runs deployment preflight, SOT milestone evidence audit, release
audit, and deterministic local staff rehearsal, then returns one JSON report with
top-level `preflight`, `sot_status`, `release_audit`, and `rehearsal` sections.
It exits non-zero if preflight fails, any SOT milestone is missing implementation
evidence, release audit has failed checks, or any rehearsal acceptance check is
false. Warnings, such as missing local TeX PDF tools, remain visible in the
report.

In the target provider environment, add `--probe-provider` to include the
explicit non-submitting QZ/Slurm connectivity probe.

## 10. Log Retention and Secret Handling

Retain these staff-private artifacts at least until grading and dispute review
are complete:

- backend service logs
- provider job IDs and safe provider metadata
- frozen manifests
- state snapshots or database backups
- export bundles
- operations logs from rehearsals and final-run launch

Do not put these values in public docs, notebooks, screenshots, or student
responses:

- student API keys
- admin bearer token
- internal callback token
- exact hidden final-eval manifest
- provider credentials or cookies
- raw provider console URLs unless explicitly sanitized

Public student APIs should expose only normalized statuses, public failure
categories, validation losses for their own runs, and documented budget fields.

## 11. Readiness Checklist

Before launch:

- `python -m scaling_backend.readiness --create-dirs` passes
- `python -m scaling_backend.sot_status` reports all SOT milestones as present
- student key CSV loaded successfully
- `python -m scaling_backend.preflight --create-dirs` passes
- `python -m scaling_backend.preflight --create-dirs --probe-provider` passes in
  the target provider environment
- admin CLI can call `queue`
- fake-provider local rehearsal passes all acceptance checks
- `python -m scaling_backend.release_audit` has no failed checks
- provider adapter guide reviewed by infrastructure owner
- frozen-manifest directory or manifest publisher is configured
- callback token verified from a worker-like environment
- exploratory train/eval manifest IDs are not the hidden final-eval manifest

Before deadline:

- export bundle from exploratory runs is readable
- staff know who can run `freeze-finals`
- students have been reminded not to submit API keys in notebooks

At deadline:

- run `freeze-finals`
- run `launch-finals`
- monitor with the background provider poll loop or manual `poll-active`
- record any staff interventions with reason and actor

After final runs:

- write export bundle with `exports --output-dir`
- archive frozen manifests, state snapshot/database backup, and export bundle
- review `grading.csv` and JSONL audit files before leaderboard publication
