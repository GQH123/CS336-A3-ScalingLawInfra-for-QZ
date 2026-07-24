# Scaling-Law Assignment Backend SoT and Implementation Plan

Date: 2026-07-16

Status: Draft for review

Related documents:

- Research brief: `docs/scaling-assignment-backend-research.md`
- Provider adapter guide: `docs/provider-adapter-guide.md`

This document is the single Source of Truth for the first version of the
summer-course scaling-law assignment backend. It combines product scope,
assignment policy, backend architecture, operational rules, and implementation
milestones.

## 1. Maintenance Policy

- Milestone summaries are written after each completed decision branch.
- This file, `docs/scaling-assignment-sot.md`, is the canonical document for
  binding decisions and implementation planning.
- The research brief is background context only. A decision is binding only if
  it appears in this SoT.
- The provider adapter guide is a focused companion document for implementing
  the internal GPU API and Slurm adapters against the backend's provider
  abstraction.

## 2. Version-One Goal

Version one is a Stanford-style fixed-compute scaling-law assignment backed by
real official training runs. Students submit training configurations to a
course-controlled API, receive validation losses from official runs, fit scaling
laws, and submit one final configuration with a predicted final validation loss
and uncertainty interval.

The backend must do five things well:

1. Accept valid student training configurations.
2. Enforce individual student budgets.
3. Run official training jobs exactly according to accepted configurations.
4. Return precise observed training results and validation losses.
5. Support a controlled final hidden evaluation workflow.

The first version optimizes for a working, fair, auditable course platform, not
for a broad research platform.

## 3. Product Scope

### Included in Version One

- One fixed-compute scaling-law assignment track.
- Individual student budgets and individual final submissions.
- API/notebook-first student workflow.
- Real hosted training runs through staff-controlled accelerator infrastructure.
- Staff-owned data preparation pipeline under `source/data/`, covering
  download manifests, blending, post-processing, held-out splitting, and
  tokenization for the training reservoir.
- CS336-compatible public API shape with small schema extensions.
- JAX/Equinox dense Transformer training runtime adapted from the local
  scaffold.
- Provider abstraction replacing Modal-specific dispatch.
- Internal GPU submission API as the first real provider target.
- Slurm adapter support for testing and alternate execution.
- Simple instructor/admin scripts and views.
- Full course export for grading, auditing, and later analysis.

### Not Included in Version One

- Multiple assignment tracks.
- Student-facing dashboard as a required feature.
- MoE or broad architecture search.
- RL/post-training or inference-time compute tracks.
- Variable tokenizer/vocabulary track.
- Student-supplied training code.
- Checkpoint/resume.
- Required W&B or external experiment tracking.
- Course SSO.

## 4. Student Assignment Policy

### Budget

Each student receives:

- 12 official accelerator-hours for exploratory scaling-law runs.
- 48 official accelerator-hours for the final hidden run.

The public assignment should not disclose the exact accelerator/GPU model before
the hardware decision is finalized. The guarantee is that official exploratory
and final runs use a consistent staff-controlled hardware type and training
stack.

The budget should be expressed as official accelerator-hours. If the actual
hardware changes before launch, the course team can preserve the same conceptual
budget while updating internal throughput expectations.

### Official Runs

Only hosted API runs count as official assignment runs. Students may inspect
reference code where provided, but extra local/private-GPU runs are not official
and should be disallowed by course policy if fairness is a concern.

The backend enforces only official API budgets. Course policy handles whether
students may run private experiments outside the official system.

### Final Submission

Each student submits:

- final training configuration
- predicted final validation loss
- lower and upper bounds for a prediction interval

Final submissions are mutable until the deadline. At the deadline, the latest
valid final submission for each student is frozen and used for the final-run
batch.
Validity includes both prediction-interval checks and training-config checks
against the same accepted-config rules used for experiments and hidden final
runs. Invalid final training configs are rejected at submission time and are not
eligible for freezing.

### Scoring

Default grading weights:

- 60% final pretraining-run validation loss
- 20% prediction quality, scored by the lower-is-better
  `prediction_quality_penalty`
- 20% methodology/writeup

For prediction scoring, let the actual final validation loss be `L`, the point
prediction be `L_hat`, and the submitted interval be `[L_lower, L_upper]`.
Valid predictions must use finite values and satisfy
`L_lower <= L_hat <= L_upper`. The interval is treated as an 80% central
prediction interval:

```text
absolute_error = abs(L_hat - L)
miss_distance = max(0, L_lower - L) + max(0, L - L_upper)
interval_score = (L_upper - L_lower) + 10 * miss_distance
prediction_quality_penalty = 0.5 * absolute_error + 0.5 * interval_score
```

This makes prediction quality a tradeoff: wide intervals pay a width penalty,
while intervals that miss the actual loss pay a larger miss-distance penalty.

Student-caused failed final runs receive the worst final-run validation-loss
score for the leaderboard component. Staff-adjudicated system failures are
excluded from automatic worst-score penalty until reviewed.

## 5. Data and Evaluation Policy

### Training Data

The course may use public-source data, but the exact public datasets, mixture,
manifest, filtering/preprocessing, tokenized shard order, and final evaluation
split remain private.

This hybrid policy avoids building a proprietary corpus from scratch while
preventing students from reconstructing the exact official experiment loop.

The version-one corpus should be a mixed-domain pretraining reservoir rather
than a single-source clone of the Stanford setup. The goal is to better
approximate real industry pretraining practice while reducing the chance that
students can exploit exact knowledge of a known reference dataset.

Recommended internal corpus direction:

- use a high-quality subset of DCLM-style data as one major component
- do not use the full DCLM volume as the entire corpus
- mix in educational/general web, long-form factual text, math/science, code,
  and a small multilingual slice
- keep the exact sources, weights, filtering rules, manifest, and tokenized
  order private
- target a version-one training reservoir of about 100B tokens

Candidate 100B-token internal mix:

- 55B tokens high-quality general web
- 20B tokens educational web
- 10B tokens diverse long-form text
- 7B tokens math/science
- 6B tokens code
- 2B tokens multilingual high-quality text

The repository includes a reusable data-pipeline scaffold under `source/data/`.
The public code can describe how data is downloaded, blended, cleaned, split, and
tokenized. The exact staff-approved production manifest should remain private.
- prefer quality-filtered splits when they exist
- if a specific quality-filtered split is unavailable, use reasonable public
  filtering/deduplication and do not block the project on perfect filtering
- keep the mixture fixed across official runs so scaling-law curves reflect
  model/data/compute effects rather than changing data distributions

### Determinism

Official runs should be deterministic:

- fixed tokenized data order
- explicit model seed
- deterministic evaluation manifests
- fixed sequence length
- fixed validation loss definition

The model seed controls model initialization. It should not change the training
data order.

### Exploratory Validation

Exploratory runs expose validation losses from an official exploratory
validation evaluation. Students can query these losses through the API as runs
progress and complete.

### Final Validation

Final leaderboard scoring uses a larger held-out validation evaluation. This
does not change the training data distribution. It only changes the evaluation
sample used for final scoring so leaderboard rankings are less sensitive to
public validation noise or repeated querying.

## 6. Public API Contract

The API should preserve the CS336-style endpoint shape and add only necessary
extensions.

Required public endpoints:

- `GET /budget`
- `POST /submit`
- `GET /experiments`
- `GET /experiment/{experiment_id}`
- `POST /final_submission`
- `GET /final_submission`

Recommended instructor/admin endpoints or scripts:

- list experiments across students
- inspect queue state
- inspect budget usage
- freeze final submissions
- launch final-run batch
- cancel experiment
- apply audited budget adjustment
- export course records

### Authentication and Privacy

- Authentication uses per-student API keys.
- Every public query is scoped to the authenticated student.
- Identical configs submitted by different students are separate experiments.
- Duplicate detection is keyed by `(student_id, config_hash)`.
- A duplicate submission may return the existing experiment ID only when the
  existing experiment belongs to the same authenticated student.

### Duplicate Config Behavior

When a student submits an identical config twice:

- return HTTP 409
- include the existing same-student experiment ID in the structured error body
- do not reserve additional budget
- do not create a new experiment

This makes duplicate requests recoverable without hiding accidental resubmits.

### Final Submission Schema Extension

The final submission request should include:

- `training_config`
- `predicted_final_loss`
- `predicted_final_loss_lower`
- `predicted_final_loss_upper`

Validation rules:

- all predicted loss values must be finite
- lower bound must be less than or equal to point estimate
- point estimate must be less than or equal to upper bound
- `training_config` must satisfy the accepted training-config rules for the
  hidden final run
- interval scoring details are handled by grading, not by training workers

## 7. Experiment Status and Failure Semantics

### Public Status Types

Recommended public status values:

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`
- `system_failed`

For `failed`, expose a typed failure reason:

- `timeout`
- `numerical`
- `resource`
- `invalid_runtime_config`
- `provider_rejected`
- `unknown_student_caused`

For `system_failed`, expose a typed system reason:

- `provider_outage`
- `worker_lost`
- `infrastructure_error`
- `admin_intervention`
- `unknown_system`

Raw provider errors should be retained in staff logs but not exposed verbatim in
the public API.

### Completed Runs

A completed exploratory run has:

- `status = completed`
- non-empty `validation_losses`
- `final_validation_loss = validation_losses[-1]`
- `used_runtime_seconds`
- completed timestamp

Only completed runs produce a valid final validation loss.

### Failed Exploratory Runs

Failed exploratory runs may expose partial validation losses as diagnostics.
They do not produce a final validation loss and should not be treated as
completed results in student analysis or grading exports.

### Timeout

Timeouts are student-caused if the accepted config exceeds its reserved runtime
or does not complete by the configured deadline. Timeout failures are charged as
the full reserved `max_runtime_seconds`.

### Numerical Failure

Numerical failures include NaN/Inf loss, repeated non-finite gradients, or other
detected instability. These are student-caused and charged by actual runtime,
clipped according to the budget policy.

### Resource Failure

Version one uses hard schema validation and soft resource warnings. Because
memory/runtime estimates are not treated as exact, an accepted config may still
fail at runtime due to resource pressure. Such failures are generally
student-caused unless staff determine there was a platform bug.

### System Failure

Clear infrastructure failures are rare exceptions. They should be reviewed by
staff and handled through audited refund or status adjustment. Version one does
not auto-retry failed jobs because ambiguous retries can distort the assignment
timeline and operational load.

## 8. Budget Accounting

Budget accounting follows the CS336-style reservation model.

Queued and running experiments reserve their full `max_runtime_seconds`.

Completed and failed experiments are charged by `used_runtime_seconds`, clipped
to policy bounds, except timeouts, which are charged as the full reserved
runtime.

Recommended budget rules:

- rejected schema/config requests: charge 0
- duplicate same-student config: charge 0
- queued/running: reserve `max_runtime_seconds`
- completed: charge actual runtime clipped to `[1, max_runtime_seconds]`
- timeout: charge `max_runtime_seconds`
- numerical/resource/student-caused failure: charge actual runtime clipped to
  `[1, max_runtime_seconds]`
- clear system failure: instructor-reviewed refund or budget adjustment
- admin cancellation: policy-specific audited adjustment

Budget updates must be auditable. Manual adjustments should be represented as
separate budget-adjustment records rather than silent edits to experiment rows.

## 9. Configuration and Resolved Metadata

### Accepted Training Config

The baseline student-configurable space follows the CS336 dense Transformer
configuration:

- architecture config
  - attention bias
  - head dimension (`head_dim`; derived only when omitted)
  - hidden size
  - intermediate/FFN size
  - number of attention heads
  - number of hidden layers
  - number of key/value heads, equal to attention heads in version one because
    grouped-query attention is not supported
  - normalization epsilon
  - RoPE theta
  - tied embeddings
  - dtype
  - vocabulary size if the runtime permits it
- optimizer config
  - AdamW or SGD
  - LR schedule
  - warmup fraction
  - final LR fraction
  - weight decay
  - betas/epsilon where applicable
  - gradient clip norm where applicable
- training config
  - train batch size
  - validation batch size
  - number of evals
  - total train tokens
  - model seed

Runtime reservation is a submission/request field, not part of the student
training-config object. The public API accepts `requested_runtime_seconds` on
`POST /submit`; resolved worker manifests carry the canonical
`max_runtime_seconds` runtime cap.

The schema should remain extensible, but version one should avoid broad
architecture search beyond the scaffold's dense Transformer family.

### Validation

Hard validation:

- JSON schema and field type validity
- positive required integer fields
- architecture shape consistency
- divisibility constraints
- optimizer parameter bounds
- finite numeric values
- budget cap

Soft warnings:

- estimated memory pressure
- estimated runtime pressure
- unusually high learning rate
- very low number of optimizer steps
- suspicious evaluation cadence

Soft warnings should be returned as metadata and logged, but should not reject
the request in version one.

### Resolved Config

Every accepted experiment should persist a resolved config snapshot:

- config hash
- parameter count
- non-embedding parameter estimate
- estimated FLOPs or compute proxy
- tokens per optimizer step
- total optimizer steps
- eval interval
- validation tokens per evaluation
- validation batches per evaluation
- resource warnings
- code version
- data manifest ID
- evaluation manifest ID
- provider manifest ID when launched

This snapshot is critical for reproducibility and grading exports.

## 10. Provider Abstraction

The local scaffold's Modal-specific dispatch should be replaced by a thin
provider abstraction.

The abstraction exists to isolate course logic from execution infrastructure.
It is not intended to support every possible provider feature.

Required operations:

- submit a frozen job manifest
- inspect provider job status
- cancel a provider job
- map provider status/errors to backend status categories
- provide log/artifact pointers

Initial concrete targets:

- QZ distributed-training API: deployment priority, using staff-built Docker
  images that contain the backend training worker
- Slurm adapter: testing and alternate execution
- fake/local adapter: unit and integration tests

The detailed provider contract lives in `docs/provider-adapter-guide.md`.

The QZ adapter should be implemented natively in this backend. The temporary
`refs/qzcli_tool/` project is a reference for CAS authentication, request
headers, payload shape, and operational behavior, but it is not a runtime package
dependency. Version one should submit distributed training jobs from explicit
staff-configured QZ resource presets. The adapter translates a frozen backend
manifest into a QZ job; it does not decide resource placement, student policy, or
budget effects.

## 11. Worker Lifecycle

A worker receives a frozen manifest and runs exactly that manifest.

Worker steps:

1. Load manifest.
2. Configure logging.
3. Load fixed data manifest and evaluation manifest.
4. If the manifest carries worker-readable tokenized train/eval index URIs,
   preflight the audited `index.json` files before trainer execution. Production
   startup uses metadata validation to reject unreadable, unsafe, size-mismatched,
   or internally inconsistent shard metadata without reading the full reservoir;
   full hash validation remains an offline audit mode.
5. Initialize model from submitted config and model seed.
6. Run training chunks under the canonical `max_runtime_seconds` watchdog where
   the worker runtime can enforce one.
7. Emit validation-loss events after each configured evaluation.
8. Emit heartbeats during long execution.
9. Finish with completed or failed payload.

The worker must not silently alter student configs. No hidden LR reduction, no
batch-size fallback, no dtype change, no model-size adjustment, and no automatic
checkpoint resume.
Tokenized-index failures are system/data-artifact failures, not student-caused
hyperparameter failures; the backend reports them under the stable
`infrastructure_error` category while retaining exact shard details only in
staff logs.
Watchdog timeouts are student-visible `timeout` failures and should be backed by
both worker-level reporting and provider-level wall-clock enforcement.

## 12. Scheduler

The exploratory scheduler should preserve fair sharing among students.

Recommended policy:

- global maximum concurrent workers
- per-student active-run fairness
- FIFO ordering within fairness rank
- queued/running runs reserve budget
- final-run queue separate from exploratory queue

Final runs should be launched only from frozen final submissions after the
deadline.

## 13. Logging and Export

Logs must be comprehensive enough to reconstruct the course.

Persist:

- submitted configs
- resolved configs
- validation-loss sequences
- train/runtime timestamps
- budget snapshots or derivable budget state
- provider job IDs and provider statuses
- provider log/detail artifact pointers for staff debugging
- worker heartbeats
- failure categories
- staff-only raw failure details
- final submissions
- final-run results
- admin overrides and budget adjustments

Required exports:

- per-student experiment summary CSV
- full experiment/event JSONL
- unified worker-event JSONL covering exploratory and final-run callbacks
- provider log/detail artifact pointers in staff JSONL and CSV summaries
- per-student budget snapshot CSV/JSONL with total, reserved, charged, and
  remaining seconds
- final submissions CSV/JSONL
- final-run results CSV/JSONL
- grading export with actual loss, predicted loss, interval,
  `prediction_quality_penalty`, `prediction_interval_score`,
  `final_run_grading_outcome`, `final_run_score_policy`, and writeup fields

## 14. Admin Workflow

Version one uses scripts and simple views.

Required commands or views:

- seed API keys from CSV
- list experiments by status/student
- inspect queue
- inspect student budget
- inspect experiment detail
- cancel experiment
- record audited refund/adjustment
- freeze final submissions
- launch final-run batch
- export course records

Every admin override must record:

- actor
- timestamp
- action type
- target student/experiment/final run
- reason
- before/after values where applicable

## 15. Security and Privacy

Student-facing APIs must not reveal:

- other students' experiment IDs
- other students' configs
- exact private data manifest
- exact private source dataset mix
- final evaluation split
- raw provider errors with infrastructure details
- API keys or internal callback credentials

Provider manifests and worker logs must treat internal API keys, callback
tokens, and data manifests as secrets.

## 16. Implementation Plan

### M0: Scaffold Audit and Adaptation Map

Goal: identify what to keep, replace, and extend in the local scaffold.

Tasks:

- inspect existing FastAPI routes, schemas, DB tables, scheduler, worker, and
  tests
- map Modal-specific code paths
- map config validation gaps against this SoT
- map budget/failure gaps against this SoT
- map backend data-loading expectations against the `source/data/` tokenized
  reservoir format
- decide package/module names for adapted backend
- identify minimal changes needed for M1 smoke run

Acceptance checks:

- adaptation map exists in this SoT or a linked implementation note
- list of files/modules to modify is known
- M1 smoke path is clearly defined

### M1: Submit-to-Result Smoke Run

Goal: one tiny valid config flows through the whole system.

Tasks:

- run local/dev API
- seed one student API key
- submit a tiny valid training config
- dispatch through fake/local or available provider adapter
- execute a tiny training/evaluation path
- write validation loss event
- mark experiment completed
- update budget accounting
- retrieve result through public API

Acceptance checks:

- `POST /submit` returns experiment ID and budget reservation
- `GET /experiment/{id}` progresses through queued/running/completed
- completed run has non-empty validation losses
- final validation loss equals last validation loss
- budget reflects actual runtime policy
- no cross-student access is possible in smoke tests

### M2: Provider Abstraction and Adapters

Goal: replace Modal-specific dispatch with provider-neutral execution.

Tasks:

- define provider interface in code (`source/backend/scaling_backend/providers/`)
- define frozen manifest schema (`manifest_version = 1` with explicit
  `run_kind`, resolved configs, data/eval manifest IDs, callback endpoint,
  runtime cap, and creation timestamp)
- implement fake/local provider for tests
- implement Slurm adapter if cluster access is available
- implement native QZ distributed-training adapter for the first deployment path
- document QZ adapter requirements in provider guide
- map provider statuses to backend statuses

Acceptance checks:

- dispatcher depends only on provider interface
- fake/local provider passes integration tests
- frozen exploratory and final manifests are persisted before provider dispatch
  and carry the same resolved worker-facing config seen by the worker
- QZ distributed adapter can build a validated distributed-training payload
- Slurm or QZ adapter can submit a test job when credentials are
  available
- provider errors are categorized without leaking raw internals to students

### M2.5: Data Pipeline and Tokenized Reservoir

Goal: prepare a staff-controlled tokenized corpus pipeline suitable for the
100B-token reservoir.

Tasks:

- finalize staff-private source manifest and licenses
- run the blend-plan dry run
- download source datasets to staff-controlled storage
- clean and exact-deduplicate documents
- hold out validation documents before tokenization
- tokenize with the same tokenizer used by the training backend
- write binary token shards and index metadata
- compute shard hashes and source accounting reports
- publish a versioned tokenized index schema with little-endian `uint32` shards
- validate production worker startup through metadata checks instead of full
  reservoir reads
- run a small sample pipeline before the full build

Acceptance checks:

- `source/data/` scripts produce a deterministic 100B blend plan
- sample download/postprocess/tokenization path completes
- full build stores tokenized train shards and validation shards separately
- public API and student docs do not reveal the exact private manifest
- `source/training` provides `course_trainer.worker:train`, which adapts the
  produced tokenized index to the Stanford CS336 training backend
- manually assembled control-node API and compute-node worker deployments use
  separate configs: `SCALING_API_CONFIG` for submission/orchestration and
  `SCALING_WORKER_CONFIG` or worker image defaults for training/data runtime

### M3: Policy and API Hardening

Goal: make public behavior match this SoT.

Tasks:

- extend final submission schema with prediction interval
- add same-student duplicate response with existing experiment ID
- implement typed failure reasons
- implement hard schema validation and soft resource warnings
- implement per-student privacy tests
- implement budget adjustment records
- implement structured error bodies
- ensure public API remains CS336-compatible where intended

Acceptance checks:

- duplicate same-student config returns 409 with same-student existing ID
- duplicate cross-student config does not reveal any other student information
- failed runs expose partial diagnostics but no final loss
- system failures can be marked/refunded through audited admin path
- final submission interval validation works

### M4: Final-Run and Admin Workflow

Goal: support the full course deadline and final evaluation process.

Tasks:

- implement final-submission freeze
- implement final-run table or equivalent status tracking
- implement separate final-run queue/batch
- implement larger held-out final evaluation manifest
- implement final-run export
- implement admin scripts/simple views
- implement cancel/refund/override audit records

Acceptance checks:

- latest valid final submission freezes at deadline
- post-freeze submissions cannot alter frozen final-run input
- final-run batch launches from frozen set
- final-run results export includes actual loss, predicted loss, interval, and
  status
- admin overrides are auditable

### M5: Documentation, Deployment Rehearsal, and Course Readiness

Goal: ensure staff and students can use the system during the one-week
assignment.

Tasks:

- write student API quickstart
- write instructor operations guide
- write provider adapter guide
- write failure/status policy section for assignment handout
- prepare example notebook
- run rehearsal with multiple seeded students
- run export/grading rehearsal
- check log retention and secret handling

Acceptance checks:

- student can submit/poll/analyze via example notebook
- instructor can seed users, inspect queue, freeze submissions, launch final
  batch, and export grading records
- provider guide is sufficient for internal API implementation handoff
- deployment rehearsal produces expected logs and exports

## 17. Risks and Mitigations

### Provider Integration Risk

Risk: internal GPU API schema is cumbersome or changes during implementation.

Mitigation: keep provider interface thin, provide a fake/local adapter for tests,
and isolate provider-specific code behind one module boundary.

### Ambiguous Failure Classification

Risk: some failures are hard to classify as student-caused or infrastructure
caused.

Mitigation: default ambiguous runtime failures to typed student-caused categories
when they follow accepted configs; reserve refunds for clear instructor-reviewed
infrastructure exceptions.

### Queue Pressure

Risk: individual 12h budgets can overload a small accelerator pool.

Mitigation: fair scheduler, per-student active-run fairness, clear examples for
budget planning, and admin queue visibility.

### Data Leakage

Risk: exact data mix or final evaluation split leaks through logs, examples, or
manifests.

Mitigation: use manifest IDs in public outputs, store exact manifests only in
staff-controlled locations, and redact worker/provider details from public API.

### Scaling-Law Noise

Risk: one-week students fit weak curves or overfit validation noise.

Mitigation: provide conceptual guidance, require prediction intervals, use larger
held-out final evaluation, and grade methodology.

### Resume/Checkpoint Contamination

Risk: non-exact resume changes training curves.

Mitigation: no checkpoint/resume in version one.

## 18. Decision Log

| Branch | Decision | Rationale | Implementation Implication |
|---|---|---|---|
| Planning process | One SoT/design/plan document | Urgent timeline benefits from a single binding artifact | All final design and milestones live here |
| Product scope | Stanford-style fixed-compute MVP | Proven assignment structure with manageable scope | Defer multi-track platform |
| Product scope | Small cohort target | Matches summer-intensive setting | Use fair queue and simple admin tools |
| Product scope | Adapt local scaffold | Existing scaffold covers API, budget, scheduler, validation, and trainer concepts | Harden and customize rather than rebuild |
| Product scope | Include data pipeline under `source/data/` | Stanford scaffold does not fully cover our data-prep needs | Implement download, blend, postprocess, split, tokenize, and planning scripts |
| Assignment | Individual fixed-compute track | Keeps grading and quota simple | Key budgets and final submissions by student |
| Assignment | 12h exploratory / 48h final | Matches CS336 precedent | Express as official accelerator-hours |
| Assignment | Do not disclose GPU model yet | Hardware details are not finalized | Public docs promise consistency, not model name |
| Assignment | Larger held-out final evaluation | Reduces leaderboard noise and overfitting | Maintain separate final eval manifest |
| Assignment | Prediction interval required | Teaches calibrated extrapolation | Extend final-submission schema |
| Scoring | 60/20/20 default | Balances leaderboard, prediction, and methodology | Grading export includes all scoring inputs |
| API | CS336-compatible plus extensions | Preserves familiarity while adding needed fields | Keep endpoint shape; extend schemas |
| API | Same-student duplicate returns 409 with existing ID | Prevents duplicate budget charge and protects privacy | Unique key is `(student_id, config_hash)` |
| API | Per-student API keys | Simple and sufficient for version one | Seed/import keys from CSV |
| Runtime | Keep JAX/Equinox trainer | Fastest path from scaffold | Avoid PyTorch rewrite |
| Runtime | Provider abstraction | Internal GPU API and Slurm must replace Modal-specific launch | Implement thin provider interface |
| Runtime | Internal GPU API first, Slurm for testing | Matches expected deployment and dev needs | Write provider adapter guide |
| Runtime | Public-source but private exact dataset manifest | Reduces data-prep burden while preserving official control | Keep exact source mix and token order private |
| Runtime | Mixed-domain corpus with DCLM subset | Better simulates industry pretraining and avoids a Stanford-identical dataset | Build a private fixed manifest from high-quality public-source components |
| Runtime | 100B-token reservoir target | Large enough for version-one final runs without requiring several hundred billion tokens | Build staff-owned tokenized reservoir and validation split pipeline |
| Runtime | Deterministic runs | Makes scaling-law analysis comparable | Freeze data order and seed semantics |
| Runtime | No checkpoint/resume | Avoids curve contamination | Handle system failures through review/refund |
| Operations | Freeze latest valid final submissions | Prevents post-deadline mutation | Add freeze workflow |
| Operations | Separate final-run queue | Keeps final scoring controlled | Implement final-run batch |
| Operations | Full course export | Supports grading and audit | Export CSV/JSONL records |
| Operations | Audited overrides | Needed for rare exceptions | Store actor/reason/before-after |
| Implementation | Vertical slice first | Exposes integration risk early | M1 submit-to-result smoke run |

## 19. Rejected and Deferred Alternatives

- Multi-track platform: deferred until the fixed-compute loop is stable.
- Simulator-first backend: rejected because the assignment requires real
  training results.
- Full student dashboard: deferred because API/notebook-first is adequate.
- Smaller pilot budget: considered, but reference budget remains viable for a
  one-week assignment.
- Same public validation for final scoring: rejected because it is noisier and
  easier to overfit.
- Team-level quotas: rejected in favor of individual assessment.
- Exact CS336 API clone: rejected because interval prediction and typed failures
  require small extensions.
- Course SSO: deferred in favor of per-student API keys.
- PyTorch rewrite: deferred because it is too risky for urgent delivery.
- Fully public data manifest: rejected because it weakens official-control and
  fairness.
- Fully private proprietary dataset: rejected because it adds unnecessary data
  burden.
- Checkpoint/resume: rejected because non-exact recovery can distort scaling
  curves.
- Required W&B: deferred because internal logs are the source of truth.
- Auto-retry system failures: rejected for version one; clear infrastructure
  failures are instructor-reviewed refund exceptions.
