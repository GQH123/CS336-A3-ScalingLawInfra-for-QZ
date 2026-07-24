# Scaling-Law Assignment Backend: Research and Introduction

Date: 2026-07-16

Workspace: `/data/_backup/_topics/fnlp_summer_training/2026/assignment-3`

Primary local references:

- Assignment handout: `cs336_assignment3_scaling.pdf`
- Stanford-style scaffold: `cs336-assignment3-scaling/`
- Public API reference implementation: `cs336-assignment3-scaling/cs336_scaling/api/public.py`
- Training config validation: `cs336-assignment3-scaling/cs336_scaling/training/training_config.py`
- Budget accounting: `cs336-assignment3-scaling/cs336_scaling/budget.py`
- Scheduler selection: `cs336-assignment3-scaling/cs336_scaling/scheduler/experiment_selector.py`

This document is a research and context export. It is not yet the final Source of Truth
or implementation plan. Its purpose is to give the next planning pass a grounded starting
point: what the original assignment does, what a comparable backend needs to guarantee,
which design choices should be made explicitly, and which assignment extensions are worth
considering.

## Executive Summary

The original CS336 assignment is a compact and effective teaching setup for empirical
scaling laws. Students receive a fixed exploratory training budget, submit model/training
configurations to a real GPU-backed API, observe validation losses, fit scaling laws, and
submit one final configuration plus a predicted validation loss for a larger hidden-budget
run.

For our summer intensive version, the first backend should preserve this core loop:

1. Students submit full training configurations.
2. The backend validates configurations and reserves budget.
3. A worker runs exactly the submitted configuration.
4. The system records validation losses over time.
5. Completed runs return a final validation loss.
6. Failed runs return clear diagnostics but no invented final loss.
7. A quota ledger prevents budget abuse.
8. A final hidden run is scored after the deadline.

The main implementation challenge is not the HTTP API itself. The main challenge is
making the result contract defensible: what counts as a completed run, what happens on
timeout, how numerical instability is reported, which failures are charged, and how to
separate student-caused failures from infrastructure failures.

The first version should therefore be a Stanford-style real-training API with a more
explicit operational policy. Multi-track assignment variants can be added later if the
schema is designed cleanly from the beginning.

## Original Assignment Model

The local handout describes the following student task:

- Students are given a 12 B200-hour exploratory budget.
- They query a hosted training API with model hyperparameters, optimizer settings, data
  scale, batch sizes, evaluation count, and maximum runtime.
- The API queues real single-B200 training runs.
- The API returns status and validation losses.
- Students fit scaling laws from their exploratory runs.
- Students submit one final configuration intended for a 48 B200-hour hidden run.
- Students also submit their predicted final validation loss.
- Leaderboard performance depends on the actual validation loss of the final hidden run.

The pedagogical point is to force students to reason about the model-size/data-size tradeoff
under fixed compute. In the simplest approximation, training compute for a dense Transformer
is often modeled as roughly proportional to `6 * N * D`, where `N` is parameter count and
`D` is training tokens. The assignment asks students to infer how the compute-optimal `N`
and `D` should scale, and to choose concrete architecture and optimizer settings that make
the predicted model train well.

Important operational details from the handout:

- Queued and running experiments reserve their full `max_runtime_seconds`.
- Completed or failed experiments are later charged by actual reported runtime, clipped to
  a minimum and maximum.
- Timeout is charged as the full `max_runtime_seconds`.
- Duplicate training configs are rejected.
- Final validation loss for a completed experiment is the final element of `status.val_losses`.
- Failed timeout experiments may show partial losses, but they are not completed runs.
- Data order is fixed and deterministic.
- The validation set is fixed for exploratory runs, while final leaderboard validation can
  use a larger or hidden validation set.
- The model seed controls initialization, not data ordering.
- The original API fixes sequence length and validation-token count.

## Local Scaffold Observations

The local `cs336-assignment3-scaling` repository is more than a client scaffold; it includes
a functional server-side architecture.

### API Shape

The public API implements:

- `POST /submit`
- `GET /budget`
- `GET /experiments`
- `GET /experiment/{experiment_id}`
- `POST /final_submission`
- `GET /final_submission`

The implementation uses FastAPI, Pydantic schemas, SQLAlchemy, and Postgres row locking.
The `/submit` endpoint locks the user row, computes current budget usage, checks whether
the new run fits in the remaining budget, rejects duplicate config hashes, inserts the
experiment, and writes a queue event.

### Configuration Validation

The `TrainingConfig` schema validates core consistency constraints:

- `training.train_tokens` must divide evenly into optimizer steps.
- Optimizer steps must divide evenly into `training.num_evals`.
- validation tokens must divide evenly by sequence length and validation batch size.
- positive architecture fields are enforced.
- `hidden_size == num_attention_heads * head_dim`.
- attention-head divisibility constraints are enforced.
- GQA is rejected in the current scaffold.
- Stanford GPU attention requires `head_dim <= 128` and a multiple of `8`;
  this also satisfies the RoPE even-dimension requirement.
- LR schedule fractions are bounded.
- AdamW hyperparameters are bounded.

This validation layer is important because it prevents wasting GPU time on malformed
requests. A summer-course backend should keep this style of validation and add any
resource preflight checks needed for the local training stack.

### Budget Accounting

The local budget logic charges:

- completed/failed experiments by `used_runtime_seconds`, clipped between 1 second and
  `max_runtime_seconds`
- queued/running experiments by reserved `max_runtime_seconds`

This is the right user-facing mental model. Students should not be able to reserve more
than their quota, and they should recover unused budget from runs that finish early.

### Scheduler Fairness

The scheduler ranks queued experiments using current running count by user plus FIFO order.
That means users with fewer running jobs get priority, while still preserving queue order
within a rough fairness envelope. This is a good default for a course environment where
students share a finite GPU pool.

### Worker Lifecycle

The worker launches real training on Modal B200 workers, writes logs and config files,
initializes a W&B run, reports internal worker events, periodically emits validation losses,
and reports completion or failure through internal endpoints.

The key teaching feature is that students do not run local training. They interact with a
fixed training environment. This makes their experimental results comparable and prevents
environment-specific implementation differences from dominating the scaling-law exercise.

## Recommended Backend Architecture for Our Version

The first version should have five bounded subsystems:

1. API gateway
2. Experiment database and event ledger
3. Scheduler/dispatcher
4. Training worker runtime
5. Instructor/admin tooling

### 1. API Gateway

Responsibilities:

- authenticate students
- validate configs
- reject malformed or impossible requests before queueing
- reserve budget atomically
- deduplicate identical configs
- expose experiment status
- accept final submissions

Recommended endpoints:

- `GET /health`
- `GET /budget`
- `POST /submit`
- `GET /experiments`
- `GET /experiment/{id}`
- `POST /final_submission`
- `GET /final_submission`
- instructor-only: `GET /admin/experiments`
- instructor-only: `POST /admin/experiments/{id}/refund`
- instructor-only: `POST /admin/experiments/{id}/cancel`
- instructor-only: `POST /admin/final_runs/launch`

The public API should return structured error bodies, not only strings. Students should be
able to programmatically distinguish validation errors, duplicate submissions, insufficient
budget, and system failures.

### 2. Experiment Database and Event Ledger

Use an append-friendly status model rather than overwriting all state without history.

Recommended tables:

- `users`
- `api_keys`
- `experiments`
- `experiment_events`
- `budget_adjustments`
- `final_submissions`
- `final_runs`

Each experiment should store:

- student ID
- canonical config JSON
- config hash
- resolved derived fields
- current status
- queued time
- dispatched time
- run ID
- used runtime
- validation losses
- failure reason, if any

Each event should store:

- experiment ID
- event type
- timestamp
- payload
- worker run ID, where applicable

The event log is useful for auditing disputes, debugging worker failures, reconstructing
leaderboard state, and publishing anonymized course statistics.

### 3. Scheduler and Dispatcher

The scheduler should make the following policy choices explicit:

- global max concurrent workers
- per-student max active workers
- whether to reserve headroom for students with zero running jobs
- whether final runs are isolated from exploratory runs
- whether instructor jobs can preempt or bypass normal queues

Recommended first-version policy:

- exploratory queue: fair-share FIFO by student active count
- final queue: launched after deadline, separate from exploratory queue
- no student-controlled priority
- no retry on student-caused failures
- automatic retry/refund on infrastructure failures

### 4. Training Worker Runtime

Workers should run from a frozen job manifest:

- config JSON
- resolved config
- code version
- container image digest
- dataset version
- tokenizer version
- validation split ID
- model seed
- submission ID
- max runtime

The worker should:

- load the fixed dataset in deterministic order
- initialize the exact model requested
- run the fixed training loop
- evaluate after each configured chunk
- emit validation-loss events
- checkpoint only for infrastructure recovery, not for user-driven tuning
- detect timeout, NaN, divergence, OOM, and worker interruption
- report final status exactly once

No hidden rescue behavior should be applied. If a student submits an LR that diverges, the
run should fail. If a model is too large to fit and the preflight should have caught it,
that is a backend bug. If the model passes preflight but OOMs due to runtime overhead, it
should be reported as a resource failure with a clear policy on budget charge/refund.

### 5. Instructor and Admin Tooling

The first version should include enough tooling for course operation:

- seed/import student API keys
- view queue state
- inspect failed runs
- refund system-caused failures
- cancel stuck jobs
- export all experiment metadata
- freeze final submissions at deadline
- launch final runs in batch
- export leaderboard results
- generate anonymized run logs for teaching

The admin tooling can be simple CLI scripts at first. A dashboard is useful but not required
for version one.

## Result and Failure Semantics

This is the most important Source-of-Truth section to define before implementation.

Recommended result contract:

| Case | API status | Final validation loss | Budget charge | Scoring use |
|---|---|---:|---:|---|
| Completed run | `completed` | last entry of `val_losses` | actual runtime clipped to policy bounds | valid |
| Schema/config validation error | HTTP 400/422 | none | 0 | invalid request |
| Duplicate config | HTTP 409 | none, or link to existing experiment | 0 | no new run |
| Insufficient budget | HTTP 400 | none | 0 | invalid request |
| Timeout | `failed_timeout` | none | full `max_runtime_seconds` | invalid for leaderboard |
| NaN/loss divergence | `failed_numerical` | none | actual runtime clipped | invalid or worst-score penalty |
| Predictable OOM caught preflight | HTTP 400/422 | none | 0 | invalid request |
| OOM after accepted launch | `failed_resource` | none | actual runtime unless instructor refunds | invalid |
| Worker preemption | `system_retrying` then retry | none until terminal status | 0 until success/final failure | not student fault |
| Infrastructure failure | `system_failed` or auto-retry | none | 0 or refunded | not student fault |
| Instructor cancellation | `cancelled` | none | policy-specific | excluded or refunded |

The backend should not fabricate a final loss for failed runs. Partial validation losses are
useful for student learning, but they should be labeled as partial diagnostics. For scoring,
a failed final submission should either be invalid or receive a fixed worst-score penalty.

For the final hidden run, stricter semantics are better:

- completed: score by hidden final validation loss
- student-caused failure: worst score or invalid final submission
- infrastructure failure: retry from scratch or from checkpoint under instructor control
- deadline missed: no final run

## Training Stability Policy

The backend should guarantee faithful execution, not successful training for every config.

Pre-launch safeguards:

- schema validation
- parameter-count computation
- memory estimate
- optimizer-state estimate
- activation-memory estimate
- batch-token divisibility checks
- max token cap
- max runtime cap
- dtype constraints
- architecture consistency constraints

Runtime safeguards:

- fixed gradient-clipping behavior if exposed in config
- finite-loss checks
- heartbeat events
- timeout watchdog
- structured exception capture
- checkpointing for infrastructure recovery
- validation-loss event logging

Stability rules should be public. If the training loop always uses gradient clipping, the
students should know. If gradient clipping is configurable, the default and allowed range
should be documented. If loss becomes NaN, the run should fail visibly rather than being
silently repaired.

## Assignment Extensions Worth Considering

The original single-track assignment is already strong. Extensions should improve learning
without turning the course into an operations burden.

### 1. Prediction Accuracy Scoring

Current-style leaderboard scoring rewards the lowest final validation loss. Add a second
score for prediction accuracy:

- submit predicted final loss
- optionally submit a confidence interval
- score absolute prediction error or a calibrated interval score

This makes the assignment about forecasting, not only hyperparameter gambling.

### 2. Fixed-Compute Track

This is the original task. Students choose model size, data size, and hyperparameters under
a fixed training budget.

Learning target:

- Chinchilla-style compute-optimal scaling
- IsoFLOP curves
- extrapolation under limited experimental budget

### 3. Fixed-Model-Size Track

Staff fixes approximate non-embedding parameter count. Students optimize:

- training tokens
- learning rate
- batch size
- warmup
- decay schedule
- regularization/stability knobs

Learning target:

- hyperparameter transfer across scales
- optimization stability
- separating architecture scaling from training-recipe scaling

### 4. Fixed-Data Track

Staff fixes unique data tokens. Students decide whether to:

- train longer with repeated data
- train a smaller model
- change LR schedule
- regularize more
- stop earlier

Learning target:

- data-constrained scaling
- repeated-token behavior
- why "more compute" is not always equivalent to "more data"

### 5. Overtraining / Inference-Cost Track

Students optimize under a deployment-aware objective, such as:

`training_cost + expected_inference_cost`

This lets students investigate when it is better to train a smaller model for more tokens,
even if that is not purely compute-optimal under pretraining loss.

Learning target:

- distinction between training-optimal and deployment-optimal models
- why real labs often overtrain smaller models

### 6. Data-Mixture Track

Give students several data buckets and allow mixture weights:

- general web
- code
- math
- multilingual
- high-quality filtered data

Learning target:

- data quality changes scaling behavior
- validation domain matters
- token count alone is not a complete description of data

This is pedagogically rich but operationally more expensive because it requires stable
data preprocessing and careful hidden evaluation design.

### 7. Hidden Distribution-Shift Track

Use one public validation distribution for exploratory feedback and a related but hidden
final validation distribution for scoring.

Learning target:

- robustness of scaling-law fits
- dangers of overfitting a leaderboard metric
- uncertainty estimation

### 8. Downstream-Performance Prediction Track

Ask students to predict downstream task performance from validation loss and scale.

Possible tasks:

- multiple-choice knowledge
- code
- math
- multilingual

Learning target:

- loss is useful but incomplete
- downstream performance can scale differently by domain
- prediction intervals matter

### 9. Test-Time Compute Track

If generation infrastructure is available, give students a fixed inference budget and let
them choose:

- model size
- sampling/search strategy
- number of attempts
- reasoning-token budget

Learning target:

- pretraining scale versus inference-time compute
- modern reasoning-model tradeoffs

This should probably not be in version one unless the course already has reliable evaluation
infrastructure for generated answers.

### 10. Stability and Reliability Track

Score students partly on successful completion rate or penalize failed final configs.

Learning target:

- a scaling law is only useful if the proposed run trains
- large-run extrapolation must include risk management

## Literature Review and Design Implications

This section summarizes public sources that are most relevant to the assignment design.

### Kaplan et al., OpenAI: Scaling Laws for Neural Language Models

Source: https://arxiv.org/abs/2001.08361

Core relevance:

- Establishes empirical power-law relationships between language-model loss, model size,
  dataset size, and compute.
- Motivates fitting smooth functional forms from smaller-scale runs.
- Gives students the baseline mental model that scaling laws are empirical forecasts rather
  than exact theoretical guarantees.

Design implication:

- The assignment should require students to justify their fitted functional form and show
  diagnostics, not merely report a final config.

### Hoffmann et al., DeepMind: Training Compute-Optimal Large Language Models

Source: https://arxiv.org/abs/2203.15556

Core relevance:

- Chinchilla-style compute-optimal scaling is the closest match to the original assignment.
- The IsoFLOP profile method is directly teachable: for each compute budget, vary model
  size, find the best point, then fit how optimal model size and data size scale with compute.
- The central lesson is that optimal model size and training tokens must be traded off,
  not maximized independently.

Design implication:

- The first version should include a fixed-compute final run and encourage or require
  IsoFLOP-style analysis.

### OpenAI: GPT-4 Technical Report

Source: https://arxiv.org/abs/2303.08774

Core relevance:

- Presents the idea that some large-run behavior can be predicted from smaller runs.
- Supports making prediction accuracy an explicit part of the score.

Design implication:

- Students should submit both a final configuration and predicted loss. A prediction-error
  score makes the exercise more faithful to real scaling-law practice.

### DeepSeek LLM: Scaling Open-Source Language Models with Longtermism

Source: https://arxiv.org/html/2401.02954

Core relevance:

- Discusses scaling-law fitting with real training runs.
- Uses IsoFLOP profiles and studies optimal model/data allocation.
- Emphasizes hyperparameter scaling and data-quality effects.
- Uses non-embedding FLOPs/token as a scale proxy, which is useful when vocabulary and
  embedding choices vary.

Design implication:

- The backend should compute and expose derived quantities such as non-embedding parameter
  count and estimated FLOPs.
- Data-mixture and data-quality tracks are valuable, but probably version-two features.

### Meta: The Llama 3 Herd of Models

Source: https://arxiv.org/abs/2407.21783

Core relevance:

- Shows that production LLM development involves data mixture, downstream evaluations,
  overtraining choices, tokenizer decisions, and scaling beyond pure validation loss.
- Useful for explaining why a course assignment can start with loss but later extend to
  downstream prediction and deployment-aware tradeoffs.

Design implication:

- The first version should keep the metric simple, but the schema should not prevent later
  downstream-eval tracks.

### Muennighoff et al.: Scaling Data-Constrained Language Models

Source: https://arxiv.org/abs/2305.16264

Core relevance:

- Directly supports a fixed-data or repeated-data assignment variant.
- Helps students see why scaling behavior changes when unique data is scarce.

Design implication:

- A fixed-data track is one of the best second tracks because it asks a different conceptual
  question without requiring a completely different backend.

### Sardana et al.: Beyond Chinchilla-Optimal

Source: https://arxiv.org/abs/2401.00448

Core relevance:

- Argues that inference costs can change what "optimal" means.
- Real deployments care about total cost, not only training compute.

Design implication:

- An overtraining or inference-cost-aware track would be a strong advanced extension.

### Kimi k1.5 / OpenAI o1: Test-Time Compute and Reasoning

Sources:

- https://arxiv.org/abs/2501.12599
- https://openai.com/index/learning-to-reason-with-llms/

Core relevance:

- Modern reasoning systems introduce another scaling axis: inference-time compute.
- This is outside the original CS336 assignment but highly relevant to current model-building
  practice.

Design implication:

- Do not put this in the first backend unless generation evaluation is already reliable.
  Keep it as a later module or capstone extension.

### DeepSeek-V3 and Kimi K2

Sources:

- https://arxiv.org/abs/2412.19437
- https://arxiv.org/abs/2507.20534

Core relevance:

- These reports are useful for thinking about operational stability, efficient scaling,
  mixture-of-experts, and large-scale training practice.

Design implication:

- Version one should probably use dense models for simplicity. However, the backend should
  record enough resolved metadata that a future MoE or sparse-model track is not impossible.

### Qwen3

Source: https://arxiv.org/abs/2505.09388

Core relevance:

- Useful for discussing thinking modes, inference control, and model-family scaling in a
  modern open-model context.

Design implication:

- Supports a later inference-budget or reasoning-budget module, not a first-version backend
  requirement.

## Recommended First-Version Scope

The first version should include:

- dense decoder-only Transformer training
- fixed tokenizer and sequence length
- fixed train/validation data order
- one exploratory budget
- one final hidden run
- public validation losses during exploratory runs
- hidden final validation set or larger final validation sample
- quota accounting
- fair scheduler
- explicit failure semantics
- final predicted loss submission
- exportable experiment logs

The first version should not include:

- multiple data mixtures
- MoE models
- RL/post-training
- inference-time search
- variable tokenizer/vocabulary
- student-supplied training code
- adaptive backend fixes to bad configs

These exclusions keep the assignment operationally feasible and make the learning objective
clear.

## Candidate Version-One Student Workflow

1. Read assignment handout and API documentation.
2. Run a small sanity-check experiment.
3. Check `/budget` and `/experiments`.
4. Design a set of exploratory runs.
5. Fit one or more scaling-law models.
6. Analyze residuals and uncertainty.
7. Select final model size and training tokens.
8. Choose architecture and optimizer hyperparameters.
9. Submit final config and predicted final loss.
10. Write methodology report with enough detail for reproduction.

The platform should support this workflow without giving students built-in scaling-law
solutions. The backend provides measurements; students provide experimental design and
analysis.

## Candidate Version-One Grading Rubric

Possible scoring components:

- 40% final hidden validation loss
- 20% predicted-loss accuracy
- 20% methodology and scaling-law analysis
- 10% experimental design under budget
- 10% robustness and reproducibility of submitted analysis

Alternative leaderboard-heavy scoring:

- 70% final hidden validation loss
- 15% predicted-loss accuracy
- 15% writeup

For an intensive course, the first rubric is better because it rewards learning and not only
leaderboard luck.

## Open Decisions for Source-of-Truth Planning

These are the decisions that should be resolved in the upcoming `superpowers:brainstorming`
and `grill-me` session.

### Product Scope

- Is version one a course-internal teaching tool or a reusable public platform?
- How many students and teams must it support?
- How many GPUs are realistically available?
- Is B200 the target hardware, or should the system abstract over local/H100/A100/B200?
- Does the course need a dashboard in version one?

### Assignment Scope

- Should version one be only the fixed-compute track?
- Should prediction accuracy be part of the leaderboard score?
- Should failed final submissions receive worst score or be invalid?
- Should students submit confidence intervals?
- Should the hidden final validation set differ from exploratory validation?

### Backend Policy

- What exact status enum should be public?
- Which failures are charged to students?
- Which failures are automatically retried or refunded?
- Should duplicate configs return HTTP 409 or the existing experiment ID?
- Should partial losses from failed runs be downloadable?
- Should instructors be able to manually override status and budget?

### Training Runtime

- Which framework should version one use?
- Which model architecture is fixed versus configurable?
- What parameter ranges are allowed?
- How will memory preflight be computed?
- Is checkpointing needed for version one?
- Is W&B or another experiment tracker required?

### Operations

- What is the final-submission deadline behavior?
- How are API keys provisioned?
- How are logs retained?
- How are final hidden runs launched and audited?
- How are disputes handled?

### Source of Truth Structure

- Where should the canonical SoT live?
- Should it include both assignment policy and implementation architecture?
- Should API schemas be generated from the SoT or merely documented by it?
- How often should milestone decisions be written back while we grill the plan?

## Suggested Next Step

Use this document as the context artifact for the next pass. The next pass should not start
by designing code. It should first define the Source-of-Truth structure and resolve the
policy decisions that determine implementation behavior:

1. maintenance policy for decision summaries
2. SoT file path and ownership
3. version-one scope boundary
4. scoring and failure semantics
5. backend architecture
6. implementation milestones

Only after those decisions are stable should we write an implementation plan.
