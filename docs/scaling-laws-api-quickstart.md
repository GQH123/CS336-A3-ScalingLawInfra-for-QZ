# Scaling Laws Project API Quickstart

Status: Draft for student distribution

This quickstart explains the public request and response format for the course
API: check budget, submit runs, poll results, organize data, and submit the
final configuration. Course staff will provide the API URL and your personal API
key before the project begins.

The English and Chinese companion Jupyter notebooks included with the project
release contain the same workflow as executable cells with sample response
objects. Use the notebook version when you want to run or copy code directly.

## 1. Setup

Install the Python `requests` dependency:

```bash
python -m pip install requests
```

Set the API URL and API key:

```python
import json
import os
import sys
import time
import requests

API_BASE_URL = os.environ["SCALING_API_BASE_URL"].rstrip("/")
API_KEY = os.environ["SCALING_API_KEY"]
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def api_request(method, path, **kwargs):
    response = requests.request(
        method,
        f"{API_BASE_URL}{path}",
        headers=HEADERS,
        **kwargs,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"raw_response": response.text}

    if response.status_code == 409:
        return body
    if response.status_code < 200 or response.status_code >= 300:
        print(
            json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True),
            file=sys.stderr,
        )
        raise SystemExit(1)
    return body
```

Your API key is personal. Do not share it or include it in your report.

## 2. Check Budget

```python
budget = api_request("GET", "/budget")
budget
```

Example response:

```json
{
  "student_id": "student-1",
  "total_seconds": 43200,
  "reserved_seconds": 0,
  "charged_seconds": 0,
  "remaining_seconds": 43200
}
```

Budget is reported in seconds. `43200` seconds equals 12 official
accelerator-hours.

## 3. Create a Small Configuration

Start with a small configuration to verify the workflow, correctness, and
stability before spending substantial budget on experiments.

```python
config = {
    "model": {
        "num_hidden_layers": 2,
        "hidden_size": 128,
    },
    "training": {
        "train_tokens": 4096,
        "num_evals": 1,
        "learning_rate": 3e-4,
    },
}
```

The API fills documented defaults such as sequence length, batch size,
evaluation count, optimizer settings, and model seed. It checks submissions
early and rejects incompatible model dimensions, divisibility errors,
non-finite numeric values, requests that exceed your remaining budget, and
similar issues.

## 4. Supported Training Configuration Format

The same training configuration object is used in two places:

- `POST /submit`: pass it as `config`
- `POST /final_submission`: pass it as `training_config`

The public configuration has exactly two top-level objects:

```python
config = {
    "model": {
        # model fields
    },
    "training": {
        # training and optimizer fields
    },
}
```

Use the documented fields below. Fields marked required must be present. Fields
marked optional may be omitted; the API fills the documented default.

Required fields:

- `model.num_hidden_layers`: positive integer, number of Transformer decoder
  blocks. Increasing this at fixed width usually increases non-embedding
  parameters and training FLOPs roughly linearly, so it is one of the main
  model-size knobs for scaling-law sweeps.
- `model.hidden_size`: positive integer, width of the residual stream, token
  embeddings, attention projections, and MLP input/output. This is a strong
  architecture-size knob; attention and MLP parameter counts grow roughly
  quadratically with it at fixed depth. It must be compatible with the attention
  head layout described below.
- `training.train_tokens`: positive integer, total number of training tokens to
  consume from the fixed course data order. The worker trains on the first
  `training.train_tokens` tokens implied by that deterministic stream; this
  field does not reshuffle data or define epochs. Larger values spend more
  optimizer work and more accelerator budget.
- `training.learning_rate`: positive finite float, peak learning rate. With the
  `cosine` schedule, training warms from zero to this value and then decays
  toward `training.learning_rate * training.final_lr_fraction`; with the
  `constant` schedule, the worker uses this value throughout training. Tune this
  together with model size and batch size: too large often produces numerical
  failure, while too small undertrains.

Optional model fields:

- `model.attention_bias`: boolean, default `false`. Adds learned bias terms to
  attention projections when enabled. Leave it fixed across comparable
  experiments so parameter-count changes are attributable to the intended
  scaling variable.
- `model.head_dim`: positive integer, default
  `model.hidden_size / model.num_attention_heads`. This is the width of each
  attention head. It is usually derived from `model.hidden_size` and
  `model.num_attention_heads`; supply it only when you want the API to check an
  explicit head layout.
- `model.intermediate_size`: positive integer, default `4 * model.hidden_size`.
  This is the hidden width of the feed-forward/SwiGLU block. Larger values
  increase non-embedding parameters and FLOPs substantially; common sweeps keep
  it as a fixed multiple of `model.hidden_size`.
- `model.num_attention_heads`: positive integer, default `1`. This is the number
  of query attention heads. At fixed `model.hidden_size`, more heads mean a
  smaller `model.head_dim`; keep `model.head_dim` valid for the GPU
  attention backend.
- `model.num_key_value_heads`: positive integer, default
  `model.num_attention_heads`. This would control grouped-query attention, but
  the current Stanford reference runtime does not support grouped-query
  attention, so set it equal to `model.num_attention_heads`.
- `model.rms_norm_eps`: positive finite float, default `1e-6`. Epsilon added in
  RMSNorm for numerical stability. The default is the reference-trainer value;
  changing it is usually an ablation rather than a scaling-law knob.
- `model.rope_theta`: positive integer, default `1000000`. Rotary-position
  embedding frequency base. Leave the default unless you are deliberately
  studying position-encoding behavior.
- `model.tie_word_embeddings`: boolean, default `false`. When true, the output
  classifier reuses the input embedding matrix, reducing parameters. Keep this
  choice fixed across runs that you want to compare directly.
- `model.dtype`: one of `float32`, `bfloat16`; default `bfloat16`. `bfloat16`
  is the normal GPU-efficient training dtype. `float32` uses more memory and
  runtime and is mainly useful for controlled numerical comparisons.
- `model.vocab_size`: positive integer, default `50432`. This must match the
  tokenized data requirements. The current course deployment uses
  `EleutherAI/gpt-neox-20b` token IDs and requires at least `50432`; increasing
  it adds embedding/output parameters that usually do not represent useful model
  capacity for this assignment.

Optional training fields:

- `training.sequence_length`: positive integer, default `1024`. Number of
  tokens in each training sequence/context window. It affects memory use and
  the number of tokens per optimizer step:
  `training.sequence_length * training.train_batch_size`.
- `training.train_batch_size`: positive integer, default `1`. Number of
  sequences in each optimizer step. At fixed `training.train_tokens`, larger
  batches produce fewer optimizer updates and usually require more memory.
- `training.validation_batch_size`: positive integer, default `1`. Number of
  validation sequences evaluated at once. It does not change the training
  objective, but it must fit memory and satisfy validation-token divisibility.
- `training.num_evals`: positive integer, default `10`. Number of training
  chunks/evaluation points. The worker trains for
  `training.train_tokens / training.num_evals` tokens per chunk and records one
  validation loss after each chunk; the final validation loss is the last entry.
  For tiny smoke tests with only a few optimizer steps, set this to `1`.
- `training.model_seed`: non-negative integer, default `0`. Controls model
  initialization only. The course data order is fixed, so changing this seed
  does not reshuffle the training stream.
- `training.optimizer`: one of `adamw`, `sgd`; default `adamw`. AdamW is the
  default reference optimizer. SGD is available for controlled comparisons but
  usually needs separate learning-rate tuning.
- `training.lr_schedule`: one of `cosine`, `constant`; default `cosine`.
  `cosine` uses `training.warmup_fraction` and
  `training.final_lr_fraction`; `constant` keeps the learning rate fixed at
  `training.learning_rate`.
- `training.weight_decay`: non-negative finite float, default `0.0`. Decoupled
  AdamW weight decay strength. It is ignored by SGD in the reference runtime.
  Common AdamW baselines often use small values such as `0.01`.
- `training.adam_beta1`: finite float in `[0, 1)`, default `0.9`. AdamW first
  moment coefficient. Ignored by SGD.
- `training.adam_beta2`: finite float in `[0, 1)`, default `0.95`. AdamW second
  moment coefficient. Ignored by SGD.
- `training.adam_epsilon`: positive finite float, default `1e-8`. AdamW
  denominator epsilon for numerical stability. Ignored by SGD.
- `training.warmup_fraction`: finite float in `[0, 1)`, default `0.0`. Fraction
  of optimizer steps used to increase the learning rate from zero to
  `training.learning_rate` under the cosine schedule.
- `training.final_lr_fraction`: finite float in `[0, 1]`, default `0.0`. Final
  learning-rate multiplier for the cosine schedule. For example, `0.1` ends at
  ten percent of the peak learning rate.
- `training.gradient_clip_norm`: positive finite float, default `1.0`. Global
  gradient-norm clipping threshold. Lower values can stabilize spikes but may
  slow learning if they clip most updates.

Important validation rules:

- If `model.head_dim` is supplied, `model.hidden_size` must equal
  `model.num_attention_heads * model.head_dim`. If `model.head_dim` is omitted,
  `model.hidden_size` must be divisible by `model.num_attention_heads`, and the
  API derives `model.head_dim`.
- `model.num_attention_heads` must be divisible by
  `model.num_key_value_heads`, and the current Stanford reference runtime
  requires `model.num_key_value_heads == model.num_attention_heads` because
  grouped-query attention is not supported.
- `model.head_dim` must be `<= 128` and a multiple of `8` for the current GPU
  attention backend. For example, use `model.num_attention_heads: 16` with
  `model.hidden_size: 2048` so `model.head_dim` is `128`.
- `model.intermediate_size` must be greater than or equal to
  `model.hidden_size`.
- `model.vocab_size` must be at least `50432` for the current
  `EleutherAI/gpt-neox-20b` tokenized dataset. Smaller vocabularies can make
  token IDs exceed the embedding table and are rejected.
- When course staff run workers with `worker_gpu_count > 1`, the Stanford FSDP
  parameter sharding also requires `model.hidden_size` and
  `model.intermediate_size` to be divisible by `worker_gpu_count`. If
  `model.tie_word_embeddings` is `false`, `model.vocab_size` must also be
  divisible by `worker_gpu_count`.
- `training.train_tokens` must be divisible by
  `training.sequence_length * training.train_batch_size`.
- `training.train_tokens` must be at most `500000000000`.
- Total optimizer steps equal `training.train_tokens` divided by
  `training.sequence_length * training.train_batch_size`.
- Total optimizer steps must be divisible by `training.num_evals`.
- Validation batches must fit the course validation-token setting; incompatible
  `training.sequence_length` or `training.validation_batch_size` values are
  rejected.
- When course staff run workers with `worker_gpu_count > 1`,
  `training.train_batch_size` and `training.validation_batch_size` must be
  divisible by `worker_gpu_count` because the Stanford trainer shards the batch
  axis across the FSDP mesh.
- All numeric fields must be finite, and fields documented as positive must be
  strictly greater than zero. In particular, `training.learning_rate` must be
  positive and finite.
- Unsupported top-level, `model`, or `training` fields are rejected, including
  legacy model field names. This helps catch misspelled field names before they
  silently change an experiment.

If a final submission uses a configuration that fails these checks, the API
returns `invalid_final_submission` and does not store that final submission.

## 5. Submit a Run

```python
submit_result = api_request(
    "POST",
    "/submit",
    json={
        "config": config,
        "requested_runtime_seconds": 300,
    },
)
submit_result
```

Example success response:

```json
{
  "experiment_id": "exp-000001",
  "status": "queued",
  "budget_reserved_seconds": 300,
  "resource_warnings": [
    "very_low_optimizer_steps"
  ],
  "resolved_config": {
    "parameter_count_estimate": 4260096,
    "tokens_per_optimizer_step": 1024,
    "total_optimizer_steps": 4,
    "resource_warnings": [
      "very_low_optimizer_steps"
    ]
  }
}
```

Queued and running experiments reserve the full requested runtime. If a run
finishes early, unused reserved budget is returned according to course rules.
`resource_warnings` are advisory and do not reject the run.

Example duplicate response:

```json
{
  "error": "duplicate_config",
  "message": "An identical configuration was already submitted.",
  "experiment_id": "exp-000001"
}
```

Duplicate detection is scoped to your API key and does not expose other
students' experiment IDs or configs.

## 6. Poll a Run

```python
experiment_id = submit_result["experiment_id"]
terminal_statuses = {"completed", "failed", "cancelled", "system_failed"}

while True:
    experiment = api_request("GET", f"/experiment/{experiment_id}")
    status = experiment["status"]
    print("status:", status)

    if status in terminal_statuses:
        break

    time.sleep(30)
```

Example completed response:

```json
{
  "experiment_id": "exp-000001",
  "status": "completed",
  "validation_losses": [4.2, 3.7],
  "final_validation_loss": 3.7,
  "failure_reason": "",
  "used_runtime_seconds": 180,
  "completed_at": "2026-07-16T15:00:00Z",
  "failed_at": "",
  "resource_warnings": []
}
```

For completed runs, use `final_validation_loss` as the run result.
`used_runtime_seconds` is the billable runtime under the course runtime rules.
`validation_losses` is an ordered history of validation evaluations during
training. It is not a confidence interval and not `[lower, upper]`; for a
completed run, `final_validation_loss` equals the last value in
`validation_losses`.

Example failed response:

```json
{
  "experiment_id": "exp-000002",
  "status": "failed",
  "validation_losses": [4.6, 4.1],
  "final_validation_loss": null,
  "failure_reason": "timeout",
  "used_runtime_seconds": 600,
  "completed_at": "",
  "failed_at": "2026-07-16T15:00:00Z",
  "resource_warnings": []
}
```

Partial validation losses from failed runs are diagnostics for debugging. They
are not final validation losses and should not be treated as completed-run
results.

## 7. List and Save Results

```python
experiments = api_request("GET", "/experiments")["experiments"]

rows = []
for exp in experiments:
    rows.append(
        {
            "experiment_id": exp["experiment_id"],
            "status": exp["status"],
            "final_loss": (
                exp["final_validation_loss"]
                if exp["status"] == "completed"
                else None
            ),
            "validation_losses": exp["validation_losses"],
            "used_runtime_seconds": exp["used_runtime_seconds"],
        }
    )
```

Keep an experiment log with each run's purpose, the scaling-law hypothesis it
tested, and how the result changed your next experimental decision. Make sure
your submitted configuration can train reliably; final training crashes caused
by configuration robustness issues make the final score invalid.

## 8. Submit the Final Configuration

Your final submission includes a training config, a predicted final loss, and a
prediction interval. Invalid configs detected by the API return
`invalid_final_submission` and are not stored.

```python
final_submission = api_request(
    "POST",
    "/final_submission",
    json={
        "training_config": config,
        "predicted_final_loss": 3.25,
        "predicted_final_loss_lower": 3.18,
        "predicted_final_loss_upper": 3.36,
    },
)
final_submission
```

Example response:

```json
{
  "student_id": "student-1",
  "training_config": {
    "model": {"num_hidden_layers": 2, "hidden_size": 128},
    "training": {"train_tokens": 4096, "num_evals": 1, "learning_rate": 0.0003}
  },
  "predicted_final_loss": 3.25,
  "predicted_final_loss_lower": 3.18,
  "predicted_final_loss_upper": 3.36,
  "updated_at": "2026-07-16T15:00:00Z",
  "frozen_at": "",
  "frozen_by": "",
  "freeze_reason": ""
}
```

You may submit multiple times before the deadline. The latest valid final
submission before the deadline is used for final evaluation.

Confirm the current final configuration:

```python
final_submission = api_request("GET", "/final_submission")
final_submission
```

After the deadline, this endpoint returns the frozen record with `frozen_at`,
`frozen_by`, and `freeze_reason` populated.

## 9. Common Errors

Invalid API key:

```json
{
  "error": "invalid_api_key",
  "message": "Invalid or missing API key."
}
```

Invalid configuration:

```json
{
  "error": "invalid_config",
  "message": "model.hidden_size must be divisible by model.num_attention_heads"
}
```

Missing final submission:

```json
{
  "error": "final_submission_not_found",
  "message": "Final submission not found."
}
```

Contact course staff if you have further questions.
