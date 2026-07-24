# Project: Scaling Laws for Language Model Training

Status: Draft for student distribution

## 1. Overview

In this project, you will study empirical scaling laws for language model
training. Your goal is to use a limited experimental budget to predict how model
validation loss changes as model size, data scale, optimizer settings, and
training compute change.

Scaling laws are useful because large training runs are too expensive to explore
by trial and error. A good scaling-law study treats smaller runs as empirical
evidence, then decides which patterns are stable enough to extrapolate. This
project asks you to make those decisions under a real budget: every exploratory
run consumes time, but every well-designed run should provide useful information
about the best final configuration.

You will not implement a training system from scratch. Instead, you will submit
training configurations to the course API. The course infrastructure will run
real language-model training jobs using a fixed training stack and a consistent
staff-controlled hardware environment. The API will return validation losses
from your official runs.

At the end of the project, you will submit one final training configuration for
a larger final pretraining run, along with your predicted final validation loss
and an uncertainty interval. Your score depends on both the actual validation
loss of your final model and the quality of your prediction.

The central question is:

> Given a fixed training-compute budget, how should we trade off model size,
> training tokens, and hyperparameters to obtain the lowest validation loss?

The project therefore asks you to solve both a compute-constrained optimization
problem and an experimental-inference problem. You are not only searching for a
good final configuration; you are also deciding which exploratory experiments
make your final prediction effective and credible.

## 2. Learning Goals

By the end of this project, you should be able to:

- explain the model-size/data-size tradeoff in language model scaling
- design small-scale experiments that support extrapolation to a larger run
- fit and critique empirical scaling-law models
- reason about uncertainty in extrapolated predictions
- choose architecture and optimizer hyperparameters under a compute budget
- distinguish completed runs, failed runs, partial diagnostics, and final scores
- write a reproducible methodology report for a scaling-law experiment

## 3. Project Structure

The project has two phases.

The first phase is for exploratory experiments, evidence collection, and
scaling-law modeling. The second phase is a single final pretraining run that
tests the configuration and prediction produced by your analysis.

### Phase 1: Exploratory Scaling-Law Experiments

Each student receives an official exploratory budget of:

- **12 official accelerator-hours**

You may use this budget to submit training configurations to the course API.
Each accepted experiment is queued and run by the course infrastructure. The API
will report the experiment status and validation losses.

You should use these exploratory runs to decide:

- which model sizes to try
- how many training tokens to use
- how to allocate compute across runs
- which optimizer and learning-rate settings are stable
- which scaling-law fitting method to trust
- how uncertain your extrapolation is

Exploratory runs do not need to be large to be useful. Small runs can establish
whether a configuration is valid, whether losses are finite, and how runtime
scales. Larger exploratory runs are useful when they reduce extrapolation error
or distinguish between competing scaling-law fits.

### Phase 2: Final Pretraining Run

At the end of the project, you will submit one final configuration for a larger
official run with budget:

- **48 official accelerator-hours**

You will also submit:

- your predicted final validation loss
- a lower bound for your prediction interval
- an upper bound for your prediction interval

After the deadline, the course staff will freeze each student's latest valid
final submission. The final runs will be launched by the course infrastructure.
The final validation losses will be hidden until results are released.

## 4. Official Compute and Hardware Policy

The project uses official accelerator-hours rather than disclosing a specific
hardware model. The course guarantees that official exploratory and final runs
use a consistent staff-controlled hardware type and training stack.

Only official API runs count for this project. You should not use private GPUs
or local training runs to obtain extra official experimental evidence. This rule
keeps the project fair across students with different hardware access.

You may use local compute for ordinary analysis tasks, such as:

- fitting scaling-law curves
- plotting results
- running notebooks
- processing your own experiment logs
- writing your report

## 5. Data and Evaluation Policy

The training data is prepared by the course staff. It is drawn from public-source
data, but the exact dataset mixture, preprocessing, tokenized order, and final
evaluation split are not disclosed.

All official runs use a fixed tokenized data order. The `model_seed` controls
model initialization, not the data order.

Exploratory runs report validation losses on the official exploratory validation
evaluation. Final leaderboard scoring uses a larger held-out validation
evaluation. This final evaluation uses the same general training/evaluation
setup, but students cannot repeatedly query it during exploration.

Because all students use a strictly controlled deterministic training
infrastructure and data pipeline, validation losses are comparable across your
runs. They are still finite-run observations, so your analysis should account
for failed runs, unstable training, and the fact that short or small-scale runs
may not reveal the same behavior as larger-budget pretraining runs.

## 6. Training Configurations

Each experiment submission contains a training configuration. The API
quickstart configuration reference provides the public schema, organized around
the following groups of parameters.

### Architecture Configuration

You will configure a dense decoder-only Transformer model. Parameters may
include:

- number of Transformer layers
- hidden size
- number of attention heads
- head dimension
- feedforward/intermediate size
- normalization settings
- RoPE settings
- embedding tying
- dtype, where supported
- vocabulary size, where supported

The API quickstart is the authoritative field reference. In the current schema,
`model.head_dim` is a supported model field. If you omit it, the API derives it
from `model.hidden_size / model.num_attention_heads`. If you provide it,
`model.hidden_size` must equal
`model.num_attention_heads * model.head_dim`.

### Optimizer Configuration

The API supports fixed optimizer families and learning-rate schedules. Parameters
may include:

- optimizer type
- peak learning rate
- warmup fraction
- final learning-rate fraction
- weight decay
- beta values, if applicable
- epsilon values, if applicable
- gradient clipping, if applicable

### Training Configuration

Training parameters include:

- training batch size
- validation batch size
- number of evaluations
- total training tokens
- model seed

The maximum runtime reservation is submitted as `requested_runtime_seconds` in
the `POST /submit` request. It is not part of the training configuration object.

The API may reject configurations that violate schema or tensor-dimension rules.
It may also return resource warnings for risky configurations, but these
warnings do not necessarily mean the configuration is invalid.

The configuration controls both the model being trained and the amount of
evidence you obtain from a run. For example, increasing model size may improve
capacity but reduce the number of optimizer steps available within a fixed
runtime. Increasing training tokens may help a model use data more effectively,
but can also make a run too long for the requested reservation. Treat each
configuration as a hypothesis about compute allocation.

## 7. API Workflow

The course staff will provide:

- API base URL
- API key for each student
- API quickstart
- sample notebook, if distributed
- full request/response schema

All API requests should include your API key in the request headers. Your API key
is personal. Do not share it.

The public API will include:

- `GET /budget`
- `POST /submit`
- `GET /experiments`
- `GET /experiment/{experiment_id}`
- `POST /final_submission`
- `GET /final_submission`

The API quickstart provides practical Python examples and sample responses. If a
sample notebook is distributed, it will show the full
submit/poll/analyze/final-submission loop.

### Typical Workflow

1. Check your remaining budget with `GET /budget`.
2. Submit a training configuration with `POST /submit`.
3. Poll experiment status with `GET /experiment/{experiment_id}`.
4. Record validation losses from completed runs.
5. Fit scaling laws using your collected results.
6. Submit your final configuration and predicted loss with
   `POST /final_submission`.
7. Confirm your current final submission with `GET /final_submission`.

## 8. Budget Accounting

Queued and running experiments reserve their full `requested_runtime_seconds`.

When an experiment finishes, the system updates budget usage according to the
final experiment status.

General rules:

- invalid requests are not charged
- duplicate identical submissions by the same student are not charged again
- queued and running jobs reserve their full requested runtime
- completed jobs are charged by actual runtime, up to the requested maximum
- timeout failures are charged as the full reserved runtime
- numerical or resource failures are charged by actual runtime, up to the
  requested maximum
- clear infrastructure failures may be reviewed by course staff

If you reserve more runtime than your experiment needs, unused reserved time is
returned after the run completes.

## 9. Experiment Status

Experiments may have statuses such as:

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`
- `system_failed`

Completed experiments include a list of validation losses. For completed
experiments, the final validation loss is the last entry in that list.

Failed exploratory runs may include partial validation losses. These partial
losses are useful diagnostics, but they are not final validation losses and
should not be treated as completed-run results.

Failure reasons may include timeout, numerical instability, resource failure, or
other runtime failures. The course API will provide structured failure
categories where possible.

## 10. Final Submission

Your final submission contains:

- final training configuration
- predicted final validation loss
- prediction interval lower bound
- prediction interval upper bound

You may resubmit before the deadline. The latest valid submission before the
deadline is the one that will be frozen and used for the final pretraining run.
The API validates final training configurations before storing them. Invalid
configs return `invalid_final_submission` and cannot be frozen.

Your prediction interval should represent your honest uncertainty about the
final validation loss. For example, if your point prediction is `2.73`, you
might submit an interval such as `[2.69, 2.79]` if that reflects your fitted
scaling-law uncertainty and model-selection risk. For scoring, course staff
interpret this interval as an 80% central prediction interval by default.

The API records your prediction. It does not generate or adjust your prediction
for you.

## 11. Deliverables

You must submit:

1. **Final API submission**
   - final training configuration
   - predicted final validation loss
   - prediction interval lower and upper bounds

2. **Methodology report**
   - description of your experimental design
   - list or summary of runs used for fitting
   - scaling-law model or models fitted
   - plots supporting your extrapolation
   - residuals or other fit-quality diagnostics
   - explanation of final model-size/data-size choice
   - explanation of optimizer/hyperparameter choices
   - uncertainty discussion
   - limitations and possible failure modes

3. **Analysis code**
   - code used to collect, clean, fit, and plot your results
   - code should be sufficient to reproduce your reported analysis from your
     API results

The exact submission portal and file format will be announced by course staff.

Your report should make your reasoning auditable. A reader should be able to see
which runs were used, which runs were excluded or downweighted, what model you
fit, and why the final configuration follows from that evidence.

## 12. Grading

Default grading weights:

- **60%** final pretraining-run validation loss
- **20%** prediction quality
- **20%** methodology report and analysis reproducibility

Prediction quality uses an explicit lower-is-better penalty. Let the actual
final validation loss be $L$, your point prediction be $\hat{L}$, and your
prediction interval be $[L_-, L_+]$. A valid prediction must use finite values
and satisfy:

$$
L_- \le \hat{L} \le L_+.
$$

Prediction quality has two components:

- Point-prediction error:

  $$
  E_{\text{point}} = |\hat{L} - L|.
  $$

- Interval penalty:

  $$
  S_{\text{interval}}
  =
  (L_+ - L_-)
  + 10 \max(0, L_- - L)
  + 10 \max(0, L - L_+).
  $$

Here, $(L_+ - L_-)$ is the width penalty. If the actual loss falls outside the
interval, the distance outside the interval is additionally multiplied by 10.
The coefficient 10 comes from the standard interval score for an 80% central
prediction interval, since $2 / 0.2 = 10$.

The raw prediction-quality penalty is:

$$
S_{\text{pred}}
=
0.5 E_{\text{point}}
+
0.5 S_{\text{interval}}.
$$

A wider interval is therefore not automatically better: it may be more likely
to cover the actual result, but it directly increases the width penalty. A
narrow or shifted interval is also risky: once the actual loss falls outside the
interval, the miss distance is penalized heavily. The best strategy is to submit
a point prediction supported by your experiments and a well-calibrated
uncertainty interval that is not artificially inflated.

The final pretraining run is expected to complete successfully. If your final
configuration fails due to timeout, numerical instability, or resource issues,
the final-run validation-loss component receives an invalid score; because there
is no trustworthy actual final loss, the prediction-quality penalty also cannot
be computed automatically. Confirmed platform or infrastructure failures are
reviewed by course staff and rerun or adjudicated separately.

## 13. Rules and Academic Integrity

This project is about experimental design and empirical reasoning. You may use
standard libraries for data analysis, plotting, optimization, and curve fitting.

Unless course staff announce a different policy:

- only official API runs count as official experimental evidence
- do not share API keys
- do not attempt to access another student's experiments
- do not attempt to bypass budget limits
- do not attempt to infer or access the final evaluation split
- do not attack, overload, scrape, or reverse-engineer the course backend
- disclose any external assistance according to the course policy

## 14. Practical Advice

Keep a structured experiment log from the beginning. For every run, record:

- experiment ID
- submitted configuration
- status
- runtime
- validation losses
- notes about why you ran it
- how it affected your next decision

Good scaling-law work is not just curve fitting. It also requires knowing which
runs are trustworthy, which runs were undertrained or unstable, and where your
extrapolation is fragile.

Prefer analyses that separate raw experimental observations from model-selection
decisions. For example, track throughput, runtime usage, failure modes, and
validation losses separately before fitting a scaling curve. When two fitted
laws make similar predictions, explain why you trusted one more than the other
or report the disagreement as part of your uncertainty.

Use failed or partial runs carefully. They can be useful evidence about
stability, memory pressure, or poor hyperparameters, but they should not be
treated as completed validation-loss observations unless the API reports a
completed result.

The winning final model is not necessarily the largest model. Under a fixed
compute budget, a model can be too small to use the data effectively or too
large to train enough. Your job is to find the best tradeoff.
