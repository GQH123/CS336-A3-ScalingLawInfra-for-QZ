# Data Pipeline for the Scaling-Law Assignment

This folder contains the proposed data preparation pipeline for a 100B-token
version-one training reservoir.

The pipeline is intentionally separate from the API backend. It prepares
staff-controlled manifests, blended text shards, post-processed documents, and
tokenized shards that the training backend can consume.

## Scope

Included:

- dataset manifest for a general 100B-token high-quality mixture
- Hugging Face streaming download helper
- deterministic token-budget blending
- text post-processing, exact-document deduplication, and held-out splitting
- tokenizer-based binary shard writer
- single-command orchestration with plan-only mode
- final prepared-data manifest for deployment wiring
- lightweight tests for deterministic logic

Not included:

- a committed 100B-token dataset
- exact private course manifest
- licenses or access approvals for third-party datasets
- near-duplicate MinHash clustering
- PII filtering beyond simple text cleanup hooks

## Suggested 100B Mix

The default manifest uses this public-source target mix:

| Slice | Weight | Target tokens |
|---|---:|---:|
| general web | 0.55 | 55B |
| educational web | 0.20 | 20B |
| long-form diverse text | 0.10 | 10B |
| math/science | 0.07 | 7B |
| code | 0.06 | 6B |
| multilingual | 0.02 | 2B |

The exact source mix and downloaded manifest should remain staff-private.

The default manifest replaces Dolma with a public directly streamable long-form
fallback because Dolma expects files to be downloaded separately and exposed
through its `DATA_DIR` loader path. The code and multilingual slices use gated
staff sources, The Stack dedup and CulturaX, so the build environment must
provide an `HF_TOKEN` that has accepted those repositories' access terms. The
manifest intentionally uses `bigcode/the-stack-dedup` rather than
`bigcode/the-stack-v2` because Stack v2 streams Software Heritage identifiers
and metadata, not file contents. CulturaX is made concrete with the `zh`
language config; change or split this config if the staff manifest should target
other languages.

## Installation

Core tests use only the Python standard library.

For real downloads and tokenization, install optional dependencies:

```bash
python -m pip install datasets transformers zstandard tqdm
```

## Run Full Pipeline

The canonical staff command prepares the complete dataset reservoir:

```bash
export PROJECT_ROOTDIR="/home/qhgao/.workspace/projects/fnlp-summer-training-2026/assignment-3"

mkdir -p /mnt/course-data/scaling-data-build
cd /mnt/course-data/scaling-data-build

PYTHONPATH="$PROJECT_ROOTDIR/source/data" python -m scaling_data.prepare \
  --manifest "$PROJECT_ROOTDIR/source/data/manifests/general_100b_mix.json" \
  --work-dir "$PWD/data_work" \
  --tokenizer EleutherAI/gpt-neox-20b \
  --target-shard-tokens 100000000 \
  --validation-rate 0.001 \
  --min-chars 200
```

For compute-node deployment, do not copy the full assignment checkout. Build a
standalone data-package bundle on the control/development node:

```bash
source/data/scripts/build_deployment_bundle.sh /tmp/scaling-laws-data-package.tar.gz
```

Extract that bundle on the compute node, for example under `/opt/scaling-laws`.
It expands to a `data/` directory containing only the data preparation package,
manifests, scripts, and metadata.

Then run the staff launcher from the mounted storage directory:

```bash
data_package_root="/opt/scaling-laws/data"
export SCALING_DATA_PACKAGE_ROOT="$data_package_root"
export HF_TOKEN="<staff Hugging Face token>"

mkdir -p /mnt/course-data/scaling-data-build
cd /mnt/course-data/scaling-data-build
"$data_package_root/scripts/prepare_full_data.sh"
```

The launcher intentionally does not store the token. It reads `HF_TOKEN` from
the shell, exports `HUGGING_FACE_HUB_TOKEN` for Hugging Face tooling
compatibility, prepends the standalone data package to `PYTHONPATH`, and then
calls `python -m scaling_data.prepare`. `PROJECT_ROOTDIR` is still accepted for
older assignment-root layouts, but it is no longer required. By default, all
generated dataset artifacts are written under the invocation directory as
`./data_work`, not under the source package. Override this only when a specific
storage path is required:

```bash
export SCALING_DATA_WORK_DIR="/mnt/course-data/scaling-data-build/data_work"
```

## Shuffle an Existing Tokenized Corpus

If the expensive tokenization stage has already completed, do not re-run
download, post-processing, or tokenization just to fix source-blocked training
order. Use the tokenized-corpus shuffle patch instead. It reads the existing
`train/index.json` plus `tokens-*.bin` files, copies fixed-size token byte
ranges into a new deterministic source-balanced order, and writes a fresh
standard tokenized `index.json`.

From the data-build working directory:

```bash
data_package_root="/home/qhgao/.workspace/projects/fnlp-summer-training-2026/assignment-3-data-package"
export SCALING_DATA_PACKAGE_ROOT="$data_package_root"

export SCALING_TOKENIZED_INPUT_INDEX="$PWD/data_work/tokenized/train/index.json"
export SCALING_TOKENIZED_OUTPUT_DIR="$PWD/data_work/tokenized_shuffled/train"
export SCALING_TOKENIZED_SHUFFLE_SEED=20260724
export SCALING_TOKENIZED_SHUFFLE_CHUNK_TOKENS=262144
export SCALING_TARGET_SHARD_TOKENS=100000000

"$data_package_root/scripts/shuffle_tokenized_train.sh"
```

The default chunk size is `262144` tokens, which is a multiple of the default
`1024` sequence length. This keeps source mixing reasonably fine-grained without
creating millions of tiny random reads for a 300G-scale tokenized corpus. Use a
smaller multiple of the sequence length only if early-prefix mixture quality is
more important than shuffle throughput.

The existing index records per-shard source token counts, not per-token source
labels. For the current blockwise build, ordinary shards are single-source and
boundary shards can be split deterministically from their recorded counts. If
you want to make boundary-shard splitting explicit, provide the current raw
source order:

```bash
export SCALING_TOKENIZED_SHUFFLE_SOURCE_ORDER="code,education,general_web,long_form,math,multilingual"
```

The validation loader also reads a contiguous prefix of its tokenized index, so
shuffle the validation split as well if it was produced by the same blockwise
pipeline:

```bash
export SCALING_TOKENIZED_INPUT_INDEX="$PWD/data_work/tokenized/validation/index.json"
export SCALING_TOKENIZED_OUTPUT_DIR="$PWD/data_work/tokenized_shuffled/validation"
export SCALING_TOKENIZED_SHUFFLE_SEED=20260725

"$data_package_root/scripts/shuffle_tokenized_train.sh"
```

After the shuffle finishes, point the control/API runtime at the shuffled train
and validation indexes:

```bash
export SCALING_TOKENIZED_TRAIN_INDEX_URI="file://$PWD/data_work/tokenized_shuffled/train/index.json"
export SCALING_TOKENIZED_VALIDATION_INDEX_URI="file://$PWD/data_work/tokenized_shuffled/validation/index.json"
```

The shuffle command does not require `HF_TOKEN`, `datasets`, `transformers`, or
access to the original JSONL files. It streams token bytes from existing shards
and recomputes output shard hashes.

The full download can legitimately spend a long time after Hugging Face prints
`Resolving data files`. The launcher prints per-source progress on stderr while
streaming records. By default it refreshes every 1,000 scanned documents. Tune
this for more frequent output:

```bash
export SCALING_PROGRESS_EVERY_DOCS=1000
```

By default, the launcher processes logical manifest sources one by one. Within
the current logical source, it fans out several download shards so all workers
focus on the same upstream corpus before moving to the next corpus:

```bash
export SCALING_DOWNLOAD_WORKERS=4
export SCALING_DOWNLOAD_SHARDS_PER_SOURCE=4
export SCALING_DOWNLOAD_SOURCE_PARALLELISM=1
```

Each shard keeps its own raw output and resume checkpoint, for example
`raw_jsonl/general_web.part000.jsonl.gz`. Records inside those files still carry
the logical `source_id`, such as `general_web`, so later tokenization and blend
accounting do not treat shard suffixes as separate corpora. Increase
`SCALING_DOWNLOAD_WORKERS`, `SCALING_DOWNLOAD_SHARDS_PER_SOURCE`, or
`SCALING_DOWNLOAD_SOURCE_PARALLELISM` only when the storage mount, Hugging Face
endpoint, and local CPU decompression can sustain the extra streams. A source
can override the default shard count in the manifest with `download_shards`.

`SCALING_DOWNLOAD_WORKERS` is still the total concurrent download-task pool. The
effective number of simultaneously active logical sources is bounded by both
`SCALING_DOWNLOAD_SOURCE_PARALLELISM` and `SCALING_DOWNLOAD_WORKERS`. When
source parallelism is greater than one, shard tasks are submitted round-robin
across the active sources so each source gets a chance to resolve and start
streaming early. For example, 18 workers, six shards per source, and source
parallelism of six can plan up to 36 source-shard tasks for the active source
set, but only 18 can run at once. The remaining planned shard state files should
show `queued` until a worker is available.

The download stage uses a process pool. Each running shard streams from Hugging
Face or a local file, writes its resume JSONL, and performs the final gzip
replacement in that same worker process. The parent process keeps the scheduler
filled, records queued/running/completed state, and aggregates failures after
the scheduled shard work finishes.

Post-processing and tokenization also use process pools by default in the full
launcher. `SCALING_POSTPROCESS_WORKERS` and `SCALING_TOKENIZE_WORKERS` default
to `SCALING_DOWNLOAD_WORKERS` when unset. Post-processing workers clean and hash
per-input-file candidates; the parent process still performs the final
deterministic global dedupe merge. Tokenization workers encode chunks of
processed records; the parent process still enforces per-source token caps and
writes the final shard indexes in stable order. Tune tokenization task size with
`SCALING_TOKENIZE_CHUNK_RECORDS` if worker dispatch overhead or memory use needs
adjustment.

For a quick source-health smoke test, use a small per-source document cap, one
download shard per source, and several active sources:

```bash
export SCALING_MAX_DOCS_PER_SOURCE=100
export SCALING_DOWNLOAD_WORKERS=6
export SCALING_DOWNLOAD_SHARDS_PER_SOURCE=1
export SCALING_DOWNLOAD_SOURCE_PARALLELISM=6
"$data_package_root/scripts/prepare_full_data.sh"
```

For the sustained full build, keep source-level parallelism at `1` unless the
remote endpoints and local storage can absorb multiple corpora at once. Usually
it is better to scale `SCALING_DOWNLOAD_SHARDS_PER_SOURCE` first so bandwidth is
spent on one corpus before moving to the next.

By default, the downloader streams each enabled upstream source until
`SCALING_MAX_DOCS_PER_SOURCE` or source exhaustion. An optional early-stop guard
can use the blend plan's per-source `target_tokens` to set an approximate raw
download budget:

```text
estimated download tokens = target_tokens * SCALING_DOWNLOAD_TOKEN_OVERFETCH_RATIO
```

`SCALING_DOWNLOAD_CHARS_PER_TOKEN` controls the lightweight estimator used while
streaming text. Exact source caps are still enforced later during tokenization,
so this early stop is only a speed and storage guard when explicitly enabled:

```bash
export SCALING_DOWNLOAD_TOKEN_OVERFETCH_RATIO=0
export SCALING_DOWNLOAD_CHARS_PER_TOKEN=4.0
```

Set `SCALING_DOWNLOAD_TOKEN_OVERFETCH_RATIO` to a positive value such as `1.15`
only when staff intentionally want approximate per-source early stopping before
tokenization.

Downloads are resumable by default. While a source is streaming, the downloader
writes an append-only checkpoint under `data_work/raw_jsonl/.resume/` and records
progress in a sidecar state file. If the process is interrupted, rerun the same
command from the same working directory. The next run will skip only the
checkpointed safe source prefix when supported, deduplicate by document hash,
and continue writing only missing records. A completed source is compressed
atomically to `data_work/raw_jsonl/<source_id>.jsonl.gz` and skipped on later
runs. If the source location changes, or if the current document/token cap is
not compatible with the completed checkpoint, the downloader rebuilds that
source instead of reusing stale raw data.

Resume state status values are operational signals:

- `queued`: the shard was planned but has not yet been assigned to a worker.
- `in_progress`: the shard is streaming or writing its resume JSONL.
- `interrupted`: the shard raised an exception or the process was stopped before
  completion.
- `completed`: the final `.jsonl.gz` was written and the shard can be reused.

During finalization, a hidden file such as
`data_work/raw_jsonl/.general_web.part000.jsonl.gz.tmp.gz` can exist while gzip
replacement is still in progress. The visible
`data_work/raw_jsonl/general_web.part000.jsonl.gz` appears only after the atomic
replace completes.

`scanned_documents` is progress telemetry. The actual resume cursor is
`resume_safe_scanned_documents`, which advances only after a streamed row has
been fully handled by skipping, filtering, deduplication, or writing. Legacy
state files without that safe cursor replay conservatively from the beginning
instead of trusting an unsafe scanned-row count.

Interrupted state files include `error_type` and `error_message` for new runs.
If a state file shows `scanned_documents: 0`, the failure happened before rows
started streaming, usually during Hugging Face dataset resolution, gated access,
missing config selection, or dataset-specific loader setup. After changing the
manifest source path/config, rerun the launcher; the source-location check will
discard stale checkpoints for that source.

If an individual shard fails while other shards remain available, the download
scheduler keeps starting pending shard tasks. After all scheduled shard work has
finished or interrupted, the stage raises one aggregate error listing the failed
download IDs.

If a state file reports that a configured `text_field` is missing from streamed
rows, the selected dataset is not exposing text under the field named in the
manifest. This is expected for `bigcode/the-stack-v2`: it streams Software
Heritage identifiers and metadata rather than file bodies. Use a content-bearing
source such as `bigcode/the-stack-dedup`, or add a separate Software Heritage
content retrieval layer before using Stack v2 directly.

Only disable resume when you intentionally want to discard checkpoints and
rebuild raw source files from scratch:

```bash
export SCALING_NO_RESUME_DOWNLOADS=1
```

Before launching the expensive full build, run a small end-to-end smoke build:

```bash
export SCALING_MAX_DOCS_PER_SOURCE=1000
"$data_package_root/scripts/prepare_full_data.sh"
```

Unset `SCALING_MAX_DOCS_PER_SOURCE` for the real reservoir build. Set
`SCALING_NO_PROGRESS=1` only when a batch runner cannot handle stderr progress
bars.

If the package is installed, use the equivalent console entrypoint:

```bash
scaling-prepare-data \
  --manifest "$PROJECT_ROOTDIR/source/data/manifests/general_100b_mix.json" \
  --work-dir "$PWD/data_work" \
  --tokenizer EleutherAI/gpt-neox-20b \
  --target-shard-tokens 100000000 \
  --validation-rate 0.001 \
  --min-chars 200 \
  --download-workers 4 \
  --download-shards-per-source 4 \
  --download-source-parallelism 1 \
  --download-token-overfetch-ratio 0 \
  --progress-every-docs 1000
```

The command runs the necessary stages in order:

1. build the blend allocation and pipeline plan
2. stream/download enabled sources into `raw_jsonl`
3. clean, exact-deduplicate, and split train/validation JSONL
4. tokenize into split-specific binary shards
5. enforce per-source training-token targets from the blend plan
6. validate tokenized indexes in metadata mode by default
7. write `data_work/prepared-data-manifest.json`

The compatibility script remains available for older notes and runbooks:

```bash
PYTHONPATH=source/data python source/data/scripts/prepare_100b_data.py \
  --manifest source/data/manifests/general_100b_mix.json \
  --work-dir data_work
```

## Plans and Runtime Artifacts

The generated plan files under `data_work/plans/` are staff-facing audit and
reproducibility artifacts. They are not loaded by the training backend. Their
job is to record the normalized source mix, token allocation, stage order, and
expected output paths before or during a data build.

The runtime handoff artifact is `data_work/prepared-data-manifest.json`. It
records the source manifest, the written plan paths, post-processing stats,
per-source training-token limits, tokenized index metadata, and API runtime
overrides. The training backend should consume the tokenized `index.json` files,
usually through the URIs recorded in this manifest:

```json
{
  "api_runtime_overrides": {
    "SCALING_TOKENIZED_TRAIN_INDEX_URI": "file:///storage/data_work/tokenized/train/index.json",
    "SCALING_TOKENIZED_VALIDATION_INDEX_URI": "file:///storage/data_work/tokenized/validation/index.json"
  }
}
```

Use plan-only mode when checking a manifest or storage path before launching
the expensive download/tokenization build:

```bash
PYTHONPATH=source/data python -m scaling_data.prepare \
  --manifest source/data/manifests/general_100b_mix.json \
  --work-dir data_work \
  --plan-only
```

`--dry-run` is kept as a deprecated alias for `--plan-only`.

## Build a Lifecycle-Test Dataset from Partial Tokenization

If full tokenization is still running but some `tokens-*.bin` shards already
exist, package a small train/validation dataset for API and trainer lifecycle
testing:

```bash
PYTHONPATH="$SCALING_DATA_PACKAGE_ROOT" python -m scaling_data.lifecycle_dataset \
  --source-tokenized-dir /path/to/data_work/tokenized \
  --output-dir /path/to/data_work/lifecycle_test_dataset \
  --train-shards 2 \
  --validation-shards 1 \
  --sequence-length 128 \
  --train-tokens 8192 \
  --validation-tokens-per-eval 4096 \
  --num-evals 2
```

When `/path/to/data_work/tokenized/validation` is not available yet, the command
uses the first `--train-shards` train shards for train and the next
`--validation-shards` train shards for validation. This is only for lifecycle
testing; it is not a final evaluation split.

The command writes:

```text
lifecycle_test_dataset/tokenized/train/index.json
lifecycle_test_dataset/tokenized/validation/index.json
lifecycle_test_dataset/api-runtime-overrides.env
lifecycle_test_dataset/api-runtime-overrides.json
lifecycle_test_dataset/lifecycle-dataset-manifest.json
lifecycle_test_dataset/README.md
```

Load the generated runtime overrides into the control-node API process before
submitting lifecycle test jobs:

```bash
source /path/to/data_work/lifecycle_test_dataset/api-runtime-overrides.env
```

The generated overrides set both exploratory and final tokenized index URIs to
the lifecycle dataset:

```text
SCALING_TOKENIZED_TRAIN_INDEX_URI
SCALING_TOKENIZED_VALIDATION_INDEX_URI
SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI
SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI
SCALING_VALIDATION_TOKENS_PER_EVAL
SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL
```

The default `--link-mode hardlink-or-copy` hardlinks shards when possible and
falls back to copying across filesystems. Use `--link-mode copy` if the output
must be independent of the source shard files.

The lower-level plan writers are still useful for debugging a single stage:

```bash
PYTHONPATH=source/data python -m scaling_data.blend \
  --manifest source/data/manifests/general_100b_mix.json \
  --output-dir data_work/blend_plan \
  --dry-run
```

```bash
PYTHONPATH=source/data python -m scaling_data.pipeline \
  --manifest source/data/manifests/general_100b_mix.json \
  --output-dir data_work/plans \
  --work-dir data_work
```

## Download Small Samples

Download a small sample from enabled manifest sources:

```bash
PYTHONPATH=source/data python -m scaling_data.download \
  --manifest source/data/manifests/general_100b_mix.json \
  --output-dir data_work/raw_jsonl \
  --max-docs-per-source 1000 \
  --download-workers 4 \
  --download-shards-per-source 4 \
  --download-source-parallelism 1
```

For a real 100B-token build, remove the sample cap and run on staff-controlled
storage. Add `--download-token-overfetch-ratio 1.15` only when staff explicitly
want the approximate per-source token-budget early stop.

For an offline rehearsal, a manifest source may use `local_path` instead of
`hf_path`. The file must be JSONL/JSONL.GZ/JSONL.ZST and can use the same
`text_field` setting as remote sources:

```json
{
  "id": "local_sample",
  "weight": 1.0,
  "local_path": "/secure/course/samples/local_sample.jsonl",
  "text_field": "text",
  "enabled": true
}
```

This makes the download/postprocess/tokenize path testable without network
access or Hugging Face dependencies before the full staff-controlled build.

## Post-Process and Split

Clean, deduplicate, and split held-out validation documents:

```bash
PYTHONPATH=source/data python -m scaling_data.postprocess \
  --input-dir data_work/raw_jsonl \
  --output-dir data_work/processed_jsonl \
  --validation-rate 0.001 \
  --min-chars 200 \
  --workers 4
```

## Tokenize

Tokenize processed JSONL shards:

```bash
PYTHONPATH=source/data python -m scaling_data.tokenize \
  --input-dir data_work/processed_jsonl \
  --output-dir data_work/tokenized \
  --tokenizer EleutherAI/gpt-neox-20b \
  --target-shard-tokens 100000000 \
  --workers 4
```

Use the same tokenizer as the model training backend.

The tokenizer writes split-specific shard directories when processed records
carry a `split` field. The normal post-processing path therefore produces:

```text
data_work/tokenized/train/index.json
data_work/tokenized/train/tokens-000000.bin
data_work/tokenized/validation/index.json
data_work/tokenized/validation/tokens-000000.bin
```

Each `index.json` records total tokens, per-source token accounting, per-shard
token counts, `uint32` dtype, little-endian byte order, shard format, and
SHA-256 hashes for every shard. These hashes and source counts are the staff
audit trail for verifying deterministic shard contents and mixture accounting
before the training backend consumes the reservoir.

The tokenized index schema is versioned:

```json
{
  "schema_version": 1,
  "token_dtype": "uint32",
  "byte_order": "little",
  "shard_format": "flat_binary_uint32_le",
  "total_tokens": 3,
  "source_token_counts": {"sample": 3},
  "shards": [
    {
      "path": "tokens-000000.bin",
      "tokens": 3,
      "dtype": "uint32",
      "byte_order": "little",
      "sha256": "64 lowercase hex characters",
      "source_token_counts": {"sample": 3}
    }
  ]
}
```

Training code should load tokenized data through the index rather than guessing
file names. `scaling_data.tokenize.load_tokenized_index(...,
validation_mode="metadata")` validates the index schema, safe relative shard
paths, token counts, local file sizes, and SHA-256 metadata format without
reading full shards. `validation_mode="full"` additionally hashes every shard
and is the staff audit mode. `scaling_data.tokenize.read_token_shard(...)` reads
a declared little-endian `uint32` shard back into token IDs. This keeps the
data-consuming runtime coupled to the audited index format, not to incidental
directory contents.

## Run Tests

```bash
PYTHONPATH=source/data python -m unittest discover -s source/data/tests -v
```

## Staff Notes

- Verify every dataset license and access condition before full download.
- Keep the final staff manifest private.
- Hold out validation documents before final tokenization.
- Remove exact and near-duplicate documents between train and validation.
- Log source IDs, processing code version, tokenizer version, and shard hashes.
