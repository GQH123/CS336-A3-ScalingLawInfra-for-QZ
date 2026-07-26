#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
default_project_rootdir="/home/qhgao/.workspace/projects/fnlp-summer-training-2026/assignment-3"
project_rootdir="${PROJECT_ROOTDIR:-$default_project_rootdir}"
execution_root="$(pwd -P)"

if [[ -n "${SCALING_DATA_PACKAGE_ROOT:-}" ]]; then
  data_package_root="$SCALING_DATA_PACKAGE_ROOT"
elif [[ -d "$project_rootdir/source/data/scaling_data" ]]; then
  data_package_root="$project_rootdir/source/data"
else
  data_package_root="$(cd "$script_dir/.." && pwd -P)"
fi

if [[ ! -d "$data_package_root/scaling_data" ]]; then
  echo "Could not find scaling_data package under: $data_package_root" >&2
  echo "Set SCALING_DATA_PACKAGE_ROOT to a copied source/data directory." >&2
  echo "PROJECT_ROOTDIR is optional and only needed for assignment-root layouts." >&2
  exit 2
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for Hugging Face-backed sources." >&2
  echo "Export HF_TOKEN in the shell before running this script." >&2
  exit 2
fi

export PYTHONPATH="$data_package_root${PYTHONPATH:+:$PYTHONPATH}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

default_cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '8')"
case "$default_cpu_count" in
  ''|*[!0-9]*) default_cpu_count=8 ;;
esac
if (( default_cpu_count < 1 )); then
  default_cpu_count=8
fi

manifest="${SCALING_DATA_MANIFEST:-$data_package_root/manifests/general_100b_mix.json}"
work_dir="${SCALING_DATA_WORK_DIR:-$execution_root/data_work}"
tokenizer="${SCALING_DATA_TOKENIZER:-EleutherAI/gpt-neox-20b}"
target_shard_tokens="${SCALING_TARGET_SHARD_TOKENS:-100000000}"
validation_rate="${SCALING_VALIDATION_RATE:-0.001}"
min_chars="${SCALING_MIN_CHARS:-200}"
progress_every_docs="${SCALING_PROGRESS_EVERY_DOCS:-1000}"
download_token_overfetch_ratio="${SCALING_DOWNLOAD_TOKEN_OVERFETCH_RATIO:-0}"
download_chars_per_token="${SCALING_DOWNLOAD_CHARS_PER_TOKEN:-4.0}"
download_workers="${SCALING_DOWNLOAD_WORKERS:-4}"
download_shards_per_source="${SCALING_DOWNLOAD_SHARDS_PER_SOURCE:-4}"
download_source_parallelism="${SCALING_DOWNLOAD_SOURCE_PARALLELISM:-1}"
postprocess_workers="${SCALING_POSTPROCESS_WORKERS:-$download_workers}"
tokenize_workers="${SCALING_TOKENIZE_WORKERS:-$default_cpu_count}"
tokenize_chunk_records="${SCALING_TOKENIZE_CHUNK_RECORDS:-4096}"
tokenize_max_pending_chunks="${SCALING_TOKENIZE_MAX_PENDING_CHUNKS:-$((tokenize_workers * 4))}"
tokenize_progress_every_chunks="${SCALING_TOKENIZE_PROGRESS_EVERY_CHUNKS:-100}"
processed_shuffle_seed="${SCALING_PROCESSED_SHUFFLE_SEED:-20260724}"
processed_shuffle_bucket_count="${SCALING_PROCESSED_SHUFFLE_BUCKET_COUNT:-4096}"
processed_shuffle_max_open_buckets="${SCALING_PROCESSED_SHUFFLE_MAX_OPEN_BUCKETS:-0}"

if [[ ! -f "$manifest" ]]; then
  echo "Data manifest does not exist: $manifest" >&2
  echo "Set SCALING_DATA_MANIFEST explicitly if the manifest is stored elsewhere." >&2
  exit 2
fi

extra_args=(
  --progress-every-docs "$progress_every_docs"
  --download-token-overfetch-ratio "$download_token_overfetch_ratio"
  --download-chars-per-token "$download_chars_per_token"
  --download-workers "$download_workers"
  --download-shards-per-source "$download_shards_per_source"
  --download-source-parallelism "$download_source_parallelism"
  --postprocess-workers "$postprocess_workers"
  --tokenize-workers "$tokenize_workers"
  --tokenize-chunk-records "$tokenize_chunk_records"
  --tokenize-max-pending-chunks "$tokenize_max_pending_chunks"
  --tokenize-progress-every-chunks "$tokenize_progress_every_chunks"
  --processed-shuffle-seed "$processed_shuffle_seed"
  --processed-shuffle-bucket-count "$processed_shuffle_bucket_count"
  --processed-shuffle-max-open-buckets "$processed_shuffle_max_open_buckets"
)

if [[ -n "${SCALING_MAX_DOCS_PER_SOURCE:-}" ]]; then
  extra_args+=(--max-docs-per-source "$SCALING_MAX_DOCS_PER_SOURCE")
fi

if [[ "${SCALING_NO_PROGRESS:-0}" == "1" ]]; then
  extra_args+=(--no-progress)
fi

if [[ "${SCALING_NO_RESUME_DOWNLOADS:-0}" == "1" ]]; then
  extra_args+=(--no-resume-downloads)
fi

if [[ "${SCALING_NO_SHUFFLE_PROCESSED_RECORDS:-0}" == "1" ]]; then
  extra_args+=(--no-shuffle-processed-records)
fi

exec python -m scaling_data.prepare \
  --manifest "$manifest" \
  --work-dir "$work_dir" \
  --tokenizer "$tokenizer" \
  --target-shard-tokens "$target_shard_tokens" \
  --validation-rate "$validation_rate" \
  --min-chars "$min_chars" \
  "${extra_args[@]}" \
  "$@"
