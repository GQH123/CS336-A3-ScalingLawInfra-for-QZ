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

default_cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '8')"
case "$default_cpu_count" in
  ''|*[!0-9]*) default_cpu_count=8 ;;
esac
if (( default_cpu_count < 1 )); then
  default_cpu_count=8
fi

export PYTHONPATH="$data_package_root${PYTHONPATH:+:$PYTHONPATH}"

manifest="${SCALING_DATA_MANIFEST:-$data_package_root/manifests/general_100b_mix.json}"
work_dir="${SCALING_DATA_WORK_DIR:-$execution_root/data_work}"
processed_dir="${SCALING_PROCESSED_JSONL_DIR:-$work_dir/processed_jsonl}"
tokenizer="${SCALING_DATA_TOKENIZER:-EleutherAI/gpt-neox-20b}"
target_shard_tokens="${SCALING_TARGET_SHARD_TOKENS:-100000000}"
processed_shuffle_seed="${SCALING_PROCESSED_SHUFFLE_SEED:-20260724}"
processed_shuffle_bucket_count="${SCALING_PROCESSED_SHUFFLE_BUCKET_COUNT:-4096}"
processed_shuffle_max_open_buckets="${SCALING_PROCESSED_SHUFFLE_MAX_OPEN_BUCKETS:-0}"
tokenize_workers="${SCALING_TOKENIZE_WORKERS:-$default_cpu_count}"
tokenize_chunk_records="${SCALING_TOKENIZE_CHUNK_RECORDS:-4096}"
tokenize_max_pending_chunks="${SCALING_TOKENIZE_MAX_PENDING_CHUNKS:-$((tokenize_workers * 4))}"
tokenize_progress_every_chunks="${SCALING_TOKENIZE_PROGRESS_EVERY_CHUNKS:-100}"

if [[ ! -f "$manifest" ]]; then
  echo "Data manifest does not exist: $manifest" >&2
  echo "Set SCALING_DATA_MANIFEST explicitly if the manifest is stored elsewhere." >&2
  exit 2
fi

if [[ ! -d "$processed_dir" ]]; then
  echo "Processed JSONL directory does not exist: $processed_dir" >&2
  echo "Set SCALING_PROCESSED_JSONL_DIR to the existing processed_jsonl directory." >&2
  exit 2
fi

args=(
  --manifest "$manifest"
  --processed-dir "$processed_dir"
  --work-dir "$work_dir"
  --tokenizer "$tokenizer"
  --target-shard-tokens "$target_shard_tokens"
  --shuffle-seed "$processed_shuffle_seed"
  --shuffle-bucket-count "$processed_shuffle_bucket_count"
  --shuffle-max-open-buckets "$processed_shuffle_max_open_buckets"
  --tokenize-workers "$tokenize_workers"
  --tokenize-chunk-records "$tokenize_chunk_records"
  --tokenize-max-pending-chunks "$tokenize_max_pending_chunks"
  --tokenize-progress-every-chunks "$tokenize_progress_every_chunks"
)

if [[ "${SCALING_NO_SHUFFLE_PROCESSED_RECORDS:-0}" == "1" ]]; then
  args+=(--no-shuffle-processed-records)
fi

if [[ "${SCALING_FORCE_RESHUFFLE_PROCESSED_RECORDS:-0}" == "1" ]]; then
  args+=(--force-reshuffle)
fi

if [[ "${SCALING_NO_PROGRESS:-0}" == "1" ]]; then
  args+=(--no-progress)
fi

exec python -m scaling_data.rebuild_tokenized "${args[@]}" "$@"
