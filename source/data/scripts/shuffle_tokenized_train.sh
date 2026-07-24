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

export PYTHONPATH="$data_package_root${PYTHONPATH:+:$PYTHONPATH}"

input_index="${SCALING_TOKENIZED_INPUT_INDEX:-$execution_root/data_work/tokenized/train/index.json}"
output_dir="${SCALING_TOKENIZED_OUTPUT_DIR:-${SCALING_SHUFFLED_TOKENIZED_TRAIN_DIR:-$execution_root/data_work/tokenized_shuffled/train}}"
seed="${SCALING_TOKENIZED_SHUFFLE_SEED:-20260724}"
chunk_tokens="${SCALING_TOKENIZED_SHUFFLE_CHUNK_TOKENS:-262144}"
target_shard_tokens="${SCALING_TARGET_SHARD_TOKENS:-100000000}"
validation_mode="${SCALING_TOKENIZED_SHUFFLE_VALIDATION_MODE:-metadata}"
max_open_shards="${SCALING_TOKENIZED_SHUFFLE_MAX_OPEN_SHARDS:-32}"

if [[ ! -f "$input_index" ]]; then
  echo "Tokenized index does not exist: $input_index" >&2
  echo "Set SCALING_TOKENIZED_INPUT_INDEX to the existing split index.json." >&2
  exit 2
fi

args=(
  --input-index "$input_index"
  --output-dir "$output_dir"
  --seed "$seed"
  --chunk-tokens "$chunk_tokens"
  --target-shard-tokens "$target_shard_tokens"
  --validation-mode "$validation_mode"
  --max-open-shards "$max_open_shards"
)

if [[ -n "${SCALING_TOKENIZED_SHUFFLE_SOURCE_ORDER:-}" ]]; then
  args+=(--source-order "$SCALING_TOKENIZED_SHUFFLE_SOURCE_ORDER")
fi

if [[ "${SCALING_TOKENIZED_SHUFFLE_OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

exec python -m scaling_data.shuffle_tokenized "${args[@]}" "$@"
