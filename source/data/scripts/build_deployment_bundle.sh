#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
data_package_root="$(cd "$script_dir/.." && pwd -P)"
output_path="${1:-$PWD/scaling-laws-data-package.tar.gz}"

mkdir -p "$(dirname "$output_path")"

tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.pytest_cache' \
  --exclude='*.egg-info' \
  -C "$data_package_root/.." \
  -czf "$output_path" \
  "$(basename "$data_package_root")"

printf '%s\n' "$output_path"
