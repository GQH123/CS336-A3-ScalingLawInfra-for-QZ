from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from scaling_data.tokenize import (
    TOKEN_BYTE_ORDER,
    TOKEN_DTYPE,
    TOKEN_SHARD_FORMAT,
    TOKENIZED_INDEX_SCHEMA_VERSION,
    load_tokenized_index,
)


DEFAULT_SOURCE_ID = "lifecycle_test"
DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_TRAIN_TOKENS = 8192
DEFAULT_VALIDATION_TOKENS_PER_EVAL = 4096
DEFAULT_NUM_EVALS = 2


class LifecycleDatasetError(ValueError):
    pass


def package_lifecycle_dataset(
    *,
    source_tokenized_dir: Path,
    output_dir: Path,
    train_shards: int = 1,
    validation_shards: int = 1,
    source_validation_dir: Path | None = None,
    link_mode: str = "hardlink-or-copy",
    overwrite: bool = False,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    train_tokens: int = DEFAULT_TRAIN_TOKENS,
    validation_tokens_per_eval: int = DEFAULT_VALIDATION_TOKENS_PER_EVAL,
    num_evals: int = DEFAULT_NUM_EVALS,
) -> dict[str, Any]:
    source_root = source_tokenized_dir.resolve()
    output_root = output_dir.resolve()
    train_source_dir = _resolve_train_dir(source_root)
    validation_source_dir = _resolve_validation_dir(
        source_root,
        source_validation_dir=source_validation_dir,
    )
    _reject_dangerous_output_dir(
        output_root,
        source_root=source_root,
        train_source_dir=train_source_dir,
        validation_source_dir=validation_source_dir,
    )
    _prepare_output_dir(output_root, overwrite=overwrite)
    train_candidates = _token_shards(train_source_dir)
    if validation_source_dir is None:
        needed = train_shards + validation_shards
        if len(train_candidates) < needed:
            raise LifecycleDatasetError(
                "not enough train shards to carve lifecycle train and validation "
                f"splits: need {needed}, found {len(train_candidates)}"
            )
        selected_train = train_candidates[:train_shards]
        selected_validation = train_candidates[train_shards:needed]
        validation_from_train = True
    else:
        validation_candidates = _token_shards(validation_source_dir)
        if len(train_candidates) < train_shards:
            raise LifecycleDatasetError(
                f"not enough train shards: need {train_shards}, found "
                f"{len(train_candidates)}"
            )
        if len(validation_candidates) < validation_shards:
            raise LifecycleDatasetError(
                f"not enough validation shards: need {validation_shards}, found "
                f"{len(validation_candidates)}"
            )
        selected_train = train_candidates[:train_shards]
        selected_validation = validation_candidates[:validation_shards]
        validation_from_train = False

    tokenized_root = output_root / "tokenized"
    train_index = _write_split(
        selected_train,
        output_dir=tokenized_root / "train",
        link_mode=link_mode,
    )
    validation_index = _write_split(
        selected_validation,
        output_dir=tokenized_root / "validation",
        link_mode=link_mode,
    )

    train_total = int(train_index["total_tokens"])
    validation_total = int(validation_index["total_tokens"])
    resolved_train_tokens = _resolve_token_budget(
        requested=train_tokens,
        available_tokens=train_total,
        sequence_length=sequence_length,
        field_name="train_tokens",
    )
    resolved_validation_tokens = _resolve_token_budget(
        requested=validation_tokens_per_eval,
        available_tokens=validation_total,
        sequence_length=sequence_length,
        field_name="validation_tokens_per_eval",
    )
    resolved_num_evals = _resolve_num_evals(
        train_tokens=resolved_train_tokens,
        sequence_length=sequence_length,
        requested=num_evals,
    )
    overrides = _runtime_overrides(
        train_index_path=tokenized_root / "train" / "index.json",
        validation_index_path=tokenized_root / "validation" / "index.json",
        validation_tokens_per_eval=resolved_validation_tokens,
    )
    student_config = _recommended_student_config(
        sequence_length=sequence_length,
        train_tokens=resolved_train_tokens,
        num_evals=resolved_num_evals,
    )
    manifest = {
        "schema_version": 1,
        "source_tokenized_dir": str(source_root),
        "source_train_dir": str(train_source_dir),
        "source_validation_dir": (
            "" if validation_source_dir is None else str(validation_source_dir)
        ),
        "validation_built_from_train_shards": validation_from_train,
        "output_dir": str(output_root),
        "tokenized_dir": str(tokenized_root),
        "train_index_path": str(tokenized_root / "train" / "index.json"),
        "validation_index_path": str(tokenized_root / "validation" / "index.json"),
        "train_index_uri": overrides["SCALING_TOKENIZED_TRAIN_INDEX_URI"],
        "validation_index_uri": overrides["SCALING_TOKENIZED_VALIDATION_INDEX_URI"],
        "train_total_tokens": train_total,
        "validation_total_tokens": validation_total,
        "selected_train_shards": [str(path) for path in selected_train],
        "selected_validation_shards": [str(path) for path in selected_validation],
        "runtime_overrides": overrides,
        "recommended_student_config": student_config,
        "recommended_submit_payload": {
            "config": student_config,
            "requested_runtime_seconds": 300,
        },
    }
    _write_json(output_root / "lifecycle-dataset-manifest.json", manifest)
    _write_json(output_root / "api-runtime-overrides.json", overrides)
    _write_env(output_root / "api-runtime-overrides.env", overrides)
    _write_readme(output_root / "README.md", manifest)

    load_tokenized_index(tokenized_root / "train" / "index.json", validation_mode="metadata")
    load_tokenized_index(
        tokenized_root / "validation" / "index.json",
        validation_mode="metadata",
    )
    return manifest


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise LifecycleDatasetError(
                f"output directory is not empty: {output_dir}; pass --overwrite "
                "to replace this lifecycle dataset"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _reject_dangerous_output_dir(
    output_dir: Path,
    *,
    source_root: Path,
    train_source_dir: Path,
    validation_source_dir: Path | None,
) -> None:
    protected = [source_root, train_source_dir]
    if validation_source_dir is not None:
        protected.append(validation_source_dir)
    for path in protected:
        if output_dir == path:
            raise LifecycleDatasetError(
                f"output directory must not be the source shard directory: {output_dir}"
            )
        if path.is_relative_to(output_dir):
            raise LifecycleDatasetError(
                "output directory must not contain the source shard directory: "
                f"output={output_dir} source={path}"
            )


def _resolve_train_dir(source_root: Path) -> Path:
    if source_root.name == "train":
        return source_root
    train_dir = source_root / "train"
    if train_dir.exists():
        return train_dir
    return source_root


def _resolve_validation_dir(
    source_root: Path,
    *,
    source_validation_dir: Path | None,
) -> Path | None:
    if source_validation_dir is not None:
        return source_validation_dir.resolve()
    validation_dir = source_root / "validation"
    if validation_dir.exists() and _token_shards(validation_dir):
        return validation_dir
    return None


def _token_shards(directory: Path) -> list[Path]:
    if not directory.exists():
        raise LifecycleDatasetError(f"token shard directory does not exist: {directory}")
    shards = sorted(directory.glob("tokens-*.bin"))
    if not shards:
        raise LifecycleDatasetError(f"no tokens-*.bin shards found in {directory}")
    return shards


def _write_split(
    shards: Sequence[Path],
    *,
    output_dir: Path,
    link_mode: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    shard_entries: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for index, source_path in enumerate(shards):
        target_name = f"tokens-{index:06d}.bin"
        target_path = output_dir / target_name
        _materialize_shard(source_path, target_path, link_mode=link_mode)
        token_count = _token_count_from_file(target_path)
        digest = _sha256_file(target_path)
        total_tokens += token_count
        source_counts[DEFAULT_SOURCE_ID] += token_count
        shard_entries.append(
            {
                "path": target_name,
                "tokens": token_count,
                "dtype": TOKEN_DTYPE,
                "byte_order": TOKEN_BYTE_ORDER,
                "sha256": digest,
                "source_token_counts": {DEFAULT_SOURCE_ID: token_count},
            }
        )
    index_payload = {
        "schema_version": TOKENIZED_INDEX_SCHEMA_VERSION,
        "token_dtype": TOKEN_DTYPE,
        "byte_order": TOKEN_BYTE_ORDER,
        "shard_format": TOKEN_SHARD_FORMAT,
        "total_tokens": total_tokens,
        "source_token_counts": dict(sorted(source_counts.items())),
        "shards": shard_entries,
    }
    _write_json(output_dir / "index.json", index_payload)
    return index_payload


def _materialize_shard(source_path: Path, target_path: Path, *, link_mode: str) -> None:
    if link_mode == "copy":
        shutil.copy2(source_path, target_path)
        return
    if link_mode == "symlink":
        target_path.symlink_to(source_path.resolve())
        return
    if link_mode == "hardlink":
        os.link(source_path, target_path)
        return
    if link_mode == "hardlink-or-copy":
        try:
            os.link(source_path, target_path)
        except OSError:
            shutil.copy2(source_path, target_path)
        return
    raise LifecycleDatasetError(
        "link_mode must be one of: copy, hardlink, hardlink-or-copy, symlink"
    )


def _token_count_from_file(path: Path) -> int:
    size = path.stat().st_size
    if size <= 0 or size % 4 != 0:
        raise LifecycleDatasetError(
            f"token shard size must be a positive multiple of 4 bytes: {path}"
        )
    return size // 4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_token_budget(
    *,
    requested: int,
    available_tokens: int,
    sequence_length: int,
    field_name: str,
) -> int:
    if sequence_length <= 0:
        raise LifecycleDatasetError("sequence_length must be positive")
    if requested <= 0:
        raise LifecycleDatasetError(f"{field_name} must be positive")
    usable = min(int(requested), int(available_tokens) - 1)
    usable -= usable % sequence_length
    if usable <= 0:
        raise LifecycleDatasetError(
            f"{field_name} cannot be satisfied: available_tokens={available_tokens}, "
            f"sequence_length={sequence_length}"
        )
    return usable


def _resolve_num_evals(
    *,
    train_tokens: int,
    sequence_length: int,
    requested: int,
) -> int:
    if requested <= 0:
        raise LifecycleDatasetError("num_evals must be positive")
    total_steps = train_tokens // sequence_length
    if requested > total_steps:
        return requested
    for candidate in range(min(requested, total_steps), 0, -1):
        if total_steps % candidate == 0:
            return candidate
    return 1


def _runtime_overrides(
    *,
    train_index_path: Path,
    validation_index_path: Path,
    validation_tokens_per_eval: int,
) -> dict[str, str]:
    train_uri = train_index_path.resolve().as_uri()
    validation_uri = validation_index_path.resolve().as_uri()
    return {
        "SCALING_DATA_MANIFEST_ID": "lifecycle-train-local",
        "SCALING_EVAL_MANIFEST_ID": "lifecycle-validation-local",
        "SCALING_FINAL_DATA_MANIFEST_ID": "lifecycle-final-train-local",
        "SCALING_FINAL_EVAL_MANIFEST_ID": "lifecycle-final-validation-local",
        "SCALING_VALIDATION_TOKENS_PER_EVAL": str(validation_tokens_per_eval),
        "SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL": str(validation_tokens_per_eval),
        "SCALING_TOKENIZED_TRAIN_INDEX_URI": train_uri,
        "SCALING_TOKENIZED_VALIDATION_INDEX_URI": validation_uri,
        "SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI": train_uri,
        "SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI": validation_uri,
    }


def _recommended_student_config(
    *,
    sequence_length: int,
    train_tokens: int,
    num_evals: int,
) -> dict[str, Any]:
    return {
        "model": {
            "num_hidden_layers": 1,
            "hidden_size": 64,
            "num_attention_heads": 1,
            "dtype": "float32",
        },
        "training": {
            "train_tokens": train_tokens,
            "sequence_length": sequence_length,
            "train_batch_size": 1,
            "validation_batch_size": 1,
            "num_evals": num_evals,
            "learning_rate": 3e-4,
            "optimizer": "adamw",
            "lr_schedule": "constant",
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_env(path: Path, values: dict[str, str]) -> None:
    lines = [f"export {key}={_shell_quote(value)}" for key, value in sorted(values.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _write_readme(path: Path, manifest: dict[str, Any]) -> None:
    overrides = manifest["runtime_overrides"]
    config = manifest["recommended_student_config"]
    text = f"""# Lifecycle Test Tokenized Dataset

This small dataset was packaged from already-written token shards so the API and
training backend can be tested before the full tokenization run finishes.

## Runtime Overrides

Load these values into the control-node API runtime:

```bash
source {path.parent / "api-runtime-overrides.env"}
```

The important index URIs are:

```text
SCALING_TOKENIZED_TRAIN_INDEX_URI={overrides["SCALING_TOKENIZED_TRAIN_INDEX_URI"]}
SCALING_TOKENIZED_VALIDATION_INDEX_URI={overrides["SCALING_TOKENIZED_VALIDATION_INDEX_URI"]}
SCALING_VALIDATION_TOKENS_PER_EVAL={overrides["SCALING_VALIDATION_TOKENS_PER_EVAL"]}
```

## Minimal Student Submit Payload

```json
{json.dumps({"config": config, "requested_runtime_seconds": 300}, indent=2, sort_keys=True)}
```

The lifecycle dataset uses the same validation index for exploratory and final
runs. That is only for backend lifecycle testing, not for final grading.
"""
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Package already-written token shards into a small train/validation "
            "dataset for API and trainer lifecycle testing."
        )
    )
    parser.add_argument("--source-tokenized-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-validation-dir", type=Path, default=None)
    parser.add_argument("--train-shards", type=int, default=1)
    parser.add_argument("--validation-shards", type=int, default=1)
    parser.add_argument(
        "--link-mode",
        choices=["copy", "hardlink", "hardlink-or-copy", "symlink"],
        default="hardlink-or-copy",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument(
        "--validation-tokens-per-eval",
        type=int,
        default=DEFAULT_VALIDATION_TOKENS_PER_EVAL,
    )
    parser.add_argument("--num-evals", type=int, default=DEFAULT_NUM_EVALS)
    args = parser.parse_args(argv)

    manifest = package_lifecycle_dataset(
        source_tokenized_dir=args.source_tokenized_dir,
        output_dir=args.output_dir,
        train_shards=args.train_shards,
        validation_shards=args.validation_shards,
        source_validation_dir=args.source_validation_dir,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
        sequence_length=args.sequence_length,
        train_tokens=args.train_tokens,
        validation_tokens_per_eval=args.validation_tokens_per_eval,
        num_evals=args.num_evals,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
