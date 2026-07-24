from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from scaling_data.io import (
    iter_input_files,
    iter_jsonl_records,
    open_text_writer,
    write_jsonl_record,
)
from scaling_data.text_processing import clean_text, document_hash


_STATS_KEYS = (
    "input_documents",
    "kept_train_documents",
    "kept_validation_documents",
    "dropped_short_documents",
    "dropped_long_documents",
    "dropped_duplicate_documents",
    "trusted_preprocessed_documents",
)


@dataclass(frozen=True)
class _PostprocessTask:
    input_index: int
    input_path: Path
    candidate_path: Path
    min_chars: int
    max_chars: int | None
    validation_rate: float


@dataclass(frozen=True)
class _PostprocessTaskResult:
    input_index: int
    candidate_path: str
    stats: dict[str, int]


def assign_split(text: str, *, validation_rate: float) -> str:
    """Assign a document to train/validation by stable content hash."""
    if not 0 <= validation_rate <= 1:
        raise ValueError("validation_rate must be in [0, 1]")
    if validation_rate == 0:
        return "train"
    if validation_rate == 1:
        return "validation"

    digest = document_hash(clean_text(text))
    bucket = int(digest[:16], 16) / float(16**16 - 1)
    return "validation" if bucket < validation_rate else "train"


def postprocess_directory(
    *,
    input_dir: Path,
    output_dir: Path,
    validation_rate: float,
    min_chars: int,
    max_chars: int | None = None,
    workers: int = 1,
) -> dict:
    worker_count = _normalize_workers(workers, field_name="workers")
    if worker_count <= 1:
        return _postprocess_directory_inline(
            input_dir=input_dir,
            output_dir=output_dir,
            validation_rate=validation_rate,
            min_chars=min_chars,
            max_chars=max_chars,
        )
    return _postprocess_directory_parallel(
        input_dir=input_dir,
        output_dir=output_dir,
        validation_rate=validation_rate,
        min_chars=min_chars,
        max_chars=max_chars,
        workers=worker_count,
    )


def _postprocess_directory_inline(
    *,
    input_dir: Path,
    output_dir: Path,
    validation_rate: float,
    min_chars: int,
    max_chars: int | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl.gz"
    validation_path = output_dir / "validation.jsonl.gz"
    seen_hashes: set[str] = set()
    stats = _empty_stats()

    with open_text_writer(train_path) as train_handle, open_text_writer(
        validation_path
    ) as validation_handle:
        for input_path in iter_input_files(input_dir):
            for record in iter_jsonl_records(input_path):
                stats["input_documents"] += 1
                raw_text = record.get("text")
                if not isinstance(raw_text, str):
                    stats["dropped_short_documents"] += 1
                    continue

                cleaned, digest, trusted_preprocessed = _cleaned_text_and_hash(
                    record,
                    raw_text,
                )
                if trusted_preprocessed:
                    stats["trusted_preprocessed_documents"] += 1
                if len(cleaned) < min_chars:
                    stats["dropped_short_documents"] += 1
                    continue
                if max_chars is not None and len(cleaned) > max_chars:
                    stats["dropped_long_documents"] += 1
                    continue

                if digest in seen_hashes:
                    stats["dropped_duplicate_documents"] += 1
                    continue
                seen_hashes.add(digest)

                split = assign_split_digest(digest, validation_rate=validation_rate)
                out_record = {
                    "text": cleaned,
                    "source_id": record.get("source_id", input_path.stem),
                    "document_hash": digest,
                    "split": split,
                }
                if split == "validation":
                    write_jsonl_record(validation_handle, out_record)
                    stats["kept_validation_documents"] += 1
                else:
                    write_jsonl_record(train_handle, out_record)
                    stats["kept_train_documents"] += 1

    stats_path = output_dir / "postprocess-stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def _postprocess_directory_parallel(
    *,
    input_dir: Path,
    output_dir: Path,
    validation_rate: float,
    min_chars: int,
    max_chars: int | None,
    workers: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = list(iter_input_files(input_dir))
    if len(input_paths) <= 1:
        return _postprocess_directory_inline(
            input_dir=input_dir,
            output_dir=output_dir,
            validation_rate=validation_rate,
            min_chars=min_chars,
            max_chars=max_chars,
        )

    train_path = output_dir / "train.jsonl.gz"
    validation_path = output_dir / "validation.jsonl.gz"
    stats = _empty_stats()
    seen_hashes: set[str] = set()

    with TemporaryDirectory(prefix=".postprocess-", dir=output_dir) as tmpdir:
        tmp_root = Path(tmpdir)
        tasks = [
            _PostprocessTask(
                input_index=index,
                input_path=input_path,
                candidate_path=tmp_root / f"candidates-{index:06d}.jsonl",
                min_chars=min_chars,
                max_chars=max_chars,
                validation_rate=validation_rate,
            )
            for index, input_path in enumerate(input_paths)
        ]
        results_by_index: dict[int, _PostprocessTaskResult] = {}
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = {
                executor.submit(_postprocess_input_file, task): task
                for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                results_by_index[result.input_index] = result
                for key, value in result.stats.items():
                    stats[key] += value

        with open_text_writer(train_path) as train_handle, open_text_writer(
            validation_path
        ) as validation_handle:
            for index in range(len(tasks)):
                result = results_by_index[index]
                for out_record in iter_jsonl_records(Path(result.candidate_path)):
                    digest = str(out_record["document_hash"])
                    if digest in seen_hashes:
                        stats["dropped_duplicate_documents"] += 1
                        continue
                    seen_hashes.add(digest)
                    if out_record.get("split") == "validation":
                        write_jsonl_record(validation_handle, out_record)
                        stats["kept_validation_documents"] += 1
                    else:
                        write_jsonl_record(train_handle, out_record)
                        stats["kept_train_documents"] += 1

    stats_path = output_dir / "postprocess-stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def _postprocess_input_file(task: _PostprocessTask) -> _PostprocessTaskResult:
    stats = _empty_stats()
    task.candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with task.candidate_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl_records(task.input_path):
            stats["input_documents"] += 1
            raw_text = record.get("text")
            if not isinstance(raw_text, str):
                stats["dropped_short_documents"] += 1
                continue

            cleaned, digest, trusted_preprocessed = _cleaned_text_and_hash(
                record,
                raw_text,
            )
            if trusted_preprocessed:
                stats["trusted_preprocessed_documents"] += 1
            if len(cleaned) < task.min_chars:
                stats["dropped_short_documents"] += 1
                continue
            if task.max_chars is not None and len(cleaned) > task.max_chars:
                stats["dropped_long_documents"] += 1
                continue

            split = assign_split_digest(digest, validation_rate=task.validation_rate)
            write_jsonl_record(
                handle,
                {
                    "text": cleaned,
                    "source_id": record.get("source_id", task.input_path.stem),
                    "document_hash": digest,
                    "split": split,
                },
            )

    return _PostprocessTaskResult(
        input_index=task.input_index,
        candidate_path=str(task.candidate_path),
        stats=stats,
    )


def _empty_stats() -> dict[str, int]:
    return {key: 0 for key in _STATS_KEYS}


def _normalize_workers(value: int, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def assign_split_digest(digest: str, *, validation_rate: float) -> str:
    if not 0 <= validation_rate <= 1:
        raise ValueError("validation_rate must be in [0, 1]")
    if validation_rate == 0:
        return "train"
    if validation_rate == 1:
        return "validation"

    bucket = int(digest[:16], 16) / float(16**16 - 1)
    return "validation" if bucket < validation_rate else "train"


def _cleaned_text_and_hash(record: dict, raw_text: str) -> tuple[str, str, bool]:
    digest = record.get("document_hash")
    if record.get("text_preprocessed") is True and _is_sha256_hex(digest):
        return raw_text, str(digest), True
    cleaned = clean_text(raw_text)
    return cleaned, document_hash(cleaned), False


def _is_sha256_hex(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, and split JSONL.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-rate", type=float, default=0.001)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    stats = postprocess_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        validation_rate=args.validation_rate,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        workers=args.workers,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
