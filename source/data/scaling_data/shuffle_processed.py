from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO

from scaling_data.io import iter_jsonl_records, open_text_writer, write_jsonl_record


DEFAULT_PROCESSED_SHUFFLE_BUCKETS = 4096
DEFAULT_MAX_OPEN_BUCKETS = 64
DEFAULT_PROCESSED_SPLITS = ("train", "validation")


class ProcessedShuffleError(ValueError):
    pass


class _BucketWriterCache:
    def __init__(self, *, bucket_dir: Path, max_open_buckets: int):
        if max_open_buckets <= 0:
            raise ValueError("max_open_buckets must be positive")
        self.bucket_dir = bucket_dir
        self.max_open_buckets = max_open_buckets
        self._handles: OrderedDict[int, TextIO] = OrderedDict()

    def write(self, bucket_index: int, record: Mapping[str, object]) -> None:
        handle = self._handle_for(bucket_index)
        write_jsonl_record(handle, dict(record))

    def close(self) -> None:
        while self._handles:
            _, handle = self._handles.popitem()
            handle.close()

    def _handle_for(self, bucket_index: int) -> TextIO:
        try:
            handle = self._handles.pop(bucket_index)
        except KeyError:
            handle = (self.bucket_dir / _bucket_filename(bucket_index)).open(
                "a",
                encoding="utf-8",
            )
        self._handles[bucket_index] = handle
        while len(self._handles) > self.max_open_buckets:
            _, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        return handle


def shuffle_processed_splits(
    *,
    input_dir: Path,
    output_dir: Path,
    seed: int,
    splits: Sequence[str] = DEFAULT_PROCESSED_SPLITS,
    bucket_count: int = DEFAULT_PROCESSED_SHUFFLE_BUCKETS,
    max_open_buckets: int = DEFAULT_MAX_OPEN_BUCKETS,
    overwrite: bool = False,
) -> dict:
    """Shuffle processed JSONL records independently inside each split.

    This is an external bucket shuffle for already post-processed records such
    as ``processed_jsonl/train.jsonl.gz`` and
    ``processed_jsonl/validation.jsonl.gz``. It moves whole JSONL records only;
    tokenization happens after this step.
    """

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ProcessedShuffleError("output_dir must be different from input_dir")
    normalized_splits = _normalize_splits(splits)
    normalized_bucket_count = _positive_int(bucket_count, "bucket_count")
    normalized_max_open_buckets = _positive_int(
        max_open_buckets,
        "max_open_buckets",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_output_dir(
        output_dir=output_dir,
        splits=normalized_splits,
        overwrite=overwrite,
    )

    stats = {
        "schema_version": 1,
        "algorithm": "processed_jsonl_record_bucket_shuffle_v1",
        "seed": int(seed),
        "bucket_count": normalized_bucket_count,
        "max_open_buckets": normalized_max_open_buckets,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": {},
    }
    with TemporaryDirectory(prefix=".processed-shuffle-", dir=output_dir) as tmpdir:
        tmp_root = Path(tmpdir)
        for split in normalized_splits:
            input_path = _split_input_path(input_dir, split)
            if input_path is None:
                continue
            split_stats = _shuffle_one_split(
                input_path=input_path,
                output_path=output_dir / f"{split}.jsonl.gz",
                tmp_root=tmp_root / split,
                split=split,
                seed=int(seed),
                bucket_count=normalized_bucket_count,
                max_open_buckets=normalized_max_open_buckets,
            )
            stats["splits"][split] = split_stats

    (output_dir / "shuffle-stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stats


def _shuffle_one_split(
    *,
    input_path: Path,
    output_path: Path,
    tmp_root: Path,
    split: str,
    seed: int,
    bucket_count: int,
    max_open_buckets: int,
) -> dict:
    tmp_root.mkdir(parents=True, exist_ok=True)
    source_document_counts: Counter[str] = Counter()
    bucket_record_counts: Counter[int] = Counter()
    writer_cache = _BucketWriterCache(
        bucket_dir=tmp_root,
        max_open_buckets=max_open_buckets,
    )
    document_count = 0
    try:
        for ordinal, record in enumerate(iter_jsonl_records(input_path)):
            key = _record_shuffle_key(
                seed=seed,
                split=split,
                record=record,
                ordinal=ordinal,
            )
            bucket_index = int(key[:16], 16) % bucket_count
            writer_cache.write(
                bucket_index,
                {
                    "key": key,
                    "record": record,
                },
            )
            document_count += 1
            bucket_record_counts[bucket_index] += 1
            source_document_counts[_normalize_source_id(record.get("source_id"))] += 1
    finally:
        writer_cache.close()

    bucket_order = sorted(
        bucket_record_counts,
        key=lambda bucket_index: _bucket_shuffle_key(
            seed=seed,
            split=split,
            bucket_index=bucket_index,
        ),
    )
    with open_text_writer(output_path) as output_handle:
        for bucket_index in bucket_order:
            bucket_path = tmp_root / _bucket_filename(bucket_index)
            wrappers = list(iter_jsonl_records(bucket_path))
            wrappers.sort(key=lambda wrapper: str(wrapper["key"]))
            for wrapper in wrappers:
                record = wrapper.get("record")
                if not isinstance(record, dict):
                    raise ProcessedShuffleError(
                        f"malformed shuffle bucket record in {bucket_path}"
                    )
                write_jsonl_record(output_handle, record)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "documents": document_count,
        "source_document_counts": dict(sorted(source_document_counts.items())),
        "nonempty_buckets": len(bucket_record_counts),
        "max_bucket_records": max(bucket_record_counts.values(), default=0),
    }


def _prepare_output_dir(
    *,
    output_dir: Path,
    splits: Sequence[str],
    overwrite: bool,
) -> None:
    existing = list(output_dir.iterdir())
    if not existing:
        return
    allowed_names = {"shuffle-stats.json"}
    allowed_names.update(f"{split}.jsonl.gz" for split in splits)
    removable = []
    for path in existing:
        if path.is_file() and path.name in allowed_names:
            removable.append(path)
            continue
        if path.is_dir() and path.name.startswith(".processed-shuffle-"):
            raise ProcessedShuffleError(
                f"found stale temporary shuffle directory: {path}"
            )
        raise FileExistsError(
            f"output_dir contains non-shuffle artifact {path}; refusing to delete it"
        )
    if not overwrite:
        raise FileExistsError(
            f"output_dir is not empty: {output_dir}; pass overwrite=True or "
            "--overwrite to replace previous shuffled processed records"
        )
    for path in removable:
        path.unlink()


def _split_input_path(input_dir: Path, split: str) -> Path | None:
    candidates = [
        input_dir / f"{split}.jsonl.gz",
        input_dir / f"{split}.jsonl.zst",
        input_dir / f"{split}.jsonl",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    if len(existing) > 1:
        raise ProcessedShuffleError(
            f"multiple input files found for split {split!r}: "
            + ", ".join(str(path) for path in existing)
        )
    return existing[0]


def _record_shuffle_key(
    *,
    seed: int,
    split: str,
    record: Mapping[str, object],
    ordinal: int,
) -> str:
    document_hash = record.get("document_hash")
    if not isinstance(document_hash, str) or not document_hash.strip():
        document_hash = json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    payload = "\0".join(
        (
            str(seed),
            split,
            str(ordinal),
            document_hash,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bucket_shuffle_key(*, seed: int, split: str, bucket_index: int) -> str:
    payload = f"{seed}\0{split}\0bucket\0{bucket_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bucket_filename(bucket_index: int) -> str:
    return f"bucket-{bucket_index:06d}.jsonl"


def _normalize_splits(splits: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_split in splits:
        split = str(raw_split).strip()
        if not split:
            raise ValueError("split names must be non-empty")
        if split not in seen:
            normalized.append(split)
            seen.add(split)
    if not normalized:
        raise ValueError("at least one split is required")
    return tuple(normalized)


def _normalize_source_id(value) -> str:
    source_id = str(value).strip() if value is not None else ""
    return source_id or "unknown"


def _positive_int(value, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _parse_splits(value: str) -> tuple[str, ...]:
    return _normalize_splits(item for item in value.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Shuffle post-processed JSONL records independently within each "
            "data split before tokenization."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--splits",
        default="train,validation",
        help="Comma-separated split names to shuffle. Defaults to train,validation.",
    )
    parser.add_argument(
        "--bucket-count",
        type=int,
        default=DEFAULT_PROCESSED_SHUFFLE_BUCKETS,
    )
    parser.add_argument(
        "--max-open-buckets",
        type=int,
        default=DEFAULT_MAX_OPEN_BUCKETS,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    stats = shuffle_processed_splits(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        splits=_parse_splits(args.splits),
        bucket_count=args.bucket_count,
        max_open_buckets=args.max_open_buckets,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
