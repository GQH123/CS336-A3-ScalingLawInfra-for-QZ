from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scaling_data.blend import build_blend_plan, load_manifest
from scaling_data.prepare import (
    _normalize_non_negative_int,
    _normalize_positive_int,
    _resolve_tokenize_max_pending_chunks,
    _tokenize_progress_callback,
    _tokenize_processed_records,
    _tokenized_index_summary,
    _train_source_token_limits,
)
from scaling_data.shuffle_processed import (
    DEFAULT_MAX_OPEN_BUCKETS,
    DEFAULT_PROCESSED_SHUFFLE_BUCKETS,
    resolve_processed_shuffle_max_open_buckets,
    reusable_processed_shuffle_stats,
    shuffle_processed_splits,
)
from scaling_data.tokenize import DEFAULT_TOKENIZE_CHUNK_RECORDS


@dataclass(frozen=True)
class RebuildTokenizedConfig:
    manifest_path: Path
    processed_dir: Path
    work_dir: Path
    tokenizer_name: str = ""
    target_shard_tokens: int = 100_000_000
    shuffle_processed_records: bool = True
    reuse_existing_shuffle: bool = True
    shuffle_seed: int = 20260724
    shuffle_bucket_count: int = DEFAULT_PROCESSED_SHUFFLE_BUCKETS
    shuffle_max_open_buckets: int = DEFAULT_MAX_OPEN_BUCKETS
    tokenize_workers: int = 1
    tokenize_chunk_records: int = DEFAULT_TOKENIZE_CHUNK_RECORDS
    tokenize_max_pending_chunks: int | None = None
    tokenize_progress_every_chunks: int = 100
    index_validation_mode: str = "metadata"
    show_progress: bool = False


def rebuild_tokenized_from_processed(
    config: RebuildTokenizedConfig,
    *,
    tokenizer_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(config.manifest_path)
    tokenizer_name = config.tokenizer_name or str(manifest.get("tokenizer", "")).strip()
    if not tokenizer_name:
        raise ValueError("tokenizer_name is required")

    blend_plan = build_blend_plan(manifest)
    train_source_token_limits = _train_source_token_limits(blend_plan)
    shuffled_processed_dir = config.work_dir / "processed_jsonl_shuffled"
    tokenized_dir = config.work_dir / "tokenized"
    tokenize_workers = _normalize_positive_int(
        config.tokenize_workers,
        field_name="tokenize_workers",
    )
    tokenize_chunk_records = _normalize_positive_int(
        config.tokenize_chunk_records,
        field_name="tokenize_chunk_records",
    )
    tokenize_max_pending_chunks = _resolve_tokenize_max_pending_chunks(
        config.tokenize_max_pending_chunks,
        workers=tokenize_workers,
    )
    shuffle_bucket_count = _normalize_positive_int(
        config.shuffle_bucket_count,
        field_name="shuffle_bucket_count",
    )
    requested_shuffle_max_open_buckets = _normalize_non_negative_int(
        config.shuffle_max_open_buckets,
        field_name="shuffle_max_open_buckets",
    )
    shuffle_max_open_buckets = resolve_processed_shuffle_max_open_buckets(
        bucket_count=shuffle_bucket_count,
        max_open_buckets=requested_shuffle_max_open_buckets,
    )
    tokenize_progress_every_chunks = _normalize_positive_int(
        config.tokenize_progress_every_chunks,
        field_name="tokenize_progress_every_chunks",
    )

    shuffle_stats: dict[str, Any] = {}
    if config.shuffle_processed_records:
        if config.reuse_existing_shuffle:
            reusable_stats = reusable_processed_shuffle_stats(
                input_dir=config.processed_dir,
                output_dir=shuffled_processed_dir,
                seed=int(config.shuffle_seed),
                bucket_count=shuffle_bucket_count,
            )
        else:
            reusable_stats = None
        if reusable_stats is not None:
            shuffle_stats = reusable_stats
            if config.show_progress:
                print(
                    f"[rebuild] reusing existing processed shuffle at "
                    f"{shuffled_processed_dir}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            if config.show_progress:
                print(
                    (
                        f"[rebuild] shuffling processed records from "
                        f"{config.processed_dir} into {shuffled_processed_dir} "
                        f"seed={int(config.shuffle_seed)} "
                        f"bucket_count={shuffle_bucket_count} "
                        f"max_open_buckets={shuffle_max_open_buckets}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            shuffle_stats = shuffle_processed_splits(
                input_dir=config.processed_dir,
                output_dir=shuffled_processed_dir,
                seed=int(config.shuffle_seed),
                bucket_count=shuffle_bucket_count,
                max_open_buckets=shuffle_max_open_buckets,
                overwrite=True,
            )
        tokenization_input_dir = shuffled_processed_dir
    else:
        tokenization_input_dir = config.processed_dir

    if config.show_progress:
        print(
            (
                f"[rebuild] tokenizing records from {tokenization_input_dir} "
                f"into {tokenized_dir} workers={tokenize_workers} "
                f"chunk_records={tokenize_chunk_records} "
                f"max_pending_chunks={tokenize_max_pending_chunks}"
            ),
            file=sys.stderr,
            flush=True,
        )
    indexes = _tokenize_processed_records(
        input_dir=tokenization_input_dir,
        output_dir=tokenized_dir,
        tokenizer_name=tokenizer_name,
        target_shard_tokens=config.target_shard_tokens,
        train_source_token_limits=train_source_token_limits,
        workers=tokenize_workers,
        chunk_records=tokenize_chunk_records,
        max_pending_chunks=tokenize_max_pending_chunks,
        progress_callback=_tokenize_progress_callback(
            enabled=config.show_progress,
            every_chunks=tokenize_progress_every_chunks,
        ),
        tokenizer_loader=tokenizer_loader,
    )
    tokenized_indexes = _tokenized_index_summary(
        tokenized_dir=tokenized_dir,
        indexes=indexes,
        validation_mode=config.index_validation_mode,
    )
    summary = {
        "status": "completed",
        "manifest_name": manifest["name"],
        "source_manifest_path": str(config.manifest_path),
        "processed_dir": str(config.processed_dir),
        "processed_records_shuffled": config.shuffle_processed_records,
        "reuse_existing_shuffle": config.reuse_existing_shuffle,
        "shuffled_processed_dir": str(shuffled_processed_dir),
        "tokenization_input_dir": str(tokenization_input_dir),
        "tokenized_dir": str(tokenized_dir),
        "tokenizer": tokenizer_name,
        "target_shard_tokens": int(config.target_shard_tokens),
        "train_source_token_limits": dict(train_source_token_limits),
        "shuffle_seed": int(config.shuffle_seed),
        "shuffle_bucket_count": shuffle_bucket_count,
        "requested_shuffle_max_open_buckets": requested_shuffle_max_open_buckets,
        "shuffle_max_open_buckets": shuffle_max_open_buckets,
        "processed_shuffle_stats": shuffle_stats,
        "tokenize_workers": tokenize_workers,
        "tokenize_chunk_records": tokenize_chunk_records,
        "tokenize_max_pending_chunks": tokenize_max_pending_chunks,
        "tokenize_progress_every_chunks": tokenize_progress_every_chunks,
        "tokenized_indexes": tokenized_indexes,
    }
    summary_path = config.work_dir / "rebuilt-tokenized-manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["rebuilt_manifest_path"] = str(summary_path)
    return summary


def main(
    argv: Sequence[str] | None = None,
    *,
    tokenizer_loader: Callable[[str], Any] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild tokenized binary shards from existing processed JSONL, "
            "optionally shuffling whole records before tokenization."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data_work/processed_jsonl"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data_work"))
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--target-shard-tokens", type=int, default=100_000_000)
    parser.add_argument(
        "--no-shuffle-processed-records",
        action="store_true",
        help="Tokenize processed records in their existing order.",
    )
    parser.add_argument(
        "--force-reshuffle",
        action="store_true",
        help=(
            "Rewrite processed_jsonl_shuffled even when a completed matching "
            "shuffle-stats.json can be reused."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260724)
    parser.add_argument(
        "--shuffle-bucket-count",
        type=int,
        default=DEFAULT_PROCESSED_SHUFFLE_BUCKETS,
    )
    parser.add_argument(
        "--shuffle-max-open-buckets",
        type=int,
        default=DEFAULT_MAX_OPEN_BUCKETS,
        help=(
            "Maximum shuffle bucket files kept open. Use 0, the default, to "
            "auto-size from the process file descriptor limit."
        ),
    )
    parser.add_argument("--tokenize-workers", type=int, default=1)
    parser.add_argument(
        "--tokenize-chunk-records",
        type=int,
        default=DEFAULT_TOKENIZE_CHUNK_RECORDS,
    )
    parser.add_argument("--tokenize-max-pending-chunks", type=int)
    parser.add_argument(
        "--tokenize-progress-every-chunks",
        type=int,
        default=100,
        help="When progress is enabled, print tokenization status every N chunks.",
    )
    parser.add_argument(
        "--index-validation-mode",
        choices=("metadata", "full"),
        default="metadata",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable rebuild stage logs and tokenization progress on stderr.",
    )
    args = parser.parse_args(argv)

    summary = rebuild_tokenized_from_processed(
        RebuildTokenizedConfig(
            manifest_path=args.manifest,
            processed_dir=args.processed_dir,
            work_dir=args.work_dir,
            tokenizer_name=args.tokenizer,
            target_shard_tokens=args.target_shard_tokens,
            shuffle_processed_records=not args.no_shuffle_processed_records,
            reuse_existing_shuffle=not args.force_reshuffle,
            shuffle_seed=args.shuffle_seed,
            shuffle_bucket_count=args.shuffle_bucket_count,
            shuffle_max_open_buckets=args.shuffle_max_open_buckets,
            tokenize_workers=args.tokenize_workers,
            tokenize_chunk_records=args.tokenize_chunk_records,
            tokenize_max_pending_chunks=args.tokenize_max_pending_chunks,
            tokenize_progress_every_chunks=args.tokenize_progress_every_chunks,
            index_validation_mode=args.index_validation_mode,
            show_progress=not args.no_progress,
        ),
        tokenizer_loader=tokenizer_loader,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
