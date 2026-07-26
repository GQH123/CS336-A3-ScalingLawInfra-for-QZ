from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from scaling_data.blend import build_blend_plan, load_manifest
from scaling_data.download import (
    DownloadTask,
    build_download_tasks,
    download_source,
    print_progress_event,
    write_queued_download_state,
)
from scaling_data.pipeline import write_pipeline_plan
from scaling_data.postprocess import postprocess_directory
from scaling_data.shuffle_processed import (
    DEFAULT_MAX_OPEN_BUCKETS,
    DEFAULT_PROCESSED_SHUFFLE_BUCKETS,
    shuffle_processed_splits,
)
from scaling_data.tokenize import (
    DEFAULT_TOKENIZE_CHUNK_RECORDS,
    load_tokenized_index,
    tokenize_directory,
    tokenize_records_by_split,
)


@dataclass(frozen=True)
class PrepareDataConfig:
    manifest_path: Path
    work_dir: Path
    tokenizer_name: str = ""
    target_shard_tokens: int = 100_000_000
    validation_rate: float = 0.001
    min_chars: int = 200
    max_chars: int | None = None
    max_docs_per_source: int | None = None
    download_token_overfetch_ratio: float = 0.0
    download_chars_per_token: float = 4.0
    download_workers: int = 4
    download_shards_per_source: int = 1
    download_source_parallelism: int = 1
    postprocess_workers: int | None = None
    tokenize_workers: int | None = None
    tokenize_chunk_records: int = DEFAULT_TOKENIZE_CHUNK_RECORDS
    tokenize_max_pending_chunks: int | None = None
    tokenize_progress_every_chunks: int = 100
    shuffle_processed_records: bool = True
    processed_shuffle_seed: int = 20260724
    processed_shuffle_bucket_count: int = DEFAULT_PROCESSED_SHUFFLE_BUCKETS
    processed_shuffle_max_open_buckets: int = DEFAULT_MAX_OPEN_BUCKETS
    resume_downloads: bool = True
    progress_every_docs: int = 1_000
    index_validation_mode: str = "metadata"
    plan_only: bool = False
    show_progress: bool = False


@dataclass(frozen=True)
class _DownloadFailure:
    download_id: str
    logical_source_id: str
    error_type: str
    error_message: str


class DownloadStageError(RuntimeError):
    def __init__(self, failures: Sequence[_DownloadFailure]):
        self.failures = tuple(failures)
        shown = "; ".join(
            (
                f"{failure.download_id} "
                f"({failure.error_type}: {failure.error_message})"
            )
            for failure in self.failures[:8]
        )
        suffix = ""
        if len(self.failures) > 8:
            suffix = f"; ... +{len(self.failures) - 8} more"
        super().__init__(
            f"{len(self.failures)} download shard(s) failed: {shown}{suffix}"
        )


def prepare_data(
    config: PrepareDataConfig,
    *,
    tokenizer_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(config.manifest_path)
    tokenizer_name = config.tokenizer_name or str(manifest.get("tokenizer", "")).strip()
    if not tokenizer_name and not config.plan_only:
        raise ValueError("tokenizer_name is required")

    raw_dir = config.work_dir / "raw_jsonl"
    processed_dir = config.work_dir / "processed_jsonl"
    shuffled_processed_dir = config.work_dir / "processed_jsonl_shuffled"
    tokenized_dir = config.work_dir / "tokenized"
    plan_dir = config.work_dir / "plans"

    pipeline_plan_path = write_pipeline_plan(
        manifest_path=config.manifest_path,
        output_dir=plan_dir,
        work_dir=config.work_dir,
    )
    blend_plan = build_blend_plan(manifest)
    blend_plan_path = _write_blend_plan(blend_plan, output_dir=plan_dir)
    train_source_token_limits = _train_source_token_limits(blend_plan)
    download_token_budgets = _download_token_budgets(
        blend_plan,
        overfetch_ratio=config.download_token_overfetch_ratio,
    )
    resolved_download_workers = _normalize_download_workers(
        requested_workers=config.download_workers,
    )
    resolved_postprocess_workers = _resolve_stage_workers(
        requested_workers=config.postprocess_workers,
        default_workers=resolved_download_workers,
        field_name="postprocess_workers",
    )
    resolved_tokenize_workers = _resolve_stage_workers(
        requested_workers=config.tokenize_workers,
        default_workers=resolved_download_workers,
        field_name="tokenize_workers",
    )
    resolved_tokenize_chunk_records = _normalize_positive_int(
        config.tokenize_chunk_records,
        field_name="tokenize_chunk_records",
    )
    resolved_tokenize_max_pending_chunks = _resolve_tokenize_max_pending_chunks(
        config.tokenize_max_pending_chunks,
        workers=resolved_tokenize_workers,
    )
    resolved_tokenize_progress_every_chunks = _normalize_positive_int(
        config.tokenize_progress_every_chunks,
        field_name="tokenize_progress_every_chunks",
    )
    resolved_processed_shuffle_bucket_count = _normalize_positive_int(
        config.processed_shuffle_bucket_count,
        field_name="processed_shuffle_bucket_count",
    )
    resolved_processed_shuffle_max_open_buckets = _normalize_positive_int(
        config.processed_shuffle_max_open_buckets,
        field_name="processed_shuffle_max_open_buckets",
    )
    tokenization_input_dir = (
        shuffled_processed_dir if config.shuffle_processed_records else processed_dir
    )
    summary: dict[str, Any] = {
        "status": "planned",
        "manifest_name": manifest["name"],
        "pipeline_plan_path": str(pipeline_plan_path),
        "blend_plan_path": str(blend_plan_path),
        "work_dir": str(config.work_dir),
        "train_source_token_limits": train_source_token_limits,
        "download_token_budgets": download_token_budgets,
        "download_workers": resolved_download_workers,
        "download_shards_per_source": _normalize_download_shards_per_source(
            config.download_shards_per_source
        ),
        "download_source_parallelism": _normalize_download_source_parallelism(
            config.download_source_parallelism
        ),
        "postprocess_workers": resolved_postprocess_workers,
        "tokenize_workers": resolved_tokenize_workers,
        "tokenize_chunk_records": resolved_tokenize_chunk_records,
        "tokenize_max_pending_chunks": resolved_tokenize_max_pending_chunks,
        "tokenize_progress_every_chunks": resolved_tokenize_progress_every_chunks,
        "processed_records_shuffled": config.shuffle_processed_records,
        "processed_dir": str(processed_dir),
        "shuffled_processed_dir": str(shuffled_processed_dir),
        "tokenization_input_dir": str(tokenization_input_dir),
        "processed_shuffle_seed": int(config.processed_shuffle_seed),
        "processed_shuffle_bucket_count": resolved_processed_shuffle_bucket_count,
        "processed_shuffle_max_open_buckets": resolved_processed_shuffle_max_open_buckets,
        "resume_downloads": config.resume_downloads,
    }
    if config.plan_only:
        return summary

    raw_dir.mkdir(parents=True, exist_ok=True)
    total_sources = len(blend_plan["sources"])
    if config.show_progress:
        print(
            (
                f"[prepare] plans written pipeline={pipeline_plan_path} "
                f"blend={blend_plan_path}"
            ),
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[prepare] downloading {total_sources} enabled sources to {raw_dir}",
            file=sys.stderr,
            flush=True,
        )
    raw_paths, download_stats = _download_sources(
        sources=blend_plan["sources"],
        output_dir=raw_dir,
        config=config,
        download_token_budgets=download_token_budgets,
    )

    if config.show_progress:
        print(
            f"[prepare] post-processing raw JSONL into {processed_dir}",
            file=sys.stderr,
            flush=True,
        )
    postprocess_stats = postprocess_directory(
        input_dir=raw_dir,
        output_dir=processed_dir,
        validation_rate=config.validation_rate,
        min_chars=config.min_chars,
        max_chars=config.max_chars,
        workers=resolved_postprocess_workers,
    )
    if config.show_progress:
        print(
            f"[prepare] post-processing complete stats={postprocess_stats}",
            file=sys.stderr,
            flush=True,
        )
    processed_shuffle_stats: dict[str, Any] = {}
    if config.shuffle_processed_records:
        if config.show_progress:
            print(
                (
                    f"[prepare] shuffling processed JSONL records from {processed_dir} "
                    f"into {shuffled_processed_dir}"
                ),
                file=sys.stderr,
                flush=True,
            )
        processed_shuffle_stats = shuffle_processed_splits(
            input_dir=processed_dir,
            output_dir=shuffled_processed_dir,
            seed=int(config.processed_shuffle_seed),
            bucket_count=resolved_processed_shuffle_bucket_count,
            max_open_buckets=resolved_processed_shuffle_max_open_buckets,
            overwrite=True,
        )
        if config.show_progress:
            print(
                f"[prepare] processed shuffle complete stats={processed_shuffle_stats}",
                file=sys.stderr,
                flush=True,
            )
    tokenization_input_dir = (
        shuffled_processed_dir if config.shuffle_processed_records else processed_dir
    )
    summary["tokenization_input_dir"] = str(tokenization_input_dir)
    summary["processed_shuffle_stats"] = processed_shuffle_stats
    if config.show_progress:
        print(
            f"[prepare] tokenizing processed records from {tokenization_input_dir} "
            f"into {tokenized_dir}",
            file=sys.stderr,
            flush=True,
        )
    indexes = _tokenize_processed_records(
        input_dir=tokenization_input_dir,
        output_dir=tokenized_dir,
        tokenizer_name=tokenizer_name,
        target_shard_tokens=config.target_shard_tokens,
        train_source_token_limits=train_source_token_limits,
        workers=resolved_tokenize_workers,
        chunk_records=resolved_tokenize_chunk_records,
        max_pending_chunks=resolved_tokenize_max_pending_chunks,
        progress_callback=_tokenize_progress_callback(
            enabled=config.show_progress,
            every_chunks=resolved_tokenize_progress_every_chunks,
        ),
        tokenizer_loader=tokenizer_loader,
    )
    tokenized_indexes = _tokenized_index_summary(
        tokenized_dir=tokenized_dir,
        indexes=indexes,
        validation_mode=config.index_validation_mode,
    )
    if config.show_progress:
        print(
            f"[prepare] tokenization complete indexes={tokenized_indexes}",
            file=sys.stderr,
            flush=True,
        )
    prepared_manifest = _prepared_manifest(
        config=config,
        manifest=manifest,
        tokenizer_name=tokenizer_name,
        raw_paths=raw_paths,
        postprocess_stats=postprocess_stats,
        pipeline_plan_path=pipeline_plan_path,
        blend_plan_path=blend_plan_path,
        train_source_token_limits=train_source_token_limits,
        download_token_budgets=download_token_budgets,
        download_stats=download_stats,
        processed_records_shuffled=config.shuffle_processed_records,
        processed_dir=processed_dir,
        shuffled_processed_dir=shuffled_processed_dir,
        tokenization_input_dir=tokenization_input_dir,
        processed_shuffle_stats=processed_shuffle_stats,
        tokenized_indexes=tokenized_indexes,
    )
    prepared_manifest_path = config.work_dir / "prepared-data-manifest.json"
    prepared_manifest_path.write_text(
        json.dumps(prepared_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary.update(
        {
            "status": "completed",
            "prepared_manifest_path": str(prepared_manifest_path),
            "raw_paths": raw_paths,
            "download_stats": download_stats,
            "postprocess_stats": postprocess_stats,
            "processed_shuffle_stats": processed_shuffle_stats,
            "tokenized_indexes": tokenized_indexes,
        }
    )
    return summary


def main(
    argv: Sequence[str] | None = None,
    *,
    tokenizer_loader: Callable[[str], Any] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full scaling-law data preparation pipeline."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("source/data/manifests/general_100b_mix.json"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data_work"))
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--target-shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--validation-rate", type=float, default=0.001)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--max-docs-per-source", type=int)
    parser.add_argument("--download-token-overfetch-ratio", type=float, default=0.0)
    parser.add_argument("--download-chars-per-token", type=float, default=4.0)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--download-shards-per-source", type=int, default=4)
    parser.add_argument("--download-source-parallelism", type=int, default=1)
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        help="Process workers for post-processing. Defaults to --download-workers.",
    )
    parser.add_argument(
        "--tokenize-workers",
        type=int,
        help="Process workers for tokenization. Defaults to --download-workers.",
    )
    parser.add_argument(
        "--tokenize-chunk-records",
        type=int,
        default=DEFAULT_TOKENIZE_CHUNK_RECORDS,
        help="Processed JSONL records per tokenization process task.",
    )
    parser.add_argument(
        "--tokenize-max-pending-chunks",
        type=int,
        help=(
            "Maximum submitted tokenization tasks waiting in the process pool. "
            "Defaults to four times --tokenize-workers."
        ),
    )
    parser.add_argument(
        "--tokenize-progress-every-chunks",
        type=int,
        default=100,
        help="When progress is enabled, print tokenization status every N chunks.",
    )
    parser.add_argument(
        "--no-shuffle-processed-records",
        action="store_true",
        help=(
            "Tokenize processed JSONL in its original postprocess order instead "
            "of first shuffling whole records within each split."
        ),
    )
    parser.add_argument("--processed-shuffle-seed", type=int, default=20260724)
    parser.add_argument(
        "--processed-shuffle-bucket-count",
        type=int,
        default=DEFAULT_PROCESSED_SHUFFLE_BUCKETS,
    )
    parser.add_argument(
        "--processed-shuffle-max-open-buckets",
        type=int,
        default=DEFAULT_MAX_OPEN_BUCKETS,
    )
    parser.add_argument(
        "--no-resume-downloads",
        action="store_true",
        help="Ignore existing download checkpoints and rebuild raw source files.",
    )
    parser.add_argument("--progress-every-docs", type=int, default=1_000)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable download progress bars and stage logs on stderr.",
    )
    parser.add_argument(
        "--index-validation-mode",
        choices=("metadata", "full"),
        default="metadata",
        help=(
            "Validate final tokenized indexes using cheap metadata checks or "
            "full shard re-hashing. Defaults to metadata."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write blend/pipeline plans and skip download, postprocess, and tokenize.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Deprecated alias for --plan-only.",
    )
    args = parser.parse_args(argv)

    summary = prepare_data(
        PrepareDataConfig(
            manifest_path=args.manifest,
            work_dir=args.work_dir,
            tokenizer_name=args.tokenizer,
            target_shard_tokens=args.target_shard_tokens,
            validation_rate=args.validation_rate,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            max_docs_per_source=args.max_docs_per_source,
            download_token_overfetch_ratio=args.download_token_overfetch_ratio,
            download_chars_per_token=args.download_chars_per_token,
            download_workers=args.download_workers,
            download_shards_per_source=args.download_shards_per_source,
            download_source_parallelism=args.download_source_parallelism,
            postprocess_workers=args.postprocess_workers,
            tokenize_workers=args.tokenize_workers,
            tokenize_chunk_records=args.tokenize_chunk_records,
            tokenize_max_pending_chunks=args.tokenize_max_pending_chunks,
            tokenize_progress_every_chunks=args.tokenize_progress_every_chunks,
            shuffle_processed_records=not args.no_shuffle_processed_records,
            processed_shuffle_seed=args.processed_shuffle_seed,
            processed_shuffle_bucket_count=args.processed_shuffle_bucket_count,
            processed_shuffle_max_open_buckets=args.processed_shuffle_max_open_buckets,
            resume_downloads=not args.no_resume_downloads,
            progress_every_docs=args.progress_every_docs,
            index_validation_mode=args.index_validation_mode,
            plan_only=args.plan_only or args.dry_run,
            show_progress=not args.no_progress,
        ),
        tokenizer_loader=tokenizer_loader,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


class _DownloadProgressReporter:
    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self._bar = None
        self._bar_mode = "scanned_docs"
        self._last_scanned = 0
        self._last_written = 0
        self._last_estimated_tokens = 0
        self._tqdm = self._load_tqdm() if enabled else None

    def callback_for_source(
        self,
        *,
        source_index: int,
        total_sources: int,
    ) -> Callable[[dict], None] | None:
        if not self.enabled:
            return None

        def callback(event: dict) -> None:
            self._handle(
                event,
                source_index=source_index,
                total_sources=total_sources,
            )

        return callback

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _handle(
        self,
        event: dict,
        *,
        source_index: int,
        total_sources: int,
    ) -> None:
        event_name = event.get("event")
        if event_name == "source_start":
            self.close()
            self._bar_mode = "scanned_docs"
            self._last_scanned = 0
            self._last_written = 0
            self._last_estimated_tokens = 0
            if self._tqdm is None:
                print(
                    f"[download] source {source_index}/{total_sources}",
                    file=sys.stderr,
                    flush=True,
                )
                print_progress_event(event)
                return
            total = event.get("max_docs")
            unit = "doc"
            if isinstance(total, int) and total > 0:
                self._bar_mode = "written_docs"
            else:
                total = event.get("max_estimated_tokens")
                if isinstance(total, int) and total > 0:
                    unit = "tok"
                    self._bar_mode = "estimated_tokens"
                else:
                    total = None
                    self._bar_mode = "scanned_docs"
            if not isinstance(total, int) or total <= 0:
                total = None
            source_id = event.get("source_id", "unknown")
            self._bar = self._tqdm(
                total=total,
                desc=f"download {source_index}/{total_sources} {source_id}",
                unit=unit,
                dynamic_ncols=True,
                leave=True,
                file=sys.stderr,
            )
            self._bar.set_postfix(scanned=0, written=0, est_tokens=0)
            self._bar.refresh()
        elif event_name == "source_progress":
            self._update_bar_or_print(event)
        elif event_name == "source_complete":
            self._update_bar_or_print(event)
            self.close()
            print_progress_event(event)

    def _update_bar_or_print(self, event: dict) -> None:
        if self._bar is None:
            print_progress_event(event)
            return
        scanned = _nonnegative_int(event.get("scanned_documents"))
        written = _nonnegative_int(event.get("written_documents"))
        estimated_tokens = _nonnegative_int(event.get("estimated_tokens"))
        if self._bar_mode == "estimated_tokens":
            delta = max(0, estimated_tokens - self._last_estimated_tokens)
        elif self._bar_mode == "written_docs":
            delta = max(0, written - self._last_written)
        else:
            delta = max(0, scanned - self._last_scanned)
        if delta:
            self._bar.update(delta)
        self._bar.set_postfix(
            scanned=scanned,
            written=written,
            est_tokens=estimated_tokens,
        )
        self._bar.refresh()
        self._last_scanned = scanned
        self._last_written = written
        self._last_estimated_tokens = estimated_tokens

    @staticmethod
    def _load_tqdm():
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return None
        return tqdm


@dataclass(frozen=True)
class _SourceDownloadPlan:
    source_index: int
    total_sources: int
    source_id: str
    tasks: list[DownloadTask]


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _download_sources(
    *,
    sources: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: PrepareDataConfig,
    download_token_budgets: Mapping[str, int],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    total_sources = len(sources)
    workers = _normalize_download_workers(requested_workers=config.download_workers)
    shards_per_source = _normalize_download_shards_per_source(
        config.download_shards_per_source
    )
    requested_source_parallelism = _normalize_download_source_parallelism(
        config.download_source_parallelism
    )
    active_source_limit = min(workers, requested_source_parallelism)
    download_stats: dict[str, dict[str, Any]] = {}
    raw_paths_by_download_id: dict[str, str] = {}
    ordered_download_ids: list[str] = []
    download_failures: list[_DownloadFailure] = []
    progress_lock = threading.Lock()

    def progress_callback(event: dict) -> None:
        if not config.show_progress:
            return
        with progress_lock:
            print_progress_event(event)

    source_plans: list[_SourceDownloadPlan] = []
    for source_index, source in enumerate(sources, start=1):
        source_id = str(source["id"])
        tasks = build_download_tasks(
            dict(source),
            shards_per_source=shards_per_source,
            max_docs=config.max_docs_per_source,
            max_estimated_tokens=download_token_budgets.get(source_id),
        )
        ordered_download_ids.extend(task.download_id for task in tasks)
        for task in tasks:
            write_queued_download_state(source=task.source, output_dir=output_dir)
        source_plans.append(
            _SourceDownloadPlan(
                source_index=source_index,
                total_sources=total_sources,
                source_id=source_id,
                tasks=tasks,
            )
        )

    if active_source_limit <= 1:
        _download_sources_sequentially(
            source_plans=source_plans,
            output_dir=output_dir,
            config=config,
            workers=workers,
            download_stats=download_stats,
            raw_paths_by_download_id=raw_paths_by_download_id,
            download_failures=download_failures,
            progress_callback=progress_callback,
            progress_lock=progress_lock,
        )
    else:
        _download_sources_with_source_parallelism(
            source_plans=source_plans,
            output_dir=output_dir,
            config=config,
            workers=workers,
            active_source_limit=active_source_limit,
            download_stats=download_stats,
            raw_paths_by_download_id=raw_paths_by_download_id,
            download_failures=download_failures,
            progress_callback=progress_callback,
            progress_lock=progress_lock,
        )

    if download_failures:
        raise DownloadStageError(download_failures)

    return (
        _ordered_raw_paths(ordered_download_ids, raw_paths_by_download_id),
        download_stats,
    )


def _download_sources_sequentially(
    *,
    source_plans: Sequence[_SourceDownloadPlan],
    output_dir: Path,
    config: PrepareDataConfig,
    workers: int,
    download_stats: dict[str, dict[str, Any]],
    raw_paths_by_download_id: dict[str, str],
    download_failures: list[_DownloadFailure],
    progress_callback: Callable[[dict], None] | None,
    progress_lock: threading.Lock,
) -> None:
    for source_plan in source_plans:
        source_workers = min(workers, len(source_plan.tasks))
        if config.show_progress:
            with progress_lock:
                print(
                    (
                        "[prepare] downloading source "
                        f"{source_plan.source_index}/{source_plan.total_sources} "
                        f"{source_plan.source_id} with {source_workers} worker(s), "
                        f"{len(source_plan.tasks)} shard(s)"
                    ),
                    file=sys.stderr,
                    flush=True,
                )

        if source_workers <= 1:
            reporter = _DownloadProgressReporter(enabled=config.show_progress)
            try:
                for task_index, task in enumerate(source_plan.tasks, start=1):
                    try:
                        downloaded_source_id, raw_path, stats = _download_one_task(
                            task=task,
                            output_dir=output_dir,
                            config=config,
                            progress_callback=reporter.callback_for_source(
                                source_index=task_index,
                                total_sources=len(source_plan.tasks),
                            ),
                        )
                    except Exception as exc:
                        download_failures.append(_download_failure(task, exc))
                    else:
                        raw_paths_by_download_id[downloaded_source_id] = raw_path
                        download_stats[downloaded_source_id] = stats
            finally:
                reporter.close()
            continue

        with ProcessPoolExecutor(max_workers=source_workers) as executor:
            futures = {
                executor.submit(
                    _download_one_task,
                    task=task,
                    output_dir=output_dir,
                    config=config,
                ): task
                for task in source_plan.tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    source_id, raw_path, stats = future.result()
                except Exception as exc:
                    download_failures.append(_download_failure(task, exc))
                else:
                    raw_paths_by_download_id[source_id] = raw_path
                    download_stats[source_id] = stats


@dataclass
class _ActiveSourceDownload:
    plan: _SourceDownloadPlan
    next_task_index: int = 0
    running_tasks: int = 0


def _download_sources_with_source_parallelism(
    *,
    source_plans: Sequence[_SourceDownloadPlan],
    output_dir: Path,
    config: PrepareDataConfig,
    workers: int,
    active_source_limit: int,
    download_stats: dict[str, dict[str, Any]],
    raw_paths_by_download_id: dict[str, str],
    download_failures: list[_DownloadFailure],
    progress_callback: Callable[[dict], None] | None,
    progress_lock: threading.Lock,
) -> None:
    pending_plans = list(source_plans)
    active_sources: list[_ActiveSourceDownload] = []
    running_futures = {}
    round_robin_index = 0

    def activate_next_source() -> bool:
        if not pending_plans or len(active_sources) >= active_source_limit:
            return False
        plan = pending_plans.pop(0)
        active_sources.append(_ActiveSourceDownload(plan=plan))
        if config.show_progress:
            with progress_lock:
                print(
                    (
                        "[prepare] activating source "
                        f"{plan.source_index}/{plan.total_sources} "
                        f"{plan.source_id} with {len(plan.tasks)} shard(s); "
                        f"active_sources={len(active_sources)}/{active_source_limit} "
                        f"worker_pool={workers}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        return True

    with ProcessPoolExecutor(max_workers=workers) as executor:
        while pending_plans or active_sources or running_futures:
            while len(running_futures) < workers:
                while len(active_sources) < active_source_limit and pending_plans:
                    activate_next_source()

                if not active_sources:
                    break

                submitted = False
                for offset in range(len(active_sources)):
                    source_index = (
                        round_robin_index + offset
                    ) % len(active_sources)
                    active_source = active_sources[source_index]
                    if active_source.next_task_index >= len(
                        active_source.plan.tasks
                    ):
                        continue
                    task = active_source.plan.tasks[
                        active_source.next_task_index
                    ]
                    active_source.next_task_index += 1
                    active_source.running_tasks += 1
                    future = executor.submit(
                        _download_one_task,
                        task=task,
                        output_dir=output_dir,
                        config=config,
                    )
                    running_futures[future] = (active_source, task)
                    round_robin_index = (source_index + 1) % len(active_sources)
                    submitted = True
                    break

                if not submitted:
                    break

            if not running_futures:
                break

            completed, _ = wait(
                running_futures,
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                active_source, task = running_futures.pop(future)
                try:
                    downloaded_source_id, raw_path, stats = future.result()
                except Exception as exc:
                    download_failures.append(_download_failure(task, exc))
                else:
                    raw_paths_by_download_id[downloaded_source_id] = raw_path
                    download_stats[downloaded_source_id] = stats
                active_source.running_tasks -= 1

            active_sources = [
                active_source
                for active_source in active_sources
                if not (
                    active_source.next_task_index == len(active_source.plan.tasks)
                    and active_source.running_tasks == 0
                )
            ]
            if active_sources:
                round_robin_index %= len(active_sources)
            else:
                round_robin_index = 0


def _download_one_task(
    *,
    task: DownloadTask,
    output_dir: Path,
    config: PrepareDataConfig,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    download_stats: dict[str, dict[str, Any]] = {}
    raw_path = download_source(
        source=task.source,
        output_dir=output_dir,
        max_docs=task.max_docs,
        max_estimated_tokens=task.max_estimated_tokens,
        chars_per_token=config.download_chars_per_token,
        min_chars=config.min_chars,
        progress_every_docs=config.progress_every_docs,
        resume=config.resume_downloads,
        progress_callback=_collect_download_stats(
            source_id=task.download_id,
            download_stats=download_stats,
            progress_callback=progress_callback,
        ),
    )
    return task.download_id, str(raw_path), dict(
        download_stats.get(task.download_id, {})
    )


def _download_failure(task: DownloadTask, exc: Exception) -> _DownloadFailure:
    return _DownloadFailure(
        download_id=task.download_id,
        logical_source_id=task.logical_source_id,
        error_type=type(exc).__name__,
        error_message=_truncate_error_message(exc),
    )


def _truncate_error_message(value, *, limit: int = 1000) -> str:
    message = "" if value is None else str(value)
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _ordered_raw_paths(
    ordered_download_ids: Sequence[str],
    raw_paths_by_download_id: Mapping[str, str],
) -> list[str]:
    return [raw_paths_by_download_id[source_id] for source_id in ordered_download_ids]


def _normalize_download_workers(
    *,
    requested_workers: int,
) -> int:
    if isinstance(requested_workers, bool):
        raise ValueError("download_workers must be a positive integer")
    parsed = int(requested_workers)
    if parsed <= 0:
        raise ValueError("download_workers must be a positive integer")
    return parsed


def _normalize_download_shards_per_source(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("download_shards_per_source must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("download_shards_per_source must be a positive integer")
    return parsed


def _normalize_download_source_parallelism(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("download_source_parallelism must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("download_source_parallelism must be a positive integer")
    return parsed


def _resolve_stage_workers(
    *,
    requested_workers: int | None,
    default_workers: int,
    field_name: str,
) -> int:
    if requested_workers is None:
        return default_workers
    return _normalize_positive_int(requested_workers, field_name=field_name)


def _normalize_positive_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _resolve_tokenize_max_pending_chunks(
    value: int | None,
    *,
    workers: int,
) -> int:
    if value is None:
        return max(1, workers * 4)
    return _normalize_positive_int(value, field_name="tokenize_max_pending_chunks")


def _tokenize_progress_callback(
    *,
    enabled: bool,
    every_chunks: int,
) -> Callable[[dict], None] | None:
    if not enabled:
        return None
    counters = {
        "submitted": 0,
        "completed": 0,
        "consumed": 0,
    }

    def callback(event: dict) -> None:
        event_name = event.get("event")
        if event_name == "tokenize_chunk_submitted":
            counters["submitted"] += 1
            return
        if event_name == "tokenize_chunk_completed":
            counters["completed"] += 1
            return
        if event_name != "tokenize_chunk_consumed":
            return
        counters["consumed"] += 1
        if counters["consumed"] != 1 and counters["consumed"] % every_chunks != 0:
            return
        print(
            (
                "[tokenize] "
                f"submitted_chunks={counters['submitted']} "
                f"completed_chunks={counters['completed']} "
                f"consumed_chunks={counters['consumed']} "
                f"pending_chunks={event.get('pending_chunks', 0)} "
                f"buffered_chunks={event.get('buffered_chunks', 0)}"
            ),
            file=sys.stderr,
            flush=True,
        )

    return callback


def _collect_download_stats(
    *,
    source_id: str,
    download_stats: dict[str, dict[str, Any]],
    progress_callback: Callable[[dict], None] | None,
    lock: threading.Lock | None = None,
) -> Callable[[dict], None]:
    def callback(event: dict) -> None:
        if progress_callback is not None:
            progress_callback(event)
        if event.get("event") == "source_complete":
            stats = {
                "logical_source_id": str(event.get("logical_source_id", source_id)),
                "scanned_documents": int(event.get("scanned_documents", 0)),
                "written_documents": int(event.get("written_documents", 0)),
                "estimated_tokens": int(event.get("estimated_tokens", 0)),
                "stop_reason": str(event.get("stop_reason", "")),
                "output_path": str(event.get("output_path", "")),
            }
            if event.get("download_num_shards") is not None:
                stats["download_num_shards"] = int(event["download_num_shards"])
            if event.get("download_shard_index") is not None:
                stats["download_shard_index"] = int(event["download_shard_index"])
            if lock is None:
                download_stats[source_id] = stats
            else:
                with lock:
                    download_stats[source_id] = stats

    return callback


def _write_blend_plan(plan: Mapping[str, Any], *, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{plan['manifest_name']}.blend-plan.json"
    output_path.write_text(json.dumps(dict(plan), indent=2) + "\n", encoding="utf-8")
    return output_path


def _tokenize_processed_records(
    *,
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str,
    target_shard_tokens: int,
    train_source_token_limits: Mapping[str, int],
    workers: int,
    chunk_records: int,
    max_pending_chunks: int,
    progress_callback: Callable[[dict], None] | None,
    tokenizer_loader: Callable[[str], Any] | None,
) -> dict[str, dict]:
    if tokenizer_loader is None:
        return tokenize_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            tokenizer_name=tokenizer_name,
            target_shard_tokens=target_shard_tokens,
            train_source_token_limits=train_source_token_limits,
            workers=workers,
            chunk_records=chunk_records,
            max_pending_chunks=max_pending_chunks,
            progress_callback=progress_callback,
        )
    return tokenize_records_by_split(
        input_dir=input_dir,
        output_dir=output_dir,
        tokenizer=tokenizer_loader(tokenizer_name),
        target_shard_tokens=target_shard_tokens,
        train_source_token_limits=train_source_token_limits,
        workers=workers,
        chunk_records=chunk_records,
        max_pending_chunks=max_pending_chunks,
        progress_callback=progress_callback,
    )


def _tokenized_index_summary(
    *,
    tokenized_dir: Path,
    indexes: Mapping[str, Mapping[str, Any]],
    validation_mode: str,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for split in sorted(indexes):
        index_path = tokenized_dir / split / "index.json"
        index = load_tokenized_index(index_path, validation_mode=validation_mode)
        summary[split] = {
            "path": str(index_path),
            "uri": index_path.resolve().as_uri(),
            "total_tokens": int(index["total_tokens"]),
            "shard_count": len(index["shards"]),
            "source_token_counts": dict(index.get("source_token_counts", {})),
        }
    return summary


def _prepared_manifest(
    *,
    config: PrepareDataConfig,
    manifest: Mapping[str, Any],
    tokenizer_name: str,
    raw_paths: list[str],
    postprocess_stats: Mapping[str, Any],
    pipeline_plan_path: Path,
    blend_plan_path: Path,
    train_source_token_limits: Mapping[str, int],
    download_token_budgets: Mapping[str, int],
    download_stats: Mapping[str, Mapping[str, Any]],
    processed_records_shuffled: bool,
    processed_dir: Path,
    shuffled_processed_dir: Path,
    tokenization_input_dir: Path,
    processed_shuffle_stats: Mapping[str, Any],
    tokenized_indexes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_name": manifest["name"],
        "source_manifest_path": str(config.manifest_path),
        "work_dir": str(config.work_dir),
        "tokenizer": tokenizer_name,
        "target_shard_tokens": config.target_shard_tokens,
        "validation_rate": config.validation_rate,
        "min_chars": config.min_chars,
        "max_chars": config.max_chars,
        "raw_paths": raw_paths,
        "postprocess_stats": dict(postprocess_stats),
        "processed_records_shuffled": bool(processed_records_shuffled),
        "processed_dir": str(processed_dir),
        "shuffled_processed_dir": str(shuffled_processed_dir),
        "tokenization_input_dir": str(tokenization_input_dir),
        "processed_shuffle_seed": int(config.processed_shuffle_seed),
        "processed_shuffle_bucket_count": _normalize_positive_int(
            config.processed_shuffle_bucket_count,
            field_name="processed_shuffle_bucket_count",
        ),
        "processed_shuffle_max_open_buckets": _normalize_positive_int(
            config.processed_shuffle_max_open_buckets,
            field_name="processed_shuffle_max_open_buckets",
        ),
        "processed_shuffle_stats": dict(processed_shuffle_stats),
        "pipeline_plan_path": str(pipeline_plan_path),
        "blend_plan_path": str(blend_plan_path),
        "train_source_token_limits": dict(train_source_token_limits),
        "download_token_overfetch_ratio": config.download_token_overfetch_ratio,
        "download_chars_per_token": config.download_chars_per_token,
        "download_workers": _normalize_download_workers(
            requested_workers=config.download_workers,
        ),
        "download_shards_per_source": _normalize_download_shards_per_source(
            config.download_shards_per_source
        ),
        "download_source_parallelism": _normalize_download_source_parallelism(
            config.download_source_parallelism
        ),
        "postprocess_workers": _resolve_stage_workers(
            requested_workers=config.postprocess_workers,
            default_workers=_normalize_download_workers(
                requested_workers=config.download_workers,
            ),
            field_name="postprocess_workers",
        ),
        "tokenize_workers": _resolve_stage_workers(
            requested_workers=config.tokenize_workers,
            default_workers=_normalize_download_workers(
                requested_workers=config.download_workers,
            ),
            field_name="tokenize_workers",
        ),
        "tokenize_chunk_records": _normalize_positive_int(
            config.tokenize_chunk_records,
            field_name="tokenize_chunk_records",
        ),
        "tokenize_max_pending_chunks": _resolve_tokenize_max_pending_chunks(
            config.tokenize_max_pending_chunks,
            workers=_resolve_stage_workers(
                requested_workers=config.tokenize_workers,
                default_workers=_normalize_download_workers(
                    requested_workers=config.download_workers,
                ),
                field_name="tokenize_workers",
            ),
        ),
        "tokenize_progress_every_chunks": _normalize_positive_int(
            config.tokenize_progress_every_chunks,
            field_name="tokenize_progress_every_chunks",
        ),
        "resume_downloads": config.resume_downloads,
        "download_token_budgets": dict(download_token_budgets),
        "download_stats": {
            source_id: dict(stats)
            for source_id, stats in download_stats.items()
        },
        "tokenized_indexes": dict(tokenized_indexes),
        "api_runtime_overrides": _api_runtime_overrides(tokenized_indexes),
    }


def _train_source_token_limits(blend_plan: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(source["id"]): int(source["target_tokens"])
        for source in blend_plan["sources"]
    }


def _download_token_budgets(
    blend_plan: Mapping[str, Any],
    *,
    overfetch_ratio: float,
) -> dict[str, int]:
    if overfetch_ratio <= 0:
        return {}
    return {
        str(source["id"]): max(1, int(int(source["target_tokens"]) * overfetch_ratio))
        for source in blend_plan["sources"]
    }


def _api_runtime_overrides(
    tokenized_indexes: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    output: dict[str, str] = {}
    train = tokenized_indexes.get("train")
    validation = tokenized_indexes.get("validation")
    if train:
        output["SCALING_TOKENIZED_TRAIN_INDEX_URI"] = str(train["uri"])
    if validation:
        output["SCALING_TOKENIZED_VALIDATION_INDEX_URI"] = str(validation["uri"])
    return output


if __name__ == "__main__":
    raise SystemExit(main())
