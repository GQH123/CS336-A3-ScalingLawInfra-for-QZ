from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Iterator

from scaling_data.blend import build_blend_plan, load_manifest
from scaling_data.io import iter_jsonl_records, write_jsonl_record
from scaling_data.text_processing import clean_text, document_hash


@dataclass
class _ResumeSnapshot:
    seen_hashes: set[str]
    written_documents: int
    estimated_tokens: int


@dataclass(frozen=True)
class DownloadTask:
    source: dict
    download_id: str
    logical_source_id: str
    max_docs: int | None
    max_estimated_tokens: int | None


@dataclass(frozen=True)
class _SourceDownloadPlan:
    source_index: int
    total_sources: int
    source_id: str
    source_location: str
    tasks: list[DownloadTask]


@dataclass
class _ActiveSourceDownload:
    plan: _SourceDownloadPlan
    next_task_index: int = 0
    running_tasks: int = 0


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


def iter_source_rows(source: dict, *, skip_rows: int = 0) -> Iterator[dict]:
    if skip_rows < 0:
        raise ValueError("skip_rows must be non-negative")
    local_path = source.get("local_path")
    if isinstance(local_path, str) and local_path.strip():
        rows = _apply_row_shard(iter_jsonl_records(Path(local_path)), source=source)
        yield from _skip_rows(rows, skip_rows=skip_rows)
        return
    yield from iter_huggingface_rows(source, skip_rows=skip_rows)


def iter_huggingface_rows(source: dict, *, skip_rows: int = 0) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for Hugging Face downloads") from exc

    kwargs = {
        "path": source["hf_path"],
        "split": source.get("split", "train"),
        "streaming": True,
    }
    if source.get("hf_name") is not None:
        kwargs["name"] = source["hf_name"]
    if source.get("hf_data_dir") is not None:
        kwargs["data_dir"] = source["hf_data_dir"]
    if source.get("hf_data_files") is not None:
        kwargs["data_files"] = source["hf_data_files"]
    if source.get("hf_revision") is not None:
        kwargs["revision"] = source["hf_revision"]
    if source.get("hf_trust_remote_code") is not None:
        kwargs["trust_remote_code"] = bool(source["hf_trust_remote_code"])

    dataset = load_dataset(**kwargs)
    shard_count, shard_index = _download_shard_spec(source)
    if shard_count > 1:
        shard = getattr(dataset, "shard", None)
        if callable(shard):
            dataset = shard(num_shards=shard_count, index=shard_index)
        else:
            dataset = _shard_rows(
                dataset,
                num_shards=shard_count,
                shard_index=shard_index,
            )
    if skip_rows:
        skip = getattr(dataset, "skip", None)
        if callable(skip):
            dataset = skip(skip_rows)
        else:
            dataset = _skip_rows(dataset, skip_rows=skip_rows)
    yield from dataset


def _skip_rows(rows: Iterator[dict], *, skip_rows: int) -> Iterator[dict]:
    for index, row in enumerate(rows):
        if index < skip_rows:
            continue
        yield row


def _apply_row_shard(rows: Iterator[dict], *, source: dict) -> Iterator[dict]:
    shard_count, shard_index = _download_shard_spec(source)
    if shard_count == 1:
        yield from rows
        return
    yield from _shard_rows(rows, num_shards=shard_count, shard_index=shard_index)


def _shard_rows(
    rows: Iterator[dict],
    *,
    num_shards: int,
    shard_index: int,
) -> Iterator[dict]:
    for index, row in enumerate(rows):
        if index % num_shards == shard_index:
            yield row


def build_download_tasks(
    source: dict,
    *,
    shards_per_source: int,
    max_docs: int | None = None,
    max_estimated_tokens: int | None = None,
) -> list[DownloadTask]:
    shard_count = _source_shard_count(source, default=shards_per_source)
    logical_source_id = _logical_source_id(source)
    base_download_id = _download_artifact_id(source)
    doc_chunks = _split_optional_limit(max_docs, parts=shard_count)
    token_chunks = _split_optional_limit(max_estimated_tokens, parts=shard_count)
    tasks: list[DownloadTask] = []
    for shard_index in range(shard_count):
        task_source = dict(source)
        if shard_count == 1:
            download_id = base_download_id
        else:
            download_id = f"{base_download_id}.part{shard_index:03d}"
            task_source["download_id"] = download_id
            task_source["download_num_shards"] = shard_count
            task_source["download_shard_index"] = shard_index
        task_max_docs = None if doc_chunks is None else doc_chunks[shard_index]
        task_max_tokens = None
        if token_chunks is not None:
            task_max_tokens = token_chunks[shard_index]
            if task_max_tokens <= 0:
                task_max_docs = 0
                task_max_tokens = None
        tasks.append(
            DownloadTask(
                source=task_source,
                download_id=download_id,
                logical_source_id=logical_source_id,
                max_docs=task_max_docs,
                max_estimated_tokens=task_max_tokens,
            )
        )
    return tasks


def _source_shard_count(source: dict, *, default: int) -> int:
    value = source.get("download_shards")
    if value is None:
        value = default
    return _positive_int(value, field_name="download_shards")


def _download_shard_spec(source: dict) -> tuple[int, int]:
    shard_count = _positive_int(
        source.get("download_num_shards", 1),
        field_name="download_num_shards",
    )
    shard_index = _nonnegative_config_int(
        source.get("download_shard_index", 0),
        field_name="download_shard_index",
    )
    if shard_index >= shard_count:
        raise ValueError("download_shard_index must be smaller than download_num_shards")
    return shard_count, shard_index


def _positive_int(value, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _nonnegative_config_int(value, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _split_optional_limit(value: int | None, *, parts: int) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("download limits must be non-negative integers")
    parsed = int(value)
    if parsed < 0:
        raise ValueError("download limits must be non-negative integers")
    base, remainder = divmod(parsed, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def _logical_source_id(source: dict) -> str:
    return str(source["id"])


def _download_artifact_id(source: dict) -> str:
    raw = source.get("download_id", source["id"])
    value = str(raw).strip()
    if not value:
        raise ValueError("download_id must not be empty")
    return value


def download_source(
    *,
    source: dict,
    output_dir: Path,
    max_docs: int | None = None,
    max_estimated_tokens: int | None = None,
    chars_per_token: float = 4.0,
    min_chars: int = 200,
    progress_every_docs: int = 1_000,
    progress_callback: Callable[[dict], None] | None = None,
    resume: bool = True,
) -> Path:
    if max_estimated_tokens is not None and max_estimated_tokens <= 0:
        raise ValueError("max_estimated_tokens must be positive when provided")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    logical_source_id = _logical_source_id(source)
    source_id = _download_artifact_id(source)
    text_field = source.get("text_field", "text")
    source_location = _source_location(source)
    output_path = output_dir / f"{source_id}.jsonl.gz"
    resume_path, state_path = _resume_artifact_paths(output_dir, source_id)

    completed_state = _load_resume_state(state_path)
    if resume and completed_state and not _state_matches_source(
        completed_state,
        source_location=source_location,
        text_field=str(text_field),
    ):
        _clear_resume_artifacts(resume_path=resume_path, state_path=state_path)
        completed_state = {}
    if (
        resume
        and output_path.exists()
        and completed_state.get("status") == "completed"
        and _completed_state_matches_request(
            completed_state,
            max_docs=max_docs,
            max_estimated_tokens=max_estimated_tokens,
        )
    ):
        _emit_source_start(
            progress_callback=progress_callback,
            source=source,
            output_path=output_path,
            max_docs=max_docs,
            max_estimated_tokens=max_estimated_tokens,
            resume=resume,
            resume_path=resume_path,
            state_path=state_path,
            resume_existing_documents=_nonnegative_int(
                completed_state.get("written_documents")
            ),
            resume_skip_rows=_nonnegative_int(
                completed_state.get("scanned_documents")
            ),
        )
        _emit_source_complete(
            progress_callback=progress_callback,
            source_id=source_id,
            logical_source_id=logical_source_id,
            source=source,
            scanned=_nonnegative_int(completed_state.get("scanned_documents")),
            written=_nonnegative_int(completed_state.get("written_documents")),
            estimated_tokens=_nonnegative_int(
                completed_state.get("estimated_tokens")
            ),
            stop_reason="already_complete",
            output_path=output_path,
        )
        return output_path

    if not resume:
        _clear_resume_artifacts(resume_path=resume_path, state_path=state_path)

    resume_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = (
        _load_resume_snapshot(resume_path, chars_per_token=chars_per_token)
        if resume
        else _ResumeSnapshot(set(), 0, 0)
    )
    seen_hashes = snapshot.seen_hashes
    resume_state = _load_resume_state(state_path) if resume else {}
    resume_skip_rows = _resume_safe_skip_rows(resume_state) if resume else 0
    if resume_skip_rows and (
        _nonnegative_int(resume_state.get("written_documents"))
        > snapshot.written_documents
    ):
        resume_skip_rows = 0
    scanned = resume_skip_rows
    resume_safe_scanned = resume_skip_rows
    written = snapshot.written_documents
    estimated_tokens = snapshot.estimated_tokens
    stop_reason = "source_exhausted"
    _emit_source_start(
        progress_callback=progress_callback,
        source=source,
        output_path=output_path,
        max_docs=max_docs,
        max_estimated_tokens=max_estimated_tokens,
        resume=resume,
        resume_path=resume_path,
        state_path=state_path,
        resume_existing_documents=written,
        resume_skip_rows=resume_skip_rows,
    )

    existing_stop_reason = _limit_stop_reason(
        written=written,
        estimated_tokens=estimated_tokens,
        max_docs=max_docs,
        max_estimated_tokens=max_estimated_tokens,
    )
    if existing_stop_reason is not None:
        return _finalize_resumed_source(
            progress_callback=progress_callback,
            source_id=source_id,
            logical_source_id=logical_source_id,
            source=source,
            resume_path=resume_path,
            state_path=state_path,
            output_path=output_path,
            source_location=source_location,
            text_field=str(text_field),
            scanned=scanned,
            written=written,
            estimated_tokens=estimated_tokens,
            stop_reason=existing_stop_reason,
        )

    _write_resume_state(
        state_path=state_path,
        source_id=source_id,
        logical_source_id=logical_source_id,
        output_path=output_path,
        resume_path=resume_path,
        source_location=source_location,
        text_field=str(text_field),
        status="in_progress",
        scanned=scanned,
        resume_safe_scanned=resume_safe_scanned,
        written=written,
        estimated_tokens=estimated_tokens,
        stop_reason="",
    )

    with resume_path.open("a", encoding="utf-8") as handle:
        try:
            for row in iter_source_rows(source, skip_rows=resume_skip_rows):
                scanned += 1
                if text_field not in row:
                    raise _missing_text_field_error(
                        source_id=source_id,
                        text_field=str(text_field),
                        row=row,
                    )
                raw_text = row.get(text_field)
                if not isinstance(raw_text, str):
                    resume_safe_scanned = scanned
                    _checkpoint_resume_if_needed(
                        handle=handle,
                        state_path=state_path,
                        source_id=source_id,
                        logical_source_id=logical_source_id,
                        output_path=output_path,
                        resume_path=resume_path,
                        source_location=source_location,
                        text_field=str(text_field),
                        progress_every_docs=progress_every_docs,
                        scanned=scanned,
                        resume_safe_scanned=resume_safe_scanned,
                        written=written,
                        estimated_tokens=estimated_tokens,
                    )
                    _maybe_emit_progress(
                        progress_callback=progress_callback,
                        progress_every_docs=progress_every_docs,
                        source_id=source_id,
                        logical_source_id=logical_source_id,
                        scanned=scanned,
                        written=written,
                        estimated_tokens=estimated_tokens,
                    )
                    continue
                cleaned = clean_text(raw_text)
                if len(cleaned) < min_chars:
                    resume_safe_scanned = scanned
                    _checkpoint_resume_if_needed(
                        handle=handle,
                        state_path=state_path,
                        source_id=source_id,
                        logical_source_id=logical_source_id,
                        output_path=output_path,
                        resume_path=resume_path,
                        source_location=source_location,
                        text_field=str(text_field),
                        progress_every_docs=progress_every_docs,
                        scanned=scanned,
                        resume_safe_scanned=resume_safe_scanned,
                        written=written,
                        estimated_tokens=estimated_tokens,
                    )
                    _maybe_emit_progress(
                        progress_callback=progress_callback,
                        progress_every_docs=progress_every_docs,
                        source_id=source_id,
                        logical_source_id=logical_source_id,
                        scanned=scanned,
                        written=written,
                        estimated_tokens=estimated_tokens,
                    )
                    continue
                digest = document_hash(cleaned)
                if digest in seen_hashes:
                    resume_safe_scanned = scanned
                    _checkpoint_resume_if_needed(
                        handle=handle,
                        state_path=state_path,
                        source_id=source_id,
                        logical_source_id=logical_source_id,
                        output_path=output_path,
                        resume_path=resume_path,
                        source_location=source_location,
                        text_field=str(text_field),
                        progress_every_docs=progress_every_docs,
                        scanned=scanned,
                        resume_safe_scanned=resume_safe_scanned,
                        written=written,
                        estimated_tokens=estimated_tokens,
                    )
                    _maybe_emit_progress(
                        progress_callback=progress_callback,
                        progress_every_docs=progress_every_docs,
                        source_id=source_id,
                        logical_source_id=logical_source_id,
                        scanned=scanned,
                        written=written,
                        estimated_tokens=estimated_tokens,
                    )
                    continue
                seen_hashes.add(digest)

                record = {
                    "text": cleaned,
                    "source_id": logical_source_id,
                    "document_hash": digest,
                    "text_preprocessed": True,
                }
                write_jsonl_record(handle, record)
                written += 1
                estimated_tokens += estimate_text_tokens(
                    cleaned,
                    chars_per_token=chars_per_token,
                )
                resume_safe_scanned = scanned
                _checkpoint_resume_if_needed(
                    handle=handle,
                    state_path=state_path,
                    source_id=source_id,
                    logical_source_id=logical_source_id,
                    output_path=output_path,
                    resume_path=resume_path,
                    source_location=source_location,
                    text_field=str(text_field),
                    progress_every_docs=progress_every_docs,
                    scanned=scanned,
                    resume_safe_scanned=resume_safe_scanned,
                    written=written,
                    estimated_tokens=estimated_tokens,
                )
                _maybe_emit_progress(
                    progress_callback=progress_callback,
                    progress_every_docs=progress_every_docs,
                    source_id=source_id,
                    logical_source_id=logical_source_id,
                    scanned=scanned,
                    written=written,
                    estimated_tokens=estimated_tokens,
                )
                stop_reason = _limit_stop_reason(
                    written=written,
                    estimated_tokens=estimated_tokens,
                    max_docs=max_docs,
                    max_estimated_tokens=max_estimated_tokens,
                ) or "source_exhausted"
                if stop_reason != "source_exhausted":
                    break
        except BaseException:
            exc_type, exc_value, _ = sys.exc_info()
            handle.flush()
            _write_resume_state(
                state_path=state_path,
                source_id=source_id,
                logical_source_id=logical_source_id,
                output_path=output_path,
                resume_path=resume_path,
                source_location=source_location,
                text_field=str(text_field),
                status="interrupted",
                scanned=scanned,
                resume_safe_scanned=resume_safe_scanned,
                written=written,
                estimated_tokens=estimated_tokens,
                stop_reason="interrupted",
                error_type="" if exc_type is None else exc_type.__name__,
                error_message=_truncate_error_message(exc_value),
            )
            raise

    return _finalize_resumed_source(
        progress_callback=progress_callback,
        source_id=source_id,
        logical_source_id=logical_source_id,
        source=source,
        resume_path=resume_path,
        state_path=state_path,
        output_path=output_path,
        source_location=source_location,
        text_field=str(text_field),
        scanned=scanned,
        written=written,
        estimated_tokens=estimated_tokens,
        stop_reason=stop_reason,
    )


def write_queued_download_state(*, source: dict, output_dir: Path) -> None:
    source_id = _download_artifact_id(source)
    logical_source_id = _logical_source_id(source)
    text_field = str(source.get("text_field", "text"))
    source_location = _source_location(source)
    output_path = output_dir / f"{source_id}.jsonl.gz"
    resume_path, state_path = _resume_artifact_paths(output_dir, source_id)
    if state_path.exists():
        return
    _write_resume_state(
        state_path=state_path,
        source_id=source_id,
        logical_source_id=logical_source_id,
        output_path=output_path,
        resume_path=resume_path,
        source_location=source_location,
        text_field=text_field,
        status="queued",
        scanned=0,
        resume_safe_scanned=0,
        written=0,
        estimated_tokens=0,
        stop_reason="",
    )


def estimate_text_tokens(text: str, *, chars_per_token: float = 4.0) -> int:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return max(1, int(len(text) / chars_per_token))


def _missing_text_field_error(
    *,
    source_id: str,
    text_field: str,
    row: dict,
) -> RuntimeError:
    keys = sorted(str(key) for key in row.keys())
    shown_keys = keys[:20]
    suffix = "" if len(keys) <= len(shown_keys) else f", ... +{len(keys) - len(shown_keys)} more"
    return RuntimeError(
        (
            f"source {source_id} text_field {text_field!r} is missing from a "
            f"streamed row; available row keys: {', '.join(shown_keys)}{suffix}"
        )
    )


def _resume_artifact_paths(output_dir: Path, source_id: str) -> tuple[Path, Path]:
    resume_dir = output_dir / ".resume"
    return resume_dir / f"{source_id}.jsonl", resume_dir / f"{source_id}.state.json"


def _clear_resume_artifacts(*, resume_path: Path, state_path: Path) -> None:
    resume_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)


def _load_resume_state(state_path: Path) -> dict:
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _resume_safe_skip_rows(state: dict) -> int:
    if "resume_safe_scanned_documents" not in state:
        return 0
    safe_scanned = _nonnegative_int(state.get("resume_safe_scanned_documents"))
    scanned = _nonnegative_int(state.get("scanned_documents"))
    if scanned and safe_scanned > scanned:
        return 0
    return safe_scanned


def _completed_state_matches_request(
    state: dict,
    *,
    max_docs: int | None,
    max_estimated_tokens: int | None,
) -> bool:
    stop_reason = str(state.get("stop_reason", ""))
    written = _nonnegative_int(state.get("written_documents"))
    estimated_tokens = _nonnegative_int(state.get("estimated_tokens"))

    if max_docs is not None and written > max_docs:
        return False
    if stop_reason == "source_exhausted":
        return True
    if stop_reason == "max_docs":
        return max_docs is not None and written == max_docs
    if stop_reason == "estimated_token_budget":
        return (
            max_estimated_tokens is not None
            and estimated_tokens >= max_estimated_tokens
        )
    return False


def _state_matches_source(
    state: dict,
    *,
    source_location: str,
    text_field: str,
) -> bool:
    if state.get("source_location") != source_location:
        return False
    stored_text_field = state.get("text_field")
    if stored_text_field is None:
        return True
    return stored_text_field == text_field


def _load_resume_snapshot(
    resume_path: Path,
    *,
    chars_per_token: float,
) -> _ResumeSnapshot:
    seen_hashes: set[str] = set()
    written = 0
    estimated_tokens = 0
    if not resume_path.exists():
        return _ResumeSnapshot(seen_hashes, written, estimated_tokens)

    good_end = 0
    with resume_path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            parsed = _resume_record_text_and_hash(record)
            if parsed is None:
                good_end = handle.tell()
                continue
            text, digest = parsed
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                written += 1
                estimated_tokens += estimate_text_tokens(
                    text,
                    chars_per_token=chars_per_token,
                )
            good_end = handle.tell()

    if good_end < resume_path.stat().st_size:
        with resume_path.open("ab") as handle:
            handle.truncate(good_end)

    return _ResumeSnapshot(seen_hashes, written, estimated_tokens)


def _resume_record_text_and_hash(record: dict) -> tuple[str, str] | None:
    raw_text = record.get("text")
    if not isinstance(raw_text, str):
        return None
    digest = record.get("document_hash")
    if record.get("text_preprocessed") is True and _is_sha256_hex(digest):
        return raw_text, str(digest)
    cleaned = clean_text(raw_text)
    return cleaned, document_hash(cleaned)


def _is_sha256_hex(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _limit_stop_reason(
    *,
    written: int,
    estimated_tokens: int,
    max_docs: int | None,
    max_estimated_tokens: int | None,
) -> str | None:
    if max_docs is not None and written >= max_docs:
        return "max_docs"
    if max_estimated_tokens is not None and estimated_tokens >= max_estimated_tokens:
        return "estimated_token_budget"
    return None


def _checkpoint_resume_if_needed(
    *,
    handle,
    state_path: Path,
    source_id: str,
    logical_source_id: str,
    output_path: Path,
    resume_path: Path,
    source_location: str,
    text_field: str,
    progress_every_docs: int,
    scanned: int,
    resume_safe_scanned: int,
    written: int,
    estimated_tokens: int,
) -> None:
    if progress_every_docs <= 0 or scanned % progress_every_docs != 0:
        return
    handle.flush()
    _write_resume_state(
        state_path=state_path,
        source_id=source_id,
        logical_source_id=logical_source_id,
        output_path=output_path,
        resume_path=resume_path,
        source_location=source_location,
        text_field=text_field,
        status="in_progress",
        scanned=scanned,
        resume_safe_scanned=resume_safe_scanned,
        written=written,
        estimated_tokens=estimated_tokens,
        stop_reason="",
    )


def _write_resume_state(
    *,
    state_path: Path,
    source_id: str,
    logical_source_id: str,
    output_path: Path,
    resume_path: Path,
    source_location: str,
    text_field: str,
    status: str,
    scanned: int,
    resume_safe_scanned: int,
    written: int,
    estimated_tokens: int,
    stop_reason: str,
    error_type: str = "",
    error_message: str = "",
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_id": source_id,
        "logical_source_id": logical_source_id,
        "source_location": source_location,
        "text_field": text_field,
        "status": status,
        "scanned_documents": scanned,
        "resume_safe_scanned_documents": resume_safe_scanned,
        "written_documents": written,
        "estimated_tokens": estimated_tokens,
        "stop_reason": stop_reason,
        "output_path": str(output_path),
        "resume_path": str(resume_path),
    }
    if error_type:
        payload["error_type"] = error_type
    if error_message:
        payload["error_message"] = error_message
    tmp_path = state_path.with_name(f".{state_path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(state_path)


def _finalize_resumed_source(
    *,
    progress_callback: Callable[[dict], None] | None,
    source_id: str,
    logical_source_id: str,
    source: dict | None,
    resume_path: Path,
    state_path: Path,
    output_path: Path,
    source_location: str,
    text_field: str,
    scanned: int,
    written: int,
    estimated_tokens: int,
    stop_reason: str,
) -> Path:
    _compress_resume_to_output(resume_path=resume_path, output_path=output_path)
    _write_resume_state(
        state_path=state_path,
        source_id=source_id,
        logical_source_id=logical_source_id,
        output_path=output_path,
        resume_path=resume_path,
        source_location=source_location,
        text_field=text_field,
        status="completed",
        scanned=scanned,
        resume_safe_scanned=scanned,
        written=written,
        estimated_tokens=estimated_tokens,
        stop_reason=stop_reason,
    )
    resume_path.unlink(missing_ok=True)
    _emit_source_complete(
        progress_callback=progress_callback,
        source_id=source_id,
        logical_source_id=logical_source_id,
        source=source,
        scanned=scanned,
        written=written,
        estimated_tokens=estimated_tokens,
        stop_reason=stop_reason,
        output_path=output_path,
    )
    return output_path


def _compress_resume_to_output(*, resume_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.gz")
    with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=1) as handle:
        if resume_path.exists():
            for record in iter_jsonl_records(resume_path):
                write_jsonl_record(handle, record)
    tmp_path.replace(output_path)


def _emit_source_start(
    *,
    progress_callback: Callable[[dict], None] | None,
    source: dict,
    output_path: Path,
    max_docs: int | None,
    max_estimated_tokens: int | None,
    resume: bool,
    resume_path: Path,
    state_path: Path,
    resume_existing_documents: int,
    resume_skip_rows: int,
) -> None:
    _emit_progress(
        progress_callback,
        {
            "event": "source_start",
            "source_id": _download_artifact_id(source),
            "logical_source_id": _logical_source_id(source),
            "source_location": _source_location(source),
            "output_path": str(output_path),
            "max_docs": max_docs,
            "max_estimated_tokens": max_estimated_tokens,
            "resume": resume,
            "resume_path": str(resume_path),
            "resume_state_path": str(state_path),
            "resume_existing_documents": resume_existing_documents,
            "resume_skip_rows": resume_skip_rows,
            "download_num_shards": source.get("download_num_shards"),
            "download_shard_index": source.get("download_shard_index"),
        },
    )


def _emit_source_complete(
    *,
    progress_callback: Callable[[dict], None] | None,
    source_id: str,
    logical_source_id: str,
    source: dict | None,
    scanned: int,
    written: int,
    estimated_tokens: int,
    stop_reason: str,
    output_path: Path,
) -> None:
    _emit_progress(
        progress_callback,
        {
            "event": "source_complete",
            "source_id": source_id,
            "logical_source_id": logical_source_id,
            "scanned_documents": scanned,
            "written_documents": written,
            "estimated_tokens": estimated_tokens,
            "stop_reason": stop_reason,
            "output_path": str(output_path),
            "download_num_shards": None if source is None else source.get("download_num_shards"),
            "download_shard_index": None if source is None else source.get("download_shard_index"),
        },
    )


def _nonnegative_int(value) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _maybe_emit_progress(
    *,
    progress_callback: Callable[[dict], None] | None,
    progress_every_docs: int,
    source_id: str,
    logical_source_id: str,
    scanned: int,
    written: int,
    estimated_tokens: int,
) -> None:
    if progress_every_docs <= 0:
        return
    if scanned % progress_every_docs != 0:
        return
    _emit_progress(
        progress_callback,
        {
            "event": "source_progress",
            "source_id": source_id,
            "logical_source_id": logical_source_id,
            "scanned_documents": scanned,
            "written_documents": written,
            "estimated_tokens": estimated_tokens,
        },
    )


def _emit_progress(
    progress_callback: Callable[[dict], None] | None,
    event: dict,
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _source_location(source: dict) -> str:
    local_path = source.get("local_path")
    if isinstance(local_path, str) and local_path.strip():
        base_location = local_path
    else:
        hf_path = source.get("hf_path")
        hf_name = source.get("hf_name")
        if hf_name is not None:
            base_location = f"{hf_path}:{hf_name}"
        else:
            base_location = str(hf_path)
        loader_parts = []
        for source_key in (
            "hf_data_dir",
            "hf_data_files",
            "hf_revision",
            "hf_trust_remote_code",
        ):
            value = source.get(source_key)
            if value is not None:
                serialized = json.dumps(value, sort_keys=True)
                loader_parts.append(f"{source_key}={serialized}")
        if loader_parts:
            base_location = f"{base_location}?" + "&".join(loader_parts)
    shard_count, shard_index = _download_shard_spec(source)
    if shard_count == 1:
        return base_location
    return f"{base_location}#download_shard={shard_index}/{shard_count}"


def _truncate_error_message(value, *, limit: int = 2000) -> str:
    message = "" if value is None else str(value)
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def print_progress_event(event: dict) -> None:
    event_name = event.get("event")
    source_id = event.get("source_id", "unknown")
    logical_source_id = event.get("logical_source_id")
    logical_suffix = (
        ""
        if not logical_source_id or logical_source_id == source_id
        else f" logical_source={logical_source_id}"
    )
    if event_name == "source_start":
        print(
            (
                f"[download] start source={source_id}{logical_suffix} "
                f"location={event.get('source_location')} "
                f"max_docs={event.get('max_docs')} "
                f"max_est_tokens={event.get('max_estimated_tokens')} "
                f"resume={event.get('resume')} "
                f"resume_existing={event.get('resume_existing_documents')} "
                f"resume_skip_rows={event.get('resume_skip_rows')} "
                f"output={event.get('output_path')}"
            ),
            file=sys.stderr,
            flush=True,
        )
    elif event_name == "source_progress":
        print(
            (
                f"[download] progress source={source_id}{logical_suffix} "
                f"scanned={event.get('scanned_documents')} "
                f"written={event.get('written_documents')} "
                f"est_tokens={event.get('estimated_tokens')}"
            ),
            file=sys.stderr,
            flush=True,
        )
    elif event_name == "source_complete":
        print(
            (
                f"[download] complete source={source_id}{logical_suffix} "
                f"scanned={event.get('scanned_documents')} "
                f"written={event.get('written_documents')} "
                f"est_tokens={event.get('estimated_tokens')} "
                f"stop_reason={event.get('stop_reason')} "
                f"output={event.get('output_path')}"
            ),
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean manifest sources.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-docs-per-source", type=int)
    parser.add_argument("--download-token-overfetch-ratio", type=float, default=0.0)
    parser.add_argument("--download-chars-per-token", type=float, default=4.0)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--download-shards-per-source", type=int, default=4)
    parser.add_argument("--download-source-parallelism", type=int, default=1)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--progress-every-docs", type=int, default=1_000)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing resume checkpoints and rebuild each source from scratch.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    plan = build_blend_plan(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    workers = _normalize_download_workers(requested_workers=args.download_workers)
    shards_per_source = _positive_int(
        args.download_shards_per_source,
        field_name="download_shards_per_source",
    )
    source_parallelism = _normalize_download_source_parallelism(
        args.download_source_parallelism
    )

    progress_lock = threading.Lock()

    def locked_progress_callback(event: dict) -> None:
        with progress_lock:
            print_progress_event(event)

    source_plans = _build_source_plans(
        sources=plan["sources"],
        shards_per_source=shards_per_source,
        max_docs_per_source=args.max_docs_per_source,
        download_token_overfetch_ratio=args.download_token_overfetch_ratio,
    )
    for source_plan in source_plans:
        for task in source_plan.tasks:
            write_queued_download_state(source=task.source, output_dir=args.output_dir)
    _download_source_plans(
        source_plans=source_plans,
        output_dir=args.output_dir,
        args=args,
        workers=workers,
        active_source_limit=min(workers, source_parallelism),
        progress_lock=progress_lock,
        locked_progress_callback=locked_progress_callback,
    )


def _build_source_plans(
    *,
    sources: Sequence[dict],
    shards_per_source: int,
    max_docs_per_source: int | None,
    download_token_overfetch_ratio: float,
) -> list[_SourceDownloadPlan]:
    total_sources = len(sources)
    source_plans: list[_SourceDownloadPlan] = []
    for source_index, source in enumerate(sources, start=1):
        source_location = source.get("local_path") or source.get("hf_path")
        tasks = build_download_tasks(
            source=source,
            shards_per_source=shards_per_source,
            max_docs=max_docs_per_source,
            max_estimated_tokens=_download_token_budget(
                source,
                overfetch_ratio=download_token_overfetch_ratio,
            ),
        )
        source_plans.append(
            _SourceDownloadPlan(
                source_index=source_index,
                total_sources=total_sources,
                source_id=str(source["id"]),
                source_location=str(source_location),
                tasks=tasks,
            )
        )
    return source_plans


def _download_source_plans(
    *,
    source_plans: Sequence[_SourceDownloadPlan],
    output_dir: Path,
    args: argparse.Namespace,
    workers: int,
    active_source_limit: int,
    progress_lock: threading.Lock,
    locked_progress_callback: Callable[[dict], None],
) -> None:
    download_failures: list[_DownloadFailure] = []
    if active_source_limit <= 1:
        _download_source_plans_sequentially(
            source_plans=source_plans,
            output_dir=output_dir,
            args=args,
            workers=workers,
            download_failures=download_failures,
            progress_lock=progress_lock,
            locked_progress_callback=locked_progress_callback,
        )
    else:
        _download_source_plans_with_source_parallelism(
            source_plans=source_plans,
            output_dir=output_dir,
            args=args,
            workers=workers,
            active_source_limit=active_source_limit,
            download_failures=download_failures,
            progress_lock=progress_lock,
            locked_progress_callback=locked_progress_callback,
        )

    if download_failures:
        raise DownloadStageError(download_failures)


def _download_source_plans_sequentially(
    *,
    source_plans: Sequence[_SourceDownloadPlan],
    output_dir: Path,
    args: argparse.Namespace,
    workers: int,
    download_failures: list[_DownloadFailure],
    progress_lock: threading.Lock,
    locked_progress_callback: Callable[[dict], None],
) -> None:
    for source_plan in source_plans:
        source_workers = min(workers, len(source_plan.tasks))
        with progress_lock:
            print(
                (
                    f"downloading source {source_plan.source_index}/"
                    f"{source_plan.total_sources} {source_plan.source_id} "
                    f"from {source_plan.source_location} "
                    f"with {source_workers} worker(s), "
                    f"{len(source_plan.tasks)} shard(s)"
                )
        )
        if source_workers <= 1:
            for task in source_plan.tasks:
                try:
                    output_path = _download_source_from_args(
                        task=task,
                        output_dir=output_dir,
                        args=args,
                        progress_callback=print_progress_event,
                    )
                except Exception as exc:
                    download_failures.append(_download_failure(task, exc))
                else:
                    print(f"wrote {output_path}")
            continue

        with ProcessPoolExecutor(max_workers=source_workers) as executor:
            futures = {
                executor.submit(
                    _download_source_from_args,
                    task=task,
                    output_dir=output_dir,
                    args=args,
                    progress_callback=None,
                ): task
                for task in source_plan.tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    output_path = future.result()
                except Exception as exc:
                    download_failures.append(_download_failure(task, exc))
                else:
                    with progress_lock:
                        print(f"wrote {output_path}")


def _download_source_plans_with_source_parallelism(
    *,
    source_plans: Sequence[_SourceDownloadPlan],
    output_dir: Path,
    args: argparse.Namespace,
    workers: int,
    active_source_limit: int,
    download_failures: list[_DownloadFailure],
    progress_lock: threading.Lock,
    locked_progress_callback: Callable[[dict], None],
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
        with progress_lock:
            print(
                (
                    f"activating source {plan.source_index}/{plan.total_sources} "
                    f"{plan.source_id} from {plan.source_location} "
                    f"with {len(plan.tasks)} shard(s); "
                    f"active_sources={len(active_sources)}/{active_source_limit} "
                    f"worker_pool={workers}"
                )
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
                        _download_source_from_args,
                        task=task,
                        output_dir=output_dir,
                        args=args,
                        progress_callback=None,
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
                    output_path = future.result()
                except Exception as exc:
                    download_failures.append(_download_failure(task, exc))
                else:
                    with progress_lock:
                        print(f"wrote {output_path}")
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


def _download_source_from_args(
    *,
    task: DownloadTask,
    output_dir: Path,
    args: argparse.Namespace,
    progress_callback: Callable[[dict], None] | None = None,
) -> Path:
    return download_source(
        source=task.source,
        output_dir=output_dir,
        max_docs=task.max_docs,
        max_estimated_tokens=task.max_estimated_tokens,
        chars_per_token=args.download_chars_per_token,
        min_chars=args.min_chars,
        progress_every_docs=args.progress_every_docs,
        progress_callback=progress_callback,
        resume=not args.no_resume,
    )


def _download_failure(task: DownloadTask, exc: Exception) -> _DownloadFailure:
    return _DownloadFailure(
        download_id=task.download_id,
        logical_source_id=task.logical_source_id,
        error_type=type(exc).__name__,
        error_message=_truncate_error_message(exc),
    )


def _normalize_download_workers(
    *,
    requested_workers: int,
) -> int:
    return _positive_int(requested_workers, field_name="download_workers")


def _normalize_download_source_parallelism(value: int) -> int:
    return _positive_int(value, field_name="download_source_parallelism")


def _download_token_budget(source: dict, *, overfetch_ratio: float) -> int | None:
    if overfetch_ratio <= 0:
        return None
    target_tokens = source.get("target_tokens")
    if target_tokens is None:
        return None
    return max(1, int(int(target_tokens) * overfetch_ratio))


if __name__ == "__main__":
    main()
