from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import signal
import sys
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urljoin, urlparse

import requests


class WorkerManifestError(ValueError):
    pass


class WorkerResultError(ValueError):
    pass


class TokenizedIndexError(ValueError):
    pass


_REQUIRED_MANIFEST_KEYS = (
    "student_id",
    "model_config",
    "data_config",
    "validation_config",
    "runtime_config",
)


def load_manifest(
    manifest_source: str,
    *,
    session=None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    source = manifest_source.strip()
    if not source:
        raise WorkerManifestError("manifest source is empty")

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        raw = _fetch_manifest(source, session=session, timeout_seconds=timeout_seconds)
    elif parsed.scheme in {"s3", "gs"}:
        raise WorkerManifestError(f"unsupported manifest URI scheme: {parsed.scheme}")
    elif source.startswith("file://"):
        parsed = urlparse(source)
        if parsed.netloc not in {"", "localhost"}:
            raise WorkerManifestError(f"unsupported file URI host: {parsed.netloc}")
        raw = Path(unquote(parsed.path)).read_text(encoding="utf-8")
    elif source.startswith(("{", "[")):
        raw = source
    else:
        path = Path(source)
        try:
            path_exists = path.exists()
        except OSError:
            path_exists = False
        if path_exists:
            raw = path.read_text(encoding="utf-8")
        else:
            raw = source

    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkerManifestError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise WorkerManifestError("manifest must be a JSON object")

    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise WorkerManifestError(
            "manifest missing required fields: " + ", ".join(missing)
        )
    _manifest_identity(manifest)
    _validate_validation_config(manifest["validation_config"])
    return manifest


def _fetch_manifest(source: str, *, session=None, timeout_seconds: int = 30) -> str:
    http = session or requests.Session()
    try:
        response = http.get(source, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise WorkerManifestError(f"manifest fetch failed: {exc}") from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise WorkerManifestError(
            f"manifest fetch failed with HTTP {response.status_code}: {response.text}"
        )
    return response.text


class WorkerCallbackClient:
    def __init__(
        self,
        *,
        callback_url: str,
        callback_token: str,
        session=None,
        timeout_seconds: int = 30,
    ):
        self.callback_url = callback_url
        self.callback_token = callback_token
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def post_event(self, payload: Mapping[str, Any]) -> None:
        headers = {}
        if self.callback_token:
            headers["Authorization"] = f"Bearer {self.callback_token}"
        response = self.session.post(
            self.callback_url,
            json=dict(payload),
            headers=headers,
            timeout=self.timeout_seconds,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(
                f"callback failed with HTTP {response.status_code}: {response.text}"
            )


class WorkerJsonlEventWriter:
    def __init__(self, *, event_log_path: str | Path):
        self.event_log_path = Path(event_log_path)

    def post_event(self, payload: Mapping[str, Any]) -> None:
        if not str(self.event_log_path).strip():
            raise RuntimeError("event_log_path is required for jsonl event sink")
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dict(payload), sort_keys=True) + "\n"
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)


class WorkerCompositeEventSink:
    def __init__(self, sinks: list[Any]):
        self.sinks = sinks

    def post_event(self, payload: Mapping[str, Any]) -> None:
        for sink in self.sinks:
            sink.post_event(payload)


Trainer = Callable[
    [Mapping[str, Any], Callable[[Mapping[str, Any]], None]],
    Mapping[str, Any],
]


def load_trainer(spec: str) -> Trainer:
    value = spec.strip()
    if ":" not in value:
        raise WorkerManifestError("trainer spec must use module:function format")
    module_name, function_name = value.split(":", 1)
    if not module_name or not function_name:
        raise WorkerManifestError("trainer spec must use module:function format")
    module = importlib.import_module(module_name)
    trainer = getattr(module, function_name)
    if not callable(trainer):
        raise WorkerManifestError(f"trainer is not callable: {value}")
    return trainer


def run_worker(
    *,
    manifest: Mapping[str, Any],
    callback_url: str,
    callback_token: str,
    trainer: Trainer,
    session=None,
    heartbeat_interval_seconds: int = 0,
    thread_factory: Callable[..., Any] = threading.Thread,
    stop_event_factory: Callable[[], Any] = threading.Event,
    heartbeat_once: bool = False,
    runtime_timeout_factory: Callable[[int], Any] | None = None,
    tokenized_index_validation_mode: str | None = None,
    event_sink: str = "http",
    event_log_path: str | Path = "",
) -> int:
    event_client = _build_worker_event_sink(
        event_sink=event_sink,
        callback_url=callback_url,
        callback_token=callback_token,
        event_log_path=event_log_path,
        session=session,
    )
    identity_key, run_id = _manifest_identity(manifest)
    callback_lock = threading.Lock()
    event_sequence = 0

    def emit_event(payload: Mapping[str, Any]) -> None:
        nonlocal event_sequence
        event = {identity_key: run_id, **dict(payload)}
        with callback_lock:
            event_sequence += 1
            event.setdefault("event_id", f"{run_id}:{event_sequence:06d}")
            event_client.post_event(event)

    stop_event = None
    heartbeat_thread = None
    heartbeat_started = False

    if heartbeat_interval_seconds > 0:
        stop_event = stop_event_factory()

        def heartbeat_loop() -> None:
            while True:
                if stop_event.wait(heartbeat_interval_seconds):
                    return
                emit_event(
                    {
                        "event_type": "heartbeat",
                        "heartbeat_interval_seconds": heartbeat_interval_seconds,
                    }
                )
                if heartbeat_once:
                    return

        heartbeat_thread = thread_factory(target=heartbeat_loop, daemon=True)

    try:
        emit_event({"event_type": "worker_started"})
        _validate_manifest_tokenized_indexes(
            manifest,
            session=session,
            validation_mode=_tokenized_index_validation_mode(
                tokenized_index_validation_mode
            ),
        )
        runtime_limit_seconds = _manifest_runtime_limit(manifest)
        if heartbeat_thread is not None:
            heartbeat_thread.start()
            heartbeat_started = True
        timeout_context = (
            runtime_timeout_factory
            if runtime_timeout_factory is not None
            else _runtime_timeout
        )
        with timeout_context(runtime_limit_seconds):
            result = _normalize_trainer_result(trainer(manifest, emit_event))
        emit_event({"event_type": "worker_completed", **result})
        return 0
    except Exception as exc:
        print(
            f"worker failed for {identity_key}={run_id}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        sys.stderr.flush()
        try:
            emit_event(
                {
                    "event_type": "worker_failed",
                    "failure_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        except Exception as callback_exc:
            print(
                "failed to report worker_failed event: "
                f"{type(callback_exc).__name__}: {callback_exc}; "
                f"original failure: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return 1
    finally:
        if stop_event is not None:
            stop_event.set()
        if heartbeat_thread is not None and heartbeat_started:
            heartbeat_thread.join(timeout=5)


def _build_worker_event_sink(
    *,
    event_sink: str,
    callback_url: str,
    callback_token: str,
    event_log_path: str | Path,
    session=None,
):
    mode = str(event_sink or "http").strip()
    if mode == "http":
        return WorkerCallbackClient(
            callback_url=callback_url,
            callback_token=callback_token,
            session=session,
        )
    if mode == "jsonl":
        return WorkerJsonlEventWriter(event_log_path=event_log_path)
    if mode == "both":
        return WorkerCompositeEventSink(
            [
                WorkerCallbackClient(
                    callback_url=callback_url,
                    callback_token=callback_token,
                    session=session,
                ),
                WorkerJsonlEventWriter(event_log_path=event_log_path),
            ]
        )
    raise WorkerManifestError("event sink must be http, jsonl, or both")


def _missing_trainer(_manifest, _emit_event):
    raise RuntimeError(
        "No trainer implementation is configured. Wire the backend training loop "
        "before using this entrypoint in a production worker image."
    )


@contextmanager
def _runtime_timeout(seconds: int):
    if (
        seconds <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def raise_timeout(_signum, _frame):
        raise TimeoutError(f"worker runtime exceeded {seconds} seconds")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _manifest_runtime_limit(manifest: Mapping[str, Any]) -> int:
    if "max_runtime_seconds" in manifest:
        return _positive_manifest_int(
            manifest.get("max_runtime_seconds"),
            field_name="max_runtime_seconds",
        )
    runtime_config = manifest.get("runtime_config")
    if not isinstance(runtime_config, Mapping):
        raise WorkerManifestError("runtime_config must be a JSON object")
    if "max_runtime_seconds" in runtime_config:
        return _positive_manifest_int(
            runtime_config.get("max_runtime_seconds"),
            field_name="runtime_config.max_runtime_seconds",
        )
    return _positive_manifest_int(
        runtime_config.get("reserved_runtime_seconds"),
        field_name="runtime_config.reserved_runtime_seconds",
    )


def _validate_manifest_tokenized_indexes(
    manifest: Mapping[str, Any],
    *,
    session=None,
    validation_mode: str = "metadata",
) -> None:
    for section_name in ("data_config", "validation_config"):
        section = manifest.get(section_name)
        if not isinstance(section, Mapping):
            continue
        source = section.get("tokenized_index_uri")
        if source is None:
            continue
        if not isinstance(source, str) or not source.strip():
            raise TokenizedIndexError(
                f"{section_name}.tokenized_index_uri must be non-empty"
            )
        _load_tokenized_index(
            source.strip(),
            session=session,
            validation_mode=validation_mode,
        )


def _load_tokenized_index(
    source: str,
    *,
    session=None,
    timeout_seconds: int = 30,
    validation_mode: str = "metadata",
) -> dict[str, Any]:
    mode = _tokenized_index_validation_mode(validation_mode)
    raw, shard_reader, shard_sizer = _read_tokenized_index_source(
        source,
        session=session,
        timeout_seconds=timeout_seconds,
    )
    try:
        index = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TokenizedIndexError(f"tokenized index is not valid JSON: {exc}") from exc
    if not isinstance(index, dict):
        raise TokenizedIndexError("tokenized index must be a JSON object")
    shards = index.get("shards")
    if not isinstance(shards, list):
        raise TokenizedIndexError("tokenized index missing shards list")
    total_tokens = 0
    for shard_number, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise TokenizedIndexError("tokenized shard entry must be an object")
        total_tokens += _validate_tokenized_shard(
            shard,
            shard_reader=shard_reader,
            shard_sizer=shard_sizer,
            shard_number=shard_number,
            validation_mode=mode,
        )
    declared_total_tokens = index.get("total_tokens")
    if declared_total_tokens is not None:
        parsed_total = _positive_token_count(
            declared_total_tokens,
            field_name="total_tokens",
            allow_zero=True,
        )
        if parsed_total != total_tokens:
            raise TokenizedIndexError(
                "tokenized index total_tokens does not match shards"
            )
    return index


def _read_tokenized_index_source(
    source: str,
    *,
    session=None,
    timeout_seconds: int,
) -> tuple[
    str,
    Callable[[str], tuple[bytes, str]],
    Callable[[str], tuple[int, str] | None],
]:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        raw = _fetch_text(
            source,
            error_type=TokenizedIndexError,
            description="tokenized index",
            session=session,
            timeout_seconds=timeout_seconds,
        )

        def read_remote_shard(relative_path: str) -> tuple[bytes, str]:
            shard_url = urljoin(source, relative_path)
            return (
                _fetch_bytes(
                    shard_url,
                    error_type=TokenizedIndexError,
                    description="tokenized shard",
                    session=session,
                    timeout_seconds=timeout_seconds,
                ),
                shard_url,
            )

        def size_remote_shard(_relative_path: str) -> tuple[int, str] | None:
            return None

        return raw, read_remote_shard, size_remote_shard
    if parsed.scheme in {"s3", "gs"}:
        raise TokenizedIndexError(
            f"unsupported tokenized index URI scheme: {parsed.scheme}"
        )
    if source.startswith("file://"):
        if parsed.netloc not in {"", "localhost"}:
            raise TokenizedIndexError(f"unsupported file URI host: {parsed.netloc}")
        index_path = Path(unquote(parsed.path))
    elif parsed.scheme:
        raise TokenizedIndexError(
            f"unsupported tokenized index URI scheme: {parsed.scheme}"
        )
    else:
        index_path = Path(source)

    try:
        raw = index_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TokenizedIndexError(
            f"tokenized index is not readable: {index_path}"
        ) from exc

    base_dir = index_path.parent

    def read_local_shard(relative_path: str) -> tuple[bytes, str]:
        shard_path = base_dir / relative_path
        try:
            return shard_path.read_bytes(), str(shard_path)
        except OSError as exc:
            raise TokenizedIndexError(
                f"tokenized shard is not readable: {shard_path}"
            ) from exc

    def size_local_shard(relative_path: str) -> tuple[int, str] | None:
        shard_path = base_dir / relative_path
        try:
            return shard_path.stat().st_size, str(shard_path)
        except OSError as exc:
            raise TokenizedIndexError(
                f"tokenized shard is not readable: {shard_path}"
            ) from exc

    return raw, read_local_shard, size_local_shard


def _validate_tokenized_shard(
    shard: Mapping[str, Any],
    *,
    shard_reader: Callable[[str], tuple[bytes, str]],
    shard_sizer: Callable[[str], tuple[int, str] | None],
    shard_number: int,
    validation_mode: str,
) -> int:
    if shard.get("dtype") != "uint32":
        raise TokenizedIndexError("only uint32 token shards are supported")
    relative_path = _tokenized_shard_relative_path(
        shard.get("path"),
        field_name=f"shards[{shard_number}].path",
    )
    token_count = _positive_token_count(
        shard.get("tokens"),
        field_name=f"shards[{shard_number}].tokens",
    )
    expected_hash = shard.get("sha256")
    if not isinstance(expected_hash, str) or not expected_hash.strip():
        raise TokenizedIndexError(f"tokenized shard missing sha256: {relative_path}")
    if not _is_sha256_hex(expected_hash):
        raise TokenizedIndexError(f"tokenized shard sha256 is invalid: {relative_path}")
    byte_order = shard.get("byte_order", "little")
    if byte_order != "little":
        raise TokenizedIndexError("only little-endian uint32 token shards are supported")

    expected_size = token_count * 4
    if validation_mode == "metadata":
        sized = shard_sizer(relative_path)
        if sized is not None:
            shard_size, shard_label = sized
            if shard_size != expected_size:
                raise TokenizedIndexError(
                    f"tokenized shard byte size mismatch: {shard_label}"
                )
        return token_count

    shard_bytes, shard_label = shard_reader(relative_path)
    if len(shard_bytes) != expected_size:
        raise TokenizedIndexError(
            f"tokenized shard byte size mismatch: {shard_label}"
        )
    actual_hash = hashlib.sha256(shard_bytes).hexdigest()
    if actual_hash != expected_hash:
        raise TokenizedIndexError(f"tokenized shard hash mismatch: {shard_label}")
    return token_count


def _tokenized_index_validation_mode(value: str | None) -> str:
    mode = str(
        value
        if value is not None and str(value).strip()
        else os.environ.get("SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE", "metadata")
    ).strip()
    if mode not in {"metadata", "full"}:
        raise TokenizedIndexError(
            "tokenized index validation mode must be metadata or full"
        )
    return mode


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _tokenized_shard_relative_path(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TokenizedIndexError(f"{field_name} must be non-empty")
    raw_path = value.strip().replace("\\", "/")
    parsed = urlparse(raw_path)
    if parsed.scheme or parsed.netloc:
        raise TokenizedIndexError(f"{field_name} must be a relative path")
    if raw_path.startswith("/"):
        raise TokenizedIndexError(f"{field_name} must be a relative path")
    relative_path = PurePosixPath(raw_path)
    if str(relative_path) in {"", "."} or ".." in relative_path.parts:
        raise TokenizedIndexError(f"{field_name} must be a safe relative path")
    return str(relative_path)


def _positive_token_count(
    value: Any,
    *,
    field_name: str,
    allow_zero: bool = False,
) -> int:
    if isinstance(value, bool):
        raise TokenizedIndexError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TokenizedIndexError(f"{field_name} must be a positive integer") from exc
    if allow_zero:
        if parsed < 0:
            raise TokenizedIndexError(f"{field_name} must be non-negative")
    elif parsed <= 0:
        raise TokenizedIndexError(f"{field_name} must be a positive integer")
    return parsed


def _fetch_text(
    source: str,
    *,
    error_type: type[ValueError],
    description: str,
    session=None,
    timeout_seconds: int = 30,
) -> str:
    response = _get_http_response(
        source,
        error_type=error_type,
        description=description,
        session=session,
        timeout_seconds=timeout_seconds,
    )
    if response.status_code < 200 or response.status_code >= 300:
        raise error_type(
            f"{description} fetch failed with HTTP {response.status_code}: {response.text}"
        )
    return response.text


def _fetch_bytes(
    source: str,
    *,
    error_type: type[ValueError],
    description: str,
    session=None,
    timeout_seconds: int = 30,
) -> bytes:
    response = _get_http_response(
        source,
        error_type=error_type,
        description=description,
        session=session,
        timeout_seconds=timeout_seconds,
    )
    if response.status_code < 200 or response.status_code >= 300:
        raise error_type(
            f"{description} fetch failed with HTTP {response.status_code}: {response.text}"
        )
    content = getattr(response, "content", None)
    if content is not None:
        return bytes(content)
    return response.text.encode("utf-8")


def _get_http_response(
    source: str,
    *,
    error_type: type[ValueError],
    description: str,
    session=None,
    timeout_seconds: int = 30,
):
    http = session or requests.Session()
    try:
        return http.get(source, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise error_type(f"{description} fetch failed: {exc}") from exc


def _normalize_trainer_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise WorkerResultError("trainer result must be a mapping")
    validation_losses = _finite_float_list(
        result.get("validation_losses"),
        field_name="validation_losses",
    )
    if not validation_losses:
        raise WorkerResultError("validation_losses must be non-empty")
    final_loss = _finite_float(
        result.get("final_validation_loss"),
        field_name="final_validation_loss",
    )
    if final_loss != validation_losses[-1]:
        raise WorkerResultError(
            "final_validation_loss must equal validation_losses[-1]"
        )
    actual_runtime_seconds = _positive_int(
        result.get("actual_runtime_seconds"),
        field_name="actual_runtime_seconds",
    )
    normalized = dict(result)
    normalized["validation_losses"] = validation_losses
    normalized["final_validation_loss"] = final_loss
    normalized["actual_runtime_seconds"] = actual_runtime_seconds
    return normalized


def _finite_float_list(value: Any, *, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise WorkerResultError(f"{field_name} must be a list")
    return [
        _finite_float(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    ]


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkerResultError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise WorkerResultError(f"{field_name} must be finite")
    return parsed


def _positive_int(value: Any, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkerResultError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise WorkerResultError(f"{field_name} must be positive")
    return parsed


def _positive_manifest_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise WorkerManifestError(f"{field_name} must be positive")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkerManifestError(f"{field_name} must be positive") from exc
    if parsed <= 0:
        raise WorkerManifestError(f"{field_name} must be positive")
    return parsed


def _validate_validation_config(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise WorkerManifestError("validation_config must be a JSON object")
    eval_manifest_id = value.get("eval_manifest_id")
    if not isinstance(eval_manifest_id, str) or not eval_manifest_id.strip():
        raise WorkerManifestError("validation_config.eval_manifest_id must be non-empty")
    _positive_manifest_int(
        value.get("validation_tokens_per_eval"),
        field_name="validation_config.validation_tokens_per_eval",
    )
    _positive_manifest_int(
        value.get("validation_batches_per_eval"),
        field_name="validation_config.validation_batches_per_eval",
    )


def _manifest_identity(manifest: Mapping[str, Any]) -> tuple[str, str]:
    experiment_id = manifest.get("experiment_id")
    if isinstance(experiment_id, str) and experiment_id.strip():
        return "experiment_id", experiment_id
    final_run_id = manifest.get("final_run_id")
    if isinstance(final_run_id, str) and final_run_id.strip():
        return "final_run_id", final_run_id
    raise WorkerManifestError("manifest missing required field: experiment_id or final_run_id")


def main(argv: list[str] | None = None, *, session=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--callback-url", required=True)
    parser.add_argument("--callback-token-env", required=True)
    parser.add_argument(
        "--trainer",
        default=os.environ.get("SCALING_WORKER_TRAINER", ""),
        help="Trainer callable as module:function, or SCALING_WORKER_TRAINER.",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=int,
        default=int(os.environ.get("SCALING_WORKER_HEARTBEAT_INTERVAL_SECONDS", "0")),
        help="Emit worker heartbeat callbacks at this interval while training.",
    )
    parser.add_argument(
        "--tokenized-index-validation-mode",
        choices=["metadata", "full"],
        default=os.environ.get(
            "SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE",
            "metadata",
        ),
        help="Validate tokenized indexes with metadata checks or full shard hashes.",
    )
    parser.add_argument(
        "--event-sink",
        choices=["http", "jsonl", "both"],
        default=os.environ.get("SCALING_WORKER_EVENT_SINK", "http"),
        help="Emit worker events over HTTP, JSONL, or both.",
    )
    parser.add_argument(
        "--event-log-path",
        default=os.environ.get("SCALING_WORKER_EVENT_LOG_PATH", ""),
        help="Shared-storage JSONL path used when --event-sink is jsonl or both.",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest_uri)
    callback_token = os.environ.get(args.callback_token_env, "")
    trainer = load_trainer(args.trainer) if args.trainer else _missing_trainer
    return run_worker(
        manifest=manifest,
        callback_url=args.callback_url,
        callback_token=callback_token,
        trainer=trainer,
        session=session,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        tokenized_index_validation_mode=args.tokenized_index_validation_mode,
        event_sink=args.event_sink,
        event_log_path=args.event_log_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
