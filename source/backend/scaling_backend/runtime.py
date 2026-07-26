from __future__ import annotations

import csv
import datetime as _dt
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import FastAPI

from scaling_backend.api import create_app
from scaling_backend.course_config import CourseConfigError, resolve_runtime_env
from scaling_backend.manifest_store import LocalManifestStore, PublishedManifestStore
from scaling_backend.providers.contracts import ProviderAdapter
from scaling_backend.providers.fake import FakeProviderAdapter
from scaling_backend.providers.qz_distributed.factory import (
    QzProviderConfigError,
    build_qz_distributed_adapter_from_env,
)
from scaling_backend.providers.slurm import (
    SlurmProviderConfigError,
    build_slurm_adapter_from_env,
)
from scaling_backend.service import ExperimentService


class RuntimeConfigError(ValueError):
    pass


_LOG = logging.getLogger(__name__)
_UVICORN_ERROR_LOG = logging.getLogger("uvicorn.error")


def load_api_keys_csv(path: str | Path) -> dict[str, str]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            required = {"student_id", "api_key"}
            missing = sorted(required - fieldnames)
            if missing:
                raise RuntimeConfigError(
                    "Student key CSV missing required column(s): "
                    + ", ".join(missing)
                )

            api_keys: dict[str, str] = {}
            for line_number, row in enumerate(reader, start=2):
                student_id = str(row.get("student_id", "")).strip()
                api_key = str(row.get("api_key", "")).strip()
                if not student_id or not api_key:
                    raise RuntimeConfigError(
                        f"Student key CSV row {line_number} must include student_id and api_key"
                    )
                if api_key in api_keys:
                    raise RuntimeConfigError(
                        f"Student key CSV contains duplicate api_key on row {line_number}"
                    )
                api_keys[api_key] = student_id
    except OSError as exc:
        raise RuntimeConfigError(f"Could not read student key CSV: {source}") from exc

    if not api_keys:
        raise RuntimeConfigError("Student key CSV must contain at least one student")
    return api_keys


def build_service_from_env(
    env: Mapping[str, str] | None = None,
    *,
    now: Callable[[], str] | None = None,
    provider: ProviderAdapter | None = None,
) -> ExperimentService:
    try:
        values = resolve_runtime_env(env or os.environ)
    except CourseConfigError as exc:
        raise RuntimeConfigError(str(exc)) from exc
    clock = now or _utc_now_iso
    selected_provider = provider or _build_provider(values, now=clock)
    manifest_base_uri = _get(values, "SCALING_MANIFEST_BASE_URI", "memory://manifests")
    local_manifest_dir = str(values.get("SCALING_LOCAL_MANIFEST_DIR", "")).strip()
    published_manifest_dir = str(
        values.get("SCALING_PUBLISHED_MANIFEST_DIR", "")
    ).strip()
    published_manifest_base_uri = str(
        values.get("SCALING_PUBLISHED_MANIFEST_BASE_URI", "")
    ).strip()
    service = ExperimentService(
        provider=selected_provider,
        total_budget_seconds=_get_int(
            values,
            "SCALING_TOTAL_BUDGET_SECONDS",
            12 * 3600,
        ),
        now=clock,
        manifest_uri_builder=_manifest_uri_builder(manifest_base_uri),
        callback_url=_require(values, "SCALING_CALLBACK_URL"),
        manifest_store=_build_manifest_store(
            local_manifest_dir=local_manifest_dir,
            published_manifest_dir=published_manifest_dir,
            published_manifest_base_uri=published_manifest_base_uri,
        ),
        max_active_experiments_per_student=_get_nonnegative_int(
            values,
            "SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT",
            0,
        ),
        max_active_experiments_global=_get_nonnegative_int(
            values,
            "SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL",
            0,
        ),
        worker_gpu_count=_worker_gpu_count(values),
        code_version=_get(values, "SCALING_CODE_VERSION", "local-dev"),
        data_manifest_id=_get(
            values,
            "SCALING_DATA_MANIFEST_ID",
            "exploratory-train-v0",
        ),
        eval_manifest_id=_get(
            values,
            "SCALING_EVAL_MANIFEST_ID",
            "exploratory-eval-v0",
        ),
        final_data_manifest_id=_get(
            values,
            "SCALING_FINAL_DATA_MANIFEST_ID",
            "final-train-v0",
        ),
        final_eval_manifest_id=_get(
            values,
            "SCALING_FINAL_EVAL_MANIFEST_ID",
            "final-eval-v0",
        ),
        validation_tokens_per_eval=_get_int(
            values,
            "SCALING_VALIDATION_TOKENS_PER_EVAL",
            262_144,
        ),
        final_validation_tokens_per_eval=_get_int(
            values,
            "SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL",
            524_288,
        ),
        train_tokenized_index_uri=_get(
            values,
            "SCALING_TOKENIZED_TRAIN_INDEX_URI",
            "",
        ),
        validation_tokenized_index_uri=_get(
            values,
            "SCALING_TOKENIZED_VALIDATION_INDEX_URI",
            "",
        ),
        final_train_tokenized_index_uri=_get(
            values,
            "SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI",
            "",
        ),
        final_validation_tokenized_index_uri=_get(
            values,
            "SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI",
            "",
        ),
    )
    snapshot_path = str(values.get("SCALING_STATE_SNAPSHOT_PATH", "")).strip()
    if snapshot_path and Path(snapshot_path).exists():
        service.load_state_snapshot(snapshot_path)
    return service


def build_app_from_env(
    env: Mapping[str, str] | None = None,
    *,
    now: Callable[[], str] | None = None,
    provider: ProviderAdapter | None = None,
) -> FastAPI:
    try:
        values = resolve_runtime_env(env or os.environ)
    except CourseConfigError as exc:
        raise RuntimeConfigError(str(exc)) from exc
    api_keys = load_api_keys_csv(_require(values, "SCALING_STUDENT_KEYS_CSV"))
    service = build_service_from_env(values, now=now, provider=provider)
    app = create_app(
        service=service,
        api_keys=api_keys,
        internal_callback_token=_require(values, "SCALING_INTERNAL_CALLBACK_TOKEN"),
        admin_api_token=_require(values, "SCALING_ADMIN_API_TOKEN"),
        worker_event_import_dir=_get(values, "SCALING_WORKER_EVENT_IMPORT_DIR", ""),
    )
    snapshot_path = str(values.get("SCALING_STATE_SNAPSHOT_PATH", "")).strip()
    if snapshot_path:
        _install_snapshot_autosave(app, service, snapshot_path)
    provider_poll_interval_seconds = _get_nonnegative_int(
        values,
        "SCALING_PROVIDER_POLL_INTERVAL_SECONDS",
        0,
    )
    if provider_poll_interval_seconds > 0:
        _install_provider_poll_loop(
            app,
            service,
            interval_seconds=provider_poll_interval_seconds,
            snapshot_path=snapshot_path,
        )
    qz_session_heartbeat_interval_seconds = _get_nonnegative_int(
        values,
        "QZ_SESSION_HEARTBEAT_INTERVAL_SECONDS",
        0,
    )
    if (
        qz_session_heartbeat_interval_seconds > 0
        and getattr(service.provider, "provider_name", "") == "qz_distributed"
    ):
        _install_qz_session_heartbeat_loop(
            app,
            service.provider,
            interval_seconds=qz_session_heartbeat_interval_seconds,
        )
    return app


def _build_provider(
    values: Mapping[str, str],
    *,
    now: Callable[[], str],
) -> ProviderAdapter:
    provider_name = _get(values, "SCALING_PROVIDER", "qz_distributed")
    if provider_name == "fake":
        return FakeProviderAdapter(
            now=now,
            artifact_base_uri=_get(
                values,
                "SCALING_FAKE_ARTIFACT_BASE_URI",
                "memory://fake-provider",
            ),
        )
    if provider_name == "qz_distributed":
        try:
            return build_qz_distributed_adapter_from_env(values, now=now)
        except QzProviderConfigError as exc:
            raise RuntimeConfigError(str(exc)) from exc
    if provider_name == "slurm":
        try:
            return build_slurm_adapter_from_env(values, now=now)
        except SlurmProviderConfigError as exc:
            raise RuntimeConfigError(str(exc)) from exc
    raise RuntimeConfigError(f"Unsupported SCALING_PROVIDER: {provider_name}")


def _build_manifest_store(
    *,
    local_manifest_dir: str,
    published_manifest_dir: str,
    published_manifest_base_uri: str,
):
    if published_manifest_dir or published_manifest_base_uri:
        if not published_manifest_dir or not published_manifest_base_uri:
            raise RuntimeConfigError(
                "SCALING_PUBLISHED_MANIFEST_DIR and "
                "SCALING_PUBLISHED_MANIFEST_BASE_URI must be set together"
            )
        return PublishedManifestStore(
            root=published_manifest_dir,
            public_base_uri=published_manifest_base_uri,
        )
    if local_manifest_dir:
        return LocalManifestStore(local_manifest_dir)
    return None


def _worker_gpu_count(values: Mapping[str, str]) -> int:
    explicit = str(values.get("SCALING_WORKER_GPU_COUNT", "")).strip()
    if explicit:
        return _get_nonnegative_int(values, "SCALING_WORKER_GPU_COUNT", 0)
    provider_name = _get(values, "SCALING_PROVIDER", "qz_distributed")
    if provider_name == "qz_distributed":
        return _get_nonnegative_int(values, "QZ_SPEC_GPU_COUNT", 0)
    if provider_name == "slurm":
        return _get_nonnegative_int(values, "SLURM_GPUS_PER_TASK", 0)
    return 0


def _install_snapshot_autosave(
    app: FastAPI,
    service: ExperimentService,
    snapshot_path: str,
) -> None:
    @app.middleware("http")
    async def save_snapshot_after_mutation(request, call_next):
        response = await call_next(request)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and response.status_code < 400:
            service.save_state_snapshot(snapshot_path)
        return response


def run_provider_poll_iteration(
    service: ExperimentService,
    *,
    snapshot_path: str = "",
) -> dict[str, list[dict[str, Any]]]:
    report = service.admin_poll_active_runs()
    if snapshot_path:
        service.save_state_snapshot(snapshot_path)
    return report


def run_qz_session_heartbeat_iteration(provider: ProviderAdapter) -> dict[str, Any]:
    if getattr(provider, "provider_name", "") != "qz_distributed":
        raise RuntimeConfigError("QZ session heartbeat requires qz_distributed provider")
    client = getattr(provider, "client", None)
    config = getattr(provider, "config", None)
    probe = getattr(client, "probe_cookie_auth", None)
    workspace_id = str(getattr(config, "workspace_id", "")).strip()
    if not callable(probe) or not workspace_id:
        raise RuntimeConfigError("QZ session heartbeat provider is missing probe support")
    probe(workspace_id=workspace_id)
    _log_qz_session_heartbeat_info("QZ session heartbeat succeeded")
    return {"ok": True, "provider": "qz_distributed"}


def _log_qz_session_heartbeat_info(message: str, *args: Any) -> None:
    _LOG.info(message, *args)
    _UVICORN_ERROR_LOG.info(message, *args)


def _install_provider_poll_loop(
    app: FastAPI,
    service: ExperimentService,
    *,
    interval_seconds: int,
    snapshot_path: str = "",
    thread_factory: Callable[..., threading.Thread] = threading.Thread,
    stop_event_factory: Callable[[], threading.Event] = threading.Event,
) -> None:
    interval = int(interval_seconds)
    if interval <= 0:
        return

    app.state.provider_poll_interval_seconds = interval
    stop_event = stop_event_factory()

    def poll_loop() -> None:
        while not stop_event.wait(interval):
            try:
                run_provider_poll_iteration(
                    service,
                    snapshot_path=snapshot_path,
                )
            except Exception:
                _LOG.exception("Background provider poll failed")

    def start_provider_poll_loop() -> None:
        if getattr(app.state, "provider_poll_thread", None) is not None:
            return
        thread = thread_factory(
            target=poll_loop,
            daemon=True,
            name="scaling-provider-poll",
        )
        app.state.provider_poll_stop_event = stop_event
        app.state.provider_poll_thread = thread
        thread.start()

    def stop_provider_poll_loop() -> None:
        stop_event.set()
        thread = getattr(app.state, "provider_poll_thread", None)
        if thread is not None:
            thread.join(timeout=5)
        app.state.provider_poll_thread = None

    _add_lifecycle_event_handler(app, "startup", start_provider_poll_loop)
    _add_lifecycle_event_handler(app, "shutdown", stop_provider_poll_loop)


def _install_qz_session_heartbeat_loop(
    app: FastAPI,
    provider: ProviderAdapter,
    *,
    interval_seconds: int,
    thread_factory: Callable[..., threading.Thread] = threading.Thread,
    stop_event_factory: Callable[[], threading.Event] = threading.Event,
) -> None:
    interval = int(interval_seconds)
    if interval <= 0:
        return

    app.state.qz_session_heartbeat_interval_seconds = interval
    stop_event = stop_event_factory()

    def heartbeat_loop() -> None:
        while True:
            try:
                run_qz_session_heartbeat_iteration(provider)
            except Exception:
                _LOG.exception("Background QZ session heartbeat failed")
                _UVICORN_ERROR_LOG.exception("Background QZ session heartbeat failed")
            if stop_event.wait(interval):
                break

    def start_qz_session_heartbeat_loop() -> None:
        if getattr(app.state, "qz_session_heartbeat_thread", None) is not None:
            return
        _log_qz_session_heartbeat_info(
            "Starting QZ session heartbeat loop interval_seconds=%s",
            interval,
        )
        thread = thread_factory(
            target=heartbeat_loop,
            daemon=True,
            name="scaling-qz-session-heartbeat",
        )
        app.state.qz_session_heartbeat_stop_event = stop_event
        app.state.qz_session_heartbeat_thread = thread
        thread.start()

    def stop_qz_session_heartbeat_loop() -> None:
        stop_event.set()
        thread = getattr(app.state, "qz_session_heartbeat_thread", None)
        if thread is not None:
            thread.join(timeout=5)
        app.state.qz_session_heartbeat_thread = None

    _add_lifecycle_event_handler(app, "startup", start_qz_session_heartbeat_loop)
    _add_lifecycle_event_handler(app, "shutdown", stop_qz_session_heartbeat_loop)


def _add_lifecycle_event_handler(
    app: FastAPI,
    event_type: str,
    handler: Callable[[], None],
) -> None:
    add_event_handler = getattr(app, "add_event_handler", None)
    if callable(add_event_handler):
        add_event_handler(event_type, handler)
        return

    router = getattr(app, "router", None)
    router_add_event_handler = getattr(router, "add_event_handler", None)
    if callable(router_add_event_handler):
        router_add_event_handler(event_type, handler)
        return

    handlers = getattr(router, f"on_{event_type}", None)
    if hasattr(handlers, "append"):
        handlers.append(handler)
        return

    raise RuntimeConfigError(
        f"FastAPI app does not support {event_type} lifecycle handlers"
    )


def _manifest_uri_builder(base_uri: str) -> Callable[[str], str]:
    normalized = base_uri.rstrip("/")

    def build(run_id: str) -> str:
        return f"{normalized}/{run_id}.json"

    return build


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _require(values: Mapping[str, str], key: str) -> str:
    value = str(values.get(key, "")).strip()
    if not value:
        raise RuntimeConfigError(f"Missing required runtime setting: {key}")
    return value


def _get(values: Mapping[str, str], key: str, default: str) -> str:
    value = str(values.get(key, "")).strip()
    return value if value else default


def _get_int(values: Mapping[str, str], key: str, default: int) -> int:
    raw = str(values.get(key, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeConfigError(f"Runtime setting {key} must be an integer") from exc
    if value <= 0:
        raise RuntimeConfigError(f"Runtime setting {key} must be positive")
    return value


def _get_nonnegative_int(values: Mapping[str, str], key: str, default: int) -> int:
    raw = str(values.get(key, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeConfigError(f"Runtime setting {key} must be an integer") from exc
    if value < 0:
        raise RuntimeConfigError(f"Runtime setting {key} must be non-negative")
    return value
