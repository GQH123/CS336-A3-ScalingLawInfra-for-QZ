from __future__ import annotations

import datetime as _dt
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from scaling_backend.providers.qz_distributed.adapter import QzDistributedAdapter
from scaling_backend.providers.qz_distributed.client import QzClient, QzClientConfig
from scaling_backend.providers.qz_distributed.payloads import (
    QzDistributedConfig,
    QzResourceSpec,
)


class QzProviderConfigError(ValueError):
    pass


_REQUIRED_KEYS = (
    "QZ_WORKSPACE_ID",
    "QZ_PROJECT_ID",
    "QZ_COMPUTE_GROUP_ID",
    "QZ_SPEC_ID",
    "QZ_SPEC_GPU_TYPE",
    "QZ_SPEC_GPU_COUNT",
    "QZ_SPEC_CPU_COUNT",
    "QZ_SPEC_MEMORY_GB",
    "QZ_IMAGE",
)


def build_qz_distributed_adapter_from_env(
    env: Mapping[str, str] | None = None,
    *,
    now: Callable[[], str] | None = None,
) -> QzDistributedAdapter:
    values = env or os.environ
    missing = [key for key in _REQUIRED_KEYS if not str(values.get(key, "")).strip()]
    auth_error = _qz_auth_requirement_error(values)
    if auth_error:
        missing.insert(0, auth_error)
    if missing:
        raise QzProviderConfigError(
            "Missing required QZ provider settings: " + ", ".join(missing)
        )

    client = QzClient(
        QzClientConfig(
            base_url=_get(values, "QZ_API_BASE_URL", "https://qz.sii.edu.cn"),
            username=_get(values, "QZ_USERNAME", ""),
            password=_get(values, "QZ_PASSWORD_ENCRYPTED", ""),
            cookie=_get(values, "QZ_COOKIE", ""),
            cookie_file_path=_get(values, "QZ_COOKIE_FILE", ""),
            request_timeout_seconds=_get_positive_int(
                values, "QZ_REQUEST_TIMEOUT_SECONDS", 60
            ),
            login_timeout_seconds=_get_positive_int(
                values, "QZ_LOGIN_TIMEOUT_SECONDS", 30
            ),
            login_max_tries=_get_positive_int(values, "QZ_LOGIN_MAX_TRIES", 3),
            proxy=_get(values, "QZ_PROXY", ""),
            session_heartbeat_interval_seconds=_get_non_negative_int(
                values, "QZ_SESSION_HEARTBEAT_INTERVAL_SECONDS", 0
            ),
        )
    )
    config = QzDistributedConfig(
        workspace_id=_require(values, "QZ_WORKSPACE_ID"),
        project_id=_require(values, "QZ_PROJECT_ID"),
        logic_compute_group_id=_require(values, "QZ_COMPUTE_GROUP_ID"),
        image=_require(values, "QZ_IMAGE"),
        image_type=_get(values, "QZ_IMAGE_TYPE", "SOURCE_PRIVATE"),
        priority=_get_positive_int(values, "QZ_PRIORITY", 10),
        shm_gi=_get_positive_int(values, "QZ_SHM_GI", 1200),
        framework=_get(values, "QZ_FRAMEWORK", "pytorch"),
    )
    spec = QzResourceSpec(
        id=_require(values, "QZ_SPEC_ID"),
        gpu_type=_require(values, "QZ_SPEC_GPU_TYPE"),
        gpu_count=_get_positive_int(values, "QZ_SPEC_GPU_COUNT", 0),
        cpu_count=_get_positive_int(values, "QZ_SPEC_CPU_COUNT", 0),
        memory_gb=_get_positive_int(values, "QZ_SPEC_MEMORY_GB", 0),
    )
    return QzDistributedAdapter(
        client=client,
        config=config,
        spec=spec,
        callback_token_env=_get(
            values, "QZ_CALLBACK_TOKEN_ENV", "SCALING_CALLBACK_TOKEN"
        ),
        callback_token_value=_get(values, "SCALING_INTERNAL_CALLBACK_TOKEN", ""),
        trainer_spec=_get(values, "QZ_WORKER_TRAINER", ""),
        heartbeat_interval_seconds=_get_non_negative_int(
            values, "QZ_WORKER_HEARTBEAT_INTERVAL_SECONDS", 0
        ),
        tokenized_index_validation_mode=_get_tokenized_index_validation_mode(values),
        worker_conda_env=_get(values, "QZ_WORKER_CONDA_ENV", ""),
        worker_conda_init=_get(values, "QZ_WORKER_CONDA_INIT", ""),
        worker_event_log_dir=_get(values, "QZ_WORKER_EVENT_LOG_DIR", ""),
        now=now or _utc_now_iso,
    )


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _require(values: Mapping[str, str], key: str) -> str:
    value = str(values.get(key, "")).strip()
    if not value:
        raise QzProviderConfigError(f"Missing required QZ provider setting: {key}")
    return value


def _qz_auth_requirement_error(values: Mapping[str, str]) -> str:
    cookie = _get(values, "QZ_COOKIE", "")
    cookie_file = _get(values, "QZ_COOKIE_FILE", "")
    username = _get(values, "QZ_USERNAME", "")
    password = _get(values, "QZ_PASSWORD_ENCRYPTED", "")
    if cookie or _cookie_file_has_value(cookie_file) or (username and password):
        return ""
    return "QZ_COOKIE, QZ_COOKIE_FILE, or QZ_USERNAME/QZ_PASSWORD_ENCRYPTED"


def _cookie_file_has_value(path: str) -> bool:
    if not path:
        return False
    try:
        return bool(Path(path).read_text(encoding="utf-8").strip())
    except OSError:
        return False


def _get(values: Mapping[str, str], key: str, default: str) -> str:
    value = str(values.get(key, "")).strip()
    return value if value else default


def _get_int(values: Mapping[str, str], key: str, default: int) -> int:
    raw = str(values.get(key, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise QzProviderConfigError(
            f"QZ provider setting {key} must be an integer"
        ) from exc


def _get_positive_int(values: Mapping[str, str], key: str, default: int) -> int:
    parsed = _get_int(values, key, default)
    if parsed <= 0:
        raise QzProviderConfigError(
            f"QZ provider setting {key} must be a positive integer"
        )
    return parsed


def _get_non_negative_int(values: Mapping[str, str], key: str, default: int) -> int:
    parsed = _get_int(values, key, default)
    if parsed < 0:
        raise QzProviderConfigError(
            f"QZ provider setting {key} must be a non-negative integer"
        )
    return parsed


def _get_tokenized_index_validation_mode(values: Mapping[str, str]) -> str:
    mode = _get(values, "SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE", "metadata")
    if mode not in {"metadata", "full"}:
        raise QzProviderConfigError(
            "SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE must be metadata or full"
        )
    return mode
