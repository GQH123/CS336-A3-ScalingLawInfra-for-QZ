from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class CourseConfigError(ValueError):
    pass


_TOP_LEVEL_KEYS = frozenset({"schema_version", "api", "worker_manifest_defaults", "qz"})

_API_FIELDS = {
    "provider": "SCALING_PROVIDER",
    "student_keys_csv": "SCALING_STUDENT_KEYS_CSV",
    "internal_callback_token": "SCALING_INTERNAL_CALLBACK_TOKEN",
    "admin_api_token": "SCALING_ADMIN_API_TOKEN",
    "callback_url": "SCALING_CALLBACK_URL",
    "manifest_base_uri": "SCALING_MANIFEST_BASE_URI",
    "local_manifest_dir": "SCALING_LOCAL_MANIFEST_DIR",
    "published_manifest_dir": "SCALING_PUBLISHED_MANIFEST_DIR",
    "published_manifest_base_uri": "SCALING_PUBLISHED_MANIFEST_BASE_URI",
    "worker_event_import_dir": "SCALING_WORKER_EVENT_IMPORT_DIR",
    "state_snapshot_path": "SCALING_STATE_SNAPSHOT_PATH",
    "total_budget_seconds": "SCALING_TOTAL_BUDGET_SECONDS",
    "max_active_experiments_per_student": (
        "SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT"
    ),
    "max_active_experiments_global": "SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL",
    "worker_gpu_count": "SCALING_WORKER_GPU_COUNT",
    "provider_poll_interval_seconds": "SCALING_PROVIDER_POLL_INTERVAL_SECONDS",
    "code_version": "SCALING_CODE_VERSION",
}

_WORKER_MANIFEST_DEFAULT_FIELDS = {
    "data_manifest_id": "SCALING_DATA_MANIFEST_ID",
    "eval_manifest_id": "SCALING_EVAL_MANIFEST_ID",
    "final_data_manifest_id": "SCALING_FINAL_DATA_MANIFEST_ID",
    "final_eval_manifest_id": "SCALING_FINAL_EVAL_MANIFEST_ID",
    "validation_tokens_per_eval": "SCALING_VALIDATION_TOKENS_PER_EVAL",
    "final_validation_tokens_per_eval": "SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL",
    "tokenized_train_index_uri": "SCALING_TOKENIZED_TRAIN_INDEX_URI",
    "tokenized_validation_index_uri": "SCALING_TOKENIZED_VALIDATION_INDEX_URI",
    "final_tokenized_train_index_uri": "SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI",
    "final_tokenized_validation_index_uri": (
        "SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI"
    ),
}

_QZ_FIELDS = {
    "api_base_url": "QZ_API_BASE_URL",
    "username": "QZ_USERNAME",
    "password_encrypted": "QZ_PASSWORD_ENCRYPTED",
    "cookie": "QZ_COOKIE",
    "workspace_id": "QZ_WORKSPACE_ID",
    "project_id": "QZ_PROJECT_ID",
    "compute_group_id": "QZ_COMPUTE_GROUP_ID",
    "spec_id": "QZ_SPEC_ID",
    "spec_gpu_type": "QZ_SPEC_GPU_TYPE",
    "spec_gpu_count": "QZ_SPEC_GPU_COUNT",
    "spec_cpu_count": "QZ_SPEC_CPU_COUNT",
    "spec_memory_gb": "QZ_SPEC_MEMORY_GB",
    "image": "QZ_IMAGE",
    "image_type": "QZ_IMAGE_TYPE",
    "priority": "QZ_PRIORITY",
    "shm_gi": "QZ_SHM_GI",
    "framework": "QZ_FRAMEWORK",
    "callback_token_env": "QZ_CALLBACK_TOKEN_ENV",
    "worker_trainer": "QZ_WORKER_TRAINER",
    "worker_heartbeat_interval_seconds": "QZ_WORKER_HEARTBEAT_INTERVAL_SECONDS",
    "worker_conda_env": "QZ_WORKER_CONDA_ENV",
    "worker_conda_init": "QZ_WORKER_CONDA_INIT",
    "worker_event_log_dir": "QZ_WORKER_EVENT_LOG_DIR",
    "worker_tokenized_index_validation_mode": (
        "SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE"
    ),
}

_VALID_WORKER_INDEX_MODES = frozenset({"metadata", "full"})


def resolve_runtime_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    values = {str(key): str(value) for key, value in (env or {}).items()}
    config_path = (
        values.get("SCALING_API_CONFIG", "").strip()
        or values.get("SCALING_COURSE_CONFIG", "").strip()
    )
    if not config_path:
        return values

    resolved = load_api_config_env(config_path)
    for key, value in values.items():
        if str(value).strip():
            resolved[str(key)] = str(value).strip()
    return resolved


def load_api_config_env(path: str | Path) -> dict[str, str]:
    source = Path(path)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CourseConfigError(f"Could not read API config: {source}") from exc
    except json.JSONDecodeError as exc:
        raise CourseConfigError(f"API config is not valid JSON: {exc}") from exc

    if not isinstance(config, Mapping):
        raise CourseConfigError("api config must be a JSON object")
    _require_schema_version(config)
    _reject_unknown_keys(config, _TOP_LEVEL_KEYS, "api_config")

    output: dict[str, str] = {}
    _load_api_section(config.get("api"), output)
    _load_worker_manifest_defaults(
        config.get("worker_manifest_defaults", {}),
        output,
    )
    _load_qz_section(config.get("qz", {}), output)
    return output


def load_course_config_env(path: str | Path) -> dict[str, str]:
    return load_api_config_env(path)


def _require_schema_version(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise CourseConfigError("api_config.schema_version must be 1")


def _load_api_section(value: Any, output: dict[str, str]) -> None:
    section = _require_mapping(value, "api")
    _reject_unknown_keys(section, _API_FIELDS.keys(), "api")
    for source_key, env_key in _API_FIELDS.items():
        _copy_scalar(section, source_key, output, env_key, section_name="api")


def _load_worker_manifest_defaults(value: Any, output: dict[str, str]) -> None:
    section = _optional_mapping(value, "worker_manifest_defaults")
    _reject_unknown_keys(
        section,
        _WORKER_MANIFEST_DEFAULT_FIELDS.keys(),
        "worker_manifest_defaults",
    )
    for source_key, env_key in _WORKER_MANIFEST_DEFAULT_FIELDS.items():
        _copy_scalar(
            section,
            source_key,
            output,
            env_key,
            section_name="worker_manifest_defaults",
        )


def _load_qz_section(value: Any, output: dict[str, str]) -> None:
    section = _optional_mapping(value, "qz")
    _reject_unknown_keys(section, _QZ_FIELDS.keys(), "qz")
    mode = _optional_string(
        section,
        "worker_tokenized_index_validation_mode",
        section_name="qz",
    )
    if mode and mode not in _VALID_WORKER_INDEX_MODES:
        raise CourseConfigError(
            "qz.worker_tokenized_index_validation_mode must be metadata or full"
        )
    for source_key, env_key in _QZ_FIELDS.items():
        _copy_scalar(section, source_key, output, env_key, section_name="qz")


def _require_mapping(value: Any, section_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CourseConfigError(f"{section_name} must be a JSON object")
    return value


def _optional_mapping(value: Any, section_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CourseConfigError(f"{section_name} must be a JSON object")
    return value


def _reject_unknown_keys(
    section: Mapping[str, Any],
    allowed: Any,
    section_name: str,
) -> None:
    allowed_keys = set(allowed)
    unknown = sorted(str(key) for key in section if str(key) not in allowed_keys)
    if unknown:
        raise CourseConfigError(
            "Unknown API config field(s): "
            + ", ".join(f"{section_name}.{key}" for key in unknown)
        )


def _copy_scalar(
    section: Mapping[str, Any],
    source_key: str,
    output: dict[str, str],
    env_key: str,
    *,
    section_name: str,
) -> None:
    if source_key not in section:
        return
    value = section[source_key]
    if isinstance(value, bool) or value is None:
        raise CourseConfigError(f"{section_name}.{source_key} must be a scalar")
    text = str(value).strip()
    if not text:
        raise CourseConfigError(f"{section_name}.{source_key} must be non-empty")
    output[env_key] = text


def _optional_string(
    section: Mapping[str, Any],
    source_key: str,
    *,
    section_name: str,
) -> str:
    if source_key not in section:
        return ""
    value = section[source_key]
    if isinstance(value, bool) or value is None:
        raise CourseConfigError(f"{section_name}.{source_key} must be a string")
    text = str(value).strip()
    if not text:
        raise CourseConfigError(f"{section_name}.{source_key} must be non-empty")
    return text
