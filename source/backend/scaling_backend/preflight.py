from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from scaling_backend.course_config import CourseConfigError, resolve_runtime_env
from scaling_backend.providers.qz_distributed.client import QzClient, QzClientConfig
from scaling_backend.runtime import RuntimeConfigError, load_api_keys_csv


REQUIRED_RUNTIME_SETTINGS = [
    "SCALING_STUDENT_KEYS_CSV",
    "SCALING_CALLBACK_URL",
    "SCALING_MANIFEST_BASE_URI",
]

REQUIRED_SECRET_SETTINGS = [
    "SCALING_INTERNAL_CALLBACK_TOKEN",
    "SCALING_ADMIN_API_TOKEN",
]

KNOWN_PROVIDERS = {"fake", "slurm", "qz_distributed"}

QZ_PROVIDER_SETTINGS = [
    "QZ_WORKSPACE_ID",
    "QZ_PROJECT_ID",
    "QZ_COMPUTE_GROUP_ID",
    "QZ_SPEC_ID",
    "QZ_SPEC_GPU_TYPE",
    "QZ_SPEC_GPU_COUNT",
    "QZ_SPEC_CPU_COUNT",
    "QZ_SPEC_MEMORY_GB",
    "QZ_IMAGE",
]

SLURM_PROVIDER_SETTINGS = [
    "SLURM_WORK_DIR",
    "SLURM_PARTITION",
    "SLURM_ACCOUNT",
    "SLURM_GPUS_PER_TASK",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEMORY_GB",
    "SLURM_TIME_LIMIT_MINUTES",
]

PLACEHOLDER_FRAGMENTS = [
    "<",
    ">",
    "change-me",
    "changeme",
    "placeholder",
    "secret",
]

TOKENIZED_INDEX_URI_SETTINGS = [
    "SCALING_TOKENIZED_TRAIN_INDEX_URI",
    "SCALING_TOKENIZED_VALIDATION_INDEX_URI",
    "SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI",
    "SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI",
]

WORKER_TOKENIZED_INDEX_VALIDATION_MODES = {"metadata", "full"}

RUNTIME_URI_SETTINGS = [
    "SCALING_CALLBACK_URL",
    "SCALING_MANIFEST_BASE_URI",
    "SCALING_PUBLISHED_MANIFEST_BASE_URI",
]


def run_preflight(
    env: Mapping[str, str] | None = None,
    *,
    create_dirs: bool = False,
    probe_provider: bool = False,
    provider_probe: Callable[[Mapping[str, str]], dict[str, Any]] | None = None,
    slurm_runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] | None = None,
) -> dict[str, Any]:
    try:
        values = resolve_runtime_env(env or os.environ)
    except CourseConfigError as exc:
        return {
            "ok": False,
            "provider": "",
            "summary": {"pass": 0, "warn": 0, "fail": 1},
            "checks": [
                _check("course_config", "fail", str(exc)),
            ],
        }
    provider = _get(values, "SCALING_PROVIDER", "qz_distributed")
    checks = [
        _check_required_runtime_settings(values),
        _check_required_secrets(values),
        _check_runtime_uris(values),
        _check_provider_name(provider),
        _check_student_keys_csv(values),
        _check_manifest_storage(values, create_dirs=create_dirs),
        _check_state_snapshot_path(values, create_dirs=create_dirs),
        _check_budget_settings(values),
        _check_manifest_ids(values),
        _check_tokenized_index_uris(values),
        _check_worker_settings(values, create_dirs=create_dirs),
        _check_provider_environment(values, provider),
    ]
    if probe_provider:
        checks.append(
            provider_probe(values)
            if provider_probe
            else _check_provider_probe(values, provider, slurm_runner=slurm_runner)
        )
    summary = {
        "pass": sum(1 for check in checks if check["status"] == "pass"),
        "warn": sum(1 for check in checks if check["status"] == "warn"),
        "fail": sum(1 for check in checks if check["status"] == "fail"),
    }
    return {
        "ok": summary["fail"] == 0,
        "provider": provider,
        "summary": summary,
        "checks": checks,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scaling_backend.preflight",
        description="Validate scaling-laws backend deployment settings without submitting jobs.",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional API runtime JSON config path, equivalent to SCALING_API_CONFIG.",
    )
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create configured manifest and snapshot parent directories if missing.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for writing the full preflight JSON report.",
    )
    parser.add_argument(
        "--probe-provider",
        action="store_true",
        help=(
            "Run a non-submitting provider connectivity probe. "
            "QZ validates QZ_COOKIE when present, otherwise performs CAS login; "
            "Slurm runs version commands only."
        ),
    )
    args = parser.parse_args(argv)

    run_env = dict(os.environ if env is None else env)
    if args.config:
        run_env["SCALING_API_CONFIG"] = args.config

    report = run_preflight(
        run_env,
        create_dirs=args.create_dirs,
        probe_provider=args.probe_provider,
    )
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


def _check_required_runtime_settings(values: Mapping[str, str]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_RUNTIME_SETTINGS if not _get(values, key, "")]
    if missing:
        return _check(
            "required_runtime_settings",
            "fail",
            "Missing required runtime setting(s): " + ", ".join(missing),
        )
    return _check(
        "required_runtime_settings",
        "pass",
        "Required runtime settings are present.",
    )


def _check_required_secrets(values: Mapping[str, str]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_SECRET_SETTINGS if not _get(values, key, "")]
    if missing:
        return _check(
            "required_secrets",
            "fail",
            "Missing required secret setting(s): " + ", ".join(missing),
        )

    secrets = {key: _get(values, key, "") for key in REQUIRED_SECRET_SETTINGS}
    if len(set(secrets.values())) != len(secrets):
        return _check(
            "required_secrets",
            "fail",
            "SCALING_INTERNAL_CALLBACK_TOKEN and SCALING_ADMIN_API_TOKEN must be distinct.",
        )
    placeholders = [
        key
        for key, value in secrets.items()
        if _looks_like_placeholder_secret(value)
    ]
    if placeholders:
        return _check(
            "required_secrets",
            "fail",
            "Secret setting(s) look like placeholders: " + ", ".join(placeholders),
        )
    return _check("required_secrets", "pass", "Required secrets are present.")


def _check_runtime_uris(values: Mapping[str, str]) -> dict[str, Any]:
    configured: list[str] = []
    errors: list[str] = []
    for key in RUNTIME_URI_SETTINGS:
        raw = _get(values, key, "")
        if not raw:
            continue
        configured.append(key)
        message = _runtime_uri_error(key, raw, values=values)
        if message:
            errors.append(f"{key}: {message}")
    if errors:
        return _check(
            "runtime_uris",
            "fail",
            "Invalid runtime URI setting(s): " + "; ".join(errors),
        )
    return _check(
        "runtime_uris",
        "pass",
        "Runtime URI settings are valid.",
        metadata={"configured": sorted(configured)},
    )


def _runtime_uri_error(key: str, value: str, *, values: Mapping[str, str]) -> str:
    parsed = urlparse(value)
    if key == "SCALING_CALLBACK_URL":
        if parsed.scheme not in {"http", "https"}:
            return "must be an http(s) URL"
        if not parsed.netloc:
            return "must include a host"
        if parsed.hostname in {"api.internal", "backend.internal"}:
            if _uses_jsonl_worker_events(values):
                return ""
            return (
                f"{parsed.hostname} is a placeholder host; set a control-node "
                "callback URL reachable from worker containers"
            )
        return ""

    if key == "SCALING_PUBLISHED_MANIFEST_BASE_URI":
        if parsed.scheme not in {"http", "https", "s3", "gs"}:
            return "must be an http(s), s3, or gs URI"
        if not parsed.netloc:
            return "must include a host or bucket"
        if parsed.hostname in {"manifests.internal"}:
            return (
                f"{parsed.hostname} is a placeholder host; set a manifest base "
                "URI reachable from worker containers"
            )
        return ""

    if key == "SCALING_MANIFEST_BASE_URI":
        if parsed.scheme == "memory":
            if not parsed.netloc and not parsed.path:
                return "memory URI must include a namespace"
            return ""
        if parsed.scheme in {"http", "https", "s3", "gs"}:
            if not parsed.netloc:
                return "must include a host or bucket"
            if parsed.hostname in {"manifests.internal"}:
                return (
                    f"{parsed.hostname} is a placeholder host; set a manifest "
                    "base URI reachable from worker containers"
                )
            return ""
        if parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                return f"unsupported file URI host: {parsed.netloc}"
            if not parsed.path:
                return "file URI must include a path"
            return ""
        if parsed.scheme:
            return f"unsupported URI scheme: {parsed.scheme}"
        return "must be an absolute URI"

    return ""


def _uses_jsonl_worker_events(values: Mapping[str, str]) -> bool:
    return bool(
        _get(values, "SCALING_WORKER_EVENT_IMPORT_DIR", "")
        and _get(values, "QZ_WORKER_EVENT_LOG_DIR", "")
    )


def _check_provider_name(provider: str) -> dict[str, Any]:
    if provider not in KNOWN_PROVIDERS:
        return _check(
            "provider_name",
            "fail",
            f"Unsupported SCALING_PROVIDER: {provider}",
        )
    return _check("provider_name", "pass", f"Provider is {provider}.")


def _check_student_keys_csv(values: Mapping[str, str]) -> dict[str, Any]:
    path = _get(values, "SCALING_STUDENT_KEYS_CSV", "")
    if not path:
        return _check(
            "student_keys_csv",
            "fail",
            "SCALING_STUDENT_KEYS_CSV is required.",
        )
    try:
        api_keys = load_api_keys_csv(path)
    except RuntimeConfigError as exc:
        return _check("student_keys_csv", "fail", str(exc))
    return _check(
        "student_keys_csv",
        "pass",
        "Student key CSV is readable.",
        metadata={"student_count": len(set(api_keys.values()))},
    )


def _check_manifest_storage(
    values: Mapping[str, str],
    *,
    create_dirs: bool,
) -> dict[str, Any]:
    local_dir = _get(values, "SCALING_LOCAL_MANIFEST_DIR", "")
    published_dir = _get(values, "SCALING_PUBLISHED_MANIFEST_DIR", "")
    published_base_uri = _get(values, "SCALING_PUBLISHED_MANIFEST_BASE_URI", "")

    if published_dir or published_base_uri:
        if not published_dir or not published_base_uri:
            return _check(
                "manifest_storage",
                "fail",
                "SCALING_PUBLISHED_MANIFEST_DIR and "
                "SCALING_PUBLISHED_MANIFEST_BASE_URI must be set together.",
            )
        status = _ensure_directory(Path(published_dir), create_dirs=create_dirs)
        if status:
            return _check("manifest_storage", "fail", status)
        return _check(
            "manifest_storage",
            "pass",
            "Published manifest directory is writable.",
            metadata={"mode": "published", "directory": published_dir},
        )

    if local_dir:
        status = _ensure_directory(Path(local_dir), create_dirs=create_dirs)
        if status:
            return _check("manifest_storage", "fail", status)
        return _check(
            "manifest_storage",
            "pass",
            "Local manifest directory is writable.",
            metadata={"mode": "local", "directory": local_dir},
        )

    return _check(
        "manifest_storage",
        "warn",
        "No manifest store directory configured; manifests will use provider/base URI only.",
        metadata={"mode": "uri_only"},
    )


def _check_state_snapshot_path(
    values: Mapping[str, str],
    *,
    create_dirs: bool,
) -> dict[str, Any]:
    snapshot_path = _get(values, "SCALING_STATE_SNAPSHOT_PATH", "")
    if not snapshot_path:
        return _check(
            "state_snapshot_path",
            "warn",
            "SCALING_STATE_SNAPSHOT_PATH is not set; process restarts will not restore local state.",
        )
    path = Path(snapshot_path)
    parent_status = _ensure_directory(path.parent, create_dirs=create_dirs)
    if parent_status:
        return _check("state_snapshot_path", "fail", parent_status)
    if path.exists() and path.is_dir():
        return _check(
            "state_snapshot_path",
            "fail",
            f"State snapshot path is a directory: {path}",
        )
    return _check(
        "state_snapshot_path",
        "pass",
        "State snapshot parent directory is writable.",
        metadata={"path": snapshot_path},
    )


def _check_budget_settings(values: Mapping[str, str]) -> dict[str, Any]:
    positive = [
        "SCALING_TOTAL_BUDGET_SECONDS",
        "SCALING_VALIDATION_TOKENS_PER_EVAL",
        "SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL",
    ]
    nonnegative = [
        "SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT",
        "SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL",
        "SCALING_PROVIDER_POLL_INTERVAL_SECONDS",
    ]
    errors: list[str] = []
    for key in positive:
        raw = _get(values, key, "")
        if raw and (not raw.isdigit() or int(raw) <= 0):
            errors.append(f"{key} must be positive")
    for key in nonnegative:
        raw = _get(values, key, "")
        if raw and (not raw.isdigit() or int(raw) < 0):
            errors.append(f"{key} must be non-negative")
    if errors:
        return _check("budget_settings", "fail", "; ".join(errors))
    return _check("budget_settings", "pass", "Budget settings are valid.")


def _check_manifest_ids(values: Mapping[str, str]) -> dict[str, Any]:
    exploratory_eval = _get(values, "SCALING_EVAL_MANIFEST_ID", "exploratory-eval-v0")
    final_eval = _get(values, "SCALING_FINAL_EVAL_MANIFEST_ID", "final-eval-v0")
    if exploratory_eval == final_eval:
        return _check(
            "manifest_ids",
            "fail",
            "SCALING_EVAL_MANIFEST_ID and SCALING_FINAL_EVAL_MANIFEST_ID must differ.",
        )
    return _check("manifest_ids", "pass", "Exploratory and final eval manifests differ.")


def _check_tokenized_index_uris(values: Mapping[str, str]) -> dict[str, Any]:
    configured: list[str] = []
    errors: list[str] = []
    for key in TOKENIZED_INDEX_URI_SETTINGS:
        raw = _get(values, key, "")
        if not raw:
            continue
        configured.append(key)
        message = _tokenized_index_uri_error(raw)
        if message:
            errors.append(f"{key}: {message}")
    if errors:
        return _check(
            "tokenized_index_uris",
            "fail",
            "Invalid tokenized index URI setting(s): " + "; ".join(errors),
        )
    return _check(
        "tokenized_index_uris",
        "pass",
        "Tokenized index URI settings are valid.",
        metadata={"configured": sorted(configured)},
    )


def _tokenized_index_uri_error(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return ""
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            return f"unsupported file URI host: {parsed.netloc}"
        if not parsed.path:
            return "file URI must include a path"
        return ""
    if parsed.scheme in {"s3", "gs"}:
        return f"unsupported URI scheme: {parsed.scheme}"
    if parsed.scheme:
        return f"unsupported URI scheme: {parsed.scheme}"
    if not value.strip():
        return "URI must be non-empty"
    return ""


def _check_worker_settings(
    values: Mapping[str, str],
    *,
    create_dirs: bool,
) -> dict[str, Any]:
    mode = _get(values, "SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE", "metadata")
    if mode not in WORKER_TOKENIZED_INDEX_VALIDATION_MODES:
        return _check(
            "worker_settings",
            "fail",
            "SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE must be metadata or full.",
        )
    worker_event_import_dir = _get(values, "SCALING_WORKER_EVENT_IMPORT_DIR", "")
    qz_worker_event_log_dir = _get(values, "QZ_WORKER_EVENT_LOG_DIR", "")
    if worker_event_import_dir or qz_worker_event_log_dir:
        if not worker_event_import_dir or not qz_worker_event_log_dir:
            return _check(
                "worker_settings",
                "fail",
                "SCALING_WORKER_EVENT_IMPORT_DIR and QZ_WORKER_EVENT_LOG_DIR "
                "must be set together for shared-storage worker events.",
            )
        status = _ensure_directory(
            Path(worker_event_import_dir),
            create_dirs=create_dirs,
        )
        if status:
            return _check("worker_settings", "fail", status)
        return _check(
            "worker_settings",
            "pass",
            "Worker settings are valid.",
            metadata={
                "tokenized_index_validation_mode": mode,
                "worker_event_channel": "jsonl",
                "worker_event_import_dir": worker_event_import_dir,
                "qz_worker_event_log_dir": qz_worker_event_log_dir,
            },
        )
    return _check(
        "worker_settings",
        "pass",
        "Worker settings are valid.",
        metadata={
            "tokenized_index_validation_mode": mode,
            "worker_event_channel": "http",
        },
    )


def _check_provider_environment(
    values: Mapping[str, str],
    provider: str,
) -> dict[str, Any]:
    if provider == "fake":
        return _check(
            "provider_environment",
            "pass",
            "Fake provider requires no external credentials.",
        )
    if provider == "slurm":
        required = SLURM_PROVIDER_SETTINGS
    elif provider == "qz_distributed":
        required = QZ_PROVIDER_SETTINGS
    else:
        required = []

    missing = [key for key in required if not _get(values, key, "")]
    if provider == "qz_distributed":
        auth_error = _qz_auth_requirement_error(values)
        if auth_error:
            missing.insert(0, auth_error)
    if missing:
        return _check(
            "provider_environment",
            "fail",
            f"{provider} provider setting(s) missing: " + ", ".join(missing),
        )
    return _check(
        "provider_environment",
        "pass",
        f"{provider} provider settings are present.",
    )


def _check_provider_probe(
    values: Mapping[str, str],
    provider: str,
    *,
    slurm_runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] | None = None,
) -> dict[str, Any]:
    if provider == "fake":
        return _check(
            "provider_probe",
            "pass",
            "Fake provider probe requires no external calls.",
            metadata={"provider": "fake"},
        )
    if provider == "slurm":
        return _check_slurm_probe(slurm_runner=slurm_runner)
    if provider == "qz_distributed":
        return _check_qz_probe(values)
    return _check(
        "provider_probe",
        "fail",
        f"Unsupported provider probe for SCALING_PROVIDER={provider}.",
        metadata={"provider": provider},
    )


def _check_slurm_probe(
    *,
    slurm_runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] | None = None,
) -> dict[str, Any]:
    runner = slurm_runner or _run_command
    commands = [["sinfo", "--version"], ["sbatch", "--version"]]
    for command in commands:
        try:
            result = runner(command)
        except OSError as exc:
            return _check(
                "provider_probe",
                "fail",
                f"Slurm probe failed to run {' '.join(command)}: {exc}",
                metadata={"provider": "slurm"},
            )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            detail = f": {stderr}" if stderr else ""
            return _check(
                "provider_probe",
                "fail",
                f"Slurm probe command failed: {' '.join(command)}{detail}",
                metadata={"provider": "slurm"},
            )
    return _check(
        "provider_probe",
        "pass",
        "Slurm command-line tools are reachable without submitting a job.",
        metadata={"provider": "slurm", "commands": ["sinfo", "sbatch"]},
    )


def _check_qz_probe(values: Mapping[str, str]) -> dict[str, Any]:
    timeout_error = _positive_int_setting(values, "QZ_REQUEST_TIMEOUT_SECONDS", 60)
    login_timeout_error = _positive_int_setting(
        values, "QZ_LOGIN_TIMEOUT_SECONDS", 30
    )
    max_tries_error = _positive_int_setting(values, "QZ_LOGIN_MAX_TRIES", 3)
    for error in [timeout_error, login_timeout_error, max_tries_error]:
        if isinstance(error, str):
            return _check(
                "provider_probe",
                "fail",
                error,
                metadata={"provider": "qz_distributed"},
            )

    config = QzClientConfig(
        base_url=_get(values, "QZ_API_BASE_URL", "https://qz.sii.edu.cn"),
        username=_get(values, "QZ_USERNAME", ""),
        password=_get(values, "QZ_PASSWORD_ENCRYPTED", ""),
        cookie=_get(values, "QZ_COOKIE", ""),
        request_timeout_seconds=timeout_error,
        login_timeout_seconds=login_timeout_error,
        login_max_tries=max_tries_error,
        proxy=_get(values, "QZ_PROXY", ""),
    )
    client = QzClient(config)
    if config.cookie:
        try:
            client.probe_cookie_auth(workspace_id=_get(values, "QZ_WORKSPACE_ID", ""))
        except Exception as exc:
            return _check(
                "provider_probe",
                "fail",
                f"QZ provider cookie probe failed: {exc}",
                metadata={"provider": "qz_distributed", "auth": "cookie"},
            )
        return _check(
            "provider_probe",
            "pass",
            "QZ cookie probe succeeded; no training job was submitted.",
            metadata={"provider": "qz_distributed", "auth": "cookie"},
        )

    try:
        client.login_with_cas()
    except Exception as exc:
        return _check(
            "provider_probe",
            "fail",
            f"QZ provider login probe failed: {exc}",
            metadata={"provider": "qz_distributed", "auth": "login"},
        )
    return _check(
        "provider_probe",
        "pass",
        "QZ CAS login succeeded; no training job was submitted.",
        metadata={"provider": "qz_distributed", "auth": "login"},
    )


def _positive_int_setting(
    values: Mapping[str, str],
    key: str,
    default: int,
) -> int | str:
    raw = _get(values, key, str(default))
    if not raw.isdigit() or int(raw) <= 0:
        return f"{key} must be a positive integer."
    return int(raw)


def _qz_auth_requirement_error(values: Mapping[str, str]) -> str:
    cookie = _get(values, "QZ_COOKIE", "")
    username = _get(values, "QZ_USERNAME", "")
    password = _get(values, "QZ_PASSWORD_ENCRYPTED", "")
    if cookie or (username and password):
        return ""
    return "QZ_COOKIE or QZ_USERNAME/QZ_PASSWORD_ENCRYPTED"


def _run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=False,
        text=True,
        capture_output=True,
    )


def _check(
    check_id: str,
    status: str,
    message: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "id": check_id,
        "status": status,
        "message": message,
    }
    if metadata:
        result["metadata"] = dict(metadata)
    return result


def _ensure_directory(path: Path, *, create_dirs: bool) -> str:
    if path.exists() and not path.is_dir():
        return f"Path exists but is not a directory: {path}"
    if not path.exists():
        if not create_dirs:
            return f"Required parent directory does not exist: {path}"
        path.mkdir(parents=True, exist_ok=True)
    test_path = path / ".scaling-preflight-write-test"
    try:
        test_path.write_text("ok\n", encoding="utf-8")
        test_path.unlink()
    except OSError as exc:
        return f"Directory is not writable: {path} ({exc})"
    return ""


def _looks_like_placeholder_secret(value: str) -> bool:
    normalized = value.strip().lower()
    return any(fragment in normalized for fragment in PLACEHOLDER_FRAGMENTS)


def _get(values: Mapping[str, str], key: str, default: str) -> str:
    value = str(values.get(key, "")).strip()
    return value if value else default


if __name__ == "__main__":
    raise SystemExit(main())
