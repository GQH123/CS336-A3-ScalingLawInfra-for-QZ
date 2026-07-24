from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from scaling_backend.config_validation import resolve_training_config
from scaling_backend.runtime import build_service_from_env


class LifecycleCheckError(ValueError):
    pass


def load_lifecycle_package(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest_path = root / "lifecycle-dataset-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LifecycleCheckError(
            f"could not read lifecycle dataset manifest: {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise LifecycleCheckError(
            f"lifecycle dataset manifest is not valid JSON: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise LifecycleCheckError("lifecycle dataset manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise LifecycleCheckError(
            "lifecycle dataset manifest must declare schema_version = 1"
        )
    _runtime_overrides(manifest)
    _recommended_submit_payload(manifest)
    return manifest


def build_lifecycle_worker_manifest(
    lifecycle_package: Mapping[str, Any],
    *,
    run_id: str = "exp-lifecycle-000001",
    student_id: str = "staff-lifecycle",
    manifest_uri: str = "memory://lifecycle/exp-lifecycle-000001.json",
    callback_url: str = "http://127.0.0.1/internal/provider-events",
    requested_runtime_seconds: int | None = None,
    code_version: str = "lifecycle-check",
    created_at: str | None = None,
) -> dict[str, Any]:
    payload = _submit_payload(
        lifecycle_package,
        requested_runtime_seconds=requested_runtime_seconds,
    )
    config = _require_mapping(payload, "config")
    runtime_seconds = _positive_int(
        payload.get("requested_runtime_seconds"),
        "recommended_submit_payload.requested_runtime_seconds",
    )
    overrides = _runtime_overrides(lifecycle_package)
    data_manifest_id = _require_env(overrides, "SCALING_DATA_MANIFEST_ID")
    eval_manifest_id = _require_env(overrides, "SCALING_EVAL_MANIFEST_ID")
    validation_tokens = _positive_int(
        _require_env(overrides, "SCALING_VALIDATION_TOKENS_PER_EVAL"),
        "SCALING_VALIDATION_TOKENS_PER_EVAL",
    )
    train_index_uri = _require_env(overrides, "SCALING_TOKENIZED_TRAIN_INDEX_URI")
    validation_index_uri = _require_env(
        overrides,
        "SCALING_TOKENIZED_VALIDATION_INDEX_URI",
    )
    resolved = resolve_training_config(
        config,
        requested_runtime_seconds=runtime_seconds,
        code_version=code_version,
        data_manifest_id=data_manifest_id,
        eval_manifest_id=eval_manifest_id,
        validation_tokens_per_eval=validation_tokens,
    )
    model_config = dict(resolved["model"])
    training_config = dict(resolved["training"])
    created = created_at or _utc_now_iso()
    return {
        "manifest_version": 1,
        "run_kind": "exploratory",
        "experiment_id": _non_empty_text(run_id, "run_id"),
        "student_id": _non_empty_text(student_id, "student_id"),
        "manifest_uri": _non_empty_text(manifest_uri, "manifest_uri"),
        "callback_url": _non_empty_text(callback_url, "callback_url"),
        "callback_base_url": _non_empty_text(callback_url, "callback_url"),
        "model_config": model_config,
        "training_config": training_config,
        "resolved_config": dict(resolved),
        "code_version": code_version,
        "data_manifest_id": data_manifest_id,
        "eval_manifest_id": eval_manifest_id,
        "data_config": {
            "train_tokens": training_config["train_tokens"],
            "tokenized_index_uri": train_index_uri,
        },
        "validation_config": {
            "eval_manifest_id": eval_manifest_id,
            "validation_tokens_per_eval": resolved["validation_tokens_per_eval"],
            "validation_batches_per_eval": resolved["validation_batches_per_eval"],
            "tokenized_index_uri": validation_index_uri,
        },
        "max_runtime_seconds": runtime_seconds,
        "runtime_config": {
            "reserved_runtime_seconds": runtime_seconds,
            "max_runtime_seconds": runtime_seconds,
        },
        "created_at": created,
    }


def inspect_lifecycle_tokenized_indexes(
    lifecycle_package: Mapping[str, Any],
    *,
    validation_mode: str = "metadata",
) -> dict[str, Any]:
    mode = _validation_mode(validation_mode)
    manifest = build_lifecycle_worker_manifest(
        lifecycle_package,
        run_id="exp-lifecycle-inspect",
        student_id="staff-lifecycle",
        manifest_uri="memory://lifecycle/inspect.json",
        callback_url="http://127.0.0.1/internal/provider-events",
    )
    previous_mode = os.environ.get("COURSE_TRAINER_TOKENIZED_INDEX_VALIDATION_MODE")
    os.environ["COURSE_TRAINER_TOKENIZED_INDEX_VALIDATION_MODE"] = mode
    try:
        from course_trainer.tokenized_data import load_training_datasets

        datasets = load_training_datasets(manifest)
    finally:
        if previous_mode is None:
            os.environ.pop("COURSE_TRAINER_TOKENIZED_INDEX_VALIDATION_MODE", None)
        else:
            os.environ["COURSE_TRAINER_TOKENIZED_INDEX_VALIDATION_MODE"] = previous_mode

    return {
        "validation_mode": mode,
        "train_index_uri": manifest["data_config"]["tokenized_index_uri"],
        "validation_index_uri": manifest["validation_config"]["tokenized_index_uri"],
        "train_total_tokens": datasets.train.total_tokens,
        "validation_total_tokens": datasets.validation.total_tokens,
        "train_shards": len(datasets.train.shards),
        "validation_shards": len(datasets.validation.shards),
    }


def verify_manifest_uses_lifecycle_package(
    worker_manifest: Mapping[str, Any],
    lifecycle_package: Mapping[str, Any],
    *,
    final: bool = False,
) -> dict[str, Any]:
    overrides = _runtime_overrides(lifecycle_package)
    train_key = (
        "SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI"
        if final
        else "SCALING_TOKENIZED_TRAIN_INDEX_URI"
    )
    validation_key = (
        "SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI"
        if final
        else "SCALING_TOKENIZED_VALIDATION_INDEX_URI"
    )
    data_manifest_key = (
        "SCALING_FINAL_DATA_MANIFEST_ID" if final else "SCALING_DATA_MANIFEST_ID"
    )
    eval_manifest_key = (
        "SCALING_FINAL_EVAL_MANIFEST_ID" if final else "SCALING_EVAL_MANIFEST_ID"
    )
    validation_tokens_key = (
        "SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL"
        if final
        else "SCALING_VALIDATION_TOKENS_PER_EVAL"
    )
    expected_train_uri = _require_env(overrides, train_key)
    expected_validation_uri = _require_env(overrides, validation_key)
    expected_data_manifest_id = _require_env(overrides, data_manifest_key)
    expected_eval_manifest_id = _require_env(overrides, eval_manifest_key)
    expected_validation_tokens = _positive_int(
        _require_env(overrides, validation_tokens_key),
        validation_tokens_key,
    )
    data_config = _require_mapping(worker_manifest, "data_config")
    validation_config = _require_mapping(worker_manifest, "validation_config")
    checks = {
        "data_manifest_id": worker_manifest.get("data_manifest_id")
        == expected_data_manifest_id,
        "eval_manifest_id": worker_manifest.get("eval_manifest_id")
        == expected_eval_manifest_id,
        "train_index_uri": data_config.get("tokenized_index_uri")
        == expected_train_uri,
        "validation_index_uri": validation_config.get("tokenized_index_uri")
        == expected_validation_uri,
        "validation_tokens_per_eval": validation_config.get(
            "validation_tokens_per_eval"
        )
        == expected_validation_tokens,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "train_index_uri": data_config.get("tokenized_index_uri", ""),
        "expected_train_index_uri": expected_train_uri,
        "validation_index_uri": validation_config.get("tokenized_index_uri", ""),
        "expected_validation_index_uri": expected_validation_uri,
        "validation_tokens_per_eval": validation_config.get(
            "validation_tokens_per_eval"
        ),
        "expected_validation_tokens_per_eval": expected_validation_tokens,
    }


def run_control_local_service_check(
    lifecycle_package: Mapping[str, Any],
    *,
    env: Mapping[str, str],
    student_id: str = "staff-lifecycle",
    requested_runtime_seconds: int | None = None,
    unique_model_seed: bool = False,
) -> dict[str, Any]:
    values = {
        str(key): str(value)
        for key, value in env.items()
        if str(value).strip()
    }
    values.update(_runtime_overrides(lifecycle_package))
    values["SCALING_PROVIDER"] = "fake"
    service = build_service_from_env(
        values,
        now=lambda: "2026-07-24T00:00:00+00:00",
    )
    payload = _submit_payload(
        lifecycle_package,
        requested_runtime_seconds=requested_runtime_seconds,
        unique_model_seed=unique_model_seed,
    )
    submitted = service.submit(
        student_id=student_id,
        config=_require_mapping(payload, "config"),
        requested_runtime_seconds=_positive_int(
            payload.get("requested_runtime_seconds"),
            "recommended_submit_payload.requested_runtime_seconds",
        ),
    )
    provider_manifest = dict(service.provider.submitted_manifests["fake-job-000001"])
    manifest_check = verify_manifest_uses_lifecycle_package(
        provider_manifest,
        lifecycle_package,
    )
    return {
        "mode": "local-service",
        "submitted": {
            "experiment_id": submitted.experiment_id,
            "status": submitted.status,
            "budget_reserved_seconds": submitted.budget_reserved_seconds,
        },
        "manifest_check": manifest_check,
        "manifest": provider_manifest,
    }


def run_control_remote_api_check(
    lifecycle_package: Mapping[str, Any],
    *,
    base_url: str,
    api_key: str,
    local_manifest_dir: str | Path = "",
    requested_runtime_seconds: int | None = None,
    unique_model_seed: bool = False,
    request_timeout_seconds: float = 30.0,
    poll_once: bool = True,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    if not base:
        raise LifecycleCheckError("base_url is required")
    if not str(api_key).strip():
        raise LifecycleCheckError("api_key is required")
    session = requests.Session()

    def api_request(method: str, path: str, payload: Mapping[str, Any] | None = None):
        response = session.request(
            method,
            f"{base}{path}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=request_timeout_seconds,
        )
        try:
            body = response.json()
        except ValueError:
            body = {"raw_response": response.text}
        if response.status_code == 409 and body.get("error") == "duplicate_config":
            return response.status_code, body
        if response.status_code < 200 or response.status_code >= 300:
            raise LifecycleCheckError(
                f"{method} {path} failed with HTTP {response.status_code}: "
                f"{json.dumps(body, ensure_ascii=False, sort_keys=True)}"
            )
        if not isinstance(body, dict):
            raise LifecycleCheckError(f"{method} {path} did not return a JSON object")
        return response.status_code, body

    _status, budget_before = api_request("GET", "/budget")
    payload = _submit_payload(
        lifecycle_package,
        requested_runtime_seconds=requested_runtime_seconds,
        unique_model_seed=unique_model_seed,
    )
    submit_status, submitted = api_request("POST", "/submit", payload)
    experiment_id = _non_empty_text(
        submitted.get("experiment_id"),
        "submit response experiment_id",
    )
    experiment = {}
    if poll_once:
        _status, experiment = api_request("GET", f"/experiment/{experiment_id}")

    manifest_summary: dict[str, Any] = {
        "available": False,
        "path": "",
        "manifest_check": {},
    }
    manifest_dir_text = str(local_manifest_dir).strip()
    if manifest_dir_text:
        manifest_path = Path(manifest_dir_text) / f"{experiment_id}.json"
        frozen_manifest = _wait_for_manifest_file(manifest_path)
        manifest_summary = {
            "available": True,
            "path": str(manifest_path),
            "manifest_check": verify_manifest_uses_lifecycle_package(
                frozen_manifest,
                lifecycle_package,
            ),
            "manifest": frozen_manifest,
        }

    return {
        "mode": "remote-api",
        "api_base_url": base,
        "budget_before": budget_before,
        "duplicate_config": submit_status == 409,
        "submitted": submitted,
        "experiment": experiment,
        "local_manifest": manifest_summary,
    }


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compute":
            summary = _compute_command(args)
        elif args.command == "control":
            summary = _control_command(args)
        else:
            raise LifecycleCheckError(f"unsupported command: {args.command}")
    except LifecycleCheckError as exc:
        print(f"[lifecycle-check] ERROR: {exc}", flush=True)
        return 1

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return int(summary.get("exit_code", 0))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspectively test lifecycle dataset wiring for the API and worker."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute = subparsers.add_parser(
        "compute",
        help="Inspect lifecycle tokenized indexes and optionally run the worker.",
    )
    compute.add_argument("--dataset-dir", required=True)
    compute.add_argument("--work-dir", required=True)
    compute.add_argument("--run-id", default="exp-lifecycle-000001")
    compute.add_argument("--student-id", default="staff-lifecycle")
    compute.add_argument(
        "--callback-url",
        default="http://127.0.0.1/internal/provider-events",
    )
    compute.add_argument("--callback-token-env", default="SCALING_CALLBACK_TOKEN")
    compute.add_argument("--trainer", default="course_trainer.worker:train")
    compute.add_argument("--heartbeat-interval-seconds", type=int, default=0)
    compute.add_argument(
        "--tokenized-index-validation-mode",
        choices=["metadata", "full"],
        default="metadata",
    )
    compute.add_argument(
        "--event-sink",
        choices=["http", "jsonl", "both"],
        default="jsonl",
    )
    compute.add_argument("--event-log-path", default="")
    compute.add_argument("--requested-runtime-seconds", type=int, default=0)
    compute.add_argument("--skip-worker-run", action="store_true")

    control = subparsers.add_parser(
        "control",
        help="Check API lifecycle wiring from a control-node shell.",
    )
    control.add_argument("--dataset-dir", required=True)
    control.add_argument(
        "--mode",
        choices=["remote-api", "local-service"],
        default="remote-api",
    )
    control.add_argument("--api-base-url", default=os.environ.get("SCALING_API_BASE_URL", ""))
    control.add_argument("--api-key", default=os.environ.get("SCALING_API_KEY", ""))
    control.add_argument("--local-manifest-dir", default=os.environ.get("SCALING_LOCAL_MANIFEST_DIR", ""))
    control.add_argument("--student-id", default="staff-lifecycle")
    control.add_argument("--requested-runtime-seconds", type=int, default=0)
    control.add_argument("--unique-model-seed", action="store_true")
    control.add_argument("--request-timeout-seconds", type=float, default=30.0)
    control.add_argument("--no-poll-once", action="store_true")
    return parser


def _compute_command(args: argparse.Namespace) -> dict[str, Any]:
    package = load_lifecycle_package(args.dataset_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    inspection = inspect_lifecycle_tokenized_indexes(
        package,
        validation_mode=args.tokenized_index_validation_mode,
    )
    inspection_path = work_dir / "lifecycle-worker-inspection.json"
    write_json(inspection_path, inspection)

    manifest_path = work_dir / "lifecycle-worker-manifest.json"
    requested_runtime_seconds = (
        args.requested_runtime_seconds if args.requested_runtime_seconds > 0 else None
    )
    manifest = build_lifecycle_worker_manifest(
        package,
        run_id=args.run_id,
        student_id=args.student_id,
        manifest_uri=manifest_path.resolve().as_uri(),
        callback_url=args.callback_url,
        requested_runtime_seconds=requested_runtime_seconds,
    )
    write_json(manifest_path, manifest)
    event_log_path = Path(args.event_log_path) if args.event_log_path else (
        work_dir / "lifecycle-worker-events.jsonl"
    )
    summary: dict[str, Any] = {
        "mode": "compute",
        "dataset_dir": str(Path(args.dataset_dir)),
        "work_dir": str(work_dir),
        "inspection_path": str(inspection_path),
        "worker_manifest_path": str(manifest_path),
        "event_log_path": str(event_log_path),
        "inspection": inspection,
        "manifest_check": verify_manifest_uses_lifecycle_package(manifest, package),
        "worker_ran": False,
        "exit_code": 0,
    }
    if args.skip_worker_run:
        return summary

    from scaling_backend.worker import run as worker_run

    worker_args = [
        "--manifest-uri",
        str(manifest_path),
        "--callback-url",
        args.callback_url,
        "--callback-token-env",
        args.callback_token_env,
        "--trainer",
        args.trainer,
        "--heartbeat-interval-seconds",
        str(args.heartbeat_interval_seconds),
        "--tokenized-index-validation-mode",
        args.tokenized_index_validation_mode,
        "--event-sink",
        args.event_sink,
        "--event-log-path",
        str(event_log_path),
    ]
    exit_code = worker_run.main(worker_args)
    summary["worker_ran"] = True
    summary["exit_code"] = exit_code
    if args.event_sink in {"jsonl", "both"} and event_log_path.exists():
        summary["event_summary"] = _summarize_event_log(event_log_path)
    return summary


def _control_command(args: argparse.Namespace) -> dict[str, Any]:
    package = load_lifecycle_package(args.dataset_dir)
    requested_runtime_seconds = (
        args.requested_runtime_seconds if args.requested_runtime_seconds > 0 else None
    )
    if args.mode == "local-service":
        return run_control_local_service_check(
            package,
            env=os.environ,
            student_id=args.student_id,
            requested_runtime_seconds=requested_runtime_seconds,
            unique_model_seed=args.unique_model_seed,
        )
    return run_control_remote_api_check(
        package,
        base_url=args.api_base_url,
        api_key=args.api_key,
        local_manifest_dir=args.local_manifest_dir,
        requested_runtime_seconds=requested_runtime_seconds,
        unique_model_seed=args.unique_model_seed,
        request_timeout_seconds=args.request_timeout_seconds,
        poll_once=not args.no_poll_once,
    )


def _submit_payload(
    lifecycle_package: Mapping[str, Any],
    *,
    requested_runtime_seconds: int | None = None,
    unique_model_seed: bool = False,
) -> dict[str, Any]:
    payload = copy.deepcopy(_recommended_submit_payload(lifecycle_package))
    if requested_runtime_seconds is not None:
        payload["requested_runtime_seconds"] = _positive_int(
            requested_runtime_seconds,
            "requested_runtime_seconds",
        )
    if unique_model_seed:
        config = _require_mapping(payload, "config")
        training = _require_mapping(config, "training")
        training["model_seed"] = int(time.time())
    return payload


def _recommended_submit_payload(lifecycle_package: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _require_mapping(lifecycle_package, "recommended_submit_payload")
    _require_mapping(payload, "config")
    _positive_int(
        payload.get("requested_runtime_seconds"),
        "recommended_submit_payload.requested_runtime_seconds",
    )
    return payload


def _runtime_overrides(lifecycle_package: Mapping[str, Any]) -> dict[str, str]:
    raw = _require_mapping(lifecycle_package, "runtime_overrides")
    values = {str(key): str(value).strip() for key, value in raw.items()}
    required = [
        "SCALING_DATA_MANIFEST_ID",
        "SCALING_EVAL_MANIFEST_ID",
        "SCALING_FINAL_DATA_MANIFEST_ID",
        "SCALING_FINAL_EVAL_MANIFEST_ID",
        "SCALING_VALIDATION_TOKENS_PER_EVAL",
        "SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL",
        "SCALING_TOKENIZED_TRAIN_INDEX_URI",
        "SCALING_TOKENIZED_VALIDATION_INDEX_URI",
        "SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI",
        "SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI",
    ]
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise LifecycleCheckError(
            "lifecycle runtime_overrides missing required keys: "
            + ", ".join(missing)
        )
    return values


def _wait_for_manifest_file(path: Path, *, max_wait_seconds: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + max_wait_seconds
    while True:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise LifecycleCheckError(
                    f"frozen manifest is not valid JSON: {path}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise LifecycleCheckError(f"frozen manifest must be a JSON object: {path}")
            return payload
        if time.monotonic() >= deadline:
            raise LifecycleCheckError(f"frozen manifest did not appear: {path}")
        time.sleep(0.1)


def _summarize_event_log(path: Path) -> dict[str, Any]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LifecycleCheckError(
                    f"{path}:{line_number}: event is not valid JSON"
                ) from exc
            if not isinstance(event, dict):
                raise LifecycleCheckError(
                    f"{path}:{line_number}: event must be a JSON object"
                )
            events.append(event)
    return {
        "events": len(events),
        "event_types": [str(event.get("event_type", "")) for event in events],
        "validation_events": sum(
            1 for event in events if event.get("event_type") == "validation"
        ),
    }


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise LifecycleCheckError(f"{key} must be a JSON object")
    return value


def _require_env(values: Mapping[str, str], key: str) -> str:
    value = str(values.get(key, "")).strip()
    if not value:
        raise LifecycleCheckError(f"missing lifecycle runtime override: {key}")
    return value


def _non_empty_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise LifecycleCheckError(f"{field_name} must be non-empty")
    return text


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise LifecycleCheckError(f"{field_name} must be positive")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LifecycleCheckError(f"{field_name} must be positive") from exc
    if parsed <= 0:
        raise LifecycleCheckError(f"{field_name} must be positive")
    return parsed


def _validation_mode(value: str) -> str:
    mode = str(value).strip()
    if mode not in {"metadata", "full"}:
        raise LifecycleCheckError("validation_mode must be metadata or full")
    return mode


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
