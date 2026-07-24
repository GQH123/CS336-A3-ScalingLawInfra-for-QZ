from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping

from scaling_backend.providers.contracts import (
    ProviderArtifacts,
    ProviderCancelResult,
    ProviderJobStatus,
    ProviderStatusSnapshot,
    ProviderSubmission,
)
from scaling_backend.providers.qz_distributed.client import QzClient
from scaling_backend.providers.qz_distributed.payloads import (
    QzDistributedConfig,
    QzResourceSpec,
    build_create_payload,
    build_worker_command,
)
from scaling_backend.providers.qz_distributed.statuses import normalize_qz_status


_ACTIVE_STATUSES = {
    ProviderJobStatus.SUBMITTED,
    ProviderJobStatus.QUEUED,
    ProviderJobStatus.RUNNING,
    ProviderJobStatus.UNKNOWN,
}
_LOG_TAIL_LIMIT = 20


class QzDistributedAdapter:
    provider_name = "qz_distributed"

    def __init__(
        self,
        *,
        client: QzClient,
        config: QzDistributedConfig,
        spec: QzResourceSpec,
        callback_token_env: str = "SCALING_CALLBACK_TOKEN",
        callback_token_value: str = "",
        trainer_spec: str = "",
        heartbeat_interval_seconds: int = 0,
        tokenized_index_validation_mode: str = "",
        worker_conda_env: str = "",
        worker_conda_init: str = "",
        worker_event_log_dir: str = "",
        now: Callable[[], str],
    ):
        self.client = client
        self.config = config
        self.spec = spec
        self.callback_token_env = callback_token_env
        self.callback_token_value = callback_token_value
        self.trainer_spec = trainer_spec
        self.heartbeat_interval_seconds = int(heartbeat_interval_seconds)
        self.tokenized_index_validation_mode = tokenized_index_validation_mode
        self.worker_conda_env = worker_conda_env
        self.worker_conda_init = worker_conda_init
        self.worker_event_log_dir = worker_event_log_dir
        self._now = now

    def submit(self, manifest: Mapping[str, Any]) -> ProviderSubmission:
        run_id = _manifest_run_id(manifest)
        manifest_uri = _require_string(manifest, "manifest_uri")
        callback_url = _require_string(manifest, "callback_url")
        event_log_path = _worker_event_log_path(self.worker_event_log_dir, run_id)
        command = build_worker_command(
            manifest_uri=manifest_uri,
            callback_url=callback_url,
            callback_token_env=self.callback_token_env,
            trainer_spec=self.trainer_spec,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
            tokenized_index_validation_mode=self.tokenized_index_validation_mode,
            event_sink="jsonl" if event_log_path else "",
            event_log_path=event_log_path,
            callback_token_value="" if event_log_path else self.callback_token_value,
            conda_env=self.worker_conda_env,
            conda_init=self.worker_conda_init,
        )
        payload = build_create_payload(
            job_name=_job_name(run_id),
            command=command,
            config=self.config,
            spec=self.spec,
            instance_count=int(manifest.get("instance_count", 1)),
        )
        result = self.client.create_train_job(payload)
        provider_job_id = _extract_job_id(result)
        return ProviderSubmission(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            submitted_at=self._now(),
            metadata=self._safe_resource_metadata(),
        )

    def get_status(self, provider_job_id: str) -> ProviderStatusSnapshot:
        detail = self.client.get_train_job_detail(provider_job_id)
        raw_status = str(
            detail.get("status")
            or detail.get("state")
            or detail.get("job_status")
            or ""
        )
        message = str(
            detail.get("message")
            or detail.get("error_message")
            or detail.get("reason")
            or ""
        )
        metadata = {
            key: detail[key]
            for key in (
                "created_at",
                "updated_at",
                "started_at",
                "finished_at",
                "node_names",
            )
            if key in detail
        }
        status = normalize_qz_status(raw_status)
        try:
            logs_metadata = _qz_logs_metadata(
                self.client.get_train_job_logs(provider_job_id)
            )
        except Exception as exc:
            logs_metadata = {"log_fetch_error": type(exc).__name__}
        metadata.update(logs_metadata)
        log_crash_signal = str(logs_metadata.get("log_crash_signal", ""))
        if log_crash_signal:
            message = _append_message_detail(message, log_crash_signal)
            if status in _ACTIVE_STATUSES:
                status = ProviderJobStatus.FAILED
        return ProviderStatusSnapshot(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            status=status,
            raw_status=raw_status,
            message=message,
            metadata=metadata,
        )

    def cancel(
        self, provider_job_id: str, reason: str, actor: str
    ) -> ProviderCancelResult:
        accepted = self.client.stop_train_job(provider_job_id)
        return ProviderCancelResult(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            accepted=accepted,
            status=ProviderJobStatus.CANCELLED if accepted else ProviderJobStatus.UNKNOWN,
            message="stop accepted" if accepted else "stop rejected",
            metadata={"reason": reason, "actor": actor},
        )

    def get_artifacts(self, provider_job_id: str) -> ProviderArtifacts:
        logs = self.client.get_train_job_logs(provider_job_id)
        return ProviderArtifacts(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            logs_uri=f"qz://train_job/{provider_job_id}/logs",
            detail_uri=f"qz://train_job/{provider_job_id}",
            metadata={"log_count": int(logs.get("total") or len(logs.get("logs", [])))},
        )

    def _safe_resource_metadata(self) -> dict[str, str]:
        return {
            "workspace_id": self.config.workspace_id,
            "project_id": self.config.project_id,
            "logic_compute_group_id": self.config.logic_compute_group_id,
            "spec_id": self.spec.id,
        }


def _require_string(manifest: Mapping[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest missing required field: {key}")
    return value


def _manifest_run_id(manifest: Mapping[str, Any]) -> str:
    if "experiment_id" in manifest:
        return _require_string(manifest, "experiment_id")
    return _require_string(manifest, "final_run_id")


def _job_name(experiment_id: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in experiment_id
    ).strip("-")
    return f"scale-{safe or 'experiment'}"


def _worker_event_log_path(root: str, run_id: str) -> str:
    base = str(root).strip().rstrip("/")
    if not base:
        return ""
    safe_run_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in run_id
    ).strip("-")
    return str(PurePosixPath(base) / f"{safe_run_id or 'run'}.jsonl")


def _extract_job_id(result: Mapping[str, Any]) -> str:
    for key in ("job_id", "id", "train_job_id"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("QZ create response did not include a train-job ID")


def _qz_logs_metadata(logs: Mapping[str, Any]) -> dict[str, Any]:
    entries = logs.get("logs", [])
    log_lines = _qz_log_lines(entries)
    total = logs.get("total")
    try:
        log_count = int(total)
    except (TypeError, ValueError):
        log_count = len(log_lines)
    tail = log_lines[-_LOG_TAIL_LIMIT:]
    metadata: dict[str, Any] = {
        "log_count": log_count,
        "log_tail": tail,
    }
    crash_signal = _qz_log_crash_signal(log_lines)
    if crash_signal:
        metadata["log_crash_signal"] = crash_signal
    return metadata


def _qz_log_lines(entries: Any) -> list[str]:
    if not isinstance(entries, list):
        return []
    lines: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            line = entry.strip()
        elif isinstance(entry, Mapping):
            line = str(
                entry.get("message")
                or entry.get("content")
                or entry.get("log")
                or entry.get("text")
                or ""
            ).strip()
        else:
            line = ""
        if line:
            lines.append(line)
    return lines


def _qz_log_crash_signal(log_lines: list[str]) -> str:
    for line in reversed(log_lines):
        match = re.search(r"\bprocessExitCode:\s*([0-9]+)\b", line)
        if match and int(match.group(1)) != 0:
            return f"processExitCode={match.group(1)}"
    for line in reversed(log_lines):
        match = re.search(r"\bexit code\s+([0-9]+)\b", line, flags=re.IGNORECASE)
        if match and int(match.group(1)) != 0:
            return f"exit_code={match.group(1)}"
    for line in reversed(log_lines):
        lowered = line.lower()
        if "segmentation fault" in lowered:
            return "segmentation fault"
        if "oomkilled" in lowered or (
            "out of memory" in lowered and "killed" in lowered
        ):
            return "worker killed by out-of-memory condition"
    return ""


def _append_message_detail(message: str, detail: str) -> str:
    if not message:
        return detail
    if detail in message:
        return message
    return f"{message}; {detail}"
