from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from scaling_backend.providers.contracts import (
    ProviderAdapter,
    ProviderJobStatus,
)


class ExperimentStatus(str, Enum):
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SYSTEM_FAILED = "system_failed"
    CANCELLED = "cancelled"
    LOST = "lost"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ExperimentEvent:
    type: str
    timestamp: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRecord:
    experiment_id: str
    student_id: str
    status: ExperimentStatus
    provider_name: str = ""
    provider_job_id: str = ""
    provider_status: ProviderJobStatus = ProviderJobStatus.UNKNOWN
    manifest: Mapping[str, Any] = field(default_factory=dict)
    events: list[ExperimentEvent] = field(default_factory=list)


class ExperimentDispatcher:
    def __init__(self, *, provider: ProviderAdapter, now: Callable[[], str]):
        self.provider = provider
        self._now = now
        self._records: dict[str, ExperimentRecord] = {}

    def submit(self, manifest: Mapping[str, Any]) -> ExperimentRecord:
        experiment_id = _require_string(manifest, "experiment_id")
        student_id = _require_string(manifest, "student_id")
        if experiment_id in self._records:
            raise ValueError(f"experiment_id already exists: {experiment_id}")

        record = ExperimentRecord(
            experiment_id=experiment_id,
            student_id=student_id,
            status=ExperimentStatus.SUBMITTED,
            manifest=dict(manifest),
            events=[
                ExperimentEvent(
                    type="experiment_submitted",
                    timestamp=self._now(),
                    metadata={"student_id": student_id},
                )
            ],
        )
        submission = self.provider.submit(manifest)
        record.provider_name = submission.provider_name
        record.provider_job_id = submission.provider_job_id
        record.provider_status = ProviderJobStatus.SUBMITTED
        record.events.append(
            ExperimentEvent(
                type="provider_submitted",
                timestamp=self._now(),
                metadata=dict(submission.metadata),
            )
        )
        self._records[experiment_id] = record
        return record

    def poll(self, experiment_id: str) -> ExperimentRecord:
        record = self._get_record(experiment_id)
        snapshot = self.provider.get_status(record.provider_job_id)
        record.provider_status = snapshot.status
        record.status = _experiment_status_from_provider(snapshot.status)
        metadata = {
            "provider_status": snapshot.status.value,
            "raw_status": snapshot.raw_status,
            "message": snapshot.message,
        }
        if snapshot.metadata:
            metadata["snapshot_metadata"] = dict(snapshot.metadata)
        record.events.append(
            ExperimentEvent(
                type="provider_status",
                timestamp=self._now(),
                metadata=metadata,
            )
        )
        return record

    def cancel(self, experiment_id: str, *, reason: str, actor: str) -> ExperimentRecord:
        record = self._get_record(experiment_id)
        result = self.provider.cancel(record.provider_job_id, reason=reason, actor=actor)
        record.provider_status = result.status
        record.status = _experiment_status_from_provider(result.status)
        record.events.append(
            ExperimentEvent(
                type="provider_cancelled",
                timestamp=self._now(),
                metadata={
                    "accepted": result.accepted,
                    "reason": reason,
                    "actor": actor,
                    "message": result.message,
                },
            )
        )
        return record

    def get(self, experiment_id: str) -> ExperimentRecord:
        return self._get_record(experiment_id)

    def _get_record(self, experiment_id: str) -> ExperimentRecord:
        try:
            return self._records[experiment_id]
        except KeyError as exc:
            raise KeyError(f"unknown experiment_id: {experiment_id}") from exc


def _experiment_status_from_provider(
    provider_status: ProviderJobStatus,
) -> ExperimentStatus:
    return {
        ProviderJobStatus.SUBMITTED: ExperimentStatus.SUBMITTED,
        ProviderJobStatus.QUEUED: ExperimentStatus.QUEUED,
        ProviderJobStatus.RUNNING: ExperimentStatus.RUNNING,
        ProviderJobStatus.SUCCEEDED: ExperimentStatus.COMPLETED,
        ProviderJobStatus.FAILED: ExperimentStatus.FAILED,
        ProviderJobStatus.CANCELLED: ExperimentStatus.CANCELLED,
        ProviderJobStatus.LOST: ExperimentStatus.LOST,
        ProviderJobStatus.UNKNOWN: ExperimentStatus.UNKNOWN,
    }[provider_status]


def _require_string(manifest: Mapping[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest missing required field: {key}")
    return value
