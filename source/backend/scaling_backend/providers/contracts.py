from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


class ProviderJobStatus(str, Enum):
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderSubmission:
    provider_name: str
    provider_job_id: str
    submitted_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderStatusSnapshot:
    provider_name: str
    provider_job_id: str
    status: ProviderJobStatus
    raw_status: str = ""
    message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCancelResult:
    provider_name: str
    provider_job_id: str
    accepted: bool
    status: ProviderJobStatus = ProviderJobStatus.UNKNOWN
    message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderArtifacts:
    provider_name: str
    provider_job_id: str
    logs_uri: str = ""
    detail_uri: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ProviderAdapter(Protocol):
    provider_name: str

    def submit(self, manifest: Mapping[str, Any]) -> ProviderSubmission:
        ...

    def get_status(self, provider_job_id: str) -> ProviderStatusSnapshot:
        ...

    def cancel(
        self, provider_job_id: str, reason: str, actor: str
    ) -> ProviderCancelResult:
        ...

    def get_artifacts(self, provider_job_id: str) -> ProviderArtifacts:
        ...

