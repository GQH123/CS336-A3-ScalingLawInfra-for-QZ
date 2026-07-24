from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from scaling_backend.providers.contracts import (
    ProviderArtifacts,
    ProviderCancelResult,
    ProviderJobStatus,
    ProviderStatusSnapshot,
    ProviderSubmission,
)


@dataclass
class _FakeJob:
    manifest: Mapping[str, object]
    poll_count: int = 0
    cancelled: bool = False


@dataclass
class FakeProviderAdapter:
    now: Callable[[], str]
    status_sequence: list[ProviderJobStatus] = field(
        default_factory=lambda: [ProviderJobStatus.SUBMITTED]
    )
    artifact_base_uri: str = "memory://fake-provider"
    provider_name: str = "fake"

    def __post_init__(self) -> None:
        self.submitted_manifests: dict[str, Mapping[str, object]] = {}
        self._jobs: dict[str, _FakeJob] = {}
        self._next_job_number = 1

    def submit(self, manifest: Mapping[str, object]) -> ProviderSubmission:
        provider_job_id = f"fake-job-{self._next_job_number:06d}"
        self._next_job_number += 1
        frozen_manifest = dict(manifest)
        self.submitted_manifests[provider_job_id] = frozen_manifest
        self._jobs[provider_job_id] = _FakeJob(manifest=frozen_manifest)
        return ProviderSubmission(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            submitted_at=self.now(),
        )

    def restore_job(
        self,
        provider_job_id: str,
        manifest: Mapping[str, object],
        *,
        cancelled: bool = False,
    ) -> None:
        frozen_manifest = dict(manifest)
        self.submitted_manifests[provider_job_id] = frozen_manifest
        self._jobs[provider_job_id] = _FakeJob(
            manifest=frozen_manifest,
            cancelled=cancelled,
        )
        prefix = "fake-job-"
        if provider_job_id.startswith(prefix):
            try:
                restored_number = int(provider_job_id.removeprefix(prefix))
            except ValueError:
                return
            self._next_job_number = max(self._next_job_number, restored_number + 1)

    def get_status(self, provider_job_id: str) -> ProviderStatusSnapshot:
        job = self._jobs[provider_job_id]
        if job.cancelled:
            status = ProviderJobStatus.CANCELLED
        else:
            index = min(job.poll_count, len(self.status_sequence) - 1)
            status = self.status_sequence[index]
            job.poll_count += 1
        return ProviderStatusSnapshot(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            status=status,
            raw_status=status.value,
        )

    def cancel(
        self, provider_job_id: str, reason: str, actor: str
    ) -> ProviderCancelResult:
        job = self._jobs[provider_job_id]
        job.cancelled = True
        return ProviderCancelResult(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            accepted=True,
            status=ProviderJobStatus.CANCELLED,
            message="cancelled",
            metadata={"reason": reason, "actor": actor},
        )

    def get_artifacts(self, provider_job_id: str) -> ProviderArtifacts:
        if provider_job_id not in self._jobs:
            raise KeyError(provider_job_id)
        base = self.artifact_base_uri.rstrip("/")
        return ProviderArtifacts(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            logs_uri=f"{base}/{provider_job_id}/logs.txt",
            detail_uri=f"{base}/{provider_job_id}/detail.json",
        )
