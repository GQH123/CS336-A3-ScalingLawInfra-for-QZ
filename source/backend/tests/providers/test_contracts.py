from scaling_backend.providers.contracts import (
    ProviderArtifacts,
    ProviderCancelResult,
    ProviderJobStatus,
    ProviderStatusSnapshot,
    ProviderSubmission,
)


def test_provider_contracts_use_stable_status_values():
    assert [status.value for status in ProviderJobStatus] == [
        "submitted",
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "lost",
        "unknown",
    ]


def test_provider_result_objects_are_plain_data_containers():
    submission = ProviderSubmission(
        provider_name="qz_distributed",
        provider_job_id="job-123",
        submitted_at="2026-07-16T15:00:00Z",
        metadata={"workspace_id": "ws-redacted"},
    )
    status = ProviderStatusSnapshot(
        provider_name="qz_distributed",
        provider_job_id="job-123",
        status=ProviderJobStatus.RUNNING,
        raw_status="RUNNING",
        message="worker started",
        metadata={"node": "gpu-node"},
    )
    cancel = ProviderCancelResult(
        provider_name="qz_distributed",
        provider_job_id="job-123",
        accepted=True,
        status=ProviderJobStatus.CANCELLED,
        message="stop accepted",
    )
    artifacts = ProviderArtifacts(
        provider_name="qz_distributed",
        provider_job_id="job-123",
        logs_uri="qz://train_job/job-123/logs",
        metadata={"detail_uri": "qz://train_job/job-123"},
    )

    assert submission.provider_job_id == status.provider_job_id
    assert cancel.status is ProviderJobStatus.CANCELLED
    assert artifacts.logs_uri == "qz://train_job/job-123/logs"

