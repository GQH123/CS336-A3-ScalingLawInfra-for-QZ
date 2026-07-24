from scaling_backend.providers.contracts import ProviderJobStatus
from scaling_backend.providers.fake import FakeProviderAdapter


def _manifest():
    return {
        "experiment_id": "exp-1",
        "student_id": "student-1",
        "manifest_uri": "memory://exp-1",
        "callback_url": "https://backend/internal/provider-events",
    }


def test_fake_provider_submits_deterministic_jobs_and_records_manifests():
    provider = FakeProviderAdapter(now=lambda: "2026-07-16T15:00:00Z")

    first = provider.submit(_manifest())
    second = provider.submit({**_manifest(), "experiment_id": "exp-2"})

    assert first.provider_name == "fake"
    assert first.provider_job_id == "fake-job-000001"
    assert first.submitted_at == "2026-07-16T15:00:00Z"
    assert second.provider_job_id == "fake-job-000002"
    assert provider.submitted_manifests["fake-job-000001"] == _manifest()


def test_fake_provider_status_sequence_sticks_at_last_state():
    provider = FakeProviderAdapter(
        now=lambda: "now",
        status_sequence=[
            ProviderJobStatus.QUEUED,
            ProviderJobStatus.RUNNING,
            ProviderJobStatus.SUCCEEDED,
        ],
    )
    submitted = provider.submit(_manifest())

    first = provider.get_status(submitted.provider_job_id)
    second = provider.get_status(submitted.provider_job_id)
    third = provider.get_status(submitted.provider_job_id)
    fourth = provider.get_status(submitted.provider_job_id)

    assert [first.status, second.status, third.status, fourth.status] == [
        ProviderJobStatus.QUEUED,
        ProviderJobStatus.RUNNING,
        ProviderJobStatus.SUCCEEDED,
        ProviderJobStatus.SUCCEEDED,
    ]
    assert fourth.raw_status == "succeeded"


def test_fake_provider_cancels_jobs_and_returns_artifact_pointers():
    provider = FakeProviderAdapter(now=lambda: "now", artifact_base_uri="memory://logs")
    submitted = provider.submit(_manifest())

    cancel = provider.cancel(
        submitted.provider_job_id,
        reason="student request",
        actor="staff",
    )
    status = provider.get_status(submitted.provider_job_id)
    artifacts = provider.get_artifacts(submitted.provider_job_id)

    assert cancel.accepted is True
    assert cancel.status is ProviderJobStatus.CANCELLED
    assert cancel.metadata == {"reason": "student request", "actor": "staff"}
    assert status.status is ProviderJobStatus.CANCELLED
    assert artifacts.logs_uri == "memory://logs/fake-job-000001/logs.txt"
    assert artifacts.detail_uri == "memory://logs/fake-job-000001/detail.json"
