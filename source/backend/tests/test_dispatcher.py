from scaling_backend.dispatcher import ExperimentDispatcher, ExperimentStatus
from scaling_backend.providers.contracts import (
    ProviderCancelResult,
    ProviderJobStatus,
    ProviderStatusSnapshot,
    ProviderSubmission,
)


class FakeProvider:
    provider_name = "fake"

    def __init__(self):
        self.submitted_manifests = []
        self.cancelled = []
        self.next_status = ProviderStatusSnapshot(
            provider_name="fake",
            provider_job_id="provider-job-1",
            status=ProviderJobStatus.RUNNING,
            raw_status="RUNNING",
        )

    def submit(self, manifest):
        self.submitted_manifests.append(manifest)
        return ProviderSubmission(
            provider_name="fake",
            provider_job_id="provider-job-1",
            submitted_at="2026-07-16T15:00:00Z",
            metadata={"queue": "test"},
        )

    def get_status(self, provider_job_id):
        assert provider_job_id == "provider-job-1"
        return self.next_status

    def cancel(self, provider_job_id, reason, actor):
        self.cancelled.append((provider_job_id, reason, actor))
        return ProviderCancelResult(
            provider_name="fake",
            provider_job_id=provider_job_id,
            accepted=True,
            status=ProviderJobStatus.CANCELLED,
            message="cancelled",
            metadata={"reason": reason, "actor": actor},
        )

    def get_artifacts(self, provider_job_id):
        raise AssertionError("not used in dispatcher tests")


def _manifest():
    return {
        "experiment_id": "exp-1",
        "student_id": "student-1",
        "manifest_uri": "s3://bucket/exp-1.json",
        "callback_url": "https://backend/internal/provider-events",
    }


def test_submit_records_provider_submission_and_initial_events():
    provider = FakeProvider()
    dispatcher = ExperimentDispatcher(provider=provider, now=lambda: "now")

    record = dispatcher.submit(_manifest())

    assert provider.submitted_manifests == [_manifest()]
    assert record.experiment_id == "exp-1"
    assert record.student_id == "student-1"
    assert record.status is ExperimentStatus.SUBMITTED
    assert record.provider_name == "fake"
    assert record.provider_job_id == "provider-job-1"
    assert record.events[0].type == "experiment_submitted"
    assert record.events[1].type == "provider_submitted"
    assert record.events[1].metadata == {"queue": "test"}


def test_submit_rejects_duplicate_experiment_id():
    dispatcher = ExperimentDispatcher(provider=FakeProvider(), now=lambda: "now")
    dispatcher.submit(_manifest())

    try:
        dispatcher.submit(_manifest())
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("duplicate experiment_id should fail")


def test_poll_updates_experiment_status_from_provider_status():
    provider = FakeProvider()
    dispatcher = ExperimentDispatcher(provider=provider, now=lambda: "now")
    dispatcher.submit(_manifest())

    provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.SUCCEEDED,
        raw_status="COMPLETED",
        message="done",
    )
    record = dispatcher.poll("exp-1")

    assert record.status is ExperimentStatus.COMPLETED
    assert record.provider_status is ProviderJobStatus.SUCCEEDED
    assert record.events[-1].type == "provider_status"
    assert record.events[-1].metadata == {
        "provider_status": "succeeded",
        "raw_status": "COMPLETED",
        "message": "done",
    }


def test_cancel_records_cancel_result_and_status():
    provider = FakeProvider()
    dispatcher = ExperimentDispatcher(provider=provider, now=lambda: "now")
    dispatcher.submit(_manifest())

    record = dispatcher.cancel("exp-1", reason="quota audit", actor="staff")

    assert provider.cancelled == [("provider-job-1", "quota audit", "staff")]
    assert record.status is ExperimentStatus.CANCELLED
    assert record.events[-1].type == "provider_cancelled"
    assert record.events[-1].metadata == {
        "accepted": True,
        "reason": "quota audit",
        "actor": "staff",
        "message": "cancelled",
    }


def test_poll_unknown_experiment_id_fails_clearly():
    dispatcher = ExperimentDispatcher(provider=FakeProvider(), now=lambda: "now")

    try:
        dispatcher.poll("missing")
    except KeyError as exc:
        assert "unknown experiment_id" in str(exc)
    else:
        raise AssertionError("unknown experiment_id should fail")

