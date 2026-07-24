import json

import pytest

from scaling_backend.manifest_store import LocalManifestStore
from scaling_backend.providers.contracts import (
    ProviderCancelResult,
    ProviderArtifacts,
    ProviderJobStatus,
    ProviderStatusSnapshot,
    ProviderSubmission,
)
from scaling_backend.providers.fake import FakeProviderAdapter
from scaling_backend.service import (
    DuplicateExperimentError,
    ExperimentService,
    StudentAccessError,
)


class FakeProvider:
    provider_name = "fake"

    def __init__(self):
        self.submitted = []
        self.cancelled = []
        self.next_status = ProviderStatusSnapshot(
            provider_name="fake",
            provider_job_id="provider-job-1",
            status=ProviderJobStatus.RUNNING,
            raw_status="RUNNING",
            message="container started",
        )

    def submit(self, manifest):
        self.submitted.append(dict(manifest))
        return ProviderSubmission(
            provider_name="fake",
            provider_job_id=f"provider-job-{len(self.submitted)}",
            submitted_at="2026-07-16T15:00:00Z",
        )

    def get_status(self, provider_job_id):
        assert provider_job_id.startswith("provider-job-")
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
        return ProviderArtifacts(
            provider_name="fake",
            provider_job_id=provider_job_id,
            logs_uri=f"memory://fake-provider/{provider_job_id}/logs.txt",
            detail_uri=f"memory://fake-provider/{provider_job_id}/detail.json",
        )


class FailOnceOnFinalSubmissionProvider(FakeProvider):
    def __init__(self, *, failing_final_submission_id):
        super().__init__()
        self.failing_final_submission_id = failing_final_submission_id
        self.failed = False

    def submit(self, manifest):
        if (
            manifest.get("run_kind") == "final"
            and manifest.get("final_submission_id") == self.failing_final_submission_id
            and not self.failed
        ):
            self.failed = True
            raise RuntimeError("simulated provider launch failure")
        return super().submit(manifest)


class FailOnceOnExploratoryProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.failed = False

    def submit(self, manifest):
        if manifest.get("run_kind") == "exploratory" and not self.failed:
            self.failed = True
            raise RuntimeError("simulated exploratory provider failure")
        return super().submit(manifest)


class FailOnSecondExploratoryProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.failed = False

    def submit(self, manifest):
        if (
            manifest.get("run_kind") == "exploratory"
            and len(self.submitted) >= 1
            and not self.failed
        ):
            self.failed = True
            raise RuntimeError("simulated queued launch failure")
        return super().submit(manifest)


def _config(train_tokens=1024):
    return {
        "model": {"num_hidden_layers": 2, "hidden_size": 128},
        "training": {
            "train_tokens": train_tokens,
            "learning_rate": 3e-4,
            "num_evals": 1,
        },
    }


def _service(**kwargs):
    return ExperimentService(
        provider=FakeProvider(),
        total_budget_seconds=12 * 3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda experiment_id: f"memory://manifests/{experiment_id}.json",
        callback_url="https://backend/internal/provider-events",
        **kwargs,
    )


def test_submit_reserves_budget_and_dispatches_frozen_manifest():
    service = _service()

    response = service.submit(
        student_id="student-1",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=600,
    )

    assert response.experiment_id == "exp-000001"
    assert response.status == "submitted"
    assert response.budget_reserved_seconds == 600
    assert response.resolved_config["tokens_per_optimizer_step"] == 1024
    assert response.resolved_config["total_optimizer_steps"] == 2
    assert response.resolved_config["data_manifest_id"] == "exploratory-train-v0"
    assert service.get_budget("student-1").remaining_seconds == 12 * 3600 - 600

    provider_manifest = service.provider.submitted[0]
    assert provider_manifest["manifest_version"] == 1
    assert provider_manifest["run_kind"] == "exploratory"
    assert provider_manifest["experiment_id"] == "exp-000001"
    assert provider_manifest["student_id"] == "student-1"
    assert provider_manifest["manifest_uri"] == "memory://manifests/exp-000001.json"
    assert provider_manifest["callback_url"] == "https://backend/internal/provider-events"
    assert provider_manifest["callback_base_url"] == "https://backend/internal/provider-events"
    assert provider_manifest["model_config"] == response.resolved_config["model"]
    assert provider_manifest["training_config"] == response.resolved_config["training"]
    assert provider_manifest["resolved_config"] == response.resolved_config
    assert provider_manifest["code_version"] == "local-dev"
    assert provider_manifest["data_manifest_id"] == "exploratory-train-v0"
    assert provider_manifest["eval_manifest_id"] == "exploratory-eval-v0"
    assert provider_manifest["data_config"] == {"train_tokens": 2048}
    assert provider_manifest["max_runtime_seconds"] == 600
    assert provider_manifest["created_at"] == "2026-07-16T15:00:00Z"
    assert provider_manifest["runtime_config"]["reserved_runtime_seconds"] == 600
    assert provider_manifest["runtime_config"]["max_runtime_seconds"] == 600


def test_submit_threads_tokenized_index_uris_into_exploratory_manifest():
    service = _service(
        train_tokenized_index_uri="file:///secure/course/tokenized/train/index.json",
        validation_tokenized_index_uri=(
            "https://storage.example/tokenized/exploratory-eval/index.json"
        ),
    )

    response = service.submit(
        student_id="student-1",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=600,
    )

    provider_manifest = service.provider.submitted[0]
    assert provider_manifest["data_config"] == {
        "train_tokens": 2048,
        "tokenized_index_uri": "file:///secure/course/tokenized/train/index.json",
    }
    assert provider_manifest["validation_config"]["tokenized_index_uri"] == (
        "https://storage.example/tokenized/exploratory-eval/index.json"
    )
    assert "tokenized_index_uri" not in response.resolved_config


def test_submit_and_results_expose_resource_warnings_as_first_class_metadata():
    service = _service()

    response = service.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {
                "train_tokens": 1024,
                "learning_rate": 0.02,
                "num_evals": 1,
            },
        },
        requested_runtime_seconds=30,
    )
    result = service.get_result("student-1", response.experiment_id)

    assert response.resource_warnings == [
        "very_low_optimizer_steps",
        "high_learning_rate",
        "very_low_runtime_budget",
    ]
    assert response.resolved_config["resource_warnings"] == response.resource_warnings
    assert result.resource_warnings == response.resource_warnings


def test_submit_writes_frozen_manifest_before_provider_dispatch(tmp_path):
    store = LocalManifestStore(tmp_path)
    service = ExperimentService(
        provider=FakeProvider(),
        total_budget_seconds=12 * 3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda experiment_id: f"memory://placeholder/{experiment_id}",
        callback_url="https://backend/internal/provider-events",
        manifest_store=store,
    )

    service.submit(
        student_id="student-1",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=600,
    )

    provider_manifest = service.provider.submitted[0]
    manifest_uri = (tmp_path / "exp-000001.json").resolve().as_uri()
    exported = service.admin_export_course_records()
    assert provider_manifest["manifest_uri"] == manifest_uri
    assert exported["experiments"][0]["resolved_config"]["provider_manifest_uri"] == manifest_uri
    assert provider_manifest["resolved_config"]["provider_manifest_uri"] == manifest_uri
    assert (tmp_path / "exp-000001.json").exists()
    persisted = json.loads((tmp_path / "exp-000001.json").read_text(encoding="utf-8"))
    assert persisted["experiment_id"] == "exp-000001"
    assert persisted["resolved_config"]["provider_manifest_uri"] == manifest_uri


def test_submit_provider_failure_rolls_back_budget_indexes_and_experiment_id():
    provider = FailOnceOnExploratoryProvider()
    service = ExperimentService(
        provider=provider,
        total_budget_seconds=12 * 3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda experiment_id: f"memory://manifests/{experiment_id}.json",
        callback_url="https://backend/internal/provider-events",
    )

    with pytest.raises(RuntimeError, match="simulated exploratory provider failure"):
        service.submit(
            student_id="student-1",
            config=_config(train_tokens=2048),
            requested_runtime_seconds=600,
        )
    retry = service.submit(
        student_id="student-1",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=600,
    )

    assert retry.experiment_id == "exp-000001"
    assert service.get_budget("student-1").reserved_seconds == 600
    assert service.get_budget("student-1").remaining_seconds == 12 * 3600 - 600
    assert service.admin_list_experiments() == [
        {
            "experiment_id": "exp-000001",
            "student_id": "student-1",
            "status": "submitted",
            "validation_losses": [],
            "final_validation_loss": None,
            "failure_reason": "",
            "staff_failure_detail": "",
            "used_runtime_seconds": None,
            "completed_at": "",
            "failed_at": "",
            "reserved_runtime_seconds": 600,
            "config_hash": service.admin_list_experiments()[0]["config_hash"],
            "provider_name": "fake",
            "provider_job_id": "provider-job-1",
            "provider_status": "submitted",
            "provider_artifacts": {
                "provider_name": "fake",
                "provider_job_id": "provider-job-1",
                "logs_uri": "memory://fake-provider/provider-job-1/logs.txt",
                "detail_uri": "memory://fake-provider/provider-job-1/detail.json",
                "metadata": {},
            },
            "resolved_config": service.admin_list_experiments()[0]["resolved_config"],
        }
    ]
    assert [manifest["experiment_id"] for manifest in provider.submitted] == [
        "exp-000001"
    ]


def test_submit_rejects_runtime_that_exceeds_remaining_budget():
    service = _service()

    with pytest.raises(ValueError, match="exceeds remaining budget"):
        service.submit(
            student_id="student-1",
            config=_config(),
            requested_runtime_seconds=12 * 3600 + 1,
        )


def test_submit_rejects_invalid_config_without_budget_or_provider_side_effects():
    service = _service()

    with pytest.raises(ValueError, match="config.model"):
        service.submit(
            student_id="student-1",
            config={"training": {"train_tokens": 1024}},
            requested_runtime_seconds=600,
        )
    accepted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    assert accepted.experiment_id == "exp-000001"
    assert service.get_budget("student-1").reserved_seconds == 600
    assert len(service.provider.submitted) == 1


def test_submit_rejects_non_finite_or_inconsistent_config_fields_without_side_effects():
    service = _service()

    with pytest.raises(ValueError, match="model.hidden_size must be divisible"):
        service.submit(
            student_id="student-1",
            config={
                "model": {
                    "num_hidden_layers": 2,
                    "hidden_size": 130,
                    "num_attention_heads": 8,
                },
                "training": {"train_tokens": 1024, "learning_rate": 3e-4},
            },
            requested_runtime_seconds=600,
        )
    with pytest.raises(
        ValueError,
        match="training.learning_rate must be positive and finite",
    ):
        service.submit(
            student_id="student-1",
            config={
                "model": {"num_hidden_layers": 2, "hidden_size": 128},
                "training": {"train_tokens": 1024, "learning_rate": float("nan")},
            },
            requested_runtime_seconds=600,
        )

    assert service.get_budget("student-1").reserved_seconds == 0
    assert service.get_budget("student-1").charged_seconds == 0
    assert len(service.provider.submitted) == 0


def test_duplicate_same_student_config_returns_existing_id_without_second_charge():
    service = _service()
    first = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    with pytest.raises(DuplicateExperimentError) as exc_info:
        service.submit(
            student_id="student-1",
            config=_config(),
            requested_runtime_seconds=600,
        )

    assert exc_info.value.experiment_id == first.experiment_id
    assert service.get_budget("student-1").reserved_seconds == 600
    assert len(service.provider.submitted) == 1


def test_duplicate_cross_student_config_creates_separate_experiment():
    service = _service()
    first = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )
    second = service.submit(
        student_id="student-2",
        config=_config(),
        requested_runtime_seconds=600,
    )

    assert first.experiment_id == "exp-000001"
    assert second.experiment_id == "exp-000002"
    assert len(service.provider.submitted) == 2


def test_submit_defers_when_student_active_experiment_limit_is_reached():
    service = _service(max_active_experiments_per_student=1)

    first = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    second = service.submit(
        student_id="student-1",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )

    assert first.experiment_id == "exp-000001"
    assert first.status == "submitted"
    assert second.experiment_id == "exp-000002"
    assert second.status == "queued"
    assert service.get_budget("student-1").reserved_seconds == 900
    assert len(service.provider.submitted) == 1
    assert [manifest["experiment_id"] for manifest in service.provider.submitted] == [
        "exp-000001"
    ]

    service.record_worker_event(
        {
            "experiment_id": first.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.4],
            "final_validation_loss": 3.4,
            "actual_runtime_seconds": 120,
        }
    )

    assert service.get_budget("student-1").reserved_seconds == 300
    assert service.get_budget("student-1").charged_seconds == 120
    assert len(service.provider.submitted) == 2
    assert [manifest["experiment_id"] for manifest in service.provider.submitted] == [
        "exp-000001",
        "exp-000002",
    ]
    assert service.get_result("student-1", second.experiment_id).status == "submitted"


def test_cancelled_experiments_do_not_count_against_student_active_limit():
    service = _service(max_active_experiments_per_student=1)
    first = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )

    service.admin_cancel_experiment(
        first.experiment_id,
        reason="staff cleanup",
        actor="staff",
    )
    second = service.submit(
        student_id="student-1",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )

    assert second.experiment_id == "exp-000002"
    assert service.get_budget("student-1").reserved_seconds == 300
    assert len(service.provider.submitted) == 2


def test_submit_defers_when_global_active_experiment_limit_is_reached():
    service = _service(max_active_experiments_global=1)

    first = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    second = service.submit(
        student_id="student-2",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )

    assert first.experiment_id == "exp-000001"
    assert first.status == "submitted"
    assert second.experiment_id == "exp-000002"
    assert second.status == "queued"
    assert service.get_budget("student-1").reserved_seconds == 600
    assert service.get_budget("student-2").reserved_seconds == 300
    assert len(service.provider.submitted) == 1
    assert [manifest["experiment_id"] for manifest in service.provider.submitted] == [
        "exp-000001"
    ]

    service.record_worker_event(
        {
            "experiment_id": first.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.4],
            "final_validation_loss": 3.4,
            "actual_runtime_seconds": 120,
        }
    )

    assert service.get_budget("student-2").reserved_seconds == 300
    assert len(service.provider.submitted) == 2
    assert [manifest["experiment_id"] for manifest in service.provider.submitted] == [
        "exp-000001",
        "exp-000002",
    ]
    assert service.get_result("student-2", second.experiment_id).status == "submitted"


def test_student_active_experiment_limit_is_independent_by_student():
    service = _service(max_active_experiments_per_student=1)

    first = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    second = service.submit(
        student_id="student-2",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )

    assert first.experiment_id == "exp-000001"
    assert second.experiment_id == "exp-000002"
    assert service.get_budget("student-1").reserved_seconds == 600
    assert service.get_budget("student-2").reserved_seconds == 300
    assert len(service.provider.submitted) == 2


def test_callbacks_record_validation_losses_and_completed_result():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "validation",
            "step": 100,
            "loss": 3.4,
        }
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.4, 3.1],
            "final_validation_loss": 3.1,
            "actual_runtime_seconds": 420,
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    assert result.status == "completed"
    assert result.validation_losses == [3.4, 3.1]
    assert result.final_validation_loss == 3.1
    assert service.get_budget("student-1").charged_seconds == 420
    assert service.get_budget("student-1").reserved_seconds == 0


def test_worker_event_event_id_makes_terminal_replay_idempotent():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )
    completed_event = {
        "experiment_id": submitted.experiment_id,
        "event_type": "worker_completed",
        "event_id": "exp-000001:000001",
        "validation_losses": [3.4],
        "final_validation_loss": 3.4,
        "actual_runtime_seconds": 120,
    }

    assert service.record_worker_event(completed_event) is True
    assert service.record_worker_event(completed_event) is False

    result = service.get_result("student-1", submitted.experiment_id)
    events = service.admin_export_course_records()["worker_events"]
    assert result.status == "completed"
    assert service.get_budget("student-1").charged_seconds == 120
    assert len(events) == 1
    assert events[0]["payload"]["event_id"] == "exp-000001:000001"


def test_completed_result_can_use_accumulated_validation_tail():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "validation",
            "step": 100,
            "loss": 3.4,
        }
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "validation",
            "step": 200,
            "loss": 3.1,
        }
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_completed",
            "actual_runtime_seconds": 420,
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    assert result.status == "completed"
    assert result.validation_losses == [3.4, 3.1]
    assert result.final_validation_loss == 3.1


def test_worker_started_heartbeat_and_validation_events_are_ledgered_for_admin_detail():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_started",
            "provider_job_id": "provider-job-1",
        }
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "heartbeat",
            "step": 42,
            "message": "still training",
        }
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "validation",
            "step": 100,
            "loss": 3.4,
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    detail = service.admin_get_experiment(submitted.experiment_id)

    assert result.status == "running"
    assert detail["status"] == "running"
    assert [event["event_type"] for event in detail["events"]] == [
        "worker_started",
        "heartbeat",
        "validation",
    ]
    assert detail["events"][1]["payload"]["message"] == "still training"
    assert detail["events"][2]["payload"]["loss"] == 3.4


def test_callback_rejects_unknown_experiment_event_type_without_mutating_state():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    with pytest.raises(ValueError, match="unknown worker event_type"):
        service.record_worker_event(
            {
                "experiment_id": submitted.experiment_id,
                "event_type": "worker_compeleted",
                "final_validation_loss": 3.1,
            }
        )

    result = service.get_result("student-1", submitted.experiment_id)
    detail = service.admin_get_experiment(submitted.experiment_id)
    assert result.status == "submitted"
    assert result.final_validation_loss is None
    assert result.validation_losses == []
    assert detail["events"] == []


def test_callback_rejects_non_finite_validation_or_final_loss():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    with pytest.raises(ValueError, match="loss values must be finite"):
        service.record_worker_event(
            {
                "experiment_id": submitted.experiment_id,
                "event_type": "validation",
                "loss": float("nan"),
            }
        )
    with pytest.raises(ValueError, match="loss values must be finite"):
        service.record_worker_event(
            {
                "experiment_id": submitted.experiment_id,
                "event_type": "worker_completed",
                "validation_losses": [3.1],
                "final_validation_loss": float("inf"),
                "actual_runtime_seconds": 420,
            }
        )

    result = service.get_result("student-1", submitted.experiment_id)
    assert result.status == "submitted"
    assert result.validation_losses == []
    assert service.get_budget("student-1").reserved_seconds == 600
    assert service.get_budget("student-1").charged_seconds == 0
    assert service.admin_get_experiment(submitted.experiment_id)["events"] == []


def test_completed_runtime_charge_is_clipped_to_positive_reserved_range():
    service = _service()
    zero_runtime = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    over_runtime = service.submit(
        student_id="student-1",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": zero_runtime.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.2],
            "final_validation_loss": 3.2,
            "actual_runtime_seconds": 0,
        }
    )
    service.record_worker_event(
        {
            "experiment_id": over_runtime.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.0],
            "final_validation_loss": 3.0,
            "actual_runtime_seconds": 9999,
        }
    )

    assert service.get_budget("student-1").charged_seconds == 601
    assert service.get_budget("student-1").reserved_seconds == 0


def test_completed_result_exposes_used_runtime_and_completion_timestamp():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.2],
            "final_validation_loss": 3.2,
            "actual_runtime_seconds": 420,
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    detail = service.admin_get_experiment(submitted.experiment_id)
    exported = service.admin_export_course_records()["experiments"][0]

    assert result.used_runtime_seconds == 420
    assert result.completed_at == "2026-07-16T15:00:00Z"
    assert result.failed_at == ""
    assert detail["used_runtime_seconds"] == 420
    assert detail["completed_at"] == "2026-07-16T15:00:00Z"
    assert detail["failed_at"] == ""
    assert exported["used_runtime_seconds"] == 420
    assert exported["completed_at"] == "2026-07-16T15:00:00Z"
    assert exported["failed_at"] == ""


def test_completed_result_derives_final_loss_from_last_validation_loss():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.5, 3.25],
            "actual_runtime_seconds": 420,
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    exported = service.admin_export_course_records()["experiments"][0]
    assert result.status == "completed"
    assert result.validation_losses == [3.5, 3.25]
    assert result.final_validation_loss == 3.25
    assert exported["final_validation_loss"] == 3.25


def test_completed_result_rejects_final_loss_that_disagrees_with_validation_tail():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    with pytest.raises(ValueError, match="final_validation_loss must equal"):
        service.record_worker_event(
            {
                "experiment_id": submitted.experiment_id,
                "event_type": "worker_completed",
                "validation_losses": [3.5, 3.25],
                "final_validation_loss": 9.9,
                "actual_runtime_seconds": 420,
            }
        )

    result = service.get_result("student-1", submitted.experiment_id)
    assert result.status == "submitted"
    assert result.validation_losses == []
    assert result.final_validation_loss is None
    assert service.get_budget("student-1").reserved_seconds == 600
    assert service.get_budget("student-1").charged_seconds == 0


def test_worker_timeout_failure_charges_reserved_runtime_and_exposes_no_final_loss():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_failed",
            "failure_type": "timeout",
            "actual_runtime_seconds": 420,
            "message": "wall clock limit reached",
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    assert result.status == "failed"
    assert result.final_validation_loss is None
    assert result.failure_reason == "timeout"
    assert result.used_runtime_seconds == 600
    assert result.completed_at == ""
    assert result.failed_at == "2026-07-16T15:00:00Z"
    assert service.get_budget("student-1").charged_seconds == 600


def test_worker_failure_detail_is_staff_only_and_round_trips_snapshot(tmp_path):
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_failed",
            "failure_type": "provider_rejected",
            "actual_runtime_seconds": 1,
            "message": "qz node bootstrap failed: token abc123",
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    detail = service.admin_get_experiment(submitted.experiment_id)
    exported = service.admin_export_course_records()["experiments"][0]

    assert result.failure_reason == "provider_rejected"
    assert not hasattr(result, "staff_failure_detail")
    assert detail["staff_failure_detail"] == "qz node bootstrap failed: token abc123"
    assert exported["staff_failure_detail"] == "qz node bootstrap failed: token abc123"

    snapshot_path = tmp_path / "state.json"
    service.save_state_snapshot(snapshot_path)
    restored = _service()
    restored.load_state_snapshot(snapshot_path)

    restored_result = restored.get_result("student-1", submitted.experiment_id)
    restored_detail = restored.admin_get_experiment(submitted.experiment_id)
    assert restored_result.failure_reason == "provider_rejected"
    assert not hasattr(restored_result, "staff_failure_detail")
    assert (
        restored_detail["staff_failure_detail"]
        == "qz node bootstrap failed: token abc123"
    )


def test_worker_numerical_failure_charges_reported_runtime_clipped_to_reservation():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_failed",
            "failure_type": "numerical",
            "actual_runtime_seconds": 420,
            "validation_losses": [3.5],
            "message": "nan loss",
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    assert result.status == "failed"
    assert result.validation_losses == [3.5]
    assert result.final_validation_loss is None
    assert result.failure_reason == "numerical"
    assert result.used_runtime_seconds == 420
    assert result.failed_at == "2026-07-16T15:00:00Z"
    assert service.get_budget("student-1").charged_seconds == 420


def test_late_worker_failure_after_completion_does_not_mutate_terminal_result():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.4, 3.2],
            "final_validation_loss": 3.2,
            "actual_runtime_seconds": 240,
        }
    )

    with pytest.raises(ValueError, match="already terminal"):
        service.record_worker_event(
            {
                "experiment_id": submitted.experiment_id,
                "event_type": "worker_failed",
                "failure_type": "numerical",
                "validation_losses": [9.9],
                "final_validation_loss": 9.9,
                "actual_runtime_seconds": 600,
                "message": "late failure callback after completion",
            }
        )

    result = service.get_result("student-1", submitted.experiment_id)
    exported = service.admin_export_course_records()["experiments"][0]
    assert result.status == "completed"
    assert result.validation_losses == [3.4, 3.2]
    assert result.final_validation_loss == 3.2
    assert result.failure_reason == ""
    assert result.used_runtime_seconds == 240
    assert result.completed_at == "2026-07-16T15:00:00Z"
    assert result.failed_at == ""
    assert exported["status"] == "completed"
    assert exported["final_validation_loss"] == 3.2
    assert service.get_budget("student-1").charged_seconds == 240


def test_late_validation_after_completion_does_not_change_validation_tail():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.4, 3.2],
            "final_validation_loss": 3.2,
            "actual_runtime_seconds": 240,
        }
    )

    with pytest.raises(ValueError, match="already terminal"):
        service.record_worker_event(
            {
                "experiment_id": submitted.experiment_id,
                "event_type": "validation",
                "loss": 1.0,
            }
        )

    result = service.get_result("student-1", submitted.experiment_id)
    assert result.status == "completed"
    assert result.validation_losses == [3.4, 3.2]
    assert result.final_validation_loss == 3.2
    assert [
        event["event_type"]
        for event in service.admin_get_experiment(submitted.experiment_id)["events"]
    ] == ["worker_completed"]


def test_late_heartbeat_or_started_after_completion_does_not_reopen_result():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.4, 3.2],
            "final_validation_loss": 3.2,
            "actual_runtime_seconds": 240,
        }
    )

    for event_type in ["heartbeat", "worker_started"]:
        with pytest.raises(ValueError, match="already terminal"):
            service.record_worker_event(
                {
                    "experiment_id": submitted.experiment_id,
                    "event_type": event_type,
                    "step": 99,
                }
            )

    result = service.get_result("student-1", submitted.experiment_id)
    assert result.status == "completed"
    assert result.validation_losses == [3.4, 3.2]
    assert result.final_validation_loss == 3.2
    assert [
        event["event_type"]
        for event in service.admin_get_experiment(submitted.experiment_id)["events"]
    ] == ["worker_completed"]


def test_worker_raw_exception_failure_type_maps_to_typed_public_reason():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_failed",
            "failure_type": "FloatingPointError",
            "actual_runtime_seconds": 123,
            "validation_losses": [3.5],
            "message": "nan loss",
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    detail = service.admin_get_experiment(submitted.experiment_id)
    assert result.failure_reason == "numerical"
    assert detail["staff_failure_detail"] == "FloatingPointError: nan loss"
    assert service.get_budget("student-1").charged_seconds == 123


def test_unknown_worker_failure_type_maps_to_unknown_student_caused_public_reason():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_failed",
            "failure_type": "CustomTrainerCrash",
            "actual_runtime_seconds": 10,
            "message": "custom trainer raised ValueError with private file path",
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    detail = service.admin_get_experiment(submitted.experiment_id)
    assert result.failure_reason == "unknown_student_caused"
    assert detail["staff_failure_detail"] == (
        "CustomTrainerCrash: custom trainer raised ValueError with private file path"
    )


def test_tokenized_index_worker_failure_maps_to_infrastructure_error():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_failed",
            "failure_type": "TokenizedIndexError",
            "message": "tokenized shard hash mismatch: /secure/course/train/tokens-000000.bin",
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    detail = service.admin_get_experiment(submitted.experiment_id)
    budget = service.get_budget("student-1")
    assert result.status == "system_failed"
    assert result.failure_reason == "infrastructure_error"
    assert result.final_validation_loss is None
    assert result.used_runtime_seconds == 0
    assert budget.reserved_seconds == 0
    assert budget.charged_seconds == 0
    assert detail["staff_failure_detail"] == (
        "TokenizedIndexError: tokenized shard hash mismatch: "
        "/secure/course/train/tokens-000000.bin"
    )


def test_tokenized_dataset_worker_failure_maps_to_infrastructure_error():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_failed",
            "failure_type": "TokenizedDatasetError",
            "message": "tokenized shard is not readable: /mnt/course-data/train/tokens-0.bin",
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    detail = service.admin_get_experiment(submitted.experiment_id)
    budget = service.get_budget("student-1")
    assert result.status == "system_failed"
    assert result.failure_reason == "infrastructure_error"
    assert result.used_runtime_seconds == 0
    assert budget.reserved_seconds == 0
    assert budget.charged_seconds == 0
    assert detail["staff_failure_detail"] == (
        "TokenizedDatasetError: tokenized shard is not readable: "
        "/mnt/course-data/train/tokens-0.bin"
    )


def test_course_trainer_config_error_maps_to_invalid_runtime_config():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_failed",
            "failure_type": "CourseTrainerConfigError",
            "actual_runtime_seconds": 1,
            "message": "Stanford CS336 reference trainer does not support lr_schedule='linear'",
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    detail = service.admin_get_experiment(submitted.experiment_id)
    assert result.status == "failed"
    assert result.failure_reason == "invalid_runtime_config"
    assert detail["staff_failure_detail"] == (
        "CourseTrainerConfigError: Stanford CS336 reference trainer does not "
        "support lr_schedule='linear'"
    )


def test_jax_sharding_value_error_maps_to_invalid_runtime_config_for_old_workers():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_failed",
            "failure_type": "ValueError",
            "actual_runtime_seconds": 1,
            "message": (
                "Sharding spec ('fsdp',) implies that array axis 0 is partitioned "
                "2 times, but does not evenly divide the dimension size 50431. "
                "Got shape: (50431, 256) and sharding NamedSharding("
                "mesh=AbstractMesh('fsdp': 2), spec=P('fsdp', None))"
            ),
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    detail = service.admin_get_experiment(submitted.experiment_id)
    assert result.status == "failed"
    assert result.failure_reason == "invalid_runtime_config"
    assert "ValueError: Sharding spec ('fsdp',)" in detail["staff_failure_detail"]


def test_worker_reported_system_failure_releases_budget_without_charge():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_failed",
            "failure_type": "infrastructure_error",
            "actual_runtime_seconds": 17,
            "message": "validation index unavailable",
        }
    )

    result = service.get_result("student-1", submitted.experiment_id)
    budget = service.get_budget("student-1")
    assert result.status == "system_failed"
    assert result.failure_reason == "infrastructure_error"
    assert result.final_validation_loss is None
    assert result.used_runtime_seconds == 17
    assert budget.reserved_seconds == 0
    assert budget.charged_seconds == 0


def test_cross_student_result_lookup_is_denied():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    with pytest.raises(StudentAccessError):
        service.get_result("student-2", submitted.experiment_id)


def test_final_submission_is_student_scoped_mutable_and_interval_checked():
    service = _service()
    first = service.set_final_submission(
        student_id="student-1",
        training_config=_config(),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    replacement_config = {
        "model": {
            "num_hidden_layers": 4,
            "hidden_size": 256,
            "num_attention_heads": 2,
        },
        "training": {"train_tokens": 4096, "learning_rate": 2e-4, "num_evals": 1},
    }
    second = service.set_final_submission(
        student_id="student-1",
        training_config=replacement_config,
        predicted_final_loss=2.6,
        predicted_final_loss_lower=2.5,
        predicted_final_loss_upper=2.8,
    )

    assert first.updated_at == "2026-07-16T15:00:00Z"
    assert service.get_final_submission("student-1") == second
    assert second.training_config == replacement_config
    with pytest.raises(KeyError):
        service.get_final_submission("student-2")
    with pytest.raises(ValueError, match="lower <= point <= upper"):
        service.set_final_submission(
            student_id="student-1",
            training_config=_config(),
            predicted_final_loss=2.6,
            predicted_final_loss_lower=2.7,
            predicted_final_loss_upper=2.8,
        )


def test_final_submission_rejects_invalid_training_config_before_freeze():
    service = _service()
    invalid_config = _config()
    invalid_config["model"]["hidden_size"] = 130
    invalid_config["model"]["num_attention_heads"] = 8

    with pytest.raises(ValueError, match="model.hidden_size must be divisible"):
        service.set_final_submission(
            student_id="student-1",
            training_config=invalid_config,
            predicted_final_loss=2.8,
            predicted_final_loss_lower=2.7,
            predicted_final_loss_upper=2.9,
        )

    with pytest.raises(KeyError):
        service.get_final_submission("student-1")

    frozen = service.freeze_final_submissions(actor="staff", reason="deadline")
    launched = service.launch_final_runs(max_runtime_seconds=48 * 3600)

    assert frozen == []
    assert launched == []
    assert service.provider.submitted == []


def test_freeze_final_submissions_blocks_mutation_and_launches_final_runs():
    service = _service(
        final_data_manifest_id="final-train-private-v1",
        final_eval_manifest_id="final-eval-hidden-v1",
    )
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.set_final_submission(
        student_id="student-2",
        training_config=_config(train_tokens=4096),
        predicted_final_loss=2.6,
        predicted_final_loss_lower=2.5,
        predicted_final_loss_upper=2.8,
    )

    frozen = service.freeze_final_submissions(actor="staff", reason="deadline")
    launched = service.launch_final_runs(max_runtime_seconds=48 * 3600)

    assert [item.final_submission_id for item in frozen] == [
        "final-submission-000001",
        "final-submission-000002",
    ]
    assert frozen[0].frozen_by == "staff"
    assert frozen[0].freeze_reason == "deadline"
    assert [run.final_run_id for run in launched] == [
        "final-run-000001",
        "final-run-000002",
    ]
    assert service.get_final_submission("student-1").frozen_at == "2026-07-16T15:00:00Z"
    assert service.get_final_submission("student-1").frozen_by == "staff"
    assert service.get_final_submission("student-1").freeze_reason == "deadline"
    assert launched[0].student_id == "student-1"
    assert launched[0].status == "submitted"
    assert launched[0].predicted_final_loss == 2.8
    assert launched[0].provider_job_id == "provider-job-1"
    assert service.provider.submitted[-2]["run_kind"] == "final"
    assert service.provider.submitted[-2]["manifest_version"] == 1
    assert service.provider.submitted[-2]["final_run_id"] == "final-run-000001"
    assert service.provider.submitted[-2]["final_submission_id"] == "final-submission-000001"
    assert service.provider.submitted[-2]["student_id"] == "student-1"
    assert service.provider.submitted[-2]["code_version"] == "local-dev"
    assert service.provider.submitted[-2]["callback_base_url"] == "https://backend/internal/provider-events"
    assert service.provider.submitted[-2]["max_runtime_seconds"] == 48 * 3600
    assert service.provider.submitted[-2]["created_at"] == "2026-07-16T15:00:00Z"
    assert service.provider.submitted[-2]["runtime_config"]["reserved_runtime_seconds"] == 48 * 3600
    assert service.provider.submitted[-2]["runtime_config"]["max_runtime_seconds"] == 48 * 3600
    assert service.provider.submitted[-2]["resolved_config"]["data_manifest_id"] == "final-train-private-v1"
    assert service.provider.submitted[-2]["data_manifest_id"] == "final-train-private-v1"
    assert service.provider.submitted[-2]["resolved_config"]["eval_manifest_id"] == "final-eval-hidden-v1"
    assert service.provider.submitted[-2]["eval_manifest_id"] == "final-eval-hidden-v1"
    assert service.admin_export_course_records()["admin_actions"] == [
        {
            "action_id": "admin-action-000001",
            "action_type": "freeze_final_submissions",
            "actor": "staff",
            "created_at": "2026-07-16T15:00:00Z",
            "reason": "deadline",
            "student_id": "",
            "experiment_id": "",
            "final_run_id": "",
            "before_status": "open",
            "after_status": "frozen",
        },
        {
            "action_id": "admin-action-000002",
            "action_type": "launch_final_runs",
            "actor": "staff",
            "created_at": "2026-07-16T15:00:00Z",
            "reason": "deadline",
            "student_id": "",
            "experiment_id": "",
            "final_run_id": "",
            "before_status": "frozen",
            "after_status": "launched",
        },
    ]

    with pytest.raises(ValueError, match="frozen"):
        service.set_final_submission(
            student_id="student-1",
            training_config=_config(train_tokens=8192),
            predicted_final_loss=2.5,
            predicted_final_loss_lower=2.4,
            predicted_final_loss_upper=2.6,
        )


def test_empty_final_submission_freeze_closes_window_and_launch_is_idempotent():
    service = _service()

    frozen = service.freeze_final_submissions(actor="staff", reason="deadline")
    repeated_freeze = service.freeze_final_submissions(
        actor="second-staff", reason="rerun"
    )
    launched = service.launch_final_runs(max_runtime_seconds=48 * 3600)
    repeated_launch = service.launch_final_runs(max_runtime_seconds=48 * 3600)

    assert frozen == []
    assert repeated_freeze == []
    assert launched == []
    assert repeated_launch == []
    assert service.provider.submitted == []
    with pytest.raises(ValueError, match="frozen"):
        service.set_final_submission(
            student_id="student-1",
            training_config=_config(),
            predicted_final_loss=2.8,
            predicted_final_loss_lower=2.7,
            predicted_final_loss_upper=2.9,
        )
    assert [
        (action["action_type"], action["actor"], action["reason"])
        for action in service.admin_export_course_records()["admin_actions"]
    ] == [
        ("freeze_final_submissions", "staff", "deadline"),
        ("launch_final_runs", "staff", "deadline"),
    ]


def test_launch_final_runs_threads_tokenized_index_uris_into_final_manifest():
    service = _service(
        final_train_tokenized_index_uri=(
            "file:///secure/course/tokenized/final-train/index.json"
        ),
        final_validation_tokenized_index_uri=(
            "https://storage.example/tokenized/final-eval/index.json"
        ),
    )
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")

    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    provider_manifest = service.provider.submitted[0]
    assert provider_manifest["data_config"] == {
        "train_tokens": 2048,
        "tokenized_index_uri": "file:///secure/course/tokenized/final-train/index.json",
    }
    assert provider_manifest["validation_config"]["tokenized_index_uri"] == (
        "https://storage.example/tokenized/final-eval/index.json"
    )
    assert "tokenized_index_uri" not in provider_manifest["resolved_config"]


def test_launch_final_runs_writes_frozen_manifest_before_provider_dispatch(tmp_path):
    store = LocalManifestStore(tmp_path)
    service = ExperimentService(
        provider=FakeProvider(),
        total_budget_seconds=12 * 3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda run_id: f"memory://placeholder/{run_id}",
        callback_url="https://backend/internal/provider-events",
        manifest_store=store,
    )
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")

    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    provider_manifest = service.provider.submitted[0]
    manifest_uri = (tmp_path / "final-run-000001.json").resolve().as_uri()
    assert provider_manifest["manifest_uri"] == manifest_uri
    assert provider_manifest["resolved_config"]["provider_manifest_uri"] == manifest_uri
    exported = service.admin_export_course_records()
    assert exported["final_runs"][0]["resolved_config"]["provider_manifest_uri"] == manifest_uri
    assert (tmp_path / "final-run-000001.json").exists()
    frozen_manifest = json.loads((tmp_path / "final-run-000001.json").read_text(encoding="utf-8"))
    assert frozen_manifest["manifest_version"] == 1
    assert frozen_manifest["run_kind"] == "final"
    assert frozen_manifest["final_run_id"] == "final-run-000001"
    assert frozen_manifest["max_runtime_seconds"] == 48 * 3600
    assert frozen_manifest["model_config"] == frozen_manifest["resolved_config"]["model"]
    assert frozen_manifest["training_config"] == frozen_manifest["resolved_config"]["training"]
    assert frozen_manifest["resolved_config"]["provider_manifest_uri"] == manifest_uri
    assert frozen_manifest["data_config"] == {"train_tokens": 2048}


def test_launch_final_runs_requires_frozen_submissions_and_is_idempotent():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )

    with pytest.raises(ValueError, match="freeze"):
        service.launch_final_runs(max_runtime_seconds=48 * 3600)

    service.freeze_final_submissions(actor="staff", reason="deadline")
    first = service.launch_final_runs(max_runtime_seconds=48 * 3600)
    second = service.launch_final_runs(max_runtime_seconds=48 * 3600)

    assert first == second
    assert len(service.provider.submitted) == 1
    assert [
        action["action_type"]
        for action in service.admin_export_course_records()["admin_actions"]
    ] == ["freeze_final_submissions", "launch_final_runs"]


def test_launch_final_runs_retry_submits_only_missing_final_runs_after_partial_failure():
    provider = FailOnceOnFinalSubmissionProvider(
        failing_final_submission_id="final-submission-000002"
    )
    service = ExperimentService(
        provider=provider,
        total_budget_seconds=12 * 3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda run_id: f"memory://manifests/{run_id}.json",
        callback_url="https://backend/internal/provider-events",
    )
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.set_final_submission(
        student_id="student-2",
        training_config=_config(train_tokens=4096),
        predicted_final_loss=2.6,
        predicted_final_loss_lower=2.5,
        predicted_final_loss_upper=2.8,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")

    with pytest.raises(RuntimeError, match="simulated provider launch failure"):
        service.launch_final_runs(max_runtime_seconds=48 * 3600)
    retried = service.launch_final_runs(max_runtime_seconds=48 * 3600)

    assert [run.final_submission_id for run in retried] == [
        "final-submission-000001",
        "final-submission-000002",
    ]
    assert [manifest["final_submission_id"] for manifest in provider.submitted] == [
        "final-submission-000001",
        "final-submission-000002",
    ]
    assert [run.final_run_id for run in retried] == [
        "final-run-000001",
        "final-run-000002",
    ]
    assert len(service.launch_final_runs(max_runtime_seconds=48 * 3600)) == 2
    assert [manifest["final_submission_id"] for manifest in provider.submitted] == [
        "final-submission-000001",
        "final-submission-000002",
    ]
    assert [
        action["action_type"]
        for action in service.admin_export_course_records()["admin_actions"]
    ] == ["freeze_final_submissions", "launch_final_runs"]


def test_state_snapshot_preserves_partial_final_launch_retry_state(tmp_path):
    provider = FailOnceOnFinalSubmissionProvider(
        failing_final_submission_id="final-submission-000002"
    )
    service = ExperimentService(
        provider=provider,
        total_budget_seconds=12 * 3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda run_id: f"memory://manifests/{run_id}.json",
        callback_url="https://backend/internal/provider-events",
    )
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.set_final_submission(
        student_id="student-2",
        training_config=_config(train_tokens=4096),
        predicted_final_loss=2.6,
        predicted_final_loss_lower=2.5,
        predicted_final_loss_upper=2.8,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")

    with pytest.raises(RuntimeError, match="simulated provider launch failure"):
        service.launch_final_runs(max_runtime_seconds=48 * 3600)
    snapshot_path = tmp_path / "state.json"
    service.save_state_snapshot(snapshot_path)

    restored = _service()
    restored.load_state_snapshot(snapshot_path)
    launched = restored.launch_final_runs(max_runtime_seconds=48 * 3600)

    assert [run.final_submission_id for run in launched] == [
        "final-submission-000001",
        "final-submission-000002",
    ]
    assert [manifest["final_submission_id"] for manifest in restored.provider.submitted] == [
        "final-submission-000002",
    ]
    assert [
        action["action_type"]
        for action in restored.admin_export_course_records()["admin_actions"]
    ] == ["freeze_final_submissions", "launch_final_runs"]


def test_final_run_worker_callback_records_actual_loss_for_grading_export():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_completed",
            "validation_losses": [2.75],
            "final_validation_loss": 2.75,
            "actual_runtime_seconds": 3600,
        }
    )
    exported = service.admin_export_course_records()

    assert exported["final_runs"][0]["status"] == "completed"
    assert exported["final_runs"][0]["actual_final_validation_loss"] == 2.75
    assert exported["final_runs"][0]["validation_losses"] == [2.75]
    assert exported["final_runs"][0]["prediction_absolute_error"] == pytest.approx(0.05)
    assert exported["final_runs"][0]["prediction_interval_covered"] is True
    assert exported["final_runs"][0]["prediction_interval_width"] == pytest.approx(0.2)
    assert exported["final_runs"][0]["prediction_interval_miss_distance"] == 0.0
    assert exported["final_runs"][0]["prediction_interval_score"] == pytest.approx(0.2)
    assert exported["final_runs"][0]["prediction_quality_penalty"] == pytest.approx(0.125)


def test_final_run_completion_derives_actual_loss_from_validation_tail():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_completed",
            "validation_losses": [2.9, 2.75],
            "actual_runtime_seconds": 3600,
        }
    )
    exported = service.admin_export_course_records()

    assert exported["final_runs"][0]["status"] == "completed"
    assert exported["final_runs"][0]["validation_losses"] == [2.9, 2.75]
    assert exported["final_runs"][0]["actual_final_validation_loss"] == 2.75
    assert exported["final_runs"][0]["prediction_absolute_error"] == pytest.approx(0.05)
    assert exported["final_runs"][0]["prediction_interval_covered"] is True
    assert exported["final_runs"][0]["prediction_quality_penalty"] == pytest.approx(0.125)


def test_final_run_completion_can_use_accumulated_validation_tail():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "validation",
            "loss": 2.9,
        }
    )
    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "validation",
            "loss": 2.75,
        }
    )
    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_completed",
            "actual_runtime_seconds": 3600,
        }
    )
    exported = service.admin_export_course_records()

    assert exported["final_runs"][0]["status"] == "completed"
    assert exported["final_runs"][0]["validation_losses"] == [2.9, 2.75]
    assert exported["final_runs"][0]["actual_final_validation_loss"] == 2.75


def test_final_run_completion_rejects_actual_loss_that_disagrees_with_validation_tail():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    with pytest.raises(ValueError, match="final_validation_loss must equal"):
        service.record_worker_event(
            {
                "final_run_id": "final-run-000001",
                "event_type": "worker_completed",
                "validation_losses": [2.9, 2.75],
                "final_validation_loss": 9.9,
                "actual_runtime_seconds": 3600,
            }
        )

    exported = service.admin_export_course_records()
    assert exported["final_runs"][0]["status"] == "submitted"
    assert exported["final_runs"][0]["validation_losses"] == []
    assert exported["final_runs"][0]["actual_final_validation_loss"] is None


def test_late_final_run_failure_after_completion_does_not_mutate_grading_result():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)
    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_completed",
            "validation_losses": [2.75],
            "final_validation_loss": 2.75,
            "actual_runtime_seconds": 3600,
        }
    )

    with pytest.raises(ValueError, match="already terminal"):
        service.record_worker_event(
            {
                "final_run_id": "final-run-000001",
                "event_type": "worker_failed",
                "failure_type": "provider_rejected",
                "validation_losses": [9.9],
                "final_validation_loss": 9.9,
                "message": "late failure callback after completion",
            }
        )

    exported = service.admin_export_course_records()
    final_run = exported["final_runs"][0]
    assert final_run["status"] == "completed"
    assert final_run["validation_losses"] == [2.75]
    assert final_run["actual_final_validation_loss"] == 2.75
    assert final_run["failure_reason"] == ""
    assert final_run["staff_failure_detail"] == ""
    assert final_run["prediction_absolute_error"] == pytest.approx(0.05)


def test_late_final_run_validation_after_completion_does_not_change_grading_tail():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)
    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_completed",
            "validation_losses": [2.75],
            "final_validation_loss": 2.75,
            "actual_runtime_seconds": 3600,
        }
    )

    with pytest.raises(ValueError, match="already terminal"):
        service.record_worker_event(
            {
                "final_run_id": "final-run-000001",
                "event_type": "validation",
                "loss": 1.0,
            }
        )

    final_run = service.admin_export_course_records()["final_runs"][0]
    assert final_run["status"] == "completed"
    assert final_run["validation_losses"] == [2.75]
    assert final_run["actual_final_validation_loss"] == 2.75
    assert final_run["prediction_absolute_error"] == pytest.approx(0.05)


def test_late_final_run_heartbeat_or_started_after_completion_is_rejected():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)
    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_completed",
            "validation_losses": [2.75],
            "final_validation_loss": 2.75,
            "actual_runtime_seconds": 3600,
        }
    )

    for event_type in ["heartbeat", "worker_started"]:
        with pytest.raises(ValueError, match="already terminal"):
            service.record_worker_event(
                {
                    "final_run_id": "final-run-000001",
                    "event_type": event_type,
                    "step": 99,
                }
            )

    final_run = service.admin_export_course_records()["final_runs"][0]
    assert final_run["status"] == "completed"
    assert final_run["validation_losses"] == [2.75]
    assert final_run["actual_final_validation_loss"] == 2.75


def test_final_run_worker_started_and_heartbeat_callbacks_are_accepted():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_started",
        }
    )
    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "heartbeat",
            "step": 1,
        }
    )

    exported = service.admin_export_course_records()
    assert exported["final_runs"][0]["status"] == "running"
    assert exported["final_runs"][0]["actual_final_validation_loss"] is None
    assert exported["final_runs"][0]["validation_losses"] == []


def test_final_run_worker_system_failure_is_staff_review_not_student_failed():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_failed",
            "failure_type": "TokenizedIndexError",
            "message": "hidden eval shard hash mismatch",
        }
    )

    exported = service.admin_export_course_records()
    final_run = exported["final_runs"][0]
    assert final_run["status"] == "system_failed"
    assert final_run["failure_reason"] == "infrastructure_error"
    assert final_run["actual_final_validation_loss"] is None
    assert final_run["staff_failure_detail"] == (
        "TokenizedIndexError: hidden eval shard hash mismatch"
    )
    assert final_run["prediction_absolute_error"] is None
    assert final_run["prediction_interval_covered"] is None
    assert final_run["prediction_interval_score"] is None
    assert final_run["prediction_quality_penalty"] is None


def test_final_run_callback_rejects_unknown_event_type_without_mutating_state():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    with pytest.raises(ValueError, match="unknown worker event_type"):
        service.record_worker_event(
            {
                "final_run_id": "final-run-000001",
                "event_type": "worker_compeleted",
                "final_validation_loss": 2.75,
            }
        )

    exported = service.admin_export_course_records()
    assert exported["final_runs"][0]["status"] == "submitted"
    assert exported["final_runs"][0]["actual_final_validation_loss"] is None
    assert exported["final_runs"][0]["validation_losses"] == []


def test_admin_poll_final_run_updates_provider_status_without_fabricating_loss():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.SUCCEEDED,
        raw_status="COMPLETED",
        message="container exited zero",
    )

    result = service.admin_poll_final_run("final-run-000001")
    exported = service.admin_export_course_records()

    assert result.status == "lost"
    assert result.actual_final_validation_loss is None
    assert result.failure_reason == "missing_worker_completed_callback"
    assert exported["final_runs"][0]["status"] == "lost"
    assert exported["final_runs"][0]["actual_final_validation_loss"] is None


def test_admin_poll_final_run_provider_failure_keeps_staff_detail():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.FAILED,
        raw_status="QZ_JOB_FAILED",
        message="container exited 137 on b200-node-3",
    )

    result = service.admin_poll_final_run("final-run-000001")
    exported = service.admin_export_course_records()

    assert result.status == "failed"
    assert result.failure_reason == "provider_failed_without_worker_callback"
    assert result.staff_failure_detail == (
        "provider_name=fake; provider_job_id=provider-job-1; "
        "provider_status=failed; raw_status=QZ_JOB_FAILED; "
        "message=container exited 137 on b200-node-3"
    )
    assert exported["final_runs"][0]["staff_failure_detail"] == result.staff_failure_detail


def test_admin_can_mark_final_run_system_failure_auditably():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)
    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "validation",
            "loss": 2.9,
        }
    )

    result = service.admin_mark_final_run_system_failed(
        "final-run-000001",
        failure_reason="infrastructure_error",
        staff_failure_detail="hidden eval shard unavailable incident INC-55",
        reason="confirmed hidden eval outage",
        actor="staff",
    )
    exported = service.admin_export_course_records()

    assert result.status == "system_failed"
    assert result.failure_reason == "infrastructure_error"
    assert result.actual_final_validation_loss is None
    assert result.validation_losses == [2.9]
    assert result.staff_failure_detail == "hidden eval shard unavailable incident INC-55"
    assert exported["final_runs"][0]["staff_failure_detail"] == result.staff_failure_detail
    assert exported["admin_actions"][-1] == {
        "action_id": "admin-action-000003",
        "action_type": "mark_final_run_system_failed",
        "actor": "staff",
        "created_at": "2026-07-16T15:00:00Z",
        "reason": "confirmed hidden eval outage",
        "student_id": "student-1",
        "experiment_id": "",
        "final_run_id": "final-run-000001",
        "before_status": "submitted",
        "after_status": "system_failed",
    }


def test_admin_cancel_final_run_stops_provider_job_and_preserves_hidden_loss_fields():
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    result = service.admin_cancel_final_run(
        "final-run-000001",
        reason="staff dry-run cleanup",
        actor="staff",
    )
    exported = service.admin_export_course_records()

    assert service.provider.cancelled == [
        ("provider-job-1", "staff dry-run cleanup", "staff")
    ]
    assert result.status == "cancelled"
    assert result.failure_reason == "admin_intervention"
    assert result.actual_final_validation_loss is None
    assert exported["final_runs"][0]["status"] == "cancelled"
    assert exported["final_runs"][0]["failure_reason"] == "admin_intervention"
    assert exported["admin_actions"] == [
        {
            "action_id": "admin-action-000001",
            "action_type": "freeze_final_submissions",
            "actor": "staff",
            "created_at": "2026-07-16T15:00:00Z",
            "reason": "deadline",
            "student_id": "",
            "experiment_id": "",
            "final_run_id": "",
            "before_status": "open",
            "after_status": "frozen",
        },
        {
            "action_id": "admin-action-000002",
            "action_type": "launch_final_runs",
            "actor": "staff",
            "created_at": "2026-07-16T15:00:00Z",
            "reason": "deadline",
            "student_id": "",
            "experiment_id": "",
            "final_run_id": "",
            "before_status": "frozen",
            "after_status": "launched",
        },
        {
            "action_id": "admin-action-000003",
            "action_type": "cancel_final_run",
            "actor": "staff",
            "created_at": "2026-07-16T15:00:00Z",
            "reason": "staff dry-run cleanup",
            "student_id": "student-1",
            "experiment_id": "",
            "final_run_id": "final-run-000001",
            "before_status": "submitted",
            "after_status": "cancelled",
        }
    ]


def test_admin_list_and_cancel_experiment_across_students():
    service = _service()
    first = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    second = service.submit(
        student_id="student-2",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )

    listed = service.admin_list_experiments()
    cancelled = service.admin_cancel_experiment(
        first.experiment_id,
        reason="quota audit",
        actor="staff",
    )
    exported = service.admin_export_course_records()

    assert [item["experiment_id"] for item in listed] == [
        first.experiment_id,
        second.experiment_id,
    ]
    assert listed[0]["student_id"] == "student-1"
    assert listed[1]["student_id"] == "student-2"
    assert cancelled.status == "cancelled"
    assert service.provider.cancelled == [("provider-job-1", "quota audit", "staff")]
    assert service.get_result("student-1", first.experiment_id).status == "cancelled"
    assert exported["admin_actions"] == [
        {
            "action_id": "admin-action-000001",
            "action_type": "cancel_experiment",
            "actor": "staff",
            "created_at": "2026-07-16T15:00:00Z",
            "reason": "quota audit",
            "student_id": "student-1",
            "experiment_id": first.experiment_id,
            "final_run_id": "",
            "before_status": "submitted",
            "after_status": "cancelled",
        }
    ]


def test_admin_cancel_completed_experiment_preserves_terminal_result():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.4, 3.1],
            "final_validation_loss": 3.1,
            "actual_runtime_seconds": 420,
        }
    )

    result = service.admin_cancel_experiment(
        submitted.experiment_id,
        reason="late cleanup request",
        actor="staff",
    )
    exported = service.admin_export_course_records()

    assert result.status == "completed"
    assert result.final_validation_loss == 3.1
    assert result.validation_losses == [3.4, 3.1]
    assert result.used_runtime_seconds == 420
    assert service.provider.cancelled == []
    assert exported["admin_actions"] == []
    assert exported["experiments"][0]["status"] == "completed"
    assert exported["experiments"][0]["final_validation_loss"] == 3.1


def test_admin_poll_experiment_updates_provider_status_into_experiment_state():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )

    result = service.admin_poll_experiment(submitted.experiment_id)

    assert result.status == "running"
    assert service.get_result("student-1", submitted.experiment_id).status == "running"
    assert service.get_budget("student-1").reserved_seconds == 600
    assert service.dispatcher.get(submitted.experiment_id).events[-1].metadata == {
        "provider_status": "running",
        "raw_status": "RUNNING",
        "message": "container started",
    }


def test_admin_poll_provider_success_without_worker_completion_marks_lost():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.SUCCEEDED,
        raw_status="COMPLETED",
        message="container exited zero",
    )

    result = service.admin_poll_experiment(submitted.experiment_id)

    assert result.status == "lost"
    assert result.final_validation_loss is None
    assert result.failure_reason == "missing_worker_completed_callback"
    assert result.used_runtime_seconds == 600
    assert result.completed_at == ""
    assert result.failed_at == "2026-07-16T15:00:00Z"
    assert service.get_budget("student-1").reserved_seconds == 0
    assert service.get_budget("student-1").charged_seconds == 600


def test_admin_poll_provider_failure_keeps_raw_detail_staff_only():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.FAILED,
        raw_status="QZ_POD_CRASHLOOP",
        message="node gpu-b200-7 kubelet: image pull secret expired",
    )

    result = service.admin_poll_experiment(submitted.experiment_id)
    detail = service.admin_get_experiment(submitted.experiment_id)

    assert result.status == "failed"
    assert result.failure_reason == "provider_failed_without_worker_callback"
    assert result.used_runtime_seconds == 600
    assert result.failed_at == "2026-07-16T15:00:00Z"
    assert not hasattr(result, "staff_failure_detail")
    assert detail["provider_artifacts"] == {
        "provider_name": "fake",
        "provider_job_id": "provider-job-1",
        "logs_uri": "memory://fake-provider/provider-job-1/logs.txt",
        "detail_uri": "memory://fake-provider/provider-job-1/detail.json",
        "metadata": {},
    }
    assert detail["staff_failure_detail"] == (
        "provider_name=fake; provider_job_id=provider-job-1; "
        "provider_status=failed; raw_status=QZ_POD_CRASHLOOP; "
        "message=node gpu-b200-7 kubelet: image pull secret expired"
    )


def test_admin_poll_provider_unknown_releases_reservation_as_terminal_system_failure():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.UNKNOWN,
        raw_status="QZ_NEW_STATUS",
        message="unknown provider status",
    )

    result = service.admin_poll_experiment(submitted.experiment_id)

    assert result.status == "unknown"
    assert result.failure_reason == "provider_status_unknown"
    assert result.used_runtime_seconds == 600
    assert result.failed_at == "2026-07-16T15:00:00Z"
    assert service.get_budget("student-1").reserved_seconds == 0
    assert service.get_budget("student-1").charged_seconds == 600


def test_admin_list_experiments_can_filter_by_student_and_status():
    service = _service()
    running = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    completed = service.submit(
        student_id="student-1",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )
    other_student = service.submit(
        student_id="student-2",
        config=_config(train_tokens=4096),
        requested_runtime_seconds=300,
    )
    service.record_worker_event(
        {
            "experiment_id": running.experiment_id,
            "event_type": "worker_started",
        }
    )
    service.record_worker_event(
        {
            "experiment_id": completed.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.0],
            "final_validation_loss": 3.0,
            "actual_runtime_seconds": 120,
        }
    )

    student_one = service.admin_list_experiments(student_id="student-1")
    running_only = service.admin_list_experiments(status="running")
    submitted_only = service.admin_list_experiments(status="submitted")

    assert [row["experiment_id"] for row in student_one] == [
        running.experiment_id,
        completed.experiment_id,
    ]
    assert [row["experiment_id"] for row in running_only] == [running.experiment_id]
    assert [row["experiment_id"] for row in submitted_only] == [
        other_student.experiment_id
    ]


def test_admin_budget_adjustment_is_audited_and_changes_remaining_budget():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )

    adjustment = service.admin_apply_budget_adjustment(
        student_id="student-1",
        seconds=-120,
        reason="refund after provider outage",
        actor="staff",
        experiment_id=submitted.experiment_id,
    )
    positive_adjustment = service.admin_apply_budget_adjustment(
        student_id="student-1",
        seconds=60,
        reason="manual penalty",
        actor="staff",
    )

    budget = service.get_budget("student-1")
    assert adjustment.adjustment_id == "budget-adjustment-000001"
    assert adjustment.seconds == -120
    assert adjustment.reason == "refund after provider outage"
    assert adjustment.before_charged_seconds == 0
    assert adjustment.after_charged_seconds == -120
    assert positive_adjustment.adjustment_id == "budget-adjustment-000002"
    assert positive_adjustment.before_charged_seconds == -120
    assert positive_adjustment.after_charged_seconds == -60
    assert budget.charged_seconds == -60
    assert budget.remaining_seconds == 12 * 3600 - 600 + 60


def test_admin_can_mark_experiment_system_failure_and_refund_auditably():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(),
        requested_runtime_seconds=600,
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_started",
        }
    )

    result = service.admin_mark_experiment_system_failed(
        submitted.experiment_id,
        failure_reason="provider_outage",
        staff_failure_detail="QZ region outage incident INC-42",
        refund_seconds=120,
        reason="confirmed provider outage",
        actor="staff",
    )
    exported = service.admin_export_course_records()
    budget = service.get_budget("student-1")

    assert result.status == "system_failed"
    assert result.failure_reason == "provider_outage"
    assert result.final_validation_loss is None
    assert result.used_runtime_seconds == 0
    assert result.completed_at == ""
    assert result.failed_at == "2026-07-16T15:00:00Z"
    assert budget.reserved_seconds == 0
    assert budget.charged_seconds == -120
    assert exported["experiments"][0]["staff_failure_detail"] == (
        "QZ region outage incident INC-42"
    )
    assert exported["budget_adjustments"] == [
        {
            "adjustment_id": "budget-adjustment-000001",
            "student_id": "student-1",
            "seconds": -120,
            "reason": "confirmed provider outage",
            "actor": "staff",
            "created_at": "2026-07-16T15:00:00Z",
            "before_charged_seconds": 0,
            "after_charged_seconds": -120,
            "experiment_id": submitted.experiment_id,
        }
    ]
    assert exported["admin_actions"] == [
        {
            "action_id": "admin-action-000001",
            "action_type": "mark_experiment_system_failed",
            "actor": "staff",
            "created_at": "2026-07-16T15:00:00Z",
            "reason": "confirmed provider outage",
            "student_id": "student-1",
            "experiment_id": submitted.experiment_id,
            "final_run_id": "",
            "before_status": "running",
            "after_status": "system_failed",
        }
    ]


def test_admin_system_failure_succeeds_when_queued_launch_fails():
    provider = FailOnSecondExploratoryProvider()
    service = ExperimentService(
        provider=provider,
        total_budget_seconds=12 * 3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda experiment_id: (
            f"memory://manifests/{experiment_id}.json"
        ),
        callback_url="https://backend/internal/provider-events",
        max_active_experiments_per_student=1,
    )
    first = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    second = service.submit(
        student_id="student-1",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )

    result = service.admin_mark_experiment_system_failed(
        first.experiment_id,
        failure_reason="worker_lost",
        staff_failure_detail="worker disappeared without terminal callback",
        refund_seconds=0,
        reason="manual cleanup after worker crash",
        actor="staff",
    )
    exported = service.admin_export_course_records()

    assert result.status == "system_failed"
    assert service.get_result("student-1", first.experiment_id).status == (
        "system_failed"
    )
    assert service.get_result("student-1", second.experiment_id).status == "queued"
    assert service.get_budget("student-1").reserved_seconds == 300
    assert [manifest["experiment_id"] for manifest in provider.submitted] == [
        first.experiment_id
    ]
    assert [action["action_type"] for action in exported["admin_actions"]] == [
        "mark_experiment_system_failed",
        "launch_queued_experiments_failed",
    ]
    assert exported["admin_actions"][1]["reason"] == (
        "mark_experiment_system_failed: queued experiment launch failed with "
        "RuntimeError: simulated queued launch failure"
    )


def test_admin_export_course_records_includes_experiments_budget_and_final_runs():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.2],
            "final_validation_loss": 3.2,
            "actual_runtime_seconds": 420,
        }
    )
    service.admin_apply_budget_adjustment(
        student_id="student-1",
        seconds=-20,
        reason="rounding refund",
        actor="staff",
    )
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=2048),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    exported = service.admin_export_course_records()

    assert exported["experiments"][0]["experiment_id"] == submitted.experiment_id
    assert exported["experiments"][0]["final_validation_loss"] == 3.2
    assert exported["experiments"][0]["provider_name"] == "fake"
    assert exported["experiments"][0]["provider_job_id"] == "provider-job-1"
    assert exported["experiments"][0]["provider_status"] == "submitted"
    assert exported["experiments"][0]["provider_artifacts"] == {
        "provider_name": "fake",
        "provider_job_id": "provider-job-1",
        "logs_uri": "memory://fake-provider/provider-job-1/logs.txt",
        "detail_uri": "memory://fake-provider/provider-job-1/detail.json",
        "metadata": {},
    }
    assert [event["event_type"] for event in exported["experiment_events"]] == [
        "worker_completed"
    ]
    assert exported["budget_adjustments"][0]["seconds"] == -20
    assert exported["budget_adjustments"][0]["before_charged_seconds"] == 420
    assert exported["budget_adjustments"][0]["after_charged_seconds"] == 400
    assert exported["budget_snapshots"] == [
        {
            "student_id": "student-1",
            "total_seconds": 12 * 3600,
            "reserved_seconds": 0,
            "charged_seconds": 400,
            "remaining_seconds": 12 * 3600 - 400,
        }
    ]
    assert [action["action_type"] for action in exported["admin_actions"]] == [
        "freeze_final_submissions",
        "launch_final_runs",
    ]
    assert exported["final_submissions"][0]["student_id"] == "student-1"
    assert exported["final_runs"][0]["final_run_id"] == "final-run-000001"
    assert exported["final_runs"][0]["predicted_final_loss"] == 2.8
    assert exported["final_runs"][0]["provider_artifacts"] == {
        "provider_name": "fake",
        "provider_job_id": "provider-job-2",
        "logs_uri": "memory://fake-provider/provider-job-2/logs.txt",
        "detail_uri": "memory://fake-provider/provider-job-2/detail.json",
        "metadata": {},
    }


def test_admin_write_course_exports_writes_export_files(tmp_path):
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.2],
            "final_validation_loss": 3.2,
            "actual_runtime_seconds": 420,
        }
    )

    manifest = service.admin_write_course_exports(tmp_path)

    assert set(manifest) == {
        "experiments_jsonl",
        "experiments_csv",
        "experiment_events_jsonl",
        "worker_events_jsonl",
        "budget_snapshots_jsonl",
        "budget_snapshots_csv",
        "budget_adjustments_jsonl",
        "admin_actions_jsonl",
        "final_submissions_jsonl",
        "final_submissions_csv",
        "final_runs_jsonl",
        "final_runs_csv",
        "grading_csv",
    }
    assert (tmp_path / "experiments.csv").exists()
    assert "exp-000001" in (tmp_path / "experiments.jsonl").read_text()
    assert "student-1" in (tmp_path / "budget_snapshots.csv").read_text()


def test_admin_queue_snapshot_separates_exploratory_and_final_runs():
    service = _service()
    first = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    second = service.submit(
        student_id="student-2",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )
    service.record_worker_event(
        {
            "experiment_id": second.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.0],
            "final_validation_loss": 3.0,
            "actual_runtime_seconds": 100,
        }
    )
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=4096),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    snapshot = service.admin_queue_snapshot()

    assert snapshot["exploratory"] == [
        {
            "experiment_id": first.experiment_id,
            "student_id": "student-1",
            "status": "submitted",
            "reserved_runtime_seconds": 600,
            "queue_position": 1,
            "fair_share_rank": 1,
        }
    ]
    assert snapshot["final"] == [
        {
            "final_run_id": "final-run-000001",
            "student_id": "student-1",
            "status": "submitted",
            "provider_job_id": "provider-job-3",
            "queue_position": 1,
        }
    ]
    assert snapshot["active_counts_by_student"] == {
        "student-1": 2,
    }


def test_admin_queue_snapshot_exposes_fifo_order_and_fair_share_rank():
    service = _service()
    first = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    second = service.submit(
        student_id="student-1",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )
    third = service.submit(
        student_id="student-2",
        config=_config(train_tokens=3072),
        requested_runtime_seconds=300,
    )

    snapshot = service.admin_queue_snapshot()

    assert [
        row["experiment_id"] for row in snapshot["exploratory"]
    ] == [
        first.experiment_id,
        second.experiment_id,
        third.experiment_id,
    ]
    assert [
        (row["experiment_id"], row["queue_position"], row["fair_share_rank"])
        for row in snapshot["exploratory"]
    ] == [
        (first.experiment_id, 1, 1),
        (second.experiment_id, 2, 3),
        (third.experiment_id, 3, 2),
    ]
    assert snapshot["fair_share_order"] == [
        {
            "experiment_id": first.experiment_id,
            "student_id": "student-1",
            "queue_position": 1,
            "fair_share_rank": 1,
        },
        {
            "experiment_id": third.experiment_id,
            "student_id": "student-2",
            "queue_position": 3,
            "fair_share_rank": 2,
        },
        {
            "experiment_id": second.experiment_id,
            "student_id": "student-1",
            "queue_position": 2,
            "fair_share_rank": 3,
        },
    ]


def test_admin_poll_active_runs_updates_exploratory_and_final_queue_items():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=4096),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    report = service.admin_poll_active_runs()

    assert report == {
        "exploratory": [
            {
                "experiment_id": submitted.experiment_id,
                "status": "running",
                "failure_reason": "",
            }
        ],
        "final": [
            {
                "final_run_id": "final-run-000001",
                "status": "running",
                "failure_reason": "",
            }
        ],
    }
    assert service.get_result("student-1", submitted.experiment_id).status == "running"
    assert service.admin_export_course_records()["final_runs"][0]["status"] == "running"


def test_admin_poll_active_runs_reports_provider_artifacts_for_closed_failed_jobs():
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.FAILED,
        raw_status="FAILED",
        message="processExitCode=139",
    )

    report = service.admin_poll_active_runs()

    assert report["exploratory"] == [
        {
            "experiment_id": submitted.experiment_id,
            "status": "failed",
            "failure_reason": "provider_failed_without_worker_callback",
            "provider_artifacts": {
                "provider_name": "fake",
                "provider_job_id": "provider-job-1",
                "logs_uri": "memory://fake-provider/provider-job-1/logs.txt",
                "detail_uri": "memory://fake-provider/provider-job-1/detail.json",
                "metadata": {},
            },
        }
    ]


def test_service_snapshot_round_trips_course_state_and_counters(tmp_path):
    service = _service()
    first = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    service.record_worker_event(
        {
            "experiment_id": first.experiment_id,
            "event_type": "worker_completed",
            "validation_losses": [3.4, 3.1],
            "final_validation_loss": 3.1,
            "actual_runtime_seconds": 420,
        }
    )
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=4096),
        predicted_final_loss=2.7,
        predicted_final_loss_lower=2.6,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=7200)
    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_completed",
            "validation_losses": [2.8, 2.65],
            "final_validation_loss": 2.65,
            "actual_runtime_seconds": 7200,
        }
    )
    service.admin_apply_budget_adjustment(
        student_id="student-1",
        seconds=-30,
        reason="rounding refund",
        actor="staff",
        experiment_id=first.experiment_id,
    )
    snapshot_path = tmp_path / "state.json"

    service.save_state_snapshot(snapshot_path)
    restored = _service()
    restored.load_state_snapshot(snapshot_path)
    second = restored.submit(
        student_id="student-2",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )

    assert restored.get_result("student-1", first.experiment_id).status == "completed"
    assert restored.get_result("student-1", first.experiment_id).final_validation_loss == 3.1
    assert restored.get_result("student-1", first.experiment_id).used_runtime_seconds == 420
    assert restored.get_result("student-1", first.experiment_id).completed_at == "2026-07-16T15:00:00Z"
    assert restored.get_budget("student-1").charged_seconds == 390
    assert restored.admin_export_course_records()["final_runs"][0]["actual_final_validation_loss"] == 2.65
    assert restored.admin_export_course_records()["budget_adjustments"][0]["reason"] == "rounding refund"
    assert restored.admin_export_course_records()["budget_adjustments"][0][
        "before_charged_seconds"
    ] == 420
    assert restored.admin_export_course_records()["budget_adjustments"][0][
        "after_charged_seconds"
    ] == 390
    assert second.experiment_id == "exp-000002"


def test_final_run_worker_events_are_exported_for_course_reconstruction(tmp_path):
    service = _service()
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=4096),
        predicted_final_loss=2.7,
        predicted_final_loss_lower=2.6,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=7200)

    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_started",
        }
    )
    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "validation",
            "step": 10,
            "loss": 2.8,
        }
    )
    service.record_worker_event(
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_completed",
            "validation_losses": [2.8, 2.65],
            "final_validation_loss": 2.65,
            "actual_runtime_seconds": 7200,
        }
    )

    events = service.admin_export_course_records()["worker_events"]

    assert [
        {
            "final_run_id": event.get("final_run_id"),
            "event_type": event["event_type"],
            "student_id": event["student_id"],
        }
        for event in events
        if event.get("final_run_id") == "final-run-000001"
    ] == [
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_started",
            "student_id": "student-1",
        },
        {
            "final_run_id": "final-run-000001",
            "event_type": "validation",
            "student_id": "student-1",
        },
        {
            "final_run_id": "final-run-000001",
            "event_type": "worker_completed",
            "student_id": "student-1",
        },
    ]

    snapshot_path = tmp_path / "state.json"
    service.save_state_snapshot(snapshot_path)
    restored = _service()
    restored.load_state_snapshot(snapshot_path)

    restored_events = restored.admin_export_course_records()["worker_events"]

    assert [
        event["event_type"]
        for event in restored_events
        if event.get("final_run_id") == "final-run-000001"
    ] == ["worker_started", "validation", "worker_completed"]


def test_state_snapshot_preserves_empty_final_submission_freeze(tmp_path):
    service = _service()
    service.freeze_final_submissions(actor="staff", reason="deadline")
    snapshot_path = tmp_path / "state.json"

    service.save_state_snapshot(snapshot_path)
    restored = _service()
    restored.load_state_snapshot(snapshot_path)

    with pytest.raises(ValueError, match="frozen"):
        restored.set_final_submission(
            student_id="student-1",
            training_config=_config(),
            predicted_final_loss=2.8,
            predicted_final_loss_lower=2.7,
            predicted_final_loss_upper=2.9,
        )
    assert restored.launch_final_runs(max_runtime_seconds=48 * 3600) == []
    assert restored.admin_export_course_records()["admin_actions"][0][
        "action_type"
    ] == "freeze_final_submissions"


def test_state_snapshot_restores_active_final_run_provider_job_for_admin_cancel(tmp_path):
    service = ExperimentService(
        provider=FakeProviderAdapter(now=lambda: "2026-07-16T15:00:00Z"),
        total_budget_seconds=12 * 3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda run_id: f"memory://manifests/{run_id}.json",
        callback_url="https://backend/internal/provider-events",
    )
    service.set_final_submission(
        student_id="student-1",
        training_config=_config(train_tokens=4096),
        predicted_final_loss=2.7,
        predicted_final_loss_lower=2.6,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=7200)
    snapshot_path = tmp_path / "state.json"

    service.save_state_snapshot(snapshot_path)
    restored = ExperimentService(
        provider=FakeProviderAdapter(now=lambda: "2026-07-16T15:00:00Z"),
        total_budget_seconds=12 * 3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda run_id: f"memory://manifests/{run_id}.json",
        callback_url="https://backend/internal/provider-events",
    )
    restored.load_state_snapshot(snapshot_path)
    result = restored.admin_cancel_final_run(
        "final-run-000001",
        reason="restart cleanup",
        actor="staff",
    )

    assert result.status == "cancelled"
    assert restored.provider.get_status("fake-job-000001").status is ProviderJobStatus.CANCELLED


def test_load_state_snapshot_rejects_unknown_failure_reason(tmp_path):
    service = _service()
    submitted = service.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    service.record_worker_event(
        {
            "experiment_id": submitted.experiment_id,
            "event_type": "worker_failed",
            "failure_type": "numerical",
            "actual_runtime_seconds": 120,
            "message": "nan loss",
        }
    )
    snapshot_path = tmp_path / "state.json"
    service.save_state_snapshot(snapshot_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["experiments"][0]["failure_reason"] = "RawProviderSecretFailure"
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = _service()
    with pytest.raises(
        ValueError,
        match="unknown failure_reason in persisted state: RawProviderSecretFailure",
    ):
        restored.load_state_snapshot(snapshot_path)


def test_save_state_snapshot_keeps_existing_snapshot_when_replacement_fails(
    monkeypatch, tmp_path
):
    snapshot_path = tmp_path / "state.json"
    original = _service()
    first = original.submit(
        student_id="student-1",
        config=_config(train_tokens=1024),
        requested_runtime_seconds=600,
    )
    original.save_state_snapshot(snapshot_path)
    before = snapshot_path.read_text(encoding="utf-8")

    updated = _service()
    updated.submit(
        student_id="student-2",
        config=_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )

    def fail_replace(self, target):
        if target == snapshot_path:
            raise OSError("simulated replace failure")
        return original_replace(self, target)

    original_replace = type(snapshot_path).replace
    monkeypatch.setattr(type(snapshot_path), "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        updated.save_state_snapshot(snapshot_path)

    restored = _service()
    restored.load_state_snapshot(snapshot_path)
    assert snapshot_path.read_text(encoding="utf-8") == before
    assert restored.get_result("student-1", first.experiment_id).status == "submitted"
    assert restored.admin_list_experiments() == [
        {
            "experiment_id": first.experiment_id,
            "student_id": "student-1",
            "status": "submitted",
            "validation_losses": [],
            "final_validation_loss": None,
            "failure_reason": "",
            "staff_failure_detail": "",
            "used_runtime_seconds": None,
            "completed_at": "",
            "failed_at": "",
            "reserved_runtime_seconds": 600,
            "config_hash": restored.admin_list_experiments()[0]["config_hash"],
            "provider_name": "fake",
            "provider_job_id": "provider-job-1",
            "provider_status": "submitted",
            "provider_artifacts": {
                "provider_name": "fake",
                "provider_job_id": "provider-job-1",
                "logs_uri": "memory://fake-provider/provider-job-1/logs.txt",
                "detail_uri": "memory://fake-provider/provider-job-1/detail.json",
                "metadata": {},
            },
            "resolved_config": restored.admin_list_experiments()[0]["resolved_config"],
        }
    ]
