from scaling_backend.providers.fake import FakeProviderAdapter
from scaling_backend.service import ExperimentService
from scaling_backend.worker.run import run_worker


class CallbackSession:
    def __init__(self, service):
        self.service = service

    def post(self, _url, json=None, headers=None, timeout=None):
        self.service.record_worker_event(json or {})

        class Response:
            status_code = 200
            text = "ok"

        return Response()


def test_submit_to_worker_callback_result_smoke_path():
    provider = FakeProviderAdapter(now=lambda: "2026-07-16T15:00:00Z")
    service = ExperimentService(
        provider=provider,
        total_budget_seconds=3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda experiment_id: f"memory://{experiment_id}",
        callback_url="https://backend/internal/provider-events",
    )
    submitted = service.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 4096, "num_evals": 1, "learning_rate": 1e-3},
        },
        requested_runtime_seconds=300,
    )

    def fake_trainer(manifest, emit_event):
        assert manifest["experiment_id"] == submitted.experiment_id
        assert manifest["model_config"] == manifest["resolved_config"]["model"]
        assert manifest["model_config"]["num_hidden_layers"] == 2
        assert manifest["model_config"]["hidden_size"] == 128
        assert manifest["training_config"] == manifest["resolved_config"]["training"]
        assert manifest["training_config"]["train_tokens"] == 4096
        assert manifest["data_config"] == {"train_tokens": 4096}
        emit_event({"event_type": "validation", "step": 1, "loss": 4.2})
        emit_event({"event_type": "validation", "step": 2, "loss": 3.7})
        return {
            "validation_losses": [4.2, 3.7],
            "final_validation_loss": 3.7,
            "actual_runtime_seconds": 180,
        }

    exit_code = run_worker(
        manifest=provider.submitted_manifests["fake-job-000001"],
        callback_url="https://backend/internal/provider-events",
        callback_token="secret",
        trainer=fake_trainer,
        session=CallbackSession(service),
    )

    assert exit_code == 0
    result = service.get_result("student-1", submitted.experiment_id)
    assert result.status == "completed"
    assert result.validation_losses == [4.2, 3.7]
    assert result.final_validation_loss == 3.7
    assert service.get_budget("student-1").charged_seconds == 180
    assert service.get_budget("student-1").remaining_seconds == 3420


def test_final_run_manifest_to_worker_callback_export_smoke_path():
    provider = FakeProviderAdapter(now=lambda: "2026-07-16T15:00:00Z")
    service = ExperimentService(
        provider=provider,
        total_budget_seconds=3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda run_id: f"memory://{run_id}",
        callback_url="https://backend/internal/provider-events",
    )
    service.set_final_submission(
        student_id="student-1",
        training_config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 4096, "num_evals": 1, "learning_rate": 1e-3},
        },
        predicted_final_loss=2.7,
        predicted_final_loss_lower=2.6,
        predicted_final_loss_upper=2.9,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=48 * 3600)

    def fake_trainer(manifest, emit_event):
        assert manifest["final_run_id"] == "final-run-000001"
        assert manifest["model_config"] == manifest["resolved_config"]["model"]
        assert manifest["model_config"]["num_hidden_layers"] == 2
        assert manifest["model_config"]["hidden_size"] == 128
        assert manifest["training_config"] == manifest["resolved_config"]["training"]
        assert manifest["training_config"]["train_tokens"] == 4096
        assert manifest["data_config"] == {"train_tokens": 4096}
        emit_event({"event_type": "validation", "step": 1, "loss": 2.8})
        return {
            "validation_losses": [2.8, 2.7],
            "final_validation_loss": 2.7,
            "actual_runtime_seconds": 48 * 3600,
        }

    exit_code = run_worker(
        manifest=provider.submitted_manifests["fake-job-000001"],
        callback_url="https://backend/internal/provider-events",
        callback_token="secret",
        trainer=fake_trainer,
        session=CallbackSession(service),
    )

    assert exit_code == 0
    exported = service.admin_export_course_records()
    assert exported["final_runs"][0]["status"] == "completed"
    assert exported["final_runs"][0]["actual_final_validation_loss"] == 2.7
    assert exported["final_runs"][0]["validation_losses"] == [2.8, 2.7]
