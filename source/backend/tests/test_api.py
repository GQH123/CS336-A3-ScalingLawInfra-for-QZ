import json

from fastapi.testclient import TestClient

from scaling_backend.api import create_app
from scaling_backend.providers.contracts import (
    ProviderCancelResult,
    ProviderArtifacts,
    ProviderJobStatus,
    ProviderStatusSnapshot,
    ProviderSubmission,
)
from scaling_backend.providers.qz_distributed.client import QzApiError
from scaling_backend.service import ExperimentService


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
        )

    def get_artifacts(self, provider_job_id):
        return ProviderArtifacts(
            provider_name="fake",
            provider_job_id=provider_job_id,
            logs_uri=f"memory://fake-provider/{provider_job_id}/logs.txt",
            detail_uri=f"memory://fake-provider/{provider_job_id}/detail.json",
        )


class RejectingQzProvider(FakeProvider):
    provider_name = "qz_distributed"

    def submit(self, manifest):
        raise QzApiError(
            "QZ API request failed",
            code=-100000,
            staff_message='framework_config[0]: quota_id "secret" not found',
        )


def _client(*, provider=None, **service_kwargs):
    service = ExperimentService(
        provider=provider or FakeProvider(),
        total_budget_seconds=3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda experiment_id: f"memory://{experiment_id}",
        callback_url="https://backend/internal/provider-events",
        **service_kwargs,
    )
    app = create_app(
        service=service,
        api_keys={"student-1-key": "student-1", "student-2-key": "student-2"},
        internal_callback_token="internal-secret",
        admin_api_token="admin-secret",
    )
    return TestClient(app), service


def _headers(key="student-1-key"):
    return {"Authorization": f"Bearer {key}"}


def _admin_headers(key="admin-secret"):
    return {"Authorization": f"Bearer {key}"}


def _config():
    return {
        "model": {"num_hidden_layers": 2, "hidden_size": 128},
        "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
    }


def test_budget_requires_api_key_and_returns_student_budget():
    client, _service = _client()

    unauthorized = client.get("/budget")
    response = client.get("/budget", headers=_headers())

    assert unauthorized.status_code == 401
    assert unauthorized.json() == {
        "error": "invalid_api_key",
        "message": "Invalid or missing API key.",
    }
    assert response.status_code == 200
    assert response.json() == {
        "student_id": "student-1",
        "total_seconds": 3600,
        "reserved_seconds": 0,
        "charged_seconds": 0,
        "remaining_seconds": 3600,
    }


def test_submit_returns_experiment_id_and_budget_reservation():
    client, _service = _client()

    response = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "experiment_id": "exp-000001",
        "status": "queued",
        "budget_reserved_seconds": 300,
        "resource_warnings": [
            "very_low_optimizer_steps",
        ],
        "resolved_config": body["resolved_config"],
    }
    assert body["resolved_config"]["total_optimizer_steps"] == 1
    assert "data_manifest_id" not in body["resolved_config"]
    assert "eval_manifest_id" not in body["resolved_config"]
    assert client.get("/budget", headers=_headers()).json()["remaining_seconds"] == 3300


def test_submit_provider_rejection_returns_structured_error_and_rolls_back():
    client, service = _client(provider=RejectingQzProvider())

    response = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": "provider_submission_failed",
        "message": (
            "Training provider rejected the submission. "
            "Course staff should inspect the control-node logs."
        ),
        "provider_error_code": -100000,
    }
    assert "quota_id" not in str(response.json())
    assert client.get("/budget", headers=_headers()).json()["remaining_seconds"] == 3600
    assert service.admin_list_experiments() == []


def test_submit_public_resolved_config_omits_private_manifest_ids():
    client, service = _client()

    response = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    )

    assert response.status_code == 200
    public_resolved = response.json()["resolved_config"]
    assert public_resolved["parameter_count_estimate"] > 0
    assert public_resolved["tokens_per_optimizer_step"] == 1024
    assert public_resolved["total_optimizer_steps"] == 1
    assert "data_manifest_id" not in public_resolved
    assert "eval_manifest_id" not in public_resolved
    assert "provider_manifest_uri" not in public_resolved
    staff_resolved = service.admin_list_experiments()[0]["resolved_config"]
    assert staff_resolved["data_manifest_id"] == "exploratory-train-v0"
    assert staff_resolved["eval_manifest_id"] == "exploratory-eval-v0"


def test_submit_and_result_payloads_expose_resource_warnings_metadata():
    client, _service = _client()

    response = client.post(
        "/submit",
        headers=_headers(),
        json={
            "config": {
                "model": {"num_hidden_layers": 2, "hidden_size": 128},
                "training": {"train_tokens": 1024, "learning_rate": 0.02, "num_evals": 1},
            },
            "requested_runtime_seconds": 30,
        },
    )
    result = client.get("/experiment/exp-000001", headers=_headers())

    assert response.status_code == 200
    assert response.json()["resource_warnings"] == [
        "very_low_optimizer_steps",
        "high_learning_rate",
        "very_low_runtime_budget",
    ]
    assert response.json()["resolved_config"]["resource_warnings"] == response.json()[
        "resource_warnings"
    ]
    assert result.status_code == 200
    assert result.json()["resource_warnings"] == response.json()["resource_warnings"]
    assert "provider_artifacts" not in result.json()


def test_submit_rejects_invalid_config_without_budget_reservation():
    client, _service = _client()

    response = client.post(
        "/submit",
        headers=_headers(),
        json={
            "config": {
                "model": {
                    "num_hidden_layers": 2,
                    "hidden_size": 130,
                    "num_attention_heads": 8,
                },
                "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
            },
            "requested_runtime_seconds": 300,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid_config",
        "message": "model.hidden_size must be divisible by model.num_attention_heads",
    }
    assert client.get("/budget", headers=_headers()).json()["remaining_seconds"] == 3600


def test_submit_rejects_unknown_request_fields():
    client, _service = _client()

    response = client.post(
        "/submit",
        headers=_headers(),
        json={
            "config": _config(),
            "requested_runtime_seconds": 300,
            "unexpected_request_field": True,
        },
    )

    assert response.status_code == 422
    assert any(
        error["type"] == "extra_forbidden" for error in response.json()["detail"]
    )
    assert client.get("/budget", headers=_headers()).json()["remaining_seconds"] == 3600


def test_submit_defers_when_student_active_experiment_limit_is_reached():
    client, service = _client(max_active_experiments_per_student=1)

    accepted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    )
    deferred = client.post(
        "/submit",
        headers=_headers(),
        json={
            "config": {
                "model": {"num_hidden_layers": 2, "hidden_size": 128},
                "training": {"train_tokens": 2048, "learning_rate": 3e-4, "num_evals": 1},
            },
            "requested_runtime_seconds": 300,
        },
    )

    assert accepted.status_code == 200
    assert deferred.status_code == 200
    assert deferred.json()["status"] == "queued"
    assert client.get("/budget", headers=_headers()).json()["reserved_seconds"] == 600
    assert len(service.provider.submitted) == 1


def test_submit_defers_when_global_active_experiment_limit_is_reached():
    client, service = _client(max_active_experiments_global=1)

    accepted = client.post(
        "/submit",
        headers=_headers("student-1-key"),
        json={"config": _config(), "requested_runtime_seconds": 300},
    )
    deferred = client.post(
        "/submit",
        headers=_headers("student-2-key"),
        json={
            "config": {
                "model": {"num_hidden_layers": 2, "hidden_size": 128},
                "training": {"train_tokens": 2048, "learning_rate": 3e-4, "num_evals": 1},
            },
            "requested_runtime_seconds": 300,
        },
    )

    assert accepted.status_code == 200
    assert deferred.status_code == 200
    assert deferred.json()["status"] == "queued"
    assert client.get("/budget", headers=_headers("student-2-key")).json()[
        "reserved_seconds"
    ] == 300
    assert len(service.provider.submitted) == 1


def test_duplicate_same_student_submit_returns_409_existing_experiment_id():
    client, _service = _client()
    payload = {"config": _config(), "requested_runtime_seconds": 300}
    first = client.post("/submit", headers=_headers(), json=payload)

    duplicate = client.post("/submit", headers=_headers(), json=payload)

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.json() == {
        "error": "duplicate_config",
        "message": "An identical configuration was already submitted.",
        "experiment_id": "exp-000001",
    }


def test_cross_student_identical_submit_does_not_reveal_other_experiment():
    client, _service = _client()
    payload = {"config": _config(), "requested_runtime_seconds": 300}
    student_one = client.post("/submit", headers=_headers("student-1-key"), json=payload)
    student_two = client.post("/submit", headers=_headers("student-2-key"), json=payload)

    assert student_one.json()["experiment_id"] == "exp-000001"
    assert student_two.json()["experiment_id"] == "exp-000002"


def test_get_experiment_returns_student_scoped_result():
    client, _service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()

    own = client.get(f"/experiment/{submitted['experiment_id']}", headers=_headers())
    other = client.get(
        f"/experiment/{submitted['experiment_id']}",
        headers=_headers("student-2-key"),
    )

    assert own.status_code == 200
    assert own.json()["status"] == "queued"
    assert own.json()["validation_losses"] == []
    assert own.json()["final_validation_loss"] is None
    assert other.status_code == 404
    assert other.json() == {
        "error": "experiment_not_found",
        "message": "Experiment not found.",
    }


def test_list_experiments_returns_only_authenticated_students_runs():
    client, _service = _client()
    first = client.post(
        "/submit",
        headers=_headers("student-1-key"),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    other = client.post(
        "/submit",
        headers=_headers("student-2-key"),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    second_config = {
        "model": {"num_hidden_layers": 3, "hidden_size": 128},
        "training": {"train_tokens": 2048, "learning_rate": 3e-4, "num_evals": 1},
    }
    second = client.post(
        "/submit",
        headers=_headers("student-1-key"),
        json={"config": second_config, "requested_runtime_seconds": 120},
    ).json()

    response = client.get("/experiments", headers=_headers("student-1-key"))

    assert response.status_code == 200
    assert response.json() == {
        "experiments": [
            {
                "experiment_id": first["experiment_id"],
                "status": "queued",
                "validation_losses": [],
                "final_validation_loss": None,
                "failure_reason": "",
                "used_runtime_seconds": None,
                "completed_at": "",
                "failed_at": "",
                "resource_warnings": [
                    "very_low_optimizer_steps",
                ],
            },
            {
                "experiment_id": second["experiment_id"],
                "status": "queued",
                "validation_losses": [],
                "final_validation_loss": None,
                "failure_reason": "",
                "used_runtime_seconds": None,
                "completed_at": "",
                "failed_at": "",
                "resource_warnings": [
                    "very_low_optimizer_steps",
                ],
            },
        ]
    }
    assert other["experiment_id"] not in {
        item["experiment_id"] for item in response.json()["experiments"]
    }


def test_internal_callback_requires_internal_token_and_updates_result():
    client, _service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()

    unauthorized = client.post(
        "/internal/provider-events",
        json={"experiment_id": submitted["experiment_id"], "event_type": "validation"},
    )
    authorized = client.post(
        "/internal/provider-events",
        headers={"Authorization": "Bearer internal-secret"},
        json={
            "experiment_id": submitted["experiment_id"],
            "event_type": "worker_completed",
            "validation_losses": [3.0],
            "final_validation_loss": 3.0,
            "actual_runtime_seconds": 120,
        },
    )
    result = client.get(f"/experiment/{submitted['experiment_id']}", headers=_headers())

    assert unauthorized.status_code == 401
    assert unauthorized.json() == {
        "error": "invalid_callback_token",
        "message": "Invalid or missing callback token.",
    }
    assert authorized.status_code == 200
    assert authorized.json() == {"ok": True}
    assert result.json()["status"] == "completed"
    assert result.json()["validation_losses"] == [3.0]
    assert result.json()["final_validation_loss"] == 3.0
    assert result.json()["used_runtime_seconds"] == 120
    assert result.json()["completed_at"] == "2026-07-16T15:00:00Z"
    assert result.json()["failed_at"] == ""


def test_internal_callback_rejects_unknown_event_type():
    client, _service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()

    response = client.post(
        "/internal/provider-events",
        headers={"Authorization": "Bearer internal-secret"},
        json={
            "experiment_id": submitted["experiment_id"],
            "event_type": "worker_compeleted",
            "final_validation_loss": 3.0,
        },
    )
    result = client.get(f"/experiment/{submitted['experiment_id']}", headers=_headers())

    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid_worker_event",
        "message": "unknown worker event_type: worker_compeleted",
    }
    assert result.json()["status"] == "queued"
    assert result.json()["final_validation_loss"] is None


def test_student_api_maps_internal_lost_status_to_system_failed():
    client, service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.SUCCEEDED,
        raw_status="COMPLETED",
        message="provider succeeded before worker callback",
    )

    admin_polled = client.post(
        f"/admin/experiments/{submitted['experiment_id']}/poll",
        headers=_admin_headers(),
    )
    public = client.get(
        f"/experiment/{submitted['experiment_id']}",
        headers=_headers(),
    )

    assert admin_polled.status_code == 200
    assert admin_polled.json()["status"] == "lost"
    assert public.status_code == 200
    assert public.json()["status"] == "system_failed"
    assert public.json()["failure_reason"] == "worker_lost"


def test_student_api_maps_internal_unknown_status_to_system_failed():
    client, service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.UNKNOWN,
        raw_status="QZ_NEW_STATUS",
        message="provider returned an unknown status",
    )

    client.post(
        f"/admin/experiments/{submitted['experiment_id']}/poll",
        headers=_admin_headers(),
    )
    public = client.get(
        f"/experiment/{submitted['experiment_id']}",
        headers=_headers(),
    )

    assert public.status_code == 200
    assert public.json()["status"] == "system_failed"
    assert public.json()["failure_reason"] == "unknown_system"


def test_student_api_sanitizes_provider_failure_without_worker_callback():
    client, service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.FAILED,
        raw_status="QZ_POD_CRASHLOOP",
        message="node gpu-b200-7 kubelet: image pull secret expired",
    )

    admin_polled = client.post(
        f"/admin/experiments/{submitted['experiment_id']}/poll",
        headers=_admin_headers(),
    )
    public = client.get(
        f"/experiment/{submitted['experiment_id']}",
        headers=_headers(),
    )

    assert admin_polled.status_code == 200
    assert admin_polled.json()["status"] == "failed"
    assert admin_polled.json()["failure_reason"] == (
        "provider_failed_without_worker_callback"
    )
    assert public.status_code == 200
    assert public.json()["status"] == "system_failed"
    assert public.json()["failure_reason"] == "unknown_system"
    assert public.json()["final_validation_loss"] is None
    assert "failure_detail" not in public.json()


def test_student_api_surfaces_traceback_tail_without_worker_callback():
    client, service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="qz_distributed",
        provider_job_id="job-f2175c46",
        status=ProviderJobStatus.FAILED,
        raw_status="RUNNING",
        message="processExitCode=1",
        metadata={
            "log_tail": [
                "worker failed for experiment_id=exp-000001: "
                "NotImplementedError: Q must be fp16/bf16/fp8_e4m3fn/"
                "fp8_e5m2, got float32",
                "Traceback (most recent call last):",
                '  File "/opt/anaconda3/envs/fnlps_a3_compute/lib/python3.12/'
                'site-packages/course_trainer/worker.py", line 354, in run',
                "NotImplementedError: Q must be fp16/bf16/fp8_e4m3fn/"
                "fp8_e5m2, got float32",
                "processExitCode: 1",
            ],
        },
    )

    admin_polled = client.post(
        f"/admin/experiments/{submitted['experiment_id']}/poll",
        headers=_admin_headers(),
    )
    public = client.get(
        f"/experiment/{submitted['experiment_id']}",
        headers=_headers(),
    )

    assert admin_polled.status_code == 200
    assert admin_polled.json()["failure_reason"] == (
        "provider_failed_without_worker_callback"
    )
    assert public.status_code == 200
    assert public.json()["status"] == "failed"
    assert public.json()["failure_reason"] == "unknown_student_caused"
    assert "staff_failure_detail" not in public.json()
    assert "job-f2175c46" not in public.json()["failure_detail"]
    assert "qz_distributed" not in public.json()["failure_detail"]
    assert "NotImplementedError: Q must be fp16/bf16" in public.json()[
        "failure_detail"
    ]
    assert "Traceback (most recent call last):" in public.json()["failure_detail"]


def test_student_api_surfaces_worker_failure_detail_without_staff_detail():
    client, _service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()

    callback = client.post(
        "/internal/provider-events",
        headers={"Authorization": "Bearer internal-secret"},
        json={
            "experiment_id": submitted["experiment_id"],
            "event_type": "worker_failed",
            "failure_type": "NotImplementedError",
            "message": "Q must be fp16/bf16/fp8_e4m3fn/fp8_e5m2, got float32",
        },
    )
    public = client.get(
        f"/experiment/{submitted['experiment_id']}",
        headers=_headers(),
    )

    assert callback.status_code == 200
    assert public.status_code == 200
    assert public.json()["status"] == "failed"
    assert public.json()["failure_reason"] == "unknown_student_caused"
    assert public.json()["failure_detail"] == (
        "NotImplementedError: Q must be fp16/bf16/fp8_e4m3fn/fp8_e5m2, "
        "got float32"
    )
    assert "staff_failure_detail" not in public.json()


def test_student_list_sanitizes_internal_provider_failure_reasons():
    client, service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.LOST,
        raw_status="QZ_NODE_LOST",
        message="node disappeared from provider control plane",
    )
    client.post(
        f"/admin/experiments/{submitted['experiment_id']}/poll",
        headers=_admin_headers(),
    )

    public = client.get("/experiments", headers=_headers())

    assert public.status_code == 200
    assert public.json()["experiments"][0]["status"] == "system_failed"
    assert public.json()["experiments"][0]["failure_reason"] == "unknown_system"


def test_worker_reported_infrastructure_failure_is_public_system_failed():
    client, _service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()

    callback = client.post(
        "/internal/provider-events",
        headers={"Authorization": "Bearer internal-secret"},
        json={
            "experiment_id": submitted["experiment_id"],
            "event_type": "worker_failed",
            "failure_type": "TokenizedIndexError",
            "message": "tokenized shard hash mismatch: /secure/course/train.bin",
        },
    )
    public = client.get(
        f"/experiment/{submitted['experiment_id']}",
        headers=_headers(),
    )
    exported = client.get("/admin/exports", headers=_admin_headers())

    assert callback.status_code == 200
    assert public.status_code == 200
    assert public.json()["status"] == "system_failed"
    assert public.json()["failure_reason"] == "infrastructure_error"
    assert public.json()["final_validation_loss"] is None
    assert public.json()["used_runtime_seconds"] == 0
    assert "staff_failure_detail" not in public.json()
    assert exported.json()["experiments"][0]["staff_failure_detail"] == (
        "TokenizedIndexError: tokenized shard hash mismatch: "
        "/secure/course/train.bin"
    )


def test_admin_can_import_worker_events_from_shared_jsonl_dir_idempotently(tmp_path):
    service = ExperimentService(
        provider=FakeProvider(),
        total_budget_seconds=3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda experiment_id: f"memory://{experiment_id}",
        callback_url="https://backend/internal/provider-events",
    )
    app = create_app(
        service=service,
        api_keys={"student-1-key": "student-1"},
        internal_callback_token="internal-secret",
        admin_api_token="admin-secret",
        worker_event_import_dir=str(tmp_path),
    )
    client = TestClient(app)
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    event_log = tmp_path / "exp-000001.jsonl"
    event_log.write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {
                    "experiment_id": submitted["experiment_id"],
                    "event_type": "worker_started",
                    "event_id": "exp-000001:000001",
                },
                {
                    "experiment_id": submitted["experiment_id"],
                    "event_type": "worker_completed",
                    "event_id": "exp-000001:000002",
                    "validation_losses": [3.2],
                    "final_validation_loss": 3.2,
                    "actual_runtime_seconds": 120,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    first_import = client.post("/admin/import-worker-events", headers=_admin_headers())
    second_import = client.post("/admin/import-worker-events", headers=_admin_headers())
    public = client.get(
        f"/experiment/{submitted['experiment_id']}",
        headers=_headers(),
    )

    assert first_import.status_code == 200
    assert first_import.json() == {
        "ok": True,
        "files": 1,
        "lines": 2,
        "imported": 2,
        "duplicates": 0,
    }
    assert second_import.status_code == 200
    assert second_import.json() == {
        "ok": True,
        "files": 1,
        "lines": 2,
        "imported": 0,
        "duplicates": 2,
    }
    assert public.status_code == 200
    assert public.json()["status"] == "completed"
    assert public.json()["final_validation_loss"] == 3.2
    assert public.json()["used_runtime_seconds"] == 120
    assert service.get_budget("student-1").charged_seconds == 120


def test_final_submission_can_be_created_read_and_replaced():
    client, _service = _client()
    first_payload = {
        "training_config": _config(),
        "predicted_final_loss": 2.7,
        "predicted_final_loss_lower": 2.6,
        "predicted_final_loss_upper": 2.9,
    }
    second_payload = {
        "training_config": {
            "model": {
                "num_hidden_layers": 4,
                "hidden_size": 256,
                "num_attention_heads": 2,
            },
            "training": {"train_tokens": 4096, "learning_rate": 2e-4, "num_evals": 1},
        },
        "predicted_final_loss": 2.5,
        "predicted_final_loss_lower": 2.4,
        "predicted_final_loss_upper": 2.8,
    }

    missing = client.get("/final_submission", headers=_headers())
    created = client.post("/final_submission", headers=_headers(), json=first_payload)
    replaced = client.post("/final_submission", headers=_headers(), json=second_payload)
    loaded = client.get("/final_submission", headers=_headers())
    other_student = client.get(
        "/final_submission", headers=_headers("student-2-key")
    )

    assert missing.status_code == 404
    assert missing.json() == {
        "error": "final_submission_not_found",
        "message": "Final submission not found.",
    }
    assert created.status_code == 200
    assert created.json()["student_id"] == "student-1"
    assert created.json()["training_config"] == first_payload["training_config"]
    assert replaced.status_code == 200
    assert loaded.status_code == 200
    assert loaded.json()["training_config"] == second_payload["training_config"]
    assert loaded.json()["predicted_final_loss"] == 2.5
    assert loaded.json()["predicted_final_loss_lower"] == 2.4
    assert loaded.json()["predicted_final_loss_upper"] == 2.8
    assert other_student.status_code == 404


def test_final_submission_read_returns_frozen_record_after_deadline_freeze():
    client, _service = _client()
    payload = {
        "training_config": _config(),
        "predicted_final_loss": 2.7,
        "predicted_final_loss_lower": 2.6,
        "predicted_final_loss_upper": 2.9,
    }
    client.post("/final_submission", headers=_headers(), json=payload)

    client.post(
        "/admin/final-submissions/freeze",
        headers=_admin_headers(),
        json={"actor": "staff", "reason": "deadline"},
    )
    loaded = client.get("/final_submission", headers=_headers())

    assert loaded.status_code == 200
    assert loaded.json()["training_config"] == payload["training_config"]
    assert loaded.json()["frozen_at"] == "2026-07-16T15:00:00Z"
    assert loaded.json()["frozen_by"] == "staff"
    assert loaded.json()["freeze_reason"] == "deadline"


def test_final_submission_rejects_non_finite_or_inverted_prediction_interval():
    client, _service = _client()
    inverted = {
        "training_config": _config(),
        "predicted_final_loss": 2.7,
        "predicted_final_loss_lower": 2.8,
        "predicted_final_loss_upper": 2.9,
    }
    non_finite = {
        "training_config": _config(),
        "predicted_final_loss": "NaN",
        "predicted_final_loss_lower": 2.6,
        "predicted_final_loss_upper": 2.9,
    }

    inverted_response = client.post(
        "/final_submission", headers=_headers(), json=inverted
    )
    non_finite_response = client.post(
        "/final_submission", headers=_headers(), json=non_finite
    )

    assert inverted_response.status_code == 400
    assert inverted_response.json() == {
        "error": "invalid_final_submission",
        "message": "prediction interval must satisfy lower <= point <= upper",
    }
    assert non_finite_response.status_code == 400
    assert non_finite_response.json() == {
        "error": "invalid_final_submission",
        "message": "predicted final loss values must be finite",
    }


def test_final_submission_rejects_invalid_training_config():
    client, _service = _client()
    invalid_config = _config()
    invalid_config["model"]["hidden_size"] = 130
    invalid_config["model"]["num_attention_heads"] = 8
    invalid_payload = {
        "training_config": invalid_config,
        "predicted_final_loss": 2.7,
        "predicted_final_loss_lower": 2.6,
        "predicted_final_loss_upper": 2.9,
    }

    response = client.post(
        "/final_submission",
        headers=_headers(),
        json=invalid_payload,
    )
    loaded = client.get("/final_submission", headers=_headers())

    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid_final_submission",
        "message": "model.hidden_size must be divisible by model.num_attention_heads",
    }
    assert loaded.status_code == 404


def test_final_submission_rejects_unknown_request_fields():
    client, _service = _client()
    payload = {
        "training_config": _config(),
        "predicted_final_loss": 2.7,
        "predicted_final_loss_lower": 2.6,
        "predicted_final_loss_upper": 2.9,
        "unexpected_request_field": True,
    }

    response = client.post("/final_submission", headers=_headers(), json=payload)
    loaded = client.get("/final_submission", headers=_headers())

    assert response.status_code == 422
    assert any(
        error["type"] == "extra_forbidden" for error in response.json()["detail"]
    )
    assert loaded.status_code == 404


def test_admin_routes_require_admin_token_and_return_queue_snapshot():
    client, _service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()

    student_auth = client.get("/admin/queue", headers=_headers())
    admin_auth = client.get("/admin/queue", headers=_admin_headers())

    assert student_auth.status_code == 401
    assert student_auth.json() == {
        "error": "invalid_admin_token",
        "message": "Invalid or missing admin token.",
    }
    assert admin_auth.status_code == 200
    assert admin_auth.json()["exploratory"] == [
        {
            "experiment_id": submitted["experiment_id"],
            "student_id": "student-1",
            "status": "submitted",
            "reserved_runtime_seconds": 300,
            "queue_position": 1,
            "fair_share_rank": 1,
        }
    ]
    assert admin_auth.json()["fair_share_order"] == [
        {
            "experiment_id": submitted["experiment_id"],
            "student_id": "student-1",
            "queue_position": 1,
            "fair_share_rank": 1,
        }
    ]


def test_admin_can_list_and_cancel_experiments():
    client, service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()

    listed = client.get("/admin/experiments", headers=_admin_headers())
    cancelled = client.post(
        f"/admin/experiments/{submitted['experiment_id']}/cancel",
        headers=_admin_headers(),
        json={"reason": "quota audit", "actor": "staff"},
    )
    student_result = client.get(
        f"/experiment/{submitted['experiment_id']}", headers=_headers()
    )

    assert listed.status_code == 200
    assert listed.json()["experiments"][0]["experiment_id"] == submitted["experiment_id"]
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert service.provider.cancelled == [("provider-job-1", "quota audit", "staff")]
    assert student_result.json()["status"] == "cancelled"


def test_admin_cancel_completed_experiment_preserves_result():
    client, service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    client.post(
        "/internal/provider-events",
        headers={"Authorization": "Bearer internal-secret"},
        json={
            "experiment_id": submitted["experiment_id"],
            "event_type": "worker_completed",
            "validation_losses": [3.2],
            "final_validation_loss": 3.2,
            "actual_runtime_seconds": 120,
        },
    )

    cancelled = client.post(
        f"/admin/experiments/{submitted['experiment_id']}/cancel",
        headers=_admin_headers(),
        json={"reason": "late cleanup request", "actor": "staff"},
    )
    student_result = client.get(
        f"/experiment/{submitted['experiment_id']}", headers=_headers()
    )
    exported = client.get("/admin/exports", headers=_admin_headers())

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "completed"
    assert cancelled.json()["final_validation_loss"] == 3.2
    assert student_result.json()["status"] == "completed"
    assert student_result.json()["final_validation_loss"] == 3.2
    assert service.provider.cancelled == []
    assert exported.json()["admin_actions"] == []


def test_admin_can_poll_experiment_provider_status():
    client, service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()

    polled = client.post(
        f"/admin/experiments/{submitted['experiment_id']}/poll",
        headers=_admin_headers(),
    )
    student_result = client.get(
        f"/experiment/{submitted['experiment_id']}", headers=_headers()
    )
    student_auth = client.post(
        f"/admin/experiments/{submitted['experiment_id']}/poll",
        headers=_headers(),
    )

    assert polled.status_code == 200
    assert polled.json()["status"] == "running"
    assert student_result.json()["status"] == "running"
    assert student_auth.status_code == 401
    assert service.dispatcher.get(submitted["experiment_id"]).events[-1].metadata == {
        "provider_status": "running",
        "raw_status": "RUNNING",
        "message": "container started",
    }


def test_admin_can_filter_experiments_and_inspect_student_budget():
    client, _service = _client()
    first = client.post(
        "/submit",
        headers=_headers("student-1-key"),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    second = client.post(
        "/submit",
        headers=_headers("student-2-key"),
        json={"config": _config(), "requested_runtime_seconds": 120},
    ).json()
    client.post(
        "/internal/provider-events",
        headers={"Authorization": "Bearer internal-secret"},
        json={
            "experiment_id": first["experiment_id"],
            "event_type": "worker_started",
        },
    )

    filtered = client.get(
        "/admin/experiments?student_id=student-1&status=running",
        headers=_admin_headers(),
    )
    budget = client.get(
        "/admin/students/student-2/budget",
        headers=_admin_headers(),
    )
    student_auth = client.get(
        "/admin/students/student-2/budget",
        headers=_headers("student-2-key"),
    )

    assert filtered.status_code == 200
    assert [row["experiment_id"] for row in filtered.json()["experiments"]] == [
        first["experiment_id"]
    ]
    assert second["experiment_id"] not in {
        row["experiment_id"] for row in filtered.json()["experiments"]
    }
    assert budget.status_code == 200
    assert budget.json()["student_id"] == "student-2"
    assert budget.json()["reserved_seconds"] == 120
    assert student_auth.status_code == 401


def test_admin_can_inspect_experiment_detail_with_events():
    client, _service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    callback = client.post(
        "/internal/provider-events",
        headers={"Authorization": "Bearer internal-secret"},
        json={
            "experiment_id": submitted["experiment_id"],
            "event_type": "heartbeat",
            "step": 1,
            "message": "alive",
        },
    )

    detail = client.get(
        f"/admin/experiments/{submitted['experiment_id']}",
        headers=_admin_headers(),
    )
    student_auth = client.get(
        f"/admin/experiments/{submitted['experiment_id']}",
        headers=_headers(),
    )

    assert callback.status_code == 200
    assert detail.status_code == 200
    assert detail.json()["experiment_id"] == submitted["experiment_id"]
    assert detail.json()["provider_artifacts"] == {
        "provider_name": "fake",
        "provider_job_id": "provider-job-1",
        "logs_uri": "memory://fake-provider/provider-job-1/logs.txt",
        "detail_uri": "memory://fake-provider/provider-job-1/detail.json",
        "metadata": {},
    }
    assert detail.json()["events"][0]["event_type"] == "heartbeat"
    assert detail.json()["events"][0]["payload"]["message"] == "alive"
    assert student_auth.status_code == 401


def test_admin_can_apply_budget_adjustments_and_export_records():
    client, _service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()

    adjusted = client.post(
        "/admin/budget-adjustments",
        headers=_admin_headers(),
        json={
            "student_id": "student-1",
            "seconds": -120,
            "reason": "provider outage refund",
            "actor": "staff",
            "experiment_id": submitted["experiment_id"],
        },
    )
    budget = client.get("/budget", headers=_headers())
    exported = client.get("/admin/exports", headers=_admin_headers())

    assert adjusted.status_code == 200
    assert adjusted.json()["adjustment_id"] == "budget-adjustment-000001"
    assert adjusted.json()["seconds"] == -120
    assert adjusted.json()["before_charged_seconds"] == 0
    assert adjusted.json()["after_charged_seconds"] == -120
    assert budget.json()["charged_seconds"] == -120
    assert exported.status_code == 200
    assert exported.json()["budget_adjustments"][0]["reason"] == "provider outage refund"
    assert exported.json()["budget_adjustments"][0]["before_charged_seconds"] == 0
    assert exported.json()["budget_adjustments"][0]["after_charged_seconds"] == -120
    assert exported.json()["budget_snapshots"] == [
        {
            "student_id": "student-1",
            "total_seconds": 3600,
            "reserved_seconds": 300,
            "charged_seconds": -120,
            "remaining_seconds": 3600 - 300 + 120,
        }
    ]


def test_admin_can_freeze_submissions_launch_final_runs_and_export_actual_loss():
    client, _service = _client()
    client.post(
        "/final_submission",
        headers=_headers(),
        json={
            "training_config": _config(),
            "predicted_final_loss": 2.7,
            "predicted_final_loss_lower": 2.6,
            "predicted_final_loss_upper": 2.9,
        },
    )

    frozen = client.post(
        "/admin/final-submissions/freeze",
        headers=_admin_headers(),
        json={"actor": "staff", "reason": "deadline"},
    )
    launched = client.post(
        "/admin/final-runs/launch",
        headers=_admin_headers(),
        json={
            "max_runtime_seconds": 48 * 3600,
            "actor": "staff",
            "reason": "deadline",
        },
    )
    callback = client.post(
        "/internal/provider-events",
        headers={"Authorization": "Bearer internal-secret"},
        json={
            "final_run_id": "final-run-000001",
            "event_type": "worker_completed",
            "validation_losses": [2.55],
            "final_validation_loss": 2.55,
            "actual_runtime_seconds": 48 * 3600,
        },
    )
    exported = client.get("/admin/exports", headers=_admin_headers())

    assert frozen.status_code == 200
    assert frozen.json()["final_submissions"][0]["frozen_by"] == "staff"
    assert launched.status_code == 200
    assert launched.json()["final_runs"][0]["final_run_id"] == "final-run-000001"
    assert callback.status_code == 200
    assert exported.json()["final_runs"][0]["actual_final_validation_loss"] == 2.55
    assert exported.json()["final_runs"][0]["prediction_absolute_error"] == 0.15
    assert exported.json()["final_runs"][0]["prediction_interval_covered"] is False
    assert exported.json()["final_runs"][0]["prediction_interval_width"] == 0.3
    assert exported.json()["final_runs"][0]["prediction_interval_miss_distance"] == 0.05
    assert exported.json()["final_runs"][0]["prediction_interval_score"] == 0.8
    assert exported.json()["final_runs"][0]["prediction_quality_penalty"] == 0.475
    assert [action["action_type"] for action in exported.json()["admin_actions"]] == [
        "freeze_final_submissions",
        "launch_final_runs",
    ]
    assert exported.json()["admin_actions"][1]["actor"] == "staff"
    assert exported.json()["admin_actions"][1]["reason"] == "deadline"


def test_admin_empty_final_submission_freeze_blocks_late_student_submission():
    client, service = _client()

    frozen = client.post(
        "/admin/final-submissions/freeze",
        headers=_admin_headers(),
        json={"actor": "staff", "reason": "deadline"},
    )
    late_submission = client.post(
        "/final_submission",
        headers=_headers(),
        json={
            "training_config": _config(),
            "predicted_final_loss": 2.7,
            "predicted_final_loss_lower": 2.6,
            "predicted_final_loss_upper": 2.9,
        },
    )
    launched = client.post(
        "/admin/final-runs/launch",
        headers=_admin_headers(),
        json={
            "max_runtime_seconds": 48 * 3600,
            "actor": "staff",
            "reason": "deadline",
        },
    )
    repeated_launch = client.post(
        "/admin/final-runs/launch",
        headers=_admin_headers(),
        json={
            "max_runtime_seconds": 48 * 3600,
            "actor": "staff",
            "reason": "deadline",
        },
    )
    exported = client.get("/admin/exports", headers=_admin_headers())

    assert frozen.status_code == 200
    assert frozen.json() == {"final_submissions": []}
    assert late_submission.status_code == 400
    assert late_submission.json()["error"] == "invalid_final_submission"
    assert "frozen" in late_submission.json()["message"]
    assert launched.status_code == 200
    assert launched.json() == {"final_runs": []}
    assert repeated_launch.status_code == 200
    assert repeated_launch.json() == {"final_runs": []}
    assert service.provider.submitted == []
    assert [
        action["action_type"]
        for action in exported.json()["admin_actions"]
    ] == ["freeze_final_submissions", "launch_final_runs"]


def test_admin_can_poll_final_run_provider_status_without_exposing_loss():
    client, service = _client()
    client.post(
        "/final_submission",
        headers=_headers(),
        json={
            "training_config": _config(),
            "predicted_final_loss": 2.7,
            "predicted_final_loss_lower": 2.6,
            "predicted_final_loss_upper": 2.9,
        },
    )
    client.post(
        "/admin/final-submissions/freeze",
        headers=_admin_headers(),
        json={"actor": "staff", "reason": "deadline"},
    )
    client.post(
        "/admin/final-runs/launch",
        headers=_admin_headers(),
        json={
            "max_runtime_seconds": 48 * 3600,
            "actor": "staff",
            "reason": "deadline",
        },
    )
    service.provider.next_status = ProviderStatusSnapshot(
        provider_name="fake",
        provider_job_id="provider-job-1",
        status=ProviderJobStatus.SUCCEEDED,
        raw_status="COMPLETED",
        message="container exited zero",
    )

    polled = client.post(
        "/admin/final-runs/final-run-000001/poll",
        headers=_admin_headers(),
    )
    student_auth = client.post(
        "/admin/final-runs/final-run-000001/poll",
        headers=_headers(),
    )

    assert polled.status_code == 200
    assert polled.json()["status"] == "lost"
    assert polled.json()["actual_final_validation_loss"] is None
    assert polled.json()["failure_reason"] == "missing_worker_completed_callback"
    assert student_auth.status_code == 401


def test_admin_can_cancel_final_run_provider_job():
    client, service = _client()
    client.post(
        "/final_submission",
        headers=_headers(),
        json={
            "training_config": _config(),
            "predicted_final_loss": 2.7,
            "predicted_final_loss_lower": 2.6,
            "predicted_final_loss_upper": 2.9,
        },
    )
    client.post(
        "/admin/final-submissions/freeze",
        headers=_admin_headers(),
        json={"actor": "staff", "reason": "deadline"},
    )
    client.post(
        "/admin/final-runs/launch",
        headers=_admin_headers(),
        json={
            "max_runtime_seconds": 48 * 3600,
            "actor": "staff",
            "reason": "deadline",
        },
    )

    cancelled = client.post(
        "/admin/final-runs/final-run-000001/cancel",
        headers=_admin_headers(),
        json={"reason": "staff dry-run cleanup", "actor": "staff"},
    )
    student_auth = client.post(
        "/admin/final-runs/final-run-000001/cancel",
        headers=_headers(),
        json={"reason": "student should not cancel hidden run", "actor": "student"},
    )

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["failure_reason"] == "admin_intervention"
    assert service.provider.cancelled == [
        ("provider-job-1", "staff dry-run cleanup", "staff")
    ]
    assert student_auth.status_code == 401


def test_admin_can_mark_final_run_system_failure():
    client, _service = _client()
    client.post(
        "/final_submission",
        headers=_headers(),
        json={
            "training_config": _config(),
            "predicted_final_loss": 2.7,
            "predicted_final_loss_lower": 2.6,
            "predicted_final_loss_upper": 2.9,
        },
    )
    client.post(
        "/admin/final-submissions/freeze",
        headers=_admin_headers(),
        json={"actor": "staff", "reason": "deadline"},
    )
    client.post(
        "/admin/final-runs/launch",
        headers=_admin_headers(),
        json={
            "max_runtime_seconds": 48 * 3600,
            "actor": "staff",
            "reason": "deadline",
        },
    )

    marked = client.post(
        "/admin/final-runs/final-run-000001/system-failure",
        headers=_admin_headers(),
        json={
            "failure_reason": "infrastructure_error",
            "staff_failure_detail": "hidden eval shard unavailable incident INC-55",
            "reason": "confirmed hidden eval outage",
            "actor": "staff",
        },
    )
    student_auth = client.post(
        "/admin/final-runs/final-run-000001/system-failure",
        headers=_headers(),
        json={
            "failure_reason": "infrastructure_error",
            "staff_failure_detail": "student should not mark hidden run",
            "reason": "student attempt",
            "actor": "student",
        },
    )
    exported = client.get("/admin/exports", headers=_admin_headers())

    assert marked.status_code == 200
    assert marked.json()["status"] == "system_failed"
    assert marked.json()["failure_reason"] == "infrastructure_error"
    assert marked.json()["actual_final_validation_loss"] is None
    assert student_auth.status_code == 401
    assert exported.json()["admin_actions"][-1]["action_type"] == (
        "mark_final_run_system_failed"
    )


def test_admin_can_mark_experiment_system_failure_and_refund():
    client, _service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()

    marked = client.post(
        f"/admin/experiments/{submitted['experiment_id']}/system-failure",
        headers=_admin_headers(),
        json={
            "failure_reason": "provider_outage",
            "staff_failure_detail": "QZ region outage incident INC-42",
            "refund_seconds": 120,
            "reason": "confirmed provider outage",
            "actor": "staff",
        },
    )
    student_auth = client.post(
        f"/admin/experiments/{submitted['experiment_id']}/system-failure",
        headers=_headers(),
        json={
            "failure_reason": "provider_outage",
            "staff_failure_detail": "student should not mark failures",
            "refund_seconds": 120,
            "reason": "student attempt",
            "actor": "student",
        },
    )
    exported = client.get("/admin/exports", headers=_admin_headers())

    assert marked.status_code == 200
    assert marked.json()["status"] == "system_failed"
    assert marked.json()["failure_reason"] == "provider_outage"
    assert marked.json()["final_validation_loss"] is None
    assert student_auth.status_code == 401
    assert exported.json()["budget_adjustments"][0]["seconds"] == -120
    assert exported.json()["admin_actions"][0]["action_type"] == (
        "mark_experiment_system_failed"
    )


def test_admin_can_poll_all_active_runs():
    client, _service = _client()
    submitted = client.post(
        "/submit",
        headers=_headers(),
        json={"config": _config(), "requested_runtime_seconds": 300},
    ).json()
    client.post(
        "/final_submission",
        headers=_headers(),
        json={
            "training_config": _config(),
            "predicted_final_loss": 2.7,
            "predicted_final_loss_lower": 2.6,
            "predicted_final_loss_upper": 2.9,
        },
    )
    client.post(
        "/admin/final-submissions/freeze",
        headers=_admin_headers(),
        json={"actor": "staff", "reason": "deadline"},
    )
    client.post(
        "/admin/final-runs/launch",
        headers=_admin_headers(),
        json={
            "max_runtime_seconds": 48 * 3600,
            "actor": "staff",
            "reason": "deadline",
        },
    )

    polled = client.post("/admin/poll-active", headers=_admin_headers())
    student_auth = client.post("/admin/poll-active", headers=_headers())

    assert polled.status_code == 200
    assert polled.json() == {
        "exploratory": [
            {
                "experiment_id": submitted["experiment_id"],
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
    assert student_auth.status_code == 401
