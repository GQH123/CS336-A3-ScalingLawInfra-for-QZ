from scaling_backend.providers.contracts import ProviderJobStatus
from scaling_backend.providers.qz_distributed.adapter import QzDistributedAdapter
from scaling_backend.providers.qz_distributed.payloads import (
    QzDistributedConfig,
    QzResourceSpec,
)


class FakeQzClient:
    def __init__(self):
        self.created_payloads = []
        self.detail = {}
        self.stopped = []
        self.logs = {"logs": [], "total": 0}

    def create_train_job(self, payload):
        self.created_payloads.append(payload)
        return {
            "job_id": "job-qz-1",
            "workspace_id": payload["workspace_id"],
            "extra": "raw-provider-field",
        }

    def get_train_job_detail(self, job_id):
        assert job_id == "job-qz-1"
        return self.detail

    def stop_train_job(self, job_id):
        self.stopped.append(job_id)
        return True

    def get_train_job_logs(self, job_id):
        assert job_id == "job-qz-1"
        return self.logs


def _adapter(client=None):
    return QzDistributedAdapter(
        client=client or FakeQzClient(),
        config=QzDistributedConfig(
            workspace_id="ws-1",
            project_id="project-1",
            logic_compute_group_id="lcg-1",
            image="registry.example.com/scaling-worker:latest",
            shm_gi=64,
        ),
        spec=QzResourceSpec(
            id="spec-1",
            gpu_type="H200",
            gpu_count=1,
            cpu_count=15,
            memory_gb=200,
        ),
        callback_token_env="SCALING_CALLBACK_TOKEN",
        now=lambda: "2026-07-16T15:00:00Z",
    )


def test_submit_can_include_configured_worker_trainer_in_command():
    client = FakeQzClient()
    adapter = QzDistributedAdapter(
        client=client,
        config=QzDistributedConfig(
            workspace_id="ws-1",
            project_id="project-1",
            logic_compute_group_id="lcg-1",
            image="registry.example.com/scaling-worker:latest",
        ),
        spec=QzResourceSpec(
            id="spec-1",
            gpu_type="H200",
            gpu_count=1,
            cpu_count=15,
            memory_gb=200,
        ),
        callback_token_env="SCALING_CALLBACK_TOKEN",
        trainer_spec="course_trainer.worker:train",
        now=lambda: "2026-07-16T15:00:00Z",
    )

    adapter.submit(
        {
            "experiment_id": "exp-0001",
            "manifest_uri": "https://manifests.internal/scaling/exp-0001.json",
            "callback_url": "https://backend.internal/internal/provider-events",
        }
    )

    assert "--trainer course_trainer.worker:train" in client.created_payloads[0]["command"]


def test_submit_can_include_configured_tokenized_index_validation_mode():
    client = FakeQzClient()
    adapter = QzDistributedAdapter(
        client=client,
        config=QzDistributedConfig(
            workspace_id="ws-1",
            project_id="project-1",
            logic_compute_group_id="lcg-1",
            image="registry.example.com/scaling-worker:latest",
        ),
        spec=QzResourceSpec(
            id="spec-1",
            gpu_type="H200",
            gpu_count=1,
            cpu_count=15,
            memory_gb=200,
        ),
        callback_token_env="SCALING_CALLBACK_TOKEN",
        tokenized_index_validation_mode="metadata",
        now=lambda: "2026-07-16T15:00:00Z",
    )

    adapter.submit(
        {
            "experiment_id": "exp-0001",
            "manifest_uri": "https://manifests.internal/scaling/exp-0001.json",
            "callback_url": "https://backend.internal/internal/provider-events",
        }
    )

    assert (
        "--tokenized-index-validation-mode metadata"
        in client.created_payloads[0]["command"]
    )


def test_submit_can_export_worker_callback_token_in_command():
    client = FakeQzClient()
    adapter = QzDistributedAdapter(
        client=client,
        config=QzDistributedConfig(
            workspace_id="ws-1",
            project_id="project-1",
            logic_compute_group_id="lcg-1",
            image="registry.example.com/scaling-worker:latest",
        ),
        spec=QzResourceSpec(
            id="spec-1",
            gpu_type="H200",
            gpu_count=1,
            cpu_count=15,
            memory_gb=200,
        ),
        callback_token_env="SCALING_CALLBACK_TOKEN",
        callback_token_value="secret-token",
        now=lambda: "2026-07-16T15:00:00Z",
    )

    adapter.submit(
        {
            "experiment_id": "exp-0001",
            "manifest_uri": "https://manifests.internal/scaling/exp-0001.json",
            "callback_url": "https://backend.internal/internal/provider-events",
        }
    )

    assert "export SCALING_CALLBACK_TOKEN=secret-token" in (
        client.created_payloads[0]["command"]
    )


def test_submit_can_wrap_worker_command_with_conda_activation():
    client = FakeQzClient()
    adapter = QzDistributedAdapter(
        client=client,
        config=QzDistributedConfig(
            workspace_id="ws-1",
            project_id="project-1",
            logic_compute_group_id="lcg-1",
            image="registry.example.com/scaling-worker:latest",
        ),
        spec=QzResourceSpec(
            id="spec-1",
            gpu_type="H200",
            gpu_count=1,
            cpu_count=15,
            memory_gb=200,
        ),
        callback_token_env="SCALING_CALLBACK_TOKEN",
        trainer_spec="course_trainer.worker:train",
        worker_conda_env="fnlps_a3_compute",
        worker_conda_init="/opt/anaconda3/etc/profile.d/conda.sh",
        now=lambda: "2026-07-16T15:00:00Z",
    )

    adapter.submit(
        {
            "experiment_id": "exp-0001",
            "manifest_uri": "https://manifests.internal/scaling/exp-0001.json",
            "callback_url": "https://backend.internal/internal/provider-events",
        }
    )

    command = client.created_payloads[0]["command"]
    assert command.startswith("bash -lc ")
    assert "conda activate fnlps_a3_compute" in command
    assert "exec python -m scaling_backend.worker.run" in command


def test_submit_can_use_shared_jsonl_worker_event_log():
    client = FakeQzClient()
    adapter = QzDistributedAdapter(
        client=client,
        config=QzDistributedConfig(
            workspace_id="ws-1",
            project_id="project-1",
            logic_compute_group_id="lcg-1",
            image="registry.example.com/scaling-worker:latest",
        ),
        spec=QzResourceSpec(
            id="spec-1",
            gpu_type="H200",
            gpu_count=1,
            cpu_count=15,
            memory_gb=200,
        ),
        callback_token_env="SCALING_CALLBACK_TOKEN",
        worker_event_log_dir="/shared/course/worker-events",
        now=lambda: "2026-07-16T15:00:00Z",
    )

    adapter.submit(
        {
            "experiment_id": "exp-0001",
            "manifest_uri": "file:///shared/course/manifests/exp-0001.json",
            "callback_url": "https://api.internal/internal/provider-events",
        }
    )

    command = client.created_payloads[0]["command"]
    assert "--event-sink jsonl" in command
    assert "--event-log-path /shared/course/worker-events/exp-0001.jsonl" in command


def test_submit_shared_jsonl_worker_event_log_does_not_export_callback_token():
    client = FakeQzClient()
    adapter = QzDistributedAdapter(
        client=client,
        config=QzDistributedConfig(
            workspace_id="ws-1",
            project_id="project-1",
            logic_compute_group_id="lcg-1",
            image="registry.example.com/scaling-worker:latest",
        ),
        spec=QzResourceSpec(
            id="spec-1",
            gpu_type="H200",
            gpu_count=1,
            cpu_count=15,
            memory_gb=200,
        ),
        callback_token_env="SCALING_CALLBACK_TOKEN",
        callback_token_value="secret-token",
        worker_event_log_dir="/shared/course/worker-events",
        now=lambda: "2026-07-16T15:00:00Z",
    )

    adapter.submit(
        {
            "experiment_id": "exp-0001",
            "manifest_uri": "file:///shared/course/manifests/exp-0001.json",
            "callback_url": "https://api.internal/internal/provider-events",
        }
    )

    command = client.created_payloads[0]["command"]
    assert "--event-sink jsonl" in command
    assert "secret-token" not in command
    assert "export SCALING_CALLBACK_TOKEN" not in command


def test_submit_builds_qz_payload_from_frozen_manifest():
    client = FakeQzClient()
    adapter = _adapter(client)
    manifest = {
        "experiment_id": "exp-0001",
        "manifest_uri": "https://manifests.internal/scaling/exp-0001.json",
        "callback_url": "https://backend.internal/internal/provider-events",
    }

    submission = adapter.submit(manifest)

    assert submission.provider_name == "qz_distributed"
    assert submission.provider_job_id == "job-qz-1"
    assert submission.submitted_at == "2026-07-16T15:00:00Z"
    assert submission.metadata == {
        "workspace_id": "ws-1",
        "project_id": "project-1",
        "logic_compute_group_id": "lcg-1",
        "spec_id": "spec-1",
    }

    payload = client.created_payloads[0]
    assert payload["name"] == "scale-exp-0001"
    assert payload["workspace_id"] == "ws-1"
    assert payload["project_id"] == "project-1"
    assert payload["logic_compute_group_id"] == "lcg-1"
    assert payload["framework_config"][0]["image"] == (
        "registry.example.com/scaling-worker:latest"
    )
    assert payload["framework_config"][0]["resource_spec_price"]["quota_id"] == "spec-1"
    assert (
        "--manifest-uri https://manifests.internal/scaling/exp-0001.json"
        in payload["command"]
    )
    assert (
        "--callback-url https://backend.internal/internal/provider-events"
        in payload["command"]
    )
    assert "--callback-token-env SCALING_CALLBACK_TOKEN" in payload["command"]


def test_submit_builds_qz_payload_for_final_run_manifest():
    client = FakeQzClient()
    adapter = _adapter(client)
    manifest = {
        "run_kind": "final",
        "final_run_id": "final-run-000001",
        "manifest_uri": "https://manifests.internal/scaling/final-run-000001.json",
        "callback_url": "https://backend.internal/internal/provider-events",
    }

    submission = adapter.submit(manifest)

    assert submission.provider_job_id == "job-qz-1"
    assert client.created_payloads[0]["name"] == "scale-final-run-000001"
    assert (
        "--manifest-uri https://manifests.internal/scaling/final-run-000001.json"
        in client.created_payloads[0]["command"]
    )


def test_submit_rejects_manifest_without_required_execution_fields():
    adapter = _adapter()

    try:
        adapter.submit({"experiment_id": "exp-0001"})
    except ValueError as exc:
        assert "manifest_uri" in str(exc)
    else:
        raise AssertionError("submit should reject incomplete manifests")


def test_get_status_normalizes_qz_detail_fields_and_keeps_staff_metadata():
    client = FakeQzClient()
    client.detail = {
        "status": "RUNNING",
        "message": "container started",
        "created_at": "2026-07-16T15:00:00Z",
        "node_names": ["gpu-1"],
    }
    adapter = _adapter(client)

    snapshot = adapter.get_status("job-qz-1")

    assert snapshot.provider_name == "qz_distributed"
    assert snapshot.provider_job_id == "job-qz-1"
    assert snapshot.status is ProviderJobStatus.RUNNING
    assert snapshot.raw_status == "RUNNING"
    assert snapshot.message == "container started"
    assert snapshot.metadata["created_at"] == "2026-07-16T15:00:00Z"
    assert snapshot.metadata["node_names"] == ["gpu-1"]


def test_get_status_marks_failed_when_qz_logs_show_process_exit_despite_running_detail():
    client = FakeQzClient()
    client.detail = {"status": "RUNNING", "message": "container still marked active"}
    client.logs = {
        "logs": [
            {"message": "Allocator (GPU_0_bfc) ran out of memory"},
            {"message": "processExitCode: 139"},
            {"message": "Segmentation fault (core dumped)"},
        ],
        "total": 3,
    }
    adapter = _adapter(client)

    snapshot = adapter.get_status("job-qz-1")

    assert snapshot.status is ProviderJobStatus.FAILED
    assert snapshot.raw_status == "RUNNING"
    assert "processExitCode=139" in snapshot.message
    assert snapshot.metadata["log_count"] == 3
    assert snapshot.metadata["log_crash_signal"] == "processExitCode=139"
    assert snapshot.metadata["log_tail"][-1] == "Segmentation fault (core dumped)"


def test_cancel_stops_qz_job_and_returns_provider_cancel_result():
    client = FakeQzClient()
    adapter = _adapter(client)

    result = adapter.cancel("job-qz-1", reason="student request", actor="admin")

    assert client.stopped == ["job-qz-1"]
    assert result.accepted is True
    assert result.status is ProviderJobStatus.CANCELLED
    assert result.metadata == {"reason": "student request", "actor": "admin"}


def test_get_artifacts_returns_qz_log_pointer_and_summary_metadata():
    client = FakeQzClient()
    client.logs = {"logs": [{"message": "loss=3.2"}], "total": 1}
    adapter = _adapter(client)

    artifacts = adapter.get_artifacts("job-qz-1")

    assert artifacts.provider_name == "qz_distributed"
    assert artifacts.provider_job_id == "job-qz-1"
    assert artifacts.logs_uri == "qz://train_job/job-qz-1/logs"
    assert artifacts.detail_uri == "qz://train_job/job-qz-1"
    assert artifacts.metadata == {"log_count": 1}
