import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scaling_backend.course_config import load_api_config_env
from scaling_backend.runtime import (
    _install_provider_poll_loop,
    _install_qz_session_heartbeat_loop,
    RuntimeConfigError,
    build_app_from_env,
    build_service_from_env,
    load_api_keys_csv,
    run_provider_poll_iteration,
    run_qz_session_heartbeat_iteration,
)


def test_load_api_keys_csv_accepts_headered_staff_roster(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text(
        "student_id,api_key\n"
        "student-1,key-1\n"
        "student-2,key-2\n",
        encoding="utf-8",
    )

    assert load_api_keys_csv(roster) == {
        "key-1": "student-1",
        "key-2": "student-2",
    }


def test_load_api_keys_csv_rejects_missing_required_columns(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text("student,token\nstudent-1,key-1\n", encoding="utf-8")

    try:
        load_api_keys_csv(roster)
    except RuntimeConfigError as exc:
        assert "student_id" in str(exc)
        assert "api_key" in str(exc)
    else:
        raise AssertionError("expected RuntimeConfigError")


def test_build_service_from_env_uses_fake_provider_and_runtime_settings():
    service = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_TOTAL_BUDGET_SECONDS": "7200",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SCALING_CODE_VERSION": "test-version",
            "SCALING_DATA_MANIFEST_ID": "train-manifest",
            "SCALING_EVAL_MANIFEST_ID": "eval-manifest",
            "SCALING_FINAL_DATA_MANIFEST_ID": "final-train-private-v1",
            "SCALING_FINAL_EVAL_MANIFEST_ID": "final-eval-hidden-v1",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    response = service.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
        },
        requested_runtime_seconds=300,
    )

    assert response.experiment_id == "exp-000001"
    assert response.resolved_config["code_version"] == "test-version"
    assert response.resolved_config["data_manifest_id"] == "train-manifest"
    assert service.provider.provider_name == "fake"
    assert service.get_budget("student-1").total_seconds == 7200
    service.set_final_submission(
        student_id="student-1",
        training_config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
        },
        predicted_final_loss=2.5,
        predicted_final_loss_lower=2.4,
        predicted_final_loss_upper=2.7,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=7200)

    final_manifest = service.provider.submitted_manifests["fake-job-000002"]
    assert final_manifest["data_manifest_id"] == "final-train-private-v1"
    assert final_manifest["eval_manifest_id"] == "final-eval-hidden-v1"
    assert final_manifest["resolved_config"]["data_manifest_id"] == "final-train-private-v1"
    assert final_manifest["resolved_config"]["eval_manifest_id"] == "final-eval-hidden-v1"


def test_api_config_can_enable_background_provider_polling(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text("student_id,api_key\nstudent-1,key-1\n", encoding="utf-8")
    config_path = tmp_path / "api-runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "api": {
                    "provider": "fake",
                    "student_keys_csv": str(roster),
                    "internal_callback_token": "internal-token",
                    "admin_api_token": "admin-token",
                    "callback_url": "https://backend/internal/provider-events",
                    "manifest_base_uri": "memory://manifests",
                    "provider_poll_interval_seconds": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    values = load_api_config_env(config_path)
    app = build_app_from_env(
        {"SCALING_API_CONFIG": str(config_path)},
        now=lambda: "2026-07-16T15:00:00Z",
    )

    assert values["SCALING_PROVIDER_POLL_INTERVAL_SECONDS"] == "30"
    with TestClient(app):
        assert app.state.provider_poll_interval_seconds == 30


def test_api_config_can_enable_qz_session_heartbeat(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text("student_id,api_key\nstudent-1,key-1\n", encoding="utf-8")
    config_path = tmp_path / "api-runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "api": {
                    "provider": "qz_distributed",
                    "student_keys_csv": str(roster),
                    "internal_callback_token": "internal-token",
                    "admin_api_token": "admin-token",
                    "callback_url": "https://backend/internal/provider-events",
                    "manifest_base_uri": "memory://manifests",
                },
                "qz": {
                    "session_heartbeat_interval_seconds": 60,
                },
            }
        ),
        encoding="utf-8",
    )
    provider = _FakeQzProvider(workspace_id="ws-1")

    values = load_api_config_env(config_path)
    app = build_app_from_env(
        {"SCALING_API_CONFIG": str(config_path)},
        now=lambda: "2026-07-16T15:00:00Z",
        provider=provider,
    )

    assert values["QZ_SESSION_HEARTBEAT_INTERVAL_SECONDS"] == "60"
    with TestClient(app):
        assert app.state.qz_session_heartbeat_interval_seconds == 60


def test_qz_session_heartbeat_zero_interval_is_disabled(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text("student_id,api_key\nstudent-1,key-1\n", encoding="utf-8")
    app = build_app_from_env(
        {
            "SCALING_PROVIDER": "qz_distributed",
            "SCALING_STUDENT_KEYS_CSV": str(roster),
            "SCALING_INTERNAL_CALLBACK_TOKEN": "internal-token",
            "SCALING_ADMIN_API_TOKEN": "admin-token",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "QZ_SESSION_HEARTBEAT_INTERVAL_SECONDS": "0",
        },
        now=lambda: "2026-07-16T15:00:00Z",
        provider=_FakeQzProvider(workspace_id="ws-1"),
    )

    with TestClient(app):
        assert not hasattr(app.state, "qz_session_heartbeat_interval_seconds")


def test_qz_session_heartbeat_iteration_uses_read_only_probe():
    provider = _FakeQzProvider(workspace_id="ws-1")

    result = run_qz_session_heartbeat_iteration(provider)

    assert result == {"ok": True, "provider": "qz_distributed"}
    assert provider.client.probes == ["ws-1"]


def test_qz_session_heartbeat_iteration_logs_success(caplog):
    provider = _FakeQzProvider(workspace_id="ws-1")

    with caplog.at_level(logging.INFO, logger="scaling_backend.runtime"):
        run_qz_session_heartbeat_iteration(provider)

    assert "QZ session heartbeat succeeded" in caplog.text


def test_qz_session_heartbeat_iteration_logs_to_uvicorn_error(caplog):
    provider = _FakeQzProvider(workspace_id="ws-1")

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        run_qz_session_heartbeat_iteration(provider)

    assert any(
        record.name == "uvicorn.error"
        and "QZ session heartbeat succeeded" in record.getMessage()
        for record in caplog.records
    )


def test_qz_session_heartbeat_loop_probes_once_before_waiting():
    events = []
    app = FastAPI()
    provider = _OrderedFakeQzProvider(workspace_id="ws-1", events=events)

    _install_qz_session_heartbeat_loop(
        app,
        provider,
        interval_seconds=900,
        thread_factory=_InlineThread,
        stop_event_factory=lambda: _StopAfterFirstWait(events),
    )

    for handler in app.router.on_startup:
        handler()
    for handler in app.router.on_shutdown:
        handler()

    assert events[:2] == [("probe", "ws-1"), ("wait", 900)]


def test_qz_session_heartbeat_loop_supports_apps_without_add_event_handler():
    app = _LifecycleOnlyApp()

    _install_qz_session_heartbeat_loop(
        app,
        _OrderedFakeQzProvider(workspace_id="ws-1", events=[]),
        interval_seconds=900,
    )

    assert len(app.router.on_startup) == 1
    assert len(app.router.on_shutdown) == 1


def test_provider_poll_loop_supports_apps_without_add_event_handler(tmp_path):
    app = _LifecycleOnlyApp()

    _install_provider_poll_loop(
        app,
        _PollService(),
        interval_seconds=30,
        snapshot_path=str(tmp_path / "snapshot.json"),
    )

    assert len(app.router.on_startup) == 1
    assert len(app.router.on_shutdown) == 1


class _FakeQzClient:
    def __init__(self):
        self.probes = []

    def probe_cookie_auth(self, *, workspace_id):
        self.probes.append(workspace_id)
        return {"list": [], "total": 0}


class _FakeQzConfig:
    def __init__(self, workspace_id):
        self.workspace_id = workspace_id


class _FakeQzProvider:
    provider_name = "qz_distributed"

    def __init__(self, *, workspace_id):
        self.client = _FakeQzClient()
        self.config = _FakeQzConfig(workspace_id)

    def submit(self, manifest):
        raise AssertionError("heartbeat must not submit provider jobs")

    def get_status(self, provider_job_id):
        raise AssertionError("heartbeat must not poll provider job status")

    def cancel(self, provider_job_id, reason, actor):
        raise AssertionError("heartbeat must not cancel provider jobs")

    def get_artifacts(self, provider_job_id):
        raise AssertionError("heartbeat must not fetch provider artifacts")


class _OrderedFakeQzClient:
    def __init__(self, events):
        self.events = events

    def probe_cookie_auth(self, *, workspace_id):
        self.events.append(("probe", workspace_id))
        return {"list": [], "total": 0}


class _OrderedFakeQzProvider(_FakeQzProvider):
    def __init__(self, *, workspace_id, events):
        self.client = _OrderedFakeQzClient(events)
        self.config = _FakeQzConfig(workspace_id)


class _StopAfterFirstWait:
    def __init__(self, events):
        self.events = events

    def wait(self, interval):
        self.events.append(("wait", interval))
        return True

    def set(self):
        self.events.append("stop")


class _InlineThread:
    def __init__(self, *, target, daemon, name):
        self.target = target
        self.daemon = daemon
        self.name = name

    def start(self):
        self.target()

    def join(self, timeout):
        pass


class _LifecycleOnlyRouter:
    def __init__(self):
        self.on_startup = []
        self.on_shutdown = []


class _LifecycleOnlyApp:
    def __init__(self):
        self.state = type("State", (), {})()
        self.router = _LifecycleOnlyRouter()


class _PollService:
    def admin_poll_active_runs(self):
        return {"exploratory": [], "final": []}

    def save_state_snapshot(self, path):
        Path(path).write_text("snapshot\n", encoding="utf-8")


def test_provider_poll_iteration_saves_snapshot_after_poll(tmp_path):
    snapshot_path = tmp_path / "state-snapshot.json"
    calls = []

    class FakeService:
        def admin_poll_active_runs(self):
            calls.append("poll")
            return {"exploratory": [], "final": []}

        def save_state_snapshot(self, path):
            calls.append(("save", str(path)))
            Path(path).write_text("snapshot\n", encoding="utf-8")

    report = run_provider_poll_iteration(
        FakeService(),
        snapshot_path=str(snapshot_path),
    )

    assert report == {"exploratory": [], "final": []}
    assert calls == ["poll", ("save", str(snapshot_path))]
    assert snapshot_path.read_text(encoding="utf-8") == "snapshot\n"


def test_build_service_from_env_threads_validation_tokens_into_manifests():
    service = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_TOTAL_BUDGET_SECONDS": "7200",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SCALING_VALIDATION_TOKENS_PER_EVAL": "262144",
            "SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL": "524288",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    response = service.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {
                "train_tokens": 1024,
                "validation_batch_size": 8,
                "learning_rate": 3e-4,
                "num_evals": 1,
            },
        },
        requested_runtime_seconds=300,
    )
    exploratory_manifest = service.provider.submitted_manifests["fake-job-000001"]

    service.set_final_submission(
        student_id="student-1",
        training_config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {
                "train_tokens": 1024,
                "validation_batch_size": 8,
                "learning_rate": 3e-4,
                "num_evals": 1,
            },
        },
        predicted_final_loss=2.5,
        predicted_final_loss_lower=2.4,
        predicted_final_loss_upper=2.7,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=7200)
    final_manifest = service.provider.submitted_manifests["fake-job-000002"]

    assert response.resolved_config["validation_tokens_per_eval"] == 262_144
    assert response.resolved_config["validation_batches_per_eval"] == 32
    assert exploratory_manifest["validation_config"] == {
        "eval_manifest_id": "exploratory-eval-v0",
        "validation_tokens_per_eval": 262_144,
        "validation_batches_per_eval": 32,
    }
    assert final_manifest["validation_config"] == {
        "eval_manifest_id": "final-eval-v0",
        "validation_tokens_per_eval": 524_288,
        "validation_batches_per_eval": 64,
    }
    assert final_manifest["resolved_config"]["validation_tokens_per_eval"] == 524_288


def test_build_service_from_env_threads_tokenized_index_uris_into_manifests():
    service = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_TOTAL_BUDGET_SECONDS": "7200",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SCALING_TOKENIZED_TRAIN_INDEX_URI": (
                "file:///secure/course/tokenized/train/index.json"
            ),
            "SCALING_TOKENIZED_VALIDATION_INDEX_URI": (
                "file:///secure/course/tokenized/eval/index.json"
            ),
            "SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI": (
                "file:///secure/course/tokenized/final-train/index.json"
            ),
            "SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI": (
                "file:///secure/course/tokenized/final-eval/index.json"
            ),
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    service.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
        },
        requested_runtime_seconds=300,
    )
    exploratory_manifest = service.provider.submitted_manifests["fake-job-000001"]
    assert exploratory_manifest["data_config"]["tokenized_index_uri"] == (
        "file:///secure/course/tokenized/train/index.json"
    )
    assert exploratory_manifest["validation_config"]["tokenized_index_uri"] == (
        "file:///secure/course/tokenized/eval/index.json"
    )

    service.set_final_submission(
        student_id="student-1",
        training_config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
        },
        predicted_final_loss=2.5,
        predicted_final_loss_lower=2.4,
        predicted_final_loss_upper=2.7,
    )
    service.freeze_final_submissions(actor="staff", reason="deadline")
    service.launch_final_runs(max_runtime_seconds=7200)
    final_manifest = service.provider.submitted_manifests["fake-job-000002"]
    assert final_manifest["data_config"]["tokenized_index_uri"] == (
        "file:///secure/course/tokenized/final-train/index.json"
    )
    assert final_manifest["validation_config"]["tokenized_index_uri"] == (
        "file:///secure/course/tokenized/final-eval/index.json"
    )


def test_build_service_from_env_rejects_nondivisible_validation_tokens():
    service = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SCALING_VALIDATION_TOKENS_PER_EVAL": "262143",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    try:
        service.submit(
            student_id="student-1",
            config={
                "model": {"num_hidden_layers": 2, "hidden_size": 128},
                "training": {
                    "train_tokens": 1024,
                    "validation_batch_size": 8,
                    "learning_rate": 3e-4,
                    "num_evals": 1,
                },
            },
            requested_runtime_seconds=300,
        )
    except ValueError as exc:
        assert "validation_tokens_per_eval" in str(exc)
        assert "validation_batch_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_service_from_env_rejects_nondivisible_batches_for_worker_gpu_count():
    service = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SCALING_WORKER_GPU_COUNT": "2",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    assert service.worker_gpu_count == 2
    try:
        service.submit(
            student_id="student-1",
            config={
                "model": {"num_hidden_layers": 2, "hidden_size": 128},
                "training": {
                    "train_tokens": 1024,
                    "learning_rate": 3e-4,
                    "num_evals": 1,
                },
            },
            requested_runtime_seconds=300,
        )
    except ValueError as exc:
        assert "training.train_batch_size" in str(exc)
        assert "worker_gpu_count=2" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_service_from_env_uses_student_active_experiment_limit_as_launch_capacity():
    service = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_TOTAL_BUDGET_SECONDS": "7200",
            "SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT": "1",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    first = service.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
        },
        requested_runtime_seconds=300,
    )
    second = service.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 2048, "learning_rate": 3e-4, "num_evals": 1},
        },
        requested_runtime_seconds=300,
    )

    assert first.status == "submitted"
    assert second.status == "queued"


def test_build_service_from_env_rejects_negative_student_active_experiment_limit():
    try:
        build_service_from_env(
            {
                "SCALING_PROVIDER": "fake",
                "SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT": "-1",
                "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
                "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            },
            now=lambda: "2026-07-16T15:00:00Z",
        )
    except RuntimeConfigError as exc:
        assert "SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT" in str(exc)
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("expected RuntimeConfigError")


def test_build_service_from_env_uses_global_active_experiment_limit_as_launch_capacity():
    service = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_TOTAL_BUDGET_SECONDS": "7200",
            "SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL": "1",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    first = service.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
        },
        requested_runtime_seconds=300,
    )
    second = service.submit(
        student_id="student-2",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 2048, "learning_rate": 3e-4, "num_evals": 1},
        },
        requested_runtime_seconds=300,
    )

    assert first.status == "submitted"
    assert second.status == "queued"


def test_build_service_from_env_rejects_negative_global_active_experiment_limit():
    try:
        build_service_from_env(
            {
                "SCALING_PROVIDER": "fake",
                "SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL": "-1",
                "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
                "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            },
            now=lambda: "2026-07-16T15:00:00Z",
        )
    except RuntimeConfigError as exc:
        assert "SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL" in str(exc)
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("expected RuntimeConfigError")


def test_build_service_from_env_can_select_slurm_provider(tmp_path):
    service = build_service_from_env(
        {
            "SCALING_PROVIDER": "slurm",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SLURM_WORK_DIR": str(tmp_path / "slurm"),
            "SLURM_PARTITION": "gpu",
            "SLURM_ACCOUNT": "course",
            "SLURM_GPUS_PER_TASK": "1",
            "SLURM_CPUS_PER_TASK": "8",
            "SLURM_MEMORY_GB": "64",
            "SLURM_TIME_LIMIT_MINUTES": "60",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    assert service.provider.provider_name == "slurm"
    assert service.provider.config.partition == "gpu"
    assert service.provider.config.account == "course"


def test_build_service_from_env_writes_manifests_when_local_dir_is_configured(tmp_path):
    manifest_dir = tmp_path / "manifests"
    service = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_TOTAL_BUDGET_SECONDS": "7200",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://placeholder",
            "SCALING_LOCAL_MANIFEST_DIR": str(manifest_dir),
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    service.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
        },
        requested_runtime_seconds=300,
    )

    assert service.provider.submitted_manifests["fake-job-000001"]["manifest_uri"] == (
        manifest_dir / "exp-000001.json"
    ).resolve().as_uri()
    assert (manifest_dir / "exp-000001.json").exists()


def test_build_service_from_env_can_publish_manifests_with_public_uris(tmp_path):
    manifest_dir = tmp_path / "published-manifests"
    service = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_TOTAL_BUDGET_SECONDS": "7200",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://placeholder",
            "SCALING_PUBLISHED_MANIFEST_DIR": str(manifest_dir),
            "SCALING_PUBLISHED_MANIFEST_BASE_URI": "https://storage.example/manifests",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    service.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
        },
        requested_runtime_seconds=300,
    )

    assert service.provider.submitted_manifests["fake-job-000001"]["manifest_uri"] == (
        "https://storage.example/manifests/exp-000001.json"
    )
    assert (manifest_dir / "exp-000001.json").exists()


def test_build_service_from_env_requires_complete_published_manifest_settings(tmp_path):
    try:
        build_service_from_env(
            {
                "SCALING_PROVIDER": "fake",
                "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
                "SCALING_MANIFEST_BASE_URI": "memory://manifests",
                "SCALING_PUBLISHED_MANIFEST_DIR": str(tmp_path / "manifests"),
            },
            now=lambda: "2026-07-16T15:00:00Z",
        )
    except RuntimeConfigError as exc:
        assert "SCALING_PUBLISHED_MANIFEST_DIR" in str(exc)
        assert "SCALING_PUBLISHED_MANIFEST_BASE_URI" in str(exc)
    else:
        raise AssertionError("expected RuntimeConfigError")


def test_build_service_from_env_loads_state_snapshot_when_configured(tmp_path):
    snapshot_path = tmp_path / "state.json"
    original = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )
    submitted = original.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
        },
        requested_runtime_seconds=300,
    )
    original.save_state_snapshot(snapshot_path)

    restored = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SCALING_STATE_SNAPSHOT_PATH": str(snapshot_path),
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    assert restored.get_result("student-1", submitted.experiment_id).status == "submitted"
    assert restored.get_budget("student-1").reserved_seconds == 300


def test_restored_fake_provider_snapshot_can_cancel_active_experiment(tmp_path):
    snapshot_path = tmp_path / "state.json"
    original = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )
    submitted = original.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
        },
        requested_runtime_seconds=300,
    )
    original.save_state_snapshot(snapshot_path)
    restored = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SCALING_STATE_SNAPSHOT_PATH": str(snapshot_path),
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    cancelled = restored.admin_cancel_experiment(
        submitted.experiment_id,
        reason="local rehearsal cleanup",
        actor="staff",
    )

    assert cancelled.status == "cancelled"
    assert restored.get_budget("student-1").reserved_seconds == 0


def test_build_app_from_env_loads_student_keys_and_admin_tokens(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text("student_id,api_key\nstudent-1,key-1\n", encoding="utf-8")
    app = build_app_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_STUDENT_KEYS_CSV": str(roster),
            "SCALING_INTERNAL_CALLBACK_TOKEN": "internal-secret",
            "SCALING_ADMIN_API_TOKEN": "admin-secret",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )
    client = TestClient(app)

    student = client.get("/budget", headers={"Authorization": "Bearer key-1"})
    admin = client.get("/admin/queue", headers={"Authorization": "Bearer admin-secret"})
    unauthorized_admin = client.get("/admin/queue", headers={"Authorization": "Bearer key-1"})

    assert student.status_code == 200
    assert student.json()["student_id"] == "student-1"
    assert admin.status_code == 200
    assert unauthorized_admin.status_code == 401


def test_build_app_from_env_configures_worker_event_import_dir(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text("student_id,api_key\nstudent-1,key-1\n", encoding="utf-8")
    worker_events = tmp_path / "worker-events"
    app = build_app_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_STUDENT_KEYS_CSV": str(roster),
            "SCALING_INTERNAL_CALLBACK_TOKEN": "internal-secret",
            "SCALING_ADMIN_API_TOKEN": "admin-secret",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SCALING_WORKER_EVENT_IMPORT_DIR": str(worker_events),
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )
    client = TestClient(app)
    client.post(
        "/submit",
        headers={"Authorization": "Bearer key-1"},
        json={
            "config": {
                "model": {"num_hidden_layers": 2, "hidden_size": 128},
                "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
            },
            "requested_runtime_seconds": 300,
        },
    )
    worker_events.mkdir()
    (worker_events / "exp-000001.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": "exp-000001",
                "event_type": "worker_completed",
                "event_id": "exp-000001:000001",
                "validation_losses": [3.2],
                "final_validation_loss": 3.2,
                "actual_runtime_seconds": 120,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    imported = client.post(
        "/admin/import-worker-events",
        headers={"Authorization": "Bearer admin-secret"},
    )

    assert imported.status_code == 200
    assert imported.json()["imported"] == 1
    assert client.get(
        "/experiment/exp-000001",
        headers={"Authorization": "Bearer key-1"},
    ).json()["status"] == "completed"


def test_build_app_from_env_autosaves_state_snapshot_after_mutating_api_calls(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text("student_id,api_key\nstudent-1,key-1\n", encoding="utf-8")
    snapshot_path = tmp_path / "state.json"
    env = {
        "SCALING_PROVIDER": "fake",
        "SCALING_STUDENT_KEYS_CSV": str(roster),
        "SCALING_INTERNAL_CALLBACK_TOKEN": "internal-secret",
        "SCALING_ADMIN_API_TOKEN": "admin-secret",
        "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
        "SCALING_MANIFEST_BASE_URI": "memory://manifests",
        "SCALING_STATE_SNAPSHOT_PATH": str(snapshot_path),
    }
    app = build_app_from_env(env, now=lambda: "2026-07-16T15:00:00Z")
    client = TestClient(app)

    submitted = client.post(
        "/submit",
        headers={"Authorization": "Bearer key-1"},
        json={
            "config": {
                "model": {"num_hidden_layers": 2, "hidden_size": 128},
                "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
            },
            "requested_runtime_seconds": 300,
        },
    )

    restored = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SCALING_STATE_SNAPSHOT_PATH": str(snapshot_path),
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    assert submitted.status_code == 200
    assert snapshot_path.exists()
    assert restored.get_result("student-1", "exp-000001").status == "submitted"
    assert restored.get_budget("student-1").reserved_seconds == 300


def test_build_app_from_env_autosaves_state_snapshot_after_worker_callback(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text("student_id,api_key\nstudent-1,key-1\n", encoding="utf-8")
    snapshot_path = tmp_path / "state.json"
    env = {
        "SCALING_PROVIDER": "fake",
        "SCALING_STUDENT_KEYS_CSV": str(roster),
        "SCALING_INTERNAL_CALLBACK_TOKEN": "internal-secret",
        "SCALING_ADMIN_API_TOKEN": "admin-secret",
        "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
        "SCALING_MANIFEST_BASE_URI": "memory://manifests",
        "SCALING_STATE_SNAPSHOT_PATH": str(snapshot_path),
    }
    app = build_app_from_env(env, now=lambda: "2026-07-16T15:00:00Z")
    client = TestClient(app)
    client.post(
        "/submit",
        headers={"Authorization": "Bearer key-1"},
        json={
            "config": {
                "model": {"num_hidden_layers": 2, "hidden_size": 128},
                "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
            },
            "requested_runtime_seconds": 300,
        },
    )

    completed = client.post(
        "/internal/provider-events",
        headers={"Authorization": "Bearer internal-secret"},
        json={
            "experiment_id": "exp-000001",
            "event_type": "worker_completed",
            "validation_losses": [3.2],
            "final_validation_loss": 3.2,
            "actual_runtime_seconds": 120,
        },
    )
    restored = build_service_from_env(
        {
            "SCALING_PROVIDER": "fake",
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SCALING_STATE_SNAPSHOT_PATH": str(snapshot_path),
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    assert completed.status_code == 200
    assert restored.get_result("student-1", "exp-000001").status == "completed"
    assert restored.get_result("student-1", "exp-000001").final_validation_loss == 3.2
    assert restored.get_budget("student-1").charged_seconds == 120


def test_build_app_from_env_requires_staff_secrets(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text("student_id,api_key\nstudent-1,key-1\n", encoding="utf-8")

    try:
        build_app_from_env(
            {
                "SCALING_PROVIDER": "fake",
                "SCALING_STUDENT_KEYS_CSV": str(roster),
                "SCALING_INTERNAL_CALLBACK_TOKEN": "internal-secret",
                "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            },
            now=lambda: "2026-07-16T15:00:00Z",
        )
    except RuntimeConfigError as exc:
        assert "SCALING_ADMIN_API_TOKEN" in str(exc)
    else:
        raise AssertionError("expected RuntimeConfigError")
