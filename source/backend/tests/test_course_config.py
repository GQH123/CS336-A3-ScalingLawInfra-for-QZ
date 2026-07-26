import json

import pytest

from scaling_backend.course_config import (
    CourseConfigError,
    load_api_config_env,
    load_course_config_env,
    resolve_runtime_env,
)
from scaling_backend.runtime import build_service_from_env


def _api_config(tmp_path, **overrides):
    config = {
        "schema_version": 1,
        "api": {
            "provider": "fake",
            "student_keys_csv": str(tmp_path / "student_keys.csv"),
            "internal_callback_token": "internal-token",
            "admin_api_token": "admin-token",
            "callback_url": "https://backend.example/internal/provider-events",
            "manifest_base_uri": "memory://manifests",
            "local_manifest_dir": str(tmp_path / "manifests"),
            "worker_event_import_dir": str(tmp_path / "worker-events"),
            "state_snapshot_path": str(tmp_path / "state" / "snapshot.json"),
            "total_budget_seconds": 43200,
            "max_active_experiments_per_student": 1,
            "max_active_experiments_global": 8,
            "worker_gpu_count": 1,
            "code_version": "course-backend-v0",
        },
        "worker_manifest_defaults": {
            "data_manifest_id": "exploratory-train-v0",
            "eval_manifest_id": "exploratory-eval-v0",
            "final_data_manifest_id": "final-train-v0",
            "final_eval_manifest_id": "final-eval-hidden-v0",
            "validation_tokens_per_eval": 262144,
            "final_validation_tokens_per_eval": 524288,
            "tokenized_train_index_uri": (
                "file:///mnt/course-data/tokenized/train/index.json"
            ),
            "tokenized_validation_index_uri": (
                "file:///mnt/course-data/tokenized/validation/index.json"
            ),
            "final_tokenized_train_index_uri": (
                "file:///mnt/course-data/tokenized/train/index.json"
            ),
            "final_tokenized_validation_index_uri": (
                "file:///mnt/course-data/tokenized/final-validation/index.json"
            ),
        },
        "qz": {
            "api_base_url": "https://qz.sii.edu.cn",
            "username": "253108120093",
            "password_encrypted": "a" * 256,
            "cookie_file_path": str(tmp_path / "qz.cookie"),
            "session_heartbeat_interval_seconds": 900,
            "workspace_id": "workspace-1",
            "project_id": "project-1",
            "compute_group_id": "compute-group-1",
            "spec_id": "spec-1",
            "spec_gpu_type": "H200",
            "spec_gpu_count": 1,
            "spec_cpu_count": 15,
            "spec_memory_gb": 200,
            "image": "registry.example.com/course/scaling-worker:latest",
            "image_type": "SOURCE_PRIVATE",
            "priority": 10,
            "shm_gi": 1200,
            "worker_trainer": "course_trainer.worker:train",
            "worker_heartbeat_interval_seconds": 60,
            "worker_conda_env": "fnlps_a3_compute",
            "worker_conda_init": "/opt/anaconda3/etc/profile.d/conda.sh",
            "worker_event_log_dir": "/shared/course/worker-events",
            "worker_tokenized_index_validation_mode": "metadata",
        },
    }
    config.update(overrides)
    return config


def test_load_api_config_env_maps_control_node_schema_to_runtime_settings(tmp_path):
    path = tmp_path / "api-runtime.json"
    path.write_text(json.dumps(_api_config(tmp_path)), encoding="utf-8")

    values = load_api_config_env(path)

    assert values["SCALING_PROVIDER"] == "fake"
    assert values["SCALING_CALLBACK_URL"] == (
        "https://backend.example/internal/provider-events"
    )
    assert values["SCALING_DATA_MANIFEST_ID"] == "exploratory-train-v0"
    assert values["SCALING_EVAL_MANIFEST_ID"] == "exploratory-eval-v0"
    assert values["SCALING_FINAL_DATA_MANIFEST_ID"] == "final-train-v0"
    assert values["SCALING_FINAL_EVAL_MANIFEST_ID"] == "final-eval-hidden-v0"
    assert values["SCALING_TOKENIZED_TRAIN_INDEX_URI"] == (
        "file:///mnt/course-data/tokenized/train/index.json"
    )
    assert values["SCALING_TOKENIZED_VALIDATION_INDEX_URI"] == (
        "file:///mnt/course-data/tokenized/validation/index.json"
    )
    assert values["SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI"] == (
        "file:///mnt/course-data/tokenized/train/index.json"
    )
    assert values["SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI"] == (
        "file:///mnt/course-data/tokenized/final-validation/index.json"
    )
    assert values["SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE"] == "metadata"
    assert values["SCALING_WORKER_GPU_COUNT"] == "1"
    assert values["QZ_WORKER_TRAINER"] == "course_trainer.worker:train"
    assert values["QZ_WORKER_HEARTBEAT_INTERVAL_SECONDS"] == "60"
    assert values["QZ_WORKER_CONDA_ENV"] == "fnlps_a3_compute"
    assert values["QZ_WORKER_CONDA_INIT"] == "/opt/anaconda3/etc/profile.d/conda.sh"
    assert values["SCALING_WORKER_EVENT_IMPORT_DIR"].endswith("worker-events")
    assert values["QZ_WORKER_EVENT_LOG_DIR"] == "/shared/course/worker-events"
    assert values["QZ_COOKIE_FILE"].endswith("qz.cookie")
    assert values["QZ_SESSION_HEARTBEAT_INTERVAL_SECONDS"] == "900"
    assert values["QZ_IMAGE"] == "registry.example.com/course/scaling-worker:latest"


def test_resolve_runtime_env_prefers_api_config_and_allows_explicit_env_override(tmp_path):
    api_path = tmp_path / "api-runtime.json"
    api_path.write_text(json.dumps(_api_config(tmp_path)), encoding="utf-8")
    compatibility_path = tmp_path / "legacy-course-runtime.json"
    compatibility_path.write_text(
        json.dumps(
            _api_config(
                tmp_path,
                api={
                    "provider": "fake",
                    "student_keys_csv": str(tmp_path / "wrong.csv"),
                    "internal_callback_token": "wrong-internal",
                    "admin_api_token": "wrong-admin",
                    "callback_url": "https://wrong.example/internal/provider-events",
                    "manifest_base_uri": "memory://wrong",
                },
            )
        ),
        encoding="utf-8",
    )

    values = resolve_runtime_env(
        {
            "SCALING_API_CONFIG": str(api_path),
            "SCALING_COURSE_CONFIG": str(compatibility_path),
            "SCALING_PROVIDER": "fake",
            "SCALING_TOKENIZED_TRAIN_INDEX_URI": (
                "file:///mnt/course-data/debug/train/index.json"
            ),
            "QZ_IMAGE": "registry.example.com/course/debug-worker:latest",
        }
    )

    assert values["SCALING_PROVIDER"] == "fake"
    assert values["SCALING_TOKENIZED_TRAIN_INDEX_URI"] == (
        "file:///mnt/course-data/debug/train/index.json"
    )
    assert values["QZ_IMAGE"] == "registry.example.com/course/debug-worker:latest"
    assert values["SCALING_TOKENIZED_VALIDATION_INDEX_URI"] == (
        "file:///mnt/course-data/tokenized/validation/index.json"
    )
    assert values["SCALING_CALLBACK_URL"] == (
        "https://backend.example/internal/provider-events"
    )


def test_api_config_rejects_unknown_fields_that_would_hide_typos(tmp_path):
    config = _api_config(tmp_path)
    config["api"]["callback"] = "https://wrong-key.example"
    path = tmp_path / "api-runtime.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(CourseConfigError, match="api.callback"):
        load_api_config_env(path)


def test_api_config_rejects_compute_node_data_root_in_control_node_config(tmp_path):
    config = _api_config(tmp_path)
    config["worker_manifest_defaults"]["tokenized_root"] = "/mnt/course-data/tokenized"
    path = tmp_path / "api-runtime.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(CourseConfigError, match="worker_manifest_defaults.tokenized_root"):
        load_api_config_env(path)


def test_legacy_course_config_loader_remains_available_for_existing_scripts(tmp_path):
    path = tmp_path / "api-runtime.json"
    path.write_text(json.dumps(_api_config(tmp_path)), encoding="utf-8")

    values = load_course_config_env(path)

    assert values["SCALING_PROVIDER"] == "fake"
    assert values["SCALING_CALLBACK_URL"] == (
        "https://backend.example/internal/provider-events"
    )


def test_build_service_from_env_loads_api_config_file(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text("student_id,api_key\nstudent-1,key-1\n", encoding="utf-8")
    path = tmp_path / "api-runtime.json"
    path.write_text(json.dumps(_api_config(tmp_path)), encoding="utf-8")

    service = build_service_from_env(
        {"SCALING_API_CONFIG": str(path)},
        now=lambda: "2026-07-17T10:00:00Z",
    )

    service.submit(
        student_id="student-1",
        config={
            "model": {"num_hidden_layers": 2, "hidden_size": 128},
            "training": {"train_tokens": 1024, "learning_rate": 3e-4, "num_evals": 1},
        },
        requested_runtime_seconds=300,
    )

    manifest = service.provider.submitted_manifests["fake-job-000001"]
    assert manifest["data_config"]["tokenized_index_uri"] == (
        "file:///mnt/course-data/tokenized/train/index.json"
    )
    assert manifest["validation_config"]["tokenized_index_uri"] == (
        "file:///mnt/course-data/tokenized/validation/index.json"
    )
