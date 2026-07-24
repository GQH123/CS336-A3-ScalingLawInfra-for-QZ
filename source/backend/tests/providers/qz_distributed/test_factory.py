import pytest

from scaling_backend.providers.qz_distributed.factory import (
    QzProviderConfigError,
    build_qz_distributed_adapter_from_env,
)
from scaling_backend.providers.qz_distributed.payloads import QzDistributedConfig


def _valid_env():
    return {
        "QZ_API_BASE_URL": "https://qz.sii.edu.cn",
        "QZ_USERNAME": "253108120093",
        "QZ_PASSWORD_ENCRYPTED": "a" * 256,
        "QZ_COOKIE": "session=abc",
        "QZ_WORKSPACE_ID": "ws-1",
        "QZ_PROJECT_ID": "project-1",
        "QZ_COMPUTE_GROUP_ID": "lcg-1",
        "QZ_SPEC_ID": "spec-1",
        "QZ_SPEC_GPU_TYPE": "H200",
        "QZ_SPEC_GPU_COUNT": "1",
        "QZ_SPEC_CPU_COUNT": "15",
        "QZ_SPEC_MEMORY_GB": "200",
        "QZ_IMAGE": "registry.example.com/scaling-worker:latest",
        "QZ_IMAGE_TYPE": "SOURCE_PRIVATE",
        "QZ_PRIORITY": "7",
        "QZ_SHM_GI": "64",
        "QZ_CALLBACK_TOKEN_ENV": "SCALING_CALLBACK_TOKEN",
        "QZ_WORKER_TRAINER": "course_trainer.worker:train",
        "QZ_WORKER_HEARTBEAT_INTERVAL_SECONDS": "60",
        "QZ_WORKER_CONDA_ENV": "fnlps_a3_compute",
        "QZ_WORKER_CONDA_INIT": "/opt/anaconda3/etc/profile.d/conda.sh",
        "QZ_WORKER_EVENT_LOG_DIR": "/shared/course/worker-events",
        "SCALING_INTERNAL_CALLBACK_TOKEN": "internal-callback-token",
        "SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE": "metadata",
    }


def test_build_adapter_from_env_parses_staff_resource_preset():
    adapter = build_qz_distributed_adapter_from_env(
        _valid_env(),
        now=lambda: "2026-07-16T15:00:00Z",
    )

    assert adapter.provider_name == "qz_distributed"
    assert adapter.config == QzDistributedConfig(
        workspace_id="ws-1",
        project_id="project-1",
        logic_compute_group_id="lcg-1",
        image="registry.example.com/scaling-worker:latest",
        image_type="SOURCE_PRIVATE",
        priority=7,
        shm_gi=64,
        framework="pytorch",
    )
    assert adapter.spec.id == "spec-1"
    assert adapter.spec.gpu_type == "H200"
    assert adapter.spec.gpu_count == 1
    assert adapter.spec.cpu_count == 15
    assert adapter.spec.memory_gb == 200
    assert adapter.trainer_spec == "course_trainer.worker:train"
    assert adapter.heartbeat_interval_seconds == 60
    assert adapter.tokenized_index_validation_mode == "metadata"
    assert adapter.worker_conda_env == "fnlps_a3_compute"
    assert adapter.worker_conda_init == "/opt/anaconda3/etc/profile.d/conda.sh"
    assert adapter.worker_event_log_dir == "/shared/course/worker-events"
    assert adapter.callback_token_value == "internal-callback-token"


def test_build_adapter_from_env_reports_missing_required_keys_together():
    with pytest.raises(QzProviderConfigError) as exc_info:
        build_qz_distributed_adapter_from_env({}, now=lambda: "now")

    message = str(exc_info.value)
    assert "QZ_COOKIE" in message
    assert "QZ_WORKSPACE_ID" in message
    assert "QZ_IMAGE" in message


def test_build_adapter_from_env_accepts_cookie_without_cas_credentials():
    env = _valid_env()
    del env["QZ_USERNAME"]
    del env["QZ_PASSWORD_ENCRYPTED"]
    env["QZ_COOKIE"] = "inspire-session=manual"

    adapter = build_qz_distributed_adapter_from_env(env, now=lambda: "now")

    assert adapter.client.cookie == "inspire-session=manual"
    assert adapter.client.config.username == ""
    assert adapter.client.config.password == ""


@pytest.mark.parametrize(
    "key,value",
    [
        ("QZ_SPEC_GPU_COUNT", "0"),
        ("QZ_SPEC_CPU_COUNT", "0"),
        ("QZ_SPEC_MEMORY_GB", "0"),
        ("QZ_PRIORITY", "0"),
        ("QZ_SHM_GI", "0"),
        ("QZ_REQUEST_TIMEOUT_SECONDS", "0"),
        ("QZ_LOGIN_TIMEOUT_SECONDS", "0"),
        ("QZ_LOGIN_MAX_TRIES", "0"),
    ],
)
def test_build_adapter_from_env_rejects_non_positive_numeric_settings(key, value):
    env = _valid_env()
    env[key] = value

    with pytest.raises(QzProviderConfigError) as exc_info:
        build_qz_distributed_adapter_from_env(
            env,
            now=lambda: "2026-07-16T15:00:00Z",
        )

    assert key in str(exc_info.value)
    assert "positive integer" in str(exc_info.value)
