import shlex

from scaling_backend.providers.qz_distributed.payloads import (
    QzDistributedConfig,
    QzResourceSpec,
    build_create_payload,
    build_worker_command,
)


def _config() -> QzDistributedConfig:
    return QzDistributedConfig(
        workspace_id="ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
        project_id="project-c67c548f-f02c-453b-ba5b-8745db6886e7",
        logic_compute_group_id="lcg-a91ad10b-415d-4abd-8170-828a2feae5d2",
        image="docker.sii.shaipower.online/course/scaling-worker:2026-07-16",
        image_type="SOURCE_PRIVATE",
        priority=10,
        shm_gi=1200,
        framework="pytorch",
    )


def _spec() -> QzResourceSpec:
    return QzResourceSpec(
        id="4dd0e854-e2a4-4253-95e6-64c13f0b5117",
        gpu_type="H200",
        gpu_count=1,
        cpu_count=15,
        memory_gb=200,
    )


def test_build_worker_command_quotes_manifest_and_callback_values():
    command = build_worker_command(
        manifest_uri="s3://bucket/manifests/exp 1.json",
        callback_url="https://backend.internal/callback?x=1",
        callback_token_env="SCALING_CALLBACK_TOKEN",
    )

    assert command == (
        "python -m scaling_backend.worker.run "
        "--manifest-uri 's3://bucket/manifests/exp 1.json' "
        "--callback-url 'https://backend.internal/callback?x=1' "
        "--callback-token-env SCALING_CALLBACK_TOKEN"
    )


def test_build_worker_command_can_include_trainer_spec():
    command = build_worker_command(
        manifest_uri="s3://bucket/manifests/exp-1.json",
        callback_url="https://backend.internal/callback",
        callback_token_env="SCALING_CALLBACK_TOKEN",
        trainer_spec="course_trainer.worker:train",
    )

    assert command.endswith("--trainer course_trainer.worker:train")


def test_build_worker_command_can_include_heartbeat_interval():
    command = build_worker_command(
        manifest_uri="s3://bucket/manifests/exp-1.json",
        callback_url="https://backend.internal/callback",
        callback_token_env="SCALING_CALLBACK_TOKEN",
        heartbeat_interval_seconds=60,
    )

    assert command.endswith("--heartbeat-interval-seconds 60")


def test_build_worker_command_can_include_tokenized_index_validation_mode():
    command = build_worker_command(
        manifest_uri="s3://bucket/manifests/exp-1.json",
        callback_url="https://backend.internal/callback",
        callback_token_env="SCALING_CALLBACK_TOKEN",
        tokenized_index_validation_mode="metadata",
    )

    assert command.endswith("--tokenized-index-validation-mode metadata")


def test_build_worker_command_can_select_jsonl_event_sink():
    command = build_worker_command(
        manifest_uri="file:///shared/manifests/exp-1.json",
        callback_url="https://api.internal/internal/provider-events",
        callback_token_env="SCALING_CALLBACK_TOKEN",
        event_sink="jsonl",
        event_log_path="/shared/worker-events/exp-1.jsonl",
    )

    assert "--event-sink jsonl" in command
    assert "--event-log-path /shared/worker-events/exp-1.jsonl" in command


def test_build_worker_command_can_export_callback_token():
    command = build_worker_command(
        manifest_uri="file:///manifests/exp-1.json",
        callback_url="https://backend.internal/callback",
        callback_token_env="SCALING_CALLBACK_TOKEN",
        callback_token_value="secret token with spaces",
    )

    assert command.startswith("bash -lc ")
    inner = shlex.split(command)[2]
    assert "export SCALING_CALLBACK_TOKEN='secret token with spaces'" in inner
    assert "exec python -m scaling_backend.worker.run" in inner


def test_build_worker_command_can_activate_conda_environment():
    command = build_worker_command(
        manifest_uri="file:///manifests/exp-1.json",
        callback_url="https://backend.internal/callback",
        callback_token_env="SCALING_CALLBACK_TOKEN",
        trainer_spec="course_trainer.worker:train",
        heartbeat_interval_seconds=60,
        tokenized_index_validation_mode="metadata",
        callback_token_value="secret-token",
        conda_env="fnlps_a3_compute",
        conda_init="/opt/anaconda3/etc/profile.d/conda.sh",
    )

    assert command.startswith("bash -lc ")
    assert "source /opt/anaconda3/etc/profile.d/conda.sh" in command
    assert "conda activate fnlps_a3_compute" in command
    assert "export SCALING_CALLBACK_TOKEN=secret-token" in command
    assert "exec python -m scaling_backend.worker.run" in command
    assert "--trainer course_trainer.worker:train" in command


def test_build_create_payload_matches_qz_distributed_training_shape():
    payload = build_create_payload(
        job_name="scale-exp-0001",
        command="python -m scaling_backend.worker.run --manifest-uri manifest.json",
        config=_config(),
        spec=_spec(),
        instance_count=1,
    )

    assert payload == {
        "name": "scale-exp-0001",
        "logic_compute_group_id": "lcg-a91ad10b-415d-4abd-8170-828a2feae5d2",
        "project_id": "project-c67c548f-f02c-453b-ba5b-8745db6886e7",
        "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
        "framework": "pytorch",
        "command": "python -m scaling_backend.worker.run --manifest-uri manifest.json",
        "task_priority": 10,
        "auto_fault_tolerance": False,
        "framework_config": [
            {
                "cpu": 15,
                "gpu_count": 1,
                "mem_gi": 200,
                "resource_spec_price": {
                    "cpu_type": "",
                    "cpu_count": 15,
                    "gpu_type": "H200",
                    "gpu_count": 1,
                    "memory_size_gib": 200,
                    "logic_compute_group_id": "lcg-a91ad10b-415d-4abd-8170-828a2feae5d2",
                    "quota_id": "4dd0e854-e2a4-4253-95e6-64c13f0b5117",
                },
                "image": "docker.sii.shaipower.online/course/scaling-worker:2026-07-16",
                "image_type": "SOURCE_PRIVATE",
                "instance_count": 1,
                "shm_gi": 1200,
            }
        ],
    }

    serialized = str(payload)
    assert "spec_id" not in serialized
