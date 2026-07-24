from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QzDistributedConfig:
    workspace_id: str
    project_id: str
    logic_compute_group_id: str
    image: str
    image_type: str = "SOURCE_PRIVATE"
    priority: int = 10
    shm_gi: int = 1200
    framework: str = "pytorch"
    auto_fault_tolerance: bool = False


@dataclass(frozen=True)
class QzResourceSpec:
    id: str
    gpu_type: str
    gpu_count: int
    cpu_count: int
    memory_gb: int


def build_worker_command(
    manifest_uri: str,
    callback_url: str,
    callback_token_env: str,
    module: str = "scaling_backend.worker.run",
    trainer_spec: str = "",
    heartbeat_interval_seconds: int = 0,
    tokenized_index_validation_mode: str = "",
    event_sink: str = "",
    event_log_path: str = "",
    callback_token_value: str = "",
    conda_env: str = "",
    conda_init: str = "",
) -> str:
    parts = [
        "python",
        "-m",
        module,
        "--manifest-uri",
        shlex.quote(manifest_uri),
        "--callback-url",
        shlex.quote(callback_url),
        "--callback-token-env",
        shlex.quote(callback_token_env),
    ]
    if event_sink:
        parts.extend(["--event-sink", shlex.quote(event_sink)])
    if event_log_path:
        parts.extend(["--event-log-path", shlex.quote(event_log_path)])
    if trainer_spec:
        parts.extend(["--trainer", shlex.quote(trainer_spec)])
    if heartbeat_interval_seconds > 0:
        parts.extend(
            ["--heartbeat-interval-seconds", shlex.quote(str(heartbeat_interval_seconds))]
        )
    if tokenized_index_validation_mode:
        parts.extend(
            [
                "--tokenized-index-validation-mode",
                shlex.quote(tokenized_index_validation_mode),
            ]
        )
    command = " ".join(parts)
    if not conda_env and not callback_token_value:
        return command

    setup_parts = []
    if callback_token_value:
        setup_parts.append(_shell_export(callback_token_env, callback_token_value))
    if conda_init:
        setup_parts.append(f"source {shlex.quote(conda_init)}")
    if conda_env:
        setup_parts.append(f"conda activate {shlex.quote(conda_env)}")
    setup_parts.append(f"exec {command}")
    return "bash -lc " + shlex.quote(" && ".join(setup_parts))


def _shell_export(name: str, value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"callback token env name is not a shell identifier: {name}")
    return f"export {name}={shlex.quote(value)}"


def build_resource_spec_price(
    spec: QzResourceSpec, logic_compute_group_id: str
) -> dict[str, Any]:
    return {
        "cpu_type": "",
        "cpu_count": int(spec.cpu_count),
        "gpu_type": spec.gpu_type,
        "gpu_count": int(spec.gpu_count),
        "memory_size_gib": int(spec.memory_gb),
        "logic_compute_group_id": logic_compute_group_id,
        "quota_id": spec.id,
    }


def build_create_payload(
    job_name: str,
    command: str,
    config: QzDistributedConfig,
    spec: QzResourceSpec,
    instance_count: int = 1,
) -> dict[str, Any]:
    resource_spec_price = build_resource_spec_price(
        spec, config.logic_compute_group_id
    )
    return {
        "name": job_name,
        "logic_compute_group_id": config.logic_compute_group_id,
        "project_id": config.project_id,
        "workspace_id": config.workspace_id,
        "framework": config.framework,
        "command": command,
        "task_priority": int(config.priority),
        "auto_fault_tolerance": bool(config.auto_fault_tolerance),
        "framework_config": [
            {
                "cpu": int(spec.cpu_count),
                "gpu_count": int(spec.gpu_count),
                "mem_gi": int(spec.memory_gb),
                "resource_spec_price": resource_spec_price,
                "image": config.image,
                "image_type": config.image_type,
                "instance_count": int(instance_count),
                "shm_gi": int(config.shm_gi),
            }
        ],
    }
