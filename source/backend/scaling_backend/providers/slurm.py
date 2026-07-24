from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scaling_backend.providers.contracts import (
    ProviderArtifacts,
    ProviderCancelResult,
    ProviderJobStatus,
    ProviderStatusSnapshot,
    ProviderSubmission,
)


class SlurmProviderConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SlurmConfig:
    work_dir: Path
    partition: str = ""
    account: str = ""
    gpus_per_task: int = 1
    cpus_per_task: int = 8
    memory_gb: int = 64
    time_limit_minutes: int = 60
    callback_token_env: str = "SCALING_CALLBACK_TOKEN"
    trainer_spec: str = ""
    heartbeat_interval_seconds: int = 0
    python_executable: str = "python"
    worker_module: str = "scaling_backend.worker.run"


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class SlurmProviderAdapter:
    provider_name = "slurm"

    def __init__(
        self,
        *,
        config: SlurmConfig,
        runner: CommandRunner | None = None,
        now: Callable[[], str],
    ):
        self.config = config
        self._runner = runner or _run_command
        self._now = now
        self._jobs: dict[str, dict[str, str]] = {}

    def submit(self, manifest: Mapping[str, Any]) -> ProviderSubmission:
        run_id = _manifest_run_id(manifest)
        manifest_uri = _require_string(manifest, "manifest_uri")
        callback_url = _require_string(manifest, "callback_url")
        job_name = _job_name(run_id)
        script_path = self._script_path(job_name)
        stdout_path = self._stdout_path(job_name)
        stderr_path = self._stderr_path(job_name)
        self._write_script(
            script_path=script_path,
            job_name=job_name,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            manifest_uri=manifest_uri,
            callback_url=callback_url,
        )

        result = self._run(["sbatch", "--parsable", str(script_path)])
        provider_job_id = _parse_sbatch_job_id(result.stdout)
        self._jobs[provider_job_id] = {
            "job_name": job_name,
            "script_path": str(script_path),
            "stdout_path": str(self._stdout_path(job_name, provider_job_id)),
            "stderr_path": str(self._stderr_path(job_name, provider_job_id)),
        }
        return ProviderSubmission(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            submitted_at=self._now(),
            metadata={
                "script_path": str(script_path),
                "stdout_path": str(self._stdout_path(job_name, provider_job_id)),
                "stderr_path": str(self._stderr_path(job_name, provider_job_id)),
            },
        )

    def get_status(self, provider_job_id: str) -> ProviderStatusSnapshot:
        squeue = self._run(["squeue", "-h", "-j", provider_job_id, "-o", "%T"])
        raw_status = _first_state(squeue.stdout)
        if not raw_status:
            sacct = self._run(
                ["sacct", "-n", "-j", provider_job_id, "--format=State", "--parsable2"]
            )
            raw_status = _first_state(sacct.stdout)
        return ProviderStatusSnapshot(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            status=normalize_slurm_status(raw_status),
            raw_status=raw_status,
        )

    def cancel(
        self, provider_job_id: str, reason: str, actor: str
    ) -> ProviderCancelResult:
        result = self._run(["scancel", provider_job_id])
        accepted = result.returncode == 0
        return ProviderCancelResult(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            accepted=accepted,
            status=ProviderJobStatus.CANCELLED if accepted else ProviderJobStatus.UNKNOWN,
            message=result.stderr.strip() or result.stdout.strip(),
            metadata={"reason": reason, "actor": actor},
        )

    def get_artifacts(self, provider_job_id: str) -> ProviderArtifacts:
        metadata = self._jobs.get(provider_job_id) or {
            "stdout_path": str(self.config.work_dir / "logs" / f"{provider_job_id}.out"),
            "stderr_path": str(self.config.work_dir / "logs" / f"{provider_job_id}.err"),
            "script_path": str(self.config.work_dir / "scripts" / f"{provider_job_id}.sbatch"),
        }
        return ProviderArtifacts(
            provider_name=self.provider_name,
            provider_job_id=provider_job_id,
            logs_uri=Path(metadata["stdout_path"]).resolve().as_uri(),
            detail_uri=Path(metadata["script_path"]).resolve().as_uri(),
            metadata={
                "stderr_uri": Path(metadata["stderr_path"]).resolve().as_uri(),
            },
        )

    def _write_script(
        self,
        *,
        script_path: Path,
        job_name: str,
        stdout_path: Path,
        stderr_path: Path,
        manifest_uri: str,
        callback_url: str,
    ) -> None:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={stdout_path}",
            f"#SBATCH --error={stderr_path}",
            f"#SBATCH --gres=gpu:{self.config.gpus_per_task}",
            f"#SBATCH --cpus-per-task={self.config.cpus_per_task}",
            f"#SBATCH --mem={self.config.memory_gb}G",
            f"#SBATCH --time={_format_time_limit(self.config.time_limit_minutes)}",
        ]
        if self.config.partition:
            lines.append(f"#SBATCH --partition={self.config.partition}")
        if self.config.account:
            lines.append(f"#SBATCH --account={self.config.account}")
        command = build_worker_command(
            python_executable=self.config.python_executable,
            worker_module=self.config.worker_module,
            manifest_uri=manifest_uri,
            callback_url=callback_url,
            callback_token_env=self.config.callback_token_env,
            trainer_spec=self.config.trainer_spec,
            heartbeat_interval_seconds=self.config.heartbeat_interval_seconds,
        )
        lines.extend(["", command, ""])
        script_path.write_text("\n".join(lines), encoding="utf-8")

    def _script_path(self, job_name: str) -> Path:
        return self.config.work_dir / "scripts" / f"{job_name}.sbatch"

    def _stdout_path(self, job_name: str, provider_job_id: str = "%j") -> Path:
        return self.config.work_dir / "logs" / f"{job_name}-{provider_job_id}.out"

    def _stderr_path(self, job_name: str, provider_job_id: str = "%j") -> Path:
        return self.config.work_dir / "logs" / f"{job_name}-{provider_job_id}.err"

    def _run(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        result = self._runner(args)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result


def build_slurm_adapter_from_env(
    env: Mapping[str, str] | None = None,
    *,
    now: Callable[[], str] | None = None,
) -> SlurmProviderAdapter:
    values = env or os.environ
    config = SlurmConfig(
        work_dir=Path(_get(values, "SLURM_WORK_DIR", "/tmp/scaling-slurm")),
        partition=_get(values, "SLURM_PARTITION", ""),
        account=_get(values, "SLURM_ACCOUNT", ""),
        gpus_per_task=_get_positive_int(values, "SLURM_GPUS_PER_TASK", 1),
        cpus_per_task=_get_positive_int(values, "SLURM_CPUS_PER_TASK", 8),
        memory_gb=_get_positive_int(values, "SLURM_MEMORY_GB", 64),
        time_limit_minutes=_get_positive_int(values, "SLURM_TIME_LIMIT_MINUTES", 60),
        callback_token_env=_get(values, "SLURM_CALLBACK_TOKEN_ENV", "SCALING_CALLBACK_TOKEN"),
        trainer_spec=_get(values, "SLURM_WORKER_TRAINER", ""),
        heartbeat_interval_seconds=_get_non_negative_int(
            values, "SLURM_WORKER_HEARTBEAT_INTERVAL_SECONDS", 0
        ),
        python_executable=_get(values, "SLURM_PYTHON", "python"),
        worker_module=_get(values, "SLURM_WORKER_MODULE", "scaling_backend.worker.run"),
    )
    return SlurmProviderAdapter(
        config=config,
        now=now or _utc_now_iso,
    )


def build_worker_command(
    *,
    python_executable: str,
    worker_module: str,
    manifest_uri: str,
    callback_url: str,
    callback_token_env: str,
    trainer_spec: str = "",
    heartbeat_interval_seconds: int = 0,
) -> str:
    parts = [
        python_executable,
        "-m",
        worker_module,
        "--manifest-uri",
        manifest_uri,
        "--callback-url",
        callback_url,
        "--callback-token-env",
        callback_token_env,
    ]
    if trainer_spec:
        parts.extend(["--trainer", trainer_spec])
    if heartbeat_interval_seconds > 0:
        parts.extend(["--heartbeat-interval-seconds", str(heartbeat_interval_seconds)])
    return " ".join(shlex.quote(part) for part in parts)


def normalize_slurm_status(raw_status: str) -> ProviderJobStatus:
    status = raw_status.strip().upper().split()[0] if raw_status else ""
    status = status.split("|", 1)[0].split("+", 1)[0]
    if status in {"PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"}:
        return ProviderJobStatus.QUEUED
    if status in {"RUNNING", "RESIZING"}:
        return ProviderJobStatus.RUNNING
    if status in {"COMPLETED"}:
        return ProviderJobStatus.SUCCEEDED
    if status in {"CANCELLED", "REVOKED"}:
        return ProviderJobStatus.CANCELLED
    if status in {
        "BOOT_FAIL",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }:
        return ProviderJobStatus.FAILED
    if not status:
        return ProviderJobStatus.UNKNOWN
    return ProviderJobStatus.UNKNOWN


def _run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _manifest_run_id(manifest: Mapping[str, Any]) -> str:
    if manifest.get("run_kind") == "final":
        return _require_string(manifest, "final_run_id")
    return _require_string(manifest, "experiment_id")


def _require_string(manifest: Mapping[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest missing required field: {key}")
    return value


def _job_name(run_id: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in run_id
    ).strip("-")
    return f"scale-{safe or 'run'}"


def _parse_sbatch_job_id(stdout: str) -> str:
    text = stdout.strip()
    match = re.search(r"(\d+)", text)
    if not match:
        raise ValueError("sbatch output did not include a Slurm job ID")
    return match.group(1)


def _first_state(stdout: str) -> str:
    for line in stdout.splitlines():
        value = line.strip()
        if value:
            return value
    return ""


def _format_time_limit(minutes: int) -> str:
    hours, remaining_minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{remaining_minutes:02d}:00"


def _get(values: Mapping[str, str], key: str, default: str) -> str:
    value = str(values.get(key, "")).strip()
    return value if value else default


def _get_positive_int(values: Mapping[str, str], key: str, default: int) -> int:
    raw = str(values.get(key, "")).strip()
    if not raw:
        parsed = default
    else:
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise SlurmProviderConfigError(
                f"Slurm provider setting {key} must be an integer"
            ) from exc
    if parsed <= 0:
        raise SlurmProviderConfigError(
            f"Slurm provider setting {key} must be a positive integer"
        )
    return parsed


def _get_non_negative_int(values: Mapping[str, str], key: str, default: int) -> int:
    raw = str(values.get(key, "")).strip()
    if not raw:
        parsed = default
    else:
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise SlurmProviderConfigError(
                f"Slurm provider setting {key} must be an integer"
            ) from exc
    if parsed < 0:
        raise SlurmProviderConfigError(
            f"Slurm provider setting {key} must be a non-negative integer"
        )
    return parsed


def _utc_now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()
