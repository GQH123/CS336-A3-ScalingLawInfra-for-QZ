import subprocess

import pytest

from scaling_backend.providers.contracts import ProviderJobStatus
from scaling_backend.providers.slurm import (
    SlurmConfig,
    SlurmProviderConfigError,
    SlurmProviderAdapter,
    build_slurm_adapter_from_env,
)


class FakeRunner:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, args):
        self.calls.append(list(args))
        if not self.results:
            raise AssertionError(f"unexpected command: {args}")
        return self.results.pop(0)


def _result(args, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _manifest():
    return {
        "manifest_version": 1,
        "run_kind": "exploratory",
        "experiment_id": "exp-000001",
        "student_id": "student-1",
        "manifest_uri": "https://storage.example/manifests/exp-000001.json",
        "callback_url": "https://backend.internal/internal/provider-events",
        "max_runtime_seconds": 5400,
    }


def _adapter(tmp_path, runner):
    return SlurmProviderAdapter(
        config=SlurmConfig(
            work_dir=tmp_path,
            partition="gpu",
            account="course",
            gpus_per_task=1,
            cpus_per_task=16,
            memory_gb=128,
            time_limit_minutes=90,
            callback_token_env="SCALING_CALLBACK_TOKEN",
            trainer_spec="course_trainer.worker:train",
            heartbeat_interval_seconds=45,
        ),
        runner=runner,
        now=lambda: "2026-07-16T15:00:00Z",
    )


def test_submit_writes_sbatch_script_and_parses_job_id(tmp_path):
    runner = FakeRunner(_result(["sbatch"], stdout="12345\n"))
    adapter = _adapter(tmp_path, runner)

    submission = adapter.submit(_manifest())

    assert submission.provider_name == "slurm"
    assert submission.provider_job_id == "12345"
    assert submission.submitted_at == "2026-07-16T15:00:00Z"
    script_path = tmp_path / "scripts" / "scale-exp-000001.sbatch"
    assert runner.calls == [["sbatch", "--parsable", str(script_path)]]
    assert submission.metadata["script_path"] == str(script_path)

    script = script_path.read_text(encoding="utf-8")
    assert "#SBATCH --job-name=scale-exp-000001" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --account=course" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=16" in script
    assert "#SBATCH --mem=128G" in script
    assert "#SBATCH --time=01:30:00" in script
    assert "--manifest-uri https://storage.example/manifests/exp-000001.json" in script
    assert "--callback-url https://backend.internal/internal/provider-events" in script
    assert "--callback-token-env SCALING_CALLBACK_TOKEN" in script
    assert "--trainer course_trainer.worker:train" in script
    assert "--heartbeat-interval-seconds 45" in script


def test_submit_supports_final_run_manifests(tmp_path):
    runner = FakeRunner(_result(["sbatch"], stdout="Submitted batch job 987\n"))
    adapter = _adapter(tmp_path, runner)

    submission = adapter.submit(
        {
            **_manifest(),
            "run_kind": "final",
            "experiment_id": "",
            "final_run_id": "final-run-000001",
            "manifest_uri": "https://storage.example/manifests/final-run-000001.json",
        }
    )

    assert submission.provider_job_id == "987"
    assert runner.calls[0][2].endswith("scale-final-run-000001.sbatch")


def test_get_status_prefers_squeue_and_normalizes_running(tmp_path):
    runner = FakeRunner(_result(["squeue"], stdout="RUNNING\n"))
    adapter = _adapter(tmp_path, runner)

    snapshot = adapter.get_status("12345")

    assert snapshot.status is ProviderJobStatus.RUNNING
    assert snapshot.raw_status == "RUNNING"
    assert runner.calls == [["squeue", "-h", "-j", "12345", "-o", "%T"]]


def test_get_status_falls_back_to_sacct_for_terminal_jobs(tmp_path):
    runner = FakeRunner(
        _result(["squeue"], stdout=""),
        _result(["sacct"], stdout="COMPLETED\n"),
    )
    adapter = _adapter(tmp_path, runner)

    snapshot = adapter.get_status("12345")

    assert snapshot.status is ProviderJobStatus.SUCCEEDED
    assert snapshot.raw_status == "COMPLETED"
    assert runner.calls == [
        ["squeue", "-h", "-j", "12345", "-o", "%T"],
        ["sacct", "-n", "-j", "12345", "--format=State", "--parsable2"],
    ]


def test_cancel_invokes_scancel_and_returns_status(tmp_path):
    runner = FakeRunner(_result(["scancel"], stdout="cancelled\n"))
    adapter = _adapter(tmp_path, runner)

    result = adapter.cancel("12345", reason="staff cleanup", actor="admin")

    assert runner.calls == [["scancel", "12345"]]
    assert result.accepted is True
    assert result.status is ProviderJobStatus.CANCELLED
    assert result.metadata == {"reason": "staff cleanup", "actor": "admin"}


def test_get_artifacts_returns_local_script_and_log_pointers(tmp_path):
    runner = FakeRunner(_result(["sbatch"], stdout="12345\n"))
    adapter = _adapter(tmp_path, runner)
    adapter.submit(_manifest())

    artifacts = adapter.get_artifacts("12345")

    assert artifacts.logs_uri == (tmp_path / "logs" / "scale-exp-000001-12345.out").resolve().as_uri()
    assert artifacts.detail_uri == (tmp_path / "scripts" / "scale-exp-000001.sbatch").resolve().as_uri()
    assert artifacts.metadata["stderr_uri"] == (
        tmp_path / "logs" / "scale-exp-000001-12345.err"
    ).resolve().as_uri()


def test_build_slurm_adapter_from_env_parses_staff_settings(tmp_path):
    adapter = build_slurm_adapter_from_env(
        {
            "SLURM_WORK_DIR": str(tmp_path),
            "SLURM_PARTITION": "gpu",
            "SLURM_ACCOUNT": "course",
            "SLURM_GPUS_PER_TASK": "2",
            "SLURM_CPUS_PER_TASK": "24",
            "SLURM_MEMORY_GB": "256",
            "SLURM_TIME_LIMIT_MINUTES": "120",
            "SLURM_CALLBACK_TOKEN_ENV": "SCALING_CALLBACK_TOKEN",
            "SLURM_WORKER_TRAINER": "course_trainer.worker:train",
            "SLURM_WORKER_HEARTBEAT_INTERVAL_SECONDS": "30",
            "SLURM_PYTHON": "python3",
        },
        now=lambda: "2026-07-16T15:00:00Z",
    )

    assert adapter.provider_name == "slurm"
    assert adapter.config.work_dir == tmp_path
    assert adapter.config.partition == "gpu"
    assert adapter.config.gpus_per_task == 2
    assert adapter.config.python_executable == "python3"
    assert adapter.config.heartbeat_interval_seconds == 30


@pytest.mark.parametrize(
    "key",
    [
        "SLURM_GPUS_PER_TASK",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEMORY_GB",
        "SLURM_TIME_LIMIT_MINUTES",
    ],
)
def test_build_slurm_adapter_from_env_rejects_non_positive_numeric_settings(
    tmp_path, key
):
    env = {
        "SLURM_WORK_DIR": str(tmp_path),
        "SLURM_GPUS_PER_TASK": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_MEMORY_GB": "64",
        "SLURM_TIME_LIMIT_MINUTES": "60",
    }
    env[key] = "0"

    with pytest.raises(SlurmProviderConfigError) as exc_info:
        build_slurm_adapter_from_env(env)

    assert key in str(exc_info.value)
    assert "positive integer" in str(exc_info.value)
