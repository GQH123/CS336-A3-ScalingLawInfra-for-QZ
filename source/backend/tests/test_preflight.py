import json
import subprocess

from scaling_backend import preflight


def _env(tmp_path, **overrides):
    roster = tmp_path / "student_keys.csv"
    roster.write_text(
        "student_id,api_key\n"
        "student-1,key-1\n"
        "student-2,key-2\n",
        encoding="utf-8",
    )
    local_manifests = tmp_path / "manifests"
    snapshot = tmp_path / "state" / "snapshot.json"
    values = {
        "SCALING_PROVIDER": "fake",
        "SCALING_STUDENT_KEYS_CSV": str(roster),
        "SCALING_INTERNAL_CALLBACK_TOKEN": "internal-token",
        "SCALING_ADMIN_API_TOKEN": "admin-token",
        "SCALING_CALLBACK_URL": "https://backend.example/internal/provider-events",
        "SCALING_MANIFEST_BASE_URI": "memory://manifests",
        "SCALING_LOCAL_MANIFEST_DIR": str(local_manifests),
        "SCALING_TOTAL_BUDGET_SECONDS": "43200",
        "SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT": "1",
        "SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL": "8",
        "SCALING_CODE_VERSION": "course-backend-v0",
        "SCALING_DATA_MANIFEST_ID": "exploratory-train-v0",
        "SCALING_EVAL_MANIFEST_ID": "exploratory-eval-v0",
        "SCALING_FINAL_DATA_MANIFEST_ID": "final-train-v0",
        "SCALING_FINAL_EVAL_MANIFEST_ID": "final-eval-hidden-v0",
        "SCALING_STATE_SNAPSHOT_PATH": str(snapshot),
    }
    values.update(overrides)
    return values


def _by_id(report):
    return {check["id"]: check for check in report["checks"]}


def test_preflight_passes_for_valid_fake_provider_environment(tmp_path):
    report = preflight.run_preflight(_env(tmp_path), create_dirs=True)
    checks = _by_id(report)

    assert report["ok"] is True
    assert report["provider"] == "fake"
    assert report["summary"] == {"pass": 12, "warn": 0, "fail": 0}
    assert checks["student_keys_csv"]["status"] == "pass"
    assert checks["student_keys_csv"]["metadata"] == {"student_count": 2}
    assert checks["required_secrets"]["status"] == "pass"
    assert checks["runtime_uris"]["status"] == "pass"
    assert checks["manifest_storage"]["status"] == "pass"
    assert checks["state_snapshot_path"]["status"] == "pass"
    assert checks["provider_environment"]["status"] == "pass"
    assert (tmp_path / "manifests").is_dir()
    assert (tmp_path / "state").is_dir()


def test_preflight_rejects_placeholder_callback_host(tmp_path):
    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_CALLBACK_URL="https://api.internal/internal/provider-events",
        )
    )

    assert report["ok"] is False
    check = _by_id(report)["runtime_uris"]
    assert check["status"] == "fail"
    assert "placeholder host" in check["message"]
    assert "worker containers" in check["message"]


def test_preflight_allows_placeholder_callback_when_qz_uses_shared_worker_events(
    tmp_path,
):
    worker_events = tmp_path / "worker-events"
    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_PROVIDER="qz_distributed",
            SCALING_CALLBACK_URL="https://api.internal/internal/provider-events",
            SCALING_WORKER_EVENT_IMPORT_DIR=str(worker_events),
            QZ_WORKER_EVENT_LOG_DIR=str(worker_events),
            QZ_API_BASE_URL="https://qz.sii.edu.cn",
            QZ_USERNAME="253108120093",
            QZ_PASSWORD_ENCRYPTED="a" * 256,
            QZ_COOKIE="session=abc",
            QZ_WORKSPACE_ID="ws-1",
            QZ_PROJECT_ID="project-1",
            QZ_COMPUTE_GROUP_ID="lcg-1",
            QZ_SPEC_ID="spec-1",
            QZ_SPEC_GPU_TYPE="H200",
            QZ_SPEC_GPU_COUNT="1",
            QZ_SPEC_CPU_COUNT="15",
            QZ_SPEC_MEMORY_GB="200",
            QZ_IMAGE="registry.example.com/scaling-worker:latest",
        ),
        create_dirs=True,
    )
    checks = _by_id(report)

    assert report["ok"] is True
    assert checks["runtime_uris"]["status"] == "pass"
    assert checks["worker_settings"]["status"] == "pass"
    assert checks["worker_settings"]["metadata"]["worker_event_channel"] == "jsonl"
    assert worker_events.is_dir()


def test_preflight_rejects_placeholder_manifest_host(tmp_path):
    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_MANIFEST_BASE_URI="https://manifests.internal/scaling",
        )
    )

    assert report["ok"] is False
    check = _by_id(report)["runtime_uris"]
    assert check["status"] == "fail"
    assert "placeholder host" in check["message"]
    assert "worker containers" in check["message"]


def test_preflight_loads_standard_api_config_file(tmp_path):
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
                    "callback_url": "https://backend.example/internal/provider-events",
                    "manifest_base_uri": "memory://manifests",
                    "local_manifest_dir": str(tmp_path / "manifests"),
                    "state_snapshot_path": str(tmp_path / "state" / "snapshot.json"),
                    "total_budget_seconds": 43200,
                    "max_active_experiments_per_student": 1,
                    "max_active_experiments_global": 8,
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
            }
        ),
        encoding="utf-8",
    )

    report = preflight.run_preflight(
        {"SCALING_API_CONFIG": str(config_path)},
        create_dirs=True,
    )
    checks = _by_id(report)

    assert report["ok"] is True
    assert checks["student_keys_csv"]["status"] == "pass"
    assert checks["tokenized_index_uris"]["status"] == "pass"
    assert checks["tokenized_index_uris"]["metadata"]["configured"] == [
        "SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI",
        "SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI",
        "SCALING_TOKENIZED_TRAIN_INDEX_URI",
        "SCALING_TOKENIZED_VALIDATION_INDEX_URI",
    ]


def test_preflight_validates_optional_tokenized_index_uri_settings(tmp_path):
    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_TOKENIZED_TRAIN_INDEX_URI=str(tmp_path / "tokenized/train/index.json"),
            SCALING_TOKENIZED_VALIDATION_INDEX_URI=(
                "file:///secure/course/tokenized/eval/index.json"
            ),
            SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI=(
                "https://storage.example/tokenized/final-train/index.json"
            ),
            SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI=(
                "http://storage.example/tokenized/final-eval/index.json"
            ),
        ),
        create_dirs=True,
    )
    checks = _by_id(report)

    assert report["ok"] is True
    assert checks["tokenized_index_uris"]["status"] == "pass"
    assert checks["tokenized_index_uris"]["metadata"] == {
        "configured": [
            "SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI",
            "SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI",
            "SCALING_TOKENIZED_TRAIN_INDEX_URI",
            "SCALING_TOKENIZED_VALIDATION_INDEX_URI",
        ]
    }


def test_preflight_validates_runtime_uri_settings(tmp_path):
    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_CALLBACK_URL="http://127.0.0.1:8000/internal/provider-events",
            SCALING_MANIFEST_BASE_URI="file:///secure/course/frozen-manifests",
            SCALING_PUBLISHED_MANIFEST_DIR=str(tmp_path / "published-manifests"),
            SCALING_PUBLISHED_MANIFEST_BASE_URI=(
                "https://storage.example/course/manifests"
            ),
        ),
        create_dirs=True,
    )
    checks = _by_id(report)

    assert report["ok"] is True
    assert checks["runtime_uris"]["status"] == "pass"
    assert checks["runtime_uris"]["metadata"] == {
        "configured": [
            "SCALING_CALLBACK_URL",
            "SCALING_MANIFEST_BASE_URI",
            "SCALING_PUBLISHED_MANIFEST_BASE_URI",
        ]
    }


def test_preflight_rejects_invalid_runtime_uri_settings(tmp_path):
    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_CALLBACK_URL="ftp://backend.example/internal/provider-events",
            SCALING_MANIFEST_BASE_URI="relative/manifests",
            SCALING_PUBLISHED_MANIFEST_DIR=str(tmp_path / "published-manifests"),
            SCALING_PUBLISHED_MANIFEST_BASE_URI="file:///secure/course/manifests",
        ),
        create_dirs=True,
    )
    checks = _by_id(report)

    assert report["ok"] is False
    assert checks["runtime_uris"]["status"] == "fail"
    assert "SCALING_CALLBACK_URL" in checks["runtime_uris"]["message"]
    assert "SCALING_MANIFEST_BASE_URI" in checks["runtime_uris"]["message"]
    assert "SCALING_PUBLISHED_MANIFEST_BASE_URI" in checks["runtime_uris"]["message"]


def test_preflight_rejects_invalid_tokenized_index_uri_settings(tmp_path):
    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_TOKENIZED_TRAIN_INDEX_URI="s3://course/tokenized/train/index.json",
            SCALING_TOKENIZED_VALIDATION_INDEX_URI=(
                "file://remote-host/secure/course/tokenized/eval/index.json"
            ),
            SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI=(
                "ftp://storage.example/tokenized/final-train/index.json"
            ),
        )
    )
    checks = _by_id(report)

    assert report["ok"] is False
    assert checks["tokenized_index_uris"]["status"] == "fail"
    assert "SCALING_TOKENIZED_TRAIN_INDEX_URI" in checks[
        "tokenized_index_uris"
    ]["message"]
    assert "SCALING_TOKENIZED_VALIDATION_INDEX_URI" in checks[
        "tokenized_index_uris"
    ]["message"]
    assert "SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI" in checks[
        "tokenized_index_uris"
    ]["message"]


def test_preflight_rejects_invalid_worker_tokenized_index_validation_mode(tmp_path):
    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE="sample",
        )
    )
    checks = _by_id(report)

    assert report["ok"] is False
    assert checks["worker_settings"]["status"] == "fail"
    assert "SCALING_WORKER_TOKENIZED_INDEX_VALIDATION_MODE" in checks[
        "worker_settings"
    ]["message"]


def test_preflight_uses_real_qz_provider_environment_names(tmp_path):
    env = _env(
        tmp_path,
        SCALING_PROVIDER="qz_distributed",
        QZ_API_BASE_URL="https://qz.sii.edu.cn",
        QZ_USERNAME="253108120093",
        QZ_PASSWORD_ENCRYPTED="a" * 256,
        QZ_COOKIE="session=abc",
        QZ_WORKSPACE_ID="ws-1",
        QZ_PROJECT_ID="project-1",
        QZ_COMPUTE_GROUP_ID="lcg-1",
        QZ_SPEC_ID="spec-1",
        QZ_SPEC_GPU_TYPE="H200",
        QZ_SPEC_GPU_COUNT="1",
        QZ_SPEC_CPU_COUNT="15",
        QZ_SPEC_MEMORY_GB="200",
        QZ_IMAGE="registry.example.com/scaling-worker:latest",
    )

    report = preflight.run_preflight(env, create_dirs=True)
    checks = _by_id(report)

    assert report["ok"] is True
    assert checks["provider_environment"]["status"] == "pass"
    assert "QZCLI" not in checks["provider_environment"]["message"]


def test_preflight_reports_missing_required_runtime_settings(tmp_path):
    env = _env(tmp_path)
    del env["SCALING_CALLBACK_URL"]

    report = preflight.run_preflight(env)
    checks = _by_id(report)

    assert report["ok"] is False
    assert checks["required_runtime_settings"]["status"] == "fail"
    assert "SCALING_CALLBACK_URL" in checks["required_runtime_settings"]["message"]


def test_preflight_does_not_probe_provider_by_default(tmp_path):
    calls = []

    def probe(_env):
        calls.append("probe")
        return preflight._check("provider_probe", "pass", "probe ran")

    report = preflight.run_preflight(
        _env(tmp_path),
        create_dirs=True,
        provider_probe=probe,
    )

    assert report["ok"] is True
    assert calls == []
    assert "provider_probe" not in _by_id(report)


def test_preflight_can_probe_fake_provider_without_external_calls(tmp_path):
    report = preflight.run_preflight(
        _env(tmp_path),
        create_dirs=True,
        probe_provider=True,
    )
    checks = _by_id(report)

    assert report["ok"] is True
    assert checks["provider_probe"]["status"] == "pass"
    assert checks["provider_probe"]["metadata"] == {"provider": "fake"}


def test_preflight_slurm_probe_runs_non_submitting_version_commands(tmp_path):
    calls = []

    def runner(args):
        calls.append(list(args))
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="version ok\n",
            stderr="",
        )

    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_PROVIDER="slurm",
            SLURM_WORK_DIR=str(tmp_path / "slurm"),
            SLURM_PARTITION="gpu",
            SLURM_ACCOUNT="course",
            SLURM_GPUS_PER_TASK="1",
            SLURM_CPUS_PER_TASK="8",
            SLURM_MEMORY_GB="64",
            SLURM_TIME_LIMIT_MINUTES="60",
        ),
        create_dirs=True,
        probe_provider=True,
        slurm_runner=runner,
    )
    checks = _by_id(report)

    assert report["ok"] is True
    assert checks["provider_probe"]["status"] == "pass"
    assert calls == [["sinfo", "--version"], ["sbatch", "--version"]]


def test_preflight_qz_probe_logs_in_without_submitting_job(tmp_path, monkeypatch):
    class FakeQzClient:
        def __init__(self, config):
            self.config = config
            self.logged_in = False

        def login_with_cas(self):
            self.logged_in = True
            return "session=fresh"

    monkeypatch.setattr(preflight, "QzClient", FakeQzClient)

    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_PROVIDER="qz_distributed",
            QZ_API_BASE_URL="https://qz.sii.edu.cn",
            QZ_USERNAME="253108120093",
            QZ_PASSWORD_ENCRYPTED="a" * 256,
            QZ_WORKSPACE_ID="ws-1",
            QZ_PROJECT_ID="project-1",
            QZ_COMPUTE_GROUP_ID="lcg-1",
            QZ_SPEC_ID="spec-1",
            QZ_SPEC_GPU_TYPE="H200",
            QZ_SPEC_GPU_COUNT="1",
            QZ_SPEC_CPU_COUNT="15",
            QZ_SPEC_MEMORY_GB="200",
            QZ_IMAGE="registry.example.com/scaling-worker:latest",
        ),
        create_dirs=True,
        probe_provider=True,
    )
    checks = _by_id(report)

    assert report["ok"] is True
    assert checks["provider_probe"]["status"] == "pass"
    assert checks["provider_probe"]["metadata"] == {
        "provider": "qz_distributed",
        "auth": "login",
    }


def test_preflight_qz_probe_uses_cookie_without_cas_credentials(
    tmp_path, monkeypatch
):
    calls = []

    class FakeQzClient:
        def __init__(self, config):
            self.config = config

        def probe_cookie_auth(self, *, workspace_id):
            calls.append(("cookie", self.config.cookie, workspace_id))
            return {"list": [], "total": 0}

        def login_with_cas(self):
            raise AssertionError("CAS login should not run when QZ_COOKIE is set")

    monkeypatch.setattr(preflight, "QzClient", FakeQzClient)

    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_PROVIDER="qz_distributed",
            QZ_API_BASE_URL="https://qz.sii.edu.cn",
            QZ_COOKIE="inspire-session=manual",
            QZ_WORKSPACE_ID="ws-1",
            QZ_PROJECT_ID="project-1",
            QZ_COMPUTE_GROUP_ID="lcg-1",
            QZ_SPEC_ID="spec-1",
            QZ_SPEC_GPU_TYPE="H200",
            QZ_SPEC_GPU_COUNT="1",
            QZ_SPEC_CPU_COUNT="15",
            QZ_SPEC_MEMORY_GB="200",
            QZ_IMAGE="registry.example.com/scaling-worker:latest",
        ),
        create_dirs=True,
        probe_provider=True,
    )
    checks = _by_id(report)

    assert report["ok"] is True
    assert calls == [("cookie", "inspire-session=manual", "ws-1")]
    assert checks["provider_environment"]["status"] == "pass"
    assert checks["provider_probe"]["status"] == "pass"
    assert checks["provider_probe"]["metadata"] == {
        "provider": "qz_distributed",
        "auth": "cookie",
    }


def test_preflight_reports_student_key_csv_errors(tmp_path):
    roster = tmp_path / "bad_student_keys.csv"
    roster.write_text("student,token\nstudent-1,key-1\n", encoding="utf-8")

    report = preflight.run_preflight(
        _env(tmp_path, SCALING_STUDENT_KEYS_CSV=str(roster))
    )
    checks = _by_id(report)

    assert report["ok"] is False
    assert checks["student_keys_csv"]["status"] == "fail"
    assert "student_id" in checks["student_keys_csv"]["message"]
    assert "api_key" in checks["student_keys_csv"]["message"]


def test_preflight_rejects_invalid_validation_token_settings(tmp_path):
    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_VALIDATION_TOKENS_PER_EVAL="0",
            SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL="not-an-int",
        )
    )
    checks = _by_id(report)

    assert report["ok"] is False
    assert checks["budget_settings"]["status"] == "fail"
    assert "SCALING_VALIDATION_TOKENS_PER_EVAL must be positive" in checks[
        "budget_settings"
    ]["message"]
    assert "SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL must be positive" in checks[
        "budget_settings"
    ]["message"]


def test_preflight_rejects_negative_provider_poll_interval(tmp_path):
    report = preflight.run_preflight(
        _env(tmp_path, SCALING_PROVIDER_POLL_INTERVAL_SECONDS="-1")
    )
    checks = _by_id(report)

    assert report["ok"] is False
    assert checks["budget_settings"]["status"] == "fail"
    assert "SCALING_PROVIDER_POLL_INTERVAL_SECONDS must be non-negative" in checks[
        "budget_settings"
    ]["message"]


def test_preflight_rejects_reused_or_placeholder_secrets(tmp_path):
    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_INTERNAL_CALLBACK_TOKEN="same-secret",
            SCALING_ADMIN_API_TOKEN="same-secret",
        )
    )
    checks = _by_id(report)

    assert report["ok"] is False
    assert checks["required_secrets"]["status"] == "fail"
    assert "must be distinct" in checks["required_secrets"]["message"]


def test_preflight_rejects_incomplete_published_manifest_settings(tmp_path):
    report = preflight.run_preflight(
        _env(
            tmp_path,
            SCALING_LOCAL_MANIFEST_DIR="",
            SCALING_PUBLISHED_MANIFEST_DIR=str(tmp_path / "published"),
            SCALING_PUBLISHED_MANIFEST_BASE_URI="",
        )
    )
    checks = _by_id(report)

    assert report["ok"] is False
    assert checks["manifest_storage"]["status"] == "fail"
    assert "must be set together" in checks["manifest_storage"]["message"]


def test_preflight_reports_snapshot_parent_without_creating_dirs(tmp_path):
    report = preflight.run_preflight(_env(tmp_path), create_dirs=False)
    checks = _by_id(report)

    assert report["ok"] is False
    assert checks["state_snapshot_path"]["status"] == "fail"
    assert "parent directory does not exist" in checks["state_snapshot_path"]["message"]


def test_preflight_cli_writes_json_report(tmp_path, capsys):
    output_path = tmp_path / "preflight.json"
    exit_code = preflight.main(
        [
            "--create-dirs",
            "--output-json",
            str(output_path),
        ],
        env=_env(tmp_path),
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert printed["ok"] is True
    assert written == printed


def test_preflight_cli_can_load_config_file(tmp_path, capsys):
    values = _env(tmp_path)
    config_path = tmp_path / "api-runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "api": {
                    "provider": values["SCALING_PROVIDER"],
                    "student_keys_csv": values["SCALING_STUDENT_KEYS_CSV"],
                    "internal_callback_token": values[
                        "SCALING_INTERNAL_CALLBACK_TOKEN"
                    ],
                    "admin_api_token": values["SCALING_ADMIN_API_TOKEN"],
                    "callback_url": values["SCALING_CALLBACK_URL"],
                    "manifest_base_uri": values["SCALING_MANIFEST_BASE_URI"],
                    "local_manifest_dir": values["SCALING_LOCAL_MANIFEST_DIR"],
                    "state_snapshot_path": values["SCALING_STATE_SNAPSHOT_PATH"],
                    "total_budget_seconds": values["SCALING_TOTAL_BUDGET_SECONDS"],
                    "max_active_experiments_per_student": values[
                        "SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT"
                    ],
                    "max_active_experiments_global": values[
                        "SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL"
                    ],
                    "code_version": values["SCALING_CODE_VERSION"],
                },
                "worker_manifest_defaults": {
                    "data_manifest_id": values["SCALING_DATA_MANIFEST_ID"],
                    "eval_manifest_id": values["SCALING_EVAL_MANIFEST_ID"],
                    "final_data_manifest_id": values[
                        "SCALING_FINAL_DATA_MANIFEST_ID"
                    ],
                    "final_eval_manifest_id": values[
                        "SCALING_FINAL_EVAL_MANIFEST_ID"
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = preflight.main(
        [
            "--config",
            str(config_path),
            "--create-dirs",
        ],
        env={},
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert _by_id(printed)["student_keys_csv"]["status"] == "pass"


def test_preflight_cli_enables_provider_probe(tmp_path, capsys):
    output_path = tmp_path / "preflight.json"
    exit_code = preflight.main(
        [
            "--create-dirs",
            "--probe-provider",
            "--output-json",
            str(output_path),
        ],
        env=_env(tmp_path),
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert _by_id(printed)["provider_probe"]["status"] == "pass"
