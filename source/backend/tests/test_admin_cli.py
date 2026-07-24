import json

from scaling_backend import admin_cli
from scaling_backend.runtime import load_api_keys_csv


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"ok": True}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [FakeResponse()])
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


def test_admin_cli_lists_filtered_experiments(capsys):
    http_client = FakeHttpClient(
        [
            FakeResponse(
                payload={"experiments": [{"experiment_id": "exp-1"}]}
            )
        ]
    )

    exit_code = admin_cli.main(
        [
            "--base-url",
            "https://backend.example/api/",
            "--admin-token",
            "admin-secret",
            "experiments",
            "--student-id",
            "student-1",
            "--status",
            "running",
        ],
        http_client=http_client,
    )

    assert exit_code == 0
    assert http_client.calls == [
        {
            "method": "GET",
            "url": "https://backend.example/api/admin/experiments",
            "headers": {"Authorization": "Bearer admin-secret"},
            "params": {"student_id": "student-1", "status": "running"},
            "json": None,
            "timeout": 30.0,
        }
    ]
    assert json.loads(capsys.readouterr().out) == {
        "experiments": [{"experiment_id": "exp-1"}]
    }


def test_admin_cli_seed_student_keys_writes_roster_csv_without_http(
    tmp_path, capsys
):
    roster = tmp_path / "roster.csv"
    output = tmp_path / "student_keys.csv"
    roster.write_text(
        "student_id,name\nstudent-1,Ada\nstudent-2,Grace\n",
        encoding="utf-8",
    )
    http_client = FakeHttpClient()

    exit_code = admin_cli.main(
        [
            "seed-student-keys",
            "--roster-csv",
            str(roster),
            "--output-csv",
            str(output),
        ],
        http_client=http_client,
    )

    assert exit_code == 0
    assert http_client.calls == []
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "student_id,api_key"
    assert [line.split(",", 1)[0] for line in lines[1:]] == [
        "student-1",
        "student-2",
    ]
    keys = [line.split(",", 1)[1] for line in lines[1:]]
    assert len(set(keys)) == 2
    assert all(key.startswith("sk-scale-") for key in keys)
    assert all(len(key) >= 32 for key in keys)
    assert load_api_keys_csv(output) == {
        keys[0]: "student-1",
        keys[1]: "student-2",
    }
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "output_csv": str(output),
        "student_count": 2,
    }
    assert keys[0] not in json.dumps(printed)


def test_admin_cli_seed_student_keys_refuses_overwrite_without_flag(
    tmp_path, capsys
):
    roster = tmp_path / "roster.csv"
    output = tmp_path / "student_keys.csv"
    roster.write_text("student_id\nstudent-1\n", encoding="utf-8")
    output.write_text("student_id,api_key\nstudent-old,key-old\n", encoding="utf-8")

    exit_code = admin_cli.main(
        [
            "seed-student-keys",
            "--roster-csv",
            str(roster),
            "--output-csv",
            str(output),
        ]
    )

    assert exit_code == 1
    assert output.read_text(encoding="utf-8") == (
        "student_id,api_key\nstudent-old,key-old\n"
    )
    assert "already exists" in capsys.readouterr().err


def test_admin_cli_seed_student_keys_rejects_duplicate_student_ids(
    tmp_path, capsys
):
    roster = tmp_path / "roster.csv"
    output = tmp_path / "student_keys.csv"
    roster.write_text(
        "student_id,name\nstudent-1,Ada\nstudent-1,Duplicate\n",
        encoding="utf-8",
    )

    exit_code = admin_cli.main(
        [
            "seed-student-keys",
            "--roster-csv",
            str(roster),
            "--output-csv",
            str(output),
        ]
    )

    assert exit_code == 1
    assert not output.exists()
    assert "duplicate student_id" in capsys.readouterr().err


def test_admin_cli_can_inspect_budget_cancel_and_export(capsys):
    http_client = FakeHttpClient(
        [
            FakeResponse(payload={"student_id": "student-1"}),
            FakeResponse(payload={"status": "cancelled"}),
            FakeResponse(payload={"experiments": []}),
        ]
    )

    budget_exit = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "student-budget",
            "student-1",
        ],
        http_client=http_client,
    )
    cancel_exit = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "cancel",
            "exp-1",
            "--reason",
            "quota audit",
            "--actor",
            "staff",
        ],
        http_client=http_client,
    )
    export_exit = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "exports",
        ],
        http_client=http_client,
    )

    assert (budget_exit, cancel_exit, export_exit) == (0, 0, 0)
    assert http_client.calls[0]["method"] == "GET"
    assert http_client.calls[0]["url"] == (
        "https://backend.example/admin/students/student-1/budget"
    )
    assert http_client.calls[1]["method"] == "POST"
    assert http_client.calls[1]["url"] == (
        "https://backend.example/admin/experiments/exp-1/cancel"
    )
    assert http_client.calls[1]["json"] == {
        "reason": "quota audit",
        "actor": "staff",
    }
    assert http_client.calls[2]["method"] == "GET"
    assert http_client.calls[2]["url"] == "https://backend.example/admin/exports"
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert printed == [
        {"student_id": "student-1"},
        {"status": "cancelled"},
        {"experiments": []},
    ]


def test_admin_cli_can_poll_experiment_and_final_run_and_cancel_final_run(capsys):
    http_client = FakeHttpClient(
        [
            FakeResponse(payload={"status": "running"}),
            FakeResponse(payload={"status": "lost"}),
            FakeResponse(payload={"status": "cancelled"}),
        ]
    )

    poll_exit = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "poll",
            "exp-1",
        ],
        http_client=http_client,
    )
    poll_final_exit = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "poll-final",
            "final-run-1",
        ],
        http_client=http_client,
    )
    cancel_final_exit = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "cancel-final",
            "final-run-1",
            "--reason",
            "staff dry-run cleanup",
            "--actor",
            "staff",
        ],
        http_client=http_client,
    )

    assert (poll_exit, poll_final_exit, cancel_final_exit) == (0, 0, 0)
    assert http_client.calls[0]["method"] == "POST"
    assert http_client.calls[0]["url"] == (
        "https://backend.example/admin/experiments/exp-1/poll"
    )
    assert http_client.calls[0]["json"] is None
    assert http_client.calls[1]["method"] == "POST"
    assert http_client.calls[1]["url"] == (
        "https://backend.example/admin/final-runs/final-run-1/poll"
    )
    assert http_client.calls[1]["json"] is None
    assert http_client.calls[2]["method"] == "POST"
    assert http_client.calls[2]["url"] == (
        "https://backend.example/admin/final-runs/final-run-1/cancel"
    )
    assert http_client.calls[2]["json"] == {
        "reason": "staff dry-run cleanup",
        "actor": "staff",
    }
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert printed == [
        {"status": "running"},
        {"status": "lost"},
        {"status": "cancelled"},
    ]


def test_admin_cli_can_poll_active_runs(capsys):
    http_client = FakeHttpClient(
        [
            FakeResponse(
                payload={
                    "exploratory": [{"experiment_id": "exp-1", "status": "running"}],
                    "final": [{"final_run_id": "final-run-1", "status": "running"}],
                }
            )
        ]
    )

    exit_code = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "poll-active",
        ],
        http_client=http_client,
    )

    assert exit_code == 0
    assert http_client.calls == [
        {
            "method": "POST",
            "url": "https://backend.example/admin/poll-active",
            "headers": {"Authorization": "Bearer admin-secret"},
            "params": None,
            "json": None,
            "timeout": 30.0,
        }
    ]
    assert json.loads(capsys.readouterr().out) == {
        "exploratory": [{"experiment_id": "exp-1", "status": "running"}],
        "final": [{"final_run_id": "final-run-1", "status": "running"}],
    }


def test_admin_cli_can_import_worker_events(capsys):
    http_client = FakeHttpClient(
        [
            FakeResponse(
                payload={
                    "ok": True,
                    "files": 1,
                    "lines": 3,
                    "imported": 2,
                    "duplicates": 1,
                }
            )
        ]
    )

    exit_code = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "import-worker-events",
        ],
        http_client=http_client,
    )

    assert exit_code == 0
    assert http_client.calls == [
        {
            "method": "POST",
            "url": "https://backend.example/admin/import-worker-events",
            "headers": {"Authorization": "Bearer admin-secret"},
            "params": None,
            "json": None,
            "timeout": 30.0,
        }
    ]
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "files": 1,
        "lines": 3,
        "imported": 2,
        "duplicates": 1,
    }


def test_admin_cli_exports_can_write_jsonl_and_csv_bundle(tmp_path, capsys):
    http_client = FakeHttpClient(
        [
            FakeResponse(
                payload={
                    "experiments": [
                        {
                            "experiment_id": "exp-1",
                            "student_id": "student-1",
                            "status": "completed",
                            "final_validation_loss": 3.2,
                        }
                    ],
                    "experiment_events": [
                        {
                            "experiment_id": "exp-1",
                            "event_type": "worker_completed",
                            "payload": {"final_validation_loss": 3.2},
                        }
                    ],
                    "worker_events": [
                        {
                            "experiment_id": "exp-1",
                            "event_type": "worker_completed",
                            "payload": {"final_validation_loss": 3.2},
                        },
                        {
                            "final_run_id": "final-run-1",
                            "event_type": "worker_completed",
                            "payload": {"actual_final_validation_loss": 2.7},
                        },
                    ],
                    "budget_snapshots": [
                        {
                            "student_id": "student-1",
                            "total_seconds": 43200,
                            "reserved_seconds": 0,
                            "charged_seconds": 300,
                            "remaining_seconds": 42900,
                        }
                    ],
                    "budget_adjustments": [],
                    "admin_actions": [],
                    "final_submissions": [
                        {
                            "final_submission_id": "final-submission-1",
                            "student_id": "student-1",
                            "predicted_final_loss": 2.8,
                        }
                    ],
                    "final_runs": [
                        {
                            "final_run_id": "final-run-1",
                            "student_id": "student-1",
                            "status": "completed",
                            "actual_final_validation_loss": 2.7,
                            "predicted_final_loss": 2.8,
                            "predicted_final_loss_lower": 2.6,
                            "predicted_final_loss_upper": 2.9,
                        }
                    ],
                }
            )
        ]
    )

    exit_code = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "exports",
            "--output-dir",
            str(tmp_path),
        ],
        http_client=http_client,
    )

    assert exit_code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["output_dir"] == str(tmp_path)
    assert set(body["paths"]) == {
        "experiments_jsonl",
        "experiments_csv",
        "experiment_events_jsonl",
        "worker_events_jsonl",
        "budget_snapshots_jsonl",
        "budget_snapshots_csv",
        "budget_adjustments_jsonl",
        "admin_actions_jsonl",
        "final_submissions_jsonl",
        "final_submissions_csv",
        "final_runs_jsonl",
        "final_runs_csv",
        "grading_csv",
    }
    assert (tmp_path / "experiments.csv").read_text(encoding="utf-8").startswith(
        "experiment_id,student_id,status"
    )
    assert "student-1" in (tmp_path / "budget_snapshots.csv").read_text(
        encoding="utf-8"
    )
    assert "final-run-1" in (tmp_path / "grading.csv").read_text(encoding="utf-8")
    assert "worker_completed" in (
        tmp_path / "experiment_events.jsonl"
    ).read_text(encoding="utf-8")
    assert "final-run-1" in (tmp_path / "worker_events.jsonl").read_text(
        encoding="utf-8"
    )


def test_admin_cli_posts_adjustment_freeze_and_launch_requests():
    http_client = FakeHttpClient([FakeResponse(), FakeResponse(), FakeResponse()])

    adjustment_exit = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "adjust-budget",
            "student-1",
            "--seconds",
            "-120",
            "--reason",
            "provider outage refund",
            "--actor",
            "staff",
            "--experiment-id",
            "exp-1",
        ],
        http_client=http_client,
    )
    freeze_exit = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "freeze-finals",
            "--reason",
            "deadline",
            "--actor",
            "staff",
        ],
        http_client=http_client,
    )
    launch_exit = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "launch-finals",
            "--max-runtime-seconds",
            "7200",
            "--reason",
            "deadline",
            "--actor",
            "staff",
        ],
        http_client=http_client,
    )

    assert (adjustment_exit, freeze_exit, launch_exit) == (0, 0, 0)
    assert http_client.calls[0]["url"] == (
        "https://backend.example/admin/budget-adjustments"
    )
    assert http_client.calls[0]["json"] == {
        "student_id": "student-1",
        "seconds": -120,
        "reason": "provider outage refund",
        "actor": "staff",
        "experiment_id": "exp-1",
    }
    assert http_client.calls[1]["url"] == (
        "https://backend.example/admin/final-submissions/freeze"
    )
    assert http_client.calls[1]["json"] == {
        "reason": "deadline",
        "actor": "staff",
    }
    assert http_client.calls[2]["url"] == (
        "https://backend.example/admin/final-runs/launch"
    )
    assert http_client.calls[2]["json"] == {
        "max_runtime_seconds": 7200,
        "reason": "deadline",
        "actor": "staff",
    }


def test_admin_cli_marks_experiment_system_failure():
    http_client = FakeHttpClient(
        [FakeResponse(payload={"status": "system_failed", "failure_reason": "provider_outage"})]
    )

    exit_code = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "mark-system-failure",
            "exp-000001",
            "--failure-reason",
            "provider_outage",
            "--staff-failure-detail",
            "QZ region outage incident INC-42",
            "--refund-seconds",
            "120",
            "--reason",
            "confirmed provider outage",
            "--actor",
            "staff",
        ],
        http_client=http_client,
    )

    assert exit_code == 0
    assert http_client.calls == [
        {
            "method": "POST",
            "url": "https://backend.example/admin/experiments/exp-000001/system-failure",
            "headers": {"Authorization": "Bearer admin-secret"},
            "params": None,
            "json": {
                "failure_reason": "provider_outage",
                "staff_failure_detail": "QZ region outage incident INC-42",
                "refund_seconds": 120,
                "reason": "confirmed provider outage",
                "actor": "staff",
            },
            "timeout": 30.0,
        }
    ]


def test_admin_cli_marks_final_run_system_failure():
    http_client = FakeHttpClient(
        [
            FakeResponse(
                payload={
                    "status": "system_failed",
                    "failure_reason": "infrastructure_error",
                }
            )
        ]
    )

    exit_code = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "admin-secret",
            "mark-final-system-failure",
            "final-run-000001",
            "--failure-reason",
            "infrastructure_error",
            "--staff-failure-detail",
            "hidden eval shard unavailable incident INC-55",
            "--reason",
            "confirmed hidden eval outage",
            "--actor",
            "staff",
        ],
        http_client=http_client,
    )

    assert exit_code == 0
    assert http_client.calls == [
        {
            "method": "POST",
            "url": "https://backend.example/admin/final-runs/final-run-000001/system-failure",
            "headers": {"Authorization": "Bearer admin-secret"},
            "params": None,
            "json": {
                "failure_reason": "infrastructure_error",
                "staff_failure_detail": "hidden eval shard unavailable incident INC-55",
                "reason": "confirmed hidden eval outage",
                "actor": "staff",
            },
            "timeout": 30.0,
        }
    ]


def test_admin_cli_returns_nonzero_for_api_errors(capsys):
    http_client = FakeHttpClient(
        [
            FakeResponse(
                status_code=401,
                payload={
                    "error": "invalid_admin_token",
                    "message": "Invalid or missing admin token.",
                },
            )
        ]
    )

    exit_code = admin_cli.main(
        [
            "--base-url",
            "https://backend.example",
            "--admin-token",
            "bad-token",
            "queue",
        ],
        http_client=http_client,
    )

    assert exit_code == 1
    assert json.loads(capsys.readouterr().err) == {
        "status_code": 401,
        "error": "invalid_admin_token",
        "message": "Invalid or missing admin token.",
    }
