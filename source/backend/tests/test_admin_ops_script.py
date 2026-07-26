from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "deploy" / "control-node" / "admin-ops.sh"


def _write_runtime_files(root: Path) -> dict[str, Path]:
    snapshot = root / "state-snapshot.json"
    worker_events = root / "worker-events"
    manifests = root / "frozen-manifests"
    worker_events.mkdir(parents=True)
    manifests.mkdir(parents=True)
    snapshot.write_text(
        json.dumps(
            {
                "snapshot_version": 1,
                "experiments": [{"experiment_id": "exp-000001"}],
                "budgets": {"student-1": {"charged_seconds": 120}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (worker_events / "events.jsonl").write_text('{"experiment_id":"exp-000001"}\n')
    (manifests / "exp-000001.json").write_text('{"experiment_id":"exp-000001"}\n')
    (root / "api-runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "api": {
                    "admin_api_token": "admin-secret",
                    "state_snapshot_path": str(snapshot),
                    "local_manifest_dir": str(manifests),
                    "worker_event_import_dir": str(worker_events),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "snapshot": snapshot,
        "worker_events": worker_events,
        "manifests": manifests,
    }


def _env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CONTROL_RUNTIME_DIR": str(root),
            "SCALING_API_CONFIG": str(root / "api-runtime.json"),
            "SCALING_ADMIN_BASE_URL": "http://127.0.0.1:9",
            "SCALING_ADMIN_API_TOKEN": "admin-secret",
        }
    )
    return env


def test_admin_ops_reset_state_requires_confirm(tmp_path):
    paths = _write_runtime_files(tmp_path)

    result = subprocess.run(
        [str(_script_path()), "reset-state"],
        env=_env(tmp_path),
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "requires --confirm" in result.stderr
    assert json.loads(paths["snapshot"].read_text(encoding="utf-8"))["experiments"] == [
        {"experiment_id": "exp-000001"}
    ]


def test_admin_ops_reset_state_archives_course_state_and_writes_empty_snapshot(tmp_path):
    paths = _write_runtime_files(tmp_path)

    result = subprocess.run(
        [str(_script_path()), "reset-state", "--confirm"],
        env=_env(tmp_path),
        text=True,
        capture_output=True,
        check=True,
    )

    snapshot = json.loads(paths["snapshot"].read_text(encoding="utf-8"))
    assert snapshot == {
        "snapshot_version": 1,
        "experiments": [],
        "budgets": {},
        "final_submissions": {},
        "frozen_final_submissions": {},
        "final_submissions_frozen": False,
        "final_runs_launched": False,
        "final_runs": [],
        "budget_adjustments": [],
        "admin_actions": [],
        "experiment_events": [],
        "worker_events": [],
        "dispatcher_records": [],
        "counters": {
            "next_experiment_number": 1,
            "next_final_submission_number": 1,
            "next_final_run_number": 1,
            "next_budget_adjustment_number": 1,
            "next_admin_action_number": 1,
        },
    }
    assert list(tmp_path.glob("state-snapshot.json.pre-reset-*"))
    assert not list(paths["worker_events"].glob("*.jsonl"))
    assert not list(paths["manifests"].glob("*.json"))
    assert list(tmp_path.glob("worker-events.archive-*"))
    assert list(tmp_path.glob("frozen-manifests.archive-*"))
    assert "reset complete" in result.stdout
