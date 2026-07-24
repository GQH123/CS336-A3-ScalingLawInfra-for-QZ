import csv
import json

from scaling_backend.rehearsal import run_local_rehearsal


def test_run_local_rehearsal_writes_report_and_course_exports(tmp_path):
    report_path = run_local_rehearsal(tmp_path)

    assert report_path == tmp_path / "rehearsal-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["students"] == ["student-1", "student-2"]
    assert report["exploratory_experiment_ids"] == ["exp-000001", "exp-000002"]
    assert report["final_run_ids"] == ["final-run-000001", "final-run-000002"]
    assert report["grading_rows"] == 2
    assert report["status_counts"] == {
        "exploratory_completed": 2,
        "final_completed": 2,
    }
    assert report["acceptance_checks"] == {
        "seeded_students": True,
        "queue_inspected": True,
        "final_submissions_frozen": True,
        "final_batch_launched": True,
        "grading_export_written": True,
        "operation_log_written": True,
        "secret_scan_passed": True,
    }
    assert report["queue_snapshots"]["before_final_launch"]["final"] == []
    assert [
        item["final_run_id"]
        for item in report["queue_snapshots"]["after_final_launch"]["final"]
    ] == ["final-run-000001", "final-run-000002"]
    assert report["log_paths"] == {
        "operations_log": str(tmp_path / "operations.log"),
    }
    assert sorted(report["export_paths"]) == [
        "admin_actions_jsonl",
        "budget_adjustments_jsonl",
        "budget_snapshots_csv",
        "budget_snapshots_jsonl",
        "experiment_events_jsonl",
        "experiments_csv",
        "experiments_jsonl",
        "final_runs_csv",
        "final_runs_jsonl",
        "final_submissions_csv",
        "final_submissions_jsonl",
        "grading_csv",
        "worker_events_jsonl",
    ]

    grading_path = tmp_path / "exports" / "grading.csv"
    with grading_path.open(newline="", encoding="utf-8") as handle:
        grading_rows = list(csv.DictReader(handle))
    assert [row["student_id"] for row in grading_rows] == ["student-1", "student-2"]
    assert [row["status"] for row in grading_rows] == ["completed", "completed"]
    assert (tmp_path / "exports" / "experiment_events.jsonl").exists()
    assert (tmp_path / "exports" / "worker_events.jsonl").exists()

    operations_log = tmp_path / "operations.log"
    assert operations_log.exists()
    log_text = operations_log.read_text(encoding="utf-8")
    for expected in [
        "seeded_students student-1,student-2",
        "queue_inspected before_final_launch",
        "freeze_final_submissions count=2",
        "launch_final_runs count=2",
        "export_course_records",
        "secret_scan_passed",
    ]:
        assert expected in log_text
    assert "rehearsal-token" not in log_text
