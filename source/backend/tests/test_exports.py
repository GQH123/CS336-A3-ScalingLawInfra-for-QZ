import csv
import json

from scaling_backend.exports import export_course_records


def _records():
    return {
        "experiments": [
            {
                "experiment_id": "exp-000001",
                "student_id": "student-1",
                "status": "completed",
                "validation_losses": [3.4, 3.2],
                "final_validation_loss": 3.2,
                "failure_reason": "",
                "used_runtime_seconds": 420,
                "completed_at": "2026-07-16T15:10:00Z",
                "failed_at": "",
                "reserved_runtime_seconds": 600,
                "config_hash": "hash-1",
                "provider_name": "fake",
                "provider_job_id": "provider-job-exp-1",
                "provider_status": "succeeded",
                "provider_artifacts": {
                    "provider_name": "fake",
                    "provider_job_id": "provider-job-exp-1",
                    "logs_uri": "memory://logs/provider-job-exp-1/logs.txt",
                    "detail_uri": "memory://logs/provider-job-exp-1/detail.json",
                    "metadata": {"stderr_uri": "memory://logs/provider-job-exp-1/stderr.txt"},
                },
                "resolved_config": {
                    "parameter_count_estimate": 1024,
                    "total_optimizer_steps": 10,
                },
            }
        ],
        "experiment_events": [
            {
                "experiment_id": "exp-000001",
                "student_id": "student-1",
                "event_type": "heartbeat",
                "timestamp": "2026-07-16T15:00:00Z",
                "payload": {"step": 1, "message": "alive"},
            }
        ],
        "worker_events": [
            {
                "experiment_id": "exp-000001",
                "student_id": "student-1",
                "event_type": "heartbeat",
                "timestamp": "2026-07-16T15:00:00Z",
                "payload": {"step": 1, "message": "alive"},
            },
            {
                "final_run_id": "final-run-000001",
                "student_id": "student-1",
                "event_type": "worker_completed",
                "timestamp": "2026-07-16T17:00:00Z",
                "payload": {"actual_final_validation_loss": 2.75},
            },
        ],
        "budget_snapshots": [
            {
                "student_id": "student-1",
                "total_seconds": 43200,
                "reserved_seconds": 0,
                "charged_seconds": 400,
                "remaining_seconds": 42800,
            }
        ],
        "budget_adjustments": [
            {
                "adjustment_id": "budget-adjustment-000001",
                "student_id": "student-1",
                "seconds": -20,
                "reason": "rounding refund",
                "actor": "staff",
                "created_at": "2026-07-16T15:00:00Z",
                "experiment_id": "exp-000001",
                "before_charged_seconds": 420,
                "after_charged_seconds": 400,
            }
        ],
        "admin_actions": [
            {
                "action_id": "admin-action-000001",
                "action_type": "cancel_experiment",
                "actor": "staff",
                "created_at": "2026-07-16T15:30:00Z",
                "reason": "quota audit",
                "student_id": "student-1",
                "experiment_id": "exp-000001",
                "final_run_id": "",
                "before_status": "submitted",
                "after_status": "cancelled",
            }
        ],
        "final_submissions": [
            {
                "final_submission_id": "final-submission-000001",
                "student_id": "student-1",
                "training_config": {"model": {"num_hidden_layers": 2}},
                "predicted_final_loss": 2.8,
                "predicted_final_loss_lower": 2.7,
                "predicted_final_loss_upper": 2.9,
                "updated_at": "2026-07-16T15:00:00Z",
                "frozen_at": "2026-07-16T16:00:00Z",
                "frozen_by": "staff",
                "freeze_reason": "deadline",
            }
        ],
        "final_runs": [
            {
                "final_run_id": "final-run-000001",
                "final_submission_id": "final-submission-000001",
                "student_id": "student-1",
                "status": "completed",
                "provider_job_id": "provider-job-1",
                "provider_artifacts": {
                    "provider_name": "fake",
                    "provider_job_id": "provider-job-1",
                    "logs_uri": "memory://logs/provider-job-1/logs.txt",
                    "detail_uri": "memory://logs/provider-job-1/detail.json",
                    "metadata": {},
                },
                "predicted_final_loss": 2.8,
                "predicted_final_loss_lower": 2.7,
                "predicted_final_loss_upper": 2.9,
                "actual_final_validation_loss": 2.75,
                "failure_reason": "",
                "resolved_config": {
                    "parameter_count_estimate": 2048,
                    "total_optimizer_steps": 20,
                },
            }
        ],
    }


def test_export_course_records_writes_jsonl_and_csv_artifacts(tmp_path):
    manifest = export_course_records(_records(), tmp_path)

    assert manifest == {
        "experiments_jsonl": str(tmp_path / "experiments.jsonl"),
        "experiments_csv": str(tmp_path / "experiments.csv"),
        "experiment_events_jsonl": str(tmp_path / "experiment_events.jsonl"),
        "worker_events_jsonl": str(tmp_path / "worker_events.jsonl"),
        "budget_snapshots_jsonl": str(tmp_path / "budget_snapshots.jsonl"),
        "budget_snapshots_csv": str(tmp_path / "budget_snapshots.csv"),
        "budget_adjustments_jsonl": str(tmp_path / "budget_adjustments.jsonl"),
        "admin_actions_jsonl": str(tmp_path / "admin_actions.jsonl"),
        "final_submissions_jsonl": str(tmp_path / "final_submissions.jsonl"),
        "final_submissions_csv": str(tmp_path / "final_submissions.csv"),
        "final_runs_jsonl": str(tmp_path / "final_runs.jsonl"),
        "final_runs_csv": str(tmp_path / "final_runs.csv"),
        "grading_csv": str(tmp_path / "grading.csv"),
    }

    experiment_rows = _read_csv(tmp_path / "experiments.csv")
    final_submission_rows = _read_csv(tmp_path / "final_submissions.csv")
    final_run_rows = _read_csv(tmp_path / "final_runs.csv")
    budget_snapshot_rows = _read_csv(tmp_path / "budget_snapshots.csv")
    grading_rows = _read_csv(tmp_path / "grading.csv")

    assert experiment_rows == [
        {
            "experiment_id": "exp-000001",
            "student_id": "student-1",
            "status": "completed",
            "final_validation_loss": "3.2",
            "failure_reason": "",
            "staff_failure_detail": "",
            "used_runtime_seconds": "420",
            "completed_at": "2026-07-16T15:10:00Z",
            "failed_at": "",
            "reserved_runtime_seconds": "600",
            "config_hash": "hash-1",
                "provider_name": "fake",
                "provider_job_id": "provider-job-exp-1",
                "provider_status": "succeeded",
                "provider_logs_uri": "memory://logs/provider-job-exp-1/logs.txt",
                "provider_detail_uri": "memory://logs/provider-job-exp-1/detail.json",
                "parameter_count_estimate": "1024",
                "total_optimizer_steps": "10",
            }
    ]
    assert final_submission_rows[0]["predicted_final_loss"] == "2.8"
    assert final_run_rows[0]["provider_job_id"] == "provider-job-1"
    assert final_run_rows[0]["provider_logs_uri"] == (
        "memory://logs/provider-job-1/logs.txt"
    )
    assert final_run_rows[0]["provider_detail_uri"] == (
        "memory://logs/provider-job-1/detail.json"
    )
    assert final_run_rows[0]["failure_reason"] == ""
    assert final_run_rows[0]["staff_failure_detail"] == ""
    assert budget_snapshot_rows == [
        {
            "student_id": "student-1",
            "total_seconds": "43200",
            "reserved_seconds": "0",
            "charged_seconds": "400",
            "remaining_seconds": "42800",
        }
    ]
    assert grading_rows == [
        {
            "student_id": "student-1",
            "final_run_id": "final-run-000001",
            "status": "completed",
            "actual_final_validation_loss": "2.75",
            "predicted_final_loss": "2.8",
            "predicted_final_loss_lower": "2.7",
            "predicted_final_loss_upper": "2.9",
            "prediction_absolute_error": "0.05",
            "prediction_interval_covered": "true",
            "prediction_interval_width": "0.2",
            "prediction_interval_miss_distance": "0.0",
            "prediction_interval_score": "0.2",
            "prediction_quality_penalty": "0.125",
            "final_run_grading_outcome": "valid_final_loss",
            "final_run_score_policy": (
                "rank by actual final validation loss and lower prediction_quality_penalty"
            ),
            "methodology_report_score": "",
            "methodology_report_notes": "",
            "analysis_code_score": "",
            "analysis_code_notes": "",
            "writeup_artifact_uri": "",
        }
    ]

    with (tmp_path / "experiments.jsonl").open() as handle:
        lines = [json.loads(line) for line in handle]
    assert lines[0]["resolved_config"]["parameter_count_estimate"] == 1024
    assert lines[0]["provider_artifacts"]["metadata"]["stderr_uri"] == (
        "memory://logs/provider-job-exp-1/stderr.txt"
    )
    with (tmp_path / "experiment_events.jsonl").open() as handle:
        event_lines = [json.loads(line) for line in handle]
    assert event_lines[0]["event_type"] == "heartbeat"
    assert event_lines[0]["payload"]["message"] == "alive"
    with (tmp_path / "worker_events.jsonl").open() as handle:
        worker_event_lines = [json.loads(line) for line in handle]
    assert [line["event_type"] for line in worker_event_lines] == [
        "heartbeat",
        "worker_completed",
    ]
    assert worker_event_lines[1]["final_run_id"] == "final-run-000001"
    with (tmp_path / "budget_snapshots.jsonl").open() as handle:
        budget_snapshot_lines = [json.loads(line) for line in handle]
    assert budget_snapshot_lines[0]["remaining_seconds"] == 42800
    with (tmp_path / "budget_adjustments.jsonl").open() as handle:
        adjustment_lines = [json.loads(line) for line in handle]
    assert adjustment_lines[0]["before_charged_seconds"] == 420
    assert adjustment_lines[0]["after_charged_seconds"] == 400
    with (tmp_path / "admin_actions.jsonl").open() as handle:
        admin_action_lines = [json.loads(line) for line in handle]
    assert admin_action_lines[0]["action_type"] == "cancel_experiment"
    assert admin_action_lines[0]["before_status"] == "submitted"
    assert admin_action_lines[0]["after_status"] == "cancelled"


def test_export_course_records_writes_empty_files_with_headers(tmp_path):
    manifest = export_course_records(
        {
            "experiments": [],
            "experiment_events": [],
            "worker_events": [],
            "budget_snapshots": [],
            "budget_adjustments": [],
            "admin_actions": [],
            "final_submissions": [],
            "final_runs": [],
        },
        tmp_path,
    )

    assert _read_csv(tmp_path / "grading.csv") == []
    assert (tmp_path / "experiments.jsonl").read_text() == ""
    assert set(manifest) == {
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


def test_grading_export_marks_student_caused_final_failures_for_worst_score(tmp_path):
    records = _records()
    records["final_runs"] = [
        {
            "final_run_id": "final-run-000001",
            "final_submission_id": "final-submission-000001",
            "student_id": "student-1",
            "status": "failed",
            "provider_job_id": "provider-job-1",
            "predicted_final_loss": 2.8,
            "predicted_final_loss_lower": 2.7,
            "predicted_final_loss_upper": 2.9,
            "actual_final_validation_loss": None,
            "failure_reason": "numerical",
        },
        {
            "final_run_id": "final-run-000002",
            "final_submission_id": "final-submission-000002",
            "student_id": "student-2",
            "status": "system_failed",
            "provider_job_id": "provider-job-2",
            "predicted_final_loss": 2.6,
            "predicted_final_loss_lower": 2.5,
            "predicted_final_loss_upper": 2.8,
            "actual_final_validation_loss": None,
            "failure_reason": "infrastructure_error",
        },
    ]

    export_course_records(records, tmp_path)

    grading_rows = _read_csv(tmp_path / "grading.csv")
    assert grading_rows[0]["final_run_grading_outcome"] == "student_failed_worst_score"
    assert grading_rows[0]["final_run_score_policy"] == (
        "assign worst final-run validation-loss score for the leaderboard component"
    )
    assert grading_rows[0]["prediction_absolute_error"] == ""
    assert grading_rows[0]["prediction_interval_covered"] == ""
    assert grading_rows[0]["prediction_interval_width"] == ""
    assert grading_rows[0]["prediction_interval_miss_distance"] == ""
    assert grading_rows[0]["prediction_interval_score"] == ""
    assert grading_rows[0]["prediction_quality_penalty"] == ""
    assert grading_rows[1]["final_run_grading_outcome"] == "staff_review_required"
    assert grading_rows[1]["final_run_score_policy"] == (
        "exclude from automatic worst-score penalty until staff adjudication"
    )


def test_grading_export_scores_prediction_interval_width_and_miss_distance(tmp_path):
    records = _records()
    records["final_runs"][0].update(
        {
            "predicted_final_loss": 2.7,
            "predicted_final_loss_lower": 2.6,
            "predicted_final_loss_upper": 2.9,
            "actual_final_validation_loss": 2.55,
        }
    )

    export_course_records(records, tmp_path)

    grading_rows = _read_csv(tmp_path / "grading.csv")
    assert grading_rows[0]["prediction_absolute_error"] == "0.15"
    assert grading_rows[0]["prediction_interval_covered"] == "false"
    assert grading_rows[0]["prediction_interval_width"] == "0.3"
    assert grading_rows[0]["prediction_interval_miss_distance"] == "0.05"
    assert grading_rows[0]["prediction_interval_score"] == "0.8"
    assert grading_rows[0]["prediction_quality_penalty"] == "0.475"


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
