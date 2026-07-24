from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from scaling_backend.prediction_scoring import compute_prediction_metrics


EXPERIMENT_CSV_FIELDS = [
    "experiment_id",
    "student_id",
    "status",
    "final_validation_loss",
    "failure_reason",
    "staff_failure_detail",
    "used_runtime_seconds",
    "completed_at",
    "failed_at",
    "reserved_runtime_seconds",
    "config_hash",
    "provider_name",
    "provider_job_id",
    "provider_status",
    "provider_logs_uri",
    "provider_detail_uri",
    "parameter_count_estimate",
    "total_optimizer_steps",
]

FINAL_SUBMISSION_CSV_FIELDS = [
    "final_submission_id",
    "student_id",
    "predicted_final_loss",
    "predicted_final_loss_lower",
    "predicted_final_loss_upper",
    "updated_at",
    "frozen_at",
    "frozen_by",
    "freeze_reason",
]

FINAL_RUN_CSV_FIELDS = [
    "final_run_id",
    "final_submission_id",
    "student_id",
    "status",
    "provider_job_id",
    "provider_logs_uri",
    "provider_detail_uri",
    "predicted_final_loss",
    "predicted_final_loss_lower",
    "predicted_final_loss_upper",
    "actual_final_validation_loss",
    "failure_reason",
    "staff_failure_detail",
    "parameter_count_estimate",
    "total_optimizer_steps",
]

BUDGET_SNAPSHOT_CSV_FIELDS = [
    "student_id",
    "total_seconds",
    "reserved_seconds",
    "charged_seconds",
    "remaining_seconds",
]

GRADING_CSV_FIELDS = [
    "student_id",
    "final_run_id",
    "status",
    "actual_final_validation_loss",
    "predicted_final_loss",
    "predicted_final_loss_lower",
    "predicted_final_loss_upper",
    "prediction_absolute_error",
    "prediction_interval_covered",
    "prediction_interval_width",
    "prediction_interval_miss_distance",
    "prediction_interval_score",
    "prediction_quality_penalty",
    "final_run_grading_outcome",
    "final_run_score_policy",
    "methodology_report_score",
    "methodology_report_notes",
    "analysis_code_score",
    "analysis_code_notes",
    "writeup_artifact_uri",
]


def export_course_records(
    records: Mapping[str, list[Mapping[str, Any]]], output_dir: str | Path
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    experiments = list(records.get("experiments", []))
    experiment_events = list(records.get("experiment_events", []))
    worker_events = list(records.get("worker_events", experiment_events))
    budget_snapshots = list(records.get("budget_snapshots", []))
    budget_adjustments = list(records.get("budget_adjustments", []))
    admin_actions = list(records.get("admin_actions", []))
    final_submissions = list(records.get("final_submissions", []))
    final_runs = list(records.get("final_runs", []))

    paths = {
        "experiments_jsonl": out / "experiments.jsonl",
        "experiments_csv": out / "experiments.csv",
        "experiment_events_jsonl": out / "experiment_events.jsonl",
        "worker_events_jsonl": out / "worker_events.jsonl",
        "budget_snapshots_jsonl": out / "budget_snapshots.jsonl",
        "budget_snapshots_csv": out / "budget_snapshots.csv",
        "budget_adjustments_jsonl": out / "budget_adjustments.jsonl",
        "admin_actions_jsonl": out / "admin_actions.jsonl",
        "final_submissions_jsonl": out / "final_submissions.jsonl",
        "final_submissions_csv": out / "final_submissions.csv",
        "final_runs_jsonl": out / "final_runs.jsonl",
        "final_runs_csv": out / "final_runs.csv",
        "grading_csv": out / "grading.csv",
    }

    _write_jsonl(paths["experiments_jsonl"], experiments)
    _write_csv(paths["experiments_csv"], EXPERIMENT_CSV_FIELDS, _experiment_rows(experiments))
    _write_jsonl(paths["experiment_events_jsonl"], experiment_events)
    _write_jsonl(paths["worker_events_jsonl"], worker_events)
    _write_jsonl(paths["budget_snapshots_jsonl"], budget_snapshots)
    _write_csv(
        paths["budget_snapshots_csv"],
        BUDGET_SNAPSHOT_CSV_FIELDS,
        _budget_snapshot_rows(budget_snapshots),
    )
    _write_jsonl(paths["budget_adjustments_jsonl"], budget_adjustments)
    _write_jsonl(paths["admin_actions_jsonl"], admin_actions)
    _write_jsonl(paths["final_submissions_jsonl"], final_submissions)
    _write_csv(
        paths["final_submissions_csv"],
        FINAL_SUBMISSION_CSV_FIELDS,
        _final_submission_rows(final_submissions),
    )
    _write_jsonl(paths["final_runs_jsonl"], final_runs)
    _write_csv(paths["final_runs_csv"], FINAL_RUN_CSV_FIELDS, _final_run_rows(final_runs))
    _write_csv(paths["grading_csv"], GRADING_CSV_FIELDS, _grading_rows(final_runs))

    return {key: str(path) for key, path in paths.items()}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _write_csv(
    path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _experiment_rows(rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        resolved = _mapping(row.get("resolved_config"))
        artifacts = _mapping(row.get("provider_artifacts"))
        yield {
            "experiment_id": row.get("experiment_id", ""),
            "student_id": row.get("student_id", ""),
            "status": row.get("status", ""),
            "final_validation_loss": row.get("final_validation_loss", ""),
            "failure_reason": row.get("failure_reason", ""),
            "staff_failure_detail": row.get("staff_failure_detail", ""),
            "used_runtime_seconds": row.get("used_runtime_seconds", ""),
            "completed_at": row.get("completed_at", ""),
            "failed_at": row.get("failed_at", ""),
            "reserved_runtime_seconds": row.get("reserved_runtime_seconds", ""),
            "config_hash": row.get("config_hash", ""),
            "provider_name": row.get("provider_name", ""),
            "provider_job_id": row.get("provider_job_id", ""),
            "provider_status": row.get("provider_status", ""),
            "provider_logs_uri": artifacts.get("logs_uri", ""),
            "provider_detail_uri": artifacts.get("detail_uri", ""),
            "parameter_count_estimate": resolved.get("parameter_count_estimate", ""),
            "total_optimizer_steps": resolved.get("total_optimizer_steps", ""),
        }


def _final_submission_rows(
    rows: Iterable[Mapping[str, Any]]
) -> Iterable[dict[str, Any]]:
    for row in rows:
        yield {
            "final_submission_id": row.get("final_submission_id", ""),
            "student_id": row.get("student_id", ""),
            "predicted_final_loss": row.get("predicted_final_loss", ""),
            "predicted_final_loss_lower": row.get("predicted_final_loss_lower", ""),
            "predicted_final_loss_upper": row.get("predicted_final_loss_upper", ""),
            "updated_at": row.get("updated_at", ""),
            "frozen_at": row.get("frozen_at", ""),
            "frozen_by": row.get("frozen_by", ""),
            "freeze_reason": row.get("freeze_reason", ""),
        }


def _final_run_rows(rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        resolved = _mapping(row.get("resolved_config"))
        artifacts = _mapping(row.get("provider_artifacts"))
        yield {
            "final_run_id": row.get("final_run_id", ""),
            "final_submission_id": row.get("final_submission_id", ""),
            "student_id": row.get("student_id", ""),
            "status": row.get("status", ""),
            "provider_job_id": row.get("provider_job_id", ""),
            "provider_logs_uri": artifacts.get("logs_uri", ""),
            "provider_detail_uri": artifacts.get("detail_uri", ""),
            "predicted_final_loss": row.get("predicted_final_loss", ""),
            "predicted_final_loss_lower": row.get("predicted_final_loss_lower", ""),
            "predicted_final_loss_upper": row.get("predicted_final_loss_upper", ""),
            "actual_final_validation_loss": row.get("actual_final_validation_loss", ""),
            "failure_reason": row.get("failure_reason", ""),
            "staff_failure_detail": row.get("staff_failure_detail", ""),
            "parameter_count_estimate": resolved.get("parameter_count_estimate", ""),
            "total_optimizer_steps": resolved.get("total_optimizer_steps", ""),
        }


def _budget_snapshot_rows(
    rows: Iterable[Mapping[str, Any]]
) -> Iterable[dict[str, Any]]:
    for row in rows:
        yield {
            "student_id": row.get("student_id", ""),
            "total_seconds": row.get("total_seconds", ""),
            "reserved_seconds": row.get("reserved_seconds", ""),
            "charged_seconds": row.get("charged_seconds", ""),
            "remaining_seconds": row.get("remaining_seconds", ""),
        }


def _grading_rows(rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        prediction_metrics = _prediction_metrics(row)
        grading_outcome, score_policy = _final_run_grading_policy(row)
        yield {
            "student_id": row.get("student_id", ""),
            "final_run_id": row.get("final_run_id", ""),
            "status": row.get("status", ""),
            "actual_final_validation_loss": row.get("actual_final_validation_loss", ""),
            "predicted_final_loss": row.get("predicted_final_loss", ""),
            "predicted_final_loss_lower": row.get("predicted_final_loss_lower", ""),
            "predicted_final_loss_upper": row.get("predicted_final_loss_upper", ""),
            "prediction_absolute_error": (
                ""
                if prediction_metrics["prediction_absolute_error"] is None
                else prediction_metrics["prediction_absolute_error"]
            ),
            "prediction_interval_covered": _csv_bool(
                prediction_metrics["prediction_interval_covered"]
            ),
            "prediction_interval_width": (
                ""
                if prediction_metrics["prediction_interval_width"] is None
                else prediction_metrics["prediction_interval_width"]
            ),
            "prediction_interval_miss_distance": (
                ""
                if prediction_metrics["prediction_interval_miss_distance"] is None
                else prediction_metrics["prediction_interval_miss_distance"]
            ),
            "prediction_interval_score": (
                ""
                if prediction_metrics["prediction_interval_score"] is None
                else prediction_metrics["prediction_interval_score"]
            ),
            "prediction_quality_penalty": (
                ""
                if prediction_metrics["prediction_quality_penalty"] is None
                else prediction_metrics["prediction_quality_penalty"]
            ),
            "final_run_grading_outcome": grading_outcome,
            "final_run_score_policy": score_policy,
            "methodology_report_score": row.get("methodology_report_score", ""),
            "methodology_report_notes": row.get("methodology_report_notes", ""),
            "analysis_code_score": row.get("analysis_code_score", ""),
            "analysis_code_notes": row.get("analysis_code_notes", ""),
            "writeup_artifact_uri": row.get("writeup_artifact_uri", ""),
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _prediction_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        "prediction_absolute_error": row.get("prediction_absolute_error"),
        "prediction_interval_covered": row.get("prediction_interval_covered"),
        "prediction_interval_width": row.get("prediction_interval_width"),
        "prediction_interval_miss_distance": row.get(
            "prediction_interval_miss_distance"
        ),
        "prediction_interval_score": row.get("prediction_interval_score"),
        "prediction_quality_penalty": row.get("prediction_quality_penalty"),
    }
    if all(value is not None for value in metrics.values()):
        return metrics
    computed = compute_prediction_metrics(
        predicted_final_loss=row.get("predicted_final_loss"),
        predicted_final_loss_lower=row.get("predicted_final_loss_lower"),
        predicted_final_loss_upper=row.get("predicted_final_loss_upper"),
        actual_final_validation_loss=row.get("actual_final_validation_loss"),
    )
    return {
        key: metrics[key] if metrics[key] is not None else computed[key]
        for key in metrics
    }


def _final_run_grading_policy(row: Mapping[str, Any]) -> tuple[str, str]:
    status = str(row.get("status", ""))
    failure_reason = str(row.get("failure_reason", ""))
    actual = _optional_float(row.get("actual_final_validation_loss"))
    if status == "completed" and actual is not None:
        return (
            "valid_final_loss",
            "rank by actual final validation loss and lower prediction_quality_penalty",
        )
    if status == "system_failed" or failure_reason in {
        "provider_outage",
        "worker_lost",
        "infrastructure_error",
        "admin_intervention",
        "unknown_system",
    }:
        return (
            "staff_review_required",
            "exclude from automatic worst-score penalty until staff adjudication",
        )
    if status in {"failed", "cancelled", "lost", "unknown"}:
        return (
            "student_failed_worst_score",
            "assign worst final-run validation-loss score for the leaderboard component",
        )
    return (
        "pending_or_unlaunched",
        "not ready for final grading export",
    )


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _csv_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)
