from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scaling_backend.providers.fake import FakeProviderAdapter
from scaling_backend.service import ExperimentService
from scaling_backend.worker.run import run_worker


STUDENTS = ("student-1", "student-2")
REHEARSAL_CALLBACK_TOKEN = "rehearsal-token"


class _CallbackSession:
    def __init__(self, service: ExperimentService):
        self.service = service

    def post(self, _url, json=None, headers=None, timeout=None):
        self.service.record_worker_event(json or {})

        class Response:
            status_code = 200
            text = "ok"

        return Response()


def run_local_rehearsal(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    operations_log = root / "operations.log"
    _append_log(operations_log, f"seeded_students {','.join(STUDENTS)}")
    provider = FakeProviderAdapter(now=lambda: "2026-07-16T15:00:00Z")
    service = ExperimentService(
        provider=provider,
        total_budget_seconds=12 * 3600,
        now=lambda: "2026-07-16T15:00:00Z",
        manifest_uri_builder=lambda run_id: f"memory://rehearsal/{run_id}.json",
        callback_url="https://backend.internal/internal/provider-events",
        code_version="local-rehearsal",
        data_manifest_id="rehearsal-exploratory-train",
        eval_manifest_id="rehearsal-exploratory-eval",
        final_data_manifest_id="rehearsal-final-train",
        final_eval_manifest_id="rehearsal-final-eval-hidden",
    )
    session = _CallbackSession(service)

    exploratory_ids = []
    for index, student_id in enumerate(STUDENTS, start=1):
        submitted = service.submit(
            student_id=student_id,
            config=_training_config(
                num_hidden_layers=1 + index,
                hidden_size=128,
                train_tokens=1024 * index,
                learning_rate=3e-4,
            ),
            requested_runtime_seconds=300,
        )
        exploratory_ids.append(submitted.experiment_id)
        _run_fake_worker(
            provider.submitted_manifests[f"fake-job-{index:06d}"],
            session=session,
            losses=[4.0 - index * 0.2, 3.8 - index * 0.2],
            runtime_seconds=120 + index * 10,
        )
        service.set_final_submission(
            student_id=student_id,
            training_config=_training_config(
                num_hidden_layers=2 + index,
                hidden_size=128,
                train_tokens=2048 * index,
                learning_rate=2e-4,
            ),
            predicted_final_loss=2.8 - index * 0.1,
            predicted_final_loss_lower=2.7 - index * 0.1,
            predicted_final_loss_upper=2.95 - index * 0.1,
        )

    queue_before_final_launch = service.admin_queue_snapshot()
    _append_log(operations_log, "queue_inspected before_final_launch")
    service.freeze_final_submissions(actor="rehearsal", reason="local rehearsal")
    _append_log(operations_log, f"freeze_final_submissions count={len(STUDENTS)}")
    final_runs = service.launch_final_runs(max_runtime_seconds=7200)
    queue_after_final_launch = service.admin_queue_snapshot()
    _append_log(operations_log, f"launch_final_runs count={len(final_runs)}")
    for offset, final_run in enumerate(final_runs, start=1):
        job_id = f"fake-job-{len(STUDENTS) + offset:06d}"
        _run_fake_worker(
            provider.submitted_manifests[job_id],
            session=session,
            losses=[2.9 - offset * 0.1, 2.75 - offset * 0.1],
            runtime_seconds=7200,
        )

    export_paths = service.admin_write_course_exports(root / "exports")
    _append_log(operations_log, "export_course_records")
    secret_scan_passed = _secret_scan_passed(
        paths=[operations_log, *(Path(path) for path in export_paths.values())],
        secrets={REHEARSAL_CALLBACK_TOKEN},
    )
    if not secret_scan_passed:
        raise RuntimeError("rehearsal secret scan failed")
    _append_log(operations_log, "secret_scan_passed")
    report = _build_report(
        service.admin_export_course_records(),
        export_paths=export_paths,
        exploratory_ids=exploratory_ids,
        final_run_ids=[run.final_run_id for run in final_runs],
        queue_snapshots={
            "before_final_launch": queue_before_final_launch,
            "after_final_launch": queue_after_final_launch,
        },
        operations_log=operations_log,
        secret_scan_passed=secret_scan_passed,
    )
    report_path = root / "rehearsal-report.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _run_fake_worker(
    manifest: Mapping[str, Any],
    *,
    session: _CallbackSession,
    losses: list[float],
    runtime_seconds: int,
) -> None:
    def trainer(_manifest, emit_event):
        for step, loss in enumerate(losses, start=1):
            emit_event({"event_type": "validation", "step": step, "loss": loss})
        return {
            "validation_losses": losses,
            "final_validation_loss": losses[-1],
            "actual_runtime_seconds": runtime_seconds,
        }

    exit_code = run_worker(
        manifest=manifest,
        callback_url="https://backend.internal/internal/provider-events",
        callback_token=REHEARSAL_CALLBACK_TOKEN,
        trainer=trainer,
        session=session,
    )
    if exit_code != 0:
        raise RuntimeError(f"rehearsal worker failed for manifest {manifest}")


def _build_report(
    records: Mapping[str, list[dict[str, Any]]],
    *,
    export_paths: Mapping[str, str],
    exploratory_ids: list[str],
    final_run_ids: list[str],
    queue_snapshots: Mapping[str, Mapping[str, Any]],
    operations_log: Path,
    secret_scan_passed: bool,
) -> dict[str, Any]:
    grading_rows = _count_csv_rows(Path(export_paths["grading_csv"]))
    final_submissions = records["final_submissions"]
    return {
        "students": list(STUDENTS),
        "exploratory_experiment_ids": exploratory_ids,
        "final_run_ids": final_run_ids,
        "grading_rows": grading_rows,
        "status_counts": {
            "exploratory_completed": sum(
                1
                for experiment in records["experiments"]
                if experiment["status"] == "completed"
            ),
            "final_completed": sum(
                1
                for final_run in records["final_runs"]
                if final_run["status"] == "completed"
            ),
        },
        "queue_snapshots": {
            key: _queue_report(snapshot)
            for key, snapshot in queue_snapshots.items()
        },
        "export_paths": dict(export_paths),
        "log_paths": {
            "operations_log": str(operations_log),
        },
        "acceptance_checks": {
            "seeded_students": len(STUDENTS) == 2,
            "queue_inspected": set(queue_snapshots) == {
                "before_final_launch",
                "after_final_launch",
            },
            "final_submissions_frozen": (
                len(final_submissions) == len(STUDENTS)
                and all(item["frozen_at"] for item in final_submissions)
                and all(item["frozen_by"] == "rehearsal" for item in final_submissions)
                and all(
                    item["freeze_reason"] == "local rehearsal"
                    for item in final_submissions
                )
            ),
            "final_batch_launched": len(final_run_ids) == len(STUDENTS),
            "grading_export_written": grading_rows == len(STUDENTS),
            "operation_log_written": operations_log.exists(),
            "secret_scan_passed": secret_scan_passed,
        },
    }


def _queue_report(snapshot: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "exploratory": [
            {
                "experiment_id": item["experiment_id"],
                "status": item["status"],
            }
            for item in snapshot["exploratory"]
        ],
        "final": [
            {
                "final_run_id": item["final_run_id"],
                "status": item["status"],
            }
            for item in snapshot["final"]
        ],
    }


def _append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _secret_scan_passed(*, paths: list[Path], secrets: set[str]) -> bool:
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8")
        if any(secret and secret in text for secret in secrets):
            return False
    return True


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def _training_config(
    *,
    num_hidden_layers: int,
    hidden_size: int,
    train_tokens: int,
    learning_rate: float,
) -> dict[str, dict[str, Any]]:
    return {
        "model": {
            "num_hidden_layers": num_hidden_layers,
            "hidden_size": hidden_size,
        },
        "training": {
            "train_tokens": train_tokens,
            "num_evals": 1,
            "learning_rate": learning_rate,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scaling_backend.rehearsal",
        description="Run a deterministic local staff rehearsal with the fake provider.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report_path = run_local_rehearsal(args.output_dir)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
