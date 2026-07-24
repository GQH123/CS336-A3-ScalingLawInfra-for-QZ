from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Sequence

import requests

from scaling_backend.exports import export_course_records


DEFAULT_TIMEOUT_SECONDS = 30.0


def main(
    argv: Sequence[str] | None = None,
    *,
    http_client: Any = requests,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "seed-student-keys":
        return _seed_student_keys_command(args)

    base_url = _required_option(
        args.base_url,
        "SCALING_ADMIN_BASE_URL",
        "--base-url or SCALING_ADMIN_BASE_URL is required",
    )
    admin_token = _required_option(
        args.admin_token,
        "SCALING_ADMIN_API_TOKEN",
        "--admin-token or SCALING_ADMIN_API_TOKEN is required",
    )
    method, path, params, payload = _request_for_args(args)
    response = http_client.request(
        method,
        _join_url(base_url, path),
        headers={"Authorization": f"Bearer {admin_token}"},
        params=params,
        json=payload,
        timeout=args.timeout,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"message": response.text}

    if response.status_code >= 400:
        print(
            json.dumps(
                {"status_code": response.status_code, **body},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    if args.command == "exports" and args.output_dir:
        paths = export_course_records(body, Path(args.output_dir))
        print(
            json.dumps(
                {
                    "output_dir": str(Path(args.output_dir)),
                    "paths": paths,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    print(json.dumps(body, ensure_ascii=False, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scaling_backend.admin_cli",
        description="Instructor/admin CLI for the scaling-laws assignment backend.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SCALING_ADMIN_BASE_URL", ""),
        help="Backend base URL, or SCALING_ADMIN_BASE_URL.",
    )
    parser.add_argument(
        "--admin-token",
        default=os.environ.get("SCALING_ADMIN_API_TOKEN", ""),
        help="Admin bearer token, or SCALING_ADMIN_API_TOKEN.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP request timeout in seconds.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_student_keys = subparsers.add_parser(
        "seed-student-keys",
        help=(
            "Generate the student_id,api_key CSV consumed by "
            "SCALING_STUDENT_KEYS_CSV from a roster CSV."
        ),
    )
    seed_student_keys.add_argument("--roster-csv", required=True)
    seed_student_keys.add_argument("--output-csv", required=True)
    seed_student_keys.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output CSV. Disabled by default.",
    )

    subparsers.add_parser("queue", help="Inspect active exploratory and final queues.")

    experiments = subparsers.add_parser(
        "experiments", help="List experiments with optional filters."
    )
    experiments.add_argument("--student-id", default="")
    experiments.add_argument("--status", default="")

    student_budget = subparsers.add_parser(
        "student-budget", help="Inspect a student's budget."
    )
    student_budget.add_argument("student_id")

    experiment = subparsers.add_parser(
        "experiment", help="Inspect one experiment with staff-only detail."
    )
    experiment.add_argument("experiment_id")

    cancel = subparsers.add_parser("cancel", help="Cancel an experiment.")
    cancel.add_argument("experiment_id")
    cancel.add_argument("--reason", required=True)
    cancel.add_argument("--actor", required=True)

    poll = subparsers.add_parser("poll", help="Poll an experiment provider job.")
    poll.add_argument("experiment_id")

    mark_system_failure = subparsers.add_parser(
        "mark-system-failure",
        help="Mark an experiment as an instructor-reviewed system failure.",
    )
    mark_system_failure.add_argument("experiment_id")
    mark_system_failure.add_argument("--failure-reason", required=True)
    mark_system_failure.add_argument("--staff-failure-detail", default="")
    mark_system_failure.add_argument("--refund-seconds", type=int, default=0)
    mark_system_failure.add_argument("--reason", required=True)
    mark_system_failure.add_argument("--actor", required=True)

    subparsers.add_parser(
        "poll-active", help="Poll all active exploratory and final provider jobs."
    )
    subparsers.add_parser(
        "import-worker-events",
        help="Import shared-storage worker event JSONL files on the control node.",
    )

    adjustment = subparsers.add_parser(
        "adjust-budget", help="Record an audited budget adjustment."
    )
    adjustment.add_argument("student_id")
    adjustment.add_argument("--seconds", type=int, required=True)
    adjustment.add_argument("--reason", required=True)
    adjustment.add_argument("--actor", required=True)
    adjustment.add_argument("--experiment-id", default="")

    freeze = subparsers.add_parser(
        "freeze-finals", help="Freeze latest valid final submissions."
    )
    freeze.add_argument("--reason", required=True)
    freeze.add_argument("--actor", required=True)

    launch = subparsers.add_parser(
        "launch-finals", help="Launch final runs from frozen submissions."
    )
    launch.add_argument("--max-runtime-seconds", type=int, required=True)
    launch.add_argument("--reason", required=True)
    launch.add_argument("--actor", required=True)

    poll_final = subparsers.add_parser(
        "poll-final", help="Poll a hidden final-run provider job."
    )
    poll_final.add_argument("final_run_id")

    mark_final_system_failure = subparsers.add_parser(
        "mark-final-system-failure",
        help="Mark a hidden final run as an instructor-reviewed system failure.",
    )
    mark_final_system_failure.add_argument("final_run_id")
    mark_final_system_failure.add_argument("--failure-reason", required=True)
    mark_final_system_failure.add_argument("--staff-failure-detail", default="")
    mark_final_system_failure.add_argument("--reason", required=True)
    mark_final_system_failure.add_argument("--actor", required=True)

    cancel_final = subparsers.add_parser(
        "cancel-final", help="Cancel a hidden final-run provider job."
    )
    cancel_final.add_argument("final_run_id")
    cancel_final.add_argument("--reason", required=True)
    cancel_final.add_argument("--actor", required=True)

    exports = subparsers.add_parser("exports", help="Export course records.")
    exports.add_argument(
        "--output-dir",
        default="",
        help=(
            "Optional directory for writing the standard JSONL/CSV export bundle. "
            "When omitted, the raw JSON payload is printed."
        ),
    )
    return parser


def _seed_student_keys_command(args: argparse.Namespace) -> int:
    roster_path = Path(args.roster_csv)
    output_path = Path(args.output_csv)
    if output_path.exists() and not args.overwrite:
        print(
            f"Output CSV already exists: {output_path}. Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return 1

    try:
        student_ids = _read_roster_student_ids(roster_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Could not read roster CSV: {roster_path}: {exc}", file=sys.stderr)
        return 1

    rows = [
        {
            "student_id": student_id,
            "api_key": f"sk-scale-{secrets.token_urlsafe(32)}",
        }
        for student_id in student_ids
    ]

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["student_id", "api_key"])
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        print(f"Could not write student key CSV: {output_path}: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_csv": str(output_path),
                "student_count": len(rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _read_roster_student_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "student_id" not in fieldnames:
            raise ValueError("Roster CSV missing required column: student_id")

        student_ids: list[str] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            student_id = str(row.get("student_id", "")).strip()
            if not student_id:
                raise ValueError(
                    f"Roster CSV row {line_number} must include student_id"
                )
            if student_id in seen:
                raise ValueError(
                    f"Roster CSV contains duplicate student_id on row {line_number}: "
                    f"{student_id}"
                )
            seen.add(student_id)
            student_ids.append(student_id)

    if not student_ids:
        raise ValueError("Roster CSV must contain at least one student")
    return student_ids


def _request_for_args(args: argparse.Namespace) -> tuple[
    str,
    str,
    dict[str, str] | None,
    dict[str, Any] | None,
]:
    if args.command == "queue":
        return "GET", "/admin/queue", None, None
    if args.command == "experiments":
        return (
            "GET",
            "/admin/experiments",
            _non_empty_params(
                {
                    "student_id": args.student_id,
                    "status": args.status,
                }
            ),
            None,
        )
    if args.command == "student-budget":
        return "GET", f"/admin/students/{args.student_id}/budget", None, None
    if args.command == "experiment":
        return "GET", f"/admin/experiments/{args.experiment_id}", None, None
    if args.command == "cancel":
        return (
            "POST",
            f"/admin/experiments/{args.experiment_id}/cancel",
            None,
            {"reason": args.reason, "actor": args.actor},
        )
    if args.command == "poll":
        return "POST", f"/admin/experiments/{args.experiment_id}/poll", None, None
    if args.command == "mark-system-failure":
        return (
            "POST",
            f"/admin/experiments/{args.experiment_id}/system-failure",
            None,
            {
                "failure_reason": args.failure_reason,
                "staff_failure_detail": args.staff_failure_detail,
                "refund_seconds": args.refund_seconds,
                "reason": args.reason,
                "actor": args.actor,
            },
        )
    if args.command == "poll-active":
        return "POST", "/admin/poll-active", None, None
    if args.command == "import-worker-events":
        return "POST", "/admin/import-worker-events", None, None
    if args.command == "adjust-budget":
        return (
            "POST",
            "/admin/budget-adjustments",
            None,
            {
                "student_id": args.student_id,
                "seconds": args.seconds,
                "reason": args.reason,
                "actor": args.actor,
                "experiment_id": args.experiment_id,
            },
        )
    if args.command == "freeze-finals":
        return (
            "POST",
            "/admin/final-submissions/freeze",
            None,
            {"reason": args.reason, "actor": args.actor},
        )
    if args.command == "launch-finals":
        return (
            "POST",
            "/admin/final-runs/launch",
            None,
            {
                "max_runtime_seconds": args.max_runtime_seconds,
                "reason": args.reason,
                "actor": args.actor,
            },
        )
    if args.command == "poll-final":
        return "POST", f"/admin/final-runs/{args.final_run_id}/poll", None, None
    if args.command == "mark-final-system-failure":
        return (
            "POST",
            f"/admin/final-runs/{args.final_run_id}/system-failure",
            None,
            {
                "failure_reason": args.failure_reason,
                "staff_failure_detail": args.staff_failure_detail,
                "reason": args.reason,
                "actor": args.actor,
            },
        )
    if args.command == "cancel-final":
        return (
            "POST",
            f"/admin/final-runs/{args.final_run_id}/cancel",
            None,
            {"reason": args.reason, "actor": args.actor},
        )
    if args.command == "exports":
        return "GET", "/admin/exports", None, None
    raise ValueError(f"unknown command: {args.command}")


def _non_empty_params(params: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in params.items() if value}


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _required_option(value: str, env_name: str, message: str) -> str:
    if value:
        return value
    env_value = os.environ.get(env_name, "")
    if env_value:
        return env_value
    raise SystemExit(message)


if __name__ == "__main__":
    raise SystemExit(main())
