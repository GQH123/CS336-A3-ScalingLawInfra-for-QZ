from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scaling_backend import preflight, release_audit, sot_status
from scaling_backend.rehearsal import run_local_rehearsal


def run_readiness_gate(
    *,
    env: Mapping[str, str] | None = None,
    create_dirs: bool = False,
    probe_provider: bool = False,
    rehearsal_output_dir: str | Path | None = None,
    release_root: str | Path | None = None,
    run_rehearsal: Callable[[Path], Path] = run_local_rehearsal,
) -> dict[str, Any]:
    values = env or os.environ
    preflight_report = preflight.run_preflight(
        values,
        create_dirs=create_dirs,
        probe_provider=probe_provider,
    )
    sot_report = sot_status.run_sot_status(release_root)
    release_report = release_audit.run_release_audit(release_root)
    rehearsal_dir = (
        Path(rehearsal_output_dir)
        if rehearsal_output_dir is not None
        else Path(tempfile.mkdtemp(prefix="scaling-readiness-rehearsal-"))
    )
    rehearsal_path = run_rehearsal(rehearsal_dir)
    rehearsal_report = json.loads(Path(rehearsal_path).read_text(encoding="utf-8"))

    checks = {
        "preflight": _preflight_check(preflight_report),
        "sot_status": _sot_status_check(sot_report),
        "release_audit": _release_audit_check(release_report),
        "rehearsal": _rehearsal_check(rehearsal_report, Path(rehearsal_path)),
    }
    summary = {
        "pass": sum(1 for check in checks.values() if check["status"] == "pass"),
        "warn": _warning_count(checks, release_report),
        "fail": sum(1 for check in checks.values() if check["status"] == "fail"),
    }
    return {
        "ok": summary["fail"] == 0,
        "summary": summary,
        "checks": checks,
        "preflight": preflight_report,
        "sot_status": sot_report,
        "release_audit": release_report,
        "rehearsal": rehearsal_report,
        "artifacts": {
            "rehearsal_report": str(rehearsal_path),
        },
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scaling_backend.readiness",
        description=(
            "Run the staff launch-readiness gate: preflight, release audit, "
            "and deterministic local rehearsal."
        ),
    )
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create configured preflight directories if missing.",
    )
    parser.add_argument(
        "--probe-provider",
        action="store_true",
        help="Include the explicit non-submitting provider probe in preflight.",
    )
    parser.add_argument(
        "--rehearsal-output-dir",
        default="",
        help="Directory for rehearsal artifacts. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--release-root",
        default="",
        help="Project root to pass to release audit. Defaults to the current source tree.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for writing the full readiness JSON report.",
    )
    args = parser.parse_args(argv)

    report = run_readiness_gate(
        env=env,
        create_dirs=args.create_dirs,
        probe_provider=args.probe_provider,
        rehearsal_output_dir=args.rehearsal_output_dir or None,
        release_root=args.release_root or None,
    )
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


def _preflight_check(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("ok") is True:
        return _check("pass", "Preflight passed.")
    return _check("fail", "Preflight failed.")


def _release_audit_check(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(report.get("summary", {}))
    if int(summary.get("fail", 0)) > 0:
        return _check("fail", "Release audit has failed checks.")
    if int(summary.get("warn", 0)) > 0:
        return _check("pass", "Release audit passed with warnings.")
    return _check("pass", "Release audit passed.")


def _sot_status_check(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(report.get("summary", {}))
    if int(summary.get("fail", 0)) > 0:
        return _check("fail", "SOT milestone status has failed milestone(s).")
    return _check("pass", "SOT milestone evidence is present.")


def _rehearsal_check(
    report: Mapping[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    acceptance = dict(report.get("acceptance_checks", {}))
    failing = sorted(key for key, value in acceptance.items() if value is not True)
    if failing:
        return _check(
            "fail",
            "Rehearsal acceptance check(s) failed: " + ", ".join(failing),
            metadata={"report_path": str(report_path), "failing_checks": failing},
        )
    return _check(
        "pass",
        "Rehearsal acceptance checks passed.",
        metadata={"report_path": str(report_path)},
    )


def _warning_count(
    checks: Mapping[str, Mapping[str, Any]],
    release_report: Mapping[str, Any],
) -> int:
    tool_warnings = sum(1 for check in checks.values() if check["status"] == "warn")
    release_warnings = int(dict(release_report.get("summary", {})).get("warn", 0))
    return tool_warnings + release_warnings


def _check(
    status: str,
    message: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {"status": status, "message": message}
    if metadata:
        result["metadata"] = dict(metadata)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
