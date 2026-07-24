from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class MilestoneSpec:
    id: str
    title: str
    evidence: tuple[str, ...]


MILESTONES = (
    MilestoneSpec(
        id="M0",
        title="Scaffold Audit and Adaptation Map",
        evidence=(
            "docs/scaling-assignment-sot.md",
            "docs/scaling-assignment-backend-research.md",
            "docs/provider-adapter-guide.md",
        ),
    ),
    MilestoneSpec(
        id="M1",
        title="Submit-to-Result Smoke Run",
        evidence=(
            "source/backend/tests/test_smoke_submit_to_result.py",
            "source/backend/scaling_backend/service.py",
            "source/backend/scaling_backend/dispatcher.py",
            "source/backend/scaling_backend/providers/fake.py",
        ),
    ),
    MilestoneSpec(
        id="M2",
        title="Provider Abstraction and Adapters",
        evidence=(
            "source/backend/scaling_backend/providers/contracts.py",
            "source/backend/scaling_backend/providers/fake.py",
            "source/backend/scaling_backend/providers/slurm.py",
            "source/backend/scaling_backend/providers/qz_distributed/adapter.py",
            "source/backend/scaling_backend/providers/qz_distributed/payloads.py",
            "source/backend/tests/providers/qz_distributed/test_adapter.py",
            "docs/provider-adapter-guide.md",
        ),
    ),
    MilestoneSpec(
        id="M2.5",
        title="Data Pipeline and Tokenized Reservoir",
        evidence=(
            "source/data/README.md",
            "source/data/manifests/general_100b_mix.json",
            "source/data/scaling_data/pipeline.py",
            "source/data/scaling_data/download.py",
            "source/data/scaling_data/postprocess.py",
            "source/data/scaling_data/tokenize.py",
            "source/data/tests/test_data_pipeline.py",
        ),
    ),
    MilestoneSpec(
        id="M3",
        title="Policy and API Hardening",
        evidence=(
            "source/backend/scaling_backend/api.py",
            "source/backend/scaling_backend/config_validation.py",
            "source/backend/scaling_backend/student_client.py",
            "source/backend/tests/test_api.py",
            "source/backend/tests/test_config_validation.py",
            "source/backend/tests/test_student_client.py",
            "source/backend/tests/test_service.py",
        ),
    ),
    MilestoneSpec(
        id="M4",
        title="Final-Run and Admin Workflow",
        evidence=(
            "source/backend/scaling_backend/admin_cli.py",
            "source/backend/scaling_backend/exports.py",
            "source/backend/tests/test_admin_cli.py",
            "source/backend/tests/test_exports.py",
            "source/backend/tests/test_smoke_submit_to_result.py",
            "docs/instructor-operations-guide.md",
        ),
    ),
    MilestoneSpec(
        id="M5",
        title="Documentation, Deployment Rehearsal, and Course Readiness",
        evidence=(
            "source/backend/scaling_backend/preflight.py",
            "source/backend/scaling_backend/readiness.py",
            "source/backend/scaling_backend/rehearsal.py",
            "source/backend/scaling_backend/release_audit.py",
            "source/backend/tests/test_preflight.py",
            "source/backend/tests/test_readiness.py",
            "source/backend/tests/test_rehearsal.py",
            "source/backend/tests/test_release_audit.py",
            "docs/scaling-laws-api-quickstart.md",
            "docs/scaling-laws-project-student-handout.md",
            "docs/instructor-operations-guide.md",
        ),
    ),
)


def run_sot_status(root: str | Path | None = None) -> dict[str, Any]:
    base = ROOT if root is None else Path(root)
    milestones = [_milestone_status(base, milestone) for milestone in MILESTONES]
    summary = {
        "pass": sum(1 for item in milestones if item["status"] == "pass"),
        "warn": sum(1 for item in milestones if item["status"] == "warn"),
        "fail": sum(1 for item in milestones if item["status"] == "fail"),
    }
    return {
        "ok": summary["fail"] == 0,
        "summary": summary,
        "milestones": milestones,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scaling_backend.sot_status",
        description="Audit local implementation evidence against the assignment SoT milestones.",
    )
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Project root to audit. Defaults to the current source tree root.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for writing the SOT milestone status JSON report.",
    )
    args = parser.parse_args(argv)

    report = run_sot_status(args.root)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


def _milestone_status(base: Path, milestone: MilestoneSpec) -> dict[str, Any]:
    missing = [relative for relative in milestone.evidence if not (base / relative).exists()]
    status = "fail" if missing else "pass"
    if missing:
        message = "Missing milestone evidence: " + ", ".join(missing)
    else:
        message = "Milestone evidence is present."
    return {
        "id": milestone.id,
        "title": milestone.title,
        "status": status,
        "message": message,
        "evidence": list(milestone.evidence),
        "missing": missing,
    }


if __name__ == "__main__":
    raise SystemExit(main())
