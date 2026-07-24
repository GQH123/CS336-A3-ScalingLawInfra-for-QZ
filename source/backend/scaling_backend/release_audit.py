from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]

STALE_PUBLIC_PATTERNS = [
    "X-API-Key",
    "architecture_config",
    "optimizer_config",
    "status_type",
    "total_train_tokens",
    "existing_experiment_id",
    "active_experiment_limit_reached",
    '"detail"',
    "Recommended Strategy",
    "Recommendation Strategy",
    "推荐策略",
    "Students may discuss",
    "学生可以讨论",
    "derived head dimension",
    "head dimension is derived from that division",
    "rather than submitted as an independent field",
    "推导得到的 head dimension",
    "而不是单独提交的字段",
    "RoPE head dimension must be even",
    "suspicious_evaluation_cadence",
]

NOTEBOOK_REQUIRED_SNIPPETS = [
    "import requests",
    "api_base_url",
    "api_key",
    "requests.request(",
    "def api_request(",
    "def make_small_config(",
    "num_attention_heads",
    '"num_evals": int(num_evals)',
    "def wait_for_experiment(",
    'api_request("GET", "/budget")',
    '"/submit"',
    '"requested_runtime_seconds": 300',
    'api_request("GET", "/experiments")',
    '"/final_submission"',
    "final_validation_loss",
    "validation_losses",
    "predicted_final_loss_lower",
    "predicted_final_loss_upper",
]

NOTEBOOK_FORBIDDEN_SNIPPETS = [
    "ScalingApiClient",
    "small_example_config",
    "scaling_backend.student_client",
    "student_client.py",
    "source/backend",
    "PYTHONPATH",
]

PUBLIC_TEXT_PATHS = [
    "docs/scaling-laws-api-quickstart.md",
    "docs/scaling-laws-api-quickstart.zh-CN.md",
    "docs/scaling-laws-project-student-handout.md",
    "docs/scaling-laws-project-student-handout.zh-CN.md",
    "docs/scaling-assignment-sot.md",
    "docs/scaling-assignment-backend-research.md",
    "docs/instructor-operations-guide.md",
    "docs/provider-adapter-guide.md",
    "docs/examples/scaling_laws_student_workflow.ipynb",
    "docs/examples/scaling_laws_student_workflow.zh-CN.ipynb",
    "docs/latex/scaling-laws-api-quickstart.tex",
    "docs/latex/scaling-laws-api-quickstart.zh-CN.tex",
    "docs/latex/scaling-laws-project-student-handout.tex",
    "docs/latex/scaling-laws-project-student-handout.zh-CN.tex",
]

PUBLIC_STUDENT_TEXT_PATHS = [
    "docs/scaling-laws-api-quickstart.md",
    "docs/scaling-laws-api-quickstart.zh-CN.md",
    "docs/scaling-laws-project-student-handout.md",
    "docs/scaling-laws-project-student-handout.zh-CN.md",
]

PUBLIC_STUDENT_INTERNAL_PATH_PATTERNS = [
    "docs/",
    "source/",
    ".ipynb",
    "PYTHONPATH",
    "student_client.py",
    "scaling_backend.student_client",
]


def run_release_audit(root: str | Path | None = None) -> dict[str, Any]:
    base = ROOT if root is None else Path(root)
    checks = [
        _check_student_quickstarts(base),
        _check_student_notebook(base),
        _check_staff_operations_docs(base),
        _check_provider_adapter_guide(base),
        _check_preflight_and_rehearsal_tools(base),
        _check_public_api_stale_terms(base),
        _check_quickstart_latex_sources(base),
        _check_manual_deployment_package(base),
        _check_latex_pdf_toolchain(),
    ]
    summary = {
        "pass": sum(1 for check in checks if check["status"] == "pass"),
        "warn": sum(1 for check in checks if check["status"] == "warn"),
        "fail": sum(1 for check in checks if check["status"] == "fail"),
    }
    return {
        "ok": summary["fail"] == 0,
        "summary": summary,
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scaling_backend.release_audit",
        description="Audit local release-readiness artifacts for the scaling-laws assignment.",
    )
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Project root to audit. Defaults to the current source tree root.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for writing the release audit JSON report.",
    )
    args = parser.parse_args(argv)

    report = run_release_audit(args.root)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


def _check_student_quickstarts(base: Path) -> dict[str, Any]:
    paths = [
        "docs/scaling-laws-api-quickstart.md",
        "docs/scaling-laws-api-quickstart.zh-CN.md",
    ]
    required = [
        "Authorization",
        "Bearer",
        "requested_runtime_seconds",
        "final_validation_loss",
        "invalid_final_submission",
        "api_request(",
        "model.attention_bias",
        "model.head_dim",
        "model.hidden_size",
        "model.intermediate_size",
        "model.num_attention_heads",
        "model.num_hidden_layers",
        "model.num_key_value_heads",
        "model.rms_norm_eps",
        "model.rope_theta",
        "model.tie_word_embeddings",
        "model.dtype",
        "model.vocab_size",
        "50432",
        "EleutherAI/gpt-neox-20b",
        "500000000000",
        '"num_evals": 1',
        "[0, 1)",
        "worker_gpu_count",
        "FSDP",
        "training.train_tokens",
        "training.sequence_length",
        "training.train_batch_size",
        "training.validation_batch_size",
        "training.num_evals",
        "training.model_seed",
        "training.learning_rate",
        "training.optimizer",
        "training.lr_schedule",
        "training.weight_decay",
        "training.adam_beta1",
        "training.adam_beta2",
        "training.adam_epsilon",
        "training.warmup_fraction",
        "training.final_lr_fraction",
        "training.gradient_clip_norm",
        "Unsupported top-level",
    ]
    base_check = _check_text_files(
        base,
        "student_quickstarts",
        paths,
        required,
        "Student quickstarts cover bearer auth, runtime requests, polling, and final submission.",
    )
    if base_check["status"] != "pass":
        return base_check

    stale_statuses: dict[str, list[str]] = {}
    for relative in paths:
        path = base / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        found = []
        for pattern in [
            '"status": "submitted"',
            '"status": "lost"',
            '"status": "unknown"',
            '"lost"',
            '"unknown"',
            '"data_manifest_id"',
            '"eval_manifest_id"',
            '"provider_manifest_uri"',
        ]:
            if pattern in text:
                found.append(pattern)
        if found:
            stale_statuses[relative] = found
    if stale_statuses:
        return _check(
            "student_quickstarts",
            "fail",
            "Student quickstarts expose internal status term(s): "
            + json.dumps(stale_statuses, sort_keys=True),
            metadata={"paths": paths, "matches": stale_statuses},
        )
    return base_check


def _check_student_notebook(base: Path) -> dict[str, Any]:
    relatives = [
        "docs/examples/scaling_laws_student_workflow.ipynb",
        "docs/examples/scaling_laws_student_workflow.zh-CN.ipynb",
    ]
    code_cell_counts: dict[str, int] = {}
    for relative in relatives:
        path = base / relative
        if not path.exists():
            return _check("student_notebook", "fail", f"Missing {relative}")
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _check("student_notebook", "fail", f"Notebook JSON is invalid: {exc}")

        code_cells = [
            cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
        ]
        code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        text = path.read_text(encoding="utf-8")
        missing = [
            snippet for snippet in NOTEBOOK_REQUIRED_SNIPPETS if snippet not in code
        ]
        if missing:
            return _check(
                "student_notebook",
                "fail",
                f"{relative} missing required workflow snippet(s): "
                + ", ".join(missing),
            )
        forbidden = [
            snippet for snippet in NOTEBOOK_FORBIDDEN_SNIPPETS if snippet in text
        ]
        if forbidden:
            return _check(
                "student_notebook",
                "fail",
                f"{relative} exposes forbidden helper-client snippet(s): "
                + ", ".join(forbidden),
            )
        for cell in code_cells:
            if cell.get("execution_count") is not None or cell.get("outputs") != []:
                return _check(
                    "student_notebook",
                    "fail",
                    "Notebook must be committed without execution counts or outputs.",
                )
            try:
                compile("".join(cell.get("source", [])), str(path), "exec")
            except SyntaxError as exc:
                return _check(
                    "student_notebook",
                    "fail",
                    f"Notebook code does not parse: {exc}",
                )
        code_cell_counts[relative] = len(code_cells)
    return _check(
        "student_notebook",
        "pass",
        "Student workflow notebooks are valid and cover submit/poll/analyze/final submission.",
        metadata={
            "paths": relatives,
            "path": relatives[0],
            "code_cell_count": code_cell_counts[relatives[0]],
            "code_cell_counts": code_cell_counts,
            "required_snippets": list(NOTEBOOK_REQUIRED_SNIPPETS),
        },
    )


def _check_staff_operations_docs(base: Path) -> dict[str, Any]:
    paths = ["docs/instructor-operations-guide.md", "source/backend/README.md"]
    required = [
        "python -m scaling_backend.preflight",
        "--probe-provider",
        "python -m scaling_backend.readiness",
        "python -m scaling_backend.rehearsal",
        "seed-student-keys",
        "freeze-finals",
        "launch-finals",
        "exports --output-dir",
        "worker_events.jsonl",
        "budget_snapshots.csv",
        "provider_logs_uri",
        "queue_position",
        "fair_share_order",
        "final_run_grading_outcome",
        "final_run_score_policy",
        "methodology_report_score",
        "writeup_artifact_uri",
        "secret",
        "Readiness Checklist",
    ]
    return _check_text_files(
        base,
        "staff_operations_docs",
        paths,
        required,
        "Staff operations docs cover preflight, rehearsal, final-run workflow, exports, and secrets.",
    )


def _check_provider_adapter_guide(base: Path) -> dict[str, Any]:
    paths = ["docs/provider-adapter-guide.md"]
    required = [
        "qz_distributed",
        "slurm",
        "fake",
        "Frozen Job Manifest",
        "Worker Event Contract",
        "Cancel Job",
        "Failure Mapping",
        "docs/scaling-laws-api-quickstart.md",
        "Stanford canonical model field names",
        "legacy model field names are not valid",
    ]
    return _check_text_files(
        base,
        "provider_adapter_guide",
        paths,
        required,
        "Provider adapter guide covers target providers and worker/provider contracts.",
    )


def _check_preflight_and_rehearsal_tools(base: Path) -> dict[str, Any]:
    paths = [
        "source/backend/scaling_backend/preflight.py",
        "source/backend/scaling_backend/readiness.py",
        "source/backend/scaling_backend/rehearsal.py",
        "source/backend/tests/test_preflight.py",
        "source/backend/tests/test_readiness.py",
        "source/backend/tests/test_rehearsal.py",
        "source/backend/scaling_backend/sot_status.py",
        "source/backend/tests/test_sot_status.py",
    ]
    missing = [path for path in paths if not (base / path).exists()]
    if missing:
        return _check(
            "preflight_and_rehearsal_tools",
            "fail",
            "Missing tool/test path(s): " + ", ".join(missing),
        )
    return _check(
        "preflight_and_rehearsal_tools",
        "pass",
        "Preflight, SOT milestone status, and local rehearsal tools plus regression tests are present.",
        metadata={"paths": paths},
    )


def _check_public_api_stale_terms(base: Path) -> dict[str, Any]:
    matches: dict[str, list[str]] = {}
    for relative in PUBLIC_TEXT_PATHS:
        path = base / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        found = [pattern for pattern in STALE_PUBLIC_PATTERNS if pattern in text]
        if found:
            matches[relative] = found
    for relative in PUBLIC_STUDENT_TEXT_PATHS:
        path = base / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        found = [
            pattern
            for pattern in PUBLIC_STUDENT_INTERNAL_PATH_PATTERNS
            if pattern in text
        ]
        if found:
            matches.setdefault(relative, []).extend(found)
    if matches:
        return _check(
            "public_api_stale_terms",
            "fail",
            "Stale public API terms or internal paths found.",
            metadata={"matches": matches},
        )
    return _check(
        "public_api_stale_terms",
        "pass",
        "No stale public API terms found in release-facing docs, LaTeX, or notebooks.",
    )


def _check_quickstart_latex_sources(base: Path) -> dict[str, Any]:
    paths = [
        "docs/latex/scaling-laws-api-quickstart.tex",
        "docs/latex/scaling-laws-api-quickstart.zh-CN.tex",
        "docs/latex/scaling-laws-project-student-handout.tex",
        "docs/latex/scaling-laws-project-student-handout.zh-CN.tex",
    ]
    manual_reference_targets = [
        "scaling-laws-api-quickstart.zh-CN.tex",
        "scaling-laws-project-student-handout.zh-CN.tex",
    ]
    missing = [path for path in paths if not (base / path).exists()]
    if missing:
        return _check(
            "quickstart_latex_sources",
            "fail",
            "Missing LaTeX source(s): " + ", ".join(missing),
        )
    generator_path = base / "docs" / "latex" / "generate_quickstart_latex.py"
    if not generator_path.exists():
        return _check(
            "quickstart_latex_sources",
            "fail",
            "Missing LaTeX generator: docs/latex/generate_quickstart_latex.py",
        )
    generator_text = generator_path.read_text(encoding="utf-8")
    regenerated_manual_targets = [
        target
        for target in manual_reference_targets
        if f'"target": LATEX_DIR / "{target}"' in generator_text
    ]
    if regenerated_manual_targets:
        return _check(
            "quickstart_latex_sources",
            "fail",
            "Manual Chinese LaTeX reference target(s) must not be regenerated: "
            + ", ".join(regenerated_manual_targets),
            metadata={"manual_reference_targets": regenerated_manual_targets},
        )
    for relative in paths:
        text = (base / relative).read_text(encoding="utf-8")
        changed_background_terms = [
            "bgcolor=CodeBlockBg",
            "\\definecolor{CodeBlockBg}",
            "\\definecolor{CodeBlockFrame}",
            "rulecolor=\\color{CodeBlockFrame}",
        ]
        changed_background_found = [
            snippet for snippet in changed_background_terms if snippet in text
        ]
        if changed_background_found:
            return _check(
                "quickstart_latex_sources",
                "fail",
                f"LaTeX source changes the original Sepia code-block background: {relative}",
                metadata={"matches": {relative: changed_background_found}},
            )
        required = [
            "\\usepackage{minted}",
            "\\usemintedstyle{monokai}",
            "\\definecolor{CodeBlockText}",
            "\\AtBeginEnvironment{minted}{\\color{CodeBlockText}}",
            "\\BeforeBeginEnvironment{minted}{\\vspace{-0.25\\baselineskip}}",
            "\\AfterEndEnvironment{minted}{\\vspace{-0.35\\baselineskip}}",
            "\\setlength{\\parskip}{4pt plus 1pt minus 1pt}",
            "bgcolor=Sepia",
            "framesep=3pt",
            "formatcom=\\color{CodeBlockText}",
            "numbersep=6pt",
            "baselinestretch=0.96",
        ]
        missing_required = [snippet for snippet in required if snippet not in text]
        if missing_required:
            return _check(
                "quickstart_latex_sources",
                "fail",
                f"LaTeX source lacks readable minted styling: {relative}",
                metadata={"missing": {relative: missing_required}},
            )
    return _check(
        "quickstart_latex_sources",
        "pass",
        "Release LaTeX sources exist and use readable minted/monokai code styling.",
        metadata={"paths": paths},
    )


def _check_latex_pdf_toolchain() -> dict[str, Any]:
    missing = [tool for tool in ["latexmk", "xelatex"] if shutil.which(tool) is None]
    if missing:
        return _check(
            "latex_pdf_toolchain",
            "warn",
            "Native PDF compilation tools unavailable: " + ", ".join(missing),
        )
    return _check(
        "latex_pdf_toolchain",
        "pass",
        "Native PDF compilation tools are available.",
    )


def _check_manual_deployment_package(base: Path) -> dict[str, Any]:
    paths = [
        "deploy/README.md",
        "deploy/control-node/api-runtime.sample.json",
        "deploy/compute-node/worker-runtime.sample.json",
        "deploy/compute-node/data-contract.md",
        "deploy/compute-node/constraints.txt",
        "deploy/compute-node/requirements-cuda12.txt",
        "source/backend/pyproject.toml",
        "source/data/pyproject.toml",
        "source/training/pyproject.toml",
        "source/training/course_trainer/worker.py",
        "source/training/course_trainer/tokenized_data.py",
    ]
    missing = [path for path in paths if not (base / path).exists()]
    if missing:
        return _check(
            "manual_deployment_package",
            "fail",
            "Manual deployment package missing required artifact(s): "
            + ", ".join(missing),
            metadata={"missing": missing},
        )

    text_requirements = {
        "deploy/README.md": [
            "Manual Container Deployment",
            "Control node",
            "Compute node",
            "SCALING_API_CONFIG",
            "--tokenized-index-validation-mode metadata",
            "source/training",
            "course_trainer.worker:train",
            "deploy/compute-node/constraints.txt",
            "deploy/compute-node/requirements-cuda12.txt",
            "--no-deps flash-hog==0.5.0",
            "chex==0.1.91",
        ],
        "deploy/compute-node/constraints.txt": [
            "chex==0.1.91",
            "furu==0.0.27",
            "flash-hog==0.5.0",
        ],
        "deploy/compute-node/requirements-cuda12.txt": [
            "chex==0.1.91",
            "jax[cuda12]==0.9.1",
            "--no-deps flash-hog==0.5.0",
            "--no-deps ./refs/cs336-assignment3-scaling",
        ],
        "deploy/compute-node/data-contract.md": [
            "Compute-Node Data Contract",
            "schema_version",
            "flat_binary_uint32_le",
        ],
        "source/backend/pyproject.toml": ["scaling-laws-backend"],
        "source/data/pyproject.toml": ["scaling-laws-data"],
        "source/training/pyproject.toml": ["scaling-laws-course-trainer"],
        "source/training/course_trainer/worker.py": [
            "StanfordLocalBackend",
            "build_stanford_training_config_payload",
            "deploy/compute-node/constraints.txt",
            "JAX CUDA backend could not initialize",
        ],
    }
    missing_snippets: dict[str, list[str]] = {}
    for relative, required in text_requirements.items():
        text = (base / relative).read_text(encoding="utf-8")
        missing_for_file = [snippet for snippet in required if snippet not in text]
        if missing_for_file:
            missing_snippets[relative] = missing_for_file
    if missing_snippets:
        return _check(
            "manual_deployment_package",
            "fail",
            "Manual deployment artifact(s) missing required text: "
            + json.dumps(missing_snippets, sort_keys=True),
            metadata={"missing": missing_snippets},
        )

    cuda12_requirements = (
        base / "deploy" / "compute-node" / "requirements-cuda12.txt"
    ).read_text(encoding="utf-8")
    forbidden_active_lines = [
        line.strip()
        for line in cuda12_requirements.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and (line.strip().startswith("flash-hog") or "jax[cuda13]" in line)
    ]
    if forbidden_active_lines:
        return _check(
            "manual_deployment_package",
            "fail",
            "CUDA 12 requirements must not actively resolve dependency metadata "
            "that installs CUDA 13 JAX plugins: "
            + json.dumps(forbidden_active_lines, sort_keys=True),
            metadata={
                "path": "deploy/compute-node/requirements-cuda12.txt",
                "forbidden_active_lines": forbidden_active_lines,
            },
        )

    try:
        config = json.loads(
            (base / "deploy" / "control-node" / "api-runtime.sample.json").read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        return _check(
            "manual_deployment_package",
            "fail",
            f"deploy/control-node/api-runtime.sample.json is invalid JSON: {exc}",
        )
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        return _check(
            "manual_deployment_package",
            "fail",
            "deploy/control-node/api-runtime.sample.json must declare schema_version = 1.",
        )
    try:
        worker_config = json.loads(
            (
                base / "deploy" / "compute-node" / "worker-runtime.sample.json"
            ).read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        return _check(
            "manual_deployment_package",
            "fail",
            f"deploy/compute-node/worker-runtime.sample.json is invalid JSON: {exc}",
        )
    if not isinstance(worker_config, dict) or worker_config.get("schema_version") != 1:
        return _check(
            "manual_deployment_package",
            "fail",
            "deploy/compute-node/worker-runtime.sample.json must declare schema_version = 1.",
        )
    return _check(
        "manual_deployment_package",
        "pass",
        "Manual deployment package includes runtime config, install metadata, and operator docs.",
        metadata={"paths": paths},
    )


def _check_text_files(
    base: Path,
    check_id: str,
    paths: list[str],
    required_terms: list[str],
    success_message: str,
) -> dict[str, Any]:
    missing_paths = [path for path in paths if not (base / path).exists()]
    if missing_paths:
        return _check(check_id, "fail", "Missing file(s): " + ", ".join(missing_paths))
    combined = "\n".join((base / path).read_text(encoding="utf-8") for path in paths)
    missing_terms = [term for term in required_terms if term not in combined]
    if missing_terms:
        return _check(
            check_id,
            "fail",
            "Missing required term(s): " + ", ".join(missing_terms),
            metadata={"paths": paths},
        )
    return _check(check_id, "pass", success_message, metadata={"paths": paths})


def _deployment_generated_artifacts(base: Path) -> list[str]:
    roots = [
        base / "source" / "backend",
        base / "source" / "data",
        base / "source" / "training",
    ]
    generated: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not _is_generated_deployment_artifact(path):
                continue
            generated.append(path.relative_to(base).as_posix())
    return sorted(generated)


def _is_generated_deployment_artifact(path: Path) -> bool:
    if path.name in {"__pycache__", ".pytest_cache"}:
        return True
    if path.is_dir() and path.name.endswith(".egg-info"):
        return True
    if path.is_file() and path.suffix in {".pyc", ".pyo"}:
        return True
    return False


def _check(
    check_id: str,
    status: str,
    message: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {"id": check_id, "status": status, "message": message}
    if metadata:
        result["metadata"] = dict(metadata)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
