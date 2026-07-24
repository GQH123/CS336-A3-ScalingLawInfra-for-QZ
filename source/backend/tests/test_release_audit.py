import json

from scaling_backend import release_audit


def _by_id(report):
    return {check["id"]: check for check in report["checks"]}


def _write_minimal_manual_deployment_package(
    base,
    *,
    cuda12_requirements: str | None = None,
) -> None:
    (base / "deploy" / "control-node").mkdir(parents=True)
    (base / "deploy" / "compute-node").mkdir(parents=True)
    (base / "source" / "backend").mkdir(parents=True)
    (base / "source" / "data").mkdir(parents=True)
    (base / "source" / "training" / "course_trainer").mkdir(parents=True)
    (base / "deploy" / "README.md").write_text(
        "Manual Container Deployment\n"
        "Control node\n"
        "Compute node\n"
        "SCALING_API_CONFIG\n"
        "--tokenized-index-validation-mode metadata\n"
        "source/training\n"
        "course_trainer.worker:train\n"
        "deploy/compute-node/constraints.txt\n"
        "deploy/compute-node/requirements-cuda12.txt\n"
        "python -m pip install --no-deps flash-hog==0.5.0\n"
        "chex==0.1.91\n",
        encoding="utf-8",
    )
    (base / "deploy" / "control-node" / "api-runtime.sample.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )
    (base / "deploy" / "compute-node" / "worker-runtime.sample.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )
    (base / "deploy" / "compute-node" / "data-contract.md").write_text(
        "Compute-Node Data Contract\nschema_version\nmetadata\nflat_binary_uint32_le\n",
        encoding="utf-8",
    )
    (base / "deploy" / "compute-node" / "constraints.txt").write_text(
        "furu==0.0.27\n"
        "chex==0.1.91\n"
        "flash-hog==0.5.0\n",
        encoding="utf-8",
    )
    (base / "deploy" / "compute-node" / "requirements-cuda12.txt").write_text(
        cuda12_requirements
        or (
            "# python -m pip install --no-deps flash-hog==0.5.0\n"
            "# python -m pip install --no-deps ./refs/cs336-assignment3-scaling\n"
            "chex==0.1.91\n"
            "jax[cuda12]==0.9.1\n"
        ),
        encoding="utf-8",
    )
    (base / "source" / "backend" / "pyproject.toml").write_text(
        "[project]\nname = 'scaling-laws-backend'\n",
        encoding="utf-8",
    )
    (base / "source" / "data" / "pyproject.toml").write_text(
        "[project]\nname = 'scaling-laws-data'\n",
        encoding="utf-8",
    )
    (base / "source" / "training" / "pyproject.toml").write_text(
        "[project]\nname = 'scaling-laws-course-trainer'\n",
        encoding="utf-8",
    )
    (base / "source" / "training" / "course_trainer" / "worker.py").write_text(
        "StanfordLocalBackend\n"
        "build_stanford_training_config_payload\n"
        "deploy/compute-node/constraints.txt\n"
        "JAX CUDA backend could not initialize\n",
        encoding="utf-8",
    )
    (base / "source" / "training" / "course_trainer" / "tokenized_data.py").write_text(
        "def load_indexed_dataset():\n    pass\n",
        encoding="utf-8",
    )


def test_release_audit_reports_m5_readiness_artifacts():
    report = release_audit.run_release_audit()
    checks = _by_id(report)

    assert report["ok"] is True
    assert report["summary"] == {"pass": 8, "warn": 1, "fail": 0}
    assert checks["student_quickstarts"]["status"] == "pass"
    assert checks["student_notebook"]["status"] == "pass"
    assert checks["staff_operations_docs"]["status"] == "pass"
    assert checks["provider_adapter_guide"]["status"] == "pass"
    assert checks["preflight_and_rehearsal_tools"]["status"] == "pass"
    assert checks["public_api_stale_terms"]["status"] == "pass"
    assert checks["quickstart_latex_sources"]["status"] == "pass"
    assert checks["manual_deployment_package"]["status"] == "pass"
    assert checks["latex_pdf_toolchain"]["status"] in {"pass", "warn"}

    notebook_metadata = checks["student_notebook"]["metadata"]
    assert notebook_metadata["path"] == "docs/examples/scaling_laws_student_workflow.ipynb"
    assert notebook_metadata["code_cell_count"] >= 6
    assert "requests.request(" in notebook_metadata["required_snippets"]
    assert '"/submit"' in notebook_metadata["required_snippets"]
    assert '"num_evals": int(num_evals)' in notebook_metadata["required_snippets"]
    assert "def wait_for_experiment(" in notebook_metadata["required_snippets"]
    assert "docs/examples/scaling_laws_student_workflow.zh-CN.ipynb" in notebook_metadata[
        "paths"
    ]


def test_release_audit_cli_writes_json_report(tmp_path, capsys):
    output_path = tmp_path / "release-audit.json"

    exit_code = release_audit.main(["--output-json", str(output_path)])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert printed == written
    assert printed["ok"] is True
    assert printed["summary"]["fail"] == 0


def test_release_audit_requires_manual_deployment_package_artifacts(tmp_path):
    (tmp_path / "deploy" / "control-node").mkdir(parents=True)
    (tmp_path / "deploy" / "compute-node").mkdir(parents=True)
    (tmp_path / "source" / "backend").mkdir(parents=True)
    (tmp_path / "source" / "data").mkdir(parents=True)
    (tmp_path / "source" / "training" / "course_trainer").mkdir(parents=True)
    (tmp_path / "deploy" / "README.md").write_text(
        "Manual Container Deployment\n"
        "Control node\n"
        "Compute node\n"
        "SCALING_API_CONFIG\n"
        "--tokenized-index-validation-mode metadata\n",
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "control-node" / "api-runtime.sample.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "compute-node" / "worker-runtime.sample.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "compute-node" / "data-contract.md").write_text(
        "Compute-Node Data Contract\nschema_version\nflat_binary_uint32_le\nmetadata\n",
        encoding="utf-8",
    )
    (tmp_path / "source" / "backend" / "pyproject.toml").write_text(
        "[project]\nname = 'scaling-laws-backend'\n",
        encoding="utf-8",
    )
    (tmp_path / "source" / "training" / "course_trainer" / "worker.py").write_text(
        "def train(manifest, emit_event):\n    raise NotImplementedError\n",
        encoding="utf-8",
    )

    check = release_audit._check_manual_deployment_package(tmp_path)

    assert check["status"] == "fail"
    assert "source/data/pyproject.toml" in check["message"]


def test_release_audit_requires_compute_node_trainer_package(tmp_path):
    (tmp_path / "deploy" / "control-node").mkdir(parents=True)
    (tmp_path / "deploy" / "compute-node").mkdir(parents=True)
    (tmp_path / "source" / "backend").mkdir(parents=True)
    (tmp_path / "source" / "data").mkdir(parents=True)
    (tmp_path / "source" / "training").mkdir(parents=True)
    (tmp_path / "deploy" / "README.md").write_text(
        "Manual Container Deployment\n"
        "Control node\n"
        "Compute node\n"
        "SCALING_API_CONFIG\n"
        "--tokenized-index-validation-mode metadata\n"
        "source/training\n"
        "course_trainer.worker:train\n"
        "deploy/compute-node/constraints.txt\n"
        "deploy/compute-node/requirements-cuda12.txt\n",
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "control-node" / "api-runtime.sample.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "compute-node" / "worker-runtime.sample.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "compute-node" / "data-contract.md").write_text(
        "Compute-Node Data Contract\nschema_version\nmetadata\nflat_binary_uint32_le\n",
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "compute-node" / "constraints.txt").write_text(
        "furu==0.0.27\n",
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "compute-node" / "requirements-cuda12.txt").write_text(
        "jax[cuda12]==0.9.1\n"
        "python -m pip install --no-deps ./refs/cs336-assignment3-scaling\n",
        encoding="utf-8",
    )
    (tmp_path / "source" / "backend" / "pyproject.toml").write_text(
        "[project]\nname = 'scaling-laws-backend'\n",
        encoding="utf-8",
    )
    (tmp_path / "source" / "data" / "pyproject.toml").write_text(
        "[project]\nname = 'scaling-laws-data'\n",
        encoding="utf-8",
    )

    check = release_audit._check_manual_deployment_package(tmp_path)

    assert check["status"] == "fail"
    assert "source/training/pyproject.toml" in check["message"]
    assert "source/training/course_trainer/worker.py" in check["message"]


def test_release_audit_requires_provider_manifest_schema_cross_reference(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "provider-adapter-guide.md").write_text(
        "\n".join(
            [
                "qz_distributed",
                "slurm",
                "fake",
                "Frozen Job Manifest",
                "Worker Event Contract",
                "Cancel Job",
                "Failure Mapping",
            ]
        ),
        encoding="utf-8",
    )

    check = release_audit._check_provider_adapter_guide(tmp_path)

    assert check["status"] == "fail"
    assert "Stanford canonical model field names" in check["message"]


def test_release_audit_detects_generated_artifacts_for_strict_package_cleanliness(
    tmp_path,
):
    (tmp_path / "deploy" / "control-node").mkdir(parents=True)
    (tmp_path / "deploy" / "compute-node").mkdir(parents=True)
    (tmp_path / "source" / "backend" / "scaling_backend").mkdir(parents=True)
    (tmp_path / "source" / "data").mkdir(parents=True)
    (tmp_path / "source" / "training" / "course_trainer").mkdir(parents=True)
    (tmp_path / "deploy" / "README.md").write_text(
        "Manual Container Deployment\n"
        "Control node\n"
        "Compute node\n"
        "SCALING_API_CONFIG\n"
        "--tokenized-index-validation-mode metadata\n"
        "source/training\n"
        "course_trainer.worker:train\n"
        "deploy/compute-node/constraints.txt\n"
        "deploy/compute-node/requirements-cuda12.txt\n",
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "control-node" / "api-runtime.sample.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "compute-node" / "worker-runtime.sample.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "compute-node" / "data-contract.md").write_text(
        "Compute-Node Data Contract\nschema_version\nmetadata\nflat_binary_uint32_le\n",
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "compute-node" / "constraints.txt").write_text(
        "furu==0.0.27\n",
        encoding="utf-8",
    )
    (tmp_path / "deploy" / "compute-node" / "requirements-cuda12.txt").write_text(
        "jax[cuda12]==0.9.1\n"
        "python -m pip install --no-deps ./refs/cs336-assignment3-scaling\n",
        encoding="utf-8",
    )
    (tmp_path / "source" / "backend" / "pyproject.toml").write_text(
        "[project]\nname = 'scaling-laws-backend'\n",
        encoding="utf-8",
    )
    (tmp_path / "source" / "data" / "pyproject.toml").write_text(
        "[project]\nname = 'scaling-laws-data'\n",
        encoding="utf-8",
    )
    (tmp_path / "source" / "training" / "pyproject.toml").write_text(
        "[project]\nname = 'scaling-laws-course-trainer'\n",
        encoding="utf-8",
    )
    (tmp_path / "source" / "training" / "course_trainer" / "worker.py").write_text(
        "StanfordLocalBackend\n"
        "build_stanford_training_config_payload\n"
        "deploy/compute-node/constraints.txt\n"
        "JAX CUDA backend could not initialize\n",
        encoding="utf-8",
    )
    (tmp_path / "source" / "training" / "course_trainer" / "tokenized_data.py").write_text(
        "def load_indexed_dataset():\n    pass\n",
        encoding="utf-8",
    )
    generated = tmp_path / "source" / "backend" / "scaling_backend" / "__pycache__"
    generated.mkdir()
    (generated / "service.cpython-312.pyc").write_bytes(b"pyc")

    generated_artifacts = release_audit._deployment_generated_artifacts(tmp_path)

    assert generated_artifacts == [
        "source/backend/scaling_backend/__pycache__",
        "source/backend/scaling_backend/__pycache__/service.cpython-312.pyc",
    ]


def test_release_audit_rejects_cuda13_resolution_in_cuda12_requirements(tmp_path):
    _write_minimal_manual_deployment_package(
        tmp_path,
        cuda12_requirements=(
            "# python -m pip install --no-deps flash-hog==0.5.0\n"
            "# python -m pip install --no-deps ./refs/cs336-assignment3-scaling\n"
            "chex==0.1.91\n"
            "jax[cuda12]==0.9.1\n"
            "flash-hog>=0.5.0\n"
            "jax[cuda13]==0.9.1\n"
        ),
    )

    check = release_audit._check_manual_deployment_package(tmp_path)

    assert check["status"] == "fail"
    assert "CUDA 12 requirements must not actively resolve" in check["message"]
    assert check["metadata"]["forbidden_active_lines"] == [
        "flash-hog>=0.5.0",
        "jax[cuda13]==0.9.1",
    ]


def test_release_audit_requires_cuda12_chex_runtime_dependency(tmp_path):
    _write_minimal_manual_deployment_package(
        tmp_path,
        cuda12_requirements=(
            "# python -m pip install --no-deps flash-hog==0.5.0\n"
            "# python -m pip install --no-deps ./refs/cs336-assignment3-scaling\n"
            "jax[cuda12]==0.9.1\n"
        ),
    )

    check = release_audit._check_manual_deployment_package(tmp_path)

    assert check["status"] == "fail"
    assert "chex==0.1.91" in check["message"]
    assert "deploy/compute-node/requirements-cuda12.txt" in check["message"]


def test_release_audit_requires_provider_probe_in_staff_docs(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "source" / "backend").mkdir(parents=True)
    staff_doc_without_probe = "\n".join(
        [
            "python -m scaling_backend.preflight",
            "python -m scaling_backend.rehearsal",
            "freeze-finals",
            "launch-finals",
            "exports --output-dir",
            "worker_events.jsonl",
            "secret",
            "Readiness Checklist",
        ]
    )
    (tmp_path / "docs" / "instructor-operations-guide.md").write_text(
        staff_doc_without_probe,
        encoding="utf-8",
    )
    (tmp_path / "source" / "backend" / "README.md").write_text(
        staff_doc_without_probe,
        encoding="utf-8",
    )

    check = release_audit._check_staff_operations_docs(tmp_path)

    assert check["status"] == "fail"
    assert "--probe-provider" in check["message"]


def test_release_audit_requires_student_key_seeding_in_staff_docs(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "source" / "backend").mkdir(parents=True)
    staff_doc_without_key_seeding = "\n".join(
        [
            "python -m scaling_backend.preflight",
            "--probe-provider",
            "python -m scaling_backend.readiness",
            "python -m scaling_backend.rehearsal",
            "freeze-finals",
            "launch-finals",
            "exports --output-dir",
            "worker_events.jsonl",
            "budget_snapshots.csv",
            "provider_logs_uri",
            "final_run_grading_outcome",
            "final_run_score_policy",
            "methodology_report_score",
            "writeup_artifact_uri",
            "secret",
            "Readiness Checklist",
        ]
    )
    (tmp_path / "docs" / "instructor-operations-guide.md").write_text(
        staff_doc_without_key_seeding,
        encoding="utf-8",
    )
    (tmp_path / "source" / "backend" / "README.md").write_text(
        staff_doc_without_key_seeding,
        encoding="utf-8",
    )

    check = release_audit._check_staff_operations_docs(tmp_path)

    assert check["status"] == "fail"
    assert "seed-student-keys" in check["message"]


def test_release_audit_requires_fair_share_queue_fields_in_staff_docs(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "source" / "backend").mkdir(parents=True)
    staff_doc_without_queue_fields = "\n".join(
        [
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
            "final_run_grading_outcome",
            "final_run_score_policy",
            "methodology_report_score",
            "writeup_artifact_uri",
            "secret",
            "Readiness Checklist",
        ]
    )
    (tmp_path / "docs" / "instructor-operations-guide.md").write_text(
        staff_doc_without_queue_fields,
        encoding="utf-8",
    )
    (tmp_path / "source" / "backend" / "README.md").write_text(
        staff_doc_without_queue_fields,
        encoding="utf-8",
    )

    check = release_audit._check_staff_operations_docs(tmp_path)

    assert check["status"] == "fail"
    assert "fair_share_order" in check["message"]


def test_release_audit_requires_budget_snapshot_export_in_staff_docs(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "source" / "backend").mkdir(parents=True)
    staff_doc_without_budget_snapshot = "\n".join(
        [
            "python -m scaling_backend.preflight",
            "--probe-provider",
            "python -m scaling_backend.readiness",
            "python -m scaling_backend.rehearsal",
            "freeze-finals",
            "launch-finals",
            "exports --output-dir",
            "worker_events.jsonl",
            "final_run_grading_outcome",
            "final_run_score_policy",
            "methodology_report_score",
            "writeup_artifact_uri",
            "secret",
            "Readiness Checklist",
        ]
    )
    (tmp_path / "docs" / "instructor-operations-guide.md").write_text(
        staff_doc_without_budget_snapshot,
        encoding="utf-8",
    )
    (tmp_path / "source" / "backend" / "README.md").write_text(
        staff_doc_without_budget_snapshot,
        encoding="utf-8",
    )

    check = release_audit._check_staff_operations_docs(tmp_path)

    assert check["status"] == "fail"
    assert "budget_snapshots.csv" in check["message"]


def test_release_audit_requires_provider_artifact_columns_in_staff_docs(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "source" / "backend").mkdir(parents=True)
    staff_doc_without_provider_artifacts = "\n".join(
        [
            "python -m scaling_backend.preflight",
            "--probe-provider",
            "python -m scaling_backend.readiness",
            "python -m scaling_backend.rehearsal",
            "freeze-finals",
            "launch-finals",
            "exports --output-dir",
            "worker_events.jsonl",
            "budget_snapshots.csv",
            "final_run_grading_outcome",
            "final_run_score_policy",
            "methodology_report_score",
            "writeup_artifact_uri",
            "secret",
            "Readiness Checklist",
        ]
    )
    (tmp_path / "docs" / "instructor-operations-guide.md").write_text(
        staff_doc_without_provider_artifacts,
        encoding="utf-8",
    )
    (tmp_path / "source" / "backend" / "README.md").write_text(
        staff_doc_without_provider_artifacts,
        encoding="utf-8",
    )

    check = release_audit._check_staff_operations_docs(tmp_path)

    assert check["status"] == "fail"
    assert "provider_logs_uri" in check["message"]


def test_release_audit_rejects_internal_statuses_in_student_quickstarts(tmp_path):
    (tmp_path / "docs").mkdir()
    stale_quickstart = "\n".join(
        [
            _student_quickstart_required_text(),
            'terminal_statuses = {"completed", "failed", "lost"}',
        ]
    )
    (tmp_path / "docs" / "scaling-laws-api-quickstart.md").write_text(
        stale_quickstart,
        encoding="utf-8",
    )
    (tmp_path / "docs" / "scaling-laws-api-quickstart.zh-CN.md").write_text(
        stale_quickstart,
        encoding="utf-8",
    )

    check = release_audit._check_student_quickstarts(tmp_path)

    assert check["status"] == "fail"
    assert "lost" in check["message"]


def test_release_audit_rejects_private_manifest_ids_in_student_quickstarts(tmp_path):
    (tmp_path / "docs").mkdir()
    leaking_quickstart = "\n".join(
        [
            _student_quickstart_required_text(),
            '"data_manifest_id": "exploratory-train-v0"',
        ]
    )
    (tmp_path / "docs" / "scaling-laws-api-quickstart.md").write_text(
        leaking_quickstart,
        encoding="utf-8",
    )
    (tmp_path / "docs" / "scaling-laws-api-quickstart.zh-CN.md").write_text(
        leaking_quickstart,
        encoding="utf-8",
    )

    check = release_audit._check_student_quickstarts(tmp_path)

    assert check["status"] == "fail"
    assert "data_manifest_id" in check["message"]


def test_release_audit_rejects_internal_paths_in_public_student_docs(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "scaling-laws-api-quickstart.md").write_text(
        "See docs/examples/scaling_laws_student_workflow.ipynb and source/backend.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "scaling-laws-project-student-handout.md").write_text(
        "Put source/backend on PYTHONPATH.\n",
        encoding="utf-8",
    )

    check = release_audit._check_public_api_stale_terms(tmp_path)

    assert check["status"] == "fail"
    assert "internal paths" in check["message"]
    assert "docs/" in check["metadata"]["matches"][
        "docs/scaling-laws-api-quickstart.md"
    ]
    assert "PYTHONPATH" in check["metadata"]["matches"][
        "docs/scaling-laws-project-student-handout.md"
    ]


def test_release_audit_rejects_removed_student_handout_strategy_terms(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "scaling-laws-project-student-handout.md").write_text(
        "## 13. Recommended Strategy\nStudents may discuss high-level ideas.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "scaling-laws-project-student-handout.zh-CN.md").write_text(
        "## 13. 推荐策略\n学生可以讨论高层想法。\n",
        encoding="utf-8",
    )

    check = release_audit._check_public_api_stale_terms(tmp_path)

    assert check["status"] == "fail"
    assert "Recommended Strategy" in check["metadata"]["matches"][
        "docs/scaling-laws-project-student-handout.md"
    ]
    assert "学生可以讨论" in check["metadata"]["matches"][
        "docs/scaling-laws-project-student-handout.zh-CN.md"
    ]


def test_release_audit_rejects_legacy_head_dim_logic_terms(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "latex").mkdir()
    (tmp_path / "docs" / "scaling-laws-project-student-handout.md").write_text(
        "The derived head dimension is not submitted.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "scaling-assignment-backend-research.md").write_text(
        "RoPE head dimension must be even.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "latex" / "scaling-laws-project-student-handout.zh-CN.tex").write_text(
        "head dimension 由该整除关系推导得到，而不是单独提交的字段。\n",
        encoding="utf-8",
    )

    check = release_audit._check_public_api_stale_terms(tmp_path)

    assert check["status"] == "fail"
    assert "derived head dimension" in check["metadata"]["matches"][
        "docs/scaling-laws-project-student-handout.md"
    ]
    assert "RoPE head dimension must be even" in check["metadata"]["matches"][
        "docs/scaling-assignment-backend-research.md"
    ]
    assert "而不是单独提交的字段" in check["metadata"]["matches"][
        "docs/latex/scaling-laws-project-student-handout.zh-CN.tex"
    ]


def test_release_audit_rejects_removed_evaluation_cadence_warning(tmp_path):
    (tmp_path / "docs" / "examples").mkdir(parents=True)
    (tmp_path / "docs" / "examples" / "scaling_laws_student_workflow.ipynb").write_text(
        '{"cells": [{"cell_type": "code", "source": ["suspicious_evaluation_cadence"]}]}',
        encoding="utf-8",
    )

    check = release_audit._check_public_api_stale_terms(tmp_path)

    assert check["status"] == "fail"
    assert "suspicious_evaluation_cadence" in check["metadata"]["matches"][
        "docs/examples/scaling_laws_student_workflow.ipynb"
    ]


def test_release_audit_rejects_unreadable_latex_code_block_styling(tmp_path):
    latex_dir = tmp_path / "docs" / "latex"
    latex_dir.mkdir(parents=True)
    _write_safe_latex_generator(latex_dir)
    old_style = "\n".join(
        [
            "\\usepackage{minted}",
            "\\usemintedstyle{monokai}",
            "\\setminted{",
            "  bgcolor=Sepia,",
            "}",
        ]
    )
    for name in [
        "scaling-laws-api-quickstart.tex",
        "scaling-laws-api-quickstart.zh-CN.tex",
        "scaling-laws-project-student-handout.tex",
        "scaling-laws-project-student-handout.zh-CN.tex",
    ]:
        (latex_dir / name).write_text(old_style, encoding="utf-8")

    check = release_audit._check_quickstart_latex_sources(tmp_path)

    assert check["status"] == "fail"
    assert "readable minted styling" in check["message"]
    missing = next(iter(check["metadata"]["missing"].values()))
    assert "\\definecolor{CodeBlockText}" in missing
    assert "\\AtBeginEnvironment{minted}{\\color{CodeBlockText}}" in missing
    assert "formatcom=\\color{CodeBlockText}" in missing
    assert "framesep=3pt" in missing


def test_release_audit_rejects_changed_latex_code_block_background(tmp_path):
    latex_dir = tmp_path / "docs" / "latex"
    latex_dir.mkdir(parents=True)
    _write_safe_latex_generator(latex_dir)
    changed_background = "\n".join(
        [
            "\\usepackage{minted}",
            "\\definecolor{CodeBlockBg}{HTML}{2B1F1F}",
            "\\definecolor{CodeBlockFrame}{HTML}{8C5A5A}",
            "\\definecolor{CodeBlockText}{HTML}{F8F8F2}",
            "\\usemintedstyle{monokai}",
            "\\AtBeginEnvironment{minted}{\\color{CodeBlockText}}",
            "\\setminted{",
            "  bgcolor=CodeBlockBg,",
            "  rulecolor=\\color{CodeBlockFrame},",
            "  formatcom=\\color{CodeBlockText},",
            "}",
        ]
    )
    for name in [
        "scaling-laws-api-quickstart.tex",
        "scaling-laws-api-quickstart.zh-CN.tex",
        "scaling-laws-project-student-handout.tex",
        "scaling-laws-project-student-handout.zh-CN.tex",
    ]:
        (latex_dir / name).write_text(changed_background, encoding="utf-8")

    check = release_audit._check_quickstart_latex_sources(tmp_path)

    assert check["status"] == "fail"
    assert "changes the original Sepia" in check["message"]
    assert "bgcolor=CodeBlockBg" in next(iter(check["metadata"]["matches"].values()))


def test_release_audit_rejects_regenerating_manual_chinese_latex_references(tmp_path):
    latex_dir = tmp_path / "docs" / "latex"
    latex_dir.mkdir(parents=True)
    _write_safe_latex_sources(latex_dir)
    (latex_dir / "generate_quickstart_latex.py").write_text(
        '"target": LATEX_DIR / "scaling-laws-project-student-handout.zh-CN.tex"\n',
        encoding="utf-8",
    )

    check = release_audit._check_quickstart_latex_sources(tmp_path)

    assert check["status"] == "fail"
    assert "must not be regenerated" in check["message"]
    assert "scaling-laws-project-student-handout.zh-CN.tex" in check["message"]


def _write_safe_latex_generator(latex_dir):
    (latex_dir / "generate_quickstart_latex.py").write_text(
        "# generated English LaTeX targets only\n",
        encoding="utf-8",
    )


def _student_quickstart_required_text():
    return "\n".join(
        [
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
    )


def _write_safe_latex_sources(latex_dir):
    safe_style = "\n".join(
        [
            "\\usepackage{minted}",
            "\\definecolor{CodeBlockText}{HTML}{F8F8F2}",
            "\\usemintedstyle{monokai}",
            "\\AtBeginEnvironment{minted}{\\color{CodeBlockText}}",
            "\\BeforeBeginEnvironment{minted}{\\vspace{-0.25\\baselineskip}}",
            "\\AfterEndEnvironment{minted}{\\vspace{-0.35\\baselineskip}}",
            "\\setlength{\\parskip}{4pt plus 1pt minus 1pt}",
            "\\setminted{",
            "  bgcolor=Sepia,",
            "  framesep=3pt,",
            "  formatcom=\\color{CodeBlockText},",
            "  numbersep=6pt,",
            "  baselinestretch=0.96,",
            "}",
        ]
    )
    for name in [
        "scaling-laws-api-quickstart.tex",
        "scaling-laws-api-quickstart.zh-CN.tex",
        "scaling-laws-project-student-handout.tex",
        "scaling-laws-project-student-handout.zh-CN.tex",
    ]:
        (latex_dir / name).write_text(safe_style, encoding="utf-8")
