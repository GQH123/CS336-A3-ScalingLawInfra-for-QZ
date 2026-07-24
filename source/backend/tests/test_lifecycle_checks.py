from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from scaling_backend.lifecycle_checks import (
    build_lifecycle_worker_manifest,
    inspect_lifecycle_tokenized_indexes,
    load_lifecycle_package,
    run_control_local_service_check,
    verify_manifest_uses_lifecycle_package,
)


def _write_index(directory: Path, tokens: list[int]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = b"".join(int(token).to_bytes(4, "little") for token in tokens)
    shard = directory / "tokens-000000.bin"
    shard.write_bytes(payload)
    index = {
        "schema_version": 1,
        "token_dtype": "uint32",
        "byte_order": "little",
        "shard_format": "flat_binary_uint32_le",
        "total_tokens": len(tokens),
        "source_token_counts": {"test": len(tokens)},
        "shards": [
            {
                "path": shard.name,
                "tokens": len(tokens),
                "dtype": "uint32",
                "byte_order": "little",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_token_counts": {"test": len(tokens)},
            }
        ],
    }
    index_path = directory / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path


def _write_lifecycle_package(root: Path) -> dict:
    train_index = _write_index(root / "tokenized" / "train", list(range(20)))
    validation_index = _write_index(root / "tokenized" / "validation", list(range(30, 38)))
    runtime_overrides = {
        "SCALING_DATA_MANIFEST_ID": "lifecycle-train-local",
        "SCALING_EVAL_MANIFEST_ID": "lifecycle-validation-local",
        "SCALING_FINAL_DATA_MANIFEST_ID": "lifecycle-final-train-local",
        "SCALING_FINAL_EVAL_MANIFEST_ID": "lifecycle-final-validation-local",
        "SCALING_VALIDATION_TOKENS_PER_EVAL": "4",
        "SCALING_FINAL_VALIDATION_TOKENS_PER_EVAL": "4",
        "SCALING_TOKENIZED_TRAIN_INDEX_URI": train_index.resolve().as_uri(),
        "SCALING_TOKENIZED_VALIDATION_INDEX_URI": validation_index.resolve().as_uri(),
        "SCALING_FINAL_TOKENIZED_TRAIN_INDEX_URI": train_index.resolve().as_uri(),
        "SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI": validation_index.resolve().as_uri(),
    }
    config = {
        "model": {
            "num_hidden_layers": 1,
            "hidden_size": 64,
            "num_attention_heads": 1,
            "dtype": "float32",
        },
        "training": {
            "train_tokens": 8,
            "sequence_length": 4,
            "train_batch_size": 1,
            "validation_batch_size": 1,
            "num_evals": 2,
            "learning_rate": 3e-4,
            "optimizer": "adamw",
            "lr_schedule": "constant",
        },
    }
    manifest = {
        "schema_version": 1,
        "output_dir": str(root),
        "tokenized_dir": str(root / "tokenized"),
        "train_index_path": str(train_index),
        "validation_index_path": str(validation_index),
        "train_index_uri": runtime_overrides["SCALING_TOKENIZED_TRAIN_INDEX_URI"],
        "validation_index_uri": runtime_overrides[
            "SCALING_TOKENIZED_VALIDATION_INDEX_URI"
        ],
        "train_total_tokens": 20,
        "validation_total_tokens": 8,
        "runtime_overrides": runtime_overrides,
        "recommended_student_config": config,
        "recommended_submit_payload": {
            "config": config,
            "requested_runtime_seconds": 300,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "lifecycle-dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "api-runtime-overrides.json").write_text(
        json.dumps(runtime_overrides, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_build_lifecycle_worker_manifest_freezes_dataset_uris_from_package(tmp_path):
    expected_package = _write_lifecycle_package(tmp_path)
    package = load_lifecycle_package(tmp_path)

    manifest = build_lifecycle_worker_manifest(
        package,
        run_id="exp-lifecycle-000001",
        student_id="staff-lifecycle",
        manifest_uri=(tmp_path / "worker-manifest.json").as_uri(),
        callback_url="http://control-node/internal/provider-events",
        code_version="test-code-version",
    )

    assert package == expected_package
    assert manifest["experiment_id"] == "exp-lifecycle-000001"
    assert manifest["student_id"] == "staff-lifecycle"
    assert manifest["data_manifest_id"] == "lifecycle-train-local"
    assert manifest["eval_manifest_id"] == "lifecycle-validation-local"
    assert manifest["code_version"] == "test-code-version"
    assert manifest["model_config"] == manifest["resolved_config"]["model"]
    assert manifest["training_config"] == manifest["resolved_config"]["training"]
    assert manifest["data_config"] == {
        "train_tokens": 8,
        "tokenized_index_uri": package["runtime_overrides"][
            "SCALING_TOKENIZED_TRAIN_INDEX_URI"
        ],
    }
    assert manifest["validation_config"] == {
        "eval_manifest_id": "lifecycle-validation-local",
        "validation_tokens_per_eval": 4,
        "validation_batches_per_eval": 1,
        "tokenized_index_uri": package["runtime_overrides"][
            "SCALING_TOKENIZED_VALIDATION_INDEX_URI"
        ],
    }

    verification = verify_manifest_uses_lifecycle_package(manifest, package)
    assert verification["ok"] is True
    assert verification["train_index_uri"] == package["train_index_uri"]
    assert verification["validation_index_uri"] == package["validation_index_uri"]


def test_inspect_lifecycle_tokenized_indexes_reads_lifecycle_shards(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "source" / "training"))
    _write_lifecycle_package(tmp_path)
    package = load_lifecycle_package(tmp_path)

    inspection = inspect_lifecycle_tokenized_indexes(package, validation_mode="full")

    assert inspection == {
        "validation_mode": "full",
        "train_index_uri": package["train_index_uri"],
        "validation_index_uri": package["validation_index_uri"],
        "train_total_tokens": 20,
        "validation_total_tokens": 8,
        "train_shards": 1,
        "validation_shards": 1,
    }


def test_control_local_service_check_submits_with_lifecycle_runtime_overrides(tmp_path):
    _write_lifecycle_package(tmp_path)
    package = load_lifecycle_package(tmp_path)

    result = run_control_local_service_check(
        package,
        env={
            "SCALING_CALLBACK_URL": "https://backend/internal/provider-events",
            "SCALING_MANIFEST_BASE_URI": "memory://manifests",
            "SCALING_TOTAL_BUDGET_SECONDS": "7200",
        },
        student_id="staff-lifecycle",
    )

    assert result["mode"] == "local-service"
    assert result["submitted"]["experiment_id"] == "exp-000001"
    assert result["submitted"]["status"] == "submitted"
    assert result["manifest_check"]["ok"] is True
    assert result["manifest"]["data_config"]["tokenized_index_uri"] == package[
        "train_index_uri"
    ]
    assert result["manifest"]["validation_config"]["tokenized_index_uri"] == package[
        "validation_index_uri"
    ]


def test_compute_lifecycle_script_writes_inspection_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[3]
    _write_lifecycle_package(tmp_path / "lifecycle")

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{root / 'source' / 'backend'}:"
        f"{root / 'source' / 'training'}:"
        f"{env.get('PYTHONPATH', '')}"
    )
    env["LIFECYCLE_DATASET_DIR"] = str(tmp_path / "lifecycle")
    env["LIFECYCLE_WORK_DIR"] = str(tmp_path / "compute-work")
    env["LIFECYCLE_SKIP_WORKER_RUN"] = "1"
    env["LIFECYCLE_TOKENIZED_INDEX_VALIDATION_MODE"] = "full"

    completed = subprocess.run(
        [str(root / "deploy" / "compute-node" / "lifecycle-test-worker-run.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    inspection = json.loads(
        (tmp_path / "compute-work" / "lifecycle-worker-inspection.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (tmp_path / "compute-work" / "lifecycle-worker-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert inspection["validation_mode"] == "full"
    assert inspection["train_total_tokens"] == 20
    assert manifest["data_config"]["tokenized_index_uri"].startswith("file://")


def test_control_lifecycle_script_runs_local_service_check(tmp_path):
    root = Path(__file__).resolve().parents[3]
    _write_lifecycle_package(tmp_path / "lifecycle")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'source' / 'backend'}:{env.get('PYTHONPATH', '')}"
    env["LIFECYCLE_DATASET_DIR"] = str(tmp_path / "lifecycle")
    env["LIFECYCLE_API_MODE"] = "local-service"
    env["SCALING_CALLBACK_URL"] = "https://backend/internal/provider-events"
    env["SCALING_MANIFEST_BASE_URI"] = "memory://manifests"
    env["SCALING_TOTAL_BUDGET_SECONDS"] = "7200"

    completed = subprocess.run(
        [str(root / "deploy" / "control-node" / "lifecycle-test-api.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["mode"] == "local-service"
    assert result["manifest_check"]["ok"] is True
