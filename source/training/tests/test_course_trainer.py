from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import pytest

from course_trainer.tokenized_data import (
    IndexedTokenDataset,
    TokenizedDatasetError,
    load_indexed_dataset,
    load_training_datasets,
)
from course_trainer.worker import (
    CourseTrainerConfigError,
    _jax_device_count,
    _install_stanford_schema_import_shim,
    _train_batch,
    _validate_batch_sharding_compatibility,
    _validate_parameter_sharding_compatibility,
    _with_config_error_for_known_jax_sharding_failure,
    build_stanford_training_config_payload,
    run_with_backend,
)


def _manifest(tmp_path: Path) -> dict:
    train_index = _write_index(tmp_path / "train", [1, 2, 3, 4, 5, 6, 7, 8])
    validation_index = _write_index(tmp_path / "validation", [9, 10, 11, 12, 13])
    return {
        "experiment_id": "exp-000001",
        "student_id": "student-1",
        "model_config": {
            "attention_bias": False,
            "head_dim": 32,
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000,
            "tie_word_embeddings": False,
            "dtype": "bfloat16",
            "vocab_size": 32000,
        },
        "training_config": {
            "train_tokens": 4096,
            "sequence_length": 512,
            "train_batch_size": 2,
            "validation_batch_size": 1,
            "num_evals": 4,
            "model_seed": 123,
            "learning_rate": 3e-4,
            "optimizer": "adamw",
            "lr_schedule": "cosine",
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "warmup_fraction": 0.05,
            "final_lr_fraction": 0.1,
            "gradient_clip_norm": 1.0,
        },
        "data_config": {
            "train_tokens": 4096,
            "tokenized_index_uri": train_index.as_uri(),
        },
        "validation_config": {
            "eval_manifest_id": "exploratory-eval-v0",
            "validation_tokens_per_eval": 512,
            "validation_batches_per_eval": 1,
            "tokenized_index_uri": validation_index.as_uri(),
        },
        "runtime_config": {"max_runtime_seconds": 300},
    }


def _write_index(directory: Path, tokens: list[int]) -> Path:
    directory.mkdir(parents=True)
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


def test_build_stanford_training_config_payload_maps_manifest_names(tmp_path):
    payload = build_stanford_training_config_payload(_manifest(tmp_path))

    assert payload == {
        "architecture_config": {
            "attention_bias": False,
            "head_dim": 32,
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000,
            "tie_word_embeddings": False,
            "dtype": "bfloat16",
            "vocab_size": 32000,
        },
        "optimizer_config": {
            "lr_scheduler": {
                "peak_value": 3e-4,
                "final_lr_frac": 0.1,
                "warmup_frac": 0.05,
                "init_value": 0.0,
            },
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-8,
            "eps_root": 1e-8,
            "grad_clip_norm": 1.0,
        },
        "train_batch_size": 2,
        "val_batch_size": 1,
        "n_evals": 4,
        "total_train_tokens": 4096,
        "max_runtime_seconds": 300,
        "model_seed": 123,
    }


def test_build_stanford_training_config_payload_rejects_unsupported_reference_options(
    tmp_path,
):
    manifest = _manifest(tmp_path)
    manifest["model_config"]["num_key_value_heads"] = 4

    with pytest.raises(CourseTrainerConfigError, match="GQA is not supported"):
        build_stanford_training_config_payload(manifest)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (
            "model_config",
            "head_dim",
            2048,
            "requires head_dim <= 128 and a multiple of 8",
        ),
        (
            "model_config",
            "head_dim",
            36,
            "requires head_dim <= 128 and a multiple of 8",
        ),
        (
            "model_config",
            "dtype",
            "float16",
            "model_config.dtype must be one of",
        ),
        (
            "training_config",
            "lr_schedule",
            "linear",
            "does not support lr_schedule='linear'",
        ),
    ],
)
def test_build_stanford_training_config_payload_rejects_other_runtime_limits(
    tmp_path,
    section,
    field,
    value,
    message,
):
    manifest = _manifest(tmp_path)
    manifest[section][field] = value

    with pytest.raises(CourseTrainerConfigError, match=message):
        build_stanford_training_config_payload(manifest)


def test_load_indexed_dataset_reads_flat_binary_uint32_shards_on_demand(tmp_path):
    index_path = _write_index(tmp_path / "train", [7, 8, 9, 10, 11])

    dataset = load_indexed_dataset(index_path.as_uri())

    assert isinstance(dataset, IndexedTokenDataset)
    assert dataset.total_tokens == 5
    assert dataset.read_range(1, 4).tolist() == [8, 9, 10]


def test_load_indexed_dataset_rejects_same_size_hash_mismatch(tmp_path):
    index_path = _write_index(tmp_path / "train", [7, 8, 9, 10, 11])
    shard_path = index_path.parent / "tokens-000000.bin"
    shard_path.write_bytes(b"\x00" * shard_path.stat().st_size)

    with pytest.raises(TokenizedDatasetError, match="hash mismatch"):
        load_indexed_dataset(index_path.as_uri(), validation_mode="full")


def test_run_with_backend_invokes_stanford_backend_and_emits_validation_events(tmp_path):
    manifest = _manifest(tmp_path)
    emitted = []

    class FakeBackend:
        def __init__(self):
            self.calls = []

        def run(self, *, manifest, training_config_payload, dataset):
            self.calls.append(
                {
                    "manifest": manifest,
                    "training_config_payload": training_config_payload,
                    "dataset": dataset,
                }
            )
            return {
                "validation_losses": [4.2, 3.8],
                "final_validation_loss": 3.8,
                "actual_runtime_seconds": 120,
            }

    backend = FakeBackend()

    result = run_with_backend(manifest, emitted.append, backend=backend)

    assert result == {
        "validation_losses": [4.2, 3.8],
        "final_validation_loss": 3.8,
        "actual_runtime_seconds": 120,
    }
    assert backend.calls[0]["manifest"] is manifest
    assert backend.calls[0]["training_config_payload"]["total_train_tokens"] == 4096
    assert backend.calls[0]["dataset"].train.total_tokens == 8
    assert backend.calls[0]["dataset"].validation.total_tokens == 5
    assert emitted == [
        {"event_type": "validation", "step": 1, "loss": 4.2},
        {"event_type": "validation", "step": 2, "loss": 3.8},
    ]


def test_run_with_backend_rejects_non_finite_validation_loss_before_emit(tmp_path):
    manifest = _manifest(tmp_path)
    emitted = []

    class FakeBackend:
        def run(self, *, manifest, training_config_payload, dataset):
            return {
                "validation_losses": [math.nan],
                "final_validation_loss": math.nan,
                "actual_runtime_seconds": 120,
            }

    with pytest.raises(
        CourseTrainerConfigError,
        match=r"validation_losses\[0\] must be finite",
    ):
        run_with_backend(manifest, emitted.append, backend=FakeBackend())

    assert emitted == []


def test_train_batch_rejects_token_ids_outside_model_vocab_before_jax(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["model_config"]["vocab_size"] = 32_000
    manifest["training_config"].update(
        {
            "train_tokens": 8,
            "sequence_length": 4,
            "train_batch_size": 1,
            "validation_batch_size": 1,
            "num_evals": 2,
        }
    )
    manifest["data_config"]["train_tokens"] = 8
    manifest["data_config"]["tokenized_index_uri"] = _write_index(
        tmp_path / "high-token-train",
        [1, 40_000, 2, 3, 4, 5, 6, 7, 8],
    ).as_uri()
    dataset = load_training_datasets(manifest)

    class SimpleTrainingConfig:
        seq_len = 4
        eval_every_tokens = 8
        model_seed = 0

    class SimpleBatch:
        def __init__(self, *, input_ids, labels):
            self.input_ids = input_ids
            self.labels = labels

    with pytest.raises(CourseTrainerConfigError) as exc_info:
        _train_batch(
            dataset,
            SimpleTrainingConfig(),
            SimpleBatch,
            chunk_idx=0,
            vocab_size=32_000,
        )

    message = str(exc_info.value)
    assert "training token ids must be < model_config.vocab_size" in message
    assert "max token id 40000" in message
    assert "model_config.vocab_size=32000" in message


def test_jax_device_count_reports_cuda_driver_mismatch_with_install_guidance():
    class FakeJax:
        @staticmethod
        def devices():
            raise RuntimeError(
                "Unable to initialize backend 'cuda': INTERNAL: "
                "no supported devices found for platform CUDA: "
                "cudaErrorInsufficientDriver"
            )

    with pytest.raises(RuntimeError) as exc_info:
        _jax_device_count(FakeJax)

    message = str(exc_info.value)
    assert "JAX CUDA backend could not initialize" in message
    assert "jax[cuda13]==0.9.1" in message
    assert "CUDA 12 compute-node install path" in message


def test_jax_device_count_rejects_cpu_fallback_with_plugin_cleanup_guidance():
    class FakeCpuDevice:
        platform = "cpu"

        def __repr__(self):
            return "CpuDevice(id=0)"

    class FakeJax:
        @staticmethod
        def devices():
            return [FakeCpuDevice()]

    with pytest.raises(RuntimeError) as exc_info:
        _jax_device_count(FakeJax)

    message = str(exc_info.value)
    assert "JAX did not expose any CUDA devices" in message
    assert "CpuDevice(id=0)" in message
    assert "PJRT_Api already exists" in message
    assert "jax-cuda12/jax-cuda13" in message


def test_jax_device_count_rejects_duplicate_cuda_plugins(monkeypatch):
    def fake_find_spec(module_name):
        if module_name in {"jax_plugins.xla_cuda12", "jax_plugins.xla_cuda13"}:
            return object()
        return None

    class FakeCudaDevice:
        platform = "cuda"

    class FakeJax:
        @staticmethod
        def devices():
            return [FakeCudaDevice()]

    monkeypatch.setattr(
        "course_trainer.worker.importlib.util.find_spec",
        fake_find_spec,
    )

    with pytest.raises(RuntimeError) as exc_info:
        _jax_device_count(FakeJax)

    message = str(exc_info.value)
    assert "Multiple JAX CUDA PJRT plugins are installed" in message
    assert "xla_cuda12" in message
    assert "xla_cuda13" in message
    assert "PJRT_Api already exists" in message


@pytest.mark.parametrize(
    ("train_batch_size", "val_batch_size", "message"),
    [
        (1, 2, "training_config.train_batch_size=1"),
        (2, 1, "training_config.validation_batch_size=1"),
    ],
)
def test_validate_batch_sharding_compatibility_rejects_nondivisible_batches(
    train_batch_size,
    val_batch_size,
    message,
):
    class FakeTrainingConfig:
        pass

    training_config = FakeTrainingConfig()
    training_config.train_batch_size = train_batch_size
    training_config.val_batch_size = val_batch_size

    with pytest.raises(CourseTrainerConfigError) as exc_info:
        _validate_batch_sharding_compatibility(training_config, device_count=2)

    error = str(exc_info.value)
    assert message in error
    assert "JAX CUDA device count 2" in error
    assert "multiples of 2" in error


@pytest.mark.parametrize(
    (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "tie_word_embeddings",
        "device_count",
        "message",
    ),
    [
        (
            50_433,
            256,
            1024,
            False,
            2,
            "model_config.vocab_size=50433",
        ),
        (
            50_432,
            256,
            1025,
            False,
            2,
            "model_config.intermediate_size=1025",
        ),
        (
            50_435,
            288,
            1280,
            False,
            5,
            "model_config.hidden_size=288",
        ),
    ],
)
def test_validate_parameter_sharding_compatibility_rejects_nondivisible_model_axes(
    vocab_size,
    hidden_size,
    intermediate_size,
    tie_word_embeddings,
    device_count,
    message,
):
    class FakeArchitectureConfig:
        pass

    class FakeTrainingConfig:
        pass

    architecture = FakeArchitectureConfig()
    architecture.vocab_size = vocab_size
    architecture.hidden_size = hidden_size
    architecture.intermediate_size = intermediate_size
    architecture.tie_word_embeddings = tie_word_embeddings
    training_config = FakeTrainingConfig()
    training_config.architecture_config = architecture

    with pytest.raises(CourseTrainerConfigError) as exc_info:
        _validate_parameter_sharding_compatibility(
            training_config, device_count=device_count
        )

    error = str(exc_info.value)
    assert message in error
    assert f"JAX CUDA device count {device_count}" in error
    assert "Stanford trainer shards model parameter axes across the fsdp mesh" in error


def test_validate_parameter_sharding_compatibility_allows_nondivisible_tied_vocab():
    class FakeArchitectureConfig:
        vocab_size = 50_433
        hidden_size = 256
        intermediate_size = 1024
        tie_word_embeddings = True

    class FakeTrainingConfig:
        architecture_config = FakeArchitectureConfig()

    _validate_parameter_sharding_compatibility(FakeTrainingConfig(), device_count=2)


def test_known_jax_sharding_value_error_is_recast_as_config_error():
    original = ValueError(
        "Sharding spec ('fsdp',) implies that array axis 0 is partitioned 2 "
        "times, but does not evenly divide the dimension size 50431. Got "
        "shape: (50431, 256) and sharding NamedSharding(mesh=AbstractMesh"
        "('fsdp': 2), spec=P('fsdp', None))"
    )

    with pytest.raises(CourseTrainerConfigError) as exc_info:
        with _with_config_error_for_known_jax_sharding_failure():
            raise original

    error = str(exc_info.value)
    assert "Stanford/JAX FSDP sharding rejected this runtime config" in error
    assert "does not evenly divide the dimension size 50431" in error


def test_stanford_schema_import_shim_allows_reference_training_config_import(
    monkeypatch,
):
    root = Path(__file__).resolve().parents[3]
    refs = root / "refs" / "cs336-assignment3-scaling"
    monkeypatch.syspath_prepend(str(refs))
    for name in list(sys.modules):
        if name == "cs336_scaling" or name.startswith("cs336_scaling."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    _install_stanford_schema_import_shim()
    module = importlib.import_module("cs336_scaling.training.training_config")

    assert module.TrainingConfig.__name__ == "TrainingConfig"
