import re

import pytest

from scaling_backend.config_validation import resolve_training_config


def _config(**overrides):
    config = {
        "model": {
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
            "vocab_size": 50_432,
        },
        "training": {
            "train_tokens": 983_040,
            "sequence_length": 1024,
            "train_batch_size": 16,
            "validation_batch_size": 8,
            "num_evals": 20,
            "model_seed": 1234,
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
    }
    for section, values in overrides.items():
        config[section].update(values)
    return config


def test_resolve_training_config_computes_reproducibility_metadata():
    resolved = resolve_training_config(
        _config(),
        requested_runtime_seconds=3600,
        code_version="test-code-version",
        data_manifest_id="train-100b-v1",
        eval_manifest_id="eval-public-v1",
        validation_tokens_per_eval=262_144,
    )

    assert resolved["parameter_count_estimate"] == 16_058_624
    assert resolved["non_embedding_parameter_count_estimate"] == 3_148_032
    assert resolved["tokens_per_optimizer_step"] == 16_384
    assert resolved["total_optimizer_steps"] == 60
    assert resolved["eval_interval_steps"] == 3
    assert resolved["validation_tokens_per_eval"] == 262_144
    assert resolved["validation_batches_per_eval"] == 32
    assert resolved["estimated_training_flops"] == 94_717_618_421_760
    assert resolved["model"] == {
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
        "vocab_size": 50_432,
    }
    assert resolved["training"] == {
        "train_tokens": 983_040,
        "sequence_length": 1024,
        "train_batch_size": 16,
        "validation_batch_size": 8,
        "num_evals": 20,
        "model_seed": 1234,
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
    }
    assert resolved["code_version"] == "test-code-version"
    assert resolved["data_manifest_id"] == "train-100b-v1"
    assert resolved["eval_manifest_id"] == "eval-public-v1"
    assert resolved["resource_warnings"] == []


def test_resolve_training_config_accepts_stanford_reference_model_schema():
    config = _config(
        model={
            "attention_bias": False,
            "head_dim": 64,
            "hidden_size": 448,
            "intermediate_size": 1280,
            "num_attention_heads": 7,
            "num_hidden_layers": 9,
            "num_key_value_heads": 7,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000,
            "tie_word_embeddings": False,
            "dtype": "bfloat16",
            "vocab_size": 50_432,
        }
    )

    resolved = resolve_training_config(
        config,
        requested_runtime_seconds=3600,
        code_version="test-code-version",
        data_manifest_id="train-100b-v1",
        eval_manifest_id="eval-public-v1",
        validation_tokens_per_eval=262_144,
    )

    assert resolved["model"] == config["model"]


def test_resolve_training_config_defaults_vocab_size_to_gpt_neox_tokenizer():
    config = _config()
    del config["model"]["vocab_size"]

    resolved = resolve_training_config(
        config,
        requested_runtime_seconds=3600,
        code_version="test-code-version",
        data_manifest_id="train-100b-v1",
        eval_manifest_id="eval-public-v1",
        validation_tokens_per_eval=262_144,
    )

    assert resolved["model"]["vocab_size"] == 50_432


def test_resolve_training_config_rejects_vocab_smaller_than_gpt_neox_tokenizer():
    with pytest.raises(ValueError, match="model.vocab_size must be at least 50432"):
        resolve_training_config(
            _config(model={"vocab_size": 32_000}),
            requested_runtime_seconds=3600,
            code_version="test-code-version",
            data_manifest_id="train-100b-v1",
            eval_manifest_id="eval-public-v1",
            validation_tokens_per_eval=262_144,
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {
                **_config(),
                "runtime": {"requested_runtime_seconds": 600},
            },
            "config contains unsupported field",
        ),
        (
            _config(model={"unsupported_model_field": 256}),
            "model contains unsupported field",
        ),
        (
            _config(training={"total_train_tokens": 983_040}),
            "training contains unsupported field",
        ),
    ],
)
def test_resolve_training_config_rejects_unknown_fields(config, message):
    with pytest.raises(ValueError, match=message):
        resolve_training_config(
            config,
            requested_runtime_seconds=3600,
            code_version="test-code-version",
            data_manifest_id="train-100b-v1",
            eval_manifest_id="eval-public-v1",
            validation_tokens_per_eval=262_144,
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("model", "num_hidden_layers", 0, "model.num_hidden_layers must be positive"),
        (
            "model",
            "hidden_size",
            255,
            r"model.hidden_size must equal model.num_attention_heads \* model.head_dim",
        ),
        ("training", "train_tokens", 0, "training.train_tokens must be positive"),
        (
            "training",
            "train_tokens",
            500_000_000_001,
            "training.train_tokens must be <= 500000000000",
        ),
        ("training", "learning_rate", 0, "training.learning_rate"),
        ("training", "learning_rate", float("inf"), "finite"),
        ("training", "warmup_fraction", 1, "training.warmup_fraction"),
        ("training", "warmup_fraction", 1.2, "training.warmup_fraction"),
        ("training", "weight_decay", -0.1, "training.weight_decay"),
        ("training", "adam_beta1", 1.2, "training.adam_beta1"),
        ("training", "adam_epsilon", 0, "training.adam_epsilon"),
        ("training", "gradient_clip_norm", 0, "training.gradient_clip_norm"),
        ("training", "model_seed", True, "training.model_seed"),
    ],
)
def test_resolve_training_config_rejects_invalid_core_fields(
    section, field, value, message
):
    config = _config(**{section: {field: value}})

    with pytest.raises(ValueError, match=message):
        resolve_training_config(
            config,
            requested_runtime_seconds=3600,
            code_version="test-code-version",
            data_manifest_id="train-100b-v1",
            eval_manifest_id="eval-public-v1",
            validation_tokens_per_eval=262_144,
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (
            "model",
            "num_key_value_heads",
            3,
            "model.num_attention_heads must be divisible by model.num_key_value_heads",
        ),
        ("model", "rms_norm_eps", 0, "model.rms_norm_eps"),
        ("model", "rope_theta", float("nan"), "model.rope_theta"),
        ("model", "dtype", "float64", "model.dtype"),
        ("model", "dtype", "float16", "model.dtype"),
        ("model", "attention_bias", "false", "model.attention_bias must be boolean"),
        ("training", "optimizer", "rmsprop", "training.optimizer"),
        ("training", "lr_schedule", "polynomial", "training.lr_schedule"),
        ("training", "lr_schedule", "linear", "training.lr_schedule"),
    ],
)
def test_resolve_training_config_rejects_invalid_extended_fields(
    section, field, value, message
):
    config = _config(**{section: {field: value}})

    with pytest.raises(ValueError, match=message):
        resolve_training_config(
            config,
            requested_runtime_seconds=3600,
            code_version="test-code-version",
            data_manifest_id="train-100b-v1",
            eval_manifest_id="eval-public-v1",
            validation_tokens_per_eval=262_144,
        )


def test_resolve_training_config_rejects_gqa_for_stanford_reference_trainer():
    with pytest.raises(ValueError, match="GQA is not supported"):
        resolve_training_config(
            _config(model={"num_key_value_heads": 4}),
            requested_runtime_seconds=3600,
            code_version="test-code-version",
            data_manifest_id="train-100b-v1",
            eval_manifest_id="eval-public-v1",
            validation_tokens_per_eval=262_144,
        )


def test_resolve_training_config_rejects_intermediate_size_below_hidden_size():
    with pytest.raises(
        ValueError,
        match=re.escape("model.intermediate_size must be >= model.hidden_size"),
    ):
        resolve_training_config(
            _config(model={"intermediate_size": 128}),
            requested_runtime_seconds=3600,
            code_version="test-code-version",
            data_manifest_id="train-100b-v1",
            eval_manifest_id="eval-public-v1",
            validation_tokens_per_eval=262_144,
        )


@pytest.mark.parametrize(
    ("model_overrides", "message"),
    [
        (
            {
                "head_dim": 2048,
                "hidden_size": 2048,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
            },
            "got 2048",
        ),
        (
            {
                "head_dim": 35,
                "hidden_size": 280,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
            },
            "got 35",
        ),
    ],
)
def test_resolve_training_config_rejects_unsupported_stanford_attention_head_dim(
    model_overrides,
    message,
):
    with pytest.raises(ValueError) as exc_info:
        resolve_training_config(
            _config(model=model_overrides),
            requested_runtime_seconds=3600,
            code_version="test-code-version",
            data_manifest_id="train-100b-v1",
            eval_manifest_id="eval-public-v1",
            validation_tokens_per_eval=262_144,
        )

    error = str(exc_info.value)
    assert "model.head_dim must be <= 128 and a multiple of 8" in error
    assert message in error
    assert "model.num_attention_heads" in error


def test_resolve_training_config_warns_on_very_low_step_count():
    resolved = resolve_training_config(
        _config(
            training={"train_tokens": 32_768, "train_batch_size": 32, "num_evals": 1}
        ),
        requested_runtime_seconds=3600,
        code_version="test-code-version",
        data_manifest_id="train-100b-v1",
        eval_manifest_id="eval-public-v1",
        validation_tokens_per_eval=262_144,
    )

    assert resolved["total_optimizer_steps"] == 1
    assert resolved["resource_warnings"] == [
        "very_low_optimizer_steps",
    ]


@pytest.mark.parametrize(
    ("training_overrides", "message"),
    [
        (
            {"train_tokens": 1_000_001},
            "training.train_tokens must be divisible by sequence_length * train_batch_size",
        ),
        (
            {"num_evals": 40},
            "total optimizer steps must be divisible by training.num_evals",
        ),
        (
            {"train_tokens": 32_768, "train_batch_size": 32, "num_evals": 20},
            "total optimizer steps must be divisible by training.num_evals",
        ),
    ],
)
def test_resolve_training_config_rejects_silent_step_rounding(
    training_overrides,
    message,
):
    with pytest.raises(ValueError, match=re.escape(message)):
        resolve_training_config(
            _config(training=training_overrides),
            requested_runtime_seconds=3600,
            code_version="test-code-version",
            data_manifest_id="train-100b-v1",
            eval_manifest_id="eval-public-v1",
            validation_tokens_per_eval=262_144,
        )


def test_resolve_training_config_rejects_validation_token_rounding():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "validation_tokens_per_eval must be divisible by "
            "sequence_length * validation_batch_size"
        ),
    ):
        resolve_training_config(
            _config(),
            requested_runtime_seconds=3600,
            code_version="test-code-version",
            data_manifest_id="train-100b-v1",
            eval_manifest_id="eval-public-v1",
            validation_tokens_per_eval=262_143,
        )


@pytest.mark.parametrize(
    ("training_overrides", "message"),
    [
        (
            {"train_batch_size": 1, "validation_batch_size": 2},
            "training.train_batch_size must be divisible by worker_gpu_count",
        ),
        (
            {"train_batch_size": 16, "validation_batch_size": 1},
            "training.validation_batch_size must be divisible by worker_gpu_count",
        ),
    ],
)
def test_resolve_training_config_rejects_batches_not_divisible_by_worker_gpu_count(
    training_overrides,
    message,
):
    with pytest.raises(ValueError, match=re.escape(message)):
        resolve_training_config(
            _config(training=training_overrides),
            requested_runtime_seconds=3600,
            code_version="test-code-version",
            data_manifest_id="train-100b-v1",
            eval_manifest_id="eval-public-v1",
            validation_tokens_per_eval=262_144,
            worker_gpu_count=2,
        )


@pytest.mark.parametrize(
    ("model_overrides", "worker_gpu_count", "message"),
    [
        (
            {"vocab_size": 50_433},
            2,
            "model.vocab_size must be divisible by worker_gpu_count",
        ),
        (
            {"intermediate_size": 1025},
            2,
            "model.intermediate_size must be divisible by worker_gpu_count",
        ),
        (
            {
                "head_dim": 32,
                "hidden_size": 288,
                "intermediate_size": 1280,
                "num_attention_heads": 9,
                "num_key_value_heads": 9,
                "vocab_size": 50_435,
            },
            5,
            "model.hidden_size must be divisible by worker_gpu_count",
        ),
    ],
)
def test_resolve_training_config_rejects_model_axes_not_divisible_by_worker_gpu_count(
    model_overrides,
    worker_gpu_count,
    message,
):
    with pytest.raises(ValueError, match=re.escape(message)):
        resolve_training_config(
            _config(
                model=model_overrides,
                training={
                    "train_batch_size": 20,
                    "validation_batch_size": 10,
                    "num_evals": 16,
                },
            ),
            requested_runtime_seconds=3600,
            code_version="test-code-version",
            data_manifest_id="train-100b-v1",
            eval_manifest_id="eval-public-v1",
            validation_tokens_per_eval=204_800,
            worker_gpu_count=worker_gpu_count,
        )


def test_resolve_training_config_allows_nondivisible_vocab_when_embeddings_are_tied():
    resolved = resolve_training_config(
        _config(
            model={"tie_word_embeddings": True, "vocab_size": 50_433},
            training={
                "train_batch_size": 20,
                "validation_batch_size": 10,
                "num_evals": 16,
            },
        ),
        requested_runtime_seconds=3600,
        code_version="test-code-version",
        data_manifest_id="train-100b-v1",
        eval_manifest_id="eval-public-v1",
        validation_tokens_per_eval=204_800,
        worker_gpu_count=2,
    )

    assert resolved["model"]["vocab_size"] == 50_433
    assert resolved["model"]["tie_word_embeddings"] is True


def test_resolve_training_config_accepts_batches_divisible_by_worker_gpu_count():
    resolved = resolve_training_config(
        _config(),
        requested_runtime_seconds=3600,
        code_version="test-code-version",
        data_manifest_id="train-100b-v1",
        eval_manifest_id="eval-public-v1",
        validation_tokens_per_eval=262_144,
        worker_gpu_count=2,
    )

    assert resolved["training"]["train_batch_size"] == 16
    assert resolved["training"]["validation_batch_size"] == 8
