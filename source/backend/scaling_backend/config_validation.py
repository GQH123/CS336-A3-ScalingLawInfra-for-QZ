from __future__ import annotations

import math
from typing import Any, Mapping


DEFAULT_MODEL = {
    "attention_bias": False,
    "num_attention_heads": 1,
    "num_key_value_heads": None,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000,
    "tie_word_embeddings": False,
    "dtype": "bfloat16",
    "intermediate_size": None,
    "vocab_size": 50_432,
}
DATASET_TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
DATASET_TOKENIZER_VOCAB_SIZE = 50_432
MAX_TRAIN_TOKENS = 500_000_000_000

DEFAULT_TRAINING = {
    "sequence_length": 1024,
    "train_batch_size": 1,
    "validation_batch_size": 1,
    "num_evals": 10,
    "model_seed": 0,
    "optimizer": "adamw",
    "lr_schedule": "cosine",
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1e-8,
    "gradient_clip_norm": 1.0,
}

ALLOWED_DTYPES = frozenset({"float32", "bfloat16"})
ALLOWED_OPTIMIZERS = frozenset({"adamw", "sgd"})
ALLOWED_LR_SCHEDULES = frozenset({"cosine", "constant"})
TOP_LEVEL_FIELDS = frozenset({"model", "training"})
MODEL_FIELDS = frozenset(
    {
        "attention_bias",
        "head_dim",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "tie_word_embeddings",
        "dtype",
        "vocab_size",
    }
)
TRAINING_FIELDS = frozenset(
    {
        "train_tokens",
        "sequence_length",
        "train_batch_size",
        "validation_batch_size",
        "num_evals",
        "model_seed",
        "learning_rate",
        "optimizer",
        "lr_schedule",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "warmup_fraction",
        "final_lr_fraction",
        "gradient_clip_norm",
    }
)


def resolve_training_config(
    config: Mapping[str, Any],
    *,
    requested_runtime_seconds: int,
    code_version: str,
    data_manifest_id: str,
    eval_manifest_id: str,
    validation_tokens_per_eval: int,
    worker_gpu_count: int = 0,
) -> dict[str, Any]:
    _reject_unknown_keys(config, TOP_LEVEL_FIELDS, "config")
    model_config = _require_mapping(config, "model")
    training_config = _require_mapping(config, "training")
    _reject_unknown_keys(model_config, MODEL_FIELDS, "model")
    _reject_unknown_keys(training_config, TRAINING_FIELDS, "training")

    num_hidden_layers = _positive_int(model_config, "model.num_hidden_layers")
    hidden_size = _positive_int(model_config, "model.hidden_size")
    num_attention_heads = _positive_int(
        model_config,
        "model.num_attention_heads",
        default=DEFAULT_MODEL["num_attention_heads"],
    )
    if "head_dim" in model_config:
        head_dim = _positive_int(model_config, "model.head_dim")
        if hidden_size != num_attention_heads * head_dim:
            raise ValueError(
                "model.hidden_size must equal "
                "model.num_attention_heads * model.head_dim"
            )
    else:
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "model.hidden_size must be divisible by model.num_attention_heads"
            )
        head_dim = hidden_size // num_attention_heads
    num_key_value_heads = _positive_int(
        model_config,
        "model.num_key_value_heads",
        default=int(DEFAULT_MODEL["num_key_value_heads"] or num_attention_heads),
    )
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            "model.num_attention_heads must be divisible by "
            "model.num_key_value_heads"
        )
    if num_key_value_heads != num_attention_heads:
        raise ValueError(
            "model.num_key_value_heads must equal model.num_attention_heads; "
            "GQA is not supported by the Stanford reference trainer"
        )
    if head_dim > 128 or head_dim % 8 != 0:
        raise ValueError(
            "model.head_dim must be <= 128 and a multiple of 8 for Stanford GPU "
            "attention; got "
            f"{head_dim}. Set model.head_dim explicitly or choose "
            "model.hidden_size and model.num_attention_heads so their quotient "
            "satisfies this constraint."
        )
    attention_bias = _optional_bool(
        model_config,
        "model.attention_bias",
        default=DEFAULT_MODEL["attention_bias"],
    )
    rms_norm_eps = _positive_finite_float(
        model_config,
        "model.rms_norm_eps",
        default=DEFAULT_MODEL["rms_norm_eps"],
    )
    rope_theta = _positive_int(
        model_config,
        "model.rope_theta",
        default=DEFAULT_MODEL["rope_theta"],
    )
    tie_word_embeddings = _optional_bool(
        model_config,
        "model.tie_word_embeddings",
        default=DEFAULT_MODEL["tie_word_embeddings"],
    )
    dtype = _choice(
        model_config,
        "model.dtype",
        allowed=ALLOWED_DTYPES,
        default=DEFAULT_MODEL["dtype"],
    )
    intermediate_size = _positive_int(
        model_config,
        "model.intermediate_size",
        default=int(DEFAULT_MODEL["intermediate_size"] or 4 * hidden_size),
    )
    if intermediate_size < hidden_size:
        raise ValueError("model.intermediate_size must be >= model.hidden_size")
    vocab_size = _positive_int(
        model_config, "model.vocab_size", default=DEFAULT_MODEL["vocab_size"]
    )
    if vocab_size < DATASET_TOKENIZER_VOCAB_SIZE:
        raise ValueError(
            "model.vocab_size must be at least "
            f"{DATASET_TOKENIZER_VOCAB_SIZE} for the current "
            f"{DATASET_TOKENIZER_NAME} tokenized dataset"
        )

    train_tokens = _positive_int(training_config, "training.train_tokens")
    if train_tokens > MAX_TRAIN_TOKENS:
        raise ValueError(f"training.train_tokens must be <= {MAX_TRAIN_TOKENS}")
    sequence_length = _positive_int(
        training_config,
        "training.sequence_length",
        default=DEFAULT_TRAINING["sequence_length"],
    )
    train_batch_size = _positive_int(
        training_config,
        "training.train_batch_size",
        default=DEFAULT_TRAINING["train_batch_size"],
    )
    validation_batch_size = _positive_int(
        training_config,
        "training.validation_batch_size",
        default=DEFAULT_TRAINING["validation_batch_size"],
    )
    num_evals = _positive_int(
        training_config, "training.num_evals", default=DEFAULT_TRAINING["num_evals"]
    )
    model_seed = _non_negative_int(
        training_config,
        "training.model_seed",
        default=DEFAULT_TRAINING["model_seed"],
    )
    learning_rate = _positive_finite_float(
        training_config, "training.learning_rate", default=None
    )
    optimizer = _choice(
        training_config,
        "training.optimizer",
        allowed=ALLOWED_OPTIMIZERS,
        default=DEFAULT_TRAINING["optimizer"],
    )
    lr_schedule = _choice(
        training_config,
        "training.lr_schedule",
        allowed=ALLOWED_LR_SCHEDULES,
        default=DEFAULT_TRAINING["lr_schedule"],
    )
    weight_decay = _non_negative_finite_float(
        training_config,
        "training.weight_decay",
        default=DEFAULT_TRAINING["weight_decay"],
    )
    adam_beta1 = _bounded_fraction(
        training_config,
        "training.adam_beta1",
        default=DEFAULT_TRAINING["adam_beta1"],
        upper_inclusive=False,
    )
    adam_beta2 = _bounded_fraction(
        training_config,
        "training.adam_beta2",
        default=DEFAULT_TRAINING["adam_beta2"],
        upper_inclusive=False,
    )
    adam_epsilon = _positive_finite_float(
        training_config,
        "training.adam_epsilon",
        default=DEFAULT_TRAINING["adam_epsilon"],
    )
    warmup_fraction = _bounded_fraction(
        training_config,
        "training.warmup_fraction",
        default=0.0,
        upper_inclusive=False,
    )
    final_lr_fraction = _bounded_fraction(
        training_config, "training.final_lr_fraction", default=0.0
    )
    gradient_clip_norm = _positive_finite_float(
        training_config,
        "training.gradient_clip_norm",
        default=DEFAULT_TRAINING["gradient_clip_norm"],
    )
    gpu_count = _non_negative_external_int(worker_gpu_count, "worker_gpu_count")
    _validate_model_sharding_compatibility(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        tie_word_embeddings=tie_word_embeddings,
        worker_gpu_count=gpu_count,
    )

    tokens_per_step = sequence_length * train_batch_size
    if train_tokens % tokens_per_step != 0:
        raise ValueError(
            "training.train_tokens must be divisible by "
            "sequence_length * train_batch_size"
        )
    if gpu_count > 1 and train_batch_size % gpu_count != 0:
        raise ValueError(
            "training.train_batch_size must be divisible by worker_gpu_count "
            "for FSDP batch-axis sharding; got "
            f"training.train_batch_size={train_batch_size} and "
            f"worker_gpu_count={gpu_count}"
        )
    total_steps = train_tokens // tokens_per_step
    if total_steps % num_evals != 0:
        raise ValueError("total optimizer steps must be divisible by training.num_evals")
    eval_interval = max(1, total_steps // num_evals)
    validation_tokens = _positive_external_int(
        validation_tokens_per_eval, "validation_tokens_per_eval"
    )
    validation_tokens_per_batch = sequence_length * validation_batch_size
    if validation_tokens % validation_tokens_per_batch != 0:
        raise ValueError(
            "validation_tokens_per_eval must be divisible by "
            "sequence_length * validation_batch_size"
        )
    if gpu_count > 1 and validation_batch_size % gpu_count != 0:
        raise ValueError(
            "training.validation_batch_size must be divisible by worker_gpu_count "
            "for FSDP batch-axis sharding; got "
            f"training.validation_batch_size={validation_batch_size} and "
            f"worker_gpu_count={gpu_count}"
        )
    validation_batches = validation_tokens // validation_tokens_per_batch
    non_embedding_params = _non_embedding_parameter_estimate(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    embedding_params = vocab_size * hidden_size
    parameter_count = non_embedding_params + embedding_params
    training_flops = 6 * parameter_count * train_tokens
    warnings = _resource_warnings(
        total_steps=total_steps,
        num_evals=num_evals,
        learning_rate=learning_rate,
        requested_runtime_seconds=requested_runtime_seconds,
    )

    return {
        "model": {
            "attention_bias": attention_bias,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": num_key_value_heads,
            "rms_norm_eps": rms_norm_eps,
            "rope_theta": rope_theta,
            "tie_word_embeddings": tie_word_embeddings,
            "dtype": dtype,
            "vocab_size": vocab_size,
        },
        "training": {
            "train_tokens": train_tokens,
            "sequence_length": sequence_length,
            "train_batch_size": train_batch_size,
            "validation_batch_size": validation_batch_size,
            "num_evals": num_evals,
            "model_seed": model_seed,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "lr_schedule": lr_schedule,
            "weight_decay": weight_decay,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "adam_epsilon": adam_epsilon,
            "warmup_fraction": warmup_fraction,
            "final_lr_fraction": final_lr_fraction,
            "gradient_clip_norm": gradient_clip_norm,
        },
        "parameter_count_estimate": parameter_count,
        "non_embedding_parameter_count_estimate": non_embedding_params,
        "tokens_per_optimizer_step": tokens_per_step,
        "total_optimizer_steps": total_steps,
        "eval_interval_steps": eval_interval,
        "validation_tokens_per_eval": validation_tokens,
        "validation_batches_per_eval": validation_batches,
        "estimated_training_flops": training_flops,
        "max_runtime_seconds": int(requested_runtime_seconds),
        "resource_warnings": warnings,
        "code_version": code_version,
        "data_manifest_id": data_manifest_id,
        "eval_manifest_id": eval_manifest_id,
    }


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be an object")
    return value


def _reject_unknown_keys(
    mapping: Mapping[str, Any], allowed: frozenset[str], section_name: str
) -> None:
    unknown = sorted(str(key) for key in mapping if str(key) not in allowed)
    if unknown:
        raise ValueError(
            f"{section_name} contains unsupported field(s): " + ", ".join(unknown)
        )


def _field(mapping: Mapping[str, Any], dotted_name: str) -> str:
    return dotted_name.split(".", 1)[1]


def _positive_int(
    mapping: Mapping[str, Any], dotted_name: str, *, default: int | None = None
) -> int:
    key = _field(mapping, dotted_name)
    value = mapping.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{dotted_name} must be positive")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{dotted_name} must be positive") from exc
    if parsed <= 0:
        raise ValueError(f"{dotted_name} must be positive")
    return parsed


def _non_negative_int(
    mapping: Mapping[str, Any], dotted_name: str, *, default: int
) -> int:
    key = _field(mapping, dotted_name)
    value = mapping.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{dotted_name} must be non-negative")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{dotted_name} must be non-negative") from exc
    if parsed < 0:
        raise ValueError(f"{dotted_name} must be non-negative")
    return parsed


def _positive_external_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be positive")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be positive") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _non_negative_external_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be non-negative")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be non-negative") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _finite_float(mapping: Mapping[str, Any], dotted_name: str) -> float:
    key = _field(mapping, dotted_name)
    try:
        parsed = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{dotted_name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{dotted_name} must be finite")
    return parsed


def _positive_finite_float(
    mapping: Mapping[str, Any], dotted_name: str, *, default: float
) -> float:
    key = _field(mapping, dotted_name)
    value = mapping.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{dotted_name} must be positive and finite") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{dotted_name} must be positive and finite")
    return parsed


def _non_negative_finite_float(
    mapping: Mapping[str, Any], dotted_name: str, *, default: float
) -> float:
    key = _field(mapping, dotted_name)
    value = mapping.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{dotted_name} must be non-negative and finite") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{dotted_name} must be non-negative and finite")
    return parsed


def _bounded_fraction(
    mapping: Mapping[str, Any],
    dotted_name: str,
    *,
    default: float,
    upper_inclusive: bool = True,
) -> float:
    key = _field(mapping, dotted_name)
    value = mapping.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{dotted_name} must be in [0, 1]") from exc
    upper_ok = parsed <= 1 if upper_inclusive else parsed < 1
    if not math.isfinite(parsed) or not 0 <= parsed or not upper_ok:
        interval = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ValueError(f"{dotted_name} must be in {interval}")
    return parsed


def _optional_bool(mapping: Mapping[str, Any], dotted_name: str, *, default: bool) -> bool:
    key = _field(mapping, dotted_name)
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{dotted_name} must be boolean")
    return value


def _choice(
    mapping: Mapping[str, Any],
    dotted_name: str,
    *,
    allowed: frozenset[str],
    default: str,
) -> str:
    key = _field(mapping, dotted_name)
    value = mapping.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{dotted_name} must be one of: {sorted(allowed)}")
    normalized = value.lower()
    if normalized not in allowed:
        raise ValueError(f"{dotted_name} must be one of: {sorted(allowed)}")
    return normalized


def _non_embedding_parameter_estimate(
    *, num_hidden_layers: int, hidden_size: int, intermediate_size: int
) -> int:
    per_layer_attention = 4 * hidden_size * hidden_size
    per_layer_ffn = 2 * hidden_size * intermediate_size
    per_layer_norms = 2 * hidden_size
    final_norm = hidden_size
    return (
        num_hidden_layers
        * (per_layer_attention + per_layer_ffn + per_layer_norms)
        + final_norm
    )


def _validate_model_sharding_compatibility(
    *,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    tie_word_embeddings: bool,
    worker_gpu_count: int,
) -> None:
    if worker_gpu_count <= 1:
        return
    _require_divisible_by_worker_gpu_count(
        field_name="model.hidden_size",
        value=hidden_size,
        worker_gpu_count=worker_gpu_count,
    )
    _require_divisible_by_worker_gpu_count(
        field_name="model.intermediate_size",
        value=intermediate_size,
        worker_gpu_count=worker_gpu_count,
    )
    if not tie_word_embeddings:
        _require_divisible_by_worker_gpu_count(
            field_name="model.vocab_size",
            value=vocab_size,
            worker_gpu_count=worker_gpu_count,
        )


def _require_divisible_by_worker_gpu_count(
    *, field_name: str, value: int, worker_gpu_count: int
) -> None:
    if value % worker_gpu_count == 0:
        return
    raise ValueError(
        f"{field_name} must be divisible by worker_gpu_count for Stanford FSDP "
        f"parameter sharding; got {field_name}={value} and "
        f"worker_gpu_count={worker_gpu_count}"
    )


def _resource_warnings(
    *,
    total_steps: int,
    num_evals: int,
    learning_rate: float,
    requested_runtime_seconds: int,
) -> list[str]:
    warnings: list[str] = []
    if total_steps < 10:
        warnings.append("very_low_optimizer_steps")
    if num_evals > total_steps:
        warnings.append("suspicious_evaluation_cadence")
    if learning_rate > 0.01:
        warnings.append("high_learning_rate")
    if requested_runtime_seconds < 60:
        warnings.append("very_low_runtime_budget")
    return warnings
