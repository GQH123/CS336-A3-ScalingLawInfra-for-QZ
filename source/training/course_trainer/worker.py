from __future__ import annotations

import math
import sys
import time
import types
import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Protocol

import numpy as np

from course_trainer.tokenized_data import TrainingDatasets, load_training_datasets


class CourseTrainerConfigError(ValueError):
    pass


MAX_TRAIN_TOKENS = 500_000_000_000


class TrainerBackend(Protocol):
    def run(
        self,
        *,
        manifest: Mapping[str, Any],
        training_config_payload: Mapping[str, Any],
        dataset: TrainingDatasets,
    ) -> Mapping[str, Any]:
        ...


def train(
    manifest: Mapping[str, Any],
    emit_event,
) -> Mapping[str, Any]:
    return run_with_backend(
        manifest,
        emit_event,
        backend=StanfordLocalBackend(),
    )


def run_with_backend(
    manifest: Mapping[str, Any],
    emit_event,
    *,
    backend: TrainerBackend,
) -> Mapping[str, Any]:
    training_config_payload = build_stanford_training_config_payload(manifest)
    dataset = load_training_datasets(manifest)
    result = dict(
        backend.run(
            manifest=manifest,
            training_config_payload=training_config_payload,
            dataset=dataset,
        )
    )
    validation_losses = _finite_loss_list(result.get("validation_losses", []))
    result["validation_losses"] = validation_losses
    for step, loss in enumerate(validation_losses, start=1):
        emit_event({"event_type": "validation", "step": step, "loss": loss})
    return result


def build_stanford_training_config_payload(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    model = _mapping(manifest, "model_config")
    training = _mapping(manifest, "training_config")
    validation = _mapping(manifest, "validation_config")
    runtime = _mapping(manifest, "runtime_config")

    num_attention_heads = _positive_int(model, "model_config.num_attention_heads")
    num_key_value_heads = _positive_int(
        model,
        "model_config.num_key_value_heads",
        default=num_attention_heads,
    )
    if num_key_value_heads != num_attention_heads:
        raise CourseTrainerConfigError(
            "Stanford CS336 reference trainer requires "
            "num_key_value_heads == num_attention_heads; GQA is not supported"
        )
    head_dim = _positive_int(model, "model_config.head_dim")
    _validate_stanford_attention_head_dim(head_dim)
    hidden_size = _positive_int(model, "model_config.hidden_size")
    if hidden_size != num_attention_heads * head_dim:
        raise CourseTrainerConfigError(
            "model_config.hidden_size must equal "
            "model_config.num_attention_heads * model_config.head_dim"
        )
    dtype = _string_choice(
        model,
        "model_config.dtype",
        allowed={"float32", "bfloat16"},
        default="bfloat16",
    )
    lr_schedule = _string_choice(
        training,
        "training_config.lr_schedule",
        allowed={"cosine", "constant", "linear"},
        default="cosine",
    )
    if lr_schedule == "linear":
        raise CourseTrainerConfigError(
            "Stanford CS336 reference trainer does not support lr_schedule='linear'"
        )
    optimizer = _string_choice(
        training,
        "training_config.optimizer",
        allowed={"adamw", "sgd"},
        default="adamw",
    )
    learning_rate = _positive_float(training, "training_config.learning_rate")
    warmup_fraction = 0.0 if lr_schedule == "constant" else _bounded_fraction(
        training,
        "training_config.warmup_fraction",
        default=0.0,
        upper_inclusive=False,
    )
    final_lr_fraction = 1.0 if lr_schedule == "constant" else _bounded_fraction(
        training,
        "training_config.final_lr_fraction",
        default=0.0,
    )
    lr_scheduler = {
        "peak_value": learning_rate,
        "final_lr_frac": final_lr_fraction,
        "warmup_frac": warmup_fraction,
        "init_value": learning_rate if lr_schedule == "constant" else 0.0,
    }
    gradient_clip_norm = _optional_positive_float(
        training,
        "training_config.gradient_clip_norm",
        default=1.0,
    )
    optimizer_config: dict[str, Any] = {
        "lr_scheduler": lr_scheduler,
        "grad_clip_norm": gradient_clip_norm,
    }
    if optimizer == "adamw":
        optimizer_config.update(
            {
                "weight_decay": _non_negative_float(
                    training,
                    "training_config.weight_decay",
                    default=0.0,
                ),
                "beta1": _bounded_fraction(
                    training,
                    "training_config.adam_beta1",
                    default=0.9,
                    upper_inclusive=False,
                ),
                "beta2": _bounded_fraction(
                    training,
                    "training_config.adam_beta2",
                    default=0.95,
                    upper_inclusive=False,
                ),
                "eps": _positive_float(
                    training,
                    "training_config.adam_epsilon",
                    default=1e-8,
                ),
                "eps_root": _positive_float(
                    training,
                    "training_config.adam_eps_root",
                    default=1e-8,
                ),
            }
        )

    return {
        "architecture_config": {
            "attention_bias": _bool(
                model,
                "model_config.attention_bias",
                default=False,
            ),
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "intermediate_size": _stanford_intermediate_size(model, hidden_size),
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": _positive_int(
                model,
                "model_config.num_hidden_layers",
            ),
            "num_key_value_heads": num_key_value_heads,
            "rms_norm_eps": _positive_float(model, "model_config.rms_norm_eps"),
            "rope_theta": _positive_int(model, "model_config.rope_theta"),
            "tie_word_embeddings": _bool(
                model,
                "model_config.tie_word_embeddings",
                default=False,
            ),
            "dtype": dtype,
            "vocab_size": _positive_int(model, "model_config.vocab_size"),
        },
        "optimizer_config": optimizer_config,
        "train_batch_size": _positive_int(training, "training_config.train_batch_size"),
        "val_batch_size": _positive_int(
            training,
            "training_config.validation_batch_size",
        ),
        "n_evals": _positive_int(training, "training_config.num_evals"),
        "total_train_tokens": _stanford_train_tokens(training),
        "max_runtime_seconds": _runtime_limit_seconds(runtime),
        "model_seed": _non_negative_int(training, "training_config.model_seed", default=0),
    }


def _validate_stanford_attention_head_dim(head_dim: int) -> None:
    if head_dim > 128 or head_dim % 8 != 0:
        raise CourseTrainerConfigError(
            "Stanford CS336 reference trainer requires head_dim <= 128 and a "
            f"multiple of 8 for GPU attention; got head_dim={head_dim}. Set "
            "model_config.head_dim explicitly, or set num_attention_heads so "
            "hidden_size / num_attention_heads satisfies this constraint. For "
            "example hidden_size=2048 with num_attention_heads=16 gives "
            "head_dim=128."
        )


def _stanford_intermediate_size(model: Mapping[str, Any], hidden_size: int) -> int:
    intermediate_size = _positive_int(model, "model_config.intermediate_size")
    if intermediate_size < hidden_size:
        raise CourseTrainerConfigError(
            "model_config.intermediate_size must be >= model_config.hidden_size"
        )
    return intermediate_size


def _stanford_train_tokens(training: Mapping[str, Any]) -> int:
    train_tokens = _positive_int(training, "training_config.train_tokens")
    if train_tokens > MAX_TRAIN_TOKENS:
        raise CourseTrainerConfigError(
            f"training_config.train_tokens must be <= {MAX_TRAIN_TOKENS}"
        )
    return train_tokens


@dataclass
class StanfordLocalBackend:
    """Local compute-node runner around the Stanford CS336 model and loop modules."""

    def run(
        self,
        *,
        manifest: Mapping[str, Any],
        training_config_payload: Mapping[str, Any],
        dataset: TrainingDatasets,
    ) -> Mapping[str, Any]:
        started = time.perf_counter()
        training = _mapping(manifest, "training_config")
        validation = _mapping(manifest, "validation_config")
        seq_len = _positive_int(training, "training_config.sequence_length")
        n_val_tokens = _positive_int(
            validation,
            "validation_config.validation_tokens_per_eval",
        )

        try:
            _install_stanford_schema_import_shim()
            import equinox.nn as nn
            import jax
            from jax.sharding import AxisType, NamedSharding
            from jax.sharding import PartitionSpec as P

            from cs336_scaling.training.data import Batch
            from cs336_scaling.training.loop import outer_loss
            from cs336_scaling.training.model.basic_model import BasicCausalLM
            from cs336_scaling.training.model.jax_utils import tree_rearrange
            from cs336_scaling.training.training_config import TrainingConfig
        except ImportError as exc:
            raise RuntimeError(
                "Stanford CS336 training dependencies are not installed. "
                "Install refs/cs336-assignment3-scaling[server] with "
                "deploy/compute-node/constraints.txt in the compute-node image."
            ) from exc

        TrainingConfigClass = _training_config_class(
            TrainingConfig,
            seq_len=seq_len,
            n_val_tokens=n_val_tokens,
        )
        try:
            training_config = TrainingConfigClass.model_validate(training_config_payload)
        except ValueError as exc:
            raise CourseTrainerConfigError(
                "Stanford CS336 reference trainer rejected this runtime config: "
                f"{exc}"
            ) from exc

        device_count = _jax_device_count(jax)
        _validate_batch_sharding_compatibility(training_config, device_count)
        _validate_parameter_sharding_compatibility(training_config, device_count)
        mesh = jax.make_mesh(
            (device_count,),
            ("fsdp",),
            axis_types=(AxisType.Explicit,),
        )
        jax.set_mesh(mesh)
        data_sharding = NamedSharding(mesh, P(None, "fsdp", None))

        def put_batched_data(batch, batch_size: int):
            return jax.device_put(
                tree_rearrange(
                    batch.to_jax(),
                    "(loops batch) ... -> loops batch ...",
                    batch=batch_size,
                ),
                data_sharding,
            )

        @jax.jit
        def make_model():
            model, state = nn.make_with_state(BasicCausalLM)(
                training_config.architecture_config,
                key=jax.random.PRNGKey(training_config.model_seed),
            )
            model = model.apply_sharding(mesh)
            return model, state

        with _with_config_error_for_known_jax_sharding_failure():
            model, state = make_model()
        opt_state = training_config.optimizer_config.build(training_config).init(model)
        val_data = put_batched_data(
            _validation_batch(
                dataset,
                training_config,
                Batch,
                vocab_size=training_config.architecture_config.vocab_size,
            ),
            batch_size=training_config.val_batch_size,
        )

        validation_losses: list[float] = []
        for chunk_idx in range(training_config.n_evals):
            train_data = put_batched_data(
                _train_batch(
                    dataset,
                    training_config,
                    Batch,
                    chunk_idx,
                    vocab_size=training_config.architecture_config.vocab_size,
                ),
                batch_size=training_config.train_batch_size,
            )
            result = outer_loss(
                model,
                state,
                train_data,
                val_data,
                training_config,
                opt_state,
            )
            model = result.model
            state = result.state
            opt_state = result.opt_state
            validation_losses.append(float(result.val_loss.item()))

        if not validation_losses:
            raise RuntimeError("training_config did not schedule any evaluations")
        return {
            "validation_losses": validation_losses,
            "final_validation_loss": validation_losses[-1],
            "actual_runtime_seconds": max(1, int(math.ceil(time.perf_counter() - started))),
        }


def _train_batch(
    dataset: TrainingDatasets,
    training_config,
    Batch,
    chunk_idx: int,
    *,
    vocab_size: int | None = None,
):
    seq_len = int(training_config.seq_len)
    tokens_per_eval = int(training_config.eval_every_tokens)
    start = chunk_idx * tokens_per_eval
    stop = start + tokens_per_eval + 1
    if stop > dataset.train.total_tokens:
        raise CourseTrainerConfigError(
            f"training token range [0, {stop}) exceeds mounted train tokens "
            f"{dataset.train.total_tokens}"
        )
    n_sequences = tokens_per_eval // seq_len
    sequence_starts = start + np.arange(n_sequences, dtype=np.int64) * seq_len
    rng = np.random.default_rng(int(training_config.model_seed) + chunk_idx)
    sequence_starts = rng.permutation(sequence_starts)
    offsets = np.arange(seq_len, dtype=np.int64)
    input_positions = sequence_starts[:, None] + offsets[None, :]
    input_ids = dataset.train.read_positions(input_positions)
    labels = dataset.train.read_positions(input_positions + 1)
    _validate_token_ids_within_vocab(
        input_ids=input_ids,
        labels=labels,
        vocab_size=vocab_size,
        context="training",
    )
    return Batch(input_ids=input_ids, labels=labels)


def _validation_batch(
    dataset: TrainingDatasets,
    training_config,
    Batch,
    *,
    vocab_size: int | None = None,
):
    seq_len = int(training_config.seq_len)
    n_tokens = int(training_config.n_val_tokens)
    stop = n_tokens + 1
    if stop > dataset.validation.total_tokens:
        raise CourseTrainerConfigError(
            f"validation token range [0, {stop}) exceeds mounted validation tokens "
            f"{dataset.validation.total_tokens}"
        )
    n_sequences = n_tokens // seq_len
    offsets = np.arange(seq_len, dtype=np.int64)
    starts = np.arange(n_sequences, dtype=np.int64) * seq_len
    input_positions = starts[:, None] + offsets[None, :]
    input_ids = dataset.validation.read_positions(input_positions)
    labels = dataset.validation.read_positions(input_positions + 1)
    _validate_token_ids_within_vocab(
        input_ids=input_ids,
        labels=labels,
        vocab_size=vocab_size,
        context="validation",
    )
    return Batch(input_ids=input_ids, labels=labels)


def _finite_loss_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        raise CourseTrainerConfigError("validation_losses must be a list")
    return [
        _finite_loss(loss, field_name=f"validation_losses[{index}]")
        for index, loss in enumerate(value)
    ]


def _finite_loss(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CourseTrainerConfigError(f"{field_name} must be finite") from exc
    if not math.isfinite(parsed):
        raise CourseTrainerConfigError(f"{field_name} must be finite")
    return parsed


def _validate_token_ids_within_vocab(
    *,
    input_ids: np.ndarray,
    labels: np.ndarray,
    vocab_size: int | None,
    context: str,
) -> None:
    if vocab_size is None:
        return
    parsed_vocab_size = int(vocab_size)
    max_input_id = int(np.max(input_ids)) if input_ids.size else -1
    max_label_id = int(np.max(labels)) if labels.size else -1
    max_token_id = max(max_input_id, max_label_id)
    if max_token_id >= parsed_vocab_size:
        raise CourseTrainerConfigError(
            f"{context} token ids must be < model_config.vocab_size; "
            f"max token id {max_token_id}, "
            f"model_config.vocab_size={parsed_vocab_size}. The configured model "
            "vocabulary must match the tokenizer used to build the mounted "
            "tokenized dataset."
        )


def _jax_device_count(jax_module) -> int:
    _check_single_jax_cuda_plugin()
    try:
        devices = list(jax_module.devices())
    except RuntimeError as exc:
        message = str(exc)
        if _looks_like_jax_cuda_driver_mismatch(message):
            raise RuntimeError(
                "JAX CUDA backend could not initialize because the installed "
                "JAX CUDA runtime is incompatible with the host NVIDIA driver. "
                "The Stanford reference package installs jax[cuda13]==0.9.1 on "
                "Linux; use the CUDA 12 compute-node install path in "
                "deploy/README.md on nodes whose nvidia-smi reports CUDA 12.x, "
                "or upgrade the NVIDIA driver and use the CUDA 13 path."
            ) from exc
        raise
    cuda_devices = [device for device in devices if _is_cuda_device(device)]
    if not cuda_devices:
        raise RuntimeError(
            "JAX did not expose any CUDA devices; saw devices: "
            f"{_format_devices(devices)}. This training worker requires GPU "
            "execution. If the smoke check printed 'PJRT_Api already exists for "
            "device type cuda', remove all jax-cuda12/jax-cuda13 plugin packages "
            "from the environment and reinstall exactly one CUDA plugin matching "
            "the node driver."
        )
    return len(cuda_devices)


def _validate_batch_sharding_compatibility(training_config, device_count: int) -> None:
    if device_count <= 1:
        return
    checks = (
        ("train_batch_size", "training_config.train_batch_size"),
        ("val_batch_size", "training_config.validation_batch_size"),
    )
    for attr, field_name in checks:
        batch_size = int(getattr(training_config, attr))
        if batch_size % device_count != 0:
            raise CourseTrainerConfigError(
                f"{field_name}={batch_size} is not divisible by the JAX CUDA "
                f"device count {device_count}. The Stanford trainer shards the "
                "data batch axis across the fsdp mesh, so set "
                "training.train_batch_size and training.validation_batch_size "
                f"to multiples of {device_count}, or run this manifest on "
                f"{batch_size} GPU(s)."
            )


def _validate_parameter_sharding_compatibility(training_config, device_count: int) -> None:
    if device_count <= 1:
        return
    architecture = training_config.architecture_config
    checks = [
        ("hidden_size", "model_config.hidden_size"),
        ("intermediate_size", "model_config.intermediate_size"),
    ]
    if not bool(getattr(architecture, "tie_word_embeddings")):
        checks.append(("vocab_size", "model_config.vocab_size"))
    for attr, field_name in checks:
        value = int(getattr(architecture, attr))
        if value % device_count != 0:
            raise CourseTrainerConfigError(
                f"{field_name}={value} is not divisible by the JAX CUDA device "
                f"count {device_count}. The Stanford trainer shards model "
                "parameter axes across the fsdp mesh, so set "
                f"{field_name} to a multiple of {device_count}, or run this "
                "manifest on a compatible GPU count."
            )


@contextmanager
def _with_config_error_for_known_jax_sharding_failure():
    try:
        yield
    except ValueError as exc:
        message = str(exc)
        if _looks_like_jax_sharding_divisibility_error(message):
            raise CourseTrainerConfigError(
                "Stanford/JAX FSDP sharding rejected this runtime config: "
                f"{message}"
            ) from exc
        raise


def _looks_like_jax_sharding_divisibility_error(message: str) -> bool:
    return (
        "Sharding spec" in message
        and "does not evenly divide the dimension size" in message
        and ("fsdp" in message or "NamedSharding" in message)
    )


def _check_single_jax_cuda_plugin() -> None:
    installed = [
        plugin
        for plugin in ("xla_cuda12", "xla_cuda13")
        if _module_exists(f"jax_plugins.{plugin}")
    ]
    if len(installed) > 1:
        raise RuntimeError(
            "Multiple JAX CUDA PJRT plugins are installed in this environment: "
            f"{', '.join(installed)}. Remove all jax-cuda12/jax-cuda13 plugin "
            "packages, then reinstall exactly one CUDA plugin matching the "
            "node's NVIDIA driver. Do not proceed while JAX prints "
            "'PJRT_Api already exists for device type cuda'."
        )


def _module_exists(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _is_cuda_device(device) -> bool:
    platform = str(getattr(device, "platform", "")).lower()
    return platform in {"cuda", "gpu"} or type(device).__name__.lower().startswith(
        "cudadevice"
    )


def _format_devices(devices) -> str:
    if not devices:
        return "[]"
    return "[" + ", ".join(repr(device) for device in devices) + "]"


def _looks_like_jax_cuda_driver_mismatch(message: str) -> bool:
    return any(
        snippet in message
        for snippet in (
            "cudaErrorInsufficientDriver",
            "no supported devices found for platform CUDA",
            "Unable to initialize backend 'cuda'",
            "PJRT_Api already exists for device type cuda",
        )
    )


def _training_config_class(base_class, *, seq_len: int, n_val_tokens: int):
    namespace = {
        "__annotations__": {
            "seq_len": ClassVar[int],
            "n_val_tokens": ClassVar[int],
        },
        "seq_len": int(seq_len),
        "n_val_tokens": int(n_val_tokens),
    }
    return type("ManifestTrainingConfig", (base_class,), namespace)


def _install_stanford_schema_import_shim() -> None:
    """Avoid the reference package's broad schemas __init__ during trainer imports."""
    if "cs336_scaling.schemas" in sys.modules:
        return
    try:
        import cs336_scaling
    except ImportError:
        return
    package_file = getattr(cs336_scaling, "__file__", None)
    if not package_file:
        return
    schemas_dir = Path(package_file).resolve().parent / "schemas"
    if not schemas_dir.exists():
        return
    schemas_package = types.ModuleType("cs336_scaling.schemas")
    schemas_package.__path__ = [str(schemas_dir)]
    schemas_package.__package__ = "cs336_scaling.schemas"
    schemas_package.__file__ = str(schemas_dir / "__init__.py")
    sys.modules["cs336_scaling.schemas"] = schemas_package


def _mapping(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = manifest.get(key)
    if not isinstance(value, Mapping):
        raise CourseTrainerConfigError(f"{key} must be an object")
    return value


def _runtime_limit_seconds(runtime: Mapping[str, Any]) -> float:
    if "max_runtime_seconds" in runtime:
        return _positive_float(runtime, "runtime_config.max_runtime_seconds")
    return _positive_float(runtime, "runtime_config.reserved_runtime_seconds")


def _field(mapping: Mapping[str, Any], dotted_name: str) -> str:
    return dotted_name.split(".", 1)[1]


def _positive_int(
    mapping: Mapping[str, Any],
    dotted_name: str,
    *,
    default: int | None = None,
) -> int:
    parsed = _non_negative_int(mapping, dotted_name, default=default)
    if parsed <= 0:
        raise CourseTrainerConfigError(f"{dotted_name} must be positive")
    return parsed


def _non_negative_int(
    mapping: Mapping[str, Any],
    dotted_name: str,
    *,
    default: int | None = None,
) -> int:
    key = _field(mapping, dotted_name)
    value = mapping.get(key, default)
    if isinstance(value, bool):
        raise CourseTrainerConfigError(f"{dotted_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CourseTrainerConfigError(f"{dotted_name} must be an integer") from exc
    if parsed < 0:
        raise CourseTrainerConfigError(f"{dotted_name} must be non-negative")
    return parsed


def _finite_float(
    mapping: Mapping[str, Any],
    dotted_name: str,
    *,
    default: float | int | None = None,
) -> float:
    key = _field(mapping, dotted_name)
    value = mapping.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CourseTrainerConfigError(f"{dotted_name} must be finite") from exc
    if not math.isfinite(parsed):
        raise CourseTrainerConfigError(f"{dotted_name} must be finite")
    return parsed


def _positive_float(
    mapping: Mapping[str, Any],
    dotted_name: str,
    *,
    default: float | int | None = None,
) -> float:
    parsed = _finite_float(mapping, dotted_name, default=default)
    if parsed <= 0:
        raise CourseTrainerConfigError(f"{dotted_name} must be positive")
    return parsed


def _optional_positive_float(
    mapping: Mapping[str, Any],
    dotted_name: str,
    *,
    default: float | None,
) -> float | None:
    key = _field(mapping, dotted_name)
    if mapping.get(key, default) is None:
        return None
    return _positive_float(mapping, dotted_name, default=default)


def _non_negative_float(
    mapping: Mapping[str, Any],
    dotted_name: str,
    *,
    default: float | int,
) -> float:
    parsed = _finite_float(mapping, dotted_name, default=default)
    if parsed < 0:
        raise CourseTrainerConfigError(f"{dotted_name} must be non-negative")
    return parsed


def _bounded_fraction(
    mapping: Mapping[str, Any],
    dotted_name: str,
    *,
    default: float,
    upper_inclusive: bool = True,
) -> float:
    parsed = _finite_float(mapping, dotted_name, default=default)
    upper_ok = parsed <= 1 if upper_inclusive else parsed < 1
    if parsed < 0 or not upper_ok:
        bracket = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise CourseTrainerConfigError(f"{dotted_name} must be in {bracket}")
    return parsed


def _bool(mapping: Mapping[str, Any], dotted_name: str, *, default: bool) -> bool:
    key = _field(mapping, dotted_name)
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise CourseTrainerConfigError(f"{dotted_name} must be boolean")
    return value


def _string_choice(
    mapping: Mapping[str, Any],
    dotted_name: str,
    *,
    allowed: set[str],
    default: str,
) -> str:
    key = _field(mapping, dotted_name)
    value = mapping.get(key, default)
    if not isinstance(value, str) or value not in allowed:
        raise CourseTrainerConfigError(
            f"{dotted_name} must be one of: {', '.join(sorted(allowed))}"
        )
    return value
