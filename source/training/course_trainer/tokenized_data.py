from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import numpy as np
from numpy.typing import NDArray


class TokenizedDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class TokenShard:
    path: Path
    tokens: int
    offset: int


@dataclass(frozen=True)
class IndexedTokenDataset:
    index_path: Path
    shards: tuple[TokenShard, ...]
    total_tokens: int

    def read_range(self, start: int, stop: int) -> NDArray[np.uint32]:
        return self.read_positions(np.arange(start, stop, dtype=np.int64))

    def read_positions(self, positions: NDArray[Any]) -> NDArray[np.uint32]:
        flat = np.asarray(positions, dtype=np.int64).reshape(-1)
        if flat.size == 0:
            return np.empty(np.asarray(positions).shape, dtype=np.uint32)
        min_pos = int(flat.min())
        max_pos = int(flat.max())
        if min_pos < 0 or max_pos >= self.total_tokens:
            raise TokenizedDatasetError(
                f"token positions must be in [0, {self.total_tokens})"
            )

        out = np.empty(flat.shape, dtype=np.uint32)
        for shard in self.shards:
            shard_start = shard.offset
            shard_stop = shard.offset + shard.tokens
            mask = (flat >= shard_start) & (flat < shard_stop)
            if not mask.any():
                continue
            mmap = np.memmap(
                shard.path,
                mode="r",
                dtype="<u4",
                shape=(shard.tokens,),
            )
            out[mask] = mmap[flat[mask] - shard_start]
        return out.reshape(np.asarray(positions).shape)


@dataclass(frozen=True)
class TrainingDatasets:
    train: IndexedTokenDataset
    validation: IndexedTokenDataset


def load_training_datasets(manifest: Mapping[str, Any]) -> TrainingDatasets:
    data_config = _mapping(manifest, "data_config")
    validation_config = _mapping(manifest, "validation_config")
    train_uri = _non_empty_string(data_config, "data_config.tokenized_index_uri")
    validation_uri = _non_empty_string(
        validation_config,
        "validation_config.tokenized_index_uri",
    )
    validation_mode = os.getenv(
        "COURSE_TRAINER_TOKENIZED_INDEX_VALIDATION_MODE",
        "metadata",
    )
    return TrainingDatasets(
        train=load_indexed_dataset(train_uri, validation_mode=validation_mode),
        validation=load_indexed_dataset(
            validation_uri,
            validation_mode=validation_mode,
        ),
    )


def load_indexed_dataset(
    index_uri: str | Path,
    *,
    validation_mode: str = "metadata",
) -> IndexedTokenDataset:
    mode = _validation_mode(validation_mode)
    index_path = _local_path_from_uri(index_uri)
    try:
        raw = index_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TokenizedDatasetError(f"tokenized index is not readable: {index_path}") from exc
    try:
        index = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TokenizedDatasetError(f"tokenized index is not valid JSON: {exc}") from exc
    if not isinstance(index, dict):
        raise TokenizedDatasetError("tokenized index must be a JSON object")
    if index.get("schema_version") != 1:
        raise TokenizedDatasetError("tokenized index must declare schema_version = 1")
    if index.get("token_dtype") not in {None, "uint32"}:
        raise TokenizedDatasetError("only uint32 tokenized indexes are supported")
    if index.get("byte_order") not in {None, "little"}:
        raise TokenizedDatasetError("only little-endian tokenized indexes are supported")
    shards_raw = index.get("shards")
    if not isinstance(shards_raw, list) or not shards_raw:
        raise TokenizedDatasetError("tokenized index must contain non-empty shards")

    shards: list[TokenShard] = []
    offset = 0
    for shard_number, shard in enumerate(shards_raw):
        if not isinstance(shard, Mapping):
            raise TokenizedDatasetError("tokenized shard entry must be an object")
        shard_path = index_path.parent / _relative_shard_path(
            shard.get("path"),
            field_name=f"shards[{shard_number}].path",
        )
        tokens = _positive_int(
            shard.get("tokens"),
            field_name=f"shards[{shard_number}].tokens",
        )
        if shard.get("dtype") != "uint32":
            raise TokenizedDatasetError("only uint32 token shards are supported")
        if shard.get("byte_order", "little") != "little":
            raise TokenizedDatasetError("only little-endian token shards are supported")
        try:
            byte_size = shard_path.stat().st_size
        except OSError as exc:
            raise TokenizedDatasetError(f"tokenized shard is not readable: {shard_path}") from exc
        expected_size = tokens * 4
        if byte_size != expected_size:
            raise TokenizedDatasetError(
                f"tokenized shard byte size mismatch: {shard_path}"
            )
        expected_hash = shard.get("sha256")
        if not isinstance(expected_hash, str) or not _is_sha256_hex(expected_hash):
            raise TokenizedDatasetError(
                f"tokenized shard sha256 is invalid: {shard_path}"
            )
        if mode == "full":
            actual_hash = _sha256_file(shard_path)
            if actual_hash != expected_hash:
                raise TokenizedDatasetError(
                    f"tokenized shard hash mismatch: {shard_path}"
                )
        shards.append(TokenShard(path=shard_path, tokens=tokens, offset=offset))
        offset += tokens

    declared_total = index.get("total_tokens")
    if declared_total is not None and _non_negative_int(
        declared_total,
        field_name="total_tokens",
    ) != offset:
        raise TokenizedDatasetError("tokenized index total_tokens does not match shards")
    return IndexedTokenDataset(
        index_path=index_path,
        shards=tuple(shards),
        total_tokens=offset,
    )


def _local_path_from_uri(value: str | Path) -> Path:
    if isinstance(value, Path):
        return value
    raw = str(value).strip()
    parsed = urlparse(raw)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise TokenizedDatasetError(f"unsupported file URI host: {parsed.netloc}")
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise TokenizedDatasetError(
            "course_trainer supports local file tokenized indexes only; "
            f"got URI scheme {parsed.scheme!r}"
        )
    return Path(raw)


def _validation_mode(value: str) -> str:
    if value not in {"metadata", "full"}:
        raise TokenizedDatasetError(
            "tokenized index validation_mode must be 'metadata' or 'full'"
        )
    return value


def _mapping(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = manifest.get(key)
    if not isinstance(value, Mapping):
        raise TokenizedDatasetError(f"{key} must be an object")
    return value


def _non_empty_string(mapping: Mapping[str, Any], dotted_name: str) -> str:
    key = dotted_name.split(".", 1)[1]
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TokenizedDatasetError(f"{dotted_name} must be non-empty")
    return value.strip()


def _relative_shard_path(value: Any, *, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TokenizedDatasetError(f"{field_name} must be non-empty")
    raw = value.strip().replace("\\", "/")
    parsed = urlparse(raw)
    if parsed.scheme or parsed.netloc or raw.startswith("/"):
        raise TokenizedDatasetError(f"{field_name} must be a relative path")
    path = Path(raw)
    if str(path) in {"", "."} or ".." in path.parts:
        raise TokenizedDatasetError(f"{field_name} must be a safe relative path")
    return path


def _positive_int(value: Any, *, field_name: str) -> int:
    parsed = _non_negative_int(value, field_name=field_name)
    if parsed <= 0:
        raise TokenizedDatasetError(f"{field_name} must be positive")
    return parsed


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TokenizedDatasetError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TokenizedDatasetError(f"{field_name} must be an integer") from exc
    if parsed < 0:
        raise TokenizedDatasetError(f"{field_name} must be non-negative")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
