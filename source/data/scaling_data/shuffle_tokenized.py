from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping, Sequence

from scaling_data.tokenize import (
    TOKEN_BYTE_ORDER,
    TOKEN_DTYPE,
    TOKEN_SHARD_FORMAT,
    TOKENIZED_INDEX_SCHEMA_VERSION,
    TokenizedIndexError,
    load_tokenized_index,
)


DEFAULT_SHUFFLE_CHUNK_TOKENS = 262_144
DEFAULT_MAX_OPEN_SHARDS = 32


@dataclass(frozen=True)
class _TokenChunk:
    shard_path: Path
    offset_tokens: int
    tokens: int
    source_id: str


class _ShardReaderCache:
    def __init__(self, *, max_open_shards: int):
        if max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        self.max_open_shards = max_open_shards
        self._handles: OrderedDict[Path, BinaryIO] = OrderedDict()

    def read_chunk(self, chunk: _TokenChunk) -> bytes:
        handle = self._handle_for(chunk.shard_path)
        handle.seek(chunk.offset_tokens * 4)
        expected_bytes = chunk.tokens * 4
        data = handle.read(expected_bytes)
        if len(data) != expected_bytes:
            raise TokenizedIndexError(
                f"could not read {chunk.tokens} token(s) from {chunk.shard_path} "
                f"at token offset {chunk.offset_tokens}"
            )
        return data

    def close(self) -> None:
        while self._handles:
            _, handle = self._handles.popitem()
            handle.close()

    def _handle_for(self, shard_path: Path) -> BinaryIO:
        try:
            handle = self._handles.pop(shard_path)
        except KeyError:
            handle = shard_path.open("rb")
        self._handles[shard_path] = handle
        while len(self._handles) > self.max_open_shards:
            _, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        return handle


class _StreamingTokenShardWriter:
    def __init__(
        self,
        *,
        output_dir: Path,
        target_shard_tokens: int,
        metadata: Mapping[str, object] | None = None,
    ):
        if target_shard_tokens <= 0:
            raise ValueError("target_shard_tokens must be positive")
        self.output_dir = output_dir
        self.target_shard_tokens = target_shard_tokens
        self.metadata = dict(metadata or {})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._shard_index = 0
        self._handle: BinaryIO | None = None
        self._hash = hashlib.sha256()
        self._tokens_in_shard = 0
        self._shard_source_token_counts: Counter[str] = Counter()
        self._source_token_counts: Counter[str] = Counter()
        self._total_tokens = 0
        self.shards: list[dict] = []

    def add_bytes(self, data: bytes, *, token_count: int, source_id: str) -> None:
        if token_count <= 0:
            return
        expected_bytes = token_count * 4
        if len(data) != expected_bytes:
            raise TokenizedIndexError(
                f"token byte range length mismatch: got {len(data)} bytes for "
                f"{token_count} token(s)"
            )
        normalized_source_id = _normalize_source_id(source_id)
        byte_offset = 0
        remaining_tokens = token_count
        while remaining_tokens:
            self._ensure_open_shard()
            capacity = self.target_shard_tokens - self._tokens_in_shard
            write_tokens = min(capacity, remaining_tokens)
            write_bytes = write_tokens * 4
            piece = data[byte_offset : byte_offset + write_bytes]
            assert self._handle is not None
            self._handle.write(piece)
            self._hash.update(piece)
            self._tokens_in_shard += write_tokens
            self._total_tokens += write_tokens
            self._source_token_counts[normalized_source_id] += write_tokens
            self._shard_source_token_counts[normalized_source_id] += write_tokens
            remaining_tokens -= write_tokens
            byte_offset += write_bytes
            if self._tokens_in_shard == self.target_shard_tokens:
                self._finish_shard()

    def close(self) -> dict:
        if self._handle is not None:
            self._finish_shard()
        index = {
            "schema_version": TOKENIZED_INDEX_SCHEMA_VERSION,
            "token_dtype": TOKEN_DTYPE,
            "byte_order": TOKEN_BYTE_ORDER,
            "shard_format": TOKEN_SHARD_FORMAT,
            "total_tokens": self._total_tokens,
            "source_token_counts": dict(sorted(self._source_token_counts.items())),
            "shards": self.shards,
        }
        if self.metadata:
            index["shuffle"] = self.metadata
        (self.output_dir / "index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return index

    def _ensure_open_shard(self) -> None:
        if self._handle is not None:
            return
        shard_path = self.output_dir / f"tokens-{self._shard_index:06d}.bin"
        self._handle = shard_path.open("xb")
        self._hash = hashlib.sha256()
        self._tokens_in_shard = 0
        self._shard_source_token_counts = Counter()

    def _finish_shard(self) -> None:
        assert self._handle is not None
        self._handle.close()
        self._handle = None
        self.shards.append(
            {
                "path": f"tokens-{self._shard_index:06d}.bin",
                "tokens": self._tokens_in_shard,
                "dtype": TOKEN_DTYPE,
                "byte_order": TOKEN_BYTE_ORDER,
                "sha256": self._hash.hexdigest(),
                "source_token_counts": dict(
                    sorted(self._shard_source_token_counts.items())
                ),
            }
        )
        self._shard_index += 1


def shuffle_tokenized_corpus(
    *,
    input_index_path: Path,
    output_dir: Path,
    seed: int,
    chunk_tokens: int = DEFAULT_SHUFFLE_CHUNK_TOKENS,
    target_shard_tokens: int | None = None,
    validation_mode: str = "metadata",
    source_order: Sequence[str] | None = None,
    overwrite: bool = False,
    max_open_shards: int = DEFAULT_MAX_OPEN_SHARDS,
) -> dict:
    """Shuffle/interleave an existing flat-binary tokenized corpus.

    The input is an existing ``index.json`` plus ``tokens-*.bin`` shards. The
    implementation copies token byte ranges; it does not decode text, run a
    tokenizer, or load the whole corpus into memory.
    """

    normalized_chunk_tokens = _positive_int(chunk_tokens, "chunk_tokens")
    input_index_path = input_index_path.resolve()
    output_dir = output_dir.resolve()
    _prepare_output_dir(
        output_dir=output_dir,
        input_dir=input_index_path.parent,
        overwrite=overwrite,
    )
    input_index = load_tokenized_index(input_index_path, validation_mode=validation_mode)
    normalized_target_shard_tokens = _target_shard_tokens(
        input_index,
        target_shard_tokens=target_shard_tokens,
    )
    inferred_source_order = _source_order(input_index, explicit_order=source_order)
    chunks = _build_token_chunks(
        index_path=input_index_path,
        index=input_index,
        chunk_tokens=normalized_chunk_tokens,
        source_order=inferred_source_order,
    )
    if not chunks:
        raise TokenizedIndexError("tokenized index contains no tokens to shuffle")

    metadata = {
        "algorithm": "source_balanced_chunk_shuffle_v1",
        "chunk_tokens": normalized_chunk_tokens,
        "input_index": str(input_index_path),
        "input_total_tokens": int(input_index["total_tokens"]),
        "seed": int(seed),
        "source_order": inferred_source_order,
        "target_shard_tokens": normalized_target_shard_tokens,
    }
    writer = _StreamingTokenShardWriter(
        output_dir=output_dir,
        target_shard_tokens=normalized_target_shard_tokens,
        metadata=metadata,
    )
    reader = _ShardReaderCache(max_open_shards=max_open_shards)
    try:
        for chunk in _balanced_chunk_schedule(chunks, seed=seed):
            writer.add_bytes(
                reader.read_chunk(chunk),
                token_count=chunk.tokens,
                source_id=chunk.source_id,
            )
    finally:
        reader.close()
    return writer.close()


def _prepare_output_dir(*, output_dir: Path, input_dir: Path, overwrite: bool) -> None:
    if output_dir == input_dir:
        raise ValueError("output_dir must be different from the input index directory")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    existing = list(output_dir.iterdir())
    if not existing:
        return
    if not overwrite:
        raise FileExistsError(
            f"output_dir is not empty: {output_dir}; pass overwrite=True or "
            "--overwrite to replace a previous shuffled tokenized output"
        )
    allowed_names = {"index.json"}
    removable = []
    for path in existing:
        if path.is_file() and (
            path.name in allowed_names
            or (path.name.startswith("tokens-") and path.name.endswith(".bin"))
        ):
            removable.append(path)
            continue
        raise FileExistsError(
            f"output_dir contains non-shuffle artifact {path}; refusing to delete it"
        )
    for path in removable:
        path.unlink()


def _target_shard_tokens(index: Mapping[str, object], *, target_shard_tokens) -> int:
    if target_shard_tokens is not None:
        return _positive_int(target_shard_tokens, "target_shard_tokens")
    shards = index.get("shards")
    if not isinstance(shards, list) or not shards:
        raise TokenizedIndexError("tokenized index missing non-empty shards list")
    return max(_positive_int(shard.get("tokens"), "shards[].tokens") for shard in shards)


def _build_token_chunks(
    *,
    index_path: Path,
    index: Mapping[str, object],
    chunk_tokens: int,
    source_order: Sequence[str],
) -> list[_TokenChunk]:
    shards = index.get("shards")
    if not isinstance(shards, list):
        raise TokenizedIndexError("tokenized index missing shards list")
    chunks: list[_TokenChunk] = []
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise TokenizedIndexError("tokenized shard entry must be an object")
        shard_path = index_path.parent / _safe_relative_shard_path(shard.get("path"))
        shard_tokens = _positive_int(shard.get("tokens"), "shards[].tokens")
        offset_tokens = 0
        for source_id, run_tokens in _source_runs_for_shard(
            shard,
            shard_tokens=shard_tokens,
            source_order=source_order,
        ):
            remaining = run_tokens
            while remaining:
                take = min(chunk_tokens, remaining)
                chunks.append(
                    _TokenChunk(
                        shard_path=shard_path,
                        offset_tokens=offset_tokens,
                        tokens=take,
                        source_id=source_id,
                    )
                )
                offset_tokens += take
                remaining -= take
        if offset_tokens != shard_tokens:
            raise TokenizedIndexError(
                f"source runs do not cover shard {shard.get('path')!r}"
            )
    return chunks


def _source_runs_for_shard(
    shard: Mapping[str, object],
    *,
    shard_tokens: int,
    source_order: Sequence[str],
) -> list[tuple[str, int]]:
    counts = _source_counts(shard.get("source_token_counts"))
    if not counts:
        return [("unknown", shard_tokens)]
    if sum(counts.values()) != shard_tokens:
        return [("unknown", shard_tokens)]
    order_index = {source_id: index for index, source_id in enumerate(source_order)}
    ordered_source_ids = sorted(
        counts,
        key=lambda source_id: (order_index.get(source_id, len(order_index)), source_id),
    )
    return [(source_id, counts[source_id]) for source_id in ordered_source_ids]


def _balanced_chunk_schedule(
    chunks: Iterable[_TokenChunk],
    *,
    seed: int,
) -> Iterable[_TokenChunk]:
    chunks_by_source: dict[str, list[_TokenChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[chunk.source_id].append(chunk)

    rng = random.Random(seed)
    source_ids = sorted(chunks_by_source)
    rng.shuffle(source_ids)
    for source_id in source_ids:
        rng.shuffle(chunks_by_source[source_id])

    weights = {
        source_id: sum(chunk.tokens for chunk in chunks_by_source[source_id])
        for source_id in source_ids
    }
    current = {source_id: 0 for source_id in source_ids}
    active = list(source_ids)
    while active:
        total_active_weight = sum(weights[source_id] for source_id in active)
        for source_id in active:
            current[source_id] += weights[source_id]
        chosen = max(
            active,
            key=lambda source_id: (
                current[source_id],
                -source_ids.index(source_id),
            ),
        )
        current[chosen] -= total_active_weight
        yield chunks_by_source[chosen].pop()
        if not chunks_by_source[chosen]:
            active.remove(chosen)


def _source_order(
    index: Mapping[str, object],
    *,
    explicit_order: Sequence[str] | None,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    if explicit_order is not None:
        for raw_source_id in explicit_order:
            source_id = _normalize_source_id(raw_source_id)
            if source_id not in seen:
                ordered.append(source_id)
                seen.add(source_id)
    shards = index.get("shards")
    if not isinstance(shards, list):
        return ordered
    for shard in shards:
        if not isinstance(shard, Mapping):
            continue
        counts = _source_counts(shard.get("source_token_counts"))
        for source_id in sorted(counts):
            if source_id not in seen:
                ordered.append(source_id)
                seen.add(source_id)
    return ordered


def _source_counts(value) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts: Counter[str] = Counter()
    for raw_source_id, raw_count in value.items():
        source_id = _normalize_source_id(raw_source_id)
        count = _positive_int(raw_count, "source_token_counts[]", allow_zero=True)
        if count:
            counts[source_id] += count
    return dict(counts)


def _normalize_source_id(value) -> str:
    source_id = str(value).strip() if value is not None else ""
    return source_id or "unknown"


def _safe_relative_shard_path(value) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TokenizedIndexError("tokenized shard path must be non-empty")
    raw_path = value.strip().replace("\\", "/")
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise TokenizedIndexError("tokenized shard path must be a safe relative path")
    return path


def _positive_int(value, field_name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if allow_zero:
        if parsed < 0:
            raise ValueError(f"{field_name} must be non-negative")
    elif parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _parse_source_order(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [_normalize_source_id(item) for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Shuffle/interleave an existing tokenized flat-binary corpus without "
            "re-tokenizing text."
        )
    )
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_SHUFFLE_CHUNK_TOKENS)
    parser.add_argument("--target-shard-tokens", type=int)
    parser.add_argument(
        "--validation-mode",
        choices=("metadata", "full"),
        default="metadata",
    )
    parser.add_argument(
        "--source-order",
        help=(
            "Optional comma-separated source order used to split mixed-source "
            "boundary shards, for example "
            "code,education,general_web,long_form,math,multilingual."
        ),
    )
    parser.add_argument("--max-open-shards", type=int, default=DEFAULT_MAX_OPEN_SHARDS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing shuffled output containing only index.json and "
            "tokens-*.bin."
        ),
    )
    args = parser.parse_args(argv)

    index = shuffle_tokenized_corpus(
        input_index_path=args.input_index,
        output_dir=args.output_dir,
        seed=args.seed,
        chunk_tokens=args.chunk_tokens,
        target_shard_tokens=args.target_shard_tokens,
        validation_mode=args.validation_mode,
        source_order=_parse_source_order(args.source_order),
        overwrite=args.overwrite,
        max_open_shards=args.max_open_shards,
    )
    print(
        json.dumps(
            {
                "index_path": str(args.output_dir / "index.json"),
                "shards": len(index["shards"]),
                "source_token_counts": index.get("source_token_counts", {}),
                "total_tokens": index["total_tokens"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
