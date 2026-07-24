from __future__ import annotations

import argparse
import hashlib
import json
import sys
from array import array
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Iterator, Mapping

from scaling_data.io import iter_input_files, iter_jsonl_records


class TokenizedIndexError(ValueError):
    pass


TOKENIZED_INDEX_SCHEMA_VERSION = 1
TOKEN_DTYPE = "uint32"
TOKEN_BYTE_ORDER = "little"
TOKEN_SHARD_FORMAT = "flat_binary_uint32_le"
DEFAULT_TOKENIZE_CHUNK_RECORDS = 512

_TOKENIZER_WORKER = None


class TokenShardWriter:
    def __init__(self, *, output_dir: Path, target_shard_tokens: int):
        if target_shard_tokens <= 0:
            raise ValueError("target_shard_tokens must be positive")
        self.output_dir = output_dir
        self.target_shard_tokens = target_shard_tokens
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = array("I")
        self._source_buffer: list[str] = []
        self._shard_index = 0
        self._total_tokens = 0
        self._source_token_counts: Counter[str] = Counter()
        self.shards: list[dict] = []

    def add(self, token_ids: list[int], *, source_id: str = "unknown") -> None:
        normalized_source_id = source_id.strip() if source_id else "unknown"
        if not normalized_source_id:
            normalized_source_id = "unknown"
        self._buffer.extend(token_ids)
        self._source_buffer.extend([normalized_source_id] * len(token_ids))
        self._total_tokens += len(token_ids)
        self._source_token_counts[normalized_source_id] += len(token_ids)
        while len(self._buffer) >= self.target_shard_tokens:
            self._flush_exact(self.target_shard_tokens)

    def close(self) -> dict:
        if self._buffer:
            self._flush_exact(len(self._buffer))
        index = {
            "schema_version": TOKENIZED_INDEX_SCHEMA_VERSION,
            "token_dtype": TOKEN_DTYPE,
            "byte_order": TOKEN_BYTE_ORDER,
            "shard_format": TOKEN_SHARD_FORMAT,
            "total_tokens": self._total_tokens,
            "source_token_counts": dict(sorted(self._source_token_counts.items())),
            "shards": self.shards,
        }
        (self.output_dir / "index.json").write_text(
            json.dumps(index, indent=2) + "\n",
            encoding="utf-8",
        )
        return index

    def _flush_exact(self, n_tokens: int) -> None:
        shard_tokens = self._buffer[:n_tokens]
        del self._buffer[:n_tokens]
        shard_sources = self._source_buffer[:n_tokens]
        del self._source_buffer[:n_tokens]
        shard_path = self.output_dir / f"tokens-{self._shard_index:06d}.bin"
        _ensure_uint32_array(shard_tokens)
        if sys.byteorder != TOKEN_BYTE_ORDER:
            shard_tokens.byteswap()
        with shard_path.open("wb") as handle:
            shard_tokens.tofile(handle)
        shard_bytes = shard_path.read_bytes()
        self.shards.append(
            {
                "path": shard_path.name,
                "tokens": n_tokens,
                "dtype": TOKEN_DTYPE,
                "byte_order": TOKEN_BYTE_ORDER,
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
                "source_token_counts": dict(sorted(Counter(shard_sources).items())),
            }
        )
        self._shard_index += 1


def tokenize_directory(
    *,
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str,
    target_shard_tokens: int,
    append_eos: bool = True,
    train_source_token_limits: Mapping[str, int] | None = None,
    workers: int = 1,
    chunk_records: int = DEFAULT_TOKENIZE_CHUNK_RECORDS,
) -> dict:
    worker_count = _normalize_workers(workers, field_name="workers")
    if worker_count > 1:
        return _tokenize_records_by_split_parallel(
            input_dir=input_dir,
            output_dir=output_dir,
            target_shard_tokens=target_shard_tokens,
            append_eos=append_eos,
            train_source_token_limits=train_source_token_limits,
            workers=worker_count,
            chunk_records=chunk_records,
            initializer=_initialize_named_tokenizer_worker,
            initargs=(tokenizer_name,),
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for tokenization") from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenize_records_by_split(
        input_dir=input_dir,
        output_dir=output_dir,
        tokenizer=tokenizer,
        target_shard_tokens=target_shard_tokens,
        append_eos=append_eos,
        train_source_token_limits=train_source_token_limits,
        workers=1,
        chunk_records=chunk_records,
    )


def tokenize_records_by_split(
    *,
    input_dir: Path,
    output_dir: Path,
    tokenizer,
    target_shard_tokens: int,
    append_eos: bool = True,
    train_source_token_limits: Mapping[str, int] | None = None,
    workers: int = 1,
    chunk_records: int = DEFAULT_TOKENIZE_CHUNK_RECORDS,
) -> dict[str, dict]:
    worker_count = _normalize_workers(workers, field_name="workers")
    if worker_count > 1:
        return _tokenize_records_by_split_parallel(
            input_dir=input_dir,
            output_dir=output_dir,
            target_shard_tokens=target_shard_tokens,
            append_eos=append_eos,
            train_source_token_limits=train_source_token_limits,
            workers=worker_count,
            chunk_records=chunk_records,
            initializer=_initialize_tokenizer_object_worker,
            initargs=(tokenizer,),
        )

    eos_token_id = tokenizer.eos_token_id
    writers: dict[str, TokenShardWriter] = {}
    normalized_train_limits = _normalize_train_source_token_limits(
        train_source_token_limits
    )
    train_source_token_counts: Counter[str] = Counter()

    def writer_for(split: str) -> TokenShardWriter:
        normalized_split = split if split in {"train", "validation"} else "train"
        if normalized_split not in writers:
            writers[normalized_split] = TokenShardWriter(
                output_dir=output_dir / normalized_split,
                target_shard_tokens=target_shard_tokens,
            )
        return writers[normalized_split]

    for path in iter_input_files(input_dir):
        for record in iter_jsonl_records(path):
            text = record.get("text")
            if not isinstance(text, str) or not text:
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if append_eos and eos_token_id is not None:
                token_ids.append(eos_token_id)
            source_id = record.get("source_id")
            split = record.get("split")
            normalized_split = split if split in {"train", "validation"} else "train"
            normalized_source_id = (
                source_id.strip() if isinstance(source_id, str) else "unknown"
            )
            if not normalized_source_id:
                normalized_source_id = "unknown"
            normalized_token_ids = [int(token_id) for token_id in token_ids]
            if normalized_split == "train" and normalized_train_limits:
                limit = normalized_train_limits.get(normalized_source_id)
                if limit is not None:
                    remaining = limit - train_source_token_counts[normalized_source_id]
                    if remaining <= 0:
                        continue
                    if len(normalized_token_ids) > remaining:
                        normalized_token_ids = normalized_token_ids[:remaining]
                    train_source_token_counts[normalized_source_id] += len(
                        normalized_token_ids
                    )
            if not normalized_token_ids:
                continue
            writer_for(normalized_split).add(
                normalized_token_ids,
                source_id=normalized_source_id,
            )

    return {
        split: writers[split].close()
        for split in sorted(writers)
    }


def _tokenize_records_by_split_parallel(
    *,
    input_dir: Path,
    output_dir: Path,
    target_shard_tokens: int,
    append_eos: bool,
    train_source_token_limits: Mapping[str, int] | None,
    workers: int,
    chunk_records: int,
    initializer,
    initargs: tuple,
) -> dict[str, dict]:
    chunk_size = _normalize_workers(chunk_records, field_name="chunk_records")
    writers: dict[str, TokenShardWriter] = {}
    normalized_train_limits = _normalize_train_source_token_limits(
        train_source_token_limits
    )
    train_source_token_counts: Counter[str] = Counter()

    def writer_for(split: str) -> TokenShardWriter:
        normalized_split = split if split in {"train", "validation"} else "train"
        if normalized_split not in writers:
            writers[normalized_split] = TokenShardWriter(
                output_dir=output_dir / normalized_split,
                target_shard_tokens=target_shard_tokens,
            )
        return writers[normalized_split]

    def consume_tokenized_records(tokenized_records: list[tuple[str, str, list[int]]]):
        for normalized_split, normalized_source_id, token_ids in tokenized_records:
            if normalized_split == "train" and normalized_train_limits:
                limit = normalized_train_limits.get(normalized_source_id)
                if limit is not None:
                    remaining = limit - train_source_token_counts[normalized_source_id]
                    if remaining <= 0:
                        continue
                    if len(token_ids) > remaining:
                        token_ids = token_ids[:remaining]
                    train_source_token_counts[normalized_source_id] += len(token_ids)
            if not token_ids:
                continue
            writer_for(normalized_split).add(
                token_ids,
                source_id=normalized_source_id,
            )

    chunks = _iter_tokenize_record_chunks(input_dir, chunk_records=chunk_size)
    pending = {}
    buffered: dict[int, list[tuple[str, str, list[int]]]] = {}
    next_chunk_to_consume = 0
    exhausted = False
    max_pending = max(1, workers * 2)

    def submit_next(executor) -> None:
        nonlocal exhausted
        if exhausted:
            return
        try:
            chunk_index, records = next(chunks)
        except StopIteration:
            exhausted = True
            return
        future = executor.submit(
            _tokenize_record_chunk_worker,
            chunk_index,
            records,
            append_eos,
        )
        pending[future] = chunk_index

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        for _ in range(max_pending):
            submit_next(executor)
            if exhausted:
                break

        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                chunk_index = pending.pop(future)
                result_index, tokenized_records = future.result()
                if result_index != chunk_index:
                    raise RuntimeError("tokenizer worker returned an unexpected chunk")
                buffered[result_index] = tokenized_records
                submit_next(executor)

            while next_chunk_to_consume in buffered:
                consume_tokenized_records(buffered.pop(next_chunk_to_consume))
                next_chunk_to_consume += 1

    return {
        split: writers[split].close()
        for split in sorted(writers)
    }


def _iter_tokenize_record_chunks(
    input_dir: Path,
    *,
    chunk_records: int,
) -> Iterator[tuple[int, list[dict]]]:
    chunk_index = 0
    records: list[dict] = []
    for path in iter_input_files(input_dir):
        for record in iter_jsonl_records(path):
            records.append(record)
            if len(records) >= chunk_records:
                yield chunk_index, records
                chunk_index += 1
                records = []
    if records:
        yield chunk_index, records


def _initialize_named_tokenizer_worker(tokenizer_name: str) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for tokenization") from exc
    _initialize_tokenizer_object_worker(AutoTokenizer.from_pretrained(tokenizer_name))


def _initialize_tokenizer_object_worker(tokenizer) -> None:
    global _TOKENIZER_WORKER
    _TOKENIZER_WORKER = tokenizer


def _tokenize_record_chunk_worker(
    chunk_index: int,
    records: list[dict],
    append_eos: bool,
) -> tuple[int, list[tuple[str, str, list[int]]]]:
    if _TOKENIZER_WORKER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    eos_token_id = _TOKENIZER_WORKER.eos_token_id
    tokenized_records: list[tuple[str, str, list[int]]] = []
    for record in records:
        text = record.get("text")
        if not isinstance(text, str) or not text:
            continue
        token_ids = _TOKENIZER_WORKER.encode(text, add_special_tokens=False)
        if append_eos and eos_token_id is not None:
            token_ids.append(eos_token_id)
        source_id = record.get("source_id")
        split = record.get("split")
        normalized_split = split if split in {"train", "validation"} else "train"
        normalized_source_id = (
            source_id.strip() if isinstance(source_id, str) else "unknown"
        )
        if not normalized_source_id:
            normalized_source_id = "unknown"
        normalized_token_ids = [int(token_id) for token_id in token_ids]
        if normalized_token_ids:
            tokenized_records.append(
                (
                    normalized_split,
                    normalized_source_id,
                    normalized_token_ids,
                )
            )
    return chunk_index, tokenized_records


def _normalize_train_source_token_limits(
    limits: Mapping[str, int] | None,
) -> dict[str, int]:
    if limits is None:
        return {}
    normalized: dict[str, int] = {}
    for raw_source_id, raw_limit in limits.items():
        if isinstance(raw_limit, bool):
            raise ValueError("train source token limits must be nonnegative integers")
        source_id = str(raw_source_id).strip()
        if not source_id:
            raise ValueError("train source token limit source IDs must be non-empty")
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "train source token limits must be nonnegative integers"
            ) from exc
        if limit < 0:
            raise ValueError("train source token limits must be nonnegative integers")
        normalized[source_id] = limit
    return normalized


def _normalize_workers(value: int, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def load_tokenized_index(index_path: Path, *, validation_mode: str = "full") -> dict:
    mode = _validation_mode(validation_mode)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise TokenizedIndexError("tokenized index must be a JSON object")
    if index.get("schema_version") != TOKENIZED_INDEX_SCHEMA_VERSION:
        raise TokenizedIndexError("tokenized index schema_version must be 1")
    if index.get("token_dtype") != TOKEN_DTYPE:
        raise TokenizedIndexError("tokenized index token_dtype must be uint32")
    if index.get("byte_order") != TOKEN_BYTE_ORDER:
        raise TokenizedIndexError("tokenized index byte_order must be little")
    if index.get("shard_format") != TOKEN_SHARD_FORMAT:
        raise TokenizedIndexError(
            "tokenized index shard_format must be flat_binary_uint32_le"
        )
    shards = index.get("shards")
    if not isinstance(shards, list):
        raise TokenizedIndexError("tokenized index missing shards list")
    base_dir = index_path.parent
    total_tokens = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise TokenizedIndexError("tokenized shard entry must be an object")
        if shard.get("dtype") != TOKEN_DTYPE:
            raise TokenizedIndexError("only uint32 token shards are supported")
        if shard.get("byte_order") != TOKEN_BYTE_ORDER:
            raise TokenizedIndexError(
                "only little-endian uint32 token shards are supported"
            )
        token_count = _positive_token_count(shard.get("tokens"), field_name="tokens")
        total_tokens += token_count
        relative_path = _safe_relative_shard_path(shard.get("path"))
        shard_path = base_dir / relative_path
        expected_hash = shard.get("sha256")
        if not isinstance(expected_hash, str) or not _is_sha256_hex(expected_hash):
            raise TokenizedIndexError(f"tokenized shard missing sha256: {shard_path}")
        expected_size = token_count * 4
        try:
            if shard_path.stat().st_size != expected_size:
                raise TokenizedIndexError(
                    f"tokenized shard byte size mismatch: {shard_path}"
                )
            if mode == "metadata":
                continue
            actual_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise TokenizedIndexError(f"tokenized shard is not readable: {shard_path}") from exc
        if actual_hash != expected_hash:
            raise TokenizedIndexError(f"tokenized shard hash mismatch: {shard_path}")
    declared_total_tokens = _positive_token_count(
        index.get("total_tokens"),
        field_name="total_tokens",
        allow_zero=True,
    )
    if declared_total_tokens != total_tokens:
        raise TokenizedIndexError("tokenized index total_tokens does not match shards")
    return index


def read_token_shard(base_dir: Path, shard: dict) -> list[int]:
    if shard.get("dtype") != TOKEN_DTYPE:
        raise TokenizedIndexError("only uint32 token shards are supported")
    if shard.get("byte_order") != TOKEN_BYTE_ORDER:
        raise TokenizedIndexError("only little-endian uint32 token shards are supported")
    shard_path = base_dir / _safe_relative_shard_path(shard.get("path"))
    tokens = array("I")
    try:
        with shard_path.open("rb") as handle:
            tokens.fromfile(handle, int(shard["tokens"]))
    except (OSError, EOFError, KeyError, ValueError) as exc:
        raise TokenizedIndexError(f"could not read token shard: {shard_path}") from exc
    _ensure_uint32_array(tokens)
    if sys.byteorder != TOKEN_BYTE_ORDER:
        tokens.byteswap()
    return list(tokens)


def _ensure_uint32_array(tokens: array) -> None:
    if tokens.itemsize != 4:
        raise TokenizedIndexError("platform array('I') is not 32 bits")


def _validation_mode(value: str) -> str:
    if value not in {"metadata", "full"}:
        raise TokenizedIndexError("validation_mode must be metadata or full")
    return value


def _positive_token_count(
    value,
    *,
    field_name: str,
    allow_zero: bool = False,
) -> int:
    if isinstance(value, bool):
        raise TokenizedIndexError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TokenizedIndexError(f"{field_name} must be a positive integer") from exc
    if allow_zero:
        if parsed < 0:
            raise TokenizedIndexError(f"{field_name} must be non-negative")
    elif parsed <= 0:
        raise TokenizedIndexError(f"{field_name} must be a positive integer")
    return parsed


def _safe_relative_shard_path(value) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TokenizedIndexError("tokenized shard path must be non-empty")
    raw_path = value.strip().replace("\\", "/")
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise TokenizedIndexError("tokenized shard path must be a safe relative path")
    return path


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize processed JSONL shards.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--no-eos", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-records", type=int, default=DEFAULT_TOKENIZE_CHUNK_RECORDS)
    args = parser.parse_args()

    index = tokenize_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        target_shard_tokens=args.target_shard_tokens,
        append_eos=not args.no_eos,
        workers=args.workers,
        chunk_records=args.chunk_records,
    )
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
