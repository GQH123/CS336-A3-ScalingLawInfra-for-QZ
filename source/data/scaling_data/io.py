from __future__ import annotations

import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def open_text_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as handle:
            yield handle
    elif path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("zstandard is required for .zst output") from exc
        with path.open("wb") as raw_handle:
            with zstd.open(raw_handle, "wt", encoding="utf-8") as handle:
                yield handle
    else:
        with path.open("w", encoding="utf-8") as handle:
            yield handle


def iter_jsonl_records(path: Path) -> Iterator[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)
    elif path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("zstandard is required for .zst input") from exc
        with path.open("rb") as raw_handle:
            with zstd.open(raw_handle, "rt", encoding="utf-8") as handle:
                for line in handle:
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def write_jsonl_record(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_input_files(input_dir: Path) -> Iterator[Path]:
    suffixes = (".jsonl", ".jsonl.gz", ".jsonl.zst")
    yield from sorted(path for path in input_dir.iterdir() if path.name.endswith(suffixes))
