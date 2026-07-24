from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse


class ManifestStore(Protocol):
    def write_manifest(self, run_id: str, manifest: Mapping[str, Any]) -> str:
        ...


class LocalManifestStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_manifest(self, run_id: str, manifest: Mapping[str, Any]) -> str:
        safe_run_id = _safe_run_id(run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        path = (self.root / f"{safe_run_id}.json").resolve()
        root = self.root.resolve()
        if root not in path.parents:
            raise ValueError(f"unsafe run_id for manifest path: {run_id}")
        _write_canonical_json_atomic(path, manifest)
        return path.as_uri()


class PublishedManifestStore:
    def __init__(self, root: str | Path, public_base_uri: str):
        self.root = Path(root)
        self.public_base_uri = _safe_public_base_uri(public_base_uri)

    def write_manifest(self, run_id: str, manifest: Mapping[str, Any]) -> str:
        safe_run_id = _safe_run_id(run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        path = (self.root / f"{safe_run_id}.json").resolve()
        root = self.root.resolve()
        if root not in path.parents:
            raise ValueError(f"unsafe run_id for manifest path: {run_id}")
        _write_canonical_json_atomic(path, manifest)
        return f"{self.public_base_uri}/{safe_run_id}.json"


def _safe_run_id(run_id: str) -> str:
    if not run_id or run_id.strip() != run_id:
        raise ValueError(f"unsafe run_id for manifest path: {run_id}")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if any(char not in allowed for char in run_id):
        raise ValueError(f"unsafe run_id for manifest path: {run_id}")
    return run_id


def _safe_public_base_uri(public_base_uri: str) -> str:
    value = public_base_uri.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https", "s3", "gs"} or not parsed.netloc:
        raise ValueError("public_base_uri must be an http(s), s3, or gs URI")
    return value


def _write_canonical_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(dict(payload), sort_keys=True, indent=2) + "\n"
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
