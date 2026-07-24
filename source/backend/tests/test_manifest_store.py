import json

from scaling_backend.manifest_store import LocalManifestStore, PublishedManifestStore


def test_local_manifest_store_writes_canonical_json_and_returns_file_uri(tmp_path):
    store = LocalManifestStore(tmp_path)
    manifest = {
        "experiment_id": "exp-000001",
        "student_id": "student-1",
        "model_config": {"hidden_size": 128, "num_hidden_layers": 2},
    }

    uri = store.write_manifest("exp-000001", manifest)

    assert uri == (tmp_path / "exp-000001.json").resolve().as_uri()
    loaded = json.loads((tmp_path / "exp-000001.json").read_text(encoding="utf-8"))
    assert loaded == manifest


def test_local_manifest_store_rejects_unsafe_run_ids(tmp_path):
    store = LocalManifestStore(tmp_path)

    try:
        store.write_manifest("../exp-1", {"experiment_id": "../exp-1"})
    except ValueError as exc:
        assert "unsafe run_id" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_local_manifest_store_keeps_existing_manifest_when_replacement_fails(
    monkeypatch, tmp_path
):
    store = LocalManifestStore(tmp_path)
    path = tmp_path / "exp-000001.json"
    first_manifest = {"experiment_id": "exp-000001", "version": 1}
    second_manifest = {"experiment_id": "exp-000001", "version": 2}
    store.write_manifest("exp-000001", first_manifest)
    before = path.read_text(encoding="utf-8")

    original_replace = type(path).replace

    def fail_replace(self, target):
        if target == path.resolve():
            raise OSError("simulated replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(type(path), "replace", fail_replace)

    try:
        store.write_manifest("exp-000001", second_manifest)
    except OSError as exc:
        assert "simulated replace failure" in str(exc)
    else:
        raise AssertionError("expected OSError")

    assert path.read_text(encoding="utf-8") == before
    assert json.loads(path.read_text(encoding="utf-8")) == first_manifest


def test_published_manifest_store_writes_local_json_and_returns_public_uri(tmp_path):
    store = PublishedManifestStore(
        root=tmp_path,
        public_base_uri="https://storage.example/course/manifests/",
    )
    manifest = {
        "experiment_id": "exp-000001",
        "student_id": "student-1",
        "model_config": {"hidden_size": 128, "num_hidden_layers": 2},
    }

    uri = store.write_manifest("exp-000001", manifest)

    assert uri == "https://storage.example/course/manifests/exp-000001.json"
    loaded = json.loads((tmp_path / "exp-000001.json").read_text(encoding="utf-8"))
    assert loaded == manifest


def test_published_manifest_store_rejects_unsafe_uri_prefix(tmp_path):
    try:
        PublishedManifestStore(root=tmp_path, public_base_uri="file:///tmp/manifests")
    except ValueError as exc:
        assert "public_base_uri" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_published_manifest_store_keeps_existing_manifest_when_replacement_fails(
    monkeypatch, tmp_path
):
    store = PublishedManifestStore(
        root=tmp_path,
        public_base_uri="https://storage.example/course/manifests/",
    )
    path = tmp_path / "exp-000001.json"
    first_manifest = {"experiment_id": "exp-000001", "version": 1}
    second_manifest = {"experiment_id": "exp-000001", "version": 2}
    store.write_manifest("exp-000001", first_manifest)
    before = path.read_text(encoding="utf-8")

    original_replace = type(path).replace

    def fail_replace(self, target):
        if target == path.resolve():
            raise OSError("simulated replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(type(path), "replace", fail_replace)

    try:
        store.write_manifest("exp-000001", second_manifest)
    except OSError as exc:
        assert "simulated replace failure" in str(exc)
    else:
        raise AssertionError("expected OSError")

    assert path.read_text(encoding="utf-8") == before
    assert json.loads(path.read_text(encoding="utf-8")) == first_manifest
