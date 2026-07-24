import hashlib
import json
import math
import sys

import pytest
import requests

from scaling_backend.worker.run import (
    WorkerCallbackClient,
    WorkerManifestError,
    load_manifest,
    load_trainer,
    main,
    run_worker,
)


class FakeResponse:
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text


class FakeSession:
    def __init__(self):
        self.gets = []
        self.posts = []
        self.response = FakeResponse()

    def get(self, url, timeout=None):
        self.gets.append({"url": url, "timeout": timeout})
        return self.response

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append(
            {"url": url, "json": json, "headers": headers or {}, "timeout": timeout}
        )
        return self.response


class RaisingGetSession(FakeSession):
    def get(self, url, timeout=None):
        self.gets.append({"url": url, "timeout": timeout})
        raise requests.Timeout("network timeout")


class RaisingPostSession(FakeSession):
    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append(
            {"url": url, "json": json, "headers": headers or {}, "timeout": timeout}
        )
        raise requests.ConnectionError("callback host unresolved")


def _manifest():
    return {
        "experiment_id": "exp-1",
        "student_id": "student-1",
        "model_config": {"num_hidden_layers": 2},
        "data_config": {"train_tokens": 1024},
        "validation_config": {
            "eval_manifest_id": "exploratory-eval-v0",
            "validation_tokens_per_eval": 262_144,
            "validation_batches_per_eval": 256,
        },
        "runtime_config": {"max_runtime_seconds": 60},
    }


def _final_manifest():
    return {
        "run_kind": "final",
        "final_run_id": "final-run-000001",
        "final_submission_id": "final-submission-000001",
        "student_id": "student-1",
        "model_config": {"num_hidden_layers": 2},
        "data_config": {"train_tokens": 1024},
        "validation_config": {
            "eval_manifest_id": "final-eval-hidden-v0",
            "validation_tokens_per_eval": 524_288,
            "validation_batches_per_eval": 512,
        },
        "runtime_config": {"max_runtime_seconds": 60},
    }


def _write_tokenized_index(index_dir, *, shard_payload=b"\x01\x00\x00\x00", sha256=None):
    index_dir.mkdir(parents=True, exist_ok=True)
    shard_path = index_dir / "tokens-000000.bin"
    shard_path.write_bytes(shard_payload)
    expected_hash = (
        sha256
        if sha256 is not None
        else hashlib.sha256(shard_payload).hexdigest()
    )
    index_path = index_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "total_tokens": len(shard_payload) // 4,
                "source_token_counts": {"test": len(shard_payload) // 4},
                "shards": [
                    {
                        "path": shard_path.name,
                        "tokens": len(shard_payload) // 4,
                        "dtype": "uint32",
                        "sha256": expected_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return index_path


def test_load_manifest_from_local_json_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")

    manifest = load_manifest(str(path))

    assert manifest["experiment_id"] == "exp-1"
    assert manifest["model_config"] == {"num_hidden_layers": 2}


def test_load_manifest_from_file_uri(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")

    manifest = load_manifest(path.resolve().as_uri())

    assert manifest["experiment_id"] == "exp-1"
    assert manifest["model_config"] == {"num_hidden_layers": 2}


def test_load_manifest_from_inline_json_string():
    manifest = load_manifest(json.dumps(_manifest()))

    assert manifest["student_id"] == "student-1"


def test_load_manifest_from_https_uri_fetches_json_with_session():
    session = FakeSession()
    session.response = FakeResponse(status_code=200, text=json.dumps(_manifest()))

    manifest = load_manifest(
        "https://storage.example/manifests/exp-1.json",
        session=session,
        timeout_seconds=9,
    )

    assert manifest["experiment_id"] == "exp-1"
    assert session.gets == [
        {"url": "https://storage.example/manifests/exp-1.json", "timeout": 9}
    ]


def test_load_manifest_rejects_remote_non_2xx_response():
    session = FakeSession()
    session.response = FakeResponse(status_code=403, text="access denied")

    with pytest.raises(WorkerManifestError, match="HTTP 403"):
        load_manifest("https://storage.example/manifests/exp-1.json", session=session)


def test_load_manifest_rejects_unsupported_remote_scheme_with_clear_message():
    with pytest.raises(WorkerManifestError, match="unsupported manifest URI scheme: s3"):
        load_manifest("s3://staff-bucket/scaling/manifests/exp-1.json")


def test_load_manifest_rejects_missing_required_sections():
    with pytest.raises(WorkerManifestError) as exc_info:
        load_manifest(json.dumps({"experiment_id": "exp-1"}))

    assert "student_id" in str(exc_info.value)
    assert "model_config" in str(exc_info.value)


@pytest.mark.parametrize(
    ("validation_config", "message"),
    [
        (None, "manifest missing required fields: validation_config"),
        ({}, "validation_config.eval_manifest_id"),
        (
            {
                "eval_manifest_id": "exploratory-eval-v0",
                "validation_tokens_per_eval": 0,
                "validation_batches_per_eval": 256,
            },
            "validation_config.validation_tokens_per_eval",
        ),
        (
            {
                "eval_manifest_id": "exploratory-eval-v0",
                "validation_tokens_per_eval": 262_144,
                "validation_batches_per_eval": "many",
            },
            "validation_config.validation_batches_per_eval",
        ),
    ],
)
def test_load_manifest_rejects_invalid_validation_config(validation_config, message):
    manifest = _manifest()
    if validation_config is None:
        del manifest["validation_config"]
    else:
        manifest["validation_config"] = validation_config

    with pytest.raises(WorkerManifestError, match=message):
        load_manifest(json.dumps(manifest))


def test_callback_client_posts_bearer_token_and_payload():
    session = FakeSession()
    client = WorkerCallbackClient(
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        session=session,
        timeout_seconds=5,
    )

    client.post_event({"event_type": "heartbeat", "experiment_id": "exp-1"})

    assert session.posts == [
        {
            "url": "https://backend/internal/provider-events",
            "json": {"event_type": "heartbeat", "experiment_id": "exp-1"},
            "headers": {"Authorization": "Bearer secret-token"},
            "timeout": 5,
        }
    ]


def test_callback_client_raises_for_non_2xx_response():
    session = FakeSession()
    session.response = FakeResponse(status_code=500, text="server exploded")
    client = WorkerCallbackClient(
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        session=session,
    )

    with pytest.raises(RuntimeError, match="callback failed"):
        client.post_event({"event_type": "heartbeat"})


def test_run_worker_can_emit_jsonl_events_without_http_callback(tmp_path):
    session = RaisingPostSession()
    event_log_path = tmp_path / "worker-events" / "exp-1.jsonl"

    def trainer(_manifest, emit_event):
        emit_event({"event_type": "validation", "step": 10, "loss": 3.14})
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.14, 3.0],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=_manifest(),
        callback_url="https://api.internal/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
        event_sink="jsonl",
        event_log_path=event_log_path,
    )

    assert exit_code == 0
    assert session.posts == []
    events = [
        json.loads(line)
        for line in event_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_type"] for event in events] == [
        "worker_started",
        "validation",
        "worker_completed",
    ]
    assert all(event["experiment_id"] == "exp-1" for event in events)
    assert all(event["event_id"].startswith("exp-1:") for event in events)
    assert len({event["event_id"] for event in events}) == len(events)


def test_run_worker_posts_completed_payload_from_trainer_result():
    session = FakeSession()

    def trainer(manifest, emit_event):
        assert manifest["validation_config"] == {
            "eval_manifest_id": "exploratory-eval-v0",
            "validation_tokens_per_eval": 262_144,
            "validation_batches_per_eval": 256,
        }
        emit_event({"event_type": "validation", "step": 10, "loss": 3.14})
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.14, 3.0],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=_manifest(),
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
    )

    assert exit_code == 0
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "validation",
        "worker_completed",
    ]
    assert session.posts[-1]["json"]["final_validation_loss"] == 3.0
    assert session.posts[-1]["json"]["validation_losses"] == [3.14, 3.0]


def test_run_worker_applies_runtime_watchdog_from_top_level_manifest_limit():
    session = FakeSession()
    manifest = _manifest()
    manifest["max_runtime_seconds"] = 42
    manifest["runtime_config"]["max_runtime_seconds"] = 60
    entered = False
    observed_limits = []

    class RecordingTimeout:
        def __enter__(self):
            nonlocal entered
            entered = True

        def __exit__(self, exc_type, exc, traceback):
            return False

    def runtime_timeout_factory(seconds):
        observed_limits.append(seconds)
        return RecordingTimeout()

    def trainer(_manifest, _emit_event):
        assert entered is True
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.0],
            "actual_runtime_seconds": 42,
        }

    exit_code = run_worker(
        manifest=manifest,
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
        runtime_timeout_factory=runtime_timeout_factory,
    )

    assert exit_code == 0
    assert observed_limits == [42]
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "worker_completed",
    ]


def test_run_worker_reports_timeout_when_runtime_watchdog_fires_before_completion():
    session = FakeSession()
    manifest = _manifest()
    manifest["max_runtime_seconds"] = 42
    trainer_called = False

    class ImmediateTimeout:
        def __enter__(self):
            raise TimeoutError("worker runtime exceeded 42 seconds")

        def __exit__(self, exc_type, exc, traceback):
            return False

    def trainer(_manifest, _emit_event):
        nonlocal trainer_called
        trainer_called = True
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.0],
            "actual_runtime_seconds": 42,
        }

    exit_code = run_worker(
        manifest=manifest,
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
        runtime_timeout_factory=lambda _seconds: ImmediateTimeout(),
    )

    assert exit_code == 1
    assert trainer_called is False
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "worker_failed",
    ]
    assert session.posts[-1]["json"]["failure_type"] == "TimeoutError"
    assert session.posts[-1]["json"]["message"] == "worker runtime exceeded 42 seconds"


def test_run_worker_rejects_bad_tokenized_index_before_trainer_runs(tmp_path):
    session = FakeSession()
    manifest = _manifest()
    manifest["data_config"]["tokenized_index_uri"] = str(
        _write_tokenized_index(tmp_path / "train", sha256="0" * 64)
    )
    trainer_called = False

    def trainer(_manifest, _emit_event):
        nonlocal trainer_called
        trainer_called = True
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.0],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=manifest,
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
        tokenized_index_validation_mode="full",
    )

    assert exit_code == 1
    assert trainer_called is False
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "worker_failed",
    ]
    assert session.posts[-1]["json"]["failure_type"] == "TokenizedIndexError"
    assert "hash mismatch" in session.posts[-1]["json"]["message"]


def test_run_worker_defaults_to_metadata_tokenized_index_validation(tmp_path):
    session = FakeSession()
    manifest = _manifest()
    manifest["data_config"]["tokenized_index_uri"] = str(
        _write_tokenized_index(tmp_path / "train", sha256="0" * 64)
    )
    trainer_called = False

    def trainer(_manifest, _emit_event):
        nonlocal trainer_called
        trainer_called = True
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.0],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=manifest,
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
    )

    assert exit_code == 0
    assert trainer_called is True
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "worker_completed",
    ]


def test_run_worker_metadata_validation_does_not_fetch_remote_shards():
    session = FakeSession()
    session.response = FakeResponse(
        status_code=200,
        text=json.dumps(
            {
                "total_tokens": 1,
                "source_token_counts": {"test": 1},
                "shards": [
                    {
                        "path": "tokens-000000.bin",
                        "tokens": 1,
                        "dtype": "uint32",
                        "byte_order": "little",
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
    )
    manifest = _manifest()
    manifest["data_config"][
        "tokenized_index_uri"
    ] = "https://storage.example/tokenized/train/index.json"

    def trainer(_manifest, _emit_event):
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.0],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=manifest,
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
    )

    assert exit_code == 0
    assert session.gets == [
        {
            "url": "https://storage.example/tokenized/train/index.json",
            "timeout": 30,
        }
    ]


def test_run_worker_accepts_valid_train_and_validation_tokenized_indexes(tmp_path):
    session = FakeSession()
    train_index = _write_tokenized_index(
        tmp_path / "train",
        shard_payload=b"\x01\x00\x00\x00\x02\x00\x00\x00",
    )
    validation_index = _write_tokenized_index(
        tmp_path / "validation",
        shard_payload=b"\x03\x00\x00\x00",
    )
    manifest = _manifest()
    manifest["data_config"]["tokenized_index_uri"] = str(train_index)
    manifest["validation_config"][
        "tokenized_index_uri"
    ] = validation_index.resolve().as_uri()
    trainer_seen_manifest = None

    def trainer(seen_manifest, _emit_event):
        nonlocal trainer_seen_manifest
        trainer_seen_manifest = seen_manifest
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.0],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=manifest,
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
    )

    assert exit_code == 0
    assert trainer_seen_manifest is manifest
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "worker_completed",
    ]


def test_run_worker_reports_remote_tokenized_index_fetch_failure_as_index_error():
    session = RaisingGetSession()
    manifest = _manifest()
    manifest["data_config"][
        "tokenized_index_uri"
    ] = "https://storage.example/tokenized/train/index.json"
    trainer_called = False

    def trainer(_manifest, _emit_event):
        nonlocal trainer_called
        trainer_called = True
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.0],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=manifest,
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
    )

    assert exit_code == 1
    assert trainer_called is False
    assert session.posts[-1]["json"]["failure_type"] == "TokenizedIndexError"
    assert "tokenized index fetch failed" in session.posts[-1]["json"]["message"]


def test_run_worker_can_emit_periodic_heartbeats_during_trainer_execution():
    session = FakeSession()

    class FakeStopEvent:
        def __init__(self):
            self._set = False

        def set(self):
            self._set = True

        def wait(self, _seconds):
            return self._set

    class FakeThread:
        def __init__(self, target, daemon=None):
            self.target = target
            self.daemon = daemon
            self.stop_event = FakeStopEvent()
            self.started = False
            self.joined = False

        def start(self):
            self.started = True

        def join(self, timeout=None):
            self.joined = True

    threads = []

    def thread_factory(target, daemon=None):
        thread = FakeThread(target=target, daemon=daemon)
        threads.append(thread)
        return thread

    stop_events = []

    def stop_event_factory():
        event = FakeStopEvent()
        stop_events.append(event)
        return event

    def trainer(_manifest, _emit_event):
        threads[0].target()
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.0],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=_manifest(),
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
        heartbeat_interval_seconds=30,
        thread_factory=thread_factory,
        stop_event_factory=stop_event_factory,
        heartbeat_once=True,
    )

    assert exit_code == 0
    assert threads[0].started is True
    assert threads[0].daemon is True
    assert threads[0].joined is True
    assert stop_events[0]._set is True
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "heartbeat",
        "worker_completed",
    ]
    assert session.posts[1]["json"]["heartbeat_interval_seconds"] == 30


def test_run_worker_callback_failure_before_heartbeat_start_does_not_join_unstarted_thread(
    capsys,
):
    session = RaisingPostSession()
    joined = False
    trainer_called = False

    class FakeStopEvent:
        def set(self):
            pass

        def wait(self, _seconds):
            return False

    class FakeThread:
        def __init__(self, target, daemon=None):
            self.target = target
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

        def join(self, timeout=None):
            nonlocal joined
            joined = True
            if not self.started:
                raise RuntimeError("cannot join thread before it is started")

    def trainer(_manifest, _emit_event):
        nonlocal trainer_called
        trainer_called = True
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.0],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=_manifest(),
        callback_url="https://api.internal/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
        heartbeat_interval_seconds=60,
        thread_factory=FakeThread,
        stop_event_factory=FakeStopEvent,
    )

    assert exit_code == 1
    assert trainer_called is False
    assert joined is False
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "worker_failed",
    ]
    stderr = capsys.readouterr().err
    assert (
        "worker failed for experiment_id=exp-1: "
        "ConnectionError: callback host unresolved"
    ) in stderr
    assert "Traceback (most recent call last):" in stderr
    assert "requests.exceptions.ConnectionError: callback host unresolved" in stderr
    assert "failed to report worker_failed event" in stderr


def test_run_worker_uses_final_run_id_for_final_manifest_callbacks():
    session = FakeSession()

    def trainer(manifest, emit_event):
        assert manifest["final_run_id"] == "final-run-000001"
        emit_event({"event_type": "validation", "step": 10, "loss": 2.75})
        return {
            "final_validation_loss": 2.7,
            "validation_losses": [2.75, 2.7],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=_final_manifest(),
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
    )

    assert exit_code == 0
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "validation",
        "worker_completed",
    ]
    assert all(call["json"]["final_run_id"] == "final-run-000001" for call in session.posts)
    assert all("experiment_id" not in call["json"] for call in session.posts)


def test_run_worker_posts_failed_payload_when_trainer_raises(capsys):
    session = FakeSession()

    def trainer(_manifest, _emit_event):
        raise FloatingPointError("nan loss")

    exit_code = run_worker(
        manifest=_manifest(),
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
    )

    assert exit_code == 1
    assert session.posts[-1]["json"]["event_type"] == "worker_failed"
    assert session.posts[-1]["json"]["failure_type"] == "FloatingPointError"
    assert session.posts[-1]["json"]["message"] == "nan loss"
    stderr = capsys.readouterr().err
    assert (
        "worker failed for experiment_id=exp-1: FloatingPointError: nan loss"
        in stderr
    )
    assert "Traceback (most recent call last):" in stderr
    assert "FloatingPointError: nan loss" in stderr


def test_run_worker_rejects_non_finite_trainer_result_before_completed_event():
    session = FakeSession()

    def trainer(_manifest, _emit_event):
        return {
            "final_validation_loss": math.inf,
            "validation_losses": [3.5, math.inf],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=_manifest(),
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
    )

    assert exit_code == 1
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "worker_failed",
    ]
    assert session.posts[-1]["json"]["failure_type"] == "WorkerResultError"


def test_run_worker_rejects_final_loss_that_is_not_last_validation_loss():
    session = FakeSession()

    def trainer(_manifest, _emit_event):
        return {
            "final_validation_loss": 3.0,
            "validation_losses": [3.2, 3.1],
            "actual_runtime_seconds": 120,
        }

    exit_code = run_worker(
        manifest=_manifest(),
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
    )

    assert exit_code == 1
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "worker_failed",
    ]
    assert session.posts[-1]["json"]["failure_type"] == "WorkerResultError"
    assert "validation_losses[-1]" in session.posts[-1]["json"]["message"]


def test_run_worker_requires_actual_runtime_seconds_in_trainer_result():
    session = FakeSession()

    def trainer(_manifest, _emit_event):
        return {"final_validation_loss": 3.0, "validation_losses": [3.2, 3.0]}

    exit_code = run_worker(
        manifest=_manifest(),
        callback_url="https://backend/internal/provider-events",
        callback_token="secret-token",
        trainer=trainer,
        session=session,
    )

    assert exit_code == 1
    assert session.posts[-1]["json"]["event_type"] == "worker_failed"
    assert session.posts[-1]["json"]["failure_type"] == "WorkerResultError"


def test_load_trainer_imports_callable_from_module_spec(tmp_path):
    trainer_module = tmp_path / "trainer_impl.py"
    trainer_module.write_text(
        "def train(manifest, emit_event):\n"
        "    emit_event({'event_type': 'validation', 'step': 1, 'loss': 3.5})\n"
        "    return {'final_validation_loss': 3.4, 'validation_losses': [3.5, 3.4]}\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        trainer = load_trainer("trainer_impl:train")
    finally:
        sys.path.remove(str(tmp_path))

    emitted = []
    result = trainer(_manifest(), emitted.append)

    assert emitted == [{"event_type": "validation", "step": 1, "loss": 3.5}]
    assert result["final_validation_loss"] == 3.4


def test_main_uses_configured_trainer_and_callback_session(monkeypatch, tmp_path):
    trainer_module = tmp_path / "trainer_main_impl.py"
    trainer_module.write_text(
        "def train(manifest, emit_event):\n"
        "    assert manifest['experiment_id'] == 'exp-1'\n"
        "    emit_event({'event_type': 'validation', 'step': 2, 'loss': 3.2})\n"
        "    return {'final_validation_loss': 3.1, 'validation_losses': [3.2, 3.1], 'actual_runtime_seconds': 120}\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    session = FakeSession()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("CALLBACK_TOKEN", "secret-token")

    exit_code = main(
        [
            "--manifest-uri",
            str(manifest_path),
            "--callback-url",
            "https://backend/internal/provider-events",
            "--callback-token-env",
            "CALLBACK_TOKEN",
            "--trainer",
            "trainer_main_impl:train",
        ],
        session=session,
    )

    assert exit_code == 0
    assert [call["json"]["event_type"] for call in session.posts] == [
        "worker_started",
        "validation",
        "worker_completed",
    ]
    assert session.posts[0]["headers"] == {"Authorization": "Bearer secret-token"}
    assert session.posts[-1]["json"]["final_validation_loss"] == 3.1


def test_main_can_select_jsonl_event_sink(monkeypatch, tmp_path):
    trainer_module = tmp_path / "trainer_jsonl_impl.py"
    trainer_module.write_text(
        "def train(manifest, emit_event):\n"
        "    emit_event({'event_type': 'validation', 'step': 2, 'loss': 3.2})\n"
        "    return {'final_validation_loss': 3.1, 'validation_losses': [3.2, 3.1], 'actual_runtime_seconds': 120}\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    event_log_path = tmp_path / "events" / "exp-1.jsonl"
    session = RaisingPostSession()
    monkeypatch.syspath_prepend(str(tmp_path))

    exit_code = main(
        [
            "--manifest-uri",
            str(manifest_path),
            "--callback-url",
            "https://api.internal/internal/provider-events",
            "--callback-token-env",
            "CALLBACK_TOKEN",
            "--trainer",
            "trainer_jsonl_impl:train",
            "--event-sink",
            "jsonl",
            "--event-log-path",
            str(event_log_path),
        ],
        session=session,
    )

    assert exit_code == 0
    assert session.posts == []
    assert [
        json.loads(line)["event_type"]
        for line in event_log_path.read_text(encoding="utf-8").splitlines()
    ] == ["worker_started", "validation", "worker_completed"]
