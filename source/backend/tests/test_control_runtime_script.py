from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


class FakeResponse:
    status_code = 400
    text = '{"error": "invalid_config", "message": "training contains unsupported field(s): seed"}'

    def json(self):
        return {
            "error": "invalid_config",
            "message": "training contains unsupported field(s): seed",
        }

    def raise_for_status(self):
        raise RuntimeError("HTTPError traceback should not be surfaced")


def _load_control_runtime_test_module():
    script_path = (
        Path(__file__).resolve().parents[3].parent
        / "assignment-3-control-runtime"
        / "test.py"
    )
    spec = importlib.util.spec_from_file_location("control_runtime_test_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_runtime_api_request_prints_bad_request_json_without_traceback(
    monkeypatch,
    capsys,
):
    module = _load_control_runtime_test_module()

    def fake_request(*_args, **_kwargs):
        return FakeResponse()

    monkeypatch.setattr(module.requests, "request", fake_request)

    with pytest.raises(SystemExit) as exc_info:
        module.api_request("POST", "/submit", json={"config": {}})

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "invalid_config",
        "message": "training contains unsupported field(s): seed",
    }
