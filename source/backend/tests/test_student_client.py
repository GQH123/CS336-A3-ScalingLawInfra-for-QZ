import pytest

from scaling_backend import student_client


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = "response text"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")


class FakeSession:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


def test_student_client_uses_bearer_auth_and_submit_envelope():
    session = FakeSession(
        [
            FakeResponse(payload={"remaining_seconds": 3600}),
            FakeResponse(
                payload={
                    "experiment_id": "exp-000001",
                    "status": "submitted",
                }
            ),
        ]
    )
    client = student_client.ScalingApiClient(
        base_url="https://backend.example/api/",
        api_key="student-secret",
        session=session,
    )

    budget = client.get_budget()
    submitted = client.submit_experiment(
        config=student_client.small_example_config(train_tokens=2048),
        requested_runtime_seconds=300,
    )

    assert budget == {"remaining_seconds": 3600}
    assert submitted["experiment_id"] == "exp-000001"
    assert session.calls == [
        {
            "method": "GET",
            "url": "https://backend.example/api/budget",
            "headers": {"Authorization": "Bearer student-secret"},
            "json": None,
            "timeout": 30.0,
        },
        {
            "method": "POST",
            "url": "https://backend.example/api/submit",
            "headers": {"Authorization": "Bearer student-secret"},
            "json": {
                "config": {
                    "model": {
                        "num_hidden_layers": 2,
                        "hidden_size": 128,
                        "num_attention_heads": 1,
                    },
                    "training": {
                        "train_tokens": 2048,
                        "learning_rate": 0.0003,
                        "num_evals": 1,
                    },
                },
                "requested_runtime_seconds": 300,
            },
            "timeout": 30.0,
        },
    ]


def test_student_client_poll_and_final_submission_helpers():
    session = FakeSession(
        [
            FakeResponse(payload={"status": "running"}),
            FakeResponse(payload={"status": "completed", "final_validation_loss": 3.2}),
            FakeResponse(payload={"final_submission_id": "final-submission-000001"}),
        ]
    )
    client = student_client.ScalingApiClient(
        base_url="https://backend.example",
        api_key="student-secret",
        session=session,
    )

    polled = client.wait_for_experiment(
        "exp-000001",
        poll_interval_seconds=0,
        terminal_statuses={"completed"},
        max_polls=2,
    )
    final = client.set_final_submission(
        training_config=student_client.small_example_config(train_tokens=4096),
        predicted_final_loss=2.8,
        predicted_final_loss_lower=2.7,
        predicted_final_loss_upper=2.9,
    )

    assert polled["status"] == "completed"
    assert final["final_submission_id"] == "final-submission-000001"
    assert session.calls[0]["url"] == "https://backend.example/experiment/exp-000001"
    assert session.calls[2]["url"] == "https://backend.example/final_submission"
    assert session.calls[2]["json"]["predicted_final_loss_upper"] == 2.9


def test_student_client_default_terminal_statuses_match_public_api_contract():
    assert student_client.DEFAULT_TERMINAL_STATUSES == {
        "completed",
        "failed",
        "cancelled",
        "system_failed",
    }


def test_student_client_http_errors_include_api_json_body():
    session = FakeSession(
        [
            FakeResponse(
                status_code=400,
                payload={
                    "error": "invalid_config",
                    "message": (
                        "training.train_batch_size must be divisible by "
                        "worker_gpu_count"
                    ),
                },
            )
        ]
    )
    client = student_client.ScalingApiClient(
        base_url="https://backend.example",
        api_key="student-secret",
        session=session,
    )

    with pytest.raises(student_client.ScalingApiError) as exc_info:
        client.submit_experiment(
            config=student_client.small_example_config(train_tokens=2048),
            requested_runtime_seconds=300,
        )

    error = exc_info.value
    assert error.status_code == 400
    assert error.body == {
        "error": "invalid_config",
        "message": "training.train_batch_size must be divisible by worker_gpu_count",
    }
    assert "HTTP 400 invalid_config" in str(error)
    assert "worker_gpu_count" in str(error)
