from __future__ import annotations

import time
from typing import Any, Iterable, Mapping

import requests


DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "system_failed",
}


class ScalingApiError(RuntimeError):
    def __init__(self, *, status_code: int, body: Mapping[str, Any]) -> None:
        self.status_code = int(status_code)
        self.body = dict(body)
        error = str(self.body.get("error", "http_error"))
        message = str(self.body.get("message", self.body))
        super().__init__(f"HTTP {self.status_code} {error}: {message}")


class ScalingApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        session: Any | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout_seconds = float(timeout_seconds)

    def get_budget(self) -> dict[str, Any]:
        return self._request("GET", "/budget")

    def submit_experiment(
        self,
        *,
        config: Mapping[str, Any],
        requested_runtime_seconds: int,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/submit",
            json={
                "config": dict(config),
                "requested_runtime_seconds": int(requested_runtime_seconds),
            },
        )

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/experiment/{experiment_id}")

    def list_experiments(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/experiments")["experiments"])

    def wait_for_experiment(
        self,
        experiment_id: str,
        *,
        poll_interval_seconds: float = 30.0,
        terminal_statuses: Iterable[str] = DEFAULT_TERMINAL_STATUSES,
        max_polls: int | None = None,
    ) -> dict[str, Any]:
        terminal = set(terminal_statuses)
        polls = 0
        while True:
            result = self.get_experiment(experiment_id)
            polls += 1
            if result.get("status") in terminal:
                return result
            if max_polls is not None and polls >= max_polls:
                return result
            time.sleep(poll_interval_seconds)

    def set_final_submission(
        self,
        *,
        training_config: Mapping[str, Any],
        predicted_final_loss: float,
        predicted_final_loss_lower: float,
        predicted_final_loss_upper: float,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/final_submission",
            json={
                "training_config": dict(training_config),
                "predicted_final_loss": float(predicted_final_loss),
                "predicted_final_loss_lower": float(predicted_final_loss_lower),
                "predicted_final_loss_upper": float(predicted_final_loss_upper),
            },
        )

    def get_final_submission(self) -> dict[str, Any]:
        return self._request("GET", "/final_submission")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self.session.request(
            method,
            self.base_url + path,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=json,
            timeout=self.timeout_seconds,
        )
        try:
            body = response.json()
        except ValueError:
            body = {"raw_response": response.text}
        if response.status_code < 200 or response.status_code >= 300:
            raise ScalingApiError(status_code=response.status_code, body=body)
        return body


def small_example_config(
    *,
    train_tokens: int = 4096,
    learning_rate: float = 3e-4,
    num_hidden_layers: int = 2,
    hidden_size: int = 128,
    num_attention_heads: int = 1,
    num_evals: int = 1,
) -> dict[str, Any]:
    return {
        "model": {
            "num_hidden_layers": int(num_hidden_layers),
            "hidden_size": int(hidden_size),
            "num_attention_heads": int(num_attention_heads),
        },
        "training": {
            "train_tokens": int(train_tokens),
            "learning_rate": float(learning_rate),
            "num_evals": int(num_evals),
        },
    }
