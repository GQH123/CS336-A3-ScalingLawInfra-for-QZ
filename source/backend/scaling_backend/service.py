from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from scaling_backend.config_validation import resolve_training_config
from scaling_backend.dispatcher import (
    ExperimentDispatcher,
    ExperimentEvent,
    ExperimentRecord,
    ExperimentStatus,
)
from scaling_backend.exports import export_course_records
from scaling_backend.manifest_store import ManifestStore
from scaling_backend.prediction_scoring import compute_prediction_metrics
from scaling_backend.providers.contracts import (
    ProviderAdapter,
    ProviderArtifacts,
    ProviderJobStatus,
)


class DuplicateExperimentError(ValueError):
    def __init__(self, experiment_id: str):
        super().__init__(f"duplicate config for this student: {experiment_id}")
        self.experiment_id = experiment_id


class StudentAccessError(PermissionError):
    pass


STUDENT_FAILURE_REASONS = frozenset(
    {
        "timeout",
        "numerical",
        "resource",
        "invalid_runtime_config",
        "provider_rejected",
        "unknown_student_caused",
    }
)

SYSTEM_FAILURE_REASONS = frozenset(
    {
        "provider_outage",
        "worker_lost",
        "infrastructure_error",
        "admin_intervention",
        "unknown_system",
    }
)

INTERNAL_FAILURE_REASONS = frozenset(
    {
        "provider_cancelled",
        "provider_failed_without_worker_callback",
        "missing_worker_completed_callback",
        "provider_lost",
        "provider_status_unknown",
        "cancel_rejected",
    }
)


@dataclass(frozen=True)
class SubmitResponse:
    experiment_id: str
    status: str
    budget_reserved_seconds: int
    resolved_config: Mapping[str, Any] = field(default_factory=dict)
    resource_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BudgetSnapshot:
    student_id: str
    total_seconds: int
    reserved_seconds: int
    charged_seconds: int

    @property
    def remaining_seconds(self) -> int:
        return self.total_seconds - self.reserved_seconds - self.charged_seconds


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    status: str
    validation_losses: list[float]
    final_validation_loss: float | None
    failure_reason: str = ""
    used_runtime_seconds: int | None = None
    completed_at: str = ""
    failed_at: str = ""
    resource_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FinalSubmission:
    final_submission_id: str
    student_id: str
    training_config: Mapping[str, Any]
    predicted_final_loss: float
    predicted_final_loss_lower: float
    predicted_final_loss_upper: float
    updated_at: str
    frozen_at: str = ""
    frozen_by: str = ""
    freeze_reason: str = ""


@dataclass(frozen=True)
class FinalRun:
    final_run_id: str
    final_submission_id: str
    student_id: str
    status: str
    provider_job_id: str
    provider_name: str
    manifest: Mapping[str, Any]
    predicted_final_loss: float
    predicted_final_loss_lower: float
    predicted_final_loss_upper: float
    resolved_config: Mapping[str, Any] = field(default_factory=dict)
    validation_losses: list[float] = field(default_factory=list)
    actual_final_validation_loss: float | None = None
    failure_reason: str = ""
    staff_failure_detail: str = ""


@dataclass(frozen=True)
class BudgetAdjustment:
    adjustment_id: str
    student_id: str
    seconds: int
    reason: str
    actor: str
    created_at: str
    before_charged_seconds: int
    after_charged_seconds: int
    experiment_id: str = ""


@dataclass(frozen=True)
class AdminAction:
    action_id: str
    action_type: str
    actor: str
    created_at: str
    reason: str
    student_id: str
    experiment_id: str = ""
    final_run_id: str = ""
    before_status: str = ""
    after_status: str = ""


@dataclass(frozen=True)
class WorkerEvent:
    student_id: str
    event_type: str
    timestamp: str
    payload: Mapping[str, Any]
    experiment_id: str = ""
    final_run_id: str = ""


@dataclass
class _BudgetState:
    total_seconds: int
    reserved_by_experiment: dict[str, int] = field(default_factory=dict)
    charged_seconds: int = 0

    @property
    def reserved_seconds(self) -> int:
        return sum(self.reserved_by_experiment.values())

    @property
    def remaining_seconds(self) -> int:
        return self.total_seconds - self.reserved_seconds - self.charged_seconds


@dataclass
class _ExperimentState:
    experiment_id: str
    student_id: str
    config_hash: str
    config: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    reserved_runtime_seconds: int
    manifest: Mapping[str, Any] = field(default_factory=dict)
    status: str = "submitted"
    validation_losses: list[float] = field(default_factory=list)
    final_validation_loss: float | None = None
    failure_reason: str = ""
    staff_failure_detail: str = ""
    used_runtime_seconds: int | None = None
    completed_at: str = ""
    failed_at: str = ""


class ExperimentService:
    def __init__(
        self,
        *,
        provider: ProviderAdapter,
        total_budget_seconds: int,
        now: Callable[[], str],
        manifest_uri_builder: Callable[[str], str],
        callback_url: str,
        code_version: str = "local-dev",
        data_manifest_id: str = "exploratory-train-v0",
        eval_manifest_id: str = "exploratory-eval-v0",
        final_data_manifest_id: str = "final-train-v0",
        final_eval_manifest_id: str = "final-eval-v0",
        validation_tokens_per_eval: int = 262_144,
        final_validation_tokens_per_eval: int = 524_288,
        train_tokenized_index_uri: str = "",
        validation_tokenized_index_uri: str = "",
        final_train_tokenized_index_uri: str = "",
        final_validation_tokenized_index_uri: str = "",
        manifest_store: ManifestStore | None = None,
        max_active_experiments_per_student: int = 0,
        max_active_experiments_global: int = 0,
        worker_gpu_count: int = 0,
    ):
        max_active = int(max_active_experiments_per_student)
        if max_active < 0:
            raise ValueError("max_active_experiments_per_student must be non-negative")
        max_active_global = int(max_active_experiments_global)
        if max_active_global < 0:
            raise ValueError("max_active_experiments_global must be non-negative")
        worker_gpus = int(worker_gpu_count)
        if worker_gpus < 0:
            raise ValueError("worker_gpu_count must be non-negative")
        validation_tokens = int(validation_tokens_per_eval)
        if validation_tokens <= 0:
            raise ValueError("validation_tokens_per_eval must be positive")
        final_validation_tokens = int(final_validation_tokens_per_eval)
        if final_validation_tokens <= 0:
            raise ValueError("final_validation_tokens_per_eval must be positive")
        self.provider = provider
        self.now = now
        self.dispatcher = ExperimentDispatcher(provider=provider, now=now)
        self.total_budget_seconds = int(total_budget_seconds)
        self.manifest_uri_builder = manifest_uri_builder
        self.callback_url = callback_url
        self.code_version = code_version
        self.data_manifest_id = data_manifest_id
        self.eval_manifest_id = eval_manifest_id
        self.final_data_manifest_id = final_data_manifest_id
        self.final_eval_manifest_id = final_eval_manifest_id
        self.validation_tokens_per_eval = validation_tokens
        self.final_validation_tokens_per_eval = final_validation_tokens
        self.train_tokenized_index_uri = _optional_private_uri(train_tokenized_index_uri)
        self.validation_tokenized_index_uri = _optional_private_uri(
            validation_tokenized_index_uri
        )
        self.final_train_tokenized_index_uri = _optional_private_uri(
            final_train_tokenized_index_uri
        )
        self.final_validation_tokenized_index_uri = _optional_private_uri(
            final_validation_tokenized_index_uri
        )
        self.manifest_store = manifest_store
        self.max_active_experiments_per_student = max_active
        self.max_active_experiments_global = max_active_global
        self.worker_gpu_count = worker_gpus
        self._experiments: dict[str, _ExperimentState] = {}
        self._student_config_index: dict[tuple[str, str], str] = {}
        self._budgets: dict[str, _BudgetState] = {}
        self._final_submissions: dict[str, FinalSubmission] = {}
        self._frozen_final_submissions: dict[str, FinalSubmission] = {}
        self._final_submissions_frozen = False
        self._final_runs_launched = False
        self._final_runs: dict[str, FinalRun] = {}
        self._budget_adjustments: list[BudgetAdjustment] = []
        self._admin_actions: list[AdminAction] = []
        self._worker_events: list[WorkerEvent] = []
        self._next_experiment_number = 1
        self._next_final_submission_number = 1
        self._next_final_run_number = 1
        self._next_budget_adjustment_number = 1
        self._next_admin_action_number = 1

    def submit(
        self,
        *,
        student_id: str,
        config: Mapping[str, Any],
        requested_runtime_seconds: int,
    ) -> SubmitResponse:
        runtime = int(requested_runtime_seconds)
        if runtime <= 0:
            raise ValueError("requested_runtime_seconds must be positive")

        config_hash = _stable_hash(config)
        duplicate_id = self._student_config_index.get((student_id, config_hash))
        if duplicate_id:
            raise DuplicateExperimentError(duplicate_id)

        resolved_config = resolve_training_config(
            config,
            requested_runtime_seconds=runtime,
            code_version=self.code_version,
            data_manifest_id=self.data_manifest_id,
            eval_manifest_id=self.eval_manifest_id,
            validation_tokens_per_eval=self.validation_tokens_per_eval,
            worker_gpu_count=self.worker_gpu_count,
        )

        budget = self._budget_for(student_id)
        if runtime > budget.remaining_seconds:
            raise ValueError("requested runtime exceeds remaining budget")

        experiment_id = self._new_experiment_id()
        state = _ExperimentState(
            experiment_id=experiment_id,
            student_id=student_id,
            config_hash=config_hash,
            config=dict(config),
            resolved_config=resolved_config,
            reserved_runtime_seconds=runtime,
            status=ExperimentStatus.QUEUED.value,
        )
        self._experiments[experiment_id] = state
        self._student_config_index[(student_id, config_hash)] = experiment_id
        budget.reserved_by_experiment[experiment_id] = runtime

        try:
            manifest = self._persist_manifest_if_configured(
                state.experiment_id,
                self._build_manifest(state),
            )
            state.resolved_config = dict(manifest["resolved_config"])
            state.manifest = dict(manifest)
            if self._can_launch_experiment(state):
                self._launch_experiment(state)
        except Exception:
            self._experiments.pop(experiment_id, None)
            self._student_config_index.pop((student_id, config_hash), None)
            budget.reserved_by_experiment.pop(experiment_id, None)
            self._next_experiment_number -= 1
            raise
        return SubmitResponse(
            experiment_id=experiment_id,
            status=state.status,
            budget_reserved_seconds=runtime,
            resolved_config=state.resolved_config,
            resource_warnings=_resource_warnings_from_resolved(state.resolved_config),
        )

    def _active_experiment_count_for_student(self, student_id: str) -> int:
        return sum(
            1
            for state in self._experiments.values()
            if state.student_id == student_id and _is_active_status(state.status)
        )

    def _active_experiment_count(self) -> int:
        return sum(
            1
            for state in self._experiments.values()
            if _is_active_status(state.status)
        )

    def _can_launch_experiment(self, state: _ExperimentState) -> bool:
        if not _is_active_status(state.status):
            return False
        if self._experiment_has_provider_job(state.experiment_id):
            return False
        if (
            self.max_active_experiments_per_student > 0
            and self._launched_experiment_count_for_student(state.student_id)
            >= self.max_active_experiments_per_student
        ):
            return False
        if (
            self.max_active_experiments_global > 0
            and self._launched_experiment_count()
            >= self.max_active_experiments_global
        ):
            return False
        return True

    def _launch_experiment(self, state: _ExperimentState) -> None:
        if self._experiment_has_provider_job(state.experiment_id):
            return
        manifest = dict(state.manifest)
        if not manifest:
            manifest = self._persist_manifest_if_configured(
                state.experiment_id,
                self._build_manifest(state),
            )
            state.resolved_config = dict(manifest["resolved_config"])
            state.manifest = dict(manifest)
        record = self.dispatcher.submit(manifest)
        state.status = record.status.value
        state.manifest = dict(record.manifest)

    def _launch_queued_experiments(self) -> None:
        while True:
            launched = False
            for state in self._queued_experiments_in_fair_share_order():
                if not self._can_launch_experiment(state):
                    continue
                self._launch_experiment(state)
                launched = True
            if not launched:
                return

    def _launch_queued_experiments_best_effort(
        self,
        *,
        trigger: str,
        experiment_id: str = "",
        final_run_id: str = "",
    ) -> None:
        try:
            self._launch_queued_experiments()
        except Exception as exc:
            detail = str(exc).strip()
            if detail:
                reason = (
                    f"{trigger}: queued experiment launch failed with "
                    f"{type(exc).__name__}: {detail[:300]}"
                )
            else:
                reason = (
                    f"{trigger}: queued experiment launch failed with "
                    f"{type(exc).__name__}"
                )
            self._record_admin_action(
                action_type="launch_queued_experiments_failed",
                actor="system",
                reason=reason,
                student_id="",
                experiment_id=experiment_id,
                final_run_id=final_run_id,
                before_status="queued",
                after_status="queued",
            )

    def _queued_experiments_in_fair_share_order(self) -> list[_ExperimentState]:
        queued_states = [
            state
            for state in self._experiments.values()
            if state.status == ExperimentStatus.QUEUED.value
            and not self._experiment_has_provider_job(state.experiment_id)
        ]
        rows = [
            {
                "experiment_id": state.experiment_id,
                "student_id": state.student_id,
                "queue_position": queue_position,
            }
            for queue_position, state in enumerate(queued_states, start=1)
        ]
        by_experiment_id = {state.experiment_id: state for state in queued_states}
        return [
            by_experiment_id[str(row["experiment_id"])]
            for row in _fair_share_order(rows)
        ]

    def _launched_experiment_count_for_student(self, student_id: str) -> int:
        return sum(
            1
            for state in self._experiments.values()
            if state.student_id == student_id
            and _is_active_status(state.status)
            and self._experiment_has_provider_job(state.experiment_id)
        )

    def _launched_experiment_count(self) -> int:
        return sum(
            1
            for state in self._experiments.values()
            if _is_active_status(state.status)
            and self._experiment_has_provider_job(state.experiment_id)
        )

    def _experiment_has_provider_job(self, experiment_id: str) -> bool:
        try:
            self.dispatcher.get(experiment_id)
        except KeyError:
            return False
        return True

    def get_budget(self, student_id: str) -> BudgetSnapshot:
        budget = self._budget_for(student_id)
        return BudgetSnapshot(
            student_id=student_id,
            total_seconds=budget.total_seconds,
            reserved_seconds=budget.reserved_seconds,
            charged_seconds=budget.charged_seconds,
        )

    def record_worker_event(self, payload: Mapping[str, Any]) -> bool:
        event_key = _worker_event_dedup_key(payload)
        if event_key is not None and self._has_worker_event_dedup_key(event_key):
            return False
        if payload.get("final_run_id"):
            return self._record_final_run_worker_event(payload)
        experiment_id = _require_string(payload, "experiment_id")
        event_type = _require_string(payload, "event_type")
        state = self._experiments[experiment_id]

        if event_type == "worker_started":
            _require_active_for_worker_event(state.status, state.experiment_id)
            self._record_experiment_event(state, event_type, payload)
            state.status = ExperimentStatus.RUNNING.value
            return True

        if event_type == "heartbeat":
            _require_active_for_worker_event(state.status, state.experiment_id)
            self._record_experiment_event(state, event_type, payload)
            state.status = ExperimentStatus.RUNNING.value
            return True

        if event_type == "validation":
            _require_active_for_worker_event(state.status, state.experiment_id)
            loss = _finite_float(payload["loss"])
            self._record_experiment_event(state, event_type, payload)
            state.validation_losses.append(loss)
            state.status = ExperimentStatus.RUNNING.value
            return True

        if event_type == "worker_completed":
            _require_active_for_terminal_worker_event(state.status, state.experiment_id)
            validation_losses = _finite_float_list(payload.get("validation_losses", []))
            completed_losses = validation_losses or list(state.validation_losses)
            final_loss = _completed_final_loss_from_payload(payload, completed_losses)
            completed_at = self._record_experiment_event(state, event_type, payload)
            state.validation_losses = completed_losses
            state.final_validation_loss = final_loss
            state.status = ExperimentStatus.COMPLETED.value
            charged_runtime = _clipped_runtime_charge(
                payload.get("actual_runtime_seconds"),
                reserved_runtime_seconds=state.reserved_runtime_seconds,
            )
            state.used_runtime_seconds = charged_runtime
            state.completed_at = completed_at
            state.failed_at = ""
            self._charge_and_release(state, charged_runtime)
            self._launch_queued_experiments_best_effort(
                trigger="worker_completed",
                experiment_id=state.experiment_id,
            )
            return True

        if event_type == "worker_failed":
            _require_active_for_terminal_worker_event(state.status, state.experiment_id)
            validation_losses = _finite_float_list(payload.get("validation_losses", []))
            failed_at = self._record_experiment_event(state, event_type, payload)
            if validation_losses:
                state.validation_losses = validation_losses
            failure_reason = _normalize_worker_failure_reason(payload)
            state.status = (
                ExperimentStatus.SYSTEM_FAILED.value
                if failure_reason in SYSTEM_FAILURE_REASONS
                else ExperimentStatus.FAILED.value
            )
            state.failure_reason = failure_reason
            state.staff_failure_detail = _staff_failure_detail_from_mapping(payload)
            if failure_reason in SYSTEM_FAILURE_REASONS:
                state.used_runtime_seconds = _clipped_system_runtime(
                    payload.get("actual_runtime_seconds"),
                    reserved_runtime_seconds=state.reserved_runtime_seconds,
                )
                state.completed_at = ""
                state.failed_at = failed_at
                self._release_reserved_runtime(state)
                self._launch_queued_experiments_best_effort(
                    trigger="worker_failed",
                    experiment_id=state.experiment_id,
                )
                return True
            if failure_reason == "timeout":
                charged_runtime = state.reserved_runtime_seconds
            else:
                charged_runtime = _clipped_runtime_charge(
                    payload.get("actual_runtime_seconds"),
                    reserved_runtime_seconds=state.reserved_runtime_seconds,
                )
            state.used_runtime_seconds = charged_runtime
            state.completed_at = ""
            state.failed_at = failed_at
            self._charge_and_release(state, charged_runtime)
            self._launch_queued_experiments_best_effort(
                trigger="worker_failed",
                experiment_id=state.experiment_id,
            )
            return True

        raise ValueError(f"unknown worker event_type: {event_type}")

    def _record_final_run_worker_event(self, payload: Mapping[str, Any]) -> bool:
        final_run_id = _require_string(payload, "final_run_id")
        event_type = _require_string(payload, "event_type")
        final_run = self._final_runs[final_run_id]

        if event_type == "worker_started":
            _require_active_for_worker_event(final_run.status, final_run_id)
            self._record_final_run_event(final_run, event_type, payload)
            self._final_runs[final_run_id] = _replace_final_run(
                final_run, status=ExperimentStatus.RUNNING.value
            )
            return True

        if event_type == "heartbeat":
            _require_active_for_worker_event(final_run.status, final_run_id)
            self._record_final_run_event(final_run, event_type, payload)
            self._final_runs[final_run_id] = _replace_final_run(
                final_run, status=ExperimentStatus.RUNNING.value
            )
            return True

        if event_type == "validation":
            _require_active_for_worker_event(final_run.status, final_run_id)
            validation_losses = [*final_run.validation_losses, _finite_float(payload["loss"])]
            self._record_final_run_event(final_run, event_type, payload)
            self._final_runs[final_run_id] = _replace_final_run(
                final_run, validation_losses=validation_losses
            )
            return True

        if event_type == "worker_completed":
            _require_active_for_terminal_worker_event(final_run.status, final_run_id)
            validation_losses = _finite_float_list(payload.get("validation_losses", []))
            completed_losses = validation_losses or list(final_run.validation_losses)
            final_loss = _completed_final_loss_from_payload(payload, completed_losses)
            self._record_final_run_event(final_run, event_type, payload)
            self._final_runs[final_run_id] = _replace_final_run(
                final_run,
                status=ExperimentStatus.COMPLETED.value,
                validation_losses=completed_losses,
                actual_final_validation_loss=final_loss,
                failure_reason="",
                staff_failure_detail="",
            )
            return True

        if event_type == "worker_failed":
            _require_active_for_terminal_worker_event(final_run.status, final_run_id)
            validation_losses = _finite_float_list(payload.get("validation_losses", []))
            failure_reason = _normalize_worker_failure_reason(payload)
            self._record_final_run_event(final_run, event_type, payload)
            self._final_runs[final_run_id] = _replace_final_run(
                final_run,
                status=(
                    ExperimentStatus.SYSTEM_FAILED.value
                    if failure_reason in SYSTEM_FAILURE_REASONS
                    else ExperimentStatus.FAILED.value
                ),
                validation_losses=validation_losses or final_run.validation_losses,
                failure_reason=failure_reason,
                staff_failure_detail=_staff_failure_detail_from_mapping(payload),
            )
            return True

        raise ValueError(f"unknown worker event_type: {event_type}")

    def get_result(self, student_id: str, experiment_id: str) -> ExperimentResult:
        state = self._experiments[experiment_id]
        if state.student_id != student_id:
            raise StudentAccessError("experiment does not belong to this student")
        return _result_from_state(state)

    def list_results(self, student_id: str) -> list[ExperimentResult]:
        return [
            _result_from_state(state)
            for state in self._experiments.values()
            if state.student_id == student_id
        ]

    def set_final_submission(
        self,
        *,
        student_id: str,
        training_config: Mapping[str, Any],
        predicted_final_loss: float,
        predicted_final_loss_lower: float,
        predicted_final_loss_upper: float,
    ) -> FinalSubmission:
        if self._final_submissions_frozen:
            raise ValueError("final submissions are frozen")
        if not isinstance(training_config, Mapping):
            raise ValueError("training_config must be an object")
        self._resolve_final_submission_config(training_config)
        point = float(predicted_final_loss)
        lower = float(predicted_final_loss_lower)
        upper = float(predicted_final_loss_upper)
        if not all(math.isfinite(value) for value in (lower, point, upper)):
            raise ValueError("predicted final loss values must be finite")
        if not lower <= point <= upper:
            raise ValueError(
                "prediction interval must satisfy lower <= point <= upper"
            )
        submission = FinalSubmission(
            final_submission_id=self._final_submission_id_for(student_id),
            student_id=student_id,
            training_config=dict(training_config),
            predicted_final_loss=point,
            predicted_final_loss_lower=lower,
            predicted_final_loss_upper=upper,
            updated_at=self.now(),
        )
        self._final_submissions[student_id] = submission
        return submission

    def get_final_submission(self, student_id: str) -> FinalSubmission:
        if student_id in self._frozen_final_submissions:
            return self._frozen_final_submissions[student_id]
        return self._final_submissions[student_id]

    def freeze_final_submissions(self, *, actor: str, reason: str) -> list[FinalSubmission]:
        if self._final_submissions_frozen:
            return list(self._frozen_final_submissions.values())
        self._final_submissions_frozen = True
        frozen_at = self.now()
        for student_id, submission in self._final_submissions.items():
            frozen = FinalSubmission(
                final_submission_id=submission.final_submission_id,
                student_id=submission.student_id,
                training_config=dict(submission.training_config),
                predicted_final_loss=submission.predicted_final_loss,
                predicted_final_loss_lower=submission.predicted_final_loss_lower,
                predicted_final_loss_upper=submission.predicted_final_loss_upper,
                updated_at=submission.updated_at,
                frozen_at=frozen_at,
                frozen_by=actor,
                freeze_reason=reason,
            )
            self._frozen_final_submissions[student_id] = frozen
        self._record_admin_action(
            action_type="freeze_final_submissions",
            actor=actor,
            reason=reason,
            student_id="",
            before_status="open",
            after_status="frozen",
        )
        return list(self._frozen_final_submissions.values())

    def launch_final_runs(
        self,
        *,
        max_runtime_seconds: int,
        actor: str = "staff",
        reason: str = "deadline",
    ) -> list[FinalRun]:
        if not self._final_submissions_frozen:
            raise ValueError("freeze final submissions before launching final runs")
        if self._final_runs_launched:
            return list(self._final_runs.values())

        final_submission_ids_with_runs = {
            final_run.final_submission_id for final_run in self._final_runs.values()
        }
        for submission in self._frozen_final_submissions.values():
            if submission.final_submission_id in final_submission_ids_with_runs:
                continue
            final_run_id = f"final-run-{self._next_final_run_number:06d}"
            resolved_config = resolve_training_config(
                submission.training_config,
                requested_runtime_seconds=int(max_runtime_seconds),
                code_version=self.code_version,
                data_manifest_id=self.final_data_manifest_id,
                eval_manifest_id=self.final_eval_manifest_id,
                validation_tokens_per_eval=self.final_validation_tokens_per_eval,
                worker_gpu_count=self.worker_gpu_count,
            )
            manifest = self._persist_manifest_if_configured(
                final_run_id,
                self._build_final_manifest(
                    final_run_id=final_run_id,
                    submission=submission,
                    max_runtime_seconds=int(max_runtime_seconds),
                    resolved_config=resolved_config,
                ),
            )
            resolved_config = dict(manifest["resolved_config"])
            provider_submission = self.provider.submit(manifest)
            self._next_final_run_number += 1
            self._final_runs[final_run_id] = FinalRun(
                final_run_id=final_run_id,
                final_submission_id=submission.final_submission_id,
                student_id=submission.student_id,
                status=ExperimentStatus.SUBMITTED.value,
                provider_job_id=provider_submission.provider_job_id,
                provider_name=provider_submission.provider_name,
                manifest=dict(manifest),
                predicted_final_loss=submission.predicted_final_loss,
                predicted_final_loss_lower=submission.predicted_final_loss_lower,
                predicted_final_loss_upper=submission.predicted_final_loss_upper,
                resolved_config=resolved_config,
            )
            final_submission_ids_with_runs.add(submission.final_submission_id)
        self._final_runs_launched = True
        self._record_admin_action(
            action_type="launch_final_runs",
            actor=actor,
            reason=reason,
            student_id="",
            before_status="frozen",
            after_status="launched",
        )
        return list(self._final_runs.values())

    def admin_list_experiments(
        self,
        *,
        student_id: str = "",
        status: str = "",
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for state in self._experiments.values():
            if student_id and state.student_id != student_id:
                continue
            if status and state.status != status:
                continue
            row = {
                "experiment_id": state.experiment_id,
                "student_id": state.student_id,
                "status": state.status,
                "validation_losses": list(state.validation_losses),
                "final_validation_loss": state.final_validation_loss,
                "failure_reason": state.failure_reason,
                "staff_failure_detail": state.staff_failure_detail,
                "used_runtime_seconds": state.used_runtime_seconds,
                "completed_at": state.completed_at,
                "failed_at": state.failed_at,
                "reserved_runtime_seconds": state.reserved_runtime_seconds,
                "config_hash": state.config_hash,
                "resolved_config": dict(state.resolved_config),
            }
            try:
                record = self.dispatcher.get(state.experiment_id)
            except KeyError:
                record = None
            if record is not None:
                row.update(
                    {
                        "provider_name": record.provider_name,
                        "provider_job_id": record.provider_job_id,
                        "provider_status": record.provider_status.value,
                        "provider_artifacts": self._provider_artifacts_payload(
                            provider_job_id=record.provider_job_id,
                            provider_name=record.provider_name,
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "provider_name": "",
                        "provider_job_id": "",
                        "provider_status": "",
                        "provider_artifacts": {},
                    }
                )
            rows.append(row)
        return rows

    def admin_get_experiment(self, experiment_id: str) -> dict[str, Any]:
        state = self._experiments[experiment_id]
        detail = _experiment_admin_payload(state)
        try:
            record = self.dispatcher.get(experiment_id)
        except KeyError:
            record = None
        if record is not None:
            detail["provider_name"] = record.provider_name
            detail["provider_job_id"] = record.provider_job_id
            detail["provider_status"] = record.provider_status.value
            detail["provider_artifacts"] = self._provider_artifacts_payload(
                provider_job_id=record.provider_job_id,
                provider_name=record.provider_name,
            )
        else:
            detail["provider_name"] = ""
            detail["provider_job_id"] = ""
            detail["provider_status"] = ""
            detail["provider_artifacts"] = {}
        detail["events"] = [
            _worker_event_payload(event)
            for event in self._worker_events
            if event.experiment_id == experiment_id
        ]
        return detail

    def admin_cancel_experiment(
        self, experiment_id: str, *, reason: str, actor: str
    ) -> ExperimentResult:
        state = self._experiments[experiment_id]
        if not _is_active_status(state.status):
            return _result_from_state(state)

        before_status = state.status
        if not self._experiment_has_provider_job(experiment_id):
            state.status = ExperimentStatus.CANCELLED.value
            state.failure_reason = "admin_intervention"
            state.staff_failure_detail = "cancelled before provider dispatch"
            self._mark_experiment_terminal_failure(
                state,
                used_runtime_seconds=0,
            )
            self._release_reserved_runtime(state)
            self._record_admin_action(
                action_type="cancel_experiment",
                actor=actor,
                reason=reason,
                student_id=state.student_id,
                experiment_id=experiment_id,
                before_status=before_status,
                after_status=state.status,
            )
            self._launch_queued_experiments_best_effort(
                trigger="cancel_experiment",
                experiment_id=experiment_id,
            )
            return _result_from_state(state)

        record = self.dispatcher.cancel(experiment_id, reason=reason, actor=actor)
        state.status = record.status.value
        launch_queued = False
        if record.status is ExperimentStatus.CANCELLED:
            state.failure_reason = "admin_intervention"
            state.staff_failure_detail = _provider_record_staff_failure_detail(record)
            self._mark_experiment_terminal_failure(
                state,
                used_runtime_seconds=0,
                failed_at=_provider_record_terminal_timestamp(record),
            )
            self._release_reserved_runtime(state)
            launch_queued = True
        self._record_admin_action(
            action_type="cancel_experiment",
            actor=actor,
            reason=reason,
            student_id=state.student_id,
            experiment_id=experiment_id,
            before_status=before_status,
            after_status=state.status,
        )
        if launch_queued:
            self._launch_queued_experiments_best_effort(
                trigger="cancel_experiment",
                experiment_id=experiment_id,
            )
        return _result_from_state(state)

    def admin_mark_experiment_system_failed(
        self,
        experiment_id: str,
        *,
        failure_reason: str,
        staff_failure_detail: str,
        refund_seconds: int,
        reason: str,
        actor: str,
    ) -> ExperimentResult:
        if failure_reason not in SYSTEM_FAILURE_REASONS:
            raise ValueError(f"failure_reason must be one of: {sorted(SYSTEM_FAILURE_REASONS)}")
        if int(refund_seconds) < 0:
            raise ValueError("refund_seconds must be non-negative")

        state = self._experiments[experiment_id]
        before_status = state.status
        state.status = ExperimentStatus.SYSTEM_FAILED.value
        state.failure_reason = failure_reason
        state.staff_failure_detail = staff_failure_detail
        state.final_validation_loss = None
        state.used_runtime_seconds = (
            0 if state.used_runtime_seconds is None else state.used_runtime_seconds
        )
        state.completed_at = ""
        state.failed_at = self.now()
        self._release_reserved_runtime(state)
        if int(refund_seconds) > 0:
            self.admin_apply_budget_adjustment(
                student_id=state.student_id,
                seconds=-int(refund_seconds),
                reason=reason,
                actor=actor,
                experiment_id=experiment_id,
            )
        self._record_admin_action(
            action_type="mark_experiment_system_failed",
            actor=actor,
            reason=reason,
            student_id=state.student_id,
            experiment_id=experiment_id,
            before_status=before_status,
            after_status=state.status,
        )
        self._launch_queued_experiments_best_effort(
            trigger="mark_experiment_system_failed",
            experiment_id=experiment_id,
        )
        return _result_from_state(state)

    def admin_poll_experiment(self, experiment_id: str) -> ExperimentResult:
        state = self._experiments[experiment_id]
        if not _is_active_status(state.status):
            return _result_from_state(state)
        if not self._experiment_has_provider_job(experiment_id):
            self._launch_queued_experiments_best_effort(
                trigger="admin_poll_experiment",
                experiment_id=experiment_id,
            )
            if not self._experiment_has_provider_job(experiment_id):
                return _result_from_state(state)

        record = self.dispatcher.poll(experiment_id)

        if record.status in {
            ExperimentStatus.SUBMITTED,
            ExperimentStatus.QUEUED,
            ExperimentStatus.RUNNING,
        }:
            state.status = record.status.value
            return _result_from_state(state)

        if record.status is ExperimentStatus.CANCELLED:
            state.status = ExperimentStatus.CANCELLED.value
            state.failure_reason = "provider_cancelled"
            state.staff_failure_detail = _provider_record_staff_failure_detail(record)
            charged_runtime = _provider_terminal_runtime_charge(state)
            self._mark_experiment_terminal_failure(
                state,
                used_runtime_seconds=charged_runtime,
                failed_at=_provider_record_terminal_timestamp(record),
            )
            self._charge_and_release(state, charged_runtime)
            self._launch_queued_experiments_best_effort(
                trigger="admin_poll_experiment",
                experiment_id=experiment_id,
            )
            return _result_from_state(state)

        if record.status is ExperimentStatus.FAILED:
            state.status = ExperimentStatus.FAILED.value
            state.failure_reason = "provider_failed_without_worker_callback"
            state.staff_failure_detail = _provider_record_staff_failure_detail(record)
            charged_runtime = _provider_terminal_runtime_charge(state)
            self._mark_experiment_terminal_failure(
                state,
                used_runtime_seconds=charged_runtime,
                failed_at=_provider_record_terminal_timestamp(record),
            )
            self._charge_and_release(state, charged_runtime)
            self._launch_queued_experiments_best_effort(
                trigger="admin_poll_experiment",
                experiment_id=experiment_id,
            )
            return _result_from_state(state)

        if record.status is ExperimentStatus.COMPLETED:
            state.status = ExperimentStatus.LOST.value
            state.failure_reason = "missing_worker_completed_callback"
            state.staff_failure_detail = _provider_record_staff_failure_detail(record)
            charged_runtime = _provider_terminal_runtime_charge(state)
            self._mark_experiment_terminal_failure(
                state,
                used_runtime_seconds=charged_runtime,
                failed_at=_provider_record_terminal_timestamp(record),
            )
            self._charge_and_release(state, charged_runtime)
            self._launch_queued_experiments_best_effort(
                trigger="admin_poll_experiment",
                experiment_id=experiment_id,
            )
            return _result_from_state(state)

        if record.status is ExperimentStatus.LOST:
            state.status = ExperimentStatus.LOST.value
            state.failure_reason = "provider_lost"
            state.staff_failure_detail = _provider_record_staff_failure_detail(record)
            charged_runtime = _provider_terminal_runtime_charge(state)
            self._mark_experiment_terminal_failure(
                state,
                used_runtime_seconds=charged_runtime,
                failed_at=_provider_record_terminal_timestamp(record),
            )
            self._charge_and_release(state, charged_runtime)
            self._launch_queued_experiments_best_effort(
                trigger="admin_poll_experiment",
                experiment_id=experiment_id,
            )
            return _result_from_state(state)

        state.status = record.status.value
        state.failure_reason = "provider_status_unknown"
        state.staff_failure_detail = _provider_record_staff_failure_detail(record)
        charged_runtime = _provider_terminal_runtime_charge(state)
        self._mark_experiment_terminal_failure(
            state,
            used_runtime_seconds=charged_runtime,
            failed_at=_provider_record_terminal_timestamp(record),
        )
        self._charge_and_release(state, charged_runtime)
        self._launch_queued_experiments_best_effort(
            trigger="admin_poll_experiment",
            experiment_id=experiment_id,
        )
        return _result_from_state(state)

    def admin_poll_final_run(self, final_run_id: str) -> FinalRun:
        final_run = self._final_runs[final_run_id]
        if not _is_active_status(final_run.status):
            return final_run
        snapshot = self.provider.get_status(final_run.provider_job_id)

        if snapshot.status in {
            ProviderJobStatus.SUBMITTED,
            ProviderJobStatus.QUEUED,
            ProviderJobStatus.RUNNING,
        }:
            updated = _replace_final_run(final_run, status=_status_value(snapshot.status))
        elif snapshot.status is ProviderJobStatus.SUCCEEDED:
            updated = _replace_final_run(
                final_run,
                status=ExperimentStatus.LOST.value,
                failure_reason="missing_worker_completed_callback",
                staff_failure_detail=_provider_snapshot_staff_failure_detail(snapshot),
            )
        elif snapshot.status is ProviderJobStatus.FAILED:
            updated = _replace_final_run(
                final_run,
                status=ExperimentStatus.FAILED.value,
                failure_reason="provider_failed_without_worker_callback",
                staff_failure_detail=_provider_snapshot_staff_failure_detail(snapshot),
            )
        elif snapshot.status is ProviderJobStatus.CANCELLED:
            updated = _replace_final_run(
                final_run,
                status=ExperimentStatus.CANCELLED.value,
                failure_reason="provider_cancelled",
                staff_failure_detail=_provider_snapshot_staff_failure_detail(snapshot),
            )
        elif snapshot.status is ProviderJobStatus.LOST:
            updated = _replace_final_run(
                final_run,
                status=ExperimentStatus.LOST.value,
                failure_reason="provider_lost",
                staff_failure_detail=_provider_snapshot_staff_failure_detail(snapshot),
            )
        else:
            updated = _replace_final_run(
                final_run,
                status=ExperimentStatus.UNKNOWN.value,
                failure_reason="provider_status_unknown",
                staff_failure_detail=_provider_snapshot_staff_failure_detail(snapshot),
            )
        self._final_runs[final_run_id] = updated
        return updated

    def admin_cancel_final_run(
        self, final_run_id: str, *, reason: str, actor: str
    ) -> FinalRun:
        final_run = self._final_runs[final_run_id]
        if not _is_active_status(final_run.status):
            return final_run
        before_status = final_run.status
        result = self.provider.cancel(final_run.provider_job_id, reason=reason, actor=actor)
        status = (
            ExperimentStatus.CANCELLED.value
            if result.accepted
            else _status_value(result.status)
        )
        failure_reason = "admin_intervention" if result.accepted else "cancel_rejected"
        updated = _replace_final_run(
            final_run,
            status=status,
            failure_reason=failure_reason,
            staff_failure_detail=_staff_failure_detail_from_mapping(
                {
                    "message": result.message,
                    "provider_name": result.provider_name,
                    "provider_job_id": result.provider_job_id,
                    **dict(result.metadata),
                }
            ),
        )
        self._final_runs[final_run_id] = updated
        self._record_admin_action(
            action_type="cancel_final_run",
            actor=actor,
            reason=reason,
            student_id=final_run.student_id,
            final_run_id=final_run_id,
            before_status=before_status,
            after_status=updated.status,
        )
        return updated

    def admin_mark_final_run_system_failed(
        self,
        final_run_id: str,
        *,
        failure_reason: str,
        staff_failure_detail: str,
        reason: str,
        actor: str,
    ) -> FinalRun:
        if failure_reason not in SYSTEM_FAILURE_REASONS:
            raise ValueError(f"failure_reason must be one of: {sorted(SYSTEM_FAILURE_REASONS)}")
        final_run = self._final_runs[final_run_id]
        before_status = final_run.status
        updated = _replace_final_run(
            final_run,
            status=ExperimentStatus.SYSTEM_FAILED.value,
            actual_final_validation_loss=None,
            failure_reason=failure_reason,
            staff_failure_detail=staff_failure_detail,
        )
        self._final_runs[final_run_id] = updated
        self._record_admin_action(
            action_type="mark_final_run_system_failed",
            actor=actor,
            reason=reason,
            student_id=final_run.student_id,
            final_run_id=final_run_id,
            before_status=before_status,
            after_status=updated.status,
        )
        return updated

    def admin_poll_active_runs(self) -> dict[str, list[dict[str, Any]]]:
        active_experiment_ids = [
            state.experiment_id
            for state in self._experiments.values()
            if _is_active_status(state.status)
        ]
        active_final_run_ids = [
            final_run.final_run_id
            for final_run in self._final_runs.values()
            if _is_active_status(final_run.status)
        ]
        exploratory = [
            self._active_experiment_poll_row(result)
            for result in (
                self.admin_poll_experiment(experiment_id)
                for experiment_id in active_experiment_ids
            )
        ]
        final = [
            self._active_final_run_poll_row(final_run)
            for final_run in (
                self.admin_poll_final_run(final_run_id)
                for final_run_id in active_final_run_ids
            )
        ]
        return {"exploratory": exploratory, "final": final}

    def _active_experiment_poll_row(self, result: ExperimentResult) -> dict[str, Any]:
        row: dict[str, Any] = {
            "experiment_id": result.experiment_id,
            "status": result.status,
            "failure_reason": result.failure_reason,
        }
        if _is_active_status(result.status):
            return row
        try:
            record = self.dispatcher.get(result.experiment_id)
        except KeyError:
            return row
        row["provider_artifacts"] = self._provider_artifacts_payload(
            provider_job_id=record.provider_job_id,
            provider_name=record.provider_name,
        )
        return row

    def _active_final_run_poll_row(self, final_run: FinalRun) -> dict[str, Any]:
        row: dict[str, Any] = {
            "final_run_id": final_run.final_run_id,
            "status": final_run.status,
            "failure_reason": final_run.failure_reason,
        }
        if _is_active_status(final_run.status):
            return row
        row["provider_artifacts"] = self._provider_artifacts_payload(
            provider_job_id=final_run.provider_job_id,
            provider_name=final_run.provider_name,
        )
        return row

    def admin_apply_budget_adjustment(
        self,
        *,
        student_id: str,
        seconds: int,
        reason: str,
        actor: str,
        experiment_id: str = "",
    ) -> BudgetAdjustment:
        budget = self._budget_for(student_id)
        before_charged_seconds = budget.charged_seconds
        after_charged_seconds = before_charged_seconds + int(seconds)
        adjustment = BudgetAdjustment(
            adjustment_id=(
                f"budget-adjustment-{self._next_budget_adjustment_number:06d}"
            ),
            student_id=student_id,
            seconds=int(seconds),
            reason=reason,
            actor=actor,
            created_at=self.now(),
            before_charged_seconds=before_charged_seconds,
            after_charged_seconds=after_charged_seconds,
            experiment_id=experiment_id,
        )
        self._next_budget_adjustment_number += 1
        self._budget_adjustments.append(adjustment)
        budget.charged_seconds = adjustment.after_charged_seconds
        return adjustment

    def admin_export_course_records(self) -> dict[str, list[dict[str, Any]]]:
        worker_events = [_worker_event_payload(event) for event in self._worker_events]
        return {
            "experiments": self.admin_list_experiments(),
            "experiment_events": [
                event
                for event in worker_events
                if event.get("experiment_id")
            ],
            "worker_events": worker_events,
            "budget_snapshots": [
                _budget_snapshot_payload(self.get_budget(student_id))
                for student_id in sorted(self._budgets)
            ],
            "budget_adjustments": [
                _budget_adjustment_payload(adjustment)
                for adjustment in self._budget_adjustments
            ],
            "admin_actions": [
                _admin_action_payload(action)
                for action in self._admin_actions
            ],
            "final_submissions": [
                _final_submission_payload(submission)
                for submission in self._frozen_final_submissions.values()
            ],
            "final_runs": [
                _final_run_payload(
                    final_run,
                    provider_artifacts=self._provider_artifacts_payload(
                        provider_job_id=final_run.provider_job_id,
                        provider_name=final_run.provider_name,
                    ),
                )
                for final_run in self._final_runs.values()
            ],
        }

    def admin_write_course_exports(self, output_dir: str | Path) -> dict[str, str]:
        return export_course_records(self.admin_export_course_records(), output_dir)

    def save_state_snapshot(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(f".{output_path.name}.tmp")
        payload = json.dumps(self._state_snapshot_payload(), sort_keys=True, indent=2) + "\n"
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_path.replace(output_path)
        except Exception:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def load_state_snapshot(self, path: str | Path) -> None:
        source_path = Path(path)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if payload.get("snapshot_version") != 1:
            raise ValueError("unsupported state snapshot version")
        self._experiments = {
            item["experiment_id"]: _experiment_state_from_payload(item)
            for item in payload.get("experiments", [])
        }
        self._student_config_index = {
            (state.student_id, state.config_hash): state.experiment_id
            for state in self._experiments.values()
        }
        self._budgets = {
            student_id: _budget_state_from_payload(item)
            for student_id, item in payload.get("budgets", {}).items()
        }
        self._final_submissions = {
            student_id: _final_submission_from_payload(item)
            for student_id, item in payload.get("final_submissions", {}).items()
        }
        self._frozen_final_submissions = {
            student_id: _final_submission_from_payload(item)
            for student_id, item in payload.get("frozen_final_submissions", {}).items()
        }
        self._final_submissions_frozen = _snapshot_bool_with_fallback(
            payload,
            "final_submissions_frozen",
            fallback=bool(self._frozen_final_submissions),
        )
        self._final_runs = {
            item["final_run_id"]: _final_run_from_payload(item)
            for item in payload.get("final_runs", [])
        }
        self._final_runs_launched = _snapshot_bool_with_fallback(
            payload,
            "final_runs_launched",
            fallback=bool(self._final_runs),
        )
        self._budget_adjustments = [
            _budget_adjustment_from_payload(item)
            for item in payload.get("budget_adjustments", [])
        ]
        self._admin_actions = [
            _admin_action_from_payload(item)
            for item in payload.get("admin_actions", [])
        ]
        self._worker_events = [
            _worker_event_from_payload(item)
            for item in payload.get("worker_events", payload.get("experiment_events", []))
        ]
        self.dispatcher._records = {
            item["experiment_id"]: _dispatcher_record_from_payload(item)
            for item in payload.get("dispatcher_records", [])
        }
        self._restore_provider_jobs_for_snapshot()
        counters = payload.get("counters", {})
        self._next_experiment_number = int(counters.get("next_experiment_number", 1))
        self._next_final_submission_number = int(
            counters.get("next_final_submission_number", 1)
        )
        self._next_final_run_number = int(counters.get("next_final_run_number", 1))
        self._next_budget_adjustment_number = int(
            counters.get("next_budget_adjustment_number", 1)
        )
        self._next_admin_action_number = int(
            counters.get("next_admin_action_number", 1)
        )

    def admin_queue_snapshot(self) -> dict[str, Any]:
        exploratory = [
            {
                "experiment_id": state.experiment_id,
                "student_id": state.student_id,
                "status": state.status,
                "reserved_runtime_seconds": state.reserved_runtime_seconds,
                "queue_position": queue_position,
            }
            for queue_position, state in enumerate(
                (
                    state
                    for state in self._experiments.values()
                    if _is_active_status(state.status)
                ),
                start=1,
            )
        ]
        fair_share_order = _fair_share_order(exploratory)
        fair_share_rank_by_experiment = {
            str(row["experiment_id"]): int(row["fair_share_rank"])
            for row in fair_share_order
        }
        exploratory = [
            {
                **row,
                "fair_share_rank": fair_share_rank_by_experiment[
                    str(row["experiment_id"])
                ],
            }
            for row in exploratory
        ]
        final = [
            {
                "final_run_id": final_run.final_run_id,
                "student_id": final_run.student_id,
                "status": final_run.status,
                "provider_job_id": final_run.provider_job_id,
                "queue_position": queue_position,
            }
            for queue_position, final_run in enumerate(
                (
                    final_run
                    for final_run in self._final_runs.values()
                    if _is_active_status(final_run.status)
                ),
                start=1,
            )
        ]
        return {
            "exploratory": exploratory,
            "final": final,
            "fair_share_order": fair_share_order,
            "active_counts_by_student": _active_counts_by_student(
                exploratory=exploratory,
                final=final,
            ),
        }

    def _state_snapshot_payload(self) -> dict[str, Any]:
        return {
            "snapshot_version": 1,
            "experiments": [
                {
                    **_experiment_admin_payload(state),
                    "manifest": dict(state.manifest),
                }
                for state in self._experiments.values()
            ],
            "budgets": {
                student_id: {
                    "total_seconds": budget.total_seconds,
                    "reserved_by_experiment": dict(budget.reserved_by_experiment),
                    "charged_seconds": budget.charged_seconds,
                }
                for student_id, budget in self._budgets.items()
            },
            "final_submissions": {
                student_id: _final_submission_payload(submission)
                for student_id, submission in self._final_submissions.items()
            },
            "frozen_final_submissions": {
                student_id: _final_submission_payload(submission)
                for student_id, submission in self._frozen_final_submissions.items()
            },
            "final_submissions_frozen": self._final_submissions_frozen,
            "final_runs_launched": self._final_runs_launched,
            "final_runs": [
                _final_run_payload(final_run)
                for final_run in self._final_runs.values()
            ],
            "budget_adjustments": [
                _budget_adjustment_payload(adjustment)
                for adjustment in self._budget_adjustments
            ],
            "admin_actions": [
                _admin_action_payload(action)
                for action in self._admin_actions
            ],
            "experiment_events": [
                _worker_event_payload(event)
                for event in self._worker_events
                if event.experiment_id
            ],
            "worker_events": [
                _worker_event_payload(event)
                for event in self._worker_events
            ],
            "dispatcher_records": [
                _dispatcher_record_payload(record)
                for record in self.dispatcher._records.values()
            ],
            "counters": {
                "next_experiment_number": self._next_experiment_number,
                "next_final_submission_number": self._next_final_submission_number,
                "next_final_run_number": self._next_final_run_number,
                "next_budget_adjustment_number": self._next_budget_adjustment_number,
                "next_admin_action_number": self._next_admin_action_number,
            },
        }

    def _restore_provider_jobs_for_snapshot(self) -> None:
        restore_job = getattr(self.provider, "restore_job", None)
        if not callable(restore_job):
            return
        for record in self.dispatcher._records.values():
            if record.provider_job_id and record.manifest:
                restore_job(
                    record.provider_job_id,
                    record.manifest,
                    cancelled=record.status is ExperimentStatus.CANCELLED,
                )
        for final_run in self._final_runs.values():
            if final_run.provider_job_id and final_run.manifest:
                restore_job(
                    final_run.provider_job_id,
                    final_run.manifest,
                    cancelled=final_run.status == ExperimentStatus.CANCELLED.value,
                )

    def _budget_for(self, student_id: str) -> _BudgetState:
        if student_id not in self._budgets:
            self._budgets[student_id] = _BudgetState(
                total_seconds=self.total_budget_seconds
            )
        return self._budgets[student_id]

    def _provider_artifacts_payload(
        self, *, provider_job_id: str, provider_name: str = ""
    ) -> dict[str, Any]:
        if not provider_job_id:
            return {}
        try:
            artifacts = self.provider.get_artifacts(provider_job_id)
        except Exception:
            return {
                "provider_name": provider_name or self.provider.provider_name,
                "provider_job_id": provider_job_id,
                "logs_uri": "",
                "detail_uri": "",
                "metadata": {},
            }
        return _provider_artifacts_payload(artifacts)

    def _new_experiment_id(self) -> str:
        experiment_id = f"exp-{self._next_experiment_number:06d}"
        self._next_experiment_number += 1
        return experiment_id

    def _record_admin_action(
        self,
        *,
        action_type: str,
        actor: str,
        reason: str,
        student_id: str,
        experiment_id: str = "",
        final_run_id: str = "",
        before_status: str = "",
        after_status: str = "",
    ) -> AdminAction:
        action = AdminAction(
            action_id=f"admin-action-{self._next_admin_action_number:06d}",
            action_type=action_type,
            actor=actor,
            created_at=self.now(),
            reason=reason,
            student_id=student_id,
            experiment_id=experiment_id,
            final_run_id=final_run_id,
            before_status=before_status,
            after_status=after_status,
        )
        self._next_admin_action_number += 1
        self._admin_actions.append(action)
        return action

    def _final_submission_id_for(self, student_id: str) -> str:
        existing = self._final_submissions.get(student_id)
        if existing:
            return existing.final_submission_id
        final_submission_id = (
            f"final-submission-{self._next_final_submission_number:06d}"
        )
        self._next_final_submission_number += 1
        return final_submission_id

    def _resolve_final_submission_config(
        self,
        training_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        return resolve_training_config(
            training_config,
            requested_runtime_seconds=self.total_budget_seconds,
            code_version=self.code_version,
            data_manifest_id=self.final_data_manifest_id,
            eval_manifest_id=self.final_eval_manifest_id,
            validation_tokens_per_eval=self.final_validation_tokens_per_eval,
            worker_gpu_count=self.worker_gpu_count,
        )

    def _build_manifest(self, state: _ExperimentState) -> dict[str, Any]:
        model_config = dict(state.resolved_config["model"])
        training_config = dict(state.resolved_config["training"])
        max_runtime_seconds = state.reserved_runtime_seconds
        return {
            "manifest_version": 1,
            "run_kind": "exploratory",
            "experiment_id": state.experiment_id,
            "student_id": state.student_id,
            "manifest_uri": self.manifest_uri_builder(state.experiment_id),
            "callback_url": self.callback_url,
            "callback_base_url": self.callback_url,
            "model_config": model_config,
            "training_config": training_config,
            "resolved_config": dict(state.resolved_config),
            "code_version": self.code_version,
            "data_manifest_id": self.data_manifest_id,
            "eval_manifest_id": self.eval_manifest_id,
            "data_config": self._data_config(training_config, final=False),
            "validation_config": self._validation_config(
                eval_manifest_id=self.eval_manifest_id,
                resolved_config=state.resolved_config,
                final=False,
            ),
            "max_runtime_seconds": max_runtime_seconds,
            "runtime_config": {
                "reserved_runtime_seconds": max_runtime_seconds,
                "max_runtime_seconds": max_runtime_seconds,
            },
            "config_hash": state.config_hash,
            "created_at": self.now(),
        }

    def _persist_manifest_if_configured(
        self,
        run_id: str,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        frozen_manifest = dict(manifest)
        if self.manifest_store:
            frozen_manifest["manifest_uri"] = self.manifest_store.write_manifest(
                run_id,
                frozen_manifest,
            )
        frozen_manifest["resolved_config"] = _with_provider_manifest_uri(
            frozen_manifest.get("resolved_config", {}),
            str(frozen_manifest["manifest_uri"]),
        )
        if self.manifest_store:
            self.manifest_store.write_manifest(run_id, frozen_manifest)
        return frozen_manifest

    def _has_worker_event_dedup_key(
        self,
        event_key: tuple[str, str, str],
    ) -> bool:
        identity_key, run_id, event_id = event_key
        for event in self._worker_events:
            if event.payload.get("event_id") != event_id:
                continue
            if identity_key == "experiment_id" and event.experiment_id == run_id:
                return True
            if identity_key == "final_run_id" and event.final_run_id == run_id:
                return True
        return False

    def _record_experiment_event(
        self,
        state: _ExperimentState,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> str:
        timestamp = self.now()
        self._worker_events.append(
            WorkerEvent(
                student_id=state.student_id,
                event_type=event_type,
                timestamp=timestamp,
                payload=dict(payload),
                experiment_id=state.experiment_id,
            )
        )
        return timestamp

    def _record_final_run_event(
        self,
        final_run: FinalRun,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> str:
        timestamp = self.now()
        self._worker_events.append(
            WorkerEvent(
                student_id=final_run.student_id,
                event_type=event_type,
                timestamp=timestamp,
                payload=dict(payload),
                final_run_id=final_run.final_run_id,
            )
        )
        return timestamp

    def _mark_experiment_terminal_failure(
        self,
        state: _ExperimentState,
        *,
        used_runtime_seconds: int,
        failed_at: str | None = None,
    ) -> None:
        state.used_runtime_seconds = int(used_runtime_seconds)
        state.completed_at = ""
        state.failed_at = self.now() if failed_at is None else failed_at

    def _build_final_manifest(
        self,
        *,
        final_run_id: str,
        submission: FinalSubmission,
        max_runtime_seconds: int,
        resolved_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        model_config = dict(resolved_config["model"])
        training_config = dict(resolved_config["training"])
        return {
            "manifest_version": 1,
            "run_kind": "final",
            "final_run_id": final_run_id,
            "final_submission_id": submission.final_submission_id,
            "student_id": submission.student_id,
            "manifest_uri": self.manifest_uri_builder(final_run_id),
            "callback_url": self.callback_url,
            "callback_base_url": self.callback_url,
            "model_config": model_config,
            "training_config": training_config,
            "resolved_config": dict(resolved_config),
            "code_version": self.code_version,
            "data_manifest_id": self.final_data_manifest_id,
            "eval_manifest_id": self.final_eval_manifest_id,
            "data_config": self._data_config(training_config, final=True),
            "validation_config": self._validation_config(
                eval_manifest_id=self.final_eval_manifest_id,
                resolved_config=resolved_config,
                final=True,
            ),
            "max_runtime_seconds": max_runtime_seconds,
            "runtime_config": {
                "reserved_runtime_seconds": max_runtime_seconds,
                "max_runtime_seconds": max_runtime_seconds,
            },
            "predicted_final_loss": submission.predicted_final_loss,
            "predicted_final_loss_lower": submission.predicted_final_loss_lower,
            "predicted_final_loss_upper": submission.predicted_final_loss_upper,
            "created_at": self.now(),
        }

    def _data_config(
        self,
        training_config: Mapping[str, Any],
        *,
        final: bool,
    ) -> dict[str, Any]:
        config = {"train_tokens": training_config["train_tokens"]}
        tokenized_index_uri = (
            self.final_train_tokenized_index_uri
            if final
            else self.train_tokenized_index_uri
        )
        if tokenized_index_uri:
            config["tokenized_index_uri"] = tokenized_index_uri
        return config

    def _validation_config(
        self,
        *,
        eval_manifest_id: str,
        resolved_config: Mapping[str, Any],
        final: bool,
    ) -> dict[str, Any]:
        config = {
            "eval_manifest_id": eval_manifest_id,
            "validation_tokens_per_eval": resolved_config[
                "validation_tokens_per_eval"
            ],
            "validation_batches_per_eval": resolved_config[
                "validation_batches_per_eval"
            ],
        }
        tokenized_index_uri = (
            self.final_validation_tokenized_index_uri
            if final
            else self.validation_tokenized_index_uri
        )
        if tokenized_index_uri:
            config["tokenized_index_uri"] = tokenized_index_uri
        return config

    def _charge_and_release(self, state: _ExperimentState, seconds: int) -> None:
        budget = self._budget_for(state.student_id)
        budget.reserved_by_experiment.pop(state.experiment_id, None)
        budget.charged_seconds += int(seconds)

    def _release_reserved_runtime(self, state: _ExperimentState) -> None:
        budget = self._budget_for(state.student_id)
        budget.reserved_by_experiment.pop(state.experiment_id, None)


def _finite_float(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("loss values must be finite")
    return parsed


def _finite_float_list(value: Any) -> list[float]:
    return [_finite_float(item) for item in value]


def _completed_final_loss_from_payload(
    payload: Mapping[str, Any],
    validation_losses: list[float],
) -> float:
    if not validation_losses:
        raise ValueError("completed runs require non-empty validation_losses")
    tail_loss = validation_losses[-1]
    if "final_validation_loss" not in payload or payload.get("final_validation_loss") is None:
        return tail_loss
    final_loss = _finite_float(payload.get("final_validation_loss"))
    if not math.isclose(final_loss, tail_loss, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("final_validation_loss must equal validation_losses[-1]")
    return final_loss


def _clipped_runtime_charge(
    value: Any, *, reserved_runtime_seconds: int
) -> int:
    actual_runtime = reserved_runtime_seconds if value is None else int(value)
    return max(1, min(actual_runtime, reserved_runtime_seconds))


def _clipped_system_runtime(
    value: Any, *, reserved_runtime_seconds: int
) -> int:
    if value is None:
        return 0
    actual_runtime = int(value)
    return max(0, min(actual_runtime, reserved_runtime_seconds))


def _provider_terminal_runtime_charge(state: _ExperimentState) -> int:
    return max(1, state.reserved_runtime_seconds)


def _status_value(status: ProviderJobStatus) -> str:
    return {
        ProviderJobStatus.SUBMITTED: ExperimentStatus.SUBMITTED.value,
        ProviderJobStatus.QUEUED: ExperimentStatus.QUEUED.value,
        ProviderJobStatus.RUNNING: ExperimentStatus.RUNNING.value,
        ProviderJobStatus.SUCCEEDED: ExperimentStatus.COMPLETED.value,
        ProviderJobStatus.FAILED: ExperimentStatus.FAILED.value,
        ProviderJobStatus.CANCELLED: ExperimentStatus.CANCELLED.value,
        ProviderJobStatus.LOST: ExperimentStatus.LOST.value,
        ProviderJobStatus.UNKNOWN: ExperimentStatus.UNKNOWN.value,
    }[status]


def _normalize_worker_failure_reason(payload: Mapping[str, Any]) -> str:
    raw_failure_type = str(payload.get("failure_type", "")).strip()
    if raw_failure_type in STUDENT_FAILURE_REASONS:
        return raw_failure_type
    if raw_failure_type in SYSTEM_FAILURE_REASONS:
        return raw_failure_type
    if raw_failure_type in INTERNAL_FAILURE_REASONS:
        return raw_failure_type

    lowered_type = raw_failure_type.lower()
    detail = _staff_failure_detail_from_mapping(payload).lower()
    if raw_failure_type in {"FloatingPointError", "ArithmeticError", "OverflowError"}:
        return "numerical"
    if raw_failure_type == "TimeoutError" or "timeout" in lowered_type:
        return "timeout"
    if raw_failure_type == "MemoryError" or "out of memory" in detail or "oom" in detail:
        return "resource"
    if raw_failure_type in {"TokenizedIndexError", "TokenizedDatasetError"}:
        return "infrastructure_error"
    if raw_failure_type == "CourseTrainerConfigError":
        return "invalid_runtime_config"
    if raw_failure_type == "ValueError" and _looks_like_jax_sharding_config_error(
        detail
    ):
        return "invalid_runtime_config"
    if raw_failure_type == "WorkerResultError" and (
        "finite" in detail or "nan" in detail or "inf" in detail
    ):
        return "numerical"
    return "unknown_student_caused"


def _looks_like_jax_sharding_config_error(detail: str) -> bool:
    return (
        "sharding spec" in detail
        and "does not evenly divide the dimension size" in detail
        and ("fsdp" in detail or "namedsharding" in detail)
    )


def _staff_failure_detail_from_mapping(payload: Mapping[str, Any]) -> str:
    raw_failure_type = str(payload.get("failure_type", "")).strip()
    for key in ("message", "detail", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            detail = value.strip()
            if raw_failure_type and raw_failure_type not in {
                *STUDENT_FAILURE_REASONS,
                *SYSTEM_FAILURE_REASONS,
                *INTERNAL_FAILURE_REASONS,
            }:
                return f"{raw_failure_type}: {detail}"
            return detail
    return raw_failure_type


def _all_known_failure_reasons() -> frozenset[str]:
    return STUDENT_FAILURE_REASONS | SYSTEM_FAILURE_REASONS | INTERNAL_FAILURE_REASONS


def _assert_known_failure_reason(reason: str) -> str:
    if reason and reason not in _all_known_failure_reasons():
        raise ValueError(f"unknown failure_reason in persisted state: {reason}")
    return reason


def _assert_known_final_failure_reason(reason: str) -> str:
    if reason and reason not in _all_known_failure_reasons():
        raise ValueError(f"unknown final-run failure_reason in persisted state: {reason}")
    return reason


def _provider_record_staff_failure_detail(record: ExperimentRecord) -> str:
    status_event = next(
        (event for event in reversed(record.events) if event.type == "provider_status"),
        None,
    )
    metadata = {} if status_event is None else dict(status_event.metadata)
    return _provider_failure_detail(
        provider_name=record.provider_name,
        provider_job_id=record.provider_job_id,
        provider_status=record.provider_status.value,
        raw_status=str(metadata.get("raw_status", "")),
        message=str(metadata.get("message", "")),
    )


def _provider_record_terminal_timestamp(record: ExperimentRecord) -> str:
    event = next(
        (
            event
            for event in reversed(record.events)
            if event.type in {"provider_status", "provider_cancelled"}
        ),
        None,
    )
    if event is None and record.events:
        event = record.events[-1]
    return "" if event is None else event.timestamp


def _provider_snapshot_staff_failure_detail(snapshot: ProviderStatusSnapshot) -> str:
    return _provider_failure_detail(
        provider_name=snapshot.provider_name,
        provider_job_id=snapshot.provider_job_id,
        provider_status=snapshot.status.value,
        raw_status=snapshot.raw_status,
        message=snapshot.message,
    )


def _provider_failure_detail(
    *,
    provider_name: str,
    provider_job_id: str,
    provider_status: str,
    raw_status: str,
    message: str,
) -> str:
    parts = [
        f"provider_name={provider_name}",
        f"provider_job_id={provider_job_id}",
        f"provider_status={provider_status}",
    ]
    if raw_status:
        parts.append(f"raw_status={raw_status}")
    if message:
        parts.append(f"message={message}")
    return "; ".join(parts)


def _result_from_state(state: _ExperimentState) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=state.experiment_id,
        status=state.status,
        validation_losses=list(state.validation_losses),
        final_validation_loss=state.final_validation_loss,
        failure_reason=state.failure_reason,
        used_runtime_seconds=state.used_runtime_seconds,
        completed_at=state.completed_at,
        failed_at=state.failed_at,
        resource_warnings=_resource_warnings_from_resolved(state.resolved_config),
    )


def _resource_warnings_from_resolved(resolved_config: Mapping[str, Any]) -> list[str]:
    return [
        str(item)
        for item in resolved_config.get("resource_warnings", [])
    ]


def _with_provider_manifest_uri(
    resolved_config: Mapping[str, Any],
    manifest_uri: str,
) -> dict[str, Any]:
    resolved = dict(resolved_config)
    resolved["provider_manifest_uri"] = manifest_uri
    return resolved


def _optional_private_uri(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _experiment_admin_payload(state: _ExperimentState) -> dict[str, Any]:
    return {
        "experiment_id": state.experiment_id,
        "student_id": state.student_id,
        "status": state.status,
        "validation_losses": list(state.validation_losses),
        "final_validation_loss": state.final_validation_loss,
        "failure_reason": state.failure_reason,
        "staff_failure_detail": state.staff_failure_detail,
        "used_runtime_seconds": state.used_runtime_seconds,
        "completed_at": state.completed_at,
        "failed_at": state.failed_at,
        "reserved_runtime_seconds": state.reserved_runtime_seconds,
        "config_hash": state.config_hash,
        "config": dict(state.config),
        "resolved_config": dict(state.resolved_config),
    }


def _is_active_status(status: str) -> bool:
    return status in {
        ExperimentStatus.SUBMITTED.value,
        ExperimentStatus.QUEUED.value,
        ExperimentStatus.RUNNING.value,
    }


def _require_active_for_terminal_worker_event(status: str, run_id: str) -> None:
    _require_active_for_worker_event(status, run_id)


def _require_active_for_worker_event(status: str, run_id: str) -> None:
    if not _is_active_status(status):
        raise ValueError(f"run already terminal: {run_id} status={status}")


def _active_counts_by_student(
    *,
    exploratory: list[Mapping[str, Any]],
    final: list[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in [*exploratory, *final]:
        student_id = str(row["student_id"])
        counts[student_id] = counts.get(student_id, 0) + 1
    return counts


def _fair_share_order(exploratory: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_student: dict[str, list[Mapping[str, Any]]] = {}
    student_order: list[str] = []
    for row in exploratory:
        student_id = str(row["student_id"])
        if student_id not in by_student:
            by_student[student_id] = []
            student_order.append(student_id)
        by_student[student_id].append(row)

    ordered: list[dict[str, Any]] = []
    round_index = 0
    while True:
        added = False
        for student_id in student_order:
            rows = by_student[student_id]
            if round_index >= len(rows):
                continue
            row = rows[round_index]
            ordered.append(
                {
                    "experiment_id": row["experiment_id"],
                    "student_id": student_id,
                    "queue_position": row["queue_position"],
                    "fair_share_rank": len(ordered) + 1,
                }
            )
            added = True
        if not added:
            break
        round_index += 1
    return ordered


def _worker_event_payload(event: WorkerEvent) -> dict[str, Any]:
    payload = {
        "student_id": event.student_id,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "payload": dict(event.payload),
    }
    if event.experiment_id:
        payload["experiment_id"] = event.experiment_id
    if event.final_run_id:
        payload["final_run_id"] = event.final_run_id
    return payload


def _budget_adjustment_payload(adjustment: BudgetAdjustment) -> dict[str, Any]:
    return {
        "adjustment_id": adjustment.adjustment_id,
        "student_id": adjustment.student_id,
        "seconds": adjustment.seconds,
        "reason": adjustment.reason,
        "actor": adjustment.actor,
        "created_at": adjustment.created_at,
        "before_charged_seconds": adjustment.before_charged_seconds,
        "after_charged_seconds": adjustment.after_charged_seconds,
        "experiment_id": adjustment.experiment_id,
    }


def _admin_action_payload(action: AdminAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "action_type": action.action_type,
        "actor": action.actor,
        "created_at": action.created_at,
        "reason": action.reason,
        "student_id": action.student_id,
        "experiment_id": action.experiment_id,
        "final_run_id": action.final_run_id,
        "before_status": action.before_status,
        "after_status": action.after_status,
    }


def _final_submission_payload(submission: FinalSubmission) -> dict[str, Any]:
    return {
        "final_submission_id": submission.final_submission_id,
        "student_id": submission.student_id,
        "training_config": dict(submission.training_config),
        "predicted_final_loss": submission.predicted_final_loss,
        "predicted_final_loss_lower": submission.predicted_final_loss_lower,
        "predicted_final_loss_upper": submission.predicted_final_loss_upper,
        "updated_at": submission.updated_at,
        "frozen_at": submission.frozen_at,
        "frozen_by": submission.frozen_by,
        "freeze_reason": submission.freeze_reason,
    }


def _final_run_payload(
    final_run: FinalRun,
    *,
    provider_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prediction_metrics = _final_run_prediction_metrics(final_run)
    payload = {
        "final_run_id": final_run.final_run_id,
        "final_submission_id": final_run.final_submission_id,
        "student_id": final_run.student_id,
        "status": final_run.status,
        "provider_job_id": final_run.provider_job_id,
        "provider_name": final_run.provider_name,
        "manifest": dict(final_run.manifest),
        "predicted_final_loss": final_run.predicted_final_loss,
        "predicted_final_loss_lower": final_run.predicted_final_loss_lower,
        "predicted_final_loss_upper": final_run.predicted_final_loss_upper,
        "validation_losses": list(final_run.validation_losses),
        "actual_final_validation_loss": final_run.actual_final_validation_loss,
        "failure_reason": final_run.failure_reason,
        "staff_failure_detail": final_run.staff_failure_detail,
        "prediction_absolute_error": prediction_metrics["prediction_absolute_error"],
        "prediction_interval_covered": prediction_metrics["prediction_interval_covered"],
        "prediction_interval_width": prediction_metrics["prediction_interval_width"],
        "prediction_interval_miss_distance": prediction_metrics[
            "prediction_interval_miss_distance"
        ],
        "prediction_interval_score": prediction_metrics["prediction_interval_score"],
        "prediction_quality_penalty": prediction_metrics[
            "prediction_quality_penalty"
        ],
        "resolved_config": dict(final_run.resolved_config),
    }
    if provider_artifacts is not None:
        payload["provider_artifacts"] = dict(provider_artifacts)
    return payload


def _final_run_prediction_metrics(final_run: FinalRun) -> dict[str, Any]:
    return compute_prediction_metrics(
        predicted_final_loss=final_run.predicted_final_loss,
        predicted_final_loss_lower=final_run.predicted_final_loss_lower,
        predicted_final_loss_upper=final_run.predicted_final_loss_upper,
        actual_final_validation_loss=final_run.actual_final_validation_loss,
    )


def _dispatcher_record_payload(record: ExperimentRecord) -> dict[str, Any]:
    return {
        "experiment_id": record.experiment_id,
        "student_id": record.student_id,
        "status": record.status.value,
        "provider_name": record.provider_name,
        "provider_job_id": record.provider_job_id,
        "provider_status": record.provider_status.value,
        "manifest": dict(record.manifest),
        "events": [
            {
                "type": event.type,
                "timestamp": event.timestamp,
                "metadata": dict(event.metadata),
            }
            for event in record.events
        ],
    }


def _provider_artifacts_payload(artifacts: ProviderArtifacts) -> dict[str, Any]:
    return {
        "provider_name": artifacts.provider_name,
        "provider_job_id": artifacts.provider_job_id,
        "logs_uri": artifacts.logs_uri,
        "detail_uri": artifacts.detail_uri,
        "metadata": dict(artifacts.metadata),
    }


def _budget_snapshot_payload(snapshot: BudgetSnapshot) -> dict[str, Any]:
    return {
        "student_id": snapshot.student_id,
        "total_seconds": snapshot.total_seconds,
        "reserved_seconds": snapshot.reserved_seconds,
        "charged_seconds": snapshot.charged_seconds,
        "remaining_seconds": snapshot.remaining_seconds,
    }


def _snapshot_bool_with_fallback(
    payload: Mapping[str, Any], key: str, *, fallback: bool
) -> bool:
    if key in payload:
        return bool(payload[key])
    return fallback


def _experiment_state_from_payload(item: Mapping[str, Any]) -> _ExperimentState:
    return _ExperimentState(
        experiment_id=str(item["experiment_id"]),
        student_id=str(item["student_id"]),
        config_hash=str(item["config_hash"]),
        config=dict(item["config"]),
        resolved_config=dict(item["resolved_config"]),
        reserved_runtime_seconds=int(item["reserved_runtime_seconds"]),
        manifest=dict(item.get("manifest", {})),
        status=str(item.get("status", "submitted")),
        validation_losses=_finite_float_list(item.get("validation_losses", [])),
        final_validation_loss=(
            None
            if item.get("final_validation_loss") is None
            else _finite_float(item.get("final_validation_loss"))
        ),
        failure_reason=_assert_known_failure_reason(
            str(item.get("failure_reason", ""))
        ),
        staff_failure_detail=str(item.get("staff_failure_detail", "")),
        used_runtime_seconds=(
            None
            if item.get("used_runtime_seconds") is None
            else int(item["used_runtime_seconds"])
        ),
        completed_at=str(item.get("completed_at", "")),
        failed_at=str(item.get("failed_at", "")),
    )


def _budget_state_from_payload(item: Mapping[str, Any]) -> _BudgetState:
    return _BudgetState(
        total_seconds=int(item["total_seconds"]),
        reserved_by_experiment={
            str(experiment_id): int(seconds)
            for experiment_id, seconds in dict(
                item.get("reserved_by_experiment", {})
            ).items()
        },
        charged_seconds=int(item.get("charged_seconds", 0)),
    )


def _final_submission_from_payload(item: Mapping[str, Any]) -> FinalSubmission:
    return FinalSubmission(
        final_submission_id=str(item["final_submission_id"]),
        student_id=str(item["student_id"]),
        training_config=dict(item["training_config"]),
        predicted_final_loss=_finite_float(item["predicted_final_loss"]),
        predicted_final_loss_lower=_finite_float(item["predicted_final_loss_lower"]),
        predicted_final_loss_upper=_finite_float(item["predicted_final_loss_upper"]),
        updated_at=str(item["updated_at"]),
        frozen_at=str(item.get("frozen_at", "")),
        frozen_by=str(item.get("frozen_by", "")),
        freeze_reason=str(item.get("freeze_reason", "")),
    )


def _final_run_from_payload(item: Mapping[str, Any]) -> FinalRun:
    return FinalRun(
        final_run_id=str(item["final_run_id"]),
        final_submission_id=str(item["final_submission_id"]),
        student_id=str(item["student_id"]),
        status=str(item["status"]),
        provider_job_id=str(item["provider_job_id"]),
        provider_name=str(item.get("provider_name", "")),
        manifest=dict(item.get("manifest", {})),
        predicted_final_loss=_finite_float(item["predicted_final_loss"]),
        predicted_final_loss_lower=_finite_float(item["predicted_final_loss_lower"]),
        predicted_final_loss_upper=_finite_float(item["predicted_final_loss_upper"]),
        resolved_config=dict(item.get("resolved_config", {})),
        validation_losses=_finite_float_list(item.get("validation_losses", [])),
        actual_final_validation_loss=(
            None
            if item.get("actual_final_validation_loss") is None
            else _finite_float(item.get("actual_final_validation_loss"))
        ),
        failure_reason=_assert_known_final_failure_reason(
            str(item.get("failure_reason", ""))
        ),
        staff_failure_detail=str(item.get("staff_failure_detail", "")),
    )


def _budget_adjustment_from_payload(item: Mapping[str, Any]) -> BudgetAdjustment:
    return BudgetAdjustment(
        adjustment_id=str(item["adjustment_id"]),
        student_id=str(item["student_id"]),
        seconds=int(item["seconds"]),
        reason=str(item["reason"]),
        actor=str(item["actor"]),
        created_at=str(item["created_at"]),
        before_charged_seconds=int(item.get("before_charged_seconds", 0)),
        after_charged_seconds=int(
            item.get(
                "after_charged_seconds",
                int(item.get("before_charged_seconds", 0)) + int(item["seconds"]),
            )
        ),
        experiment_id=str(item.get("experiment_id", "")),
    )


def _admin_action_from_payload(item: Mapping[str, Any]) -> AdminAction:
    return AdminAction(
        action_id=str(item["action_id"]),
        action_type=str(item["action_type"]),
        actor=str(item["actor"]),
        created_at=str(item["created_at"]),
        reason=str(item["reason"]),
        student_id=str(item["student_id"]),
        experiment_id=str(item.get("experiment_id", "")),
        final_run_id=str(item.get("final_run_id", "")),
        before_status=str(item.get("before_status", "")),
        after_status=str(item.get("after_status", "")),
    )


def _worker_event_from_payload(item: Mapping[str, Any]) -> WorkerEvent:
    return WorkerEvent(
        student_id=str(item["student_id"]),
        event_type=str(item["event_type"]),
        timestamp=str(item["timestamp"]),
        payload=dict(item.get("payload", {})),
        experiment_id=str(item.get("experiment_id", "")),
        final_run_id=str(item.get("final_run_id", "")),
    )


def _dispatcher_record_from_payload(item: Mapping[str, Any]) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=str(item["experiment_id"]),
        student_id=str(item["student_id"]),
        status=ExperimentStatus(str(item.get("status", "submitted"))),
        provider_name=str(item.get("provider_name", "")),
        provider_job_id=str(item.get("provider_job_id", "")),
        provider_status=ProviderJobStatus(str(item.get("provider_status", "unknown"))),
        manifest=dict(item.get("manifest", {})),
        events=[
            ExperimentEvent(
                type=str(event["type"]),
                timestamp=str(event["timestamp"]),
                metadata=dict(event.get("metadata", {})),
            )
            for event in item.get("events", [])
        ],
    )


def _replace_final_run(final_run: FinalRun, **updates: Any) -> FinalRun:
    values = {
        "final_run_id": final_run.final_run_id,
        "final_submission_id": final_run.final_submission_id,
        "student_id": final_run.student_id,
        "status": final_run.status,
        "provider_job_id": final_run.provider_job_id,
        "provider_name": final_run.provider_name,
        "manifest": final_run.manifest,
        "predicted_final_loss": final_run.predicted_final_loss,
        "predicted_final_loss_lower": final_run.predicted_final_loss_lower,
        "predicted_final_loss_upper": final_run.predicted_final_loss_upper,
        "resolved_config": final_run.resolved_config,
        "validation_losses": final_run.validation_losses,
        "actual_final_validation_loss": final_run.actual_final_validation_loss,
        "failure_reason": final_run.failure_reason,
        "staff_failure_detail": final_run.staff_failure_detail,
    }
    values.update(updates)
    return FinalRun(**values)


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _worker_event_dedup_key(
    payload: Mapping[str, Any],
) -> tuple[str, str, str] | None:
    event_id = payload.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        return None
    final_run_id = payload.get("final_run_id")
    if isinstance(final_run_id, str) and final_run_id.strip():
        return ("final_run_id", final_run_id.strip(), event_id.strip())
    experiment_id = payload.get("experiment_id")
    if isinstance(experiment_id, str) and experiment_id.strip():
        return ("experiment_id", experiment_id.strip(), event_id.strip())
    return None


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"payload missing required field: {key}")
    return value
