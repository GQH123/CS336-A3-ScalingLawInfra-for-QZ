from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from fastapi import Depends, FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from scaling_backend.providers.qz_distributed.client import QzApiError
from scaling_backend.service import (
    DuplicateExperimentError,
    ExperimentService,
    StudentAccessError,
)


_LOG = logging.getLogger(__name__)


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubmitRequest(StrictRequestModel):
    config: dict[str, Any]
    requested_runtime_seconds: int = Field(gt=0)


class FinalSubmissionRequest(StrictRequestModel):
    training_config: dict[str, Any]
    predicted_final_loss: float
    predicted_final_loss_lower: float
    predicted_final_loss_upper: float


class AdminCancelExperimentRequest(StrictRequestModel):
    reason: str
    actor: str


class AdminMarkExperimentSystemFailureRequest(StrictRequestModel):
    failure_reason: str
    staff_failure_detail: str = ""
    refund_seconds: int = Field(ge=0)
    reason: str
    actor: str


class AdminMarkFinalRunSystemFailureRequest(StrictRequestModel):
    failure_reason: str
    staff_failure_detail: str = ""
    reason: str
    actor: str


class AdminBudgetAdjustmentRequest(StrictRequestModel):
    student_id: str
    seconds: int
    reason: str
    actor: str
    experiment_id: str = ""


class AdminFreezeFinalSubmissionsRequest(StrictRequestModel):
    actor: str
    reason: str


class AdminLaunchFinalRunsRequest(StrictRequestModel):
    max_runtime_seconds: int = Field(gt=0)
    actor: str
    reason: str


def create_app(
    *,
    service: ExperimentService,
    api_keys: Mapping[str, str],
    internal_callback_token: str,
    admin_api_token: str | None = None,
    worker_event_import_dir: str = "",
) -> FastAPI:
    app = FastAPI(title="Scaling Laws Assignment Backend")

    def current_student(authorization: str | None = Header(default=None)) -> str:
        token = _bearer_token(authorization)
        student_id = api_keys.get(token)
        if not student_id:
            raise PublicApiError(
                status_code=401,
                error="invalid_api_key",
                message="Invalid or missing API key.",
            )
        return student_id

    def require_internal(authorization: str | None = Header(default=None)) -> None:
        token = _bearer_token(authorization)
        if token != internal_callback_token:
            raise PublicApiError(
                status_code=401,
                error="invalid_callback_token",
                message="Invalid or missing callback token.",
            )

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        token = _bearer_token(authorization)
        if not admin_api_token or token != admin_api_token:
            raise PublicApiError(
                status_code=401,
                error="invalid_admin_token",
                message="Invalid or missing admin token.",
            )

    @app.exception_handler(PublicApiError)
    def public_api_error_handler(_request, exc: PublicApiError):
        return _error_response(
            status_code=exc.status_code,
            error=exc.error,
            message=exc.message,
            **exc.extra,
        )

    @app.get("/budget")
    def get_budget(student_id: str = Depends(current_student)):
        budget = service.get_budget(student_id)
        return {
            "student_id": budget.student_id,
            "total_seconds": budget.total_seconds,
            "reserved_seconds": budget.reserved_seconds,
            "charged_seconds": budget.charged_seconds,
            "remaining_seconds": budget.remaining_seconds,
        }

    @app.post("/submit")
    def submit(request: SubmitRequest, student_id: str = Depends(current_student)):
        try:
            result = service.submit(
                student_id=student_id,
                config=request.config,
                requested_runtime_seconds=request.requested_runtime_seconds,
            )
        except DuplicateExperimentError as exc:
            return _error_response(
                status_code=409,
                error="duplicate_config",
                message="An identical configuration was already submitted.",
                experiment_id=exc.experiment_id,
            )
        except QzApiError as exc:
            _LOG.warning(
                "Provider submission failed during /submit: code=%s staff_message=%s",
                exc.code,
                exc.staff_message,
                exc_info=True,
            )
            raise PublicApiError(
                status_code=502,
                error="provider_submission_failed",
                message=(
                    "Training provider rejected the submission. "
                    "Course staff should inspect the control-node logs."
                ),
                provider_error_code=exc.code,
            ) from exc
        except ValueError as exc:
            raise PublicApiError(
                status_code=400,
                error="invalid_config",
                message=str(exc),
            ) from exc
        return {
            "experiment_id": result.experiment_id,
            "status": _public_experiment_status(result.status),
            "budget_reserved_seconds": result.budget_reserved_seconds,
            "resource_warnings": result.resource_warnings,
            "resolved_config": _public_resolved_config(result.resolved_config),
        }

    @app.get("/experiment/{experiment_id}")
    def get_experiment(
        experiment_id: str, student_id: str = Depends(current_student)
    ):
        try:
            result = service.get_result(student_id, experiment_id)
        except (KeyError, StudentAccessError) as exc:
            raise PublicApiError(
                status_code=404,
                error="experiment_not_found",
                message="Experiment not found.",
            ) from exc
        return _experiment_result_payload(result)

    @app.get("/experiments")
    def list_experiments(student_id: str = Depends(current_student)):
        return {
            "experiments": [
                _experiment_result_payload(result)
                for result in service.list_results(student_id)
            ]
        }

    @app.post("/final_submission")
    def set_final_submission(
        request: FinalSubmissionRequest,
        student_id: str = Depends(current_student),
    ):
        try:
            submission = service.set_final_submission(
                student_id=student_id,
                training_config=request.training_config,
                predicted_final_loss=request.predicted_final_loss,
                predicted_final_loss_lower=request.predicted_final_loss_lower,
                predicted_final_loss_upper=request.predicted_final_loss_upper,
            )
        except ValueError as exc:
            raise PublicApiError(
                status_code=400,
                error="invalid_final_submission",
                message=str(exc),
            ) from exc
        return _final_submission_payload(submission)

    @app.get("/final_submission")
    def get_final_submission(student_id: str = Depends(current_student)):
        try:
            submission = service.get_final_submission(student_id)
        except KeyError as exc:
            raise PublicApiError(
                status_code=404,
                error="final_submission_not_found",
                message="Final submission not found.",
            ) from exc
        return _final_submission_payload(submission)

    @app.post("/internal/provider-events")
    def provider_events(
        payload: dict[str, Any], _authorized: None = Depends(require_internal)
    ):
        try:
            service.record_worker_event(payload)
        except (KeyError, ValueError) as exc:
            raise PublicApiError(
                status_code=400,
                error="invalid_worker_event",
                message=str(exc),
            ) from exc
        return {"ok": True}

    @app.get("/admin/queue")
    def admin_queue(_authorized: None = Depends(require_admin)):
        return service.admin_queue_snapshot()

    @app.post("/admin/poll-active")
    def admin_poll_active(_authorized: None = Depends(require_admin)):
        return service.admin_poll_active_runs()

    @app.post("/admin/import-worker-events")
    def admin_import_worker_events(_authorized: None = Depends(require_admin)):
        if not worker_event_import_dir.strip():
            raise PublicApiError(
                status_code=400,
                error="worker_event_import_not_configured",
                message="Worker event import directory is not configured.",
            )
        try:
            report = _import_worker_events_from_jsonl_dir(
                service,
                worker_event_import_dir,
            )
        except (OSError, ValueError, KeyError) as exc:
            raise PublicApiError(
                status_code=400,
                error="invalid_worker_event_import",
                message=str(exc),
            ) from exc
        return {"ok": True, **report}

    @app.get("/admin/experiments")
    def admin_experiments(
        student_id: str = "",
        status: str = "",
        _authorized: None = Depends(require_admin),
    ):
        return {
            "experiments": service.admin_list_experiments(
                student_id=student_id,
                status=status,
            )
        }

    @app.get("/admin/students/{student_id}/budget")
    def admin_get_student_budget(
        student_id: str,
        _authorized: None = Depends(require_admin),
    ):
        budget = service.get_budget(student_id)
        return {
            "student_id": budget.student_id,
            "total_seconds": budget.total_seconds,
            "reserved_seconds": budget.reserved_seconds,
            "charged_seconds": budget.charged_seconds,
            "remaining_seconds": budget.remaining_seconds,
        }

    @app.get("/admin/experiments/{experiment_id}")
    def admin_get_experiment(
        experiment_id: str,
        _authorized: None = Depends(require_admin),
    ):
        try:
            return service.admin_get_experiment(experiment_id)
        except KeyError as exc:
            raise PublicApiError(
                status_code=404,
                error="experiment_not_found",
                message="Experiment not found.",
            ) from exc

    @app.post("/admin/experiments/{experiment_id}/cancel")
    def admin_cancel_experiment(
        experiment_id: str,
        request: AdminCancelExperimentRequest,
        _authorized: None = Depends(require_admin),
    ):
        try:
            result = service.admin_cancel_experiment(
                experiment_id,
                reason=request.reason,
                actor=request.actor,
            )
        except KeyError as exc:
            raise PublicApiError(
                status_code=404,
                error="experiment_not_found",
                message="Experiment not found.",
            ) from exc
        except ValueError as exc:
            raise PublicApiError(
                status_code=400,
                error="invalid_admin_operation",
                message=str(exc),
            ) from exc
        return _admin_experiment_result_payload(result)

    @app.post("/admin/experiments/{experiment_id}/system-failure")
    def admin_mark_experiment_system_failure(
        experiment_id: str,
        request: AdminMarkExperimentSystemFailureRequest,
        _authorized: None = Depends(require_admin),
    ):
        try:
            result = service.admin_mark_experiment_system_failed(
                experiment_id,
                failure_reason=request.failure_reason,
                staff_failure_detail=request.staff_failure_detail,
                refund_seconds=request.refund_seconds,
                reason=request.reason,
                actor=request.actor,
            )
        except KeyError as exc:
            raise PublicApiError(
                status_code=404,
                error="experiment_not_found",
                message="Experiment not found.",
            ) from exc
        except ValueError as exc:
            raise PublicApiError(
                status_code=400,
                error="invalid_admin_operation",
                message=str(exc),
            ) from exc
        return _admin_experiment_result_payload(result)

    @app.post("/admin/experiments/{experiment_id}/poll")
    def admin_poll_experiment(
        experiment_id: str,
        _authorized: None = Depends(require_admin),
    ):
        try:
            result = service.admin_poll_experiment(experiment_id)
        except KeyError as exc:
            raise PublicApiError(
                status_code=404,
                error="experiment_not_found",
                message="Experiment not found.",
            ) from exc
        except ValueError as exc:
            raise PublicApiError(
                status_code=400,
                error="invalid_admin_operation",
                message=str(exc),
            ) from exc
        return _admin_experiment_result_payload(result)

    @app.post("/admin/budget-adjustments")
    def admin_apply_budget_adjustment(
        request: AdminBudgetAdjustmentRequest,
        _authorized: None = Depends(require_admin),
    ):
        adjustment = service.admin_apply_budget_adjustment(
            student_id=request.student_id,
            seconds=request.seconds,
            reason=request.reason,
            actor=request.actor,
            experiment_id=request.experiment_id,
        )
        return _budget_adjustment_payload(adjustment)

    @app.post("/admin/final-submissions/freeze")
    def admin_freeze_final_submissions(
        request: AdminFreezeFinalSubmissionsRequest,
        _authorized: None = Depends(require_admin),
    ):
        submissions = service.freeze_final_submissions(
            actor=request.actor,
            reason=request.reason,
        )
        return {
            "final_submissions": [
                _admin_final_submission_payload(submission)
                for submission in submissions
            ]
        }

    @app.post("/admin/final-runs/launch")
    def admin_launch_final_runs(
        request: AdminLaunchFinalRunsRequest,
        _authorized: None = Depends(require_admin),
    ):
        try:
            final_runs = service.launch_final_runs(
                max_runtime_seconds=request.max_runtime_seconds,
                actor=request.actor,
                reason=request.reason,
            )
        except ValueError as exc:
            raise PublicApiError(
                status_code=400,
                error="invalid_admin_operation",
                message=str(exc),
            ) from exc
        return {
            "final_runs": [
                _final_run_payload(final_run)
                for final_run in final_runs
            ]
        }

    @app.post("/admin/final-runs/{final_run_id}/poll")
    def admin_poll_final_run(
        final_run_id: str,
        _authorized: None = Depends(require_admin),
    ):
        try:
            final_run = service.admin_poll_final_run(final_run_id)
        except KeyError as exc:
            raise PublicApiError(
                status_code=404,
                error="final_run_not_found",
                message="Final run not found.",
            ) from exc
        except ValueError as exc:
            raise PublicApiError(
                status_code=400,
                error="invalid_admin_operation",
                message=str(exc),
            ) from exc
        return _final_run_payload(final_run)

    @app.post("/admin/final-runs/{final_run_id}/system-failure")
    def admin_mark_final_run_system_failure(
        final_run_id: str,
        request: AdminMarkFinalRunSystemFailureRequest,
        _authorized: None = Depends(require_admin),
    ):
        try:
            final_run = service.admin_mark_final_run_system_failed(
                final_run_id,
                failure_reason=request.failure_reason,
                staff_failure_detail=request.staff_failure_detail,
                reason=request.reason,
                actor=request.actor,
            )
        except KeyError as exc:
            raise PublicApiError(
                status_code=404,
                error="final_run_not_found",
                message="Final run not found.",
            ) from exc
        except ValueError as exc:
            raise PublicApiError(
                status_code=400,
                error="invalid_admin_operation",
                message=str(exc),
            ) from exc
        return _final_run_payload(final_run)

    @app.post("/admin/final-runs/{final_run_id}/cancel")
    def admin_cancel_final_run(
        final_run_id: str,
        request: AdminCancelExperimentRequest,
        _authorized: None = Depends(require_admin),
    ):
        try:
            final_run = service.admin_cancel_final_run(
                final_run_id,
                reason=request.reason,
                actor=request.actor,
            )
        except KeyError as exc:
            raise PublicApiError(
                status_code=404,
                error="final_run_not_found",
                message="Final run not found.",
            ) from exc
        except ValueError as exc:
            raise PublicApiError(
                status_code=400,
                error="invalid_admin_operation",
                message=str(exc),
            ) from exc
        return _final_run_payload(final_run)

    @app.get("/admin/exports")
    def admin_exports(_authorized: None = Depends(require_admin)):
        return service.admin_export_course_records()

    return app


class PublicApiError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        error: str,
        message: str,
        **extra: Any,
    ):
        self.status_code = status_code
        self.error = error
        self.message = message
        self.extra = extra


def _error_response(
    *, status_code: int, error: str, message: str, **extra: Any
) -> JSONResponse:
    content = {"error": error, "message": message}
    content.update(extra)
    return JSONResponse(status_code=status_code, content=content)


def _experiment_result_payload(result) -> dict[str, Any]:
    public_failure_reason = _public_failure_reason(
        status=result.status,
        failure_reason=result.failure_reason,
        failure_detail=getattr(result, "failure_detail", ""),
    )
    payload = _raw_experiment_result_payload(result)
    payload["status"] = _public_experiment_status(
        result.status,
        result.failure_reason,
        getattr(result, "failure_detail", ""),
    )
    payload["failure_reason"] = public_failure_reason
    return payload


def _public_resolved_config(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    hidden_keys = {
        "data_manifest_id",
        "eval_manifest_id",
        "provider_manifest_uri",
    }
    return {
        key: value
        for key, value in dict(resolved_config).items()
        if key not in hidden_keys
    }


def _admin_experiment_result_payload(result) -> dict[str, Any]:
    return _raw_experiment_result_payload(result)


def _raw_experiment_result_payload(result) -> dict[str, Any]:
    payload = {
        "experiment_id": result.experiment_id,
        "status": result.status,
        "validation_losses": result.validation_losses,
        "final_validation_loss": result.final_validation_loss,
        "failure_reason": result.failure_reason,
        "used_runtime_seconds": result.used_runtime_seconds,
        "completed_at": result.completed_at,
        "failed_at": result.failed_at,
        "resource_warnings": result.resource_warnings,
    }
    failure_detail = getattr(result, "failure_detail", "")
    if failure_detail:
        payload["failure_detail"] = failure_detail
    return payload


def _final_submission_payload(submission) -> dict[str, Any]:
    return {
        "student_id": submission.student_id,
        "training_config": submission.training_config,
        "predicted_final_loss": submission.predicted_final_loss,
        "predicted_final_loss_lower": submission.predicted_final_loss_lower,
        "predicted_final_loss_upper": submission.predicted_final_loss_upper,
        "updated_at": submission.updated_at,
        "frozen_at": submission.frozen_at,
        "frozen_by": submission.frozen_by,
        "freeze_reason": submission.freeze_reason,
    }


def _admin_final_submission_payload(submission) -> dict[str, Any]:
    payload = _final_submission_payload(submission)
    payload.update(
        {
            "final_submission_id": submission.final_submission_id,
            "frozen_at": submission.frozen_at,
            "frozen_by": submission.frozen_by,
            "freeze_reason": submission.freeze_reason,
        }
    )
    return payload


def _final_run_payload(final_run) -> dict[str, Any]:
    return {
        "final_run_id": final_run.final_run_id,
        "final_submission_id": final_run.final_submission_id,
        "student_id": final_run.student_id,
        "status": final_run.status,
        "provider_job_id": final_run.provider_job_id,
        "predicted_final_loss": final_run.predicted_final_loss,
        "predicted_final_loss_lower": final_run.predicted_final_loss_lower,
        "predicted_final_loss_upper": final_run.predicted_final_loss_upper,
        "validation_losses": final_run.validation_losses,
        "actual_final_validation_loss": final_run.actual_final_validation_loss,
        "failure_reason": final_run.failure_reason,
        "staff_failure_detail": final_run.staff_failure_detail,
        "resolved_config": final_run.resolved_config,
    }


def _budget_adjustment_payload(adjustment) -> dict[str, Any]:
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


def _public_experiment_status(
    status: str,
    failure_reason: str = "",
    failure_detail: str = "",
) -> str:
    if status == "submitted":
        return "queued"
    if failure_detail and failure_reason == "provider_failed_without_worker_callback":
        return "failed"
    if status in {"lost", "unknown"}:
        return "system_failed"
    if failure_reason in _INTERNAL_PROVIDER_FAILURE_REASONS:
        return "system_failed"
    return status


def _public_failure_reason(
    *,
    status: str,
    failure_reason: str,
    failure_detail: str = "",
) -> str:
    if failure_detail and failure_reason == "provider_failed_without_worker_callback":
        return "unknown_student_caused"
    if failure_reason in _INTERNAL_PROVIDER_FAILURE_REASONS:
        return "unknown_system"
    if status == "lost":
        if failure_reason == "missing_worker_completed_callback":
            return "worker_lost"
        return "unknown_system"
    if status == "unknown":
        return "unknown_system"
    return failure_reason


_INTERNAL_PROVIDER_FAILURE_REASONS = frozenset(
    {
        "provider_cancelled",
        "provider_failed_without_worker_callback",
        "provider_lost",
        "provider_status_unknown",
    }
)


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return ""
    return authorization[len(prefix) :].strip()


def _import_worker_events_from_jsonl_dir(
    service: ExperimentService,
    directory: str,
) -> dict[str, int]:
    root = Path(directory)
    if not root.exists():
        raise ValueError(f"worker event import directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"worker event import path is not a directory: {root}")

    files = 0
    lines = 0
    imported = 0
    duplicates = 0
    for path in sorted(root.glob("*.jsonl")):
        if not path.is_file():
            continue
        files += 1
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                lines += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: worker event is not valid JSON"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"{path}:{line_number}: worker event must be a JSON object"
                    )
                if service.record_worker_event(payload):
                    imported += 1
                else:
                    duplicates += 1
    return {
        "files": files,
        "lines": lines,
        "imported": imported,
        "duplicates": duplicates,
    }
