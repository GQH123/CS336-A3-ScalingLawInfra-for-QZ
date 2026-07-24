from __future__ import annotations

from scaling_backend.providers.contracts import ProviderJobStatus


_STATUS_MAP = {
    "CREATED": ProviderJobStatus.SUBMITTED,
    "SUBMITTED": ProviderJobStatus.SUBMITTED,
    "PENDING": ProviderJobStatus.QUEUED,
    "QUEUED": ProviderJobStatus.QUEUED,
    "QUEUING": ProviderJobStatus.QUEUED,
    "WAITING": ProviderJobStatus.QUEUED,
    "RUNNING": ProviderJobStatus.RUNNING,
    "STARTED": ProviderJobStatus.RUNNING,
    "SUCCEEDED": ProviderJobStatus.SUCCEEDED,
    "SUCCESS": ProviderJobStatus.SUCCEEDED,
    "COMPLETED": ProviderJobStatus.SUCCEEDED,
    "FINISHED": ProviderJobStatus.SUCCEEDED,
    "FAILED": ProviderJobStatus.FAILED,
    "ERROR": ProviderJobStatus.FAILED,
    "STOPPED": ProviderJobStatus.CANCELLED,
    "CANCELED": ProviderJobStatus.CANCELLED,
    "CANCELLED": ProviderJobStatus.CANCELLED,
}


def normalize_qz_status(raw_status: object) -> ProviderJobStatus:
    if raw_status is None:
        return ProviderJobStatus.UNKNOWN
    normalized = str(raw_status).strip().upper()
    if not normalized:
        return ProviderJobStatus.UNKNOWN
    return _STATUS_MAP.get(normalized, ProviderJobStatus.UNKNOWN)

