from scaling_backend.providers.contracts import ProviderJobStatus
from scaling_backend.providers.qz_distributed.statuses import normalize_qz_status


def test_normalize_qz_status_handles_common_platform_statuses():
    assert normalize_qz_status("PENDING") is ProviderJobStatus.QUEUED
    assert normalize_qz_status("QUEUING") is ProviderJobStatus.QUEUED
    assert normalize_qz_status("RUNNING") is ProviderJobStatus.RUNNING
    assert normalize_qz_status("SUCCEEDED") is ProviderJobStatus.SUCCEEDED
    assert normalize_qz_status("COMPLETED") is ProviderJobStatus.SUCCEEDED
    assert normalize_qz_status("FAILED") is ProviderJobStatus.FAILED
    assert normalize_qz_status("STOPPED") is ProviderJobStatus.CANCELLED
    assert normalize_qz_status("CANCELLED") is ProviderJobStatus.CANCELLED


def test_normalize_qz_status_is_case_and_whitespace_tolerant():
    assert normalize_qz_status(" running ") is ProviderJobStatus.RUNNING
    assert normalize_qz_status("") is ProviderJobStatus.UNKNOWN
    assert normalize_qz_status(None) is ProviderJobStatus.UNKNOWN
    assert normalize_qz_status("some-new-platform-state") is ProviderJobStatus.UNKNOWN

