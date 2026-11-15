import pytest
from pydantic import ValidationError

from vision_bill.config import LLMSettings, OcrSettings, WorkerSettings


def test_required_ocr_requires_enabled():
    with pytest.raises(ValidationError):
        OcrSettings(enabled=False, required=True)
    assert OcrSettings(enabled=True, required=True).required


def test_no_implicit_fallback():
    assert OcrSettings().enabled is False
    assert OcrSettings().allow_vision_fallback is False


def test_worker_attempt_budget_fits_claim_lease():
    worker = WorkerSettings()
    assert worker.attempt_timeout_seconds == 600
    assert worker.max_attempts == 3
    assert worker.retry_backoff_seconds == (30, 60)
    assert worker.retry_backoff_cap_seconds == 300
    assert worker.claim_lease_seconds == 900
    with pytest.raises(ValidationError):
        WorkerSettings(attempt_timeout_seconds=601)
    with pytest.raises(ValidationError):
        WorkerSettings(claim_lease_seconds=600)
    for change in (
        {"max_attempts": 0},
        {"max_attempts": 4},
        {"retry_backoff_seconds": [0, 60]},
        {"retry_backoff_cap_seconds": 0},
    ):
        with pytest.raises(ValidationError):
            WorkerSettings(**change)


def test_formatter_limit_is_separate_from_context_and_ocr_limits():
    llm = LLMSettings(host="local", api_key="none", model_name="model", temperature=0)
    assert llm.max_output_tokens == 8192
    with pytest.raises(ValidationError):
        LLMSettings(
            host="local", api_key="none", model_name="model", temperature=0, max_output_tokens=0
        )
