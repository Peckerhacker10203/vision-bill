"""Submission and mutation contracts. Internal claims never serve as response DTOs."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, computed_field, field_validator

from .db.image import ImagePart, ImageRow
from .db.receipt import ReceiptRow
from .extraction import InferenceEnvelope
from .ocr import OcrPageMetadata, OcrResponse, Sha256
from .review import FindingAcknowledgement, ReviewEvaluation

TERMINAL_STATES = frozenset({"analyzed", "needs_review", "failed", "timed_out", "unreadable"})


class AccessScope(BaseModel):
    user_id: UUID
    can_see_all: bool = False


class SubmissionStatus(BaseModel):
    """Explicit public projection; extra DB fields are intentionally discarded."""

    id: UUID
    original_filename: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    image_path: str | None = None
    thumbnail_path: str | None = None
    status: str = "pending"
    error: str | None = None
    receipt_id: UUID | None = None
    bypass_review: bool = False
    user_id: UUID | None = None
    created_at: datetime | None = None
    analyzed_at: datetime | None = None
    processing_at: datetime | None = None
    additional_images: list[ImagePart] = Field(default_factory=list)
    model_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=3)
    retry_at: datetime | None = None
    result_code: str | None = None
    wait_reason: str | None = None
    draft_available: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATES


class AnalysisClaim(BaseModel):
    image: ImageRow
    generation: int = Field(ge=1)
    attempt_number: int = Field(ge=1)
    claim_token: UUID = Field(repr=False)
    lease_expires_at: datetime


class RetryRequest(BaseModel):
    expected_generation: int = Field(ge=1)
    request_key: str = Field(min_length=1)


class DraftPatch(BaseModel):
    expected_revision: int = Field(ge=1)
    changes: dict[str, Any]


class DraftFinalize(BaseModel):
    expected_revision: int = Field(ge=1)
    request_key: str = Field(min_length=1)


class ReviewRequest(BaseModel):
    expected_operation_revision: int | None = Field(default=None, ge=1)
    expected_review_revision: int | None = Field(default=None, ge=1)
    acknowledgements: list[FindingAcknowledgement] = Field(default_factory=list)


class SubmissionCreate(BaseModel):
    id: UUID
    parts: list[ImagePart] = Field(min_length=1)
    source_manifest: list[dict[str, Any]]
    requested_options: dict[str, Any]
    effective_options: dict[str, Any]
    request_fingerprint: str
    model_id: str | None = None
    bypass_review: bool = False


class EnqueueResult(BaseModel):
    submission: SubmissionStatus
    created: bool


class OcrCheckpoint(BaseModel):
    ordered_source_sha256: list[Sha256]
    profile_id: str
    runtime_fingerprint: str
    pages: list[dict[str, Any]]

    @field_validator("pages", mode="before")
    @classmethod
    def metadata_only(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [
                OcrPageMetadata.model_validate(
                    page.model_dump() if isinstance(page, OcrResponse) else page
                ).model_dump(mode="json")
                for page in value
            ]
        return value


class AttemptOutcome(BaseModel):
    kind: Literal["complete", "partial", "retry", "terminal"]
    envelope: InferenceEnvelope | None = None
    result_code: str | None = None
    retry_delay_seconds: int | None = Field(default=None, ge=0)
    review: ReviewEvaluation | None = None


class CommitResult(BaseModel):
    stale: bool
    submission: SubmissionStatus | None = None
    receipt: ReceiptRow | None = None
