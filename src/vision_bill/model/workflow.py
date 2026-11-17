"""Submission and mutation contracts. Internal claims never serve as response DTOs."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from .db.image import ImagePart, ImageRow
from .db.receipt import ReceiptRow
from .extraction import InferenceEnvelope
from .ocr import OcrPageMetadata, OcrResponse, Sha256, WorkingImageMetadata
from .review import FindingAcknowledgement, ReviewEvaluation, SourceView

TERMINAL_STATES = frozenset({"analyzed", "needs_review", "failed", "timed_out", "unreadable"})


def assert_metadata_only(value: Any) -> None:
    """Reject transport image payloads from arbitrary persisted metadata."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError("image bytes belong in saved artifacts")  # noqa: TRY004
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"png_base64", "image_base64", "image_bytes"}:
                raise ValueError("raw image payload belongs in saved artifacts")
            assert_metadata_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_metadata_only(item)


class SourcePartBinding(BaseModel):
    """Original-file hash attested by the application, in exact capture order."""

    photo_index: int = Field(ge=0)
    source_sha256: Sha256
    image_path: str = Field(min_length=1)


class PersistedSourcePart(SourcePartBinding):
    part_id: UUID


class PersistedWorkingView(BaseModel):
    """Caller attests that the exact validated PNG was saved before DB commit.

    DB owns identity/metadata checks, never file or codec I/O. Artifact creation
    and cleanup of unreferenced files belong to the caller/FileProvider.
    """

    model_config = ConfigDict(extra="forbid")
    part_id: UUID
    photo_index: int = Field(ge=0)
    source_sha256: Sha256
    artifact_path: str = Field(min_length=1)
    metadata: WorkingImageMetadata

    @model_validator(mode="before")
    @classmethod
    def no_payload(cls, value: Any) -> Any:
        if isinstance(value, dict):
            assert_metadata_only(value)
        return value


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
    latest_extraction_id: UUID | None = None
    source_views: list[SourceView] = Field(default_factory=list)

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
    policy: Literal["preserve", "current"] = "preserve"


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
    """Upload identity and the first resolved execution snapshot.

    The caller fingerprints ordered ORIGINAL hashes plus analysis-relevant
    explicitly supplied options/model selection. requested_options contains
    those explicit values (including an explicit bypass flag); effective_options,
    model_id and bypass_review preserve the first resolution of server defaults.
    Generated paths, thumbnails and database IDs are never request identity.
    """

    id: UUID
    parts: list[ImagePart] = Field(min_length=1)
    source_manifest: list[dict[str, Any]]
    requested_options: dict[str, Any]
    effective_options: dict[str, Any]
    request_fingerprint: str
    model_id: str | None = None
    bypass_review: bool = False

    @model_validator(mode="after")
    def metadata_only(self) -> "SubmissionCreate":
        assert_metadata_only(self.model_dump())
        return self


class EnqueueResult(BaseModel):
    submission: SubmissionStatus
    created: bool


class OcrCheckpoint(BaseModel):
    ordered_source_sha256: list[Sha256]
    profile_id: str
    runtime_fingerprint: str
    pages: list[dict[str, Any]]
    persisted_views: list[PersistedWorkingView] = Field(default_factory=list)

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
    persisted_views: list[PersistedWorkingView] = Field(default_factory=list)

    @model_validator(mode="after")
    def required_result(self) -> "AttemptOutcome":
        if self.kind == "complete" and (
            self.envelope is None
            or self.envelope.receipt is None
            or self.review is None
            or not self.review.can_finalize
        ):
            raise ValueError("complete requires a receipt and finalizable review")
        if self.kind == "partial" and (self.envelope is None or self.envelope.draft is None):
            raise ValueError("partial requires an extraction draft")
        return self


class CommitResult(BaseModel):
    stale: bool
    submission: SubmissionStatus | None = None
    receipt: ReceiptRow | None = None


class DeletedResource(BaseModel):
    """Owned file identities returned only after the authorized DB deletion commits."""
    image_id: UUID | None = None
    receipt_id: UUID | None = None
    parts: list[ImagePart] = Field(default_factory=list)
    working_views: list[PersistedWorkingView] = Field(default_factory=list)
    thumbnail_sha256: dict[str, str | None] = Field(default_factory=dict)
