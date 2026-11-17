"""Versioned OCR wire types, portable to the sidecar without application imports."""

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONTRACT_VERSION: Literal[2] = 2

__all__ = [
    "CONTRACT_VERSION",
    "BoundingBox",
    "CoordinateSpace",
    "OcrBlock",
    "OcrCapabilities",
    "OcrPageMetadata",
    "OcrResponse",
    "Point",
    "Sha256",
    "SourceRef",
    "WorkerReadiness",
    "WorkingImage",
    "WorkingImageMetadata",
]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CoordinateSpace = Literal["original", "working", "unknown"]
Point = Annotated[
    list[Annotated[float, Field(allow_inf_nan=False)]], Field(min_length=2, max_length=2)
]
BoundingBox = Annotated[list[Point], Field(min_length=3)]


class SourceRef(BaseModel):
    part_id: UUID
    photo_index: int = Field(ge=0)
    source_sha256: Sha256
    view_id: UUID | None = None
    coordinate_space: CoordinateSpace = "unknown"
    block_id: str | None = None
    bbox: BoundingBox | None = None
    text_span: tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]] | None = None

    @field_validator("text_span")
    @classmethod
    def ordered_span(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is not None and value[1] < value[0]:
            raise ValueError("text span end must not precede start")
        return value


class OcrBlock(BaseModel):
    id: str
    label: str
    text: str
    bbox: BoundingBox


class WorkingImageMetadata(BaseModel):
    """Persisted view identity and geometry; safe for formatter and public metadata."""

    view_id: UUID
    media_type: Literal["image/png"] = "image/png"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: Sha256
    transforms: list[dict[str, Any]] = Field(default_factory=list)
    inverse_mapping: None = None


class WorkingImage(WorkingImageMetadata):
    png_base64: str = Field(repr=False)


class OcrPageMetadata(BaseModel):
    """Validated OCR page after the working PNG has been persisted separately."""

    contract_version: Literal[2] = CONTRACT_VERSION
    request_id: UUID
    photo_index: int = Field(ge=0)
    source_sha256: Sha256
    text: str
    blocks: list[OcrBlock]
    working_image: WorkingImageMetadata
    profile_id: str
    runtime_fingerprint: str
    telemetry: dict[str, Any] = Field(default_factory=dict)


class OcrResponse(OcrPageMetadata):
    model_config = ConfigDict(extra="forbid")
    working_image: WorkingImage


class OcrCapabilities(BaseModel):
    enabled: bool
    configured: bool
    reachable: bool
    ready: bool
    can_load: bool = False
    profile_id: str | None = None
    runtime_fingerprint: str | None = None
    reason_code: str | None = None
    ocr_max_new_tokens: int | None = Field(default=None, gt=0)


class WorkerReadiness(BaseModel):
    runtime_fingerprint: str | None = None
    ready: bool
    generation: int = Field(ge=1)
    profile_id: str | None = None
    reason_code: str | None = None
