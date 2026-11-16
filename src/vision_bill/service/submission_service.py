"""One fenced attempt, ordered checkpoint recovery, and owner-scoped draft operations."""

import asyncio
import base64
import hashlib
import logging
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import UUID

import httpx

from ..config import OcrSettings, Settings
from ..helper.deadline import check_deadline, run_sync_before
from ..model.db.image import ImageRow
from ..model.db.receipt import ReceiptRow
from ..model.experiment import ExperimentOptions
from ..model.extraction import DraftView, ExtractionDraft, InferenceEnvelope
from ..model.ocr import OcrPageMetadata, OcrResponse, SourceRef, WorkingImage, WorkingImageMetadata
from ..model.review import OriginalAvailability
from ..model.workflow import (
    AccessScope,
    AnalysisClaim,
    AttemptOutcome,
    CommitResult,
    DraftFinalize,
    DraftPatch,
    OcrCheckpoint,
    PersistedSourcePart,
    PersistedWorkingView,
    SourcePartBinding,
)
from ..provider.llm.errors import AnalysisTimeoutError, UnreadableReceiptError
from ..provider.ocr.base import compact_ocr_text
from .extraction_pipeline import OcrRuntimeChangedError, OriginalSourceChangedError
from .image_service import ImageService
from .receipt_review import admission_review
from .receipt_service import ReceiptService

logger = logging.getLogger(__name__)


class StaleClaimError(Exception):
    """A worker lost its fence; it must not publish any outcome."""


class SourceIntegrityError(Exception):
    """Original bytes/path/order no longer match the durable submission."""


class SubmissionService:
    def __init__(self, settings: Settings, receipts: ReceiptService, images: ImageService):
        self.settings = settings
        self.worker = settings.worker
        self.receipts = receipts
        self.db = receipts.workflow_db
        self.images = images

    def legacy_snapshot(self) -> dict[str, Any]:
        return {
            "ocr": self.settings.ocr.model_dump(mode="json"),
            "temperature": self.settings.llm.temperature,
            "max_output_tokens": self.settings.llm.max_output_tokens,
        }

    async def ready_for(self, image: ImageRow) -> bool:
        """Readiness precedes claim, and therefore never spends the attempt budget."""
        snapshot = image.effective_options or self.legacy_snapshot()
        try:
            ocr = OcrSettings.model_validate(snapshot["ocr"])
        except (ValueError, KeyError):
            # Claim malformed saved options to record a terminal safe diagnostic.
            return True
        try:
            if not await self.receipts.check_connection():
                return False
            capability = "vision"
            selected = image.model_id or self.settings.llm.model_name
            if ocr.enabled:
                capabilities = await self.receipts.ocr_capabilities(ocr)
                if not capabilities.ready and (ocr.required or not ocr.allow_vision_fallback):
                    return False
                if capabilities.ready:
                    capability = "text"
                else:
                    selected = ocr.vision_fallback_model_id or selected
            models = await self.receipts.get_available_models(capability=capability)
            return any(model.id == selected for model in models) or (
                not image.effective_options and bool(models)
            )
        except (httpx.HTTPError, OSError, ValueError, KeyError):
            return False

    async def process_claim(self, claim: AnalysisClaim) -> CommitResult:
        deadline = monotonic() + self.worker.attempt_timeout_seconds
        owned: list[PersistedWorkingView] = []
        try:
            async with asyncio.timeout(max(0, deadline - monotonic())):
                return await self._process_claim(claim, deadline, owned)
        except TimeoutError:
            # A hung DB cannot extend the complete attempt. Lease recovery owns
            # any still-uncommitted outcome; stale fencing prevents a late write.
            return CommitResult(stale=True)
        finally:
            if owned and monotonic() < deadline:
                # The DB locks the submission before checking references, including
                # transactions whose commit acknowledgement was lost. Uncertain
                # outcomes retain artifacts; never delete a referenced file.
                try:
                    async with asyncio.timeout(min(5.0, max(0, deadline - monotonic()))):
                        unused = await self.db.unreferenced_views(claim.image.id, owned)
                        for view in unused:
                            await asyncio.to_thread(
                                self.images.remove_unreferenced_working_image,
                                claim.image.id,
                                view.artifact_path,
                                view.metadata,
                            )
                except Exception:  # noqa: BLE001 - a safe orphan is preferable to lost evidence
                    logger.warning("Working artifact cleanup deferred")

    async def _process_claim(
        self, claim: AnalysisClaim, deadline: float, owned: list[PersistedWorkingView]
    ) -> CommitResult:
        try:
            # Reserve a small part of the same absolute budget for durable error
            # bookkeeping. Providers, callbacks and threads retain the original deadline.
            reserve = min(5.0, self.worker.attempt_timeout_seconds * 0.05)
            async with asyncio.timeout(max(0, deadline - monotonic() - reserve)):
                outcome = await self._attempt(claim, deadline, owned)
                check_deadline(deadline)
                return await self.db.finish_attempt(claim, outcome)
        except StaleClaimError:
            return CommitResult(stale=True)
        except (SourceIntegrityError, OriginalSourceChangedError):
            outcome = AttemptOutcome(kind="terminal", result_code="source_integrity_failed")
        except (TimeoutError, AnalysisTimeoutError):
            outcome = AttemptOutcome(kind="retry", result_code="timed_out")
        except UnreadableReceiptError:
            outcome = AttemptOutcome(kind="terminal", result_code="unreadable")
        except OcrRuntimeChangedError:
            outcome = AttemptOutcome(kind="retry", result_code="ocr_runtime_changed")
        except (ValueError, KeyError):
            outcome = AttemptOutcome(kind="terminal", result_code="analysis_contract_failed")
        except Exception:  # noqa: BLE001 - no receipt content in error responses or logs
            outcome = AttemptOutcome(kind="retry", result_code="analysis_unavailable")
        check_deadline(deadline)
        async with asyncio.timeout(max(0, deadline - monotonic())):
            return await self.db.finish_attempt(claim, outcome)

    @staticmethod
    def _bindings(image: ImageRow) -> list[SourcePartBinding]:
        try:
            return [
                SourcePartBinding(
                    photo_index=i,
                    image_path=p.image_path,
                    source_sha256=hashlib.sha256(Path(p.image_path).read_bytes()).hexdigest(),
                )
                for i, p in enumerate(image.parts())
            ]
        except OSError as exc:
            raise SourceIntegrityError from exc

    async def _attempt(
        self, claim: AnalysisClaim, deadline: float, owned: list[PersistedWorkingView]
    ) -> AttemptOutcome:
        image = claim.image
        if not image.effective_options:
            # Resolve legacy model selection from affirmative provider discovery.
            # This is repeated under the attempt budget and saved exactly once.
            legacy_ocr = OcrSettings.model_validate(self.legacy_snapshot()["ocr"])
            models = await self.receipts.get_available_models(
                capability="text" if legacy_ocr.enabled else "vision"
            )
            preferred = image.model_id or self.settings.llm.model_name
            selected = next(
                (m.id for m in models if m.id == preferred), models[0].id if models else None
            )
            if selected is None:
                raise RuntimeError("model_unavailable")
            saved = await self.db.ensure_execution_snapshot(
                claim, model_id=selected, effective_options=self.legacy_snapshot()
            )
            if saved is None:
                raise StaleClaimError
            image = saved
        snapshot = image.effective_options
        if set(snapshot) != {"ocr", "temperature", "max_output_tokens"} or not image.model_id:
            raise ValueError("execution_snapshot_invalid")
        ocr = OcrSettings.model_validate(snapshot["ocr"])
        options = ExperimentOptions(
            temperature=snapshot["temperature"], allow_ocr_fallback=ocr.allow_vision_fallback
        )
        limit = snapshot["max_output_tokens"]
        if type(limit) is not int or limit <= 0:
            raise ValueError("execution_snapshot_invalid")
        bindings = await run_sync_before(deadline, self._bindings, image)
        try:
            parts = await self.db.bind_source_parts(claim, bindings)
        except ValueError as exc:
            raise SourceIntegrityError from exc
        if parts is None:
            raise StaleClaimError
        hashes = [p.source_sha256 for p in parts]
        refs = [
            SourceRef(
                part_id=p.part_id,
                photo_index=p.photo_index,
                source_sha256=p.source_sha256,
                coordinate_space="original",
            )
            for p in parts
        ]
        pages, manifests = await self._resume(claim, parts, ocr, deadline)

        async def save_page(page: OcrResponse) -> None:
            manifest = await self._save_view(
                image.id, page.photo_index, page.source_sha256, page.working_image, parts, deadline
            )
            owned.append(manifest)
            pages.append(page)
            manifests.append(manifest)
            checkpoint = OcrCheckpoint(
                ordered_source_sha256=hashes,
                profile_id=page.profile_id,
                runtime_fingerprint=page.runtime_fingerprint,
                pages=[p.model_dump() for p in pages],
                persisted_views=list(manifests),
            )
            if not await self.db.save_ocr_checkpoint(claim, checkpoint):
                raise StaleClaimError

        tags = await self.receipts.list_tags()
        for retry in range(2):
            try:
                envelope = await self.receipts.extract_receipt(
                    image.model_id,
                    [Path(p.image_path) for p in parts],
                    options=options,
                    deadline=deadline,
                    tags=tags,
                    checkpoint=save_page,
                    resume_pages=list(pages),
                    source_refs=refs,
                    ocr_settings=ocr,
                    max_output_tokens=limit,
                )
                break
            except OcrRuntimeChangedError:
                if not await self.db.clear_ocr_checkpoint(claim):
                    raise StaleClaimError from None
                pages.clear()
                manifests.clear()
                if retry:
                    raise
        else:
            raise OcrRuntimeChangedError("ocr_runtime_changed")
        if envelope.provenance.ordered_source_sha256 != hashes:
            raise SourceIntegrityError
        for view in envelope.input_views:
            manifest = await self._save_view(
                image.id, view.photo_index, view.source_sha256, view.working_image, parts, deadline
            )
            owned.append(manifest)
            manifests.append(manifest)
        # Check originals again before accepting any receipt; preserve bytes and paths.
        if await run_sync_before(deadline, self._bindings, image) != bindings:
            raise SourceIntegrityError
        return self._outcome(envelope, manifests)

    async def _save_view(
        self,
        image_id: UUID,
        index: int,
        digest: str,
        working: WorkingImage,
        parts: list[PersistedSourcePart],
        deadline: float,
    ) -> PersistedWorkingView:
        if index >= len(parts) or index < 0 or parts[index].source_sha256 != digest:
            raise SourceIntegrityError
        path = await self.images.save_working_image(image_id, working, deadline=deadline)
        return PersistedWorkingView(
            part_id=parts[index].part_id,
            photo_index=index,
            source_sha256=digest,
            artifact_path=str(path),
            metadata=WorkingImageMetadata.model_validate(working.model_dump()),
        )

    async def _resume(
        self,
        claim: AnalysisClaim,
        parts: list[PersistedSourcePart],
        ocr: OcrSettings,
        deadline: float,
    ) -> tuple[list[OcrResponse], list[PersistedWorkingView]]:
        checkpoint = await self.db.get_ocr_checkpoint(claim)
        if checkpoint is None:
            return [], []
        if checkpoint.ordered_source_sha256 != [p.source_sha256 for p in parts]:
            raise SourceIntegrityError
        try:
            capabilities = await self.receipts.ocr_capabilities(ocr)
            if (
                not ocr.enabled
                or not capabilities.ready
                or not capabilities.runtime_fingerprint
                or capabilities.profile_id != checkpoint.profile_id
                or capabilities.runtime_fingerprint != checkpoint.runtime_fingerprint
            ):
                raise ValueError("checkpoint_runtime_changed")
            if not 0 < len(checkpoint.pages) <= len(parts) or len(
                checkpoint.persisted_views
            ) != len(checkpoint.pages):
                raise ValueError("checkpoint_prefix_invalid")
            result: list[OcrResponse] = []
            seen: set[UUID] = set()
            for index, raw in enumerate(checkpoint.pages):
                page = OcrPageMetadata.model_validate(raw)
                compact_ocr_text({"text": page.text})
                view = checkpoint.persisted_views[index]
                if (
                    page.photo_index != index
                    or page.source_sha256 != parts[index].source_sha256
                    or page.profile_id != checkpoint.profile_id
                    or page.runtime_fingerprint != checkpoint.runtime_fingerprint
                    or view.metadata != page.working_image
                    or view.photo_index != index
                    or view.part_id != parts[index].part_id
                    or view.source_sha256 != parts[index].source_sha256
                    or view.metadata.view_id in seen
                ):
                    raise ValueError("checkpoint_identity_invalid")
                seen.add(view.metadata.view_id)
                png = await run_sync_before(
                    deadline,
                    self.images.read_working_image,
                    claim.image.id,
                    view.artifact_path,
                    view.metadata,
                )
                payload = page.model_dump()
                payload["working_image"]["png_base64"] = base64.b64encode(png).decode()
                result.append(OcrResponse.model_validate(payload))
            return result, list(checkpoint.persisted_views)
        except TimeoutError:
            raise
        except (ValueError, OSError):
            if not await self.db.clear_ocr_checkpoint(claim):
                raise StaleClaimError from None
            return [], []

    @staticmethod
    def _outcome(envelope: InferenceEnvelope, views: list[PersistedWorkingView]) -> AttemptOutcome:
        document = envelope.draft
        if document is None and envelope.receipt is not None:
            document = ExtractionDraft.model_validate(envelope.receipt.model_dump())
        if document is None:
            code = envelope.findings[-1].code if envelope.findings else "analysis_incomplete"
            if code not in {
                "ocr_unavailable",
                "ocr_page_failed",
                "ocr_disabled",
                "formatter_model_unavailable",
                "vision_model_unavailable",
                "vision_fallback_unavailable",
            }:
                code = "analysis_incomplete"
            return AttemptOutcome(
                kind="retry", result_code=code, envelope=envelope, persisted_views=views
            )
        review = admission_review(document, originals_valid=True, findings=envelope.findings)
        if envelope.receipt is not None and review.can_finalize:
            return AttemptOutcome(
                kind="complete", envelope=envelope, review=review, persisted_views=views
            )
        envelope.draft = document
        return AttemptOutcome(
            kind="partial",
            envelope=envelope,
            review=review,
            result_code="needs_review",
            persisted_views=views,
        )

    async def get_draft(self, image_id: UUID, *, scope: AccessScope) -> DraftView | None:
        return await self.db.get_draft(image_id, scope=scope)

    async def patch_draft(
        self, image_id: UUID, patch: DraftPatch, *, scope: AccessScope
    ) -> DraftView | None:
        return await self.db.patch_draft(image_id, patch, scope=scope)

    async def finalize_draft(
        self, image_id: UUID, request: DraftFinalize, *, scope: AccessScope
    ) -> ReceiptRow | None:
        draft = await self.db.get_draft(image_id, scope=scope)
        if draft is None:
            return None
        availability = []
        if draft.finalized_receipt_id is None:
            parts = await self.db.get_source_parts(image_id, scope=scope)

            def inspect() -> list[OriginalAvailability]:
                result = []
                for part in parts:
                    try:
                        digest = hashlib.sha256(Path(part.image_path).read_bytes()).hexdigest()
                        result.append(
                            OriginalAvailability(
                                part_id=part.part_id,
                                available=True,
                                sha256_matches=digest == part.source_sha256,
                            )
                        )
                    except OSError:
                        result.append(
                            OriginalAvailability(
                                part_id=part.part_id, available=False, sha256_matches=False
                            )
                        )
                return result

            availability = await run_sync_before(monotonic() + 30, inspect)
        return await self.db.finalize_draft(image_id, request, scope=scope, originals=availability)
