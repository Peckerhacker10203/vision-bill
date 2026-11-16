"""Durable submissions and fenced, single-connection extraction transactions.

Files are saved by the application before supplying PersistedWorkingView. This
provider validates their identities and stores metadata only; it never decodes
or opens images. A failed transaction may leave an unreferenced file for the
application's artifact cleanup, but never a partially committed receipt.
"""

import json
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from pydantic import TypeAdapter

from ...config import PGSettings, WorkerSettings
from ...model.db.image import ImageRow
from ...model.ocr import OcrPageMetadata, Sha256, SourceRef, WorkingImageMetadata
from ...model.workflow import (
    AccessScope,
    AnalysisClaim,
    AttemptOutcome,
    CommitResult,
    EnqueueResult,
    OcrCheckpoint,
    PersistedSourcePart,
    PersistedWorkingView,
    RetryRequest,
    SourcePartBinding,
    SubmissionCreate,
    SubmissionStatus,
    assert_metadata_only,
)
from .receipt_db import ReceiptDB


def _json(value: Any) -> str:
    assert_metadata_only(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decoded(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _status(row: Any) -> SubmissionStatus:
    return SubmissionStatus.model_validate(ImageRow.model_validate(dict(row)).model_dump())


class _ExpiredClaim(Exception):
    """Force rollback if a once-live lease expires before the final seal."""


def _nested_source_refs(value: Any) -> list[SourceRef]:
    """Validate references in flexible provenance/overlap metadata too."""
    refs: list[SourceRef] = []
    if isinstance(value, dict):
        if "part_id" in value:
            refs.append(SourceRef.model_validate(value))
        for item in value.values():
            refs.extend(_nested_source_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_nested_source_refs(item))
    return refs


def _nested_view_ids(value: Any) -> list[UUID]:
    """View identities in flexible metadata must belong to this submission too."""
    ids: list[UUID] = []
    if isinstance(value, dict):
        if value.get("view_id") is not None:
            ids.append(UUID(str(value["view_id"])))
        for item in value.values():
            ids.extend(_nested_view_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.extend(_nested_view_ids(item))
    return ids


class WorkflowDB:
    def __init__(self, settings: PGSettings, *, worker_settings: WorkerSettings | None = None):
        self._settings = settings
        self._worker = worker_settings or WorkerSettings()
        self._pool: asyncpg.Pool | None = None
        self._receipts = ReceiptDB(settings)

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Workflow database is not initialized")
        return self._pool

    @property
    def is_ready(self) -> bool:
        return self._pool is not None

    async def init_db(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(dsn=self._settings.pg_dsn)

    async def destroy_db(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def get_submission(
        self, image_id: UUID, *, scope: AccessScope
    ) -> SubmissionStatus | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM images WHERE id=$1 AND (user_id=$2 OR $3)",
                image_id,
                scope.user_id,
                scope.can_see_all,
            )
        return _status(row) if row else None

    async def get_source_parts(
        self,
        image_id: UUID,
        *,
        scope: AccessScope,
    ) -> list[PersistedSourcePart]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT p.* FROM image_source_parts p JOIN images i
                ON i.id=p.image_id WHERE i.id=$1 AND (i.user_id=$2 OR $3)
                ORDER BY p.photo_index""",
                image_id,
                scope.user_id,
                scope.can_see_all,
            )
        return [PersistedSourcePart.model_validate(dict(row)) for row in rows]

    async def bind_source_parts(
        self,
        claim: AnalysisClaim,
        bindings: list[SourcePartBinding],
    ) -> list[PersistedSourcePart] | None:
        """Initialize unknown legacy source identities inside the live claim fence.

        The caller hashes originals under its attempt deadline. Paths must equal
        the stored ordered originals; established identities cannot be changed.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            row = await self._lock_claim(conn, claim)
            if row is None:
                return None
            originals = ImageRow.model_validate(dict(row)).parts()
            if len(bindings) != len(originals):
                raise ValueError("source binding count mismatch")
            for index, binding in enumerate(bindings):
                if (
                    binding.photo_index != index
                    or binding.image_path != originals[index].image_path
                ):
                    raise ValueError("source binding order/path mismatch")
            manifest = [
                {"photo_index": b.photo_index, "source_sha256": b.source_sha256} for b in bindings
            ]
            old_manifest = _decoded(row["source_manifest"])
            if old_manifest and [(v["photo_index"], v["source_sha256"]) for v in old_manifest] != [
                (b.photo_index, b.source_sha256) for b in bindings
            ]:
                raise ValueError("established source hashes cannot be changed")
            existing = await self._parts(conn, row["id"])
            if existing:
                if [SourcePartBinding.model_validate(p.model_dump()) for p in existing] != bindings:
                    raise ValueError("established source identities cannot be changed")
                return existing
            for index, binding in enumerate(bindings):
                await conn.execute(
                    """INSERT INTO image_source_parts(image_id,photo_index,
                    source_sha256,image_path,metadata) VALUES($1,$2,$3,$4,$5::jsonb)""",
                    row["id"],
                    index,
                    binding.source_sha256,
                    binding.image_path,
                    _json(originals[index].model_dump(mode="json")),
                )
            await conn.execute(
                "UPDATE images SET source_manifest=$2::jsonb WHERE id=$1",
                row["id"],
                _json(manifest),
            )
            return await self._parts(conn, row["id"])

    async def enqueue_submission(
        self,
        submission: SubmissionCreate,
        *,
        scope: AccessScope,
        idempotency_key: str | None,
    ) -> EnqueueResult:
        if idempotency_key is not None and not idempotency_key.strip():
            raise ValueError("idempotency key must not be empty")
        manifest = submission.source_manifest
        if len(manifest) != len(submission.parts):
            raise ValueError("source manifest must match ordered parts")
        for index, source in enumerate(manifest):
            if source.get("photo_index") != index:
                raise ValueError("source photo indices must be ordered and contiguous")
            TypeAdapter(Sha256).validate_python(source.get("source_sha256"))
        async with self.pool.acquire() as conn, conn.transaction():
            if idempotency_key is not None:
                # Serialize absent keys too, rather than racing SELECT then INSERT.
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1,0))",
                    f"upload:{scope.user_id}:{idempotency_key}",
                )
                old = await conn.fetchrow(
                    "SELECT * FROM images WHERE user_id=$1 AND idempotency_key=$2",
                    scope.user_id,
                    idempotency_key,
                )
                if old:
                    expected = (
                        submission.request_fingerprint,
                        [item["source_sha256"] for item in manifest],
                        submission.requested_options,
                    )
                    actual = (
                        old["request_fingerprint"],
                        [item["source_sha256"] for item in _decoded(old["source_manifest"])],
                        _decoded(old["requested_options"]),
                    )
                    if actual != expected:
                        raise ValueError("upload idempotency payload conflict")
                    return EnqueueResult(submission=_status(old), created=False)
            primary = submission.parts[0]
            row = await conn.fetchrow(
                """INSERT INTO images(id,original_filename,media_type,
                size_bytes,image_path,thumbnail_path,user_id,bypass_review,additional_images,
                model_id,source_manifest,requested_options,effective_options,request_fingerprint,
                idempotency_key,max_attempts) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11::jsonb,
                $12::jsonb,$13::jsonb,$14,$15,$16) RETURNING *""",
                submission.id,
                primary.original_filename,
                primary.media_type,
                primary.size_bytes,
                primary.image_path,
                primary.thumbnail_path,
                scope.user_id,
                submission.bypass_review,
                _json([part.model_dump(mode="json") for part in submission.parts[1:]]),
                submission.model_id,
                _json(manifest),
                _json(submission.requested_options),
                _json(submission.effective_options),
                submission.request_fingerprint,
                idempotency_key,
                self._worker.max_attempts,
            )
            for index, part in enumerate(submission.parts):
                await conn.execute(
                    """INSERT INTO image_source_parts(image_id,photo_index,
                    source_sha256,image_path,metadata) VALUES($1,$2,$3,$4,$5::jsonb)""",
                    submission.id,
                    index,
                    manifest[index]["source_sha256"],
                    part.image_path,
                    _json(part.model_dump(mode="json")),
                )
            return EnqueueResult(submission=_status(row), created=True)

    async def claim_next(self, *, lease_seconds: int = 900) -> AnalysisClaim | None:
        if lease_seconds <= self._worker.attempt_timeout_seconds:
            raise ValueError("claim lease must exceed the complete attempt deadline")
        async with self.pool.acquire() as conn, conn.transaction():
            while True:
                row = await conn.fetchrow("""SELECT * FROM images WHERE receipt_id IS NULL AND
                    NOT EXISTS (SELECT 1 FROM receipts r WHERE r.image_id=images.id) AND
                    ((status='pending' AND (retry_at IS NULL OR retry_at<=clock_timestamp())) OR
                     (status='processing' AND lease_expires_at<=clock_timestamp()))
                    ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1""")
                if row is None:
                    return None
                if row["status"] == "processing":
                    await conn.execute(
                        """INSERT INTO receipt_extractions(image_id,generation,
                        attempt_number,outcome_kind,result_code) VALUES($1,$2,$3,$4,'lease_expired')
                        ON CONFLICT(image_id,generation,attempt_number) DO NOTHING""",
                        row["id"],
                        row["generation"],
                        row["attempt_count"],
                        "terminal" if row["attempt_count"] >= row["max_attempts"] else "retry",
                    )
                if row["attempt_count"] >= row["max_attempts"]:
                    await conn.execute(
                        """UPDATE images SET status='failed', result_code='attempt_limit',
                        claim_token=NULL,lease_expires_at=NULL,processing_at=NULL WHERE id=$1""",
                        row["id"],
                    )
                    continue
                claimed = await conn.fetchrow(
                    """UPDATE images SET status='processing',
                    processing_at=clock_timestamp(),claim_token=$2,
                    lease_expires_at=clock_timestamp()+$3::double precision*interval '1 second',
                    attempt_count=attempt_count+1,retry_at=NULL,wait_reason=NULL
                    WHERE id=$1 RETURNING *""",
                    row["id"],
                    uuid4(),
                    lease_seconds,
                )
                return AnalysisClaim(
                    image=ImageRow.model_validate(dict(claimed)),
                    generation=claimed["generation"],
                    attempt_number=claimed["attempt_count"],
                    claim_token=claimed["claim_token"],
                    lease_expires_at=claimed["lease_expires_at"],
                )

    @staticmethod
    async def _lock_claim(conn: asyncpg.Connection, claim: AnalysisClaim) -> Any:
        row = await conn.fetchrow("SELECT * FROM images WHERE id=$1 FOR UPDATE", claim.image.id)
        if row is None:
            return None
        live = await conn.fetchval(
            "SELECT $1::timestamptz>clock_timestamp()", row["lease_expires_at"]
        )
        if (
            row["status"] != "processing"
            or row["generation"] != claim.generation
            or row["attempt_count"] != claim.attempt_number
            or row["claim_token"] != claim.claim_token
            or not live
        ):
            return None
        return row

    @staticmethod
    async def _parts(conn: asyncpg.Connection, image_id: UUID) -> list[PersistedSourcePart]:
        rows = await conn.fetch(
            "SELECT * FROM image_source_parts WHERE image_id=$1 ORDER BY photo_index", image_id
        )
        return [PersistedSourcePart.model_validate(dict(row)) for row in rows]

    @staticmethod
    async def _save_views(
        conn: asyncpg.Connection,
        image_id: UUID,
        parts: list[PersistedSourcePart],
        views: list[PersistedWorkingView],
    ) -> dict[UUID, PersistedWorkingView]:
        bound: dict[UUID, PersistedWorkingView] = {}
        for view in views:
            if view.photo_index >= len(parts):
                raise ValueError("working view photo index does not match real source parts")
            part = parts[view.photo_index]
            if view.part_id != part.part_id or view.source_sha256 != part.source_sha256:
                raise ValueError("working view does not match real source part")
            metadata = view.metadata.model_dump(mode="json")
            await conn.execute(
                """INSERT INTO extraction_working_views(view_id,image_id,part_id,
                artifact_path,metadata) VALUES($1,$2,$3,$4,$5::jsonb) ON CONFLICT(view_id) DO NOTHING""",
                view.metadata.view_id,
                image_id,
                view.part_id,
                view.artifact_path,
                _json(metadata),
            )
            saved = await conn.fetchrow(
                "SELECT * FROM extraction_working_views WHERE view_id=$1", view.metadata.view_id
            )
            if (
                saved["image_id"] != image_id
                or saved["part_id"] != view.part_id
                or saved["artifact_path"] != view.artifact_path
                or _decoded(saved["metadata"]) != metadata
            ):
                raise ValueError("immutable working artifact conflict")
            bound[view.metadata.view_id] = view
        return bound

    @staticmethod
    def _require_view(
        parts: list[PersistedSourcePart],
        views: dict[UUID, PersistedWorkingView],
        photo_index: int,
        source_sha256: str,
        metadata: WorkingImageMetadata,
    ) -> None:
        if photo_index >= len(parts) or parts[photo_index].source_sha256 != source_sha256:
            raise ValueError("source index/hash mismatch")
        saved = views.get(metadata.view_id)
        projected = WorkingImageMetadata.model_validate(metadata.model_dump()).model_dump(
            mode="json"
        )
        if (
            saved is None
            or saved.photo_index != photo_index
            or saved.source_sha256 != source_sha256
            or saved.metadata.model_dump(mode="json") != projected
        ):
            raise ValueError("missing or mismatched saved working artifact")

    @staticmethod
    async def _validate_refs(
        conn: asyncpg.Connection,
        image_id: UUID,
        parts: list[PersistedSourcePart],
        references: list[SourceRef],
    ) -> None:
        for ref in references:
            if ref.photo_index >= len(parts):
                raise ValueError("source reference photo index mismatch")
            part = parts[ref.photo_index]
            if ref.part_id != part.part_id or ref.source_sha256 != part.source_sha256:
                raise ValueError("source reference must bind to a real source part")
            if ref.view_id is not None:
                exists = await conn.fetchval(
                    """SELECT EXISTS(SELECT 1 FROM extraction_working_views
                    WHERE view_id=$1 AND image_id=$2 AND part_id=$3)""",
                    ref.view_id,
                    image_id,
                    ref.part_id,
                )
                if not exists:
                    raise ValueError("source reference view does not match saved artifact")
            if ref.coordinate_space == "working" and ref.view_id is None:
                raise ValueError("working source reference requires saved view")

    async def _validate_metadata_refs(
        self,
        conn: asyncpg.Connection,
        image_id: UUID,
        parts: list[PersistedSourcePart],
        value: Any,
    ) -> None:
        await self._validate_refs(conn, image_id, parts, _nested_source_refs(value))
        for view_id in set(_nested_view_ids(value)):
            if not await conn.fetchval(
                """SELECT EXISTS(SELECT 1 FROM extraction_working_views
                WHERE view_id=$1 AND image_id=$2)""",
                view_id,
                image_id,
            ):
                raise ValueError("source reference view does not match saved artifact")

    async def get_ocr_checkpoint(self, claim: AnalysisClaim) -> OcrCheckpoint | None:
        """Read internal metadata under a live fence, including saved-view paths.

        The application checks the actual OCR profile/runtime before reuse and
        loads saved artifacts through FileProvider. This never reconstructs PNGs.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            row = await self._lock_claim(conn, claim)
            if row is None or row["ocr_checkpoint"] is None:
                return None
            return OcrCheckpoint.model_validate(_decoded(row["ocr_checkpoint"]))

    async def save_ocr_checkpoint(self, claim: AnalysisClaim, checkpoint: OcrCheckpoint) -> bool:
        async with self.pool.acquire() as conn, conn.transaction():
            row = await self._lock_claim(conn, claim)
            if row is None:
                return False
            parts = await self._parts(conn, row["id"])
            if checkpoint.ordered_source_sha256 != [part.source_sha256 for part in parts]:
                raise ValueError("checkpoint ordered source hashes mismatch")
            views = await self._save_views(conn, row["id"], parts, checkpoint.persisted_views)
            for index, raw in enumerate(checkpoint.pages):
                page = OcrPageMetadata.model_validate(raw)
                if (
                    page.photo_index != index
                    or page.profile_id != checkpoint.profile_id
                    or page.runtime_fingerprint != checkpoint.runtime_fingerprint
                ):
                    raise ValueError("checkpoint page order/profile mismatch")
                self._require_view(
                    parts, views, page.photo_index, page.source_sha256, page.working_image
                )
            payload = checkpoint.model_dump(mode="json")
            await self._validate_metadata_refs(conn, row["id"], parts, payload)
            old = _decoded(row["ocr_checkpoint"])
            if old is not None and (
                old["ordered_source_sha256"] != payload["ordered_source_sha256"]
                or old["profile_id"] != payload["profile_id"]
                or old["runtime_fingerprint"] != payload["runtime_fingerprint"]
                or payload["pages"][: len(old["pages"])] != old["pages"]
            ):
                raise ValueError("immutable checkpoint source/profile/page conflict")
            await conn.execute(
                "UPDATE images SET ocr_checkpoint=$2::jsonb WHERE id=$1", row["id"], _json(payload)
            )
            return True

    async def finish_attempt(self, claim: AnalysisClaim, outcome: AttemptOutcome) -> CommitResult:
        try:
            return await self._finish_attempt_transaction(claim, outcome)
        except _ExpiredClaim:
            return CommitResult(stale=True)

    async def _finish_attempt_transaction(
        self,
        claim: AnalysisClaim,
        outcome: AttemptOutcome,
    ) -> CommitResult:
        async with self.pool.acquire() as conn, conn.transaction():
            row = await self._lock_claim(conn, claim)
            if row is None:
                return CommitResult(stale=True)
            # Lock and preserve human work, including an association added after claim.
            existing = await conn.fetchrow(
                "SELECT id FROM receipts WHERE image_id=$1 FOR UPDATE", row["id"]
            )
            draft = await conn.fetchrow(
                "SELECT * FROM receipt_drafts WHERE image_id=$1 FOR UPDATE", row["id"]
            )
            if row["receipt_id"] is not None or existing is not None:
                raise ValueError("a linked receipt must never be overwritten")
            if draft is not None and (
                draft["human_edited"] or draft["finalized_receipt_id"] is not None
            ):
                raise ValueError("an edited or finalized draft must never be overwritten")
            parts = await self._parts(conn, row["id"])
            views = await self._save_views(conn, row["id"], parts, outcome.persisted_views)
            envelope = outcome.envelope
            payload = None
            if envelope is not None:
                if envelope.provenance.ordered_source_sha256 != [
                    p.source_sha256 for p in parts
                ] or envelope.provenance.source_image_count != len(parts):
                    raise ValueError("envelope ordered source hashes mismatch")
                for page in envelope.ocr_pages:
                    self._require_view(
                        parts, views, page.photo_index, page.source_sha256, page.working_image
                    )
                for view in envelope.input_views:
                    self._require_view(
                        parts, views, view.photo_index, view.source_sha256, view.working_image
                    )
                payload = envelope.model_dump(mode="json", exclude={"ocr_pages", "input_views"})
                payload["ocr_pages"] = [
                    OcrPageMetadata.model_validate(page.model_dump()).model_dump(mode="json")
                    for page in envelope.ocr_pages
                ]
                payload["input_views"] = [
                    view.model_dump(mode="json") for view in outcome.persisted_views
                ]
            review_payload = outcome.review.model_dump(mode="json") if outcome.review else None
            saved_views = [view.model_dump(mode="json") for view in outcome.persisted_views]
            await self._validate_metadata_refs(
                conn,
                row["id"],
                parts,
                {"envelope": payload, "review": review_payload, "saved_views": saved_views},
            )
            extraction_id = await conn.fetchval(
                """INSERT INTO receipt_extractions(image_id,generation,
                attempt_number,outcome_kind,result_code,envelope,review,source_views)
                VALUES($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb,$8::jsonb) RETURNING id""",
                row["id"],
                claim.generation,
                claim.attempt_number,
                outcome.kind,
                outcome.result_code,
                _json(payload) if payload is not None else None,
                _json(review_payload) if review_payload is not None else None,
                _json(saved_views),
            )
            receipt = None
            status = "failed"
            delay = None
            result_code = outcome.result_code
            if outcome.kind == "complete":
                assert (
                    envelope is not None
                    and envelope.receipt is not None
                    and outcome.review is not None
                )
                assert payload is not None
                verified = row["bypass_review"] and outcome.review.can_auto_verify
                receipt = await self._receipts.insert_receipt(
                    conn,
                    envelope.receipt,
                    image_id=row["id"],
                    user_id=row["user_id"],
                    status="verified" if verified else "unverified",
                    verified=verified,
                )
                record = await conn.fetchrow(
                    """UPDATE receipts SET extraction_id=$2,amount_policy=$3::jsonb,
                    findings=$4::jsonb,field_sources=$5::jsonb,checks_version=$6,verification_source=$7
                    WHERE id=$1 RETURNING *""",
                    receipt.id,
                    extraction_id,
                    envelope.amount_policy.model_dump_json(),
                    _json([f.model_dump(mode="json") for f in outcome.review.findings]),
                    _json(payload["field_sources"]),
                    outcome.review.checks_version,
                    "automatic" if verified else None,
                )
                receipt = self._receipts._receipt_row_from_record(record)
                await conn.execute(
                    """INSERT INTO receipt_review_revisions(receipt_id,operation_revision,
                    review_revision,event_kind,snapshot) VALUES($1,1,1,'extracted',$2::jsonb)""",
                    receipt.id,
                    _json(
                        {
                            "receipt": receipt.model_dump(mode="json"),
                            "extraction_id": str(extraction_id),
                            "field_sources": payload["field_sources"],
                        }
                    ),
                )
                status = "analyzed"
            elif outcome.kind == "partial":
                assert envelope is not None and envelope.draft is not None
                assert payload is not None
                await conn.execute(
                    """INSERT INTO receipt_drafts(image_id,extraction_id,generation,document,
                    amount_policy,findings,field_sources) VALUES($1,$2,$3,$4::jsonb,$5::jsonb,$6::jsonb,$7::jsonb)
                    ON CONFLICT(image_id) DO UPDATE SET extraction_id=EXCLUDED.extraction_id,
                    generation=EXCLUDED.generation,revision=receipt_drafts.revision+1,
                    document=EXCLUDED.document,amount_policy=EXCLUDED.amount_policy,
                    findings=EXCLUDED.findings,field_sources=EXCLUDED.field_sources""",
                    row["id"],
                    extraction_id,
                    claim.generation,
                    envelope.draft.model_dump_json(),
                    envelope.amount_policy.model_dump_json(),
                    _json(
                        [finding.model_dump(mode="json") for finding in outcome.review.findings]
                        if outcome.review
                        else payload["findings"]
                    ),
                    _json(payload["field_sources"]),
                )
                status = "needs_review"
            elif outcome.kind == "retry" and row["attempt_count"] < row["max_attempts"]:
                status = "pending"
                delay = min(
                    outcome.retry_delay_seconds
                    if outcome.retry_delay_seconds is not None
                    else self._worker.retry_backoff_seconds[claim.attempt_number - 1],
                    self._worker.retry_backoff_cap_seconds,
                )
            elif outcome.kind == "retry":
                result_code = result_code or "attempt_limit"
            elif outcome.result_code in {"timed_out", "unreadable"}:
                status = outcome.result_code
            updated = await conn.fetchrow(
                """UPDATE images SET status=$2,receipt_id=$3,result_code=$4,
                error=$4,claim_token=NULL,lease_expires_at=NULL,processing_at=NULL,
                retry_at=CASE WHEN $5::double precision IS NULL THEN NULL ELSE
                    clock_timestamp()+$5::double precision*interval '1 second' END,
                analyzed_at=CASE WHEN $2::varchar IN ('analyzed','needs_review') THEN clock_timestamp() ELSE analyzed_at END,
                draft_available=$6 WHERE id=$1 AND lease_expires_at>clock_timestamp()
                AND generation=$7 AND claim_token=$8 AND status='processing' RETURNING *""",
                row["id"],
                status,
                receipt.id if receipt else None,
                result_code,
                delay,
                status == "needs_review",
                claim.generation,
                claim.claim_token,
            )
            if updated is None:
                raise _ExpiredClaim
            return CommitResult(stale=False, submission=_status(updated), receipt=receipt)

    async def retry_submission(
        self,
        image_id: UUID,
        request: RetryRequest,
        *,
        scope: AccessScope,
    ) -> SubmissionStatus | None:
        async with self.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM images WHERE id=$1 AND (user_id=$2 OR $3) FOR UPDATE",
                image_id,
                scope.user_id,
                scope.can_see_all,
            )
            if row is None:
                return None
            old = await conn.fetchrow(
                "SELECT * FROM image_retry_requests WHERE image_id=$1 AND request_key=$2",
                image_id,
                request.request_key,
            )
            if old is not None:
                if old["expected_generation"] != request.expected_generation:
                    raise ValueError("retry key payload conflict")
                return SubmissionStatus.model_validate(_decoded(old["result"]))
            if row["generation"] != request.expected_generation:
                raise ValueError("retry generation conflict")
            if row["receipt_id"] is not None or await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM receipts WHERE image_id=$1)", image_id
            ):
                raise ValueError("a linked receipt cannot be retried")
            draft = await conn.fetchrow(
                "SELECT * FROM receipt_drafts WHERE image_id=$1 FOR UPDATE", image_id
            )
            if draft and (draft["human_edited"] or draft["finalized_receipt_id"] is not None):
                raise ValueError("an edited draft cannot be retried")
            if row["status"] not in {"failed", "timed_out", "unreadable", "needs_review"}:
                raise ValueError("submission is not terminal/retryable")
            updated = await conn.fetchrow(
                """UPDATE images SET generation=generation+1,attempt_count=0,
                status='pending',retry_at=NULL,claim_token=NULL,lease_expires_at=NULL,processing_at=NULL,
                analyzed_at=NULL,error=NULL,result_code=NULL,wait_reason=NULL,ocr_checkpoint=NULL
                WHERE id=$1 RETURNING *""",
                image_id,
            )
            result = _status(updated)
            await conn.execute(
                """INSERT INTO image_retry_requests(image_id,request_key,expected_generation,result)
                VALUES($1,$2,$3,$4::jsonb)""",
                image_id,
                request.request_key,
                request.expected_generation,
                result.model_dump_json(),
            )
            return result
