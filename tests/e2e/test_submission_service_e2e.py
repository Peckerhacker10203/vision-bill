"""Actual PostgreSQL transactions, actual PNGs, typed pipeline and deterministic inference."""

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image
from test_extraction_pipeline import formatter, sidecar

from vision_bill.api.drafts import router
from vision_bill.config import OcrSettings
from vision_bill.model.db.image import ImagePart
from vision_bill.model.workflow import SubmissionCreate
from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User
from vision_bill.service.extraction_pipeline import ExtractionPipeline
from vision_bill.service.image_service import ImageService
from vision_bill.service.receipt_service import ReceiptService
from vision_bill.service.submission_service import SubmissionService

from .test_workflow_db_e2e import db as db  # noqa: PLC0414 - pytest fixture re-export
from .test_workflow_db_e2e import migrate
from .test_workflow_db_e2e import postgres as postgres  # noqa: PLC0414 - pytest fixture re-export
from .test_workflow_db_e2e import (
    workflow_db as workflow_db,  # noqa: PLC0414 - pytest fixture re-export
)

pytestmark = pytest.mark.e2e


async def setup(w, scope, settings, tmp_path, *, ocr=True):
    paths = [tmp_path / f"{i}.png" for i in range(2)]
    for i, path in enumerate(paths):
        Image.new("RGB", (4, 5), (i, 0, 0)).save(path)
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    config = OcrSettings(enabled=ocr, required=ocr)
    upload = SubmissionCreate(
        id=uuid4(),
        parts=[ImagePart(image_path=str(p)) for p in paths],
        source_manifest=[{"photo_index": i, "source_sha256": h} for i, h in enumerate(hashes)],
        requested_options={},
        effective_options={
            "ocr": config.model_dump(mode="json"),
            "temperature": 0.13,
            "max_output_tokens": 1234,
        },
        request_fingerprint=uuid4().hex,
        model_id="fixture",
        bypass_review=True,
    )
    image = (await w.enqueue_submission(upload, scope=scope, idempotency_key=None)).submission
    llm, native = formatter(not ocr), sidecar()
    native.get_capabilities.return_value.runtime_fingerprint = "runtime"
    receipts = ReceiptService(settings.images, settings.pg, llm)
    receipts._workflow_db = w
    receipts.list_tags = AsyncMock(return_value=[])
    receipts.ocr_capabilities = native.get_capabilities

    async def extract(model, paths, *, ocr_settings, **kwargs):
        return await ExtractionPipeline(llm, native if ocr else None, ocr_settings).extract(
            model, paths, **kwargs
        )

    receipts.extract_receipt = extract
    images = ImageService(settings.images)
    service = SubmissionService(settings, receipts, images)
    return image, paths, llm, native, service


@pytest.mark.parametrize("change", ["none", "runtime", "corrupt", "source", "race"])
async def test_checkpoint_reload_reuses_only_exact_prefix(workflow_db, settings, tmp_path, change):
    w, scope = workflow_db
    image, paths, _llm, native, service = await setup(w, scope, settings, tmp_path)
    original_extract = native.extract_page.side_effect

    async def fail_second(*args, **kwargs):
        if kwargs["photo_index"] == 1:
            raise TimeoutError()
        return await original_extract(*args, **kwargs)

    native.extract_page.side_effect = fail_second
    first = await service.process_claim(await w.claim_next())
    assert first.submission.status == "pending" and not first.submission.terminal
    async with w.pool.acquire() as conn:
        checkpoint = await conn.fetchval("SELECT ocr_checkpoint FROM images WHERE id=$1", image.id)
        assert "png_base64" not in checkpoint
        await conn.execute("UPDATE images SET retry_at=clock_timestamp() WHERE id=$1", image.id)
    claim = await w.claim_next()
    saved = await w.get_ocr_checkpoint(claim)
    if change == "runtime":
        native.get_capabilities.return_value.runtime_fingerprint = "new-runtime"
    if change == "race":
        old = native.get_capabilities.return_value.model_copy()
        new = old.model_copy(update={"runtime_fingerprint": "new-runtime"})
        native.get_capabilities.side_effect = [old, new, new]
    if change == "corrupt":
        Path(saved.persisted_views[0].artifact_path).write_bytes(b"corrupt")
    if change == "source":
        paths[0].write_bytes(b"changed original")

    async def page(*args, **kwargs):
        result = await original_extract(*args, **kwargs)
        if change in {"runtime", "race"}:
            result.runtime_fingerprint = "new-runtime"
        return result

    native.extract_page = AsyncMock(side_effect=page)
    second = await service.process_claim(claim)
    if change == "source":
        assert second.submission.status == "failed"
        assert second.submission.result_code == "source_integrity_failed"
        native.extract_page.assert_not_awaited()
        return
    assert second.submission.status == "analyzed"
    assert not second.receipt.verified
    assert native.extract_page.await_count == (1 if change == "none" else 2)
    assert all(p.exists() for p in paths)
    status = await w.get_submission(image.id, scope=scope)
    extraction = await w.get_extraction(image.id, status.latest_extraction_id, scope=scope)
    assert len(extraction["envelope"]["ocr_pages"]) == 2
    assert "artifact_path" not in str(extraction) and "png_base64" not in str(extraction)
    assert status.source_views
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 2
        assert await conn.fetchval("SELECT count(*) FROM extraction_working_views") == (
            2 if change == "none" else 3
        )


async def test_direct_source_routes_owner_admin_integrity_and_exact_bytes(
    workflow_db, settings, tmp_path
):
    w, scope = workflow_db
    image, _paths, _llm, _native, service = await setup(w, scope, settings, tmp_path, ocr=False)
    result = await service.process_claim(await w.claim_next())
    assert result.receipt and not result.receipt.verified
    status = await w.get_submission(image.id, scope=scope)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/images")
    app.state.submission_service = service
    user = User(id=scope.user_id, username="owner")
    app.dependency_overrides[get_current_user] = lambda: user
    extraction_url = f"/api/v1/images/{image.id}/extractions/{status.latest_extraction_id}"
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.get(extraction_url)
        assert response.status_code == 200 and "artifact_path" not in response.text
        view = status.source_views[0]
        manifest = await w.get_working_view(image.id, view.view_id, scope=scope)
        png = await client.get(view.url)
        assert png.content == Path(manifest.artifact_path).read_bytes()
        assert png.headers["content-type"] == "image/png"
        user = User(id=uuid4(), username="other")
        assert (await client.get(view.url)).status_code == 404
        assert (await client.get(extraction_url)).status_code == 404
        user.can_see_all = True
        assert (await client.get(view.url)).status_code == 200
        Path(manifest.artifact_path).write_bytes(b"tampered")
        assert (await client.get(view.url)).status_code == 404


async def test_cancelled_formatter_retains_checkpoint_and_no_receipt(
    workflow_db, settings, tmp_path
):
    w, scope = workflow_db
    _image, _paths, llm, _native, service = await setup(w, scope, settings, tmp_path)
    entered = asyncio.Event()

    async def hang(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    llm._client.chat.side_effect = hang
    claim = await w.claim_next()
    task = asyncio.create_task(service.process_claim(claim))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len((await w.get_ocr_checkpoint(claim)).pages) == 2
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 0
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 0


@pytest.mark.parametrize("old_status", ["pending", "failed"])
async def test_migrated_legacy_first_resolution_persisted_once(db, settings, tmp_path, old_status):
    from vision_bill.provider.db.workflow_db import WorkflowDB

    migrate(db, "0010")
    path = tmp_path / "legacy.png"
    Image.new("RGB", (4, 5)).save(path)
    conn = await asyncpg.connect(db.pg_dsn)
    image_id = await conn.fetchval(
        "INSERT INTO images(status,image_path) VALUES($1,$2) RETURNING id", old_status, str(path)
    )
    await conn.close()
    migrate(db)
    w = WorkflowDB(db)
    await w.init_db()
    try:
        llm = formatter()
        settings.llm.model_name = "fixture"
        receipts = ReceiptService(settings.images, db, llm)
        receipts._workflow_db = w
        receipts.list_tags = AsyncMock(return_value=[])
        service = SubmissionService(settings, receipts, ImageService(settings.images))
        claim = await w.claim_next()
        snapshot = service.legacy_snapshot()
        first = await w.ensure_execution_snapshot(
            claim, model_id="fixture", effective_options=snapshot
        )
        settings.llm.model_name = "changed"
        settings.llm.temperature = 0.99
        repeated = await w.ensure_execution_snapshot(
            claim, model_id="changed", effective_options=service.legacy_snapshot()
        )
        assert repeated.model_id == first.model_id == "fixture"
        assert repeated.effective_options == snapshot
        result = await service.process_claim(claim)
        assert result.receipt and result.submission.attempt_count == 1
        assert result.submission.id == image_id and path.exists()
        assert await w.claim_next() is None
    finally:
        await w.destroy_db()
