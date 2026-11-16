"""Durable execution uses the saved snapshot and one complete attempt budget."""

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from PIL import Image

from vision_bill.config import OcrSettings
from vision_bill.model.db.image import ImageRow
from vision_bill.model.workflow import AnalysisClaim, CommitResult, PersistedSourcePart


def context(settings, tmp_path):
    from vision_bill.service.image_service import ImageService
    from vision_bill.service.submission_service import SubmissionService

    path = tmp_path / "original.png"
    Image.new("RGB", (4, 5)).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    image = ImageRow(
        id=uuid4(),
        image_path=str(path),
        model_id="saved-model",
        effective_options={
            "ocr": OcrSettings().model_dump(mode="json"),
            "temperature": 0.13,
            "max_output_tokens": 1234,
        },
    )
    claim = AnalysisClaim(
        image=image,
        generation=1,
        attempt_number=1,
        claim_token=uuid4(),
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=900),
    )
    part = PersistedSourcePart(
        part_id=uuid4(), photo_index=0, source_sha256=digest, image_path=str(path)
    )
    db = SimpleNamespace(
        bind_source_parts=AsyncMock(return_value=[part]),
        get_ocr_checkpoint=AsyncMock(return_value=None),
        finish_attempt=AsyncMock(return_value=CommitResult(stale=False)),
        clear_ocr_checkpoint=AsyncMock(return_value=True),
        save_ocr_checkpoint=AsyncMock(return_value=True),
    )
    receipt = SimpleNamespace(
        workflow_db=db, extract_receipt=AsyncMock(), list_tags=AsyncMock(return_value=[])
    )
    service = SubmissionService(settings, receipt, ImageService(settings.images))
    return service, receipt, db, claim, part


async def test_original_mismatch_terminal_and_safe(settings, tmp_path):
    service, receipt, db, claim, _part = context(settings, tmp_path)
    db.bind_source_parts.side_effect = ValueError("sensitive original mismatch")
    await service.process_claim(claim)
    receipt.extract_receipt.assert_not_awaited()
    outcome = db.finish_attempt.call_args.args[1]
    assert outcome.kind == "terminal" and outcome.result_code == "source_integrity_failed"


async def test_saved_snapshot_is_forwarded_without_current_defaults(settings, tmp_path):
    service, receipt, db, claim, part = context(settings, tmp_path)
    receipt.list_tags.return_value = ["coffee", "food"]
    receipt.extract_receipt.side_effect = TimeoutError()
    await service.process_claim(claim)
    args, kwargs = receipt.extract_receipt.call_args
    assert args[0] == "saved-model"
    assert kwargs["tags"] == ["coffee", "food"]
    assert kwargs["options"].temperature == 0.13
    assert kwargs["max_output_tokens"] == 1234
    assert kwargs["ocr_settings"].enabled is False
    assert kwargs["source_refs"][0].part_id == part.part_id
    assert db.finish_attempt.call_args.args[1].kind == "retry"


async def test_hung_attempt_includes_binding_in_original_budget(settings, tmp_path):
    service, _receipt, db, claim, _part = context(settings, tmp_path)
    # A short controlled budget exercises exactly the same 600-second outer timer.
    service.worker = settings.worker.model_copy(update={"attempt_timeout_seconds": 0.2})
    entered = asyncio.Event()

    async def hang(*args):
        entered.set()
        await asyncio.Event().wait()

    db.bind_source_parts.side_effect = hang
    async with asyncio.timeout(0.8):
        await service.process_claim(claim)
    assert entered.is_set()
    assert db.finish_attempt.call_args.args[1].result_code == "timed_out"


async def test_working_artifact_validates_png_hash_dimensions_and_submission(settings, tmp_path):
    import base64
    from time import monotonic

    from test_ocr_provider import wire_page

    from vision_bill.model.ocr import WorkingImage, WorkingImageMetadata
    from vision_bill.service.image_service import ImageService

    service = ImageService(settings.images)
    image_id = uuid4()
    working = WorkingImage.model_validate(wire_page()["working_image"])
    path = await service.save_working_image(image_id, working, deadline=monotonic() + 2)
    metadata = WorkingImageMetadata.model_validate(working.model_dump())
    assert service.read_working_image(image_id, str(path), metadata) == base64.b64decode(
        working.png_base64
    )
    with pytest.raises(ValueError):
        service.read_working_image(uuid4(), str(path), metadata)
    with pytest.raises(ValueError):
        service.read_working_image(image_id, str(path), metadata.model_copy(update={"width": 999}))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        service.read_working_image(image_id, str(path), metadata)


async def test_stale_checkpoint_cleans_only_unreferenced_new_artifact(settings, tmp_path):
    from test_ocr_provider import wire_page

    from vision_bill.model.ocr import OcrResponse

    service, receipt, db, claim, part = context(settings, tmp_path)
    db.save_ocr_checkpoint.return_value = False
    db.unreferenced_views = AsyncMock(side_effect=lambda image_id, views: views)

    async def extract(*args, **kwargs):
        await kwargs["checkpoint"](OcrResponse.model_validate(wire_page(0, part.source_sha256)))
        raise AssertionError("must stop before formatting")

    receipt.extract_receipt.side_effect = extract
    result = await service.process_claim(claim)
    assert result.stale
    assert not list((tmp_path / "uploads" / "working").glob("*/*.png"))
    db.finish_attempt.assert_not_awaited()


async def test_hung_outcome_commit_cannot_extend_complete_attempt_budget(settings, tmp_path):
    service, receipt, db, claim, _part = context(settings, tmp_path)
    service.worker = settings.worker.model_copy(update={"attempt_timeout_seconds": 0.02})
    receipt.extract_receipt.side_effect = TimeoutError()

    async def hang(*args):
        await asyncio.Event().wait()

    db.finish_attempt.side_effect = hang
    async with asyncio.timeout(0.3):
        result = await service.process_claim(claim)
    assert result.stale


async def test_preclaim_ocr_text_model_is_eligible_but_missing_model_waits(settings, tmp_path):
    from vision_bill.model.ocr import OcrCapabilities
    from vision_bill.provider.llm.base import ModelInfo

    service, receipts, _db, claim, _part = context(settings, tmp_path)
    claim.image.effective_options["ocr"] = OcrSettings(enabled=True, required=True).model_dump(
        mode="json"
    )
    receipts.check_connection = AsyncMock(return_value=True)
    receipts.get_available_models = AsyncMock(return_value=[ModelInfo("saved-model")])
    receipts.ocr_capabilities = AsyncMock(
        return_value=OcrCapabilities(
            enabled=True,
            configured=True,
            reachable=True,
            ready=True,
            profile_id="profile",
            runtime_fingerprint="runtime",
        )
    )
    assert await service.ready_for(claim.image)
    receipts.get_available_models.assert_awaited_once_with(capability="text")
    receipts.get_available_models.return_value = []
    assert not await service.ready_for(claim.image)


@pytest.mark.parametrize("unreadable", [True, False])
async def test_worker_errors_follow_durable_retry_policy(settings, tmp_path, unreadable):
    from vision_bill.provider.llm.errors import AnalysisTimeoutError, UnreadableReceiptError

    service, receipt, db, claim, _part = context(settings, tmp_path)
    receipt.extract_receipt.side_effect = (
        UnreadableReceiptError("sensitive text")
        if unreadable
        else AnalysisTimeoutError("sensitive text")
    )
    await service.process_claim(claim)
    outcome = db.finish_attempt.call_args.args[1]
    assert outcome.kind == ("terminal" if unreadable else "retry")
    assert outcome.result_code == ("unreadable" if unreadable else "timed_out")


async def test_worker_passes_every_original_in_capture_order(settings, tmp_path):
    from vision_bill.model.db.image import ImagePart

    service, receipt, db, claim, part = context(settings, tmp_path)
    second = tmp_path / "second.png"
    Image.new("RGB", (6, 7)).save(second)
    digest = hashlib.sha256(second.read_bytes()).hexdigest()
    claim.image.additional_images = [ImagePart(image_path=str(second))]
    db.bind_source_parts.return_value.append(
        PersistedSourcePart(
            part_id=uuid4(), photo_index=1, source_sha256=digest, image_path=str(second)
        )
    )
    receipt.extract_receipt.side_effect = TimeoutError()
    await service.process_claim(claim)
    assert receipt.extract_receipt.call_args.args[1] == [Path(part.image_path), second]


async def test_preclaim_malformed_provider_response_does_not_spend_attempt(settings, tmp_path):
    service, receipts, _db, claim, _part = context(settings, tmp_path)
    receipts.check_connection = AsyncMock(return_value=True)
    receipts.get_available_models = AsyncMock(side_effect=ValueError("malformed provider JSON"))
    assert not await service.ready_for(claim.image)


async def test_controlled_clock_exhausts_original_600_seconds_before_commit(
    settings, tmp_path, monkeypatch
):
    import vision_bill.helper.deadline as deadlines
    import vision_bill.service.submission_service as submissions

    service, receipt, db, claim, _part = context(settings, tmp_path)
    clock = [100.0]
    monkeypatch.setattr(submissions, "monotonic", lambda: clock[0])
    monkeypatch.setattr(deadlines, "monotonic", lambda: clock[0])

    async def expired(*args, **kwargs):
        assert kwargs["deadline"] == 700.0
        clock[0] = 700.01
        raise TimeoutError()

    receipt.extract_receipt.side_effect = expired
    result = await service.process_claim(claim)
    assert result.stale
    db.finish_attempt.assert_not_awaited()


async def test_checkpoint_transport_failure_does_not_clear_cache_or_run_ocr(settings, tmp_path):
    import asyncpg

    service, receipts, db, claim, _part = context(settings, tmp_path)
    db.get_ocr_checkpoint.side_effect = asyncpg.ConnectionDoesNotExistError("connection lost")
    await service.process_claim(claim)
    db.clear_ocr_checkpoint.assert_not_awaited()
    receipts.extract_receipt.assert_not_awaited()
    outcome = db.finish_attempt.call_args.args[1]
    assert outcome.kind == "retry" and outcome.result_code == "analysis_unavailable"


@pytest.mark.parametrize("permission", [None, False, True, "true"])
async def test_saved_fallback_permission_used_in_readiness_and_execution(
    settings, tmp_path, permission
):
    from vision_bill.model.ocr import OcrCapabilities
    from vision_bill.provider.llm.base import ModelInfo

    service, receipts, db, claim, _ = context(settings, tmp_path)
    claim.image.effective_options["ocr"] = OcrSettings(
        enabled=True, allow_vision_fallback=True
    ).model_dump(mode="json")
    if permission is not None:
        claim.image.effective_options["allow_ocr_fallback"] = permission
    snapshot = dict(claim.image.effective_options)
    receipts.check_connection = AsyncMock(return_value=True)
    receipts.ocr_capabilities = AsyncMock(
        return_value=OcrCapabilities(enabled=True, configured=True, reachable=True, ready=False)
    )
    receipts.get_available_models = AsyncMock(return_value=[ModelInfo("saved-model")])
    assert await service.ready_for(claim.image) is (
        permission is True or isinstance(permission, str)
    )
    receipts.extract_receipt.side_effect = TimeoutError()
    await service.process_claim(claim)
    if isinstance(permission, str):
        receipts.extract_receipt.assert_not_awaited()
        assert db.finish_attempt.call_args.args[1].result_code == "analysis_contract_failed"
    else:
        assert receipts.extract_receipt.call_args.kwargs["options"].allow_ocr_fallback is (
            permission is True
        )
    assert claim.image.effective_options == snapshot
