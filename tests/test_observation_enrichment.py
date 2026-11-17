"""All-photo checkpoints and optional labeling at the real provider boundaries."""

import asyncio
import hashlib
from decimal import Decimal
from pathlib import Path
from time import monotonic
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from handoff_fixtures import install_owned_bindings
from ollama import ChatResponse, ProcessResponse
from PIL import Image
from test_experiment_scoring import receipt
from test_extraction_pipeline import formatter, pipeline, sidecar
from test_gpu_handoff import enable_managed_ocr
from test_submission_service import context

from vision_bill.config import OcrSettings
from vision_bill.model.db.image import ImagePart
from vision_bill.model.experiment import ExperimentOptions
from vision_bill.model.workflow import PersistedSourcePart
from vision_bill.service.experiment_execution import ExperimentFailure, execute_submission
from vision_bill.service.extraction_pipeline import ExtractionPipeline
from vision_bill.service.gpu_handoff import current_owner
from vision_bill.service.receipt_service import ReceiptService


def observation_provider():
    raw = receipt().model_dump(mode="json")
    del raw["category"]
    raw["line_items"][0]["spending_category"] = "milk"
    return formatter(payload=raw)


def chat_reply(content):
    return ChatResponse(message={"role": "assistant", "content": content})


@pytest.mark.parametrize("save_succeeds", [True, False])
@pytest.mark.parametrize("mode", ["direct_vision", "ocr"])
async def test_submission_checkpoints_all_views_before_text_labels(
    settings, tmp_path, save_succeeds, mode,
):
    service, _, db, claim, first = context(settings, tmp_path)
    second_path = tmp_path / "second.png"
    Image.new("RGB", (6, 7)).save(second_path)
    second = PersistedSourcePart(
        part_id=uuid4(), photo_index=1, image_path=str(second_path),
        source_sha256=hashlib.sha256(second_path.read_bytes()).hexdigest(),
    )
    claim.image.additional_images = [ImagePart(image_path=str(second_path))]
    db.bind_source_parts.return_value = [first, second]
    db.unreferenced_views = AsyncMock(return_value=[])
    llm = observation_provider()
    receipts = ReceiptService(settings.images, settings.pg, llm)
    if mode == "ocr":
        settings.ocr = OcrSettings(
            enabled=True, service_url="http://local-ocr", manage_model_lifecycle=False,
        )
        claim.image.effective_options["ocr"] = settings.ocr.model_dump(mode="json")
        receipts._ocr_settings = settings.ocr
        receipts._ocr_provider = sidecar()
    receipts.list_tags = AsyncMock(return_value=[])
    service.receipts = receipts
    events = []

    async def save(claim_arg, checkpoint, *, deadline=None):
        await asyncio.sleep(0)
        assert events == []
        assert claim_arg == claim
        assert deadline > asyncio.get_running_loop().time()
        assert checkpoint.source_parts == [first, second]
        assert checkpoint.model_id == "fixture"
        assert checkpoint.envelope.receipt.total == Decimal("4.00")
        assert checkpoint.envelope.draft.category is None
        assert checkpoint.envelope.classification_enrichment.state == "pending"
        assert checkpoint.envelope.provenance.used_pipeline == mode
        assert len(checkpoint.envelope.ocr_pages) == (2 if mode == "ocr" else 0)
        assert db.save_ocr_checkpoint.await_count == (2 if mode == "ocr" else 0)
        assert "png_base64" not in checkpoint.model_dump_json()
        assert [view.photo_index for view in checkpoint.persisted_views] == [0, 1]
        for view in checkpoint.persisted_views:
            assert Path(view.artifact_path).is_file()
            assert service.images.read_working_image(claim.image.id, view.artifact_path, view.metadata)
        if save_succeeds:
            events.append("checkpoint_committed")
        return save_succeeds

    db.save_observation_checkpoint = AsyncMock(side_effect=save)
    formatted = llm._client.chat.return_value

    async def chat(**request):
        messages = request["messages"]
        prompt = messages[0]["content"]
        if not prompt.startswith("Classify these ordered"):
            if mode == "ocr":
                assert "PHOTO 0" in prompt and "PHOTO 1" in prompt
                assert all("images" not in message for message in messages)
            else:
                assert len(messages[0]["images"]) == 2
            return formatted
        events.append("classification_dispatched")
        assert "checkpoint_committed" in events
        assert all("images" not in message for message in messages)
        return chat_reply('{"category":"grocery","line_items":[{}]}')

    llm._client.chat.side_effect = chat
    claim.image.model_id = "fixture"
    result = await service.process_claim(claim)
    if save_succeeds:
        assert events == ["checkpoint_committed", "classification_dispatched"]
        outcome = db.finish_attempt.call_args.args[1]
        assert outcome.envelope.receipt.total == Decimal("4.00")
        assert outcome.envelope.receipt.line_items[0].spending_category == "milk"
        assert outcome.envelope.classification_enrichment.state == "unavailable"
        assert len(outcome.persisted_views) == 2
    else:
        assert events == []
        assert result.stale
        db.finish_attempt.assert_not_awaited()


def managed_pipeline(monkeypatch, tmp_path):
    llm = observation_provider()
    enable_managed_ocr(monkeypatch)
    install_owned_bindings(monkeypatch)
    ocr = sidecar()
    ocr.managed_lifecycle = True
    ocr.claim, ocr.park = AsyncMock(), AsyncMock()
    settings = OcrSettings(enabled=True, coordination_path=tmp_path / "handoff")
    return llm, ExtractionPipeline(llm, ocr, settings)


@pytest.mark.parametrize("release_succeeds", [True, False])
async def test_classification_local_timeout_requires_settlement_and_verified_release(
    monkeypatch, tmp_path, release_succeeds,
):
    llm, extract = managed_pipeline(monkeypatch, tmp_path)
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    entered = asyncio.Event()
    finish_response = asyncio.Event()
    events = []
    owners = []
    resident = False
    formatted = llm._client.chat.return_value

    async def checkpoint(observed):
        assert observed.input_views[0].photo_index == 0
        assert observed.receipt.total == Decimal("4.00")
        owners.append(current_owner())
        events.append("checkpoint_committed")

    async def chat(**request):
        nonlocal resident
        resident = True
        if request["messages"][0].get("images"):
            return formatted
        assert events == ["checkpoint_committed"]
        events.append("classification_dispatched")
        entered.set()
        await finish_response.wait()
        events.append("classification_settled")
        return chat_reply('{}')

    async def release(**request):
        nonlocal resident
        assert request == {"model": "fixture", "keep_alive": 0}
        assert events[-1] == "classification_settled"
        resident = not release_succeeds
        events.append("release_requested")

    async def residency():
        if "release_requested" in events:
            events.append("release_verified")
        return ProcessResponse(models=[{"model": "fixture:latest"}] if resident else [])

    llm._client.chat.side_effect = chat
    llm._client.generate = AsyncMock(side_effect=release)
    llm._client.ps.side_effect = residency
    deadline = monotonic() + 60.5

    async def finish_after_classification_deadline():
        await entered.wait()
        await asyncio.sleep(max(0, deadline - 60 - monotonic()) + 0.03)
        finish_response.set()

    async with asyncio.timeout(4), asyncio.TaskGroup() as tasks:
        responder = tasks.create_task(finish_after_classification_deadline())
        try:
            if release_succeeds:
                result = await extract.extract(
                    "fixture", [path], options=ExperimentOptions(pipeline="direct_vision"),
                    deadline=deadline, tags=[], observation_checkpoint=checkpoint,
                )
                assert result.classification_enrichment.state == "unavailable"
                assert result.receipt.total == Decimal("4.00")
            else:
                with pytest.raises(RuntimeError, match="model_release_unconfirmed"):
                    await extract.extract(
                        "fixture", [path], options=ExperimentOptions(pipeline="direct_vision"),
                        deadline=deadline, tags=[], observation_checkpoint=checkpoint,
                    )
        finally:
            responder.cancel()
    assert events == [
        "checkpoint_committed", "classification_dispatched", "classification_settled",
        "release_requested", "release_verified",
    ]
    assert owners[0] is not None and owners[0].request_sequence == 3


@pytest.mark.parametrize("failure", ["proxy", "transport_timeout", "cancel"])
async def test_ambiguous_classification_keeps_durable_fence(
    monkeypatch, tmp_path, failure,
):
    llm, extract = managed_pipeline(monkeypatch, tmp_path)
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    formatted = llm._client.chat.return_value
    entered = asyncio.Event()
    fail_response = asyncio.Event()
    checkpoints = []
    completed = []
    request = httpx.Request("POST", "http://local/api/chat")
    proxy_error = httpx.HTTPStatusError(
        "ambiguous proxy response", request=request, response=httpx.Response(503, request=request),
    )
    error = proxy_error if failure == "proxy" else httpx.ReadTimeout("in flight", request=request)

    async def checkpoint(observed):
        checkpoints.append(observed)

    async def chat(**request):
        if request["messages"][0].get("images"):
            return formatted
        assert checkpoints[0].classification_enrichment.state == "pending"
        entered.set()
        if failure == "cancel":
            await fail_response.wait()
        raise error

    async def run():
        completed.append(await extract.extract(
            "fixture", [path], options=ExperimentOptions(pipeline="direct_vision"),
            deadline=monotonic() + 90, tags=[], observation_checkpoint=checkpoint,
        ))

    llm._client.chat.side_effect = chat
    if failure == "cancel":
        task = asyncio.create_task(run())
        try:
            async with asyncio.timeout(3):
                await entered.wait()
                task.cancel()
                await asyncio.sleep(0)
                fail_response.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    else:
        with pytest.raises(type(error)):
            await run()
    journal = extract.coordinator().read_journal()
    assert journal.unsettled and journal.models == ["fixture"]
    assert len(checkpoints) == 1
    assert completed == []


async def test_benchmark_failure_metadata_retains_admitted_observations(settings, tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    llm = observation_provider()
    llm._client.chat.side_effect = [
        llm._client.chat.return_value, httpx.ReadTimeout("ambiguous classification"),
    ]
    service = ReceiptService(settings.images, settings.pg, llm)
    with pytest.raises(ExperimentFailure) as raised:
        await execute_submission(service, "fixture", [path], ExperimentOptions(), [])
    assert raised.value.metadata["draft"]["total"] == "4.00"
    assert raised.value.metadata["classification_enrichment"]["state"] == "pending"


async def test_short_work_budget_skips_label_dispatch(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 5)).save(path)
    llm = observation_provider()
    result = await pipeline(llm).extract(
        "fixture", [path], options=ExperimentOptions(),
        deadline=monotonic() + 2, tags=[],
    )
    assert llm._client.chat.await_count == 1
    assert result.receipt.total == Decimal("4.00")
    assert result.classification_enrichment.state == "skipped"
    assert result.classification_enrichment.reason_code == "classification_budget_exhausted"
