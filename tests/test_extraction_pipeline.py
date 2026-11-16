"""All-photo extraction policy and source-aware formatter integration."""

import asyncio
import hashlib
import importlib.util
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ollama import ChatResponse
from PIL import Image
from test_experiment_scoring import receipt
from test_ocr_provider import wire_page

from vision_bill.config import OcrSettings
from vision_bill.model.experiment import ExperimentOptions
from vision_bill.model.ocr import OcrCapabilities, OcrResponse
from vision_bill.provider.llm.base import ModelInfo
from vision_bill.provider.llm.ollama import OllamaProvider


@pytest.fixture
def photos(tmp_path):
    paths = [tmp_path / f"{i}.png" for i in range(2)]
    for i, path in enumerate(paths):
        Image.new("RGB", (4, 5), (i, 0, 0)).save(path)
    return paths


def formatter(vision=True, payload=None):
    provider = OllamaProvider("http://local")
    model = ModelInfo("fixture")
    model.capabilities = ["completion", "vision"] if vision else ["completion"]
    provider.get_available_models = AsyncMock(return_value=[model])
    payload = payload or receipt().model_dump(mode="json")
    for row in payload["line_items"]:
        row.setdefault("source_image", 0)
    payload.setdefault("duplicate_candidates", [])
    provider._client.chat = AsyncMock(
        return_value=ChatResponse(
            model="actual-model",
            done_reason="stop",
            eval_count=42,
            message={"role": "assistant", "content": json.dumps(payload)},
        )
    )
    return provider


def pipeline(provider, ocr=None, **settings):
    assert importlib.util.find_spec("vision_bill.service.extraction_pipeline"), (
        "shared pipeline missing"
    )
    from vision_bill.service.extraction_pipeline import ExtractionPipeline

    return ExtractionPipeline(provider, ocr, OcrSettings(enabled=ocr is not None, **settings))


def sidecar(fail_at=None, blank=False):
    async def page(path, *, photo_index, source_sha256, deadline, expected_profile_id=None):
        if photo_index == fail_at:
            if blank:
                return OcrResponse.model_validate(wire_page(photo_index, source_sha256, text=" "))
            raise ValueError("sensitive receipt content must not be exposed")
        return OcrResponse.model_validate(wire_page(photo_index, source_sha256))

    return SimpleNamespace(
        get_capabilities=AsyncMock(
            return_value=OcrCapabilities(
                enabled=True,
                configured=True,
                reachable=True,
                ready=True,
                profile_id="fixed-profile",
            )
        ),
        extract_page=AsyncMock(side_effect=page),
    )


async def test_explicit_direct_never_calls_ocr(photos):
    ocr, llm = sidecar(), formatter()
    result = await pipeline(llm, ocr).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="direct_vision", include_classification=False),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.receipt is not None
    ocr.extract_page.assert_not_awaited()
    ocr.get_capabilities.assert_not_awaited()
    assert result.provenance.source_image_count == 2


@pytest.mark.parametrize("blank", [False, True])
async def test_second_photo_failure_is_not_a_complete_prefix(photos, blank):
    ocr, llm = sidecar(1, blank), formatter()
    checkpoint = AsyncMock()
    result = await pipeline(llm, ocr, required=True, allow_vision_fallback=True).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="ocr", allow_ocr_fallback=True),
        deadline=time.monotonic() + 2,
        tags=[],
        checkpoint=checkpoint,
    )
    assert result.receipt is None
    assert result.provenance.source_image_count == 2
    assert len(result.ocr_pages) == 1
    checkpoint.assert_awaited_once()
    llm._client.chat.assert_not_awaited()
    assert not result.provenance.fallback_used


async def test_optional_failure_falls_back_for_whole_receipt(photos):
    llm = formatter()
    result = await pipeline(llm, sidecar(1), allow_vision_fallback=True).extract(
        "fixture",
        photos,
        options=ExperimentOptions(
            pipeline="ocr", allow_ocr_fallback=True, include_classification=False
        ),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.receipt is not None
    assert result.provenance.fallback_used
    assert result.provenance.fallback_reason == "ocr_page_failed"
    assert len(llm._client.chat.call_args.kwargs["messages"][0]["images"]) == 2


async def test_text_only_ocr_formats_all_marked_photos_without_png(photos):
    llm = formatter(False)
    result = await pipeline(llm, sidecar()).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="ocr", include_classification=False),
        deadline=time.monotonic() + 2,
        tags=["private_tag"],
    )
    assert result.receipt is not None
    message = llm._client.chat.call_args.kwargs["messages"][0]
    prompt = message["content"]
    assert "PHOTO 0" in prompt and "PHOTO 1" in prompt
    assert "png_base64" not in prompt and "private_tag" not in prompt
    assert "images" not in message
    assert len(result.ocr_pages) == 2
    assert result.provenance.model_id == "fixture"
    assert result.telemetry.calls[0]["actual_model"] == "actual-model"
    assert result.telemetry.output_tokens == 42


async def test_text_only_model_never_receives_fallback_images(photos):
    llm = formatter(False)
    result = await pipeline(llm, sidecar(1), allow_vision_fallback=True).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="ocr", allow_ocr_fallback=True),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.receipt is None
    assert result.provenance.fallback_reason == "vision_fallback_unavailable"
    llm._client.chat.assert_not_awaited()


async def test_checkpoint_failure_is_not_swallowed_or_fallback(photos):
    llm = formatter()
    checkpoint = AsyncMock(side_effect=OSError("storage failed"))
    with pytest.raises(OSError, match="storage failed"):
        await pipeline(llm, sidecar(), allow_vision_fallback=True).extract(
            "fixture",
            photos,
            options=ExperimentOptions(pipeline="ocr", allow_ocr_fallback=True),
            deadline=time.monotonic() + 2,
            tags=[],
            checkpoint=checkpoint,
        )
    llm._client.chat.assert_not_awaited()


async def test_one_deadline_covers_ocr_and_formatter(photos):
    llm, ocr = formatter(), sidecar()

    async def slow(*args, **kwargs):
        await asyncio.sleep(0.04)
        return await sidecar().extract_page(*args, **kwargs)

    ocr.extract_page.side_effect = slow
    with pytest.raises(TimeoutError):
        await pipeline(llm, ocr).extract(
            "fixture",
            photos,
            options=ExperimentOptions(pipeline="ocr"),
            deadline=time.monotonic() + 0.02,
            tags=[],
        )
    llm._client.chat.assert_not_awaited()


async def test_real_gif_direct_uses_rgb_png_and_preserves_original_provenance(tmp_path):
    path = tmp_path / "fake.png"
    Image.new("RGB", (3, 5), "red").save(
        path, format="GIF", save_all=True, append_images=[Image.new("RGB", (3, 5), "blue")]
    )
    original = path.read_bytes()
    llm = formatter()

    async def inspect(**kwargs):
        transported = kwargs["messages"][0]["images"][0]
        from io import BytesIO

        assert isinstance(transported, bytes)
        with Image.open(BytesIO(transported)) as decoded:
            assert decoded.format == "PNG" and decoded.mode == "RGB"
        return ChatResponse(message={"role": "assistant", "content": receipt().model_dump_json()})

    llm._client.chat.side_effect = inspect
    result = await pipeline(llm).extract(
        "fixture",
        [path],
        options=ExperimentOptions(pipeline="direct_vision", include_classification=False),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert path.read_bytes() == original
    assert result.provenance.ordered_source_sha256 == [hashlib.sha256(original).hexdigest()]
    assert result.field_sources["input"] == []
    assert result.receipt is not None
    assert result.input_views[0].working_image.transforms[0]["frame_count"] == 2
    assert result.input_views[0].source_sha256 == hashlib.sha256(original).hexdigest()


async def test_ocr_discovery_accepts_text_model_and_records_known_context(photos):
    llm = formatter(False)
    llm.get_available_models.return_value[0].context_window_tokens = 32768
    llm.get_available_models.return_value[0].digest = "model-digest"
    result = await pipeline(llm, sidecar()).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="ocr", include_classification=False),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    llm.get_available_models.assert_awaited_once_with(capability="text")
    assert result.telemetry.context_window_tokens == 32768
    assert result.provenance.model_digest == "model-digest"


async def test_explicit_vision_fallback_model_is_used_for_text_formatter(photos):
    llm = formatter(False)
    vision = ModelInfo("vision-fallback")
    vision.capabilities = ["vision"]
    llm.get_available_models.return_value.append(vision)
    result = await pipeline(
        llm, sidecar(1), allow_vision_fallback=True, vision_fallback_model_id="vision-fallback"
    ).extract(
        "fixture",
        photos,
        options=ExperimentOptions(
            pipeline="ocr", allow_ocr_fallback=True, include_classification=False
        ),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.provenance.model_id == "fixture"
    assert llm._client.chat.call_args.kwargs["model"] == "vision-fallback"
    assert result.telemetry.calls[0]["requested_model"] == "vision-fallback"
    assert len(result.input_views) == 2


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "GIF", "HEIF"])
async def test_ocr_transport_keeps_every_original_format(monkeypatch, tmp_path, fmt):
    import re

    import httpx
    from pillow_heif import register_heif_opener
    from test_ocr_provider import http_client

    from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider

    register_heif_opener()
    path = tmp_path / "misleading.dat"
    Image.new("RGB", (5, 7), "red").save(path, format=fmt)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp", "GIF": "gif", "HEIF": "heif"}[fmt]

    def handler(request):
        assert raw in request.content
        assert f"Content-Type: image/{mime}".encode() in request.content
        request_id = re.search(rb'name="request_id"\r\n\r\n([^\r]+)', request.content)[1].decode()
        return httpx.Response(200, json=wire_page(digest=digest, request_id=request_id))

    http_client(monkeypatch, handler)
    result = await PaddleOcrProvider("http://local").extract_page(
        path, photo_index=0, source_sha256=digest, deadline=time.monotonic() + 2
    )
    assert result.source_sha256 == digest and path.read_bytes() == raw


async def test_fallback_preserves_ocr_failure_as_stable_finding(photos):
    result = await pipeline(formatter(), sidecar(1), allow_vision_fallback=True).extract(
        "fixture",
        photos,
        options=ExperimentOptions(
            pipeline="ocr", allow_ocr_fallback=True, include_classification=False
        ),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.receipt is not None
    assert any(f.code == "ocr_page_failed" for f in result.findings)


async def test_ocr_and_formatter_limits_remain_separate(photos):
    from vision_bill.model.ocr import OcrCapabilities

    ocr = sidecar()
    assert "ocr_max_new_tokens" in OcrCapabilities.model_fields, "typed OCR limit missing"
    ocr.get_capabilities.return_value = OcrCapabilities(
        enabled=True,
        configured=True,
        reachable=True,
        ready=True,
        profile_id="fixed-profile",
        ocr_max_new_tokens=4096,
    )
    result = await pipeline(formatter(False), ocr).extract(
        "fixture",
        photos,
        options=ExperimentOptions(pipeline="ocr", include_classification=False),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.telemetry.ocr_max_new_tokens == 4096
    assert result.telemetry.max_output_tokens == 8192
    assert result.telemetry.context_window_tokens is None


@pytest.mark.parametrize("fallback", [False, True])
async def test_configured_standard_openai_vision_model_reaches_inference(
    settings, photos, fallback
):
    from vision_bill.config import LLMProviderEnum
    from vision_bill.provider.factory import get_llm_provider

    declared = settings.llm.model_copy(update={"provider": LLMProviderEnum.OPENAI})
    # Assignment must be a supported settings field, not an ignored extra.
    assert "vision_model_ids" in type(declared).model_fields
    declared.vision_model_ids = ["configured-vision"]
    llm = get_llm_provider(declared)
    llm._client.models.with_raw_response.list = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(
                content=json.dumps(
                    {"data": [{"id": "configured-vision", "object": "model"}]}
                ).encode()
            )
        )
    )
    payload = receipt().model_dump(mode="json")
    for row in payload["line_items"]:
        row["source_image"] = 0
    payload["duplicate_candidates"] = []
    llm._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            model="actual-vision",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload))
                )
            ],
        )
    )
    result = await pipeline(
        llm, sidecar(1) if fallback else None, allow_vision_fallback=fallback
    ).extract(
        "configured-vision",
        photos,
        options=ExperimentOptions(
            pipeline="ocr" if fallback else "direct_vision",
            allow_ocr_fallback=fallback,
            include_classification=False,
        ),
        deadline=time.monotonic() + 2,
        tags=[],
    )
    assert result.receipt is not None
    assert result.provenance.fallback_used is fallback
    assert len(llm._client.chat.completions.create.call_args.kwargs["messages"][0]["content"]) == 3


@pytest.mark.parametrize("stage", ["hash", "normalization"])
async def test_blocked_preparation_obeys_deadline_and_cleans_owned_files(
    monkeypatch, tmp_path, photos, stage
):
    import tempfile
    import threading
    from pathlib import Path

    from PIL import ImageOps

    gate, entered, cleaned = threading.Event(), threading.Event(), threading.Event()
    timer = threading.Timer(0.5, gate.set)
    original_read, original_transpose = Path.read_bytes, ImageOps.exif_transpose
    cleanup = tempfile.TemporaryDirectory.cleanup

    def cleanup_observed(instance):
        cleanup(instance)
        cleaned.set()

    def read(path):
        if stage == "hash" and path == photos[0]:
            entered.set()
            gate.wait(0.5)
        return original_read(path)

    def transpose(image, *args, **kwargs):
        entered.set()
        gate.wait(0.5)
        return original_transpose(image, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read)
    if stage == "normalization":
        monkeypatch.setattr(ImageOps, "exif_transpose", transpose)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", cleanup_observed)
    llm = formatter()
    timer.start()
    began = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await pipeline(llm).extract(
                "fixture",
                photos,
                options=ExperimentOptions(pipeline="direct_vision", include_classification=False),
                deadline=began + 0.08,
                tags=[],
            )
        assert entered.is_set()
        assert time.monotonic() - began < 0.3
        llm._client.chat.assert_not_awaited()
    finally:
        gate.set()
        timer.cancel()
    if stage == "normalization":
        assert await asyncio.to_thread(cleaned.wait, 0.5)
        assert not list(tmp_path.glob("vision-bill-vision-*"))
    assert llm.experiment_trace is None


@pytest.mark.parametrize("late_stage", ["repair", "enrichment"])
async def test_budget_exhausted_after_all_ocr_during_late_call(photos, late_stage):
    llm, ocr = formatter(False), sidecar()
    first = llm._client.chat.return_value
    if late_stage == "repair":
        first = ChatResponse(
            message={"role": "assistant", "content": '{"merchant_name":"partial"}'}
        )
    entered = asyncio.Event()

    async def late(**kwargs):
        if llm._client.chat.await_count == 1:
            return first
        entered.set()
        await asyncio.Event().wait()

    llm._client.chat.side_effect = late
    checkpoint = AsyncMock()
    began = time.monotonic()
    with pytest.raises(TimeoutError):
        await pipeline(llm, ocr).extract(
            "fixture",
            photos,
            options=ExperimentOptions(
                pipeline="ocr", include_classification=late_stage == "enrichment"
            ),
            deadline=began + 0.15,
            tags=["food"],
            checkpoint=checkpoint,
        )
    assert checkpoint.await_count == 2
    assert entered.is_set() and llm._client.chat.await_count == 2
    assert time.monotonic() - began < 0.5


async def test_ollama_serializes_exact_prepared_png_after_temporary_cleanup(
    monkeypatch, tmp_path, photos
):
    import base64
    import tempfile
    from io import BytesIO

    import httpx

    transported = []
    payload = receipt().model_dump(mode="json")
    for row in payload["line_items"]:
        row["source_image"] = 0
    payload["duplicate_candidates"] = []

    def handler(request):
        assert not list(tmp_path.glob("vision-bill-vision-*"))
        body = json.loads(request.content)
        transported.extend(base64.b64decode(data) for data in body["messages"][0]["images"])
        for png in transported:
            with Image.open(BytesIO(png)) as image:
                assert image.format == "PNG" and image.mode == "RGB"
        return httpx.Response(
            200,
            json={
                "model": "actual",
                "message": {"role": "assistant", "content": json.dumps(payload)},
                "done": True,
            },
        )

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    llm = OllamaProvider("http://local")
    model = ModelInfo("fixture")
    model.capabilities = ["vision"]
    llm.get_available_models = AsyncMock(return_value=[model])
    async with httpx.AsyncClient(
        base_url="http://local", transport=httpx.MockTransport(handler)
    ) as client:
        llm._client._client = client
        result = await pipeline(llm).extract(
            "fixture",
            photos,
            options=ExperimentOptions(pipeline="direct_vision", include_classification=False),
            deadline=time.monotonic() + 2,
            tags=[],
        )
    assert result.receipt is not None
    assert transported == [
        base64.b64decode(view.working_image.png_base64) for view in result.input_views
    ]
    assert [view.source_sha256 for view in result.input_views] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in photos
    ]
