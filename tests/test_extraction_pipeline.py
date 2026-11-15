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
        with Image.open(transported) as decoded:
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
