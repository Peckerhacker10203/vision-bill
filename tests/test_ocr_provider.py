"""Typed sidecar transport contract, exercised through real HTTP serialization."""

import base64
import hashlib
import time
from io import BytesIO
from uuid import uuid4

import httpx
import pytest
from PIL import Image

from vision_bill.config import OcrSettings
from vision_bill.provider.factory import get_ocr_provider
from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider


def image_file(tmp_path, fmt="PNG"):
    path = tmp_path / "misleading.jpg"
    Image.new("RGB", (3, 4), "white").save(path, format=fmt)
    return path


def wire_page(index=0, digest="a" * 64, request_id=None, text="Milk 2.00"):
    data = BytesIO()
    Image.new("RGB", (3, 4)).save(data, format="PNG")
    raw = data.getvalue()
    return {
        "contract_version": 2,
        "request_id": str(request_id or uuid4()),
        "photo_index": index,
        "source_sha256": digest,
        "text": text,
        "blocks": [],
        "working_image": {
            "view_id": str(uuid4()),
            "width": 3,
            "height": 4,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "png_base64": base64.b64encode(raw).decode(),
        },
        "profile_id": "fixed-profile",
        "runtime_fingerprint": "runtime",
        "telemetry": {},
    }


def http_client(monkeypatch, handler):
    original = httpx.AsyncClient
    calls = []

    def make(**kwargs):
        calls.append(kwargs)
        return original(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", make)
    return calls


def test_disabled_ocr_constructs_no_client():
    assert get_ocr_provider(OcrSettings()) is None


async def test_readiness_requires_info_and_ready(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request.url.path)
        if request.url.path == "/info":
            return httpx.Response(
                200, json={"contract_version": 2, "profile": {"profile_id": "fixed-profile"}}
            )
        return httpx.Response(
            503,
            json={
                "ready": False,
                "generation": 1,
                "profile_id": "fixed-profile",
                "reason_code": "ocr_starting",
            },
        )

    calls = http_client(monkeypatch, handler)
    provider = PaddleOcrProvider("http://local", managed_lifecycle=False)
    assert hasattr(provider, "get_capabilities"), "typed readiness is missing"
    capabilities = await provider.get_capabilities()
    assert not capabilities.ready and capabilities.reachable
    assert capabilities.reason_code == "ocr_starting"
    assert requests == ["/info", "/ready"]
    assert all(call["trust_env"] is False for call in calls)


async def test_original_png_bytes_mime_hash_and_remaining_deadline(monkeypatch, tmp_path):
    path = image_file(tmp_path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    def handler(request):
        body = request.content
        assert b"Content-Type: image/png" in body
        assert raw in body and digest.encode() in body
        import re

        request_id = re.search(rb'name="request_id"\r\n\r\n([^\r]+)', body)[1].decode()
        timeout = int(re.search(rb'name="timeout_ms"\r\n\r\n([^\r]+)', body)[1])
        assert 0 < timeout <= 2000
        return httpx.Response(200, json=wire_page(digest=digest, request_id=request_id))

    calls = http_client(monkeypatch, handler)
    provider = PaddleOcrProvider("http://local", managed_lifecycle=False)
    assert hasattr(provider, "extract_page"), "typed page extraction is missing"
    page = await provider.extract_page(
        path,
        photo_index=0,
        source_sha256=digest,
        deadline=time.monotonic() + 2,
        expected_profile_id="fixed-profile",
    )
    assert page.source_sha256 == digest
    assert calls[0]["trust_env"] is False


@pytest.mark.parametrize(
    "change", ["photo_index", "source_sha256", "profile_id", "request_id", "text", "png_base64"]
)
async def test_rejects_malformed_or_reordered_result(monkeypatch, tmp_path, change):
    path = image_file(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def handler(request):
        import re

        request_id = re.search(rb'name="request_id"\r\n\r\n([^\r]+)', request.content)[1].decode()
        page = wire_page(digest=digest, request_id=request_id)
        if change == "png_base64":
            page["working_image"][change] = "not png"
        else:
            page[change] = {
                "photo_index": 1,
                "source_sha256": "b" * 64,
                "profile_id": "other",
                "request_id": str(uuid4()),
                "text": " ",
            }[change]
        return httpx.Response(200, json=page)

    http_client(monkeypatch, handler)
    provider = PaddleOcrProvider("http://local", managed_lifecycle=False)
    assert hasattr(provider, "extract_page"), "typed page extraction is missing"
    with pytest.raises(ValueError):
        await provider.extract_page(
            path,
            photo_index=0,
            source_sha256=digest,
            deadline=time.monotonic() + 2,
            expected_profile_id="fixed-profile",
        )


async def test_paddleocr_extract_text_http_failure(monkeypatch, tmp_path):
    path = image_file(tmp_path)
    http_client(monkeypatch, lambda request: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        await PaddleOcrProvider("http://local", managed_lifecycle=False).extract_text(path)


@pytest.mark.parametrize("body", [[], {"contract_version": 2, "profile": {}, "worker": "bad"}])
async def test_malformed_info_is_unready_instead_of_crashing(monkeypatch, body):
    def handler(request):
        return httpx.Response(
            200,
            json=body
            if request.url.path == "/info"
            else {"ready": True, "generation": 1, "profile_id": "profile"},
        )

    http_client(monkeypatch, handler)
    result = await PaddleOcrProvider("http://local", managed_lifecycle=False).get_capabilities()
    assert not result.ready and result.reason_code == "ocr_unavailable"


async def test_ocr_limit_comes_from_matching_ready_profile(monkeypatch):
    def handler(request):
        if request.url.path == "/info":
            return httpx.Response(
                200,
                json={
                    "contract_version": 2,
                    "profile": {"max_new_tokens": 4096},
                    "worker": {"profile_id": "fixed-profile"},
                },
            )
        return httpx.Response(
            200, json={"ready": True, "generation": 1, "profile_id": "fixed-profile"}
        )

    http_client(monkeypatch, handler)
    capabilities = await PaddleOcrProvider("http://local", managed_lifecycle=False).get_capabilities()
    assert capabilities.ready
    assert getattr(capabilities, "ocr_max_new_tokens", None) == 4096


async def test_original_byte_preparation_is_bounded_before_http(monkeypatch, tmp_path):
    import asyncio
    import threading
    from pathlib import Path

    path = image_file(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original = Path.read_bytes
    gate, entered, finished = threading.Event(), threading.Event(), threading.Event()

    def read(selected):
        entered.set()
        gate.wait(0.5)
        try:
            return original(selected)
        finally:
            finished.set()

    calls = http_client(monkeypatch, lambda request: httpx.Response(500))
    monkeypatch.setattr(Path, "read_bytes", read)
    timer = threading.Timer(0.5, gate.set)
    timer.start()
    began = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await PaddleOcrProvider("http://local", managed_lifecycle=False).extract_page(
                path, photo_index=0, source_sha256=digest, deadline=began + 0.08
            )
        assert entered.is_set()
        assert time.monotonic() - began < 0.3
    finally:
        gate.set()
        timer.cancel()
    assert await asyncio.to_thread(finished.wait, 0.5)
    assert calls == []


async def test_capabilities_exposes_qualified_live_runtime(monkeypatch):
    def handler(request):
        ready = {
            "ready": True,
            "generation": 2,
            "profile_id": "fixed-profile",
            "runtime_fingerprint": "qualified-runtime",
        }
        return httpx.Response(
            200,
            json=(
                {"contract_version": 2, "profile": {}, "worker": ready}
                if request.url.path == "/info"
                else ready
            ),
        )

    http_client(monkeypatch, handler)
    assert (
        await PaddleOcrProvider("http://local", managed_lifecycle=False).get_capabilities()
    ).runtime_fingerprint == "qualified-runtime"
