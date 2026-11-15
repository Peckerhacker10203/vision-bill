import hashlib
import threading
from io import BytesIO
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_ocr_worker_lifecycle import FakeContext, manager


def png():
    output = BytesIO()
    Image.new("RGB", (2, 2)).save(output, "PNG")
    return output.getvalue()


def fields(raw):
    return {
        "request_id": str(uuid4()),
        "photo_index": "0",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "timeout_ms": "500",
    }


def test_feature_requires_lifecycle_api():
    import importlib.util

    assert importlib.util.find_spec("ocr_service.api") is not None, "lifecycle API missing"


def test_health_does_not_imply_readiness_and_metadata_is_safe():
    from ocr_service.api import create_app

    worker = manager(FakeContext(init_fail=True))
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["reason_code"] == "ocr_init_failed"
        info = client.get("/info").json()
        assert info["contract_version"] == 2
        assert "model_dirs" not in str(info)
        assert "exception" not in str(info)


def test_valid_png_returns_contract_v2_and_cleans_file():
    from ocr_service.api import create_app

    context = FakeContext()
    worker = manager(context)
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        raw = png()
        data = fields(raw)
        result = client.post("/ocr", files={"image": ("receipt.png", raw, "image/png")}, data=data)
        assert result.status_code == 200, result.text
        assert result.json()["contract_version"] == 2
        assert result.json()["request_id"] == data["request_id"]
        assert not context.requests[0].original_path.exists()


@pytest.mark.parametrize(
    "case,status",
    [("hash", 422), ("bytes", 415), ("jpeg", 415), ("oversize", 413), ("timeout", 422)],
)
def test_malformed_or_oversized_payload_rejected_before_inference(case, status):
    from ocr_service.api import create_app

    context = FakeContext()
    worker = manager(context)
    with TestClient(create_app(worker_factory=lambda: worker, max_upload_bytes=100)) as client:
        raw = png()
        data = fields(raw)
        media_type = "image/png"
        if case == "hash":
            data["source_sha256"] = "0" * 64
        if case == "bytes":
            raw = b"not png"
            data = fields(raw)
        if case == "jpeg":
            media_type = "image/jpeg"
        if case == "oversize":
            raw = b"x" * 101
        if case == "timeout":
            data["timeout_ms"] = "600001"
        result = client.post("/ocr", files={"image": ("x.png", raw, media_type)}, data=data)
        assert result.status_code == status, result.text
        assert not context.requests


def test_timeout_terminates_before_http_removes_input():
    from ocr_service.api import create_app

    context = FakeContext("hang")
    worker = manager(context)
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        raw = png()
        data = fields(raw) | {"timeout_ms": "25"}
        result = client.post("/ocr", files={"image": ("x.png", raw, "image/png")}, data=data)
        assert result.status_code == 504
        assert result.json()["detail"] == "ocr_deadline_exceeded"
        assert not context.requests[0].original_path.exists()


def test_shutdown_does_not_block_event_loop():
    import asyncio

    from ocr_service.api import create_app

    from vision_bill.model.ocr import WorkerReadiness

    entered, release = threading.Event(), threading.Event()

    class SlowStop:
        def start(self):
            pass

        def stop(self):
            entered.set()
            assert release.wait(1)

        def readiness(self):
            return WorkerReadiness(ready=False, generation=1, reason_code="ocr_stopped")

    async def scenario():
        app = create_app(worker_factory=SlowStop)

        async def heartbeat():
            await asyncio.to_thread(entered.wait, 1)
            release.set()

        task = asyncio.create_task(heartbeat())
        async with app.router.lifespan_context(app):
            pass
        await task

    asyncio.run(scenario())


def test_shutdown_during_delayed_factory_never_starts_worker():
    import asyncio

    from ocr_service.api import create_app

    entered, release = threading.Event(), threading.Event()
    events = []

    class Worker:
        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    def factory():
        entered.set()
        assert release.wait(1)
        return Worker()

    async def scenario():
        app = create_app(worker_factory=factory)
        async with app.router.lifespan_context(app):
            assert await asyncio.to_thread(entered.wait, 1)
            asyncio.get_running_loop().call_soon(release.set)

    asyncio.run(scenario())
    assert "start" not in events
    assert "stop" in events


def test_busy_worker_keeps_http_health_and_ready_responsive():
    from concurrent.futures import ThreadPoolExecutor

    from ocr_service.api import create_app

    context = FakeContext("hang")
    worker = manager(context)
    with TestClient(create_app(worker_factory=lambda: worker)) as client:
        raw = png()
        with ThreadPoolExecutor(1) as pool:
            job = pool.submit(
                client.post, "/ocr", files={"image": ("x.png", raw, "image/png")}, data=fields(raw)
            )
            assert context.entered.wait(1)
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 200
            context.release.set()
            assert job.result(1).status_code == 200


@pytest.mark.parametrize("change", ["hash", "size", "base64"])
def test_working_png_integrity_must_match_wire_metadata(change, tmp_path):
    from fastapi import HTTPException
    from ocr_service.api import validate_result
    from test_ocr_worker_lifecycle import request, response

    result = response(request(tmp_path))
    if change == "hash":
        result.working_image.sha256 = "0" * 64
    elif change == "size":
        result.working_image.width = 3
    else:
        result.working_image.png_base64 = "invalid-base64"
    with pytest.raises(HTTPException) as error:
        validate_result(result)
    assert error.value.status_code == 502
    assert error.value.detail == "ocr_invalid_result"


def test_explicit_runtime_startup_inputs_are_consumed(tmp_path, monkeypatch):
    from ocr_service.api import configured_worker
    from ocr_service.profile import RuntimeSettings

    runtime = RuntimeSettings(
        accelerator="rocm",
        gpu_index=0,
        model_dirs={},
        cache_dir=tmp_path,
        qualification_manifest=tmp_path / "sealed.json",
        expected_packages={},
        expected_paddle_commit="abcdef1",
        image_identity="test-image",
    )
    settings = tmp_path / "runtime.json"
    settings.write_text(runtime.model_dump_json())
    monkeypatch.setenv("OCR_RUNTIME_SETTINGS_PATH", str(settings))
    monkeypatch.setenv("OCR_QUALIFICATION_IMAGE_PATH", str(tmp_path / "smoke.png"))
    monkeypatch.setenv("OCR_STARTUP_TIMEOUT_SECONDS", "17")
    worker = configured_worker()
    assert worker._runtime == runtime
    assert worker._startup_timeout == 17
    assert worker._smoke_path == tmp_path / "smoke.png"
    worker.stop()
