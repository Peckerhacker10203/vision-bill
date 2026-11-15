"""Original-image HTTP ingress, validated working PNGs, and worker lifecycle."""

import asyncio
import base64
import binascii
import hashlib
import os
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from vision_bill.model.ocr import OcrResponse

from .profile import RuntimeSettings, load_profile
from .worker import OcrRequest, OcrWorkerError, OcrWorkerManager

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SOURCE_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/heic": "HEIF",
    "image/heif": "HEIF",
    "image/gif": "GIF",
}


def configured_worker() -> OcrWorkerManager:
    runtime_path = os.environ.get("OCR_RUNTIME_SETTINGS_PATH")
    smoke_path = os.environ.get("OCR_QUALIFICATION_IMAGE_PATH")
    if not runtime_path or not smoke_path:
        # Missing configuration is surfaced by the child's stable init failure.
        return OcrWorkerManager()
    return OcrWorkerManager(
        runtime=RuntimeSettings.model_validate_json(Path(runtime_path).read_bytes()),
        smoke_path=Path(smoke_path),
        startup_timeout=float(os.environ.get("OCR_STARTUP_TIMEOUT_SECONDS", "600")),
    )


def validate_png(raw: bytes, *, limit: int) -> None:
    if len(raw) > limit:
        raise HTTPException(413, "ocr_payload_too_large")
    try:
        with Image.open(BytesIO(raw)) as decoded:
            if decoded.format != "PNG" or decoded.width * decoded.height > MAX_IMAGE_PIXELS:
                raise ValueError("Invalid PNG dimensions or format")
            if getattr(decoded, "n_frames", 1) != 1:
                raise ValueError("Only deliberate single-frame PNG inputs are accepted")
            decoded.verify()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise HTTPException(415, "ocr_invalid_png") from None


def validate_source(raw: bytes, media_type: str | None, *, limit: int) -> None:
    if len(raw) > limit:
        raise HTTPException(413, "ocr_payload_too_large")
    if media_type not in SOURCE_FORMATS:
        raise HTTPException(415, "ocr_unsupported_media_type")
    from pillow_heif.as_plugin import register_heif_opener

    register_heif_opener()
    try:
        with Image.open(BytesIO(raw)) as decoded:
            if (
                decoded.format != SOURCE_FORMATS[media_type]
                or decoded.width * decoded.height > MAX_IMAGE_PIXELS
            ):
                raise ValueError("Invalid source format or dimensions")
            decoded.verify()
        with Image.open(BytesIO(raw)) as decoded:
            decoded.seek(0)
            decoded.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise HTTPException(415, "ocr_invalid_image") from None


def validate_result(result: OcrResponse) -> None:
    try:
        raw = base64.b64decode(result.working_image.png_base64, validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(502, "ocr_invalid_result") from None
    try:
        validate_png(raw, limit=MAX_UPLOAD_BYTES)
        view = result.working_image
        with Image.open(BytesIO(raw)) as decoded:
            if decoded.size != (view.width, view.height):
                raise ValueError("Result dimensions differ")
        if hashlib.sha256(raw).hexdigest() != view.sha256:
            raise ValueError("Result hash differs")
    except (HTTPException, ValueError):
        raise HTTPException(502, "ocr_invalid_result") from None


def create_app(
    *,
    worker_factory: Callable[[], OcrWorkerManager] = configured_worker,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> FastAPI:
    worker: OcrWorkerManager | None = None
    startup_failed = False
    closing = False

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal closing

        async def initialize() -> None:
            nonlocal worker, startup_failed
            try:
                worker = await asyncio.to_thread(worker_factory)
                if closing:
                    await asyncio.to_thread(worker.stop)
                    return
                await asyncio.to_thread(worker.start)
            except Exception:  # noqa: BLE001 - safe lifecycle failure boundary
                startup_failed = True

        initialization = asyncio.create_task(initialize())
        try:
            yield
        finally:
            closing = True
            # The factory only parses local settings; awaiting it prevents late
            # publication of an orphan manager. Native initialization observes stop.
            if worker is not None:
                await asyncio.to_thread(worker.stop)
            await initialization
            if worker is not None:
                await asyncio.to_thread(worker.stop)

    app = FastAPI(title="PaddleOCR-VL-1.6 Service", version="2.0.0", lifespan=lifespan)

    def ready_body() -> dict[str, object]:
        if worker is None:
            return {
                "ready": False,
                "generation": 1,
                "profile_id": None,
                "reason_code": "ocr_init_failed" if startup_failed else "ocr_starting",
            }
        return worker.readiness().model_dump()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        body = ready_body()
        return JSONResponse(body, status_code=200 if body["ready"] else 503)

    @app.get("/info")
    async def info() -> dict[str, object]:
        return {
            "contract_version": 2,
            "profile": load_profile(Path(__file__).parents[1] / "profile.json").model_dump(),
            "worker": ready_body(),
            "max_upload_bytes": max_upload_bytes,
            "max_pending": 4,
            "max_timeout_ms": 600000,
        }

    @app.post("/ocr", response_model=OcrResponse)
    def ocr(
        image: Annotated[UploadFile, File()],
        request_id: Annotated[UUID, Form()],
        photo_index: Annotated[int, Form(ge=0)],
        source_sha256: Annotated[str, Form(pattern=r"^[0-9a-f]{64}$")],
        timeout_ms: Annotated[int, Form(gt=0, le=600000)],
    ) -> OcrResponse:
        deadline = time.monotonic() + timeout_ms / 1000
        if worker is None:
            raise HTTPException(503, "ocr_init_failed" if startup_failed else "ocr_starting")
        raw = image.file.read(max_upload_bytes + 1)
        validate_source(raw, image.content_type, limit=max_upload_bytes)
        if hashlib.sha256(raw).hexdigest() != source_sha256:
            raise HTTPException(422, "ocr_source_hash_mismatch")
        with tempfile.NamedTemporaryFile(
            prefix="vision-bill-upload-", suffix=".source", delete=False
        ) as f:
            f.write(raw)
            path = Path(f.name)
        cleanup_safe = True
        try:
            result = worker.infer(
                OcrRequest(request_id, path, photo_index, source_sha256),
                deadline_monotonic=deadline,
            )
            validate_result(result)
            return result
        except OcrWorkerError as exc:
            cleanup_safe = exc.cleanup_safe
            raise HTTPException(exc.status_code, exc.code) from None
        finally:
            if cleanup_safe:
                path.unlink(missing_ok=True)

    return app
