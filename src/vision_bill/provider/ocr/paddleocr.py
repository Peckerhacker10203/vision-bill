"""Local v2 OCR transport. Original bytes and identities cross this boundary unchanged."""

import asyncio
import base64
import binascii
import hashlib
from io import BytesIO
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

import httpx
from PIL import Image

from ...helper.deadline import check_deadline, run_sync_before
from ...model.ocr import OcrCapabilities, OcrResponse, WorkerReadiness
from .base import OcrProvider, compact_ocr_text


class PaddleOcrProvider(OcrProvider):
    def __init__(self, service_url: str, timeout: int = 300):
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout

    async def get_capabilities(self) -> OcrCapabilities:
        reachable = False
        try:
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                info_response = await client.get(f"{self.service_url}/info")
                info_response.raise_for_status()
                reachable = True
                info = await asyncio.to_thread(info_response.json)
                if (
                    not isinstance(info, dict)
                    or info.get("contract_version") != 2
                    or not isinstance(info.get("profile"), dict)
                    or not isinstance(info.get("worker", {}), dict)
                ):
                    raise ValueError("ocr_contract_mismatch")
                response = await client.get(f"{self.service_url}/ready")
                if response.status_code not in (200, 503):
                    response.raise_for_status()
                ready = await asyncio.to_thread(
                    WorkerReadiness.model_validate_json, response.content
                )
                info_profile = info.get("worker", {}).get("profile_id")
                if ready.ready and (
                    response.status_code != 200
                    or not ready.profile_id
                    or (info_profile is not None and ready.profile_id != info_profile)
                ):
                    raise ValueError("ocr_profile_mismatch")
                reported_limit = info["profile"].get("max_new_tokens")
                ocr_limit = (
                    reported_limit
                    if ready.ready
                    and info_profile == ready.profile_id
                    and type(reported_limit) is int
                    and reported_limit > 0
                    else None
                )
                return OcrCapabilities(
                    enabled=True,
                    configured=True,
                    reachable=True,
                    ready=ready.ready,
                    profile_id=ready.profile_id,
                    reason_code=ready.reason_code,
                    runtime_fingerprint=ready.runtime_fingerprint if ready.ready else None,
                    ocr_max_new_tokens=ocr_limit,
                )
        except (httpx.HTTPError, OSError, ValueError):
            return OcrCapabilities(
                enabled=True,
                configured=True,
                reachable=reachable,
                ready=False,
                reason_code="ocr_unavailable",
            )

    async def extract_page(
        self,
        image: Path,
        *,
        photo_index: int,
        source_sha256: str,
        deadline: float,
        expected_profile_id: str | None = None,
    ) -> OcrResponse:
        raw, media_type = await run_sync_before(
            deadline, self._source_payload, image, source_sha256
        )
        remaining = min(float(self.timeout), deadline - monotonic())
        if remaining <= 0:
            raise TimeoutError("ocr_deadline")
        request_id = uuid4()
        async with asyncio.timeout(max(0, deadline - monotonic())):
            async with httpx.AsyncClient(timeout=remaining, trust_env=False) as client:
                response = await client.post(
                    f"{self.service_url}/ocr",
                    files={"image": (image.name, raw, media_type)},
                    data={
                        "request_id": str(request_id),
                        "photo_index": str(photo_index),
                        "source_sha256": source_sha256,
                        "timeout_ms": str(max(1, min(600000, int(remaining * 1000)))),
                    },
                )
                response.raise_for_status()
        check_deadline(deadline)
        return await run_sync_before(
            deadline,
            self._validated_response,
            response.content,
            request_id,
            photo_index,
            source_sha256,
            expected_profile_id,
        )

    @staticmethod
    def _source_payload(image: Path, source_sha256: str) -> tuple[bytes, str]:
        """Pure original-byte/MIME preparation; no caller objects are mutated."""
        raw = image.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source_sha256:
            raise ValueError("ocr_source_hash_mismatch")
        from pillow_heif.as_plugin import register_heif_opener

        register_heif_opener()
        with Image.open(BytesIO(raw)) as decoded:
            media_type = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
                "HEIF": "image/heif",
                "GIF": "image/gif",
            }.get(decoded.format or "")
        if media_type is None:
            raise ValueError("ocr_unsupported_media_type")
        return raw, media_type

    @staticmethod
    def _validated_response(
        content: bytes,
        request_id: object,
        photo_index: int,
        source_sha256: str,
        expected_profile_id: str | None,
    ) -> OcrResponse:
        """Decode/validate PNG and typed page in an isolated synchronous worker."""
        page = OcrResponse.model_validate_json(content)
        if (
            page.request_id != request_id
            or page.photo_index != photo_index
            or page.source_sha256 != source_sha256
            or (expected_profile_id is not None and page.profile_id != expected_profile_id)
        ):
            raise ValueError("ocr_identity_mismatch")
        compact_ocr_text({"text": page.text})
        try:
            png = base64.b64decode(page.working_image.png_base64, validate=True)
            with Image.open(BytesIO(png)) as working:
                if (
                    working.format != "PNG"
                    or working.size != (page.working_image.width, page.working_image.height)
                    or getattr(working, "n_frames", 1) != 1
                ):
                    raise ValueError("ocr_invalid_working_image")
                working.verify()
            if hashlib.sha256(png).hexdigest() != page.working_image.sha256:
                raise ValueError("ocr_invalid_working_image")
        except (OSError, binascii.Error, SyntaxError) as exc:
            raise ValueError("ocr_invalid_working_image") from exc
        return page

    async def extract_text(self, image: Path) -> dict[str, Any]:
        """Compatibility API; live sidecars still require a validated v2 response."""
        deadline = monotonic() + self.timeout
        digest = await run_sync_before(
            deadline, lambda: hashlib.sha256(image.read_bytes()).hexdigest()
        )
        page = await self.extract_page(
            image, photo_index=0, source_sha256=digest, deadline=deadline
        )
        return await run_sync_before(deadline, page.model_dump, mode="json")

    async def check_connection(self) -> bool:
        return (await self.get_capabilities()).ready
