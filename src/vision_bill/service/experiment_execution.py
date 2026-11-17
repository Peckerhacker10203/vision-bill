"""Benchmark execution through the same public receipt-service paths as uploads.

Forced managed OCR requires production OCR participation in the shared handoff.
Unmanaged CPU/replay extensions may be selected independently of production OCR.
All pipeline policy overrides are local to the experiment.
"""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter
from typing import TYPE_CHECKING, Any

from ..config import OcrSettings
from ..model.experiment import ExperimentOptions, ExtractionPipeline
from ..model.handoff import HandoffStatus, HandoffWait
from ..model.receipt import Receipt
from .execution_policy import EXECUTION_POLICY_VERSION
from .extraction_pipeline import ExtractionPipeline as ReceiptExtractionPipeline
from .gpu_handoff import admission_deadline

if TYPE_CHECKING:
    from .receipt_service import ReceiptService


@dataclass
class ExperimentResult:
    receipt: Receipt
    attempts: int
    elapsed_ms: float
    metadata: dict[str, Any]


class ExperimentFailure(ValueError):
    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


def ocr_fingerprint(provider: Any) -> str | None:
    """Identify the published OCR extension's client without exposing its URL.

    The full pipeline identity separately binds the extension's advertised
    profile and runtime fingerprint; this hash binds local client settings.
    """
    if provider is None:
        return None
    identity = {
        "provider": type(provider).__name__,
        "endpoint": getattr(provider, "service_url", None),
        "timeout": getattr(provider, "timeout", None),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def effective_policy(
    service: "ReceiptService",
    options: ExperimentOptions,
) -> tuple[Any, OcrSettings, ExperimentOptions]:
    """Keep upload policy, explicit direct vision and strict OCR distinct."""
    settings = service._ocr_settings.model_copy(update={
        "allow_vision_fallback": False, "vision_fallback_model_id": None,
    })
    options = options.model_copy(update={"allow_ocr_fallback": False})
    ocr = service._ocr_provider
    if options.pipeline == ExtractionPipeline.OCR:
        settings = settings.model_copy(
            update={
                "enabled": True,
                "required": True,
                "allow_vision_fallback": False,
            }
        )
        if ocr is None:
            from ..provider.factory import get_ocr_provider

            ocr = get_ocr_provider(settings)
    # Direct vision still coordinates with the configured OCR sidecar; it simply
    # skips its load phase. This also protects direct benchmark jobs.
    return ocr, settings, options


async def execute_submission(
    service: "ReceiptService",
    model_id: str,
    paths: list[Path],
    options: ExperimentOptions,
    tags: list[str],
) -> ExperimentResult:
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("Receipt image or additional photo is unavailable")
    started = perf_counter()
    probe = copy(service)
    probe._ocr_provider, probe._ocr_settings, effective_options = effective_policy(service, options)
    provider = service._provider.for_experiment(effective_options)
    probe._provider = provider
    probe._experiment_tags = list(tags)
    metadata: dict[str, Any] = {
        "requested_pipeline": options.pipeline.value,
        "requested_options": options.model_dump(mode="json"),
        "fallback_used": False,
        "image_count": len(paths),
        "provider": type(provider).__name__,
        "options": provider.experiment_options.model_dump(mode="json")
        if provider.experiment_options
        else options.model_dump(mode="json"),
    }
    try:
        async with asyncio.timeout(provider.analysis_timeout_seconds):
            envelope = await probe.extract_receipt(
                model_id,
                paths,
                options=effective_options,
                deadline=monotonic() + provider.analysis_timeout_seconds,
                tags=tags,
            )
            metadata.update(envelope.provenance.model_dump(mode="json"))
            metadata["calls"] = envelope.telemetry.calls
            metadata["telemetry"] = envelope.telemetry.model_dump(mode="json")
            metadata["source_projection_version"] = envelope.source_projection_version
            metadata["draft"] = envelope.draft.model_dump(mode="json") if envelope.draft else None
            metadata["raw_field_sources"] = {
                key: [ref.model_dump(mode="json") for ref in refs]
                for key, refs in envelope.raw_field_sources.items()
            }
            metadata["field_sources"] = {
                key: [ref.model_dump(mode="json") for ref in refs]
                for key, refs in envelope.field_sources.items()
            }
            metadata["overlap_decisions"] = envelope.overlap_decisions
            metadata["findings"] = [f.model_dump(mode="json") for f in envelope.findings]
            from ..model.ocr import OcrPageMetadata

            metadata["ocr_pages"] = [
                OcrPageMetadata.model_validate(page.model_dump()).model_dump(mode="json")
                for page in envelope.ocr_pages
            ]
            if envelope.receipt is None:
                code = envelope.findings[-1].code if envelope.findings else "receipt_incomplete"
                raise ValueError(code)
            receipt = envelope.receipt
    except HandoffWait:
        raise
    except Exception as exc:
        metadata.setdefault("calls", provider.experiment_trace or [])
        metadata["elapsed_ms"] = (perf_counter() - started) * 1000
        code = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        raise ExperimentFailure(code, metadata) from exc
    metadata["llm_ms"] = sum(call.get("elapsed_ms", 0) for call in metadata["calls"])
    elapsed = (perf_counter() - started) * 1000
    metadata["elapsed_ms"] = elapsed
    return ExperimentResult(receipt, len(metadata["calls"]) or 1, elapsed, metadata)


async def pipeline_identity(
    service: "ReceiptService", options: ExperimentOptions, *, qualify: bool = False,
    managed_model_ids: Sequence[str] = (),
    expected_binding_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze effective local execution contracts without receipt/reference observations."""
    import inspect

    from ..helper.document_image import NORMALIZER_VERSION, document_rgb
    from ..model.extraction import (
        EXTRACTION_SCHEMA_VERSION,
        SOURCE_PROJECTION_VERSION,
        ReceiptInput,
    )
    from ..provider.llm.experiment import source_extraction_prompt

    ocr, settings, effective_options = effective_policy(service, options)
    provider = service._provider.for_experiment(effective_options)
    handoff = ReceiptExtractionPipeline(provider, ocr, settings).coordinator()
    binding_sha256 = None
    if handoff is not None:
        binding = handoff.require_binding(expected_binding_sha256)
        if qualify and any(model not in binding.model_ids for model in managed_model_ids):
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="model_handoff_model_not_adopted",
            ))
        binding_sha256 = handoff.binding_sha256(binding)
    elif expected_binding_sha256 is not None:
        raise HandoffWait(HandoffStatus(
            state="configuration_required", reason_code="execution_policy_changed",
        ))
    capabilities = None
    if ocr is not None and options.pipeline != ExtractionPipeline.DIRECT:
        if qualify and handoff is not None:
            deadline = monotonic() + provider.analysis_timeout_seconds
            async with handoff.session(
                deadline, model_ids=managed_model_ids, expected_binding_sha256=binding_sha256,
                lock_deadline=admission_deadline(deadline),
            ) as owner:
                await owner.ocr_phase()
                qualified = await ocr.get_capabilities()
                if not qualified.ready or not qualified.runtime_fingerprint:
                    raise ValueError("ocr_qualification_required")
                capabilities = qualified.model_dump(mode="json")
        else:
            capabilities = (await ocr.get_capabilities()).model_dump(mode="json")
        if getattr(ocr, "managed_lifecycle", False):
            # Residency and availability are observations, not frozen identity.
            capabilities = {key: capabilities[key] for key in (
                "profile_id", "runtime_fingerprint", "ocr_max_new_tokens"
            )}
    return {
        "execution_policy_version": EXECUTION_POLICY_VERSION,
        "managed_binding_sha256": binding_sha256,
        "version": 1,
        "normalizer_version": NORMALIZER_VERSION,
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "source_projection_version": SOURCE_PROJECTION_VERSION,
        "source_prompt_sha256": hashlib.sha256(
            inspect.getsource(source_extraction_prompt).encode()
        ).hexdigest(),
        "input_schema_sha256": hashlib.sha256(
            json.dumps(ReceiptInput.model_json_schema(), sort_keys=True).encode()
        ).hexdigest(),
        "preprocessing_sha256": hashlib.sha256(
            (
                inspect.getsource(document_rgb)
                + inspect.getsource(provider._prepare_receipt_input)
                + inspect.getsource(provider._authorized_source_refs)
            ).encode()
        ).hexdigest(),
        "ocr_client": ocr_fingerprint(ocr),
        "ocr_runtime": capabilities,
        "ocr_settings": settings.model_dump(mode="json", exclude={"service_url"}),
        "requested_options": options.model_dump(mode="json"),
        "options": (provider.experiment_options or effective_options).model_dump(mode="json"),
        "formatter_output_tokens": provider.max_output_tokens,
        "context_window_tokens": getattr(provider, "context_window_tokens", None),
        "analysis_timeout_seconds": provider.analysis_timeout_seconds,
    }
