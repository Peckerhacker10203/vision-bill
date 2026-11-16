import asyncio
import base64
import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter
from typing import Any, TypeVar
from uuid import uuid4

from PIL import Image, ImageOps
from pydantic import ValidationError

from ...helper.deadline import check_deadline, run_sync_before
from ...helper.duplicates import verify_duplicate
from ...model.experiment import ExperimentOptions
from ...model.extraction import (
    SOURCE_PROJECTION_VERSION,
    DerivedInputView,
    ExtractionDraft,
    InferenceEnvelope,
    InferenceTelemetry,
    PipelineProvenance,
    ReceiptInput,
)
from ...model.item_category import CATEGORY_CODES, CATEGORY_DEFINITIONS
from ...model.ocr import SourceRef, WorkingImage
from ...model.receipt import Receipt
from ...model.receipt_extraction import DuplicateCandidate, MultiImageReceipt, validate_overlap
from ...model.review import AmountPolicy, ReviewFinding
from .errors import UnreadableReceiptError
from .experiment import extraction_prompt, source_extraction_prompt

TEXT_CAPABILITIES = {"completion", "text", "chat", "vision", "multimodal"}

T = TypeVar("T")


@dataclass
class ModelInfo:
    def __init__(self, id: str, parameter_size: str | None = None):
        self.id = id
        self.parameter_size = parameter_size

    id: str
    parameter_size: str | None = None
    capabilities: list[str] | None = None
    digest: str | None = None
    context_window_tokens: int | None = None


@dataclass
class AnalysisResult:
    receipt: Receipt
    attempts: int
    elapsed_ms: float
    model_digest: str | None = None


class _PartialDraftError(ValueError):
    def __init__(self, envelope: InferenceEnvelope, *, incomplete: bool = False):
        super().__init__("receipt_incomplete")
        self.envelope = envelope
        self.incomplete = incomplete


class LLMProvider(ABC):
    """Interface every LLM backend must implement."""

    supports_thinking_control: bool = False
    analysis_timeout_seconds: float = 600
    experiment_options: ExperimentOptions | None = None
    experiment_trace: list[dict[str, Any]] | None = None
    max_output_tokens: int = 8192

    def for_experiment(self, options: ExperimentOptions) -> "LLMProvider":
        """Copy request settings; never mutate the live upload provider."""
        probe = copy(self)
        temperature = options.temperature
        if temperature is None:
            temperature = getattr(self, "_temperature", 0.0)
        probe.experiment_options = options.model_copy(update={"temperature": temperature})
        probe.experiment_trace = []
        probe.update_runtime_settings(temperature=temperature)
        return probe

    def update_runtime_settings(self, *, temperature: float) -> None:
        """Apply settings which are safe to change without rebuilding a provider."""

    @abstractmethod
    async def get_available_models(self, *, capability: str = "vision") -> list[ModelInfo]:
        """Return the models this provider currently exposes."""

    @abstractmethod
    async def analyse_receipt_from_model(
        self, model_id: str, image: Path | Sequence[Path], tags: Sequence[str] | None = None
    ) -> Receipt:
        """Send an image + prompt to the given model and return the result.

        ``tags`` is the allowed line-item tag vocabulary; when provided it is
        embedded in the prompt so the model only emits known tags.
        """

    @abstractmethod
    async def analyse_receipt_from_ocr_text(
        self, model_id: str, ocr_result: dict[str, Any], tags: Sequence[str] | None = None
    ) -> Receipt:
        """Send OCR text + prompt to the given model and return the result.

        Used in the two-stage OCR → LLM pipeline where OCR has already
        extracted text from the image and the LLM formats it into a Receipt.
        """

    @abstractmethod
    async def send_message(self, model_id: str, messages: Sequence[Mapping[str, Any]]) -> str:
        """Send a message to the given model and return the response."""

    @abstractmethod
    async def check_connection(self) -> bool:
        """Return True if the backend is reachable, False otherwise."""

    async def analyse_receipt_with_metadata(
        self, model_id: str, image: Path | Sequence[Path], tags: Sequence[str] | None = None
    ) -> AnalysisResult:
        """Compatibility wrapper for providers which do not expose attempt telemetry."""
        started = perf_counter()
        receipt = await self.analyse_receipt_from_model(model_id, image, tags=tags)
        return AnalysisResult(
            receipt=receipt, attempts=1, elapsed_ms=(perf_counter() - started) * 1000
        )

    async def _repair_response(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        parse: Callable[[str], T],
        *,
        deadline: float,
        attempts: list[dict[str, Any]],
    ) -> T:
        """One parser/repair loop for live typed input and legacy/replay wrappers."""
        last_error: ValueError | None = None
        partial_error: _PartialDraftError | None = None
        for attempt in range(3):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("extraction_deadline")
            before = len(self.experiment_trace or [])
            async with asyncio.timeout(remaining):
                content = await self.send_message(model_id, messages)
            check_deadline(deadline)
            call = (
                dict(self.experiment_trace[-1])
                if self.experiment_trace and len(self.experiment_trace) > before
                else {}
            )
            # These bounded artifacts are owner-scoped by the caller, never general logs.
            call.update(
                attempt=attempt + 1,
                requested_model=model_id,
                raw_response=content[:200000],
                response_truncated=len(content) > 200000,
            )
            attempts.append(call)
            try:
                if not content or len(content) > 200000:
                    raise ValueError("invalid_response_size")
                parsed = await run_sync_before(deadline, parse, content)
                if isinstance(parsed, InferenceEnvelope):
                    self._record_supplied_fields(call, parsed)
                return parsed
            except ValueError as exc:
                if isinstance(exc, _PartialDraftError):
                    partial_error = exc
                    self._record_supplied_fields(call, exc.envelope)
                    if exc.incomplete:
                        return exc.envelope  # type: ignore[return-value]
                last_error = exc
                messages = [
                    *messages,
                    {"role": "assistant", "content": content[:200000]},
                    {
                        "role": "user",
                        "content": "That response was not valid JSON matching the required schema. "
                        "Preserve missing values as null and include ordered source_image indices. "
                        "Please respond again with ONLY corrected JSON.",
                    },
                ]
        if partial_error is not None:
            raise partial_error
        raise ValueError("Failed to get a valid response after 3 attempts") from last_error

    @staticmethod
    def _record_supplied_fields(call: dict[str, Any], envelope: InferenceEnvelope) -> None:
        if envelope.draft is not None:
            call["supplied_fields"] = sorted(envelope.draft.model_fields_set)
            call["null_fields"] = sorted(
                key
                for key in envelope.draft.model_fields_set
                if getattr(envelope.draft, key) is None
            )

    async def _legacy_analysis(
        self, model_id: str, messages: list[dict[str, Any]], *, image_count: int = 1
    ) -> AnalysisResult:
        """Explicit compatibility/replay path for historical Receipt-only responses."""
        started = perf_counter()
        attempts: list[dict[str, Any]] = []

        def parse(content: str) -> Receipt:
            if not content.lstrip().startswith(("{", "```")) and re.search(
                r"provide.*image", content, re.IGNORECASE | re.DOTALL
            ):
                raise ValueError("Please ensure the model supports vision capabilities")
            return self.parse_llm_response(content, image_count=image_count)

        try:
            receipt = await self._repair_response(
                model_id,
                messages,
                parse,
                deadline=monotonic() + self.analysis_timeout_seconds,
                attempts=attempts,
            )
        except ValueError as exc:
            if attempts and re.search(
                r"provide.*image",
                str(attempts[-1].get("raw_response", "")),
                re.IGNORECASE | re.DOTALL,
            ):
                raise ValueError("Please ensure the model supports vision capabilities") from exc
            raise
        return AnalysisResult(receipt, len(attempts), (perf_counter() - started) * 1000)

    async def extract_receipt_input(
        self, model_id: str, input: ReceiptInput, *, tags: Sequence[str], deadline: float
    ) -> InferenceEnvelope:
        """Extract observations first; classification is an independent enrichment call."""
        started = perf_counter()
        probe = copy(self)
        probe.experiment_trace = []
        input, messages, views = await run_sync_before(
            deadline, probe._prepare_receipt_input, input
        )
        count = len(input.image_paths)
        provenance = PipelineProvenance(
            requested_pipeline=input.mode,
            used_pipeline=input.mode,
            source_image_count=count,
            ordered_source_sha256=[r.source_sha256 for r in input.source_refs],
            model_id=model_id,
        )
        result = InferenceEnvelope(
            provenance=provenance,
            telemetry=InferenceTelemetry(elapsed_ms=0, max_output_tokens=self.max_output_tokens),
            field_sources={"input": input.source_refs},
        )
        attempts: list[dict[str, Any]] = []

        try:
            result = await probe._repair_response(
                model_id,
                messages,
                partial(probe._parse_receipt_input, input=input, prototype=result),
                deadline=deadline,
                attempts=attempts,
            )
        except ValueError as exc:
            if isinstance(exc, _PartialDraftError):
                result = exc.envelope
            # A validated nullable draft is useful even when no complete Receipt can be built.
            result.findings.append(
                ReviewFinding(
                    id="formatter_invalid_response",
                    code="formatter_invalid_response",
                    severity="review",
                    field_path="draft",
                    source_refs=input.source_refs,
                )
            )
        result.telemetry = InferenceTelemetry(
            stage_ms={"formatter": (perf_counter() - started) * 1000},
            elapsed_ms=(perf_counter() - started) * 1000,
            max_output_tokens=self.max_output_tokens,
            calls=attempts,
            finish_reason=attempts[-1].get("finish_reason") if attempts else None,
            output_tokens=attempts[-1].get("completion_tokens", attempts[-1].get("eval_count"))
            if attempts
            else None,
        )
        if result.receipt is not None and (
            self.experiment_options is None or self.experiment_options.include_classification
        ):
            result = await probe._enrich_classification(model_id, result, tags, deadline)
        check_deadline(deadline)
        result.input_views = views
        result.telemetry.elapsed_ms = (perf_counter() - started) * 1000
        return result

    def _prepare_receipt_input(
        self, input: ReceiptInput
    ) -> tuple[ReceiptInput, list[dict[str, Any]], list[DerivedInputView]]:
        """Own temporary derivatives until all transport content is materialized as bytes/data."""
        input = input.model_copy(deep=True)
        count = len(input.image_paths)
        if input.mode == "ocr":
            if len(input.ocr_pages) != count or [p.photo_index for p in input.ocr_pages] != list(
                range(count)
            ):
                raise ValueError("incomplete_ocr_input")
            prompt = source_extraction_prompt(
                input, source_refs=self._authorized_source_refs(input)
            )
            return input, [{"role": "user", "content": prompt}], []
        from pillow_heif.as_plugin import register_heif_opener

        register_heif_opener()
        views: list[DerivedInputView] = []
        with TemporaryDirectory(prefix="vision-bill-vision-") as directory:
            paths: list[Path] = []
            for index, source in enumerate(input.image_paths):
                raw = Path(source).read_bytes()
                source_hash = hashlib.sha256(raw).hexdigest()
                target = Path(directory) / f"{index}.png"
                with Image.open(BytesIO(raw)) as original:
                    original.seek(0)
                    transforms = [
                        {
                            "operation": "first_frame_exif_rgb",
                            "frame_count": getattr(original, "n_frames", 1),
                            "frame_index": 0,
                            "exif_orientation": original.getexif().get(274),
                            "source_format": original.format,
                        }
                    ]
                    working = ImageOps.exif_transpose(original).convert("RGB")
                    working.save(target, format="PNG")
                png = target.read_bytes()
                digest = hashlib.sha256(png).hexdigest()
                # Each persisted preparation owns a new immutable artifact path, including retries.
                view_id = uuid4()
                views.append(
                    DerivedInputView(
                        photo_index=index,
                        source_sha256=source_hash,
                        working_image=WorkingImage(
                            view_id=view_id,
                            sha256=digest,
                            width=working.width,
                            height=working.height,
                            transforms=transforms,
                            png_base64=base64.b64encode(png).decode(),
                        ),
                    )
                )
                paths.append(target)
            messages = self._build_image_messages(paths, [])
            # Ollama defers Path reads until chat serialization; detach before our directory closes.
            for message in messages:
                if "images" in message:
                    message["images"] = [path.read_bytes() for path in message["images"]]
            prompt = source_extraction_prompt(input)
            if isinstance(messages[0]["content"], str):
                messages[0]["content"] = prompt
            else:
                messages[0]["content"][0]["text"] = prompt
            return input, messages, views

    @staticmethod
    def _authorized_source_refs(input: ReceiptInput) -> list[SourceRef]:
        """Alternate OCR views do not change ordered input/provenance cardinality."""
        authorized = list(input.source_refs)
        if input.mode != "ocr":
            return authorized
        for ref in input.source_refs:
            if ref.coordinate_space != "original":
                continue
            page = next(
                (
                    page
                    for page in input.ocr_pages
                    if page.photo_index == ref.photo_index
                    and page.source_sha256 == ref.source_sha256
                ),
                None,
            )
            if page is None:
                raise ValueError("ocr_source_reference_mismatch")
            working_ref = SourceRef(
                part_id=ref.part_id,
                photo_index=ref.photo_index,
                source_sha256=ref.source_sha256,
                coordinate_space="working",
                view_id=page.working_image.view_id,
            )
            if working_ref not in authorized:
                authorized.append(working_ref)
        return authorized

    def _parse_receipt_input(
        self, content: str, *, input: ReceiptInput, prototype: InferenceEnvelope
    ) -> InferenceEnvelope:
        """Pure worker result; no caller envelope or attempt list is mutated."""
        result = prototype.model_copy(deep=True)
        count = len(input.image_paths)
        raw = content.strip().removeprefix("```json").removesuffix("```").strip()
        payload = json.loads(raw, parse_float=Decimal)
        if not isinstance(payload, dict):
            raise ValueError("invalid_output_object")  # noqa: TRY004 - malformed model output
        if isinstance(payload.get("error"), dict) and payload["error"].get("code") == "unreadable":
            raise UnreadableReceiptError("receipt_unreadable")
        values = payload.get("draft", payload)
        if not isinstance(values, dict):
            raise ValueError("invalid_draft")  # noqa: TRY004 - malformed model output
        draft = ExtractionDraft.model_validate(values)
        field_sources: dict[str, list[SourceRef]] = {"input": input.source_refs}
        supplied_sources = payload.get("field_sources", {})
        if not isinstance(supplied_sources, dict):
            raise ValueError("invalid_field_sources")  # noqa: TRY004 - malformed model output
        for key, refs in supplied_sources.items():
            if not isinstance(refs, list):
                raise ValueError("invalid_field_sources")  # noqa: TRY004 - malformed model output
            validated = [SourceRef.model_validate(ref) for ref in refs]
            for ref in validated:
                known_refs = [
                    known
                    for known in self._authorized_source_refs(input)
                    if ref.part_id == known.part_id
                    and ref.photo_index == known.photo_index
                    and ref.source_sha256 == known.source_sha256
                    and ref.view_id == known.view_id
                    and ref.coordinate_space == known.coordinate_space
                ]
                if not known_refs:
                    raise ValueError("unknown_source_reference")
                geometry = (ref.block_id, ref.bbox, ref.text_span)
                if geometry == (None, None, None) or any(
                    geometry == (known.block_id, known.bbox, known.text_span)
                    for known in known_refs
                ):
                    continue
                page = next(
                    (
                        page
                        for page in input.ocr_pages
                        if page.photo_index == ref.photo_index
                        and page.source_sha256 == ref.source_sha256
                        and page.working_image.view_id == ref.view_id
                    ),
                    None,
                )
                if input.mode != "ocr" or ref.coordinate_space != "working" or page is None:
                    raise ValueError("unsupported_source_geometry")
                block = next((block for block in page.blocks if block.id == ref.block_id), None)
                if block is None or (ref.bbox is not None and ref.bbox != block.bbox):
                    raise ValueError("unknown_source_block")
                if ref.text_span is not None:
                    # Native transcripts concatenate ordered blocks with one newline. Bind to
                    # this block's actual occurrence, not an equal repeated product elsewhere.
                    start = 0
                    for observed in page.blocks:
                        if observed.id == block.id:
                            break
                        start += len(observed.text) + 1
                    expected_span = (start, start + len(block.text))
                    if (
                        page.text != "\n".join(observed.text for observed in page.blocks)
                        or ref.text_span != expected_span
                    ):
                        raise ValueError("unsupported_source_span")
            field_sources[str(key)] = validated
        # Retain original ordered rows in the draft and all proposed overlap decisions.
        candidates = values.get("duplicate_candidates", [])
        if not isinstance(candidates, list) or any(
            not isinstance(pair, dict) for pair in candidates
        ):
            raise ValueError("invalid_overlap_candidates")
        decisions = payload.get("overlap_decisions", [])
        if not isinstance(decisions, list) or any(not isinstance(d, dict) for d in decisions):
            raise ValueError("invalid_overlap_decisions")
        if count > 1:
            rows = values.get("line_items")
            if rows is not None:
                sources = [row.get("source_image") for row in rows]
                if any(
                    type(i) is not int or not 0 <= i < count for i in sources
                ) or sources != sorted(sources):
                    raise ValueError("invalid_row_source_order")
                validate_overlap(
                    sources, [DuplicateCandidate.model_validate(pair) for pair in candidates]
                )
                for index, source in enumerate(sources):
                    if source < len(input.source_refs):
                        field_sources.setdefault(f"line_items.{index}", [input.source_refs[source]])
        result.amount_policy = AmountPolicy.model_validate(payload.get("amount_policy", {}))
        result.draft = draft
        result.field_sources = field_sources
        result.raw_field_sources = {key: list(refs) for key, refs in field_sources.items()}
        result.source_projection_version = SOURCE_PROJECTION_VERSION
        result.overlap_decisions = decisions + [
            dict(pair, verified=None, removed_index=None)
            for pair in candidates
            if isinstance(pair, dict)
        ]
        # These financial observations cannot inherit final-business EUR/zero defaults.
        observed_values = {
            **values,
            **{key: getattr(draft, key) for key in ("currency", "discount_total", "tax_total")},
        }
        try:
            if count > 1:
                multi = MultiImageReceipt.model_validate(observed_values)
                receipt = multi.reconciled(count)
                verified_groups: dict[int, bool] = {}
                for pair in multi.duplicate_candidates:
                    source = multi.line_items[pair.first_index].source_image
                    matches = verify_duplicate(
                        multi.line_items[pair.first_index], multi.line_items[pair.second_index]
                    )
                    verified_groups[source] = verified_groups.get(source, True) and matches
                result.overlap_decisions = decisions + [
                    dict(
                        pair.model_dump(),
                        verified=verified_groups[multi.line_items[pair.first_index].source_image],
                        removed_index=pair.second_index
                        if verified_groups[multi.line_items[pair.first_index].source_image]
                        else None,
                    )
                    for pair in multi.duplicate_candidates
                ]
                removed = {
                    pair.second_index
                    for pair in multi.duplicate_candidates
                    if verified_groups[multi.line_items[pair.first_index].source_image]
                }
                positions = {
                    old: new
                    for new, old in enumerate(
                        index for index in range(len(multi.line_items)) if index not in removed
                    )
                }
                projected: dict[str, list[SourceRef]] = {}
                for key, refs in field_sources.items():
                    parts = key.split(".")
                    if len(parts) >= 2 and parts[0] == "line_items" and parts[1].isdigit():
                        old = int(parts[1])
                        if old not in positions:
                            continue
                        parts[1] = str(positions[old])
                        key = ".".join(parts)
                    projected[key] = refs
                result.field_sources = projected
            else:
                receipt = Receipt.model_validate(observed_values)
        except ValidationError as exc:
            incomplete = all(
                error["type"] == "missing" or error.get("input") is None for error in exc.errors()
            )
            raise _PartialDraftError(result, incomplete=incomplete) from exc
        except ValueError as exc:
            raise _PartialDraftError(result) from exc
        result.receipt = receipt
        return result

    def _build_image_messages(
        self, image: Path | Sequence[Path], tags: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def _classification_prompt(result: InferenceEnvelope, tags: Sequence[str]) -> str:
        assert result.receipt is not None
        return (
            "Classify these ordered receipt rows. Return JSON with category and line_items "
            "(spending_category and tags only, same row count). Allowed tags: "
            + json.dumps(list(tags))
            + "; categories: "
            + json.dumps(CATEGORY_DEFINITIONS)
            + "; receipt: "
            + result.receipt.model_dump_json()
        )

    @staticmethod
    def _parse_classification(content: str, result: InferenceEnvelope) -> InferenceEnvelope:
        result = result.model_copy(deep=True)
        assert result.receipt is not None
        try:
            payload = json.loads(content)
            rows = payload.get("line_items", [])
            if len(rows) != len(result.receipt.line_items):
                return result
            updated = result.receipt.model_dump()
            updated["category"] = payload.get("category", updated["category"])
            for row, additions in zip(updated["line_items"], rows, strict=True):
                for key in ("spending_category", "tags"):
                    if key in additions:
                        row[key] = additions[key]
            result.receipt = Receipt.model_validate(updated)
            if result.draft is not None:
                result.draft.category = result.receipt.category
                # The original draft rows include overlap; classification targets only final rows.
                # Keep per-row enrichment on Receipt rather than misassigning it to overlap rows.
        except (ValueError, TypeError, AttributeError):
            return result

        return result

    async def _enrich_classification(
        self, model_id: str, result: InferenceEnvelope, tags: Sequence[str], deadline: float
    ) -> InferenceEnvelope:
        prompt = await run_sync_before(deadline, self._classification_prompt, result, tuple(tags))
        before = len(self.experiment_trace or [])
        async with asyncio.timeout(max(0, deadline - monotonic())):
            content = await self.send_message(model_id, [{"role": "user", "content": prompt}])
        check_deadline(deadline)
        call = (
            dict(self.experiment_trace[-1])
            if self.experiment_trace and len(self.experiment_trace) > before
            else {}
        )
        call.update(
            stage="classification",
            requested_model=model_id,
            raw_response=content[:200000],
            response_truncated=len(content) > 200000,
        )
        result = await run_sync_before(deadline, self._parse_classification, content, result)
        result.telemetry.calls.append(call)
        if isinstance(call.get("elapsed_ms"), (float, int)):
            result.telemetry.stage_ms["classification"] = call["elapsed_ms"]
        return result

    def _ocr_result_to_text(self, ocr_result: dict[str, Any]) -> str:
        """Historical harness adapter, shared by both provider backends."""
        from ..ocr.base import compact_ocr_text

        return compact_ocr_text(ocr_result)

    def _build_ocr_formatting_prompt(self, ocr_text: str, tags: Sequence[str] | None = None) -> str:
        """Historical harness prompt adapter; live typed inputs use source_extraction_prompt."""
        return self.build_ocr_prompt(ocr_text, tags)

    def build_ocr_prompt(self, ocr_text: str, tags: Sequence[str] | None = None) -> str:
        return (
            "The input below is OCR TEXT, not an image. Extract the actual receipt values. "
            "Return one populated receipt JSON object. Do NOT return a JSON schema: "
            "properties, $defs, title and type are schema vocabulary, not receipt fields. "
            "No image is needed.\n\n"
            + self.build_prompt(tags, ocr=True)
            + "\n\nOCR-extracted receipt text:\n---\n"
            + ocr_text
            + "\n---\nReturn the populated receipt object, not the schema."
        )

    def build_prompt(
        self, tags: Sequence[str] | None = None, *, image_count: int = 1, ocr: bool = False
    ) -> str:
        """Build a prompt for the LLM to analyze an image.

        When ``tags`` is provided, the prompt gives the model the full tag
        vocabulary and lets it suggest one new tag when nothing fits; otherwise
        the model may invent short free-form tags.
        """
        if self.experiment_options and not self.experiment_options.include_classification:
            return extraction_prompt(image_count, ocr=ocr)
        if tags:
            tag_instruction = (
                "Add short tags to each line_item when useful. Prefer tags from this list: "
                f"{', '.join(tags)}. If no tag in the list fits, you may suggest a single new "
                "short tag (lowercase snake_case, at most three words); suggested tags are "
                "reviewed by a human before they become standard tags."
            )
        else:
            tag_instruction = "Add short free-form tags to each line_item when useful."
        categories = "; ".join(f"{code}: {CATEGORY_DEFINITIONS[code]}" for code in CATEGORY_CODES)
        schema = MultiImageReceipt if image_count > 1 else Receipt
        overlap_instruction = (
            "Extract EVERY purchased row separately from EACH photo, including overlap rows. "
            "Do NOT remove duplicates yourself. Set source_image to its zero-based photo index. "
            "Group line_items by photo, then top-to-bottom within each photo. "
            "In duplicate_candidates identify only the SAME printed rows visible at the end "
            "of one photo and the start of the next. Use zero-based line_items array indices. "
            "Include the entire ordered overlap, not isolated matching products. "
            "Use an empty candidate list when there is no clear visual overlap. "
            "Receipt-level totals must come from the printed receipt, not the duplicated rows."
            if image_count > 1
            else "List line_items in the exact top-to-bottom order on the receipt."
        )
        input_description = (
            "OCR text of ONE receipt" if ocr else "one or more images of ONE receipt"
        )
        source_instruction = (
            "OCR text may contain overlapping photo transcripts. Return ONE complete receipt."
            if ocr
            else "Images are ordered from top to bottom and may overlap. Return ONE complete receipt."
        )
        count_instruction = "" if ocr else f"There are {image_count} input photos. "
        return f"""You are a receipt analyser. You will be given {input_description}.
    {source_instruction}
    {count_instruction}{overlap_instruction}
    Preserve genuinely repeated purchases, even when descriptions and prices are identical.
    If the receipt is too blurry, cut off, or unreadable to extract reliably, return ONLY
    {{"error": {{"code": "unreadable", "message": "Explain which part needs a clearer photo"}}}}.
    This error response is an alternative to the receipt schema. Do not invent missing values.
    You should analyze the receipt and provide a machine-readable JSON response based on the receipt.
    Determine merchant_name from text that is actually printed on the receipt (store name in the
    header or footer, the address line, a website, or any other written hint).
    Do not guess or infer the company name from a logo, photo, or design; if no printed name exists,
    use the best available written hint.
    {tag_instruction}
    Set exactly one spending_category on each line_item using only these codes:
    {categories}.
    Use unknown when the text does not support a category; use deposit for Pfand lines.
    Set the top-level category to the single best category for the whole purchase.
    Use positive unit_price and total_price for deposit/Pfand charges; use negative prices
    for refunds, credits, or deposits returned to the customer.
    Tag every deposit charge or return with `deposit`, and set spending_category to deposit.
    Do not add any additional text or commentary. Only provide the JSON response matching this schema:

    {schema.model_json_schema()}
    """

    def parse_llm_response(self, response: str, *, image_count: int = 1) -> Receipt:
        """Parse the LLM response into a Receipt object."""
        raw_text = response.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error = payload["error"]
            if error.get("code") == "unreadable" and isinstance(error.get("message"), str):
                raise UnreadableReceiptError(error["message"][:1000])
        try:
            if image_count > 1:
                return MultiImageReceipt.model_validate_json(raw_text).reconciled(image_count)
            return Receipt.model_validate_json(raw_text)
        except ValidationError as e:
            # retry logic: feed error back to the LLM, or raise
            raise ValueError(f"LLM output failed validation: {e}")
