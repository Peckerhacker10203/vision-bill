"""Asynchronous, restart-safe local benchmark execution."""
import asyncio
import json
import logging
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID

from ..model.benchmark import (
    BenchmarkCreate,
    BenchmarkRun,
    BenchmarkStatus,
    GoldSnapshot,
    ReferencePreview,
)
from ..model.experiment import ExperimentOptions, ExtractionPipeline
from ..model.receipt import LineItem, Receipt, TaxLine
from ..model.workflow import AccessScope
from ..provider.db.benchmark_db import BenchmarkDB
from ..provider.llm.base import LLMProvider
from ..provider.llm.experiment import endpoint_fingerprint
from .benchmark_scoring import PROMPT_VERSION, dataset_fingerprint, score_receipts
from .experiment_execution import (
    ExperimentFailure,
    image_digest,
    ocr_fingerprint,
    pipeline_identity,
)
from .experiment_scoring import DETAIL_SCORING_VERSION, detailed_score, detailed_score_v3
from .receipt_service import ReceiptService

logger = logging.getLogger(__name__)


class BenchmarkService:
    """Owns benchmark task execution; normal receipt/image records are read-only."""

    def __init__(self, provider: LLMProvider, receipt_service: ReceiptService):
        self._provider = provider
        self._receipts = receipt_service
        self._db = BenchmarkDB(receipt_service.pool, receipt_service._db)
        self._worker: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._worker is None:
            self._stopping.clear()
            self._worker = asyncio.create_task(self._work(), name="benchmark-worker")

    async def stop(self) -> None:
        self._stopping.set()
        if self._worker:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def create_run(self, request: BenchmarkCreate, *, scope: AccessScope) -> BenchmarkRun:
        use_ocr = request.experiment.pipeline == ExtractionPipeline.OCR or (
            request.experiment.pipeline == ExtractionPipeline.UPLOAD and self._receipts._ocr_settings.enabled
        )
        models = await self._provider.get_available_models(capability="text" if use_ocr else "vision")
        available = {m.id: m for m in models}
        selected_models = request.model_ids or list(available)
        if not selected_models:
            raise ValueError("No vision models are available")
        unknown = set(selected_models) - set(available)
        if unknown:
            raise ValueError(f"Selected model(s) unavailable: {', '.join(sorted(unknown))}")
        selected_ids = await self._select_receipts(request, scope=scope)
        if not selected_ids:
            raise ValueError("No current qualified references match the selection")
        if request.experiment.think is not None and not self._provider.supports_thinking_control:
            raise ValueError("Thinking override is supported only by the Ollama provider")
        if request.experiment.pipeline == ExtractionPipeline.OCR and not self.capabilities()["ocr_available"]:
            raise ValueError("OCR extension is not configured; choose normal upload or direct vision")
        tags = await self._receipts.list_tags()
        inputs = [await self._capture_input(receipt_id, scope=scope) for receipt_id in selected_ids]
        expected = [Receipt.model_validate(item["expected"]) for item in inputs]
        probe = self._provider.for_experiment(request.experiment)
        options = probe.experiment_options or request.experiment
        counts = {len(item["images"]) for item in inputs}
        for item in inputs:
            item["pipeline"] = options.pipeline.value
        manifest = {"identity": await pipeline_identity(self._receipts, options), "options": options.model_dump(mode="json"), "tags": tags, "inputs": inputs,
                    "provider": type(self._provider).__name__,
                    "endpoint_fingerprint": endpoint_fingerprint(self._provider),
                    "ocr_fingerprint": ocr_fingerprint(getattr(self._receipts, "_ocr_provider", None)),
                    "prompt_hashes": {str(count): sha256(probe.build_prompt(tags, image_count=count).encode()).hexdigest() for count in counts},
                    "validation_schema_sha256": sha256(json.dumps(Receipt.model_json_schema(), sort_keys=True).encode()).hexdigest(),
                    "thinking_default": "server-controlled" if options.think is None else "explicit"}
        return await self._db.create_run(
            model_ids=selected_models, receipt_ids=selected_ids, fingerprint=dataset_fingerprint(expected),
            model_digests={m.id: m.digest for m in models}, timeout=request.request_timeout_seconds,
            policy=request.council_policy, absolute=request.council_absolute_threshold,
            relative=request.council_relative_threshold, apply_flags=request.apply_council_flags,
            prompt_version=PROMPT_VERSION, scoring_version=DETAIL_SCORING_VERSION, manifest=manifest, owner_id=scope.user_id, scope=scope,
        )

    async def get_status(self, run_id: UUID, *, scope: AccessScope) -> BenchmarkStatus | None:
        run = await self._db.run(run_id, scope=scope)
        if not run:
            return None
        counts = await self._db.task_counts(run_id)
        return BenchmarkStatus(run=run, summaries=await self._db.summaries(run_id), **counts)

    async def list_runs(self, *, scope: AccessScope) -> list[BenchmarkRun]:
        return await self._db.runs(scope=scope)

    async def reevaluate(self, run_id: UUID, receipt_id: UUID, model_id: str, *, scope: AccessScope) -> dict[str, Any]:
        """Transient inspection: extraction and field diff are returned, never stored."""
        run = await self._db.run(run_id, scope=scope)
        if not run or receipt_id not in run.receipt_ids or model_id not in run.model_ids:
            raise ValueError("Receipt/model is not part of this benchmark run")
        manifest = await self._db.manifest(run_id)
        if manifest:
            available = {m.id: m for m in await self._provider.get_available_models(capability="all")}
            current = available.get(model_id)
            digest = (await self._run_digests(run)).get(model_id)
            if current is None or (digest and current.digest and digest != current.digest):
                raise ValueError("Selected model is unavailable or its digest changed; create a new run")
            expected, result = await asyncio.wait_for(
                self._run_experiment(manifest, receipt_id, model_id), timeout=await self._timeout(run))
            detail = self._score(manifest, receipt_id, expected, result.receipt)
            item = next(item for item in manifest["inputs"] if item["receipt_id"] == str(receipt_id))
            return {"receipt_id": receipt_id, "model_id": model_id, **detail,
                    "expected": item["expected"] if item.get("reference") else expected.model_dump(mode="json"),
                    "actual": result.receipt.model_dump(mode="json"),
                    "metadata": result.metadata}
        expected, image_path = await self._expected_receipt_and_path(receipt_id)
        if not image_path or not Path(image_path).is_file():
            raise FileNotFoundError("Verified receipt image is unavailable")
        result = await self._provider.analyse_receipt_with_metadata(model_id, Path(image_path))
        actual = result.receipt
        scores = score_receipts(expected, actual)
        expected_dump, actual_dump = expected.model_dump(mode="json"), actual.model_dump(mode="json")
        diff = {key: {"expected": expected_dump[key], "actual": actual_dump[key]} for key in expected_dump if expected_dump[key] != actual_dump[key]}
        return {"receipt_id": receipt_id, "model_id": model_id, "scores": scores, "expected": expected_dump, "actual": actual_dump, "diff": diff}

    async def _select_receipts(self, request: BenchmarkCreate, *, scope: AccessScope) -> list[UUID]:
        if request.receipt_ids is not None:
            result = []
            for receipt_id in request.receipt_ids:
                row = await self._receipts.get_receipt_by_id(receipt_id, user_id=scope.user_id, can_see_all=scope.can_see_all)
                if row is None or not row.verified:
                    raise ValueError("Explicit receipt is unavailable or not verified")
                result.append(receipt_id)
            return result
        candidates = await self._db.reference_candidates(
            scope=scope, category=request.category, max_source_confidence=request.max_source_confidence,
        )
        selected = []
        for receipt_id in candidates:
            try:
                await self._db.current_reference(receipt_id, scope=scope)
            except (ValueError, OSError):
                continue  # Filtered selection excludes stale/ineligible references.
            selected.append(receipt_id)
            if request.limit is not None and len(selected) >= request.limit:
                break
        return selected

    async def _expected_receipt_and_path(self, receipt_id: UUID) -> tuple[Receipt, str | None]:
        details = await self._receipts.get_receipt_with_details(receipt_id)
        if details is None or not details.receipt.verified:
            raise ValueError("Benchmark data must be a verified receipt")
        r = details.receipt
        return Receipt.model_validate({"confidence": r.confidence, "merchant_name": r.merchant_name, "merchant_address": r.merchant_address, "receipt_number": r.receipt_number, "date": r.date, "time": r.time, "currency": r.currency, "category": r.category, "line_items": [LineItem.model_validate(x.model_dump()) for x in details.line_items], "taxes": [TaxLine.model_validate(x.model_dump()) for x in details.taxes], "subtotal": r.subtotal, "discount_total": r.discount_total, "tax_total": r.tax_total, "tip": r.tip, "total": r.total, "payment_method": r.payment_method}), details.image_path

    async def _expected_receipt(self, receipt_id: UUID) -> Receipt:
        return (await self._expected_receipt_and_path(receipt_id))[0]

    async def _work(self) -> None:
        while not self._stopping.is_set():
            try:
                task = await self._db.lease_task()
                if task is None:
                    await asyncio.sleep(1)
                    continue
                await self._execute(dict(task))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Benchmark worker iteration failed")
                await asyncio.sleep(5)

    async def _execute(self, task: dict[str, Any]) -> None:
        try:
            models = {m.id: m for m in await self._provider.get_available_models(capability="all")}
        except Exception:  # noqa: BLE001 - provider SDK exception types vary by transport version
            await self._db.retry_task(task)
            return
        run = await self._db.run(task["run_id"])
        model = models.get(task["model_id"])
        if run is None:
            return
        expected_digest = (await self._run_digests(run)).get(task["model_id"])
        if model is None:
            await self._db.retry_task(task, waiting=True)
            return
        if expected_digest and model.digest and expected_digest != model.digest:
            await self._db.retry_task(task, waiting=True)
            return
        manifest = await self._db.manifest(task["run_id"])
        if manifest:
            await self._execute_experiment(task, manifest, run)
            return
        try:
            expected, image_path = await self._expected_receipt_and_path(task["receipt_id"])
            if not image_path or not Path(image_path).is_file():
                raise FileNotFoundError("Image missing")
            result = await asyncio.wait_for(self._provider.analyse_receipt_with_metadata(task["model_id"], Path(image_path)), timeout=await self._timeout(run))
            await self._db.finish_task(task, success=True, score=score_receipts(expected, result.receipt)["overall"], confidence=result.receipt.confidence, attempts=result.attempts, latency_ms=result.elapsed_ms)
        except (ValueError, FileNotFoundError):
            await self._db.finish_task(task, success=False)
        except Exception:
            logger.warning("Transient benchmark failure for run=%s model=%s receipt=%s", task["run_id"], task["model_id"], task["receipt_id"], exc_info=True)
            await self._db.retry_task(task)

    async def _run_digests(self, run: BenchmarkRun) -> dict[str, str | None]:
        async with self._receipts.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT model_digests FROM benchmark_runs WHERE id=$1", run.id)
        if not row:
            return {}
        value = row["model_digests"]
        return dict(json.loads(value) if isinstance(value, str) else value)

    async def _timeout(self, run: BenchmarkRun) -> int:
        async with self._receipts.pool.acquire() as conn:
            return int(await conn.fetchval("SELECT request_timeout_seconds FROM benchmark_runs WHERE id=$1", run.id))

    def capabilities(self) -> dict[str, bool]:
        return {"ocr_available": bool(self._receipts._ocr_settings.service_url),
                "ocr_enabled": self._receipts._ocr_settings.enabled,
                "thinking_control": self._provider.supports_thinking_control}

    async def results(self, run_id: UUID, *, limit: int, offset: int, scope: AccessScope) -> list[dict[str, Any]]:
        if await self._db.run(run_id, scope=scope) is None:
            raise ValueError("Benchmark run not found")
        return await self._db.results(run_id, limit=limit, offset=offset)

    async def reference_preview(self, receipt_id: UUID, *, scope: AccessScope) -> ReferencePreview:
        return await self._db.reference_preview(receipt_id, scope=scope)

    async def qualify_reference(self, receipt_id: UUID, expected_review_revision: int,
                                expected_document_sha256: str, *, scope: AccessScope) -> GoldSnapshot:
        return await self._db.qualify_reference(receipt_id, expected_review_revision,
                                                expected_document_sha256, scope=scope)

    async def _capture_input(self, receipt_id: UUID, *, scope: AccessScope) -> dict[str, Any]:
        reference = await self._db.current_reference(receipt_id, scope=scope)
        return {"receipt_id": str(receipt_id), "expected": reference.expected,
                "images": reference.sources, "reference": reference.model_dump(mode="json"),
                "scored_fields": reference.scored_fields}

    async def _run_experiment(self, manifest: dict[str, Any], receipt_id: UUID, model_id: str) -> tuple[Receipt, Any]:
        if manifest.get("endpoint_fingerprint") != endpoint_fingerprint(self._provider):
            raise ValueError("Provider endpoint changed since run creation; create a new run")
        if manifest["provider"] != type(self._provider).__name__:
            raise ValueError("Provider changed since run creation; create a new run")
        item = next((x for x in manifest["inputs"] if x["receipt_id"] == str(receipt_id)), None)
        if item is None:
            raise ValueError("Receipt is absent from the frozen benchmark inputs")
        if item["pipeline"] != "direct_vision" and manifest.get("ocr_fingerprint") != ocr_fingerprint(getattr(self._receipts, "_ocr_provider", None)):
            raise ValueError("OCR provider endpoint or client settings changed; create a new run")
        schema_hash = sha256(json.dumps(Receipt.model_json_schema(), sort_keys=True).encode()).hexdigest()
        if manifest["validation_schema_sha256"] != schema_hash:
            raise ValueError("Receipt validation schema changed; create a new run")
        paths = [Path(image["path"]) for image in item["images"]]
        for path, image in zip(paths, item["images"], strict=True):
            if await asyncio.to_thread(image_digest, path) != image["sha256"]:
                raise ValueError("Receipt image changed since run creation")
        options = ExperimentOptions.model_validate(manifest["options"])
        if manifest.get("identity") is not None and manifest["identity"] != await pipeline_identity(self._receipts, options):
            raise ValueError("Effective pipeline identity changed; create a new run")
        probe = self._provider.for_experiment(options)
        current_prompt = sha256(probe.build_prompt(list(manifest["tags"]), image_count=len(paths)).encode()).hexdigest()
        if current_prompt != manifest["prompt_hashes"][str(len(paths))]:
            raise ValueError("Extraction prompt changed since run creation; create a new run")
        options = options.model_copy(update={"pipeline": ExtractionPipeline(item["pipeline"])})
        # The outer run deadline includes OCR, LLM generation and repair attempts.
        result = await self._receipts.analyse_submission_with_metadata(model_id, paths, options, list(manifest["tags"]))
        if manifest.get("identity") is not None:
            if result.metadata.get("ordered_source_sha256") != [image["sha256"] for image in item["images"]]:
                raise ValueError("Executed source identity differs from frozen input")
            runtime = manifest["identity"].get("ocr_runtime")
            if runtime and result.metadata.get("used_pipeline") == "ocr" and any(
                runtime.get(key) != result.metadata.get(key)
                for key in ("profile_id", "runtime_fingerprint")
            ):
                raise ValueError("Executed OCR runtime differs from frozen input")
        result.metadata["requested_pipeline"] = manifest["options"]["pipeline"]
        result.metadata["resolved_pipeline"] = result.metadata.get("used_pipeline", item["pipeline"])
        result.metadata["image_hashes"] = [image["sha256"] for image in item["images"]]
        return Receipt.model_validate(item["expected"]), result

    @staticmethod
    def _score(manifest: dict[str, Any], receipt_id: UUID, expected: Receipt, actual: Receipt | None) -> dict[str, Any]:
        item = next(x for x in manifest["inputs"] if x["receipt_id"] == str(receipt_id))
        if "reference" not in item and actual is not None:
            return detailed_score_v3(expected, actual, classification=manifest["options"]["include_classification"])
        return detailed_score(item["expected"], actual, classification=manifest["options"]["include_classification"],
                              scored_fields=item.get("scored_fields"))

    async def _execute_experiment(self, task: dict[str, Any], manifest: dict[str, Any], run: BenchmarkRun) -> None:
        from time import perf_counter
        started = perf_counter()
        try:
            expected, result = await asyncio.wait_for(
                self._run_experiment(manifest, task["receipt_id"], task["model_id"]),
                timeout=await self._timeout(run))
            detail = self._score(manifest, task["receipt_id"], expected, result.receipt)
            payload = {**detail, "expected": next(x["expected"] for x in manifest["inputs"] if x["receipt_id"] == str(task["receipt_id"])),
                       "actual": result.receipt.model_dump(mode="json"), "metadata": result.metadata,
                       "legacy_scores": score_receipts(expected, result.receipt)}
        except (ExperimentFailure, ValueError, OSError, TimeoutError) as exc:
            metadata = exc.metadata if isinstance(exc, ExperimentFailure) else {"options": manifest["options"]}
            elapsed = (perf_counter() - started) * 1000
            metadata["elapsed_ms"] = elapsed
            metadata["requested_pipeline"] = manifest["options"]["pipeline"]
            item = next(x for x in manifest["inputs"] if x["receipt_id"] == str(task["receipt_id"]))
            failure_score = self._score(manifest, task["receipt_id"], Receipt.model_validate(item["expected"]), None) if "reference" in item else {}
            await self._db.finish_task(task, success=False, latency_ms=elapsed,
                                       attempts=len(metadata.get("calls", [])) or 1,
                                       result={**failure_score, "error": str(exc) or type(exc).__name__, "metadata": metadata})

            return
        await self._db.finish_task(task, success=True, score=detail["scores"]["normalized"],
                                   confidence=result.receipt.confidence, attempts=result.attempts,
                                   latency_ms=result.elapsed_ms, result=payload)
