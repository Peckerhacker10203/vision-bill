"""Persistence for benchmark control-plane data only.

This module deliberately never writes to receipts, line_items, taxes or images.
"""
import asyncio
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg

from ...model.benchmark import (
    BenchmarkRun,
    BenchmarkSummary,
    CouncilPolicy,
    GoldSnapshot,
    ReferencePreview,
)
from ...model.workflow import AccessScope
from .receipt_db import ReceiptDB


class BenchmarkDB:
    def __init__(self, pool: asyncpg.Pool, receipts: ReceiptDB | None = None):
        self._pool = pool
        self._receipts = receipts

    async def create_run(self, *, model_ids: list[str], receipt_ids: list[UUID], fingerprint: str,
                         model_digests: dict[str, str | None], timeout: int, policy: CouncilPolicy,
                         absolute: Any, relative: Any, apply_flags: bool, prompt_version: str,
                         scoring_version: str, manifest: dict[str, Any] | None = None, owner_id: UUID | None = None, scope: AccessScope | None = None) -> BenchmarkRun:
        async with self._pool.acquire() as conn, conn.transaction():
            references = [item for item in (manifest or {}).get("inputs", []) if "reference" in item]
            if references:
                if scope is None or scope.user_id != owner_id:
                    raise ValueError("reference_scope_required")
                for item in sorted(references, key=lambda item: item["receipt_id"]):
                    current = await self._reference_preview(conn, UUID(item["receipt_id"]), scope)
                    reference = item["reference"]
                    if current.review_revision != reference["review_revision"] or current.document_sha256 != reference["document_sha256"]:
                        raise ValueError("reference_changed_before_run_creation")
                    if not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM benchmark_references WHERE id=$1 AND receipt_id=$2 AND document_sha256=$3)", UUID(reference["id"]), UUID(item["receipt_id"]), current.document_sha256):
                        raise ValueError("reference_not_qualified")
            row = await conn.fetchrow("""INSERT INTO benchmark_runs
                (model_ids, receipt_ids, dataset_fingerprint, model_digests, request_timeout_seconds,
                 council_policy, council_absolute_threshold, council_relative_threshold, apply_council_flags,
                 prompt_version, scoring_version, experiment_manifest, owner_id)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13) RETURNING *""",
                json.dumps(model_ids),
                json.dumps([str(receipt_id) for receipt_id in receipt_ids]),
                fingerprint,
                json.dumps(model_digests),
                timeout,
                policy.value,
                absolute,
                relative,
                apply_flags, prompt_version, scoring_version,
                json.dumps(manifest) if manifest is not None else None, owner_id)
            await conn.executemany("INSERT INTO benchmark_tasks (run_id, model_id, receipt_id) VALUES ($1,$2,$3)", [(row["id"], m, r) for m in model_ids for r in receipt_ids])
            await conn.executemany("INSERT INTO benchmark_summaries (run_id, model_id, model_digest) VALUES ($1,$2,$3)", [(row["id"], m, model_digests.get(m)) for m in model_ids])
        return self._run(row)

    async def run(self, run_id: UUID, *, scope: AccessScope | None = None) -> BenchmarkRun | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM benchmark_runs WHERE id=$1 AND ($2 OR owner_id=$3)", run_id, scope is None or scope.can_see_all, scope.user_id if scope else None)
        return self._run(row) if row else None

    async def runs(self, *, scope: AccessScope) -> list[BenchmarkRun]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM benchmark_runs WHERE ($1 OR owner_id=$2) ORDER BY created_at DESC, id DESC", scope.can_see_all, scope.user_id)
        return [self._run(r) for r in rows]

    async def task_counts(self, run_id: UUID) -> dict[str, int]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT status, count(*) AS n FROM benchmark_tasks WHERE run_id=$1 GROUP BY status", run_id)
            terminal = await conn.fetchval("SELECT COALESCE(sum(completed), 0) FROM benchmark_summaries WHERE run_id=$1", run_id)
        counts = {str(r["status"]): int(r["n"]) for r in rows}
        return {**{name: counts.get(name, 0) for name in ("queued", "running", "waiting_for_model", "retrying")}, "terminal": int(terminal)}

    async def summaries(self, run_id: UUID) -> list[BenchmarkSummary]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""SELECT s.*,r.scoring_version,
                (SELECT count(*) FROM benchmark_results b WHERE b.run_id=s.run_id
                 AND b.model_id=s.model_id AND b.payload->>'full_receipt_correct'='true') AS full_correct
                FROM benchmark_summaries s JOIN benchmark_runs r ON r.id=s.run_id
                WHERE s.run_id=$1 ORDER BY s.model_id""", run_id)
        return [BenchmarkSummary(
            all_case_average_score=(r["total_score"] / r["completed"] if r["completed"] and r["scoring_version"] == "4" else None),
            full_receipt_correct_rate=(r["full_correct"] / r["completed"] if r["completed"] and r["scoring_version"] == "4" else None),
            model_id=r["model_id"], model_digest=r["model_digest"], completed=r["completed"], succeeded=r["succeeded"], failed=r["failed"], average_score=(r["total_score"] / r["succeeded"] if r["succeeded"] else None), average_confidence=(r["total_confidence"] / r["succeeded"] if r["succeeded"] else None), average_attempts=(r["total_attempts"] / r["completed"] if r["completed"] else None), average_latency_ms=(r["total_latency_ms"] / r["completed"] if r["completed"] else None), council_candidates=r["council_candidates"], council_findings=r["council_findings"]) for r in rows]

    async def lease_task(self) -> Mapping[str, Any] | None:
        now = datetime.now(UTC)
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow("""SELECT * FROM benchmark_tasks WHERE
                (status IN ('queued','retrying','waiting_for_model') AND (retry_at IS NULL OR retry_at <= $1))
                OR (status='running' AND leased_until < $1)
                ORDER BY run_id, receipt_id FOR UPDATE SKIP LOCKED LIMIT 1""", now)
            if not row:
                return None
            await conn.execute("UPDATE benchmark_tasks SET status='running', attempts=attempts+1, started_at=$2, leased_until=$3 WHERE run_id=$1 AND model_id=$4 AND receipt_id=$5", row["run_id"], now, now + timedelta(seconds=900), row["model_id"], row["receipt_id"])
            return dict(row)

    async def retry_task(self, task: Mapping[str, Any], waiting: bool = False) -> None:
        delay = min(5 * 2 ** max(0, int(task["attempts"])), 300)
        async with self._pool.acquire() as conn:
            await conn.execute("UPDATE benchmark_tasks SET status=$1, leased_until=NULL, retry_at=$2 WHERE run_id=$3 AND model_id=$4 AND receipt_id=$5", "waiting_for_model" if waiting else "retrying", datetime.now(UTC) + timedelta(seconds=delay), task["run_id"], task["model_id"], task["receipt_id"])

    async def finish_task(self, task: Mapping[str, Any], *, success: bool, score: float = 0, confidence: int = 0, attempts: int = 1, latency_ms: float = 0, result: dict[str, Any] | None = None) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            deleted = await conn.execute("DELETE FROM benchmark_tasks WHERE run_id=$1 AND model_id=$2 AND receipt_id=$3", task["run_id"], task["model_id"], task["receipt_id"])
            if deleted == "DELETE 0":
                return  # A recovered lease must not count the same task twice.
            if result is not None:
                await conn.execute(
                    "INSERT INTO benchmark_results (run_id, model_id, receipt_id, success, payload) "
                    "VALUES ($1,$2,$3,$4,$5) ON CONFLICT (run_id, model_id, receipt_id) DO NOTHING",
                    task["run_id"], task["model_id"], task["receipt_id"], success, json.dumps(result))
            await conn.execute("""UPDATE benchmark_summaries SET completed=completed+1, succeeded=succeeded+$1, failed=failed+$2, total_score=total_score+$3, total_confidence=total_confidence+$4, total_attempts=total_attempts+$5, total_latency_ms=total_latency_ms+$6 WHERE run_id=$7 AND model_id=$8""", int(success), int(not success), score, confidence, attempts, latency_ms, task["run_id"], task["model_id"])
            await conn.execute("""UPDATE benchmark_runs SET status='completed', completed_at=now()
                WHERE id=$1 AND NOT EXISTS (SELECT 1 FROM benchmark_tasks WHERE run_id=$1)""", task["run_id"])

    @staticmethod
    def _run(row: Mapping[str, Any]) -> BenchmarkRun:
        model_ids = json.loads(row["model_ids"]) if isinstance(row["model_ids"], str) else row["model_ids"]
        receipt_ids = (
            json.loads(row["receipt_ids"])
            if isinstance(row["receipt_ids"], str)
            else row["receipt_ids"]
        )
        manifest = row.get("experiment_manifest")
        if isinstance(manifest, str):
            manifest = json.loads(manifest)
        experiment = {k: v for k, v in manifest.items() if k not in {"inputs", "tags"}} if manifest else None
        return BenchmarkRun(experiment=experiment, id=row["id"], status=row["status"], model_ids=list(model_ids), receipt_ids=list(receipt_ids), dataset_fingerprint=row["dataset_fingerprint"], prompt_version=row["prompt_version"], scoring_version=row["scoring_version"], council_policy=row["council_policy"], apply_council_flags=row["apply_council_flags"], created_at=row["created_at"], completed_at=row["completed_at"])

    async def manifest(self, run_id: UUID) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            value = await conn.fetchval("SELECT experiment_manifest FROM benchmark_runs WHERE id=$1", run_id)
        if isinstance(value, str):
            value = json.loads(value)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("Invalid benchmark manifest")
        return dict(value)

    async def results(self, run_id: UUID, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM benchmark_results WHERE run_id=$1 ORDER BY model_id, receipt_id LIMIT $2 OFFSET $3",
                run_id, limit, offset)
        return [{"model_id": row["model_id"], "receipt_id": str(row["receipt_id"]),
                 "success": row["success"],
                 **(json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"])}
                for row in rows]

    async def _reference_preview(
        self, conn: asyncpg.Connection, receipt_id: UUID, scope: AccessScope,
    ) -> ReferencePreview:
        from ...model.db.image import ImageRow
        from ...model.extraction import EXTRACTION_SCHEMA_VERSION
        from ...service.experiment_execution import image_digest
        from ...service.experiment_scoring import DETAIL_SCORING_VERSION, OBSERVABLE_FIELDS
        from ...service.receipt_review import (
            evaluate_receipt,
            material_document,
            persistent_findings,
            receipt_from_draft,
            validate_acknowledgements,
        )
        assert self._receipts is not None
        row = await self._receipts._lock_review_receipt(conn, receipt_id, scope.user_id, scope.can_see_all)
        if row is None:
            raise ValueError('Receipt not found')
        current = self._receipts._receipt_row_from_record(row)
        if not current.verified or current.status != 'verified' or current.verification_source != 'human':
            raise ValueError('Current human verification required')
        # A surviving human flag after a clean edit is prerequisite provenance, not fresh
        # consent. The separate immutable qualification below attests the CURRENT state.
        if not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM receipt_review_revisions WHERE receipt_id=$1 AND event_kind='verified' AND actor_user_id IS NOT NULL)", receipt_id):
            raise ValueError('Ordinary human verification provenance required')
        document = await self._receipts._review_document(conn, row)
        await self._receipts._validate_review_sources(conn, current.image_id, current.field_sources, current.overlap_decisions)
        originals = await self._receipts._review_originals(conn, current.image_id)
        evaluation = evaluate_receipt(document, current.amount_policy, current.field_sources,
                                      current.overlap_decisions, originals,
                                      retained_findings=persistent_findings(current.findings))
        validate_acknowledgements(evaluation, current.acknowledgements)
        image_row = await conn.fetchrow('SELECT * FROM images WHERE id=$1', current.image_id)
        if image_row is None or not originals:
            raise ValueError('original_integrity_failed')
        image = ImageRow.model_validate(dict(image_row))
        durable = {part['photo_index']: part['source_sha256'] for part in await conn.fetch('SELECT photo_index,source_sha256 FROM image_source_parts WHERE image_id=$1 ORDER BY photo_index', current.image_id)}
        sources = []
        for part, original in zip(image.parts(), originals, strict=True):
            if not original.available or not original.sha256_matches:
                raise ValueError('original_integrity_failed')
            actual_hash = await asyncio.to_thread(image_digest, Path(part.image_path))
            if original.photo_index in durable and durable[original.photo_index] != actual_hash:
                raise ValueError('original_integrity_failed')
            sources.append({'part_id': str(original.part_id) if original.part_id else None,
                            'photo_index': original.photo_index,
                            'path': part.image_path,
                            'sha256': actual_hash})
        expected = receipt_from_draft(document).model_dump(mode='json')
        # Snapshot original NUMERIC observations before public float-compatible adapters.
        items = await conn.fetch('SELECT quantity FROM line_items WHERE receipt_id=$1 ORDER BY position,id', receipt_id)
        taxes = await conn.fetch('SELECT rate FROM taxes WHERE receipt_id=$1 ORDER BY position,id', receipt_id)
        for item, raw in zip(expected['line_items'], items, strict=True):
            item['quantity'] = str(raw['quantity'])
        for tax, raw in zip(expected['taxes'], taxes, strict=True):
            tax['rate'] = str(raw['rate']) if raw['rate'] is not None else None
        material = material_document(document)
        material['line_items'] = [{k: item[k] for k in ('description','quantity','unit_price','total_price')}
                                  for item in expected['line_items']]
        material['taxes'] = expected['taxes']
        # A default value is ambiguous only without retained observation evidence.
        # Nullable raw drafts distinguish explicit zero/EUR from schema-filled defaults.
        observed = {path.split('.')[0] for path, refs in current.field_sources.items() if refs}
        raw_draft = await conn.fetchval(
            "SELECT envelope->'draft' FROM receipt_extractions WHERE id=$1 AND image_id=$2",
            current.extraction_id, current.image_id,
        )
        if isinstance(raw_draft, str):
            raw_draft = json.loads(raw_draft)
        if isinstance(raw_draft, dict):
            observed.update(key for key, value in raw_draft.items() if value is not None)
        if current.amount_policy.tax_information == 'explicit_zero':
            observed.add('tax_total')
        # Unknown legacy defaults remain excluded. The final union is server-derived;
        # confirmation cannot remove known observations, and its hash binds the mask.
        mandatory = {'merchant_name', 'date', 'line_items', 'subtotal', 'total'}
        defaults = {'currency': 'EUR', 'payment_method': 'unknown', 'discount_total': '0', 'tax_total': '0', 'taxes': []}
        scored = [key for key in OBSERVABLE_FIELDS if key in mandatory or key in observed or
                  (material.get(key) is not None and key not in defaults) or
                  (key in defaults and material.get(key) != defaults[key])]
        def digest(value: Any) -> str:
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        identity = {'version': 'gold-1', 'material': material,
                    'sources': [{k: source[k] for k in ('part_id','photo_index','sha256')} for source in sources],
                    'schema_version': EXTRACTION_SCHEMA_VERSION, 'scored_fields': scored}
        return ReferencePreview(receipt_identity=receipt_id, owner_id=current.user_id,
                                review_revision=current.review_revision, document_sha256=digest(identity),
                                expected_sha256=digest(expected), expected=expected, scored_fields=scored,
                                sources=sources, schema_version=EXTRACTION_SCHEMA_VERSION,
                                scoring_version=DETAIL_SCORING_VERSION,
                                findings=[f.model_dump(mode='json') for f in evaluation.findings],
                                acknowledgements=[a.model_dump(mode='json') for a in current.acknowledgements])

    async def reference_preview(self, receipt_id: UUID, *, scope: AccessScope) -> ReferencePreview:
        async with self._pool.acquire() as conn, conn.transaction():
            return await self._reference_preview(conn, receipt_id, scope)

    @staticmethod
    def _reference(row: Mapping[str, Any]) -> GoldSnapshot:
        values = dict(row)
        for key in ('expected', 'scored_fields', 'sources', 'findings', 'acknowledgements'):
            if isinstance(values[key], str):
                values[key] = json.loads(values[key])
        return GoldSnapshot.model_validate(values)

    async def qualify_reference(self, receipt_id: UUID, expected_review_revision: int,
                                expected_document_sha256: str, *, scope: AccessScope) -> GoldSnapshot:
        async with self._pool.acquire() as conn, conn.transaction():
            preview = await self._reference_preview(conn, receipt_id, scope)
            if preview.review_revision != expected_review_revision or preview.document_sha256 != expected_document_sha256:
                raise ValueError('reference_confirmation_conflict')
            saved = await conn.fetchrow('''INSERT INTO benchmark_references
                (receipt_id,owner_id,qualified_by,receipt_identity,owner_identity,actor_identity,
                 review_revision,document_sha256,expected_sha256,expected,scored_fields,sources,
                 schema_version,scoring_version,findings,acknowledgements)
                VALUES($1,$2,$3,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                ON CONFLICT(receipt_identity,review_revision,document_sha256) DO NOTHING RETURNING *''',
                receipt_id, preview.owner_id, scope.user_id, preview.review_revision,
                preview.document_sha256, preview.expected_sha256, json.dumps(preview.expected),
                json.dumps(preview.scored_fields), json.dumps(preview.sources), preview.schema_version,
                preview.scoring_version, json.dumps(preview.findings), json.dumps(preview.acknowledgements))
            if saved is None:
                saved = await conn.fetchrow('SELECT * FROM benchmark_references WHERE receipt_identity=$1 AND review_revision=$2 AND document_sha256=$3',
                                            receipt_id, expected_review_revision, expected_document_sha256)
            assert saved is not None
            return self._reference(saved)

    async def current_reference(self, receipt_id: UUID, *, scope: AccessScope) -> GoldSnapshot:
        async with self._pool.acquire() as conn, conn.transaction():
            preview = await self._reference_preview(conn, receipt_id, scope)
            row = await conn.fetchrow('''SELECT * FROM benchmark_references WHERE receipt_id=$1
                AND review_revision=$2 AND document_sha256=$3 AND ($4 OR owner_id=$5)''',
                receipt_id, preview.review_revision, preview.document_sha256, scope.can_see_all, scope.user_id)
            if row is None:
                raise ValueError('Receipt has no current explicitly qualified reference')
            return self._reference(row)

    async def reference_candidates(
        self, *, scope: AccessScope, category: str | None, max_source_confidence: int | None,
    ) -> list[UUID]:
        """Scoped qualified candidates; current eligibility is checked before limiting."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch('''SELECT r.id FROM receipts r
                WHERE ($1 OR r.user_id=$2) AND r.verified AND r.status='verified'
                AND ($3::text IS NULL OR r.category=$3)
                AND ($4::integer IS NULL OR r.confidence<=$4)
                AND EXISTS(SELECT 1 FROM benchmark_references b WHERE b.receipt_id=r.id
                    AND ($1 OR b.owner_id=$2))
                ORDER BY r.date DESC,r.id DESC''', scope.can_see_all, scope.user_id,
                category, max_source_confidence)
        return [row['id'] for row in rows]
