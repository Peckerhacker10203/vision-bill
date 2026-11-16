"""Real PostgreSQL 18: migrations, queue locks, fencing and transaction rollback.

Each test owns a new database inside one isolated, disposable Docker container.
No operator connection settings are used. Run explicitly with -m e2e.
"""

import asyncio
import importlib
import os
import secrets
import subprocess
import time
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from vision_bill.config import PGSettings
from vision_bill.model.db.image import ImagePart
from vision_bill.model.extraction import InferenceEnvelope, InferenceTelemetry, PipelineProvenance
from vision_bill.model.receipt import Receipt
from vision_bill.model.review import ReviewEvaluation
from vision_bill.model.workflow import AccessScope, AttemptOutcome, RetryRequest, SubmissionCreate

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def postgres():
    name = "vision-bill-task7-" + secrets.token_hex(6)
    password = secrets.token_hex(24)
    subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "-e",
            "POSTGRES_USER=workflow_test",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "-e",
            "POSTGRES_DB=workflow_admin",
            "-p",
            "127.0.0.1::5432",
            "postgres:18",
        ],
        check=True,
        capture_output=True,
    )
    try:
        port = int(
            subprocess.check_output(["docker", "port", name, "5432/tcp"], text=True)
            .strip()
            .rsplit(":", 1)[1]
        )
        for _ in range(100):
            if (
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        name,
                        "pg_isready",
                        "-h",
                        "127.0.0.1",
                        "-U",
                        "workflow_test",
                        "-d",
                        "workflow_admin",
                    ],
                    check=False,
                    capture_output=True,
                ).returncode
                == 0
            ):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("isolated PostgreSQL not ready")
        yield PGSettings(
            user="workflow_test",
            password=password,
            host="127.0.0.1",
            port=port,
            db="workflow_admin",
        )
    finally:
        subprocess.run(["docker", "rm", "--force", name], check=True, capture_output=True)


def migrate(settings, revision="head", direction="upgrade", check=True):
    env = {
        **os.environ,
        "PG__USER": settings.user,
        "PG__PASSWORD": settings.password,
        "PG__HOST": settings.host,
        "PG__PORT": str(settings.port),
        "PG__DB": settings.db,
        "AUTH__SECRET_KEY": secrets.token_hex(32),
        "VISION_BILL_CONFIG_PATH": "/tmp/vb-task7-no-config",
    }
    return subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "alembic", direction, revision],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.fixture
async def db(postgres):
    admin = await asyncpg.connect(postgres.pg_dsn)
    dbname = "case_" + secrets.token_hex(8)
    await admin.execute(f'CREATE DATABASE "{dbname}"')
    settings = postgres.model_copy(update={"db": dbname})
    try:
        yield settings
    finally:
        await admin.execute(f'DROP DATABASE "{dbname}" WITH (FORCE)')
        await admin.close()


@pytest.fixture
async def workflow_db(db):
    migrate(db)
    module = importlib.import_module("vision_bill.provider.db.workflow_db")
    provider = module.WorkflowDB(db)
    await provider.init_db()
    owner = uuid4()
    async with provider.pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users(id, username, hashed_password) VALUES($1,$2,'test')",
            owner,
            str(owner),
        )
    try:
        yield provider, AccessScope(user_id=owner)
    finally:
        await provider.destroy_db()


def submission():
    return SubmissionCreate(
        id=uuid4(),
        parts=[ImagePart(image_path="/private/original.png")],
        source_manifest=[{"photo_index": 0, "source_sha256": "a" * 64}],
        requested_options={},
        effective_options={},
        request_fingerprint="same-request",
    )


async def enqueue(w, scope):
    return (await w.enqueue_submission(submission(), scope=scope, idempotency_key=None)).submission


def complete(*, total="90071992547.91"):
    receipt = Receipt(
        confidence=90,
        merchant_name="Merchant",
        date="2026-01-01",
        currency="EUR",
        line_items=[
            {"description": "item", "quantity": 1, "unit_price": total, "total_price": total}
        ],
        subtotal=total,
        total=total,
        taxes=[{"name": "VAT", "amount": "0.01"}],
    )
    return AttemptOutcome(
        kind="complete",
        envelope=InferenceEnvelope(
            receipt=receipt,
            provenance=PipelineProvenance(
                requested_pipeline="direct_vision",
                used_pipeline="direct_vision",
                source_image_count=1,
                ordered_source_sha256=["a" * 64],
                model_id="test",
            ),
            telemetry=InferenceTelemetry(elapsed_ms=1, max_output_tokens=8192),
        ),
        review=ReviewEvaluation(can_finalize=True, can_auto_verify=False),
    )


async def test_bootstrap_and_empty_downgrade(db):
    migrate(db)
    conn = await asyncpg.connect(db.pg_dsn)
    assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0013"
    await conn.close()
    migrate(db, "0010", "downgrade")
    migrate(db)


async def test_due_only_skip_locked_concurrent_claims(workflow_db):
    w, scope = workflow_db
    first, second, future = [await enqueue(w, scope) for _ in range(3)]
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET retry_at=now()+interval '1 hour' WHERE id=$1", future.id
        )
        async with conn.transaction():
            await conn.fetchrow("SELECT id FROM images WHERE id=$1 FOR UPDATE", first.id)
            claimed = await asyncio.wait_for(w.claim_next(), timeout=2)
            assert claimed.image.id == second.id
    claimed = await w.claim_next()
    assert claimed.image.id == first.id
    assert await w.claim_next() is None


async def test_concurrent_claims_unique(workflow_db):
    w, scope = workflow_db
    ids = {(await enqueue(w, scope)).id for _ in range(5)}
    claims = await asyncio.gather(*(w.claim_next() for _ in range(8)))
    assert {c.image.id for c in claims if c is not None} == ids
    assert sum(c is not None for c in claims) == 5


async def test_expiry_tokens_generation_and_three_attempt_limit(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim1 = await w.claim_next()
    assert (
        await w.finish_attempt(
            claim1.model_copy(update={"claim_token": uuid4()}), AttemptOutcome(kind="retry")
        )
    ).stale
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET lease_expires_at=now()-interval '1 second' WHERE id=$1", image.id
        )
    assert (await w.finish_attempt(claim1, complete())).stale
    claim2 = await w.claim_next()
    assert claim2.attempt_number == 2
    assert (await w.finish_attempt(claim1, complete())).stale
    await w.finish_attempt(claim2, AttemptOutcome(kind="retry", retry_delay_seconds=0))
    claim3 = await w.claim_next()
    assert claim3.attempt_number == 3
    result = await w.finish_attempt(claim3, AttemptOutcome(kind="retry", retry_delay_seconds=0))
    assert result.submission.status == "failed" and result.submission.attempt_count == 3
    assert await w.claim_next() is None
    retried = await w.retry_submission(
        image.id, RetryRequest(expected_generation=1, request_key="retry"), scope=scope
    )
    assert retried.generation == 2
    assert (await w.finish_attempt(claim3, complete())).stale


async def test_upload_idempotency_is_owner_scoped_and_conflicts(workflow_db):
    w, scope = workflow_db
    request = submission()
    results = await asyncio.gather(
        *(w.enqueue_submission(request, scope=scope, idempotency_key="upload") for _ in range(3))
    )
    assert sum(r.created for r in results) == 1
    assert len({r.submission.id for r in results}) == 1
    altered = request.model_copy(update={"request_fingerprint": "changed"})
    with pytest.raises(ValueError, match="conflict"):
        await w.enqueue_submission(altered, scope=scope, idempotency_key="upload")
    async with w.pool.acquire() as conn:
        other = await conn.fetchval(
            "INSERT INTO users(username,hashed_password) VALUES('other','test') RETURNING id"
        )
    different = await w.enqueue_submission(
        submission(), scope=AccessScope(user_id=other), idempotency_key="upload"
    )
    assert different.created


async def test_retry_key_replay_survives_later_generations(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    await w.finish_attempt(await w.claim_next(), AttemptOutcome(kind="terminal"))
    request = RetryRequest(expected_generation=1, request_key="r1")
    first = await w.retry_submission(image.id, request, scope=scope)
    await w.finish_attempt(await w.claim_next(), AttemptOutcome(kind="terminal"))
    await w.retry_submission(
        image.id, RetryRequest(expected_generation=2, request_key="r2"), scope=scope
    )
    replay = await w.retry_submission(image.id, request, scope=scope)
    assert replay == first and replay.generation == 2
    with pytest.raises(ValueError, match="conflict"):
        await w.retry_submission(
            image.id, request.model_copy(update={"expected_generation": 3}), scope=scope
        )
    assert await w.retry_submission(image.id, request, scope=AccessScope(user_id=uuid4())) is None


async def test_finish_atomic_decimal_and_immutable_history(workflow_db, db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    results = await asyncio.gather(
        w.finish_attempt(claim, complete()), w.finish_attempt(claim, complete())
    )
    assert sum(r.stale for r in results) == 1
    row = next(r.receipt for r in results if not r.stale)
    assert row.total == Decimal("90071992547.91")
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval("SELECT total_price FROM line_items WHERE receipt_id=$1", row.id)
            == row.total
        )
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 1
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 1
        with pytest.raises(asyncpg.RaiseError):
            await conn.execute("UPDATE receipt_review_revisions SET event_kind='tamper'")
    with pytest.raises(ValueError, match="receipt"):
        await w.retry_submission(
            image.id, RetryRequest(expected_generation=1, request_key="never"), scope=scope
        )
    assert migrate(db, "0010", "downgrade", check=False).returncode != 0
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0013"
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 1


async def test_child_failure_rolls_back_everything(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    outcome = complete()
    outcome.envelope.receipt.line_items[0].description = "x" * 256
    with pytest.raises(asyncpg.StringDataRightTruncationError):
        await w.finish_attempt(claim, outcome)
    async with w.pool.acquire() as conn:
        for table in ("receipts", "receipt_extractions", "receipt_review_revisions"):
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0
        assert (
            await conn.fetchval("SELECT status FROM images WHERE id=$1", image.id) == "processing"
        )


async def test_live_lease_must_still_hold_when_sealing(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    async with w.pool.acquire() as conn:
        await conn.execute("""CREATE FUNCTION delay_child() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN PERFORM pg_sleep(0.5); RETURN NEW; END $$""")
        await conn.execute(
            "CREATE TRIGGER delay_child BEFORE INSERT ON line_items FOR EACH ROW EXECUTE FUNCTION delay_child()"
        )
        await conn.execute(
            "UPDATE images SET lease_expires_at=clock_timestamp()+interval '0.2 second' WHERE id=$1",
            image.id,
        )
    assert (await w.finish_attempt(claim, complete())).stale
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 0
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 0


def wire_page():
    from vision_bill.model.ocr import OcrResponse

    return OcrResponse(
        request_id=uuid4(),
        photo_index=0,
        source_sha256="a" * 64,
        text="Merchant",
        blocks=[],
        profile_id="profile",
        runtime_fingerprint="runtime",
        working_image={
            "view_id": uuid4(),
            "width": 2,
            "height": 3,
            "sha256": "b" * 64,
            "png_base64": "private-image-bytes",
        },
    )


async def saved_view(w, scope, image, page):
    from vision_bill.model.ocr import WorkingImageMetadata
    from vision_bill.model.workflow import PersistedWorkingView

    parts = await w.get_source_parts(image.id, scope=scope)
    assert len(parts) == 1
    return PersistedWorkingView(
        part_id=parts[0].part_id,
        photo_index=0,
        source_sha256="a" * 64,
        artifact_path="/private/saved-working.png",
        metadata=WorkingImageMetadata.model_validate(page.working_image.model_dump()),
    )


async def test_checkpoints_bind_real_parts_profile_and_fence(workflow_db):
    from vision_bill.model.workflow import OcrCheckpoint

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    checkpoint = OcrCheckpoint(
        ordered_source_sha256=["a" * 64],
        profile_id="profile",
        runtime_fingerprint="runtime",
        pages=[page],
        persisted_views=[view],
    )
    assert await w.save_ocr_checkpoint(claim, checkpoint)
    assert await w.save_ocr_checkpoint(claim, checkpoint)
    with pytest.raises(ValueError, match="profile"):
        await w.save_ocr_checkpoint(claim, checkpoint.model_copy(update={"profile_id": "changed"}))
    with pytest.raises(ValueError, match="real source"):
        await w.save_ocr_checkpoint(
            claim,
            checkpoint.model_copy(
                update={"persisted_views": [view.model_copy(update={"part_id": uuid4()})]}
            ),
        )
    assert not await w.save_ocr_checkpoint(
        claim.model_copy(update={"claim_token": uuid4()}), checkpoint
    )
    async with w.pool.acquire() as conn:
        raw = await conn.fetchval("SELECT ocr_checkpoint::text FROM images WHERE id=$1", image.id)
        assert "png_base64" not in raw and "private-image-bytes" not in raw
    assert await w.get_source_parts(image.id, scope=AccessScope(user_id=uuid4())) == []


async def test_finish_requires_saved_artifacts_and_real_source_refs(workflow_db):
    from vision_bill.model.extraction import DerivedInputView
    from vision_bill.model.ocr import SourceRef

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    outcome = complete()
    outcome.envelope.ocr_pages = [page]
    outcome.envelope.input_views = [
        DerivedInputView(photo_index=0, source_sha256="a" * 64, working_image=page.working_image)
    ]
    with pytest.raises(ValueError, match="saved working artifact"):
        await w.finish_attempt(claim, outcome)
    outcome.persisted_views = [view]
    ref = SourceRef(part_id=uuid4(), photo_index=0, source_sha256="a" * 64)
    outcome.envelope.field_sources = {"total": [ref]}
    with pytest.raises(ValueError, match="real source part"):
        await w.finish_attempt(claim, outcome)
    ref.part_id = view.part_id
    ref.view_id = view.metadata.view_id
    ref.coordinate_space = "working"
    assert not (await w.finish_attempt(claim, outcome)).stale
    async with w.pool.acquire() as conn:
        raw = await conn.fetchval("SELECT envelope::text FROM receipt_extractions")
        assert "png_base64" not in raw and "private-image-bytes" not in raw
        assert str(view.part_id) in raw and str(view.metadata.view_id) in raw


async def test_partial_draft_human_edits_and_receipt_link_are_preserved(workflow_db):
    from vision_bill.model.extraction import ExtractionDraft
    from vision_bill.model.review import ReviewFinding

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    envelope = complete().envelope
    envelope.receipt = None
    envelope.draft = ExtractionDraft(merchant_name="uncertain", total="1.23")
    finding = ReviewFinding(
        id="missing", code="missing_value", severity="review", field_path="date"
    )
    review = ReviewEvaluation(findings=[finding], can_finalize=False, can_auto_verify=False)
    result = await w.finish_attempt(
        claim, AttemptOutcome(kind="partial", envelope=envelope, review=review)
    )
    assert result.submission.status == "needs_review" and result.submission.draft_available
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT findings->0->>'id' FROM receipt_drafts WHERE image_id=$1", image.id
            )
            == "missing"
        )
        await conn.execute(
            "UPDATE receipt_drafts SET human_edited=TRUE,revision=2 WHERE image_id=$1", image.id
        )
    with pytest.raises(ValueError, match="edited draft"):
        await w.retry_submission(
            image.id, RetryRequest(expected_generation=1, request_key="human"), scope=scope
        )
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT document->>'total' FROM receipt_drafts WHERE image_id=$1", image.id
            )
            == "1.23"
        )


async def test_upgrade_0010_preserves_legacy_content_paths_and_ambiguity(db):
    migrate(db, "0010")
    conn = await asyncpg.connect(db.pg_dsn)
    ids = []
    for state in ("pending", "failed", "processing", "analyzed", "failed", "processing"):
        ids.append(
            await conn.fetchval(
                "INSERT INTO images(status,error,image_path) VALUES($1,'legacy error','/legacy/path') RETURNING id",
                state,
            )
        )
    receipt_id = await conn.fetchval(
        """INSERT INTO receipts(merchant_name,date,total,status,verified,image_id)
        VALUES('old','2020-01-01',12345678901.23,'verified',TRUE,$1) RETURNING id""",
        ids[4],
    )
    # Simulate historical duplicates explicitly preserved by migration 0002.
    await conn.execute("DROP TRIGGER receipts_image_id_unique ON receipts")
    await conn.execute("DROP INDEX ux_receipts_image_id")
    duplicate = await conn.fetchval(
        """INSERT INTO receipts(merchant_name,date,total,image_id)
        VALUES('duplicate','2020-01-01',9.87,$1) RETURNING id""",
        ids[4],
    )
    reverse = await conn.fetchval(
        "INSERT INTO receipts(merchant_name,date,total) VALUES('reverse','2020-01-01',2.34) RETURNING id"
    )
    await conn.execute("UPDATE images SET receipt_id=$2 WHERE id=$1", ids[5], reverse)
    await conn.close()
    migrate(db)
    conn = await asyncpg.connect(db.pg_dsn)
    states = [await conn.fetchval("SELECT status FROM images WHERE id=$1", image) for image in ids]
    assert states == ["pending", "pending", "pending", "analyzed", "analyzed", "analyzed"]
    receipt = await conn.fetchrow("SELECT * FROM receipts WHERE id=$1", receipt_id)
    assert (
        receipt["total"] == Decimal("12345678901.23")
        and receipt["verified"]
        and receipt["verification_source"] == "legacy"
    )
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM receipts WHERE id=ANY($1::uuid[])",
            [receipt_id, duplicate, reverse],
        )
        == 3
    )
    for image in ids:
        row = await conn.fetchrow("SELECT * FROM images WHERE id=$1", image)
        assert row["image_path"] == "/legacy/path" and row["user_id"] is None
        assert row["attempt_count"] == 0 and row["max_attempts"] == 3
    assert (
        await conn.fetchval("SELECT legacy_events->0->>'error' FROM images WHERE id=$1", ids[1])
        == "legacy error"
    )
    assert (
        await conn.fetchval(
            "SELECT legacy_events->0->>'receipt_count' FROM images WHERE id=$1", ids[4]
        )
        == "2"
    )
    await conn.close()


async def test_bind_legacy_parts_is_fenced_stable_and_cannot_rewrite(workflow_db):
    from vision_bill.model import workflow

    w, scope = workflow_db
    async with w.pool.acquire() as conn:
        image_id = await conn.fetchval(
            "INSERT INTO images(user_id,image_path) VALUES($1,'/legacy/original') RETURNING id",
            scope.user_id,
        )
    claim = await w.claim_next()
    assert hasattr(workflow, "SourcePartBinding"), "typed binding contract missing"
    binding = workflow.SourcePartBinding(
        photo_index=0, source_sha256="c" * 64, image_path="/legacy/original"
    )
    parts = await w.bind_source_parts(claim, [binding])
    assert parts[0].image_path == "/legacy/original" and parts[0].source_sha256 == "c" * 64
    assert (await w.bind_source_parts(claim, [binding])) == parts
    assert (
        await w.bind_source_parts(claim.model_copy(update={"claim_token": uuid4()}), [binding])
        is None
    )
    for key, value in (("source_sha256", "d" * 64), ("image_path", "/wrong"), ("photo_index", 1)):
        with pytest.raises(ValueError, match="source"):
            await w.bind_source_parts(claim, [binding.model_copy(update={key: value})])
    assert await w.get_source_parts(image_id, scope=scope) == parts


async def test_legacy_image_mutators_cannot_bypass_durable_fence(workflow_db):
    from vision_bill.provider.db.image_db import ImageDB

    w, scope = workflow_db
    image = await enqueue(w, scope)
    legacy = ImageDB(w._settings)
    legacy._pool = w.pool
    assert await legacy.claim_for_analysis(image.id) is None
    claim = await w.claim_next()
    await legacy.mark_failed(image.id, "old worker")
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval("SELECT status FROM images WHERE id=$1", image.id) == "processing"
        )
    assert not (await w.finish_attempt(claim, complete())).stale


async def test_expired_third_lease_seals_terminal_without_fourth_claim(workflow_db):
    w, scope = workflow_db
    image = await enqueue(w, scope)
    for attempt in range(1, 4):
        claim = await w.claim_next()
        assert claim.attempt_number == attempt
        async with w.pool.acquire() as conn:
            await conn.execute(
                "UPDATE images SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
                image.id,
            )
    assert await w.claim_next() is None
    async with w.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM images WHERE id=$1", image.id)
        assert row["status"] == "failed" and row["attempt_count"] == 3
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 3


async def test_worker_backoff_configuration_is_honored(workflow_db):
    from vision_bill.config import WorkerSettings
    from vision_bill.provider.db.workflow_db import WorkflowDB

    w, scope = workflow_db
    configured = WorkflowDB(
        w._settings,
        worker_settings=WorkerSettings(retry_backoff_seconds=(3, 7), retry_backoff_cap_seconds=5),
    )
    configured._pool = w.pool
    await enqueue(configured, scope)
    claim = await configured.claim_next()
    result = await configured.finish_attempt(claim, AttemptOutcome(kind="retry"))
    async with w.pool.acquire() as conn:
        delay = await conn.fetchval(
            "SELECT EXTRACT(EPOCH FROM retry_at-clock_timestamp()) FROM images WHERE id=$1",
            result.submission.id,
        )
        assert Decimal(2) < delay <= Decimal(3)
        await conn.execute(
            "UPDATE images SET retry_at=clock_timestamp() WHERE id=$1", result.submission.id
        )
    result = await configured.finish_attempt(
        await configured.claim_next(), AttemptOutcome(kind="retry", retry_delay_seconds=99)
    )
    async with w.pool.acquire() as conn:
        delay = await conn.fetchval(
            "SELECT EXTRACT(EPOCH FROM retry_at-clock_timestamp()) FROM images WHERE id=$1",
            result.submission.id,
        )
        assert Decimal(4) < delay <= Decimal(5)


async def test_sealed_extraction_cannot_be_deleted_directly(workflow_db):
    w, scope = workflow_db
    await enqueue(w, scope)
    result = await w.finish_attempt(await w.claim_next(), complete())
    async with w.pool.acquire() as conn:
        with pytest.raises(asyncpg.RaiseError):
            await conn.execute("DELETE FROM receipt_extractions")
        await conn.execute("DELETE FROM receipts WHERE id=$1", result.receipt.id)
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 0


async def test_checkpoint_read_is_fenced_and_preserves_saved_metadata(workflow_db):
    from vision_bill.model.workflow import OcrCheckpoint

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    checkpoint = OcrCheckpoint(
        ordered_source_sha256=["a" * 64],
        profile_id="profile",
        runtime_fingerprint="runtime",
        pages=[page],
        persisted_views=[view],
    )
    await w.save_ocr_checkpoint(claim, checkpoint)
    assert await w.get_ocr_checkpoint(claim) == checkpoint
    assert await w.get_ocr_checkpoint(claim.model_copy(update={"claim_token": uuid4()})) is None
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
            image.id,
        )
    assert await w.get_ocr_checkpoint(claim) is None


async def test_numeric_update_and_owner_admin_scope(workflow_db):
    from vision_bill.provider.db.receipt_db import ReceiptDB

    w, scope = workflow_db
    image = await enqueue(w, scope)
    outcome = complete()
    result = await w.finish_attempt(await w.claim_next(), outcome)
    receipts = ReceiptDB(w._settings)
    receipts._pool = w.pool
    updated = outcome.envelope.receipt.model_copy(
        update={"total": Decimal("90071992547.93"), "subtotal": Decimal("90071992547.93")}
    )
    saved = await receipts.update_receipt(result.receipt.id, updated, user_id=scope.user_id)
    assert saved.total == Decimal("90071992547.93")
    other = AccessScope(user_id=uuid4())
    assert await w.get_submission(image.id, scope=other) is None
    assert (
        await w.get_submission(image.id, scope=other.model_copy(update={"can_see_all": True}))
    ).id == image.id
    async with w.pool.acquire() as conn:
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                """INSERT INTO receipt_extractions(image_id,generation,attempt_number,outcome_kind)
                VALUES($1,1,1,'complete')""",
                image.id,
            )


async def test_overlap_metadata_cannot_smuggle_fabricated_source_refs(workflow_db):
    w, scope = workflow_db
    await enqueue(w, scope)
    outcome = complete()
    outcome.envelope.overlap_decisions = [
        {"source_refs": [{"part_id": str(uuid4()), "photo_index": 0, "source_sha256": "a" * 64}]}
    ]
    with pytest.raises(ValueError, match="real source part"):
        await w.finish_attempt(await w.claim_next(), outcome)


async def test_upload_replay_ignores_new_storage_paths_and_changed_resolved_defaults(workflow_db):
    w, scope = workflow_db
    first = submission().model_copy(
        update={"effective_options": {"pipeline": "direct_vision"}, "model_id": "first-default"}
    )
    saved = await w.enqueue_submission(first, scope=scope, idempotency_key="stable-request")
    replay = first.model_copy(
        update={
            "id": uuid4(),
            "parts": [ImagePart(image_path="/new/random/path", thumbnail_path="/new/thumb")],
            "effective_options": {"pipeline": "ocr"},
            "model_id": "new-default",
            "bypass_review": True,
        }
    )
    result = await w.enqueue_submission(replay, scope=scope, idempotency_key="stable-request")
    assert not result.created and result.submission.id == saved.submission.id
    assert result.submission.model_id == "first-default" and not result.submission.bypass_review
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT effective_options->>'pipeline' FROM images WHERE id=$1", saved.submission.id
            )
            == "direct_vision"
        )
    explicit_changed = replay.model_copy(
        update={"requested_options": {"model_id": "explicitly-other"}}
    )
    with pytest.raises(ValueError, match="conflict"):
        await w.enqueue_submission(explicit_changed, scope=scope, idempotency_key="stable-request")
