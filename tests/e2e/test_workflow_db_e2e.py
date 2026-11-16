"""Real PostgreSQL 18: migrations, queue locks, fencing and transaction rollback.

Each test owns a new database inside one isolated, disposable Docker container.
No operator connection settings are used. Run explicitly with -m e2e.
"""

import asyncio
import importlib
import os
import secrets
import subprocess
import sys
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
        [sys.executable, "-m", "alembic", direction, revision],
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
    from vision_bill.service.image_service import ImageService
    provider = module.WorkflowDB(db, original_checker=ImageService.check_originals)
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
    assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0017"
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
        assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0017"
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


async def test_upgrade_0010_preserves_precise_local_times_and_legacy_evidence(db, tmp_path):
    from datetime import time as Time
    from vision_bill.provider.db.receipt_db import ReceiptDB

    migrate(db, "0010")
    conn = await asyncpg.connect(db.pg_dsn)
    owner = await conn.fetchval("INSERT INTO users(username,hashed_password) VALUES($1,'test') RETURNING id", str(uuid4()))
    records = []
    times = [Time(14, 30), Time(14, 30, 15), Time(14, 30, 15, 100000), Time(14, 30, 0, 1)]
    for index, value in enumerate(times):
        original = tmp_path / f"legacy-{index}.png"
        original.write_bytes(f"original legacy {index}".encode())
        image = await conn.fetchval("INSERT INTO images(user_id,image_path,status) VALUES($1,$2,'analyzed') RETURNING id", owner, str(original))
        receipt = await conn.fetchval("""INSERT INTO receipts(user_id,image_id,merchant_name,date,time,subtotal,total,tax_total,currency,verified,status)
            VALUES($1,$2,'Legacy','2020-01-01',$3,12.34,12.34,1.23,'EUR',TRUE,'verified') RETURNING id""", owner, image, value)
        await conn.execute("UPDATE images SET receipt_id=$2 WHERE id=$1", image, receipt)
        for position in range(2):
            await conn.execute("""INSERT INTO line_items(receipt_id,description,original_description,quantity,unit_price,total_price,position)
                VALUES($1,$2::text,$2::text,1.2345,6.17,6.17,$3)""", receipt, f"line-{position}", position)
            await conn.execute("INSERT INTO taxes(receipt_id,name,rate,amount) VALUES($1,$2,0.1234,0.615)", receipt, f"tax-{position}")
        records.append((receipt, image, original))
    run = await conn.fetchval("""INSERT INTO benchmark_runs(status,model_ids,receipt_ids,dataset_fingerprint,prompt_version,scoring_version,experiment_manifest)
        VALUES('completed','["legacy"]','[]','frozen','old','3','{"frozen_time":"14:30:15.1"}') RETURNING id""")
    await conn.execute("""INSERT INTO benchmark_results(run_id,model_id,receipt_id,success,payload)
        VALUES($1,'legacy',$2,TRUE,'{"time":"14:30:15.1","score":0.875}')""", run, records[0][0])
    queries = [
        "SELECT id,user_id,image_id,time,subtotal,total,tax_total,currency,verified,status FROM receipts ORDER BY id",
        "SELECT id,receipt_id,description,quantity,unit_price,total_price,position FROM line_items ORDER BY receipt_id,position",
        "SELECT id,receipt_id,name,rate,amount FROM taxes ORDER BY receipt_id,id",
        "SELECT id,user_id,image_path,receipt_id FROM images ORDER BY id",
        "SELECT experiment_manifest FROM benchmark_runs WHERE id=$1",
        "SELECT payload FROM benchmark_results WHERE run_id=$1",
    ]
    before = [await conn.fetch(query, *([run] if "$1" in query else [])) for query in queries]
    await conn.close()
    migrate(db)
    conn = await asyncpg.connect(db.pg_dsn)
    try:
        assert [await conn.fetch(query, *([run] if "$1" in query else [])) for query in queries] == before
        reader = ReceiptDB(db)
        reader._pool = await asyncpg.create_pool(db.pg_dsn, min_size=1, max_size=2)
        try:
            for index, (receipt, image, original) in enumerate(records):
                detail = await reader.get_receipt_with_details(receipt, user_id=owner)
                assert detail.receipt.time == times[index].isoformat()
                assert detail.receipt.verification_source == "legacy"
                assert [item.description for item in detail.line_items] == ["line-0", "line-1"]
                assert original.read_bytes() == f"original legacy {index}".encode()
                assert await conn.fetchval("SELECT image_path FROM images WHERE id=$1", image) == str(original)
        finally:
            await reader.destroy_db()
        assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0017"
    finally:
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


@pytest.mark.parametrize("path", ["terminal_review", "checkpoint_page", "finish_page"])
@pytest.mark.parametrize("forged", ["part", "view"])
async def test_every_persisted_review_and_page_reference_is_validated(workflow_db, path, forged):
    from vision_bill.model.review import ReviewFinding
    from vision_bill.model.workflow import OcrCheckpoint

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    reference = {
        "part_id": str(uuid4()) if forged == "part" else str(view.part_id),
        "photo_index": 0,
        "source_sha256": "a" * 64,
        "view_id": str(uuid4()) if forged == "view" else str(view.metadata.view_id),
    }
    if path == "terminal_review":
        finding = ReviewFinding(
            id="forged",
            code="source",
            severity="integrity",
            field_path="total",
            source_refs=[reference],
        )
        review = ReviewEvaluation(findings=[finding], can_finalize=False, can_auto_verify=False)
        with pytest.raises(ValueError, match="source reference"):
            await w.finish_attempt(
                claim, AttemptOutcome(kind="terminal", review=review, persisted_views=[view])
            )
    else:
        page.telemetry = {"elapsed_ms": 1, "nested": {"source_refs": [reference]}}
        if path == "checkpoint_page":
            checkpoint = OcrCheckpoint(
                ordered_source_sha256=["a" * 64],
                profile_id="profile",
                runtime_fingerprint="runtime",
                pages=[page],
                persisted_views=[view],
            )
            with pytest.raises(ValueError, match="source reference"):
                await w.save_ocr_checkpoint(claim, checkpoint)
        else:
            outcome = complete()
            outcome.envelope.ocr_pages = [page]
            outcome.persisted_views = [view]
            with pytest.raises(ValueError, match="source reference"):
                await w.finish_attempt(claim, outcome)
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipt_extractions") == 0
        assert await conn.fetchval("SELECT count(*) FROM extraction_working_views") == 0
        assert (
            await conn.fetchval("SELECT ocr_checkpoint FROM images WHERE id=$1", image.id) is None
        )


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
    from vision_bill.api.receipts import ReceiptUpdate
    from vision_bill.provider.db.receipt_db import ReceiptDB
    from vision_bill.service.image_service import ImageService

    w, scope = workflow_db
    image = await enqueue(w, scope)
    outcome = complete()
    result = await w.finish_attempt(await w.claim_next(), outcome)
    receipts = ReceiptDB(w._settings, original_checker=ImageService.check_originals)
    receipts._pool = w.pool
    updated = ReceiptUpdate.model_validate(dict(outcome.envelope.receipt.model_dump(), total="90071992547.93", subtotal="90071992547.93", expected_operation_revision=1))
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


async def test_untouched_partial_drafts_allow_repeated_generations_with_monotonic_revision(
    workflow_db,
):
    from vision_bill.model.extraction import ExtractionDraft

    w, scope = workflow_db
    image = await enqueue(w, scope)
    for generation in range(1, 4):
        envelope = complete().envelope
        envelope.receipt = None
        envelope.draft = ExtractionDraft(merchant_name=f"generation {generation}")
        result = await w.finish_attempt(
            await w.claim_next(), AttemptOutcome(kind="partial", envelope=envelope)
        )
        assert result.submission.generation == generation
        async with w.pool.acquire() as conn:
            draft = await conn.fetchrow("SELECT * FROM receipt_drafts WHERE image_id=$1", image.id)
            assert draft["revision"] == generation and not draft["human_edited"]
        retried = await w.retry_submission(
            image.id,
            RetryRequest(expected_generation=generation, request_key=f"generation-{generation}"),
            scope=scope,
        )
        assert retried.generation == generation + 1


@pytest.mark.parametrize("legacy_rows", [False, True])
async def test_downgrade_preserves_exact_0010_terminal_status_contract(db, legacy_rows):
    migrate(db, "0010")
    conn = await asyncpg.connect(db.pg_dsn)
    if legacy_rows:
        await conn.execute(
            "INSERT INTO images(status,image_path) VALUES('timed_out','/legacy/timeout'),('unreadable','/legacy/unreadable')"
        )
    await conn.close()
    migrate(db)
    migrate(db, "0010", "downgrade")
    conn = await asyncpg.connect(db.pg_dsn)
    assert await conn.fetchval("SELECT version_num FROM alembic_version") == "0010"
    for state in ("pending", "processing", "analyzed", "failed", "timed_out", "unreadable"):
        row_id = await conn.fetchval("INSERT INTO images(status) VALUES($1) RETURNING id", state)
        await conn.execute("UPDATE images SET status=$2 WHERE id=$1", row_id, state)
    if legacy_rows:
        assert (
            await conn.fetchval("SELECT status FROM images WHERE image_path='/legacy/timeout'")
            == "timed_out"
        )
        assert (
            await conn.fetchval("SELECT status FROM images WHERE image_path='/legacy/unreadable'")
            == "unreadable"
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await conn.execute("INSERT INTO images(status) VALUES('needs_review')")
    await conn.close()


async def test_review_actor_deletion_nulls_only_actor_preserves_history_and_owner(workflow_db):
    w, scope = workflow_db
    await enqueue(w, scope)
    result = await w.finish_attempt(await w.claim_next(), complete())
    async with w.pool.acquire() as conn:
        actor = await conn.fetchval(
            "INSERT INTO users(username,hashed_password,is_admin) VALUES('reviewer','test',TRUE) RETURNING id"
        )
        revision = await conn.fetchval(
            """INSERT INTO receipt_review_revisions(receipt_id,operation_revision,
            review_revision,event_kind,actor_user_id,snapshot) VALUES($1,2,1,'verified',$2,'{"content":"immutable"}') RETURNING id""",
            result.receipt.id,
            actor,
        )
        for statement in (
            "actor_user_id=NULL",
            "event_kind='tampered'",
            "snapshot='{}'::jsonb",
            "operation_revision=99",
        ):
            with pytest.raises(asyncpg.RaiseError, match="immutable"):
                await conn.execute(
                    f"UPDATE receipt_review_revisions SET {statement} WHERE id=$1", revision
                )
        await conn.execute("DELETE FROM users WHERE id=$1", actor)
        history = await conn.fetchrow(
            "SELECT * FROM receipt_review_revisions WHERE id=$1", revision
        )
        assert (
            history["actor_user_id"] is None and history["snapshot"] == '{"content": "immutable"}'
        )
        assert history["operation_revision"] == 2 and history["event_kind"] == "verified"
        assert (
            await conn.fetchval("SELECT user_id FROM receipts WHERE id=$1", result.receipt.id)
            == scope.user_id
        )
        with pytest.raises(asyncpg.RaiseError, match="immutable"):
            await conn.execute(
                "UPDATE receipt_review_revisions SET snapshot='{}'::jsonb WHERE id=$1", revision
            )
        await conn.execute("DELETE FROM receipts WHERE id=$1", result.receipt.id)
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 0


@pytest.mark.parametrize("terminal_review", [False, True])
async def test_reference_validation_preserves_real_references_and_ordinary_telemetry(
    workflow_db, terminal_review
):
    from vision_bill.model.review import ReviewFinding
    from vision_bill.model.workflow import OcrCheckpoint

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    reference = {
        "part_id": str(view.part_id),
        "photo_index": 0,
        "source_sha256": "a" * 64,
        "view_id": str(view.metadata.view_id),
    }
    review = ReviewEvaluation(
        findings=[
            ReviewFinding(
                id="real",
                code="review",
                severity="review",
                field_path="total",
                source_refs=[reference],
            )
        ],
        can_finalize=False,
        can_auto_verify=False,
    )
    if terminal_review:
        result = await w.finish_attempt(
            claim, AttemptOutcome(kind="terminal", review=review, persisted_views=[view])
        )
        assert not result.stale
    else:
        page.telemetry = {
            "elapsed_ms": 12,
            "provider": {"profile": "native"},
            "reference": reference,
        }
        checkpoint = OcrCheckpoint(
            ordered_source_sha256=["a" * 64],
            profile_id="profile",
            runtime_fingerprint="runtime",
            pages=[page],
            persisted_views=[view],
        )
        assert await w.save_ocr_checkpoint(claim, checkpoint)
        outcome = complete()
        outcome.envelope.ocr_pages = [page]
        outcome.persisted_views = [view]
        assert not (await w.finish_attempt(claim, outcome)).stale


async def partial_draft(w, scope, original_path=None):
    from vision_bill.model.extraction import ExtractionDraft

    request = submission()
    if original_path is not None:
        from hashlib import sha256
        original_path.write_bytes(b"original fixture")
        request.parts[0].image_path = str(original_path)
        request.source_manifest[0]["source_sha256"] = sha256(original_path.read_bytes()).hexdigest()
    image = (await w.enqueue_submission(request, scope=scope, idempotency_key=None)).submission
    outcome = complete()
    outcome.envelope.provenance.ordered_source_sha256 = [request.source_manifest[0]["source_sha256"]]
    outcome.envelope.draft = ExtractionDraft(merchant_name="observed", total=None)
    outcome.envelope.receipt = None
    await w.finish_attempt(
        await w.claim_next(), AttemptOutcome(kind="partial", envelope=outcome.envelope)
    )
    return image


async def test_draft_null_cas_human_edit_and_owner_scope(workflow_db):
    from vision_bill.model.workflow import DraftPatch

    w, scope = workflow_db
    image = await partial_draft(w, scope)
    draft = await w.get_draft(image.id, scope=scope)
    assert draft.document.total is None and draft.revision == 1
    assert await w.get_draft(image.id, scope=AccessScope(user_id=uuid4())) is None
    patched = await w.patch_draft(
        image.id, DraftPatch(expected_revision=1, changes={"merchant_name": None}), scope=scope
    )
    assert patched.revision == 2 and patched.document.merchant_name is None
    assert (await w.get_draft(image.id, scope=scope)).document.merchant_name is None
    with pytest.raises(ValueError, match="revision"):
        await w.patch_draft(
            image.id, DraftPatch(expected_revision=1, changes={"total": "1.20"}), scope=scope
        )
    with pytest.raises(ValueError, match="edited"):
        await w.retry_submission(
            image.id, RetryRequest(expected_generation=1, request_key="retry"), scope=scope
        )
    async with w.pool.acquire() as conn:
        assert await conn.fetchval(
            "SELECT human_edited FROM receipt_drafts WHERE image_id=$1", image.id
        )


async def test_clear_checkpoint_is_fenced_and_retains_immutable_views(workflow_db):
    from vision_bill.model.workflow import OcrCheckpoint

    w, scope = workflow_db
    image = await enqueue(w, scope)
    claim = await w.claim_next()
    page = wire_page()
    view = await saved_view(w, scope, image, page)
    await w.save_ocr_checkpoint(
        claim,
        OcrCheckpoint(
            ordered_source_sha256=["a" * 64],
            profile_id="profile",
            runtime_fingerprint="runtime",
            pages=[page],
            persisted_views=[view],
        ),
    )
    assert not await w.clear_ocr_checkpoint(claim.model_copy(update={"claim_token": uuid4()}))
    assert await w.get_ocr_checkpoint(claim) is not None
    assert await w.clear_ocr_checkpoint(claim)
    assert await w.get_ocr_checkpoint(claim) is None
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM extraction_working_views") == 1


async def test_draft_finalize_missing_then_once_unverified_and_rollback(workflow_db, tmp_path):
    from vision_bill.model.review import OriginalAvailability
    from vision_bill.model.workflow import DraftFinalize, DraftPatch

    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / "original.png")
    parts = await w.get_source_parts(image.id, scope=scope)
    originals = [
        OriginalAvailability(part_id=p.part_id, available=True, sha256_matches=True) for p in parts
    ]
    request = DraftFinalize(expected_revision=1, request_key="final")
    with pytest.raises(ValueError, match="incomplete"):
        await w.finalize_draft(image.id, request, scope=scope, originals=originals)
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 0
    document = complete().envelope.receipt.model_dump(mode="json")
    await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes=document), scope=scope)
    request = DraftFinalize(expected_revision=2, request_key="final")
    with pytest.raises(ValueError, match="original"):
        await w.finalize_draft(image.id, request, scope=scope, originals=[])
    first, repeated = await asyncio.gather(
        *(w.finalize_draft(image.id, request, scope=scope, originals=originals) for _ in range(2))
    )
    assert first.id == repeated.id and not first.verified and first.status == "unverified"
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 1
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 1
        assert (
            await conn.fetchval("SELECT receipt_id FROM images WHERE id=$1", image.id) == first.id
        )


async def test_draft_finalization_rolls_back_after_receipt_insert(workflow_db, monkeypatch, tmp_path):
    from vision_bill.model.review import OriginalAvailability
    from vision_bill.model.workflow import DraftFinalize, DraftPatch

    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / "original.png")
    document = complete().envelope.receipt.model_dump(mode="json")
    await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes=document), scope=scope)
    originals = [
        OriginalAvailability(part_id=p.part_id, available=True, sha256_matches=True)
        for p in await w.get_source_parts(image.id, scope=scope)
    ]
    insert = w._receipts.insert_receipt

    async def interrupted(*args, **kwargs):
        await insert(*args, **kwargs)
        raise asyncio.CancelledError()

    monkeypatch.setattr(w._receipts, "insert_receipt", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await w.finalize_draft(
            image.id,
            DraftFinalize(expected_revision=2, request_key="final"),
            scope=scope,
            originals=originals,
        )
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 0
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 0
        assert (
            await conn.fetchval(
                "SELECT finalized_receipt_id FROM receipt_drafts WHERE image_id=$1", image.id
            )
            is None
        )
        assert (
            await conn.fetchval("SELECT status FROM images WHERE id=$1", image.id) == "needs_review"
        )


@pytest.mark.parametrize("due_kind", ["retry", "lease"])
async def test_next_wake_includes_overdue_retry_and_defers_unavailable_candidate(
    workflow_db, due_kind
):
    w, scope = workflow_db
    first, second = await enqueue(w, scope), await enqueue(w, scope)
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET retry_at=clock_timestamp()-interval '30 seconds' WHERE id=$1",
            first.id,
        )
        await conn.execute(
            "UPDATE images SET retry_at=clock_timestamp()+interval '60 seconds' WHERE id=$1",
            second.id,
        )
    if due_kind == "lease":
        claim = await w.claim_next(image_id=first.id)
        assert claim is not None
        async with w.pool.acquire() as conn:
            await conn.execute(
                "UPDATE images SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
                first.id,
            )
    assert await w.next_wake_delay(300) == 0
    delay = await w.next_wake_delay(300, deferred_image_ids=[first.id])
    assert 55 < delay <= 60
    assert await w.next_wake_delay(300, deferred_image_ids=[first.id, second.id]) == 300


async def test_long_cycle_rescans_overdue_retry_using_real_wake_query(
    workflow_db, settings, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import vision_bill.service.analysis_scheduler as module

    w, scope = workflow_db
    first, second = await enqueue(w, scope), await enqueue(w, scope)
    clock = [0.0]
    runs, waits = [], []
    provider_ready = [True]

    async def overdue():
        async with w.pool.acquire() as conn:
            await conn.execute(
                "UPDATE images SET retry_at=clock_timestamp()-interval '1 second' WHERE id=$1",
                first.id,
            )

    async def process(claim):
        runs.append((claim.image.id, clock[0]))
        if claim.image.id == second.id:
            clock[0] += 60
            await overdue()
            return await w.finish_attempt(
                claim, AttemptOutcome(kind="terminal", result_code="unreadable")
            )
        result = await w.finish_attempt(
            claim, AttemptOutcome(kind="retry", result_code="timed_out")
        )
        if claim.attempt_number == 2:
            provider_ready[0] = False
            await overdue()
        return result

    async def wait_for(awaitable, *, timeout):
        awaitable.close()
        waits.append(timeout)
        if len(waits) == 4:
            raise asyncio.CancelledError
        if len(waits) > 1:
            clock[0] += timeout
        raise TimeoutError

    service = SimpleNamespace(
        ready_for=AsyncMock(side_effect=lambda image: provider_ready[0]),
        process_claim=process,
    )
    scheduler = module.AnalysisScheduler(
        settings, None, SimpleNamespace(workflow_db=w), None, submission_service=service
    )
    monkeypatch.setattr(
        module, "asyncio", SimpleNamespace(wait_for=wait_for, timeout=asyncio.timeout)
    )
    with pytest.raises(asyncio.CancelledError):
        await scheduler._run_forever()
    assert runs == [(first.id, 0), (second.id, 0), (first.id, 60)]
    assert waits == [300, 0, 0, 300]
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT attempt_count FROM images WHERE id=$1", first.id) == 2
        assert await conn.fetchval("SELECT status FROM images WHERE id=$1", first.id) == "pending"

async def test_draft_policy_and_sources_are_versioned_partial_changes(workflow_db, tmp_path):
    from vision_bill.model.workflow import DraftPatch
    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / 'original.png')
    policy = {'tax_basis': 'gross', 'discount_treatment': 'included_in_lines',
              'tip_treatment': 'none', 'subtotal_origin': 'printed', 'tax_information': 'explicit_zero'}
    saved = await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes={
        'merchant_name': None, 'amount_policy': policy, 'field_sources': {}, 'overlap_decisions': []
    }), scope=scope)
    assert saved.revision == 2 and saved.document.merchant_name is None and saved.document.total is None
    assert saved.amount_policy.model_dump() == policy
    assert saved.field_sources == {} and saved.overlap_decisions == []
    assert any(f.code == 'missing_mandatory_field' for f in saved.findings)
    assert not any(f.code == 'unknown_tax_basis' for f in saved.findings)
    with pytest.raises(ValueError, match='revision'):
        await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes={'amount_policy': {'tax_basis':'net'}}), scope=scope)
    current = await w.get_draft(image.id, scope=scope)
    assert current.revision == 2 and current.amount_policy.tax_basis == 'gross'
    preserved = await w.patch_draft(image.id, DraftPatch(expected_revision=2, changes={'total':'0.000001'}), scope=scope)
    assert preserved.amount_policy == saved.amount_policy and preserved.document.merchant_name is None


@pytest.mark.parametrize('field', ['field_sources', 'overlap_decisions'])
async def test_draft_foreign_source_replacement_rolls_back(workflow_db, tmp_path, field):
    from vision_bill.model.workflow import DraftPatch
    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / 'original.png')
    foreign = {'part_id':str(uuid4()),'photo_index':0,'source_sha256':'a'*64,
               'view_id':str(uuid4()),'coordinate_space':'working','bbox':[[1,1],[2,1],[2,2]]}
    metadata = {'total':[foreign]} if field == 'field_sources' else [{'source_refs':[foreign]}]
    with pytest.raises(ValueError, match='source'):
        await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes={'total':'99',field:metadata}), scope=scope)
    current = await w.get_draft(image.id, scope=scope)
    assert current.revision == 1 and current.document.total is None
    async with w.pool.acquire() as conn:
        assert not await conn.fetchval('SELECT human_edited FROM receipt_drafts WHERE image_id=$1', image.id)

async def test_draft_review_metadata_http_contract(workflow_db, tmp_path):
    from types import SimpleNamespace

    import httpx
    from fastapi import FastAPI

    from vision_bill.api.drafts import router
    from vision_bill.security.dependencies import get_current_user
    from vision_bill.security.models import User
    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / 'original.png')
    app = FastAPI()
    app.include_router(router, prefix='/api/v1/images')
    app.dependency_overrides[get_current_user] = lambda: User(id=scope.user_id, username='owner', is_admin=False)
    app.state.submission_service = SimpleNamespace(db=w, patch_draft=w.patch_draft, get_draft=w.get_draft)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        path = f'/api/v1/images/{image.id}/draft'
        saved = await client.patch(path, json={'expected_revision':1,'changes':{
            'amount_policy':{'tax_basis':'net'},'merchant_name':None,'field_sources':{},'overlap_decisions':[]}})
        assert saved.status_code == 200
        assert saved.json()['amount_policy']['tax_basis'] == 'net'
        assert saved.json()['document']['merchant_name'] is None
        assert saved.json()['revision'] == 2
        assert (await client.patch(path, json={'expected_revision':1,'changes':{'total':'4'}})).status_code == 409
        foreign = {'part_id':str(uuid4()),'photo_index':0,'source_sha256':'a'*64,'coordinate_space':'unknown'}
        assert (await client.patch(path, json={'expected_revision':2,'changes':{'total':'99','field_sources':{'total':[foreign]}}})).status_code == 422
        current = (await client.get(path)).json()
        assert current['revision'] == 2 and current['document']['total'] is None


async def test_draft_overlap_migration_backfill_and_guard(db):
    import json
    migrate(db, '0014')
    from vision_bill.provider.db.workflow_db import WorkflowDB
    w = WorkflowDB(db)
    await w.init_db()
    try:
        async with w.pool.acquire() as conn:
            owner = await conn.fetchval("INSERT INTO users(username,hashed_password) VALUES('draft-test','hash') RETURNING id")
            image = await conn.fetchval("INSERT INTO images(image_path,user_id) VALUES('/private-original',$1) RETURNING id", owner)
            decisions = [{'decision':'retain', 'positions':[0,1], 'reason':'printed twice'}]
            extraction = await conn.fetchval("""INSERT INTO receipt_extractions(image_id,generation,attempt_number,outcome_kind,envelope,source_views)
                VALUES($1,1,1,'partial',$2::jsonb,'[]') RETURNING id""", image, json.dumps({'overlap_decisions':decisions}))
            await conn.execute("""INSERT INTO receipt_drafts(image_id,extraction_id,generation,document,amount_policy,findings,field_sources)
                VALUES($1,$2,1,'{}','{}','[]','{}')""", image, extraction)
        migrate(db)
        async with w.pool.acquire() as conn:
            assert json.loads(await conn.fetchval('SELECT overlap_decisions FROM receipt_drafts WHERE image_id=$1',image)) == decisions
            assert json.loads(await conn.fetchval('SELECT envelope FROM receipt_extractions WHERE id=$1',extraction)) == {'overlap_decisions':decisions}
        assert migrate(db,'0014','downgrade',check=False).returncode != 0
    finally:
        await w.destroy_db()

async def test_draft_validates_nested_metadata_before_typed_projection(workflow_db, tmp_path):
    from vision_bill.model.workflow import DraftPatch
    w, scope = workflow_db
    image = await partial_draft(w, scope, tmp_path / 'original.png')
    part = (await w.get_source_parts(image.id, scope=scope))[0]
    ref = {'part_id':str(part.part_id),'photo_index':0,'source_sha256':part.source_sha256,
           'coordinate_space':'unknown','extra':{'part_id':str(uuid4()),'photo_index':0,
           'source_sha256':'a'*64,'coordinate_space':'unknown'}}
    with pytest.raises(ValueError, match='source'):
        await w.patch_draft(image.id, DraftPatch(expected_revision=1, changes={'field_sources':{'total':[ref]},'total':'9'}), scope=scope)
    current = await w.get_draft(image.id, scope=scope)
    assert current.revision == 1 and current.document.total is None
