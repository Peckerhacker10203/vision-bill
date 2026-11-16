"""Owned PostgreSQL barriers prove atomic upload recovery and artifact retention."""

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from tests.e2e import test_submission_service_e2e as submissions
from tests.e2e.test_benchmark_claims_e2e import wait_blocked
from vision_bill.model.workflow import AttemptOutcome

postgres, db, workflow_db = submissions.postgres, submissions.db, submissions.workflow_db
pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("phase,attempt", [("finish", 1), ("finish", 3), ("checkpoint", 1),
                                          ("finish_atomic", 1), ("checkpoint_atomic", 1), ("finish_release", 1),
                                          ("references", 1), ("reference_release", 1), ("tags", 1)])
async def test_upload_deadline_atomic_recovery_and_artifacts(workflow_db, settings, tmp_path, phase, attempt):
    w, scope = workflow_db
    image, paths, llm, native, service = await submissions.setup(w, scope, settings, tmp_path)
    originals = [p.read_bytes() for p in paths]
    source_hashes = [hashlib.sha256(data).hexdigest() for data in originals]
    if attempt == 3:
        async with w.pool.acquire() as conn:
            await conn.execute("UPDATE images SET attempt_count=2 WHERE id=$1", image.id)
    claim = await w.claim_next()
    assert claim.attempt_number == attempt
    blocker = await w.pool.acquire()
    tx = blocker.transaction()
    await tx.start()
    reset_active = 0
    rollback_active = 0
    release_entered = asyncio.Event()
    slow = True

    class ControlledConnection(asyncpg.Connection):
        reference_read = False
        tag_read = False
        sealed = False

        async def fetch(self, query, *args, **kwargs):
            self.tag_read = "FROM tags" in query
            return await super().fetch(query, *args, **kwargs)

        async def fetchrow(self, query, *args, **kwargs):
            self.reference_read = query == "SELECT id FROM images WHERE id=$1 FOR UPDATE"
            self.sealed = query.startswith("UPDATE images SET status=$2")
            return await super().fetchrow(query, *args, **kwargs)

        async def execute(self, query, *args, **kwargs):
            nonlocal rollback_active
            if query == "ROLLBACK;" and slow:
                rollback_active += 1
                try:
                    await asyncio.sleep(.4)
                finally:
                    rollback_active -= 1
            return await super().execute(query, *args, **kwargs)

    async def reset(conn):
        nonlocal reset_active
        if slow and ((phase == "reference_release" and conn.reference_read) or (phase == "tags" and conn.tag_read) or (phase == "finish_release" and conn.sealed)):
            reset_active += 1
            release_entered.set()
            try:
                await asyncio.sleep(.6)
            finally:
                reset_active -= 1

    pool = await asyncpg.create_pool(w._settings.pg_dsn, min_size=0, max_size=2,
                                    connection_class=ControlledConnection, reset=reset)
    peer = await pool.acquire()
    peer_pid = await peer.fetchval("SELECT pg_backend_pid()")
    original_pool = w._pool
    w._pool = pool
    # Exercise the real ReceiptService -> ReceiptDB hidden read, not setup's mock.
    del service.receipts.list_tags
    service.receipts._db._pool = pool
    service.worker = settings.worker.model_copy(update={"attempt_timeout_seconds": .4})
    original_attempt = service._attempt
    original_page = native.extract_page.side_effect
    original_save = w.save_ocr_checkpoint

    async def lock_image():
        await blocker.fetchval("SELECT id FROM images WHERE id=$1 FOR UPDATE", image.id)

    async def finish_barrier(*args, **kwargs):
        outcome = await original_attempt(*args, **kwargs)
        await lock_image()
        return outcome

    async def checkpoint_barrier(*args, **kwargs):
        page = await original_page(*args, **kwargs)
        if kwargs["photo_index"] == 1:
            await lock_image()
        return page

    async def stale_checkpoint(*args, **kwargs):
        async with original_pool.acquire() as conn:
            await conn.execute("UPDATE images SET claim_token=$2 WHERE id=$1", image.id, uuid4())
        assert not await original_save(*args, **kwargs)
        if phase == "references":
            await lock_image()
        return False

    if phase == "finish":
        service._attempt = finish_barrier
    elif phase == "checkpoint":
        native.extract_page.side_effect = checkpoint_barrier
    elif phase in {"references", "reference_release"}:
        w.save_ocr_checkpoint = stale_checkpoint
    elif phase == "tags":
        await blocker.execute("LOCK TABLE tags IN ACCESS EXCLUSIVE MODE")
    elif phase in {"finish_atomic", "checkpoint_atomic"}:
        async with original_pool.acquire() as conn:
            await conn.execute("""CREATE FUNCTION deadline_barrier() RETURNS trigger AS $$
                BEGIN PERFORM pg_advisory_xact_lock(1400); RETURN NEW; END;
                $$ LANGUAGE plpgsql""")
            if phase == "finish_atomic":
                await conn.execute("""CREATE TRIGGER deadline_barrier BEFORE INSERT
                    ON receipt_review_revisions FOR EACH ROW EXECUTE FUNCTION deadline_barrier()""")
            else:
                await conn.execute("""CREATE TRIGGER deadline_barrier BEFORE UPDATE OF ocr_checkpoint
                    ON images FOR EACH ROW WHEN (OLD.ocr_checkpoint IS NOT NULL)
                    EXECUTE FUNCTION deadline_barrier()""")
        await blocker.execute("SELECT pg_advisory_xact_lock(1400)")

    started = asyncio.get_running_loop().time()
    work = asyncio.create_task(service.process_claim(claim))
    try:
        if phase in {"reference_release", "finish_release"}:
            await asyncio.wait_for(release_entered.wait(), 2)
        else:
            await wait_blocked(original_pool, 1)
        result = await asyncio.wait_for(work, .7)
        assert result.stale
        assert asyncio.get_running_loop().time() - started < .55
        assert reset_active == rollback_active == 0
        assert await peer.fetchval("SELECT pg_backend_pid()") == peer_pid
        assert pool._queue.qsize() == 1
        if phase == "finish_release":
            # COMMIT succeeded but its release/acknowledgement did not complete
            # inside the budget. Retain every artifact and accept no duplicate.
            slow = False
            async with original_pool.acquire() as conn:
                for table in ("receipts", "receipt_extractions", "receipt_review_revisions"):
                    assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 1
                assert await conn.fetchval("SELECT count(*) FROM extraction_working_views") == 2
            status = await w.get_submission(image.id, scope=scope)
            assert status.status == "analyzed" and status.receipt_id
            assert await w.claim_next() is None
            assert (await w.finish_attempt(claim, AttemptOutcome(kind="retry", result_code="timed_out"))).stale
            assert len(list((tmp_path / "uploads" / "working").glob("*/*.png"))) == 2
            for view in status.source_views:
                saved = await w.get_working_view(image.id, view.view_id, scope=scope)
                assert service.images.read_working_image(image.id, saved.artifact_path, saved.metadata)
            assert [p.read_bytes() for p in paths] == originals
            return
        async with original_pool.acquire() as conn:
            for table in ("receipts", "receipt_extractions", "receipt_review_revisions", "line_items", "taxes", "receipt_drafts"):
                assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0
            checkpoint = await conn.fetchval("SELECT ocr_checkpoint FROM images WHERE id=$1", image.id)
            saved = await conn.fetch("SELECT artifact_path FROM extraction_working_views WHERE image_id=$1", image.id)
            row = await conn.fetchrow("SELECT status,attempt_count FROM images WHERE id=$1", image.id)
        assert row["status"] == "processing" and row["attempt_count"] == attempt
        prefix = 2 if phase.startswith("finish") else 1 if phase.startswith("checkpoint") else 0
        assert len(saved) == prefix
        assert (len(json.loads(checkpoint)["pages"]) if checkpoint else 0) == prefix
        files = list((tmp_path / "uploads" / "working").glob("*/*.png"))
        assert len(files) == (2 if prefix else 0 if phase == "tags" else 1)
        before = {p: p.read_bytes() for p in files}
        assert all(Path(row["artifact_path"]).exists() for row in saved)
        assert [p.read_bytes() for p in paths] == originals
        await tx.rollback()
        slow = False
        service._attempt = original_attempt
        native.extract_page.side_effect = original_page
        w.save_ocr_checkpoint = original_save
        service.worker = settings.worker
        native.extract_page.reset_mock()
        async with original_pool.acquire() as conn:
            await conn.execute("UPDATE images SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1", image.id)
        recovered = await w.claim_next()
        if attempt == 3:
            assert recovered is None
            status = await w.get_submission(image.id, scope=scope)
            assert status.terminal and status.status == "failed" and status.attempt_count == 3
            native.extract_page.assert_not_awaited()
            assert await w.claim_next() is None
            async with original_pool.acquire() as conn:
                history = await conn.fetch("SELECT attempt_number,result_code,outcome_kind FROM receipt_extractions")
                assert [dict(row) for row in history] == [{"attempt_number": 3, "result_code": "lease_expired", "outcome_kind": "terminal"}]
        else:
            assert recovered.attempt_number == 2
            final = await service.process_claim(recovered)
            assert final.receipt is not None and final.submission.status == "analyzed"
            assert native.extract_page.await_count == 2 - prefix
            assert (await w.finish_attempt(claim, AttemptOutcome(kind="retry", result_code="timed_out"))).stale
            assert (await w.finish_attempt(recovered, AttemptOutcome(kind="retry", result_code="timed_out"))).stale
            async with original_pool.acquire() as conn:
                assert await conn.fetchval("SELECT count(*) FROM receipts") == 1
                history = await conn.fetch("SELECT attempt_number,result_code,outcome_kind FROM receipt_extractions ORDER BY attempt_number")
                assert len(history) == 2
                assert dict(history[0]) == {"attempt_number": 1, "result_code": "lease_expired", "outcome_kind": "retry"}
                assert history[1]["attempt_number"] == 2 and history[1]["outcome_kind"] == "complete"
                assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 1
        assert all(path.read_bytes() == data for path, data in before.items())
        assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == source_hashes
    finally:
        slow = False
        if not work.done():
            work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        if blocker.is_in_transaction():
            await tx.rollback()
        await original_pool.release(blocker)
        await pool.release(peer)
        await pool.close()
        w._pool = original_pool
