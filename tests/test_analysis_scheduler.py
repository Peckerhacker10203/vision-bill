"""Durable readiness gates precede claims; retries wake at their due time."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from vision_bill.model.db.image import ImageRow
from vision_bill.model.workflow import CommitResult, SubmissionStatus
from vision_bill.service.analysis_scheduler import AnalysisScheduler


def context(settings):
    image = ImageRow(id=uuid4())
    claim = SimpleNamespace(image=image)
    db = SimpleNamespace(
        is_ready=True,
        due_candidates=AsyncMock(return_value=[image]),
        claim_next=AsyncMock(return_value=claim),
        next_wake_delay=AsyncMock(return_value=0.01),
    )
    receipts = SimpleNamespace(workflow_db=db)
    service = SimpleNamespace(
        ready_for=AsyncMock(return_value=True),
        process_claim=AsyncMock(
            return_value=CommitResult(
                stale=False,
                submission=SubmissionStatus(id=image.id, status="pending", result_code="timed_out"),
            )
        ),
    )
    scheduler = AnalysisScheduler(settings, None, receipts, None, submission_service=service)
    return scheduler, db, service, image


async def test_provider_down_waits_before_claim_without_attempt(settings):
    scheduler, db, service, _image = context(settings)
    service.ready_for.return_value = False
    assert await scheduler.process_pending() == []
    db.claim_next.assert_not_awaited()
    service.process_claim.assert_not_awaited()


async def test_pending_backoff_is_nonterminal_and_fenced_claim_used(settings):
    scheduler, db, service, image = context(settings)
    result = await scheduler.process_pending()
    assert result[0].status == "pending" and result[0].error == "timed_out"
    db.claim_next.assert_awaited_once_with(lease_seconds=900, image_id=image.id)
    service.process_claim.assert_awaited_once_with(db.claim_next.return_value)


async def test_earliest_retry_wakes_without_waiting_300_seconds(settings):
    scheduler, db, _service, _image = context(settings)
    done = asyncio.Event()
    count = 0

    async def scan():
        nonlocal count
        count += 1
        if count > 1:
            done.set()
        return []

    scheduler.process_pending = scan
    await scheduler.start()
    try:
        async with asyncio.timeout(0.3):
            await done.wait()
    finally:
        await scheduler.stop()
    assert count >= 2
    db.next_wake_delay.assert_awaited()


async def test_wake_trigger_and_stop_are_idempotent(settings):
    scheduler, db, _service, _image = context(settings)
    db.next_wake_delay.return_value = 300
    await scheduler.start()
    task = scheduler._task
    await scheduler.start()
    assert scheduler._task is task
    await asyncio.sleep(0)
    scheduler.trigger()
    await asyncio.sleep(0.01)
    await scheduler.stop()
    await scheduler.stop()
    assert scheduler._task is None


async def test_database_unavailable_never_claims(settings):
    scheduler, db, _service, _image = context(settings)
    db.is_ready = False
    assert await scheduler.process_pending() == []
    db.due_candidates.assert_not_awaited()
