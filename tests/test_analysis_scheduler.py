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


async def test_retry_due_during_long_second_job_rescans_without_provider_down_hot_loop(
    settings, monkeypatch
):
    from datetime import UTC, datetime, timedelta

    import vision_bill.service.analysis_scheduler as module

    scheduler, db, service, first = context(settings)
    second = ImageRow(id=uuid4())
    clock = [0.0]
    due = {first.id: 0.0, second.id: 0.0}
    runs = []
    waits = []
    done = asyncio.Event()
    ready = {first.id: True, second.id: True}

    async def candidates():
        return [image for image in [first, second] if due.get(image.id, 999) <= clock[0]]

    async def claim_next(*, image_id, lease_seconds):
        return SimpleNamespace(image=first if image_id == first.id else second)

    async def process(claim):
        runs.append((claim.image.id, clock[0]))
        if claim.image.id == first.id and len(runs) == 1:
            due[first.id] = clock[0] + 30
            return CommitResult(
                stale=False,
                submission=SubmissionStatus(
                    id=first.id,
                    status="pending",
                    retry_at=datetime.now(UTC) + timedelta(seconds=30),
                ),
            )
        if claim.image.id == second.id:
            clock[0] += 60
            del due[second.id]
        else:
            ready[first.id] = False
            due[first.id] = clock[0]  # another overdue retry, but provider then goes down
        return CommitResult(
            stale=False, submission=SubmissionStatus(id=claim.image.id, status="pending")
        )

    async def wake_delay(maximum, *, deferred_image_ids=()):
        eligible = [
            when - clock[0] for image_id, when in due.items() if image_id not in deferred_image_ids
        ]
        return min(maximum, max(0, min(eligible))) if eligible else maximum

    async def wait_for(awaitable, *, timeout):
        waits.append(timeout)
        awaitable.close()
        if len(waits) >= 4:
            done.set()
            raise asyncio.CancelledError()
        if timeout:
            clock[0] += timeout if len(waits) > 1 else 0
        raise TimeoutError()

    db.due_candidates.side_effect = candidates
    db.claim_next.side_effect = claim_next
    db.next_wake_delay.side_effect = wake_delay
    service.ready_for.side_effect = lambda image: ready[image.id]
    service.process_claim.side_effect = process
    monkeypatch.setattr(module.asyncio, "wait_for", wait_for)
    task = asyncio.create_task(scheduler._run_forever())
    await done.wait()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert runs == [(first.id, 0), (second.id, 0), (first.id, 60)]
    assert db.claim_next.await_count == 3
    assert waits == [300, 0, 0, 300]
