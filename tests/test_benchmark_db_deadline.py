"""Installed asyncpg transaction/pool lifecycle with only wire I/O simulated."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import pytest

from vision_bill.config import WorkerSettings
from vision_bill.provider.db.benchmark_db import BenchmarkDB
from vision_bill.service.benchmark_service import BenchmarkService


class WireConnection(asyncpg.Connection):
    """Real Transaction and PoolConnectionHolder; controllable network waits."""

    def __init__(self, phase):
        self.phase = phase
        self._loop = asyncio.get_running_loop()
        self._pool_release_ctr = 0
        self._top_xact = None
        self._proxy = None
        self._listeners = {}
        self._log_listeners = set()
        self._protocol = SimpleNamespace(is_in_transaction=lambda: False, queries_count=0,
                                         _is_cancelling=lambda: False)
        self.closed = False
        self.commands = []
        self.active_io = 0
        self.terminated = False
        self.fetch_count = 0
        self.commit_at = None

    async def pause(self, seconds):
        self.active_io += 1
        try:
            await asyncio.sleep(seconds)
        finally:
            self.active_io -= 1

    def is_closed(self):
        return self.closed

    def terminate(self):
        self.closed = self.terminated = True
        if self._proxy is not None:
            self._proxy._holder._release_on_close()

    async def close(self, *, timeout=None):
        # Connection close after a failed acquire/setup is network I/O too.
        if self.phase in {"acquire_setup", "acquire_init"}:
            await self.pause(.3)
        self.terminate()

    async def _reset(self):
        pass

    async def reset(self, *, timeout=None):
        if self.phase in {"release", "read_release"}:
            await self.pause(.3)

    async def execute(self, sql, *args, **kwargs):
        self.commands.append(sql)
        if sql == "COMMIT;":
            self.commit_at = self._loop.time()
        if sql == "ROLLBACK;":
            await self.pause(.3)
        return "DELETE 1" if sql.startswith("DELETE") else "UPDATE 1"

    async def fetchrow(self, sql, *args, **kwargs):
        self.fetch_count += 1
        if self.phase == "read_release":
            await self.pause(.02)
            return None
        if self.phase == "rollback":
            await self.pause(10)  # outer deadline cancels the exhausted-run lock
        elif self.phase == "release" and self.fetch_count == 1:
            await self.pause(.9)  # release starts just before the deadline
        return {"inference_attempt_count": 1}

    async def fetchval(self, sql, *args, **kwargs):
        await self.pause(.02)
        return 600 if "request_timeout_seconds" in sql else None


@pytest.mark.parametrize("phase", ["rollback", "release", "acquire_setup", "acquire_init"])
async def test_exhausted_worker_bounds_actual_asyncpg_cleanup_and_preserves_pool_capacity(phase):
    made = []
    async def connect(*args, **kwargs):
        conn = WireConnection(phase)
        made.append(conn)
        return conn
    async def setup(conn):
        if phase == "acquire_setup":
            await conn.pause(10)
    async def init(conn):
        if phase == "acquire_init":
            await conn.pause(10)
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect,
                                    connection_class=WireConnection, setup=setup, init=init)
    service = object.__new__(BenchmarkService)
    service._settings = WorkerSettings(attempt_timeout_seconds=1)
    service._receipts = SimpleNamespace(pool=pool)
    service._db = BenchmarkDB(pool)
    service._db.run = AsyncMock(return_value=SimpleNamespace(id=uuid4()))
    service._db.manifest = AsyncMock(return_value=None)
    service._timeout = AsyncMock(return_value=600)
    task = {"run_id": uuid4(), "model_id": "m", "receipt_id": uuid4(), "claim_token": uuid4(),
            "inference_attempt_count": 3, "inference_started": False, "attempts": 0}
    started = asyncio.get_running_loop().time()
    try:
        await service._execute(task)
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 1.15, f"budget=1s, elapsed={elapsed:.3f}s, phase={phase}"
        assert made[0].terminated
        assert made[0].active_io == 0, "no shielded background cleanup may escape"
        assert made[0].commit_at is None or made[0].commit_at < started + 1
        # A discarded borrowed connection returns its slot, not the whole pool.
        async with asyncio.timeout(.2):
            pool._setup = None
            pool._init = None
            pool._holders[0]._setup = None
            conn = await pool.acquire()
            assert conn._con is not made[0]
            conn.terminate()
    finally:
        pool.terminate()


@pytest.mark.parametrize("operation", ["run", "manifest", "timeout", "digests", "legacy_receipt"])
async def test_read_preflight_release_uses_its_explicit_deadline(operation, tmp_path):
    from vision_bill.config import ImageSettings, PGSettings
    from vision_bill.service.receipt_service import ReceiptService
    made = []
    async def connect(*args, **kwargs):
        made.append(WireConnection("read_release"))
        return made[-1]
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect, connection_class=WireConnection)
    receipts = ReceiptService(ImageSettings(save_dir=str(tmp_path), tmp_dir=str(tmp_path)), PGSettings(user="test", password="test"), AsyncMock())
    receipts._db._pool = pool
    service = BenchmarkService(AsyncMock(), receipts)
    run = SimpleNamespace(id=uuid4())
    started = asyncio.get_running_loop().time()
    deadline = started + .1
    calls = {
        "run": lambda: service._db.run(run.id, deadline=deadline),
        "manifest": lambda: service._db.manifest(run.id, deadline=deadline),
        "timeout": lambda: service._timeout(run, deadline=deadline),
        "digests": lambda: service._run_digests(run, deadline=deadline),
        "legacy_receipt": lambda: service._expected_receipt_and_path(run.id, deadline=deadline),
    }
    try:
        with pytest.raises(TimeoutError):
            await calls[operation]()
        assert asyncio.get_running_loop().time() - started < .2
        assert made[0].terminated and made[0].active_io == 0
        assert pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_bounded_normal_release_reuses_connection_and_detaches_old_proxy():
    from vision_bill.provider.db.deadline import connection
    made = []
    async def connect(*args, **kwargs):
        made.append(WireConnection("responsive"))
        return made[-1]
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect, connection_class=WireConnection)
    try:
        async with connection(pool, deadline=asyncio.get_running_loop().time()+1, transaction=True) as first:
            raw = first._con
            await first.execute("UPDATE harmless")
        assert first._con is None and not raw.terminated
        async with connection(pool, deadline=asyncio.get_running_loop().time()+1) as second:
            assert second._con is raw and second is not first
        assert second._con is None and len(made) == 1 and pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_slow_rollback_preserves_original_error_and_leaves_no_cleanup_task():
    from vision_bill.provider.db.deadline import connection
    made = []
    async def connect(*args, **kwargs):
        made.append(WireConnection("responsive"))
        return made[-1]
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect, connection_class=WireConnection)
    started = asyncio.get_running_loop().time()
    try:
        with pytest.raises(ValueError, match="original failure"):
            async with connection(pool, deadline=started+.1, transaction=True):
                raise ValueError("original failure")
        assert asyncio.get_running_loop().time() - started < .2
        assert made[0].terminated and made[0].active_io == 0
        assert "ROLLBACK;" in made[0].commands and "COMMIT;" not in made[0].commands
        assert pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_external_cancellation_is_preserved_during_bounded_rollback():
    from vision_bill.provider.db.deadline import connection
    made = []
    entered = asyncio.Event()
    async def connect(*args, **kwargs):
        made.append(WireConnection("responsive"))
        return made[-1]
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect, connection_class=WireConnection)
    started = asyncio.get_running_loop().time()
    async def work():
        async with connection(pool, deadline=started+.1, transaction=True):
            entered.set()
            await asyncio.Event().wait()
    task = asyncio.create_task(work())
    try:
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and asyncio.get_running_loop().time() - started < .2
        assert made[0].terminated and made[0].active_io == 0
        assert pool._queue.qsize() == 1
    finally:
        pool.terminate()


async def test_unsupported_pool_fails_closed_without_public_cleanup_fallback():
    from vision_bill.provider.db.deadline import connection
    class UnsupportedPool:
        def acquire(self):
            raise AssertionError("unbounded fallback must never run")
    with pytest.raises(RuntimeError, match="Unsupported asyncpg deadline adapter"):
        async with connection(UnsupportedPool(), deadline=asyncio.get_running_loop().time()+1):
            pytest.fail("unsupported pool cannot be borrowed")


async def test_cancelled_pool_queue_wait_does_not_disrupt_another_borrower():
    made = []
    async def connect(*args, **kwargs):
        made.append(WireConnection("responsive"))
        return made[-1]
    pool = await asyncpg.create_pool(min_size=0, max_size=1, connect=connect,
                                    connection_class=WireConnection)
    borrowed = await pool.acquire()
    service = object.__new__(BenchmarkService)
    service._settings = WorkerSettings(attempt_timeout_seconds=1)
    service._db = BenchmarkDB(pool)
    # Exercise the actual first run-read acquisition, before inference or reserve.
    started = asyncio.get_running_loop().time()
    try:
        await service._execute({"run_id": uuid4()})
        assert asyncio.get_running_loop().time() - started < 1.15
        assert not borrowed.is_closed() and not made[0].terminated
        await pool.release(borrowed)
        async with asyncio.timeout(.2):
            reused = await pool.acquire()
            assert reused._con is made[0]
            await pool.release(reused)
    finally:
        pool.terminate()
