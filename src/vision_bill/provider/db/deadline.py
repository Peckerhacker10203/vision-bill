"""Attempt-local asyncpg lifecycle without unbounded or shielded cleanup.

Only callers supplying an absolute loop-clock deadline use this adapter. The
ordinary API path keeps asyncpg's public contexts. Private pool coupling lives
here because public acquire can await close after setup failure, and public
release shields a new task. Neither can enforce cancellation during cleanup.
The adapter is checked against the locked asyncpg version and fails closed on
an upgrade until its lifecycle tests have been reviewed.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any, TypeVar

import asyncpg

T = TypeVar("T")


def _remaining(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError("database_attempt_deadline")
    return remaining


async def _within(deadline: float, operation: Callable[[], Awaitable[T]]) -> T:
    _remaining(deadline)  # Do not even start COMMIT/ROLLBACK after expiry.
    async with asyncio.timeout_at(deadline):
        return await operation()


def _discard(holder: Any) -> None:
    # Connection.terminate() aborts the transport synchronously. Its normal
    # callback detaches the proxy and returns the slot. Explicit release is
    # idempotent, also covering connect/init failures before a proxy exists.
    if holder._con is not None:
        with suppress(Exception):
            holder._con.terminate()
    holder._release_on_close()


async def _acquire(pool: asyncpg.Pool, deadline: float) -> Any:
    if asyncpg.__version__ != "0.31.0" or not isinstance(pool, asyncpg.Pool):
        raise RuntimeError("Unsupported asyncpg deadline adapter; review locked lifecycle")
    pool._check_init()
    if pool._closing:
        raise asyncpg.InterfaceError("pool is closing")
    holder = await _within(deadline, pool._queue.get)
    if not isinstance(holder, asyncpg.pool.PoolConnectionHolder):
        pool._queue.put_nowait(holder)
        raise TypeError("Unsupported asyncpg pool holder")
    # Own the slot during connect/init/setup as well as during queries. No
    # other borrower or whole-pool state is terminated on this task's timeout.
    holder._in_use = pool._loop.create_future()
    holder._maybe_cancel_inactive_callback()
    try:
        async with asyncio.timeout_at(deadline):
            _remaining(deadline)
            if holder._con is None or holder._con.is_closed() or holder._generation != pool._generation:
                if holder._con is not None:
                    holder._con.terminate()
                    holder._con = None
                kwargs = dict(pool._connect_kwargs)
                kwargs["timeout"] = min(kwargs.get("timeout") or 60, _remaining(deadline))
                holder._con = await pool._connect(
                    *pool._connect_args, loop=pool._loop,
                    connection_class=pool._connection_class, record_class=pool._record_class,
                    **kwargs,
                )
                if not isinstance(holder._con, pool._connection_class):
                    raise TypeError("Unexpected asyncpg connection class")
                holder._generation = pool._generation
                # Retain the connection BEFORE init so cancellation discards it
                # instead of entering asyncpg's unbounded graceful-close handler.
                if pool._init is not None:
                    await pool._init(holder._con)
            holder._proxy = asyncpg.pool.PoolConnectionProxy(holder, holder._con)
            if holder._setup is not None:
                await holder._setup(holder._proxy)
            _remaining(deadline)
            return holder
    except BaseException:
        _discard(holder)
        raise


@asynccontextmanager
async def connection(
    pool: asyncpg.Pool, *, deadline: float | None = None, transaction: bool = False,
) -> AsyncIterator[asyncpg.Connection]:
    if deadline is None:
        async with pool.acquire() as conn:
            if transaction:
                async with conn.transaction():
                    yield conn
            else:
                yield conn
        return

    holder = await _acquire(pool, deadline)
    conn = holder._proxy
    failed = False
    try:
        if transaction:
            tx = conn.transaction()
            try:
                await _within(deadline, tx.start)
            except BaseException:
                _discard(holder)
                raise
            try:
                async with asyncio.timeout_at(deadline):
                    yield conn
            except BaseException:
                try:
                    await _within(deadline, tx.rollback)
                except BaseException:  # noqa: BLE001 - discard, preserve original body error
                    _discard(holder)
                raise  # Keep the original cancellation/error, never a closed-tx error.
            else:
                try:
                    await _within(deadline, tx.commit)
                except BaseException:
                    _discard(holder)
                    raise
        else:
            async with asyncio.timeout_at(deadline):
                yield conn
    except BaseException:
        failed = True
        raise
    finally:
        if holder._in_use is not None:
            try:
                _remaining(deadline)
                conn._con._on_release()
                # Same reset/generation/query-limit bookkeeping as Pool.release,
                # awaited directly: cancellation leaves no shielded task behind.
                await _within(deadline, lambda: holder.release(_remaining(deadline)))
            except BaseException:
                _discard(holder)
                if not failed:
                    raise
