"""Run blocking calls off the event loop on pools split by class of work.

The default asyncio executor is one pool of ``min(32, cpu + 4)`` threads. A
runtime package install holds a thread for up to ten minutes, and every request
waiting on ``runtime_packages._INSTALL_LOCK`` holds another one while it blocks.
On a two-core host that leaves four threads for everything else, so a knowledge
index rebuild queues behind a pip download and the symptom is "the server got
slow but the CPU is idle".

Splitting by class of work makes that starvation impossible by construction: a
ten-minute install can only ever occupy :attr:`Pool.SUBPROCESS`.

Do not offload everything. A SQLite row read is tens of microseconds; a thread
hop plus future scheduling costs more than the query. ``run_blocking`` also adds
nothing to SQLite throughput because :class:`octop.infra.db.pool.SqlitePool`
serializes every statement behind one lock — the only gain is that the waiting
happens off the loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from functools import partial
from time import perf_counter
from typing import Any, cast

from octop.infra.metrics import METRICS

logger = logging.getLogger(__name__)


class Pool(StrEnum):
    """Class of blocking work. Each value owns a separate thread pool.

    The split is by *how long the call can run*, not by what it uses:

    ``DB``
        Repository calls. Split from IO only so a slow query is visible on its
        own counter; SQLite serializes these behind one lock anyway.
    ``IO``
        Blocking calls that finish in milliseconds to seconds — file IO, network
        calls that carry their own timeout, and child processes that pass an
        explicit ``timeout`` (``adb``, ``screencap``). A hang here is a bug or a
        bounded timeout, never a ten-minute wait.
    ``SUBPROCESS``
        Child work whose runtime nobody bounded: package installers and model
        downloads. Two threads, on purpose — see the module docstring.

    Putting a bounded ``adb`` call on ``SUBPROCESS`` looks tidier but shrinks
    console and screen-mirror throughput from eight threads to two, so it is
    the wrong trade even though both spawn a child process.
    """

    DB = "db"
    IO = "io"
    SUBPROCESS = "subprocess"


# SQLite serializes writes anyway, so a wide DB pool only deepens the queue;
# PostgreSQL can use a few more. Long operations get the narrowest pool.
_DEFAULT_WORKERS = {Pool.DB: 4, Pool.IO: 8, Pool.SUBPROCESS: 2}
_WORKER_ENV = {
    Pool.DB: "OCTOP_OFFLOAD_DB_WORKERS",
    Pool.IO: "OCTOP_OFFLOAD_IO_WORKERS",
    Pool.SUBPROCESS: "OCTOP_OFFLOAD_SUBPROCESS_WORKERS",
}
_DEFAULT_SLOW_MS = 250.0
_SLOW_ENV = "OCTOP_OFFLOAD_SLOW_MS"

_pools: dict[Pool, ThreadPoolExecutor] = {}
_pools_lock = threading.Lock()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return default


def slow_threshold_ms() -> float:
    """Calls slower than this are counted and logged."""
    raw = os.environ.get(_SLOW_ENV, "").strip()
    try:
        return float(raw) if raw else _DEFAULT_SLOW_MS
    except ValueError:
        return _DEFAULT_SLOW_MS


def executor_for(pool: Pool) -> ThreadPoolExecutor:
    """Return the executor for *pool*, creating it on first use."""
    with _pools_lock:
        existing = _pools.get(pool)
        if existing is not None:
            return existing
        created = ThreadPoolExecutor(
            max_workers=_env_int(_WORKER_ENV[pool], _DEFAULT_WORKERS[pool]),
            thread_name_prefix=f"octop-{pool.value}",
        )
        _pools[pool] = created
        return created


async def run_blocking[T](
    fn: Callable[..., T],
    *args: Any,
    pool: Pool = Pool.IO,
    wait_timeout: float | None = None,
    label: str | None = None,
    **kwargs: Any,
) -> T:
    """Call ``fn(*args, **kwargs)`` on *pool* and return its result.

    *wait_timeout* stops waiting; it cannot stop the thread. Pass it only where
    abandoning the result is safe — for subprocess work put the timeout on the
    child process instead, or the call keeps running invisibly.

    It is deliberately not called ``timeout``: many blocking functions take a
    ``timeout`` of their own, and a same-named parameter here would silently
    swallow it instead of forwarding it.
    """
    loop = asyncio.get_running_loop()
    name = label or getattr(fn, "__qualname__", None) or repr(fn)
    started = perf_counter()
    future = loop.run_in_executor(executor_for(pool), partial(fn, *args, **kwargs))
    try:
        if wait_timeout is None:
            return await future
        return cast("T", await asyncio.wait_for(future, wait_timeout))
    finally:
        _record(name, pool, (perf_counter() - started) * 1000.0)


async def run_db[T](fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous repository call on the dedicated DB pool.

    Repository signatures stay synchronous: making them async would touch forty
    repositories and every caller for no SQLite throughput gain.
    """
    return await run_blocking(fn, *args, pool=Pool.DB, **kwargs)


def _record(name: str, pool: Pool, elapsed_ms: float) -> None:
    METRICS.inc("blocking_offload_total")
    if elapsed_ms < slow_threshold_ms():
        return
    METRICS.inc("blocking_offload_slow_total")
    logger.warning("blocking call %s on the %s pool took %.0fms", name, pool.value, elapsed_ms)


def shutdown_pools(*, wait: bool = False) -> None:
    """Release the pools so a restarted loop (tests, reload) builds fresh ones.

    ``wait=False`` does not kill work already running — an install in flight is
    bounded by its own subprocess timeout, exactly as before this module existed.
    """
    with _pools_lock:
        pools = list(_pools.items())
        _pools.clear()
    for _pool, executor in pools:
        executor.shutdown(wait=wait)
