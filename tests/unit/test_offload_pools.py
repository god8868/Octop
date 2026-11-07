"""Offload pools: isolation between classes of work, metrics and shutdown."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import pytest

from octop.infra.metrics import METRICS
from octop.infra.utils import offload
from octop.infra.utils.offload import Pool, run_blocking, run_db


@pytest.fixture(autouse=True)
def _fresh_pools() -> Iterator[None]:
    offload.shutdown_pools()
    before = METRICS.snapshot()
    yield
    offload.shutdown_pools()
    for key, value in before.items():
        METRICS.set(key, value)


def test_each_class_of_work_gets_its_own_executor() -> None:
    assert offload.executor_for(Pool.DB) is not offload.executor_for(Pool.IO)
    assert offload.executor_for(Pool.IO) is not offload.executor_for(Pool.SUBPROCESS)
    assert offload.executor_for(Pool.DB) is offload.executor_for(Pool.DB)


def _thread_name() -> str:
    return threading.current_thread().name


def test_db_work_runs_on_its_own_threads() -> None:
    assert asyncio.run(run_db(_thread_name)).startswith("octop-db")


def test_generic_blocking_work_runs_on_the_io_threads() -> None:
    assert asyncio.run(run_blocking(_thread_name)).startswith("octop-io")


def test_a_fully_saturated_io_pool_does_not_starve_the_db_pool() -> None:
    """The whole point of splitting: long work cannot block short work.

    Every ``Pool.IO`` thread is held here, so a DB call sharing that pool would
    hang until the timeout. It returns promptly because the pools differ.
    """
    release = threading.Event()
    started: list[int] = []
    started_lock = threading.Lock()

    def hold() -> None:
        with started_lock:
            started.append(1)
        release.wait(timeout=5)

    async def scenario() -> str:
        workers = offload.executor_for(Pool.IO)._max_workers
        tasks = [asyncio.create_task(run_blocking(hold)) for _ in range(workers)]
        for _ in range(200):
            with started_lock:
                if len(started) == workers:
                    break
            await asyncio.sleep(0.01)
        try:
            assert len(started) == workers, "the IO pool never filled up"
            return await asyncio.wait_for(run_db(lambda: "db still works"), timeout=2)
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    assert asyncio.run(scenario()) == "db still works"


def test_run_db_uses_the_db_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Pool] = []
    real = offload.executor_for

    def spy(pool: Pool) -> ThreadPoolExecutor:
        seen.append(pool)
        return real(pool)

    monkeypatch.setattr(offload, "executor_for", spy)

    assert asyncio.run(run_db(lambda: 7)) == 7
    assert seen == [Pool.DB]


def test_results_and_arguments_survive_the_thread_hop() -> None:
    async def scenario() -> int:
        return await run_blocking(lambda a, b, *, extra=0: a + b + extra, 2, 3, extra=4)

    assert asyncio.run(scenario()) == 9


def test_offloads_are_counted() -> None:
    before = METRICS.blocking_offload_total

    asyncio.run(run_blocking(lambda: None))

    assert METRICS.blocking_offload_total == before + 1


def test_a_slow_call_is_counted_and_named(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("OCTOP_OFFLOAD_SLOW_MS", "10")
    before = METRICS.blocking_offload_slow_total

    def slow_repository_read() -> str:
        sleep(0.05)
        return "done"

    with caplog.at_level(logging.WARNING, logger="octop.infra.utils.offload"):
        asyncio.run(run_blocking(slow_repository_read))

    assert METRICS.blocking_offload_slow_total == before + 1
    assert any("slow_repository_read" in record.message for record in caplog.records)


def test_a_fast_call_is_not_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("OCTOP_OFFLOAD_SLOW_MS", "10000")

    with caplog.at_level(logging.WARNING, logger="octop.infra.utils.offload"):
        asyncio.run(run_blocking(lambda: None))

    assert caplog.records == []


def test_a_timeout_stops_the_wait() -> None:
    """Documented behaviour: the caller gives up, the thread keeps going."""

    async def scenario() -> None:
        with pytest.raises(TimeoutError):
            await run_blocking(sleep, 0.3, wait_timeout=0.05)

    asyncio.run(scenario())


def test_pool_sizes_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTOP_OFFLOAD_SUBPROCESS_WORKERS", "5")

    assert offload.executor_for(Pool.SUBPROCESS)._max_workers == 5


def test_an_exception_crosses_the_thread_boundary() -> None:
    def boom() -> None:
        raise ValueError("repository failed")

    async def scenario() -> None:
        with pytest.raises(ValueError, match="repository failed"):
            await run_db(boom)

    asyncio.run(scenario())
