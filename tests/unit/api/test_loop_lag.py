"""Event-loop stall detection: sampling maths, attribution and rescheduling."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator

import pytest

from octop.api.middleware import loop_lag
from octop.infra.metrics import METRICS


@pytest.fixture(autouse=True)
def _reset_module_state() -> Iterator[None]:
    loop_lag.stop_watchdog()
    before = METRICS.snapshot()
    yield
    loop_lag.stop_watchdog()
    for key, value in before.items():
        METRICS.set(key, value)


def _stalls() -> int:
    return METRICS.loop_lag_over_100ms_total


def test_a_free_loop_reports_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="octop.api.middleware.loop_lag"):
        loop_lag._sample(10.005, 10.0, limit_ms=100.0)

    assert caplog.records == []


def test_a_stall_is_counted_and_named(caplog: pytest.LogCaptureFixture) -> None:
    loop_lag._enter("POST /api/knowledge/bases/1/documents")

    with caplog.at_level(logging.WARNING, logger="octop.api.middleware.loop_lag"):
        loop_lag._sample(10.900, 10.0)

    assert METRICS.loop_lag_over_100ms_total >= 1
    assert METRICS.loop_lag_max_ms >= 900
    assert any(
        "POST /api/knowledge/bases/1/documents" in record.message for record in caplog.records
    )


def test_a_stall_without_a_request_says_so(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="octop.api.middleware.loop_lag"):
        loop_lag._sample(10.900, 10.0)

    assert any("no request in flight" in record.message for record in caplog.records)


def test_the_oldest_request_wins_the_blame(caplog: pytest.LogCaptureFixture) -> None:
    loop_lag._enter("GET /api/agents")
    loop_lag._enter("GET /api/health")

    with caplog.at_level(logging.WARNING, logger="octop.api.middleware.loop_lag"):
        loop_lag._sample(10.900, 10.0)

    assert any("GET /api/agents" in record.message for record in caplog.records)


def test_the_deadline_is_rebased_after_a_stall() -> None:
    """A one-second stall must not report nine extra fake stalls."""
    deadline = loop_lag._sample(11.0, 10.0)

    assert deadline == pytest.approx(11.0 + loop_lag._INTERVAL_S)


def test_interleaved_requests_do_not_remove_each_other() -> None:
    """Each exit must clear its own entry, not whatever was registered last.

    Requests interleave at every ``await``: with a module-level token, the first
    request to finish removes the second request's entry and leaks its own for
    the process lifetime.
    """

    async def scenario() -> None:
        async def request(label: str, gate: asyncio.Event, done: asyncio.Event) -> None:
            loop_lag._enter(label)
            await gate.wait()
            loop_lag._leave()
            done.set()

        first_gate, second_gate = asyncio.Event(), asyncio.Event()
        first_done, second_done = asyncio.Event(), asyncio.Event()
        asyncio.create_task(request("first", first_gate, first_done))
        asyncio.create_task(request("second", second_gate, second_done))
        await asyncio.sleep(0)

        assert len(loop_lag._in_flight) == 2
        first_gate.set()
        await first_done.wait()
        assert [entry.label for entry in loop_lag._in_flight.values()] == ["second"]

        second_gate.set()
        await second_done.wait()
        assert loop_lag._in_flight == {}

    asyncio.run(scenario())


def test_the_watchdog_reschedules_itself() -> None:
    async def scenario() -> None:
        loop_lag._ensure_watchdog()
        assert loop_lag._expected is not None
        # A second call must not start a duplicate sampler.
        first = loop_lag._expected
        loop_lag._ensure_watchdog()
        assert loop_lag._expected == first
        loop_lag.stop_watchdog()
        assert loop_lag._expected is None

    asyncio.run(scenario())


def test_threshold_can_be_tuned_at_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTOP_LOOP_LAG_MS", "50")

    assert loop_lag.threshold_ms() == 50.0
