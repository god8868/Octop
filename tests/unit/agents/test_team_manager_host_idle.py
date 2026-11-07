"""Host-idle waiting must back off instead of polling at a fixed 50ms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from octop.infra.agents.teams import team_manager as team_manager_module
from octop.infra.agents.teams.team_manager import (
    _HOST_IDLE_POLL_MAX_SEC,
    _HOST_IDLE_POLL_SEC,
    TeamManager,
)


class _FakeLoop:
    """Deterministic clock so the wait deadline is reached without real time."""

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now


class _FakeAsyncio:
    """Records requested sleeps and advances the fake clock by the same amount."""

    def __init__(self) -> None:
        self.loop = _FakeLoop()
        self.delays: list[float] = []

    def get_running_loop(self) -> _FakeLoop:
        return self.loop

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)
        self.loop.now += delay


def _make_manager(*, host_busy: Any = False) -> TeamManager:
    """Build a TeamManager whose host-busy probe is fully controllable."""
    probe = host_busy if callable(host_busy) else (lambda _host: host_busy)
    gateway = SimpleNamespace(ws_hub=SimpleNamespace(is_turn_active=lambda _room: False))
    return TeamManager(
        agent_manager=SimpleNamespace(is_agent_active=probe),
        thread_registry=MagicMock(),
        user_repo=MagicMock(),
        gateway=gateway,
    )


@pytest.mark.asyncio
async def test_wait_host_idle_returns_true_without_sleeping(monkeypatch) -> None:
    """An idle host is detected on the first probe — no backoff delay at all."""
    mgr = _make_manager(host_busy=False)
    fake = _FakeAsyncio()
    monkeypatch.setattr(team_manager_module, "asyncio", fake)

    assert await mgr._wait_host_dispatch_idle("HOST", "room") is True
    assert fake.delays == []


@pytest.mark.asyncio
async def test_wait_host_idle_returns_true_once_host_clears(monkeypatch) -> None:
    """Busy → busy → idle needs two waits, doubling from the initial poll."""
    states = iter([True, True, False])
    mgr = _make_manager(host_busy=lambda _host: next(states))
    fake = _FakeAsyncio()
    monkeypatch.setattr(team_manager_module, "asyncio", fake)

    assert await mgr._wait_host_dispatch_idle("HOST", "room") is True
    assert fake.delays == [pytest.approx(0.05), pytest.approx(0.1)]


@pytest.mark.asyncio
async def test_wait_host_idle_backs_off_and_caps(monkeypatch) -> None:
    """A host that never clears must hit the cap and stop at the deadline."""
    mgr = _make_manager(host_busy=True)
    fake = _FakeAsyncio()
    monkeypatch.setattr(team_manager_module, "asyncio", fake)

    assert await mgr._wait_host_dispatch_idle("HOST", "room") is False

    assert fake.delays[0] == pytest.approx(_HOST_IDLE_POLL_SEC)
    assert fake.delays[1] == pytest.approx(0.1)
    assert fake.delays[2] == pytest.approx(0.2)
    assert fake.delays[3] == pytest.approx(_HOST_IDLE_POLL_MAX_SEC)
    assert max(fake.delays) == pytest.approx(_HOST_IDLE_POLL_MAX_SEC)
    # Fixed 50ms polling cost ~900 wakeups for a 45s wait; backoff must be far
    # below that while still reaching the deadline.
    assert len(fake.delays) < 200
    assert fake.loop.now >= 45.0


@pytest.mark.asyncio
async def test_wait_host_idle_never_exceeds_cap(monkeypatch) -> None:
    """Every delay after the first must stay within the documented cap."""
    mgr = _make_manager(host_busy=True)
    fake = _FakeAsyncio()
    monkeypatch.setattr(team_manager_module, "asyncio", fake)

    await mgr._wait_host_dispatch_idle("HOST", "room")
    assert all(delay <= _HOST_IDLE_POLL_MAX_SEC for delay in fake.delays)
