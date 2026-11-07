"""Measure event-loop stalls and name the request that was running.

Nothing in the server could answer "was the loop blocked, for how long, and
while serving what" — so every concurrency fix was unverifiable and every
"it feels slow" report was unfalsifiable. This samples the drift between the
time a timer was due and the time it actually ran: on a free loop that drift
stays at a few microseconds, and anything above the threshold is time some
synchronous call stole from every other task.

It is deliberately built on a self-rescheduling ``call_later`` rather than a
background task: a pending timer is dropped silently when the loop closes,
while a pending task logs "Task was destroyed but it is pending" on every
server shutdown and on every TestClient teardown.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from fastapi import Request

from octop.infra.metrics import METRICS

logger = logging.getLogger(__name__)

_INSTALL_ATTR = "_octop_loop_lag_installed"
_INTERVAL_S = 0.1
_DEFAULT_THRESHOLD_MS = 100.0
_THRESHOLD_ENV = "OCTOP_LOOP_LAG_MS"

_expected: float | None = None
# ``_expected`` is measured on a specific loop's clock, so a fresh loop cannot
# inherit it: without this the watchdog would decide it is already running and
# silently stop sampling, and the counters would read zero forever — the most
# misleading possible report.
_watchdog_loop: asyncio.AbstractEventLoop | None = None
_in_flight: dict[int, _InFlight] = {}
_counter = 0
# Requests interleave at every ``await``, so the token has to travel with the
# request instead of living in a module global — otherwise one request's exit
# removes another's entry and the original leaks for the process lifetime.
_token: ContextVar[int] = ContextVar("octop_loop_lag_token", default=0)


@dataclass(frozen=True)
class _InFlight:
    label: str
    started: float


def threshold_ms() -> float:
    """Stall length that counts as a problem. Calibrated from Phase 0 data."""
    raw = os.environ.get(_THRESHOLD_ENV, "").strip()
    try:
        return float(raw) if raw else _DEFAULT_THRESHOLD_MS
    except ValueError:
        return _DEFAULT_THRESHOLD_MS


def install(app: Any, server: Any = None) -> None:
    """Record loop stalls for every request served by *app*."""
    if getattr(app, _INSTALL_ATTR, False):
        return
    setattr(app, _INSTALL_ATTR, True)

    @app.middleware("http")  # type: ignore[untyped-decorator]
    async def _loop_lag(request: Request, call_next: Any) -> Any:
        _ensure_watchdog()
        _enter(f"{request.method} {request.url.path}")
        try:
            return await call_next(request)
        finally:
            _leave()


def _enter(label: str) -> None:
    global _counter
    _counter += 1
    _in_flight[_counter] = _InFlight(label=label, started=perf_counter())
    _token.set(_counter)


def _leave() -> None:
    _in_flight.pop(_token.get(), None)


def oldest_in_flight() -> _InFlight | None:
    """The request that has been running longest — the likely stall owner."""
    if not _in_flight:
        return None
    return min(_in_flight.values(), key=lambda item: item.started)


def _sample(now: float, expected: float, *, limit_ms: float | None = None) -> float:
    """Record one drift sample and return the next deadline.

    The next deadline is always ``now + interval``, never ``expected +
    interval``: after a one-second stall a catch-up burst of ten immediate
    callbacks would report nine fake stalls and measure its own backlog.
    """
    drift_ms = (now - expected) * 1000.0
    if drift_ms >= (threshold_ms() if limit_ms is None else limit_ms):
        _report(drift_ms)
    return now + _INTERVAL_S


def _report(drift_ms: float) -> None:
    METRICS.inc("loop_lag_over_100ms_total")
    if drift_ms > METRICS.loop_lag_max_ms:
        METRICS.set("loop_lag_max_ms", int(drift_ms))
    culprit = oldest_in_flight()
    if culprit is None:
        logger.warning("event loop blocked %.0fms (no request in flight)", drift_ms)
        return
    logger.warning(
        "event loop blocked %.0fms while serving %s (that request has been running %.0fms)",
        drift_ms,
        culprit.label,
        (perf_counter() - culprit.started) * 1000.0,
    )


def _ensure_watchdog() -> None:
    """Start sampling on the running loop; safe to call on every request."""
    global _expected, _watchdog_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - install() outside a loop
        return
    if _watchdog_loop is loop and _expected is not None:
        return
    _watchdog_loop = loop
    _expected = loop.time() + _INTERVAL_S
    loop.call_later(_INTERVAL_S, _tick, loop)


def _tick(loop: asyncio.AbstractEventLoop) -> None:
    global _expected
    if _expected is None or loop.is_closed():
        return
    _expected = _sample(loop.time(), _expected)
    loop.call_later(_INTERVAL_S, _tick, loop)


def stop_watchdog() -> None:
    """Stop rescheduling; the loop drops the pending timer when it closes."""
    global _expected, _watchdog_loop
    _expected = None
    _watchdog_loop = None
    _in_flight.clear()
