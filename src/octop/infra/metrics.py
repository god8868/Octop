"""In-memory metrics counters."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class Metrics:
    messages_total: int = 0
    stream_errors_total: int = 0
    cron_runs_total: int = 0
    cron_errors_total: int = 0
    agent_active: int = 0
    # Event-loop stalls: a healthy server keeps both at zero.
    loop_lag_over_100ms_total: int = 0
    loop_lag_max_ms: int = 0
    # Offload pools, so a starved thread pool shows up before users report it.
    blocking_offload_total: int = 0
    blocking_offload_slow_total: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def set(self, name: str, value: int) -> None:
        with self._lock:
            setattr(self, name, value)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "messages_total": self.messages_total,
                "stream_errors_total": self.stream_errors_total,
                "cron_runs_total": self.cron_runs_total,
                "cron_errors_total": self.cron_errors_total,
                "agent_active": self.agent_active,
                "loop_lag_over_100ms_total": self.loop_lag_over_100ms_total,
                "loop_lag_max_ms": self.loop_lag_max_ms,
                "blocking_offload_total": self.blocking_offload_total,
                "blocking_offload_slow_total": self.blocking_offload_slow_total,
            }


METRICS = Metrics()
