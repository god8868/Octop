"""In-process in-flight peer jobs for team roster locks."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Protocol

logger = logging.getLogger(__name__)


class TeamJobSink(Protocol):
    """Where finished and started dispatches are recorded for later audit."""

    def record_begin(
        self,
        *,
        job_id: str,
        team_agent_id: str,
        member_agent_id: str,
    ) -> None: ...

    def record_end(self, *, job_id: str) -> None: ...


class TeamJobTracker:
    """Tracks in-flight host→member dispatches.

    Prefer a stable ``job_id`` (inbox id) so begin/end is idempotent. Pair
    counters remain for callers that have no job id.

    Roster locks live in this object and nowhere else: a restart still clears
    them. An optional *sink* records start/finish to the ``team_jobs`` table so a
    restart can explain and clean up what it inherited, but the sink is never
    read back — making it a source of truth would let a stale row block or
    release a member forever.
    """

    def __init__(self, sink: TeamJobSink | None = None) -> None:
        self._lock = Lock()
        self._counts: dict[tuple[str, str], int] = {}
        self._jobs: dict[str, tuple[str, str]] = {}
        self._sink = sink

    def begin(
        self,
        team_agent_id: str,
        member_agent_id: str,
        job_id: str | None = None,
    ) -> None:
        key = (team_agent_id, member_agent_id)
        with self._lock:
            if job_id:
                if job_id in self._jobs:
                    return
                self._jobs[job_id] = key
            else:
                self._counts[key] = self._counts.get(key, 0) + 1
        if job_id:
            self._record_begin(job_id, key)

    def end(
        self,
        team_agent_id: str,
        member_agent_id: str,
        job_id: str | None = None,
    ) -> None:
        key = (team_agent_id, member_agent_id)
        with self._lock:
            if job_id:
                self._jobs.pop(job_id, None)
            else:
                current = self._counts.get(key, 0) - 1
                if current <= 0:
                    self._counts.pop(key, None)
                else:
                    self._counts[key] = current
        if job_id:
            self._record_end(job_id)

    def _record_begin(self, job_id: str, key: tuple[str, str]) -> None:
        """Best-effort audit write; dispatch must never fail because of it."""
        sink = self._sink
        if sink is None:
            return
        try:
            sink.record_begin(job_id=job_id, team_agent_id=key[0], member_agent_id=key[1])
        except Exception as exc:
            logger.warning("could not record team job %s start: %s", job_id, exc)

    def _record_end(self, job_id: str) -> None:
        sink = self._sink
        if sink is None:
            return
        try:
            sink.record_end(job_id=job_id)
        except Exception as exc:
            logger.warning("could not record team job %s end: %s", job_id, exc)

    def is_busy(self, team_agent_id: str, member_agent_id: str) -> bool:
        key = (team_agent_id, member_agent_id)
        with self._lock:
            if self._counts.get(key, 0) > 0:
                return True
            return any(pair == key for pair in self._jobs.values())

    def is_member_busy(self, member_agent_id: str) -> bool:
        with self._lock:
            if any(
                count > 0 and member == member_agent_id
                for (_, member), count in self._counts.items()
            ):
                return True
            return any(member == member_agent_id for (_, member) in self._jobs.values())

    def busy_member_ids(self, team_agent_id: str) -> set[str]:
        with self._lock:
            busy = {
                member
                for (team, member), count in self._counts.items()
                if team == team_agent_id and count > 0
            }
            busy.update(member for team, member in self._jobs.values() if team == team_agent_id)
            return busy
