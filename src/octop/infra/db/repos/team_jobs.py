"""Data-access layer for the team job audit trail.

``TeamJobTracker`` keeps roster locks in process memory only; this table exists
so a restart can explain and clean up after itself. It is never read while
deciding whether a member is busy — see ``docs`` in ``teams/jobs.py``.
"""

from __future__ import annotations

import time

from octop.infra.db.pool import DatabasePool

ACTIVE = "active"
DONE = "done"
ORPHANED = "orphaned"


class TeamJobsRepo:
    """Persist host→member dispatch start/finish events."""

    def __init__(self, db: DatabasePool) -> None:
        self._db = db

    def record_begin(
        self,
        *,
        job_id: str,
        team_agent_id: str,
        member_agent_id: str,
        started_at: int | None = None,
    ) -> None:
        """Insert an ``active`` row; a replayed job id keeps the original row."""
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO team_jobs"
                "(job_id, team_agent_id, member_agent_id, status, created_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (job_id) DO NOTHING",
                (
                    job_id,
                    team_agent_id,
                    member_agent_id,
                    ACTIVE,
                    started_at if started_at is not None else int(time.time()),
                ),
            )

    def record_end(self, *, job_id: str, finished_at: int | None = None) -> None:
        """Close an ``active`` row.

        Rows already retired as orphaned stay orphaned: their real outcome is
        unknown, and pretending they finished would fabricate history.
        """
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE team_jobs SET status = ?, finished_at = ? WHERE job_id = ? AND status = ?",
                (
                    DONE,
                    finished_at if finished_at is not None else int(time.time()),
                    job_id,
                    ACTIVE,
                ),
            )

    def retire_active(self, *, retired_at: int | None = None) -> int:
        """Mark every leftover ``active`` row as orphaned and report how many."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE team_jobs SET status = ?, finished_at = ? WHERE status = ?",
                (
                    ORPHANED,
                    retired_at if retired_at is not None else int(time.time()),
                    ACTIVE,
                ),
            )
            return int(getattr(cursor, "rowcount", 0) or 0)

    def status_of(self, job_id: str) -> str | None:
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT status FROM team_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return None if row is None else str(row[0])
