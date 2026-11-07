"""``team_jobs`` repository and the startup sweep that retires leftovers."""

from __future__ import annotations

from pathlib import Path

import pytest

from octop.infra.db.migrate import run_migrations
from octop.infra.db.pool import SqlitePool
from octop.infra.db.repos.team_jobs import ACTIVE, DONE, ORPHANED, TeamJobsRepo


@pytest.fixture
def db(tmp_path: Path) -> SqlitePool:
    pool = SqlitePool(tmp_path / "octop.db")
    run_migrations(pool)
    yield pool
    pool.close()


def _rows(db: SqlitePool) -> list[tuple[str, str, object]]:
    with db.connect() as conn:
        return [
            (str(row["job_id"]), str(row["status"]), row["finished_at"])
            for row in conn.execute("SELECT job_id, status, finished_at FROM team_jobs").fetchall()
        ]


def test_record_begin_creates_an_active_row(db: SqlitePool) -> None:
    repo = TeamJobsRepo(db)

    repo.record_begin(job_id="job-1", team_agent_id="team-1", member_agent_id="member-1")

    assert repo.status_of("job-1") == ACTIVE
    assert len(_rows(db)) == 1


def test_record_begin_is_idempotent(db: SqlitePool) -> None:
    repo = TeamJobsRepo(db)

    repo.record_begin(job_id="job-1", team_agent_id="team-1", member_agent_id="member-1")
    repo.record_begin(job_id="job-1", team_agent_id="team-1", member_agent_id="member-1")

    assert len(_rows(db)) == 1


def test_record_end_marks_done(db: SqlitePool) -> None:
    repo = TeamJobsRepo(db)
    repo.record_begin(job_id="job-1", team_agent_id="team-1", member_agent_id="member-1")

    repo.record_end(job_id="job-1", finished_at=1234)

    assert repo.status_of("job-1") == DONE
    assert _rows(db) == [("job-1", DONE, 1234)]


def test_record_end_of_an_unknown_job_writes_nothing(db: SqlitePool) -> None:
    repo = TeamJobsRepo(db)

    repo.record_end(job_id="never-started")

    assert _rows(db) == []


def test_retire_active_marks_leftovers_orphaned(db: SqlitePool) -> None:
    repo = TeamJobsRepo(db)
    repo.record_begin(job_id="job-1", team_agent_id="team-1", member_agent_id="member-1")
    repo.record_begin(job_id="job-2", team_agent_id="team-1", member_agent_id="member-2")
    repo.record_end(job_id="job-2", finished_at=10)

    retired = repo.retire_active(retired_at=99)

    assert retired == 1
    assert repo.status_of("job-1") == ORPHANED
    assert repo.status_of("job-2") == DONE


def test_an_orphan_is_never_rewritten_as_done(db: SqlitePool) -> None:
    repo = TeamJobsRepo(db)
    repo.record_begin(job_id="job-1", team_agent_id="team-1", member_agent_id="member-1")
    repo.retire_active(retired_at=99)

    # A late reply for the orphaned turn must not fabricate a clean finish.
    repo.record_end(job_id="job-1", finished_at=200)

    assert repo.status_of("job-1") == ORPHANED
    assert _rows(db) == [("job-1", ORPHANED, 99)]


def test_startup_retires_rows_the_previous_process_left_active(db: SqlitePool) -> None:
    repo = TeamJobsRepo(db)
    repo.record_begin(job_id="job-1", team_agent_id="team-1", member_agent_id="member-1")

    run_migrations(db)

    assert repo.status_of("job-1") == ORPHANED


def test_startup_leaves_finished_rows_alone(db: SqlitePool) -> None:
    repo = TeamJobsRepo(db)
    repo.record_begin(job_id="job-1", team_agent_id="team-1", member_agent_id="member-1")
    repo.record_end(job_id="job-1", finished_at=10)

    run_migrations(db)

    assert _rows(db) == [("job-1", DONE, 10)]
