"""Team job tracker: roster locks stay in memory, the audit sink is optional."""

from __future__ import annotations

import logging

import pytest

from octop.infra.agents.teams import TeamJobTracker


class _Sink:
    """Records the audit calls the tracker makes."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.fail = False

    def record_begin(self, *, job_id: str, team_agent_id: str, member_agent_id: str) -> None:
        if self.fail:
            raise RuntimeError("database is down")
        self.events.append(("begin", f"{job_id}|{team_agent_id}|{member_agent_id}"))

    def record_end(self, *, job_id: str) -> None:
        if self.fail:
            raise RuntimeError("database is down")
        self.events.append(("end", job_id))


def test_begin_and_end_are_recorded() -> None:
    sink = _Sink()
    jobs = TeamJobTracker(sink)

    jobs.begin("team-1", "member-1", job_id="job-1")
    jobs.end("team-1", "member-1", job_id="job-1")

    assert sink.events == [("begin", "job-1|team-1|member-1"), ("end", "job-1")]


def test_replayed_job_id_is_recorded_once() -> None:
    sink = _Sink()
    jobs = TeamJobTracker(sink)

    jobs.begin("team-1", "member-1", job_id="job-1")
    jobs.begin("team-1", "member-1", job_id="job-1")

    assert sink.events == [("begin", "job-1|team-1|member-1")]
    assert jobs.is_busy("team-1", "member-1") is True


def test_pair_counter_dispatch_is_not_recorded() -> None:
    sink = _Sink()
    jobs = TeamJobTracker(sink)

    jobs.begin("team-1", "member-1")
    jobs.end("team-1", "member-1")

    # Nothing stable to key a row on, and these dispatches never had an inbox id.
    assert sink.events == []
    assert jobs.is_busy("team-1", "member-1") is False


def test_end_of_an_unknown_job_still_closes_the_row() -> None:
    sink = _Sink()
    jobs = TeamJobTracker(sink)

    # A reply can arrive for a job this process never saw (started before a
    # restart); closing the row is what keeps the trail from staying open.
    jobs.end("team-1", "member-1", job_id="job-from-before-restart")

    assert sink.events == [("end", "job-from-before-restart")]


def test_sink_failure_does_not_break_dispatch(caplog: pytest.LogCaptureFixture) -> None:
    sink = _Sink()
    sink.fail = True
    jobs = TeamJobTracker(sink)

    with caplog.at_level(logging.WARNING, logger="octop.infra.agents.teams.jobs"):
        jobs.begin("team-1", "member-1", job_id="job-1")
        assert jobs.is_busy("team-1", "member-1") is True
        jobs.end("team-1", "member-1", job_id="job-1")

    assert jobs.is_busy("team-1", "member-1") is False
    assert len(caplog.records) == 2
    assert all("could not record team job" in record.message for record in caplog.records)


def test_tracker_without_a_sink_behaves_as_before() -> None:
    jobs = TeamJobTracker()

    jobs.begin("team-1", "member-1", job_id="job-1")

    assert jobs.is_busy("team-1", "member-1") is True
    assert jobs.busy_member_ids("team-1") == {"member-1"}
