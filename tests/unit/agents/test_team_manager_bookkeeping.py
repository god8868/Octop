"""Peer-prompt and live-reply bookkeeping stays bounded and self-cleaning."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from octop.infra.agents.teams.team_manager import (
    _LIVE_HOST_REPLY_MAX,
    _PEER_PROMPT_MAX,
    TeamManager,
)


def _make_manager() -> TeamManager:
    return TeamManager(
        agent_manager=SimpleNamespace(is_agent_active=lambda _id: False),
        thread_registry=MagicMock(),
        user_repo=MagicMock(),
        gateway=None,
    )


def test_stash_peer_prompt_evicts_oldest_over_capacity() -> None:
    mgr = _make_manager()

    for index in range(_PEER_PROMPT_MAX + 40):
        mgr.stash_peer_prompt("agent", f"thread-{index}", question="q", dispatch="d")

    assert len(mgr._peer_prompts) == _PEER_PROMPT_MAX
    assert len(mgr._peer_prompt_seq) == _PEER_PROMPT_MAX
    # The 40 oldest entries are the ones that had to go.
    assert ("agent", "thread-0") not in mgr._peer_prompts
    assert ("agent", "thread-39") not in mgr._peer_prompts
    assert ("agent", "thread-40") in mgr._peer_prompts
    assert ("agent", f"thread-{_PEER_PROMPT_MAX + 39}") in mgr._peer_prompts


def test_take_peer_prompt_returns_value_and_clears_seq() -> None:
    mgr = _make_manager()
    mgr.stash_peer_prompt("agent", "thread", question=" hello ", dispatch=" go ")

    assert mgr.take_peer_prompt("agent", "thread") == ("hello", "go")
    assert ("agent", "thread") not in mgr._peer_prompt_seq
    assert mgr.take_peer_prompt("agent", "thread") is None


def test_stash_peer_prompt_ignores_blank_payload() -> None:
    mgr = _make_manager()
    mgr.stash_peer_prompt("agent", "thread", question="  ", dispatch="  ")
    assert mgr._peer_prompts == {}
    mgr.stash_peer_prompt("agent", None, question="q", dispatch="d")
    assert mgr._peer_prompts == {}


def test_discard_peer_prompt_releases_unconsumed_stash() -> None:
    mgr = _make_manager()
    mgr.stash_peer_prompt("agent", "thread", question="q", dispatch="d")

    mgr.discard_peer_prompt("agent", "thread")

    assert mgr._peer_prompts == {}
    assert mgr._peer_prompt_seq == {}


def test_live_host_reply_capacity_is_bounded() -> None:
    mgr = _make_manager()

    for index in range(_LIVE_HOST_REPLY_MAX + 25):
        mgr._live_host_reply_counter += 1
        mgr._live_host_replies[f"room-{index}"] = mgr._live_host_reply_counter
        if len(mgr._live_host_replies) > _LIVE_HOST_REPLY_MAX:
            mgr._evict_oldest_live_replies()

    assert len(mgr._live_host_replies) == _LIVE_HOST_REPLY_MAX
    assert "room-0" not in mgr._live_host_replies
    assert f"room-{_LIVE_HOST_REPLY_MAX + 24}" in mgr._live_host_replies


def test_take_live_host_reply_is_single_use() -> None:
    mgr = _make_manager()
    mgr._live_host_replies["room"] = 1

    assert mgr._take_live_host_reply("room") is True
    assert mgr._take_live_host_reply("room") is False
    assert mgr._take_live_host_reply("") is False
    assert mgr._live_host_replies == {}


def test_evict_noop_below_capacity() -> None:
    mgr = _make_manager()
    mgr._live_host_replies["room"] = 7

    mgr._evict_oldest_live_replies()

    assert mgr._live_host_replies == {"room": 7}
