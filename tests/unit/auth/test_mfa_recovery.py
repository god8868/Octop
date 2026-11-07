"""Recovery code atomic single-spend, and the cost of hashing them."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from octop.infra.auth.mfa import recovery as recovery_mod
from octop.infra.db.migrate import run_migrations
from octop.infra.db.pool import SqlitePool
from octop.infra.db.repos.user_recovery_codes import UserRecoveryCodeRepo
from octop.infra.db.repos.users import UserRepo
from octop.infra.users.password import hash_password


@pytest.fixture()
def repo(tmp_path: Path):
    pool = SqlitePool(tmp_path / "octop.db")
    run_migrations(pool)
    users = UserRepo(pool)
    uid = users.create(
        username="alice",
        password_hash=hash_password("Password1"),
        role="user",
    )
    return uid, UserRecoveryCodeRepo(pool)


async def test_recovery_atomic_double_spend(repo) -> None:
    uid, codes_repo = repo
    plain = await recovery_mod.issue_codes(codes_repo, uid)
    assert len(plain) == recovery_mod.RECOVERY_CODE_COUNT
    code = plain[0]
    assert await recovery_mod.try_spend_code(codes_repo, uid, code)
    assert not await recovery_mod.try_spend_code(codes_repo, uid, code)
    assert codes_repo.count_unused(uid) == recovery_mod.RECOVERY_CODE_COUNT - 1


async def test_issuing_codes_does_not_freeze_the_event_loop(repo) -> None:
    """Ten argon2 hashes are ~660ms of CPU; the loop must keep turning.

    A ticker counts how often the loop got to reschedule while the hashes were
    being computed. Inline, it would be 0 or 1 — the whole server stops for the
    duration. Offloaded, it is dozens. The threshold is deliberately far below
    the expected count and far above the broken case, so a slow CI box cannot
    turn this into a flake while a regression cannot slip through.
    """
    uid, codes_repo = repo
    ticks = 0
    running = True

    async def ticker() -> None:
        nonlocal ticks
        while running:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await recovery_mod.issue_codes(codes_repo, uid)
    finally:
        running = False
        await task

    assert ticks >= 10, f"event loop only got {ticks} turns during ten argon2 hashes"
