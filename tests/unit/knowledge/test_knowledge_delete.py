"""Knowledge-base deletion must stay retryable when the file side fails."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from octop.infra.db.migrate import run_migrations
from octop.infra.db.pool import SqlitePool
from octop.infra.db.repos.knowledge import KnowledgeRepo
from octop.infra.db.repos.settings import SettingsRepo
from octop.infra.db.repos.users import UserRepo
from octop.infra.knowledge import service as service_module
from octop.infra.knowledge.files import (
    delete_knowledge_base_files,
    knowledge_base_dir,
)
from octop.infra.knowledge.service import KnowledgeService
from octop.infra.utils.paths import PathLayout


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KnowledgeService:
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path / "home"))
    pool = SqlitePool(tmp_path / "octop.db")
    run_migrations(pool)
    services = SimpleNamespace(
        knowledge_repo=KnowledgeRepo(pool),
        settings_repo=SettingsRepo(pool),
        user_repo=UserRepo(pool),
        paths=PathLayout.from_env(),
    )
    monkeypatch.setattr(service_module, "assert_knowledge_usable", lambda *_args: None)
    return KnowledgeService(services)


@pytest.fixture
def owned_base(service: KnowledgeService) -> tuple[str, int]:
    """Create a base owned by a fresh user; return ``(kb_id, owner_user_id)``."""
    owner = service._services.user_repo.create(username="owner", password_hash="h", role="user")
    base = service.create_base(owner_user_id=owner, name="Docs", shared=False)
    return base.id, owner


def test_delete_base_removes_knowledge_dir(
    service: KnowledgeService, owned_base: tuple[str, int]
) -> None:
    kb_id, owner = owned_base
    target = knowledge_base_dir(kb_id)
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.sqlite").write_bytes(b"fake index")

    service.delete_base(kb_id, actor_user_id=owner)

    assert not target.exists()
    assert service._services.knowledge_repo.get_base(kb_id) is None


def test_delete_base_keeps_metadata_when_file_delete_fails(
    service: KnowledgeService, owned_base: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file-side failure must abort before the existence anchor is dropped."""
    kb_id, owner = owned_base

    class _BoomBlobs:
        def delete_base(self, _kb_id: str, *, ignore_errors: bool = True) -> None:
            raise OSError("file is locked")

    monkeypatch.setattr(KnowledgeService, "_blobs", lambda _self: _BoomBlobs())

    with pytest.raises(OSError, match="locked"):
        service.delete_base(kb_id, actor_user_id=owner)

    assert service._services.knowledge_repo.get_base(kb_id) is not None


def test_delete_base_survives_group_access_failure(
    service: KnowledgeService,
    owned_base: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ACL reset failure is logged but must not block the removal."""
    kb_id, owner = owned_base
    seen: list[str] = []

    class _BoomGroups:
        def __init__(self, _repos: Any) -> None:
            pass

        def clear(self, _resource: str, target: str) -> None:
            seen.append(target)
            raise RuntimeError("acl down")

    monkeypatch.setattr("octop.infra.users.groups.GroupAccess", _BoomGroups)
    service._services.repos = SimpleNamespace()

    with caplog.at_level(logging.WARNING, logger="octop.infra.knowledge.service"):
        service.delete_base(kb_id, actor_user_id=owner)

    assert seen == [kb_id]
    assert service._services.knowledge_repo.get_base(kb_id) is None
    assert "group access cleanup failed" in caplog.text


def test_delete_base_removes_files_before_metadata(
    service: KnowledgeService, owned_base: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    kb_id, owner = owned_base
    order: list[str] = []

    class _OrderedBlobs:
        def delete_base(self, _kb_id: str, *, ignore_errors: bool = True) -> None:
            order.append(f"blobs(ignore_errors={ignore_errors})")

    repo = service._services.knowledge_repo
    original_delete = repo.delete_base

    def _recording_delete(target: str) -> None:
        order.append("repo")
        original_delete(target)

    monkeypatch.setattr(KnowledgeService, "_blobs", lambda _self: _OrderedBlobs())
    monkeypatch.setattr(repo, "delete_base", _recording_delete)

    service.delete_base(kb_id, actor_user_id=owner)

    assert order == ["blobs(ignore_errors=False)", "repo"]


def test_delete_knowledge_base_files_missing_dir_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A base that never received an upload has no directory; that is not an error."""
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path / "home"))

    delete_knowledge_base_files("never-created")
    delete_knowledge_base_files("never-created", ignore_errors=False)


def test_delete_knowledge_base_files_reports_locked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ignore_errors=False`` must surface a failure instead of hiding it."""
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path / "home"))
    target = knowledge_base_dir("locked")
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.sqlite").write_bytes(b"x")

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("still open")

    monkeypatch.setattr("octop.infra.knowledge.files.shutil.rmtree", _boom)

    with pytest.raises(OSError, match="still open"):
        delete_knowledge_base_files("locked", ignore_errors=False)
