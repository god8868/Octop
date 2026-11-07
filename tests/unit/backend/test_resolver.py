"""Tests for backend resolver helpers."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from deepagents.backends import CompositeBackend
from octop_harness.backends import resolve_backend
from octop_harness.backends.workspace import BackendWorkspace

from octop.infra.backend.resolver import (
    _is_host_root,
    backend_spec_supports_execution,
    default_agent_backend_spec,
    resolve_agent_backend_spec,
    windows_neutralize_host_root,
)


def test_default_agent_backend_spec_posix_uses_host_root(tmp_path: Path) -> None:
    ws = tmp_path / "agents" / "AGT001"
    ws.mkdir(parents=True)
    with patch("octop.infra.backend.resolver.os", SimpleNamespace(name="posix")):
        spec = default_agent_backend_spec(ws)
    assert spec == {"type": "local_shell", "root_dir": "/", "virtual_mode": True}


@pytest.mark.skipif(os.name != "posix", reason="POSIX host-root default cwd is '/'")
def test_default_agent_backend_resolve_host_root_wraps_artifacts(tmp_path: Path) -> None:
    ws = tmp_path / "agents" / "AGT001"
    ws.mkdir(parents=True)
    backend = resolve_backend(default_agent_backend_spec(ws), workspace_dir=ws)

    assert isinstance(backend, CompositeBackend)
    assert backend.artifacts_root == str(ws.resolve())
    assert backend_spec_supports_execution({"type": "local_shell", "root_dir": "/"})
    assert str(getattr(backend, "cwd", None)) == "/"

    history = ws / "conversation_history" / "thread.md"
    result = backend.write(str(history), "context summary")
    assert result.error is None
    assert history.read_text(encoding="utf-8") == "context summary"


def test_default_agent_backend_resolve_workspace_file_ops(tmp_path: Path) -> None:
    """Workspace mkdir/move/delete via BackendWorkspace — both host-root and scoped defaults."""
    ws = tmp_path / "agents" / "AGT001"
    ws.mkdir(parents=True)
    backend = resolve_backend(default_agent_backend_spec(ws), workspace_dir=ws)
    cwd = getattr(backend, "cwd", None)
    if os.name == "nt":
        assert Path(str(cwd)).resolve() == ws.resolve()
    else:
        assert str(cwd) == "/"
    workspace = BackendWorkspace(backend, ws)
    workspace.mkdir("source")
    (ws / "source" / "note.txt").write_text("ok", encoding="utf-8")
    workspace.move("source", "moved")
    assert (ws / "moved" / "note.txt").read_text(encoding="utf-8") == "ok"
    workspace.delete("moved")
    assert not (ws / "moved").exists()


def test_default_agent_backend_spec_windows_scopes_to_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "agents" / "AGT001"
    ws.mkdir(parents=True)
    with patch("octop.infra.backend.resolver.os", SimpleNamespace(name="nt")):
        spec = default_agent_backend_spec(ws)
    assert spec == {
        "type": "local_shell",
        "root_dir": str(ws.resolve()),
        "virtual_mode": True,
    }


def test_is_host_root_matches_explicit_host_roots() -> None:
    for root in ("/", "\\", "", "  /  "):
        assert _is_host_root(root), root


def test_is_host_root_ignores_drive_root_and_missing() -> None:
    # An explicit drive root is a deliberate user choice, not the dashboard sentinel.
    assert not _is_host_root("D:\\")
    assert not _is_host_root("C:/")
    # harness falls back to workspace_dir on its own; do not rewrite absent config.
    assert not _is_host_root(None)


def test_windows_neutralize_is_passthrough_on_posix(tmp_path: Path) -> None:
    spec = {"type": "local_shell", "root_dir": "/", "virtual_mode": True}
    with patch("octop.infra.backend.resolver.os", SimpleNamespace(name="posix")):
        out = windows_neutralize_host_root(spec, workspace_dir=tmp_path)
    assert out == spec


def test_windows_neutralize_local_shell_host_root_scopes_to_workspace(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "agents" / "AGT001"
    ws.mkdir(parents=True)
    with patch("octop.infra.backend.resolver.os", SimpleNamespace(name="nt")):
        out = windows_neutralize_host_root(
            {"type": "local_shell", "root_dir": "/", "virtual_mode": True},
            workspace_dir=ws,
        )
    assert out == {
        "type": "local_shell",
        "root_dir": str(ws.resolve()),
        "virtual_mode": True,
    }


def test_windows_neutralize_filesystem_empty_root_scopes_to_workspace(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "agents" / "AGT001"
    ws.mkdir(parents=True)
    with patch("octop.infra.backend.resolver.os", SimpleNamespace(name="nt")):
        out = windows_neutralize_host_root(
            {"type": "filesystem", "root_dir": "", "virtual_mode": True},
            workspace_dir=ws,
        )
    # Preserve filesystem type; only rewrite the host-root sentinel.
    assert out == {
        "type": "filesystem",
        "root_dir": str(ws.resolve()),
        "virtual_mode": True,
    }


def test_windows_neutralize_keeps_explicit_drive_root(tmp_path: Path) -> None:
    spec = {"type": "local_shell", "root_dir": "D:\\develop", "virtual_mode": True}
    with patch("octop.infra.backend.resolver.os", SimpleNamespace(name="nt")):
        out = windows_neutralize_host_root(spec, workspace_dir=tmp_path)
    assert out == spec


def test_windows_neutralize_keeps_missing_root_dir(tmp_path: Path) -> None:
    spec = {"type": "local_shell", "virtual_mode": True}
    with patch("octop.infra.backend.resolver.os", SimpleNamespace(name="nt")):
        out = windows_neutralize_host_root(spec, workspace_dir=tmp_path)
    assert out == spec


def test_windows_neutralize_composite_default_host_root_scoped(tmp_path: Path) -> None:
    ws = tmp_path / "agents" / "AGT001"
    ws.mkdir(parents=True)
    spec: dict[str, Any] = {
        "type": "composite",
        "default": {"type": "local_shell", "root_dir": "/", "virtual_mode": True},
        "routes": {},
    }
    with patch("octop.infra.backend.resolver.os", SimpleNamespace(name="nt")):
        out = windows_neutralize_host_root(spec, workspace_dir=ws)
    assert out["default"] == {
        "type": "local_shell",
        "root_dir": str(ws.resolve()),
        "virtual_mode": True,
    }
    assert out["routes"] == {}


def test_windows_neutralize_composite_healthy_default_kept(tmp_path: Path) -> None:
    spec: dict[str, Any] = {
        "type": "composite",
        "default": {"type": "local_shell", "virtual_mode": True},
        "routes": {
            "/project/": {
                "type": "local_shell",
                "root_dir": "D:\\develop",
                "virtual_mode": True,
            }
        },
    }
    with patch("octop.infra.backend.resolver.os", SimpleNamespace(name="nt")):
        out = windows_neutralize_host_root(spec, workspace_dir=tmp_path)
    assert out == spec


# ---------------------------------------------------------------------------
# resolve_agent_backend_spec — named refs, cycles, nesting depth
# ---------------------------------------------------------------------------


def _fake_repo(specs: dict[str, Any]) -> Any:
    """Repo stub: ``get_by_name`` returns a marker keyed by the stored spec map."""
    return SimpleNamespace(get_by_name=lambda name: name if name in specs else None)


def _patch_rows(monkeypatch: pytest.MonkeyPatch, specs: dict[str, Any]) -> None:
    """Replace the row→spec adapter so tests describe backends as plain dicts."""
    monkeypatch.setattr(
        "octop.infra.backend.resolver.row_to_backend_spec",
        lambda row: specs.get(row),
    )


def _resolve(spec: Any, specs: dict[str, Any]) -> Any:
    return resolve_agent_backend_spec(spec, repo=_fake_repo(specs))


def test_resolve_self_reference_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs: dict[str, Any] = {"A": {"type": "named", "name": "A"}}
    _patch_rows(monkeypatch, specs)

    with pytest.raises(ValueError, match="cycle"):
        _resolve({"type": "named", "name": "A"}, specs)


def test_resolve_two_node_cycle_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs: dict[str, Any] = {
        "A": {"type": "named", "name": "B"},
        "B": {"type": "named", "name": "A"},
    }
    _patch_rows(monkeypatch, specs)

    with pytest.raises(ValueError, match="A -> B -> A"):
        _resolve({"type": "named", "name": "A"}, specs)


def test_resolve_three_node_cycle_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs: dict[str, Any] = {
        "A": {"type": "named", "name": "B"},
        "B": {"type": "named", "name": "C"},
        "C": {"type": "named", "name": "A"},
    }
    _patch_rows(monkeypatch, specs)

    with pytest.raises(ValueError, match="A -> B -> C -> A"):
        _resolve({"type": "named", "name": "A"}, specs)


def test_resolve_diamond_reference_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two branches reaching the same backend is legal; only cycles are not."""
    specs: dict[str, Any] = {
        "A": {
            "type": "composite",
            "default": {"type": "named", "name": "B"},
            "routes": {"/c/": {"type": "named", "name": "C"}},
        },
        "B": {"type": "composite", "default": {"type": "named", "name": "D"}, "routes": {}},
        "C": {"type": "composite", "default": {"type": "named", "name": "D"}, "routes": {}},
        "D": {"type": "local_shell", "root_dir": "/shared", "virtual_mode": True},
    }
    _patch_rows(monkeypatch, specs)

    resolved = _resolve({"type": "named", "name": "A"}, specs)
    assert resolved["default"]["default"]["root_dir"] == "/shared"
    assert resolved["routes"]["/c/"]["default"]["root_dir"] == "/shared"


def test_resolve_composite_route_cycle_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs: dict[str, Any] = {
        "A": {
            "type": "composite",
            "default": {"type": "local_shell", "root_dir": "/", "virtual_mode": True},
            "routes": {"/loop/": {"type": "named", "name": "A"}},
        },
    }
    _patch_rows(monkeypatch, specs)

    with pytest.raises(ValueError, match="cycle"):
        _resolve({"type": "named", "name": "A"}, specs)


def test_resolve_named_chain_within_depth_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs: dict[str, Any] = {
        "A": {"type": "named", "name": "B"},
        "B": {"type": "named", "name": "C"},
        "C": {"type": "named", "name": "D"},
        "D": {"type": "named", "name": "E"},
        "E": {"type": "local_shell", "root_dir": "/end", "virtual_mode": True},
    }
    _patch_rows(monkeypatch, specs)

    assert _resolve({"type": "named", "name": "A"}, specs)["root_dir"] == "/end"


def test_resolve_nesting_over_depth_limit_raises() -> None:
    spec: Any = {"type": "local_shell", "root_dir": "/leaf", "virtual_mode": True}
    for _ in range(18):
        spec = {"type": "composite", "default": spec, "routes": {}}

    with pytest.raises(ValueError, match="nesting exceeds"):
        resolve_agent_backend_spec(spec, repo=None)


def test_resolve_plain_spec_is_returned_clean() -> None:
    """Non-named specs keep their fields but drop a stray ``name``."""
    assert resolve_agent_backend_spec(
        {"type": "local_shell", "root_dir": "/x", "name": "stale"}, repo=None
    ) == {"type": "local_shell", "root_dir": "/x"}
