"""Bundled Chromium stays off until the user confirms."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from octop.infra.control.browser.bundled import (
    browser_enabled,
    bundled_browser,
    confirm_browser_enabled,
)


def test_missing_marker_is_already_enabled(tmp_path: Path) -> None:
    marker = tmp_path / "missing.json"
    assert bundled_browser(marker) is False
    assert browser_enabled(marker=marker, root=tmp_path) is True


def test_bundled_browser_requires_confirm(tmp_path: Path) -> None:
    marker = tmp_path / "bundled.json"
    marker.write_text(json.dumps({"browser": True}), encoding="utf-8")
    home = tmp_path / "home"
    assert browser_enabled(marker=marker, root=home) is False
    confirm_browser_enabled(home, marker=marker)
    assert (home / "browser.enabled").is_file()
    assert browser_enabled(marker=marker, root=home) is True


def test_confirm_is_a_noop_without_a_bundled_browser(tmp_path: Path) -> None:
    home = tmp_path / "home"
    confirm_browser_enabled(home, marker=tmp_path / "missing.json")
    assert not (home / "browser.enabled").exists()


def test_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from octop.infra.control.browser import bundled as bundled_module

    marker = tmp_path / "env.json"
    marker.write_text(json.dumps({"browser": True}), encoding="utf-8")
    monkeypatch.setenv(bundled_module._ENV_OVERRIDE, str(marker))
    monkeypatch.setattr(bundled_module, "_ETC_MARKER", tmp_path / "no-etc.json")

    assert bundled_browser() is True


def test_install_root_marker_is_discovered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Desktop/portable builds ship the manifest beside the install root."""
    from octop.infra.control.browser import bundled as bundled_module

    home = tmp_path / "home"
    home.mkdir()
    (home / bundled_module._MARKER_NAME).write_text(json.dumps({"browser": True}), encoding="utf-8")
    monkeypatch.delenv(bundled_module._ENV_OVERRIDE, raising=False)
    monkeypatch.setenv("OCTOP_HOME", str(home))
    monkeypatch.setattr(bundled_module, "_ETC_MARKER", tmp_path / "no-etc.json")

    assert bundled_browser() is True


def test_no_marker_anywhere_is_not_bundled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from octop.infra.control.browser import bundled as bundled_module

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv(bundled_module._ENV_OVERRIDE, raising=False)
    monkeypatch.setenv("OCTOP_HOME", str(home))
    monkeypatch.setattr(bundled_module, "_ETC_MARKER", tmp_path / "no-etc.json")

    assert bundled_browser() is False
