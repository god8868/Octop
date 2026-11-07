"""Channel plugin fingerprints must track changes without reading every asset."""

from __future__ import annotations

import os
from pathlib import Path

from octop.infra.agents.plugins.manager import _channel_fingerprint


def _write(root: Path, rel: str, data: bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_fingerprint_changes_when_size_changes(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    _write(plugin, "assets/app.js", b"x" * 10)
    before = _channel_fingerprint(plugin)

    _write(plugin, "assets/app.js", b"x" * 11)

    assert _channel_fingerprint(plugin) != before


def test_fingerprint_changes_when_mtime_changes(tmp_path: Path) -> None:
    """Same size, rewritten file: mtime is what catches it."""
    plugin = tmp_path / "plugin"
    target = _write(plugin, "assets/app.js", b"abcd")
    os.utime(target, (1_600_000_000, 1_600_000_000))
    before = _channel_fingerprint(plugin)

    os.utime(target, (1_600_000_100, 1_600_000_100))

    assert _channel_fingerprint(plugin) != before


def test_fingerprint_is_stable_across_calls(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    _write(plugin, "plugin.yaml", b"id: demo\n")
    _write(plugin, "assets/app.js", b"console.log(1)\n")

    assert _channel_fingerprint(plugin) == _channel_fingerprint(plugin)


def test_fingerprint_changes_when_a_file_is_added(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    _write(plugin, "plugin.yaml", b"id: demo\n")
    before = _channel_fingerprint(plugin)

    _write(plugin, "assets/extra.css", b"body{}")

    assert _channel_fingerprint(plugin) != before


def test_fingerprint_ignores_pycache_and_git(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    _write(plugin, "plugin.yaml", b"id: demo\n")
    before = _channel_fingerprint(plugin)

    _write(plugin, "__pycache__/mod.cpython-312.pyc", b"junk")
    _write(plugin, ".git/HEAD", b"ref: refs/heads/main\n")

    assert _channel_fingerprint(plugin) == before


def test_fingerprint_of_empty_dir_is_constant(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()

    assert _channel_fingerprint(plugin) == _channel_fingerprint(plugin)
