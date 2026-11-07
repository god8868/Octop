"""Opt-in for a browser that was baked into the image.

The binary can be present without Browser AI being turned on. Confirmation
writes a marker under the Octop home (the data volume in Docker). Hosts
without the image marker keep the previous behavior.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from octop.infra.utils.paths import PathLayout

_ETC_MARKER = Path("/etc/octop/bundled-components.json")
# Kept as a module attribute for callers that import the historical path.
MARKER = _ETC_MARKER
FLAG_NAME = "browser.enabled"
_MARKER_NAME = "bundled-components.json"
_ENV_OVERRIDE = "OCTOP_BUNDLED_COMPONENTS"


def _default_marker() -> Path:
    """First bundled-components manifest that actually exists for this install.

    Docker images bake it under ``/etc/octop``. Desktop and portable builds ship
    it beside the install root instead, where the historical constant never
    resolved — the browser then silently behaved as "not bundled".
    """
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override)
    if os.name != "nt" and _ETC_MARKER.is_file():
        return _ETC_MARKER
    try:
        beside_install = PathLayout.from_env().root / _MARKER_NAME
    except OSError:  # pragma: no cover - unreadable home falls through
        return _ETC_MARKER
    if beside_install.is_file():
        return beside_install
    return _ETC_MARKER


def bundled_browser(marker: Path | None = None) -> bool:
    """True when this install shipped Chromium and still requires a confirm."""
    target = marker if marker is not None else _default_marker()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return bool(isinstance(payload, dict) and payload.get("browser") is True)


def _flag_path(root: Path | None) -> Path:
    home = root if root is not None else PathLayout.from_env().root
    return home / FLAG_NAME


def browser_enabled(*, marker: Path | None = None, root: Path | None = None) -> bool:
    """False only for a bundled browser the user has not confirmed."""
    if not bundled_browser(marker):
        return True
    return _flag_path(root).is_file()


def confirm_browser_enabled(root: Path | None = None, *, marker: Path | None = None) -> None:
    """Record that the user enabled the bundled browser."""
    if not bundled_browser(marker):
        return
    path = _flag_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1\n", encoding="utf-8")
