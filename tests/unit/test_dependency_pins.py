"""``octop-harness`` must stay on a bounded major version.

The runtime patches a lot of private harness symbols. Without an upper bound a
routine dependency refresh can jump a major release and break those patches
silently — the symptom is wrong agent behaviour, not an import error at boot.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _dependencies() -> list[str]:
    payload = tomllib.loads((_repo() / "pyproject.toml").read_text(encoding="utf-8"))
    return list(payload["project"]["dependencies"])


def _harness_requirements() -> list[str]:
    return [item for item in _dependencies() if item.startswith("octop-harness")]


def test_octop_harness_keeps_an_upper_bound() -> None:
    requirements = _harness_requirements()

    assert len(requirements) == 1, requirements
    assert re.search(r"<\s*\d", requirements[0]), (
        f"octop-harness needs an upper bound: {requirements[0]!r}"
    )


def test_lock_file_records_the_same_bound() -> None:
    """A pyproject edit without ``uv lock`` leaves ``uv sync --frozen`` broken."""
    requirement = _harness_requirements()[0]
    specifier = re.sub(r"^octop-harness(\[[^\]]*\])?", "", requirement)

    lock = (_repo() / "uv.lock").read_text(encoding="utf-8")

    assert f'specifier = "{specifier}"' in lock
