"""The OpenSandbox SDK install must not run on the event loop.

Saving an admin backend used to download and install the SDK inline, freezing
every other request in the single-process server for the length of the install.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from octop.api.routers import storage_backends
from octop.infra.backend import opensandbox_deps
from octop.infra.errors import ErrorCode, OctopError


@pytest.fixture
def installed_in(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record which thread the install ran on."""
    seen: list[int] = []

    def fake(*, allow_install: bool = True) -> str:
        seen.append(threading.get_ident())
        return "ready"

    monkeypatch.setattr(opensandbox_deps, "ensure_opensandbox_deps", fake)
    return seen


async def test_the_install_runs_off_the_event_loop(installed_in: list[int]) -> None:
    await storage_backends._ensure_opensandbox_sdk("opensandbox")

    assert installed_in, "the SDK install never ran"
    assert installed_in[0] != threading.get_ident()


async def test_other_backend_kinds_install_nothing(installed_in: list[int]) -> None:
    for kind in ("s3", "tos", ""):
        await storage_backends._ensure_opensandbox_sdk(kind)

    assert installed_in == []


async def test_a_failed_install_still_maps_to_the_storage_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*, allow_install: bool = True) -> Any:
        raise RuntimeError("pip could not reach the index")

    monkeypatch.setattr(opensandbox_deps, "ensure_opensandbox_deps", boom)

    with pytest.raises(OctopError) as caught:
        await storage_backends._ensure_opensandbox_sdk("opensandbox")

    assert caught.value.code == ErrorCode.STORAGE_BACKEND_DEPS_FAILED
