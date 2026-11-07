"""The OpenSandbox SDK install must not run on the event loop.

Saving an admin backend used to download and install the SDK inline, freezing
every other request in the single-process server for the length of the install.
It now runs on the subprocess pool, so the wait cannot starve the DB or IO pools
either.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from octop.api.routers import storage_backends
from octop.infra.backend import opensandbox_deps
from octop.infra.errors import ErrorCode, OctopError
from octop.infra.utils import runtime_packages


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake a missing SDK whose synchronous install flips it to importable."""
    state: dict[str, Any] = {"available": False, "threads": [], "thread_names": [], "calls": 0}

    def fake_install(
        spec: Any,
        *,
        is_satisfied: Any = None,
        import_modules: Any = (),
        timeout: int = 600,
    ) -> str:
        state["calls"] += 1
        state["threads"].append(threading.get_ident())
        state["thread_names"].append(threading.current_thread().name)
        state["available"] = True
        return "installed"

    monkeypatch.setattr(opensandbox_deps, "opensandbox_sdk_available", lambda: state["available"])
    monkeypatch.setattr(runtime_packages, "install_packages", fake_install)
    return state


async def test_the_install_runs_off_the_event_loop(sdk: dict[str, Any]) -> None:
    await storage_backends._ensure_opensandbox_sdk("opensandbox")

    assert sdk["threads"], "the SDK install never ran"
    assert sdk["threads"][0] != threading.get_ident()


async def test_the_install_uses_the_subprocess_pool(sdk: dict[str, Any]) -> None:
    """Not just any thread: the narrowest pool, so a ten-minute install cannot
    take a slot that a knowledge-base query or a repository call needs."""
    await storage_backends._ensure_opensandbox_sdk("opensandbox")

    assert [name.split("_")[0] for name in sdk["thread_names"]] == ["octop-subprocess"]


async def test_an_importable_sdk_installs_nothing(sdk: dict[str, Any]) -> None:
    sdk["available"] = True

    await storage_backends._ensure_opensandbox_sdk("opensandbox")

    assert sdk["calls"] == 0


async def test_other_backend_kinds_install_nothing(sdk: dict[str, Any]) -> None:
    for kind in ("s3", "tos", ""):
        await storage_backends._ensure_opensandbox_sdk(kind)

    assert sdk["calls"] == 0


async def test_a_failed_install_still_maps_to_the_storage_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("pip could not reach the index")

    monkeypatch.setattr(opensandbox_deps, "opensandbox_sdk_available", lambda: False)
    monkeypatch.setattr(runtime_packages, "install_packages", boom)

    with pytest.raises(OctopError) as caught:
        await storage_backends._ensure_opensandbox_sdk("opensandbox")

    assert caught.value.code == ErrorCode.STORAGE_BACKEND_DEPS_FAILED
