"""On-demand install of the official ``opensandbox`` SDK."""

from __future__ import annotations

import importlib.util

from octop.infra.utils.runtime_packages import (
    PackageInstallSpec,
    install_packages,
    install_packages_async,
)

_OPENSANDBOX_SPEC = PackageInstallSpec(packages=("opensandbox>=0.1.16,<0.2",))

_MISSING_MSG = "OpenSandbox SDK is not installed. Enable this backend to install it automatically."
_LOAD_FAILED_MSG = (
    "OpenSandbox SDK was installed but could not be loaded. Restart the server and try again."
)


def opensandbox_sdk_available() -> bool:
    return importlib.util.find_spec("opensandbox") is not None


def ensure_opensandbox_deps(*, allow_install: bool = True) -> str:
    """Ensure the OpenSandbox SDK can be imported.

    Admin enable / probe / agent start pass ``allow_install=True``. Returns
    ``\"ready\"`` or ``\"installed\"``. Also installs filesystem helpers when the
    pinned harness build does not provide them.
    """
    from octop.infra.backend.opensandbox_mkdir import install_opensandbox_fs

    install_opensandbox_fs()
    if opensandbox_sdk_available():
        return "ready"
    if not allow_install:
        raise RuntimeError(_MISSING_MSG)
    outcome = install_packages(
        _OPENSANDBOX_SPEC,
        is_satisfied=opensandbox_sdk_available,
        import_modules=("opensandbox",),
    )
    if not opensandbox_sdk_available():
        raise RuntimeError(_LOAD_FAILED_MSG)
    return outcome


async def ensure_opensandbox_deps_async(*, allow_install: bool = True) -> str:
    """Async twin of :func:`ensure_opensandbox_deps` for request handlers.

    Only the install leaves the event loop. Both checks here are cheap by
    construction — ``find_spec`` never executes the module, and the fs helpers
    are attribute assignments — so offloading the whole function would buy
    nothing and cost an extra thread hop.

    The sync twin stays for the two callers that are themselves synchronous
    (backend probe, agent workspace fallback).
    """
    from octop.infra.backend.opensandbox_mkdir import install_opensandbox_fs

    install_opensandbox_fs()
    if opensandbox_sdk_available():
        return "ready"
    if not allow_install:
        raise RuntimeError(_MISSING_MSG)
    outcome = await install_packages_async(
        _OPENSANDBOX_SPEC,
        is_satisfied=opensandbox_sdk_available,
        import_modules=("opensandbox",),
    )
    if not opensandbox_sdk_available():
        raise RuntimeError(_LOAD_FAILED_MSG)
    return outcome
