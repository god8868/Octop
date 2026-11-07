"""Guards for the execution boundary: what may block the event loop.

Both rules come from the same failure. A single-process asyncio server does not
degrade gracefully when one handler blocks — every agent turn, stream and health
check stops with it, and the only symptom is "it got slow". These tests make the
boundary explicit so a future edit cannot quietly widen it.

The allowlist in :data:`_DEFAULT_EXECUTOR_BASELINE` is a burn-down list: every
entry is a place that still uses the shared default executor instead of
:func:`octop.infra.utils.offload.run_blocking`. It may only shrink. Converting a
call site means deleting its entry — a stale entry fails the test, so the list
cannot drift out of date the way a comment would.
"""

from __future__ import annotations

import ast
import asyncio
import threading
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

#: Synchronous entry points that must never be called from an ``async def``.
#: ``subprocess.Popen`` is deliberately absent: it returns as soon as the child
#: is forked, so its cost is milliseconds, not unbounded. The unbounded part is
#: ``wait``/``communicate``, and those are reached through helpers this rule
#: does not need to guess about.
_FORBIDDEN_IN_ASYNC = frozenset(
    {
        "install_packages",
        "ensure_opensandbox_deps",
        "ensure_ocr_deps",
        "ensure_local_embedding_deps",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "time.sleep",
        "sleep",
    }
)

#: ``run_in_executor(None, ...)`` still present, keyed by file then function.
#: 23 call sites across 17 files when this guard was written. Only shrinks.
_DEFAULT_EXECUTOR_BASELINE: dict[str, dict[str, int]] = {
    "src/octop/api/routers/auth.py": {"_authenticate_ldap": 1},
    "src/octop/api/routers/auth_ldap.py": {"test_ldap_config": 1},
    "src/octop/api/routers/auth_oauth.py": {
        "oauth_bind_start": 1,
        "oauth_start": 1,
        "test_oauth_provider": 1,
    },
    "src/octop/api/routers/auth_oidc.py": {"oidc_start": 1, "test_oidc_config": 1},
    "src/octop/api/routers/desktop/settings.py": {"set_desktop_geometry": 1},
    "src/octop/api/routers/ollama_models.py": {"_run_pull_in_background": 1},
    "src/octop/api/routers/terminal.py": {"_destroy_session": 1},
    "src/octop/infra/agents/memory/slim.py": {"_run_postgres": 1, "inspect_storage": 1},
    "src/octop/infra/agents/providers/onnx_service.py": {
        "_run": 1,
        "probe_local_model": 1,
    },
    "src/octop/infra/agents/settings/langfuse.py": {"test_connection": 1},
    "src/octop/infra/auth/mfa/reauth.py": {"prove_password": 1},
    "src/octop/infra/auth/sso/service.py": {"handle_callback": 1},
    "src/octop/infra/connectors/gateway/agently_auth.py": {"_spawn": 1},
    "src/octop/infra/connectors/gateway/agently_watch.py": {"_listen": 2},
    "src/octop/infra/knowledge/jobs.py": {"_run": 1},
    "src/octop/infra/knowledge/retrieve.py": {"retrieve_context": 1},
    "src/octop/infra/server.py": {"_boot_runtime": 1},
}


@dataclass(frozen=True, order=True)
class _Sighting:
    file: str
    function: str

    def __str__(self) -> str:
        return f"{self.file}::{self.function}"


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_files() -> list[Path]:
    root = _repo() / "src" / "octop"
    if not root.is_dir():  # pragma: no cover - installed-only test run
        pytest.skip(f"source tree not found at {root}")
    return sorted(root.rglob("*.py"))


def _callee_name(node: ast.Call) -> str:
    """Dotted name of the call target, or ``""`` when it is not a plain name."""
    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    parts.reverse()
    return ".".join(parts)


def _visit(
    node: ast.AST,
    rel: str,
    enclosing: str,
    in_async: bool,
    default_executor: list[_Sighting],
    direct_blocking: list[tuple[_Sighting, str]],
) -> None:
    """Walk the tree, tracking the innermost enclosing function."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        enclosing = node.name
        in_async = isinstance(node, ast.AsyncFunctionDef)
    if isinstance(node, ast.Call):
        name = _callee_name(node)
        if name.endswith("run_in_executor") and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value is None:
                default_executor.append(_Sighting(rel, enclosing))
        if in_async and name in _FORBIDDEN_IN_ASYNC:
            direct_blocking.append((_Sighting(rel, enclosing), name))
    for child in ast.iter_child_nodes(node):
        _visit(child, rel, enclosing, in_async, default_executor, direct_blocking)


@lru_cache(maxsize=1)
def _collect() -> tuple[tuple[_Sighting, ...], tuple[tuple[_Sighting, str], ...]]:
    """Return default-executor uses and forbidden synchronous calls in async code."""
    default_executor: list[_Sighting] = []
    direct_blocking: list[tuple[_Sighting, str]] = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(_repo()).as_posix()
        _visit(tree, rel, "", False, default_executor, direct_blocking)
    return tuple(default_executor), tuple(direct_blocking)


def test_no_default_executor_use_outside_the_baseline() -> None:
    """``run_in_executor(None, ...)`` shares one pool with everything else.

    A ten-minute install on that pool starves unrelated work, so new offloads
    must name a pool via ``run_blocking``. Existing call sites are listed above
    and the list may only shrink.
    """
    found, _ = _collect()
    actual = Counter((sighting.file, sighting.function) for sighting in found)
    allowed = Counter(
        (file, function)
        for file, functions in _DEFAULT_EXECUTOR_BASELINE.items()
        for function, count in functions.items()
        for _ in range(count)
    )

    added = [f"{file}::{fn}" for (file, fn), n in (actual - allowed).items() for _ in range(n)]
    removed = [f"{file}::{fn}" for (file, fn), n in (allowed - actual).items() for _ in range(n)]

    assert not added, (
        "new run_in_executor(None, ...) — use run_blocking(..., pool=Pool.X) instead, "
        f"so this work cannot starve another pool: {sorted(added)}"
    )
    assert not removed, (
        "the baseline still lists call sites that are gone; delete these entries so the "
        f"list keeps shrinking: {sorted(removed)}"
    )


def test_async_code_never_calls_a_blocking_primitive_directly() -> None:
    """These run for seconds to minutes; calling them freezes the whole server."""
    _, violations = _collect()

    assert not violations, "\n".join(
        f"{sighting} calls {name}() directly — await its async twin, or offload it "
        "with run_blocking(..., pool=Pool.SUBPROCESS)"
        for sighting, name in violations
    )


def test_every_install_entry_point_reaches_the_subprocess_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule above is only worth having if the sanctioned path really is off-loop."""
    from octop.infra.agents.providers import onnx_service
    from octop.infra.backend import opensandbox_deps
    from octop.infra.knowledge import ocr
    from octop.infra.utils import runtime_packages

    threads: list[str] = []
    state = {"installed": False}

    def record(*_args: Any, **_kwargs: Any) -> str:
        threads.append(threading.current_thread().name)
        state["installed"] = True
        return "installed"

    monkeypatch.setattr(runtime_packages, "install_packages", record)
    monkeypatch.setattr(ocr, "ensure_ocr_deps", record)
    monkeypatch.setattr(onnx_service, "ensure_local_embedding_deps", record)
    monkeypatch.setattr(opensandbox_deps, "opensandbox_sdk_available", lambda: state["installed"])

    async def scenario() -> None:
        # The SDK facade checks availability before and after installing, so it
        # has to run while the fake still reports "missing" — before any of the
        # other entry points have flipped the flag.
        await opensandbox_deps.ensure_opensandbox_deps_async(allow_install=True)
        await runtime_packages.install_packages_async(("example",))
        await ocr.ensure_ocr_deps_async(backend="onnx")
        await onnx_service.ensure_local_embedding_deps_async(allow_install=True)

    asyncio.run(scenario())

    # install_packages_async, ocr, onnx, and the sdk facade via install_packages_async.
    assert len(threads) == 4, threads
    assert [name.split("_")[0] for name in threads] == ["octop-subprocess"] * 4
