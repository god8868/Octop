"""Schema, dimension pinning and ANN-index lifecycle for the pgvector index.

There is no PostgreSQL in unit runs, so every connection is a recorder: the
tests assert the statement sequence the module emits, which is exactly the part
that can break silently (an unindexed column looks just like an indexed one).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest

from octop.config import DatabaseConfig
from octop.infra.knowledge import pg_index
from octop.infra.knowledge.pg_index import (
    PgVectorKnowledgeIndex,
    _at_least,
    _index_plan,
    _vector_dims,
    _version_tuple,
)

_DB = DatabaseConfig(driver="postgresql", host="db", port=5432, database="octop", user="u")
_ALTER = "ALTER COLUMN embedding TYPE"
_CREATE_INDEX = "CREATE INDEX CONCURRENTLY"
_ITERATIVE = "SET LOCAL hnsw.iterative_scan"


class _Result:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


class _FakeCursor:
    def __init__(self, owner: _FakeConnection) -> None:
        self._owner = owner

    def executemany(self, sql: str, rows: Any) -> None:
        self._owner.statements.append(" ".join(sql.split()))

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeConnection:
    """Records SQL and answers the catalog probes the module issues."""

    def __init__(
        self,
        *,
        version: str = "0.8.0",
        column_type: str = "vector",
        index_valid: bool | None = None,
        table_has_rows: bool = False,
        fail_on: str | None = None,
        hits: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self.version = version
        self.column_type = column_type
        self.index_valid = index_valid
        self.table_has_rows = table_has_rows
        self.fail_on = fail_on
        self.hits = hits or []
        self.statements: list[str] = []

    def execute(self, sql: str, params: Any = None) -> _Result:
        flat = " ".join(sql.split())
        self.statements.append(flat)
        if self.fail_on and self.fail_on in flat:
            raise RuntimeError("ddl failed")
        if "typname = 'vector'" in flat:
            return _Result([(1,)])
        if "extversion FROM pg_extension" in flat:
            return _Result([(self.version, self.column_type, self.index_valid)])
        if "EXISTS (SELECT 1 FROM" in flat:
            return _Result([(self.table_has_rows,)])
        if "AS score" in flat:
            return _Result(self.hits)
        return _Result([])

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture(autouse=True)
def _clear_module_memos() -> Iterator[None]:
    pg_index._INDEX_ATTEMPTED.clear()
    pg_index._ITERATIVE_SCAN_DISABLED.clear()
    yield
    pg_index._INDEX_ATTEMPTED.clear()
    pg_index._ITERATIVE_SCAN_DISABLED.clear()


def _open(monkeypatch: pytest.MonkeyPatch, conn: _FakeConnection) -> PgVectorKnowledgeIndex:
    monkeypatch.setattr(pg_index, "adopt_legacy_content_tables", lambda database: None)
    monkeypatch.setattr(PgVectorKnowledgeIndex, "_connect", lambda self: conn)
    monkeypatch.setattr(PgVectorKnowledgeIndex, "_ddl_connect", lambda self: conn)
    return PgVectorKnowledgeIndex("kb1", _DB)


def _statements(conn: _FakeConnection, needle: str) -> list[int]:
    return [index for index, sql in enumerate(conn.statements) if needle in sql]


# --- pure helpers -----------------------------------------------------------


def test_version_tuple_parsing() -> None:
    assert _version_tuple("0.8.0") == (0, 8, 0)
    assert _version_tuple("0.7.4") == (0, 7, 4)
    assert _version_tuple("0.5.0-dev") == (0, 5, 0)
    assert _version_tuple(None) == ()
    assert _version_tuple("") == ()


def test_at_least_pads_short_versions() -> None:
    assert _at_least((0, 8), (0, 8, 0)) is True
    assert _at_least((0, 0, 9), (0, 5, 0)) is False
    assert _at_least((), (0, 5, 0)) is False


def test_vector_dims_reads_declared_width() -> None:
    assert _vector_dims("vector(1024)") == 1024
    assert _vector_dims("vector") is None
    assert _vector_dims("") is None


def test_index_plan_prefers_hnsw_and_falls_back() -> None:
    assert _index_plan((0, 8, 0)) == ("hnsw", "")
    method, options = _index_plan((0, 4, 2))
    assert method == "ivfflat"
    assert "lists = 100" in options


# --- dimension pinning ------------------------------------------------------


def test_untyped_column_is_pinned_before_the_first_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConnection(column_type="vector", table_has_rows=False)
    index = _open(monkeypatch, conn)
    assert _statements(conn, _ALTER) == []

    index.replace_doc_chunks("doc", ["a"], [[1.0, 0.0, 0.0]])

    alter = _statements(conn, _ALTER)
    assert len(alter) == 1
    assert "vector(3)" in conn.statements[alter[0]]
    insert = _statements(conn, "INSERT INTO public.knowledge_chunks")
    assert insert and alter[0] < insert[0]


def test_pinned_column_builds_the_hnsw_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConnection(column_type="vector", table_has_rows=False)
    index = _open(monkeypatch, conn)

    index.replace_doc_chunks("doc", ["a"], [[1.0, 0.0, 0.0]])

    created = [conn.statements[i] for i in _statements(conn, _CREATE_INDEX)]
    assert len(created) == 1
    assert "USING hnsw (embedding vector_cosine_ops)" in created[0]
    assert "idx_knowledge_chunks_embedding" in created[0]


def test_existing_rows_skip_the_table_rewrite(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    conn = _FakeConnection(column_type="vector", table_has_rows=True)
    index = _open(monkeypatch, conn)

    with caplog.at_level(logging.WARNING, logger="octop.infra.knowledge.pg_index"):
        index.replace_doc_chunks("doc", ["a"], [[1.0, 0.0, 0.0]])

    assert _statements(conn, _ALTER) == []
    assert _statements(conn, _CREATE_INDEX) == []
    assert any("no fixed dimension" in record.message for record in caplog.records)
    # The write still happens, exactly as before the change.
    assert _statements(conn, "INSERT INTO public.knowledge_chunks")


def test_dimension_mismatch_warns_and_still_writes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    conn = _FakeConnection(column_type="vector(3)")
    index = _open(monkeypatch, conn)

    with caplog.at_level(logging.WARNING, logger="octop.infra.knowledge.pg_index"):
        index.replace_doc_chunks("doc", ["a"], [[1.0, 0.0, 0.0, 0.0]])

    assert any("dimension mismatch" in record.message for record in caplog.records)
    assert _statements(conn, "INSERT INTO public.knowledge_chunks")


# --- index building ---------------------------------------------------------


def test_old_pgvector_uses_ivfflat(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConnection(version="0.4.2", column_type="vector", table_has_rows=False)
    index = _open(monkeypatch, conn)

    index.replace_doc_chunks("doc", ["a"], [[1.0, 0.0]])

    created = [conn.statements[i] for i in _statements(conn, _CREATE_INDEX)]
    assert len(created) == 1
    assert "USING ivfflat" in created[0]
    assert "lists = 100" in created[0]


def test_typed_column_builds_the_index_without_touching_the_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConnection(column_type="vector(2)")
    _open(monkeypatch, conn)

    assert _statements(conn, _ALTER) == []
    assert len(_statements(conn, _CREATE_INDEX)) == 1


def test_invalid_index_is_dropped_before_the_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConnection(column_type="vector(2)", index_valid=False)
    _open(monkeypatch, conn)

    dropped = _statements(conn, "DROP INDEX CONCURRENTLY")
    created = _statements(conn, _CREATE_INDEX)
    assert len(dropped) == 1 and len(created) == 1
    assert dropped[0] < created[0]


def test_valid_index_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConnection(column_type="vector(2)", index_valid=True)
    _open(monkeypatch, conn)

    assert _statements(conn, _CREATE_INDEX) == []
    assert _statements(conn, "DROP INDEX CONCURRENTLY") == []


def test_index_build_failure_keeps_writes_working(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    conn = _FakeConnection(column_type="vector", table_has_rows=False, fail_on=_CREATE_INDEX)
    index = _open(monkeypatch, conn)

    with caplog.at_level(logging.WARNING, logger="octop.infra.knowledge.pg_index"):
        index.replace_doc_chunks("doc", ["a"], [[1.0, 0.0]])

    assert any("could not build" in record.message for record in caplog.records)
    assert _statements(conn, "INSERT INTO public.knowledge_chunks")


def test_index_attempt_is_not_repeated_for_the_same_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConnection(column_type="vector", table_has_rows=False, fail_on=_CREATE_INDEX)
    first = _open(monkeypatch, conn)
    first.replace_doc_chunks("doc", ["a"], [[1.0, 0.0]])
    second = _open(monkeypatch, conn)
    second.replace_doc_chunks("doc", ["a"], [[1.0, 0.0]])

    assert len(_statements(conn, _CREATE_INDEX)) == 1


# --- search -----------------------------------------------------------------


def test_search_requests_iterative_scan_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConnection(
        version="0.8.0",
        column_type="vector(3)",
        index_valid=True,
        hits=[("doc:0", "doc", 0, "honey", {}, 0.9)],
    )
    index = _open(monkeypatch, conn)

    hits = index.search([1.0, 0.0, 0.0], k=1)

    relaxed = _statements(conn, _ITERATIVE)
    scored = _statements(conn, "AS score")
    assert len(relaxed) == 1 and len(scored) == 1
    assert relaxed[0] < scored[0]
    assert hits[0].text == "honey"


def test_search_skips_iterative_scan_on_old_pgvector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConnection(version="0.7.4", column_type="vector(3)", index_valid=True)
    index = _open(monkeypatch, conn)

    index.search([1.0, 0.0, 0.0], k=1)

    assert _statements(conn, _ITERATIVE) == []


def test_search_skips_iterative_scan_without_an_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConnection(version="0.8.0", column_type="vector", index_valid=None)
    index = _open(monkeypatch, conn)

    index.search([1.0, 0.0, 0.0], k=1)

    assert _statements(conn, _ITERATIVE) == []


def test_search_keeps_working_when_the_guc_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConnection(
        version="0.8.0",
        column_type="vector(3)",
        index_valid=True,
        fail_on=_ITERATIVE,
        hits=[("doc:0", "doc", 0, "honey", {}, 0.9)],
    )
    index = _open(monkeypatch, conn)

    first = index.search([1.0, 0.0, 0.0], k=1)
    second = index.search([1.0, 0.0, 0.0], k=1)

    assert first[0].text == "honey" and second[0].text == "honey"
    # The rejection is remembered: the second query does not retry the GUC.
    assert conn.statements.count("SET LOCAL hnsw.iterative_scan = relaxed_order") == 1
