"""pgvector chunk index for one knowledge base.

``CREATE EXTENSION vector`` is an operator step (see
``docker/postgres/init-vector.sql`` and ADR 002). This module only creates
``public.knowledge_chunks`` once the ``vector`` type exists.

The table declares ``embedding vector`` without a width because the embedding
model is configuration, not schema. pgvector refuses to index a ``vector`` that
has no declared dimensions, so the ANN index can only be built after the column
is pinned to ``vector(N)`` on the first indexed document.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from octop.config import DatabaseConfig
from octop.infra.db.file_store import KNOWLEDGE_CHUNKS, adopt_legacy_content_tables
from octop.infra.knowledge.index import Hit

logger = logging.getLogger(__name__)

_TABLE = f"public.{KNOWLEDGE_CHUNKS}"
_EMBEDDING_INDEX = "idx_knowledge_chunks_embedding"
# HNSW arrived in pgvector 0.5.0; older servers only have ivfflat.
_HNSW_MIN_VERSION = (0, 5, 0)
# Iterative scan (0.8.0) keeps a filtered graph walk going until k rows match.
_ITERATIVE_SCAN_MIN_VERSION = (0, 8, 0)
# pgvector needs a list count for ivfflat; the default of 100 suits the row
# counts a single control plane holds.
_IVFFLAT_OPTIONS = "WITH (lists = 100)"
_VECTOR_TYPE = re.compile(r"^vector\((\d+)\)$")

# Process-wide memos, keyed by connection string. One DDL attempt per install
# keeps a permission error from retrying on every search, and one warning per
# server keeps an old pgvector from logging on every query.
_INDEX_ATTEMPTED: set[str] = set()
_ITERATIVE_SCAN_DISABLED: set[str] = set()

_METADATA_SQL = f"""
SELECT
    (SELECT extversion FROM pg_extension WHERE extname = 'vector'),
    (SELECT format_type(a.atttypid, a.atttypmod)
       FROM pg_attribute a
      WHERE a.attrelid = '{_TABLE}'::regclass
        AND a.attname = 'embedding'
        AND NOT a.attisdropped),
    (SELECT i.indisvalid
       FROM pg_index i
       JOIN pg_class c ON c.oid = i.indexrelid
      WHERE i.indrelid = '{_TABLE}'::regclass
        AND c.relname = '{_EMBEDDING_INDEX}')
"""


def _version_tuple(raw: object) -> tuple[int, ...]:
    """Parse ``extversion`` into comparable integers, ignoring suffixes."""
    parts: list[int] = []
    for piece in str(raw or "").split("."):
        digits = "".join(character for character in piece if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _at_least(version: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    if not version:
        return False
    padded = version + (0,) * max(0, len(minimum) - len(version))
    return padded >= minimum


def _vector_dims(column_type: object) -> int | None:
    """Declared width of the column, or ``None`` when it is a bare ``vector``."""
    match = _VECTOR_TYPE.match(str(column_type or "").strip())
    return int(match.group(1)) if match else None


def _index_plan(version: tuple[int, ...]) -> tuple[str, str]:
    """Access method and its options for the installed pgvector."""
    if _at_least(version, _HNSW_MIN_VERSION):
        return "hnsw", ""
    return "ivfflat", _IVFFLAT_OPTIONS


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".8g") for value in values) + "]"


class PgVectorKnowledgeIndex:
    """Cosine search over ``public.knowledge_chunks`` for one knowledge base."""

    def __init__(self, kb_id: str, database: DatabaseConfig) -> None:
        self._kb_id = kb_id
        self._database = database
        self._conninfo = database.postgresql_conninfo()
        self._vector_version: tuple[int, ...] = ()
        self._embedding_dims: int | None = None
        self._embedding_index_valid = False
        self._embedding_index_invalid = False
        self._ensure_schema()

    def _connect(self) -> Any:
        import psycopg

        return psycopg.connect(self._conninfo)

    def _ddl_connect(self) -> Any:
        """Autocommit connection: ``CREATE INDEX CONCURRENTLY`` forbids a transaction."""
        import psycopg

        return psycopg.connect(self._conninfo, autocommit=True)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            present = conn.execute("SELECT 1 FROM pg_type WHERE typname = 'vector'").fetchone()
            if present is None:
                raise RuntimeError(
                    "pgvector is not installed on this PostgreSQL. "
                    "Run CREATE EXTENSION vector (docker/postgres/init-vector.sql) "
                    "before using knowledge search with database.driver=postgresql."
                )
            adopt_legacy_content_tables(self._database)
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    kb_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector NOT NULL,
                    meta_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    PRIMARY KEY (kb_id, chunk_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_doc
                ON {_TABLE} (kb_id, doc_id)
                """
            )
            row = conn.execute(_METADATA_SQL).fetchone()
            conn.commit()
        if row is not None:
            self._vector_version = _version_tuple(row[0])
            self._embedding_dims = _vector_dims(row[1])
            self._embedding_index_valid = row[2] is True
            self._embedding_index_invalid = row[2] is False
        self._maybe_create_embedding_index()

    def _pin_embedding_dims(self, dims: int) -> bool:
        """Give the untyped ``embedding`` column a width so it can be indexed.

        This is what makes an ANN index possible at all. ``ALTER COLUMN TYPE``
        rewrites the table under an ACCESS EXCLUSIVE lock, so it only runs while
        the table still holds no chunks; an install that already has data gets
        an actionable warning rather than a multi-minute stall of live searches.
        """
        with self._connect() as conn:
            row = conn.execute(f"SELECT EXISTS (SELECT 1 FROM {_TABLE})").fetchone()
            conn.commit()
        if row is not None and row[0]:
            logger.warning(
                "knowledge chunks stay on sequential scan: %s.embedding has no fixed "
                "dimension and the table already holds rows. Pin it during a maintenance "
                "window, and the ANN index builds itself on the next start: "
                "ALTER TABLE %s ALTER COLUMN embedding TYPE vector(%d) "
                "USING embedding::vector(%d);",
                _TABLE,
                _TABLE,
                dims,
                dims,
            )
            return False
        try:
            with self._ddl_connect() as conn:
                conn.execute(
                    f"ALTER TABLE {_TABLE} ALTER COLUMN embedding TYPE vector({dims}) "
                    f"USING embedding::vector({dims})"
                )
        except Exception as exc:
            logger.warning(
                "could not pin %s.embedding to vector(%d) (%s); knowledge search keeps "
                "working without an ANN index",
                _TABLE,
                dims,
                exc,
            )
            return False
        self._embedding_dims = dims
        logger.info("pinned knowledge chunk embeddings to %d dimensions", dims)
        return True

    def _ensure_embedding_dims(self, dims: int) -> None:
        """Reconcile the column width with the vectors about to be written."""
        if not dims:
            return
        if self._embedding_dims is None:
            self._pin_embedding_dims(dims)
            return
        if self._embedding_dims != dims:
            logger.warning(
                "knowledge chunk dimension mismatch: column is vector(%d) but this "
                "document embedded to %d; the document may fail to index",
                self._embedding_dims,
                dims,
            )

    def _maybe_create_embedding_index(self) -> None:
        """Build the approximate-nearest-neighbour index once the width is known."""
        if self._embedding_dims is None or self._embedding_index_valid:
            return
        if self._conninfo in _INDEX_ATTEMPTED:
            return
        _INDEX_ATTEMPTED.add(self._conninfo)
        kind, options = _index_plan(self._vector_version)
        suffix = f" {options}" if options else ""
        statement = (
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_EMBEDDING_INDEX} "
            f"ON {_TABLE} USING {kind} (embedding vector_cosine_ops){suffix}"
        )
        try:
            with self._ddl_connect() as conn:
                if self._embedding_index_invalid:
                    # A previously interrupted CONCURRENTLY build leaves an
                    # invalid index behind; IF NOT EXISTS would silently keep it.
                    conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_EMBEDDING_INDEX}")
                conn.execute(statement)
        except Exception as exc:
            logger.warning(
                "could not build the pgvector %s index on %s (%s); knowledge search "
                "stays correct but reads the whole table",
                kind,
                _TABLE,
                exc,
            )
            return
        self._embedding_index_valid = True
        logger.info("built pgvector %s index %s", kind, _EMBEDDING_INDEX)

    def _relax_index_scan(self, conn: Any) -> None:
        """Keep the graph walk going until the ``kb_id`` filter is satisfied.

        The index is on ``embedding`` alone — HNSW has no multicolumn form — so
        ``WHERE kb_id`` is applied after the walk and a plain scan can stop early
        with fewer than ``k`` neighbours. Iterative scan fixes that in 0.8+;
        older servers keep the previous behaviour exactly.
        """
        if not self._embedding_index_valid or self._conninfo in _ITERATIVE_SCAN_DISABLED:
            return
        if not _at_least(self._vector_version, _ITERATIVE_SCAN_MIN_VERSION):
            return
        try:
            conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        except Exception:
            _ITERATIVE_SCAN_DISABLED.add(self._conninfo)
            conn.rollback()
            logger.warning(
                "pgvector rejected hnsw.iterative_scan; kb-filtered knowledge search "
                "may return fewer than k hits"
            )

    def replace_doc_chunks(
        self,
        doc_id: str,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        *,
        metadata: Sequence[dict[str, object]] | None = None,
    ) -> None:
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata and texts must have the same length")
        from psycopg.types.json import Jsonb

        if embeddings:
            # Pin the column and build the index before writing: on a fresh
            # install both are instant while the table is empty, and every later
            # insert just maintains the index.
            self._ensure_embedding_dims(len(embeddings[0]))
            self._maybe_create_embedding_index()

        rows: list[tuple[object, ...]] = []
        for ordinal, (text, embedding) in enumerate(zip(texts, embeddings, strict=True)):
            vector = [float(value) for value in embedding]
            if not vector:
                raise ValueError("embedding cannot be empty")
            meta = metadata[ordinal] if metadata is not None else {}
            rows.append(
                (
                    self._kb_id,
                    f"{doc_id}:{ordinal}",
                    doc_id,
                    ordinal,
                    text,
                    _vector_literal(vector),
                    Jsonb(meta),
                )
            )
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {_TABLE} WHERE kb_id = %s AND doc_id = %s",
                (self._kb_id, doc_id),
            )
            if rows:
                with conn.cursor() as cur:
                    cur.executemany(
                        f"""
                        INSERT INTO {_TABLE}
                            (kb_id, chunk_id, doc_id, ordinal, text, embedding, meta_json)
                        VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                        """,
                        rows,
                    )
            conn.commit()

    def list_doc_chunks(self, doc_id: str) -> list[tuple[int, str]]:
        """Return stored slices for one document, in ordinal order."""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT ordinal, text FROM {_TABLE}
                WHERE kb_id = %s AND doc_id = %s
                ORDER BY ordinal
                """,
                (self._kb_id, doc_id),
            ).fetchall()
        return [(int(ordinal), str(text)) for ordinal, text in rows]

    def delete_doc(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {_TABLE} WHERE kb_id = %s AND doc_id = %s",
                (self._kb_id, doc_id),
            )
            conn.commit()

    def delete_kb(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {_TABLE} WHERE kb_id = %s",
                (self._kb_id,),
            )
            conn.commit()

    def search(self, query_vec: Sequence[float], k: int) -> list[Hit]:
        if k <= 0:
            return []
        query = [float(value) for value in query_vec]
        if not query:
            raise ValueError("query vector cannot be empty")
        if all(value == 0 for value in query):
            raise ValueError("query vector cannot be zero")
        literal = _vector_literal(query)
        with self._connect() as conn:
            self._relax_index_scan(conn)
            rows = conn.execute(
                f"""
                SELECT chunk_id, doc_id, ordinal, text, meta_json,
                       1 - (embedding <=> %s::vector) AS score
                FROM {_TABLE}
                WHERE kb_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (literal, self._kb_id, literal, k),
            ).fetchall()
        hits: list[Hit] = []
        for chunk_id, doc_id, ordinal, text, meta_json, score in rows:
            decoded = meta_json
            if isinstance(decoded, str):
                decoded = json.loads(decoded)
            hits.append(
                Hit(
                    chunk_id=str(chunk_id),
                    doc_id=str(doc_id),
                    ordinal=int(ordinal),
                    text=str(text),
                    score=float(score),
                    metadata=decoded if isinstance(decoded, dict) else {},
                )
            )
        return hits
