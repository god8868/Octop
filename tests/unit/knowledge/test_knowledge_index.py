"""Behavioural tests for the SQLite knowledge sidecar index."""

from __future__ import annotations

from pathlib import Path

import pytest

from octop.infra.knowledge.index import KnowledgeIndex


@pytest.fixture
def index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KnowledgeIndex:
    """Point ``PathLayout`` at a temp dir so each test gets a clean sidecar."""
    from octop.infra.utils.paths import PathLayout

    monkeypatch.setattr(PathLayout, "from_env", classmethod(lambda _cls: PathLayout(tmp_path)))
    return KnowledgeIndex("kb1")


def _seed(index: KnowledgeIndex, vectors: dict[str, list[float]]) -> None:
    """Store one single-chunk document per entry, keyed by document id."""
    for doc_id, vector in vectors.items():
        index.replace_doc_chunks(
            doc_id,
            [f"text of {doc_id}"],
            [vector],
            metadata=[{"doc": doc_id}],
        )


def test_search_returns_top_k_by_cosine(index: KnowledgeIndex) -> None:
    _seed(
        index,
        {
            "same": [1.0, 0.0, 0.0],
            "near": [0.9, 0.1, 0.0],
            "orthogonal": [0.0, 1.0, 0.0],
        },
    )

    hits = index.search([1.0, 0.0, 0.0], k=2)

    assert [hit.doc_id for hit in hits] == ["same", "near"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.9 / (0.9**2 + 0.1**2) ** 0.5)
    assert hits[0].text == "text of same"
    assert hits[0].metadata == {"doc": "same"}


def test_search_skips_dimension_mismatch(index: KnowledgeIndex) -> None:
    _seed(index, {"good": [1.0, 0.0], "wrong": [1.0, 0.0, 0.0]})

    hits = index.search([1.0, 0.0], k=5)

    assert [hit.doc_id for hit in hits] == ["good"]


def test_search_keeps_zero_norm_chunk_with_zero_score(index: KnowledgeIndex) -> None:
    """A zero vector has no direction, but it must not silently vanish."""
    _seed(index, {"real": [1.0, 0.0], "zero": [0.0, 0.0]})

    hits = index.search([1.0, 0.0], k=5)

    assert [hit.doc_id for hit in hits] == ["real", "zero"]
    assert hits[1].score == 0.0


def test_search_empty_index_returns_empty(index: KnowledgeIndex) -> None:
    assert index.search([1.0, 0.0], k=3) == []


def test_search_k_larger_than_rows(index: KnowledgeIndex) -> None:
    _seed(index, {"a": [1.0, 0.0], "b": [0.0, 1.0]})

    assert len(index.search([1.0, 0.0], k=10)) == 2


def test_search_k_non_positive_returns_empty(index: KnowledgeIndex) -> None:
    _seed(index, {"a": [1.0, 0.0]})

    assert index.search([1.0, 0.0], k=0) == []
    assert index.search([1.0, 0.0], k=-1) == []


def test_search_rejects_empty_and_zero_query(index: KnowledgeIndex) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        index.search([], k=1)
    with pytest.raises(ValueError, match="cannot be zero"):
        index.search([0.0, 0.0], k=1)


def test_search_non_dict_metadata_falls_back_to_empty(index: KnowledgeIndex) -> None:
    index.replace_doc_chunks("doc", ["body"], [[1.0, 0.0]], metadata=[{"x": 1}])
    # Overwrite meta_json with a JSON scalar to exercise the fallback.
    with index._connect() as conn:
        conn.execute("UPDATE chunks SET meta_json = '[]'")

    hits = index.search([1.0, 0.0], k=1)

    assert hits[0].metadata == {}


def test_delete_doc_removes_its_chunks(index: KnowledgeIndex) -> None:
    _seed(index, {"a": [1.0, 0.0], "b": [0.0, 1.0]})

    index.delete_doc("a")

    assert [hit.doc_id for hit in index.search([1.0, 0.0], k=5)] == ["b"]
