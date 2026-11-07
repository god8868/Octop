"""Filesystem layout and safe document persistence for knowledge bases."""

from __future__ import annotations

import shutil
from pathlib import Path

from octop.infra.utils.paths import PathLayout


def knowledge_base_dir(kb_id: str) -> Path:
    return PathLayout.from_env().knowledge_dir / kb_id


def documents_dir(kb_id: str) -> Path:
    path = knowledge_base_dir(kb_id) / "docs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def document_path(kb_id: str, doc_id: str, filename: str) -> Path:
    suffix = Path(filename).suffix.lower()
    return documents_dir(kb_id) / f"{doc_id}{suffix}"


def write_document(kb_id: str, doc_id: str, filename: str, content: bytes) -> Path:
    path = document_path(kb_id, doc_id, filename)
    path.write_bytes(content)
    return path


def delete_document_file(kb_id: str, doc_id: str, filename: str) -> None:
    document_path(kb_id, doc_id, filename).unlink(missing_ok=True)


def delete_knowledge_base_files(kb_id: str, *, ignore_errors: bool = True) -> None:
    """Remove a knowledge base's local directory.

    A base created before its first upload has no directory yet, so a missing
    path is never an error. ``ignore_errors=False`` lets callers that are about
    to drop the metadata row surface permission or file-lock failures instead of
    leaving an orphan directory behind.
    """
    target = knowledge_base_dir(kb_id)
    if not target.exists():
        return
    shutil.rmtree(target, ignore_errors=ignore_errors)
