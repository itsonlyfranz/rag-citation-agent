"""Fallback search: a provider protocol plus a local snapshot implementation (offline default)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from .index import chunk_document
from .models import Chunk, Document, RetrievedChunk
from .index import TfidfIndex


class SearchProvider(Protocol):
    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        """Return search-origin chunks; `coverage` is comparable to the KB's."""
        ...


class SnapshotPage(BaseModel):
    title: str
    url: str
    content: str


class SnapshotSearch:
    """Stands in for a web search API by searching a local JSON snapshot of pages.

    Swap in a real engine by implementing `SearchProvider`; the agent only sees
    `RetrievedChunk`s whose chunks carry `origin="search"` and a URL.
    """

    def __init__(self, pages: list[SnapshotPage]):
        chunks: list[Chunk] = []
        for i, page in enumerate(pages):
            doc = Document(id=f"web{i}", title=page.title, text=page.content, uri=page.url)
            chunks.extend(chunk_document(doc, origin="search"))
        self._index = TfidfIndex(chunks)

    @classmethod
    def from_file(cls, path: Path) -> "SnapshotSearch":
        return cls([SnapshotPage.model_validate(p) for p in json.loads(path.read_text(encoding="utf-8"))])

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        return self._index.search(query, k)
