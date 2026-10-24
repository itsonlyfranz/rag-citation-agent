import json
from pathlib import Path

import pytest

from rag_citation_agent.config import Settings
from rag_citation_agent.index import KnowledgeBase, load_documents
from rag_citation_agent.search import SnapshotSearch

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    base = KnowledgeBase()
    base.add(load_documents([ROOT / "data" / "knowledge_base"]))
    return base


@pytest.fixture(scope="session")
def search() -> SnapshotSearch:
    return SnapshotSearch.from_file(ROOT / "data" / "web_snapshot" / "results.json")


@pytest.fixture
def settings() -> Settings:
    return Settings()


def reply(*claims: tuple[str, str, str], answerable: bool = True) -> str:
    """Build a model reply: each claim is (text, source_id, quote)."""
    return json.dumps(
        {
            "answerable": answerable,
            "claims": [{"text": t, "citations": [{"source": s, "quote": q}]} for t, s, q in claims],
        }
    )
