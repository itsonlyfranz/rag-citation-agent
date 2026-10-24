"""Settings loaded from the environment (and an optional gitignored .env)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-sonnet-5-5"


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines; no variable expansion, no dependency on python-dotenv."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("\"'")
    return values


class Settings(BaseModel):
    anthropic_api_key: str | None = None
    model: str = DEFAULT_MODEL
    top_k: int = Field(4, ge=1)
    # Below this query coverage the knowledge base is treated as not knowing the topic.
    min_retrieval: float = Field(0.35, ge=0, le=1)
    # Overall confidence below this flags the answer and triggers the search fallback.
    min_confidence: float = Field(0.6, ge=0, le=1)
    # Minimum share of a claim's content terms that must appear in its cited text.
    min_support: float = Field(0.6, ge=0, le=1)
    # Multiplier applied to answers built from search results instead of the curated KB.
    fallback_trust: float = Field(0.9, ge=0, le=1)
    index_path: Path = Path(".rag/index.json")
    search_snapshot: Path = Path("data/web_snapshot/results.json")

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, dotenv: Path | None = Path(".env")
    ) -> "Settings":
        merged: dict[str, str] = dict(load_dotenv(dotenv)) if dotenv else {}
        merged.update(os.environ if env is None else env)
        fields = {
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "model": "RAG_MODEL",
            "top_k": "RAG_TOP_K",
            "min_retrieval": "RAG_MIN_RETRIEVAL",
            "min_confidence": "RAG_MIN_CONFIDENCE",
            "min_support": "RAG_MIN_SUPPORT",
            "fallback_trust": "RAG_FALLBACK_TRUST",
            "index_path": "RAG_INDEX_PATH",
            "search_snapshot": "RAG_SEARCH_SNAPSHOT",
        }
        kwargs = {f: merged[e] for f, e in fields.items() if merged.get(e)}
        return cls(**kwargs)
