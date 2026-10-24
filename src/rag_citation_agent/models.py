"""Pydantic data models shared across the pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    title: str
    text: str
    uri: str | None = None


class Chunk(BaseModel):
    id: str
    doc_id: str
    title: str
    text: str
    origin: Literal["kb", "search"] = "kb"
    uri: str | None = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    # Cosine similarity, used for ranking.
    score: float
    # IDF-weighted share of the query's terms found in the chunk (0..1). Terms the
    # corpus has never seen count against it, so off-topic questions score ~0.
    coverage: float


class Citation(BaseModel):
    number: int
    source_id: str
    chunk_id: str
    title: str
    uri: str | None = None
    origin: Literal["kb", "search"] = "kb"
    quote: str


class Claim(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    support: float = 0.0
    verified: bool = False
    issues: list[str] = Field(default_factory=list)


class Confidence(BaseModel):
    retrieval: float
    verification: float
    support: float
    overall: float
    level: Literal["high", "medium", "low"]


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"


class Answer(BaseModel):
    question: str
    status: AnswerStatus
    text: str
    claims: list[Claim] = Field(default_factory=list)
    removed_claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence
    low_confidence: bool
    used_fallback: bool = False
    flags: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
