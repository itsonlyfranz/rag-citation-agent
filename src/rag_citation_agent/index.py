"""Local TF-IDF inverted index and its JSON-persisted store. No external services."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .models import Chunk, Document, RetrievedChunk
from .text import content_tokens


def chunk_document(doc: Document, max_chars: int = 600, origin: str = "kb") -> list[Chunk]:
    """Pack whole paragraphs into chunks of at most ~max_chars (long paragraphs split by sentence)."""
    from .text import split_sentences

    units: list[str] = []
    for para in re.split(r"\n\s*\n", doc.text):
        para = " ".join(para.split())
        if not para or para.startswith("#"):
            continue
        units.extend(split_sentences(para) if len(para) > max_chars else [para])

    chunks: list[Chunk] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current} {unit}".strip()
    if current:
        chunks.append(current)
    return [
        Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, title=doc.title, text=text, origin=origin, uri=doc.uri)  # type: ignore[arg-type]
        for i, text in enumerate(chunks)
    ]


class TfidfIndex:
    """Sparse TF-IDF vectors over an inverted index: queries only touch chunks sharing a term."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        n = len(chunks)
        term_counts = [Counter(content_tokens(f"{c.title} {c.text}")) for c in chunks]
        df: Counter[str] = Counter(t for counts in term_counts for t in counts)
        self._unseen_idf = math.log(1 + n) + 1
        self._idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self._terms = [set(counts) for counts in term_counts]
        self._postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, counts in enumerate(term_counts):
            weights = {t: (1 + math.log(tf)) * self._idf[t] for t, tf in counts.items()}
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            for t, w in weights.items():
                self._postings[t].append((i, w / norm))

    def idf(self, term: str) -> float:
        return self._idf.get(term, self._unseen_idf)

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        q_terms = set(content_tokens(query))
        if not q_terms:
            return []
        q_weights = {t: self.idf(t) for t in q_terms}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for t, qw in q_weights.items():
            for i, dw in self._postings.get(t, ()):
                scores[i] += (qw / q_norm) * dw
        total = sum(q_weights.values())
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            RetrievedChunk(
                chunk=self.chunks[i],
                score=round(score, 4),
                coverage=round(sum(q_weights[t] for t in q_terms & self._terms[i]) / total, 4),
            )
            for i, score in ranked
        ]


class KnowledgeBase:
    """Documents -> chunks -> index, persisted as a single JSON file."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks: list[Chunk] = chunks or []
        self._index = TfidfIndex(self.chunks)

    def add(self, docs: list[Document], max_chars: int = 600) -> int:
        replaced = {d.id for d in docs}
        kept = [c for c in self.chunks if c.doc_id not in replaced]
        new = [c for d in docs for c in chunk_document(d, max_chars)]
        self.chunks = kept + new
        self._index = TfidfIndex(self.chunks)
        return len(new)

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        return self._index.search(query, k)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "[" + ",\n".join(c.model_dump_json() for c in self.chunks) + "]", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        import json

        return cls([Chunk.model_validate(c) for c in json.loads(path.read_text(encoding="utf-8"))])

    def __len__(self) -> int:
        return len(self.chunks)


def load_documents(paths: list[Path]) -> list[Document]:
    """Read .md/.txt files (recursing into directories); the title is the first '# ' heading."""
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(q for q in p.rglob("*") if q.suffix in {".md", ".txt"}) if p.is_dir() else [p])
    docs = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        heading = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), None)
        docs.append(Document(id=f.stem, title=heading or f.stem.replace("-", " ").title(), text=text, uri=f.name))
    return docs
