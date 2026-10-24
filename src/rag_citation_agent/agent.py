"""The orchestration: retrieve -> generate -> verify -> score -> (fallback search) -> answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import confidence as conf
from .config import Settings
from .generation import MalformedOutput, generate
from .grounding import verify
from .index import KnowledgeBase
from .llm import LLMClient
from .models import Answer, AnswerStatus, Citation, Claim, Confidence, RetrievedChunk
from .search import SearchProvider

ABSTAIN_TEXT = "I couldn't find a sufficiently supported answer to that in the available sources."


@dataclass
class Attempt:
    """The outcome of answering from one set of context chunks."""

    context: list[RetrievedChunk]
    claims: list[Claim] = field(default_factory=list)
    confidence: Confidence = field(
        default_factory=lambda: Confidence(retrieval=0, verification=0, support=0, overall=0, level="low")
    )
    flags: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    @property
    def verified(self) -> list[Claim]:
        return [c for c in self.claims if c.verified]


class RAGAgent:
    def __init__(
        self,
        kb: KnowledgeBase,
        llm: LLMClient,
        search: SearchProvider | None = None,
        settings: Settings | None = None,
    ):
        self.kb, self.llm, self.search = kb, llm, search
        self.settings = settings or Settings()

    def ask(self, question: str) -> Answer:
        s = self.settings
        trace: list[str] = []

        hits = self.kb.search(question, s.top_k)
        best = hits[0].coverage if hits else 0.0
        trace.append(f"kb: {len(hits)} chunks, best query coverage {best:.2f}")

        if best >= s.min_retrieval:
            attempt = self._attempt(question, hits, trust=1.0)
            trace += attempt.trace
        else:
            attempt = Attempt(context=hits, flags=["weak_retrieval"])
            trace.append(f"kb coverage below {s.min_retrieval}: skipping generation")

        used_fallback = False
        if attempt.confidence.overall < s.min_confidence:
            if self.search is None:
                trace.append("no search provider configured: cannot fall back")
            else:
                found = self.search.search(question, s.top_k)
                trace.append(f"fallback search: {len(found)} results")
                if found:
                    merged = sorted(
                        [*found, *(h for h in hits if h.coverage >= s.min_retrieval)],
                        key=lambda r: (-r.coverage, -r.score),
                    )[: s.top_k]
                    second = self._attempt(question, merged, trust=s.fallback_trust)
                    trace += second.trace
                    if second.confidence.overall > attempt.confidence.overall:
                        attempt, used_fallback = second, True
                        trace.append("fallback answer adopted")
                    else:
                        trace.append("fallback answer was not better; keeping original")

        return self._finalize(question, attempt, used_fallback, trace)

    def _attempt(self, question: str, context: list[RetrievedChunk], trust: float) -> Attempt:
        s = self.settings
        out = Attempt(context=context)
        try:
            raw = generate(self.llm, question, context)
        except MalformedOutput as exc:
            out.flags.append("malformed_model_output")
            out.trace.append(f"generation failed: {exc}")
            return out
        out.claims = verify(raw, context, s.min_support)
        n_bad = len(out.claims) - len(out.verified)
        out.trace.append(
            f"generated {len(out.claims)} claims, {len(out.verified)} verified"
            + ("" if raw.answerable else " (model reported unanswerable)")
        )
        if n_bad:
            out.flags.append("removed_unsupported_claims")
        by_chunk = {r.chunk.id: r.coverage for r in context}
        cited = [by_chunk[c.chunk_id] for cl in out.verified for c in cl.citations]
        retrieval = max(cited, default=0.0)
        out.confidence = conf.score(out.claims, retrieval, trust, s.min_confidence)
        out.trace.append(f"confidence {out.confidence.overall:.2f} ({out.confidence.level})")
        return out

    def _finalize(self, question: str, attempt: Attempt, used_fallback: bool, trace: list[str]) -> Answer:
        s = self.settings
        flags = list(attempt.flags)
        good = attempt.verified
        citations: list[Citation] = []
        numbering: dict[tuple[str, str], int] = {}
        sentences: list[str] = []
        for claim in good:
            marks = []
            for cit in claim.citations:
                key = (cit.chunk_id, cit.quote)
                if key not in numbering:
                    numbering[key] = len(numbering) + 1
                    citations.append(cit.model_copy(update={"number": numbering[key]}))
                marks.append(f"[{numbering[key]}]")
            sentences.append(f"{claim.text}{''.join(marks)}")
            claim.citations = [c.model_copy(update={"number": numbering[(c.chunk_id, c.quote)]}) for c in claim.citations]

        low = attempt.confidence.overall < s.min_confidence
        if low:
            flags.append("low_confidence")
        if used_fallback:
            flags.append("answered_from_search")
        if good:
            text, status = " ".join(sentences), AnswerStatus.ANSWERED
        else:
            text, status = ABSTAIN_TEXT, AnswerStatus.ABSTAINED
        return Answer(
            question=question,
            status=status,
            text=text,
            claims=good,
            removed_claims=[c for c in attempt.claims if not c.verified],
            citations=citations,
            confidence=attempt.confidence,
            low_confidence=low,
            used_fallback=used_fallback,
            flags=flags,
            trace=trace,
        )
