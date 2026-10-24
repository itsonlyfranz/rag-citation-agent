"""The agent as a LangGraph StateGraph: retrieve -> generate -> verify/score -> (search fallback) -> answer.

    START -> retrieve --(coverage ok)--> answer_from_kb --+
                 |                                        |
                 +--(weak)--> weak_retrieval -------------+
                                                          v
                              confident? --yes--> finalize -> END
                                  |no
                         search provider? --none--> finalize
                                  |
                               search --(no results)--> finalize
                                  |
                            answer_from_search -> finalize
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import confidence as conf
from .config import Settings
from .generation import MalformedOutput, generate
from .grounding import verify
from .index import KnowledgeBase
from .llm import LLMClient
from .models import Answer, AnswerStatus, Citation, Claim, Confidence, RetrievedChunk
from .search import SearchProvider

ABSTAIN_TEXT = "I couldn't find a sufficiently supported answer to that in the available sources."


class Attempt(BaseModel):
    """The outcome of answering from one set of context chunks."""

    context: list[RetrievedChunk] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    confidence: Confidence = Confidence(retrieval=0, verification=0, support=0, overall=0, level="low")
    flags: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)

    @property
    def verified(self) -> list[Claim]:
        return [c for c in self.claims if c.verified]


class AgentState(BaseModel):
    """Typed graph state. `trace` accumulates across nodes; every other field is last-write-wins."""

    question: str
    hits: list[RetrievedChunk] = Field(default_factory=list)
    attempt: Attempt = Field(default_factory=Attempt)
    found: list[RetrievedChunk] = Field(default_factory=list)
    used_fallback: bool = False
    trace: Annotated[list[str], operator.add] = Field(default_factory=list)
    answer: Answer | None = None


class RAGAgent:
    def __init__(
        self,
        kb: KnowledgeBase,
        llm: LLMClient,
        search: SearchProvider | None = None,
        settings: Settings | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.kb, self.llm, self.search = kb, llm, search
        self.settings = settings or Settings()
        # Checkpointing is opt-in: with a saver, every node's state is kept per thread for replay/inspection.
        self.checkpointer = checkpointer
        self.graph = self._build()

    # --- graph wiring --------------------------------------------------------------------------

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("answer_from_kb", self._answer_from_kb)
        g.add_node("weak_retrieval", self._weak_retrieval)
        g.add_node("search", self._search)
        g.add_node("answer_from_search", self._answer_from_search)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "retrieve")
        g.add_conditional_edges("retrieve", self._route_retrieval, ["answer_from_kb", "weak_retrieval"])
        g.add_conditional_edges("answer_from_kb", self._route_confidence, ["search", "finalize"])
        g.add_conditional_edges("weak_retrieval", self._route_confidence, ["search", "finalize"])
        g.add_conditional_edges("search", self._route_search, ["answer_from_search", "finalize"])
        g.add_edge("answer_from_search", "finalize")
        g.add_edge("finalize", END)
        return g.compile(checkpointer=self.checkpointer)

    def ask(self, question: str, thread_id: str | None = None) -> Answer:
        config = {
            "configurable": {"thread_id": thread_id or uuid.uuid4().hex},
            "recursion_limit": self.settings.recursion_limit,
        }
        result = self.graph.invoke(AgentState(question=question), config)
        return result["answer"]

    def history(self, thread_id: str) -> list[str]:
        """Nodes executed on a thread, oldest first, read back from the checkpointer."""
        if self.checkpointer is None:
            raise RuntimeError("history() needs a checkpointer: construct RAGAgent(..., checkpointer=MemorySaver())")
        config: Any = {"configurable": {"thread_id": thread_id}}
        snapshots = reversed(list(self.graph.get_state_history(config)))
        return [n for snap in snapshots for n in snap.next if n != "__start__"]

    # --- routing --------------------------------------------------------------------------------

    def _route_retrieval(self, state: AgentState) -> Literal["answer_from_kb", "weak_retrieval"]:
        best = state.hits[0].coverage if state.hits else 0.0
        return "answer_from_kb" if best >= self.settings.min_retrieval else "weak_retrieval"

    def _route_confidence(self, state: AgentState) -> Literal["search", "finalize"]:
        if state.attempt.confidence.overall >= self.settings.min_confidence:
            return "finalize"
        return "search" if self.search is not None else "finalize"

    def _route_search(self, state: AgentState) -> Literal["answer_from_search", "finalize"]:
        return "answer_from_search" if state.found else "finalize"

    # --- nodes ----------------------------------------------------------------------------------

    def _retrieve(self, state: AgentState) -> dict:
        hits = self.kb.search(state.question, self.settings.top_k)
        best = hits[0].coverage if hits else 0.0
        return {"hits": hits, "trace": [f"kb: {len(hits)} chunks, best query coverage {best:.2f}"]}

    def _weak_retrieval(self, state: AgentState) -> dict:
        # Out-of-domain question: spend no LLM call on it.
        return {
            "attempt": Attempt(context=state.hits, flags=["weak_retrieval"]),
            "trace": [f"kb coverage below {self.settings.min_retrieval}: skipping generation"],
        }

    def _answer_from_kb(self, state: AgentState) -> dict:
        attempt = self._attempt(state.question, state.hits, trust=1.0)
        return {"attempt": attempt, "trace": attempt.trace}

    def _search(self, state: AgentState) -> dict:
        assert self.search is not None
        s = self.settings
        found = self.search.search(state.question, s.top_k)
        merged = sorted(
            [*found, *(h for h in state.hits if h.coverage >= s.min_retrieval)],
            key=lambda r: (-r.coverage, -r.score),
        )[: s.top_k]
        return {"found": merged if found else [], "trace": [f"fallback search: {len(found)} results"]}

    def _answer_from_search(self, state: AgentState) -> dict:
        second = self._attempt(state.question, state.found, trust=self.settings.fallback_trust)
        trace = list(second.trace)
        if second.confidence.overall > state.attempt.confidence.overall:
            trace.append("fallback answer adopted")
            return {"attempt": second, "used_fallback": True, "trace": trace}
        trace.append("fallback answer was not better; keeping original")
        return {"trace": trace}

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
        out.confidence = conf.score(out.claims, max(cited, default=0.0), trust, s.min_confidence)
        out.trace.append(f"confidence {out.confidence.overall:.2f} ({out.confidence.level})")
        return out

    def _finalize(self, state: AgentState) -> dict:
        attempt, trace = state.attempt, list(state.trace)
        if attempt.confidence.overall < self.settings.min_confidence and self.search is None:
            trace.append("no search provider configured: cannot fall back")
        flags = list(attempt.flags)
        citations: list[Citation] = []
        numbering: dict[tuple[str, str], int] = {}
        sentences: list[str] = []
        good: list[Claim] = []
        for claim in attempt.verified:
            marks, numbered = [], []
            for cit in claim.citations:
                key = (cit.chunk_id, cit.quote)
                if key not in numbering:
                    numbering[key] = len(numbering) + 1
                    citations.append(cit.model_copy(update={"number": numbering[key]}))
                marks.append(f"[{numbering[key]}]")
                numbered.append(cit.model_copy(update={"number": numbering[key]}))
            sentences.append(f"{claim.text}{''.join(marks)}")
            good.append(claim.model_copy(update={"citations": numbered}))

        low = attempt.confidence.overall < self.settings.min_confidence
        if low:
            flags.append("low_confidence")
        if state.used_fallback:
            flags.append("answered_from_search")
        answered = bool(good)
        answer = Answer(
            question=state.question,
            status=AnswerStatus.ANSWERED if answered else AnswerStatus.ABSTAINED,
            text=" ".join(sentences) if answered else ABSTAIN_TEXT,
            claims=good,
            removed_claims=[c for c in attempt.claims if not c.verified],
            citations=citations,
            confidence=attempt.confidence,
            low_confidence=low,
            used_fallback=state.used_fallback,
            flags=flags,
            trace=trace,
        )
        return {"answer": answer}
