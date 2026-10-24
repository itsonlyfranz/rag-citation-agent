import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from conftest import reply
from rag_citation_agent import RAGAgent
from rag_citation_agent.llm import ScriptedLLMClient
from rag_citation_agent.models import Chunk, RetrievedChunk
from rag_citation_agent.offline import ExtractiveLLMClient


def nodes(agent, thread):
    return agent.history(thread)


def test_graph_has_explicit_nodes(kb, search):
    graph = RAGAgent(kb, ExtractiveLLMClient(), search).graph.get_graph()
    assert {"retrieve", "answer_from_kb", "weak_retrieval", "search", "answer_from_search", "finalize"} <= set(graph.nodes)


def test_checkpointer_records_the_path_for_a_grounded_answer(kb, search):
    agent = RAGAgent(kb, ExtractiveLLMClient(), search, checkpointer=MemorySaver())
    agent.ask("How long are audit logs retained on the Enterprise plan?", thread_id="t1")
    assert nodes(agent, "t1") == ["retrieve", "answer_from_kb", "finalize"]


def test_checkpointer_records_the_fallback_path(kb, search):
    agent = RAGAgent(kb, ExtractiveLLMClient(), search, checkpointer=MemorySaver())
    agent.ask("Which regions does Helios Cloud operate data centers in?", thread_id="t2")
    assert nodes(agent, "t2") == ["retrieve", "weak_retrieval", "search", "answer_from_search", "finalize"]


def test_threads_are_isolated(kb, search):
    agent = RAGAgent(kb, ExtractiveLLMClient(), search, checkpointer=MemorySaver())
    agent.ask("How long are audit logs retained on the Enterprise plan?", thread_id="a")
    agent.ask("Who won the 2022 football World Cup?", thread_id="b")
    assert "search" not in nodes(agent, "a") and "search" in nodes(agent, "b")


def test_history_requires_a_checkpointer(kb, search):
    agent = RAGAgent(kb, ExtractiveLLMClient(), search)
    agent.ask("How long are audit logs retained on the Enterprise plan?", thread_id="t1")
    with pytest.raises(RuntimeError, match="checkpointer"):
        agent.history("t1")


def test_recursion_limit_is_enforced(kb, settings):
    agent = RAGAgent(kb, ExtractiveLLMClient(), None, settings.model_copy(update={"recursion_limit": 1}))
    with pytest.raises(GraphRecursionError):
        agent.ask("How long are audit logs retained on the Enterprise plan?")


def test_reused_thread_answer_reflects_only_the_current_question(kb, search):
    fresh = RAGAgent(kb, ExtractiveLLMClient(), search).ask("How long are audit logs retained on the Enterprise plan?")
    agent = RAGAgent(kb, ExtractiveLLMClient(), search, checkpointer=MemorySaver())
    agent.ask("Which regions does Helios Cloud operate data centers in?", thread_id="s")
    again = agent.ask("How long are audit logs retained on the Enterprise plan?", thread_id="s")
    assert again.trace == fresh.trace
    assert again.used_fallback is fresh.used_fallback is False and again.flags == fresh.flags


class _StubKB:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=4):
        return self.hits


def test_retrieval_gate_uses_the_best_coverage_not_the_top_score():
    text = "Audit logs are retained for 400 days on the Enterprise plan."
    hits = [
        RetrievedChunk(chunk=Chunk(id="x#0", doc_id="x", title="X", text="Enterprise."), score=0.9, coverage=0.1),
        RetrievedChunk(chunk=Chunk(id="y#0", doc_id="y", title="Y", text=text), score=0.4, coverage=0.9),
    ]
    llm = ScriptedLLMClient([reply(("Enterprise audit logs are retained for 400 days.", "S2", "for 400 days on the Enterprise plan"))])
    a = RAGAgent(_StubKB(hits), llm).ask("How long are audit logs retained on the Enterprise plan?")
    assert len(llm.calls) == 1 and a.citations and a.citations[0].chunk_id == "y#0"
