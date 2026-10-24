import pytest
from langgraph.errors import GraphRecursionError

from rag_citation_agent import RAGAgent
from rag_citation_agent.offline import ExtractiveLLMClient


def nodes(agent, thread):
    return agent.history(thread)


def test_graph_has_explicit_nodes(kb, search):
    graph = RAGAgent(kb, ExtractiveLLMClient(), search).graph.get_graph()
    assert {"retrieve", "answer_from_kb", "weak_retrieval", "search", "answer_from_search", "finalize"} <= set(graph.nodes)


def test_checkpointer_records_the_path_for_a_grounded_answer(kb, search):
    agent = RAGAgent(kb, ExtractiveLLMClient(), search)
    agent.ask("How long are audit logs retained on the Enterprise plan?", thread_id="t1")
    assert nodes(agent, "t1") == ["retrieve", "answer_from_kb", "finalize"]


def test_checkpointer_records_the_fallback_path(kb, search):
    agent = RAGAgent(kb, ExtractiveLLMClient(), search)
    agent.ask("Which regions does Helios Cloud operate data centers in?", thread_id="t2")
    assert nodes(agent, "t2") == ["retrieve", "weak_retrieval", "search", "answer_from_search", "finalize"]


def test_threads_are_isolated(kb, search):
    agent = RAGAgent(kb, ExtractiveLLMClient(), search)
    agent.ask("How long are audit logs retained on the Enterprise plan?", thread_id="a")
    agent.ask("Who won the 2022 football World Cup?", thread_id="b")
    assert "search" not in nodes(agent, "a") and "search" in nodes(agent, "b")


def test_recursion_limit_is_enforced(kb, settings):
    agent = RAGAgent(kb, ExtractiveLLMClient(), None, settings.model_copy(update={"recursion_limit": 1}))
    with pytest.raises(GraphRecursionError):
        agent.ask("How long are audit logs retained on the Enterprise plan?")
