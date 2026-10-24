"""Scripted walkthrough of the four headline behaviours."""

from __future__ import annotations

import json
from pathlib import Path

from .agent import RAGAgent
from .config import Settings
from .index import KnowledgeBase, load_documents
from .llm import LLMClient
from .offline import ExtractiveLLMClient
from .render import render
from .search import SnapshotSearch


class HallucinatingLLMClient:
    """Simulates a misbehaving model: a faithful answer plus one invented, fake-cited claim."""

    def __init__(self) -> None:
        self._honest = ExtractiveLLMClient(max_claims=1)

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        data = json.loads(self._honest.complete(system, user, max_tokens))
        data["answerable"] = True
        data["claims"].append(
            {
                "text": "Enterprise customers also get 24/7 phone support with a 5 minute response time.",
                "citations": [{"source": "S1", "quote": "24/7 phone support with a 5 minute response time"}],
            }
        )
        return json.dumps(data)


def run_demo(root: Path, llm: LLMClient, settings: Settings) -> list[str]:
    kb = KnowledgeBase()
    kb.add(load_documents([root / "data" / "knowledge_base"]))
    search = SnapshotSearch.from_file(root / "data" / "web_snapshot" / "results.json")

    scenarios: list[tuple[str, str, LLMClient]] = [
        ("1. Grounded answer with verified citations", "How long are audit logs retained on the Enterprise plan?", llm),
        ("2. Topic missing from the KB -> search fallback", "Which regions does Helios Cloud operate data centers in?", llm),
        ("3. Model invents a claim -> removed, answer flagged", "What support does Helios Cloud give Enterprise customers?", HallucinatingLLMClient()),
        ("4. Unanswerable -> abstain instead of guessing", "Who won the 2022 football World Cup?", llm),
    ]
    out = []
    for title, question, client in scenarios:
        answer = RAGAgent(kb, client, search, settings).ask(question)
        out.append(f"=== {title}\n{render(answer, verbose=True)}\n")
    return out
