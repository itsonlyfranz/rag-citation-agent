"""Regenerate the agent flow diagram in README.md from the compiled LangGraph graph.

    python scripts/render_graph.py           # rewrite the diagram block in README.md
    python scripts/render_graph.py --check   # exit 1 if README.md's nodes or edges differ from the graph
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from rag_citation_agent import RAGAgent
from rag_citation_agent.index import KnowledgeBase
from rag_citation_agent.offline import ExtractiveLLMClient
from rag_citation_agent.search import SnapshotSearch

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
BLOCK = re.compile(
    r"(<!-- flow-diagram:start -->\n```mermaid\n)(.*?)(```\n<!-- flow-diagram:end -->)", re.DOTALL
)
NODE = re.compile(r"^\s*(\w+)\(", re.M)
EDGE = re.compile(r"^\s*(\w+)\s+(-->|-\.->)\s+(\w+);", re.M)

Shape = tuple[frozenset[str], frozenset[tuple[str, str, bool]]]


def _graph():
    """The compiled graph with every optional edge (a search provider is attached)."""
    return RAGAgent(KnowledgeBase(), ExtractiveLLMClient(), SnapshotSearch([])).graph.get_graph()


def render_flow() -> str:
    """Mermaid source of the compiled graph."""
    return _graph().draw_mermaid()


def graph_shape() -> Shape:
    """Node ids and (source, target, conditional) edges of the compiled graph."""
    graph = _graph()
    return frozenset(graph.nodes), frozenset((e.source, e.target, e.conditional) for e in graph.edges)


def mermaid_shape(source: str) -> Shape:
    """Node ids and (source, target, conditional) edges declared in Mermaid flowchart source."""
    edges = frozenset((a, b, arrow == "-.->") for a, arrow, b in EDGE.findall(source))
    return frozenset(NODE.findall(source)), edges


def readme_flow(text: str) -> str | None:
    """The flow diagram currently embedded in README text, or None if the markers are missing."""
    match = BLOCK.search(text)
    return match.group(2) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if README.md differs from the compiled graph")
    args = parser.parse_args()

    text = README.read_text()
    embedded = readme_flow(text)
    if embedded is None:
        print("README.md has no flow-diagram markers", file=sys.stderr)
        return 1
    if args.check:
        if mermaid_shape(embedded) != graph_shape():
            print("README.md flow diagram is stale; run scripts/render_graph.py", file=sys.stderr)
            return 1
        return 0
    flow = render_flow()
    README.write_text(BLOCK.sub(lambda m: m.group(1) + flow + m.group(3), text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
