"""Regenerate the agent flow diagram in README.md from the compiled LangGraph graph.

    python scripts/render_graph.py           # rewrite the diagram block in README.md
    python scripts/render_graph.py --check   # exit 1 if README.md is out of date
    python scripts/render_graph.py --print   # print the Mermaid source only
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


def render_flow() -> str:
    """Mermaid source of the compiled graph, with every optional edge (a search provider is attached)."""
    agent = RAGAgent(KnowledgeBase(), ExtractiveLLMClient(), SnapshotSearch([]))
    return agent.graph.get_graph().draw_mermaid()


def readme_flow(text: str) -> str | None:
    """The flow diagram currently embedded in README text, or None if the markers are missing."""
    match = BLOCK.search(text)
    return match.group(2) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if README.md differs from the compiled graph")
    parser.add_argument("--print", action="store_true", help="print the Mermaid source instead of editing README.md")
    args = parser.parse_args()

    flow = render_flow()
    if args.print:
        print(flow, end="")
        return 0
    text = README.read_text()
    if readme_flow(text) is None:
        print("README.md has no flow-diagram markers", file=sys.stderr)
        return 1
    if args.check:
        if readme_flow(text) != flow:
            print("README.md flow diagram is stale; run scripts/render_graph.py", file=sys.stderr)
            return 1
        return 0
    README.write_text(BLOCK.sub(lambda m: m.group(1) + flow + m.group(3), text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
