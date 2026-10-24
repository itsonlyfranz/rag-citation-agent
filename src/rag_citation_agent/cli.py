"""Command line: `rag-agent ingest | ask | demo`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import RAGAgent
from .config import Settings
from .demo import run_demo
from .index import KnowledgeBase, load_documents
from .llm import AnthropicClient, LLMClient
from .offline import ExtractiveLLMClient
from .render import render
from .search import SnapshotSearch


def _llm(settings: Settings, offline: bool) -> LLMClient:
    if offline:
        return ExtractiveLLMClient()
    return AnthropicClient(settings.anthropic_api_key, settings.model)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-agent", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="chunk and index documents into the local store")
    ing.add_argument("paths", nargs="+", type=Path)

    ask = sub.add_parser("ask", help="ask a question against the indexed documents")
    ask.add_argument("question")
    ask.add_argument("--offline", action="store_true", help="use the built-in extractive model (no API key)")
    ask.add_argument("--no-search", action="store_true", help="disable the search fallback")
    ask.add_argument("--json", action="store_true", help="print the full Answer as JSON")
    ask.add_argument("-v", "--verbose", action="store_true", help="show the pipeline trace")

    demo = sub.add_parser("demo", help="run the four headline scenarios on the sample corpus")
    demo.add_argument("--offline", action="store_true", help="use the built-in extractive model (no API key)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    if args.command == "ingest":
        kb = KnowledgeBase.load(settings.index_path) if settings.index_path.is_file() else KnowledgeBase()
        added = kb.add(load_documents(args.paths))
        kb.save(settings.index_path)
        print(f"indexed {added} chunks ({len(kb)} total) -> {settings.index_path}")
        return 0

    try:
        llm = _llm(settings, args.offline)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "demo":
        print("\n".join(run_demo(Path.cwd(), llm, settings)))
        return 0

    if not settings.index_path.is_file():
        print(f"error: no index at {settings.index_path}; run `rag-agent ingest <paths>` first", file=sys.stderr)
        return 2
    search = None
    if not args.no_search and settings.search_snapshot.is_file():
        search = SnapshotSearch.from_file(settings.search_snapshot)
    agent = RAGAgent(KnowledgeBase.load(settings.index_path), llm, search, settings)
    answer = agent.ask(args.question)
    print(answer.model_dump_json(indent=2) if args.json else render(answer, args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
