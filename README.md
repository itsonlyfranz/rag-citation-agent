# RAG Agent with Citation Grounding

A RAG agent that verifies every claim against a verbatim source quote, flags low-confidence answers, and falls back to search.

**Technologies:** Python 3.11+, LangGraph, langchain-anthropic (Claude `claude-sonnet-5-5`), Pydantic v2, pytest, local TF-IDF retrieval
**Category:** AI agents / RAG · **Shows:** you can prevent hallucinations at scale

## Highlights

- **LangGraph agent:** the flow is a `StateGraph` with typed Pydantic state, six explicit nodes, and conditional edges for the retrieval gate, the confidence check, and the search fallback. An opt-in checkpointer (e.g. `RAGAgent(..., checkpointer=MemorySaver())`) records each question's path, queryable via `agent.history(thread_id)`, and a recursion limit bounds every run.
- **Claim-level citation verification:** each generated claim must carry a verbatim quote from a retrieved source; claims with unknown sources, fabricated quotes, unsupported terms, or numbers absent from the quote are removed before the user sees them.
- **Confidence from verifiable signals:** `trust x verification x (retrieval + support) / 2`, ignoring the model's self-reported confidence. In the demo, one invented claim out of two drops an answer from high confidence to 0.42 and flags it.
- **Search fallback with abstention:** weak KB coverage skips the LLM call entirely and routes to a `SearchProvider`; if nothing verifies, the agent abstains rather than guesses.
- **Fully offline and swappable:** the LLM is behind a one-method protocol and retrieval/search are local, so all 56 tests and the 4-scenario demo run with no network or API key.
- **Prompt-injection mitigation:** retrieved text is escaped inside `<source>` tags and declared untrusted (covered by a test).

Most RAG demos stop at "stuff chunks into a prompt and cite whatever the model says." This project
treats the model's output as **untrusted** and verifies every claim against the retrieved text before
it reaches the user. A claim that cannot be traced to a verbatim quote in a real source is removed,
and the answer is scored, flagged, and (if weak) retried against search results.

## What it demonstrates

| Capability | How it works | Proven by |
|---|---|---|
| **Retrieve context** | Local TF-IDF inverted index over chunked documents, persisted as JSON. No services. | `tests/test_retrieval.py` |
| **Answers with sources** | Model returns claims, each with a source id and a *verbatim quote*. Final text carries numbered `[n]` citations pointing at title, URI and quote. | `test_grounded_answer_carries_numbered_verified_citations`, `test_citation_numbers_are_shared_…` |
| **Prevent hallucinations** | Every claim is checked: source exists, quote is literally in the source, claim terms are covered by the source, every number appears in the quote. Failures are dropped, never shown. | `tests/test_grounding.py`, `test_fabricated_claim_is_removed_…`, `test_answer_with_only_fabricated_claims_abstains` |
| **Flag low confidence** | Confidence is computed from verifiable signals (below) and compared to a configurable threshold. | `tests/test_confidence.py`, `test_low_confidence_is_flagged_…`, `test_threshold_is_configurable` |
| **Fallback to search** | Weak retrieval or low confidence triggers a `SearchProvider`; the better-scoring answer wins and is labelled as search-sourced. | `test_missing_kb_topic_skips_generation_…`, `test_low_confidence_kb_answer_triggers_fallback_…`, `test_fallback_that_is_no_better_…` |
| **Graph flow and checkpoints** | Explicit nodes/edges; opt-in per-thread checkpoint history; recursion limit. | `tests/test_graph.py` |
| **Abstain instead of guessing** | No verified claims -> an explicit "couldn't find a supported answer", never a made-up one. | `test_unanswerable_question_abstains_even_with_search` |

## Architecture

The agent is a LangGraph `StateGraph` (`agent.py`). Nodes: `retrieve`, `answer_from_kb`, `weak_retrieval`,
`search`, `answer_from_search`, `finalize`. Conditional edges choose the path after retrieval (coverage gate),
after scoring (confidence threshold / search provider available), and after search (any results?). Equivalent
flow, drawn as a pipeline:

```
question
   |
   v
KnowledgeBase.search ---- best query coverage < min_retrieval? ----+
   | (TF-IDF, top-k chunks)                                        |
   v                                                               |
generate  (LLMClient -> JSON claims + quotes; 1 retry on bad JSON) |
   |                                                               |
   v                                                               |
verify    (source exists . quote verbatim . term support . numbers)|
   |                                                               |
   v                                                               |
confidence = verification x (retrieval + support) / 2              |
   |                                                               |
   +-- >= min_confidence --------------------------> Answer        |
   |                                                    ^          |
   +-- < min_confidence <---------------------------------+--------+
          |
          v
   SearchProvider.search -> merge with KB hits -> generate -> verify -> confidence
          |
          v
   keep whichever attempt scored higher (search answers carry a trust discount)
          |
          v
   Answer: text with [n] citations, removed claims, confidence, flags, trace
```

Components (in `src/rag_citation_agent/`), each with one responsibility:

- `models.py` Pydantic v2 models (`Chunk`, `Claim`, `Citation`, `Confidence`, `Answer`).
- `index.py` chunking, TF-IDF inverted index, JSON-persisted `KnowledgeBase`.
- `generation.py` prompt (sources are escaped and declared untrusted) and strict JSON parsing.
- `grounding.py` the claim verifier. `confidence.py` the scoring formula.
- `search.py` `SearchProvider` protocol + local `SnapshotSearch`.
- `agent.py` the LangGraph `StateGraph`: typed `AgentState`, nodes, conditional edges, optional checkpointer. `llm.py` the `LLMClient` protocol, `AnthropicClient` (wraps `ChatAnthropic`), `ScriptedLLMClient`.
- `offline.py` a deterministic extractive client so the CLI/demo work with no key.

### Confidence

```
overall = trust x verification x (retrieval + support) / 2
```

- **verification**: fraction of generated claims that passed citation checks. Multiplying (not averaging) means
  one invented claim out of two pushes an answer under the default 0.6 threshold.
- **retrieval**: IDF-weighted query coverage of the best *cited* chunk. Unknown query terms count against it.
- **support**: mean term overlap between verified claims and their cited sources.
- **trust**: 1.0 for the curated KB, `fallback_trust` (default 0.9) for search results.

The model's self-reported confidence is deliberately not used: it is exactly the signal that fails when it hallucinates.

## Quickstart

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/) (or plain `pip install -e '.[dev]'`):

```bash
uv sync --extra dev          # or: python -m venv .venv && .venv/bin/pip install -e '.[dev]'
uv run pytest                # fully offline: no network, no API key
```

To use Claude, copy `.env.example` to `.env` (gitignored) and set `ANTHROPIC_API_KEY`.
The default model is `claude-sonnet-5-5`; override with `RAG_MODEL`.

## Demo

`rag-agent demo` runs four scenarios on the bundled sample corpus (a fictional product, "Helios Cloud",
so the model cannot answer from memory). `--offline` swaps in the built-in extractive model so no key is needed:

```bash
uv run rag-agent demo --offline
```

1. **Grounded answer**: audit-log retention, with verified `[n]` citations and high confidence.
2. **Search fallback**: data-center regions are not in the KB; the agent skips generation, searches, and labels the answer `[SEARCH FALLBACK]`.
3. **Hallucination caught**: a *simulated misbehaving model* adds an invented "24/7 phone support" claim with a fabricated quote. It is removed, shown as `REMOVED unverifiable claim`, and the answer is flagged `[LOW CONFIDENCE]`.
4. **Abstain**: an off-topic question gets an explicit refusal instead of a guess.

Use it on your own documents:

```bash
uv run rag-agent ingest path/to/docs          # .md/.txt, recursive -> .rag/index.json
uv run rag-agent ask "your question" -v       # add --offline for no key, --json for the full Answer
```

`-v` prints the pipeline trace (coverage, claims generated/verified, confidence, fallback decisions).

## Design decisions and trade-offs

- **Verify, don't trust.** Prompting for citations is not enough; models fabricate plausible quotes. Requiring a
  *verbatim* quote makes verification a cheap deterministic string check that scales linearly with answer size and needs
  no second LLM call.
- **Lexical support is a proxy, not entailment.** Term overlap plus a number check catches invented facts, but not a
  subtle negation flipped inside a real quote. An NLI model or LLM judge could be added behind `verify()`; I kept the
  default deterministic, free, and testable.
- **Gate before generating.** If the KB has no coverage of the question, the agent makes no LLM call at all. At scale this
  is both the cheapest and the safest path for out-of-domain traffic.
- **TF-IDF over embeddings.** Zero dependencies, offline, deterministic tests, and the IDF-weighted coverage score doubles as
  an interpretable "does the KB know this topic" signal. The cost is weaker semantic matching (synonyms). `KnowledgeBase`
  is the seam to swap in dense or hybrid retrieval.
- **LangGraph where it fits.** The branching (gate, confidence check, optional fallback) is exactly what conditional edges
  express, and an optional checkpointer gives a replayable per-question path. I did not add interrupts or parallel branches:
  there is no human-in-the-loop step, and the fallback deliberately runs only after the KB attempt fails (parallel
  search would spend a search and an LLM call on every question). Checkpointing is off by default so a long-lived agent
  holds no per-question state; pass a saver (`MemorySaver` for tests, a persistent one in production) to enable `history()`.
- **Search is a protocol.** `SnapshotSearch` searches a local JSON file so everything runs offline; production would
  implement `SearchProvider` against a real search API. Search answers get a trust discount and a visible flag because web
  content is less vetted than a curated KB.
- **Untrusted context.** Retrieved text is HTML-escaped inside `<source>` tags and the system prompt says to treat it as data,
  reducing prompt-injection from documents (tested). It is mitigation, not a guarantee.
- **Fail closed.** Malformed JSON gets one retry, then abstains. Unverifiable claims are removed rather than annotated.
- **Not covered:** multi-hop questions, conversation history, re-ranking, and large-scale index sharding. The inverted index
  only touches chunks sharing a query term, but the JSON store is loaded fully in memory.

## Tests

`uv run pytest` runs 56 offline tests. The LLM sits behind the one-method `LLMClient` protocol; tests use
`ScriptedLLMClient` (canned replies, records prompts) and a fake chat model for `AnthropicClient`, so no network or API key
is ever needed. The table above maps each capability to the tests that prove it.
