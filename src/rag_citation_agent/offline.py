"""A deterministic, no-network LLMClient that answers by extracting sentences from the sources.

It lets the CLI and demo run without an API key. It is intentionally a faithful model: it only
ever quotes what is in the context, so grounded answers verify and unanswerable questions abstain.
"""

from __future__ import annotations

import json
import re
from html import unescape

from .text import content_tokens, split_sentences

_SOURCE = re.compile(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
_QUESTION = re.compile(r"Question: (.*?)\n\nSources:", re.DOTALL)


class ExtractiveLLMClient:
    def __init__(self, min_overlap: float = 0.5, max_claims: int = 2):
        self.min_overlap = min_overlap
        self.max_claims = max_claims

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        question = _QUESTION.search(user)
        q_terms = set(content_tokens(question.group(1))) if question else set()
        scored: list[tuple[float, str, str]] = []
        for sid, body in _SOURCE.findall(user):
            for sentence in split_sentences(unescape(body)):
                if q_terms:
                    overlap = len(q_terms & set(content_tokens(sentence))) / len(q_terms)
                    scored.append((overlap, sid, sentence))
        scored.sort(key=lambda t: -t[0])
        picked = [t for t in scored if t[0] >= self.min_overlap][: self.max_claims]
        claims = [{"text": s, "citations": [{"source": sid, "quote": s}]} for _, sid, s in picked]
        return json.dumps({"answerable": bool(claims), "claims": claims})
