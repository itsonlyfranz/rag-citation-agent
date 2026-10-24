"""Citation verification: the layer that makes hallucinated claims impossible to ship silently.

A claim survives only if (1) it cites a source that was actually provided, (2) the quote is a
verbatim passage of that source, (3) the claim's content terms are covered by the cited source text,
and (4) every number in the claim appears in the quoted passage itself.
"""

from __future__ import annotations

from .generation import RawAnswer, RawClaim
from .models import Citation, Claim, RetrievedChunk
from .text import content_tokens, normalize, numbers


def support_score(claim: str, evidence: str) -> float:
    """Share of the claim's content terms found in the evidence (lexical entailment proxy)."""
    terms = set(content_tokens(claim))
    if not terms:
        return 0.0
    return len(terms & set(content_tokens(evidence))) / len(terms)


def verify_claim(raw: RawClaim, context: dict[str, RetrievedChunk], min_support: float) -> Claim:
    issues: list[str] = []
    verified: list[tuple[str, RetrievedChunk, str]] = []
    for cit in raw.citations:
        hit = context.get(cit.source)
        if hit is None:
            issues.append(f"unknown source {cit.source!r}")
        elif not content_tokens(cit.quote) or f" {normalize(cit.quote)} " not in f" {normalize(hit.chunk.text)} ":
            issues.append(f"quote for {cit.source} is not in the source")
        else:
            verified.append((cit.source, hit, cit.quote.strip()))
    if not raw.citations:
        issues.append("no citations")

    quoted = " ".join(q for _, _, q in verified)
    source_text = " ".join(hit.chunk.text for _, hit, _ in verified)
    support = support_score(raw.text, source_text) if verified else 0.0
    if verified and support < min_support:
        issues.append(f"claim not supported by its sources (support {support:.2f})")
    unsupported_numbers = numbers(raw.text) - numbers(quoted)
    if verified and unsupported_numbers:
        issues.append(f"numbers not in quotes: {sorted(unsupported_numbers)}")

    ok = bool(verified) and not any(i.startswith(("claim not", "numbers")) for i in issues)
    citations = [
        Citation(
            number=0, source_id=sid, chunk_id=hit.chunk.id, title=hit.chunk.title,
            uri=hit.chunk.uri, origin=hit.chunk.origin, quote=quote,
        )
        for sid, hit, quote in verified
    ]
    return Claim(text=raw.text.strip(), citations=citations if ok else [], support=round(support, 3), verified=ok, issues=issues)


def verify(raw: RawAnswer, context: list[RetrievedChunk], min_support: float) -> list[Claim]:
    by_id = {f"S{i}": r for i, r in enumerate(context, 1)}
    return [verify_claim(c, by_id, min_support) for c in raw.claims if c.text.strip()]
