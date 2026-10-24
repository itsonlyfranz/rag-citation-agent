"""Plain-text rendering of an Answer for the terminal."""

from __future__ import annotations

from .models import Answer


def render(a: Answer, verbose: bool = False) -> str:
    c = a.confidence
    lines = [
        f"Q: {a.question}",
        f"A: {a.text}",
        f"   status={a.status.value}  confidence={c.overall:.2f} ({c.level})"
        + ("  [LOW CONFIDENCE]" if a.low_confidence else "")
        + ("  [SEARCH FALLBACK]" if a.used_fallback else ""),
    ]
    for cit in a.citations:
        where = cit.uri or cit.chunk_id
        lines.append(f'   [{cit.number}] {cit.title} ({cit.origin}: {where}) "{cit.quote}"')
    for claim in a.removed_claims:
        lines.append(f"   REMOVED unverifiable claim: {claim.text!r} -- {'; '.join(claim.issues)}")
    if a.flags:
        lines.append(f"   flags: {', '.join(a.flags)}")
    if verbose:
        lines += [f"   trace: {t}" for t in a.trace]
    return "\n".join(lines)
