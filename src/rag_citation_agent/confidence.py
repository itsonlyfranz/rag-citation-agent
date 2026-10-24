"""Confidence scoring from verifiable signals only (the model's self-reported confidence is ignored)."""

from __future__ import annotations

from .models import Claim, Confidence


def score(claims: list[Claim], retrieval: float, trust: float = 1.0, min_confidence: float = 0.6) -> Confidence:
    """overall = trust * verification * (retrieval + support) / 2

    - verification: fraction of generated claims that survived citation checks. Multiplying by it
      means a single fabricated claim among two drags the answer below the default threshold.
    - retrieval: query coverage of the best cited chunk (is the topic in the context at all?).
    - support: mean lexical support of the verified claims.
    - trust: <1 for answers built from search results rather than the curated KB.
    """
    if not claims:
        return Confidence(retrieval=retrieval, verification=0.0, support=0.0, overall=0.0, level="low")
    good = [c for c in claims if c.verified]
    verification = len(good) / len(claims)
    support = sum(c.support for c in good) / len(good) if good else 0.0
    overall = round(trust * verification * (retrieval + support) / 2, 3)
    level = "high" if overall >= max(0.8, min_confidence) else "medium" if overall >= min_confidence else "low"
    return Confidence(
        retrieval=round(retrieval, 3), verification=round(verification, 3),
        support=round(support, 3), overall=overall, level=level,
    )
