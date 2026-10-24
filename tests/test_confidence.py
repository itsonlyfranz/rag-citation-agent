from rag_citation_agent.confidence import score
from rag_citation_agent.models import Claim


def c(verified=True, support=1.0):
    return Claim(text="t", verified=verified, support=support)


def test_fully_verified_answer_is_high():
    s = score([c(), c()], retrieval=0.9)
    assert s.level == "high" and s.overall == 0.95


def test_one_fabricated_claim_in_two_drops_below_threshold():
    s = score([c(), c(verified=False, support=0)], retrieval=0.9)
    assert s.verification == 0.5 and s.level == "low"


def test_no_claims_is_zero_and_low():
    s = score([], retrieval=0.9)
    assert s.overall == 0 and s.level == "low"


def test_search_trust_penalty_lowers_score():
    assert score([c()], 0.8, trust=0.9).overall < score([c()], 0.8).overall


def test_threshold_controls_level():
    claims = [c(support=0.7)]
    assert score(claims, 0.7, min_confidence=0.6).level == "medium"
    assert score(claims, 0.7, min_confidence=0.9).level == "low"
