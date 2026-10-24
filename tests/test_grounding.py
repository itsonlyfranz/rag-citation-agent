from rag_citation_agent.generation import RawAnswer, RawCitation, RawClaim, parse_answer, MalformedOutput
from rag_citation_agent.grounding import support_score, verify
from rag_citation_agent.models import Chunk, RetrievedChunk
import pytest

SRC = "Audit logs are retained for 90 days on the Team plan and for 400 days on the Enterprise plan."


def ctx():
    chunk = Chunk(id="d#0", doc_id="d", title="Doc", text=SRC)
    return [RetrievedChunk(chunk=chunk, score=0.5, coverage=0.9)]


def claim(text, quote=SRC, source="S1"):
    return RawAnswer(claims=[RawClaim(text=text, citations=[RawCitation(source=source, quote=quote)])])


def check(raw, min_support=0.6):
    return verify(raw, ctx(), min_support)[0]


def test_verbatim_quote_and_supported_claim_verifies():
    c = check(claim("Enterprise audit logs are retained for 400 days.", "for 400 days on the Enterprise plan"))
    assert c.verified and c.citations[0].chunk_id == "d#0" and c.issues == []


def test_quote_matching_ignores_case_and_whitespace():
    assert check(claim("Logs are retained 90 days.", "AUDIT   logs are retained\nfor 90 days")).verified


def test_fabricated_quote_is_rejected():
    c = check(claim("Logs are kept forever.", "Audit logs are kept forever on every plan."))
    assert not c.verified and c.citations == [] and "not in the source" in c.issues[0]


def test_unknown_source_id_is_rejected():
    c = check(claim("x logs", source="S9"))
    assert not c.verified and "unknown source" in c.issues[0]


def test_claim_without_citations_is_rejected():
    c = check(RawAnswer(claims=[RawClaim(text="Logs exist.")]))
    assert not c.verified and "no citations" in c.issues


def test_real_quote_but_unrelated_claim_is_rejected():
    c = check(claim("The company was founded in Berlin by two engineers.", "Audit logs are retained for 90 days"))
    assert not c.verified and any("not supported" in i for i in c.issues)


def test_number_not_in_quote_is_rejected_even_when_words_match():
    c = check(claim("Enterprise audit logs are retained for 500 days.", "for 400 days on the Enterprise plan"))
    assert not c.verified and any("numbers not in quotes" in i for i in c.issues)


def test_support_score_bounds():
    assert support_score("alpha beta", "alpha beta gamma") == 1.0
    assert support_score("alpha beta", "gamma") == 0.0
    assert support_score("the of", "anything") == 0.0


def test_parse_answer_accepts_fenced_json_and_rejects_garbage():
    assert parse_answer('```json\n{"answerable": false, "claims": []}\n```').answerable is False
    for bad in ["no json here", '{"claims": "nope"}', "{broken"]:
        with pytest.raises(MalformedOutput):
            parse_answer(bad)


@pytest.mark.parametrize("text,quote", [
    ("Audit logs are retained for 40 days on the Enterprise plan.", "40"),
    ("Audit logs are retained for 40 days on the Enterprise plan.", "40 days on the Enterprise plan"),
    ("Audit logs are retained for 3 days on the Team plan.", "for 9"),
])
def test_quote_must_match_whole_tokens(text, quote):
    c = check(claim(text, quote))
    assert not c.verified and "not in the source" in c.issues[0]


def test_quote_without_content_terms_is_rejected():
    c = check(claim("Audit logs are retained on the Enterprise plan.", "the"))
    assert not c.verified and "not in the source" in c.issues[0]
