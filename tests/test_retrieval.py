from pathlib import Path

from rag_citation_agent.index import KnowledgeBase, TfidfIndex, chunk_document, load_documents
from rag_citation_agent.models import Document
from rag_citation_agent.text import content_tokens, numbers, split_sentences


def test_relevant_chunk_ranks_first_with_high_coverage(kb):
    top = kb.search("How long are audit logs retained on the Enterprise plan?")[0]
    assert top.chunk.uri == "security-and-data.md"
    assert "400 days" in top.chunk.text
    assert top.coverage > 0.6


def test_off_topic_query_has_no_hits_or_low_coverage(kb):
    hits = kb.search("Who won the 2022 football World Cup?")
    assert not hits or hits[0].coverage < 0.2


def test_unseen_terms_lower_coverage(kb):
    known = kb.search("refund window billing")[0].coverage
    mixed = kb.search("refund window billing zeppelin quokka")[0].coverage
    assert mixed < known


def test_chunking_respects_size_and_keeps_provenance():
    doc = Document(id="d", title="T", text="# T\n\n" + "\n\n".join(f"Paragraph number {i} has words." for i in range(30)), uri="d.md")
    chunks = chunk_document(doc, max_chars=100)
    assert len(chunks) > 1 and all(len(c.text) <= 100 for c in chunks)
    assert {c.uri for c in chunks} == {"d.md"} and chunks[0].id == "d#0"


def test_long_paragraph_is_split_by_sentence():
    doc = Document(id="d", title="T", text=" ".join(f"Sentence {i} is here." for i in range(40)))
    assert len(chunk_document(doc, max_chars=120)) > 3


def test_reingesting_a_document_replaces_its_chunks():
    kb = KnowledgeBase()
    kb.add([Document(id="a", title="A", text="alpha beta gamma")])
    kb.add([Document(id="a", title="A", text="delta epsilon")])
    assert len(kb) == 1 and kb.search("alpha") == []


def test_store_roundtrips_through_json(kb, tmp_path):
    path = tmp_path / "nested" / "index.json"
    kb.save(path)
    loaded = KnowledgeBase.load(path)
    q = "How long do refunds take?"
    assert [r.chunk.id for r in loaded.search(q)] == [r.chunk.id for r in kb.search(q)]


def test_load_documents_reads_titles_from_headings(tmp_path):
    (tmp_path / "x.md").write_text("# My Title\n\nbody text", encoding="utf-8")
    (tmp_path / "ignored.png").write_bytes(b"")
    docs = load_documents([tmp_path])
    assert [(d.id, d.title) for d in docs] == [("x", "My Title")]


def test_text_helpers():
    assert content_tokens("The Logs were retained") == ["log", "retain"]
    assert numbers("Pay $10,000 or 99.95% by day 30") == {"10000", "99.95", "30"}
    assert split_sentences("One thing. Another thing! A third?") == ["One thing.", "Another thing!", "A third?"]


def test_empty_index_and_stopword_only_query():
    assert TfidfIndex([]).search("anything") == []
    kb = KnowledgeBase()
    kb.add([Document(id="a", title="A", text="alpha beta")])
    assert kb.search("the and of") == []


def test_same_named_files_in_different_directories_get_distinct_ids(tmp_path):
    for sub, body in (("a", "alpha beta gamma"), ("b", "delta epsilon zeta")):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "notes.md").write_text(body, encoding="utf-8")
    docs = load_documents([tmp_path / "a", tmp_path / "b"])
    assert len({d.id for d in docs}) == 2
    kb = KnowledgeBase()
    kb.add(docs)
    assert len({c.id for c in kb.chunks}) == len(kb.chunks) == 2
    assert kb.search("alpha") and kb.search("delta")


def test_reingesting_a_nested_file_alone_replaces_its_chunks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs" / "guides").mkdir(parents=True)
    (tmp_path / "docs" / "a.md").write_text("alpha beta gamma", encoding="utf-8")
    setup = tmp_path / "docs" / "guides" / "setup.md"
    setup.write_text("delta epsilon zeta", encoding="utf-8")
    kb = KnowledgeBase()
    kb.add(load_documents([Path("docs")]))
    setup.write_text("omega sigma tau", encoding="utf-8")
    kb.add(load_documents([Path("docs/guides/setup.md")]))
    assert len(kb) == 2 and kb.search("delta") == [] and kb.search("omega")
