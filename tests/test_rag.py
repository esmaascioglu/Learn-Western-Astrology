"""RAG orchestration: prompt construction, citation surfacing, no-context guard.

The retriever and the LLM are stubbed so these test the orchestration logic only,
with no DB or network calls.
"""
from __future__ import annotations

import app.rag as rag
from app.retrieval.search import Retrieved


def _ctx(n: int, content: str) -> Retrieved:
    return Retrieved(
        chunk_id=n, document_id=1, content=content, page_start=n, page_end=n,
        metadata={}, score=0.5 / n, book_title="Book", book_author="Author",
    )


def test_format_sources_numbers_from_one():
    text = rag._format_sources([_ctx(1, "first"), _ctx(2, "second")])
    assert "[1] Book (Author), p.1\nfirst" in text
    assert "[2] Book (Author), p.2\nsecond" in text


def test_messages_carry_system_prompt_and_sources():
    msgs = rag._messages("why?", [_ctx(1, "grounding text")])
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "grounding text" in msgs[1]["content"]
    assert "why?" in msgs[1]["content"]


def test_answer_sources_expose_citation_and_score():
    ans = rag.Answer(text="x", contexts=[_ctx(1, "a"), _ctx(2, "b")])
    sources = ans.sources
    assert [s["n"] for s in sources] == [1, 2]
    assert sources[0]["citation"] == "Book (Author), p.1"


def test_answer_returns_no_context_message_when_retrieval_empty(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda question, **kw: [])
    out = rag.answer("anything")
    assert out.contexts == []
    assert out.text == rag._NO_CONTEXT


def test_answer_generates_from_retrieved_context(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda question, **kw: [_ctx(1, "grounding")])
    monkeypatch.setattr(rag.gateway, "complete", lambda messages: "the answer [1]")
    out = rag.answer("question")
    assert out.text == "the answer [1]"
    assert len(out.contexts) == 1
