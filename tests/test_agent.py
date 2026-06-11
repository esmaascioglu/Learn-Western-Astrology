"""Phase-1 agent wiring: the search tool's contract and graph construction.

The retriever itself is stubbed (it's already covered by eval/retrieval_metrics.py
and tests/test_rag.py) — these tests check the AGENT-FACING surface: what the
tool returns to the LLM, its schema, and that the graph builds with the expected
tool attached. No DB, no real OpenAI/Langfuse calls.
"""
from __future__ import annotations

import app.agent.tools as tools
from app.agent.graph import build_agent, get_callbacks
from app.retrieval.search import Retrieved


def _ctx(n: int, content: str) -> Retrieved:
    return Retrieved(
        chunk_id=n, document_id=1, content=content, page_start=n, page_end=n,
        metadata={}, score=0.5 / n, book_title="Book", book_author="Author",
    )


# --- search_knowledge_base tool --------------------------------------------
def test_tool_schema_exposes_a_query_string_argument():
    # This is what the LLM actually sees when deciding how to call the tool.
    schema = tools.search_knowledge_base.args_schema.model_json_schema()
    assert schema["properties"]["query"]["type"] == "string"


def test_tool_formats_results_with_numbered_citations(monkeypatch):
    monkeypatch.setattr(tools, "retrieve", lambda query: [_ctx(1, "Saturn text")])
    out = tools.search_knowledge_base.invoke({"query": "Saturn in Taurus"})
    assert "[1] Book (Author), p.1" in out
    assert "Saturn text" in out


def test_tool_reports_no_results_plainly(monkeypatch):
    monkeypatch.setattr(tools, "retrieve", lambda query: [])
    out = tools.search_knowledge_base.invoke({"query": "something obscure"})
    assert "No relevant passages" in out


# --- graph construction -----------------------------------------------------
def test_build_agent_wires_the_search_tool(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = build_agent()
    # The compiled graph must include our tool node so the LLM can call it.
    tool_node = agent.nodes["tools"].bound
    tool_names = {t.name for t in tool_node.tools_by_name.values()}
    assert tool_names == {"search_knowledge_base"}


def test_get_callbacks_empty_without_langfuse_keys(monkeypatch):
    # obs.ENABLED is computed at import time from settings; this project's test
    # .env has no LANGFUSE_* keys, so tracing is off and callbacks() is a no-op.
    from app import obs

    monkeypatch.setattr(obs, "ENABLED", False)
    assert get_callbacks() == []
