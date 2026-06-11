"""The agent's only tool: search the Phase-0 hybrid-retrieval knowledge base.

The docstring below IS the tool's interface to the model — LangChain turns it
(plus the type hints) into the JSON schema the LLM sees when deciding whether
and how to call this tool. Keep it accurate: the model has no other information
about what this tool does or when to use it.
"""
from __future__ import annotations

from langchain_core.tools import tool

from app.retrieval.search import retrieve


@tool
def search_knowledge_base(query: str) -> str:
    """Search the Western astrology reference library for passages relevant to
    a question about astrological concepts, placements, signs, houses, aspects,
    or transits.

    Use this whenever the user asks something that requires looking up what the
    source books say — which is almost always. Write a focused query describing
    the astrological concept you need (e.g. "Saturn in Taurus personality" or
    "what is a decanate"), not the user's full sentence.

    Returns numbered passages, each with a citation (book title and page). Use
    these citation numbers when answering. If no passages are returned, say so
    plainly instead of guessing.
    """
    results = retrieve(query)
    if not results:
        return "No relevant passages found in the knowledge base."
    return "\n\n".join(f"[{i}] {r.citation}\n{r.content}" for i, r in enumerate(results, 1))
