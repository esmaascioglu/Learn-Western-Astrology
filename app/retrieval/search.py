"""Hybrid retrieval: dense + full-text + SPLADE sparse, fused with Reciprocal
Rank Fusion (RRF) in the `hybrid_search` SQL function.

Three complementary legs, no reranker: dense (semantic paraphrase), FTS (exact
lexical, heading-weighted), and SPLADE (learned-sparse, term-expanded — closes
the vocabulary gap). Results are returned directly in RRF order.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from app import obs
from app.config import get_settings
from app.ingestion.embed import embed_query
from app.ingestion.store import connect
from app.retrieval.query import focus_query
from app.retrieval.splade import encode_query, to_pgvector


@dataclass
class Retrieved:
    chunk_id: int
    document_id: int
    content: str
    page_start: int
    page_end: int
    metadata: dict
    score: float  # RRF score from hybrid_search
    book_title: str
    book_author: str | None
    selected: bool = False  # in the final top-n (used by the inspector)

    @property
    def citation(self) -> str:
        pages = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )
        # Author is included so every citation the LLM sees (and can repeat back to
        # the user) names the book AND its author, not just a bare title — needed
        # for proper attribution of the copyrighted source material.
        book = f"{self.book_title} ({self.book_author})" if self.book_author else self.book_title
        return f"{book}, {pages}"


def _embed_literal(question: str) -> str:
    """Embed the query and format it as a pgvector literal for ::vector casts."""
    emb = embed_query(question)
    return "[" + ",".join(f"{x:.8f}" for x in emb) + "]"


def _sparse(question: str):
    """SPLADE-encode the query into a pgvector SparseVector for the sparse leg."""
    return to_pgvector(encode_query(question))


@obs.observe(as_type="retriever", name="hybrid-retrieve", capture_output=False)
def retrieve(
    question: str,
    top_n: int | None = None,
    metadata_filter: dict | None = None,
) -> list[Retrieved]:
    s = get_settings()
    top_n = top_n or s.top_n

    # Lexical legs (FTS + SPLADE) search on the FOCUSED query — generic framing
    # words stripped so the discriminating terms drive ranking. The dense leg keeps
    # the full question (it matches on paraphrase/context, not keywords).
    focused = focus_query(question)
    with connect() as conn:
        rows = conn.execute(
            "select id, document_id, content, page_start, page_end, metadata, score "
            "from hybrid_search(%s, %s::vector, %s::sparsevec, %s::int, "
            "filter => %s::jsonb)",
            (focused, _embed_literal(question), _sparse(focused), top_n,
             json.dumps(metadata_filter or {})),
        ).fetchall()
        docs = dict(conn.execute(
            "select id, title || '\x1f' || coalesce(author,'') from documents"
        ).fetchall())

    results: list[Retrieved] = []
    for r in rows:
        if r[6] < s.relevance_floor:
            continue
        title, _, author = docs[r[1]].partition("\x1f")
        results.append(Retrieved(
            r[0], r[1], r[2], r[3], r[4], r[5], float(r[6]), title, author or None,
            selected=True,
        ))

    if obs.ENABLED:
        obs.client().update_current_span(
            output=[{"citation": r.citation, "score": round(r.score, 4)} for r in results]
        )
    return results


def retrieve_debug(
    question: str, top_k: int | None = None, top_n: int | None = None
) -> list[Retrieved]:
    """Return all top_k hybrid candidates (RRF order), marking the final top_n.
    For inspecting a single question's retrieval."""
    s = get_settings()
    top_k = top_k or s.retrieval_top_k
    top_n = top_n or s.top_n

    with connect() as conn:
        focused = focus_query(question)
        rows = conn.execute(
            "select id, document_id, content, page_start, page_end, metadata, score "
            "from hybrid_search(%s, %s::vector, %s::sparsevec, %s::int)",
            (focused, _embed_literal(question), _sparse(focused), top_k),
        ).fetchall()
        docs = dict(conn.execute("select id, title from documents").fetchall())

    cands = [
        Retrieved(r[0], r[1], r[2], r[3], r[4], r[5], float(r[6]), docs[r[1]], None)
        for r in rows
    ]
    for c in cands[:top_n]:
        c.selected = True
    return cands
