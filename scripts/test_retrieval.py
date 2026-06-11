"""Retrieval test bench — edit QUESTIONS, run, and for every retrieved chunk see
*why* it came back: keyword (FTS) vs semantic (vector) contribution, which query
terms matched the heading vs the body, plus full metadata and content.

    .venv/bin/python scripts/test_retrieval.py

This mirrors production retrieval (same OR-FTS + heading-weighted + SPLADE RRF
fusion as `hybrid_search`), but breaks the fused score into its three legs (FTS /
dense / SPLADE) so you can reason about why each result came back.
"""
from __future__ import annotations

import re

from rich.console import Console

from app.config import get_settings
from app.ingestion.embed import embed_query
from app.ingestion.store import connect
from app.retrieval.query import focus_query
from app.retrieval.splade import encode_query, to_pgvector

# ─────────────────────────  EDIT HERE  ─────────────────────────
QUESTIONS = [
    "Which combinations in a natal chart make you stubborn?",
    "What makes someone confident in their natal chart?",
    "Transiting Sun conjunct natal Sun",
    "What does Venus in Gemini mean in a natal chart?",
    "Saturn transit through the seventh house",
]
TOP_K = 12          # how many candidates to display per question
PREVIEW_CHARS = 260  # content preview length
# ───────────────────────────────────────────────────────────────

console = Console()
_STOP = {"what", "does", "do", "mean", "means", "is", "are", "the", "a", "an", "in",
         "of", "my", "your", "his", "her", "to", "and", "for", "how", "with", "when",
         "it", "that", "this", "i", "you", "natal", "chart"}

# Replicates hybrid_search but EXPOSES each leg: FTS rank/score, dense rank/sim,
# and SPLADE rank/dot. Weights mirror production (fts 0.7 / dense 1.0 / sparse 1.0).
DIAG_SQL = """
with q as (
    select to_tsquery('english',
        replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | ')) as tsq
),
fts as (
    select c.id,
           ts_rank_cd(c.fts, q.tsq) as frank,
           row_number() over (order by ts_rank_cd(c.fts, q.tsq) desc) as fpos
    from chunks c, q
    where c.fts @@ q.tsq
    order by frank desc
    limit %(leg)s
),
sem as (
    select c.id,
           1 - (c.embedding <=> %(emb)s::vector) as sim,
           row_number() over (order by c.embedding <=> %(emb)s::vector) as spos
    from chunks c
    order by c.embedding <=> %(emb)s::vector
    limit %(leg)s
),
spl as (
    select c.id,
           (c.sparse_embedding <#> %(sp)s::sparsevec) * -1 as sdot,
           row_number() over (order by c.sparse_embedding <#> %(sp)s::sparsevec) as sppos
    from chunks c
    where c.sparse_embedding is not null
    order by c.sparse_embedding <#> %(sp)s::sparsevec
    limit %(leg)s
),
ids as (
    select id from fts union select id from sem union select id from spl
),
comb as (
    select i.id,
           coalesce(0.7 / (50 + fts.fpos), 0) +
           coalesce(1.0 / (50 + sem.spos), 0) +
           coalesce(1.0 / (50 + spl.sppos), 0) as rrf,
           fts.fpos, fts.frank, sem.spos, sem.sim, spl.sppos, spl.sdot
    from ids i
    left join fts on fts.id = i.id
    left join sem on sem.id = i.id
    left join spl on spl.id = i.id
)
select comb.rrf, comb.fpos, comb.frank, comb.spos, comb.sim, comb.sppos, comb.sdot,
       c.content, c.metadata, d.title, c.page_start, c.page_end
from comb
join chunks c on c.id = comb.id
join documents d on d.id = c.document_id
order by comb.rrf desc
limit %(topk)s
"""


def _query_terms(q: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", q.lower()) if w not in _STOP and len(w) > 2]


def _why(md: dict, content: str, fpos, spos, sppos, q: str) -> str:
    sec = (md.get("section") or "").lower()
    body = content.lower()
    qt = list(dict.fromkeys(_query_terms(q)))  # unique terms, order preserved
    in_head = [t for t in qt if t in sec]
    in_body = [t for t in qt if t in body and t not in in_head]
    parts = []
    if fpos is not None:
        kw = f"KEYWORD (FTS #{fpos})"
        if in_head:
            kw += f" — matched in heading: {in_head}"
        if in_body:
            kw += f" — in body: {in_body}"
        parts.append(kw)
    if spos is not None:
        parts.append(f"SEMANTIC (vector #{spos})")
    if sppos is not None:
        parts.append(f"SPLADE (sparse #{sppos})")
    if not parts:
        parts.append("(unclear)")
    return "  +  ".join(parts)


def run(q: str) -> None:
    # Lexical legs use the focused query (framing words stripped); dense uses the
    # full question — mirrors production retrieve().
    qf = focus_query(q)
    emb = "[" + ",".join(f"{x:.8f}" for x in embed_query(q)) + "]"
    sp = to_pgvector(encode_query(qf))
    with connect() as conn:
        rows = conn.execute(
            DIAG_SQL, {"q": qf, "emb": emb, "sp": sp, "leg": TOP_K * 2, "topk": TOP_K}
        ).fetchall()

    top_n = get_settings().top_n
    console.rule(f"[bold cyan]{q}[/]")
    focus_note = f"  [dim]→ lexical focus: '{qf}'[/]" if qf != q.lower() else ""
    console.print(f"[dim]{len(rows)} candidates · top {top_n} go to the LLM[/]{focus_note}")

    for i, (rrf, fpos, frank, spos, sim, sppos, sdot, content, md, title, p0, p1) \
            in enumerate(rows, 1):
        sel = "[green]✓ to LLM[/]" if i <= top_n else "[dim]·[/]"
        fts = f"FTS #{fpos} (rank {frank:.3f})" if fpos is not None else "FTS —"
        vec = f"VEC #{spos} (sim {sim:.3f})" if spos is not None else "VEC —"
        spl = f"SPLADE #{sppos} (dot {sdot:.2f})" if sppos is not None else "SPLADE —"
        pages = f"p.{p0}" if p0 == p1 else f"pp.{p0}-{p1}"
        console.print(
            f"\n[bold]#{i}[/] {sel}  RRF=[bold]{rrf:.4f}[/]   "
            f"[magenta]{fts}[/] | [blue]{vec}[/] | [green]{spl}[/]"
        )
        console.print(
            f"   [cyan]{md.get('section', '?')}[/]  "
            f"([yellow]{md.get('indicator_type', '-')}[/]/{md.get('context', '-')})  "
            f"· {title[:30]} {pages}"
        )
        console.print(f"   [yellow]WHY:[/] {_why(md, content, fpos, spos, sppos, q)}")
        preview = content.replace("\n", " ")[:PREVIEW_CHARS]
        console.print("   " + preview + "…", markup=False)


if __name__ == "__main__":
    for question in QUESTIONS:
        run(question)
    console.print()
