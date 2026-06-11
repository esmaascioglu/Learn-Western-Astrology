"""Command-line entrypoint:  `astro <command>`."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from app.ingestion.books import BOOKS
from app.ingestion.chunk import chunk_book

app = typer.Typer(add_completion=False, help="Learn-Western-Astrology pipeline.")
console = Console()


@app.command()
def inspect(
    samples: int = 3,
    grep: str = typer.Option(None, help="Only show chunks whose content matches this regex."),
):
    """Chunk locally and print stats/samples — for sanity checks."""
    import re as _re

    for book in BOOKS:
        chunks = chunk_book(book)
        toks = sorted(c.token_count for c in chunks)
        pages = max(c.page_end for c in chunks)
        console.rule(f"[bold]{book.title}[/]  ({book.slug})")
        console.print(
            f"strategy: [cyan]{book.chunk_strategy}[/] · pages: {pages} · chunks: {len(chunks)} · "
            f"tokens min/median/max: {toks[0]}/{toks[len(toks)//2]}/{toks[-1]}"
        )
        shown = chunks
        if grep:
            rx = _re.compile(grep, _re.IGNORECASE)
            shown = [c for c in chunks if rx.search(c.content)]
            console.print(f"[dim]{len(shown)} chunks match /{grep}/[/]")
        for c in shown[:samples]:
            console.print(
                f"\n[dim]— chunk {c.chunk_index} · pp.{c.page_start}-{c.page_end} · "
                f"{c.token_count} tok · {c.metadata}[/]"
            )
            console.print(c.content[:500] + ("…" if len(c.content) > 500 else ""))


@app.command()
def retrieve(
    question: str = typer.Argument(..., help="The question to inspect retrieval for."),
    top_k: int = typer.Option(20, help="Hybrid candidates to consider."),
    top_n: int = typer.Option(6, help="Passages sent to the LLM."),
    all_full: bool = typer.Option(False, help="Show full text for non-selected candidates too."),
):
    """Inspect retrieval for ONE question: show the top selected chunks (full text)
    and the remaining candidates, all with their hybrid (RRF) scores."""
    from app.retrieval.search import retrieve_debug

    cands = retrieve_debug(question, top_k=top_k, top_n=top_n)
    if not cands:
        console.print("[red]No candidates returned.[/]")
        return

    console.rule(f"[bold]{question}[/]")
    console.print(f"[dim]{len(cands)} hybrid candidates (RRF order) → top {top_n} selected[/]\n")

    selected = [c for c in cands if c.selected]
    dropped = [c for c in cands if not c.selected]

    console.print("[bold green]══ SELECTED (sent to the LLM) ══[/]")
    for i, c in enumerate(selected, 1):
        console.print(f"\n[green]#{i}[/]  score=[bold]{c.score:.4f}[/]  [cyan]{c.citation}[/]"
                      f"  [dim]tags={c.metadata}[/]")
        console.print(c.content)

    console.print("\n[bold yellow]══ NOT SELECTED ══[/]")
    for c in dropped:
        console.print(f"\n[yellow]·[/] score=[bold]{c.score:.4f}[/]  [cyan]{c.citation}[/]")
        console.print(c.content if all_full else c.content[:160].replace("\n", " ") + "…")


@app.command()
def ingest(
    reembed: bool = typer.Option(False, help="Ignore the embedding cache and re-embed."),
    upsert: bool = typer.Option(True, help="Upsert to Supabase after embedding."),
):
    """Full pipeline: chunk → embed (OpenAI dense + SPLADE sparse, cached) → upsert.

    Both vector sets are cached to artifacts/ so a DB failure never costs a
    re-embed. Dense embeddings cost OpenAI calls; SPLADE is computed locally
    (free) but is also cached so retries / back-fills don't recompute it.
    The DB connection is opened only for the (fast) upsert.
    """
    from app.ingestion import cache

    table = Table("book", "pages", "chunks", "dense", "sparse", "upsert")
    for book in BOOKS:
        if cache.exists(book.slug) and not reembed:
            chunks, embeddings, sparse = cache.load(book.slug)
            embed_status = "cached"
        else:
            from app.ingestion.embed import embed_documents

            chunks = chunk_book(book)
            console.print(f"[cyan]embedding[/] {book.slug}: {len(chunks)} chunks (OpenAI)…")
            embeddings = embed_documents([c.content for c in chunks])
            sparse = None
            cache.save(book.slug, chunks, embeddings)
            embed_status = "✓ fresh"

        # SPLADE sparse vectors: compute locally if missing, then (re)cache.
        if sparse is None:
            from app.retrieval.splade import encode

            console.print(f"[cyan]SPLADE[/] {book.slug}: encoding {len(chunks)} chunks (local)…")
            sparse = encode([c.content for c in chunks])
            cache.save(book.slug, chunks, embeddings, sparse)
            sparse_status = "✓ fresh"
        else:
            sparse_status = "cached"

        num_pages = max(c.page_end for c in chunks)
        upsert_status = "skipped"
        if upsert:
            from app.ingestion.store import connect, replace_chunks, upsert_document

            with connect() as conn:
                doc_id = upsert_document(conn, book, num_pages=num_pages)
                replace_chunks(conn, doc_id, chunks, embeddings, sparse)
                conn.commit()
            upsert_status = "✓ upserted"

        table.add_row(book.slug, str(num_pages), str(len(chunks)),
                      embed_status, sparse_status, upsert_status)
    console.print(table)


@app.command()
def chat():
    """Interactive chat with the astrology tutor agent (LangGraph, Phase 1).

    Each line you send is one turn. The agent decides whether to call
    search_knowledge_base (shown as "→ searching: ..."); the conversation
    persists for the life of this process via an in-memory checkpointer.
    Type /exit to quit.
    """
    import uuid

    from langchain_core.messages import HumanMessage

    from app import obs
    from app.agent.graph import build_agent, get_callbacks

    agent = build_agent()
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "callbacks": get_callbacks(),
    }

    console.print(
        "[bold cyan]Astrology Tutor[/] — ask about signs, placements, houses, "
        "aspects, transits. [dim]/exit to quit[/]\n"
    )
    try:
        while True:
            try:
                question = console.input("[bold]you>[/] ")
            except (EOFError, KeyboardInterrupt):
                break
            question = question.strip()
            if not question:
                continue
            if question.lower() in {"/exit", "exit", "quit"}:
                break

            for step in agent.stream(
                {"messages": [HumanMessage(question)]}, config=config, stream_mode="values"
            ):
                msg = step["messages"][-1]
                if msg.type == "ai" and msg.tool_calls:
                    for call in msg.tool_calls:
                        query = call["args"].get("query", "")
                        console.print(f"[dim]→ searching: {query!r}[/]")
                elif msg.type == "ai":
                    console.print(f"\n[bold green]tutor>[/] {msg.content}\n")
    finally:
        obs.flush()


if __name__ == "__main__":
    app()
