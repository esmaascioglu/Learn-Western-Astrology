"""Supabase / Postgres persistence for documents and chunks."""
from __future__ import annotations

import json
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings
from app.ingestion.books import Book
from app.ingestion.chunk import Chunk


@contextmanager
def connect():
    s = get_settings()
    if not s.database_url:
        raise RuntimeError("DATABASE_URL is not set — fill it in .env")
    conn = psycopg.connect(
        s.database_url,
        connect_timeout=15,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def upsert_document(conn: psycopg.Connection, book: Book, num_pages: int) -> int:
    row = conn.execute(
        """
        insert into documents (slug, title, author, source_path, num_pages)
        values (%s, %s, %s, %s, %s)
        on conflict (slug) do update
            set title = excluded.title,
                author = excluded.author,
                source_path = excluded.source_path,
                num_pages = excluded.num_pages
        returning id
        """,
        (book.slug, book.title, book.author, str(book.path), num_pages),
    ).fetchone()
    return row[0]


def replace_chunks(conn: psycopg.Connection, document_id: int, chunks: list[Chunk],
                   embeddings: list[list[float]],
                   sparse: list[dict[int, float]] | None = None) -> None:
    """Idempotent: clears the book's existing chunks, then inserts fresh ones.

    `sparse` is the per-chunk SPLADE vector ({term_id: weight}); when omitted the
    sparse_embedding column is left null (FTS + dense legs still work)."""
    from app.retrieval.splade import to_pgvector

    sparse = sparse or [None] * len(chunks)
    conn.execute("delete from chunks where document_id = %s", (document_id,))
    with conn.cursor() as cur:
        cur.executemany(
            """
            insert into chunks
                (document_id, chunk_index, content, page_start, page_end,
                 metadata, token_count, embedding, sparse_embedding)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    document_id,
                    c.chunk_index,
                    c.content,
                    c.page_start,
                    c.page_end,
                    json.dumps(c.metadata),
                    c.token_count,
                    emb,
                    to_pgvector(sp) if sp is not None else None,
                )
                for c, emb, sp in zip(chunks, embeddings, sparse, strict=True)
            ],
        )
