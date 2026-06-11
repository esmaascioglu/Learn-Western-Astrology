"""Chunking validation report.

For every book: runs its chunk strategy, prints a coverage summary (chunk counts,
token distribution, indicator_type breakdown) to the console, and writes a CSV of
RANDOM sample chunks with full metadata + a content preview so the chunking can be
reviewed/approved before embedding.

Run:
  .venv/bin/python scripts/chunk_report.py                 # 25 samples/book
  .venv/bin/python scripts/chunk_report.py --samples 40 --seed 7
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter

from rich.console import Console
from rich.table import Table

from app.config import PROJECT_ROOT
from app.ingestion.books import BOOKS
from app.ingestion.chunk import chunk_book

console = Console()

CSV_FIELDS = [
    "book", "strategy", "chunk_index", "page_start", "page_end", "token_count",
    "indicator_type", "context", "transiting", "aspect", "natal", "planet",
    "sign", "house", "section", "content_preview",
]


def _row(book, chunks, c) -> dict:
    md = c.metadata
    preview = c.content.replace("\n", " ").strip()
    return {
        "book": book.slug,
        "strategy": book.chunk_strategy,
        "chunk_index": c.chunk_index,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "token_count": c.token_count,
        "indicator_type": md.get("indicator_type", ""),
        "context": md.get("context", ""),
        "transiting": md.get("transiting", ""),
        "aspect": md.get("aspect", ""),
        "natal": md.get("natal", ""),
        "planet": md.get("planet", ""),
        "sign": md.get("sign", ""),
        "house": md.get("house", ""),
        "section": md.get("section", ""),
        "content_preview": preview[:300] + ("…" if len(preview) > 300 else ""),
    }


def main(n_samples: int, seed: int) -> None:
    rng = random.Random(seed)
    rows: list[dict] = []

    for book in BOOKS:
        chunks = chunk_book(book)
        toks = sorted(c.token_count for c in chunks)
        types = Counter(c.metadata.get("indicator_type", "—") for c in chunks)
        empty_meta = sum(1 for c in chunks if not c.metadata.get("indicator_type"))

        console.rule(f"[bold]{book.slug}[/]  ·  {book.chunk_strategy}")
        console.print(
            f"chunks: [bold]{len(chunks)}[/] · tokens min/median/max: "
            f"{toks[0]}/{toks[len(toks)//2]}/{toks[-1]} · untyped: {empty_meta}"
        )
        t = Table("indicator_type", "count", show_edge=False)
        for it, cnt in types.most_common():
            t.add_row(str(it), str(cnt))
        console.print(t)

        sample = rng.sample(chunks, min(n_samples, len(chunks)))
        sample.sort(key=lambda c: c.chunk_index)
        rows.extend(_row(book, chunks, c) for c in sample)

    out = PROJECT_ROOT / "artifacts" / "chunk_report.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    console.print(f"\n[green]Wrote {len(rows)} sampled chunks → {out}[/]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=25, help="random chunks per book")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.samples, args.seed)
