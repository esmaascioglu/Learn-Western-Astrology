"""Registry of source books. Edit titles/authors here as you verify them."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import DATA_DIR


@dataclass(frozen=True)
class Book:
    slug: str
    filename: str
    title: str
    author: str | None = None
    # Chunking strategy (see app/ingestion/chunk.py):
    #   "transit_sections" — aspect + house-ingress headings (Hand)
    #   "toc_sections"     — table-of-contents sections via page offset (Libra)
    #   "font_sections"    — native-font heading hierarchy + run-in (Woolfolk)
    #   "token_windows"    — fallback for unstructured text
    chunk_strategy: str = "token_windows"
    chunk_target: int = 450  # target tokens for sub-splitting long sections

    @property
    def path(self):
        return DATA_DIR / self.filename


BOOKS: list[Book] = [
    Book(
        slug="planets-in-transit",
        filename="robert-hand-planets-in-transitspdf_compress.pdf",
        title="Planets in Transit: Life Cycles for Living",
        author="Robert Hand",
        chunk_strategy="transit_sections",
    ),
    Book(
        slug="astrology-technics-ethics",
        filename="astrologyitstech00libr.pdf",
        title="Astrology: Its Technics and Ethics",
        author="C. Aq. Libra",
        chunk_strategy="toc_sections",
        chunk_target=450,
    ),
    Book(
        slug="only-astrology-book",
        filename="theonlyastrologybookyouwilleverneed.pdf",
        title="The Only Astrology Book You'll Ever Need",
        author="Joanna Martine Woolfolk",
        chunk_strategy="font_sections",
        chunk_target=450,
    ),
]

BOOKS_BY_SLUG = {b.slug: b for b in BOOKS}
