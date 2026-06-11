"""PDF text extraction + cleaning (PyMuPDF).

Produces a list of cleaned pages. The books have real text layers, so no OCR is
needed. Cleaning handles the usual PDF artifacts: running headers/footers,
bare page numbers, and words hyphenated across line breaks.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Page:
    number: int  # 1-based PDF page number
    text: str


_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_BARE_PAGE_NUM = re.compile(r"^\s*[ivxlcdmIVXLCDM\d]{1,6}\s*$")
_WS = re.compile(r"[ \t]+")
_MULTI_BLANK = re.compile(r"\n{3,}")


def _detect_running_lines(raw_pages: list[str], threshold: float = 0.4) -> set[str]:
    """Find header/footer lines repeated across many pages (likely chrome)."""
    counts: Counter[str] = Counter()
    n = len(raw_pages)
    for text in raw_pages:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # only the first/last few lines can be running headers/footers
        for ln in lines[:2] + lines[-2:]:
            if 3 <= len(ln) <= 80:
                counts[ln] += 1
    return {ln for ln, c in counts.items() if c >= max(3, threshold * n)}


def _clean_page(text: str, running: set[str]) -> str:
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)  # de-hyphenate across line breaks
    # Rebuild paragraphs: blank lines separate paragraphs; within a paragraph,
    # line breaks become spaces. Drop running headers/footers and page numbers.
    paragraphs: list[str] = []
    current: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if s in running or _BARE_PAGE_NUM.match(s):
            continue
        current.append(s)
    if current:
        paragraphs.append(" ".join(current))

    cleaned = "\n\n".join(_WS.sub(" ", p).strip() for p in paragraphs if p.strip())
    return _MULTI_BLANK.sub("\n\n", cleaned).strip()


def clean_lines(path: str | Path) -> list[tuple[int, str]]:
    """Return (page_number, line_text) preserving line boundaries (so section
    headings stay on their own line) with running headers/footers and bare page
    numbers removed. Used by the structure-aware chunkers."""
    doc = fitz.open(path)
    try:
        raw = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
    finally:
        doc.close()
    running = _detect_running_lines(raw)
    out: list[tuple[int, str]] = []
    for i, text in enumerate(raw):
        for ln in text.splitlines():
            s = ln.strip()
            if not s or s in running or _BARE_PAGE_NUM.match(s):
                continue
            out.append((i + 1, s))
    return out


def reflow(lines: list[str]) -> str:
    """Join lines into prose, healing words hyphenated across line breaks."""
    buf = ""
    for ln in lines:
        if buf[-1:] == "-" and buf[-2:-1].isalpha():
            buf = buf[:-1] + ln.lstrip()
        else:
            buf = f"{buf} {ln}".strip() if buf else ln
    return _WS.sub(" ", buf).strip()


def parse_pdf(path: str | Path) -> list[Page]:
    doc = fitz.open(path)
    try:
        raw = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
    finally:
        doc.close()
    running = _detect_running_lines(raw)
    pages = []
    for i, text in enumerate(raw):
        cleaned = _clean_page(text, running)
        if cleaned:
            pages.append(Page(number=i + 1, text=cleaned))
    return pages
