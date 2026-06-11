"""Measure document structure to design a per-document, heading-aware chunker.

For each PDF reports:
  1. Paragraph token distribution (min/median/mean/p90/p95/max).
  2. Font-size histogram (body vs. larger heading sizes), sampled mid-document.
  3. Sample texts at heading-candidate sizes — to see if sections are split by font.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import fitz
import tiktoken

from app.ingestion.books import BOOKS
from app.ingestion.parse import parse_pdf

enc = tiktoken.get_encoding("cl100k_base")


def ntok(s: str) -> int:
    return len(enc.encode(s))


def para_stats(slug, path):
    pages = parse_pdf(path)
    toks = sorted(ntok(p) for pg in pages for p in pg.text.split("\n\n") if p.strip())
    n = len(toks)

    def pct(f: float) -> int:  # f-quantile of the sorted token counts
        return toks[int(f * (n - 1))]

    print(f"\n[{slug}]  paragraphs={n}")
    print(f"  tokens  min={toks[0]}  median={pct(.5)}  mean={sum(toks)//n}  "
          f"p90={pct(.9)}  p95={pct(.95)}  max={toks[-1]}")


def font_stats(slug, path, page_lo, page_hi):
    doc = fitz.open(path)
    size_chars = Counter()          # rounded size -> total chars
    size_samples = defaultdict(list)
    for i in range(page_lo, min(page_hi, doc.page_count)):
        d = doc.load_page(i).get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join(s["text"] for s in line.get("spans", []))
                for s in line.get("spans", []):
                    sz = round(s["size"], 1)
                    txt = s["text"].strip()
                    size_chars[sz] += len(txt)
                    if txt and len(size_samples[sz]) < 6:
                        size_samples[sz].append(line_text.strip()[:60])
    doc.close()
    body = size_chars.most_common(1)[0][0]
    print(f"  body font size ≈ {body}")
    print("  size : chars : samples (larger-than-body = heading candidates)")
    for sz in sorted(size_chars, reverse=True):
        if size_chars[sz] < 30:
            continue
        flag = "  <-- HEADINGS?" if sz > body + 0.4 else ""
        print(f"   {sz:>5} : {size_chars[sz]:>6} : {size_samples[sz][:3]}{flag}")


# How many extracted lines look like Hand transit headings "<Planet> <Aspect> <Planet>"
ASPECTS = "Conjunct|Sextile|Square|Trine|Opposition|Conjunction|Quincunx|Semisextile"
PLANETS = "Sun|Moon|Mercury|Venus|Mars|Jupiter|Saturn|Uranus|Neptune|Pluto|Ascendant|Midheaven"
HEAD_RE = re.compile(rf"^({PLANETS})\s+({ASPECTS})\s+({PLANETS})\b")


def heading_line_scan(slug, path):
    doc = fitz.open(path)
    hits = 0
    samples = []
    for i in range(doc.page_count):
        for ln in doc.load_page(i).get_text("text").splitlines():
            if HEAD_RE.match(ln.strip()):
                hits += 1
                if len(samples) < 8:
                    samples.append(ln.strip()[:50])
    doc.close()
    print(f"  '<Planet> <Aspect> <Planet>' heading-lines found: {hits}")
    print(f"  samples: {samples}")


if __name__ == "__main__":
    for book in BOOKS:
        print("=" * 72)
        para_stats(book.slug, book.path)
        # sample the middle of each book (skip front matter)
        font_stats(book.slug, book.path, page_lo=90, page_hi=110)
        heading_line_scan(book.slug, book.path)
