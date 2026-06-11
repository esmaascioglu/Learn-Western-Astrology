"""Measure the *natural* chunk unit for each book, to size chunkers from data."""
from __future__ import annotations

import re

import fitz
import tiktoken

from app.ingestion.books import BOOKS_BY_SLUG

enc = tiktoken.get_encoding("cl100k_base")


def ntok(s: str) -> int:
    return len(enc.encode(s))


def dist(label, toks):
    toks = sorted(t for t in toks if t > 0)
    if not toks:
        print(f"  {label}: none")
        return
    n = len(toks)

    def pct(f: float) -> int:  # f-quantile of the sorted token counts
        return toks[int(f * (n - 1))]

    print(f"  {label}: n={n}  min={toks[0]} median={pct(.5)} mean={sum(toks)//n} "
          f"p90={pct(.9)} p95={pct(.95)} max={toks[-1]}")


def raw_text(path):
    doc = fitz.open(path)
    t = "\n".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))
    doc.close()
    return t


# --- Hand: split on aspect headings, measure section sizes ---
ASPECTS = "Conjunct|Sextile|Square|Trine|Opposition|Conjunction|Quincunx|Semisextile"
PLANETS = "Sun|Moon|Mercury|Venus|Mars|Jupiter|Saturn|Uranus|Neptune|Pluto|Ascendant|Midheaven"
HEAD_RE = re.compile(rf"^\s*({PLANETS})\s+({ASPECTS})\s+({PLANETS})\s*$", re.M)

print("=" * 60, "\n[planets-in-transit] aspect-section sizes")
text = raw_text(BOOKS_BY_SLUG["planets-in-transit"].path)
starts = [m.start() for m in HEAD_RE.finditer(text)]
sections = [text[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
dist("sections", [ntok(s) for s in sections])

# --- Libra: real paragraphs (blank-line) + heading candidates ---
print("=" * 60, "\n[astrology-technics-ethics] paragraph + heading scan")
ltext = raw_text(BOOKS_BY_SLUG["astrology-technics-ethics"].path)
paras = [p for p in re.split(r"\n\s*\n", ltext) if p.strip()]
dist("blank-line paragraphs", [ntok(p) for p in paras])

# candidate headings: short lines, mostly Title Case or ALL CAPS, or Chapter/roman
cap_re = re.compile(r"^[A-Z][A-Za-z .,'-]{2,40}$")
chap_re = re.compile(r"^\s*(CHAPTER|Chapter|[IVXLC]+\.)\b")
heads = []
for ln in ltext.splitlines():
    s = ln.strip()
    if not s:
        continue
    words = s.split()
    title_case = 2 <= len(words) <= 6 and s == s.title()
    all_caps = s.isupper() and 3 <= len(s) <= 40
    if chap_re.match(s) or title_case or all_caps:
        heads.append(s)
print(f"  heading-candidate lines: {len(heads)}")
print(f"  samples: {heads[:14]}")
