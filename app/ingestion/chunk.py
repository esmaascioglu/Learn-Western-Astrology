"""Per-document, structure-aware chunking.

Every book is split on its *own* natural units and tagged with a typed
"indicator" metadata schema so retrieval can filter precisely:

  - Hand "Planets in Transit"  -> `transit_sections`: split on transit headings
        "<Planet> <Aspect> <Planet>" (aspect) AND "<Planet> in the Nth House"
        (house ingress). context = transit.
  - Libra "Technics and Ethics" -> `toc_sections`: table-of-contents sections via
        printed->PDF page offset. context = natal.
  - Woolfolk "Only Astrology Book" -> `font_sections`: native-font heading
        hierarchy + ALL-CAPS run-in "PLANET IN SIGN" headings, with sign
        breadcrumb. context = natal (planet-in-sign, sign chapters, ascendant).
  - fallback `token_windows` for unstructured text.

Metadata schema (only the relevant keys are set per chunk):
  indicator_type : transit_aspect | angle_aspect | house_ingress |
                   planet_in_sign | sign_chapter | sign_section | ascendant |
                   house_meaning | aspect_meaning | sign_meaning | section
  context        : natal | transit
  transiting / natal / aspect / house / planet / sign / section
  planets / signs / aspects : entities mentioned in the chunk body (from _tag)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import fitz
import numpy as np
import tiktoken

from app.ingestion.books import Book
from app.ingestion.parse import clean_lines, reflow

_ENC = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# --- astrology vocabulary -------------------------------------------------
PLANETS = [
    "sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn",
    "uranus", "neptune", "pluto", "chiron", "lilith",
    "north node", "south node", "ascendant", "midheaven",
]
SIGNS = [
    "aries", "taurus", "gemini", "cancer", "leo", "virgo", "libra",
    "scorpio", "sagittarius", "capricorn", "aquarius", "pisces",
]
ASPECTS = [
    "conjunction", "conjunct", "opposition", "trine", "square", "sextile",
    "quincunx", "semisextile", "semisquare", "sesquiquadrate",
]


def _vocab_matcher(terms: list[str]) -> re.Pattern[str]:
    parts = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


_PLANET_RE = _vocab_matcher(PLANETS)
_SIGN_RE = _vocab_matcher(SIGNS)
_ASPECT_RE = _vocab_matcher(ASPECTS)


def _tag(text: str) -> dict:
    """Entities mentioned in the chunk body (planets/signs/aspects)."""
    low = text.lower()
    tags: dict = {}
    for key, rx in (("planets", _PLANET_RE), ("signs", _SIGN_RE), ("aspects", _ASPECT_RE)):
        found = sorted({m.group(1).lower() for m in rx.finditer(low)})
        if found:
            tags[key] = found
    return tags


# --- heading patterns -----------------------------------------------------
_PLANET_ALT = ("Sun|Moon|Mercury|Venus|Mars|Jupiter|Saturn|Uranus|Neptune|Pluto|"
               "Ascendant|Midheaven")
_SIGN_ALT = ("Aries|Taurus|Gemini|Cancer|Leo|Virgo|Libra|Scorpio|Sagittarius|"
             "Capricorn|Aquarius|Pisces")
_ASPECT_ALT = "Conjunct|Sextile|Square|Trine|Opposition|Quincunx|Semisextile|Semisquare"
_ORD_ALT = ("First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|"
            "Eleventh|Twelfth")
_ORDINALS = {w: i + 1 for i, w in enumerate(_ORD_ALT.split("|"))}
_SIGNS_SET = {s for s in SIGNS}

# transit aspect / angle headings, e.g. "Sun Conjunct Moon", "Sun Conjunct Ascendant"
HEAD_RE = re.compile(rf"^({_PLANET_ALT})\s+({_ASPECT_ALT})\s+({_PLANET_ALT})$", re.IGNORECASE)
# transit through a house, e.g. "Saturn in the Seventh House"
HOUSE_RE = re.compile(rf"^({_PLANET_ALT})\s+in\s+the\s+({_ORD_ALT})\s+House$", re.IGNORECASE)
# planet-in-sign as a full line, e.g. "Moon in Aries"
_PIS_RE = re.compile(rf"^({_PLANET_ALT})\s+in\s+({_SIGN_ALT})$", re.IGNORECASE)
# planet-in-sign as ALL-CAPS run-in heading, e.g. "MERCURY IN ARIES This is ..."
_CAPS_PIS_RE = re.compile(rf"^({_PLANET_ALT.upper()})\s+IN\s+({_SIGN_ALT.upper()})\b")
# ascendant / rising, e.g. "Aries Ascendant", "Taurus Rising"
_ASC_RE = re.compile(rf"^({_SIGN_ALT})\s+(Ascendant|Rising)\b", re.IGNORECASE)


@dataclass
class Chunk:
    chunk_index: int
    content: str
    page_start: int
    page_end: int
    token_count: int
    metadata: dict = field(default_factory=dict)


Line = tuple[int, str]  # (page_number, text)


def _windows(body: list[Line], target: int, overlap: int, heading: str | None):
    """Pack body lines into ~target-token windows; prepend `heading` to each."""
    out: list[tuple[str, int, int]] = []
    win: list[Line] = []
    wtok = 0

    def make() -> tuple[str, int, int]:
        text = reflow([t for _, t in win])
        content = f"{heading}\n\n{text}" if heading else text
        pages = [p for p, _ in win]
        return content, min(pages), max(pages)

    for page, text in body:
        utok = n_tokens(text)
        if win and wtok + utok > target:
            out.append(make())
            carry: list[Line] = []
            ctok = 0
            for u in reversed(win):
                t = n_tokens(u[1])
                if ctok + t > overlap:
                    break
                carry.insert(0, u)
                ctok += t
            win, wtok = list(carry), sum(n_tokens(t) for _, t in carry)
        win.append((page, text))
        wtok += utok
    if win:
        out.append(make())
    return out


# Lookup tables / indexes (ephemeris date→sign charts, "What Is Your Moon Sign?",
# the book index) are pure noise for an interpretive RAG — their function is the
# ephemeris tool's job. They are dense with uppercase codes (PIS, ARI) and
# numbers, so we drop chunks whose tokens are mostly non-lowercase.
_JUNK_RATIO_MAX = 0.5
_MIN_TOKENS = 12


def _is_useful(content: str) -> bool:
    toks = content.split()
    if len(toks) < _MIN_TOKENS:
        return False
    junk = sum(1 for t in toks if not any(c.islower() for c in t))
    return junk / len(toks) <= _JUNK_RATIO_MAX


def _emit(chunks: list[Chunk], body: list[Line], target: int, overlap: int,
          heading: str | None, meta: dict, start_idx: int) -> int:
    """Sub-split a section into chunks, attach metadata, return next index.
    Table/index junk is dropped (see _is_useful)."""
    idx = start_idx
    for content, ps, pe in _windows(body, target, overlap, heading):
        if not _is_useful(content):
            continue
        md = dict(meta)
        md.update(_tag(content))
        chunks.append(Chunk(idx, content, ps, pe, n_tokens(content), md))
        idx += 1
    return idx


# ========================================================================
# Hand — transit sections (aspects + house ingress)
# ========================================================================
def _hand_meta(head: str | None) -> dict:
    if head and (m := HEAD_RE.match(head)):
        transiting, aspect, natal = (g.lower() for g in m.groups())
        itype = "angle_aspect" if natal in ("ascendant", "midheaven") else "transit_aspect"
        return {"indicator_type": itype, "context": "transit", "transiting": transiting,
                "aspect": aspect, "natal": natal, "section": head}
    if head and (m := HOUSE_RE.match(head)):
        return {"indicator_type": "house_ingress", "context": "transit",
                "transiting": m.group(1).lower(),
                "house": _ORDINALS[m.group(2).title()], "section": head}
    return {"context": "transit"}


def _chunk_transit_sections(book: Book) -> list[Chunk]:
    sections: list[tuple[str | None, list[Line]]] = []
    head: str | None = None
    body: list[Line] = []
    for page, text in clean_lines(book.path):
        if HEAD_RE.match(text) or HOUSE_RE.match(text):
            sections.append((head, body))
            head, body = text, []
        else:
            body.append((page, text))
    sections.append((head, body))

    chunks: list[Chunk] = []
    idx = 0
    for head, body in sections:
        if not body:
            continue
        idx = _emit(chunks, body, target=900, overlap=80, heading=head,
                    meta=_hand_meta(head), start_idx=idx)
    return chunks


# ========================================================================
# Libra — table-of-contents sections
# ========================================================================
_TOC_RE = re.compile(r"([A-Z][^.;\n]{4,55}?);\s*(\d+)\)")


def _parse_toc(path) -> tuple[list[tuple[int, str]], int]:
    doc = fitz.open(path)
    try:
        toc_text = "\n".join(doc.load_page(i).get_text("text") for i in range(11, 15))
        by_page: dict[int, str] = {}
        for title, page in _TOC_RE.findall(toc_text):
            by_page.setdefault(int(page), re.sub(r"\s+", " ", title).strip())
        toc = sorted(by_page.items())

        offsets = []
        for page, title in toc:
            if len(title.split()) < 3:
                continue
            for i in range(15, doc.page_count):
                lines = doc.load_page(i).get_text("text").splitlines()
                if any(ln.strip().rstrip(".;:").lower() == title.lower() for ln in lines):
                    offsets.append(i - page)
                    break
        offset = int(np.median(offsets)) if offsets else 17
    finally:
        doc.close()
    return toc, offset


def _libra_indicator(title: str) -> str:
    t = title.lower()
    if "house" in t:
        return "house_meaning"
    if "ascendant" in t or "rising" in t:
        return "ascendant"
    if "aspect" in t:
        return "aspect_meaning"
    if "sign" in t or "zodiac" in t or "decanate" in t:
        return "sign_meaning"
    if "planet" in t or "ruler" in t or "sphere" in t:
        return "planet_meaning"
    return "section"


def _chunk_toc_sections(book: Book) -> list[Chunk]:
    toc, offset = _parse_toc(book.path)
    lines = clean_lines(book.path)
    bounds = sorted((p + offset, t) for p, t in toc)

    chunks: list[Chunk] = []
    idx = 0
    for i, (start, title) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else 1 << 30
        body = [(p, t) for p, t in lines if start <= p < end]
        if not body:
            continue
        meta = {"indicator_type": _libra_indicator(title), "context": "natal", "section": title}
        idx = _emit(chunks, body, target=book.chunk_target, overlap=60,
                    heading=title, meta=meta, start_idx=idx)
    return chunks


# ========================================================================
# Woolfolk — native-font heading hierarchy + run-in PLANET IN SIGN
# ========================================================================
def _font_lines(path) -> list[tuple[int, str, float]]:
    """(page, text, max_font_size) per line, preserving order."""
    doc = fitz.open(path)
    out = []
    try:
        for i in range(doc.page_count):
            for block in doc.load_page(i).get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(s["text"] for s in spans).strip()
                    if not spans or not text:
                        continue
                    size = round(max(s["size"] for s in spans), 1)
                    out.append((i + 1, text, size))
    finally:
        doc.close()
    return out


def _is_running_header(text: str) -> bool:
    # e.g. "16 • Sun Sign Astrology" or "Sun Signs • 17"
    return bool(re.match(r"^\d+\s*[•·]", text) or re.search(r"[•·]\s*\d+$", text))


def _classify_font_heading(head: str, current_sign: str | None) -> dict:
    if m := _PIS_RE.match(head):
        return {"indicator_type": "planet_in_sign", "context": "natal",
                "planet": m.group(1).lower(), "sign": m.group(2).lower(), "section": head}
    if m := _ASC_RE.match(head):
        return {"indicator_type": "ascendant", "context": "natal",
                "sign": m.group(1).lower(), "section": head}
    if head.strip().lower() in _SIGNS_SET:
        return {"indicator_type": "sign_chapter", "context": "natal",
                "sign": head.strip().lower(), "section": head}
    return {"indicator_type": "sign_section", "context": "natal",
            "sign": current_sign, "section": head}


def _chunk_font_sections(book: Book) -> list[Chunk]:
    raw = _font_lines(book.path)
    body_size = Counter(sz for _, _, sz in raw).most_common(1)[0][0]
    heading_min = body_size + 1.5

    chunks: list[Chunk] = []
    idx = 0
    cur_head: str | None = None
    cur_meta: dict = {"context": "natal"}
    cur_body: list[Line] = []
    cur_sign: str | None = None

    def flush() -> None:
        nonlocal idx
        if cur_head is None and not cur_body:
            return
        heading = cur_head
        if heading and cur_meta.get("indicator_type") == "sign_section" and cur_sign:
            heading = f"{cur_sign.title()} › {cur_head}"  # breadcrumb for generic subsections
        idx = _emit(chunks, cur_body, target=book.chunk_target, overlap=60,
                    heading=heading, meta=cur_meta, start_idx=idx)

    for page, text, size in raw:
        if _is_running_header(text):
            continue
        caps = _CAPS_PIS_RE.match(text)
        font_heading = size >= heading_min and len(text) >= 3 and len(text.split()) <= 12
        if caps:
            flush()
            heading = f"{caps.group(1).title()} in {caps.group(2).title()}"
            remainder = text[caps.end():].strip()
            cur_head = heading
            cur_meta = {"indicator_type": "planet_in_sign", "context": "natal",
                        "planet": caps.group(1).lower(), "sign": caps.group(2).lower(),
                        "section": heading}
            cur_body = [(page, remainder)] if remainder else []
        elif font_heading:
            flush()
            cur_head = text
            cur_meta = _classify_font_heading(text, cur_sign)
            cur_body = []
            if cur_meta["indicator_type"] == "sign_chapter":
                cur_sign = cur_meta["sign"]
        else:
            cur_body.append((page, text))
    flush()
    return chunks


def _chunk_token_windows(book: Book) -> list[Chunk]:
    chunks: list[Chunk] = []
    _emit(chunks, clean_lines(book.path), target=book.chunk_target, overlap=60,
          heading=None, meta={}, start_idx=0)
    return chunks


_STRATEGIES = {
    "transit_sections": _chunk_transit_sections,
    "toc_sections": _chunk_toc_sections,
    "font_sections": _chunk_font_sections,
    "token_windows": _chunk_token_windows,
}


def chunk_book(book: Book) -> list[Chunk]:
    return _STRATEGIES[book.chunk_strategy](book)
